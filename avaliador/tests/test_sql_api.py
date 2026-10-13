import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SQL_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'lab02-sql'
sys.path.insert(0, str(SQL_DIR))
# Os labs têm runtimes de mesmo nome; ambos exportam as funções utilizadas aqui.
from serve_sql import SqlEngine, fields
from prepare_sql import SYSTEM, compact_schema
from score_sql import execute_readonly


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content']) for m in messages)))


class SqlApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.db = self.root / 'demo.sqlite'
        with sqlite3.connect(self.db) as con:
            con.executescript('CREATE TABLE items(name TEXT); INSERT INTO items VALUES ("A"),("A"),("B");')
        self.payload = {'db_id': 'demo', 'question': 'List names', 'evidence': ''}
        user = 'Schema (PK numbers identify composite-key order):\n' + compact_schema(self.db) + '\nEvidence:\n\nQuestion:\nList names'
        row = {'id': 'test:1', 'db_id': 'demo', 'difficulty': 'simple', 'eligible': True,
               'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}],
               'gold': 'SELECT name FROM items'}
        self.engine = SqlEngine.__new__(SqlEngine)
        self.engine.databases = {'demo': self.db}
        self.engine.schemas = {'demo': compact_schema(self.db)}
        self.engine.rows, self.engine.examples = [row], [fields(row)]
        self.engine.recorded, self.engine.case_reports = {}, {}
        self.engine.config = {'max_seq': 2048, 'max_new': 512}
        self.engine.tok = Tokenizer()

    def test_real_sqlite_rejects_writes_attach_pragmas_and_multiple_statements(self):
        self.assertEqual(execute_readonly(self.db, 'SELECT name FROM items'), [('A',), ('A',), ('B',)])
        for sql in ['DELETE FROM items', 'DROP TABLE items', "ATTACH ':memory:' AS other", 'PRAGMA user_version', 'SELECT 1; SELECT 2', "SELECT load_extension('x')"]:
            with self.subTest(sql=sql), self.assertRaises((sqlite3.Error, ValueError)):
                execute_readonly(self.db, sql)
        self.assertEqual(execute_readonly(self.db, 'SELECT count(*) FROM items'), [(3,)])

    def test_generation_receives_only_prompt_then_reference_compares_set_and_bag(self):
        received = []
        def generate(context):
            received.append(context)
            return {'sql': 'SELECT DISTINCT name FROM items', 'status': 'ok', 'latency_s': .1}
        self.engine.generate = generate
        result = self.engine.predict(self.payload)
        self.assertNotIn('gold', json.dumps(received))
        self.assertEqual(received[0]['messages'], self.engine.rows[0]['messages'])
        self.assertTrue(result['reference']['ex_set'])
        self.assertFalse(result['reference']['ex_bag'])
        self.assertEqual(result['slm']['execution']['columns'], ['name'])
        changed = {**self.payload, 'question': 'List unique names'}
        self.assertNotIn('reference', self.engine.predict(changed))

    def test_context_and_database_boundaries(self):
        for payload in [{**self.payload, 'gold': 'SELECT 1'}, {**self.payload, 'db_id': str(self.db)}]:
            with self.assertRaises(ValueError): self.engine.context(payload)
        self.engine.config['max_seq'] = 10
        self.engine.generate = lambda _: self.fail('Não deveria gerar')
        self.assertEqual(self.engine.predict(self.payload)['slm']['status'], 'context_exceeded')

    def test_preview_truncates_without_changing_comparison_data(self):
        execution, full = self.engine.query('demo', 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<101) SELECT x FROM n')
        self.assertTrue(execution['truncated'])
        self.assertEqual(len(execution['rows']), 100)
        self.assertEqual(len(full), 101)
        self.assertEqual(self.engine.execute({'db_id': 'demo', 'sql': 'DELETE FROM items'})['status'], 'sql_error')


if __name__ == '__main__':
    unittest.main()
