"""API local do adaptador BIRD; bancos permitidos vêm do split de teste."""
import argparse
import json
import sqlite3
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from lab_runtime import MODEL, read_jsonl, tokenizer, load_model, prompt_ids
from prepare_sql import SYSTEM, compact_schema
from score_sql import execute_readonly, clean_sql


def fields(row):
    content = row['messages'][1]['content']
    schema, tail = content.split('\nEvidence:\n', 1)
    evidence, question = tail.rsplit('\nQuestion:\n', 1)
    return {'id': row['id'], 'db_id': row['db_id'], 'question': question,
            'evidence': evidence, 'difficulty': row['difficulty'],
            'eligible': row.get('eligible', True)}


class SqlEngine:
    def __init__(self, data, adapter, results):
        self.data = Path(data)
        self.adapter = str(adapter)
        self.config = json.loads((self.data / 'length_report.json').read_text())
        self.rows = read_jsonl(self.data / 'test.jsonl')
        self.examples = [fields(row) for row in self.rows]
        manifest = json.loads((self.data / 'db_manifest.json').read_text())
        self.databases = {db: Path(manifest[db]['path']) for db in sorted({r['db_id'] for r in self.rows})}
        self.schemas = {db: compact_schema(path) for db, path in self.databases.items()}
        self.recorded, self.metrics, self.case_reports = {}, {}, {}
        for name in ('base', 'tuned', 'llm'):
            path = Path(results, 'test_' + name + '.jsonl')
            if path.exists():
                self.recorded[name] = {r['id']: r for r in read_jsonl(path)}
            case_path = Path(results, name + '.cases.jsonl')
            if case_path.exists():
                self.case_reports[name] = {r['id']: r for r in read_jsonl(case_path)}
            path = Path(results, name + '.summary.json')
            if path.exists():
                self.metrics[name] = json.loads(path.read_text())
        self.tok = tokenizer()
        self.model, _ = load_model(adapter)
        self.model.eval()

    def health(self):
        return {'status': 'ready', 'model': MODEL, 'adapter': Path(self.adapter).parent.name,
                'databases': len(self.databases), 'max_seq': self.config['max_seq'],
                'max_new': self.config['max_new']}

    def catalog(self):
        return {'databases': [{'id': db, 'schema': self.schemas[db]} for db in self.databases],
                'examples': self.examples, 'metrics': self.metrics}

    def database(self, db):
        if db not in self.databases:
            raise ValueError('Selecione um banco do Mini-Dev cadastrado no serviço.')
        return self.databases[db]

    def context(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'db_id', 'question', 'evidence'}:
            raise ValueError('Entrada deve conter somente db_id, question e evidence.')
        for key, limit in [('db_id', 100), ('question', 8000), ('evidence', 12000)]:
            if not isinstance(payload[key], str) or len(payload[key]) > limit:
                raise ValueError('Campo inválido: ' + key)
        if not payload['question'].strip():
            raise ValueError('Informe a pergunta.')
        self.database(payload['db_id'])
        user = ('Schema (PK numbers identify composite-key order):\n' + self.schemas[payload['db_id']]
                + '\nEvidence:\n' + payload['evidence'] + '\nQuestion:\n' + payload['question'])
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
        ids = prompt_ids(self.tok, messages)
        return {'messages': messages, 'input_tokens': len(ids),
                'eligible': len(ids) + self.config['max_new'] <= self.config['max_seq'],
                'max_seq': self.config['max_seq'], 'max_new': self.config['max_new']}

    def query(self, db, sql):
        start = time.perf_counter()
        try:
            table = execute_readonly(self.database(db), sql, timeout=15, with_columns=True)
            return {'status': 'ok', 'columns': table['columns'], 'rows': [[v.hex() if isinstance(v, bytes) else v for v in row] for row in table['rows'][:100]],
                    'row_count': len(table['rows']), 'truncated': len(table['rows']) > 100,
                    'latency_s': time.perf_counter() - start}, table['rows']
        except (sqlite3.Error, ValueError) as error:
            return {'status': 'sql_error', 'error': str(error),
                    'latency_s': time.perf_counter() - start}, None

    def generate(self, context):
        import torch
        ids = torch.tensor([prompt_ids(self.tok, context['messages'])], device=self.model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.config['max_new'], do_sample=False,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        tokens = generated[0, ids.shape[1]:].tolist()
        text = self.tok.decode(tokens, skip_special_tokens=True).strip()
        status = 'output_limit' if len(tokens) >= self.config['max_new'] and tokens[-1] != self.tok.eos_token_id else 'ok'
        slm = {'status': status, 'sql': clean_sql(text), 'latency_s': time.perf_counter() - start,
               'input_tokens': ids.shape[1], 'output_tokens': len(tokens)}
        return slm

    def predict(self, payload):
        context = self.context(payload)
        if not context['eligible']:
            return {'context': context, 'slm': {'status': 'context_exceeded', 'sql': ''}}
        slm = self.generate(context)
        status, text = slm['status'], slm['sql']
        result = {'context': context, 'slm': slm}
        got = None
        if status == 'ok':
            slm['execution'], got = self.query(payload['db_id'], text)
        # Gold e predições registradas só são consultados depois da geração.
        reference = next((row for row, example in zip(self.rows, self.examples)
                          if all(payload[k] == example[k] for k in payload)), None)
        if reference:
            execution, gold = self.query(payload['db_id'], reference['gold'])
            result['reference'] = {'id': reference['id'], 'gold_sql': reference['gold'], 'execution': execution,
                'ex_set': set(got) == set(gold) if got is not None and gold is not None else None,
                'ex_bag': Counter(got) == Counter(gold) if got is not None and gold is not None else None}
            result['recorded'] = {name: {'text': rows[reference['id']]['text'],
                'status': rows[reference['id']]['status'], 'latency_s': rows[reference['id']].get('latency_s'),
                'model': rows[reference['id']].get('model', MODEL), 'source': 'benchmark registrado',
                **{k: v for k, v in self.case_reports.get(name, {}).get(reference['id'], {}).items()
                   if k in ('correct', 'bag_correct', 'sql_error', 'reference_valid')}}
                for name, rows in self.recorded.items() if reference['id'] in rows}
        return result

    def execute(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'db_id', 'sql'}:
            raise ValueError('Entrada deve conter somente db_id e sql.')
        if not isinstance(payload['sql'], str) or not 0 < len(payload['sql']) <= 16000:
            raise ValueError('SQL inválido.')
        self.database(payload['db_id'])
        return self.query(payload['db_id'], payload['sql'])[0]


def create_server(engine, port):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body):
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            route = {'/health': engine.health, '/catalog': engine.catalog}.get(self.path)
            if not route:
                self.answer(404, {'error': 'Rota desconhecida.'})
                return
            self.answer(200, route())

        def do_POST(self):
            route = {'/context': engine.context, '/predict': engine.predict, '/execute': engine.execute}.get(self.path)
            if not route:
                self.answer(404, {'error': 'Rota desconhecida.'})
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 100000:
                    raise ValueError('Tamanho de entrada inválido.')
                self.answer(200, route(json.loads(self.rfile.read(size))))
            except (ValueError, TypeError, KeyError) as error:
                self.answer(400, {'error': str(error)})
            except Exception:
                self.answer(500, {'error': 'Falha na inferência SQL. Confira o log Python.'})
    return HTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--results', default='results')
    parser.add_argument('--port', type=int, default=8082)
    args = parser.parse_args()
    engine = SqlEngine(args.data, args.adapter, args.results)
    print(f'API SQL pronta em http://127.0.0.1:{args.port}', flush=True)
    with create_server(engine, args.port) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
