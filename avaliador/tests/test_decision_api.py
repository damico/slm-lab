"""Contract tests for the UI routes, using an engine substitute and no GPU."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'lab04-decisoes'))
from serve_decisions import DecisionEngine, create_server
from demo_outcomes import history_fingerprint


class Builder:
    def build(self, history):
        if not isinstance(history, list):
            raise ValueError('history deve ser uma lista.')
        return {'text': 'same prompt', 'eligible': len(history) < 10,
                'input_ids': [1, 2], 'dropped_turns': 0, 'policy_ids': ['refund']}


class Model:
    model_key = 'test-adapter'
    def __call__(self, text):
        assert text == 'same prompt'
        return {'logits': [3, 0]}


class DecisionApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = DecisionEngine.__new__(DecisionEngine)
        self.engine.data = Path(self.temp.name)
        self.engine.builder = Builder()
        self.engine.model = Model()
        self.engine.bundle = {'labels': ['refund', 'retrieve_utterance'], 'actions': ['refund'],
            'model': 'test-model', 'max_seq': 2048, 'policy_docs': [{'id': 'refund', 'text': 'Check the purchase.'}]}
        self.engine.cal = {'temperature': 1.0, 'threshold': .99}
        self.server = create_server(self.engine, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urllib.request.urlopen(urllib.request.Request(self.url + path, data=data,
            headers={'Content-Type':'application/json'})) as response:
            return json.load(response)

    def test_context_and_calibrated_decision_share_the_prompt(self):
        context = self.call('/context', {'history': []})
        self.assertEqual(context['prompt'], 'same prompt')
        self.assertEqual(context['policies'][0]['text'], 'Check the purchase.')
        result = self.call('/decide', {'history': []})
        self.assertEqual(result['choice'], 'refund')
        self.assertTrue(result['requires_review'])
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
        self.assertEqual(self.call('/health')['status'], 'ready')

    def test_context_exceeded_returns_review_without_prediction(self):
        result = self.call('/decide', {'history': [{}] * 10})
        self.assertEqual(result['reason'], 'context_exceeded')
        self.assertIsNone(result['choice'])

    def test_examples_keep_gold_outside_history(self):
        demo = self.engine.data / 'demo'
        demo.mkdir()
        (demo / 'case_00.json').write_text(json.dumps({'history':[{'role':'customer','text':'Hello'}]}))
        (self.engine.data / 'demo_expected.json').write_text(json.dumps([{'id':'1:2','expected':'refund'}]))
        examples = self.call('/examples')['examples']
        self.assertEqual(examples[0]['expected'], 'refund')
        self.assertNotIn('expected', examples[0]['history'][0])

    def test_reference_outcome_is_exposed_only_for_matching_history(self):
        demo = self.engine.data / 'demo'
        demo.mkdir()
        history = [{'role': 'customer', 'text': 'Hello'}]
        path = demo / 'case_01.json'
        path.write_text(json.dumps({'history': history}))
        (self.engine.data / 'demo_expected.json').write_text(json.dumps([
            {'id': 'other:1', 'expected': 'refund'}, {'id': '1:2', 'expected': 'refund'}]))
        outcome = {'id': '1:2', 'history_sha256': history_fingerprint(history),
                   'delivered_messages': [{'role': 'agent', 'text': 'Recorded reply'}]}
        (self.engine.data / 'demo_outcomes.json').write_text(json.dumps({'1:2': outcome}))
        self.assertEqual(self.call('/examples')['examples'][0]['outcome'], outcome)
        path.write_text(json.dumps({'history': [{'role': 'customer', 'text': 'Edited'}]}))
        self.assertIsNone(self.call('/examples')['examples'][0]['outcome'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/context', {'history': history, 'outcome': outcome})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_rejects_gold_and_unknown_routes(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/context', {'history':[], 'label':'refund'})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/missing')
        self.assertEqual(error.exception.code, 404)
        error.exception.close()


if __name__ == '__main__':
    unittest.main()
