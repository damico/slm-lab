import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/lab05-finqa'))
spec=importlib.util.spec_from_file_location('finqa_web',ROOT/'scripts/lab05-finqa/serve_finqa_web.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)

class FinqaWebTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);data=self.root/'data';results=self.root/'results';data.mkdir();results.mkdir()
        self.context={'question':'What was revenue growth as a percentage number?',
            'sources':{'table_1':'Revenue previous: 100; current: 118'},'table':[['Revenue','100','118']]}
        self.program={'evidence':['table_1'],'program':'subtract(118, 100), divide(#0, const_100), multiply(#1, const_100)'}
        row={'id':'case:1','document':'report/1','context':self.context,'gold':18,'gold_output':self.program,'eligible':True}
        (data/'test.jsonl').write_text(json.dumps(row)+'\n');(data/'config.json').write_text('{"top_k":12,"max_seq":2048,"max_new":512}')
        for backend in ['base','sft','distilled','teacher']:
            p=results/f'test_{backend}.jsonl';p.write_text(json.dumps({'id':'case:1','status':'ok','text':json.dumps(self.program),'latency_s':1})+'\n')
            p.with_suffix('.meta.json').write_text(json.dumps({'signature':{'input_sha256':api.file_hash(data/'test.jsonl')}}))
        self.lab=api.FinqaLab(data,results);self.payload={'context':self.context,'example_id':'case:1','mode':'recorded'}
    def test_recorded_calculation_executes_with_four_conditions_without_loading_gpu(self):
        r=self.lab.predict(self.payload)
        self.assertEqual(r['slm']['execution']['answer'],18);self.assertEqual(len(r['slm']['execution']['trace']),3)
        self.assertTrue(r['slm']['correct']);self.assertEqual(len(r['recorded']),4)
        self.assertFalse(self.lab.health()['model_loaded']);self.assertTrue(r['requires_review'])
    def test_catalog_and_example_do_not_expose_references(self):
        for value in [self.lab.catalog(),self.lab.example(self.lab.rows[0])]:
            text=json.dumps(value);self.assertNotIn('gold',text);self.assertNotIn('program',text)
    def test_edited_context_drops_reference_and_recordings_from_real_inference(self):
        context={**self.context,'question':'New question about revenue growth as a percentage number?'}
        payload={**self.payload,'context':context}
        with self.assertRaises(ValueError):self.lab.predict(payload)
        seen=[]
        class FakeGenerator:
            tok=None
            def __call__(inner,prompt,max_new):seen.append(prompt);return {'status':'ok','text':json.dumps(self.program)}
        self.lab.enable_gpu=True;self.lab.generator=FakeGenerator()
        with patch.object(api,'prompt_ids',return_value=[1,2]):r=self.lab.predict({**payload,'mode':'specialist'})
        self.assertIsNone(r['reference']);self.assertEqual(r['recorded'],{});self.assertNotIn('correct',r['slm'])
        self.assertEqual(seen,[api.messages(context)])
    def test_gold_program_and_unknown_nested_context_fields_are_rejected(self):
        for bad in [{**self.payload,'gold':18},{**self.payload,'program':'add(1,2)'},
                    {**self.payload,'context':{**self.context,'answer':18}},
                    {**self.payload,'example_id':'unknown'}]:
            with self.assertRaises(ValueError):self.lab.predict(bad)
    def test_truncated_and_ungrounded_programs_never_become_an_answer(self):
        for status,text in [('output_limit',json.dumps(self.program)),('ok','{"evidence":["table_1"],"program":"add(999, 118)"}')]:
            r=self.lab.inspect({'status':status,'text':text},self.context)
            self.assertFalse(r['program_valid']);self.assertIsNone(r['execution'])
    def test_wrong_benchmark_hash_is_rejected(self):
        (self.root/'results/test_distilled.meta.json').write_text('{"signature":{"input_sha256":"wrong"}}')
        with self.assertRaises(ValueError):api.FinqaLab(self.root/'data',self.root/'results')

if __name__=='__main__':unittest.main()
