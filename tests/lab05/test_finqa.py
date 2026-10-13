import json
import sys
import tempfile
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/lab05-finqa'))
from finqa_core import source_map,execute,parse_output,retrieve,messages,TEACHER,save_json,write_rows,read_rows,file_hash,read_json
from prepare_finqa import convert
from build_distillation import select_pairs,main as build_main
from evaluate_finqa import score
from serve_finqa import context_from_payload


class Tokenizer:
    def encode(self,text,**kwargs):return list(text.encode())
    def apply_chat_template(self,messages,**kwargs):return list(json.dumps(messages).encode())


class FinqaTest(unittest.TestCase):
    def setUp(self):
        self.raw={'id':'report.pdf-1','filename':'report.pdf','pre_text':[], 'post_text':[],
            'table':[['Year','Revenue'],['Previous','100'],['Current','118']],
            'qa':{'question':'Revenue increased from 100 to 118. What was percentage growth?',
                  'gold_inds':{'table_1':'old','table_2':'new'},'program':'subtract(118, 100), divide(#0, 100), multiply(#1, const_100)','exe_ans':18}}
        self.context={'sources':source_map(self.raw),'question':self.raw['qa']['question'],'table':self.raw['table']}
        self.output={'evidence':['table_1','table_2'],'program':self.raw['qa']['program']}

    def test_growth_and_trace_and_percent_convention(self):
        result=execute(self.output,self.context)
        self.assertEqual(result['answer'],18)
        self.assertEqual([step['result'] for step in result['trace']],[18,.18,18])
        self.assertEqual(execute({'evidence':['table_1'],'program':'divide(10%, const_2)'},
            {**self.context,'question':'Half of 10%?'})['answer'],.05)

    def test_rejects_forged_sources_numbers_code_and_future_references(self):
        for payload in [dict(self.output,evidence=['text_999']),dict(self.output,program='add(500, const_1)'),
            dict(self.output,program='add(#0, const_1)'),dict(self.output,program='divide(100, 0)'),
            dict(self.output,program="__import__('os', none)"),dict(self.output,program='add(100, 118); print(1)'),
            dict(self.output,program='exp(100, 100)')]:
            with self.subTest(payload=payload),self.assertRaises((ValueError,ZeroDivisionError)):
                execute(payload,self.context)
        with self.assertRaises(ValueError):parse_output('{"evidence":["table_1"],"program":"add(100,100)","program":"add(1,1)"}')

    def test_table_row_needs_citation_and_uses_all_values(self):
        raw={**self.raw,'table':[['Year','A','B'],['revenue','100','118']]}
        context={**self.context,'sources':source_map(raw),'table':raw['table']}
        self.assertEqual(execute({'evidence':['table_1'],'program':'table_average(revenue, none)'},context)['answer'],109)
        with self.assertRaises(ValueError):execute({'evidence':['table_0'],'program':'table_sum(revenue, none)'},context)

    def test_gold_mutation_does_not_change_prompt_or_eligibility(self):
        tok={'student':Tokenizer(),'teacher':Tokenizer()}
        row=convert(self.raw,'test',tok,10000,512,12)
        changed={**self.raw,'qa':{**self.raw['qa'],'program':'add(1, 1)','gold_inds':{'text_999':'LEAK_MARKER'},'exe_ans':999}}
        other=convert(changed,'test',tok,10000,512,12)
        self.assertEqual(row['messages'],other['messages'])
        self.assertEqual(row['eligible'],other['eligible'])
        self.assertNotIn('LEAK_MARKER',json.dumps(other['messages']))

    def test_pair_selection_filters_wrong_teacher_and_preserves_ids(self):
        tok=Tokenizer();row=convert(self.raw,'train',{'student':tok},10000,512,12)
        good={'id':row['id'],'model':TEACHER,'status':'ok','text':json.dumps(self.output)}
        gold,teacher,audit=select_pairs([row],[good],tok,{'max_seq':10000})
        self.assertEqual([r['id'] for r in gold],[r['id'] for r in teacher])
        self.assertTrue(audit[0]['approved'])
        wrong={**good,'text':json.dumps({**self.output,'program':'add(100, 118)'})}
        self.assertFalse(select_pairs([row],[wrong],tok,{'max_seq':10000})[2][0]['approved'])
        with self.assertRaises(ValueError):select_pairs([{**row,'split':'test'}],[good],tok,{'max_seq':10000})
        with self.assertRaises(ValueError):select_pairs([row],[good,good],tok,{'max_seq':10000})

    def test_file_pipeline_pairs_and_scores_missing_context_as_failure(self):
        tok=Tokenizer();row=convert(self.raw,'train',{'student':tok},10000,512,12)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';pred=root/'teacher.jsonl';paired=root/'paired'
            config={'max_seq':10000,'max_new':512,'models':{TEACHER:'teacher-revision'}}
            save_json(data/'config.json',config)
            write_rows(data/'train_candidates.jsonl',[row])
            write_rows(data/'sft_available/validation.jsonl',[{**row,'split':'validation'}])
            prediction={'id':row['id'],'model':TEACHER,'status':'ok','text':json.dumps(self.output)}
            write_rows(pred,[prediction])
            save_json(pred.with_suffix('.meta.json'),{'signature':{'model':TEACHER,'revision':'teacher-revision',
                'max_seq':10000,'max_new':512,'input_sha256':file_hash(data/'train_candidates.jsonl')}})
            with patch('build_distillation.tokenizer',return_value=tok),patch.object(sys,'argv',
                ['build_distillation.py','--data',str(data),'--predictions',str(pred),'--out',str(paired)]),redirect_stdout(io.StringIO()):
                build_main()
            self.assertEqual([r['id'] for r in read_rows(paired/'sft/train.jsonl')],
                             [r['id'] for r in read_rows(paired/'distilled/train.jsonl')])
            self.assertTrue((paired/'review.jsonl').exists())
            test=root/'test.jsonl';predictions=root/'predictions.jsonl';out=root/'metrics'
            write_rows(test,[{**row,'split':'test'},{**row,'id':'outside-context','split':'test','eligible':False}])
            write_rows(predictions,[prediction])
            with redirect_stdout(io.StringIO()):score(test,predictions,out)
            summary=read_json(str(out)+'.summary.json')
            self.assertEqual(summary['n_total'],2)
            self.assertEqual(summary['accuracy_eligible'],1)
            self.assertEqual(summary['accuracy_all_valid'],.5)

    def test_api_rejects_gold_and_uses_question_and_raw_sources(self):
        payload={k:self.raw[k] for k in ['pre_text','post_text','table']}
        payload['question']=self.raw['qa']['question']
        self.assertEqual(context_from_payload(payload)['question'],payload['question'])
        with self.assertRaises(ValueError):context_from_payload({**payload,'gold':18})


if __name__=='__main__':unittest.main()
