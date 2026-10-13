import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'datasets'))
from catalog_service import Catalog,normalize


class DatasetCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cache=self.root/'cache'
        self.roots={lab:self.root/lab for lab in ['01','02','03','04','05']}
        self.tool={'id':'tools:1','eligible':True,'messages':[{'role':'user','content':'Tools:\n[{"name":"weather","parameters":{"city":"string"}}]\nRequest:\nWeather in Paris'}],
            'gold':[{'name':'weather','arguments':{'city':'Paris'}}]}
        self.write(self.roots['01']/'data/train.jsonl',[{**self.tool,'id':f'tools:{i}'} for i in range(25)])
        self.write(self.roots['02']/'data/train.jsonl',[{'id':'sql:1','eligible':False,'db_id':'school',
            'messages':[{'role':'user','content':'Schema (PK numbers identify composite-key order):\nstudent(name TEXT)\nEvidence:\n\nQuestion:\nHow many students?'}],'gold':'SELECT COUNT(*) FROM student'}])
        image=self.cache/'originals/03/images/receipt.png';image.parent.mkdir(parents=True)
        image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1XkAAAAASUVORK5CYII='))
        self.write(self.cache/'originals/03/train.jsonl',[{'id':'receipt:1','transcript':'SHOP\nTOTAL 10',
            'entities':{'company':'SHOP','date':'01/01/2026','address':'Street','total':'10'},'image_path':str(image)}])
        history={'history':[{'role':'customer','text':'Please cancel my subscription'}],'guideline_excerpts':['Check account']}
        self.write(self.roots['04']/'data/train.jsonl',[{'id':'4989:5','label':'cancel-subscription','text':'Header\nAllowed labels\n'+json.dumps(history),'eligible':True}])
        self.write(self.cache/'originals/05/train.jsonl',[{'id':'finqa:1','table':[['Year','Revenue'],['2025','118']],
            'pre_text':['Revenue was 100 before.'],'post_text':[],'qa':{'question':'Growth?',
                'gold_inds':{'table_1':'118'},'program':'subtract(118, 100)','exe_ans':18}}])
        self.catalog=Catalog(self.cache,self.roots);self.catalog.refresh()

    def write(self,path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_all_five_labs_have_explicit_origins_and_splits(self):
        labs=self.catalog.catalog()['labs']
        self.assertEqual(len(labs),5);self.assertTrue(all(lab['available'] for lab in labs))
        self.assertEqual(labs[2]['versions'][0]['origin'],'original')
        self.assertEqual(labs[4]['train_count'],1)

    def test_paginated_search_filter_and_random_case(self):
        result=self.catalog.samples('01-prepared-train',q='Paris',page=2,size=20)
        self.assertEqual(result['total'],25);self.assertEqual(len(result['items']),5)
        self.assertEqual(self.catalog.samples('02-prepared-train',eligible='yes')['total'],0)
        self.assertEqual(self.catalog.samples('02-prepared-train',eligible='no')['total'],1)
        self.assertEqual(self.catalog.samples('01-prepared-train',q='absent')['total'],0)
        self.assertEqual(len(self.catalog.samples('01-prepared-train',random_case=True)['items']),1)
        self.assertEqual(self.catalog.samples('01-prepared-train',q='Paris" OR DROP TABLE samples')['total'],0)
        with self.assertRaises(ValueError):self.catalog.samples('../../etc/passwd')

    def test_input_and_gold_normalized_for_each_lab(self):
        tool=self.catalog.detail('01-prepared-train',0)
        self.assertEqual(tool['question'],'Weather in Paris');self.assertEqual(tool['reference'][0]['name'],'weather')
        sql=self.catalog.detail('02-prepared-train',0)
        self.assertEqual(sql['question'],'How many students?');self.assertEqual(sql['schema'],'student(name TEXT)')
        decision=self.catalog.detail('04-prepared-train',0)
        self.assertEqual(decision['history'][0]['role'],'customer');self.assertEqual(decision['reference'],'cancel-subscription')
        finance=self.catalog.detail('05-original-train',0)
        self.assertEqual(finance['expected_result'],18);self.assertNotIn('gold_inds',finance.get('sources',{}))

    def test_image_only_from_registered_receipt_roots(self):
        receipt=self.catalog.detail('03-original-train',0)
        self.assertTrue(receipt['has_image']);self.assertNotIn('image_path',receipt)
        self.assertEqual(self.catalog.image('03-original-train',0)[1],'image/png')
        secret=self.root/'secret.png';secret.write_bytes(b'\x89PNGsecret')
        self.assertIsNone(self.catalog.image_path({'image_path':str(secret)}))
        with self.assertRaises(ValueError):self.catalog.image('01-prepared-train',0)

    def test_changed_sources_are_not_read_with_stale_offsets(self):
        self.write(self.roots['01']/'data/train.jsonl',[{**self.tool,'id':'new'}])
        with self.assertRaises(ValueError):self.catalog.detail('01-prepared-train',0)
        self.catalog.refresh()
        self.assertEqual(self.catalog.samples('01-prepared-train')['total'],1)
        self.assertEqual(self.catalog.detail('01-prepared-train',0)['id'],'new')


if __name__=='__main__':unittest.main()
