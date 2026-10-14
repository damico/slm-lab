import base64
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/lab03-recibos'))
spec=importlib.util.spec_from_file_location('receipts_api',ROOT/'scripts/lab03-recibos/serve_receipts.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)


class ReceiptsApiTest(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name);data=self.root/'data';results=self.root/'results';data.mkdir();results.mkdir()
        self.gold={'company':'SHOP','date':'01/02/2018','address':'12 STREET','total':'9.90'}
        raw=self.root/'raw/images';raw.mkdir(parents=True);(raw/'receipt.img').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
        row={'id':'test:1','text':'SHOP\nTOTAL 9.90','image_path':'raw/images/receipt.img',
             'messages':api.messages('SHOP\nTOTAL 9.90'),'answer':json.dumps(self.gold)}
        (data/'test.jsonl').write_text(json.dumps(row)+'\n');(data/'length_report.json').write_text('{"max_seq":2048,"max_new":512}')
        (results/'test_especialista.jsonl').write_text(json.dumps({'id':'test:1','text':json.dumps(self.gold),'status':'ok','latency_s':2})+'\n')
        self.lab=api.ReceiptsLab(data,results);self.payload={'text':row['text'],'example_id':'test:1','mode':'recorded'}
    def test_recorded_fields_match_reference_without_loading_a_model(self):
        result=self.lab.predict(self.payload)
        self.assertTrue(result['slm']['correct']);self.assertEqual(result['slm']['fields'],self.gold)
        self.assertTrue(result['requires_review']);self.assertFalse(self.lab.health()['live_enabled'])
    def test_edited_ocr_drops_reference_and_never_sends_it_to_generator(self):
        changed={**self.payload,'text':'NEW RECEIPT'}
        with self.assertRaises(ValueError):self.lab.predict(changed)
        seen=[];self.lab.generator=lambda prompt:seen.append(prompt) or {'status':'ok','text':json.dumps(self.gold)}
        result=self.lab.predict({**changed,'mode':'specialist'})
        self.assertIsNone(result['reference']);self.assertEqual(result['recorded'],{})
        self.assertNotIn('correct',result['slm']);self.assertNotIn('12 STREET',json.dumps(seen))
    def test_gold_and_arbitrary_image_paths_are_rejected(self):
        with self.assertRaises(ValueError):self.lab.predict({**self.payload,'reference':self.gold})
        self.lab.by_id['test:1']['image_path']='../outside.jpg'
        with self.assertRaises(ValueError):self.lab.image('test:1')
    def test_incomplete_and_invalid_outputs_do_not_become_fields(self):
        self.assertFalse(api.inspect_output(json.dumps(self.gold),'output_limit')['schema_valid'])
        self.assertEqual(api.inspect_output('{"total":"9.90"}')['status'],'invalid_output')
    def test_uploaded_bytes_use_a_temporary_file_and_have_no_reference(self):
        seen=[]
        def extract(path,lang,psm):
            seen.append(path);self.assertEqual(path.read_bytes(),b'uploaded receipt')
            return 'OCR TEXT','image_ocr'
        with patch.object(api,'extract_text',side_effect=extract):
            result=api.ocr_upload({'filename':'../user.png','content':base64.b64encode(b'uploaded receipt').decode()})
        self.assertEqual(result['text'],'OCR TEXT');self.assertFalse(result['reference_available'])
        self.assertFalse(seen[0].exists())
        with self.assertRaises(ValueError):api.ocr_upload({'filename':'file.py','content':'AAAA'})
    def test_http_catalog_has_no_gold_and_serves_only_registered_images(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),api.make_handler(self.lab));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            with urllib.request.urlopen(url+'/catalog') as r:data=json.load(r)
            self.assertNotIn('answer',data['examples'][0]);self.assertNotIn('reference',data['examples'][0])
            with urllib.request.urlopen(url+'/image?id=test:1') as r:self.assertEqual(r.headers['Content-Type'],'image/png')
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+'/image?id=../../etc/passwd')
            self.assertEqual(error.exception.code,404)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
