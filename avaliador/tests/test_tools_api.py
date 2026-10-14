import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/lab01-ferramentas'))
spec=importlib.util.spec_from_file_location('tools_api',ROOT/'scripts/lab01-ferramentas/serve_tools.py')
tools_api=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools_api)


class ToolsApiTest(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();root=Path(self.folder.name)
        self.data=root/'data';self.results=root/'results';self.data.mkdir();self.results.mkdir()
        self.tools=[{'name':'weather','description':'Weather','parameters':{'type':'object'}}]
        self.gold=[{'name':'weather','arguments':{'city':'Paris'}}]
        messages=[{'role':'system','content':'Select calls'},
                  {'role':'user','content':'Tools:\n'+json.dumps(self.tools)+'\nRequest:\nWeather in Paris'}]
        row={'id':'xlam:1','messages':messages,'gold':self.gold,'tool_names':['weather']}
        (self.data/'test.jsonl').write_text(json.dumps(row)+'\n')
        (self.results/'test_tuned.jsonl').write_text(json.dumps({'id':'xlam:1','status':'ok','text':json.dumps(self.gold),'latency_s':1.0})+'\n')
        self.lab=tools_api.ToolsLab(self.data,self.results)
        self.input={'query':'Weather in Paris','tools':self.tools,'mode':'recorded'}
    def tearDown(self):self.folder.cleanup()
    def test_recorded_mode_needs_no_model_and_reports_proposed_not_executed_calls(self):
        result=self.lab.predict(self.input)
        self.assertFalse(self.lab.health()['live_enabled'])
        self.assertEqual(result['slm']['calls'],self.gold)
        self.assertTrue(result['correct']);self.assertFalse(result['executed'])
    def test_edited_inputs_never_reuse_reference_or_recorded_prediction(self):
        changed={**self.input,'query':'Weather in Tokyo'}
        with self.assertRaises(ValueError):self.lab.predict(changed)
        seen=[]
        self.lab.generator=lambda messages:seen.append(messages) or {'text':'[]','status':'ok'}
        result=self.lab.predict({**changed,'mode':'specialist'})
        self.assertIsNone(result['reference']);self.assertIsNone(result['correct']);self.assertEqual(result['recorded'],{})
        self.assertNotIn('Paris',json.dumps(seen));self.assertIn('Tokyo',json.dumps(seen))
    def test_rejects_gold_duplicate_names_and_malformed_contract(self):
        for payload in [{**self.input,'gold':self.gold},{**self.input,'tools':self.tools*2}]:
            with self.assertRaises(ValueError):self.lab.predict(payload)
        self.assertEqual(tools_api.inspect_output('[{"name":"weather","arguments":{},"extra":1}]',self.tools)['status'],'invalid_output')
        self.assertEqual(tools_api.inspect_output('[{"name":"delete","arguments":{}}]',self.tools)['status'],'unknown_tool')
    def test_http_catalog_has_no_gold_and_unknown_routes_are_rejected(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),tools_api.make_handler(self.lab))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            with urllib.request.urlopen(url+'/catalog') as response:catalog=json.load(response)
            self.assertNotIn('gold',catalog['examples'][0]);self.assertNotIn('reference',catalog['examples'][0])
            request=urllib.request.Request(url+'/predict',json.dumps(self.input).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:self.assertTrue(json.load(response)['correct'])
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
