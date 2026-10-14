"""API de recibos: OCR/benchmark em CPU e adaptador real sob demanda."""
import argparse
import base64
import hashlib
import io
import json
import random
import tempfile
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from lab_runtime import read_jsonl,tokenizer,prompt_ids,load_model,MODEL
from receipt_common import FIELDS,messages,normalize_text,normalize_field,parse_fields,extract_text,safe_csv


def inspect_output(text,status='ok'):
    if not isinstance(text,str):raise ValueError('Resposta deve ser texto.')
    result={'status':status,'text':text,'fields':{},'schema_valid':False}
    if status=='ok':
        try:result.update(fields=parse_fields(text),schema_valid=True)
        except (ValueError,TypeError):result['status']='invalid_output'
    return result


class Generator:
    def __init__(self,adapter,config):
        self.adapter=adapter;self.config=config;self.model=None;self.tok=None
    def __call__(self,prompt):
        import torch
        if self.tok is None:self.tok=tokenizer()
        ids=prompt_ids(self.tok,prompt);limit=self.config['max_new']
        if len(ids)+limit>self.config['max_seq']:return {'status':'context_exceeded','text':'','input_tokens':len(ids)}
        if self.model is None:
            if json.loads((Path(self.adapter)/'adapter_config.json').read_text())['base_model_name_or_path']!=MODEL:
                raise ValueError('Adaptador de outra base.')
            self.model,_=load_model(self.adapter);self.model.eval()
            for key in ['temperature','top_p','top_k']:setattr(self.model.generation_config,key,None)
        x=torch.tensor([ids],device=self.model.device);torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            y=self.model.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=limit,do_sample=False,
                pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize();tokens=y[0,len(ids):].tolist()
        return {'status':'output_limit' if len(tokens)>=limit and tokens[-1]!=self.tok.eos_token_id else 'ok',
                'text':self.tok.decode(tokens,skip_special_tokens=True).strip(),
                'latency_s':time.perf_counter()-start,'input_tokens':len(ids),'output_tokens':len(tokens)}


class ReceiptsLab:
    def __init__(self,data,results,generator=None):
        data=Path(data).resolve();self.root=data.parent;self.generator=generator;self.lock=threading.Lock()
        self.rows=read_jsonl(data/'test.jsonl');self.by_id={r['id']:r for r in self.rows}
        self.config=json.loads((data/'length_report.json').read_text())
        self.examples=[{'id':r['id'],'text':r['text'],'has_image':bool(r.get('image_path')),
            'ocr_empty':r.get('ocr_empty',False)} for r in self.rows]
        self.predictions={};self.metrics={};sha=hashlib.sha256((data/'test.jsonl').read_bytes()).hexdigest()
        for key,name,summary in [('base','test_original','original'),('specialist','test_especialista','especialista'),('llm','test_llm','llm')]:
            path=Path(results)/(name+'.jsonl')
            if path.exists():
                predictions=read_jsonl(path)
                if len({r['id'] for r in predictions})!=len(predictions) or any(r['id'] not in self.by_id for r in predictions):
                    raise ValueError('IDs incompatíveis: '+str(path))
                meta=path.with_suffix('.meta.json')
                if meta.exists() and json.loads(meta.read_text())['input_sha256']!=sha:
                    raise ValueError('Predições de outro dataset: '+str(path))
                self.predictions[key]={r['id']:r for r in predictions}
            path=Path(results)/(summary+'.summary.json')
            if path.exists():self.metrics[key]=json.loads(path.read_text())
    def health(self):
        return {'status':'ready','live_enabled':self.generator is not None,
            'model_loaded':bool(getattr(self.generator,'model',None)),'examples':len(self.rows),
            'recorded_available':bool(self.predictions.get('specialist'))}
    def context(self,payload):
        if not isinstance(payload,dict) or set(payload)-{'text','example_id','mode'}:
            raise ValueError('Envie somente texto, ID do exemplo e modo; gabarito não é entrada.')
        text=payload.get('text');mode=payload.get('mode','recorded');example_id=payload.get('example_id')
        if not isinstance(text,str) or len(text)>20000 or mode not in ('recorded','specialist'):
            raise ValueError('Texto ou modo inválido.')
        if example_id is not None and (not isinstance(example_id,str) or example_id not in self.by_id):
            raise ValueError('Exemplo inexistente.')
        text=normalize_text(text);row=self.by_id.get(example_id)
        if row and normalize_text(row['text'])!=text:row=None
        if not text and row is None:raise ValueError('O OCR não encontrou texto utilizável.')
        prompt=row['messages'] if row else messages(text)
        return mode,row,{'messages':prompt,'max_new':self.config['max_new'],
            'reference_available':row is not None,'ocr_empty':not bool(text)}
    @staticmethod
    def compare(result,row):
        expected=parse_fields(row['answer'])
        fields={key:bool(result['schema_valid'] and normalize_field(result['fields'][key])==normalize_field(expected[key])) for key in FIELDS}
        return {'field_correct':fields,'correct':all(fields.values())}
    def predict(self,payload):
        mode,row,context=self.context(payload)
        if mode=='recorded':
            if row is None:raise ValueError('Entrada editada. Ative inferência real para extrair novos campos.')
            raw=self.predictions.get('specialist',{}).get(row['id'])
            if raw is None:raise ValueError('Resultado registrado ausente.')
        else:
            if self.generator is None:raise ValueError('Inferência real desativada enquanto outro laboratório usa as GPUs.')
            if context['ocr_empty']:raw={'text':'','status':'empty_input'}
            else:
                if not self.lock.acquire(blocking=False):raise ValueError('Há uma inferência em andamento. Aguarde.')
                try:raw=self.generator(context['messages'])
                finally:self.lock.release()
        result=inspect_output(raw.get('text',''),raw.get('status','ok'))
        result.update({key:raw[key] for key in ['latency_s','input_tokens','output_tokens','usage'] if key in raw})
        recorded={}
        if row:
            result.update(self.compare(result,row))
            for name,items in self.predictions.items():
                pred=items.get(row['id'])
                if pred:
                    item=inspect_output(pred.get('text',''),pred.get('status','ok'))
                    item.update(self.compare(item,row));item['latency_s']=pred.get('latency_s');recorded[name]=item
        return {'id':row['id'] if row else None,'mode':mode,'slm':result,'recorded':recorded,
                'context':context,'reference':parse_fields(row['answer']) if row else None,'requires_review':True}
    def image(self,identifier):
        row=self.by_id.get(identifier)
        if not row or not row.get('image_path'):raise ValueError('Imagem inexistente.')
        path=Path(row['image_path']);path=path if path.is_absolute() else self.root/path
        path=path.resolve()
        if not path.is_relative_to((self.root/'raw/images').resolve()):raise ValueError('Imagem fora da pasta cadastrada.')
        data=path.read_bytes()
        if data.startswith(b'\xff\xd8\xff'):return data,'image/jpeg'
        if data.startswith(b'\x89PNG\r\n\x1a\n'):return data,'image/png'
        from PIL import Image
        out=io.BytesIO()
        with Image.open(path) as im:im.convert('RGB').save(out,format='PNG')
        return out.getvalue(),'image/png'


def ocr_upload(payload):
    if not isinstance(payload,dict) or set(payload)!={'filename','content'}:raise ValueError('Contrato de upload inválido.')
    name=payload['filename'];content=payload['content']
    if not isinstance(name,str) or not isinstance(content,str) or len(content)>12000000:raise ValueError('Upload excessivo.')
    extension=Path(name).suffix.lower()
    if extension not in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff','.pdf'}:raise ValueError('Use imagem ou PDF de uma página.')
    data=base64.b64decode(content,validate=True)
    if not 0<len(data)<=8000000:raise ValueError('Use até 8 MB por recibo.')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/('receipt'+extension);path.write_bytes(data);start=time.perf_counter()
        text,method=extract_text(path,'eng',6)
    return {'text':text,'method':method,'latency_s':time.perf_counter()-start,
            'sha256':hashlib.sha256(data).hexdigest(),'reference_available':False}


def make_handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body,mime='application/json; charset=utf-8'):
            data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            url=urlparse(self.path)
            try:
                if url.path=='/health':self.reply(200,lab.health())
                elif url.path=='/catalog':self.reply(200,{'examples':lab.examples,'metrics':lab.metrics})
                elif url.path=='/example/random':self.reply(200,random.choice(lab.examples))
                elif url.path=='/image':
                    data,mime=lab.image(parse_qs(url.query).get('id',[''])[0]);self.reply(200,data,mime)
                else:self.reply(404,{'message':'Rota inexistente.'})
            except (ValueError,OSError):self.reply(404,{'message':'Imagem não disponível.'})
        def do_POST(self):
            if self.path not in ('/predict','/ocr','/validate'):self.reply(404,{'message':'Rota inexistente.'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=(12001000 if self.path=='/ocr' else 128000):raise ValueError('Entrada excessiva.')
                payload=json.loads(self.rfile.read(size))
                if self.path=='/predict':result=lab.predict(payload)
                elif self.path=='/ocr':result=ocr_upload(payload)
                else:
                    if not isinstance(payload,dict) or set(payload)!={'text','status'}:raise ValueError('Contrato inválido.')
                    result=inspect_output(payload['text'],payload['status'])
                self.reply(200,result)
            except (ValueError,TypeError,KeyError,OSError) as error:self.reply(400,{'message':str(error)})
            except Exception:
                import traceback;traceback.print_exc();self.reply(503,{'message':'Falha no OCR ou modelo. Consulte o log da API de recibos.'})
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');parser.add_argument('--results',default='results')
    parser.add_argument('--adapter',default='runs/main/final_adapter');parser.add_argument('--enable-gpu',action='store_true')
    parser.add_argument('--port',type=int,default=8083);args=parser.parse_args()
    config=json.loads((Path(args.data)/'length_report.json').read_text())
    generator=Generator(args.adapter,config) if args.enable_gpu else None
    lab=ReceiptsLab(args.data,args.results,generator)
    print('API recibos pronta em localhost:'+str(args.port)+' / GPU habilitada: '+str(args.enable_gpu),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(lab)).serve_forever()


if __name__=='__main__':main()
