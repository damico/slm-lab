"""API xLAM: benchmark em CPU; adaptador real opcional, carregado sob demanda."""
import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from lab_runtime import read_jsonl,prompt_ids,tokenizer,load_model
from prepare_tools import SYSTEM
from score_tools import parse_calls,canonical


def validate_input(payload):
    if not isinstance(payload,dict) or set(payload)-{'query','tools','mode'}:
        raise ValueError('Envie somente query, tools e mode; gabarito não é entrada.')
    query=payload.get('query');tools=payload.get('tools');mode=payload.get('mode','recorded')
    if not isinstance(query,str) or not query.strip() or len(query)>8000:
        raise ValueError('Escreva uma solicitação com até 8.000 caracteres.')
    if not isinstance(tools,list) or not 1<=len(tools)<=64 or len(json.dumps(tools))>100000:
        raise ValueError('Envie um catálogo JSON com 1 a 64 ferramentas.')
    names=[]
    for tool in tools:
        if not isinstance(tool,dict) or not isinstance(tool.get('name'),str) or not tool['name'].strip():
            raise ValueError('Cada ferramenta precisa de name preenchido.')
        if 'parameters' in tool and not isinstance(tool['parameters'],dict):
            raise ValueError('parameters deve ser um objeto JSON.')
        names.append(tool['name'])
    if len(set(names))!=len(names):raise ValueError('Nomes de ferramentas duplicados.')
    if mode not in ('recorded','specialist'):raise ValueError('Modo inválido.')
    return query.strip(),tools,mode


def inspect_output(text,tools,status='ok'):
    if not isinstance(text,str):raise ValueError('Resposta deve ser texto.')
    result={'status':status,'text':text,'calls':[],'contract_valid':False,'known_tools':False}
    if status!='ok':return result
    try:
        calls=parse_calls(text);names={t['name'] for t in tools}
        result.update(calls=calls,contract_valid=True,known_tools=all(c['name'] in names for c in calls))
        if not result['known_tools']:result['status']='unknown_tool'
    except (ValueError,TypeError):result['status']='invalid_output'
    return result


class Generator:
    def __init__(self,adapter,max_seq,max_new):
        self.adapter=adapter;self.max_seq=max_seq;self.max_new=max_new;self.model=None;self.tok=None
    def __call__(self,messages):
        import torch
        if self.tok is None:self.tok=tokenizer()
        ids=prompt_ids(self.tok,messages)
        if len(ids)+self.max_new>self.max_seq:
            return {'status':'context_exceeded','text':'','input_tokens':len(ids)}
        if self.model is None:
            config=json.loads((Path(self.adapter)/'adapter_config.json').read_text())
            from lab_runtime import MODEL
            if config['base_model_name_or_path']!=MODEL:raise ValueError('Adaptador de outra base.')
            self.model,_=load_model(self.adapter);self.model.eval()
            for key in ['temperature','top_p','top_k']:setattr(self.model.generation_config,key,None)
        x=torch.tensor([ids],device=self.model.device)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            y=self.model.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=self.max_new,
                do_sample=False,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize();tokens=y[0,len(ids):].tolist()
        return {'text':self.tok.decode(tokens,skip_special_tokens=True).strip(),
                'status':'output_limit' if len(tokens)>=self.max_new and tokens[-1]!=self.tok.eos_token_id else 'ok',
                'latency_s':time.perf_counter()-start,'input_tokens':len(ids),'output_tokens':len(tokens)}


class ToolsLab:
    def __init__(self,data,results,generator=None):
        self.rows=read_jsonl(Path(data)/'test.jsonl');self.generator=generator;self.lock=threading.Lock()
        self.examples=[];self.matches={};self.by_id={r['id']:r for r in self.rows}
        for row in self.rows:
            user=next(m['content'] for m in row['messages'] if m['role']=='user')
            tools,separator,query=user.removeprefix('Tools:\n').rpartition('\nRequest:\n')
            if not separator:raise ValueError('Formato do prompt xLAM inesperado.')
            entry={'id':row['id'],'query':query,'tools':json.loads(tools)}
            self.examples.append(entry);self.matches[self.key(query,entry['tools'])]=row
        self.predictions={};self.metrics={}
        for key,name in [('base','test_base'),('specialist','test_tuned'),('llm','test_llm')]:
            path=Path(results)/(name+'.jsonl')
            if path.exists():
                rows=read_jsonl(path)
                if len({r['id'] for r in rows})!=len(rows) or any(r['id'] not in self.by_id for r in rows):
                    raise ValueError('IDs incompatíveis em '+str(path))
                meta=path.with_suffix('.meta.json')
                if meta.exists():
                    import hashlib
                    if json.loads(meta.read_text())['input_sha256']!=hashlib.sha256((Path(data)/'test.jsonl').read_bytes()).hexdigest():
                        raise ValueError('Predições de outro conjunto: '+str(path))
                self.predictions[key]={r['id']:r for r in rows}
            summary=Path(results)/({'base':'base','specialist':'tuned','llm':'llm'}[key]+'.summary.json')
            if summary.exists():self.metrics[key]=json.loads(summary.read_text())
    @staticmethod
    def key(query,tools):return json.dumps([query,tools],sort_keys=True,ensure_ascii=False)
    def health(self):
        return {'status':'ready','live_enabled':self.generator is not None,
                'model_loaded':bool(getattr(self.generator,'model',None)),
                'examples':len(self.examples),'recorded_available':bool(self.predictions.get('specialist'))}
    def context(self,payload):
        query,tools,mode=validate_input(payload);row=self.matches.get(self.key(query,tools))
        messages=row['messages'] if row else [{'role':'system','content':SYSTEM},
            {'role':'user','content':'Tools:\n'+json.dumps(tools,ensure_ascii=False)+'\nRequest:\n'+query}]
        return query,tools,mode,row,{'messages':messages,'max_new':512,'reference_available':row is not None}
    def predict(self,payload):
        query,tools,mode,row,context=self.context(payload)
        if mode=='recorded':
            if row is None:raise ValueError('O exemplo foi editado. Para uma solicitação nova, ative a inferência real.')
            raw=self.predictions.get('specialist',{}).get(row['id'])
            if raw is None:raise ValueError('Não há predição registrada para este exemplo.')
        else:
            if self.generator is None:raise ValueError('Inferência real desativada para preservar as GPUs. Use o benchmark registrado.')
            if not self.lock.acquire(blocking=False):raise ValueError('Há uma inferência em andamento. Aguarde.')
            try:raw=self.generator(context['messages'])
            finally:self.lock.release()
        result=inspect_output(raw.get('text',''),tools,raw.get('status','ok'))
        result.update({key:raw[key] for key in ['latency_s','input_tokens','output_tokens','usage'] if key in raw})
        recorded={}
        if row:
            for backend,items in self.predictions.items():
                pred=items.get(row['id'])
                if pred:
                    item=inspect_output(pred.get('text',''),tools,pred.get('status','ok'))
                    item['correct']=item['status']=='ok' and canonical(item['calls'])==canonical(row['gold'])
                    item['latency_s']=pred.get('latency_s');recorded[backend]=item
        return {'id':row['id'] if row else None,'mode':mode,'slm':result,'context':context,
                'reference':row['gold'] if row else None,'recorded':recorded,
                'correct':result['status']=='ok' and canonical(result['calls'])==canonical(row['gold']) if row else None,
                'executed':False}


def make_handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,body):
            data=json.dumps(body,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/health':self.reply(200,lab.health())
            elif path=='/catalog':self.reply(200,{'examples':lab.examples[:60],'metrics':lab.metrics,'total':len(lab.examples)})
            elif path=='/example/random':self.reply(200,random.choice(lab.examples))
            else:self.reply(404,{'message':'Rota inexistente.'})
        def do_POST(self):
            if self.path not in ('/predict','/context','/validate'):
                self.reply(404,{'message':'Rota inexistente.'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=150000:raise ValueError('Entrada ausente ou excessiva.')
                payload=json.loads(self.rfile.read(size))
                if self.path=='/predict':result=lab.predict(payload)
                elif self.path=='/context':result=lab.context(payload)[4]
                else:
                    if not isinstance(payload,dict) or set(payload)!={'text','tools','status'}:raise ValueError('Contrato inválido.')
                    validate_input({'query':'validate','tools':payload['tools']})
                    result=inspect_output(payload['text'],payload['tools'],payload['status'])
                self.reply(200,result)
            except (ValueError,TypeError,KeyError):self.reply(400,{'message':'Confira solicitação, catálogo JSON e disponibilidade do modo selecionado.'})
            except Exception:
                import traceback;traceback.print_exc()
                self.reply(503,{'message':'Falha na inferência. Consulte o log da API de ferramentas.'})
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');parser.add_argument('--results',default='results')
    parser.add_argument('--adapter',default='runs/main_r16_lr1e4/final_adapter');parser.add_argument('--enable-gpu',action='store_true')
    parser.add_argument('--port',type=int,default=8081);args=parser.parse_args()
    config=json.loads((Path(args.data)/'length_report.json').read_text())
    gen=Generator(args.adapter,config['max_seq'],config['max_new']) if args.enable_gpu else None
    lab=ToolsLab(args.data,args.results,gen)
    print('API ferramentas pronta em localhost:'+str(args.port)+' / GPU habilitada: '+str(args.enable_gpu),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(lab)).serve_forever()


if __name__=='__main__':main()
