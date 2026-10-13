"""API de perguntas financeiras: geração local e cálculo restrito; sem PDFs."""
import argparse
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from finqa_core import STUDENT,read_json,retrieve,messages,parse_output,execute
from finqa_runtime import Generator,prompt_ids


def context_from_payload(payload,top_k=12):
    if not isinstance(payload,dict) or set(payload)!={'question','pre_text','post_text','table'}:
        raise ValueError('Informe somente question, pre_text, post_text e table')
    if not isinstance(payload['question'],str) or not 0<len(payload['question'])<=8000:raise ValueError('Pergunta inválida')
    for key in ['pre_text','post_text']:
        if not isinstance(payload[key],list) or len(payload[key])>256 or any(not isinstance(s,str) or len(s)>4000 for s in payload[key]):raise ValueError('Texto inválido')
    table=payload['table']
    if not isinstance(table,list) or not table or len(table)>128 or any(not isinstance(row,list) or not 1<=len(row)<=32 or any(not isinstance(v,str) or len(v)>2000 for v in row) for row in table):raise ValueError('Tabela inválida')
    raw={**payload,'qa':{'question':payload['question']}}
    return {'question':payload['question'],'sources':retrieve(raw,top_k),'table':table}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data')
    parser.add_argument('--adapter',required=True);parser.add_argument('--port',type=int,default=8085)
    args=parser.parse_args();config=read_json(args.data+'/config.json');generator=Generator(STUDENT,args.adapter)
    class Handler(BaseHTTPRequestHandler):
        def answer(self,status,body):
            raw=json.dumps(body,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path=='/health':self.answer(200,{'status':'ready','model':STUDENT})
            else:self.answer(404,{'error':'Rota desconhecida'})
        def do_POST(self):
            if self.path!='/predict':self.answer(404,{'error':'Rota desconhecida'});return
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=100000:raise ValueError('Tamanho inválido')
                context=context_from_payload(json.loads(self.rfile.read(size)),config['top_k']);prompt=messages(context)
                if len(prompt_ids(generator.tok,prompt))+config['max_new']>config['max_seq']:
                    self.answer(200,{'status':'context_exceeded','requires_review':True});return
                prediction=generator(prompt,config['max_new']);result=None
                if prediction['status']=='ok':
                    try:result=execute(parse_output(prediction['text']),context)
                    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as error:
                        prediction.update(status='invalid_program',error=str(error))
                self.answer(200,{'prediction':prediction,'execution':result,'requires_review':True,
                    'note':'Cálculo executado com fontes citadas; revise a interpretação e as unidades antes de usar.'})
            except (ValueError,TypeError,KeyError) as error:self.answer(400,{'error':str(error)})
            except Exception:self.answer(500,{'error':'Falha no modelo; confira o log do servidor'})
    print(f'API FinQA em http://127.0.0.1:{args.port}/predict',flush=True)
    with HTTPServer(('127.0.0.1',args.port),Handler) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
