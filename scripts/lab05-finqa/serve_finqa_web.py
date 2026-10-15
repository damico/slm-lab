"""Exploração FinQA: benchmark preservado em CPU e aluno real opcional em GPU."""
import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from finqa_core import STUDENT, correct, execute, file_hash, messages, parse_output, read_json, read_rows
from finqa_runtime import Generator, prompt_ids
from serve_finqa import context_from_payload


def checked_context(payload):
    if not isinstance(payload, dict) or set(payload) != {'question', 'sources', 'table'}:
        raise ValueError('Informe somente pergunta, fontes e tabela; não envie gabaritos.')
    question, sources, table = payload['question'], payload['sources'], payload['table']
    if not isinstance(question, str) or not question.strip() or len(question) > 8000:
        raise ValueError('Pergunta inválida.')
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 50 or any(
        not isinstance(k, str) or not 0 < len(k) <= 120 or not isinstance(v, str) or len(v) > 8000 for k, v in sources.items()):
        raise ValueError('Fontes inválidas.')
    if not isinstance(table, list) or not 1 <= len(table) <= 128 or any(
        not isinstance(row, list) or not 1 <= len(row) <= 32 or any(not isinstance(v, str) or len(v) > 2000 for v in row) for row in table):
        raise ValueError('Tabela inválida.')
    return {'question': question, 'sources': sources, 'table': table}


class FinqaLab:
    def __init__(self, data, results, adapter=None, enable_gpu=False):
        self.data, self.results = Path(data), Path(results)
        self.config = read_json(self.data/'config.json')
        self.rows = read_rows(self.data/'test.jsonl')
        self.by_id = {r['id']: r for r in self.rows}
        if len(self.by_id) != len(self.rows): raise ValueError('IDs de teste duplicados.')
        self.predictions = {}; self.metrics = {}
        for backend in ['base', 'sft', 'distilled', 'teacher']:
            path = self.results/f'test_{backend}.jsonl'
            if path.exists():
                if not path.with_suffix('.meta.json').exists() or read_json(path.with_suffix('.meta.json'))['signature']['input_sha256'] != file_hash(self.data/'test.jsonl'):
                    raise ValueError('Benchmark de outro arquivo de teste.')
                rows = read_rows(path); mapped = {p['id']: p for p in rows}
                if len(mapped) != len(rows) or not mapped.keys() <= self.by_id.keys(): raise ValueError('IDs de previsão inválidos.')
                self.predictions[backend] = mapped
            summary = self.results/f'{backend}.summary.json'
            if summary.exists(): self.metrics[backend] = read_json(summary)
        self.adapter = adapter; self.enable_gpu = enable_gpu; self.generator = None
        self.lock = threading.Lock()

    def health(self):
        return {'status':'ready', 'live_enabled':self.enable_gpu, 'model_loaded':self.generator is not None,
                'examples':len(self.rows), 'recorded_available':bool(self.predictions), 'model':STUDENT}

    def example(self, row):
        return {'id':row['id'], 'document':row.get('document'), 'context':row['context'], 'eligible':row.get('eligible', True)}

    def catalog(self):
        return {'examples':[{'id':r['id'], 'question':r['context']['question']} for r in self.rows],
                'metrics':self.metrics, **self.health()}

    def inspect(self, prediction, context, reference=None):
        out = {k:prediction[k] for k in ['status','text','latency_s','input_tokens','output_tokens'] if k in prediction}
        out['execution'] = None; out['program_valid'] = False
        if out.get('status') == 'ok':
            try:
                parsed = parse_output(out.get('text','')); out['program'] = parsed['program']
                out['execution'] = execute(parsed, context); out['program_valid'] = True
            except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as error:
                out.update(status='invalid_program', error=str(error))
        if reference is not None:
            out['correct'] = bool(out['execution'] and correct(out['execution']['answer'], reference['gold']))
        return out

    def predict(self, payload):
        if not isinstance(payload, dict): raise ValueError('Entrada inválida.')
        if set(payload) == {'question','pre_text','post_text','table'}:
            payload = {'context':context_from_payload(payload, self.config['top_k']), 'mode':'specialist'}
        if not set(payload) <= {'context','example_id','mode'} or 'context' not in payload:
            raise ValueError('Campos desconhecidos; não envie referências, programas ou respostas.')
        context = checked_context(payload['context']); mode = payload.get('mode','recorded')
        if mode not in ('recorded','specialist'): raise ValueError('Modo inválido.')
        example_id = payload.get('example_id')
        if example_id is not None and (not isinstance(example_id,str) or example_id not in self.by_id):
            raise ValueError('Exemplo desconhecido.')
        row = self.by_id.get(example_id)
        if row is not None and row['context'] != context: row = None
        recorded = {key:self.inspect(predictions[row['id']], context, row) for key, predictions in self.predictions.items()
                    if row is not None and row['id'] in predictions}
        if mode == 'recorded':
            if row is None or 'distilled' not in recorded: raise ValueError('Use um exemplo original para consultar o benchmark. Entradas editadas precisam de inferência real.')
            slm = recorded['distilled']
        else:
            if not self.enable_gpu: raise ValueError('Inferência real desabilitada; habilite a GPU no serviço.')
            if not self.lock.acquire(blocking=False): raise ValueError('Modelo ocupado; aguarde o teste atual.')
            try:
                if self.generator is None: self.generator = Generator(STUDENT, self.adapter)
                prompt = messages(context)
                if len(prompt_ids(self.generator.tok, prompt)) + self.config['max_new'] > self.config['max_seq']:
                    prediction = {'status':'context_exceeded', 'text':''}
                else: prediction = self.generator(prompt, self.config['max_new'])
                slm = self.inspect(prediction, context, row)
            finally: self.lock.release()
        reference = {'answer':row['gold'], 'program':row['gold_output']['program'], 'evidence':row['gold_output']['evidence']} if row else None
        return {'mode':mode, 'example_id':row['id'] if row else None, 'slm':slm, 'recorded':recorded,
                'context':context, 'reference':reference, 'requires_review':True,
                'note':'Confira interpretação e unidades. Programa executável não garante resposta correta.'}


def handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body):
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/health': self.answer(200,lab.health())
            elif parsed.path == '/catalog': self.answer(200,lab.catalog())
            elif parsed.path == '/example/random': self.answer(200,lab.example(random.choice(lab.rows)))
            elif parsed.path == '/example':
                row = lab.by_id.get(parse_qs(parsed.query).get('id',[''])[0])
                self.answer(200,lab.example(row)) if row else self.answer(404,{'message':'Exemplo inexistente.'})
            else: self.answer(404,{'message':'Rota desconhecida.'})
        def do_POST(self):
            if self.path != '/predict': self.answer(404,{'message':'Rota desconhecida.'}); return
            try:
                size = int(self.headers.get('Content-Length',0))
                if not 0 < size <= 150000: raise ValueError('Entrada deve ter até 150 KB.')
                self.answer(200, lab.predict(json.loads(self.rfile.read(size))))
            except (ValueError, TypeError, KeyError) as error: self.answer(400,{'message':str(error)})
            except Exception: self.answer(503,{'message':'Falha na inferência; consulte o log da API FinQA.'})
    return Handler


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data');p.add_argument('--results',default='results')
    p.add_argument('--adapter',default='runs/main_distilled/final_adapter');p.add_argument('--enable-gpu',action='store_true')
    p.add_argument('--port',type=int,default=8085);args=p.parse_args()
    lab=FinqaLab(args.data,args.results,args.adapter,args.enable_gpu)
    print(f'FinQA web / porta {args.port} / GPU habilitada: {args.enable_gpu}',flush=True)
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler(lab)) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass

if __name__ == '__main__': main()
