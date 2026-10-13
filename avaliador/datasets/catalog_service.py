"""Explorador de datasets local, indexado em SQLite, sem modelos ou CUDA."""
from contextlib import contextmanager
import argparse
import json
import mimetypes
import os
import random
import re
import sqlite3
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs

LABS = [
 ('01','Ferramentas','xLAM','Solicitações, ferramentas disponíveis e chamadas esperadas.'),
 ('02','SQL','BIRD','Perguntas sobre bancos, esquemas e consultas de referência.'),
 ('03','Recibos','SROIE','Recibos, texto e campos que queremos extrair.'),
 ('04','Decisões','ABCD','Históricos de atendimento e a próxima etapa anotada.'),
 ('05','Cálculos financeiros','FinQA','Tabelas, perguntas, evidências e programas de cálculo.')]
SPLITS={'train':'Treino','validation':'Validação','test':'Teste','calibration':'Calibração','selection':'Seleção do limiar'}


def object_value(value):
    if isinstance(value,str):
        try:return json.loads(value)
        except ValueError:return value
    return value


def normalize(lab,row):
    """Separação explícita entre entrada e referência; nunca inferência."""
    result={'id':str(row.get('id',row.get('source_key',''))),'metadata':{},'raw':row}
    result['metadata']={key:row[key] for key in ['eligible','prompt_tokens','difficulty','db_id','label','split',
        'dropped_turns','supervision','reference_valid','document','page','source_key','ocr_empty'] if key in row}
    user=next((m.get('content','') for m in row.get('messages',[]) if m.get('role')=='user'),'')
    if lab=='01':
        tools,separator,question=user.removeprefix('Tools:\n').rpartition('\nRequest:\n')
        result.update(question=question if separator else row.get('query',user),
            tools=object_value(tools) if separator else object_value(row.get('tools',[])),
            reference=object_value(row.get('gold',row.get('answer',row.get('answers',[])))))
    elif lab=='02':
        body,separator,question=user.rpartition('\nQuestion:\n');schema,_,evidence=body.partition('\nEvidence:\n')
        result.update(question=question if separator else row.get('question',user),
            schema=schema.removeprefix('Schema (PK numbers identify composite-key order):\n'),evidence=evidence,
            reference=row.get('gold',row.get('answer','')),db_id=row.get('db_id',''))
    elif lab=='03':
        text=row.get('text',row.get('transcript',''))
        if not text and user:
            decoded=object_value(user);text=decoded.get('receipt_text',user) if isinstance(decoded,dict) else user
        result.update(question='Quais são a empresa, a data, o endereço e o total deste recibo?',
            receipt_text=text,reference=object_value(row.get('answer',row.get('entities',{}))),
            image_path=row.get('image_path'),text_origin=row.get('text_origin','Texto usado no laboratório'))
    elif lab=='04':
        context=row.get('context',{})
        if not context:
            text=row.get('text','');pos=text.find('\n{')
            if pos>=0:
                try:context=json.loads(text[pos+1:])
                except ValueError:context={}
        history=row.get('history',context.get('history',[]))
        latest=next((e.get('text','') for e in reversed(history) if e.get('role')=='customer'),'Histórico de atendimento')
        result.update(question=latest,history=history,rules=context.get('guideline_excerpts',[]),
            reference=row.get('label',row.get('expected','')),policies=row.get('policy_ids',[]))
    elif lab=='05':
        context=row.get('context',{});qa=row.get('qa',{})
        reference=object_value(row['answer']) if row.get('supervision')=='teacher' else row.get('gold_output')
        if reference is None:reference={'evidence':list(qa.get('gold_inds',{})),'program':qa.get('program','')}
        result.update(question=context.get('question',qa.get('question',row.get('question',''))),
            table=context.get('table',row.get('table',[])),sources=context.get('sources',{}),
            pre_text=row.get('pre_text',[]),post_text=row.get('post_text',[]),reference=reference,
            expected_result=row.get('gold',qa.get('exe_ans')),reference_sources=qa.get('gold_inds',{}))
    return result


class Catalog:
    def __init__(self,cache,roots):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.database=self.cache/'index.sqlite';self.roots=roots
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''CREATE TABLE IF NOT EXISTS datasets(
                id TEXT PRIMARY KEY,lab TEXT,version TEXT,version_label TEXT,split TEXT,origin TEXT,
                path TEXT,signature TEXT,total INTEGER,eligible INTEGER);
                CREATE TABLE IF NOT EXISTS samples(dataset TEXT,ordinal INTEGER,identifier TEXT,
                    title TEXT,preview TEXT,eligible INTEGER,label TEXT,offset INTEGER,length INTEGER,
                    PRIMARY KEY(dataset,ordinal));
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text,tokenize='unicode61 remove_diacritics 2');
                CREATE INDEX IF NOT EXISTS sample_scope ON samples(dataset,eligible,ordinal);''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.database,timeout=60);db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()
    def discover(self):
        files=[]
        for lab,root in self.roots.items():
            for split in SPLITS:
                path=root/'data'/(split+'.jsonl')
                if path.exists():files.append((lab,'prepared','Dados preparados do laboratório',split,'prepared',path))
            if lab=='05':
                for folder in ['paired_main','paired_pilot']:
                    for mode in ['sft','distilled']:
                        for split in ['train','validation']:
                            path=root/folder/mode/(split+'.jsonl')
                            if path.exists():files.append((lab,folder+'_'+mode,
                                ('Treino completo' if folder=='paired_main' else 'Piloto')+' / '+('gabaritos' if mode=='sft' else 'professor'),split,'prepared',path))
        for lab,_,_,_ in LABS:
            for split in ['train','validation','test']:
                path=self.cache/'originals'/lab/(split+'.jsonl')
                if path.exists():files.append((lab,'original','Fonte original / ainda não preparada',split,'original',path))
        return files
    def refresh(self):
        files=self.discover();present=set()
        for lab,version,label,split,origin,path in files:
            dataset=f'{lab}-{version}-{split}';present.add(dataset)
            stat=path.stat();signature=f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
            with self.connect() as db:
                old=db.execute('SELECT signature FROM datasets WHERE id=?',(dataset,)).fetchone()
                if old and old['signature']==signature:continue
                db.execute('DELETE FROM search WHERE rowid IN (SELECT rowid FROM samples WHERE dataset=?)',(dataset,))
                db.execute('DELETE FROM samples WHERE dataset=?',(dataset,))
                total=0;eligible=0
                with path.open('rb') as stream:
                    while True:
                        offset=stream.tell();line=stream.readline()
                        if not line:break
                        if not line.strip():continue
                        row=json.loads(line);item=normalize(lab,row)
                        title=item.get('question') or item['id'];preview=re.sub(r'\s+',' ',title)[:180]
                        flag=int(row['eligible']) if isinstance(row.get('eligible'),bool) else None
                        cursor=db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)',
                            (dataset,total,item['id'],title[:160],preview,flag,str(row.get('label',row.get('db_id',''))),offset,len(line)))
                        search=json.dumps({k:v for k,v in item.items() if k not in ('raw','image_path')},ensure_ascii=False)
                        db.execute('INSERT INTO search(rowid,text) VALUES(?,?)',(cursor.lastrowid,search[:20000]))
                        total+=1;eligible+=flag==1
                db.execute('INSERT OR REPLACE INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (dataset,lab,version,label,split,origin,str(path.resolve()),signature,total,eligible))
                print(f'Indexado {dataset}: {total} exemplos',flush=True)
        with self.connect() as db:
            for row in db.execute('SELECT id FROM datasets').fetchall():
                if row['id'] not in present:
                    db.execute('DELETE FROM search WHERE rowid IN (SELECT rowid FROM samples WHERE dataset=?)',(row['id'],))
                    db.execute('DELETE FROM samples WHERE dataset=?',(row['id'],));db.execute('DELETE FROM datasets WHERE id=?',(row['id'],))
    def catalog(self):
        with self.connect() as db:datasets=[dict(row) for row in db.execute('SELECT * FROM datasets ORDER BY lab,origin DESC,version,split')]
        labs=[]
        for code,title,name,description in LABS:
            versions={}
            for row in datasets:
                if row['lab']!=code:continue
                version=versions.setdefault(row['version'],{'id':row['version'],'label':row['version_label'],
                    'origin':row['origin'],'splits':[]})
                version['splits'].append({'id':row['split'],'label':SPLITS[row['split']],
                    'dataset_id':row['id'],'count':row['total'],'eligible':row['eligible']})
            variants=sorted(versions.values(),key=lambda v:(v['id']!='prepared',v['origin']=='original',v['id']))
            train=sum(s['count'] for s in variants[0]['splits'] if s['id']=='train') if variants else 0
            labs.append({'id':code,'title':title,'dataset':name,'description':description,'available':bool(versions),
                'train_count':train,'versions':variants})
        return {'labs':labs,'read_only':True,'gpu_required':False}
    def scope(self,db,dataset):
        row=db.execute('SELECT * FROM datasets WHERE id=?',(dataset,)).fetchone()
        if not row:raise ValueError('Dataset desconhecido')
        path=Path(row['path']);stat=path.stat()
        current=f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
        if current!=row['signature']:raise ValueError('Dados locais mudaram. Atualize o índice antes de explorar.')
        return row
    def samples(self,dataset,q='',page=1,size=20,eligible='all',random_case=False):
        if len(q)>120 or page<1 or not 1<=size<=30 or eligible not in ('all','yes','no'):raise ValueError('Filtro inválido')
        with self.connect() as db:
            scope=self.scope(db,dataset);params=[dataset];where='s.dataset=?';join=''
            if eligible!='all':where+=' AND s.eligible=?';params.append(int(eligible=='yes'))
            words=re.findall(r'\w+',q)[:12]
            if words:
                join=' JOIN search ON search.rowid=s.rowid';where+=' AND search MATCH ?'
                params.append(' AND '.join('"'+word+'"*' for word in words))
            total=db.execute('SELECT count(*) FROM samples s'+join+' WHERE '+where,params).fetchone()[0]
            offset=(page-1)*size
            if random_case:
                offset=random.randrange(total) if total else 0;size=1
            rows=db.execute('SELECT s.ordinal,s.identifier,s.title,s.preview,s.eligible,s.label FROM samples s'+join+
                ' WHERE '+where+' ORDER BY s.ordinal LIMIT ? OFFSET ?',params+[size,offset]).fetchall()
            return {'dataset_id':dataset,'lab':scope['lab'],'origin':scope['origin'],'total':total,'page':page,
                    'page_size':size,'offset':offset,'items':[dict(row) for row in rows]}
    def detail(self,dataset,ordinal):
        with self.connect() as db:
            scope=self.scope(db,dataset);record=db.execute('SELECT * FROM samples WHERE dataset=? AND ordinal=?',(dataset,ordinal)).fetchone()
            if not record:raise ValueError('Exemplo inexistente')
            with Path(scope['path']).open('rb') as stream:stream.seek(record['offset']);row=json.loads(stream.read(record['length']))
            item=normalize(scope['lab'],row)
            item.update(dataset_id=dataset,ordinal=ordinal,lab=scope['lab'],split=scope['split'],origin=scope['origin'])
            item['has_image']=bool(self.image_path(item))
            item.pop('image_path',None)
            return item
    def image_path(self,item):
        if item.get('lab','03')!='03' or not item.get('image_path'):return None
        path=Path(item['image_path'])
        if not path.is_absolute():path=self.roots['03']/path
        path=path.resolve()
        allowed=[(self.roots['03']/'raw/images').resolve(),(self.cache/'originals/03/images').resolve()]
        if not any(path.is_relative_to(root) for root in allowed) or not path.is_file():return None
        return path
    def image(self,dataset,ordinal):
        with self.connect() as db:
            scope=self.scope(db,dataset)
            if scope['lab']!='03':raise ValueError('Imagem não disponível')
            record=db.execute('SELECT offset,length FROM samples WHERE dataset=? AND ordinal=?',(dataset,ordinal)).fetchone()
            if not record:raise ValueError('Exemplo inexistente')
            with Path(scope['path']).open('rb') as stream:stream.seek(record['offset']);row=json.loads(stream.read(record['length']))
        path=self.image_path(normalize('03',row))
        if not path:raise ValueError('Imagem não disponível')
        raw=path.read_bytes()
        if raw.startswith(b'\x89PNG'):mime='image/png'
        elif raw.startswith(b'\xff\xd8\xff'):mime='image/jpeg'
        elif raw.startswith((b'GIF87a',b'GIF89a')):mime='image/gif'
        elif raw.startswith(b'RIFF') and raw[8:12]==b'WEBP':mime='image/webp'
        else:raise ValueError('Formato de imagem não suportado')
        return raw,mime


def create_server(catalog,port):
    class Handler(BaseHTTPRequestHandler):
        def answer(self,status,payload,mime='application/json; charset=utf-8'):
            raw=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            request=urlparse(self.path);params=parse_qs(request.query)
            get=lambda key,default='':params.get(key,[default])[0]
            try:
                if request.path=='/health':self.answer(200,{'status':'ready','gpu_required':False})
                elif request.path=='/catalog':self.answer(200,catalog.catalog())
                elif request.path=='/samples':self.answer(200,catalog.samples(get('dataset'),get('q'),int(get('page','1')),int(get('size','20')),get('eligible','all'),get('random')=='true'))
                elif request.path=='/detail':self.answer(200,catalog.detail(get('dataset'),int(get('ordinal'))))
                elif request.path=='/image':
                    raw,mime=catalog.image(get('dataset'),int(get('ordinal')));self.answer(200,raw,mime)
                else:self.answer(404,{'error':'Rota desconhecida'})
            except (ValueError,TypeError,FileNotFoundError) as error:self.answer(400,{'error':str(error)})
            except Exception:self.answer(500,{'error':'Não foi possível ler o dataset local'})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',default=str(Path.home()/'slm-labs/.dataset-explorer'))
    parser.add_argument('--port',type=int,default=8086);parser.add_argument('--index-only',action='store_true');args=parser.parse_args()
    shared=Path(os.environ.get('LAB_SHARED_DIR',str(Path.home()/'slm-labs')))
    roots={lab:Path(os.environ.get('LAB'+lab+'_DIR',str(shared/folder))) for lab,folder in
        [('01','01-tools'),('02','02-sql'),('03','03-recibos'),('05','05-finqa')]}
    roots['04']=Path(os.environ.get('LAB04_DIR',str(Path.home()/'lab04-decisoes')))
    catalog=Catalog(args.cache,roots);catalog.refresh()
    if args.index_only:return
    print(f'Explorador de datasets em http://127.0.0.1:{args.port} / somente leitura / sem GPU',flush=True)
    with create_server(catalog,args.port) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
