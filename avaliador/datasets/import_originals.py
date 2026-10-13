"""Importa fontes originais dos labs ainda não preparados, sem OCR/modelos/treino."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

FINQA_REVISION='0f16e2867befa6840783e58be38c9efb9229d742'


def write(path,rows):
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def import_receipts(out):
    from datasets import load_dataset,Image
    from huggingface_hub import HfApi,hf_hub_download
    repo='jsdnrs/ICDAR2019-SROIE';info=HfApi().dataset_info(repo);revision=info.sha
    ds=load_dataset(repo,revision=revision)
    images=out/'images';images.mkdir()
    for split in ['train','test']:
        def rows():
            for index,row in enumerate(ds[split].cast_column('image',Image(decode=False))):
                key=str(row.get('key',index));uid=hashlib.sha256((split+':'+key).encode()).hexdigest()[:24]
                image=row['image'];content=image.get('bytes')
                if content is None:content=Path(image['path']).read_bytes()
                path=images/(uid+'.img');path.write_bytes(content)
                # O destino final é registrado antes da renomeação da pasta temporária.
                image_path=out.parent/'03'/'images'/path.name
                yield {'id':split+':'+key,'source_key':key,'transcript':'\n'.join(row['words']),
                    'entities':row['entities'],'image_path':str(image_path.resolve()),
                    'text_origin':'Transcrição anotada do SROIE; não é OCR executado pelo laboratório'}
        write(out/(split+'.jsonl'),rows())
        print('Importado SROIE:',split,flush=True)
    card=hf_hub_download(repo,'README.md',repo_type='dataset',revision=revision)
    shutil.copyfile(card,out/'SOURCE_README.md')
    return {'dataset':repo,'revision':revision,'origin':'original',
        'source':'https://huggingface.co/datasets/'+repo,'splits':'Divisão original train/test; sem validação reservada nem deduplicação do laboratório'}


def import_finqa(out):
    for filename,split in [('train.json','train'),('dev.json','validation'),('test.json','test')]:
        url=f'https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REVISION}/dataset/{filename}'
        with urllib.request.urlopen(url,timeout=120) as response:raw=response.read()
        rows=json.loads(raw)
        # Só campos originais necessários à exploração; não usar model_input anotado como entrada do modelo.
        write(out/(split+'.jsonl'),({'id':r['id'],'pre_text':r['pre_text'],'post_text':r['post_text'],
            'table':r['table'],'qa':{key:r['qa'][key] for key in ['question','program','gold_inds','exe_ans']}}
            for r in rows))
        print('Importado FinQA:',split,len(rows),flush=True)
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REVISION}/LICENSE',timeout=30) as response:
        (out/'LICENSE').write_bytes(response.read())
    return {'dataset':'FinQA','revision':FINQA_REVISION,'origin':'original',
        'source':'https://github.com/czyssrs/FinQA','splits':'Divisão original; contexto completo, sem recuperação/limite de tokens do Lab 05'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lab',choices=['03','05','all'],default='all')
    parser.add_argument('--cache',default=str(Path.home()/'slm-labs/.dataset-explorer'));args=parser.parse_args()
    cache=Path(args.cache)/'originals';cache.mkdir(parents=True,exist_ok=True)
    for lab in ['03','05'] if args.lab=='all' else [args.lab]:
        target=cache/lab
        if target.exists():print(f'Lab {lab}: fonte já importada; preservada.');continue
        stage=cache/(lab+'.pending')
        if stage.exists():raise FileExistsError('Importação incompleta: '+str(stage)+'; preserve/renomeie para tentar novamente')
        stage.mkdir()
        manifest=import_receipts(stage) if lab=='03' else import_finqa(stage)
        (stage/'source.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        stage.rename(target)
        print(f'Lab {lab}: fonte original disponível em {target}',flush=True)


if __name__=='__main__':main()
