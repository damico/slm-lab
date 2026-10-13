"""Baixa revisão FinQA fixada, trava modelos e prepara prompts sem respostas."""
import argparse
import json
import urllib.request
from pathlib import Path
from finqa_core import (FINQA_REVISION, STUDENT, TEACHER, read_json, save_json, write_rows,
    source_map, retrieve, messages, execute, correct, digest, file_hash)


def convert(raw, split, tokenizers, max_seq, max_new, top_k):
    selected = retrieve(raw, top_k)
    context = {'question':raw['qa']['question'], 'sources':selected, 'table':raw['table']}
    prompt = messages(context)
    lengths = {name:len(tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,enable_thinking=False))
               for name,tok in tokenizers.items()}
    eligible = all(n+max_new <= max_seq for n in lengths.values())
    full = {'question':context['question'],'sources':source_map(raw),'table':raw['table']}
    gold = {'evidence':list(raw['qa']['gold_inds']), 'program':raw['qa']['program']}
    reference_valid = False; error = None
    try:
        # O resultado do programa de referência é checado separadamente da elegibilidade.
        reference = execute({'evidence':[next(iter(full['sources']))], 'program':gold['program']},full,grounded=False)
        reference_valid = correct(reference['answer'], raw['qa']['exe_ans'])
        if not reference_valid: error='Programa original diverge de exe_ans na convenção local'
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc: error=str(exc)
    answer=json.dumps(gold,ensure_ascii=False,separators=(',',':'))
    target_valid=False; target_error=None
    try:
        result=execute(gold,context)
        target_valid=correct(result['answer'],raw['qa']['exe_ans'])
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc: target_error=str(exc)
    answer_fits = all(lengths[name]+len(tok.encode(answer,add_special_tokens=False))+1 <= max_seq
                      for name,tok in tokenizers.items())
    page=raw.get('filename',raw['id'].split('.pdf')[0])
    document=page.rsplit('/',1)[0] if '/' in page else page
    return {'id':raw['id'],'split':split,'document':document,'page':page,
            'context':context,'messages':prompt,'eligible':eligible,'prompt_tokens':lengths,
            'gold':raw['qa']['exe_ans'],'gold_output':gold,'answer':answer,
            'reference_valid':reference_valid,'reference_error':error,
            'sft_eligible':eligible and reference_valid and target_valid and answer_fits,
            'target_error':target_error}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',default='raw/finqa');parser.add_argument('--out',default='data')
    parser.add_argument('--max-seq',type=int,default=2048);parser.add_argument('--max-new',type=int,default=512)
    parser.add_argument('--top-k',type=int,default=12);parser.add_argument('--skip-download',action='store_true')
    args=parser.parse_args()
    if args.max_seq <= args.max_new or args.top_k < 1: raise ValueError('Janela/recuperação inválida')
    raw=Path(args.raw);out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use outra pasta --out para preservar os dados')
    raw.mkdir(parents=True,exist_ok=True)
    for name in ['train.json','dev.json','test.json','LICENSE']:
        path=raw/name
        if not args.skip_download:
            remote=('dataset/'+name) if name.endswith('.json') else name
            url=f'https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REVISION}/{remote}'
            with urllib.request.urlopen(url,timeout=120) as response: payload=response.read()
            path.write_bytes(payload)
        if not path.exists(): raise FileNotFoundError(path)
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    lock_path=Path('revisions.json')
    lock=read_json(lock_path) if lock_path.exists() else {}
    toks={}
    for name in [STUDENT,TEACHER]:
        if name not in lock: lock[name]=HfApi().model_info(name).sha;save_json(lock_path,lock)
        toks[name]=AutoTokenizer.from_pretrained(name,revision=lock[name])
    report={'finqa_revision':FINQA_REVISION,'models':lock,'arguments':vars(args),
            'retriever':'TF-IDF local por documento, unigrams+bigrams, somente pergunta/contexto',
            'raw_sha256':{name:file_hash(raw/name) for name in ['train.json','dev.json','test.json','LICENSE']},'splits':{}}
    groups={};all_ids=set()
    for split,name,expected in [('train','train.json',6251),('validation','dev.json',883),('test','test.json',1147)]:
        original=read_json(raw/name)
        if len(original)!=expected: raise ValueError('Contagem inesperada do split fixado: '+split)
        rows=[]
        for source in original:
            if source['id'] in all_ids: raise ValueError('ID repetido entre splits')
            all_ids.add(source['id']);rows.append(convert(source,split,toks,args.max_seq,args.max_new,args.top_k))
        rows.sort(key=lambda r:digest('42:'+r['id']))
        write_rows(out/(split+'.jsonl'),rows)
        usable=[r for r in rows if r['sft_eligible']]
        if split != 'test': write_rows(out/'sft_available'/(split+'.jsonl'),usable)
        if split == 'train': write_rows(out/'train_candidates.jsonl',usable)
        if not usable and split!='test': raise ValueError('Nenhum alvo SFT utilizável: '+split)
        groups[split]={r['document'] for r in rows}
        report['splits'][split]={'total':len(rows),'eligible':sum(r['eligible'] for r in rows),
            'reference_valid':sum(r['reference_valid'] for r in rows),'sft_available':len(usable),
            'gold_sources_retrieved':sum(set(r['gold_output']['evidence'])<=set(r['context']['sources']) for r in rows)}
    report['document_overlap']={f'{a}/{b}':len(groups[a]&groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    save_json(out/'config.json',{'max_seq':args.max_seq,'max_new':args.max_new,'top_k':args.top_k,'models':lock})
    save_json(out/'source_report.json',report)
    demo={'question':'Revenue increased from 100 million to 118 million. What was the percentage growth?',
          'pre_text':[],'post_text':[],'table':[['Year','Revenue (million)'],['Previous','100'],['Current','118']]}
    save_json(out/'demo_input.json',demo)
    save_json(out/'demo_prediction.json',{'evidence':['table_1','table_2'],
        'program':'subtract(118, 100), divide(#0, 100), multiply(#1, const_100)'})
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
