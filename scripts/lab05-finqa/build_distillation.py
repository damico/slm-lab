"""Aprova respostas do professor e produz dois treinos com os mesmos IDs."""
import argparse
import json
from pathlib import Path
from finqa_core import (TEACHER, STUDENT, read_json, read_rows, write_rows, save_json,
    parse_output, execute, correct, file_hash, digest)
from finqa_runtime import tokenizer, prompt_ids


def select_pairs(rows, predictions, tok, config):
    if any(r['split']!='train' for r in rows):raise ValueError('Destilação aceita somente split train')
    ids={r['id'] for r in rows}
    if len(ids)!=len(rows) or len({p['id'] for p in predictions})!=len(predictions):raise ValueError('IDs duplicados')
    if any(p['id'] not in ids for p in predictions):raise ValueError('Predição fora do treino')
    pred={p['id']:p for p in predictions};gold_rows=[];teacher_rows=[];audit=[]
    for row in rows:
        reason=None;p=pred.get(row['id'])
        try:
            if not p:raise ValueError('not_generated')
            if p.get('model')!=TEACHER:raise ValueError('not_teacher')
            if not row['sft_eligible']:raise ValueError('gold_target_unavailable')
            if p['status']!='ok':raise ValueError(p['status'])
            target=parse_output(p['text']);execution=execute(target,row['context'])
            if not correct(execution['answer'],row['gold']):raise ValueError('wrong_execution')
            answer=json.dumps(target,ensure_ascii=False,separators=(',',':'))
            if len(prompt_ids(tok,row['messages']))+len(tok.encode(answer,add_special_tokens=False))+1>config['max_seq']:
                raise ValueError('teacher_target_too_long')
        except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc:reason=str(exc)
        audit.append({'id':row['id'],'approved':reason is None,'reason':reason})
        if reason is None:
            gold_rows.append({**row,'supervision':'gold'})
            teacher_rows.append({**row,'answer':answer,'supervision':'teacher','teacher_execution':execution})
    return gold_rows,teacher_rows,audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data')
    parser.add_argument('--predictions',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--review-size',type=int,default=20);args=parser.parse_args()
    data=Path(args.data);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use outra pasta --out')
    meta=read_json(Path(args.predictions).with_suffix('.meta.json'))
    if meta['signature']['model']!=TEACHER:raise ValueError('Metadados não são do professor Qwen3-8B')
    config=read_json(data/'config.json')
    if any(meta['signature'][key]!=config[key] for key in ['max_seq','max_new']):raise ValueError('Janela do professor mudou')
    if meta['signature']['revision']!=config['models'][TEACHER]:raise ValueError('Revisão do professor mudou')
    if meta['signature']['input_sha256']!=file_hash(data/'train_candidates.jsonl'):raise ValueError('Professor recebeu outro conjunto')
    rows=read_rows(data/'train_candidates.jsonl');predictions=read_rows(args.predictions)
    gold,teacher,audit=select_pairs(rows,predictions,tokenizer(STUDENT),config)
    write_rows(out/'audit.jsonl',audit)
    if not teacher:raise ValueError('Nenhuma resposta aprovada. Consulte audit.jsonl e inspecione o piloto')
    validation=read_rows(data/'sft_available/validation.jsonl')
    if not validation:raise ValueError('Validação vazia')
    for name,training in [('sft',gold),('distilled',teacher)]:
        write_rows(out/name/'train.jsonl',training);write_rows(out/name/'validation.jsonl',validation)
        save_json(out/name/'config.json',config)
    review=sorted(teacher,key=lambda r:digest('review:'+r['id']))[:args.review_size]
    write_rows(out/'review.jsonl',[{'id':r['id'],'question':r['context']['question'],'sources':r['context']['sources'],
        'teacher_output':r['answer'],'teacher_execution':r['teacher_execution'],'gold_output':r['gold_output'],'gold':r['gold']} for r in review])
    save_json(out/'selection_report.json',{'candidates':len(rows),'generated':len(predictions),'approved':len(teacher),
        'approval_rate_generated':len(teacher)/len(predictions),'paired_ids':[r['id'] for r in teacher],
        'validation_supervision':'mesmos gabaritos nos dois treinos',
        'train_input_sha256':file_hash(data/'train_candidates.jsonl'),'predictions_sha256':file_hash(args.predictions),
        'teacher_meta':meta,'review_cases':len(review),'arguments':vars(args)})
    print(f'Aprovados {len(teacher)}/{len(predictions)}; dois treinos com IDs idênticos. Revise {out}/review.jsonl antes do treino.')


if __name__=='__main__':main()
