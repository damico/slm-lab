"""Execução verificável, métricas locais e comparação pareada por documento."""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from finqa_core import read_rows, read_json, save_json, write_rows, parse_output, execute, correct, source_map, file_hash


def percentile(values,p):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*p;i=math.floor(x);j=math.ceil(x)
    return values[i]+(values[j]-values[i])*(x-i)


def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z/den*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [center-half,center+half]


def score(reference,predictions,out):
    rows=read_rows(reference);pred=read_rows(predictions);mapping={p['id']:p for p in pred}
    if len(mapping)!=len(pred) or not set(mapping)<={r['id'] for r in rows}:raise ValueError('IDs inválidos ou duplicados')
    cases=[]
    for row in rows:
        p=mapping.get(row['id'],{'status':'missing','text':''})
        case={'id':row['id'],'document':row['document'],'eligible':row['eligible'],
              'reference_valid':row['reference_valid'],'status':p['status'],'program_valid':False,'correct':False}
        if p['status']=='ok':
            try:
                output=parse_output(p['text']);result=execute(output,row['context'])
                case.update(program_valid=True,answer=result['answer'],trace=result['trace'],evidence=result['evidence'],
                    correct=correct(result['answer'],row['gold']))
                gold_set=set(row['gold_output']['evidence']);pred_set=set(output['evidence'])
                case['evidence_recall']=len(gold_set & pred_set)/len(gold_set) if gold_set else None
                case['exact_program']=output['program']==row['gold_output']['program']
            except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc:case['error']=str(exc)
        cases.append(case)
    valid=[c for c in cases if c['reference_valid']];eligible=[c for c in valid if c['eligible']]
    k=sum(c['correct'] for c in eligible);lat=[p['latency_s'] for p in pred if p['status']=='ok' and 'latency_s' in p]
    summary={'metric':'local_grounded_execution_accuracy_not_official_FinQA',
        'n_total':len(cases),'n_reference_valid':len(valid),'n_eligible_valid':len(eligible),'n_correct':k,
        'accuracy_eligible':k/len(eligible) if eligible else None,
        'accuracy_all_valid':sum(c['correct'] for c in valid)/len(valid) if valid else None,
        'coverage':sum(c['eligible'] for c in cases)/len(cases) if cases else None,
        'program_valid_rate':sum(c['program_valid'] for c in eligible)/len(eligible) if eligible else None,
        'wilson95':wilson(k,len(eligible)),'latency_success_n':len(lat),
        'p50_s':statistics.median(lat) if lat else None,'p95_s':percentile(lat,.95),
        'input_tokens':sum(p.get('input_tokens',0) for p in pred),'output_tokens':sum(p.get('output_tokens',0) for p in pred),
        'reference_sha256':file_hash(reference),'predictions_sha256':file_hash(predictions)}
    write_rows(str(out)+'.cases.jsonl',cases);save_json(str(out)+'.summary.json',summary)
    print(json.dumps(summary,indent=2))


def compare(a,b):
    left={c['id']:c for c in read_rows(a)};right={c['id']:c for c in read_rows(b)}
    if left.keys()!=right.keys():raise ValueError('Compare os mesmos IDs')
    groups=defaultdict(list)
    for key,l in left.items():
        r=right[key]
        if (l['eligible'],l['reference_valid'],l['document'])!=(r['eligible'],r['reference_valid'],r['document']):raise ValueError('Referências distintas')
        if l['eligible'] and l['reference_valid']:groups[l['document']].append(int(r['correct'])-int(l['correct']))
    if not groups:raise ValueError('Sem casos comparáveis')
    clusters=list(groups.values());n=sum(len(c) for c in clusters);rng=random.Random(42);boot=[]
    for _ in range(2000):
        sample=rng.choices(clusters,k=len(clusters))
        boot.append(sum(sum(c) for c in sample)/sum(len(c) for c in sample))
    print(json.dumps({'n':n,'documents':len(clusters),'delta_B_minus_A':sum(sum(c) for c in clusters)/n,
                     'cluster_bootstrap95':[percentile(boot,.025),percentile(boot,.975)]},indent=2))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    sc=sub.add_parser('score');sc.add_argument('reference');sc.add_argument('predictions');sc.add_argument('--out',required=True)
    cp=sub.add_parser('compare');cp.add_argument('a');cp.add_argument('b')
    ex=sub.add_parser('execute');ex.add_argument('--input',required=True);ex.add_argument('--prediction',required=True)
    args=parser.parse_args()
    if args.command=='score':score(args.reference,args.predictions,args.out)
    elif args.command=='compare':compare(args.a,args.b)
    else:
        raw=read_json(args.input);context={'question':raw['question'],'sources':source_map(raw),'table':raw['table']}
        print(json.dumps(execute(read_json(args.prediction),context),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
