"""Contagens e projeção de duração com latências realmente registradas."""
import argparse
import json
import statistics
from pathlib import Path
from finqa_core import read_json,read_rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data')
    parser.add_argument('--predictions');args=parser.parse_args()
    data=Path(args.data);report=read_json(data/'source_report.json')
    print(json.dumps(report['splits'],indent=2))
    candidates=len(read_rows(data/'train_candidates.jsonl'))
    print(f'Candidatos do professor: {candidates}; treino nativo: 6251; validação: 883; teste: 1147.')
    if args.predictions:
        rows=read_rows(args.predictions);times=[r['latency_s'] for r in rows if r['status']=='ok' and 'latency_s' in r]
        if times:
            mean=statistics.mean(times)
            print(f'{len(times)} gerações ok; média {mean:.2f}s; soma {sum(times)/60:.1f}min.')
            print(f'Projeção para todos os candidatos nesse perfil: {candidates*mean/3600:.2f}h; não inclui carregamento, validação ou falhas.')
        meta=Path(args.predictions).with_suffix('.meta.json')
        if meta.exists():print(json.dumps(read_json(meta).get('gpu_peak',{}),indent=2))


if __name__=='__main__':main()
