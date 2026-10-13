"""Conta chamadas xLAM e estima custo/tempo a partir do consumo informado pela API."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='?', const='data/test.jsonl', default='data/test.jsonl')
    parser.add_argument('--predictions', default='results/test_llm.jsonl')
    parser.add_argument('--input-price', type=float, default=0.40, help='USD por milhão de tokens de entrada')
    parser.add_argument('--output-price', type=float, default=1.60, help='USD por milhão de tokens de saída')
    args = parser.parse_args()
    rows = read_rows(args.test)
    eligible = sum(bool(row.get('eligible', True)) for row in rows)
    print(f'Teste: {eligible} chamadas previstas em {len(rows)} linhas totais; fora do contexto: {len(rows)-eligible}.')
    path = Path(args.predictions)
    if not path.exists():
        return
    results = read_rows(path)
    used = [r for r in results if r.get('usage')]
    input_tokens = sum(r['usage'].get('prompt_tokens', 0) for r in used)
    output_tokens = sum(r['usage'].get('completion_tokens', 0) for r in used)
    cost = (input_tokens * args.input_price + output_tokens * args.output_price) / 1_000_000
    print(f'Arquivo: {path}; linhas registradas: {len(results)}; respostas com consumo informado: {len(used)}.')
    print(f'Tokens: {input_tokens} entrada, {output_tokens} saída.')
    print(f'Custo estimado em USD pelas tarifas configuradas: ${cost:.6f}.')
    if used:
        print(f'Projeção para {eligible} chamadas com o consumo médio deste arquivo: ${cost / len(used) * eligible:.4f}.')
    timed = [r['latency_s'] for r in results if r.get('status') == 'ok' and 'latency_s' in r]
    if timed:
        mean = sum(timed) / len(timed)
        print(f'Latência média das {len(timed)} respostas ok: {mean:.3f} s; ritmo sequencial: {60/mean:.1f} chamadas/min.' if mean > 0 else 'Latência média: 0 s.')
        print(f'Projeção sequencial para {eligible} chamadas nesse ritmo: {mean * eligible / 60:.1f} min.')
    print('Projeções usam o perfil observado; um piloto de uma chamada não representa a diversidade do teste.')


if __name__ == '__main__':
    main()
