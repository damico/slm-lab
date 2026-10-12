"""Counts eligible BIRD API requests and summarizes OpenAI token usage."""
import argparse
import json
from pathlib import Path

INPUT_PRICE_PER_MILLION = 0.40
OUTPUT_PRICE_PER_MILLION = 1.60


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--predictions", default="results/test_llm.jsonl")
    args = parser.parse_args()
    rows = read_rows(args.test)
    eligible = sum(bool(row.get("eligible")) for row in rows)
    print(f"Teste: {eligible} chamadas previstas em {len(rows)} linhas totais.")
    path = Path(args.predictions)
    if not path.exists():
        return
    results = read_rows(path)
    usage_rows = [row.get("usage", {}) for row in results if row.get("usage")]
    input_tokens = sum(row.get("prompt_tokens", 0) for row in usage_rows)
    output_tokens = sum(row.get("completion_tokens", 0) for row in usage_rows)
    cost = (input_tokens * INPUT_PRICE_PER_MILLION +
            output_tokens * OUTPUT_PRICE_PER_MILLION) / 1_000_000
    print(f"Predições concluídas: {len(results)}; respostas com uso informado: {len(usage_rows)}.")
    print(f"Tokens: {input_tokens} entrada, {output_tokens} saída.")
    print(f"Custo OpenAI estimado em USD (gpt-4.1-mini): ${cost:.4f}.")


if __name__ == "__main__":
    main()
