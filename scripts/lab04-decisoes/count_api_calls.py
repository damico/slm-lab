"""Conta as linhas que gerariam chamadas ao endpoint remoto do laboratório 04."""
import argparse
import json
from pathlib import Path


def count(path, limit=0):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return len(rows), sum(bool(row.get("eligible")) for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    args = parser.parse_args()
    probe_rows, probe_calls = count(args.validation, limit=3)
    test_rows, test_calls = count(args.test)
    print(f"Probe: {probe_calls} chamadas ({probe_rows} primeiras linhas da validação).")
    print(f"Teste completo: {test_calls} chamadas ({test_rows} linhas do teste).")
    print(f"Total estimado: {probe_calls + test_calls} chamadas, sem contar retries HTTP.")
