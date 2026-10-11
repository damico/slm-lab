import json
import math
import statistics
from pathlib import Path
from lab_runtime import read_jsonl, write_jsonl


def predictions(path, references):
    rows = read_jsonl(path)
    result = {r["id"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError("IDs de predicao duplicados.")
    extra = set(result) - {r["id"] for r in references}
    if extra:
        raise ValueError("IDs fora do conjunto de referencia: " + repr(sorted(extra)[:5]))
    return result


def wilson(k, n):
    if not n:
        return None
    z, p = 1.96, k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(center - half, 5), round(center + half, 5)]


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    index = (len(v) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return v[lo] + (v[hi] - v[lo]) * (index - lo)


def report(cases, predicted, output, extra=None):
    valid = [c for c in cases if c.get("reference_valid", True)]
    eligible = [c for c in valid if c.get("eligible", True)]
    lat = [p["latency_s"] for p in predicted.values() if p.get("status") == "ok"
           and "latency_s" in p]
    k = sum(c["correct"] for c in eligible)
    summary = {
        "n_total": len(cases), "n_reference_valid": len(valid),
        "n_eligible_valid": len(eligible), "n_correct": k,
        "accuracy_eligible": k / len(eligible) if eligible else None,
        "wilson95_eligible": wilson(k, len(eligible)),
        "accuracy_all_valid": sum(c["correct"] for c in valid) / len(valid) if valid else None,
        "latency_success_n": len(lat), "p50_s": statistics.median(lat) if lat else None,
        "p95_s": percentile(lat, 0.95),
        "api_prompt_tokens": sum(p.get("usage", {}).get("prompt_tokens", 0) for p in predicted.values()),
        "api_completion_tokens": sum(p.get("usage", {}).get("completion_tokens", 0) for p in predicted.values()),
        **(extra or {})}
    write_jsonl(output + ".cases.jsonl", cases)
    Path(output + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    import random
    ap = argparse.ArgumentParser(description="Diferenca pareada B - A com bootstrap.")
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a = {x["id"]: x for x in read_jsonl(args.a)}
    b = {x["id"]: x for x in read_jsonl(args.b)}
    if a.keys() != b.keys():
        raise ValueError("Compare os mesmos IDs, sem misturar limit e teste completo.")
    flags = lambda c: (c.get("eligible", True), c.get("reference_valid", True))
    if any(flags(a[k]) != flags(b[k]) for k in a):
        raise ValueError("Elegibilidade/referencias diferentes entre execucoes.")
    ids = [k for k in a if all(flags(a[k]))]
    if not ids:
        raise ValueError("Nenhum caso comparavel.")
    delta = [int(b[k]["correct"]) - int(a[k]["correct"]) for k in ids]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(2000))
    print(json.dumps({"n": len(delta), "delta_B_minus_A": sum(delta) / len(delta),
                      "bootstrap95": [percentile(boot, .025), percentile(boot, .975)],
                      "B_wins": delta.count(1), "A_wins": delta.count(-1)}, indent=2))
