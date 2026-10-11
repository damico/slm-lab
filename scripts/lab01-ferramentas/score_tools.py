import argparse
import json
import re
from collections import Counter
from lab_runtime import read_jsonl
from metrics import predictions, report


def no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Chave JSON duplicada.")
        out[key] = value
    return out


def parse_calls(text):
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    value = json.loads(text, object_pairs_hook=no_duplicate_keys,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, list):
        raise ValueError("Era esperado um array de chamadas.")
    for call in value:
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
            raise ValueError("Chamada fora do contrato.")
    return value


def canonical(calls):
    # Ordem das chamadas independentes nao conta; multiplicidade conta.
    return Counter(json.dumps(c, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False) for c in calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("prediction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    refs = read_jsonl(args.reference)
    pred = predictions(args.prediction, refs)
    cases = []
    for row in refs:
        p = pred.get(row["id"], {"status": "missing", "text": ""})
        c = {"id": row["id"], "eligible": row.get("eligible", True), "correct": False,
             "schema_valid": False, "tool_names_correct": False, "unknown_tool": False,
             "status": p["status"]}
        try:
            if p["status"] != "ok":
                raise ValueError("Predicao incompleta/ausente.")
            calls = parse_calls(p["text"])
            c["schema_valid"] = True
            c["unknown_tool"] = any(x["name"] not in row["tool_names"] for x in calls)
            c["tool_names_correct"] = Counter(x["name"] for x in calls) == Counter(x["name"] for x in row["gold"])
            c["correct"] = canonical(calls) == canonical(row["gold"])
        except (ValueError, TypeError, KeyError) as exc:
            c["parse_error"] = str(exc)
        cases.append(c)
    selected = [c for c in cases if c["eligible"]]
    rates = {key + "_rate": sum(c[key] for c in selected) / len(selected) if selected else None
             for key in ("schema_valid", "tool_names_correct", "unknown_tool")}
    report(cases, pred, args.out, rates)


if __name__ == "__main__":
    main()
