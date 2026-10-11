"""Exact match normalizado por campo e documento; sem julgar com outro LLM."""
import argparse
import json
from lab_runtime import read_jsonl
from metrics import predictions, report
from receipt_common import FIELDS, normalize_field, parse_fields


def score(refs, preds):
    cases = []
    for row in refs:
        expected = parse_fields(row["answer"])
        pred = preds.get(row["id"], {})
        actual, schema_ok = {}, False
        if pred.get("status") == "ok":
            try:
                actual, schema_ok = parse_fields(pred.get("text", "")), True
            except (ValueError, TypeError):
                pass
        field_ok = {k: bool(row["eligible"] and schema_ok and
                       normalize_field(actual[k]) == normalize_field(expected[k])) for k in FIELDS}
        cases.append({"id": row["id"], "eligible": row["eligible"],
                      "correct": all(field_ok.values()), "fields": field_ok,
                      "schema_valid": schema_ok, "expected": expected, "predicted": actual,
                      "company_seen_in_train": row.get("company_seen_in_train", False),
                      "status": pred.get("status", "missing_prediction")})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("predictions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    refs = read_jsonl(args.reference)
    predicted = predictions(args.predictions, refs)
    cases = score(refs, predicted)
    eligible = [r for r in cases if r["eligible"]]
    n = len(eligible)
    by_merchant = {}
    for flag in (True, False):
        subset = [r for r in eligible if r["company_seen_in_train"] == flag]
        by_merchant["seen" if flag else "unseen"] = {
            "n": len(subset), "document_accuracy": sum(r["correct"] for r in subset) / len(subset) if subset else None}
    report(cases, predicted, args.out, {
        "metric": "all_four_fields_normalized_exact_match",
        "schema_valid_rate": sum(r["schema_valid"] for r in eligible) / n if n else None,
        "field_accuracy": {k: sum(r["fields"][k] for r in eligible) / n if n else None for k in FIELDS},
        "merchant_groups": by_merchant})


if __name__ == "__main__":
    main()
