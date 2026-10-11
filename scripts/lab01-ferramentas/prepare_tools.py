import argparse
import json
import re
from collections import Counter
from pathlib import Path
from lab_runtime import MODEL, digest, lock_repo, finalize_splits

SYSTEM = (
    "Select the required calls using only the supplied tools. Return ONLY a JSON array "
    'of objects with exactly these keys: "name" and "arguments". '
    "Use exact tool and parameter names. Include all requested independent calls. "
    "Do not include explanations or Markdown."
)


def unpack(value):
    return json.loads(value) if isinstance(value, str) else value


def group_id(tools):
    # Agrupa o catalogo inteiro; nao significa que cada API seja inedita no teste.
    canonical = sorted(tools, key=lambda t: t["name"])
    return digest(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--train-limit", type=int, default=0)
    args = ap.parse_args()
    lock_repo(MODEL)
    repo = "Salesforce/xlam-function-calling-60k"
    revision = lock_repo(repo, "dataset")
    dataset = load_dataset(repo, revision=revision, split="train")
    splits = {"train": [], "validation": [], "test": []}
    counts, seen = Counter(), set()
    for i, row in enumerate(dataset):
        query = row["query"].strip()
        key = digest(re.sub(r"\s+", " ", query).casefold())
        if key in seen:
            counts["duplicate_query"] += 1
            continue
        seen.add(key)
        tools, answers = unpack(row["tools"]), unpack(row["answers"])
        if not isinstance(tools, list) or not isinstance(answers, list):
            raise ValueError("Formato mudou: inspecione os campos tools/answers.")
        names = {t["name"] for t in tools}
        if any(a["name"] not in names or not isinstance(a["arguments"], dict) for a in answers):
            counts["invalid_reference"] += 1
            continue
        group = group_id(tools)
        bucket = int(group[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        normalized = [{"name": a["name"], "arguments": a["arguments"]} for a in answers]
        record = {
            "id": "xlam:" + str(row.get("id", i)), "group": group,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Tools:\n" +
                          json.dumps(tools, ensure_ascii=False, separators=(",", ":")) +
                          "\nRequest:\n" + query}],
            "answer": json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
            "gold": normalized, "tool_names": sorted(names)}
        splits[split].append(record)
    for name in splits:
        splits[name].sort(key=lambda r: digest("42:" + r["id"]))
    if args.train_limit:
        splits["train"] = splits["train"][:args.train_limit]
    groups = {k: {r["group"] for r in v} for k, v in splits.items()}
    assert not (groups["train"] & groups["test"])
    assert not (groups["train"] & groups["validation"])
    finalize_splits(splits, args.out, args.max_seq, args.max_new)
    Path(args.out, "source_report.json").write_text(json.dumps({
        "dataset": repo, "revision": revision, "raw_count": len(dataset),
        "pre_length_counts": {k: len(v) for k, v in splits.items()},
        "group_counts": {k: len(v) for k, v in groups.items()},
        "dropped": dict(counts), "arguments": vars(args)}, indent=2))


if __name__ == "__main__":
    main()
