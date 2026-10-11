import argparse
import collections
import gzip
import json
import urllib.request
from pathlib import Path
from decision_core import (MODEL, ABCD_REVISION, ContextBuilder, conversation_cases,
    digest, file_hash, policy_documents, read_json, save_json, tokenizer, write_rows)


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("data/abcd_v1.1.json.gz", "data/ontology.json", "data/guidelines.json", "LICENSE"):
        target = folder / Path(name).name
        if not target.exists():
            url = f"https://raw.githubusercontent.com/asappresearch/abcd/{ABCD_REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "ABCD-training-lab"})
            with urllib.request.urlopen(request, timeout=180) as response:
                content = response.read()
            target.write_bytes(content)
        print("Fonte:", target, flush=True)


def main():
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/abcd")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-conversations", type=int, default=2000)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--policy-tokens", type=int, default=400)
    args = ap.parse_args()
    out, raw = Path(args.out), Path(args.raw)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use --out novo; nao misture preparacoes.")
    out.mkdir(parents=True, exist_ok=True)
    download(raw)
    with gzip.open(raw / "abcd_v1.1.json.gz", "rt", encoding="utf-8") as f:
        original = json.load(f)
    ids = [str(c["convo_id"]) for s in ("train", "dev", "test") for c in original[s]]
    if len(set(ids)) != len(ids):
        raise ValueError("IDs de conversa repetidos entre splits.")
    # Remove do treino conversas inteiras identicas a dev/test ou a outro treino.
    fingerprint = lambda c: digest(json.dumps(c["original"], ensure_ascii=False, sort_keys=True))
    seen = {fingerprint(c) for s in ("dev", "test") for c in original[s]}
    train, removed = [], []
    for c in sorted(original["train"], key=lambda c: digest("42:" + str(c["convo_id"]))):
        h = fingerprint(c)
        if h in seen:
            removed.append(c["convo_id"])
        else:
            train.append(c)
            seen.add(h)
    if args.train_conversations:
        train = train[:args.train_conversations]
    dev = sorted(original["dev"], key=lambda c: digest("42:" + str(c["convo_id"])))
    a, b = len(dev) // 2, len(dev) * 3 // 4
    splits = {"train": train, "validation": dev[:a], "calibration": dev[a:b],
              "selection": dev[b:], "test": original["test"]}
    ontology = read_json(raw / "ontology.json")
    actions = sorted({name for group in ontology["actions"].values() for name in group})
    revision_file = Path("model_revision.json")
    if not revision_file.exists():
        save_json(revision_file, {"model": MODEL, "revision": HfApi().model_info(MODEL).sha})
    rev = read_json(revision_file)
    if rev["model"] != MODEL:
        raise ValueError("model_revision.json pertence a outro modelo.")
    bundle = {"model": MODEL, "model_revision": rev["revision"], "abcd_revision": ABCD_REVISION,
        "labels": actions + ["retrieve_utterance"], "actions": actions,
        "max_seq": args.max_seq, "policy_tokens": args.policy_tokens,
        "policy_docs": policy_documents(read_json(raw / "guidelines.json"))}
    save_json(out / "bundle.json", bundle)
    builder = ContextBuilder(bundle, tokenizer(bundle))
    counts, demos = {}, []
    for name, conversations in splits.items():
        rows, labels, eligible_labels = [], collections.Counter(), collections.Counter()
        for c in conversations:
            for sample in conversation_cases(c):
                if sample["label"] not in bundle["labels"]:
                    raise ValueError("Rotulo fora da ontologia: " + sample["label"])
                features = builder.build(sample["history"])
                features.pop("input_ids")
                row = {k: v for k, v in sample.items() if k != "history"}
                row.update(features, split=name, label_id=bundle["labels"].index(sample["label"]))
                rows.append(row)
                labels[sample["label"]] += 1
                if row["eligible"]:
                    eligible_labels[sample["label"]] += 1
                if name == "test" and len(demos) < 12 and row["eligible"] and row["label"] != "retrieve_utterance":
                    demos.append({"id": sample["id"], "history": sample["history"], "expected": sample["label"]})
        write_rows(out / (name + ".jsonl"), rows)
        counts[name] = {"conversations": len(conversations), "decisions": len(rows),
            "eligible": sum(r["eligible"] for r in rows),
            "with_history_window": sum(r["dropped_turns"] > 0 for r in rows),
            "class_counts": dict(labels), "eligible_class_counts": dict(eligible_labels)}
        print(name, counts[name]["decisions"], "decisoes", flush=True)
    missing = set(bundle["labels"]) - set(counts["train"]["eligible_class_counts"])
    if missing:
        raise ValueError("Aumente --train-conversations; classes ausentes: " + repr(missing))
    report = {"arguments": vars(args), "splits": counts, "removed_train_duplicates": removed,
              "source_hashes": {p.name: file_hash(p) for p in raw.iterdir() if p.is_file()},
              "bundle_sha256": file_hash(out / "bundle.json"),
              "source_conversations": {s: len(v) for s, v in original.items()}}
    save_json(out / "preparation_report.json", report)
    (out / "demo").mkdir(exist_ok=True)
    for i, d in enumerate(demos):
        save_json(out / "demo" / f"case_{i:02d}.json", {"history": d["history"]})
    save_json(out / "demo_expected.json", [{"id": d["id"], "expected": d["expected"]} for d in demos])


if __name__ == "__main__":
    main()
