"""SROIE: preserva teste original, executa OCR e cria JSONL para o SLM."""
import argparse
import hashlib
import json
from pathlib import Path
from lab_runtime import MODEL, digest, finalize_splits, lock_repo, read_jsonl, write_jsonl
from receipt_common import (FIELDS, extract_text, file_sha256, messages,
                            normalize_field, normalize_text, parse_fields, run_text)

DATASET = "jsdnrs/ICDAR2019-SROIE"


def main():
    from datasets import Image, load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--mode", choices=["ocr", "transcript"], default="ocr")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use uma pasta --out nova para preservar os splits.")
    out.mkdir(parents=True, exist_ok=True)
    revision = lock_repo(DATASET, "dataset")
    lock_repo(MODEL)
    ds = load_dataset(DATASET, revision=revision)
    if not {"train", "test"}.issubset(ds):
        raise ValueError("Distribuicao mudou: esperados train e test.")
    inventory = {s: len(ds[s]) for s in ("train", "test")}
    print("Splits baixados:", inventory, flush=True)
    image_dir, cache = Path("raw/images"), Path("raw/ocr_cache")
    image_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    tess = run_text(["tesseract", "--version"]).splitlines()[0] if args.mode == "ocr" else None
    preprocessing = {"mode": args.mode, "lang": args.lang, "psm": args.psm,
                     "tesseract": tess, "code_sha256": file_sha256("receipt_common.py")}
    rows = {"train": [], "test": []}
    seen_ids = set()
    for split in rows:
        for raw in ds[split].cast_column("image", Image(decode=False)):
            key = str(raw["key"])
            if key in seen_ids:
                raise ValueError("Chave repetida no dataset: " + key)
            seen_ids.add(key)
            uid = split + "_" + digest(key)[:16]
            image = raw["image"]
            content = image["bytes"] if image.get("bytes") is not None else Path(image["path"]).read_bytes()
            image_hash = hashlib.sha256(content).hexdigest()
            path = image_dir / (uid + ".img")  # Pillow identifica o formato pelos bytes.
            path.write_bytes(content)
            transcript = normalize_text("\n".join(raw["words"]))
            if args.mode == "ocr":
                cache_id = digest(image_hash + json.dumps(preprocessing, sort_keys=True))
                cached = cache / (cache_id + ".txt")
                if cached.exists():
                    text = cached.read_text(encoding="utf-8")
                else:
                    text, _ = extract_text(path, args.lang, args.psm)
                    cached.write_text(text, encoding="utf-8")
            else:
                text = transcript
            entities = {k: raw["entities"][k] for k in FIELDS}
            if any(not isinstance(v, str) or not v.strip() for v in entities.values()):
                raise ValueError("Referencia ausente: " + key)
            answer = json.dumps(entities, ensure_ascii=False)
            parse_fields(answer)
            rows[split].append({"id": uid, "source_key": key,
                "messages": messages(text), "answer": answer, "text": text,
                "image_path": str(path), "image_sha256": image_hash,
                "transcript_sha256": digest(transcript), "ocr_empty": not bool(text.strip())})
            if len(rows[split]) % 50 == 0:
                print(split, len(rows[split]), flush=True)
    # Teste inteiro preservado. Duplicatas textuais/imagens do teste saem do treino.
    test_hashes = {r[k] for r in rows["test"] for k in ("image_sha256", "transcript_sha256")}
    unique, removed, fingerprints = [], [], set(test_hashes)
    for row in sorted(rows["train"], key=lambda r: r["id"]):
        marks = {row["image_sha256"], row["transcript_sha256"]}
        if marks & fingerprints:
            removed.append(row["id"])
        else:
            unique.append(row)
            fingerprints.update(marks)
    unique.sort(key=lambda r: digest("42:" + r["id"]))
    n_val = max(1, round(len(unique) * .1))
    splits = {"train": unique[n_val:], "validation": unique[:n_val], "test": rows["test"]}
    finalize_splits(splits, out, args.max_seq, args.max_new)
    # Usa o treino efetivamente aceito pelo limite de contexto.
    companies = {normalize_field(json.loads(r["answer"])["company"])
                 for r in read_jsonl(out / "train.jsonl")}
    for name in splits:
        accepted = read_jsonl(out / (name + ".jsonl"))
        for row in accepted:
            row["company_seen_in_train"] = normalize_field(json.loads(row["answer"])["company"]) in companies
        write_jsonl(out / (name + ".jsonl"), accepted)
    for row in splits["test"]:
        row["company_seen_in_train"] = normalize_field(json.loads(row["answer"])["company"]) in companies
    # Modelo nao recebe gabarito, source_key nem company_seen_in_train: so messages.
    manifest = {"dataset": DATASET, "revision": revision, "original_counts": inventory,
                "preprocessing": preprocessing, "split_seed": 42,
                "removed_train_duplicates": removed,
                "ocr_empty": {s: sum(r["ocr_empty"] for r in v) for s, v in splits.items()},
                "test_known_merchant": sum(r["company_seen_in_train"] for r in splits["test"])}
    (out / "dataset_report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    demo = out / "demo_images"
    demo.mkdir(exist_ok=True)
    # Ordem por ID; nao seleciona exemplos com base no acerto do modelo.
    for row in sorted(splits["test"], key=lambda r: r["id"])[:10]:
        from PIL import Image as PILImage
        with PILImage.open(row["image_path"]) as im:
            im.convert("RGB").save(demo / (row["id"] + ".png"))
    write_jsonl(out / "demo_manifest.jsonl", sorted(splits["test"], key=lambda r: r["id"])[:10])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
