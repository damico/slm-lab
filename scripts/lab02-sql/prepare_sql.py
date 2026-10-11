import argparse
import json
import sqlite3
from pathlib import Path
from lab_runtime import MODEL, digest, lock_repo, finalize_splits

SYSTEM = (
    "Generate one read-only SQLite SELECT query that answers the question using the "
    "provided schema and evidence. Return only SQL, without explanations or Markdown. "
    "Use SQLite syntax. Do not modify data."
)


def index_databases(root):
    out = {}
    for p in sorted(Path(root).rglob("*.sqlite")):
        if p.stem in out:
            raise ValueError("Banco duplicado: " + p.stem + ". Use uma unica copia do pacote.")
        out[p.stem] = p.resolve()
    if not out:
        raise ValueError("Nenhum .sqlite em " + str(root) + "; confira ZIPs internos.")
    return out


def compact_schema(path):
    # Extrai somente metadados, nunca utiliza o SQL correto para selecionar tabelas.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        lines = []
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            cols = list(con.execute("PRAGMA table_info(" + quoted + ")"))
            body = ", ".join('"' + c[1].replace('"', '""') + '" ' + (c[2] or "ANY")
                             + (" PK" + str(c[5]) if c[5] else "") for c in cols)
            lines.append(quoted + "(" + body + ")")
            for fk in con.execute("PRAGMA foreign_key_list(" + quoted + ")"):
                lines.append("  FK group=" + str(fk[0]) + " part=" + str(fk[1]) + ": "
                             + str(fk[3]) + " -> " + str(fk[2]) + "." + str(fk[4]))
    return "\n".join(lines)


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    lock_repo(MODEL)
    train_repo, test_repo = "birdsql/bird23-train-filtered", "birdsql/bird_mini_dev"
    tr_rev, te_rev = lock_repo(train_repo, "dataset"), lock_repo(test_repo, "dataset")
    source = load_dataset(train_repo, revision=tr_rev, split="train")
    test = load_dataset(test_repo, revision=te_rev, split="mini_dev_sqlite")
    train_ids, test_ids = set(source["db_id"]), set(test["db_id"])
    if train_ids & test_ids:
        raise ValueError("Sobreposicao inesperada entre bancos de treino e mini-dev.")
    ids = sorted(train_ids, key=lambda x: digest("42:" + x))
    validation_ids = set(ids[:max(1, round(len(ids) * 0.1))])
    indexes = {"train": index_databases(args.train_root), "test": index_databases(args.test_root)}
    missing = (train_ids - indexes["train"].keys()) | (test_ids - indexes["test"].keys())
    if missing:
        raise ValueError("Faltam bancos SQLite: " + repr(sorted(missing)))
    schemas, splits = {}, {"train": [], "validation": [], "test": []}
    manifest, seen = {}, set()
    for origin, dataset in (("train", source), ("test", test)):
        for i, row in enumerate(dataset):
            db = row["db_id"]
            path = indexes[origin][db]
            if db not in schemas:
                schemas[db] = compact_schema(path)
                manifest[db] = {"path": str(path), "size_bytes": path.stat().st_size}
            unique = digest(db + "\n" + row["question"].strip())
            if unique in seen:
                continue
            seen.add(unique)
            split = "test" if origin == "test" else "validation" if db in validation_ids else "train"
            rid = origin + ":" + str(row.get("question_id", i))
            user = ("Schema (PK numbers identify composite-key order):\n" + schemas[db]
                    + "\nEvidence:\n" + (row.get("evidence") or "")
                    + "\nQuestion:\n" + row["question"])
            splits[split].append({"id": rid, "db_id": db, "db_path": str(path),
                "difficulty": row.get("difficulty", "unspecified"),
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "answer": row["SQL"].strip(), "gold": row["SQL"].strip()})
    for split in splits:
        splits[split].sort(key=lambda r: digest("42:" + r["id"]))
    finalize_splits(splits, args.out, args.max_seq, args.max_new)
    Path(args.out, "db_manifest.json").write_text(json.dumps(manifest, indent=2))
    Path(args.out, "source_report.json").write_text(json.dumps({
        "train_repo": train_repo, "train_revision": tr_rev,
        "test_repo": test_repo, "test_revision": te_rev,
        "validation_db_ids": sorted(validation_ids), "test_db_ids": sorted(test_ids),
        "source_rows": len(source), "mini_dev_rows": len(test),
        "pre_length_counts": {k: len(v) for k, v in splits.items()},
        "arguments": vars(args)}, indent=2))


if __name__ == "__main__":
    main()
