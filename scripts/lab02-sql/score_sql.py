import argparse
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from lab_runtime import read_jsonl
from metrics import predictions, report


def clean_sql(text):
    text = text.strip()
    fence = re.fullmatch(r"```(?:sql|sqlite)?\s*\n(.*?)\n```", text, re.S | re.I)
    return fence.group(1).strip() if fence else text


def authorize(action, arg1, arg2, database, trigger):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
    if action == sqlite3.SQLITE_FUNCTION:
        function = (arg2 or arg1 or "").lower()
        if function in {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def execute_readonly(path, sql, timeout=15, max_rows=100000):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA busy_timeout=1000")
        con.enable_load_extension(False)
        con.set_authorizer(authorize)
        deadline = time.monotonic() + timeout
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cur = con.execute(clean_sql(sql))  # execute rejeita multiplas instrucoes.
        if cur.description is None:
            raise ValueError("A instrucao nao retornou uma tabela.")
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("Limite de linhas atingido.")
        return rows
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("prediction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=15)
    args = ap.parse_args()
    refs = read_jsonl(args.reference)
    pred = predictions(args.prediction, refs)
    cases = []
    for row in refs:
        p = pred.get(row["id"], {"status": "missing", "text": ""})
        c = {"id": row["id"], "db_id": row["db_id"], "difficulty": row["difficulty"],
             "eligible": row.get("eligible", True), "reference_valid": True,
             "correct": False, "bag_correct": False, "executable": False,
             "status": p["status"]}
        try:
            gold = execute_readonly(row["db_path"], row["gold"], args.timeout)
            c["gold_empty"] = len(gold) == 0
        except (sqlite3.Error, ValueError) as exc:
            c.update(reference_valid=False, reference_error=str(exc))
            cases.append(c)
            continue
        if p["status"] == "ok":
            try:
                got = execute_readonly(row["db_path"], p["text"], args.timeout)
                c["executable"] = True
                c["correct"] = set(got) == set(gold)
                c["bag_correct"] = Counter(got) == Counter(gold)
            except (sqlite3.Error, ValueError) as exc:
                c["sql_error"] = str(exc)
        cases.append(c)
    eligible = [c for c in cases if c["eligible"] and c["reference_valid"]]
    by_difficulty = {}
    for level in sorted({c["difficulty"] for c in eligible}):
        subset = [c for c in eligible if c["difficulty"] == level]
        by_difficulty[level] = {"n": len(subset), "ex_set": sum(c["correct"] for c in subset) / len(subset)}
    report(cases, pred, args.out, {
        "metric": "local_EX_set_not_official_BIRD_score",
        "execution_rate": sum(c["executable"] for c in eligible) / len(eligible) if eligible else None,
        "ex_bag": sum(c["bag_correct"] for c in eligible) / len(eligible) if eligible else None,
        "gold_empty_n": sum(c.get("gold_empty", False) for c in eligible),
        "by_difficulty": by_difficulty})


if __name__ == "__main__":
    main()
