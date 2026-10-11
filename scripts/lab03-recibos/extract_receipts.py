"""Aplicacao local: um arquivo ou pasta -> resultados.jsonl + despesas.csv."""
import argparse
import csv
import json
import time
from pathlib import Path
from lab_runtime import load_model, prompt_ids, tokenizer
from receipt_common import FIELDS, extract_text, file_sha256, messages, parse_fields, safe_csv


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Arquivo ou pasta; um recibo por arquivo.")
    ap.add_argument("--adapter", help="Omitir para usar o modelo original.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    origin, out = Path(args.input).resolve(), Path(args.out)
    if not origin.exists():
        raise FileNotFoundError(origin)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf", ".txt"}
    files = sorted(p for p in origin.iterdir() if p.suffix.lower() in extensions and p.is_file()) if origin.is_dir() else [origin]
    if not files:
        raise ValueError("Nenhum recibo encontrado.")
    out.mkdir(parents=True, exist_ok=False)
    (out / "textos").mkdir()
    tok, loaded = tokenizer(), None
    meta = {"arguments": vars(args), "revisions": json.loads(Path("revisions.json").read_text()),
            "timing": "OCR + tokenizacao + inferencia + validacao; exclui carga do modelo"}
    with (out / "resultados.jsonl").open("w", encoding="utf-8") as jf, \
         (out / "despesas.csv").open("w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["arquivo", "status", *FIELDS, "segundos"])
        writer.writeheader()
        for index, path in enumerate(files):
            record = {"arquivo": path.name, "sha256": file_sha256(path), "status": "error"}
            start = time.perf_counter()
            loading_seconds = 0
            try:
                text, method = extract_text(path, args.lang, args.psm, args.force_ocr)
                (out / "textos" / f"{index:04d}.txt").write_text(text, encoding="utf-8")
                record.update(text_file=f"textos/{index:04d}.txt", extraction_method=method)
                if not text:
                    raise ValueError("OCR/texto vazio.")
                ids = prompt_ids(tok, messages(text))
                if len(ids) + args.max_new > args.max_seq:
                    record["status"] = "context_exceeded"
                    raise ValueError("Recibo excede contexto; nao foi truncado.")
                if loaded is None:
                    load_start = time.perf_counter()
                    loaded, _ = load_model(args.adapter)
                    loaded.eval()
                    torch.cuda.synchronize()
                    loading_seconds = time.perf_counter() - load_start
                x = torch.tensor([ids], device=loaded.device)
                with torch.inference_mode():
                    y = loaded.generate(x, attention_mask=torch.ones_like(x),
                        max_new_tokens=args.max_new, do_sample=False,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                answer = y[0, x.shape[1]:].tolist()
                raw = tok.decode(answer, skip_special_tokens=True).strip()
                record["raw_output"] = raw
                if len(answer) >= args.max_new and answer[-1] != tok.eos_token_id:
                    record["status"] = "output_limit"
                    raise ValueError("Geracao atingiu limite de saida.")
                record["fields"] = parse_fields(raw)
                record["status"] = "schema_valid"  # Nao significa campos corretos.
            except Exception as exc:
                record["error"] = type(exc).__name__ + ": " + str(exc)
            record["segundos"] = round(time.perf_counter() - start - loading_seconds, 4)
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()
            writer.writerow({k: safe_csv(v) for k, v in {
                "arquivo": path.name, "status": record["status"],
                **{k: record.get("fields", {}).get(k) for k in FIELDS},
                "segundos": record["segundos"]}.items()})
            cf.flush()
            print(json.dumps(record, ensure_ascii=False, indent=2))
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
