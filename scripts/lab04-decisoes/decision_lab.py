"""Treino QLoRA com cabeca de classificacao; baselines e inferencia."""
import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from decision_core import (adapter_hash, digest, file_hash, meta_path, read_json,
                          read_rows, save_json, tokenizer, write_rows)


def gpu_model(bundle, classifier=True):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel.")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    cls = AutoModelForSequenceClassification if classifier else AutoModelForCausalLM
    extra = {"num_labels": len(bundle["labels"]), "problem_type": "single_label_classification"} if classifier else {}
    model = cls.from_pretrained(bundle["model"], revision=bundle["model_revision"],
        torch_dtype=dtype, device_map={"": rank}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["score"] if classifier else None), **extra)
    if classifier and isinstance(model.score, bnb.nn.Linear4bit):
        raise RuntimeError("A cabeca score deve permanecer em ponto flutuante.")
    model.config.pad_token_id = tokenizer(bundle).pad_token_id
    model.config.use_cache = not classifier
    return model, bf16


def initialize_training_devices():
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=5))
    print(f"DDP: rank local {rank}, GPU {rank}, grupo inicializado.", flush=True)


def train(args):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    initialize_training_devices()
    set_seed(args.seed)
    bundle = read_json(Path(args.data, "bundle.json"))
    tok = tokenizer(bundle)
    parts = {}
    for split in ("train", "validation"):
        rows = [r for r in read_rows(Path(args.data, split + ".jsonl")) if r["eligible"]]
        if split == "validation" and args.eval_limit:
            rows = sorted(rows, key=lambda r: digest(r["id"]))[:args.eval_limit]
        encoded = []
        for row in rows:
            ids = tok.encode(row["text"], add_special_tokens=False)
            if len(ids) > bundle["max_seq"]:
                raise ValueError("Comprimento mudou; refaca a preparacao.")
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": row["label_id"]})
        if not encoded:
            raise ValueError("Split vazio: " + split)
        parts[split] = Dataset.from_list(encoded)
    model, bf16 = gpu_model(bundle)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                   gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=args.rank,
        lora_alpha=2 * args.rank, lora_dropout=.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"]))
    model.print_trainable_parameters()
    if not any("score" in n and p.requires_grad for n, p in model.named_parameters()):
        raise RuntimeError("Cabeca de classificacao nao foi habilitada para treino.")
    options = TrainingArguments(output_dir=args.run, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr,
        warmup_ratio=.05, lr_scheduler_type="cosine", weight_decay=.01, max_grad_norm=1.0,
        optim="adamw_torch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, prediction_loss_only=True, logging_steps=10, logging_nan_inf_filter=False,
        report_to="none", seed=args.seed, data_seed=args.seed, label_names=["labels"])
    trainer = Trainer(model=model, args=options, processing_class=tok,
        train_dataset=parts["train"], eval_dataset=parts["validation"],
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=8))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    print(f"DDP: rank local {os.environ.get('LOCAL_RANK', '0')}, Trainer concluido.", flush=True)
    if trainer.is_world_process_zero():
        folder = Path(args.run, "final_adapter")
        trainer.save_model(str(folder))
        tok.save_pretrained(folder)
        save_json(folder / "bundle.json", bundle)
        save_json(Path(args.run, "run_report.json"), {"arguments": vars(args), "metrics": result.metrics,
            "seconds": time.perf_counter() - start,
            "rank0_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "bundle_sha256": file_hash(Path(args.data, "bundle.json")),
            "train_sha256": file_hash(Path(args.data, "train.jsonl")),
            "validation_sha256": file_hash(Path(args.data, "validation.jsonl"))})
        print("Adaptador final e run_report.json salvos.", flush=True)


def baseline(args):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(Path(args.data, "train.jsonl")) if r["eligible"]]
    model = make_pipeline(TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"))
    model.fit([r["text"] for r in rows], [r["label_id"] for r in rows])
    joblib.dump(model, out)
    save_json(meta_path(out), {"bundle_sha256": file_hash(Path(args.data, "bundle.json")),
                             "train_sha256": file_hash(Path(args.data, "train.jsonl"))})


def remote_call(text):
    import urllib.request
    import urllib.error
    extras = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))
    if {"model", "messages"} & set(extras):
        raise ValueError("LLM_EXTRA_JSON nao pode substituir model/messages.")
    body = {"model": os.environ["LLM_MODEL"], "messages": [{"role": "user", "content": text}], **extras}
    if not any(k in body for k in ("max_tokens", "max_completion_tokens")):
        body["max_tokens"] = 32
    request = urllib.request.Request(os.environ["LLM_URL"], json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.load(response)
            choice = raw["choices"][0]
            return (choice["message"].get("content") or "").strip(), raw.get("usage", {}), choice.get("finish_reason")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError("Falha HTTP: " + str(exc.code)) from exc
            time.sleep(2 ** attempt)


class Predictor:
    def __init__(self, bundle, backend, artifact=None):
        self.bundle, self.backend = bundle, backend
        if backend == "api":
            self.model_key = "api:" + os.environ["LLM_MODEL"]
        elif backend == "tfidf":
            import joblib
            self.model = joblib.load(artifact)  # Somente o arquivo gerado localmente neste lab.
            self.model_key = "tfidf:" + file_hash(artifact)
        else:
            from transformers import set_seed
            from peft import PeftModel
            set_seed(42)
            self.tok = tokenizer(bundle)
            self.model, _ = gpu_model(bundle, classifier=backend == "classifier")
            if backend == "classifier":
                if read_json(Path(artifact, "bundle.json")) != bundle:
                    raise ValueError("Adaptador e bundle pertencem a preparacoes diferentes.")
                self.model = PeftModel.from_pretrained(self.model, artifact, is_trainable=False)
                self.model_key = "adapter:" + adapter_hash(artifact)
            else:
                self.model_key = "base:" + bundle["model_revision"]
            self.model.eval()

    def __call__(self, text):
        labels = self.bundle["labels"]
        start = time.perf_counter()
        result = {"status": "ok"}
        if self.backend == "api":
            raw, usage, finish = remote_call(text)
            result.update(raw=raw, usage=usage, choice=raw if raw in labels else None)
            if raw not in labels or finish == "length":
                result["status"] = "invalid_output"
        elif self.backend == "tfidf":
            values = self.model.decision_function([text])[0]
            logits = [-1e9] * len(labels)
            for k, v in zip(self.model.classes_, values):
                logits[int(k)] = float(v)
            result.update(logits=logits, choice=labels[max(range(len(labels)), key=logits.__getitem__)])
        else:
            import torch
            if self.backend == "classifier":
                encoded = self.tok(text, add_special_tokens=False, return_tensors="pt")
            else:
                ids = self.tok.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                encoded = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.inference_mode():
                if self.backend == "classifier":
                    values = self.model(**encoded).logits[0].float().cpu().tolist()
                    result.update(logits=values, choice=labels[max(range(len(labels)), key=values.__getitem__)])
                else:
                    y = self.model.generate(**encoded, max_new_tokens=32, do_sample=False,
                        pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
                    raw = self.tok.decode(y[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    result.update(raw=raw, choice=raw if raw in labels else None)
                    if raw not in labels:
                        result["status"] = "invalid_output"
            torch.cuda.synchronize()
        result["latency_s"] = time.perf_counter() - start
        return result


def predict(args):
    bundle_path = Path(args.data, "bundle.json")
    bundle = read_json(bundle_path)
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.backend in ("classifier", "tfidf") and not args.artifact:
        raise ValueError("Passe --artifact para classifier/tfidf.")
    if args.backend == "tfidf" and read_json(meta_path(args.artifact))["bundle_sha256"] != file_hash(bundle_path):
        raise ValueError("Baseline e bundle diferentes.")
    model = Predictor(bundle, args.backend, args.artifact)
    first = next((r for r in rows if r["eligible"]), None)
    if first and args.backend != "api":
        model(first["text"])  # Aquecimento fora da medida.
    if args.backend in ("classifier", "base"):
        import torch
        torch.cuda.reset_peak_memory_stats()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            result = {"id": row["id"], "status": "context_exceeded"}
            if row["eligible"]:
                try:
                    result.update(model(row["text"]))
                except Exception as exc:
                    if args.backend != "api":
                        raise
                    result.update(status="api_error", error=type(exc).__name__)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(row["id"], result["status"], flush=True)
    meta = {"model_key": model.model_key, "arguments": vars(args), "labels": bundle["labels"],
            "bundle_sha256": file_hash(bundle_path), "input_sha256": file_hash(args.input)}
    if args.backend == "api":
        meta["api_extra"] = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))
    if args.backend in ("classifier", "base"):
        meta.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    save_json(meta_path(out), meta)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--eval-steps", type=int, default=100)
    t.add_argument("--eval-limit", type=int, default=512)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume")
    b = sub.add_parser("baseline")
    b.add_argument("--data", default="data")
    b.add_argument("--output", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", default="data")
    p.add_argument("--backend", choices=["classifier", "base", "tfidf", "api"], required=True)
    p.add_argument("--artifact")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "baseline": baseline, "predict": predict}[a.command](a)


if __name__ == "__main__":
    main()
