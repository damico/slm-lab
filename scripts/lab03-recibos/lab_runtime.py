"""Runtime do laboratorio de extracao de recibos. Python 3.11/3.12."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-1.7B"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_repo(repo, kind="model"):
    # Preparacao em um processo; treinos apenas leem revisions.json.
    from huggingface_hub import HfApi
    path = Path("revisions.json")
    lock = json.loads(path.read_text()) if path.exists() else {}
    key = kind + ":" + repo
    if key not in lock:
        api = HfApi()
        info = api.model_info(repo) if kind == "model" else api.dataset_info(repo)
        lock[key] = info.sha
        path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock[key]


def tokenizer():
    from transformers import AutoTokenizer
    lock = json.loads(Path("revisions.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=lock["model:" + MODEL])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def prompt_ids(tok, messages):
    return tok.apply_chat_template(messages, tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)


def finalize_splits(splits, out, max_seq=2048, max_new=512):
    # Teste: elegibilidade depende SO do prompt, nunca da resposta correta.
    tok = tokenizer()
    stats = {"max_seq": max_seq, "max_new": max_new, "splits": {}}
    Path(out).mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        accepted, rejected = [], []
        for r in rows:
            n_prompt = len(prompt_ids(tok, r["messages"]))
            r["prompt_tokens"] = n_prompt
            r["eligible"] = n_prompt + max_new <= max_seq
            if name in ("train", "validation"):
                n_answer = len(tok.encode(r["answer"], add_special_tokens=False)) + 1
                if not r["eligible"] or n_prompt + n_answer > max_seq:
                    rejected.append({"id": r["id"], "reason": "length", "prompt": n_prompt,
                                     "answer": n_answer})
                    continue
            accepted.append(r)
        write_jsonl(Path(out) / (name + ".jsonl"), accepted)
        write_jsonl(Path(out) / (name + "_excluded.jsonl"), rejected)
        stats["splits"][name] = {"written": len(accepted), "excluded": len(rejected),
                                 "eligible": sum(r["eligible"] for r in accepted)}
    for name in ("train", "validation"):
        if not stats["splits"][name]["written"]:
            raise ValueError("Split vazio depois do limite de contexto: " + name)
    Path(out, "length_report.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def load_model(adapter=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("Este caminho requer CUDA. Verifique driver e torch.")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    lock = json.loads(Path("revisions.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=lock["model:" + MODEL], torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        device_map={"": rank}, attn_implementation="sdpa")
    model.config.use_cache = not training
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, bf16


def train(args):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    class ReceiptTrainer(Trainer):
        def _restore_adapter(self,checkpoint,model=None):
            net=self.model if model is None else model
            net.load_adapter(checkpoint,net.active_adapter,torch_device='cpu',is_trainable=True)
        def _load_best_model(self):
            self._restore_adapter(self.state.best_model_checkpoint)
        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            self._restore_adapter(resume_from_checkpoint,model)
        def _save_checkpoint(self,model,trial):
            super()._save_checkpoint(model,trial)
            if torch.distributed.is_initialized():
                selection=[(self.state.best_model_checkpoint,self.state.best_metric,self.state.best_global_step)
                           if self.is_world_process_zero() else None]
                torch.distributed.broadcast_object_list(selection,src=0)
                self.state.best_model_checkpoint,self.state.best_metric,self.state.best_global_step=selection[0]
    set_seed(args.seed)
    tok = tokenizer()
    config = json.loads(Path(args.data, "length_report.json").read_text())
    def encode(row):
        prefix = prompt_ids(tok, row["messages"])
        answer = tok.encode(row["answer"], add_special_tokens=False) + [tok.eos_token_id]
        ids = prefix + answer
        if len(ids) > config["max_seq"]:
            raise ValueError("Dados mudaram: execute a preparacao novamente.")
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": [-100] * len(prefix) + answer}
    parts = {}
    for split in ("train", "validation"):
        rows = read_jsonl(Path(args.data, split + ".jsonl"))
        if split == "validation" and args.eval_limit:
            rows = sorted(rows, key=lambda r: digest(str(args.seed) + r["id"]))[:args.eval_limit]
        parts[split] = Dataset.from_list([encode(r) for r in rows])
    model, bf16 = load_model(training=True)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=0.05, target_modules="all-linear", bias="none"))
    model.print_trainable_parameters()
    ta = TrainingArguments(
        output_dir=args.run, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        max_grad_norm=1.0, optim="adamw_torch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True, logging_steps=10,
        logging_nan_inf_filter=False, report_to="none", seed=args.seed,
        data_seed=args.seed, dataloader_num_workers=0, label_names=["labels"])
    trainer = ReceiptTrainer(model=model, args=ta, train_dataset=parts["train"],
                      eval_dataset=parts["validation"], processing_class=tok,
                      data_collator=DataCollatorForSeq2Seq(
                          tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=args.resume or None)
        print(f"DDP: rank local {os.environ.get('LOCAL_RANK','0')}, Trainer concluído.",flush=True)
        if torch.distributed.is_initialized():torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        if trainer.is_world_process_zero():
            final = str(Path(args.run, "final_adapter"))
            trainer.model.save_pretrained(final)
            tok.save_pretrained(final)
            details = {**result.metrics, "seconds": time.perf_counter() - start,
                       "rank0_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                       "arguments": vars(args), "data_config": config,
                       "revisions": json.loads(Path("revisions.json").read_text()),
                       "data_hashes": {s: file_hash(Path(args.data, s + ".jsonl"))
                                       for s in ("train", "validation")}}
            Path(args.run, "run_report.json").write_text(json.dumps(details, indent=2))

            print("Adaptador final e relatório gravados.",flush=True)
        if torch.distributed.is_initialized():torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def remote_call(messages, max_new):
    import urllib.request
    import urllib.error
    endpoint = os.environ["LLM_URL"]  # URL completa de chat/completions.
    extra = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))
    if {"messages", "model"} & set(extra):
        raise ValueError("LLM_EXTRA_JSON nao pode sobrescrever model/messages.")
    body = {"model": os.environ["LLM_MODEL"], "messages": messages, **extra}
    if not any(k in body for k in ("max_tokens", "max_completion_tokens")):
        body["max_tokens"] = max_new
    headers = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"],
               "Content-Type": "application/json"}
    for attempt in range(3):
        request = urllib.request.Request(endpoint, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.load(response)
            choice = raw["choices"][0]
            return (choice["message"].get("content") or "", raw.get("usage", {}),
                    choice.get("finish_reason"), raw.get("model", body["model"]))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError("HTTP " + str(exc.code) + ": verifique endpoint/modelo/parametros.") from exc
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def predict(args):
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Use outro --output para preservar a execucao anterior.")
    report = {"arguments": vars(args), "input_sha256": file_hash(args.input),
              "revisions": json.loads(Path("revisions.json").read_text()),
              "llm_model": os.environ.get("LLM_MODEL") if args.api else None,
              "llm_url": os.environ.get("LLM_URL") if args.api else None,
              "api_extra": json.loads(os.environ.get("LLM_EXTRA_JSON", "{}")) if args.api else None}
    if args.api and args.adapter:
        raise ValueError("Nao combine --api e --adapter.")
    if not args.api:
        import torch
        from transformers import set_seed
        set_seed(args.seed)
        tok = tokenizer()
        model, _ = load_model(args.adapter)
        model.eval()
        gen = dict(max_new_tokens=args.max_new, do_sample=args.sample,
                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if args.sample:
            gen.update(temperature=0.7, top_p=0.8, top_k=20)
        # Aquecimento fora das medidas; usa somente o prompt do primeiro caso elegivel.
        first = next((r for r in rows if r.get("eligible", True)), None)
        if first:
            warm = torch.tensor([prompt_ids(tok, first["messages"])], device=model.device)
            with torch.inference_mode():
                model.generate(warm, attention_mask=torch.ones_like(warm),
                               max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            result = {"id": row["id"], "status": "ok", "text": ""}
            if not row.get("eligible", True):
                result["status"] = "context_exceeded"
            else:
                start = time.perf_counter()
                if args.api:
                    try:
                        text, usage, finish, model_name = remote_call(row["messages"], args.max_new)
                        result.update(text=text, usage=usage, finish_reason=finish, model=model_name)
                        if finish == "length":
                            result["status"] = "output_limit"
                    except Exception as exc:
                        result.update(status="api_error", error=type(exc).__name__)
                        print("Falha de API:", row["id"], str(exc), flush=True)
                else:
                    x = torch.tensor([prompt_ids(tok, row["messages"])], device=model.device)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        y = model.generate(x, attention_mask=torch.ones_like(x), **gen)
                    torch.cuda.synchronize()
                    generated = y[0, x.shape[1]:].tolist()
                    result.update(text=tok.decode(generated, skip_special_tokens=True).strip(),
                                  output_tokens=len(generated), input_tokens=x.shape[1])
                    if len(generated) >= args.max_new and generated[-1] != tok.eos_token_id:
                        result["status"] = "output_limit"
                result["latency_s"] = time.perf_counter() - start
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(result["id"], result["status"], flush=True)
    if not args.api:
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    out.with_suffix(".meta.json").write_text(json.dumps(report, indent=2))


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
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--eval-limit", type=int, default=256)
    t.add_argument("--eval-steps", type=int, default=20)
    t.add_argument("--resume")
    p = sub.add_parser("predict")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter")
    p.add_argument("--api", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()
    train(a) if a.command == "train" else predict(a)


if __name__ == "__main__":
    main()
