# Laboratório 2, SLM especialista em consultas SQL

**José Damico · Laboratório local · Versão 1.0 · 10/10/2026**

**Objetivo:** ajustar um Qwen3-1.7B para converter pergunta, esquema e evidência em SQL; executar as consultas geradas e comparar o resultado com o gabarito, com o modelo original e com um LLM maior.

## 1. Resultado que você construirá

Uma interface de linguagem natural para consultas de leitura sobre bancos. A especialização avaliada é **text-to-SQL em bancos cujos esquemas são fornecidos no prompt**. O treino ensina uma tarefa; não incorpora automaticamente os dados atuais do banco nos pesos.

O treino usa [birdsql/bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered), com 6.601 exemplos filtrados do BIRD. A avaliação final usa o split `mini_dev_sqlite` de [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), com 500 perguntas e seus bancos. Uma parte dos bancos do conjunto de treino fica reservada para validação. O Mini-Dev é tratado como teste final deste laboratório; apesar do nome, não será usado para ajustar hiperparâmetros.

A página oficial do conjunto filtrado mostra ganhos de ajuste de um Qwen2.5-3B sobre a base. É evidência externa de utilidade do dataset, não uma previsão para nosso Qwen3-1.7B/QLoRA. O desempenho precisa ser medido.

Começaremos com **SQLite**, o caminho mais direto para executar os bancos do benchmark sem configurar um servidor. Sua preferência por PostgreSQL faz sentido para a integração futura, mas SQL aprendido/avaliado em SQLite não deve ser anunciado como validado em PostgreSQL. Ao final há uma orientação de expansão para o dialeto PostgreSQL.

```bash
mkdir -p "$HOME/slm-labs/02-sql"
cd "$HOME/slm-labs/02-sql"
```

Este arquivo inclui todos os scripts de que precisa e pode ser utilizado sem executar o laboratório de ferramentas.

## 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB

Este roteiro considera **Linux/Ubuntu, Bash, Python 3.11 ou 3.12, duas placas de 8 GB e aproximadamente 32 GB de RAM**. Se o computador estiver em Windows, execute em Ubuntu/WSL2 com CUDA funcionando, ou use Ubuntu nativo; os comandos abaixo não são de PowerShell.

Use o ambiente compartilhado descrito em [Preparação geral](preparacao-geral.md). Execute os dois laboratórios em sequência quando cada um utilizar as duas GPUs. Feche outras inferências que ocupem VRAM. Reserve inicialmente 25–40 GB de disco para ambiente, cache do modelo e checkpoints; no laboratório BIRD, acrescente espaço para os bancos e os ZIPs, conforme a seção de download.

**As memórias das placas não se somam automaticamente.** O treinamento usa Distributed Data Parallel (DDP): um processo e uma cópia quantizada do modelo em cada GPU. Cada exemplo precisa caber nos 8 GB de uma placa. A segunda GPU distribui exemplos e sincroniza os gradientes dos adaptadores. O ganho de velocidade precisa ser medido; não se presume 2×.

Verifique o driver existente:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
```

Se `nvidia-smi` funciona, não há razão para reinstalar o driver para começar. A indicação “CUDA Version” desse comando descreve a capacidade do driver; o PyTorch instala seu runtime CUDA. A preparação geral usa a distribuição oficial cu128 do PyTorch 2.7.1, que requer um driver compatível. Se seu driver não for compatível, consulte a matriz oficial indicada nas referências antes de trocar a distribuição.

Prepare primeiro o ambiente compartilhado conforme [Preparação geral](preparacao-geral.md). Dentro da pasta do laboratório criada na seção 1, ative esse ambiente:

```bash
source ~/slm-labs/.venv/bin/activate
cp ~/slm-labs/requirements.txt ./requirements.txt
cp ~/slm-labs/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p data runs logs results raw
export TOKENIZERS_PARALLELISM=false
```

As versões foram fixadas para tornar a receita explícita, não por serem as versões mais recentes. O arquivo `requirements-resolved.txt` registra também as dependências transitivas realmente instaladas. A receita precisa passar pelo teste curto no seu equipamento antes de um treino longo.

```bash
python - <<'PY'
import torch
import transformers, peft, accelerate, bitsandbytes, datasets
print('Torch:', torch.__version__, 'CUDA do PyTorch:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA indisponivel'
assert torch.cuda.device_count() >= 2, 'As duas GPUs precisam estar visiveis'
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), 'GiB',
          'BF16:', torch.cuda.is_bf16_supported())
PY
```

O script escolhe BF16 quando o runtime informa suporte; caso contrário, usa FP16. Não usa FlashAttention como dependência adicional. `attn_implementation="sdpa"` deixa o PyTorch selecionar um caminho de atenção compatível.

## 3. Como executar os scripts deste documento

Copie **cada bloco que começa com `cat > ... <<'PY'` integralmente** para o terminal Bash, incluindo o `PY` final. Isso criará os arquivos Python na pasta atual. Crie todos os scripts antes de seguir os comandos das seções seguintes. Os arquivos estão embutidos no Markdown; não é necessário baixar código de um repositório deste tutorial.

O `lab_runtime.py` contém carregamento, treinamento e inferência. O `metrics.py` calcula métricas, intervalos de Wilson e diferenças pareadas. Os demais scripts são específicos do laboratório.

O modelo inicial é **Qwen/Qwen3-1.7B**, em modo sem raciocínio explícito (`enable_thinking=False`). O treino aplica a perda somente à resposta; tokens do prompt e padding recebem rótulo `-100`. A base fica quantizada em NF4 e congelada; os adaptadores LoRA são treinados nas camadas lineares. Isso é fine-tuning supervisionado com QLoRA, e não pré-treinamento do zero.

Na primeira preparação, os scripts registram os commits do modelo e datasets em `revisions.json`. Reexecuções reutilizam essas revisões. Preserve esse arquivo com os resultados.

## 4. Baixar os exemplos e os bancos

**`load_dataset(...)` baixa perguntas e SQLs, mas não substitui o download dos bancos.** Sem os arquivos `.sqlite`, não há esquema real nem avaliação por execução.

Planeje espaço: o BIRD trabalha com bancos grandes, na ordem de dezenas de GB. Como você manterá os ZIPs e as cópias extraídas, reserve aproximadamente **100 GB livres adicionais como margem inicial**, conferindo o tamanho efetivo dos pacotes antes de baixar. Isso é planejamento de disco, não uma medida feita neste ambiente.

O link oficial de treino, referenciado no card do dataset filtrado, é:

```bash
curl -fL --retry 3 -C - \
  'https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip' \
  -o raw/bird_train.zip
```

Para o Mini-Dev, use **Download BIRD Mini-Dev Complete Package** na [página oficial](https://huggingface.co/datasets/birdsql/bird_mini_dev). O link apontado na consulta desta versão é [minidev_0703.zip no Google Drive](https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view). Faça o download pelo navegador e copie-o para `raw/bird_mini.zip` dentro desta pasta. Não salve a página HTML do Drive como se fosse um ZIP. Se o link tiver mudado, siga o link atual do card e registre a mudança.

O script de extração abaixo lida com ZIPs internos; os caminhos finais são encontrados recursivamente. Crie todos os scripts da seção 5 e então execute:

```bash
python extract_archives.py raw/bird_train.zip raw/bird_train
python extract_archives.py raw/bird_mini.zip raw/bird_mini
sha256sum raw/bird_train.zip raw/bird_mini.zip > raw/archive-sha256.txt
python prepare_sql.py --train-root raw/bird_train --test-root raw/bird_mini \
  --out data --max-seq 2048 --max-new 512
cat data/source_report.json
cat data/length_report.json
```

O extrator exige um destino vazio para evitar mistura de versões. Se interromper a extração, use outra pasta de destino e ajuste os argumentos de preparação, ou remova conscientemente apenas a extração incompleta. Não apague os arquivos originais que pretende preservar.

A preparação:

1. Resolve revisões dos dois datasets e do modelo.
2. Localiza os bancos de treino e Mini-Dev; interrompe se faltar algum ou houver duplicidade.
3. Extrai nomes, tipos, chaves primárias e relações de chaves estrangeiras.
4. Reserva aproximadamente 10% dos **bancos** de treino para validação; os demais servem ao treino.
5. Verifica que bancos do Mini-Dev não coincidem com os bancos do conjunto de treino.
6. Monta o prompt com esquema, pergunta e `evidence`, sem utilizar o SQL correto para escolher tabelas ou colunas.
7. Registra limites de contexto, exclusões e caminhos reais dos bancos.

O campo `evidence` é uma informação adicional fornecida pelo benchmark. Todos os modelos recebem a mesma evidência. No uso real essa informação pode não existir, e você precisaria de descrições ou catálogo de dados. Portanto, o resultado deve ser descrito como **text-to-SQL com esquema e evidência fornecidos**.

O esquema compacto evita parte do custo de contexto, mas ainda haverá bancos grandes que não cabem em 2.048 tokens. Esses casos são contabilizados, não truncados. Descrições completas de colunas e amostras de valores não são incluídas nesta primeira receita. Não compare seus números diretamente com sistemas que usam recuperação de valores, amostras, múltiplas tentativas ou correção via execução.

```bash
python - <<'PY'
from lab_runtime import read_jsonl
r = read_jsonl('data/train.jsonl')[0]
print(r['messages'][1]['content'])
print('\nSQL de treino:', r['answer'])
print('\nBanco:', r['db_path'])
PY
```

## 5. Arquivos Python necessários

Crie os cinco arquivos abaixo. `lab_runtime.py` e `metrics.py` são independentes das cópias do laboratório 1; execute todos os comandos a partir da pasta `02-sql`.

### lab_runtime.py

```bash
cat > lab_runtime.py <<'PY'
"""Runtime comum aos dois laboratorios. Python 3.11/3.12."""
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
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True, logging_steps=10,
        logging_nan_inf_filter=False, report_to="none", seed=args.seed,
        data_seed=args.seed, dataloader_num_workers=0, label_names=["labels"])
    trainer = Trainer(model=model, args=ta, train_dataset=parts["train"],
                      eval_dataset=parts["validation"], processing_class=tok,
                      data_collator=DataCollatorForSeq2Seq(
                          tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    if trainer.is_world_process_zero():
        final = str(Path(args.run, "final_adapter"))
        trainer.save_model(final)
        tok.save_pretrained(final)
        details = {**result.metrics, "seconds": time.perf_counter() - start,
                   "rank0_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                   "arguments": vars(args), "data_config": config,
                   "revisions": json.loads(Path("revisions.json").read_text()),
                   "data_hashes": {s: file_hash(Path(args.data, s + ".jsonl"))
                                   for s in ("train", "validation")}}
        Path(args.run, "run_report.json").write_text(json.dumps(details, indent=2))


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
PY
```

### metrics.py

```bash
cat > metrics.py <<'PY'
import json
import math
import statistics
from pathlib import Path
from lab_runtime import read_jsonl, write_jsonl


def predictions(path, references):
    rows = read_jsonl(path)
    result = {r["id"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError("IDs de predicao duplicados.")
    extra = set(result) - {r["id"] for r in references}
    if extra:
        raise ValueError("IDs fora do conjunto de referencia: " + repr(sorted(extra)[:5]))
    return result


def wilson(k, n):
    if not n:
        return None
    z, p = 1.96, k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(center - half, 5), round(center + half, 5)]


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    index = (len(v) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return v[lo] + (v[hi] - v[lo]) * (index - lo)


def report(cases, predicted, output, extra=None):
    valid = [c for c in cases if c.get("reference_valid", True)]
    eligible = [c for c in valid if c.get("eligible", True)]
    lat = [p["latency_s"] for p in predicted.values() if p.get("status") == "ok"
           and "latency_s" in p]
    k = sum(c["correct"] for c in eligible)
    summary = {
        "n_total": len(cases), "n_reference_valid": len(valid),
        "n_eligible_valid": len(eligible), "n_correct": k,
        "accuracy_eligible": k / len(eligible) if eligible else None,
        "wilson95_eligible": wilson(k, len(eligible)),
        "accuracy_all_valid": sum(c["correct"] for c in valid) / len(valid) if valid else None,
        "latency_success_n": len(lat), "p50_s": statistics.median(lat) if lat else None,
        "p95_s": percentile(lat, 0.95),
        "api_prompt_tokens": sum(p.get("usage", {}).get("prompt_tokens", 0) for p in predicted.values()),
        "api_completion_tokens": sum(p.get("usage", {}).get("completion_tokens", 0) for p in predicted.values()),
        **(extra or {})}
    write_jsonl(output + ".cases.jsonl", cases)
    Path(output + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    import random
    ap = argparse.ArgumentParser(description="Diferenca pareada B - A com bootstrap.")
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a = {x["id"]: x for x in read_jsonl(args.a)}
    b = {x["id"]: x for x in read_jsonl(args.b)}
    if a.keys() != b.keys():
        raise ValueError("Compare os mesmos IDs, sem misturar limit e teste completo.")
    flags = lambda c: (c.get("eligible", True), c.get("reference_valid", True))
    if any(flags(a[k]) != flags(b[k]) for k in a):
        raise ValueError("Elegibilidade/referencias diferentes entre execucoes.")
    ids = [k for k in a if all(flags(a[k]))]
    if not ids:
        raise ValueError("Nenhum caso comparavel.")
    delta = [int(b[k]["correct"]) - int(a[k]["correct"]) for k in ids]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(2000))
    print(json.dumps({"n": len(delta), "delta_B_minus_A": sum(delta) / len(delta),
                      "bootstrap95": [percentile(boot, .025), percentile(boot, .975)],
                      "B_wins": delta.count(1), "A_wins": delta.count(-1)}, indent=2))
PY
```

### extract_archives.py

```bash
cat > extract_archives.py <<'PY'
"""Extrai ZIPs oficiais, inclusive arquivos ZIP internos, sem escapar do destino."""
import argparse
import zipfile
from pathlib import Path


def unpack(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Caminho inesperado no ZIP: " + item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Link simbolico inesperado no ZIP.")
        z.extractall(destination)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("destination")
    a = ap.parse_args()
    root = Path(a.destination)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use um diretorio de extracao vazio.")
    unpack(Path(a.archive), root)
    seen = set()
    while True:
        todo = [p for p in root.rglob("*.zip") if str(p.resolve()) not in seen]
        if not todo:
            break
        for p in todo:
            print("Extraindo arquivo interno:", p)
            seen.add(str(p.resolve()))
            unpack(p, p.parent)
    print("Bancos SQLite encontrados:", len(list(root.rglob("*.sqlite"))))
PY
```

### prepare_sql.py

```bash
cat > prepare_sql.py <<'PY'
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
PY
```

### score_sql.py

```bash
cat > score_sql.py <<'PY'
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
PY
```

## 6. Verificar memória e treinar

A configuração inicial é: **contexto total 2.048 tokens, reserva de até 512 tokens para saída, LoRA rank 16, learning rate 0,0001, microbatch 1, acumulação 8 e duas épocas**. É um ponto de partida, não uma garantia de encaixe em 8 GB nem uma configuração ótima demonstrada. O contexto inclui o catálogo/esquema, a pergunta e a resposta.

Primeiro, faça um teste de 10 passos em uma GPU. Ajuste o índice se a GPU 1 não estiver livre:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu \
  --max-steps 10 --accum 1 --eval-limit 32 \
  2>&1 | tee logs/smoke_1gpu.log
```

Depois, verifique o caminho de duas GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
```

Os adaptadores desses testes curtos são apenas diagnósticos. O treinamento principal começa novamente do modelo original em outra pasta:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc_per_node=2 lab_runtime.py train \
  --data data --run runs/main_r16_lr1e4 \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-limit 256 \
  2>&1 | tee logs/main_train.log
kill "$LAB_MONITOR_PID"
```

O batch efetivo é `1 × 8 × 2 = 16` exemplos por atualização. Se treinar em uma GPU e quiser preservar esse batch, use `--accum 16`. Isso não torna o tempo de execução idêntico.

A cada 100 passos, o Trainer avalia a perda em até 256 exemplos fixos da validação e salva um checkpoint. Ao terminar, carrega o melhor checkpoint observado por `eval_loss` e salva `runs/main_r16_lr1e4/final_adapter`. Em testes com menos de 100 passos pode não haver avaliação; nesse caso, o adaptador final é apenas o último estado. A perda ajuda a selecionar checkpoints, mas a avaliação funcional da seção 7 é a medida importante.

Não existe previsão confiável de duração sem executar o teste curto. Use os segundos por passo do seu log e estime o total de passos com `ceil(n_treino / 16) × épocas`; some avaliação e gravação. Não extrapole apenas do primeiro passo, que inclui aquecimento.

Para retomar um treino interrompido, use uma pasta de checkpoint realmente existente:

```bash
python - <<'PY'
from pathlib import Path
print('\n'.join(str(p) for p in sorted(Path('runs/main_r16_lr1e4').glob('checkpoint-*'))))
PY
# Exemplo de forma, substituindo pelo caminho que o comando acima mostrou:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 lab_runtime.py train \
#   --data data --run runs/main_r16_lr1e4 --epochs 2 --accum 8 \
#   --resume runs/main_r16_lr1e4/checkpoint-N
```

Preserve dados, revisões e hiperparâmetros ao retomar. O arquivo `run_report.json` registra configuração, hashes dos dados, tempo e pico de memória alocada no processo da GPU de rank 0. `gpu_train.csv` registra o uso observado das duas placas, incluindo consumo de outros processos.

## 7. Comparar o modelo original, o ajustado e um LLM maior

Use primeiro a **validação** para ajustar prompt, taxa de aprendizado, número de épocas e decodificação. Faça poucas experiências deliberadas: por exemplo, uma e duas épocas, ou learning rates `0.00005` e `0.0001`. Troque o nome de `--run` a cada treino. Não selecione a melhor configuração olhando o teste final.

Para uma verificação funcional rápida, crie um conjunto de até 50 casos elegíveis da validação:

```bash
python - <<'PY'
from lab_runtime import read_jsonl, write_jsonl
rows = [r for r in read_jsonl('data/validation.jsonl') if r['eligible']][:50]
assert rows, 'Nao ha validacao elegivel'
write_jsonl('data/validation_demo.jsonl', rows)
PY
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
```

Avalie esses arquivos com o avaliador da seção 8, apontando a referência para `data/validation_demo.jsonl`. Depois de escolher sua configuração, execute o **teste completo**, uma vez por condição:

```bash
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
```

Os dois modelos locais usam a mesma revisão da base, quantização, prompt, GPU e limite de saída. O padrão do tutorial é decodificação gulosa para uma primeira comparação determinística, sem modo thinking. A documentação Qwen recomenda amostragem no uso geral: `--sample` usa temperatura 0,7, top-p 0,8 e top-k 20. Se escolher amostragem na validação, aplique-a a ambas as condições e repita com sementes fixas, por exemplo 42, 43 e 44, em arquivos distintos. Não escolha a melhor resposta entre tentativas sem declarar um experimento diferente.

Para o **LLM maior**, o script aceita um endpoint HTTPS de chat completions com formato compatível. Escolha o modelo forte que você efetivamente pretende substituir ou complementar; registre provedor, identificador, data e parâmetros. O comando a seguir é configurável: os valores entre `<...>` devem ser substituídos. A chave não deve ser escrita no Markdown nem em Git.

```bash
export LLM_URL='https://<seu-provedor>/<caminho>/chat/completions'
export LLM_MODEL='<identificador-real-do-LLM-maior>'
read -r -s -p 'Chave da API: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":512}'
```

Um exemplo de endpoint real com esse formato é `https://openrouter.ai/api/v1/chat/completions`; consulte o catálogo do provedor para definir `LLM_MODEL` e os custos. Também é possível usar um servidor próprio compatível. Alguns modelos de raciocínio não aceitam `temperature` ou usam `max_completion_tokens`; ajuste `LLM_EXTRA_JSON` à documentação do modelo. Se o orçamento inclui tokens internos de raciocínio, 512 pode ser insuficiente: estabeleça um orçamento adequado **na validação**, registre-o e mantenha-o no teste. O script transmite o JSON adicional como informado.

Faça primeiro uma chamada para confirmar compatibilidade e custo:

```bash
python lab_runtime.py predict --api --limit 1 \
  --input data/validation_demo.jsonl --output results/api_smoke.jsonl
```

Esse arquivo de uma chamada serve apenas para inspeção. Não o pontue contra os 50 casos como se fosse uma avaliação completa. Se houver `api_error`, corrija credenciais, endpoint ou parâmetros antes de continuar. Então:

```bash
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
unset LLM_API_KEY
```

As chamadas de API podem ser cobradas pelo provedor. `--limit` permite um piloto, mas a comparação final deve utilizar o mesmo conjunto de IDs para todos. Para interromper e refazer uma execução, use outro arquivo de saída: o script impede sobrescrita acidental. Ele não retoma uma inferência parcial automaticamente.

**Protocolo de comparação:** todos recebem a mesma informação no prompt e produzem uma resposta textual no contrato especificado. Este script não ativa o modo nativo de tool calling ou saídas estruturadas do provedor. Um LLM com suporte nativo pode obter outro resultado; essa configuração merece uma linha adicional na comparação de produção, com seu próprio cliente e adaptação de formato. Não apresente a condição de prompt uniforme como o melhor desempenho possível de cada plataforma.

A preparação marca `eligible=false` quando o prompt mais a reserva de saída excede o contexto escolhido. Esses casos continuam no teste, recebem `context_exceeded` em todas as condições e aparecem na cobertura. Assim, não se corta silenciosamente um esquema ou catálogo. A elegibilidade do teste não usa o tamanho da resposta correta. Os dados de treino/validação também excluem respostas que excedam o contexto; as contagens ficam nos relatórios.

## 8. Avaliar executando as consultas

```bash
python score_sql.py data/validation_demo.jsonl results/val_base.jsonl --out results/val_base
python score_sql.py data/validation_demo.jsonl results/val_tuned.jsonl --out results/val_tuned
```

Depois do treinamento/configuração final e das três inferências completas:

```bash
python score_sql.py data/test.jsonl results/test_base.jsonl --out results/base
python score_sql.py data/test.jsonl results/test_tuned.jsonl --out results/tuned
python score_sql.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/base.cases.jsonl results/tuned.cases.jsonl
python metrics.py results/llm.cases.jsonl results/tuned.cases.jsonl
```

Cada consulta é executada com conexão `mode=ro`, `PRAGMA query_only=ON`, extensões desativadas, autorizador que permite leituras e funções SQL, limite de linhas e um limite de tempo acompanhado pelo SQLite. Comandos de escrita, `ATTACH`, alterações de PRAGMA e carregamento de extensão são recusados. Use os bancos públicos locais do laboratório; não aponte esse avaliador para um banco de produção. Esse isolamento é parte da implementação de um executor de consultas geradas, não uma validação de autorização empresarial.

O timeout padrão é 15 segundos por consulta. Ele usa o progress handler do SQLite, não um sandbox de processo com limite rígido de CPU/memória. Para um serviço exposto a usuários, coloque o executor em um processo/container isolado com limites do sistema. Para consultas legítimas mais lentas, defina antecipadamente outro `--timeout` e aplique o mesmo valor em todas as condições.

Métricas:

- **`accuracy_eligible` / EX-set local:** compara o conjunto de linhas retornadas pela consulta gerada com o conjunto do SQL de referência. O texto do SQL pode ser diferente.
- **`ex_bag`:** compara também a multiplicidade das linhas retornadas.
- **`execution_rate`:** consulta válida e executável, mesmo que responda à pergunta errada.
- **`by_difficulty`:** acerto por nível de dificuldade do Mini-Dev.
- **`gold_empty_n`:** referências que retornaram vazio, úteis para revisar acertos acidentais.

As métricas locais não verificam ordem das linhas e não aplicam tolerância numérica a diferenças de ponto flutuante. EX-set ignora duplicatas; EX-bag as preserva. Consultas diferentes podem coincidir por acaso em um único estado do banco. Uma resposta com ordenação incorreta pode passar aqui; revise casos com ordenação e resultados vazios e, para alegações formais, use testes adicionais e o avaliador oficial. A comparação de tuplas mantém a ordem das colunas.

Se a própria referência falhar ou exceder o limite, o caso é registrado como `reference_valid=false` e não entra no denominador válido. Investigue esses casos **independentemente do resultado dos modelos**. As três condições precisam ter o mesmo conjunto de referências válidas; o comparador pareado rejeita divergências. Uma consulta do modelo que falha diante de referência válida conta como erro.

**Este avaliador é uma medida local explícita, não uma implementação certificada do score oficial BIRD.** Para publicar uma posição ou comparar com o leaderboard, utilize a versão correspondente dos dados e o [avaliador oficial](https://github.com/bird-bench/mini_dev/tree/main/evaluation). Não misture os 500 casos SQLite do Mini-Dev clássico com outras extensões que incluem CRUD.

### Expansão posterior para PostgreSQL

O Mini-Dev também disponibiliza o split `mini_dev_pg` e material de construção dos bancos PostgreSQL. Faça essa expansão em outro experimento: importe o pacote oficial em um banco separado, use um usuário somente leitura, ajuste prompt/dialeto e troque o executor SQLite por um cliente PostgreSQL. Treinar apenas com SQL SQLite e avaliar PostgreSQL mede transferência entre dialetos, não um especialista PostgreSQL plenamente treinado.

Para integrar ao InfiniteStack, acrescente seleção autorizada de schemas, descrições de colunas, limite de execução e validação de consulta no backend. Uma demonstração de três perguntas de negócio contra um banco de exemplo é suficiente para mostrar o caminho de produto; ela permanece separada da avaliação quantitativa BIRD.

## 9. Interpretar os resultados e demonstrar valor

Cada avaliação produz `*.summary.json` e `*.cases.jsonl`. A tabela final da apresentação deve ser preenchida com seus números:

| Condição | N elegíveis/válidos | Acerto funcional | IC 95% | Latência p50/p95 | Pico VRAM | Custo por 1.000 solicitações |
|---|---:|---:|---|---|---|---|
| SLM original | medir | medir | calcular | medir | medir | estimar com premissas |
| SLM ajustado | medir | medir | calcular | medir | medir | estimar com premissas |
| LLM maior | mesmo conjunto | medir | calcular | medir ponta a ponta | não observável em API | calcular pelo consumo real |

`accuracy_eligible` usa apenas casos elegíveis e com referência válida; `accuracy_all_valid` também penaliza os casos que não couberam no contexto. Mostre ambos e a cobertura. Predições ausentes, erros de API e respostas que atingiram o limite de saída contam como falhas, não desaparecem da acurácia. Latências p50/p95 são calculadas sobre requisições bem-sucedidas; o número desse subconjunto aparece no relatório.

Os tempos locais excluem carregamento do modelo e incluem geração após um aquecimento. O tempo de API inclui rede, serviço e eventuais tentativas. Isso mede a experiência desses dois caminhos, não uma comparação isolada da velocidade dos chips. Para latência mais estável, faça três passagens separadas, sem outro processo utilizando as GPUs, e reporte a dispersão. Não compare tokens/segundo de tokenizadores diferentes como se fossem uma unidade linguística idêntica.

Os scripts registram o consumo de tokens retornado pela API. Com preços vigentes por milhão de tokens de entrada e saída, calcule:

`custo_API = tokens_entrada / 1e6 × preço_entrada + tokens_saída / 1e6 × preço_saída`

Depois normalize pelo número de solicitações. Inclua tarifas específicas de cache/raciocínio quando existirem. Custo local não é zero: declare energia, amortização do equipamento, tempo de operação e quantas inferências amortizam o treinamento. `power.draw` mede as GPUs, não o computador inteiro; não o apresente como consumo total da máquina.

O intervalo de Wilson fornecido é uma aproximação binomial por exemplo. Para diferença entre modelos, use a comparação pareada da seção 8. O bootstrap também é por exemplo; correlações dentro do mesmo catálogo ou banco podem estreitar artificialmente o intervalo. Para publicação científica, adicione bootstrap por grupo/banco e um teste independente mais amplo.

Resultados possíveis que justificam uma boa aula:

1. O ajuste melhora claramente o pequeno modelo e aproxima sua qualidade do LLM maior.
2. O especialista supera o LLM maior **no recorte avaliado**, com diferença sustentada pelos dados.
3. O especialista mantém qualidade suficiente para um requisito definido previamente, com uma vantagem operacional medida.
4. O LLM maior continua melhor, e o laboratório mostra onde a especialização não compensou.

Não escolha somente exemplos em que o SLM vence. Selecione para a demonstração: um acerto compartilhado, uma melhoria após o ajuste, uma falha do especialista e um caso fora do escopo. Use o teste apenas após congelar as escolhas. Modelos podem ter visto benchmarks públicos no pré-treinamento; a separação feita aqui impede vazamento no **seu fine-tuning**, mas não certifica ausência de contaminação anterior.

O idioma dos datasets é inglês. Apresente os resultados quantitativos em inglês. Uma demonstração em português pode ser incluída separadamente como teste de transferência, sem afirmar que houve especialização em português. Traduzir todo o dataset mudaria o experimento e exigiria nova revisão.

## 10. Problemas frequentes

| Sintoma | Ação concreta |
|---|---|
| `CUDA out of memory` | Pare o treino, confira processos em `nvidia-smi`, mantenha microbatch 1 e refaça a preparação com `--max-seq 1536` ou `1024`. Isso reduz cobertura: use os novos dados em todas as condições. Acumulação de gradientes não reduz a memória de uma sequência individual. |
| Ainda não cabe | Reduza `--rank` para 8; considere limitar LoRA a `q_proj/k_proj/v_proj/o_proj` como nova configuração documentada. Evite trocar silenciosamente o modelo durante a comparação. |
| Travamento/erro NCCL somente com duas GPUs | Confira `nvidia-smi topo -m` e teste uma GPU. Para diagnóstico de P2P, tente apenas nessa execução `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun ...`. Pode reduzir desempenho. Não altere IOMMU/BIOS como primeira medida. |
| Apenas uma GPU usada | Use o comando `torchrun --nproc_per_node=2`; a inferência com `predict` é deliberadamente de uma GPU. |
| Perda `nan`/`inf` | Confira precisão escolhida, reduza learning rate e investigue o log; não continue treinando esperando correção espontânea. |
| Erro de acesso ao xLAM | Entre no Hugging Face, aceite as condições do dataset e execute `hf auth login` no mesmo usuário/ambiente. |
| Muitos casos fora do contexto | Inspecione `length_report.json`. Aumentar contexto eleva memória; um esquema/catálogo mais compacto deve usar regras disponíveis na inferência, nunca tabelas/funções reveladas pela resposta correta. |
| Bom treino, validação pior | Reduza épocas/LR e confira o mascaramento da perda, separação por grupos e diversidade. Não altere o teste para melhorar o resultado. |
| JSON/SQL truncado | Ajuste `--max-new` e a reserva `--max-new` na preparação de forma coordenada; refaça todas as condições. Use outro diretório de dados e resultados. |
| Modelos geram explicações | Verifique que está usando o mesmo template, `enable_thinking=False`, prompt e revisão registrados. Não acrescente um pós-processamento inteligente só ao especialista. |

O script usa um mapa explícito `device_map={"": LOCAL_RANK}` para treinar. Não o substitua por `device_map="auto"` em DDP: esse modo pode espalhar um modelo por GPUs e conflitar com um processo por placa.

## 11. Roteiro para a apresentação de uma hora

| Minutos | Demonstração |
|---|---|
| 0–7 | Problema, aplicação real, hipótese e baseline |
| 7–15 | Dataset pronto, estrutura de um exemplo e divisão dos conjuntos |
| 15–25 | Quantização, LoRA, perda somente na resposta e distribuição entre GPUs |
| 25–35 | Comandos utilizados, logs, memória e curva de validação do treino já concluído |
| 35–48 | Comparação completa e demonstração de três ou quatro casos |
| 48–55 | Falhas, custo, limites e como integrar |
| 55–60 | Perguntas |

O treinamento completo é feito antes. Leve adaptador, cache do modelo, datasets e resultados em disco. A parte local deve funcionar sem internet quando os arquivos já estiverem no cache; o LLM por API deve ter suas respostas previamente registradas para a aula não depender da conexão.

Se mostrar **os dois laboratórios na mesma hora**, use aproximadamente 10 minutos de fundamentos comuns, 20 minutos de ferramentas, 20 minutos de SQL e 10 minutos de comparação e perguntas.

## 12. O que preservar ao terminar

Guarde o Markdown, scripts extraídos, `requirements.txt`, `requirements-resolved.txt`, `revisions.json`, relatórios de preparação, JSONLs utilizados, logs, `run_report.json`, adaptador final, predições, metadados de inferência e métricas. Registre a versão do driver e qual GPU serviu cada medição.

O adaptador sozinho não é um modelo completo. Para reutilizá-lo, carregue a mesma revisão do Qwen base e depois o adaptador, como faz `load_model`. Não é necessário mesclar os pesos para a demonstração. Uma futura conversão para GGUF/Ollama deve ser tratada como outro artefato, com nova avaliação, porque a quantização e o runtime mudam.

**Validação deste tutorial:** sintaxe dos scripts e lógica de avaliação/preparação verificadas com exemplos controlados. O treinamento CUDA, os downloads completos e as chamadas a um LLM pago não foram executados no ambiente de elaboração. Os testes curtos acima são as verificações necessárias no seu laboratório; não há acurácia, duração ou consumo de VRAM inventados neste documento.

## 13. Referências primárias

Fontes consultadas em 10/10/2026; preserve revisões e hashes dos pacotes utilizados.

- [BIRD, conjunto de treino filtrado](https://huggingface.co/datasets/birdsql/bird23-train-filtered).
- [BIRD, Mini-Dev e splits por dialeto](https://huggingface.co/datasets/birdsql/bird_mini_dev).
- [BIRD, repositório Mini-Dev e avaliadores](https://github.com/bird-bench/mini_dev).
- [BIRD, artigo original](https://arxiv.org/abs/2305.03111).
- [Python, sqlite3, URI, autorizador e progress handler](https://docs.python.org/3/library/sqlite3.html).

- [Qwen3-1.7B, modelo, template e modo sem thinking](https://huggingface.co/Qwen/Qwen3-1.7B).
- [PEFT 0.17, quantização e QLoRA](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
- [Transformers 4.55.4, Trainer e TrainingArguments](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer).
- [PyTorch, instalações de versões anteriores](https://pytorch.org/get-started/previous-versions/).
- [bitsandbytes 0.47, requisitos de instalação](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/installation).
- [Hugging Face Hub 0.34.4, autenticação e CLI](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/guides/cli).
- [OpenRouter, formato da API usado como exemplo de endpoint compatível](https://openrouter.ai/docs/api_reference/overview).
