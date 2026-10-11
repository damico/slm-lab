#!/usr/bin/env bash
# Comandos extraídos do Markdown correspondente. Execute etapas em ordem.
set -eo pipefail
LAB_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_SHARED_DIR="${LAB_SHARED_DIR:-$HOME/slm-labs}"
LAB_WORK_DIR="${LAB_WORK_DIR:-$HOME/slm-labs/02-sql}"

usage() {
    cat <<'HELP'
Uso: bash laboratorio-02-slm-sql-bird.sh --list
     bash laboratorio-02-slm-sql-bird.sh --steps 1-5
     bash laboratorio-02-slm-sql-bird.sh --steps 6,7,8

As etapas correspondem aos blocos Bash do Markdown, na mesma ordem.
Escolha as etapas adequadas ao experimento; há alternativas, API e servidor.
Execute etapas dependentes em ordem. A configuração da API e suas chamadas
precisam estar na mesma execução para preservar as variáveis e a chave.
O servidor permanece em primeiro plano; faça a chamada HTTP em outro terminal.
Logs: pasta de trabalho/logs/laboratorio-02-slm-sql-bird-DATA-HORA-PID.log (stdout e stderr).
LAB_WORK_DIR e LAB_SHARED_DIR permitem ajustar as pastas.
HELP
}
list_steps() {
    cat <<'STEPS'
01 — 1. Resultado que você construirá
02 — 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
03 — 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
04 — 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
05 — 3. Como executar os scripts deste documento
06 — 4. Baixar os exemplos e os bancos
07 — 4. Baixar os exemplos e os bancos
08 — 4. Baixar os exemplos e os bancos
09 — 6. Verificar memória e treinar
10 — 6. Verificar memória e treinar
11 — 6. Verificar memória e treinar
12 — 6. Verificar memória e treinar
13 — 7. Comparar o modelo original, o ajustado e um LLM maior
14 — 7. Comparar o modelo original, o ajustado e um LLM maior
15 — 7. Comparar o modelo original, o ajustado e um LLM maior
16 — 7. Comparar o modelo original, o ajustado e um LLM maior
17 — 7. Comparar o modelo original, o ajustado e um LLM maior
18 — 8. Avaliar executando as consultas
19 — 8. Avaliar executando as consultas
STEPS
}

# Bloco 1 de laboratorio-02-slm-sql-bird.md: 1. Resultado que você construirá
etapa_1() {
mkdir -p "$LAB_WORK_DIR"
cd "$LAB_WORK_DIR"
}

# Bloco 2 de laboratorio-02-slm-sql-bird.md: 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
etapa_2() {
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
}

# Bloco 3 de laboratorio-02-slm-sql-bird.md: 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
etapa_3() {
source "$LAB_SHARED_DIR"/.venv/bin/activate
cp "$LAB_SHARED_DIR"/requirements.txt ./requirements.txt
cp "$LAB_SHARED_DIR"/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p data runs logs results raw
export TOKENIZERS_PARALLELISM=false
}

# Bloco 4 de laboratorio-02-slm-sql-bird.md: 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB
etapa_4() {
python "$LAB_REPO_DIR"/verificar_ambiente.py
}

# Bloco 5 de laboratorio-02-slm-sql-bird.md: 3. Como executar os scripts deste documento
etapa_5() {
cp -n "$LAB_REPO_DIR"/scripts/lab02-sql/*.py ./
}

# Bloco 6 de laboratorio-02-slm-sql-bird.md: 4. Baixar os exemplos e os bancos
etapa_6() {
mkdir -p raw
curl -fL --retry 3 -C - \
  'https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip' \
  -o raw/bird_train.zip
}

# Bloco 7 de laboratorio-02-slm-sql-bird.md: 4. Baixar os exemplos e os bancos
etapa_7() {
python extract_archives.py raw/bird_train.zip raw/bird_train
python extract_archives.py raw/bird_mini.zip raw/bird_mini
sha256sum raw/bird_train.zip raw/bird_mini.zip > raw/archive-sha256.txt
python prepare_sql.py --train-root raw/bird_train --test-root raw/bird_mini \
  --out data --max-seq 2048 --max-new 512
cat data/source_report.json
cat data/length_report.json
}

# Bloco 8 de laboratorio-02-slm-sql-bird.md: 4. Baixar os exemplos e os bancos
etapa_8() {
python inspecionar_dados.py
}

# Bloco 9 de laboratorio-02-slm-sql-bird.md: 6. Verificar memória e treinar
etapa_9() {
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu \
  --max-steps 10 --accum 1 --eval-limit 32 \
  2>&1 | tee logs/smoke_1gpu.log
}

# Bloco 10 de laboratorio-02-slm-sql-bird.md: 6. Verificar memória e treinar
etapa_10() {
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
}

# Bloco 11 de laboratorio-02-slm-sql-bird.md: 6. Verificar memória e treinar
etapa_11() {
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc_per_node=2 lab_runtime.py train \
  --data data --run runs/main_r16_lr1e4 \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-limit 256 \
  2>&1 | tee logs/main_train.log
kill "$LAB_MONITOR_PID"
}

# Bloco 12 de laboratorio-02-slm-sql-bird.md: 6. Verificar memória e treinar
etapa_12() {
python listar_checkpoints.py
# Exemplo de forma, substituindo pelo caminho que o comando acima mostrou:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 lab_runtime.py train \
#   --data data --run runs/main_r16_lr1e4 --epochs 2 --accum 8 \
#   --resume runs/main_r16_lr1e4/checkpoint-N
}

# Bloco 13 de laboratorio-02-slm-sql-bird.md: 7. Comparar o modelo original, o ajustado e um LLM maior
etapa_13() {
python preparar_demo.py
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
}

# Bloco 14 de laboratorio-02-slm-sql-bird.md: 7. Comparar o modelo original, o ajustado e um LLM maior
etapa_14() {
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
}

# Bloco 15 de laboratorio-02-slm-sql-bird.md: 7. Comparar o modelo original, o ajustado e um LLM maior
etapa_15() {
export LLM_URL='https://<seu-provedor>/<caminho>/chat/completions'
export LLM_MODEL='<identificador-real-do-LLM-maior>'
read -r -s -p 'Chave da API: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":512}'
}

# Bloco 16 de laboratorio-02-slm-sql-bird.md: 7. Comparar o modelo original, o ajustado e um LLM maior
etapa_16() {
python lab_runtime.py predict --api --limit 1 \
  --input data/validation_demo.jsonl --output results/api_smoke.jsonl
}

# Bloco 17 de laboratorio-02-slm-sql-bird.md: 7. Comparar o modelo original, o ajustado e um LLM maior
etapa_17() {
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
unset LLM_API_KEY
}

# Bloco 18 de laboratorio-02-slm-sql-bird.md: 8. Avaliar executando as consultas
etapa_18() {
python score_sql.py data/validation_demo.jsonl results/val_base.jsonl --out results/val_base
python score_sql.py data/validation_demo.jsonl results/val_tuned.jsonl --out results/val_tuned
}

# Bloco 19 de laboratorio-02-slm-sql-bird.md: 8. Avaliar executando as consultas
etapa_19() {
python score_sql.py data/test.jsonl results/test_base.jsonl --out results/base
python score_sql.py data/test.jsonl results/test_tuned.jsonl --out results/tuned
python score_sql.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/base.cases.jsonl results/tuned.cases.jsonl
python metrics.py results/llm.cases.jsonl results/tuned.cases.jsonl
}

if [[ $# == 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --list && $# == 1 ]]; then
    list_steps
    exit 0
fi
if [[ ${1:-} != --steps || $# != 2 ]]; then
    usage >&2
    exit 2
fi
LAB_SELECTED_STEPS=()
IFS=',' read -r -a LAB_PARTS <<< "$2"
for LAB_PART in "${LAB_PARTS[@]}"; do
    if [[ $LAB_PART =~ ^([0-9]+)-([0-9]+)$ ]]; then
        LAB_FIRST=$((10#${BASH_REMATCH[1]}))
        LAB_LAST=$((10#${BASH_REMATCH[2]}))
        if (( LAB_FIRST > LAB_LAST )); then
            echo "Intervalo inválido: $LAB_PART" >&2
            exit 2
        fi
    elif [[ $LAB_PART =~ ^[0-9]+$ ]]; then
        LAB_FIRST=$((10#$LAB_PART))
        LAB_LAST=$LAB_FIRST
    else
        echo "Etapa inválida: $LAB_PART" >&2
        exit 2
    fi
    if (( LAB_FIRST < 1 || LAB_LAST > 19 )); then
        echo "Use etapas entre 1 e 19." >&2
        exit 2
    fi
    for ((LAB_STEP=LAB_FIRST; LAB_STEP<=LAB_LAST; LAB_STEP++)); do
        LAB_SELECTED_STEPS+=("$LAB_STEP")
    done
done
if (( ${#LAB_SELECTED_STEPS[@]} == 0 )); then
    echo 'Selecione pelo menos uma etapa.' >&2
    exit 2
fi
mkdir -p "$LAB_WORK_DIR/logs"
cd "$LAB_WORK_DIR"
LAB_LOG_FILE="$LAB_WORK_DIR/logs/laboratorio-02-slm-sql-bird-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee -a "$LAB_LOG_FILE") 2>&1
cleanup() {
    local lab_status=$?
    trap - EXIT
    for lab_pid in "${LAB_MONITOR_PID:-}" "${LAB03_MONITOR_PID:-}" "${LAB04_MONITOR_PID:-}"; do
        if [[ -n $lab_pid ]]; then
            kill "$lab_pid" 2>/dev/null || true
            wait "$lab_pid" 2>/dev/null || true
        fi
    done
    echo "Fim: código $lab_status. Log: $LAB_LOG_FILE"
    exit "$lab_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Log: $LAB_LOG_FILE"
echo "Pasta de trabalho: $LAB_WORK_DIR"
# Etapas posteriores também recebem o ambiente necessário em uma nova execução.
if [[ -f "$LAB_SHARED_DIR/.venv/bin/activate" ]]; then
    source "$LAB_SHARED_DIR/.venv/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_THREAD_LIMIT=1
for LAB_STEP in "${LAB_SELECTED_STEPS[@]}"; do
    echo "=== Etapa $LAB_STEP — $(date --iso-8601=seconds) ==="
    "etapa_$LAB_STEP"
done
