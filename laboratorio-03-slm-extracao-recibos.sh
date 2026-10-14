#!/usr/bin/env bash
# Comandos extraídos do Markdown correspondente. Execute etapas em ordem.
set -eo pipefail
LAB_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_SHARED_DIR="${LAB_SHARED_DIR:-$HOME/slm-labs}"
LAB_WORK_DIR="${LAB_WORK_DIR:-$HOME/lab03-recibos}"

usage() {
    cat <<'HELP'
Uso: bash laboratorio-03-slm-extracao-recibos.sh --list
     bash laboratorio-03-slm-extracao-recibos.sh --steps 1-5
     bash laboratorio-03-slm-extracao-recibos.sh --steps 6,7,8

As etapas correspondem aos blocos Bash do Markdown, na mesma ordem.
Escolha as etapas adequadas ao experimento; há alternativas, API e servidor.
Execute etapas dependentes em ordem. A configuração da API e suas chamadas
precisam estar na mesma execução para preservar as variáveis e a chave.
O servidor permanece em primeiro plano; faça a chamada HTTP em outro terminal.
Logs: pasta de trabalho/logs/laboratorio-03-slm-extracao-recibos-DATA-HORA-PID.log (stdout e stderr).
LAB_WORK_DIR e LAB_SHARED_DIR permitem ajustar as pastas.
HELP
}
list_steps() {
    cat <<'STEPS'
01 - 3. Ambiente e memória
02 - 3. Ambiente e memória
03 - 3. Ambiente e memória
04 - 4. Copiar os scripts do laboratório
05 - 5. Preparar os dados sem anotar exemplos
06 - 5. Preparar os dados sem anotar exemplos
07 - 6. Treinar nas duas RTX 3070
08 - 6. Treinar nas duas RTX 3070
09 - 7. Validar antes de abrir o teste
10 - 8. Comparação objetiva: original, especialista e LLM maior
11 - 8. Comparação objetiva: original, especialista e LLM maior
12 - 8. Comparação objetiva: original, especialista e LLM maior
13 - 9. Usar a aplicação de verdade
14 - 9. Usar a aplicação de verdade
15 - 9. Usar a aplicação de verdade
STEPS
}

# Bloco 1 de laboratorio-03-slm-extracao-recibos.md: 3. Ambiente e memória
etapa_1() {
mkdir -p "$LAB_WORK_DIR"
cd "$LAB_WORK_DIR"
cp "$LAB_REPO_DIR"/laboratorio-03-slm-extracao-recibos.md ./laboratorio.md
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
python3 --version
}

# Bloco 2 de laboratorio-03-slm-extracao-recibos.md: 3. Ambiente e memória
etapa_2() {
source "$LAB_SHARED_DIR"/.venv/bin/activate
cp "$LAB_REPO_DIR"/requirements.txt ./requirements.txt
python -m pip freeze > requirements-resolved.txt
mkdir -p runs logs results raw
export TOKENIZERS_PARALLELISM=false
export OMP_THREAD_LIMIT=1
tesseract --version > logs/tesseract-version.txt 2>&1
pdftotext -v > logs/poppler-version.txt 2>&1
}

# Bloco 3 de laboratorio-03-slm-extracao-recibos.md: 3. Ambiente e memória
etapa_3() {
python "$LAB_REPO_DIR"/verificar_ambiente.py
}

# Bloco 4 de laboratorio-03-slm-extracao-recibos.md: 4. Copiar os scripts do laboratório
etapa_4() {
cp -n "$LAB_REPO_DIR"/scripts/lab03-recibos/*.py ./
}

# Bloco 5 de laboratorio-03-slm-extracao-recibos.md: 5. Preparar os dados sem anotar exemplos
etapa_5() {
set -o pipefail
python prepare_receipts.py --out data --mode ocr --lang eng --psm 6 \
  --max-seq 2048 --max-new 512 2>&1 | tee logs/prepare.log
}

# Bloco 6 de laboratorio-03-slm-extracao-recibos.md: 5. Preparar os dados sem anotar exemplos
etapa_6() {
python inspecionar_dados.py
}

# Bloco 7 de laboratorio-03-slm-extracao-recibos.md: 6. Treinar nas duas RTX 3070
etapa_7() {
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu --max-steps 10 --accum 1 \
  --eval-steps 5 --eval-limit 8 2>&1 | tee logs/smoke_1gpu.log

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-steps 10 --eval-limit 16 \
  2>&1 | tee logs/smoke_2gpu.log
}

# Bloco 8 de laboratorio-03-slm-extracao-recibos.md: 6. Treinar nas duas RTX 3070
etapa_8() {
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB03_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc-per-node=2 -- lab_runtime.py train \
  --data data --run runs/main --epochs 3 --accum 8 --rank 16 \
  --lr 0.0001 --eval-steps 20 --eval-limit 256 \
  2>&1 | tee logs/main_train.log
kill "$LAB03_MONITOR_PID"
}

# Bloco 9 de laboratorio-03-slm-extracao-recibos.md: 7. Validar antes de abrir o teste
etapa_9() {
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/validation.jsonl --output results/val_original.jsonl
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/validation.jsonl --output results/val_especialista.jsonl \
  --adapter runs/main/final_adapter
python score_receipts.py data/validation.jsonl results/val_original.jsonl --out results/val_original
python score_receipts.py data/validation.jsonl results/val_especialista.jsonl --out results/val_especialista
}

# Bloco 10 de laboratorio-03-slm-extracao-recibos.md: 8. Comparação objetiva: original, especialista e LLM maior
etapa_10() {
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_original.jsonl
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_especialista.jsonl \
  --adapter runs/main/final_adapter
}

# Bloco 11 de laboratorio-03-slm-extracao-recibos.md: 8. Comparação objetiva: original, especialista e LLM maior
etapa_11() {
export LLM_URL='https://SEU_ENDPOINT/v1/chat/completions'
export LLM_MODEL='Qwen/Qwen3-32B'
read -r -s -p 'Chave do endpoint: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
python lab_runtime.py predict --api \
  --input data/validation.jsonl --output results/val_llm.jsonl --limit 3
}

# Bloco 12 de laboratorio-03-slm-extracao-recibos.md: 8. Comparação objetiva: original, especialista e LLM maior
etapa_12() {
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
python score_receipts.py data/test.jsonl results/test_original.jsonl --out results/original
python score_receipts.py data/test.jsonl results/test_especialista.jsonl --out results/especialista
python score_receipts.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/original.cases.jsonl results/especialista.cases.jsonl
python metrics.py results/llm.cases.jsonl results/especialista.cases.jsonl
}

# Bloco 13 de laboratorio-03-slm-extracao-recibos.md: 9. Usar a aplicação de verdade
etapa_13() {
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py data/demo_images \
  --adapter runs/main/final_adapter --out saida-demo
}

# Bloco 14 de laboratorio-03-slm-extracao-recibos.md: 9. Usar a aplicação de verdade
etapa_14() {
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py /caminho/recibo.png \
  --adapter runs/main/final_adapter --out saida-recibo
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py /caminho/recibo.pdf \
  --adapter runs/main/final_adapter --out saida-pdf
}

# Bloco 15 de laboratorio-03-slm-extracao-recibos.md: 9. Usar a aplicação de verdade
etapa_15() {
cat > exemplo_sintetico.txt <<'TXT'
DEMO SUPPLIES LTD
10 DEMO ROAD
DATE 08/10/2026
SUBTOTAL 84.81
TAX 5.09
TOTAL RM 89.90
CASH 100.00
CHANGE 10.10
TXT
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py exemplo_sintetico.txt \
  --adapter runs/main/final_adapter --out saida-sintetico
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
    if (( LAB_FIRST < 1 || LAB_LAST > 15 )); then
        echo "Use etapas entre 1 e 15." >&2
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
LAB_LOG_FILE="$LAB_WORK_DIR/logs/laboratorio-03-slm-extracao-recibos-$(date +%Y%m%d-%H%M%S)-$$.log"
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
    echo "=== Etapa $LAB_STEP - $(date --iso-8601=seconds) ==="
    "etapa_$LAB_STEP"
done
