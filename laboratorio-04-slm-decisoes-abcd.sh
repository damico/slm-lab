#!/usr/bin/env bash
# Comandos extraídos do Markdown correspondente. Execute etapas em ordem.
set -eo pipefail
LAB_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_SHARED_DIR="${LAB_SHARED_DIR:-$HOME/slm-labs}"
LAB_WORK_DIR="${LAB_WORK_DIR:-$HOME/lab04-decisoes}"

usage() {
    cat <<'HELP'
Uso: bash laboratorio-04-slm-decisoes-abcd.sh --list
     bash laboratorio-04-slm-decisoes-abcd.sh --steps 1-5
     bash laboratorio-04-slm-decisoes-abcd.sh --steps 6,7,8

As etapas correspondem aos blocos Bash do Markdown, na mesma ordem.
Escolha as etapas adequadas ao experimento; há alternativas, API e servidor.
Execute etapas dependentes em ordem. A configuração da API e suas chamadas
precisam estar na mesma execução para preservar as variáveis e a chave.
O servidor permanece em primeiro plano; faça a chamada HTTP em outro terminal.
Logs: pasta de trabalho/logs/laboratorio-04-slm-decisoes-abcd-DATA-HORA-PID.log (stdout e stderr).
LAB_WORK_DIR e LAB_SHARED_DIR permitem ajustar as pastas.
HELP
}
list_steps() {
    cat <<'STEPS'
01 - 4. Preparar o ambiente nas duas placas de 8 GB
02 - 4. Preparar o ambiente nas duas placas de 8 GB
03 - 4. Preparar o ambiente nas duas placas de 8 GB
04 - 5. Copiar os scripts do laboratório
05 - 6. Baixar e preparar os dados
06 - 6. Baixar e preparar os dados
07 - 6. Baixar e preparar os dados
08 - 7. Teste curto e treinamento
09 - 7. Teste curto e treinamento
10 - 8. Validar o especialista e treinar uma referência simples
11 - 8. Validar o especialista e treinar uma referência simples
12 - 9. Calibrar probabilidades e escolher quando revisar
13 - 9. Calibrar probabilidades e escolher quando revisar
14 - 10. Comparar com o modelo original e um LLM maior
15 - 10. Comparar com o modelo original e um LLM maior
16 - 10. Comparar com o modelo original e um LLM maior
17 - 11. Usar a CLI e a API local
18 - 11. Usar a CLI e a API local
19 - 11. Usar a CLI e a API local
20 - 11. Usar a CLI e a API local
STEPS
}

# Bloco 1 de laboratorio-04-slm-decisoes-abcd.md: 4. Preparar o ambiente nas duas placas de 8 GB
etapa_1() {
mkdir -p "$LAB_WORK_DIR"
cd "$LAB_WORK_DIR"
cp "$LAB_REPO_DIR"/laboratorio-04-slm-decisoes-abcd.md ./laboratorio.md
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
}

# Bloco 2 de laboratorio-04-slm-decisoes-abcd.md: 4. Preparar o ambiente nas duas placas de 8 GB
etapa_2() {
source "$LAB_SHARED_DIR"/.venv/bin/activate
cp "$LAB_SHARED_DIR"/requirements.txt ./requirements.txt
cp "$LAB_SHARED_DIR"/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p raw runs results logs
export TOKENIZERS_PARALLELISM=false
}

# Bloco 3 de laboratorio-04-slm-decisoes-abcd.md: 4. Preparar o ambiente nas duas placas de 8 GB
etapa_3() {
python "$LAB_REPO_DIR"/verificar_ambiente.py
}

# Bloco 4 de laboratorio-04-slm-decisoes-abcd.md: 5. Copiar os scripts do laboratório
etapa_4() {
cp -n "$LAB_REPO_DIR"/scripts/lab04-decisoes/*.py ./
}

# Bloco 5 de laboratorio-04-slm-decisoes-abcd.md: 6. Baixar e preparar os dados
etapa_5() {
set -o pipefail
python prepare_abcd.py --out data --train-conversations 2000 \
  --max-seq 2048 --policy-tokens 400 2>&1 | tee logs/prepare.log
}

# Bloco 6 de laboratorio-04-slm-decisoes-abcd.md: 6. Baixar e preparar os dados
etapa_6() {
python inspecionar_dados.py
}

# Bloco 7 de laboratorio-04-slm-decisoes-abcd.md: 6. Baixar e preparar os dados
etapa_7() {
python prepare_abcd.py --out data_full --train-conversations 0 \
  --max-seq 2048 --policy-tokens 400
}

# Bloco 8 de laboratorio-04-slm-decisoes-abcd.md: 7. Teste curto e treinamento
etapa_8() {
CUDA_VISIBLE_DEVICES=1 python decision_lab.py train --data data \
  --run runs/smoke_1gpu --max-steps 10 --accum 1 --eval-steps 5 --eval-limit 16 \
  2>&1 | tee logs/smoke_1gpu.log
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  decision_lab.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-steps 10 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
}

# Bloco 9 de laboratorio-04-slm-decisoes-abcd.md: 7. Teste curto e treinamento
etapa_9() {
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB04_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc-per-node=2 -- decision_lab.py train \
  --data data --run runs/main --epochs 2 --accum 8 --rank 16 --lr 0.0001 \
  --eval-steps 100 --eval-limit 512 2>&1 | tee logs/main_train.log
kill "$LAB04_MONITOR_PID"
}

# Bloco 10 de laboratorio-04-slm-decisoes-abcd.md: 8. Validar o especialista e treinar uma referência simples
etapa_10() {
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/validation.jsonl --output results/val_specialist.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_specialist.jsonl \
  --out results/val_specialist
}

# Bloco 11 de laboratorio-04-slm-decisoes-abcd.md: 8. Validar o especialista e treinar uma referência simples
etapa_11() {
python decision_lab.py baseline --data data --output runs/tfidf.joblib
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/validation.jsonl --output results/val_tfidf.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_tfidf.jsonl \
  --out results/val_tfidf
}

# Bloco 12 de laboratorio-04-slm-decisoes-abcd.md: 9. Calibrar probabilidades e escolher quando revisar
etapa_12() {
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/calibration.jsonl --output results/calibration_logits.jsonl
python evaluate_decisions.py calibrate data/calibration.jsonl results/calibration_logits.jsonl \
  --out runs/temperature.json
}

# Bloco 13 de laboratorio-04-slm-decisoes-abcd.md: 9. Calibrar probabilidades e escolher quando revisar
etapa_13() {
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/selection.jsonl --output results/selection_logits.jsonl
python evaluate_decisions.py select data/selection.jsonl results/selection_logits.jsonl \
  --calibration runs/temperature.json --target-error 0.10 --min-selected 100 \
  --out runs/decision_policy.json
}

# Bloco 14 de laboratorio-04-slm-decisoes-abcd.md: 10. Comparar com o modelo original e um LLM maior
etapa_14() {
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/test.jsonl --output results/test_specialist.jsonl
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data --backend base \
  --input data/test.jsonl --output results/test_original.jsonl
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/test.jsonl --output results/test_tfidf.jsonl
}

# Bloco 15 de laboratorio-04-slm-decisoes-abcd.md: 10. Comparar com o modelo original e um LLM maior
etapa_15() {
export LLM_URL='https://api.openai.com/v1/chat/completions'
export LLM_MODEL='gpt-4.1-mini'
read -r -s -p 'Chave da API OpenAI: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":32}'
python count_api_calls.py
python decision_lab.py predict --data data --backend api \
  --input data/validation.jsonl --output results/val_api_probe.jsonl --limit 3
}

# Bloco 16 de laboratorio-04-slm-decisoes-abcd.md: 10. Comparar com o modelo original e um LLM maior
etapa_16() {
python decision_lab.py predict --data data --backend api \
  --input data/test.jsonl --output results/test_llm.jsonl
python evaluate_decisions.py score data/test.jsonl results/test_specialist.jsonl \
  --calibration runs/decision_policy.json --out results/specialist
python evaluate_decisions.py score data/test.jsonl results/test_original.jsonl --out results/original
python evaluate_decisions.py score data/test.jsonl results/test_tfidf.jsonl --out results/tfidf
python evaluate_decisions.py score data/test.jsonl results/test_llm.jsonl --out results/llm
python evaluate_decisions.py compare results/original.cases.jsonl results/specialist.cases.jsonl
python evaluate_decisions.py compare results/llm.cases.jsonl results/specialist.cases.jsonl
}

# Bloco 17 de laboratorio-04-slm-decisoes-abcd.md: 11. Usar a CLI e a API local
etapa_17() {
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json \
  --input data/demo/case_00.json
}

# Bloco 18 de laboratorio-04-slm-decisoes-abcd.md: 11. Usar a CLI e a API local
etapa_18() {
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json --port 8084
}

# Bloco 19 de laboratorio-04-slm-decisoes-abcd.md: 11. Usar a CLI e a API local
etapa_19() {
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' \
  --data-binary @data/demo/case_00.json
}

# Bloco 20 de laboratorio-04-slm-decisoes-abcd.md: 11. Usar a CLI e a API local
etapa_20() {
cat > meu_caso.json <<'JSON'
{
  "history": [
    {"role":"agent","text":"How can I help you?"},
    {"role":"customer","text":"I want to return a shirt because it is the wrong size."},
    {"role":"agent","text":"May I have your full name?"},
    {"role":"customer","text":"Alex Example"}
  ]
}
JSON
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' --data-binary @meu_caso.json
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
    if (( LAB_FIRST < 1 || LAB_LAST > 20 )); then
        echo "Use etapas entre 1 e 20." >&2
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
LAB_LOG_FILE="$LAB_WORK_DIR/logs/laboratorio-04-slm-decisoes-abcd-$(date +%Y%m%d-%H%M%S)-$$.log"
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
