#!/usr/bin/env bash
# Blocos extraídos do roteiro Lab 05. Escolha etapas e revise a supervisão antes do treino.
set -eo pipefail
LAB_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_SHARED_DIR="${LAB_SHARED_DIR:-$HOME/slm-labs}"
LAB_WORK_DIR="${LAB_WORK_DIR:-$HOME/slm-labs/05-finqa}"
list_steps() {
cat <<'STEPS'
01 - 2. Ambiente único e separação das GPUs
02 - 2. Ambiente único e separação das GPUs
03 - 3. Preparar dados e travar revisões
04 - 4. Conferir o executor sem carregar modelos
05 - 5. Professor: smoke test e piloto offline
06 - 5. Professor: smoke test e piloto offline
07 - 6. Validar a supervisão e construir o par experimental
08 - 7. Treino do aluno: smoke test e piloto pareado
09 - 7. Treino do aluno: smoke test e piloto pareado
10 - 8. Conferir as quatro condições na validação
11 - 8. Conferir as quatro condições na validação
12 - 9. Destilação completa e treinos finais
13 - 9. Destilação completa e treinos finais
14 - 10. Teste completo com os mesmos 1.147 exemplos
15 - 10. Teste completo com os mesmos 1.147 exemplos
16 - 11. Medidas e interpretação
17 - 12. API do aluno funcionando sozinho
18 - 12. API do aluno funcionando sozinho
STEPS
}

etapa_1() {
mkdir -p "$LAB_WORK_DIR"
cd "$LAB_WORK_DIR"
source "$LAB_SHARED_DIR/.venv"/bin/activate
for LAB05_SCRIPT in "$LAB_REPO_DIR"/scripts/lab05-finqa/*.py; do
  if [ ! -e "./${LAB05_SCRIPT##*/}" ]; then
    cp "$LAB05_SCRIPT" ./
  fi
done
cp "$LAB_REPO_DIR"/laboratorio-05-slm-financeiro-finqa.md ./laboratorio.md
cp "$LAB_REPO_DIR"/requirements.txt ./requirements.txt
python -m pip freeze > requirements-resolved.txt
mkdir -p raw logs runs results
set -o pipefail
export TOKENIZERS_PARALLELISM=false
}

etapa_2() {
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
nvidia-smi
df -h .
}

etapa_3() {
python prepare_finqa.py --raw raw/finqa --out data \
  --max-seq 2048 --max-new 512 --top-k 12 \
  2>&1 | tee logs/prepare.log
python inspect_finqa.py --data data
}

etapa_4() {
python evaluate_finqa.py execute \
  --input data/demo_input.json --prediction data/demo_prediction.json \
  2>&1 | tee logs/demo_executor.log
}

etapa_5() {
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_smoke.jsonl --limit 3 \
  2>&1 | tee logs/teacher_smoke.log
python inspect_finqa.py --data data --predictions results/teacher_smoke.jsonl
}

etapa_6() {
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_pilot.jsonl --limit 200 \
  2>&1 | tee logs/teacher_pilot.log
python inspect_finqa.py --data data --predictions results/teacher_pilot.jsonl
}

etapa_7() {
python build_distillation.py --data data \
  --predictions results/teacher_pilot.jsonl --out paired_pilot \
  2>&1 | tee logs/filter_pilot.log
python -m json.tool paired_pilot/selection_report.json
head -n 3 paired_pilot/review.jsonl
}

etapa_8() {
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -- finqa_runtime.py train \
  --data paired_pilot/distilled --run runs/smoke_distilled \
  --max-steps 10 --accum 8 --eval-steps 5 --eval-limit 16 \
  2>&1 | tee logs/smoke_distilled.log
}

etapa_9() {
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_pilot/sft --run runs/pilot_sft \
  --max-steps 40 --accum 8 --rank 16 --lr 0.0001 --eval-steps 10 --eval-limit 128 \
  2>&1 | tee logs/pilot_sft.log
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_pilot/distilled --run runs/pilot_distilled \
  --max-steps 40 --accum 8 --rank 16 --lr 0.0001 --eval-steps 10 --eval-limit 128 \
  2>&1 | tee logs/pilot_distilled.log
}

etapa_10() {
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_base.jsonl
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_sft.jsonl \
  --adapter runs/pilot_sft/final_adapter
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/validation.jsonl --output results/val_distilled.jsonl \
  --adapter runs/pilot_distilled/final_adapter
CUDA_VISIBLE_DEVICES=0,1 python finqa_runtime.py predict --data data \
  --model Qwen/Qwen3-8B --shard --gpu-memory 5GiB \
  --input data/validation.jsonl --output results/val_teacher.jsonl
}

etapa_11() {
python evaluate_finqa.py score data/validation.jsonl results/val_base.jsonl --out results/val_base
python evaluate_finqa.py score data/validation.jsonl results/val_sft.jsonl --out results/val_sft
python evaluate_finqa.py score data/validation.jsonl results/val_distilled.jsonl --out results/val_distilled
python evaluate_finqa.py score data/validation.jsonl results/val_teacher.jsonl --out results/val_teacher
python evaluate_finqa.py compare results/val_sft.cases.jsonl results/val_distilled.cases.jsonl
}

etapa_12() {
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python finqa_runtime.py predict --data data --model Qwen/Qwen3-8B \
  --shard --gpu-memory 5GiB --input data/train_candidates.jsonl \
  --output results/teacher_train.jsonl \
  2>&1 | tee logs/teacher_train.log
python inspect_finqa.py --data data --predictions results/teacher_train.jsonl
python build_distillation.py --data data \
  --predictions results/teacher_train.jsonl --out paired_main \
  2>&1 | tee logs/filter_main.log
python -m json.tool paired_main/selection_report.json
head -n 3 paired_main/review.jsonl
}

etapa_13() {
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_main/sft --run runs/main_sft \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-steps 100 --eval-limit 128 \
  2>&1 | tee logs/main_sft.log
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  python -m torch.distributed.run --standalone --nproc-per-node=2 -- \
  finqa_runtime.py train --data paired_main/distilled --run runs/main_distilled \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-steps 100 --eval-limit 128 \
  2>&1 | tee logs/main_distilled.log
}

etapa_14() {
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_base.jsonl
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_sft.jsonl \
  --adapter runs/main_sft/final_adapter
CUDA_VISIBLE_DEVICES=1 python finqa_runtime.py predict --data data \
  --input data/test.jsonl --output results/test_distilled.jsonl \
  --adapter runs/main_distilled/final_adapter
CUDA_VISIBLE_DEVICES=0,1 python finqa_runtime.py predict --data data \
  --model Qwen/Qwen3-8B --shard --gpu-memory 5GiB \
  --input data/test.jsonl --output results/test_teacher.jsonl
}

etapa_15() {
python evaluate_finqa.py score data/test.jsonl results/test_base.jsonl --out results/base
python evaluate_finqa.py score data/test.jsonl results/test_sft.jsonl --out results/sft
python evaluate_finqa.py score data/test.jsonl results/test_distilled.jsonl --out results/distilled
python evaluate_finqa.py score data/test.jsonl results/test_teacher.jsonl --out results/teacher
python evaluate_finqa.py compare results/base.cases.jsonl results/sft.cases.jsonl
python evaluate_finqa.py compare results/sft.cases.jsonl results/distilled.cases.jsonl
python evaluate_finqa.py compare results/teacher.cases.jsonl results/distilled.cases.jsonl
}

etapa_16() {
cd "$LAB_WORK_DIR"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_phase.csv
}

etapa_17() {
CUDA_VISIBLE_DEVICES=1 python serve_finqa.py --data data \
  --adapter runs/main_distilled/final_adapter --port 8085 \
  2>&1 | tee logs/api_finqa.log
}

etapa_18() {
cd "$LAB_WORK_DIR"
curl -fsS http://127.0.0.1:8085/predict \
  -H 'Content-Type: application/json' --data-binary @data/demo_input.json
}

if [[ $# == 1 && $1 == --list ]]; then list_steps; exit 0; fi
if [[ $# != 2 || $1 != --steps ]]; then
  echo "Uso: bash laboratorio-05-slm-financeiro-finqa.sh --list | --steps 1-4 | --steps 5,6" >&2
  echo "Servidor e monitor ficam em primeiro plano; execute em terminais separados." >&2
  exit 2
fi
LAB_SELECTED=()
IFS=',' read -r -a LAB_PARTS <<< "$2"
for LAB_PART in "${LAB_PARTS[@]}"; do
  if [[ $LAB_PART =~ ^([0-9]+)-([0-9]+)$ ]]; then
    LAB_FIRST=$((10#${BASH_REMATCH[1]})); LAB_LAST=$((10#${BASH_REMATCH[2]}))
  elif [[ $LAB_PART =~ ^[0-9]+$ ]]; then
    LAB_FIRST=$((10#$LAB_PART)); LAB_LAST=$LAB_FIRST
  else echo "Etapa inválida: $LAB_PART" >&2; exit 2; fi
  if (( LAB_FIRST < 1 || LAB_LAST > 18 || LAB_FIRST > LAB_LAST )); then echo "Intervalo inválido" >&2; exit 2; fi
  for ((LAB_STEP=LAB_FIRST;LAB_STEP<=LAB_LAST;LAB_STEP++)); do LAB_SELECTED+=("$LAB_STEP"); done
done
mkdir -p "$LAB_WORK_DIR/logs"
cd "$LAB_WORK_DIR"
LAB_LOG_FILE="$LAB_WORK_DIR/logs/laboratorio-05-slm-financeiro-finqa-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee -a "$LAB_LOG_FILE") 2>&1
cleanup() { local lab_status=$?; trap - EXIT; echo "Fim: código $lab_status. Log: $LAB_LOG_FILE"; exit "$lab_status"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$LAB_SHARED_DIR/.venv/bin/activate"
export TOKENIZERS_PARALLELISM=false
for LAB_STEP in "${LAB_SELECTED[@]}"; do
  echo "Etapa $LAB_STEP / log: $LAB_LOG_FILE"
  "etapa_$LAB_STEP"
done
