#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO_DIR="$(cd -- "$AVALIADOR_DIR/.." && pwd)"
LAB05_DIR="${LAB05_DIR:-$HOME/slm-labs/05-finqa}"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
LAB05_ADAPTER="${LAB05_ADAPTER:-$LAB05_DIR/runs/main_distilled/final_adapter}"
LAB05_OPTIONS=()
if [[ ${LAB05_ENABLE_GPU:-0} == 1 ]]; then
  [[ -f "$LAB05_ADAPTER/adapter_config.json" ]] || { echo 'Adaptador ausente.' >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${LAB05_GPU:-0}"
  LAB05_OPTIONS+=(--enable-gpu)
else export CUDA_VISIBLE_DEVICES=""; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$AVALIADOR_DIR/logs"
LAB05_LOG="$AVALIADOR_DIR/logs/finqa-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "API FinQA / GPU habilitada: ${LAB05_ENABLE_GPU:-0} / log: $LAB05_LOG"
cd "$LAB05_DIR"
exec "$LAB_VENV_DIR/bin/python" -u "$AVALIADOR_DIR/gpu_worker.py" --lab 05 \
  --data data --results results --adapter "$LAB05_ADAPTER" --port "${LAB05_API_PORT:-8085}" \
  "${LAB05_OPTIONS[@]}" > >(tee "$LAB05_LOG") 2>&1
