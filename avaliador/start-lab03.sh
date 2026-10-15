#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO_DIR="$(cd -- "$AVALIADOR_DIR/.." && pwd)"
LAB03_DIR="${LAB03_DIR:-$HOME/lab03-recibos}"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
LAB03_ADAPTER="${LAB03_ADAPTER:-$LAB03_DIR/runs/main/final_adapter}"
LAB03_OPTIONS=()
if [[ ${LAB03_ENABLE_GPU:-0} == 1 ]]; then
  [[ -f "$LAB03_ADAPTER/adapter_config.json" ]] || { echo 'Adaptador ausente.' >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${LAB03_GPU:-0}"
  LAB03_OPTIONS+=(--enable-gpu)
else export CUDA_VISIBLE_DEVICES=""; fi
export TOKENIZERS_PARALLELISM=false
mkdir -p "$AVALIADOR_DIR/logs"
LAB03_LOG="$AVALIADOR_DIR/logs/receipts-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "API de recibos / GPU habilitada: ${LAB03_ENABLE_GPU:-0} / log: $LAB03_LOG"
cd "$LAB03_DIR"
exec "$LAB_VENV_DIR/bin/python" -u "$AVALIADOR_DIR/gpu_worker.py" --lab 03 \
  --data data --results results --adapter "$LAB03_ADAPTER" --port "${LAB03_API_PORT:-8083}" \
  "${LAB03_OPTIONS[@]}" > >(tee "$LAB03_LOG") 2>&1
