#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO_DIR="$(cd -- "$AVALIADOR_DIR/.." && pwd)"
LAB01_DIR="${LAB01_DIR:-$HOME/slm-labs/01-tools}"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
LAB01_ADAPTER="${LAB01_ADAPTER:-$LAB01_DIR/runs/main_r16_lr1e4/final_adapter}"
LAB01_OPTIONS=()
if [[ ${LAB01_ENABLE_GPU:-0} == 1 ]]; then
  [[ -f "$LAB01_ADAPTER/adapter_config.json" ]] || { echo 'Adaptador ausente.' >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${LAB01_GPU:-0}"
  LAB01_OPTIONS+=(--enable-gpu)
else
  export CUDA_VISIBLE_DEVICES=""
fi
mkdir -p "$AVALIADOR_DIR/logs"
LAB01_LOG="$AVALIADOR_DIR/logs/tools-$(date +%Y%m%d-%H%M%S)-$$.log"
export TOKENIZERS_PARALLELISM=false
cd "$LAB01_DIR"
echo "API de ferramentas / GPU habilitada: ${LAB01_ENABLE_GPU:-0} / log: $LAB01_LOG"
exec "$LAB_VENV_DIR/bin/python" -u "$LAB_REPO_DIR/scripts/lab01-ferramentas/serve_tools.py" \
  --data data --results results --adapter "$LAB01_ADAPTER" --port "${LAB01_API_PORT:-8081}" \
  "${LAB01_OPTIONS[@]}" > >(tee "$LAB01_LOG") 2>&1
