#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO_DIR="$(cd -- "$AVALIADOR_DIR/.." && pwd)"
LAB04_DIR="${LAB04_DIR:-$HOME/lab04-decisoes}"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
for LAB_FILE in "$LAB_VENV_DIR/bin/python" "$LAB04_DIR/data/bundle.json" "$LAB04_DIR/runs/main/final_adapter/adapter_config.json" "$LAB04_DIR/runs/decision_policy.json"; do
  if [[ ! -f "$LAB_FILE" ]]; then
    echo "Arquivo necessário ausente: $LAB_FILE" >&2
    exit 1
  fi
done
mkdir -p "$AVALIADOR_DIR/logs"
LAB04_LOG="$AVALIADOR_DIR/logs/python-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Carregando o adaptador real de $LAB04_DIR/runs/main/final_adapter"
echo "Log: $LAB04_LOG"
export CUDA_VISIBLE_DEVICES="${LAB04_GPU:-0}"
export TOKENIZERS_PARALLELISM=false
exec "$LAB_VENV_DIR/bin/python" -u "$LAB_REPO_DIR/scripts/lab04-decisoes/serve_decisions.py" \
  --data "$LAB04_DIR/data" --adapter "$LAB04_DIR/runs/main/final_adapter" \
  --calibration "$LAB04_DIR/runs/decision_policy.json" --port "${LAB04_API_PORT:-8084}" \
  > >(tee "$LAB04_LOG") 2>&1
