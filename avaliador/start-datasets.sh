#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$HOME/slm-labs/.dataset-explorer}"
mkdir -p "$AVALIADOR_DIR/logs"
DATASET_LOG="$AVALIADOR_DIR/logs/datasets-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Explorando datasets locais, sem carregar modelos. Log: $DATASET_LOG"
export CUDA_VISIBLE_DEVICES=""
exec "$LAB_VENV_DIR/bin/python" -u "$AVALIADOR_DIR/datasets/catalog_service.py" \
  --cache "$DATASET_CACHE_DIR" --port "${DATASET_API_PORT:-8086}" > >(tee "$DATASET_LOG") 2>&1
