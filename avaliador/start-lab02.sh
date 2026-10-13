#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO_DIR="$(cd -- "$AVALIADOR_DIR/.." && pwd)"
LAB02_DIR="${LAB02_DIR:-$HOME/slm-labs/02-sql}"
LAB_VENV_DIR="${LAB_VENV_DIR:-$HOME/slm-labs/.venv}"
LAB02_ADAPTER="${LAB02_ADAPTER:-$LAB02_DIR/runs/main_r16_lr1e4/final_adapter}"
for LAB_FILE in "$LAB_VENV_DIR/bin/python" "$LAB02_DIR/revisions.json" "$LAB02_DIR/data/test.jsonl" "$LAB02_ADAPTER/adapter_config.json"; do
  if [[ ! -f "$LAB_FILE" ]]; then echo "Arquivo necessário ausente: $LAB_FILE" >&2; exit 1; fi
done
mkdir -p "$AVALIADOR_DIR/logs"
LAB02_LOG="$AVALIADOR_DIR/logs/sql-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Carregando o adaptador SQL: $LAB02_ADAPTER"
echo "Log: $LAB02_LOG"
export CUDA_VISIBLE_DEVICES="${LAB02_GPU:-1}"
export TOKENIZERS_PARALLELISM=false
cd "$LAB02_DIR"
exec "$LAB_VENV_DIR/bin/python" -u "$LAB_REPO_DIR/scripts/lab02-sql/serve_sql.py" \
  --data "$LAB02_DIR/data" --adapter "$LAB02_ADAPTER" --results "$LAB02_DIR/results" \
  --port "${LAB02_API_PORT:-8082}" > >(tee "$LAB02_LOG") 2>&1
