#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$AVALIADOR_DIR/logs"
export LAB01_ENABLE_GPU=1 LAB03_ENABLE_GPU=1 LAB05_ENABLE_GPU=1
export LAB01_GPU="${LAB01_GPU:-0}" LAB02_GPU="${LAB02_GPU:-1}" LAB03_GPU="${LAB03_GPU:-0}" LAB04_GPU="${LAB04_GPU:-0}" LAB05_GPU="${LAB05_GPU:-1}"
for LAB_NUMBER in 01 02 03 04 05; do
  LAB_PORT_VAR="LAB${LAB_NUMBER}_API_PORT"
  LAB_DEFAULT_PORT=$((8080 + 10#$LAB_NUMBER))
  LAB_PORT="${!LAB_PORT_VAR:-$LAB_DEFAULT_PORT}"
  if LAB_HEALTH="$(curl --max-time 2 -fsS "http://127.0.0.1:$LAB_PORT/health" 2>/dev/null)"; then
    if [[ "$LAB_HEALTH" == *'"live_enabled": false'* ]]; then
      echo "Lab $LAB_NUMBER está em modo CPU. Encerre essa API antes de habilitar todas as inferências." >&2
      exit 1
    fi
    echo "Lab $LAB_NUMBER já ativo na porta $LAB_PORT."
    continue
  fi
  LAB_START_LOG="$AVALIADOR_DIR/logs/start-lab${LAB_NUMBER}-$(date +%Y%m%d-%H%M%S).log"
  nohup bash "$AVALIADOR_DIR/start-lab${LAB_NUMBER}.sh" > "$LAB_START_LOG" 2>&1 </dev/null &
  LAB_START_PID=$!
  echo "$LAB_START_PID" > "$AVALIADOR_DIR/logs/lab${LAB_NUMBER}.pid"
  echo "Iniciando Lab $LAB_NUMBER / PID $LAB_START_PID / porta $LAB_PORT / $LAB_START_LOG"
done
printf 'Confira /api/lab01/status, /api/lab02/status, /api/lab03/status, /api/status e /api/lab05/status no avaliador.\n'
