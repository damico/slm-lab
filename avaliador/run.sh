#!/usr/bin/env bash
set -euo pipefail
AVALIADOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$AVALIADOR_DIR"
mkdir -p logs
AVALIADOR_LOG="$AVALIADOR_DIR/logs/web-$(date +%Y%m%d-%H%M%S)-$$.log"
if ! command -v mvn >/dev/null || ! command -v java >/dev/null; then
  echo 'Instale Java 17 ou superior e Maven, conforme README.md.' >&2
  exit 1
fi
mvn -B -ntp -DskipTests package 2>&1 | tee "$AVALIADOR_LOG"
echo "Avaliador: http://localhost:${AVALIADOR_PORT:-8080}"
echo "Log: $AVALIADOR_LOG"
exec java -jar target/avaliador-1.0.0.jar > >(tee -a "$AVALIADOR_LOG") 2>&1
