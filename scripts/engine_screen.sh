#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${ENGINE_SCREEN_SESSION:-mc_engine_bybit}"
ENV_FILE="${ENGINE_ENV_FILE:-$ROOT_DIR/state/bybit.env}"
ENTRY_FILE="${ENGINE_ENTRY_FILE:-$ROOT_DIR/main_engine_mc_v2_final.py}"
PY_BIN="${ENGINE_PY_BIN:-$ROOT_DIR/.venv/bin/python}"
PID_FILE="${ENGINE_PID_FILE:-$ROOT_DIR/engine.pid}"
LOG_FILE="${ENGINE_LOG_FILE:-$ROOT_DIR/state/codex_engine.log}"
PORT="${ENGINE_PORT:-9999}"

usage() {
  cat <<'EOF'
Usage: scripts/engine_screen.sh {start|stop|restart|status|logs}

Environment overrides:
  ENGINE_SCREEN_SESSION  default: mc_engine_bybit
  ENGINE_ENV_FILE        default: state/bybit.env
  ENGINE_ENTRY_FILE      default: main_engine_mc_v2_final.py
  ENGINE_PY_BIN          default: .venv/bin/python
  ENGINE_PID_FILE        default: engine.pid
  ENGINE_LOG_FILE        default: state/codex_engine.log
  ENGINE_PORT            default: 9999
EOF
}

require_prereqs() {
  command -v screen >/dev/null 2>&1 || { echo "[ERR] screen not found"; exit 1; }
  [ -f "$ENV_FILE" ] || { echo "[ERR] missing env file: $ENV_FILE"; exit 1; }
  [ -f "$ENTRY_FILE" ] || { echo "[ERR] missing entry file: $ENTRY_FILE"; exit 1; }
  [ -x "$PY_BIN" ] || { echo "[ERR] missing python bin: $PY_BIN"; exit 1; }
}

find_pid() {
  pgrep -f "Python.*${ENTRY_FILE}" | head -n 1 || true
}

write_pid_file() {
  local pid
  pid="$(find_pid)"
  if [ -n "$pid" ]; then
    echo "$pid" > "$PID_FILE"
  fi
}

stop_engine() {
  if screen -ls 2>/dev/null | grep -q "[[:space:]]${SESSION_NAME}[[:space:]]"; then
    screen -S "$SESSION_NAME" -X quit || true
  fi
  pkill -f "$ENTRY_FILE" 2>/dev/null || true
  rm -f "$PID_FILE"
}

start_engine() {
  mkdir -p "$(dirname "$LOG_FILE")"
  touch "$LOG_FILE"

  stop_engine
  sleep 1

  local cmd
  cmd="cd \"$ROOT_DIR\" && set -a && source \"$ENV_FILE\" && set +a && exec \"$PY_BIN\" -u \"$ENTRY_FILE\" >> \"$LOG_FILE\" 2>&1"
  screen -dmS "$SESSION_NAME" /bin/bash -lc "$cmd"

  local ok=0
  for _ in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/"; then
      ok=1
      break
    fi
    sleep 0.5
  done

  write_pid_file
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"

  if [ "$ok" -eq 1 ] && [ -n "$pid" ]; then
    echo "[OK] engine started pid=${pid} session=${SESSION_NAME} port=${PORT}"
    return 0
  fi

  echo "[ERR] engine failed health-check on port ${PORT}"
  echo "[INFO] tail log: ${LOG_FILE}"
  tail -n 80 "$LOG_FILE" || true
  return 1
}

status_engine() {
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  local cmd=""
  if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  fi
  if [ -z "$pid" ] || [ -z "$cmd" ] || [[ "$cmd" != *"Python"* ]] || [[ "$cmd" != *"$ENTRY_FILE"* ]]; then
    write_pid_file
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi
  if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
    ps -p "$pid" -o pid,ppid,stat,etime,command
  else
    echo "[INFO] engine.pid missing or stale"
  fi
  echo "[INFO] screen sessions:"
  screen -ls || true
  echo "[INFO] port ${PORT}:"
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || true
  echo "[INFO] dashboard http:"
  curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:${PORT}/" || true
}

logs_engine() {
  tail -n 120 "$LOG_FILE"
}

main() {
  require_prereqs
  local action="${1:-}"
  case "$action" in
    start) start_engine ;;
    stop) stop_engine ;;
    restart) start_engine ;;
    status) status_engine ;;
    logs) logs_engine ;;
    *) usage; exit 1 ;;
  esac
}

main "${1:-}"
