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
HEALTH_TIMEOUT_SEC="${ENGINE_HEALTH_TIMEOUT_SEC:-90}"

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
  find_pids | tail -n 1
}

find_pids() {
  local entry_base
  entry_base="$(basename "$ENTRY_FILE")"
  {
    pgrep -f "Python.*${ENTRY_FILE}" || true
    pgrep -f "python.*${ENTRY_FILE}" || true
    pgrep -f "Python.*${entry_base}" || true
    pgrep -f "python.*${entry_base}" || true
  } | awk '!seen[$0]++' | sort -n
}

collect_descendants() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    collect_descendants "$child"
  done
  echo "$pid"
}

kill_pid_tree() {
  local pid="$1"
  local sig="${2:-TERM}"
  [ -n "$pid" ] || return 0

  local target
  for target in $(collect_descendants "$pid" | awk '!seen[$0]++'); do
    kill "-${sig}" "$target" 2>/dev/null || true
  done
}

find_orphan_worker_pids() {
  local pid cwd
  ps -axo pid,ppid,command | awk '$2 == 1 && ($0 ~ /Python\.app\/Contents\/MacOS\/Python -c from multiprocessing\.spawn import spawn_main/ || $0 ~ /Python\.app\/Contents\/MacOS\/Python -c from multiprocessing\.resource_tracker import main;main/) { print $1 }' | while read -r pid; do
    [ -n "${pid:-}" ] || continue
    cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | awk '/^n/{print substr($0,2)}' | head -n 1)"
    if [ "$cwd" = "$ROOT_DIR" ]; then
      echo "$pid"
    fi
  done | awk '!seen[$0]++' | sort -n
}

read_lock_owner_pid() {
  local lock_file="$ROOT_DIR/state/locks/trading_engine.lock"
  [ -f "$lock_file" ] || return 0
  python3 - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$lock_file")
try:
    obj = json.loads(p.read_text(encoding="utf-8"))
    pid = int(obj.get("pid")) if isinstance(obj, dict) and obj.get("pid") is not None else None
    if pid:
        print(pid)
except Exception:
    pass
PY
}

write_pid_file() {
  local pid
  pid="$(find_pid)"
  if [ -n "$pid" ]; then
    echo "$pid" > "$PID_FILE"
  fi
}

stop_engine() {
  local entry_base
  entry_base="$(basename "$ENTRY_FILE")"
  local pid
  if screen -ls 2>/dev/null | grep -q "[[:space:]]${SESSION_NAME}[[:space:]]"; then
    screen -S "$SESSION_NAME" -X quit || true
  fi

  for pid in $(find_pids); do
    kill_pid_tree "$pid" TERM
  done
  sleep 1
  for pid in $(find_pids); do
    kill_pid_tree "$pid" KILL
  done

  local owner_pid
  owner_pid="$(read_lock_owner_pid)"
  if [ -n "${owner_pid:-}" ] && ps -p "$owner_pid" >/dev/null 2>&1; then
    kill_pid_tree "$owner_pid" TERM
    sleep 0.5
    if ps -p "$owner_pid" >/dev/null 2>&1; then
      kill_pid_tree "$owner_pid" KILL
    fi
  fi

  local orphan_pid
  for orphan_pid in $(find_orphan_worker_pids); do
    kill -TERM "$orphan_pid" 2>/dev/null || true
  done
  sleep 0.5
  for orphan_pid in $(find_orphan_worker_pids); do
    kill -KILL "$orphan_pid" 2>/dev/null || true
  done

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
  local attempts=0
  local sleep_sec=0.5
  attempts="$(python3 - <<PY
import math
print(max(1, int(math.ceil(float(${HEALTH_TIMEOUT_SEC}) / ${sleep_sec}))))
PY
)"
  for _ in $(seq 1 "${attempts}"); do
    local code
    code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 "http://127.0.0.1:${PORT}/" || true)"
    if [ -n "$code" ] && [ "$code" != "000" ]; then
      ok=1
      break
    fi
    sleep "${sleep_sec}"
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
