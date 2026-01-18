#!/usr/bin/env bash
# scripts/cleanup_root_files.sh
# 안전한 루트 정리 스크립트
# - 기본: dry-run (무엇을 할지 출력)
# - 실제 실행: ./scripts/cleanup_root_files.sh --run
# 작성자: 자동 생성

set -euo pipefail
ARCHIVE_DIR="$HOME/codex_quant_archives/roots_$(date +%F)"
DRY_RUN=true

if [[ "${1-}" == "--run" ]]; then
  DRY_RUN=false
fi

echo "Archive dir: $ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

# 안전 보존(보관해서 레포에서 제거 권장)
ARCHIVE_AND_REMOVE=(
  ".venv311"
  "engine.log"
  "engine.pid"
  "nohup.out"
  "logs"
  "formatted_payload.json"
  "formatted_payload_3.json"
  "python main.py"
  ".pytest_cache"
  "__pycache__"
  "test_crash.log"
)

# 즉시 삭제 가능한 캐시/임시
DELETE_IMMEDIATE=(
  ".DS_Store"
  ".continue"
  ".cursor"
  ".aider.chat.history.md"
  ".aider.input.history"
)

# 보존 (삭제하지 않음)
KEEP=(
  "main.py"
  "config.py"
  "README.md"
  "README_DEV_SETUP.md"
  "requirements.txt"
  "requirements-dev.txt"
  "requirements-lock.txt"
  "core"
  "engines"
  "state"
  "scripts"
  "benchmarks"
  "docs"
  "models"
  "trainers"
  "patches"
  "tests"
  "dashboard_v2.html"
)

echo
echo "=== Dry run: planned actions ==="
for f in "${ARCHIVE_AND_REMOVE[@]}"; do
  if [[ -e "$PWD/$f" ]]; then
    echo "[ARCHIVE->REMOVE] $f -> $ARCHIVE_DIR/"
  else
    echo "[MISSING] $f"
  fi
done
for f in "${DELETE_IMMEDIATE[@]}"; do
  if [[ -e "$PWD/$f" ]]; then
    echo "[DELETE] $f"
  fi
done

if $DRY_RUN; then
  echo
  echo "DRY RUN complete. To perform actions, run: ./scripts/cleanup_root_files.sh --run"
  exit 0
fi

# Confirm
read -p "Proceed with archive+remove? (y/N): " confirm
if [[ "$confirm" != "y" ]]; then
  echo "Aborting. No changes made."
  exit 1
fi

# Perform archive + move
echo "Starting archive and remove..."
for f in "${ARCHIVE_AND_REMOVE[@]}"; do
  if [[ -e "$PWD/$f" ]]; then
    echo "Archiving $f"
    # preserve path inside archive dir
    target="$ARCHIVE_DIR/$f"
    mkdir -p "$(dirname "$target")"
    mv "$PWD/$f" "$target"
  fi
done

# compress archived files where appropriate
echo "Compressing archived logs and large files..."
find "$ARCHIVE_DIR" -type f -name "*.log" -o -name "nohup.out" -o -name "*.json" | while read -r f; do
  if [[ -f "$f" ]]; then
    gzip -9 "$f" || true
  fi
done

# delete immediate files
for f in "${DELETE_IMMEDIATE[@]}"; do
  if [[ -e "$PWD/$f" ]]; then
    echo "Deleting $f"
    rm -rf "$PWD/$f"
  fi
done

# Update .gitignore (append if missing)
GITIGNORE=.gitignore
echo "Ensuring .venv311/ is in $GITIGNORE"
grep -qxF ".venv311/" "$GITIGNORE" || echo ".venv311/" >> "$GITIGNORE"

echo "Cleanup complete. Archived items are in: $ARCHIVE_DIR"

echo "Next recommended steps:"
echo "  - Commit .gitignore changes: git add .gitignore && git commit -m 'Ignore local venv and large runtime files'"
echo "  - If sensitive files were committed (e.g. env files), consider removing from git history using git filter-repo or BFG."

exit 0
