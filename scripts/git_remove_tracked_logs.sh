#!/usr/bin/env bash
set -euo pipefail

# git_remove_tracked_logs.sh
# - Identifies git-tracked files under `logs/` that do NOT contain sensitive patterns
# - By default prints the git rm commands (dry-run). Use --run to execute.

SENSITIVE='PMAKER_LOAD|PMAKER_AUTO|BATCH_PIPE|STATE_LOAD|paper_balance|paper_equity|ORDERBOOK|HEARTBEAT'
RUN=0

if [[ "${1:-}" == "--run" ]]; then
  RUN=1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "No logs directory found at '$LOG_DIR'. Exiting." >&2
  exit 0
fi

echo "Finding git-tracked log files..."
# Use portable read loop to populate TRACKED (avoid mapfile for macOS bash)
TRACKED=()
while IFS= read -r line; do
  TRACKED+=("$line")
done <<< "$(git -C "$PROJECT_ROOT" ls-files "$LOG_DIR" 2>/dev/null || true)"

if [[ ${#TRACKED[@]} -eq 0 ]]; then
  echo "No tracked files under logs/. Nothing to remove via git." 
  exit 0
fi

echo "Total tracked files under logs/: ${#TRACKED[@]}"

TO_REMOVE=()
for f in "${TRACKED[@]}"; do
  if ! grep -E -q "$SENSITIVE" "$PROJECT_ROOT/$f" 2>/dev/null; then
    TO_REMOVE+=("$f")
  fi
done

if [[ ${#TO_REMOVE[@]} -eq 0 ]]; then
  echo "No tracked log files found that are free of sensitive patterns. Nothing to remove."
  exit 0
fi

echo "The following tracked files are candidates for git removal (archived first recommended):"
for f in "${TO_REMOVE[@]}"; do
  echo "  $f"
done

if [[ $RUN -ne 1 ]]; then
  echo
  echo "Dry-run. To remove these files from git run:"
  echo "  bash $0 --run"
  echo "Or to remove but keep on disk: use 'git rm --cached <file>' for each file instead of 'git rm'"
  exit 0
fi

echo "Removing files from git and committing..."
git -C "$PROJECT_ROOT" rm -v "${TO_REMOVE[@]}"
git -C "$PROJECT_ROOT" commit -m "Remove archived non-sensitive logs from git"

echo "Done. Remember to push the commit: git push"

exit 0
