#!/usr/bin/env bash
set -euo pipefail

# archive_non_sensitive_logs.sh
# - Scans `logs/` for files that do NOT contain sensitive patterns
# - Moves them to an archive dir, compresses the archive, produces a sha256
# - By default does a dry-run; pass --run to perform actions

SENSITIVE='PMAKER_LOAD|PMAKER_AUTO|BATCH_PIPE|STATE_LOAD|paper_balance|paper_equity|ORDERBOOK|HEARTBEAT'
RUN=0

if [[ "${1:-}" == "--run" ]]; then
  RUN=1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
ARCHIVE_ROOT="${HOME}/codex_quant_archives"
ARCHIVE_SUBDIR="logs_$(date +%F_%H%M%S)"

echo "Project root: $PROJECT_ROOT"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "No logs directory found at '$LOG_DIR'. Exiting." >&2
  exit 0
fi

echo "Scanning '$LOG_DIR' for non-sensitive log files..."

# Build list of candidate files (those that do NOT match sensitive patterns)
# Use portable read loop instead of `mapfile` (macOS bash lacks mapfile)
CANDIDATES=()
while IFS= read -r line; do
  CANDIDATES+=("$line")
done <<< "$(grep -E -L "$SENSITIVE" "$LOG_DIR"/* 2>/dev/null || true)"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "No non-sensitive files found. Nothing to archive."
  exit 0
fi

echo "Found ${#CANDIDATES[@]} candidate files:" 
for f in "${CANDIDATES[@]}"; do
  echo "  - $f"
done

echo
if [[ $RUN -ne 1 ]]; then
  echo "Dry-run mode. To perform archive + removal run:"
  echo "  bash $0 --run"
  exit 0
fi

mkdir -p "$ARCHIVE_ROOT/$ARCHIVE_SUBDIR"

echo "Moving candidates to archive dir: $ARCHIVE_ROOT/$ARCHIVE_SUBDIR"
for f in "${CANDIDATES[@]}"; do
  mv -v "$f" "$ARCHIVE_ROOT/$ARCHIVE_SUBDIR/"
done

cd "$ARCHIVE_ROOT"
ARCHIVE_NAME="logs_archive_${ARCHIVE_SUBDIR}.tar.gz"
tar czf "$ARCHIVE_NAME" "$ARCHIVE_SUBDIR"
shasum -a 256 "$ARCHIVE_NAME" > "$ARCHIVE_NAME".sha256

echo "Archive created: $ARCHIVE_ROOT/$ARCHIVE_NAME"
echo "Checksum: $ARCHIVE_ROOT/$ARCHIVE_NAME.sha256"

echo "Archive step complete. You can now delete the original moved files from the archive dir if you want to free space locally."
echo "If you want to remove the archive directory contents after verifying the tarball, run:\n  rm -rf \"$ARCHIVE_ROOT/$ARCHIVE_SUBDIR\""

exit 0
