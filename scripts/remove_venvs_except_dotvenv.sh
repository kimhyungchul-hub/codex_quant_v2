#!/usr/bin/env bash
set -euo pipefail

# remove_venvs_except_dotvenv.sh
# - Finds venv-like directories under the repo by looking for
#   - 'bin/activate' files, or
#   - 'pyvenv.cfg' files, or
#   - executable 'bin/python*' files
# - Excludes the `.venv` directory
# - Dry-run by default; pass --run to archive (tar.gz + sha256) and remove each candidate
# - This script is destructive when run with --run. Use with care.

RUN=0
if [[ "${1:-}" == "--run" ]]; then
  RUN=1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXCLUDE_DIR="$ROOT/.venv"
ARCHIVE_ROOT="$ROOT/archive/venvs_cleanup_$(date +%F_%H%M%S)"

echo "Repository root: $ROOT"
echo "Excluding: $EXCLUDE_DIR"

TMPLIST=$(mktemp)
trap 'rm -f "$TMPLIST"' EXIT

# Find activate files
find . -maxdepth 4 -type f -path '*/bin/activate' -print0 2>/dev/null | \
  while IFS= read -r -d '' f; do
    vdir="$(cd "$(dirname "$f")/.." && pwd)"
    echo "$vdir" >> "$TMPLIST"
  done

# Find pyvenv.cfg
find . -maxdepth 4 -type f -name 'pyvenv.cfg' -print0 2>/dev/null | \
  while IFS= read -r -d '' f; do
    vdir="$(cd "$(dirname "$f")" && pwd)"
    echo "$vdir" >> "$TMPLIST"
  done

# Find executable python binaries under */bin/python*
find . -maxdepth 4 -type f -path '*/bin/python*' -perm -111 -print0 2>/dev/null | \
  while IFS= read -r -d '' f; do
    vdir="$(cd "$(dirname "$f")/.." && pwd)"
    echo "$vdir" >> "$TMPLIST"
  done

# Deduplicate and filter
sort -u "$TMPLIST" -o "$TMPLIST" || true

CANDIDATES=()
while IFS= read -r line; do
  # Normalize
  dir="$(cd "$line" 2>/dev/null && pwd || echo "$line")"
  # Skip if empty or root
  [[ -z "$dir" || "$dir" == "." ]] && continue
  # Skip the excluded .venv
  if [[ "$dir" == "$EXCLUDE_DIR" ]]; then
    continue
  fi
  # Only include directories that actually exist
  if [[ -d "$dir" ]]; then
    CANDIDATES+=("$dir")
  fi
done < "$TMPLIST"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "No venv-like directories found (excluding .venv). Nothing to do."
  exit 0
fi

echo "Found ${#CANDIDATES[@]} candidate venv directories (excluding .venv):"
for d in "${CANDIDATES[@]}"; do
  echo "  - $d"
done

if [[ $RUN -ne 1 ]]; then
  echo
  echo "Dry-run mode. To archive and remove these directories run:" 
  echo "  bash $0 --run"
  exit 0
fi

mkdir -p "$ARCHIVE_ROOT"
echo "Archiving to: $ARCHIVE_ROOT"

for d in "${CANDIDATES[@]}"; do
  name="$(basename "$d")"
  TAR_NAME="$ARCHIVE_ROOT/${name}.tar.gz"
  echo "Archiving $d -> $TAR_NAME"
  # create tar.gz from parent dir to preserve folder name
  tar -C "$(dirname "$d")" -czf "$TAR_NAME" "$(basename "$d")"
  shasum -a 256 "$TAR_NAME" > "$TAR_NAME".sha256
  echo "Removing directory $d"
  rm -rf "$d"
done

echo "All candidates archived and removed. Archive directory: $ARCHIVE_ROOT"
echo "Verify the tar.gz files and their .sha256 before permanently deleting backups."

exit 0
