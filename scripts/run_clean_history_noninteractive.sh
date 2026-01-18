#!/usr/bin/env bash
set -euo pipefail

# Non-interactive wrapper for scripts/clean_history_with_git_filter_repo.sh
# Usage:
#   bash scripts/run_clean_history_noninteractive.sh --tool=filter-repo --yes --remove-files="path1,path2"
#
# This script calls the interactive helper in a non-interactive way by
# preparing the environment and invoking git-filter-repo or BFG with the
# provided file list. It does NOT push to remotes.

TOOL="filter-repo"
CONFIRM="no"
REMOVE_FILES=""

for arg in "$@"; do
  case $arg in
    --tool=*) TOOL="${arg#*=}" ;;
    --yes) CONFIRM="yes" ;;
    --remove-files=*) REMOVE_FILES="${arg#*=}" ;;
    --help) echo "Usage: $0 --tool=(filter-repo|bfg) --yes --remove-files='file1,file2'"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

if [ "$CONFIRM" != "yes" ]; then
  echo "Pass --yes to run non-interactively." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Please commit or stash them first." >&2
  git status --porcelain
  exit 1
fi

ROOT_DIR=$(pwd)
BACKUP_DIR="${ROOT_DIR}/../codex_quant_mirror_backup.git"

echo "Creating mirror backup at: ${BACKUP_DIR}"
git clone --mirror "${ROOT_DIR}" "${BACKUP_DIR}"

IFS=',' read -r -a FILE_ARRAY <<< "$REMOVE_FILES"

if [ "$TOOL" = "filter-repo" ]; then
  if ! command -v git-filter-repo >/dev/null 2>&1; then
    echo "git-filter-repo not found. Install: https://github.com/newren/git-filter-repo" >&2
    exit 3
  fi

  PATH_ARGS=()
  for f in "${FILE_ARRAY[@]}"; do
    PATH_ARGS+=(--paths "$f")
  done

  echo "Running git-filter-repo --invert-paths --strip-blobs-bigger-than 50M"
  git filter-repo "${PATH_ARGS[@]}" --invert-paths --strip-blobs-bigger-than 50M

elif [ "$TOOL" = "bfg" ]; then
  if ! command -v bfg >/dev/null 2>&1; then
    echo "BFG not found. Install: https://rtyley.github.io/bfg-repo-cleaner/" >&2
    exit 3
  fi

  TMP_BARE="${ROOT_DIR}/../codex_quant_bare.git"
  git clone --mirror "${ROOT_DIR}" "${TMP_BARE}"
  cd "${TMP_BARE}"

  # Build a comma-separated list for BFG patterns
  PATTERNS=""
  for f in "${FILE_ARRAY[@]}"; do
    if [ -n "$PATTERNS" ]; then PATTERNS=","$PATTERNS; fi
    PATTERNS="$PATTERNS$f"
  done

  echo "Running BFG to strip blobs bigger than 100M and remove specified files"
  bfg --strip-blobs-bigger-than 100M --delete-files "$PATTERNS" .
  git reflog expire --expire=now --all && git gc --prune=now --aggressive

else
  echo "Unknown tool: $TOOL" >&2
  exit 2
fi

echo "Cleanup complete locally. Inspect the repo; push is a manual step."
echo "Suggested checks:
  git count-objects -vH
  git log --stat --summary | head -n 200
  git verify-pack -v .git/objects/pack/pack-*.idx | sort -k3 -n | tail -n 20
"

exit 0
