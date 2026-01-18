#!/usr/bin/env bash
set -euo pipefail

# clean_history_with_git_filter_repo.sh
#
# Safe helper to remove large files from git history using git-filter-repo
# or BFG. This script does not push changes automatically by default — it
# creates a mirror backup and runs the filter. Read the file before running.

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install git and retry." >&2
  exit 2
fi

echo "This script will rewrite the repository history in-place."
echo "It will first create a mirror backup in ../codex_quant_mirror_backup.git"
echo
read -p "Continue? (type 'yes' to proceed): " confirm
if [ "$confirm" != "yes" ]; then
  echo "Aborting.";
  exit 0
fi

# 1) Ensure working tree is clean
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree has uncommitted changes. Please commit or stash them first." >&2
  git status --porcelain
  exit 1
fi

ROOT_DIR=$(pwd)
BACKUP_DIR="${ROOT_DIR}/../codex_quant_mirror_backup.git"

echo "Creating a local mirror backup at: ${BACKUP_DIR}"
git clone --mirror "${ROOT_DIR}" "${BACKUP_DIR}"

echo "Backup created. Next: choose the tool to use for history cleanup."
echo "  1) git-filter-repo (recommended)
  2) BFG Repo-Cleaner (alternative)
  "
read -p "Select tool [1/2]: " tool

if [ "$tool" = "1" ]; then
  if ! command -v git-filter-repo >/dev/null 2>&1; then
    echo "git-filter-repo not found. Install it first. See: https://github.com/newren/git-filter-repo" >&2
    exit 3
  fi

  # Suggested list of explicit paths to remove from history (from prior push errors)
  # You can edit this list before running the script.
  FILES_TO_REMOVE=(
    "engine_final.log"
    "state/engine_run.log"
    "engine_ev_fixed.log"
    "engine_stdout_final.log.old"
  )

  # Build --paths args
  PATH_ARGS=()
  for f in "${FILES_TO_REMOVE[@]}"; do
    PATH_ARGS+=(--paths "${f}")
  done

  echo "Running git-filter-repo to remove specific files and strip blobs >50M"
  echo "(This will rewrite all refs in the current repository)."

  # Remove the listed paths and strip any blobs larger than 50M (tune as needed)
  git filter-repo "${PATH_ARGS[@]}" --invert-paths --strip-blobs-bigger-than 50M

  echo "filter-repo complete. Review the repository locally before pushing."
  echo "Suggested next steps:
  - Inspect history with: git log --all -- <path>
  - Run: git count-objects -vH
  - If OK, force-push to remote: git push --force --all && git push --force --tags
  "

elif [ "$tool" = "2" ]; then
  if ! command -v bfg >/dev/null 2>&1; then
    echo "BFG not found. Install it first: https://rtyley.github.io/bfg-repo-cleaner/" >&2
    exit 3
  fi

  echo "Using BFG: create a bare clone, run BFG, then push back."
  echo "This script will create a temporary bare clone and run BFG against it."

  TMP_BARE="${ROOT_DIR}/../codex_quant_bare.git"
  git clone --mirror "${ROOT_DIR}" "${TMP_BARE}"
  cd "${TMP_BARE}"

  # Example: remove blobs bigger than 100M
  bfg --strip-blobs-bigger-than 100M .
  git reflog expire --expire=now --all && git gc --prune=now --aggressive

  echo "BFG run completed. Mirror is at: ${TMP_BARE}. Inspect before pushing back."
  echo "To push cleaned mirror back to upstream: git push --mirror <remote>"

else
  echo "Invalid selection. Exiting." >&2
  exit 4
fi

echo "DONE. Remember: rewriting history requires all collaborators to re-clone." 
