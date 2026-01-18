#!/usr/bin/env bash
set -euo pipefail

# consolidate_venvs.sh
# - Finds existing venv-like directories under the repo (looking for */bin/activate)
# - Creates a single `.venv` using Python 3.11 and installs dependencies
# - Archives/moves old venv dirs into `archive/venvs_<timestamp>/`
# - Dry-run by default; pass --run to perform actions. Requires a working python3.11.

RUN=0
if [[ "${1:-}" == "--run" ]]; then
  RUN=1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_ROOT="$ROOT/archive/venvs_$(date +%F_%H%M%S)"

echo "Scanning repository for virtualenvs (looking for */bin/activate)..."
# Portable: avoid `mapfile` (not present in macOS bash)
ACTIVATE_PATHS=()
while IFS= read -r p; do
  ACTIVATE_PATHS+=("$p")
done <<< "$(find "$ROOT" -maxdepth 4 -type f -name activate -path "*/bin/activate" 2>/dev/null || true)"

VENV_DIRS=()
for p in "${ACTIVATE_PATHS[@]}"; do
  vdir="$(dirname "${p}")/.."
  vdir_abs="$(cd "$vdir" && pwd)"
  VENV_DIRS+=("$vdir_abs")
done

# Remove duplicates
if [[ ${#VENV_DIRS[@]} -gt 0 ]]; then
  IFS=$'\n' read -r -d '' -a VENV_DIRS < <(printf "%s\n" "${VENV_DIRS[@]}" | awk '!x[$0]++' && printf '\0') || true
fi

echo "Detected venv directories:"
for d in "${VENV_DIRS[@]}"; do
  echo "  - $d"
done

if [[ ${#VENV_DIRS[@]} -eq 0 ]]; then
  echo "No virtualenvs found under the repo (no */bin/activate).";
else
  echo "Found ${#VENV_DIRS[@]} venv(s)."
fi

if [[ $RUN -ne 1 ]]; then
  echo "\nDry-run: To create consolidated .venv and archive old venvs run:" 
  echo "  bash $0 --run"
  exit 0
fi

# Ensure python3.11 is available
if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; print(sys.version_info[:2])' 2>/dev/null | grep -q "3, 11"; then
  PY=python3
else
  echo "python3.11 not found. Please install Python 3.11 (e.g. 'brew install python@3.11') and retry." >&2
  exit 1
fi

cd "$ROOT"

TARGET_VENV="$ROOT/.venv"
if [[ -d "$TARGET_VENV" ]]; then
  echo "Target venv $TARGET_VENV already exists. Skipping creation.";
else
  echo "Creating new venv at $TARGET_VENV using $PY..."
  $PY -m venv "$TARGET_VENV"
fi

echo "Activating target venv and installing dependencies..."
source "$TARGET_VENV/bin/activate"

# Prefer lockfile if present
if [[ -f "$ROOT/requirements-lock.txt" ]]; then
  echo "Installing from requirements-lock.txt (preferred)"
  pip install --upgrade pip
  pip install -r "$ROOT/requirements-lock.txt"
elif [[ -f "$ROOT/requirements.txt" ]]; then
  echo "Installing from requirements.txt"
  pip install --upgrade pip
  pip install -r "$ROOT/requirements.txt"
else
  echo "No requirements file found. Attempting to aggregate from found venvs..."
  TMP_REQ="/tmp/codex_quant_venv_merged_requirements_$(date +%s).txt"
  rm -f "$TMP_REQ"
  for v in "${VENV_DIRS[@]}"; do
    if [[ -x "$v/bin/pip" ]]; then
      echo "Exporting $v packages..."
      "$v/bin/pip" freeze >> "$TMP_REQ" || true
    fi
  done
  if [[ -f "$TMP_REQ" ]]; then
    # dedupe
    awk '!x[$0]++' "$TMP_REQ" > "$TMP_REQ".dedup
    pip install -r "$TMP_REQ".dedup || true
    rm -f "$TMP_REQ" "$TMP_REQ".dedup
  else
    echo "No package lists could be aggregated. You may need to manually pip install needed packages.";
  fi
fi

deactivate || true

echo "Archiving old venv directories to $ARCHIVE_ROOT"
mkdir -p "$ARCHIVE_ROOT"
for v in "${VENV_DIRS[@]}"; do
  name=$(basename "$v")
  dest="$ARCHIVE_ROOT/$name"
  echo "Moving $v -> $dest"
  mv "$v" "$dest"
done

echo "Ensure .venv/ is in .gitignore"
GITIGNORE="$ROOT/.gitignore"
grep -qxF ".venv/" "$GITIGNORE" || echo ".venv/" >> "$GITIGNORE"

echo "Consolidation complete. Activate the new environment with:\n  source .venv/bin/activate"
echo "Verify Python version: .venv/bin/python --version"

exit 0
