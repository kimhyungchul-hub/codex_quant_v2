#!/usr/bin/env bash
# scripts/move_benchmarks.sh
# Dry-run by default. Use --run to perform moves, --commit to commit the changes.
# Moves benchmark_*.py files from repo root into the `benchmarks/` folder using git mv.

set -euo pipefail
DRY_RUN=true
DO_COMMIT=false

for arg in "$@"; do
  case "$arg" in
    --run) DRY_RUN=false ;;
    --commit) DO_COMMIT=true ;;
    -h|--help)
      echo "Usage: $0 [--run] [--commit]"
      echo "  --run    : actually perform git mv (dry-run by default)"
      echo "  --commit : run 'git commit -m' after moving (requires --run)"
      exit 0 ;;
  esac
done

ROOT_FILES=(
  "benchmark_global_batching.py"
  "benchmark_napv_jax.py"
  "benchmark_proper.py"
  "benchmark_realistic.py"
  "benchmark_scalability.py"
)

TARGET_DIR="benchmarks"

echo "Target dir: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

for f in "${ROOT_FILES[@]}"; do
  if [[ ! -e "$PWD/$f" ]]; then
    echo "[MISSING] $f"
    continue
  fi

  if $DRY_RUN; then
    echo "[DRY-RUN] git mv $f $TARGET_DIR/"
  else
    echo "[MOVE] git mv $f $TARGET_DIR/"
    git mv "$f" "$TARGET_DIR/"
  fi
done

if ! $DRY_RUN && $DO_COMMIT; then
  echo "Committing changes..."
  git add "$TARGET_DIR"
  git commit -m "chore: move benchmark scripts to benchmarks/"
  echo "Committed."
fi

echo "Done. Dry-run=$DRY_RUN, committed=$DO_COMMIT"
