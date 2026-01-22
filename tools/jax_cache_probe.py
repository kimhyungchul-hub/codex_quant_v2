#!/usr/bin/env python3
import bootstrap  # ensure JAX/XLA env is set before jax imports
import argparse
import os
import shutil
import time
from pathlib import Path


def _dir_stats(root: Path) -> tuple[int, int]:
    files = 0
    size = 0
    if not root.exists():
        return 0, 0
    for p in root.rglob('*'):
        if p.is_file():
            files += 1
            try:
                size += p.stat().st_size
            except OSError:
                pass
    return files, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default='/tmp/jax_cache_test_copilot')
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.clean:
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache_dir)

    import jax
    import jax.numpy as jnp

    try:
        jax.config.update('jax_enable_compilation_cache', True)
    except Exception:
        pass

    try:
        jax.config.update('jax_compilation_cache_dir', str(cache_dir))
    except Exception:
        pass

    before_files, before_size = _dir_stats(cache_dir)

    from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator
    from engines.mc.leverage_optimizer_jax import _warmup_jit_cache

    t0 = time.time()

    try:
        GlobalBatchEvaluator().warmup()
    except Exception as e:
        print('GlobalBatchEvaluator().warmup() failed:', e)

    try:
        _warmup_jit_cache()
    except Exception as e:
        print('_warmup_jit_cache() failed:', e)

    @jax.jit
    def f(x):
        return jnp.sin(x).sum() + (x @ x)

    x = jnp.ones((1024,), dtype=jnp.float32)
    _ = f(x).block_until_ready()

    elapsed_ms = int((time.time() - t0) * 1000)

    after_files, after_size = _dir_stats(cache_dir)

    print('cache_dir:', cache_dir)
    print('jax_version:', jax.__version__)
    print('elapsed_ms:', elapsed_ms)
    print('cache_files_before:', before_files)
    print('cache_files_after:', after_files)
    print('cache_bytes_before:', before_size)
    print('cache_bytes_after:', after_size)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
