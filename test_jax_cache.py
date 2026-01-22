
import os
import time

def main():
    import engines.mc.jax_backend as jax_backend
    jax_backend.ensure_jax()
    jax = jax_backend.jax
    jnp = jax_backend.jnp

    cache_dir = "/Users/jeonghwakim/codex_quant/state/jax_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass

    print(f"JAX Version: {getattr(jax, '__version__', 'unknown')}")
    try:
        print(f"JAX Cache Dir from config: {jax.config.jax_compilation_cache_dir}")
    except Exception:
        pass

    @jax.jit
    def f(x):
        return jnp.sin(x) + jnp.cos(x)

    x = jnp.ones((1000, 1000))

    t0 = time.time()
    y = f(x).block_until_ready()
    print(f"First run (compile): {time.time() - t0:.4f}s")

    t1 = time.time()
    y = f(x).block_until_ready()
    print(f"Second run (no compile): {time.time() - t1:.4f}s")

    print("Checking cache directory...")
    files = os.listdir(cache_dir)
    print(f"Cache files: {len(files)}")
    for f in files[:5]:
        print(f" - {f}")


if __name__ == '__main__':
    main()
