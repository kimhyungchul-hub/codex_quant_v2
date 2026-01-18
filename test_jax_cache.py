
import os
import time
import jax
import jax.numpy as jnp

cache_dir = "/Users/jeonghwakim/codex_quant/state/jax_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
# Needs to be set before jax imports usually, but we try both
jax.config.update("jax_compilation_cache_dir", cache_dir)

print(f"JAX Version: {jax.__version__}")
print(f"JAX Cache Dir from config: {jax.config.jax_compilation_cache_dir}")

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
