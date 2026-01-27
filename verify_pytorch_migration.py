#!/usr/bin/env python3
"""
PyTorch Backend Verification Script
==================================

JAX → PyTorch 전환 후 핵심 기능들이 정상 작동하는지 검증합니다.
"""

import sys
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_torch_backend():
    """PyTorch backend 기본 기능 테스트"""
    print("=== PyTorch Backend Test ===")
    
    try:
        from engines.mc.torch_backend import ensure_torch, get_torch_device, _TORCH_OK
        
        ensure_torch()
        device = get_torch_device()
        
        print(f"PyTorch Available: {_TORCH_OK}")
        print(f"Device: {device}")
        
        if not _TORCH_OK:
            print("❌ PyTorch not available!")
            return False
            
        # Test simple tensor operations
        import torch
        x = torch.randn(10, 10, device=device)
        y = torch.sum(x)
        print(f"✅ Simple tensor operation successful: sum = {y.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch backend test failed: {e}")
        return False

def test_path_simulation():
    """Path simulation 기능 테스트"""
    print("\n=== Path Simulation Test ===")
    
    try:
        from engines.mc.monte_carlo_engine import MonteCarloEngine
        
        engine = MonteCarloEngine()
        
        # Test single path simulation
        paths = engine.simulate_paths_price(
            seed=42,
            s0=100.0,
            mu=0.1,
            sigma=0.2,
            n_paths=1000,
            n_steps=60,
            dt=1.0/365/24/60,  # 1 minute
            return_torch=False
        )
        
        print(f"✅ Path simulation successful: shape {paths.shape}")
        print(f"   Initial price: {paths[0, 0]:.2f}")
        print(f"   Final price mean: {np.mean(paths[:, -1]):.2f}")
        print(f"   Final price std: {np.std(paths[:, -1]):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path simulation test failed: {e}")
        return False

def test_batch_simulation():
    """Batch simulation 기능 테스트"""
    print("\n=== Batch Simulation Test ===")
    
    try:
        from engines.mc.monte_carlo_engine import MonteCarloEngine
        
        engine = MonteCarloEngine()
        
        # Test batch path simulation
        n_symbols = 3
        seeds = np.array([42, 43, 44])
        s0s = np.array([100.0, 200.0, 50.0])
        mus = np.array([0.1, 0.05, 0.15])
        sigmas = np.array([0.2, 0.3, 0.25])
        
        paths_batch = engine.simulate_paths_price_batch(
            seeds=seeds,
            s0s=s0s,
            mus=mus,
            sigmas=sigmas,
            n_paths=500,
            n_steps=30,
            dt=1.0/365/24/60  # 1 minute
        )
        
        print(f"✅ Batch simulation successful: shape {paths_batch.shape}")
        
        for i in range(n_symbols):
            final_prices = paths_batch[i, :, -1]
            if hasattr(final_prices, 'numpy'):
                final_prices = final_prices.numpy()
            print(f"   Symbol {i}: S0={s0s[i]:.1f}, Final mean={np.mean(final_prices):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch simulation test failed: {e}")
        return False

def test_memory_usage():
    """메모리 사용량 테스트"""
    print("\n=== Memory Usage Test ===")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from engines.mc.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        
        # Large simulation test
        paths = engine.simulate_paths_price_batch(
            seeds=np.arange(10),
            s0s=np.full(10, 100.0),
            mus=np.full(10, 0.1),
            sigmas=np.full(10, 0.2),
            n_paths=2000,
            n_steps=360,  # 6 hours
            dt=1.0/365/24/60  # 1 minute
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        print(f"✅ Memory test completed:")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Memory used: {memory_used:.1f} MB")
        
        # Clean up
        del paths
        gc.collect()
        
        # Check for memory leaks
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_retained = final_memory - initial_memory
        
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Memory retained: {memory_retained:.1f} MB")
        
        if memory_used < 2000:  # Less than 2GB
            print("✅ Memory usage is within acceptable limits")
            return True
        else:
            print("⚠️ High memory usage detected")
            return False
        
    except ImportError:
        print("⚠️ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """메인 검증 함수"""
    print("🔥 JAX → PyTorch Migration Verification")
    print("=" * 50)
    
    tests = [
        ("PyTorch Backend", test_torch_backend),
        ("Path Simulation", test_path_simulation),
        ("Batch Simulation", test_batch_simulation),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JAX → PyTorch migration successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())