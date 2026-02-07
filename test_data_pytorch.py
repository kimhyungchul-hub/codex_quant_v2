#!/usr/bin/env python3
"""
Simple data collection test - PyTorch backend verification
OHLCV 로딩, PyTorch 계산, 기본 통계 출력
"""
import asyncio
import sys
sys.path.insert(0, "/Users/jeonghwakim/codex_quant")

from core.orchestrator import build_exchange, LiveOrchestrator

async def test_data():
    print("=== 1. Exchange 초기화 ===")
    exchange = await build_exchange()
    print(f"✅ Exchange: {exchange.id}")
    
    print("\n=== 2. Orchestrator 초기화 ===")
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    orch = LiveOrchestrator(
        symbols=symbols,
        balance=10000.0,
        exchange=exchange
    )
    print(f"✅ Orchestrator: {len(orch.symbols)} symbols")
    
    print("\n=== 3. OHLCV preload ===")
    await orch.preload_all_ohlcv()
    
    print("\n=== 4. Loaded data check ===")
    for sym in symbols:
        if sym in orch._ohlcv_buffer:
            data = orch._ohlcv_buffer[sym]
            closes = data['closes']
            print(f"✅ {sym}: {len(closes)} candles, close={closes[-1]:.2f}")
        else:
            print(f"❌ {sym}: No data")
    
    print("\n=== 5. PyTorch verification ===")
    from engines.mc.jax_backend import _TORCH_OK, _TORCH_DEVICE
    print(f"PyTorch available: {_TORCH_OK}")
    print(f"Device: {_TORCH_DEVICE}")
    
    if _TORCH_OK:
        from engines.mc.tail_sampling import MonteCarloTailSamplingMixin
        import torch
        
        mixin = MonteCarloTailSamplingMixin()
        samples = mixin._sample_increments_torch(
            (1000, 100), 
            mode='gaussian', 
            df=6.0, 
            bootstrap_returns=None
        )
        print(f"✅ GPU sampling: shape={samples.shape}, device={samples.device}")
        print(f"   Mean: {torch.mean(samples):.4f}, Std: {torch.std(samples):.4f}")
    
    await exchange.close()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_data())
