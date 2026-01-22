# server.py
# Engine Server - JAX 엔진을 분리된 프로세스로 실행
# 봇을 재시작해도 엔진은 메모리에 유지되어 재시작 시간이 0.1초 수준으로 단축됩니다.
#
# 실행 방법:
#   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
#
# 주의사항:
#   - JAX는 GPU 메모리를 공유하지 못하므로 workers=1 권장
#   - 이 서버는 웬만해선 끄지 않습니다

from __future__ import annotations

import bootstrap  # ensure JAX/XLA env is set before jax imports

import os
import sys
import time
from pathlib import Path
from typing import Any

# [OPTIMIZATION] JAX Persistent Cache Configuration
# 이 설정은 반드시 'import jax'가 실행되기 전에 선언되어야 합니다.
_BASE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _BASE_DIR / ".jax_cache"
_CACHE_DIR.mkdir(exist_ok=True)

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_CACHE_DIR))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MAX_ENTRY_SIZE_BYTES", "-1")

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# EngineHub 로딩 (JAX 엔진 초기화 포함)
print("[EngineServer] Initializing EngineHub... (JAX compilation may take a while)")
_init_start = time.perf_counter()

from engines.engine_hub import EngineHub

_hub: EngineHub | None = None
_init_time: float = 0.0


def get_hub() -> EngineHub:
    """Lazy singleton for EngineHub."""
    global _hub, _init_time
    if _hub is None:
        t0 = time.perf_counter()
        _hub = EngineHub()
        _init_time = time.perf_counter() - t0
        print(f"[EngineServer] EngineHub initialized in {_init_time:.2f}s")
    return _hub


# FastAPI 앱 생성
app = FastAPI(
    title="Codex Quant Engine Server",
    description="JAX-based Monte Carlo Engine Server for trading decisions",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 엔진 미리 로드 (warm-up)."""
    print("[EngineServer] Starting warm-up...")
    hub = get_hub()
    
    # 간단한 더미 컨텍스트로 JIT 컴파일 유도
    dummy_ctx = {
        "symbol": "WARMUP/USDT:USDT",
        "price": 100.0,
        "mu": 0.0001,
        "sigma": 0.02,
        "spread": 0.0001,
        "regime": "chop",
        "bid": 99.99,
        "ask": 100.01,
        "closes": [100.0] * 20,
        "highs": [101.0] * 20,
        "lows": [99.0] * 20,
        "volumes": [1000.0] * 20,
        "timestamps": list(range(20)),
        "orderbook": {"bids": [[99.9, 10]], "asks": [[100.1, 10]]},
    }
    
    try:
        # 첫 번째 호출로 JIT 컴파일
        _ = hub.decide(dummy_ctx)
        print("[EngineServer] Warm-up completed successfully")
    except Exception as e:
        print(f"[EngineServer] Warm-up failed (non-critical): {e}")
    
    total_init = time.perf_counter() - _init_start
    print(f"[EngineServer] Total initialization time: {total_init:.2f}s")


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트."""
    return {
        "status": "ok",
        "init_time": _init_time,
        "engines": [e.name for e in get_hub().engines],
    }


@app.get("/status")
async def status():
    """상세 상태 정보."""
    hub = get_hub()
    return {
        "status": "ok",
        "engine_count": len(hub.engines),
        "engines": [
            {"name": e.name, "weight": getattr(e, "weight", 1.0)}
            for e in hub.engines
        ],
        "init_time": _init_time,
    }


@app.post("/decide")
async def decide(ctx: dict = Body(...)):
    """
    단일 컨텍스트에 대한 의사결정.
    
    Args:
        ctx: 심볼 컨텍스트 (price, mu, sigma, orderbook 등)
    
    Returns:
        action, ev, confidence, reason 등을 포함한 의사결정 결과
    """
    try:
        hub = get_hub()
        t0 = time.perf_counter()
        result = hub.decide(ctx)
        dt_ms = (time.perf_counter() - t0) * 1000
        
        # 성능 로깅
        if dt_ms > 100:
            print(f"[EngineServer] /decide took {dt_ms:.2f}ms for {ctx.get('symbol', 'UNKNOWN')}")
        
        return JSONResponse(content=result)
    except Exception as e:
        print(f"[EngineServer] /decide error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={
                "action": "WAIT",
                "ev": 0.0,
                "confidence": 0.0,
                "reason": f"Engine error: {str(e)}",
                "meta": {},
            },
            status_code=200,  # 에러여도 200 반환 (fallback 처리)
        )


@app.post("/decide_batch")
async def decide_batch(ctx_list: list[dict] = Body(...)):
    """
    배치 의사결정 - 여러 심볼을 한 번에 처리.
    GPU 배치 처리로 효율성 극대화.
    
    Args:
        ctx_list: 심볼 컨텍스트 리스트
    
    Returns:
        의사결정 결과 리스트
    """
    try:
        if not ctx_list:
            return JSONResponse(content=[])
        
        hub = get_hub()
        t0 = time.perf_counter()
        results = hub.decide_batch(ctx_list)
        dt_ms = (time.perf_counter() - t0) * 1000
        
        print(f"[EngineServer] /decide_batch: {len(ctx_list)} symbols in {dt_ms:.2f}ms")
        
        return JSONResponse(content=results)
    except Exception as e:
        print(f"[EngineServer] /decide_batch error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: 개별 처리
        fallback_results = []
        for ctx in ctx_list:
            try:
                result = get_hub().decide(ctx)
                fallback_results.append(result)
            except Exception:
                fallback_results.append({
                    "action": "WAIT",
                    "ev": 0.0,
                    "confidence": 0.0,
                    "reason": "Engine batch error fallback",
                    "meta": {},
                })
        
        return JSONResponse(content=fallback_results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()
    
    print(f"[EngineServer] Starting on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,  # JAX는 멀티프로세스 GPU 공유 불가
        log_level="info",
    )
