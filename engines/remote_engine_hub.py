# engines/remote_engine_hub.py
# RemoteEngineHub - 원격 엔진 서버와 통신하는 클라이언트
#
# 이 클래스는 EngineHub와 동일한 인터페이스를 제공하지만,
# 실제 연산은 원격 서버(server.py)에서 수행됩니다.
#
# 사용법:
#   from engines.remote_engine_hub import RemoteEngineHub
#   hub = RemoteEngineHub(url="http://localhost:8000")
#   result = hub.decide(ctx)

from __future__ import annotations

import os
import time
from typing import Any

import requests


class RemoteEngineHub:
    """
    원격 엔진 서버와 HTTP로 통신하는 클라이언트.
    
    EngineHub와 동일한 인터페이스(decide, decide_batch)를 제공하여
    main_engine_mc_v2_final.py에서 쉽게 교체 가능.
    
    장점:
    - 봇 재시작 시 JAX 엔진 재로딩 불필요 (0.1초 수준 재시작)
    - 엔진 서버와 봇 로직 분리로 독립적 업데이트 가능
    - 서버 장애 시 자동 폴백 처리
    
    Args:
        url: 엔진 서버 URL (기본: http://localhost:8000)
        timeout_single: 단일 요청 타임아웃 (초)
        timeout_batch: 배치 요청 타임아웃 (초)
        retry_count: 재시도 횟수
        fallback_local: 서버 불가 시 로컬 EngineHub 폴백 여부
    """
    
    def __init__(
        self,
        url: str | None = None,
        timeout_single: float = 2.0,
        timeout_batch: float = 10.0,
        retry_count: int = 2,
        fallback_local: bool = False,
    ):
        self.url = url or os.environ.get("ENGINE_SERVER_URL", "http://localhost:8000")
        self.timeout_single = timeout_single
        self.timeout_batch = timeout_batch
        self.retry_count = retry_count
        self.fallback_local = fallback_local
        
        self._local_hub = None  # Lazy loading for fallback
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        
        # 연결 상태 추적
        self._last_error: str | None = None
        self._consecutive_errors: int = 0
        self._last_success_ts: float = 0.0
        
        # 헬스체크
        self._check_health()
    
    def _check_health(self) -> bool:
        """서버 헬스체크."""
        try:
            resp = self._session.get(f"{self.url}/health", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[RemoteEngineHub] Connected to {self.url}")
                print(f"[RemoteEngineHub] Engines: {data.get('engines', [])}")
                self._consecutive_errors = 0
                self._last_success_ts = time.time()
                return True
        except Exception as e:
            print(f"[RemoteEngineHub] Health check failed: {e}")
            self._last_error = str(e)
        return False
    
    def _get_local_hub(self):
        """로컬 EngineHub를 lazy-load."""
        if self._local_hub is None:
            print("[RemoteEngineHub] Loading local EngineHub as fallback...")
            from engines.engine_hub import EngineHub
            self._local_hub = EngineHub()
        return self._local_hub
    
    def _default_result(self, reason: str = "Engine unavailable") -> dict:
        """기본 WAIT 결과 반환."""
        return {
            "action": "WAIT",
            "ev": 0.0,
            "confidence": 0.0,
            "reason": reason,
            "meta": {},
        }
    
    def decide(self, ctx: dict) -> dict:
        """
        단일 컨텍스트에 대한 의사결정.
        
        Args:
            ctx: 심볼 컨텍스트
        
        Returns:
            의사결정 결과 (action, ev, confidence, reason, meta 등)
        """
        for attempt in range(self.retry_count + 1):
            try:
                resp = self._session.post(
                    f"{self.url}/decide",
                    json=ctx,
                    timeout=self.timeout_single,
                )
                if resp.status_code == 200:
                    self._consecutive_errors = 0
                    self._last_success_ts = time.time()
                    return resp.json()
                else:
                    self._last_error = f"HTTP {resp.status_code}"
            except requests.exceptions.Timeout:
                self._last_error = "Timeout"
                print(f"[RemoteEngineHub] decide timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                self._last_error = "Connection refused"
                print(f"[RemoteEngineHub] decide connection error (attempt {attempt + 1})")
            except Exception as e:
                self._last_error = str(e)
                print(f"[RemoteEngineHub] decide error: {e}")
        
        self._consecutive_errors += 1
        
        # 폴백 처리
        if self.fallback_local:
            print("[RemoteEngineHub] Falling back to local EngineHub")
            return self._get_local_hub().decide(ctx)
        
        return self._default_result(f"Engine error: {self._last_error}")
    
    def decide_batch(self, ctx_list: list[dict]) -> list[dict]:
        """
        배치 의사결정 - 여러 심볼을 한 번에 처리.
        
        Args:
            ctx_list: 심볼 컨텍스트 리스트
        
        Returns:
            의사결정 결과 리스트
        """
        if not ctx_list:
            return []
        
        for attempt in range(self.retry_count + 1):
            try:
                resp = self._session.post(
                    f"{self.url}/decide_batch",
                    json=ctx_list,
                    timeout=self.timeout_batch,
                )
                if resp.status_code == 200:
                    self._consecutive_errors = 0
                    self._last_success_ts = time.time()
                    return resp.json()
                else:
                    self._last_error = f"HTTP {resp.status_code}"
            except requests.exceptions.Timeout:
                self._last_error = "Timeout"
                print(f"[RemoteEngineHub] decide_batch timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                self._last_error = "Connection refused"
                print(f"[RemoteEngineHub] decide_batch connection error (attempt {attempt + 1})")
            except Exception as e:
                self._last_error = str(e)
                print(f"[RemoteEngineHub] decide_batch error: {e}")
        
        self._consecutive_errors += 1
        
        # 폴백: 개별 처리 시도
        print("[RemoteEngineHub] Batch failed, trying individual requests...")
        results = []
        for ctx in ctx_list:
            try:
                result = self.decide(ctx)
                results.append(result)
            except Exception:
                results.append(self._default_result("Batch fallback error"))
        
        # 여전히 실패하면 로컬 폴백
        if self.fallback_local and self._consecutive_errors > 3:
            print("[RemoteEngineHub] Multiple failures, using local EngineHub")
            return self._get_local_hub().decide_batch(ctx_list)
        
        return results
    
    def is_connected(self) -> bool:
        """서버 연결 상태 확인."""
        return self._consecutive_errors < 3 and (time.time() - self._last_success_ts) < 60
    
    def get_status(self) -> dict:
        """연결 상태 정보 반환."""
        return {
            "url": self.url,
            "connected": self.is_connected(),
            "consecutive_errors": self._consecutive_errors,
            "last_error": self._last_error,
            "last_success": self._last_success_ts,
        }


# 편의 함수: 환경 변수에 따라 적절한 Hub 반환
def create_engine_hub(use_remote: bool | None = None) -> "RemoteEngineHub | EngineHub":
    """
    환경 설정에 따라 RemoteEngineHub 또는 EngineHub를 반환.
    
    Args:
        use_remote: None이면 환경 변수 USE_REMOTE_ENGINE으로 결정
    
    Returns:
        적절한 엔진 허브 인스턴스
    """
    if use_remote is None:
        use_remote = os.environ.get("USE_REMOTE_ENGINE", "0").lower() in ("1", "true", "yes")
    
    if use_remote:
        print("[create_engine_hub] Using RemoteEngineHub")
        return RemoteEngineHub()
    else:
        print("[create_engine_hub] Using local EngineHub")
        from engines.engine_hub import EngineHub
        return EngineHub()
