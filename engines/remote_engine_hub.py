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
import config
import asyncio
import atexit
import pickle
import time
from itertools import count
from multiprocessing import Process
from typing import Any, Dict, Optional

import requests

from core.ring_buffer import SharedMemoryRingBuffer

os.environ.setdefault("MC_STREAMING_PARALLEL", "1")
os.environ.setdefault("MC_STREAMING_PARALLEL_WORKERS", "1")


_ATEXIT_REGISTERED = False
_PROCESS_HUBS: set["ProcessEngineHub"] = set()


def _register_atexit_cleanup():
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return

    def _cleanup_all():
        hubs = list(_PROCESS_HUBS)
        for hub in hubs:
            try:
                hub.close()
            except Exception:
                pass

    atexit.register(_cleanup_all)
    _ATEXIT_REGISTERED = True


def _maybe_set_affinity(cpu_ids: Optional[list[int]]) -> None:
    """Set CPU affinity when the platform supports it."""
    if not cpu_ids:
        return
    try:
        os.sched_setaffinity(0, set(int(c) for c in cpu_ids))
    except Exception:
        # macOS does not support sched_setaffinity; ignore silently.
        return


class MCEngineWorker(Process):
    """독립 프로세스에서 MonteCarloEngine을 실행하는 워커."""

    def __init__(
        self,
        req_name: str,
        resp_name: str,
        size: int,
        slot_size: int,
        cpu_affinity: Optional[list[int]],
        env: Dict[str, str],
    ) -> None:
        super().__init__(daemon=True)
        self.req_name = req_name
        self.resp_name = resp_name
        self.size = size
        self.slot_size = slot_size
        self.cpu_affinity = cpu_affinity
        self.env = env

    def _pack(self, obj: dict) -> bytes:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return payload

    def _unpack(self, data: bytes) -> dict:
        return pickle.loads(data)

    def run(self) -> None:  # type: ignore[override]
        for k, v in self.env.items():
            os.environ[k] = v
        try:
            import bootstrap  # noqa: F401
        except Exception:
            pass

        _maybe_set_affinity(self.cpu_affinity)

        try:
            req_buf = SharedMemoryRingBuffer(name=self.req_name, size=self.size, slot_size=self.slot_size, create=False)
            resp_buf = SharedMemoryRingBuffer(name=self.resp_name, size=self.size, slot_size=self.slot_size, create=False, overwrite=True)
        except Exception as e:
            print(f"[MCEngineWorker] ring buffer attach failed: {e}")
            return

        try:
            from engines.mc.monte_carlo_engine import MonteCarloEngine

            engine = MonteCarloEngine()
        except Exception as e:  # pragma: no cover - defensive
            try:
                resp_buf.write(self._pack({"id": -1, "ok": False, "error": f"engine init failed: {e}"}))
            finally:
                req_buf.close()
                resp_buf.close()
            return

        while True:
            raw = req_buf.read()
            if raw is None:
                time.sleep(0)
                continue

            try:
                msg = self._unpack(raw)
            except Exception as e:
                resp_buf.write(self._pack({"id": -1, "ok": False, "error": f"decode failed: {e}"}))
                continue

            msg_type = msg.get("type")
            req_id = msg.get("id", -1)

            if msg_type == "stop":
                resp_buf.write(self._pack({"id": req_id, "ok": True, "result": "stopped"}))
                break

            t0 = time.perf_counter()
            try:
                if msg_type == "ping":
                    result = {"status": "pong"}
                elif msg_type == "decide_batch":
                    payload = msg.get("payload") or []
                    result = engine.decide_batch(payload)
                elif msg_type == "decide":
                    payload = msg.get("payload") or {}
                    result = engine.decide(payload)
                else:
                    raise ValueError(f"unknown msg type: {msg_type}")
                resp = {
                    "id": req_id,
                    "ok": True,
                    "result": result,
                    "dt_ms": int((time.perf_counter() - t0) * 1000),
                }
            except Exception as e:  # pragma: no cover - defensive
                resp = {"id": req_id, "ok": False, "error": str(e)}

            try:
                resp_buf.write(self._pack(resp))
            except Exception:
                continue

        req_buf.close()
        resp_buf.close()


def _worker_env_copy() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k in {"MC_VERBOSE_PRINT", "MC_TAIL_MODE"}
    }


class _ProcessWorkerClient:
    def __init__(
        self,
        name_suffix: str,
        size: int,
        slot_size: int,
        timeout_single: float,
        timeout_batch: float,
        cpu_affinity: Optional[list[int]],
        env: dict[str, str],
    ) -> None:
        self.timeout_single = timeout_single
        self.timeout_batch = timeout_batch
        self._id_gen = count(1)
        self._pending: Dict[int, dict] = {}
        self._last_error: Optional[str] = None
        self._last_success_ts: float = 0.0
        self._closed = False

        self._req_name = f"mc_req_{name_suffix}"
        self._resp_name = f"mc_resp_{name_suffix}"

        self._req_buf = SharedMemoryRingBuffer(
            name=self._req_name, size=size, slot_size=slot_size, create=True, overwrite=True
        )
        self._resp_buf = SharedMemoryRingBuffer(
            name=self._resp_name, size=size, slot_size=slot_size, create=True, overwrite=True
        )

        self._proc = MCEngineWorker(
            req_name=self._req_name,
            resp_name=self._resp_name,
            size=size,
            slot_size=slot_size,
            cpu_affinity=cpu_affinity,
            env=env,
        )
        self._proc.start()

    def _pack(self, obj: dict) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _poll_once(self) -> None:
        while True:
            raw = self._resp_buf.read()
            if raw is None:
                return
            try:
                resp = pickle.loads(raw)
            except Exception:
                continue
            rid = resp.get("id")
            if rid is None:
                continue
            self._pending[rid] = resp

    def _submit(self, msg: dict) -> int:
        req_id = next(self._id_gen)
        payload = {"id": req_id, **msg}
        self._req_buf.write(self._pack(payload))
        return req_id

    def _process_response(self, resp: dict, msg: dict) -> Any:
        if not resp.get("ok", False):
            self._last_error = resp.get("error")
            raise RuntimeError(resp.get("error", "worker error"))
        self._last_success_ts = time.time()
        self._last_error = None
        return resp.get("result")

    def _await_response_sync(self, msg: dict, timeout: float) -> Any:
        req_id = self._submit(msg)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            resp = self._pending.pop(req_id, None)
            if resp is None:
                self._poll_once()
                resp = self._pending.pop(req_id, None)
            if resp:
                return self._process_response(resp, msg)
            time.sleep(0)
        self._last_error = "timeout"
        raise TimeoutError(f"worker response timeout (msg={msg.get('type')})")

    async def _await_response_async(self, msg: dict, timeout: float) -> Any:
        req_id = self._submit(msg)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            resp = self._pending.pop(req_id, None)
            if resp is None:
                self._poll_once()
                resp = self._pending.pop(req_id, None)
            if resp:
                return self._process_response(resp, msg)
            await asyncio.sleep(0)
        self._last_error = "timeout"
        raise TimeoutError(f"worker response timeout (msg={msg.get('type')})")

    def send(self, msg: dict, timeout: float) -> Any:
        return self._await_response_sync(msg, timeout)

    async def send_async(self, msg: dict, timeout: float) -> Any:
        return await self._await_response_async(msg, timeout)

    def is_connected(self) -> bool:
        return bool(self._proc.is_alive()) and (time.time() - self._last_success_ts) < 60

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.send({"type": "stop"}, timeout=min(self.timeout_single, self.timeout_batch, 2.0))
        except Exception:
            pass
        try:
            if self._proc.is_alive():
                self._proc.join(timeout=5.0)
                if self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=1.0)
        except Exception:
            pass
        for buf in (self._req_buf, self._resp_buf):
            try:
                buf.close(unlink=True)
            except Exception:
                pass


class ProcessEngineHub:
    """Run MonteCarloEngine in a separate process via shared memory."""

    def __init__(
        self,
        capacity: int = 32,
        slot_size: int = 131072,
        timeout_single: float = 2.0,
        timeout_batch: float = 10.0,
        cpu_affinity: Optional[list[int]] = None,
        worker_count: int = 1,
        startup_timeout: float | None = None,
        startup_retries: int = 2,
    ) -> None:
        self.capacity = capacity
        self.slot_size = slot_size
        self.timeout_single = timeout_single
        self.timeout_batch = timeout_batch
        self.cpu_affinity = cpu_affinity
        self._closed = False
        self._last_error: Optional[str] = None
        self._last_success_ts: float = 0.0
        self._worker_count = max(1, int(worker_count))
        self._next_worker_idx = 0
        try:
            default_startup_timeout = max(float(self.timeout_single) * 3.0, 8.0)
        except Exception:
            default_startup_timeout = 8.0
        if startup_timeout is None:
            try:
                startup_timeout = float(
                    os.environ.get("MC_ENGINE_STARTUP_TIMEOUT_SEC", default_startup_timeout)
                    or default_startup_timeout
                )
            except Exception:
                startup_timeout = float(default_startup_timeout)
        self._startup_timeout = max(1.0, float(startup_timeout))
        try:
            startup_retries = int(
                os.environ.get("MC_ENGINE_STARTUP_RETRIES", startup_retries) or startup_retries
            )
        except Exception:
            startup_retries = 2
        self._startup_retries = max(1, int(startup_retries))

        env_copy = _worker_env_copy()
        self._workers: list[_ProcessWorkerClient] = []
        for idx in range(self._worker_count):
            suffix = f"{os.getpid()}_{int(time.time() * 1000)}_{idx}"
            worker = _ProcessWorkerClient(
                suffix,
                size=self.capacity,
                slot_size=self.slot_size,
                timeout_single=self.timeout_single,
                timeout_batch=self.timeout_batch,
                cpu_affinity=self.cpu_affinity,
                env=env_copy,
            )
            self._workers.append(worker)

        _PROCESS_HUBS.add(self)
        _register_atexit_cleanup()

        # Ensure every worker is ready before serving requests.
        try:
            self._ensure_workers_ready()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _ensure_workers_ready(self) -> None:
        for idx, worker in enumerate(self._workers):
            last_exc: Exception | None = None
            for attempt in range(1, self._startup_retries + 1):
                try:
                    _ = worker.send({"type": "ping"}, timeout=self._startup_timeout)
                    self._last_success_ts = worker._last_success_ts
                    self._last_error = None
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    self._last_error = str(exc)
                    if attempt < self._startup_retries:
                        time.sleep(min(1.0, 0.2 * attempt))
            if last_exc is not None:
                raise TimeoutError(
                    f"worker[{idx}] startup ping failed after {self._startup_retries} attempts "
                    f"(timeout={self._startup_timeout:.1f}s): {last_exc}"
                )

    def _next_worker(self) -> _ProcessWorkerClient:
        worker = self._workers[self._next_worker_idx % len(self._workers)]
        self._next_worker_idx = (self._next_worker_idx + 1) % len(self._workers)
        return worker

    def decide(self, ctx: dict) -> dict:
        worker = self._next_worker()
        result = worker.send({"type": "decide", "payload": ctx}, timeout=self.timeout_single)
        self._last_success_ts = worker._last_success_ts
        self._last_error = worker._last_error
        return result

    def decide_batch(self, ctx_list: list[dict]) -> list[dict]:
        if not ctx_list:
            return []
        worker = self._next_worker()
        result = worker.send({"type": "decide_batch", "payload": ctx_list}, timeout=self.timeout_batch)
        self._last_success_ts = worker._last_success_ts
        self._last_error = worker._last_error
        return result

    async def decide_batch_async(self, ctx_list: list[dict], timeout: float | None = None) -> list[dict]:
        if not ctx_list:
            return []
        worker = self._next_worker()
        result = await worker.send_async(
            {"type": "decide_batch", "payload": ctx_list},
            timeout if timeout is not None else self.timeout_batch,
        )
        self._last_success_ts = worker._last_success_ts
        self._last_error = worker._last_error
        return result

    def is_connected(self) -> bool:
        return any(worker.is_connected() for worker in self._workers)

    def get_status(self) -> dict:
        return {
            "mode": "process",
            "workers": len(self._workers),
            "connected": self.is_connected(),
            "last_error": self._last_error,
            "last_success": self._last_success_ts,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass

        _PROCESS_HUBS.discard(self)


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
        self.url = url or str(getattr(config, "ENGINE_SERVER_URL", "http://localhost:8000"))
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
def create_engine_hub(
    use_remote: bool | None = None,
    use_process: bool | None = None,
    cpu_affinity: Optional[list[int]] = None,
) -> "RemoteEngineHub | ProcessEngineHub | EngineHub":
    """
    환경 설정에 따라 원격/프로세스/로컬 엔진 허브를 반환.
    """
    if use_remote is None:
        use_remote = bool(getattr(config, "USE_REMOTE_ENGINE", False))

    if use_process is None:
        use_process = bool(getattr(config, "USE_PROCESS_ENGINE", True))

    if cpu_affinity is None:
        affinity_env = str(getattr(config, "MC_ENGINE_CPU_AFFINITY", "")).strip()
        if affinity_env:
            try:
                cpu_affinity = [int(x) for x in affinity_env.split(",") if x.strip()]
            except Exception:
                cpu_affinity = None

    if use_remote:
        print("[create_engine_hub] Using RemoteEngineHub")
        return RemoteEngineHub()

    if use_process:
        print("[create_engine_hub] Using ProcessEngineHub (shared memory IPC)")
        try:
            worker_count = int(os.environ.get("MC_STREAMING_PARALLEL_WORKERS", "1") or 1)
        except Exception:
            worker_count = 1
        worker_count = max(1, worker_count)
        return ProcessEngineHub(
            capacity=int(getattr(config, "MC_ENGINE_SHM_SLOTS", 32)),
            slot_size=int(getattr(config, "MC_ENGINE_SHM_SLOT_SIZE", 131072)),
            timeout_single=float(getattr(config, "MC_ENGINE_TIMEOUT_SINGLE", 2.0)),
            timeout_batch=float(getattr(config, "MC_ENGINE_TIMEOUT_BATCH", 10.0)),
            cpu_affinity=cpu_affinity,
            worker_count=worker_count,
        )

    print("[create_engine_hub] Using local EngineHub")
    from engines.engine_hub import EngineHub

    return EngineHub()


async def _zero_blocking_smoke() -> None:
    """Lightweight non-blocking smoke test for process hub.

    Main loop spins at 10Hz while MC worker handles a decide_batch_async request.
    """

    hub = ProcessEngineHub(capacity=8, slot_size=131072, timeout_batch=5.0)

    async def producer():
        ctx = {
            "symbol": "TEST/USDT:USDT",
            "price": 100.0,
            "mu": 0.0,
            "sigma": 0.2,
            "closes": [100.0] * 60,
            "regime": "chop",
            "direction": 1,
            "in_position": False,
            "ts_ms": int(time.time() * 1000),
        }
        try:
            res = await hub.decide_batch_async([ctx], timeout=2.0)
            print("[SMOKE] decide_batch_async result", res)
        except Exception as e:
            print("[SMOKE] decide_batch_async error", e)

    async def ticker():
        for i in range(10):
            print(f"[SMOKE] main loop tick {i}")
            await asyncio.sleep(0.1)

    await asyncio.gather(producer(), ticker())


if __name__ == "__main__":
    asyncio.run(_zero_blocking_smoke())
