from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any

import fcntl


def _now_ms() -> int:
    return int(time.time() * 1000)


class SingletonProcessLock:
    """
    Simple non-blocking singleton lock backed by fcntl flock.
    """

    def __init__(self, lock_path: str | Path):
        self.lock_path = Path(lock_path)
        self._fd = None
        self.owner_pid: int | None = None
        self.owner_info: dict[str, Any] | None = None

    def _read_owner_info(self) -> None:
        self.owner_pid = None
        self.owner_info = None
        try:
            text = self.lock_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"pid": int(text)}
            if isinstance(payload, dict):
                pid = payload.get("pid")
                try:
                    self.owner_pid = int(pid) if pid is not None else None
                except Exception:
                    self.owner_pid = None
                self.owner_info = payload
        except Exception:
            return

    def acquire(self, *, role: str | None = None, extra: dict[str, Any] | None = None) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self.lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                fd.close()
            except Exception:
                pass
            self._read_owner_info()
            return False

        payload: dict[str, Any] = {
            "pid": int(os.getpid()),
            "started_at_ms": _now_ms(),
        }
        if role:
            payload["role"] = str(role)
        if isinstance(extra, dict):
            payload.update(extra)
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(json.dumps(payload, ensure_ascii=False))
            fd.flush()
        except Exception:
            pass

        self._fd = fd
        self.owner_pid = int(os.getpid())
        self.owner_info = payload
        atexit.register(self.release)
        return True

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fd.close()
        except Exception:
            pass

