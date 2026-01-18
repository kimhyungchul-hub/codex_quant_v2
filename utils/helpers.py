import time
import math
import numpy as np
from typing import Any
import os
from pathlib import Path

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return float(default)

def _load_env_file(path: str) -> bool:
    """
    Minimal .env loader:
      - supports KEY=VALUE lines
      - ignores blank lines and comments starting with '#'
      - strips optional single/double quotes around VALUE
      - does not override already-set env vars
    """
    try:
        p = Path(path).expanduser()
        if not p.exists() or not p.is_file():
            return False
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            if not key:
                continue
            if key in os.environ and str(os.environ.get(key, "")).strip() != "":
                continue
            val = v.strip()
            if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                val = val[1:-1]
            os.environ[key] = val
        return True
    except Exception:
        return False

def _load_env_file_override(path: str) -> bool:
    """
    Same as `_load_env_file`, but always overrides existing env vars.
    Intended for `state/bybit.env` so live/paper runtime can be controlled
    without being blocked by `.env` defaults.
    """
    try:
        p = Path(path).expanduser()
        if not p.exists() or not p.is_file():
            return False
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            if not key:
                continue
            val = v.strip()
            if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                val = val[1:-1]
            os.environ[key] = val
        return True
    except Exception:
        return False

def now_ms() -> int:
    return int(time.time() * 1000)

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _sanitize_for_json(obj, _depth: int = 0):
    """
    payload 전체를 JSON-친화적인 값으로 정규화한다.
    - NaN / Inf -> None (null)
    - numpy scalar/array -> Python 기본형 + 리스트
    - 알 수 없는 타입 -> str(...) 또는 None
    """
    if _depth > 10:
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v, _depth + 1) for v in obj]

    # numpy / float check
    try:
        if isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return [_sanitize_for_json(x, _depth + 1) for x in obj.tolist()]
    except Exception:
        pass

    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj

    return str(obj)

def _calc_rsi(closes, period: int = 14):
    if len(closes) < period + 1:
        return 50.0
    try:
        c = np.asarray(closes, dtype=np.float64)
        diff = np.diff(c)
        up = np.where(diff > 0, diff, 0.0)
        dn = np.where(diff < 0, -diff, 0.0)

        ma_up = np.mean(up[-period:])
        ma_dn = np.mean(dn[-period:])
        if ma_dn == 0:
            return 100.0
        rs = ma_up / ma_dn
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return 50.0
