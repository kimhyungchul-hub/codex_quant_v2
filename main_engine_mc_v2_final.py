from __future__ import annotations

import os
import sys

# OpenMP shared-memory failures can abort the process on some systems.
# Enforce SHM-disabled defaults before any ML libraries import.
_kmp_shm = os.environ.get("KMP_SHM_DISABLE", "").strip().lower()
if _kmp_shm not in ("1", "true", "yes", "on"):
    os.environ["KMP_SHM_DISABLE"] = "1"
_kmp_use_shm = os.environ.get("KMP_USE_SHM", "").strip().lower()
if _kmp_use_shm in ("", "1", "true", "yes", "on"):
    os.environ["KMP_USE_SHM"] = "0"
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import bootstrap

from pathlib import Path
import config
from engines.mc.config import config as mc_config
from concurrent.futures import ThreadPoolExecutor
from engines.mc.constants import DECIDE_BATCH_TIMEOUT_SEC as MC_DECIDE_BATCH_TIMEOUT_SEC

# ============================================================================
# GPU Thread Pool (prevents asyncio blocking during GPU operations)
# ============================================================================
# GPU 연산은 메인 asyncio 루프를 블로킹하므로 별도 스레드에서 실행
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_worker")

# Timeout (seconds) for decide_batch GPU/remote calls. Centralized in engines.mc.constants
DECIDE_BATCH_TIMEOUT_SEC = MC_DECIDE_BATCH_TIMEOUT_SEC

import asyncio
import json
import hmac
import hashlib
import subprocess
import time
import math
import random
import re
import uuid
import threading
import datetime
import numpy as np
import sqlite3
import shutil
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from collections import deque
from typing import Optional
from aiohttp import web
import ccxt.async_support as ccxt
import aiohttp

from engines.engine_hub import EngineHub
from engines.remote_engine_hub import RemoteEngineHub, create_engine_hub
from core.risk_manager import RiskManager
from utils.alpha_models import (
    compute_mlofi,
    VPINState,
    update_vpin_state,
    init_vpin_state_from_series,
    KalmanCVState,
    update_kalman_cv,
    ParticleFilterState,
    update_particle_filter,
    estimate_hurst_vr,
    GARCHState,
    update_garch_state,
    BayesMeanState,
    update_bayes_mean,
    estimate_ar1_next_return,
    HawkesState,
    update_hawkes_state,
    GaussianHMMState,
    update_gaussian_hmm,
    compute_ou_drift,
    load_causal_weights,
    apply_causal_adjustment,
    load_weight_vector,
    load_direction_model,
    predict_direction_logistic,
    _annualize_mu,
    _annualize_sigma,
)


# 원격 엔진 서버 사용 여부 (환경 변수로 제어)
USE_REMOTE_ENGINE = bool(getattr(config, "USE_REMOTE_ENGINE", False))
ENGINE_SERVER_URL = str(getattr(config, "ENGINE_SERVER_URL", "http://localhost:8000"))

# -------------------------------------------------------------------
# aiohttp 일부 macOS 환경에서 TCP keepalive 설정 시 OSError(22)가 날 수 있다.
# (dashboard 접속 불가 증상). best-effort로 무시 처리.
# -------------------------------------------------------------------
try:
    import aiohttp.tcp_helpers as _aiohttp_tcp_helpers

    _orig_tcp_keepalive = _aiohttp_tcp_helpers.tcp_keepalive

    def _tcp_keepalive_safe(transport):
        try:
            return _orig_tcp_keepalive(transport)
        except OSError:
            return None

    _aiohttp_tcp_helpers.tcp_keepalive = _tcp_keepalive_safe
except Exception:
    pass
from engines.mc_engine import mc_first_passage_tp_sl_jax
from engines.mc_risk import compute_cvar, kelly_with_cvar, PyramidTracker, ExitPolicy, should_exit_position
from regime import adjust_mu_sigma, time_regime, get_regime_mu_sigma
from engines.running_stats import RunningStats
from engines.kelly_allocator import KellyAllocator
from core.continuous_opportunity import ContinuousOpportunityChecker
from core.multi_timeframe_scoring import check_position_switching
from core.database_manager import DatabaseManager, TradingMode

PORT = 9999
DASHBOARD_HOST = str(os.environ.get("DASHBOARD_HOST", "127.0.0.1")).strip() or "127.0.0.1"

SYMBOLS = list(config.SYMBOLS)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

# Prefer the actively edited template inside templates/, but keep fallbacks
# for legacy copies that might still exist.
_dashboard_candidates = [
    TEMPLATE_DIR / "dashboard_v2.html",
    TEMPLATE_DIR / "dashboard.html",
    BASE_DIR / "dashboard_v2.html",
    BASE_DIR / "dashboard.html",
]
DASHBOARD_FILE = next((p for p in _dashboard_candidates if p.exists()), None)
if DASHBOARD_FILE is None:
    raise FileNotFoundError("Dashboard template not found in templates/ or base directory.")

# ---- OHLCV settings
TIMEFRAME = "1m"
OHLCV_PRELOAD_LIMIT = 240   # 시작 시 한 번에 채울 캔들 수
OHLCV_REFRESH_LIMIT = 2     # 이후 갱신은 최소로
OHLCV_SLEEP_SEC = 30        # 더 촘촘한 1분봉 갱신 주기

# ---- Orderbook settings (대시보드 "Orderbook Ready" 해결)
ORDERBOOK_DEPTH = 5
ORDERBOOK_SLEEP_SEC = 2.0     # 심볼 전체를 한 바퀴 도는 주기(대략)

# ---- Networking / Retry (Bybit public endpoints are sensitive)
CCXT_TIMEOUT_MS = 20000
MAX_RETRY = 4
RETRY_BASE_SEC = 0.5
MAX_INFLIGHT_REQ = 1
ORDERBOOK_MAX_INFLIGHT_REQ = 5

# ---- Risk / Execution settings
ENABLE_LIVE_ORDERS = bool(getattr(config, "ENABLE_LIVE_ORDERS", False))  # 실제 주문 호출 토글
LIVE_BALANCE_SYNC_SEC = float(getattr(config, "LIVE_BALANCE_SYNC_SEC", os.environ.get("LIVE_BALANCE_SYNC_SEC", 5.0)) or 5.0)
DEFAULT_LEVERAGE = float(getattr(config, "DEFAULT_LEVERAGE", 5.0) or 5.0)
MAX_LEVERAGE = float(getattr(config, "MAX_LEVERAGE", 50.0) or 50.0)
LEVERAGE_MIN = float(os.environ.get("LEVERAGE_MIN", 1.0) or 1.0)
MAINT_MARGIN_RATE = float(getattr(config, "MAINT_MARGIN_RATE", 0.005) or 0.005)
LIQUIDATION_BUFFER = float(getattr(config, "LIQUIDATION_BUFFER", 0.0025) or 0.0025)
LOSS_STREAK_LIMIT = 3
ALERT_THROTTLE_SEC = 30
ERROR_BURST_LIMIT = 3
ERROR_BURST_WINDOW_SEC = 120
DEFAULT_SIZE_FRAC = 0.10            # balance 대비 기본 진입 비중 (더 공격적)
MAX_POSITION_HOLD_SEC = int(getattr(config, "MAX_POSITION_HOLD_SEC", 600))  # 보유 상한(환경변수 기반)
POSITION_HOLD_MIN_SEC = int(getattr(config, "POSITION_HOLD_MIN_SEC", 0))  # 최소 보유 시간 (초)
POSITION_CAP_ENABLED = False        # 포지션 개수 제한 비활성화(무제한 진입)
EXPOSURE_CAP_ENABLED = True         # 노출 한도 사용
MAX_CONCURRENT_POSITIONS = 99999
MAX_NOTIONAL_EXPOSURE = float(getattr(config, "MAX_NOTIONAL_EXPOSURE", 10.0))  # 기본 총 노출 한도 (잔고 대비 배수)
LIVE_MAX_NOTIONAL_EXPOSURE = float(getattr(config, "LIVE_MAX_NOTIONAL_EXPOSURE", 10.0))  # 라이브 모드 최소 1000%
REBALANCE_THRESHOLD_FRAC = float(getattr(config, "REBALANCE_THRESHOLD_FRAC", 0.02))
REBALANCE_MIN_INTERVAL_SEC = float(getattr(config, "REBALANCE_MIN_INTERVAL_SEC", 0.0) or 0.0)
REBALANCE_MIN_NOTIONAL = float(getattr(config, "REBALANCE_MIN_NOTIONAL", 0.0) or 0.0)

def _normalize_sym_key(sym: str) -> str:
    s = str(sym or "").strip()
    if not s:
        return s
    if "/" not in s:
        # Handle raw Bybit symbols like BTCUSDT/BTCUSDC
        if s.endswith("USDT") and len(s) > 4:
            base = s[:-4]
            return f"{base}/USDT:USDT"
        if s.endswith("USDC") and len(s) > 4:
            base = s[:-4]
            return f"{base}/USDC:USDC"
        return f"{s}/USDT:USDT"
    if ":" not in s:
        # Normalize "BTC/USDT" -> "BTC/USDT:USDT"
        try:
            quote = s.split("/")[-1]
            if quote:
                return f"{s}:{quote}"
        except Exception:
            pass
    return s

def _parse_sym_float_map(env_key: str) -> dict[str, float]:
    raw = str(os.environ.get(env_key, "")).strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
        elif "=" in part:
            k, v = part.split("=", 1)
        else:
            continue
        key = _normalize_sym_key(k.strip())
        try:
            out[key] = float(v.strip())
        except Exception:
            continue
    return out

REBALANCE_THRESHOLD_MAP = _parse_sym_float_map("REBALANCE_THRESHOLD_MAP")
REBALANCE_MIN_INTERVAL_MAP = _parse_sym_float_map("REBALANCE_MIN_INTERVAL_MAP")
REBALANCE_MIN_NOTIONAL_MAP = _parse_sym_float_map("REBALANCE_MIN_NOTIONAL_MAP")
REBALANCE_ENABLED = str(os.environ.get("REBALANCE_ENABLE", os.environ.get("REBALANCE_ENABLED", "1"))).strip().lower() in ("1", "true", "yes", "on")
EV_DROP_THRESHOLD = 0.0003          # EV 급락 exit 감지 임계
K_LEV = float(getattr(config, "K_LEV", 2000.0))  # 레버리지 스케일 (환경변수 기반)
EV_EXIT_FLOOR = {"bull": -0.0003, "bear": -0.0003, "chop": -0.0002, "volatile": -0.0002}
EV_DROP = {"bull": 0.0010, "bear": 0.0010, "chop": 0.0008, "volatile": 0.0008}
PSL_RISE = {"bull": 0.05, "bear": 0.05, "chop": 0.03, "volatile": 0.03}
MAX_DRAWDOWN_LIMIT = float(os.environ.get("MAX_DRAWDOWN_LIMIT", 0.10) or 0.10)  # Kill Switch 기준 (10% DD)
EXECUTION_MODE = "maker_dynamic"   # maker_dynamic | market

# ---- Portfolio Selection (TOP N 종목 선택)
TOP_N_SYMBOLS = int(getattr(config, "TOP_N_SYMBOLS", 4))  # 상위 N개 종목만 진입 (<=0이면 전체)
USE_KELLY_ALLOCATION = bool(getattr(config, "USE_KELLY_ALLOCATION", True))
KELLY_PORTFOLIO_FRAC = float(os.environ.get("KELLY_PORTFOLIO_FRAC", 1.0) or 1.0)
KELLY_PORTFOLIO_FRAC = max(0.0, min(1.0, KELLY_PORTFOLIO_FRAC))

if TOP_N_SYMBOLS <= 0 or TOP_N_SYMBOLS > len(SYMBOLS):
    TOP_N_SYMBOLS = len(SYMBOLS)
USE_CONTINUOUS_OPPORTUNITY = bool(getattr(config, "USE_CONTINUOUS_OPPORTUNITY", True))
SWITCHING_COST_MULT = float(getattr(config, "SWITCHING_COST_MULT", 2.0))  # 교체 비용 승수 (수수료 × 승수)
MAKER_TIMEOUT_SEC = 2.0
VOLATILITY_MARKET_THRESHOLD = 0.012

# ---- Consensus / Spread settings
CONSENSUS_THRESHOLD = 1.0           # 합의 점수 임계값(가중 득표 합)
RSI_PERIOD = 14
RSI_LONG = 60.0
RSI_SHORT = 40.0
SPREAD_LOOKBACK = 60
SPREAD_Z_ENTRY = 2.0
SPREAD_Z_EXIT = 0.5
SPREAD_SIZE_FRAC = 0.02
SPREAD_HOLD_SEC = 600
SPREAD_ENABLED = False  # 스프레드 비활성화 (Unified Score 전용 모드)
SPREAD_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("SOL/USDT:USDT", "BNB/USDT:USDT"),
]
# 스프레드 상한 (entry gate)
SPREAD_PCT_MAX = float(getattr(config, "SPREAD_PCT_MAX", 0.0005) or 0.0005)  # 0.05%
MIN_ENTRY_SCORE = float(getattr(config, "MIN_ENTRY_SCORE", 0.0) or 0.0)
MIN_ENTRY_NOTIONAL = float(getattr(config, "MIN_ENTRY_NOTIONAL", 0.0) or 0.0)
MIN_ENTRY_NOTIONAL_PCT = float(getattr(config, "MIN_ENTRY_NOTIONAL_PCT", 0.0) or 0.0)
MIN_ENTRY_EXPOSURE_PCT = float(getattr(config, "MIN_ENTRY_EXPOSURE_PCT", 0.0) or 0.0)
MIN_LIQ_SCORE = float(getattr(config, "MIN_LIQ_SCORE", 0.0) or 0.0)
# Tick-level microstructure features (short-term direction/volatility)
TICK_BUFFER_SEC = float(getattr(config, "TICK_BUFFER_SEC", 300.0) or 300.0)
TICK_LOOKBACK_SEC = float(getattr(config, "TICK_LOOKBACK_SEC", 30.0) or 30.0)
TICK_MIN_SAMPLES = int(getattr(config, "TICK_MIN_SAMPLES", 8) or 8)
TICK_BREAKOUT_LOOKBACK_SEC = float(getattr(config, "TICK_BREAKOUT_LOOKBACK_SEC", 60.0) or 60.0)
TICK_BREAKOUT_BUFFER_PCT = float(getattr(config, "TICK_BREAKOUT_BUFFER_PCT", 0.0004) or 0.0004)
MIN_TICK_VOL = float(getattr(config, "MIN_TICK_VOL", 0.0) or 0.0)
PRE_MC_ENABLED = bool(getattr(config, "PRE_MC_ENABLED", False))
PRE_MC_BLOCK_ON_FAIL = bool(getattr(config, "PRE_MC_BLOCK_ON_FAIL", True))
PRE_MC_MIN_EXPECTED_PNL = float(getattr(config, "PRE_MC_MIN_EXPECTED_PNL", 0.0) or 0.0)
PRE_MC_MIN_CVAR = float(getattr(config, "PRE_MC_MIN_CVAR", -0.05) or -0.05)
PRE_MC_MAX_LIQ_PROB = float(getattr(config, "PRE_MC_MAX_LIQ_PROB", 0.05) or 0.05)
PRE_MC_SIZE_SCALE = float(getattr(config, "PRE_MC_SIZE_SCALE", 0.5) or 0.5)

# Bybit USDT-Perp 기본 수수료
# Maker 주문 우선 사용으로 수수료 절감 (0.12% → 0.02%)
BYBIT_TAKER_FEE = 0.0006  # 0.06% per side (시장가)
BYBIT_MAKER_FEE = 0.0001  # 0.01% per side (지정가)
TAKER_FEE_RATE = BYBIT_TAKER_FEE  # alias for portfolio switching cost
MAKER_FEE_RATE = BYBIT_MAKER_FEE  # alias for portfolio switching cost
USE_MAKER_ORDERS = bool(getattr(config, "USE_MAKER_ORDERS", True))

# EV auto-tuning (p95 of recent EVs)
EV_TUNE_WINDOW_SEC = 30 * 60   # 30 minutes
EV_TUNE_PCTL = 95
EV_TUNE_MIN_SAMPLES = 40
EV_ENTER_FLOOR_MIN = 0.0008   # 조금 완화
EV_ENTER_FLOOR_MAX = 0.0025   # 상한도 소폭 완화

# 엔트리 히스테리시스 / 쿨다운
COOLDOWN_SEC = 60
ENTRY_STREAK_MIN = 1
# cooldown presets used by _mark_exit_and_cooldown()
COOLDOWN_TP_SEC = 30
COOLDOWN_RISK_SEC = 120

def now_ms() -> int:
    return int(time.time() * 1000)


class LiveOrchestrator:
    def __init__(self, exchange, data_exchange=None):
        # 환경 변수에 따라 로컬 또는 원격 엔진 허브 선택
        if USE_REMOTE_ENGINE:
            print(f"[LiveOrchestrator] Using RemoteEngineHub @ {ENGINE_SERVER_URL}")
            self.hub = RemoteEngineHub(url=ENGINE_SERVER_URL, fallback_local=True)
        else:
            # 기본값: 프로세스 분리 허브를 사용하여 GIL 차단을 제거
            use_process = bool(getattr(config, "USE_PROCESS_ENGINE", True))
            cpu_affinity = None
            affinity_env = str(getattr(config, "MC_ENGINE_CPU_AFFINITY", "")).strip()
            if affinity_env:
                try:
                    cpu_affinity = [int(x) for x in affinity_env.split(",") if x.strip()]
                except Exception:
                    cpu_affinity = None
            self.hub = create_engine_hub(use_remote=False, use_process=use_process, cpu_affinity=cpu_affinity)
            print(f"[LiveOrchestrator] Using {'ProcessEngineHub' if use_process else 'EngineHub'}")
        self.exchange = exchange
        self.data_exchange = data_exchange or exchange
        self._net_sem = asyncio.Semaphore(MAX_INFLIGHT_REQ)
        self._ob_sem = asyncio.Semaphore(ORDERBOOK_MAX_INFLIGHT_REQ)
        self._last_ok = {"tickers": 0, "ohlcv": {s: 0 for s in SYMBOLS}, "ob": {s: 0 for s in SYMBOLS}}

        self.clients = set()
        # 마지막으로 전송된 full_update 메시지(문자열). 새로 연결된 클라이언트에 즉시 재전송 가능
        self._last_broadcast_msg = None
        self.logs = deque(maxlen=300)
        self.anomalies = deque(maxlen=200)
        self._loss_streak = 0
        self._last_alert_ts = {}
        self._error_burst = 0
        self._last_error_ts = 0.0
        self._telegram_token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "")).strip()
        self._telegram_chat_id = str(getattr(config, "TELEGRAM_CHAT_ID", "")).strip()
        self._telegram_enabled = bool(self._telegram_token and self._telegram_chat_id)

        self.balance = 10_000.0
        self.positions = {}  # sym -> position dict (demo/paper)
        self._live_wallet_balance = None
        self._live_equity = None
        self._live_free_balance = None
        self._live_total_initial_margin = None
        self._live_total_maintenance_margin = None
        self._last_live_sync_ms = None
        self._last_live_sync_err = None
        self._live_equity_history = deque(maxlen=20_000)
        self._live_balance_key_warned = False
        self._live_balance_stale_warned = False
        self._live_balance_cache_warned = False
        self._hold_eval_lock = threading.Lock()
        self._rebalance_on_next_cycle = False
        self._force_rebalance_cycle = False
        self._last_positions_sync_ms: int | None = None
        self._exchange_positions_view: list[dict] = []
        self._exchange_positions_by_symbol: dict[str, dict] = {}
        self._exchange_positions_ts: int | None = None
        self._exchange_positions_source: str | None = None
        self._ws_positions_cache: dict[str, dict] = {}
        self._ws_positions_last_ms: int | None = None
        self._ws_positions_last_err: str | None = None
        self._ws_positions_connected: bool = False
        self.leverage = DEFAULT_LEVERAGE
        self.max_leverage = MAX_LEVERAGE
        self.enable_orders = bool(ENABLE_LIVE_ORDERS)
        self._trading_mode = TradingMode.LIVE if self.enable_orders else TradingMode.PAPER
        self.max_positions = MAX_CONCURRENT_POSITIONS
        if self._trading_mode == TradingMode.LIVE:
            # Keep live exposure cap at least 1000% unless user explicitly sets higher global cap.
            self.max_notional_frac = max(float(MAX_NOTIONAL_EXPOSURE), float(LIVE_MAX_NOTIONAL_EXPOSURE))
        else:
            self.max_notional_frac = float(MAX_NOTIONAL_EXPOSURE)
        self.position_cap_enabled = POSITION_CAP_ENABLED
        self.exposure_cap_enabled = EXPOSURE_CAP_ENABLED
        self.default_size_frac = DEFAULT_SIZE_FRAC
        self.safety_mode = False
        self._emergency_stop_handled = False
        self._emergency_stop_ts: int | None = None
        self.max_drawdown_limit = MAX_DRAWDOWN_LIMIT
        self.initial_equity = None
        self._is_hedge_mode = False
        self._position_mode = "oneway"
        self.risk_manager = RiskManager(self)
        # 수수료 설정 (Bybit taker/maker)
        self.fee_taker = BYBIT_TAKER_FEE
        self.fee_maker = BYBIT_MAKER_FEE
        # Maker 주문 사용시 수수료 0.02% (왕복), Taker는 0.12%
        self.fee_mode = "maker" if USE_MAKER_ORDERS else "taker"
        self._decision_log_every = int(getattr(config, "DECISION_LOG_EVERY", 10))
        self._decision_cycle = 0
        self._last_mc_ms: float | None = None
        self._last_mc_ts: int | None = None
        self._last_mc_ctxs: int | None = None
        self._last_mc_status: str | None = None
        self._last_mc_backend: str | None = None
        self._last_mc_device: str | None = None
        self._cycle_free_balance: float | None = None
        self._cycle_reserved_margin: float = 0.0
        self._cycle_reserved_notional: float = 0.0
        self._cycle_balance_ts: int | None = None
        self._dashboard_fast = str(os.environ.get("DASHBOARD_FAST_LOOP", "1")).strip().lower() in ("1", "true", "yes", "on")
        self._dashboard_refresh_sec = float(os.environ.get("DASHBOARD_REFRESH_SEC", 0.5) or 0.5)
        self._last_decisions: dict[str, dict] = {}
        self._last_rows: list[dict] = []
        # Keep a compact per-symbol ctx snapshot so dashboard_fast can render alpha fields
        # without rebuilding rows from ctx=None.
        self._last_ctx_by_sym: dict[str, dict] = {}
        self.spread_pairs = SPREAD_PAIRS
        self.spread_enabled = SPREAD_ENABLED
        self._last_actions = {}
        self._last_rebalance_ts: dict[str, float] = {}
        self._cooldown_until = {s: 0.0 for s in SYMBOLS}
        self._entry_streak = {s: 0 for s in SYMBOLS}
        self._last_exit_kind = {s: "NONE" for s in SYMBOLS}
        self._streak = {s: 0 for s in SYMBOLS}
        self._ev_tune_hist = {s: deque(maxlen=2000) for s in SYMBOLS}  # (ts, ev)
        self._ev_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._cvar_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._ofi_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_thr_ema: dict[tuple[str, str], float] = {}
        self._ev_thr_ema_ts: dict[tuple[str, str], int] = {}
        self._ev_drop_state: dict[str, dict] = {s: {"prev": None, "streak": 0} for s in SYMBOLS}
        self._spread_regime_hist: dict[tuple[str, str], deque] = {}
        self._liq_regime_hist: dict[tuple[str, str], deque] = {}
        self._ofi_resid_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._cvar_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._ema_ev: dict[str, tuple[float, int]] = {}
        self._ema_psl: dict[str, tuple[float, int]] = {}
        self._exit_bad_ticks: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._dyn_leverage = {s: self.leverage for s in SYMBOLS}
        self._last_leverage_sync_ms_by_sym: dict[str, int] = {}
        self._last_leverage_target_by_sym: dict[str, float] = {}
        self._leverage_sync_fail_by_sym: dict[str, dict] = {}
        self._balance_reject_110007_by_sym: dict[str, dict] = {}
        self._symbol_quality_cache_by_sym: dict[str, dict] = {}
        self._lev_floor_sticky_counts: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._lev_diag_last_ms_by_sym: dict[str, int] = {}
        self._leverage_sync_min_interval_ms = int(os.environ.get("LIVE_LEVERAGE_SYNC_MIN_INTERVAL_MS", 15_000) or 15_000)
        self._last_close_ts: dict[str, int] = {}
        self._last_open_ts: dict[str, int] = {}
        self._last_exchange_pos = {}
        self._live_missing_pos_counts: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._outside_universe_positions: dict[str, dict] = {}
        self._empty_positions_fetches: int = 0
        self._last_nonempty_pos_fetch_ms: int = 0
        self._external_close_missing_counts: dict[str, int] = {}
        self._last_dynamic_universe_ms: int = 0
        self._runtime_universe = set(SYMBOLS)
        self._last_universe_refresh_ms = 0
        self._universe_refresh_sec = float(os.environ.get("UNIVERSE_REFRESH_SEC", 5.0) or 5.0)
        # Hybrid auto-tuning (entry floor / confidence scale)
        self._hybrid_entry_floor_dyn: float | None = None
        self._hybrid_entry_floor_dyn_ts: int = 0
        self._hybrid_conf_scale_dyn: float | None = None
        self._hybrid_conf_scale_dyn_ts: int = 0
        self._last_portfolio_joint_ts = 0.0
        self._last_portfolio_report = None
        self._portfolio_joint_task = None
        self.portfolio_joint_interval_sec = float(os.environ.get("PORTFOLIO_JOINT_INTERVAL_SEC", 180) or 180)
        self.trade_tape = deque(maxlen=20_000)
        self.eval_history = deque(maxlen=5_000)  # 예측 vs 실제 품질 평가용
        event_min_score = float(getattr(config, "EVENT_EXIT_SCORE", -0.0005))
        event_max_p_sl = float(getattr(config, "EVENT_EXIT_MAX_P_SL", 0.55))
        event_max_abs_cvar = float(getattr(config, "EVENT_EXIT_MAX_ABS_CVAR", 0.010))
        self.exit_policy = ExitPolicy(
            min_event_score=event_min_score,
            max_event_p_sl=event_max_p_sl,
            min_event_p_tp=0.30,
            grace_sec=20,
            max_hold_sec=0,
            time_stop_mult=2.2,
            max_abs_event_cvar_r=event_max_abs_cvar,
        )
        self.mc_cache = {}  # (sym, side, regime, price_bucket) -> (ts, meta)
        self.mc_cache_ttl = 2.0  # seconds

        self.market = {s: {"price": None, "ts": 0} for s in SYMBOLS}
        # Tick-level buffer (for short-term direction/volatility)
        try:
            _price_refresh = float(os.environ.get("PRICE_REFRESH_SEC", 1.0) or 1.0)
        except Exception:
            _price_refresh = 1.0
        self._tick_buffer_sec = max(5.0, float(TICK_BUFFER_SEC))
        tick_maxlen = int(max(16, (self._tick_buffer_sec / max(_price_refresh, 0.2)) * 2.0))
        self._tick_buffer = {s: deque(maxlen=tick_maxlen) for s in SYMBOLS}
        self._tick_feature_cache: dict[str, tuple[int, dict]] = {}

        # 1m close buffer (preload로 한 번에 채움)
        self.ohlcv_buffer = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}
        self.ohlcv_open = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}
        self.ohlcv_high = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}
        self.ohlcv_low = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}
        self.ohlcv_volume = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}

        # orderbook 상태(대시보드 표기용)
        self.orderbook = {s: {"ts": 0, "ready": False, "bids": [], "asks": []} for s in SYMBOLS}

        # alpha state (per-symbol)
        self._alpha_state: dict[str, dict] = {}
        self._alpha_mlofi_weights = load_weight_vector(getattr(mc_config, "mlofi_weight_path", "") or "")
        self._alpha_mlofi_last_reload_ms = 0
        self._alpha_mlofi_reload_sec = float(getattr(mc_config, "mlofi_reload_sec", 60.0) or 60.0)
        self._alpha_mlofi_weights_mtime: float | None = None
        self._alpha_causal_weights = load_causal_weights(getattr(mc_config, "causal_weights_path", "") or "")
        self._alpha_causal_last_reload_ms = 0
        self._alpha_causal_reload_sec = float(getattr(mc_config, "causal_reload_sec", 60.0) or 60.0)
        self._alpha_causal_weights_mtime: float | None = None
        self._alpha_dir_last_reload_ms = 0
        self._alpha_dir_reload_sec = float(getattr(mc_config, "alpha_direction_reload_sec", 60.0) or 60.0)
        self._alpha_dir_model_path = str(getattr(mc_config, "alpha_direction_model_path", "") or "").strip()
        self._alpha_dir_model_mtime: float | None = None
        self._alpha_dir_model: dict[str, object] = load_direction_model(self._alpha_dir_model_path)
        self._alpha_garch_last_reload_ms = 0
        self._alpha_garch_reload_sec = float(getattr(mc_config, "garch_param_reload_sec", 86400.0) or 86400.0)
        self._alpha_garch_path = str(getattr(mc_config, "garch_param_path", "") or "").strip()
        self._alpha_garch_mtime: float | None = None
        self._alpha_garch_override: dict[str, float] | None = None
        self._alpha_garch_symbol_overrides: dict[str, dict[str, float]] = {}
        self._alpha_garch_fit_enabled = bool(getattr(mc_config, "garch_daily_fit_enabled", False))
        self._alpha_garch_fit_interval_sec = float(getattr(mc_config, "garch_fit_interval_sec", 86400.0) or 86400.0)
        self._alpha_garch_fit_data_glob = str(getattr(mc_config, "garch_fit_data_glob", "data/*.csv") or "data/*.csv")
        self._alpha_garch_fit_lookback = int(getattr(mc_config, "garch_fit_lookback", 4000) or 4000)
        self._alpha_garch_fit_min_obs = int(getattr(mc_config, "garch_fit_min_obs", 300) or 300)
        self._alpha_garch_fit_timeout_sec = float(getattr(mc_config, "garch_fit_timeout_sec", 120.0) or 120.0)
        self._alpha_garch_fit_allow_fallback = bool(getattr(mc_config, "garch_fit_allow_fallback", True))
        self._alpha_garch_fit_last_ms = 0
        self._alpha_garch_fit_start_ms = 0
        self._alpha_garch_fit_proc: subprocess.Popen[str] | None = None
        self._alpha_ml_model = None
        self._alpha_ml_ready = False
        self._alpha_causal_feature_names = ["ofi", "mlofi", "vpin", "mu_kf", "hurst", "mu_ml"]
        self._alpha_train_mlofi = deque(maxlen=int(os.environ.get("ALPHA_WEIGHT_MAX_SAMPLES", 200000) or 200000))
        self._alpha_train_causal = deque(maxlen=int(os.environ.get("ALPHA_WEIGHT_MAX_SAMPLES", 200000) or 200000))
        self._alpha_train_last_save_ms = 0
        self._alpha_train_save_interval_sec = float(os.environ.get("ALPHA_WEIGHT_SAMPLE_SAVE_SEC", 300) or 300)
        self._alpha_train_last_train_ms = 0
        self._alpha_train_interval_sec = float(os.environ.get("ALPHA_WEIGHT_TRAIN_INTERVAL_SEC", 1800) or 1800)
        self._alpha_train_min_samples = int(os.environ.get("ALPHA_WEIGHT_TRAIN_MIN_SAMPLES", 2000) or 2000)
        self._alpha_train_min_new_samples = int(os.environ.get("ALPHA_WEIGHT_TRAIN_MIN_NEW_SAMPLES", 300) or 300)
        self._alpha_train_last_mlofi_n = 0
        self._alpha_train_last_causal_n = 0
        self._alpha_train_last_window_key: str | None = None
        self._alpha_train_last_exit_closed = 0
        self._alpha_train_state_path: Path | None = None
        self._alpha_mlofi_train_path = str(os.environ.get("ALPHA_MLOFI_TRAIN_PATH", "state/mlofi_train_samples.npz"))
        self._alpha_causal_train_path = str(os.environ.get("ALPHA_CAUSAL_TRAIN_PATH", "state/causal_train_samples.npz"))
        self._alpha_mlofi_file_samples = 0
        self._alpha_causal_file_samples = 0
        self._alpha_reload_check_last_ms = 0
        self._alpha_reload_check_interval_sec = float(os.environ.get("ALPHA_RELOAD_CHECK_SEC", 1.0) or 1.0)
        self._alpha_garch_fit_last_window_key: str | None = None
        self._load_alpha_training_samples()

        # OHLCV freshness / dedupe
        self._last_kline_ts = {s: 0 for s in SYMBOLS}      # 마지막 캔들 timestamp(ms)
        self._last_kline_ok_ms = {s: 0 for s in SYMBOLS}   # 마지막으로 buffer 갱신 성공한 시각(ms)
        self._preloaded = {s: False for s in SYMBOLS}

        self._last_feed_ok_ms = 0
        self._equity_history = deque(maxlen=20_000)
        self._state_json_load_errors: set[str] = set()
        # persistence
        self.state_dir = BASE_DIR / "state"
        self.state_dir.mkdir(exist_ok=True)
        mode_suffix = "live" if self._trading_mode == TradingMode.LIVE else "paper"
        self.state_files = {
            "equity": self.state_dir / f"equity_history_{mode_suffix}.json",
            "trade": self.state_dir / f"trade_tape_{mode_suffix}.json",
            "eval": self.state_dir / f"eval_history_{mode_suffix}.json",
            "positions": self.state_dir / f"positions_{mode_suffix}.json",
            "balance": self.state_dir / f"balance_{mode_suffix}.json",
        }
        self._exit_trade_types = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}
        self._closed_trade_count = 0
        self._reval_baseline_path = self.state_dir / "reval_baseline.json"
        self._reval_baseline_closed = 0
        self._reval_baseline_last_load_ms = 0
        self._reval_baseline_reload_sec = float(os.environ.get("REVAL_BASELINE_RELOAD_SEC", 10.0) or 10.0)
        # NewExit anchor is intentionally independent from rolling reval baseline.
        # It provides a monotonic "non-resetting" counter for dashboard visibility.
        self._new_exit_anchor_path = self.state_dir / "new_exit_anchor.json"
        self._new_exit_anchor_closed = 0
        self._new_exit_anchor_loaded = False
        self._reval_target_min = int(os.environ.get("REVAL_TARGET_MIN_NEW_CLOSED", 200) or 200)
        self._reval_target_max = int(os.environ.get("REVAL_TARGET_MAX_NEW_CLOSED", 300) or 300)
        self._auto_reval_status_path = self.state_dir / "auto_reval_db_report.json"
        self._auto_reval_status_reload_sec = float(os.environ.get("AUTO_REVAL_STATUS_RELOAD_SEC", 5.0) or 5.0)
        self._auto_reval_status_last_load_ms = 0
        self._auto_reval_status_cache: dict = {}
        self._auto_tune_override_path = self.state_dir / "auto_tune_overrides.json"
        self._auto_tune_status_path = self.state_dir / "auto_tune_status.json"
        self._auto_tune_reload_sec = float(os.environ.get("AUTO_TUNE_RELOAD_SEC", 5.0) or 5.0)
        self._auto_tune_last_reload_ms = 0
        self._auto_tune_overrides_mtime: float | None = None
        self._auto_tune_status_mtime: float | None = None
        self._auto_tune_overrides_cache: dict[str, object] = {}
        self._auto_tune_status_cache: dict[str, object] = {}
        self._auto_tune_last_apply_ms = 0
        self._auto_tune_last_batch_id = 0
        self._event_exit_strict_consistency = str(
            os.environ.get("EVENT_EXIT_STRICT_CONSISTENCY_MODE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        try:
            self._event_alignment_window = int(os.environ.get("EVENT_EXIT_ALIGNMENT_WINDOW", 500) or 500)
        except Exception:
            self._event_alignment_window = 500
        self._event_alignment_window = max(50, min(5000, int(self._event_alignment_window)))
        self._event_alignment_samples: deque[dict[str, object]] = deque(maxlen=self._event_alignment_window)
        self._event_alignment_anchor_path = self.state_dir / "event_alignment_anchor.json"
        self._event_alignment_report_path = self.state_dir / "event_alignment_split_report.json"
        try:
            self._event_alignment_report_write_sec = float(os.environ.get("EVENT_EXIT_ALIGNMENT_REPORT_WRITE_SEC", 5.0) or 5.0)
        except Exception:
            self._event_alignment_report_write_sec = 5.0
        self._event_alignment_report_write_sec = max(0.5, float(self._event_alignment_report_write_sec))
        self._event_alignment_report_last_write_ms = 0
        self._event_alignment_anchor_loaded = False
        self._event_alignment_anchor_ts_ms = 0
        self._event_alignment_anchor_trade_id = 0
        self._event_alignment_since_anchor: dict[str, object] = {}
        self._alpha_train_state_path = self.state_dir / "alpha_train_state.json"
        self._legacy_delev_last_sync_ms_by_sym: dict[str, int] = {}
        self._last_state_persist_ms = 0
        self._migrate_legacy_state_files()
        self._load_persistent_state()
        # 러닝 통계
        self.stats = RunningStats(maxlen=5000)
        
        # ---- SQLite Database (JSON 대체) - state/ 고정 + 모드별 DB 파일 분리
        self.data_dir = self.state_dir
        self.data_dir.mkdir(exist_ok=True)
        db_name = "bot_data_live.db" if self._trading_mode == TradingMode.LIVE else "bot_data_paper.db"
        self._db_path = str(self.data_dir / db_name)
        self._maybe_migrate_legacy_db()
        self.db = DatabaseManager(db_path=self._db_path)
        self._bootstrap_state_from_db_if_empty()
        self._refresh_closed_trade_counter()
        self._load_event_alignment_anchor(force=False)
        self._bootstrap_event_alignment_samples(force=True)
        self._bootstrap_event_alignment_since_anchor(force=True)
        self._persist_event_alignment_report(force=True)
        self._load_alpha_train_state(force=True)
        self._load_reval_baseline(force=True)
        self._load_new_exit_anchor(force=True)
        self._archive_outside_universe_positions(source="startup")
        self._init_initial_equity()
        self._maybe_reset_initial_equity_on_start()
        
        # ---- Portfolio Management (TOP N 선택 + Kelly 배분 + 교체 비용 평가)
        self.kelly_allocator = KellyAllocator(max_leverage=MAX_LEVERAGE, half_kelly=0.5)
        self.opportunity_checker = ContinuousOpportunityChecker(self)
        self._symbol_scores: dict[str, float] = {}  # sym -> score (EV or NAPV)
        self._symbol_hold_scores: dict[str, float] = {}  # sym -> score for current side (hold)
        self._symbol_ranks: dict[str, int] = {}     # sym -> rank (1=best)
        self._top_n_symbols: list[str] = []         # TOP N 종목 리스트
        self._last_ranking_ts = 0                   # 마지막 순위 갱신 시각
        self._kelly_allocations: dict[str, float] = {}  # sym -> allocation weight

        # Hold-vs-exit evaluation cache
        self._hold_eval_last_ts: dict[str, int] = {}
        self._entry_link_infer_cache: dict[tuple[str, str, int], str] = {}
        self._mc_engine_cache = None
        self._mc_engine_warned = False
        
        # ---- SoA (Structure of Arrays) Pre-allocation for Zero-Copy Batch Ingestion ----
        # CRITICAL: 메모리 재할당 방지 & JAX Static Shape 유지
        from engines.mc.constants import STATIC_MAX_SYMBOLS
        self._batch_max_symbols = max(len(SYMBOLS), STATIC_MAX_SYMBOLS)  # JAX JIT stability
        self._sym_to_idx = {s: i for i, s in enumerate(SYMBOLS)}  # O(1) lookup
        # Pre-allocated numpy arrays (reused each cycle)
        self._batch_prices = np.zeros(self._batch_max_symbols, dtype=np.float64)
        self._batch_mus = np.zeros(self._batch_max_symbols, dtype=np.float64)
        self._batch_sigmas = np.ones(self._batch_max_symbols, dtype=np.float64) * 0.2  # default vol
        self._batch_leverages = np.ones(self._batch_max_symbols, dtype=np.float64) * DEFAULT_LEVERAGE
        self._batch_ofi_scores = np.zeros(self._batch_max_symbols, dtype=np.float64)
        self._batch_valid_mask = np.zeros(self._batch_max_symbols, dtype=bool)  # which slots are valid
        self._log(f"[INIT] Portfolio Management: TOP_N={TOP_N_SYMBOLS}, Kelly={USE_KELLY_ALLOCATION}, Opportunity={USE_CONTINUOUS_OPPORTUNITY}")
        self._log(f"[INIT] SoA Batch Arrays: max_symbols={self._batch_max_symbols}, shape=({self._batch_max_symbols},)")

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "INFO", "msg": text})
        print(text, flush=True)

    def _log_err(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "ERROR", "msg": text})
        print(text, flush=True)

    def _normalize_entry_time_ms(self, entry_time, default: int | None = None) -> int:
        """Normalize entry_time to ms (handle seconds or microseconds)."""
        if default is None:
            default = now_ms()
        try:
            t = int(entry_time)
        except Exception:
            return int(default)
        # If value looks like seconds, convert to ms. If it looks like micros, downscale.
        if t < 1_000_000_000_000:
            t = t * 1000
        elif t > 100_000_000_000_000:
            t = int(t / 1000)
        return int(t)

    def _extract_entry_time_ms_from_info(self, info: dict | None, default: int | None = None) -> int:
        """Extract best-effort entry timestamp in ms from exchange info payload."""
        if default is None:
            default = now_ms()
        if not isinstance(info, dict):
            return int(default)
        cands: list[int] = []
        for key in ("createdTimeMs", "createdTime", "updateTime", "updatedTime"):
            val = info.get(key)
            if val is None:
                continue
            t = self._normalize_entry_time_ms(val, default=default)
            if t > 0:
                cands.append(int(t))
        if not cands:
            return int(default)
        # Prefer the most recent exchange timestamp so stale createdTime does not inflate age.
        t_best = max(cands)
        now_t = now_ms()
        if t_best > now_t + 10 * 60 * 1000:
            return int(default)
        return int(t_best)

    def _infer_entry_link_id_from_db(self, sym: str, side: str, entry_time_ms: int | None) -> str | None:
        """
        Best-effort linkage recovery for exchange-synced positions that do not carry entry_link_id.
        Uses recent DB ENTER/SPREAD rows for the same symbol+side and nearest timestamp match.
        """
        if not sym:
            return None
        side_u = str(side or "").upper()
        if side_u not in ("LONG", "SHORT"):
            return None
        bucket = int((int(entry_time_ms) if entry_time_ms else 0) // 60000)
        cache_key = (str(sym), side_u, bucket)
        cached = self._entry_link_infer_cache.get(cache_key)
        if cached is not None:
            return cached or None
        if not hasattr(self, "db") or self.db is None:
            return None
        try:
            lookback = int(os.environ.get("ENTRY_LINK_INFER_LOOKBACK", 6000) or 6000)
        except Exception:
            lookback = 6000
        lookback = max(200, min(20000, int(lookback)))
        try:
            tol_sec = float(os.environ.get("ENTRY_LINK_INFER_TOL_SEC", 21600) or 21600)
        except Exception:
            tol_sec = 21600.0
        try:
            future_slack_sec = float(os.environ.get("ENTRY_LINK_INFER_FUTURE_SLACK_SEC", 120) or 120)
        except Exception:
            future_slack_sec = 120.0
        tol_ms = int(max(60.0, tol_sec) * 1000.0)
        future_slack_ms = int(max(0.0, future_slack_sec) * 1000.0)
        ref_ts = int(entry_time_ms or 0)
        best_link = ""
        best_gap = None
        try:
            rows = self.db.get_recent_trades(limit=lookback, mode=self._trading_mode, symbol=str(sym))
        except Exception:
            rows = []
        for row in rows or []:
            try:
                action_u = str(row.get("action") or "").upper()
                if action_u not in ("ENTER", "SPREAD"):
                    continue
                row_side = str(row.get("side") or "").upper()
                if row_side != side_u:
                    continue
                link = row.get("entry_link_id") or row.get("trade_uid")
                if not link:
                    continue
                ts_row = int(row.get("timestamp_ms") or 0)
                if ts_row <= 0:
                    continue
                if ref_ts > 0:
                    if ts_row > (ref_ts + future_slack_ms):
                        continue
                    gap = abs(ref_ts - ts_row)
                    if gap > tol_ms:
                        continue
                else:
                    gap = 0
                if best_gap is None or gap < best_gap:
                    best_gap = int(gap)
                    best_link = str(link)
            except Exception:
                continue
        self._entry_link_infer_cache[cache_key] = str(best_link or "")
        return best_link or None

    def _is_auto_env(self, value) -> bool:
        try:
            return str(value).strip().lower() in ("auto", "dyn", "adaptive")
        except Exception:
            return False

    def _get_hybrid_entry_floor(self) -> float:
        env_val = os.environ.get("HYBRID_ENTRY_FLOOR", os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
        auto_entry = self._is_auto_env(env_val) or str(os.environ.get("HYBRID_AUTO_ENTRY", "0")).strip().lower() in ("1", "true", "yes", "on")
        if auto_entry and self._hybrid_entry_floor_dyn is not None:
            return float(self._hybrid_entry_floor_dyn)
        try:
            return float(env_val or 0.0)
        except Exception:
            try:
                return float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
            except Exception:
                return 0.0

    def _get_hybrid_conf_scale(self) -> float:
        env_val = os.environ.get("HYBRID_CONF_SCALE", "0.01")
        auto_conf = self._is_auto_env(env_val) or str(os.environ.get("HYBRID_AUTO_CONF", "0")).strip().lower() in ("1", "true", "yes", "on")
        if auto_conf and self._hybrid_conf_scale_dyn is not None:
            return float(self._hybrid_conf_scale_dyn)
        try:
            return float(env_val or 0.01)
        except Exception:
            return 0.01

    def _update_hybrid_auto_tuning(self, scores_arr, ts_ms: int) -> None:
        try:
            use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
            hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
            hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")
            try:
                entry_floor_eff = self._get_hybrid_entry_floor() if hybrid_only else float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
                if MIN_ENTRY_SCORE > 0:
                    entry_floor_eff = float(max(entry_floor_eff, MIN_ENTRY_SCORE))
            except Exception:
                entry_floor_eff = None
            decision_meta["entry_floor_eff"] = entry_floor_eff
            decision_meta["min_entry_score"] = MIN_ENTRY_SCORE if MIN_ENTRY_SCORE > 0 else None
            decision_meta["min_entry_notional"] = MIN_ENTRY_NOTIONAL if MIN_ENTRY_NOTIONAL > 0 else None
            decision_meta["min_liq_score"] = MIN_LIQ_SCORE if MIN_LIQ_SCORE > 0 else None
            if not hybrid_only:
                return

            scores = np.asarray(scores_arr, dtype=float)
            if scores.size == 0:
                return

            # --- Entry floor auto-tune ---
            entry_env = os.environ.get("HYBRID_ENTRY_FLOOR", "")
            auto_entry = self._is_auto_env(entry_env) or str(os.environ.get("HYBRID_AUTO_ENTRY", "0")).strip().lower() in ("1", "true", "yes", "on")
            if auto_entry:
                min_samples = int(os.environ.get("HYBRID_ENTRY_MIN_SAMPLES", 10))
                if scores.size >= min_samples:
                    entry_pct = float(os.environ.get("HYBRID_ENTRY_PCT", 75.0))
                    entry_pct = float(max(50.0, min(99.0, entry_pct)))
                    raw_thr = float(np.percentile(scores, entry_pct))
                    half_life = float(os.environ.get("HYBRID_ENTRY_EMA_HALFLIFE_SEC", 600.0))
                    prev = self._hybrid_entry_floor_dyn
                    prev_ts = int(self._hybrid_entry_floor_dyn_ts or ts_ms)
                    dt_sec = max(1.0, (ts_ms - prev_ts) / 1000.0)
                    if half_life <= 0:
                        ema = raw_thr
                    else:
                        alpha = 1.0 - math.exp(-math.log(2.0) * dt_sec / max(half_life, 1e-6))
                        ema = raw_thr if prev is None else (alpha * raw_thr + (1.0 - alpha) * float(prev))
                    self._hybrid_entry_floor_dyn = float(ema)
                    self._hybrid_entry_floor_dyn_ts = int(ts_ms)

            # --- Confidence scale auto-tune ---
            conf_env = os.environ.get("HYBRID_CONF_SCALE", "")
            auto_conf = self._is_auto_env(conf_env) or str(os.environ.get("HYBRID_AUTO_CONF", "0")).strip().lower() in ("1", "true", "yes", "on")
            if auto_conf:
                min_samples = int(os.environ.get("HYBRID_CONF_MIN_SAMPLES", 10))
                if scores.size >= min_samples:
                    conf_target = float(os.environ.get("HYBRID_CONF_TARGET", 0.8))
                    conf_target = float(max(0.55, min(0.95, conf_target)))
                    conf_pct = float(os.environ.get("HYBRID_CONF_SCORE_PCT", 75.0))
                    conf_pct = float(max(50.0, min(99.0, conf_pct)))
                    pos_scores = scores[scores > 0.0]
                    if pos_scores.size >= min_samples:
                        score_ref = float(np.percentile(pos_scores, conf_pct))
                    else:
                        score_ref = float(np.percentile(np.abs(scores), conf_pct))
                    score_ref = float(max(score_ref, 1e-6))
                    x = float(max(1e-6, min(0.999, 2.0 * conf_target - 1.0)))
                    atanh_x = 0.5 * math.log((1.0 + x) / (1.0 - x))
                    scale_raw = score_ref / max(atanh_x, 1e-6)
                    scale_min = float(os.environ.get("HYBRID_CONF_SCALE_MIN", 1e-6))
                    scale_raw = float(max(scale_min, scale_raw))
                    half_life = float(os.environ.get("HYBRID_CONF_EMA_HALFLIFE_SEC", 600.0))
                    prev = self._hybrid_conf_scale_dyn
                    prev_ts = int(self._hybrid_conf_scale_dyn_ts or ts_ms)
                    dt_sec = max(1.0, (ts_ms - prev_ts) / 1000.0)
                    if half_life <= 0:
                        ema = scale_raw
                    else:
                        alpha = 1.0 - math.exp(-math.log(2.0) * dt_sec / max(half_life, 1e-6))
                        ema = scale_raw if prev is None else (alpha * scale_raw + (1.0 - alpha) * float(prev))
                    self._hybrid_conf_scale_dyn = float(ema)
                    self._hybrid_conf_scale_dyn_ts = int(ts_ms)
        except Exception:
            return

    def _should_alert(self, key: str, *, now_ts: float | None = None, throttle_sec: int | float = ALERT_THROTTLE_SEC) -> bool:
        now_ts = time.time() if now_ts is None else float(now_ts)
        last = float(self._last_alert_ts.get(key, 0.0) or 0.0)
        if (now_ts - last) < float(throttle_sec):
            return False
        self._last_alert_ts[key] = now_ts
        return True

    async def _send_telegram(self, message: str):
        if not self._telegram_enabled:
            return
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {"chat_id": self._telegram_chat_id, "text": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status >= 300:
                        self._log_err(f"[ALERT] Telegram send failed: {resp.status}")
        except Exception as e:
            self._log_err(f"[ALERT] Telegram send error: {e}")

    def _enqueue_alert(self, title: str, message: str):
        if not self._telegram_enabled:
            return
        asyncio.create_task(self._send_telegram(f"{title}\n{message}"))

    def _register_anomaly(self, kind: str, severity: str, message: str, data: dict | None = None):
        ts = now_ms()
        entry = {
            "ts": ts,
            "kind": str(kind),
            "severity": str(severity),
            "message": str(message),
            "data": data or {},
        }
        self.anomalies.append(entry)
        if self._should_alert(f"{kind}:{severity}"):
            self._enqueue_alert(f"[ALERT:{severity}] {kind}", message)

    def _note_runtime_error(self, context: str, err_text: str):
        now_ts = time.time()
        if (now_ts - self._last_error_ts) > ERROR_BURST_WINDOW_SEC:
            self._error_burst = 0
        self._last_error_ts = now_ts
        self._error_burst += 1
        self._register_anomaly("runtime_error", "critical", f"{context}: {err_text}")
        if self._error_burst >= ERROR_BURST_LIMIT:
            self.safety_mode = True
            self.enable_orders = False
            self._register_anomaly("safety_mode", "critical", "error burst -> safety mode")

    async def _ccxt_call(self, label: str, fn, *args, semaphore=None, **kwargs):
        """
        Best-effort CCXT call with:
          - concurrency cap (semaphore)
          - retry with exponential backoff + jitter
        """
        for attempt in range(1, MAX_RETRY + 1):
            try:
                async with (semaphore or self._net_sem):
                    return await fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                msg_low = msg.lower()
                is_retryable = any(k in msg for k in [
                    "RequestTimeout", "DDoSProtection", "ExchangeNotAvailable",
                    "NetworkError", "ETIMEDOUT", "ECONNRESET", "502", "503", "504"
                ])
                # Bybit occasionally returns timestamp drift / recv_window errors.
                # Try time-difference sync and retry rather than hard-failing live sizing.
                is_time_skew = (
                    "retcode\":10002" in msg_low
                    or "recv_window" in msg_low
                    or "req_timestamp" in msg_low
                    or "server_timestamp" in msg_low
                )
                if "bybit get https://" in msg_low or "bybit post https://" in msg_low:
                    is_retryable = True
                if is_time_skew:
                    is_retryable = True
                    ex_obj = getattr(fn, "__self__", None)
                    if ex_obj is not None and hasattr(ex_obj, "load_time_difference"):
                        try:
                            await ex_obj.load_time_difference()
                            self._log("[LIVE_BAL] exchange time difference refreshed after timestamp error")
                        except Exception:
                            pass
                if (attempt >= MAX_RETRY) or (not is_retryable):
                    raise
                backoff = (RETRY_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                self._log_err(f"[WARN] {label} retry {attempt}/{MAX_RETRY} err={msg} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)

    async def init_exchange_settings(self):
        try:
            await self.exchange.load_markets()
        except Exception:
            pass
        await self._sync_position_mode()
        try:
            sync_global = str(os.environ.get("SYNC_GLOBAL_LEVERAGE_ON_START", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            sync_global = False
        if sync_global:
            await self._sync_leverage()
        else:
            self._log("[EXCHANGE] global leverage sync disabled; per-order leverage only")

    async def _sync_position_mode(self):
        if not self.enable_orders:
            self._log("[EXCHANGE] live orders disabled; skipping position mode sync")
            return
        if not getattr(self.exchange, "apiKey", None) or not getattr(self.exchange, "secret", None):
            self._log("[EXCHANGE] apiKey/secret missing; skipping position mode sync")
            return
        if not hasattr(self.exchange, "fetch_position_mode"):
            return
        try:
            mode = await self.exchange.fetch_position_mode()
            hedge = False
            if isinstance(mode, bool):
                hedge = bool(mode)
            elif isinstance(mode, str):
                hedge = mode.lower() in ("hedge", "both", "dual")
            elif isinstance(mode, dict):
                if "hedgeMode" in mode:
                    hedge = bool(mode.get("hedgeMode"))
                elif "positionMode" in mode:
                    hedge = str(mode.get("positionMode")).lower() in ("hedge", "both", "dual")
                elif "mode" in mode:
                    hedge = str(mode.get("mode")).lower() in ("hedge", "both", "dual")
            self._is_hedge_mode = bool(hedge)
            self._position_mode = "hedge" if self._is_hedge_mode else "oneway"
            self._log(f"[EXCHANGE] position mode: {self._position_mode}")
        except Exception as e:
            self._log_err(f"[EXCHANGE] fetch_position_mode failed: {e}")

    async def _sync_leverage(self):
        if not self.enable_orders:
            self._log("[EXCHANGE] live orders disabled; skipping leverage sync")
            return
        if not getattr(self.exchange, "apiKey", None) or not getattr(self.exchange, "secret", None):
            self._log("[EXCHANGE] apiKey/secret missing; skipping leverage sync")
            return
        if not hasattr(self.exchange, "set_leverage"):
            return
        target = float(self.leverage)
        for sym in SYMBOLS:
            try:
                await self._ccxt_call("set_leverage", self.exchange.set_leverage, int(target), sym)
                self._log(f"[EXCHANGE] leverage synced: {sym} -> {target:.1f}x")
            except Exception as e:
                self._log_err(f"[EXCHANGE] set_leverage failed: {sym} target={target:.1f}x err={e}")

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def _migrate_legacy_state_files(self) -> None:
        """Migrate legacy JSON state files (non-mode) into mode-specific files if missing."""
        legacy = {
            "equity": self.state_dir / "equity_history.json",
            "trade": self.state_dir / "trade_tape.json",
            "eval": self.state_dir / "eval_history.json",
            "positions": self.state_dir / "positions.json",
            "balance": self.state_dir / "balance.json",
        }
        for key, legacy_path in legacy.items():
            new_path = self.state_files.get(key)
            if not new_path:
                continue
            if new_path.exists():
                continue
            if legacy_path.exists():
                try:
                    shutil.copy2(legacy_path, new_path)
                    self._log(f"[STATE_MIGRATE] {legacy_path.name} -> {new_path.name}")
                except Exception as e:
                    self._log_err(f"[ERR] migrate state {legacy_path.name}: {e}")

    def _db_has_rows(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with sqlite3.connect(str(path)) as conn:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]
                for tbl in tables:
                    if not tbl or tbl.startswith("sqlite_"):
                        continue
                    try:
                        row = conn.execute(f"SELECT 1 FROM {tbl} LIMIT 1").fetchone()
                        if row is not None:
                            return True
                    except Exception:
                        continue
        except Exception:
            return True
        return False

    def _legacy_db_paths(self) -> list[Path]:
        paths: list[Path] = []
        legacy_dir = Path("/tmp/codex_quant_db")
        if legacy_dir.exists():
            for name in ("bot_data.db", "bot_data_live.db", "bot_data_paper.db"):
                p = legacy_dir / name
                if p.exists():
                    paths.append(p)
        return paths

    def _table_has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
            return any(r[1] == column for r in cur.fetchall())
        except Exception:
            return False

    def _copy_legacy_table(
        self,
        dest_conn: sqlite3.Connection,
        table: str,
        *,
        where: str | None = None,
    ) -> None:
        cols = []
        try:
            cur = dest_conn.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cur.fetchall()]
        except Exception:
            cols = []
        if not cols:
            return
        col_list = ",".join(cols)
        sql = f"INSERT OR IGNORE INTO {table} ({col_list}) SELECT {col_list} FROM legacy.{table}"
        if where:
            sql += f" WHERE {where}"
        dest_conn.execute(sql)

    def _copy_from_legacy_db(self, legacy_path: Path, live_path: Path, paper_path: Path) -> None:
        try:
            legacy_conn = sqlite3.connect(str(legacy_path))
        except Exception as e:
            self._log_err(f"[DB_MIGRATE] open legacy failed: {legacy_path} err={e}")
            return
        try:
            cur = legacy_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            legacy_tables = [r[0] for r in cur.fetchall() if r and r[0] and not r[0].startswith("sqlite_")]
        except Exception:
            legacy_tables = []

        name = legacy_path.name.lower()
        if "paper" in name:
            targets = [("paper", paper_path)]
        elif "live" in name:
            targets = [("live", live_path)]
        else:
            targets = [("live", live_path), ("paper", paper_path)]

        for mode, dest_path in targets:
            try:
                dest_conn = sqlite3.connect(str(dest_path))
                dest_conn.execute(f"ATTACH DATABASE '{str(legacy_path)}' AS legacy")
                for tbl in legacy_tables:
                    has_mode = self._table_has_column(legacy_conn, tbl, "trading_mode")
                    if not has_mode:
                        if tbl == "slippage_analysis" and mode != "live":
                            continue
                        self._copy_legacy_table(dest_conn, tbl)
                        continue
                    # Filter by trading_mode when present
                    self._copy_legacy_table(dest_conn, tbl, where=f"trading_mode='{mode}'")
                dest_conn.commit()
                dest_conn.execute("DETACH DATABASE legacy")
                dest_conn.close()
            except Exception as e:
                self._log_err(f"[DB_MIGRATE] copy failed: {legacy_path} -> {dest_path} err={e}")
        try:
            legacy_conn.close()
        except Exception:
            pass

    def _maybe_migrate_legacy_db(self) -> None:
        marker = self.state_dir / "db_migration_v1.done"
        if marker.exists():
            return
        legacy_paths = self._legacy_db_paths()
        if not legacy_paths:
            return
        live_path = self.state_dir / "bot_data_live.db"
        paper_path = self.state_dir / "bot_data_paper.db"

        force = str(os.environ.get("DB_MIGRATE_FORCE", "0")).strip().lower() in ("1", "true", "yes", "on")
        live_has = self._db_has_rows(live_path)
        paper_has = self._db_has_rows(paper_path)
        if (live_has or paper_has) and not force:
            return

        # Ensure destination DBs exist with schema
        try:
            DatabaseManager(db_path=str(live_path))
            DatabaseManager(db_path=str(paper_path))
        except Exception as e:
            self._log_err(f"[DB_MIGRATE] init dest failed: {e}")
            return

        for legacy in legacy_paths:
            self._log(f"[DB_MIGRATE] copying from {legacy} -> state/")
            self._copy_from_legacy_db(legacy, live_path, paper_path)

        try:
            marker.write_text(json.dumps({"ts": now_ms(), "legacy": [str(p) for p in legacy_paths]}))
        except Exception:
            pass

    def _load_json(self, path: Path, default):
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            try:
                self._state_json_load_errors.add(str(path))
            except Exception:
                pass
            self._log_err(f"[ERR] load {path.name}: {e}")
        return default

    def _write_json_atomic(self, path: Path, payload) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(path)

    def _is_exit_trade_record(self, rec: dict | None) -> bool:
        if not isinstance(rec, dict):
            return False
        ttype = str(rec.get("ttype") or rec.get("type") or rec.get("action") or "").upper()
        return ttype in self._exit_trade_types

    def _refresh_closed_trade_counter(self) -> None:
        tape_count = 0
        db_count = 0
        try:
            tape_count = int(sum(1 for r in (self.trade_tape or []) if self._is_exit_trade_record(r)))
        except Exception:
            tape_count = 0
        try:
            if hasattr(self, "db") and self.db is not None:
                db_count = int(self.db.count_closed_trades(mode=self._trading_mode))
        except Exception:
            db_count = 0
        self._closed_trade_count = int(max(tape_count, db_count))

    def _load_reval_baseline(self, force: bool = False) -> None:
        ts_ms = now_ms()
        if not force:
            try:
                if ts_ms - int(self._reval_baseline_last_load_ms or 0) < int(max(1.0, self._reval_baseline_reload_sec) * 1000.0):
                    return
            except Exception:
                pass
        self._reval_baseline_last_load_ms = int(ts_ms)
        baseline = 0
        try:
            p = self._reval_baseline_path
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                baseline = int((payload or {}).get("baseline_closed", 0) or 0)
        except Exception:
            baseline = 0
        self._reval_baseline_closed = max(0, int(baseline))

    def _load_new_exit_anchor(self, force: bool = False) -> None:
        if self._new_exit_anchor_loaded and not force:
            return
        anchor = None
        try:
            p = self._new_exit_anchor_path
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    anchor = int((payload or {}).get("anchor_closed", 0) or 0)
        except Exception:
            anchor = None
        if anchor is None:
            try:
                anchor = int(self._reval_baseline_closed or 0)
            except Exception:
                anchor = 0
        try:
            closed_total = int(self._closed_trade_count or 0)
        except Exception:
            closed_total = 0
        anchor = max(0, min(int(anchor), int(max(0, closed_total))))
        self._new_exit_anchor_closed = int(anchor)
        self._new_exit_anchor_loaded = True
        try:
            if not self._new_exit_anchor_path.exists():
                payload = {
                    "anchor_closed": int(anchor),
                    "anchor_set_ts": int(now_ms()),
                    "note": "NewExit cumulative anchor (independent from rolling reval baseline)",
                }
                self._write_json_atomic(self._new_exit_anchor_path, payload)
        except Exception:
            pass

    def _load_alpha_train_state(self, force: bool = False) -> None:
        path = self._alpha_train_state_path
        if path is None:
            return
        closed_now = max(0, int(self._closed_trade_count or 0))
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    raw_closed = int((payload or {}).get("last_exit_closed", closed_now) or closed_now)
                    self._alpha_train_last_exit_closed = max(0, min(int(raw_closed), int(closed_now)))
                    raw_ms = int((payload or {}).get("last_train_ms", 0) or 0)
                    if raw_ms > 0:
                        self._alpha_train_last_train_ms = max(int(self._alpha_train_last_train_ms or 0), int(raw_ms))
                    return
            except Exception:
                pass
        self._alpha_train_last_exit_closed = int(closed_now)
        if force:
            self._persist_alpha_train_state(reason="bootstrap")

    def _persist_alpha_train_state(self, reason: str | None = None) -> None:
        path = self._alpha_train_state_path
        if path is None:
            return
        payload = {
            "last_exit_closed": int(max(0, int(self._alpha_train_last_exit_closed or 0))),
            "last_train_ms": int(max(0, int(self._alpha_train_last_train_ms or 0))),
            "reason": str(reason or ""),
            "saved_ms": int(now_ms()),
        }
        try:
            self._write_json_atomic(path, payload)
        except Exception:
            pass

    def _reval_status(self, ts_ms: int | None = None) -> dict:
        try:
            self._load_reval_baseline(force=False)
        except Exception:
            pass
        try:
            self._load_new_exit_anchor(force=False)
        except Exception:
            pass
        baseline = max(0, int(self._reval_baseline_closed or 0))
        closed_total = max(0, int(self._closed_trade_count or 0))
        target_min = max(1, int(self._reval_target_min or 200))
        target_max = max(target_min, int(self._reval_target_max or max(300, target_min)))
        batch_new_total = max(0, closed_total - baseline)
        remaining = max(0, target_min - batch_new_total)
        anchor_closed = max(0, min(int(self._new_exit_anchor_closed or 0), int(closed_total)))
        cumulative_new_total = max(0, int(closed_total) - int(anchor_closed))
        return {
            "baseline_closed": int(baseline),
            "closed_total": int(closed_total),
            "anchor_closed": int(anchor_closed),
            # Keep legacy field for compatibility (batch progress from rolling baseline).
            "new_closed_total": int(batch_new_total),
            "new_closed_total_batch": int(batch_new_total),
            # Non-resetting cumulative counter for dashboard/operator visibility.
            "new_closed_total_cum": int(cumulative_new_total),
            "target_min": int(target_min),
            "target_max": int(target_max),
            "remaining_to_min": int(remaining),
            "ready": bool(batch_new_total >= target_min),
            "timestamp": int(ts_ms or now_ms()),
        }

    def _load_auto_reval_status(self, force: bool = False) -> None:
        ts_ms = now_ms()
        if not force:
            try:
                if ts_ms - int(self._auto_reval_status_last_load_ms or 0) < int(max(1.0, self._auto_reval_status_reload_sec) * 1000.0):
                    return
            except Exception:
                pass
        self._auto_reval_status_last_load_ms = int(ts_ms)
        payload: dict | None = None
        try:
            p = self._auto_reval_status_path
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    payload = obj
        except Exception:
            payload = None
        if isinstance(payload, dict):
            self._auto_reval_status_cache = payload

    def _auto_reval_status(self, ts_ms: int | None = None) -> dict:
        try:
            self._load_auto_reval_status(force=False)
        except Exception:
            pass
        def _to_int(v, default: int = 0) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)
        raw = self._auto_reval_status_cache if isinstance(self._auto_reval_status_cache, dict) else {}
        cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
        summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
        summary_kpi = summary.get("batch_kpi") if isinstance(summary.get("batch_kpi"), dict) else {}
        summary_kpi_delta = summary.get("batch_kpi_delta") if isinstance(summary.get("batch_kpi_delta"), dict) else {}
        runs = raw.get("runs") if isinstance(raw.get("runs"), list) else []
        progress = raw.get("progress") if isinstance(raw.get("progress"), dict) else {}
        reports_total = 3
        reports_ready = 0
        for r in runs:
            if not isinstance(r, dict):
                continue
            try:
                if int(r.get("code", -1)) == 0:
                    reports_ready += 1
            except Exception:
                continue
        reports_ready = max(0, min(reports_total, int(reports_ready)))

        target_new = _to_int(cfg.get("target_new"), int(self._reval_target_min or 200))
        new_closed = _to_int(raw.get("new_closed_total"), 0)
        new_closed_cum = _to_int(raw.get("new_closed_total_cum"), _to_int(progress.get("processed_new_closed_total"), new_closed))
        remaining = _to_int(raw.get("remaining_to_target"), max(0, int(target_new) - int(new_closed)))
        completed_batches = _to_int(progress.get("completed_batches"), 0)
        completed_reports_total = _to_int(progress.get("completed_reports_total"), 0)
        status_ts = _to_int(raw.get("heartbeat_ts_ms"), _to_int(raw.get("timestamp_ms"), 0))
        now_ts = int(ts_ms or now_ms())
        stale_sec = None
        if status_ts > 0:
            stale_sec = max(0.0, (float(now_ts) - float(status_ts)) / 1000.0)

        return {
            "exists": bool(self._auto_reval_status_path.exists()),
            "ready": bool(raw.get("ready") is True),
            "reports_ready": int(reports_ready),
            "reports_total": int(reports_total),
            "new_closed_total": int(new_closed),
            "new_closed_total_cum": int(max(0, new_closed_cum)),
            "target_new": int(max(1, target_new)),
            "remaining_to_target": int(max(0, remaining)),
            "timeout": bool(raw.get("timeout") is True),
            "baseline_closed": _to_int(raw.get("baseline_closed"), 0),
            "closed_total": _to_int(raw.get("closed_total"), 0),
            "completed_batches": int(max(0, completed_batches)),
            "completed_reports_total": int(max(0, completed_reports_total)),
            "direction_hit": self._safe_float(summary_kpi.get("direction_hit"), None),
            "entry_issue_ratio": self._safe_float(summary_kpi.get("entry_issue_ratio"), None),
            "avg_exit_regret": self._safe_float(summary_kpi.get("avg_exit_regret"), None),
            "direction_hit_delta": self._safe_float(summary_kpi_delta.get("direction_hit"), None),
            "entry_issue_ratio_delta": self._safe_float(summary_kpi_delta.get("entry_issue_ratio"), None),
            "avg_exit_regret_delta": self._safe_float(summary_kpi_delta.get("avg_exit_regret"), None),
            "timestamp": int(status_ts if status_ts > 0 else now_ts),
            "stale_sec": stale_sec,
        }

    def _normalize_auto_tune_env_value(self, value):
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
                return f"{value:.10g}"
            return f"{value:.8f}".rstrip("0").rstrip(".")
        if value is None:
            return None
        txt = str(value).strip()
        return txt if txt else None

    def _maybe_reload_auto_tune_overrides(self, ts_ms: int) -> None:
        try:
            interval_ms = int(max(1.0, float(self._auto_tune_reload_sec)) * 1000.0)
        except Exception:
            interval_ms = 5000
        if int(ts_ms) - int(self._auto_tune_last_reload_ms or 0) < int(interval_ms):
            return
        self._auto_tune_last_reload_ms = int(ts_ms)

        # status cache (for dashboard)
        try:
            st = self._auto_tune_status_path
            if st.exists():
                st_m = float(st.stat().st_mtime)
                if self._auto_tune_status_mtime is None or st_m > float(self._auto_tune_status_mtime):
                    with st.open("r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        self._auto_tune_status_cache = obj
                        self._auto_tune_status_mtime = st_m
                        try:
                            b_id = int((obj or {}).get("batch_id", 0) or 0)
                            if b_id > int(self._auto_tune_last_batch_id or 0):
                                self._auto_tune_last_batch_id = int(b_id)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            p = self._auto_tune_override_path
            if not p.exists():
                return
            mt = float(p.stat().st_mtime)
            if self._auto_tune_overrides_mtime is not None and mt <= float(self._auto_tune_overrides_mtime):
                return
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            payload = raw if isinstance(raw, dict) else {}
            ov = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else payload
            if not isinstance(ov, dict):
                return
            applied = 0
            cleaned: dict[str, object] = {}
            for k, v in ov.items():
                key = str(k or "").strip().upper()
                if not key:
                    continue
                if not all(ch.isalnum() or ch == "_" for ch in key):
                    continue
                val_txt = self._normalize_auto_tune_env_value(v)
                if val_txt is None:
                    continue
                os.environ[key] = str(val_txt)
                cleaned[key] = v
                applied += 1
            self._auto_tune_overrides_cache = cleaned
            self._auto_tune_overrides_mtime = mt
            self._auto_tune_last_apply_ms = int(ts_ms)
            if applied > 0:
                try:
                    batch_id = int((payload or {}).get("batch_id", 0) or 0)
                except Exception:
                    batch_id = 0
                if batch_id > int(self._auto_tune_last_batch_id or 0):
                    self._auto_tune_last_batch_id = int(batch_id)
                self._log(
                    f"[AUTO_TUNE] overrides applied: n={applied} "
                    f"batch={batch_id} file={str(p)}"
                )
        except Exception as e:
            self._log_err(f"[AUTO_TUNE] override reload failed: {e}")

    def _auto_tune_status(self, ts_ms: int | None = None) -> dict:
        now_ts = int(ts_ms or now_ms())
        raw = self._auto_tune_status_cache if isinstance(self._auto_tune_status_cache, dict) else {}
        retrain = raw.get("retrain") if isinstance(raw.get("retrain"), dict) else {}
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        try:
            stale_sec = None
            r_ts = int(raw.get("timestamp_ms", 0) or 0)
            if r_ts > 0:
                stale_sec = max(0.0, (float(now_ts) - float(r_ts)) / 1000.0)
        except Exception:
            stale_sec = None
        return {
            "enabled": bool(int(os.environ.get("AUTO_TUNE_ENABLED", "1") or 1) == 1),
            "exists": bool(self._auto_tune_override_path.exists()),
            "override_count": int(len(self._auto_tune_overrides_cache or {})),
            "last_apply_ms": int(self._auto_tune_last_apply_ms or 0),
            "last_batch_id": int(self._auto_tune_last_batch_id or 0),
            "actions": (raw.get("actions") if isinstance(raw.get("actions"), list) else []),
            "retrain_requested": bool(retrain.get("requested") is True),
            "retrain_ran": bool(retrain.get("ran") is True),
            "retrain_rc": retrain.get("rc"),
            "retrain_reason": str(retrain.get("reason") or ""),
            "entry_issue_rate": self._safe_float(metrics.get("entry_issue_rate"), None),
            "entry_issue_ratio": self._safe_float(metrics.get("entry_issue_ratio"), self._safe_float(metrics.get("entry_issue_rate"), None)),
            "miss_rate": self._safe_float(metrics.get("miss_rate"), None),
            "direction_hit": self._safe_float(metrics.get("direction_hit"), None),
            "avg_exit_regret": self._safe_float(metrics.get("avg_exit_regret"), None),
            "direction_hit_delta": self._safe_float(metrics.get("direction_hit_delta"), None),
            "entry_issue_ratio_delta": self._safe_float(metrics.get("entry_issue_ratio_delta"), None),
            "avg_exit_regret_delta": self._safe_float(metrics.get("avg_exit_regret_delta"), None),
            "early_like_rate": self._safe_float(metrics.get("early_like_rate"), None),
            "event_mc_exit_rate": self._safe_float(metrics.get("event_mc_exit_rate"), None),
            "liq_like_rate": self._safe_float(metrics.get("liq_like_rate"), None),
            "avg_roe": self._safe_float(metrics.get("avg_roe"), None),
            "timestamp": int(raw.get("timestamp_ms", 0) or 0),
            "stale_sec": stale_sec,
        }

    @staticmethod
    def _coerce_opt_bool(val):
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, float)):
            try:
                return bool(int(val))
            except Exception:
                return None
        if isinstance(val, str):
            txt = str(val).strip().lower()
            if txt in ("1", "true", "yes", "on", "y", "t"):
                return True
            if txt in ("0", "false", "no", "off", "n", "f"):
                return False
        return None

    def _is_event_exit_reason(self, reason: str | None) -> bool:
        try:
            r = str(reason or "").strip().lower()
            return (
                r.startswith("event_mc_exit")
                or r.startswith("mc_exit:")
                or r.startswith("event_exit:")
            )
        except Exception:
            return False

    def _new_event_alignment_counter(self) -> dict[str, object]:
        return {
            "samples": 0,
            "known_samples": 0,
            "aligned": 0,
            "blocked_samples": 0,
            "blocked_to_event_exit": 0,
            "safe_samples": 0,
            "safe_to_non_event_exit": 0,
            "strict_mode_samples": 0,
            "latest_reason": None,
            "latest_symbol": None,
            "latest_ts_ms": None,
            "latest_trade_id": None,
        }

    def _update_event_alignment_counter(self, counter: dict[str, object], sample: dict[str, object]) -> None:
        if not isinstance(counter, dict) or not isinstance(sample, dict):
            return
        try:
            counter["samples"] = int(counter.get("samples", 0) or 0) + 1
        except Exception:
            counter["samples"] = 1
        known = self._coerce_opt_bool(sample.get("entry_event_exit_ok"))
        if known is not None:
            counter["known_samples"] = int(counter.get("known_samples", 0) or 0) + 1
            if sample.get("aligned") is True:
                counter["aligned"] = int(counter.get("aligned", 0) or 0) + 1
            if known is False:
                counter["blocked_samples"] = int(counter.get("blocked_samples", 0) or 0) + 1
                if bool(sample.get("is_event_exit")):
                    counter["blocked_to_event_exit"] = int(counter.get("blocked_to_event_exit", 0) or 0) + 1
            elif known is True:
                counter["safe_samples"] = int(counter.get("safe_samples", 0) or 0) + 1
                if not bool(sample.get("is_event_exit")):
                    counter["safe_to_non_event_exit"] = int(counter.get("safe_to_non_event_exit", 0) or 0) + 1
        if bool(sample.get("strict_mode")):
            counter["strict_mode_samples"] = int(counter.get("strict_mode_samples", 0) or 0) + 1
        counter["latest_reason"] = sample.get("reason")
        counter["latest_symbol"] = sample.get("symbol")
        counter["latest_ts_ms"] = sample.get("ts_ms")
        counter["latest_trade_id"] = sample.get("trade_id")

    def _event_alignment_counter_to_status(self, counter: dict[str, object]) -> dict[str, object]:
        if not isinstance(counter, dict):
            counter = {}
        n_total = int(counter.get("samples", 0) or 0)
        n_known = int(counter.get("known_samples", 0) or 0)
        n_aligned = int(counter.get("aligned", 0) or 0)
        blocked_n = int(counter.get("blocked_samples", 0) or 0)
        blocked_to_event_n = int(counter.get("blocked_to_event_exit", 0) or 0)
        safe_n = int(counter.get("safe_samples", 0) or 0)
        safe_to_non_event_n = int(counter.get("safe_to_non_event_exit", 0) or 0)
        strict_true_n = int(counter.get("strict_mode_samples", 0) or 0)
        align_rate = (float(n_aligned) / float(n_known)) if n_known > 0 else None
        blocked_rate = (float(blocked_to_event_n) / float(blocked_n)) if blocked_n > 0 else None
        safe_rate = (float(safe_to_non_event_n) / float(safe_n)) if safe_n > 0 else None
        return {
            "samples": int(n_total),
            "known_samples": int(n_known),
            "aligned": int(n_aligned),
            "alignment_rate": align_rate,
            "blocked_samples": int(blocked_n),
            "blocked_to_event_exit": int(blocked_to_event_n),
            "blocked_to_event_exit_rate": blocked_rate,
            "safe_samples": int(safe_n),
            "safe_to_non_event_exit": int(safe_to_non_event_n),
            "safe_to_non_event_exit_rate": safe_rate,
            "strict_mode_samples": int(strict_true_n),
            "latest_reason": counter.get("latest_reason"),
            "latest_symbol": counter.get("latest_symbol"),
            "latest_ts_ms": counter.get("latest_ts_ms"),
            "latest_trade_id": counter.get("latest_trade_id"),
        }

    def _latest_trade_id_for_mode(self) -> int:
        db_path = str(getattr(self, "_db_path", "") or "")
        if not db_path:
            return 0
        try:
            mode_val = getattr(self._trading_mode, "value", None)
        except Exception:
            mode_val = None
        query = "SELECT MAX(id) FROM trades"
        params: list[object] = []
        if mode_val:
            query += " WHERE trading_mode = ?"
            params.append(mode_val)
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(query, params)
                row = cur.fetchone()
            return int((row[0] if row else 0) or 0)
        except Exception:
            return 0

    def _load_event_alignment_anchor(self, force: bool = False) -> None:
        if self._event_alignment_anchor_loaded and not force:
            return
        anchor_ts = None
        anchor_trade_id = None
        try:
            env_ts = int(os.environ.get("EVENT_EXIT_ALIGNMENT_ANCHOR_TS_MS", 0) or 0)
        except Exception:
            env_ts = 0
        try:
            env_id = int(os.environ.get("EVENT_EXIT_ALIGNMENT_ANCHOR_TRADE_ID", 0) or 0)
        except Exception:
            env_id = 0
        try:
            p = self._event_alignment_anchor_path
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    anchor_ts = int((payload or {}).get("anchor_ts_ms", 0) or 0)
                    anchor_trade_id = int((payload or {}).get("anchor_trade_id", 0) or 0)
        except Exception:
            anchor_ts = None
            anchor_trade_id = None
        if env_ts > 0:
            anchor_ts = int(env_ts)
        if env_id > 0:
            anchor_trade_id = int(env_id)
        if anchor_ts is None or int(anchor_ts) <= 0:
            anchor_ts = int(now_ms())
        if anchor_trade_id is None or int(anchor_trade_id) < 0:
            anchor_trade_id = 0
        if int(anchor_trade_id) == 0:
            anchor_trade_id = int(max(0, self._latest_trade_id_for_mode()))
        self._event_alignment_anchor_ts_ms = int(max(0, int(anchor_ts)))
        self._event_alignment_anchor_trade_id = int(max(0, int(anchor_trade_id)))
        self._event_alignment_anchor_loaded = True
        self._event_alignment_since_anchor = self._new_event_alignment_counter()
        try:
            payload = {
                "anchor_ts_ms": int(self._event_alignment_anchor_ts_ms),
                "anchor_trade_id": int(self._event_alignment_anchor_trade_id),
                "saved_ms": int(now_ms()),
                "mode": getattr(self._trading_mode, "value", None),
                "note": "event alignment split anchor (post-patch new EXIT window)",
            }
            self._write_json_atomic(self._event_alignment_anchor_path, payload)
        except Exception:
            pass

    def _sample_is_since_alignment_anchor(self, sample: dict[str, object]) -> bool:
        if not self._event_alignment_anchor_loaded:
            return False
        if not isinstance(sample, dict):
            return False
        aid = int(max(0, int(self._event_alignment_anchor_trade_id or 0)))
        ats = int(max(0, int(self._event_alignment_anchor_ts_ms or 0)))
        sid = None
        sts = None
        try:
            if sample.get("trade_id") is not None:
                sid = int(sample.get("trade_id"))
        except Exception:
            sid = None
        try:
            if sample.get("ts_ms") is not None:
                sts = int(sample.get("ts_ms"))
        except Exception:
            sts = None
        if aid > 0 and sid is not None:
            return bool(int(sid) > int(aid))
        if ats > 0 and sts is not None:
            return bool(int(sts) >= int(ats))
        if sid is not None and aid <= 0:
            return True
        if sts is not None and ats <= 0:
            return True
        return False

    def _bootstrap_event_alignment_since_anchor(self, force: bool = False) -> None:
        if not force and isinstance(self._event_alignment_since_anchor, dict) and int(self._event_alignment_since_anchor.get("samples", 0) or 0) > 0:
            return
        if not hasattr(self, "db") or self.db is None:
            return
        if not self._event_alignment_anchor_loaded:
            self._load_event_alignment_anchor(force=False)
        self._event_alignment_since_anchor = self._new_event_alignment_counter()
        try:
            mode_val = getattr(self._trading_mode, "value", None)
        except Exception:
            mode_val = None
        aid = int(max(0, int(self._event_alignment_anchor_trade_id or 0)))
        ats = int(max(0, int(self._event_alignment_anchor_ts_ms or 0)))
        try:
            boot_limit = int(os.environ.get("EVENT_EXIT_ALIGNMENT_ANCHOR_BOOTSTRAP_LIMIT", 200000) or 200000)
        except Exception:
            boot_limit = 200000
        boot_limit = max(1000, int(boot_limit))
        db_path = str(getattr(self, "_db_path", "") or "")
        if not db_path:
            return
        q = (
            "SELECT id, symbol, entry_reason, timestamp_ms, raw_data "
            "FROM trades WHERE action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')"
        )
        params: list[object] = []
        if mode_val:
            q += " AND trading_mode = ?"
            params.append(mode_val)
        if aid > 0 and ats > 0:
            q += " AND (id > ? OR timestamp_ms >= ?)"
            params.extend([aid, ats])
        elif aid > 0:
            q += " AND id > ?"
            params.append(aid)
        elif ats > 0:
            q += " AND timestamp_ms >= ?"
            params.append(ats)
        q += " ORDER BY timestamp_ms ASC LIMIT ?"
        params.append(boot_limit)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(q, params)
                rows = cur.fetchall()
        except Exception:
            rows = []
        for row in rows or []:
            r = dict(row) if row is not None else {}
            raw = {}
            try:
                raw_txt = r.get("raw_data")
                if raw_txt:
                    raw = json.loads(raw_txt) if isinstance(raw_txt, str) else (raw_txt if isinstance(raw_txt, dict) else {})
            except Exception:
                raw = {}
            known = self._coerce_opt_bool(raw.get("entry_event_exit_ok"))
            is_event_exit = self._is_event_exit_reason(r.get("entry_reason"))
            aligned = None
            if known is not None:
                aligned = (not bool(known)) == bool(is_event_exit)
            sample = {
                "ts_ms": int(r.get("timestamp_ms") or 0),
                "symbol": str(r.get("symbol") or ""),
                "reason": str(r.get("entry_reason") or ""),
                "entry_event_exit_ok": known,
                "is_event_exit": bool(is_event_exit),
                "aligned": aligned,
                "event_mode": str(raw.get("event_exit_mode") or ""),
                "event_hit": self._coerce_opt_bool(raw.get("event_exit_hit")),
                "strict_mode": self._coerce_opt_bool(raw.get("event_exit_strict_mode")),
                "trade_id": int(r.get("id") or 0),
                "source": "bootstrap_since_anchor",
            }
            self._update_event_alignment_counter(self._event_alignment_since_anchor, sample)

    def _persist_event_alignment_report(self, force: bool = False, status: dict | None = None) -> None:
        ts_ms = int(now_ms())
        if not force:
            try:
                if ts_ms - int(self._event_alignment_report_last_write_ms or 0) < int(float(self._event_alignment_report_write_sec) * 1000.0):
                    return
            except Exception:
                pass
        if not isinstance(status, dict):
            overall_counter = self._new_event_alignment_counter()
            for s in list(self._event_alignment_samples):
                self._update_event_alignment_counter(overall_counter, s)
            status = {
                "enabled": bool(self._event_exit_strict_mode_enabled()),
                "window": int(self._event_alignment_window),
                **self._event_alignment_counter_to_status(overall_counter),
                "since_anchor": self._event_alignment_counter_to_status(self._event_alignment_since_anchor),
                "anchor_ts_ms": int(self._event_alignment_anchor_ts_ms or 0),
                "anchor_trade_id": int(self._event_alignment_anchor_trade_id or 0),
            }
        payload = {
            "updated_ms": int(ts_ms),
            "mode": getattr(self._trading_mode, "value", None),
            "enabled": bool(status.get("enabled")),
            "window": int(status.get("window", self._event_alignment_window) or self._event_alignment_window),
            "anchor": {
                "anchor_ts_ms": int(status.get("anchor_ts_ms", self._event_alignment_anchor_ts_ms) or 0),
                "anchor_trade_id": int(status.get("anchor_trade_id", self._event_alignment_anchor_trade_id) or 0),
            },
            "overall_window": {
                k: status.get(k)
                for k in (
                    "samples",
                    "known_samples",
                    "aligned",
                    "alignment_rate",
                    "blocked_samples",
                    "blocked_to_event_exit",
                    "blocked_to_event_exit_rate",
                    "safe_samples",
                    "safe_to_non_event_exit",
                    "safe_to_non_event_exit_rate",
                    "strict_mode_samples",
                    "latest_reason",
                    "latest_symbol",
                    "latest_ts_ms",
                    "latest_trade_id",
                )
            },
            "since_anchor": dict(status.get("since_anchor") or {}),
        }
        try:
            self._write_json_atomic(self._event_alignment_report_path, payload)
            self._event_alignment_report_last_write_ms = int(ts_ms)
        except Exception:
            pass

    def _append_event_alignment_sample(
        self,
        *,
        reason: str | None,
        entry_event_exit_ok,
        symbol: str | None = None,
        ts_ms: int | None = None,
        event_mode: str | None = None,
        event_hit=None,
        strict_mode=None,
        trade_id: int | None = None,
        source: str | None = None,
    ) -> None:
        known = self._coerce_opt_bool(entry_event_exit_ok)
        is_event_exit = self._is_event_exit_reason(reason)
        aligned = None
        if known is not None:
            aligned = (not bool(known)) == bool(is_event_exit)
        rec = {
            "ts_ms": int(ts_ms or now_ms()),
            "symbol": str(symbol or ""),
            "reason": str(reason or ""),
            "entry_event_exit_ok": known,
            "is_event_exit": bool(is_event_exit),
            "aligned": aligned,
            "event_mode": str(event_mode or ""),
            "event_hit": self._coerce_opt_bool(event_hit),
            "strict_mode": self._coerce_opt_bool(strict_mode),
            "trade_id": None,
            "source": str(source or ""),
        }
        try:
            if trade_id is not None:
                rec["trade_id"] = int(trade_id)
        except Exception:
            rec["trade_id"] = None
        self._event_alignment_samples.append(rec)
        if self._sample_is_since_alignment_anchor(rec):
            self._update_event_alignment_counter(self._event_alignment_since_anchor, rec)

    def _bootstrap_event_alignment_samples(self, force: bool = False) -> None:
        if (not force) and self._event_alignment_samples:
            return
        if not hasattr(self, "db") or self.db is None:
            return
        try:
            limit = int(max(100, self._event_alignment_window * 3))
            rows = self.db.get_recent_trades(limit=limit, mode=self._trading_mode)
        except Exception:
            return
        if not isinstance(rows, list) or not rows:
            return
        self._event_alignment_samples.clear()
        exit_types = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}
        for row in rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").upper()
            if action not in exit_types:
                continue
            raw = {}
            try:
                raw_txt = row.get("raw_data")
                if raw_txt:
                    raw = json.loads(raw_txt) if isinstance(raw_txt, str) else (raw_txt if isinstance(raw_txt, dict) else {})
            except Exception:
                raw = {}
            entry_event_ok = raw.get("entry_event_exit_ok")
            event_mode = raw.get("event_exit_mode")
            event_hit = raw.get("event_exit_hit")
            strict_mode = raw.get("event_exit_strict_mode")
            self._append_event_alignment_sample(
                reason=row.get("entry_reason"),
                entry_event_exit_ok=entry_event_ok,
                symbol=row.get("symbol"),
                ts_ms=row.get("timestamp_ms"),
                event_mode=event_mode,
                event_hit=event_hit,
                strict_mode=strict_mode,
                trade_id=row.get("id"),
                source="bootstrap_window",
            )

    def _event_alignment_status(self) -> dict:
        if not self._event_alignment_samples:
            self._bootstrap_event_alignment_samples(force=False)
        if not self._event_alignment_anchor_loaded:
            self._load_event_alignment_anchor(force=False)
        if not isinstance(self._event_alignment_since_anchor, dict) or not self._event_alignment_since_anchor:
            self._bootstrap_event_alignment_since_anchor(force=True)
        samples = list(self._event_alignment_samples)
        overall_counter = self._new_event_alignment_counter()
        for s in samples:
            self._update_event_alignment_counter(overall_counter, s)
        overall = self._event_alignment_counter_to_status(overall_counter)
        since_anchor = self._event_alignment_counter_to_status(self._event_alignment_since_anchor)
        status = {
            "enabled": bool(self._event_exit_strict_mode_enabled()),
            "window": int(self._event_alignment_window),
            "samples": overall.get("samples"),
            "known_samples": overall.get("known_samples"),
            "aligned": overall.get("aligned"),
            "alignment_rate": overall.get("alignment_rate"),
            "blocked_samples": overall.get("blocked_samples"),
            "blocked_to_event_exit": overall.get("blocked_to_event_exit"),
            "blocked_to_event_exit_rate": overall.get("blocked_to_event_exit_rate"),
            "safe_samples": overall.get("safe_samples"),
            "safe_to_non_event_exit": overall.get("safe_to_non_event_exit"),
            "safe_to_non_event_exit_rate": overall.get("safe_to_non_event_exit_rate"),
            "strict_mode_samples": overall.get("strict_mode_samples"),
            "latest_reason": overall.get("latest_reason"),
            "latest_symbol": overall.get("latest_symbol"),
            "latest_ts_ms": overall.get("latest_ts_ms"),
            "latest_trade_id": overall.get("latest_trade_id"),
            "since_anchor": since_anchor,
            "anchor_ts_ms": int(self._event_alignment_anchor_ts_ms or 0),
            "anchor_trade_id": int(self._event_alignment_anchor_trade_id or 0),
            "report_path": str(self._event_alignment_report_path),
        }
        self._persist_event_alignment_report(force=False, status=status)
        return status

    def _load_persistent_state(self):
        balance_loaded = False
        # balance
        bal = self._load_json(self.state_files.get("balance"), None)
        if isinstance(bal, (int, float)):
            self.balance = float(bal)
            balance_loaded = True

        # equity history
        eq = self._load_json(self.state_files["equity"], [])
        for item in eq:
            try:
                t = int(item.get("time", 0))
                v = float(item.get("equity", 0.0))
                self._equity_history.append({"time": t, "equity": v})
            except Exception:
                continue

        # trade tape
        trades = self._load_json(self.state_files["trade"], [])
        for t in trades:
            self.trade_tape.append(t)

        # eval history
        evals = self._load_json(self.state_files["eval"], [])
        for e in evals:
            self.eval_history.append(e)

        # positions
        poss = self._load_json(self.state_files.get("positions"), [])
        if isinstance(poss, list):
            for p in poss:
                try:
                    sym = str(p.get("symbol"))
                    if not sym:
                        continue
                    p["entry_price"] = float(p.get("entry_price", 0.0))
                    p["quantity"] = float(p.get("quantity", 0.0))
                    p["notional"] = float(p.get("notional", 0.0))
                    p["leverage"] = float(p.get("leverage", self.leverage))
                    p["cap_frac"] = float(p.get("cap_frac", 0.0))
                    p["entry_time"] = self._normalize_entry_time_ms(p.get("entry_time", now_ms()), default=now_ms())
                    p["hold_limit"] = int(p.get("hold_limit", MAX_POSITION_HOLD_SEC * 1000))
                    self.positions[sym] = p
                except Exception:
                    continue

        # fallback balance from equity history if not loaded and no positions
        if (not balance_loaded) and self._equity_history and not self.positions:
            self.balance = float(self._equity_history[-1]["equity"])

    def _bootstrap_state_from_db_if_empty(self) -> None:
        if not hasattr(self, "db") or self.db is None:
            return
        mode = self._trading_mode if hasattr(self, "_trading_mode") else TradingMode.PAPER
        # Trade tape
        if not self.trade_tape:
            try:
                limit = int(getattr(self.trade_tape, "maxlen", 20000) or 20000)
                rows = self.db.get_recent_trades(limit=limit, mode=mode)
                for row in rows:
                    rec = None
                    raw = row.get("raw_data")
                    if raw:
                        try:
                            rec = json.loads(raw)
                        except Exception:
                            rec = None
                    if rec is None:
                        rec = row
                    self.trade_tape.append(rec)
                if rows:
                    self._log(f"[STATE_LOAD] trade_tape loaded from SQLite: {len(rows)} rows")
                    trade_path = self.state_files.get("trade")
                    need_repair = bool(trade_path and (not Path(trade_path).exists() or str(trade_path) in self._state_json_load_errors))
                    if need_repair and trade_path is not None:
                        try:
                            self._write_json_atomic(Path(trade_path), list(self.trade_tape))
                            self._state_json_load_errors.discard(str(trade_path))
                            self._log(f"[STATE_REPAIR] rewrote {Path(trade_path).name} from SQLite bootstrap")
                        except Exception as repair_err:
                            self._log_err(f"[ERR] repair {Path(trade_path).name}: {repair_err}")
            except Exception as e:
                self._log_err(f"[ERR] load trade_tape from SQLite: {e}")

        # Equity history
        if not self._equity_history:
            try:
                limit = int(getattr(self._equity_history, "maxlen", 20000) or 20000)
                rows = self.db.get_equity_history(limit=limit, mode=mode)
                for row in rows:
                    ts = row.get("timestamp_ms") or row.get("time") or row.get("ts")
                    eq = row.get("total_equity")
                    if eq is None:
                        eq = row.get("equity")
                    if eq is None:
                        eq = row.get("wallet_balance")
                    if ts is None or eq is None:
                        continue
                    try:
                        self._equity_history.append({"time": int(ts), "equity": float(eq)})
                    except Exception:
                        continue
                if rows:
                    self._log(f"[STATE_LOAD] equity_history loaded from SQLite: {len(rows)} rows")
            except Exception as e:
                self._log_err(f"[ERR] load equity_history from SQLite: {e}")

        # Positions
        if not self.positions:
            try:
                pos_map = self.db.get_all_positions(mode=mode)
                if pos_map:
                    for sym, pos in pos_map.items():
                        self.positions[sym] = pos
                    self._log(f"[STATE_LOAD] positions loaded from SQLite: {len(pos_map)} symbols")
            except Exception as e:
                self._log_err(f"[ERR] load positions from SQLite: {e}")

    def _init_initial_equity(self):
        if self.initial_equity is not None:
            return
        if self._equity_history:
            try:
                self.initial_equity = float(self._equity_history[-1]["equity"])
                return
            except Exception:
                pass
        self.initial_equity = float(self.balance)

    def _estimate_current_equity(self) -> float:
        unreal = 0.0
        for sym, pos in self.positions.items():
            try:
                px = self.market.get(sym, {}).get("price")
            except Exception:
                px = None
            if px is None:
                continue
            try:
                entry = float(pos.get("entry_price", 0.0))
                qty = float(pos.get("quantity", 0.0))
                side = str(pos.get("side", "")).upper()
            except Exception:
                continue
            pnl = (px - entry) * qty if side == "LONG" else (entry - px) * qty
            unreal += float(pnl)
        try:
            bal = float(self.balance)
        except Exception:
            bal = 0.0
        return float(bal) + float(unreal)

    def _exposure_cap_balance(self) -> float:
        """Balance base for exposure cap (total equity = wallet + unrealized)."""
        try:
            eq = float(self._estimate_current_equity())
            if math.isfinite(eq) and eq > 0:
                return eq
        except Exception:
            pass
        try:
            return float(self.balance)
        except Exception:
            return 0.0

    def _sizing_balance(self) -> float:
        """Balance used for order sizing (prefer live free balance when available)."""
        if self.enable_orders:
            # Guard against stale/missing live balance (avoid over-allocating)
            try:
                stale_sec = float(os.environ.get("LIVE_BALANCE_STALE_SEC", 15.0) or 15.0)
            except Exception:
                stale_sec = 15.0
            try:
                cache_max_age_sec = float(os.environ.get("LIVE_BALANCE_CACHE_MAX_AGE_SEC", 180.0) or 180.0)
            except Exception:
                cache_max_age_sec = 180.0
            last_ms = int(self._last_live_sync_ms or 0)
            age_ms = (now_ms() - last_ms) if last_ms else (10**12)
            is_stale = (not last_ms) or age_ms > int(stale_sec * 1000)
            if is_stale:
                if not self._live_balance_stale_warned:
                    self._log_err("[LIVE_BAL] balance stale/missing; blocking live sizing")
                    self._live_balance_stale_warned = True
                # If sync is briefly stale but we still have recent cached free/wallet,
                # continue sizing from cache to avoid long entry blackouts.
                if last_ms and age_ms <= int(max(cache_max_age_sec, stale_sec) * 1000):
                    try:
                        free = self._live_free_balance
                        if free is not None:
                            free_val = float(free)
                            if math.isfinite(free_val) and free_val > 0:
                                if not self._live_balance_cache_warned:
                                    self._log("[LIVE_BAL] using cached free balance while sync is stale")
                                    self._live_balance_cache_warned = True
                                return free_val
                    except Exception:
                        pass
                    try:
                        wallet = self._live_wallet_balance
                        if wallet is not None:
                            wallet_val = float(wallet)
                            if math.isfinite(wallet_val) and wallet_val > 0:
                                if not self._live_balance_cache_warned:
                                    self._log("[LIVE_BAL] using cached wallet balance while sync is stale")
                                    self._live_balance_cache_warned = True
                                return wallet_val
                    except Exception:
                        pass
                try:
                    allow_fallback = str(os.environ.get("LIVE_ALLOW_BALANCE_FALLBACK", "0")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    allow_fallback = False
                if not allow_fallback:
                    return 0.0
            else:
                self._live_balance_stale_warned = False
                self._live_balance_cache_warned = False
            try:
                free = self._live_free_balance
                if free is not None:
                    free_val = float(free)
                    if math.isfinite(free_val) and free_val > 0:
                        return free_val
            except Exception:
                pass
            try:
                allow_fallback = str(os.environ.get("LIVE_ALLOW_BALANCE_FALLBACK", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                allow_fallback = False
            if not allow_fallback:
                return 0.0
        try:
            return float(self.balance)
        except Exception:
            return 0.0

    def _cycle_available_balance(self) -> float:
        """Available balance within current decision cycle (free - reserved)."""
        base = self._cycle_free_balance
        if base is None:
            base = self._sizing_balance()
        try:
            base_val = float(base or 0.0)
        except Exception:
            base_val = 0.0
        if not self.enable_orders:
            return base_val
        try:
            reserved = float(self._cycle_reserved_margin or 0.0)
        except Exception:
            reserved = 0.0
        return max(0.0, base_val - reserved)

    def _reserve_cycle_margin(self, notional: float, leverage: float) -> None:
        """Reserve margin in the current decision cycle to avoid oversubscription."""
        if not self.enable_orders:
            return
        try:
            lev = float(leverage)
        except Exception:
            lev = 0.0
        if lev <= 0:
            return
        try:
            margin = float(notional) / max(lev, 1e-6)
        except Exception:
            return
        if not math.isfinite(margin) or margin <= 0:
            return
        try:
            self._cycle_reserved_margin = float(self._cycle_reserved_margin or 0.0) + margin
            self._cycle_reserved_notional = float(self._cycle_reserved_notional or 0.0) + float(notional)
        except Exception:
            pass

    def _maybe_reset_initial_equity_on_start(self) -> None:
        if not bool(getattr(config, "RESET_INITIAL_EQUITY_ON_START", False)):
            return
        equity = self._estimate_current_equity()
        if equity <= 0:
            try:
                equity = float(self.balance)
            except Exception:
                equity = 1.0
        self.initial_equity = float(equity)
        try:
            self.risk_manager.initial_equity = float(equity)
            self.risk_manager._emergency_stop_triggered = False
            self.risk_manager._dd_stop_triggered = False
        except Exception:
            pass
        self.safety_mode = False
        self._emergency_stop_handled = False
        self._emergency_stop_ts = None
        try:
            self._log(f"[RISK] initial_equity reset on start: {equity:.2f}")
        except Exception:
            pass

    def clear_safety_mode(self, *, reset_equity: bool = False) -> dict:
        # Manual override from dashboard.
        try:
            self.safety_mode = False
            self._emergency_stop_handled = False
            self._emergency_stop_ts = None
        except Exception:
            pass
        try:
            if hasattr(self, "risk_manager") and self.risk_manager is not None:
                self.risk_manager._emergency_stop_triggered = False
                self.risk_manager._dd_stop_triggered = False
        except Exception:
            pass

        reset_val = None
        if reset_equity:
            try:
                equity = float(self._estimate_current_equity())
            except Exception:
                equity = 0.0
            if equity <= 0:
                try:
                    equity = float(self.balance)
                except Exception:
                    equity = 1.0
            if equity <= 0:
                equity = 1.0
            try:
                self.initial_equity = float(equity)
            except Exception:
                pass
            try:
                if hasattr(self, "risk_manager") and self.risk_manager is not None:
                    self.risk_manager.initial_equity = float(equity)
            except Exception:
                pass
            reset_val = float(equity)

        try:
            self._log(f"[RISK] safety_mode cleared by dashboard reset_equity={reset_equity} equity={reset_val if reset_equity else 'n/a'}")
        except Exception:
            pass
        return {
            "safety_mode": bool(getattr(self, "safety_mode", False)),
            "reset_equity": bool(reset_equity),
            "equity": reset_val,
        }

    def _persist_state(self, force: bool = False):
        now = now_ms()
        if not force and (now - self._last_state_persist_ms < 10_000):
            return
        self._last_state_persist_ms = now
        
        # ---- JSON 저장 (기존 호환성 유지)
        try:
            self._write_json_atomic(self.state_files["equity"], list(self._equity_history))
            self._write_json_atomic(self.state_files["trade"], list(self.trade_tape))
            self._write_json_atomic(self.state_files["eval"], list(self.eval_history))
            self._write_json_atomic(self.state_files["positions"], list(self.positions.values()))
            self._write_json_atomic(self.state_files["balance"], self.balance)
        except Exception as e:
            self._log_err(f"[ERR] persist state (JSON): {e}")
        
        # ---- SQLite 저장 (비동기)
        try:
            # Equity 저장
            total_unrealized = sum(
                float(p.get("pnl", 0) or 0) for p in self.positions.values()
            )
            equity_data = {
                "timestamp_ms": now,
                "total_equity": float(self.balance) + total_unrealized,
                "wallet_balance": float(self.balance),
                "unrealized_pnl": total_unrealized,
                "position_count": len(self.positions),
                "total_notional": sum(
                    float(p.get("notional", 0) or 0) for p in self.positions.values()
                ),
            }
            self.db.log_equity_background(equity_data, mode=self._trading_mode)
            
            # Positions 저장
            for sym, pos in self.positions.items():
                self.db.save_position_background(sym, pos, mode=self._trading_mode)
        except Exception as e:
            self._log_err(f"[ERR] persist state (SQLite): {e}")

    def _compute_returns_and_vol(self, prices):
        if prices is None or len(prices) < 10:
            return None, None

        log_returns = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 and p1 and p0 > 0 and p1 > 0:
                log_returns.append(math.log(p1 / p0))

        if len(log_returns) < 5:
            return None, None

        mu = sum(log_returns) / len(log_returns)
        var = sum((r - mu) ** 2 for r in log_returns) / len(log_returns)
        sigma = math.sqrt(var)
        return mu, sigma

    @staticmethod
    def _annualize_mu_sigma(mu_bar: float, sigma_bar: float, bar_seconds: float) -> tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = mu_bar * bars_per_year
        sigma_annual = sigma_bar * math.sqrt(bars_per_year)
        return float(mu_base), float(sigma_annual)

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _extract_filter_states(decision: dict) -> dict:
        """decision 객체에서 filter_states 추출 (여러 위치에서 찾기)"""
        # 기본값: 모든 필터 통과 (회색 표시)
        default_filter_states = {
            "unified": None,
            "spread": None,
            "event_cvar": None,
            "event_exit": None,
            "both_ev_neg": None,
            "gross_ev": None,
            "net_expectancy": None,
            "dir_gate": None,
            "symbol_quality": None,
            "liq": None,
            "min_notional": None,
            "min_exposure": None,
            "tick_vol": None,
            "fee": None,
            "top_n": None,
            "pre_mc": None,
            "cap": None,
            "cap_safety": None,
            "cap_positions": None,
            "cap_exposure": None,
        }
        
        if not decision:
            return default_filter_states
        
        # 1. decision 최상위
        if decision.get("filter_states"):
            return decision["filter_states"]
        # 2. decision.meta
        meta = decision.get("meta") or {}
        if meta.get("filter_states"):
            return meta["filter_states"]
        # 3. decision.details 내부
        for d in decision.get("details", []):
            if isinstance(d, dict):
                if d.get("filter_states"):
                    return d["filter_states"]
                # details 내부의 meta
                dm = d.get("meta") or {}
                if dm.get("filter_states"):
                    return dm["filter_states"]
        
        # 아무것도 찾지 못하면 기본값 반환
        return default_filter_states
    def _row(self, sym, price, ts, decision, candles, ctx=None, *, log_filters: bool = True):
        # decision loop can call _row with ctx=None during early bootstrap/error fallback
        # paths; normalize to avoid dashboard feed crashes.
        ctx = ctx or {}
        status = "WAIT"
        ai = "-"
        mc = "-"
        conf = 0.0

        mc_meta = {}
        if decision:
            status = decision.get("action", "WAIT")  # LONG/SHORT/WAIT
            conf = self._safe_float(decision.get("confidence", 0.0), 0.0)
            mc = decision.get("reason", "") or "-"
            # details에서 mc 메타 뽑기
            for d in decision.get("details", []):
                if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                    mc_meta = d.get("meta", {}) or {}
                    break
            # details에 없으면 decision.meta 사용 (엔진이 meta를 직접 주는 경우)
            if not mc_meta:
                mc_meta = decision.get("meta", {}) or {}

        trade_age = ts - self.market[sym]["ts"] if self.market[sym]["ts"] else None
        kline_age = ts - self._last_kline_ok_ms[sym] if self._last_kline_ok_ms[sym] else None

        ob_age = ts - self.orderbook[sym]["ts"] if self.orderbook[sym]["ts"] else None
        ob_ready = bool(self.orderbook[sym]["ready"])

        meta = decision.get("meta", {}) if decision else {}
        ev = self._safe_float(decision.get("ev", meta.get("ev", 0.0)) if decision else 0.0, 0.0)
        kelly = self._safe_float(meta.get("kelly", 0.0), 0.0)
        regime = (ctx or {}).get("regime") or meta.get("regime") or "-"
        mode = "alpha"
        if self.positions.get(sym, {}).get("tag") == "spread":
            mode = "spread"
        elif decision and any(d.get("_engine") == "consensus" for d in decision.get("details", [])):
            mode = "consensus"
        action_type = self.positions.get(sym, {}).get("tag") == "spread" and "SPREAD" or (self.positions.get(sym) and "HOLD") or self._last_actions.get(sym, "-")

        def _opt_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        pos = self.positions.get(sym)
        ex_pos = None
        try:
            if self.enable_orders and self._exchange_positions_ts is not None:
                ex_pos = self._exchange_positions_by_symbol.get(sym)
        except Exception:
            ex_pos = None
        pos_side = pos.get("side") if pos else "-"
        pos_pnl = None
        pos_roe = None
        pos_lev = pos.get("leverage") if pos else self._dyn_leverage.get(sym)
        pos_sf = pos.get("size_frac") if pos else None
        pos_cap_frac = pos.get("cap_frac") if pos else None
        if pos and price:
            entry = float(pos.get("entry_price", price))
            qty = float(pos.get("quantity", 0.0))
            pnl = ((price - entry) * qty) if pos_side == "LONG" else ((entry - price) * qty)
            notional = float(pos.get("notional", 0.0))
            lev_safe = float(pos.get("entry_leverage", pos.get("leverage", self.leverage)) or 1.0)
            base_notional = notional / max(lev_safe, 1e-6)
            pos_pnl = pnl
            pos_roe = pnl / base_notional if base_notional else 0.0

        if ex_pos is not None:
            ex_side = str(ex_pos.get("side") or "")
            if ex_side:
                pos_side = ex_side
            ex_lev = _opt_float(ex_pos.get("leverage"))
            if ex_lev is not None:
                pos_lev = ex_lev
            ex_sf = _opt_float(ex_pos.get("size_frac"))
            if ex_sf is not None:
                pos_sf = ex_sf
            ex_cap = _opt_float(ex_pos.get("cap_frac"))
            if ex_cap is not None:
                pos_cap_frac = ex_cap
            ex_pnl = _opt_float(ex_pos.get("unrealized_pnl"))
            if ex_pnl is None and price is not None:
                try:
                    ex_entry = float(ex_pos.get("entry_price", price))
                    ex_qty = float(ex_pos.get("quantity", 0.0))
                    ex_pnl = (float(price) - ex_entry) * ex_qty if pos_side == "LONG" else (ex_entry - float(price)) * ex_qty
                except Exception:
                    ex_pnl = None
            ex_roe = _opt_float(ex_pos.get("roe"))
            if ex_roe is None and ex_pnl is not None:
                ex_margin = _opt_float(ex_pos.get("margin"))
                if ex_margin is not None and ex_margin > 0:
                    ex_roe = float(ex_pnl) / float(ex_margin)
                else:
                    ex_notional = _opt_float(ex_pos.get("notional"))
                    ex_lev_for_roe = _opt_float(ex_pos.get("entry_leverage"))
                    if ex_lev_for_roe is None or ex_lev_for_roe <= 0:
                        ex_lev_for_roe = _opt_float(ex_pos.get("leverage"))
                    if (ex_notional is not None) and (ex_lev_for_roe is not None) and ex_lev_for_roe > 0:
                        ex_base = ex_notional / max(float(ex_lev_for_roe), 1e-6)
                        if ex_base > 0:
                            ex_roe = float(ex_pnl) / float(ex_base)
            if ex_pnl is not None:
                pos_pnl = float(ex_pnl)
            if ex_roe is not None:
                pos_roe = float(ex_roe)

        hybrid_score = _opt_float(meta.get("hybrid_score"))
        if hybrid_score is None:
            hybrid_score = _opt_float(mc_meta.get("hybrid_score") if mc_meta else None)
        hybrid_score_logw = _opt_float(meta.get("hybrid_score_logw"))
        if hybrid_score_logw is None:
            hybrid_score_logw = _opt_float(mc_meta.get("hybrid_score_logw") if mc_meta else None)
        hybrid_score_hold = _opt_float(meta.get("hybrid_score_hold"))
        if hybrid_score_hold is None:
            hybrid_score_hold = _opt_float(mc_meta.get("hybrid_score_hold") if mc_meta else None)
        hybrid_score_long = _opt_float(meta.get("hybrid_score_long"))
        if hybrid_score_long is None:
            hybrid_score_long = _opt_float(mc_meta.get("hybrid_score_long") if mc_meta else None)
        hybrid_score_short = _opt_float(meta.get("hybrid_score_short"))
        if hybrid_score_short is None:
            hybrid_score_short = _opt_float(mc_meta.get("hybrid_score_short") if mc_meta else None)
        hybrid_entry_floor = _opt_float(meta.get("hybrid_entry_floor"))
        if hybrid_entry_floor is None:
            hybrid_entry_floor = _opt_float(mc_meta.get("hybrid_entry_floor") if mc_meta else None)

        opt_leverage = _opt_float(meta.get("optimal_leverage"))
        if opt_leverage is None and decision:
            opt_leverage = _opt_float(decision.get("optimal_leverage") or decision.get("leverage"))
        decision_leverage = _opt_float(decision.get("leverage") if decision else None)
        if decision_leverage is None:
            decision_leverage = _opt_float(meta.get("lev"))
        dyn_leverage = _opt_float(self._dyn_leverage.get(sym))
        lev_source = meta.get("lev_source")
        liq_prob_long = _opt_float(meta.get("liq_prob_long"))
        if liq_prob_long is None and decision:
            liq_prob_long = _opt_float(decision.get("liq_prob_long"))
        liq_prob_short = _opt_float(meta.get("liq_prob_short"))
        if liq_prob_short is None and decision:
            liq_prob_short = _opt_float(decision.get("liq_prob_short"))
        liq_price_long = _opt_float(meta.get("liq_price_long"))
        if liq_price_long is None and decision:
            liq_price_long = _opt_float(decision.get("liq_price_long"))
        liq_price_short = _opt_float(meta.get("liq_price_short"))
        if liq_price_short is None and decision:
            liq_price_short = _opt_float(decision.get("liq_price_short"))

        unified_score = _opt_float(decision.get("unified_score") if decision else None)
        if unified_score is None:
            unified_score = _opt_float(meta.get("unified_score"))
        if unified_score is None and hybrid_score is not None:
            unified_score = hybrid_score
        unified_score_hold = _opt_float(decision.get("unified_score_hold") if decision else None)
        if unified_score_hold is None:
            unified_score_hold = _opt_float(meta.get("unified_score_hold"))
        if unified_score_hold is None:
            unified_score_hold = _opt_float(mc_meta.get("unified_score_hold") if mc_meta else None)
        unified_t_star = _opt_float(meta.get("unified_t_star"))
        if unified_t_star is None:
            unified_t_star = _opt_float(mc_meta.get("unified_t_star") if mc_meta else None)
        event_p_tp = _opt_float(meta.get("event_p_tp"))
        event_p_sl = _opt_float(meta.get("event_p_sl"))
        event_p_timeout = _opt_float(meta.get("event_p_timeout"))
        event_t_median = _opt_float(meta.get("event_t_median"))
        event_ev_r = _opt_float(meta.get("event_ev_r"))
        event_cvar_r = _opt_float(meta.get("event_cvar_r"))
        event_ev_pct = _opt_float(meta.get("event_ev_pct"))
        event_cvar_pct = _opt_float(meta.get("event_cvar_pct"))
        event_unified_score = _opt_float(meta.get("event_unified_score"))
        event_exit_score = _opt_float(meta.get("event_exit_score"))
        event_exit_min_score = _opt_float(meta.get("event_exit_min_score"))
        event_exit_max_cvar = _opt_float(meta.get("event_exit_max_cvar"))
        event_exit_max_p_sl = _opt_float(meta.get("event_exit_max_p_sl"))
        horizon_weights = meta.get("horizon_weights")
        ev_by_h = meta.get("ev_by_horizon")
        win_by_h = meta.get("win_by_horizon")
        cvar_by_h = meta.get("cvar_by_horizon")
        horizon_seq = meta.get("horizon_seq")
        mu_alpha_row = _opt_float(meta.get("mu_alpha"))
        if mu_alpha_row is None:
            mu_alpha_row = _opt_float(mc_meta.get("mu_alpha"))
        if mu_alpha_row is None:
            mu_alpha_row = _opt_float(ctx.get("mu_alpha"))
        if mu_alpha_row is None:
            # Dashboard fallback only: keep alpha column alive even when engine meta omits mu_alpha.
            mu_alpha_row = _opt_float(ctx.get("mu_causal"))
        if mu_alpha_row is None:
            mu_alpha_row = _opt_float(ctx.get("mu_kf"))
        mu_alpha_raw_row = _opt_float(meta.get("mu_alpha_raw"))
        if mu_alpha_raw_row is None:
            mu_alpha_raw_row = _opt_float(mc_meta.get("mu_alpha_raw"))
        if mu_alpha_raw_row is None:
            mu_alpha_raw_row = _opt_float(ctx.get("mu_alpha_raw"))
        if mu_alpha_raw_row is None:
            mu_alpha_raw_row = mu_alpha_row
        entry_quality_row = _opt_float(meta.get("entry_quality_score"))
        if entry_quality_row is None and pos:
            entry_quality_row = _opt_float(pos.get("entry_quality_score"))
        one_way_row = _opt_float(meta.get("one_way_move_score"))
        if one_way_row is None and pos:
            one_way_row = _opt_float(pos.get("one_way_move_score"))
        leverage_signal_row = _opt_float(meta.get("leverage_signal_score"))
        if leverage_signal_row is None and pos:
            leverage_signal_row = _opt_float(pos.get("leverage_signal_score"))
        
        # WAIT 결정 디버그용
        mc_meta_final = meta if meta.get("policy_direction_reason") else (mc_meta or {})
        direction_reason = mc_meta_final.get("policy_direction_reason")
        
        # TP 게이트 정보 추가 (진입 차단 주원인)
        tp_v = mc_meta_final.get("policy_tp_5m_long") if status != "SHORT" else mc_meta_final.get("policy_tp_5m_short")
        tp_blocked = mc_meta_final.get("policy_tp_gate_block_long") if status != "SHORT" else mc_meta_final.get("policy_tp_gate_block_short")
        
        if direction_reason:
            if tp_blocked:
                direction_reason += f" | TP_GATED({(tp_v*100):.1f}%)"
            elif tp_v is not None:
                direction_reason += f" | TP:{(tp_v*100):.1f}%"
        
        score_long = _opt_float(mc_meta_final.get("hybrid_score_long") or mc_meta_final.get("policy_ev_score_long"))
        score_short = _opt_float(mc_meta_final.get("hybrid_score_short") or mc_meta_final.get("policy_ev_score_short"))
        score_threshold = _opt_float(mc_meta_final.get("policy_score_threshold_eff"))

        row = {
            "symbol": sym,
            "price": price,
            "status": status,
            "ai": ai,
            "mc": mc,
            "conf": conf,
            "ev": ev,
            "kelly": kelly,
            "regime": regime,
            "mode": mode,
            "action_type": action_type,
            "hybrid_score": hybrid_score,
            "hybrid_score_logw": hybrid_score_logw,
            "hybrid_score_hold": hybrid_score_hold,
            "hybrid_score_long": hybrid_score_long,
            "hybrid_score_short": hybrid_score_short,
            "hybrid_entry_floor": hybrid_entry_floor,
            "unified_score": unified_score if unified_score is not None else ev,
            "unified_score_hold": unified_score_hold,
            "unified_t_star": unified_t_star,
            "candles": candles,
            "event_p_tp": event_p_tp,
            "event_p_sl": event_p_sl,
            "event_p_timeout": event_p_timeout,
            "event_t_median": event_t_median,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_ev_pct": event_ev_pct,
            "event_cvar_pct": event_cvar_pct,
            "event_unified_score": event_unified_score,
            "event_exit_score": event_exit_score,
            "event_exit_min_score": event_exit_min_score,
            "event_exit_max_cvar": event_exit_max_cvar,
            "event_exit_max_p_sl": event_exit_max_p_sl,
            "horizon_weights": horizon_weights,
            "ev_by_horizon": ev_by_h,
            "win_by_horizon": win_by_h,
            "cvar_by_horizon": cvar_by_h,
            "horizon_seq": horizon_seq,

            # freshness
            "trade_age": trade_age,
            "kline_age": kline_age,
            "orderbook_age": ob_age,
            "orderbook_ready": ob_ready,

            "pos_side": pos_side,
            "pos_pnl": pos_pnl,
            "pos_roe": pos_roe,
            "pos_tag": pos.get("tag") if pos else None,
            "pos_leverage": pos_lev,
            "pos_size_frac": pos_sf,
            "pos_cap_frac": pos_cap_frac,
            "opt_leverage": opt_leverage,
            "decision_leverage": decision_leverage,
            "dyn_leverage": dyn_leverage,
            "lev_source": lev_source,
            "entry_quality_score": entry_quality_row,
            "one_way_move_score": one_way_row,
            "leverage_signal_score": leverage_signal_row,
            "liq_prob_long": liq_prob_long,
            "liq_prob_short": liq_prob_short,
            "liq_price_long": liq_price_long,
            "liq_price_short": liq_price_short,

            # MC diagnostics
            "mc_h_desc": mc_meta.get("best_horizon_desc") or mc_meta.get("best_horizon") or "-",
            "mc_ev": self._safe_float(mc_meta.get("ev", 0.0), 0.0),
            "mc_win_rate": self._safe_float(mc_meta.get("win_rate", 0.0), 0.0),
            "mc_be_win_rate": self._safe_float(mc_meta.get("be_win_rate", 0.0), 0.0),
            "mc_tp": self._safe_float(mc_meta.get("tp", 0.0), 0.0),
            "mc_sl": self._safe_float(mc_meta.get("sl", 0.0), 0.0),
            "mc_hit_rate": self._safe_float(mc_meta.get("hit_rate", 0.0), 0.0),

            # alpha components (for dashboard)
            "mu_alpha": mu_alpha_row,
            "mu_alpha_raw": mu_alpha_raw_row,
            "alpha_mlofi": _opt_float(ctx.get("mlofi")),
            "alpha_vpin": _opt_float(ctx.get("vpin")),
            "alpha_mu_kf": _opt_float(ctx.get("mu_kf")),
            "alpha_hurst": _opt_float(ctx.get("hurst")),
            "alpha_mu_ml": _opt_float(ctx.get("mu_ml")),
            "alpha_mu_bayes": _opt_float(ctx.get("mu_bayes")),
            "alpha_mu_ar": _opt_float(ctx.get("mu_ar")),
            "alpha_mu_pf": _opt_float(ctx.get("mu_pf")),
            "alpha_mu_ou": _opt_float(ctx.get("mu_ou")),
            "alpha_hawkes_boost": _opt_float(ctx.get("hawkes_boost")),
            "alpha_sigma_garch": _opt_float(ctx.get("sigma_garch")),
            "alpha_mu_dir_prob_long": _opt_float(ctx.get("mu_dir_prob_long")),
            "alpha_mu_dir_edge": _opt_float(ctx.get("mu_dir_edge")),
            "alpha_mu_dir_conf": _opt_float(ctx.get("mu_dir_conf")),
            # Backward-compatible aliases for older dashboard/report consumers.
            "vpin": _opt_float(ctx.get("vpin")),
            "hurst": _opt_float(ctx.get("hurst")),
            "mu_dir_prob_long": _opt_float(ctx.get("mu_dir_prob_long")),
            "mu_dir_edge": _opt_float(ctx.get("mu_dir_edge")),
            "mu_dir_conf": _opt_float(ctx.get("mu_dir_conf")),
            "alpha_hmm_state": ctx.get("hmm_state"),
            "alpha_hmm_conf": _opt_float(ctx.get("hmm_conf")),
            "alpha_hmm_sign": _opt_float(ctx.get("hmm_regime_sign")),

            "details": (decision.get("details", []) if decision else []),
            
            # ✅ 진입 필터 상태 (신호등 표시용)
            "filter_states": self._extract_filter_states(decision),

            # ✅ 필터 툴팁용 메타
            "entry_floor_eff": _opt_float(meta.get("entry_floor_eff")),
            "min_entry_score": _opt_float(meta.get("min_entry_score")),
            "spread_pct": _opt_float(meta.get("spread_pct")),
            "spread_cap": _opt_float(meta.get("spread_cap")),
            "cvar_floor": _opt_float(meta.get("cvar_floor")),
            "net_expectancy_effective": _opt_float(meta.get("net_expectancy_effective")),
            "net_expectancy_min": _opt_float(meta.get("net_expectancy_min")),
            "net_expectancy_raw": _opt_float(meta.get("net_expectancy_raw")),
            "net_expectancy_fee_cost": _opt_float(meta.get("net_expectancy_fee_cost")),
            "net_expectancy_vpin": _opt_float(meta.get("net_expectancy_vpin")),
            "net_expectancy_dir_conf": _opt_float(meta.get("net_expectancy_dir_conf")),
            "symbol_quality_score": _opt_float(meta.get("symbol_quality_score")),
            "symbol_quality_sample_n": meta.get("symbol_quality_sample_n"),
            "symbol_quality_expectancy": _opt_float(meta.get("symbol_quality_expectancy")),
            "symbol_quality_gross_expectancy": _opt_float(meta.get("symbol_quality_gross_expectancy")),
            "symbol_quality_hit_rate": _opt_float(meta.get("symbol_quality_hit_rate")),
            "symbol_quality_reject_ratio": _opt_float(meta.get("symbol_quality_reject_ratio")),
            "symbol_quality_min_exits": meta.get("symbol_quality_min_exits"),
            "symbol_quality_min_score": _opt_float(meta.get("symbol_quality_min_score")),
            "liq_score": _opt_float(meta.get("liq_score")),
            "min_liq_score": _opt_float(meta.get("min_liq_score")),
            "est_notional": _opt_float(meta.get("est_notional")),
            "min_entry_notional": _opt_float(meta.get("min_entry_notional")),
            "top_n_active": meta.get("top_n_active"),
            "top_n_rank": meta.get("top_n_rank"),
            "top_n_limit": meta.get("top_n_limit"),
            "top_n_ok": meta.get("top_n_ok"),
            "pre_mc_active": meta.get("pre_mc_active"),
            "pre_mc_ok": meta.get("pre_mc_ok"),
            "pre_mc_reason": meta.get("pre_mc_reason"),
            "pre_mc_expected_pnl": _opt_float(meta.get("pre_mc_expected_pnl")),
            "pre_mc_expected_pnl_source": meta.get("pre_mc_expected_pnl_source"),
            "pre_mc_cvar": _opt_float(meta.get("pre_mc_cvar")),
            "pre_mc_prob_liq": _opt_float(meta.get("pre_mc_prob_liq")),
            "pre_mc_prob_liq_source": meta.get("pre_mc_prob_liq_source"),
            "pre_mc_min_expected_pnl": _opt_float(meta.get("pre_mc_min_expected_pnl")),
            "pre_mc_min_cvar": _opt_float(meta.get("pre_mc_min_cvar")),
            "pre_mc_max_liq_prob": _opt_float(meta.get("pre_mc_max_liq_prob")),
            "pre_mc_blocked": meta.get("pre_mc_blocked"),
            "pre_mc_scaled": meta.get("pre_mc_scaled"),
            "pre_mc_scale": _opt_float(meta.get("pre_mc_scale")),
            "pre_mc_block_on_fail": meta.get("pre_mc_block_on_fail"),
            "total_open_notional": _opt_float(meta.get("total_open_notional")),
            "exposure_cap_limit": _opt_float(meta.get("exposure_cap_limit")),
            "positions_count": meta.get("positions_count"),
            "max_positions": meta.get("max_positions"),
            "safety_mode": meta.get("safety_mode"),
            "hybrid_only": meta.get("hybrid_only"),
            
            # ✅ WAIT 결정 디버그용
            "direction_reason": direction_reason,
            "score_long": score_long,
            "score_short": score_short,
            "score_threshold": score_threshold,
        }
        
        # 디버깅: filter_states 확인 (옵션 로그)
        if log_filters:
            fs = self._extract_filter_states(decision)
            if fs:
                blocked = [k for k, v in fs.items() if v == False]
                if blocked:
                    try:
                        use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
                        hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
                        hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        hybrid_only = False
                    if hybrid_only:
                        blocked = ["hybrid" if k == "unified" else k for k in blocked]
                if blocked:
                    self._log(f"[FILTER] {sym} blocked: {blocked}")
                else:
                    # all_pass일 때 Action과 점수 정보 추가 (None-safe)
                    try:
                        best_ev_long = _opt_float(mc_meta_final.get("policy_best_ev_long")) if mc_meta_final else None
                        best_ev_short = _opt_float(mc_meta_final.get("policy_best_ev_short")) if mc_meta_final else None
                        ev_best = None
                        if status == "LONG":
                            ev_best = best_ev_long
                        elif status == "SHORT":
                            ev_best = best_ev_short
                        else:
                            ev_best = max(best_ev_long or 0.0, best_ev_short or 0.0)
                    except Exception:
                        ev_best = None
                    ev_best_txt = f"{(ev_best or 0.0):.5f}" if ev_best is not None else "-"
                    try:
                        use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
                        hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
                        hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        hybrid_only = False
                    score_label = "HYB" if hybrid_only else "UNI"
                    score_val = hybrid_score if (hybrid_only and hybrid_score is not None) else (unified_score or 0.0)
                    floor_val = hybrid_entry_floor if hybrid_only else float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
                    floor_txt = f"{(floor_val or 0.0):.5f}" if floor_val is not None else "-"
                    self._log(
                        f"[FILTER] {sym} all_pass | Action:{status} | {score_label}:{(score_val or 0.0):.5f} | FLOOR:{floor_txt} | EV_best:{ev_best_txt} | EV_sum:{(ev or 0.0):.5f}"
                    )
            else:
                self._log(f"[FILTER] {sym} NO filter_states extracted!")
        
        return row

    def _total_open_notional(self) -> float:
        return sum(float(pos.get("notional", 0.0)) for pos in self.positions.values())

    def _is_managed_position(self, pos: dict | None) -> bool:
        if not pos:
            return True
        if pos.get("managed") is False:
            return False
        try:
            manage_synced = bool(getattr(config, "MANAGE_SYNCED_POSITIONS", True))
        except Exception:
            manage_synced = True
        if not manage_synced:
            src = str(pos.get("pos_source") or "").lower()
            if src in ("exchange_sync", "exchange", "sync"):
                return False
        return True

    def _top_n_status(self, sym: str) -> dict:
        active = bool(TOP_N_SYMBOLS < len(SYMBOLS))
        rank = self._symbol_ranks.get(sym)
        ok = None
        if active and rank is not None:
            try:
                ok = int(rank) <= int(TOP_N_SYMBOLS)
            except Exception:
                ok = None
        return {"active": active, "rank": rank, "limit": int(TOP_N_SYMBOLS), "ok": ok}

    def _decision_metric_float(self, decision: dict | None, meta: dict | None, keys: tuple[str, ...], default=None):
        def _coerce(v):
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        if isinstance(meta, dict):
            for k in keys:
                out = _coerce(meta.get(k))
                if out is not None:
                    return out
        if isinstance(decision, dict):
            for k in keys:
                out = _coerce(decision.get(k))
                if out is not None:
                    return out
            details = decision.get("details")
            if isinstance(details, list):
                for d in details:
                    if not isinstance(d, dict):
                        continue
                    dm = d.get("meta")
                    if not isinstance(dm, dict):
                        continue
                    for k in keys:
                        out = _coerce(dm.get(k))
                        if out is not None:
                            return out
        return default

    def _symbol_quality_state(self, sym: str, ts_ms: int | None = None) -> dict[str, object]:
        now_ts = int(ts_ms if ts_ms is not None else now_ms())
        try:
            refresh_sec = float(os.environ.get("SYMBOL_QUALITY_REFRESH_SEC", 15.0) or 15.0)
        except Exception:
            refresh_sec = 15.0
        refresh_ms = int(max(1.0, refresh_sec) * 1000.0)
        cache = self._symbol_quality_cache_by_sym if isinstance(getattr(self, "_symbol_quality_cache_by_sym", None), dict) else {}
        cached = cache.get(sym)
        if isinstance(cached, dict):
            try:
                if (now_ts - int(cached.get("ts", 0) or 0)) < refresh_ms:
                    return cached
            except Exception:
                pass

        try:
            exit_lookback = int(os.environ.get("SYMBOL_QUALITY_LOOKBACK_EXITS", 320) or 320)
        except Exception:
            exit_lookback = 320
        try:
            reject_lookback = int(os.environ.get("SYMBOL_QUALITY_REJECT_LOOKBACK", 320) or 320)
        except Exception:
            reject_lookback = 320
        try:
            min_exits = int(os.environ.get("SYMBOL_QUALITY_MIN_EXITS", 40) or 40)
        except Exception:
            min_exits = 40
        try:
            min_score = float(os.environ.get("SYMBOL_QUALITY_MIN_SCORE", 0.42) or 0.42)
        except Exception:
            min_score = 0.42
        try:
            exp_ref = float(os.environ.get("SYMBOL_QUALITY_EXPECTANCY_REF", 0.0010) or 0.0010)
        except Exception:
            exp_ref = 0.0010
        try:
            reject_ref = float(os.environ.get("SYMBOL_QUALITY_REJECT_REF", 0.35) or 0.35)
        except Exception:
            reject_ref = 0.35
        try:
            w_exp = float(os.environ.get("SYMBOL_QUALITY_WEIGHT_EXPECTANCY", 0.55) or 0.55)
        except Exception:
            w_exp = 0.55
        try:
            w_hit = float(os.environ.get("SYMBOL_QUALITY_WEIGHT_HITRATE", 0.30) or 0.30)
        except Exception:
            w_hit = 0.30
        try:
            w_rej = float(os.environ.get("SYMBOL_QUALITY_WEIGHT_REJECT", 0.15) or 0.15)
        except Exception:
            w_rej = 0.15
        w_sum = float(max(1e-6, w_exp + w_hit + w_rej))
        w_exp, w_hit, w_rej = float(w_exp / w_sum), float(w_hit / w_sum), float(w_rej / w_sum)
        try:
            time_filter_enabled = str(os.environ.get("SYMBOL_QUALITY_TIME_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            time_filter_enabled = True
        try:
            time_window_hours = int(os.environ.get("SYMBOL_QUALITY_TIME_WINDOW_HOURS", 1) or 1)
        except Exception:
            time_window_hours = 1
        time_window_hours = int(max(0, min(12, time_window_hours)))
        try:
            time_min_exits = int(os.environ.get("SYMBOL_QUALITY_TIME_MIN_EXITS", 24) or 24)
        except Exception:
            time_min_exits = 24
        time_min_exits = int(max(1, min(time_min_exits, max(1, exit_lookback))))
        try:
            time_weight = float(os.environ.get("SYMBOL_QUALITY_TIME_WEIGHT", 0.30) or 0.30)
        except Exception:
            time_weight = 0.30
        time_weight = float(max(0.0, min(0.80, time_weight)))
        try:
            hard_neg_exp_floor = float(os.environ.get("SYMBOL_QUALITY_EXPECTANCY_HARD_FLOOR", -0.0015) or -0.0015)
        except Exception:
            hard_neg_exp_floor = -0.0015
        try:
            hard_neg_exp_penalty = float(os.environ.get("SYMBOL_QUALITY_EXPECTANCY_HARD_PENALTY", 0.22) or 0.22)
        except Exception:
            hard_neg_exp_penalty = 0.22
        try:
            hard_reject_floor = float(os.environ.get("SYMBOL_QUALITY_REJECT_HARD", 0.55) or 0.55)
        except Exception:
            hard_reject_floor = 0.55
        try:
            hard_reject_penalty = float(os.environ.get("SYMBOL_QUALITY_REJECT_PENALTY", 0.18) or 0.18)
        except Exception:
            hard_reject_penalty = 0.18
        try:
            gross_filter_enabled = str(os.environ.get("SYMBOL_QUALITY_GROSS_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            gross_filter_enabled = True
        try:
            gross_ref = float(os.environ.get("SYMBOL_QUALITY_GROSS_REF", exp_ref) or exp_ref)
        except Exception:
            gross_ref = float(exp_ref)
        try:
            gross_weight = float(os.environ.get("SYMBOL_QUALITY_GROSS_WEIGHT", 0.25) or 0.25)
        except Exception:
            gross_weight = 0.25
        gross_weight = float(max(0.0, min(0.80, gross_weight)))
        try:
            gross_time_weight = float(os.environ.get("SYMBOL_QUALITY_GROSS_TIME_WEIGHT", gross_weight) or gross_weight)
        except Exception:
            gross_time_weight = gross_weight
        gross_time_weight = float(max(0.0, min(0.80, gross_time_weight)))
        try:
            gross_hard_floor = float(os.environ.get("SYMBOL_QUALITY_GROSS_HARD_FLOOR", -0.0008) or -0.0008)
        except Exception:
            gross_hard_floor = -0.0008
        try:
            gross_hard_penalty = float(os.environ.get("SYMBOL_QUALITY_GROSS_HARD_PENALTY", 0.18) or 0.18)
        except Exception:
            gross_hard_penalty = 0.18
        try:
            gross_gate_floor = float(os.environ.get("SYMBOL_QUALITY_GROSS_GATE_FLOOR", -0.0002) or -0.0002)
        except Exception:
            gross_gate_floor = -0.0002
        try:
            gross_gate_floor_time = float(os.environ.get("SYMBOL_QUALITY_GROSS_GATE_FLOOR_TIME", gross_gate_floor) or gross_gate_floor)
        except Exception:
            gross_gate_floor_time = gross_gate_floor

        exits_seen = 0
        reject_seen = 0
        roe_samples: list[float] = []
        gross_roe_samples: list[float] = []
        hit_count = 0
        exits_seen_time = 0
        reject_seen_time = 0
        roe_samples_time: list[float] = []
        gross_roe_samples_time: list[float] = []
        hit_count_time = 0
        now_hour = int((int(now_ts) // 3_600_000) % 24)
        def _hour_ok(ts_value) -> bool:
            if not time_filter_enabled:
                return False
            try:
                rec_ts = int(float(ts_value))
            except Exception:
                return False
            if rec_ts <= 0:
                return False
            rec_hour = int((rec_ts // 3_600_000) % 24)
            dist = abs(int(rec_hour) - int(now_hour))
            dist = min(dist, 24 - dist)
            return bool(dist <= int(time_window_hours))
        tape = list(self.trade_tape or [])
        for rec in reversed(tape):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("symbol") or "") != str(sym):
                continue
            ttype = str(rec.get("ttype") or rec.get("type") or rec.get("action") or "").upper()
            rec_ts = rec.get("timestamp_ms")
            time_ok = _hour_ok(rec_ts)
            if ttype == "ORDER_REJECT":
                if reject_seen < reject_lookback:
                    reject_seen += 1
                if time_ok and reject_seen_time < reject_lookback:
                    reject_seen_time += 1
            elif ttype == "EXIT":
                if exits_seen < exit_lookback:
                    reason_l = str(rec.get("reason") or "").strip().lower()
                    if (
                        "external_close" in reason_l
                        or "exchange_manual_close" in reason_l
                        or "manual_cleanup" in reason_l
                        or "rebalance" in reason_l
                    ):
                        continue
                    exits_seen += 1
                    roe_v = self._safe_float(rec.get("roe"), None)
                    if roe_v is None:
                        roe_v = self._safe_float(rec.get("realized_r"), None)
                    if roe_v is not None:
                        roe_f = float(roe_v)
                        roe_samples.append(roe_f)
                        if roe_f > 0:
                            hit_count += 1
                    gross_roe_val = None
                    try:
                        roe_net = self._safe_float(rec.get("roe"), None)
                        fee_v = self._safe_float(rec.get("fee"), 0.0) or 0.0
                        notional_v = self._safe_float(rec.get("notional"), None)
                        lev_v = self._safe_float(rec.get("leverage"), None)
                        if roe_net is not None:
                            gross_roe_val = float(roe_net)
                            if notional_v is not None and lev_v is not None and float(lev_v) > 0:
                                margin_v = float(notional_v) / max(float(lev_v), 1e-9)
                                if margin_v > 0:
                                    gross_roe_val = float(gross_roe_val + float(fee_v) / float(margin_v))
                        if gross_roe_val is None:
                            pnl_gross_v = self._safe_float(rec.get("pnl_gross"), None)
                            if pnl_gross_v is None:
                                pnl_gross_v = self._safe_float(rec.get("realized_pnl"), None)
                            if pnl_gross_v is not None and notional_v is not None and lev_v is not None and float(lev_v) > 0:
                                margin_v = float(notional_v) / max(float(lev_v), 1e-9)
                                if margin_v > 0:
                                    gross_roe_val = float(float(pnl_gross_v) / float(margin_v))
                    except Exception:
                        gross_roe_val = None
                    if gross_roe_val is not None:
                        gross_roe_samples.append(float(gross_roe_val))
                if time_ok and exits_seen_time < exit_lookback:
                    reason_l_t = str(rec.get("reason") or "").strip().lower()
                    if (
                        "external_close" in reason_l_t
                        or "exchange_manual_close" in reason_l_t
                        or "manual_cleanup" in reason_l_t
                        or "rebalance" in reason_l_t
                    ):
                        continue
                    exits_seen_time += 1
                    roe_t = self._safe_float(rec.get("roe"), None)
                    if roe_t is None:
                        roe_t = self._safe_float(rec.get("realized_r"), None)
                    if roe_t is not None:
                        roe_tf = float(roe_t)
                        roe_samples_time.append(roe_tf)
                        if roe_tf > 0:
                            hit_count_time += 1
                    gross_roe_time_val = None
                    try:
                        roe_net_t = self._safe_float(rec.get("roe"), None)
                        fee_t = self._safe_float(rec.get("fee"), 0.0) or 0.0
                        notional_t = self._safe_float(rec.get("notional"), None)
                        lev_t = self._safe_float(rec.get("leverage"), None)
                        if roe_net_t is not None:
                            gross_roe_time_val = float(roe_net_t)
                            if notional_t is not None and lev_t is not None and float(lev_t) > 0:
                                margin_t = float(notional_t) / max(float(lev_t), 1e-9)
                                if margin_t > 0:
                                    gross_roe_time_val = float(gross_roe_time_val + float(fee_t) / float(margin_t))
                        if gross_roe_time_val is None:
                            pnl_gross_t = self._safe_float(rec.get("pnl_gross"), None)
                            if pnl_gross_t is None:
                                pnl_gross_t = self._safe_float(rec.get("realized_pnl"), None)
                            if pnl_gross_t is not None and notional_t is not None and lev_t is not None and float(lev_t) > 0:
                                margin_t = float(notional_t) / max(float(lev_t), 1e-9)
                                if margin_t > 0:
                                    gross_roe_time_val = float(float(pnl_gross_t) / float(margin_t))
                    except Exception:
                        gross_roe_time_val = None
                    if gross_roe_time_val is not None:
                        gross_roe_samples_time.append(float(gross_roe_time_val))
            if exits_seen >= exit_lookback and reject_seen >= reject_lookback:
                break

        sample_n = int(len(roe_samples))
        expectancy = float(sum(roe_samples) / sample_n) if sample_n > 0 else None
        gross_sample_n = int(len(gross_roe_samples))
        gross_expectancy = float(sum(gross_roe_samples) / gross_sample_n) if gross_sample_n > 0 else None
        hit_rate = float(hit_count / sample_n) if sample_n > 0 else None
        denom = float(max(1, exits_seen + reject_seen))
        reject_ratio = float(reject_seen / denom)
        sample_n_time = int(len(roe_samples_time))
        expectancy_time = float(sum(roe_samples_time) / sample_n_time) if sample_n_time > 0 else None
        gross_sample_n_time = int(len(gross_roe_samples_time))
        gross_expectancy_time = float(sum(gross_roe_samples_time) / gross_sample_n_time) if gross_sample_n_time > 0 else None
        hit_rate_time = float(hit_count_time / sample_n_time) if sample_n_time > 0 else None
        denom_time = float(max(1, exits_seen_time + reject_seen_time))
        reject_ratio_time = float(reject_seen_time / denom_time) if time_filter_enabled else None
        hard_reject_count = 0
        try:
            hard_reject_count = int((self._balance_reject_110007_by_sym.get(sym) or {}).get("count", 0) or 0)
        except Exception:
            hard_reject_count = 0
        if expectancy is None:
            exp_score = 0.5
        else:
            exp_scale = max(1e-6, float(abs(exp_ref)))
            exp_score = float(0.5 + 0.5 * math.tanh(float(expectancy) / exp_scale))
        hit_score = float(max(0.0, min(1.0, hit_rate if hit_rate is not None else 0.5)))
        reject_scale = max(1e-6, float(reject_ref))
        reject_score = float(max(0.0, min(1.0, 1.0 - (float(reject_ratio) / reject_scale))))
        reject_score = float(max(0.0, reject_score - min(0.45, 0.04 * float(max(0, hard_reject_count)))))
        score_base = float(w_exp * exp_score + w_hit * hit_score + w_rej * reject_score)
        gross_score = None
        if gross_filter_enabled:
            if gross_expectancy is None:
                gross_score = 0.5
            else:
                gross_scale = max(1e-6, float(abs(gross_ref)))
                gross_score = float(0.5 + 0.5 * math.tanh(float(gross_expectancy) / gross_scale))
            gw = float(max(0.0, min(0.80, gross_weight)))
            score_base = float((1.0 - gw) * score_base + gw * float(gross_score))

        score_time = None
        gross_time_score = None
        if time_filter_enabled and sample_n_time >= int(max(1, time_min_exits)):
            if expectancy_time is None:
                exp_score_t = 0.5
            else:
                exp_scale_t = max(1e-6, float(abs(exp_ref)))
                exp_score_t = float(0.5 + 0.5 * math.tanh(float(expectancy_time) / exp_scale_t))
            hit_score_t = float(max(0.0, min(1.0, hit_rate_time if hit_rate_time is not None else 0.5)))
            rej_ratio_t = float(reject_ratio_time) if reject_ratio_time is not None else float(reject_ratio)
            reject_score_t = float(max(0.0, min(1.0, 1.0 - (rej_ratio_t / reject_scale))))
            score_time = float(w_exp * exp_score_t + w_hit * hit_score_t + w_rej * reject_score_t)
            if gross_filter_enabled:
                if gross_expectancy_time is None:
                    gross_time_score = 0.5
                else:
                    gross_scale_t = max(1e-6, float(abs(gross_ref)))
                    gross_time_score = float(0.5 + 0.5 * math.tanh(float(gross_expectancy_time) / gross_scale_t))
                gtw = float(max(0.0, min(0.80, gross_time_weight)))
                score_time = float((1.0 - gtw) * score_time + gtw * float(gross_time_score))

        score = float(score_base)
        if score_time is not None:
            tw = float(max(0.0, min(0.80, time_weight)))
            score = float((1.0 - tw) * score + tw * float(score_time))

        # Apply hard penalties only when quality is materially weak.
        if expectancy is not None and float(expectancy) < float(hard_neg_exp_floor):
            exp_denom = max(1e-6, abs(float(hard_neg_exp_floor)))
            exp_def = float(max(0.0, float(hard_neg_exp_floor) - float(expectancy)) / exp_denom)
            score = float(score - min(0.45, max(0.0, float(hard_neg_exp_penalty)) * min(1.0, exp_def)))
        if float(reject_ratio) > float(hard_reject_floor):
            rej_def = float(max(0.0, float(reject_ratio) - float(hard_reject_floor)) / max(1e-6, 1.0 - float(hard_reject_floor)))
            score = float(score - min(0.40, max(0.0, float(hard_reject_penalty)) * min(1.0, rej_def)))
        if gross_filter_enabled and gross_expectancy is not None and float(gross_expectancy) < float(gross_hard_floor):
            gross_denom = max(1e-6, abs(float(gross_hard_floor)))
            gross_def = float(max(0.0, float(gross_hard_floor) - float(gross_expectancy)) / gross_denom)
            score = float(score - min(0.40, max(0.0, float(gross_hard_penalty)) * min(1.0, gross_def)))
        score = float(max(0.0, min(1.0, score)))
        quality_ok = None if sample_n < int(max(1, min_exits)) else bool(score >= float(min_score))
        if quality_ok is not None and quality_ok and gross_filter_enabled and gross_expectancy is not None:
            if float(gross_expectancy) < float(gross_gate_floor):
                quality_ok = False
        if (
            quality_ok is not None
            and quality_ok
            and time_filter_enabled
            and gross_filter_enabled
            and sample_n_time >= int(max(1, time_min_exits))
            and gross_expectancy_time is not None
        ):
            if float(gross_expectancy_time) < float(gross_gate_floor_time):
                quality_ok = False
        payload = {
            "ts": int(now_ts),
            "enabled": True,
            "ok": quality_ok,
            "score": float(max(0.0, min(1.0, score))),
            "sample_n": int(sample_n),
            "sample_exits": int(exits_seen),
            "sample_rejects": int(reject_seen),
            "expectancy": expectancy,
            "gross_expectancy": gross_expectancy,
            "gross_sample_n": int(gross_sample_n),
            "gross_score": float(gross_score) if gross_score is not None else None,
            "hit_rate": hit_rate,
            "reject_ratio": float(reject_ratio),
            "time_filter_enabled": bool(time_filter_enabled),
            "time_bucket_hour_utc": int(now_hour),
            "time_window_hours": int(time_window_hours),
            "time_sample_n": int(sample_n_time),
            "time_expectancy": expectancy_time,
            "time_gross_expectancy": gross_expectancy_time,
            "time_gross_sample_n": int(gross_sample_n_time),
            "time_gross_score": float(gross_time_score) if gross_time_score is not None else None,
            "time_hit_rate": hit_rate_time,
            "time_reject_ratio": float(reject_ratio_time) if reject_ratio_time is not None else None,
            "time_score": float(score_time) if score_time is not None else None,
            "gross_filter_enabled": bool(gross_filter_enabled),
            "gross_weight": float(gross_weight),
            "gross_time_weight": float(gross_time_weight),
            "gross_gate_floor": float(gross_gate_floor),
            "gross_gate_floor_time": float(gross_gate_floor_time),
            "hard_reject_count": int(hard_reject_count),
            "min_exits": int(max(1, min_exits)),
            "min_score": float(min_score),
        }
        self._symbol_quality_cache_by_sym[str(sym)] = payload
        return payload

    def _pre_mc_status(self) -> dict:
        def _maybe_float(val):
            try:
                if val is None:
                    return None
                return float(val)
            except Exception:
                return None

        status = {
            "active": bool(PRE_MC_ENABLED),
            "ok": None,
            "expected_pnl": None,
            "expected_pnl_source": None,
            "cvar": None,
            "prob_liq": None,
            "prob_liq_source": None,
            "min_expected_pnl": float(PRE_MC_MIN_EXPECTED_PNL),
            "min_cvar": float(PRE_MC_MIN_CVAR),
            "max_liq_prob": float(PRE_MC_MAX_LIQ_PROB),
            "reason": None,
        }

        if not status["active"]:
            status["reason"] = "disabled"
            return status

        report = self._last_portfolio_report
        if not report:
            status["reason"] = "report_missing"
            return status

        exp_pnl = report.get("rebalance_expected_portfolio_pnl")
        exp_src = "rebalance"
        if exp_pnl is None:
            exp_pnl = report.get("expected_portfolio_pnl")
            exp_src = "base"
        cvar = report.get("cvar")
        liq_prob = report.get("rebalance_prob_account_liq_proxy")
        liq_src = "rebalance"
        if liq_prob is None:
            liq_prob = report.get("prob_account_liquidation_proxy")
            liq_src = "base"

        exp_pnl = _maybe_float(exp_pnl)
        cvar = _maybe_float(cvar)
        liq_prob = _maybe_float(liq_prob)

        status["expected_pnl"] = exp_pnl
        status["expected_pnl_source"] = exp_src if exp_pnl is not None else None
        status["cvar"] = cvar
        status["prob_liq"] = liq_prob
        status["prob_liq_source"] = liq_src if liq_prob is not None else None

        metrics_present = any(m is not None for m in (exp_pnl, cvar, liq_prob))
        if not metrics_present:
            status["reason"] = "metrics_missing"
            return status

        ok = True
        reasons = []
        if exp_pnl is not None and exp_pnl < float(PRE_MC_MIN_EXPECTED_PNL):
            ok = False
            reasons.append("expected_pnl")
        if cvar is not None and cvar < float(PRE_MC_MIN_CVAR):
            ok = False
            reasons.append("cvar")
        if liq_prob is not None and liq_prob > float(PRE_MC_MAX_LIQ_PROB):
            ok = False
            reasons.append("liq_prob")

        status["ok"] = ok
        status["reason"] = "ok" if ok else ",".join(reasons)
        return status

    def _can_enter_position(self, notional: float) -> tuple[bool, str]:
        if self.safety_mode:
            return False, "safety mode"
        if self.position_cap_enabled and len(self.positions) >= self.max_positions:
            return False, "max positions reached"
        if self.exposure_cap_enabled and (self._total_open_notional() + notional) > (self._exposure_cap_balance() * self.max_notional_frac):
            return False, "exposure capped"
        return True, ""

    def _min_filter_states(self, sym: str, decision: dict, ts_ms: int) -> dict:
        meta = (decision.get("meta") or {}) if decision else {}
        regime = str(meta.get("regime") or "chop")
        try:
            pos_now = self.positions.get(sym) or {}
            pos_qty_now = float(pos_now.get("quantity", pos_now.get("qty", 0.0)) or 0.0)
        except Exception:
            pos_qty_now = 0.0
        has_pos = pos_qty_now != 0.0
        is_entry = bool(decision and decision.get("action") in ("LONG", "SHORT") and (not has_pos))
        spread_pct = meta.get("spread_pct")
        event_cvar_r = meta.get("event_cvar_r")
        event_exit_score = meta.get("event_exit_score")
        event_cvar_pct = meta.get("event_cvar_pct")
        event_p_sl = meta.get("event_p_sl")
        event_p_tp = meta.get("event_p_tp")
        use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
        hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
        hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")
        regime_raw = str(
            meta.get("event_exit_dynamic_regime_bucket")
            or meta.get("regime")
            or (decision.get("regime") if isinstance(decision, dict) else None)
            or regime
            or "chop"
        ).strip().lower()
        regime_keys: list[str] = []
        if ("bull" in regime_raw) or ("trend" in regime_raw):
            regime_keys.extend(["BULL", "TREND"])
        elif "bear" in regime_raw:
            regime_keys.extend(["BEAR", "TREND"])
        elif ("mean" in regime_raw) or ("revert" in regime_raw):
            regime_keys.extend(["MEAN_REVERT", "CHOP"])
        elif ("volatile" in regime_raw) or ("random" in regime_raw) or ("noise" in regime_raw):
            regime_keys.extend(["VOLATILE", "RANDOM"])
        else:
            regime_keys.append("CHOP")
        if "TREND" in regime_keys:
            regime_keys = [k for k in regime_keys if k != "CHOP"] + ["TREND"]
        regime_keys = list(dict.fromkeys([str(k).upper() for k in regime_keys if str(k).strip()]))

        def _env_regime_float(base_key: str, default: float) -> float:
            keys = [f"{str(base_key).upper()}_{rk}" for rk in regime_keys]
            keys.append(str(base_key).upper())
            for k in keys:
                try:
                    raw = os.environ.get(k)
                    if raw is None or str(raw).strip() == "":
                        continue
                    return float(raw)
                except Exception:
                    continue
            return float(default)

        def _env_regime_bool(base_key: str, default: bool) -> bool:
            keys = [f"{str(base_key).upper()}_{rk}" for rk in regime_keys]
            keys.append(str(base_key).upper())
            for k in keys:
                try:
                    raw = os.environ.get(k)
                    if raw is None or str(raw).strip() == "":
                        continue
                    return str(raw).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    continue
            return bool(default)
        unified_score = None
        if hybrid_only:
            unified_score = decision.get("hybrid_score") if decision else None
            if unified_score is None:
                unified_score = meta.get("hybrid_score")
        if unified_score is None:
            unified_score = decision.get("unified_score") if decision else None
        if unified_score is None:
            unified_score = meta.get("unified_score")
        if unified_score is None:
            unified_score = decision.get("ev", 0.0) if decision else 0.0

        # Thresholds
        if hybrid_only:
            unified_floor = self._get_hybrid_entry_floor()
        else:
            unified_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
        if MIN_ENTRY_SCORE > 0:
            unified_floor = float(max(unified_floor, MIN_ENTRY_SCORE))
        spread_cap_map = {"bull": 0.0020, "bear": 0.0020, "chop": 0.0012, "volatile": 0.0008}
        spread_cap = spread_cap_map.get(regime, SPREAD_PCT_MAX)
        cvar_floor_map = {"bull": -1.2, "bear": -1.2, "chop": -1.0, "volatile": -0.8}
        cvar_floor_regime = cvar_floor_map.get(regime, -1.0)

        unified_ok = float(unified_score or 0.0) >= unified_floor
        spread_enabled = spread_pct is not None
        spread_ok = (float(spread_pct) <= float(spread_cap)) if spread_enabled else None
        if hybrid_only:
            event_cvar_ok = None
        else:
            event_cvar_ok = True if event_cvar_r is None else float(event_cvar_r) >= float(cvar_floor_regime)
        exit_policy_dyn, exit_policy_diag = self._build_dynamic_exit_policy(
            sym,
            pos_now if has_pos else None,
            decision,
            ctx={"regime": regime},
        )
        min_score_thr_val = self._safe_float(meta.get("event_exit_min_score"), float(exit_policy_dyn.min_event_score))
        max_cvar_thr_val = self._safe_float(meta.get("event_exit_max_cvar"), float(exit_policy_dyn.max_abs_event_cvar_r))
        max_p_sl_thr_val = self._safe_float(meta.get("event_exit_max_p_sl"), float(exit_policy_dyn.max_event_p_sl))
        min_p_tp_thr_val = self._safe_float(meta.get("event_exit_min_p_tp"), float(exit_policy_dyn.min_event_p_tp))
        min_score_thr = float(min_score_thr_val if min_score_thr_val is not None else exit_policy_dyn.min_event_score)
        max_cvar_thr = float(max_cvar_thr_val if max_cvar_thr_val is not None else exit_policy_dyn.max_abs_event_cvar_r)
        max_p_sl_thr = float(max_p_sl_thr_val if max_p_sl_thr_val is not None else exit_policy_dyn.max_event_p_sl)
        min_p_tp_thr = float(min_p_tp_thr_val if min_p_tp_thr_val is not None else exit_policy_dyn.min_event_p_tp)
        exit_policy_dyn.min_event_score = min_score_thr
        exit_policy_dyn.max_abs_event_cvar_r = max_cvar_thr
        exit_policy_dyn.max_event_p_sl = max_p_sl_thr
        exit_policy_dyn.min_event_p_tp = min_p_tp_thr
        self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_policy_diag)
        if isinstance(decision, dict):
            decision["meta"] = meta
        event_exit_ok = None
        if is_entry:
            event_exit_ok = self._coerce_opt_bool(meta.get("event_exit_ok"))
            if event_exit_ok is not None and isinstance(meta, dict):
                meta["event_exit_eval_source"] = "entry_precheck_gate"
                decision["meta"] = meta
        if event_exit_ok is None:
            try:
                if event_exit_score is not None or event_cvar_pct is not None or event_p_sl is not None or event_p_tp is not None:
                    event_exit_ok = True
                    if event_exit_score is not None and float(event_exit_score) <= float(min_score_thr):
                        event_exit_ok = False
                    if event_cvar_pct is not None and abs(float(event_cvar_pct)) >= float(max_cvar_thr):
                        event_exit_ok = False
                    if event_p_sl is not None and float(event_p_sl) >= float(max_p_sl_thr):
                        event_exit_ok = False
                    if event_p_tp is not None and float(event_p_tp) <= float(min_p_tp_thr):
                        event_exit_ok = False
            except Exception:
                event_exit_ok = None
            if isinstance(meta, dict) and event_exit_ok is not None:
                meta["event_exit_eval_source"] = "threshold_only"
                decision["meta"] = meta
        if is_entry:
            try:
                entry_event_filter_on = str(
                    os.environ.get("ENTRY_EVENT_EXIT_FILTER_ENABLED", "1")
                ).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                entry_event_filter_on = True
            if not entry_event_filter_on:
                event_exit_ok = None
                if isinstance(meta, dict):
                    meta["entry_event_exit_filter_enabled"] = False
                    decision["meta"] = meta
            elif isinstance(meta, dict):
                meta["entry_event_exit_filter_enabled"] = True
                decision["meta"] = meta

        # Allow exceptionally strong entries to bypass strict pre-entry event filter.
        if is_entry and event_exit_ok is False:
            try:
                bypass_on = str(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                bypass_on = True
            if bypass_on:
                conf_now = self._safe_float(decision.get("confidence"), None)
                if conf_now is None:
                    conf_now = self._safe_float(meta.get("confidence"), 0.0)
                dir_conf_now = self._safe_float(meta.get("mu_dir_conf"), None)
                if dir_conf_now is None:
                    dir_conf_now = self._safe_float(decision.get("mu_dir_conf"), 0.0)
                dir_edge_now = abs(self._safe_float(meta.get("mu_dir_edge"), 0.0) or 0.0)
                ev_gap_now = abs(
                    self._safe_float(meta.get("policy_ev_gap"), None)
                    or self._safe_float(meta.get("ev_gap"), None)
                    or 0.0
                )
                entry_q_now = self._safe_float(meta.get("entry_quality_score"), 0.0) or 0.0
                p_sl_now = self._safe_float(event_p_sl, None)
                p_tp_now = self._safe_float(event_p_tp, None)
                try:
                    min_conf = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_MIN_CONF", 0.72) or 0.72)
                except Exception:
                    min_conf = 0.72
                try:
                    min_dir_conf = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_MIN_DIR_CONF", 0.60) or 0.60)
                except Exception:
                    min_dir_conf = 0.60
                try:
                    min_edge = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_MIN_DIR_EDGE", 0.12) or 0.12)
                except Exception:
                    min_edge = 0.12
                try:
                    min_gap = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_MIN_EV_GAP", 0.0018) or 0.0018)
                except Exception:
                    min_gap = 0.0018
                try:
                    min_entry_q = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_MIN_ENTRY_Q", 0.70) or 0.70)
                except Exception:
                    min_entry_q = 0.70
                try:
                    hard_psl = float(os.environ.get("ENTRY_EVENT_EXIT_BYPASS_HARD_PSL", 0.985) or 0.985)
                except Exception:
                    hard_psl = 0.985
                strong_entry = bool(
                    float(conf_now or 0.0) >= float(min_conf)
                    and float(dir_conf_now or 0.0) >= float(min_dir_conf)
                    and float(dir_edge_now or 0.0) >= float(min_edge)
                    and float(ev_gap_now or 0.0) >= float(min_gap)
                    and float(entry_q_now or 0.0) >= float(min_entry_q)
                )
                hard_adverse = bool(
                    (p_sl_now is not None and float(p_sl_now) >= float(hard_psl))
                    or (
                        p_sl_now is not None
                        and p_tp_now is not None
                        and float(p_tp_now) <= float(p_sl_now) * 0.65
                    )
                )
                if strong_entry and (not hard_adverse):
                    event_exit_ok = True
                    meta["event_entry_bypass"] = True
                    meta["event_entry_bypass_conf"] = float(conf_now or 0.0)
                    meta["event_entry_bypass_dir_conf"] = float(dir_conf_now or 0.0)
                    meta["event_entry_bypass_dir_edge"] = float(dir_edge_now or 0.0)
                    meta["event_entry_bypass_ev_gap"] = float(ev_gap_now or 0.0)
                    meta["event_entry_bypass_entry_q"] = float(entry_q_now or 0.0)
                    if isinstance(decision, dict):
                        decision["meta"] = meta
        liq_ok = None
        if MIN_LIQ_SCORE > 0:
            try:
                ob = self.orderbook.get(sym)
                if not ob or not ob.get("ready"):
                    liq_ok = None
                else:
                    liq_ok = float(self._liquidity_score(sym)) >= float(MIN_LIQ_SCORE)
            except Exception:
                liq_ok = None
        min_notional_ok = None
        min_exposure_ok = None
        tick_vol_ok = None
        both_ev_neg_ok = None
        gross_ev_ok = None
        net_expectancy_ok = None
        dir_gate_ok = None
        symbol_quality_ok = None
        lev_floor_lock_ok = None
        fee_ok = None
        fee_cost = None
        fee_ev = None
        fee_roundtrip = None
        fee_mult = None
        # Entry capacity/safety gate (exposure cap / max positions / safety mode)
        cap_safety_ok = not bool(self.safety_mode)
        cap_positions_ok = None
        if self.position_cap_enabled:
            cap_positions_ok = len(self.positions) < self.max_positions
        cap_exposure_ok = None
        try:
            lev_val = decision.get("leverage") or meta.get("leverage") or self._dyn_leverage.get(sym) or self.leverage
            lev_val = float(lev_val) if lev_val else float(self.leverage)
        except Exception:
            lev_val = float(self.leverage)
        try:
            min_lev = float(LEVERAGE_MIN)
        except Exception:
            min_lev = 1.0
        try:
            max_lev = float(self.max_leverage or lev_val)
        except Exception:
            max_lev = lev_val
        if min_lev > 0:
            lev_val = max(min_lev, lev_val)
        if max_lev > 0:
            lev_val = min(max_lev, lev_val)
        if is_entry:
            try:
                lev_floor_lock_on = str(os.environ.get("LEVERAGE_FLOOR_LOCK_ENTRY_BLOCK", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                lev_floor_lock_on = True
            if lev_floor_lock_on:
                try:
                    lev_floor_ref = float(os.environ.get("LEVERAGE_TARGET_MIN", min_lev) or min_lev)
                except Exception:
                    lev_floor_ref = float(min_lev)
                lev_floor_ref = max(float(min_lev), float(lev_floor_ref))
                try:
                    lev_floor_eps = float(os.environ.get("LEVERAGE_FLOOR_LOCK_EPS", 0.001) or 0.001)
                except Exception:
                    lev_floor_eps = 0.001
                try:
                    lev_floor_min_sticky = int(os.environ.get("LEVERAGE_FLOOR_LOCK_MIN_STICKY", 3) or 3)
                except Exception:
                    lev_floor_min_sticky = 3
                try:
                    lev_floor_max_gap = float(os.environ.get("LEVERAGE_FLOOR_LOCK_MAX_EV_GAP", 0.0008) or 0.0008)
                except Exception:
                    lev_floor_max_gap = 0.0008
                try:
                    lev_floor_max_conf = float(os.environ.get("LEVERAGE_FLOOR_LOCK_MAX_CONF", 0.60) or 0.60)
                except Exception:
                    lev_floor_max_conf = 0.60
                try:
                    lev_floor_max_dir_conf = float(os.environ.get("LEVERAGE_FLOOR_LOCK_MAX_DIR_CONF", 0.58) or 0.58)
                except Exception:
                    lev_floor_max_dir_conf = 0.58
                try:
                    lev_floor_max_score_edge = float(os.environ.get("LEVERAGE_FLOOR_LOCK_MAX_SCORE_EDGE", 0.0004) or 0.0004)
                except Exception:
                    lev_floor_max_score_edge = 0.0004
                try:
                    sticky_cnt = int(self._lev_floor_sticky_counts.get(sym, 0) or 0)
                except Exception:
                    sticky_cnt = 0
                ev_gap_now = self._safe_float(meta.get("policy_ev_gap"), None)
                if ev_gap_now is None:
                    ev_gap_now = self._safe_float(decision.get("ev_gap"), None)
                if ev_gap_now is None:
                    ev_gap_now = self._safe_float(meta.get("ev_gap"), 0.0)
                dir_conf_now = self._safe_float(meta.get("mu_dir_conf"), None)
                if dir_conf_now is None:
                    dir_conf_now = self._safe_float(decision.get("mu_dir_conf"), 0.0)
                conf_now = self._safe_float(decision.get("confidence"), 0.0)
                score_edge = float((unified_score or 0.0) - (unified_floor or 0.0))
                near_floor = float(lev_val) <= float(lev_floor_ref + max(0.0, lev_floor_eps))
                weak_signal = (
                    abs(float(ev_gap_now or 0.0)) <= float(max(0.0, lev_floor_max_gap))
                    and float(conf_now or 0.0) <= float(max(0.0, lev_floor_max_conf))
                    and float(dir_conf_now or 0.0) <= float(max(0.0, lev_floor_max_dir_conf))
                    and float(score_edge) <= float(max(0.0, lev_floor_max_score_edge))
                )
                lev_floor_lock_ok = not (near_floor and weak_signal and int(sticky_cnt) >= int(max(1, lev_floor_min_sticky)))
                try:
                    meta["lev_floor_lock_ok"] = bool(lev_floor_lock_ok)
                    meta["lev_floor_lock_near_floor"] = bool(near_floor)
                    meta["lev_floor_lock_weak_signal"] = bool(weak_signal)
                    meta["lev_floor_lock_score_edge"] = float(score_edge)
                    meta["lev_floor_lock_sticky_cnt"] = int(sticky_cnt)
                    meta["lev_floor_lock_min_sticky"] = int(max(1, lev_floor_min_sticky))
                    meta["lev_floor_lock_floor"] = float(lev_floor_ref)
                    decision["meta"] = meta
                except Exception:
                    pass
            else:
                lev_floor_lock_ok = True
        try:
            _, est_notional, _ = self._calc_position_size(decision, 1.0, lev_val, symbol=sym, use_cycle_reserve=is_entry)
        except Exception:
            try:
                size_frac = decision.get("size_frac") or meta.get("size_fraction") or self.default_size_frac
                size_frac = float(size_frac or 0.0)
            except Exception:
                size_frac = float(self.default_size_frac or 0.0)
            est_notional = max(0.0, float(self.balance) * float(size_frac) * float(lev_val))
        eff_balance = self._sizing_balance()
        # Exposure cap should be based on total equity (wallet + unrealized).
        try:
            cap_balance = float(self._exposure_cap_balance())
        except Exception:
            cap_balance = float(self.balance) if self.balance is not None else float(eff_balance or 0.0)
        min_notional_floor = float(MIN_ENTRY_NOTIONAL) if MIN_ENTRY_NOTIONAL > 0 else 0.0
        if MIN_ENTRY_NOTIONAL_PCT > 0:
            try:
                min_notional_floor = max(min_notional_floor, float(eff_balance) * float(MIN_ENTRY_NOTIONAL_PCT))
            except Exception:
                pass
        try:
            min_notional_hard_min = float(os.environ.get("MIN_ENTRY_NOTIONAL_HARD_MIN", 0.5) or 0.5)
        except Exception:
            min_notional_hard_min = 0.5
        min_notional_floor = float(max(0.0, min_notional_hard_min, min_notional_floor)) if min_notional_floor > 0 else 0.0
        try:
            min_notional_cap_ratio = float(os.environ.get("MIN_ENTRY_NOTIONAL_BALANCE_CAP_RATIO", 0.15) or 0.15)
        except Exception:
            min_notional_cap_ratio = 0.15
        if min_notional_floor > 0 and float(min_notional_cap_ratio) > 0:
            try:
                cap_floor = float(max(0.0, float(eff_balance) * float(min_notional_cap_ratio)))
                if cap_floor > 0:
                    min_notional_floor = float(max(min_notional_hard_min, min(min_notional_floor, cap_floor)))
            except Exception:
                pass
        if min_notional_floor > 0:
            try:
                min_notional_ok = float(est_notional) >= float(min_notional_floor)
            except Exception:
                min_notional_ok = None
        min_exposure_floor = float(MIN_ENTRY_EXPOSURE_PCT) if MIN_ENTRY_EXPOSURE_PCT > 0 else 0.0
        if min_exposure_floor > 0:
            try:
                min_exposure_ok = float(est_notional) >= (float(cap_balance) * float(min_exposure_floor))
            except Exception:
                min_exposure_ok = None
        if MIN_TICK_VOL > 0:
            try:
                tick_vol = meta.get("tick_vol")
                tick_vol_ok = float(tick_vol) >= float(MIN_TICK_VOL)
            except Exception:
                tick_vol_ok = None
        if self.exposure_cap_enabled:
            try:
                cap_exposure_ok = (self._total_open_notional() + float(est_notional)) <= (float(cap_balance) * float(self.max_notional_frac))
            except Exception:
                cap_exposure_ok = None
        if is_entry:
            side_now = str((decision or {}).get("action") or "").upper()
            long_edge_raw = self._decision_metric_float(
                decision,
                meta,
                ("policy_ev_mix_long", "policy_ev_score_long", "unified_score_long", "hybrid_score_long"),
                default=None,
            )
            short_edge_raw = self._decision_metric_float(
                decision,
                meta,
                ("policy_ev_mix_short", "policy_ev_score_short", "unified_score_short", "hybrid_score_short"),
                default=None,
            )
            long_edge = float(long_edge_raw) if long_edge_raw is not None else None
            short_edge = float(short_edge_raw) if short_edge_raw is not None else None
            try:
                base_fee = float(os.environ.get("HYBRID_BASE_FEE_RATE", 0.0002) or 0.0002)
            except Exception:
                base_fee = 0.0002
            try:
                slippage_bps = float(os.environ.get("HYBRID_SLIPPAGE_BPS", 0.0) or 0.0)
            except Exception:
                slippage_bps = 0.0
            fee_est = float(2.0 * (base_fee + slippage_bps * 1e-4))
            long_gross = float(long_edge + fee_est) if long_edge is not None else None
            short_gross = float(short_edge + fee_est) if short_edge is not None else None
            side_net = long_edge if side_now == "LONG" else short_edge if side_now == "SHORT" else None
            side_gross = long_gross if side_now == "LONG" else short_gross if side_now == "SHORT" else None
            max_net = None
            max_gross = None
            if long_edge is not None and short_edge is not None:
                max_net = float(max(long_edge, short_edge))
            elif long_edge is not None:
                max_net = float(long_edge)
            elif short_edge is not None:
                max_net = float(short_edge)
            if long_gross is not None and short_gross is not None:
                max_gross = float(max(long_gross, short_gross))
            elif long_gross is not None:
                max_gross = float(long_gross)
            elif short_gross is not None:
                max_gross = float(short_gross)

            both_ev_neg_on = _env_regime_bool("ENTRY_BOTH_EV_NEG_FILTER_ENABLED", True)
            both_ev_neg_floor = _env_regime_float("ENTRY_BOTH_EV_NEG_NET_FLOOR", 0.0)
            if both_ev_neg_on:
                if long_edge is not None and short_edge is not None:
                    both_ev_neg_ok = bool(
                        not (
                            float(long_edge) <= float(both_ev_neg_floor)
                            and float(short_edge) <= float(both_ev_neg_floor)
                        )
                    )
                else:
                    both_ev_neg_ok = None

            gross_filter_on = _env_regime_bool("ENTRY_GROSS_EV_FILTER_ENABLED", True)
            gross_floor = _env_regime_float("ENTRY_GROSS_EV_MIN", 0.0)
            gross_side_only = _env_regime_bool("ENTRY_GROSS_EV_REQUIRE_SIDE", True)
            if gross_filter_on:
                gross_ref = side_gross if gross_side_only else max_gross
                if gross_ref is None and (not gross_side_only):
                    gross_ref = side_gross
                gross_ev_ok = (None if gross_ref is None else bool(float(gross_ref) >= float(gross_floor)))

            if isinstance(meta, dict):
                meta["entry_ev_long_net"] = float(long_edge) if long_edge is not None else None
                meta["entry_ev_short_net"] = float(short_edge) if short_edge is not None else None
                meta["entry_ev_side_net"] = float(side_net) if side_net is not None else None
                meta["entry_ev_max_net"] = float(max_net) if max_net is not None else None
                meta["entry_ev_long_gross"] = float(long_gross) if long_gross is not None else None
                meta["entry_ev_short_gross"] = float(short_gross) if short_gross is not None else None
                meta["entry_ev_side_gross"] = float(side_gross) if side_gross is not None else None
                meta["entry_ev_max_gross"] = float(max_gross) if max_gross is not None else None
                meta["entry_ev_fee_est"] = float(fee_est)
                meta["entry_both_ev_neg_filter_enabled"] = bool(both_ev_neg_on)
                meta["entry_both_ev_neg_floor"] = float(both_ev_neg_floor)
                meta["entry_both_ev_neg_ok"] = both_ev_neg_ok
                meta["entry_gross_filter_enabled"] = bool(gross_filter_on)
                meta["entry_gross_min"] = float(gross_floor)
                meta["entry_gross_require_side"] = bool(gross_side_only)
                meta["entry_gross_ok"] = gross_ev_ok
                decision["meta"] = meta

            try:
                net_filter_on = str(os.environ.get("ENTRY_NET_EXPECTANCY_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                net_filter_on = True
            if net_filter_on:
                side_now = str((decision or {}).get("action") or "").upper()
                edge_raw = None
                if side_now == "LONG":
                    edge_raw = self._decision_metric_float(
                        decision,
                        meta,
                        ("policy_ev_mix_long", "policy_ev_score_long", "unified_score_long", "hybrid_score_long"),
                        default=None,
                    )
                elif side_now == "SHORT":
                    edge_raw = self._decision_metric_float(
                        decision,
                        meta,
                        ("policy_ev_mix_short", "policy_ev_score_short", "unified_score_short", "hybrid_score_short"),
                        default=None,
                    )
                if edge_raw is None:
                    edge_raw = self._decision_metric_float(
                        decision,
                        meta,
                        ("event_ev_pct", "policy_ev_mix", "policy_ev_side", "pred_ev", "ev"),
                        default=None,
                    )

                net_floor = _env_regime_float("ENTRY_NET_EXPECTANCY_MIN", 0.0)
                vpin_start = _env_regime_float("ENTRY_NET_EXPECTANCY_VPIN_START", 0.65)
                vpin_bump = _env_regime_float("ENTRY_NET_EXPECTANCY_VPIN_BUMP", 0.0004)
                conf_ref = _env_regime_float("ENTRY_NET_EXPECTANCY_DIR_CONF_REF", 0.58)
                conf_bump = _env_regime_float("ENTRY_NET_EXPECTANCY_DIR_CONF_BUMP", 0.0004)
                floor_eff = float(net_floor)
                vpin_val = self._decision_metric_float(decision, meta, ("vpin", "alpha_vpin"), default=None)
                if vpin_val is not None:
                    vpin_clip = float(max(0.0, min(1.0, vpin_val)))
                    if vpin_clip > float(vpin_start):
                        vpin_scale = (float(vpin_clip) - float(vpin_start)) / max(1e-6, 1.0 - float(vpin_start))
                        floor_eff += float(max(0.0, vpin_bump) * vpin_scale)
                dir_conf_val = self._decision_metric_float(decision, meta, ("mu_dir_conf", "pred_mu_dir_conf"), default=None)
                if dir_conf_val is not None and float(conf_ref) > 1e-6 and float(dir_conf_val) < float(conf_ref):
                    conf_deficit = (float(conf_ref) - float(dir_conf_val)) / float(conf_ref)
                    floor_eff += float(max(0.0, conf_bump) * max(0.0, conf_deficit))
                net_edge = None if edge_raw is None else float(edge_raw) - float(fee_est)
                require_signal = _env_regime_bool("ENTRY_NET_EXPECTANCY_REQUIRE_SIGNAL", False)
                if net_edge is None:
                    net_expectancy_ok = (False if require_signal else None)
                else:
                    net_expectancy_ok = bool(float(net_edge) >= float(floor_eff))
                if isinstance(meta, dict):
                    meta["net_expectancy_filter_enabled"] = True
                    meta["net_expectancy_raw"] = float(edge_raw) if edge_raw is not None else None
                    meta["net_expectancy_fee_cost"] = float(fee_est)
                    meta["net_expectancy_effective"] = float(net_edge) if net_edge is not None else None
                    meta["net_expectancy_min"] = float(floor_eff)
                    meta["net_expectancy_vpin"] = float(vpin_val) if vpin_val is not None else None
                    meta["net_expectancy_dir_conf"] = float(dir_conf_val) if dir_conf_val is not None else None
                    meta["net_expectancy_regime"] = str(regime_raw)
                    meta["net_expectancy_regime_keys"] = list(regime_keys)
                    meta["net_expectancy_vpin_start"] = float(vpin_start)
                    meta["net_expectancy_vpin_bump"] = float(vpin_bump)
                    meta["net_expectancy_dir_conf_ref"] = float(conf_ref)
                    meta["net_expectancy_dir_conf_bump"] = float(conf_bump)
                    meta["net_expectancy_require_signal"] = bool(require_signal)
                    meta["net_expectancy_ok"] = net_expectancy_ok
                    decision["meta"] = meta
            if str(os.environ.get("SYMBOL_QUALITY_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on"):
                sq = self._symbol_quality_state(sym, int(ts_ms))
                symbol_quality_ok = sq.get("ok")
                if isinstance(meta, dict):
                    meta["symbol_quality_enabled"] = True
                    meta["symbol_quality_score"] = sq.get("score")
                    meta["symbol_quality_ok"] = sq.get("ok")
                    meta["symbol_quality_sample_n"] = sq.get("sample_n")
                    meta["symbol_quality_expectancy"] = sq.get("expectancy")
                    meta["symbol_quality_gross_expectancy"] = sq.get("gross_expectancy")
                    meta["symbol_quality_hit_rate"] = sq.get("hit_rate")
                    meta["symbol_quality_reject_ratio"] = sq.get("reject_ratio")
                    meta["symbol_quality_min_exits"] = sq.get("min_exits")
                    meta["symbol_quality_min_score"] = sq.get("min_score")
                    decision["meta"] = meta
            elif isinstance(meta, dict):
                meta["symbol_quality_enabled"] = False
                decision["meta"] = meta
        # Direction quality gate (stage-2): in small-gap zones, block low-confidence/low-edge entries.
        if is_entry:
            try:
                dir_gate_enabled = str(os.environ.get("ALPHA_DIRECTION_GATE_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                dir_gate_enabled = True
            if dir_gate_enabled and bool(getattr(mc_config, "alpha_direction_use", False)):
                try:
                    small_gap_thr = float(os.environ.get("ALPHA_DIRECTION_GATE_SMALL_GAP", 0.0012) or 0.0012)
                except Exception:
                    small_gap_thr = 0.0012
                try:
                    min_conf = float(os.environ.get("ALPHA_DIRECTION_GATE_MIN_CONF", 0.56) or 0.56)
                except Exception:
                    min_conf = 0.56
                try:
                    min_edge = float(os.environ.get("ALPHA_DIRECTION_GATE_MIN_EDGE", 0.08) or 0.08)
                except Exception:
                    min_edge = 0.08
                try:
                    min_side_prob = float(os.environ.get("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", 0.52) or 0.52)
                except Exception:
                    min_side_prob = 0.52
                try:
                    weak_mu_thr = float(getattr(mc_config, "alpha_direction_gate_weak_mu", 0.02) or 0.02)
                except Exception:
                    weak_mu_thr = 0.02
                try:
                    weak_gap_thr = float(getattr(mc_config, "alpha_direction_gate_weak_gap", 0.0020) or 0.0020)
                except Exception:
                    weak_gap_thr = 0.0020
                try:
                    conf_boost = float(getattr(mc_config, "alpha_direction_gate_conf_boost", 0.05) or 0.05)
                except Exception:
                    conf_boost = 0.05
                try:
                    edge_boost = float(getattr(mc_config, "alpha_direction_gate_edge_boost", 0.03) or 0.03)
                except Exception:
                    edge_boost = 0.03
                try:
                    side_prob_boost = float(getattr(mc_config, "alpha_direction_gate_side_prob_boost", 0.03) or 0.03)
                except Exception:
                    side_prob_boost = 0.03
                def _meta_or_decision_float(*keys: str, default: float = 0.0) -> float:
                    for k in keys:
                        try:
                            if isinstance(meta, dict) and meta.get(k) is not None:
                                return float(meta.get(k))
                        except Exception:
                            pass
                        try:
                            if isinstance(decision, dict) and decision.get(k) is not None:
                                return float(decision.get(k))
                        except Exception:
                            pass
                    details = decision.get("details") if isinstance(decision, dict) else None
                    if isinstance(details, list):
                        for d in details:
                            if not isinstance(d, dict):
                                continue
                            dm = d.get("meta")
                            if not isinstance(dm, dict):
                                continue
                            for k in keys:
                                try:
                                    if dm.get(k) is not None:
                                        return float(dm.get(k))
                                except Exception:
                                    pass
                    return float(default)

                def _meta_or_decision_raw(*keys: str):
                    for k in keys:
                        try:
                            if isinstance(meta, dict) and meta.get(k) is not None:
                                return meta.get(k)
                        except Exception:
                            pass
                        try:
                            if isinstance(decision, dict) and decision.get(k) is not None:
                                return decision.get(k)
                        except Exception:
                            pass
                    details = decision.get("details") if isinstance(decision, dict) else None
                    if isinstance(details, list):
                        for d in details:
                            if not isinstance(d, dict):
                                continue
                            dm = d.get("meta")
                            if not isinstance(dm, dict):
                                continue
                            for k in keys:
                                try:
                                    if dm.get(k) is not None:
                                        return dm.get(k)
                                except Exception:
                                    pass
                    return None

                ev_gap = _meta_or_decision_float("policy_ev_gap", "ev_gap", default=0.0)
                dir_conf = _meta_or_decision_float("mu_dir_conf", default=0.0)
                dir_edge = _meta_or_decision_float("mu_dir_edge", default=0.0)
                dir_prob_long = _meta_or_decision_float("mu_dir_prob_long", default=0.5)
                mu_alpha_abs = abs(_meta_or_decision_float("mu_alpha", "pred_mu_alpha", default=0.0))
                dir_conf_raw = _meta_or_decision_raw("mu_dir_conf")
                dir_edge_raw = _meta_or_decision_raw("mu_dir_edge")
                dir_prob_raw = _meta_or_decision_raw("mu_dir_prob_long")
                regime_now = str((meta or {}).get("regime") or regime or "").strip().lower()
                conf_req = float(min_conf)
                edge_req = float(min_edge)
                side_prob_req = float(min_side_prob)
                adaptive_strength = 0.0
                adaptive_hit = None
                adaptive_entry_issue = None
                adaptive_hit_delta = None
                adaptive_entry_issue_delta = None
                adaptive_enabled = False
                # Enforce sane lower bounds even when env thresholds are accidentally too permissive.
                conf_req = max(conf_req, float(getattr(mc_config, "alpha_direction_confidence_floor", 0.45) or 0.45))
                edge_req = max(edge_req, 0.05)
                side_prob_req = max(side_prob_req, 0.52)
                try:
                    adaptive_enabled = str(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_GATE", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    adaptive_enabled = True
                if adaptive_enabled:
                    try:
                        adapt_max_stale = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_MAX_STALE_SEC", 1800) or 1800)
                    except Exception:
                        adapt_max_stale = 1800.0
                    try:
                        adapt_hit_ref = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_HIT_REF", 0.53) or 0.53)
                    except Exception:
                        adapt_hit_ref = 0.53
                    try:
                        adapt_issue_ref = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_ENTRY_ISSUE_REF", 0.50) or 0.50)
                    except Exception:
                        adapt_issue_ref = 0.50
                    try:
                        adapt_hit_good = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_HIT_GOOD", 0.58) or 0.58)
                    except Exception:
                        adapt_hit_good = 0.58
                    try:
                        adapt_issue_good = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_ENTRY_ISSUE_GOOD", 0.40) or 0.40)
                    except Exception:
                        adapt_issue_good = 0.40
                    try:
                        adapt_conf_gain = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_CONF_GAIN", 0.06) or 0.06)
                    except Exception:
                        adapt_conf_gain = 0.06
                    try:
                        adapt_edge_gain = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_EDGE_GAIN", 0.04) or 0.04)
                    except Exception:
                        adapt_edge_gain = 0.04
                    try:
                        adapt_prob_gain = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_SIDE_PROB_GAIN", 0.03) or 0.03)
                    except Exception:
                        adapt_prob_gain = 0.03
                    try:
                        adapt_max_tighten = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_MAX_TIGHTEN", 0.30) or 0.30)
                    except Exception:
                        adapt_max_tighten = 0.30
                    try:
                        adapt_max_relax = float(os.environ.get("ALPHA_DIRECTION_ADAPTIVE_MAX_RELAX", 0.20) or 0.20)
                    except Exception:
                        adapt_max_relax = 0.20
                    try:
                        tune_status = self._auto_tune_status(int(ts_ms))
                    except Exception:
                        tune_status = {}
                    if not isinstance(tune_status, dict):
                        tune_status = {}
                    try:
                        stale_sec = self._safe_float(tune_status.get("stale_sec"), None)
                    except Exception:
                        stale_sec = None
                    if stale_sec is None or float(stale_sec) <= float(max(0.0, adapt_max_stale)):
                        adaptive_hit = self._safe_float(tune_status.get("direction_hit"), None)
                        adaptive_entry_issue = self._safe_float(
                            tune_status.get("entry_issue_ratio"),
                            self._safe_float(tune_status.get("entry_issue_rate"), None),
                        )
                        adaptive_hit_delta = self._safe_float(tune_status.get("direction_hit_delta"), None)
                        adaptive_entry_issue_delta = self._safe_float(tune_status.get("entry_issue_ratio_delta"), None)
                        if adaptive_hit is not None:
                            adaptive_strength += (float(adapt_hit_ref) - float(adaptive_hit)) / max(1e-6, float(adapt_hit_ref))
                        if adaptive_entry_issue is not None:
                            adaptive_strength += (float(adaptive_entry_issue) - float(adapt_issue_ref)) / max(1e-6, (1.0 - float(adapt_issue_ref)))
                        if adaptive_hit_delta is not None and float(adaptive_hit_delta) < 0.0:
                            adaptive_strength += abs(float(adaptive_hit_delta)) / max(1e-6, float(adapt_hit_ref))
                        if adaptive_entry_issue_delta is not None and float(adaptive_entry_issue_delta) > 0.0:
                            adaptive_strength += float(adaptive_entry_issue_delta) / max(1e-6, (1.0 - float(adapt_issue_ref)))
                        # Good recent performance relaxes thresholds slightly.
                        if (
                            adaptive_hit is not None
                            and adaptive_entry_issue is not None
                            and float(adaptive_hit) >= float(adapt_hit_good)
                            and float(adaptive_entry_issue) <= float(adapt_issue_good)
                        ):
                            relax_h = (float(adaptive_hit) - float(adapt_hit_good)) / max(1e-6, 1.0 - float(adapt_hit_good))
                            relax_i = (float(adapt_issue_good) - float(adaptive_entry_issue)) / max(1e-6, float(adapt_issue_good))
                            adaptive_strength -= 0.50 * max(0.0, min(relax_h, relax_i))
                        adaptive_strength = float(max(-abs(adapt_max_relax), min(abs(adapt_max_tighten), adaptive_strength)))
                        if adaptive_strength != 0.0:
                            conf_req = float(conf_req + float(adapt_conf_gain) * adaptive_strength)
                            edge_req = float(edge_req + float(adapt_edge_gain) * adaptive_strength)
                            side_prob_req = float(side_prob_req + float(adapt_prob_gain) * adaptive_strength)
                if regime_now in ("chop", "volatile"):
                    conf_req = float(min(0.99, conf_req + 0.02))
                    edge_req = float(min(0.99, edge_req + 0.01))
                    side_prob_req = float(min(0.99, side_prob_req + 0.01))
                weak_edge_zone = bool((mu_alpha_abs < max(0.0, weak_mu_thr)) and (abs(float(ev_gap)) < max(0.0, weak_gap_thr)))
                if weak_edge_zone:
                    conf_req = float(min(0.99, conf_req + max(0.0, conf_boost)))
                    edge_req = float(min(0.99, edge_req + max(0.0, edge_boost)))
                    side_prob_req = float(min(0.99, side_prob_req + max(0.0, side_prob_boost)))
                gate_needed = bool(abs(float(ev_gap)) < max(0.0, float(small_gap_thr)) or weak_edge_zone)
                dir_signal_available = bool(
                    (dir_conf_raw is not None)
                    or (dir_edge_raw is not None)
                    or (dir_prob_raw is not None)
                )
                if gate_needed:
                    try:
                        require_signal = str(os.environ.get("ALPHA_DIRECTION_GATE_REQUIRE_SIGNAL", "0")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        require_signal = False
                    if not dir_signal_available:
                        dir_gate_ok = False if require_signal else None
                    else:
                        side = str((decision or {}).get("action") or "").upper()
                        if side == "LONG":
                            side_prob = float(dir_prob_long)
                        elif side == "SHORT":
                            side_prob = float(1.0 - dir_prob_long)
                        else:
                            side_prob = 0.5
                        dir_gate_ok = bool(
                            (dir_conf >= float(conf_req))
                            and (abs(float(dir_edge)) >= float(edge_req))
                            and (float(side_prob) >= float(side_prob_req))
                        )
                else:
                    dir_gate_ok = True
                if isinstance(meta, dict):
                    meta["dir_gate_small_gap_thr"] = float(small_gap_thr)
                    meta["dir_gate_weak_mu_thr"] = float(weak_mu_thr)
                    meta["dir_gate_weak_gap_thr"] = float(weak_gap_thr)
                    meta["dir_gate_weak_zone"] = bool(weak_edge_zone)
                    meta["dir_gate_needed"] = bool(gate_needed)
                    meta["dir_gate_min_conf"] = float(conf_req)
                    meta["dir_gate_min_edge"] = float(edge_req)
                    meta["dir_gate_min_side_prob"] = float(side_prob_req)
                    meta["dir_gate_prob_long"] = float(dir_prob_long)
                    meta["dir_gate_mu_alpha_abs"] = float(mu_alpha_abs)
                    meta["dir_gate_signal_available"] = bool(dir_signal_available)
                    meta["dir_gate_require_signal"] = bool(require_signal if gate_needed else False)
                    meta["dir_gate_adaptive_enabled"] = bool(adaptive_enabled)
                    meta["dir_gate_adaptive_strength"] = float(adaptive_strength)
                    meta["dir_gate_adaptive_direction_hit"] = adaptive_hit
                    meta["dir_gate_adaptive_entry_issue_ratio"] = adaptive_entry_issue
                    meta["dir_gate_adaptive_direction_hit_delta"] = adaptive_hit_delta
                    meta["dir_gate_adaptive_entry_issue_delta"] = adaptive_entry_issue_delta
                    meta["dir_gate_ok"] = (None if dir_gate_ok is None else bool(dir_gate_ok))
                    decision["meta"] = meta
        # Fee/EV gate for short-term trading (optional)
        if is_entry:
            try:
                fee_filter_on = str(os.environ.get("FEE_FILTER_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                fee_filter_on = False
            if fee_filter_on:
                try:
                    base_fee = float(os.environ.get("HYBRID_BASE_FEE_RATE", 0.0002) or 0.0002)
                except Exception:
                    base_fee = 0.0002
                try:
                    slippage_bps = float(os.environ.get("HYBRID_SLIPPAGE_BPS", 0.0) or 0.0)
                except Exception:
                    slippage_bps = 0.0
                fee_roundtrip = float(2.0 * (base_fee + slippage_bps * 1e-4))
                try:
                    fee_mult = float(os.environ.get("FEE_FILTER_MULT", 1.0) or 1.0)
                except Exception:
                    fee_mult = 1.0
                try:
                    fee_extra = float(os.environ.get("FEE_FILTER_EXTRA", 0.0) or 0.0)
                except Exception:
                    fee_extra = 0.0
                fee_cost = float(fee_roundtrip) * float(fee_mult) + float(fee_extra)
                try:
                    fee_ev = float(meta.get("event_ev_pct")) if meta.get("event_ev_pct") is not None else None
                except Exception:
                    fee_ev = None
                if fee_ev is not None:
                    fee_ev_eff = fee_ev
                    try:
                        use_lev = str(os.environ.get("FEE_FILTER_USE_LEVERAGE", "1")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        use_lev = True
                    lev_for_fee = None
                    if use_lev:
                        try:
                            lev_for_fee = float(decision.get("leverage") or meta.get("leverage") or meta.get("lev") or self._dyn_leverage.get(sym) or self.leverage)
                        except Exception:
                            lev_for_fee = None
                        try:
                            lev_cap = float(os.environ.get("FEE_FILTER_LEV_CAP", 0.0) or 0.0)
                        except Exception:
                            lev_cap = 0.0
                        if lev_for_fee is not None and lev_cap > 0:
                            lev_for_fee = float(min(lev_for_fee, lev_cap))
                        if lev_for_fee is not None:
                            try:
                                fee_ev_eff = float(fee_ev) * float(max(lev_for_fee, 1e-6))
                            except Exception:
                                fee_ev_eff = fee_ev
                    try:
                        fee_ok = float(fee_ev_eff) >= float(fee_cost)
                    except Exception:
                        fee_ok = None
                # expose fee filter fields for dashboard/tooltips
                try:
                    if decision is not None and isinstance(meta, dict):
                        meta["fee_filter_enabled"] = True
                        meta["fee_filter_cost"] = fee_cost
                        meta["fee_filter_ev"] = fee_ev
                        meta["fee_filter_ev_eff"] = fee_ev_eff if fee_ev is not None else None
                        meta["fee_filter_lev"] = lev_for_fee
                        meta["fee_filter_mult"] = fee_mult
                        meta["fee_filter_base"] = fee_roundtrip
                        decision["meta"] = meta
                except Exception:
                    pass
        # Cost-gate integration (NX/FEE): avoid hard double-gating unless explicitly requested.
        if is_entry:
            try:
                cost_gate_mode = str(os.environ.get("ENTRY_COST_GATE_MODE", "nx_preferred")).strip().lower()
            except Exception:
                cost_gate_mode = "nx_preferred"
            nx_active = net_expectancy_ok is not None
            fee_active = fee_ok is not None
            integration_applied = False
            if cost_gate_mode in ("nx_preferred", "nx_primary", "integrated"):
                if nx_active and fee_active:
                    # NX already subtracts execution cost via net_edge = edge_raw - fee_est.
                    # Keep NX as the single hard gate and demote fee to informational meta.
                    fee_ok = None
                    integration_applied = True
            elif cost_gate_mode in ("fee_preferred", "fee_primary"):
                if nx_active and fee_active:
                    net_expectancy_ok = None
                    integration_applied = True
            elif cost_gate_mode in ("soft_or", "or"):
                if nx_active and fee_active:
                    merged_ok = bool(net_expectancy_ok) or bool(fee_ok)
                    net_expectancy_ok = merged_ok
                    fee_ok = None
                    integration_applied = True
            # modes: both/strict -> keep both as-is.
            if isinstance(meta, dict):
                meta["entry_cost_gate_mode"] = str(cost_gate_mode)
                meta["entry_cost_gate_nx_active"] = bool(nx_active)
                meta["entry_cost_gate_fee_active"] = bool(fee_active)
                meta["entry_cost_gate_integration_applied"] = bool(integration_applied)
                if integration_applied:
                    meta["entry_cost_gate_note"] = (
                        "fee_suppressed_by_nx"
                        if cost_gate_mode in ("nx_preferred", "nx_primary", "integrated")
                        else "nx_suppressed_by_fee"
                        if cost_gate_mode in ("fee_preferred", "fee_primary")
                        else "nx_fee_soft_or"
                    )
                decision["meta"] = meta
        cap_ok = cap_safety_ok
        if cap_positions_ok is False:
            cap_ok = False
        if cap_exposure_ok is False:
            cap_ok = False
        out = {
            "unified": unified_ok,
            "spread": spread_ok,
            "event_cvar": event_cvar_ok,
            "event_exit": event_exit_ok,
            "both_ev_neg": both_ev_neg_ok,
            "gross_ev": gross_ev_ok,
            "net_expectancy": net_expectancy_ok,
            "dir_gate": dir_gate_ok,
            "symbol_quality": symbol_quality_ok,
            "lev_floor_lock": lev_floor_lock_ok,
            "liq": liq_ok,
            "min_notional": min_notional_ok,
            "min_exposure": min_exposure_ok,
            "tick_vol": tick_vol_ok,
            "fee": fee_ok,
            "top_n": None,
            "pre_mc": None,
            "cap": cap_ok,
            "cap_safety": cap_safety_ok,
            "cap_positions": cap_positions_ok,
            "cap_exposure": cap_exposure_ok,
        }
        if not is_entry:
            # Entry-only filters shouldn't show as blocked for non-entry decisions.
            for k in ("unified", "spread", "event_cvar", "event_exit", "both_ev_neg", "gross_ev", "net_expectancy", "dir_gate", "symbol_quality", "lev_floor_lock", "liq", "min_notional", "min_exposure", "tick_vol", "fee", "top_n", "pre_mc", "cap", "cap_safety", "cap_positions", "cap_exposure"):
                out[k] = None
        return out

    def _normalize_filter_overrides(self, overrides) -> set[str]:
        if overrides is None:
            return set()
        if isinstance(overrides, str):
            parts = [p.strip().lower() for p in overrides.replace(";", ",").split(",")]
            return {p for p in parts if p}
        if isinstance(overrides, (list, tuple, set)):
            out = set()
            for item in overrides:
                if item is None:
                    continue
                if isinstance(item, str):
                    val = item.strip().lower()
                else:
                    try:
                        val = str(item).strip().lower()
                    except Exception:
                        val = ""
                if val:
                    out.add(val)
            return {p for p in out if p}
        try:
            val = str(overrides).strip().lower()
            return {val} if val else set()
        except Exception:
            return set()

    def _extract_entry_overrides(self, decision: dict | None) -> set[str]:
        if not decision:
            return set()
        raw = decision.get("entry_override_filters")
        if raw is None:
            meta = decision.get("meta") or {}
            raw = meta.get("entry_override_filters") or meta.get("override_filters") or meta.get("override_filters_entry")
        return self._normalize_filter_overrides(raw)

    def _entry_permit(self, sym: str, decision: dict, ts_ms: int) -> tuple[bool, str]:
        fs = None
        if decision:
            if isinstance(decision.get("filter_states"), dict):
                fs = decision.get("filter_states")
            else:
                meta = decision.get("meta") or {}
                if isinstance(meta.get("filter_states"), dict):
                    fs = meta.get("filter_states")
        if fs is None:
            fs = self._min_filter_states(sym, decision, ts_ms)
        overrides = self._extract_entry_overrides(decision)
        if overrides and decision:
            meta = dict(decision.get("meta") or {})
            meta["entry_override_filters"] = sorted(overrides)
            decision["meta"] = meta
        if fs.get("top_n") is None and "top_n" not in overrides:
            top_n = self._top_n_status(sym)
            if top_n.get("ok") is not None:
                fs = dict(fs)
                fs["top_n"] = top_n.get("ok")
        if fs.get("pre_mc") is None and "pre_mc" not in overrides:
            pre_mc = self._pre_mc_status()
            if pre_mc.get("ok") is not None:
                fs = dict(fs)
                fs["pre_mc"] = pre_mc.get("ok")
        if overrides:
            fs = dict(fs)
            for key in overrides:
                if key in fs and fs.get(key) is False:
                    fs[key] = None
        if decision and fs is not decision.get("filter_states"):
            decision["filter_states"] = fs
            meta = dict(decision.get("meta") or {})
            meta["filter_states"] = fs
            decision["meta"] = meta
        if fs.get("spread") is False and "spread" not in overrides:
            return False, "spread_cap"
        if fs.get("event_cvar") is False and "event_cvar" not in overrides:
            return False, "event_cvar_floor"
        if fs.get("event_exit") is False and "event_exit" not in overrides:
            return False, "event_mc_exit"
        if fs.get("both_ev_neg") is False and "both_ev_neg" not in overrides:
            return False, "both_ev_negative"
        if fs.get("gross_ev") is False and "gross_ev" not in overrides:
            return False, "gross_ev"
        if fs.get("net_expectancy") is False and "net_expectancy" not in overrides:
            return False, "net_expectancy"
        if fs.get("dir_gate") is False and "dir_gate" not in overrides:
            return False, "direction_conf"
        if fs.get("symbol_quality") is False and "symbol_quality" not in overrides:
            return False, "symbol_quality"
        if fs.get("unified") is False and "unified" not in overrides:
            return False, "unified_floor"
        if fs.get("liq") is False and "liq" not in overrides:
            return False, "liquidity"
        if fs.get("min_notional") is False and "min_notional" not in overrides:
            return False, "min_notional"
        if fs.get("min_exposure") is False and "min_exposure" not in overrides:
            return False, "min_exposure"
        if fs.get("fee") is False and "fee" not in overrides:
            return False, "fee"
        if fs.get("top_n") is False and "top_n" not in overrides:
            return False, "top_n"
        if fs.get("pre_mc") is False and "pre_mc" not in overrides:
            return False, "pre_mc"
        if fs.get("cap") is False and "cap" not in overrides:
            return False, "cap"
        return True, ""

    def _calc_position_size(self, decision: dict, price: float, leverage: float, size_frac_override: float | None = None, symbol: str | None = None, use_cycle_reserve: bool = False, ignore_caps: bool = False) -> tuple[float, float, float]:
        meta = (decision or {}).get("meta", {}) or {}
        size_frac = size_frac_override if size_frac_override is not None else decision.get("size_frac") or meta.get("size_fraction") or self.default_size_frac
        base_balance = self._cycle_available_balance() if use_cycle_reserve else self._sizing_balance()

        # ---- Kelly 배분 적용 (psi/kelly 우선, 종목당 cap 미적용) ----
        if USE_KELLY_ALLOCATION and symbol and symbol in self._kelly_allocations:
            try:
                total_cap = float(os.environ.get("KELLY_TOTAL_EXPOSURE", self.max_notional_frac or 1.0))
            except Exception:
                total_cap = float(self.max_notional_frac or 1.0)
            lev_safe = float(leverage) if leverage and float(leverage) > 0 else 1.0
            kelly_frac = float(self._kelly_allocations[symbol])
            # Total notional cap (e.g., 5.0 = 500%) distributed purely by Kelly weights.
            size_frac = max(0.0, kelly_frac * total_cap / max(lev_safe, 1e-6))
        else:
            cap_frac = meta.get("regime_cap_frac")
            if cap_frac is not None:
                try:
                    size_frac = min(size_frac, float(cap_frac))
                except Exception:
                    pass
        
        # 상한 제거: 신호가 강하면 엔진이 제시한 비중을 그대로 사용
        size_frac = float(max(0.0, size_frac))
        # Stage-4: dynamic exposure scaling (delever in toxic/low-confidence states).
        try:
            dyn_exp_enabled = str(os.environ.get("EXPOSURE_DYNAMIC_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            dyn_exp_enabled = True
        if dyn_exp_enabled:
            try:
                conf_soft = float(os.environ.get("EXPOSURE_CONF_SOFT", 0.56) or 0.56)
            except Exception:
                conf_soft = 0.56
            try:
                vpin_soft = float(os.environ.get("EXPOSURE_VPIN_SOFT", 0.70) or 0.70)
            except Exception:
                vpin_soft = 0.70
            try:
                vpin_hard = float(os.environ.get("EXPOSURE_VPIN_HARD", 0.90) or 0.90)
            except Exception:
                vpin_hard = 0.90
            try:
                gain_tox = float(os.environ.get("EXPOSURE_DELEV_GAIN_TOX", 0.55) or 0.55)
            except Exception:
                gain_tox = 0.55
            try:
                gain_conf = float(os.environ.get("EXPOSURE_DELEV_GAIN_CONF", 0.45) or 0.45)
            except Exception:
                gain_conf = 0.45
            try:
                hard_scale = float(os.environ.get("EXPOSURE_DELEV_HARD_SCALE", 0.70) or 0.70)
            except Exception:
                hard_scale = 0.70
            try:
                min_scale = float(os.environ.get("EXPOSURE_DYNAMIC_MIN_SCALE", 0.30) or 0.30)
            except Exception:
                min_scale = 0.30
            min_scale = float(max(0.05, min(1.0, min_scale)))
            try:
                confidence = float(decision.get("confidence") if decision and decision.get("confidence") is not None else meta.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = float(max(0.0, min(1.0, confidence)))
            vpin_val = self._safe_float(meta.get("vpin"), None)
            if vpin_val is None:
                vpin_val = self._safe_float(meta.get("event_exit_dynamic_vpin"), None)
            if vpin_val is None:
                vpin_val = self._safe_float(meta.get("alpha_vpin"), None)
            if vpin_val is not None:
                vpin_val = float(max(0.0, min(1.0, vpin_val)))
            tox = 0.0
            if vpin_val is not None and float(vpin_val) > float(vpin_soft):
                tox = float(max(0.0, float(vpin_val) - float(vpin_soft)) / max(1e-6, 1.0 - float(vpin_soft)))
            conf_def = float(max(0.0, float(conf_soft) - float(confidence)) / max(1e-6, float(conf_soft)))
            exp_scale = float((1.0 - gain_tox * tox) * (1.0 - gain_conf * conf_def))
            if (vpin_val is not None and float(vpin_val) >= float(vpin_hard)) and confidence < conf_soft:
                exp_scale *= float(max(0.10, min(1.0, hard_scale)))
            exp_scale = float(max(min_scale, min(1.0, exp_scale)))
            size_frac = float(size_frac * exp_scale)
            try:
                meta["exposure_dynamic_enabled"] = bool(dyn_exp_enabled)
                meta["exposure_dynamic_scale"] = float(exp_scale)
                meta["exposure_dynamic_vpin"] = vpin_val
                meta["exposure_dynamic_confidence"] = float(confidence)
                meta["exposure_dynamic_tox"] = float(tox)
                meta["exposure_dynamic_conf_deficit"] = float(conf_def)
                if isinstance(decision, dict):
                    decision["meta"] = meta
            except Exception:
                pass
        # Cap-aware sizing for new entries: scale to remaining exposure capacity
        if self.exposure_cap_enabled and use_cycle_reserve and (not ignore_caps):
            try:
                total_cap = float(os.environ.get("KELLY_TOTAL_EXPOSURE", self.max_notional_frac or 1.0))
            except Exception:
                total_cap = float(self.max_notional_frac or 1.0)
            try:
                eff_cap = float(min(total_cap, float(self.max_notional_frac or total_cap)))
            except Exception:
                eff_cap = float(total_cap)
            try:
                cap_balance = float(self._exposure_cap_balance())
            except Exception:
                cap_balance = 0.0
            if cap_balance > 0 and eff_cap > 0:
                try:
                    open_notional = float(self._total_open_notional() or 0.0)
                except Exception:
                    open_notional = 0.0
                remaining_frac = max(0.0, eff_cap - (open_notional / max(cap_balance, 1e-9)))
                cap_size_frac = remaining_frac / max(float(leverage or 1.0), 1e-6)
                if cap_size_frac < size_frac:
                    size_frac = max(0.0, cap_size_frac)
        # Order sizing uses available balance (live_free_balance) when live, with a safety haircut.
        try:
            sizing_pad = float(os.environ.get("LIVE_ORDER_SAFETY_PAD", 0.98) or 0.98)
        except Exception:
            sizing_pad = 0.98
        sizing_pad = min(max(sizing_pad, 0.1), 1.0)
        notional = float(max(0.0, base_balance * sizing_pad * size_frac * leverage))
        qty = float(notional / price) if price and notional > 0 else 0.0
        return size_frac, notional, qty

    #def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
    def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
        def _f(x, default=0.0):
            try:
                if x is None:
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        regime = ctx.get("regime") or "chop"
        sym = ctx.get("symbol")
        sigma_raw = _f(ctx.get("sigma"), 0.0)
        try:
            sigma_cap_for_lev = float(os.environ.get("LEVERAGE_SIGMA_CAP_FOR_RISK", 0.20) or 0.20)
        except Exception:
            sigma_cap_for_lev = 0.20
        sigma_cap_for_lev = float(max(0.01, min(5.0, sigma_cap_for_lev)))
        sigma = float(min(max(0.0, float(sigma_raw)), sigma_cap_for_lev))
        meta = decision.get("meta") or {}
        ev = _f(decision.get("ev"), 0.0)
        cvar = _f(meta.get("cvar05", decision.get("cvar")), 0.0)
        event_p_sl = _f(meta.get("event_p_sl"), 0.0)
        spread_pct = _f(meta.get("spread_pct", ctx.get("spread_pct")), 0.0002)
        execution_cost = _f(meta.get("execution_cost", meta.get("fee_rt")), 0.0)
        slippage_pct = _f(meta.get("slippage_pct", 0.0), 0.0)
        event_cvar_pct = _f(meta.get("event_cvar_pct"), 0.0)
        confidence = float(max(0.0, min(1.0, _f(decision.get("confidence"), 0.0))))
        unified_score = decision.get("unified_score")
        if unified_score is None:
            unified_score = meta.get("unified_score")
        unified_score = _f(unified_score, 0.0)
        action = str(decision.get("action") or "").upper()
        side_sign = 1.0 if action == "LONG" else (-1.0 if action == "SHORT" else 0.0)
        hmm_sign = _f(ctx.get("hmm_regime_sign", meta.get("hmm_regime_sign")), 0.0)
        hmm_conf = float(max(0.0, min(1.0, _f(ctx.get("hmm_conf", meta.get("hmm_conf")), 0.0))))
        vpin_val = self._safe_float(meta.get("vpin"), None)
        if vpin_val is None:
            vpin_val = self._safe_float(ctx.get("vpin"), None)
        if vpin_val is None:
            vpin_val = self._safe_float(meta.get("event_exit_dynamic_vpin"), None)
        if vpin_val is not None:
            vpin_val = float(max(0.0, min(1.0, vpin_val)))
        hurst_val = self._safe_float(meta.get("hurst"), None)
        if hurst_val is None:
            hurst_val = self._safe_float(ctx.get("hurst"), None)
        if hurst_val is not None:
            hurst_val = float(max(0.0, min(1.0, hurst_val)))
        mu_align = self._safe_float(meta.get("event_exit_dynamic_mu_alignment"), None)
        if mu_align is None and side_sign != 0.0:
            mu_alpha = self._safe_float(meta.get("mu_adjusted"), None)
            if mu_alpha is None:
                mu_alpha = self._safe_float(meta.get("mu_alpha"), None)
            if mu_alpha is None:
                mu_alpha = self._safe_float(ctx.get("mu_alpha"), None)
            if mu_alpha is None:
                mu_alpha = self._safe_float(ctx.get("mu_base"), 0.0) or 0.0
            mu_align = float(side_sign * float(mu_alpha or 0.0))
        mu_align = float(mu_align or 0.0)
        dir_conf = self._safe_float(meta.get("mu_dir_conf"), None)
        if dir_conf is None:
            dir_conf = self._safe_float(ctx.get("mu_dir_conf"), None)
        dir_edge = self._safe_float(meta.get("mu_dir_edge"), None)
        if dir_edge is None:
            dir_edge = self._safe_float(ctx.get("mu_dir_edge"), None)
        dir_prob_long = self._safe_float(meta.get("mu_dir_prob_long"), None)
        if dir_prob_long is None:
            dir_prob_long = self._safe_float(ctx.get("mu_dir_prob_long"), None)
        ev_gap = self._safe_float(meta.get("policy_ev_gap"), None)
        if ev_gap is None:
            ev_gap = self._safe_float(meta.get("ev_gap"), None)
        if ev_gap is None:
            ev_gap = self._safe_float(ctx.get("ev_gap"), 0.0)
        tick_trend = self._safe_float(meta.get("tick_trend"), None)
        if tick_trend is None:
            tick_trend = self._safe_float(ctx.get("tick_trend"), 0.0)
        tick_breakout_active = bool(meta.get("tick_breakout_active", ctx.get("tick_breakout_active")))
        tick_breakout_dir = self._safe_float(meta.get("tick_breakout_dir"), None)
        if tick_breakout_dir is None:
            tick_breakout_dir = self._safe_float(ctx.get("tick_breakout_dir"), 0.0)
        tick_breakout_score = self._safe_float(meta.get("tick_breakout_score"), None)
        if tick_breakout_score is None:
            tick_breakout_score = self._safe_float(ctx.get("tick_breakout_score"), 0.0)

        def _clip01(x: float) -> float:
            return float(max(0.0, min(1.0, float(x))))

        try:
            edge_ref = float(os.environ.get("LEVERAGE_ENTRY_QUALITY_EDGE_REF", 0.16) or 0.16)
        except Exception:
            edge_ref = 0.16
        edge_ref = max(edge_ref, 1e-6)
        try:
            gap_ref = float(os.environ.get("LEVERAGE_ENTRY_QUALITY_EV_GAP_REF", 0.0030) or 0.0030)
        except Exception:
            gap_ref = 0.0030
        gap_ref = max(gap_ref, 1e-6)
        try:
            mu_ref = float(os.environ.get("LEVERAGE_ENTRY_QUALITY_MU_ALIGN_REF", 0.0015) or 0.0015)
        except Exception:
            mu_ref = 0.0015
        mu_ref = max(mu_ref, 1e-6)

        conf_n = _clip01(confidence)
        dir_conf_n = _clip01(dir_conf if dir_conf is not None else confidence)
        edge_n = _clip01(abs(float(dir_edge or 0.0)) / edge_ref)
        gap_n = _clip01(abs(float(ev_gap or 0.0)) / gap_ref)
        mu_n = _clip01(max(0.0, math.tanh(float(mu_align) / mu_ref)))
        entry_quality_score = float(
            0.32 * conf_n
            + 0.28 * dir_conf_n
            + 0.18 * edge_n
            + 0.12 * gap_n
            + 0.10 * mu_n
        )
        entry_quality_score = _clip01(entry_quality_score)

        try:
            breakout_ref = float(os.environ.get("LEVERAGE_ONEWAY_BREAKOUT_REF", 0.0012) or 0.0012)
        except Exception:
            breakout_ref = 0.0012
        breakout_ref = max(breakout_ref, 1e-8)
        try:
            trend_ref = float(os.environ.get("LEVERAGE_ONEWAY_TREND_REF", 1.5) or 1.5)
        except Exception:
            trend_ref = 1.5
        trend_ref = max(trend_ref, 1e-6)
        breakout_align = 0.0
        if side_sign != 0.0 and tick_breakout_active:
            breakout_sign = float(side_sign) * float(tick_breakout_dir or 0.0)
            breakout_mag = min(2.0, abs(float(tick_breakout_score or 0.0)) / breakout_ref)
            breakout_align = breakout_mag if breakout_sign > 0.0 else (-0.7 * breakout_mag)
        breakout_n = _clip01(0.5 + 0.5 * breakout_align)
        trend_n = _clip01(0.5 + 0.5 * math.tanh((float(side_sign) * float(tick_trend or 0.0)) / trend_ref))
        hurst_trend_n = 0.5 if hurst_val is None else _clip01((float(hurst_val) - 0.45) / 0.20)
        one_way_raw = float(0.45 * breakout_n + 0.35 * trend_n + 0.20 * hurst_trend_n)
        tox_scale = 1.0
        if vpin_val is not None:
            tox_scale = float(max(0.40, 1.0 - 0.45 * _clip01((float(vpin_val) - 0.55) / 0.45)))
        one_way_move_score = _clip01(one_way_raw * tox_scale)
        try:
            signal_quality_w = float(os.environ.get("LEVERAGE_SIGNAL_QUALITY_WEIGHT", 0.62) or 0.62)
        except Exception:
            signal_quality_w = 0.62
        signal_quality_w = float(max(0.0, min(1.0, signal_quality_w)))
        leverage_signal_score = _clip01(
            signal_quality_w * float(entry_quality_score) + (1.0 - signal_quality_w) * float(one_way_move_score)
        )

        # risk = max(|CVaR|, |event_cvar_pct|) + 0.7*spread + 0.5*slippage + 0.5*sigma (+ p_sl 가중)
        risk_score = max(abs(cvar), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma + 0.2 * event_p_sl
        if risk_score <= 1e-6:
            risk_score = 1e-6

        try:
            lev_regime_bull = float(os.environ.get("LEVERAGE_REGIME_MAX_BULL", self.max_leverage) or self.max_leverage)
        except Exception:
            lev_regime_bull = float(self.max_leverage)
        try:
            lev_regime_bear = float(os.environ.get("LEVERAGE_REGIME_MAX_BEAR", self.max_leverage) or self.max_leverage)
        except Exception:
            lev_regime_bear = float(self.max_leverage)
        try:
            lev_regime_chop = float(os.environ.get("LEVERAGE_REGIME_MAX_CHOP", min(self.max_leverage, 30.0)) or min(self.max_leverage, 30.0))
        except Exception:
            lev_regime_chop = float(min(self.max_leverage, 30.0))
        try:
            lev_regime_volatile = float(os.environ.get("LEVERAGE_REGIME_MAX_VOLATILE", min(self.max_leverage, 20.0)) or min(self.max_leverage, 20.0))
        except Exception:
            lev_regime_volatile = float(min(self.max_leverage, 20.0))
        lev_max_map = {
            "bull": min(float(self.max_leverage), max(1.0, lev_regime_bull)),
            "bear": min(float(self.max_leverage), max(1.0, lev_regime_bear)),
            "chop": min(float(self.max_leverage), max(1.0, lev_regime_chop)),
            "volatile": min(float(self.max_leverage), max(1.0, lev_regime_volatile)),
        }
        lev_max = float(lev_max_map.get(regime, self.max_leverage))

        # EV/risk 기반 기본 레버리지 + 신뢰도/점수/HMM 정렬 보정
        lev_raw = (max(ev - execution_cost, 0.0) / risk_score) * K_LEV
        try:
            conf_gain = float(getattr(mc_config, "leverage_conf_gain", 0.8))
        except Exception:
            conf_gain = 0.8
        lev_raw *= max(0.55, (1.0 + conf_gain * (confidence - 0.5)))

        try:
            score_ref = float(getattr(mc_config, "leverage_score_ref", 0.003) or 0.003)
        except Exception:
            score_ref = 0.003
        score_ref = max(score_ref, 1e-6)
        score_term = math.tanh(float(unified_score) / score_ref)
        if score_term >= 0:
            lev_raw *= (1.0 + 0.7 * score_term)
        else:
            lev_raw *= max(0.60, 1.0 + 0.4 * score_term)

        try:
            hmm_gain = float(getattr(mc_config, "leverage_hmm_gain", 0.5))
        except Exception:
            hmm_gain = 0.5
        if side_sign != 0.0 and hmm_conf > 0.0:
            hmm_align = side_sign * (1.0 if hmm_sign > 0 else (-1.0 if hmm_sign < 0 else 0.0))
            if hmm_align > 0:
                lev_raw *= (1.0 + hmm_gain * hmm_conf)
            elif hmm_align < 0:
                lev_raw *= max(0.50, 1.0 - 0.8 * hmm_gain * hmm_conf)

        # tail risk guard
        lev_raw *= max(0.40, 1.0 - 0.65 * max(event_p_sl, 0.0))

        # Stage-4: confidence/toxicity/volatility aware deleveraging + liquidation-distance floor.
        try:
            dyn_lev_enabled = str(os.environ.get("LEVERAGE_DYNAMIC_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            dyn_lev_enabled = True
        try:
            lev_target_min = float(os.environ.get("LEVERAGE_TARGET_MIN", max(1.0, LEVERAGE_MIN)) or max(1.0, LEVERAGE_MIN))
        except Exception:
            lev_target_min = max(1.0, float(LEVERAGE_MIN))
        try:
            lev_target_max = float(os.environ.get("LEVERAGE_TARGET_MAX", min(50.0, float(self.max_leverage))) or min(50.0, float(self.max_leverage)))
        except Exception:
            lev_target_max = min(50.0, float(self.max_leverage))
        lev_target_min = max(1.0, float(lev_target_min))
        lev_target_max = max(lev_target_min, min(float(lev_max), float(lev_target_max)))
        lev_target_max_base = float(lev_target_max)
        lev_signal_target = float(lev_target_min)
        lev_raw_ev = float(lev_raw)
        lev_signal_blend = 0.0
        if action in ("LONG", "SHORT"):
            try:
                lev_signal_pow = float(os.environ.get("LEVERAGE_SIGNAL_SCORE_POW", 1.20) or 1.20)
            except Exception:
                lev_signal_pow = 1.20
            lev_signal_pow = float(max(0.2, min(3.0, lev_signal_pow)))
            lev_signal_target = float(
                lev_target_min
                + (lev_target_max - lev_target_min) * (float(leverage_signal_score) ** lev_signal_pow)
            )
            try:
                lev_signal_blend = float(os.environ.get("LEVERAGE_SIGNAL_BLEND", 0.55) or 0.55)
            except Exception:
                lev_signal_blend = 0.55
            lev_signal_blend = float(max(0.0, min(1.0, lev_signal_blend)))
            lev_raw = float((1.0 - lev_signal_blend) * float(lev_raw_ev) + lev_signal_blend * float(lev_signal_target))
        aggressive_open = False
        if action in ("LONG", "SHORT"):
            try:
                aggressive_on = str(os.environ.get("LEVERAGE_AGGRESSIVE_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                aggressive_on = False
            if aggressive_on:
                try:
                    aggressive_cap = float(os.environ.get("LEVERAGE_AGGRESSIVE_MAX", self.max_leverage) or self.max_leverage)
                except Exception:
                    aggressive_cap = float(self.max_leverage)
                aggressive_cap = max(lev_target_max_base, min(float(self.max_leverage), float(aggressive_cap)))
                try:
                    aggressive_min_conf = float(os.environ.get("LEVERAGE_AGGRESSIVE_MIN_CONF", 0.72) or 0.72)
                except Exception:
                    aggressive_min_conf = 0.72
                try:
                    aggressive_min_score = float(os.environ.get("LEVERAGE_AGGRESSIVE_MIN_SCORE", 0.0012) or 0.0012)
                except Exception:
                    aggressive_min_score = 0.0012
                try:
                    aggressive_min_mu_align = float(os.environ.get("LEVERAGE_AGGRESSIVE_MIN_MU_ALIGN", 0.0) or 0.0)
                except Exception:
                    aggressive_min_mu_align = 0.0
                try:
                    aggressive_hurst_min = float(os.environ.get("LEVERAGE_AGGRESSIVE_HURST_MIN", 0.55) or 0.55)
                except Exception:
                    aggressive_hurst_min = 0.55
                try:
                    aggressive_vpin_max = float(os.environ.get("LEVERAGE_AGGRESSIVE_MAX_VPIN", 0.55) or 0.55)
                except Exception:
                    aggressive_vpin_max = 0.55
                try:
                    aggressive_event_psl_max = float(os.environ.get("LEVERAGE_AGGRESSIVE_MAX_EVENT_PSL", 0.30) or 0.30)
                except Exception:
                    aggressive_event_psl_max = 0.30
                vpin_ok = True if vpin_val is None else float(vpin_val) <= float(aggressive_vpin_max)
                hurst_ok = True if hurst_val is None else float(hurst_val) >= float(aggressive_hurst_min)
                if (
                    confidence >= float(aggressive_min_conf)
                    and float(unified_score) >= float(aggressive_min_score)
                    and float(mu_align) >= float(aggressive_min_mu_align)
                    and vpin_ok
                    and hurst_ok
                    and float(event_p_sl) <= float(aggressive_event_psl_max)
                ):
                    aggressive_open = True
                    lev_target_max = float(max(lev_target_max_base, min(float(lev_max), float(aggressive_cap))))
        tox = 0.0
        conf_def = 0.0
        sigma_stress = 0.0
        lev_scale = 1.0
        if dyn_lev_enabled:
            try:
                vpin_soft = float(os.environ.get("LEVERAGE_VPIN_SOFT", 0.65) or 0.65)
            except Exception:
                vpin_soft = 0.65
            try:
                vpin_hard = float(os.environ.get("LEVERAGE_VPIN_HARD", 0.90) or 0.90)
            except Exception:
                vpin_hard = 0.90
            try:
                conf_soft = float(os.environ.get("LEVERAGE_CONF_SOFT", 0.56) or 0.56)
            except Exception:
                conf_soft = 0.56
            try:
                sigma_soft = float(os.environ.get("LEVERAGE_SIGMA_SOFT", 0.035) or 0.035)
            except Exception:
                sigma_soft = 0.035
            try:
                sigma_hard = float(os.environ.get("LEVERAGE_SIGMA_HARD", 0.080) or 0.080)
            except Exception:
                sigma_hard = 0.080
            try:
                delev_gain_tox = float(os.environ.get("LEVERAGE_DELEV_GAIN_TOX", 0.65) or 0.65)
            except Exception:
                delev_gain_tox = 0.65
            try:
                delev_gain_conf = float(os.environ.get("LEVERAGE_DELEV_GAIN_CONF", 0.50) or 0.50)
            except Exception:
                delev_gain_conf = 0.50
            try:
                delev_gain_sigma = float(os.environ.get("LEVERAGE_DELEV_GAIN_SIGMA", 0.55) or 0.55)
            except Exception:
                delev_gain_sigma = 0.55
            try:
                delev_hard_scale = float(os.environ.get("LEVERAGE_DELEV_HARD_SCALE", 0.55) or 0.55)
            except Exception:
                delev_hard_scale = 0.55
            try:
                min_scale = float(os.environ.get("LEVERAGE_DYNAMIC_MIN_SCALE", 0.20) or 0.20)
            except Exception:
                min_scale = 0.20
            min_scale = float(max(0.05, min(1.0, min_scale)))
            try:
                vpin_cap_for_delev = float(os.environ.get("LEVERAGE_VPIN_CAP_FOR_DELEV", 0.92) or 0.92)
            except Exception:
                vpin_cap_for_delev = 0.92
            vpin_cap_for_delev = float(max(0.50, min(1.0, vpin_cap_for_delev)))
            vpin_for_lev = vpin_val
            if vpin_for_lev is not None:
                vpin_for_lev = float(min(float(vpin_for_lev), float(vpin_cap_for_delev)))
            if vpin_for_lev is not None and vpin_for_lev > vpin_soft:
                tox = float(max(0.0, vpin_for_lev - vpin_soft) / max(1e-6, 1.0 - vpin_soft))
            conf_def = float(max(0.0, conf_soft - confidence) / max(1e-6, conf_soft))
            if sigma > sigma_soft:
                sigma_stress = float(max(0.0, sigma - sigma_soft) / max(1e-6, sigma_hard - sigma_soft))
            try:
                sigma_stress_clip = float(os.environ.get("LEVERAGE_SIGMA_STRESS_CLIP", 1.0) or 1.0)
            except Exception:
                sigma_stress_clip = 1.0
            sigma_stress_clip = float(max(0.10, min(5.0, sigma_stress_clip)))
            sigma_stress = float(max(0.0, min(sigma_stress, sigma_stress_clip)))
            lev_scale = float((1.0 - delev_gain_tox * tox) * (1.0 - delev_gain_conf * conf_def) * (1.0 - delev_gain_sigma * sigma_stress))
            if (vpin_for_lev is not None and vpin_for_lev >= vpin_hard) and (confidence < conf_soft):
                lev_scale *= float(max(0.10, min(1.0, delev_hard_scale)))
            # Trend+alignment boost is only allowed in non-toxic states.
            try:
                h_high = float(getattr(mc_config, "hurst_high", 0.55))
            except Exception:
                h_high = 0.55
            try:
                trend_boost = float(os.environ.get("LEVERAGE_TREND_ALIGN_BOOST", 0.10) or 0.10)
            except Exception:
                trend_boost = 0.10
            if (hurst_val is not None and hurst_val >= h_high) and (mu_align > 0.0) and (confidence >= conf_soft) and ((vpin_for_lev is None) or (vpin_for_lev < vpin_soft)):
                lev_scale *= float(1.0 + max(0.0, trend_boost))
            lev_scale = float(max(min_scale, min(1.25, lev_scale)))
            lev_raw = float(lev_raw * lev_scale)

        # If recent entries are repeatedly rejected by available-balance (110007), temporarily de-leverage.
        reject_scale = 1.0
        reject_count = 0
        if sym:
            try:
                reject_state = self._balance_reject_110007_by_sym.get(sym) or {}
                reject_ts = int(reject_state.get("ts", 0) or 0)
                reject_count = int(reject_state.get("count", 0) or 0)
            except Exception:
                reject_ts = 0
                reject_count = 0
            try:
                reject_cooldown_sec = float(os.environ.get("LEVERAGE_BALANCE_REJECT_COOLDOWN_SEC", 300.0) or 300.0)
            except Exception:
                reject_cooldown_sec = 300.0
            try:
                reject_step_scale = float(os.environ.get("LEVERAGE_BALANCE_REJECT_SCALE", 0.82) or 0.82)
            except Exception:
                reject_step_scale = 0.82
            try:
                reject_min_count = int(os.environ.get("LEVERAGE_BALANCE_REJECT_MIN_COUNT", 2) or 2)
            except Exception:
                reject_min_count = 2
            try:
                reject_min_scale = float(os.environ.get("LEVERAGE_BALANCE_REJECT_MIN_SCALE", 0.45) or 0.45)
            except Exception:
                reject_min_scale = 0.45
            try:
                reject_relief_signal = float(os.environ.get("LEVERAGE_BALANCE_REJECT_RELIEF_SIGNAL", 0.60) or 0.60)
            except Exception:
                reject_relief_signal = 0.60
            try:
                reject_relief_conf = float(os.environ.get("LEVERAGE_BALANCE_REJECT_RELIEF_CONF", 0.66) or 0.66)
            except Exception:
                reject_relief_conf = 0.66
            try:
                reject_relief_pow = float(os.environ.get("LEVERAGE_BALANCE_REJECT_RELIEF_POW", 0.55) or 0.55)
            except Exception:
                reject_relief_pow = 0.55
            reject_min_count = int(max(1, min(10, reject_min_count)))
            reject_min_scale = float(max(0.10, min(1.0, reject_min_scale)))
            reject_relief_signal = float(max(0.0, min(1.0, reject_relief_signal)))
            reject_relief_conf = float(max(0.0, min(1.0, reject_relief_conf)))
            reject_relief_pow = float(max(0.10, min(1.0, reject_relief_pow)))
            if reject_ts > 0 and reject_count > 0:
                age_ms = max(0, now_ms() - reject_ts)
                if age_ms <= int(max(1.0, reject_cooldown_sec) * 1000.0):
                    if reject_count >= reject_min_count:
                        reject_step_scale = min(max(reject_step_scale, 0.10), 1.0)
                        reject_scale = max(reject_min_scale, float(reject_step_scale) ** float(min(6, reject_count)))
                        # High-quality signals get softer rejection de-leverage to avoid floor lock.
                        if float(leverage_signal_score) >= reject_relief_signal and float(confidence) >= reject_relief_conf:
                            reject_scale = max(reject_scale, float(reject_scale) ** float(reject_relief_pow))
                    lev_raw = float(lev_raw * reject_scale)
                else:
                    try:
                        self._balance_reject_110007_by_sym.pop(sym, None)
                    except Exception:
                        pass

        # Enforce minimum liquidation distance using leverage cap.
        try:
            maint_rate = float(os.environ.get("MAINT_MARGIN_RATE", MAINT_MARGIN_RATE) or MAINT_MARGIN_RATE)
        except Exception:
            maint_rate = float(MAINT_MARGIN_RATE)
        try:
            liq_buffer = float(os.environ.get("LIQUIDATION_BUFFER", LIQUIDATION_BUFFER) or LIQUIDATION_BUFFER)
        except Exception:
            liq_buffer = float(LIQUIDATION_BUFFER)
        try:
            min_liq_dist = float(os.environ.get("LEVERAGE_MIN_LIQ_DISTANCE", 0.015) or 0.015)
        except Exception:
            min_liq_dist = 0.015
        min_liq_dist_base = float(min_liq_dist)
        # High-quality / low-toxicity entries may use a tighter liquidation distance floor.
        if action in ("LONG", "SHORT"):
            try:
                hq_conf = float(os.environ.get("LEVERAGE_HIGH_QUALITY_MIN_CONF", 0.74) or 0.74)
            except Exception:
                hq_conf = 0.74
            try:
                hq_signal = float(os.environ.get("LEVERAGE_HIGH_QUALITY_MIN_SIGNAL", 0.82) or 0.82)
            except Exception:
                hq_signal = 0.82
            try:
                hq_entry_q = float(os.environ.get("LEVERAGE_HIGH_QUALITY_MIN_ENTRY_Q", 0.80) or 0.80)
            except Exception:
                hq_entry_q = 0.80
            try:
                hq_vpin_cap = float(os.environ.get("LEVERAGE_HIGH_QUALITY_MAX_VPIN", 0.45) or 0.45)
            except Exception:
                hq_vpin_cap = 0.45
            try:
                min_liq_dist_hq = float(os.environ.get("LEVERAGE_MIN_LIQ_DISTANCE_HIGH_QUALITY", 0.010) or 0.010)
            except Exception:
                min_liq_dist_hq = 0.010
            if (
                float(confidence) >= float(hq_conf)
                and float(leverage_signal_score) >= float(hq_signal)
                and float(entry_quality_score) >= float(hq_entry_q)
                and ((vpin_for_lev is None) or (float(vpin_for_lev) <= float(hq_vpin_cap)))
            ):
                min_liq_dist = min(float(min_liq_dist), float(min_liq_dist_hq))
        min_liq_dist = float(max(0.001, min(0.20, min_liq_dist)))
        liq_cap = float(lev_target_max)
        denom = float(max(1e-6, maint_rate + liq_buffer + min_liq_dist))
        if denom > 0:
            liq_cap = min(float(lev_target_max), float(max(1.0, 1.0 / denom)))

        lev_floor = float(max(1.0, min(lev_target_min, lev_max)))
        lev_cap = float(max(lev_floor, min(float(lev_max), float(lev_target_max), float(liq_cap))))
        lev_quality_cap = float(lev_cap)
        lev_quality_low_conf = False
        lev_quality_high_tox = False
        if action in ("LONG", "SHORT"):
            try:
                low_conf_thr = float(os.environ.get("LEVERAGE_LOW_CONF_THRESHOLD", 0.58) or 0.58)
            except Exception:
                low_conf_thr = 0.58
            try:
                low_signal_thr = float(os.environ.get("LEVERAGE_LOW_SIGNAL_THRESHOLD", 0.48) or 0.48)
            except Exception:
                low_signal_thr = 0.48
            try:
                high_vpin_thr = float(os.environ.get("LEVERAGE_HIGH_VPIN_THRESHOLD", 0.82) or 0.82)
            except Exception:
                high_vpin_thr = 0.82
            try:
                low_conf_cap = float(os.environ.get("LEVERAGE_LOW_CONF_CAP", 3.0) or 3.0)
            except Exception:
                low_conf_cap = 3.0
            try:
                high_vpin_cap = float(os.environ.get("LEVERAGE_HIGH_VPIN_CAP", 2.0) or 2.0)
            except Exception:
                high_vpin_cap = 2.0
            lev_quality_low_conf = bool(
                float(confidence) < float(low_conf_thr)
                or float(leverage_signal_score) < float(low_signal_thr)
            )
            lev_quality_high_tox = bool(
                (vpin_for_lev is not None)
                and float(vpin_for_lev) >= float(high_vpin_thr)
            )
            if lev_quality_low_conf:
                lev_quality_cap = min(float(lev_quality_cap), max(1.0, float(low_conf_cap)))
            if lev_quality_high_tox:
                lev_quality_cap = min(float(lev_quality_cap), max(1.0, float(high_vpin_cap)))
        if lev_quality_cap < lev_floor:
            lev_floor = float(max(1.0, lev_quality_cap))
        lev_cap = float(max(lev_floor, min(lev_cap, lev_quality_cap)))
        lev = float(max(lev_floor, min(lev_cap, lev_raw)))
        try:
            meta["lev_dynamic_enabled"] = bool(dyn_lev_enabled)
            meta["lev_target_min"] = float(lev_floor)
            meta["lev_target_max"] = float(lev_target_max)
            meta["lev_target_max_base"] = float(lev_target_max_base)
            meta["lev_aggressive_open"] = bool(aggressive_open)
            meta["lev_liq_cap"] = float(liq_cap)
            meta["lev_raw_before_caps"] = float(lev_raw)
            meta["lev_raw_ev_component"] = float(lev_raw_ev)
            meta["lev_signal_target"] = float(lev_signal_target)
            meta["lev_signal_blend"] = float(lev_signal_blend)
            meta["entry_quality_score"] = float(entry_quality_score)
            meta["one_way_move_score"] = float(one_way_move_score)
            meta["leverage_signal_score"] = float(leverage_signal_score)
            meta["lev_dynamic_scale"] = float(lev_scale)
            meta["lev_dynamic_tox"] = float(tox)
            meta["lev_dynamic_conf_deficit"] = float(conf_def)
            meta["lev_dynamic_sigma_stress"] = float(sigma_stress)
            meta["lev_balance_reject_scale"] = float(reject_scale)
            meta["lev_balance_reject_count"] = int(reject_count)
            meta["lev_balance_reject_min_count"] = int(reject_min_count if "reject_min_count" in locals() else 0)
            meta["lev_balance_reject_min_scale"] = float(reject_min_scale if "reject_min_scale" in locals() else 0.0)
            meta["lev_balance_reject_relief_signal"] = float(reject_relief_signal if "reject_relief_signal" in locals() else 0.0)
            meta["lev_balance_reject_relief_conf"] = float(reject_relief_conf if "reject_relief_conf" in locals() else 0.0)
            meta["lev_balance_reject_relief_pow"] = float(reject_relief_pow if "reject_relief_pow" in locals() else 0.0)
            meta["lev_dynamic_vpin"] = vpin_val
            meta["lev_dynamic_vpin_for_lev"] = vpin_for_lev if "vpin_for_lev" in locals() else vpin_val
            meta["lev_dynamic_hurst"] = hurst_val
            meta["lev_dynamic_mu_align"] = float(mu_align)
            meta["lev_sigma_raw"] = float(sigma_raw)
            meta["lev_sigma_used"] = float(sigma)
            meta["lev_sigma_stress_clip"] = float(sigma_stress_clip if "sigma_stress_clip" in locals() else 1.0)
            meta["lev_min_liq_dist_base"] = float(min_liq_dist_base)
            meta["lev_min_liq_dist_used"] = float(min_liq_dist)
            meta["lev_quality_cap"] = float(lev_quality_cap)
            meta["lev_quality_low_conf"] = bool(lev_quality_low_conf)
            meta["lev_quality_high_tox"] = bool(lev_quality_high_tox)
            decision["meta"] = meta
        except Exception:
            pass
        if sym:
            floor_sticky = int(self._lev_floor_sticky_counts.get(sym, 0) or 0)
            if action in ("LONG", "SHORT"):
                if float(lev) <= float(lev_floor) + 1e-9:
                    floor_sticky = min(10_000, floor_sticky + 1)
                else:
                    floor_sticky = max(0, floor_sticky - 1)
            else:
                floor_sticky = 0
            self._lev_floor_sticky_counts[sym] = int(floor_sticky)
            try:
                meta = decision.get("meta") or {}
                meta["lev_floor_sticky_count"] = int(floor_sticky)
                decision["meta"] = meta
            except Exception:
                pass
            try:
                lev_diag_on = str(os.environ.get("LEVERAGE_DIAG_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                lev_diag_on = True
            if lev_diag_on and action in ("LONG", "SHORT"):
                try:
                    lev_diag_interval_sec = float(os.environ.get("LEVERAGE_DIAG_INTERVAL_SEC", 120.0) or 120.0)
                except Exception:
                    lev_diag_interval_sec = 120.0
                try:
                    lev_diag_floor_sticky = int(os.environ.get("LEVERAGE_DIAG_FLOOR_STICKY", 3) or 3)
                except Exception:
                    lev_diag_floor_sticky = 3
                now_diag_ms = now_ms()
                prev_diag_ms = int(self._lev_diag_last_ms_by_sym.get(sym, 0) or 0)
                if now_diag_ms - prev_diag_ms >= int(max(5.0, lev_diag_interval_sec) * 1000.0):
                    if floor_sticky >= max(1, lev_diag_floor_sticky) or float(lev) <= float(lev_floor) + 1e-9:
                        self._log(
                            f"[LEV_DIAG] {sym} lev={float(lev):.2f}x floor={float(lev_floor):.2f} cap={float(lev_cap):.2f} "
                            f"raw={float(lev_raw):.3f} conf={float(confidence):.3f} score={float(unified_score):.6f} "
                            f"entry_q={float(entry_quality_score):.3f} oneway={float(one_way_move_score):.3f} "
                            f"lev_sig={float(leverage_signal_score):.3f} "
                            f"vpin={float(vpin_val) if vpin_val is not None else -1.0:.3f} "
                            f"hurst={float(hurst_val) if hurst_val is not None else -1.0:.3f} "
                            f"mu_align={float(mu_align):.6f} reject_scale={float(reject_scale):.3f} floor_sticky={int(floor_sticky)}"
                        )
                    self._lev_diag_last_ms_by_sym[sym] = int(now_diag_ms)
            self._dyn_leverage[sym] = lev
        return lev

    def _direction_bias(self, closes) -> int:
        if not closes:
            return 1
        price_now = closes[-1]
        fast_period = min(10, len(closes))
        slow_period = min(40, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        bias = 1 if fast >= slow else -1
        # 레짐 힌트: 상승장에서는 롱 우선, 하락장에서는 숏 우선 (더 민감하게)
        if len(closes) >= 20:
            slope_short = closes[-1] - closes[-20]
            pct_short = slope_short / max(price_now, 1e-6)
            if pct_short > 0.002:
                bias = 1
            elif pct_short < -0.002:
                bias = -1
        if len(closes) >= 60:
            slope_long = closes[-1] - closes[-60]
            pct_long = slope_long / max(price_now, 1e-6)
            if pct_long > 0.004:
                bias = 1
            elif pct_long < -0.004:
                bias = -1
        # 최근 방향 비율로 편향 교정
        if len(closes) >= 15:
            rets = [1 if closes[i] >= closes[i - 1] else -1 for i in range(len(closes) - 14, len(closes))]
            up_ratio = sum(1 for r in rets if r > 0) / len(rets)
            if up_ratio >= 0.6:
                bias = 1
            elif up_ratio <= 0.4:
                bias = -1
        return bias

    def _infer_regime(self, closes) -> str:
        if not closes or len(closes) < 30:
            return "chop"
        fast_period = min(80, len(closes))
        slow_period = min(200, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        slope_short = closes[-1] - closes[max(0, len(closes) - 6)]
        slope_long = closes[-1] - closes[max(0, len(closes) - 40)]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, min(len(closes), 180))]
        vol = float(np.std(rets)) if rets else 0.0
        # 변동성 높고 추세 약하면 volatile
        if vol > 0.01 and abs(slope_short) < closes[-1] * 0.0015:
            return "volatile"
        # 강한 상승/하락을 더 민감하게
        if fast > slow and slope_long > 0 and slope_short > 0:
            return "bull"
        if fast < slow and slope_long < 0 and slope_short < 0:
            return "bear"
        return "chop"

    def _compute_ofi_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym)
        if not ob:
            return 0.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        # 심도별 가중(상위호가 가중치 ↑)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        bid_vol = sum(float(b[1]) * weights[i] for i, b in enumerate(bids[: len(weights)]) if len(b) >= 2)
        ask_vol = sum(float(a[1]) * weights[i] for i, a in enumerate(asks[: len(weights)]) if len(a) >= 2)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / denom)

    def _get_alpha_state(self, sym: str) -> dict:
        st = self._alpha_state.get(sym)
        if st is None:
            g_ovr = (self._alpha_garch_symbol_overrides.get(sym) or self._alpha_garch_override or {})
            st = {
                "vpin": VPINState(bucket_size=0.0, window=int(getattr(mc_config, "vpin_window", 50))),
                "garch": GARCHState(
                    omega=float(g_ovr.get("omega", getattr(mc_config, "garch_omega", 1e-6))),
                    alpha=float(g_ovr.get("alpha", getattr(mc_config, "garch_alpha", 0.05))),
                    beta=float(g_ovr.get("beta", getattr(mc_config, "garch_beta", 0.90))),
                    var=max(float(g_ovr.get("var", 1e-6) or 1e-6), 1e-10),
                ),
                "bayes": BayesMeanState(mu_mean=0.0, mu_var=float(getattr(mc_config, "bayes_obs_var", 1e-4))),
                "pf": ParticleFilterState(n_particles=int(getattr(mc_config, "pf_particles", 128))),
                "kf": KalmanCVState(Q=float(getattr(mc_config, "kf_q", 1e-6)), R=float(getattr(mc_config, "kf_r", 1e-4))),
                "hawkes": HawkesState(
                    lambda_buy=0.0,
                    lambda_sell=0.0,
                    alpha=float(getattr(mc_config, "hawkes_alpha", 0.5)),
                    beta=float(getattr(mc_config, "hawkes_beta", 2.0)),
                ),
                "hmm": GaussianHMMState(
                    n_states=int(getattr(mc_config, "hmm_states", 3)),
                    adapt_lr=float(getattr(mc_config, "hmm_adapt_lr", 0.02)),
                ),
                "hurst_rets": deque(maxlen=int(getattr(mc_config, "hurst_window", 120))),
                "last_price": None,
                "last_ts": None,
                "last_kline_ts": 0,
                "mlofi_prev": None,
                "last_hurst_ts": 0,
                "last_pf_ts": 0,
                "last_vpin_ts": 0,
                "last_garch_ts": 0,
                "last_ml_ts": 0,
                "last_hmm_ts": 0,
                "hurst_last": 0.0,
                "mu_pf_last": 0.0,
                "mu_ml_last": 0.0,
                "vpin_last": 0.0,
                "sigma_garch_last": 0.0,
                "hmm_last": {},
                "alpha_cache": {},
            }
            self._alpha_state[sym] = st
        return st

    def _parse_hurst_taus(self) -> list[int]:
        try:
            raw = str(getattr(mc_config, "hurst_taus", "2,4,8,16"))
            return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        except Exception:
            return [2, 4, 8, 16]

    @staticmethod
    def _parse_utc_hours(raw: str | None) -> list[int]:
        out: list[int] = []
        if raw is None:
            return out
        for tok in str(raw).split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                h = int(t)
            except Exception:
                continue
            if 0 <= h <= 23:
                out.append(h)
        return sorted(set(out))

    @staticmethod
    def _utc_window_key(ts_ms: int, window_hours: list[int], window_min: float) -> str | None:
        if not window_hours:
            return None
        try:
            win = max(1.0, float(window_min))
        except Exception:
            win = 60.0
        dt = datetime.datetime.utcfromtimestamp(max(0, int(ts_ms)) / 1000.0)
        for h in window_hours:
            delta_min = abs((int(dt.hour) * 60 + int(dt.minute)) - (int(h) * 60))
            # wrap-around distance (e.g. 23:58 vs 00:02)
            delta_min = min(delta_min, 24 * 60 - delta_min)
            if delta_min <= win:
                return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}-{int(h):02d}"
        return None

    def _safe_float(self, value, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            out = float(value)
            if not np.isfinite(out):
                return default
            return out
        except Exception:
            return default

    def _effective_min_hold_sec(self, pos: dict | None) -> float:
        """
        Base min-hold + optional opt_hold ratio floor.
        Useful to reduce premature exits before the modeled hold horizon.
        """
        try:
            base_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
        except Exception:
            base_sec = float(POSITION_HOLD_MIN_SEC)
        base_sec = max(0.0, float(base_sec))

        if not isinstance(pos, dict):
            return base_sec

        try:
            ratio = float(os.environ.get("EXIT_MIN_HOLD_RATIO_TO_OPT", 0.60) or 0.60)
        except Exception:
            ratio = 0.60
        ratio = max(0.0, min(1.5, float(ratio)))

        if ratio <= 0.0:
            return base_sec

        opt_hold = (
            pos.get("opt_hold_curr_sec")
            or pos.get("opt_hold_entry_sec")
            or pos.get("opt_hold_sec")
        )
        opt_hold_f = self._safe_float(opt_hold, None)
        if opt_hold_f is None or opt_hold_f <= 0:
            return base_sec

        dyn_sec = float(opt_hold_f) * float(ratio)
        try:
            dyn_cap = float(os.environ.get("EXIT_MIN_HOLD_RATIO_CAP_SEC", 1800) or 1800)
        except Exception:
            dyn_cap = 1800.0
        dyn_cap = max(0.0, float(dyn_cap))
        if dyn_cap > 0:
            dyn_sec = min(float(dyn_sec), float(dyn_cap))
        return max(base_sec, float(dyn_sec))

    def _advance_exit_confirmation(
        self,
        pos: dict | None,
        key: str,
        *,
        triggered: bool,
        ts_ms: int,
        required_ticks: int,
        reset_sec: float,
    ) -> tuple[bool, int]:
        """Stateful confirmation gate to prevent one-tick noisy exits."""
        if not isinstance(pos, dict):
            return bool(triggered), 1 if triggered else 0
        req = max(1, int(required_ticks))
        cnt_key = f"{key}_confirm_count"
        last_key = f"{key}_confirm_last_ms"
        if req <= 1:
            if not triggered:
                pos[cnt_key] = 0
                pos[last_key] = 0
                return False, 0
            pos[cnt_key] = 1
            pos[last_key] = int(ts_ms)
            return True, 1

        if not triggered:
            pos[cnt_key] = 0
            pos[last_key] = 0
            return False, 0

        try:
            prev_cnt = int(pos.get(cnt_key, 0) or 0)
        except Exception:
            prev_cnt = 0
        try:
            prev_ts = int(pos.get(last_key, 0) or 0)
        except Exception:
            prev_ts = 0
        try:
            reset_ms = int(max(0.0, float(reset_sec)) * 1000.0)
        except Exception:
            reset_ms = 0
        if reset_ms > 0 and prev_ts > 0 and (int(ts_ms) - int(prev_ts)) > reset_ms:
            prev_cnt = 0
        cnt = int(prev_cnt) + 1
        pos[cnt_key] = int(cnt)
        pos[last_key] = int(ts_ms)
        return bool(cnt >= req), int(cnt)

    def _resolve_exit_mode_state(
        self,
        exit_diag: dict | None,
        *,
        shock_threshold_env: str = "EVENT_EXIT_SHOCK_FAST_THRESHOLD",
        shock_threshold_default: float = 1.0,
    ) -> dict[str, object]:
        diag = exit_diag if isinstance(exit_diag, dict) else {}
        shock_score = max(0.0, self._safe_float(diag.get("shock_score"), 0.0) or 0.0)
        noise_mode = bool(diag.get("noise_mode"))
        mode_hint = str(diag.get("shock_bucket") or "").strip().lower()
        try:
            shock_thr = float(os.environ.get(shock_threshold_env, shock_threshold_default) or shock_threshold_default)
        except Exception:
            shock_thr = float(shock_threshold_default)
        if mode_hint not in ("shock", "normal", "noise"):
            mode = "noise" if noise_mode else "normal"
        else:
            mode = mode_hint
        # Always prioritize adverse shock branch when score crosses threshold.
        if float(shock_score) >= float(shock_thr):
            mode = "shock"
        elif mode == "normal" and noise_mode:
            mode = "noise"
        return {
            "mode": str(mode),
            "shock_score": float(shock_score),
            "noise_mode": bool(noise_mode),
            "shock_threshold": float(shock_thr),
        }

    def _event_exit_strict_mode_enabled(self) -> bool:
        try:
            self._event_exit_strict_consistency = str(
                os.environ.get("EVENT_EXIT_STRICT_CONSISTENCY_MODE", "1")
            ).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return bool(self._event_exit_strict_consistency)

    def _evaluate_event_exit_core(
        self,
        *,
        event_score: float,
        event_cvar_pct: float,
        event_p_sl: float,
        event_p_tp: float,
        metrics_available: bool,
        policy: ExitPolicy,
        exit_diag: dict | None,
        strict_mode: bool,
        shock_threshold_env: str = "EVENT_EXIT_SHOCK_MODE_THRESHOLD",
        shock_threshold_default: float = 1.0,
    ) -> dict[str, object]:
        event_mode_state = self._resolve_exit_mode_state(
            exit_diag,
            shock_threshold_env=shock_threshold_env,
            shock_threshold_default=shock_threshold_default,
        )
        shock_score_now = float(event_mode_state.get("shock_score") or 0.0)
        noise_mode_now = bool(event_mode_state.get("noise_mode"))
        event_mode = str(event_mode_state.get("mode") or "normal")

        score_thr = float(policy.min_event_score)
        cvar_thr = float(policy.max_abs_event_cvar_r)
        psl_thr = float(policy.max_event_p_sl)
        ptp_thr = float(policy.min_event_p_tp)
        if strict_mode:
            if event_mode == "shock":
                try:
                    score_thr += float(os.environ.get("EVENT_EXIT_SCORE_OFFSET_SHOCK", 0.0008) or 0.0008)
                except Exception:
                    pass
                try:
                    psl_thr *= float(os.environ.get("EVENT_EXIT_MAX_P_SL_SCALE_SHOCK", 0.92) or 0.92)
                except Exception:
                    pass
                try:
                    cvar_thr *= float(os.environ.get("EVENT_EXIT_MAX_CVAR_SCALE_SHOCK", 0.85) or 0.85)
                except Exception:
                    pass
                try:
                    ptp_thr *= float(os.environ.get("EVENT_EXIT_MIN_P_TP_SCALE_SHOCK", 1.08) or 1.08)
                except Exception:
                    pass
            elif event_mode == "noise":
                try:
                    score_thr += float(os.environ.get("EVENT_EXIT_SCORE_OFFSET_NOISE", -0.0006) or -0.0006)
                except Exception:
                    pass
                try:
                    psl_thr *= float(os.environ.get("EVENT_EXIT_MAX_P_SL_SCALE_NOISE", 1.08) or 1.08)
                except Exception:
                    pass
                try:
                    cvar_thr *= float(os.environ.get("EVENT_EXIT_MAX_CVAR_SCALE_NOISE", 1.08) or 1.08)
                except Exception:
                    pass
                try:
                    ptp_thr *= float(os.environ.get("EVENT_EXIT_MIN_P_TP_SCALE_NOISE", 0.92) or 0.92)
                except Exception:
                    pass
            else:
                try:
                    score_thr += float(os.environ.get("EVENT_EXIT_SCORE_OFFSET_NORMAL", 0.0) or 0.0)
                except Exception:
                    pass
                try:
                    psl_thr *= float(os.environ.get("EVENT_EXIT_MAX_P_SL_SCALE_NORMAL", 1.0) or 1.0)
                except Exception:
                    pass
                try:
                    cvar_thr *= float(os.environ.get("EVENT_EXIT_MAX_CVAR_SCALE_NORMAL", 1.0) or 1.0)
                except Exception:
                    pass
                try:
                    ptp_thr *= float(os.environ.get("EVENT_EXIT_MIN_P_TP_SCALE_NORMAL", 1.0) or 1.0)
                except Exception:
                    pass

        score_thr = float(np.clip(score_thr, -0.05, 0.02))
        psl_thr = float(np.clip(psl_thr, 0.25, 0.99))
        cvar_thr = float(np.clip(cvar_thr, 0.002, 0.25))
        ptp_thr = float(np.clip(ptp_thr, 0.05, 0.90))

        score_hit = bool(metrics_available and (float(event_score) <= float(score_thr)))
        cvar_hit = bool(metrics_available and (abs(float(event_cvar_pct)) >= float(cvar_thr)))
        psl_hit = bool(metrics_available and (float(event_p_sl) >= float(psl_thr)))
        ptp_hit = bool(metrics_available and (float(event_p_tp) <= float(ptp_thr)))
        event_exit_hit = bool(score_hit or cvar_hit or psl_hit or ptp_hit)
        return {
            "mode": str(event_mode),
            "shock_score": float(shock_score_now),
            "noise_mode": bool(noise_mode_now),
            "strict_mode": bool(strict_mode),
            "threshold_score": float(score_thr),
            "threshold_cvar": float(cvar_thr),
            "threshold_psl": float(psl_thr),
            "threshold_ptp": float(ptp_thr),
            "hit_score": bool(score_hit),
            "hit_cvar": bool(cvar_hit),
            "hit_psl": bool(psl_hit),
            "hit_ptp": bool(ptp_hit),
            "event_exit_hit": bool(event_exit_hit),
        }

    def _get_exit_confirmation_rule(
        self,
        prefix: str,
        mode: str,
        *,
        default_normal: int = 2,
        default_shock: int = 1,
        default_noise: int = 3,
        default_reset_sec: float = 180.0,
    ) -> tuple[int, float, dict[str, int]]:
        mode_norm = str(mode or "normal").strip().lower()
        if mode_norm not in ("shock", "noise", "normal"):
            mode_norm = "normal"
        try:
            ticks_shock = int(os.environ.get(f"{prefix}_CONFIRM_TICKS_SHOCK", default_shock) or default_shock)
        except Exception:
            ticks_shock = int(default_shock)
        try:
            ticks_normal = int(os.environ.get(f"{prefix}_CONFIRM_TICKS_NORMAL", default_normal) or default_normal)
        except Exception:
            ticks_normal = int(default_normal)
        try:
            ticks_noise = int(os.environ.get(f"{prefix}_CONFIRM_TICKS_NOISE", default_noise) or default_noise)
        except Exception:
            ticks_noise = int(default_noise)
        try:
            reset_sec = float(os.environ.get(f"{prefix}_CONFIRM_RESET_SEC", default_reset_sec) or default_reset_sec)
        except Exception:
            reset_sec = float(default_reset_sec)
        if mode_norm == "shock":
            required = int(max(1, ticks_shock))
        elif mode_norm == "noise":
            required = int(max(1, ticks_noise))
        else:
            required = int(max(1, ticks_normal))
        return required, float(max(0.0, reset_sec)), {
            "shock": int(max(1, ticks_shock)),
            "normal": int(max(1, ticks_normal)),
            "noise": int(max(1, ticks_noise)),
        }

    def _event_exit_gate_decision(
        self,
        *,
        pos: dict,
        ts_ms: int,
        event_exit_hit: bool,
        event_mode: str,
        shock_score: float,
        p_sl_evt: float,
        cvar_pct_evt: float,
        p_tp_evt: float | None = None,
        ptp_thr: float | None = None,
        meta: dict | None = None,
        confirm_key: str = "event_exit",
    ) -> dict[str, object]:
        """
        Shared gating for event-exit decisions.
        Applies t* progress guard + shock/noise confirmation rules so precheck/runtime stay consistent.
        """
        m = meta if isinstance(meta, dict) else {}
        mode_norm = str(event_mode or "normal").strip().lower()
        if mode_norm not in ("shock", "normal", "noise"):
            mode_norm = "normal"
        try:
            shock_fast_threshold = float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD", 1.0) or 1.0)
        except Exception:
            shock_fast_threshold = 1.0
        try:
            hard_psl = float(os.environ.get("EVENT_EXIT_HARD_PSL", 0.90) or 0.90)
        except Exception:
            hard_psl = 0.90
        try:
            hard_cvar = float(os.environ.get("EVENT_EXIT_HARD_CVAR", 0.030) or 0.030)
        except Exception:
            hard_cvar = 0.030
        try:
            hard_ptp_scale = float(os.environ.get("EVENT_EXIT_HARD_P_TP_SCALE", 0.75) or 0.75)
        except Exception:
            hard_ptp_scale = 0.75
        try:
            hard_ptp_floor = float(os.environ.get("EVENT_EXIT_HARD_P_TP_FLOOR", 0.08) or 0.08)
        except Exception:
            hard_ptp_floor = 0.08
        ptp_low_adverse = False
        try:
            p_tp_now = self._safe_float(p_tp_evt, None)
            if p_tp_now is not None:
                ptp_base = self._safe_float(ptp_thr, None)
                if ptp_base is None or float(ptp_base) <= 0:
                    ptp_base = float(hard_ptp_floor)
                ptp_cut = max(float(hard_ptp_floor), float(ptp_base) * float(hard_ptp_scale))
                ptp_low_adverse = bool(float(p_tp_now) <= float(ptp_cut))
        except Exception:
            ptp_low_adverse = False
        severe_adverse = bool(
            (float(p_sl_evt) >= float(hard_psl))
            or (abs(float(cvar_pct_evt)) >= float(hard_cvar))
            or bool(ptp_low_adverse)
        )
        try:
            fast_only_shock = str(os.environ.get("EXIT_FAST_ONLY_SHOCK", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            fast_only_shock = True

        # Default idle confirm state
        confirm_ticks_normal, confirm_reset_sec, confirm_ticks_map = self._get_exit_confirmation_rule(
            "EVENT_EXIT",
            "normal",
            default_normal=2,
            default_shock=1,
            default_noise=3,
            default_reset_sec=180.0,
        )

        if not bool(event_exit_hit):
            _confirmed, _cnt = self._advance_exit_confirmation(
                pos,
                str(confirm_key or "event_exit"),
                triggered=False,
                ts_ms=int(ts_ms),
                required_ticks=int(max(1, confirm_ticks_normal)),
                reset_sec=float(max(0.0, confirm_reset_sec)),
            )
            return {
                "allow_exit": False,
                "guard_reason": "idle",
                "event_mode": str(mode_norm),
                "shock_score": float(shock_score),
                "severe_adverse": bool(severe_adverse),
                "severe_adverse_ptp_low": bool(ptp_low_adverse),
                "fast_only_shock": bool(fast_only_shock),
                "confirm_mode": "idle",
                "confirm_required": int(max(1, confirm_ticks_normal)),
                "confirm_count": int(_cnt),
                "confirm_ok": False,
                "guarded_by_tstar": False,
                "guard_progress": None,
                "guard_required": None,
                "guard_hold_target_sec": None,
                "guard_age_sec": None,
                "guard_remaining_sec": None,
            }

        # t* progress guard: avoid early exits unless adverse shock is sufficiently strong.
        try:
            tstar_guard_enabled = str(os.environ.get("EVENT_MC_TSTAR_GUARD_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            tstar_guard_enabled = True
        if tstar_guard_enabled:
            try:
                h_low = float(getattr(mc_config, "hurst_low", 0.45))
                h_high = float(getattr(mc_config, "hurst_high", 0.55))
            except Exception:
                h_low, h_high = 0.45, 0.55
            try:
                prog_trend = float(os.environ.get("EVENT_MC_TSTAR_MIN_PROGRESS_TREND", 0.45) or 0.45)
            except Exception:
                prog_trend = 0.45
            try:
                prog_random = float(os.environ.get("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", 0.60) or 0.60)
            except Exception:
                prog_random = 0.60
            try:
                prog_mr = float(os.environ.get("EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", 0.70) or 0.70)
            except Exception:
                prog_mr = 0.70
            try:
                bypass_shock = float(os.environ.get("EVENT_MC_TSTAR_BYPASS_SHOCK", 1.20) or 1.20)
            except Exception:
                bypass_shock = 1.20
            try:
                bypass_psl = float(os.environ.get("EVENT_MC_TSTAR_BYPASS_PSL", 0.92) or 0.92)
            except Exception:
                bypass_psl = 0.92
            try:
                entry_ts_ms = self._normalize_entry_time_ms((pos or {}).get("entry_time"), default=ts_ms)
                age_sec = max(0.0, (int(ts_ms) - int(entry_ts_ms)) / 1000.0)
            except Exception:
                age_sec = 0.0
            hold_target_sec = None
            try:
                hold_target_sec = float(
                    (pos or {}).get("opt_hold_curr_sec")
                    or (pos or {}).get("opt_hold_entry_sec")
                    or (pos or {}).get("opt_hold_sec")
                    or m.get("event_hold_target_sec")
                    or m.get("policy_horizon_eff_sec")
                    or m.get("best_h")
                    or 0.0
                )
            except Exception:
                hold_target_sec = 0.0
            if not hold_target_sec or hold_target_sec <= 0:
                try:
                    hold_target_sec = float(
                        (pos or {}).get("policy_min_hold_eff_sec")
                        or self._effective_min_hold_sec(pos or {})
                        or os.environ.get("EXIT_MIN_HOLD_SEC", 0.0)
                        or 0.0
                    )
                except Exception:
                    hold_target_sec = 0.0
            try:
                fallback_floor = float(os.environ.get("EVENT_MC_TSTAR_MIN_HOLD_FALLBACK_SEC", 0.0) or 0.0)
            except Exception:
                fallback_floor = 0.0
            if fallback_floor > 0:
                hold_target_sec = float(max(float(hold_target_sec or 0.0), fallback_floor))
            if hold_target_sec and hold_target_sec > 0:
                progress = float(age_sec / max(float(hold_target_sec), 1e-6))
                hurst_now = self._safe_float(m.get("event_exit_dynamic_hurst"), None)
                if hurst_now is None:
                    hurst_now = self._safe_float(m.get("hurst"), None)
                regime_req = prog_random
                if hurst_now is not None and float(hurst_now) > h_high:
                    regime_req = prog_trend
                elif hurst_now is not None and float(hurst_now) < h_low:
                    regime_req = prog_mr
                remaining_sec = max(0.0, float(hold_target_sec) - float(age_sec))
                severe_adverse_guard = bool(float(p_sl_evt) >= float(bypass_psl) or bool(ptp_low_adverse))
                if (float(progress) < float(regime_req)) and (float(shock_score) < float(bypass_shock)) and (not severe_adverse_guard):
                    _confirmed, _cnt = self._advance_exit_confirmation(
                        pos,
                        str(confirm_key or "event_exit"),
                        triggered=False,
                        ts_ms=int(ts_ms),
                        required_ticks=int(max(1, confirm_ticks_map.get("normal", 2))),
                        reset_sec=float(max(0.0, confirm_reset_sec)),
                    )
                    return {
                        "allow_exit": False,
                        "guard_reason": "tstar_progress",
                        "event_mode": str(mode_norm),
                        "shock_score": float(shock_score),
                        "severe_adverse": bool(severe_adverse),
                        "severe_adverse_ptp_low": bool(ptp_low_adverse),
                        "fast_only_shock": bool(fast_only_shock),
                        "confirm_mode": "guarded",
                        "confirm_required": int(max(1, confirm_ticks_map.get("normal", 2))),
                        "confirm_count": int(_cnt),
                        "confirm_ok": False,
                        "guarded_by_tstar": True,
                        "guard_progress": float(progress),
                        "guard_required": float(regime_req),
                        "guard_hold_target_sec": float(hold_target_sec),
                        "guard_age_sec": float(age_sec),
                        "guard_remaining_sec": float(remaining_sec),
                    }

        confirm_mode = str(mode_norm)
        if (not fast_only_shock) and (float(shock_score) >= float(shock_fast_threshold) or bool(severe_adverse)):
            confirm_mode = "shock"

        confirm_required, confirm_reset_sec, _ticks = self._get_exit_confirmation_rule(
            "EVENT_EXIT",
            confirm_mode,
            default_normal=2,
            default_shock=1,
            default_noise=3,
            default_reset_sec=180.0,
        )
        if confirm_mode != "shock":
            try:
                non_shock_min = int(os.environ.get("EVENT_EXIT_MIN_CONFIRM_NON_SHOCK", 3) or 3)
            except Exception:
                non_shock_min = 3
            try:
                noise_min = int(os.environ.get("EVENT_EXIT_MIN_CONFIRM_NOISE", non_shock_min) or non_shock_min)
            except Exception:
                noise_min = non_shock_min
            if confirm_mode == "noise":
                confirm_required = int(max(confirm_required, max(1, noise_min)))
            else:
                confirm_required = int(max(confirm_required, max(1, non_shock_min)))

        confirm_ok, confirm_cnt = self._advance_exit_confirmation(
            pos,
            str(confirm_key or "event_exit"),
            triggered=True,
            ts_ms=int(ts_ms),
            required_ticks=int(max(1, confirm_required)),
            reset_sec=float(max(0.0, confirm_reset_sec)),
        )
        return {
            "allow_exit": bool(confirm_ok),
            "guard_reason": "confirm_pending" if not bool(confirm_ok) else "confirmed",
            "event_mode": str(mode_norm),
            "shock_score": float(shock_score),
            "severe_adverse": bool(severe_adverse),
            "severe_adverse_ptp_low": bool(ptp_low_adverse),
            "fast_only_shock": bool(fast_only_shock),
            "confirm_mode": str(confirm_mode),
            "confirm_required": int(max(1, confirm_required)),
            "confirm_count": int(confirm_cnt),
            "confirm_ok": bool(confirm_ok),
            "guarded_by_tstar": False,
            "guard_progress": None,
            "guard_required": None,
            "guard_hold_target_sec": None,
            "guard_age_sec": None,
            "guard_remaining_sec": None,
        }

    @staticmethod
    def _new_trade_uid() -> str:
        return uuid.uuid4().hex

    def _capture_position_observability(
        self,
        pos: dict | None,
        *,
        decision: dict | None = None,
        ctx: dict | None = None,
    ) -> None:
        """
        ENTRY/EXIT 분석에 필요한 핵심 피처를 포지션에 지속적으로 스냅샷한다.
        EXIT 시점에 raw_data/DB로 그대로 흘려보내기 위한 최소 집합이다.
        """
        if not isinstance(pos, dict):
            return
        meta = dict((decision or {}).get("meta") or {}) if isinstance(decision, dict) else {}
        if not meta and isinstance(decision, dict):
            details = decision.get("details")
            if isinstance(details, list):
                for d in details:
                    if not isinstance(d, dict):
                        continue
                    m2 = d.get("meta")
                    if isinstance(m2, dict) and m2:
                        meta = dict(m2)
                        break
        ctx = dict(ctx or {})

        def _pick_float(*vals):
            for v in vals:
                fv = self._safe_float(v, None)
                if fv is not None:
                    return float(fv)
            return None

        def _pick_str(*vals):
            for v in vals:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            return None

        def _pick_bool(*vals):
            for v in vals:
                bv = self._coerce_opt_bool(v)
                if bv is not None:
                    return bool(bv)
            return None

        regime = _pick_str(ctx.get("regime"), meta.get("regime"), pos.get("regime"))
        session = _pick_str(ctx.get("session"), meta.get("session"), pos.get("session"))
        if regime is not None:
            pos["regime"] = regime
        if session is not None:
            pos["session"] = session
        entry_id = _pick_str(pos.get("entry_id"), pos.get("entry_link_id"))
        if entry_id is not None:
            pos["entry_id"] = entry_id
            pos["entry_link_id"] = entry_id

        # Alpha snapshots
        vpin = _pick_float(
            meta.get("vpin"),
            (decision or {}).get("vpin") if isinstance(decision, dict) else None,
            ctx.get("vpin"),
            meta.get("event_exit_dynamic_vpin"),
            pos.get("alpha_vpin"),
        )
        hurst = _pick_float(
            meta.get("hurst"),
            (decision or {}).get("hurst") if isinstance(decision, dict) else None,
            ctx.get("hurst"),
            meta.get("event_exit_dynamic_hurst"),
            pos.get("alpha_hurst"),
        )
        if vpin is not None:
            pos["alpha_vpin"] = float(max(0.0, min(1.0, vpin)))
        if hurst is not None:
            pos["alpha_hurst"] = float(max(0.0, min(1.0, hurst)))

        mu_alpha = _pick_float(
            meta.get("mu_alpha"),
            (decision or {}).get("mu_alpha") if isinstance(decision, dict) else None,
            meta.get("event_exit_dynamic_mu_alpha"),
            pos.get("pred_mu_alpha"),
        )
        mu_alpha_raw = _pick_float(
            meta.get("mu_alpha_raw"),
            (decision or {}).get("mu_alpha_raw") if isinstance(decision, dict) else None,
            pos.get("pred_mu_alpha_raw"),
        )
        mu_dir_conf = _pick_float(
            meta.get("mu_dir_conf"),
            (decision or {}).get("mu_dir_conf") if isinstance(decision, dict) else None,
            pos.get("pred_mu_dir_conf"),
        )
        mu_dir_edge = _pick_float(
            meta.get("mu_dir_edge"),
            (decision or {}).get("mu_dir_edge") if isinstance(decision, dict) else None,
            pos.get("pred_mu_dir_edge"),
        )
        if mu_alpha is not None:
            pos["pred_mu_alpha"] = mu_alpha
        if mu_alpha_raw is not None:
            pos["pred_mu_alpha_raw"] = mu_alpha_raw
        if mu_dir_conf is not None:
            pos["pred_mu_dir_conf"] = mu_dir_conf
        if mu_dir_edge is not None:
            pos["pred_mu_dir_edge"] = mu_dir_edge
        try:
            mu_dir_prob_long = _pick_float(
                meta.get("mu_dir_prob_long"),
                (decision or {}).get("mu_dir_prob_long") if isinstance(decision, dict) else None,
                pos.get("pred_mu_dir_prob_long"),
            )
        except Exception:
            mu_dir_prob_long = None
        if mu_dir_prob_long is not None:
            pos["pred_mu_dir_prob_long"] = float(max(0.0, min(1.0, mu_dir_prob_long)))
        entry_quality_score = _pick_float(
            meta.get("entry_quality_score"),
            (decision or {}).get("entry_quality_score") if isinstance(decision, dict) else None,
            pos.get("entry_quality_score"),
        )
        one_way_move_score = _pick_float(
            meta.get("one_way_move_score"),
            (decision or {}).get("one_way_move_score") if isinstance(decision, dict) else None,
            pos.get("one_way_move_score"),
        )
        leverage_signal_score = _pick_float(
            meta.get("leverage_signal_score"),
            (decision or {}).get("leverage_signal_score") if isinstance(decision, dict) else None,
            pos.get("leverage_signal_score"),
        )
        if entry_quality_score is not None:
            pos["entry_quality_score"] = float(max(0.0, min(1.0, entry_quality_score)))
        if one_way_move_score is not None:
            pos["one_way_move_score"] = float(max(0.0, min(1.0, one_way_move_score)))
        if leverage_signal_score is not None:
            pos["leverage_signal_score"] = float(max(0.0, min(1.0, leverage_signal_score)))

        # Exit policy thresholds (dynamic 포함)
        score_th = _pick_float(
            meta.get("policy_score_threshold_eff"),
            meta.get("score_threshold"),
            (decision or {}).get("policy_score_threshold") if isinstance(decision, dict) else None,
            pos.get("policy_score_threshold"),
        )
        event_min_score = _pick_float(
            meta.get("event_exit_min_score"),
            (decision or {}).get("event_exit_min_score") if isinstance(decision, dict) else None,
            pos.get("policy_event_exit_min_score"),
        )
        dd_floor_dyn = _pick_float(
            meta.get("unrealized_dd_floor_dyn"),
            (decision or {}).get("unrealized_dd_floor_dyn") if isinstance(decision, dict) else None,
            pos.get("policy_unrealized_dd_floor"),
        )
        if score_th is not None:
            pos["policy_score_threshold"] = score_th
        if event_min_score is not None:
            pos["policy_event_exit_min_score"] = event_min_score
        if dd_floor_dyn is not None:
            pos["policy_unrealized_dd_floor"] = dd_floor_dyn

        # Event exit snapshots (precheck/runtime). Keep latest values for ENTER↔EXIT consistency analysis.
        event_score = _pick_float(meta.get("event_exit_score"), pos.get("event_exit_score"))
        event_cvar_pct = _pick_float(meta.get("event_cvar_pct"), pos.get("event_cvar_pct"))
        event_p_sl = _pick_float(meta.get("event_p_sl"), pos.get("event_p_sl"))
        event_p_tp = _pick_float(meta.get("event_p_tp"), pos.get("event_p_tp"))
        event_thr_score = _pick_float(
            meta.get("event_exit_threshold_score"),
            meta.get("event_exit_min_score"),
            pos.get("event_exit_threshold_score"),
        )
        event_thr_cvar = _pick_float(
            meta.get("event_exit_threshold_cvar"),
            meta.get("event_exit_max_cvar"),
            pos.get("event_exit_threshold_cvar"),
        )
        event_thr_psl = _pick_float(
            meta.get("event_exit_threshold_psl"),
            meta.get("event_exit_max_p_sl"),
            pos.get("event_exit_threshold_psl"),
        )
        event_thr_ptp = _pick_float(
            meta.get("event_exit_threshold_ptp"),
            meta.get("event_exit_min_p_tp"),
            pos.get("event_exit_threshold_ptp"),
        )
        event_mode = _pick_str(
            meta.get("event_exit_mode"),
            meta.get("event_exit_dynamic_shock_bucket"),
            pos.get("event_exit_mode"),
        )
        event_hit = _pick_bool(meta.get("event_exit_hit"), pos.get("event_exit_hit"))
        entry_event_exit_ok = _pick_bool(meta.get("event_exit_ok"), pos.get("entry_event_exit_ok"))
        strict_mode = _pick_bool(meta.get("event_exit_strict_mode"), pos.get("event_exit_strict_mode"))
        event_eval_source = _pick_str(meta.get("event_eval_source"), pos.get("event_eval_source"))
        event_pre_allow = _pick_bool(meta.get("event_precheck_allow_exit_now"), pos.get("event_precheck_allow_exit_now"))
        event_pre_guard = _pick_str(meta.get("event_precheck_guard_reason"), pos.get("event_precheck_guard_reason"))
        event_pre_confirm_mode = _pick_str(meta.get("event_precheck_confirm_mode"), pos.get("event_precheck_confirm_mode"))
        event_pre_confirm_required = _pick_float(meta.get("event_precheck_confirm_required"), pos.get("event_precheck_confirm_required"))
        event_pre_confirm_count = _pick_float(meta.get("event_precheck_confirm_count"), pos.get("event_precheck_confirm_count"))
        event_pre_confirmed = _pick_bool(meta.get("event_precheck_confirmed"), pos.get("event_precheck_confirmed"))
        event_pre_shock_score = _pick_float(meta.get("event_precheck_shock_score"), pos.get("event_precheck_shock_score"))
        event_pre_severe_adverse = _pick_bool(meta.get("event_precheck_severe_adverse"), pos.get("event_precheck_severe_adverse"))
        event_pre_severe_ptp = _pick_bool(
            meta.get("event_precheck_severe_adverse_ptp_low"),
            pos.get("event_precheck_severe_adverse_ptp_low"),
        )
        event_pre_guard_progress = _pick_float(meta.get("event_precheck_guard_progress"), pos.get("event_precheck_guard_progress"))
        event_pre_guard_required = _pick_float(meta.get("event_precheck_guard_required"), pos.get("event_precheck_guard_required"))
        event_pre_guard_hold = _pick_float(meta.get("event_precheck_guard_hold_target_sec"), pos.get("event_precheck_guard_hold_target_sec"))
        event_pre_guard_age = _pick_float(meta.get("event_precheck_guard_age_sec"), pos.get("event_precheck_guard_age_sec"))
        event_pre_guard_remaining = _pick_float(
            meta.get("event_precheck_guard_remaining_sec"),
            pos.get("event_precheck_guard_remaining_sec"),
        )
        event_guard = _pick_str(meta.get("event_exit_guard"), pos.get("event_exit_guard"))
        event_guard_progress = _pick_float(meta.get("event_exit_guard_progress"), pos.get("event_exit_guard_progress"))
        event_guard_required = _pick_float(meta.get("event_exit_guard_required"), pos.get("event_exit_guard_required"))
        event_guard_hold = _pick_float(meta.get("event_exit_guard_hold_target_sec"), pos.get("event_exit_guard_hold_target_sec"))
        event_guard_age = _pick_float(meta.get("event_exit_guard_age_sec"), pos.get("event_exit_guard_age_sec"))
        event_guard_remaining = _pick_float(meta.get("event_exit_guard_remaining_sec"), pos.get("event_exit_guard_remaining_sec"))
        event_confirm_mode = _pick_str(meta.get("event_exit_confirm_mode"), pos.get("event_exit_confirm_mode"))
        event_confirm_required = _pick_float(meta.get("event_exit_confirm_required"), pos.get("event_exit_confirm_required"))
        event_confirm_count = _pick_float(meta.get("event_exit_confirm_count"), pos.get("event_exit_confirm_count"))
        event_confirmed = _pick_bool(meta.get("event_exit_confirmed"), pos.get("event_exit_confirmed"))
        event_hold_target = _pick_float(meta.get("event_hold_target_sec"), pos.get("event_hold_target_sec"))
        event_hold_remaining = _pick_float(meta.get("event_hold_remaining_sec"), pos.get("event_hold_remaining_sec"))
        if event_score is not None:
            pos["event_exit_score"] = float(event_score)
        if event_cvar_pct is not None:
            pos["event_cvar_pct"] = float(event_cvar_pct)
        if event_p_sl is not None:
            pos["event_p_sl"] = float(max(0.0, min(1.0, event_p_sl)))
        if event_p_tp is not None:
            pos["event_p_tp"] = float(max(0.0, min(1.0, event_p_tp)))
        if event_thr_score is not None:
            pos["event_exit_threshold_score"] = float(event_thr_score)
        if event_thr_cvar is not None:
            pos["event_exit_threshold_cvar"] = float(event_thr_cvar)
        if event_thr_psl is not None:
            pos["event_exit_threshold_psl"] = float(event_thr_psl)
        if event_thr_ptp is not None:
            pos["event_exit_threshold_ptp"] = float(event_thr_ptp)
        if event_mode is not None:
            pos["event_exit_mode"] = str(event_mode)
        if event_hit is not None:
            pos["event_exit_hit"] = bool(event_hit)
        if entry_event_exit_ok is not None:
            pos["entry_event_exit_ok"] = bool(entry_event_exit_ok)
        if strict_mode is not None:
            pos["event_exit_strict_mode"] = bool(strict_mode)
        if event_eval_source is not None:
            pos["event_eval_source"] = str(event_eval_source)
        if event_pre_allow is not None:
            pos["event_precheck_allow_exit_now"] = bool(event_pre_allow)
        if event_pre_guard is not None:
            pos["event_precheck_guard_reason"] = str(event_pre_guard)
        if event_pre_confirm_mode is not None:
            pos["event_precheck_confirm_mode"] = str(event_pre_confirm_mode)
        if event_pre_confirm_required is not None:
            pos["event_precheck_confirm_required"] = int(max(1, round(float(event_pre_confirm_required))))
        if event_pre_confirm_count is not None:
            pos["event_precheck_confirm_count"] = int(max(0, round(float(event_pre_confirm_count))))
        if event_pre_confirmed is not None:
            pos["event_precheck_confirmed"] = bool(event_pre_confirmed)
        if event_pre_shock_score is not None:
            pos["event_precheck_shock_score"] = float(event_pre_shock_score)
        if event_pre_severe_adverse is not None:
            pos["event_precheck_severe_adverse"] = bool(event_pre_severe_adverse)
        if event_pre_severe_ptp is not None:
            pos["event_precheck_severe_adverse_ptp_low"] = bool(event_pre_severe_ptp)
        if event_pre_guard_progress is not None:
            pos["event_precheck_guard_progress"] = float(event_pre_guard_progress)
        if event_pre_guard_required is not None:
            pos["event_precheck_guard_required"] = float(event_pre_guard_required)
        if event_pre_guard_hold is not None:
            pos["event_precheck_guard_hold_target_sec"] = float(event_pre_guard_hold)
        if event_pre_guard_age is not None:
            pos["event_precheck_guard_age_sec"] = float(event_pre_guard_age)
        if event_pre_guard_remaining is not None:
            pos["event_precheck_guard_remaining_sec"] = float(event_pre_guard_remaining)
        if event_guard is not None:
            pos["event_exit_guard"] = str(event_guard)
        if event_guard_progress is not None:
            pos["event_exit_guard_progress"] = float(event_guard_progress)
        if event_guard_required is not None:
            pos["event_exit_guard_required"] = float(event_guard_required)
        if event_guard_hold is not None:
            pos["event_exit_guard_hold_target_sec"] = float(event_guard_hold)
        if event_guard_age is not None:
            pos["event_exit_guard_age_sec"] = float(event_guard_age)
        if event_guard_remaining is not None:
            pos["event_exit_guard_remaining_sec"] = float(event_guard_remaining)
        if event_confirm_mode is not None:
            pos["event_exit_confirm_mode"] = str(event_confirm_mode)
        if event_confirm_required is not None:
            pos["event_exit_confirm_required"] = int(max(1, round(float(event_confirm_required))))
        if event_confirm_count is not None:
            pos["event_exit_confirm_count"] = int(max(0, round(float(event_confirm_count))))
        if event_confirmed is not None:
            pos["event_exit_confirmed"] = bool(event_confirmed)
        if event_hold_target is not None:
            pos["event_hold_target_sec"] = float(event_hold_target)
        if event_hold_remaining is not None:
            pos["event_hold_remaining_sec"] = float(event_hold_remaining)
        pos["event_eval_ts_ms"] = int(now_ms())

        # Keep key observability fields non-null for downstream analytics.
        if not pos.get("regime"):
            pos["regime"] = "chop"
        if pos.get("alpha_vpin") is None:
            pos["alpha_vpin"] = 0.0
        if pos.get("alpha_hurst") is None:
            pos["alpha_hurst"] = 0.5
        if pos.get("pred_mu_alpha") is None:
            pos["pred_mu_alpha"] = 0.0
        if pos.get("pred_mu_dir_conf") is None:
            pos["pred_mu_dir_conf"] = 0.0
        if pos.get("pred_mu_dir_prob_long") is None:
            pos["pred_mu_dir_prob_long"] = 0.5
        if pos.get("entry_quality_score") is None:
            pos["entry_quality_score"] = 0.0
        if pos.get("one_way_move_score") is None:
            pos["one_way_move_score"] = 0.0
        if pos.get("leverage_signal_score") is None:
            pos["leverage_signal_score"] = 0.0
        if pos.get("policy_score_threshold") is None:
            try:
                pos["policy_score_threshold"] = float(getattr(mc_config, "score_exit_threshold", 0.0) or 0.0)
            except Exception:
                pos["policy_score_threshold"] = 0.0
        if pos.get("policy_event_exit_min_score") is None:
            try:
                pos["policy_event_exit_min_score"] = float(getattr(config, "EVENT_EXIT_SCORE", -0.0005))
            except Exception:
                pos["policy_event_exit_min_score"] = -0.0005
        if pos.get("policy_unrealized_dd_floor") is None:
            try:
                pos["policy_unrealized_dd_floor"] = float(os.environ.get("UNREALIZED_DD_EXIT_ROE", -0.02) or -0.02)
            except Exception:
                pos["policy_unrealized_dd_floor"] = -0.02
        if not pos.get("entry_id") and pos.get("entry_link_id"):
            pos["entry_id"] = str(pos.get("entry_link_id"))
        if not pos.get("entry_link_id") and pos.get("entry_id"):
            pos["entry_link_id"] = str(pos.get("entry_id"))

        pos["obs_last_update_ms"] = int(now_ms())

    def _exit_side_sign(self, pos: dict | None, decision: dict | None) -> tuple[float, str]:
        side = ""
        if isinstance(pos, dict):
            side = str(pos.get("side") or "").upper()
            qty = self._safe_float(pos.get("quantity", pos.get("qty", 0.0)), 0.0) or 0.0
            if side in ("LONG", "SHORT") and abs(float(qty)) > 0:
                return (1.0 if side == "LONG" else -1.0), side
        if isinstance(decision, dict):
            side = str(decision.get("action") or "").upper()
            if side in ("LONG", "SHORT"):
                return (1.0 if side == "LONG" else -1.0), side
        return 0.0, ""

    def _build_dynamic_exit_policy(
        self,
        sym: str,
        pos: dict | None,
        decision: dict | None,
        *,
        ctx: dict | None = None,
    ) -> tuple[ExitPolicy, dict]:
        base = self.exit_policy
        policy = ExitPolicy(
            min_event_score=float(base.min_event_score),
            max_event_p_sl=float(base.max_event_p_sl),
            time_stop_mult=float(base.time_stop_mult),
            min_event_p_tp=float(base.min_event_p_tp),
            grace_sec=int(base.grace_sec),
            max_hold_sec=int(base.max_hold_sec),
            max_abs_event_cvar_r=float(base.max_abs_event_cvar_r),
        )

        meta = (decision.get("meta") or {}) if isinstance(decision, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        ctx = ctx or {}
        side_sign, side = self._exit_side_sign(pos, decision)
        entry_quality_score = self._safe_float(meta.get("entry_quality_score"), None)
        if entry_quality_score is None and isinstance(pos, dict):
            entry_quality_score = self._safe_float(pos.get("entry_quality_score"), None)
        if entry_quality_score is not None:
            entry_quality_score = float(max(0.0, min(1.0, entry_quality_score)))
        one_way_move_score = self._safe_float(meta.get("one_way_move_score"), None)
        if one_way_move_score is None and isinstance(pos, dict):
            one_way_move_score = self._safe_float(pos.get("one_way_move_score"), None)
        if one_way_move_score is not None:
            one_way_move_score = float(max(0.0, min(1.0, one_way_move_score)))

        hurst = self._safe_float(meta.get("hurst"), None)
        if hurst is None:
            hurst = self._safe_float(ctx.get("hurst"), None)
        if hurst is not None:
            hurst = float(max(0.0, min(1.0, hurst)))

        vpin = self._safe_float(meta.get("vpin"), None)
        if vpin is None:
            vpin = self._safe_float(ctx.get("vpin"), None)
        if vpin is not None:
            vpin = float(max(0.0, min(1.0, vpin)))

        mu_alpha = self._safe_float(meta.get("mu_adjusted"), None)
        if mu_alpha is None:
            mu_alpha = self._safe_float(meta.get("mu_alpha"), None)
        if mu_alpha is None:
            mu_alpha = self._safe_float(ctx.get("mu_alpha"), None)
        if mu_alpha is None:
            mu_alpha = self._safe_float(ctx.get("mu_base"), 0.0) or 0.0
        mu_alpha = float(mu_alpha)

        tick_vol = self._safe_float(meta.get("tick_vol"), None)
        if tick_vol is None:
            tick_vol = self._safe_float(ctx.get("tick_vol"), None)
        tick_trend = self._safe_float(meta.get("tick_trend"), None)
        if tick_trend is None:
            tick_trend = self._safe_float(ctx.get("tick_trend"), None)
        tick_breakout_active = bool(meta.get("tick_breakout_active", ctx.get("tick_breakout_active")))
        tick_breakout_dir = self._safe_float(meta.get("tick_breakout_dir"), None)
        if tick_breakout_dir is None:
            tick_breakout_dir = self._safe_float(ctx.get("tick_breakout_dir"), 0.0)
        tick_breakout_score = self._safe_float(meta.get("tick_breakout_score"), None)
        if tick_breakout_score is None:
            tick_breakout_score = self._safe_float(ctx.get("tick_breakout_score"), 0.0)
        hmm_state = meta.get("hmm_state")
        if hmm_state is None:
            hmm_state = ctx.get("hmm_state")
        hmm_conf = self._safe_float(meta.get("hmm_conf"), None)
        if hmm_conf is None:
            hmm_conf = self._safe_float(ctx.get("hmm_conf"), 0.0)
        hmm_sign = self._safe_float(meta.get("hmm_regime_sign"), None)
        if hmm_sign is None:
            hmm_sign = self._safe_float(ctx.get("hmm_regime_sign"), 0.0)

        try:
            h_low = float(getattr(mc_config, "hurst_low", 0.45))
            h_high = float(getattr(mc_config, "hurst_high", 0.55))
            vpin_extreme = float(getattr(mc_config, "vpin_extreme_threshold", 0.90))
            align_scale = max(float(getattr(mc_config, "exit_policy_mu_align_scale", 1.5) or 1.5), 1e-6)
        except Exception:
            h_low, h_high, vpin_extreme, align_scale = 0.45, 0.55, 0.90, 1.5

        dynamic_enabled = bool(getattr(mc_config, "exit_policy_dynamic_enabled", True))
        mode_tags = []
        regime_bucket = "random"
        if dynamic_enabled:
            score_mult = 1.0
            p_sl_add = 0.0
            cvar_mult = 1.0
            time_mult = 1.0
            p_tp_add = 0.0

            if hurst is not None:
                if hurst > h_high:
                    mode_tags.append("trend")
                    regime_bucket = "trend"
                    score_mult *= 1.20
                    p_sl_add += 0.05
                    cvar_mult *= 1.15
                    time_mult *= 1.20
                elif hurst < h_low:
                    mode_tags.append("mean_revert")
                    regime_bucket = "mean_revert"
                    score_mult *= 0.90
                    p_sl_add -= 0.04
                    cvar_mult *= 0.90
                    time_mult *= 0.90
                else:
                    mode_tags.append("random")
                    score_mult *= 0.95
                    p_sl_add -= 0.02
                    cvar_mult *= 0.95

            if vpin is not None:
                tox = float(max(0.0, vpin - 0.5) / 0.5)
                score_mult *= (1.0 - 0.18 * tox)
                p_sl_add -= 0.06 * tox
                cvar_mult *= (1.0 - 0.16 * tox)
                time_mult *= (1.0 - 0.12 * tox)
                p_tp_add += 0.04 * tox
                if vpin >= vpin_extreme:
                    if hurst is not None and hurst < h_low:
                        mode_tags.append("vpin_extreme_ou")
                        score_mult *= 1.20
                        p_sl_add += 0.08
                        cvar_mult *= 1.20
                        time_mult *= 1.15
                        p_tp_add -= 0.04
                    else:
                        mode_tags.append("vpin_extreme_risk")
                        score_mult *= 0.75
                        p_sl_add -= 0.10
                        cvar_mult *= 0.75
                        time_mult *= 0.80
                        p_tp_add += 0.08

            mu_align_raw = float(side_sign * mu_alpha) if side_sign != 0.0 else 0.0
            mu_align = math.tanh(mu_align_raw / align_scale)
            if mu_align >= 0.0:
                score_mult *= (1.0 + 0.35 * mu_align)
                p_sl_add += 0.08 * mu_align
                cvar_mult *= (1.0 + 0.25 * mu_align)
                time_mult *= (1.0 + 0.30 * mu_align)
                p_tp_add -= 0.04 * mu_align
            else:
                bad = abs(mu_align)
                score_mult *= max(0.50, 1.0 - 0.50 * bad)
                p_sl_add -= 0.12 * bad
                cvar_mult *= max(0.55, 1.0 - 0.45 * bad)
                time_mult *= max(0.60, 1.0 - 0.35 * bad)
                p_tp_add += 0.08 * bad

            # Respect high-quality entries in normal/noise states, but tighten low-quality entries.
            q_entry = 0.0 if entry_quality_score is None else float(entry_quality_score)
            q_oneway = 0.0 if one_way_move_score is None else float(one_way_move_score)
            q_combo = float(max(0.0, min(1.0, 0.65 * q_entry + 0.35 * q_oneway)))
            if q_combo >= 0.70:
                score_mult *= 1.08
                p_sl_add += 0.03
                cvar_mult *= 1.07
                time_mult *= 1.10
                p_tp_add -= 0.02
                mode_tags.append("entry_quality_hold")
            elif q_combo <= 0.35:
                score_mult *= 0.92
                p_sl_add -= 0.04
                cvar_mult *= 0.90
                time_mult *= 0.88
                p_tp_add += 0.03
                mode_tags.append("entry_quality_weak")

            # Shock-vs-noise split: tighten exits only on adverse shock, loosen in noisy shakeouts.
            shock_score = 0.0
            noise_mode = False
            try:
                shock_enabled = bool(getattr(mc_config, "exit_policy_shock_enabled", True))
                shock_breakout = float(getattr(mc_config, "exit_policy_shock_breakout_score", 0.0008))
                shock_trend_z = float(getattr(mc_config, "exit_policy_shock_trend_z", 2.0))
                noise_trend_z = float(getattr(mc_config, "exit_policy_noise_trend_z", 1.0))
                noise_vpin_cap = float(getattr(mc_config, "exit_policy_noise_vpin_cap", 0.80))
            except Exception:
                shock_enabled, shock_breakout, shock_trend_z, noise_trend_z, noise_vpin_cap = True, 0.0008, 2.0, 1.0, 0.80

            if shock_enabled:
                if side_sign != 0.0 and tick_breakout_active and tick_breakout_dir is not None:
                    against_breakout = float(tick_breakout_dir) * float(side_sign) < 0.0
                    if against_breakout and abs(float(tick_breakout_score or 0.0)) >= shock_breakout:
                        shock_score += min(2.0, 0.9 + abs(float(tick_breakout_score or 0.0)) / max(shock_breakout, 1e-8))
                    elif (not against_breakout) and abs(float(tick_breakout_score or 0.0)) >= shock_breakout:
                        shock_score -= 0.3

                if side_sign != 0.0 and tick_trend is not None:
                    trend_align = float(side_sign) * float(tick_trend)
                    if trend_align <= -abs(shock_trend_z):
                        shock_score += min(1.5, abs(trend_align) / max(abs(shock_trend_z), 1e-8))
                    elif trend_align >= abs(shock_trend_z):
                        shock_score -= 0.2

                if side_sign != 0.0 and hmm_conf is not None and float(hmm_conf) > 0.0:
                    hmm_align = float(side_sign) * (1.0 if (hmm_sign or 0.0) > 0 else (-1.0 if (hmm_sign or 0.0) < 0 else 0.0))
                    if hmm_align < 0:
                        shock_score += 0.8 * float(hmm_conf)
                    elif hmm_align > 0:
                        shock_score -= 0.4 * float(hmm_conf)

                if tick_breakout_active is False and tick_trend is not None and abs(float(tick_trend)) <= abs(noise_trend_z):
                    if (vpin is None) or (float(vpin) <= float(noise_vpin_cap)):
                        noise_mode = True

                if shock_score >= 1.0:
                    mode_tags.append("shock_guard")
                    score_mult *= max(0.45, 1.0 - 0.22 * shock_score)
                    p_sl_add -= min(0.18, 0.06 * shock_score)
                    cvar_mult *= max(0.55, 1.0 - 0.20 * shock_score)
                    time_mult *= max(0.55, 1.0 - 0.18 * shock_score)
                    p_tp_add += min(0.12, 0.04 * shock_score)
                elif noise_mode:
                    mode_tags.append("noise_hold")
                    score_mult *= 1.08
                    p_sl_add += 0.04
                    cvar_mult *= 1.08
                    time_mult *= 1.12
                    p_tp_add -= 0.03
            else:
                shock_score = 0.0
                noise_mode = False

            policy.min_event_score = float(policy.min_event_score * score_mult)
            # Regime-specific min-score offset (negative => fewer early exits, positive => faster exits).
            def _env_float_chain(keys: tuple[str, ...], default: float) -> float:
                for k in keys:
                    try:
                        v = os.environ.get(k)
                        if v is None or str(v).strip() == "":
                            continue
                        return float(v)
                    except Exception:
                        continue
                return float(default)
            score_off_trend = _env_float_chain(("EVENT_EXIT_SCORE_OFFSET_TREND",), -0.0012)
            score_off_mean_revert = _env_float_chain(
                ("EVENT_EXIT_SCORE_OFFSET_MEAN_REVERT", "EVENT_EXIT_SCORE_OFFSET_CHOP"),
                -0.0008,
            )
            score_off_random = _env_float_chain(
                ("EVENT_EXIT_SCORE_OFFSET_RANDOM", "EVENT_EXIT_SCORE_OFFSET_VOLATILE"),
                -0.0003,
            )
            try:
                score_off_vpin_ou = float(os.environ.get("EVENT_EXIT_SCORE_OFFSET_VPIN_EXTREME_OU", -0.0010) or -0.0010)
            except Exception:
                score_off_vpin_ou = -0.0010
            try:
                score_off_vpin_risk = float(os.environ.get("EVENT_EXIT_SCORE_OFFSET_VPIN_EXTREME_RISK", 0.0010) or 0.0010)
            except Exception:
                score_off_vpin_risk = 0.0010
            score_offset = score_off_random
            if regime_bucket == "trend":
                score_offset = score_off_trend
            elif regime_bucket == "mean_revert":
                score_offset = score_off_mean_revert
            if vpin is not None and vpin >= vpin_extreme:
                if regime_bucket == "mean_revert":
                    score_offset += score_off_vpin_ou
                else:
                    score_offset += score_off_vpin_risk
            policy.min_event_score = float(policy.min_event_score + score_offset)
            policy.max_event_p_sl = float(policy.max_event_p_sl + p_sl_add)
            policy.max_abs_event_cvar_r = float(policy.max_abs_event_cvar_r * cvar_mult)
            policy.time_stop_mult = float(policy.time_stop_mult * time_mult)
            policy.min_event_p_tp = float(policy.min_event_p_tp + p_tp_add)
        else:
            shock_score = 0.0
            noise_mode = False
            score_offset = 0.0

        policy.min_event_score = float(np.clip(policy.min_event_score, -0.05, 0.01))
        policy.max_event_p_sl = float(np.clip(policy.max_event_p_sl, 0.25, 0.98))
        policy.max_abs_event_cvar_r = float(np.clip(policy.max_abs_event_cvar_r, 0.002, 0.20))
        policy.time_stop_mult = float(np.clip(policy.time_stop_mult, 1.0, 6.0))
        policy.min_event_p_tp = float(np.clip(policy.min_event_p_tp, 0.10, 0.80))

        mu_align_raw = float(side_sign * mu_alpha) if side_sign != 0.0 else 0.0
        try:
            shock_fast_threshold = float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD", 1.0) or 1.0)
        except Exception:
            shock_fast_threshold = 1.0
        shock_bucket = "shock" if float(shock_score) >= float(shock_fast_threshold) else ("noise" if noise_mode else "normal")
        diag = {
            "symbol": str(sym),
            "side": side,
            "dynamic_enabled": bool(dynamic_enabled),
            "regime_bucket": str(regime_bucket),
            "mode": ",".join(mode_tags) if mode_tags else ("dynamic" if dynamic_enabled else "static"),
            "shock_bucket": str(shock_bucket),
            "hurst": hurst,
            "vpin": vpin,
            "mu_alpha": float(mu_alpha),
            "mu_alignment": float(mu_align_raw),
            "entry_quality_score": entry_quality_score,
            "one_way_move_score": one_way_move_score,
            "tick_vol": tick_vol,
            "tick_trend": tick_trend,
            "tick_breakout_active": bool(tick_breakout_active),
            "tick_breakout_dir": float(tick_breakout_dir or 0.0),
            "tick_breakout_score": float(tick_breakout_score or 0.0),
            "shock_score": float(shock_score),
            "noise_mode": bool(noise_mode),
            "hmm_state": hmm_state,
            "hmm_conf": float(hmm_conf or 0.0),
            "hmm_sign": float(hmm_sign or 0.0),
            "min_event_score": float(policy.min_event_score),
            "min_event_score_offset": float(score_offset),
            "max_event_p_sl": float(policy.max_event_p_sl),
            "max_abs_event_cvar_r": float(policy.max_abs_event_cvar_r),
            "time_stop_mult": float(policy.time_stop_mult),
            "min_event_p_tp": float(policy.min_event_p_tp),
        }
        return policy, diag

    def _attach_dynamic_exit_meta(self, meta: dict, policy: ExitPolicy, diag: dict) -> None:
        if not isinstance(meta, dict):
            return
        meta["event_exit_dynamic_enabled"] = bool(diag.get("dynamic_enabled"))
        meta["event_exit_dynamic_regime_bucket"] = str(diag.get("regime_bucket") or "")
        meta["event_exit_dynamic_mode"] = str(diag.get("mode") or "static")
        meta["event_exit_dynamic_hurst"] = diag.get("hurst")
        meta["event_exit_dynamic_vpin"] = diag.get("vpin")
        meta["event_exit_dynamic_mu_alpha"] = diag.get("mu_alpha")
        meta["event_exit_dynamic_mu_alignment"] = diag.get("mu_alignment")
        meta["entry_quality_score"] = diag.get("entry_quality_score")
        meta["one_way_move_score"] = diag.get("one_way_move_score")
        meta["event_exit_dynamic_tick_vol"] = diag.get("tick_vol")
        meta["event_exit_dynamic_tick_trend"] = diag.get("tick_trend")
        meta["event_exit_dynamic_tick_breakout_active"] = bool(diag.get("tick_breakout_active"))
        meta["event_exit_dynamic_tick_breakout_dir"] = diag.get("tick_breakout_dir")
        meta["event_exit_dynamic_tick_breakout_score"] = diag.get("tick_breakout_score")
        meta["event_exit_dynamic_shock_score"] = diag.get("shock_score")
        meta["event_exit_dynamic_shock_bucket"] = str(diag.get("shock_bucket") or "normal")
        meta["event_exit_dynamic_noise_mode"] = bool(diag.get("noise_mode"))
        meta["event_exit_dynamic_hmm_state"] = diag.get("hmm_state")
        meta["event_exit_dynamic_hmm_conf"] = diag.get("hmm_conf")
        meta["event_exit_dynamic_hmm_sign"] = diag.get("hmm_sign")
        meta["event_exit_min_score"] = float(policy.min_event_score)
        meta["event_exit_min_score_offset"] = float(diag.get("min_event_score_offset", 0.0) or 0.0)
        meta["event_exit_max_p_sl"] = float(policy.max_event_p_sl)
        meta["event_exit_max_cvar"] = float(policy.max_abs_event_cvar_r)
        meta["event_exit_time_stop_mult"] = float(policy.time_stop_mult)
        meta["event_exit_min_p_tp"] = float(policy.min_event_p_tp)

    def _sanitize_garch_params(self, raw: dict | None) -> dict[str, float] | None:
        if not isinstance(raw, dict):
            return None
        try:
            omega = float(raw.get("omega"))
            alpha = float(raw.get("alpha"))
            beta = float(raw.get("beta"))
        except Exception:
            return None
        if not np.isfinite(omega) or not np.isfinite(alpha) or not np.isfinite(beta):
            return None
        omega = max(float(omega), 1e-12)
        alpha = min(max(float(alpha), 1e-6), 0.999)
        beta = min(max(float(beta), 1e-6), 0.999)
        ab_sum = alpha + beta
        if ab_sum >= 0.999:
            scale = 0.999 / max(ab_sum, 1e-9)
            alpha *= scale
            beta *= scale
        out = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta)}
        var0 = self._safe_float(raw.get("var0"), None)
        if var0 is None:
            var0 = self._safe_float(raw.get("var"), None)
        if var0 is not None and float(var0) > 0:
            out["var"] = max(float(var0), 1e-10)
        return out

    def _maybe_reload_causal_weights(self, ts_ms: int) -> None:
        path = str(getattr(mc_config, "causal_weights_path", "") or "").strip()
        if not path:
            return
        try:
            reload_sec = float(getattr(mc_config, "causal_reload_sec", self._alpha_causal_reload_sec) or self._alpha_causal_reload_sec)
        except Exception:
            reload_sec = float(self._alpha_causal_reload_sec)
        interval_ms = int(max(1.0, reload_sec) * 1000.0)
        if ts_ms - int(self._alpha_causal_last_reload_ms or 0) < interval_ms:
            return
        self._alpha_causal_last_reload_ms = int(ts_ms)
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            return
        prev_mtime = self._alpha_causal_weights_mtime
        if prev_mtime is not None and mtime <= float(prev_mtime):
            return
        weights = load_causal_weights(path)
        if not weights:
            return
        self._alpha_causal_weights = weights
        self._alpha_causal_weights_mtime = mtime
        try:
            self._log(f"[ALPHA_CAUSAL] reloaded weights: path={path} n={len(weights)}")
        except Exception:
            pass

    def _maybe_reload_mlofi_weights(self, ts_ms: int) -> None:
        path = str(getattr(mc_config, "mlofi_weight_path", "") or "").strip()
        if not path:
            return
        try:
            reload_sec = float(getattr(mc_config, "mlofi_reload_sec", self._alpha_mlofi_reload_sec) or self._alpha_mlofi_reload_sec)
        except Exception:
            reload_sec = float(self._alpha_mlofi_reload_sec)
        interval_ms = int(max(1.0, reload_sec) * 1000.0)
        if ts_ms - int(self._alpha_mlofi_last_reload_ms or 0) < interval_ms:
            return
        self._alpha_mlofi_last_reload_ms = int(ts_ms)
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            return
        prev_mtime = self._alpha_mlofi_weights_mtime
        if prev_mtime is not None and mtime <= float(prev_mtime):
            return
        weights = load_weight_vector(path)
        if weights is None:
            return
        try:
            arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception:
            return
        if arr.size == 0 or (not np.all(np.isfinite(arr))):
            return
        self._alpha_mlofi_weights = arr
        self._alpha_mlofi_weights_mtime = mtime
        try:
            self._log(f"[ALPHA_MLOFI] reloaded weights: path={path} n={int(arr.size)}")
        except Exception:
            pass

    def _maybe_reload_direction_model(self, ts_ms: int) -> None:
        path = str(self._alpha_dir_model_path or getattr(mc_config, "alpha_direction_model_path", "") or "").strip()
        if not path:
            return
        try:
            reload_sec = float(getattr(mc_config, "alpha_direction_reload_sec", self._alpha_dir_reload_sec) or self._alpha_dir_reload_sec)
        except Exception:
            reload_sec = float(self._alpha_dir_reload_sec)
        interval_ms = int(max(1.0, reload_sec) * 1000.0)
        if ts_ms - int(self._alpha_dir_last_reload_ms or 0) < interval_ms:
            return
        self._alpha_dir_last_reload_ms = int(ts_ms)
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            return
        prev_mtime = self._alpha_dir_model_mtime
        if prev_mtime is not None and mtime <= float(prev_mtime):
            return
        model = load_direction_model(path)
        if not model:
            return
        self._alpha_dir_model = model
        self._alpha_dir_model_mtime = mtime
        try:
            n_feat = len(model.get("feature_names") or [])
            self._log(f"[ALPHA_DIR] reloaded direction model: path={path} n_feat={n_feat}")
        except Exception:
            pass

    def _maybe_reload_garch_params(self, ts_ms: int) -> None:
        path = str(self._alpha_garch_path or getattr(mc_config, "garch_param_path", "") or "").strip()
        if not path:
            return
        try:
            reload_sec = float(getattr(mc_config, "garch_param_reload_sec", self._alpha_garch_reload_sec) or self._alpha_garch_reload_sec)
        except Exception:
            reload_sec = float(self._alpha_garch_reload_sec)
        interval_ms = int(max(1.0, reload_sec) * 1000.0)
        if ts_ms - int(self._alpha_garch_last_reload_ms or 0) < interval_ms:
            return
        self._alpha_garch_last_reload_ms = int(ts_ms)
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            return
        prev_mtime = self._alpha_garch_mtime
        if prev_mtime is not None and mtime <= float(prev_mtime):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        payload = data if isinstance(data, dict) else {}
        global_raw = None
        if isinstance(payload.get("garch"), dict):
            global_raw = payload.get("garch") or {}
        elif isinstance(payload, dict):
            global_raw = payload
        override = self._sanitize_garch_params(global_raw)
        if not override:
            return

        sym_raw = payload.get("symbols") if isinstance(payload.get("symbols"), dict) else {}
        sym_overrides: dict[str, dict[str, float]] = {}
        if isinstance(sym_raw, dict):
            for k, v in sym_raw.items():
                if not isinstance(v, dict):
                    continue
                sv = self._sanitize_garch_params(v)
                if sv:
                    sym_overrides[str(k)] = sv

        self._alpha_garch_override = dict(override)
        self._alpha_garch_symbol_overrides = sym_overrides
        self._alpha_garch_mtime = mtime
        for sym_key, st in self._alpha_state.items():
            g = st.get("garch")
            if isinstance(g, GARCHState):
                sovr = self._alpha_garch_symbol_overrides.get(sym_key) or override
                g.omega = float(sovr["omega"])
                g.alpha = float(sovr["alpha"])
                g.beta = float(sovr["beta"])
                if sovr.get("var") is not None:
                    g.var = max(float(sovr["var"]), 1e-10)
        try:
            var_text = f" var0={override['var']:.2e}" if override.get("var") is not None else ""
            self._log(
                f"[ALPHA_GARCH] reloaded params: path={path} "
                f"omega={override['omega']:.2e} alpha={override['alpha']:.4f} beta={override['beta']:.4f}"
                f"{var_text} symbols={len(sym_overrides)}"
            )
        except Exception:
            pass

    def _maybe_load_alpha_ml(self) -> None:
        if self._alpha_ml_ready:
            return
        self._alpha_ml_ready = True
        path = str(getattr(mc_config, "ml_model_path", "") or "").strip()
        if not path or not os.path.exists(path):
            # optional auto-build when model missing
            if str(os.environ.get("ML_AUTO_BUILD", "0")).strip().lower() in ("1", "true", "yes", "on"):
                try:
                    from scripts.build_mu_alpha_model import build_mu_alpha_model
                    build_mu_alpha_model(
                        out_path=path or "state/mu_alpha_model.pt",
                        seq_len=int(os.environ.get("ML_SEQ_LEN", 64) or 64),
                        hidden=int(os.environ.get("ML_HIDDEN", 32) or 32),
                        layers=int(os.environ.get("ML_LAYERS", 2) or 2),
                        epochs=int(os.environ.get("ML_EPOCHS", 3) or 3),
                        batch_size=int(os.environ.get("ML_BATCH", 128) or 128),
                        lr=float(os.environ.get("ML_LR", 1e-3) or 1e-3),
                        dropout=float(os.environ.get("ML_DROPOUT", 0.1) or 0.1),
                        max_samples=int(os.environ.get("ML_MAX_SAMPLES", 200000) or 200000),
                    )
                except Exception:
                    return
            if not path or not os.path.exists(path):
                return
        try:
            import torch  # lazy import
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            try:
                model = torch.jit.load(path, map_location=device)
            except Exception:
                model = torch.load(path, map_location=device)
            if hasattr(model, "eval"):
                model.eval()
            self._alpha_ml_model = model
        except Exception:
            self._alpha_ml_model = None

    def _predict_mu_ml(self, closes: list[float], volumes: list[float] | None = None) -> float:
        """
        ML model spec:
        - Input: normalized feature sequence, shape (1, seq_len, n_features)
          - feature[0]: log return
          - feature[1]: log volume change (optional)
          - feature[2]: abs return (optional)
        - Output: per-bar expected log-return (mu_bar), scalar
        - Runtime annualizes with bar_seconds=60
        """
        if not bool(getattr(mc_config, "alpha_use_ml", False)):
            return 0.0
        self._maybe_load_alpha_ml()
        if self._alpha_ml_model is None:
            return 0.0
        if closes is None or len(closes) < 8:
            return 0.0
        try:
            import torch
            x = np.asarray(closes, dtype=np.float32)
            if x.size < 8:
                return 0.0
            rets = np.diff(np.log(np.maximum(x, 1e-12)))
            if rets.size < 4:
                return 0.0
            model = self._alpha_ml_model
            n_features = int(getattr(model, "n_features", 1) or 1)
            seq_len = int(getattr(model, "seq_len", int(os.environ.get("ML_SEQ_LEN", 64) or 64)))
            feats = rets
            if n_features > 1:
                if volumes is not None and len(volumes) >= len(closes):
                    v = np.asarray(volumes, dtype=np.float32)
                    v = v[-x.size:]
                    v = np.where(np.isfinite(v), v, 0.0)
                    v = np.maximum(v, 1e-12)
                    v_rets = np.diff(np.log(v))
                else:
                    v_rets = np.zeros_like(rets)
                abs_rets = np.abs(rets)
                feats = np.stack([rets, v_rets, abs_rets], axis=1)
                if n_features < feats.shape[1]:
                    feats = feats[:, :n_features]
            if feats.ndim == 1:
                feats = feats[:, None]
            if seq_len > 0 and feats.shape[0] > seq_len:
                feats = feats[-seq_len:]
            if feats.shape[0] < 4:
                return 0.0
            mean = feats.mean(axis=0, keepdims=True)
            std = feats.std(axis=0, keepdims=True) + 1e-6
            feats = (feats - mean) / std
            seq = torch.tensor(feats, dtype=torch.float32).view(1, feats.shape[0], feats.shape[1])
            with torch.no_grad():
                out = model(seq)
            # accept scalar or 1x1
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            out = np.asarray(out).reshape(-1)
            if out.size == 0:
                return 0.0
            mu_bar = float(out[0])
            return float(_annualize_mu(mu_bar, 60.0))
        except Exception:
            return 0.0

    def _update_alpha_state(
        self,
        sym: str,
        ts: int,
        price: float,
        closes: list[float],
        volumes: list[float],
        *,
        fast_mode: bool = False,
    ) -> dict:
        st = self._get_alpha_state(sym)
        out: dict = {}
        ofi_score = float(self._compute_ofi_score(sym))
        try:
            reload_interval_ms = int(max(0.2, float(self._alpha_reload_check_interval_sec)) * 1000.0)
        except Exception:
            reload_interval_ms = 1000
        last_reload_check = int(self._alpha_reload_check_last_ms or 0)
        if last_reload_check <= 0 or (int(ts) - last_reload_check) >= reload_interval_ms:
            self._alpha_reload_check_last_ms = int(ts)
            self._maybe_reload_mlofi_weights(int(ts))
            self._maybe_reload_causal_weights(int(ts))
            self._maybe_reload_direction_model(int(ts))
            self._maybe_reload_garch_params(int(ts))

        if fast_mode:
            alpha_cache = st.get("alpha_cache")
            if isinstance(alpha_cache, dict) and alpha_cache:
                out.update(alpha_cache)
                out["alpha_fast_mode"] = True
                return out

        # MLOFI from orderbook delta
        if bool(getattr(mc_config, "alpha_use_mlofi", False)):
            ob = self.orderbook.get(sym) or {}
            prev_ob = st.get("mlofi_prev")
            levels = int(getattr(mc_config, "mlofi_levels", 5))
            weights = self._alpha_mlofi_weights
            # If learned weights collapse near zero, fall back to exponential profile.
            if weights is not None:
                try:
                    w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
                    if w_arr.size == 0 or (not np.all(np.isfinite(w_arr))) or float(np.sum(np.abs(w_arr))) < 1e-6:
                        weights = None
                except Exception:
                    weights = None
            if weights is None:
                lam = float(getattr(mc_config, "mlofi_decay_lambda", 0.4))
                weights = np.exp(-lam * np.arange(levels, dtype=np.float64))
            mlofi_val, mlofi_vec = compute_mlofi(prev_ob, ob, levels=levels, weights=weights)
            out["mlofi"] = float(mlofi_val)
            out["mlofi_vec"] = [float(x) for x in np.asarray(mlofi_vec).tolist()]
            # store shallow copy of top levels
            st["mlofi_prev"] = {
                "bids": (ob.get("bids") or [])[:levels],
                "asks": (ob.get("asks") or [])[:levels],
            }

        # Kalman filter (update per tick)
        if bool(getattr(mc_config, "alpha_use_kf", False)):
            last_ts = st.get("last_ts")
            dt = 1.0
            if last_ts is not None:
                dt = max(1e-3, (float(ts) - float(last_ts)) / 1000.0)
            kf_state = st.get("kf")
            kf_state.Q = float(getattr(mc_config, "kf_q", kf_state.Q))
            kf_state.R = float(getattr(mc_config, "kf_r", kf_state.R))
            kf_state, vel = update_kalman_cv(kf_state, float(price), dt)
            st["kf"] = kf_state
            if price > 0:
                mu_bar = (float(vel) * 60.0) / float(price)  # per-bar log-return approx
                out["mu_kf"] = float(_annualize_mu(mu_bar, 60.0))
            st["last_ts"] = ts

        # Hawkes (event from price change)
        if bool(getattr(mc_config, "alpha_use_hawkes", False)):
            last_price = st.get("last_price")
            event = 0
            if last_price is not None:
                if float(price) > float(last_price):
                    event = 1
                elif float(price) < float(last_price):
                    event = -1
            dt = 1.0
            if st.get("last_ts") is not None:
                dt = max(1e-3, (float(ts) - float(st.get("last_ts"))) / 1000.0)
            hawkes_state = st.get("hawkes")
            hawkes_state.alpha = float(getattr(mc_config, "hawkes_alpha", hawkes_state.alpha))
            hawkes_state.beta = float(getattr(mc_config, "hawkes_beta", hawkes_state.beta))
            hawkes_state, boost = update_hawkes_state(hawkes_state, event, dt)
            st["hawkes"] = hawkes_state
            out["hawkes_lambda_buy"] = float(hawkes_state.lambda_buy)
            out["hawkes_lambda_sell"] = float(hawkes_state.lambda_sell)
            out["hawkes_boost"] = float(boost)
            st["last_price"] = float(price)
            st["last_ts"] = ts

        # Per-candle updates (use kline timestamp)
        kline_ts = int(self._last_kline_ts.get(sym, 0) or 0)
        if kline_ts and kline_ts != int(st.get("last_kline_ts") or 0):
            st["last_kline_ts"] = kline_ts
            if closes is not None and len(closes) >= 2:
                ret = float(math.log(max(closes[-1], 1e-12) / max(closes[-2], 1e-12)))
                # training samples (one-step ahead)
                if str(os.environ.get("ALPHA_WEIGHT_COLLECT", "0")).strip().lower() in ("1", "true", "yes", "on"):
                    prev_vec = st.get("mlofi_last_vec")
                    if prev_vec is not None:
                        try:
                            self._alpha_train_mlofi.append((np.asarray(prev_vec, dtype=np.float64), float(ret)))
                        except Exception:
                            pass
                    prev_causal = st.get("causal_last_vec")
                    if prev_causal is not None:
                        try:
                            prev_regime = str(st.get("causal_last_regime") or "chop")
                            self._alpha_train_causal.append(
                                (
                                    np.asarray(prev_causal, dtype=np.float64),
                                    float(ret),
                                    prev_regime,
                                    str(sym),
                                    int(kline_ts),
                                )
                            )
                        except Exception:
                            pass
                # Hurst series
                if bool(getattr(mc_config, "alpha_use_hurst", False)):
                    st["hurst_rets"].append(ret)
                    try:
                        hurst_update_sec = float(getattr(mc_config, "alpha_hurst_update_sec", 60.0) or 60.0)
                    except Exception:
                        hurst_update_sec = 60.0
                    hurst_interval_ms = int(max(0.0, hurst_update_sec) * 1000.0)
                    last_hurst_ts = int(st.get("last_hurst_ts") or 0)
                    should_update_hurst = (
                        hurst_interval_ms <= 0
                        or last_hurst_ts <= 0
                        or (int(kline_ts) - last_hurst_ts) >= hurst_interval_ms
                    )
                    if should_update_hurst:
                        taus = self._parse_hurst_taus()
                        h = estimate_hurst_vr(st["hurst_rets"], taus)
                        if h is not None:
                            out["hurst"] = float(h)
                            st["hurst_last"] = float(h)
                        st["last_hurst_ts"] = int(kline_ts)
                    else:
                        out["hurst"] = float(st.get("hurst_last") or 0.0)

                # GARCH volatility
                if bool(getattr(mc_config, "alpha_use_garch", False)):
                    try:
                        garch_update_sec = float(getattr(mc_config, "alpha_garch_update_sec", 60.0) or 60.0)
                    except Exception:
                        garch_update_sec = 60.0
                    garch_interval_ms = int(max(0.0, garch_update_sec) * 1000.0)
                    last_garch_ts = int(st.get("last_garch_ts") or 0)
                    should_update_garch = (
                        garch_interval_ms <= 0
                        or last_garch_ts <= 0
                        or (int(kline_ts) - last_garch_ts) >= garch_interval_ms
                    )
                    if should_update_garch:
                        g_state = st.get("garch")
                        g_ovr = self._alpha_garch_symbol_overrides.get(sym) or self._alpha_garch_override or {}
                        g_state.omega = float(g_ovr.get("omega", getattr(mc_config, "garch_omega", g_state.omega)))
                        g_state.alpha = float(g_ovr.get("alpha", getattr(mc_config, "garch_alpha", g_state.alpha)))
                        g_state.beta = float(g_ovr.get("beta", getattr(mc_config, "garch_beta", g_state.beta)))
                        if g_ovr.get("var") is not None and float(g_state.var) <= 1e-10:
                            g_state.var = max(float(g_ovr.get("var")), 1e-10)
                        if float(g_state.var) <= 1e-10 and closes is not None and len(closes) >= 8:
                            seed = np.asarray(closes[-min(len(closes), 120) :], dtype=np.float64)
                            seed_rets = np.diff(np.log(np.maximum(seed, 1e-12)))
                            if seed_rets.size >= 2:
                                g_state.var = max(float(np.var(seed_rets)), 1e-10)
                        g_state, sigma_bar = update_garch_state(g_state, ret)
                        st["garch"] = g_state
                        st["sigma_garch_last"] = float(_annualize_sigma(float(sigma_bar), 60.0))
                        st["last_garch_ts"] = int(kline_ts)
                    out["sigma_garch"] = float(st.get("sigma_garch_last") or 0.0)

                # Bayesian mean update
                if bool(getattr(mc_config, "alpha_use_bayes", False)):
                    b_state = st.get("bayes")
                    obs_var = float(getattr(mc_config, "bayes_obs_var", 1e-4))
                    if out.get("sigma_garch") is not None:
                        # use garch variance proxy if available (de-annualize)
                        sigma_bar = float(out["sigma_garch"]) / math.sqrt(31536000.0 / 60.0)
                        obs_var = max(obs_var, sigma_bar * sigma_bar)
                    b_state, mu_mean, mu_var = update_bayes_mean(b_state, ret, obs_var)
                    st["bayes"] = b_state
                    out["mu_bayes"] = float(_annualize_mu(float(mu_mean), 60.0))
                    out["mu_bayes_var"] = float(mu_var)

                # AR(1) forecast
                if bool(getattr(mc_config, "alpha_use_arima", False)):
                    rets = list(st.get("hurst_rets") or [])
                    mu_ar = estimate_ar1_next_return(rets[-120:] if len(rets) > 0 else rets)
                    out["mu_ar"] = float(_annualize_mu(float(mu_ar), 60.0))

                # Particle filter
                if bool(getattr(mc_config, "alpha_use_pf", False)):
                    try:
                        pf_update_sec = float(getattr(mc_config, "alpha_pf_update_sec", 60.0) or 60.0)
                    except Exception:
                        pf_update_sec = 60.0
                    pf_interval_ms = int(max(0.0, pf_update_sec) * 1000.0)
                    last_pf_ts = int(st.get("last_pf_ts") or 0)
                    should_update_pf = (
                        pf_interval_ms <= 0
                        or last_pf_ts <= 0
                        or (int(kline_ts) - last_pf_ts) >= pf_interval_ms
                    )
                    if should_update_pf:
                        pf_state = st.get("pf")
                        pf_state, mu_pf = update_particle_filter(pf_state, ret)
                        st["pf"] = pf_state
                        st["mu_pf_last"] = float(_annualize_mu(float(mu_pf), 60.0))
                        st["last_pf_ts"] = int(kline_ts)
                    out["mu_pf"] = float(st.get("mu_pf_last") or 0.0)

                # VPIN update (use candle volume)
                if bool(getattr(mc_config, "alpha_use_vpin", False)):
                    try:
                        vpin_update_sec = float(getattr(mc_config, "alpha_vpin_update_sec", 60.0) or 60.0)
                    except Exception:
                        vpin_update_sec = 60.0
                    vpin_interval_ms = int(max(0.0, vpin_update_sec) * 1000.0)
                    last_vpin_ts = int(st.get("last_vpin_ts") or 0)
                    should_update_vpin = (
                        vpin_interval_ms <= 0
                        or last_vpin_ts <= 0
                        or (int(kline_ts) - last_vpin_ts) >= vpin_interval_ms
                    )
                    if should_update_vpin:
                        vpin_state = st.get("vpin")
                        if vpin_state.bucket_size <= 0 and volumes is not None and len(volumes) >= 2:
                            total_vol = float(np.sum(np.asarray(volumes[-int(getattr(mc_config, "vpin_bucket_count", 50)) :], dtype=np.float64)))
                            bucket_count = int(getattr(mc_config, "vpin_bucket_count", 50))
                            vpin_state.bucket_size = max(total_vol / max(bucket_count, 1), 1e-8)
                        # sigma from recent returns
                        rets = list(st.get("hurst_rets") or [])
                        sigma = float(np.std(rets[-60:])) if len(rets) >= 5 else 1e-6
                        dp = float(closes[-1] - closes[-2])
                        vol = float(volumes[-1]) if volumes else 0.0
                        vpin_val = update_vpin_state(vpin_state, dp, vol, sigma)
                        st["vpin"] = vpin_state
                        st["vpin_last"] = float(vpin_val)
                        st["last_vpin_ts"] = int(kline_ts)
                    out["vpin"] = float(st.get("vpin_last") or 0.0)

                # Hidden Markov Model regime (online Gaussian filter)
                if bool(getattr(mc_config, "alpha_use_hmm", False)):
                    try:
                        hmm_update_sec = float(getattr(mc_config, "alpha_hmm_update_sec", 60.0) or 60.0)
                    except Exception:
                        hmm_update_sec = 60.0
                    hmm_interval_ms = int(max(0.0, hmm_update_sec) * 1000.0)
                    last_hmm_ts = int(st.get("last_hmm_ts") or 0)
                    should_update_hmm = (
                        hmm_interval_ms <= 0
                        or last_hmm_ts <= 0
                        or (int(kline_ts) - last_hmm_ts) >= hmm_interval_ms
                    )
                    if should_update_hmm:
                        hmm_state = st.get("hmm")
                        try:
                            hmm_state.n_states = int(getattr(mc_config, "hmm_states", hmm_state.n_states))
                        except Exception:
                            pass
                        try:
                            hmm_state.adapt_lr = float(getattr(mc_config, "hmm_adapt_lr", hmm_state.adapt_lr))
                        except Exception:
                            pass
                        hmm_state, hmm_info = update_gaussian_hmm(hmm_state, ret)
                        st["hmm"] = hmm_state
                        st["last_hmm_ts"] = int(kline_ts)
                        st["hmm_last"] = {
                            "state_idx": int(hmm_info.get("state_idx", 0)),
                            "state_label": str(hmm_info.get("state_label", "chop")),
                            "confidence": float(hmm_info.get("confidence", 0.0) or 0.0),
                            "regime_sign": int(hmm_info.get("regime_sign", 0) or 0),
                            "means": [float(x) for x in (hmm_info.get("means") or [])],
                        }
                    hmm_last = st.get("hmm_last") or {}
                    out["hmm_state_idx"] = int(hmm_last.get("state_idx", 0))
                    out["hmm_state"] = str(hmm_last.get("state_label", "chop"))
                    out["hmm_conf"] = float(hmm_last.get("confidence", 0.0) or 0.0)
                    out["hmm_regime_sign"] = int(hmm_last.get("regime_sign", 0) or 0)
                    out["hmm_means"] = [float(x) for x in (hmm_last.get("means") or [])]

                # update training feature caches
                if str(os.environ.get("ALPHA_WEIGHT_COLLECT", "0")).strip().lower() in ("1", "true", "yes", "on"):
                    if "mlofi_vec" in out and out.get("mlofi_vec") is not None:
                        st["mlofi_last_vec"] = np.asarray(out.get("mlofi_vec"), dtype=np.float64)
                    feat_vals = [
                        float(ofi_score),
                        float(out.get("mlofi", 0.0) or 0.0),
                        float(out.get("vpin", 0.0) or 0.0),
                        float(out.get("mu_kf", 0.0) or 0.0),
                        float(out.get("hurst", 0.0) or 0.0),
                        float(out.get("mu_ml", 0.0) or 0.0),
                    ]
                    st["causal_last_vec"] = np.asarray(feat_vals, dtype=np.float64)

        # OU drift (mean reversion)
        if bool(getattr(mc_config, "alpha_use_hurst", False)):
            win = int(getattr(mc_config, "ou_mean_window", 60))
            if closes is not None and len(closes) >= 2 and win >= 2:
                mean_price = float(np.mean(np.asarray(closes[-win:], dtype=np.float64)))
                theta = float(getattr(mc_config, "ou_theta", 0.3))
                mu_ou = compute_ou_drift(float(price), mean_price, theta)
                if price > 0:
                    mu_bar = mu_ou / float(price)
                    out["mu_ou"] = float(_annualize_mu(mu_bar, 60.0))

        # ML prediction (optional)
        if bool(getattr(mc_config, "alpha_use_ml", False)):
            try:
                ml_update_sec = float(getattr(mc_config, "alpha_ml_update_sec", 5.0) or 5.0)
            except Exception:
                ml_update_sec = 5.0
            ml_interval_ms = int(max(0.0, ml_update_sec) * 1000.0)
            last_ml_ts = int(st.get("last_ml_ts") or 0)
            should_update_ml = (
                ml_interval_ms <= 0
                or last_ml_ts <= 0
                or (int(ts) - last_ml_ts) >= ml_interval_ms
            )
            if should_update_ml:
                mu_ml = self._predict_mu_ml(closes, volumes)
                st["mu_ml_last"] = float(mu_ml)
                st["last_ml_ts"] = int(ts)
            out["mu_ml"] = float(st.get("mu_ml_last") or 0.0)

        # Refresh causal feature cache after all components computed
        if str(os.environ.get("ALPHA_WEIGHT_COLLECT", "0")).strip().lower() in ("1", "true", "yes", "on"):
            feat_vals = [
                float(ofi_score),
                float(out.get("mlofi", 0.0) or 0.0),
                float(out.get("vpin", 0.0) or 0.0),
                float(out.get("mu_kf", 0.0) or 0.0),
                float(out.get("hurst", 0.0) or 0.0),
                float(out.get("mu_ml", 0.0) or 0.0),
            ]
            st["causal_last_vec"] = np.asarray(feat_vals, dtype=np.float64)
            try:
                h_val = float(out.get("hurst", st.get("hurst_last", 0.5)) or 0.5)
            except Exception:
                h_val = 0.5
            try:
                vpin_val = float(out.get("vpin", st.get("vpin_last", 0.0)) or 0.0)
            except Exception:
                vpin_val = 0.0
            try:
                h_low = float(getattr(mc_config, "hurst_low", 0.45))
                h_high = float(getattr(mc_config, "hurst_high", 0.55))
            except Exception:
                h_low, h_high = 0.45, 0.55
            if vpin_val >= 0.85:
                st["causal_last_regime"] = "volatile"
            elif h_val > h_high:
                st["causal_last_regime"] = "trend"
            elif h_val < h_low:
                st["causal_last_regime"] = "mean_revert"
            else:
                st["causal_last_regime"] = "chop"
            if "mlofi_vec" in out and out.get("mlofi_vec") is not None:
                st["mlofi_last_vec"] = np.asarray(out.get("mlofi_vec"), dtype=np.float64)

        # Causal adjustment (optional)
        if bool(getattr(mc_config, "alpha_use_causal", False)):
            features = {
                "ofi": float(ofi_score),
                "mlofi": float(out.get("mlofi", 0.0) or 0.0),
                "vpin": float(out.get("vpin", 0.0) or 0.0),
                "mu_kf": float(out.get("mu_kf", 0.0) or 0.0),
                "hurst": float(out.get("hurst", 0.0) or 0.0),
            }
            mu_causal = apply_causal_adjustment(self._alpha_causal_weights, features)
            out["mu_causal"] = float(mu_causal)

        # Direction model (logistic baseline): predict long/short edge for mu_alpha sign correction.
        if bool(getattr(mc_config, "alpha_direction_use", False)):
            model = self._alpha_dir_model if isinstance(self._alpha_dir_model, dict) else {}
            if model:
                try:
                    feat_dir = {
                        "ofi": float(ofi_score),
                        "mlofi": float(out.get("mlofi", 0.0) or 0.0),
                        "vpin": float(out.get("vpin", 0.0) or 0.0),
                        "mu_kf": float(out.get("mu_kf", 0.0) or 0.0),
                        "hurst": float(out.get("hurst", 0.0) or 0.0),
                        "mu_ml": float(out.get("mu_ml", 0.0) or 0.0),
                        "mu_bayes": float(out.get("mu_bayes", 0.0) or 0.0),
                        "mu_ar": float(out.get("mu_ar", 0.0) or 0.0),
                        "mu_pf": float(out.get("mu_pf", 0.0) or 0.0),
                        "mu_ou": float(out.get("mu_ou", 0.0) or 0.0),
                        "hawkes_boost": float(out.get("hawkes_boost", 0.0) or 0.0),
                        "sigma_garch": float(out.get("sigma_garch", 0.0) or 0.0),
                        "_regime": str(st.get("causal_last_regime") or "chop"),
                    }
                    p_long, edge, conf = predict_direction_logistic(model, feat_dir)
                    out["mu_dir_prob_long"] = float(p_long)
                    out["mu_dir_edge"] = float(edge)
                    out["mu_dir_conf"] = float(conf)
                except Exception:
                    pass

        # Keep previous alpha outputs between candle updates to avoid zero-flicker.
        cache_keys = (
            "mlofi",
            "vpin",
            "mu_kf",
            "hurst",
            "sigma_garch",
            "mu_bayes",
            "mu_bayes_var",
            "mu_ar",
            "mu_pf",
            "mu_ou",
            "mu_ml",
            "mu_causal",
            "hawkes_boost",
            "hawkes_lambda_buy",
            "hawkes_lambda_sell",
            "mu_dir_prob_long",
            "mu_dir_edge",
            "mu_dir_conf",
            "hmm_state_idx",
            "hmm_state",
            "hmm_conf",
            "hmm_regime_sign",
            "hmm_means",
        )
        alpha_cache = st.get("alpha_cache")
        if not isinstance(alpha_cache, dict):
            alpha_cache = {}
        for key in cache_keys:
            if key in out:
                alpha_cache[key] = out[key]
        for key in cache_keys:
            if key not in out and key in alpha_cache:
                out[key] = alpha_cache[key]
        st["alpha_cache"] = alpha_cache
        out["alpha_fast_mode"] = False

        return out

    def _load_alpha_training_samples(self) -> None:
        """Load persisted alpha training samples into in-memory deques at startup."""
        def _load_npz(path: str, target: deque) -> int:
            if not path:
                return 0
            if not os.path.exists(path):
                return 0
            try:
                with np.load(path, allow_pickle=True) as data:
                    X = data.get("X")
                    y = data.get("y")
                    regimes = data.get("regimes")
                    symbols = data.get("symbols")
                    timestamps = data.get("timestamps")
                    if X is None or y is None:
                        return 0
                    X = np.asarray(X)
                    y = np.asarray(y)
                    if regimes is not None:
                        regimes = np.asarray(regimes, dtype=object)
                    if symbols is not None:
                        symbols = np.asarray(symbols, dtype=object)
                    if timestamps is not None:
                        timestamps = np.asarray(timestamps)
                n = int(min(len(X), len(y)))
                if n <= 0:
                    return 0
                take = int(min(n, target.maxlen or n))
                start = n - take
                target.clear()
                for i in range(start, n):
                    if target is self._alpha_train_causal:
                        reg_val = "chop"
                        try:
                            if regimes is not None and i < len(regimes):
                                reg_val = str(regimes[i])
                        except Exception:
                            reg_val = "chop"
                        sym_val = ""
                        ts_val = 0
                        try:
                            if symbols is not None and i < len(symbols):
                                sym_val = str(symbols[i] or "")
                        except Exception:
                            sym_val = ""
                        try:
                            if timestamps is not None and i < len(timestamps):
                                ts_val = int(float(timestamps[i]))
                        except Exception:
                            ts_val = 0
                        if sym_val:
                            target.append((np.asarray(X[i], dtype=np.float64), float(y[i]), reg_val, sym_val, ts_val))
                        else:
                            target.append((np.asarray(X[i], dtype=np.float64), float(y[i]), reg_val))
                    else:
                        target.append((np.asarray(X[i], dtype=np.float64), float(y[i])))
                return n
            except Exception as e:
                try:
                    self._log_err(f"[ALPHA_TRAIN] load samples failed path={path}: {e}")
                except Exception:
                    pass
                return 0

        self._alpha_mlofi_file_samples = _load_npz(self._alpha_mlofi_train_path, self._alpha_train_mlofi)
        self._alpha_causal_file_samples = _load_npz(self._alpha_causal_train_path, self._alpha_train_causal)

        if self._alpha_mlofi_file_samples or self._alpha_causal_file_samples:
            try:
                self._log(
                    "[ALPHA_TRAIN] samples restored "
                    f"mlofi={self._alpha_mlofi_file_samples} "
                    f"causal={self._alpha_causal_file_samples}"
                )
            except Exception:
                pass
        try:
            self._alpha_train_last_mlofi_n = int(len(self._alpha_train_mlofi))
            self._alpha_train_last_causal_n = int(len(self._alpha_train_causal))
        except Exception:
            self._alpha_train_last_mlofi_n = 0
            self._alpha_train_last_causal_n = 0

    def _maybe_fit_garch_params(self, ts_ms: int) -> None:
        if not bool(self._alpha_garch_fit_enabled):
            return
        out_path = str(self._alpha_garch_path or getattr(mc_config, "garch_param_path", "") or "").strip()
        if not out_path:
            return

        proc = self._alpha_garch_fit_proc
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                timeout_ms = int(max(10.0, float(self._alpha_garch_fit_timeout_sec)) * 1000.0)
                start_ms = int(self._alpha_garch_fit_start_ms or 0)
                if start_ms > 0 and timeout_ms > 0 and (int(ts_ms) - start_ms) > timeout_ms:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        stdout, stderr = proc.communicate(timeout=1.0)
                    except Exception:
                        stdout, stderr = "", ""
                    self._alpha_garch_fit_proc = None
                    self._alpha_garch_fit_start_ms = 0
                    self._alpha_garch_fit_last_ms = int(ts_ms)
                    self._log_err(
                        "[ALPHA_GARCH] daily fit timeout; process killed "
                        f"stdout={str(stdout).strip()[:160]} stderr={str(stderr).strip()[:160]}"
                    )
                return

            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                stdout, stderr = "", ""
            self._alpha_garch_fit_proc = None
            self._alpha_garch_fit_start_ms = 0
            self._alpha_garch_fit_last_ms = int(ts_ms)
            if rc == 0:
                try:
                    self._log(f"[ALPHA_GARCH] daily fit done: {str(stdout).strip()[:220]}")
                except Exception:
                    pass
                # Force immediate reload after a successful fit output update.
                self._alpha_garch_last_reload_ms = 0
                self._maybe_reload_garch_params(int(ts_ms))
            else:
                try:
                    self._log_err(
                        "[ALPHA_GARCH] daily fit failed "
                        f"rc={rc} stdout={str(stdout).strip()[:160]} stderr={str(stderr).strip()[:160]}"
                    )
                except Exception:
                    pass
            return

        interval_ms = int(max(1.0, float(self._alpha_garch_fit_interval_sec)) * 1000.0)
        if int(ts_ms) - int(self._alpha_garch_fit_last_ms or 0) < interval_ms:
            return
        fit_hours = self._parse_utc_hours(os.environ.get("GARCH_FIT_HOURS_UTC"))
        try:
            fit_window_min = float(os.environ.get("GARCH_FIT_WINDOW_MIN", 90.0) or 90.0)
        except Exception:
            fit_window_min = 90.0
        fit_window_key = self._utc_window_key(int(ts_ms), fit_hours, fit_window_min) if fit_hours else None
        if fit_hours and fit_window_key is None:
            return
        if fit_window_key and self._alpha_garch_fit_last_window_key == fit_window_key:
            return

        lookback = int(max(256, int(self._alpha_garch_fit_lookback or 4000)))
        min_obs = int(max(64, int(self._alpha_garch_fit_min_obs or 300)))
        cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "fit_garch_params.py"),
            "--input-glob",
            str(self._alpha_garch_fit_data_glob or "data/*.csv"),
            "--out",
            str(out_path),
            "--lookback",
            str(lookback),
            "--min-obs",
            str(min_obs),
            "--allow-fallback",
            "1" if bool(self._alpha_garch_fit_allow_fallback) else "0",
        ]
        try:
            self._alpha_garch_fit_proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._alpha_garch_fit_start_ms = int(ts_ms)
            self._alpha_garch_fit_last_ms = int(ts_ms)
            self._alpha_garch_fit_last_window_key = fit_window_key
            self._log(f"[ALPHA_GARCH] daily fit started: {' '.join(cmd)}")
        except Exception as e:
            self._alpha_garch_fit_proc = None
            self._alpha_garch_fit_start_ms = 0
            self._alpha_garch_fit_last_ms = int(ts_ms)
            self._log_err(f"[ALPHA_GARCH] daily fit launch failed: {e}")

    def _maybe_persist_alpha_samples(self, ts: int) -> None:
        if str(os.environ.get("ALPHA_WEIGHT_COLLECT", "0")).strip().lower() not in ("1", "true", "yes", "on"):
            return
        if ts - int(self._alpha_train_last_save_ms or 0) < int(self._alpha_train_save_interval_sec * 1000):
            return
        self._alpha_train_last_save_ms = ts
        try:
            if self._alpha_train_mlofi:
                X = np.stack([x for x, _ in self._alpha_train_mlofi], axis=0)
                y = np.asarray([y for _, y in self._alpha_train_mlofi], dtype=np.float64)
                np.savez_compressed(self._alpha_mlofi_train_path, X=X, y=y)
                self._alpha_mlofi_file_samples = int(len(y))
            if self._alpha_train_causal:
                X_rows = []
                y_rows = []
                regimes = []
                symbols = []
                timestamps = []
                for item in self._alpha_train_causal:
                    try:
                        if isinstance(item, (list, tuple)):
                            if len(item) >= 2:
                                X_rows.append(np.asarray(item[0], dtype=np.float64))
                                y_rows.append(float(item[1]))
                                if len(item) >= 3:
                                    regimes.append(str(item[2]))
                                else:
                                    regimes.append("chop")
                                sym_val = ""
                                ts_val = 0
                                if len(item) >= 4:
                                    try:
                                        sym_val = str(item[3] or "")
                                    except Exception:
                                        sym_val = ""
                                if len(item) >= 5:
                                    try:
                                        ts_val = int(float(item[4]))
                                    except Exception:
                                        ts_val = 0
                                symbols.append(sym_val)
                                timestamps.append(ts_val)
                    except Exception:
                        continue
                if X_rows and y_rows:
                    X = np.stack(X_rows, axis=0)
                    y = np.asarray(y_rows, dtype=np.float64)
                    symbols_arr = np.asarray(symbols, dtype=object)
                    timestamps_arr = np.asarray(timestamps, dtype=np.int64)
                    try:
                        fee_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_FEE_BPS", 0.0) or 0.0)
                    except Exception:
                        fee_bps = 0.0
                    try:
                        slippage_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", 0.0) or 0.0)
                    except Exception:
                        slippage_bps = 0.0
                    try:
                        extra_cost_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_COST_BPS", 0.0) or 0.0)
                    except Exception:
                        extra_cost_bps = 0.0
                    cost_ret = float(max(0.0, (fee_bps + slippage_bps + extra_cost_bps) / 10_000.0))
                    y_net = np.sign(y) * np.maximum(np.abs(y) - cost_ret, 0.0)
                    y_hstar_net = np.zeros_like(y_net, dtype=np.float64)
                    y_hstar = np.zeros_like(y_net, dtype=np.float64)
                    hstar_bars = np.zeros((int(y_net.shape[0]),), dtype=np.int32)
                    try:
                        hstar_on = str(os.environ.get("ALPHA_DIRECTION_HSTAR_LABEL_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        hstar_on = True
                    if hstar_on and int(y.shape[0]) > 8:
                        try:
                            hbar_raw = str(os.environ.get("ALPHA_DIRECTION_HSTAR_BARS", "1,2,3,5,8,13,21,34") or "")
                        except Exception:
                            hbar_raw = "1,2,3,5,8,13,21,34"
                        hbars: list[int] = []
                        for tok in str(hbar_raw).split(","):
                            tok = str(tok).strip()
                            if not tok:
                                continue
                            try:
                                h = int(float(tok))
                            except Exception:
                                continue
                            if h > 0:
                                hbars.append(h)
                        if not hbars:
                            hbars = [1, 2, 3, 5, 8, 13, 21, 34]
                        hbars = sorted(set(hbars))
                        idx_by_sym: dict[str, list[int]] = {}
                        for i, sym in enumerate(symbols_arr):
                            s = str(sym or "")
                            if not s:
                                continue
                            idx_by_sym.setdefault(s, []).append(int(i))
                        for idxs in idx_by_sym.values():
                            if len(idxs) <= 1:
                                continue
                            arr = np.asarray(idxs, dtype=np.int64)
                            if timestamps_arr.shape[0] == y.shape[0]:
                                ts_sub = np.asarray(timestamps_arr[arr], dtype=np.int64)
                                order = np.argsort(ts_sub, kind="mergesort")
                                arr = arr[order]
                            rets = np.asarray(y[arr], dtype=np.float64)
                            pref = np.concatenate(([0.0], np.cumsum(rets, dtype=np.float64)))
                            m = int(arr.shape[0])
                            for j in range(m - 1):
                                best_score = -1e18
                                best_cum = 0.0
                                best_h = 0
                                for h in hbars:
                                    if (j + h) > m:
                                        break
                                    cum_r = float(pref[j + h] - pref[j])
                                    score = abs(cum_r) - cost_ret
                                    if score > best_score:
                                        best_score = float(score)
                                        best_cum = float(cum_r)
                                        best_h = int(h)
                                if best_h > 0:
                                    net_mag = max(0.0, abs(float(best_cum)) - cost_ret)
                                    net_lbl = math.copysign(net_mag, float(best_cum)) if net_mag > 0 else 0.0
                                    src_idx = int(arr[j])
                                    y_hstar_net[src_idx] = float(net_lbl)
                                    y_hstar[src_idx] = float(best_cum)
                                    hstar_bars[src_idx] = int(best_h)
                    np.savez_compressed(
                        self._alpha_causal_train_path,
                        X=X,
                        y=y,
                        y_net=np.asarray(y_net, dtype=np.float64),
                        y_fee_adj=np.asarray(y_net, dtype=np.float64),
                        y_hstar=np.asarray(y_hstar, dtype=np.float64),
                        y_hstar_net=np.asarray(y_hstar_net, dtype=np.float64),
                        hstar_bars=np.asarray(hstar_bars, dtype=np.int32),
                        regimes=np.asarray(regimes, dtype=object),
                        symbols=symbols_arr,
                        timestamps=timestamps_arr,
                        feature_names=np.asarray(self._alpha_causal_feature_names, dtype=object),
                    )
                    self._alpha_causal_file_samples = int(len(y))
        except Exception as e:
            try:
                self._log_err(f"[ALPHA_TRAIN] persist failed: {e}")
            except Exception:
                pass

    def _maybe_train_alpha_weights(self, ts: int) -> None:
        if str(os.environ.get("ALPHA_WEIGHT_TRAIN_ENABLED", "0")).strip().lower() not in ("1", "true", "yes", "on"):
            return
        m_n = int(len(self._alpha_train_mlofi))
        c_n = int(len(self._alpha_train_causal))
        if m_n < self._alpha_train_min_samples and c_n < self._alpha_train_min_samples:
            return
        min_new = int(max(0, int(self._alpha_train_min_new_samples or 0)))
        new_m = max(0, m_n - int(self._alpha_train_last_mlofi_n or 0))
        new_c = max(0, c_n - int(self._alpha_train_last_causal_n or 0))
        closed_now = int(max(0, int(self._closed_trade_count or 0)))
        prev_exit_anchor = int(max(0, int(self._alpha_train_last_exit_closed or 0)))
        new_exits = int(max(0, closed_now - prev_exit_anchor))
        try:
            force_on_new_exits = str(os.environ.get("ALPHA_TRAIN_FORCE_ON_NEW_EXITS", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            force_on_new_exits = True
        try:
            trigger_new_exits = int(os.environ.get("ALPHA_TRAIN_TRIGGER_NEW_EXITS", 200) or 200)
        except Exception:
            trigger_new_exits = 200
        force_train = bool(force_on_new_exits and trigger_new_exits > 0 and new_exits >= trigger_new_exits)
        force_reason = None
        if force_train:
            force_reason = f"new_exits:{new_exits}/{trigger_new_exits}"
        interval_ms = int(self._alpha_train_interval_sec * 1000)
        if not force_train and ts - int(self._alpha_train_last_train_ms or 0) < interval_ms:
            return
        if force_train:
            try:
                force_min_sec = float(os.environ.get("ALPHA_TRAIN_FORCE_MIN_SEC", 300.0) or 300.0)
            except Exception:
                force_min_sec = 300.0
            if ts - int(self._alpha_train_last_train_ms or 0) < int(max(0.0, force_min_sec) * 1000.0):
                return
        if min_new > 0 and max(new_m, new_c) < min_new and not force_train:
            return
        train_hours = self._parse_utc_hours(os.environ.get("ALPHA_WEIGHT_TRAIN_HOURS_UTC"))
        try:
            train_window_min = float(os.environ.get("ALPHA_WEIGHT_TRAIN_WINDOW_MIN", 90.0) or 90.0)
        except Exception:
            train_window_min = 90.0
        try:
            force_ignore_window = str(os.environ.get("ALPHA_TRAIN_FORCE_IGNORE_WINDOW", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            force_ignore_window = True
        train_window_key = self._utc_window_key(int(ts), train_hours, train_window_min) if train_hours else None
        if train_hours and train_window_key is None and not (force_train and force_ignore_window):
            return
        if train_window_key and self._alpha_train_last_window_key == train_window_key and not force_train:
            return
        self._alpha_train_last_train_ms = ts
        try:
            from scripts.train_alpha_weights import train_mlofi_weights, train_causal_weights, train_mu_direction_model

            lam = float(os.environ.get("ALPHA_WEIGHT_RIDGE_LAMBDA", 1e-3) or 1e-3)
            mlofi_out = str(getattr(mc_config, "mlofi_weight_path", "") or "state/mlofi_weights.json")
            causal_out = str(getattr(mc_config, "causal_weights_path", "") or "state/causal_weights.json")
            mlofi_out_npy = str(os.environ.get("MLOFI_WEIGHT_NPY", "state/mlofi_weights.npy"))
            dir_out = str(getattr(mc_config, "alpha_direction_model_path", "") or "state/mu_direction_model.json")
            dir_min_samples = int(os.environ.get("ALPHA_DIRECTION_MIN_SAMPLES", max(500, self._alpha_train_min_samples)) or max(500, self._alpha_train_min_samples))
            dir_train_enabled = str(os.environ.get("ALPHA_DIRECTION_TRAIN_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")

            train_mlofi_weights(self._alpha_mlofi_train_path, mlofi_out, mlofi_out_npy, lam)
            train_causal_weights(self._alpha_causal_train_path, causal_out, lam)
            if dir_train_enabled:
                train_mu_direction_model(self._alpha_causal_train_path, dir_out, lam, dir_min_samples)

            # reload weights
            self._alpha_mlofi_weights = load_weight_vector(mlofi_out)
            self._alpha_causal_weights = load_causal_weights(causal_out)
            if os.path.exists(dir_out):
                self._alpha_dir_model = load_direction_model(dir_out)
                try:
                    self._alpha_dir_model_mtime = float(os.path.getmtime(dir_out))
                except Exception:
                    self._alpha_dir_model_mtime = self._alpha_dir_model_mtime
            self._alpha_train_last_mlofi_n = int(m_n)
            self._alpha_train_last_causal_n = int(c_n)
            self._alpha_train_last_exit_closed = int(closed_now)
            self._alpha_train_last_window_key = train_window_key
            self._persist_alpha_train_state(reason=(force_reason or "scheduled"))
            try:
                self._log(
                    "[ALPHA_TRAIN] weights refreshed: "
                    f"mlofi={mlofi_out} causal={causal_out} dir={dir_out} "
                    f"samples(m,c)=({m_n},{c_n}) new=({new_m},{new_c}) "
                    f"new_exits={new_exits} force={bool(force_train)} reason={force_reason or '-'} "
                    f"window={train_window_key}"
                )
            except Exception:
                pass
        except Exception as e:
            try:
                self._log_err(f"[ALPHA_TRAIN] train failed: {e}")
            except Exception:
                pass

    def _liquidity_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym)
        if not ob:
            return 1.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2) + sum(float(a[1]) for a in asks[:5] if len(a) >= 2)
        return float(max(vol, 1.0))

    def _position_idx_for_side(self, position_side: str) -> int:
        if self._is_hedge_mode:
            return 1 if position_side == "LONG" else 2
        return 0

    def _should_force_market(self, sym: str) -> bool:
        ob = self.orderbook.get(sym)
        if not ob or not ob.get("ready"):
            return True
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return True
        try:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
        except Exception:
            return True
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        if mid <= 0:
            return True
        spread_pct = (ask - bid) / mid
        if spread_pct > SPREAD_PCT_MAX:
            return True
        closes = list(self.ohlcv_buffer.get(sym, []))
        mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
        if sigma_bar is not None and float(sigma_bar) > float(VOLATILITY_MARKET_THRESHOLD):
            return True
        return False

    def _maybe_place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        position_side: str | None = None,
        execution_type: str | None = None,
    ):
        if not self.enable_orders or quantity <= 0:
            return
        asyncio.create_task(
            self._execute_order(
                symbol,
                side,
                quantity,
                reduce_only,
                position_side=position_side,
                execution_type=execution_type,
            )
        )

    def _handle_entry_order_failure(self, symbol: str, side: str, quantity: float, err_text: str):
        pos = self.positions.get(symbol)
        if not pos:
            return
        if "110007" in str(err_text):
            try:
                lev = float(pos.get("leverage", self.leverage) or self.leverage)
            except Exception:
                lev = float(self.leverage)
            try:
                notional = float(pos.get("notional", 0.0) or 0.0)
            except Exception:
                notional = 0.0
            req_margin = (notional / max(lev, 1e-6)) if notional > 0 else 0.0
            self._log_err(
                f"[ORDER_110007] {symbol} insufficient available balance | "
                f"qty={float(quantity):.6f} lev={lev:.2f} est_notional={notional:.2f} est_margin={req_margin:.2f} "
                f"live_free={self._live_free_balance} live_wallet={self._live_wallet_balance} live_equity={self._live_equity}"
            )
            self._register_anomaly(
                "order_insufficient_balance",
                "warn",
                f"{symbol} order rejected(110007): available balance 부족",
                {
                    "qty": float(quantity),
                    "leverage": float(lev),
                    "est_notional": float(notional),
                    "est_margin": float(req_margin),
                    "live_free_balance": self._live_free_balance,
                },
            )
        try:
            pos["order_status"] = "failed"
            pos["order_error"] = err_text
            pos["order_error_ts"] = now_ms()
        except Exception:
            pass
        fee_entry = 0.0
        try:
            fee_entry = float(pos.get("fee_paid", 0.0) or 0.0)
        except Exception:
            fee_entry = 0.0
        if fee_entry:
            try:
                self.balance += float(fee_entry)
            except Exception:
                pass
        self.positions.pop(symbol, None)
        try:
            price = float(pos.get("entry_price") or 0.0)
        except Exception:
            price = 0.0
        try:
            self._record_trade(
                "ORDER_REJECT",
                symbol,
                side,
                price,
                float(quantity),
                pos,
                pnl=0.0,
                fee=-fee_entry if fee_entry else None,
                reason=f"order_reject:{err_text}",
            )
        except Exception:
            pass
        self._register_anomaly("order_reject", "warn", f"{symbol} order rejected: {err_text}")

    def _entry_order_price_hint(self, symbol: str, side: str) -> float | None:
        px = None
        try:
            ob = self.orderbook.get(symbol) or {}
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if str(side).upper() == "LONG" and asks:
                px = float(asks[0][0])
            elif str(side).upper() == "SHORT" and bids:
                px = float(bids[0][0])
            elif bids and asks:
                px = 0.5 * (float(bids[0][0]) + float(asks[0][0]))
        except Exception:
            px = None
        if px is None:
            try:
                px = float((self.market.get(symbol) or {}).get("price"))
            except Exception:
                px = None
        if px is None or (not math.isfinite(px)) or px <= 0:
            return None
        return float(px)

    @staticmethod
    def _normalize_regime_bucket(regime: str | None) -> str:
        s = str(regime or "").strip().lower()
        if not s:
            return "chop"
        if s in ("trend", "trending", "bull", "bear", "persistent"):
            return "trend"
        if ("vol" in s) or ("shock" in s) or ("panic" in s):
            return "volatile"
        if ("mean" in s) or ("revert" in s) or ("anti" in s):
            return "mean_revert"
        return "chop"

    def _regime_min_qty_upsize_limit(self, regime_hint: str | None = None) -> tuple[float, str]:
        try:
            base = float(os.environ.get("LIVE_MIN_QTY_MAX_UPSIZE", 5.0) or 5.0)
        except Exception:
            base = 5.0
        bucket = self._normalize_regime_bucket(regime_hint)
        env_key_map = {
            "trend": "LIVE_MIN_QTY_MAX_UPSIZE_TREND",
            "chop": "LIVE_MIN_QTY_MAX_UPSIZE_CHOP",
            "volatile": "LIVE_MIN_QTY_MAX_UPSIZE_VOLATILE",
            "mean_revert": "LIVE_MIN_QTY_MAX_UPSIZE_MEAN_REVERT",
        }
        key = env_key_map.get(bucket)
        if key:
            try:
                raw = os.environ.get(key)
                if raw is not None and str(raw).strip() != "":
                    base = float(raw)
            except Exception:
                pass
        return float(max(1.0, base)), bucket

    def _regime_entry_min_notional(self, regime_hint: str | None = None) -> tuple[float, str]:
        try:
            base = float(os.environ.get("LIVE_ENTRY_MIN_NOTIONAL", os.environ.get("MIN_ENTRY_NOTIONAL", 0.0)) or 0.0)
        except Exception:
            base = 0.0
        bucket = self._normalize_regime_bucket(regime_hint)
        env_key_map = {
            "trend": "LIVE_ENTRY_MIN_NOTIONAL_TREND",
            "chop": "LIVE_ENTRY_MIN_NOTIONAL_CHOP",
            "volatile": "LIVE_ENTRY_MIN_NOTIONAL_VOLATILE",
            "mean_revert": "LIVE_ENTRY_MIN_NOTIONAL_MEAN_REVERT",
        }
        key = env_key_map.get(bucket)
        if key:
            try:
                raw = os.environ.get(key)
                if raw is not None and str(raw).strip() != "":
                    base = float(raw)
            except Exception:
                pass
        return float(max(0.0, base)), bucket

    def _symbol_amount_constraints(self, symbol: str) -> dict:
        market = {}
        try:
            if hasattr(self.exchange, "market"):
                market = self.exchange.market(symbol) or {}
        except Exception:
            market = {}
        if not market:
            try:
                market = (getattr(self.exchange, "markets", {}) or {}).get(symbol) or {}
            except Exception:
                market = {}
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        cost_limits = limits.get("cost") or {}
        precision = market.get("precision") or {}

        min_qty = self._safe_float(amount_limits.get("min"), None)
        max_qty = self._safe_float(amount_limits.get("max"), None)
        qty_step = self._safe_float(precision.get("amount"), None)
        min_cost = self._safe_float(cost_limits.get("min"), None)

        if qty_step is not None and qty_step <= 0:
            qty_step = None
        if min_qty is not None and min_qty <= 0:
            min_qty = None
        if max_qty is not None and max_qty <= 0:
            max_qty = None
        if min_cost is not None and min_cost <= 0:
            min_cost = None

        if qty_step is not None:
            min_qty = max(float(min_qty or 0.0), float(qty_step))
        return {
            "min_qty": float(min_qty) if min_qty is not None else None,
            "max_qty": float(max_qty) if max_qty is not None else None,
            "qty_step": float(qty_step) if qty_step is not None else None,
            "min_cost": float(min_cost) if min_cost is not None else None,
        }

    @staticmethod
    def _round_qty_to_step(quantity: float, step: float, *, up: bool = False) -> float:
        try:
            q = Decimal(str(float(quantity)))
            s = Decimal(str(float(step)))
            if q <= 0 or s <= 0:
                return 0.0
            units = (q / s).to_integral_value(rounding=ROUND_UP if up else ROUND_DOWN)
            out = units * s
            return float(out)
        except Exception:
            return float(quantity)

    @staticmethod
    def _parse_min_qty_precision(err_text: str) -> float | None:
        try:
            m = re.search(r"minimum amount precision of ([0-9eE+\\.-]+)", str(err_text), flags=re.IGNORECASE)
            if not m:
                return None
            v = float(m.group(1))
            if not math.isfinite(v) or v <= 0:
                return None
            return float(v)
        except Exception:
            return None

    def _to_amount_precision(self, symbol: str, quantity: float, *, up: bool = False, fallback_step: float | None = None) -> float:
        qty = float(quantity)
        if qty <= 0:
            return 0.0
        if hasattr(self.exchange, "amount_to_precision"):
            try:
                out = float(self.exchange.amount_to_precision(symbol, qty))
                if math.isfinite(out) and out > 0:
                    return out
                return 0.0
            except Exception as e:
                min_from_err = self._parse_min_qty_precision(str(e))
                if up and min_from_err is not None:
                    return self._round_qty_to_step(max(qty, float(min_from_err)), float(min_from_err), up=True)
        if fallback_step is not None and fallback_step > 0:
            return self._round_qty_to_step(qty, fallback_step, up=up)
        return qty

    def _precision_aware_entry_quantity(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: float,
        *,
        price_hint: float | None = None,
        context: str = "entry",
        regime_hint: str | None = None,
    ) -> float:
        try:
            qty_raw = float(quantity)
        except Exception:
            return 0.0
        if (not math.isfinite(qty_raw)) or qty_raw <= 0:
            return 0.0

        constraints = self._symbol_amount_constraints(symbol)
        min_qty = constraints.get("min_qty")
        max_qty = constraints.get("max_qty")
        qty_step = constraints.get("qty_step")
        min_cost = constraints.get("min_cost")

        px = price_hint
        if px is None or (not math.isfinite(float(px))) or float(px) <= 0:
            px = self._entry_order_price_hint(symbol, side)
        px = float(px) if (px is not None and math.isfinite(float(px)) and float(px) > 0) else None

        try:
            min_mode = str(os.environ.get("LIVE_MIN_QTY_MODE", "raise")).strip().lower()
        except Exception:
            min_mode = "raise"
        allow_raise = min_mode in ("raise", "up", "ceil", "upsize")
        max_upsize, regime_bucket = self._regime_min_qty_upsize_limit(regime_hint)

        qty = self._guard_entry_quantity_by_free_balance(symbol, side, qty_raw, leverage)
        qty = self._to_amount_precision(symbol, qty, up=False, fallback_step=qty_step)
        if qty_step is not None and qty_step > 0:
            qty = self._round_qty_to_step(qty, qty_step, up=False)
        if max_qty is not None and qty > max_qty:
            qty = self._round_qty_to_step(max_qty, qty_step or max_qty, up=False)
            qty = self._to_amount_precision(symbol, qty, up=False, fallback_step=qty_step)

        req_qty = 0.0
        if min_qty is not None and min_qty > 0:
            req_qty = max(req_qty, float(min_qty))
        if min_cost is not None and min_cost > 0 and px is not None and px > 0:
            req_qty = max(req_qty, float(min_cost) / float(px))
        if qty_step is not None and qty_step > 0 and req_qty > 0:
            req_qty = self._round_qty_to_step(req_qty, qty_step, up=True)

        if req_qty > 0 and qty < req_qty:
            if not allow_raise:
                self._log(
                    f"[ENTRY_SKIP_MIN_QTY] {symbol} {context} qty={qty_raw:.8f}->{qty:.8f} "
                    f"required={req_qty:.8f} mode=skip regime={regime_bucket}"
                )
                return 0.0
            baseline = max(qty, 1e-12)
            upsize_ratio = float(req_qty / baseline)
            if upsize_ratio > max_upsize:
                self._log(
                    f"[ENTRY_SKIP_MIN_QTY] {symbol} {context} upsize_ratio={upsize_ratio:.2f} "
                    f"limit={max_upsize:.2f} qty={qty:.8f} required={req_qty:.8f} regime={regime_bucket}"
                )
                return 0.0
            qty = self._round_qty_to_step(req_qty, qty_step or req_qty, up=True)
            qty = self._to_amount_precision(symbol, qty, up=True, fallback_step=qty_step)
            qty = self._guard_entry_quantity_by_free_balance(symbol, side, qty, leverage)
            qty = self._to_amount_precision(symbol, qty, up=False, fallback_step=qty_step)
            if qty_step is not None and qty_step > 0:
                qty = self._round_qty_to_step(qty, qty_step, up=False)
            if qty < req_qty:
                self._log(
                    f"[ENTRY_SKIP_MIN_QTY] {symbol} {context} cannot satisfy required qty after balance guard "
                    f"(qty={qty:.8f}, required={req_qty:.8f}, regime={regime_bucket})"
                )
                return 0.0

        if max_qty is not None and qty > max_qty:
            qty = self._round_qty_to_step(max_qty, qty_step or max_qty, up=False)
            qty = self._to_amount_precision(symbol, qty, up=False, fallback_step=qty_step)
        if (not math.isfinite(qty)) or qty <= 0:
            return 0.0

        if abs(float(qty) - float(qty_raw)) > 1e-12:
            try:
                notional = float(qty * px) if px is not None else None
            except Exception:
                notional = None
            msg = f"[ENTRY_QTY_NORM] {symbol} {context} qty={qty_raw:.8f}->{qty:.8f}"
            if notional is not None and math.isfinite(notional):
                msg += f" notional={notional:.4f}"
            self._log(msg)
        return float(qty)

    def _guard_entry_quantity_by_free_balance(self, symbol: str, side: str, quantity: float, leverage: float) -> float:
        try:
            qty = float(quantity)
        except Exception:
            return quantity
        if qty <= 0:
            return qty
        px = self._entry_order_price_hint(symbol, side)
        if px is None:
            return qty
        try:
            lev = float(leverage) if leverage and float(leverage) > 0 else float(self.leverage)
        except Exception:
            lev = float(self.leverage)
        if lev <= 0:
            lev = 1.0
        try:
            free_bal = float(self._sizing_balance() or 0.0)
        except Exception:
            free_bal = 0.0
        if free_bal <= 0:
            return qty
        try:
            guard = float(os.environ.get("LIVE_ORDER_BALANCE_GUARD", 0.90) or 0.90)
        except Exception:
            guard = 0.90
        guard = min(max(guard, 0.10), 0.99)
        req_notional = px * qty
        max_notional = free_bal * lev * guard
        if (not math.isfinite(req_notional)) or (not math.isfinite(max_notional)) or max_notional <= 0:
            return qty
        if req_notional <= max_notional:
            return qty
        new_qty = max_notional / max(px, 1e-12)
        try:
            if hasattr(self.exchange, "amount_to_precision"):
                new_qty = float(self.exchange.amount_to_precision(symbol, new_qty))
        except Exception:
            pass
        if (not math.isfinite(new_qty)) or new_qty <= 0:
            return qty
        if new_qty >= qty:
            return qty
        self._log(
            f"[ORDER_GUARD] {symbol} qty scaled by free balance: {qty:.6f} -> {new_qty:.6f} "
            f"(free={free_bal:.4f}, lev={lev:.2f}, guard={guard:.2f})"
        )
        return float(new_qty)

    async def _submit_limit_close(self, symbol: str, price: float) -> dict:
        """Place a reduce-only limit close for an open position."""
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "error": "position_not_found"}
        try:
            qty = float(pos.get("quantity", 0.0))
        except Exception:
            qty = 0.0
        if qty <= 0:
            return {"ok": False, "error": "no_quantity"}
        side = str(pos.get("side") or "LONG").upper()
        order_side = "sell" if side == "LONG" else "buy"
        if not self.enable_orders:
            try:
                self._close_position(symbol, float(price), "manual_limit_close", exit_kind="MANUAL", skip_order=True)
            except Exception as e:
                return {"ok": False, "error": str(e)}
            return {"ok": True, "mode": "paper"}
        params = {
            "reduceOnly": True,
            "positionIdx": self._position_idx_for_side(side),
        }
        try:
            order = await self._ccxt_call(
                "limit_close",
                self.exchange.create_order,
                symbol,
                "limit",
                order_side,
                qty,
                float(price),
                params,
            )
            try:
                pos["order_status"] = "closing"
                pos["close_order_px"] = float(price)
                pos["close_order_ts"] = now_ms()
            except Exception:
                pass
            self._log(f"[ORDER] limit close {symbol} {order_side} qty={qty:.6f} px={float(price):.4f}")
            return {"ok": True, "order": order}
        except Exception as e:
            self._log_err(f"[ERR] limit close {symbol}: {e}")
            return {"ok": False, "error": str(e)}

    async def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        position_side: str | None = None,
        execution_type: str | None = None,
    ):
        """
        실 주문 호출. ENABLE_LIVE_ORDERS가 True일 때만 create_order를 호출한다.
        """
        order_side = "buy" if side == "LONG" else "sell"
        exec_type = execution_type or EXECUTION_MODE
        pos_side = position_side or side
        lev_for_order = float(self._dyn_leverage.get(symbol, self.leverage) or self.leverage)
        params = {
            "reduceOnly": reduce_only,
            "positionIdx": self._position_idx_for_side(pos_side),
        }
        try:
            if not reduce_only:
                try:
                    target_lev = None
                    pos = self.positions.get(symbol)
                    if pos is not None:
                        target_lev = pos.get("leverage")
                    if target_lev is None:
                        target_lev = self._dyn_leverage.get(symbol, self.leverage)
                    if target_lev is not None:
                        try:
                            lev_for_order = float(target_lev)
                        except Exception:
                            lev_for_order = float(self.leverage)
                        lev_ok = await self._sync_symbol_leverage(symbol, float(target_lev))
                        if not lev_ok:
                            fail_meta = self._leverage_sync_fail_by_sym.get(symbol, {}) or {}
                            err = fail_meta.get("err", "set_leverage_failed")
                            try:
                                fallback_on_fail = bool(getattr(mc_config, "leverage_sync_fallback_on_fail", True))
                            except Exception:
                                fallback_on_fail = True
                            fallback_applied = False
                            if fallback_on_fail:
                                err_l = str(err).lower()
                                if ("110012" in err_l) or ("110013" in err_l) or ("ab not enough for new leverage" in err_l) or ("risk limit" in err_l):
                                    try:
                                        suggested_lev = fail_meta.get("suggested_leverage")
                                        fallback_lev = float(
                                            suggested_lev
                                            or self._last_leverage_target_by_sym.get(symbol)
                                            or self._dyn_leverage.get(symbol)
                                            or self.leverage
                                        )
                                    except Exception:
                                        fallback_lev = float(self.leverage)
                                    lev_for_order = max(1.0, float(fallback_lev))
                                    self._dyn_leverage[symbol] = float(lev_for_order)
                                    if pos is not None:
                                        pos["leverage"] = float(lev_for_order)
                                    fallback_applied = True
                                    self._log_err(
                                        f"[ORDER] leverage sync failed; fallback leverage used: "
                                        f"{symbol} err={err} fallback={lev_for_order:.2f}x"
                                    )
                            try:
                                block_on_fail = str(os.environ.get("BLOCK_ENTRY_ON_LEVERAGE_SYNC_FAIL", "1")).strip().lower() in ("1", "true", "yes", "on")
                            except Exception:
                                block_on_fail = True
                            if block_on_fail and (not fallback_applied):
                                self._log_err(f"[ORDER] leverage sync failed; entry blocked: {symbol} err={err}")
                                self._handle_entry_order_failure(symbol, side, quantity, f"leverage_sync_failed:{err}")
                                return
                except Exception:
                    pass
                try:
                    regime_hint = ((self.positions.get(symbol) or {}).get("regime"))
                except Exception:
                    regime_hint = None
                quantity = self._precision_aware_entry_quantity(
                    symbol,
                    side,
                    quantity,
                    lev_for_order,
                    context="order_submit",
                    regime_hint=regime_hint,
                )
                if quantity <= 0:
                    self._handle_entry_order_failure(symbol, side, quantity, "insufficient_balance_or_min_qty")
                    return
            if exec_type == "maker_dynamic" and not self._should_force_market(symbol):
                ob = self.orderbook.get(symbol)
                bids = ob.get("bids") if ob else []
                asks = ob.get("asks") if ob else []
                price = None
                if order_side == "buy" and bids:
                    price = float(bids[0][0])
                elif order_side == "sell" and asks:
                    price = float(asks[0][0])
                if price is not None and price > 0:
                    limit_order = await self.exchange.create_order(symbol, "limit", order_side, quantity, price, params)
                    order_id = limit_order.get("id") if isinstance(limit_order, dict) else None
                    self._log(f"[ORDER] maker {symbol} {order_side} qty={quantity:.6f} px={price:.4f} reduce_only={reduce_only}")
                    await asyncio.sleep(MAKER_TIMEOUT_SEC)
                    if order_id:
                        try:
                            fetched = await self.exchange.fetch_order(order_id, symbol)
                            remaining = float(fetched.get("remaining", 0.0) or 0.0)
                            status = str(fetched.get("status", ""))
                        except Exception:
                            remaining = float(quantity)
                            status = "unknown"
                        if status.lower() == "closed" or remaining <= 0:
                            self._log(f"[ORDER] maker filled: {symbol} {order_side} id={order_id}")
                            if not reduce_only:
                                pos = self.positions.get(symbol)
                                if pos is not None:
                                    pos["order_status"] = "ack"
                                    pos["order_ack_ts"] = now_ms()
                                try:
                                    self._balance_reject_110007_by_sym.pop(symbol, None)
                                except Exception:
                                    pass
                            return
                        try:
                            await self.exchange.cancel_order(order_id, symbol)
                        except Exception as cancel_err:
                            self._log_err(f"[ORDER] maker cancel failed: {symbol} id={order_id} err={cancel_err}")
                        if remaining > 0:
                            await self.exchange.create_order(symbol, "market", order_side, remaining, None, params)
                            self._log(f"[ORDER] taker fallback: {symbol} {order_side} qty={remaining:.6f} reduce_only={reduce_only}")
                            if not reduce_only:
                                pos = self.positions.get(symbol)
                                if pos is not None:
                                    pos["order_status"] = "ack"
                                    pos["order_ack_ts"] = now_ms()
                                try:
                                    self._balance_reject_110007_by_sym.pop(symbol, None)
                                except Exception:
                                    pass
                            return
            await self.exchange.create_order(symbol, "market", order_side, quantity, None, params)
            self._log(f"[ORDER] {symbol} {order_side} {quantity:.6f} reduce_only={reduce_only}")
            if not reduce_only:
                pos = self.positions.get(symbol)
                if pos is not None:
                    pos["order_status"] = "ack"
                    pos["order_ack_ts"] = now_ms()
                try:
                    self._balance_reject_110007_by_sym.pop(symbol, None)
                except Exception:
                    pass
        except Exception as e:
            err_text = str(e)
            if (not reduce_only) and ("110007" in err_text):
                try:
                    retry_shrink = float(os.environ.get("LIVE_ORDER_RETRY_SHRINK", 0.85) or 0.85)
                except Exception:
                    retry_shrink = 0.85
                retry_shrink = min(max(retry_shrink, 0.10), 0.95)
                try:
                    retry_max_attempts = int(os.environ.get("LIVE_ORDER_RETRY_MAX_ATTEMPTS", 3) or 3)
                except Exception:
                    retry_max_attempts = 3
                retry_max_attempts = max(1, min(6, retry_max_attempts))
                retry_qty_prev = float(quantity)
                retry_qty_curr = float(quantity)
                retry_last_err = None
                for retry_idx in range(retry_max_attempts):
                    retry_qty_curr = float(retry_qty_curr) * retry_shrink
                    retry_qty_curr = self._precision_aware_entry_quantity(
                        symbol,
                        side,
                        retry_qty_curr,
                        lev_for_order,
                        context="order_retry_110007",
                        regime_hint=regime_hint if 'regime_hint' in locals() else None,
                    )
                    if retry_qty_curr <= 0 or retry_qty_curr >= retry_qty_prev:
                        break
                    try:
                        await self.exchange.create_order(symbol, "market", order_side, retry_qty_curr, None, params)
                        self._log(
                            f"[ORDER_RETRY_110007] {symbol} {order_side} retry success "
                            f"attempt={retry_idx + 1}/{retry_max_attempts} qty={float(quantity):.6f}->{float(retry_qty_curr):.6f}"
                        )
                        pos = self.positions.get(symbol)
                        if pos is not None:
                            pos["order_status"] = "ack"
                            pos["order_ack_ts"] = now_ms()
                        try:
                            self._balance_reject_110007_by_sym.pop(symbol, None)
                        except Exception:
                            pass
                        return
                    except Exception as retry_err:
                        retry_last_err = retry_err
                        retry_qty_prev = float(retry_qty_curr)
                        continue
                try:
                    now_rej = now_ms()
                    rej_state = self._balance_reject_110007_by_sym.get(symbol) or {}
                    prev_count = int(rej_state.get("count", 0) or 0)
                    prev_ts = int(rej_state.get("ts", 0) or 0)
                    if prev_ts > 0 and (now_rej - prev_ts) > 600_000:
                        prev_count = 0
                    self._balance_reject_110007_by_sym[symbol] = {
                        "count": int(min(20, prev_count + 1)),
                        "ts": int(now_rej),
                    }
                except Exception:
                    pass
                if retry_last_err is not None:
                    err_text = f"{err_text} | retry_err={retry_last_err}"
            self._log_err(f"[ERR] order {symbol} {order_side}: {err_text}")
            if not reduce_only:
                self._handle_entry_order_failure(symbol, side, quantity, err_text)

    async def _sync_symbol_leverage(self, symbol: str, target: float, force: bool = False) -> bool:
        if not self.enable_orders:
            return True
        if not getattr(self.exchange, "apiKey", None) or not getattr(self.exchange, "secret", None):
            return True
        if not hasattr(self.exchange, "set_leverage"):
            return True
        try:
            lev = float(target)
        except Exception:
            return False
        if lev <= 0:
            return False
        try:
            step = float(getattr(mc_config, "leverage_sync_step", 1.0) or 1.0)
        except Exception:
            step = 1.0
        if step > 0:
            lev = round(lev / step) * step
        lev = float(max(1.0, min(float(self.max_leverage or lev), lev)))
        now = now_ms()
        last_ts = int(self._last_leverage_sync_ms_by_sym.get(symbol, 0) or 0)
        last_target = float(self._last_leverage_target_by_sym.get(symbol, 0.0) or 0.0)
        if not force:
            if (now - last_ts) < int(self._leverage_sync_min_interval_ms) and abs(lev - last_target) < 1e-6:
                return True
        try:
            await self._ccxt_call("set_leverage", self.exchange.set_leverage, float(lev), symbol)
            self._last_leverage_sync_ms_by_sym[symbol] = now
            self._last_leverage_target_by_sym[symbol] = float(lev)
            self._leverage_sync_fail_by_sym.pop(symbol, None)
            try:
                self._log(f"[EXCHANGE] leverage synced: {symbol} -> {lev:.1f}x (per-order)")
            except Exception:
                pass
            return True
        except Exception as e:
            err_text = str(e)
            # 일부 마켓은 정수 레버리지 강제. float 실패 시 int로 1회 fallback.
            try_int_retry = abs(float(lev) - round(float(lev))) > 1e-9 and (
                "precision" in err_text.lower()
                or "invalid" in err_text.lower()
                or "parameter" in err_text.lower()
            )
            if try_int_retry:
                lev_int = float(int(round(lev)))
                try:
                    await self._ccxt_call("set_leverage", self.exchange.set_leverage, lev_int, symbol)
                    self._last_leverage_sync_ms_by_sym[symbol] = now
                    self._last_leverage_target_by_sym[symbol] = lev_int
                    self._leverage_sync_fail_by_sym.pop(symbol, None)
                    self._log(f"[EXCHANGE] leverage synced: {symbol} -> {lev_int:.1f}x (int-fallback)")
                    return True
                except Exception as e2:
                    err_text = f"{err_text} | int_fallback_err={e2}"
            # Bybit risk-limit cap case (e.g. 110013 maxLeverage [2500]): auto retry with parsed max leverage.
            suggested_lev = None
            if "110013" in err_text and "maxleverage" in err_text.lower():
                try:
                    m = re.search(r"maxLeverage\s*\[(\d+)\]", err_text, flags=re.IGNORECASE)
                except Exception:
                    m = None
                if m:
                    try:
                        raw_cap = float(m.group(1))
                        cap_lev = raw_cap / 100.0 if raw_cap >= 100.0 else raw_cap
                        cap_lev = float(max(1.0, min(float(self.max_leverage or cap_lev), cap_lev)))
                        suggested_lev = float(cap_lev)
                    except Exception:
                        suggested_lev = None
                if suggested_lev is not None and abs(float(suggested_lev) - float(lev)) > 1e-6:
                    try:
                        await self._ccxt_call("set_leverage", self.exchange.set_leverage, float(suggested_lev), symbol)
                        self._last_leverage_sync_ms_by_sym[symbol] = now
                        self._last_leverage_target_by_sym[symbol] = float(suggested_lev)
                        self._leverage_sync_fail_by_sym.pop(symbol, None)
                        self._log(
                            f"[EXCHANGE] leverage capped by risk-limit: {symbol} "
                            f"{lev:.1f}x -> {suggested_lev:.1f}x"
                        )
                        return True
                    except Exception as e3:
                        err_text = f"{err_text} | risk_cap_fallback_err={e3}"
            if "110043" in err_text or "leverage not modified" in err_text:
                self._last_leverage_sync_ms_by_sym[symbol] = now
                self._last_leverage_target_by_sym[symbol] = float(lev)
                self._leverage_sync_fail_by_sym.pop(symbol, None)
                return True
            self._leverage_sync_fail_by_sym[symbol] = {
                "ts": now,
                "target": float(lev),
                "suggested_leverage": suggested_lev,
                "err": err_text,
            }
            self._log_err(f"[EXCHANGE] set_leverage failed: {symbol} target={lev:.1f}x err={e}")
            return False

    async def _maybe_sync_legacy_5x_deleverage(self, ts_ms: int) -> None:
        """Gradually step down residual high-leverage positions (e.g. stale 5x) without waiting for new entries."""
        try:
            enabled = str(os.environ.get("LEGACY_5X_DELEVERAGE_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            enabled = True
        if not enabled:
            return
        try:
            from_lev = float(os.environ.get("LEGACY_5X_DELEVERAGE_FROM", 4.9) or 4.9)
        except Exception:
            from_lev = 4.9
        try:
            min_gap = float(os.environ.get("LEGACY_5X_DELEVERAGE_MIN_GAP", 0.20) or 0.20)
        except Exception:
            min_gap = 0.20
        try:
            step_lev = float(os.environ.get("LEGACY_5X_DELEVERAGE_STEP", 1.0) or 1.0)
        except Exception:
            step_lev = 1.0
        try:
            interval_sec = float(os.environ.get("LEGACY_5X_DELEVERAGE_INTERVAL_SEC", 120.0) or 120.0)
        except Exception:
            interval_sec = 120.0
        try:
            max_per_cycle = int(os.environ.get("LEGACY_5X_DELEVERAGE_MAX_PER_CYCLE", 2) or 2)
        except Exception:
            max_per_cycle = 2
        max_per_cycle = max(1, min(10, int(max_per_cycle)))
        interval_ms = int(max(0.0, float(interval_sec)) * 1000.0)
        try:
            fallback_target = float(
                os.environ.get("LEGACY_5X_DELEVERAGE_FALLBACK_TARGET", os.environ.get("LEVERAGE_TARGET_MIN", os.environ.get("LEVERAGE_MIN", 2.0)))
                or 2.0
            )
        except Exception:
            fallback_target = 2.0
        fallback_target = max(1.0, float(fallback_target))
        min_gap = max(0.05, float(min_gap))
        step_lev = max(min_gap, float(step_lev))

        candidates: list[tuple[float, str, float, float, float]] = []
        for sym, pos in list((self.positions or {}).items()):
            if not isinstance(pos, dict):
                continue
            if not self._is_managed_position(pos):
                continue
            qty = self._safe_float(pos.get("quantity", pos.get("qty", 0.0)), 0.0) or 0.0
            if abs(float(qty)) <= 0.0:
                continue
            curr_lev = self._safe_float(pos.get("leverage"), None)
            if curr_lev is None or float(curr_lev) <= 0.0:
                curr_lev = self._safe_float(self._dyn_leverage.get(sym), self.leverage)
            curr_lev = float(curr_lev or 0.0)
            if curr_lev < float(from_lev):
                continue
            dyn_target = self._safe_float(self._dyn_leverage.get(sym), None)
            if dyn_target is None or dyn_target <= 0.0:
                dyn_target = fallback_target
            desired = float(max(fallback_target, min(curr_lev, float(dyn_target))))
            if desired >= (curr_lev - min_gap):
                continue
            last_ms = int(self._legacy_delev_last_sync_ms_by_sym.get(sym, 0) or 0)
            if interval_ms > 0 and (int(ts_ms) - last_ms) < interval_ms:
                continue
            step_target = float(max(desired, curr_lev - step_lev))
            if step_target >= (curr_lev - 1e-6):
                continue
            gap = float(curr_lev - step_target)
            candidates.append((gap, sym, curr_lev, step_target, desired))

        if not candidates:
            return
        candidates.sort(key=lambda x: float(x[0]), reverse=True)
        for _gap, sym, curr_lev, step_target, desired in candidates[:max_per_cycle]:
            self._legacy_delev_last_sync_ms_by_sym[sym] = int(ts_ms)
            ok = await self._sync_symbol_leverage(sym, float(step_target), force=True)
            if ok:
                pos = self.positions.get(sym)
                if isinstance(pos, dict):
                    pos["leverage"] = float(step_target)
                self._dyn_leverage[sym] = float(step_target)
                self._log(
                    f"[LEGACY_DELEV] {sym} leverage {curr_lev:.2f}x -> {step_target:.2f}x "
                    f"(target={desired:.2f}x)"
                )
            else:
                self._log_err(
                    f"[LEGACY_DELEV] set_leverage failed: {sym} current={curr_lev:.2f}x "
                    f"step_target={step_target:.2f}x desired={desired:.2f}x"
                )

    def _enter_position(
        self,
        sym: str,
        side: str,
        price: float,
        decision: dict,
        ts: int,
        ctx: dict | None = None,
        *,
        size_frac_override: float | None = None,
        hold_limit_override: int | None = None,
        tag: str | None = None,
        leverage_override: float | None = None,
    ):
        lev = leverage_override if leverage_override is not None else self.leverage
        override_filters = self._extract_entry_overrides(decision)
        ignore_caps = bool({"cap", "cap_exposure", "cap_positions"} & set(override_filters))
        size_frac, notional, qty = self._calc_position_size(
            decision,
            price,
            lev,
            size_frac_override=size_frac_override,
            symbol=sym,
            use_cycle_reserve=True,
            ignore_caps=ignore_caps,
        )
        qty = self._precision_aware_entry_quantity(
            sym,
            side,
            qty,
            lev,
            price_hint=price,
            context="entry_plan",
            regime_hint=((ctx or {}).get("regime") or ((decision.get("meta") or {}).get("regime") if decision else None)),
        )
        notional = float(max(0.0, qty * float(price))) if qty > 0 and price else 0.0
        regime_hint_entry = (ctx or {}).get("regime") or ((decision.get("meta") or {}).get("regime") if decision else None)
        min_entry_notional, regime_bucket = self._regime_entry_min_notional(regime_hint_entry)
        if min_entry_notional > 0.0 and notional < min_entry_notional:
            self._log(
                f"[ENTRY_SKIP_MIN_NOTIONAL] {sym} notional={notional:.4f} < floor={min_entry_notional:.4f} "
                f"(qty={qty:.8f}, px={price:.8f}, regime={regime_bucket})"
            )
            return
        can_enter, reason = self._can_enter_position(notional)
        if (not can_enter) and ignore_caps and reason == "exposure capped":
            can_enter = True
            reason = ""
        if not can_enter or qty <= 0:
            self._log(f"[{sym}] skip entry ({reason})")
            return

        # Reserve margin for this cycle to avoid oversubscription on live orders.
        self._reserve_cycle_margin(notional, lev)

        ctx = ctx or {}

        meta = (decision.get("meta") or {}) if decision else {}
        if not isinstance(meta, dict):
            meta = {}
        if not meta and decision:
            details = decision.get("details")
            if isinstance(details, list):
                for d in details:
                    if not isinstance(d, dict):
                        continue
                    m2 = d.get("meta")
                    if isinstance(m2, dict) and m2:
                        meta = dict(m2)
                        break
        if decision and not isinstance(decision.get("meta"), dict) and meta:
            decision["meta"] = meta

        def _dget(name: str, default=None):
            if not decision:
                return default
            if isinstance(meta, dict) and (meta.get(name) is not None):
                return meta.get(name)
            val = decision.get(name)
            return default if val is None else val

        entry_score = None
        entry_hold_score = None
        entry_floor = None
        entry_quality_score = None
        one_way_move_score = None
        leverage_signal_score = None
        opt_hold_sec = None
        opt_hold_src = None
        if decision:
            entry_score = decision.get("unified_score")
            if entry_score is None:
                entry_score = meta.get("unified_score")
            if entry_score is None:
                entry_score = meta.get("hybrid_score")
            entry_hold_score = meta.get("unified_score_hold")
            if entry_hold_score is None:
                entry_hold_score = meta.get("hybrid_score_hold")
            entry_floor = meta.get("entry_floor_eff")
            if entry_floor is None:
                entry_floor = meta.get("hybrid_entry_floor")
            if entry_floor is None:
                entry_floor = meta.get("score_threshold")
            opt_hold_sec = meta.get("opt_hold_sec")
            if opt_hold_sec is None:
                opt_hold_sec = meta.get("unified_t_star")
            if opt_hold_sec is None:
                opt_hold_sec = meta.get("policy_horizon_eff_sec") or meta.get("best_h")
            try:
                if opt_hold_sec is not None:
                    opt_hold_sec = float(opt_hold_sec)
                    if not math.isfinite(opt_hold_sec) or opt_hold_sec <= 0:
                        opt_hold_sec = None
            except Exception:
                opt_hold_sec = None
            if opt_hold_sec is not None:
                opt_hold_src = "entry_meta"
            entry_quality_score = _dget("entry_quality_score")
            one_way_move_score = _dget("one_way_move_score")
            leverage_signal_score = _dget("leverage_signal_score")

        def _clip01(x: float) -> float:
            return float(max(0.0, min(1.0, float(x))))

        if entry_quality_score is None or float(self._safe_float(entry_quality_score, 0.0) or 0.0) <= 0.0:
            try:
                edge_ref = float(os.environ.get("LEVERAGE_ENTRY_QUALITY_EDGE_REF", 0.16) or 0.16)
            except Exception:
                edge_ref = 0.16
            edge_ref = max(edge_ref, 1e-6)
            try:
                gap_ref = float(os.environ.get("LEVERAGE_ENTRY_QUALITY_EV_GAP_REF", 0.0030) or 0.0030)
            except Exception:
                gap_ref = 0.0030
            gap_ref = max(gap_ref, 1e-6)
            conf_v = float(max(0.0, min(1.0, self._safe_float(decision.get("confidence") if decision else 0.0, 0.0) or 0.0)))
            dir_conf_v = float(max(0.0, min(1.0, self._safe_float(_dget("mu_dir_conf"), conf_v) or conf_v)))
            dir_edge_v = abs(float(self._safe_float(_dget("mu_dir_edge"), 0.0) or 0.0))
            ev_gap_v = abs(float(self._safe_float(_dget("policy_ev_gap", _dget("ev_gap", 0.0)), 0.0) or 0.0))
            edge_n = _clip01(dir_edge_v / edge_ref)
            gap_n = _clip01(ev_gap_v / gap_ref)
            entry_quality_score = _clip01(0.45 * conf_v + 0.30 * dir_conf_v + 0.15 * edge_n + 0.10 * gap_n)
        else:
            entry_quality_score = _clip01(self._safe_float(entry_quality_score, 0.0) or 0.0)

        if one_way_move_score is None or float(self._safe_float(one_way_move_score, 0.0) or 0.0) <= 0.0:
            tick_trend = float(self._safe_float((ctx or {}).get("tick_trend"), 0.0) or 0.0)
            tick_breakout_active = bool((ctx or {}).get("tick_breakout_active"))
            tick_breakout_dir = float(self._safe_float((ctx or {}).get("tick_breakout_dir"), 0.0) or 0.0)
            tick_breakout_score = abs(float(self._safe_float((ctx or {}).get("tick_breakout_score"), 0.0) or 0.0))
            side_sign = 1.0 if str(side).upper() == "LONG" else -1.0
            try:
                breakout_ref = float(os.environ.get("LEVERAGE_ONEWAY_BREAKOUT_REF", 0.0012) or 0.0012)
            except Exception:
                breakout_ref = 0.0012
            breakout_ref = max(breakout_ref, 1e-8)
            try:
                trend_ref = float(os.environ.get("LEVERAGE_ONEWAY_TREND_REF", 1.5) or 1.5)
            except Exception:
                trend_ref = 1.5
            trend_ref = max(trend_ref, 1e-6)
            breakout_sign = side_sign * tick_breakout_dir
            breakout_n = _clip01((tick_breakout_score / breakout_ref) * (1.0 if (tick_breakout_active and breakout_sign > 0.0) else 0.0))
            trend_n = _clip01(0.5 + 0.5 * math.tanh((side_sign * tick_trend) / trend_ref))
            one_way_move_score = _clip01(0.55 * breakout_n + 0.45 * trend_n)
        else:
            one_way_move_score = _clip01(self._safe_float(one_way_move_score, 0.0) or 0.0)

        if leverage_signal_score is None or float(self._safe_float(leverage_signal_score, 0.0) or 0.0) <= 0.0:
            try:
                q_w = float(os.environ.get("LEVERAGE_SIGNAL_QUALITY_WEIGHT", 0.62) or 0.62)
            except Exception:
                q_w = 0.62
            q_w = float(max(0.0, min(1.0, q_w)))
            leverage_signal_score = _clip01(q_w * float(entry_quality_score) + (1.0 - q_w) * float(one_way_move_score))
        else:
            leverage_signal_score = _clip01(self._safe_float(leverage_signal_score, 0.0) or 0.0)
        try:
            lev_diag_on = str(os.environ.get("LEVERAGE_DIAG_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            lev_diag_on = True
        if lev_diag_on and float(lev) <= 1.000001:
            self._log(
                f"[LEV_ENTRY] {sym} lev={float(lev):.2f}x d_lev={self._safe_float(decision.get('leverage') if decision else None, 0.0):.2f} "
                f"eq={float(entry_quality_score):.3f} ls={float(leverage_signal_score):.3f} "
                f"meta_lev_source={_dget('lev_source')} meta_lev_raw={self._safe_float(_dget('lev_raw_before_caps'), 0.0):.3f} "
                f"meta_lev_sig_tgt={self._safe_float(_dget('lev_signal_target'), 0.0):.3f}"
            )
        opt_hold_entry_sec = int(round(opt_hold_sec)) if opt_hold_sec is not None else None
        entry_link_id = self._new_trade_uid()
        pos = {
            "symbol": sym,
            "side": side,
            "entry_price": float(price),
            "entry_time": ts,
            "quantity": qty,
            "notional": notional,
            "size_frac": size_frac,
            "tag": tag,
            "leverage": lev,
            "entry_leverage": lev,
            "entry_notional": notional,
            "cap_frac": float(notional / self.balance) if self.balance else 0.0,
            "fee_paid": notional * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker),
            "order_status": "submitted" if self.enable_orders else "paper",
            "order_submit_ts": now_ms(),
            "regime": (ctx or {}).get("regime") or _dget("regime"),
            "session": (ctx or {}).get("session") or _dget("session"),
            "entry_id": entry_link_id,
            "entry_link_id": entry_link_id,
            # 예측 스냅샷
            "pred_win": decision.get("confidence") if decision else None,
            "pred_ev": decision.get("ev") if decision else None,
            "pred_event_ev_r": _dget("event_ev_r"),
            "pred_event_p_tp": _dget("event_p_tp"),
            "pred_event_p_sl": _dget("event_p_sl"),
            "pred_mu_alpha": _dget("mu_alpha"),
            "pred_mu_alpha_raw": _dget("mu_alpha_raw"),
            "pred_mu_dir_conf": _dget("mu_dir_conf"),
            "pred_mu_dir_edge": _dget("mu_dir_edge"),
            "pred_mu_dir_prob_long": _dget("mu_dir_prob_long"),
            "pred_hmm_state": (ctx or {}).get("hmm_state") if decision else None,
            "pred_hmm_conf": (ctx or {}).get("hmm_conf") if decision else None,
            "alpha_vpin": (ctx or {}).get("vpin") if decision else None,
            "alpha_hurst": (ctx or {}).get("hurst") if decision else None,
            "consensus_used": self._consensus_used_flag(decision),
            "entry_score": entry_score,
            "entry_score_hold": entry_hold_score,
            "entry_floor": entry_floor,
            "entry_quality_score": float(entry_quality_score) if entry_quality_score is not None else None,
            "one_way_move_score": float(one_way_move_score) if one_way_move_score is not None else None,
            "leverage_signal_score": float(leverage_signal_score) if leverage_signal_score is not None else None,
            "policy_score_threshold": _dget("policy_score_threshold_eff", _dget("policy_score_threshold")),
            "policy_event_exit_min_score": _dget("event_exit_min_score"),
            "policy_unrealized_dd_floor": _dget("unrealized_dd_floor_dyn"),
            "lev_source": _dget("lev_source"),
            "lev_raw_before_caps": _dget("lev_raw_before_caps"),
            "lev_raw_ev_component": _dget("lev_raw_ev_component"),
            "lev_signal_target": _dget("lev_signal_target"),
            "lev_signal_blend": _dget("lev_signal_blend"),
            "lev_target_min": _dget("lev_target_min"),
            "lev_target_max": _dget("lev_target_max"),
            "lev_target_max_base": _dget("lev_target_max_base"),
            "lev_liq_cap": _dget("lev_liq_cap"),
            "lev_dynamic_scale": _dget("lev_dynamic_scale"),
            "lev_dynamic_tox": _dget("lev_dynamic_tox"),
            "lev_dynamic_conf_deficit": _dget("lev_dynamic_conf_deficit"),
            "lev_dynamic_sigma_stress": _dget("lev_dynamic_sigma_stress"),
            "lev_dynamic_vpin": _dget("lev_dynamic_vpin"),
            "lev_dynamic_vpin_for_lev": _dget("lev_dynamic_vpin_for_lev"),
            "lev_dynamic_hurst": _dget("lev_dynamic_hurst"),
            "lev_dynamic_mu_align": _dget("lev_dynamic_mu_align"),
            "lev_balance_reject_scale": _dget("lev_balance_reject_scale"),
            "lev_balance_reject_count": _dget("lev_balance_reject_count"),
            "lev_balance_reject_min_count": _dget("lev_balance_reject_min_count"),
            "lev_balance_reject_min_scale": _dget("lev_balance_reject_min_scale"),
            "lev_balance_reject_relief_signal": _dget("lev_balance_reject_relief_signal"),
            "lev_balance_reject_relief_conf": _dget("lev_balance_reject_relief_conf"),
            "lev_balance_reject_relief_pow": _dget("lev_balance_reject_relief_pow"),
            "lev_sigma_raw": _dget("lev_sigma_raw"),
            "lev_sigma_used": _dget("lev_sigma_used"),
            "lev_sigma_stress_clip": _dget("lev_sigma_stress_clip"),
            "opt_hold_sec": int(round(opt_hold_sec)) if opt_hold_sec is not None else None,
            "opt_hold_src": opt_hold_src,
            "opt_hold_entry_sec": opt_hold_entry_sec,
            "opt_hold_entry_src": opt_hold_src,
            "opt_hold_curr_sec": int(round(opt_hold_sec)) if opt_hold_sec is not None else None,
            "opt_hold_curr_remaining_sec": int(round(opt_hold_sec)) if opt_hold_sec is not None else None,
            "opt_hold_curr_src": opt_hold_src,
        }
        self._capture_position_observability(pos, decision=decision, ctx=ctx)
        # 진입 수수료 선반영
        fee_entry = pos["fee_paid"]
        self.balance -= fee_entry
        self.positions[sym] = pos
        self._rebalance_on_next_cycle = True
        self._log(
            f"[{sym}] ENTER {side} qty={qty:.4f} notional={notional:.2f} lev={float(lev):.2f}x "
            f"fee={fee_entry:.4f} size={size_frac:.2%} tag={tag or '-'}"
        )
        self._last_actions[sym] = "ENTER"
        try:
            self._last_open_ts[sym] = int(ts)
        except Exception:
            pass
        self._maybe_place_order(sym, side, qty, reduce_only=False, position_side=side)
        entry_type = "SPREAD" if tag == "spread" else "ENTER"
        self._record_trade(entry_type, sym, side, price, qty, pos, fee=fee_entry)
        self._persist_state(force=True)

    def _close_position(self, sym: str, price: float, reason: str, exit_kind: str = "MANUAL", *, skip_order: bool = False):
        pos = self.positions.pop(sym, None)
        if not pos or price is None:
            return
        qty = float(pos.get("quantity", 0.0))
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        notional_entry = float(pos.get("notional", 0.0))
        fee_exit = price * qty * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker)
        pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        pnl_net = pnl - fee_exit
        self.balance += pnl_net
        self._log(f"[{sym}] EXIT {side} qty={qty:.4f} pnl={pnl_net:.2f} fee={fee_exit:.4f} ({reason})")
        self._last_actions[sym] = "EXIT"
        exit_side = "SHORT" if side == "LONG" else "LONG"
        if not skip_order:
            self._maybe_place_order(sym, exit_side, qty, reduce_only=True, position_side=side)
        realized_r = pnl_net / notional_entry if notional_entry else 0.0
        hit = 1 if pnl > 0 else 0
        self._record_trade(
            "EXIT",
            sym,
            side,
            price,
            qty,
            pos,
            pnl=pnl_net,
            fee=fee_exit,
            reason=reason,
            realized_r=realized_r,
            hit=hit,
            exit_kind=exit_kind,
        )
        # 재진입 쿨다운 (종류별)
        self._mark_exit_and_cooldown(sym, exit_kind=exit_kind, ts_ms=now_ms())
        try:
            self._last_close_ts[sym] = int(now_ms())
        except Exception:
            pass
        # 예측 vs 실제 기록
        pred_win = pos.get("pred_win")
        pred_ev = pos.get("pred_ev")
        pred_event_ev_r = pos.get("pred_event_ev_r")
        pred_event_p_tp = pos.get("pred_event_p_tp")
        pred_event_p_sl = pos.get("pred_event_p_sl")
        brier_sl = None
        if pred_event_p_sl is not None:
            try:
                brier_sl = (float(pred_event_p_sl) - (1 - hit)) ** 2
            except Exception:
                brier_sl = None
        self.eval_history.append({
            "hit": hit,
            "pred_win": pred_win,
            "pred_ev": pred_ev,
            "pred_event_ev_r": pred_event_ev_r,
            "pred_event_p_tp": pred_event_p_tp,
            "pred_event_p_sl": pred_event_p_sl,
            "realized_r": realized_r,
            "brier_sl": brier_sl,
        })
        try:
            log_payload = {
                "regime": pos.get("regime"),
                "session": pos.get("session"),
                "EV": pred_ev,
                "event_p_tp": pred_event_p_tp,
                "event_p_sl": pred_event_p_sl,
                "realized_r": realized_r,
                "consensus_used": pos.get("consensus_used"),
            }
            self._log(f"[METRIC] {json.dumps(log_payload)}")
        except Exception:
            pass
        self._persist_state(force=True)

    def _classify_external_close_reason(
        self,
        sym: str,
        pos: dict,
        *,
        miss_cnt: int = 0,
        source: str = "fetch_positions_loop",
        now_ts: int | None = None,
    ) -> tuple[str, str, dict]:
        now_ts = int(now_ts or now_ms())
        cause = "manual_cleanup"
        reason = "exchange_close_manual_cleanup"
        detail = {
            "symbol": sym,
            "source": source,
            "miss_cnt": int(miss_cnt or 0),
            "live_miss_cnt": int(self._live_missing_pos_counts.get(sym, 0) or 0),
            "safety_mode": bool(getattr(self, "safety_mode", False)),
            "pos_source": (pos or {}).get("pos_source"),
            "managed": bool((pos or {}).get("managed", True)),
        }
        # Positions synced/imported from exchange and not actively managed by bot are treated as sync-origin closes.
        if (not bool((pos or {}).get("managed", True))) or str((pos or {}).get("pos_source") or "").lower().startswith("exchange_sync"):
            cause = "external_sync"
            reason = "exchange_close_external_sync"

        try:
            liq_need = int(os.environ.get("LIVE_LIQUIDATION_MISS_COUNT", 2) or 2)
        except Exception:
            liq_need = 2
        liq_need = max(1, liq_need)
        live_miss_next = int(self._live_missing_pos_counts.get(sym, 0) or 0) + 1

        mark_px = self._safe_float(
            (self.market.get(sym) or {}).get("price")
            or (pos or {}).get("current_price")
            or (pos or {}).get("entry_price"),
            None,
        )
        liq_px = self._safe_float((pos or {}).get("liq_price"), None)
        side = str((pos or {}).get("side") or "LONG").upper()
        near_liq = False
        liq_gap = None
        try:
            liq_buf = float(os.environ.get("EXTERNAL_CLOSE_LIQ_BUFFER_PCT", 0.0030) or 0.0030)
        except Exception:
            liq_buf = 0.0030
        liq_buf = max(0.0, float(liq_buf))
        if liq_px is not None and mark_px is not None and mark_px > 0:
            try:
                liq_gap = abs(float(mark_px) - float(liq_px)) / max(abs(float(mark_px)), 1e-9)
            except Exception:
                liq_gap = None
            if side == "LONG":
                near_liq = float(mark_px) <= float(liq_px) * (1.0 + liq_buf)
            else:
                near_liq = float(mark_px) >= float(liq_px) * (1.0 - liq_buf)

        risk_forced = bool(getattr(self, "safety_mode", False)) or (live_miss_next >= liq_need) or bool(near_liq)
        if risk_forced:
            cause = "risk_forced"
            reason = "exchange_close_risk_forced"

        detail.update(
            {
                "reason": reason,
                "cause": cause,
                "live_miss_next": int(live_miss_next),
                "live_liq_miss_need": int(liq_need),
                "liq_price": liq_px,
                "mark_price": mark_px,
                "near_liq": bool(near_liq),
                "liq_gap": liq_gap,
                "side": side,
                "ts": int(now_ts),
            }
        )
        return cause, reason, detail

    def _record_external_close(
        self,
        sym: str,
        price: float,
        reason: str = "exchange_manual_close",
        *,
        cause: str | None = None,
        source: str = "fetch_positions_loop",
        detail: dict | None = None,
    ):
        """Record exchange-side close without touching wallet balance."""
        pos = self.positions.pop(sym, None)
        if not pos or price is None:
            return
        try:
            qty = float(pos.get("quantity", 0.0))
        except Exception:
            qty = 0.0
        if qty <= 0:
            return
        try:
            entry = float(pos.get("entry_price", price))
        except Exception:
            entry = float(price)
        side = str(pos.get("side") or "LONG")
        external_cause = str(cause or "manual_cleanup")
        detail_payload = dict(detail or {})
        if not detail_payload:
            try:
                ext_cause, ext_reason, ext_detail = self._classify_external_close_reason(
                    sym,
                    pos,
                    miss_cnt=0,
                    source=source,
                    now_ts=now_ms(),
                )
                external_cause = ext_cause
                reason = str(reason or ext_reason)
                detail_payload = dict(ext_detail or {})
            except Exception:
                detail_payload = {}
        pos["external_close_cause"] = external_cause
        pos["external_close_source"] = str(source)
        pos["external_close_detail"] = detail_payload
        if detail_payload.get("miss_cnt") is not None:
            pos["external_close_miss_cnt"] = detail_payload.get("miss_cnt")
        pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        try:
            entry_t = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
            pos["age_sec"] = max(0.0, (now_ms() - int(entry_t)) / 1000.0)
        except Exception:
            pass
        self._record_trade("EXIT", sym, side, price, qty, pos, pnl=float(pnl), fee=0.0, reason=reason, exit_kind="EXTERNAL")
        try:
            if self.db is not None:
                evt_type = f"exchange_close_{external_cause}"
                snap = dict(pos)
                snap["reason"] = reason
                snap["external_close_cause"] = external_cause
                snap["external_close_source"] = source
                snap["external_close_detail"] = detail_payload
                self.db.log_position_event(sym, evt_type, snap, mode=self._trading_mode)
        except Exception:
            pass
        if external_cause == "risk_forced":
            self._register_anomaly(
                "exchange_close_risk",
                "critical",
                f"{sym} exchange-side risk-forced close ({reason})",
                detail_payload or {"symbol": sym, "reason": reason},
            )
        elif external_cause == "manual_cleanup":
            self._register_anomaly(
                "exchange_close_manual",
                "warn",
                f"{sym} exchange-side manual cleanup close",
                detail_payload or {"symbol": sym, "reason": reason},
            )
        else:
            self._register_anomaly(
                "exchange_close_sync",
                "warn",
                f"{sym} exchange sync close ({source})",
                detail_payload or {"symbol": sym, "reason": reason},
            )
        try:
            self._last_actions[sym] = "SYNC_CLOSE"
            self._last_close_ts[sym] = now_ms()
            self._live_missing_pos_counts.pop(sym, None)
            self._external_close_missing_counts.pop(sym, None)
        except Exception:
            pass
        self._persist_state(force=True)

    def _liquidate_all_positions(self):
        if not self.positions:
            return
        self.safety_mode = True
        self._log_err("[RISK] Emergency liquidation triggered")
        for sym in list(self.positions.keys()):
            pos = self.positions.get(sym)
            if not pos:
                continue
            px = self.market.get(sym, {}).get("price")
            if px is None:
                self._log_err(f"[RISK] skip liquidation (no price): {sym}")
                continue
            self._close_position(sym, float(px), "emergency_stop", exit_kind="KILL")

    def _rebalance_position(self, sym: str, price: float, decision: dict, leverage_override: float | None = None):
        """
        기존 포지션이 있을 때 목표 비중과 현 포지션이 크게 다르면 수량/노출을 조정한다.
        실제 주문은 ENABLE_LIVE_ORDERS에 따라 별도 처리.
        """
        if not REBALANCE_ENABLED:
            return
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        lev = leverage_override if leverage_override is not None else pos.get("leverage", self.leverage)
        target_size_frac, target_notional, target_qty = self._calc_position_size(decision, price, lev, symbol=sym)
        if target_notional <= 0:
            # 목표 노출이 0이면 전량 청산
            self._close_position(sym, price, "rebalance to zero")
            self._last_rebalance_ts[sym] = time.time()
            return
        curr_notional = float(pos.get("notional", 0.0))
        now = time.time()
        cap_forced_reduce = False
        # Enforce total exposure cap during rebalances as well
        if self.exposure_cap_enabled:
            try:
                cap_balance = float(self._exposure_cap_balance())
            except Exception:
                cap_balance = float(self.balance or 0.0)
            try:
                cap_total = float(cap_balance) * float(self.max_notional_frac)
            except Exception:
                cap_total = float(cap_balance) * float(self.max_notional_frac or 0.0)
            try:
                open_notional = float(self._total_open_notional() or 0.0)
            except Exception:
                open_notional = 0.0
            # Allow this position to use remaining capacity (remove its current notional first).
            cap_remaining = cap_total - max(0.0, (open_notional - curr_notional))
            if cap_remaining < 0:
                cap_remaining = 0.0
            if target_notional > cap_remaining:
                prev_target_notional = float(target_notional)
                target_notional = float(cap_remaining)
                target_qty = float(target_notional / price) if price and target_notional > 0 else 0.0
                if target_notional < prev_target_notional:
                    cap_forced_reduce = True
                try:
                    base_balance = float(self._sizing_balance())
                except Exception:
                    base_balance = float(self.balance or 0.0)
                if base_balance > 0 and lev:
                    target_size_frac = float(target_notional) / max(base_balance * max(float(lev), 1e-6), 1e-6)
        sym_thr = float(REBALANCE_THRESHOLD_MAP.get(sym, REBALANCE_THRESHOLD_FRAC))
        sym_min_interval = float(REBALANCE_MIN_INTERVAL_MAP.get(sym, REBALANCE_MIN_INTERVAL_SEC) or 0.0)
        sym_min_notional = float(REBALANCE_MIN_NOTIONAL_MAP.get(sym, REBALANCE_MIN_NOTIONAL) or 0.0)
        last_ts = self._last_rebalance_ts.get(sym)
        if sym_min_interval > 0 and last_ts is not None:
            if (now - float(last_ts)) < float(sym_min_interval):
                return
        abs_delta = abs(target_notional - curr_notional)
        if sym_min_notional > 0 and abs_delta < float(sym_min_notional):
            return
        delta = abs_delta / curr_notional if curr_notional else 1.0
        # 항상 레버리지/메타 업데이트
        if leverage_override is not None:
            if self.enable_orders:
                pos["target_leverage"] = float(leverage_override)
            else:
                pos["leverage"] = float(leverage_override)
        if delta < sym_thr:
            return
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        curr_qty = float(pos.get("quantity", 0.0))
        delta_qty = float(target_qty) - float(curr_qty)

        # Live mode: place rebalance orders, but do not mutate local qty/notional or book synthetic PnL.
        # Exchange sync should be the source of truth for realized changes.
        try:
            live_order_only = self.enable_orders and (
                str(os.environ.get("LIVE_REBALANCE_ORDER_ONLY", "1")).strip().lower() in ("1", "true", "yes", "on")
            )
        except Exception:
            live_order_only = bool(self.enable_orders)
        if live_order_only:
            if abs(float(delta_qty)) <= 1e-12:
                return
            if float(delta_qty) < 0.0:
                close_qty = self._to_amount_precision(sym, abs(float(delta_qty)), up=False)
                if close_qty > 0:
                    exit_side = "SHORT" if side == "LONG" else "LONG"
                    self._maybe_place_order(sym, exit_side, close_qty, reduce_only=True, position_side=side)
                    self._log(
                        f"[{sym}] REBAL_ORDER reduce qty={close_qty:.6f} "
                        f"target_notional={target_notional:.2f} curr_notional={curr_notional:.2f}"
                    )
            else:
                add_qty = self._to_amount_precision(sym, float(delta_qty), up=False)
                if add_qty > 0:
                    self._maybe_place_order(sym, side, add_qty, reduce_only=False, position_side=side)
                    self._log(
                        f"[{sym}] REBAL_ORDER add qty={add_qty:.6f} "
                        f"target_notional={target_notional:.2f} curr_notional={curr_notional:.2f}"
                    )
            pos["target_notional"] = float(target_notional)
            pos["target_size_frac"] = float(target_size_frac)
            pos["target_qty"] = float(target_qty)
            self._last_actions[sym] = "REBAL_ORDER"
            self._last_rebalance_ts[sym] = now
            self._persist_state(force=False)
            return

        if target_notional < curr_notional and curr_notional > 0:
            # Avoid premature partial exits from rebalance unless exposure-cap forces a cut.
            try:
                reb_respect_min_hold = str(os.environ.get("REBALANCE_RESPECT_MIN_HOLD", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                reb_respect_min_hold = True
            if reb_respect_min_hold and not cap_forced_reduce:
                min_hold_sec_eff = self._effective_min_hold_sec(pos)
                if min_hold_sec_eff > 0:
                    try:
                        entry_ts_ms = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
                        age_sec = max(0.0, (now_ms() - int(entry_ts_ms)) / 1000.0)
                    except Exception:
                        age_sec = 0.0
                    if age_sec < float(min_hold_sec_eff):
                        self._last_rebalance_ts[sym] = now
                        return
            # 부분 청산: 줄이는 비율만큼 실현 손익을 balance에 반영
            reduce_ratio = 1.0 - (target_notional / curr_notional)
            close_qty = curr_qty * reduce_ratio
            close_notional = curr_notional * reduce_ratio
            entry_fee_total = float(pos.get("fee_paid", 0.0) or 0.0)
            entry_fee_alloc = entry_fee_total * (close_notional / curr_notional) if curr_notional > 0 else 0.0
            pnl_realized = (price - entry) * close_qty if side == "LONG" else (entry - price) * close_qty
            fee_partial = close_notional * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker)
            pnl_realized_net = pnl_realized - fee_partial
            self.balance += pnl_realized_net

            # 남은 포지션 업데이트
            pos["quantity"] = max(curr_qty - close_qty, 0.0)
            pos["notional"] = max(target_notional, 0.0)
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(pos["notional"] / self.balance) if self.balance else 0.0
            if entry_fee_total:
                pos["fee_paid"] = max(0.0, entry_fee_total - entry_fee_alloc)

            # 부분 청산 기록
            close_pos = dict(pos)
            close_pos["notional"] = close_notional
            close_pos["quantity"] = close_qty
            close_pos["leverage"] = float(pos.get("leverage", lev) or lev)
            close_pos["entry_leverage"] = pos.get("entry_leverage")
            close_pos["fee_paid"] = entry_fee_alloc
            self._log(f"[{sym}] PARTIAL EXIT by REBAL qty={close_qty:.4f} pnl={pnl_realized_net:.2f} fee={fee_partial:.4f}")
            self._last_actions[sym] = "REBAL_EXIT"
            self._record_trade("REBAL_EXIT", sym, side, price, close_qty, close_pos, pnl=pnl_realized_net, fee=fee_partial, reason="rebalance partial")
            self._cooldown_until[sym] = 0
            self._last_rebalance_ts[sym] = now
        else:
            # 노출 확대/동일: 포지션만 갱신
            pos["notional"] = target_notional
            pos["quantity"] = target_qty
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(target_notional / self.balance) if self.balance else 0.0
            self._log(f"[{sym}] REBALANCE qty={target_qty:.4f} notional={target_notional:.2f} size={target_size_frac:.2%}")
            self._last_actions[sym] = "REBAL"
            # 기록용 스냅샷
            pnl_now = (price - entry) * target_qty if side == "LONG" else (entry - price) * target_qty
            self._record_trade("REBAL", sym, side, price, target_qty, pos, pnl=pnl_now, reason="rebalance")
            self._last_rebalance_ts[sym] = now

        # 실제 리밸런싱 주문 경로 (옵션)
        adj_qty = max(0.0, target_qty - float(pos.get("quantity", 0.0)))
        if adj_qty > 0:
            self._maybe_place_order(sym, pos["side"], adj_qty, reduce_only=False, position_side=pos["side"])
        self._persist_state(force=True)

    def _maybe_exit_position(
        self,
        sym: str,
        price: float,
        decision: dict,
        ts: int,
        *,
        allow_extra_exits: bool = True,
        ctx: dict | None = None,
    ):
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        # Exchange liquidation price hard stop (if available)
        try:
            use_ex_liq = str(os.environ.get("EXCHANGE_LIQ_SL", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_ex_liq = True
        if use_ex_liq:
            liq_price = pos.get("liq_price")
            if liq_price is None:
                liq_price = pos.get("liqPrice")
            if liq_price is None:
                try:
                    liq_price = (pos.get("meta") or {}).get("liq_price")
                except Exception:
                    liq_price = None
            try:
                liq_price = float(liq_price) if liq_price is not None else None
            except Exception:
                liq_price = None
            if liq_price is not None and liq_price > 0:
                try:
                    buf_pct = float(os.environ.get("EXCHANGE_LIQ_SL_BUFFER_PCT", 0.0) or 0.0)
                except Exception:
                    buf_pct = 0.0
                side = str(pos.get("side") or "")
                if side == "LONG":
                    trigger = float(price) <= float(liq_price) * (1.0 + max(0.0, buf_pct))
                else:
                    trigger = float(price) >= float(liq_price) * (1.0 - max(0.0, buf_pct))
                if trigger:
                    self._close_position(sym, float(price), "exchange_liq_sl", exit_kind="SL")
                    return
        try:
            hold_only = str(os.environ.get("HOLD_EVAL_ONLY", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            hold_only = False
        if hold_only:
            action = (decision or {}).get("action") or "WAIT"
            if action == "EXIT":
                meta = (decision or {}).get("meta") or {}
                reason = str((decision or {}).get("reason") or "")
                if meta.get("switch_to") or reason.startswith("SWITCH_TO_"):
                    self._close_position(sym, price, reason or "switch_exit")
            return
        action = (decision or {}).get("action") or "WAIT"
        self._capture_position_observability(pos, decision=decision, ctx=ctx)
        min_hold_sec_eff = self._effective_min_hold_sec(pos)
        try:
            pos["policy_min_hold_eff_sec"] = float(min_hold_sec_eff)
        except Exception:
            pass
        if action == "EXIT":
            entry_ts = pos.get("entry_time", ts)
            age_sec = (ts - entry_ts) / 1000.0 if entry_ts else 0.0
            if min_hold_sec_eff > 0 and age_sec < float(min_hold_sec_eff):
                return
            meta = (decision or {}).get("meta") or {}
            exit_policy_dyn, exit_diag = self._build_dynamic_exit_policy(sym, pos, decision, ctx=ctx)
            self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_diag)
            hybrid_mode_state = self._resolve_exit_mode_state(
                exit_diag,
                shock_threshold_env="HYBRID_EXIT_SHOCK_FAST_THRESHOLD",
                shock_threshold_default=1.0,
            )
            confirm_mode = str(hybrid_mode_state.get("mode") or "normal")
            confirm_required, confirm_reset_sec, _hybrid_ticks = self._get_exit_confirmation_rule(
                "HYBRID_EXIT",
                confirm_mode,
                default_normal=2,
                default_shock=1,
                default_noise=3,
                default_reset_sec=180.0,
            )
            confirm_ok, confirm_cnt = self._advance_exit_confirmation(
                pos,
                "hybrid_exit",
                triggered=True,
                ts_ms=int(ts),
                required_ticks=confirm_required,
                reset_sec=float(max(0.0, confirm_reset_sec)),
            )
            meta["hybrid_exit_confirm_mode"] = str(confirm_mode)
            meta["hybrid_exit_confirm_required"] = int(confirm_required)
            meta["hybrid_exit_confirm_count"] = int(confirm_cnt)
            meta["hybrid_exit_confirmed"] = bool(confirm_ok)
            meta["hybrid_exit_shock_score"] = float(hybrid_mode_state.get("shock_score") or 0.0)
            meta["hybrid_exit_noise_mode"] = bool(hybrid_mode_state.get("noise_mode"))
            decision["meta"] = meta
            if not confirm_ok:
                return
            self._close_position(sym, price, "hybrid_exit")
            return
        else:
            confirm_ticks_normal, confirm_reset_sec, _hybrid_ticks_idle = self._get_exit_confirmation_rule(
                "HYBRID_EXIT",
                "normal",
                default_normal=2,
                default_shock=1,
                default_noise=3,
                default_reset_sec=180.0,
            )
            _hyb_ok, _hyb_cnt = self._advance_exit_confirmation(
                pos,
                "hybrid_exit",
                triggered=False,
                ts_ms=int(ts),
                required_ticks=int(max(1, confirm_ticks_normal)),
                reset_sec=float(max(0.0, confirm_reset_sec)),
            )
            try:
                meta = (decision or {}).get("meta") or {}
                meta["hybrid_exit_confirm_mode"] = "idle"
                meta["hybrid_exit_confirm_required"] = int(max(1, confirm_ticks_normal))
                meta["hybrid_exit_confirm_count"] = int(_hyb_cnt)
                meta["hybrid_exit_confirmed"] = bool(_hyb_ok)
                decision["meta"] = meta
            except Exception:
                pass
        if not allow_extra_exits:
            return
        age_ms = ts - pos.get("entry_time", ts)
        entry = float(pos.get("entry_price", price))
        qty = float(pos.get("quantity", 0.0))
        side = pos.get("side")
        notional = float(pos.get("notional", 0.0))
        pnl_unreal = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
        base_notional = notional / max(lev_safe, 1e-6) if notional else 0.0
        roe_unreal = pnl_unreal / base_notional if base_notional else 0.0
        unified_score_hold = None
        if decision:
            unified_score_hold = decision.get("unified_score_hold")
            if unified_score_hold is None:
                if side == "LONG":
                    unified_score_hold = decision.get("unified_score_long")
                elif side == "SHORT":
                    unified_score_hold = decision.get("unified_score_short")
        unified_score_hold = float(unified_score_hold) if unified_score_hold is not None else None

        unified_score_rev = None
        if decision:
            if side == "LONG":
                unified_score_rev = decision.get("unified_score_short")
            elif side == "SHORT":
                unified_score_rev = decision.get("unified_score_long")
        unified_score_rev = float(unified_score_rev) if unified_score_rev is not None else None

        exit_reasons = []
        min_hold_ms = max(0.0, float(min_hold_sec_eff) * 1000.0)
        hold_ok = (age_ms >= min_hold_ms) if min_hold_ms > 0 else True
        exit_policy_dyn, exit_diag = self._build_dynamic_exit_policy(sym, pos, decision, ctx=ctx)
        if isinstance(decision, dict):
            meta = dict(decision.get("meta") or {})
            self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_diag)
            decision["meta"] = meta
        cash_mode_state = self._resolve_exit_mode_state(
            exit_diag,
            shock_threshold_env="UNIFIED_CASH_EXIT_SHOCK_FAST_THRESHOLD",
            shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
        )
        entry_quality_score = float(max(0.0, min(1.0, self._safe_float(pos.get("entry_quality_score"), 0.0) or 0.0)))
        one_way_move_score = float(max(0.0, min(1.0, self._safe_float(pos.get("one_way_move_score"), 0.0) or 0.0)))
        leverage_signal_score = float(max(0.0, min(1.0, self._safe_float(pos.get("leverage_signal_score"), 0.0) or 0.0)))
        try:
            opt_hold_sec = float(
                pos.get("opt_hold_curr_sec")
                or pos.get("opt_hold_entry_sec")
                or pos.get("opt_hold_sec")
                or 0.0
            )
        except Exception:
            opt_hold_sec = 0.0
        hold_progress = 1.0
        if opt_hold_sec > 0:
            hold_progress = float(max(0.0, age_ms / 1000.0) / max(opt_hold_sec, 1e-6))
        shock_mode_now = str(cash_mode_state.get("mode") or "normal")
        shock_score_now = float(cash_mode_state.get("shock_score") or 0.0)
        mu_align_now = float(self._safe_float(exit_diag.get("mu_alignment"), 0.0) or 0.0)
        try:
            respect_enabled = str(os.environ.get("EXIT_RESPECT_ENTRY_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            respect_enabled = True
        try:
            respect_min_q = float(os.environ.get("EXIT_RESPECT_ENTRY_MIN_Q", 0.72) or 0.72)
        except Exception:
            respect_min_q = 0.72
        try:
            respect_min_signal = float(os.environ.get("EXIT_RESPECT_ENTRY_MIN_SIGNAL", 0.70) or 0.70)
        except Exception:
            respect_min_signal = 0.70
        try:
            respect_min_progress = float(os.environ.get("EXIT_RESPECT_ENTRY_MIN_PROGRESS", 0.65) or 0.65)
        except Exception:
            respect_min_progress = 0.65
        try:
            respect_mu_opposed = float(os.environ.get("EXIT_RESPECT_ENTRY_MU_OPPOSED", -0.0010) or -0.0010)
        except Exception:
            respect_mu_opposed = -0.0010
        try:
            respect_flip_margin = float(os.environ.get("EXIT_RESPECT_ENTRY_FLIP_MARGIN", 0.0010) or 0.0010)
        except Exception:
            respect_flip_margin = 0.0010
        high_quality_entry = bool(
            entry_quality_score >= float(max(0.0, respect_min_q))
            and leverage_signal_score >= float(max(0.0, respect_min_signal))
        )

        def _respect_entry_guard(reason_tag: str, reverse_edge: float | None = None) -> bool:
            if not respect_enabled:
                return False
            if not high_quality_entry:
                return False
            if shock_mode_now == "shock" or shock_score_now >= 1.0:
                return False
            # Strong opposite alpha should still be allowed to exit.
            if float(mu_align_now) <= float(respect_mu_opposed):
                return False
            if reverse_edge is not None and float(reverse_edge) >= float(respect_flip_margin):
                return False
            return bool(hold_progress < float(max(0.0, respect_min_progress)))

        if isinstance(decision, dict):
            meta = dict(decision.get("meta") or {})
            meta["entry_quality_score"] = float(entry_quality_score)
            meta["one_way_move_score"] = float(one_way_move_score)
            meta["leverage_signal_score"] = float(leverage_signal_score)
            meta["entry_respect_guard_enabled"] = bool(respect_enabled)
            meta["entry_respect_guard_high_quality"] = bool(high_quality_entry)
            meta["entry_respect_guard_hold_progress"] = float(hold_progress)
            meta["entry_respect_guard_mode"] = str(shock_mode_now)
            meta["entry_respect_guard_shock_score"] = float(shock_score_now)
            decision["meta"] = meta
        def _decision_metric_float(*keys: str, default: float | None = None) -> float | None:
            for k in keys:
                try:
                    if isinstance(decision, dict):
                        m = decision.get("meta")
                        if isinstance(m, dict) and m.get(k) is not None:
                            return float(m.get(k))
                except Exception:
                    pass
                try:
                    if isinstance(decision, dict) and decision.get(k) is not None:
                        return float(decision.get(k))
                except Exception:
                    pass
            return default

        cash_normal_required, cash_confirm_reset_sec, _cash_ticks = self._get_exit_confirmation_rule(
            "UNIFIED_CASH_EXIT",
            "normal",
            default_normal=2,
            default_shock=1,
            default_noise=3,
            default_reset_sec=120.0,
        )
        flip_normal_required, flip_confirm_reset_sec, _flip_ticks = self._get_exit_confirmation_rule(
            "UNIFIED_FLIP",
            "normal",
            default_normal=3,
            default_shock=1,
            default_noise=4,
            default_reset_sec=180.0,
        )
        unified_cash_candidate = False
        unified_flip_candidate = False
        if action in ("LONG", "SHORT") and action != pos["side"]:
            reverse_edge = None
            try:
                if unified_score_hold is not None and unified_score_rev is not None:
                    reverse_edge = float(unified_score_rev) - float(unified_score_hold)
            except Exception:
                reverse_edge = None
            unified_flip_candidate = bool(
                hold_ok and (unified_score_hold is None or unified_score_rev is None or unified_score_rev > unified_score_hold)
            )
            flip_confirm_mode = "idle"
            flip_confirm_required = int(max(1, flip_normal_required))
            flip_confirm_cnt = 0
            flip_confirm_ok = False
            flip_guarded = False
            flip_guard_reason = ""
            flip_guard_progress = None
            flip_guard_required = None
            flip_guard_hold_target_sec = None
            if unified_flip_candidate:
                reverse_edge_v = float(reverse_edge) if reverse_edge is not None else 0.0
                flip_mode_state = self._resolve_exit_mode_state(
                    {
                        "shock_score": float(cash_mode_state.get("shock_score") or 0.0),
                        "noise_mode": bool(cash_mode_state.get("noise_mode")),
                        "shock_bucket": str(cash_mode_state.get("mode") or "normal"),
                    },
                    shock_threshold_env="UNIFIED_FLIP_SHOCK_FAST_THRESHOLD",
                    shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
                )
                flip_confirm_mode = str(flip_mode_state.get("mode") or "normal")
                flip_confirm_required, flip_confirm_reset_sec, _flip_ticks_dyn = self._get_exit_confirmation_rule(
                    "UNIFIED_FLIP",
                    flip_confirm_mode,
                    default_normal=3,
                    default_shock=1,
                    default_noise=4,
                    default_reset_sec=180.0,
                )
                try:
                    flip_min_reverse_edge = float(os.environ.get("UNIFIED_FLIP_MIN_REVERSE_EDGE", 0.0008) or 0.0008)
                except Exception:
                    flip_min_reverse_edge = 0.0008
                try:
                    flip_bypass_reverse_edge = float(os.environ.get("UNIFIED_FLIP_BYPASS_REVERSE_EDGE", 0.0030) or 0.0030)
                except Exception:
                    flip_bypass_reverse_edge = 0.0030
                try:
                    flip_min_opp_prob = float(os.environ.get("UNIFIED_FLIP_MIN_OPPOSITE_SIDE_PROB", 0.56) or 0.56)
                except Exception:
                    flip_min_opp_prob = 0.56
                try:
                    flip_min_dir_conf = float(
                        os.environ.get("UNIFIED_FLIP_MIN_DIR_CONF", os.environ.get("ALPHA_DIRECTION_GATE_MIN_CONF", 0.58)) or 0.58
                    )
                except Exception:
                    flip_min_dir_conf = 0.58
                try:
                    flip_min_dir_edge = float(
                        os.environ.get("UNIFIED_FLIP_MIN_DIR_EDGE", os.environ.get("ALPHA_DIRECTION_GATE_MIN_EDGE", 0.06)) or 0.06
                    )
                except Exception:
                    flip_min_dir_edge = 0.06
                dir_conf_now = float(max(0.0, min(1.0, self._safe_float(_decision_metric_float("mu_dir_conf", default=0.0), 0.0) or 0.0)))
                dir_edge_now = float(abs(self._safe_float(_decision_metric_float("mu_dir_edge", default=0.0), 0.0) or 0.0))
                dir_prob_long_now = float(max(0.0, min(1.0, self._safe_float(_decision_metric_float("mu_dir_prob_long", default=0.5), 0.5) or 0.5)))
                opp_side_prob = float(dir_prob_long_now) if str(action).upper() == "LONG" else float(1.0 - dir_prob_long_now)
                if (flip_confirm_mode != "shock") and (reverse_edge_v < float(max(0.0, flip_min_reverse_edge))):
                    flip_guarded = True
                    flip_guard_reason = "reverse_edge"
                if (not flip_guarded) and (flip_confirm_mode != "shock") and (reverse_edge_v < float(max(0.0, flip_bypass_reverse_edge))):
                    if opp_side_prob < float(max(0.5, flip_min_opp_prob)):
                        flip_guarded = True
                        flip_guard_reason = "opp_side_prob"
                    elif dir_conf_now < float(max(0.5, flip_min_dir_conf)):
                        flip_guarded = True
                        flip_guard_reason = "dir_conf"
                    elif dir_edge_now < float(max(0.0, flip_min_dir_edge)):
                        flip_guarded = True
                        flip_guard_reason = "dir_edge"

                if (not flip_guarded) and flip_confirm_mode != "shock":
                    try:
                        flip_guard_enabled = str(os.environ.get("UNIFIED_FLIP_TSTAR_GUARD_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        flip_guard_enabled = True
                    if flip_guard_enabled:
                        try:
                            h_low = float(getattr(mc_config, "hurst_low", 0.45))
                            h_high = float(getattr(mc_config, "hurst_high", 0.55))
                        except Exception:
                            h_low, h_high = 0.45, 0.55
                        try:
                            flip_prog_trend = float(os.environ.get("UNIFIED_FLIP_MIN_PROGRESS_TREND", 0.55) or 0.55)
                        except Exception:
                            flip_prog_trend = 0.55
                        try:
                            flip_prog_random = float(os.environ.get("UNIFIED_FLIP_MIN_PROGRESS_RANDOM", 0.75) or 0.75)
                        except Exception:
                            flip_prog_random = 0.75
                        try:
                            flip_prog_mr = float(os.environ.get("UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT", 0.85) or 0.85)
                        except Exception:
                            flip_prog_mr = 0.85
                        try:
                            flip_bypass_shock = float(os.environ.get("UNIFIED_FLIP_BYPASS_SHOCK", 1.15) or 1.15)
                        except Exception:
                            flip_bypass_shock = 1.15
                        try:
                            flip_hold_fallback = float(os.environ.get("UNIFIED_FLIP_MIN_HOLD_FALLBACK_SEC", 60) or 60)
                        except Exception:
                            flip_hold_fallback = 60.0
                        hold_target = None
                        try:
                            hold_target = float(
                                pos.get("opt_hold_curr_sec")
                                or pos.get("opt_hold_entry_sec")
                                or pos.get("opt_hold_sec")
                                or pos.get("policy_min_hold_eff_sec")
                                or 0.0
                            )
                        except Exception:
                            hold_target = 0.0
                        if not hold_target or hold_target <= 0:
                            try:
                                hold_target = float(self._effective_min_hold_sec(pos))
                            except Exception:
                                hold_target = 0.0
                        if flip_hold_fallback > 0:
                            hold_target = float(max(float(hold_target or 0.0), float(flip_hold_fallback)))
                        if hold_target > 0:
                            age_sec = max(0.0, float(age_ms) / 1000.0)
                            progress = float(age_sec / max(float(hold_target), 1e-6))
                            hurst_now = self._safe_float(exit_diag.get("hurst"), None)
                            req_progress = float(flip_prog_random)
                            if hurst_now is not None and float(hurst_now) > h_high:
                                req_progress = float(flip_prog_trend)
                            elif hurst_now is not None and float(hurst_now) < h_low:
                                req_progress = float(flip_prog_mr)
                            flip_shock = float(flip_mode_state.get("shock_score") or 0.0)
                            if (progress < float(req_progress)) and (flip_shock < float(flip_bypass_shock)) and (
                                reverse_edge_v < float(max(0.0, flip_bypass_reverse_edge))
                            ):
                                flip_guarded = True
                                flip_guard_reason = "tstar_progress"
                                flip_guard_progress = float(progress)
                                flip_guard_required = float(req_progress)
                                flip_guard_hold_target_sec = float(hold_target)

                flip_confirm_ok, flip_confirm_cnt = self._advance_exit_confirmation(
                    pos,
                    "unified_flip_exit",
                    triggered=bool(unified_flip_candidate and (not flip_guarded)),
                    ts_ms=int(ts),
                    required_ticks=int(max(1, flip_confirm_required)),
                    reset_sec=float(max(0.0, flip_confirm_reset_sec)),
                )
                if isinstance(decision, dict):
                    meta = dict(decision.get("meta") or {})
                    meta["unified_flip_candidate"] = bool(unified_flip_candidate)
                    meta["unified_flip_reverse_edge"] = float(reverse_edge_v)
                    meta["unified_flip_opp_side_prob"] = float(opp_side_prob)
                    meta["unified_flip_dir_conf"] = float(dir_conf_now)
                    meta["unified_flip_dir_edge"] = float(dir_edge_now)
                    meta["unified_flip_confirm_mode"] = str(flip_confirm_mode)
                    meta["unified_flip_confirm_required"] = int(flip_confirm_required)
                    meta["unified_flip_confirm_count"] = int(flip_confirm_cnt)
                    meta["unified_flip_confirmed"] = bool(flip_confirm_ok)
                    meta["unified_flip_guarded"] = bool(flip_guarded)
                    meta["unified_flip_guard_reason"] = str(flip_guard_reason or "")
                    if flip_guard_progress is not None:
                        meta["unified_flip_guard_progress"] = float(flip_guard_progress)
                        meta["unified_flip_guard_required"] = float(flip_guard_required or 0.0)
                        meta["unified_flip_guard_hold_target_sec"] = float(flip_guard_hold_target_sec or 0.0)
                    decision["meta"] = meta
                if flip_confirm_ok:
                    if _respect_entry_guard("unified_flip", reverse_edge=reverse_edge):
                        if isinstance(decision, dict):
                            meta = dict(decision.get("meta") or {})
                            meta["entry_respect_guard_blocked"] = "unified_flip"
                            meta["entry_respect_guard_reverse_edge"] = float(reverse_edge) if reverse_edge is not None else None
                            decision["meta"] = meta
                    else:
                        exit_reasons.append("unified_flip")
        elif action == "WAIT":
            if unified_score_hold is not None:
                if hold_ok:
                    cash_exit_enabled = bool(getattr(mc_config, "policy_cash_exit_enabled", False))
                    cash_exit_score = float(getattr(mc_config, "policy_cash_exit_score", 0.0))
                    entry_floor = pos.get("entry_floor")
                    entry_score = pos.get("entry_score_hold")
                    if entry_score is None:
                        entry_score = pos.get("entry_score")
                    score_drop = float(os.environ.get("EXIT_SCORE_DROP", 0.0) or 0.0)
                    should_exit = False
                    if cash_exit_enabled and unified_score_hold <= cash_exit_score:
                        should_exit = True
                    if entry_floor is not None:
                        try:
                            if unified_score_hold <= float(entry_floor):
                                should_exit = True
                        except Exception:
                            pass
                    if score_drop > 0 and entry_score is not None:
                        try:
                            if unified_score_hold <= float(entry_score) - float(score_drop):
                                should_exit = True
                        except Exception:
                            pass
                    if should_exit:
                        unified_cash_candidate = True
                        cash_confirm_mode = str(cash_mode_state.get("mode") or "normal")
                        cash_confirm_required, cash_confirm_reset_sec, _cash_ticks_dyn = self._get_exit_confirmation_rule(
                            "UNIFIED_CASH_EXIT",
                            cash_confirm_mode,
                            default_normal=2,
                            default_shock=1,
                            default_noise=3,
                            default_reset_sec=120.0,
                        )
                        cash_confirm_ok, cash_confirm_cnt = self._advance_exit_confirmation(
                            pos,
                            "unified_cash_exit",
                            triggered=True,
                            ts_ms=int(ts),
                            required_ticks=cash_confirm_required,
                            reset_sec=float(max(0.0, cash_confirm_reset_sec)),
                        )
                        if isinstance(decision, dict):
                            meta = dict(decision.get("meta") or {})
                            meta["unified_cash_exit_candidate"] = True
                            meta["unified_cash_exit_confirm_mode"] = str(cash_confirm_mode)
                            meta["unified_cash_exit_confirm_required"] = int(cash_confirm_required)
                            meta["unified_cash_exit_confirm_count"] = int(cash_confirm_cnt)
                            meta["unified_cash_exit_confirmed"] = bool(cash_confirm_ok)
                            meta["unified_cash_exit_shock_score"] = float(cash_mode_state.get("shock_score") or 0.0)
                            meta["unified_cash_exit_noise_mode"] = bool(cash_mode_state.get("noise_mode"))
                            decision["meta"] = meta
                        if cash_confirm_ok:
                            if _respect_entry_guard("unified_cash", reverse_edge=None):
                                if isinstance(decision, dict):
                                    meta = dict(decision.get("meta") or {})
                                    meta["entry_respect_guard_blocked"] = "unified_cash"
                                    decision["meta"] = meta
                            else:
                                exit_reasons.append("unified_cash")
        if not unified_flip_candidate:
            _flip_ok, _flip_cnt = self._advance_exit_confirmation(
                pos,
                "unified_flip_exit",
                triggered=False,
                ts_ms=int(ts),
                required_ticks=int(max(1, flip_normal_required)),
                reset_sec=float(max(0.0, flip_confirm_reset_sec)),
            )
            if isinstance(decision, dict):
                meta = dict(decision.get("meta") or {})
                meta["unified_flip_candidate"] = False
                meta["unified_flip_confirm_mode"] = "idle"
                meta["unified_flip_confirm_required"] = int(max(1, flip_normal_required))
                meta["unified_flip_confirm_count"] = int(_flip_cnt)
                meta["unified_flip_confirmed"] = bool(_flip_ok)
                meta["unified_flip_guarded"] = False
                decision["meta"] = meta
        if not unified_cash_candidate:
            _cash_ok, _cash_cnt = self._advance_exit_confirmation(
                pos,
                "unified_cash_exit",
                triggered=False,
                ts_ms=int(ts),
                required_ticks=int(max(1, cash_normal_required)),
                reset_sec=float(max(0.0, cash_confirm_reset_sec)),
            )
            if isinstance(decision, dict):
                meta = dict(decision.get("meta") or {})
                meta["unified_cash_exit_candidate"] = False
                meta["unified_cash_exit_confirm_mode"] = "idle"
                meta["unified_cash_exit_confirm_required"] = int(max(1, cash_normal_required))
                meta["unified_cash_exit_confirm_count"] = int(_cash_cnt)
                meta["unified_cash_exit_confirmed"] = bool(_cash_ok)
                decision["meta"] = meta
        # 공격적 손실 컷 (미실현 ROE 기준)
        if hold_ok:
            try:
                dd_floor_base = float(os.environ.get("UNREALIZED_DD_EXIT_ROE", -0.02) or -0.02)
            except Exception:
                dd_floor_base = -0.02
            dd_floor = float(dd_floor_base)
            dd_mode = "normal"
            dd_reg_mult = 1.0
            try:
                if bool(getattr(mc_config, "exit_policy_dynamic_enabled", True)):
                    mu_align = self._safe_float(exit_diag.get("mu_alignment"), 0.0) or 0.0
                    vpin_val = self._safe_float(exit_diag.get("vpin"), None)
                    hurst_val = self._safe_float(exit_diag.get("hurst"), None)
                    shock_score = max(0.0, self._safe_float(exit_diag.get("shock_score"), 0.0) or 0.0)
                    noise_mode = bool(exit_diag.get("noise_mode"))
                    hmm_sign = self._safe_float(exit_diag.get("hmm_sign"), 0.0) or 0.0
                    hmm_conf = max(0.0, min(1.0, self._safe_float(exit_diag.get("hmm_conf"), 0.0) or 0.0))
                    side_sign = 1.0 if str(side).upper() == "LONG" else (-1.0 if str(side).upper() == "SHORT" else 0.0)
                    align_scale = max(float(getattr(mc_config, "exit_policy_mu_align_scale", 1.5) or 1.5), 1e-6)
                    mu_term = math.tanh(float(mu_align) / float(align_scale))
                    # mu alignment가 좋으면 손실 허용폭 확대, 나쁘면 손실컷 강화
                    dd_floor = float(dd_floor * (1.0 + 0.55 * mu_term))
                    if vpin_val is not None:
                        vpin_val = float(max(0.0, min(1.0, vpin_val)))
                        vpin_tox = float(max(0.0, vpin_val - 0.5) / 0.5)
                        # 고독성 구간은 기본적으로 빠르게 컷
                        dd_floor = float(dd_floor * (1.0 - 0.30 * vpin_tox))
                        # 단, VPIN extreme + mean-revert 레짐 + 음의 mu 정렬은 역발상 진입기회로 여겨 완화
                        vpin_ext = float(getattr(mc_config, "vpin_extreme_threshold", 0.90))
                        h_low = float(getattr(mc_config, "hurst_low", 0.45))
                        if vpin_val >= vpin_ext and hurst_val is not None and float(hurst_val) < h_low and mu_term < 0.0:
                            dd_floor = float(dd_floor * 1.25)
                    # Shock vs noise split: in adverse shocks cut faster, in noisy shakeouts allow more room.
                    dd_mode_state = self._resolve_exit_mode_state(
                        {
                            "shock_score": float(shock_score),
                            "noise_mode": bool(noise_mode),
                            "shock_bucket": "noise" if bool(noise_mode) else "normal",
                        },
                        shock_threshold_env="UNREALIZED_DD_SHOCK_SCORE_THRESHOLD",
                        shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
                    )
                    dd_mode = str(dd_mode_state.get("mode") or "normal")
                    if shock_score > 0.0:
                        dd_floor = float(dd_floor * max(0.50, 1.0 - 0.22 * min(shock_score, 2.5)))
                    elif noise_mode:
                        dd_floor = float(dd_floor * 1.12)
                    if dd_mode == "shock":
                        try:
                            dd_floor *= float(os.environ.get("UNREALIZED_DD_MULT_SHOCK", 0.78) or 0.78)
                        except Exception:
                            pass
                    elif dd_mode == "noise":
                        try:
                            dd_floor *= float(os.environ.get("UNREALIZED_DD_MULT_NOISE", 1.12) or 1.12)
                        except Exception:
                            pass
                    else:
                        try:
                            dd_floor *= float(os.environ.get("UNREALIZED_DD_MULT_NORMAL", 1.00) or 1.00)
                        except Exception:
                            pass
                    # Regime-aware DD multiplier (auto-tuner can drive *_TREND/*_CHOP/*_VOLATILE).
                    try:
                        reg_bucket = str(exit_diag.get("regime_bucket") or "").strip().lower()
                    except Exception:
                        reg_bucket = ""
                    dd_reg_mult = 1.0
                    try:
                        if reg_bucket == "trend":
                            dd_reg_mult = float(os.environ.get("UNREALIZED_DD_REGIME_MULT_TREND", 1.0) or 1.0)
                        elif reg_bucket == "mean_revert":
                            raw = os.environ.get("UNREALIZED_DD_REGIME_MULT_MEAN_REVERT")
                            if raw is None or str(raw).strip() == "":
                                raw = os.environ.get("UNREALIZED_DD_REGIME_MULT_CHOP", "1.0")
                            dd_reg_mult = float(raw or 1.0)
                        else:
                            raw = os.environ.get("UNREALIZED_DD_REGIME_MULT_RANDOM")
                            if raw is None or str(raw).strip() == "":
                                raw = os.environ.get("UNREALIZED_DD_REGIME_MULT_VOLATILE", "1.0")
                            dd_reg_mult = float(raw or 1.0)
                    except Exception:
                        dd_reg_mult = 1.0
                    dd_floor = float(dd_floor * dd_reg_mult)
                    # HMM disagreement implies higher chance that holding is wrong-side.
                    if side_sign != 0.0 and hmm_conf > 0.0:
                        hmm_dir = 1.0 if hmm_sign > 0 else (-1.0 if hmm_sign < 0 else 0.0)
                        hmm_align = side_sign * hmm_dir
                        if hmm_align < 0:
                            dd_floor = float(dd_floor * max(0.55, 1.0 - 0.30 * hmm_conf))
                        elif hmm_align > 0 and noise_mode:
                            dd_floor = float(dd_floor * (1.0 + 0.15 * hmm_conf))
                    dd_floor = float(np.clip(dd_floor, -0.20, -0.002))
                    if isinstance(decision, dict):
                        meta = dict(decision.get("meta") or {})
                        meta["unrealized_dd_floor_base"] = float(dd_floor_base)
                        meta["unrealized_dd_floor_dyn"] = float(dd_floor)
                        meta["unrealized_dd_mu_alignment"] = float(mu_align)
                        meta["unrealized_dd_vpin"] = vpin_val
                        meta["unrealized_dd_hurst"] = hurst_val
                        meta["unrealized_dd_shock_score"] = float(shock_score)
                        meta["unrealized_dd_noise_mode"] = bool(noise_mode)
                        meta["unrealized_dd_mode"] = str(dd_mode)
                        meta["unrealized_dd_regime_mult"] = float(dd_reg_mult)
                        meta["unrealized_dd_hmm_sign"] = float(hmm_sign)
                        meta["unrealized_dd_hmm_conf"] = float(hmm_conf)
                        decision["meta"] = meta
            except Exception:
                dd_floor = float(dd_floor_base)
                dd_mode = "normal"
            dd_triggered = bool(roe_unreal <= dd_floor)
            dd_guard_progress = None
            dd_guard_required = None
            dd_guard_hold_target_sec = None
            dd_guarded = False
            if dd_triggered and dd_mode != "shock":
                try:
                    dd_guard_enabled = str(os.environ.get("UNREALIZED_DD_TSTAR_GUARD_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    dd_guard_enabled = True
                if dd_guard_enabled:
                    try:
                        h_low = float(getattr(mc_config, "hurst_low", 0.45))
                        h_high = float(getattr(mc_config, "hurst_high", 0.55))
                    except Exception:
                        h_low, h_high = 0.45, 0.55
                    try:
                        dd_prog_trend = float(os.environ.get("UNREALIZED_DD_TSTAR_MIN_PROGRESS_TREND", 0.45) or 0.45)
                    except Exception:
                        dd_prog_trend = 0.45
                    try:
                        dd_prog_random = float(os.environ.get("UNREALIZED_DD_TSTAR_MIN_PROGRESS_RANDOM", 0.60) or 0.60)
                    except Exception:
                        dd_prog_random = 0.60
                    try:
                        dd_prog_mr = float(os.environ.get("UNREALIZED_DD_TSTAR_MIN_PROGRESS_MEAN_REVERT", 0.70) or 0.70)
                    except Exception:
                        dd_prog_mr = 0.70
                    try:
                        dd_bypass_shock = float(os.environ.get("UNREALIZED_DD_TSTAR_BYPASS_SHOCK", 1.15) or 1.15)
                    except Exception:
                        dd_bypass_shock = 1.15
                    try:
                        dd_bypass_floor_mult = float(os.environ.get("UNREALIZED_DD_TSTAR_BYPASS_FLOOR_MULT", 1.30) or 1.30)
                    except Exception:
                        dd_bypass_floor_mult = 1.30
                    try:
                        entry_ts_ms = self._normalize_entry_time_ms(pos.get("entry_time"), default=ts)
                        age_sec_dd = max(0.0, (int(ts) - int(entry_ts_ms)) / 1000.0)
                    except Exception:
                        age_sec_dd = 0.0
                    hold_target_dd = None
                    try:
                        hold_target_dd = float(
                            pos.get("opt_hold_curr_sec")
                            or pos.get("opt_hold_entry_sec")
                            or pos.get("opt_hold_sec")
                            or pos.get("policy_min_hold_eff_sec")
                            or 0.0
                        )
                    except Exception:
                        hold_target_dd = 0.0
                    if not hold_target_dd or hold_target_dd <= 0:
                        try:
                            hold_target_dd = float(self._effective_min_hold_sec(pos))
                        except Exception:
                            hold_target_dd = 0.0
                    try:
                        dd_fallback_floor = float(os.environ.get("UNREALIZED_DD_TSTAR_MIN_HOLD_FALLBACK_SEC", 0.0) or 0.0)
                    except Exception:
                        dd_fallback_floor = 0.0
                    if dd_fallback_floor > 0:
                        hold_target_dd = float(max(float(hold_target_dd or 0.0), dd_fallback_floor))
                    if hold_target_dd and hold_target_dd > 0:
                        progress_dd = float(age_sec_dd / max(float(hold_target_dd), 1e-6))
                        hurst_now_dd = self._safe_float(exit_diag.get("hurst"), None)
                        if hurst_now_dd is None:
                            hurst_now_dd = self._safe_float((decision or {}).get("hurst"), None) if isinstance(decision, dict) else None
                        req_progress_dd = dd_prog_random
                        if hurst_now_dd is not None and float(hurst_now_dd) > h_high:
                            req_progress_dd = dd_prog_trend
                        elif hurst_now_dd is not None and float(hurst_now_dd) < h_low:
                            req_progress_dd = dd_prog_mr
                        shock_now_dd = self._safe_float(exit_diag.get("shock_score"), 0.0) or 0.0
                        severe_dd = bool(roe_unreal <= float(dd_floor) * float(max(1.0, dd_bypass_floor_mult)))
                        if (progress_dd < float(req_progress_dd)) and (float(shock_now_dd) < float(dd_bypass_shock)) and (not severe_dd):
                            dd_triggered = False
                            dd_guarded = True
                            dd_guard_progress = float(progress_dd)
                            dd_guard_required = float(req_progress_dd)
                            dd_guard_hold_target_sec = float(hold_target_dd)
            dd_confirm_required, dd_confirm_reset_sec, _dd_ticks = self._get_exit_confirmation_rule(
                "UNREALIZED_DD",
                dd_mode,
                default_normal=2,
                default_shock=1,
                default_noise=3,
                default_reset_sec=180.0,
            )
            if dd_mode != "shock":
                try:
                    dd_non_shock_min = int(os.environ.get("UNREALIZED_DD_MIN_CONFIRM_NON_SHOCK", 3) or 3)
                except Exception:
                    dd_non_shock_min = 3
                try:
                    dd_noise_min = int(os.environ.get("UNREALIZED_DD_MIN_CONFIRM_NOISE", dd_non_shock_min) or dd_non_shock_min)
                except Exception:
                    dd_noise_min = dd_non_shock_min
                if dd_mode == "noise":
                    dd_confirm_required = int(max(dd_confirm_required, max(1, dd_noise_min)))
                else:
                    dd_confirm_required = int(max(dd_confirm_required, max(1, dd_non_shock_min)))
            dd_confirm_ok, dd_confirm_cnt = self._advance_exit_confirmation(
                pos,
                "unrealized_dd_exit",
                triggered=bool(dd_triggered),
                ts_ms=int(ts),
                required_ticks=int(dd_confirm_required),
                reset_sec=float(max(0.0, dd_confirm_reset_sec)),
            )
            if isinstance(decision, dict):
                meta = dict(decision.get("meta") or {})
                meta["unrealized_dd_triggered"] = bool(dd_triggered)
                meta["unrealized_dd_confirm_mode"] = str(dd_mode if dd_triggered else "idle")
                meta["unrealized_dd_confirm_required"] = int(dd_confirm_required)
                meta["unrealized_dd_confirm_count"] = int(dd_confirm_cnt)
                meta["unrealized_dd_confirmed"] = bool(dd_confirm_ok)
                meta["unrealized_dd_guarded_by_tstar"] = bool(dd_guarded)
                if dd_guarded:
                    meta["unrealized_dd_guard_progress"] = dd_guard_progress
                    meta["unrealized_dd_guard_required"] = dd_guard_required
                    meta["unrealized_dd_guard_hold_target_sec"] = dd_guard_hold_target_sec
                decision["meta"] = meta
            if dd_triggered and dd_confirm_ok:
                try:
                    dd_severe_mult = float(os.environ.get("UNREALIZED_DD_SEVERE_MULT", 1.40) or 1.40)
                except Exception:
                    dd_severe_mult = 1.40
                severe_loss = bool(roe_unreal <= float(dd_floor) * float(max(1.0, dd_severe_mult)))
                if _respect_entry_guard(f"unrealized_dd_{dd_mode}", reverse_edge=None) and (dd_mode != "shock") and (not severe_loss):
                    if isinstance(decision, dict):
                        meta = dict(decision.get("meta") or {})
                        meta["entry_respect_guard_blocked"] = f"unrealized_dd_{dd_mode}"
                        meta["unrealized_dd_severe_loss"] = bool(severe_loss)
                        decision["meta"] = meta
                else:
                    exit_reasons.append(f"unrealized_dd_{dd_mode}")
        if exit_reasons:
            self._close_position(sym, price, ", ".join(exit_reasons))

    def _record_trade(
        self,
        ttype: str,
        sym: str,
        side: str,
        price: float,
        qty: float,
        pos: dict,
        pnl: float | None = None,
        fee: float | None = None,
        reason: str | None = None,
        realized_r: float | None = None,
        hit: int | None = None,
        exit_kind: str | None = None,
        **_ignored,
    ):
        ts = time.strftime("%H:%M:%S")
        # Backstop: keep observability payload populated even on legacy/side-path exits.
        if isinstance(pos, dict):
            try:
                self._capture_position_observability(pos, decision=None, ctx=None)
            except Exception:
                pass
            if pos.get("policy_min_hold_eff_sec") is None:
                try:
                    pos["policy_min_hold_eff_sec"] = float(self._effective_min_hold_sec(pos))
                except Exception:
                    pass
        # Ensure hold duration is captured for DB logs
        hold_duration_sec = None
        try:
            if pos is not None:
                hold_duration_sec = pos.get("age_sec")
                if hold_duration_sec is None:
                    entry_ts = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
                    hold_duration_sec = max(0.0, (now_ms() - int(entry_ts)) / 1000.0)
                    pos["age_sec"] = float(hold_duration_sec)
                hold_duration_sec = float(hold_duration_sec) if hold_duration_sec is not None else None
        except Exception:
            hold_duration_sec = None
        opt_hold_entry = pos.get("opt_hold_entry_sec") if isinstance(pos, dict) else None
        if opt_hold_entry is None and isinstance(pos, dict):
            opt_hold_entry = pos.get("opt_hold_sec")
        opt_hold_curr = pos.get("opt_hold_curr_sec") if isinstance(pos, dict) else None
        if opt_hold_curr is None and isinstance(pos, dict):
            opt_hold_curr = pos.get("opt_hold_sec")
        opt_hold_rem = pos.get("opt_hold_curr_remaining_sec") if isinstance(pos, dict) else None
        opt_hold_ratio = None
        try:
            if hold_duration_sec is not None and opt_hold_entry is not None and float(opt_hold_entry) > 0:
                opt_hold_ratio = float(hold_duration_sec) / float(opt_hold_entry)
        except Exception:
            opt_hold_ratio = None
        # normalize numeric fields
        pnl_val = None if pnl is None else float(pnl)
        notional = pos.get("notional")
        lev = pos.get("leverage")
        entry_lev = pos.get("entry_leverage")
        lev_for_roe = entry_lev
        try:
            if lev_for_roe in (None, 0):
                lev_for_roe = lev
            lev_for_roe = float(lev_for_roe) if lev_for_roe is not None else None
        except Exception:
            lev_for_roe = None
        base_notional = None
        if notional is not None and lev_for_roe not in (None, 0):
            try:
                base_notional = float(notional) / float(max(lev_for_roe, 1e-6))
            except Exception:
                base_notional = None
        roe_val = None
        if pnl_val is not None and base_notional:
            try:
                roe_val = pnl_val / base_notional
            except Exception:
                roe_val = None
        trade_uid = self._new_trade_uid()
        ttype_u = str(ttype or "").upper()
        open_types = {"ENTER", "SPREAD"}
        entry_link_id = None
        if isinstance(pos, dict):
            entry_link_id = pos.get("entry_link_id")
            if not entry_link_id:
                entry_link_id = pos.get("entry_id")
        if ttype_u in open_types and isinstance(pos, dict):
            if not entry_link_id:
                entry_link_id = trade_uid
                pos["entry_link_id"] = entry_link_id
                pos["entry_id"] = entry_link_id
        if not entry_link_id and isinstance(pos, dict):
            try:
                e_ts = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
            except Exception:
                e_ts = now_ms()
            entry_link_id = f"{sym}:{side}:{int(e_ts)}"
            pos["entry_link_id"] = entry_link_id
            pos["entry_id"] = entry_link_id
        entry_id = str(entry_link_id) if entry_link_id else None
        if isinstance(pos, dict):
            if entry_id and not pos.get("entry_id"):
                pos["entry_id"] = entry_id
            if entry_id and not pos.get("entry_link_id"):
                pos["entry_link_id"] = entry_id
        # For exit-type records, adjust PnL to include entry fee allocation
        exit_types = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}
        entry_fee_alloc = None
        pnl_total = pnl_val
        if ttype and str(ttype).upper() in exit_types and pnl_val is not None:
            try:
                entry_fee_alloc = float(pos.get("fee_paid", 0.0) or 0.0)
            except Exception:
                entry_fee_alloc = None
            if entry_fee_alloc is not None:
                pnl_total = float(pnl_val) - float(entry_fee_alloc)
                if base_notional:
                    try:
                        roe_val = pnl_total / base_notional
                    except Exception:
                        pass

        entry = {
            "time": ts,
            "type": ttype,
            "ttype": ttype,
            "trade_uid": trade_uid,
            "entry_id": entry_id,
            "entry_link_id": entry_link_id,
            "symbol": sym,
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "pnl": pnl_total,
            "pnl_gross": pnl_val,
            "entry_fee_alloc": entry_fee_alloc,
            "roe": roe_val,
            "notional": notional,
            "leverage": lev_for_roe if lev_for_roe not in (None, 0) else lev,
            "leverage_effective": lev,
            "entry_leverage": lev_for_roe if lev_for_roe not in (None, 0) else None,
            "fee": None if fee is None else float(fee),
            "tag": pos.get("tag"),
            "reason": reason,
            "exit_kind": exit_kind,
            "pred_win": pos.get("pred_win"),
            "pred_ev": pos.get("pred_ev"),
            "pred_event_ev_r": pos.get("pred_event_ev_r"),
            "pred_event_p_tp": pos.get("pred_event_p_tp"),
            "pred_event_p_sl": pos.get("pred_event_p_sl"),
            "entry_event_exit_ok": pos.get("entry_event_exit_ok"),
            "event_exit_hit": pos.get("event_exit_hit"),
            "event_exit_mode": pos.get("event_exit_mode"),
            "event_exit_score": pos.get("event_exit_score"),
            "event_cvar_pct": pos.get("event_cvar_pct"),
            "event_p_tp": pos.get("event_p_tp"),
            "event_p_sl": pos.get("event_p_sl"),
            "event_exit_threshold_score": pos.get("event_exit_threshold_score"),
            "event_exit_threshold_psl": pos.get("event_exit_threshold_psl"),
            "event_exit_threshold_cvar": pos.get("event_exit_threshold_cvar"),
            "event_exit_threshold_ptp": pos.get("event_exit_threshold_ptp"),
            "event_exit_strict_mode": pos.get("event_exit_strict_mode"),
            "event_eval_source": pos.get("event_eval_source"),
            "event_eval_ts_ms": pos.get("event_eval_ts_ms"),
            "event_precheck_allow_exit_now": pos.get("event_precheck_allow_exit_now"),
            "event_precheck_guard_reason": pos.get("event_precheck_guard_reason"),
            "event_precheck_confirm_mode": pos.get("event_precheck_confirm_mode"),
            "event_precheck_confirm_required": pos.get("event_precheck_confirm_required"),
            "event_precheck_confirm_count": pos.get("event_precheck_confirm_count"),
            "event_precheck_confirmed": pos.get("event_precheck_confirmed"),
            "event_precheck_shock_score": pos.get("event_precheck_shock_score"),
            "event_precheck_severe_adverse": pos.get("event_precheck_severe_adverse"),
            "event_precheck_severe_adverse_ptp_low": pos.get("event_precheck_severe_adverse_ptp_low"),
            "event_precheck_guard_progress": pos.get("event_precheck_guard_progress"),
            "event_precheck_guard_required": pos.get("event_precheck_guard_required"),
            "event_precheck_guard_hold_target_sec": pos.get("event_precheck_guard_hold_target_sec"),
            "event_precheck_guard_age_sec": pos.get("event_precheck_guard_age_sec"),
            "event_precheck_guard_remaining_sec": pos.get("event_precheck_guard_remaining_sec"),
            "event_exit_guard": pos.get("event_exit_guard"),
            "event_exit_guard_progress": pos.get("event_exit_guard_progress"),
            "event_exit_guard_required": pos.get("event_exit_guard_required"),
            "event_exit_guard_hold_target_sec": pos.get("event_exit_guard_hold_target_sec"),
            "event_exit_guard_age_sec": pos.get("event_exit_guard_age_sec"),
            "event_exit_guard_remaining_sec": pos.get("event_exit_guard_remaining_sec"),
            "event_exit_confirm_mode": pos.get("event_exit_confirm_mode"),
            "event_exit_confirm_required": pos.get("event_exit_confirm_required"),
            "event_exit_confirm_count": pos.get("event_exit_confirm_count"),
            "event_exit_confirmed": pos.get("event_exit_confirmed"),
            "event_hold_target_sec": pos.get("event_hold_target_sec"),
            "event_hold_remaining_sec": pos.get("event_hold_remaining_sec"),
            "pred_mu_alpha": pos.get("pred_mu_alpha"),
            "pred_mu_alpha_raw": pos.get("pred_mu_alpha_raw"),
            "pred_mu_dir_conf": pos.get("pred_mu_dir_conf"),
            "pred_mu_dir_edge": pos.get("pred_mu_dir_edge"),
            "pred_hmm_state": pos.get("pred_hmm_state"),
            "pred_hmm_conf": pos.get("pred_hmm_conf"),
            "regime": pos.get("regime"),
            "alpha_vpin": pos.get("alpha_vpin"),
            "alpha_hurst": pos.get("alpha_hurst"),
            "policy_score_threshold": pos.get("policy_score_threshold"),
            "policy_event_exit_min_score": pos.get("policy_event_exit_min_score"),
            "policy_unrealized_dd_floor": pos.get("policy_unrealized_dd_floor"),
            "policy_min_hold_eff_sec": pos.get("policy_min_hold_eff_sec"),
            "consensus_used": pos.get("consensus_used"),
            "realized_r": float(realized_r) if realized_r is not None else (pnl_total / base_notional if (pnl_total is not None and base_notional) else None),
            "hit": int(hit) if hit is not None else None,
            "hold_duration_sec": hold_duration_sec,
            "opt_hold_entry_sec": opt_hold_entry,
            "opt_hold_curr_sec": opt_hold_curr,
            "opt_hold_curr_remaining_sec": opt_hold_rem,
            "opt_hold_ratio": opt_hold_ratio,
            "external_close_cause": pos.get("external_close_cause"),
            "external_close_source": pos.get("external_close_source"),
            "external_close_miss_cnt": pos.get("external_close_miss_cnt"),
            "external_close_detail": pos.get("external_close_detail"),
            "entry_quality_score": pos.get("entry_quality_score"),
            "one_way_move_score": pos.get("one_way_move_score"),
            "leverage_signal_score": pos.get("leverage_signal_score"),
            "lev_source": pos.get("lev_source"),
            "lev_raw_before_caps": pos.get("lev_raw_before_caps"),
            "lev_raw_ev_component": pos.get("lev_raw_ev_component"),
            "lev_signal_target": pos.get("lev_signal_target"),
            "lev_signal_blend": pos.get("lev_signal_blend"),
            "lev_target_min": pos.get("lev_target_min"),
            "lev_target_max": pos.get("lev_target_max"),
            "lev_target_max_base": pos.get("lev_target_max_base"),
            "lev_liq_cap": pos.get("lev_liq_cap"),
            "lev_dynamic_scale": pos.get("lev_dynamic_scale"),
            "lev_dynamic_tox": pos.get("lev_dynamic_tox"),
            "lev_dynamic_conf_deficit": pos.get("lev_dynamic_conf_deficit"),
            "lev_dynamic_sigma_stress": pos.get("lev_dynamic_sigma_stress"),
            "lev_dynamic_vpin": pos.get("lev_dynamic_vpin"),
            "lev_dynamic_vpin_for_lev": pos.get("lev_dynamic_vpin_for_lev"),
            "lev_dynamic_hurst": pos.get("lev_dynamic_hurst"),
            "lev_dynamic_mu_align": pos.get("lev_dynamic_mu_align"),
            "lev_balance_reject_scale": pos.get("lev_balance_reject_scale"),
            "lev_balance_reject_count": pos.get("lev_balance_reject_count"),
            "lev_balance_reject_min_count": pos.get("lev_balance_reject_min_count"),
            "lev_balance_reject_min_scale": pos.get("lev_balance_reject_min_scale"),
            "lev_balance_reject_relief_signal": pos.get("lev_balance_reject_relief_signal"),
            "lev_balance_reject_relief_conf": pos.get("lev_balance_reject_relief_conf"),
            "lev_balance_reject_relief_pow": pos.get("lev_balance_reject_relief_pow"),
            "lev_sigma_raw": pos.get("lev_sigma_raw"),
            "lev_sigma_used": pos.get("lev_sigma_used"),
            "lev_sigma_stress_clip": pos.get("lev_sigma_stress_clip"),
        }
        self.trade_tape.append(entry)
        if self._is_exit_trade_record(entry):
            try:
                self._closed_trade_count = int(self._closed_trade_count) + 1
            except Exception:
                self._refresh_closed_trade_counter()
            try:
                self._append_event_alignment_sample(
                    reason=reason,
                    entry_event_exit_ok=pos.get("entry_event_exit_ok") if isinstance(pos, dict) else None,
                    symbol=sym,
                    ts_ms=now_ms(),
                    event_mode=(pos.get("event_exit_mode") if isinstance(pos, dict) else None),
                    event_hit=(pos.get("event_exit_hit") if isinstance(pos, dict) else None),
                    strict_mode=(pos.get("event_exit_strict_mode") if isinstance(pos, dict) else None),
                )
            except Exception:
                pass
        
        # ---- SQLite 저장 (비동기)
        try:
            trade_data = {
                "symbol": sym,
                "side": side,
                "action": ttype,  # "ENTRY" or "EXIT"
                "fill_price": float(price),
                "qty": float(qty),
                "notional": notional,
                "leverage": lev_for_roe if lev_for_roe not in (None, 0) else lev,
                "leverage_effective": lev,
                "entry_leverage": lev_for_roe if lev_for_roe not in (None, 0) else None,
                "timestamp_ms": int(time.time() * 1000),
                "fee": float(fee) if fee else None,
                "realized_pnl": pnl_val,
                "roe": roe_val,
                "hold_duration_sec": hold_duration_sec,
                "trade_uid": trade_uid,
                "entry_id": entry_id,
                "entry_link_id": entry_link_id,
                "entry_ev": pos.get("pred_ev"),
                "entry_kelly": pos.get("kelly"),
                "entry_reason": reason,
                "regime": pos.get("regime"),
                "alpha_vpin": pos.get("alpha_vpin"),
                "alpha_hurst": pos.get("alpha_hurst"),
                "pred_mu_alpha": pos.get("pred_mu_alpha"),
                "pred_mu_dir_conf": pos.get("pred_mu_dir_conf"),
                "policy_score_threshold": pos.get("policy_score_threshold"),
                "policy_event_exit_min_score": pos.get("policy_event_exit_min_score"),
                "policy_unrealized_dd_floor": pos.get("policy_unrealized_dd_floor"),
                "policy_min_hold_eff_sec": pos.get("policy_min_hold_eff_sec"),
                "entry_event_exit_ok": pos.get("entry_event_exit_ok"),
                "event_exit_hit": pos.get("event_exit_hit"),
                "event_exit_mode": pos.get("event_exit_mode"),
                "event_exit_score": pos.get("event_exit_score"),
                "event_cvar_pct": pos.get("event_cvar_pct"),
                "event_p_tp": pos.get("event_p_tp"),
                "event_p_sl": pos.get("event_p_sl"),
                "event_exit_threshold_score": pos.get("event_exit_threshold_score"),
                "event_exit_threshold_psl": pos.get("event_exit_threshold_psl"),
                "event_exit_threshold_cvar": pos.get("event_exit_threshold_cvar"),
                "event_exit_threshold_ptp": pos.get("event_exit_threshold_ptp"),
                "event_exit_strict_mode": pos.get("event_exit_strict_mode"),
                "event_eval_source": pos.get("event_eval_source"),
                "event_eval_ts_ms": pos.get("event_eval_ts_ms"),
                "event_precheck_allow_exit_now": pos.get("event_precheck_allow_exit_now"),
                "event_precheck_guard_reason": pos.get("event_precheck_guard_reason"),
                "event_precheck_confirm_mode": pos.get("event_precheck_confirm_mode"),
                "event_precheck_confirm_required": pos.get("event_precheck_confirm_required"),
                "event_precheck_confirm_count": pos.get("event_precheck_confirm_count"),
                "event_precheck_confirmed": pos.get("event_precheck_confirmed"),
                "event_precheck_shock_score": pos.get("event_precheck_shock_score"),
                "event_precheck_severe_adverse": pos.get("event_precheck_severe_adverse"),
                "event_precheck_severe_adverse_ptp_low": pos.get("event_precheck_severe_adverse_ptp_low"),
                "event_precheck_guard_progress": pos.get("event_precheck_guard_progress"),
                "event_precheck_guard_required": pos.get("event_precheck_guard_required"),
                "event_precheck_guard_hold_target_sec": pos.get("event_precheck_guard_hold_target_sec"),
                "event_precheck_guard_age_sec": pos.get("event_precheck_guard_age_sec"),
                "event_precheck_guard_remaining_sec": pos.get("event_precheck_guard_remaining_sec"),
                "event_exit_guard": pos.get("event_exit_guard"),
                "event_exit_guard_progress": pos.get("event_exit_guard_progress"),
                "event_exit_guard_required": pos.get("event_exit_guard_required"),
                "event_exit_guard_hold_target_sec": pos.get("event_exit_guard_hold_target_sec"),
                "event_exit_guard_age_sec": pos.get("event_exit_guard_age_sec"),
                "event_exit_guard_remaining_sec": pos.get("event_exit_guard_remaining_sec"),
                "event_exit_confirm_mode": pos.get("event_exit_confirm_mode"),
                "event_exit_confirm_required": pos.get("event_exit_confirm_required"),
                "event_exit_confirm_count": pos.get("event_exit_confirm_count"),
                "event_exit_confirmed": pos.get("event_exit_confirmed"),
                "event_hold_target_sec": pos.get("event_hold_target_sec"),
                "event_hold_remaining_sec": pos.get("event_hold_remaining_sec"),
                "external_close_cause": pos.get("external_close_cause"),
                "external_close_source": pos.get("external_close_source"),
                "external_close_miss_cnt": pos.get("external_close_miss_cnt"),
                "external_close_detail": pos.get("external_close_detail"),
                "entry_quality_score": pos.get("entry_quality_score"),
                "one_way_move_score": pos.get("one_way_move_score"),
                "leverage_signal_score": pos.get("leverage_signal_score"),
                "lev_source": pos.get("lev_source"),
                "lev_raw_before_caps": pos.get("lev_raw_before_caps"),
                "lev_raw_ev_component": pos.get("lev_raw_ev_component"),
                "lev_signal_target": pos.get("lev_signal_target"),
                "lev_signal_blend": pos.get("lev_signal_blend"),
                "lev_target_min": pos.get("lev_target_min"),
                "lev_target_max": pos.get("lev_target_max"),
                "lev_target_max_base": pos.get("lev_target_max_base"),
                "lev_liq_cap": pos.get("lev_liq_cap"),
                "lev_dynamic_scale": pos.get("lev_dynamic_scale"),
                "lev_dynamic_tox": pos.get("lev_dynamic_tox"),
                "lev_dynamic_conf_deficit": pos.get("lev_dynamic_conf_deficit"),
                "lev_dynamic_sigma_stress": pos.get("lev_dynamic_sigma_stress"),
                "lev_dynamic_vpin": pos.get("lev_dynamic_vpin"),
                "lev_dynamic_vpin_for_lev": pos.get("lev_dynamic_vpin_for_lev"),
                "lev_dynamic_hurst": pos.get("lev_dynamic_hurst"),
                "lev_dynamic_mu_align": pos.get("lev_dynamic_mu_align"),
                "lev_balance_reject_scale": pos.get("lev_balance_reject_scale"),
                "lev_balance_reject_count": pos.get("lev_balance_reject_count"),
                "lev_balance_reject_min_count": pos.get("lev_balance_reject_min_count"),
                "lev_balance_reject_min_scale": pos.get("lev_balance_reject_min_scale"),
                "lev_balance_reject_relief_signal": pos.get("lev_balance_reject_relief_signal"),
                "lev_balance_reject_relief_conf": pos.get("lev_balance_reject_relief_conf"),
                "lev_balance_reject_relief_pow": pos.get("lev_balance_reject_relief_pow"),
                "lev_sigma_raw": pos.get("lev_sigma_raw"),
                "lev_sigma_used": pos.get("lev_sigma_used"),
                "lev_sigma_stress_clip": pos.get("lev_sigma_stress_clip"),
            }
            self.db.log_trade_background(trade_data, mode=self._trading_mode)
        except Exception as e:
            self._log_err(f"[DB] log_trade failed: {e}")
        
        if ttype == "EXIT" and pnl_val is not None:
            if pnl_val < 0:
                self._loss_streak += 1
                if self._loss_streak >= LOSS_STREAK_LIMIT:
                    self._register_anomaly(
                        "loss_streak",
                        "warn",
                        f"{sym} 연속 손절 {self._loss_streak}회",
                        {"symbol": sym, "loss_streak": self._loss_streak},
                    )
            else:
                self._loss_streak = 0
        if reason:
            reason_l = str(reason).lower()
            if "liquidat" in reason_l or "emergency_stop" in reason_l or "drawdown" in reason_l or "kill" in reason_l:
                self._register_anomaly(
                    "liquidation",
                    "critical",
                    f"{sym} 강제 청산 감지 ({reason})",
                    {"symbol": sym, "reason": reason},
                )
        self._persist_state(force=True)

    def _consensus_used_flag(self, decision: dict) -> bool:
        meta = decision.get("meta") or {}
        if meta.get("consensus_used"):
            return True
        for d in decision.get("details", []) or []:
            m = d.get("meta") or {}
            if m.get("consensus_used"):
                return True
        return False

    def _ema_update(self, store: dict, key, x: float, half_life_sec: float, ts_ms: int) -> float:
        prev = store.get(key)
        if prev is None:
            store[key] = (float(x), ts_ms)
            return float(x)
        prev_val, prev_ts = prev
        dt_sec = max(1.0, (ts_ms - prev_ts) / 1000.0)
        alpha = 1.0 - math.exp(-math.log(2) * dt_sec / max(half_life_sec, 1e-6))
        new_val = alpha * float(x) + (1.0 - alpha) * float(prev_val)
        store[key] = (new_val, ts_ms)
        return new_val

    @staticmethod
    def _calc_rsi(closes, period: int = RSI_PERIOD) -> float | None:
        if closes is None or len(closes) <= period:
            return None
        gains = []
        losses = []
        for i in range(1, period + 1):
            delta = closes[-i] - closes[-i - 1]
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _consensus_action(self, decision: dict, ctx: dict) -> tuple[str, float]:
        """
        여러 지표를 투표/가중합하여 방향 합의 점수와 액션을 결정.
        score > CONSENSUS_THRESHOLD => LONG, score < -CONSENSUS_THRESHOLD => SHORT, else WAIT
        """
        score = 0.0
        base_action = decision.get("action") if decision else "WAIT"
        ev = float(decision.get("ev", 0.0)) if decision else 0.0
        meta = (decision.get("meta") or {}) if decision else {}
        cvar = float(meta.get("cvar05", decision.get("cvar", 0.0)) if decision else 0.0)
        win = float(decision.get("confidence", 0.0)) if decision else 0.0
        direction = int(ctx.get("direction", 1))
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))
        ofi = float(ctx.get("ofi_score", 0.0))
        rsi = self._calc_rsi(ctx.get("closes"), period=RSI_PERIOD)
        closes = ctx.get("closes") or []
        sym = ctx.get("symbol")
        spread_pct = self._safe_float(meta.get("spread_pct", ctx.get("spread_pct")), 0.0)
        liq = self._liquidity_score(sym) if sym else 1.0
        p_sl = float(meta.get("event_p_sl", 0.0) or 0.0)

        if ev <= 0:
            return "WAIT", score

        # --- history update for z-scores (regime/session residual 기반) ---
        def _rz(val, hist_deque: deque | None, default_scale: float = 0.001):
            try:
                if hist_deque is None or len(hist_deque) < 5:
                    return float(val / max(default_scale, 1e-6))
                arr = np.asarray(hist_deque, dtype=np.float64)
                mean = float(arr.mean())
                std = float(arr.std())
                std = std if std > 1e-6 else default_scale
                return float((val - mean) / std)
            except Exception:
                return float(val / max(default_scale, 1e-6))

        key = (regime, session)
        ev_hist_rs = self._ev_regime_hist_rs.setdefault(key, deque(maxlen=500))
        cvar_hist_rs = self._cvar_regime_hist_rs.setdefault(key, deque(maxlen=500))
        ev_hist_rs.append(ev)
        cvar_hist_rs.append(cvar)

        ofi_hist_base = self._ofi_regime_hist.setdefault(key, deque(maxlen=500))
        ofi_hist_base.append(ofi)
        ofi_mean = float(np.mean(ofi_hist_base)) if ofi_hist_base else 0.0
        ofi_residual = ofi - ofi_mean
        ofi_resid_hist = self._ofi_resid_hist.setdefault(key, deque(maxlen=500))
        ofi_resid_hist.append(ofi_residual)

        spread_hist = self._spread_regime_hist.setdefault(key, deque(maxlen=500))
        spread_hist.append(spread_pct)
        liq_hist = self._liq_regime_hist.setdefault(key, deque(maxlen=500))
        liq_hist.append(liq)

        z_ev = _rz(ev, ev_hist_rs)
        z_cvar = _rz(-cvar, cvar_hist_rs, default_scale=0.0005)
        z_ofi = _rz(ofi_residual, ofi_resid_hist, default_scale=0.0005)
        z_spread = _rz(spread_pct, spread_hist, default_scale=0.0002)
        z_liq = _rz(liq, liq_hist, default_scale=1.0)

        bias = float(direction)

        score = (
            1.2 * z_ev
            + 0.9 * z_cvar  # -CVaR를 넣어 양수일수록 좋게
            - 0.7 * p_sl
            + 0.5 * z_ofi
            + 0.5 * bias
            + 0.3 * z_liq
            - 0.6 * z_spread
        )

        # RSI as mild tie-breaker
        if rsi is not None:
            if rsi >= RSI_LONG:
                score += 0.2
            elif rsi <= RSI_SHORT:
                score -= 0.2

        action = "WAIT"
        if score >= CONSENSUS_THRESHOLD:
            action = "LONG"
        elif score <= -CONSENSUS_THRESHOLD:
            action = "SHORT"
        else:
            action = base_action if base_action in ("LONG", "SHORT") else "WAIT"
        return action, score

    def _spread_signal(self, base: str, quote: str) -> tuple[str, str, float] | None:
        """
        단순 비율 mean-reversion 스프레드.
        ratio = base/quote. z > entry: short base / long quote, z < -entry: long base / short quote.
        """
        base_closes = list(self.ohlcv_buffer.get(base) or [])
        quote_closes = list(self.ohlcv_buffer.get(quote) or [])
        if len(base_closes) < SPREAD_LOOKBACK + 1 or len(quote_closes) < SPREAD_LOOKBACK + 1:
            return None
        ratio = [b / q for b, q in zip(base_closes[-SPREAD_LOOKBACK:], quote_closes[-SPREAD_LOOKBACK:]) if q]
        if len(ratio) < SPREAD_LOOKBACK:
            return None
        mean = sum(ratio) / len(ratio)
        var = sum((x - mean) ** 2 for x in ratio) / len(ratio)
        std = math.sqrt(max(var, 1e-9))
        latest = ratio[-1]
        z = (latest - mean) / std if std > 0 else 0.0
        if abs(z) < SPREAD_Z_ENTRY:
            return None
        if z > 0:
            return ("SHORT", "LONG", z)
        return ("LONG", "SHORT", z)

    def _manage_spreads(self, ts: int):
        """
        스프레드 진입/청산. 페어 양쪽 포지션이 모두 없을 때만 진입.
        """
        if not self.spread_enabled:
            return
        for base, quote in self.spread_pairs:
            base_px = self.market.get(base, {}).get("price")
            quote_px = self.market.get(quote, {}).get("price")
            if base_px is None or quote_px is None:
                continue
            signal = self._spread_signal(base, quote)
            base_pos = self.positions.get(base)
            quote_pos = self.positions.get(quote)

            # 청산 조건: 태그가 spread이고 z가 수렴하거나 보유 초과
            if (base_pos and base_pos.get("tag") == "spread") or (quote_pos and quote_pos.get("tag") == "spread"):
                # 재계산 z
                zinfo = self._spread_signal(base, quote)
                z_now = zinfo[2] if zinfo else 0.0
                should_exit = abs(z_now) <= SPREAD_Z_EXIT
                age_ok = False
                if base_pos:
                    age_ok = age_ok or (ts - base_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if quote_pos:
                    age_ok = age_ok or (ts - quote_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if should_exit or age_ok:
                    if base_pos:
                        self._close_position(base, float(base_px), f"spread exit z={z_now:.2f}")
                    if quote_pos:
                        self._close_position(quote, float(quote_px), f"spread exit z={z_now:.2f}")
                    continue

            # 진입: 두 심볼 모두 포지션 없고 스프레드 신호가 명확할 때
            if signal and not base_pos and not quote_pos:
                base_side, quote_side, z = signal
                hold_override = min(SPREAD_HOLD_SEC, MAX_POSITION_HOLD_SEC)
                ctx_spread = {"regime": "spread", "session": time_regime()}
                meta_spread = {"regime": "spread", "session": time_regime()}
                self._enter_position(
                    base, base_side, float(base_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._enter_position(
                    quote, quote_side, float(quote_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._log(f"[SPREAD] {base}/{quote} z={z:.2f} -> {base_side}/{quote_side}")
                self._last_actions[base] = "SPREAD_ENTER"
                self._last_actions[quote] = "SPREAD_ENTER"

    # ======================================================================
    # [REFACTOR] 3-Stage Decision Pipeline: Context → Decide → Apply
    # ======================================================================

    def _build_decision_context(self, sym: str, ts: int) -> dict | None:
        """
        Stage 1: 단일 심볼에 대한 의사결정 컨텍스트 생성.
        Returns None if symbol data is not ready (e.g., price=None).
        """
        price = self.market[sym]["price"]
        closes = list(self.ohlcv_buffer[sym])
        opens = list(self.ohlcv_open[sym])
        highs = list(self.ohlcv_high[sym])
        lows = list(self.ohlcv_low[sym])
        volumes = list(self.ohlcv_volume[sym])
        candles = len(closes)

        if price is None:
            return None

        mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
        regime = self._infer_regime(closes)

        # Orderbook spread
        spread_pct = None
        ob = self.orderbook.get(sym)
        if ob and ob.get("ready"):
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if bids and asks and len(bids[0]) >= 2 and len(asks[0]) >= 2:
                bid = float(bids[0][0])
                ask = float(asks[0][0])
                mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
                if mid > 0:
                    spread_pct = (ask - bid) / mid

        # Annualized μ/σ
        mu_base, sigma = (0.0, 0.0)
        if mu_bar is not None and sigma_bar is not None:
            mu_base, sigma = self._annualize_mu_sigma(mu_bar, sigma_bar, bar_seconds=60.0)

        # Regime table blend
        session = time_regime()
        mu_tab, sig_tab = get_regime_mu_sigma(regime, session, symbol=sym)
        if mu_tab is not None and sig_tab is not None:
            w = 0.35
            mu_base = float((1.0 - w) * float(mu_base) + w * float(mu_tab))
            sigma = float(max(1e-6, (1.0 - w) * float(sigma) + w * float(sig_tab)))

        ofi_score = float(self._compute_ofi_score(sym))
        impulse = self._compute_impulse_features(closes, opens, highs, lows, volumes)
        rt_breakout = self._compute_realtime_breakout_features(float(price), highs, lows, volumes)
        tick_feats = self._compute_tick_features(sym, ts)
        alpha_feats = self._update_alpha_state(sym, ts, float(price), closes, volumes)
        pos = self.positions.get(sym, {})
        pos_size_frac = pos.get("size_frac") if pos else None
        if pos_size_frac is None:
            pos_size_frac = self._kelly_allocations.get(sym, self.default_size_frac)
        try:
            pos_size_frac = float(pos_size_frac)
        except Exception:
            pos_size_frac = float(self.default_size_frac)
        pos_leverage = pos.get("leverage") if pos else None
        if pos_leverage is None:
            pos_leverage = self._dyn_leverage.get(sym, self.leverage)
        try:
            pos_leverage = float(pos_leverage)
        except Exception:
            pos_leverage = float(self.leverage)
        tuner = getattr(self, "tuner", None)
        regime_params = None
        if tuner and hasattr(tuner, "get_params"):
            try:
                regime_params = tuner.get_params(regime)
            except Exception:
                pass

        # Bootstrap returns for tail
        bootstrap_returns = None
        if closes is not None and len(closes) >= 64:
            try:
                x = np.asarray(closes, dtype=np.float64)
                bootstrap_returns = np.diff(np.log(np.maximum(x, 1e-12))).astype(np.float64)[-512:]
            except Exception:
                pass

        is_dev_mode = bool(getattr(config, "DEV_MODE", False))

        ctx_out = {
            "symbol": sym,
            "price": float(price),
            "bar_seconds": 60.0,
            "closes": closes,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "candles": candles,
            "direction": self._direction_bias(closes),
            "regime": regime,
            "ofi_score": float(ofi_score),
            "liquidity_score": self._liquidity_score(sym),
            "leverage": None,
            "size_frac": pos_size_frac,
            "leverage": pos_leverage,
            "mu_base": float(mu_base),
            "sigma": float(max(sigma, 0.0)),
            "regime_params": regime_params,
            "session": session,
            "spread_pct": spread_pct,
            "use_torch": True,
            "use_jax": False,
            "tail_mode": "student_t",
            "tail_model": "student_t",
            "tail_df": 6.0,
            "student_t_df": 6.0,
            "bootstrap_returns": bootstrap_returns,
            "ev": None,
            "equity": float(self.balance),
            "hybrid_entry_floor_dyn": float(self._hybrid_entry_floor_dyn) if self._hybrid_entry_floor_dyn is not None else None,
            "hybrid_conf_scale_dyn": float(self._hybrid_conf_scale_dyn) if self._hybrid_conf_scale_dyn is not None else None,
        }
        if alpha_feats:
            ctx_out.update(alpha_feats)
        if impulse:
            ctx_out.update(impulse)
        if rt_breakout:
            ctx_out.update(rt_breakout)
        if tick_feats:
            ctx_out.update(tick_feats)
        return ctx_out

    def _build_batch_context_soa(self, ts: int) -> tuple[list[dict], np.ndarray]:
        """
        [SoA OPTIMIZATION] Structure of Arrays 방식으로 배치 컨텍스트 생성.
        
        Returns:
            - ctx_list: 유효한 심볼의 컨텍스트 리스트 (기존 호환성 유지)
            - valid_indices: 유효한 심볼의 인덱스 배열 (GPU 배열 인덱싱용)
        
        CRITICAL OPTIMIZATION:
        - Pre-allocated 배열에 직접 값 할당 (메모리 재할당 없음)
        - Dict 생성 최소화 (필수 필드만 포함)
        - O(1) 심볼 인덱스 조회
        """
        # Reset valid mask and arrays
        self._batch_valid_mask.fill(False)

        try:
            ctx_latency_budget_ms = float(os.environ.get("DECISION_CTX_LATENCY_BUDGET_MS", 0.0) or 0.0)
        except Exception:
            ctx_latency_budget_ms = 0.0
        ctx_budget_enabled = bool(ctx_latency_budget_ms > 0.0)
        try:
            skip_heavy_after_budget = str(os.environ.get("DECISION_CTX_SKIP_HEAVY_ALPHA_ON_BUDGET", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            skip_heavy_after_budget = True
        ctx_t0 = time.perf_counter()
        budget_triggered = False
        fast_alpha_count = 0

        ctx_list = []
        valid_indices = []

        for sym in SYMBOLS:
            idx = self._sym_to_idx.get(sym)
            if idx is None:
                continue

            price = self.market[sym]["price"]
            if price is None:
                continue

            closes = list(self.ohlcv_buffer[sym])
            opens = list(self.ohlcv_open[sym])
            highs = list(self.ohlcv_high[sym])
            lows = list(self.ohlcv_low[sym])
            volumes = list(self.ohlcv_volume[sym])
            candles = len(closes)

            # Compute mu/sigma directly into pre-allocated arrays
            mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
            regime = self._infer_regime(closes)

            # Annualized μ/σ
            mu_base, sigma = (0.0, 0.0)
            if mu_bar is not None and sigma_bar is not None:
                mu_base, sigma = self._annualize_mu_sigma(mu_bar, sigma_bar, bar_seconds=60.0)

            # Regime table blend
            session = time_regime()
            mu_tab, sig_tab = get_regime_mu_sigma(regime, session, symbol=sym)
            if mu_tab is not None and sig_tab is not None:
                w = 0.35
                mu_base = float((1.0 - w) * float(mu_base) + w * float(mu_tab))
                sigma = float(max(1e-6, (1.0 - w) * float(sigma) + w * float(sig_tab)))

            ofi_score = float(self._compute_ofi_score(sym))
            impulse = self._compute_impulse_features(closes, opens, highs, lows, volumes)
            rt_breakout = self._compute_realtime_breakout_features(float(price), highs, lows, volumes)
            tick_feats = self._compute_tick_features(sym, ts)
            elapsed_ms = (time.perf_counter() - ctx_t0) * 1000.0
            alpha_fast_mode = bool(
                ctx_budget_enabled
                and skip_heavy_after_budget
                and elapsed_ms >= float(ctx_latency_budget_ms)
            )
            if alpha_fast_mode:
                budget_triggered = True
                fast_alpha_count += 1
            alpha_feats = self._update_alpha_state(sym, ts, float(price), closes, volumes, fast_mode=alpha_fast_mode)
            pos = self.positions.get(sym, {})
            pos_qty = float(pos.get("quantity", pos.get("qty", 0.0)) or 0.0)
            pos_side_val = 0
            if pos_qty != 0.0:
                side = str(pos.get("side", "")).upper()
                if side == "LONG":
                    pos_side_val = 1
                elif side == "SHORT":
                    pos_side_val = -1
            pos_size_frac = pos.get("size_frac") if pos else None
            if pos_size_frac is None:
                pos_size_frac = self._kelly_allocations.get(sym, self.default_size_frac)
            try:
                pos_size_frac = float(pos_size_frac)
            except Exception:
                pos_size_frac = float(self.default_size_frac)
            pos_leverage = pos.get("leverage") if pos else None
            if pos_leverage is None:
                pos_leverage = self._dyn_leverage.get(sym, self.leverage)
            try:
                pos_leverage = float(pos_leverage)
            except Exception:
                pos_leverage = float(self.leverage)

            # Fill pre-allocated arrays (Zero-Copy style)
            self._batch_prices[idx] = float(price)
            self._batch_mus[idx] = float(mu_base)
            self._batch_sigmas[idx] = float(max(sigma, 1e-6))
            self._batch_ofi_scores[idx] = ofi_score
            self._batch_valid_mask[idx] = True

            valid_indices.append(idx)

            # Build minimal ctx dict for backward compatibility
            # (향후 완전 SoA 전환 시 제거 가능)
            is_dev_mode = bool(getattr(config, "DEV_MODE", False))
            ctx = {
                "symbol": sym,
                "price": float(price),
                "bar_seconds": 60.0,
                "closes": closes,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
                "candles": candles,
                "direction": self._direction_bias(closes),
                "regime": regime,
                "ofi_score": ofi_score,
                "liquidity_score": self._liquidity_score(sym),
                "leverage": pos_leverage,
                "size_frac": pos_size_frac,
                "mu_base": float(mu_base),
                "sigma": float(max(sigma, 0.0)),
                "equity": float(self.balance),
                "session": session,
                "use_torch": True,
                "use_jax": False,
                "ev": None,
                "position_side": pos_side_val,
                "has_position": bool(pos_qty != 0.0),
                "_soa_idx": idx,  # SoA 배열 인덱스 참조
            }
            if alpha_feats:
                ctx.update(alpha_feats)
            if impulse:
                ctx.update(impulse)
            if rt_breakout:
                ctx.update(rt_breakout)
            if tick_feats:
                ctx.update(tick_feats)
            if self._hybrid_entry_floor_dyn is not None:
                ctx["hybrid_entry_floor_dyn"] = float(self._hybrid_entry_floor_dyn)
            if self._hybrid_conf_scale_dyn is not None:
                ctx["hybrid_conf_scale_dyn"] = float(self._hybrid_conf_scale_dyn)
            ctx_list.append(ctx)

        try:
            ctx_elapsed_ms = (time.perf_counter() - ctx_t0) * 1000.0
            self._last_ctx_build_ms = float(ctx_elapsed_ms)
            self._last_ctx_fast_alpha_count = int(fast_alpha_count)
            self._last_ctx_budget_enabled = bool(ctx_budget_enabled)
            self._last_ctx_budget_ms = float(ctx_latency_budget_ms)
            if budget_triggered and self._should_alert(
                "ctx_latency_budget",
                now_ts=time.time(),
                throttle_sec=30.0,
            ):
                self._log(
                    f"[LATENCY] ctx budget exceeded ({ctx_elapsed_ms:.1f}ms >= {ctx_latency_budget_ms:.1f}ms); "
                    f"alpha cache fallback symbols={int(fast_alpha_count)}"
                )
        except Exception:
            pass

        return ctx_list, np.array(valid_indices, dtype=np.int32)

    def get_batch_arrays(self) -> dict:
        """
        [SoA API] Pre-allocated 배열들을 반환.
        EngineHub.decide_batch_arrays()에서 사용.
        """
        return {
            "prices": self._batch_prices,
            "mus": self._batch_mus,
            "sigmas": self._batch_sigmas,
            "leverages": self._batch_leverages,
            "ofi_scores": self._batch_ofi_scores,
            "valid_mask": self._batch_valid_mask,
            "max_symbols": self._batch_max_symbols,
        }

    def _apply_decision(
        self,
        sym: str,
        decision: dict,
        ctx: dict,
        ts: int,
        log_this_cycle: bool,
    ) -> dict:
        """
        Stage 3: 의사결정 결과를 포지션/주문에 적용.
        Returns a row dict for dashboard broadcast.
        개별 심볼 예외는 내부에서 격리하여 처리.
        """
        price = ctx["price"]
        candles = ctx.get("candles", 0)
        regime = ctx.get("regime", "chop")
        session = ctx.get("session", "OFF")
        ofi_score = ctx.get("ofi_score", 0.0)

        try:
            # DEBUG: TP/SL keys from decision.meta
            if log_this_cycle and decision:
                meta = decision.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                if not meta:
                    details = decision.get("details")
                    if isinstance(details, list):
                        for d in details:
                            if not isinstance(d, dict):
                                continue
                            m2 = d.get("meta")
                            if isinstance(m2, dict) and m2:
                                meta = dict(m2)
                                break
                if meta and not isinstance(decision.get("meta"), dict):
                    decision["meta"] = meta
                keys = [
                    "mc_tp", "mc_sl", "tp", "sl",
                    "profit_target", "stop_loss",
                    "tp_pct", "sl_pct", "params",
                ]
                picked = {}
                for k in keys:
                    if k in meta:
                        picked[k] = meta.get(k)
                params = meta.get("params") or {}
                if isinstance(params, dict):
                    for k in ["profit_target", "stop_loss", "tp_pct", "sl_pct", "tp", "sl", "n_paths"]:
                        if k in params:
                            picked[f"params.{k}"] = params.get(k)
                self._log(f"[DBG_META_TPSL] {sym} action={decision.get('action')} picked={json.dumps(picked, ensure_ascii=False)}")

            ctx["ev"] = decision.get("ev", 0.0)
            side = decision.get("action_type") or decision.get("action")

            # MC cache
            price_bucket = round(float(price), 3)
            cache_key = (sym, side, regime, price_bucket)
            now_cache = time.time()
            cached = self.mc_cache.get(cache_key)
            if cached and (now_cache - cached[0] <= self.mc_cache_ttl):
                decision_meta = decision.get("meta") or {}
                decision_meta.update(cached[1])
                decision["meta"] = decision_meta
            else:
                self.mc_cache[cache_key] = (now_cache, decision.get("meta", {}))

            # Running stats update
            def _s(val, default=0.0):
                try:
                    return float(val) if val is not None else float(default)
                except Exception:
                    return float(default)

            self.stats.push("ev", (regime, session), _s(decision.get("ev")))
            self.stats.push("cvar", (regime, session), _s(decision.get("cvar")))
            spread_val = _s(decision.get("meta", {}).get("spread_pct", ctx.get("spread_pct")), 0.0)
            self.stats.push("spread", (regime, session), spread_val)
            self.stats.push("liq", (regime, session), _s(self._liquidity_score(sym)))
            self.stats.push("ofi", (regime, session), _s(ofi_score))
            ofi_mean = self.stats.ema_update("ofi_mean", (regime, session), ofi_score, half_life_sec=900)
            ofi_res = ofi_score - ofi_mean
            self.stats.push("ofi_res", (regime, session), ofi_res)

            # Dynamic leverage (force optimal leverage if present, else risk-based fallback)
            try:
                if decision is not None and isinstance(decision.get("meta"), dict):
                    opt_lev = decision.get("meta", {}).get("optimal_leverage")
                    if opt_lev is not None and decision.get("leverage") is None:
                        decision["leverage"] = float(opt_lev)
                        meta_tmp = dict(decision.get("meta") or {})
                        if meta_tmp.get("lev_source") is None:
                            meta_tmp["lev_source"] = "optimal_leverage"
                        decision["meta"] = meta_tmp
            except Exception:
                pass
            # If leverage stays at static default, force one pass of risk-based dynamic leverage.
            try:
                force_dyn_default = str(os.environ.get("LEVERAGE_DYNAMIC_OVERRIDE_DEFAULT", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                force_dyn_default = True
            try:
                if force_dyn_default and decision is not None:
                    def _fopt(v):
                        try:
                            if v is None:
                                return None
                            return float(v)
                        except Exception:
                            return None
                    meta_tmp = decision.get("meta") or {}
                    action_now = str(decision.get("action") or "").upper()
                    lev_now = _fopt(decision.get("leverage"))
                    if lev_now is None:
                        lev_now = _fopt(meta_tmp.get("lev") or meta_tmp.get("optimal_leverage"))
                    default_lev = _fopt(getattr(config, "DEFAULT_LEVERAGE", self.leverage))
                    if default_lev is None:
                        default_lev = _fopt(self.leverage)
                    if default_lev is None:
                        default_lev = 5.0
                    if action_now in ("LONG", "SHORT") and lev_now is not None and abs(float(lev_now) - float(default_lev)) < 1e-6:
                        dyn = self._dynamic_leverage_risk(decision, ctx)
                        if dyn is not None and math.isfinite(float(dyn)):
                            decision["leverage"] = float(dyn)
                            meta_tmp = dict(decision.get("meta") or {})
                            meta_tmp["lev"] = float(dyn)
                            meta_tmp["lev_source"] = "dynamic_risk_override_default"
                            meta_tmp["lev_override_from"] = float(lev_now)
                            decision["meta"] = meta_tmp
                            self._log(
                                f"[LEVERAGE] {sym} override default {float(lev_now):.2f}x -> {float(dyn):.2f}x "
                                f"(source=dynamic_risk_override_default)"
                            )
            except Exception:
                pass
            # If still no leverage, compute dynamic leverage from risk
            try:
                meta_tmp = decision.get("meta") or {}
                if decision is not None and decision.get("leverage") is None and meta_tmp.get("lev") is None:
                    dyn = self._dynamic_leverage_risk(decision, ctx)
                    if dyn is not None:
                        decision["leverage"] = float(dyn)
                        meta_tmp = dict(decision.get("meta") or {})
                        meta_tmp["lev"] = float(dyn)
                        if meta_tmp.get("lev_source") is None:
                            meta_tmp["lev_source"] = "dynamic_risk"
                        decision["meta"] = meta_tmp
            except Exception:
                pass
            try:
                meta_tmp = dict(decision.get("meta") or {})
                if meta_tmp.get("lev_source") is None:
                    if decision.get("leverage") is not None:
                        meta_tmp["lev_source"] = "decision"
                    elif meta_tmp.get("lev") is not None:
                        meta_tmp["lev_source"] = "meta"
                    decision["meta"] = meta_tmp
            except Exception:
                pass
            try:
                lev_fallback_min = float(os.environ.get("LEVERAGE_TARGET_MIN", os.environ.get("LEVERAGE_MIN", LEVERAGE_MIN)) or LEVERAGE_MIN)
            except Exception:
                try:
                    lev_fallback_min = float(LEVERAGE_MIN)
                except Exception:
                    lev_fallback_min = float(self.leverage)
            dyn_leverage = float(decision.get("leverage") or decision.get("meta", {}).get("lev") or lev_fallback_min or self.leverage)
            # Always run one dynamic-risk pass for entry actions so leverage signal/meta stay populated.
            try:
                action_now = str(decision.get("action") or "").upper() if decision else ""
                force_dynamic_pass = str(os.environ.get("LEVERAGE_FORCE_DYNAMIC_PASS", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                action_now = ""
                force_dynamic_pass = True
            if force_dynamic_pass and action_now in ("LONG", "SHORT") and decision is not None:
                try:
                    dyn_pass = self._dynamic_leverage_risk(decision, ctx)
                except Exception as e:
                    dyn_pass = None
                    self._log_err(f"[LEVERAGE] dynamic_risk_forced_pass failed: {sym} err={e}")
                if dyn_pass is not None and math.isfinite(float(dyn_pass)):
                    dyn_leverage = float(dyn_pass)
                    decision["leverage"] = float(dyn_pass)
                    meta_tmp = dict(decision.get("meta") or {})
                    meta_tmp["lev"] = float(dyn_pass)
                    lev_src_prev = str(meta_tmp.get("lev_source") or "").strip()
                    if (not lev_src_prev) or lev_src_prev.startswith("optimal") or lev_src_prev.startswith("decision"):
                        meta_tmp["lev_source"] = "dynamic_risk_forced_pass"
                    decision["meta"] = meta_tmp
            try:
                min_lev = float(LEVERAGE_MIN)
            except Exception:
                min_lev = 1.0
            try:
                max_lev = float(self.max_leverage or self.leverage or dyn_leverage)
            except Exception:
                max_lev = dyn_leverage
            if min_lev > 0:
                if dyn_leverage < min_lev:
                    meta_tmp = dict(decision.get("meta") or {})
                    meta_tmp["lev_min_applied"] = float(min_lev)
                    decision["meta"] = meta_tmp
                dyn_leverage = max(min_lev, dyn_leverage)
            if max_lev > 0:
                dyn_leverage = min(max_lev, dyn_leverage)
            ctx["leverage"] = dyn_leverage

            # Regime size cap (Kelly 모드에서는 개별 종목 상한을 제거)
            decision = dict(decision)
            decision_meta = dict(decision.get("meta") or {})
            if USE_KELLY_ALLOCATION:
                cap_frac_regime = 1.0
                sz = KELLY_PORTFOLIO_FRAC
            else:
                cap_map = {"bull": 0.25, "bear": 0.25, "chop": 0.10, "volatile": 0.08}
                cap_frac_regime = cap_map.get(regime, 0.10)
                sz = decision.get("size_frac") or decision_meta.get("size_fraction") or self.default_size_frac
            decision_meta["regime_cap_frac"] = cap_frac_regime
            decision["meta"] = decision_meta
            decision["size_frac"] = float(min(max(0.0, sz), cap_frac_regime))

            use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
            hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
            hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")

            # Hard gates: spread_pct, event_cvar_r
            spread_pct_now = decision_meta.get("spread_pct", ctx.get("spread_pct"))
            ev_cvar_r = decision_meta.get("event_cvar_r")
            spread_cap_map = {"bull": 0.0020, "bear": 0.0020, "chop": 0.0012, "volatile": 0.0008}
            spread_cap = spread_cap_map.get(regime, SPREAD_PCT_MAX)
            if spread_pct_now is not None and spread_cap is not None and spread_pct_now > spread_cap:
                decision["action"] = "WAIT"
                decision["reason"] = f"{decision.get('reason', '')} | spread_cap"

            cvar_floor_map = {"bull": -1.2, "bear": -1.2, "chop": -1.0, "volatile": -0.8}
            cvar_floor_regime = cvar_floor_map.get(regime, -1.0)
            if (not hybrid_only) and ev_cvar_r is not None and ev_cvar_r < cvar_floor_regime:
                decision["action"] = "WAIT"
                decision["reason"] = f"{decision.get('reason', '')} | event_cvar_floor"

            # Dynamic EV entry threshold
            ev_net_now = float(decision.get("ev", 0.0) or 0.0)
            hist_sym = self._ev_tune_hist[sym]
            hist_sym.append((ts, ev_net_now))
            cutoff = ts - int(EV_TUNE_WINDOW_SEC * 1000)
            while hist_sym and hist_sym[0][0] < cutoff:
                hist_sym.popleft()

            regime_key = (regime or "chop", session or "OFF")
            hist_reg = self._ev_regime_hist.setdefault(regime_key, deque(maxlen=4000))
            hist_reg.append((ts, ev_net_now))
            while hist_reg and hist_reg[0][0] < cutoff:
                hist_reg.popleft()

            dyn_enter_floor = None
            ev_vals = [x[1] for x in hist_reg]
            if len(ev_vals) >= EV_TUNE_MIN_SAMPLES:
                try:
                    raw_thr = float(np.percentile(ev_vals, 80))
                    prev = self._ev_thr_ema.get(regime_key)
                    prev_ts = self._ev_thr_ema_ts.get(regime_key, ts)
                    dt_sec = max(1.0, (ts - prev_ts) / 1000.0)
                    alpha = 1.0 - math.exp(-math.log(2) * dt_sec / 600.0)
                    ema = raw_thr if prev is None else (alpha * raw_thr + (1 - alpha) * prev)
                    ema = float(max(EV_ENTER_FLOOR_MIN, min(EV_ENTER_FLOOR_MAX, ema)))
                    self._ev_thr_ema[regime_key] = ema
                    self._ev_thr_ema_ts[regime_key] = ts
                    dyn_enter_floor = ema
                except Exception:
                    pass

            if dyn_enter_floor is not None:
                decision = dict(decision)
                meta_tmp = dict(decision.get("meta") or {})
                meta_tmp["ev_entry_threshold_dyn"] = dyn_enter_floor
                decision["meta"] = meta_tmp

            # Consensus action
            consensus_action, consensus_score = self._consensus_action(decision, ctx)
            if decision.get("action") == "WAIT" and consensus_action in ("LONG", "SHORT") and decision.get("ev", 0.0) > 0 and decision.get("confidence", 0.0) >= 0.60:
                decision = dict(decision)
                decision["action"] = consensus_action
                decision["reason"] = f"{decision.get('reason', '')} | consensus {consensus_action} score={consensus_score:.2f}"
                decision["details"] = decision.get("details", [])
                decision["details"].append({
                    "_engine": "consensus",
                    "_weight": 0.5,
                    "action": consensus_action,
                    "ev": decision.get("ev", 0.0),
                    "confidence": decision.get("confidence", 0.0),
                    "reason": f"consensus {consensus_score:.2f}",
                    "meta": {"consensus_score": consensus_score, "consensus_used": True},
                })
                decision["size_frac"] = decision.get("size_frac") or decision.get("meta", {}).get("size_fraction") or self.default_size_frac

            # Entry filters + TOP-N / pre-MC status (after consensus)
            decision_meta = dict(decision.get("meta") or {})
            pos_now = self.positions.get(sym) or {}
            try:
                pos_qty_now = float(pos_now.get("quantity", pos_now.get("qty", 0.0)) or 0.0)
            except Exception:
                pos_qty_now = 0.0
            has_pos = pos_qty_now != 0.0
            is_entry = (decision.get("action") in ("LONG", "SHORT")) and (not has_pos)

            # Attach tick-level diagnostics for UI/debug
            try:
                decision_meta.setdefault("tick_ret", ctx.get("tick_ret"))
                decision_meta.setdefault("tick_vol", ctx.get("tick_vol"))
                decision_meta.setdefault("tick_trend", ctx.get("tick_trend"))
                decision_meta.setdefault("tick_dir", ctx.get("tick_dir"))
                decision_meta.setdefault("tick_breakout_active", ctx.get("tick_breakout_active"))
                decision_meta.setdefault("tick_breakout_dir", ctx.get("tick_breakout_dir"))
                decision_meta.setdefault("tick_breakout_score", ctx.get("tick_breakout_score"))
                decision_meta.setdefault("tick_breakout_level", ctx.get("tick_breakout_level"))
                decision_meta.setdefault("tick_window_sec", ctx.get("tick_window_sec"))
                decision_meta.setdefault("tick_samples", ctx.get("tick_samples"))
                decision_meta.setdefault("min_tick_vol", float(MIN_TICK_VOL) if MIN_TICK_VOL > 0 else None)
            except Exception:
                pass

            top_n_status = self._top_n_status(sym)
            pre_mc_status = self._pre_mc_status()

            pre_mc_scaled = False
            if is_entry and pre_mc_status.get("ok") is False:
                try:
                    size_before = float(decision.get("size_frac") or decision_meta.get("size_fraction") or self.default_size_frac)
                except Exception:
                    size_before = float(self.default_size_frac or 0.0)
                size_after = float(max(0.0, size_before * float(PRE_MC_SIZE_SCALE)))
                decision["size_frac"] = size_after
                decision_meta["pre_mc_size_before"] = size_before
                decision_meta["pre_mc_size_after"] = size_after
                decision_meta["pre_mc_scale"] = float(PRE_MC_SIZE_SCALE)
                pre_mc_scaled = True

            if decision:
                # Pre-entry event MC filter (same thresholds as event_mc_exit)
                try:
                    event_entry_filter = str(os.environ.get("EVENT_ENTRY_FILTER", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    event_entry_filter = True
                if is_entry and event_entry_filter:
                    try:
                        evt_meta = self._compute_event_entry_metrics(sym, decision, ctx, float(price))
                        if isinstance(evt_meta, dict):
                            decision_meta.update(evt_meta)
                            decision["meta"] = decision_meta
                    except Exception as e:
                        self._log_err(f"[ERR] event_entry_metrics {sym}: {e}")
                fs = self._min_filter_states(sym, decision, ts)
                if top_n_status.get("ok") is not None:
                    fs["top_n"] = bool(top_n_status.get("ok"))
                else:
                    fs["top_n"] = None
                if pre_mc_status.get("ok") is not None:
                    fs["pre_mc"] = bool(pre_mc_status.get("ok"))
                else:
                    fs["pre_mc"] = None
                decision["filter_states"] = fs
                decision_meta["filter_states"] = fs
                decision_meta["entry_filter_eval"] = bool(is_entry)

            if hybrid_only:
                entry_floor = self._get_hybrid_entry_floor()
            else:
                entry_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
            entry_floor_eff = float(max(entry_floor, MIN_ENTRY_SCORE)) if MIN_ENTRY_SCORE > 0 else float(entry_floor)

            decision_meta["unified_entry_floor"] = float(entry_floor)
            decision_meta["entry_floor_eff"] = entry_floor_eff
            decision_meta["min_entry_score"] = float(MIN_ENTRY_SCORE)
            decision_meta["hybrid_only"] = bool(hybrid_only)

            try:
                decision_meta["spread_pct"] = float(spread_pct_now) if spread_pct_now is not None else None
            except Exception:
                decision_meta["spread_pct"] = None
            try:
                decision_meta["spread_cap"] = float(spread_cap) if spread_cap is not None else None
            except Exception:
                decision_meta["spread_cap"] = None
            try:
                decision_meta["event_cvar_r"] = float(ev_cvar_r) if ev_cvar_r is not None else None
            except Exception:
                decision_meta["event_cvar_r"] = None
            decision_meta["cvar_floor"] = None if hybrid_only else float(cvar_floor_regime)

            liq_score = None
            try:
                ob = self.orderbook.get(sym)
                if ob and ob.get("ready"):
                    liq_score = float(self._liquidity_score(sym))
            except Exception:
                liq_score = None
            decision_meta["liq_score"] = liq_score
            decision_meta["min_liq_score"] = float(MIN_LIQ_SCORE) if MIN_LIQ_SCORE > 0 else None

            est_notional = None
            try:
                lev_val = decision.get("leverage") or decision_meta.get("leverage") or decision_meta.get("lev") or self._dyn_leverage.get(sym) or self.leverage
                lev_val = float(lev_val) if lev_val else float(self.leverage)
                _, est_notional, _ = self._calc_position_size(decision, 1.0, lev_val, symbol=sym, use_cycle_reserve=is_entry)
            except Exception:
                est_notional = None
            decision_meta["est_notional"] = float(est_notional) if est_notional is not None else None
            min_notional_floor = float(MIN_ENTRY_NOTIONAL) if MIN_ENTRY_NOTIONAL > 0 else 0.0
            if MIN_ENTRY_NOTIONAL_PCT > 0:
                try:
                    min_notional_floor = max(min_notional_floor, float(self.balance) * float(MIN_ENTRY_NOTIONAL_PCT))
                except Exception:
                    pass
            decision_meta["min_entry_notional"] = float(min_notional_floor) if min_notional_floor > 0 else None
            decision_meta["min_entry_notional_pct"] = float(MIN_ENTRY_NOTIONAL_PCT) if MIN_ENTRY_NOTIONAL_PCT > 0 else None
            min_exposure_floor = float(MIN_ENTRY_EXPOSURE_PCT) if MIN_ENTRY_EXPOSURE_PCT > 0 else 0.0
            decision_meta["min_entry_exposure_pct"] = float(MIN_ENTRY_EXPOSURE_PCT) if MIN_ENTRY_EXPOSURE_PCT > 0 else None
            if min_exposure_floor > 0:
                try:
                    cap_balance = float(self._exposure_cap_balance())
                except Exception:
                    cap_balance = float(self.balance or 0.0)
                decision_meta["min_entry_exposure_notional"] = float(cap_balance) * float(min_exposure_floor)
            else:
                decision_meta["min_entry_exposure_notional"] = None

            decision_meta["safety_mode"] = bool(self.safety_mode)
            decision_meta["positions_count"] = int(len(self.positions))
            decision_meta["max_positions"] = int(self.max_positions) if self.position_cap_enabled else None
            try:
                decision_meta["total_open_notional"] = float(self._total_open_notional())
            except Exception:
                decision_meta["total_open_notional"] = None
            if self.exposure_cap_enabled:
                try:
                    decision_meta["exposure_cap_limit"] = float(self._exposure_cap_balance()) * float(self.max_notional_frac)
                except Exception:
                    decision_meta["exposure_cap_limit"] = None
            else:
                decision_meta["exposure_cap_limit"] = None

            decision_meta["top_n_active"] = bool(top_n_status.get("active"))
            decision_meta["top_n_rank"] = top_n_status.get("rank")
            decision_meta["top_n_limit"] = top_n_status.get("limit")
            decision_meta["top_n_ok"] = top_n_status.get("ok")

            decision_meta["pre_mc_active"] = bool(pre_mc_status.get("active"))
            decision_meta["pre_mc_ok"] = pre_mc_status.get("ok")
            decision_meta["pre_mc_reason"] = pre_mc_status.get("reason")
            decision_meta["pre_mc_expected_pnl"] = pre_mc_status.get("expected_pnl")
            decision_meta["pre_mc_expected_pnl_source"] = pre_mc_status.get("expected_pnl_source")
            decision_meta["pre_mc_cvar"] = pre_mc_status.get("cvar")
            decision_meta["pre_mc_prob_liq"] = pre_mc_status.get("prob_liq")
            decision_meta["pre_mc_prob_liq_source"] = pre_mc_status.get("prob_liq_source")
            decision_meta["pre_mc_min_expected_pnl"] = float(PRE_MC_MIN_EXPECTED_PNL)
            decision_meta["pre_mc_min_cvar"] = float(PRE_MC_MIN_CVAR)
            decision_meta["pre_mc_max_liq_prob"] = float(PRE_MC_MAX_LIQ_PROB)
            decision_meta["pre_mc_scaled"] = bool(pre_mc_scaled)
            decision_meta["pre_mc_block_on_fail"] = bool(PRE_MC_BLOCK_ON_FAIL)
            decision_meta["pre_mc_blocked"] = False

            if is_entry and decision.get("action") in ("LONG", "SHORT") and decision.get("filter_states"):
                fs = decision.get("filter_states") or {}
                blocked = []
                if fs.get("spread") is False:
                    blocked.append("spread")
                if fs.get("event_cvar") is False:
                    blocked.append("event_cvar")
                try:
                    block_event_mc_exit = str(os.environ.get("ENTRY_BLOCK_EVENT_MC_EXIT", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    block_event_mc_exit = True
                if block_event_mc_exit and fs.get("event_exit") is False:
                    blocked.append("event_mc_exit")
                if fs.get("net_expectancy") is False:
                    blocked.append("net_expectancy")
                if fs.get("dir_gate") is False:
                    blocked.append("direction_conf")
                if fs.get("symbol_quality") is False:
                    blocked.append("symbol_quality")
                if fs.get("lev_floor_lock") is False:
                    blocked.append("lev_floor_lock")
                if fs.get("unified") is False:
                    blocked.append("unified")
                if fs.get("liq") is False:
                    blocked.append("liquidity")
                if fs.get("min_notional") is False:
                    blocked.append("min_notional")
                if fs.get("min_exposure") is False:
                    blocked.append("min_exposure")
                if fs.get("tick_vol") is False:
                    blocked.append("tick_vol")
                if fs.get("top_n") is False:
                    blocked.append("top_n")
                if fs.get("cap") is False:
                    blocked.append("cap")
                if PRE_MC_BLOCK_ON_FAIL and fs.get("pre_mc") is False:
                    blocked.append("pre_mc")
                decision_meta["pre_mc_blocked"] = bool(PRE_MC_BLOCK_ON_FAIL and fs.get("pre_mc") is False)
                if blocked:
                    decision["action"] = "WAIT"
                    reason = decision.get("reason", "")
                    suffix = f"filter_block:{','.join(blocked)}"
                    decision["reason"] = f"{reason} | {suffix}" if reason else suffix
                    decision_meta["entry_blocked_filters"] = blocked
                else:
                    decision_meta["entry_blocked_filters"] = []
            elif decision:
                # Not an entry attempt; keep explicit marker for UI consistency
                decision_meta.setdefault("entry_blocked_filters", [])

            decision["meta"] = decision_meta

            # HOLD/EXIT using event MC
            exited_by_event = False
            pos = self.positions.get(sym)
            pos_managed = self._is_managed_position(pos)
            try:
                hold_only = str(os.environ.get("HOLD_EVAL_ONLY", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                hold_only = False
            if pos and decision and ctx.get("mu_base") is not None and ctx.get("sigma", 0.0) > 0:
                if pos_managed and not hold_only:
                    exited_by_event = self._evaluate_event_exit(sym, pos, decision, ctx, ts, price)

            # Hold-vs-exit using entry-eval t* (align hold/exit with entry logic)
            if (not exited_by_event) and sym in self.positions and decision:
                pos = self.positions.get(sym) or {}
                eval_only = not pos_managed
                if self._check_hold_vs_exit(sym, pos, decision, ctx, ts, price, eval_only=eval_only):
                    exited_by_event = True

            # Policy-based event MC exit
            if (not exited_by_event) and pos_managed and sym in self.positions and decision and (not hold_only):
                pos = self.positions.get(sym) or {}
                meta = decision.get("meta") or {}
                age_sec = (ts - int(pos.get("entry_time", ts))) / 1000.0
                exit_policy_dyn, exit_diag = self._build_dynamic_exit_policy(sym, pos, decision, ctx=ctx)
                self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_diag)
                decision["meta"] = meta
                try:
                    min_hold_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
                except Exception:
                    min_hold_sec = float(POSITION_HOLD_MIN_SEC)
                if min_hold_sec > 0 and age_sec < float(min_hold_sec):
                    do_exit, reason = False, "min_hold"
                else:
                    try:
                        shared_gate_on = str(os.environ.get("EVENT_EXIT_POLICY_USE_SHARED_GATE", "1")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        shared_gate_on = True
                    if shared_gate_on:
                        event_score_pol = self._safe_float(meta.get("event_exit_score"), None)
                        event_cvar_pol = self._safe_float(meta.get("event_cvar_pct"), None)
                        event_psl_pol = self._safe_float(meta.get("event_p_sl"), None)
                        event_ptp_pol = self._safe_float(meta.get("event_p_tp"), None)
                        metrics_available_pol = any(
                            v is not None
                            for v in (event_score_pol, event_cvar_pol, event_psl_pol, event_ptp_pol)
                        )
                        if metrics_available_pol:
                            strict_mode_pol = self._event_exit_strict_mode_enabled()
                            core_pol = self._evaluate_event_exit_core(
                                event_score=float(event_score_pol or 0.0),
                                event_cvar_pct=float(event_cvar_pol or 0.0),
                                event_p_sl=float(event_psl_pol or 0.0),
                                event_p_tp=float(event_ptp_pol or 0.0),
                                metrics_available=True,
                                policy=exit_policy_dyn,
                                exit_diag=exit_diag,
                                strict_mode=bool(strict_mode_pol),
                                shock_threshold_env="EVENT_EXIT_SHOCK_MODE_THRESHOLD",
                                shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
                            )
                            gate_pol = self._event_exit_gate_decision(
                                pos=pos,
                                ts_ms=int(ts),
                                event_exit_hit=bool(core_pol.get("event_exit_hit")),
                                event_mode=str(core_pol.get("mode") or "normal"),
                                shock_score=float(core_pol.get("shock_score") or 0.0),
                                p_sl_evt=float(event_psl_pol or 0.0),
                                cvar_pct_evt=float(event_cvar_pol or 0.0),
                                p_tp_evt=float(event_ptp_pol or 0.0),
                                ptp_thr=float(core_pol.get("threshold_ptp") or 0.0),
                                meta=meta,
                                confirm_key="event_exit_policy",
                            )
                            meta["event_eval_source"] = "exit_policy_shared"
                            meta["event_exit_mode"] = str(gate_pol.get("event_mode") or core_pol.get("mode") or "normal")
                            meta["event_exit_hit"] = bool(core_pol.get("event_exit_hit"))
                            meta["event_exit_guard"] = str(gate_pol.get("guard_reason") or "")
                            meta["event_exit_confirm_mode"] = str(gate_pol.get("confirm_mode") or "idle")
                            meta["event_exit_confirm_required"] = int(gate_pol.get("confirm_required") or 1)
                            meta["event_exit_confirm_count"] = int(gate_pol.get("confirm_count") or 0)
                            meta["event_exit_confirmed"] = bool(gate_pol.get("confirm_ok"))
                            meta["event_exit_guard_progress"] = gate_pol.get("guard_progress")
                            meta["event_exit_guard_required"] = gate_pol.get("guard_required")
                            meta["event_exit_guard_hold_target_sec"] = gate_pol.get("guard_hold_target_sec")
                            meta["event_exit_guard_age_sec"] = gate_pol.get("guard_age_sec")
                            meta["event_exit_guard_remaining_sec"] = gate_pol.get("guard_remaining_sec")
                            decision["meta"] = meta
                            do_exit = bool(gate_pol.get("allow_exit"))
                            if do_exit:
                                reason = f"event_mc_exit_{meta.get('event_exit_mode') or 'normal'}"
                            else:
                                reason = f"gate:{str(gate_pol.get('guard_reason') or 'hold')}"
                        else:
                            do_exit, reason = should_exit_position(pos, meta, age_sec=age_sec, policy=exit_policy_dyn)
                    else:
                        do_exit, reason = should_exit_position(pos, meta, age_sec=age_sec, policy=exit_policy_dyn)
                if do_exit:
                    try:
                        self._capture_position_observability(pos, decision=decision, ctx=ctx)
                    except Exception:
                        pass
                    reason_txt = str(reason or "")
                    if reason_txt.lower().startswith("event_mc_exit"):
                        close_reason = reason_txt
                    else:
                        close_reason = f"MC_EXIT:{reason_txt}"
                    self._close_position(sym, float(price), close_reason)
                    exited_by_event = True
                else:
                    exited_by_event = self._check_ema_ev_exit(sym, decision, regime, price, ts)

            if pos_managed:
                self._maybe_exit_position(sym, float(price), decision, ts, allow_extra_exits=True, ctx=ctx)

            # Keep desired leverage separately; do not overwrite live position leverage metadata.
            if sym in self.positions and pos_managed:
                self.positions[sym]["target_leverage"] = float(dyn_leverage)
            self._dyn_leverage[sym] = dyn_leverage

            if not exited_by_event:
                if pos_managed and REBALANCE_ENABLED and sym in self.positions:
                    if decision.get("action") in ("LONG", "SHORT") or bool(getattr(self, "_force_rebalance_cycle", False)):
                        self._rebalance_position(sym, float(price), decision, leverage_override=dyn_leverage)

                # EV drop exit
                if pos_managed and sym in self.positions:
                    exited_by_event = self._check_ev_drop_exit(sym, decision, regime, price, ts)

                if decision.get("action") in ("LONG", "SHORT") and sym not in self.positions:
                    try:
                        dyn_entry = self._dynamic_leverage_risk(decision, ctx)
                    except Exception as e:
                        dyn_entry = None
                        self._log_err(f"[LEVERAGE] dynamic_risk_pre_entry failed: {sym} err={e}")
                    if dyn_entry is not None and math.isfinite(float(dyn_entry)):
                        dyn_leverage = float(dyn_entry)
                        ctx["leverage"] = float(dyn_leverage)
                        decision["leverage"] = float(dyn_leverage)
                        meta_tmp = dict(decision.get("meta") or {})
                        meta_tmp["lev"] = float(dyn_leverage)
                        lev_src_prev = str(meta_tmp.get("lev_source") or "").strip()
                        if (not lev_src_prev) or lev_src_prev.startswith("decision"):
                            meta_tmp["lev_source"] = "dynamic_risk_pre_entry"
                        decision["meta"] = meta_tmp
                        self._dyn_leverage[sym] = float(dyn_leverage)
                    permit, deny_reason = self._entry_permit(sym, decision, ts)
                    if permit:
                        self._enter_position(sym, decision["action"], float(price), decision, ts, ctx=ctx, leverage_override=dyn_leverage)
                    else:
                        if log_this_cycle:
                            self._log(f"[{sym}] skip entry (permit: {deny_reason}) ev={decision.get('ev', 0):.4f} win={decision.get('confidence', 0):.2f}")

            if log_this_cycle and decision:
                meta = decision.get("meta") or {}
                self._log(
                    f"[DECISION] {sym} action={decision.get('action')} "
                    f"ev={decision.get('ev', 0.0):.4f} "
                    f"win={decision.get('confidence', 0.0):.2f} "
                    f"size={decision.get('size_frac', meta.get('size_fraction', 0.0)):.3f} "
                    f"reason={decision.get('reason', '')}"
                )

            if decision:
                self._last_decisions[sym] = decision
            return self._row(sym, float(price), ts, decision, candles, ctx=ctx)

        except Exception as e:
            import traceback
            err_text = f"{e} {traceback.format_exc()}"
            self._log_err(f"[ERR] _apply_decision {sym}: {err_text}")
            self._note_runtime_error(f"apply_decision:{sym}", err_text)
            return self._row(sym, ctx.get("price"), ts, None, ctx.get("candles", 0), ctx=ctx)

    def _evaluate_event_exit(self, sym: str, pos: dict, decision: dict, ctx: dict, ts: int, price: float) -> bool:
        """Event-based MC exit evaluation. Returns True if exited."""
        try:
            min_hold_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
        except Exception:
            min_hold_sec = float(POSITION_HOLD_MIN_SEC)
        try:
            entry_ts = int(pos.get("entry_time") or 0)
        except Exception:
            entry_ts = 0
        if entry_ts and min_hold_sec > 0:
            age_sec = (ts - entry_ts) / 1000.0
            if age_sec < float(min_hold_sec):
                return False
        meta = decision.get("meta") or {}
        if not meta:
            for d in decision.get("details", []):
                if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                    meta = d.get("meta") or {}
                    break
        exit_policy_dyn, exit_diag = self._build_dynamic_exit_policy(sym, pos, decision, ctx=ctx)
        self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_diag)
        decision["meta"] = meta
        regime_now = str(ctx.get("regime", "chop"))
        mu_evt_src = "ctx.mu_base"
        mu_evt_base = ctx.get("mu_base")
        if meta.get("mu_adjusted") is not None:
            mu_evt_src = "meta.mu_adjusted"
            mu_evt_base = meta.get("mu_adjusted")
        elif meta.get("mu_alpha") is not None:
            mu_evt_src = "meta.mu_alpha"
            mu_evt_base = meta.get("mu_alpha")
        elif ctx.get("mu_alpha") is not None:
            mu_evt_src = "ctx.mu_alpha"
            mu_evt_base = ctx.get("mu_alpha")
        sigma_evt_base = (
            meta.get("sigma_sim")
            or meta.get("sigma_annual")
            or ctx.get("sigma_sim")
            or ctx.get("sigma")
            or 0.0
        )
        try:
            sigma_evt_base = max(float(sigma_evt_base), 1e-6)
        except Exception:
            sigma_evt_base = 1e-6
        if mu_evt_src == "meta.mu_adjusted":
            try:
                mu_evt = float(mu_evt_base or 0.0)
            except Exception:
                mu_evt = 0.0
            sigma_evt = float(sigma_evt_base)
        else:
            mu_evt, sigma_evt = adjust_mu_sigma(
                float(mu_evt_base or 0.0),
                float(sigma_evt_base),
                regime_now,
            )
        meta["event_mu_src"] = str(mu_evt_src)
        meta["event_mu_used"] = float(mu_evt)
        meta["event_sigma_used"] = float(sigma_evt)
        try:
            bar_sec = float(ctx.get("bar_seconds", 60.0) or 60.0)
        except Exception:
            bar_sec = 60.0
        try:
            step_sec = float(os.environ.get("EVENT_MC_STEP_SEC", bar_sec) or bar_sec)
        except Exception:
            step_sec = bar_sec
        if step_sec <= 0:
            step_sec = bar_sec if bar_sec > 0 else 60.0
        max_horizon_sec = None
        try:
            horizon_seq = meta.get("horizon_seq")
            if isinstance(horizon_seq, (list, tuple)) and horizon_seq:
                max_horizon_sec = max(float(h) for h in horizon_seq if h is not None)
        except Exception:
            max_horizon_sec = None
        try:
            env_max_h = float(os.environ.get("EVENT_MC_MAX_HORIZON_SEC", 0) or 0)
        except Exception:
            env_max_h = 0.0
        if env_max_h > 0:
            max_horizon_sec = env_max_h
        # If available, align event MC horizon to optimal hold (t*)
        try:
            use_tstar = str(os.environ.get("EVENT_MC_USE_TSTAR", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_tstar = True
        hold_target_sec = None
        if use_tstar:
            try:
                hold_target_sec = float(
                    pos.get("opt_hold_curr_sec")
                    or pos.get("opt_hold_sec")
                    or pos.get("opt_hold_entry_sec")
                    or 0.0
                )
            except Exception:
                hold_target_sec = 0.0
            if not hold_target_sec:
                try:
                    hold_target_sec = float(
                        meta.get("opt_hold_sec")
                        or meta.get("unified_t_star")
                        or meta.get("policy_horizon_eff_sec")
                        or meta.get("best_h")
                        or 0.0
                    )
                except Exception:
                    hold_target_sec = 0.0
            if hold_target_sec and entry_ts:
                try:
                    age_sec = (ts - entry_ts) / 1000.0
                except Exception:
                    age_sec = 0.0
                remaining_sec = max(1.0, float(hold_target_sec) - float(age_sec))
                max_horizon_sec = float(remaining_sec)
                meta["event_hold_target_sec"] = float(hold_target_sec)
                meta["event_hold_remaining_sec"] = float(remaining_sec)
        if max_horizon_sec and step_sec > 0:
            max_steps_evt = max(1, int(round(float(max_horizon_sec) / float(step_sec))))
        else:
            try:
                max_steps_evt = int(os.environ.get("EVENT_MC_MAX_STEPS", 600) or 600)
            except Exception:
                max_steps_evt = 600
        dt_evt = float(step_sec) / 31536000.0
        seed_evt = int(time.time()) ^ hash(sym)
        entry = float(pos.get("entry_price", price))
        price_now = float(price)

        params_meta = meta.get("params") or {}
        tp_pct = decision.get("mc_tp") or meta.get("mc_tp") or params_meta.get("profit_target") or 0.001
        sl_pct = decision.get("mc_sl") or meta.get("mc_sl") or (tp_pct * 0.8)
        tp_pct = float(max(tp_pct, 1e-6))
        sl_pct = float(max(sl_pct, 1e-6))

        tp_rem = max((entry * (1 + tp_pct) / price_now) - 1.0, 1e-6)
        sl_rem = max(1.0 - (entry * (1 - sl_pct) / price_now), 1e-6)

        m_evt = mc_first_passage_tp_sl_jax(
            s0=price_now,
            tp_pct=tp_rem,
            sl_pct=sl_rem,
            mu=mu_evt,
            sigma=sigma_evt,
            dt=dt_evt,
            max_steps=int(max_steps_evt),
            n_paths=int(decision.get("n_paths", meta.get("params", {}).get("n_paths", 2048))),
            seed=seed_evt,
            dist=ctx.get("tail_model", "student_t"),
            df=ctx.get("tail_df", 6.0),
            boot_rets=ctx.get("bootstrap_returns"),
        )

        if m_evt:
            ev_r_evt = float(m_evt.get("event_ev_r", 0.0) or 0.0)
            cvar_r_evt = float(m_evt.get("event_cvar_r", 0.0) or 0.0)
            p_sl_evt = float(m_evt.get("event_p_sl", 0.0) or 0.0)
            p_tp_evt = float(m_evt.get("event_p_tp", 0.0) or 0.0)
            ev_pct_evt = ev_r_evt * sl_rem
            cvar_pct_evt = cvar_r_evt * sl_rem
            t_med_evt = float(m_evt.get("event_t_median", 0.0) or 0.0)
            t_med_evt_sec = float(t_med_evt) * float(step_sec) if t_med_evt > 0 else 0.0
            tau_evt = float(max(1.0, t_med_evt_sec if t_med_evt_sec > 0 else 1.0))
            lambda_evt = float(getattr(mc_config, "unified_risk_lambda", 1.0))
            rho_evt = float(getattr(mc_config, "unified_rho", 0.0))
            event_score = float(ev_pct_evt - lambda_evt * abs(cvar_pct_evt) - rho_evt * tau_evt)
        else:
            ev_pct_evt, cvar_pct_evt, p_sl_evt, p_tp_evt = 0.0, 0.0, 0.0, 0.0
            event_score = 0.0

        strict_mode = self._event_exit_strict_mode_enabled()
        core_eval = self._evaluate_event_exit_core(
            event_score=float(event_score),
            event_cvar_pct=float(cvar_pct_evt),
            event_p_sl=float(p_sl_evt),
            event_p_tp=float(p_tp_evt),
            metrics_available=bool(m_evt),
            policy=exit_policy_dyn,
            exit_diag=exit_diag,
            strict_mode=bool(strict_mode),
            shock_threshold_env="EVENT_EXIT_SHOCK_MODE_THRESHOLD",
            shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
        )
        shock_score_now = float(core_eval.get("shock_score") or 0.0)
        noise_mode_now = bool(core_eval.get("noise_mode"))
        event_mode = str(core_eval.get("mode") or "normal")
        score_thr = float(core_eval.get("threshold_score"))
        cvar_thr = float(core_eval.get("threshold_cvar"))
        psl_thr = float(core_eval.get("threshold_psl"))
        ptp_thr = float(core_eval.get("threshold_ptp"))
        score_hit = bool(core_eval.get("hit_score"))
        cvar_hit = bool(core_eval.get("hit_cvar"))
        psl_hit = bool(core_eval.get("hit_psl"))
        ptp_hit = bool(core_eval.get("hit_ptp"))
        event_exit_hit = bool(core_eval.get("event_exit_hit"))
        meta["event_exit_hit"] = bool(event_exit_hit)
        meta["event_exit_hit_score"] = bool(score_hit)
        meta["event_exit_hit_cvar"] = bool(cvar_hit)
        meta["event_exit_hit_psl"] = bool(psl_hit)
        meta["event_exit_hit_ptp"] = bool(ptp_hit)
        meta["event_eval_source"] = "exit_runtime"
        meta["event_exit_strict_mode"] = bool(core_eval.get("strict_mode"))
        meta["event_exit_mode"] = str(event_mode)
        meta["event_exit_score"] = float(event_score)
        meta["event_cvar_pct"] = float(cvar_pct_evt)
        meta["event_p_sl"] = float(p_sl_evt)
        meta["event_p_tp"] = float(p_tp_evt)
        meta["event_exit_min_score"] = float(score_thr)
        meta["event_exit_max_cvar"] = float(cvar_thr)
        meta["event_exit_max_p_sl"] = float(psl_thr)
        meta["event_exit_min_p_tp"] = float(ptp_thr)
        meta["event_exit_threshold_score"] = float(score_thr)
        meta["event_exit_threshold_psl"] = float(psl_thr)
        meta["event_exit_threshold_cvar"] = float(cvar_thr)
        meta["event_exit_threshold_ptp"] = float(ptp_thr)
        meta["event_eval_ts_ms"] = int(now_ms())
        gate_eval = self._event_exit_gate_decision(
            pos=pos,
            ts_ms=int(ts),
            event_exit_hit=bool(event_exit_hit),
            event_mode=str(event_mode),
            shock_score=float(shock_score_now),
            p_sl_evt=float(p_sl_evt),
            cvar_pct_evt=float(cvar_pct_evt),
            p_tp_evt=float(p_tp_evt),
            ptp_thr=float(ptp_thr),
            meta=meta,
            confirm_key="event_exit",
        )
        meta["event_exit_confirm_mode"] = str(gate_eval.get("confirm_mode") or "idle")
        meta["event_exit_confirm_required"] = int(gate_eval.get("confirm_required") or 1)
        meta["event_exit_confirm_count"] = int(gate_eval.get("confirm_count") or 0)
        meta["event_exit_confirmed"] = bool(gate_eval.get("confirm_ok"))
        meta["event_exit_shock_score"] = float(gate_eval.get("shock_score") or 0.0)
        meta["event_exit_noise_mode"] = bool(noise_mode_now)
        meta["event_exit_mode"] = str(gate_eval.get("event_mode") or event_mode)
        meta["event_exit_fast_only_shock"] = bool(gate_eval.get("fast_only_shock"))
        meta["event_exit_severe_adverse"] = bool(gate_eval.get("severe_adverse"))
        meta["event_exit_severe_adverse_ptp_low"] = bool(gate_eval.get("severe_adverse_ptp_low"))
        guard_reason = str(gate_eval.get("guard_reason") or "")
        meta["event_exit_guard"] = guard_reason if guard_reason and guard_reason != "idle" else None
        meta["event_exit_guard_progress"] = gate_eval.get("guard_progress")
        meta["event_exit_guard_required"] = gate_eval.get("guard_required")
        meta["event_exit_guard_hold_target_sec"] = gate_eval.get("guard_hold_target_sec")
        meta["event_exit_guard_age_sec"] = gate_eval.get("guard_age_sec")
        meta["event_exit_guard_remaining_sec"] = gate_eval.get("guard_remaining_sec")
        decision["meta"] = meta
        if not bool(gate_eval.get("allow_exit")):
            return False

        allow_flip = str(os.environ.get("EVENT_MC_ALLOW_FLIP", "1")).strip().lower() in ("1", "true", "yes", "on")
        if allow_flip:
            def _maybe_float(val):
                try:
                    if val is None:
                        return None
                    return float(val)
                except Exception:
                    return None

            side = str(pos.get("side") or decision.get("action") or "").upper()
            if side in ("LONG", "SHORT"):
                opp_side = "SHORT" if side == "LONG" else "LONG"
                score_long = _maybe_float(meta.get("hybrid_score_long") or meta.get("unified_score_long") or meta.get("policy_ev_score_long"))
                score_short = _maybe_float(meta.get("hybrid_score_short") or meta.get("unified_score_short") or meta.get("policy_ev_score_short"))
                cur_score = score_long if side == "LONG" else score_short
                opp_score = score_short if side == "LONG" else score_long
                try:
                    flip_min_score = float(os.environ.get("EVENT_MC_FLIP_MIN_SCORE", 0.0) or 0.0)
                except Exception:
                    flip_min_score = 0.0
                try:
                    flip_margin = float(os.environ.get("EVENT_MC_FLIP_MARGIN", 0.0) or 0.0)
                except Exception:
                    flip_margin = 0.0
                if opp_score is not None and opp_score >= flip_min_score and (cur_score is None or opp_score >= (cur_score + flip_margin)):
                    override_raw = os.environ.get("EVENT_MC_FLIP_OVERRIDE_FILTERS", "top_n,cap")
                    override_filters = self._normalize_filter_overrides(override_raw)
                    meta_for_flip = dict(meta or {})
                    meta_for_flip["entry_override_filters"] = sorted(override_filters) if override_filters else []
                    meta_for_flip.setdefault("hybrid_score", opp_score)
                    meta_for_flip.setdefault("unified_score", opp_score)
                    meta_for_flip["event_mc_flip"] = True
                    lev_val = float(decision.get("leverage") or decision.get("optimal_leverage") or pos.get("leverage") or self.leverage or 1.0)
                    flip_decision = {
                        "action": opp_side,
                        "ev": float(opp_score),
                        "confidence": float(decision.get("confidence", 0.0) or 0.0),
                        "unified_score": float(opp_score),
                        "hybrid_score": float(opp_score),
                        "leverage": lev_val,
                        "reason": "event_mc_exit_flip",
                        "meta": meta_for_flip,
                    }
                    self._log(
                        f"[{sym}] EXIT+FLIP by MC "
                        f"(Score={event_score*100:.4f}%, EV%={ev_pct_evt*100:.2f}%, CVaR%={cvar_pct_evt*100:.2f}%, "
                        f"P_SL={p_sl_evt:.2f}, P_TP={p_tp_evt:.2f}, {side}->{opp_side}, score={opp_score:.4f})"
                    )
                    self._capture_position_observability(pos, decision=decision, ctx=ctx)
                    self._close_position(sym, price_now, "event_mc_exit_flip", exit_kind="RISK")
                    permit, deny = self._entry_permit(sym, flip_decision, ts)
                    if permit:
                        self._enter_position(sym, opp_side, price_now, flip_decision, ts, ctx=ctx, leverage_override=lev_val)
                    else:
                        self._log(f"[{sym}] event_mc_exit flip blocked: {deny}")
                    return True
        self._log(
            f"[{sym}] EXIT by MC "
            f"(Score={event_score*100:.4f}%, "
            f"EV%={ev_pct_evt*100:.2f}%, "
            f"CVaR%={cvar_pct_evt*100:.2f}%, "
            f"P_SL={p_sl_evt:.2f}, "
            f"P_TP={p_tp_evt:.2f}, mode={event_mode})"
        )
        self._capture_position_observability(pos, decision=decision, ctx=ctx)
        self._close_position(sym, price_now, f"event_mc_exit_{event_mode}")
        return True

    def _compute_event_entry_metrics(self, sym: str, decision: dict, ctx: dict, price: float) -> dict:
        """Pre-entry event MC metrics using the same thresholds as event_mc_exit."""
        meta = decision.get("meta") or {}
        if not meta:
            for d in decision.get("details", []):
                if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                    meta = d.get("meta") or {}
                    break
        exit_policy_dyn, exit_diag = self._build_dynamic_exit_policy(sym, None, decision, ctx=ctx)
        self._attach_dynamic_exit_meta(meta, exit_policy_dyn, exit_diag)
        decision["meta"] = meta
        regime_now = str(ctx.get("regime", "chop"))
        mu_evt_src = "ctx.mu_base"
        mu_evt_base = ctx.get("mu_base")
        if meta.get("mu_adjusted") is not None:
            mu_evt_src = "meta.mu_adjusted"
            mu_evt_base = meta.get("mu_adjusted")
        elif meta.get("mu_alpha") is not None:
            mu_evt_src = "meta.mu_alpha"
            mu_evt_base = meta.get("mu_alpha")
        elif ctx.get("mu_alpha") is not None:
            mu_evt_src = "ctx.mu_alpha"
            mu_evt_base = ctx.get("mu_alpha")
        sigma_evt_base = (
            meta.get("sigma_sim")
            or meta.get("sigma_annual")
            or ctx.get("sigma_sim")
            or ctx.get("sigma")
            or 0.0
        )
        try:
            sigma_evt_base = max(float(sigma_evt_base), 1e-6)
        except Exception:
            sigma_evt_base = 1e-6
        if mu_evt_src == "meta.mu_adjusted":
            try:
                mu_evt = float(mu_evt_base or 0.0)
            except Exception:
                mu_evt = 0.0
            sigma_evt = float(sigma_evt_base)
        else:
            mu_evt, sigma_evt = adjust_mu_sigma(
                float(mu_evt_base or 0.0),
                float(sigma_evt_base),
                regime_now,
            )
        meta["event_mu_src"] = str(mu_evt_src)
        meta["event_mu_used"] = float(mu_evt)
        meta["event_sigma_used"] = float(sigma_evt)
        try:
            bar_sec = float(ctx.get("bar_seconds", 60.0) or 60.0)
        except Exception:
            bar_sec = 60.0
        try:
            step_sec = float(os.environ.get("EVENT_MC_STEP_SEC", bar_sec) or bar_sec)
        except Exception:
            step_sec = bar_sec
        if step_sec <= 0:
            step_sec = bar_sec if bar_sec > 0 else 60.0
        max_horizon_sec = None
        try:
            horizon_seq = meta.get("horizon_seq")
            if isinstance(horizon_seq, (list, tuple)) and horizon_seq:
                max_horizon_sec = max(float(h) for h in horizon_seq if h is not None)
        except Exception:
            max_horizon_sec = None
        try:
            env_max_h = float(os.environ.get("EVENT_MC_MAX_HORIZON_SEC", 0) or 0)
        except Exception:
            env_max_h = 0.0
        if env_max_h > 0:
            max_horizon_sec = env_max_h
        try:
            use_tstar = str(os.environ.get("EVENT_MC_USE_TSTAR", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_tstar = True
        if use_tstar:
            try:
                tstar = float(
                    meta.get("opt_hold_sec")
                    or meta.get("unified_t_star")
                    or meta.get("policy_horizon_eff_sec")
                    or meta.get("best_h")
                    or 0.0
                )
            except Exception:
                tstar = 0.0
            if tstar and tstar > 0:
                max_horizon_sec = float(tstar)
        if max_horizon_sec and step_sec > 0:
            max_steps_evt = max(1, int(round(float(max_horizon_sec) / float(step_sec))))
        else:
            try:
                max_steps_evt = int(os.environ.get("EVENT_MC_MAX_STEPS", 600) or 600)
            except Exception:
                max_steps_evt = 600
        dt_evt = float(step_sec) / 31536000.0
        seed_evt = int(time.time()) ^ hash(sym)
        params_meta = meta.get("params") or {}
        tp_pct = decision.get("mc_tp") or meta.get("mc_tp") or params_meta.get("profit_target") or 0.001
        sl_pct = decision.get("mc_sl") or meta.get("mc_sl") or (tp_pct * 0.8)
        tp_pct = float(max(tp_pct, 1e-6))
        sl_pct = float(max(sl_pct, 1e-6))

        # Entry at current price => remaining TP/SL equals configured TP/SL.
        tp_rem = float(tp_pct)
        sl_rem = float(sl_pct)

        ev_r_evt = meta.get("event_ev_r")
        cvar_r_evt = meta.get("event_cvar_r")
        p_sl_evt = meta.get("event_p_sl")
        p_tp_evt = meta.get("event_p_tp")
        t_med_evt = meta.get("event_t_median")
        metrics_available = True

        if ev_r_evt is None or cvar_r_evt is None or p_sl_evt is None or p_tp_evt is None:
            m_evt = mc_first_passage_tp_sl_jax(
                s0=float(price),
                tp_pct=tp_rem,
                sl_pct=sl_rem,
                mu=mu_evt,
                sigma=sigma_evt,
                dt=dt_evt,
                max_steps=int(max_steps_evt),
                n_paths=int(decision.get("n_paths", meta.get("params", {}).get("n_paths", 2048))),
                seed=seed_evt,
                dist=ctx.get("tail_model", "student_t"),
                df=ctx.get("tail_df", 6.0),
                boot_rets=ctx.get("bootstrap_returns"),
            )
            if m_evt:
                ev_r_evt = float(m_evt.get("event_ev_r", 0.0) or 0.0)
                cvar_r_evt = float(m_evt.get("event_cvar_r", 0.0) or 0.0)
                p_sl_evt = float(m_evt.get("event_p_sl", 0.0) or 0.0)
                p_tp_evt = float(m_evt.get("event_p_tp", 0.0) or 0.0)
                t_med_evt = float(m_evt.get("event_t_median", 0.0) or 0.0)
                t_med_evt = float(t_med_evt) * float(step_sec) if t_med_evt > 0 else 0.0
            else:
                ev_r_evt, cvar_r_evt, p_sl_evt, p_tp_evt, t_med_evt = 0.0, 0.0, 0.0, 0.0, 0.0
                metrics_available = False
        else:
            ev_r_evt = float(ev_r_evt or 0.0)
            cvar_r_evt = float(cvar_r_evt or 0.0)
            p_sl_evt = float(p_sl_evt or 0.0)
            p_tp_evt = float(p_tp_evt or 0.0)
            t_med_evt = float(t_med_evt or 0.0)

        ev_pct_evt = float(ev_r_evt) * float(sl_rem)
        cvar_pct_evt = float(cvar_r_evt) * float(sl_rem)
        tau_evt = float(max(1.0, float(t_med_evt) if t_med_evt else 1.0))
        lambda_evt = float(getattr(mc_config, "unified_risk_lambda", 1.0))
        rho_evt = float(getattr(mc_config, "unified_rho", 0.0))
        event_score = float(ev_pct_evt - lambda_evt * abs(cvar_pct_evt) - rho_evt * tau_evt)

        strict_mode = self._event_exit_strict_mode_enabled()
        core_eval = self._evaluate_event_exit_core(
            event_score=float(event_score),
            event_cvar_pct=float(cvar_pct_evt),
            event_p_sl=float(p_sl_evt),
            event_p_tp=float(p_tp_evt),
            metrics_available=bool(metrics_available),
            policy=exit_policy_dyn,
            exit_diag=exit_diag,
            strict_mode=bool(strict_mode),
            shock_threshold_env="EVENT_EXIT_SHOCK_MODE_THRESHOLD",
            shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
        )
        ts_eval_ms = int(now_ms())
        try:
            hold_target_pre = float(
                meta.get("opt_hold_sec")
                or meta.get("unified_t_star")
                or meta.get("policy_horizon_eff_sec")
                or meta.get("best_h")
                or 0.0
            )
        except Exception:
            hold_target_pre = 0.0
        shadow_pos = {
            "entry_time": int(ts_eval_ms),
            "opt_hold_curr_sec": float(hold_target_pre) if hold_target_pre > 0 else None,
            "opt_hold_entry_sec": float(hold_target_pre) if hold_target_pre > 0 else None,
            "policy_min_hold_eff_sec": float(hold_target_pre) if hold_target_pre > 0 else None,
        }
        precheck_gate = self._event_exit_gate_decision(
            pos=shadow_pos,
            ts_ms=int(ts_eval_ms),
            event_exit_hit=bool(core_eval.get("event_exit_hit")),
            event_mode=str(core_eval.get("mode") or "normal"),
            shock_score=float(core_eval.get("shock_score") or 0.0),
            p_sl_evt=float(p_sl_evt),
            cvar_pct_evt=float(cvar_pct_evt),
            p_tp_evt=float(p_tp_evt),
            ptp_thr=float(core_eval.get("threshold_ptp")),
            meta=meta,
            confirm_key="event_exit_precheck",
        )
        event_exit_ok = not bool(precheck_gate.get("allow_exit"))
        pre_guard_reason = str(precheck_gate.get("guard_reason") or "")

        return {
            "event_ev_r": float(ev_r_evt),
            "event_cvar_r": float(cvar_r_evt),
            "event_p_tp": float(p_tp_evt),
            "event_p_sl": float(p_sl_evt),
            "event_ev_pct": float(ev_pct_evt),
            "event_cvar_pct": float(cvar_pct_evt),
            "event_exit_score": float(event_score),
            "event_exit_min_score": float(core_eval.get("threshold_score")),
            "event_exit_max_cvar": float(core_eval.get("threshold_cvar")),
            "event_exit_max_p_sl": float(core_eval.get("threshold_psl")),
            "event_exit_ok": bool(event_exit_ok),
            "event_exit_hit": bool(core_eval.get("event_exit_hit")),
            "event_exit_hit_score": bool(core_eval.get("hit_score")),
            "event_exit_hit_cvar": bool(core_eval.get("hit_cvar")),
            "event_exit_hit_psl": bool(core_eval.get("hit_psl")),
            "event_exit_hit_ptp": bool(core_eval.get("hit_ptp")),
            "event_exit_mode": str(core_eval.get("mode") or "normal"),
            "event_exit_threshold_score": float(core_eval.get("threshold_score")),
            "event_exit_threshold_psl": float(core_eval.get("threshold_psl")),
            "event_exit_threshold_cvar": float(core_eval.get("threshold_cvar")),
            "event_exit_threshold_ptp": float(core_eval.get("threshold_ptp")),
            "event_exit_strict_mode": bool(core_eval.get("strict_mode")),
            "event_eval_source": "entry_precheck",
            "event_eval_ts_ms": int(ts_eval_ms),
            "event_precheck_allow_exit_now": bool(precheck_gate.get("allow_exit")),
            "event_precheck_guard_reason": pre_guard_reason if pre_guard_reason and pre_guard_reason != "idle" else None,
            "event_precheck_confirm_mode": str(precheck_gate.get("confirm_mode") or "idle"),
            "event_precheck_confirm_required": int(precheck_gate.get("confirm_required") or 1),
            "event_precheck_confirm_count": int(precheck_gate.get("confirm_count") or 0),
            "event_precheck_confirmed": bool(precheck_gate.get("confirm_ok")),
            "event_precheck_shock_score": float(precheck_gate.get("shock_score") or 0.0),
            "event_precheck_severe_adverse": bool(precheck_gate.get("severe_adverse")),
            "event_precheck_severe_adverse_ptp_low": bool(precheck_gate.get("severe_adverse_ptp_low")),
            "event_precheck_fast_only_shock": bool(precheck_gate.get("fast_only_shock")),
            "event_precheck_guard_progress": precheck_gate.get("guard_progress"),
            "event_precheck_guard_required": precheck_gate.get("guard_required"),
            "event_precheck_guard_hold_target_sec": precheck_gate.get("guard_hold_target_sec"),
            "event_precheck_guard_age_sec": precheck_gate.get("guard_age_sec"),
            "event_precheck_guard_remaining_sec": precheck_gate.get("guard_remaining_sec"),
            "event_exit_confirm_mode": str(precheck_gate.get("confirm_mode") or "idle"),
            "event_exit_confirm_required": int(precheck_gate.get("confirm_required") or 1),
            "event_exit_confirm_count": int(precheck_gate.get("confirm_count") or 0),
            "event_exit_confirmed": bool(precheck_gate.get("confirm_ok")),
            "event_exit_guard": pre_guard_reason if pre_guard_reason and pre_guard_reason != "idle" else None,
            "event_exit_guard_progress": precheck_gate.get("guard_progress"),
            "event_exit_guard_required": precheck_gate.get("guard_required"),
            "event_exit_guard_hold_target_sec": precheck_gate.get("guard_hold_target_sec"),
            "event_exit_guard_age_sec": precheck_gate.get("guard_age_sec"),
            "event_exit_guard_remaining_sec": precheck_gate.get("guard_remaining_sec"),
            "event_exit_dynamic_mode": str(exit_diag.get("mode") or "static"),
            "event_exit_dynamic_hurst": exit_diag.get("hurst"),
            "event_exit_dynamic_vpin": exit_diag.get("vpin"),
            "event_exit_dynamic_mu_alpha": exit_diag.get("mu_alpha"),
            "event_exit_dynamic_mu_alignment": exit_diag.get("mu_alignment"),
            "event_exit_time_stop_mult": float(exit_policy_dyn.time_stop_mult),
            "event_exit_min_p_tp": float(core_eval.get("threshold_ptp")),
            "event_tp_pct": float(tp_pct),
            "event_sl_pct": float(sl_pct),
            "event_t_median": float(t_med_evt) if t_med_evt is not None else None,
        }

    def _check_ema_ev_exit(self, sym: str, decision: dict, regime: str, price: float, ts: int) -> bool:
        """EMA-based EV/PSL deterioration exit. Returns True if exited."""
        try:
            min_hold_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
        except Exception:
            min_hold_sec = float(POSITION_HOLD_MIN_SEC)
        if min_hold_sec > 0:
            pos = self.positions.get(sym) or {}
            try:
                entry_ts = int(pos.get("entry_time") or 0)
            except Exception:
                entry_ts = 0
            if entry_ts:
                age_sec = (ts - entry_ts) / 1000.0
                if age_sec < float(min_hold_sec):
                    return False
        meta = decision.get("meta") or {}
        ev_now = float(decision.get("unified_score", decision.get("ev", 0.0)) or 0.0)
        p_sl_now = float(meta.get("event_p_sl", 0.0) or 0.0)
        ev_ema = self._ema_update(self._ema_ev, sym, ev_now, half_life_sec=30, ts_ms=ts)
        psl_ema = self._ema_update(self._ema_psl, sym, p_sl_now, half_life_sec=30, ts_ms=ts)
        d_ev = ev_now - ev_ema
        d_psl = p_sl_now - psl_ema
        ev_floor_reg = EV_EXIT_FLOOR.get(regime, -0.0002)
        ev_drop_reg = EV_DROP.get(regime, 0.0008)
        psl_rise_reg = PSL_RISE.get(regime, 0.03)
        persist_need = 1 if regime in ("bull", "bear") else 2

        if (ev_now < ev_floor_reg) and (d_ev < -ev_drop_reg) and (d_psl > psl_rise_reg):
            self._exit_bad_ticks[sym] = self._exit_bad_ticks.get(sym, 0) + 1
        else:
            self._exit_bad_ticks[sym] = 0

        if self._exit_bad_ticks[sym] >= persist_need:
            self._close_position(sym, float(price), "ev_psl_ema_exit", exit_kind="RISK")
            return True
        return False

    def _check_ev_drop_exit(self, sym: str, decision: dict, regime: str, price: float, ts: int) -> bool:
        """EV drop exit logic. Returns True if exited."""
        try:
            min_hold_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
        except Exception:
            min_hold_sec = float(POSITION_HOLD_MIN_SEC)
        if min_hold_sec > 0:
            pos = self.positions.get(sym) or {}
            try:
                entry_ts = int(pos.get("entry_time") or 0)
            except Exception:
                entry_ts = 0
            if entry_ts:
                age_sec = (ts - entry_ts) / 1000.0
                if age_sec < float(min_hold_sec):
                    return False
        ev_now = float(decision.get("unified_score", decision.get("ev", 0.0)) or 0.0)
        meta_now = decision.get("meta") or {}
        ev_floor = float(meta_now.get("ev_entry_threshold_dyn") or meta_now.get("ev_entry_threshold") or 0.0)
        prev_ev = self._ev_drop_state[sym].get("prev")
        delta_ev = None
        if prev_ev is not None:
            delta_ev = ev_now - prev_ev

        strong_signal = decision.get("confidence", 0.0) >= 0.65
        needed_ticks = 1 if strong_signal else 2
        if ev_now < ev_floor and (delta_ev is not None) and (delta_ev < -EV_DROP_THRESHOLD):
            self._ev_drop_state[sym]["streak"] += 1
        else:
            self._ev_drop_state[sym]["streak"] = 0
        self._ev_drop_state[sym]["prev"] = ev_now

        if self._ev_drop_state[sym]["streak"] >= needed_ticks:
            self._close_position(sym, float(price), "ev_drop_exit", exit_kind="RISK")
            return True
        return False

    def _decide_v3(self, ctx: dict) -> dict:
        """
        v3 flow:
          1) Execution gate (spread)
          2) Alpha candidate (direction + OFI) -> choose side or WAIT
          3) MC validate at leverage=1 using EngineHub.decide(ctx)
          4) Risk-based leverage compute
          5) (optional) Re-run hub.decide with final leverage for reporting/meta
        Returns a decision dict compatible with existing downstream code.
        """

        sym = ctx.get("symbol")
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))

        # -------------------------
        # Stage-1: Execution gate
        # -------------------------
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # fallback (2bp)
        spread_pct = float(spread_pct)

        spread_cap_map = {
            "bull": 0.0020,
            "bear": 0.0020,
            "chop": 0.0012,
            "volatile": 0.0008,
        }
        spread_cap = spread_cap_map.get(regime, 0.0012)
        if spread_pct > spread_cap:
            return {
                "action": "WAIT",
                "reason": "v3_exec_gate_spread",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {"regime": regime, "session": session, "spread_pct": spread_pct, "spread_cap": spread_cap},
                "details": [],
            }

        # -------------------------
        # Stage-2: Alpha candidate (direction + OFI residual-ish)
        # -------------------------
        direction = float(ctx.get("direction", 1))  # +1 / -1
        ofi = float(ctx.get("ofi_score", 0.0))
        ofi_mean = self.stats.ema_update("ofi_mean_v3", (regime, session), ofi, half_life_sec=900)
        ofi_res = ofi - float(ofi_mean)

        # simple scale proxy using recent abs-mean from stats buffer if available
        # (if not enough samples, it'll behave like "raw")
        try:
            z_ofi = self.stats.robust_z("ofi_res", (regime, session), ofi_res, fallback=ofi_res / 0.001)
        except Exception:
            z_ofi = ofi_res / 0.001

        alpha_long = 0.55 * direction + 0.35 * z_ofi
        alpha_short = -0.55 * direction - 0.35 * z_ofi
        alpha_max = max(alpha_long, alpha_short)

        alpha_min_map = {"bull": 0.25, "bear": 0.25, "chop": 0.40, "volatile": 0.55}
        alpha_min = alpha_min_map.get(regime, 0.40)
        if alpha_max < alpha_min:
            return {
                "action": "WAIT",
                "reason": "v3_alpha_gate",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {
                    "regime": regime, "session": session,
                    "alpha_long": alpha_long, "alpha_short": alpha_short,
                    "alpha_min": alpha_min, "z_ofi": z_ofi,
                    "spread_pct": spread_pct,
                },
                "details": [],
            }

        alpha_side = "LONG" if alpha_long >= alpha_short else "SHORT"

        # -------------------------
        # Stage-3: MC validate at leverage=1 (using existing EngineHub)
        # -------------------------
        ctx1 = dict(ctx)
        ctx1["leverage"] = 1.0
        decision1 = self.hub.decide(ctx1)

        ev1 = float(decision1.get("ev", 0.0) or 0.0)
        win1 = float(decision1.get("confidence", 0.0) or 0.0)
        cvar1 = float(decision1.get("cvar", 0.0) or 0.0)
        meta1 = decision1.get("meta") or {}
        
        # ✅ DEBUG: Log MC direction vs alpha_side for verification
        mc_dir_stage3 = int(meta1.get("direction", 0))
        mc_side_stage3 = "LONG" if mc_dir_stage3 == 1 else "SHORT" if mc_dir_stage3 == -1 else "WAIT"
        mu_alpha = float(meta1.get("mu_alpha", 0.0) or 0.0)
        if mc_side_stage3 != alpha_side:
            print(f"[DIR_MISMATCH] {sym} | alpha_side={alpha_side} mc_side={mc_side_stage3} mu={mu_alpha:.4f} (MC direction will be used)", flush=True)

        p_sl = float(meta1.get("event_p_sl", 0.0) or 0.0)
        event_cvar_r = meta1.get("event_cvar_r")
        event_cvar_r = float(event_cvar_r) if event_cvar_r is not None else -999.0

        # v3 lev=1 gates (완화된 버전)
        ev1_floor = {"bull": 0.0002, "bear": 0.0002, "chop": 0.0005, "volatile": 0.0008}.get(regime, 0.0005)
        win1_floor = {"bull": 0.50, "bear": 0.50, "chop": 0.52, "volatile": 0.53}.get(regime, 0.52)
        cvar1_floor = {"bull": -0.010, "bear": -0.011, "chop": -0.008, "volatile": -0.007}.get(regime, -0.010)
        psl_max = {"bull": 0.42, "bear": 0.40, "chop": 0.35, "volatile": 0.32}.get(regime, 0.40)
        event_cvar_r_floor = {"bull": -1.20, "bear": -1.15, "chop": -1.05, "volatile": -0.95}.get(regime, -1.10)

        if ev1 < ev1_floor or win1 < win1_floor or cvar1 < cvar1_floor or p_sl > psl_max or event_cvar_r < event_cvar_r_floor:
            # decision1 포맷 유지 + reason만 v3로 덮어쓰기
            d = dict(decision1)
            d["action"] = "WAIT"
            d["reason"] = "v3_mc1_gate"
            m = dict(meta1)
            m.update({
                "v3_alpha_side": alpha_side,
                "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
                "ev1_floor": ev1_floor, "win1_floor": win1_floor, "cvar1_floor": cvar1_floor,
                "psl_max": psl_max, "event_cvar_r_floor": event_cvar_r_floor,
                "spread_pct": spread_pct,
            })
            d["meta"] = m
            return d

        # -------------------------
        # Stage-4: Risk-based leverage (use lev=1 metrics)
        # -------------------------
        sl_pct = float(meta1.get("mc_sl") or meta1.get("sl_pct") or 0.002)
        event_cvar_pct = event_cvar_r * sl_pct

        # best-effort slippage from meta if present
        slippage_pct = float(meta1.get("slippage_pct", 0.0) or 0.0)
        sigma = float(ctx.get("sigma", 0.0) or 0.0)

        risk = max(abs(cvar1), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma
        if risk < 1e-9:
            risk = 1e-9

        # leverage bounds per regime
        lev_max = {"bull": 20.0, "bear": 18.0, "chop": 10.0, "volatile": 8.0}.get(regime, 10.0)
        lev_raw = (max(ev1, 0.0) / risk) * K_LEV
        lev = float(max(1.0, min(lev_max, lev_raw)))

        # -------------------------
        # Stage-5: Re-run hub.decide with final leverage (reporting & size_frac)
        # -------------------------
        ctxF = dict(ctx)
        ctxF["leverage"] = lev
        decisionF = self.hub.decide(ctxF)

        # ✅ FIX: Use MC engine's direction (from EV comparison), NOT alpha_side from _direction_bias
        # MC engine calculates ev_long vs ev_short based on mu (drift), which is the true signal
        decisionF = dict(decisionF)
        mc_direction = int((decisionF.get("meta") or {}).get("direction", 0))
        if mc_direction == 1:
            mc_side = "LONG"
        elif mc_direction == -1:
            mc_side = "SHORT"
        else:
            mc_side = "WAIT"
        
        # Only use mc_side if MC engine gave a valid direction, otherwise fall back to WAIT
        if decisionF.get("action") in ("LONG", "SHORT") and mc_side in ("LONG", "SHORT"):
            decisionF["action"] = mc_side
        else:
            decisionF["action"] = "WAIT"

        metaF = dict(decisionF.get("meta") or {})
        metaF.update({
            "v3_used": True,
            "v3_alpha_side": alpha_side,
            "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
            "risk": risk, "lev_raw": lev_raw, "lev": lev,
            "event_cvar_pct": event_cvar_pct,
            "spread_pct": spread_pct,
        })
        decisionF["meta"] = metaF

        # expose leverage/size_frac for your existing sizing code
        decisionF["leverage"] = lev

        # size cap by regime (your code already applies regime_cap_frac later, but we can prefill)
        cap_map = {"bull": 0.25, "bear": 0.22, "chop": 0.10, "volatile": 0.08}
        metaF.setdefault("regime_cap_frac", cap_map.get(regime, 0.10))

        return decisionF

    async def fetch_prices_loop(self):
        try:
            interval = float(os.environ.get("PRICE_REFRESH_SEC", 1.0) or 1.0)
        except Exception:
            interval = 1.0
        interval = max(0.2, interval)
        while True:
            try:
                tickers = await self._ccxt_call("fetch_tickers", self.data_exchange.fetch_tickers, SYMBOLS)
                ts = now_ms()
                ok_any = False
                for s in SYMBOLS:
                    last = (tickers.get(s) or {}).get("last")
                    if last is not None:
                        px = float(last)
                        self.market[s]["price"] = px
                        self.market[s]["ts"] = ts
                        self._append_tick_price(s, px, ts)
                        ok_any = True
                if ok_any:
                    self._last_feed_ok_ms = ts
            except Exception as e:
                self._log_err(f"[ERR] fetch_tickers: {e}")
            await asyncio.sleep(interval)  # ✅ too tight can hit Bybit limits

    async def preload_all_ohlcv(self, limit: int = OHLCV_PRELOAD_LIMIT):
        """
        서버 시작 전에 OHLCV를 미리 채워서 'candles 부족 / 준비중' 시간을 없앰.
        """
        for sym in SYMBOLS:
            try:
                ohlcv = await self.data_exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
                if not ohlcv:
                    continue
                self.ohlcv_buffer[sym].clear()
                self.ohlcv_open[sym].clear()
                self.ohlcv_high[sym].clear()
                self.ohlcv_low[sym].clear()
                self.ohlcv_volume[sym].clear()
                last_ts = 0
                for c in ohlcv:
                    ts_ms = int(c[0])
                    open_price = float(c[1])
                    high_price = float(c[2])
                    low_price = float(c[3])
                    close_price = float(c[4])
                    vol_val = float(c[5]) if len(c) > 5 else 0.0
                    self.ohlcv_open[sym].append(open_price)
                    self.ohlcv_high[sym].append(high_price)
                    self.ohlcv_low[sym].append(low_price)
                    self.ohlcv_volume[sym].append(vol_val)
                    self.ohlcv_buffer[sym].append(close_price)
                    last_ts = ts_ms
                self._last_kline_ts[sym] = last_ts
                self._last_kline_ok_ms[sym] = now_ms()
                self._preloaded[sym] = True
                self._log(f"[PRELOAD] {sym} candles={len(self.ohlcv_buffer[sym])}")
            except Exception as e:
                self._log_err(f"[ERR] preload_ohlcv {sym}: {e}")

    async def fetch_ohlcv_loop(self):
        """
        - preload은 main()에서 한 번에 수행
        - 이후 최신 1분봉만 dedupe 갱신
        """
        while True:
            start = now_ms()
            try:
                for sym in SYMBOLS:
                    try:
                        ohlcv = await self._ccxt_call(
                            f"fetch_ohlcv {sym}",
                            self.data_exchange.fetch_ohlcv,
                            sym, timeframe=TIMEFRAME, limit=OHLCV_REFRESH_LIMIT
                        )
                        if not ohlcv:
                            continue
                        last = ohlcv[-1]
                        ts_ms = int(last[0])
                        open_price = float(last[1])
                        high_price = float(last[2])
                        low_price = float(last[3])
                        close_price = float(last[4])
                        vol_val = float(last[5]) if len(last) > 5 else 0.0

                        if ts_ms != self._last_kline_ts[sym]:
                            self.ohlcv_open[sym].append(open_price)
                            self.ohlcv_high[sym].append(high_price)
                            self.ohlcv_low[sym].append(low_price)
                            self.ohlcv_volume[sym].append(vol_val)
                            self.ohlcv_buffer[sym].append(close_price)
                            self._last_kline_ts[sym] = ts_ms
                            self._last_kline_ok_ms[sym] = now_ms()
                    except Exception as e_sym:
                        self._log_err(f"[ERR] fetch_ohlcv {sym}: {e_sym}")
            except Exception as e:
                self._log_err(f"[ERR] fetch_ohlcv(loop): {e}")

            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(1.0, OHLCV_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    async def fetch_orderbook_loop(self):
        """
        대시보드 Orderbook Ready = NO 해결용.
        - 심볼별 fetch_order_book 수행
        """
        while True:
            start = now_ms()
            symbols = list(SYMBOLS)
            tasks = [
                self._ccxt_call(
                    f"fetch_orderbook {sym}",
                    self.data_exchange.fetch_order_book,
                    sym, limit=ORDERBOOK_DEPTH,
                    semaphore=self._ob_sem,
                )
                for sym in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(symbols, results):
                if isinstance(result, Exception):
                    # rate limit/권한 이슈 등은 ready False로 유지
                    self.orderbook[sym]["ready"] = False
                    msg = str(result)
                    if ("RequestTimeout" not in msg) and ("DDoSProtection" not in msg) and ("RateLimit" not in msg):
                        extra = ""
                        try:
                            status = getattr(result, "status", None)
                            body = getattr(result, "body", None)
                            if status is not None:
                                extra += f" status={status}"
                            if body:
                                extra += f" body={str(body)[:300]}"
                        except Exception:
                            pass
                        self._log_err(f"[ERR] fetch_orderbook {sym}: {repr(result)}{extra}")
                    continue

                ob = result or {}
                bids = (ob.get("bids") or [])[:ORDERBOOK_DEPTH]
                asks = (ob.get("asks") or [])[:ORDERBOOK_DEPTH]
                ready = bool(bids) and bool(asks)
                self.orderbook[sym]["bids"] = bids
                self.orderbook[sym]["asks"] = asks
                self.orderbook[sym]["ready"] = ready
                self.orderbook[sym]["ts"] = now_ms()

            elapsed = (now_ms() - start) / 1000.0
            # 전체 한바퀴 주기를 ORDERBOOK_SLEEP_SEC에 맞춤
            sleep_left = max(0.0, ORDERBOOK_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    def _normalize_exchange_positions(self, raw_positions: list[dict]) -> dict[str, dict]:
        out = {}
        for p in raw_positions or []:
            try:
                sym = p.get("symbol") or p.get("info", {}).get("symbol")
            except Exception:
                sym = None
            if not sym:
                continue
            sym = _normalize_sym_key(sym)
            size = None
            for key in ("contracts", "positionAmt", "size"):
                if size is None:
                    try:
                        size = p.get(key)
                    except Exception:
                        size = None
            if size is None:
                try:
                    size = p.get("info", {}).get("size")
                except Exception:
                    size = None
            try:
                size_f = float(size or 0.0)
            except Exception:
                size_f = 0.0
            p["size_f"] = size_f
            # Best-effort liquidation price from raw payload
            if "liqPrice" not in p and "liq_price" not in p:
                try:
                    liq_raw = p.get("info", {}).get("liqPrice")
                except Exception:
                    liq_raw = None
                if liq_raw is not None:
                    p["liqPrice"] = liq_raw
            out[str(sym)] = p
        return out

    def _set_exchange_positions_view(self, positions: list[dict], now_ts: int, source: str) -> None:
        view = list(positions or [])
        by_symbol: dict[str, dict] = {}
        for p in view:
            if not isinstance(p, dict):
                continue
            sym = _normalize_sym_key(p.get("symbol"))
            if not sym:
                continue
            by_symbol[sym] = p
        self._exchange_positions_view = view
        self._exchange_positions_by_symbol = by_symbol
        self._exchange_positions_ts = int(now_ts)
        self._exchange_positions_source = str(source or "rest")

    def _extract_exchange_position_risk_metrics(
        self,
        ex: dict,
        info: dict,
        *,
        notional: float | None = None,
        leverage: float | None = None,
        mark_px: float | None = None,
        entry_px: float | None = None,
        qty: float | None = None,
        side: str | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Best-effort extraction of unrealized PnL, margin and ROE from exchange payload."""

        def _pick_num(*vals):
            for v in vals:
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    return fv
            return None

        unrealized_pnl = _pick_num(
            ex.get("unrealizedPnl"),
            ex.get("unrealisedPnl"),
            ex.get("upl"),
            info.get("unrealizedPnl"),
            info.get("unrealisedPnl"),
            info.get("curUnrealisedPnl"),
            info.get("curUnrealizedPnl"),
            info.get("upl"),
        )
        margin = _pick_num(
            ex.get("initialMargin"),
            ex.get("margin"),
            ex.get("positionMargin"),
            info.get("positionIM"),
            info.get("positionIMByMp"),
            info.get("initialMargin"),
            info.get("positionMargin"),
            info.get("positionBalance"),
            info.get("positionMM"),
            info.get("positionMaintMargin"),
        )
        if margin is not None and margin <= 0:
            margin = None
        roe = _pick_num(
            ex.get("roe"),
            ex.get("percentage"),
            ex.get("unrealizedPnlPcnt"),
            ex.get("unrealisedPnlPcnt"),
            info.get("roe"),
            info.get("positionRoi"),
            info.get("positionROI"),
            info.get("unrealizedPnlPcnt"),
            info.get("unrealisedPnlPcnt"),
            info.get("unrealizedProfitRate"),
        )
        if roe is not None and abs(float(roe)) > 5.0:
            # Some exchanges report percent points (e.g. 12.3) instead of fraction (0.123).
            roe = float(roe) / 100.0

        if unrealized_pnl is None and (mark_px is not None) and (entry_px is not None) and (qty is not None) and side:
            try:
                if str(side).upper() == "LONG":
                    unrealized_pnl = (float(mark_px) - float(entry_px)) * float(qty)
                elif str(side).upper() == "SHORT":
                    unrealized_pnl = (float(entry_px) - float(mark_px)) * float(qty)
            except Exception:
                unrealized_pnl = None

        if margin is None:
            try:
                if (notional is not None) and (leverage is not None) and float(leverage) > 0:
                    margin = float(notional) / max(float(leverage), 1e-6)
            except Exception:
                margin = None
        if margin is not None and margin <= 0:
            margin = None

        if roe is None and (unrealized_pnl is not None) and (margin is not None) and float(margin) > 0:
            try:
                roe = float(unrealized_pnl) / float(margin)
            except Exception:
                roe = None

        return unrealized_pnl, margin, roe

    def _build_exchange_positions_view(self, pos_map: dict[str, dict], now_ts: int) -> list[dict]:
        """Build a lightweight positions list from exchange data for dashboard display."""
        out: list[dict] = []
        for sym, ex in (pos_map or {}).items():
            try:
                size = float(ex.get("size_f", 0.0) or 0.0)
            except Exception:
                size = 0.0
            if abs(size) < 1e-9:
                continue
            info = ex.get("info") or {}
            side_raw = ex.get("side") or info.get("side")
            side = None
            if side_raw:
                s = str(side_raw).lower()
                if "long" in s or "buy" in s:
                    side = "LONG"
                elif "short" in s or "sell" in s:
                    side = "SHORT"
            if side is None:
                side = "SHORT" if size < 0 else "LONG"
            qty = abs(size)

            def _pick(*vals):
                for v in vals:
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isfinite(fv) and fv > 0:
                        return fv
                return None

            entry_px = _pick(ex.get("entryPrice"), ex.get("entry_price"), info.get("entryPrice"), info.get("avgPrice"))
            mark_px = _pick(ex.get("markPrice"), ex.get("lastPrice"), info.get("markPrice"), info.get("lastPrice"))
            if entry_px is None:
                entry_px = mark_px
            liq_px = _pick(ex.get("liqPrice"), ex.get("liq_price"), info.get("liqPrice"))
            notional = _pick(ex.get("notional"), info.get("positionValue"))
            if notional is None and entry_px is not None:
                notional = abs(qty * entry_px)
            if notional is None and mark_px is not None:
                notional = abs(qty * mark_px)
            leverage = _pick(ex.get("leverage"), info.get("leverage")) or float(self.leverage)
            unrealized_pnl, margin, roe = self._extract_exchange_position_risk_metrics(
                ex,
                info,
                notional=notional,
                leverage=leverage,
                mark_px=mark_px,
                entry_px=entry_px,
                qty=qty,
                side=side,
            )

            entry_time = self._extract_entry_time_ms_from_info(info, default=now_ts)

            cap_frac = float(notional / self.balance) if (self.balance and notional) else 0.0
            size_frac = 0.0
            try:
                size_frac = float(notional or 0.0) / max(float(self.balance) * max(float(leverage), 1e-6), 1e-6)
            except Exception:
                size_frac = 0.0
            try:
                prior = self.positions.get(sym) or {}
            except Exception:
                prior = {}
            entry_leverage = prior.get("entry_leverage")
            if entry_leverage in (None, 0):
                entry_leverage = float(leverage)
            entry_notional = prior.get("entry_notional")
            if entry_notional in (None, 0):
                entry_notional = float(notional or 0.0)

            out.append({
                "symbol": sym,
                "side": side,
                "entry_price": float(entry_px or 0.0),
                "entry_time": int(entry_time),
                "quantity": float(qty),
                "notional": float(notional or 0.0),
                "entry_notional": float(entry_notional or 0.0),
                "size_frac": float(size_frac),
                "leverage": float(leverage),
                "entry_leverage": float(entry_leverage),
                "cap_frac": float(cap_frac),
                "current": float(mark_px) if mark_px is not None else None,
                "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
                "margin": float(margin) if margin is not None else None,
                "roe": float(roe) if roe is not None else None,
                "liq_price": float(liq_px) if liq_px is not None else None,
                "pos_source": "exchange",
            })
        return out

    def _positions_fetch_params(self) -> dict:
        """Best-effort params for Bybit position fetches (unified account needs category)."""
        try:
            cat = str(os.environ.get("BYBIT_POSITIONS_CATEGORY", "")).strip()
        except Exception:
            cat = ""
        if not cat:
            try:
                if any((":USDT" in s) or str(s).endswith("USDT") for s in SYMBOLS):
                    cat = "linear"
            except Exception:
                cat = ""
        if not cat:
            return {}
        return {"category": cat}

    def _bybit_ws_endpoint(self) -> str:
        try:
            override = str(os.environ.get("BYBIT_WS_PRIVATE_ENDPOINT", "")).strip()
        except Exception:
            override = ""
        if override:
            return override
        try:
            use_testnet = bool(getattr(config, "BYBIT_TESTNET", False))
        except Exception:
            use_testnet = False
        return "wss://stream-testnet.bybit.com/v5/private" if use_testnet else "wss://stream.bybit.com/v5/private"

    def _bybit_ws_signature(self, api_key: str, api_secret: str, expires_ms: int, mode: str = "realtime") -> str:
        # Bybit WS v5 uses HMAC SHA256; default payload follows legacy GET/realtime scheme.
        if mode == "v5":
            payload = f"{expires_ms}{api_key}"
        else:
            payload = f"GET/realtime{expires_ms}"
        return hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _apply_ws_positions(self, entries: list[dict], now_ts: int, reset: bool = False) -> None:
        if reset:
            self._ws_positions_cache = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            raw_sym = item.get("symbol") or item.get("symbolName") or item.get("info", {}).get("symbol")
            sym = _normalize_sym_key(raw_sym)
            if not sym:
                continue
            size_val = item.get("size")
            if size_val is None:
                size_val = item.get("positionAmt")
            if size_val is None:
                size_val = item.get("qty")
            try:
                size = float(size_val or 0.0)
            except Exception:
                size = 0.0
            side_raw = item.get("side") or item.get("positionSide")
            side = None
            if side_raw:
                s = str(side_raw).lower()
                if "long" in s or "buy" in s:
                    side = "LONG"
                elif "short" in s or "sell" in s:
                    side = "SHORT"
            if side is None:
                side = "SHORT" if size < 0 else "LONG"
            size_abs = abs(size)
            if size_abs < 1e-12:
                self._ws_positions_cache.pop(sym, None)
                continue
            ex = {
                "symbol": sym,
                "side": side,
                "size_f": size_abs if side == "LONG" else -size_abs,
                "entryPrice": item.get("entryPrice") or item.get("avgPrice"),
                "markPrice": item.get("markPrice") or item.get("lastPrice"),
                "notional": item.get("positionValue") or item.get("notional"),
                "leverage": item.get("leverage"),
                "unrealizedPnl": (
                    item.get("unrealizedPnl")
                    if item.get("unrealizedPnl") is not None
                    else item.get("unrealisedPnl")
                ),
                "initialMargin": item.get("positionIM") if item.get("positionIM") is not None else item.get("positionMargin"),
                "roe": item.get("positionRoi") if item.get("positionRoi") is not None else item.get("roe"),
                "liqPrice": item.get("liqPrice") or item.get("liq_price"),
                "info": item,
            }
            self._ws_positions_cache[sym] = ex
        try:
            self._set_exchange_positions_view(
                self._build_exchange_positions_view(self._ws_positions_cache, now_ts),
                now_ts,
                "ws",
            )
            self._ws_positions_last_ms = int(now_ts)
        except Exception:
            pass

    async def bybit_ws_positions_loop(self) -> None:
        """Optional Bybit private WS positions stream for faster dashboard sync."""
        while True:
            enabled = str(os.environ.get("BYBIT_WS_POSITIONS", "0")).strip().lower() in ("1", "true", "yes", "on")
            if not enabled:
                await asyncio.sleep(2.0)
                continue
            if not self.enable_orders:
                await asyncio.sleep(2.0)
                continue
            api_key = getattr(self.exchange, "apiKey", None)
            api_secret = getattr(self.exchange, "secret", None)
            if not api_key or not api_secret:
                self._log_err("[BYBIT_WS] apiKey/secret missing; skip WS positions")
                await asyncio.sleep(5.0)
                continue
            endpoint = self._bybit_ws_endpoint()
            try:
                reconnect_sec = float(os.environ.get("BYBIT_WS_RECONNECT_SEC", 5.0) or 5.0)
            except Exception:
                reconnect_sec = 5.0
            try:
                auth_mode = str(os.environ.get("BYBIT_WS_AUTH_MODE", "realtime") or "realtime").strip().lower()
            except Exception:
                auth_mode = "realtime"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(endpoint, heartbeat=20) as ws:
                        self._ws_positions_connected = True
                        self._ws_positions_last_err = None
                        expires_ms = int(time.time() * 1000) + 10_000
                        sign = self._bybit_ws_signature(str(api_key), str(api_secret), expires_ms, mode=auth_mode)
                        await ws.send_json({"op": "auth", "args": [api_key, expires_ms, sign]})
                        await ws.send_json({"op": "subscribe", "args": ["position"]})
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                except Exception:
                                    continue
                                if isinstance(data, dict):
                                    op = data.get("op")
                                    if op == "auth":
                                        if not data.get("success", False):
                                            self._log_err(f"[BYBIT_WS] auth failed: {data}")
                                            break
                                    if op == "subscribe":
                                        if not data.get("success", True):
                                            self._log_err(f"[BYBIT_WS] subscribe failed: {data}")
                                            break
                                    topic = data.get("topic") or ""
                                    if str(topic).startswith("position"):
                                        entries = data.get("data")
                                        if isinstance(entries, dict):
                                            entries = [entries]
                                        if isinstance(entries, list):
                                            reset = str(data.get("type") or "").lower() == "snapshot"
                                            self._apply_ws_positions(entries, now_ms(), reset=reset)
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
            except Exception as e:
                self._ws_positions_last_err = str(e)
                self._log_err(f"[BYBIT_WS] positions loop error: {e}")
            finally:
                self._ws_positions_connected = False
            await asyncio.sleep(max(1.0, reconnect_sec))

    def _ensure_symbol_state(self, sym: str) -> None:
        if sym not in self.market:
            self.market[sym] = {"price": None, "ts": 0}
        if sym not in self.ohlcv_buffer:
            self.ohlcv_buffer[sym] = deque(maxlen=OHLCV_PRELOAD_LIMIT)
        if sym not in self.ohlcv_open:
            self.ohlcv_open[sym] = deque(maxlen=OHLCV_PRELOAD_LIMIT)
        if sym not in self.ohlcv_high:
            self.ohlcv_high[sym] = deque(maxlen=OHLCV_PRELOAD_LIMIT)
        if sym not in self.ohlcv_low:
            self.ohlcv_low[sym] = deque(maxlen=OHLCV_PRELOAD_LIMIT)
        if sym not in self.ohlcv_volume:
            self.ohlcv_volume[sym] = deque(maxlen=OHLCV_PRELOAD_LIMIT)
        if sym not in self.orderbook:
            self.orderbook[sym] = {"ts": 0, "ready": False, "bids": [], "asks": []}
        if sym not in self._last_kline_ts:
            self._last_kline_ts[sym] = 0
        if sym not in self._last_kline_ok_ms:
            self._last_kline_ok_ms[sym] = 0
        if sym not in self._preloaded:
            self._preloaded[sym] = False
        if sym not in self._cooldown_until:
            self._cooldown_until[sym] = 0.0
        if sym not in self._entry_streak:
            self._entry_streak[sym] = 0
        if sym not in self._last_exit_kind:
            self._last_exit_kind[sym] = "NONE"
        if sym not in self._streak:
            self._streak[sym] = 0
        if sym not in self._ev_tune_hist:
            self._ev_tune_hist[sym] = deque(maxlen=2000)
        if sym not in self._ev_hist:
            self._ev_hist[sym] = deque(maxlen=400)
        if sym not in self._cvar_hist:
            self._cvar_hist[sym] = deque(maxlen=400)
        if sym not in self._ev_drop_state:
            self._ev_drop_state[sym] = {"prev": None, "streak": 0}
        if sym not in self._exit_bad_ticks:
            self._exit_bad_ticks[sym] = 0
        if sym not in self._dyn_leverage:
            self._dyn_leverage[sym] = float(self.leverage)
        if sym not in self._live_missing_pos_counts:
            self._live_missing_pos_counts[sym] = 0
        # alpha state
        if sym not in self._alpha_state:
            self._get_alpha_state(sym)
        if sym not in self._last_ok.get("ohlcv", {}):
            self._last_ok.setdefault("ohlcv", {})[sym] = 0
        if sym not in self._last_ok.get("ob", {}):
            self._last_ok.setdefault("ob", {})[sym] = 0

    def _extract_ticker_volume(self, t: dict) -> float | None:
        def _f(x):
            try:
                return float(x)
            except Exception:
                return None
        if not isinstance(t, dict):
            return None
        v = _f(t.get("quoteVolume"))
        if v is None:
            info = t.get("info") or {}
            v = _f(info.get("turnover24h")) or _f(info.get("turnover24hUSDT")) or _f(info.get("volume24h"))
        if v is None:
            base = _f(t.get("baseVolume"))
            last = _f(t.get("last"))
            if base is not None and last is not None:
                v = float(base) * float(last)
        return v

    def _apply_dynamic_universe(self, new_syms: list[str], *, reason: str = "top_volume") -> None:
        global SYMBOLS, TOP_N_SYMBOLS
        # keep order, remove duplicates
        seen = set()
        cleaned = []
        for s in new_syms:
            if not s or s in seen:
                continue
            cleaned.append(s)
            seen.add(s)
        if not cleaned:
            return
        if cleaned == SYMBOLS:
            return
        old_syms = list(SYMBOLS)
        old_set = set(old_syms)
        new_set = set(cleaned)
        add = [s for s in cleaned if s not in old_set]

        # ensure state for new symbols
        for sym in add:
            self._ensure_symbol_state(sym)

        # archive positions outside new universe
        self._runtime_universe = new_set
        try:
            self._archive_outside_universe_positions(universe=new_set, source=reason)
        except Exception:
            pass

        SYMBOLS = cleaned
        # update TOP_N limit to avoid out-of-range
        if TOP_N_SYMBOLS <= 0 or TOP_N_SYMBOLS > len(SYMBOLS):
            TOP_N_SYMBOLS = len(SYMBOLS)
        # refresh index map for SoA
        try:
            self._sym_to_idx = {s: i for i, s in enumerate(SYMBOLS)}
        except Exception:
            pass
        if len(SYMBOLS) > int(self._batch_max_symbols):
            self._log_err(f"[UNIVERSE] dynamic symbols={len(SYMBOLS)} exceed batch_max={self._batch_max_symbols}")

        self._log(f"[UNIVERSE] dynamic update: {len(old_set)} -> {len(new_set)} ({reason})")

    def _allow_empty_position_sync(self, now_ts: int) -> bool:
        try:
            confirm_count = int(os.environ.get("SYNC_EMPTY_CONFIRM_COUNT", 2) or 2)
        except Exception:
            confirm_count = 2
        try:
            grace_sec = float(os.environ.get("SYNC_EMPTY_GRACE_SEC", 30.0) or 30.0)
        except Exception:
            grace_sec = 30.0
        confirm_count = max(1, confirm_count)
        grace_ms = int(max(0.0, grace_sec) * 1000.0)
        last_ok = int(self._last_nonempty_pos_fetch_ms or 0)
        if last_ok and (now_ts - last_ok) < grace_ms:
            return False
        return self._empty_positions_fetches >= confirm_count

    async def _sync_positions_from_exchange(
        self,
        *,
        overwrite: bool = True,
        reason: str = "startup",
        raw_positions: list[dict] | None = None,
        keep_pending: bool = False,
        include_all: bool | None = None,
    ) -> None:
        if not self.enable_orders:
            return
        if not getattr(self.exchange, "apiKey", None) or not getattr(self.exchange, "secret", None):
            self._log_err("[SYNC_POS] apiKey/secret missing; skip exchange position sync")
            return
        if include_all is None:
            include_all = bool(getattr(config, "SYNC_POSITIONS_ALL", True))
        try:
            fetch_params = self._positions_fetch_params()
            raw = raw_positions
            if raw is None:
                try:
                    if include_all:
                        if fetch_params:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions, None, fetch_params)
                        else:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions)
                    else:
                        if fetch_params:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions, SYMBOLS, fetch_params)
                        else:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions, SYMBOLS)
                except Exception as e:
                    msg = str(e)
                    if "fetchPositions() does not accept an array" in msg or "does not accept an array" in msg:
                        if fetch_params:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions, None, fetch_params)
                        else:
                            raw = await self._ccxt_call("fetch_positions(sync)", self.exchange.fetch_positions)
                    else:
                        raise
            pos_map = self._normalize_exchange_positions(raw if isinstance(raw, list) else [])
        except Exception as e:
            self._log_err(f"[SYNC_POS] fetch_positions failed: {e}")
            return

        now_ts = now_ms()
        new_positions: dict[str, dict] = {}
        for sym, ex in pos_map.items():
            try:
                size = float(ex.get("size_f", 0.0) or 0.0)
            except Exception:
                size = 0.0
            if abs(size) < 1e-9:
                continue
            info = ex.get("info") or {}
            side_raw = ex.get("side") or info.get("side")
            side = None
            if side_raw:
                s = str(side_raw).lower()
                if "long" in s or "buy" in s:
                    side = "LONG"
                elif "short" in s or "sell" in s:
                    side = "SHORT"
            if side is None:
                side = "SHORT" if size < 0 else "LONG"
            qty = abs(size)
            def _pick(*vals):
                for v in vals:
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isfinite(fv) and fv > 0:
                        return fv
                return None
            entry_px = _pick(ex.get("entryPrice"), ex.get("entry_price"), info.get("entryPrice"), info.get("avgPrice"))
            mark_px = _pick(ex.get("markPrice"), ex.get("lastPrice"), info.get("markPrice"), info.get("lastPrice"))
            if entry_px is None:
                entry_px = mark_px
            liq_px = _pick(ex.get("liqPrice"), ex.get("liq_price"), info.get("liqPrice"))
            notional = _pick(ex.get("notional"), info.get("positionValue"))
            if notional is None and entry_px is not None:
                notional = abs(qty * entry_px)
            if notional is None and mark_px is not None:
                notional = abs(qty * mark_px)
            leverage = _pick(ex.get("leverage"), info.get("leverage")) or float(self.leverage)
            unrealized_pnl, margin, roe_ex = self._extract_exchange_position_risk_metrics(
                ex,
                info,
                notional=notional,
                leverage=leverage,
                mark_px=mark_px,
                entry_px=entry_px,
                qty=qty,
                side=side,
            )
            entry_time = self._extract_entry_time_ms_from_info(info, default=now_ts)
            cap_frac = float(notional / self.balance) if (self.balance and notional) else 0.0
            size_frac = 0.0
            try:
                size_frac = float(notional or 0.0) / max(float(self.balance) * max(float(leverage), 1e-6), 1e-6)
            except Exception:
                size_frac = 0.0

            pos = {
                "symbol": sym,
                "side": side,
                "entry_price": float(entry_px or 0.0),
                "entry_time": int(entry_time),
                "quantity": float(qty),
                "notional": float(notional or 0.0),
                "size_frac": float(size_frac),
                "tag": None,
                "leverage": float(leverage),
                "entry_leverage": float(leverage),
                "entry_notional": float(notional or 0.0),
                "cap_frac": float(cap_frac),
                "fee_paid": 0.0,
                "order_status": "ack",
                "order_ack_ts": int(now_ts),
                "pos_source": "exchange_sync",
                "liq_price": float(liq_px) if liq_px is not None else None,
                "margin": float(margin) if margin is not None else None,
                "unrealized_pnl": float(unrealized_pnl) if unrealized_pnl is not None else None,
                "roe_ex": float(roe_ex) if roe_ex is not None else None,
                "current_price": float(mark_px) if mark_px is not None else None,
                "managed": bool(getattr(config, "MANAGE_SYNCED_POSITIONS", True)),
            }
            # Preserve hold-eval metadata across exchange syncs (avoid wiping t* fields)
            try:
                prior = self.positions.get(sym) or {}
                for key in (
                    "entry_id",
                    "entry_link_id",
                    "entry_leverage",
                    "entry_notional",
                    "opt_hold_entry_sec",
                    "opt_hold_entry_src",
                    "opt_hold_curr_sec",
                    "opt_hold_curr_src",
                    "opt_hold_curr_remaining_sec",
                    "opt_hold_sec",
                    "opt_hold_src",
                    "hold_ev_tstar",
                    "hold_ev_raw_tstar",
                    "hold_eval_ts",
                    "hold_eval_h_pick",
                    "entry_score",
                    "entry_score_hold",
                    "entry_floor",
                    "pred_ev",
                    "pred_event_ev_r",
                    "entry_event_exit_ok",
                    "event_exit_hit",
                    "event_exit_mode",
                    "event_exit_score",
                    "event_cvar_pct",
                    "event_p_tp",
                    "event_p_sl",
                    "event_exit_threshold_score",
                    "event_exit_threshold_psl",
                    "event_exit_threshold_cvar",
                    "event_exit_threshold_ptp",
                    "event_exit_strict_mode",
                    "event_eval_source",
                    "event_eval_ts_ms",
                    "event_precheck_allow_exit_now",
                    "event_precheck_guard_reason",
                    "event_precheck_confirm_mode",
                    "event_precheck_confirm_required",
                    "event_precheck_confirm_count",
                    "event_precheck_confirmed",
                    "event_precheck_shock_score",
                    "event_precheck_severe_adverse",
                    "event_precheck_severe_adverse_ptp_low",
                    "event_precheck_guard_progress",
                    "event_precheck_guard_required",
                    "event_precheck_guard_hold_target_sec",
                    "event_precheck_guard_age_sec",
                    "event_precheck_guard_remaining_sec",
                    "event_exit_guard",
                    "event_exit_guard_progress",
                    "event_exit_guard_required",
                    "event_exit_guard_hold_target_sec",
                    "event_exit_guard_age_sec",
                    "event_exit_guard_remaining_sec",
                    "event_exit_confirm_mode",
                    "event_exit_confirm_required",
                    "event_exit_confirm_count",
                    "event_exit_confirmed",
                    "event_hold_target_sec",
                    "event_hold_remaining_sec",
                    "pred_mu_alpha",
                    "pred_mu_alpha_raw",
                    "pred_mu_dir_conf",
                    "pred_mu_dir_edge",
                    "pred_hmm_state",
                    "pred_hmm_conf",
                    "alpha_vpin",
                    "alpha_hurst",
                    "policy_score_threshold",
                    "policy_event_exit_min_score",
                    "policy_unrealized_dd_floor",
                    "regime",
                    "session",
                ):
                    if key in prior and prior.get(key) is not None:
                        pos[key] = prior.get(key)
                if pos.get("entry_leverage") in (None, 0):
                    pos["entry_leverage"] = float(leverage)
                if pos.get("entry_notional") in (None, 0):
                    pos["entry_notional"] = float(notional or 0.0)
            except Exception:
                pass
            if not pos.get("entry_link_id"):
                try:
                    inferred_link = self._infer_entry_link_id_from_db(sym, side, int(entry_time))
                except Exception:
                    inferred_link = None
                if inferred_link:
                    pos["entry_link_id"] = inferred_link
            if not pos.get("entry_id") and pos.get("entry_link_id"):
                pos["entry_id"] = pos.get("entry_link_id")
            if not pos.get("entry_link_id") and pos.get("entry_id"):
                pos["entry_link_id"] = pos.get("entry_id")
            new_positions[sym] = pos

        if keep_pending and self.positions:
            try:
                grace_sec = float(getattr(config, "LIVE_PENDING_SYNC_GRACE_SEC", 30.0) or 30.0)
            except Exception:
                grace_sec = 30.0
            for sym, pos in list(self.positions.items()):
                if sym in new_positions:
                    continue
                try:
                    status = str(pos.get("order_status") or "")
                    submit_ts = int(pos.get("order_submit_ts") or 0)
                except Exception:
                    status = ""
                    submit_ts = 0
                if status and status != "ack":
                    if submit_ts and (now_ts - submit_ts) < int(grace_sec * 1000):
                        new_positions[sym] = pos

        if overwrite and (not new_positions) and self.positions:
            if not self._allow_empty_position_sync(int(now_ts)):
                self._log("[SYNC_POS] skip empty overwrite (guard)")
                return
        if overwrite:
            self.positions = new_positions
        else:
            self.positions.update(new_positions)
        try:
            self._set_exchange_positions_view(list(new_positions.values()), int(now_ts), "rest")
        except Exception:
            pass
        try:
            for sym in new_positions.keys():
                self._external_close_missing_counts.pop(sym, None)
        except Exception:
            pass
        for sym in new_positions.keys():
            self._last_actions[sym] = "SYNC"
            self._last_open_ts[sym] = int(now_ts)
        try:
            self._archive_outside_universe_positions(source=f"exchange_sync:{reason}")
        except Exception:
            pass
        try:
            if hasattr(self, "db") and self.db is not None and overwrite:
                try:
                    self.db.delete_positions_by_mode(self._trading_mode)
                except Exception:
                    pass
                for sym, pos in self.positions.items():
                    self.db.save_position_background(sym, pos, mode=self._trading_mode)
        except Exception:
            pass
        try:
            self._persist_state(force=True)
        except Exception:
            pass
        self._last_positions_sync_ms = int(now_ts)
        self._log(f"[SYNC_POS] exchange positions synced ({reason}) count={len(new_positions)} overwrite={overwrite}")

    def _extract_usdt_wallet_equity_from_balance(self, bal):
        wallet = None
        equity = None
        free = None
        try:
            if isinstance(bal, dict):
                usdt = bal.get("USDT")
                if isinstance(usdt, dict):
                    try:
                        free = float(usdt.get("free")) if usdt.get("free") is not None else None
                    except Exception:
                        free = None
                    try:
                        equity = float(usdt.get("total")) if usdt.get("total") is not None else None
                    except Exception:
                        equity = None
                info = bal.get("info")
                if isinstance(info, dict):
                    res = info.get("result")
                    if isinstance(res, dict):
                        lst = res.get("list")
                        if isinstance(lst, list) and lst:
                            item = lst[0]
                            if isinstance(item, dict):
                                try:
                                    if item.get("totalWalletBalance") is not None:
                                        wallet = float(item.get("totalWalletBalance"))
                                except Exception:
                                    pass
                                try:
                                    if item.get("totalEquity") is not None:
                                        equity = float(item.get("totalEquity"))
                                except Exception:
                                    pass
                                try:
                                    if item.get("totalAvailableBalance") is not None:
                                        free = float(item.get("totalAvailableBalance"))
                                    elif item.get("availableBalance") is not None:
                                        free = float(item.get("availableBalance"))
                                except Exception:
                                    pass
        except Exception:
            pass
        return wallet, equity, free

    def _extract_margin_metrics_from_balance(self, bal):
        im = None
        mm = None
        try:
            if isinstance(bal, dict):
                info = bal.get("info")
                if isinstance(info, dict):
                    res = info.get("result")
                    if isinstance(res, dict):
                        lst = res.get("list")
                        if isinstance(lst, list) and lst:
                            item = lst[0]
                            if isinstance(item, dict):
                                try:
                                    if item.get("totalInitialMargin") is not None:
                                        im = float(item.get("totalInitialMargin"))
                                except Exception:
                                    im = None
                                try:
                                    if item.get("totalMaintenanceMargin") is not None:
                                        mm = float(item.get("totalMaintenanceMargin"))
                                except Exception:
                                    mm = None
        except Exception:
            pass
        return im, mm

    async def fetch_balance_loop(self):
        interval = float(LIVE_BALANCE_SYNC_SEC) if LIVE_BALANCE_SYNC_SEC else 5.0
        interval = max(1.0, interval)
        while True:
            if not self.enable_orders:
                await asyncio.sleep(interval)
                continue
            if not getattr(self.exchange, "apiKey", None) or not getattr(self.exchange, "secret", None):
                if not self._live_balance_key_warned:
                    self._log("[LIVE_BAL] apiKey/secret missing; skipping live balance sync")
                    self._live_balance_key_warned = True
                await asyncio.sleep(interval)
                continue
            self._live_balance_key_warned = False
            try:
                bal = await self._ccxt_call("fetch_balance(live)", self.exchange.fetch_balance)
                wallet, equity, free = self._extract_usdt_wallet_equity_from_balance(bal)
                im, mm = self._extract_margin_metrics_from_balance(bal)
                equity_val = equity if equity is not None else wallet
                if wallet is not None:
                    self.balance = float(wallet)
                elif equity_val is not None:
                    self.balance = float(equity_val)
                self._live_wallet_balance = wallet
                self._live_equity = equity_val
                self._live_free_balance = free
                self._live_total_initial_margin = im
                self._live_total_maintenance_margin = mm
                ts_ms = now_ms()
                self._last_live_sync_ms = ts_ms
                self._last_live_sync_err = None
                try:
                    self.risk_manager.update_account_summary(
                        wallet_balance=wallet,
                        total_equity=equity,
                        free_balance=free,
                        total_initial_margin=im,
                        total_maintenance_margin=mm,
                    )
                except Exception:
                    pass
                try:
                    eq_val = equity_val if equity_val is not None else (wallet if wallet is not None else self.balance)
                    eq_val = float(eq_val)
                    if math.isfinite(eq_val):
                        self._live_equity_history.append({"time": ts_ms, "equity": eq_val})
                except Exception:
                    pass
            except Exception as e:
                self._last_live_sync_err = str(e)
                self._log_err(f"[ERR] fetch_balance_loop: {e}")
                self._note_runtime_error("fetch_balance_loop", str(e))
            await asyncio.sleep(interval)

    async def fetch_positions_loop(self):
        if not self.enable_orders:
            while True:
                await asyncio.sleep(float(getattr(config, "LIVE_LIQUIDATION_SYNC_SEC", 10.0) or 10.0))
        while True:
            try:
                interval = float(getattr(config, "LIVE_LIQUIDATION_SYNC_SEC", 10.0) or 10.0)
            except Exception:
                interval = 10.0
            try:
                fetch_params = self._positions_fetch_params()
                try:
                    if fetch_params:
                        raw = await self._ccxt_call("fetch_positions", self.exchange.fetch_positions, SYMBOLS, fetch_params)
                    else:
                        raw = await self._ccxt_call("fetch_positions", self.exchange.fetch_positions, SYMBOLS)
                except Exception as e:
                    msg = str(e)
                    # Bybit: fetchPositions does not accept an array with more than one symbol.
                    if "fetchPositions() does not accept an array" in msg or "does not accept an array" in msg:
                        if fetch_params:
                            raw = await self._ccxt_call("fetch_positions", self.exchange.fetch_positions, None, fetch_params)
                        else:
                            raw = await self._ccxt_call("fetch_positions", self.exchange.fetch_positions)
                    else:
                        raise
                pos_map = self._normalize_exchange_positions(raw if isinstance(raw, list) else [])
                now_ts = now_ms()
                try:
                    self._set_exchange_positions_view(
                        self._build_exchange_positions_view(pos_map, now_ts),
                        int(now_ts),
                        "rest",
                    )
                except Exception:
                    pass
                if pos_map:
                    self._empty_positions_fetches = 0
                    self._last_nonempty_pos_fetch_ms = int(now_ts)
                elif self.positions:
                    self._empty_positions_fetches += 1
                    if not self._allow_empty_position_sync(int(now_ts)):
                        await asyncio.sleep(interval)
                        continue
                # Record external/manual closes before sync overwrites local state.
                try:
                    for sym, pos in list(self.positions.items()):
                        try:
                            qty = float(pos.get("quantity", 0.0))
                        except Exception:
                            qty = 0.0
                        if qty == 0.0:
                            continue
                        try:
                            status = str(pos.get("order_status") or "")
                            submit_ts = int(pos.get("order_submit_ts") or 0)
                        except Exception:
                            status = ""
                            submit_ts = 0
                        if status and status != "ack":
                            # skip pending entries
                            if submit_ts and (now_ts - submit_ts) < 30_000:
                                continue
                        ex = pos_map.get(sym)
                        ex_size = 0.0
                        if ex is not None:
                            try:
                                ex_size = float(ex.get("size_f", 0.0))
                            except Exception:
                                ex_size = 0.0
                        if ex is not None and abs(ex_size) >= 1e-9:
                            try:
                                self._external_close_missing_counts.pop(sym, None)
                            except Exception:
                                pass
                            continue
                        if ex is None or abs(ex_size) < 1e-9:
                            try:
                                confirm_cnt = int(os.environ.get("EXTERNAL_CLOSE_CONFIRM_COUNT", 2) or 2)
                            except Exception:
                                confirm_cnt = 2
                            confirm_cnt = max(1, confirm_cnt)
                            miss_cnt = int(self._external_close_missing_counts.get(sym, 0) or 0) + 1
                            self._external_close_missing_counts[sym] = miss_cnt
                            if miss_cnt < confirm_cnt:
                                continue
                            try:
                                entry_ts = int(pos.get("entry_time") or 0)
                            except Exception:
                                entry_ts = 0
                            try:
                                grace_sec = float(os.environ.get("EXTERNAL_CLOSE_GRACE_SEC", 10.0) or 10.0)
                            except Exception:
                                grace_sec = 10.0
                            if entry_ts and (now_ts - entry_ts) < int(grace_sec * 1000):
                                continue
                            px = self.market.get(sym, {}).get("price") or pos.get("current_price") or pos.get("entry_price")
                            if px is None:
                                continue
                            try:
                                ext_cause, ext_reason, ext_detail = self._classify_external_close_reason(
                                    sym,
                                    pos,
                                    miss_cnt=int(miss_cnt),
                                    source="fetch_positions_loop",
                                    now_ts=int(now_ts),
                                )
                            except Exception:
                                ext_cause, ext_reason, ext_detail = ("manual_cleanup", "exchange_close_manual_cleanup", {"miss_cnt": int(miss_cnt)})
                            self._record_external_close(
                                sym,
                                float(px),
                                reason=ext_reason,
                                cause=ext_cause,
                                source="fetch_positions_loop",
                                detail=ext_detail,
                            )
                            try:
                                self._external_close_missing_counts.pop(sym, None)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    if bool(getattr(config, "SYNC_POSITIONS_POLL", True)):
                        poll_sec = float(getattr(config, "SYNC_POSITIONS_POLL_SEC", 30.0) or 30.0)
                        last_sync = int(self._last_positions_sync_ms or 0)
                        current_ms = now_ms()
                        if (not last_sync) or (current_ms - last_sync) >= int(poll_sec * 1000):
                            try:
                                overwrite_poll = str(os.environ.get("SYNC_POSITIONS_OVERWRITE", "0")).strip().lower() in ("1", "true", "yes", "on")
                            except Exception:
                                overwrite_poll = False
                            try:
                                keep_pending = str(os.environ.get("SYNC_POSITIONS_KEEP_PENDING", "1")).strip().lower() in ("1", "true", "yes", "on")
                            except Exception:
                                keep_pending = True
                            include_all = bool(getattr(config, "SYNC_POSITIONS_ALL", True))
                            await self._sync_positions_from_exchange(
                                overwrite=overwrite_poll,
                                reason="poll",
                                raw_positions=None if include_all else (raw if isinstance(raw, list) else None),
                                keep_pending=keep_pending,
                                include_all=include_all,
                            )
                except Exception:
                    pass
                now_ts = now_ms()
                # detect unexpected exchange-side closes (possible liquidation)
                for sym, pos in list(self.positions.items()):
                    try:
                        qty = float(pos.get("quantity", 0.0))
                    except Exception:
                        qty = 0.0
                    if qty == 0.0:
                        continue
                    # Skip liquidation check for pending entry orders (not yet acknowledged)
                    try:
                        status = str(pos.get("order_status") or "")
                        submit_ts = int(pos.get("order_submit_ts") or 0)
                    except Exception:
                        status = ""
                        submit_ts = 0
                    if status and status != "ack":
                        if submit_ts and (now_ts - submit_ts) < 30_000:
                            continue
                    ex = pos_map.get(sym)
                    ex_size = 0.0
                    if ex is not None:
                        try:
                            ex_size = float(ex.get("size_f", 0.0))
                        except Exception:
                            ex_size = 0.0
                    # reset missing counter if exchange position exists
                    if ex is not None and abs(ex_size) > 1e-9:
                        self._live_missing_pos_counts[sym] = 0
                        continue
                    # if exchange reports no position but we think it's open
                    if ex is None or abs(ex_size) < 1e-9:
                        last_close = int(self._last_close_ts.get(sym, 0) or 0)
                        # ignore if we just closed locally
                        if last_close and (now_ts - last_close) < 10_000:
                            continue
                        # grace period after entry (avoid false liquidation)
                        try:
                            entry_ts = int(pos.get("entry_time") or 0)
                        except Exception:
                            entry_ts = 0
                        try:
                            grace_sec = float(os.environ.get("LIVE_LIQUIDATION_GRACE_SEC", 15.0) or 15.0)
                        except Exception:
                            grace_sec = 15.0
                        if entry_ts and (now_ts - entry_ts) < int(grace_sec * 1000):
                            continue
                        # require multiple consecutive misses before marking liquidation
                        try:
                            miss_need = int(os.environ.get("LIVE_LIQUIDATION_MISS_COUNT", 2) or 2)
                        except Exception:
                            miss_need = 2
                        miss_need = max(1, miss_need)
                        miss_cnt = int(self._live_missing_pos_counts.get(sym, 0) or 0) + 1
                        self._live_missing_pos_counts[sym] = miss_cnt
                        if miss_cnt < miss_need:
                            continue
                        px = self.market.get(sym, {}).get("price")
                        self._log_err(f"[LIVE_LIQ] {sym} exchange position missing (size=0). Marking as liquidation.")
                        self._register_anomaly("exchange_liquidation", "critical", f"{sym} exchange position closed (liq?)")
                        try:
                            liq_trigger_safety = str(os.environ.get("LIVE_LIQUIDATION_TRIGGER_SAFETY", "1")).strip().lower() in ("1", "true", "yes", "on")
                        except Exception:
                            liq_trigger_safety = True
                        if liq_trigger_safety and (not bool(getattr(self, "safety_mode", False))):
                            self.safety_mode = True
                            self._register_anomaly("safety_mode", "critical", f"exchange liquidation -> safety mode ({sym})")
                            self._log_err(f"[RISK] safety_mode ON due to exchange liquidation: {sym}")
                        if px is None:
                            try:
                                px = float(pos.get("entry_price") or 0.0)
                            except Exception:
                                px = None
                        if px is None:
                            self.positions.pop(sym, None)
                        else:
                            self._close_position(sym, float(px), "exchange_liquidation", exit_kind="KILL", skip_order=True)
            except Exception as e:
                self._log_err(f"[ERR] fetch_positions_loop: {e}")
                self._note_runtime_error("fetch_positions_loop", str(e))
            await asyncio.sleep(interval)

    async def refresh_top_volume_universe_loop(self):
        while True:
            try:
                enabled = str(os.environ.get("DYNAMIC_UNIVERSE_TOP_VOLUME", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                enabled = False
            if not enabled:
                await asyncio.sleep(5.0)
                continue
            try:
                refresh_sec = float(os.environ.get("TOP_VOLUME_REFRESH_SEC", 1800.0) or 1800.0)
            except Exception:
                refresh_sec = 1800.0
            refresh_sec = max(60.0, refresh_sec)
            try:
                top_n = int(os.environ.get("TOP_VOLUME_COUNT", 30) or 30)
            except Exception:
                top_n = 30
            top_n = max(1, top_n)
            try:
                suffix = str(os.environ.get("TOP_VOLUME_SYMBOL_SUFFIX", ":USDT") or ":USDT").strip()
            except Exception:
                suffix = ":USDT"
            try:
                min_quote = float(os.environ.get("TOP_VOLUME_MIN_QUOTE", 0.0) or 0.0)
            except Exception:
                min_quote = 0.0

            try:
                tickers = await self._ccxt_call("fetch_tickers(universe)", self.data_exchange.fetch_tickers)
                if not isinstance(tickers, dict):
                    tickers = {}
                candidates = []
                for sym, t in tickers.items():
                    if not sym:
                        continue
                    if suffix and (suffix not in sym):
                        continue
                    vol = self._extract_ticker_volume(t)
                    if vol is None:
                        continue
                    if min_quote > 0 and float(vol) < float(min_quote):
                        continue
                    candidates.append((float(vol), str(sym)))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    new_syms = [s for _, s in candidates[:top_n]]
                    # If a symbol drops from top volume but still has an open position,
                    # delay replacement until that position is closed.
                    try:
                        active_syms = []
                        for sym, pos in (self.positions or {}).items():
                            try:
                                qty = float(pos.get("quantity", 0.0) or 0.0)
                            except Exception:
                                qty = 0.0
                            if abs(qty) > 1e-9:
                                active_syms.append(sym)
                        if active_syms:
                            outside_active = [s for s in active_syms if s not in new_syms]
                            if outside_active:
                                keep_n = max(0, len(new_syms) - len(outside_active))
                                if keep_n < len(new_syms):
                                    new_syms = new_syms[:keep_n]
                                for s in outside_active:
                                    if s not in new_syms:
                                        new_syms.append(s)
                    except Exception:
                        pass
                    self._apply_dynamic_universe(new_syms, reason="top_volume")
            except Exception as e:
                self._log_err(f"[UNIVERSE] top_volume update failed: {e}")

            await asyncio.sleep(refresh_sec)

    async def hold_eval_loop(self):
        """Background hold-eval refresh for open positions (async, eval-only)."""
        while True:
            try:
                enabled = str(os.environ.get("HOLD_EVAL_ASYNC_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                enabled = True
            if not enabled:
                await asyncio.sleep(1.0)
                continue
            try:
                interval = float(os.environ.get("HOLD_EVAL_ASYNC_INTERVAL_SEC", 0) or 0.0)
            except Exception:
                interval = 0.0
            if interval <= 0:
                try:
                    interval = float(os.environ.get("HOLD_EVAL_INTERVAL_SEC", 30.0) or 30.0)
                except Exception:
                    interval = 30.0
            try:
                positions = list((self.positions or {}).items())
            except Exception:
                positions = []
            if not positions:
                await asyncio.sleep(max(1.0, interval))
                continue
            ts = now_ms()
            for sym, pos in positions:
                try:
                    if not pos:
                        continue
                    ctx = None
                    try:
                        ctx = self._build_decision_context(sym, ts)
                    except Exception:
                        ctx = None
                    if not ctx:
                        continue
                    price = ctx.get("price") or self.market.get(sym, {}).get("price") or pos.get("entry_price")
                    if price is None:
                        continue
                    # Async background loop should never submit live orders from a worker thread.
                    # Keep it observation-only; execution happens in the main decision loop.
                    eval_only = True
                    await asyncio.to_thread(
                        self._check_hold_vs_exit,
                        sym,
                        pos,
                        {},
                        ctx,
                        ts,
                        float(price),
                        eval_only=eval_only,
                    )
                except Exception as e:
                    self._log_err(f"[HOLD_EVAL_ASYNC] {sym} error: {e}")
            await asyncio.sleep(max(1.0, interval))

    def _mark_exit_and_cooldown(self, sym: str, exit_kind: str, ts_ms: int):
        """
        exit_kind: "TP" | "TIMEOUT" | "SL" | "KILL" | "MANUAL"
        """
        k = (exit_kind or "MANUAL").upper()
        self._last_exit_kind[sym] = k
        # cooldown rule disabled
        self._cooldown_until[sym] = 0
        self._streak[sym] = 0

    def _archive_outside_universe_positions(self, *, universe: set[str] | None = None, source: str = "runtime") -> int:
        """Archive and remove positions that are not in the given universe (or default SYMBOLS)."""
        uni = universe if universe is not None else set(SYMBOLS)
        outside = [s for s in self.positions.keys() if s not in uni]
        if not outside:
            return 0
        try:
            dedupe_sec = float(os.environ.get("OUTSIDE_UNIVERSE_ARCHIVE_DEDUPE_SEC", 180.0) or 180.0)
        except Exception:
            dedupe_sec = 180.0
        dedupe_ms = int(max(0.0, dedupe_sec) * 1000.0)
        archived = 0
        archived_syms: list[str] = []
        for sym in outside:
            pos = self.positions.get(sym) or {}
            qty_now = self._safe_float(pos.get("quantity") or pos.get("size"), None)
            px_now = self._safe_float(pos.get("entry_price"), None)
            side_now = str(pos.get("side") or "")
            ts_now = now_ms()
            prev = self._outside_universe_positions.get(sym) or {}
            prev_ts = int(prev.get("archive_ts") or 0)
            prev_qty = self._safe_float(prev.get("quantity"), None)
            prev_px = self._safe_float(prev.get("entry_price"), None)
            prev_side = str(prev.get("side") or "")
            same_qty = (
                qty_now is not None
                and prev_qty is not None
                and abs(float(qty_now) - float(prev_qty)) <= max(1e-9, abs(float(qty_now)) * 1e-6)
            )
            same_px = (
                px_now is not None
                and prev_px is not None
                and abs(float(px_now) - float(prev_px)) <= max(1e-9, abs(float(px_now)) * 1e-6)
            )
            if dedupe_ms > 0 and prev_ts > 0 and (ts_now - prev_ts) < dedupe_ms and prev_side == side_now and same_qty and same_px:
                try:
                    self.positions.pop(sym, None)
                except Exception:
                    pass
                continue
            snap = dict(pos)
            snap["archive_reason"] = "outside_universe"
            snap["archive_source"] = source
            snap["archive_ts"] = ts_now
            # Track for dashboard warning
            self._outside_universe_positions[sym] = {
                "symbol": sym,
                "side": snap.get("side"),
                "quantity": snap.get("quantity") or snap.get("size"),
                "entry_price": snap.get("entry_price"),
                "notional": snap.get("notional"),
                "archive_ts": snap.get("archive_ts"),
                "archive_source": source,
            }
            # Persist archive event
            try:
                if self.db is not None:
                    self.db.log_position_event(sym, "archive_outside_universe", snap, mode=self._trading_mode)
                    self.db.delete_position(sym)
            except Exception as e:
                self._log_err(f"[ERR] archive_outside_universe DB: {sym} err={e}")
            # Remove from in-memory positions
            try:
                self.positions.pop(sym, None)
            except Exception:
                pass
            self._last_close_ts[sym] = now_ms()
            self._register_anomaly("outside_universe_position", "warning", f"{sym} archived (outside universe)")
            archived += 1
            archived_syms.append(sym)
        if archived_syms:
            self._log(f"[ARCHIVE] outside universe positions: {archived_syms}")
        return archived

    def _compute_momentum_z(self, closes: list[float]) -> float:
        if not closes or len(closes) < 6:
            return 0.0
        try:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            if rets.size < 5:
                return 0.0
            window = int(min(20, rets.size))
            subset = rets[-window:]
            mean_r = float(subset.mean())
            std_r = float(subset.std())
            if std_r <= 1e-9:
                return 0.0
            return float((subset[-1] - mean_r) / std_r)
        except Exception:
            return 0.0

    def _compute_impulse_features(
        self,
        closes: list[float],
        opens: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> dict:
        try:
            lookback = int(os.environ.get("IMPULSE_LOOKBACK", 20) or 20)
        except Exception:
            lookback = 20
        lookback = max(5, lookback)
        try:
            body_thr = float(os.environ.get("IMPULSE_BODY_PCT", 0.003) or 0.003)
        except Exception:
            body_thr = 0.003
        try:
            vol_mult = float(os.environ.get("IMPULSE_VOL_MULT", 2.0) or 2.0)
        except Exception:
            vol_mult = 2.0
        try:
            score_min = float(os.environ.get("IMPULSE_SCORE_MIN", 0.66) or 0.66)
        except Exception:
            score_min = 0.66

        if not closes or not opens or not highs or not lows or not volumes:
            return {
                "momentum_z": 0.0,
                "impulse_score": 0.0,
                "impulse_dir": 0,
                "impulse_active": False,
                "impulse_body_pct": None,
                "impulse_vol_ratio": None,
            }

        n = min(len(closes), len(opens), len(highs), len(lows), len(volumes))
        if n < lookback + 2:
            return {
                "momentum_z": self._compute_momentum_z(closes),
                "impulse_score": 0.0,
                "impulse_dir": 0,
                "impulse_active": False,
                "impulse_body_pct": None,
                "impulse_vol_ratio": None,
            }

        closes = closes[-n:]
        opens = opens[-n:]
        highs = highs[-n:]
        lows = lows[-n:]
        volumes = volumes[-n:]

        curr_close = float(closes[-1])
        curr_open = float(opens[-1])
        curr_vol = float(volumes[-1])
        prev_high = float(max(highs[-(lookback + 1):-1]))
        prev_low = float(min(lows[-(lookback + 1):-1]))

        body_pct = 0.0
        if curr_open > 0:
            body_pct = float((curr_close - curr_open) / curr_open)

        vol_mean = float(np.mean(volumes[-(lookback + 1):-1])) if volumes[-(lookback + 1):-1] else 0.0
        vol_ratio = float(curr_vol / vol_mean) if vol_mean > 0 else 0.0

        breakout_long = curr_close > prev_high
        breakout_short = curr_close < prev_low
        body_long = body_pct >= body_thr
        body_short = body_pct <= -body_thr
        vol_ok = vol_ratio >= vol_mult if vol_mult > 0 else True

        breakout_score = 1.0 if (breakout_long or breakout_short) else 0.0
        body_score = 0.0
        if body_thr > 0:
            body_score = min(1.0, abs(body_pct) / body_thr)
        vol_score = 0.0
        if vol_mult > 0:
            vol_score = min(1.0, vol_ratio / vol_mult) if vol_ratio > 0 else 0.0
        impulse_score = float((breakout_score + body_score + vol_score) / 3.0)

        impulse_dir = 0
        if breakout_long and body_long and vol_ok:
            impulse_dir = 1
        elif breakout_short and body_short and vol_ok:
            impulse_dir = -1
        impulse_active = bool(impulse_dir != 0 and impulse_score >= score_min)

        return {
            "momentum_z": self._compute_momentum_z(closes),
            "impulse_score": float(impulse_score),
            "impulse_dir": int(impulse_dir),
            "impulse_active": bool(impulse_active),
            "impulse_body_pct": float(body_pct),
            "impulse_vol_ratio": float(vol_ratio),
        }

    def _compute_realtime_breakout_features(
        self,
        price: float,
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> dict:
        try:
            enabled = str(os.environ.get("REALTIME_BREAKOUT_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            enabled = True
        if not enabled or price is None or price <= 0:
            return {
                "rt_breakout_active": False,
                "rt_breakout_dir": 0,
                "rt_breakout_score": 0.0,
                "rt_breakout_strength": None,
                "rt_breakout_level": None,
                "rt_breakout_vol_ratio": None,
            }
        try:
            lookback = int(os.environ.get("REALTIME_BREAKOUT_LOOKBACK", 20) or 20)
        except Exception:
            lookback = 20
        lookback = max(5, lookback)
        try:
            buffer_pct = float(os.environ.get("REALTIME_BREAKOUT_BUFFER_PCT", 0.0005) or 0.0005)
        except Exception:
            buffer_pct = 0.0005
        try:
            score_min = float(os.environ.get("REALTIME_BREAKOUT_SCORE_MIN", 0.66) or 0.66)
        except Exception:
            score_min = 0.66
        try:
            vol_mult = float(os.environ.get("REALTIME_BREAKOUT_VOL_MULT", 1.5) or 1.5)
        except Exception:
            vol_mult = 1.5

        if not highs or not lows:
            return {
                "rt_breakout_active": False,
                "rt_breakout_dir": 0,
                "rt_breakout_score": 0.0,
                "rt_breakout_strength": None,
                "rt_breakout_level": None,
                "rt_breakout_vol_ratio": None,
            }
        n = min(len(highs), len(lows), len(volumes) if volumes else len(highs))
        if n < lookback + 2:
            return {
                "rt_breakout_active": False,
                "rt_breakout_dir": 0,
                "rt_breakout_score": 0.0,
                "rt_breakout_strength": None,
                "rt_breakout_level": None,
                "rt_breakout_vol_ratio": None,
            }
        highs = highs[-n:]
        lows = lows[-n:]
        vols = volumes[-n:] if volumes else [0.0] * n

        prev_high = float(max(highs[-(lookback + 1):-1]))
        prev_low = float(min(lows[-(lookback + 1):-1]))
        level_long = float(prev_high * (1.0 + buffer_pct))
        level_short = float(prev_low * (1.0 - buffer_pct))

        breakout_long = price >= level_long
        breakout_short = price <= level_short
        dir_val = 1 if breakout_long else (-1 if breakout_short else 0)

        strength = None
        if dir_val == 1 and level_long > 0:
            strength = float((price - level_long) / level_long)
        elif dir_val == -1 and level_short > 0:
            strength = float((level_short - price) / level_short)

        vol_ratio = None
        try:
            if vols and len(vols) > lookback:
                vol_mean = float(np.mean(vols[-(lookback + 1):-1]))
                vol_last = float(vols[-1])
                if vol_mean > 0:
                    vol_ratio = float(vol_last / vol_mean)
        except Exception:
            vol_ratio = None

        vol_ok = True
        if vol_mult > 0 and vol_ratio is not None:
            vol_ok = vol_ratio >= vol_mult

        breakout_score = 1.0 if dir_val != 0 else 0.0
        strength_score = 0.0
        if strength is not None:
            strength_score = min(1.0, abs(float(strength)) / max(1e-9, float(buffer_pct) * 2.0))
        vol_score = 0.0
        if vol_ratio is not None and vol_mult > 0:
            vol_score = min(1.0, float(vol_ratio) / float(vol_mult))
        rt_score = float((breakout_score + strength_score + vol_score) / 3.0)

        active = bool(dir_val != 0 and rt_score >= score_min and vol_ok)
        level = level_long if dir_val == 1 else (level_short if dir_val == -1 else None)

        return {
            "rt_breakout_active": bool(active),
            "rt_breakout_dir": int(dir_val),
            "rt_breakout_score": float(rt_score),
            "rt_breakout_strength": float(strength) if strength is not None else None,
            "rt_breakout_level": float(level) if level is not None else None,
            "rt_breakout_vol_ratio": float(vol_ratio) if vol_ratio is not None else None,
        }

    def _append_tick_price(self, sym: str, price: float, ts_ms: int) -> None:
        buf = self._tick_buffer.get(sym)
        if buf is None:
            return
        try:
            px = float(price)
        except Exception:
            return
        ts_val = int(ts_ms)
        buf.append((ts_val, px))
        try:
            cutoff = ts_val - int(self._tick_buffer_sec * 1000.0)
            while buf and buf[0][0] < cutoff:
                buf.popleft()
        except Exception:
            pass

    def _compute_tick_features(self, sym: str, ts_ms: int | None = None) -> dict:
        default = {
            "tick_ret": 0.0,
            "tick_vol": 0.0,
            "tick_mom": 0.0,
            "tick_trend": 0.0,
            "tick_dir": 0,
            "tick_samples": 0,
            "tick_window_sec": float(TICK_LOOKBACK_SEC),
            "tick_breakout_active": False,
            "tick_breakout_dir": 0,
            "tick_breakout_score": 0.0,
            "tick_breakout_level": None,
        }
        buf = self._tick_buffer.get(sym)
        if not buf or len(buf) < 2:
            return default
        try:
            now_ms = int(ts_ms or buf[-1][0])
        except Exception:
            now_ms = int(buf[-1][0])

        cache = self._tick_feature_cache.get(sym)
        if cache and cache[0] == now_ms:
            return cache[1]

        try:
            lookback_sec = float(os.environ.get("TICK_LOOKBACK_SEC", TICK_LOOKBACK_SEC) or TICK_LOOKBACK_SEC)
        except Exception:
            lookback_sec = float(TICK_LOOKBACK_SEC)
        lookback_sec = max(1.0, lookback_sec)
        cutoff = now_ms - int(lookback_sec * 1000.0)
        vals = [p for t, p in buf if t >= cutoff]
        if len(vals) < int(max(2, TICK_MIN_SAMPLES)):
            out = dict(default)
            out["tick_samples"] = len(vals)
            out["tick_window_sec"] = float(lookback_sec)
            self._tick_feature_cache[sym] = (now_ms, out)
            return out

        try:
            arr = np.asarray(vals, dtype=np.float64)
            rets = np.diff(np.log(np.maximum(arr, 1e-12)))
            tick_ret = float(math.log(arr[-1] / max(arr[0], 1e-12)))
            tick_vol = float(rets.std()) if rets.size else 0.0
            tick_mom = float(rets.mean()) if rets.size else 0.0
        except Exception:
            tick_ret = 0.0
            tick_vol = 0.0
            tick_mom = 0.0
        tick_trend = 0.0
        if tick_vol > 0:
            tick_trend = float(tick_ret / max(tick_vol, 1e-9))
        tick_trend = float(max(-8.0, min(8.0, tick_trend)))
        tick_dir = 1 if tick_trend > 0 else (-1 if tick_trend < 0 else 0)

        # Tick breakout: recent high/low vs last price
        try:
            breakout_lb = float(os.environ.get("TICK_BREAKOUT_LOOKBACK_SEC", TICK_BREAKOUT_LOOKBACK_SEC) or TICK_BREAKOUT_LOOKBACK_SEC)
        except Exception:
            breakout_lb = float(TICK_BREAKOUT_LOOKBACK_SEC)
        breakout_lb = max(1.0, breakout_lb)
        cutoff_b = now_ms - int(breakout_lb * 1000.0)
        vals_b = [p for t, p in buf if t >= cutoff_b]
        tick_breakout_active = False
        tick_breakout_dir = 0
        tick_breakout_score = 0.0
        tick_breakout_level = None
        try:
            buffer_pct = float(os.environ.get("TICK_BREAKOUT_BUFFER_PCT", TICK_BREAKOUT_BUFFER_PCT) or TICK_BREAKOUT_BUFFER_PCT)
        except Exception:
            buffer_pct = float(TICK_BREAKOUT_BUFFER_PCT)
        if len(vals_b) >= 3:
            last = float(vals_b[-1])
            prev = vals_b[:-1]
            hi = max(prev)
            lo = min(prev)
            if hi > 0 and last > hi * (1.0 + buffer_pct):
                tick_breakout_active = True
                tick_breakout_dir = 1
                tick_breakout_level = float(hi)
                tick_breakout_score = float((last - hi) / max(hi, 1e-12))
            elif lo > 0 and last < lo * (1.0 - buffer_pct):
                tick_breakout_active = True
                tick_breakout_dir = -1
                tick_breakout_level = float(lo)
                tick_breakout_score = float((lo - last) / max(lo, 1e-12))

        out = {
            "tick_ret": float(tick_ret),
            "tick_vol": float(tick_vol),
            "tick_mom": float(tick_mom),
            "tick_trend": float(tick_trend),
            "tick_dir": int(tick_dir),
            "tick_samples": int(len(vals)),
            "tick_window_sec": float(lookback_sec),
            "tick_breakout_active": bool(tick_breakout_active),
            "tick_breakout_dir": int(tick_breakout_dir),
            "tick_breakout_score": float(tick_breakout_score),
            "tick_breakout_level": float(tick_breakout_level) if tick_breakout_level is not None else None,
        }
        self._tick_feature_cache[sym] = (now_ms, out)
        return out

    def _read_symbols_csv_from_file(self, path: Path) -> list[str]:
        try:
            if not path.exists():
                return []
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return []
        val = ""
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("SYMBOLS_CSV="):
                val = s.split("=", 1)[1].strip()
        if not val:
            return []
        parts = [p.strip() for p in val.split(",")]
        return [p for p in parts if p]

    def _refresh_runtime_universe(self) -> None:
        now = now_ms()
        if (now - int(self._last_universe_refresh_ms or 0)) < int(self._universe_refresh_sec * 1000):
            return
        self._last_universe_refresh_ms = now

        # Prefer runtime overrides from state/bybit.env when running with bybit profile
        candidates = []
        use_bybit_env = False
        try:
            if str(getattr(config, "ENV_PROFILE", "")).strip().lower() == "bybit":
                use_bybit_env = True
            elif str(getattr(config, "ENV_ACTIVE", "")).endswith(".env.bybit"):
                use_bybit_env = True
        except Exception:
            use_bybit_env = False
        if use_bybit_env:
            bybit_env = self.state_dir / "bybit.env"
            candidates.append(bybit_env)
        if getattr(config, "ENV_ACTIVE", None):
            try:
                candidates.append(Path(str(config.ENV_ACTIVE)))
            except Exception:
                pass

        new_syms: list[str] = []
        for path in candidates:
            new_syms = self._read_symbols_csv_from_file(path)
            if new_syms:
                break
        if not new_syms:
            return

        new_set = set(new_syms)
        if not new_set or new_set == self._runtime_universe:
            return

        self._runtime_universe = new_set
        self._log(f"[UNIVERSE] runtime update: {len(new_set)} symbols")
        self._register_anomaly("universe_change", "info", f"runtime universe updated: {len(new_set)} symbols")
        self._archive_outside_universe_positions(universe=new_set, source="universe_change")

    def _compute_portfolio(self):
        unreal = 0.0
        total_notional = 0.0
        pos_list = []
        use_exchange = False
        try:
            use_exchange = bool(self.enable_orders) and str(os.environ.get("DASHBOARD_USE_EXCHANGE_POSITIONS", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_exchange = False
        if use_exchange and self._exchange_positions_ts is not None:
            for pos in list(self._exchange_positions_view or []):
                sym = pos.get("symbol")
                px = pos.get("current")
                if px is None:
                    px = pos.get("current_price")
                if px is None:
                    try:
                        px = self.market.get(sym, {}).get("price")
                    except Exception:
                        px = None
                if px is None:
                    try:
                        px = float(pos.get("entry_price") or 0.0)
                    except Exception:
                        px = None
                if px is None:
                    continue
                entry = float(pos.get("entry_price", px))
                side = str(pos.get("side") or "")
                qty = float(pos.get("quantity", 0.0))
                notional = float(pos.get("notional", 0.0))
                pnl_ex = self._safe_float(pos.get("unrealized_pnl"), None)
                pnl = (
                    ((px - entry) * qty) if side == "LONG" else ((entry - px) * qty)
                ) if pnl_ex is None else float(pnl_ex)
                unreal += pnl
                total_notional += notional
                lev = float(pos.get("leverage", self.leverage))
                margin = self._safe_float(pos.get("margin"), None)
                if margin is None or margin <= 0:
                    base_notional = notional / max(lev, 1e-6)
                else:
                    base_notional = float(margin)
                roe_ex = self._safe_float(pos.get("roe"), None)
                roe_val = roe_ex if roe_ex is not None else (float(pnl / base_notional) if base_notional else 0.0)
                try:
                    entry_t = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
                    age_sec = max(0.0, (now_ms() - int(entry_t)) / 1000.0)
                except Exception:
                    entry_t = now_ms()
                    age_sec = 0.0
                pos_list.append({
                    "symbol": sym,
                    "side": side,
                    "entry_price": entry,
                    "entry_time": int(entry_t),
                    "current": float(px),
                    "pnl": float(pnl),
                    "roe": float(roe_val) if roe_val is not None else None,
                    "leverage": lev,
                    "entry_leverage": pos.get("entry_leverage"),
                    "quantity": qty,
                    "notional": notional,
                    "margin": float(base_notional) if base_notional else None,
                    "cap_frac": float(notional / self.balance) if self.balance else 0.0,
                    "size_frac": pos.get("size_frac"),
                    "tag": pos.get("tag"),
                    "age_sec": age_sec,
                    "opt_hold_entry_sec": pos.get("opt_hold_entry_sec"),
                    "opt_hold_curr_sec": pos.get("opt_hold_curr_sec"),
                    "opt_hold_curr_remaining_sec": pos.get("opt_hold_curr_remaining_sec"),
                    "hold_eval_ts": pos.get("hold_eval_ts"),
                    "pos_source": pos.get("pos_source"),
                })
        else:
            for sym, pos in self.positions.items():
                px = None
                try:
                    px = self.market.get(sym, {}).get("price")
                except Exception:
                    px = None
                if px is None:
                    try:
                        px = float(pos.get("entry_price") or 0.0)
                    except Exception:
                        px = None
                if px is None:
                    continue
                entry = float(pos.get("entry_price", px))
                side = str(pos.get("side") or "")
                qty = float(pos.get("quantity", 0.0))
                notional = float(pos.get("notional", 0.0))
                pnl = ((px - entry) * qty) if side == "LONG" else ((entry - px) * qty)
                unreal += pnl
                total_notional += notional
                lev = float(pos.get("leverage", self.leverage))
                lev_for_roe = self._safe_float(pos.get("entry_leverage"), None)
                if lev_for_roe is None or lev_for_roe <= 0:
                    lev_for_roe = lev
                base_notional = notional / max(float(lev_for_roe), 1e-6)
                entry_t = self._normalize_entry_time_ms(pos.get("entry_time"), default=now_ms())
                age_sec = max(0.0, (now_ms() - int(entry_t)) / 1000.0)
                opt_hold_entry = pos.get("opt_hold_entry_sec") or pos.get("opt_hold_sec")
                opt_hold_curr = pos.get("opt_hold_curr_sec") or pos.get("opt_hold_sec") or opt_hold_entry
                opt_hold_rem = pos.get("opt_hold_curr_remaining_sec")
                if opt_hold_rem is None and opt_hold_curr is not None:
                    try:
                        opt_hold_rem = max(0.0, float(opt_hold_curr) - float(age_sec))
                    except Exception:
                        opt_hold_rem = None
                pos_list.append({
                    "symbol": sym,
                    "side": side,
                    "entry_price": entry,
                    "entry_time": int(entry_t),
                    "current": float(px),
                    "pnl": float(pnl),
                    "roe": float(pnl / base_notional) if base_notional else 0.0,
                    "leverage": lev,
                    "entry_leverage": lev_for_roe,
                    "quantity": qty,
                    "notional": notional,
                    "margin": float(base_notional) if base_notional else None,
                    "cap_frac": float(notional / self.balance) if self.balance else 0.0,
                    "size_frac": pos.get("size_frac"),
                    "tag": pos.get("tag"),
                    "age_sec": age_sec,
                    "opt_hold_entry_sec": opt_hold_entry,
                    "opt_hold_curr_sec": opt_hold_curr,
                    "opt_hold_curr_remaining_sec": opt_hold_rem,
                    "hold_eval_ts": pos.get("hold_eval_ts"),
                })
        equity = float(self.balance) + float(unreal)
        self._equity_history.append({"time": now_ms(), "equity": equity})
        util = float(total_notional / self.balance) if self.balance else 0.0
        self._persist_state(force=False)
        return equity, unreal, util, pos_list

    def _compute_eval_metrics(self):
        # Brier score, hit rate, avg EV vs realized R
        if not self.eval_history:
            return {"brier": None, "hit_rate": None, "avg_ev": None, "avg_realized_r": None, "avg_event_ev_r": None}
        briers = []
        hits = []
        evs = []
        evr = []
        realized = []
        for e in self.eval_history:
            hit = e.get("hit")
            pred_win = e.get("pred_win")
            if pred_win is not None and hit is not None:
                try:
                    briers.append((float(pred_win) - float(hit)) ** 2)
                except Exception:
                    pass
            if hit is not None:
                hits.append(float(hit))
            if e.get("pred_ev") is not None:
                evs.append(float(e.get("pred_ev")))
            if e.get("pred_event_ev_r") is not None:
                evr.append(float(e.get("pred_event_ev_r")))
            if e.get("realized_r") is not None:
                realized.append(float(e.get("realized_r")))
        return {
            "brier": float(np.mean(briers)) if briers else None,
            "hit_rate": float(np.mean(hits)) if hits else None,
            "avg_ev": float(np.mean(evs)) if evs else None,
            "avg_realized_r": float(np.mean(realized)) if realized else None,
            "avg_event_ev_r": float(np.mean(evr)) if evr else None,
        }

    def _run_portfolio_joint_sync(self, symbols: list[str], ai_scores: dict[str, float]) -> dict | None:
        if not symbols:
            return None
        try:
            from engines.mc.portfolio_joint_sim import PortfolioJointSimEngine, PortfolioConfig
        except Exception as e:
            self._log_err(f"[PORTFOLIO_JOINT] import failed: {e}")
            return None

        ohlcv_map: dict[str, list[tuple]] = {}
        for sym in symbols:
            candles = self.ohlcv_buffer.get(sym, [])
            if not candles:
                continue
            try:
                if isinstance(candles[0], dict):
                    ohlcv_map[sym] = [
                        (
                            float(c.get("open")),
                            float(c.get("high")),
                            float(c.get("low")),
                            float(c.get("close")),
                            float(c.get("volume")),
                        )
                        for c in candles
                    ]
                else:
                    ohlcv_map[sym] = [
                        (
                            float(c[1]),
                            float(c[2]),
                            float(c[3]),
                            float(c[4]),
                            float(c[5]) if len(c) > 5 else float(c[4]),
                        )
                        for c in candles
                    ]
            except Exception:
                continue

        if not ohlcv_map:
            return None

        try:
            cfg = PortfolioConfig(
                days=int(getattr(mc_config, "portfolio_days", 3)),
                simulations=int(getattr(mc_config, "portfolio_simulations", 12000)),
                batch_size=int(getattr(mc_config, "portfolio_batch_size", 4000)),
                block_size=int(getattr(mc_config, "portfolio_block_size", 12)),
                min_history=int(getattr(mc_config, "portfolio_min_history", 180)),
                drift_k=float(getattr(mc_config, "portfolio_drift_k", 0.35)),
                score_clip=float(getattr(mc_config, "portfolio_score_clip", 1.0)),
                tilt_strength=float(getattr(mc_config, "portfolio_tilt_strength", 0.6)),
                use_jumps=bool(getattr(mc_config, "portfolio_use_jumps", True)),
                p_jump_market=float(getattr(mc_config, "portfolio_p_jump_market", 0.005)),
                p_jump_idio=float(getattr(mc_config, "portfolio_p_jump_idio", 0.007)),
                target_leverage=float(getattr(mc_config, "portfolio_target_leverage", 10.0)),
                individual_cap=float(getattr(mc_config, "portfolio_individual_cap", 3.0)),
                risk_aversion=float(getattr(mc_config, "portfolio_risk_aversion", 0.5)),
                var_alpha=float(getattr(mc_config, "portfolio_var_alpha", 0.05)),
                market_factor_scale=float(getattr(mc_config, "portfolio_market_factor_scale", 1.0)),
                residual_scale=float(getattr(mc_config, "portfolio_residual_scale", 1.0)),
                leverage=float(self.max_leverage),
                rebalance_sim_enabled=bool(getattr(mc_config, "portfolio_rebalance_sim_enabled", False)),
                rebalance_interval=int(getattr(mc_config, "portfolio_rebalance_interval", 1)),
                rebalance_fee_bps=float(getattr(mc_config, "portfolio_rebalance_fee_bps", 6.0)),
                rebalance_slippage_bps=float(getattr(mc_config, "portfolio_rebalance_slip_bps", 4.0)),
                rebalance_score_noise=float(getattr(mc_config, "portfolio_rebalance_score_noise", 0.0)),
                rebalance_min_score=float(getattr(mc_config, "portfolio_rebalance_min_score", 0.0)),
                seed=None,
            )

            engine = PortfolioJointSimEngine(ohlcv_map, ai_scores, cfg)
            weights, report = engine.build_portfolio(symbols)
            report = dict(report)
            report["weights"] = weights
            report["target_leverage"] = float(cfg.target_leverage)
            return report
        except Exception as e:
            import traceback

            self._log_err(f"[PORTFOLIO_JOINT] run failed: {e}")
            try:
                self._log_err(traceback.format_exc())
            except Exception:
                pass
            return None

    async def _maybe_portfolio_joint(self, symbols: list[str], ai_scores: dict[str, float]) -> dict | None:
        now = time.time()
        if (now - float(self._last_portfolio_joint_ts)) < float(self.portfolio_joint_interval_sec):
            return self._last_portfolio_report
        report = await asyncio.to_thread(self._run_portfolio_joint_sync, symbols, ai_scores)
        if report:
            self._last_portfolio_joint_ts = now
            self._last_portfolio_report = report
        return report

    def _alpha_hit_status(self) -> dict:
        """Collect AlphaHit trainer/runtime metrics for the dashboard."""
        status = {
            "enabled": False,
            "disable_reason": None,
            "signal_boost": bool(getattr(mc_config, "alpha_signal_boost", False)),
            "alpha_scaling_factor": float(getattr(mc_config, "alpha_scaling_factor", 1.0)),
            "mu_alpha_cap": float(getattr(mc_config, "mu_alpha_cap", 5.0)),
            "beta": float(getattr(mc_config, "alpha_hit_beta", 1.0)),
            "min_buffer": int(getattr(mc_config, "alpha_hit_min_buffer", 1024)),
            "warmup_samples": int(getattr(mc_config, "alpha_hit_warmup_samples", 512)),
            "max_buffer": int(getattr(mc_config, "alpha_hit_max_buffer", 200000)),
            "buffer_size": 0,
            "total_samples": 0,
            "total_train_steps": 0,
            "last_loss": None,
            "ema_loss": None,
            "warmup_done": False,
            "trainer_device": None,
            "trainer_name": None,
            "replay_path": str(getattr(mc_config, "alpha_hit_replay_path", "") or ""),
            "replay_save_every": int(getattr(mc_config, "alpha_hit_replay_save_every", 2000)),
            "replay_exists": False,
            "replay_size_bytes": None,
        }

        replay_path = status["replay_path"]
        if replay_path:
            rp = Path(replay_path)
            status["replay_exists"] = rp.exists()
            if status["replay_exists"]:
                try:
                    status["replay_size_bytes"] = int(rp.stat().st_size)
                except Exception:
                    status["replay_size_bytes"] = None

        for eng in getattr(self.hub, "engines", []):
            if not getattr(eng, "alpha_hit_enabled", False):
                if status["disable_reason"] is None:
                    status["disable_reason"] = getattr(eng, "alpha_hit_disable_reason", None)
                continue
            status["enabled"] = True
            status["beta"] = float(getattr(eng, "alpha_hit_beta", status["beta"]))
            status["trainer_name"] = getattr(eng, "name", status["trainer_name"])
            trainer = getattr(eng, "alpha_hit_trainer", None)
            if trainer is not None:
                try:
                    status["buffer_size"] = int(getattr(trainer, "buffer_size", status["buffer_size"]))
                except Exception:
                    status["buffer_size"] = status["buffer_size"]
                cfg_obj = getattr(trainer, "cfg", None)
                if cfg_obj is not None:
                    status["warmup_samples"] = int(getattr(cfg_obj, "warmup_samples", status["warmup_samples"]))
                    status["replay_save_every"] = int(getattr(cfg_obj, "replay_save_every", status["replay_save_every"]))
                status["warmup_done"] = bool(getattr(trainer, "is_warmed_up", status["warmup_done"]))
                status["total_samples"] = int(getattr(trainer, "_total_samples", status["total_samples"]))
                status["total_train_steps"] = int(getattr(trainer, "_total_train_steps", status["total_train_steps"]))
                status["last_loss"] = getattr(trainer, "last_loss", status["last_loss"])
                status["ema_loss"] = getattr(trainer, "ema_loss", status["ema_loss"])
                dev = getattr(trainer, "device", status["trainer_device"])
                status["trainer_device"] = str(dev) if dev is not None else None
            status["warmup_done"] = status["warmup_done"] or (
                status["warmup_samples"] > 0 and status["buffer_size"] >= status["warmup_samples"]
            )
            break

        if not status["enabled"] and status["disable_reason"] is None:
            if not bool(getattr(mc_config, "alpha_hit_enable", True)):
                status["disable_reason"] = "ALPHA_HIT_ENABLE=0"
            else:
                use_torch_env = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower()
                if use_torch_env in ("0", "false", "no", "off"):
                    status["disable_reason"] = "MC_USE_TORCH=0"

        if status["min_buffer"] <= 0:
            status["warmup_done"] = True

        return status

    def _alpha_weight_status(self) -> dict:
        mlofi_path = str(getattr(mc_config, "mlofi_weight_path", "") or "state/mlofi_weights.json")
        causal_path = str(getattr(mc_config, "causal_weights_path", "") or "state/causal_weights.json")
        dir_path = str(getattr(mc_config, "alpha_direction_model_path", "") or "state/mu_direction_model.json")
        closed_now = int(max(0, int(self._closed_trade_count or 0)))
        last_exit_anchor = int(max(0, int(self._alpha_train_last_exit_closed or 0)))
        new_exits_since_last = int(max(0, closed_now - last_exit_anchor))
        trigger_new_exits = int(max(1, int(os.environ.get("ALPHA_TRAIN_TRIGGER_NEW_EXITS", 200) or 200)))
        remaining_force = int(max(0, trigger_new_exits - new_exits_since_last))
        force_ready = bool(new_exits_since_last >= trigger_new_exits)
        train_state_reason = ""
        try:
            p = self._alpha_train_state_path
            if p is not None and p.exists():
                with p.open("r", encoding="utf-8") as f:
                    st = json.load(f)
                if isinstance(st, dict):
                    train_state_reason = str(st.get("reason") or "")
        except Exception:
            train_state_reason = ""
        return {
            "collect_enabled": str(os.environ.get("ALPHA_WEIGHT_COLLECT", "0")).strip().lower() in ("1", "true", "yes", "on"),
            "train_enabled": str(os.environ.get("ALPHA_WEIGHT_TRAIN_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on"),
            "mlofi_samples": int(max(len(self._alpha_train_mlofi), int(self._alpha_mlofi_file_samples or 0))),
            "causal_samples": int(max(len(self._alpha_train_causal), int(self._alpha_causal_file_samples or 0))),
            "min_samples": int(self._alpha_train_min_samples),
            "last_save_ms": int(self._alpha_train_last_save_ms or 0),
            "last_train_ms": int(self._alpha_train_last_train_ms or 0),
            "mlofi_weight_path": mlofi_path,
            "causal_weight_path": causal_path,
            "direction_model_path": dir_path,
            "mlofi_weight_exists": bool(os.path.exists(mlofi_path)),
            "causal_weight_exists": bool(os.path.exists(causal_path)),
            "direction_model_exists": bool(os.path.exists(dir_path)),
            "direction_model_loaded": bool(self._alpha_dir_model),
            "ridge_lambda": float(os.environ.get("ALPHA_WEIGHT_RIDGE_LAMBDA", 1e-3) or 1e-3),
            "train_interval_sec": float(self._alpha_train_interval_sec),
            "save_interval_sec": float(self._alpha_train_save_interval_sec),
            "closed_total": int(closed_now),
            "new_exits_since_last_train": int(new_exits_since_last),
            "trigger_new_exits": int(trigger_new_exits),
            "remaining_to_force_train": int(remaining_force),
            "force_train_ready": bool(force_ready),
            "last_train_reason": str(train_state_reason),
        }

    def _get_local_mc_engine(self):
        """Return local MC engine if available (not process/remote)."""
        if self._mc_engine_cache is not None:
            return self._mc_engine_cache
        hub = getattr(self, "hub", None)
        engines = getattr(hub, "engines", None)
        if not engines:
            if not self._mc_engine_warned:
                self._log("[HOLD_EVAL] local EngineHub not available (process/remote). Hold-vs-exit skipped.")
                self._mc_engine_warned = True
            return None
        for eng in engines:
            if hasattr(eng, "evaluate_entry_metrics") and hasattr(eng, "_get_params"):
                self._mc_engine_cache = eng
                return eng
        if not self._mc_engine_warned:
            self._log("[HOLD_EVAL] MC engine not found in hub.")
            self._mc_engine_warned = True
        return None

    def _check_hold_vs_exit(
        self,
        sym: str,
        pos: dict,
        decision: dict,
        ctx: dict,
        ts: int,
        price: float,
        *,
        eval_only: bool = False,
    ) -> bool:
        """Re-run entry-eval for open positions and compare hold vs exit at t*."""
        try:
            enabled = str(os.environ.get("HOLD_EXIT_EVAL_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            enabled = False
        if not enabled:
            return False
        if not pos or not ctx:
            return False

        try:
            min_hold_sec = float(os.environ.get("EXIT_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC) or POSITION_HOLD_MIN_SEC)
        except Exception:
            min_hold_sec = float(POSITION_HOLD_MIN_SEC)
        try:
            entry_ts = int(pos.get("entry_time") or 0)
        except Exception:
            entry_ts = 0
        age_sec = 0.0
        if entry_ts:
            age_sec = (ts - entry_ts) / 1000.0
        allow_exit = True
        if entry_ts and min_hold_sec > 0 and age_sec < float(min_hold_sec):
            allow_exit = False

        try:
            interval_sec = float(os.environ.get("HOLD_EVAL_INTERVAL_SEC", 30.0) or 30.0)
        except Exception:
            interval_sec = 30.0
        last_ts = int(self._hold_eval_last_ts.get(sym, 0) or 0)
        if last_ts and (ts - last_ts) < int(max(1.0, interval_sec) * 1000.0):
            return False
        self._hold_eval_last_ts[sym] = int(ts)

        mc_engine = self._get_local_mc_engine()
        if mc_engine is None:
            return False

        side = str(pos.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return False

        try:
            ctx_eval = dict(ctx)
            ctx_eval["symbol"] = sym
            ctx_eval["price"] = float(price)
            ctx_eval["direction"] = 1 if side == "LONG" else -1
            try:
                ctx_eval["leverage"] = float(pos.get("leverage", self.leverage) or self.leverage)
            except Exception:
                ctx_eval["leverage"] = float(self.leverage)
            # Avoid side effects on alpha/EMA updates during auxiliary eval
            ctx_eval["_mu_alpha_ema_skip_update"] = True
            # Optional faster evaluation
            try:
                n_paths = int(os.environ.get("HOLD_EVAL_N_PATHS", 0) or 0)
            except Exception:
                n_paths = 0
            if n_paths > 0:
                ctx_eval["n_paths"] = int(n_paths)

            regime_ctx = str(ctx_eval.get("regime", "chop"))
            params = mc_engine._get_params(regime_ctx, ctx_eval)
            # deterministic seed (same 5s window as engine)
            seed_window = int(ts // 5000)
            seed = int((hash(sym) ^ seed_window) & 0xFFFFFFFF)
            metrics = mc_engine.evaluate_entry_metrics(ctx_eval, params, seed=seed)
        except Exception as e:
            self._log_err(f"[HOLD_EVAL] {sym} eval error: {e}")
            return False

        if not isinstance(metrics, dict):
            return False

        meta = metrics.get("meta") if isinstance(metrics.get("meta"), dict) else {}
        h_list = (
            metrics.get("horizon_seq")
            or meta.get("horizon_seq")
            or metrics.get("policy_horizons")
            or meta.get("policy_horizons")
        )
        if not h_list:
            return False

        ev_long_vec = metrics.get("ev_by_horizon_long") or meta.get("ev_by_horizon_long")
        ev_short_vec = metrics.get("ev_by_horizon_short") or meta.get("ev_by_horizon_short")
        if not ev_long_vec or not ev_short_vec:
            return False

        def _pick_tstar_for(side_label: str):
            if side_label == "LONG":
                t_val = metrics.get("unified_t_star_long") or meta.get("unified_t_star_long")
            else:
                t_val = metrics.get("unified_t_star_short") or meta.get("unified_t_star_short")
            if t_val is None:
                t_val = metrics.get("unified_t_star") or meta.get("unified_t_star")
            if t_val is None:
                t_val = metrics.get("policy_horizon_eff_sec") or meta.get("policy_horizon_eff_sec")
            if t_val is None:
                t_val = metrics.get("best_h") or meta.get("best_h")
            try:
                t_val = float(t_val) if t_val is not None else None
            except Exception:
                t_val = None
            return t_val

        tstar_cur = _pick_tstar_for(side)
        if tstar_cur is None or tstar_cur <= 0:
            return False
        try:
            t_min = float(os.environ.get("HOLD_EVAL_MIN_HORIZON_SEC", 0.0) or 0.0)
        except Exception:
            t_min = 0.0
        try:
            t_max = float(os.environ.get("HOLD_EVAL_MAX_HORIZON_SEC", 0.0) or 0.0)
        except Exception:
            t_max = 0.0
        if t_min > 0:
            tstar_cur = max(tstar_cur, t_min)
        if t_max > 0:
            tstar_cur = min(tstar_cur, t_max)

        # Pick EV at t* (nearest horizon)
        try:
            h_list_f = [float(h) for h in h_list]
            idx_cur = min(range(len(h_list_f)), key=lambda i: abs(h_list_f[i] - float(tstar_cur)))
            ev_cur_at_t = float(ev_long_vec[idx_cur] if side == "LONG" else ev_short_vec[idx_cur])
            h_pick = float(h_list_f[idx_cur])
        except Exception:
            return False

        # Adjust EV to remove entry cost (hold already paid entry fee)
        try:
            add_entry = str(os.environ.get("HOLD_EVAL_ADD_ENTRY_COST", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            add_entry = True
        cost_base = metrics.get("fee_roundtrip_total")
        if cost_base is None:
            cost_base = metrics.get("execution_cost") or meta.get("execution_cost") or meta.get("fee_roundtrip_total")
        try:
            lev_val = float(ctx_eval.get("leverage") or 1.0)
        except Exception:
            lev_val = 1.0
        try:
            cost_entry_roe = float(cost_base or 0.0) * float(lev_val) / 2.0
        except Exception:
            cost_entry_roe = 0.0
        cost_exit_roe = float(cost_entry_roe)
        hold_ev = float(ev_cur_at_t) + (cost_entry_roe if add_entry else 0.0)
        exit_now_ev = -float(cost_exit_roe)

        try:
            margin = float(os.environ.get("HOLD_EVAL_EXIT_MARGIN", 0.0) or 0.0)
        except Exception:
            margin = 0.0

        # Persist hold horizon on position for later event MC alignment
        try:
            if pos.get("opt_hold_entry_sec") is None:
                pos["opt_hold_entry_sec"] = int(round(float(tstar_cur)))
                pos["opt_hold_entry_src"] = pos.get("opt_hold_src") or "hold_eval"
            pos["opt_hold_sec"] = int(round(float(tstar_cur)))
            pos["opt_hold_src"] = "hold_eval"
            pos["opt_hold_curr_sec"] = int(round(float(tstar_cur)))
            pos["opt_hold_curr_src"] = "hold_eval"
            pos["opt_hold_curr_remaining_sec"] = (
                int(round(max(0.0, float(tstar_cur) - float(age_sec))))
                if age_sec is not None
                else None
            )
            pos["hold_ev_tstar"] = float(hold_ev)
            pos["hold_ev_raw_tstar"] = float(ev_cur_at_t)
            pos["hold_eval_ts"] = int(ts)
            pos["hold_eval_h_pick"] = int(round(h_pick))
        except Exception:
            pass

        # Guard hold-vs-exit from firing too early relative to t* horizon.
        if allow_exit and tstar_cur and float(tstar_cur) > 0:
            try:
                hold_progress_min = float(os.environ.get("HOLD_EVAL_MIN_PROGRESS_TO_EXIT", 0.85) or 0.85)
            except Exception:
                hold_progress_min = 0.85
            hold_progress_min = float(max(0.0, min(1.5, hold_progress_min)))
            try:
                hold_progress = float(age_sec) / max(float(tstar_cur), 1e-6)
            except Exception:
                hold_progress = 0.0
            if hold_progress < hold_progress_min:
                allow_exit = False
                try:
                    pos["hold_eval_exit_guard_progress"] = float(hold_progress)
                    pos["hold_eval_exit_guard_required"] = float(hold_progress_min)
                except Exception:
                    pass

        if eval_only:
            return False

        hold_exit_diag = {}
        try:
            hold_decision_for_exit = {"action": side, "meta": dict(meta)}
            _hold_pol, hold_exit_diag = self._build_dynamic_exit_policy(
                sym,
                pos,
                hold_decision_for_exit,
                ctx=ctx_eval,
            )
        except Exception:
            hold_exit_diag = {}
        hold_mode_state = self._resolve_exit_mode_state(
            hold_exit_diag,
            shock_threshold_env="HOLD_EVAL_EXIT_SHOCK_FAST_THRESHOLD",
            shock_threshold_default=self._safe_float(os.environ.get("EVENT_EXIT_SHOCK_FAST_THRESHOLD"), 1.0) or 1.0,
        )
        hold_confirm_normal, hold_confirm_reset_sec, _hold_ticks = self._get_exit_confirmation_rule(
            "HOLD_EVAL_EXIT",
            "normal",
            default_normal=2,
            default_shock=1,
            default_noise=3,
            default_reset_sec=300.0,
        )

        # Optional flip: if opposite side has stronger EV at its own t*
        try:
            allow_flip = str(os.environ.get("HOLD_EVAL_ALLOW_FLIP", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            allow_flip = True
        if allow_exit and allow_flip:
            opp_side = "SHORT" if side == "LONG" else "LONG"
            tstar_opp = _pick_tstar_for(opp_side)
            if tstar_opp is not None:
                if t_min > 0:
                    tstar_opp = max(tstar_opp, t_min)
                if t_max > 0:
                    tstar_opp = min(tstar_opp, t_max)
            ev_opp_at_t = None
            if tstar_opp is not None and tstar_opp > 0:
                try:
                    idx_opp = min(range(len(h_list_f)), key=lambda i: abs(h_list_f[i] - float(tstar_opp)))
                    ev_opp_at_t = float(ev_short_vec[idx_opp] if opp_side == "SHORT" else ev_long_vec[idx_opp])
                except Exception:
                    ev_opp_at_t = None
            try:
                flip_margin = float(os.environ.get("HOLD_EVAL_FLIP_MARGIN", 0.0) or 0.0)
            except Exception:
                flip_margin = 0.0
            try:
                flip_min_ev = float(os.environ.get("HOLD_EVAL_FLIP_MIN_EV", 0.0) or 0.0)
            except Exception:
                flip_min_ev = 0.0
            if ev_opp_at_t is not None and ev_opp_at_t > flip_min_ev:
                # Opp EV is net-of-entry cost; subtract exit cost to flip
                flip_ev = float(ev_opp_at_t) - float(cost_exit_roe)
                if flip_ev > (hold_ev + flip_margin):
                    meta_for_flip = dict(meta)
                    score_long = metrics.get("unified_score_long") or meta.get("unified_score_long")
                    score_short = metrics.get("unified_score_short") or meta.get("unified_score_short")
                    flip_score = score_short if opp_side == "SHORT" else score_long
                    if flip_score is None:
                        flip_score = float(ev_opp_at_t)
                    meta_for_flip.setdefault("hybrid_score", flip_score)
                    meta_for_flip.setdefault("unified_score", flip_score)
                    meta_for_flip.setdefault("regime", ctx_eval.get("regime", meta_for_flip.get("regime")))
                    flip_decision = {
                        "action": opp_side,
                        "ev": float(ev_opp_at_t),
                        "confidence": float(metrics.get("confidence", 0.0) or 0.0),
                        "unified_score": float(flip_score) if flip_score is not None else float(ev_opp_at_t),
                        "hybrid_score": float(flip_score) if flip_score is not None else float(ev_opp_at_t),
                        "leverage": float(ctx_eval.get("leverage") or lev_val),
                        "meta": meta_for_flip,
                    }
                    self._log(
                        f"[{sym}] EXIT+FLIP by HOLD_EVAL "
                        f"(cur={side} t*={int(round(tstar_cur))}s ev@t*={hold_ev*100:.4f}% -> "
                        f"{opp_side} t*={int(round(tstar_opp)) if tstar_opp else '-'}s ev@t*={float(ev_opp_at_t)*100:.4f}%)"
                    )
                    self._close_position(sym, float(price), "hold_vs_exit_flip", exit_kind="RISK")
                    permit, _deny = self._entry_permit(sym, flip_decision, ts)
                    if permit:
                        self._enter_position(sym, opp_side, float(price), flip_decision, ts, ctx=ctx, leverage_override=lev_val)
                    return True

        hold_exit_candidate = bool(allow_exit and hold_ev <= (exit_now_ev + margin))
        if hold_exit_candidate:
            shock_score = float(hold_mode_state.get("shock_score") or 0.0)
            noise_mode = bool(hold_mode_state.get("noise_mode"))
            confirm_mode = str(hold_mode_state.get("mode") or "normal")
            confirm_required, hold_confirm_reset_sec, _hold_ticks_dyn = self._get_exit_confirmation_rule(
                "HOLD_EVAL_EXIT",
                confirm_mode,
                default_normal=2,
                default_shock=1,
                default_noise=3,
                default_reset_sec=300.0,
            )
            confirm_ok, confirm_cnt = self._advance_exit_confirmation(
                pos,
                "hold_eval_exit",
                triggered=True,
                ts_ms=int(ts),
                required_ticks=confirm_required,
                reset_sec=float(max(0.0, hold_confirm_reset_sec)),
            )
            try:
                pos["hold_eval_exit_candidate"] = True
                pos["hold_eval_exit_confirm_mode"] = str(confirm_mode)
                pos["hold_eval_exit_confirm_required"] = int(confirm_required)
                pos["hold_eval_exit_confirm_count"] = int(confirm_cnt)
                pos["hold_eval_exit_confirmed"] = bool(confirm_ok)
                pos["hold_eval_exit_shock_score"] = float(shock_score)
                pos["hold_eval_exit_noise_mode"] = bool(noise_mode)
            except Exception:
                pass
            if not confirm_ok:
                return False
            self._log(
                f"[{sym}] EXIT by HOLD_EVAL "
                f"(t*={int(round(tstar_cur))}s, ev@t*={hold_ev*100:.4f}%, exit_now={exit_now_ev*100:.4f}%)"
            )
            self._close_position(sym, float(price), "hold_vs_exit", exit_kind="RISK")
            return True

        _hold_conf_ok, _hold_conf_cnt = self._advance_exit_confirmation(
            pos,
            "hold_eval_exit",
            triggered=False,
            ts_ms=int(ts),
            required_ticks=int(max(1, hold_confirm_normal)),
            reset_sec=float(max(0.0, hold_confirm_reset_sec)),
        )
        try:
            pos["hold_eval_exit_candidate"] = False
            pos["hold_eval_exit_confirm_mode"] = "idle"
            pos["hold_eval_exit_confirm_required"] = int(max(1, hold_confirm_normal))
            pos["hold_eval_exit_confirm_count"] = int(_hold_conf_cnt)
            pos["hold_eval_exit_confirmed"] = bool(_hold_conf_ok)
        except Exception:
            pass

        return False

    def _refresh_mc_device_info(self) -> None:
        """Refresh MC backend/device info for dashboard display."""
        if USE_REMOTE_ENGINE:
            self._last_mc_backend = "remote"
            self._last_mc_device = "remote"
            return
        use_torch_env = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower()
        use_torch = use_torch_env in ("1", "true", "yes", "on")
        if not use_torch:
            self._last_mc_backend = "numpy"
            self._last_mc_device = "cpu"
            return
        try:
            from engines.mc.torch_backend import _TORCH_OK, get_torch_device, torch
            if not _TORCH_OK:
                self._last_mc_backend = "numpy"
                self._last_mc_device = "cpu"
                return
            dev = get_torch_device()
            dev_type = getattr(dev, "type", None) or "cpu"
            self._last_mc_backend = "torch"
            self._last_mc_device = str(dev_type)
            # Best-effort warm cache: no extra info needed for dashboard
        except Exception:
            self._last_mc_backend = None
            self._last_mc_device = None

    async def broadcast(self, rows):
        try:
            # DEBUG: rows 개수 확인
            if not rows:
                self._log("[WARN] broadcast: rows empty!")
            
            equity, unreal, util, pos_list = self._compute_portfolio()
            ts = now_ms()
            if self.risk_manager.check_emergency_stop(equity):
                if not self._emergency_stop_handled:
                    self._register_anomaly("kill_switch", "critical", f"DD stop triggered equity={equity:.2f}")
                    self._liquidate_all_positions()
                    self.safety_mode = True
                    self._emergency_stop_handled = True
                    self._emergency_stop_ts = ts
                    self._log_err("[RISK] Emergency stop engaged: trading halted, dashboard stays online")
                else:
                    self.safety_mode = True
            eval_metrics = self._compute_eval_metrics()

            feed_connected = (self._last_feed_ok_ms > 0) and (ts - self._last_feed_ok_ms < 10_000)
            feed = {
                "connected": bool(feed_connected),
                "last_msg_age": (ts - self._last_feed_ok_ms) if self._last_feed_ok_ms else None
            }

            # Debug: announce broadcast attempt
            try:
                clients_now = len(self.clients)
                # If no clients currently connected, wait briefly (non-blocking) for a short window
                # to allow newly-connected dashboards to receive this update.
                if clients_now == 0:
                    wait_deadline = time.time() + 0.5  # seconds total to wait
                    while time.time() < wait_deadline and len(self.clients) == 0:
                        # yield control to event loop briefly without blocking
                        try:
                            await asyncio.sleep(0.05)
                        except Exception:
                            time.sleep(0.05)
                            break
                    clients_now = len(self.clients)
                self._log(f"[BROADCAST] clients={clients_now} rows={len(rows)}")
            except Exception:
                pass

            try:
                hist_limit = max(100, int(getattr(config, "DASHBOARD_HISTORY_MAX", 3000) or 3000))
            except Exception:
                hist_limit = 3000
            try:
                tape_limit = max(100, int(getattr(config, "DASHBOARD_TRADE_TAPE_MAX", 500) or 500))
            except Exception:
                tape_limit = 500
            include_details = bool(getattr(config, "DASHBOARD_INCLUDE_DETAILS", False))

            # Keep websocket payload compact so dashboard rendering stays responsive.
            market_rows = rows
            if not include_details:
                compact_rows = []
                for r in rows or []:
                    if isinstance(r, dict) and "details" in r:
                        rc = dict(r)
                        rc.pop("details", None)
                        compact_rows.append(rc)
                    else:
                        compact_rows.append(r)
                market_rows = compact_rows

            eq_history = list(self._equity_history)[-hist_limit:]
            live_history = list(self._live_equity_history)[-hist_limit:]
            trade_tape = list(self.trade_tape)[-tape_limit:]

            payload = {
                "type": "full_update",
                "server_time": ts,
                "kill_switch": bool(self.safety_mode),
                "engine": {
                    "modules_ok": True,
                    "ws_clients": len(self.clients),
                    "loop_ms": None,
                    "safety_mode": bool(self.safety_mode),
                    "emergency_stop_ts": self._emergency_stop_ts,
                    "env_profile": getattr(config, "ENV_PROFILE", None),
                    "env_file": getattr(config, "ENV_FILE", None),
                    "env_active": getattr(config, "ENV_ACTIVE", None),
                    "env_sources": getattr(config, "ENV_SOURCES", None),
                    "record_mode": getattr(self._trading_mode, "value", None),
                    "record_db": getattr(self, "_db_path", None),
                    "outside_universe_positions": list(self._outside_universe_positions.values()),
                    "mc_last_ms": self._last_mc_ms,
                    "mc_last_ts": self._last_mc_ts,
                    "mc_last_ctxs": self._last_mc_ctxs,
                    "mc_status": self._last_mc_status,
                    "mc_backend": self._last_mc_backend,
                    "mc_device": self._last_mc_device,
                },
                # Alpha Hit ML 및 Signal Boost 상태
                "alpha_hit": self._alpha_hit_status(),
                "alpha_weights": self._alpha_weight_status(),
                "reval": self._reval_status(ts),
                "auto_reval": self._auto_reval_status(ts),
                "auto_tune": self._auto_tune_status(ts),
                "event_alignment": self._event_alignment_status(),
                "feed": feed,
                "market": market_rows,
                "portfolio": {
                    "balance": float(self.balance),
                    "equity": float(equity),
                    "unrealized_pnl": float(unreal),
                    "utilization": util,
                    "utilization_cap": float(self.max_notional_frac) if self.exposure_cap_enabled else None,
                    "positions_count": int(len(pos_list)),
                    "positions": pos_list,
                    "positions_source": (
                        "exchange_ws" if (self.enable_orders and self._exchange_positions_source == "ws")
                        else ("exchange" if (self.enable_orders and self._exchange_positions_ts is not None) else "engine")
                    ),
                    "positions_sync_ts": self._exchange_positions_ts,
                    "history": eq_history,
                    "live_wallet_balance": self._live_wallet_balance,
                    "live_equity": self._live_equity,
                    "live_free_balance": self._live_free_balance,
                    "live_total_initial_margin": self._live_total_initial_margin,
                    "live_total_maintenance_margin": self._live_total_maintenance_margin,
                    "live_last_sync_ms": self._last_live_sync_ms,
                    "live_last_sync_err": self._last_live_sync_err,
                    "live_history": live_history,
                    "pre_mc": None,
                },
                "eval_metrics": eval_metrics,
                "logs": list(self.logs),
                "trade_tape": trade_tape,
                "alerts": list(self.anomalies),
            }

            # Attach compact portfolio pre-MC summary (if available)
            try:
                if self._last_portfolio_report:
                    r = self._last_portfolio_report
                    payload["portfolio"]["pre_mc"] = {
                        "expected_pnl": r.get("expected_portfolio_pnl"),
                        "cvar": r.get("cvar"),
                        "prob_account_liq": r.get("prob_account_liquidation_proxy"),
                        "rebalance_expected_pnl": r.get("rebalance_expected_portfolio_pnl"),
                        "rebalance_turnover_mean": r.get("rebalance_turnover_mean"),
                        "rebalance_turnover_p95": r.get("rebalance_turnover_p95"),
                        "rebalance_cost_mean": r.get("rebalance_cost_mean"),
                    }
            except Exception:
                pass

            msg = json.dumps(payload, ensure_ascii=False)
            # 캐시: 새로 연결되는 클라이언트에 즉시 전송하기 위해 마지막 페이로드 보관
            try:
                self._last_broadcast_msg = msg
            except Exception:
                self._last_broadcast_msg = None
            # Debug: write last broadcast to a temp file for offline inspection (truncated legacy dump)
            try:
                with open('/tmp/last_broadcast.json', 'w', encoding='utf-8') as _f:
                    _f.write(msg[:200000])
            except Exception:
                pass

            # Write a compact diagnostics JSON (non-truncated) focused on rebalance fields
            try:
                diag = {
                    "ts": int(time.time() * 1000),
                    "market_diag": [
                        {
                            "symbol": r.get("symbol"),
                            "rebalance_decision": r.get("rebalance_decision"),
                            "rebalance_weight": r.get("rebalance_weight"),
                            "rebalance_delta_ev": r.get("rebalance_delta_ev"),
                            "rebalance_exec_cost": r.get("rebalance_exec_cost"),
                            "rebalance_target_leverage": r.get("rebalance_target_leverage"),
                            "rebalance_allow_trade": r.get("rebalance_allow_trade"),
                            "status": r.get("status"),
                            "unified": r.get("unified_score"),
                            "reason": r.get("direction_reason"),
                            "sL": r.get("score_long"),
                            "sS": r.get("score_short"),
                            "sT": r.get("score_threshold"),
                        }
                        for r in rows
                    ],
                }
                with open('/tmp/last_broadcast_diag.json', 'w', encoding='utf-8') as _f:
                    _f.write(json.dumps(diag, ensure_ascii=False))
            except Exception:
                pass
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)
            try:
                self._log(f"[BROADCAST_DONE] sent rows={len(rows)} clients_now={len(self.clients)}")
            except Exception:
                pass
        except Exception as e:
            import traceback
            err_text = f"{e} {traceback.format_exc()}"
            self._log_err(f"[ERR] broadcast: {err_text}")
            self._note_runtime_error("broadcast", err_text)

    async def dashboard_loop(self):
        """Fast UI refresh loop using latest decisions + market data."""
        try:
            interval = float(self._dashboard_refresh_sec)
        except Exception:
            interval = 0.5
        interval = max(0.1, interval)
        while True:
            try:
                if not self._dashboard_fast:
                    await asyncio.sleep(1.0)
                    continue
                if len(self.clients) == 0:
                    await asyncio.sleep(interval)
                    continue
                ts = now_ms()
                rows = []
                for sym in SYMBOLS:
                    price = self.market[sym]["price"]
                    if price is None:
                        continue
                    decision = self._last_decisions.get(sym)
                    candles = len(self.ohlcv_buffer[sym])
                    ctx_dash = self._last_ctx_by_sym.get(sym)
                    row = self._row(sym, float(price), ts, decision, candles, ctx=ctx_dash, log_filters=False)
                    rows.append(row)
                if rows:
                    self._last_rows = rows
                    await self.broadcast(rows)
            except Exception as e:
                self._log_err(f"[ERR] dashboard_loop: {e}")
                self._note_runtime_error("dashboard_loop", str(e))
            await asyncio.sleep(interval)

    async def decision_loop(self):
        """
        [REFACTORED] 3-Stage Decision Pipeline:
          1. Context Collection: _build_decision_context() per symbol
          2. Batch Decision: hub.decide_batch(ctx_list)
          3. Result Application: _apply_decision() per symbol (isolated exceptions)
        """
        while True:
            try:
                rows = []
                ts = now_ms()
                try:
                    self._maybe_reload_auto_tune_overrides(int(ts))
                except Exception as e:
                    self._log_err(f"[AUTO_TUNE] reload loop error: {e}")
                try:
                    self._force_rebalance_cycle = bool(self._rebalance_on_next_cycle)
                    self._rebalance_on_next_cycle = False
                except Exception:
                    self._force_rebalance_cycle = False
                try:
                    self._cycle_free_balance = self._sizing_balance()
                    self._cycle_reserved_margin = 0.0
                    self._cycle_reserved_notional = 0.0
                    self._cycle_balance_ts = ts
                except Exception:
                    pass
                self._decision_cycle = (self._decision_cycle + 1) % self._decision_log_every
                log_this_cycle = (self._decision_cycle == 0)
                try:
                    self._refresh_runtime_universe()
                except Exception as e:
                    self._log_err(f"[ERR] refresh_universe: {e}")

                # ====== Stage 1: Context Collection (SoA Optimized) ======
                # [OPTIMIZATION] _build_batch_context_soa: Pre-allocated arrays + minimal dict creation
                ctx_list = []
                ctx_sym_map = {}  # sym -> ctx for later lookup
                valid_indices = None
                
                try:
                    ctx_list, valid_indices = self._build_batch_context_soa(ts)
                    # Cycle debug: log ctx_list size
                    try:
                        self._log(f"[CYCLE] ts={ts} ctx_count={len(ctx_list)} valid_indices_count={0 if valid_indices is None else len(valid_indices)}")
                    except Exception:
                        pass
                    ctx_sym_map = {ctx["symbol"]: ctx for ctx in ctx_list}
                    try:
                        # Store only dashboard-relevant alpha/regime fields (avoid copying large arrays).
                        compact_map: dict[str, dict] = {}
                        for _sym, _ctx in ctx_sym_map.items():
                            compact_map[_sym] = {
                                "regime": _ctx.get("regime"),
                                "mlofi": _ctx.get("mlofi"),
                                "vpin": _ctx.get("vpin"),
                                "mu_kf": _ctx.get("mu_kf"),
                                "hurst": _ctx.get("hurst"),
                                "mu_ml": _ctx.get("mu_ml"),
                                "mu_bayes": _ctx.get("mu_bayes"),
                                "mu_ar": _ctx.get("mu_ar"),
                                "mu_pf": _ctx.get("mu_pf"),
                                "mu_ou": _ctx.get("mu_ou"),
                                "hawkes_boost": _ctx.get("hawkes_boost"),
                                "sigma_garch": _ctx.get("sigma_garch"),
                                "mu_dir_prob_long": _ctx.get("mu_dir_prob_long"),
                                "mu_dir_edge": _ctx.get("mu_dir_edge"),
                                "mu_dir_conf": _ctx.get("mu_dir_conf"),
                                "hmm_state": _ctx.get("hmm_state"),
                                "hmm_conf": _ctx.get("hmm_conf"),
                                "hmm_regime_sign": _ctx.get("hmm_regime_sign"),
                            }
                        if compact_map:
                            self._last_ctx_by_sym.update(compact_map)
                    except Exception:
                        pass

                    # 유효하지 않은 심볼들은 빈 row로 채움
                    valid_syms = set(ctx_sym_map.keys())
                    for sym in SYMBOLS:
                        if sym not in valid_syms:
                            candles = len(self.ohlcv_buffer[sym])
                            rows.append(self._row(sym, None, ts, None, candles))
                except Exception as e:
                    import traceback
                    self._log_err(f"[ERR] build_batch_ctx: {e} {traceback.format_exc()}")
                    # Fallback: 개별 빌드 (기존 방식)
                    for sym in SYMBOLS:
                        try:
                            ctx = self._build_decision_context(sym, ts)
                            if ctx is None:
                                candles = len(self.ohlcv_buffer[sym])
                                rows.append(self._row(sym, None, ts, None, candles))
                                continue
                            ctx_list.append(ctx)
                            ctx_sym_map[sym] = ctx
                        except Exception as e2:
                            self._log_err(f"[ERR] build_ctx {sym}: {e2}")
                            candles = len(self.ohlcv_buffer[sym])
                            rows.append(self._row(sym, None, ts, None, candles))

                # ====== Stage 2: Batch Decision Execution (GPU in separate thread) ======
                batch_decisions = []
                if ctx_list:
                    streaming = bool(getattr(config, "MC_STREAMING_MODE", False))
                    stream_batch = int(getattr(config, "MC_STREAMING_BATCH_SIZE", 4) or 4)
                    if stream_batch <= 0:
                        stream_batch = 1
                    stream_broadcast = bool(getattr(config, "MC_STREAMING_BROADCAST", True))

                    async def _run_decide_batch(ctx_subset: list[dict]) -> tuple[list[dict], str, float]:
                        if not ctx_subset:
                            return [], "empty", 0.0
                        t0_mc = time.perf_counter()
                        mc_status = "ok"
                        decisions: list[dict] = []
                        try:
                            # 우선 비동기 프로세스 허브가 있으면 polling 방식으로 사용
                            if hasattr(self.hub, "decide_batch_async") and asyncio.iscoroutinefunction(getattr(self.hub, "decide_batch_async")):
                                try:
                                    decisions = await asyncio.wait_for(
                                        self.hub.decide_batch_async(ctx_subset, timeout=DECIDE_BATCH_TIMEOUT_SEC),
                                        timeout=DECIDE_BATCH_TIMEOUT_SEC,
                                    )
                                except asyncio.TimeoutError:
                                    mc_status = "timeout"
                                    self._log_err(f"[ERR] decide_batch: timeout after {DECIDE_BATCH_TIMEOUT_SEC}s")
                                    self._note_runtime_error("decide_batch", f"timeout {DECIDE_BATCH_TIMEOUT_SEC}s")
                                    decisions = [{"action": "WAIT", "ev": 0.0, "reason": "batch_timeout"} for _ in ctx_subset]
                            else:
                                # GPU 연산을 별도 스레드에서 실행하여 asyncio 블로킹 방지
                                loop = asyncio.get_event_loop()
                                try:
                                    decisions = await asyncio.wait_for(
                                        loop.run_in_executor(
                                            GPU_EXECUTOR,
                                            self.hub.decide_batch,
                                            ctx_subset,
                                        ),
                                        timeout=DECIDE_BATCH_TIMEOUT_SEC,
                                    )
                                except asyncio.TimeoutError:
                                    mc_status = "timeout"
                                    self._log_err(f"[ERR] decide_batch: timeout after {DECIDE_BATCH_TIMEOUT_SEC}s")
                                    self._note_runtime_error("decide_batch", f"timeout {DECIDE_BATCH_TIMEOUT_SEC}s")
                                    decisions = [{"action": "WAIT", "ev": 0.0, "reason": "batch_timeout"} for _ in ctx_subset]
                        except Exception as e:
                            import traceback
                            mc_status = "error"
                            self._log_err(f"[ERR] decide_batch: {e} {traceback.format_exc()}")
                            self._note_runtime_error("decide_batch", str(e))
                            decisions = [{"action": "WAIT", "ev": 0.0, "reason": "batch_error"} for _ in ctx_subset]
                        elapsed_ms = (time.perf_counter() - t0_mc) * 1000.0
                        return decisions, mc_status, elapsed_ms

                    # Lightweight cycle log for debugging latency and ctx size
                    try:
                        self._log(f"[CYCLE] cycle={self._decision_cycle} ts={ts} ctx_count={len(ctx_list)} valid_indices={len(valid_indices) if valid_indices is not None else 'N/A'} streaming={streaming} batch={stream_batch}")
                    except Exception:
                        pass

                    if streaming:
                        batch_decisions = [None for _ in ctx_list]

                        # Prepare row cache for streaming UI updates
                        rows_cache: dict[str, dict] = {}
                        try:
                            if getattr(self, "_last_rows", None):
                                for r in self._last_rows:
                                    if isinstance(r, dict) and r.get("symbol"):
                                        rows_cache[r["symbol"]] = r
                        except Exception:
                            pass
                        for r in rows:
                            if isinstance(r, dict) and r.get("symbol"):
                                rows_cache[r["symbol"]] = r

                        def _rows_from_cache() -> list[dict]:
                            out = []
                            for s in SYMBOLS:
                                row = rows_cache.get(s)
                                if row is None:
                                    try:
                                        candles = len(self.ohlcv_buffer[s])
                                    except Exception:
                                        candles = 0
                                    row = self._row(s, None, ts, None, candles, log_filters=False)
                                out.append(row)
                            return out

                        mc_total_ms = 0.0
                        mc_status_final = "ok"
                        processed = 0

                        for start in range(0, len(ctx_list), stream_batch):
                            chunk = ctx_list[start:start + stream_batch]
                            decisions, mc_status, elapsed_ms = await _run_decide_batch(chunk)

                            # Update MC timing/progress for streaming
                            mc_total_ms += float(elapsed_ms or 0.0)
                            processed += len(chunk)
                            try:
                                self._last_mc_ms = mc_total_ms
                                self._last_mc_ts = ts
                                self._last_mc_ctxs = processed
                                if mc_status == "error":
                                    mc_status_final = "error"
                                elif mc_status == "timeout" and mc_status_final != "error":
                                    mc_status_final = "timeout"
                                self._last_mc_status = "partial" if processed < len(ctx_list) else mc_status_final
                            except Exception:
                                pass

                            # If chunk failed, reuse last good decisions to avoid all-zero UI
                            if decisions:
                                try:
                                    for i, dec in enumerate(decisions):
                                        reason = str(dec.get("reason", ""))
                                        if reason in ("batch_timeout", "batch_error"):
                                            sym = chunk[i]["symbol"]
                                            cached = self._last_decisions.get(sym)
                                            if cached:
                                                decisions[i] = cached
                                except Exception:
                                    pass

                            # Cache decisions + update streaming rows
                            for i, dec in enumerate(decisions):
                                idx = start + i
                                if idx >= len(batch_decisions):
                                    continue
                                batch_decisions[idx] = dec
                                sym = chunk[i]["symbol"]
                                try:
                                    row = self._row(sym, chunk[i].get("price"), ts, dec, chunk[i].get("candles", 0), ctx=chunk[i], log_filters=False)
                                except Exception:
                                    row = self._row(sym, chunk[i].get("price"), ts, None, chunk[i].get("candles", 0), ctx=chunk[i], log_filters=False)
                                rows_cache[sym] = row

                            if stream_broadcast and not self._dashboard_fast:
                                try:
                                    await self.broadcast(_rows_from_cache())
                                except Exception as e:
                                    self._log_err(f"[ERR] broadcast(stream): {e}")
                                    self._note_runtime_error("broadcast_stream", str(e))

                        # Final MC status after streaming chunks
                        try:
                            self._last_mc_ms = mc_total_ms
                            self._last_mc_ts = ts
                            self._last_mc_ctxs = len(ctx_list)
                            self._last_mc_status = mc_status_final
                            self._refresh_mc_device_info()
                        except Exception:
                            pass

                        # Fill missing decisions (if any)
                        for i, dec in enumerate(batch_decisions):
                            if dec is None:
                                batch_decisions[i] = {"action": "WAIT", "ev": 0.0, "reason": "batch_missing"}
                    else:
                        decisions, mc_status, elapsed_ms = await _run_decide_batch(ctx_list)
                        batch_decisions = decisions
                        try:
                            self._last_mc_ms = float(elapsed_ms or 0.0)
                            self._last_mc_ts = ts
                            self._last_mc_ctxs = len(ctx_list)
                            self._last_mc_status = mc_status
                            self._refresh_mc_device_info()
                        except Exception:
                            pass

                # If batch failed, reuse last good decisions to avoid all-zero UI
                if batch_decisions:
                    try:
                        for i, dec in enumerate(batch_decisions):
                            reason = str(dec.get("reason", ""))
                            if reason in ("batch_timeout", "batch_error"):
                                sym = ctx_list[i]["symbol"]
                                cached = self._last_decisions.get(sym)
                                if cached:
                                    batch_decisions[i] = cached
                    except Exception:
                        pass

                # ====== Stage 2.5: Portfolio Ranking & TOP N Selection + Kelly Allocation ======
                
                # ✅ UnifiedScore 분포 통계 및 필터 분석 (항상 실행)
                if batch_decisions:
                    # Always update auto-tuning from latest scores (log throttling handled separately)
                    use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
                    hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
                    hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")

                    def _pick_score(dec: dict, keys: list[str]) -> float:
                        for k in keys:
                            v = dec.get(k)
                            if v is not None:
                                return float(v)
                        meta = dec.get("meta") or {}
                        for k in keys:
                            v = meta.get(k)
                            if v is not None:
                                return float(v)
                        for d in dec.get("details", []):
                            m = d.get("meta") or {}
                            for k in keys:
                                v = m.get(k)
                                if v is not None:
                                    return float(v)
                        return float(dec.get("ev", 0.0) or 0.0)

                    scores_list = []
                    for i, dec in enumerate(batch_decisions):
                        if hybrid_only:
                            score = _pick_score(dec, ["hybrid_score", "unified_score"])
                        else:
                            score = _pick_score(dec, ["unified_score"])
                        scores_list.append(score)
                    if scores_list:
                        scores_arr = np.asarray(scores_list, dtype=float)
                        self._update_hybrid_auto_tuning(scores_arr, now_ms())

                    # 10분마다 통계 출력
                    if not hasattr(self, '_last_score_stats_log_ms'):
                        self._last_score_stats_log_ms = 0 # 즉시 출력 유도
                    
                    now = now_ms()
                    if (now - self._last_score_stats_log_ms) >= 600_000:  # 10분
                        self._last_score_stats_log_ms = now

                        sym_ev_map = {}
                        for i, dec in enumerate(batch_decisions):
                            score = float(dec.get("unified_score", dec.get("ev", 0.0)) or 0.0)
                            sym_ev_map[ctx_list[i]["symbol"]] = (dec.get("action", "WAIT"), score, dec.get("ev", 0.0))
                            
                        if scores_list:
                            scores_arr = np.asarray(scores_list, dtype=float)
                            
                            # 필터 차단 통계
                            filter_stats = {"unified": 0, "spread": 0, "event_cvar": 0, "event_exit": 0, "both_ev_neg": 0, "gross_ev": 0, "net_expectancy": 0, "dir_gate": 0, "symbol_quality": 0, "liq": 0, "min_notional": 0, "min_exposure": 0, "fee": 0, "top_n": 0, "pre_mc": 0, "cap": 0}
                            for i, dec in enumerate(batch_decisions):
                                fs = dec.get("filter_states", {})
                                if not isinstance(fs, dict):
                                    meta = dec.get("meta", {})
                                    fs = meta.get("filter_states", {})
                                for key in filter_stats:
                                    if fs.get(key) is False:
                                        filter_stats[key] += 1

                            self._log(f"[SCORE_STATS] Distribution (n={len(scores_arr)}): Mean={scores_arr.mean():.6f} | Median={np.median(scores_arr):.6f} | Max={scores_arr.max():.6f}")
                            
                            # 현재 threshold 통과율
                            if hybrid_only:
                                entry_floor = self._get_hybrid_entry_floor()
                                label = "HYBRID_ENTRY_FLOOR"
                            else:
                                entry_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
                                label = "UNIFIED_ENTRY_FLOOR"
                            pass_count = (scores_arr >= entry_floor).sum()
                            pass_rate = (pass_count / len(scores_arr)) * 100
                            self._log(f"  Current {label}={entry_floor:.6f}: {pass_count}/{len(scores_arr)} ({pass_rate:.1f}% pass)")
                            # Auto-tune debug
                            if hybrid_only and (self._hybrid_entry_floor_dyn is not None or self._hybrid_conf_scale_dyn is not None):
                                self._log(f"  HYBRID_AUTO entry_floor={self._hybrid_entry_floor_dyn} conf_scale={self._hybrid_conf_scale_dyn}")
                            
                            # 상세 디버깅 (상위 3개 심볼 상태)
                            sorted_by_score = sorted(sym_ev_map.items(), key=lambda x: x[1][1], reverse=True)[:3]
                            top_debug = [f"{s}:Act={v[0]}/Score={v[1]:.5f}/EV={v[2]:.5f}" for s, v in sorted_by_score]
                            self._log(f"  Top Symbols: {top_debug}")

                if batch_decisions and USE_KELLY_ALLOCATION:
                    try:
                        # 2.5.1 — 모든 심볼의 UnifiedScore 수집
                        sym_score_map: dict[str, float] = {}
                        sym_hold_map: dict[str, float] = {}
                        
                        if hybrid_only:
                            entry_floor = self._get_hybrid_entry_floor()
                        else:
                            entry_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
                        for i, dec in enumerate(batch_decisions):
                            sym = ctx_list[i]["symbol"]
                            if hybrid_only:
                                score_val = _pick_score(dec, ["hybrid_score", "unified_score"])
                            else:
                                score_val = _pick_score(dec, ["unified_score"])
                            sym_score_map[sym] = score_val
                            if hybrid_only:
                                hold_val = _pick_score(dec, ["hybrid_score_hold", "unified_score_hold", "hybrid_score", "unified_score"])
                            else:
                                hold_val = _pick_score(dec, ["unified_score_hold", "unified_score"])
                            sym_hold_map[sym] = float(hold_val if hold_val is not None else score_val)
                        
                        # 2.5.2 — UnifiedScore 기준 순위 정렬 및 TOP N 선택
                        sorted_syms = sorted(sym_score_map.keys(), key=lambda s: sym_score_map[s], reverse=True)
                        self._symbol_scores = sym_score_map.copy()
                        self._symbol_hold_scores = sym_hold_map.copy()
                        self._symbol_ranks = {s: rank + 1 for rank, s in enumerate(sorted_syms)}
                        self._top_n_symbols = sorted_syms[:TOP_N_SYMBOLS]
                        
                        # Always log TOP N selection (critical for debugging)
                        top_info = [(s, f"{sym_score_map[s]:.4f}") for s in self._top_n_symbols]
                        self._log(f"[PORTFOLIO] TOP {TOP_N_SYMBOLS}: {top_info}")
                        
                        # (Logging block moved to pre-check)

                        
                        # 2.5.3 — Kelly 배분 계산 (UnifiedScore 비례 배분)
                        if self._top_n_symbols:
                            n_top = len(self._top_n_symbols)
                            
                            scores = np.array([sym_score_map[s] for s in self._top_n_symbols])
                            self._log(f"[KELLY_DEBUG] Scores for TOP {n_top}: {dict(zip(self._top_n_symbols, scores.tolist()))}")
                            
                            # UnifiedScore 비례 배분 (entry_floor 기준 양수만 반영)
                            scores_adj = scores - float(entry_floor)
                            scores_positive = np.clip(scores_adj, 0, None)
                            # Kelly score transform to amplify relative differences
                            try:
                                mode = str(os.environ.get("KELLY_SCORE_TRANSFORM", "linear")).strip().lower()
                                pos_only = str(os.environ.get("KELLY_SCORE_POSITIVE_ONLY", "1")).strip().lower() in ("1", "true", "yes", "on")
                                hyb_sens = 1.0
                                if hybrid_only:
                                    try:
                                        hyb_sens = float(os.environ.get("KELLY_HYB_SENSITIVITY", 1.0) or 1.0)
                                    except Exception:
                                        hyb_sens = 1.0
                                    hyb_sens = max(0.25, min(hyb_sens, 5.0))
                                base = scores_positive if pos_only else scores_adj.copy()
                                if mode == "power_norm":
                                    gamma = float(os.environ.get("KELLY_SCORE_GAMMA", 2.0) or 2.0) * hyb_sens
                                    max_pos = float(np.max(base)) if base is not None and base.size > 0 else 0.0
                                    if max_pos > 0:
                                        base = (base / max_pos) ** gamma
                                elif mode in ("softmax", "zscore_softmax"):
                                    if pos_only:
                                        mask = base > 0
                                    else:
                                        mask = np.ones_like(base, dtype=bool)
                                    if np.any(mask):
                                        if mode == "zscore_softmax":
                                            std_floor = float(os.environ.get("KELLY_SCORE_STD_FLOOR", 1e-8) or 1e-8)
                                            mu = float(np.mean(base[mask]))
                                            std = float(np.std(base[mask]))
                                            std = max(std, std_floor)
                                            z = (base - mu) / std
                                        else:
                                            z = base
                                        if hyb_sens != 1.0:
                                            z = z * hyb_sens
                                        temp = float(os.environ.get("KELLY_SCORE_TEMP", 1.0) or 1.0)
                                        z = np.clip(z, -20.0, 20.0)
                                        w = np.exp(z / max(temp, 1e-6))
                                        if pos_only:
                                            w = w * mask
                                        base = w
                                else:
                                    base = scores_positive
                                if base is None:
                                    base = scores_positive
                            except Exception:
                                base = scores_positive

                            base = np.where(np.isfinite(base), base, 0.0)
                            total_score = float(np.sum(base))
                            
                            if total_score > 0:
                                kelly_norm = base / total_score
                            else:
                                # 점수가 모두 0이거나 음수면 균등 배분
                                kelly_norm = np.ones(n_top) / n_top
                            
                            self._kelly_allocations = {s: float(kelly_norm[i]) for i, s in enumerate(self._top_n_symbols)}
                            
                            # Always log Kelly allocations (critical for debugging)
                            alloc_info = [(s, f"{self._kelly_allocations[s]:.2%}") for s in self._top_n_symbols]
                            self._log(f"[KELLY] Allocations: {alloc_info}")
                        
                        # 2.5.4 — TOP N에 없는 심볼의 진입 차단 (action을 WAIT로 변경)
                        for i, dec in enumerate(batch_decisions):
                            sym = ctx_list[i]["symbol"]
                            action = dec.get("action", "WAIT")
                            
                            if action in ("LONG", "SHORT") and sym not in self._top_n_symbols:
                                # 현재 포지션이 없는 경우에만 차단 (기존 포지션 청산은 허용)
                                pos = self.positions.get(sym, {})
                                if pos.get("qty", 0) == 0:
                                    dec["action"] = "WAIT"
                                    dec["reason"] = f"NOT_IN_TOP_{TOP_N_SYMBOLS}"
                                    # Always log blocked symbols (critical for debugging)
                                    self._log(f"[PORTFOLIO] {sym} rank={self._symbol_ranks.get(sym)} → WAIT (not in TOP {TOP_N_SYMBOLS})")

                        # Optional: portfolio joint + rebalance pre-sim (background)
                        try:
                            if bool(getattr(mc_config, "portfolio_enabled", False)):
                                if (self._portfolio_joint_task is None) or self._portfolio_joint_task.done():
                                    symbols_for_sim = self._top_n_symbols or list(sym_score_map.keys())
                                    self._portfolio_joint_task = asyncio.create_task(
                                        self._maybe_portfolio_joint(symbols_for_sim, sym_score_map)
                                    )
                        except Exception:
                            pass
                        
                    except Exception as e:
                        import traceback
                        self._log_err(f"[ERR] portfolio_ranking: {e} {traceback.format_exc()}")
                        self._note_runtime_error("portfolio_ranking", str(e))

                # ====== Stage 2.6: Continuous Opportunity Evaluation (Switching Cost) ======
                if batch_decisions and USE_CONTINUOUS_OPPORTUNITY:
                    try:
                        dec_by_sym = {ctx_list[i]["symbol"]: dec for i, dec in enumerate(batch_decisions)}
                        held_syms = []
                        for s, p in self.positions.items():
                            qty = p.get("quantity")
                            if qty is None:
                                qty = p.get("qty")
                            try:
                                qty = float(qty or 0.0)
                            except Exception:
                                qty = 0.0
                            if abs(qty) > 0:
                                held_syms.append(s)

                        # Candidates: not held, actionable, and not blocked by entry filters
                        candidates = []
                        for sym, dec in dec_by_sym.items():
                            if sym in self.positions:
                                continue
                            action = dec.get("action")
                            if action not in ("LONG", "SHORT"):
                                continue
                            meta = dec.get("meta") or {}
                            blocked = meta.get("entry_blocked_filters") or []
                            if blocked:
                                continue
                            if hybrid_only:
                                score = _pick_score(dec, ["hybrid_score", "unified_score"])
                            else:
                                score = _pick_score(dec, ["unified_score"])
                            candidates.append((float(score), sym))

                        if not candidates or not held_syms:
                            pass
                        else:
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            best_score, best_sym = candidates[0]

                            # Find worst current position by expected value proxy
                            worst_sym = None
                            worst_score = None
                            for held_sym in held_syms:
                                pos = self.positions.get(held_sym) or {}
                                if not self._is_managed_position(pos):
                                    continue
                                hold_score = pos.get("hold_ev_tstar")
                                if hold_score is None:
                                    dec = dec_by_sym.get(held_sym) or {}
                                    if hybrid_only:
                                        hold_score = _pick_score(dec, ["hybrid_score_hold", "unified_score_hold", "hybrid_score", "unified_score"])
                                    else:
                                        hold_score = _pick_score(dec, ["unified_score_hold", "unified_score"])
                                if hold_score is None:
                                    hold_score = pos.get("pred_ev", 0.0) or 0.0
                                try:
                                    hold_score = float(hold_score)
                                except Exception:
                                    hold_score = 0.0
                                if (worst_score is None) or (hold_score < worst_score):
                                    worst_score = hold_score
                                    worst_sym = held_sym

                            if worst_sym is not None:
                                try:
                                    switch_margin = float(os.environ.get("SWITCH_EV_MARGIN", 0.0) or 0.0)
                                except Exception:
                                    switch_margin = 0.0
                                if best_score > (worst_score + switch_margin):
                                    if log_this_cycle:
                                        self._log(
                                            f"[SWITCH] {worst_sym}(hold={worst_score:.5f}) → "
                                            f"{best_sym}(new={best_score:.5f}) margin={switch_margin:.5f}"
                                        )
                                    # close worst
                                    for i, dec in enumerate(batch_decisions):
                                        if ctx_list[i]["symbol"] == worst_sym:
                                            dec["action"] = "EXIT"
                                            dec["reason"] = f"SWITCH_TO_{best_sym}"
                                            dec.setdefault("meta", {})["switch_to"] = best_sym
                                            break
                                    # annotate candidate (entry will proceed via normal flow)
                                    cand_dec = dec_by_sym.get(best_sym)
                                    if isinstance(cand_dec, dict):
                                        cand_dec.setdefault("meta", {})["switch_from"] = worst_sym
                                else:
                                    if log_this_cycle:
                                        self._log(
                                            f"[HOLD] {worst_sym}(hold={worst_score:.5f}) kept vs "
                                            f"{best_sym}(new={best_score:.5f})"
                                        )
                    except Exception as e:
                        import traceback
                        self._log_err(f"[ERR] opportunity_eval: {e} {traceback.format_exc()}")
                        self._note_runtime_error("opportunity_eval", str(e))

                # ====== Stage 3: Result Application (per-symbol isolation) ======
                for i, decision in enumerate(batch_decisions):
                    ctx = ctx_list[i]
                    sym = ctx["symbol"]
                    try:
                        row = self._apply_decision(sym, decision, ctx, ts, log_this_cycle)
                        rows.append(row)
                    except Exception as e:
                        import traceback
                        self._log_err(f"[ERR] apply_decision {sym}: {e} {traceback.format_exc()}")
                        self._note_runtime_error(f"apply_decision:{sym}", str(e))
                        rows.append(self._row(sym, ctx.get("price"), ts, None, ctx.get("candles", 0)))

                # Legacy residual leverage correction: step down stale high leverage (e.g. 5x leftovers).
                try:
                    await self._maybe_sync_legacy_5x_deleverage(int(ts))
                except Exception as e:
                    self._log_err(f"[ERR] legacy_delev_sync: {e}")
                    self._note_runtime_error("legacy_delev_sync", str(e))

                # ====== Stage 4: Spread Management & Broadcast (always execute) ======
                try:
                    if self.spread_enabled:
                        self._manage_spreads(ts)
                except Exception as e:
                    self._log_err(f"[ERR] manage_spreads: {e}")
                    self._note_runtime_error("manage_spreads", str(e))
                # Reset cycle rebalance flag
                try:
                    self._force_rebalance_cycle = False
                except Exception:
                    pass
                
                try:
                    self._last_rows = rows
                    if not self._dashboard_fast:
                        await self.broadcast(rows)
                except Exception as e:
                    import traceback
                    self._log_err(f"[ERR] broadcast: {e} {traceback.format_exc()}")
                    self._note_runtime_error("broadcast", str(e))
                    await asyncio.sleep(0.1)

                # Alpha weight persistence/training (offline ridge)
                try:
                    self._maybe_fit_garch_params(ts)
                    self._maybe_persist_alpha_samples(ts)
                    self._maybe_train_alpha_weights(ts)
                except Exception as e:
                    self._log_err(f"[ALPHA_TRAIN] loop error: {e}")

                # Cycle debug dump: record last cycle info for offline inspection
                try:
                    cycle_info = {
                        "ts": ts,
                        "ctx_count": len(ctx_list) if ctx_list is not None else 0,
                        "batch_decisions": len(batch_decisions) if batch_decisions is not None else 0,
                        "rows_sent": len(rows),
                    }
                    with open('/tmp/last_cycle.json', 'w', encoding='utf-8') as _cf:
                        _cf.write(json.dumps(cycle_info))
                except Exception:
                    pass

                # If no contexts were built, throttle the loop to avoid tight spin.
                if not ctx_list:
                    await asyncio.sleep(0.2)

            except Exception as e:
                # Catch-all to prevent the decision loop from dying.
                import traceback
                err_text = f"{e} {traceback.format_exc()}"
                self._log_err(f"[ERR] decision_loop top-level: {err_text}")
                self._note_runtime_error("decision_loop", str(e))
                # Try to send a minimal update to keep dashboard alive
                try:
                    err_row = [self._row('SYSTEM', None, now_ms(), None, 0)]
                    await self.broadcast(err_row)
                except Exception:
                    pass
                await asyncio.sleep(1.0)


async def index_handler(request):
    return web.FileResponse(str(DASHBOARD_FILE))


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    orch: LiveOrchestrator = request.app["orchestrator"]
    orch.clients.add(ws)
    await ws.send_str(json.dumps({"type": "init", "msg": "connected"}, ensure_ascii=False))
    try:
        orch._log(f"[WS_CONNECT] client_id={id(ws)} clients_now={len(orch.clients)}")
        # write a short client-connect snapshot for offline debugging
        try:
            with open('/tmp/ws_clients.json', 'w', encoding='utf-8') as _cf:
                _cf.write(json.dumps({"ts": now_ms(), "clients": len(orch.clients)}, ensure_ascii=False))
        except Exception:
            pass
    except Exception:
        pass

    # If we have a cached last broadcast, send it immediately so UI lights update on connect
    try:
        if getattr(orch, "_last_broadcast_msg", None):
            await ws.send_str(orch._last_broadcast_msg)
            orch._log(f"[WS_CONNECT] sent cached full_update to client_id={id(ws)}")
    except Exception:
        pass

    async for _ in ws:
        pass

    orch.clients.discard(ws)
    try:
        orch._log(f"[WS_DISCONNECT] client_id={id(ws)} clients_now={len(orch.clients)}")
        try:
            with open('/tmp/ws_clients.json', 'w', encoding='utf-8') as _cf:
                _cf.write(json.dumps({"ts": now_ms(), "clients": len(orch.clients)}, ensure_ascii=False))
        except Exception:
            pass
    except Exception:
        pass
    return ws


async def liquidate_all_handler(request):
    orch = request.app.get("orchestrator")
    if orch is None or not hasattr(orch, "_liquidate_all_positions"):
        return web.json_response({"ok": False, "error": "orchestrator_not_ready"}, status=503)
    try:
        orch._liquidate_all_positions()
        pos_count = len(getattr(orch, "positions", {}) or {})
        return web.json_response({"ok": True, "positions": pos_count})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def liquidate_all_safety_handler(request):
    orch = request.app.get("orchestrator")
    if orch is None:
        return web.json_response({"ok": False, "error": "orchestrator_not_ready"}, status=503)
    try:
        if hasattr(orch, "_liquidate_all_positions"):
            orch._liquidate_all_positions()
        # Ensure safety mode even if no positions
        if hasattr(orch, "safety_mode"):
            orch.safety_mode = True
        try:
            if hasattr(orch, "_register_anomaly"):
                orch._register_anomaly("safety_mode", "critical", "dashboard liquidate_all_safety")
        except Exception:
            pass
        try:
            if hasattr(orch, "_log_err"):
                orch._log_err("[RISK] safety_mode ON via dashboard liquidate_all_safety")
        except Exception:
            pass
        pos_count = len(getattr(orch, "positions", {}) or {})
        return web.json_response({"ok": True, "positions": pos_count})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def close_limit_handler(request):
    orch = request.app.get("orchestrator")
    if orch is None:
        return web.json_response({"ok": False, "error": "orchestrator_not_ready"}, status=503)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    sym = (payload.get("symbol") or "").strip()
    price = payload.get("price")
    if not sym or price is None:
        return web.json_response({"ok": False, "error": "symbol_or_price_missing"}, status=400)
    try:
        price_f = float(price)
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_price"}, status=400)
    if price_f <= 0:
        return web.json_response({"ok": False, "error": "invalid_price"}, status=400)
    result = await orch._submit_limit_close(sym, price_f)
    status = 200 if result.get("ok") else 400
    return web.json_response(result, status=status)


async def safety_mode_handler(request):
    orch = request.app.get("orchestrator")
    if orch is None or not hasattr(orch, "clear_safety_mode"):
        return web.json_response({"ok": False, "error": "orchestrator_not_ready"}, status=503)

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return None

    data = {}
    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not data:
        try:
            data = dict(request.rel_url.query)
        except Exception:
            data = {}

    action = str(data.get("action") or "").strip().lower()
    enabled = _as_bool(data.get("enabled"))
    if enabled is True or action in ("on", "enable", "true"):
        return web.json_response({"ok": False, "error": "enable_not_supported"}, status=400)

    reset_equity = _as_bool(data.get("reset_equity"))
    reset_equity = bool(reset_equity) if reset_equity is not None else False
    try:
        result = orch.clear_safety_mode(reset_equity=reset_equity)
        payload = {"ok": True}
        if isinstance(result, dict):
            payload.update(result)
        return web.json_response(payload)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def main():
    ex_cfg = {
        "enableRateLimit": True,
        "timeout": CCXT_TIMEOUT_MS,
    }
    ex_cfg.setdefault("options", {})
    ex_cfg["options"]["adjustForTimeDifference"] = True
    try:
        recv_window = int(os.environ.get("BYBIT_RECV_WINDOW", 15000) or 15000)
    except Exception:
        recv_window = 15000
    recv_window = max(5000, recv_window)
    ex_cfg["options"]["recvWindow"] = recv_window
    ex_cfg["options"]["recv_window"] = recv_window
    try:
        default_type = str(os.environ.get("BYBIT_DEFAULT_TYPE", "swap")).strip().lower()
    except Exception:
        default_type = "swap"
    try:
        default_settle = str(os.environ.get("BYBIT_DEFAULT_SETTLE", "USDT")).strip()
    except Exception:
        default_settle = "USDT"
    if default_type or default_settle:
        if default_type:
            ex_cfg["options"]["defaultType"] = default_type
        if default_settle:
            ex_cfg["options"]["defaultSettle"] = default_settle
    live_enabled = bool(getattr(config, "ENABLE_LIVE_ORDERS", False))
    trade_cfg = dict(ex_cfg)
    if live_enabled:
        trade_cfg.update({
            "apiKey": getattr(config, "API_KEY", "") or "",
            "secret": getattr(config, "API_SECRET", "") or "",
        })
    exchange = ccxt.bybit(trade_cfg)
    data_exchange = ccxt.bybit(ex_cfg)
    if bool(getattr(config, "BYBIT_TESTNET", False)):
        exchange.set_sandbox_mode(True)
        data_exchange.set_sandbox_mode(True)
    # Create a lightweight stub orchestrator so the web server (WS) can accept
    # connections immediately. The real LiveOrchestrator will be created in
    # background and replace `app["orchestrator"]`.
    from types import SimpleNamespace
    orchestrator = SimpleNamespace()
    orchestrator.clients = set()

    runner = None
    site = None
    try:
        # Start the web server immediately so the dashboard can connect quickly.
        app = web.Application()
        app["orchestrator"] = orchestrator
        app.add_routes([
            web.get("/", index_handler),
            web.get("/ws", ws_handler),
            web.post("/api/liquidate_all", liquidate_all_handler),
            web.post("/api/liquidate_all_safety", liquidate_all_safety_handler),
            web.post("/api/close_limit", close_limit_handler),
            web.post("/api/safety_mode", safety_mode_handler),
        ])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, DASHBOARD_HOST, PORT)
        await site.start()

        print(f"🚀 Dashboard: http://{DASHBOARD_HOST}:{PORT}")
        print(f"📄 Serving: {DASHBOARD_FILE.name}")

        # Perform long-running initialization in background so server is responsive.
        async def _background_init():
            try:
                # Instantiate the real orchestrator (this may spawn workers/processes)
                # On macOS shared_memory/process workers can be unstable; prefer
                # in-process EngineHub where possible to avoid startup failures.
                if not bool(getattr(config, "USE_PROCESS_ENGINE_SET", False)):
                    config.USE_PROCESS_ENGINE = False
                real_orch = LiveOrchestrator(exchange, data_exchange=data_exchange)

                # Transfer any connected WS clients from the stub to the real orchestrator
                try:
                    real_orch.clients = orchestrator.clients
                except Exception:
                    pass

                # Replace the app reference so handlers use the real orchestrator
                app["orchestrator"] = real_orch

                # Perform heavy initialization (exchange settings, OHLCV preload)
                await real_orch.init_exchange_settings()
                try:
                    if bool(getattr(config, "SYNC_POSITIONS_ON_START", False)):
                        await real_orch._sync_positions_from_exchange(overwrite=True, reason="startup")
                except Exception as e:
                    try:
                        real_orch._log_err(f"[SYNC_POS] startup sync failed: {e}")
                    except Exception:
                        pass
                await real_orch.preload_all_ohlcv(limit=OHLCV_PRELOAD_LIMIT)
                real_orch._persist_state(force=True)

                # Start periodic loops and decision pipeline on the real orchestrator
                asyncio.create_task(real_orch.fetch_prices_loop())
                asyncio.create_task(real_orch.fetch_ohlcv_loop())
                asyncio.create_task(real_orch.fetch_orderbook_loop())
                asyncio.create_task(real_orch.fetch_balance_loop())
                asyncio.create_task(real_orch.fetch_positions_loop())
                asyncio.create_task(real_orch.bybit_ws_positions_loop())
                asyncio.create_task(real_orch.refresh_top_volume_universe_loop())
                asyncio.create_task(real_orch.hold_eval_loop())
                asyncio.create_task(real_orch.decision_loop())
                asyncio.create_task(real_orch.dashboard_loop())

                print("✅ Background init completed: fetch/decision loops started")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ERR] background_init: {e}\n{tb}")

        # Schedule but don't await: server is responsive immediately
        asyncio.create_task(_background_init())

        # Keep running until cancelled
        await asyncio.Future()
    except OSError as e:
        print(f"[ERR] Failed to bind on port {PORT}: {e}")
    finally:
        try:
            await exchange.close()
            if data_exchange is not exchange:
                await data_exchange.close()
        except Exception:
            pass
        if site is not None:
            try:
                await site.stop()
            except Exception:
                pass
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
