import os
from pathlib import Path

from utils.helpers import _env_bool, _env_int, _env_float, _load_env_file, _load_env_file_override

BASE_DIR = Path(__file__).resolve().parent

ENV_SOURCES = []

def _record_env(path: str, mode: str, loaded: bool) -> None:
    ENV_SOURCES.append({
        "path": str(path),
        "mode": str(mode),
        "loaded": bool(loaded),
    })

# Load local `.env` first (developer convenience).
_base_env_path = str(BASE_DIR / ".env")
_base_loaded = _load_env_file(_base_env_path)
_record_env(_base_env_path, "base", _base_loaded)

# Optional profile env support (e.g. .env.scalp or .env.midterm).
# Priority: ENV_FILE (explicit path) > ENV_PROFILE (.env.<profile>) > fallback to .env.scalp if no .env exists.
_env_file = str(os.environ.get("ENV_FILE", "")).strip()
_env_profile = str(os.environ.get("ENV_PROFILE", "")).strip()
if not _env_profile:
    _env_profile = str(os.environ.get("ENVPROFILE", "")).strip()
if not _env_profile:
    _profile_file = BASE_DIR / ".env.profile"
    try:
        if _profile_file.exists():
            _env_profile = _profile_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except Exception:
        _env_profile = ""
if _env_file:
    _override_loaded = _load_env_file_override(_env_file)
    _record_env(_env_file, "override_env_file", _override_loaded)
elif _env_profile:
    _profile_path = str(BASE_DIR / f".env.{_env_profile}")
    _profile_loaded = _load_env_file_override(_profile_path)
    _record_env(_profile_path, "override_env_profile", _profile_loaded)
elif not (BASE_DIR / ".env").exists():
    _fallback_path = str(BASE_DIR / ".env.scalp")
    _fallback_loaded = _load_env_file(_fallback_path)
    _record_env(_fallback_path, "fallback_scalp", _fallback_loaded)

# bybit api (loaded from env or state/bybit.env)
def _load_api_keys() -> tuple[str, str]:
    api_key = os.environ.get("BYBIT_API_KEY") or os.environ.get("API_KEY") or ""
    api_secret = os.environ.get("BYBIT_API_SECRET") or os.environ.get("API_SECRET") or ""
    return api_key, api_secret

API_KEY, API_SECRET = _load_api_keys()
# Load runtime overrides from state/bybit.env only when running with the bybit profile.
_use_bybit_env = False
if _env_profile and _env_profile.strip().lower() == "bybit":
    _use_bybit_env = True
if _env_file and Path(_env_file).name == ".env.bybit":
    _use_bybit_env = True
if str(os.environ.get("USE_BYBIT_ENV", "")).strip().lower() in ("1", "true", "yes", "on"):
    _use_bybit_env = True

# NOTE: `state/bybit.env.example` is a template (often with testnet defaults).
if _use_bybit_env:
    if not (BASE_DIR / "state" / "bybit.env").exists():
        if not (BASE_DIR / ".env").exists():
            _bybit_example = str(BASE_DIR / "state" / "bybit.env.example")
            _bybit_loaded = _load_env_file(_bybit_example)
            _record_env(_bybit_example, "bybit_example", _bybit_loaded)
    else:
        # `state/bybit.env` must override `.env` when keys overlap.
        _bybit_path = str(BASE_DIR / "state" / "bybit.env")
        _bybit_loaded = _load_env_file_override(_bybit_path)
        _record_env(_bybit_path, "bybit_override", _bybit_loaded)

# Refresh API keys after any env override
API_KEY, API_SECRET = _load_api_keys()

ENV_PROFILE = _env_profile
ENV_FILE = _env_file
ENV_ACTIVE = None
for src in reversed(ENV_SOURCES):
    if src.get("loaded"):
        ENV_ACTIVE = src.get("path")
        break

def _env_symbols(defaults: list[str]) -> list[str]:
    raw = str(os.environ.get("SYMBOLS_CSV", "")).strip()
    if not raw:
        return list(defaults)
    parts = [p.strip() for p in raw.split(",")]
    syms = [p for p in parts if p]
    return syms or list(defaults)

# -------------------------------------------------------------------
# Constants & Configuration
# -------------------------------------------------------------------
PORT = 9999
DASHBOARD_HISTORY_MAX = _env_int("DASHBOARD_HISTORY_MAX", 3000)
DASHBOARD_TRADE_TAPE_MAX = _env_int("DASHBOARD_TRADE_TAPE_MAX", 500)
DASHBOARD_INCLUDE_DETAILS = _env_bool("DASHBOARD_INCLUDE_DETAILS", False)

DECISION_REFRESH_SEC = _env_float("DECISION_REFRESH_SEC", 2.0)
DECISION_EVAL_MIN_INTERVAL_SEC = _env_float("DECISION_EVAL_MIN_INTERVAL_SEC", DECISION_REFRESH_SEC)
DECISION_WORKER_SLEEP_SEC = _env_float("DECISION_WORKER_SLEEP_SEC", 0.0)
DECISION_MIN_CANDLES = _env_int("DECISION_MIN_CANDLES", 20)
DECISION_MAX_INFLIGHT = _env_int("DECISION_MAX_INFLIGHT", 10)
LOG_STDOUT = _env_bool("LOG_STDOUT", False)
DEBUG_MU_SIGMA = _env_bool("DEBUG_MU_SIGMA", False)
DEBUG_TPSL_META = _env_bool("DEBUG_TPSL_META", False)
DEBUG_ROW = _env_bool("DEBUG_ROW", False)
DECISION_LOG_EVERY = _env_int("DECISION_LOG_EVERY", 10)
MC_STREAMING_MODE = _env_bool("MC_STREAMING_MODE", False)
MC_STREAMING_BATCH_SIZE = _env_int("MC_STREAMING_BATCH_SIZE", 4)
MC_STREAMING_BROADCAST = _env_bool("MC_STREAMING_BROADCAST", True)
MAINT_MARGIN_RATE = _env_float("MAINT_MARGIN_RATE", 0.005)
LIQUIDATION_BUFFER = _env_float("LIQUIDATION_BUFFER", 0.0025)
LIQUIDATION_SCORE_PENALTY = _env_float("LIQUIDATION_SCORE_PENALTY", 0.01)
LIVE_LIQUIDATION_SYNC_SEC = _env_float("LIVE_LIQUIDATION_SYNC_SEC", 10.0)

# Tick-level microstructure settings (short-term direction/volatility)
TICK_BUFFER_SEC = _env_float("TICK_BUFFER_SEC", 300.0)
TICK_LOOKBACK_SEC = _env_float("TICK_LOOKBACK_SEC", 30.0)
TICK_MIN_SAMPLES = _env_int("TICK_MIN_SAMPLES", 8)
TICK_BREAKOUT_LOOKBACK_SEC = _env_float("TICK_BREAKOUT_LOOKBACK_SEC", 60.0)
TICK_BREAKOUT_BUFFER_PCT = _env_float("TICK_BREAKOUT_BUFFER_PCT", 0.0004)
MIN_TICK_VOL = _env_float("MIN_TICK_VOL", 0.0)

# ✅ DEV_MODE에서는 n_paths를 줄여서 빠른 피드백 제공
_DEV_MODE = _env_bool("DEV_MODE", False)
_DEFAULT_N_PATHS_LIVE = 1024 if _DEV_MODE else 16384
_DEFAULT_N_PATHS_EXIT = 512 if _DEV_MODE else 4096
MC_N_PATHS_LIVE = _env_int("MC_N_PATHS_LIVE", _DEFAULT_N_PATHS_LIVE)
MC_N_PATHS_EXIT = _env_int("MC_N_PATHS_EXIT", _DEFAULT_N_PATHS_EXIT)
MC_TAIL_MODE = str(os.environ.get("MC_TAIL_MODE", "student_t")).strip().lower() or "student_t"
MC_STUDENT_T_DF = _env_float("MC_STUDENT_T_DF", 6.0)
DECIDE_BATCH_TIMEOUT_SEC = _env_float("DECIDE_BATCH_TIMEOUT_SEC", 5.0)

SYMBOLS = _env_symbols([
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT",
    "NEAR/USDT:USDT", "UNI/USDT:USDT", "APT/USDT:USDT", "SUI/USDT:USDT",
    "SAND/USDT:USDT", "LDO/USDT:USDT", "CRV/USDT:USDT", "HBAR/USDT:USDT",
    "ICP/USDT:USDT", "GMT/USDT:USDT",
])
TEMPLATE_DIR = BASE_DIR / "templates"
DASHBOARD_FILE = BASE_DIR / "dashboard_v2.html"


# OHLCV settings
TIMEFRAME = "1m"
OHLCV_PRELOAD_LIMIT = 240
OHLCV_REFRESH_LIMIT = 2
OHLCV_SLEEP_SEC = _env_float("OHLCV_SLEEP_SEC", 30.0)
PRELOAD_ON_START = _env_bool("PRELOAD_ON_START", True)

# Orderbook settings
ORDERBOOK_DEPTH = 5
ORDERBOOK_SLEEP_SEC = _env_float("ORDERBOOK_SLEEP_SEC", 2.0)
ORDERBOOK_SYMBOL_INTERVAL_SEC = _env_float("ORDERBOOK_SYMBOL_INTERVAL_SEC", 0.35)
TICKER_SLEEP_SEC = _env_float("TICKER_SLEEP_SEC", 1.0)

# Networking / Retry
CCXT_TIMEOUT_MS = 20000
MAX_RETRY = 4
RETRY_BASE_SEC = 0.5
MAX_INFLIGHT_REQ = 20

# Risk / Execution settings
ENABLE_LIVE_ORDERS = _env_bool("ENABLE_LIVE_ORDERS", False)
DEFAULT_LEVERAGE = _env_float("DEFAULT_LEVERAGE", 5.0)
MAX_LEVERAGE = _env_float("MAX_LEVERAGE", 100.0)
DEFAULT_SIZE_FRAC = _env_float("DEFAULT_SIZE_FRAC", 0.05)
MAX_POSITION_HOLD_SEC = _env_int("MAX_POSITION_HOLD_SEC", 600)
POSITION_HOLD_MIN_SEC = _env_int("POSITION_HOLD_MIN_SEC", 120)
POSITION_HOLD_HARD_CAP_SEC = _env_int("POSITION_HOLD_HARD_CAP_SEC", max(1800, int(MAX_POSITION_HOLD_SEC)))
POSITION_CAP_ENABLED = _env_bool("POSITION_CAP_ENABLED", False)
EXPOSURE_CAP_ENABLED = _env_bool("EXPOSURE_CAP_ENABLED", True)
MAX_CONCURRENT_POSITIONS = _env_int("MAX_CONCURRENT_POSITIONS", 99999)
MAX_NOTIONAL_EXPOSURE = _env_float("MAX_NOTIONAL_EXPOSURE", 10.0)
LIVE_MAX_NOTIONAL_EXPOSURE = _env_float("LIVE_MAX_NOTIONAL_EXPOSURE", 10.0)
RESET_INITIAL_EQUITY_ON_START = _env_bool("RESET_INITIAL_EQUITY_ON_START", False)
SYNC_POSITIONS_ON_START = _env_bool("SYNC_POSITIONS_ON_START", True)
SYNC_POSITIONS_ALL = _env_bool("SYNC_POSITIONS_ALL", True)
SYNC_POSITIONS_POLL = _env_bool("SYNC_POSITIONS_POLL", True)
SYNC_POSITIONS_POLL_SEC = _env_float("SYNC_POSITIONS_POLL_SEC", 30.0)
LIVE_PENDING_SYNC_GRACE_SEC = _env_float("LIVE_PENDING_SYNC_GRACE_SEC", 30.0)
MANAGE_SYNCED_POSITIONS = _env_bool("MANAGE_SYNCED_POSITIONS", True)
KELLY_COV_LOOKBACK = _env_int("KELLY_COV_LOOKBACK", 100)
KELLY_COV_ESTIMATOR = str(os.environ.get("KELLY_COV_ESTIMATOR", "ledoit_wolf")).strip().lower()
REBALANCE_THRESHOLD_FRAC = _env_float("REBALANCE_THRESHOLD_FRAC", 0.02)
REBALANCE_MIN_INTERVAL_SEC = _env_float("REBALANCE_MIN_INTERVAL_SEC", 0.0)
REBALANCE_MIN_NOTIONAL = _env_float("REBALANCE_MIN_NOTIONAL", 0.0)
EV_DROP_THRESHOLD = 0.0003
K_LEV = 2000.0
EV_EXIT_FLOOR = {"bull": -0.0003, "bear": -0.0003, "chop": -0.0002, "volatile": -0.0002}
EV_DROP = {"bull": 0.0010, "bear": 0.0010, "chop": 0.0008, "volatile": 0.0008}
PSL_RISE = {"bull": 0.05, "bear": 0.05, "chop": 0.03, "volatile": 0.03}
DEFAULT_TP_PCT = _env_float("DEFAULT_TP_PCT", 0.006)
DEFAULT_SL_PCT = _env_float("DEFAULT_SL_PCT", 0.005)

EXEC_MODE = str(os.environ.get("EXEC_MODE", "maker_then_market")).strip().lower()
USE_MAKER_ORDERS = _env_bool("USE_MAKER_ORDERS", True)
MAKER_TIMEOUT_MS = _env_int("MAKER_TIMEOUT_MS", 1500)
MAKER_RETRIES = _env_int("MAKER_RETRIES", 2)
MAKER_POLL_MS = _env_int("MAKER_POLL_MS", 200)
POLICY_SCORE_TRAILING_FACTOR = _env_float("POLICY_SCORE_TRAILING_FACTOR", 0.6)
POLICY_SCORE_FLIP_MARGIN = _env_float("POLICY_SCORE_FLIP_MARGIN", 0.001)
UNIFIED_RISK_LAMBDA = _env_float("UNIFIED_RISK_LAMBDA", 1.0)
UNIFIED_RHO = _env_float("UNIFIED_RHO", 0.0)
EVENT_EXIT_SCORE = _env_float("EVENT_EXIT_SCORE", -0.0005)
EVENT_EXIT_MAX_P_SL = _env_float("EVENT_EXIT_MAX_P_SL", 0.55)
EVENT_EXIT_MAX_ABS_CVAR = _env_float("EVENT_EXIT_MAX_ABS_CVAR", 0.010)

CONSENSUS_THRESHOLD = 0.0005
RSI_PERIOD = 14
RSI_LONG = 60.0
RSI_SHORT = 40.0
SPREAD_LOOKBACK = 60
SPREAD_Z_ENTRY = 2.0
SPREAD_Z_EXIT = 0.5
SPREAD_SIZE_FRAC = 0.02
SPREAD_HOLD_SEC = 600
SPREAD_ENABLED = _env_bool("SPREAD_ENABLED", False)
SPREAD_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("SOL/USDT:USDT", "BNB/USDT:USDT"),
]
SPREAD_PCT_MAX = _env_float("SPREAD_PCT_MAX", 0.0005)
SPREAD_ENTRY_MAX = _env_float("SPREAD_ENTRY_MAX", 0.0)
MIN_ENTRY_SCORE = _env_float("MIN_ENTRY_SCORE", 0.0)
MIN_ENTRY_NOTIONAL = _env_float("MIN_ENTRY_NOTIONAL", 0.0)
MIN_ENTRY_NOTIONAL_PCT = _env_float("MIN_ENTRY_NOTIONAL_PCT", 0.0)
MIN_ENTRY_EXPOSURE_PCT = _env_float("MIN_ENTRY_EXPOSURE_PCT", 0.0)
MIN_LIQ_SCORE = _env_float("MIN_LIQ_SCORE", 0.0)
PRE_MC_ENABLED = _env_bool("PRE_MC_ENABLED", False)
PRE_MC_BLOCK_ON_FAIL = _env_bool("PRE_MC_BLOCK_ON_FAIL", True)
PRE_MC_MIN_EXPECTED_PNL = _env_float("PRE_MC_MIN_EXPECTED_PNL", 0.0)
PRE_MC_MIN_CVAR = _env_float("PRE_MC_MIN_CVAR", -0.05)
PRE_MC_MAX_LIQ_PROB = _env_float("PRE_MC_MAX_LIQ_PROB", 0.05)
PRE_MC_SIZE_SCALE = _env_float("PRE_MC_SIZE_SCALE", 0.5)

BYBIT_TAKER_FEE = 0.0006
BYBIT_MAKER_FEE = 0.0001

EV_TUNE_WINDOW_SEC = 30 * 60
EV_TUNE_PCTL = 95
EV_TUNE_MIN_SAMPLES = 40
EV_ENTER_FLOOR_MIN = 0.0008
EV_ENTER_FLOOR_MAX = 0.0025
EV_ENTRY_THRESHOLD = _env_float("EV_ENTRY_THRESHOLD", 0.002)

COOLDOWN_SEC = _env_int("COOLDOWN_SEC", 60)
ENTRY_STREAK_MIN = 1
COOLDOWN_TP_SEC = 30
COOLDOWN_RISK_SEC = 120

# Portfolio / ranking
TOP_N_SYMBOLS = _env_int("TOP_N_SYMBOLS", 4)
USE_KELLY_ALLOCATION = _env_bool("USE_KELLY_ALLOCATION", True)
USE_CONTINUOUS_OPPORTUNITY = _env_bool("USE_CONTINUOUS_OPPORTUNITY", True)
SWITCHING_COST_MULT = _env_float("SWITCHING_COST_MULT", 2.0)
PORTFOLIO_TOP_N = _env_int("PORTFOLIO_TOP_N", 4)
PORTFOLIO_SWITCH_COST_MULT = _env_float("PORTFOLIO_SWITCH_COST_MULT", 1.2)
PORTFOLIO_KELLY_CAP = _env_float("PORTFOLIO_KELLY_CAP", 5.0)
PORTFOLIO_JOINT_INTERVAL_SEC = _env_float("PORTFOLIO_JOINT_INTERVAL_SEC", 15.0)
PORTFOLIO_JOINT_SIM_ENABLED = _env_bool("PORTFOLIO_JOINT_SIM_ENABLED", False)

# Engine routing / runtime
USE_REMOTE_ENGINE = _env_bool("USE_REMOTE_ENGINE", False)
ENGINE_SERVER_URL = str(os.environ.get("ENGINE_SERVER_URL", "http://localhost:8000")).strip()
USE_PROCESS_ENGINE = _env_bool("USE_PROCESS_ENGINE", True)
USE_PROCESS_ENGINE_SET = "USE_PROCESS_ENGINE" in os.environ
MC_ENGINE_CPU_AFFINITY = str(os.environ.get("MC_ENGINE_CPU_AFFINITY", "")).strip()
MC_ENGINE_SHM_SLOTS = _env_int("MC_ENGINE_SHM_SLOTS", 32)
MC_ENGINE_SHM_SLOT_SIZE = _env_int("MC_ENGINE_SHM_SLOT_SIZE", 131072)
MC_ENGINE_TIMEOUT_SINGLE = _env_float("MC_ENGINE_TIMEOUT_SINGLE", 2.0)
MC_ENGINE_TIMEOUT_BATCH = _env_float("MC_ENGINE_TIMEOUT_BATCH", 10.0)

# Notifications
TELEGRAM_BOT_TOKEN = str(os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
TELEGRAM_CHAT_ID = str(os.environ.get("TELEGRAM_CHAT_ID", "")).strip()

# Exchange mode
BYBIT_TESTNET = _env_bool("BYBIT_TESTNET", False)
DATA_BYBIT_TESTNET = _env_bool("DATA_BYBIT_TESTNET", False)

# Live sync
LIVE_SYNC_SLEEP_SEC = _env_float("LIVE_SYNC_SLEEP_SEC", 2.0)

# Paper trading controls
PAPER_TRADING = _env_bool("PAPER_TRADING", True)
PAPER_FLAT_ON_WAIT = _env_bool("PAPER_FLAT_ON_WAIT", True)
PAPER_USE_ENGINE_SIZING = _env_bool("PAPER_USE_ENGINE_SIZING", True)
PAPER_ENGINE_SIZE_MULT = _env_float("PAPER_ENGINE_SIZE_MULT", 1.0)
PAPER_ENGINE_SIZE_MIN_FRAC = _env_float("PAPER_ENGINE_SIZE_MIN_FRAC", 0.005)
PAPER_ENGINE_SIZE_MAX_FRAC = _env_float("PAPER_ENGINE_SIZE_MAX_FRAC", 0.20)
PAPER_SIZE_FRAC = _env_float("PAPER_SIZE_FRAC", DEFAULT_SIZE_FRAC)
PAPER_LEVERAGE = _env_float("PAPER_LEVERAGE", DEFAULT_LEVERAGE)
PAPER_FEE_ROUNDTRIP = _env_float("PAPER_FEE_ROUNDTRIP", 0.0)
PAPER_SLIPPAGE_BPS = _env_float("PAPER_SLIPPAGE_BPS", 0.0)
PAPER_MIN_HOLD_SEC = _env_int("PAPER_MIN_HOLD_SEC", POSITION_HOLD_MIN_SEC)
PAPER_MAX_HOLD_SEC = _env_int("PAPER_MAX_HOLD_SEC", MAX_POSITION_HOLD_SEC)
PAPER_MAX_POSITIONS = _env_int("PAPER_MAX_POSITIONS", MAX_CONCURRENT_POSITIONS)
PAPER_EXIT_POLICY_ONLY = _env_bool("PAPER_EXIT_POLICY_ONLY", True)
PAPER_EXIT_POLICY_HORIZON_SEC = _env_int("PAPER_EXIT_POLICY_HORIZON_SEC", 1800)
PAPER_EXIT_POLICY_MIN_HOLD_SEC = _env_int("PAPER_EXIT_POLICY_MIN_HOLD_SEC", 180)
PAPER_EXIT_POLICY_DECISION_DT_SEC = _env_int("PAPER_EXIT_POLICY_DECISION_DT_SEC", 5)
PAPER_EXIT_POLICY_FLIP_CONFIRM_TICKS = _env_int("PAPER_EXIT_POLICY_FLIP_CONFIRM_TICKS", 3)
PAPER_EXIT_POLICY_HOLD_BAD_TICKS = _env_int("PAPER_EXIT_POLICY_HOLD_BAD_TICKS", 3)
PAPER_EXIT_POLICY_SCORE_MARGIN = _env_float("PAPER_EXIT_POLICY_SCORE_MARGIN", 0.0001)
PAPER_EXIT_POLICY_SOFT_FLOOR = _env_float("PAPER_EXIT_POLICY_SOFT_FLOOR", -0.001)
PAPER_EXIT_POLICY_P_POS_ENTER_FLOOR = _env_float("PAPER_EXIT_POLICY_P_POS_ENTER_FLOOR", 0.52)
PAPER_EXIT_POLICY_P_POS_HOLD_FLOOR = _env_float("PAPER_EXIT_POLICY_P_POS_HOLD_FLOOR", 0.50)
PAPER_EXIT_POLICY_DD_STOP_ENABLED = _env_bool("PAPER_EXIT_POLICY_DD_STOP_ENABLED", True)
PAPER_EXIT_POLICY_DD_STOP_ROE = _env_float("PAPER_EXIT_POLICY_DD_STOP_ROE", -0.02)
