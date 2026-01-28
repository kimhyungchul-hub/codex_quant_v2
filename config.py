import os
from pathlib import Path

from utils.helpers import _env_bool, _env_int, _env_float, _load_env_file, _load_env_file_override

BASE_DIR = Path(__file__).resolve().parent

# Load local `.env` first (developer convenience).
_load_env_file(str(BASE_DIR / ".env"))

# bybit api
API_KEY = "cITWygtNEBO1zxHgXH"
API_SECRET = "AKnWtmPRF59YU5FhNMPjwcSQZHd05cpaYY6R5"
# 환경변수로 자동 주입 (기존 코드 호환)
os.environ["BYBIT_API_KEY"] = API_KEY
os.environ["BYBIT_API_SECRET"] = API_SECRET
# Load runtime overrides early so all config values can be controlled from state/bybit.env.
# NOTE: `state/bybit.env.example` is a template (often with testnet defaults). We only load it when no
# real env file exists (to avoid accidentally enabling testnet when the project is configured via `.env`).
if not (BASE_DIR / "state" / "bybit.env").exists():
    if not (BASE_DIR / ".env").exists():
        _load_env_file(str(BASE_DIR / "state" / "bybit.env.example"))
else:
    # `state/bybit.env` must override `.env` when keys overlap.
    _load_env_file_override(str(BASE_DIR / "state" / "bybit.env"))

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
DASHBOARD_HISTORY_MAX = 50000
DASHBOARD_TRADE_TAPE_MAX = 2000
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
KELLY_COV_LOOKBACK = _env_int("KELLY_COV_LOOKBACK", 100)
KELLY_COV_ESTIMATOR = str(os.environ.get("KELLY_COV_ESTIMATOR", "ledoit_wolf")).strip().lower()
REBALANCE_THRESHOLD_FRAC = 0.02
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

CONSENSUS_THRESHOLD = 0.0005
RSI_PERIOD = 14
RSI_LONG = 60.0
RSI_SHORT = 40.0
SPREAD_LOOKBACK = 60
SPREAD_Z_ENTRY = 2.0
SPREAD_Z_EXIT = 0.5
SPREAD_SIZE_FRAC = 0.02
SPREAD_HOLD_SEC = 600
SPREAD_ENABLED = False
SPREAD_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("SOL/USDT:USDT", "BNB/USDT:USDT"),
]
SPREAD_PCT_MAX = 0.0005
SPREAD_ENTRY_MAX = _env_float("SPREAD_ENTRY_MAX", 0.0)

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
