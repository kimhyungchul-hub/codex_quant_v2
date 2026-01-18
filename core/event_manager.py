from __future__ import annotations
import asyncio
import json
import time
import math
import random
import os
import numpy as np
from pathlib import Path
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
from engines.engine_hub import EngineHub
from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
from utils.alpha_features import build_alpha_features
from engines.simulation_methods import mc_first_passage_tp_sl_jax
from engines.mc_risk import compute_cvar, kelly_with_cvar, PyramidTracker, ExitPolicy, should_exit_position
from regime import adjust_mu_sigma, time_regime, get_regime_mu_sigma
from engines.running_stats import RunningStats
from engines.mc.monte_carlo_engine import MonteCarloEngine
from config import *
from core.dashboard_server import DashboardServer
from core.data_manager import DataManager
from utils.helpers import now_ms, _safe_float, _sanitize_for_json, _calc_rsi, _load_env_file, _env_bool, _env_int, _env_float
from engines.pmaker_manager import PMakerManager

class EventManager:
    # 이벤트 관련 로직
    pass
