from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from utils.helpers import now_ms


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class StateFiles:
    equity: Path
    trade: Path
    positions: Path
    balance: Path


class StateStore:
    """Best-effort persistence for orchestrator state.

    Keeps file IO isolated from the orchestrator so runtime/trading logic can be refactored independently.
    """

    def __init__(
        self,
        *,
        state_dir: Path,
        log: Optional[LogFn] = None,
        log_err: Optional[LogFn] = None,
        trade_tape_maxlen: int = 20_000,
        persist_min_interval_ms: int = 5_000,
        stale_position_threshold_ms: int = 24 * 3600 * 1000,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

        self.files = StateFiles(
            equity=self.state_dir / "equity_history.json",
            trade=self.state_dir / "trade_tape.json",
            positions=self.state_dir / "positions.json",
            balance=self.state_dir / "balance.json",
        )

        self._log = log
        self._log_err = log_err

        self.trade_tape_maxlen = int(trade_tape_maxlen)
        self.persist_min_interval_ms = int(persist_min_interval_ms)
        self.stale_position_threshold_ms = int(stale_position_threshold_ms)

        self._last_persist_ms = 0

    def _info(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _err(self, msg: str) -> None:
        if self._log_err is not None:
            self._log_err(msg)

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._err(f"[ERR] load {path.name}: {e}")
        return default

    def load(self) -> Tuple[Optional[float], Dict[str, Dict[str, Any]], Deque[Dict[str, Any]]]:
        """Return (balance, positions, trade_tape). Missing files yield defaults."""
        balance: Optional[float] = None
        positions: Dict[str, Dict[str, Any]] = {}
        trade_tape: Deque[Dict[str, Any]] = deque(maxlen=self.trade_tape_maxlen)

        # balance
        try:
            val = self._load_json(self.files.balance, None)
            if val is not None:
                balance = float(val)
        except Exception:
            balance = None

        # positions (with stale check)
        try:
            pos_data = self._load_json(self.files.positions, {})
            if isinstance(pos_data, list):
                pos_data = {}

            if isinstance(pos_data, dict) and pos_data:
                now = now_ms()
                valid_positions: Dict[str, Dict[str, Any]] = {}
                discarded = 0
                for sym, p in pos_data.items():
                    if not isinstance(p, dict):
                        continue
                    try:
                        entry_time = int(p.get("time") or p.get("entry_time") or 0)
                    except Exception:
                        entry_time = 0
                    if entry_time and (now - entry_time > self.stale_position_threshold_ms):
                        discarded += 1
                        continue
                    valid_positions[str(sym)] = p

                if discarded > 0:
                    self._info(f"[WARN] Discarded {discarded} stale positions (>24h old) from state.")
                positions = valid_positions
        except Exception:
            positions = {}

        # trade tape
        try:
            tape = self._load_json(self.files.trade, [])
            if isinstance(tape, list):
                trade_tape = deque(tape, maxlen=self.trade_tape_maxlen)
        except Exception:
            pass

        return balance, positions, trade_tape

    def persist(self, *, balance: float, positions: Dict[str, Any], trade_tape: Deque[Any], force: bool = False) -> None:
        ts = now_ms()
        if (not force) and self.persist_min_interval_ms > 0:
            if ts - int(self._last_persist_ms) < int(self.persist_min_interval_ms):
                return
        self._last_persist_ms = int(ts)

        # JSON persistence has been disabled in favor of SQLite-backed persistence.
        # This method intentionally performs no file writes to avoid dual-writing
        # and potential mode-mixing between paper/live storage.
        self._info("[STATE_STORE] JSON writes disabled (DB-only mode)")
        return

    def reset(self, *, initial_balance: float = 10_000.0) -> None:
        """Reset persisted state to a fresh paper baseline."""
        # No-op: we intentionally do not write JSON files when resetting state.
        # Downstream systems should use the DatabaseManager for persistence.
        self._info("[STATE_STORE] reset skipped — JSON writes disabled (DB-only mode)")
        return
