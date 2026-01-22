"""
StateMixin - 상태 저장/로드 기능
================================

LiveOrchestrator에서 분리된 상태 관리 메서드들.
- JSON 파일 로드/저장 (_load_json, _persist_state)
- 영구 상태 복원 (_load_persistent_state)
- 초기 자산 설정 (_init_initial_equity)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class StateMixin:
    """상태 저장/로드 믹스인"""

    def _load_json(self, path: Path, default: Any) -> Any:
        """
        JSON 파일을 안전하게 로드.
        
        Args:
            path: 파일 경로
            default: 로드 실패 시 반환할 기본값
            
        Returns:
            파싱된 JSON 또는 기본값
        """
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log_err(f"[ERR] load {path.name}: {e}")
        return default

    def _load_persistent_state(self) -> None:
        """
        영구 저장된 상태를 복원.
        
        로드 순서:
        1. balance (잔고)
        2. equity_history (자산 이력)
        3. trade_tape (거래 기록)
        4. eval_history (평가 이력)
        5. positions (포지션)
        """
        from . import now_ms
        
        balance_loaded = False
        
        # 1. balance
        bal = self._load_json(self.state_files.get("balance"), None)
        if isinstance(bal, (int, float)):
            self.balance = float(bal)
            balance_loaded = True

        # 2. equity history
        eq = self._load_json(self.state_files["equity"], [])
        for item in eq:
            try:
                t = int(item.get("time", 0))
                v = float(item.get("equity", 0.0))
                self._equity_history.append({"time": t, "equity": v})
            except Exception:
                continue

        # 3. trade tape
        trades = self._load_json(self.state_files["trade"], [])
        for t in trades:
            self.trade_tape.append(t)

        # 4. eval history
        evals = self._load_json(self.state_files["eval"], [])
        for e in evals:
            self.eval_history.append(e)

        # 5. positions
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
                    p["entry_time"] = int(p.get("entry_time", now_ms()))
                    p["hold_limit"] = int(p.get("hold_limit", self.MAX_POSITION_HOLD_SEC * 1000))
                    self.positions[sym] = p
                except Exception:
                    continue

        # Fallback: 잔고가 로드되지 않았고 포지션이 없으면 마지막 equity 사용
        if (not balance_loaded) and self._equity_history and not self.positions:
            self.balance = float(self._equity_history[-1]["equity"])

    def _init_initial_equity(self) -> None:
        """
        초기 자산(initial_equity) 설정.
        
        드로다운 계산 등에 사용되는 기준 자산.
        """
        if self.initial_equity is not None:
            return
        
        # equity_history에서 마지막 값 사용
        if self._equity_history:
            try:
                self.initial_equity = float(self._equity_history[-1]["equity"])
                return
            except Exception:
                pass
        
        # 없으면 현재 잔고 사용
        self.initial_equity = float(self.balance)

    def _persist_state(self, force: bool = False) -> None:
        """
        현재 상태를 파일로 저장.
        
        Args:
            force: True면 쿨다운 무시하고 즉시 저장
        """
        from . import now_ms
        
        now = now_ms()
        
        # 10초 쿨다운 (강제가 아닌 경우)
        if not force and (now - self._last_state_persist_ms < 10_000):
            return
        
        self._last_state_persist_ms = now
        
        try:
            # equity history
            with self.state_files["equity"].open("w", encoding="utf-8") as f:
                json.dump(list(self._equity_history), f, ensure_ascii=False)
            
            # trade tape
            with self.state_files["trade"].open("w", encoding="utf-8") as f:
                json.dump(list(self.trade_tape), f, ensure_ascii=False)
            
            # eval history
            with self.state_files["eval"].open("w", encoding="utf-8") as f:
                json.dump(list(self.eval_history), f, ensure_ascii=False)
            
            # positions
            with self.state_files["positions"].open("w", encoding="utf-8") as f:
                json.dump(list(self.positions.values()), f, ensure_ascii=False)
            
            # balance
            with self.state_files["balance"].open("w", encoding="utf-8") as f:
                json.dump(self.balance, f, ensure_ascii=False)
                
        except Exception as e:
            self._log_err(f"[ERR] persist state: {e}")

    def _mark_exit_and_cooldown(self, sym: str, exit_kind: str, ts_ms: int) -> None:
        """
        포지션 종료 후 재진입 쿨다운 설정.
        
        Args:
            sym: 심볼
            exit_kind: 종료 유형 ("RISK", "KILL", "MANUAL" 등)
            ts_ms: 현재 타임스탬프 (ms)
        """
        import time
        
        self._last_exit_kind[sym] = exit_kind
        
        # 종료 유형에 따른 쿨다운 시간 결정
        cooldown_map = {
            "RISK": self.COOLDOWN_SEC * 2,
            "KILL": self.COOLDOWN_SEC * 3,
        }
        cooldown_sec = cooldown_map.get(exit_kind, self.COOLDOWN_SEC)
        
        self._cooldown_until[sym] = time.time() + cooldown_sec
        self._entry_streak[sym] = 0
