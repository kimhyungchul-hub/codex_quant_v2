"""
DashboardMixin - 대시보드 기능
==============================

LiveOrchestrator에서 분리된 대시보드/모니터링 메서드들.
- 행 데이터 생성 (_row)
- 포트폴리오 계산 (_compute_portfolio)
- 평가 메트릭 계산 (_compute_eval_metrics)
- 브로드캐스트 (broadcast)
"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass


class DashboardMixin:
    """대시보드 믹스인"""

    def _row(
        self,
        sym: str,
        decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        대시보드 테이블용 행 데이터 생성.
        
        Args:
            sym: 심볼
            decision: 엔진 결정 결과 (있으면 사용)
            
        Returns:
            대시보드 행 dict
        """
        price = self._latest_prices.get(sym, 0.0)
        pos = self.positions.get(sym)
        
        # 기본 값
        row = {
            "symbol": sym,
            "price": price,
            "action": "HOLD",
            "ev": 0.0,
            "evph_p": 0.0,
            "ev_score_p": 0.0,
            "napv_p": 0.0,
            "evp": 0.0,
            "unified_score": 0.0,
            "unified_score_long": 0.0,
            "unified_score_short": 0.0,
            "unified_score_hold": 0.0,
            "unified_t_star": 0.0,
            "rank": 0,
            "hold_mean_sec": 0.0,
            "optimal_horizon_sec": 0.0,
            "cap_frac": 0.0,
            "leverage": self.leverage,
            "in_position": pos is not None,
            "position_side": None,
            "position_pnl": 0.0,
            "position_pnl_pct": 0.0,
            "hold_sec": 0.0,
        }
        
        # 결정 정보 병합
        if decision:
            row["action"] = decision.get("action", "HOLD")
            row["ev"] = decision.get("ev", 0.0)
            row["evph_p"] = decision.get("evph_p", 0.0)
            row["ev_score_p"] = decision.get("ev_score_p", 0.0)
            row["napv_p"] = decision.get("napv_p", 0.0)
            row["evp"] = decision.get("evp", 0.0)
            row["unified_score"] = decision.get("unified_score", decision.get("ev", 0.0))
            row["unified_score_long"] = decision.get("unified_score_long", 0.0)
            row["unified_score_short"] = decision.get("unified_score_short", 0.0)
            row["unified_score_hold"] = decision.get("unified_score_hold", 0.0)
            row["unified_t_star"] = decision.get("unified_t_star", 0.0)
            row["rank"] = decision.get("rank", 0)
            row["hold_mean_sec"] = decision.get("hold_mean_sec", 0.0)
            row["optimal_horizon_sec"] = decision.get("optimal_horizon_sec", 0.0)
            row["cap_frac"] = decision.get("cap_frac", 0.0)
            row["leverage"] = decision.get("leverage", self.leverage)
        
        # 포지션 정보
        if pos:
            row["position_side"] = pos.get("action")
            entry_price = float(pos.get("entry_price", price))
            qty = float(pos.get("quantity", 0.0))
            action = pos.get("action", "BUY")
            entry_time = int(pos.get("entry_time", 0))
            
            if action == "BUY":
                pnl = (price - entry_price) * qty
            else:
                pnl = (entry_price - price) * qty
            
            pnl_pct = (price - entry_price) / entry_price if entry_price else 0.0
            if action == "SELL":
                pnl_pct = -pnl_pct
            
            row["position_pnl"] = pnl
            row["position_pnl_pct"] = pnl_pct
            row["hold_sec"] = (int(time.time() * 1000) - entry_time) / 1000
        
        return row

    def _compute_portfolio(self) -> Dict[str, Any]:
        """
        포트폴리오 요약 계산.
        
        Returns:
            포트폴리오 요약 dict
        """
        unrealized = self._unrealized_pnl()
        equity = self.balance + unrealized
        
        # 드로다운
        initial = self.initial_equity or equity
        drawdown = (initial - equity) / initial if initial > 0 else 0.0
        
        # 오늘 PnL (단순화: 마지막 equity_history와 비교)
        today_pnl = 0.0
        if self._equity_history:
            last_eq = float(self._equity_history[-1].get("equity", equity))
            today_pnl = equity - last_eq
        
        # 포지션 요약
        long_count = sum(1 for p in self.positions.values() if p.get("action") == "BUY")
        short_count = sum(1 for p in self.positions.values() if p.get("action") == "SELL")
        total_notional = self._total_open_notional()
        
        return {
            "balance": self.balance,
            "equity": equity,
            "unrealized_pnl": unrealized,
            "today_pnl": today_pnl,
            "drawdown": drawdown,
            "position_count": len(self.positions),
            "long_count": long_count,
            "short_count": short_count,
            "total_notional": total_notional,
            "capital_util": total_notional / equity if equity > 0 else 0.0,
            "kill_switch": self._kill_switch_on,
        }

    def _compute_eval_metrics(self) -> Dict[str, Any]:
        """
        평가 메트릭 계산.
        
        Returns:
            평가 메트릭 dict
        """
        trades = list(self.trade_tape)
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
                "profit_factor": 0.0,
            }
        
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total = len(pnls)
        win_rate = len(wins) / total if total > 0 else 0.0
        avg_pnl = sum(pnls) / total if total > 0 else 0.0
        
        # Sharpe (단순화)
        import numpy as np
        arr = np.array(pnls)
        sharpe = float(np.mean(arr) / np.std(arr)) if np.std(arr) > 0 else 0.0
        
        # Profit Factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        return {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

    async def broadcast(self, force: bool = False) -> None:
        """
        대시보드로 현재 상태 브로드캐스트.
        
        Args:
            force: True면 쿨다운 무시
        """
        now = time.time()
        
        # 1초 쿨다운 (강제가 아닌 경우)
        if not force and (now - self._last_broadcast_time < 1.0):
            return
        
        self._last_broadcast_time = now
        
        # 행 데이터 수집
        rows: List[Dict[str, Any]] = []
        for sym in self.symbols:
            decision = self._latest_decisions.get(sym)
            row = self._row(sym, decision)
            rows.append(row)
        
        # 포트폴리오 요약
        portfolio = self._compute_portfolio()
        
        # 메트릭
        metrics = self._compute_eval_metrics()
        
        # 알림
        alerts = list(self._alert_queue) if hasattr(self, "_alert_queue") else []
        
        payload = {
            "timestamp": int(now * 1000),
            "rows": rows,
            "portfolio": portfolio,
            "metrics": metrics,
            "alerts": alerts[-20:],  # 최근 20개만
        }
        
        # 대시보드 서버로 전송
        if hasattr(self, "_dashboard_server") and self._dashboard_server:
            try:
                await self._dashboard_server.broadcast(payload)
            except Exception:
                pass  # 브로드캐스트 실패는 무시
