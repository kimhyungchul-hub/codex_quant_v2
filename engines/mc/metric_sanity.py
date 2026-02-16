"""MC Metric Sanity Monitor — MC 파이프라인 수치 이상 감지 & 자동 튜닝.

모든 MC 산출/소비 수치를 주기적으로 검증하여:
  1. 수학적으로 불가능한 값 (probability > 1, negative sigma 등) 탐지
  2. 논리적 불일치 (mu↔direction mismatch, TP+SL+timeout ≠ 1 등) 탐지
  3. 통계적 이상 (모든 EV 음수, drift 소멸, cost > EV 등) 탐지
  4. 자동 튜닝 (반복 이상 시 파라미터 완화/조정)
  5. 대시보드 알림 payload 생성

Usage:
    monitor = MCMetricSanityMonitor()
    alerts = monitor.check_batch(batch_decisions)
    auto_fixes = monitor.auto_tune(alerts)
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mc_sanity")


@dataclass
class SanityAlert:
    """Single anomaly detected in MC pipeline."""
    category: str          # e.g. "probability", "consistency", "statistical"
    metric: str            # e.g. "event_p_tp"
    severity: str          # "INFO", "WARN", "CRITICAL"
    symbol: str            # affected symbol or "ALL"
    message: str           # human-readable description
    value: Optional[float] = None
    expected_range: Optional[str] = None
    auto_fixable: bool = False
    fix_description: Optional[str] = None
    timestamp_ms: int = 0


@dataclass
class SanityReport:
    """Batch sanity check report."""
    timestamp_ms: int = 0
    total_symbols: int = 0
    alerts: List[SanityAlert] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    all_ok: bool = True
    # Summary stats
    ev_mean: float = 0.0
    ev_positive_ratio: float = 0.0
    mu_abs_mean: float = 0.0
    sigma_mean: float = 0.0
    cost_mean: float = 0.0
    edge_gap_mean: float = 0.0  # edge - cost gap
    tp_sl_balance: float = 0.0  # avg(p_tp - p_sl)


class MCMetricSanityMonitor:
    """Comprehensive MC metric validation system."""

    def __init__(self) -> None:
        self._issue_streaks: Dict[str, int] = {}
        self._last_fix_ms: int = 0
        self._history: List[SanityReport] = []  # keep last N reports
        self._max_history: int = 30

    def check_batch(self, batch_decisions: List[Dict[str, Any]]) -> SanityReport:
        """Run all sanity checks on batch decisions from MC.

        Args:
            batch_decisions: list of decision dicts from decide_batch,
                             each containing 'meta' with MC metrics.

        Returns:
            SanityReport with all detected anomalies.
        """
        ts = int(time.time() * 1000)
        report = SanityReport(timestamp_ms=ts, total_symbols=len(batch_decisions or []))
        alerts: List[SanityAlert] = []

        if not batch_decisions:
            report.all_ok = True
            return report

        # Aggregate metrics from all decisions
        all_mu: List[float] = []
        all_sigma: List[float] = []
        all_ev: List[float] = []
        all_edge_raw: List[float] = []
        all_cost: List[float] = []
        all_p_tp: List[float] = []
        all_p_sl: List[float] = []
        all_p_timeout: List[float] = []
        all_event_ev_r: List[float] = []
        all_win: List[float] = []
        all_kelly: List[float] = []
        all_unified: List[float] = []
        all_cvar: List[float] = []
        all_dampen_ratio: List[float] = []

        for dec in batch_decisions:
            m = dec.get("meta") or {}
            sym = str(m.get("symbol") or dec.get("symbol") or "?")

            # === 1. PROBABILITY RANGE CHECKS ===
            alerts.extend(self._check_probability_ranges(m, sym, ts))

            # === 2. DIRECTION CONSISTENCY ===
            alerts.extend(self._check_direction_consistency(m, dec, sym, ts))

            # === 3. PER-SYMBOL VALUE RANGE CHECKS ===
            alerts.extend(self._check_value_ranges(m, sym, ts))

            # === 4. COST vs EV CONSISTENCY ===
            alerts.extend(self._check_cost_ev_consistency(m, sym, ts))

            # === 5. DAMPEN CHAIN MONITORING ===
            alerts.extend(self._check_dampen_ratio(m, sym, ts))

            # Aggregate for global checks
            _append_float(all_mu, m, "mu_alpha", "pred_mu_alpha")
            _append_float(all_sigma, m, "sigma", "sigma_adj", "sigma_sim", "sigma_annual")
            _append_float(all_ev, m, "unified_score", "ev", "policy_ev_mix")
            _append_float(all_edge_raw, m, "policy_ev_mix_long", "policy_ev_mix_short", "unified_score_long", "unified_score_short")
            _append_float(all_cost, m, "fee_roundtrip_total", "execution_cost")
            _append_float(all_p_tp, m, "event_p_tp")
            _append_float(all_p_sl, m, "event_p_sl")
            _append_float(all_p_timeout, m, "event_p_timeout")
            _append_float(all_event_ev_r, m, "event_ev_r")
            _append_float(all_win, m, "win")
            _append_float(all_kelly, m, "kelly", "kelly_frac")
            _append_float(all_unified, m, "unified_score")
            _append_float(all_cvar, m, "cvar", "cvar_95")
            _append_float(all_dampen_ratio, m, "total_dampen_ratio")

        # === 6. GLOBAL STATISTICAL CHECKS ===
        alerts.extend(self._check_global_statistics(
            all_mu, all_sigma, all_ev, all_edge_raw, all_cost,
            all_p_tp, all_p_sl, all_unified, all_cvar, all_dampen_ratio,
            ts
        ))

        # Build report
        report.alerts = alerts
        report.all_ok = all(a.severity == "INFO" for a in alerts)

        # Summary stats
        if all_ev:
            ev_arr = np.array(all_ev)
            report.ev_mean = float(np.mean(ev_arr))
            report.ev_positive_ratio = float(np.sum(ev_arr > 0) / len(ev_arr))
        if all_mu:
            report.mu_abs_mean = float(np.mean(np.abs(all_mu)))
        if all_sigma:
            report.sigma_mean = float(np.mean(all_sigma))
        if all_cost:
            report.cost_mean = float(np.mean(all_cost))
        if all_edge_raw and all_cost:
            min_len = min(len(all_edge_raw), len(all_cost))
            gaps = [all_edge_raw[i] - all_cost[i] for i in range(min_len)]
            report.edge_gap_mean = float(np.mean(gaps)) if gaps else 0.0
        if all_p_tp and all_p_sl:
            min_len = min(len(all_p_tp), len(all_p_sl))
            report.tp_sl_balance = float(np.mean([all_p_tp[i] - all_p_sl[i] for i in range(min_len)]))

        # Update streaks
        seen_metrics = set()
        for a in alerts:
            if a.severity in ("WARN", "CRITICAL"):
                key = f"{a.category}:{a.metric}"
                seen_metrics.add(key)
                self._issue_streaks[key] = self._issue_streaks.get(key, 0) + 1
        # Reset cleared metrics
        for key in list(self._issue_streaks.keys()):
            if key not in seen_metrics:
                self._issue_streaks[key] = 0

        # History
        self._history.append(report)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return report

    def auto_tune(self, report: SanityReport) -> List[str]:
        """Apply automatic parameter adjustments for recurring issues.

        Returns list of human-readable fix descriptions.
        """
        fixes: List[str] = []
        ts = int(time.time() * 1000)

        # Rate limit: at most once per 3 minutes
        if (ts - self._last_fix_ms) < 180_000:
            return fixes

        auto_fixable = [a for a in report.alerts if a.auto_fixable and a.severity in ("WARN", "CRITICAL")]
        if not auto_fixable:
            return fixes

        # Group fixes by type
        fix_map: Dict[str, List[SanityAlert]] = {}
        for a in auto_fixable:
            key = f"{a.category}:{a.metric}"
            fix_map.setdefault(key, []).append(a)

        for key, alerts_list in fix_map.items():
            streak = self._issue_streaks.get(key, 0)
            if streak < 2:
                continue  # need at least 2 consecutive occurrences

            for a in alerts_list[:1]:  # pick first
                fix = self._apply_fix(a, streak)
                if fix:
                    fixes.append(fix)
                    report.fixes_applied.append(fix)

        if fixes:
            self._last_fix_ms = ts

        return fixes

    def get_dashboard_payload(self, report: SanityReport) -> Dict[str, Any]:
        """Convert report to dashboard-friendly JSON."""
        critical = [a for a in report.alerts if a.severity == "CRITICAL"]
        warn = [a for a in report.alerts if a.severity == "WARN"]
        info = [a for a in report.alerts if a.severity == "INFO"]

        trend_ev = "stable"
        if len(self._history) >= 3:
            recent = [r.ev_mean for r in self._history[-3:]]
            if all(r < 0 for r in recent):
                if recent[-1] < recent[0]:
                    trend_ev = "deteriorating"
                else:
                    trend_ev = "recovering"

        return {
            "mc_sanity": {
                "timestamp_ms": report.timestamp_ms,
                "all_ok": report.all_ok,
                "critical_count": len(critical),
                "warn_count": len(warn),
                "info_count": len(info),
                "alerts": [
                    {
                        "category": a.category,
                        "metric": a.metric,
                        "severity": a.severity,
                        "symbol": a.symbol,
                        "message": a.message,
                        "value": a.value,
                    }
                    for a in (critical + warn)[:10]  # top 10 for dashboard
                ],
                "fixes_applied": report.fixes_applied,
                "summary": {
                    "ev_mean": round(report.ev_mean, 6),
                    "ev_pos_ratio": round(report.ev_positive_ratio, 3),
                    "mu_abs_mean": round(report.mu_abs_mean, 4),
                    "sigma_mean": round(report.sigma_mean, 4),
                    "cost_mean": round(report.cost_mean, 6),
                    "edge_gap_mean": round(report.edge_gap_mean, 6),
                    "tp_sl_balance": round(report.tp_sl_balance, 4),
                    "trend_ev": trend_ev,
                },
            }
        }

    # ─── Per-Symbol Checks ───────────────────────────────────────

    def _check_probability_ranges(self, m: dict, sym: str, ts: int) -> List[SanityAlert]:
        """Check that probabilities are in [0, 1] and sum to ~1."""
        alerts: List[SanityAlert] = []
        p_tp = _safe_float(m, "event_p_tp")
        p_sl = _safe_float(m, "event_p_sl")
        p_to = _safe_float(m, "event_p_timeout")

        for name, val in [("event_p_tp", p_tp), ("event_p_sl", p_sl), ("event_p_timeout", p_to)]:
            if val is not None and (val < -0.001 or val > 1.001):
                alerts.append(SanityAlert(
                    category="probability", metric=name, severity="CRITICAL",
                    symbol=sym, message=f"{name}={val:.4f} 범위 초과 [0,1]",
                    value=val, expected_range="[0.0, 1.0]",
                    timestamp_ms=ts,
                ))

        if p_tp is not None and p_sl is not None and p_to is not None:
            prob_sum = p_tp + p_sl + p_to
            if abs(prob_sum - 1.0) > 0.05:
                alerts.append(SanityAlert(
                    category="probability", metric="prob_sum", severity="WARN",
                    symbol=sym, message=f"P(TP)+P(SL)+P(Timeout)={prob_sum:.4f} ≠ 1.0",
                    value=prob_sum, expected_range="≈1.0",
                    timestamp_ms=ts,
                ))

        # Win rate sanity
        win = _safe_float(m, "win")
        if win is not None and (win < 0.0 or win > 1.0):
            alerts.append(SanityAlert(
                category="probability", metric="win", severity="CRITICAL",
                symbol=sym, message=f"win={win:.4f} 범위 초과",
                value=win, expected_range="[0.0, 1.0]",
                timestamp_ms=ts,
            ))

        # ppos horizons
        for side in ("long", "short"):
            key = f"ppos_{side}_h"
            arr = m.get(key)
            if isinstance(arr, (list, tuple)):
                for i, v in enumerate(arr):
                    if v is not None and (float(v) < -0.01 or float(v) > 1.01):
                        alerts.append(SanityAlert(
                            category="probability", metric=f"ppos_{side}[{i}]", severity="WARN",
                            symbol=sym, message=f"ppos_{side}[{i}]={float(v):.4f} 범위 초과",
                            value=float(v), expected_range="[0.0, 1.0]",
                            timestamp_ms=ts,
                        ))

        return alerts

    def _check_direction_consistency(self, m: dict, dec: dict, sym: str, ts: int) -> List[SanityAlert]:
        """Verify direction matches mu_alpha sign."""
        alerts: List[SanityAlert] = []
        direction = str(m.get("direction") or dec.get("action") or "").upper()
        mu = _safe_float(m, "mu_alpha", "pred_mu_alpha")
        ev_long = _safe_float(m, "unified_score_long", "policy_ev_score_long", "policy_ev_mix_long")
        ev_short = _safe_float(m, "unified_score_short", "policy_ev_score_short", "policy_ev_mix_short")

        if mu is not None and direction in ("LONG", "SHORT"):
            # Strong mu contradicts direction
            if direction == "LONG" and mu < -1.0:
                alerts.append(SanityAlert(
                    category="consistency", metric="dir_mu_mismatch", severity="WARN",
                    symbol=sym, message=f"LONG but mu_alpha={mu:.4f} (강한 음수)",
                    value=mu, timestamp_ms=ts,
                ))
            elif direction == "SHORT" and mu > 1.0:
                alerts.append(SanityAlert(
                    category="consistency", metric="dir_mu_mismatch", severity="WARN",
                    symbol=sym, message=f"SHORT but mu_alpha={mu:.4f} (강한 양수)",
                    value=mu, timestamp_ms=ts,
                ))

        # EV direction consistency
        if ev_long is not None and ev_short is not None and direction in ("LONG", "SHORT"):
            if direction == "LONG" and ev_short > ev_long + 0.001:
                alerts.append(SanityAlert(
                    category="consistency", metric="dir_ev_mismatch", severity="INFO",
                    symbol=sym, message=f"LONG chosen but ev_short({ev_short:.6f}) > ev_long({ev_long:.6f})",
                    timestamp_ms=ts,
                ))
            elif direction == "SHORT" and ev_long > ev_short + 0.001:
                alerts.append(SanityAlert(
                    category="consistency", metric="dir_ev_mismatch", severity="INFO",
                    symbol=sym, message=f"SHORT chosen but ev_long({ev_long:.6f}) > ev_short({ev_short:.6f})",
                    timestamp_ms=ts,
                ))

        return alerts

    def _check_value_ranges(self, m: dict, sym: str, ts: int) -> List[SanityAlert]:
        """Check that MC-derived values are in reasonable ranges."""
        alerts: List[SanityAlert] = []

        # sigma: must be positive, typically 0.05 ~ 5.0 annualized
        sigma = _safe_float(m, "sigma", "sigma_adj", "sigma_sim", "sigma_annual")
        if sigma is not None:
            if sigma <= 0:
                alerts.append(SanityAlert(
                    category="range", metric="sigma", severity="CRITICAL",
                    symbol=sym, message=f"sigma={sigma:.6f} ≤ 0 (불가능)",
                    value=sigma, expected_range="(0, 10]",
                    timestamp_ms=ts,
                ))
            elif sigma > 10.0:
                alerts.append(SanityAlert(
                    category="range", metric="sigma", severity="WARN",
                    symbol=sym, message=f"sigma={sigma:.4f} > 10.0 (비정상적으로 높음)",
                    value=sigma, expected_range="(0, 10]",
                    timestamp_ms=ts,
                ))

        # kelly: typically [-1, 1], extreme values indicate model failure
        kelly = _safe_float(m, "kelly", "kelly_frac")
        if kelly is not None and abs(kelly) > 2.0:
            alerts.append(SanityAlert(
                category="range", metric="kelly", severity="WARN",
                symbol=sym, message=f"kelly={kelly:.4f} 극단값 (|kelly| > 2.0)",
                value=kelly, expected_range="[-1, 1]",
                timestamp_ms=ts,
            ))

        # event_ev_r: R-multiple, -1 to tp/sl ratio, typically [-1, 10]
        ev_r = _safe_float(m, "event_ev_r")
        if ev_r is not None and (ev_r < -2.0 or ev_r > 20.0):
            alerts.append(SanityAlert(
                category="range", metric="event_ev_r", severity="WARN",
                symbol=sym, message=f"event_ev_r={ev_r:.4f} 범위 이상",
                value=ev_r, expected_range="[-1, 10]",
                timestamp_ms=ts,
            ))

        # fee/cost: must be positive, typically < 0.3%
        fee = _safe_float(m, "fee_roundtrip_total", "execution_cost")
        if fee is not None:
            if fee < 0:
                alerts.append(SanityAlert(
                    category="range", metric="fee_roundtrip", severity="CRITICAL",
                    symbol=sym, message=f"fee_roundtrip={fee:.6f} < 0 (불가능)",
                    value=fee, expected_range="[0, 0.003]",
                    timestamp_ms=ts,
                ))
            elif fee > 0.005:
                alerts.append(SanityAlert(
                    category="range", metric="fee_roundtrip", severity="WARN",
                    symbol=sym, message=f"fee_roundtrip={fee:.6f} > 0.5% (비정상적으로 높음)",
                    value=fee, expected_range="[0, 0.003]",
                    timestamp_ms=ts,
                ))

        # leverage: sanity
        lev = _safe_float(m, "optimal_leverage", "leverage")
        if lev is not None and (lev < 0 or lev > 100):
            alerts.append(SanityAlert(
                category="range", metric="leverage", severity="WARN",
                symbol=sym, message=f"leverage={lev:.1f} 범위 이상",
                value=lev, expected_range="[0, 50]",
                timestamp_ms=ts,
            ))

        return alerts

    def _check_cost_ev_consistency(self, m: dict, sym: str, ts: int) -> List[SanityAlert]:
        """Check that cost doesn't dominate EV in pathological ways."""
        alerts: List[SanityAlert] = []

        ev_best = _safe_float(m, "ev_best", "ev_expected")
        cost = _safe_float(m, "fee_roundtrip_total", "execution_cost")
        ev_unified = _safe_float(m, "unified_score")

        if ev_best is not None and cost is not None and cost > 0:
            # If best-horizon EV is more than 10x the cost, something is off
            if abs(ev_best) > cost * 20:
                alerts.append(SanityAlert(
                    category="consistency", metric="ev_cost_ratio", severity="INFO",
                    symbol=sym,
                    message=f"EV/cost ratio extreme: ev_best={ev_best:.6f}, cost={cost:.6f}, ratio={ev_best/cost:.1f}x",
                    value=ev_best / cost,
                    timestamp_ms=ts,
                ))

        # If unified_score is very negative but action != WAIT, anomaly
        action = str(m.get("action") or "").upper()
        if ev_unified is not None and ev_unified < -0.01 and action in ("LONG", "SHORT"):
            alerts.append(SanityAlert(
                category="consistency", metric="score_action_mismatch", severity="WARN",
                symbol=sym,
                message=f"action={action} but unified_score={ev_unified:.6f} (강한 음수)",
                value=ev_unified,
                timestamp_ms=ts,
            ))

        return alerts

    def _check_dampen_ratio(self, m: dict, sym: str, ts: int) -> List[SanityAlert]:
        """Check if mu_alpha dampening is destroying signal."""
        alerts: List[SanityAlert] = []
        dampen = _safe_float(m, "total_dampen_ratio", "mu_dampen_ratio")
        if dampen is not None and dampen < 0.05:
            alerts.append(SanityAlert(
                category="statistical", metric="dampen_ratio", severity="WARN",
                symbol=sym,
                message=f"total_dampen_ratio={dampen:.4f} < 0.05 — 신호 사실상 소멸",
                value=dampen,
                expected_range="[0.05, 1.0]",
                auto_fixable=True,
                fix_description="Reduce Hurst/chop dampening strength",
                timestamp_ms=ts,
            ))
        return alerts

    # ─── Global Checks ───────────────────────────────────────

    def _check_global_statistics(
        self,
        all_mu, all_sigma, all_ev, all_edge, all_cost,
        all_p_tp, all_p_sl, all_unified, all_cvar, all_dampen,
        ts: int,
    ) -> List[SanityAlert]:
        alerts: List[SanityAlert] = []

        # G1: 모든 EV 음수 (진입 불가 상태)
        if len(all_ev) >= 5:
            ev_arr = np.array(all_ev)
            pos_ratio = float(np.sum(ev_arr > 0)) / len(ev_arr)
            if pos_ratio == 0:
                alerts.append(SanityAlert(
                    category="statistical", metric="ev_all_negative", severity="WARN",
                    symbol="ALL",
                    message=f"모든 EV 음수 (mean={float(np.mean(ev_arr)):.6f}, max={float(np.max(ev_arr)):.6f}) — 진입 불가",
                    value=float(np.mean(ev_arr)),
                    auto_fixable=True,
                    fix_description="NX floor 완화 또는 cost 모델 점검",
                    timestamp_ms=ts,
                ))
            elif pos_ratio < 0.1:
                alerts.append(SanityAlert(
                    category="statistical", metric="ev_mostly_negative", severity="INFO",
                    symbol="ALL",
                    message=f"EV 양수 비율 {pos_ratio*100:.0f}% (mean={float(np.mean(ev_arr)):.6f})",
                    value=pos_ratio,
                    timestamp_ms=ts,
                ))

        # G2: mu_alpha 거의 0 (신호 사라짐)
        if len(all_mu) >= 5:
            mu_abs = float(np.mean(np.abs(all_mu)))
            if mu_abs < 0.01:
                alerts.append(SanityAlert(
                    category="statistical", metric="mu_alpha_near_zero", severity="WARN",
                    symbol="ALL",
                    message=f"mu_alpha |mean|={mu_abs:.6f} — 방향 신호 거의 없음 (dampening 과다 가능성)",
                    value=mu_abs,
                    auto_fixable=True,
                    fix_description="mu_alpha dampening 완화",
                    timestamp_ms=ts,
                ))

        # G3: Cost가 Edge보다 항상 큼 (구조적 진입 불가)
        if len(all_edge) >= 5 and len(all_cost) >= 5:
            min_len = min(len(all_edge), len(all_cost))
            gaps = [all_edge[i] - all_cost[i] for i in range(min_len)]
            if all(g < 0 for g in gaps):
                avg_gap = float(np.mean(gaps))
                alerts.append(SanityAlert(
                    category="statistical", metric="edge_cost_structural", severity="WARN",
                    symbol="ALL",
                    message=f"Edge < Cost 구조적 차이: avg gap={avg_gap:.6f} — 비용 모델 과다 또는 EV 과소 평가",
                    value=avg_gap,
                    auto_fixable=True,
                    fix_description="execution cost 파라미터 점검 (delay_k, adverse_k)",
                    timestamp_ms=ts,
                ))

        # G4: sigma 다양성 부족
        if len(all_sigma) >= 5:
            sigma_arr = np.array(all_sigma)
            sigma_cv = float(np.std(sigma_arr) / max(np.mean(sigma_arr), 1e-12))
            if sigma_cv < 0.01:
                alerts.append(SanityAlert(
                    category="statistical", metric="sigma_homogeneous", severity="INFO",
                    symbol="ALL",
                    message=f"sigma CV={sigma_cv:.4f} — 모든 심볼 동일 σ (GARCH 미분화)",
                    value=sigma_cv,
                    timestamp_ms=ts,
                ))

        # G5: TP 편향 검사
        if len(all_p_tp) >= 5 and len(all_p_sl) >= 5:
            tp_arr = np.array(all_p_tp)
            sl_arr = np.array(all_p_sl[:len(all_p_tp)])
            tp_mean = float(np.mean(tp_arr))
            sl_mean = float(np.mean(sl_arr))
            if tp_mean < 0.05 and sl_mean > 0.5:
                alerts.append(SanityAlert(
                    category="statistical", metric="tp_sl_imbalance", severity="WARN",
                    symbol="ALL",
                    message=f"TP확률 극저={tp_mean:.4f}, SL확률 극고={sl_mean:.4f} — TP 도달 불가능 상태",
                    value=tp_mean,
                    auto_fixable=True,
                    fix_description="TP_PCT 조정 또는 Barrier 계산 점검",
                    timestamp_ms=ts,
                ))

        # G6: dampen ratio 전체 저조
        if len(all_dampen) >= 5:
            dampen_arr = np.array(all_dampen)
            dampen_mean = float(np.mean(dampen_arr))
            if dampen_mean < 0.1:
                alerts.append(SanityAlert(
                    category="statistical", metric="dampen_all_low", severity="WARN",
                    symbol="ALL",
                    message=f"전체 dampen_ratio 평균={dampen_mean:.4f} — 감쇠 체인이 신호를 소멸시킴",
                    value=dampen_mean,
                    auto_fixable=True,
                    fix_description="MC_HURST_RANDOM_DAMPEN_MAX, CHOP_GUARD 완화",
                    timestamp_ms=ts,
                ))

        # G7: CVaR 극단 체크
        if len(all_cvar) >= 5:
            cvar_arr = np.array(all_cvar)
            cvar_min = float(np.min(cvar_arr))
            if cvar_min < -0.5:
                alerts.append(SanityAlert(
                    category="statistical", metric="cvar_extreme", severity="WARN",
                    symbol="ALL",
                    message=f"CVaR 극단값={cvar_min:.4f} — 테일 리스크 비정상",
                    value=cvar_min,
                    timestamp_ms=ts,
                ))

        return alerts

    # ─── Auto-Fix Implementations ─────────────────────────────

    def _apply_fix(self, alert: SanityAlert, streak: int) -> Optional[str]:
        """Apply a specific fix based on alert type. Returns description or None."""
        try:
            metric = alert.metric

            if metric == "ev_all_negative" and streak >= 3:
                # 비용 모델 파라미터 완화
                try:
                    cur_delay_k = float(os.environ.get("ORDER_EXEC_DELAY_PENALTY_BPS_K", "2.4"))
                except Exception:
                    cur_delay_k = 2.4
                new_delay_k = max(0.5, cur_delay_k * 0.8)
                os.environ["ORDER_EXEC_DELAY_PENALTY_BPS_K"] = f"{new_delay_k:.2f}"

                try:
                    cur_adverse_k = float(os.environ.get("ORDER_EXEC_ADVERSE_BPS_K", "3.5"))
                except Exception:
                    cur_adverse_k = 3.5
                new_adverse_k = max(0.5, cur_adverse_k * 0.8)
                os.environ["ORDER_EXEC_ADVERSE_BPS_K"] = f"{new_adverse_k:.2f}"

                return f"delay_k: {cur_delay_k:.1f}→{new_delay_k:.1f}, adverse_k: {cur_adverse_k:.1f}→{new_adverse_k:.1f}"

            elif metric == "mu_alpha_near_zero" and streak >= 3:
                # Hurst dampening 완화
                try:
                    cur_hurst = float(os.environ.get("MC_HURST_RANDOM_DAMPEN_MAX", "0.5"))
                except Exception:
                    cur_hurst = 0.5
                new_hurst = max(0.1, cur_hurst * 0.7)
                os.environ["MC_HURST_RANDOM_DAMPEN_MAX"] = f"{new_hurst:.3f}"
                return f"MC_HURST_RANDOM_DAMPEN_MAX: {cur_hurst:.3f}→{new_hurst:.3f}"

            elif metric == "dampen_all_low" and streak >= 2:
                # Chop guard 완화
                try:
                    cur_chop_floor = float(os.environ.get("CHOP_MU_FLOOR", "0.1"))
                except Exception:
                    cur_chop_floor = 0.1
                new_chop_floor = max(0.01, cur_chop_floor * 0.5)
                os.environ["CHOP_MU_FLOOR"] = f"{new_chop_floor:.4f}"
                return f"CHOP_MU_FLOOR: {cur_chop_floor:.4f}→{new_chop_floor:.4f}"

            elif metric == "edge_cost_structural" and streak >= 3:
                # Delay/adverse penalty 축소
                try:
                    cur_delay_k = float(os.environ.get("ORDER_EXEC_DELAY_PENALTY_BPS_K", "2.4"))
                except Exception:
                    cur_delay_k = 2.4
                new_delay_k = max(0.5, cur_delay_k * 0.75)
                os.environ["ORDER_EXEC_DELAY_PENALTY_BPS_K"] = f"{new_delay_k:.2f}"
                return f"ORDER_EXEC_DELAY_PENALTY_BPS_K: {cur_delay_k:.2f}→{new_delay_k:.2f}"

            elif metric == "tp_sl_imbalance" and streak >= 3:
                # TP 너무 tight하면 조정 힌트만
                logger.warning(
                    "[MC_SANITY_FIX] TP/SL imbalance detected. Consider increasing DEFAULT_TP_PCT."
                )
                return "Hint: DEFAULT_TP_PCT 증가 권장 (현재 TP 도달률 극저)"

            elif metric == "dampen_ratio" and streak >= 2:
                return None  # 개별 심볼은 글로벌 fix에서 처리

        except Exception as e:
            logger.error(f"[MC_SANITY_FIX] Fix failed for {alert.metric}: {e}")
        return None


# ─── Utilities ──────────────────────────────────────────────────

def _safe_float(m: dict, *keys) -> Optional[float]:
    """Extract first valid float from dict by key priority."""
    for k in keys:
        try:
            v = m.get(k)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _append_float(target: list, m: dict, *keys):
    """Append first valid float from dict to list."""
    v = _safe_float(m, *keys)
    if v is not None:
        target.append(v)
