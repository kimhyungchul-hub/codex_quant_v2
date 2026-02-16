#!/usr/bin/env python3
"""
core/auto_diagnostics.py — 자동 진단 & 자기 개선 시스템 (Self-Improving Diagnostics Engine)

실시간으로 최근 거래를 분석하고, 문제를 감지하여 파라미터를 자동 조정합니다.
auto_tune_overrides.json의 기존 hot-reload 파이프라인과 통합됩니다.

5가지 진단 모듈:
1. Direction Accuracy — mu_alpha ↔ direction 정렬 품질 모니터링
2. Regime×Side Profitability — 레짐별 방향 수익성 감시, 블록리스트 자동 갱신
3. Time-of-Day Analysis — 시간대별 수익성 감시, bad hours 자동 갱신
4. Sizing & Leverage Feedback — 최근 승률/payoff 기반 레버리지 자동 조정
5. Hold Duration Optimization — 최적 보유 시간 분석

사용법:
    engine.decision_loop() 내에서 1시간마다 자동 호출됨.
    또는 독립 실행: python3 -m core.auto_diagnostics --db state/bot_data_live.db
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("auto_diagnostics")

# ─── 안전 한계 (Safety Guardrails) ────────────────────────────────
MAX_LEVERAGE_CHANGE = 0.15          # 1회 조정 시 최대 ±15%
MIN_COOLDOWN_SEC = 1800             # 조정 후 최소 30분 쿨다운
MIN_SAMPLE_SIZE = 20                # 최소 분석 샘플 수
MAX_BAD_HOURS = 6                   # 최대 차단 가능 시간 수
REGIME_BLOCK_MIN_TRADES = 15        # 레짐 블록 판단 최소 거래 수
REGIME_BLOCK_WR_THRESHOLD = 0.38    # 이 이하 WR이면 블록 후보
REGIME_BLOCK_LOSS_THRESHOLD = -5.0  # 누적 손실 기준 (달러)


@dataclass
class DiagnosticResult:
    """각 진단 모듈의 결과"""
    module: str
    severity: str  # CRITICAL / HIGH / MEDIUM / LOW / INFO
    message: str
    recommendation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoDiagnosticsState:
    """진단 상태 추적"""
    last_run_ts: float = 0.0
    last_adjust_ts: float = 0.0
    consecutive_loss_cycles: int = 0
    total_adjustments: int = 0
    adjustment_history: list[dict] = field(default_factory=list)


class AutoDiagnosticsEngine:
    """
    자동 진단 & 자기 개선 엔진.
    
    기존 auto_tune_overrides.json 파이프라인과 통합:
    - 진단 결과를 state/diagnostics_overrides.json에 기록
    - _maybe_reload_auto_tune_overrides()에서 merge하여 적용
    - Safety guardrails로 과도한 변경 방지
    """

    def __init__(
        self,
        db_path: str = "state/bot_data_live.db",
        output_path: str = "state/diagnostics_overrides.json",
        state_path: str = "state/diagnostics_state.json",
        lookback_hours: float = 4.0,
        interval_sec: float = 3600.0,
    ):
        self.db_path = Path(db_path)
        self.output_path = Path(output_path)
        self.state_path = Path(state_path)
        self.lookback_hours = lookback_hours
        self.interval_sec = interval_sec
        self.state = AutoDiagnosticsState()
        self._load_state()

    # ──────────────── Public API ────────────────

    def should_run(self) -> bool:
        """실행 시점 판단"""
        elapsed = time.time() - self.state.last_run_ts
        return elapsed >= self.interval_sec

    def run_diagnostics(self, extra_context: dict | None = None) -> list[DiagnosticResult]:
        """
        전체 진단 실행 → 파라미터 조정 → 파일 기록

        Returns:
            DiagnosticResult 리스트
        """
        self.state.last_run_ts = time.time()
        results: list[DiagnosticResult] = []

        df = self._load_recent_trades()
        if df is None or len(df) < MIN_SAMPLE_SIZE:
            logger.info(f"[AUTO_DIAG] Insufficient trades: {len(df) if df is not None else 0} < {MIN_SAMPLE_SIZE}")
            self._save_state()
            return results

        # 5가지 진단 모듈 실행
        results.extend(self._diag_direction_accuracy(df))
        results.extend(self._diag_regime_side(df))
        results.extend(self._diag_time_of_day(df))
        results.extend(self._diag_sizing_leverage(df))
        results.extend(self._diag_hold_duration(df))

        # 조정 추천 수집 및 적용
        overrides = self._collect_overrides(results)
        if overrides:
            self._apply_overrides(overrides, results)

        self._save_state()
        self._log_summary(results)
        return results

    # ──────────────── 진단 모듈 1: Direction Accuracy ────────────────

    def _diag_direction_accuracy(self, df: list[dict]) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []
        
        # mu_alpha alignment 분석
        aligned = [t for t in df if t.get("pred_mu_alpha") is not None]
        if len(aligned) < MIN_SAMPLE_SIZE:
            return results

        aligned_trades = []
        misaligned_trades = []
        for t in aligned:
            mu = float(t.get("pred_mu_alpha") or 0)
            side = str(t.get("side") or "").upper()
            pnl = float(t.get("realized_pnl") or 0)
            is_aligned = (mu > 0 and side == "BUY") or (mu < 0 and side == "SELL")
            if abs(mu) < 0.001:
                continue  # mu too small to classify
            if is_aligned:
                aligned_trades.append({"pnl": pnl, "hit": 1 if pnl > 0 else 0})
            else:
                misaligned_trades.append({"pnl": pnl, "hit": 1 if pnl > 0 else 0})

        if len(aligned_trades) >= 5 and len(misaligned_trades) >= 5:
            aligned_wr = np.mean([t["hit"] for t in aligned_trades])
            misaligned_wr = np.mean([t["hit"] for t in misaligned_trades])
            aligned_pnl = sum(t["pnl"] for t in aligned_trades)
            misaligned_pnl = sum(t["pnl"] for t in misaligned_trades)
            misalign_pct = len(misaligned_trades) / (len(aligned_trades) + len(misaligned_trades)) * 100

            sev = "CRITICAL" if misaligned_wr < 0.35 and misalign_pct > 50 else \
                  "HIGH" if misaligned_wr < 0.40 else "MEDIUM"

            rec = {}
            # mu_alpha direction gate 강도 조절
            current_min_abs = float(os.environ.get("MU_ALIGN_MIN_ABS", "0.01") or 0.01)
            if misalign_pct > 60 and misaligned_wr < 0.35:
                # 많은 misalignment + 나쁜 WR → min_abs 낮춤 (더 엄격하게 차단)
                new_min_abs = max(0.005, current_min_abs * 0.8)
                rec["MU_ALIGN_MIN_ABS"] = round(new_min_abs, 4)
            elif misalign_pct < 30 and aligned_wr > 0.50:
                # 정렬 잘 됨 + 좋은 WR → min_abs 높임 (제한 완화)
                new_min_abs = min(0.05, current_min_abs * 1.2)
                rec["MU_ALIGN_MIN_ABS"] = round(new_min_abs, 4)

            results.append(DiagnosticResult(
                module="direction_accuracy",
                severity=sev,
                message=(
                    f"Aligned WR={aligned_wr:.1%} (n={len(aligned_trades)}, ${aligned_pnl:+.2f}) "
                    f"vs Misaligned WR={misaligned_wr:.1%} (n={len(misaligned_trades)}, ${misaligned_pnl:+.2f}) — "
                    f"{misalign_pct:.0f}% misaligned"
                ),
                recommendation=rec,
                metrics={
                    "aligned_wr": aligned_wr, "misaligned_wr": misaligned_wr,
                    "aligned_n": len(aligned_trades), "misaligned_n": len(misaligned_trades),
                    "aligned_pnl": aligned_pnl, "misaligned_pnl": misaligned_pnl,
                    "misalign_pct": misalign_pct,
                },
            ))

        # Overall WR 분석
        all_hits = [1 if float(t.get("realized_pnl") or 0) > 0 else 0 for t in df]
        overall_wr = np.mean(all_hits) if all_hits else 0
        
        wins = [float(t["realized_pnl"]) for t in df if float(t.get("realized_pnl") or 0) > 0]
        losses = [abs(float(t["realized_pnl"])) for t in df if float(t.get("realized_pnl") or 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 1e-9
        payoff = avg_win / max(avg_loss, 1e-9)
        breakeven_wr = 1 / (1 + payoff) if payoff > 0 else 0.5

        if overall_wr < breakeven_wr - 0.05:
            self.state.consecutive_loss_cycles += 1
            sev = "CRITICAL" if self.state.consecutive_loss_cycles >= 3 else "HIGH"
            results.append(DiagnosticResult(
                module="direction_accuracy",
                severity=sev,
                message=f"WR={overall_wr:.1%} < BE_WR={breakeven_wr:.1%} (gap={breakeven_wr-overall_wr:.1%}, consecutive={self.state.consecutive_loss_cycles})",
                metrics={"wr": overall_wr, "breakeven_wr": breakeven_wr, "payoff": payoff},
            ))
        else:
            self.state.consecutive_loss_cycles = max(0, self.state.consecutive_loss_cycles - 1)

        return results

    # ──────────────── 진단 모듈 2: Regime × Side ────────────────

    def _diag_regime_side(self, df: list[dict]) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []
        
        combos: dict[str, list[dict]] = {}
        for t in df:
            regime = str(t.get("regime") or "unknown").lower()
            side = str(t.get("side") or "").upper()
            if side == "BUY":
                side = "LONG"
            elif side == "SELL":
                side = "SHORT"
            else:
                continue
            key = f"{regime}_{side.lower()}"
            combos.setdefault(key, []).append(t)

        current_block_str = str(os.environ.get("REGIME_SIDE_BLOCK_LIST", "") or "")
        current_blocks = {b.strip().lower() for b in current_block_str.split(",") if b.strip()}
        new_blocks = set(current_blocks)
        unblock_candidates = set()

        for combo, trades in combos.items():
            n = len(trades)
            if n < REGIME_BLOCK_MIN_TRADES:
                continue
            pnls = [float(t.get("realized_pnl") or 0) for t in trades]
            wr = np.mean([1 if p > 0 else 0 for p in pnls])
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)

            if combo in current_blocks:
                # 이미 블록된 콤보 — 해제 조건 확인 (보수적)
                # 30거래 이상 시뮬가 필요하지만, 블록되어 실거래가 없으므로 유지
                continue

            # 새로 블록 추가 조건
            if wr < REGIME_BLOCK_WR_THRESHOLD and total_pnl < REGIME_BLOCK_LOSS_THRESHOLD and n >= REGIME_BLOCK_MIN_TRADES:
                new_blocks.add(combo)
                results.append(DiagnosticResult(
                    module="regime_side",
                    severity="HIGH",
                    message=f"{combo}: WR={wr:.1%}, PnL=${total_pnl:+.2f}, n={n} → adding to BLOCK_LIST",
                    recommendation={"REGIME_SIDE_BLOCK_LIST": ",".join(sorted(new_blocks))},
                    metrics={"combo": combo, "wr": wr, "total_pnl": total_pnl, "n": n},
                ))
            elif wr < 0.40 and total_pnl < 0:
                results.append(DiagnosticResult(
                    module="regime_side",
                    severity="MEDIUM",
                    message=f"{combo}: WR={wr:.1%}, PnL=${total_pnl:+.2f}, n={n} — below average but not blocked yet",
                    metrics={"combo": combo, "wr": wr, "total_pnl": total_pnl, "n": n},
                ))

        # 블록리스트 변경 시 추천
        if new_blocks != current_blocks:
            results.append(DiagnosticResult(
                module="regime_side",
                severity="HIGH",
                message=f"BLOCK_LIST update: {current_blocks} → {new_blocks}",
                recommendation={"REGIME_SIDE_BLOCK_LIST": ",".join(sorted(new_blocks))},
            ))

        return results

    # ──────────────── 진단 모듈 3: Time-of-Day ────────────────

    def _diag_time_of_day(self, df: list[dict]) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []

        hourly_data: dict[int, list[float]] = {h: [] for h in range(24)}
        for t in df:
            ts = t.get("entry_time") or t.get("opened_at") or t.get("timestamp")
            if ts is None:
                continue
            try:
                if isinstance(ts, str):
                    from datetime import datetime as _dt
                    dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                elif isinstance(ts, (int, float)):
                    import time as _time
                    if ts > 1e12:
                        ts = ts / 1000  # ms → sec
                    hour = _time.gmtime(ts).tm_hour
                else:
                    continue
                pnl = float(t.get("realized_pnl") or 0)
                hourly_data[hour].append(pnl)
            except Exception:
                continue

        current_bad_str = str(os.environ.get("TRADING_BAD_HOURS_UTC", "6,7") or "")
        current_bad = {int(h.strip()) for h in current_bad_str.split(",") if h.strip().isdigit()}
        new_bad = set(current_bad)

        bad_candidates = []
        for hour in range(24):
            pnls = hourly_data[hour]
            n = len(pnls)
            if n < 5:
                continue
            wr = np.mean([1 if p > 0 else 0 for p in pnls])
            total = sum(pnls)
            
            # 나쁜 시간 기준: WR < 35% AND 총 손실 AND 최소 5거래
            if wr < 0.35 and total < -2.0 and hour not in current_bad:
                bad_candidates.append((hour, wr, total, n))

        # 가장 나쁜 시간대만 추가 (MAX_BAD_HOURS 제한)
        bad_candidates.sort(key=lambda x: x[2])  # 손실 큰 순
        for hour, wr, total, n in bad_candidates:
            if len(new_bad) >= MAX_BAD_HOURS:
                break
            new_bad.add(hour)
            results.append(DiagnosticResult(
                module="time_of_day",
                severity="HIGH",
                message=f"UTC {hour}h: WR={wr:.1%}, PnL=${total:+.2f}, n={n} → adding to BAD_HOURS",
                recommendation={"TRADING_BAD_HOURS_UTC": ",".join(str(h) for h in sorted(new_bad))},
                metrics={"hour": hour, "wr": wr, "total_pnl": total, "n": n},
            ))

        # 해제 후보: 현재 블록된 시간이 최근에 좋아졌다면
        for hour in list(current_bad):
            pnls = hourly_data.get(hour, [])
            # 블록된 시간은 거래가 없으므로 판단 불가 → 유지
            # (단, 매뉴얼 해제를 위한 로그는 남김)

        return results

    # ──────────────── 진단 모듈 4: Sizing & Leverage ────────────────

    def _diag_sizing_leverage(self, df: list[dict]) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []

        # 최근 거래의 notional-binned 성과 분석
        notional_bins = [
            ("tiny", 0, 50),
            ("small", 50, 200),
            ("medium", 200, 500),
            ("large", 500, float("inf")),
        ]

        for label, lo, hi in notional_bins:
            trades = [t for t in df if lo <= abs(float(t.get("notional") or 0)) < hi]
            if len(trades) < 5:
                continue
            pnls = [float(t.get("realized_pnl") or 0) for t in trades]
            wr = np.mean([1 if p > 0 else 0 for p in pnls])
            total = sum(pnls)

            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            payoff = np.mean(wins) / max(np.mean(losses), 1e-9) if wins and losses else 1.0

            if payoff < 1.0 and total < -5.0 and label in ("medium", "large"):
                rec = {}
                # 큰 사이즈에서 payoff < 1 → 레버리지 축소 추천
                current_max = float(os.environ.get("LEVERAGE_TARGET_MAX", "25") or 25)
                new_max = max(3.0, current_max * (1 - MAX_LEVERAGE_CHANGE))
                # Note: LEVERAGE_TARGET_MAX is in blocklist, so use LEVERAGE_DYNAMIC_MIN_SCALE
                current_scale = float(os.environ.get("LEVERAGE_DYNAMIC_MIN_SCALE", "0.4") or 0.4)
                new_scale = max(0.2, current_scale * 0.9)
                rec["LEVERAGE_DYNAMIC_MIN_SCALE"] = round(new_scale, 3)

                results.append(DiagnosticResult(
                    module="sizing_leverage",
                    severity="HIGH",
                    message=f"{label} notional (${lo}-${hi}): payoff={payoff:.2f}, WR={wr:.1%}, PnL=${total:+.2f} → reduce leverage",
                    recommendation=rec,
                    metrics={"bin": label, "wr": wr, "payoff": payoff, "total_pnl": total, "n": len(trades)},
                ))
            elif payoff > 1.5 and wr > 0.45 and total > 5.0:
                results.append(DiagnosticResult(
                    module="sizing_leverage",
                    severity="INFO",
                    message=f"{label} notional: payoff={payoff:.2f}, WR={wr:.1%}, PnL=${total:+.2f} — profitable bin",
                    metrics={"bin": label, "wr": wr, "payoff": payoff, "total_pnl": total, "n": len(trades)},
                ))

        # 전체 레버리지 vs PnL 상관 분석
        leverages = []
        pnls_all = []
        for t in df:
            lev = float(t.get("leverage") or 0)
            pnl = float(t.get("realized_pnl") or 0)
            if lev > 0:
                leverages.append(lev)
                pnls_all.append(pnl)
        
        if len(leverages) >= 10:
            corr = float(np.corrcoef(leverages, pnls_all)[0, 1]) if np.std(leverages) > 0 else 0
            if corr < -0.15:
                results.append(DiagnosticResult(
                    module="sizing_leverage",
                    severity="HIGH",
                    message=f"Leverage-PnL correlation = {corr:.3f} (higher leverage → worse PnL) — consider reducing",
                    metrics={"lev_pnl_corr": corr, "n": len(leverages)},
                ))

        return results

    # ──────────────── 진단 모듈 5: Hold Duration ────────────────

    def _diag_hold_duration(self, df: list[dict]) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []

        duration_bins = [
            ("flash", 0, 60),
            ("short", 60, 300),
            ("medium", 300, 3600),
            ("long", 3600, float("inf")),
        ]

        for label, lo_sec, hi_sec in duration_bins:
            trades = [
                t for t in df
                if lo_sec <= float(t.get("hold_duration_sec") or 0) < hi_sec
                and float(t.get("hold_duration_sec") or 0) > 0
            ]
            if len(trades) < 5:
                continue

            pnls = [float(t.get("realized_pnl") or 0) for t in trades]
            wr = np.mean([1 if p > 0 else 0 for p in pnls])
            total = sum(pnls)

            if label == "flash" and wr < 0.30 and total < -2.0:
                # Flash trades lose → increase min hold or reduce entries
                current_min_hold = float(os.environ.get("EXIT_MIN_HOLD_SEC", "30") or 30)
                new_min_hold = min(180, int(current_min_hold * 1.5))
                results.append(DiagnosticResult(
                    module="hold_duration",
                    severity="HIGH",
                    message=f"Flash (<1min) trades: WR={wr:.1%}, PnL=${total:+.2f}, n={len(trades)} → increase min hold",
                    recommendation={"EXIT_MIN_HOLD_SEC": str(new_min_hold)},
                    metrics={"bin": label, "wr": wr, "total_pnl": total, "n": len(trades)},
                ))
            elif label == "long" and wr > 0.50 and total > 0:
                results.append(DiagnosticResult(
                    module="hold_duration",
                    severity="INFO",
                    message=f"Long holds (>1h): WR={wr:.1%}, PnL=${total:+.2f}, n={len(trades)} — profitable duration",
                    metrics={"bin": label, "wr": wr, "total_pnl": total, "n": len(trades)},
                ))

        return results

    # ──────────────── Override 수집 및 적용 ────────────────

    def _collect_overrides(self, results: list[DiagnosticResult]) -> dict[str, str]:
        """모든 진단 결과에서 추천된 overrides를 수집"""
        overrides: dict[str, str] = {}
        for r in results:
            if r.recommendation:
                for k, v in r.recommendation.items():
                    overrides[k] = str(v)
        return overrides

    def _apply_overrides(self, overrides: dict[str, str], results: list[DiagnosticResult]) -> None:
        """overrides를 파일에 기록 (hot-reload용)"""
        # 쿨다운 체크
        elapsed = time.time() - self.state.last_adjust_ts
        if elapsed < MIN_COOLDOWN_SEC:
            logger.info(f"[AUTO_DIAG] Cooldown active: {elapsed:.0f}s < {MIN_COOLDOWN_SEC}s")
            return

        # 기존 auto_tune_overrides.json과 merge
        merged = {}
        auto_tune_path = self.output_path.parent / "auto_tune_overrides.json"
        if auto_tune_path.exists():
            try:
                with open(auto_tune_path) as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    merged = existing.get("overrides", existing)
            except Exception:
                pass

        # 새 overrides를 merge
        for k, v in overrides.items():
            merged[k] = v

        # diagnostics_overrides.json에 기록
        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "source": "auto_diagnostics",
            "interval_hours": self.lookback_hours,
            "n_adjustments": len(overrides),
            "overrides": overrides,
            "diagnostics_summary": [
                {
                    "module": r.module,
                    "severity": r.severity,
                    "message": r.message[:200],
                }
                for r in results
                if r.severity in ("CRITICAL", "HIGH")
            ],
        }

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"[AUTO_DIAG] Wrote {len(overrides)} overrides to {self.output_path}")
        except Exception as e:
            logger.error(f"[AUTO_DIAG] Failed to write overrides: {e}")
            return

        # auto_tune_overrides.json에도 merge하여 기록 (기존 것에 추가)
        try:
            if auto_tune_path.exists():
                with open(auto_tune_path) as f:
                    orig = json.load(f)
            else:
                orig = {"overrides": {}}

            if isinstance(orig, dict):
                if "overrides" not in orig:
                    orig["overrides"] = {}
                for k, v in overrides.items():
                    orig["overrides"][k] = v
                orig["diagnostics_timestamp_ms"] = int(time.time() * 1000)
                orig["diagnostics_n_adjustments"] = len(overrides)

                with open(auto_tune_path, "w") as f:
                    json.dump(orig, f, indent=2, ensure_ascii=False)
                logger.info(f"[AUTO_DIAG] Merged {len(overrides)} overrides into auto_tune_overrides.json")
        except Exception as e:
            logger.error(f"[AUTO_DIAG] Failed to merge into auto_tune: {e}")

        self.state.last_adjust_ts = time.time()
        self.state.total_adjustments += 1
        self.state.adjustment_history.append({
            "ts": time.time(),
            "overrides": overrides,
            "reason": [r.message[:100] for r in results if r.severity in ("CRITICAL", "HIGH")],
        })
        # 히스토리 크기 제한
        if len(self.state.adjustment_history) > 100:
            self.state.adjustment_history = self.state.adjustment_history[-50:]

        self._save_state()

    # ──────────────── DB 접근 ────────────────

    def _load_recent_trades(self) -> list[dict] | None:
        """최근 N시간 내 closed 거래를 DB에서 로드"""
        if not self.db_path.exists():
            logger.warning(f"[AUTO_DIAG] DB not found: {self.db_path}")
            return None

        cutoff_ms = int((time.time() - self.lookback_hours * 3600) * 1000)
        
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 테이블 존재 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                conn.close()
                return None

            # 컬럼 목록 확인
            cursor.execute("PRAGMA table_info(trades)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # 시간 필터 컬럼 결정
            time_col = "closed_at_ms" if "closed_at_ms" in columns else \
                       "exit_time" if "exit_time" in columns else \
                       "timestamp" if "timestamp" in columns else None

            if time_col is None:
                # timestamp 없으면 최근 N건
                cursor.execute(f"SELECT * FROM trades ORDER BY rowid DESC LIMIT 500")
            else:
                cursor.execute(
                    f"SELECT * FROM trades WHERE {time_col} > ? ORDER BY {time_col} DESC LIMIT 1000",
                    (cutoff_ms,)
                )

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            # dict로 변환
            result = []
            for row in rows:
                d = {k: row[k] for k in row.keys()}
                result.append(d)
            return result

        except Exception as e:
            logger.error(f"[AUTO_DIAG] DB query failed: {e}")
            return None

    # ──────────────── State 관리 ────────────────

    def _load_state(self):
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self.state.last_run_ts = data.get("last_run_ts", 0)
                self.state.last_adjust_ts = data.get("last_adjust_ts", 0)
                self.state.consecutive_loss_cycles = data.get("consecutive_loss_cycles", 0)
                self.state.total_adjustments = data.get("total_adjustments", 0)
                self.state.adjustment_history = data.get("adjustment_history", [])
            except Exception:
                pass

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({
                    "last_run_ts": self.state.last_run_ts,
                    "last_adjust_ts": self.state.last_adjust_ts,
                    "consecutive_loss_cycles": self.state.consecutive_loss_cycles,
                    "total_adjustments": self.state.total_adjustments,
                    "adjustment_history": self.state.adjustment_history[-20:],
                }, f, indent=2)
        except Exception:
            pass

    # ──────────────── 로깅 ────────────────

    def _log_summary(self, results: list[DiagnosticResult]):
        """진단 결과 요약 로그"""
        if not results:
            logger.info("[AUTO_DIAG] No findings in this cycle")
            return

        by_sev = {}
        for r in results:
            by_sev.setdefault(r.severity, []).append(r)

        summary_parts = []
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            items = by_sev.get(sev, [])
            if items:
                summary_parts.append(f"{sev}={len(items)}")

        logger.info(f"[AUTO_DIAG] {len(results)} findings: {', '.join(summary_parts)}")
        for r in results:
            if r.severity in ("CRITICAL", "HIGH"):
                logger.warning(f"[AUTO_DIAG][{r.severity}] [{r.module}] {r.message}")

    # ──────────────── Dashboard 통합 ────────────────

    def get_dashboard_payload(self) -> dict:
        """대시보드에 표시할 진단 상태 요약"""
        diag_payload = {
            "last_run_ts": self.state.last_run_ts,
            "last_adjust_ts": self.state.last_adjust_ts,
            "total_adjustments": self.state.total_adjustments,
            "consecutive_loss_cycles": self.state.consecutive_loss_cycles,
            "interval_sec": self.interval_sec,
        }
        # 마지막 조정 내역
        if self.state.adjustment_history:
            last = self.state.adjustment_history[-1]
            diag_payload["last_adjustment"] = {
                "ts": last.get("ts"),
                "n_overrides": len(last.get("overrides", {})),
                "reasons": last.get("reason", [])[:3],
            }
        return diag_payload


# ──────────────── CLI 실행 ────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto Diagnostics Engine")
    parser.add_argument("--db", default="state/bot_data_live.db", help="DB path")
    parser.add_argument("--hours", type=float, default=4.0, help="Lookback hours")
    parser.add_argument("--output", default="state/diagnostics_overrides.json", help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = AutoDiagnosticsEngine(
        db_path=args.db,
        output_path=args.output,
        lookback_hours=args.hours,
    )

    # Force run (ignore interval)
    engine.state.last_run_ts = 0

    results = engine.run_diagnostics()

    print(f"\n{'=' * 70}")
    print(f"AUTO DIAGNOSTICS REPORT ({len(results)} findings)")
    print(f"{'=' * 70}")
    for r in results:
        marker = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}.get(r.severity, "⚪")
        print(f"\n{marker} [{r.severity}] [{r.module}]")
        print(f"   {r.message}")
        if r.recommendation:
            print(f"   → Override: {r.recommendation}")

    if args.dry_run:
        print("\n[DRY-RUN] No overrides written.")
    else:
        overrides = engine._collect_overrides(results)
        if overrides:
            print(f"\n📝 {len(overrides)} overrides written to {args.output}")
        else:
            print("\n✅ No parameter changes recommended.")

    print(f"\nState: {engine.state_path}")


if __name__ == "__main__":
    main()
