#!/usr/bin/env python3
"""
Trade Logic Diagnostic System (통계 진단 시스템)
=================================================
모든 과거 매매 데이터를 분석하여 현재 매매 로직의 문제점을 자동 진단하고
수정방안을 제시하는 시스템.

Usage:
    python3 scripts/trade_diagnostics.py [--db state/bot_data_live.db] [--output /tmp/diagnostics_report.txt]

Output:
    - 콘솔: 요약 리포트
    - /tmp/diagnostics_report.txt: 상세 리포트 
    - JSON: /tmp/diagnostics_params.json: 추천 파라미터 
"""
import sqlite3
import numpy as np
import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ==============================================================================
# DIAGNOSTIC MODULE REGISTRY
# ==============================================================================

class DiagnosticResult:
    """Single diagnostic finding"""
    def __init__(self, severity: str, category: str, finding: str, 
                 evidence: str, recommendation: str, param_changes: dict = None):
        self.severity = severity  # CRITICAL / HIGH / MEDIUM / LOW / INFO
        self.category = category
        self.finding = finding
        self.evidence = evidence
        self.recommendation = recommendation
        self.param_changes = param_changes or {}

    def __repr__(self):
        return f"[{self.severity}] {self.category}: {self.finding}"


class TradeDiagnostics:
    """Main diagnostic engine"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.findings: list[DiagnosticResult] = []
        self.param_recommendations: dict = {}
        self._load_data()
    
    def _load_data(self):
        """Load trade data"""
        self.exits = [dict(r) for r in self.conn.execute("""
            SELECT * FROM trades
            WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
              AND action IN ('EXIT', 'REBAL_EXIT')
            ORDER BY timestamp_ms ASC
        """).fetchall()]
        
        self.entries = [dict(r) for r in self.conn.execute("""
            SELECT * FROM trades
            WHERE action IN ('LONG', 'SHORT')
            ORDER BY timestamp_ms ASC
        """).fetchall()]
        
        self.equity = [dict(r) for r in self.conn.execute("""
            SELECT * FROM equity_history
            ORDER BY timestamp_ms ASC
        """).fetchall()]
        
        print(f"Loaded: {len(self.exits)} exits, {len(self.entries)} entries, {len(self.equity)} equity snapshots")
    
    # ==========================================================================
    # DIAGNOSTIC 1: Direction Accuracy
    # ==========================================================================
    def diag_direction_accuracy(self):
        """Check overall direction hit rate and breakeven requirements"""
        if not self.exits:
            return
        
        wins = [t for t in self.exits if float(t['realized_pnl']) > 0]
        losses = [t for t in self.exits if float(t['realized_pnl']) < 0]
        wr = len(wins) / len(self.exits) if self.exits else 0
        
        avg_win = np.mean([float(t['realized_pnl']) for t in wins]) if wins else 0
        avg_loss = np.mean([float(t['realized_pnl']) for t in losses]) if losses else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        be_wr = abs(avg_loss) / (abs(avg_loss) + avg_win) if (avg_win + abs(avg_loss)) > 0 else 0.5
        
        total_pnl = sum(float(t['realized_pnl']) for t in self.exits)
        
        self.findings.append(DiagnosticResult(
            severity="CRITICAL" if wr < be_wr else "INFO",
            category="Direction",
            finding=f"WR={wr:.1%} vs Breakeven={be_wr:.1%} (gap={wr-be_wr:+.1%}), Payoff={payoff:.3f}",
            evidence=f"n={len(self.exits)}, avg_win=${avg_win:.4f}, avg_loss=${avg_loss:.4f}, total=${total_pnl:.2f}",
            recommendation="WR 개선 또는 Payoff ratio 개선 필요" if wr < be_wr else "정상 범위"
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 2: Confidence Calibration
    # ==========================================================================
    def diag_confidence_calibration(self):
        """Check if predicted confidence matches actual hit rate"""
        cal_data = []
        for t in self.exits:
            conf = t.get('pred_mu_dir_conf')
            if conf is None: continue
            c = float(conf)
            pnl = float(t['realized_pnl'])
            cal_data.append((c, 1 if pnl > 0 else 0))
        
        if len(cal_data) < 100:
            return
        
        confs = np.array([x[0] for x in cal_data])
        hits = np.array([x[1] for x in cal_data])
        corr = np.corrcoef(confs, hits)[0, 1]
        
        # Calibration by bucket
        buckets = defaultdict(lambda: {"hit": 0, "total": 0})
        for c, h in cal_data:
            b = round(c * 10) / 10  # 0.1 resolution
            buckets[b]["hit"] += h
            buckets[b]["total"] += 1
        
        miscal_count = 0
        for b, data in buckets.items():
            if data["total"] >= 20:
                actual_hr = data["hit"] / data["total"]
                gap = abs(actual_hr - b)
                if gap > 0.15:
                    miscal_count += 1
        
        severity = "CRITICAL" if corr < -0.03 else "HIGH" if miscal_count > 3 else "MEDIUM"
        
        self.findings.append(DiagnosticResult(
            severity=severity,
            category="Confidence",
            finding=f"Corr(conf, hit)={corr:+.4f}, {miscal_count} miscalibrated buckets",
            evidence=f"n={len(cal_data)} trades with confidence data",
            recommendation="확신도 모델 비활성화 또는 재학습 필요" if corr < 0 else "확신도 모델 정상",
            param_changes={"DIRECTION_CONF_DISABLED": "1"} if corr < -0.03 else {}
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 3: mu_alpha Alignment
    # ==========================================================================
    def diag_mu_alignment(self):
        """Check if mu_alpha sign matches trading direction"""
        aligned_pnl = []
        misaligned_pnl = []
        
        for t in self.exits:
            mu = t.get('pred_mu_alpha')
            side = str(t.get('side') or '').upper()
            pnl = float(t['realized_pnl'])
            if mu is None or side not in ('LONG', 'SHORT'):
                continue
            m = float(mu)
            if (m > 0 and side == "LONG") or (m < 0 and side == "SHORT"):
                aligned_pnl.append(pnl)
            else:
                misaligned_pnl.append(pnl)
        
        total = len(aligned_pnl) + len(misaligned_pnl)
        if total < 100:
            return
        
        misalign_pct = len(misaligned_pnl) / total
        aligned_total = sum(aligned_pnl)
        misaligned_total = sum(misaligned_pnl)
        
        aligned_wr = sum(1 for p in aligned_pnl if p > 0) / len(aligned_pnl) if aligned_pnl else 0
        misaligned_wr = sum(1 for p in misaligned_pnl if p > 0) / len(misaligned_pnl) if misaligned_pnl else 0
        
        severity = "CRITICAL" if misalign_pct > 0.5 else "HIGH" if misalign_pct > 0.3 else "MEDIUM"
        
        self.findings.append(DiagnosticResult(
            severity=severity,
            category="mu_alpha",
            finding=f"Misaligned: {misalign_pct:.1%} of trades, WR(aligned)={aligned_wr:.1%} vs WR(misaligned)={misaligned_wr:.1%}",
            evidence=f"aligned: n={len(aligned_pnl)} total=${aligned_total:.2f} | misaligned: n={len(misaligned_pnl)} total=${misaligned_total:.2f}",
            recommendation="mu_alpha 부호 정합성 하드 게이트 추가 필요" if misalign_pct > 0.3 else "정상",
            param_changes={"MU_ALPHA_DIRECTION_GATE": "1"} if misalign_pct > 0.3 else {}
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 4: Regime × Side Profitability
    # ==========================================================================
    def diag_regime_side(self):
        """Find losing regime×side combinations"""
        combos = defaultdict(list)
        for t in self.exits:
            regime = str(t.get('regime') or '?').lower()
            side = str(t.get('side') or '?').upper()
            pnl = float(t['realized_pnl'])
            combos[f"{regime}_{side}"].append(pnl)
        
        block_list = []
        for key, data in sorted(combos.items(), key=lambda x: sum(x[1])):
            if len(data) < 20:
                continue
            w = sum(1 for p in data if p > 0)
            wr = w / len(data)
            total = sum(data)
            avg = np.mean(data)
            
            if total < -10 and wr < 0.40:
                block_list.append(key)
                self.findings.append(DiagnosticResult(
                    severity="HIGH" if total < -50 else "MEDIUM",
                    category="Regime×Side",
                    finding=f"{key}: n={len(data)}, WR={wr:.1%}, total=${total:.2f}",
                    evidence=f"avg=${avg:.4f}, {len(data)} trades",
                    recommendation=f"'{key}' 조합 진입 차단 또는 사이징 축소"
                ))
        
        if block_list:
            self.param_recommendations["REGIME_SIDE_BLOCK_LIST"] = ",".join(block_list)
    
    # ==========================================================================
    # DIAGNOSTIC 5: Notional/Leverage Sizing
    # ==========================================================================
    def diag_sizing(self):
        """Analyze notional sizing impact on PnL"""
        buckets = defaultdict(list)
        for t in self.exits:
            not_val = float(t.get('notional') or 0)
            pnl = float(t['realized_pnl'])
            if not_val < 20: b = "<$20"
            elif not_val < 50: b = "$20-50"
            elif not_val < 100: b = "$50-100"
            elif not_val < 200: b = "$100-200"
            elif not_val < 500: b = "$200-500"
            else: b = "$500+"
            buckets[b].append(pnl)
        
        for b in ["$500+", "$200-500", "$100-200"]:
            data = buckets.get(b, [])
            if len(data) < 10:
                continue
            total = sum(data)
            wins = [p for p in data if p > 0]
            losses = [p for p in data if p < 0]
            payoff = abs(np.mean(wins) / np.mean(losses)) if losses else 0
            
            if total < -50 or payoff < 1.0:
                self.findings.append(DiagnosticResult(
                    severity="HIGH" if total < -100 else "MEDIUM",
                    category="Sizing",
                    finding=f"Notional {b}: payoff={payoff:.3f}, total=${total:.2f}",
                    evidence=f"n={len(data)}, avg_win=${np.mean(wins) if wins else 0:.4f}, avg_loss=${np.mean(losses) if losses else 0:.4f}",
                    recommendation=f"Notional {b} 제한 강화 필요"
                ))
    
    # ==========================================================================
    # DIAGNOSTIC 6: Hold Duration Optimization
    # ==========================================================================
    def diag_hold_duration(self):
        """Find optimal hold duration by regime"""
        for regime_filter in ['chop', 'bull', 'bear']:
            reg_exits = [t for t in self.exits if str(t.get('regime','')).lower() == regime_filter]
            if len(reg_exits) < 50:
                continue
            
            short_hold = [t for t in reg_exits 
                          if t.get('hold_duration_sec') is not None 
                          and float(t['hold_duration_sec']) < 60]
            long_hold = [t for t in reg_exits 
                         if t.get('hold_duration_sec') is not None 
                         and 300 <= float(t['hold_duration_sec']) <= 3600]
            
            if short_hold and long_hold:
                short_wr = sum(1 for t in short_hold if float(t['realized_pnl']) > 0) / len(short_hold)
                long_wr = sum(1 for t in long_hold if float(t['realized_pnl']) > 0) / len(long_hold)
                short_total = sum(float(t['realized_pnl']) for t in short_hold)
                long_total = sum(float(t['realized_pnl']) for t in long_hold)
                
                if short_total < -10 and long_total > 0:
                    self.findings.append(DiagnosticResult(
                        severity="MEDIUM",
                        category="HoldDuration",
                        finding=f"{regime_filter}: <1min loses ${short_total:.2f} (WR={short_wr:.1%}), 5-60min gains ${long_total:.2f} (WR={long_wr:.1%})",
                        evidence=f"n_short={len(short_hold)}, n_long={len(long_hold)}",
                        recommendation=f"{regime_filter} 레짐에서 최소 보유 시간 증가 권장"
                    ))
    
    # ==========================================================================
    # DIAGNOSTIC 7: Time-of-Day Analysis
    # ==========================================================================
    def diag_time_of_day(self):
        """Find losing hours"""
        hour_pnl = defaultdict(list)
        for t in self.exits:
            ts = t.get('timestamp_ms')
            pnl = float(t['realized_pnl'])
            if ts:
                dt = datetime.fromtimestamp(float(ts)/1000, tz=timezone.utc)
                hour_pnl[dt.hour].append(pnl)
        
        bad_hours = []
        for h in range(24):
            data = hour_pnl.get(h, [])
            if len(data) >= 30:
                total = sum(data)
                wr = sum(1 for p in data if p > 0) / len(data)
                if total < -100 and wr < 0.35:
                    bad_hours.append(h)
                    self.findings.append(DiagnosticResult(
                        severity="MEDIUM",
                        category="TimeOfDay",
                        finding=f"UTC {h:02d}:00 — WR={wr:.1%}, total=${total:.2f}, n={len(data)}",
                        evidence=f"avg=${np.mean(data):.4f}",
                        recommendation=f"UTC {h:02d} 시에 진입 회피 또는 사이징 축소"
                    ))
        
        if bad_hours:
            self.param_recommendations["TRADING_BAD_HOURS_UTC"] = ",".join(str(h) for h in bad_hours)
    
    # ==========================================================================
    # DIAGNOSTIC 8: Entry Quality Score Effectiveness
    # ==========================================================================
    def diag_entry_quality(self):
        """Check if entry_quality_score actually predicts PnL"""
        eq_data = []
        for t in self.exits:
            eq = t.get('entry_quality_score')
            pnl = float(t['realized_pnl'])
            if eq is not None:
                eq_data.append((float(eq), pnl))
        
        if len(eq_data) < 100:
            return
        
        scores = np.array([x[0] for x in eq_data])
        pnls = np.array([x[1] for x in eq_data])
        corr = np.corrcoef(scores, pnls)[0, 1]
        
        # Compare low vs high quality
        low_q = [p for s, p in eq_data if s < 0.3]
        high_q = [p for s, p in eq_data if s >= 0.5]
        
        if low_q and high_q:
            low_total = sum(low_q)
            high_total = sum(high_q)
            
            if low_total < -20:
                opt_threshold = 0.3
                # Try to find better threshold
                for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    below = [p for s, p in eq_data if s < thr]
                    if below and sum(below) < -10:
                        opt_threshold = thr
                        break
                
                self.findings.append(DiagnosticResult(
                    severity="MEDIUM",
                    category="EntryQuality",
                    finding=f"Corr(quality, pnl)={corr:+.4f}, Low(<0.3): ${low_total:.2f}, High(≥0.5): ${high_total:.2f}",
                    evidence=f"n_low={len(low_q)}, n_high={len(high_q)}",
                    recommendation=f"entry_quality_score ≥ {opt_threshold} 최소 임계치 설정",
                    param_changes={"MIN_ENTRY_QUALITY_SCORE": str(opt_threshold)}
                ))
    
    # ==========================================================================
    # DIAGNOSTIC 9: Entry EV Effectiveness  
    # ==========================================================================
    def diag_entry_ev(self):
        """Check if entry EV predicts actual profitability"""
        ev_data = []
        for t in self.exits:
            ev = t.get('entry_ev')
            pnl = float(t['realized_pnl'])
            if ev is not None:
                ev_data.append((float(ev), pnl))
        
        if len(ev_data) < 100:
            return
        
        evs = np.array([x[0] for x in ev_data])
        pnls = np.array([x[1] for x in ev_data])
        corr = np.corrcoef(evs, pnls)[0, 1]
        
        # Find optimal EV threshold
        best_thr = 0
        best_edge = -float('inf')
        for thr in np.arange(-0.01, 0.15, 0.005):
            above = [p for e, p in ev_data if e >= thr]
            if len(above) >= 50:
                avg_pnl = np.mean(above)
                if avg_pnl > best_edge:
                    best_edge = avg_pnl
                    best_thr = thr
        
        self.findings.append(DiagnosticResult(
            severity="HIGH" if corr < 0.01 else "INFO",
            category="EntryEV",
            finding=f"Corr(entry_ev, pnl)={corr:+.4f}, Optimal EV threshold={best_thr:.4f} → avg_pnl=${best_edge:.4f}",
            evidence=f"n={len(ev_data)} trades with EV data",
            recommendation=f"ENTRY_GROSS_EV_MIN={best_thr:.4f} 설정 권장" if best_thr > 0 else "EV 임계치 정상",
            param_changes={"ENTRY_GROSS_EV_MIN": f"{best_thr:.4f}"} if best_thr > 0.001 else {}
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 10: Drawdown & Risk Metrics
    # ==========================================================================
    def diag_risk_metrics(self):
        """Compute drawdown, Sharpe, and risk-adjusted metrics"""
        if len(self.equity) < 10:
            return
        
        equities = [float(e.get('total_equity') or 0) for e in self.equity if float(e.get('total_equity') or 0) > 0]
        if not equities:
            return
        
        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # Recent trend (last 1000 points)
        recent = equities[-min(1000, len(equities)):]
        trend = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
        
        self.findings.append(DiagnosticResult(
            severity="HIGH" if max_dd > 0.5 else "MEDIUM" if max_dd > 0.3 else "INFO",
            category="Risk",
            finding=f"Max Drawdown={max_dd:.1%}, Recent trend={trend:+.1%}",
            evidence=f"Peak equity=${max(equities):.2f}, Current=${equities[-1]:.2f}",
            recommendation="심각한 드로다운 — 포지션 사이징 축소 필요" if max_dd > 0.5 else "정상"
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 11: Consecutive Losses & Tilt Detection
    # ==========================================================================
    def diag_streaks(self):
        """Detect dangerous loss streaks"""
        pnls = [float(t['realized_pnl']) for t in self.exits]
        
        max_loss_streak = 0
        current_streak = 0
        for p in pnls:
            if p < 0:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0
        
        # After long loss streaks, does behavior change (tilt)?
        # Check if trades get bigger after losses
        window = 10
        for i in range(window, len(pnls) - 1):
            recent_streak = sum(1 for p in pnls[i-window:i] if p < 0)
        
        self.findings.append(DiagnosticResult(
            severity="HIGH" if max_loss_streak > 20 else "MEDIUM" if max_loss_streak > 10 else "LOW",
            category="Streaks",
            finding=f"Max consecutive losses: {max_loss_streak}",
            evidence=f"Total trades: {len(pnls)}",
            recommendation="연속 손절 방지 circuit breaker 추가" if max_loss_streak > 15 else "정상"
        ))
    
    # ==========================================================================
    # DIAGNOSTIC 12: Symbol-Level Analysis
    # ==========================================================================
    def diag_symbols(self):
        """Find consistently losing symbols"""
        sym_data = defaultdict(list)
        for t in self.exits:
            sym = str(t.get('symbol') or '?')
            pnl = float(t['realized_pnl'])
            sym_data[sym].append(pnl)
        
        bad_symbols = []
        for sym, data in sorted(sym_data.items(), key=lambda x: sum(x[1])):
            if len(data) < 20:
                continue
            total = sum(data)
            wr = sum(1 for p in data if p > 0) / len(data)
            if total < -10 and wr < 0.35:
                bad_symbols.append(sym)
                self.findings.append(DiagnosticResult(
                    severity="MEDIUM",
                    category="Symbol",
                    finding=f"{sym}: n={len(data)}, WR={wr:.1%}, total=${total:.2f}",
                    evidence=f"avg=${np.mean(data):.4f}",
                    recommendation=f"{sym} 블랙리스트 추가 고려"
                ))
    
    # ==========================================================================
    # RUN ALL DIAGNOSTICS
    # ==========================================================================
    def run_all(self):
        """Run all diagnostic checks"""
        print("\n" + "="*70)
        print("  TRADE LOGIC DIAGNOSTICS — 매매 로직 자동 진단 시스템")
        print("="*70)
        
        diagnostics = [
            ("Direction Accuracy", self.diag_direction_accuracy),
            ("Confidence Calibration", self.diag_confidence_calibration),
            ("mu_alpha Alignment", self.diag_mu_alignment),
            ("Regime × Side", self.diag_regime_side),
            ("Sizing Analysis", self.diag_sizing),
            ("Hold Duration", self.diag_hold_duration),
            ("Time of Day", self.diag_time_of_day),
            ("Entry Quality Score", self.diag_entry_quality),
            ("Entry EV", self.diag_entry_ev),
            ("Risk Metrics", self.diag_risk_metrics),
            ("Loss Streaks", self.diag_streaks),
            ("Symbol Analysis", self.diag_symbols),
        ]
        
        for name, func in diagnostics:
            try:
                func()
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        self.findings.sort(key=lambda f: severity_order.get(f.severity, 5))
        
        return self.findings
    
    def print_report(self):
        """Print formatted report"""
        print("\n" + "="*70)
        print("  DIAGNOSTIC FINDINGS — 진단 결과")
        print("="*70)
        
        for i, f in enumerate(self.findings, 1):
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}.get(f.severity, "❓")
            print(f"\n{icon} [{f.severity}] #{i} — {f.category}")
            print(f"   Finding: {f.finding}")
            print(f"   Evidence: {f.evidence}")
            print(f"   Fix: {f.recommendation}")
            if f.param_changes:
                for k, v in f.param_changes.items():
                    print(f"   → Set {k}={v}")
        
        # Aggregate parameter recommendations
        all_params = {}
        for f in self.findings:
            all_params.update(f.param_changes)
        all_params.update(self.param_recommendations)
        
        if all_params:
            print("\n" + "="*70)
            print("  RECOMMENDED PARAMETER CHANGES — 추천 파라미터 변경")
            print("="*70)
            for k, v in sorted(all_params.items()):
                print(f"  {k}={v}")
            
            # Save to JSON
            json_path = "/tmp/diagnostics_params.json"
            with open(json_path, "w") as fp:
                json.dump(all_params, fp, indent=2)
            print(f"\n  → Saved to {json_path}")
        
        # Summary counts
        counts = defaultdict(int)
        for f in self.findings:
            counts[f.severity] += 1
        
        print("\n" + "="*70)
        print(f"  SUMMARY: {counts.get('CRITICAL',0)} CRITICAL, {counts.get('HIGH',0)} HIGH, "
              f"{counts.get('MEDIUM',0)} MEDIUM, {counts.get('LOW',0)} LOW, {counts.get('INFO',0)} INFO")
        print("="*70)
    
    def save_report(self, path: str):
        """Save detailed report to file"""
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.print_report()
        sys.stdout = old_stdout
        
        with open(path, "w") as f:
            f.write(buffer.getvalue())
        print(f"\nReport saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Trade Logic Diagnostic System")
    parser.add_argument("--db", default="state/bot_data_live.db", help="Path to SQLite database")
    parser.add_argument("--output", default="/tmp/diagnostics_report.txt", help="Output report path")
    args = parser.parse_args()
    
    diag = TradeDiagnostics(args.db)
    diag.run_all()
    diag.print_report()
    diag.save_report(args.output)


if __name__ == "__main__":
    main()
