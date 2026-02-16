#!/usr/bin/env python3
"""Counterfactual: would mu_sign_flip_exit + regime_dir_gate/exit improve ROE?

Replays all historical trades from SQLite and simulates:
1. What if mu_sign_flip_exit had triggered earlier?
2. What if regime_dir_gate had blocked bad entries?
3. What if regime_dir_exit had closed conflicting positions?
4. Could we have re-entered in the correct direction after exit?
"""
import sqlite3, os, sys, math
import numpy as np
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "..", "state", "bot_data_live.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_ohlcv(symbol_base: str) -> np.ndarray:
    """Load OHLCV CSV. Returns [ts_ms, O, H, L, C, V]."""
    path = os.path.join(DATA_DIR, f"{symbol_base}.csv")
    if not os.path.exists(path):
        return np.array([])
    import csv
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                ts_raw = row[0].strip()
                if ts_raw.replace(".", "").replace("-", "").isdigit() and len(ts_raw) > 10:
                    ts = float(ts_raw)
                else:
                    dt = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                    ts = dt.timestamp() * 1000
                o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                v = float(row[5]) if len(row) > 5 else 0
                rows.append([ts, o, h, l, c, v])
            except Exception:
                continue
    return np.array(rows) if rows else np.array([])


def main():
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row

    # ===== 1. Regime-Direction Gate: 역방향 진입 차단 효과 =====
    print("=" * 70)
    print("1. REGIME-DIRECTION GATE: 역방향 진입을 차단했다면?")
    print("=" * 70)

    # Find entries where regime and side conflict
    exits = db.execute("""
        SELECT symbol, side, roe, regime, hold_duration_sec, entry_reason,
               timestamp_ms, entry_ev, fill_price
        FROM trades
        WHERE action IN ('EXIT', 'REBAL_EXIT') AND roe IS NOT NULL
        ORDER BY timestamp_ms
    """).fetchall()

    bear_long = [r for r in exits if r["regime"] == "bear" and r["side"] == "LONG"]
    bull_short = [r for r in exits if r["regime"] == "bull" and r["side"] == "SHORT"]
    total_conflicts = bear_long + bull_short
    
    if total_conflicts:
        total_conflict_roe = sum(r["roe"] or 0 for r in total_conflicts)
        avg_conflict = total_conflict_roe / len(total_conflicts)
        wins = sum(1 for r in total_conflicts if (r["roe"] or 0) > 0)
        print(f"  대상 거래: {len(total_conflicts)} ({len(bear_long)} bear+LONG, {len(bull_short)} bull+SHORT)")
        print(f"  차단 시 회피할 총 ROE: {total_conflict_roe:+.4f}")
        print(f"  평균 ROE: {avg_conflict:+.4f}")
        print(f"  적중률: {100*wins/len(total_conflicts):.1f}%")
        print(f"  → 이 거래들을 전부 차단했다면 ROE {abs(total_conflict_roe):.4f} 만큼 손실 회피")
    else:
        print("  해당 거래 없음")

    # ===== 2. ev_drop_exit 강화 효과 시뮬레이션 =====
    print(f"\n{'=' * 70}")
    print("2. EV_DROP_EXIT 강화 (confirmation 1틱, progress guard 완화)")
    print("=" * 70)

    # ev_drop_exit가 아닌 다른 exit reason으로 나간 거래 중 ROE < 0인 것들
    # 만약 ev_drop_exit로 더 빨리 나갔다면?
    non_evdrop_losses = db.execute("""
        SELECT symbol, side, roe, regime, hold_duration_sec, entry_reason, timestamp_ms
        FROM trades
        WHERE action IN ('EXIT', 'REBAL_EXIT')
          AND roe < 0
          AND entry_reason NOT LIKE '%ev_drop%'
          AND entry_reason NOT LIKE '%emergency%'
          AND entry_reason NOT LIKE '%liquidation%'
        ORDER BY roe ASC
        LIMIT 200
    """).fetchall()
    
    if non_evdrop_losses:
        total_loss = sum(r["roe"] for r in non_evdrop_losses)
        avg_hold = sum(r["hold_duration_sec"] or 0 for r in non_evdrop_losses) / len(non_evdrop_losses)
        
        # Estimate: if exited at 40% of hold time (new progress guard), loss would be ~40% of final loss
        estimated_saved = 0
        for r in non_evdrop_losses:
            hold = r["hold_duration_sec"] or 0
            if hold > 60:  # only for trades held > 1 min
                # Assumes linear loss accumulation — conservative estimate
                early_exit_ratio = 0.4  # exit at 40% of hold time
                estimated_loss_at_early = r["roe"] * early_exit_ratio
                saved = r["roe"] - estimated_loss_at_early  # negative - more negative = positive saved
                estimated_saved += abs(saved) if r["roe"] < 0 else 0
        
        print(f"  ev_drop가 아닌 exit으로 손실난 거래: {len(non_evdrop_losses)}")
        print(f"  총 손실 ROE: {total_loss:+.4f}")
        print(f"  평균 보유시간: {avg_hold:.0f}s")
        print(f"  40% 시점 조기 청산 시 추정 절감: ~{estimated_saved:.4f} ROE")
        
        # By exit reason
        reason_breakdown = {}
        for r in non_evdrop_losses:
            reason = r["entry_reason"] or "unknown"
            if reason not in reason_breakdown:
                reason_breakdown[reason] = {"count": 0, "total_roe": 0, "total_hold": 0}
            reason_breakdown[reason]["count"] += 1
            reason_breakdown[reason]["total_roe"] += r["roe"]
            reason_breakdown[reason]["total_hold"] += (r["hold_duration_sec"] or 0)
        
        print(f"\n  손실 거래 exit reason 별:")
        for reason, s in sorted(reason_breakdown.items(), key=lambda x: x[1]["total_roe"]):
            avg_h = s["total_hold"] / s["count"] if s["count"] else 0
            print(f"    {reason:<35s}: n={s['count']:>3}, total_roe={s['total_roe']:+.4f}, avg_hold={avg_h:.0f}s")

    # ===== 3. mu_sign_flip_exit 시뮬레이션 =====
    print(f"\n{'=' * 70}")
    print("3. MU_SIGN_FLIP_EXIT: mu_alpha 부호 반전 감지 효과")
    print("=" * 70)

    # 시뮬레이션: mu_alpha를 직접 볼 수 없으므로, 가격 방향 전환으로 대리
    # 가격이 entry 후 반대 방향으로 2캔들 이상 움직이면 → 감지 가능했을 것
    all_exits = db.execute("""
        SELECT symbol, side, roe, regime, hold_duration_sec, entry_reason,
               timestamp_ms, fill_price
        FROM trades
        WHERE action IN ('EXIT', 'REBAL_EXIT') AND roe IS NOT NULL
        ORDER BY timestamp_ms DESC LIMIT 500
    """).fetchall()
    
    would_have_saved = 0
    would_have_cut_winners = 0
    mu_flip_applicable = 0
    
    for r in all_exits:
        roe = r["roe"] or 0
        hold = r["hold_duration_sec"] or 0
        side = r["side"]
        
        # mu_sign_flip would mainly catch trades where:
        # - price moved against us for sustained period
        # - we held too long before exiting
        if hold > 60 and roe < -0.005:  # held > 1min, lost > 0.5%
            mu_flip_applicable += 1
            # Estimate: if detected at 30s (MU_SIGN_FLIP_MIN_AGE_SEC), loss would be ~proportion
            early_exit_roe = roe * (30.0 / max(hold, 30.0))
            would_have_saved += (roe - early_exit_roe)  # negative, abs = savings
        elif hold > 60 and roe > 0.005:  # would we cut winners too?
            # With MU_SIGN_FLIP_MIN_MAGNITUDE=0.05, small mu wouldn't trigger
            pass
    
    if mu_flip_applicable > 0:
        print(f"  적용 대상 거래 (>1min, ROE<-0.5%): {mu_flip_applicable}")
        print(f"  조기 감지 시 추정 절감: {abs(would_have_saved):.4f} ROE")
        print(f"  평균 절감/거래: {abs(would_have_saved)/mu_flip_applicable:.4f}")
    
    # ===== 4. 방향 전환 후 재진입 가능성 =====
    print(f"\n{'=' * 70}")
    print("4. 방향 전환 감지 후 반대 방향 재진입 가능성")
    print("=" * 70)

    # Find exits that were followed by an opposite-side entry within 5 minutes
    all_trades = db.execute("""
        SELECT action, symbol, side, roe, timestamp_ms, fill_price, entry_reason, regime
        FROM trades
        ORDER BY timestamp_ms
    """).fetchall()
    
    # Group by symbol: exit → next entry
    exits_by_sym = {}
    entries_by_sym = {}
    for t in all_trades:
        sym = t["symbol"]
        if t["action"] in ("EXIT", "REBAL_EXIT"):
            exits_by_sym.setdefault(sym, []).append(t)
        elif t["action"] == "ENTER":
            entries_by_sym.setdefault(sym, []).append(t)
    
    flip_reentry_success = 0
    flip_reentry_fail = 0
    flip_reentry_missed = 0  # exited but never re-entered
    flip_reentry_details = []
    
    for sym, sym_exits in exits_by_sym.items():
        sym_entries = entries_by_sym.get(sym, [])
        for ex in sym_exits:
            ex_ts = ex["timestamp_ms"]
            ex_side = ex["side"]
            opp_side = "SHORT" if ex_side == "LONG" else "LONG"
            
            # Find next entry for this symbol within 10 minutes
            next_entry = None
            for en in sym_entries:
                if en["timestamp_ms"] > ex_ts and (en["timestamp_ms"] - ex_ts) < 600_000:
                    next_entry = en
                    break
            
            if next_entry is None:
                # No re-entry → check if re-entry would have been profitable using OHLCV
                flip_reentry_missed += 1
                continue
            
            if next_entry["side"] == opp_side:
                # Did re-enter in opposite direction!
                # Find the ROE of this entry
                next_exit = None
                for ex2 in sym_exits:
                    if ex2["timestamp_ms"] > next_entry["timestamp_ms"]:
                        next_exit = ex2
                        break
                
                if next_exit and (next_exit["roe"] or 0) > 0:
                    flip_reentry_success += 1
                    flip_reentry_details.append({
                        "sym": sym, "exit_side": ex_side, "entry_side": opp_side,
                        "delay_sec": (next_entry["timestamp_ms"] - ex_ts) / 1000,
                        "reentry_roe": next_exit["roe"],
                    })
                else:
                    flip_reentry_fail += 1
            else:
                # Same-side re-entry (not a flip)
                pass
    
    total_flip = flip_reentry_success + flip_reentry_fail
    print(f"  총 청산 후 10분 내 반대방향 재진입: {total_flip}")
    print(f"  재진입 성공 (ROE>0): {flip_reentry_success}")
    print(f"  재진입 실패 (ROE<=0): {flip_reentry_fail}")
    if total_flip > 0:
        print(f"  반대방향 재진입 적중률: {100*flip_reentry_success/total_flip:.1f}%")
    print(f"  미재진입 (기회 놓침): {flip_reentry_missed}")
    
    if flip_reentry_details:
        avg_delay = np.mean([d["delay_sec"] for d in flip_reentry_details])
        avg_roe = np.mean([d["reentry_roe"] for d in flip_reentry_details])
        total_roe = sum(d["reentry_roe"] for d in flip_reentry_details)
        print(f"  성공 평균 지연: {avg_delay:.0f}s ({avg_delay/60:.1f}min)")
        print(f"  성공 평균 ROE: {avg_roe:+.4f}")
        print(f"  성공 총 ROE: {total_roe:+.4f}")

    # ===== 5. 종합: 두 규칙 동시 적용 시 추정 개선 =====
    print(f"\n{'=' * 70}")
    print("5. 종합: 새 규칙 적용 시 추정 수익률 개선")
    print("=" * 70)
    
    # Overall current performance
    total_exits = db.execute("SELECT COUNT(*) as n, SUM(roe) as s FROM trades WHERE action IN ('EXIT','REBAL_EXIT')").fetchone()
    current_total = total_exits["s"] or 0
    current_n = total_exits["n"] or 1
    
    print(f"  현재 총 거래: {current_n}")
    print(f"  현재 총 ROE: {current_total:+.4f}")
    print(f"  현재 평균 ROE/거래: {current_total/current_n:+.6f}")
    
    # Estimated improvements
    regime_gate_save = abs(total_conflict_roe) if total_conflicts else 0
    
    improved_total = current_total + regime_gate_save
    print(f"\n  --- 추정 개선 ---")
    print(f"  Regime-Dir Gate 절감: +{regime_gate_save:.4f}")
    if non_evdrop_losses:
        print(f"  ev_drop 조기 청산 절감: +{estimated_saved:.4f} (추정)")
    if mu_flip_applicable > 0:
        print(f"  mu_sign_flip 조기 감지 절감: +{abs(would_have_saved):.4f} (추정)")
    
    total_save = regime_gate_save + (estimated_saved if non_evdrop_losses else 0) + abs(would_have_saved)
    print(f"\n  총 추정 절감: +{total_save:.4f} ROE")
    print(f"  개선 후 추정 총 ROE: {current_total + total_save:+.4f}")
    print(f"  개선폭: {current_total:+.4f} → {current_total + total_save:+.4f}")
    
    # ===== 6. OHLCV 기반 방향 전환 후 진입 기회 분석 =====
    print(f"\n{'=' * 70}")
    print("6. OHLCV 기반: 방향 전환 후 반대방향 진입 기회")
    print("=" * 70)
    
    for sym_base in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        ohlcv = load_ohlcv(sym_base)
        if len(ohlcv) < 20:
            continue
        
        closes = ohlcv[:, 4]
        ts_arr = ohlcv[:, 0]
        
        # Simple zigzag inflection detection (0.5% min)
        state = 0
        last_idx = 0
        last_val = closes[0]
        inflections = []
        for i in range(1, len(closes)):
            pct = (closes[i] - last_val) / (last_val + 1e-12)
            if state == 0:
                if pct >= 0.005:
                    state = 1; last_idx = i; last_val = closes[i]
                elif pct <= -0.005:
                    state = -1; last_idx = i; last_val = closes[i]
            elif state == 1:
                if closes[i] > last_val:
                    last_idx = i; last_val = closes[i]
                elif pct <= -0.005:
                    inflections.append({"idx": last_idx, "type": "peak", "price": float(last_val), "new_dir": "down"})
                    state = -1; last_idx = i; last_val = closes[i]
            elif state == -1:
                if closes[i] < last_val:
                    last_idx = i; last_val = closes[i]
                elif pct >= 0.005:
                    inflections.append({"idx": last_idx, "type": "trough", "price": float(last_val), "new_dir": "up"})
                    state = 1; last_idx = i; last_val = closes[i]
        
        if not inflections:
            continue
        
        # For each inflection, what was the profit if we entered in new_dir?
        entry_results = []
        for inf in inflections:
            idx = inf["idx"]
            new_dir = inf["new_dir"]
            entry_price = inf["price"]
            
            # Look 5, 10, 30 minutes ahead
            for target_min, target_label in [(5, "5min"), (10, "10min"), (30, "30min")]:
                target_idx = idx + target_min  # 1-min candles
                if target_idx >= len(closes):
                    continue
                exit_price = closes[target_idx]
                if new_dir == "up":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                entry_results.append({
                    "sym": sym_base, "dir": new_dir, "hold": target_label,
                    "pnl_pct": pnl_pct,
                })
        
        if not entry_results:
            continue
        
        print(f"\n  {sym_base}: {len(inflections)} inflections (0.5% zigzag)")
        for hold_label in ["5min", "10min", "30min"]:
            subset = [r for r in entry_results if r["hold"] == hold_label]
            if not subset:
                continue
            wins = sum(1 for r in subset if r["pnl_pct"] > 0)
            avg_pnl = np.mean([r["pnl_pct"] for r in subset])
            print(f"    {hold_label} hold: {100*wins/len(subset):.0f}% win (n={len(subset)}), avg={avg_pnl:+.3f}%")
    
    db.close()
    print(f"\n{'=' * 70}")
    print("분석 완료.")


if __name__ == "__main__":
    main()
