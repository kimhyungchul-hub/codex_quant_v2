#!/usr/bin/env python3
"""
Counterfactual validation: Simulate the user's proposed strategy changes
against historical trades to estimate PnL impact.

Strategy: "Few symbols + High conviction + Long hold + High leverage + Concentrated capital"
"""
import sqlite3, json, sys, math
from datetime import datetime
from collections import defaultdict

DB = "state/bot_data_live.db"

# ── Current parameters (baseline) ──────────────────────────────────────
CURRENT = {
    "TOP_N_SYMBOLS": 20,
    "UNIFIED_ENTRY_FLOOR": -0.0003,
    "ENTRY_GROSS_EV_MIN": -0.00005,
    "ENTRY_NET_EXPECTANCY_MIN": -0.0008,
    "POSITION_HOLD_MIN_SEC": 60,
    "MAX_POSITION_HOLD_SEC": 1800,
    "MU_SIGN_FLIP_MIN_AGE_SEC": 120,
    "MU_SIGN_FLIP_CONFIRM_TICKS": 5,
    "LEVERAGE_TARGET_MAX": 6.5,
    "DEFAULT_TP_PCT": 0.015,
    "DEFAULT_SL_PCT": 0.012,
    "NOTIONAL_RECLAMP_BASE_FRAC": 0.16,
}

# ── Proposed parameters ────────────────────────────────────────────────
PROPOSED = {
    "TOP_N_SYMBOLS": 6,
    "UNIFIED_ENTRY_FLOOR": 0.001,
    "ENTRY_GROSS_EV_MIN": 0.0005,
    "ENTRY_NET_EXPECTANCY_MIN": 0.0,
    "POSITION_HOLD_MIN_SEC": 300,
    "MAX_POSITION_HOLD_SEC": 7200,
    "MU_SIGN_FLIP_MIN_AGE_SEC": 300,
    "MU_SIGN_FLIP_CONFIRM_TICKS": 8,
    "LEVERAGE_TARGET_MAX": 15.0,
    "DEFAULT_TP_PCT": 0.025,
    "DEFAULT_SL_PCT": 0.018,
    "NOTIONAL_RECLAMP_BASE_FRAC": 0.28,
}

# ── Top performing symbols (from golden analysis) ──────────────────────
# Best symbols: BERA($31), AAVE($7), POWER($6), BTR($6), HYPE($1), ZRO($1)
# Core high-cap always active: BTC, ETH
TOP_SYMBOLS = {"BERA", "AAVE", "POWER", "BTR", "HYPE", "ZRO", "BTC", "ETH"}

# Worst symbols to avoid
AVOID_SYMBOLS = {"ADA", "AVAX", "UNI", "FARTCOIN", "CLO", "SOL", "ME", "ASTER", "LINK", "SUI"}


def load_trades(db_path, limit=4000):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    trades = db.execute(
        "SELECT id, timestamp_ms, symbol, side, action, realized_pnl, "
        "hold_duration_sec, entry_ev, roe, regime, pred_mu_alpha, pred_mu_dir_conf, "
        "alpha_vpin, entry_quality_score, one_way_move_score, leverage_signal_score, "
        "raw_data "
        "FROM trades WHERE action='EXIT' AND realized_pnl IS NOT NULL "
        "ORDER BY timestamp_ms DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return list(reversed(trades))  # chronological


def parse_trade(t):
    rd = json.loads(t["raw_data"]) if t["raw_data"] else {}
    sym = (t["symbol"] or "").split("/")[0]
    return {
        "id": t["id"],
        "ts": t["timestamp_ms"],
        "symbol": sym,
        "side": t["side"],
        "pnl": float(t["realized_pnl"]) if t["realized_pnl"] else 0,
        "roe": float(t["roe"]) if t["roe"] else 0,
        "hold_sec": float(t["hold_duration_sec"]) if t["hold_duration_sec"] else 0,
        "entry_ev": float(t["entry_ev"]) if t["entry_ev"] else None,
        "regime": t["regime"] or rd.get("regime", ""),
        "mu": float(t["pred_mu_alpha"]) if t["pred_mu_alpha"] else None,
        "conf": float(t["pred_mu_dir_conf"]) if t["pred_mu_dir_conf"] else None,
        "vpin": float(t["alpha_vpin"]) if t["alpha_vpin"] else None,
        "leverage": float(rd.get("leverage") or rd.get("entry_leverage") or 0),
        "reason": rd.get("exit_reason", rd.get("reason", "")),
        "eq_score": float(t["entry_quality_score"]) if t["entry_quality_score"] else 0,
        "lev_score": float(t["leverage_signal_score"]) if t["leverage_signal_score"] else 0,
    }


def simulate_counterfactual(trades_raw):
    trades = [parse_trade(t) for t in trades_raw]
    print(f"Total trades loaded: {len(trades)}")
    
    results = {
        "baseline": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
        "cf_symbol_filter": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
        "cf_ev_filter": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
        "cf_combined": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
        "cf_avoid_worst": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
        "cf_hold_filter": {"pnl": 0, "cnt": 0, "wins": 0, "syms": set()},
    }
    
    # Buckets for detailed analysis
    hold_bucket_pnl = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0})
    lev_bucket_pnl = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0})
    
    for t in trades:
        pnl = t["pnl"]
        sym = t["symbol"]
        ev = t["entry_ev"]
        hold = t["hold_sec"]
        lev = t["leverage"]
        
        # Baseline (all trades)
        results["baseline"]["pnl"] += pnl
        results["baseline"]["cnt"] += 1
        if pnl > 0: results["baseline"]["wins"] += 1
        results["baseline"]["syms"].add(sym)
        
        # CF1: Only top symbols
        if sym in TOP_SYMBOLS:
            results["cf_symbol_filter"]["pnl"] += pnl
            results["cf_symbol_filter"]["cnt"] += 1
            if pnl > 0: results["cf_symbol_filter"]["wins"] += 1
            results["cf_symbol_filter"]["syms"].add(sym)
        
        # CF2: Only positive EV trades
        if ev is not None and ev > 0.001:
            results["cf_ev_filter"]["pnl"] += pnl
            results["cf_ev_filter"]["cnt"] += 1
            if pnl > 0: results["cf_ev_filter"]["wins"] += 1
            results["cf_ev_filter"]["syms"].add(sym)
        
        # CF3: Combined (top symbols + positive EV)
        if sym in TOP_SYMBOLS and (ev is not None and ev > 0.0005):
            results["cf_combined"]["pnl"] += pnl
            results["cf_combined"]["cnt"] += 1
            if pnl > 0: results["cf_combined"]["wins"] += 1
            results["cf_combined"]["syms"].add(sym)
        
        # CF4: Avoid worst symbols
        if sym not in AVOID_SYMBOLS:
            results["cf_avoid_worst"]["pnl"] += pnl
            results["cf_avoid_worst"]["cnt"] += 1
            if pnl > 0: results["cf_avoid_worst"]["wins"] += 1
            results["cf_avoid_worst"]["syms"].add(sym)
        
        # CF5: Only long-hold trades (>300s)
        if hold > 300:
            results["cf_hold_filter"]["pnl"] += pnl
            results["cf_hold_filter"]["cnt"] += 1
            if pnl > 0: results["cf_hold_filter"]["wins"] += 1
            results["cf_hold_filter"]["syms"].add(sym)
        
        # Hold duration buckets
        if hold < 60: bkt = "<1m"
        elif hold < 300: bkt = "1-5m"
        elif hold < 900: bkt = "5-15m"
        elif hold < 3600: bkt = "15m-1h"
        else: bkt = ">1h"
        hold_bucket_pnl[bkt]["pnl"] += pnl
        hold_bucket_pnl[bkt]["cnt"] += 1
        if pnl > 0: hold_bucket_pnl[bkt]["wins"] += 1
        
        # Leverage buckets
        if lev <= 0: bkt = "unknown"
        elif lev < 3: bkt = "<3x"
        elif lev < 8: bkt = "3-8x"
        elif lev < 15: bkt = "8-15x"
        else: bkt = ">=15x"
        lev_bucket_pnl[bkt]["pnl"] += pnl
        lev_bucket_pnl[bkt]["cnt"] += 1
        if pnl > 0: lev_bucket_pnl[bkt]["wins"] += 1
    
    # Print results
    print("\n" + "="*80)
    print("COUNTERFACTUAL COMPARISON")
    print("="*80)
    
    for name, d in results.items():
        cnt = d["cnt"]
        if cnt == 0:
            print(f"\n  {name:25s}: NO TRADES")
            continue
        wr = d["wins"] / cnt * 100
        avg = d["pnl"] / cnt
        print(f"\n  {name:25s}: PnL=${d['pnl']:+8.2f}  cnt={cnt:5d}  "
              f"WR={wr:5.1f}%  avg=${avg:+.4f}  syms={len(d['syms'])}")
    
    # Calculate deltas
    base_pnl = results["baseline"]["pnl"]
    print(f"\n{'='*80}")
    print("DELTA vs BASELINE")
    print(f"{'='*80}")
    for name, d in results.items():
        if name == "baseline":
            continue
        delta = d["pnl"] - base_pnl
        saved_trades = results["baseline"]["cnt"] - d["cnt"]
        print(f"  {name:25s}: delta=${delta:+8.2f}  trades_saved={saved_trades:+d}")
    
    # Hold duration analysis
    print(f"\n{'='*80}")
    print("HOLD DURATION vs PNL")
    print(f"{'='*80}")
    for bkt in ["<1m", "1-5m", "5-15m", "15m-1h", ">1h"]:
        d = hold_bucket_pnl[bkt]
        if d["cnt"] == 0:
            continue
        wr = d["wins"] / d["cnt"] * 100
        avg = d["pnl"] / d["cnt"]
        print(f"  {bkt:10s}: PnL=${d['pnl']:+8.2f}  cnt={d['cnt']:5d}  WR={wr:5.1f}%  avg=${avg:+.4f}")
    
    # Leverage analysis
    print(f"\n{'='*80}")
    print("LEVERAGE vs PNL")
    print(f"{'='*80}")
    for bkt in ["unknown", "<3x", "3-8x", "8-15x", ">=15x"]:
        d = lev_bucket_pnl[bkt]
        if d["cnt"] == 0:
            continue
        wr = d["wins"] / d["cnt"] * 100
        avg = d["pnl"] / d["cnt"]
        print(f"  {bkt:10s}: PnL=${d['pnl']:+8.2f}  cnt={d['cnt']:5d}  WR={wr:5.1f}%  avg=${avg:+.4f}")
    
    # Regime analysis
    regime_stats = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0})
    for t in trades:
        r = t["regime"] or "unknown"
        regime_stats[r]["pnl"] += t["pnl"]
        regime_stats[r]["cnt"] += 1
        if t["pnl"] > 0: regime_stats[r]["wins"] += 1
    
    print(f"\n{'='*80}")
    print("REGIME vs PNL")
    print(f"{'='*80}")
    for r, d in sorted(regime_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        if d["cnt"] == 0:
            continue
        wr = d["wins"] / d["cnt"] * 100
        avg = d["pnl"] / d["cnt"]
        print(f"  {r:12s}: PnL=${d['pnl']:+8.2f}  cnt={d['cnt']:5d}  WR={wr:5.1f}%  avg=${avg:+.4f}")
    
    # Combined optimal scenario: top symbols + avoid chop + positive EV + hold>300s
    optimal = {"pnl": 0, "cnt": 0, "wins": 0}
    for t in trades:
        sym = t["symbol"]
        ev = t["entry_ev"]
        hold = t["hold_sec"]
        regime = t["regime"]
        
        # Optimal: Top symbol or not in avoid list, EV > 0, not micro-hold
        in_top = sym in TOP_SYMBOLS
        not_worst = sym not in AVOID_SYMBOLS
        has_ev = ev is not None and ev > 0.0005
        decent_hold = hold > 120
        not_chop = regime != "chop"
        
        if not_worst and has_ev and decent_hold:
            optimal["pnl"] += t["pnl"]
            optimal["cnt"] += 1
            if t["pnl"] > 0: optimal["wins"] += 1
    
    if optimal["cnt"] > 0:
        wr = optimal["wins"] / optimal["cnt"] * 100
        avg = optimal["pnl"] / optimal["cnt"]
        delta = optimal["pnl"] - base_pnl
        print(f"\n{'='*80}")
        print(f"OPTIMAL SCENARIO (no-worst + EV>0.0005 + hold>120s)")
        print(f"  PnL=${optimal['pnl']:+8.2f}  cnt={optimal['cnt']:5d}  WR={wr:5.1f}%  avg=${avg:+.4f}")
        print(f"  vs baseline: delta=${delta:+8.2f}  trades_saved={results['baseline']['cnt'] - optimal['cnt']:+d}")
    
    # Aggressive optimal: top symbols + EV>0.001 + hold>300s + scale leverage
    scale_optimal = {"pnl": 0, "cnt": 0, "wins": 0}
    for t in trades:
        sym = t["symbol"]
        ev = t["entry_ev"]
        hold = t["hold_sec"]
        lev = t["leverage"]
        
        in_top = sym in TOP_SYMBOLS
        not_worst = sym not in AVOID_SYMBOLS
        has_ev = ev is not None and ev > 0.001
        decent_hold = hold > 300
        
        if in_top and has_ev:
            # Scale: if we had used higher leverage on these specific trades
            # Assume average leverage would increase from ~5x to ~12x
            pnl_scaled = t["pnl"]
            if lev > 0 and lev < 12:
                scale = min(12.0 / max(lev, 1.0), 3.0)  # Cap at 3x scaling
                pnl_scaled = t["pnl"] * scale
            
            scale_optimal["pnl"] += pnl_scaled
            scale_optimal["cnt"] += 1
            if pnl_scaled > 0: scale_optimal["wins"] += 1
    
    if scale_optimal["cnt"] > 0:
        wr = scale_optimal["wins"] / scale_optimal["cnt"] * 100
        avg = scale_optimal["pnl"] / scale_optimal["cnt"]
        print(f"\n{'='*80}")
        print(f"AGGRESSIVE SCENARIO (top_syms + EV>0.001 + leverage scaled to 12x)")
        print(f"  PnL=${scale_optimal['pnl']:+8.2f}  cnt={scale_optimal['cnt']:5d}  WR={wr:5.1f}%  avg=${avg:+.4f}")
        print(f"  vs baseline: delta=${scale_optimal['pnl'] - base_pnl:+8.2f}")
    
    # === Summary recommendation ===
    print(f"\n{'='*80}")
    print("RECOMMENDATION SUMMARY")
    print(f"{'='*80}")
    
    best_scenario = max(
        [(n, d) for n, d in results.items() if n != "baseline" and d["cnt"] > 0],
        key=lambda x: x[1]["pnl"]
    )
    print(f"  Best standalone filter: {best_scenario[0]} (PnL improvement: ${best_scenario[1]['pnl'] - base_pnl:+.2f})")
    
    if optimal["cnt"] > 0:
        print(f"  Best combined (no-worst+EV+hold): ${optimal['pnl']:+.2f} ({optimal['cnt']} trades)")
    if scale_optimal["cnt"] > 0:
        print(f"  Best aggressive (top+EV+leverage): ${scale_optimal['pnl']:+.2f} ({scale_optimal['cnt']} trades)")
    
    # Print confidence level
    if optimal["cnt"] > 100:
        confidence = "HIGH"
    elif optimal["cnt"] > 30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW (small sample)"
    print(f"  Confidence: {confidence} (based on {optimal['cnt']} trades)")
    
    return results, optimal, scale_optimal


if __name__ == "__main__":
    trades = load_trades(DB, limit=4000)
    results, optimal, aggressive = simulate_counterfactual(trades)
