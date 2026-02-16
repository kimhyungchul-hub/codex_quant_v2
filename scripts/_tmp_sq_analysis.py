#!/usr/bin/env python3
"""Analyze SQ filter stats, capital utilization, and leverage."""
import sqlite3, os, math

DB = os.path.join(os.path.dirname(__file__), "..", "state", "bot_data_live.db")

def main():
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row

    # 1. Balance
    r = db.execute("SELECT total_equity, wallet_balance FROM equity_history ORDER BY timestamp_ms DESC LIMIT 1").fetchone()
    bal = float(r["total_equity"] or r["wallet_balance"] or 0) if r else 0
    print(f"=== Balance: ${bal:.2f} ===\n")

    # 2. SQ stats per symbol
    print("=" * 70)
    print("1. Symbol Quality (최근 320 EXIT 기준)")
    print("=" * 70)
    syms = ["SOL/USDT:USDT","XRP/USDT:USDT","DOGE/USDT:USDT","ADA/USDT:USDT",
            "SUI/USDT:USDT","LINK/USDT:USDT","BCH/USDT:USDT","BTC/USDT:USDT",
            "ETH/USDT:USDT","BERA/USDT:USDT","ME/USDT:USDT","0G/USDT:USDT",
            "BTR/USDT:USDT","PIPPIN/USDT:USDT","ZRO/USDT:USDT","MOVE/USDT:USDT",
            "XPL/USDT:USDT","MOODENG/USDT:USDT"]
    exp_ref = 0.0010
    reject_ref = 0.35
    min_score_th = 0.34  # from env

    for s in sorted(syms):
        rows = db.execute(
            "SELECT roe, timestamp_ms FROM trades WHERE symbol=? AND action IN ('EXIT','REBAL_EXIT') AND roe IS NOT NULL ORDER BY timestamp_ms DESC LIMIT 320",
            (s,),
        ).fetchall()
        rejects = db.execute(
            "SELECT COUNT(*) as n FROM trades WHERE symbol=? AND action='ORDER_REJECT' ORDER BY timestamp_ms DESC LIMIT 320",
            (s,),
        ).fetchone()
        reject_n = int(rejects["n"]) if rejects else 0

        if not rows:
            print(f"  {s:<25s}: no trades")
            continue

        roes = [float(r["roe"]) for r in rows]
        n = len(roes)
        hits = sum(1 for r in roes if r > 0)
        exp = sum(roes) / n
        hit_rate = hits / n
        total = sum(roes)
        denom = max(1, n + reject_n)
        reject_ratio = reject_n / denom

        # Compute SQ score (simplified version matching engine logic)
        exp_score = 0.5 + 0.5 * math.tanh(exp / max(1e-6, exp_ref))
        hit_score = max(0.0, min(1.0, hit_rate))
        reject_score = max(0.0, min(1.0, 1.0 - (reject_ratio / max(1e-6, reject_ref))))
        w_exp, w_hit, w_rej = 0.55, 0.30, 0.15
        score = w_exp * exp_score + w_hit * hit_score + w_rej * reject_score

        # Penalties
        if exp < -0.0015:
            score -= 0.22
        if reject_ratio > 0.65:
            score -= 0.12

        blocked = "BLOCKED" if (n >= 24 and score < min_score_th) else ("PASS" if n >= 24 else "SKIP(n<24)")
        print(f"  {s:<25s}: n={n:>3}, hit={100*hit_rate:.1f}%, exp={exp:+.6f}, rej={reject_ratio:.2f}, "
              f"score={score:.3f} {'<' if score < min_score_th else '>='} {min_score_th} → {blocked}")

    # 3. Capital utilization
    print(f"\n{'=' * 70}")
    print("2. Capital Utilization & Leverage (최근 100 거래)")
    print("=" * 70)

    recent = db.execute(
        "SELECT symbol, side, notional, timestamp_ms FROM trades "
        "WHERE action='ENTER' AND timestamp_ms IS NOT NULL ORDER BY timestamp_ms DESC LIMIT 100"
    ).fetchall()

    if recent:
        notionals = [float(r["notional"] or 0) for r in recent if r["notional"]]
        avg_notional = sum(notionals) / len(notionals) if notionals else 0
        max_notional = max(notionals) if notionals else 0
        print(f"  평균 notional: ${avg_notional:.2f} (max: ${max_notional:.2f})")
        if bal > 0:
            print(f"  평균 자본이용률: {100*avg_notional/bal:.1f}% of balance")
            print(f"  최대 자본이용률: {100*max_notional/bal:.1f}% of balance")

    # 4. Capital tier (for current balance)
    tiers = [500.0, 1500.0, 3000.0, 6000.0, 9000.0]
    caps = [2.0, 2.8, 4.0, 5.5, 7.5, 9.5]
    idx = 0
    for th in tiers:
        if bal < th:
            break
        idx += 1
    tier_cap = caps[min(idx, len(caps) - 1)]
    print(f"\n  현재 잔고 기준 CAPITAL_TIER: balance=${bal:.2f} → cap={tier_cap:.1f}x (tier_idx={idx})")

    min_notional_tiers = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    min_not = min_notional_tiers[min(idx, len(min_notional_tiers) - 1)]
    print(f"  CAPITAL_TIER min_notional: ${min_not:.2f}")

    # 5. Active positions (from live engine)
    print(f"\n{'=' * 70}")
    print("3. 현재 Active Positions & Total Exposure")
    print("=" * 70)
    pos_rows = db.execute(
        "SELECT symbol, side, leverage, notional, entry_price, roe FROM positions WHERE side IS NOT NULL"
    ).fetchall()
    total_notional = 0
    for p in pos_rows:
        not_v = float(p["notional"] or 0)
        total_notional += not_v
        lev_v = float(p["leverage"] or 1)
        roe_v = p["roe"] if p["roe"] is not None else "N/A"
        print(f"  {p['symbol']:<25s} {p['side']:<6s} lev={lev_v:.1f}x notional=${not_v:.2f} roe={roe_v}")
    if pos_rows:
        if bal > 0:
            print(f"  총 노출: ${total_notional:.2f} ({100*total_notional/bal:.1f}% of balance)")
    else:
        print("  포지션 없음")

    # 6. Recent LEV_DIAG from engine log
    print(f"\n{'=' * 70}")
    print("4. 최근 LEV_DIAG (레버리지 진단)")
    print("=" * 70)
    try:
        import subprocess
        out = subprocess.check_output(
            ["grep", "LEV_DIAG", "/tmp/engine.log"],
            text=True, timeout=5
        ).strip().split("\n")
        for line in out[-10:]:
            print(f"  {line.strip()}")
    except Exception as e:
        print(f"  로그 접근 실패: {e}")

    db.close()

if __name__ == "__main__":
    main()
