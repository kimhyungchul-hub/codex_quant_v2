#!/usr/bin/env python3
"""Test mu_alpha magnitude thresholds for profitable signal strength."""
import sqlite3

DB = "state/bot_data_live.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
trades = conn.execute("""
    SELECT side, pred_mu_alpha, realized_pnl, hold_duration_sec, regime
    FROM trades WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
      AND pred_mu_alpha IS NOT NULL AND pred_mu_alpha != 0
""").fetchall()

print("=== Aligned trades by mu_alpha magnitude ===")
for threshold in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    aligned = [t for t in trades if (
        (t['pred_mu_alpha'] > threshold and t['side'] == 'LONG') or
        (t['pred_mu_alpha'] < -threshold and t['side'] == 'SHORT')
    )]
    if not aligned:
        print(f"|mu|>{threshold:.2f}: 0 trades")
        continue
    wins = sum(1 for t in aligned if t['realized_pnl'] > 0)
    pnl = sum(t['realized_pnl'] for t in aligned)
    wr = wins / len(aligned) * 100
    avg_w = sum(t['realized_pnl'] for t in aligned if t['realized_pnl'] > 0) / max(wins, 1)
    losses = [t for t in aligned if t['realized_pnl'] < 0]
    avg_l = sum(t['realized_pnl'] for t in losses) / max(len(losses), 1) if losses else -0.01
    rr = abs(avg_w / min(avg_l, -0.01))
    bep = 1 / (1 + rr) * 100
    edge = wr - bep
    sign = "+" if edge > 0 else ""
    profit = "✅" if edge > 0 else "❌"
    print(f"|mu|>{threshold:.2f}: {len(aligned):>5} trades, WR={wr:>5.1f}%, "
          f"PnL=${pnl:>9.2f}, R:R={rr:.2f}, BEP={bep:.1f}%, edge={sign}{edge:.1f}pp {profit}")

# Also test using ONLY DirectionModel approach: strong mu + same direction
print("\n=== Strong signal + aligned: PnL by hold duration ===")
strong_aligned = [t for t in trades if (
    (t['pred_mu_alpha'] > 0.10 and t['side'] == 'LONG') or
    (t['pred_mu_alpha'] < -0.10 and t['side'] == 'SHORT')
)]
for bucket, lo, hi in [('<1min', 0, 60), ('1-5min', 60, 300), ('5-10min', 300, 600),
                        ('10-30min', 600, 1800), ('>30min', 1800, 999999)]:
    bt = [t for t in strong_aligned if lo <= (t['hold_duration_sec'] or 0) < hi]
    if bt:
        wins = sum(1 for t in bt if t['realized_pnl'] > 0)
        pnl = sum(t['realized_pnl'] for t in bt)
        wr = wins / len(bt) * 100
        print(f"  {bucket:<10}: {len(bt):>4} trades, WR={wr:.1f}%, PnL=${pnl:.2f}")

conn.close()
