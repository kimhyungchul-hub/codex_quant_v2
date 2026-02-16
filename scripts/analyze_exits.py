#!/usr/bin/env python3
"""Analyze exit reasons and hold duration from trade history."""
import sqlite3, sys

DB = "state/bot_data_live.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# 1. Entry reason analysis
print("=== Entry Reason Analysis ===")
rows = conn.execute("""
    SELECT entry_reason, COUNT(*) as cnt, 
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(AVG(realized_pnl), 4) as avg_pnl,
           ROUND(SUM(realized_pnl), 2) as total_pnl,
           ROUND(AVG(hold_duration_sec), 0) as avg_hold
    FROM trades 
    WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
    GROUP BY entry_reason 
    ORDER BY total_pnl ASC
    LIMIT 25
""").fetchall()
print(f"{'Reason':<50} {'Cnt':>5} {'WR%':>6} {'AvgPnL':>9} {'Total':>10} {'Hold':>6}")
print('-'*92)
for r in rows:
    reason = str(r['entry_reason'] or 'none')[:49]
    wr = r['wins']/r['cnt']*100 if r['cnt']>0 else 0
    hold = r['avg_hold'] or 0
    print(f"{reason:<50} {r['cnt']:>5} {wr:>5.1f}% {r['avg_pnl']:>9.4f} {r['total_pnl']:>10.2f} {hold:>5.0f}s")

# 2. Hold duration distribution
print("\n=== Hold Duration Distribution ===")
rows2 = conn.execute("""
    SELECT 
        CASE 
            WHEN hold_duration_sec < 60 THEN 'A_<1min'
            WHEN hold_duration_sec < 300 THEN 'B_1-5min'
            WHEN hold_duration_sec < 600 THEN 'C_5-10min'
            WHEN hold_duration_sec < 1800 THEN 'D_10-30min'
            WHEN hold_duration_sec < 3600 THEN 'E_30-60min'
            ELSE 'F_>60min'
        END as bucket,
        COUNT(*) as cnt,
        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
        ROUND(AVG(realized_pnl), 4) as avg_pnl,
        ROUND(SUM(realized_pnl), 2) as total_pnl,
        ROUND(AVG(roe), 6) as avg_roe
    FROM trades 
    WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
    GROUP BY bucket
    ORDER BY bucket
""").fetchall()
for r in rows2:
    wr = r['wins']/r['cnt']*100 if r['cnt']>0 else 0
    label = r['bucket'][2:]
    print(f"  {label:<12} cnt={r['cnt']:>5} WR={wr:>5.1f}% avg_pnl={r['avg_pnl']:>8.4f} total={r['total_pnl']:>10.2f} avg_roe={r['avg_roe']:>8.4f}")

# 3. ROE distribution
print("\n=== ROE Distribution ===")
rows3 = conn.execute("""
    SELECT 
        CASE 
            WHEN roe < -0.05 THEN 'A_<-5pct'
            WHEN roe < -0.02 THEN 'B_-5to-2pct'
            WHEN roe < -0.01 THEN 'C_-2to-1pct'
            WHEN roe < 0 THEN 'D_-1to0pct'
            WHEN roe < 0.01 THEN 'E_0to1pct'
            WHEN roe < 0.02 THEN 'F_1to2pct'
            WHEN roe < 0.05 THEN 'G_2to5pct'
            ELSE 'H_>5pct'
        END as bucket,
        COUNT(*) as cnt,
        ROUND(AVG(realized_pnl), 4) as avg_pnl,
        ROUND(SUM(realized_pnl), 2) as total_pnl
    FROM trades 
    WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
    GROUP BY bucket
    ORDER BY bucket
""").fetchall()
for r in rows3:
    label = r['bucket'][2:]
    print(f"  {label:<14} cnt={r['cnt']:>5} avg_pnl={r['avg_pnl']:>8.4f} total={r['total_pnl']:>10.2f}")

# 4. Average winning PnL vs losing PnL (profit factor)
print("\n=== Win/Loss Ratio ===")
wl = conn.execute("""
    SELECT 
        SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
        SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss,
        AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
        AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as win_cnt,
        COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as loss_cnt
    FROM trades 
    WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
""").fetchone()
pf = wl['gross_profit'] / max(wl['gross_loss'], 0.01)
rr = abs(wl['avg_win'] / min(wl['avg_loss'], -0.01))
print(f"  Wins:  {wl['win_cnt']:>6} trades, avg=${wl['avg_win']:.4f}")
print(f"  Loss:  {wl['loss_cnt']:>6} trades, avg=${wl['avg_loss']:.4f}")
print(f"  Profit Factor: {pf:.4f}")
print(f"  Reward/Risk Ratio: {rr:.4f}")
print(f"  BEP WR at R:R={rr:.2f}: {1/(1+rr)*100:.1f}%")

# 5. Recent 24h trades
print("\n=== Recent 24h ===")
r24 = conn.execute("""
    SELECT COUNT(*) as cnt,
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(realized_pnl), 4) as total_pnl,
           ROUND(AVG(hold_duration_sec), 0) as avg_hold
    FROM trades 
    WHERE realized_pnl IS NOT NULL AND realized_pnl != 0
      AND timestamp_ms > (strftime('%s','now') * 1000 - 86400000)
""").fetchone()
if r24['cnt'] > 0:
    wr = r24['wins']/r24['cnt']*100
    print(f"  Trades: {r24['cnt']}, WR: {wr:.1f}%, PnL: ${r24['total_pnl']:.4f}, AvgHold: {r24['avg_hold']}s")
else:
    print("  No trades in last 24h")

conn.close()
