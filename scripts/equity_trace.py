#!/usr/bin/env python3
import sqlite3, os
from datetime import datetime

os.chdir("/Users/jeonghwakim/codex_quant_clean")
conn = sqlite3.connect("state/bot_data_live.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT timestamp_ms, total_equity, position_count, total_notional FROM equity_history ORDER BY timestamp_ms").fetchall()

print(f"Total equity records: {len(rows)}")
step = max(1, len(rows) // 50)
print(f"{'Date':<20} {'Equity':>10} {'Pos':>4} {'Notional':>12}")
print("-" * 50)
for i in range(0, len(rows), step):
    r = rows[i]
    dt = datetime.fromtimestamp(r['timestamp_ms']/1000).strftime('%m/%d %H:%M')
    eq = r['total_equity'] or 0
    pos = r['position_count'] or 0
    notional = r['total_notional'] or 0
    print(f"{dt:<20} ${eq:>9.2f} {pos:>4} ${notional:>11.2f}")

print(f"\n--- Last 10 records ---")
for r in rows[-10:]:
    dt = datetime.fromtimestamp(r['timestamp_ms']/1000).strftime('%m/%d %H:%M')
    eq = r['total_equity'] or 0
    pos = r['position_count'] or 0
    notional = r['total_notional'] or 0
    print(f"{dt:<20} ${eq:>9.2f} {pos:>4} ${notional:>11.2f}")

conn.close()
