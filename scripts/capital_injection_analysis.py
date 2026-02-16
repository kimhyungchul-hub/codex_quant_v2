#!/usr/bin/env python3
"""Capital injection before/after analysis + time-series performance breakdown"""
import sqlite3, json, statistics, os
from datetime import datetime
from collections import defaultdict

os.chdir("/Users/jeonghwakim/codex_quant_clean")
conn = sqlite3.connect("state/bot_data_live.db")
conn.row_factory = sqlite3.Row

injection_ts = 1770429459938  # 2026-02-07 15:57:39

before = conn.execute(
    "SELECT * FROM trades WHERE action != 'OPEN' AND timestamp_ms < ? ORDER BY timestamp_ms DESC LIMIT 360",
    (injection_ts,)
).fetchall()

after = conn.execute(
    "SELECT * FROM trades WHERE action != 'OPEN' AND timestamp_ms >= ? ORDER BY timestamp_ms ASC",
    (injection_ts,)
).fetchall()

def safe_json(raw):
    if not raw: return {}
    try: return json.loads(raw) if isinstance(raw, str) else raw
    except: return {}

def analyze_period(data, label):
    pnls = [r['realized_pnl'] for r in data if r['realized_pnl'] is not None]
    roes = [r['roe'] for r in data if r['roe'] is not None]
    notionals = [r['notional'] for r in data if r['notional'] is not None]
    holds = [r['hold_duration_sec'] for r in data if r['hold_duration_sec'] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    exit_reasons = defaultdict(lambda: {'n':0,'pnl':0.0,'wins':0})
    for r in data:
        raw = safe_json(r['raw_data'])
        reason = raw.get('exit_reason', r['action'] or 'UNKNOWN')
        exit_reasons[reason]['n'] += 1
        if r['realized_pnl'] is not None:
            exit_reasons[reason]['pnl'] += r['realized_pnl']
            if r['realized_pnl'] > 0:
                exit_reasons[reason]['wins'] += 1
    
    long_pnl = sum(r['realized_pnl'] for r in data if r['side']=='LONG' and r['realized_pnl'])
    short_pnl = sum(r['realized_pnl'] for r in data if r['side']=='SHORT' and r['realized_pnl'])
    long_n = sum(1 for r in data if r['side']=='LONG')
    short_n = sum(1 for r in data if r['side']=='SHORT')
    long_wins = sum(1 for r in data if r['side']=='LONG' and r['realized_pnl'] and r['realized_pnl'] > 0)
    short_wins = sum(1 for r in data if r['side']=='SHORT' and r['realized_pnl'] and r['realized_pnl'] > 0)
    long_total = sum(1 for r in data if r['side']=='LONG' and r['realized_pnl'] is not None)
    short_total = sum(1 for r in data if r['side']=='SHORT' and r['realized_pnl'] is not None)
    
    if len(data) > 1:
        ts_start = datetime.fromtimestamp(data[0]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
        ts_end = datetime.fromtimestamp(data[-1]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
        ts_range = f"{ts_start} ~ {ts_end}"
    else:
        ts_range = "N/A"
    
    print(f"\n{'='*80}")
    print(f" {label} ({len(data)}trades, {ts_range})")
    print(f"{'='*80}")
    
    if not pnls:
        print("  No PnL data")
        return
    
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else float('inf')
    print(f"  Total PnL:    ${sum(pnls):.2f}")
    print(f"  Win Rate:     {len(wins)}/{len(pnls)} = {len(wins)/len(pnls)*100:.1f}%")
    if roes:
        print(f"  Avg ROE:      {statistics.mean(roes)*100:.3f}%")
    print(f"  Profit Factor: {pf:.3f}")
    if notionals:
        print(f"  Avg Notional: ${statistics.mean(notionals):.2f}")
    if holds:
        print(f"  Avg Hold:     {statistics.mean(holds)/60:.1f}min")
    
    long_wr = f"{long_wins/long_total*100:.1f}%" if long_total > 0 else "N/A"
    short_wr = f"{short_wins/short_total*100:.1f}%" if short_total > 0 else "N/A"
    print(f"  LONG:  {long_n} trades, WR={long_wr}, PnL=${long_pnl:.2f}")
    print(f"  SHORT: {short_n} trades, WR={short_wr}, PnL=${short_pnl:.2f}")
    
    print(f"\n  --- Exit Reasons ---")
    for reason, s in sorted(exit_reasons.items(), key=lambda x: x[1]['pnl']):
        if s['n'] > 0 and reason not in ('ENTER', 'ORDER_REJECT', 'OPEN'):
            wr = s['wins']/s['n']*100 if s['n'] > 0 else 0
            print(f"    {reason:<35} {s['n']:>4} trades  WR:{wr:>5.1f}%  PnL:${s['pnl']:>8.2f}")

# Main analysis
print("=" * 80)
print(" CAPITAL INJECTION IMPACT ANALYSIS")
print("=" * 80)
print(f" Injection detected: {datetime.fromtimestamp(injection_ts/1000)}")
print(f" Before trades: {len(before)}")
print(f" After trades:  {len(after)}")

analyze_period(before, "BEFORE Injection (360 trades)")
analyze_period(after[:120], "AFTER Injection: First 120 trades")
analyze_period(after[:360], "AFTER Injection: First 360 trades")
if len(after) > 360:
    analyze_period(after[360:720], "AFTER Injection: 360~720 trades")
if len(after) > 720:
    analyze_period(after[720:1080], "AFTER Injection: 720~1080 trades")

# Time-series performance breakdown (360-trade windows)
print(f"\n{'='*80}")
print(f" TIME-SERIES PERFORMANCE (360-trade windows after injection)")
print(f"{'='*80}")
n_after = len(after)
print(f" Total after injection: {n_after} trades\n")
print(f" {'Window':<12} {'Period':<28} {'PnL':>10} {'WR':>7} {'PF':>7} {'ROE':>8} {'Notional':>10} {'Hold':>8}")
print(f" {'-'*12} {'-'*28} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")

for i in range(0, min(n_after, 9000), 360):
    chunk = after[i:i+360]
    if not chunk: break
    pnls = [r['realized_pnl'] for r in chunk if r['realized_pnl'] is not None]
    wins = [p for p in pnls if p > 0]
    roes = [r['roe'] for r in chunk if r['roe'] is not None]
    nots = [r['notional'] for r in chunk if r['notional'] is not None]
    holds = [r['hold_duration_sec'] for r in chunk if r['hold_duration_sec'] is not None]
    if not pnls: continue
    losses_v = [p for p in pnls if p <= 0]
    pf = abs(sum(wins)/sum(losses_v)) if losses_v and sum(losses_v) != 0 else float('inf')
    ts_s = datetime.fromtimestamp(chunk[0]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
    ts_e = datetime.fromtimestamp(chunk[-1]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
    avg_not = statistics.mean(nots) if nots else 0
    avg_hold = statistics.mean(holds)/60 if holds else 0
    wr = len(wins)/len(pnls)*100 if pnls else 0
    roe = statistics.mean(roes)*100 if roes else 0
    window = f"{i}-{i+len(chunk)}"
    period = f"{ts_s}~{ts_e}"
    print(f" {window:<12} {period:<28} ${sum(pnls):>8.2f} {wr:>6.1f}% {pf:>6.3f} {roe:>7.3f}% ${avg_not:>8.0f} {avg_hold:>6.1f}m")

# Notional distribution change
print(f"\n{'='*80}")
print(f" NOTIONAL SIZE DISTRIBUTION CHANGE")
print(f"{'='*80}")
for label, data in [("Before Injection", before), ("After 0-1000", after[:1000]), ("After 1000-3000", after[1000:3000]), ("After 6000+", after[6000:])]:
    if not data: continue
    nots = [r['notional'] for r in data if r['notional'] is not None]
    if not nots: continue
    print(f"\n  {label} ({len(data)} trades):")
    print(f"    Mean: ${statistics.mean(nots):.2f}")
    print(f"    Median: ${statistics.median(nots):.2f}")
    print(f"    Min: ${min(nots):.2f}, Max: ${max(nots):.2f}")
    # Buckets
    buckets = {'<$50': 0, '$50-100': 0, '$100-200': 0, '$200-500': 0, '$500+': 0}
    for n in nots:
        if n < 50: buckets['<$50'] += 1
        elif n < 100: buckets['$50-100'] += 1
        elif n < 200: buckets['$100-200'] += 1
        elif n < 500: buckets['$200-500'] += 1
        else: buckets['$500+'] += 1
    for k, v in buckets.items():
        pct = v/len(nots)*100
        print(f"    {k}: {v} ({pct:.1f}%)")

# Leverage distribution change
print(f"\n{'='*80}")
print(f" LEVERAGE DISTRIBUTION CHANGE")
print(f"{'='*80}")
for label, data in [("Before", before), ("After 0-2000", after[:2000]), ("After 6000+", after[6000:])]:
    if not data: continue
    levs = []
    for r in data:
        raw = safe_json(r['raw_data'])
        lev = raw.get('leverage') or raw.get('leverage_used')
        if lev is not None:
            levs.append(float(lev))
    if levs:
        print(f"  {label}: Mean={statistics.mean(levs):.2f}x, Median={statistics.median(levs):.2f}x, Max={max(levs):.0f}x ({len(levs)} samples)")

# regime-based performance change
print(f"\n{'='*80}")
print(f" REGIME-BASED PERFORMANCE")
print(f"{'='*80}")
for label, data in [("Before", before), ("After recent 360", after[-360:])]:
    regimes = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0, 'total_with_pnl': 0})
    for r in data:
        reg = r['regime'] or 'unknown'
        regimes[reg]['n'] += 1
        if r['realized_pnl'] is not None:
            regimes[reg]['pnl'] += r['realized_pnl']
            regimes[reg]['total_with_pnl'] += 1
            if r['realized_pnl'] > 0:
                regimes[reg]['wins'] += 1
    print(f"\n  {label}:")
    for reg, s in sorted(regimes.items(), key=lambda x: x[1]['pnl']):
        wr = s['wins']/s['total_with_pnl']*100 if s['total_with_pnl'] > 0 else 0
        print(f"    {reg:<12} {s['n']:>4} trades  WR:{wr:>5.1f}%  PnL:${s['pnl']:>8.2f}")

# Direction hit analysis
print(f"\n{'='*80}")
print(f" DIRECTION HIT RATE TREND")
print(f"{'='*80}")
for i in range(0, min(n_after, 3600), 720):
    chunk = after[i:i+720]
    if not chunk: break
    dir_hits = 0
    dir_total = 0
    for r in chunk:
        raw = safe_json(r['raw_data'])
        dh = raw.get('direction_hit')
        if dh is not None:
            dir_total += 1
            if dh: dir_hits += 1
    if dir_total > 0:
        ts_s = datetime.fromtimestamp(chunk[0]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
        ts_e = datetime.fromtimestamp(chunk[-1]['timestamp_ms']/1000).strftime('%m/%d %H:%M')
        print(f"  [{i}-{i+len(chunk)}] {ts_s}~{ts_e}: {dir_hits}/{dir_total} = {dir_hits/dir_total*100:.1f}%")

conn.close()
print("\n[DONE]")
