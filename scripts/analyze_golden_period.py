#!/usr/bin/env python3
"""Analyze the golden trading period ($350->$380 equity surge) and extract winning patterns."""
import sqlite3, json, time, sys
from datetime import datetime
from collections import defaultdict

DB = "state/bot_data_live.db"
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

# 1. Equity history
rows = db.execute("SELECT timestamp_ms, total_equity FROM equity_history ORDER BY timestamp_ms").fetchall()
print(f"EQUITY_COUNT: {len(rows)}")
eqs = [(r['timestamp_ms'], float(r['total_equity'])) for r in rows if r['total_equity']]
print("=== EQUITY 340-390 RANGE ===")
for ts, eq in eqs:
    dt = datetime.fromtimestamp(ts/1000).strftime('%m-%d %H:%M')
    if 340 <= eq <= 390:
        print(f"  EQ {dt} = {eq:.2f}")

# 2. All exits
trades = db.execute(
    "SELECT id, timestamp_ms, symbol, side, realized_pnl, hold_duration_sec, raw_data "
    "FROM trades WHERE action='EXIT' AND realized_pnl IS NOT NULL ORDER BY timestamp_ms"
).fetchall()
print(f"\nTOTAL_EXITS: {len(trades)}")

# 3. Hourly grouping
hourly = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0, "syms": set(), "hold_avg": 0, "holds": []})
for t in trades:
    dt = datetime.fromtimestamp(t['timestamp_ms']/1000)
    hk = dt.strftime('%m-%d %H')
    p = float(t['realized_pnl'])
    h = float(t['hold_duration_sec']) if t['hold_duration_sec'] else 0
    hourly[hk]["pnl"] += p
    hourly[hk]["cnt"] += 1
    if p > 0:
        hourly[hk]["wins"] += 1
    hourly[hk]["syms"].add(t['symbol'].split('/')[0])
    hourly[hk]["holds"].append(h)

top = sorted(hourly.items(), key=lambda x: x[1]["pnl"], reverse=True)[:15]
print("\n=== TOP 15 BEST HOURS ===")
for h, d in top:
    wr = d["wins"]/d["cnt"]*100
    avg_hold = sum(d["holds"])/len(d["holds"]) if d["holds"] else 0
    syms = sorted(d["syms"])[:5]
    print(f"  {h}: pnl={d['pnl']:.3f} cnt={d['cnt']} wr={wr:.0f}% hold_avg={avg_hold:.0f}s syms={len(d['syms'])} {syms}")

worst = sorted(hourly.items(), key=lambda x: x[1]["pnl"])[:10]
print("\n=== WORST 10 HOURS ===")
for h, d in worst:
    wr = d["wins"]/d["cnt"]*100
    avg_hold = sum(d["holds"])/len(d["holds"]) if d["holds"] else 0
    print(f"  {h}: pnl={d['pnl']:.3f} cnt={d['cnt']} wr={wr:.0f}% hold_avg={avg_hold:.0f}s")

# 4. Deep-dive into winning trades from the golden period
# Find the best 4-hour window
windows = []
hourly_list = sorted(hourly.items())
for i in range(len(hourly_list) - 3):
    w_pnl = sum(hourly_list[j][1]["pnl"] for j in range(i, i+4))
    w_cnt = sum(hourly_list[j][1]["cnt"] for j in range(i, i+4))
    w_wins = sum(hourly_list[j][1]["wins"] for j in range(i, i+4))
    windows.append((hourly_list[i][0], hourly_list[i+3][0], w_pnl, w_cnt, w_wins))

windows.sort(key=lambda x: x[2], reverse=True)
print("\n=== TOP 5 BEST 4-HOUR WINDOWS ===")
for start, end, pnl, cnt, wins in windows[:5]:
    wr = wins/cnt*100 if cnt else 0
    print(f"  {start} to {end}: pnl={pnl:.3f} cnt={cnt} wr={wr:.0f}%")

# 5. Analyze the golden window trades in detail
if windows:
    best_start, best_end = windows[0][0], windows[0][1]
    print(f"\n=== GOLDEN WINDOW DETAIL: {best_start} to {best_end} ===")
    golden_trades = []
    for t in trades:
        dt = datetime.fromtimestamp(t['timestamp_ms']/1000)
        hk = dt.strftime('%m-%d %H')
        if best_start <= hk <= best_end:
            p = float(t['realized_pnl'])
            h = float(t['hold_duration_sec']) if t['hold_duration_sec'] else 0
            rd = json.loads(t['raw_data']) if t['raw_data'] else {}
            golden_trades.append({
                "sym": t['symbol'], "side": t['side'], "pnl": p,
                "hold_sec": h, "reason": rd.get('exit_reason', rd.get('reason', '')),
                "regime": rd.get('regime', ''), "ev": rd.get('entry_ev'),
                "mu": rd.get('pred_mu_alpha'), "conf": rd.get('pred_mu_dir_conf'),
                "lev": rd.get('leverage', rd.get('entry_leverage')),
            })
    
    # Aggregate golden stats
    wins = [t for t in golden_trades if t['pnl'] > 0]
    losses = [t for t in golden_trades if t['pnl'] <= 0]
    print(f"  Total: {len(golden_trades)}, Wins: {len(wins)}, Losses: {len(losses)}")
    print(f"  Win Rate: {len(wins)/len(golden_trades)*100:.1f}%")
    print(f"  Avg PnL: ${sum(t['pnl'] for t in golden_trades)/len(golden_trades):.4f}")
    print(f"  Avg Hold(win): {sum(t['hold_sec'] for t in wins)/max(1,len(wins)):.0f}s")
    print(f"  Avg Hold(loss): {sum(t['hold_sec'] for t in losses)/max(1,len(losses)):.0f}s")
    
    sym_cnt = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0})
    for t in golden_trades:
        sym = t['sym'].split('/')[0]
        sym_cnt[sym]["pnl"] += t['pnl']
        sym_cnt[sym]["cnt"] += 1
        if t['pnl'] > 0: sym_cnt[sym]["wins"] += 1
    
    print("\n  Symbol breakdown:")
    for sym, d in sorted(sym_cnt.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = d["wins"]/d["cnt"]*100
        print(f"    {sym}: pnl=${d['pnl']:.3f} cnt={d['cnt']} wr={wr:.0f}%")
    
    # Regime breakdown
    regime_cnt = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0})
    for t in golden_trades:
        r = t.get('regime', 'unknown')
        regime_cnt[r]["pnl"] += t['pnl']
        regime_cnt[r]["cnt"] += 1
        if t['pnl'] > 0: regime_cnt[r]["wins"] += 1
    
    print("\n  Regime breakdown:")
    for r, d in sorted(regime_cnt.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = d["wins"]/d["cnt"]*100
        print(f"    {r}: pnl=${d['pnl']:.3f} cnt={d['cnt']} wr={wr:.0f}%")
    
    # Hold duration buckets for winners
    print("\n  Hold duration buckets (winners only):")
    hold_bkts = {"<30s": 0, "30-120s": 0, "120-300s": 0, "300-600s": 0, ">600s": 0}
    hold_bkt_pnl = {"<30s": 0, "30-120s": 0, "120-300s": 0, "300-600s": 0, ">600s": 0}
    for t in wins:
        h = t['hold_sec']
        if h < 30: bkt = "<30s"
        elif h < 120: bkt = "30-120s"
        elif h < 300: bkt = "120-300s"
        elif h < 600: bkt = "300-600s"
        else: bkt = ">600s"
        hold_bkts[bkt] += 1
        hold_bkt_pnl[bkt] += t['pnl']
    
    for bkt in ["<30s", "30-120s", "120-300s", "300-600s", ">600s"]:
        print(f"    {bkt}: cnt={hold_bkts[bkt]} pnl=${hold_bkt_pnl[bkt]:.3f}")

    # Leverage analysis
    lev_sum = defaultdict(lambda: {"pnl": 0, "cnt": 0})
    for t in golden_trades:
        lev = t.get('lev')
        if lev is not None:
            try:
                l = float(lev)
                if l < 2: bkt = "<2x"
                elif l < 5: bkt = "2-5x"
                elif l < 10: bkt = "5-10x"
                else: bkt = ">=10x"
            except: bkt = "unknown"
        else: bkt = "unknown"
        lev_sum[bkt]["pnl"] += t['pnl']
        lev_sum[bkt]["cnt"] += 1
    
    print("\n  Leverage breakdown:")
    for bkt, d in sorted(lev_sum.items()):
        print(f"    {bkt}: pnl=${d['pnl']:.3f} cnt={d['cnt']}")

# 6. Compare winning vs losing trade characteristics across ALL trades
print("\n=== GLOBAL WIN vs LOSS CHARACTERISTICS ===")
all_wins = [t for t in trades if float(t['realized_pnl']) > 0]
all_losses = [t for t in trades if float(t['realized_pnl']) <= 0]
print(f"  Total wins: {len(all_wins)}, Total losses: {len(all_losses)}")
print(f"  Win rate: {len(all_wins)/len(trades)*100:.1f}%")

# Hold duration comparison
avg_hold_w = sum(float(t['hold_duration_sec'] or 0) for t in all_wins) / max(1, len(all_wins))
avg_hold_l = sum(float(t['hold_duration_sec'] or 0) for t in all_losses) / max(1, len(all_losses))
print(f"  Avg hold (win): {avg_hold_w:.0f}s")
print(f"  Avg hold (loss): {avg_hold_l:.0f}s")

# Analyze raw_data for leverage and mu patterns in wins vs losses
win_levs, loss_levs = [], []
win_mus, loss_mus = [], []
win_confs, loss_confs = [], []
for t in all_wins[-2000:]:
    rd = json.loads(t['raw_data']) if t['raw_data'] else {}
    lev = rd.get('leverage') or rd.get('entry_leverage')
    mu = rd.get('pred_mu_alpha')
    conf = rd.get('pred_mu_dir_conf')
    if lev: win_levs.append(float(lev))
    if mu: win_mus.append(float(mu))
    if conf: win_confs.append(float(conf))

for t in all_losses[-2000:]:
    rd = json.loads(t['raw_data']) if t['raw_data'] else {}
    lev = rd.get('leverage') or rd.get('entry_leverage')
    mu = rd.get('pred_mu_alpha')
    conf = rd.get('pred_mu_dir_conf')
    if lev: loss_levs.append(float(lev))
    if mu: loss_mus.append(float(mu))
    if conf: loss_confs.append(float(conf))

if win_levs: print(f"  Avg leverage (win): {sum(win_levs)/len(win_levs):.2f}x")
if loss_levs: print(f"  Avg leverage (loss): {sum(loss_levs)/len(loss_levs):.2f}x")
if win_mus: print(f"  Avg |mu| (win): {sum(abs(m) for m in win_mus)/len(win_mus):.3f}")
if loss_mus: print(f"  Avg |mu| (loss): {sum(abs(m) for m in loss_mus)/len(loss_mus):.3f}")
if win_confs: print(f"  Avg conf (win): {sum(win_confs)/len(win_confs):.4f}")
if loss_confs: print(f"  Avg conf (loss): {sum(loss_confs)/len(loss_confs):.4f}")

# Top winning symbols
sym_total = defaultdict(lambda: {"pnl": 0, "cnt": 0, "wins": 0, "avg_hold_w": [], "avg_hold_l": []})
for t in trades[-4000:]:
    sym = t['symbol'].split('/')[0]
    p = float(t['realized_pnl'])
    h = float(t['hold_duration_sec'] or 0)
    sym_total[sym]["pnl"] += p
    sym_total[sym]["cnt"] += 1
    if p > 0:
        sym_total[sym]["wins"] += 1
        sym_total[sym]["avg_hold_w"].append(h)
    else:
        sym_total[sym]["avg_hold_l"].append(h)

print("\n=== SYMBOL RANKINGS (recent 4000 exits) ===")
for sym, d in sorted(sym_total.items(), key=lambda x: x[1]["pnl"], reverse=True)[:15]:
    wr = d["wins"]/d["cnt"]*100
    ahw = sum(d["avg_hold_w"])/max(1,len(d["avg_hold_w"]))
    ahl = sum(d["avg_hold_l"])/max(1,len(d["avg_hold_l"]))
    print(f"  {sym:10s}: pnl=${d['pnl']:+.3f} cnt={d['cnt']:4d} wr={wr:.0f}% hold_w={ahw:.0f}s hold_l={ahl:.0f}s")

print("\n... WORST 10 ...")
for sym, d in sorted(sym_total.items(), key=lambda x: x[1]["pnl"])[:10]:
    wr = d["wins"]/d["cnt"]*100
    print(f"  {sym:10s}: pnl=${d['pnl']:+.3f} cnt={d['cnt']:4d} wr={wr:.0f}%")

db.close()
