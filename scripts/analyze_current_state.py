#!/usr/bin/env python3
"""현재 엔진 상태 분석: 레버리지/포지션 사이즈/노출도"""
import re, sys

lines = open('/tmp/engine.log','r',errors='ignore').readlines()

sizing = [l for l in lines if 'SIZING_INPUT' in l]
entries = [l for l in lines if '] ENTER ' in l]
exits = [l for l in lines if '] EXIT ' in l]
filters_pass = [l for l in lines if 'FILTER' in l and 'all_pass' in l]
filters_block = [l for l in lines if 'FILTER' in l and 'blocked' in l]
lev_diag = [l for l in lines if 'LEV_DIAG' in l]
gate_rej = [l for l in lines if 'gate_reject' in l]
skip_min = [l for l in lines if 'ENTRY_SKIP_MIN' in l]

print(f"Total lines: {len(lines)}")
print(f"SIZING_INPUT: {len(sizing)}")
print(f"ENTRIES: {len(entries)}")
print(f"EXITS: {len(exits)}")
print(f"FILTER all_pass: {len(filters_pass)}")
print(f"FILTER blocked: {len(filters_block)}")
print(f"LEV_DIAG: {len(lev_diag)}")
print(f"gate_reject: {len(gate_rej)}")
print(f"SKIP_MIN_*: {len(skip_min)}")
print()

# Entry analysis
print("=== RECENT ENTRIES ===")
for e in entries[-15:]:
    print(e.strip()[:140])
print()

# LEV_DIAG
print("=== LEV_DIAG SAMPLES ===")
for l in lev_diag[-10:]:
    print(l.strip()[:160])
print()

# SIZING_INPUT parsing
print("=== SIZING_INPUT STATS ===")
levs = []
evs = []
kellys = []
confs = []
for s in sizing:
    m_lev = re.search(r'lev_in=([0-9.]+)', s)
    m_ev = re.search(r'ev=([0-9.\-]+)', s)
    m_kelly = re.search(r'kelly=([0-9.\-]+)', s)
    m_conf = re.search(r'conf=([0-9.]+)', s)
    if m_lev: levs.append(float(m_lev.group(1)))
    if m_ev: evs.append(float(m_ev.group(1)))
    if m_kelly: kellys.append(float(m_kelly.group(1)))
    if m_conf: confs.append(float(m_conf.group(1)))

if levs:
    import numpy as np
    levs = np.array(levs)
    print(f"  Leverage: mean={np.mean(levs):.2f} median={np.median(levs):.2f} min={np.min(levs):.2f} max={np.max(levs):.2f}")
    print(f"  Lev distribution: 1x={sum(levs<=1.1)}, 2-6x={sum((levs>1.1)&(levs<=6))}, 7-10x={sum((levs>6)&(levs<=10))}, >10x={sum(levs>10)}")
if evs:
    evs = np.array(evs)
    print(f"  EV: mean={np.mean(evs):.6f} median={np.median(evs):.6f} min={np.min(evs):.6f} max={np.max(evs):.6f}")
if kellys:
    kellys = np.array(kellys)
    print(f"  Kelly: mean={np.mean(kellys):.6f} median={np.median(kellys):.6f} min={np.min(kellys):.6f} max={np.max(kellys):.6f}")
if confs:
    confs = np.array(confs)
    print(f"  Confidence: mean={np.mean(confs):.4f} median={np.median(confs):.4f}")
print()

# Entry size analysis
print("=== ENTRY SIZE ANALYSIS ===")
for e in entries[-15:]:
    m_not = re.search(r'notional=([0-9.]+)', e)
    m_lev = re.search(r'lev=([0-9.]+)x', e)
    m_size = re.search(r'size=([0-9.]+)%', e)
    m_sym = re.search(r'\[([A-Z0-9/]+:USDT)\]', e)
    sym = m_sym.group(1) if m_sym else '?'
    notional = float(m_not.group(1)) if m_not else 0
    lev = float(m_lev.group(1)) if m_lev else 0
    size_pct = float(m_size.group(1)) if m_size else 0
    print(f"  {sym:25s} notional={notional:8.2f} lev={lev:5.1f}x size={size_pct:5.2f}%")

# Config analysis
print("\n=== CURRENT ENV CONFIG ===")
import os
env_keys = [
    'MAX_TOTAL_EXPOSURE_PCT', 'MAX_SINGLE_EXPOSURE_PCT',
    'UNI_LEV_MAX', 'UNI_LEV_MAX_CHOP', 'UNI_LEV_MAX_TREND',
    'UNI_MAX_POS_FRAC', 'UNI_MAX_POS_FRAC_CHOP',
    'UNI_HARD_NOTIONAL_CAP', 'UNI_MAX_TOTAL_EXPOSURE',
    'TOP_N_SYMBOLS', 'K_LEV',
    'DEFAULT_TP_PCT', 'DEFAULT_SL_PCT',
]
for k in env_keys:
    v = os.environ.get(k, 'UNSET')
    print(f"  {k}: {v}")
