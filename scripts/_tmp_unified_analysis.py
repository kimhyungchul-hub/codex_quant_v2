#!/usr/bin/env python3
"""Unified Score threshold analysis"""

floor = -0.00030  # UNIFIED_ENTRY_FLOOR
min_entry = 0.0000005  # MIN_ENTRY_SCORE
chop_add = 0.0015  # CHOP_ENTRY_FLOOR_ADD default
bear_add = 0.0010

effective_chop = max(floor, min_entry) + chop_add
effective_bear = max(floor, min_entry) + bear_add
effective_other = max(floor, min_entry)

print("=" * 60)
print("Unified Score Threshold Analysis")
print("=" * 60)
print(f"UNIFIED_ENTRY_FLOOR  = {floor}")
print(f"MIN_ENTRY_SCORE      = {min_entry}")
print(f"CHOP_ENTRY_FLOOR_ADD = {chop_add}")
print(f"BEAR_ENTRY_FLOOR_ADD = {bear_add}")
print()
print(f"Effective floor (chop):  max({floor}, {min_entry}) + {chop_add} = {effective_chop:.6f}")
print(f"Effective floor (bear):  max({floor}, {min_entry}) + {bear_add} = {effective_bear:.6f}")
print(f"Effective floor (other): max({floor}, {min_entry}) = {effective_other:.7f}")
print()
print(f"Score stats from log: Mean=0.000141, Median=0.000064, Max=0.000419")
print(f"Score 0.000141 >= {effective_chop:.6f}(chop)? {0.000141 >= effective_chop}")
print(f"Score 0.000419 >= {effective_chop:.6f}(chop)? {0.000419 >= effective_chop}")
print()
print(f">>> ALL unified scores ({0.000419:.6f} max) < chop threshold ({effective_chop:.6f})")
print(f">>> CHOP_ENTRY_FLOOR_ADD=0.0015 is the PRIMARY BLOCKER")
print(f">>> In chop regime, EVERY symbol fails unified filter")
