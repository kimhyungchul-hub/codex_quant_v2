#!/usr/bin/env python3
"""
실시간 UnifiedScore 분포 분석 스크립트

사용법:
  python scripts/analyze_unified_score_live.py

출력:
  - UnifiedScore 분포 (percentile, min, max, mean, median)
  - 필터 통과율 (unified, spread, event_cvar, cooldown)
  - TOP N 진입 가능 심볼 리스트
"""

import asyncio
import json
from pathlib import Path
import sys
import time
import requests
import numpy as np

# API 엔드포인트
API_BASE = "http://localhost:9999"

async def fetch_status():
    """대시보드 API에서 현재 상태 가져오기"""
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] API 호출 실패: {e}")
        return None

def analyze_scores(data):
    """UnifiedScore 분포 분석"""
    if not data or "market" not in data:
        print("[WARN] 데이터 없음")
        return
    
    market = data["market"]
    if not market:
        print("[WARN] market 배열 비어있음")
        return
    
    scores = []
    filters = {
        "unified": {"pass": 0, "block": 0},
        "spread": {"pass": 0, "block": 0},
        "event_cvar": {"pass": 0, "block": 0},
        "cooldown": {"pass": 0, "block": 0},
    }
    
    symbols_data = []
    
    for row in market:
        sym = row.get("symbol", "?")
        unified = row.get("unified_score")
        
        if unified is not None:
            scores.append(float(unified))
        
        # 필터 상태 집계
        fs = row.get("filter_states", {})
        for key in filters:
            state = fs.get(key, True)  # 기본값: True (통과)
            if state:
                filters[key]["pass"] += 1
            else:
                filters[key]["block"] += 1
        
        # 심볼별 상세 정보
        symbols_data.append({
            "symbol": sym,
            "unified_score": unified,
            "status": row.get("status", "?"),
            "filter_states": fs,
            "ev": row.get("ev"),
            "mc": row.get("mc", ""),
        })
    
    # ====== 분포 통계 ======
    if scores:
        scores_arr = np.array(scores)
        print("\n" + "="*60)
        print("📊 UnifiedScore 분포 (현재 시점)")
        print("="*60)
        print(f"  Count     : {len(scores)}")
        print(f"  Mean      : {scores_arr.mean():.6f}")
        print(f"  Median    : {np.median(scores_arr):.6f}")
        print(f"  Std       : {scores_arr.std():.6f}")
        print(f"  Min       : {scores_arr.min():.6f}")
        print(f"  Max       : {scores_arr.max():.6f}")
        print(f"  P05       : {np.percentile(scores_arr, 5):.6f}")
        print(f"  P25       : {np.percentile(scores_arr, 25):.6f}")
        print(f"  P50       : {np.percentile(scores_arr, 50):.6f}")
        print(f"  P75       : {np.percentile(scores_arr, 75):.6f}")
        print(f"  P95       : {np.percentile(scores_arr, 95):.6f}")
    else:
        print("\n[WARN] UnifiedScore 데이터가 없습니다!")
    
    # ====== 필터 통과율 ======
    print("\n" + "="*60)
    print("🚦 필터 통과율")
    print("="*60)
    for key, counts in filters.items():
        total = counts["pass"] + counts["block"]
        if total > 0:
            pass_rate = (counts["pass"] / total) * 100
            print(f"  {key:12s}: {counts['pass']:2d}/{total:2d} ({pass_rate:.1f}% pass)")
        else:
            print(f"  {key:12s}: N/A")
    
    # ====== 진입 가능 심볼 (모든 필터 통과) ======
    print("\n" + "="*60)
    print("✅ 진입 가능 심볼 (모든 필터 통과)")
    print("="*60)
    
    can_enter = []
    for sd in symbols_data:
        fs = sd["filter_states"]
        all_pass = all(fs.get(k, True) for k in ["unified", "spread", "event_cvar", "cooldown"])
        if all_pass and sd["status"] in ("LONG", "SHORT"):
            can_enter.append(sd)
    
    if can_enter:
        # UnifiedScore 기준 정렬
        can_enter_sorted = sorted(can_enter, key=lambda x: x["unified_score"] or -999, reverse=True)
        for rank, sd in enumerate(can_enter_sorted[:10], 1):
            print(f"  #{rank} {sd['symbol']:8s} | Score: {sd['unified_score']:8.6f} | {sd['status']:5s} | EV: {sd['ev']:.6f}")
    else:
        print("  (없음)")
    
    # ====== 필터 차단 심볼 (디버깅용) ======
    print("\n" + "="*60)
    print("❌ 필터 차단 심볼 (상위 10개)")
    print("="*60)
    
    blocked = []
    for sd in symbols_data:
        fs = sd["filter_states"]
        blocked_filters = [k for k in ["unified", "spread", "event_cvar", "cooldown"] if not fs.get(k, True)]
        if blocked_filters:
            blocked.append({**sd, "blocked_by": blocked_filters})
    
    # UnifiedScore 기준 정렬 (점수가 높은데 차단된 것부터)
    blocked_sorted = sorted(blocked, key=lambda x: x["unified_score"] or -999, reverse=True)
    
    for rank, sd in enumerate(blocked_sorted[:10], 1):
        blocked_str = ", ".join(sd["blocked_by"])
        print(f"  #{rank} {sd['symbol']:8s} | Score: {sd['unified_score']:8.6f} | Blocked: [{blocked_str}] | MC: {sd['mc']}")
    
    if not blocked:
        print("  (없음)")
    
    # ====== 최적 threshold 제안 ======
    if scores:
        print("\n" + "="*60)
        print("💡 최적 Threshold 제안")
        print("="*60)
        
        # P25 (하위 75% 차단)
        p25 = np.percentile(scores_arr, 25)
        print(f"  Conservative (P25): {p25:.6f}  ← 상위 75% 신호만 진입")
        
        # P50 (하위 50% 차단)
        p50 = np.percentile(scores_arr, 50)
        print(f"  Moderate (P50)    : {p50:.6f}  ← 상위 50% 신호만 진입")
        
        # P75 (상위 25%만 진입)
        p75 = np.percentile(scores_arr, 75)
        print(f"  Aggressive (P75)  : {p75:.6f}  ← 상위 25% 신호만 진입")
        
        # Mean
        mean = scores_arr.mean()
        print(f"  Balanced (Mean)   : {mean:.6f}  ← 평균 이상 신호만 진입")
        
        # 현재 설정
        current = -0.0001
        current_pass = (scores_arr >= current).sum()
        current_pass_rate = (current_pass / len(scores_arr)) * 100
        print(f"\n  Current ({current:.6f}): {current_pass}/{len(scores_arr)} ({current_pass_rate:.1f}% pass)")

async def main():
    print("="*60)
    print("🔍 UnifiedScore 실시간 분석 도구")
    print("="*60)
    print(f"API: {API_BASE}")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 단일 스냅샷
    data = await fetch_status()
    if data:
        analyze_scores(data)
    else:
        print("[ERROR] 데이터를 가져올 수 없습니다. 엔진이 실행 중인지 확인하세요.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ 분석 완료")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
