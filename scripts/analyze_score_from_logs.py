#!/usr/bin/env python3
"""
로그 파일 기반 UnifiedScore 분포 분석

사용법:
  python scripts/analyze_score_from_logs.py [로그파일경로]
  
예시:
  python scripts/analyze_score_from_logs.py engine_stdout_final.log
  
기능:
  - 최근 N개 의사결정에서 UnifiedScore 추출
  - 필터 차단 원인 집계
  - 최적 threshold 자동 제안
"""

import re
import sys
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

def parse_log_file(log_path: Path, max_lines: int = 5000):
    """로그 파일에서 UnifiedScore 및 필터 정보 추출"""
    scores = []
    filter_blocks = defaultdict(int)
    filter_pass_count = 0
    direction_reasons = Counter()
    tp_blocks = 0
    
    # 최근 max_lines만 읽기
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = lines[-max_lines:]
    except FileNotFoundError:
        print(f"[ERROR] 로그 파일을 찾을 수 없습니다: {log_path}")
        return None
    except Exception as e:
        print(f"[ERROR] 로그 파일 읽기 실패: {e}")
        return None
    
    # Pattern: [FILTER] SYMBOL blocked: ['filter1', 'filter2']
    filter_pattern = re.compile(r"\[FILTER\]\s+(\w+)\s+blocked:\s+\[([^\]]+)\]")
    
    # Pattern: [FILTER] SYMBOL all_pass
    pass_pattern = re.compile(r"\[FILTER\]\s+(\w+)\s+all_pass")
    
    # Pattern: UnifiedScore 값 (다양한 위치에서 출력될 수 있음)
    # 예: unified_score=0.001234 또는 unified_score: 0.001234
    score_pattern = re.compile(r"unified_score[=:]\s*([-\d.]+)")
    
    # Pattern: direction_reason 추출
    # 예: direction_reason: EV_LOW | TP_GATED(2.5%)
    reason_pattern = re.compile(r"direction_reason[=:]\s*([^\n|]+)")
    
    # Pattern: TP_GATED 추출
    tp_gated_pattern = re.compile(r"TP_GATED\(")
    
    for line in lines:
        # 필터 차단 추출
        match_block = filter_pattern.search(line)
        if match_block:
            sym = match_block.group(1)
            blocked_filters_str = match_block.group(2)
            # 'unified', 'spread' 같은 필터명 추출
            blocked_filters = [f.strip().strip("'\"") for f in blocked_filters_str.split(",")]
            for flt in blocked_filters:
                filter_blocks[flt] += 1
        
        # 필터 통과 추출
        match_pass = pass_pattern.search(line)
        if match_pass:
            filter_pass_count += 1
        
        # UnifiedScore 값 추출
        match_score = score_pattern.search(line)
        if match_score:
            try:
                score_val = float(match_score.group(1))
                scores.append(score_val)
            except ValueError:
                pass
        
        # Direction 사유 추출
        match_reason = reason_pattern.search(line)
        if match_reason:
            reason = match_reason.group(1).strip()
            direction_reasons[reason] += 1
        
        # TP_GATED 카운트
        if tp_gated_pattern.search(line):
            tp_blocks += 1
    
    return {
        "scores": scores,
        "filter_blocks": dict(filter_blocks),
        "filter_pass_count": filter_pass_count,
        "direction_reasons": direction_reasons.most_common(10),
        "tp_blocks": tp_blocks,
    }

def analyze_and_recommend(data):
    """분석 결과 출력 및 threshold 추천"""
    if not data:
        return
    
    scores = data["scores"]
    filter_blocks = data["filter_blocks"]
    filter_pass_count = data["filter_pass_count"]
    direction_reasons = data["direction_reasons"]
    tp_blocks = data["tp_blocks"]
    
    print("="*70)
    print("📊 UnifiedScore 분포 분석 (최근 로그 기반)")
    print("="*70)
    
    if not scores:
        print("\n[WARN] UnifiedScore 데이터를 찾을 수 없습니다.")
        print("  → 로그에 unified_score 출력이 없거나, 로그 파일이 오래되었을 수 있습니다.")
        print("  → main_engine_mc_v2_final.py에서 unified_score를 로깅하는지 확인하세요.")
    else:
        scores_arr = np.array(scores)
        print(f"\n📈 분포 통계 (샘플 수: {len(scores)})")
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
        
        # 음수/양수 비율
        neg_count = (scores_arr < 0).sum()
        pos_count = (scores_arr >= 0).sum()
        print(f"\n  음수: {neg_count} ({neg_count/len(scores)*100:.1f}%)")
        print(f"  양수: {pos_count} ({pos_count/len(scores)*100:.1f}%)")
    
    # 필터 차단 통계
    print("\n" + "="*70)
    print("🚦 필터 차단 통계")
    print("="*70)
    
    if filter_blocks:
        total_blocks = sum(filter_blocks.values())
        print(f"\n  총 차단 횟수: {total_blocks}")
        print(f"  통과 횟수   : {filter_pass_count}")
        print(f"\n  필터별 차단 횟수:")
        for flt, count in sorted(filter_blocks.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_blocks) * 100
            print(f"    {flt:15s}: {count:4d} ({pct:5.1f}%)")
    else:
        print("\n  (필터 차단 로그 없음)")
    
    # Direction 사유 분석
    if direction_reasons:
        print("\n" + "="*70)
        print("📋 진입 차단 사유 TOP 10")
        print("="*70)
        for reason, count in direction_reasons:
            print(f"  {reason:60s}: {count:4d}")
    
    if tp_blocks > 0:
        print(f"\n⚠️  TP_GATED (TP 확률 부족) 차단: {tp_blocks}회")
        print("  → TP 확률(policy_tp_5m)이 15% 미만이어서 진입이 차단되었습니다.")
        print("  → MC 시뮬레이션 파라미터(TP_MULTIPLIER 등)를 조정하거나,")
        print("  → POLICY_P_TP_ENTER_MIN을 낮춰서 TP 게이트를 완화할 수 있습니다.")
    
    # 최적 threshold 제안
    if scores:
        print("\n" + "="*70)
        print("💡 최적 UNIFIED_ENTRY_FLOOR 제안")
        print("="*70)
        
        p25 = np.percentile(scores_arr, 25)
        p50 = np.percentile(scores_arr, 50)
        p75 = np.percentile(scores_arr, 75)
        mean = scores_arr.mean()
        median = np.median(scores_arr)
        
        current = -0.0001
        
        print(f"\n  현재 설정: {current:.6f}")
        current_pass = (scores_arr >= current).sum()
        current_pass_rate = (current_pass / len(scores_arr)) * 100
        print(f"    → {current_pass}/{len(scores_arr)} ({current_pass_rate:.1f}%) 통과\n")
        
        print("  권장 옵션:")
        print(f"    1. Conservative (P25): UNIFIED_ENTRY_FLOOR={p25:.6f}")
        print(f"       → 상위 75% 신호만 진입 (강력한 필터)")
        
        print(f"\n    2. Moderate (P50)    : UNIFIED_ENTRY_FLOOR={p50:.6f}")
        print(f"       → 상위 50% 신호만 진입 (균형)")
        
        print(f"\n    3. Balanced (Mean)   : UNIFIED_ENTRY_FLOOR={mean:.6f}")
        print(f"       → 평균 이상 신호만 진입 (추천)")
        
        print(f"\n    4. Aggressive (P75)  : UNIFIED_ENTRY_FLOOR={p75:.6f}")
        print(f"       → 상위 25% 신호만 진입 (고위험·고수익)")
        
        # 어느 정도가 적절한지 가이드
        print("\n  📌 선택 가이드:")
        print("    - 현재 진입이 거의 없다면: Mean 또는 P50 추천")
        print("    - 손실이 많다면: P75 (더 강한 필터)")
        print("    - 기회를 더 잡고 싶다면: P25 또는 0.0")
        
        # 실제 적용 명령어
        print("\n  🛠️ 적용 방법:")
        print(f"    echo 'UNIFIED_ENTRY_FLOOR={mean:.6f}' >> .env.midterm")
        print(f"    # 또는 .env.midterm 파일을 직접 수정 후 엔진 재시작")

def main():
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        # 기본 로그 파일 경로
        candidates = [
            Path("engine_stdout_final.log"),
            Path("nohup.out"),
            Path("logs/engine.log"),
        ]
        log_file = None
        for candidate in candidates:
            if candidate.exists():
                log_file = candidate
                break
        
        if log_file is None:
            print("[ERROR] 로그 파일을 찾을 수 없습니다.")
            print("사용법: python scripts/analyze_score_from_logs.py [로그파일경로]")
            sys.exit(1)
    
    print(f"📁 로그 파일: {log_file}")
    print(f"🔍 분석 중...\n")
    
    data = parse_log_file(log_file, max_lines=10000)
    if data:
        analyze_and_recommend(data)
    
    print("\n" + "="*70)
    print("✅ 분석 완료")
    print("="*70)

if __name__ == "__main__":
    main()
