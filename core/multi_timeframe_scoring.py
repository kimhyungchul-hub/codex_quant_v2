"""
Multi-Timeframe Scoring System
===============================

고빈도 매매(HFT) 및 퀀트 트레이딩 시스템에서 여러 시간대(5m, 10m, 30m, 1h)의 
몬테카를로 시뮬레이션 결과를 통합하고 비용 효율적인 트레이딩 결정을 내리기 위한 핵심 함수들.

Author: Codex Quant Team
Date: 2026-01-01
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# 기본 가중치: 장기 추세(1h) > 중기(30m) > 단기(10m, 5m)
DEFAULT_WEIGHTS = {
    '5m': 0.40,   # 40% - 단기 신호 강조
    '10m': 0.30,  # 30% - 단기 추세
    '30m': 0.20,  # 20% - 중기 추세
    '1h': 0.10,   # 10% - 장기 추세 비중 축소
}


def calculate_consensus_score(
    scores_dict: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Task 1: 통합 점수 산출 (Consensus Score)
    
    여러 시간대의 점수를 가중 평균하여 하나의 조화된 점수(Harmonic Score)를 산출합니다.
    기존의 Max() 방식을 버리고 시간대별 특성을 반영한 가중 합산 방식을 사용합니다.
    
    Parameters
    ----------
    scores_dict : Dict[str, float]
        시간대별 점수 딕셔너리. 예: {'5m': 10.2, '10m': 5.5, '30m': -2.0, '1h': 8.0}
    weights : Optional[Dict[str, float]]
        시간대별 가중치. None이면 DEFAULT_WEIGHTS 사용.
        
    Returns
    -------
    float
        가중 평균된 통합 점수 (consensus_score)
        
    Design Rationale
    ----------------
    - 장기 추세(1h)가 살아있어야 높은 점수를 받도록 40% 가중치 부여
    - 단기(5m)가 너무 역행하면 전체 점수를 깎지만, 영향력은 10%로 제한
    - 이를 통해 단기 노이즈에 의한 과도한 리밸런싱 방지
    
    Cold Start Handling
    -------------------
    - 일부 timeframe 데이터가 누락된 경우, 가용한 데이터로만 가중 평균 계산
    - 가중치를 재정규화(normalize)하여 합이 1.0이 되도록 조정
    - 모든 데이터가 누락된 경우 0.0 반환
    
    Examples
    --------
    >>> scores = {'5m': 10.0, '10m': 8.0, '30m': 6.0, '1h': 12.0}
    >>> calculate_consensus_score(scores)
    9.2  # 0.1*10 + 0.2*8 + 0.3*6 + 0.4*12 = 9.2
    
    >>> scores = {'10m': 5.0, '1h': 10.0}  # 일부 누락
    >>> calculate_consensus_score(scores)
    8.333...  # (0.2*5 + 0.4*10) / (0.2 + 0.4) = 8.33
    """
    if not scores_dict:
        return 0.0
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # 가용한 시간대와 해당 가중치만 추출
    available_scores = []
    available_weights = []
    
    for timeframe, weight in weights.items():
        if timeframe in scores_dict:
            available_scores.append(scores_dict[timeframe])
            available_weights.append(weight)
    
    # Cold Start: 데이터가 없으면 0.0 반환
    if not available_scores:
        return 0.0
    
    # 가중치 정규화 (합이 1.0이 되도록)
    available_scores = np.array(available_scores, dtype=np.float64)
    available_weights = np.array(available_weights, dtype=np.float64)
    
    weight_sum = np.sum(available_weights)
    if weight_sum == 0.0:
        return 0.0
    
    normalized_weights = available_weights / weight_sum
    
    # 가중 평균 계산
    consensus_score = np.dot(available_scores, normalized_weights)
    
    return float(consensus_score)


def calculate_advanced_metrics(
    score_history: List[float],
    window_seconds: int = 300
) -> Tuple[float, float]:
    """
    Task 2: 그룹 및 랭크 산정 (Advanced Metrics)
    
    2초마다 갱신되는 consensus_score의 노이즈를 제거하고,
    자본 할당(Group)과 순위(Rank)를 결정하는 지표를 계산합니다.
    
    Parameters
    ----------
    score_history : List[float]
        최근 5분간(또는 window_seconds)의 consensus_score 리스트 (Time Series).
        시간 순서대로 정렬되어 있어야 함 (oldest -> newest).
    window_seconds : int, optional
        분석 윈도우 크기 (초). 기본값 300초 (5분).
        현재는 사용하지 않지만, 향후 동적 윈도우 조정 시 활용 가능.
        
    Returns
    -------
    Tuple[float, float]
        (group_score, rank_score)
        - group_score: 안정성 기반 점수 (Risk-Adjusted EWMA)
        - rank_score: 모멘텀 기반 점수 (Fast EWMA)
        
    Design Rationale
    ----------------
    Group Score (안정성):
        - EWMA(span=150) - 1.0 * StdDev(score_history)
        - 약 5분 평균 - 변동성 페널티
        - 의도: 안정적으로 높은 점수를 유지하는 종목에 자본 배분
        - 변동성이 크면 페널티를 주어 자본 할당 감소
        
    Rank Score (모멘텀):
        - EWMA(span=15)
        - 약 30초의 빠른 평균
        - 의도: 최근 모멘텀이 강한 종목을 우선 순위에 배치
        - 그룹 진입 경쟁에서 사용
        
    Cold Start Handling
    -------------------
    - 최소 5개 샘플 필요
    - 샘플 부족 시 (0.0, 0.0) 반환
    - EWMA 계산 시 데이터 길이가 span보다 짧아도 작동 (pandas 기본 동작)
    
    Examples
    --------
    >>> history = [10.0] * 150  # 안정적으로 10.0 유지
    >>> group_score, rank_score = calculate_advanced_metrics(history)
    >>> group_score  # ~10.0 (변동성 0이므로 페널티 없음)
    >>> rank_score   # ~10.0
    
    >>> history = list(range(150))  # 0부터 149까지 증가
    >>> group_score, rank_score = calculate_advanced_metrics(history)
    >>> group_score  # 평균 - 높은 StdDev 페널티
    >>> rank_score   # 최근 값에 가까움 (빠른 EWMA)
    """
    # Cold Start: 최소 샘플 수 검증
    if len(score_history) < 1:
        return (0.0, 0.0)
    
    # numpy array로 변환 (벡터 연산 최적화)
    scores = np.array(score_history, dtype=np.float64)
    
    # Group Score: Risk-Adjusted EWMA
    # EWMA(span=900) - 1.0 * StdDev
    # 30분 윈도우 (900개 샘플 = 1800초 / 2초)
    span_long = 900
    alpha_long = 2.0 / (span_long + 1)  # EWMA의 smoothing factor
    
    # EWMA 계산: exponentially weighted moving average
    ewma_long = _calculate_ewma(scores, alpha_long)
    
    # Standard Deviation (전체 이력 기준)
    std_dev = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    
    # Group Score = EWMA - 1.0 * StdDev (안정성 기반)
    group_score = ewma_long - 1.0 * std_dev
    
    # Rank Score: Fast EWMA
    # EWMA(span=30) - 최근 1분 (60초 / 2초 = 30 샘플)
    span_fast = 30
    alpha_fast = 2.0 / (span_fast + 1)
    
    rank_score = _calculate_ewma(scores, alpha_fast)
    
    return (float(group_score), float(rank_score), float(ewma_long), float(std_dev))


def _calculate_ewma(values: np.ndarray, alpha: float) -> float:
    """
    Exponentially Weighted Moving Average (EWMA) 계산 헬퍼 함수.
    
    Parameters
    ----------
    values : np.ndarray
        시계열 데이터 (oldest -> newest)
    alpha : float
        Smoothing factor. alpha = 2 / (span + 1)
        
    Returns
    -------
    float
        가장 최근 시점의 EWMA 값
        
    Notes
    -----
    EWMA 공식: S_t = alpha * x_t + (1 - alpha) * S_{t-1}
    S_0 = x_0 (첫 번째 값으로 초기화)
    """
    if len(values) == 0:
        return 0.0
    
    ewma = values[0]  # 초기값
    
    for val in values[1:]:
        ewma = alpha * val + (1.0 - alpha) * ewma
    
    return ewma


def check_position_switching(
    current_pos: Dict[str, any],
    candidate_pos: Dict[str, any],
    fee_rate: float,
    scaling_factor: float = 4.0
) -> bool:
    """
    Task 3: 포지션 교체 판단 (Switching Logic with Hysteresis)
    
    단순히 랭크가 높다고 교체하지 말고, "교체 비용을 상회하는 이익"이 있을 때만 
    교체하도록 히스테리시스(Hysteresis)를 적용합니다.
    
    Parameters
    ----------
    current_pos : Dict[str, any]
        현재 보유 포지션 정보. 필수 키: 'group_score'
    candidate_pos : Dict[str, any]
        교체 대상 후보 정보. 필수 키: 'group_score'
    fee_rate : float
        편도 수수료율. 예: 0.0005 (0.05%)
    scaling_factor : float, optional
        히스테리시스 배율. 기본값 4.0
        수수료의 N배 이상 마진이 있어야 교체 허용
        
    Returns
    -------
    bool
        True: 교체 실행, False: 현재 포지션 유지
        
    Design Rationale
    ----------------
    Switching Cost:
        - fee_rate * 4.0 * scaling_factor
        - 양방향 수수료(roundtrip) + 실행 비용 + 슬리피지를 고려한 안전 마진
        - 예: fee_rate=0.0005일 때, 최소 0.002 (0.2%) 이상 개선되어야 교체
        
    Condition:
        - candidate['group_score'] > current['group_score'] + Switching_Cost
        - 후보의 안정성 점수가 전환 비용을 상회하는 경우에만 교체
        
    Why Group Score?:
        - Rank Score는 단기 모멘텀이라 노이즈가 많음
        - Group Score는 안정성 기반이므로 자본 재배치 결정에 적합
        
    Examples
    --------
    >>> current = {'group_score': 10.0}
    >>> candidate = {'group_score': 10.5}
    >>> fee_rate = 0.0005
    >>> check_position_switching(current, candidate, fee_rate)
    False  # 0.5 < 0.002 * 4 = 0.008이므로 교체 안 함
    
    >>> candidate = {'group_score': 12.0}
    >>> check_position_switching(current, candidate, fee_rate)
    True  # 2.0 > 0.008이므로 교체
    """
    # 필수 필드 검증
    if 'group_score' not in current_pos:
        raise ValueError("current_pos must have 'group_score' field")
    if 'group_score' not in candidate_pos:
        raise ValueError("candidate_pos must have 'group_score' field")
    
    current_score = float(current_pos['group_score'])
    candidate_score = float(candidate_pos['group_score'])
    
    # Switching Cost 계산
    # fee_rate는 편도이므로, roundtrip = fee_rate * 2
    # 추가 안전 마진을 위해 scaling_factor 적용
    switching_cost = fee_rate * 4.0 * scaling_factor
    
    # 교체 조건: 후보가 (현재 + 전환비용)보다 높은가?
    should_switch = candidate_score > (current_score + switching_cost)
    
    return should_switch


def check_exit_condition(
    position: Dict[str, any],
    current_scores_dict: Dict[str, float],
    consensus_score: float,
    exit_threshold: float = -50.0
) -> Tuple[bool, str]:
    """
    Task 4: 일관성 있는 청산 (Context-Aware Exit)
    
    진입 당시의 시간대(Time Horizon)를 기억하고, 해당 시간대의 논리가 깨졌을 때 청산합니다.
    
    Parameters
    ----------
    position : Dict[str, any]
        포지션 객체. 필수 키: 'entry_tag' (진입 시 주력 시간대, 예: '1h')
        Optional: 'sym', 'side' (로깅용)
    current_scores_dict : Dict[str, float]
        실시간 시간대별 점수. 예: {'5m': 5.0, '10m': 3.0, '30m': -2.0, '1h': -10.0}
    consensus_score : float
        현재의 통합 점수 (calculate_consensus_score 결과)
    exit_threshold : float, optional
        Hard Stop 임계값. 기본값 -50.0
        consensus_score가 이 값 이하로 떨어지면 무조건 청산
        
    Returns
    -------
    Tuple[bool, str]
        (should_exit: bool, reason: str)
        - should_exit: True이면 청산 실행
        - reason: 청산 사유 (로깅 및 분석용)
        
    Design Rationale
    ----------------
    Timeframe Check:
        - 진입 시 태그된 시간대(entry_tag)의 점수가 0 이하로 떨어지면 청산
        - 예: '1h' 태그로 진입했는데 현재 '1h' 점수가 음수 → "진입 논리 붕괴"
        - 이는 진입 근거가 사라졌음을 의미
        
    Hard Stop:
        - consensus_score < exit_threshold (예: -50)이면 태그와 무관하게 즉시 청산
        - 심각한 손실 방지 (손절)
        - 모든 시간대가 동시에 나빠진 극단적 상황 대응
        
    Cold Start Handling:
        - entry_tag가 없는 경우 (레거시 포지션): consensus_score만 사용
        - current_scores_dict에 entry_tag가 없는 경우: 경고 후 consensus만 사용
        
    Examples
    --------
    >>> pos = {'entry_tag': '1h', 'sym': 'BTC/USDT'}
    >>> scores = {'5m': 5.0, '10m': 3.0, '30m': 2.0, '1h': -5.0}
    >>> consensus = calculate_consensus_score(scores)  # 약 -0.5
    >>> should_exit, reason = check_exit_condition(pos, scores, consensus)
    >>> should_exit
    True
    >>> reason
    'Timeframe Logic Broken: 1h score=-5.0 <= 0'
    
    >>> scores = {'5m': -60.0, '10m': -55.0, '30m': -58.0, '1h': -62.0}
    >>> consensus = calculate_consensus_score(scores)  # 약 -59
    >>> should_exit, reason = check_exit_condition(pos, scores, consensus)
    >>> should_exit
    True
    >>> reason
    'Hard Stop: consensus_score=-59.0 < -50.0'
    """
    # Hard Stop: consensus_score가 심각하게 훼손된 경우 (최우선)
    if consensus_score < exit_threshold:
        reason = f"Hard Stop: consensus_score={consensus_score:.1f} < {exit_threshold}"
        return (True, reason)
    
    # Timeframe Check: entry_tag 기반 청산
    entry_tag = position.get('entry_tag')
    
    # Cold Start: entry_tag가 없는 경우 (레거시 포지션)
    if entry_tag is None:
        # consensus_score만으로 판단 (Hard Stop은 이미 체크했으므로 여기서는 False)
        return (False, "No exit condition met (no entry_tag)")
    
    # entry_tag에 해당하는 현재 점수 확인
    if entry_tag not in current_scores_dict:
        # Cold Start: entry_tag 시간대 데이터가 없는 경우
        # 경고하지만 청산하지는 않음 (데이터 누락일 수 있음)
        return (False, f"Warning: entry_tag '{entry_tag}' not in current_scores_dict")
    
    entry_timeframe_score = current_scores_dict[entry_tag]
    
    # 진입 시간대의 점수가 0 이하로 떨어지면 청산
    # 0 이하 = 해당 시간대의 기대 수익이 없거나 음수
    if entry_timeframe_score <= 0.0:
        sym = position.get('sym', 'UNKNOWN')
        reason = f"Timeframe Logic Broken: {entry_tag} score={entry_timeframe_score:.1f} <= 0 [{sym}]"
        return (True, reason)
    
    # 모든 조건을 통과 → 청산하지 않음
    return (False, "No exit condition met")


# ============================================================================
# Utility Functions (향후 확장용)
# ============================================================================

def get_best_entry_tag(scores_dict: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> str:
    """
    가장 기여도가 높은 시간대를 반환 (entry_tag 결정용).
    
    Parameters
    ----------
    scores_dict : Dict[str, float]
        시간대별 점수
    weights : Optional[Dict[str, float]]
        가중치 (None이면 DEFAULT_WEIGHTS)
        
    Returns
    -------
    str
        가장 기여도가 높은 시간대 키 (예: '1h')
        
    Notes
    -----
    기여도 = score * weight
    진입 시 이 timeframe의 논리가 가장 강했음을 의미
    """
    if not scores_dict:
        return '1h'  # 기본값
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    best_tag = None
    best_contribution = -float('inf')
    
    for timeframe, weight in weights.items():
        if timeframe in scores_dict:
            contribution = scores_dict[timeframe] * weight
            if contribution > best_contribution:
                best_contribution = contribution
                best_tag = timeframe
    
    return best_tag if best_tag is not None else '1h'
