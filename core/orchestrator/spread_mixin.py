"""
SpreadMixin - 스프레드/차익거래 기능
====================================

LiveOrchestrator에서 분리된 스프레드 관련 메서드들.
- 스프레드 시그널 계산 (_spread_signal)
- 스프레드 포지션 관리 (_manage_spreads)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    pass


class SpreadMixin:
    """스프레드/차익거래 믹스인"""

    def _spread_signal(
        self,
        sym1: str,
        sym2: str,
        lookback: int = 20
    ) -> Tuple[float, float, float]:
        """
        두 심볼 간 스프레드 시그널 계산.
        
        페어 트레이딩을 위한 z-score 기반 시그널.
        
        Args:
            sym1: 첫 번째 심볼
            sym2: 두 번째 심볼
            lookback: 이동평균 계산에 사용할 기간
            
        Returns:
            (z_score, spread_mean, spread_std)
        """
        import numpy as np
        
        ohlcv1 = self._ohlcv_buffer.get(sym1, {})
        ohlcv2 = self._ohlcv_buffer.get(sym2, {})
        
        closes1 = ohlcv1.get("closes", [])
        closes2 = ohlcv2.get("closes", [])
        
        if len(closes1) < lookback or len(closes2) < lookback:
            return 0.0, 0.0, 0.0
        
        # 최근 데이터만 사용
        c1 = np.array(closes1[-lookback:], dtype=np.float64)
        c2 = np.array(closes2[-lookback:], dtype=np.float64)
        
        # 가격 비율 기반 스프레드
        # 로그 스프레드: log(P1) - log(P2)
        log_spread = np.log(c1) - np.log(c2)
        
        spread_mean = float(np.mean(log_spread))
        spread_std = float(np.std(log_spread))
        
        if spread_std < 1e-8:
            return 0.0, spread_mean, spread_std
        
        # 현재 스프레드의 z-score
        current_spread = float(log_spread[-1])
        z_score = (current_spread - spread_mean) / spread_std
        
        return z_score, spread_mean, spread_std

    def _manage_spreads(
        self,
        pairs: List[Tuple[str, str]],
        entry_z: float = 2.0,
        exit_z: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        스프레드 포지션 관리 시그널 생성.
        
        Args:
            pairs: 페어 리스트 [(sym1, sym2), ...]
            entry_z: 진입 z-score 임계값
            exit_z: 청산 z-score 임계값
            
        Returns:
            수행할 액션 리스트
        """
        actions: List[Dict[str, Any]] = []
        
        for sym1, sym2 in pairs:
            z_score, mean, std = self._spread_signal(sym1, sym2)
            
            # 스프레드 포지션 상태 확인
            spread_key = f"{sym1}/{sym2}"
            spread_pos = self._spread_positions.get(spread_key)
            
            if spread_pos is None:
                # 진입 시그널
                if z_score > entry_z:
                    # 스프레드가 높음: sym1 매도, sym2 매수
                    actions.append({
                        "type": "SPREAD_ENTRY",
                        "pair": spread_key,
                        "leg1": {"symbol": sym1, "action": "SELL"},
                        "leg2": {"symbol": sym2, "action": "BUY"},
                        "z_score": z_score,
                    })
                elif z_score < -entry_z:
                    # 스프레드가 낮음: sym1 매수, sym2 매도
                    actions.append({
                        "type": "SPREAD_ENTRY",
                        "pair": spread_key,
                        "leg1": {"symbol": sym1, "action": "BUY"},
                        "leg2": {"symbol": sym2, "action": "SELL"},
                        "z_score": z_score,
                    })
            else:
                # 청산 시그널
                entry_z_stored = spread_pos.get("entry_z", 0.0)
                
                # 평균 회귀 완료
                if abs(z_score) < exit_z:
                    actions.append({
                        "type": "SPREAD_EXIT",
                        "pair": spread_key,
                        "z_score": z_score,
                        "reason": "MEAN_REVERSION",
                    })
                # 손절: 반대 방향으로 더 벌어짐
                elif (entry_z_stored > 0 and z_score > entry_z_stored * 1.5) or \
                     (entry_z_stored < 0 and z_score < entry_z_stored * 1.5):
                    actions.append({
                        "type": "SPREAD_EXIT",
                        "pair": spread_key,
                        "z_score": z_score,
                        "reason": "STOP_LOSS",
                    })
        
        return actions

    def _cointegration_test(
        self,
        sym1: str,
        sym2: str,
        lookback: int = 100
    ) -> Tuple[bool, float]:
        """
        두 심볼의 공적분 테스트 (간소화).
        
        Args:
            sym1: 첫 번째 심볼
            sym2: 두 번째 심볼
            lookback: 테스트에 사용할 기간
            
        Returns:
            (공적분 여부, 상관계수)
        """
        import numpy as np
        
        ohlcv1 = self._ohlcv_buffer.get(sym1, {})
        ohlcv2 = self._ohlcv_buffer.get(sym2, {})
        
        closes1 = ohlcv1.get("closes", [])
        closes2 = ohlcv2.get("closes", [])
        
        if len(closes1) < lookback or len(closes2) < lookback:
            return False, 0.0
        
        c1 = np.array(closes1[-lookback:], dtype=np.float64)
        c2 = np.array(closes2[-lookback:], dtype=np.float64)
        
        # 로그 수익률 상관
        ret1 = np.diff(np.log(c1))
        ret2 = np.diff(np.log(c2))
        
        corr = float(np.corrcoef(ret1, ret2)[0, 1])
        
        # 간소화된 공적분 판단: 높은 상관관계 + 스프레드 정상성
        # 실제로는 ADF 테스트 등을 사용해야 함
        is_cointegrated = abs(corr) > 0.7
        
        return is_cointegrated, corr
