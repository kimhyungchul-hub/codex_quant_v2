"""
OrchestratorBase - 오케스트레이터 기본 클래스
=============================================

모든 믹스인이 공유하는 상태와 속성을 정의.
실제 속성 초기화는 LiveOrchestrator.__init__에서 수행.
"""

from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    import ccxt.async_support as ccxt


def now_ms() -> int:
    """현재 시간을 밀리초로 반환."""
    return int(time.time() * 1000)


class OrchestratorBase:
    """
    오케스트레이터 기본 클래스.
    
    모든 믹스인이 참조하는 공통 상태/속성을 선언.
    실제 값 할당은 LiveOrchestrator.__init__에서 수행.
    
    Attributes:
        # 설정
        symbols: 거래 심볼 리스트
        leverage: 기본 레버리지
        balance: 현재 잔고 (USDT)
        initial_equity: 초기 자산 (드로다운 계산용)
        
        # 제한
        MAX_POSITIONS: 최대 동시 포지션 수
        MAX_CAPITAL_UTIL: 최대 자본 활용률 (0~1)
        MAX_NOTIONAL_PER_POSITION: 포지션당 최대 명목가치
        MIN_NOTIONAL: 최소 명목가치
        MAX_LEVERAGE: 최대 레버리지
        MAX_POSITION_HOLD_SEC: 최대 보유 시간 (초)
        
        # 리스크
        STOP_LOSS_PCT: 손절 비율
        TAKE_PROFIT_PCT: 익절 비율
        TRAILING_STOP_PCT: 트레일링 스탑 비율
        MAX_DRAWDOWN: 최대 허용 드로다운
        COOLDOWN_SEC: 재진입 쿨다운 (초)
        
        # 필터
        MIN_CONFIDENCE: 최소 신뢰도
        MIN_EV: 최소 기대값
        
        # 상태
        positions: 현재 포지션 {symbol: pos_dict}
        trade_tape: 거래 기록 리스트
        eval_history: 평가 이력 리스트
        
        # 버퍼
        _latest_prices: 최신 가격 {symbol: price}
        _latest_decisions: 최신 결정 {symbol: decision}
        _ohlcv_buffer: OHLCV 데이터 {symbol: ohlcv_dict}
        _orderbook_buffer: 호가창 {symbol: orderbook_dict}
        _equity_history: 자산 이력 deque
        
        # 내부 상태
        _shutdown: 종료 플래그
        _kill_switch_on: 킬스위치 상태
        _cooldown_until: 쿨다운 타임스탬프 {symbol: timestamp}
        _entry_streak: 연속 진입 횟수 {symbol: count}
        _last_exit_kind: 마지막 청산 유형 {symbol: kind}
        
        # 스프레드
        _spread_positions: 스프레드 포지션 {pair: pos_dict}
        
        # 외부 연결
        exchange: CCXT 거래소 인스턴스
        hub: 엔진 허브
        _dashboard_server: 대시보드 서버
    """

    # ──────────────────────────────────────────────────────────────
    # 설정 기본값 (클래스 레벨)
    # ──────────────────────────────────────────────────────────────
    
    # 포지션 제한
    MAX_POSITIONS: int = 10
    MAX_CAPITAL_UTIL: float = 0.8
    MAX_NOTIONAL_PER_POSITION: float = 10000.0
    MIN_NOTIONAL: float = 10.0
    MAX_LEVERAGE: float = 20.0
    MAX_POSITION_HOLD_SEC: int = 3600
    
    # 리스크 파라미터
    STOP_LOSS_PCT: float = 0.02
    TAKE_PROFIT_PCT: float = 0.05
    TRAILING_STOP_PCT: float = 0.015
    MAX_DRAWDOWN: float = 0.15
    COOLDOWN_SEC: int = 60
    
    # 필터
    MIN_CONFIDENCE: float = 0.6
    MIN_EV: float = 0.001
    
    # 데이터 수집
    OHLCV_TIMEFRAME: str = "1m"
    OHLCV_PRELOAD_LIMIT: int = 100
    OHLCV_INTERVAL_SEC: int = 60
    ORDERBOOK_DEPTH: int = 10
    
    # 텔레그램 (선택)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    def __init__(self) -> None:
        """
        기본 상태 초기화.
        
        Note: 이 메서드는 믹스인 체인에서 호출됨.
              실제 LiveOrchestrator에서 추가 초기화 수행.
        """
        # 심볼 및 설정
        self.symbols: List[str] = []
        self.leverage: float = 1.0
        self.balance: float = 0.0
        self.initial_equity: Optional[float] = None
        
        # 포지션 및 기록
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_tape: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []
        
        # 로깅
        self.logs: deque = deque(maxlen=300)
        
        # 버퍼
        self._latest_prices: Dict[str, float] = {}
        self._latest_decisions: Dict[str, Dict[str, Any]] = {}
        self._ohlcv_buffer: Dict[str, Dict[str, Any]] = {}
        self._orderbook_buffer: Dict[str, Dict[str, Any]] = {}
        self._equity_history: deque = deque(maxlen=10000)
        
        # 내부 상태
        self._shutdown: bool = False
        self._kill_switch_on: bool = False
        self._cooldown_until: Dict[str, float] = {}
        self._entry_streak: Dict[str, int] = {}
        self._last_exit_kind: Dict[str, str] = {}
        
        # 알림
        self._alert_queue: deque = deque(maxlen=100)
        
        # 스프레드
        self._spread_positions: Dict[str, Dict[str, Any]] = {}
        
        # 타이밍
        self._last_state_persist_ms: int = 0
        self._last_broadcast_time: float = 0.0
        self._ticker_interval_sec: float = 0.5
        self._ohlcv_interval_sec: float = 60.0
        self._orderbook_interval_sec: float = 2.0
        
        # 외부 연결 (나중에 설정)
        self.exchange: Optional[Any] = None
        self.hub: Optional[Any] = None
        self._dashboard_server: Optional[Any] = None
        
        # 상태 파일
        self.state_files: Dict[str, Path] = {}
