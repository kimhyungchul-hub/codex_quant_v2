"""
core/orchestrator - 오케스트레이터 모듈
======================================

LiveOrchestrator 클래스를 믹스인 조합으로 구성.

Usage:
    from core.orchestrator import LiveOrchestrator
    
    orchestrator = LiveOrchestrator(
        symbols=["BTC/USDT", "ETH/USDT"],
        balance=10000.0,
        ...
    )
"""

from .base import OrchestratorBase, now_ms
from .logging_mixin import LoggingMixin
from .exchange_mixin import ExchangeMixin
from .state_mixin import StateMixin
from .market_data_mixin import MarketDataMixin
from .position_mixin import PositionMixin
from .risk_mixin import RiskMixin
from .decision_mixin import DecisionMixin
from .dashboard_mixin import DashboardMixin
from .data_loop_mixin import DataLoopMixin
from .spread_mixin import SpreadMixin


class LiveOrchestrator(
    LoggingMixin,
    ExchangeMixin,
    StateMixin,
    MarketDataMixin,
    PositionMixin,
    RiskMixin,
    DecisionMixin,
    DashboardMixin,
    DataLoopMixin,
    SpreadMixin,
    OrchestratorBase,
):
    """
    라이브 트레이딩 오케스트레이터.
    
    믹스인 조합으로 구성된 메인 클래스.
    각 믹스인은 독립적인 기능 영역을 담당:
    
    - LoggingMixin: 로깅, 알림, 텔레그램
    - ExchangeMixin: 거래소 API, CCXT 호출
    - StateMixin: 상태 저장/로드, 영구 저장
    - MarketDataMixin: 시장 데이터 계산, 변동성, 레짐
    - PositionMixin: 포지션 관리, 진입/청산
    - RiskMixin: 리스크 관리, 손절/익절
    - DecisionMixin: 의사결정, 합의 로직
    - DashboardMixin: 대시보드, 모니터링
    - DataLoopMixin: 데이터 수집 루프
    - SpreadMixin: 스프레드/차익거래
    - OrchestratorBase: 공통 상태/속성
    
    MRO (Method Resolution Order):
        1. LiveOrchestrator
        2. LoggingMixin
        3. ExchangeMixin
        4. StateMixin
        5. MarketDataMixin
        6. PositionMixin
        7. RiskMixin
        8. DecisionMixin
        9. DashboardMixin
        10. DataLoopMixin
        11. SpreadMixin
        12. OrchestratorBase
        13. object
    """

    def __init__(
        self,
        symbols: list,
        balance: float,
        leverage: float = 1.0,
        exchange: object = None,
        hub: object = None,
        state_dir: str = "state",
        **kwargs,
    ):
        """
        오케스트레이터 초기화.
        
        Args:
            symbols: 거래 심볼 리스트
            balance: 초기 잔고 (USDT)
            leverage: 기본 레버리지
            exchange: CCXT 거래소 인스턴스 (선택)
            hub: 엔진 허브 (선택)
            state_dir: 상태 저장 디렉토리
            **kwargs: 추가 설정 (클래스 속성 오버라이드)
        """
        # 기본 클래스 초기화
        super().__init__()
        
        # 필수 설정
        self.symbols = list(symbols)
        self.balance = float(balance)
        self.leverage = float(leverage)
        self.exchange = exchange
        self.hub = hub
        
        # 추가 속성 (main.py 호환성)
        import asyncio
        self.data_exchange = kwargs.get("data_exchange", exchange)  # 데이터 전용 거래소
        self._net_sem = asyncio.Semaphore(10)  # 네트워크 동시성 제한
        self.enable_orders = False  # 실제 주문 여부 (paper trading 기본)
        self.paper_trading_enabled = True  # 페이퍼 트레이딩 활성화
        
        # kwargs로 클래스 속성 오버라이드
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # 상태 파일 설정
        from pathlib import Path
        state_path = Path(state_dir)
        state_path.mkdir(parents=True, exist_ok=True)
        
        self.state_files = {
            "balance": state_path / "balance.json",
            "equity": state_path / "equity_history.json",
            "trade": state_path / "trade_tape.json",
            "eval": state_path / "eval_history.json",
            "positions": state_path / "positions.json",
        }
        
        # 영구 상태 로드
        self._load_persistent_state()
        
        # 초기 자산 설정
        self._init_initial_equity()
        
        self._log(f"[INIT] LiveOrchestrator: {len(self.symbols)} symbols, balance={self.balance:.2f}")

    async def start(self) -> None:
        """
        오케스트레이터 시작.
        
        모든 비동기 루프를 시작하고 대시보드를 활성화.
        """
        import asyncio
        
        self._shutdown = False
        
        self._log("[START] LiveOrchestrator starting...")
        
        # 거래소 설정 초기화
        if self.exchange:
            await self.init_exchange_settings()
        
        # OHLCV 프리로드
        await self.preload_all_ohlcv()
        
        # 데이터 수집 루프 시작
        tasks = [
            asyncio.create_task(self.fetch_prices_loop()),
            asyncio.create_task(self.fetch_ohlcv_loop()),
            asyncio.create_task(self.fetch_orderbook_loop()),
        ]
        
        self._log("[START] Data loops started")
        
        # 메인 루프는 별도로 실행 (main_engine에서 호출)
        return tasks

    async def stop(self) -> None:
        """
        오케스트레이터 정지.
        
        모든 루프를 종료하고 상태를 저장.
        """
        self._shutdown = True
        
        # 상태 저장
        self._persist_state(force=True)
        
        self._log("[STOP] LiveOrchestrator stopped")


# 하위 호환성을 위한 별칭
Orchestrator = LiveOrchestrator


# Import helper functions from the original orchestrator.py module
# (These are at module level in core/orchestrator.py)
import sys
import os
from pathlib import Path

# Add core directory to path temporarily to import from orchestrator.py file
_core_dir = Path(__file__).parent.parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

# Import directly from the orchestrator.py file (not the package)
try:
    import importlib.util
    _orch_file = _core_dir / "orchestrator.py"
    _spec = importlib.util.spec_from_file_location("_orchestrator_module", _orch_file)
    _orch_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_orch_mod)
    build_exchange = _orch_mod.build_exchange
    build_data_exchange = _orch_mod.build_data_exchange
except Exception as e:
    # Fallback: define dummy functions
    async def build_exchange():
        raise NotImplementedError(f"build_exchange import failed: {e}")
    async def build_data_exchange():
        raise NotImplementedError(f"build_data_exchange import failed: {e}")


__all__ = [
    "LiveOrchestrator",
    "Orchestrator",
    "OrchestratorBase",
    "now_ms",
    # 믹스인 (필요시 개별 사용)
    "LoggingMixin",
    "ExchangeMixin",
    "StateMixin",
    "MarketDataMixin",
    "PositionMixin",
    "RiskMixin",
    "DecisionMixin",
    "DashboardMixin",
    "DataLoopMixin",
    "SpreadMixin",
    # Helper functions
    "build_exchange",
    "build_data_exchange",
]
