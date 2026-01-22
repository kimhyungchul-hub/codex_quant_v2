"""
test_orchestrator_mixins.py
===========================

core/orchestrator/ 믹스인 분리 후 통합 테스트.

테스트 항목:
1. LiveOrchestrator import 및 인스턴스 생성
2. 각 믹스인 메서드 호출 가능 여부
3. 상태 저장/로드 동작
4. 시장 데이터 계산 정확성
5. 의사결정 컨텍스트 생성
"""

import sys
import tempfile
from pathlib import Path

import pytest

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestOrchestratorImport:
    """Import 및 기본 인스턴스화 테스트"""

    def test_import_all_mixins(self):
        """모든 믹스인 개별 import"""
        from core.orchestrator.base import OrchestratorBase, now_ms
        from core.orchestrator.logging_mixin import LoggingMixin
        from core.orchestrator.exchange_mixin import ExchangeMixin
        from core.orchestrator.state_mixin import StateMixin
        from core.orchestrator.market_data_mixin import MarketDataMixin
        from core.orchestrator.position_mixin import PositionMixin
        from core.orchestrator.risk_mixin import RiskMixin
        from core.orchestrator.decision_mixin import DecisionMixin
        from core.orchestrator.dashboard_mixin import DashboardMixin
        from core.orchestrator.data_loop_mixin import DataLoopMixin
        from core.orchestrator.spread_mixin import SpreadMixin

        assert OrchestratorBase is not None
        assert now_ms() > 0

    def test_import_live_orchestrator(self):
        """통합 LiveOrchestrator import"""
        from core.orchestrator import LiveOrchestrator

        assert LiveOrchestrator is not None

    def test_mro_order(self):
        """MRO 순서 확인"""
        from core.orchestrator import LiveOrchestrator

        mro_names = [c.__name__ for c in LiveOrchestrator.__mro__]

        assert mro_names[0] == "LiveOrchestrator"
        assert "LoggingMixin" in mro_names
        assert "OrchestratorBase" in mro_names
        assert mro_names[-1] == "object"

    def test_instantiate_orchestrator(self):
        """인스턴스 생성 테스트"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT", "ETH/USDT"],
                balance=10000.0,
                leverage=5.0,
                state_dir=tmpdir,
            )

            assert orch.balance == 10000.0
            assert orch.leverage == 5.0
            assert len(orch.symbols) == 2


class TestLoggingMixin:
    """LoggingMixin 테스트"""

    def test_log_methods(self):
        """_log, _log_err 메서드"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,
                state_dir=tmpdir,
            )

            orch._log("Test INFO message")
            orch._log_err("Test ERROR message")

            # 로그가 기록되었는지 확인
            assert len(orch.logs) >= 2


class TestMarketDataMixin:
    """MarketDataMixin 테스트"""

    def test_compute_returns_and_vol(self):
        """수익률/변동성 계산"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,
                state_dir=tmpdir,
            )

            closes = [100.0 + i * 0.1 for i in range(50)]
            mu, sigma = orch._compute_returns_and_vol(closes, lookback=20)

            assert isinstance(mu, float)
            assert isinstance(sigma, float)
            assert sigma >= 0

    def test_direction_bias(self):
        """방향성 편향 계산"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,
                state_dir=tmpdir,
            )

            # 상승 추세
            up_closes = [100.0 + i for i in range(20)]
            bias = orch._direction_bias(up_closes)
            assert bias > 0

            # 하락 추세
            down_closes = [100.0 - i for i in range(20)]
            bias = orch._direction_bias(down_closes)
            assert bias < 0

    def test_infer_regime(self):
        """레짐 추정"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,
                state_dir=tmpdir,
            )

            # 낮은 변동성
            stable_closes = [100.0 + i * 0.001 for i in range(50)]
            regime = orch._infer_regime(stable_closes)
            assert regime == "LOW_VOL"

    def test_compute_ofi_score(self):
        """OFI 점수 계산"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,
                state_dir=tmpdir,
            )

            # 매수 우세 호가창
            orderbook = {
                "bids": [[100.0, 10.0], [99.9, 8.0], [99.8, 5.0]],
                "asks": [[100.1, 3.0], [100.2, 2.0], [100.3, 1.0]],
            }
            ofi = orch._compute_ofi_score(orderbook)
            assert ofi > 0  # 매수 우세

            # 빈 호가창
            ofi_empty = orch._compute_ofi_score({})
            assert ofi_empty == 0.0


class TestStateMixin:
    """StateMixin 테스트"""

    def test_persist_and_load_state(self):
        """상태 저장 및 로드"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            # 첫 번째 인스턴스에서 상태 저장
            orch1 = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=5000.0,
                state_dir=tmpdir,
            )
            orch1.balance = 6000.0  # 잔고 변경
            orch1._persist_state(force=True)

            # 두 번째 인스턴스에서 상태 로드
            orch2 = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=1000.0,  # 다른 초기값
                state_dir=tmpdir,
            )

            # 저장된 잔고가 로드되어야 함
            assert orch2.balance == 6000.0


class TestPositionMixin:
    """PositionMixin 테스트"""

    def test_total_open_notional(self):
        """총 노출 계산"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT", "ETH/USDT"],
                balance=10000.0,
                state_dir=tmpdir,
            )

            # 포지션 추가
            orch.positions["BTC/USDT"] = {"notional": 1000.0}
            orch.positions["ETH/USDT"] = {"notional": 500.0}

            total = orch._total_open_notional()
            assert total == 1500.0

    def test_can_enter_position(self):
        """진입 가능 여부 판단"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT", "ETH/USDT"],
                balance=10000.0,
                state_dir=tmpdir,
            )

            # 빈 상태에서 진입 가능
            can = orch._can_enter_position("BTC/USDT", 1000.0)
            assert can is True

            # 이미 포지션 보유 시 진입 불가
            orch.positions["BTC/USDT"] = {"notional": 1000.0}
            can = orch._can_enter_position("BTC/USDT", 500.0)
            assert can is False


class TestRiskMixin:
    """RiskMixin 테스트"""

    def test_dynamic_leverage_risk(self):
        """동적 레버리지 조정"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=10000.0,
                leverage=10.0,
                state_dir=tmpdir,
            )

            # 높은 변동성 -> 레버리지 감소
            lev = orch._dynamic_leverage_risk(10.0, 0.5, confidence=0.8)
            assert lev < 10.0

            # 낮은 변동성 -> 레버리지 유지/증가
            lev_low = orch._dynamic_leverage_risk(10.0, 0.1, confidence=0.8)
            assert lev_low >= lev


class TestDecisionMixin:
    """DecisionMixin 테스트"""

    def test_consensus_action(self):
        """합의 액션 도출"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=10000.0,
                state_dir=tmpdir,
            )

            decisions = [
                {"action": "BUY", "confidence": 0.8},
                {"action": "BUY", "confidence": 0.7},
                {"action": "SELL", "confidence": 0.3},
            ]

            action, ratio = orch._consensus_action(decisions)
            assert action == "BUY"
            assert ratio > 0.5

    def test_ema_update(self):
        """EMA 업데이트"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT"],
                balance=10000.0,
                state_dir=tmpdir,
            )

            # EMA 업데이트
            new_ema = orch._ema_update(100.0, 110.0, alpha=0.5)
            assert new_ema == 105.0  # 0.5 * 110 + 0.5 * 100


class TestSpreadMixin:
    """SpreadMixin 테스트"""

    def test_spread_signal(self):
        """스프레드 시그널 계산"""
        from core.orchestrator import LiveOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            orch = LiveOrchestrator(
                symbols=["BTC/USDT", "ETH/USDT"],
                balance=10000.0,
                state_dir=tmpdir,
            )

            # OHLCV 버퍼 설정
            orch._ohlcv_buffer["BTC/USDT"] = {
                "closes": [100.0 + i * 0.1 for i in range(30)]
            }
            orch._ohlcv_buffer["ETH/USDT"] = {
                "closes": [50.0 + i * 0.05 for i in range(30)]
            }

            z, mean, std = orch._spread_signal("BTC/USDT", "ETH/USDT")

            assert isinstance(z, float)
            assert isinstance(mean, float)
            assert isinstance(std, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
