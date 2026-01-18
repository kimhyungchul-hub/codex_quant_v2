# engines/dummy_engine.py
from typing import Any, Dict
from engines.base import BaseEngine


class DummyEngine(BaseEngine):
    """
    Fallback 더미 엔진 (엔진이 로드되지 않았을 때 사용)
    """
    name = "dummy"
    weight = 0.0  # 가중치 0으로 설정하여 실제 결정에 영향 없음

    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        항상 WAIT 액션을 반환
        """
        return {
            "action": "WAIT",
            "ev": 0.0,
            "confidence": 0.0,
            "reason": "dummy engine (no engines loaded)",
            "meta": {},
        }

