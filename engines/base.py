# engines/base.py
from typing import Any, Dict


class BaseEngine:
    """
    모든 엔진의 기본 클래스
    """
    name: str = "base"
    weight: float = 1.0

    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        컨텍스트를 받아서 결정을 반환
        """
        raise NotImplementedError("Subclass must implement decide()")

