"""
LoggingMixin - 로깅 및 알림 기능
================================

LiveOrchestrator에서 분리된 로깅/알림 관련 메서드들.
- 로그 출력 (_log, _log_err)
- 텔레그램 알림 (_send_telegram, _enqueue_alert)
- 이상 징후 기록 (_register_anomaly, _note_runtime_error)
- 알림 스로틀링 (_should_alert)
"""

from __future__ import annotations
import asyncio
import time
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from typing import Optional


class LoggingMixin:
    """로깅 및 알림 믹스인"""
    
    # 상수 (OrchestratorBase에서 초기화됨)
    ALERT_THROTTLE_SEC: int = 300
    ERROR_BURST_WINDOW_SEC: float = 60.0
    ERROR_BURST_LIMIT: int = 10
    
    def _log(self, text: str) -> None:
        """INFO 레벨 로그 출력"""
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "INFO", "msg": text})
        print(text)

    def _log_err(self, text: str) -> None:
        """ERROR 레벨 로그 출력"""
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "ERROR", "msg": text})
        print(text)

    def _should_alert(
        self,
        key: str,
        *,
        now_ts: Optional[float] = None,
        throttle_sec: int | float = 300,
    ) -> bool:
        """
        동일 키로 반복 알림을 방지하는 스로틀링 체크.
        
        Args:
            key: 알림 종류를 구분하는 키 (예: "loss_streak:warn")
            now_ts: 현재 타임스탬프 (생략 시 time.time())
            throttle_sec: 최소 간격 (초)
        
        Returns:
            True면 알림 발송 허용, False면 스로틀링됨
        """
        now_ts = time.time() if now_ts is None else float(now_ts)
        last = float(self._last_alert_ts.get(key, 0.0) or 0.0)
        if (now_ts - last) < float(throttle_sec):
            return False
        self._last_alert_ts[key] = now_ts
        return True

    async def _send_telegram(self, message: str) -> None:
        """텔레그램 메시지 전송"""
        if not self._telegram_enabled:
            return
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {"chat_id": self._telegram_chat_id, "text": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status >= 300:
                        self._log_err(f"[ALERT] Telegram send failed: {resp.status}")
        except Exception as e:
            self._log_err(f"[ALERT] Telegram send error: {e}")

    def _enqueue_alert(self, title: str, message: str) -> None:
        """
        텔레그램 알림을 비동기로 전송 예약.
        
        Args:
            title: 알림 제목
            message: 알림 본문
        """
        if not self._telegram_enabled:
            return
        asyncio.create_task(self._send_telegram(f"{title}\n{message}"))

    def _register_anomaly(
        self,
        kind: str,
        severity: str,
        message: str,
        data: dict | None = None,
    ) -> None:
        """
        이상 징후를 기록하고 필요시 알림 발송.
        
        Args:
            kind: 이상 유형 (예: "loss_streak", "liquidation")
            severity: 심각도 ("info", "warn", "critical")
            message: 설명 메시지
            data: 추가 데이터
        """
        from . import now_ms
        
        ts = now_ms()
        entry = {
            "ts": ts,
            "kind": str(kind),
            "severity": str(severity),
            "message": str(message),
            "data": data or {},
        }
        self.anomalies.append(entry)
        
        # 스로틀링 후 알림 발송
        if self._should_alert(f"{kind}:{severity}"):
            self._enqueue_alert(f"[ALERT:{severity}] {kind}", message)

    def _note_runtime_error(self, context: str, err_text: str) -> None:
        """
        런타임 에러 기록 및 에러 버스트 시 안전 모드 전환.
        
        Args:
            context: 에러 발생 컨텍스트 (예: "decision_loop", "broadcast")
            err_text: 에러 메시지
        """
        now_ts = time.time()
        
        # 버스트 윈도우 리셋
        if (now_ts - self._last_error_ts) > self.ERROR_BURST_WINDOW_SEC:
            self._error_burst = 0
        
        self._last_error_ts = now_ts
        self._error_burst += 1
        
        # 이상 징후 기록
        self._register_anomaly("runtime_error", "critical", f"{context}: {err_text}")
        
        # 에러 버스트 시 안전 모드
        if self._error_burst >= self.ERROR_BURST_LIMIT:
            self.safety_mode = True
            self.enable_orders = False
            self._register_anomaly("safety_mode", "critical", "error burst -> safety mode")
