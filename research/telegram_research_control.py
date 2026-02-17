"""
research/telegram_research_control.py
-----------------------------------
Telegram command/control bridge for research.runner.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("research.telegram_control")


class TelegramResearchControl:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.offset_file = self.project_root / "state" / "telegram_research_offset.json"
        self.token = self._env_or_file("TELEGRAM_BOT_TOKEN")
        self.chat_id = self._env_or_file("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self._offset = self._load_offset()

    def _env_or_file(self, name: str, default: str = "") -> str:
        v = str(os.environ.get(name, "") or "").strip()
        if v:
            return v
        env_path = self.project_root / "state" / "bybit.env"
        if not env_path.exists():
            return str(default)
        try:
            for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = raw.strip()
                if (not s) or s.startswith("#") or ("=" not in s):
                    continue
                k, val = s.split("=", 1)
                if str(k).strip() == name:
                    return str(val).strip()
        except Exception:
            return str(default)
        return str(default)

    def _load_offset(self) -> int:
        try:
            if not self.offset_file.exists():
                return 0
            row = json.loads(self.offset_file.read_text(encoding="utf-8"))
            return int(row.get("offset") or 0)
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        try:
            self.offset_file.parent.mkdir(parents=True, exist_ok=True)
            self.offset_file.write_text(json.dumps({"offset": int(offset)}, ensure_ascii=False, indent=2), encoding="utf-8")
            self._offset = int(offset)
        except Exception:
            pass

    def _api_call(self, method: str, payload: dict | None = None, timeout: float = 15.0) -> dict | None:
        if not self.enabled:
            return None
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="ignore"))
        except HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                detail = str(e)
            logger.warning(f"Telegram HTTP error ({method}): {e.code} {detail}")
        except URLError as e:
            logger.warning(f"Telegram network error ({method}): {e}")
        except Exception as e:
            logger.warning(f"Telegram call failed ({method}): {e}")
        return None

    def _chunk_text(self, text: str, limit: int = 3500) -> list[str]:
        s = str(text or "").strip()
        if not s:
            return []
        if len(s) <= limit:
            return [s]
        out: list[str] = []
        buf = ""
        for line in s.splitlines():
            ln = line + "\n"
            if len(buf) + len(ln) > limit and buf:
                out.append(buf.rstrip())
                buf = ""
            if len(ln) > limit:
                if buf:
                    out.append(buf.rstrip())
                    buf = ""
                for i in range(0, len(ln), limit):
                    out.append(ln[i : i + limit].rstrip())
            else:
                buf += ln
        if buf.strip():
            out.append(buf.rstrip())
        return out

    def send_text(self, text: str) -> bool:
        if not self.enabled:
            return False
        ok_any = False
        for chunk in self._chunk_text(text):
            resp = self._api_call(
                "sendMessage",
                {
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=10.0,
            )
            if isinstance(resp, dict) and bool(resp.get("ok")):
                ok_any = True
        return ok_any

    def help_text(self) -> str:
        return (
            "[Research Telegram Control]\n"
            "명령어:\n"
            "- /rq_help : 도움말\n"
            "- /rq_status : 현재 연구 설정/주기\n"
            "- /rq_models : OpenAI 사용 가능 모델 목록\n"
            "- /rq_ask <질문> : OpenAI 자유 질의응답\n"
            "- /rq_review : 즉시 OpenAI 리뷰 실행\n"
            "- /rq_review <프롬프트> : 사용자 프롬프트를 추가해 즉시 리뷰\n"
            "- /rq_apply all : 최신 OpenAI 제안 전체 env 변경 적용\n"
            "- /rq_apply 1,3 : 최신 OpenAI 제안 중 1번과 3번만 적용\n"
            "- /rq_cf_now : 다음 CF 연구 사이클 즉시 시작\n"
        )

    def _parse_apply_selection(self, arg: str) -> dict[str, Any]:
        s = str(arg or "").strip()
        if not s:
            return {"ok": False, "reason": "missing_selection"}
        if s.lower() == "all":
            return {"ok": True, "selection": "all"}
        ids: list[int] = []
        for tok in re.split(r"[\s,]+", s):
            t = tok.strip()
            if not t:
                continue
            if t.isdigit():
                ids.append(int(t))
        ids = sorted(set(i for i in ids if i > 0))
        if not ids:
            return {"ok": False, "reason": "invalid_indices"}
        return {"ok": True, "selection": ids}

    def _parse_command(self, text: str) -> dict | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        low = raw.lower()

        if low in ("/rq_help", "/research_help", "연구 도움말", "연구 도움"):
            return {"type": "help"}
        if low in ("/rq_status", "/research_status", "연구 상태"):
            return {"type": "status"}
        if low in ("/rq_models", "/openai_models", "모델 목록"):
            return {"type": "list_models"}
        if low in ("/rq_cf_now", "/research_now", "연구 지금"):
            return {"type": "run_cf_now"}

        if low.startswith("/rq_ask"):
            parts = raw.split(maxsplit=1)
            q = parts[1].strip() if len(parts) > 1 else ""
            if not q:
                return {"type": "error", "message": "사용법: /rq_ask <질문>"}
            return {"type": "ask_openai", "question": q}

        if low.startswith("/rq_review") or low.startswith("/rq_prompt"):
            parts = raw.split(maxsplit=1)
            prompt = parts[1].strip() if len(parts) > 1 else ""
            return {"type": "run_openai_review", "prompt": prompt or None}

        if low.startswith("/rq_apply"):
            parts = raw.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            parsed = self._parse_apply_selection(arg)
            if not parsed.get("ok"):
                return {"type": "error", "message": "사용법: /rq_apply all 또는 /rq_apply 1,3"}
            return {"type": "apply_openai_suggestions", "selection": parsed.get("selection")}

        return None

    def poll_actions(self) -> list[dict]:
        if not self.enabled:
            return []
        resp = self._api_call(
            "getUpdates",
            {
                "offset": int(self._offset) + 1,
                "timeout": 0,
                "allowed_updates": ["message"],
            },
            timeout=20.0,
        )
        if not isinstance(resp, dict) or not bool(resp.get("ok")):
            return []
        rows = resp.get("result")
        if not isinstance(rows, list):
            return []

        actions: list[dict] = []
        max_offset = int(self._offset)
        for row in rows:
            if not isinstance(row, dict):
                continue
            update_id = int(row.get("update_id") or 0)
            if update_id > max_offset:
                max_offset = update_id

            msg = row.get("message")
            if not isinstance(msg, dict):
                continue
            text = msg.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
            chat_id = str(chat.get("id") or "").strip()
            if self.chat_id and chat_id != str(self.chat_id):
                continue

            action = self._parse_command(text)
            if not isinstance(action, dict):
                continue
            action["update_id"] = update_id
            action["chat_id"] = chat_id
            action["text"] = text.strip()
            actions.append(action)

        if max_offset > self._offset:
            self._save_offset(max_offset)
        return actions
