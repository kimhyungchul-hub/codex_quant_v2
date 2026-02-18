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
import secrets
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("research.telegram_control")


class TelegramResearchControl:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.offset_file = self.project_root / "state" / "telegram_research_offset.json"
        self.history_file = self.project_root / "state" / "telegram_research_chat_history.json"
        self.pending_write_file = self.project_root / "state" / "telegram_research_pending_writes.json"
        self.audit_log_file = self.project_root / "state" / "telegram_research_fs_audit.jsonl"
        self.token = self._env_or_file("TELEGRAM_BOT_TOKEN")
        self.chat_id = self._env_or_file("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self._offset = self._load_offset()
        try:
            self.write_confirm_ttl_sec = int(self._env_or_file("TELEGRAM_WRITE_CONFIRM_TTL_SEC", "600") or 600)
        except Exception:
            self.write_confirm_ttl_sec = 600
        try:
            self.read_max_bytes = int(self._env_or_file("TELEGRAM_READ_MAX_BYTES", "24000") or 24000)
        except Exception:
            self.read_max_bytes = 24000
        try:
            self.write_max_bytes = int(self._env_or_file("TELEGRAM_WRITE_MAX_BYTES", "24000") or 24000)
        except Exception:
            self.write_max_bytes = 24000

    def _audit_fs(
        self,
        *,
        chat_id: str,
        action: str,
        path: str | None,
        status: str,
        reason: str = "",
        nonce: str | None = None,
        bytes_count: int | None = None,
    ) -> None:
        row = {
            "ts": int(time.time()),
            "chat_id": str(chat_id or ""),
            "action": str(action or ""),
            "path": str(path or ""),
            "status": str(status or ""),
            "reason": str(reason or ""),
            "nonce": str(nonce or ""),
            "bytes": int(bytes_count or 0),
        }
        try:
            self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _load_pending_writes(self) -> dict[str, dict]:
        try:
            if not self.pending_write_file.exists():
                return {}
            row = json.loads(self.pending_write_file.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                return {str(k): v for k, v in row.items() if isinstance(v, dict)}
        except Exception:
            return {}
        return {}

    def _save_pending_writes(self, rows: dict[str, dict]) -> None:
        try:
            self.pending_write_file.parent.mkdir(parents=True, exist_ok=True)
            self.pending_write_file.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _is_protected_path(self, p: Path) -> tuple[bool, str]:
        try:
            rel = p.resolve().relative_to(self.project_root)
        except Exception:
            return True, "system_path_blocked"
        rel_posix = rel.as_posix()
        name = p.name.lower()
        parts = [x.lower() for x in p.parts]
        if name == "bybit.env" or rel_posix == "state/bybit.env":
            return True, "protected_bybit_env"
        if ".git" in parts or rel_posix.startswith(".git/") or rel_posix == ".git":
            return True, "protected_git"
        if ".ssh" in parts:
            return True, "protected_ssh"
        if name in {"id_rsa", "id_ed25519", "authorized_keys", "known_hosts"}:
            return True, "protected_ssh"
        if name.endswith(".pem") or name.endswith(".ppk"):
            return True, "protected_ssh"
        return False, ""

    def _resolve_workspace_path(self, raw_path: str) -> tuple[Path | None, str]:
        s = str(raw_path or "").strip().strip("'").strip('"')
        if not s:
            return None, "missing_path"
        try:
            p = Path(s)
            if p.is_absolute():
                rp = p.resolve()
            else:
                rp = (self.project_root / p).resolve()
        except Exception:
            return None, "invalid_path"
        try:
            _ = rp.relative_to(self.project_root)
        except Exception:
            return None, "system_path_blocked"
        blocked, reason = self._is_protected_path(rp)
        if blocked:
            return None, reason
        return rp, ""

    def read_workspace_file(self, *, path: str, chat_id: str) -> dict[str, Any]:
        rp, reason = self._resolve_workspace_path(path)
        if rp is None:
            self._audit_fs(chat_id=chat_id, action="read", path=path, status="deny", reason=reason)
            return {"status": "error", "reason": reason}
        if not rp.exists():
            self._audit_fs(chat_id=chat_id, action="read", path=str(rp), status="error", reason="not_found")
            return {"status": "error", "reason": "not_found", "path": str(rp)}
        if rp.is_dir():
            self._audit_fs(chat_id=chat_id, action="read", path=str(rp), status="error", reason="is_directory")
            return {"status": "error", "reason": "is_directory", "path": str(rp)}
        try:
            raw = rp.read_bytes()
            truncated = False
            if len(raw) > self.read_max_bytes:
                raw = raw[: self.read_max_bytes]
                truncated = True
            text = raw.decode("utf-8", errors="replace")
            self._audit_fs(chat_id=chat_id, action="read", path=str(rp), status="ok", bytes_count=len(raw))
            return {
                "status": "ok",
                "path": str(rp),
                "text": text,
                "truncated": bool(truncated),
            }
        except Exception as e:
            self._audit_fs(chat_id=chat_id, action="read", path=str(rp), status="error", reason=str(e)[:200])
            return {"status": "error", "reason": f"read_failed: {e}", "path": str(rp)}

    def stage_workspace_write(self, *, path: str, content: str, chat_id: str) -> dict[str, Any]:
        rp, reason = self._resolve_workspace_path(path)
        if rp is None:
            self._audit_fs(chat_id=chat_id, action="write_stage", path=path, status="deny", reason=reason)
            return {"status": "error", "reason": reason}
        payload = str(content or "")
        if not payload:
            self._audit_fs(chat_id=chat_id, action="write_stage", path=str(rp), status="error", reason="empty_content")
            return {"status": "error", "reason": "empty_content"}
        raw = payload.encode("utf-8", errors="replace")
        if len(raw) > self.write_max_bytes:
            self._audit_fs(chat_id=chat_id, action="write_stage", path=str(rp), status="error", reason="content_too_large", bytes_count=len(raw))
            return {"status": "error", "reason": f"content_too_large(max={self.write_max_bytes}B)"}

        pending = self._load_pending_writes()
        now_ts = int(time.time())
        ttl = max(60, int(self.write_confirm_ttl_sec))
        # prune expired
        keep: dict[str, dict] = {}
        for k, v in pending.items():
            exp = int(v.get("expires_at") or 0)
            if exp > now_ts:
                keep[k] = v
        pending = keep

        nonce = secrets.token_hex(4)
        pending[nonce] = {
            "chat_id": str(chat_id or ""),
            "path": str(rp),
            "content": payload,
            "created_at": now_ts,
            "expires_at": now_ts + ttl,
            "sha256": sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        self._save_pending_writes(pending)
        self._audit_fs(
            chat_id=chat_id,
            action="write_stage",
            path=str(rp),
            status="staged",
            nonce=nonce,
            bytes_count=len(raw),
        )
        return {
            "status": "ok",
            "nonce": nonce,
            "path": str(rp),
            "sha256": sha256(raw).hexdigest(),
            "bytes": len(raw),
            "expires_in_sec": ttl,
        }

    def confirm_workspace_write(self, *, nonce: str, chat_id: str) -> dict[str, Any]:
        key = str(nonce or "").strip()
        if not key:
            self._audit_fs(chat_id=chat_id, action="write_confirm", path="", status="error", reason="missing_nonce")
            return {"status": "error", "reason": "missing_nonce"}
        pending = self._load_pending_writes()
        row = pending.get(key)
        if not isinstance(row, dict):
            self._audit_fs(chat_id=chat_id, action="write_confirm", path="", status="error", reason="nonce_not_found", nonce=key)
            return {"status": "error", "reason": "nonce_not_found"}
        now_ts = int(time.time())
        exp = int(row.get("expires_at") or 0)
        if exp <= now_ts:
            pending.pop(key, None)
            self._save_pending_writes(pending)
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(row.get("path") or ""), status="error", reason="nonce_expired", nonce=key)
            return {"status": "error", "reason": "nonce_expired"}
        owner_chat = str(row.get("chat_id") or "")
        if owner_chat and str(chat_id or "") != owner_chat:
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(row.get("path") or ""), status="deny", reason="chat_id_mismatch", nonce=key)
            return {"status": "error", "reason": "chat_id_mismatch"}

        target = Path(str(row.get("path") or ""))
        blocked, reason = self._is_protected_path(target)
        if blocked:
            pending.pop(key, None)
            self._save_pending_writes(pending)
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(target), status="deny", reason=reason, nonce=key)
            return {"status": "error", "reason": reason}
        try:
            _ = target.resolve().relative_to(self.project_root)
        except Exception:
            pending.pop(key, None)
            self._save_pending_writes(pending)
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(target), status="deny", reason="system_path_blocked", nonce=key)
            return {"status": "error", "reason": "system_path_blocked"}

        content = str(row.get("content") or "")
        raw = content.encode("utf-8", errors="replace")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            pending.pop(key, None)
            self._save_pending_writes(pending)
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(target), status="ok", nonce=key, bytes_count=len(raw))
            return {
                "status": "ok",
                "path": str(target),
                "bytes": len(raw),
                "sha256": str(row.get("sha256") or ""),
            }
        except Exception as e:
            self._audit_fs(chat_id=chat_id, action="write_confirm", path=str(target), status="error", reason=str(e)[:200], nonce=key)
            return {"status": "error", "reason": f"write_failed: {e}", "path": str(target)}

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
            "- /rq_read <상대경로> : 워크스페이스 파일 읽기(보호경로 제외)\n"
            "- /rq_write <상대경로>\\n<내용> : 쓰기 요청 스테이징\n"
            "- /rq_confirm <nonce> : 스테이징된 쓰기 승인/실행\n"
            "- /rq_cf_now : 다음 CF 연구 사이클 즉시 시작\n"
            "- 일반 문장 : 자동으로 /rq_ask 처리\n"
            "- 자연어 '모두 적용해'류 : 자동으로 /rq_apply all 처리\n"
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
        if low in ("/start", "/help"):
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

        if low.startswith("/rq_read"):
            parts = raw.split(maxsplit=1)
            p = parts[1].strip() if len(parts) > 1 else ""
            if not p:
                return {"type": "error", "message": "사용법: /rq_read <상대경로>"}
            return {"type": "read_workspace_file", "path": p}

        if low.startswith("/rq_write"):
            body = raw[len("/rq_write") :].lstrip()
            if not body:
                return {"type": "error", "message": "사용법: /rq_write <상대경로>\\n<내용>"}
            first_line, sep, rest = body.partition("\n")
            p = first_line.strip()
            content = rest if sep else ""
            if not p or not content:
                return {"type": "error", "message": "사용법: /rq_write <상대경로>\\n<내용>"}
            return {"type": "stage_workspace_write", "path": p, "content": content}

        if low.startswith("/rq_confirm"):
            parts = raw.split(maxsplit=1)
            nonce = parts[1].strip() if len(parts) > 1 else ""
            if not nonce:
                return {"type": "error", "message": "사용법: /rq_confirm <nonce>"}
            return {"type": "confirm_workspace_write", "nonce": nonce}

        if raw.startswith("/"):
            return {"type": "error", "message": "알 수 없는 명령입니다. /rq_help 를 사용하세요."}

        # Natural-language routing: "apply all" intent.
        if any(k in low for k in ("모두 적용", "전부 적용", "전체 적용", "all apply", "apply all")):
            return {"type": "apply_openai_suggestions", "selection": "all"}

        # Fallback: treat plain text as /rq_ask.
        return {"type": "ask_openai", "question": raw}

    def _load_history(self) -> list[dict]:
        try:
            if not self.history_file.exists():
                return []
            rows = json.loads(self.history_file.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            return []
        return []

    def _save_history(self, rows: list[dict]) -> None:
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(json.dumps(rows[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def append_chat_turn(self, *, role: str, text: str, chat_id: str, source: str = "telegram") -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        rows = self._load_history()
        rows.append(
            {
                "ts": int(time.time()),
                "role": str(role or "user"),
                "text": msg[:4000],
                "chat_id": str(chat_id or self.chat_id or ""),
                "source": str(source or "telegram"),
            }
        )
        self._save_history(rows)

    def get_recent_chat_context(self, *, chat_id: str, n: int = 8) -> list[dict]:
        rows = self._load_history()
        cid = str(chat_id or self.chat_id or "")
        filtered = [r for r in rows if str(r.get("chat_id") or "") == cid]
        out = []
        for r in filtered[-max(1, int(n)):]:
            out.append(
                {
                    "role": str(r.get("role") or "user"),
                    "text": str(r.get("text") or ""),
                    "ts": int(r.get("ts") or 0),
                }
            )
        return out

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
