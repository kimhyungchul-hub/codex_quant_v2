"""
research/openai_reviewer.py — OpenAI API 코드 리뷰 자동화
===========================================================
주기적으로:
  1. git commit + push
  2. 핵심 전략 파일을 OpenAI 모델에 전송
  3. 구조화된 개선 제안(JSON) 수신
  4. 문서화 (docs/OPENAI_REVIEW.md)
  5. (옵션) 안전 범위 내 env 제안 자동 적용

환경변수:
  - OPENAI_API_KEY (없으면 비활성화)
  - OPENAI_REVIEW_MODEL (기본: gpt-4.1-mini)
  - OPENAI_API_URL (기본: https://api.openai.com/v1/responses)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("research.openai_reviewer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_OUTPUT = PROJECT_ROOT / "docs" / "OPENAI_REVIEW.md"
REVIEW_HISTORY = PROJECT_ROOT / "state" / "openai_review_history.json"
REVIEW_LATEST = PROJECT_ROOT / "state" / "openai_review_latest.json"
QA_HISTORY = PROJECT_ROOT / "state" / "openai_qa_history.json"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"

# Files to review (relative to project root)
REVIEW_FILES = [
    "engines/mc/decision.py",
    "engines/mc/config.py",
    "engines/mc/regime_policy.py",
    "engines/mc/entry_evaluation.py",
    "core/orchestrator.py",
    "state/bybit.env",
]
# Supplementary context (shorter files)
CONTEXT_FILES = [
    "docs/SIGNAL_PIPELINE_REFERENCE.md",
    "docs/RESEARCH_FINDINGS.md",
]

MAX_CHARS_PER_FILE = 15000
MAX_TOTAL_CHARS = 120000


def _get_api_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY") or _read_env_key()


def _runtime_model() -> str:
    model = str(os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL).strip()
    return model or DEFAULT_OPENAI_MODEL


def _runtime_api_url() -> str:
    url = str(os.environ.get("OPENAI_API_URL", DEFAULT_OPENAI_API_URL) or DEFAULT_OPENAI_API_URL).strip()
    return url or DEFAULT_OPENAI_API_URL


def _read_env_key() -> Optional[str]:
    env_path = PROJECT_ROOT / "state" / "bybit.env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("OPENAI_API_KEY="):
            return s.split("=", 1)[1].strip()
    return None


def is_available() -> bool:
    return bool(_get_api_key())


def _read_file_truncated(path: Path, max_chars: int = MAX_CHARS_PER_FILE) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            half = max_chars // 2
            return content[:half] + f"\n\n... ({len(content) - max_chars} chars truncated) ...\n\n" + content[-half:]
        return content
    except Exception:
        return f"(Error reading {path})"


def _build_review_prompt(findings: list[dict], apply_history: list[dict]) -> str:
    """Build comprehensive review prompt for OpenAI."""
    parts = []
    parts.append("""# Codex Quant 전략 코드 리뷰 요청

당신은 금융 공학(Financial Engineering) 전문가이며 Python/JAX 시스템 아키텍트입니다.
아래 코인 자동 매매 봇의 핵심 전략 코드를 분석하고, 구체적인 개선 제안을 해주세요.

## 출력 형식 (반드시 JSON)
```json
{
  "summary": "전체 분석 요약 (1-2문장)",
  "risk_score": 7,
  "suggestions": [
    {
      "priority": "P1",
      "category": "logic|param|risk|perf",
      "title": "제안 제목",
      "description": "상세 설명",
      "env_changes": {"VARIABLE_NAME": "new_value"},
      "code_changes": "변경이 필요한 코드 파일과 수정 내용 (텍스트 설명)",
      "expected_impact": "예상 효과",
      "confidence": 0.8
    }
  ],
  "warnings": ["주의사항 1", "주의사항 2"]
}
```

## 최근 CF 분석 결과
""")

    if findings:
        for f in findings[:5]:
            parts.append(
                f"- **{f.get('title')}**: PnL +${f.get('improvement_pct', 0):.2f}, "
                f"신뢰도 {f.get('confidence', 0):.0%}, 파라미터: {f.get('param_changes')}"
            )
    else:
        parts.append("(아직 CF 결과 없음)")

    if apply_history:
        parts.append("\n## 최근 자동 적용 이력")
        for r in apply_history[-3:]:
            parts.append(f"- {r.get('finding_id')}: {r.get('env_changes')} → {r.get('status')}")

    parts.append("\n## 핵심 전략 파일\n")

    total_chars = sum(len(p) for p in parts)
    for fpath in REVIEW_FILES:
        full = PROJECT_ROOT / fpath
        if full.exists():
            remaining = MAX_TOTAL_CHARS - total_chars
            if remaining < 2000:
                parts.append(f"\n### {fpath}\n(생략: 프롬프트 크기 초과)\n")
                continue
            content = _read_file_truncated(full, min(MAX_CHARS_PER_FILE, remaining))
            section = f"\n### {fpath}\n```\n{content}\n```\n"
            parts.append(section)
            total_chars += len(section)

    for fpath in CONTEXT_FILES:
        full = PROJECT_ROOT / fpath
        if full.exists():
            remaining = MAX_TOTAL_CHARS - total_chars
            if remaining < 1000:
                break
            content = _read_file_truncated(full, min(5000, remaining))
            section = f"\n### {fpath} (참고)\n```\n{content}\n```\n"
            parts.append(section)
            total_chars += len(section)

    parts.append("""
## 분석 관점
1. **로직 오류**: 방향(direction) 결정, EV 계산, 레짐 처리에서 논리적 모순이 있는가?
2. **파라미터 최적화**: 현재 bybit.env 설정 중 명백히 비효율적인 값이 있는가?
3. **리스크 관리**: 과도한 레버리지, 부족한 손절, 포지션 사이징 문제가 있는가?
4. **성능**: 불필요한 연산, 메모리 누수, 병목 지점이 있는가?
5. **데이터 파이프라인**: None/NaN 전파, 부호 규약 혼동, 이중 차감 등의 함정이 있는가?

JSON만 출력하세요. 마크다운이나 설명 텍스트 없이 순수 JSON만.
""")

    return "\n".join(parts)


def _extract_response_text(resp_json: dict) -> Optional[str]:
    """Extract text from OpenAI Responses API payload (with compatibility fallbacks)."""
    out_text = resp_json.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text
    if isinstance(out_text, list):
        joined = "\n".join(str(x) for x in out_text if x is not None).strip()
        if joined:
            return joined

    output = resp_json.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                txt = block.get("text") or block.get("output_text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        if parts:
            return "\n".join(parts)

    # chat-completions compatibility fallback
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        txt = block.get("text")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt)
                if parts:
                    return "\n".join(parts)
    return None


def _call_openai(prompt: str, api_key: str, *, model: str, api_url: str) -> Optional[str]:
    def _request(payload: dict) -> Optional[str]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return _extract_response_text(data)

    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0.3,
        "max_output_tokens": 8192,
    }
    try:
        return _request(payload)
    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="ignore")[:400]
        except Exception:
            detail = str(e)
        if e.code == 400 and "temperature" in str(detail).lower() and "not supported" in str(detail).lower():
            # Some models (e.g., codex variants) reject temperature in Responses API.
            try:
                payload2 = dict(payload)
                payload2.pop("temperature", None)
                return _request(payload2)
            except Exception as e2:
                logger.error(f"OpenAI retry without temperature failed: {e2}")
        logger.error(f"OpenAI API HTTP error: {e.code} {detail}")
    except URLError as e:
        logger.error(f"OpenAI API network error: {e}")
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
    return None


def _parse_openai_response(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse OpenAI response: {cleaned[:200]}...")
    return None


def _save_review(review: dict, timestamp: float, *, model: str, source: str = "scheduled") -> None:
    history = []
    if REVIEW_HISTORY.exists():
        try:
            history = json.loads(REVIEW_HISTORY.read_text(encoding="utf-8"))
        except Exception:
            history = []
    entry = {"timestamp": float(timestamp), "model": str(model), "source": str(source), "review": review}
    history.append(entry)
    history = history[-50:]
    REVIEW_HISTORY.write_text(json.dumps(history, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    REVIEW_LATEST.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# OpenAI Code Review — {time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))}",
        "",
        f"**Model:** {model}",
        f"**Risk Score:** {review.get('risk_score', '?')}/10",
        f"**Summary:** {review.get('summary', '?')}",
        "",
        "## Suggestions",
        "",
    ]

    for s in review.get("suggestions", []):
        lines.append(f"### [{s.get('priority', '?')}] {s.get('title', '?')}")
        lines.append(
            f"**Category:** {s.get('category', '?')} | "
            f"**Confidence:** {s.get('confidence', 0):.0%}"
        )
        lines.append(f"\n{s.get('description', '')}")
        if s.get("env_changes"):
            lines.append(f"\n**Env Changes:** `{s['env_changes']}`")
        if s.get("code_changes"):
            lines.append(f"\n**Code Changes:** {s['code_changes']}")
        lines.append(f"\n**Expected Impact:** {s.get('expected_impact', '?')}")
        lines.append("")

    if review.get("warnings"):
        lines.append("## Warnings")
        for w in review["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("\n---\n*Auto-generated by research/openai_reviewer.py*")
    REVIEW_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"OpenAI review saved to {REVIEW_OUTPUT}")


def git_push_all(message: str = "auto: pre-OpenAI review snapshot") -> bool:
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )
        result = subprocess.run(["git", "push"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=60)
        if result.returncode == 0:
            logger.info(f"Git push: {message}")
            return True
        logger.warning(f"Git push failed: {result.stderr.decode(errors='ignore')[:200]}")
        return False
    except Exception as e:
        logger.error(f"Git error: {e}")
        return False


def _apply_env_suggestions(
    suggestions: list[dict],
    *,
    min_confidence: float = 0.7,
    source: str = "openai_review",
    reason: str = "",
) -> list[dict]:
    """Apply env variable changes from OpenAI suggestions (safe keys only)."""
    from research.auto_apply import (
        _apply_runtime_updates,
        _clamp_value,
        _format_env_value,
        _read_env_value,
        PARAM_BOUNDS,
    )

    updates: dict[str, str] = {}
    source_map: dict[str, str] = {}
    for s in suggestions:
        if float(s.get("confidence", 0) or 0.0) < float(min_confidence):
            continue
        env_changes = s.get("env_changes", {})
        if not env_changes:
            continue
        for key, val in env_changes.items():
            if key not in PARAM_BOUNDS:
                logger.info(f"OpenAI suggestion skipped (unknown key): {key}={val}")
                continue
            try:
                fval = float(val)
                clamped = _clamp_value(key, fval)
                old = _read_env_value(key)
                new_str = _format_env_value(clamped, key)
                if old is not None and str(old).strip() == str(new_str).strip():
                    continue
                updates[str(key)] = str(new_str)
                source_map[str(key)] = str(s.get("title") or source)
            except (ValueError, TypeError):
                continue
    if not updates:
        return []

    changed = _apply_runtime_updates(
        updates,
        source=str(source or "openai_review"),
        reason=str(reason or "openai_suggestions"),
        batch_id=f"openai-{int(time.time())}",
    )
    applied: list[dict] = []
    for key, payload in (changed or {}).items():
        if not isinstance(payload, dict):
            continue
        old = payload.get("old")
        new = payload.get("new")
        logger.info(f"[OPENAI_APPLY] {key}: {old} -> {new}")
        applied.append(
            {
                "key": str(key),
                "old": old,
                "new": new,
                "source": source_map.get(str(key), str(source)),
            }
        )
    return applied


def get_latest_review_entry() -> Optional[dict]:
    try:
        if REVIEW_LATEST.exists():
            row = json.loads(REVIEW_LATEST.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                return row
    except Exception:
        pass
    try:
        if REVIEW_HISTORY.exists():
            rows = json.loads(REVIEW_HISTORY.read_text(encoding="utf-8"))
            if isinstance(rows, list) and rows:
                last = rows[-1]
                if isinstance(last, dict):
                    return last
    except Exception:
        pass
    return None


def list_available_models() -> list[str]:
    api_key = _get_api_key()
    if not api_key:
        return []
    api_url = _runtime_api_url().strip()
    if "/v1/" in api_url:
        base = api_url.split("/v1/", 1)[0].rstrip("/")
        models_url = f"{base}/v1/models"
    else:
        models_url = "https://api.openai.com/v1/models"
    req = Request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        logger.error(f"Failed to list OpenAI models: {e}")
        return []
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        mid = str(r.get("id") or "").strip()
        if mid:
            ids.append(mid)
    return sorted(set(ids))


def _append_qa_history(row: dict) -> None:
    rows: list[dict] = []
    try:
        if QA_HISTORY.exists():
            loaded = json.loads(QA_HISTORY.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [x for x in loaded if isinstance(x, dict)]
    except Exception:
        rows = []
    rows.append(row)
    rows = rows[-200:]
    QA_HISTORY.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def ask_openai_question(question: str, *, context: dict | None = None) -> dict:
    q = str(question or "").strip()
    if not q:
        return {"status": "error", "reason": "empty_question"}
    api_key = _get_api_key()
    if not api_key:
        return {"status": "disabled", "reason": "no_api_key"}
    model = str(os.environ.get("OPENAI_ASK_MODEL") or _runtime_model()).strip() or _runtime_model()
    api_url = _runtime_api_url()

    prompt_parts = [
        "# Codex Quant OpenAI Q&A",
        "당신은 기관급 퀀트 트레이딩 시스템 연구 어시스턴트입니다.",
        "요청 언어가 한국어면 한국어로 답하고, 추정은 '추정'이라고 명시하세요.",
        "답변은 짧고 실행 가능한 형태로 작성하세요.",
        "",
        "## 사용자 질문",
        q,
    ]
    if isinstance(context, dict) and context:
        try:
            prompt_parts.extend(
                [
                    "",
                    "## 현재 실행 컨텍스트(JSON)",
                    json.dumps(context, ensure_ascii=False, indent=2)[:10000],
                ]
            )
        except Exception:
            pass
    prompt_parts.extend(
        [
            "",
            "## 응답 형식",
            "- 핵심 답변 3~10줄",
            "- 필요 시 숫자/임계치/명령어를 포함",
        ]
    )
    prompt = "\n".join(prompt_parts)
    text = _call_openai(prompt, api_key, model=model, api_url=api_url)
    if not text:
        return {"status": "error", "reason": "api_call_failed", "model": model}
    answer = str(text).strip()
    out = {
        "status": "ok",
        "model": model,
        "question": q[:800],
        "answer": answer[:12000],
        "timestamp": float(time.time()),
    }
    try:
        _append_qa_history(out)
    except Exception:
        pass
    return out


def apply_latest_review_suggestions(
    *,
    selection: str | list[int] | None = None,
    allow_low_confidence: bool = True,
) -> dict:
    latest = get_latest_review_entry()
    if not latest:
        return {"status": "error", "reason": "no_latest_review"}
    review = latest.get("review") if isinstance(latest.get("review"), dict) else {}
    suggestions = review.get("suggestions") if isinstance(review.get("suggestions"), list) else []
    if not suggestions:
        return {"status": "error", "reason": "no_suggestions"}

    selected: list[dict] = []
    selected_ids: list[int] = []
    if selection is None:
        return {"status": "error", "reason": "selection_required"}

    if isinstance(selection, str) and selection.strip().lower() == "all":
        selected = [s for s in suggestions if isinstance(s, dict)]
        selected_ids = list(range(1, len(selected) + 1))
    else:
        raw_ids = selection if isinstance(selection, list) else []
        for i in raw_ids:
            try:
                idx = int(i)
            except Exception:
                continue
            if 1 <= idx <= len(suggestions):
                s = suggestions[idx - 1]
                if isinstance(s, dict):
                    selected.append(s)
                    selected_ids.append(idx)
    if not selected:
        return {"status": "error", "reason": "empty_selection"}

    from research.auto_apply import backup_env

    backup_env()
    min_conf = -1.0 if allow_low_confidence else 0.7
    applied = _apply_env_suggestions(
        selected,
        min_confidence=min_conf,
        source="openai_manual_apply",
        reason=f"manual_selection:{selection}",
    )
    if applied:
        keys = ", ".join(sorted({str(a.get("key") or "") for a in applied if a.get("key")}))
        git_push_all(f"auto: OpenAI manual apply ({len(applied)} params) [{keys}]")
    return {
        "status": "ok",
        "selected_count": len(selected),
        "selected_indices": selected_ids,
        "applied_count": len(applied),
        "applied_env_changes": applied,
        "review_timestamp": float(latest.get("timestamp") or 0.0),
        "model": str(latest.get("model") or _runtime_model()),
    }


def run_openai_review(
    findings: list[dict] | None = None,
    apply_history: list[dict] | None = None,
    auto_apply_env: bool = False,
    prompt_override: str | None = None,
    source: str = "scheduled",
) -> dict:
    """Run one OpenAI review cycle and return dashboard status."""
    api_key = _get_api_key()
    if not api_key:
        return {"status": "disabled", "reason": "no_api_key"}

    model = _runtime_model()
    api_url = _runtime_api_url()
    ts = time.time()
    status = {"status": "running", "timestamp": ts, "model": model, "source": str(source)}

    # Phase 1: snapshot
    git_push_all(f"auto: pre-openai-review snapshot {time.strftime('%Y-%m-%d %H:%M')}")

    # Phase 2: prompt/call
    prompt = _build_review_prompt(findings or [], apply_history or [])
    if prompt_override:
        prompt = (
            f"{prompt}\n\n## 사용자 추가 분석 요청\n"
            f"{str(prompt_override).strip()[:12000]}\n"
        )
        status["prompt_override"] = str(prompt_override).strip()[:500]
    logger.info(f"Calling OpenAI ({model}), prompt={len(prompt)} chars...")

    response_text = _call_openai(prompt, api_key, model=model, api_url=api_url)
    if not response_text:
        status["status"] = "error"
        status["reason"] = "api_call_failed"
        return status

    # Phase 3: parse
    review = _parse_openai_response(response_text)
    if not review:
        status["status"] = "error"
        status["reason"] = "parse_failed"
        status["raw_response"] = response_text[:500]
        return status

    # Phase 4: save
    _save_review(review, ts, model=model, source=source)
    status["status"] = "ok"
    status["summary"] = review.get("summary", "")
    status["risk_score"] = review.get("risk_score", 0)
    status["n_suggestions"] = len(review.get("suggestions", []))
    status["warnings"] = review.get("warnings", [])
    status["suggestions"] = review.get("suggestions", [])
    status["review_output_path"] = str(REVIEW_OUTPUT)
    status["review_timestamp"] = float(ts)

    # Phase 5: optional apply
    if auto_apply_env and review.get("suggestions"):
        from research.auto_apply import backup_env

        backup_env()
        applied = _apply_env_suggestions(
            review["suggestions"],
            min_confidence=0.7,
            source="openai_auto_apply",
            reason="run_openai_review:auto_apply_env",
        )
        status["applied_env_changes"] = applied
        if applied:
            git_push_all(
                f"auto: OpenAI env changes — {len(applied)} params "
                f"({', '.join(a['key'] for a in applied)})"
            )

    # Phase 6: final push
    git_push_all(f"auto: OpenAI review complete — risk={review.get('risk_score', '?')}/10")

    return status
