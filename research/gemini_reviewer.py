"""
research/gemini_reviewer.py — Gemini API 코드 리뷰 자동화
===========================================================
1시간마다:
  1. git commit + push
  2. 핵심 전략 파일을 Gemini에 전송
  3. 구조화된 개선 제안 수신
  4. 문서화 (docs/GEMINI_REVIEW.md)
  5. 승인된 변경은 자동 적용 + git push

환경변수: GEMINI_API_KEY (없으면 비활성화)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("research.gemini_reviewer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_OUTPUT = PROJECT_ROOT / "docs" / "GEMINI_REVIEW.md"
REVIEW_HISTORY = PROJECT_ROOT / "state" / "gemini_review_history.json"

# Gemini API
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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

MAX_CHARS_PER_FILE = 15000  # Truncate large files
MAX_TOTAL_CHARS = 120000     # Total prompt size limit


def _get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or _read_env_key()


def _read_env_key() -> Optional[str]:
    env_path = PROJECT_ROOT / "state" / "bybit.env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("GEMINI_API_KEY="):
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


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    """Call Gemini API and return response text."""
    url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
    except URLError as e:
        logger.error(f"Gemini API error: {e}")
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
    return None


def _build_review_prompt(findings: list[dict], apply_history: list[dict]) -> str:
    """Build comprehensive review prompt for Gemini."""
    parts = []
    parts.append("""# Codex Quant 전략 코드 리뷰 요청

당신은 금융 공학(Financial Engineering) 전문가이며 Python/JAX 시스템 아키텍트입니다.
아래 코인 자동 매매 봇의 핵심 전략 코드를 분석하고, 구체적인 개선 제안을 해주세요.

## 출력 형식 (반드시 JSON)
```json
{
  "summary": "전체 분석 요약 (1-2문장)",
  "risk_score": 7,  // 1-10 (10=매우 위험)
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
            parts.append(f"- **{f.get('title')}**: PnL +${f.get('improvement_pct', 0):.2f}, "
                        f"신뢰도 {f.get('confidence', 0):.0%}, 파라미터: {f.get('param_changes')}")
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


def _parse_gemini_response(text: str) -> Optional[dict]:
    """Parse Gemini's JSON response."""
    if not text:
        return None
    # Try to extract JSON from response
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse Gemini response: {text[:200]}...")
    return None


def _save_review(review: dict, timestamp: float):
    """Save review to history and generate markdown."""
    # History
    history = []
    if REVIEW_HISTORY.exists():
        try:
            history = json.loads(REVIEW_HISTORY.read_text())
        except Exception:
            pass
    history.append({"timestamp": timestamp, "review": review})
    history = history[-50:]  # Keep last 50
    REVIEW_HISTORY.write_text(json.dumps(history, indent=2, default=str, ensure_ascii=False))

    # Markdown
    lines = [
        f"# Gemini Code Review — {time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))}",
        "",
        f"**Model:** {GEMINI_MODEL}",
        f"**Risk Score:** {review.get('risk_score', '?')}/10",
        f"**Summary:** {review.get('summary', '?')}",
        "",
        "## Suggestions",
        "",
    ]
    for s in review.get("suggestions", []):
        lines.append(f"### [{s.get('priority', '?')}] {s.get('title', '?')}")
        lines.append(f"**Category:** {s.get('category', '?')} | "
                     f"**Confidence:** {s.get('confidence', 0):.0%}")
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

    lines.append(f"\n---\n*Auto-generated by research/gemini_reviewer.py*")
    REVIEW_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Gemini review saved to {REVIEW_OUTPUT}")


def git_push_all(message: str = "auto: pre-Gemini review snapshot") -> bool:
    """Git add all, commit, and push."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT),
                       capture_output=True, timeout=10)
        result = subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "push"], cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"Git push: {message}")
            return True
        else:
            # Try with force if needed
            logger.warning(f"Git push failed, trying force: {result.stderr.decode()[:200]}")
            return False
    except Exception as e:
        logger.error(f"Git error: {e}")
        return False


def _apply_env_suggestions(suggestions: list[dict]) -> list[dict]:
    """Apply env variable changes from Gemini suggestions (safe ones only)."""
    from research.auto_apply import (
        _read_env_value, _update_env_value, _clamp_value,
        backup_env, PARAM_BOUNDS,
    )
    applied = []
    for s in suggestions:
        if s.get("confidence", 0) < 0.7:
            continue
        env_changes = s.get("env_changes", {})
        if not env_changes:
            continue
        for key, val in env_changes.items():
            # Safety: only apply if key is in our known bounds
            if key not in PARAM_BOUNDS:
                logger.info(f"Gemini suggestion skipped (unknown key): {key}={val}")
                continue
            try:
                fval = float(val)
                clamped = _clamp_value(key, fval)
                old = _read_env_value(key)
                if old is not None and abs(float(old) - clamped) < 1e-8:
                    continue
                logger.info(f"[GEMINI_APPLY] {key}: {old} → {clamped}")
                _update_env_value(key, str(clamped))
                applied.append({"key": key, "old": old, "new": str(clamped),
                               "source": s.get("title")})
            except (ValueError, TypeError):
                continue
    return applied


def run_gemini_review(
    findings: list[dict] | None = None,
    apply_history: list[dict] | None = None,
    auto_apply_env: bool = False,
) -> dict:
    """
    Run a complete Gemini review cycle.
    Returns status dict for dashboard.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"status": "disabled", "reason": "no_api_key"}

    ts = time.time()
    status = {"status": "running", "timestamp": ts, "model": GEMINI_MODEL}

    # Phase 1: Git push
    git_push_all(f"auto: pre-review snapshot {time.strftime('%Y-%m-%d %H:%M')}")

    # Phase 2: Build prompt and call Gemini
    prompt = _build_review_prompt(findings or [], apply_history or [])
    logger.info(f"Calling Gemini ({GEMINI_MODEL}), prompt={len(prompt)} chars...")

    response_text = _call_gemini(prompt, api_key)
    if not response_text:
        status["status"] = "error"
        status["reason"] = "api_call_failed"
        return status

    # Phase 3: Parse response
    review = _parse_gemini_response(response_text)
    if not review:
        status["status"] = "error"
        status["reason"] = "parse_failed"
        status["raw_response"] = response_text[:500]
        return status

    # Phase 4: Save review
    _save_review(review, ts)
    status["status"] = "ok"
    status["summary"] = review.get("summary", "")
    status["risk_score"] = review.get("risk_score", 0)
    status["n_suggestions"] = len(review.get("suggestions", []))
    status["warnings"] = review.get("warnings", [])

    # Phase 5: Auto-apply env suggestions (if enabled)
    if auto_apply_env and review.get("suggestions"):
        from research.auto_apply import backup_env
        backup_env()
        applied = _apply_env_suggestions(review["suggestions"])
        status["applied_env_changes"] = applied
        if applied:
            git_push_all(
                f"auto: Gemini env changes — {len(applied)} params "
                f"({', '.join(a['key'] for a in applied)})"
            )

    # Phase 6: Final git push with review doc
    git_push_all(f"auto: Gemini review complete — risk={review.get('risk_score', '?')}/10")

    return status
