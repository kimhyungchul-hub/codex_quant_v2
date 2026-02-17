#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_ANALYST_MODEL = "gemini-2.0-flash"
DEFAULT_CODER_MODEL = "gemini-2.0-flash"
DEFAULT_VALIDATOR_CMD = "python scripts/cf_validator.py --oos-ratio 0.2"
DEFAULT_TARGET_FILE = "engines/mc_risk.py"
ALLOWED_TARGET_PREFIXES = ("engines/", "core/", "research/", "utils/", "models/")

ANALYST_PROMPT_TEMPLATE = """
당신은 세계 최고 수준의 퀀트 트레이딩 펌에서 근무하는 '시니어 퀀트 리서처'입니다.
최근 우리 트레이딩 봇의 운영 데이터와 에러/손실 로그를 분석하여 로직 개선 가설을 세워야 합니다.

[최근 운영 요약]
- 기간: 최근 7일(가능한 범위에서 산출)
- 승률: {win_rate:.2f}%
- Sharpe Ratio(근사): {sharpe_ratio:.4f}
- 최대 낙폭(MDD): {mdd_pct:.2f}%
- 주요 손실 원인 로그:
{recent_loss_logs}

[지시사항]
이 데이터를 바탕으로 현재 매매 엔진의 약점을 진단하고 "코드 수정 가능한 가설 3개"를 제시하세요.
모호한 조언이 아니라, 어떤 파일의 어떤 로직을 어떻게 바꿀지 구체적으로 작성하세요.
우선순위가 가장 높은 1개를 반드시 선택하세요.

[중요 출력 규칙]
반드시 아래 JSON 객체 하나만 출력하세요. 마크다운, 코드블록, 설명 문장 금지.
{
  "hypotheses": [
    {
      "id": "H1",
      "title": "짧은 제목",
      "diagnosis": "문제 진단",
      "target_file_hint": "{target_file_hint}",
      "code_change": "수정 로직 요약",
      "expected_effect": "예상 성과 영향"
    },
    {
      "id": "H2",
      "title": "짧은 제목",
      "diagnosis": "문제 진단",
      "target_file_hint": "{target_file_hint}",
      "code_change": "수정 로직 요약",
      "expected_effect": "예상 성과 영향"
    },
    {
      "id": "H3",
      "title": "짧은 제목",
      "diagnosis": "문제 진단",
      "target_file_hint": "{target_file_hint}",
      "code_change": "수정 로직 요약",
      "expected_effect": "예상 성과 영향"
    }
  ],
  "priority_pick": "H1",
  "selection_reason": "왜 이 가설이 우선인지 근거"
}
""".strip()

CODER_PROMPT_TEMPLATE = """
당신은 기관급 퀀트 시스템을 개발하는 '시니어 파이썬/금융 공학 엔지니어'입니다.
퀀트 리서처가 다음과 같은 개선 가설을 제시했습니다.

[리서처의 가설]
{hypothesis}

[수정해야 할 원본 코드: {target_file}]
```python
{original_code}
```

[지시사항]
위 가설을 적용하여 원본 코드를 수정하세요.
문법적 오류가 없는 완전한 Python 코드를 작성해야 합니다.
기존 시스템 아키텍처(함수 시그니처, 의존성, 외부 호출 계약)를 절대 파괴하지 마세요.
응답은 반드시 수정된 전체 코드 내용만을 python 코드 블록 하나로 반환하세요.
코드 블록 외부 텍스트(설명/사족)는 절대 출력하지 마세요.
""".strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout_sec: Optional[float] = None,
    check: bool = True,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=shell,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {cmd}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")
    return cp


def _git(cmd: list[str], repo_root: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *cmd], cwd=repo_root, check=check)


def _git_output(cmd: list[str], repo_root: Path) -> str:
    return _git(cmd, repo_root).stdout.strip()


def _get_api_key(repo_root: Path) -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    env_path = repo_root / "state" / "bybit.env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("GEMINI_API_KEY="):
                return s.split("=", 1)[1].strip()
    except Exception:
        return None
    return None


def _call_gemini(
    *,
    model: str,
    prompt: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: float = 120.0,
) -> str:
    url = GEMINI_API_URL.format(model=model) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    data: Optional[dict[str, Any]] = None
    last_err = ""
    for attempt in range(1, 4):
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            raise RuntimeError(f"Gemini request failed: {e}") from e
        except URLError as e:
            last_err = str(e)
            if attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            raise RuntimeError(f"Gemini request failed: {e}") from e
    if data is None:
        raise RuntimeError(f"Gemini request failed after retries: {last_err}")

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini returned empty parts: {data}")
    text = "".join(str(p.get("text", "")) for p in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned empty text")
    return text


def _extract_python_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    m = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    found: Optional[dict[str, Any]] = None
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            found = obj
    if found is not None:
        return found
    raise ValueError("failed to parse JSON object")


def _normalize_stage1_hypotheses(obj: dict[str, Any]) -> dict[str, Any]:
    rows = obj.get("hypotheses")
    if not isinstance(rows, list):
        raise ValueError("stage1 response missing 'hypotheses' list")
    if len(rows) < 3:
        raise ValueError("stage1 requires at least 3 hypotheses")

    normalized: list[dict[str, str]] = []
    for i, item in enumerate(rows[:3], start=1):
        if not isinstance(item, dict):
            item = {"title": str(item)}
        hid = str(item.get("id") or f"H{i}").strip() or f"H{i}"
        title = str(item.get("title") or f"H{i} hypothesis").strip() or f"H{i} hypothesis"
        diagnosis = str(item.get("diagnosis") or item.get("rationale") or item.get("problem") or "").strip()
        code_change = str(item.get("code_change") or item.get("change_plan") or item.get("proposal") or "").strip()
        target_hint = str(item.get("target_file_hint") or item.get("target_file") or "").strip()
        expected = str(item.get("expected_effect") or "").strip()
        normalized.append(
            {
                "id": hid,
                "title": title,
                "diagnosis": diagnosis,
                "code_change": code_change,
                "target_file_hint": target_hint,
                "expected_effect": expected,
            }
        )

    priority_pick = str(obj.get("priority_pick") or "").strip()
    selection_reason = str(obj.get("selection_reason") or "").strip()

    selected = normalized[0]
    if priority_pick:
        for row in normalized:
            if row["id"] == priority_pick:
                selected = row
                break
    if selected is normalized[0] and priority_pick:
        # allow 1-based index fallback
        try:
            idx = int(priority_pick) - 1
            if 0 <= idx < len(normalized):
                selected = normalized[idx]
        except Exception:
            pass

    if not selection_reason:
        selection_reason = "priority_pick 기준으로 선택"

    return {
        "hypotheses": normalized,
        "priority_pick": selected["id"],
        "selection_reason": selection_reason,
        "selected": selected,
    }


def _render_selected_hypothesis(stage1_pack: dict[str, Any]) -> str:
    selected = stage1_pack.get("selected") if isinstance(stage1_pack.get("selected"), dict) else {}
    pick = str(stage1_pack.get("priority_pick") or selected.get("id") or "H1")
    reason = str(stage1_pack.get("selection_reason") or "")
    return (
        f"[선택 가설: {pick}] {selected.get('title')}\n"
        f"- diagnosis: {selected.get('diagnosis')}\n"
        f"- target_file_hint: {selected.get('target_file_hint')}\n"
        f"- code_change: {selected.get('code_change')}\n"
        f"- expected_effect: {selected.get('expected_effect')}\n"
        f"- selection_reason: {reason}"
    ).strip()


def _build_retry_hypothesis_context(
    *,
    base_hypothesis: str,
    attempt_idx: int,
    attempt_error: str = "",
    eval_payload: Optional[dict[str, Any]] = None,
) -> str:
    lines = [
        base_hypothesis.strip(),
        "",
        f"[재시도 컨텍스트] attempt={attempt_idx}",
    ]
    if attempt_error:
        lines.append(f"- 이전 시도 에러: {attempt_error}")
    if isinstance(eval_payload, dict):
        delta = eval_payload.get("delta") if isinstance(eval_payload.get("delta"), dict) else {}
        gate = {
            "strict_gain_pass": eval_payload.get("strict_gain_pass"),
            "oos_gate_pass": eval_payload.get("oos_gate_pass"),
            "improved": eval_payload.get("improved"),
        }
        lines.append(f"- 이전 검증 delta: {json.dumps(delta, ensure_ascii=False)}")
        lines.append(f"- 이전 검증 gate: {json.dumps(gate, ensure_ascii=False)}")
    lines.append(
        "- 요청: 기존 아키텍처를 유지한 채, 위 실패 원인을 반영해 더 보수적이고 검증 통과 가능성이 높은 수정안을 다시 작성하세요."
    )
    return "\n".join(lines).strip()


def _resolve_target(repo_root: Path, target_file: str, allow_any_target: bool) -> Path:
    target = (repo_root / target_file).resolve()
    if not str(target).startswith(str(repo_root.resolve()) + os.sep):
        raise ValueError(f"target_file is outside repository: {target_file}")
    if not target.exists():
        raise FileNotFoundError(f"target_file not found: {target_file}")
    rel = target.relative_to(repo_root).as_posix()
    if not allow_any_target:
        if not rel.endswith(".py"):
            raise ValueError(f"target_file must be a Python file: {rel}")
        if not rel.startswith(ALLOWED_TARGET_PREFIXES):
            raise ValueError(
                f"target_file '{rel}' is not in allowed prefixes {ALLOWED_TARGET_PREFIXES}. "
                "Use --allow-any-target to override."
            )
    return target


def _load_alpha_report(repo_root: Path) -> dict[str, Any]:
    candidates = [
        repo_root / "state" / "alpha_pipeline_report_latest.json",
        repo_root / "state" / "alpha_pipeline_report_now.json",
        repo_root / "state" / "alpha_pipeline_report.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = _read_json(p)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


def _compute_sharpe_proxy(repo_root: Path, recent_n: int = 2000) -> float:
    p = repo_root / "state" / "eval_history_live.json"
    if not p.exists():
        return 0.0
    try:
        rows = _read_json(p)
    except Exception:
        return 0.0
    if not isinstance(rows, list):
        return 0.0
    vals: list[float] = []
    for r in rows[-max(int(recent_n), 1) :]:
        if not isinstance(r, dict):
            continue
        v = r.get("realized_r")
        try:
            if v is None:
                continue
            vals.append(float(v))
        except Exception:
            continue
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)
    std = math.sqrt(max(var, 1e-12))
    return float((mean / std) * math.sqrt(len(vals)))


def _compute_mdd_percent(repo_root: Path, lookback_days: int = 7) -> float:
    p = repo_root / "state" / "equity_history_live.json"
    if not p.exists():
        return 0.0
    try:
        rows = _read_json(p)
    except Exception:
        return 0.0
    if not isinstance(rows, list) or not rows:
        return 0.0

    now_ms = int(time.time() * 1000)
    cutoff = now_ms - max(lookback_days, 1) * 24 * 60 * 60 * 1000
    filtered: list[tuple[int, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            t = int(r.get("time"))
            eq = float(r.get("equity"))
        except Exception:
            continue
        if t >= cutoff:
            filtered.append((t, eq))
    if len(filtered) < 2:
        filtered = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                t = int(r.get("time"))
                eq = float(r.get("equity"))
            except Exception:
                continue
            filtered.append((t, eq))
    if len(filtered) < 2:
        return 0.0
    filtered.sort(key=lambda x: x[0])
    peak = filtered[0][1]
    mdd = 0.0
    for _, eq in filtered:
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return float(mdd * 100.0)


def _summarize_loss_logs(repo_root: Path, limit: int = 10) -> str:
    path = repo_root / "state" / "trade_tape_live.json"
    if not path.exists():
        return "- (trade_tape_live.json 없음)"
    try:
        rows = _read_json(path)
    except Exception as e:
        return f"- (trade_tape_live.json 읽기 실패: {e})"
    if not isinstance(rows, list):
        return "- (trade_tape_live.json 형식 오류)"

    out: list[str] = []
    for r in reversed(rows):
        if not isinstance(r, dict):
            continue
        ttype = str(r.get("ttype") or r.get("type") or "").upper()
        if ttype not in {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}:
            continue
        rr: Optional[float]
        rr_raw = r.get("realized_r")
        rr = None
        if rr_raw is not None:
            try:
                rr = float(rr_raw)
            except Exception:
                rr = None
        if rr is None:
            pnl = r.get("pnl")
            notional = r.get("notional")
            lev = r.get("leverage")
            try:
                if pnl is not None and notional and lev:
                    base = float(notional) / max(float(lev), 1e-9)
                    rr = float(pnl) / max(base, 1e-9)
            except Exception:
                rr = None
        if rr is None or rr >= 0:
            continue
        symbol = str(r.get("symbol") or "?")
        side = str(r.get("side") or "?")
        reason = str(r.get("reason") or "unknown")
        hold = r.get("hold_duration_sec")
        hold_txt = ""
        try:
            if hold is not None:
                hold_txt = f", hold={float(hold):.1f}s"
        except Exception:
            hold_txt = ""
        out.append(f"- reason={reason}, symbol={symbol}, side={side}, realized_r={rr:.5f}{hold_txt}")
        if len(out) >= max(int(limit), 1):
            break
    if not out:
        return "- (최근 손실 로그를 찾지 못함)"
    out.reverse()
    return "\n".join(out)


@dataclass
class EvalMetric:
    closed_trades: int
    win_rate: float
    avg_metric: float
    sharpe: float
    max_dd: float
    oos_win_rate: Optional[float] = None
    oos_avg_metric: Optional[float] = None
    oos_sharpe: Optional[float] = None
    oos_max_dd: Optional[float] = None
    source: str = ""


def _extract_eval_metric(report: dict[str, Any]) -> EvalMetric:
    if not isinstance(report, dict):
        raise ValueError("validator report must be a JSON object")

    # Format A: scripts/evaluate_alpha_pipeline.py
    perf = report.get("performance")
    if isinstance(perf, dict):
        overall = perf.get("overall")
        if isinstance(overall, dict):
            return EvalMetric(
                closed_trades=int(perf.get("closed_trades", 0) or 0),
                win_rate=float(overall.get("win_rate", 0.0) or 0.0),
                avg_metric=float(overall.get("avg_realized_r", 0.0) or 0.0),
                sharpe=float(overall.get("sharpe", 0.0) or 0.0),
                max_dd=float(overall.get("max_dd", 0.0) or 0.0),
                source="evaluate_alpha_pipeline",
            )

    # Format B: scripts/cf_validator.py
    overall = report.get("overall")
    if isinstance(overall, dict):
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        oos = report.get("out_of_sample") if isinstance(report.get("out_of_sample"), dict) else {}

        def _pick_float(srcs: list[dict[str, Any]], keys: list[str], default: Optional[float] = 0.0) -> Optional[float]:
            for src in srcs:
                for k in keys:
                    if k in src and src.get(k) is not None:
                        try:
                            return float(src.get(k))
                        except Exception:
                            continue
            return default

        return EvalMetric(
            closed_trades=int(overall.get("n", report.get("n_trades", 0)) or 0),
            win_rate=float(_pick_float([summary, overall], ["wr", "win_rate"], 0.0) or 0.0),
            avg_metric=float(_pick_float([summary, overall], ["avg_pnl", "avg_realized_r"], 0.0) or 0.0),
            sharpe=float(_pick_float([summary, overall], ["sharpe"], 0.0) or 0.0),
            max_dd=float(_pick_float([summary, overall], ["max_dd"], 0.0) or 0.0),
            oos_win_rate=_pick_float([summary, oos], ["oos_wr", "wr", "win_rate"], None),
            oos_avg_metric=_pick_float([summary, oos], ["oos_avg_pnl", "avg_pnl", "avg_realized_r"], None),
            oos_sharpe=_pick_float([summary, oos], ["oos_sharpe", "sharpe"], None),
            oos_max_dd=_pick_float([summary, oos], ["oos_max_dd", "max_dd"], None),
            source=str(report.get("source") or "cf_engine_metrics"),
        )

    # Format C: research.runner.run_cf_cycle() raw output
    baseline = report.get("baseline")
    if isinstance(baseline, dict):
        return EvalMetric(
            closed_trades=int(baseline.get("n", 0) or 0),
            win_rate=float(baseline.get("wr", 0.0) or 0.0),
            avg_metric=float(baseline.get("avg_pnl", 0.0) or 0.0),
            sharpe=float(baseline.get("sharpe", 0.0) or 0.0),
            max_dd=float(baseline.get("max_dd", 0.0) or 0.0),
            source="research.runner.run_cf_cycle",
        )

    raise ValueError("validator report format not recognized")


def _run_validator_command(repo_root: Path, cmd: str, timeout_sec: float) -> dict[str, Any]:
    cp = _run(cmd, cwd=repo_root, shell=True, timeout_sec=timeout_sec, check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"validator command failed ({cp.returncode}): {cmd}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    stdout = cp.stdout.strip()
    if not stdout:
        raise RuntimeError("validator command produced empty output")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # Some validators print logs before JSON. Scan for the last decodable JSON object.
        decoder = json.JSONDecoder()
        parsed: Optional[dict[str, Any]] = None
        for i, ch in enumerate(stdout):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(stdout[i:])
            except Exception:
                continue
            if isinstance(obj, dict):
                parsed = obj
        if parsed is not None:
            return parsed
        raise RuntimeError("validator output is not valid JSON")


def _send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def _send_slack(text: str, webhook_url: str) -> bool:
    if not webhook_url.strip():
        return False
    body = json.dumps({"text": text}).encode("utf-8")
    req = Request(webhook_url.strip(), data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def _validate_generated_code(code: str, filename: str) -> None:
    try:
        compile(code, filename, "exec")
    except SyntaxError as e:
        raise RuntimeError(f"generated code has syntax error: {e}") from e


def _build_stage1_context(repo_root: Path) -> dict[str, Any]:
    alpha_report = _load_alpha_report(repo_root)
    perf = alpha_report.get("performance", {}) if isinstance(alpha_report, dict) else {}
    overall = perf.get("overall", {}) if isinstance(perf, dict) else {}
    win_rate = float(overall.get("win_rate", 0.0) or 0.0) * 100.0
    sharpe_ratio = _compute_sharpe_proxy(repo_root)
    mdd_pct = _compute_mdd_percent(repo_root, lookback_days=7)
    recent_loss_logs = _summarize_loss_logs(repo_root, limit=10)
    return {
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "mdd_pct": mdd_pct,
        "recent_loss_logs": recent_loss_logs,
    }


def _generate_stage1_pack(
    *,
    repo_root: Path,
    api_key: str,
    model: str,
    temperature: float,
    target_file_hint: str,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    stage1_ctx = _build_stage1_context(repo_root)
    prompt = ANALYST_PROMPT_TEMPLATE
    prompt = prompt.replace("{win_rate:.2f}", f"{float(stage1_ctx.get('win_rate', 0.0)):.2f}")
    prompt = prompt.replace("{sharpe_ratio:.4f}", f"{float(stage1_ctx.get('sharpe_ratio', 0.0)):.4f}")
    prompt = prompt.replace("{mdd_pct:.2f}", f"{float(stage1_ctx.get('mdd_pct', 0.0)):.2f}")
    prompt = prompt.replace("{recent_loss_logs}", str(stage1_ctx.get("recent_loss_logs", "")))
    prompt = prompt.replace("{target_file_hint}", str(target_file_hint))
    last_err = ""
    for _ in range(max(1, int(max_attempts))):
        raw = _call_gemini(
            model=model,
            prompt=prompt,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=3072,
        )
        try:
            obj = _extract_json_object(raw)
            pack = _normalize_stage1_hypotheses(obj)
            return stage1_ctx, pack, raw
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"failed to build Stage1 hypotheses pack: {last_err}")


def _evaluate_candidate_gate(
    *,
    baseline_metric: EvalMetric,
    candidate_metric: EvalMetric,
    min_winrate_delta: float,
    min_avg_metric_delta: float,
    min_sharpe_delta: float,
    min_maxdd_delta: float,
    min_oos_sharpe_delta: float,
    min_oos_maxdd_delta: float,
) -> dict[str, Any]:
    win_delta = candidate_metric.win_rate - baseline_metric.win_rate
    avg_metric_delta = candidate_metric.avg_metric - baseline_metric.avg_metric
    sharpe_delta = candidate_metric.sharpe - baseline_metric.sharpe
    max_dd_delta = candidate_metric.max_dd - baseline_metric.max_dd

    oos_sharpe_delta: Optional[float] = None
    oos_max_dd_delta: Optional[float] = None
    if (candidate_metric.oos_sharpe is not None) and (baseline_metric.oos_sharpe is not None):
        oos_sharpe_delta = float(candidate_metric.oos_sharpe - baseline_metric.oos_sharpe)
    if (candidate_metric.oos_max_dd is not None) and (baseline_metric.oos_max_dd is not None):
        oos_max_dd_delta = float(candidate_metric.oos_max_dd - baseline_metric.oos_max_dd)

    oos_data_available = bool((oos_sharpe_delta is not None) and (oos_max_dd_delta is not None))
    oos_threshold_active = bool(
        oos_data_available
        or abs(float(min_oos_sharpe_delta)) > 0.0
        or abs(float(min_oos_maxdd_delta)) > 0.0
    )
    if oos_threshold_active:
        oos_pass = (
            (oos_sharpe_delta is not None)
            and (oos_max_dd_delta is not None)
            and (oos_sharpe_delta >= float(min_oos_sharpe_delta))
            and (oos_max_dd_delta >= float(min_oos_maxdd_delta))
        )
    else:
        oos_pass = True

    strict_gain = bool(
        (win_delta > 0.0)
        or (avg_metric_delta > 0.0)
        or (sharpe_delta > 0.0)
        or (max_dd_delta > 0.0)
    )
    improved = (
        (win_delta >= float(min_winrate_delta))
        and (avg_metric_delta >= float(min_avg_metric_delta))
        and (sharpe_delta >= float(min_sharpe_delta))
        and (max_dd_delta >= float(min_maxdd_delta))
        and bool(strict_gain)
        and bool(oos_pass)
    )
    return {
        "delta": {
            "win_rate": win_delta,
            "avg_metric": avg_metric_delta,
            "sharpe": sharpe_delta,
            "max_dd": max_dd_delta,
            "oos_sharpe": oos_sharpe_delta,
            "oos_max_dd": oos_max_dd_delta,
        },
        "oos_gate_active": bool(oos_threshold_active),
        "oos_data_available": bool(oos_data_available),
        "oos_gate_pass": bool(oos_pass),
        "strict_gain_required": True,
        "strict_gain_pass": bool(strict_gain),
        "improved": bool(improved),
    }


def _create_branch(repo_root: Path, branch_name: str) -> None:
    _git(["checkout", "-b", branch_name], repo_root)


def _checkout_branch(repo_root: Path, branch: str) -> None:
    _git(["checkout", branch], repo_root)


def _delete_branch(repo_root: Path, branch: str) -> None:
    _git(["branch", "-D", branch], repo_root, check=False)


def _create_pr(repo_root: Path, base_branch: str, head_branch: str, title: str, body: str) -> tuple[bool, str]:
    if shutil.which("gh") is None:
        return False, "gh CLI not found"
    cp = _run(
        [
            "gh",
            "pr",
            "create",
            "--base",
            base_branch,
            "--head",
            head_branch,
            "--title",
            title,
            "--body",
            body,
        ],
        cwd=repo_root,
        check=False,
    )
    if cp.returncode != 0:
        return False, cp.stderr.strip() or cp.stdout.strip() or "gh pr create failed"
    return True, cp.stdout.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Quant Agent (Stage 1~4 with strict sandbox + human approval gate)")
    p.add_argument("--target-file", default=DEFAULT_TARGET_FILE, help="Target logic file to modify")
    p.add_argument("--allow-any-target", action="store_true", help="Allow target file outside default prefixes")
    p.add_argument("--branch-prefix", default="agent-fix", help="Isolated git branch prefix")
    p.add_argument("--analyst-model", default=DEFAULT_ANALYST_MODEL)
    p.add_argument("--coder-model", default=DEFAULT_CODER_MODEL)
    p.add_argument("--analyst-temperature", type=float, default=0.7)
    p.add_argument("--coder-temperature", type=float, default=0.1)
    p.add_argument("--max-target-chars", type=int, default=40000)
    p.add_argument(
        "--validator-cmd",
        default=DEFAULT_VALIDATOR_CMD,
        help="Must print JSON report to stdout (cf_validator or evaluate_alpha_pipeline compatible)",
    )
    p.add_argument("--validator-timeout-sec", type=float, default=300.0)
    p.add_argument("--min-winrate-delta", type=float, default=0.0, help="Candidate - baseline threshold")
    p.add_argument(
        "--min-avg-r-delta",
        type=float,
        default=0.0,
        help="Candidate - baseline threshold (legacy name; now interpreted as avg metric delta)",
    )
    p.add_argument("--min-avg-metric-delta", type=float, default=None, help="Candidate - baseline avg metric threshold")
    p.add_argument("--min-sharpe-delta", type=float, default=0.0, help="Candidate - baseline Sharpe threshold")
    p.add_argument(
        "--min-maxdd-delta",
        type=float,
        default=0.0,
        help="Candidate - baseline max_dd threshold (higher is better; closer to 0 means lower drawdown)",
    )
    p.add_argument("--min-oos-sharpe-delta", type=float, default=0.0, help="Candidate - baseline OOS Sharpe threshold")
    p.add_argument(
        "--min-oos-maxdd-delta",
        type=float,
        default=0.0,
        help="Candidate - baseline OOS max_dd threshold (higher is better)",
    )
    p.add_argument("--allow-dirty", action="store_true", help="Skip clean working tree check")
    p.add_argument("--skip-push", action="store_true", help="Do not push branch to origin")
    p.add_argument("--skip-pr", action="store_true", help="Do not create GitHub PR")
    p.add_argument("--keep-failed-branch", action="store_true", help="Do not delete failed branch")
    p.add_argument(
        "--max-recode-retries",
        type=int,
        default=2,
        help="Retry code generation+validation up to N times after failures (total attempts = 1+N)",
    )
    p.add_argument("--dry-run", action="store_true", help="Run Stage 1 only and exit")
    p.add_argument("--out-dir", default="state/autonomous_agent", help="Run artifact output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], Path.cwd()))
    os.chdir(repo_root)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{run_id}.json"

    result: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _now_iso(),
        "repo_root": str(repo_root),
        "config": vars(args),
        "status": "started",
    }
    _write_json(out_path, result)

    base_branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    result["base_branch"] = base_branch

    try:
        if not args.allow_dirty:
            dirty = _git_output(["status", "--porcelain"], repo_root)
            if dirty:
                raise RuntimeError("working tree is dirty. Commit/stash first, or use --allow-dirty.")

        target_path = _resolve_target(repo_root, args.target_file, args.allow_any_target)
        target_rel = target_path.relative_to(repo_root).as_posix()
        result["target_file"] = target_rel

        api_key = _get_api_key(repo_root)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found (env or state/bybit.env)")

        print("\n=== Stage 1: The Analyst (3 Hypotheses + Priority Pick) ===")
        stage1_ctx, stage1_pack, stage1_raw = _generate_stage1_pack(
            repo_root=repo_root,
            api_key=api_key,
            model=args.analyst_model,
            temperature=args.analyst_temperature,
            target_file_hint=target_rel,
            max_attempts=2,
        )
        selected_hypothesis = _render_selected_hypothesis(stage1_pack)
        result["stage1"] = {
            "context": stage1_ctx,
            "hypotheses": stage1_pack.get("hypotheses"),
            "priority_pick": stage1_pack.get("priority_pick"),
            "selection_reason": stage1_pack.get("selection_reason"),
            "selected": stage1_pack.get("selected"),
            "raw_response": stage1_raw,
        }
        _write_json(out_path, result)

        if args.dry_run:
            result["status"] = "dry_run_complete"
            result["finished_at"] = _now_iso()
            _write_json(out_path, result)
            print("Dry run complete. Stage 1 hypothesis saved:", out_path)
            return 0

        min_avg_metric_delta = (
            float(args.min_avg_metric_delta)
            if args.min_avg_metric_delta is not None
            else float(args.min_avg_r_delta)
        )

        print("\n=== Stage 3(baseline): Validator on base branch ===")
        baseline_report = _run_validator_command(repo_root, args.validator_cmd, args.validator_timeout_sec)
        baseline_metric = _extract_eval_metric(baseline_report)
        result["baseline"] = {
            "metric": baseline_metric.__dict__,
            "report": baseline_report,
        }
        _write_json(out_path, result)

        print("\n=== Stage 2: The Coder (Branch + Iterative Code Generation) ===")
        branch_name = f"{args.branch_prefix}-{run_id}"
        _create_branch(repo_root, branch_name)
        result["agent_branch"] = branch_name
        _write_json(out_path, result)

        initial_code = target_path.read_text(encoding="utf-8")
        if len(initial_code) > int(args.max_target_chars):
            raise RuntimeError(
                f"target file too large ({len(initial_code)} chars). "
                f"Limit={args.max_target_chars}. Scope down target file."
            )

        total_attempts = max(1, 1 + int(args.max_recode_retries))
        attempts: list[dict[str, Any]] = []
        retry_hypothesis_context = selected_hypothesis
        final_candidate_report: Optional[dict[str, Any]] = None
        final_candidate_metric: Optional[EvalMetric] = None
        final_eval_gate: Optional[dict[str, Any]] = None
        commits_made = 0

        for attempt_idx in range(1, total_attempts + 1):
            print(f"\n=== Stage 2/3 attempt {attempt_idx}/{total_attempts}: Coder + Validator ===")
            attempt_row: dict[str, Any] = {
                "attempt": attempt_idx,
                "hypothesis_context": retry_hypothesis_context,
            }
            pre_attempt_code = target_path.read_text(encoding="utf-8")
            if len(pre_attempt_code) > int(args.max_target_chars):
                raise RuntimeError(
                    f"target file too large ({len(pre_attempt_code)} chars). "
                    f"Limit={args.max_target_chars}. Scope down target file."
                )

            try:
                coder_prompt = CODER_PROMPT_TEMPLATE.format(
                    hypothesis=retry_hypothesis_context.strip(),
                    target_file=target_rel,
                    original_code=pre_attempt_code,
                )
                coder_resp = _call_gemini(
                    model=args.coder_model,
                    prompt=coder_prompt,
                    api_key=api_key,
                    temperature=args.coder_temperature,
                    max_output_tokens=8192,
                )
                new_code = _extract_python_code(coder_resp)
                _validate_generated_code(new_code, target_rel)
                target_path.write_text(new_code, encoding="utf-8")
                _run([sys.executable, "-m", "py_compile", target_rel], cwd=repo_root, check=True)
            except Exception as e:
                target_path.write_text(pre_attempt_code, encoding="utf-8")
                attempt_row["status"] = "codegen_failed"
                attempt_row["error"] = str(e)
                attempts.append(attempt_row)
                retry_hypothesis_context = _build_retry_hypothesis_context(
                    base_hypothesis=selected_hypothesis,
                    attempt_idx=attempt_idx,
                    attempt_error=str(e),
                )
                _write_json(out_path, {**result, "attempts": attempts})
                continue

            _git(["add", target_rel], repo_root)
            diff_cached = _git_output(["diff", "--cached", "--name-only"], repo_root)
            if not diff_cached.strip():
                attempt_row["status"] = "no_diff"
                attempt_row["error"] = "generated code produced no git diff"
                attempts.append(attempt_row)
                retry_hypothesis_context = _build_retry_hypothesis_context(
                    base_hypothesis=selected_hypothesis,
                    attempt_idx=attempt_idx,
                    attempt_error="generated code produced no git diff",
                )
                _write_json(out_path, {**result, "attempts": attempts})
                continue

            commit_msg = f"Auto-Agent: attempt {attempt_idx}/{total_attempts} apply hypothesis to {target_rel}"
            _git(["commit", "-m", commit_msg], repo_root)
            commits_made += 1
            attempt_row["commit_message"] = commit_msg

            try:
                candidate_report = _run_validator_command(repo_root, args.validator_cmd, args.validator_timeout_sec)
                candidate_metric = _extract_eval_metric(candidate_report)
                eval_gate = _evaluate_candidate_gate(
                    baseline_metric=baseline_metric,
                    candidate_metric=candidate_metric,
                    min_winrate_delta=float(args.min_winrate_delta),
                    min_avg_metric_delta=float(min_avg_metric_delta),
                    min_sharpe_delta=float(args.min_sharpe_delta),
                    min_maxdd_delta=float(args.min_maxdd_delta),
                    min_oos_sharpe_delta=float(args.min_oos_sharpe_delta),
                    min_oos_maxdd_delta=float(args.min_oos_maxdd_delta),
                )
                attempt_row["status"] = "improved" if bool(eval_gate.get("improved")) else "validated_not_improved"
                attempt_row["candidate_metric"] = candidate_metric.__dict__
                attempt_row["evaluation"] = eval_gate
                attempts.append(attempt_row)

                final_candidate_report = candidate_report
                final_candidate_metric = candidate_metric
                final_eval_gate = eval_gate

                _write_json(out_path, {**result, "attempts": attempts})
                if bool(eval_gate.get("improved")):
                    break

                retry_hypothesis_context = _build_retry_hypothesis_context(
                    base_hypothesis=selected_hypothesis,
                    attempt_idx=attempt_idx,
                    eval_payload=eval_gate,
                )
            except Exception as e:
                attempt_row["status"] = "validator_failed"
                attempt_row["error"] = str(e)
                attempts.append(attempt_row)
                retry_hypothesis_context = _build_retry_hypothesis_context(
                    base_hypothesis=selected_hypothesis,
                    attempt_idx=attempt_idx,
                    attempt_error=str(e),
                )
                _write_json(out_path, {**result, "attempts": attempts})
                continue

        improved = bool(final_eval_gate.get("improved")) if isinstance(final_eval_gate, dict) else False

        stage3_payload: dict[str, Any] = {
            "baseline_metric": baseline_metric.__dict__,
            "threshold": {
                "min_winrate_delta": float(args.min_winrate_delta),
                "min_avg_metric_delta": float(min_avg_metric_delta),
                "min_sharpe_delta": float(args.min_sharpe_delta),
                "min_maxdd_delta": float(args.min_maxdd_delta),
                "min_oos_sharpe_delta": float(args.min_oos_sharpe_delta),
                "min_oos_maxdd_delta": float(args.min_oos_maxdd_delta),
            },
            "attempts_total": int(total_attempts),
            "attempts_used": int(len(attempts)),
            "attempts": attempts,
            "improved": bool(improved),
        }
        if final_candidate_metric is not None and final_eval_gate is not None:
            stage3_payload.update(
                {
                    "candidate_metric": final_candidate_metric.__dict__,
                    "delta": final_eval_gate.get("delta"),
                    "oos_gate_active": final_eval_gate.get("oos_gate_active"),
                    "oos_data_available": final_eval_gate.get("oos_data_available"),
                    "oos_gate_pass": final_eval_gate.get("oos_gate_pass"),
                    "strict_gain_required": final_eval_gate.get("strict_gain_required"),
                    "strict_gain_pass": final_eval_gate.get("strict_gain_pass"),
                    "candidate_report": final_candidate_report,
                }
            )
        else:
            stage3_payload.update(
                {
                    "candidate_metric": None,
                    "delta": None,
                    "oos_gate_active": False,
                    "oos_data_available": False,
                    "oos_gate_pass": False,
                    "strict_gain_required": True,
                    "strict_gain_pass": False,
                    "candidate_report": None,
                    "error": "all attempts failed before successful validation",
                }
            )

        result["stage2"] = {
            "target_file": target_rel,
            "total_attempts": int(total_attempts),
            "commits_made": int(commits_made),
            "selected_hypothesis_id": stage1_pack.get("priority_pick"),
        }
        result["stage3"] = stage3_payload
        _write_json(out_path, result)

        print("\n=== Stage 4: The Manager (Human Approval Gate) ===")
        push_done = False
        pr_status: dict[str, Any] = {"created": False}
        alerts = {"telegram": False, "slack": False}

        if improved:
            if not args.skip_push:
                _run(["git", "push", "-u", "origin", branch_name], cwd=repo_root, check=True)
                push_done = True

            if not args.skip_pr:
                title = f"[Auto-Agent] Candidate fix: {target_rel}"
                body = (
                    "Automated quant-agent candidate update.\n\n"
                    "Safety gates passed in sandbox validator.\n"
                    "No auto-merge performed. Human review is required."
                )
                ok, detail = _create_pr(repo_root, base_branch, branch_name, title, body)
                pr_status = {"created": ok, "detail": detail}

            delta_obj = result.get("stage3", {}).get("delta") if isinstance(result.get("stage3"), dict) else {}
            win_delta = float((delta_obj or {}).get("win_rate", 0.0) or 0.0)
            avg_metric_delta = float((delta_obj or {}).get("avg_metric", 0.0) or 0.0)
            sharpe_delta = float((delta_obj or {}).get("sharpe", 0.0) or 0.0)
            max_dd_delta = float((delta_obj or {}).get("max_dd", 0.0) or 0.0)
            msg = (
                "[Auto-Agent] Validation passed.\n"
                f"branch={branch_name}\n"
                f"target={target_rel}\n"
                f"delta_win_rate={win_delta:.6f}\n"
                f"delta_avg_metric={avg_metric_delta:.6f}\n"
                f"delta_sharpe={sharpe_delta:.6f}\n"
                f"delta_max_dd={max_dd_delta:.6f}\n"
                "Manual review + approval required before merge."
            )
            alerts["telegram"] = _send_telegram(msg)
            slack_url = os.environ.get("SLACK_WEBHOOK_URL", "")
            alerts["slack"] = _send_slack(msg, slack_url) if slack_url else False
        else:
            if not args.keep_failed_branch:
                _checkout_branch(repo_root, base_branch)
                _delete_branch(repo_root, branch_name)

        # Always move back to base branch after execution.
        current = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        if current != base_branch:
            _checkout_branch(repo_root, base_branch)

        result["stage4"] = {
            "improved": bool(improved),
            "push_done": push_done,
            "pr": pr_status,
            "alerts": alerts,
            "branch_kept": bool(improved or args.keep_failed_branch),
        }
        result["status"] = "success" if improved else "no_improvement"
        result["finished_at"] = _now_iso()
        _write_json(out_path, result)

        if improved:
            print(f"SUCCESS: Candidate validated. Human review requested. report={out_path}")
            return 0
        print(f"NO IMPROVEMENT: Candidate rejected. report={out_path}")
        return 2

    except Exception as e:
        # Best-effort return to the base branch.
        try:
            cur = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
            if cur != base_branch:
                _checkout_branch(repo_root, base_branch)
        except Exception:
            pass
        result["status"] = "error"
        result["error"] = str(e)
        result["finished_at"] = _now_iso()
        _write_json(out_path, result)
        print(f"ERROR: {e}\nreport={out_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
