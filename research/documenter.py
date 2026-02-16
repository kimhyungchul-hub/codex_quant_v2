"""
research/documenter.py — Auto-Documentation Module
====================================================
CF 연구 결과를 자동으로 문서화하고 기존 docs에 반영.
- Findings → docs/RESEARCH_FINDINGS.md (주기적 갱신)
- 중요 발견 → copilot-instructions.md Change Log 형식 출력
- CODE_MAP_v2.md 업데이트 제안 생성
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("research.documenter")

FINDINGS_DOC_PATH = "docs/RESEARCH_FINDINGS.md"
RESEARCH_LOG_PATH = "state/research_findings.json"


def save_findings_json(findings: list[dict], path: str = RESEARCH_LOG_PATH):
    """Save findings to JSON for persistence."""
    try:
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        # Merge by finding_id
        existing_ids = {f["finding_id"] for f in existing}
        for f in findings:
            if f.get("finding_id") not in existing_ids:
                existing.append(f)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Saved {len(findings)} findings to {path}")
    except Exception as e:
        logger.error(f"Failed to save findings: {e}")


def generate_findings_markdown(
    findings: list[dict],
    baseline: dict,
    baseline_by_regime: dict,
    output_path: str = FINDINGS_DOC_PATH,
):
    """Generate/update the research findings markdown document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Research Findings — Counterfactual Analysis",
        f"",
        f"> Auto-generated: {now}",
        f"> Baseline: {baseline.get('n', 0)} trades, PnL=${baseline.get('pnl', 0):.2f}, "
        f"WR={baseline.get('wr', 0):.1%}, R:R={baseline.get('rr', 0):.2f}",
        f"",
        f"## Pipeline Stage Impact Summary",
        f"",
    ]

    # Group findings by stage
    by_stage: dict[str, list[dict]] = {}
    for f in findings:
        stage = f.get("stage", "unknown")
        by_stage.setdefault(stage, []).append(f)

    stage_descriptions = {
        "leverage": "레버리지 결정",
        "tp_sl": "TP/SL 타겟",
        "hold_duration": "보유 시간",
        "entry_filter": "진입 필터",
        "direction": "방향 결정",
        "vpin_filter": "VPIN 필터",
        "exit_reason": "청산 로직",
        "capital_allocation": "자본 분배",
        "regime_multiplier": "레짐 보정",
    }

    for stage, stage_findings in by_stage.items():
        desc = stage_descriptions.get(stage, stage)
        best = max(stage_findings, key=lambda f: f.get("improvement_pct", 0))
        lines.append(f"### {stage.upper()} — {desc}")
        lines.append(f"")
        lines.append(f"**Best Finding:** {best.get('title', '')}")
        lines.append(f"- Improvement: ${best.get('improvement_pct', 0):+.2f}")
        lines.append(f"- Confidence: {best.get('confidence', 0):.0%}")
        lines.append(f"- Parameters: `{json.dumps(best.get('param_changes', {}))}`")
        lines.append(f"")
        if best.get("recommendation"):
            lines.append(f"```")
            lines.append(best["recommendation"])
            lines.append(f"```")
            lines.append(f"")
        # Comparison table
        bl = best.get("baseline_metrics", {})
        im = best.get("improved_metrics", {})
        lines.append(f"| Metric | Baseline | CF | Delta |")
        lines.append(f"|--------|----------|----|----|")
        for k in ["n", "pnl", "wr", "rr", "edge", "sharpe", "pf"]:
            bv = bl.get(k, 0)
            iv = im.get(k, 0)
            dv = iv - bv if isinstance(bv, (int, float)) else 0
            fmt = ".4f" if k in ("wr", "edge") else ".2f" if k in ("pnl", "rr", "sharpe", "pf") else "d"
            lines.append(f"| {k} | {bv:{fmt}} | {iv:{fmt}} | {dv:+{fmt}} |")
        lines.append(f"")

    # Regime performance
    lines.append(f"## Regime Performance Breakdown")
    lines.append(f"")
    lines.append(f"| Regime | N | PnL | WR | R:R | Edge |")
    lines.append(f"|--------|---|-----|----|----|------|")
    for regime, m in baseline_by_regime.items():
        lines.append(
            f"| {regime} | {m.get('n', 0)} | ${m.get('pnl', 0):.2f} | "
            f"{m.get('wr', 0):.1%} | {m.get('rr', 0):.2f} | {m.get('edge', 0):+.1%} |"
        )
    lines.append(f"")

    # Action items
    lines.append(f"## 🎯 Recommended Actions")
    lines.append(f"")
    for i, f in enumerate(sorted(findings, key=lambda x: x.get("improvement_pct", 0), reverse=True)[:5], 1):
        lines.append(f"{i}. **{f.get('title', '')}** (ΔPnL: ${f.get('improvement_pct', 0):+.2f}, confidence: {f.get('confidence', 0):.0%})")
        if f.get("param_changes"):
            for pk, pv in f["param_changes"].items():
                lines.append(f"   - `{pk}` = `{pv}`")
        lines.append(f"")

    content = "\n".join(lines)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Generated {output_path} ({len(findings)} findings)")
    except Exception as e:
        logger.error(f"Failed to write {output_path}: {e}")

    return content


def generate_changelog_entry(findings: list[dict]) -> str:
    """Generate a Change Log entry for copilot-instructions.md."""
    if not findings:
        return ""
    now = datetime.now().strftime("%Y-%m-%d")
    lines = [f"### [{now}] Research Engine — CF 분석 결과"]
    lines.append(f"**발견:** {len(findings)}개의 유의미한 파라미터 최적화 발견")
    lines.append(f"")
    for f in findings[:5]:
        lines.append(f"- **{f.get('stage', '').upper()}**: {f.get('title', '')} "
                     f"(ΔPnL: ${f.get('improvement_pct', 0):+.2f}, 신뢰도: {f.get('confidence', 0):.0%})")
    lines.append(f"")
    lines.append(f"**영향 파일:** `research/cf_engine.py`, `docs/RESEARCH_FINDINGS.md`")
    return "\n".join(lines)
