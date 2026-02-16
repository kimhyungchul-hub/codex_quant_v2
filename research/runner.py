"""
research/runner.py — Main Research Engine Runner (v2)
======================================================
5분마다 CF 시뮬레이션 → 조건 충족 시 자동 적용 (보수적 모드)
1시간마다 Gemini 코드 리뷰 → git push

Usage:
  python -m research.runner              # 기본 (5분 주기 + auto-apply)
  python -m research.runner --once       # 1회만 실행
  python -m research.runner --stage leverage  # 특정 스테이지만
  python -m research.runner --no-dashboard    # 대시보드 없이
  python -m research.runner --no-auto-apply   # 자동 적용 비활성화
  python -m research.runner --no-gemini       # Gemini 리뷰 비활성화
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.cf_engine import CFEngine, TradeLoader, ALL_SIMULATORS, compute_metrics_by_regime
from research.documenter import (
    save_findings_json,
    generate_findings_markdown,
    generate_changelog_entry,
)

logger = logging.getLogger("research.runner")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
DEFAULT_DB = os.path.join(PROJECT_ROOT, "state", "bot_data_live.db")
CF_INTERVAL_SEC = 300          # 5분마다 CF 분석
GEMINI_INTERVAL_SEC = 3600     # 1시간마다 Gemini 리뷰
MAX_COMBOS_PER_STAGE = 220
DASHBOARD_PORT = 9998
FINDINGS_OUTPUT = os.path.join(PROJECT_ROOT, "docs", "RESEARCH_FINDINGS.md")
FINDINGS_JSON = os.path.join(PROJECT_ROOT, "state", "research_findings.json")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def run_cf_cycle(
    db_path: str = DEFAULT_DB,
    stage_filter: str | None = None,
    max_combos: int = MAX_COMBOS_PER_STAGE,
) -> dict:
    """Run one complete CF analysis cycle and return results."""
    t0 = time.perf_counter()

    loader = TradeLoader(db_path)
    trades = loader.load_trades()
    if not trades:
        logger.warning("No trades found. Skipping CF cycle.")
        return {"status": "no_trades"}

    simulators = ALL_SIMULATORS
    if stage_filter:
        simulators = [s for s in ALL_SIMULATORS if s.stage_name == stage_filter]
        if not simulators:
            logger.error(f"Unknown stage: {stage_filter}")
            return {"status": "unknown_stage"}

    engine = CFEngine(trades, simulators)
    sweep_progress = {}

    for sim in simulators:
        sweep_progress[sim.stage_name] = {"done": 0, "total": 1, "status": "running"}
        try:
            _update_dashboard({
                "sweep_progress": sweep_progress,
                "running": True,
                "stage_current": sim.stage_name,
            })
            engine._run_stage(sim, max_combos)
            sweep_progress[sim.stage_name] = {"done": 1, "total": 1, "status": "done"}
        except Exception as e:
            logger.error(f"Stage {sim.stage_name} failed: {e}")
            sweep_progress[sim.stage_name] = {"done": 0, "total": 1, "status": "error"}

    elapsed = time.perf_counter() - t0
    findings = engine.get_top_findings(20)
    apply_history_snapshot = []
    try:
        from research.auto_apply import get_apply_history
        apply_history_snapshot = get_apply_history()
    except Exception:
        apply_history_snapshot = []

    result = {
        "status": "ok",
        "elapsed_sec": round(elapsed, 1),
        "running": False,
        "baseline": engine.baseline,
        "baseline_by_regime": engine.baseline_by_regime,
        "n_trades": len(trades),
        "n_results": len(engine.results),
        "n_findings": len(findings),
        "findings": [asdict(f) for f in findings],
        "top_10": [asdict(f) for f in findings[:10]],
        "sweep_progress": sweep_progress,
        "apply_history": apply_history_snapshot,
        "cycle_count": getattr(run_cf_cycle, "_cycle", 0),
        "last_update_ts": time.time(),
    }
    run_cf_cycle._cycle = getattr(run_cf_cycle, "_cycle", 0) + 1

    try:
        save_findings_json([asdict(f) for f in findings], FINDINGS_JSON)
    except Exception as e:
        logger.error(f"Failed to save findings JSON: {e}")

    try:
        generate_findings_markdown(
            [asdict(f) for f in findings],
            engine.baseline,
            engine.baseline_by_regime,
            FINDINGS_OUTPUT,
        )
    except Exception as e:
        logger.error(f"Failed to generate markdown: {e}")

    try:
        changelog = generate_changelog_entry([asdict(f) for f in findings[:5]])
        if changelog:
            logger.info(f"\n{'='*60}\nCHANGE LOG ENTRY:\n{'='*60}\n{changelog}\n{'='*60}")
    except Exception:
        pass

    _update_dashboard(result)

    logger.info(
        f"CF cycle complete: {len(findings)} findings in {elapsed:.1f}s "
        f"(trades={len(trades)}, results={len(engine.results)})"
    )

    return result


# ─────────────────────────────────────────────────────────────────
# Dashboard Integration
# ─────────────────────────────────────────────────────────────────

_dashboard_update_fn = None


def _update_dashboard(data: dict):
    global _dashboard_update_fn
    if _dashboard_update_fn:
        try:
            _dashboard_update_fn(data)
        except Exception:
            pass


def start_dashboard_thread(port: int = DASHBOARD_PORT):
    global _dashboard_update_fn
    try:
        from research.dashboard import update_state, start_dashboard_thread as _start
        _dashboard_update_fn = update_state
        t = _start(port=port)
        logger.info(f"Research dashboard started on http://0.0.0.0:{port}")
        return t
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Auto-Apply Integration
# ─────────────────────────────────────────────────────────────────

def run_auto_apply(findings: list[dict]) -> dict:
    """Run auto-apply cycle with safety guards."""
    try:
        from research.auto_apply import auto_apply_cycle, get_apply_history
        status = auto_apply_cycle(findings)
        _update_dashboard({
            "auto_apply": status,
            "apply_history": get_apply_history(),
        })
        return status
    except Exception as e:
        logger.error(f"Auto-apply error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


# ─────────────────────────────────────────────────────────────────
# Gemini Review Integration
# ─────────────────────────────────────────────────────────────────

def run_gemini_cycle(findings: list[dict]) -> dict:
    """Run Gemini code review."""
    try:
        from research.gemini_reviewer import run_gemini_review, is_available
        if not is_available():
            logger.debug("Gemini API key not set — skipping review")
            return {"status": "disabled"}
        from research.auto_apply import get_apply_history
        status = run_gemini_review(
            findings=findings,
            apply_history=get_apply_history(),
            auto_apply_env=False,  # Conservative: Gemini suggests only
        )
        _update_dashboard({"gemini_review": status})
        return status
    except Exception as e:
        logger.error(f"Gemini review error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


# ─────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Codex Quant Research Engine v2")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--stage", default=None, help="Run specific stage only")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--no-auto-apply", action="store_true", help="Disable auto-apply")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini review")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="Dashboard port")
    parser.add_argument("--interval", type=int, default=CF_INTERVAL_SEC, help="CF cycle interval (sec)")
    parser.add_argument("--gemini-interval", type=int, default=GEMINI_INTERVAL_SEC, help="Gemini review interval (sec)")
    parser.add_argument("--max-combos", type=int, default=MAX_COMBOS_PER_STAGE, help="Max parameter combos per stage")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Codex Quant Research Engine v2 — Auto-Optimize")
    logger.info(f"  DB: {args.db}")
    logger.info(f"  Stage: {args.stage or 'ALL ({} simulators)'.format(len(ALL_SIMULATORS))}")
    logger.info(f"  CF interval: {args.interval}s ({args.interval//60}min)")
    logger.info(f"  Auto-apply: {'ON (qualifying findings all apply, 30min monitor each, rollback on -$10)' if not args.no_auto_apply else 'OFF'}")
    logger.info(f"  Gemini review: {'ON' if not args.no_gemini else 'OFF'} (every {args.gemini_interval//60}min)")
    logger.info(f"  Dashboard: {'OFF' if args.no_dashboard else f'http://0.0.0.0:{args.port}'}")
    logger.info("=" * 60)

    if not args.no_dashboard:
        start_dashboard_thread(args.port)
        time.sleep(1)

    last_gemini_ts = 0
    last_findings: list[dict] = []

    try:
        while True:
            # ── Phase 1: CF Analysis (expanded stage simulators) ──
            result = run_cf_cycle(
                db_path=args.db,
                stage_filter=args.stage,
                max_combos=args.max_combos,
            )

            findings = result.get("findings", [])
            if findings:
                last_findings = findings

            if findings:
                print("\n" + "─" * 60)
                print(f"  TOP FINDINGS (cycle {result.get('cycle_count', 0)})")
                print("─" * 60)
                for i, f in enumerate(findings[:5], 1):
                    pnl = f.get("improvement_pct", 0)
                    conf = f.get("confidence", 0)
                    stage = f.get("stage", "?")
                    applied = " [APPLIED]" if f.get("applied") else ""
                    print(f"  {i}. [{stage}] ΔPnL: ${pnl:+.2f} | Conf: {conf:.0%}{applied}")
                print("─" * 60)

            # ── Phase 2: Auto-Apply (confidence ≥80%, ΔPnL ≥$100) ──
            if not args.no_auto_apply and findings:
                apply_status = run_auto_apply(findings)
                action = apply_status.get("last_action")
                if action and action not in ("no_qualifying_findings", "cooldown"):
                    logger.info(f"[AUTO_APPLY] {action}")

            # ── Phase 3: Gemini Review (hourly) ──
            if not args.no_gemini and (time.time() - last_gemini_ts) >= args.gemini_interval:
                logger.info("Starting Gemini code review...")
                gemini_status = run_gemini_cycle(last_findings)
                last_gemini_ts = time.time()
                if gemini_status.get("status") == "ok":
                    logger.info(
                        f"[GEMINI] Review complete: {gemini_status.get('n_suggestions', 0)} suggestions, "
                        f"risk={gemini_status.get('risk_score', '?')}/10"
                    )
                elif gemini_status.get("status") != "disabled":
                    logger.warning(f"[GEMINI] {gemini_status}")

            if args.once:
                break

            logger.info(f"Next CF cycle in {args.interval}s...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Research engine stopped by user")


if __name__ == "__main__":
    main()
