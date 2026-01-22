from __future__ import annotations

import bootstrap  # ensure JAX/XLA env is present for any spawned checks

import argparse
import json
import sys
import time
from typing import Any

import requests


def _safe_get(url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        return None, str(exc)


def _find_market_row(market: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    for row in market:
        if str(row.get("symbol")) == symbol:
            return row
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def check_ready(base_url: str, symbol: str, timeout: float, strict: bool) -> tuple[bool, dict[str, Any]]:
    status_url = f"{base_url.rstrip('/')}/api/status"
    score_url = f"{base_url.rstrip('/')}/api/score_debug?symbol={symbol}"

    status, status_err = _safe_get(status_url, timeout)
    score, score_err = _safe_get(score_url, timeout)

    diagnostics: dict[str, Any] = {
        "status_url": status_url,
        "score_url": score_url,
        "status_error": status_err,
        "score_error": score_err,
    }

    if status is None or score is None:
        diagnostics["ready"] = False
        return False, diagnostics

    market = status.get("market") or []
    row = _find_market_row(market, symbol)
    feed = status.get("feed") or {}
    engine = status.get("engine") or {}

    price = _as_float(row.get("price")) if row else None
    ev = _as_float(row.get("ev")) if row else None
    ev_raw = _as_float(row.get("ev_raw")) if row else None
    win_rate = _as_float(row.get("mc_win_rate")) if row else None
    cvar = _as_float(row.get("mc_cvar")) if row else None
    candles = row.get("candles") if row else None
    orderbook_ready = bool(row.get("orderbook_ready")) if row else False
    kline_age = row.get("kline_age") if row else None

    decision_cache = score.get("decision_cache") or {}
    ctx = decision_cache.get("ctx") if isinstance(decision_cache, dict) else None

    diagnostics.update(
        {
            "symbol": symbol,
            "feed_connected": bool(feed.get("connected")),
            "feed_last_msg_age": feed.get("last_msg_age"),
            "engine_mc_ready": bool(engine.get("mc_ready")),
            "market_row_found": row is not None,
            "price": price,
            "candles": candles,
            "kline_age": kline_age,
            "orderbook_ready": orderbook_ready,
            "ev": ev,
            "ev_raw": ev_raw,
            "win_rate": win_rate,
            "cvar": cvar,
            "decision_ctx_present": ctx is not None,
            "skip_reason": score.get("skip_reason"),
            "inputs": score.get("inputs"),
        }
    )

 

    # Core readiness checks
    has_price = price is not None and price > 0
    has_ctx = ctx is not None
    # 20개 고정이 아닌 인자값으로 변경 (기본값 완화)
    has_candles = candles is not None and int(candles) >= min_candles
    has_mc_ready = bool(engine.get("mc_ready"))

    # 실패 원인을 추적하기 위한 리스트
    fail_reasons = []
    if not has_price: fail_reasons.append("Missing Price")
    if not has_ctx: fail_reasons.append("Missing Decision Context")
    if not has_candles: fail_reasons.append(f"Insufficient Candles ({candles}/{min_candles})")
    if not has_mc_ready: fail_reasons.append("MC Engine Not Ready")

    ready = len(fail_reasons) == 0

    if strict:
        if not bool(feed.get("connected")): fail_reasons.append("Feed Disconnected")
        if not orderbook_ready: fail_reasons.append("Orderbook Not Ready")
        ready = len(fail_reasons) == 0

    diagnostics["ready"] = ready
    diagnostics["fail_reasons"] = fail_reasons  # 진단 정보에 추가
    
    return ready, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime system readiness checker for codex_quant")
    parser.add_argument("--base-url", default="http://localhost:9999", help="Dashboard base URL")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Symbol to validate")
    parser.add_argument("--timeout", type=float, default=2.0, help="Request timeout seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval seconds")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Max polling duration seconds")
    parser.add_argument("--strict", action="store_true", help="Require feed connected + orderbook ready")
    parser.add_argument("--json", action="store_true", help="Print JSON diagnostics each poll")
    args = parser.parse_args()

    start = time.time()
    last_diag: dict[str, Any] = {}

    while time.time() - start < float(args.max_seconds):
        ready, diag = check_ready(args.base_url, args.symbol, args.timeout, args.strict)
        last_diag = diag

        if args.json:
            print(json.dumps(diag, ensure_ascii=False))
        else:
            status = "READY" if ready else "WAIT"
            reasons = ", ".join(diag.get("fail_reasons", []))
            print(
                f"[{status}] {diag.get('symbol')} | Status: {reasons if reasons else 'All OK'} "
                f"ctx={diag.get('decision_ctx_present')} candles={diag.get('candles')} "
                f"feed={diag.get('feed_connected')} ob={diag.get('orderbook_ready')} "
                f"ev={diag.get('ev')} ev_raw={diag.get('ev_raw')}"
            )

        if ready:
            print("READY")
            return 0

        time.sleep(float(args.interval))

    print("TIMEOUT")
    if last_diag:
        print(json.dumps(last_diag, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())