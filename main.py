from __future__ import annotations

import argparse
import asyncio
import os

# JAX platform 설정 - GPU 전용 모드 (CPU 폴백 없음)
# 반드시 JAX import보다 먼저 실행됩니다. GPU가 없으면 초기화에서 실패합니다.
platform_env = os.environ.get("JAX_PLATFORMS", "").strip()
if not platform_env:
    # On macOS we request the Metal backend explicitly (uppercase 'METAL').
    # This enforces GPU-only execution; initialization will fail if Metal is unavailable.
    os.environ["JAX_PLATFORMS"] = "METAL"

import config
from core.dashboard_server import DashboardServer
from core.orchestrator import LiveOrchestrator, build_exchange, build_data_exchange
import core.orchestrator as orch_mod
print(f"🚀 [INIT_DEBUG] LiveOrchestrator file: {orch_mod.__file__}")
from engines import (
    alpha_features_methods,
    cvar_methods,
    evaluation_methods,
    exit_policy_methods,
    probability_methods,
    running_stats_methods,
    simulation_methods,
)
from engines.mc import alpha_hit as mc_alpha_hit
from engines.mc import decision as mc_decision
from engines.mc import entry_evaluation as mc_entry_evaluation
from engines.mc import execution_costs as mc_execution_costs
from engines.mc import execution_mix as mc_execution_mix
from engines.mc import exit_policy as mc_exit_policy
from engines.mc import first_passage as mc_first_passage
from engines.mc import monte_carlo_engine as mc_engine_module
from engines.mc import path_simulation as mc_path_simulation
from engines.mc import policy_weights as mc_policy_weights
from engines.mc import runtime_params as mc_runtime_params
from engines.mc import signal_features as mc_signal_features
from engines.mc import tail_sampling as mc_tail_sampling


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="codex_quant entrypoint")
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="Only import modules and exit (no network).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help='Comma-separated symbols (e.g. "BTC/USDT:USDT,ETH/USDT:USDT").',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard port override (default: config.PORT).",
    )
    parser.add_argument(
        "--decision-refresh-sec",
        type=float,
        default=None,
        help="Decision loop interval seconds (default: config.DECISION_REFRESH_SEC).",
    )
    parser.add_argument("--paper", dest="paper", action="store_true", help="Enable paper trading loop.")
    parser.add_argument("--no-paper", dest="paper", action="store_false", help="Disable paper trading loop.")
    parser.set_defaults(paper=None)
    parser.add_argument("--paper-size-frac", type=float, default=None, help="Default paper position size fraction (0~1).")
    parser.add_argument("--paper-leverage", type=float, default=None, help="Default paper leverage.")
    parser.add_argument("--mc-n-paths-live", type=int, default=None, help="MC live n_paths override (ctx.n_paths).")
    parser.add_argument("--mc-n-paths-exit", type=int, default=None, help="MC exit-policy n_paths override.")
    parser.add_argument(
        "--mc-tail-mode",
        type=str,
        default=None,
        choices=("gaussian", "student_t", "bootstrap"),
        help="MC tail distribution mode.",
    )
    parser.add_argument("--mc-student-t-df", type=float, default=None, help="MC student_t df (>= 2.1).")
    parser.add_argument("--mc-use-jax", dest="mc_use_jax", action="store_true", help="Force enable JAX in MC.")
    parser.add_argument("--mc-no-jax", dest="mc_use_jax", action="store_false", help="Disable JAX in MC.")
    parser.set_defaults(mc_use_jax=None)
    parser.add_argument(
        "--exec-mode",
        type=str,
        default=None,
        choices=("market", "maker_then_market"),
        help="Execution mode used by MC cost model.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable aiohttp dashboard server.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip OHLCV preload step.",
    )
    return parser.parse_args()


def _apply_default_run_mode() -> None:
    """
    Default run mode (mirrors `scripts/run_main_paper_mainnet.sh`):
      - mainnet market data
      - no real orders (paper trading)
      - engine optimal sizing/leverage enabled
    """
    os.environ["BYBIT_TESTNET"] = "0"
    os.environ["ENABLE_LIVE_ORDERS"] = "0"
    os.environ["PAPER_TRADING"] = "1"
    os.environ["PAPER_USE_ENGINE_SIZING"] = "1"
    os.environ.setdefault("PAPER_EXIT_POLICY_ONLY", "1")

    # `config` is imported at module import time, so we also align its runtime values.
    config.ENABLE_LIVE_ORDERS = False


async def main() -> None:
    args = _parse_args()

    # Ensure the split-out modules are importable (refactor smoke check).
    _ = (
        alpha_features_methods,
        cvar_methods,
        evaluation_methods,
        exit_policy_methods,
        probability_methods,
        running_stats_methods,
        simulation_methods,
        mc_alpha_hit,
        mc_decision,
        mc_engine_module,
        mc_entry_evaluation,
        mc_execution_costs,
        mc_execution_mix,
        mc_exit_policy,
        mc_first_passage,
        mc_path_simulation,
        mc_policy_weights,
        mc_runtime_params,
        mc_signal_features,
        mc_tail_sampling,
        LiveOrchestrator,
        DashboardServer,
    )

    if args.imports_only:
        print("Imports OK: core/* + engines/*_methods + engines/mc/*")
        return

    if (
        os.environ.get("BYBIT_TESTNET") is None
        and os.environ.get("ENABLE_LIVE_ORDERS") is None
        and os.environ.get("PAPER_TRADING") is None
    ):
        _apply_default_run_mode()

    if args.exec_mode:
        # evaluation code reads EXEC_MODE from env at runtime
        os.environ["EXEC_MODE"] = str(args.exec_mode).strip().lower()

    exchange = await build_exchange()
    data_exchange = exchange
    if os.environ.get("DATA_BYBIT_TESTNET") is not None:
        try:
            data_exchange = await build_data_exchange()
        except Exception as e:
            print(f"[WARN] build_data_exchange failed; using trade exchange data: {e}")
            data_exchange = exchange

    if data_exchange is not exchange:
        def _api_base(ex):
            urls = getattr(ex, "urls", None) or {}
            api = urls.get("api")
            if isinstance(api, dict):
                pub = api.get("public")
                if isinstance(pub, dict):
                    return pub.get("v5") or next(iter(pub.values()), None) or str(pub)
                return pub or str(api)
            return api

        def _sandbox_flag(ex):
            opt = getattr(ex, "options", None) or {}
            return bool(opt.get("sandboxMode")) or bool(getattr(ex, "sandbox", False))

        print(
            f"[EXCHANGE_MODE] BYBIT_TESTNET={os.environ.get('BYBIT_TESTNET','')} DATA_BYBIT_TESTNET={os.environ.get('DATA_BYBIT_TESTNET','')}",
            flush=True,
        )
        print(
            f"[EXCHANGE_URLS] trade_sandbox={_sandbox_flag(exchange)} trade_api={_api_base(exchange)} | data_sandbox={_sandbox_flag(data_exchange)} data_api={_api_base(data_exchange)}",
            flush=True,
        )
    symbols = None
    if args.symbols:
        parts = [p.strip() for p in str(args.symbols).split(",")]
        symbols = [p for p in parts if p]

    # Validate symbols against exchange markets to avoid startup stalls/errors
    # (e.g. one invalid symbol breaks fetch_tickers for all symbols).
    try:
        timeout_sec = float(getattr(config, "CCXT_TIMEOUT_MS", 20_000)) / 1000.0
        await asyncio.wait_for(exchange.load_markets(), timeout=max(5.0, timeout_sec))
        markets = getattr(exchange, "markets", None) or {}
        base_syms = symbols if symbols else list(config.SYMBOLS)
        valid = [s for s in base_syms if s in markets]
        invalid = [s for s in base_syms if s not in markets]
        if invalid and valid:
            print(f"[WARN] Dropping unsupported symbols: {', '.join(invalid)}")
            symbols = valid
    except Exception as e:
        print(f"[WARN] load_markets failed; using symbols as-is: {e}")

    if data_exchange is not exchange and symbols:
        try:
            timeout_sec = float(getattr(config, "CCXT_TIMEOUT_MS", 20_000)) / 1000.0
            await asyncio.wait_for(data_exchange.load_markets(), timeout=max(5.0, timeout_sec))
            markets = getattr(data_exchange, "markets", None) or {}
            valid = [s for s in symbols if s in markets]
            invalid = [s for s in symbols if s not in markets]
            if invalid and valid:
                print(f"[DATA] Dropping symbols missing on data feed: {', '.join(invalid)}")
                symbols = valid
        except Exception as e:
            print(f"[WARN] data_exchange.load_markets failed; using symbols as-is: {e}")

    orchestrator = LiveOrchestrator(exchange, symbols if symbols else config.SYMBOLS, data_exchange=data_exchange)

    # CLI overrides (runtime tuning defaults)
    if args.decision_refresh_sec is not None:
        orchestrator.decision_refresh_sec = float(max(0.2, float(args.decision_refresh_sec)))
    if args.paper is not None:
        orchestrator.paper_trading_enabled = bool(args.paper) and (not orchestrator.enable_orders)
    if args.paper_size_frac is not None:
        orchestrator.paper_size_frac_default = float(max(0.0, min(1.0, float(args.paper_size_frac))))
    if args.paper_leverage is not None:
        orchestrator.paper_leverage_default = float(max(0.0, float(args.paper_leverage)))
    if args.mc_n_paths_live is not None:
        orchestrator.mc_n_paths_live = int(max(200, min(200000, int(args.mc_n_paths_live))))
    if args.mc_n_paths_exit is not None:
        orchestrator.mc_n_paths_exit = int(max(200, min(200000, int(args.mc_n_paths_exit))))
        orchestrator._apply_mc_runtime_to_engines()
    if args.mc_use_jax is not None:
        orchestrator.mc_use_jax = bool(args.mc_use_jax)
    if args.mc_tail_mode is not None:
        orchestrator.mc_tail_mode = str(args.mc_tail_mode).strip().lower()
    if args.mc_student_t_df is not None:
        orchestrator.mc_student_t_df = float(max(2.1, float(args.mc_student_t_df)))

    if not args.no_dashboard:
        orchestrator.dashboard = DashboardServer(orchestrator, port=args.port)
        await orchestrator.dashboard.start()

    try:
        if config.PRELOAD_ON_START and (not args.no_preload):
            await orchestrator.data.preload_all_ohlcv(limit=int(config.OHLCV_PRELOAD_LIMIT))

        tasks = [
            asyncio.create_task(orchestrator.data.fetch_prices_loop()),
            asyncio.create_task(orchestrator.data.fetch_ohlcv_loop()),
            asyncio.create_task(orchestrator.data.fetch_orderbook_loop()),
            asyncio.create_task(orchestrator.live_sync_loop()) if bool(orchestrator.enable_orders) else None,
            asyncio.create_task(orchestrator.decision_worker_loop()),
            asyncio.create_task(orchestrator.decision_loop()),
        ]
        tasks = [t for t in tasks if t is not None]
        await asyncio.gather(*tasks)
    finally:
        try:
            if orchestrator.dashboard is not None:
                await orchestrator.dashboard.stop()
        finally:
            await exchange.close()
            if data_exchange is not exchange:
                await data_exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
