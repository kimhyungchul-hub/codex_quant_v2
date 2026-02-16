#!/usr/bin/env python3
"""Diagnose all 4 issues: auto_reval, mc_health, edge filter, leverage/capital."""
import asyncio, json, sys
try:
    import aiohttp
except ImportError:
    print("aiohttp not installed"); sys.exit(1)

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect('http://127.0.0.1:9999/ws', timeout=15) as ws:
            for _ in range(50):
                msg = await ws.receive(timeout=15)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                d = json.loads(msg.data)
                if d.get('type') != 'full_update':
                    continue

                # 1. Auto Reval
                ar = d.get('auto_reval') or {}
                print("=" * 60)
                print("=== AUTO REVAL ===")
                print(json.dumps(ar, ensure_ascii=False, indent=2))

                # 2. MC Health
                mc = d.get('mc_health') or {}
                print("\n" + "=" * 60)
                print("=== MC HEALTH ===")
                print(json.dumps(mc, ensure_ascii=False, indent=2))

                # 3. Edge / Filter info (all symbols)
                rows = d.get('market') or []
                print("\n" + "=" * 60)
                print(f"=== MARKET FILTERS ({len(rows)} symbols) ===")
                for r in rows:
                    meta = r.get('meta') or {}
                    print(json.dumps({
                        'sym': r.get('symbol'),
                        'action': r.get('action'),
                        'filter': r.get('filter_block') or meta.get('filter_block'),
                        'ev': r.get('ev'),
                        'lev': r.get('leverage'),
                        'notional': r.get('notional'),
                        'edge': meta.get('edge'),
                        'net_eff': meta.get('net_expectancy_effective'),
                        'net_min': meta.get('net_expectancy_min'),
                        'lev_cf_best': meta.get('lev_cf_best'),
                        'capital_cf_best': meta.get('capital_cf_best'),
                    }, ensure_ascii=False))

                # 4. Adaptive entry
                ad = d.get('adaptive_entry') or {}
                print("\n" + "=" * 60)
                print("=== ADAPTIVE ENTRY ===")
                print(json.dumps(ad, ensure_ascii=False, indent=2))

                return
    print("NO_FULL_UPDATE")

asyncio.run(main())
