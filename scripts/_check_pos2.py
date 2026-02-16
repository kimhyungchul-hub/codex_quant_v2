import asyncio, json, aiohttp

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect('http://127.0.0.1:9999/ws', timeout=10) as ws:
            for _ in range(60):
                msg = await ws.receive(timeout=10)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                d = json.loads(msg.data)
                if d.get('type') != 'full_update':
                    continue
                rows = d.get('market') or []
                positions = []
                for r in rows:
                    meta = r.get('meta') or {}
                    qty = r.get('qty', 0)
                    if qty is None:
                        qty = 0
                    if float(qty) != 0:
                        positions.append({
                            'sym': r.get('symbol'),
                            'side': r.get('side'),
                            'qty': qty,
                            'pnl_pct': r.get('pnl_pct'),
                            'action': r.get('action'),
                            'mu_alpha': meta.get('mu_alpha'),
                            'mu_alpha_raw': meta.get('mu_alpha_raw'),
                            'mu_dir_prob_long': meta.get('mu_dir_prob_long'),
                            'direction': meta.get('direction'),
                            'ev_long': meta.get('policy_ev_score_long'),
                            'ev_short': meta.get('policy_ev_score_short'),
                            'unified_score': meta.get('unified_score'),
                            'regime': meta.get('regime'),
                            'regime_pol': meta.get('regime_policy_regime'),
                        })
                print(f"Total positions: {len(positions)}")
                for p in positions:
                    print(json.dumps(p, ensure_ascii=False))
                if not positions:
                    print("No open positions found in market rows")
                    # Show first 3 rows for debug
                    for r in rows[:3]:
                        print(f"  sample: sym={r.get('symbol')} qty={r.get('qty')} side={r.get('side')}")
                return

asyncio.run(main())
