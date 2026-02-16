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
                for r in rows:
                    meta = r.get('meta') or {}
                    if r.get('qty', 0) != 0 or meta.get('qty', 0) != 0:
                        print(json.dumps({
                            'sym': r.get('symbol'),
                            'side': r.get('side'),
                            'qty': r.get('qty'),
                            'pnl_pct': r.get('pnl_pct'),
                            'action': r.get('action'),
                            'mu_alpha': meta.get('mu_alpha'),
                            'mu_alpha_raw': meta.get('mu_alpha_raw'),
                            'mu_dir_prob_long': meta.get('mu_dir_prob_long'),
                            'mu_dir_conf': meta.get('mu_dir_conf'),
                            'direction': meta.get('direction'),
                            'ev': meta.get('ev'),
                            'unified_score': meta.get('unified_score'),
                            'regime': meta.get('regime'),
                            'regime_policy_regime': meta.get('regime_policy_regime'),
                        }, ensure_ascii=False))
                return

asyncio.run(main())
