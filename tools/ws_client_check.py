import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

import aiohttp

WS_URL = os.environ.get("WS_URL", "ws://localhost:9999/ws")
DURATION = int(os.environ.get("WS_CHECK_SEC", "60"))

async def run():
    msg_count = 0
    full_update_count = 0
    init_count = 0
    samples = []
    end_time = datetime.now() + timedelta(seconds=DURATION)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL, heartbeat=30) as ws:
                print(f"Connected to {WS_URL}, listening for {DURATION}s...")
                while datetime.now() < end_time:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout= max(1, (end_time - datetime.now()).total_seconds()))
                    except asyncio.TimeoutError:
                        break
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        print("WebSocket closed by server")
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        print("WebSocket error", msg)
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        text = msg.data
                        msg_count += 1
                        try:
                            obj = json.loads(text)
                        except Exception:
                            obj = {"raw": text}
                        mtype = obj.get("type") if isinstance(obj, dict) else None
                        if mtype == "full_update":
                            full_update_count += 1
                        if mtype == "init":
                            init_count += 1
                        if len(samples) < 5:
                            samples.append(obj)
                        # lightweight log
                        print(f"[{msg_count}] {mtype or 'text'} -- keys={list(obj.keys()) if isinstance(obj, dict) else 'raw'}")
                    else:
                        # binary or others
                        print(f"Received non-text message: {msg.type}")
    except Exception as e:
        print(f"Exception while connecting/receiving: {e}")
        raise

    print("\n--- Summary ---")
    print(f"Total messages: {msg_count}")
    print(f"init messages: {init_count}")
    print(f"full_update messages: {full_update_count}")
    print("Sample messages (up to 5):")
    for s in samples:
        print(json.dumps(s, ensure_ascii=False) if isinstance(s, dict) else str(s))

if __name__ == '__main__':
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(0)
