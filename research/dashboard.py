"""
research/dashboard.py — Research Dashboard Server (stdlib only)
================================================================
Python 표준 라이브러리만 사용하는 경량 WebSocket + HTTP 대시보드.
FastAPI/uvicorn 의존성 없이 동작.
"""
from __future__ import annotations

import asyncio
import hashlib
import base64
import json
import logging
import os
import struct
import time
from pathlib import Path
from threading import Thread
from typing import Optional

logger = logging.getLogger("research.dashboard")

# Global state
_state: dict = {
    "baseline": {},
    "findings": [],
    "sweep_progress": {},
    "last_update_ts": 0,
    "running": False,
    "cycle_count": 0,
}
_ws_clients: list = []
_dashboard_html: str = ""


def update_state(data: dict):
    _state.update(data)
    _state["last_update_ts"] = time.time()


def _load_html() -> str:
    global _dashboard_html
    if not _dashboard_html:
        html_path = Path(__file__).parent / "dashboard.html"
        if html_path.exists():
            _dashboard_html = html_path.read_text(encoding="utf-8")
        else:
            _dashboard_html = "<html><body><h1>Research Dashboard — HTML not found</h1></body></html>"
    return _dashboard_html


async def _ws_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        request_data = b""
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            request_data += line
            if line == b"\r\n" or not line:
                break

        request_text = request_data.decode("utf-8", errors="replace")
        if "upgrade" in request_text.lower() and "websocket" in request_text.lower():
            await _handle_websocket(reader, writer, request_text)
        elif "GET /api/state" in request_text:
            await _handle_api(writer)
        else:
            await _handle_http(writer)
    except Exception as e:
        logger.debug(f"Connection error: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_http(writer: asyncio.StreamWriter):
    html = _load_html()
    body = html.encode("utf-8")
    resp = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + body
    writer.write(resp)
    await writer.drain()


async def _handle_api(writer: asyncio.StreamWriter):
    body = json.dumps(_state, default=str, ensure_ascii=False).encode("utf-8")
    resp = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + body
    writer.write(resp)
    await writer.drain()


async def _handle_websocket(reader, writer, request):
    key = ""
    for line in request.split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
    if not key:
        return
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(hashlib.sha1((key + magic).encode()).digest()).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    writer.write(resp.encode())
    await writer.drain()
    _ws_clients.append(writer)
    logger.info(f"WS client connected (total: {len(_ws_clients)})")
    try:
        await _ws_send(writer, json.dumps(_state, default=str, ensure_ascii=False))
        while True:
            try:
                data = await asyncio.wait_for(_ws_read_frame(reader), timeout=60)
                if data is None:
                    break
                if data == "ping":
                    await _ws_send(writer, "pong")
            except asyncio.TimeoutError:
                await _ws_send(writer, json.dumps({"type": "heartbeat", "ts": time.time()}))
    except Exception:
        pass
    finally:
        if writer in _ws_clients:
            _ws_clients.remove(writer)


async def _ws_read_frame(reader) -> Optional[str]:
    try:
        header = await reader.readexactly(2)
        if header[0] & 0x0F == 0x8:
            return None
        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", await reader.readexactly(8))[0]
        mask = await reader.readexactly(4) if masked else b"\x00" * 4
        payload = await reader.readexactly(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return payload.decode("utf-8", errors="replace")
    except Exception:
        return None


async def _ws_send(writer, message: str):
    try:
        data = message.encode("utf-8")
        frame = bytearray([0x81])
        length = len(data)
        if length < 126:
            frame.append(length)
        elif length < 65536:
            frame.append(126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(127)
            frame.extend(struct.pack(">Q", length))
        frame.extend(data)
        writer.write(bytes(frame))
        await writer.drain()
    except Exception:
        if writer in _ws_clients:
            _ws_clients.remove(writer)


async def broadcast_state():
    if not _ws_clients:
        return
    payload = json.dumps(_state, default=str, ensure_ascii=False)
    dead = []
    for w in list(_ws_clients):
        try:
            await _ws_send(w, payload)
        except Exception:
            dead.append(w)
    for w in dead:
        if w in _ws_clients:
            _ws_clients.remove(w)


async def _periodic_broadcast():
    while True:
        await asyncio.sleep(5)
        try:
            await broadcast_state()
        except Exception:
            pass


async def _start_server(host: str, port: int):
    server = await asyncio.start_server(_ws_handler, host, port)
    logger.info(f"Research dashboard on http://{host}:{port}")
    asyncio.create_task(_periodic_broadcast())
    async with server:
        await server.serve_forever()


def run_dashboard(host: str = "0.0.0.0", port: int = 9998):
    asyncio.run(_start_server(host, port))


def start_dashboard_thread(host: str = "0.0.0.0", port: int = 9998):
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start_server(host, port))
    t = Thread(target=_run, daemon=True, name="research-dashboard")
    t.start()
    logger.info(f"Dashboard thread started on http://{host}:{port}")
    return t
