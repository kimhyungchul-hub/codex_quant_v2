import asyncio
import time
from pathlib import Path

from core.database_manager import DatabaseManager, TradingMode

ASYNC_WRITES = 1000
FOREGROUND_INTERVAL_SEC = 0.01
FOREGROUND_RUN_SEC = 1.0
JITTER_LIMIT_MS = 50.0


async def _run_async(db_path: Path):
    db = DatabaseManager(str(db_path))

    # Fire-and-forget async writes
    for i in range(ASYNC_WRITES):
        db.log_trade_background(
            {
                "symbol": "ASYNC/TEST",
                "side": "LONG",
                "price": 100.0,
                "qty": 1.0,
                "order_id": f"bg-{i}",
                "timestamp_ms": int(time.time() * 1000),
            },
            mode=TradingMode.PAPER,
        )

    # Foreground loop should not be blocked by DB writes
    ticks = []
    start = time.perf_counter()
    while (time.perf_counter() - start) < FOREGROUND_RUN_SEC:
        ticks.append(time.perf_counter())
        await asyncio.sleep(FOREGROUND_INTERVAL_SEC)

    # Wait for all pending background tasks to finish (bounded wait)
    pending = list(db._pending_tasks)
    if pending:
        await asyncio.wait(pending, timeout=5.0)
    await asyncio.sleep(0.05)

    # Count written rows
    with db._get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol='ASYNC/TEST'"
        ).fetchone()[0]

    intervals_ms = [
        (ticks[i] - ticks[i - 1]) * 1000.0 for i in range(1, len(ticks))
    ]
    jitter_ms = max(intervals_ms) if intervals_ms else 0.0

    return count, jitter_ms


def test_verify_async_db(tmp_path):
    db_path = tmp_path / "async_logging.db"
    count, jitter_ms = asyncio.run(_run_async(db_path))

    assert count >= ASYNC_WRITES, f"Expected {ASYNC_WRITES}+ rows, got {count}"
    assert jitter_ms <= JITTER_LIMIT_MS, (
        f"Main loop jitter too high: {jitter_ms:.2f} ms > {JITTER_LIMIT_MS:.2f} ms"
    )
