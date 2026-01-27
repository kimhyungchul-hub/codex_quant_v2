from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
logger = logging.getLogger(__name__)

# ============================================================================
# SQLite Integrity Check & Recovery
# ============================================================================
DEFAULT_DB_CANDIDATES = [
    STATE_DIR / "bot_data.db",
    STATE_DIR / "bot_data_paper.db",
    STATE_DIR / "bot_data_live.db",
]


def check_integrity(db_path: Path) -> tuple[bool, str]:
    """Run PRAGMA integrity_check and REINDEX on index corruption."""
    if not db_path.exists():
        return True, "missing"

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        result = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        ok = str(result).lower() == "ok"
        if not ok:
            conn.execute("REINDEX;")
            conn.commit()
        return ok, str(result)
    except Exception as exc:  # pragma: no cover - best-effort bootstrap
        return False, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def run_integrity_checks(db_candidates: list[Path] | None = None) -> None:
    paths = db_candidates or DEFAULT_DB_CANDIDATES
    for db_path in paths:
        ok, detail = check_integrity(db_path)
        if detail == "missing":
            continue
        prefix = "✅" if ok else "⚠️"
        msg = f"{prefix} [BOOTSTRAP] integrity_check {db_path}: {detail}"
        print(msg)
        if not ok:
            logger.warning(msg)


try:
    run_integrity_checks()
except Exception as exc:  # pragma: no cover - fail open
    print(f"⚠️  [BOOTSTRAP] integrity check skipped: {exc}")
