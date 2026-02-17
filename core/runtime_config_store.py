from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_key(key: Any) -> str:
    txt = str(key or "").strip().upper()
    if not txt:
        return ""
    if not all(ch.isalnum() or ch == "_" for ch in txt):
        return ""
    return txt


def _normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
            return f"{value:.10g}"
        return f"{value:.8f}".rstrip("0").rstrip(".")
    txt = str(value).strip()
    return txt if txt else None


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn


def _ensure_schema_conn(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            source TEXT,
            revision INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            updated_at_ms INTEGER NOT NULL,
            source TEXT,
            reason TEXT,
            batch_id TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_config_updated ON runtime_config(updated_at_ms)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runtime_cfg_hist_key_ts ON runtime_config_history(key, updated_at_ms)")
    conn.commit()


def ensure_runtime_config_schema(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        _ensure_schema_conn(conn)


def get_runtime_config_values(
    db_path: str | Path,
    keys: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    with _connect(db_path) as conn:
        _ensure_schema_conn(conn)
        cur = conn.cursor()
        if keys is None:
            rows = cur.execute("SELECT key, value FROM runtime_config").fetchall()
        else:
            norm_keys = [_normalize_key(k) for k in keys]
            norm_keys = [k for k in norm_keys if k]
            if not norm_keys:
                return {}
            placeholders = ",".join("?" for _ in norm_keys)
            rows = cur.execute(
                f"SELECT key, value FROM runtime_config WHERE key IN ({placeholders})",
                tuple(norm_keys),
            ).fetchall()
        for row in rows:
            key = str(row["key"] or "").strip().upper()
            val = str(row["value"] or "")
            if key:
                out[key] = val
    return out


def get_runtime_config_last_change_id(db_path: str | Path) -> int:
    with _connect(db_path) as conn:
        _ensure_schema_conn(conn)
        row = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM runtime_config_history").fetchone()
        if not row:
            return 0
        return int(row["m"] or 0)


def get_runtime_config_changes_since(
    db_path: str | Path,
    last_change_id: int,
    *,
    limit: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    cap = max(1, min(int(limit), 5000))
    out: list[dict[str, Any]] = []
    new_last = int(last_change_id or 0)
    with _connect(db_path) as conn:
        _ensure_schema_conn(conn)
        rows = conn.execute(
            """
            SELECT id, key, old_value, new_value, updated_at_ms, source, reason, batch_id
            FROM runtime_config_history
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(last_change_id or 0), int(cap)),
        ).fetchall()
        for row in rows:
            rid = int(row["id"] or 0)
            if rid > new_last:
                new_last = rid
            out.append(
                {
                    "id": rid,
                    "key": str(row["key"] or "").strip().upper(),
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                    "updated_at_ms": int(row["updated_at_ms"] or 0),
                    "source": str(row["source"] or ""),
                    "reason": str(row["reason"] or ""),
                    "batch_id": str(row["batch_id"] or ""),
                }
            )
    return out, new_last


def set_runtime_config_values(
    db_path: str | Path,
    updates: dict[str, Any],
    *,
    source: str = "manual",
    reason: str = "",
    batch_id: str | None = None,
    updated_at_ms: int | None = None,
) -> dict[str, dict[str, str | None]]:
    changed: dict[str, dict[str, str | None]] = {}
    if not isinstance(updates, dict) or not updates:
        return changed

    ts_ms = int(updated_at_ms or _now_ms())
    src = str(source or "manual")
    why = str(reason or "")
    batch = str(batch_id or "")

    with _connect(db_path) as conn:
        _ensure_schema_conn(conn)
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        for raw_key, raw_val in updates.items():
            key = _normalize_key(raw_key)
            if not key:
                continue
            value = _normalize_value(raw_val)
            if value is None:
                continue

            row = cur.execute(
                "SELECT value, revision FROM runtime_config WHERE key = ?",
                (key,),
            ).fetchone()
            old_value = row["value"] if row else None
            old_revision = int(row["revision"] or 0) if row else 0
            if old_value == value:
                continue
            new_revision = int(old_revision + 1)

            cur.execute(
                """
                INSERT INTO runtime_config (key, value, updated_at_ms, source, revision)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at_ms = excluded.updated_at_ms,
                    source = excluded.source,
                    revision = excluded.revision
                """,
                (key, value, ts_ms, src, new_revision),
            )
            cur.execute(
                """
                INSERT INTO runtime_config_history (
                    key, old_value, new_value, updated_at_ms, source, reason, batch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, old_value, value, ts_ms, src, why, batch),
            )
            changed[key] = {"old": old_value, "new": value}
        conn.commit()

    return changed

