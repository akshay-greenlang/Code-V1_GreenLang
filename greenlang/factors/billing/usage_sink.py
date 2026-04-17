# -*- coding: utf-8 -*-
"""Optional durable usage events (C5) when GL_FACTORS_USAGE_SQLITE is set."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Optional


def _usage_path() -> Optional[Path]:
    raw = os.getenv("GL_FACTORS_USAGE_SQLITE", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def record_path_hit(path: str, api_key: Optional[str], tier: Optional[str]) -> None:
    p = _usage_path()
    if not p:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    key_hash = ""
    if api_key:
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                api_key_hash TEXT,
                tier TEXT,
                hit_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO api_usage_events (path, api_key_hash, tier) VALUES (?, ?, ?)",
            (path, key_hash or None, tier),
        )
        conn.commit()
    finally:
        conn.close()
