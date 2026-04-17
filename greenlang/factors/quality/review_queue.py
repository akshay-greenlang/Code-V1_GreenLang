# -*- coding: utf-8 -*-
"""QA review queue persistence (Q3) — SQLite qa_reviews table."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Dict


def enqueue_review(
    conn: sqlite3.Connection,
    *,
    edition_id: str,
    factor_id: str,
    status: str,
    payload: Dict[str, Any],
) -> str:
    rid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO qa_reviews (review_id, edition_id, factor_id, status, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (rid, edition_id, factor_id, status, json.dumps(payload, sort_keys=True, default=str)),
    )
    return rid
