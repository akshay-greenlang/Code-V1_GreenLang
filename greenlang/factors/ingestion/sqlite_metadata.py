# -*- coding: utf-8 -*-
"""SQLite helpers for raw_artifacts + ingest_runs (D4)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Dict, Optional


def insert_raw_artifact(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    sha256: str,
    storage_uri: str,
    bytes_size: int,
    url: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    aid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO raw_artifacts (artifact_id, source_id, url, content_type, sha256, bytes_size, storage_uri)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (aid, source_id, url, content_type, sha256, bytes_size, storage_uri),
    )
    return aid


def insert_ingest_run(
    conn: sqlite3.Connection,
    *,
    artifact_id: Optional[str],
    edition_id: Optional[str],
    parser_id: str,
    status: str,
    row_counts: Dict[str, Any],
    owner: str = "",
    error: str = "",
) -> str:
    rid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ingest_runs (run_id, artifact_id, edition_id, parser_id, status, row_counts_json, owner, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rid,
            artifact_id,
            edition_id,
            parser_id,
            status,
            json.dumps(row_counts, sort_keys=True),
            owner,
            error,
        ),
    )
    return rid


def upsert_factor_lineage(
    conn: sqlite3.Connection,
    edition_id: str,
    factor_id: str,
    artifact_id: Optional[str],
    ingest_run_id: Optional[str],
    lineage: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO factor_lineage (edition_id, factor_id, artifact_id, ingest_run_id, lineage_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(edition_id, factor_id) DO UPDATE SET
            artifact_id=excluded.artifact_id,
            ingest_run_id=excluded.ingest_run_id,
            lineage_json=excluded.lineage_json
        """,
        (
            edition_id,
            factor_id,
            artifact_id,
            ingest_run_id,
            json.dumps(lineage, sort_keys=True, default=str),
        ),
    )
