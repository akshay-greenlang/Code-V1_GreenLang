# -*- coding: utf-8 -*-
"""
Audit logging for connector operations (F060).

Records all connector API calls, factor fetches, and errors for
compliance and debugging. Entries are immutable once written.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectorAuditEntry:
    """A single audit log entry for a connector operation."""

    connector_id: str
    operation: str  # "fetch_metadata", "fetch_factors", "health_check"
    timestamp: str
    tenant_id: str = "default"
    license_key_hash: Optional[str] = None
    request_factor_count: int = 0
    response_factor_count: int = 0
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectorAuditLog:
    """
    Append-only audit log for connector operations.

    Stores entries in SQLite (local) or can be extended to PostgreSQL.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._entries: List[ConnectorAuditEntry] = []
        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS connector_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                connector_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                license_key_hash TEXT,
                request_factor_count INTEGER DEFAULT 0,
                response_factor_count INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                success BOOLEAN NOT NULL DEFAULT 1,
                error TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cal_connector_ts "
            "ON connector_audit_log (connector_id, timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cal_tenant_ts "
            "ON connector_audit_log (tenant_id, timestamp DESC)"
        )
        conn.commit()
        conn.close()

    def log(self, entry: ConnectorAuditEntry) -> None:
        """Append an audit entry."""
        self._entries.append(entry)
        if self._db_path:
            self._persist(entry)
        logger.debug(
            "Connector audit: %s %s success=%s latency=%dms factors=%d",
            entry.connector_id, entry.operation, entry.success,
            entry.latency_ms, entry.response_factor_count,
        )

    def _persist(self, entry: ConnectorAuditEntry) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT INTO connector_audit_log (
                connector_id, operation, timestamp, tenant_id,
                license_key_hash, request_factor_count, response_factor_count,
                latency_ms, success, error, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.connector_id,
                entry.operation,
                entry.timestamp,
                entry.tenant_id,
                entry.license_key_hash,
                entry.request_factor_count,
                entry.response_factor_count,
                entry.latency_ms,
                int(entry.success),
                entry.error,
                json.dumps(entry.metadata, default=str),
            ),
        )
        conn.commit()
        conn.close()

    def query(
        self,
        *,
        connector_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        operation: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> List[ConnectorAuditEntry]:
        """Query audit entries from in-memory log."""
        results = list(self._entries)
        if connector_id:
            results = [e for e in results if e.connector_id == connector_id]
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if operation:
            results = [e for e in results if e.operation == operation]
        if success_only:
            results = [e for e in results if e.success]
        return results[-limit:]

    @property
    def entries(self) -> List[ConnectorAuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
