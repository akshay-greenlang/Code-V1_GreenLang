# -*- coding: utf-8 -*-
"""
Climate Ledger - Core Ledger Facade
=====================================

Wraps ``greenlang.data_commons.provenance.ProvenanceTracker`` and
``greenlang.utilities.provenance.ledger.write_run_ledger`` behind a clean
v3 product API for immutable audit trails.

The ``ClimateLedger`` class is the primary entry point for recording,
verifying, and exporting provenance chains within the Climate Ledger
product module.

Two storage backends are supported:

- ``"memory"`` -- in-process only (default, dev/test).
- ``"sqlite"`` -- append-only SQLite table mirroring every record
  emitted by the in-memory ``ProvenanceTracker``.  Schema matches the
  ``climate_ledger_entries`` table created by migration V439.

Postgres support is a thin shim over the same schema and is scheduled
for Phase 2.1 follow-up; today the ``"postgres"`` backend falls back
to the SQLite writer with a warning.

Example::

    >>> from greenlang.climate_ledger.ledger import ClimateLedger
    >>> ledger = ClimateLedger(
    ...     agent_name="scope1-calc",
    ...     storage_backend="sqlite",
    ...     sqlite_path="out/ledger.sqlite",
    ... )
    >>> chain_hash = ledger.record_entry("emission", "e-001", "calculate", "abc123")
    >>> valid, chain = ledger.verify("e-001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.data_commons.provenance import ProvenanceTracker
from greenlang.utilities.provenance.ledger import write_run_ledger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite backend (append-only, chain-hash indexed)
# ---------------------------------------------------------------------------


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS climate_ledger_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name    TEXT    NOT NULL,
    entity_type   TEXT    NOT NULL,
    entity_id     TEXT    NOT NULL,
    operation     TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL,
    chain_hash    TEXT    NOT NULL,
    metadata_json TEXT    NOT NULL DEFAULT '{}',
    recorded_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cle_entity
    ON climate_ledger_entries (entity_type, entity_id, id);
CREATE INDEX IF NOT EXISTS idx_cle_chain
    ON climate_ledger_entries (chain_hash);
CREATE INDEX IF NOT EXISTS idx_cle_agent
    ON climate_ledger_entries (agent_name, recorded_at);
-- An append-only trigger that forbids UPDATE/DELETE on existing rows.
-- SQLite lacks table-level immutability; this approximates it.
CREATE TRIGGER IF NOT EXISTS trg_cle_no_update
BEFORE UPDATE ON climate_ledger_entries
BEGIN
    SELECT RAISE(ABORT, 'climate_ledger_entries is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_cle_no_delete
BEFORE DELETE ON climate_ledger_entries
BEGIN
    SELECT RAISE(ABORT, 'climate_ledger_entries is append-only');
END;
"""


class _SQLiteLedgerBackend:
    """Append-only SQLite sink for Climate Ledger entries.

    The backend is intentionally minimal: it mirrors every record written
    by the in-memory ``ProvenanceTracker``.  The tracker owns the chain
    hash calculation; this class only persists it.
    """

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        # One connection per backend, guarded by a lock.  The ledger is
        # write-heavy-append-only so contention is low; a lock avoids
        # the SQLite "database is locked" error on concurrent writes.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,  # autocommit; we wrap in explicit transactions
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SQLITE_SCHEMA)

    def append(
        self,
        *,
        agent_name: str,
        entity_type: str,
        entity_id: str,
        operation: str,
        content_hash: str,
        chain_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a record.  Returns the inserted row id."""
        payload = json.dumps(metadata or {}, sort_keys=True, default=str)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO climate_ledger_entries (
                    agent_name, entity_type, entity_id, operation,
                    content_hash, chain_hash, metadata_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_name,
                    entity_type,
                    entity_id,
                    operation,
                    content_hash,
                    chain_hash,
                    payload,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def read_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT id, agent_name, entity_type, entity_id, operation,
                           content_hash, chain_hash, metadata_json, recorded_at
                    FROM climate_ledger_entries
                    WHERE entity_id = ?
                    ORDER BY id ASC
                    """,
                    (entity_id,),
                )
            )
        return [self._row_to_dict(r) for r in rows]

    def read_all(self, limit: int = 10_000) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT id, agent_name, entity_type, entity_id, operation,
                           content_hash, chain_hash, metadata_json, recorded_at
                    FROM climate_ledger_entries
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
            )
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            (value,) = self._conn.execute(
                "SELECT COUNT(*) FROM climate_ledger_entries"
            ).fetchone()
        return int(value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_dict(row: Tuple[Any, ...]) -> Dict[str, Any]:
        (
            row_id,
            agent_name,
            entity_type,
            entity_id,
            operation,
            content_hash,
            chain_hash,
            metadata_json,
            recorded_at,
        ) = row
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except (TypeError, ValueError):
            metadata = {}
        return {
            "id": row_id,
            "agent_name": agent_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "content_hash": content_hash,
            "chain_hash": chain_hash,
            "metadata": metadata,
            "recorded_at": recorded_at,
        }


# ---------------------------------------------------------------------------
# Public ClimateLedger facade
# ---------------------------------------------------------------------------


_SUPPORTED_BACKENDS = {"memory", "sqlite", "postgres"}


class ClimateLedger:
    """Unified Climate Ledger for immutable provenance tracking.

    Provides a product-grade API over the lower-level
    ``ProvenanceTracker`` (chain-hashing) and ``write_run_ledger``
    (JSONL run records) infrastructure.

    Attributes:
        agent_name: Identifier for the agent using this ledger instance.
        storage_backend: ``"memory"`` | ``"sqlite"`` | ``"postgres"``.
        tracker: The underlying ``ProvenanceTracker`` performing
            SHA-256 chain hashing.
        sqlite_backend: The SQLite sink, present when ``storage_backend``
            is ``"sqlite"`` (or when ``"postgres"`` falls back).

    Example::

        >>> ledger = ClimateLedger(
        ...     agent_name="ghg-inventory",
        ...     storage_backend="sqlite",
        ...     sqlite_path="out/ledger.sqlite",
        ... )
        >>> h = ledger.record_entry("facility", "f-042", "ingest", "deadbeef")
        >>> ok, chain = ledger.verify("f-042")
        >>> assert ok
    """

    def __init__(
        self,
        agent_name: str,
        storage_backend: str = "memory",
        sqlite_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if storage_backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "Unsupported storage_backend %r; choose from %s"
                % (storage_backend, sorted(_SUPPORTED_BACKENDS))
            )

        self.agent_name = agent_name
        self.storage_backend = storage_backend
        self.tracker = ProvenanceTracker(agent_name=agent_name)

        self.sqlite_backend: Optional[_SQLiteLedgerBackend] = None
        if storage_backend == "sqlite":
            if sqlite_path is None:
                raise ValueError(
                    "storage_backend='sqlite' requires sqlite_path to be set"
                )
            self.sqlite_backend = _SQLiteLedgerBackend(sqlite_path)
        elif storage_backend == "postgres":
            # Postgres writes go through the same schema (migration V439).
            # The connection pool isn't yet wired here -- callers that
            # truly need Postgres should initialize via a future
            # ClimateLedger.from_dsn() factory (tracked in Phase 2.1
            # follow-up).  We still honour the contract by falling back
            # to a best-effort SQLite sink so the CLI doesn't silently
            # drop records.
            logger.warning(
                "ClimateLedger postgres backend not yet wired; "
                "falling back to sqlite_path=%s",
                sqlite_path,
            )
            if sqlite_path is not None:
                self.sqlite_backend = _SQLiteLedgerBackend(sqlite_path)

        logger.info(
            "ClimateLedger initialized (agent=%s, backend=%s)",
            agent_name,
            storage_backend,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry in the ledger.

        Delegates the chain-hash computation to ``ProvenanceTracker.record()``
        and, if the SQLite backend is active, appends an immutable row to
        ``climate_ledger_entries``.

        Returns the chain hash (SHA-256 hex) linking this entry to the
        previous one.
        """
        user_id = "system"
        if metadata:
            user_id = json.dumps(metadata, sort_keys=True, default=str)

        chain_hash = self.tracker.record(
            entity_type=entity_type,
            entity_id=entity_id,
            action=operation,
            data_hash=content_hash,
            user_id=user_id,
        )

        if self.sqlite_backend is not None:
            self.sqlite_backend.append(
                agent_name=self.agent_name,
                entity_type=entity_type,
                entity_id=entity_id,
                operation=operation,
                content_hash=content_hash,
                chain_hash=chain_hash,
                metadata=metadata,
            )

        logger.debug(
            "Ledger entry recorded: %s/%s op=%s chain=%s",
            entity_type,
            entity_id,
            operation,
            chain_hash[:16],
        )
        return chain_hash

    def verify(
        self,
        entity_id: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of an entity's provenance chain.

        When a SQLite backend is active, the persisted rows are cross-checked
        against the in-memory chain so that tampering at rest is detected.
        """
        valid_in_memory, chain = self.tracker.verify_chain(entity_id)

        if self.sqlite_backend is None:
            return valid_in_memory, chain

        persisted = self.sqlite_backend.read_entity(entity_id)

        # Equal length + matching chain hashes in order == consistent.
        if len(persisted) != len(chain):
            logger.warning(
                "Ledger length mismatch: memory=%d sqlite=%d entity=%s",
                len(chain),
                len(persisted),
                entity_id,
            )
            return False, chain

        for mem_entry, disk_entry in zip(chain, persisted):
            mem_hash = mem_entry.get("chain_hash") or mem_entry.get("hash")
            if mem_hash != disk_entry["chain_hash"]:
                logger.warning(
                    "Ledger chain-hash drift at entity=%s (memory=%s sqlite=%s)",
                    entity_id,
                    mem_hash,
                    disk_entry["chain_hash"],
                )
                return False, chain

        return valid_in_memory, chain

    def export(
        self,
        entity_id: Optional[str] = None,
        format: str = "json",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Export provenance chain data.

        When *entity_id* is provided, returns that entity's chain as a list
        of entry dicts.  When omitted, returns the full global chain
        wrapped in a summary dict.  The SQLite backend is preferred when
        active; otherwise the in-memory tracker is used.
        """
        if format != "json":
            raise ValueError(
                "Unsupported export format %r; only 'json' is supported" % format
            )

        if entity_id is not None:
            if self.sqlite_backend is not None:
                return self.sqlite_backend.read_entity(entity_id)
            return self.tracker.get_chain(entity_id)

        if self.sqlite_backend is not None:
            entries = self.sqlite_backend.read_all(limit=10_000)
            return {
                "agent_name": self.agent_name,
                "entry_count": len(entries),
                "entity_count": len({e["entity_id"] for e in entries}),
                "entries": entries,
                "storage_backend": self.storage_backend,
            }

        return {
            "agent_name": self.agent_name,
            "entry_count": self.tracker.entry_count,
            "entity_count": self.tracker.entity_count,
            "entries": self.tracker.get_global_chain(limit=10_000),
            "storage_backend": self.storage_backend,
        }

    def write_run_record(
        self,
        result: Any,
        ctx: Any,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Write a deterministic JSONL run record to disk."""
        path_arg: Optional[Path] = None
        if output_path is not None:
            path_arg = Path(output_path)

        written = write_run_ledger(result, ctx, output_path=path_arg)
        logger.info("Run record written to %s", written)
        return written

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Total number of provenance entries across all entities."""
        if self.sqlite_backend is not None:
            return self.sqlite_backend.count()
        return self.tracker.entry_count

    @property
    def entity_count(self) -> int:
        """Number of unique entities tracked in this ledger."""
        if self.sqlite_backend is not None:
            return len({e["entity_id"] for e in self.sqlite_backend.read_all()})
        return self.tracker.entity_count

    def close(self) -> None:
        """Close any underlying connections.  Safe to call multiple times."""
        if self.sqlite_backend is not None:
            self.sqlite_backend.close()
            self.sqlite_backend = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "ClimateLedger(agent_name=%r, backend=%s, entries=%d, entities=%d)"
            % (self.agent_name, self.storage_backend, self.entry_count, self.entity_count)
        )
