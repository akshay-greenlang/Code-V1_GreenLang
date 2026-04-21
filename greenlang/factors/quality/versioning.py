# -*- coding: utf-8 -*-
"""
Per-factor version chain (Phase F6).

Non-negotiable #2: every factor change creates an immutable new version.
The version chain is an append-only linked list: each new version
references its predecessor by ``previous_version_hash`` and is anchored
by a SHA-256 chain hash.

Chain-hash formula::

    chain_hash = sha256(
        factor_id + factor_version + content_hash + previous_chain_hash
    )

The chain is also surfaced to callers via the Explain endpoint and
backs the per-factor rollback feature in :mod:`rollback`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class VersionEntry:
    """One row in the per-factor version chain."""

    factor_id: str
    factor_version: str
    content_hash: str
    previous_version: Optional[str]          # factor_version of the predecessor
    previous_chain_hash: Optional[str]
    chain_hash: str
    changed_by: str
    change_reason: str
    migration_notes: Optional[str]
    deprecation_message: Optional[str]
    recorded_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "content_hash": self.content_hash,
            "previous_version": self.previous_version,
            "previous_chain_hash": self.previous_chain_hash,
            "chain_hash": self.chain_hash,
            "changed_by": self.changed_by,
            "change_reason": self.change_reason,
            "migration_notes": self.migration_notes,
            "deprecation_message": self.deprecation_message,
            "recorded_at": self.recorded_at,
        }


def compute_chain_hash(
    *,
    factor_id: str,
    factor_version: str,
    content_hash: str,
    previous_chain_hash: Optional[str],
) -> str:
    """SHA-256 over the chain-identifying fields."""
    payload = json.dumps(
        {
            "factor_id": factor_id,
            "factor_version": factor_version,
            "content_hash": content_hash,
            "previous": previous_chain_hash or "",
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# SQLite backend with append-only trigger
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_version_chain (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id             TEXT NOT NULL,
    factor_version        TEXT NOT NULL,
    content_hash          TEXT NOT NULL,
    previous_version      TEXT,
    previous_chain_hash   TEXT,
    chain_hash            TEXT NOT NULL,
    changed_by            TEXT NOT NULL,
    change_reason         TEXT NOT NULL,
    migration_notes       TEXT,
    deprecation_message   TEXT,
    recorded_at           TEXT NOT NULL,
    UNIQUE (factor_id, factor_version)
);
CREATE INDEX IF NOT EXISTS idx_fvc_factor ON factor_version_chain (factor_id, id);
CREATE INDEX IF NOT EXISTS idx_fvc_chain  ON factor_version_chain (chain_hash);

-- Append-only: forbid any UPDATE or DELETE once a row is written.
CREATE TRIGGER IF NOT EXISTS trg_fvc_no_update
BEFORE UPDATE ON factor_version_chain
BEGIN
    SELECT RAISE(ABORT, 'factor_version_chain is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_fvc_no_delete
BEFORE DELETE ON factor_version_chain
BEGIN
    SELECT RAISE(ABORT, 'factor_version_chain is append-only');
END;
"""


class VersioningError(RuntimeError):
    pass


class FactorVersionChain:
    """Thread-safe SQLite-backed per-factor version chain.

    Identical pattern to Phase 2.1 Climate Ledger's ``_SQLiteLedgerBackend``
    — WAL, autocommit, no-update / no-delete triggers.
    """

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Append + read
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        factor_id: str,
        factor_version: str,
        content_hash: str,
        changed_by: str,
        change_reason: str,
        migration_notes: Optional[str] = None,
        deprecation_message: Optional[str] = None,
    ) -> VersionEntry:
        """Append a new version row.  Raises if (factor_id, version) exists."""
        with self._lock:
            prev_row = self._conn.execute(
                """
                SELECT factor_version, chain_hash FROM factor_version_chain
                WHERE factor_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (factor_id,),
            ).fetchone()
            prev_version = prev_row[0] if prev_row else None
            prev_chain_hash = prev_row[1] if prev_row else None

            chain_hash = compute_chain_hash(
                factor_id=factor_id,
                factor_version=factor_version,
                content_hash=content_hash,
                previous_chain_hash=prev_chain_hash,
            )
            recorded_at = datetime.now(timezone.utc).isoformat()
            try:
                self._conn.execute(
                    """
                    INSERT INTO factor_version_chain (
                        factor_id, factor_version, content_hash,
                        previous_version, previous_chain_hash, chain_hash,
                        changed_by, change_reason, migration_notes,
                        deprecation_message, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        factor_id, factor_version, content_hash,
                        prev_version, prev_chain_hash, chain_hash,
                        changed_by, change_reason, migration_notes,
                        deprecation_message, recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise VersioningError(
                    "version %r already exists for %r" % (factor_version, factor_id)
                ) from exc

        return VersionEntry(
            factor_id=factor_id,
            factor_version=factor_version,
            content_hash=content_hash,
            previous_version=prev_version,
            previous_chain_hash=prev_chain_hash,
            chain_hash=chain_hash,
            changed_by=changed_by,
            change_reason=change_reason,
            migration_notes=migration_notes,
            deprecation_message=deprecation_message,
            recorded_at=recorded_at,
        )

    def chain(self, factor_id: str) -> List[VersionEntry]:
        """Return the ordered list of versions for ``factor_id``."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT factor_id, factor_version, content_hash, previous_version,
                       previous_chain_hash, chain_hash, changed_by, change_reason,
                       migration_notes, deprecation_message, recorded_at
                FROM factor_version_chain
                WHERE factor_id = ?
                ORDER BY id ASC
                """,
                (factor_id,),
            ).fetchall()
        return [VersionEntry(*row) for row in rows]

    def verify_chain(self, factor_id: str) -> bool:
        """Recompute every hash in the chain; return True if intact."""
        prev_hash: Optional[str] = None
        for entry in self.chain(factor_id):
            expected = compute_chain_hash(
                factor_id=entry.factor_id,
                factor_version=entry.factor_version,
                content_hash=entry.content_hash,
                previous_chain_hash=prev_hash,
            )
            if expected != entry.chain_hash:
                return False
            prev_hash = entry.chain_hash
        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # GAP-5 additions: rollback-aware helpers
    #
    # These read-only additions surface the version chain to the
    # rollback workflow without touching the append-only invariants.
    # ------------------------------------------------------------------

    def get_version_chain(self, factor_id: str) -> List[VersionEntry]:
        """Alias for :meth:`chain` — preferred name used by rollback.py."""
        return self.chain(factor_id)

    def get_version_entry(
        self, factor_id: str, factor_version: str
    ) -> Optional[VersionEntry]:
        """Return a specific version entry, or ``None`` if absent."""
        for entry in self.chain(factor_id):
            if entry.factor_version == factor_version:
                return entry
        return None

    def is_rollback_available(
        self, factor_id: str, factor_version: str
    ) -> bool:
        """True when ``factor_version`` exists *earlier* in the chain.

        A version is eligible for rollback iff a later version has
        superseded it — i.e. it is not the current head.
        """
        entries = self.chain(factor_id)
        if not entries:
            return False
        target = next(
            (e for e in entries if e.factor_version == factor_version), None
        )
        if target is None:
            return False
        head = entries[-1]
        return head.factor_version != factor_version

    def mark_rollback_available(
        self, factor_id: str, factor_version: str
    ) -> Dict[str, Any]:
        """Rollback-eligibility hook consulted by the rollback service.

        The hook is intentionally a pure read: it reports whether the
        named version can be used as a rollback target and surfaces the
        current head for the caller's plan UI.  It does **not** mutate
        the chain — that remains append-only per CTO non-negotiable #2.
        """
        entries = self.chain(factor_id)
        if not entries:
            return {
                "factor_id": factor_id,
                "factor_version": factor_version,
                "available": False,
                "reason": "no version chain",
                "current_version": None,
            }
        head = entries[-1]
        available = self.is_rollback_available(factor_id, factor_version)
        reason = "ok" if available else (
            "target is current head" if head.factor_version == factor_version
            else "version not in chain"
        )
        return {
            "factor_id": factor_id,
            "factor_version": factor_version,
            "available": available,
            "reason": reason,
            "current_version": head.factor_version,
            "current_content_hash": head.content_hash,
        }


__all__ = [
    "FactorVersionChain",
    "VersionEntry",
    "VersioningError",
    "compute_chain_hash",
]
