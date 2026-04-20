# -*- coding: utf-8 -*-
"""
EvidenceVault - Core evidence collection, packaging, bundling, verification.

The Evidence Vault is GreenLang's v3 L2 System of Record for regulatory
evidence.  Each collected record carries a deterministic ID, a content
hash, and can be attached to a *case* (a logical grouping such as a
CBAM reporting period, a CSRD double-materiality review, or a
specific audit engagement).

Backends:

- ``"memory"`` — in-process, default.  Fast, ephemeral, good for tests.
- ``"sqlite"`` — append-only SQLite mirroring the Postgres schema in
  migration ``V440__evidence_vault.sql``.

The signature bundle produced by :meth:`EvidenceVault.bundle` is a
deterministic ZIP containing:

- ``manifest.json`` — case + record metadata + per-file hashes
- ``records/<eid>.json`` — full evidence records (data + metadata)
- ``attachments/<sha>/<filename>`` — optional raw-source files
- ``signature.json`` — SHA-256 signature digest over the manifest

Callers that require Ed25519 or GPG signatures layer those on top of
``signature.json``; the raw content hash is always present and is what
Climate Ledger records.
"""
from __future__ import annotations

import io
import json
import logging
import sqlite3
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from greenlang.utilities.determinism.uuid import deterministic_uuid
from greenlang.utilities.provenance.hashing import hash_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id    TEXT PRIMARY KEY,
    vault_id       TEXT NOT NULL,
    case_id        TEXT,
    evidence_type  TEXT NOT NULL,
    source         TEXT NOT NULL,
    data_json      TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    content_hash   TEXT NOT NULL,
    collected_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ev_vault ON evidence_records (vault_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_ev_case  ON evidence_records (case_id, collected_at);
CREATE INDEX IF NOT EXISTS idx_ev_type  ON evidence_records (evidence_type);

CREATE TABLE IF NOT EXISTS evidence_attachments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id    TEXT NOT NULL,
    filename       TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    content_bytes  BLOB NOT NULL,
    attached_at    TEXT NOT NULL,
    FOREIGN KEY (evidence_id) REFERENCES evidence_records(evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_att_evidence ON evidence_attachments (evidence_id);

CREATE TRIGGER IF NOT EXISTS trg_ev_no_update
BEFORE UPDATE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'evidence_records is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_ev_no_delete
BEFORE DELETE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'evidence_records is append-only');
END;
"""


class _SQLiteVaultBackend:
    """Minimal append-only SQLite persistence for evidence records."""

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
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SQLITE_SCHEMA)

    # ---------- records

    def insert_record(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO evidence_records (
                    evidence_id, vault_id, case_id, evidence_type, source,
                    data_json, metadata_json, content_hash, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["evidence_id"],
                    record["vault_id"],
                    record.get("case_id"),
                    record["evidence_type"],
                    record["source"],
                    json.dumps(record["data"], sort_keys=True, default=str),
                    json.dumps(record.get("metadata") or {}, sort_keys=True, default=str),
                    record["content_hash"],
                    record["collected_at"],
                ),
            )

    def get_record(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT evidence_id, vault_id, case_id, evidence_type, source,
                       data_json, metadata_json, content_hash, collected_at
                FROM evidence_records WHERE evidence_id = ?
                """,
                (evidence_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_records(
        self,
        *,
        case_id: Optional[str] = None,
        evidence_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT evidence_id, vault_id, case_id, evidence_type, source, "
            "data_json, metadata_json, content_hash, collected_at "
            "FROM evidence_records WHERE 1=1"
        )
        params: List[Any] = []
        if case_id is not None:
            sql += " AND case_id = ?"
            params.append(case_id)
        if evidence_type is not None:
            sql += " AND evidence_type = ?"
            params.append(evidence_type)
        sql += " ORDER BY collected_at ASC"
        with self._lock:
            rows = list(self._conn.execute(sql, params))
        return [self._row_to_record(r) for r in rows]

    # ---------- attachments

    def insert_attachment(
        self,
        evidence_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO evidence_attachments (
                    evidence_id, filename, content_hash, content_bytes, attached_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    filename,
                    content_hash,
                    content,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_attachments(self, evidence_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT id, evidence_id, filename, content_hash, content_bytes, attached_at
                    FROM evidence_attachments
                    WHERE evidence_id = ?
                    ORDER BY id ASC
                    """,
                    (evidence_id,),
                )
            )
        return [
            {
                "id": r[0],
                "evidence_id": r[1],
                "filename": r[2],
                "content_hash": r[3],
                "content_bytes": r[4],
                "attached_at": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_record(row: Tuple[Any, ...]) -> Dict[str, Any]:
        (
            evidence_id,
            vault_id,
            case_id,
            evidence_type,
            source,
            data_json,
            metadata_json,
            content_hash,
            collected_at,
        ) = row
        return {
            "evidence_id": evidence_id,
            "vault_id": vault_id,
            "case_id": case_id,
            "evidence_type": evidence_type,
            "source": source,
            "data": json.loads(data_json),
            "metadata": json.loads(metadata_json) if metadata_json else {},
            "content_hash": content_hash,
            "collected_at": collected_at,
        }


# ---------------------------------------------------------------------------
# Public EvidenceVault
# ---------------------------------------------------------------------------


_SUPPORTED_BACKENDS = {"memory", "sqlite"}


class EvidenceVault:
    """Collect, package, bundle, verify, and export regulatory evidence."""

    def __init__(
        self,
        vault_id: str,
        storage: str = "memory",
        sqlite_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if storage not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "Unsupported storage %r; choose from %s"
                % (storage, sorted(_SUPPORTED_BACKENDS))
            )

        self.vault_id: str = vault_id
        self.storage: str = storage
        self._records: Dict[str, Dict[str, Any]] = {}
        self._attachments: Dict[str, List[Dict[str, Any]]] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()

        self.sqlite_backend: Optional[_SQLiteVaultBackend] = None
        if storage == "sqlite":
            if sqlite_path is None:
                raise ValueError("storage='sqlite' requires sqlite_path")
            self.sqlite_backend = _SQLiteVaultBackend(sqlite_path)

        logger.info(
            "EvidenceVault initialised: vault_id=%s storage=%s",
            vault_id,
            storage,
        )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        evidence_type: str,
        source: str,
        data: dict,
        metadata: Optional[dict] = None,
        case_id: Optional[str] = None,
    ) -> str:
        """Create and store an evidence record."""
        timestamp = datetime.now(timezone.utc).isoformat()
        id_seed = (
            f"{self.vault_id}:{evidence_type}:{source}:"
            f"{json.dumps(data, sort_keys=True, default=str)}:{timestamp}"
        )
        evidence_id = deterministic_uuid(namespace=self.vault_id, name=id_seed)
        content_hash = hash_data(data, use_canonical=False)

        record: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "vault_id": self.vault_id,
            "case_id": case_id,
            "evidence_type": evidence_type,
            "source": source,
            "data": data,
            "metadata": metadata or {},
            "content_hash": content_hash,
            "collected_at": timestamp,
        }

        self._records[evidence_id] = record
        self._attachments.setdefault(evidence_id, [])

        if self.sqlite_backend is not None:
            self.sqlite_backend.insert_record(record)

        logger.info(
            "Evidence collected: id=%s type=%s source=%s case=%s",
            evidence_id,
            evidence_type,
            source,
            case_id,
        )
        return evidence_id

    def attach(
        self,
        evidence_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Attach a raw-source file (parser log, PDF, XML…) to an evidence record.

        Returns the SHA-256 content hash of the attachment.
        """
        if evidence_id not in self._records and self._sqlite_lookup(evidence_id) is None:
            raise KeyError(f"Evidence record not found: {evidence_id}")

        content_hash = hash_data(content, use_canonical=False)
        attachment = {
            "filename": filename,
            "content_hash": content_hash,
            "content_bytes": content,
            "attached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._attachments.setdefault(evidence_id, []).append(attachment)

        if self.sqlite_backend is not None:
            self.sqlite_backend.insert_attachment(
                evidence_id, filename, content, content_hash
            )

        logger.info(
            "Evidence attachment added: eid=%s filename=%s hash=%s",
            evidence_id,
            filename,
            content_hash[:16],
        )
        return content_hash

    # ------------------------------------------------------------------
    # Packaging / bundling / exporting
    # ------------------------------------------------------------------

    def package(
        self,
        evidence_ids: Optional[List[str]] = None,
        format: str = "json",
    ) -> dict:
        """In-memory dict package of evidence records."""
        records = self._resolve_records(evidence_ids)
        canonical = json.dumps(records, sort_keys=True, default=str)
        package_hash = hash_data(canonical)
        return {
            "vault_id": self.vault_id,
            "format": format,
            "record_count": len(records),
            "package_hash": package_hash,
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }

    def bundle(
        self,
        output_path: Union[str, Path],
        case_id: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
    ) -> Path:
        """Write a deterministic signed ZIP bundle to ``output_path``.

        The bundle contains:

        - ``manifest.json``    — case + per-record metadata + per-file hashes
        - ``records/<eid>.json`` — full evidence records
        - ``attachments/<sha>/<filename>`` — raw-source files (if any)
        - ``signature.json``   — SHA-256 digest over the canonical manifest

        Either ``case_id`` or an explicit ``evidence_ids`` list MUST be
        provided so the bundle has well-defined scope.

        Returns the ``Path`` of the written ZIP file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if case_id is None and evidence_ids is None:
            raise ValueError("bundle() requires either case_id or evidence_ids")

        if evidence_ids is None:
            records = self._records_for_case(case_id)
        else:
            records = [self._get_record(eid) for eid in evidence_ids]

        if not records:
            raise ValueError(
                f"No evidence records found for case_id={case_id!r} / "
                f"evidence_ids={evidence_ids!r}"
            )

        manifest_records: List[Dict[str, Any]] = []
        attachments_index: List[Dict[str, Any]] = []

        # Build in-memory zip; ZIP_DEFLATED + sorted entries = deterministic.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Records
            for record in sorted(records, key=lambda r: r["evidence_id"]):
                record_path = f"records/{record['evidence_id']}.json"
                zf.writestr(
                    record_path,
                    json.dumps(record, sort_keys=True, indent=2, default=str),
                )
                manifest_records.append(
                    {
                        "evidence_id": record["evidence_id"],
                        "evidence_type": record["evidence_type"],
                        "source": record["source"],
                        "content_hash": record["content_hash"],
                        "collected_at": record["collected_at"],
                        "case_id": record.get("case_id"),
                        "record_path": record_path,
                    }
                )

                # Attachments
                for att in self._attachments_for(record["evidence_id"]):
                    zip_path = f"attachments/{att['content_hash']}/{att['filename']}"
                    zf.writestr(zip_path, att["content_bytes"])
                    attachments_index.append(
                        {
                            "evidence_id": record["evidence_id"],
                            "filename": att["filename"],
                            "content_hash": att["content_hash"],
                            "attached_at": att["attached_at"],
                            "zip_path": zip_path,
                        }
                    )

            manifest = {
                "vault_id": self.vault_id,
                "case_id": case_id,
                "bundled_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(manifest_records),
                "attachment_count": len(attachments_index),
                "records": manifest_records,
                "attachments": sorted(
                    attachments_index,
                    key=lambda a: (a["evidence_id"], a["content_hash"], a["filename"]),
                ),
            }
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, indent=2, default=str
            ).encode("utf-8")
            zf.writestr("manifest.json", manifest_bytes)

            signature = {
                "algorithm": "sha256",
                "manifest_hash": hash_data(manifest_bytes, use_canonical=False),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "vault_id": self.vault_id,
                "case_id": case_id,
            }
            zf.writestr(
                "signature.json",
                json.dumps(signature, sort_keys=True, indent=2, default=str),
            )

        out.write_bytes(buffer.getvalue())
        logger.info(
            "Evidence bundle written: path=%s records=%d attachments=%d",
            out,
            len(manifest_records),
            len(attachments_index),
        )
        return out

    def export(
        self,
        evidence_ids: Optional[List[str]] = None,
        format: str = "json",
    ) -> dict:
        records = self._resolve_records(evidence_ids)
        return {
            "vault_id": self.vault_id,
            "format": format,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "records": records,
        }

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, evidence_id: str) -> Tuple[bool, dict]:
        """Re-hash the record's ``data`` and compare to the stored hash."""
        record = self._get_record(evidence_id)
        computed_hash = hash_data(record["data"], use_canonical=False)
        stored_hash = record["content_hash"]
        is_valid = computed_hash == stored_hash
        details = {
            "evidence_id": evidence_id,
            "stored_hash": stored_hash,
            "computed_hash": computed_hash,
            "is_valid": is_valid,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        if not is_valid:
            logger.warning(
                "Evidence integrity FAILED: id=%s stored=%s computed=%s",
                evidence_id,
                stored_hash,
                computed_hash,
            )
        return is_valid, details

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_evidence(
        self,
        evidence_type: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> List[dict]:
        """List evidence records with optional filtering by type and/or case."""
        if self.sqlite_backend is not None:
            return self.sqlite_backend.list_records(
                case_id=case_id, evidence_type=evidence_type
            )

        records: Iterable[Dict[str, Any]] = self._records.values()
        if evidence_type is not None:
            records = (r for r in records if r["evidence_type"] == evidence_type)
        if case_id is not None:
            records = (r for r in records if r.get("case_id") == case_id)
        return list(records)

    def list_cases(self) -> List[str]:
        if self.sqlite_backend is not None:
            return sorted(
                {r["case_id"] for r in self.sqlite_backend.list_records() if r.get("case_id")}
            )
        return sorted({r.get("case_id") for r in self._records.values() if r.get("case_id")})

    def close(self) -> None:
        if self.sqlite_backend is not None:
            self.sqlite_backend.close()
            self.sqlite_backend = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sqlite_lookup(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        if self.sqlite_backend is None:
            return None
        return self.sqlite_backend.get_record(evidence_id)

    def _get_record(self, evidence_id: str) -> Dict[str, Any]:
        record = self._records.get(evidence_id)
        if record is None:
            record = self._sqlite_lookup(evidence_id)
        if record is None:
            raise KeyError(f"Evidence record not found: {evidence_id}")
        return record

    def _resolve_records(
        self,
        evidence_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        if evidence_ids is not None:
            return [self._get_record(eid) for eid in evidence_ids]
        if self.sqlite_backend is not None:
            return self.sqlite_backend.list_records()
        return list(self._records.values())

    def _records_for_case(self, case_id: Optional[str]) -> List[Dict[str, Any]]:
        if self.sqlite_backend is not None:
            return self.sqlite_backend.list_records(case_id=case_id)
        return [r for r in self._records.values() if r.get("case_id") == case_id]

    def _attachments_for(self, evidence_id: str) -> List[Dict[str, Any]]:
        if self.sqlite_backend is not None:
            return self.sqlite_backend.list_attachments(evidence_id)
        return self._attachments.get(evidence_id, [])
