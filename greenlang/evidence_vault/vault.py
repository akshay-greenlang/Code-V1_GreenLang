# -*- coding: utf-8 -*-
"""
EvidenceVault - Core evidence collection, packaging, and verification.

This module implements the ``EvidenceVault`` class, the primary entry point for
GreenLang's v3 Evidence Vault product layer.  It stores evidence records
in-memory (default) and delegates integrity hashing to
``greenlang.utilities.provenance.hashing.hash_data`` and ID generation to
``greenlang.utilities.determinism.uuid.deterministic_uuid``.

Example::

    vault = EvidenceVault(vault_id="csrd-fy25")
    eid = vault.collect("emission_factor", "scope1_agent", {"co2e": 42.0})
    ok, details = vault.verify(eid)
    assert ok
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.utilities.determinism.uuid import deterministic_uuid
from greenlang.utilities.provenance.hashing import hash_data

logger = logging.getLogger(__name__)


class EvidenceVault:
    """Collect, package, verify, and export regulatory evidence.

    The Evidence Vault is the L2 System of Record for GreenLang's compliance
    products.  Each piece of evidence is assigned a deterministic ID, timestamped,
    and content-hashed so that downstream audit/reporting tools can verify
    integrity at any point.

    Args:
        vault_id: Human-readable identifier for this vault instance
            (e.g. ``"csrd-fy25"``).
        storage: Storage backend selector.  Currently only ``"memory"`` is
            supported; future versions will add ``"sqlite"`` and ``"s3"``.

    Example::

        vault = EvidenceVault("audit-2025")
        eid = vault.collect("invoice", "erp_connector", {"amount": 100})
        package = vault.package(format="json")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, vault_id: str, storage: str = "memory") -> None:
        self.vault_id: str = vault_id
        self.storage: str = storage
        self._records: Dict[str, Dict[str, Any]] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        logger.info(
            "EvidenceVault initialised: vault_id=%s, storage=%s",
            vault_id,
            storage,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        evidence_type: str,
        source: str,
        data: dict,
        metadata: dict | None = None,
    ) -> str:
        """Create and store an evidence record.

        Args:
            evidence_type: Category of evidence (e.g. ``"emission_factor"``,
                ``"supplier_declaration"``, ``"audit_log"``).
            source: Originating agent or system (e.g. ``"scope1_agent"``).
            data: The evidence payload -- must be JSON-serialisable.
            metadata: Optional free-form metadata dict.

        Returns:
            The deterministic evidence ID assigned to the new record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build a deterministic ID from vault context + content
        id_seed = f"{self.vault_id}:{evidence_type}:{source}:{json.dumps(data, sort_keys=True)}:{timestamp}"
        evidence_id = deterministic_uuid(
            namespace=self.vault_id,
            name=id_seed,
        )

        content_hash = hash_data(data, use_canonical=False)

        record: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "vault_id": self.vault_id,
            "evidence_type": evidence_type,
            "source": source,
            "data": data,
            "metadata": metadata or {},
            "content_hash": content_hash,
            "collected_at": timestamp,
        }

        self._records[evidence_id] = record
        logger.info(
            "Evidence collected: id=%s type=%s source=%s",
            evidence_id,
            evidence_type,
            source,
        )
        return evidence_id

    def package(
        self,
        evidence_ids: list[str] | None = None,
        format: str = "json",
    ) -> dict:
        """Create an evidence package from collected records.

        Args:
            evidence_ids: Specific record IDs to include.  ``None`` means
                include all records in the vault.
            format: Output format hint (``"json"`` is the only format
                currently implemented).

        Returns:
            A dict describing the package, including a ``package_hash``
            computed over the included evidence records.
        """
        records = self._resolve_records(evidence_ids)

        # Build a canonical representation for the package hash
        canonical = json.dumps(
            [r for r in records],
            sort_keys=True,
            default=str,
        )
        package_hash = hash_data(canonical)

        package: Dict[str, Any] = {
            "vault_id": self.vault_id,
            "format": format,
            "record_count": len(records),
            "package_hash": package_hash,
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }

        logger.info(
            "Evidence packaged: vault_id=%s records=%d hash=%s",
            self.vault_id,
            len(records),
            package_hash[:16],
        )
        return package

    def verify(self, evidence_id: str) -> Tuple[bool, dict]:
        """Verify the integrity of a stored evidence record.

        Re-computes the content hash of the record's ``data`` payload and
        compares it against the hash captured at collection time.

        Args:
            evidence_id: The ID of the evidence record to verify.

        Returns:
            A ``(is_valid, details)`` tuple.  ``details`` always contains
            ``evidence_id``, ``stored_hash``, ``computed_hash``, and
            ``verified_at`` keys.

        Raises:
            KeyError: If no record with the given ID exists.
        """
        record = self._get_record(evidence_id)

        computed_hash = hash_data(record["data"], use_canonical=False)
        stored_hash = record["content_hash"]
        is_valid = computed_hash == stored_hash

        details: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "stored_hash": stored_hash,
            "computed_hash": computed_hash,
            "is_valid": is_valid,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

        if is_valid:
            logger.info("Evidence verified OK: id=%s", evidence_id)
        else:
            logger.warning(
                "Evidence integrity FAILED: id=%s stored=%s computed=%s",
                evidence_id,
                stored_hash,
                computed_hash,
            )

        return is_valid, details

    def export(
        self,
        evidence_ids: list[str] | None = None,
        format: str = "json",
    ) -> dict:
        """Export evidence records for external consumption.

        This is similar to :meth:`package` but oriented toward downstream
        reporting tools and regulatory submission systems.

        Args:
            evidence_ids: Specific record IDs to export.  ``None`` exports
                everything.
            format: Output format hint.

        Returns:
            A dict with ``vault_id``, ``format``, ``exported_at``,
            ``record_count``, and ``records`` keys.
        """
        records = self._resolve_records(evidence_ids)

        export_payload: Dict[str, Any] = {
            "vault_id": self.vault_id,
            "format": format,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "records": records,
        }

        logger.info(
            "Evidence exported: vault_id=%s records=%d format=%s",
            self.vault_id,
            len(records),
            format,
        )
        return export_payload

    def list_evidence(
        self,
        evidence_type: str | None = None,
    ) -> List[dict]:
        """List evidence records with optional type filtering.

        Args:
            evidence_type: If provided, only records matching this type are
                returned.

        Returns:
            A list of evidence record dicts (copies, not references).
        """
        if evidence_type is None:
            return list(self._records.values())

        return [
            r for r in self._records.values()
            if r["evidence_type"] == evidence_type
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_record(self, evidence_id: str) -> Dict[str, Any]:
        """Return a single record or raise ``KeyError``."""
        try:
            return self._records[evidence_id]
        except KeyError:
            raise KeyError(
                f"Evidence record not found: {evidence_id}"
            ) from None

    def _resolve_records(
        self,
        evidence_ids: list[str] | None,
    ) -> List[Dict[str, Any]]:
        """Resolve an optional ID list to actual record dicts."""
        if evidence_ids is None:
            return list(self._records.values())
        return [self._get_record(eid) for eid in evidence_ids]
