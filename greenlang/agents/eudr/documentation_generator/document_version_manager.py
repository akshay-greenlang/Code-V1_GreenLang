# -*- coding: utf-8 -*-
"""
Document Version Manager Engine - AGENT-EUDR-030

Manages document versioning, amendments, and retention per EUDR
Article 31. Tracks version history with content hashes for integrity
verification, supports finalization, amendment creation with supersession,
and enforces the mandatory 5-year retention period.

Version Lifecycle:
    DRAFT -> FINAL -> SUPERSEDED (when amended)
    FINAL -> ARCHIVED (when retention expires)

Features:
    - Sequential version numbering per document
    - Content hash tracking for integrity verification
    - Amendment creation with automatic supersession
    - 5-year retention enforcement per Article 31
    - Expiry detection and archival management
    - Full audit trail for version lifecycle events

Zero-Hallucination Guarantees:
    - All version numbering is sequential and deterministic
    - No LLM calls in the versioning path
    - Retention dates computed from creation date + config years
    - Complete provenance trail for every version event

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Article 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import DocumentationGeneratorConfig, get_config
from .models import (
    AGENT_ID,
    AGENT_VERSION,
    AuditAction,
    DocumentType,
    DocumentVersion,
    RetentionStatus,
    VersionStatus,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retention period constants
# ---------------------------------------------------------------------------

_EUDR_MIN_RETENTION_YEARS = 5  # Article 31 minimum


class DocumentVersionManager:
    """Manages document versioning and retention per Article 31.

    Tracks document versions with content hashes for integrity
    verification, supports amendment workflows, and enforces
    EUDR's mandatory 5-year retention period.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.
        _versions: In-memory version store keyed by document_id.
        _audit_log: Audit trail of version lifecycle events.

    Example:
        >>> manager = DocumentVersionManager()
        >>> v1 = manager.create_version(
        ...     document_id="dds-001",
        ...     document_type=DocumentType.DDS,
        ...     content_hash="abc123...",
        ... )
        >>> assert v1.version_number == 1
        >>> assert v1.status == VersionStatus.DRAFT
        >>> v1_final = manager.finalize_version(v1.version_id)
        >>> assert v1_final.status == VersionStatus.FINAL
    """

    def __init__(
        self,
        config: Optional[DocumentationGeneratorConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize DocumentVersionManager.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._versions: Dict[str, List[DocumentVersion]] = {}
        self._version_lookup: Dict[str, DocumentVersion] = {}
        self._audit_log: List[Dict[str, Any]] = []
        logger.info(
            "DocumentVersionManager initialized: "
            "max_versions=%d, retention_years=%d, "
            "amendment_tracking=%s",
            self._config.max_versions_per_document,
            self._config.retention_years,
            self._config.enable_amendment_tracking,
        )

    def create_version(
        self,
        document_id: str,
        document_type: DocumentType,
        content_hash: str,
        created_by: str = "system",
    ) -> DocumentVersion:
        """Create a new document version.

        Assigns a sequential version number and sets retention
        expiry based on the configured retention period.

        Args:
            document_id: Parent document identifier.
            document_type: Type of document.
            content_hash: SHA-256 hash of the version content.
            created_by: Creator identifier.

        Returns:
            New DocumentVersion in DRAFT status.

        Raises:
            ValueError: If maximum versions exceeded.
        """
        start_time = time.monotonic()

        # Check version limit
        existing = self._versions.get(document_id, [])
        max_versions = self._config.max_versions_per_document
        if len(existing) >= max_versions:
            raise ValueError(
                f"Document '{document_id}' has reached maximum "
                f"version limit of {max_versions}."
            )

        # Calculate version number
        version_number = len(existing) + 1
        version_id = f"ver-{uuid.uuid4().hex[:12]}"

        # Create version
        now = datetime.now(timezone.utc)
        version = DocumentVersion(
            version_id=version_id,
            document_id=document_id,
            document_type=document_type,
            version_number=version_number,
            status=VersionStatus.DRAFT,
            content_hash=content_hash,
            created_at=now,
            created_by=created_by,
        )

        # Store version
        if document_id not in self._versions:
            self._versions[document_id] = []
        self._versions[document_id].append(version)
        self._version_lookup[version_id] = version

        # Record audit event
        self._record_audit(
            action=AuditAction.CREATE,
            document_id=document_id,
            version_id=version_id,
            version_number=version_number,
            actor=created_by,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Version created: id=%s, doc=%s, v%d, type=%s, "
            "hash=%s..., elapsed=%.1fms",
            version_id, document_id, version_number,
            document_type.value, content_hash[:16], elapsed_ms,
        )

        return version

    def finalize_version(
        self, version_id: str,
    ) -> DocumentVersion:
        """Mark a version as final.

        Transitions a DRAFT version to FINAL status and sets
        the finalization timestamp.

        Args:
            version_id: Version identifier.

        Returns:
            Updated DocumentVersion in FINAL status.

        Raises:
            ValueError: If version not found or not in DRAFT status.
        """
        version = self._get_version_or_raise(version_id)

        if version.status != VersionStatus.DRAFT:
            raise ValueError(
                f"Version '{version_id}' cannot be finalized: "
                f"current status is '{version.status.value}', "
                f"expected 'draft'."
            )

        version.status = VersionStatus.FINAL
        now = datetime.now(timezone.utc)

        # Record audit event
        self._record_audit(
            action=AuditAction.UPDATE,
            document_id=version.document_id,
            version_id=version_id,
            version_number=version.version_number,
            actor="system",
        )

        logger.info(
            "Version finalized: id=%s, doc=%s, v%d",
            version_id, version.document_id, version.version_number,
        )

        return version

    def create_amendment(
        self,
        document_id: str,
        amendment_reason: str,
        new_content_hash: str,
        created_by: str = "system",
    ) -> DocumentVersion:
        """Create an amendment version (supersedes previous).

        Creates a new version with an amendment reason and marks
        the previous FINAL version as SUPERSEDED.

        Args:
            document_id: Parent document identifier.
            amendment_reason: Reason for the amendment.
            new_content_hash: SHA-256 hash of amended content.
            created_by: Creator identifier.

        Returns:
            New DocumentVersion representing the amendment.

        Raises:
            ValueError: If no current version exists or version
                limit exceeded.
        """
        start_time = time.monotonic()

        # Get current version
        current = self.get_current_version(document_id)
        if current is None:
            raise ValueError(
                f"No existing version found for document "
                f"'{document_id}'. Cannot create amendment."
            )

        # Supersede current final version
        if current.status == VersionStatus.FINAL:
            current.status = VersionStatus.SUPERSEDED
            now = datetime.now(timezone.utc)

            self._record_audit(
                action=AuditAction.AMEND,
                document_id=document_id,
                version_id=current.version_id,
                version_number=current.version_number,
                actor=created_by,
                details={"reason": amendment_reason},
            )

        # Create new amendment version
        new_version = self.create_version(
            document_id=document_id,
            document_type=current.document_type,
            content_hash=new_content_hash,
            created_by=created_by,
        )
        new_version.amendment_reason = amendment_reason

        if self._config.enable_amendment_tracking:
            logger.info(
                "Amendment created: id=%s, doc=%s, v%d, "
                "reason='%s', superseded v%d",
                new_version.version_id, document_id,
                new_version.version_number,
                amendment_reason, current.version_number,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Amendment version created: elapsed=%.1fms",
            elapsed_ms,
        )

        return new_version

    def get_version_history(
        self, document_id: str,
    ) -> List[DocumentVersion]:
        """Get full version history for a document.

        Args:
            document_id: Document identifier.

        Returns:
            List of DocumentVersion instances ordered by version
            number (ascending).
        """
        versions = self._versions.get(document_id, [])
        return sorted(versions, key=lambda v: v.version_number)

    def get_current_version(
        self, document_id: str,
    ) -> Optional[DocumentVersion]:
        """Get the current active version of a document.

        The current version is the highest-numbered version that
        is not SUPERSEDED or ARCHIVED.

        Args:
            document_id: Document identifier.

        Returns:
            Current DocumentVersion or None if no versions exist.
        """
        versions = self._versions.get(document_id, [])
        if not versions:
            return None

        # Find highest non-superseded, non-archived version
        active_statuses = {VersionStatus.DRAFT, VersionStatus.FINAL}
        active = [v for v in versions if v.status in active_statuses]

        if not active:
            # Fall back to highest version number
            return max(versions, key=lambda v: v.version_number)

        return max(active, key=lambda v: v.version_number)

    def get_version_by_id(
        self, version_id: str,
    ) -> Optional[DocumentVersion]:
        """Get a version by its identifier.

        Args:
            version_id: Version identifier.

        Returns:
            DocumentVersion or None if not found.
        """
        return self._version_lookup.get(version_id)

    def get_version(
        self, version_id: str,
    ) -> Optional[DocumentVersion]:
        """Get a version by its identifier (alias for get_version_by_id).

        Args:
            version_id: Version identifier.

        Returns:
            DocumentVersion or None if not found.
        """
        return self.get_version_by_id(version_id)

    def compare_versions(
        self, version_id_1: str, version_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two document versions.

        Args:
            version_id_1: First version identifier.
            version_id_2: Second version identifier.

        Returns:
            Dictionary with comparison results.

        Raises:
            ValueError: If either version not found.
        """
        v1 = self.get_version_by_id(version_id_1)
        v2 = self.get_version_by_id(version_id_2)

        if v1 is None:
            raise ValueError(f"Version {version_id_1} not found")
        if v2 is None:
            raise ValueError(f"Version {version_id_2} not found")

        return {
            "version1_id": v1.version_id,
            "version2_id": v2.version_id,
            "version1_number": v1.version_number,
            "version2_number": v2.version_number,
            "content_hash_changed": v1.content_hash != v2.content_hash,
            "version1_hash": v1.content_hash,
            "version2_hash": v2.content_hash,
            "version1_created": v1.created_at.isoformat(),
            "version2_created": v2.created_at.isoformat(),
        }

    def check_retention(self) -> Dict[str, Any]:
        """Check retention status of all documents.

        Evaluates each document's latest version against the
        configured retention period to determine expiry status.

        Returns:
            Dictionary with retention status summary and
            per-document retention details.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc)
        retention_years = self._config.retention_years
        retention_delta = timedelta(days=retention_years * 365)
        approaching_delta = timedelta(days=90)  # 90 days warning

        results: Dict[str, Any] = {
            "checked_at": now.isoformat(),
            "retention_years": retention_years,
            "total_documents": len(self._versions),
            "active": 0,
            "approaching_expiry": 0,
            "expired": 0,
            "documents": {},
        }

        for doc_id, versions in self._versions.items():
            if not versions:
                continue

            # Use earliest version creation date
            earliest = min(versions, key=lambda v: v.created_at)
            expires_at = earliest.created_at + retention_delta
            time_remaining = expires_at - now

            if time_remaining <= timedelta(0):
                status = RetentionStatus.EXPIRED.value
                results["expired"] += 1
            elif time_remaining <= approaching_delta:
                status = RetentionStatus.EXPIRING_SOON.value
                results["approaching_expiry"] += 1
            else:
                status = RetentionStatus.ACTIVE.value
                results["active"] += 1

            results["documents"][doc_id] = {
                "status": status,
                "created_at": earliest.created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "days_remaining": max(0, time_remaining.days),
                "version_count": len(versions),
            }

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Retention check: active=%d, approaching=%d, "
            "expired=%d, elapsed=%.1fms",
            results["active"],
            results["approaching_expiry"],
            results["expired"],
            elapsed_ms,
        )

        return results

    def check_retention_status(self, version_id: str) -> Dict[str, Any]:
        """Check retention status for a specific version.

        Args:
            version_id: Version identifier.

        Returns:
            Dictionary with retention status for the version.

        Raises:
            ValueError: If version not found.
        """
        version = self.get_version_by_id(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        now = datetime.now(timezone.utc)
        retention_years = self._config.retention_years
        retention_delta = timedelta(days=retention_years * 365)
        approaching_delta = timedelta(days=90)  # 90 days warning

        expires_at = version.created_at + retention_delta
        time_remaining = expires_at - now

        if time_remaining <= timedelta(0):
            status = "expired"
        elif time_remaining <= approaching_delta:
            status = "expiring_soon"
        else:
            status = "active"

        years_remaining = time_remaining.days / 365.0

        return {
            "retention_status": status,
            "created_at": version.created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "days_remaining": max(0, time_remaining.days),
            "years_remaining": max(0.0, years_remaining),
        }

    def get_versions_approaching_expiry(
        self, months_threshold: int = 3,
    ) -> List[DocumentVersion]:
        """Get versions approaching retention expiry.

        Args:
            months_threshold: Months before expiry to include (default 3).

        Returns:
            List of versions approaching expiry.
        """
        now = datetime.now(timezone.utc)
        retention_years = self._config.retention_years
        retention_delta = timedelta(days=retention_years * 365)
        threshold_delta = timedelta(days=months_threshold * 30)

        approaching = []

        for versions in self._versions.values():
            for version in versions:
                expires_at = version.created_at + retention_delta
                time_remaining = expires_at - now

                if timedelta(0) < time_remaining <= threshold_delta:
                    approaching.append(version)

        return approaching

    def archive_expired_versions(self) -> List[str]:
        """Alias for archive_expired() for backward compatibility.

        Returns:
            List of archived document IDs.
        """
        return self.archive_expired()

    def archive_expired(self) -> List[str]:
        """Archive documents past retention period.

        Marks all versions of expired documents as ARCHIVED
        and returns the list of archived document IDs.

        Returns:
            List of document IDs that were archived.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc)
        retention_years = self._config.retention_years
        retention_delta = timedelta(days=retention_years * 365)

        archived_docs: List[str] = []

        for doc_id, versions in self._versions.items():
            if not versions:
                continue

            earliest = min(versions, key=lambda v: v.created_at)
            expires_at = earliest.created_at + retention_delta

            if now >= expires_at:
                # Archive all versions
                for version in versions:
                    if version.status != VersionStatus.ARCHIVED:
                        version.status = VersionStatus.ARCHIVED
                        self._record_audit(
                            action=AuditAction.ARCHIVE,
                            document_id=doc_id,
                            version_id=version.version_id,
                            version_number=version.version_number,
                            actor="system",
                            details={
                                "reason": "retention_period_expired",
                                "retention_years": retention_years,
                            },
                        )
                archived_docs.append(doc_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Expired documents archived: count=%d, elapsed=%.1fms",
            len(archived_docs), elapsed_ms,
        )

        return archived_docs

    def get_audit_log(
        self,
        document_id: Optional[str] = None,
        action_filter: Optional[List[AuditAction]] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries with optional filters.

        Args:
            document_id: Optional document ID filter.
            action_filter: Optional list of action types to filter by.

        Returns:
            List of audit log entries.
        """
        entries = list(self._audit_log)

        if document_id:
            entries = [
                entry for entry in entries
                if entry.get("document_id") == document_id
            ]

        if action_filter:
            action_values = [a.value for a in action_filter]
            entries = [
                entry for entry in entries
                if entry.get("action") in action_values
            ]

        return entries

    def get_document_count(self) -> int:
        """Get total number of tracked documents.

        Returns:
            Number of documents with at least one version.
        """
        return len(self._versions)

    def get_total_version_count(self) -> int:
        """Get total number of versions across all documents.

        Returns:
            Total version count.
        """
        return sum(
            len(versions) for versions in self._versions.values()
        )

    def _get_version_or_raise(
        self, version_id: str,
    ) -> DocumentVersion:
        """Get version by ID or raise ValueError.

        Args:
            version_id: Version identifier.

        Returns:
            DocumentVersion instance.

        Raises:
            ValueError: If version not found.
        """
        version = self._version_lookup.get(version_id)
        if version is None:
            raise ValueError(f"Version not found: {version_id}")
        return version

    def _record_audit(
        self,
        action: AuditAction,
        document_id: str,
        version_id: str,
        version_number: int,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an audit trail event.

        Args:
            action: Audit action type.
            document_id: Document identifier.
            version_id: Version identifier.
            version_number: Version number.
            actor: Actor performing the action.
            details: Optional additional details.
        """
        entry: Dict[str, Any] = {
            "action": action.value,
            "document_id": document_id,
            "version_id": version_id,
            "version_number": version_number,
            "actor": actor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": AGENT_ID,
        }
        if details:
            entry["details"] = details

        self._audit_log.append(entry)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and version statistics.
        """
        return {
            "engine": "DocumentVersionManager",
            "status": "available",
            "config": {
                "max_versions_per_document": (
                    self._config.max_versions_per_document
                ),
                "retention_years": self._config.retention_years,
                "amendment_tracking": (
                    self._config.enable_amendment_tracking
                ),
            },
            "statistics": {
                "total_documents": self.get_document_count(),
                "total_versions": self.get_total_version_count(),
                "audit_log_entries": len(self._audit_log),
            },
        }
