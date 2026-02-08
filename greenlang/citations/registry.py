# -*- coding: utf-8 -*-
"""
Citation Registry - AGENT-FOUND-005: Citations & Evidence

Core registry that manages citations with CRUD operations, filtering,
search, and versioning with full provenance tracking.

Integrates with:
    - VerificationEngine for citation verification
    - ProvenanceTracker for audit trails
    - Metrics for Prometheus observability
    - CitationsConfig for configuration

Zero-Hallucination Guarantees:
    - All values are explicitly stored, never inferred
    - Complete version history with SHA-256 provenance
    - Deterministic content hashing

Example:
    >>> from greenlang.citations.registry import CitationRegistry
    >>> registry = CitationRegistry()
    >>> citation = registry.create(
    ...     citation_type="emission_factor",
    ...     source_authority="defra",
    ...     metadata={"title": "DEFRA GHG Factors 2024"},
    ...     effective_date="2024-01-01",
    ...     user_id="analyst",
    ... )
    >>> print(registry.count)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.citations.config import CitationsConfig, get_config
from greenlang.citations.models import (
    ChangeLogEntry,
    ChangeType,
    Citation,
    CitationMetadata,
    CitationType,
    CitationVersion,
    RegulatoryFramework,
    SourceAuthority,
    VerificationStatus,
)
from greenlang.citations.provenance import ProvenanceTracker
from greenlang.citations.metrics import (
    record_operation,
    update_citations_count,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class CitationRegistry:
    """Core registry for managing citations.

    Provides CRUD operations, filtering, search, and version management
    with full provenance tracking and content hashing.

    Attributes:
        config: CitationsConfig instance.
        provenance: ProvenanceTracker instance.
        _citations: Internal storage of citations by ID.
        _versions: Version history per citation ID.
        _change_log: Ordered change log entries.

    Example:
        >>> registry = CitationRegistry()
        >>> c = registry.create(
        ...     citation_type="emission_factor",
        ...     source_authority="epa",
        ...     metadata={"title": "EPA eGRID 2024"},
        ...     effective_date="2024-01-01",
        ... )
        >>> print(c.citation_id)
    """

    def __init__(
        self,
        config: Optional[CitationsConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize CitationRegistry.

        Args:
            config: Optional config. Uses global config if None.
            provenance: Optional provenance tracker. Creates new one if None.
        """
        self.config = config or get_config()
        self.provenance = provenance or ProvenanceTracker()
        self._citations: Dict[str, Citation] = {}
        self._versions: Dict[str, List[CitationVersion]] = {}
        self._change_log: List[ChangeLogEntry] = []
        logger.info("CitationRegistry initialized")

    def create(
        self,
        citation_type: str,
        source_authority: str,
        metadata: Dict[str, Any],
        effective_date: str,
        user_id: str = "system",
        change_reason: str = "Initial creation",
        citation_id: Optional[str] = None,
        expiration_date: Optional[str] = None,
        supersedes: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Citation:
        """Create a new citation in the registry.

        Args:
            citation_type: Type of citation (e.g., "emission_factor").
            source_authority: Source authority (e.g., "defra").
            metadata: Citation metadata dictionary (must include "title").
            effective_date: ISO format date string (YYYY-MM-DD).
            user_id: User creating the citation.
            change_reason: Reason for creation.
            citation_id: Optional pre-assigned citation ID.
            expiration_date: Optional ISO format expiration date.
            supersedes: Optional citation ID this supersedes.
            regulatory_frameworks: Optional list of framework strings.
            abstract: Optional abstract text.
            key_values: Optional key values dictionary.
            notes: Optional notes text.

        Returns:
            Created Citation object.

        Raises:
            ValueError: If citation_id already exists or capacity exceeded.
        """
        start = time.monotonic()

        # Check capacity
        if len(self._citations) >= self.config.max_citations:
            raise ValueError(
                f"Registry at capacity ({self.config.max_citations} citations)"
            )

        # Build citation metadata
        citation_metadata = CitationMetadata(**metadata)

        # Parse enums
        ct = CitationType(citation_type)
        sa = SourceAuthority(source_authority)
        eff_date = date.fromisoformat(effective_date)
        exp_date = (
            date.fromisoformat(expiration_date)
            if expiration_date
            else None
        )
        frameworks = [
            RegulatoryFramework(fw)
            for fw in (regulatory_frameworks or [])
        ]

        # Build citation
        citation = Citation(
            citation_type=ct,
            source_authority=sa,
            metadata=citation_metadata,
            effective_date=eff_date,
            expiration_date=exp_date,
            supersedes=supersedes,
            regulatory_frameworks=frameworks,
            abstract=abstract,
            key_values=key_values or {},
            notes=notes,
            created_by=user_id,
        )

        # Allow caller to set citation_id
        if citation_id:
            if citation_id in self._citations:
                raise ValueError(f"Citation {citation_id} already exists")
            citation.citation_id = citation_id

        # Check for duplicate ID (auto-generated)
        if citation.citation_id in self._citations:
            raise ValueError(
                f"Citation {citation.citation_id} already exists"
            )

        # Calculate content hash
        citation.content_hash = citation.calculate_content_hash()

        # Handle supersession
        if supersedes and supersedes in self._citations:
            old_citation = self._citations[supersedes]
            old_citation.superseded_by = citation.citation_id
            old_citation.verification_status = VerificationStatus.SUPERSEDED
            old_citation.updated_at = _utcnow()

        # Store
        self._citations[citation.citation_id] = citation

        # Create initial version
        version = CitationVersion(
            version_number=1,
            citation_id=citation.citation_id,
            snapshot=citation.model_dump(mode="json"),
            created_by=user_id,
            change_reason=change_reason,
            change_type=ChangeType.CREATE,
            provenance_hash=self._compute_hash(
                {"citation_id": citation.citation_id, "version": 1},
            ),
        )
        self._versions[citation.citation_id] = [version]

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="citation",
                entity_id=citation.citation_id,
                action="create",
                data_hash=citation.content_hash,
                user_id=user_id,
                metadata={"change_reason": change_reason},
            )
            self._append_change_log(
                user_id=user_id,
                change_type=ChangeType.CREATE,
                entity_type="citation",
                entity_id=citation.citation_id,
                old_value=None,
                new_value=citation.metadata.title,
                change_reason=change_reason,
            )

        # Update metrics
        update_citations_count(len(self._citations))

        duration = time.monotonic() - start
        record_operation("create", "success", duration)
        logger.info("Created citation: %s", citation.citation_id)

        return citation

    def get(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by ID.

        Args:
            citation_id: The citation identifier.

        Returns:
            Citation or None if not found.
        """
        start = time.monotonic()
        result = self._citations.get(citation_id)
        duration = time.monotonic() - start
        record_operation("get", "success" if result else "not_found", duration)
        return result

    def update(
        self,
        citation_id: str,
        user_id: str = "system",
        reason: str = "Citation update",
        metadata: Optional[Dict[str, Any]] = None,
        expiration_date: Optional[str] = None,
        abstract: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
    ) -> Citation:
        """Update a citation's fields.

        Args:
            citation_id: The citation to update.
            user_id: User making the change.
            reason: Reason for the change.
            metadata: Optional new metadata fields to merge.
            expiration_date: Optional new expiration date.
            abstract: Optional new abstract.
            key_values: Optional new key values to merge.
            notes: Optional new notes.
            regulatory_frameworks: Optional new frameworks list.

        Returns:
            Updated Citation object.

        Raises:
            ValueError: If citation not found.
        """
        start = time.monotonic()

        citation = self._citations.get(citation_id)
        if citation is None:
            raise ValueError(f"Citation {citation_id} not found")

        old_hash = citation.content_hash

        # Apply updates
        if metadata is not None:
            merged = citation.metadata.model_dump()
            merged.update(metadata)
            citation.metadata = CitationMetadata(**merged)

        if expiration_date is not None:
            citation.expiration_date = date.fromisoformat(expiration_date)

        if abstract is not None:
            citation.abstract = abstract

        if key_values is not None:
            citation.key_values.update(key_values)

        if notes is not None:
            citation.notes = notes

        if regulatory_frameworks is not None:
            citation.regulatory_frameworks = [
                RegulatoryFramework(fw) for fw in regulatory_frameworks
            ]

        # Recalculate hash and timestamp
        citation.content_hash = citation.calculate_content_hash()
        citation.updated_at = _utcnow()

        # Create new version
        versions = self._versions.get(citation_id, [])
        new_version = CitationVersion(
            version_number=len(versions) + 1,
            citation_id=citation_id,
            snapshot=citation.model_dump(mode="json"),
            created_by=user_id,
            change_reason=reason,
            change_type=ChangeType.UPDATE,
            provenance_hash=self._compute_hash({
                "citation_id": citation_id,
                "version": len(versions) + 1,
                "previous_hash": old_hash,
            }),
            parent_version_id=(
                versions[-1].version_id if versions else None
            ),
        )
        versions.append(new_version)
        self._versions[citation_id] = versions

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="citation",
                entity_id=citation_id,
                action="update",
                data_hash=citation.content_hash,
                user_id=user_id,
                metadata={"change_reason": reason},
            )
            self._append_change_log(
                user_id=user_id,
                change_type=ChangeType.UPDATE,
                entity_type="citation",
                entity_id=citation_id,
                old_value=old_hash,
                new_value=citation.content_hash,
                change_reason=reason,
            )

        duration = time.monotonic() - start
        record_operation("update", "success", duration)
        logger.info("Updated citation: %s", citation_id)

        return citation

    def delete(
        self,
        citation_id: str,
        user_id: str = "system",
        reason: str = "Deletion",
    ) -> bool:
        """Delete a citation from the registry.

        Args:
            citation_id: The citation to delete.
            user_id: User performing the deletion.
            reason: Reason for deletion.

        Returns:
            True if deleted, False if not found.
        """
        start = time.monotonic()

        citation = self._citations.get(citation_id)
        if citation is None:
            record_operation("delete", "not_found", time.monotonic() - start)
            return False

        old_hash = citation.content_hash
        del self._citations[citation_id]

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="citation",
                entity_id=citation_id,
                action="delete",
                data_hash=old_hash or "",
                user_id=user_id,
                metadata={"change_reason": reason},
            )
            self._append_change_log(
                user_id=user_id,
                change_type=ChangeType.DELETE,
                entity_type="citation",
                entity_id=citation_id,
                old_value=old_hash,
                new_value=None,
                change_reason=reason,
            )

        # Update metrics
        update_citations_count(len(self._citations))

        duration = time.monotonic() - start
        record_operation("delete", "success", duration)
        logger.info("Deleted citation: %s", citation_id)

        return True

    def list(
        self,
        citation_type: Optional[str] = None,
        source_authority: Optional[str] = None,
        verification_status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Citation]:
        """List citations with optional filtering.

        Args:
            citation_type: Optional citation type filter.
            source_authority: Optional source authority filter.
            verification_status: Optional verification status filter.
            search: Optional search term for title/abstract.

        Returns:
            Filtered list of citations.
        """
        start = time.monotonic()
        citations = list(self._citations.values())

        if citation_type is not None:
            ct = CitationType(citation_type)
            citations = [c for c in citations if c.citation_type == ct]

        if source_authority is not None:
            sa = SourceAuthority(source_authority)
            citations = [c for c in citations if c.source_authority == sa]

        if verification_status is not None:
            vs = VerificationStatus(verification_status)
            citations = [
                c for c in citations if c.verification_status == vs
            ]

        if search is not None:
            term = search.lower()
            citations = [
                c for c in citations
                if (
                    term in c.metadata.title.lower()
                    or (c.abstract and term in c.abstract.lower())
                )
            ]

        duration = time.monotonic() - start
        record_operation("list", "success", duration)
        return citations

    def get_by_source(self, source_authority: str) -> List[Citation]:
        """Get all citations from a specific source authority.

        Args:
            source_authority: Source authority string value.

        Returns:
            List of citations from the specified authority.
        """
        sa = SourceAuthority(source_authority)
        return [
            c for c in self._citations.values()
            if c.source_authority == sa
        ]

    def get_by_framework(self, framework: str) -> List[Citation]:
        """Get all citations relevant to a regulatory framework.

        Args:
            framework: Regulatory framework string value.

        Returns:
            List of citations linked to the framework.
        """
        rf = RegulatoryFramework(framework)
        return [
            c for c in self._citations.values()
            if rf in c.regulatory_frameworks
        ]

    def get_valid(self, reference_date: Optional[date] = None) -> List[Citation]:
        """Get all currently valid citations.

        Args:
            reference_date: Date to check validity against. Uses today if None.

        Returns:
            List of valid citations.
        """
        return [
            c for c in self._citations.values()
            if c.is_valid(reference_date)
        ]

    def search(self, query: str) -> List[Citation]:
        """Search citations by title, abstract, and notes.

        Args:
            query: Search query string.

        Returns:
            List of matching citations.
        """
        start = time.monotonic()
        term = query.lower()
        results = [
            c for c in self._citations.values()
            if (
                term in c.metadata.title.lower()
                or (c.abstract and term in c.abstract.lower())
                or (c.notes and term in c.notes.lower())
                or any(
                    term in author.lower()
                    for author in c.metadata.authors
                )
            )
        ]
        duration = time.monotonic() - start
        record_operation("search", "success", duration)
        return results

    def get_versions(self, citation_id: str) -> List[CitationVersion]:
        """Get version history for a citation.

        Args:
            citation_id: The citation identifier.

        Returns:
            List of CitationVersion objects.

        Raises:
            ValueError: If citation not found.
        """
        if citation_id not in self._citations:
            raise ValueError(f"Citation {citation_id} not found")
        return list(self._versions.get(citation_id, []))

    def get_change_log(
        self,
        entity_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ChangeLogEntry]:
        """Get the change log, optionally filtered by entity.

        Args:
            entity_id: Optional filter by entity ID.
            limit: Maximum entries to return.

        Returns:
            List of ChangeLogEntry records, newest first.
        """
        entries = list(self._change_log)
        if entity_id is not None:
            entries = [e for e in entries if e.entity_id == entity_id]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    @property
    def count(self) -> int:
        """Return the number of citations in the registry."""
        return len(self._citations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_change_log(
        self,
        user_id: str,
        change_type: ChangeType,
        entity_type: str,
        entity_id: str,
        old_value: Any,
        new_value: Any,
        change_reason: str,
    ) -> None:
        """Append a change log entry.

        Args:
            user_id: User who made the change.
            change_type: Type of change.
            entity_type: Type of entity changed.
            entity_id: ID of the changed entity.
            old_value: Previous value.
            new_value: New value.
            change_reason: Reason for change.
        """
        entry = ChangeLogEntry(
            user_id=user_id,
            change_type=change_type,
            entity_type=entity_type,
            entity_id=entity_id,
            old_value=old_value,
            new_value=new_value,
            change_reason=change_reason,
        )
        # Compute provenance hash
        entry_data = {
            "user": user_id,
            "type": change_type.value,
            "entity": entity_id,
            "old": old_value,
            "new": new_value,
            "reason": change_reason,
            "timestamp": entry.timestamp.isoformat(),
        }
        entry.provenance_hash = self._compute_hash(entry_data)
        self._change_log.append(entry)

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            data: Data to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()


__all__ = [
    "CitationRegistry",
]
