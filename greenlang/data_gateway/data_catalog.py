# -*- coding: utf-8 -*-
"""
Data Catalog Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Maintains a unified catalog of all data assets accessible through the
gateway. Supports domain-based organization, tag-based filtering,
and keyword search across catalog entries.

Zero-Hallucination Guarantees:
    - All catalog operations use deterministic lookups
    - Search uses exact keyword matching (no ML/LLM)
    - Domain and tag indexing uses standard set operations
    - SHA-256 provenance hashes on all catalog operations

Example:
    >>> from greenlang.data_gateway.data_catalog import DataCatalogEngine
    >>> catalog = DataCatalogEngine()
    >>> catalog_id = catalog.register_entry(entry)
    >>> results = catalog.search("emissions")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

def _make_catalog_entry(
    catalog_id: str,
    name: str,
    source_id: str,
    source_type: str,
    domain: str = "",
    description: str = "",
    schema_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    owner: str = "",
    data_classification: str = "internal",
) -> Dict[str, Any]:
    """Create a DataCatalogEntry dictionary.

    Args:
        catalog_id: Unique catalog entry identifier.
        name: Human-readable data asset name.
        source_id: Associated data source ID.
        source_type: Type of data source.
        domain: Business domain (e.g. emissions, supply_chain).
        description: Data asset description.
        schema_id: Associated schema definition ID.
        tags: Organizational tags.
        metadata: Additional metadata.
        owner: Data asset owner.
        data_classification: Data classification level.

    Returns:
        DataCatalogEntry dictionary.
    """
    now = _utcnow().isoformat()
    return {
        "catalog_id": catalog_id,
        "name": name,
        "source_id": source_id,
        "source_type": source_type,
        "domain": domain,
        "description": description,
        "schema_id": schema_id,
        "tags": tags or [],
        "metadata": metadata or {},
        "owner": owner,
        "data_classification": data_classification,
        "created_at": now,
        "updated_at": now,
    }


class DataCatalogEngine:
    """Unified data catalog engine.

    Maintains a registry of data assets with domain-based organization,
    tag-based filtering, and keyword search capabilities.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _entries: In-memory catalog entry storage.
        _domain_index: Index of catalog IDs by domain.
        _tag_index: Index of catalog IDs by tag.

    Example:
        >>> catalog = DataCatalogEngine()
        >>> cid = catalog.register_entry({"name": "emissions", ...})
        >>> results = catalog.search("emissions")
        >>> assert len(results) > 0
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize DataCatalogEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

        logger.info("DataCatalogEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_entry(
        self,
        entry: Dict[str, Any],
    ) -> str:
        """Register a new catalog entry.

        Args:
            entry: Catalog entry definition with keys:
                name (str): Asset name (required).
                source_id (str): Source ID (required).
                source_type (str): Source type (required).
                domain (str): Business domain.
                description (str): Description.
                schema_id (str): Associated schema.
                tags (List[str]): Tags.
                metadata (Dict): Additional metadata.
                owner (str): Owner.
                data_classification (str): Classification level.

        Returns:
            Generated catalog_id.

        Raises:
            ValueError: If required fields are missing.
        """
        name = entry.get("name", "")
        source_id = entry.get("source_id", "")
        source_type = entry.get("source_type", "")

        if not name:
            raise ValueError("Catalog entry name is required")
        if not source_id:
            raise ValueError("Catalog entry source_id is required")
        if not source_type:
            raise ValueError("Catalog entry source_type is required")

        catalog_id = self._generate_catalog_id()
        domain = entry.get("domain", "").lower().strip()
        tags = [t.lower().strip() for t in entry.get("tags", [])]

        catalog_entry = _make_catalog_entry(
            catalog_id=catalog_id,
            name=name,
            source_id=source_id,
            source_type=source_type,
            domain=domain,
            description=entry.get("description", ""),
            schema_id=entry.get("schema_id"),
            tags=tags,
            metadata=entry.get("metadata"),
            owner=entry.get("owner", ""),
            data_classification=entry.get(
                "data_classification", "internal"
            ),
        )

        # Store entry
        self._entries[catalog_id] = catalog_entry

        # Update domain index
        if domain:
            if domain not in self._domain_index:
                self._domain_index[domain] = set()
            self._domain_index[domain].add(catalog_id)

        # Update tag index
        for tag in tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(catalog_id)

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(catalog_entry)
            self._provenance.record(
                entity_type="catalog_entry",
                entity_id=catalog_id,
                action="template_creation",
                data_hash=data_hash,
            )

        logger.info(
            "Registered catalog entry %s: name=%s, domain=%s, tags=%s",
            catalog_id, name, domain, tags,
        )
        return catalog_id

    def get_entry(
        self,
        catalog_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a catalog entry by ID.

        Args:
            catalog_id: Catalog entry identifier.

        Returns:
            DataCatalogEntry dictionary or None if not found.
        """
        return self._entries.get(catalog_id)

    def list_entries(
        self,
        domain: Optional[str] = None,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List catalog entries with optional filters.

        Args:
            domain: Filter by business domain.
            source_type: Filter by source type.
            tags: Filter by tags (entries must have ALL specified tags).

        Returns:
            List of DataCatalogEntry dictionaries.
        """
        # Start with domain-filtered set or all entries
        if domain:
            domain = domain.lower().strip()
            catalog_ids = self._domain_index.get(domain, set())
            results = [
                self._entries[cid] for cid in catalog_ids
                if cid in self._entries
            ]
        else:
            results = list(self._entries.values())

        # Filter by source type
        if source_type:
            source_type = source_type.lower().strip()
            results = [
                e for e in results
                if e.get("source_type", "").lower() == source_type
            ]

        # Filter by tags (AND logic: entry must have all specified tags)
        if tags:
            tag_set = {t.lower().strip() for t in tags}
            results = [
                e for e in results
                if tag_set.issubset(set(e.get("tags", [])))
            ]

        results.sort(key=lambda e: e.get("name", ""))
        return results

    def search(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search catalog entries by keyword.

        Searches across name, description, domain, tags, and source_type
        using case-insensitive substring matching.

        Args:
            query: Search query string.

        Returns:
            List of matching DataCatalogEntry dictionaries.
        """
        if not query:
            return list(self._entries.values())

        query_lower = query.lower().strip()
        results: List[Dict[str, Any]] = []

        for entry in self._entries.values():
            # Search across multiple fields
            searchable = " ".join([
                entry.get("name", ""),
                entry.get("description", ""),
                entry.get("domain", ""),
                entry.get("source_type", ""),
                entry.get("owner", ""),
                " ".join(entry.get("tags", [])),
            ]).lower()

            if query_lower in searchable:
                results.append(entry)

        results.sort(key=lambda e: e.get("name", ""))
        return results

    def update_entry(
        self,
        catalog_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update a catalog entry.

        Args:
            catalog_id: Catalog entry identifier.
            updates: Dictionary of fields to update.

        Returns:
            Updated DataCatalogEntry or None if not found.
        """
        entry = self._entries.get(catalog_id)
        if entry is None:
            return None

        old_domain = entry.get("domain", "")
        old_tags = set(entry.get("tags", []))

        # Apply updates
        allowed_fields = {
            "name", "description", "domain", "tags",
            "metadata", "owner", "data_classification",
            "schema_id",
        }
        for key, value in updates.items():
            if key in allowed_fields:
                if key == "tags" and isinstance(value, list):
                    value = [t.lower().strip() for t in value]
                if key == "domain" and isinstance(value, str):
                    value = value.lower().strip()
                entry[key] = value

        entry["updated_at"] = _utcnow().isoformat()

        # Update domain index
        new_domain = entry.get("domain", "")
        if old_domain != new_domain:
            if old_domain and old_domain in self._domain_index:
                self._domain_index[old_domain].discard(catalog_id)
            if new_domain:
                if new_domain not in self._domain_index:
                    self._domain_index[new_domain] = set()
                self._domain_index[new_domain].add(catalog_id)

        # Update tag index
        new_tags = set(entry.get("tags", []))
        removed_tags = old_tags - new_tags
        added_tags = new_tags - old_tags

        for tag in removed_tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(catalog_id)

        for tag in added_tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(catalog_id)

        logger.info("Updated catalog entry %s", catalog_id)
        return entry

    def remove_entry(self, catalog_id: str) -> bool:
        """Remove a catalog entry.

        Args:
            catalog_id: Catalog entry identifier.

        Returns:
            True if entry was removed, False if not found.
        """
        entry = self._entries.pop(catalog_id, None)
        if entry is None:
            return False

        # Clean up domain index
        domain = entry.get("domain", "")
        if domain and domain in self._domain_index:
            self._domain_index[domain].discard(catalog_id)

        # Clean up tag index
        for tag in entry.get("tags", []):
            if tag in self._tag_index:
                self._tag_index[tag].discard(catalog_id)

        logger.info("Removed catalog entry %s", catalog_id)
        return True

    def get_domains(self) -> List[str]:
        """List all unique domains in the catalog.

        Returns:
            Sorted list of domain names.
        """
        domains = set()
        for entry in self._entries.values():
            domain = entry.get("domain", "")
            if domain:
                domains.add(domain)
        return sorted(domains)

    def get_tags(self) -> List[str]:
        """List all unique tags in the catalog.

        Returns:
            Sorted list of tag names.
        """
        tags = set()
        for entry in self._entries.values():
            for tag in entry.get("tags", []):
                if tag:
                    tags.add(tag)
        return sorted(tags)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_catalog_id(self) -> str:
        """Generate a unique catalog entry identifier.

        Returns:
            Catalog ID in format "CAT-{hex12}".
        """
        return f"CAT-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the total number of catalog entries."""
        return len(self._entries)

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dictionary with catalog counts by domain and source type.
        """
        by_domain: Dict[str, int] = {}
        by_source_type: Dict[str, int] = {}

        for entry in self._entries.values():
            domain = entry.get("domain", "unclassified")
            by_domain[domain] = by_domain.get(domain, 0) + 1

            stype = entry.get("source_type", "unknown")
            by_source_type[stype] = by_source_type.get(stype, 0) + 1

        return {
            "total_entries": len(self._entries),
            "total_domains": len(self.get_domains()),
            "total_tags": len(self.get_tags()),
            "by_domain": by_domain,
            "by_source_type": by_source_type,
        }


__all__ = [
    "DataCatalogEngine",
]
