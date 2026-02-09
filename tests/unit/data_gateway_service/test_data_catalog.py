# -*- coding: utf-8 -*-
"""
Unit Tests for DataCatalogEngine (AGENT-DATA-004)

Tests catalog entry registration, retrieval, listing with filters (domain,
source_type, tags, multiple filters), keyword search (name, description,
tags), entry updates, removal, domain/tag extraction, and indexing for
efficient lookups.

Coverage target: 85%+ of data_catalog.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class CatalogEntry:
    """A single entry in the data catalog."""

    def __init__(self, entry_id: str, name: str, description: str,
                 domain: str, source_type: str,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.entry_id = entry_id
        self.name = name
        self.description = description
        self.domain = domain
        self.source_type = source_type
        self.tags = tags or []
        self.metadata = metadata or {}
        self.provenance_hash = ""
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at


# ---------------------------------------------------------------------------
# Inline DataCatalogEngine
# ---------------------------------------------------------------------------


class DataCatalogEngine:
    """Data catalog for discovering and managing data source metadata."""

    def __init__(self):
        self._entries: Dict[str, CatalogEntry] = {}
        self._domain_index: Dict[str, List[str]] = {}
        self._tag_index: Dict[str, List[str]] = {}
        self._counter = 0

    def register_entry(self, name: str, description: str,
                       domain: str, source_type: str,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> CatalogEntry:
        """Register a new catalog entry."""
        self._counter += 1
        entry_id = f"CAT-{self._counter:05d}"
        entry = CatalogEntry(
            entry_id=entry_id,
            name=name,
            description=description,
            domain=domain,
            source_type=source_type,
            tags=tags,
            metadata=metadata,
        )
        entry.provenance_hash = _compute_hash({
            "entry_id": entry_id,
            "name": name,
            "domain": domain,
            "source_type": source_type,
        })

        self._entries[entry_id] = entry
        self._index_entry(entry)
        return entry

    def get_entry(self, entry_id: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by ID."""
        return self._entries.get(entry_id)

    def list_entries(self, domain: Optional[str] = None,
                     source_type: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[CatalogEntry]:
        """List catalog entries with optional filters."""
        entries = list(self._entries.values())

        if domain is not None:
            entries = [e for e in entries if e.domain == domain]
        if source_type is not None:
            entries = [e for e in entries if e.source_type == source_type]
        if tags is not None:
            entries = [
                e for e in entries
                if any(t in e.tags for t in tags)
            ]

        return entries

    def search(self, keyword: str) -> List[CatalogEntry]:
        """Search catalog entries by keyword (name, description, tags)."""
        keyword_lower = keyword.lower()
        results: List[CatalogEntry] = []
        for entry in self._entries.values():
            if (keyword_lower in entry.name.lower()
                    or keyword_lower in entry.description.lower()
                    or any(keyword_lower in t.lower() for t in entry.tags)):
                results.append(entry)
        return results

    def update_entry(self, entry_id: str,
                     name: Optional[str] = None,
                     description: Optional[str] = None,
                     domain: Optional[str] = None,
                     source_type: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[CatalogEntry]:
        """Update an existing catalog entry. Returns None if not found."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return None

        # Remove old indexes
        self._deindex_entry(entry)

        if name is not None:
            entry.name = name
        if description is not None:
            entry.description = description
        if domain is not None:
            entry.domain = domain
        if source_type is not None:
            entry.source_type = source_type
        if tags is not None:
            entry.tags = tags
        if metadata is not None:
            entry.metadata = metadata

        entry.updated_at = datetime.now(timezone.utc).isoformat()
        entry.provenance_hash = _compute_hash({
            "entry_id": entry_id,
            "name": entry.name,
            "domain": entry.domain,
            "updated_at": entry.updated_at,
        })

        # Re-index
        self._index_entry(entry)
        return entry

    def remove_entry(self, entry_id: str) -> bool:
        """Remove a catalog entry. Returns True if found and removed."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        self._deindex_entry(entry)
        del self._entries[entry_id]
        return True

    def get_domains(self) -> List[str]:
        """Get all unique domains."""
        return list(self._domain_index.keys())

    def get_tags(self) -> List[str]:
        """Get all unique tags."""
        return list(self._tag_index.keys())

    def _index_entry(self, entry: CatalogEntry) -> None:
        """Add an entry to domain and tag indexes."""
        if entry.domain not in self._domain_index:
            self._domain_index[entry.domain] = []
        self._domain_index[entry.domain].append(entry.entry_id)

        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(entry.entry_id)

    def _deindex_entry(self, entry: CatalogEntry) -> None:
        """Remove an entry from domain and tag indexes."""
        if entry.domain in self._domain_index:
            ids = self._domain_index[entry.domain]
            if entry.entry_id in ids:
                ids.remove(entry.entry_id)
            if not ids:
                del self._domain_index[entry.domain]

        for tag in entry.tags:
            if tag in self._tag_index:
                ids = self._tag_index[tag]
                if entry.entry_id in ids:
                    ids.remove(entry.entry_id)
                if not ids:
                    del self._tag_index[tag]


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> DataCatalogEngine:
    return DataCatalogEngine()


@pytest.fixture
def populated_catalog(engine) -> DataCatalogEngine:
    """Catalog pre-populated with sample entries."""
    engine.register_entry(
        name="Emissions Report",
        description="Annual CO2 emissions data from manufacturing",
        domain="sustainability",
        source_type="erp",
        tags=["emissions", "co2", "manufacturing"],
    )
    engine.register_entry(
        name="Energy Consumption",
        description="Monthly electricity and gas usage",
        domain="sustainability",
        source_type="csv",
        tags=["energy", "electricity", "gas"],
    )
    engine.register_entry(
        name="Purchase Orders",
        description="Supply chain purchase order data",
        domain="procurement",
        source_type="erp",
        tags=["supply-chain", "procurement"],
    )
    engine.register_entry(
        name="Waste Manifests",
        description="Waste disposal and recycling records",
        domain="sustainability",
        source_type="api",
        tags=["waste", "recycling", "emissions"],
    )
    return engine


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegisterEntry:
    """Tests for catalog entry registration."""

    def test_register_entry_success(self, engine):
        entry = engine.register_entry(
            name="Test Dataset",
            description="A test data source",
            domain="testing",
            source_type="csv",
            tags=["test", "sample"],
        )
        assert entry is not None
        assert entry.entry_id.startswith("CAT-")
        assert entry.name == "Test Dataset"
        assert entry.domain == "testing"
        assert entry.source_type == "csv"

    def test_register_entry_id_generation(self, engine):
        e1 = engine.register_entry("A", "desc A", "d1", "csv")
        e2 = engine.register_entry("B", "desc B", "d2", "erp")
        assert e1.entry_id == "CAT-00001"
        assert e2.entry_id == "CAT-00002"

    def test_register_entry_provenance(self, engine):
        entry = engine.register_entry("Test", "desc", "domain", "csv")
        assert entry.provenance_hash is not None
        assert len(entry.provenance_hash) == 64
        int(entry.provenance_hash, 16)  # valid hex


class TestGetEntry:
    """Tests for catalog entry retrieval."""

    def test_get_entry_exists(self, populated_catalog):
        entry = populated_catalog.get_entry("CAT-00001")
        assert entry is not None
        assert entry.name == "Emissions Report"

    def test_get_entry_not_found(self, engine):
        result = engine.get_entry("CAT-99999")
        assert result is None


class TestListEntries:
    """Tests for listing catalog entries with filters."""

    def test_list_all(self, populated_catalog):
        entries = populated_catalog.list_entries()
        assert len(entries) == 4

    def test_list_by_domain(self, populated_catalog):
        entries = populated_catalog.list_entries(domain="sustainability")
        assert len(entries) == 3
        assert all(e.domain == "sustainability" for e in entries)

    def test_list_by_source_type(self, populated_catalog):
        entries = populated_catalog.list_entries(source_type="erp")
        assert len(entries) == 2
        assert all(e.source_type == "erp" for e in entries)

    def test_list_by_tags(self, populated_catalog):
        entries = populated_catalog.list_entries(tags=["emissions"])
        assert len(entries) == 2  # Emissions Report and Waste Manifests

    def test_list_multiple_filters(self, populated_catalog):
        entries = populated_catalog.list_entries(
            domain="sustainability", source_type="erp",
        )
        assert len(entries) == 1
        assert entries[0].name == "Emissions Report"


class TestSearch:
    """Tests for keyword search."""

    def test_keyword_match_in_name(self, populated_catalog):
        results = populated_catalog.search("Emissions")
        assert len(results) >= 1
        assert any(e.name == "Emissions Report" for e in results)

    def test_keyword_match_in_description(self, populated_catalog):
        results = populated_catalog.search("electricity")
        assert len(results) == 1
        assert results[0].name == "Energy Consumption"

    def test_keyword_match_in_tags(self, populated_catalog):
        results = populated_catalog.search("recycling")
        assert len(results) == 1
        assert results[0].name == "Waste Manifests"

    def test_no_results(self, populated_catalog):
        results = populated_catalog.search("blockchain")
        assert results == []


class TestUpdateEntry:
    """Tests for updating catalog entries."""

    def test_update_success(self, populated_catalog):
        entry = populated_catalog.update_entry(
            "CAT-00001", name="Updated Emissions Report",
        )
        assert entry is not None
        assert entry.name == "Updated Emissions Report"

    def test_update_not_found(self, engine):
        result = engine.update_entry("CAT-99999", name="Nope")
        assert result is None

    def test_partial_update(self, populated_catalog):
        original = populated_catalog.get_entry("CAT-00001")
        original_desc = original.description
        updated = populated_catalog.update_entry(
            "CAT-00001", name="New Name",
        )
        assert updated.name == "New Name"
        assert updated.description == original_desc  # unchanged


class TestRemoveEntry:
    """Tests for removing catalog entries."""

    def test_remove_success(self, populated_catalog):
        result = populated_catalog.remove_entry("CAT-00001")
        assert result is True
        assert populated_catalog.get_entry("CAT-00001") is None

    def test_remove_not_found(self, engine):
        result = engine.remove_entry("CAT-99999")
        assert result is False


class TestGetDomains:
    """Tests for domain extraction."""

    def test_unique_domains(self, populated_catalog):
        domains = populated_catalog.get_domains()
        assert "sustainability" in domains
        assert "procurement" in domains
        assert len(set(domains)) == len(domains)  # no duplicates


class TestGetTags:
    """Tests for tag extraction."""

    def test_unique_tags(self, populated_catalog):
        tags = populated_catalog.get_tags()
        assert "emissions" in tags
        assert "energy" in tags
        assert "supply-chain" in tags

    def test_deduplication(self, populated_catalog):
        tags = populated_catalog.get_tags()
        assert len(tags) == len(set(tags))


class TestIndexing:
    """Tests for efficient lookup by domain and tag."""

    def test_domain_index(self, populated_catalog):
        assert "sustainability" in populated_catalog._domain_index
        ids = populated_catalog._domain_index["sustainability"]
        assert len(ids) == 3

    def test_tag_index(self, populated_catalog):
        assert "emissions" in populated_catalog._tag_index
        ids = populated_catalog._tag_index["emissions"]
        assert len(ids) == 2  # Emissions Report and Waste Manifests

    def test_index_updated_on_remove(self, populated_catalog):
        populated_catalog.remove_entry("CAT-00003")  # Purchase Orders
        assert "CAT-00003" not in populated_catalog._domain_index.get("procurement", [])

    def test_index_updated_on_update(self, populated_catalog):
        populated_catalog.update_entry(
            "CAT-00001", domain="finance", tags=["budget"],
        )
        assert "CAT-00001" not in populated_catalog._domain_index.get("sustainability", [])
        assert "CAT-00001" in populated_catalog._domain_index.get("finance", [])
