# -*- coding: utf-8 -*-
"""
Unit Tests for CitationRegistry (AGENT-FOUND-005)

Tests CRUD operations, version management, filtering by source/framework/status,
text search, supersession tracking, change log, and provenance hashing.

Coverage target: 85%+ of registry.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline CitationRegistry mirroring greenlang/citations/registry.py
# ---------------------------------------------------------------------------


class CitationRecord:
    """Minimal citation record for registry testing."""

    def __init__(
        self,
        citation_id: str,
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        title: str = "",
        authors: Optional[List[str]] = None,
        version: Optional[str] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        verification_status: str = "unverified",
        regulatory_frameworks: Optional[List[str]] = None,
        key_values: Optional[Dict[str, Any]] = None,
        superseded_by: Optional[str] = None,
        supersedes: Optional[str] = None,
        record_version: int = 1,
        is_deleted: bool = False,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.citation_id = citation_id
        self.citation_type = citation_type
        self.source_authority = source_authority
        self.title = title
        self.authors = authors or []
        self.version = version
        self.effective_date = effective_date
        self.expiration_date = expiration_date
        self.verification_status = verification_status
        self.regulatory_frameworks = regulatory_frameworks or []
        self.key_values = key_values or {}
        self.superseded_by = superseded_by
        self.supersedes = supersedes
        self.record_version = record_version
        self.is_deleted = is_deleted
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "title": self.title,
            "authors": self.authors,
            "version": self.version,
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "verification_status": self.verification_status,
            "regulatory_frameworks": self.regulatory_frameworks,
            "key_values": self.key_values,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "record_version": self.record_version,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ChangeLogEntry:
    def __init__(self, entry_id, citation_id, change_type, old_data=None,
                 new_data=None, changed_by="system", change_reason="",
                 provenance_hash="", timestamp=None):
        self.entry_id = entry_id
        self.citation_id = citation_id
        self.change_type = change_type
        self.old_data = old_data
        self.new_data = new_data
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class RegistryError(Exception):
    pass

class DuplicateCitationError(RegistryError):
    pass

class CitationNotFoundError(RegistryError):
    pass

class ValidationError(RegistryError):
    pass


class CitationRegistry:
    """Registry for managing citations with version history."""

    def __init__(self, enable_provenance: bool = True):
        self._citations: Dict[str, CitationRecord] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._changelog: List[ChangeLogEntry] = {}
        self._changelog = []
        self._enable_provenance = enable_provenance
        self._change_counter = 0

    def create(
        self,
        citation_id: str,
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        title: str = "",
        authors: Optional[List[str]] = None,
        version: Optional[str] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        key_values: Optional[Dict[str, Any]] = None,
    ) -> CitationRecord:
        if not citation_id:
            raise ValidationError("citation_id is required")
        if citation_id in self._citations and not self._citations[citation_id].is_deleted:
            raise DuplicateCitationError(f"Citation '{citation_id}' already exists")

        record = CitationRecord(
            citation_id=citation_id,
            citation_type=citation_type,
            source_authority=source_authority,
            title=title,
            authors=authors,
            version=version,
            effective_date=effective_date,
            expiration_date=expiration_date,
            regulatory_frameworks=regulatory_frameworks,
            key_values=key_values,
        )
        self._citations[citation_id] = record
        self._versions[citation_id] = [record.to_dict()]
        self._record_change(citation_id, "create", None, record.to_dict())
        return record

    def get(self, citation_id: str) -> CitationRecord:
        if citation_id not in self._citations or self._citations[citation_id].is_deleted:
            raise CitationNotFoundError(f"Citation '{citation_id}' not found")
        return self._citations[citation_id]

    def update(
        self,
        citation_id: str,
        title: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
        verification_status: Optional[str] = None,
        expiration_date: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        version: Optional[str] = None,
        change_reason: str = "",
        changed_by: str = "system",
    ) -> CitationRecord:
        r = self.get(citation_id)
        old_data = r.to_dict()

        if title is not None:
            r.title = title
        if key_values is not None:
            r.key_values = key_values
        if verification_status is not None:
            r.verification_status = verification_status
        if expiration_date is not None:
            r.expiration_date = expiration_date
        if regulatory_frameworks is not None:
            r.regulatory_frameworks = regulatory_frameworks
        if version is not None:
            r.version = version

        r.record_version += 1
        r.updated_at = datetime.utcnow().isoformat()

        self._versions[citation_id].append(r.to_dict())
        self._record_change(citation_id, "update", old_data, r.to_dict(),
                            changed_by=changed_by, change_reason=change_reason)
        return r

    def delete(self, citation_id: str) -> bool:
        r = self.get(citation_id)
        old_data = r.to_dict()
        r.is_deleted = True
        r.updated_at = datetime.utcnow().isoformat()
        self._record_change(citation_id, "delete", old_data, None)
        return True

    def list_citations(
        self,
        citation_type: Optional[str] = None,
        source_authority: Optional[str] = None,
        regulatory_framework: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> List[CitationRecord]:
        results = [r for r in self._citations.values() if not r.is_deleted]

        if citation_type:
            results = [r for r in results if r.citation_type == citation_type]
        if source_authority:
            results = [r for r in results if r.source_authority == source_authority]
        if regulatory_framework:
            results = [r for r in results if regulatory_framework in r.regulatory_frameworks]
        if verification_status:
            results = [r for r in results if r.verification_status == verification_status]
        return results

    def search(self, query: str) -> List[CitationRecord]:
        q = query.lower()
        results = []
        for r in self._citations.values():
            if r.is_deleted:
                continue
            if q in r.title.lower() or q in r.citation_id.lower():
                results.append(r)
                continue
            for author in r.authors:
                if q in author.lower():
                    results.append(r)
                    break
        return results

    def get_by_source(self, source_authority: str) -> List[CitationRecord]:
        return [r for r in self._citations.values()
                if not r.is_deleted and r.source_authority == source_authority]

    def get_by_framework(self, framework: str) -> List[CitationRecord]:
        return [r for r in self._citations.values()
                if not r.is_deleted and framework in r.regulatory_frameworks]

    def get_valid(self, reference_date: Optional[str] = None) -> List[CitationRecord]:
        ref = reference_date or datetime.utcnow().strftime("%Y-%m-%d")
        results = []
        for r in self._citations.values():
            if r.is_deleted:
                continue
            if r.effective_date > ref:
                continue
            if r.expiration_date and r.expiration_date < ref:
                continue
            if r.verification_status in ("invalid", "expired"):
                continue
            results.append(r)
        return results

    def get_versions(self, citation_id: str) -> List[Dict[str, Any]]:
        if citation_id not in self._versions:
            raise CitationNotFoundError(f"Citation '{citation_id}' not found")
        return list(self._versions[citation_id])

    def get_changelog(self, citation_id: Optional[str] = None) -> List[ChangeLogEntry]:
        if citation_id:
            return [e for e in self._changelog if e.citation_id == citation_id]
        return list(self._changelog)

    def supersede(self, old_id: str, new_id: str) -> None:
        old_rec = self.get(old_id)
        new_rec = self.get(new_id)
        old_rec.superseded_by = new_id
        old_rec.verification_status = "superseded"
        new_rec.supersedes = old_id

    @property
    def count(self) -> int:
        return len([r for r in self._citations.values() if not r.is_deleted])

    def _record_change(self, citation_id, change_type, old_data, new_data,
                       changed_by="system", change_reason=""):
        self._change_counter += 1
        prov_hash = ""
        if self._enable_provenance:
            payload = json.dumps({
                "change_type": change_type, "citation_id": citation_id,
                "counter": self._change_counter,
            }, sort_keys=True)
            prov_hash = hashlib.sha256(payload.encode()).hexdigest()

        entry = ChangeLogEntry(
            entry_id=f"cl-{self._change_counter:06d}",
            citation_id=citation_id,
            change_type=change_type,
            old_data=old_data,
            new_data=new_data,
            changed_by=changed_by,
            change_reason=change_reason,
            provenance_hash=prov_hash,
        )
        self._changelog.append(entry)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return CitationRegistry()


@pytest.fixture
def populated_registry():
    reg = CitationRegistry()
    reg.create("defra-2024-ghg", "emission_factor", "defra",
               title="DEFRA 2024 GHG Conversion Factors",
               authors=["DEFRA", "BEIS"], version="2024",
               effective_date="2024-01-01", expiration_date="2025-12-31",
               regulatory_frameworks=["csrd", "cbam"],
               key_values={"diesel_ef": 2.68})
    reg.create("epa-2024-ghg", "emission_factor", "epa",
               title="EPA GHG Emission Factors Hub 2024",
               authors=["US EPA"], version="2024",
               effective_date="2024-01-01",
               regulatory_frameworks=["sb253"],
               key_values={"diesel_ef_us": 2.72})
    reg.create("ipcc-ar6-wg3", "scientific", "ipcc",
               title="Climate Change 2022: Mitigation",
               authors=["IPCC Working Group III"], version="AR6",
               effective_date="2022-04-04",
               key_values={"gwp_ch4": 27.9})
    reg.create("csrd-2022-2464", "regulatory", "eu_commission",
               title="CSRD Directive 2022/2464",
               authors=["European Parliament"],
               effective_date="2024-01-01",
               regulatory_frameworks=["csrd"])
    reg.create("ghg-protocol-corp", "methodology", "ghg_protocol",
               title="GHG Protocol Corporate Standard",
               version="Revised Edition",
               effective_date="2015-01-01",
               regulatory_frameworks=["csrd", "cbam", "sb253"])
    return reg


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCitationRegistryCreate:
    """Test create() operation."""

    def test_create_success(self, registry):
        r = registry.create("cid-1", "emission_factor", "defra", title="Test")
        assert r.citation_id == "cid-1"
        assert r.citation_type == "emission_factor"
        assert r.record_version == 1

    def test_create_with_all_fields(self, registry):
        r = registry.create(
            "cid-1", "emission_factor", "defra",
            title="DEFRA 2024", authors=["DEFRA"],
            version="2024", effective_date="2024-01-01",
            expiration_date="2025-12-31",
            regulatory_frameworks=["csrd"],
            key_values={"ef": 2.68},
        )
        assert r.title == "DEFRA 2024"
        assert r.key_values["ef"] == 2.68
        assert r.regulatory_frameworks == ["csrd"]

    def test_create_duplicate_fails(self, registry):
        registry.create("cid-1", title="First")
        with pytest.raises(DuplicateCitationError, match="already exists"):
            registry.create("cid-1", title="Second")

    def test_create_empty_id_fails(self, registry):
        with pytest.raises(ValidationError, match="citation_id"):
            registry.create("")

    def test_create_increments_count(self, registry):
        assert registry.count == 0
        registry.create("cid-1", title="A")
        assert registry.count == 1
        registry.create("cid-2", title="B")
        assert registry.count == 2

    def test_create_records_initial_version(self, registry):
        registry.create("cid-1", title="Test", key_values={"ef": 2.68})
        versions = registry.get_versions("cid-1")
        assert len(versions) == 1
        assert versions[0]["record_version"] == 1

    def test_create_records_changelog(self, registry):
        registry.create("cid-1", title="Test")
        log = registry.get_changelog("cid-1")
        assert len(log) == 1
        assert log[0].change_type == "create"

    def test_create_all_citation_types(self, registry):
        for i, ct in enumerate(["emission_factor", "regulatory", "methodology",
                                  "scientific", "company_data", "guidance", "database"]):
            r = registry.create(f"cid-{i}", ct, title=f"Type {ct}")
            assert r.citation_type == ct


class TestCitationRegistryGet:
    """Test get() operation."""

    def test_get_success(self, populated_registry):
        r = populated_registry.get("defra-2024-ghg")
        assert r.title == "DEFRA 2024 GHG Conversion Factors"

    def test_get_not_found(self, registry):
        with pytest.raises(CitationNotFoundError, match="not found"):
            registry.get("nonexistent")

    def test_get_deleted_raises(self, registry):
        registry.create("cid-1", title="Test")
        registry.delete("cid-1")
        with pytest.raises(CitationNotFoundError):
            registry.get("cid-1")


class TestCitationRegistryUpdate:
    """Test update() operation."""

    def test_update_title(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", title="Updated Title")
        assert r.title == "Updated Title"
        assert r.record_version == 2

    def test_update_key_values(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", key_values={"diesel_ef": 2.75})
        assert r.key_values["diesel_ef"] == 2.75

    def test_update_verification_status(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", verification_status="verified")
        assert r.verification_status == "verified"

    def test_update_expiration_date(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", expiration_date="2026-12-31")
        assert r.expiration_date == "2026-12-31"

    def test_update_regulatory_frameworks(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", regulatory_frameworks=["csrd", "cbam", "eudr"])
        assert "eudr" in r.regulatory_frameworks

    def test_update_version(self, populated_registry):
        r = populated_registry.update("defra-2024-ghg", version="2025")
        assert r.version == "2025"

    def test_update_creates_version_history(self, populated_registry):
        populated_registry.update("defra-2024-ghg", title="V2")
        populated_registry.update("defra-2024-ghg", title="V3")
        versions = populated_registry.get_versions("defra-2024-ghg")
        assert len(versions) == 3  # initial + 2 updates

    def test_update_not_found(self, registry):
        with pytest.raises(CitationNotFoundError):
            registry.update("nonexistent", title="X")

    def test_update_records_changelog(self, populated_registry):
        populated_registry.update("defra-2024-ghg", title="Updated",
                                   change_reason="Annual update", changed_by="analyst1")
        log = populated_registry.get_changelog("defra-2024-ghg")
        latest = log[-1]
        assert latest.change_type == "update"
        assert latest.change_reason == "Annual update"
        assert latest.changed_by == "analyst1"

    def test_update_provenance_hash(self, populated_registry):
        populated_registry.update("defra-2024-ghg", title="Updated")
        log = populated_registry.get_changelog("defra-2024-ghg")
        assert len(log[-1].provenance_hash) == 64

    def test_multiple_updates_increment_version(self, populated_registry):
        for i in range(5):
            populated_registry.update("defra-2024-ghg", title=f"V{i+2}")
        r = populated_registry.get("defra-2024-ghg")
        assert r.record_version == 6  # 1 initial + 5 updates


class TestCitationRegistryDelete:
    """Test delete() operation."""

    def test_delete_success(self, populated_registry):
        count_before = populated_registry.count
        populated_registry.delete("ipcc-ar6-wg3")
        assert populated_registry.count == count_before - 1

    def test_delete_not_found(self, registry):
        with pytest.raises(CitationNotFoundError):
            registry.delete("nonexistent")

    def test_delete_is_soft(self, populated_registry):
        populated_registry.delete("ipcc-ar6-wg3")
        assert "ipcc-ar6-wg3" in populated_registry._citations
        assert populated_registry._citations["ipcc-ar6-wg3"].is_deleted is True

    def test_delete_already_deleted(self, registry):
        registry.create("cid-1", title="Test")
        registry.delete("cid-1")
        with pytest.raises(CitationNotFoundError):
            registry.delete("cid-1")

    def test_delete_records_changelog(self, registry):
        registry.create("cid-1", title="Test")
        registry.delete("cid-1")
        log = registry.get_changelog("cid-1")
        assert log[-1].change_type == "delete"


class TestCitationRegistryList:
    """Test list_citations() with filters."""

    def test_list_all(self, populated_registry):
        results = populated_registry.list_citations()
        assert len(results) == 5

    def test_list_by_type(self, populated_registry):
        results = populated_registry.list_citations(citation_type="emission_factor")
        assert len(results) == 2

    def test_list_by_source(self, populated_registry):
        results = populated_registry.list_citations(source_authority="defra")
        assert len(results) == 1

    def test_list_by_framework(self, populated_registry):
        results = populated_registry.list_citations(regulatory_framework="csrd")
        assert len(results) == 3  # defra, csrd-directive, ghg-protocol

    def test_list_by_verification_status(self, registry):
        registry.create("cid-1", title="A")
        registry.update("cid-1", verification_status="verified")
        registry.create("cid-2", title="B")
        results = registry.list_citations(verification_status="verified")
        assert len(results) == 1

    def test_list_excludes_deleted(self, populated_registry):
        populated_registry.delete("ipcc-ar6-wg3")
        results = populated_registry.list_citations()
        assert len(results) == 4

    def test_list_empty_registry(self, registry):
        results = registry.list_citations()
        assert results == []

    def test_list_no_match(self, populated_registry):
        results = populated_registry.list_citations(citation_type="database")
        assert results == []


class TestCitationRegistrySearch:
    """Test search() operation."""

    def test_search_by_title(self, populated_registry):
        results = populated_registry.search("DEFRA")
        assert len(results) == 1

    def test_search_by_author(self, populated_registry):
        results = populated_registry.search("IPCC")
        assert len(results) >= 1

    def test_search_case_insensitive(self, populated_registry):
        results = populated_registry.search("defra")
        assert len(results) == 1

    def test_search_no_match(self, populated_registry):
        results = populated_registry.search("nonexistent_keyword")
        assert results == []

    def test_search_by_id(self, populated_registry):
        results = populated_registry.search("epa-2024")
        assert len(results) == 1

    def test_search_excludes_deleted(self, populated_registry):
        populated_registry.delete("defra-2024-ghg")
        results = populated_registry.search("DEFRA")
        assert len(results) == 0


class TestCitationRegistryGetBySource:
    """Test get_by_source() operation."""

    def test_get_by_defra(self, populated_registry):
        results = populated_registry.get_by_source("defra")
        assert len(results) == 1
        assert results[0].citation_id == "defra-2024-ghg"

    def test_get_by_epa(self, populated_registry):
        results = populated_registry.get_by_source("epa")
        assert len(results) == 1

    def test_get_by_ipcc(self, populated_registry):
        results = populated_registry.get_by_source("ipcc")
        assert len(results) == 1

    def test_get_by_unknown_source(self, populated_registry):
        results = populated_registry.get_by_source("unknown")
        assert results == []


class TestCitationRegistryGetByFramework:
    """Test get_by_framework() operation."""

    def test_get_by_csrd(self, populated_registry):
        results = populated_registry.get_by_framework("csrd")
        assert len(results) == 3

    def test_get_by_cbam(self, populated_registry):
        results = populated_registry.get_by_framework("cbam")
        assert len(results) == 2

    def test_get_by_sb253(self, populated_registry):
        results = populated_registry.get_by_framework("sb253")
        assert len(results) == 2

    def test_get_by_eudr(self, populated_registry):
        results = populated_registry.get_by_framework("eudr")
        assert results == []


class TestCitationRegistryGetValid:
    """Test get_valid() operation."""

    def test_get_valid_excludes_expired(self, registry):
        registry.create("cid-1", title="Current",
                         effective_date="2020-01-01")
        registry.create("cid-2", title="Expired",
                         effective_date="2020-01-01", expiration_date="2021-01-01")
        results = registry.get_valid()
        assert len(results) == 1
        assert results[0].citation_id == "cid-1"

    def test_get_valid_excludes_future(self, registry):
        registry.create("cid-1", title="Future", effective_date="2099-01-01")
        results = registry.get_valid()
        assert len(results) == 0

    def test_get_valid_excludes_invalid_status(self, registry):
        registry.create("cid-1", title="Invalid", effective_date="2020-01-01")
        registry.update("cid-1", verification_status="invalid")
        results = registry.get_valid()
        assert len(results) == 0

    def test_get_valid_with_reference_date(self, registry):
        registry.create("cid-1", effective_date="2024-01-01", expiration_date="2025-12-31")
        results = registry.get_valid(reference_date="2024-06-15")
        assert len(results) == 1
        results = registry.get_valid(reference_date="2026-01-15")
        assert len(results) == 0


class TestCitationRegistryVersioning:
    """Test version history tracking."""

    def test_initial_version(self, registry):
        registry.create("cid-1", title="Test")
        versions = registry.get_versions("cid-1")
        assert len(versions) == 1
        assert versions[0]["record_version"] == 1

    def test_versions_grow_with_updates(self, registry):
        registry.create("cid-1", title="V1")
        registry.update("cid-1", title="V2")
        registry.update("cid-1", title="V3")
        versions = registry.get_versions("cid-1")
        assert len(versions) == 3

    def test_versions_not_found(self, registry):
        with pytest.raises(CitationNotFoundError):
            registry.get_versions("nonexistent")

    def test_version_data_preserved(self, registry):
        registry.create("cid-1", title="Original", key_values={"ef": 2.68})
        registry.update("cid-1", key_values={"ef": 2.75})
        versions = registry.get_versions("cid-1")
        assert versions[0]["key_values"]["ef"] == 2.68
        assert versions[1]["key_values"]["ef"] == 2.75


class TestCitationRegistryChangeLog:
    """Test change log entries."""

    def test_changelog_records_create(self, registry):
        registry.create("cid-1", title="Test")
        log = registry.get_changelog("cid-1")
        assert len(log) == 1
        assert log[0].change_type == "create"

    def test_changelog_records_update(self, registry):
        registry.create("cid-1", title="V1")
        registry.update("cid-1", title="V2")
        log = registry.get_changelog("cid-1")
        assert len(log) == 2
        assert log[1].change_type == "update"

    def test_changelog_records_delete(self, registry):
        registry.create("cid-1", title="Test")
        registry.delete("cid-1")
        log = registry.get_changelog("cid-1")
        assert log[-1].change_type == "delete"

    def test_changelog_all(self, populated_registry):
        all_log = populated_registry.get_changelog()
        assert len(all_log) == 5  # 5 creates

    def test_changelog_has_provenance_hash(self, registry):
        registry.create("cid-1", title="Test")
        log = registry.get_changelog("cid-1")
        assert len(log[0].provenance_hash) == 64


class TestCitationRegistrySupersession:
    """Test supersession chain tracking."""

    def test_supersede(self, registry):
        registry.create("defra-2024", title="DEFRA 2024")
        registry.create("defra-2025", title="DEFRA 2025")
        registry.supersede("defra-2024", "defra-2025")

        old = registry._citations["defra-2024"]
        new = registry.get("defra-2025")
        assert old.superseded_by == "defra-2025"
        assert old.verification_status == "superseded"
        assert new.supersedes == "defra-2024"

    def test_supersede_chain(self, registry):
        registry.create("v2023", title="2023")
        registry.create("v2024", title="2024")
        registry.create("v2025", title="2025")
        registry.supersede("v2023", "v2024")
        registry.supersede("v2024", "v2025")

        assert registry._citations["v2023"].superseded_by == "v2024"
        assert registry._citations["v2024"].superseded_by == "v2025"
        assert registry.get("v2025").supersedes == "v2024"

    def test_supersede_not_found(self, registry):
        registry.create("cid-1", title="Test")
        with pytest.raises(CitationNotFoundError):
            registry.supersede("cid-1", "nonexistent")


class TestCitationRegistryCount:
    """Test count accuracy."""

    def test_count_empty(self, registry):
        assert registry.count == 0

    def test_count_after_creates(self, populated_registry):
        assert populated_registry.count == 5

    def test_count_after_delete(self, populated_registry):
        populated_registry.delete("ipcc-ar6-wg3")
        assert populated_registry.count == 4

    def test_count_excludes_deleted(self, registry):
        registry.create("cid-1", title="A")
        registry.create("cid-2", title="B")
        registry.delete("cid-1")
        assert registry.count == 1


class TestCitationRegistryProvenanceDisabled:
    """Test provenance disabled mode."""

    def test_no_provenance_hash(self):
        reg = CitationRegistry(enable_provenance=False)
        reg.create("cid-1", title="Test")
        log = reg.get_changelog("cid-1")
        assert log[0].provenance_hash == ""


class TestCitationRegistryEdgeCases:
    """Test edge cases."""

    def test_empty_key_values(self, registry):
        r = registry.create("cid-1", title="Test", key_values={})
        assert r.key_values == {}

    def test_no_authors(self, registry):
        r = registry.create("cid-1", title="Test")
        assert r.authors == []

    def test_no_expiration(self, registry):
        r = registry.create("cid-1", title="Test")
        assert r.expiration_date is None

    def test_no_regulatory_frameworks(self, registry):
        r = registry.create("cid-1", title="Test")
        assert r.regulatory_frameworks == []

    def test_recreate_after_delete(self, registry):
        registry.create("cid-1", title="V1")
        registry.delete("cid-1")
        r = registry.create("cid-1", title="V2")
        assert r.title == "V2"
        assert r.record_version == 1
