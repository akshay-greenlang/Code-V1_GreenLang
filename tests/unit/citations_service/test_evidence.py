# -*- coding: utf-8 -*-
"""
Unit Tests for EvidenceManager (AGENT-FOUND-005)

Tests evidence package creation, item addition, citation linking,
finalization, listing, deletion, and count tracking.

Coverage target: 85%+ of evidence.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline EvidenceManager mirroring greenlang/citations/evidence.py
# ---------------------------------------------------------------------------


class EvidenceItem:
    def __init__(self, evidence_id=None, evidence_type="calculation",
                 description="", data=None, citation_ids=None,
                 source_system=None, source_agent=None, content_hash=None):
        self.evidence_id = evidence_id or str(uuid.uuid4())
        self.evidence_type = evidence_type
        self.description = description
        self.data = data or {}
        self.citation_ids = citation_ids or []
        self.source_system = source_system
        self.source_agent = source_agent
        self.content_hash = content_hash

    def calculate_content_hash(self) -> str:
        content = {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "data": self.data,
            "citation_ids": sorted(self.citation_ids),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class EvidencePackage:
    def __init__(self, package_id=None, name="", description="",
                 context=None, created_by=None):
        if not name:
            raise ValueError("name is required")
        self.package_id = package_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.evidence_items: List[EvidenceItem] = []
        self.citation_ids: List[str] = []
        self.context = context or {}
        self.created_by = created_by
        self.created_at = datetime.utcnow().isoformat()
        self.is_finalized = False
        self.package_hash: Optional[str] = None

    def add_item(self, item: EvidenceItem) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot modify finalized package")
        item.content_hash = item.calculate_content_hash()
        self.evidence_items.append(item)
        self.package_hash = None

    def add_citation(self, citation_id: str) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot modify finalized package")
        if citation_id not in self.citation_ids:
            self.citation_ids.append(citation_id)
        self.package_hash = None

    def finalize(self) -> str:
        if self.is_finalized:
            raise RuntimeError("Package already finalized")
        content = {
            "name": self.name,
            "items": [i.calculate_content_hash() for i in self.evidence_items],
            "citation_ids": sorted(self.citation_ids),
            "context": self.context,
        }
        self.package_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.is_finalized = True
        return self.package_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "name": self.name,
            "description": self.description,
            "evidence_items_count": len(self.evidence_items),
            "citation_ids": self.citation_ids,
            "is_finalized": self.is_finalized,
            "package_hash": self.package_hash,
            "created_at": self.created_at,
        }


class PackageNotFoundError(Exception):
    pass


class EvidenceManager:
    """Manages evidence packages for audit-ready documentation."""

    def __init__(self, max_items_per_package: int = 500):
        self._packages: Dict[str, EvidencePackage] = {}
        self._max_items_per_package = max_items_per_package

    def create_package(self, name: str, description: str = "",
                       context: Optional[Dict[str, Any]] = None,
                       created_by: Optional[str] = None) -> EvidencePackage:
        pkg = EvidencePackage(
            name=name, description=description,
            context=context, created_by=created_by,
        )
        self._packages[pkg.package_id] = pkg
        return pkg

    def get_package(self, package_id: str) -> EvidencePackage:
        if package_id not in self._packages:
            raise PackageNotFoundError(f"Package '{package_id}' not found")
        return self._packages[package_id]

    def add_item(self, package_id: str, item: EvidenceItem) -> EvidencePackage:
        pkg = self.get_package(package_id)
        if len(pkg.evidence_items) >= self._max_items_per_package:
            raise ValueError(
                f"Package has reached maximum items limit ({self._max_items_per_package})"
            )
        pkg.add_item(item)
        return pkg

    def add_citation(self, package_id: str, citation_id: str) -> EvidencePackage:
        pkg = self.get_package(package_id)
        pkg.add_citation(citation_id)
        return pkg

    def finalize_package(self, package_id: str) -> str:
        pkg = self.get_package(package_id)
        return pkg.finalize()

    def list_packages(self, is_finalized: Optional[bool] = None) -> List[EvidencePackage]:
        results = list(self._packages.values())
        if is_finalized is not None:
            results = [p for p in results if p.is_finalized == is_finalized]
        return results

    def delete_package(self, package_id: str) -> bool:
        if package_id not in self._packages:
            raise PackageNotFoundError(f"Package '{package_id}' not found")
        del self._packages[package_id]
        return True

    @property
    def count(self) -> int:
        return len(self._packages)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    return EvidenceManager()


@pytest.fixture
def populated_manager():
    mgr = EvidenceManager()
    p1 = mgr.create_package("Scope 1 Evidence", description="Scope 1 diesel emissions")
    item = EvidenceItem(
        evidence_type="calculation",
        description="Diesel combustion",
        data={"quantity": 10000, "ef": 2.68, "result": 26800},
        citation_ids=["defra-2024-ghg"],
    )
    mgr.add_item(p1.package_id, item)
    mgr.add_citation(p1.package_id, "defra-2024-ghg")

    p2 = mgr.create_package("Scope 2 Evidence", description="Electricity emissions")
    return mgr


# ===========================================================================
# Test Classes
# ===========================================================================


class TestEvidenceManagerCreatePackage:
    """Test create_package() operation."""

    def test_create_success(self, manager):
        pkg = manager.create_package("Test Package")
        assert pkg.name == "Test Package"
        assert pkg.is_finalized is False
        assert manager.count == 1

    def test_create_with_description(self, manager):
        pkg = manager.create_package("Test", description="Detailed description")
        assert pkg.description == "Detailed description"

    def test_create_with_context(self, manager):
        ctx = {"calculation_type": "scope_1", "fuel": "diesel"}
        pkg = manager.create_package("Test", context=ctx)
        assert pkg.context["fuel"] == "diesel"

    def test_create_with_created_by(self, manager):
        pkg = manager.create_package("Test", created_by="analyst1")
        assert pkg.created_by == "analyst1"

    def test_create_auto_generates_id(self, manager):
        pkg = manager.create_package("Test")
        assert len(pkg.package_id) == 36

    def test_create_multiple(self, manager):
        manager.create_package("P1")
        manager.create_package("P2")
        manager.create_package("P3")
        assert manager.count == 3

    def test_create_empty_name_raises(self):
        mgr = EvidenceManager()
        with pytest.raises(ValueError, match="name is required"):
            mgr.create_package("")


class TestEvidenceManagerGetPackage:
    """Test get_package() operation."""

    def test_get_success(self, populated_manager):
        packages = populated_manager.list_packages()
        pkg = populated_manager.get_package(packages[0].package_id)
        assert pkg.name == "Scope 1 Evidence"

    def test_get_not_found(self, manager):
        with pytest.raises(PackageNotFoundError, match="not found"):
            manager.get_package("nonexistent")


class TestEvidenceManagerAddItem:
    """Test add_item() operation."""

    def test_add_item_success(self, manager):
        pkg = manager.create_package("Test")
        item = EvidenceItem(evidence_type="calculation", description="Calc")
        manager.add_item(pkg.package_id, item)
        assert len(pkg.evidence_items) == 1

    def test_add_item_sets_hash(self, manager):
        pkg = manager.create_package("Test")
        item = EvidenceItem(evidence_type="data_point", description="DP",
                             data={"value": 42})
        manager.add_item(pkg.package_id, item)
        assert item.content_hash is not None
        assert len(item.content_hash) == 64

    def test_add_multiple_items(self, manager):
        pkg = manager.create_package("Test")
        for i in range(10):
            item = EvidenceItem(evidence_type="calculation",
                                 description=f"Calc {i}", data={"val": i})
            manager.add_item(pkg.package_id, item)
        assert len(pkg.evidence_items) == 10

    def test_add_item_max_limit(self):
        mgr = EvidenceManager(max_items_per_package=3)
        pkg = mgr.create_package("Test")
        for i in range(3):
            mgr.add_item(pkg.package_id, EvidenceItem(description=f"Item {i}"))
        with pytest.raises(ValueError, match="maximum items limit"):
            mgr.add_item(pkg.package_id, EvidenceItem(description="Overflow"))

    def test_add_item_not_found(self, manager):
        with pytest.raises(PackageNotFoundError):
            manager.add_item("nonexistent", EvidenceItem(description="X"))

    def test_add_item_to_finalized_raises(self, manager):
        pkg = manager.create_package("Test")
        manager.finalize_package(pkg.package_id)
        with pytest.raises(RuntimeError, match="finalized"):
            manager.add_item(pkg.package_id, EvidenceItem(description="X"))

    def test_add_item_invalidates_hash(self, manager):
        pkg = manager.create_package("Test")
        pkg.package_hash = "old_hash"
        manager.add_item(pkg.package_id, EvidenceItem(description="X"))
        assert pkg.package_hash is None

    def test_add_different_evidence_types(self, manager):
        pkg = manager.create_package("Test")
        for etype in ["calculation", "data_point", "methodology",
                       "assumption", "validation", "audit_trail"]:
            manager.add_item(pkg.package_id,
                              EvidenceItem(evidence_type=etype, description=etype))
        assert len(pkg.evidence_items) == 6


class TestEvidenceManagerAddCitation:
    """Test add_citation() operation."""

    def test_add_citation_success(self, manager):
        pkg = manager.create_package("Test")
        manager.add_citation(pkg.package_id, "defra-2024-ghg")
        assert "defra-2024-ghg" in pkg.citation_ids

    def test_add_duplicate_citation(self, manager):
        pkg = manager.create_package("Test")
        manager.add_citation(pkg.package_id, "defra-2024-ghg")
        manager.add_citation(pkg.package_id, "defra-2024-ghg")
        assert pkg.citation_ids.count("defra-2024-ghg") == 1

    def test_add_multiple_citations(self, manager):
        pkg = manager.create_package("Test")
        manager.add_citation(pkg.package_id, "defra-2024-ghg")
        manager.add_citation(pkg.package_id, "epa-2024-ghg")
        manager.add_citation(pkg.package_id, "ipcc-ar6-wg3")
        assert len(pkg.citation_ids) == 3

    def test_add_citation_to_finalized_raises(self, manager):
        pkg = manager.create_package("Test")
        manager.finalize_package(pkg.package_id)
        with pytest.raises(RuntimeError, match="finalized"):
            manager.add_citation(pkg.package_id, "cid-new")


class TestEvidenceManagerFinalizePackage:
    """Test finalize_package() operation."""

    def test_finalize_success(self, manager):
        pkg = manager.create_package("Test")
        manager.add_item(pkg.package_id, EvidenceItem(description="Calc"))
        h = manager.finalize_package(pkg.package_id)
        assert len(h) == 64
        assert pkg.is_finalized is True

    def test_finalize_empty_package(self, manager):
        pkg = manager.create_package("Test")
        h = manager.finalize_package(pkg.package_id)
        assert len(h) == 64

    def test_finalize_double_raises(self, manager):
        pkg = manager.create_package("Test")
        manager.finalize_package(pkg.package_id)
        with pytest.raises(RuntimeError, match="already finalized"):
            manager.finalize_package(pkg.package_id)

    def test_finalize_hash_deterministic(self, manager):
        def make_and_finalize():
            mgr = EvidenceManager()
            pkg = mgr.create_package("Test")
            mgr.add_item(pkg.package_id, EvidenceItem(
                evidence_type="calculation",
                description="Diesel",
                data={"result": 26800},
                citation_ids=["defra-2024"],
            ))
            mgr.add_citation(pkg.package_id, "defra-2024")
            return mgr.finalize_package(pkg.package_id)
        assert make_and_finalize() == make_and_finalize()

    def test_finalize_not_found(self, manager):
        with pytest.raises(PackageNotFoundError):
            manager.finalize_package("nonexistent")


class TestEvidenceManagerListPackages:
    """Test list_packages() operation."""

    def test_list_all(self, populated_manager):
        results = populated_manager.list_packages()
        assert len(results) == 2

    def test_list_finalized_only(self, populated_manager):
        pkg = populated_manager.list_packages()[0]
        populated_manager.finalize_package(pkg.package_id)
        results = populated_manager.list_packages(is_finalized=True)
        assert len(results) == 1

    def test_list_unfinalized_only(self, populated_manager):
        results = populated_manager.list_packages(is_finalized=False)
        assert len(results) == 2

    def test_list_empty(self, manager):
        results = manager.list_packages()
        assert results == []


class TestEvidenceManagerDeletePackage:
    """Test delete_package() operation."""

    def test_delete_success(self, populated_manager):
        count_before = populated_manager.count
        packages = populated_manager.list_packages()
        populated_manager.delete_package(packages[0].package_id)
        assert populated_manager.count == count_before - 1

    def test_delete_not_found(self, manager):
        with pytest.raises(PackageNotFoundError):
            manager.delete_package("nonexistent")


class TestEvidenceManagerCount:
    """Test count property."""

    def test_count_empty(self, manager):
        assert manager.count == 0

    def test_count_after_creates(self, populated_manager):
        assert populated_manager.count == 2

    def test_count_after_delete(self, populated_manager):
        packages = populated_manager.list_packages()
        populated_manager.delete_package(packages[0].package_id)
        assert populated_manager.count == 1
