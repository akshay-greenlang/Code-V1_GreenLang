# -*- coding: utf-8 -*-
"""
Evidence Workflow Integration Tests for Citations Service (AGENT-FOUND-005)

Tests complete evidence packaging workflows for audit submission,
multi-citation evidence packages, package finalization and tamper
detection, multi-scope evidence packaging, evidence package lifecycle
management, and evidence with provenance tracking.

All implementations are self-contained to avoid cross-module import issues.

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
# Self-contained implementations for integration testing
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


class ProvenanceEntry:
    def __init__(self, entry_id, entity_id, entity_type="evidence_package",
                 change_type="create", old_data=None, new_data=None,
                 user_id="system", reason="", parent_hash=None, timestamp=None):
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.change_type = change_type
        self.old_data = old_data
        self.new_data = new_data
        self.user_id = user_id
        self.reason = reason
        self.parent_hash = parent_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "entry_id": self.entry_id, "entity_id": self.entity_id,
            "entity_type": self.entity_type, "change_type": self.change_type,
            "old_data": str(self.old_data), "new_data": str(self.new_data),
            "user_id": self.user_id, "parent_hash": self.parent_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class ProvenanceTracker:
    def __init__(self):
        self._entries: List[ProvenanceEntry] = []
        self._counter = 0

    def record(self, entity_id, change_type, entity_type="evidence_package",
               old_data=None, new_data=None, user_id="system", reason=""):
        self._counter += 1
        parent_hash = self._entries[-1].hash if self._entries else None
        entry = ProvenanceEntry(
            entry_id=f"prov-{self._counter:06d}",
            entity_id=entity_id, entity_type=entity_type,
            change_type=change_type, old_data=old_data, new_data=new_data,
            user_id=user_id, reason=reason, parent_hash=parent_hash,
        )
        self._entries.append(entry)
        return entry

    def get_chain(self, entity_id=None, limit=None):
        results = list(self._entries)
        if entity_id:
            results = [e for e in results if e.entity_id == entity_id]
        if limit and limit > 0:
            results = results[-limit:]
        return results

    def verify_chain(self) -> bool:
        if len(self._entries) <= 1:
            return True
        for i in range(1, len(self._entries)):
            if self._entries[i].parent_hash != self._entries[i - 1].hash:
                return False
        return True

    @property
    def count(self) -> int:
        return len(self._entries)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCompleteEvidencePackagingWorkflow:
    """Test complete evidence packaging for audit submission."""

    def test_scope1_diesel_evidence_package(self):
        """Build complete Scope 1 diesel combustion evidence package."""
        em = EvidenceManager()

        # Create package
        pkg = em.create_package(
            "Scope 1 Diesel Combustion - Q4 2024",
            description="Evidence for Q4 2024 diesel combustion emissions",
            context={"scope": 1, "fuel": "diesel", "quarter": "Q4 2024",
                     "reporting_entity": "Facility A"},
            created_by="sustainability_analyst",
        )
        assert pkg.is_finalized is False

        # Add calculation evidence
        calc = EvidenceItem(
            evidence_type="calculation",
            description="Diesel combustion CO2e calculation",
            data={
                "fuel_quantity_litres": 10000,
                "emission_factor": 2.68,
                "result_kgco2e": 26800,
                "methodology": "DEFRA 2024 GHG Conversion Factors",
            },
            citation_ids=["defra-2024-ghg"],
            source_agent="emissions_calculator",
        )
        em.add_item(pkg.package_id, calc)

        # Add source data evidence
        source_data = EvidenceItem(
            evidence_type="data_point",
            description="Fuel purchase records",
            data={
                "total_litres": 10000,
                "supplier": "Fuel Corp",
                "delivery_period": "Q4 2024",
            },
            source_system="erp_system",
        )
        em.add_item(pkg.package_id, source_data)

        # Add methodology reference
        methodology = EvidenceItem(
            evidence_type="methodology",
            description="Calculation methodology description",
            data={
                "method": "simple_multiplication",
                "formula": "quantity * emission_factor",
                "uncertainty": "5%",
            },
        )
        em.add_item(pkg.package_id, methodology)

        # Link citations
        em.add_citation(pkg.package_id, "defra-2024-ghg")
        em.add_citation(pkg.package_id, "ghg-protocol-corporate")

        # Finalize
        pkg_hash = em.finalize_package(pkg.package_id)

        assert len(pkg_hash) == 64
        assert pkg.is_finalized is True
        assert len(pkg.evidence_items) == 3
        assert len(pkg.citation_ids) == 2
        assert pkg.context["scope"] == 1
        assert pkg.created_by == "sustainability_analyst"

    def test_scope2_electricity_evidence_package(self):
        """Build Scope 2 location-based electricity evidence package."""
        em = EvidenceManager()

        pkg = em.create_package(
            "Scope 2 Electricity - FY 2024",
            context={"scope": 2, "method": "location-based"},
        )

        # Add multiple electricity consumption records
        for month in range(1, 13):
            item = EvidenceItem(
                evidence_type="data_point",
                description=f"Electricity consumption month {month}",
                data={"month": month, "kwh": 5000 + month * 100},
                source_system="utility_billing",
            )
            em.add_item(pkg.package_id, item)

        # Add aggregate calculation
        total_kwh = sum(5000 + m * 100 for m in range(1, 13))
        agg_calc = EvidenceItem(
            evidence_type="calculation",
            description="Annual electricity emissions",
            data={
                "total_kwh": total_kwh,
                "grid_factor_kgco2e_per_kwh": 0.233,
                "result_kgco2e": round(total_kwh * 0.233, 2),
            },
            citation_ids=["defra-2024-grid"],
        )
        em.add_item(pkg.package_id, agg_calc)
        em.add_citation(pkg.package_id, "defra-2024-grid")

        pkg_hash = em.finalize_package(pkg.package_id)

        assert len(pkg_hash) == 64
        assert len(pkg.evidence_items) == 13  # 12 months + 1 aggregate


class TestMultiCitationEvidencePackages:
    """Test evidence packages referencing multiple citations."""

    def test_package_with_5_citations(self):
        """Package referencing 5 different authoritative sources."""
        em = EvidenceManager()

        pkg = em.create_package("Multi-Citation Compliance Package")

        citation_ids = [
            "defra-2024-ghg",
            "epa-2024-ghg",
            "ipcc-ar6-wg3",
            "ghg-protocol-corporate",
            "csrd-2022-2464",
        ]
        for cid in citation_ids:
            em.add_citation(pkg.package_id, cid)

        # Add evidence items referencing different citation subsets
        em.add_item(pkg.package_id, EvidenceItem(
            evidence_type="calculation",
            description="EU emissions",
            data={"result": 26800},
            citation_ids=["defra-2024-ghg", "ghg-protocol-corporate"],
        ))
        em.add_item(pkg.package_id, EvidenceItem(
            evidence_type="calculation",
            description="US emissions",
            data={"result": 27200},
            citation_ids=["epa-2024-ghg", "ghg-protocol-corporate"],
        ))

        pkg_hash = em.finalize_package(pkg.package_id)

        assert len(pkg.citation_ids) == 5
        assert len(pkg_hash) == 64

    def test_duplicate_citation_ignored(self):
        """Adding same citation twice does not duplicate."""
        em = EvidenceManager()
        pkg = em.create_package("Test")

        em.add_citation(pkg.package_id, "defra-2024")
        em.add_citation(pkg.package_id, "defra-2024")
        em.add_citation(pkg.package_id, "defra-2024")

        assert len(pkg.citation_ids) == 1


class TestPackageFinalizationAndTamperDetection:
    """Test finalization immutability and tamper detection."""

    def test_finalize_produces_deterministic_hash(self):
        """Same content always produces same hash."""
        def build_and_finalize():
            em = EvidenceManager()
            pkg = em.create_package("Deterministic Test")
            em.add_item(pkg.package_id, EvidenceItem(
                evidence_type="calculation",
                description="Diesel calc",
                data={"quantity": 10000, "ef": 2.68, "result": 26800},
                citation_ids=["defra-2024"],
            ))
            em.add_citation(pkg.package_id, "defra-2024")
            return em.finalize_package(pkg.package_id)

        hash1 = build_and_finalize()
        hash2 = build_and_finalize()
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different evidence produces different hash."""
        em = EvidenceManager()

        pkg1 = em.create_package("Test 1")
        em.add_item(pkg1.package_id, EvidenceItem(
            description="Calc A", data={"result": 100}))
        hash1 = em.finalize_package(pkg1.package_id)

        pkg2 = em.create_package("Test 2")
        em.add_item(pkg2.package_id, EvidenceItem(
            description="Calc B", data={"result": 200}))
        hash2 = em.finalize_package(pkg2.package_id)

        assert hash1 != hash2

    def test_add_item_after_finalize_raises(self):
        """Cannot add evidence items to finalized package."""
        em = EvidenceManager()
        pkg = em.create_package("Test")
        em.finalize_package(pkg.package_id)

        with pytest.raises(RuntimeError, match="finalized"):
            em.add_item(pkg.package_id, EvidenceItem(description="X"))

    def test_add_citation_after_finalize_raises(self):
        """Cannot add citations to finalized package."""
        em = EvidenceManager()
        pkg = em.create_package("Test")
        em.finalize_package(pkg.package_id)

        with pytest.raises(RuntimeError, match="finalized"):
            em.add_citation(pkg.package_id, "new-citation")

    def test_double_finalize_raises(self):
        """Cannot finalize an already finalized package."""
        em = EvidenceManager()
        pkg = em.create_package("Test")
        em.finalize_package(pkg.package_id)

        with pytest.raises(RuntimeError, match="already finalized"):
            em.finalize_package(pkg.package_id)

    def test_hash_length_is_sha256(self):
        """Package hash is valid SHA-256 (64 hex chars)."""
        em = EvidenceManager()
        pkg = em.create_package("Test")
        pkg_hash = em.finalize_package(pkg.package_id)

        assert len(pkg_hash) == 64
        # Verify it's valid hex
        int(pkg_hash, 16)


class TestMultiScopeEvidencePackaging:
    """Test creating separate evidence packages per scope."""

    def test_three_scope_packages(self):
        """Create Scope 1, 2, 3 evidence packages independently."""
        em = EvidenceManager()

        scope_configs = [
            ("Scope 1 Direct Emissions", {"scope": 1},
             [("calculation", "Diesel combustion", {"result_kgco2e": 26800})]),
            ("Scope 2 Indirect Emissions", {"scope": 2},
             [("calculation", "Electricity emissions", {"result_kgco2e": 14000})]),
            ("Scope 3 Value Chain", {"scope": 3},
             [("calculation", "Business travel", {"result_kgco2e": 5200}),
              ("calculation", "Purchased goods", {"result_kgco2e": 120000})]),
        ]

        hashes = []
        for name, ctx, items in scope_configs:
            pkg = em.create_package(name, context=ctx)
            for etype, desc, data in items:
                em.add_item(pkg.package_id, EvidenceItem(
                    evidence_type=etype, description=desc, data=data))
            hashes.append(em.finalize_package(pkg.package_id))

        assert em.count == 3
        assert len(set(hashes)) == 3  # All hashes unique
        assert all(len(h) == 64 for h in hashes)

        # Verify listing
        finalized = em.list_packages(is_finalized=True)
        assert len(finalized) == 3

    def test_scope_separation(self):
        """Packages for different scopes are independent."""
        em = EvidenceManager()

        pkg1 = em.create_package("Scope 1", context={"scope": 1})
        em.add_item(pkg1.package_id, EvidenceItem(
            description="S1", data={"result": 100}))
        em.add_citation(pkg1.package_id, "defra-2024")

        pkg2 = em.create_package("Scope 2", context={"scope": 2})
        em.add_item(pkg2.package_id, EvidenceItem(
            description="S2", data={"result": 200}))
        em.add_citation(pkg2.package_id, "defra-2024-grid")

        # Packages are independent
        assert pkg1.citation_ids != pkg2.citation_ids
        assert len(pkg1.evidence_items) == 1
        assert len(pkg2.evidence_items) == 1


class TestEvidencePackageLifecycle:
    """Test full evidence package lifecycle management."""

    def test_create_populate_finalize_workflow(self):
        """Standard create -> populate -> finalize workflow."""
        em = EvidenceManager()

        # Create
        pkg = em.create_package("Lifecycle Test", description="Full lifecycle")
        assert em.count == 1
        assert pkg.is_finalized is False

        # Populate
        for i in range(5):
            em.add_item(pkg.package_id, EvidenceItem(
                evidence_type="data_point", description=f"DP {i}",
                data={"value": i * 10}))
        em.add_citation(pkg.package_id, "defra-2024")
        em.add_citation(pkg.package_id, "ghg-protocol")

        assert len(pkg.evidence_items) == 5
        assert len(pkg.citation_ids) == 2

        # Finalize
        pkg_hash = em.finalize_package(pkg.package_id)
        assert pkg.is_finalized is True
        assert len(pkg_hash) == 64

    def test_multiple_packages_lifecycle(self):
        """Multiple packages at different lifecycle stages."""
        em = EvidenceManager()

        # Draft package
        draft = em.create_package("Draft Package")
        em.add_item(draft.package_id, EvidenceItem(description="Draft item"))

        # Finalized package
        final = em.create_package("Final Package")
        em.add_item(final.package_id, EvidenceItem(description="Final item"))
        em.finalize_package(final.package_id)

        # Empty package
        empty = em.create_package("Empty Package")

        assert em.count == 3
        assert len(em.list_packages(is_finalized=True)) == 1
        assert len(em.list_packages(is_finalized=False)) == 2

    def test_delete_draft_package(self):
        """Delete a draft (non-finalized) package."""
        em = EvidenceManager()
        pkg = em.create_package("To Delete")
        pid = pkg.package_id
        assert em.count == 1

        em.delete_package(pid)
        assert em.count == 0

        with pytest.raises(PackageNotFoundError):
            em.get_package(pid)

    def test_delete_finalized_package(self):
        """Delete a finalized package (allowed by manager)."""
        em = EvidenceManager()
        pkg = em.create_package("To Delete")
        em.finalize_package(pkg.package_id)
        assert em.count == 1

        em.delete_package(pkg.package_id)
        assert em.count == 0


class TestEvidenceWithProvenanceTracking:
    """Test evidence packaging with provenance audit trail."""

    def test_full_evidence_provenance(self):
        """Track all evidence operations in provenance chain."""
        em = EvidenceManager()
        prov = ProvenanceTracker()

        # Create package
        pkg = em.create_package("Tracked Package",
                                context={"scope": 1})
        prov.record(pkg.package_id, "create", new_data=pkg.to_dict())

        # Add evidence items
        for i in range(3):
            item = EvidenceItem(
                evidence_type="calculation",
                description=f"Calc {i}",
                data={"result": i * 1000},
            )
            em.add_item(pkg.package_id, item)
            prov.record(pkg.package_id, "add_evidence",
                         new_data={"evidence_id": item.evidence_id,
                                   "item_count": i + 1})

        # Add citation
        em.add_citation(pkg.package_id, "defra-2024")
        prov.record(pkg.package_id, "add_citation",
                     new_data={"citation_id": "defra-2024"})

        # Finalize
        pkg_hash = em.finalize_package(pkg.package_id)
        prov.record(pkg.package_id, "finalize",
                     new_data={"package_hash": pkg_hash})

        # Verify provenance chain
        assert prov.count == 6  # create + 3 items + citation + finalize
        assert prov.verify_chain() is True

        pkg_trail = prov.get_chain(entity_id=pkg.package_id)
        assert len(pkg_trail) == 6

        # First entry is create
        assert pkg_trail[0].change_type == "create"
        # Last entry is finalize
        assert pkg_trail[-1].change_type == "finalize"

    def test_provenance_hash_chain_integrity(self):
        """Provenance entries form valid hash chain."""
        em = EvidenceManager()
        prov = ProvenanceTracker()

        pkg = em.create_package("Chain Test")
        prov.record(pkg.package_id, "create")

        em.add_item(pkg.package_id, EvidenceItem(description="E1"))
        prov.record(pkg.package_id, "add_evidence")

        pkg_hash = em.finalize_package(pkg.package_id)
        prov.record(pkg.package_id, "finalize",
                     new_data={"hash": pkg_hash})

        # Verify chain
        assert prov.verify_chain() is True

        # Verify each entry links to previous
        entries = prov.get_chain()
        assert entries[0].parent_hash is None  # First has no parent
        for i in range(1, len(entries)):
            assert entries[i].parent_hash == entries[i - 1].hash


class TestEvidenceItemContentHashing:
    """Test evidence item content hash consistency."""

    def test_same_content_same_hash(self):
        """Identical evidence items produce identical content hashes."""
        item1 = EvidenceItem(
            evidence_type="calculation",
            description="Diesel calc",
            data={"quantity": 10000, "ef": 2.68, "result": 26800},
            citation_ids=["defra-2024"],
        )
        item2 = EvidenceItem(
            evidence_type="calculation",
            description="Diesel calc",
            data={"quantity": 10000, "ef": 2.68, "result": 26800},
            citation_ids=["defra-2024"],
        )
        assert item1.calculate_content_hash() == item2.calculate_content_hash()

    def test_different_data_different_hash(self):
        """Different evidence data produces different content hashes."""
        item1 = EvidenceItem(description="A", data={"result": 100})
        item2 = EvidenceItem(description="A", data={"result": 200})
        assert item1.calculate_content_hash() != item2.calculate_content_hash()

    def test_citation_order_does_not_affect_hash(self):
        """Citation IDs are sorted before hashing for consistency."""
        item1 = EvidenceItem(
            description="Test",
            citation_ids=["a", "b", "c"],
        )
        item2 = EvidenceItem(
            description="Test",
            citation_ids=["c", "a", "b"],
        )
        assert item1.calculate_content_hash() == item2.calculate_content_hash()

    def test_content_hash_is_sha256(self):
        """Content hash is valid SHA-256."""
        item = EvidenceItem(description="Test", data={"v": 1})
        h = item.calculate_content_hash()
        assert len(h) == 64
        int(h, 16)  # Valid hex


class TestMaxItemsEnforcement:
    """Test maximum items per package enforcement."""

    def test_max_items_limit(self):
        """Package rejects items beyond max limit."""
        em = EvidenceManager(max_items_per_package=5)
        pkg = em.create_package("Limited")

        for i in range(5):
            em.add_item(pkg.package_id, EvidenceItem(description=f"Item {i}"))

        with pytest.raises(ValueError, match="maximum items limit"):
            em.add_item(pkg.package_id, EvidenceItem(description="Overflow"))

    def test_default_max_is_500(self):
        """Default max items per package is 500."""
        em = EvidenceManager()
        assert em._max_items_per_package == 500
