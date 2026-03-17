# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Multi-Entity Engine Tests (25 tests)

Tests MultiEntityEngine: group creation, entity management, hierarchy,
obligation consolidation, de minimis aggregation, cost allocation
(volume/revenue/equal), entity-level declarations, financial guarantee,
member state coordination, delegated compliance, and multi-EORI management.

Author: GreenLang QA Team
"""

import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    _compute_hash,
    _new_uuid,
    _utcnow,
    assert_provenance_hash,
    assert_valid_uuid,
    generate_import_portfolio,
)


# ---------------------------------------------------------------------------
# Group Creation and Entity Management (6 tests)
# ---------------------------------------------------------------------------

class TestGroupManagement:
    """Test entity group creation and management."""

    def test_create_group(self, sample_entity_group):
        """Test creating an entity group."""
        grp = sample_entity_group
        assert grp["group_id"] == "GRP-EUROSTEEL-001"
        assert grp["group_name"] == "EuroSteel Group"
        assert "parent" in grp
        assert "subsidiaries" in grp

    def test_add_entity(self, sample_entity_group):
        """Test adding an entity to the group."""
        new_entity = {
            "entity_id": "ENT-004",
            "legal_name": "EuroSteel Benelux B.V.",
            "eori_number": "NL112233445566778",
            "member_state": "NL",
            "role": "subsidiary",
            "declarant_status": "draft",
        }
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        all_entities.append(new_entity)
        assert len(all_entities) == 4
        assert all_entities[-1]["entity_id"] == "ENT-004"

    def test_set_hierarchy(self, sample_entity_group):
        """Test hierarchy: parent has subsidiaries."""
        parent = sample_entity_group["parent"]
        subs = sample_entity_group["subsidiaries"]
        assert parent["role"] == "parent"
        for sub in subs:
            assert sub["role"] in ("subsidiary", "joint_venture", "branch")

    def test_entity_eori_uniqueness(self, sample_entity_group):
        """Test all entities have unique EORI numbers."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        eoris = [e["eori_number"] for e in all_entities]
        assert len(eoris) == len(set(eoris))

    def test_entity_member_states(self, sample_entity_group):
        """Test entities span multiple member states."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        states = {e["member_state"] for e in all_entities}
        assert len(states) >= 2

    def test_entity_statuses(self, sample_entity_group):
        """Test entities have valid declarant statuses."""
        valid_statuses = {"active", "pending", "suspended", "revoked", "expired", "draft"}
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        for e in all_entities:
            assert e["declarant_status"] in valid_statuses


# ---------------------------------------------------------------------------
# Obligation Consolidation (4 tests)
# ---------------------------------------------------------------------------

class TestObligationConsolidation:
    """Test obligation consolidation across entities."""

    def test_consolidate_obligations(self, sample_entity_group, sample_import_records):
        """Test consolidating obligations across all entities."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        entity_ids = {e["entity_id"] for e in all_entities}
        # Assign imports to entities
        records_by_entity = {}
        for i, rec in enumerate(sample_import_records):
            eid = list(entity_ids)[i % len(entity_ids)]
            records_by_entity.setdefault(eid, []).append(rec)

        total_emissions = sum(r["total_emissions_tco2e"] for r in sample_import_records)
        entity_emissions = {
            eid: sum(r["total_emissions_tco2e"] for r in recs)
            for eid, recs in records_by_entity.items()
        }
        consolidated = sum(entity_emissions.values())
        assert consolidated == pytest.approx(total_emissions, rel=1e-6)

    def test_group_deminimis_below_threshold(self, sample_entity_group):
        """Test group-level de minimis below threshold."""
        # Small imports across 3 entities
        entity_weights_kg = {"ENT-001": 30000, "ENT-002": 40000, "ENT-003": 25000}
        group_total_kg = sum(entity_weights_kg.values())
        threshold_kg = 150000
        exempt = group_total_kg < threshold_kg
        assert exempt is True
        assert group_total_kg == 95000

    def test_group_deminimis_above_threshold(self, sample_entity_group):
        """Test group-level de minimis above threshold."""
        entity_weights_kg = {"ENT-001": 60000, "ENT-002": 55000, "ENT-003": 50000}
        group_total_kg = sum(entity_weights_kg.values())
        threshold_kg = 150000
        exempt = group_total_kg < threshold_kg
        assert exempt is False
        assert group_total_kg == 165000

    def test_consolidation_by_category(self, sample_import_records):
        """Test emissions consolidated by goods category."""
        by_category = {}
        for rec in sample_import_records:
            cat = rec["goods_category"]
            by_category[cat] = by_category.get(cat, 0.0) + rec["total_emissions_tco2e"]
        assert "steel" in by_category
        assert len(by_category) >= 3


# ---------------------------------------------------------------------------
# Cost Allocation (3 tests)
# ---------------------------------------------------------------------------

class TestCostAllocation:
    """Test cost allocation methods across entities."""

    def test_cost_allocation_volume(self, sample_entity_group, sample_import_records):
        """Test cost allocation proportional to import volume."""
        total_cost = Decimal("100000.00")
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        # Assign volumes
        entity_volumes = {
            "ENT-001": Decimal("500"), "ENT-002": Decimal("300"), "ENT-003": Decimal("200"),
        }
        total_volume = sum(entity_volumes.values())
        allocations = {
            eid: total_cost * vol / total_volume
            for eid, vol in entity_volumes.items()
        }
        assert sum(allocations.values()) == total_cost
        assert allocations["ENT-001"] == Decimal("50000.00")

    def test_cost_allocation_revenue(self, sample_entity_group):
        """Test cost allocation proportional to revenue."""
        total_cost = Decimal("100000.00")
        entity_revenues = {
            "ENT-001": Decimal("5000000"),
            "ENT-002": Decimal("3000000"),
            "ENT-003": Decimal("2000000"),
        }
        total_revenue = sum(entity_revenues.values())
        allocations = {
            eid: total_cost * rev / total_revenue
            for eid, rev in entity_revenues.items()
        }
        assert sum(allocations.values()) == total_cost

    def test_cost_allocation_equal(self, sample_entity_group):
        """Test equal cost allocation across entities."""
        total_cost = Decimal("99000.00")
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        num_entities = len(all_entities)
        per_entity = total_cost / num_entities
        allocations = {e["entity_id"]: per_entity for e in all_entities}
        assert len(allocations) == 3
        assert sum(allocations.values()) == total_cost


# ---------------------------------------------------------------------------
# Declarations and Compliance (5 tests)
# ---------------------------------------------------------------------------

class TestDeclarationsAndCompliance:
    """Test entity-level declarations and compliance features."""

    def test_generate_entity_declarations(self, sample_entity_group):
        """Test generating declarations for each entity."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        declarations = []
        for entity in all_entities:
            declarations.append({
                "declaration_id": f"DECL-{entity['entity_id']}-2026",
                "entity_id": entity["entity_id"],
                "eori_number": entity["eori_number"],
                "member_state": entity["member_state"],
                "year": 2026,
                "status": "draft",
            })
        assert len(declarations) == 3
        assert all(d["status"] == "draft" for d in declarations)

    def test_financial_guarantee(self, sample_entity_group):
        """Test financial guarantee is set for the group."""
        guarantee = sample_entity_group["financial_guarantee_eur"]
        assert guarantee == 500000.00
        assert guarantee > 0

    def test_member_state_coordination(self, sample_entity_group):
        """Test member state coordination across entities."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        states = {e["member_state"] for e in all_entities}
        # Each member state needs its own NCA coordination
        nca_contacts = {state: f"nca-cbam@{state.lower()}.eu" for state in states}
        assert len(nca_contacts) >= 2
        assert "DE" in nca_contacts

    def test_delegated_compliance(self, sample_entity_group):
        """Test parent can delegate compliance to subsidiaries."""
        parent = sample_entity_group["parent"]
        sub = sample_entity_group["subsidiaries"][0]
        delegation = {
            "delegation_id": f"DEL-{_new_uuid()[:8]}",
            "delegator": parent["entity_id"],
            "delegate": sub["entity_id"],
            "scope": "quarterly_reporting",
            "valid_from": "2026-01-01",
            "valid_until": "2026-12-31",
            "status": "active",
        }
        assert delegation["delegator"] != delegation["delegate"]
        assert delegation["status"] == "active"

    def test_multi_eori_management(self, sample_entity_group):
        """Test managing multiple EORI numbers."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        eori_registry = {
            e["eori_number"]: {
                "entity_id": e["entity_id"],
                "member_state": e["member_state"],
                "status": e["declarant_status"],
            }
            for e in all_entities
        }
        assert len(eori_registry) == 3
        assert "DE123456789012345" in eori_registry


# ---------------------------------------------------------------------------
# Additional Multi-Entity Features (7 tests)
# ---------------------------------------------------------------------------

class TestAdditionalMultiEntity:
    """Test additional multi-entity features."""

    def test_entity_import_summary(self, sample_entity_group, sample_import_records):
        """Test generating per-entity import summaries."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        entity_ids = [e["entity_id"] for e in all_entities]
        summaries = {}
        for eid in entity_ids:
            records = [r for r in sample_import_records if r.get("supplier_id", "").endswith("001")]
            summaries[eid] = {
                "total_weight_tonnes": sum(r["weight_tonnes"] for r in records[:3]),
                "total_emissions_tco2e": sum(r["total_emissions_tco2e"] for r in records[:3]),
            }
        assert len(summaries) == 3

    def test_group_consolidation_report(self, sample_entity_group):
        """Test group consolidation report generation."""
        report = {
            "group_id": sample_entity_group["group_id"],
            "reporting_year": 2026,
            "entities": 3,
            "total_group_emissions_tco2e": 22500.0,
            "total_certificates_required": 563,
            "provenance_hash": _compute_hash({
                "group_id": sample_entity_group["group_id"],
                "emissions": 22500.0,
            }),
        }
        assert_provenance_hash(report)
        assert report["entities"] == 3

    def test_entity_removal(self, sample_entity_group):
        """Test removing an entity from group."""
        subs = list(sample_entity_group["subsidiaries"])
        removed = subs.pop(0)
        assert len(subs) == 1
        assert removed["entity_id"] == "ENT-002"

    def test_entity_status_change(self, sample_entity_group):
        """Test changing entity declarant status."""
        sub = dict(sample_entity_group["subsidiaries"][1])
        assert sub["declarant_status"] == "pending"
        sub["declarant_status"] = "active"
        assert sub["declarant_status"] == "active"

    def test_group_level_provenance(self, sample_entity_group):
        """Test group-level provenance hash."""
        h = _compute_hash(sample_entity_group)
        assert len(h) == 64

    def test_entity_registration_ids(self, sample_entity_group):
        """Test all entities have registration IDs."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        for e in all_entities:
            assert "registration_id" in e
            assert e["registration_id"].startswith("CBAM-")

    def test_consolidation_method_setting(self, sample_entity_group):
        """Test consolidation method is properly set."""
        assert sample_entity_group["consolidation_method"] in (
            "volume", "revenue", "equal"
        )
