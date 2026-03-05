# -*- coding: utf-8 -*-
"""
Unit tests for BoundaryManager -- ISO 14064-1:2018 Clause 5.

Tests organizational boundary, operational boundary, entity hierarchy,
inventory CRUD, equity share consolidation, and significance assessment
with 30+ individual test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from services.boundary_manager import BoundaryManager
from services.config import (
    ConsolidationApproach,
    GWPSource,
    ISOCategory,
    ISO14064AppConfig,
    SignificanceLevel,
)


class TestOrganizationCRUD:
    """Test organization creation, update, and deletion."""

    def test_create_organization(self, boundary_manager):
        org = boundary_manager.create_organization("Test Corp", "tech", "US")
        assert org.name == "Test Corp"
        assert org.industry == "tech"
        assert org.country == "US"
        assert len(org.id) == 36

    def test_create_organization_strips_whitespace(self, boundary_manager):
        org = boundary_manager.create_organization("  Acme  ", "  Tech  ", "  us  ")
        assert org.name == "Acme"
        assert org.industry == "tech"
        assert org.country == "US"

    def test_duplicate_organization_raises(self, boundary_manager):
        boundary_manager.create_organization("UniqueOrg", "tech", "US")
        with pytest.raises(ValueError, match="already exists"):
            boundary_manager.create_organization("uniqueorg", "energy", "UK")

    def test_get_organization(self, boundary_manager, sample_org):
        found = boundary_manager.get_organization(sample_org.id)
        assert found is not None
        assert found.id == sample_org.id
        assert found.name == sample_org.name

    def test_get_nonexistent_organization_returns_none(self, boundary_manager):
        assert boundary_manager.get_organization("nonexistent") is None

    def test_update_organization_name(self, boundary_manager, sample_org):
        updated = boundary_manager.update_organization(sample_org.id, name="New Name")
        assert updated.name == "New Name"

    def test_update_organization_duplicate_name_raises(self, boundary_manager):
        boundary_manager.create_organization("Org A", "tech", "US")
        org_b = boundary_manager.create_organization("Org B", "energy", "UK")
        with pytest.raises(ValueError, match="already exists"):
            boundary_manager.update_organization(org_b.id, name="Org A")

    def test_update_nonexistent_org_raises(self, boundary_manager):
        with pytest.raises(ValueError, match="not found"):
            boundary_manager.update_organization("bad-id", name="X")

    def test_delete_organization(self, boundary_manager, sample_org):
        result = boundary_manager.delete_organization(sample_org.id)
        assert result is True
        assert boundary_manager.get_organization(sample_org.id) is None

    def test_delete_nonexistent_org_raises(self, boundary_manager):
        with pytest.raises(ValueError, match="not found"):
            boundary_manager.delete_organization("bad-id")

    def test_list_organizations(self, boundary_manager):
        boundary_manager.create_organization("Org1", "tech", "US")
        boundary_manager.create_organization("Org2", "energy", "DE")
        orgs = boundary_manager.list_organizations()
        assert len(orgs) >= 2


class TestEntityCRUD:
    """Test entity management within organizations."""

    def test_add_entity(self, boundary_manager, sample_org):
        entity = boundary_manager.add_entity(
            org_id=sample_org.id,
            name="Plant A",
            entity_type="facility",
            country="US",
            ownership_pct=Decimal("100.0"),
        )
        assert entity.name == "Plant A"
        assert entity.entity_type == "facility"
        assert entity.country == "US"

    def test_add_entity_to_nonexistent_org_raises(self, boundary_manager):
        with pytest.raises(ValueError, match="not found"):
            boundary_manager.add_entity(
                org_id="bad-org",
                name="X",
                entity_type="facility",
                country="US",
            )

    def test_add_child_entity(self, boundary_manager, sample_org, sample_entity):
        child = boundary_manager.add_entity(
            org_id=sample_org.id,
            name="Sub-unit B",
            entity_type="operation",
            country="US",
            parent_id=sample_entity.id,
        )
        assert child.parent_id == sample_entity.id

    def test_add_entity_with_invalid_parent_raises(self, boundary_manager, sample_org):
        with pytest.raises(ValueError, match="not found"):
            boundary_manager.add_entity(
                org_id=sample_org.id,
                name="X",
                entity_type="facility",
                country="US",
                parent_id="nonexistent-parent",
            )

    def test_get_entity(self, boundary_manager, sample_org, sample_entity):
        found = boundary_manager.get_entity(sample_org.id, sample_entity.id)
        assert found is not None
        assert found.id == sample_entity.id

    def test_update_entity_fields(self, boundary_manager, sample_org, sample_entity):
        updated = boundary_manager.update_entity(
            sample_org.id, sample_entity.id,
            name="Updated Factory",
            employees=600,
        )
        assert updated.name == "Updated Factory"
        assert updated.employees == 600

    def test_delete_entity_reparents_children(self, boundary_manager, sample_org, sample_entity):
        child = boundary_manager.add_entity(
            org_id=sample_org.id,
            name="Child",
            entity_type="operation",
            country="US",
            parent_id=sample_entity.id,
        )
        boundary_manager.delete_entity(sample_org.id, sample_entity.id)
        # Child should be reparented to sample_entity's parent (None)
        refreshed = boundary_manager.get_entity(sample_org.id, child.id)
        assert refreshed.parent_id is None

    def test_get_entity_hierarchy(self, boundary_manager, sample_org, sample_entity):
        boundary_manager.add_entity(
            org_id=sample_org.id,
            name="Sub-plant",
            entity_type="operation",
            country="US",
            parent_id=sample_entity.id,
        )
        hierarchy = boundary_manager.get_entity_hierarchy(sample_org.id)
        assert len(hierarchy) >= 1
        # Root node should have children
        root = hierarchy[0]
        assert "children" in root


class TestOrganizationalBoundary:
    """Test setting consolidation approach per ISO 14064-1 Clause 5.1."""

    def test_set_operational_control(self, boundary_manager, sample_org, sample_entity):
        boundary = boundary_manager.set_organizational_boundary(
            sample_org.id, "operational_control",
        )
        assert boundary.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert sample_entity.id in boundary.entity_ids

    def test_set_equity_share(self, boundary_manager, sample_org, sample_entity):
        boundary = boundary_manager.set_organizational_boundary(
            sample_org.id, "equity_share",
        )
        assert boundary.consolidation_approach == ConsolidationApproach.EQUITY_SHARE

    def test_invalid_approach_raises(self, boundary_manager, sample_org):
        with pytest.raises(ValueError):
            boundary_manager.set_organizational_boundary(
                sample_org.id, "invalid_approach",
            )

    def test_get_organizational_boundary(self, boundary_manager, sample_org, sample_entity):
        boundary_manager.set_organizational_boundary(
            sample_org.id, "operational_control",
        )
        boundary = boundary_manager.get_organizational_boundary(sample_org.id)
        assert boundary is not None
        assert boundary.org_id == sample_org.id


class TestOperationalBoundary:
    """Test operational boundary with category inclusion/exclusion."""

    def test_set_with_all_categories(self, boundary_manager, sample_org):
        decisions = [
            {"category": "category_1_direct", "included": True, "significance": "significant"},
            {"category": "category_2_energy", "included": True, "significance": "significant"},
            {"category": "category_3_transport", "included": True, "significance": "significant"},
        ]
        boundary = boundary_manager.set_operational_boundary(sample_org.id, decisions)
        assert len(boundary.categories) >= 3

    def test_exclude_mandatory_category_raises(self, boundary_manager, sample_org):
        decisions = [
            {"category": "category_1_direct", "included": False, "significance": "not_significant"},
        ]
        with pytest.raises(ValueError, match="mandatory"):
            boundary_manager.set_operational_boundary(sample_org.id, decisions)

    def test_exclude_indirect_without_justification_raises(self, boundary_manager, sample_org):
        decisions = [
            {"category": "category_3_transport", "included": False, "significance": "not_significant"},
        ]
        with pytest.raises(ValueError, match="justification"):
            boundary_manager.set_operational_boundary(sample_org.id, decisions)

    def test_exclude_indirect_with_justification(self, boundary_manager, sample_org):
        decisions = [
            {
                "category": "category_3_transport",
                "included": False,
                "significance": "not_significant",
                "justification": "Less than 1% of total emissions",
            },
        ]
        boundary = boundary_manager.set_operational_boundary(sample_org.id, decisions)
        excluded = [c for c in boundary.categories if not c.included]
        assert len(excluded) >= 1


class TestInventoryCRUD:
    """Test inventory creation and management."""

    def test_create_inventory(self, boundary_manager, sample_org):
        inv = boundary_manager.create_inventory(
            org_id=sample_org.id,
            reporting_year=2025,
        )
        assert inv.org_id == sample_org.id
        assert inv.reporting_year == 2025
        assert inv.gwp_source == GWPSource.AR5

    def test_create_inventory_ar6(self, boundary_manager, sample_org):
        inv = boundary_manager.create_inventory(
            org_id=sample_org.id,
            reporting_year=2025,
            gwp_source="ar6",
        )
        assert inv.gwp_source == GWPSource.AR6

    def test_duplicate_inventory_raises(self, boundary_manager, sample_org):
        boundary_manager.create_inventory(sample_org.id, 2025)
        with pytest.raises(ValueError, match="already exists"):
            boundary_manager.create_inventory(sample_org.id, 2025)

    def test_inventory_for_nonexistent_org_raises(self, boundary_manager):
        with pytest.raises(ValueError, match="not found"):
            boundary_manager.create_inventory("nonexistent-org", 2025)

    def test_list_inventories_sorted(self, boundary_manager, sample_org):
        boundary_manager.create_inventory(sample_org.id, 2023)
        boundary_manager.create_inventory(sample_org.id, 2025)
        boundary_manager.create_inventory(sample_org.id, 2024)
        inventories = boundary_manager.list_inventories(sample_org.id)
        years = [inv.reporting_year for inv in inventories]
        assert years == [2025, 2024, 2023]


class TestEquityShareConsolidation:
    """Test equity share calculation per Clause 5.1."""

    def test_full_ownership(self, boundary_manager, sample_org, sample_entity):
        adjusted = boundary_manager.calculate_equity_share(
            sample_entity.id,
            Decimal("10000"),
        )
        assert adjusted == Decimal("10000.0000")

    def test_partial_ownership(self, boundary_manager, sample_org):
        entity = boundary_manager.add_entity(
            org_id=sample_org.id,
            name="JV",
            entity_type="joint_venture",
            country="CN",
            ownership_pct=Decimal("51.0"),
        )
        adjusted = boundary_manager.calculate_equity_share(
            entity.id, Decimal("10000"),
        )
        assert adjusted == Decimal("5100.0000")

    def test_invalid_ownership_pct_raises(self, boundary_manager, sample_entity):
        with pytest.raises(ValueError, match="Invalid ownership"):
            boundary_manager.calculate_equity_share(
                sample_entity.id,
                Decimal("10000"),
                ownership_pct_override=Decimal("150"),
            )

    def test_consolidate_entity_emissions_control_approach(
        self, boundary_manager, sample_org, sample_entity
    ):
        # Control approach: 100% of all controlled entities
        emissions = {sample_entity.id: Decimal("5000")}
        result = boundary_manager.consolidate_entity_emissions(
            sample_org.id, emissions,
        )
        assert result["total_tco2e"] == Decimal("5000")
        assert result["approach"] == "operational_control"


class TestCategorySignificanceAssessment:
    """Test basic category significance assessment."""

    def test_significant_category(self, boundary_manager):
        result = boundary_manager.assess_category_significance(
            ISOCategory.CATEGORY_3_TRANSPORT,
            Decimal("500"),
            Decimal("10000"),
        )
        assert result["is_significant"] is True
        assert result["percentage_of_total"] == "5.00"

    def test_not_significant_category(self, boundary_manager):
        result = boundary_manager.assess_category_significance(
            ISOCategory.CATEGORY_6_OTHER,
            Decimal("5"),
            Decimal("10000"),
        )
        assert result["is_significant"] is False

    def test_zero_total_emissions(self, boundary_manager):
        result = boundary_manager.assess_category_significance(
            ISOCategory.CATEGORY_3_TRANSPORT,
            Decimal("100"),
            Decimal("0"),
        )
        assert result["percentage_of_total"] == "0"
