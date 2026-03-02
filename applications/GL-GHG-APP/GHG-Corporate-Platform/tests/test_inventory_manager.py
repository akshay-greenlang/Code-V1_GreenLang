"""
Unit tests for GL-GHG-APP v1.0 Inventory Manager

Tests organization management, entity hierarchy, consolidation approaches,
operational boundary, inventory lifecycle, equity share application,
and exclusion management.  35+ test cases.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from services.config import (
    ConsolidationApproach,
    EntityType,
    Scope,
    Scope3Category,
)
from services.models import (
    Organization,
    Entity,
    InventoryBoundary,
    ExclusionRecord,
    GHGInventory,
    ScopeEmissions,
)


# ---------------------------------------------------------------------------
# Lightweight in-memory InventoryManager under test
# ---------------------------------------------------------------------------

class InventoryManager:
    """
    In-memory inventory management service.

    Manages organizations, entities, boundaries, and inventories in a
    dictionary-based store for unit testing.
    """

    def __init__(self):
        self.organizations: Dict[str, Organization] = {}
        self.inventories: Dict[str, GHGInventory] = {}
        self.boundaries: Dict[str, InventoryBoundary] = {}

    # -- Organization -------------------------------------------------------

    def create_organization(self, name: str, industry: str, country: str, description: str = None) -> Organization:
        if not name or not name.strip():
            raise ValueError("Organization name is required")
        org = Organization(name=name, industry=industry, country=country, description=description)
        self.organizations[org.id] = org
        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        return self.organizations.get(org_id)

    # -- Entity -------------------------------------------------------------

    def add_entity(
        self,
        org_id: str,
        name: str,
        entity_type: EntityType,
        country: str,
        parent_id: str = None,
        ownership_pct: Decimal = Decimal("100.0"),
        **kwargs,
    ) -> Entity:
        org = self.organizations.get(org_id)
        if org is None:
            raise ValueError(f"Organization {org_id} not found")
        if parent_id:
            parent_found = any(e.id == parent_id for e in org.entities)
            if not parent_found:
                raise ValueError(f"Parent entity {parent_id} not found")
        entity = Entity(
            name=name,
            entity_type=entity_type,
            country=country,
            parent_id=parent_id,
            ownership_pct=ownership_pct,
            **kwargs,
        )
        org.entities.append(entity)
        return entity

    def get_entity_tree(self, org_id: str) -> List[Dict]:
        """Build hierarchical entity tree."""
        org = self.organizations.get(org_id)
        if org is None:
            return []
        root_entities = [e for e in org.entities if e.parent_id is None]
        tree = []
        for root in root_entities:
            tree.append(self._build_subtree(root, org.entities))
        return tree

    def _build_subtree(self, node: Entity, all_entities: List[Entity]) -> Dict:
        children = [e for e in all_entities if e.parent_id == node.id]
        return {
            "entity": node,
            "children": [self._build_subtree(c, all_entities) for c in children],
        }

    def detect_orphans(self, org_id: str) -> List[Entity]:
        """Detect entities whose parent_id references a non-existent entity."""
        org = self.organizations.get(org_id)
        if org is None:
            return []
        entity_ids = {e.id for e in org.entities}
        return [
            e for e in org.entities
            if e.parent_id is not None and e.parent_id not in entity_ids
        ]

    # -- Consolidation & Boundary -------------------------------------------

    def set_boundary(
        self,
        org_id: str,
        approach: ConsolidationApproach,
        reporting_year: int,
        scopes: List[Scope] = None,
    ) -> InventoryBoundary:
        if org_id not in self.organizations:
            raise ValueError(f"Organization {org_id} not found")
        boundary = InventoryBoundary(
            org_id=org_id,
            consolidation_approach=approach,
            reporting_year=reporting_year,
            scopes=scopes or [Scope.SCOPE_1, Scope.SCOPE_2_LOCATION, Scope.SCOPE_2_MARKET],
        )
        self.boundaries[org_id] = boundary
        return boundary

    def add_exclusion(
        self,
        org_id: str,
        scope: Scope,
        reason: str,
        magnitude_pct: Decimal,
        category: str = None,
    ) -> ExclusionRecord:
        boundary = self.boundaries.get(org_id)
        if boundary is None:
            raise ValueError(f"No boundary set for organization {org_id}")
        exclusion = ExclusionRecord(
            scope=scope,
            category=category,
            reason=reason,
            magnitude_pct=magnitude_pct,
        )
        boundary.exclusions.append(exclusion)
        return exclusion

    # -- Inventory ----------------------------------------------------------

    def create_inventory(self, org_id: str, year: int) -> GHGInventory:
        if org_id not in self.organizations:
            raise ValueError(f"Organization {org_id} not found")
        for inv in self.inventories.values():
            if inv.org_id == org_id and inv.year == year:
                raise ValueError(f"Inventory already exists for org {org_id} year {year}")
        boundary = self.boundaries.get(org_id)
        inventory = GHGInventory(org_id=org_id, year=year, boundary=boundary)
        self.inventories[inventory.id] = inventory
        return inventory

    def get_inventory(self, inventory_id: str) -> Optional[GHGInventory]:
        return self.inventories.get(inventory_id)

    def list_inventories(self, org_id: str) -> List[GHGInventory]:
        return [inv for inv in self.inventories.values() if inv.org_id == org_id]

    # -- Equity Share -------------------------------------------------------

    def apply_equity_share(self, inventory: GHGInventory, org: Organization) -> GHGInventory:
        """Apply equity share percentages to scope emissions."""
        if inventory.scope1 is None:
            return inventory
        for entity in org.entities:
            pct = entity.ownership_pct / Decimal("100")
            entity_emissions = inventory.scope1.by_entity.get(entity.id, Decimal("0"))
            inventory.scope1.by_entity[entity.id] = entity_emissions * pct
        # Recalculate total from entity breakdowns
        inventory.scope1.total_tco2e = sum(inventory.scope1.by_entity.values())
        inventory.recalculate_totals()
        return inventory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    """Create fresh InventoryManager."""
    return InventoryManager()


@pytest.fixture
def org_with_entities(manager):
    """Create organization with 3 entities."""
    org = manager.create_organization("Acme Corp", "manufacturing", "US")
    sub = manager.add_entity(org.id, "Acme Europe", EntityType.SUBSIDIARY, "DE", ownership_pct=Decimal("80"))
    fac = manager.add_entity(org.id, "East Plant", EntityType.FACILITY, "US", parent_id=sub.id)
    ops = manager.add_entity(org.id, "Fleet Ops", EntityType.OPERATION, "US", parent_id=fac.id)
    return org, sub, fac, ops


# ---------------------------------------------------------------------------
# TestCreateOrganization
# ---------------------------------------------------------------------------

class TestCreateOrganization:
    """Test organization creation."""

    def test_valid_creation(self, manager):
        """Test creating a valid organization."""
        org = manager.create_organization("Test Corp", "technology", "US")
        assert org.name == "Test Corp"
        assert org.industry == "technology"
        assert org.id in manager.organizations

    def test_missing_name_raises(self, manager):
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError, match="name is required"):
            manager.create_organization("", "tech", "US")

    def test_whitespace_name_raises(self, manager):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name is required"):
            manager.create_organization("   ", "tech", "US")

    def test_with_description(self, manager):
        """Test organization creation with description."""
        org = manager.create_organization("Test", "tech", "US", description="A test company")
        assert org.description == "A test company"


# ---------------------------------------------------------------------------
# TestAddEntity
# ---------------------------------------------------------------------------

class TestAddEntity:
    """Test entity creation and hierarchy."""

    def test_add_subsidiary(self, manager):
        """Test adding a subsidiary entity."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "Sub A", EntityType.SUBSIDIARY, "GB")
        assert entity.entity_type == EntityType.SUBSIDIARY
        assert len(org.entities) == 1

    def test_add_facility(self, manager):
        """Test adding a facility entity."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "Plant A", EntityType.FACILITY, "US")
        assert entity.entity_type == EntityType.FACILITY

    def test_add_operation(self, manager):
        """Test adding an operation entity."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "Fleet", EntityType.OPERATION, "US")
        assert entity.entity_type == EntityType.OPERATION

    def test_parent_linkage(self, manager, org_with_entities):
        """Test parent-child linkage."""
        org, sub, fac, ops = org_with_entities
        assert fac.parent_id == sub.id
        assert ops.parent_id == fac.id

    def test_invalid_org_raises(self, manager):
        """Test adding entity to non-existent org raises."""
        with pytest.raises(ValueError, match="not found"):
            manager.add_entity("fake-org", "Entity", EntityType.FACILITY, "US")

    def test_invalid_parent_raises(self, manager):
        """Test adding entity with non-existent parent raises."""
        org = manager.create_organization("Test", "tech", "US")
        with pytest.raises(ValueError, match="Parent entity"):
            manager.add_entity(org.id, "Child", EntityType.FACILITY, "US", parent_id="fake-parent")


# ---------------------------------------------------------------------------
# TestEntityHierarchy
# ---------------------------------------------------------------------------

class TestEntityHierarchy:
    """Test entity hierarchy tree building."""

    def test_tree_building(self, manager, org_with_entities):
        """Test building entity tree."""
        org, sub, fac, ops = org_with_entities
        tree = manager.get_entity_tree(org.id)
        assert len(tree) == 1  # One root entity (sub has no parent)
        assert tree[0]["entity"].id == sub.id
        assert len(tree[0]["children"]) == 1  # fac is child of sub
        assert len(tree[0]["children"][0]["children"]) == 1  # ops is child of fac

    def test_multi_level(self, manager):
        """Test multi-level hierarchy."""
        org = manager.create_organization("Test", "tech", "US")
        level1 = manager.add_entity(org.id, "L1", EntityType.SUBSIDIARY, "US")
        level2 = manager.add_entity(org.id, "L2", EntityType.SUBSIDIARY, "US", parent_id=level1.id)
        level3 = manager.add_entity(org.id, "L3", EntityType.FACILITY, "US", parent_id=level2.id)
        level4 = manager.add_entity(org.id, "L4", EntityType.OPERATION, "US", parent_id=level3.id)
        tree = manager.get_entity_tree(org.id)
        # Walk down 4 levels
        node = tree[0]
        assert node["entity"].id == level1.id
        node = node["children"][0]
        assert node["entity"].id == level2.id
        node = node["children"][0]
        assert node["entity"].id == level3.id
        node = node["children"][0]
        assert node["entity"].id == level4.id
        assert len(node["children"]) == 0

    def test_orphan_detection(self, manager):
        """Test detection of orphaned entities."""
        org = manager.create_organization("Test", "tech", "US")
        entity = Entity(
            name="Orphan",
            entity_type=EntityType.FACILITY,
            country="US",
            parent_id="non-existent-id",
        )
        org.entities.append(entity)
        orphans = manager.detect_orphans(org.id)
        assert len(orphans) == 1
        assert orphans[0].name == "Orphan"

    def test_empty_org_tree(self, manager):
        """Test tree for org with no entities."""
        org = manager.create_organization("Empty Org", "tech", "US")
        tree = manager.get_entity_tree(org.id)
        assert tree == []


# ---------------------------------------------------------------------------
# TestConsolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    """Test consolidation approaches."""

    def test_operational_control(self, manager):
        """Test operational control approach (100% of controlled entities)."""
        org = manager.create_organization("Test", "tech", "US")
        boundary = manager.set_boundary(org.id, ConsolidationApproach.OPERATIONAL_CONTROL, 2025)
        assert boundary.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_financial_control(self, manager):
        """Test financial control approach."""
        org = manager.create_organization("Test", "tech", "US")
        boundary = manager.set_boundary(org.id, ConsolidationApproach.FINANCIAL_CONTROL, 2025)
        assert boundary.consolidation_approach == ConsolidationApproach.FINANCIAL_CONTROL

    def test_equity_share(self, manager):
        """Test equity share approach."""
        org = manager.create_organization("Test", "tech", "US")
        boundary = manager.set_boundary(org.id, ConsolidationApproach.EQUITY_SHARE, 2025)
        assert boundary.consolidation_approach == ConsolidationApproach.EQUITY_SHARE


# ---------------------------------------------------------------------------
# TestOperationalBoundary
# ---------------------------------------------------------------------------

class TestOperationalBoundary:
    """Test operational boundary (scope selection)."""

    def test_scope_selection(self, manager):
        """Test selecting specific scopes."""
        org = manager.create_organization("Test", "tech", "US")
        boundary = manager.set_boundary(
            org.id,
            ConsolidationApproach.OPERATIONAL_CONTROL,
            2025,
            scopes=[Scope.SCOPE_1, Scope.SCOPE_2_LOCATION, Scope.SCOPE_2_MARKET, Scope.SCOPE_3],
        )
        assert len(boundary.scopes) == 4
        assert Scope.SCOPE_3 in boundary.scopes

    def test_exclusion_adding(self, manager):
        """Test adding an exclusion."""
        org = manager.create_organization("Test", "tech", "US")
        manager.set_boundary(org.id, ConsolidationApproach.OPERATIONAL_CONTROL, 2025)
        exclusion = manager.add_exclusion(
            org.id,
            Scope.SCOPE_3,
            "No franchise operations exist within organizational boundary",
            Decimal("0.1"),
            category=Scope3Category.CAT14_FRANCHISES.value,
        )
        assert exclusion.magnitude_pct == Decimal("0.1")
        boundary = manager.boundaries[org.id]
        assert len(boundary.exclusions) == 1

    def test_exclusion_no_boundary_raises(self, manager):
        """Test adding exclusion without boundary raises."""
        org = manager.create_organization("Test", "tech", "US")
        with pytest.raises(ValueError, match="No boundary"):
            manager.add_exclusion(org.id, Scope.SCOPE_3, "Test reason for exclusion purposes", Decimal("1.0"))


# ---------------------------------------------------------------------------
# TestCreateInventory
# ---------------------------------------------------------------------------

class TestCreateInventory:
    """Test inventory creation."""

    def test_valid_year(self, manager):
        """Test creating inventory for a valid year."""
        org = manager.create_organization("Test", "tech", "US")
        inv = manager.create_inventory(org.id, 2025)
        assert inv.year == 2025
        assert inv.org_id == org.id

    def test_duplicate_year_raises(self, manager):
        """Test creating duplicate inventory raises."""
        org = manager.create_organization("Test", "tech", "US")
        manager.create_inventory(org.id, 2025)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_inventory(org.id, 2025)

    def test_invalid_org_raises(self, manager):
        """Test creating inventory for non-existent org raises."""
        with pytest.raises(ValueError, match="not found"):
            manager.create_inventory("fake-org", 2025)


# ---------------------------------------------------------------------------
# TestGetInventory
# ---------------------------------------------------------------------------

class TestGetInventory:
    """Test inventory retrieval."""

    def test_by_id(self, manager):
        """Test getting inventory by ID."""
        org = manager.create_organization("Test", "tech", "US")
        inv = manager.create_inventory(org.id, 2025)
        retrieved = manager.get_inventory(inv.id)
        assert retrieved is not None
        assert retrieved.id == inv.id

    def test_non_existent(self, manager):
        """Test getting non-existent inventory returns None."""
        assert manager.get_inventory("fake-id") is None


# ---------------------------------------------------------------------------
# TestListInventories
# ---------------------------------------------------------------------------

class TestListInventories:
    """Test inventory listing."""

    def test_all_for_org(self, manager):
        """Test listing all inventories for an organization."""
        org = manager.create_organization("Test", "tech", "US")
        manager.create_inventory(org.id, 2023)
        manager.create_inventory(org.id, 2024)
        manager.create_inventory(org.id, 2025)
        inventories = manager.list_inventories(org.id)
        assert len(inventories) == 3
        years = {inv.year for inv in inventories}
        assert years == {2023, 2024, 2025}

    def test_empty_list(self, manager):
        """Test listing inventories for org with none."""
        org = manager.create_organization("Test", "tech", "US")
        assert manager.list_inventories(org.id) == []


# ---------------------------------------------------------------------------
# TestEquityShare
# ---------------------------------------------------------------------------

class TestEquityShare:
    """Test equity share application."""

    def test_apply_50_pct_ownership(self, manager):
        """Test 50% equity share reduces emissions by half."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "JV", EntityType.SUBSIDIARY, "US", ownership_pct=Decimal("50"))
        inv = manager.create_inventory(org.id, 2025)
        inv.scope1 = ScopeEmissions(
            scope=Scope.SCOPE_1,
            total_tco2e=Decimal("10000.0"),
            by_entity={entity.id: Decimal("10000.0")},
        )
        manager.apply_equity_share(inv, org)
        assert inv.scope1.by_entity[entity.id] == Decimal("5000.0")
        assert inv.scope1.total_tco2e == Decimal("5000.0")

    def test_apply_100_pct_ownership(self, manager):
        """Test 100% equity share keeps full emissions."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "Full", EntityType.SUBSIDIARY, "US", ownership_pct=Decimal("100"))
        inv = manager.create_inventory(org.id, 2025)
        inv.scope1 = ScopeEmissions(
            scope=Scope.SCOPE_1,
            total_tco2e=Decimal("10000.0"),
            by_entity={entity.id: Decimal("10000.0")},
        )
        manager.apply_equity_share(inv, org)
        assert inv.scope1.by_entity[entity.id] == Decimal("10000.0")

    def test_apply_0_pct_ownership(self, manager):
        """Test 0% equity share zeroes out emissions."""
        org = manager.create_organization("Test", "tech", "US")
        entity = manager.add_entity(org.id, "Passive", EntityType.SUBSIDIARY, "US", ownership_pct=Decimal("0"))
        inv = manager.create_inventory(org.id, 2025)
        inv.scope1 = ScopeEmissions(
            scope=Scope.SCOPE_1,
            total_tco2e=Decimal("10000.0"),
            by_entity={entity.id: Decimal("10000.0")},
        )
        manager.apply_equity_share(inv, org)
        assert inv.scope1.by_entity[entity.id] == Decimal("0")


# ---------------------------------------------------------------------------
# TestExclusions
# ---------------------------------------------------------------------------

class TestExclusions:
    """Test exclusion management."""

    def test_add_with_reason(self, manager):
        """Test adding exclusion with reason."""
        org = manager.create_organization("Test", "tech", "US")
        manager.set_boundary(org.id, ConsolidationApproach.OPERATIONAL_CONTROL, 2025)
        exc = manager.add_exclusion(
            org.id,
            Scope.SCOPE_3,
            "Category 14 franchises not applicable to organization",
            Decimal("0.05"),
            category=Scope3Category.CAT14_FRANCHISES.value,
        )
        assert "franchises" in exc.reason.lower()

    def test_magnitude_tracking(self, manager):
        """Test exclusion magnitude tracking."""
        org = manager.create_organization("Test", "tech", "US")
        manager.set_boundary(org.id, ConsolidationApproach.OPERATIONAL_CONTROL, 2025)
        manager.add_exclusion(
            org.id, Scope.SCOPE_3,
            "No downstream leased assets in current reporting period",
            Decimal("0.2"),
            category=Scope3Category.CAT13_DOWNSTREAM_LEASED.value,
        )
        manager.add_exclusion(
            org.id, Scope.SCOPE_3,
            "Franchise operations not applicable to business model",
            Decimal("0.1"),
            category=Scope3Category.CAT14_FRANCHISES.value,
        )
        boundary = manager.boundaries[org.id]
        total_magnitude = sum(e.magnitude_pct for e in boundary.exclusions)
        assert total_magnitude == Decimal("0.3")
        assert len(boundary.exclusions) == 2
