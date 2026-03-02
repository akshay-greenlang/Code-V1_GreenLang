"""
Inventory Manager -- Organizational Boundary and GHG Inventory Setup

Implements GHG Protocol Corporate Standard Chapters 3-5:
  - Chapter 3: Organizational boundaries (equity share, financial control,
    operational control)
  - Chapter 4: Operational boundaries (Scope 1, 2, 3)
  - Chapter 5: Tracking emissions over time

Uses in-memory storage for v1.0.  All mutations are logged and
provenance-tracked via SHA-256 hashes.

Example:
    >>> mgr = InventoryManager(config)
    >>> org = mgr.create_organization(CreateOrganizationRequest(...))
    >>> inv = mgr.create_inventory(org.id, 2025)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import (
    ConsolidationApproach,
    GHGAppConfig,
    Scope,
)
from .models import (
    AddEntityRequest,
    CreateInventoryRequest,
    CreateOrganizationRequest,
    Entity,
    ExclusionRecord,
    ExclusionRequest,
    GHGInventory,
    InventoryBoundary,
    Organization,
    ScopeEmissions,
    SetBoundaryRequest,
    UpdateEntityRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class InventoryManager:
    """
    Manages organizational boundary and GHG inventory lifecycle.

    Responsibilities:
      - CRUD for organizations and their entity hierarchies
      - Setting consolidation and operational boundaries
      - Creating and managing GHG inventories per reporting year
      - Equity share calculations
      - Scope/category exclusion management

    All data is stored in-memory (dictionaries) for v1.0.
    """

    def __init__(self, config: Optional[GHGAppConfig] = None) -> None:
        """Initialize InventoryManager with optional config."""
        self.config = config or GHGAppConfig()
        self._organizations: Dict[str, Organization] = {}
        self._inventories: Dict[str, GHGInventory] = {}
        self._boundaries: Dict[str, InventoryBoundary] = {}
        logger.info("InventoryManager initialized")

    # ------------------------------------------------------------------
    # Organization CRUD
    # ------------------------------------------------------------------

    def create_organization(
        self,
        data: CreateOrganizationRequest,
    ) -> Organization:
        """
        Create a new organization.

        Args:
            data: Organization creation request.

        Returns:
            Newly created Organization.

        Raises:
            ValueError: If organization name already exists.
        """
        start = datetime.utcnow()

        for org in self._organizations.values():
            if org.name.lower() == data.name.strip().lower():
                raise ValueError(f"Organization '{data.name}' already exists")

        org = Organization(
            name=data.name.strip(),
            industry=data.industry.strip().lower(),
            country=data.country.strip().upper(),
            description=data.description,
        )
        self._organizations[org.id] = org

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Created organization '%s' (id=%s) in %.1f ms",
            org.name,
            org.id,
            elapsed_ms,
        )
        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Retrieve an organization by ID."""
        return self._organizations.get(org_id)

    def list_organizations(self) -> List[Organization]:
        """List all organizations."""
        return list(self._organizations.values())

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(
        self,
        org_id: str,
        data: AddEntityRequest,
    ) -> Entity:
        """
        Add an entity (subsidiary/facility/operation) to an organization.

        Args:
            org_id: Parent organization ID.
            data: Entity creation request.

        Returns:
            Newly created Entity.

        Raises:
            ValueError: If organization not found or parent entity invalid.
        """
        org = self._get_org_or_raise(org_id)

        if data.parent_id:
            self._get_entity_or_raise(org, data.parent_id)

        entity = Entity(
            name=data.name.strip(),
            entity_type=data.entity_type,
            parent_id=data.parent_id,
            ownership_pct=data.ownership_pct,
            country=data.country.strip().upper(),
            employees=data.employees,
            revenue=data.revenue,
            floor_area_m2=data.floor_area_m2,
            production_units=data.production_units,
            production_unit_name=data.production_unit_name,
        )
        org.entities.append(entity)
        org.updated_at = _now()

        logger.info(
            "Added entity '%s' (type=%s) to org '%s'",
            entity.name,
            entity.entity_type.value,
            org.name,
        )
        return entity

    def update_entity(
        self,
        org_id: str,
        entity_id: str,
        data: UpdateEntityRequest,
    ) -> Entity:
        """
        Update an existing entity's attributes.

        Only non-None fields in the request are applied.

        Args:
            org_id: Organization ID.
            entity_id: Entity ID to update.
            data: Fields to update.

        Returns:
            Updated Entity.
        """
        org = self._get_org_or_raise(org_id)
        entity = self._get_entity_or_raise(org, entity_id)

        update_fields = data.model_dump(exclude_none=True)
        for field_name, value in update_fields.items():
            setattr(entity, field_name, value)
        entity.updated_at = _now()
        org.updated_at = _now()

        logger.info("Updated entity '%s' fields: %s", entity.name, list(update_fields.keys()))
        return entity

    def remove_entity(self, org_id: str, entity_id: str) -> bool:
        """
        Remove an entity from an organization.

        Children of the removed entity are re-parented to the removed
        entity's parent (or become root-level if no parent).

        Args:
            org_id: Organization ID.
            entity_id: Entity ID to remove.

        Returns:
            True if removed successfully.

        Raises:
            ValueError: If organization or entity not found.
        """
        org = self._get_org_or_raise(org_id)
        entity = self._get_entity_or_raise(org, entity_id)

        # Re-parent children
        for child in org.entities:
            if child.parent_id == entity_id:
                child.parent_id = entity.parent_id
                child.updated_at = _now()

        org.entities = [e for e in org.entities if e.id != entity_id]
        org.updated_at = _now()

        logger.info("Removed entity '%s' from org '%s'", entity.name, org.name)
        return True

    def get_entity_hierarchy(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Build a tree-structured entity hierarchy.

        Returns a list of root-level entities, each with a ``children``
        list recursively populated.

        Args:
            org_id: Organization ID.

        Returns:
            List of entity tree nodes.
        """
        org = self._get_org_or_raise(org_id)
        entity_map: Dict[str, Dict[str, Any]] = {}

        for e in org.entities:
            entity_map[e.id] = {
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type.value,
                "parent_id": e.parent_id,
                "ownership_pct": str(e.ownership_pct),
                "country": e.country,
                "employees": e.employees,
                "revenue": str(e.revenue) if e.revenue else None,
                "active": e.active,
                "children": [],
            }

        roots: List[Dict[str, Any]] = []
        for eid, node in entity_map.items():
            pid = node["parent_id"]
            if pid and pid in entity_map:
                entity_map[pid]["children"].append(node)
            else:
                roots.append(node)

        logger.info(
            "Built entity hierarchy for org '%s': %d total, %d roots",
            org.name,
            len(entity_map),
            len(roots),
        )
        return roots

    # ------------------------------------------------------------------
    # Boundary Management
    # ------------------------------------------------------------------

    def set_consolidation_approach(
        self,
        org_id: str,
        approach: ConsolidationApproach,
        reporting_year: int,
    ) -> InventoryBoundary:
        """
        Set the consolidation approach for an organization.

        Per GHG Protocol Ch 3, the approach determines which entities
        are included and how their emissions are allocated.

        Args:
            org_id: Organization ID.
            approach: Consolidation approach.
            reporting_year: Year for the boundary.

        Returns:
            Updated or new InventoryBoundary.
        """
        org = self._get_org_or_raise(org_id)

        boundary = self._get_or_create_boundary(org_id, reporting_year)
        boundary.consolidation_approach = approach
        boundary.updated_at = _now()

        # Under equity share, ownership_pct matters; under control it is 100/0
        if approach in (
            ConsolidationApproach.OPERATIONAL_CONTROL,
            ConsolidationApproach.FINANCIAL_CONTROL,
        ):
            boundary.entity_ids = [e.id for e in org.entities if e.active]
        else:
            boundary.entity_ids = [e.id for e in org.entities if e.active and e.ownership_pct > 0]

        logger.info(
            "Set consolidation approach '%s' for org '%s' year %d (%d entities)",
            approach.value,
            org.name,
            reporting_year,
            len(boundary.entity_ids),
        )
        return boundary

    def set_operational_boundary(
        self,
        org_id: str,
        scopes: List[Scope],
        reporting_year: int,
    ) -> InventoryBoundary:
        """
        Set the operational boundary (which scopes to report).

        Per GHG Protocol Ch 4, Scope 1 and Scope 2 are mandatory.
        Scope 3 is optional but recommended.

        Args:
            org_id: Organization ID.
            scopes: Scopes to include.
            reporting_year: Year for the boundary.

        Returns:
            Updated InventoryBoundary.

        Raises:
            ValueError: If mandatory Scope 1 is missing.
        """
        self._get_org_or_raise(org_id)

        if Scope.SCOPE_1 not in scopes:
            raise ValueError("Scope 1 is mandatory per GHG Protocol")

        # Scope 2 at least one method is mandatory
        has_scope2 = (
            Scope.SCOPE_2_LOCATION in scopes
            or Scope.SCOPE_2_MARKET in scopes
        )
        if not has_scope2:
            raise ValueError(
                "At least one Scope 2 method (location or market) is mandatory"
            )

        boundary = self._get_or_create_boundary(org_id, reporting_year)
        boundary.scopes = scopes
        boundary.updated_at = _now()

        logger.info(
            "Set operational boundary for org '%s' year %d: %s",
            org_id,
            reporting_year,
            [s.value for s in scopes],
        )
        return boundary

    # ------------------------------------------------------------------
    # Inventory Lifecycle
    # ------------------------------------------------------------------

    def create_inventory(
        self,
        org_id: str,
        year: int,
        request: Optional[CreateInventoryRequest] = None,
    ) -> GHGInventory:
        """
        Create a new GHG inventory for an organization-year.

        Initializes empty scope emissions structures and links the
        current boundary configuration.

        Args:
            org_id: Organization ID.
            year: Reporting year.
            request: Optional creation parameters.

        Returns:
            Newly created GHGInventory.

        Raises:
            ValueError: If inventory already exists for this org-year.
        """
        start = datetime.utcnow()
        self._get_org_or_raise(org_id)

        # Check for existing inventory
        for inv in self._inventories.values():
            if inv.org_id == org_id and inv.year == year:
                raise ValueError(
                    f"Inventory already exists for org {org_id} year {year}"
                )

        boundary = self._boundaries.get(f"{org_id}_{year}")

        inventory = GHGInventory(
            org_id=org_id,
            year=year,
            boundary=boundary,
            scope1=ScopeEmissions(scope=Scope.SCOPE_1),
            scope2_location=ScopeEmissions(scope=Scope.SCOPE_2_LOCATION),
            scope2_market=ScopeEmissions(scope=Scope.SCOPE_2_MARKET),
            scope3=ScopeEmissions(scope=Scope.SCOPE_3),
        )
        self._inventories[inventory.id] = inventory

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Created inventory (id=%s) for org '%s' year %d in %.1f ms",
            inventory.id,
            org_id,
            year,
            elapsed_ms,
        )
        return inventory

    def get_inventory(self, inventory_id: str) -> Optional[GHGInventory]:
        """Retrieve an inventory by ID."""
        return self._inventories.get(inventory_id)

    def list_inventories(self, org_id: str) -> List[GHGInventory]:
        """List all inventories for an organization, sorted by year desc."""
        inventories = [
            inv for inv in self._inventories.values()
            if inv.org_id == org_id
        ]
        return sorted(inventories, key=lambda i: i.year, reverse=True)

    def update_inventory(
        self,
        inventory_id: str,
        **kwargs: Any,
    ) -> GHGInventory:
        """
        Update inventory fields and recalculate totals.

        Args:
            inventory_id: Inventory ID.
            **kwargs: Fields to update.

        Returns:
            Updated GHGInventory.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        for key, value in kwargs.items():
            if hasattr(inventory, key):
                setattr(inventory, key, value)

        inventory.recalculate_totals()
        inventory.updated_at = _now()
        inventory.provenance_hash = _sha256(
            f"{inventory.org_id}:{inventory.year}:{inventory.grand_total_tco2e}"
        )

        logger.info("Updated inventory %s, new total: %s tCO2e", inventory_id, inventory.grand_total_tco2e)
        return inventory

    def finalize_inventory(self, inventory_id: str) -> GHGInventory:
        """
        Mark an inventory as final (no further edits without recalculation).

        Args:
            inventory_id: Inventory ID.

        Returns:
            Finalized GHGInventory.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        if inventory.status == "verified":
            raise ValueError("Cannot modify a verified inventory")

        inventory.status = "final"
        inventory.recalculate_totals()
        inventory.updated_at = _now()
        inventory.provenance_hash = _sha256(
            f"final:{inventory.org_id}:{inventory.year}:{inventory.grand_total_tco2e}"
        )

        logger.info("Finalized inventory %s", inventory_id)
        return inventory

    # ------------------------------------------------------------------
    # Equity Share Calculations
    # ------------------------------------------------------------------

    def apply_equity_share(
        self,
        entity_id: str,
        emissions: Decimal,
        ownership_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Apply equity share allocation to entity emissions.

        Per GHG Protocol Ch 3, under the equity share approach,
        a company accounts for emissions proportional to its
        ownership percentage.

        Args:
            entity_id: Entity ID (used to look up ownership if pct not given).
            emissions: Gross emissions before equity adjustment.
            ownership_pct: Override ownership percentage (0-100).

        Returns:
            Adjusted emissions (tCO2e).
        """
        if ownership_pct is None:
            ownership_pct = self._lookup_ownership_pct(entity_id)

        if ownership_pct < 0 or ownership_pct > 100:
            raise ValueError(f"Invalid ownership_pct: {ownership_pct}")

        adjusted = emissions * (ownership_pct / Decimal("100"))

        logger.debug(
            "Equity share: entity=%s, gross=%.2f, pct=%.1f%%, adjusted=%.2f",
            entity_id,
            emissions,
            ownership_pct,
            adjusted,
        )
        return adjusted

    # ------------------------------------------------------------------
    # Exclusion Management
    # ------------------------------------------------------------------

    def get_exclusions(self, org_id: str, year: int) -> List[ExclusionRecord]:
        """Get all exclusions for an organization boundary."""
        boundary = self._boundaries.get(f"{org_id}_{year}")
        if boundary is None:
            return []
        return boundary.exclusions

    def add_exclusion(
        self,
        org_id: str,
        year: int,
        request: ExclusionRequest,
    ) -> ExclusionRecord:
        """
        Add a scope/category exclusion to the boundary.

        Per GHG Protocol, exclusions must be justified and their
        estimated magnitude disclosed.

        Args:
            org_id: Organization ID.
            year: Reporting year.
            request: Exclusion details.

        Returns:
            Created ExclusionRecord.
        """
        boundary = self._get_or_create_boundary(org_id, year)

        exclusion = ExclusionRecord(
            scope=request.scope,
            category=request.category,
            reason=request.reason,
            magnitude_pct=request.magnitude_pct,
        )
        boundary.exclusions.append(exclusion)
        boundary.updated_at = _now()

        logger.info(
            "Added exclusion for org '%s' year %d: scope=%s, category=%s, magnitude=%.1f%%",
            org_id,
            year,
            request.scope.value,
            request.category or "all",
            request.magnitude_pct,
        )
        return exclusion

    def remove_exclusion(
        self,
        org_id: str,
        year: int,
        exclusion_id: str,
    ) -> bool:
        """Remove an exclusion by ID."""
        boundary = self._boundaries.get(f"{org_id}_{year}")
        if boundary is None:
            raise ValueError(f"No boundary found for org {org_id} year {year}")

        original_count = len(boundary.exclusions)
        boundary.exclusions = [
            e for e in boundary.exclusions if e.id != exclusion_id
        ]
        removed = len(boundary.exclusions) < original_count

        if removed:
            boundary.updated_at = _now()
            logger.info("Removed exclusion %s from org '%s' year %d", exclusion_id, org_id, year)
        return removed

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_org_or_raise(self, org_id: str) -> Organization:
        """Retrieve organization or raise ValueError."""
        org = self._organizations.get(org_id)
        if org is None:
            raise ValueError(f"Organization not found: {org_id}")
        return org

    def _get_entity_or_raise(self, org: Organization, entity_id: str) -> Entity:
        """Retrieve entity within an organization or raise ValueError."""
        for entity in org.entities:
            if entity.id == entity_id:
                return entity
        raise ValueError(f"Entity not found: {entity_id} in org {org.id}")

    def _get_inventory_or_raise(self, inventory_id: str) -> GHGInventory:
        """Retrieve inventory or raise ValueError."""
        inventory = self._inventories.get(inventory_id)
        if inventory is None:
            raise ValueError(f"Inventory not found: {inventory_id}")
        return inventory

    def _get_or_create_boundary(
        self,
        org_id: str,
        reporting_year: int,
    ) -> InventoryBoundary:
        """Get existing boundary or create a new one."""
        key = f"{org_id}_{reporting_year}"
        if key not in self._boundaries:
            self._boundaries[key] = InventoryBoundary(
                org_id=org_id,
                consolidation_approach=self.config.default_consolidation_approach,
                reporting_year=reporting_year,
            )
        return self._boundaries[key]

    def _lookup_ownership_pct(self, entity_id: str) -> Decimal:
        """Look up ownership percentage for an entity across all orgs."""
        for org in self._organizations.values():
            for entity in org.entities:
                if entity.id == entity_id:
                    return entity.ownership_pct
        raise ValueError(f"Entity not found: {entity_id}")
