"""
Boundary Manager -- ISO 14064-1:2018 Clause 5 Organizational & Operational Boundaries

Implements ISO 14064-1:2018 boundary management:
  - Clause 5.1: Organizational boundaries (equity share, financial control,
    operational control)
  - Clause 5.2: Operational boundaries (6 ISO categories with significance
    assessment and inclusion/exclusion decisions)
  - Clause 5.3: Reporting period validation (minimum 12 months)

Uses in-memory storage for v1.0.  All mutations are logged and
provenance-tracked via SHA-256 hashes.

Example:
    >>> mgr = BoundaryManager(config)
    >>> org = mgr.create_organization("Acme Corp", "manufacturing", "US")
    >>> boundary = mgr.set_organizational_boundary(org.id, "operational_control")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ConsolidationApproach,
    GWPSource,
    ISOCategory,
    ISO14064AppConfig,
    ISO_CATEGORY_NAMES,
    InventoryStatus,
    SignificanceLevel,
)
from .models import (
    CategoryInclusion,
    Entity,
    ISOInventory,
    OperationalBoundary,
    Organization,
    OrganizationalBoundary,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class BoundaryManager:
    """
    Manages organizational and operational boundaries per ISO 14064-1:2018.

    Responsibilities:
      - CRUD for organizations and their entity hierarchies
      - Setting the consolidation approach (Clause 5.1)
      - Setting operational boundaries with category inclusion/exclusion (Clause 5.2)
      - Creating and managing GHG inventories per reporting year
      - Equity share consolidation calculations
      - Reporting period validation (minimum 12 months)
      - Entity hierarchy traversal for consolidation

    All data is stored in-memory (dictionaries) for v1.0.
    """

    def __init__(self, config: Optional[ISO14064AppConfig] = None) -> None:
        """
        Initialize BoundaryManager with optional config.

        Args:
            config: Application configuration.  Defaults are used if None.
        """
        self.config = config or ISO14064AppConfig()
        self._organizations: Dict[str, Organization] = {}
        self._entities: Dict[str, Entity] = {}
        self._org_boundaries: Dict[str, OrganizationalBoundary] = {}
        self._op_boundaries: Dict[str, OperationalBoundary] = {}
        self._inventories: Dict[str, ISOInventory] = {}
        logger.info("BoundaryManager initialized")

    # ------------------------------------------------------------------
    # Organization CRUD
    # ------------------------------------------------------------------

    def create_organization(
        self,
        name: str,
        industry: str,
        country: str,
        description: Optional[str] = None,
    ) -> Organization:
        """
        Create a new organization.

        Args:
            name: Legal entity name.
            industry: Industry sector (e.g. manufacturing, energy).
            country: ISO 3166-1 alpha-2/3 HQ country code.
            description: Optional company description.

        Returns:
            Newly created Organization.

        Raises:
            ValueError: If organization name already exists.
        """
        start = datetime.now(timezone.utc)
        clean_name = name.strip()

        for org in self._organizations.values():
            if org.name.lower() == clean_name.lower():
                raise ValueError(f"Organization '{clean_name}' already exists")

        org = Organization(
            name=clean_name,
            industry=industry.strip().lower(),
            country=country.strip().upper(),
            description=description,
        )
        self._organizations[org.id] = org

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Created organization '%s' (id=%s) in %.1f ms",
            org.name, org.id, elapsed_ms,
        )
        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Retrieve an organization by ID."""
        return self._organizations.get(org_id)

    def update_organization(
        self,
        org_id: str,
        name: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Organization:
        """
        Update an existing organization's attributes.

        Only non-None parameters are applied.

        Args:
            org_id: Organization ID.
            name: New legal entity name.
            industry: New industry sector.
            country: New HQ country code.
            description: New description.

        Returns:
            Updated Organization.

        Raises:
            ValueError: If organization not found or duplicate name.
        """
        org = self._get_org_or_raise(org_id)

        if name is not None:
            clean_name = name.strip()
            for other in self._organizations.values():
                if other.id != org_id and other.name.lower() == clean_name.lower():
                    raise ValueError(f"Organization '{clean_name}' already exists")
            org.name = clean_name

        if industry is not None:
            org.industry = industry.strip().lower()
        if country is not None:
            org.country = country.strip().upper()
        if description is not None:
            org.description = description

        org.updated_at = _now()
        logger.info("Updated organization '%s' (id=%s)", org.name, org.id)
        return org

    def delete_organization(self, org_id: str) -> bool:
        """
        Delete an organization and all associated data.

        Args:
            org_id: Organization ID.

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If organization not found.
        """
        self._get_org_or_raise(org_id)

        # Remove associated boundaries
        keys_to_remove = [
            k for k, v in self._org_boundaries.items()
            if v.org_id == org_id
        ]
        for key in keys_to_remove:
            del self._org_boundaries[key]

        op_keys_to_remove = [
            k for k, v in self._op_boundaries.items()
            if v.org_id == org_id
        ]
        for key in op_keys_to_remove:
            del self._op_boundaries[key]

        # Remove associated inventories
        inv_keys_to_remove = [
            k for k, v in self._inventories.items()
            if v.org_id == org_id
        ]
        for key in inv_keys_to_remove:
            del self._inventories[key]

        # Remove associated entities
        entity_keys_to_remove = [
            k for k, v in self._entities.items()
            if k in [e.id for e in self._organizations[org_id].entities]
        ]
        for key in entity_keys_to_remove:
            del self._entities[key]

        del self._organizations[org_id]
        logger.info("Deleted organization %s and all associated data", org_id)
        return True

    def list_organizations(self) -> List[Organization]:
        """List all organizations."""
        return list(self._organizations.values())

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(
        self,
        org_id: str,
        name: str,
        entity_type: str,
        country: str,
        parent_id: Optional[str] = None,
        ownership_pct: Decimal = Decimal("100.0"),
        employees: Optional[int] = None,
        revenue: Optional[Decimal] = None,
        floor_area_m2: Optional[Decimal] = None,
    ) -> Entity:
        """
        Add an entity (subsidiary/facility/operation) to an organization.

        Args:
            org_id: Parent organization ID.
            name: Entity name.
            entity_type: Type (subsidiary, facility, operation, joint_venture).
            country: ISO 3166-1 country code.
            parent_id: Parent entity ID for hierarchy (None = root level).
            ownership_pct: Equity share percentage (0-100).
            employees: Full-time equivalents.
            revenue: Annual revenue in USD.
            floor_area_m2: Floor area in square metres.

        Returns:
            Newly created Entity.

        Raises:
            ValueError: If organization not found or parent entity invalid.
        """
        org = self._get_org_or_raise(org_id)

        if parent_id is not None:
            self._get_entity_in_org_or_raise(org, parent_id)

        entity = Entity(
            name=name.strip(),
            entity_type=entity_type.strip().lower(),
            parent_id=parent_id,
            ownership_pct=ownership_pct,
            country=country.strip().upper(),
            employees=employees,
            revenue=revenue,
            floor_area_m2=floor_area_m2,
        )
        org.entities.append(entity)
        self._entities[entity.id] = entity
        org.updated_at = _now()

        logger.info(
            "Added entity '%s' (type=%s, ownership=%.1f%%) to org '%s'",
            entity.name, entity.entity_type, ownership_pct, org.name,
        )
        return entity

    def get_entity(self, org_id: str, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity within an organization."""
        org = self._organizations.get(org_id)
        if org is None:
            return None
        for entity in org.entities:
            if entity.id == entity_id:
                return entity
        return None

    def update_entity(
        self,
        org_id: str,
        entity_id: str,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        ownership_pct: Optional[Decimal] = None,
        country: Optional[str] = None,
        employees: Optional[int] = None,
        revenue: Optional[Decimal] = None,
        floor_area_m2: Optional[Decimal] = None,
        active: Optional[bool] = None,
    ) -> Entity:
        """
        Update an existing entity's attributes.

        Only non-None parameters are applied.

        Args:
            org_id: Organization ID.
            entity_id: Entity ID.
            name: New entity name.
            entity_type: New entity type.
            parent_id: New parent entity ID.
            ownership_pct: New equity share percentage.
            country: New country code.
            employees: New employee count.
            revenue: New annual revenue.
            floor_area_m2: New floor area.
            active: New active status.

        Returns:
            Updated Entity.

        Raises:
            ValueError: If organization or entity not found.
        """
        org = self._get_org_or_raise(org_id)
        entity = self._get_entity_in_org_or_raise(org, entity_id)

        if name is not None:
            entity.name = name.strip()
        if entity_type is not None:
            entity.entity_type = entity_type.strip().lower()
        if parent_id is not None:
            if parent_id != entity_id:
                self._get_entity_in_org_or_raise(org, parent_id)
            entity.parent_id = parent_id
        if ownership_pct is not None:
            entity.ownership_pct = ownership_pct
        if country is not None:
            entity.country = country.strip().upper()
        if employees is not None:
            entity.employees = employees
        if revenue is not None:
            entity.revenue = revenue
        if floor_area_m2 is not None:
            entity.floor_area_m2 = floor_area_m2
        if active is not None:
            entity.active = active

        entity.updated_at = _now()
        org.updated_at = _now()

        logger.info("Updated entity '%s' (id=%s) in org '%s'", entity.name, entity_id, org.name)
        return entity

    def delete_entity(self, org_id: str, entity_id: str) -> bool:
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
        entity = self._get_entity_in_org_or_raise(org, entity_id)

        # Re-parent children
        for child in org.entities:
            if child.parent_id == entity_id:
                child.parent_id = entity.parent_id
                child.updated_at = _now()

        org.entities = [e for e in org.entities if e.id != entity_id]
        self._entities.pop(entity_id, None)
        org.updated_at = _now()

        logger.info("Removed entity '%s' from org '%s'", entity.name, org.name)
        return True

    def get_entity_hierarchy(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Build a tree-structured entity hierarchy for an organization.

        Returns a list of root-level entities, each with a ``children``
        list recursively populated.

        Args:
            org_id: Organization ID.

        Returns:
            List of entity tree nodes with nested children.

        Raises:
            ValueError: If organization not found.
        """
        org = self._get_org_or_raise(org_id)
        entity_map: Dict[str, Dict[str, Any]] = {}

        for e in org.entities:
            entity_map[e.id] = {
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
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
            org.name, len(entity_map), len(roots),
        )
        return roots

    # ------------------------------------------------------------------
    # Organizational Boundary (ISO 14064-1 Clause 5.1)
    # ------------------------------------------------------------------

    def set_organizational_boundary(
        self,
        org_id: str,
        consolidation_approach: str,
    ) -> OrganizationalBoundary:
        """
        Set the organizational boundary (consolidation approach).

        Per ISO 14064-1 Clause 5.1, the organization shall select one of:
          - operational control
          - financial control
          - equity share

        Under control approaches, all active entities are included at 100%.
        Under equity share, entities are included proportional to ownership.

        Args:
            org_id: Organization ID.
            consolidation_approach: One of 'operational_control',
                'financial_control', 'equity_share'.

        Returns:
            Created or updated OrganizationalBoundary.

        Raises:
            ValueError: If organization not found or invalid approach.
        """
        org = self._get_org_or_raise(org_id)
        approach = ConsolidationApproach(consolidation_approach)

        # Determine which entities to include
        if approach in (
            ConsolidationApproach.OPERATIONAL_CONTROL,
            ConsolidationApproach.FINANCIAL_CONTROL,
        ):
            entity_ids = [e.id for e in org.entities if e.active]
        else:
            # Equity share: include only entities with ownership > 0
            entity_ids = [
                e.id for e in org.entities
                if e.active and e.ownership_pct > 0
            ]

        boundary = OrganizationalBoundary(
            org_id=org_id,
            consolidation_approach=approach,
            entity_ids=entity_ids,
        )
        self._org_boundaries[org_id] = boundary

        logger.info(
            "Set organizational boundary for org '%s': approach=%s, %d entities",
            org.name, approach.value, len(entity_ids),
        )
        return boundary

    def get_organizational_boundary(
        self,
        org_id: str,
    ) -> Optional[OrganizationalBoundary]:
        """
        Retrieve the organizational boundary for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            OrganizationalBoundary or None if not set.
        """
        return self._org_boundaries.get(org_id)

    # ------------------------------------------------------------------
    # Operational Boundary (ISO 14064-1 Clause 5.2)
    # ------------------------------------------------------------------

    def set_operational_boundary(
        self,
        org_id: str,
        category_decisions: List[Dict[str, Any]],
    ) -> OperationalBoundary:
        """
        Set the operational boundary with category inclusion/exclusion.

        Per ISO 14064-1 Clause 5.2, Categories 1 and 2 are mandatory.
        Categories 3-6 require significance assessment. If excluded,
        a justification must be provided.

        Args:
            org_id: Organization ID.
            category_decisions: List of dicts with keys:
                - category (str): ISO category value.
                - included (bool): Whether to include.
                - significance (str): significant / not_significant.
                - justification (str): Required if excluded.

        Returns:
            Created or updated OperationalBoundary.

        Raises:
            ValueError: If organization not found, or mandatory categories
                are excluded without justification.
        """
        self._get_org_or_raise(org_id)

        inclusions: List[CategoryInclusion] = []
        for decision in category_decisions:
            cat = ISOCategory(decision["category"])
            included = decision.get("included", True)
            significance_str = decision.get("significance", "significant")
            significance = SignificanceLevel(significance_str)
            justification = decision.get("justification")

            # Categories 1 and 2 are mandatory
            if cat in (ISOCategory.CATEGORY_1_DIRECT, ISOCategory.CATEGORY_2_ENERGY):
                if not included:
                    raise ValueError(
                        f"{ISO_CATEGORY_NAMES[cat]} is mandatory and cannot be excluded"
                    )

            # Exclusions require justification
            if not included and not justification:
                raise ValueError(
                    f"Exclusion of {ISO_CATEGORY_NAMES[cat]} requires a justification"
                )

            inclusions.append(CategoryInclusion(
                category=cat,
                included=included,
                significance=significance,
                justification=justification,
            ))

        # Ensure Categories 1 and 2 are present (add if missing)
        present_categories = {ci.category for ci in inclusions}
        for mandatory_cat in (ISOCategory.CATEGORY_1_DIRECT, ISOCategory.CATEGORY_2_ENERGY):
            if mandatory_cat not in present_categories:
                inclusions.append(CategoryInclusion(
                    category=mandatory_cat,
                    included=True,
                    significance=SignificanceLevel.SIGNIFICANT,
                ))

        boundary = OperationalBoundary(
            org_id=org_id,
            categories=inclusions,
        )
        self._op_boundaries[org_id] = boundary

        included_count = sum(1 for ci in inclusions if ci.included)
        logger.info(
            "Set operational boundary for org '%s': %d/%d categories included",
            org_id, included_count, len(inclusions),
        )
        return boundary

    def get_operational_boundary(
        self,
        org_id: str,
    ) -> Optional[OperationalBoundary]:
        """
        Retrieve the operational boundary for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            OperationalBoundary or None if not set.
        """
        return self._op_boundaries.get(org_id)

    # ------------------------------------------------------------------
    # Inventory CRUD
    # ------------------------------------------------------------------

    def create_inventory(
        self,
        org_id: str,
        reporting_year: int,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        gwp_source: str = "ar5",
    ) -> ISOInventory:
        """
        Create a new ISO 14064-1 GHG inventory for an organization-year.

        Validates that:
          - The organization exists.
          - No duplicate inventory for the same org-year.
          - The reporting period is at least the configured minimum months.

        Args:
            org_id: Organization ID.
            reporting_year: Reporting year (e.g. 2025).
            period_start: Optional reporting period start (defaults to Jan 1).
            period_end: Optional reporting period end (defaults to Dec 31).
            gwp_source: GWP assessment report ('ar5' or 'ar6').

        Returns:
            Newly created ISOInventory.

        Raises:
            ValueError: If validation fails.
        """
        start = datetime.now(timezone.utc)
        self._get_org_or_raise(org_id)

        # Check for duplicate inventory
        for inv in self._inventories.values():
            if inv.org_id == org_id and inv.reporting_year == reporting_year:
                raise ValueError(
                    f"Inventory already exists for org {org_id} year {reporting_year}"
                )

        # Determine consolidation approach from organizational boundary
        org_boundary = self._org_boundaries.get(org_id)
        consolidation = (
            org_boundary.consolidation_approach
            if org_boundary
            else self.config.default_consolidation_approach
        )

        # Validate reporting period
        if period_start and period_end:
            self._validate_reporting_period(period_start, period_end)

        inventory = ISOInventory(
            org_id=org_id,
            reporting_year=reporting_year,
            period_start=period_start,
            period_end=period_end,
            consolidation_approach=consolidation,
            gwp_source=GWPSource(gwp_source),
        )
        self._inventories[inventory.id] = inventory

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Created inventory (id=%s) for org '%s' year %d (GWP=%s) in %.1f ms",
            inventory.id, org_id, reporting_year, gwp_source, elapsed_ms,
        )
        return inventory

    def get_inventory(self, inventory_id: str) -> Optional[ISOInventory]:
        """Retrieve an inventory by ID."""
        return self._inventories.get(inventory_id)

    def list_inventories(self, org_id: str) -> List[ISOInventory]:
        """List all inventories for an organization, sorted by year descending."""
        inventories = [
            inv for inv in self._inventories.values()
            if inv.org_id == org_id
        ]
        return sorted(inventories, key=lambda i: i.reporting_year, reverse=True)

    # ------------------------------------------------------------------
    # Reporting Period Validation
    # ------------------------------------------------------------------

    def _validate_reporting_period(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """
        Validate that the reporting period meets minimum length requirements.

        Per ISO 14064-1, the reporting period should be at least 12 months
        to allow meaningful comparison across years.

        Args:
            period_start: Period start date.
            period_end: Period end date.

        Raises:
            ValueError: If period is too short or end precedes start.
        """
        if period_end <= period_start:
            raise ValueError("Reporting period end must be after start")

        # Calculate duration in months (approximate)
        months = (
            (period_end.year - period_start.year) * 12
            + (period_end.month - period_start.month)
        )
        if period_end.day >= period_start.day:
            pass  # Full month included
        else:
            months -= 1

        min_months = self.config.minimum_completeness_pct  # Re-use config field
        # Use the standard minimum of 12 months
        if months < 12:
            raise ValueError(
                f"Reporting period must be at least 12 months, "
                f"got approximately {months} months "
                f"(from {period_start.date()} to {period_end.date()})"
            )

        logger.debug(
            "Reporting period validated: %s to %s (~%d months)",
            period_start.date(), period_end.date(), months,
        )

    # ------------------------------------------------------------------
    # Entity Consolidation Calculations
    # ------------------------------------------------------------------

    def calculate_equity_share(
        self,
        entity_id: str,
        gross_emissions: Decimal,
        ownership_pct_override: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Apply equity share consolidation to entity emissions.

        Per ISO 14064-1 Clause 5.1, under the equity share approach,
        an organization accounts for emissions proportional to its
        ownership percentage in the entity.

        Args:
            entity_id: Entity ID.
            gross_emissions: Gross emissions before equity adjustment (tCO2e).
            ownership_pct_override: Override ownership percentage (0-100).

        Returns:
            Adjusted emissions (tCO2e) = gross * (ownership_pct / 100).

        Raises:
            ValueError: If entity not found or invalid percentage.
        """
        if ownership_pct_override is not None:
            pct = ownership_pct_override
        else:
            pct = self._lookup_ownership_pct(entity_id)

        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid ownership percentage: {pct}")

        adjusted = (gross_emissions * pct / Decimal("100")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        logger.debug(
            "Equity share: entity=%s, gross=%.4f, pct=%.1f%%, adjusted=%.4f",
            entity_id, gross_emissions, pct, adjusted,
        )
        return adjusted

    def consolidate_entity_emissions(
        self,
        org_id: str,
        entity_emissions: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Consolidate emissions from all entities using the selected approach.

        Applies the organizational boundary's consolidation approach to
        each entity's emissions and produces a consolidated total.

        Args:
            org_id: Organization ID.
            entity_emissions: Dict of entity_id to gross emissions (tCO2e).

        Returns:
            Dict with:
              - total_tco2e: Consolidated total.
              - by_entity: Per-entity adjusted emissions.
              - approach: Consolidation approach used.

        Raises:
            ValueError: If organization not found.
        """
        start = datetime.now(timezone.utc)
        org = self._get_org_or_raise(org_id)

        org_boundary = self._org_boundaries.get(org_id)
        approach = (
            org_boundary.consolidation_approach
            if org_boundary
            else self.config.default_consolidation_approach
        )

        by_entity: Dict[str, Decimal] = {}
        total = Decimal("0")

        for entity in org.entities:
            if not entity.active:
                continue
            if entity.id not in entity_emissions:
                continue

            gross = entity_emissions[entity.id]

            if approach == ConsolidationApproach.EQUITY_SHARE:
                adjusted = self.calculate_equity_share(entity.id, gross)
            else:
                # Control approaches: 100% of controlled entities
                adjusted = gross

            by_entity[entity.id] = adjusted
            total += adjusted

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Consolidated %d entities for org '%s' (approach=%s): total=%.4f tCO2e in %.1f ms",
            len(by_entity), org.name, approach.value, total, elapsed_ms,
        )

        return {
            "total_tco2e": total,
            "by_entity": by_entity,
            "approach": approach.value,
            "entity_count": len(by_entity),
        }

    # ------------------------------------------------------------------
    # Significance Assessment Helpers
    # ------------------------------------------------------------------

    def assess_category_significance(
        self,
        category: ISOCategory,
        category_emissions: Decimal,
        total_emissions: Decimal,
    ) -> Dict[str, Any]:
        """
        Assess whether an indirect category is significant.

        Per ISO 14064-1 Clause 5.2.2, indirect emission categories (3-6)
        shall be assessed for significance. A category is significant if
        its emissions exceed the configured threshold percentage.

        Args:
            category: ISO 14064-1 category.
            category_emissions: Emissions for this category (tCO2e).
            total_emissions: Total inventory emissions (tCO2e).

        Returns:
            Dict with significance assessment result.
        """
        if total_emissions <= 0:
            pct = Decimal("0")
        else:
            pct = (category_emissions / total_emissions * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

        threshold = self.config.significance_threshold_percent
        is_significant = pct >= threshold

        significance = (
            SignificanceLevel.SIGNIFICANT
            if is_significant
            else SignificanceLevel.NOT_SIGNIFICANT
        )

        result = {
            "category": category.value,
            "category_name": ISO_CATEGORY_NAMES.get(category, category.value),
            "emissions_tco2e": str(category_emissions),
            "percentage_of_total": str(pct),
            "threshold_pct": str(threshold),
            "significance": significance.value,
            "is_significant": is_significant,
        }

        logger.info(
            "Significance assessment: %s = %.2f%% of total -> %s",
            category.value, pct, significance.value,
        )
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_org_or_raise(self, org_id: str) -> Organization:
        """Retrieve organization or raise ValueError."""
        org = self._organizations.get(org_id)
        if org is None:
            raise ValueError(f"Organization not found: {org_id}")
        return org

    def _get_entity_in_org_or_raise(
        self,
        org: Organization,
        entity_id: str,
    ) -> Entity:
        """Retrieve entity within an organization or raise ValueError."""
        for entity in org.entities:
            if entity.id == entity_id:
                return entity
        raise ValueError(
            f"Entity not found: {entity_id} in org {org.id}"
        )

    def _lookup_ownership_pct(self, entity_id: str) -> Decimal:
        """Look up ownership percentage for an entity across all orgs."""
        for org in self._organizations.values():
            for entity in org.entities:
                if entity.id == entity_id:
                    return entity.ownership_pct
        raise ValueError(f"Entity not found: {entity_id}")

    def _get_inventory_or_raise(self, inventory_id: str) -> ISOInventory:
        """Retrieve inventory or raise ValueError."""
        inventory = self._inventories.get(inventory_id)
        if inventory is None:
            raise ValueError(f"Inventory not found: {inventory_id}")
        return inventory
