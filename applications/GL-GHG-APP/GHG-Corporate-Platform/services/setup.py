"""
GHG Corporate Platform -- Service Facade

This module provides the ``GHGPlatform`` class, a unified facade that
composes all service engines and manages their shared state.  It is the
single entry point for the API layer and external integrations.

Features:
  - Creates and wires all service instances with shared in-memory stores
  - Provides convenience methods that orchestrate multi-engine workflows
  - Manages application lifecycle (initialization, configuration, shutdown)
  - Computes dashboard metrics from live inventory data

Example:
    >>> from services.setup import GHGPlatform
    >>> platform = GHGPlatform()
    >>> org = platform.create_organization(CreateOrganizationRequest(...))
    >>> inv = platform.create_inventory(org.id, 2025)
    >>> platform.aggregate_all(inv.id)
    >>> dashboard = platform.get_dashboard(org.id, 2025)
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import GHGAppConfig, ReportFormat, Scope
from .models import (
    AddEntityRequest,
    AddFindingRequest,
    CompletenessResult,
    CreateInventoryRequest,
    CreateOrganizationRequest,
    DashboardMetrics,
    Entity,
    ExclusionRequest,
    GHGInventory,
    IntensityMetric,
    Organization,
    RecalculateBaseYearRequest,
    Report,
    SetBaseYearRequest,
    SetTargetRequest,
    Target,
    UncertaintyResult,
    UpdateEntityRequest,
    VerificationRecord,
    _now,
)

from .inventory_manager import InventoryManager
from .base_year_manager import BaseYearManager
from .scope_aggregator import ScopeAggregator
from .intensity_calculator import IntensityCalculator
from .uncertainty_engine import UncertaintyEngine
from .completeness_checker import CompletenessChecker
from .report_generator import ReportGenerator
from .verification_workflow import VerificationWorkflow
from .target_tracker import TargetTracker

logger = logging.getLogger(__name__)


class GHGPlatform:
    """
    Unified facade composing all GHG Corporate Platform engines.

    Holds shared in-memory stores and wires every engine to
    the same data references so that changes propagate immediately.

    Attributes:
        config: Application configuration.
        inventory_mgr: Organization and inventory management.
        base_year_mgr: Base year and recalculation management.
        aggregator: Scope 1/2/3 emissions aggregation.
        intensity_calc: Intensity metric calculation.
        uncertainty: Monte Carlo uncertainty engine.
        completeness: GHG Protocol completeness checker.
        reporter: Multi-format report generator.
        verification: Verification workflow management.
        targets: Target tracking and SBTi alignment.
    """

    def __init__(self, config: Optional[GHGAppConfig] = None) -> None:
        """
        Initialize the GHG Platform with all engines.

        Args:
            config: Optional configuration override.
        """
        self.config = config or GHGAppConfig()

        # Initialize InventoryManager first (owns the shared stores)
        self.inventory_mgr = InventoryManager(self.config)

        # Shared references for cross-engine coordination
        inventory_store = self.inventory_mgr._inventories

        # Wire all engines to the shared inventory store
        self.base_year_mgr = BaseYearManager(self.config, inventory_store)
        self.aggregator = ScopeAggregator(self.config, inventory_store)
        self.intensity_calc = IntensityCalculator(self.config, inventory_store)
        self.uncertainty = UncertaintyEngine(self.config)
        self.completeness = CompletenessChecker(self.config)
        self.reporter = ReportGenerator(self.config, inventory_store)
        self.verification = VerificationWorkflow(self.config, inventory_store)
        self.targets = TargetTracker(self.config, inventory_store)

        logger.info(
            "GHGPlatform v%s initialized with all %d engines",
            self.config.app_version,
            9,
        )

    # ------------------------------------------------------------------
    # Organization & Boundary Shortcuts
    # ------------------------------------------------------------------

    def create_organization(self, data: CreateOrganizationRequest) -> Organization:
        """Create a new organization."""
        return self.inventory_mgr.create_organization(data)

    def add_entity(self, org_id: str, data: AddEntityRequest) -> Entity:
        """Add an entity to an organization."""
        return self.inventory_mgr.add_entity(org_id, data)

    def update_entity(
        self,
        org_id: str,
        entity_id: str,
        data: UpdateEntityRequest,
    ) -> Entity:
        """Update an entity."""
        return self.inventory_mgr.update_entity(org_id, entity_id, data)

    # ------------------------------------------------------------------
    # Inventory Lifecycle Shortcuts
    # ------------------------------------------------------------------

    def create_inventory(self, org_id: str, year: int) -> GHGInventory:
        """Create a new GHG inventory."""
        return self.inventory_mgr.create_inventory(org_id, year)

    def get_inventory(self, inventory_id: str) -> Optional[GHGInventory]:
        """Get an inventory by ID."""
        return self.inventory_mgr.get_inventory(inventory_id)

    # ------------------------------------------------------------------
    # Aggregation Orchestration
    # ------------------------------------------------------------------

    def aggregate_all(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> GHGInventory:
        """
        Aggregate all scopes, compute intensity, uncertainty, completeness.

        This is the full calculation pipeline:
          1. Aggregate Scope 1/2/3 emissions
          2. Calculate intensity metrics
          3. Run uncertainty assessment
          4. Check completeness

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional pre-loaded entity data.

        Returns:
            Fully calculated GHGInventory.
        """
        # Step 1: Scope aggregation
        inventory = self.aggregator.aggregate_all(inventory_id, entity_data)

        # Step 2: Intensity metrics
        org = self._resolve_organization(inventory.org_id)
        if org:
            metrics = self._calculate_org_intensities(inventory, org)
            inventory.intensity_metrics = metrics

        # Step 3: Uncertainty
        uncertainty_result = self.uncertainty.run_monte_carlo(inventory)
        inventory.uncertainty = uncertainty_result

        # Step 4: Completeness
        completeness_result = self.completeness.check_completeness(inventory)
        inventory.completeness = completeness_result
        inventory.data_quality_score = completeness_result.data_quality_score

        inventory.updated_at = _now()

        logger.info(
            "Full pipeline complete for inventory %s: total=%.2f tCO2e, "
            "DQ=%.1f, completeness=%.1f%%",
            inventory_id,
            inventory.grand_total_tco2e,
            inventory.data_quality_score,
            completeness_result.overall_pct,
        )
        return inventory

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard(
        self,
        org_id: str,
        year: int,
    ) -> DashboardMetrics:
        """
        Generate dashboard metrics for an organization-year.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            DashboardMetrics for frontend consumption.
        """
        inventories = self.inventory_mgr.list_inventories(org_id)
        current = None
        for inv in inventories:
            if inv.year == year:
                current = inv
                break

        if current is None:
            return DashboardMetrics(
                org_id=org_id,
                year=year,
            )

        # YoY change
        yoy = self.base_year_mgr.calculate_yoy_change(org_id, year)
        yoy_pct = None
        if yoy.get("change_pct"):
            try:
                yoy_pct = Decimal(str(yoy["change_pct"]))
            except Exception:
                pass

        # Top categories
        top_cats: List[Dict[str, Any]] = []
        if current.scope3 and current.scope3.by_category:
            sorted_cats = sorted(
                current.scope3.by_category.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for cat_name, cat_value in sorted_cats:
                top_cats.append({
                    "category": cat_name,
                    "tco2e": str(cat_value),
                })

        # Target progress
        target_progress = None
        org_targets = self.targets.get_targets(org_id)
        if org_targets:
            progress_values = [
                t.current_progress_pct for t in org_targets
                if t.current_emissions is not None
            ]
            if progress_values:
                target_progress = sum(progress_values) / Decimal(str(len(progress_values)))

        return DashboardMetrics(
            org_id=org_id,
            year=year,
            total_emissions=current.grand_total_tco2e,
            scope1_total=current.scope1.total_tco2e if current.scope1 else Decimal("0"),
            scope2_location_total=(
                current.scope2_location.total_tco2e if current.scope2_location else Decimal("0")
            ),
            scope2_market_total=(
                current.scope2_market.total_tco2e if current.scope2_market else Decimal("0")
            ),
            scope3_total=current.scope3.total_tco2e if current.scope3 else Decimal("0"),
            yoy_change_pct=yoy_pct,
            intensity_metrics=current.intensity_metrics,
            data_quality_score=current.data_quality_score,
            completeness_pct=(
                current.completeness.overall_pct if current.completeness else Decimal("0")
            ),
            target_progress_pct=target_progress,
            top_categories=top_cats,
            scope3_breakdown=(
                {k: v for k, v in current.scope3.by_category.items()}
                if current.scope3
                else {}
            ),
            biogenic_co2=sum(
                s.biogenic_co2
                for s in [current.scope1, current.scope2_location, current.scope3]
                if s is not None
            ),
        )

    # ------------------------------------------------------------------
    # Health / Info
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Return platform health status.

        Returns:
            Dict with engine status and metadata.
        """
        return {
            "status": "healthy",
            "version": self.config.app_version,
            "engines": {
                "inventory_manager": "ok",
                "base_year_manager": "ok",
                "scope_aggregator": "ok",
                "intensity_calculator": "ok",
                "uncertainty_engine": "ok",
                "completeness_checker": "ok",
                "report_generator": "ok",
                "verification_workflow": "ok",
                "target_tracker": "ok",
            },
            "inventory_count": len(self.inventory_mgr._inventories),
            "organization_count": len(self.inventory_mgr._organizations),
        }

    def get_platform_info(self) -> Dict[str, Any]:
        """Return platform metadata."""
        return {
            "name": self.config.app_name,
            "version": self.config.app_version,
            "mrv_agents_integrated": 28,
            "scope1_categories": 8,
            "scope2_agents": 5,
            "scope3_categories": 15,
            "supported_formats": [f.value for f in ReportFormat],
            "scopes_supported": [s.value for s in Scope],
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _resolve_organization(self, org_id: str) -> Optional[Organization]:
        """Resolve organization from ID."""
        return self.inventory_mgr.get_organization(org_id)

    def _calculate_org_intensities(
        self,
        inventory: GHGInventory,
        org: Organization,
    ) -> List[IntensityMetric]:
        """Calculate intensity metrics using organization data."""
        total_revenue = Decimal("0")
        total_fte = 0
        total_floor = Decimal("0")
        total_production = Decimal("0")
        production_unit = None

        for entity in org.entities:
            if not entity.active:
                continue
            if entity.revenue:
                total_revenue += entity.revenue
            if entity.employees:
                total_fte += entity.employees
            if entity.floor_area_m2:
                total_floor += entity.floor_area_m2
            if entity.production_units:
                total_production += entity.production_units
                if entity.production_unit_name:
                    production_unit = entity.production_unit_name

        # Convert revenue to millions
        revenue_millions = None
        if total_revenue > 0:
            revenue_millions = (total_revenue / Decimal("1000000")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return self.intensity_calc.calculate_all_intensities(
            inventory=inventory,
            org_revenue_million_usd=revenue_millions,
            org_fte_count=total_fte if total_fte > 0 else None,
            org_floor_area_m2=total_floor if total_floor > 0 else None,
            org_production_units=total_production if total_production > 0 else None,
            org_production_unit_name=production_unit,
        )
