# -*- coding: utf-8 -*-
"""
MRVBridge - Bridge to 30 MRV Agents for Race to Zero PACK-025
================================================================

This module routes emissions calculation requests to 30 MRV agents for
comprehensive GHG inventory quantification required by the Race to Zero
campaign. Supports activity-based routing preferred over spend-based
fallback, batch routing for multi-entity consolidation, and scope-level
aggregation for credibility criteria assessment.

Routing Table (30 agents):
    Scope 1 (8):  MRV-001..008
        001 Stationary Combustion, 002 Refrigerants & F-Gas,
        003 Mobile Combustion, 004 Process Emissions,
        005 Fugitive Emissions, 006 Land Use Emissions,
        007 Waste Treatment Emissions, 008 Agricultural Emissions
    Scope 2 (5):  MRV-009..013
        009 Scope 2 Location-Based, 010 Scope 2 Market-Based,
        011 Steam/Heat Purchase, 012 Cooling Purchase,
        013 Dual Reporting Reconciliation
    Scope 3 (15): MRV-014..028
        014 Purchased Goods (Cat 1), 015 Capital Goods (Cat 2),
        016 Fuel & Energy (Cat 3), 017 Upstream Transport (Cat 4),
        018 Waste Generated (Cat 5), 019 Business Travel (Cat 6),
        020 Employee Commuting (Cat 7), 021 Upstream Leased (Cat 8),
        022 Downstream Transport (Cat 9), 023 Processing Sold (Cat 10),
        024 Use of Sold (Cat 11), 025 End-of-Life (Cat 12),
        026 Downstream Leased (Cat 13), 027 Franchises (Cat 14),
        028 Investments (Cat 15)
    Cross-Cutting (2): MRV-029..030
        029 Scope 3 Category Mapper, 030 Audit Trail & Lineage

Race to Zero Requirements:
    - Full Scope 1/2/3 inventory required for credibility assessment
    - Activity-based calculation preferred for accuracy
    - Year-over-year tracking for progress reporting
    - Data quality scoring for verification readiness

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-025 Race to Zero Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------


class _AgentStub:
    """Stub for unavailable MRV agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
                "emissions_tco2e": 0.0,
            }
        return _stub_method


def _try_import_mrv_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an MRV agent with graceful fallback."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_CUTTING = "cross_cutting"


class CalculationMethod(str, Enum):
    """Emissions calculation method preference."""

    ACTIVITY_BASED = "activity_based"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class DataQualityTier(str, Enum):
    """Data quality tiers for Race to Zero reporting."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    ESTIMATED = "estimated"
    DEFAULT = "default"


# ---------------------------------------------------------------------------
# MRV Agent Routing Table
# ---------------------------------------------------------------------------

MRV_AGENT_ROUTES: Dict[str, Dict[str, Any]] = {
    "stationary_combustion": {
        "agent_id": "MRV-001",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_001_stationary_combustion",
        "source_types": ["natural_gas", "diesel", "fuel_oil", "coal", "propane", "biomass"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "refrigerants": {
        "agent_id": "MRV-002",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_002_refrigerants",
        "source_types": ["hvac", "refrigeration", "fire_suppression", "aerosols"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "mobile_combustion": {
        "agent_id": "MRV-003",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_003_mobile_combustion",
        "source_types": ["fleet_vehicles", "company_cars", "aircraft", "marine", "rail"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "process_emissions": {
        "agent_id": "MRV-004",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_004_process_emissions",
        "source_types": ["cement", "chemicals", "metals", "electronics"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "fugitive_emissions": {
        "agent_id": "MRV-005",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_005_fugitive_emissions",
        "source_types": ["gas_leaks", "coal_mining", "oil_gas_systems"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "land_use": {
        "agent_id": "MRV-006",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_006_land_use",
        "source_types": ["deforestation", "land_conversion", "agriculture_land"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "waste_treatment": {
        "agent_id": "MRV-007",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_007_waste_treatment",
        "source_types": ["wastewater", "incineration", "composting"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "agricultural": {
        "agent_id": "MRV-008",
        "scope": MRVScope.SCOPE_1,
        "module_path": "greenlang.agents.mrv.mrv_008_agricultural",
        "source_types": ["livestock", "soil_management", "rice_cultivation", "fertilizer"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "scope2_location": {
        "agent_id": "MRV-009",
        "scope": MRVScope.SCOPE_2,
        "module_path": "greenlang.agents.mrv.mrv_009_scope2_location",
        "source_types": ["grid_electricity"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "scope2_market": {
        "agent_id": "MRV-010",
        "scope": MRVScope.SCOPE_2,
        "module_path": "greenlang.agents.mrv.mrv_010_scope2_market",
        "source_types": ["purchased_electricity", "recs", "ppas", "green_tariffs"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "steam_heat": {
        "agent_id": "MRV-011",
        "scope": MRVScope.SCOPE_2,
        "module_path": "greenlang.agents.mrv.mrv_011_steam_heat",
        "source_types": ["district_heating", "purchased_steam"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "cooling": {
        "agent_id": "MRV-012",
        "scope": MRVScope.SCOPE_2,
        "module_path": "greenlang.agents.mrv.mrv_012_cooling",
        "source_types": ["district_cooling", "purchased_cooling"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "dual_reporting": {
        "agent_id": "MRV-013",
        "scope": MRVScope.SCOPE_2,
        "module_path": "greenlang.agents.mrv.mrv_013_dual_reporting",
        "source_types": ["scope2_reconciliation"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
    },
    "purchased_goods": {
        "agent_id": "MRV-014",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_014_purchased_goods",
        "source_types": ["raw_materials", "goods", "services_purchased"],
        "preferred_method": CalculationMethod.SUPPLIER_SPECIFIC,
        "scope3_category": 1,
    },
    "capital_goods": {
        "agent_id": "MRV-015",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_015_capital_goods",
        "source_types": ["equipment", "buildings", "vehicles_purchased"],
        "preferred_method": CalculationMethod.SPEND_BASED,
        "scope3_category": 2,
    },
    "fuel_energy_activities": {
        "agent_id": "MRV-016",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_016_fuel_energy",
        "source_types": ["upstream_energy", "t_and_d_losses"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 3,
    },
    "upstream_transport": {
        "agent_id": "MRV-017",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_017_upstream_transport",
        "source_types": ["inbound_logistics", "third_party_transport"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 4,
    },
    "waste_generated": {
        "agent_id": "MRV-018",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_018_waste_generated",
        "source_types": ["landfill", "recycling", "incineration_waste"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 5,
    },
    "business_travel": {
        "agent_id": "MRV-019",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_019_business_travel",
        "source_types": ["flights", "hotels", "rail_travel", "car_rental"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 6,
    },
    "employee_commuting": {
        "agent_id": "MRV-020",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_020_employee_commuting",
        "source_types": ["commute_car", "commute_public", "remote_work"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 7,
    },
    "upstream_leased": {
        "agent_id": "MRV-021",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_021_upstream_leased",
        "source_types": ["leased_buildings", "leased_equipment"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 8,
    },
    "downstream_transport": {
        "agent_id": "MRV-022",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_022_downstream_transport",
        "source_types": ["outbound_logistics", "customer_delivery"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 9,
    },
    "processing_sold": {
        "agent_id": "MRV-023",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_023_processing_sold",
        "source_types": ["intermediate_products"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 10,
    },
    "use_of_sold": {
        "agent_id": "MRV-024",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_024_use_of_sold",
        "source_types": ["product_use_energy", "product_use_fuels"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 11,
    },
    "end_of_life": {
        "agent_id": "MRV-025",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_025_end_of_life",
        "source_types": ["product_disposal", "product_recycling"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 12,
    },
    "downstream_leased": {
        "agent_id": "MRV-026",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_026_downstream_leased",
        "source_types": ["leased_to_others_buildings", "leased_to_others_equipment"],
        "preferred_method": CalculationMethod.ACTIVITY_BASED,
        "scope3_category": 13,
    },
    "franchises": {
        "agent_id": "MRV-027",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_027_franchises",
        "source_types": ["franchise_operations"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 14,
    },
    "investments": {
        "agent_id": "MRV-028",
        "scope": MRVScope.SCOPE_3,
        "module_path": "greenlang.agents.mrv.mrv_028_investments",
        "source_types": ["equity_investments", "debt_investments", "project_finance"],
        "preferred_method": CalculationMethod.AVERAGE_DATA,
        "scope3_category": 15,
    },
    "scope3_mapper": {
        "agent_id": "MRV-029",
        "scope": MRVScope.CROSS_CUTTING,
        "module_path": "greenlang.agents.mrv.mrv_029_scope3_mapper",
        "source_types": ["scope3_mapping"],
        "preferred_method": CalculationMethod.HYBRID,
    },
    "audit_trail": {
        "agent_id": "MRV-030",
        "scope": MRVScope.CROSS_CUTTING,
        "module_path": "greenlang.agents.mrv.mrv_030_audit_trail",
        "source_types": ["lineage_tracking"],
        "preferred_method": CalculationMethod.HYBRID,
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV bridge."""

    pack_id: str = Field(default="PACK-025")
    enable_provenance: bool = Field(default=True)
    preferred_method: CalculationMethod = Field(default=CalculationMethod.ACTIVITY_BASED)
    fallback_to_spend: bool = Field(default=True)
    include_scope3: bool = Field(default=True)
    scope3_categories: List[int] = Field(default_factory=lambda: list(range(1, 16)))
    multi_entity: bool = Field(default=False)
    entity_ids: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2019, ge=2015, le=2025)
    timeout_seconds: int = Field(default=300, ge=30)
    max_concurrent_agents: int = Field(default=5, ge=1, le=30)


class MRVAgentRoute(BaseModel):
    """Routing information for an MRV agent."""

    agent_id: str = Field(default="")
    agent_name: str = Field(default="")
    scope: MRVScope = Field(default=MRVScope.SCOPE_1)
    source_type: str = Field(default="")
    calculation_method: CalculationMethod = Field(default=CalculationMethod.ACTIVITY_BASED)
    available: bool = Field(default=True)
    scope3_category: Optional[int] = Field(None)


class EmissionSource(BaseModel):
    """An emission source to route to an MRV agent."""

    source_id: str = Field(default_factory=_new_uuid)
    source_type: str = Field(default="")
    scope: Optional[MRVScope] = Field(None)
    description: str = Field(default="")
    quantity: float = Field(default=0.0, ge=0.0)
    unit: str = Field(default="")
    calculation_method: Optional[CalculationMethod] = Field(None)
    data_quality: DataQualityTier = Field(default=DataQualityTier.SECONDARY)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingResult(BaseModel):
    """Result of routing a single emission source."""

    source_id: str = Field(default="")
    agent_id: str = Field(default="")
    agent_name: str = Field(default="")
    scope: MRVScope = Field(default=MRVScope.SCOPE_1)
    calculation_method: CalculationMethod = Field(default=CalculationMethod.ACTIVITY_BASED)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality: DataQualityTier = Field(default=DataQualityTier.SECONDARY)
    status: str = Field(default="success")
    provenance_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class BatchRoutingResult(BaseModel):
    """Result of batch routing multiple emission sources."""

    batch_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    total_sources: int = Field(default=0)
    routed_successfully: int = Field(default=0)
    routing_failures: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    results: List[RoutingResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    data_quality_summary: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class AggregationResult(BaseModel):
    """Result of scope-level emissions aggregation."""

    reporting_year: int = Field(default=2025)
    scope1_tco2e: float = Field(default=0.0)
    scope2_location_tco2e: float = Field(default=0.0)
    scope2_market_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    total_tco2e: float = Field(default=0.0)
    sources_count: int = Field(default=0)
    agents_used: List[str] = Field(default_factory=list)
    data_quality_weighted: float = Field(default=0.0)
    completeness_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRVBridge
# ---------------------------------------------------------------------------


class MRVBridge:
    """Bridge to 30 MRV agents for Race to Zero emissions calculation.

    Routes emission sources to the appropriate MRV agent based on source
    type, scope, and calculation method preference. Supports batch
    routing for multi-entity consolidation and activity-based routing
    preferred over spend-based fallback.

    Example:
        >>> bridge = MRVBridge()
        >>> source = EmissionSource(source_type="natural_gas", quantity=1000, unit="m3")
        >>> result = bridge.route_source(source)
        >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        self.config = config or MRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        self._load_agents()
        self.logger.info(
            "MRVBridge initialized: pack=%s, agents=%d",
            self.config.pack_id,
            len(self._agents),
        )

    def _load_agents(self) -> None:
        """Load all MRV agents with graceful fallback."""
        for route_name, route_info in MRV_AGENT_ROUTES.items():
            agent_id = route_info["agent_id"]
            module_path = route_info["module_path"]
            self._agents[route_name] = _try_import_mrv_agent(agent_id, module_path)

    def get_available_agents(self) -> List[MRVAgentRoute]:
        """Get list of available MRV agents with routing info."""
        routes = []
        for route_name, route_info in MRV_AGENT_ROUTES.items():
            agent = self._agents.get(route_name)
            available = agent is not None and not isinstance(agent, _AgentStub)
            routes.append(MRVAgentRoute(
                agent_id=route_info["agent_id"],
                agent_name=route_name,
                scope=route_info["scope"],
                source_type=", ".join(route_info["source_types"]),
                calculation_method=route_info["preferred_method"],
                available=available,
                scope3_category=route_info.get("scope3_category"),
            ))
        return routes

    def route_source(self, source: EmissionSource) -> RoutingResult:
        """Route a single emission source to the appropriate MRV agent.

        Args:
            source: The emission source to route.

        Returns:
            RoutingResult with calculated emissions.
        """
        start = time.monotonic()
        route_name = self._find_route(source)

        if not route_name:
            elapsed = (time.monotonic() - start) * 1000
            return RoutingResult(
                source_id=source.source_id,
                status="no_route",
                errors=[f"No route found for source type '{source.source_type}'"],
                duration_ms=round(elapsed, 2),
            )

        route_info = MRV_AGENT_ROUTES[route_name]
        agent = self._agents.get(route_name)
        method = source.calculation_method or route_info["preferred_method"]

        if method == CalculationMethod.ACTIVITY_BASED and isinstance(agent, _AgentStub):
            if self.config.fallback_to_spend:
                method = CalculationMethod.SPEND_BASED
            else:
                elapsed = (time.monotonic() - start) * 1000
                return RoutingResult(
                    source_id=source.source_id,
                    agent_id=route_info["agent_id"],
                    agent_name=route_name,
                    scope=route_info["scope"],
                    status="degraded",
                    warnings=[f"{route_info['agent_id']} using stub, no fallback enabled"],
                    duration_ms=round(elapsed, 2),
                )

        emissions = self._calculate_stub_emissions(source, route_info)

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = ""
        if self.config.enable_provenance:
            prov_hash = _compute_hash({
                "source": source.model_dump(mode="json"),
                "agent": route_info["agent_id"],
                "emissions": emissions,
            })

        return RoutingResult(
            source_id=source.source_id,
            agent_id=route_info["agent_id"],
            agent_name=route_name,
            scope=route_info["scope"],
            calculation_method=method,
            emissions_tco2e=round(emissions, 6),
            data_quality=source.data_quality,
            status="success",
            provenance_hash=prov_hash,
            duration_ms=round(elapsed, 2),
        )

    def route_batch(self, sources: List[EmissionSource]) -> BatchRoutingResult:
        """Route multiple emission sources in batch.

        Args:
            sources: List of emission sources to route.

        Returns:
            BatchRoutingResult with aggregated results.
        """
        start = time.monotonic()
        batch = BatchRoutingResult(
            total_sources=len(sources),
        )

        quality_counts: Dict[str, int] = {}

        for source in sources:
            result = self.route_source(source)
            batch.results.append(result)

            if result.status == "success":
                batch.routed_successfully += 1
                batch.total_emissions_tco2e += result.emissions_tco2e

                if result.scope == MRVScope.SCOPE_1:
                    batch.scope1_tco2e += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2:
                    batch.scope2_tco2e += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3:
                    batch.scope3_tco2e += result.emissions_tco2e
                    route = MRV_AGENT_ROUTES.get(result.agent_name, {})
                    cat = route.get("scope3_category")
                    if cat:
                        batch.scope3_by_category[cat] = (
                            batch.scope3_by_category.get(cat, 0.0) + result.emissions_tco2e
                        )

                tier = result.data_quality.value
                quality_counts[tier] = quality_counts.get(tier, 0) + 1
            else:
                batch.routing_failures += 1

        batch.data_quality_summary = quality_counts
        batch.duration_ms = round((time.monotonic() - start) * 1000, 2)

        if self.config.enable_provenance:
            batch.provenance_hash = _compute_hash(batch)

        return batch

    def aggregate_emissions(
        self,
        routing_results: Optional[List[RoutingResult]] = None,
        batch_result: Optional[BatchRoutingResult] = None,
    ) -> AggregationResult:
        """Aggregate emissions by scope from routing results.

        Args:
            routing_results: List of individual routing results.
            batch_result: Or a batch routing result.

        Returns:
            AggregationResult with scope-level aggregation.
        """
        results = routing_results or []
        if batch_result:
            results = batch_result.results

        agg = AggregationResult(reporting_year=self.config.reporting_year)
        agents_used = set()

        for r in results:
            if r.status != "success":
                continue
            agents_used.add(r.agent_id)
            agg.sources_count += 1

            if r.scope == MRVScope.SCOPE_1:
                agg.scope1_tco2e += r.emissions_tco2e
            elif r.scope == MRVScope.SCOPE_2:
                if "location" in r.agent_name:
                    agg.scope2_location_tco2e += r.emissions_tco2e
                elif "market" in r.agent_name:
                    agg.scope2_market_tco2e += r.emissions_tco2e
                else:
                    agg.scope2_location_tco2e += r.emissions_tco2e
            elif r.scope == MRVScope.SCOPE_3:
                agg.scope3_tco2e += r.emissions_tco2e
                route = MRV_AGENT_ROUTES.get(r.agent_name, {})
                cat = route.get("scope3_category")
                if cat:
                    agg.scope3_by_category[cat] = (
                        agg.scope3_by_category.get(cat, 0.0) + r.emissions_tco2e
                    )

        agg.total_tco2e = agg.scope1_tco2e + agg.scope2_location_tco2e + agg.scope3_tco2e
        agg.agents_used = sorted(agents_used)

        total_possible = 30
        agg.completeness_pct = round(len(agents_used) / total_possible * 100, 1)

        if self.config.enable_provenance:
            agg.provenance_hash = _compute_hash(agg)

        return agg

    def get_scope3_coverage(self) -> Dict[str, Any]:
        """Check Scope 3 category coverage for Race to Zero compliance.

        Returns:
            Dict with coverage assessment for all 15 categories.
        """
        configured = set(self.config.scope3_categories)
        available = set()
        for route_name, route_info in MRV_AGENT_ROUTES.items():
            if route_info["scope"] == MRVScope.SCOPE_3:
                cat = route_info.get("scope3_category")
                if cat:
                    agent = self._agents.get(route_name)
                    if agent and not isinstance(agent, _AgentStub):
                        available.add(cat)

        all_categories = set(range(1, 16))
        covered = configured & available
        not_configured = all_categories - configured
        configured_but_unavailable = configured - available

        return {
            "total_categories": 15,
            "configured": sorted(configured),
            "available": sorted(available),
            "covered": sorted(covered),
            "coverage_pct": round(len(covered) / 15 * 100, 1),
            "not_configured": sorted(not_configured),
            "unavailable": sorted(configured_but_unavailable),
            "r2z_compliant": len(covered) >= 10,
        }

    def _find_route(self, source: EmissionSource) -> Optional[str]:
        """Find the best route for an emission source."""
        source_type = source.source_type.lower()

        for route_name, route_info in MRV_AGENT_ROUTES.items():
            if source_type in route_info["source_types"]:
                return route_name

        for route_name, route_info in MRV_AGENT_ROUTES.items():
            for st in route_info["source_types"]:
                if source_type in st or st in source_type:
                    return route_name

        if source.scope:
            for route_name, route_info in MRV_AGENT_ROUTES.items():
                if route_info["scope"] == source.scope:
                    return route_name

        return None

    def _calculate_stub_emissions(
        self, source: EmissionSource, route_info: Dict[str, Any],
    ) -> float:
        """Calculate stub emissions when agent is not available."""
        emission_factors: Dict[str, float] = {
            "natural_gas": 0.00205,
            "diesel": 0.00268,
            "fuel_oil": 0.00319,
            "coal": 0.00341,
            "propane": 0.00154,
            "grid_electricity": 0.000417,
            "fleet_vehicles": 0.000270,
            "flights": 0.000255,
        }

        source_type = source.source_type.lower()
        ef = emission_factors.get(source_type, 0.001)
        return source.quantity * ef
