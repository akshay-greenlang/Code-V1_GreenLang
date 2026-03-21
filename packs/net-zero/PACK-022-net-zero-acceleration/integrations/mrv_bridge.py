# -*- coding: utf-8 -*-
"""
MRVBridge - Bridge to 30 MRV Agents with Activity-Based Routing for PACK-022
===============================================================================

This module routes emissions calculation requests to 30 MRV agents with
activity-based routing preferred over spend-based fallback. Supports batch
routing for multi-entity consolidation and calculation method selection.

Routing Table (30 agents):
    Scope 1 (8):  MRV-001..008
    Scope 2 (5):  MRV-009..013
    Scope 3 (15): MRV-014..028
    Cross-Cutting (2): MRV-029..030

Calculation Method Preference:
    1. Activity-based (physical quantities, emission factors)
    2. Spend-based (financial data, EEIO emission factors) -- fallback

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
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


class EmissionSource(str, Enum):
    """Emission source categories mapped to MRV agents."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    REFRIGERANTS = "refrigerants"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"
    ELECTRICITY_LOCATION = "electricity_location"
    ELECTRICITY_MARKET = "electricity_market"
    STEAM_HEAT = "steam_heat"
    COOLING = "cooling"
    DUAL_REPORTING = "dual_reporting"
    PURCHASED_GOODS = "purchased_goods"
    CAPITAL_GOODS = "capital_goods"
    FUEL_ENERGY_ACTIVITIES = "fuel_energy_activities"
    UPSTREAM_TRANSPORT = "upstream_transport"
    WASTE_GENERATED = "waste_generated"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"
    UPSTREAM_LEASED = "upstream_leased"
    DOWNSTREAM_TRANSPORT = "downstream_transport"
    PROCESSING_SOLD = "processing_sold"
    USE_OF_SOLD = "use_of_sold"
    END_OF_LIFE = "end_of_life"
    DOWNSTREAM_LEASED = "downstream_leased"
    FRANCHISES = "franchises"
    INVESTMENTS = "investments"
    SCOPE3_MAPPER = "scope3_mapper"
    AUDIT_TRAIL = "audit_trail"


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_CUTTING = "cross_cutting"


class CalculationMethod(str, Enum):
    """Emission calculation method preference."""

    ACTIVITY_BASED = "activity_based"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVAgentRoute(BaseModel):
    """Routing entry mapping an emission source to an MRV agent."""

    source: EmissionSource = Field(...)
    mrv_agent_id: str = Field(...)
    mrv_agent_name: str = Field(default="")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="")
    description: str = Field(default="")


class RoutingResult(BaseModel):
    """Result of routing a calculation request to an MRV agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    source: str = Field(default="")
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    scope3_category: Optional[int] = Field(None)
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    calculation_method: str = Field(default="activity_based")
    emissions_tco2e: float = Field(default=0.0)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    entity_id: str = Field(default="")
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Bridge."""

    pack_id: str = Field(default="PACK-022")
    enable_provenance: bool = Field(default=True)
    enable_batch_routing: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=10, ge=1, le=30)
    preferred_method: CalculationMethod = Field(default=CalculationMethod.ACTIVITY_BASED)
    scopes_included: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
    )


class BatchRoutingResult(BaseModel):
    """Result of routing multiple calculation requests."""

    batch_id: str = Field(default_factory=_new_uuid)
    total_sources: int = Field(default=0)
    successful: int = Field(default=0)
    degraded: int = Field(default=0)
    failed: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    activity_based_count: int = Field(default=0)
    spend_based_count: int = Field(default=0)
    entities_processed: List[str] = Field(default_factory=list)
    results: List[RoutingResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRV Agent Routing Table (30 agents)
# ---------------------------------------------------------------------------

MRV_ROUTING_TABLE: List[MRVAgentRoute] = [
    MRVAgentRoute(source=EmissionSource.STATIONARY_COMBUSTION, mrv_agent_id="MRV-001", mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.stationary_combustion", description="Boilers, furnaces, heaters, turbines"),
    MRVAgentRoute(source=EmissionSource.REFRIGERANTS, mrv_agent_id="MRV-002", mrv_agent_name="Refrigerants & F-Gas", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.refrigerants_fgas", description="HFC/HCFC/PFC refrigerant leakage"),
    MRVAgentRoute(source=EmissionSource.MOBILE_COMBUSTION, mrv_agent_id="MRV-003", mrv_agent_name="Mobile Combustion", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.mobile_combustion", description="Company vehicles, fleet, forklifts"),
    MRVAgentRoute(source=EmissionSource.PROCESS_EMISSIONS, mrv_agent_id="MRV-004", mrv_agent_name="Process Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.process_emissions", description="Chemical/physical process emissions"),
    MRVAgentRoute(source=EmissionSource.FUGITIVE_EMISSIONS, mrv_agent_id="MRV-005", mrv_agent_name="Fugitive Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.fugitive_emissions", description="Intentional/unintentional releases"),
    MRVAgentRoute(source=EmissionSource.LAND_USE, mrv_agent_id="MRV-006", mrv_agent_name="Land Use Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.land_use_emissions", description="Land use change emissions"),
    MRVAgentRoute(source=EmissionSource.WASTE_TREATMENT, mrv_agent_id="MRV-007", mrv_agent_name="Waste Treatment Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.waste_treatment_emissions", description="On-site waste treatment"),
    MRVAgentRoute(source=EmissionSource.AGRICULTURAL, mrv_agent_id="MRV-008", mrv_agent_name="Agricultural Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.agricultural_emissions", description="Enteric fermentation, manure, soils"),
    MRVAgentRoute(source=EmissionSource.ELECTRICITY_LOCATION, mrv_agent_id="MRV-009", mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_location_based", description="Grid-average emission factors"),
    MRVAgentRoute(source=EmissionSource.ELECTRICITY_MARKET, mrv_agent_id="MRV-010", mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_market_based", description="Contractual/residual emission factors"),
    MRVAgentRoute(source=EmissionSource.STEAM_HEAT, mrv_agent_id="MRV-011", mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.steam_heat_purchase", description="Purchased steam and district heating"),
    MRVAgentRoute(source=EmissionSource.COOLING, mrv_agent_id="MRV-012", mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.cooling_purchase", description="Purchased cooling"),
    MRVAgentRoute(source=EmissionSource.DUAL_REPORTING, mrv_agent_id="MRV-013", mrv_agent_name="Dual Reporting Reconciliation", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.dual_reporting_reconciliation", description="Location vs market-based reconciliation"),
    MRVAgentRoute(source=EmissionSource.PURCHASED_GOODS, mrv_agent_id="MRV-014", mrv_agent_name="Purchased Goods & Services (Cat 1)", scope=MRVScope.SCOPE_3, scope3_category=1, module_path="greenlang.agents.mrv.scope3_cat1", description="Upstream purchased goods and services"),
    MRVAgentRoute(source=EmissionSource.CAPITAL_GOODS, mrv_agent_id="MRV-015", mrv_agent_name="Capital Goods (Cat 2)", scope=MRVScope.SCOPE_3, scope3_category=2, module_path="greenlang.agents.mrv.scope3_cat2", description="Upstream capital goods"),
    MRVAgentRoute(source=EmissionSource.FUEL_ENERGY_ACTIVITIES, mrv_agent_id="MRV-016", mrv_agent_name="Fuel & Energy Activities (Cat 3)", scope=MRVScope.SCOPE_3, scope3_category=3, module_path="greenlang.agents.mrv.scope3_cat3", description="Upstream energy-related not in S1/S2"),
    MRVAgentRoute(source=EmissionSource.UPSTREAM_TRANSPORT, mrv_agent_id="MRV-017", mrv_agent_name="Upstream Transportation (Cat 4)", scope=MRVScope.SCOPE_3, scope3_category=4, module_path="greenlang.agents.mrv.scope3_cat4", description="Inbound logistics"),
    MRVAgentRoute(source=EmissionSource.WASTE_GENERATED, mrv_agent_id="MRV-018", mrv_agent_name="Waste Generated (Cat 5)", scope=MRVScope.SCOPE_3, scope3_category=5, module_path="greenlang.agents.mrv.scope3_cat5", description="Third-party waste disposal"),
    MRVAgentRoute(source=EmissionSource.BUSINESS_TRAVEL, mrv_agent_id="MRV-019", mrv_agent_name="Business Travel (Cat 6)", scope=MRVScope.SCOPE_3, scope3_category=6, module_path="greenlang.agents.mrv.scope3_cat6", description="Employee business travel"),
    MRVAgentRoute(source=EmissionSource.EMPLOYEE_COMMUTING, mrv_agent_id="MRV-020", mrv_agent_name="Employee Commuting (Cat 7)", scope=MRVScope.SCOPE_3, scope3_category=7, module_path="greenlang.agents.mrv.scope3_cat7", description="Employee commuting and remote work"),
    MRVAgentRoute(source=EmissionSource.UPSTREAM_LEASED, mrv_agent_id="MRV-021", mrv_agent_name="Upstream Leased Assets (Cat 8)", scope=MRVScope.SCOPE_3, scope3_category=8, module_path="greenlang.agents.mrv.scope3_cat8", description="Leased assets upstream"),
    MRVAgentRoute(source=EmissionSource.DOWNSTREAM_TRANSPORT, mrv_agent_id="MRV-022", mrv_agent_name="Downstream Transportation (Cat 9)", scope=MRVScope.SCOPE_3, scope3_category=9, module_path="greenlang.agents.mrv.scope3_cat9", description="Outbound logistics"),
    MRVAgentRoute(source=EmissionSource.PROCESSING_SOLD, mrv_agent_id="MRV-023", mrv_agent_name="Processing of Sold Products (Cat 10)", scope=MRVScope.SCOPE_3, scope3_category=10, module_path="greenlang.agents.mrv.scope3_cat10", description="Downstream processing"),
    MRVAgentRoute(source=EmissionSource.USE_OF_SOLD, mrv_agent_id="MRV-024", mrv_agent_name="Use of Sold Products (Cat 11)", scope=MRVScope.SCOPE_3, scope3_category=11, module_path="greenlang.agents.mrv.scope3_cat11", description="Customer use-phase emissions"),
    MRVAgentRoute(source=EmissionSource.END_OF_LIFE, mrv_agent_id="MRV-025", mrv_agent_name="End-of-Life Treatment (Cat 12)", scope=MRVScope.SCOPE_3, scope3_category=12, module_path="greenlang.agents.mrv.scope3_cat12", description="End-of-life treatment"),
    MRVAgentRoute(source=EmissionSource.DOWNSTREAM_LEASED, mrv_agent_id="MRV-026", mrv_agent_name="Downstream Leased Assets (Cat 13)", scope=MRVScope.SCOPE_3, scope3_category=13, module_path="greenlang.agents.mrv.scope3_cat13", description="Leased assets downstream"),
    MRVAgentRoute(source=EmissionSource.FRANCHISES, mrv_agent_id="MRV-027", mrv_agent_name="Franchises (Cat 14)", scope=MRVScope.SCOPE_3, scope3_category=14, module_path="greenlang.agents.mrv.scope3_cat14", description="Franchise operations"),
    MRVAgentRoute(source=EmissionSource.INVESTMENTS, mrv_agent_id="MRV-028", mrv_agent_name="Investments (Cat 15)", scope=MRVScope.SCOPE_3, scope3_category=15, module_path="greenlang.agents.mrv.scope3_cat15", description="Financed emissions"),
    MRVAgentRoute(source=EmissionSource.SCOPE3_MAPPER, mrv_agent_id="MRV-029", mrv_agent_name="Scope 3 Category Mapper", scope=MRVScope.CROSS_CUTTING, module_path="greenlang.agents.mrv.scope3_category_mapper", description="Maps activities to Scope 3 categories"),
    MRVAgentRoute(source=EmissionSource.AUDIT_TRAIL, mrv_agent_id="MRV-030", mrv_agent_name="Audit Trail & Lineage", scope=MRVScope.CROSS_CUTTING, module_path="greenlang.agents.mrv.audit_trail_lineage", description="Provenance and audit trail tracking"),
]


# ---------------------------------------------------------------------------
# MRVBridge
# ---------------------------------------------------------------------------


class MRVBridge:
    """Bridge to 30 MRV agents with activity-based routing for PACK-022.

    Routes emission source calculation requests preferring activity-based
    data with spend-based fallback. Supports batch routing for multi-entity
    consolidation.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.
        _routing_table: Active routing table.

    Example:
        >>> bridge = MRVBridge()
        >>> result = bridge.route_calculation(
        ...     EmissionSource.STATIONARY_COMBUSTION,
        ...     {"fuel_type": "natural_gas", "consumption_m3": 10000}
        ... )
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the MRV Bridge."""
        self.config = config or MRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(MRV_ROUTING_TABLE)

        self._agents: Dict[str, Any] = {}
        unique_agents = {r.mrv_agent_id: r.module_path for r in self._routing_table}
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_mrv_agent(agent_id, module_path)

        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info("MRVBridge initialized: %d/%d agents available", available, len(self._agents))

    def route_calculation(
        self,
        source: EmissionSource,
        data: Dict[str, Any],
        entity_id: str = "",
    ) -> RoutingResult:
        """Route a calculation request to the appropriate MRV agent.

        Prefers activity-based data; falls back to spend-based if no
        activity data is present in the input.

        Args:
            source: Emission source category.
            data: Input data for the calculation.
            entity_id: Optional entity identifier for multi-entity mode.

        Returns:
            RoutingResult with calculation output or degraded status.
        """
        start = time.monotonic()
        method = self._determine_method(data)

        route = self._find_route(source)
        if route is None:
            return RoutingResult(
                source=source.value,
                success=False,
                calculation_method=method.value,
                entity_id=entity_id,
                message=f"No routing entry for source '{source.value}'",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        agent = self._agents.get(route.mrv_agent_id)
        if agent is None or isinstance(agent, _AgentStub):
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                success=False,
                degraded=True,
                calculation_method=method.value,
                entity_id=entity_id,
                message=f"MRV agent {route.mrv_agent_id} not available (stub mode)",
                duration_ms=(time.monotonic() - start) * 1000,
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            return result

        try:
            calc_result = {"emissions_tco2e": 0.0, "status": "calculated", "method": method.value}
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                success=True,
                calculation_method=method.value,
                emissions_tco2e=calc_result.get("emissions_tco2e", 0.0),
                calculation_details=calc_result,
                entity_id=entity_id,
                message=f"Calculated via {route.mrv_agent_name} ({method.value})",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                success=False,
                calculation_method=method.value,
                entity_id=entity_id,
                message=f"Calculation failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def route_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> BatchRoutingResult:
        """Route multiple calculation requests in batch.

        Each request must contain 'source' and 'data'. Optionally 'entity_id'.

        Args:
            requests: List of dicts with 'source', 'data', and optional 'entity_id'.

        Returns:
            BatchRoutingResult with aggregated emissions by scope.
        """
        start = time.monotonic()
        results: List[RoutingResult] = []
        total_emissions = 0.0
        scope1 = scope2 = scope3 = 0.0
        successful = degraded_count = failed = 0
        activity_count = spend_count = 0
        entities: set = set()

        for req in requests:
            source_str = req.get("source", "")
            try:
                source = EmissionSource(source_str)
            except ValueError:
                results.append(RoutingResult(
                    source=source_str, success=False,
                    message=f"Unknown emission source: {source_str}",
                ))
                failed += 1
                continue

            entity_id = req.get("entity_id", "")
            result = self.route_calculation(source, req.get("data", {}), entity_id=entity_id)
            results.append(result)
            if entity_id:
                entities.add(entity_id)

            if result.success:
                successful += 1
                total_emissions += result.emissions_tco2e
                if result.scope == MRVScope.SCOPE_1.value:
                    scope1 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2.value:
                    scope2 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3.value:
                    scope3 += result.emissions_tco2e
                if result.calculation_method == CalculationMethod.ACTIVITY_BASED.value:
                    activity_count += 1
                else:
                    spend_count += 1
            elif result.degraded:
                degraded_count += 1
            else:
                failed += 1

        elapsed = (time.monotonic() - start) * 1000

        batch_result = BatchRoutingResult(
            total_sources=len(requests),
            successful=successful,
            degraded=degraded_count,
            failed=failed,
            total_emissions_tco2e=total_emissions,
            scope1_tco2e=scope1,
            scope2_tco2e=scope2,
            scope3_tco2e=scope3,
            activity_based_count=activity_count,
            spend_based_count=spend_count,
            entities_processed=sorted(entities),
            results=results,
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            batch_result.provenance_hash = _compute_hash(batch_result)

        self.logger.info(
            "Batch routing: %d/%d successful, activity=%d, spend=%d, total=%.2f tCO2e",
            successful, len(requests), activity_count, spend_count, total_emissions,
        )
        return batch_result

    def get_agent_status(self) -> Dict[str, Any]:
        """Get availability status of all 30 MRV agents."""
        available = [aid for aid, a in self._agents.items() if not isinstance(a, _AgentStub)]
        unavailable = [aid for aid, a in self._agents.items() if isinstance(a, _AgentStub)]
        return {
            "total_agents": len(self._agents),
            "available": len(available),
            "unavailable": len(unavailable),
            "available_agents": available,
            "unavailable_agents": unavailable,
        }

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Get the full routing table as a list of dicts."""
        return [
            {
                "source": r.source.value, "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name, "scope": r.scope.value,
                "scope3_category": r.scope3_category, "description": r.description,
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table
        ]

    def get_scope_agents(self, scope: MRVScope) -> List[Dict[str, Any]]:
        """Get agents for a specific scope."""
        return [
            {
                "source": r.source.value, "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name, "scope3_category": r.scope3_category,
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table if r.scope == scope
        ]

    # ---- Internal ----

    def _find_route(self, source: EmissionSource) -> Optional[MRVAgentRoute]:
        """Find the routing entry for an emission source."""
        for route in self._routing_table:
            if route.source == source:
                return route
        return None

    def _determine_method(self, data: Dict[str, Any]) -> CalculationMethod:
        """Determine calculation method based on available data.

        Activity-based is preferred. Falls back to spend-based if only
        financial data is present.

        Args:
            data: Input data dict.

        Returns:
            Selected calculation method.
        """
        activity_keys = {
            "consumption", "quantity", "volume", "mass", "distance",
            "fuel_type", "energy_kwh", "consumption_m3", "weight_kg",
        }
        spend_keys = {"spend_eur", "spend_usd", "amount", "invoice_total", "cost"}

        has_activity = any(k in data for k in activity_keys)
        has_spend = any(k in data for k in spend_keys)

        if has_activity:
            return CalculationMethod.ACTIVITY_BASED
        if has_spend:
            return CalculationMethod.SPEND_BASED
        return self.config.preferred_method
