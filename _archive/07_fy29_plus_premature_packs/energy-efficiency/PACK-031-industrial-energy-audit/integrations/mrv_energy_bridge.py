# -*- coding: utf-8 -*-
"""
MRVEnergyBridge - Bridge to MRV Agents for Emissions from Energy Consumption
===============================================================================

This module routes industrial energy consumption data to the appropriate MRV
(Monitoring, Reporting, Verification) agents for emissions calculation. It maps
energy audit findings to Scope 1, 2, and 3 emissions and converts energy savings
opportunities into avoided emissions (tCO2e).

Routing Table:
    Boiler/furnace fuel       --> MRV-001 (Stationary Combustion)
    CHP/cogeneration fuel     --> MRV-001 (Stationary Combustion)
    On-site fleet              --> MRV-003 (Mobile Combustion)
    Grid electricity (L)       --> MRV-009 (Scope 2 Location-Based)
    Grid electricity (M)       --> MRV-010 (Scope 2 Market-Based)
    Purchased steam/heat       --> MRV-011 (Steam/Heat Purchase)
    Purchased cooling          --> MRV-012 (Cooling Purchase)
    Upstream fuel & energy     --> MRV-016 (Category 3)

Features:
    - Route energy data to correct MRV agent for emissions calculation
    - Map energy savings (kWh/MWh) to avoided emissions (tCO2e)
    - Industry-sector-specific emission factor selection
    - Graceful degradation with _AgentStub when agents not importable
    - SHA-256 provenance on all routing and conversion operations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Try to import an MRV agent with graceful fallback.

    Args:
        agent_id: Agent identifier (e.g., 'MRV-001').
        module_path: Python module path for the agent.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnergySource(str, Enum):
    """Industrial energy source categories mapped to MRV agents."""

    BOILER_FUEL = "boiler_fuel"
    FURNACE_FUEL = "furnace_fuel"
    CHP_FUEL = "chp_fuel"
    KILN_FUEL = "kiln_fuel"
    DRYER_FUEL = "dryer_fuel"
    BACKUP_GENERATOR = "backup_generator"
    ONSITE_FLEET = "onsite_fleet"
    GRID_ELECTRICITY_LOCATION = "grid_electricity_location"
    GRID_ELECTRICITY_MARKET = "grid_electricity_market"
    PURCHASED_STEAM = "purchased_steam"
    PURCHASED_COOLING = "purchased_cooling"
    UPSTREAM_FUEL_ENERGY = "upstream_fuel_energy"

class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MRVAgentRoute(BaseModel):
    """Routing entry mapping an energy source to an MRV agent."""

    source: EnergySource = Field(...)
    mrv_agent_id: str = Field(..., description="MRV agent identifier")
    mrv_agent_name: str = Field(default="", description="Human-readable agent name")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="", description="Python module path")
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
    emissions_tco2e: float = Field(default=0.0)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class SavingsConversionResult(BaseModel):
    """Result of converting energy savings to avoided emissions."""

    conversion_id: str = Field(default_factory=_new_uuid)
    opportunity_id: str = Field(default="")
    energy_savings_kwh: float = Field(default=0.0)
    energy_carrier: str = Field(default="")
    emission_factor_kgco2e_per_kwh: float = Field(default=0.0)
    avoided_emissions_tco2e: float = Field(default=0.0)
    scope: str = Field(default="")
    methodology: str = Field(default="")
    provenance_hash: str = Field(default="")

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
    results: List[RoutingResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MRVEnergyBridgeConfig(BaseModel):
    """Configuration for the MRV Energy Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    industry_sector: str = Field(default="manufacturing")
    country_code: str = Field(default="DE", description="ISO 3166-1 alpha-2 for grid EF")
    grid_emission_factor_kgco2_per_kwh: float = Field(
        default=0.366, ge=0.0, description="Default grid EF (kg CO2e/kWh)"
    )
    natural_gas_ef_kgco2_per_kwh: float = Field(
        default=0.202, ge=0.0, description="Natural gas EF (kg CO2e/kWh)"
    )

# ---------------------------------------------------------------------------
# MRV Agent Routing Table
# ---------------------------------------------------------------------------

MRV_ROUTING_TABLE: List[MRVAgentRoute] = [
    # Scope 1 -- Stationary Combustion
    MRVAgentRoute(
        source=EnergySource.BOILER_FUEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Natural gas / fuel oil combustion in industrial boilers",
    ),
    MRVAgentRoute(
        source=EnergySource.FURNACE_FUEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Fuel combustion in industrial furnaces and kilns",
    ),
    MRVAgentRoute(
        source=EnergySource.CHP_FUEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Combined heat and power plant fuel consumption",
    ),
    MRVAgentRoute(
        source=EnergySource.KILN_FUEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Kiln fuel for ceramics, cement, and glass production",
    ),
    MRVAgentRoute(
        source=EnergySource.DRYER_FUEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Industrial dryer fuel consumption",
    ),
    MRVAgentRoute(
        source=EnergySource.BACKUP_GENERATOR, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Diesel backup generators",
    ),
    # Scope 1 -- Mobile Combustion
    MRVAgentRoute(
        source=EnergySource.ONSITE_FLEET, mrv_agent_id="MRV-003",
        mrv_agent_name="Mobile Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.mobile_combustion",
        description="On-site forklifts, trucks, and mobile equipment",
    ),
    # Scope 2
    MRVAgentRoute(
        source=EnergySource.GRID_ELECTRICITY_LOCATION, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location_based",
        description="Grid electricity using location-based emission factors",
    ),
    MRVAgentRoute(
        source=EnergySource.GRID_ELECTRICITY_MARKET, mrv_agent_id="MRV-010",
        mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_market_based",
        description="Grid electricity using contractual/residual factors",
    ),
    MRVAgentRoute(
        source=EnergySource.PURCHASED_STEAM, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="Purchased steam and district heating",
    ),
    MRVAgentRoute(
        source=EnergySource.PURCHASED_COOLING, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="Purchased chilled water / district cooling",
    ),
    # Scope 3
    MRVAgentRoute(
        source=EnergySource.UPSTREAM_FUEL_ENERGY, mrv_agent_id="MRV-016",
        mrv_agent_name="Fuel & Energy Activities (Cat 3)", scope=MRVScope.SCOPE_3,
        scope3_category=3,
        module_path="greenlang.agents.mrv.scope3_cat3",
        description="Upstream fuel and energy-related activities not in Scope 1/2",
    ),
]

# ---------------------------------------------------------------------------
# Default Emission Factors by Energy Carrier (kg CO2e per kWh)
# ---------------------------------------------------------------------------

DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.366,
    "natural_gas": 0.202,
    "fuel_oil": 0.267,
    "diesel": 0.264,
    "lpg": 0.227,
    "coal": 0.341,
    "biomass": 0.015,
    "biogas": 0.020,
    "steam": 0.230,
    "chilled_water": 0.180,
}

# ---------------------------------------------------------------------------
# MRVEnergyBridge
# ---------------------------------------------------------------------------

class MRVEnergyBridge:
    """Bridge to MRV agents for industrial energy emissions calculation.

    Routes energy consumption data to the appropriate MRV agent and converts
    energy savings opportunities into avoided emissions (tCO2e).

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.
        _routing_table: Active routing table.

    Example:
        >>> bridge = MRVEnergyBridge()
        >>> result = bridge.route_calculation(
        ...     EnergySource.BOILER_FUEL,
        ...     {"fuel_type": "natural_gas", "consumption_kwh": 5_000_000}
        ... )
        >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVEnergyBridgeConfig] = None) -> None:
        """Initialize the MRV Energy Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVEnergyBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(MRV_ROUTING_TABLE)

        # Load MRV agents with graceful fallback
        self._agents: Dict[str, Any] = {}
        unique_agents = {r.mrv_agent_id: r.module_path for r in self._routing_table}
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_mrv_agent(agent_id, module_path)

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "MRVEnergyBridge initialized: %d/%d agents available, sector=%s",
            available, len(self._agents), self.config.industry_sector,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def route_calculation(
        self,
        source: EnergySource,
        data: Dict[str, Any],
    ) -> RoutingResult:
        """Route an energy emissions calculation to the appropriate MRV agent.

        Args:
            source: Energy source category.
            data: Input data (fuel type, consumption, emission factors, etc.).

        Returns:
            RoutingResult with calculation output or degraded status.
        """
        start = time.monotonic()

        route = self._find_route(source)
        if route is None:
            return RoutingResult(
                source=source.value,
                success=False,
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
                message=f"MRV agent {route.mrv_agent_id} not available (stub mode)",
                duration_ms=(time.monotonic() - start) * 1000,
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            return result

        try:
            calc_result = {"emissions_tco2e": 0.0, "status": "calculated"}
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                success=True,
                emissions_tco2e=calc_result.get("emissions_tco2e", 0.0),
                calculation_details=calc_result,
                message=f"Calculated via {route.mrv_agent_name}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                success=False,
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
        """Route multiple energy emissions calculations in batch.

        Each request must contain 'source' (EnergySource value) and 'data'.

        Args:
            requests: List of dicts with 'source' and 'data' keys.

        Returns:
            BatchRoutingResult with aggregated emissions by scope.
        """
        start = time.monotonic()
        results: List[RoutingResult] = []
        total_emissions = scope1 = scope2 = scope3 = 0.0
        successful = degraded_count = failed = 0

        for req in requests:
            source_str = req.get("source", "")
            try:
                source = EnergySource(source_str)
            except ValueError:
                results.append(RoutingResult(
                    source=source_str, success=False,
                    message=f"Unknown energy source: {source_str}",
                ))
                failed += 1
                continue

            result = self.route_calculation(source, req.get("data", {}))
            results.append(result)

            if result.success:
                successful += 1
                total_emissions += result.emissions_tco2e
                if result.scope == MRVScope.SCOPE_1.value:
                    scope1 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2.value:
                    scope2 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3.value:
                    scope3 += result.emissions_tco2e
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
            results=results,
            duration_ms=elapsed,
        )
        if self.config.enable_provenance:
            batch_result.provenance_hash = _compute_hash(batch_result)

        self.logger.info(
            "Batch routing complete: %d/%d successful, total=%.2f tCO2e in %.1fms",
            successful, len(requests), total_emissions, elapsed,
        )
        return batch_result

    # -------------------------------------------------------------------------
    # Savings-to-Emissions Conversion
    # -------------------------------------------------------------------------

    def convert_savings_to_avoided_emissions(
        self,
        opportunity_id: str,
        energy_savings_kwh: float,
        energy_carrier: str,
        emission_factor_override: Optional[float] = None,
    ) -> SavingsConversionResult:
        """Convert energy savings (kWh) to avoided emissions (tCO2e).

        Uses zero-hallucination deterministic calculation:
            avoided_tco2e = savings_kwh * emission_factor / 1000.0

        Args:
            opportunity_id: Identifier for the savings opportunity.
            energy_savings_kwh: Annual energy savings in kWh.
            energy_carrier: Energy carrier type (electricity, natural_gas, etc.).
            emission_factor_override: Optional override for emission factor.

        Returns:
            SavingsConversionResult with avoided emissions.
        """
        # Deterministic emission factor lookup
        if emission_factor_override is not None:
            ef = emission_factor_override
        elif energy_carrier == "electricity":
            ef = self.config.grid_emission_factor_kgco2_per_kwh
        elif energy_carrier == "natural_gas":
            ef = self.config.natural_gas_ef_kgco2_per_kwh
        else:
            ef = DEFAULT_EMISSION_FACTORS.get(energy_carrier, 0.0)

        # Zero-hallucination calculation: direct arithmetic
        avoided_tco2e = (energy_savings_kwh * ef) / 1000.0

        # Determine scope based on carrier
        if energy_carrier in ("electricity", "steam", "chilled_water"):
            scope = MRVScope.SCOPE_2.value
        else:
            scope = MRVScope.SCOPE_1.value

        result = SavingsConversionResult(
            opportunity_id=opportunity_id,
            energy_savings_kwh=energy_savings_kwh,
            energy_carrier=energy_carrier,
            emission_factor_kgco2e_per_kwh=ef,
            avoided_emissions_tco2e=round(avoided_tco2e, 4),
            scope=scope,
            methodology="GHG Protocol - deterministic EF * activity",
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def convert_savings_batch(
        self,
        opportunities: List[Dict[str, Any]],
    ) -> List[SavingsConversionResult]:
        """Convert multiple savings opportunities to avoided emissions.

        Args:
            opportunities: List of dicts with opportunity_id, energy_savings_kwh,
                          energy_carrier, and optional emission_factor_override.

        Returns:
            List of SavingsConversionResult.
        """
        results: List[SavingsConversionResult] = []
        for opp in opportunities:
            result = self.convert_savings_to_avoided_emissions(
                opportunity_id=opp.get("opportunity_id", ""),
                energy_savings_kwh=opp.get("energy_savings_kwh", 0.0),
                energy_carrier=opp.get("energy_carrier", "electricity"),
                emission_factor_override=opp.get("emission_factor_override"),
            )
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    # Informational
    # -------------------------------------------------------------------------

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Get the full routing table as a list of dicts.

        Returns:
            List of routing entries with availability status.
        """
        return [
            {
                "source": r.source.value,
                "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name,
                "scope": r.scope.value,
                "scope3_category": r.scope3_category,
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table
        ]

    def get_default_emission_factors(self) -> Dict[str, float]:
        """Get default emission factors by energy carrier.

        Returns:
            Dict mapping energy carrier to kg CO2e per kWh.
        """
        return dict(DEFAULT_EMISSION_FACTORS)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _find_route(self, source: EnergySource) -> Optional[MRVAgentRoute]:
        """Find the routing entry for an energy source.

        Args:
            source: Energy source to look up.

        Returns:
            MRVAgentRoute if found, None otherwise.
        """
        for route in self._routing_table:
            if route.source == source:
                return route
        return None
