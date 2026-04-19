# -*- coding: utf-8 -*-
"""
MRVBuildingBridge - Bridge to MRV Agents for Building Emissions Calculation
=============================================================================

This module routes building energy consumption data to the appropriate MRV
(Monitoring, Reporting, Verification) agents for emissions calculation. It maps
building energy data to Scope 1, 2, and 3 emissions and converts energy savings
from retrofit measures into avoided emissions (tCO2e).

Routing Table (8 Building MRV Agents):
    Commercial building energy   --> GL-MRV-BLD-001 (Commercial Buildings)
    Residential building energy  --> GL-MRV-BLD-002 (Residential Buildings)
    Industrial/warehouse energy  --> GL-MRV-BLD-003 (Industrial Buildings)
    HVAC system emissions        --> GL-MRV-BLD-004 (HVAC Systems)
    Lighting system emissions    --> GL-MRV-BLD-005 (Lighting Systems)
    Embodied carbon / materials  --> GL-MRV-BLD-006 (Building Materials)
    Operational energy           --> GL-MRV-BLD-007 (Building Operations)
    Smart building / BMS         --> GL-MRV-BLD-008 (Smart Buildings)

Fallback MRV Routes:
    On-site gas boilers          --> MRV-001 (Stationary Combustion)
    Grid electricity (location)  --> MRV-009 (Scope 2 Location-Based)
    Grid electricity (market)    --> MRV-010 (Scope 2 Market-Based)
    Purchased heat/steam         --> MRV-011 (Steam/Heat Purchase)
    Purchased cooling            --> MRV-012 (Cooling Purchase)
    Refrigerant leakage          --> MRV-002 (Refrigerants & F-Gas)

Features:
    - Route building energy data to correct MRV agent for emissions calculation
    - Map energy savings (kWh) to avoided emissions (tCO2e)
    - Building-type-specific emission factor selection
    - Graceful degradation with _AgentStub when agents not importable
    - SHA-256 provenance on all routing and conversion operations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-032 Building Energy Assessment
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
        agent_id: Agent identifier (e.g., 'MRV-BLD-001').
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

class BuildingEnergySource(str, Enum):
    """Building energy source categories mapped to MRV agents."""

    GAS_HEATING = "gas_heating"
    OIL_HEATING = "oil_heating"
    BIOMASS_HEATING = "biomass_heating"
    ELECTRIC_HEATING = "electric_heating"
    HEAT_PUMP = "heat_pump"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    GRID_ELECTRICITY_LOCATION = "grid_electricity_location"
    GRID_ELECTRICITY_MARKET = "grid_electricity_market"
    ON_SITE_PV = "on_site_pv"
    SOLAR_THERMAL = "solar_thermal"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    EMBODIED_CARBON = "embodied_carbon"
    WATER_HEATING_GAS = "water_heating_gas"
    WATER_HEATING_ELECTRIC = "water_heating_electric"
    COOKING_GAS = "cooking_gas"

class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"

class BuildingMRVAgentType(str, Enum):
    """Building-specific MRV agent types."""

    COMMERCIAL = "GL-MRV-BLD-001"
    RESIDENTIAL = "GL-MRV-BLD-002"
    INDUSTRIAL = "GL-MRV-BLD-003"
    HVAC = "GL-MRV-BLD-004"
    LIGHTING = "GL-MRV-BLD-005"
    MATERIALS = "GL-MRV-BLD-006"
    OPERATIONS = "GL-MRV-BLD-007"
    SMART_BUILDING = "GL-MRV-BLD-008"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MRVAgentRoute(BaseModel):
    """Routing entry mapping an energy source to an MRV agent."""

    source: BuildingEnergySource = Field(...)
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
    measure_id: str = Field(default="")
    measure_name: str = Field(default="")
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
    building_id: str = Field(default="")
    total_sources: int = Field(default=0)
    successful: int = Field(default=0)
    degraded: int = Field(default=0)
    failed: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    results: List[RoutingResult] = Field(default_factory=list)
    savings_conversions: List[SavingsConversionResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MRVBuildingBridgeConfig(BaseModel):
    """Configuration for the MRV Building Bridge."""

    pack_id: str = Field(default="PACK-032")
    enable_provenance: bool = Field(default=True)
    building_type: str = Field(default="commercial_office")
    country_code: str = Field(default="GB", description="ISO 3166-1 alpha-2 for grid EF")
    grid_emission_factor_kgco2_per_kwh: float = Field(
        default=0.233, ge=0.0, description="Default grid EF (kg CO2e/kWh)"
    )
    natural_gas_ef_kgco2_per_kwh: float = Field(
        default=0.202, ge=0.0, description="Natural gas EF (kg CO2e/kWh)"
    )
    oil_ef_kgco2_per_kwh: float = Field(
        default=0.274, ge=0.0, description="Heating oil EF (kg CO2e/kWh)"
    )
    biomass_ef_kgco2_per_kwh: float = Field(
        default=0.015, ge=0.0, description="Biomass EF (kg CO2e/kWh)"
    )
    district_heating_ef_kgco2_per_kwh: float = Field(
        default=0.170, ge=0.0, description="District heating EF (kg CO2e/kWh)"
    )

# ---------------------------------------------------------------------------
# MRV Agent Routing Table
# ---------------------------------------------------------------------------

MRV_ROUTING_TABLE: List[MRVAgentRoute] = [
    # Scope 1 -- On-site combustion
    MRVAgentRoute(
        source=BuildingEnergySource.GAS_HEATING, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Natural gas combustion in building boilers",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.OIL_HEATING, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Heating oil combustion in building boilers",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.BIOMASS_HEATING, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Biomass combustion in building heating systems",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.WATER_HEATING_GAS, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Gas-fired domestic hot water systems",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.COOKING_GAS, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Gas cooking in commercial kitchens",
    ),
    # Scope 1 -- Refrigerant leakage
    MRVAgentRoute(
        source=BuildingEnergySource.REFRIGERANT_LEAKAGE, mrv_agent_id="MRV-002",
        mrv_agent_name="Refrigerants & F-Gas", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.refrigerants_fgas",
        description="HVAC refrigerant leakage from chillers and split systems",
    ),
    # Scope 2 -- Purchased electricity (location-based)
    MRVAgentRoute(
        source=BuildingEnergySource.GRID_ELECTRICITY_LOCATION, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location",
        description="Grid electricity (location-based emission factor)",
    ),
    # Scope 2 -- Purchased electricity (market-based)
    MRVAgentRoute(
        source=BuildingEnergySource.GRID_ELECTRICITY_MARKET, mrv_agent_id="MRV-010",
        mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_market",
        description="Grid electricity (market-based emission factor with RECs/GOs)",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.ELECTRIC_HEATING, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location",
        description="Electric heating systems (resistive/heat pump)",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.HEAT_PUMP, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location",
        description="Air/ground source heat pump electricity consumption",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.WATER_HEATING_ELECTRIC, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location",
        description="Electric water heating (immersion, heat pump DHW)",
    ),
    # Scope 2 -- Purchased heat/cooling
    MRVAgentRoute(
        source=BuildingEnergySource.DISTRICT_HEATING, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="Purchased district heating for building",
    ),
    MRVAgentRoute(
        source=BuildingEnergySource.DISTRICT_COOLING, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="Purchased district cooling for building",
    ),
    # Scope 3 -- Embodied carbon
    MRVAgentRoute(
        source=BuildingEnergySource.EMBODIED_CARBON, mrv_agent_id="GL-MRV-BLD-006",
        mrv_agent_name="Building Materials", scope=MRVScope.SCOPE_3,
        scope3_category=1,
        module_path="greenlang.agents.mrv.building_materials",
        description="Embodied carbon in building materials (A1-A5, B1-B5, C1-C4)",
    ),
]

# Building-type to primary MRV agent mapping
BUILDING_TYPE_AGENT_MAP: Dict[str, str] = {
    "commercial_office": "GL-MRV-BLD-001",
    "retail_building": "GL-MRV-BLD-001",
    "hotel_hospitality": "GL-MRV-BLD-001",
    "healthcare_facility": "GL-MRV-BLD-001",
    "education_building": "GL-MRV-BLD-001",
    "residential_multifamily": "GL-MRV-BLD-002",
    "mixed_use_development": "GL-MRV-BLD-001",
    "public_sector_building": "GL-MRV-BLD-001",
    "industrial_warehouse": "GL-MRV-BLD-003",
    "data_centre": "GL-MRV-BLD-001",
}

# Country-level grid emission factors (kg CO2e/kWh) -- 2024 reference
GRID_EMISSION_FACTORS: Dict[str, float] = {
    "GB": 0.233, "DE": 0.366, "FR": 0.052, "NL": 0.328,
    "IT": 0.257, "ES": 0.146, "SE": 0.008, "NO": 0.007,
    "PL": 0.623, "BE": 0.143, "AT": 0.086, "DK": 0.116,
    "FI": 0.061, "IE": 0.296, "PT": 0.142, "CZ": 0.383,
    "RO": 0.261, "HU": 0.209, "GR": 0.349, "BG": 0.398,
    "US": 0.390, "CA": 0.120, "AU": 0.656, "JP": 0.457,
    "KR": 0.415, "CN": 0.555, "IN": 0.708, "BR": 0.074,
    "ZA": 0.928, "SG": 0.408,
}

# ---------------------------------------------------------------------------
# MRVBuildingBridge
# ---------------------------------------------------------------------------

class MRVBuildingBridge:
    """Routes building energy data to MRV agents for emissions calculation.

    Supports 8 building-specific MRV agents plus fallback to standard MRV agents
    for stationary combustion, Scope 2, and purchased energy. Converts energy
    savings from retrofit measures into avoided emissions (tCO2e).

    Attributes:
        config: Bridge configuration.
        _agents: Loaded MRV agent instances (or stubs).

    Example:
        >>> bridge = MRVBuildingBridge()
        >>> result = bridge.route_calculation("gas_heating", {"energy_kwh": 50000})
        >>> assert result.success
    """

    def __init__(self, config: Optional[MRVBuildingBridgeConfig] = None) -> None:
        """Initialize the MRV Building Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVBuildingBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        self._load_agents()

        self.logger.info(
            "MRVBuildingBridge initialized: building_type=%s, country=%s, "
            "grid_ef=%.3f",
            self.config.building_type,
            self.config.country_code,
            self.config.grid_emission_factor_kgco2_per_kwh,
        )

    def _load_agents(self) -> None:
        """Load MRV agents with graceful fallback to stubs."""
        seen: set = set()
        for route in MRV_ROUTING_TABLE:
            if route.mrv_agent_id not in seen:
                self._agents[route.mrv_agent_id] = _try_import_mrv_agent(
                    route.mrv_agent_id, route.module_path
                )
                seen.add(route.mrv_agent_id)

        # Load building-specific MRV agents
        for agent_type in BuildingMRVAgentType:
            if agent_type.value not in self._agents:
                self._agents[agent_type.value] = _try_import_mrv_agent(
                    agent_type.value,
                    f"greenlang.agents.mrv.building.{agent_type.name.lower()}",
                )

    def _get_route(self, source_name: str) -> Optional[MRVAgentRoute]:
        """Find the routing entry for an energy source.

        Args:
            source_name: Energy source name.

        Returns:
            MRVAgentRoute or None.
        """
        for route in MRV_ROUTING_TABLE:
            if route.source.value == source_name:
                return route
        return None

    def _get_emission_factor(self, source: BuildingEnergySource) -> float:
        """Get emission factor for an energy source.

        Args:
            source: Energy source type.

        Returns:
            Emission factor in kg CO2e/kWh.
        """
        country = self.config.country_code
        ef_map: Dict[str, float] = {
            BuildingEnergySource.GAS_HEATING.value: self.config.natural_gas_ef_kgco2_per_kwh,
            BuildingEnergySource.OIL_HEATING.value: self.config.oil_ef_kgco2_per_kwh,
            BuildingEnergySource.BIOMASS_HEATING.value: self.config.biomass_ef_kgco2_per_kwh,
            BuildingEnergySource.WATER_HEATING_GAS.value: self.config.natural_gas_ef_kgco2_per_kwh,
            BuildingEnergySource.COOKING_GAS.value: self.config.natural_gas_ef_kgco2_per_kwh,
            BuildingEnergySource.ELECTRIC_HEATING.value: GRID_EMISSION_FACTORS.get(
                country, self.config.grid_emission_factor_kgco2_per_kwh
            ),
            BuildingEnergySource.HEAT_PUMP.value: GRID_EMISSION_FACTORS.get(
                country, self.config.grid_emission_factor_kgco2_per_kwh
            ),
            BuildingEnergySource.WATER_HEATING_ELECTRIC.value: GRID_EMISSION_FACTORS.get(
                country, self.config.grid_emission_factor_kgco2_per_kwh
            ),
            BuildingEnergySource.GRID_ELECTRICITY_LOCATION.value: GRID_EMISSION_FACTORS.get(
                country, self.config.grid_emission_factor_kgco2_per_kwh
            ),
            BuildingEnergySource.GRID_ELECTRICITY_MARKET.value: GRID_EMISSION_FACTORS.get(
                country, self.config.grid_emission_factor_kgco2_per_kwh
            ),
            BuildingEnergySource.DISTRICT_HEATING.value: self.config.district_heating_ef_kgco2_per_kwh,
            BuildingEnergySource.DISTRICT_COOLING.value: 0.150,
        }
        return ef_map.get(source.value, 0.0)

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def route_calculation(
        self,
        source_name: str,
        data: Dict[str, Any],
    ) -> RoutingResult:
        """Route a single energy calculation to the appropriate MRV agent.

        Args:
            source_name: Energy source identifier.
            data: Calculation data (must include 'energy_kwh').

        Returns:
            RoutingResult with emissions and provenance.
        """
        start_time = time.monotonic()
        result = RoutingResult(source=source_name)

        route = self._get_route(source_name)
        if route is None:
            result.message = f"No MRV route for source '{source_name}'"
            result.duration_ms = (time.monotonic() - start_time) * 1000
            return result

        result.mrv_agent_id = route.mrv_agent_id
        result.scope = route.scope.value
        result.scope3_category = route.scope3_category

        agent = self._agents.get(route.mrv_agent_id)
        if agent is None:
            result.message = f"Agent {route.mrv_agent_id} not loaded"
            result.duration_ms = (time.monotonic() - start_time) * 1000
            return result

        try:
            # Check if agent is a stub
            if isinstance(agent, _AgentStub):
                result.degraded = True
                # Use deterministic fallback calculation
                energy_kwh = data.get("energy_kwh", 0.0)
                try:
                    source_enum = BuildingEnergySource(source_name)
                    ef = self._get_emission_factor(source_enum)
                except ValueError:
                    ef = self.config.grid_emission_factor_kgco2_per_kwh
                emissions_tco2e = energy_kwh * ef / 1000.0
                result.emissions_tco2e = round(emissions_tco2e, 6)
                result.success = True
                result.message = f"Degraded: stub calculation for {route.mrv_agent_id}"
                result.calculation_details = {
                    "method": "stub_ef_multiplication",
                    "energy_kwh": energy_kwh,
                    "emission_factor_kgco2_per_kwh": ef,
                    "emissions_kgco2e": round(energy_kwh * ef, 3),
                }
            else:
                # Real agent call
                agent_result = agent.calculate(data)
                result.emissions_tco2e = agent_result.get("emissions_tco2e", 0.0)
                result.success = True
                result.calculation_details = agent_result
                result.message = f"Calculated via {route.mrv_agent_id}"

        except Exception as exc:
            result.message = f"Agent {route.mrv_agent_id} failed: {exc}"
            self.logger.error(
                "MRV routing failed: source=%s, agent=%s, error=%s",
                source_name, route.mrv_agent_id, exc,
            )

        result.duration_ms = (time.monotonic() - start_time) * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash({
                "source": source_name,
                "agent": route.mrv_agent_id,
                "emissions": result.emissions_tco2e,
                "data": data,
            })

        return result

    def route_batch(
        self,
        sources: List[Dict[str, Any]],
        building_id: str = "",
    ) -> BatchRoutingResult:
        """Route multiple energy source calculations in batch.

        Args:
            sources: List of dicts with 'source' and 'data' keys.
            building_id: Building identifier for tracking.

        Returns:
            BatchRoutingResult with aggregated emissions.
        """
        start_time = time.monotonic()
        batch = BatchRoutingResult(building_id=building_id, total_sources=len(sources))

        for src in sources:
            source_name = src.get("source", "")
            data = src.get("data", {})
            result = self.route_calculation(source_name, data)
            batch.results.append(result)

            if result.success:
                batch.successful += 1
                batch.total_emissions_tco2e += result.emissions_tco2e

                if result.scope == MRVScope.SCOPE_1.value:
                    batch.scope1_tco2e += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2.value:
                    batch.scope2_tco2e += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3.value:
                    batch.scope3_tco2e += result.emissions_tco2e

                if result.degraded:
                    batch.degraded += 1
            else:
                batch.failed += 1

        # Round aggregated values
        batch.total_emissions_tco2e = round(batch.total_emissions_tco2e, 6)
        batch.scope1_tco2e = round(batch.scope1_tco2e, 6)
        batch.scope2_tco2e = round(batch.scope2_tco2e, 6)
        batch.scope3_tco2e = round(batch.scope3_tco2e, 6)

        batch.duration_ms = (time.monotonic() - start_time) * 1000

        if self.config.enable_provenance:
            batch.provenance_hash = _compute_hash(batch)

        self.logger.info(
            "Batch routing complete: building=%s, total=%d, ok=%d, "
            "degraded=%d, failed=%d, emissions=%.3f tCO2e",
            building_id, len(sources), batch.successful,
            batch.degraded, batch.failed, batch.total_emissions_tco2e,
        )
        return batch

    # -------------------------------------------------------------------------
    # Energy-to-Emissions Conversion
    # -------------------------------------------------------------------------

    def convert_energy_to_emissions(
        self,
        energy_kwh: float,
        energy_source: str,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert energy consumption to CO2e emissions.

        Zero-hallucination: uses deterministic emission factor multiplication.

        Args:
            energy_kwh: Energy consumption in kWh.
            energy_source: Energy source type.
            country_code: Optional country override for grid EF.

        Returns:
            Dict with emissions breakdown.
        """
        try:
            source_enum = BuildingEnergySource(energy_source)
        except ValueError:
            return {
                "error": f"Unknown energy source '{energy_source}'",
                "emissions_kgco2e": 0.0,
                "emissions_tco2e": 0.0,
            }

        country = country_code or self.config.country_code
        ef = self._get_emission_factor(source_enum)

        # Override grid EF for specific country
        if source_enum in (
            BuildingEnergySource.GRID_ELECTRICITY_LOCATION,
            BuildingEnergySource.GRID_ELECTRICITY_MARKET,
            BuildingEnergySource.ELECTRIC_HEATING,
            BuildingEnergySource.HEAT_PUMP,
            BuildingEnergySource.WATER_HEATING_ELECTRIC,
        ):
            ef = GRID_EMISSION_FACTORS.get(country, ef)

        emissions_kgco2e = energy_kwh * ef
        emissions_tco2e = emissions_kgco2e / 1000.0

        return {
            "energy_kwh": energy_kwh,
            "energy_source": energy_source,
            "country_code": country,
            "emission_factor_kgco2e_per_kwh": ef,
            "emissions_kgco2e": round(emissions_kgco2e, 3),
            "emissions_tco2e": round(emissions_tco2e, 6),
            "methodology": "ef_multiplication",
            "provenance_hash": _compute_hash({
                "energy_kwh": energy_kwh,
                "source": energy_source,
                "ef": ef,
            }),
        }

    def convert_savings_to_avoided_emissions(
        self,
        measures: List[Dict[str, Any]],
    ) -> List[SavingsConversionResult]:
        """Convert retrofit energy savings to avoided emissions.

        Args:
            measures: List of retrofit measures with 'energy_savings_kwh',
                      'energy_carrier', and optional 'measure_id'/'measure_name'.

        Returns:
            List of SavingsConversionResult.
        """
        results: List[SavingsConversionResult] = []
        for measure in measures:
            savings_kwh = measure.get("energy_savings_kwh", measure.get("annual_saving_kwh", 0.0))
            carrier = measure.get("energy_carrier", measure.get("category", "electricity"))
            measure_id = measure.get("measure_id", _new_uuid()[:8])
            measure_name = measure.get("measure_name", measure.get("measure", ""))

            # Map carrier to source for EF lookup
            carrier_source_map: Dict[str, str] = {
                "electricity": "grid_electricity_location",
                "gas": "gas_heating",
                "natural_gas": "gas_heating",
                "oil": "oil_heating",
                "heating_oil": "oil_heating",
                "biomass": "biomass_heating",
                "district_heating": "district_heating",
                "district_cooling": "district_cooling",
                "hvac": "grid_electricity_location",
                "lighting": "grid_electricity_location",
                "envelope": "gas_heating",
                "renewable": "grid_electricity_location",
            }
            source_name = carrier_source_map.get(carrier, "grid_electricity_location")

            try:
                source_enum = BuildingEnergySource(source_name)
                ef = self._get_emission_factor(source_enum)
            except ValueError:
                ef = self.config.grid_emission_factor_kgco2_per_kwh

            avoided_tco2e = savings_kwh * ef / 1000.0

            # Determine scope
            scope = "scope_2"
            if source_name in ("gas_heating", "oil_heating", "biomass_heating"):
                scope = "scope_1"

            conv = SavingsConversionResult(
                measure_id=measure_id,
                measure_name=measure_name,
                energy_savings_kwh=savings_kwh,
                energy_carrier=carrier,
                emission_factor_kgco2e_per_kwh=ef,
                avoided_emissions_tco2e=round(avoided_tco2e, 6),
                scope=scope,
                methodology="ef_multiplication_savings",
            )

            if self.config.enable_provenance:
                conv.provenance_hash = _compute_hash(conv)

            results.append(conv)

        self.logger.info(
            "Converted %d measures to avoided emissions: total=%.3f tCO2e",
            len(results),
            sum(r.avoided_emissions_tco2e for r in results),
        )
        return results

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Return the complete MRV routing table for inspection.

        Returns:
            List of routing entries as dicts.
        """
        return [
            {
                "source": r.source.value,
                "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name,
                "scope": r.scope.value,
                "scope3_category": r.scope3_category,
                "description": r.description,
            }
            for r in MRV_ROUTING_TABLE
        ]

    def get_available_agents(self) -> Dict[str, bool]:
        """Check which MRV agents are available (not stubs).

        Returns:
            Dict mapping agent_id to availability boolean.
        """
        return {
            agent_id: not isinstance(agent, _AgentStub)
            for agent_id, agent in self._agents.items()
        }
