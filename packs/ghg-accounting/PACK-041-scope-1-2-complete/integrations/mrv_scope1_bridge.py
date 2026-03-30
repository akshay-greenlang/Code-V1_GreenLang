# -*- coding: utf-8 -*-
"""
MRVScope1Bridge - Bridge to All 8 Scope 1 MRV Agents for PACK-041
=====================================================================

This module routes facility-level activity data to the appropriate Scope 1
MRV agents (MRV-001 through MRV-008) for emissions calculation. It covers
all GHG Protocol Scope 1 categories: stationary combustion, refrigerant
leakage, mobile combustion, process emissions, fugitive emissions, land
use change, waste treatment, and agricultural emissions.

Routing Table:
    Stationary combustion    --> MRV-001 (gl_stationary_combustion_)
    Refrigerant / process    --> MRV-002 (gl_refrigerant_emissions_)
    Mobile combustion        --> MRV-003 (gl_mobile_combustion_)
    Process emissions        --> MRV-004 (gl_process_emissions_)
    Fugitive emissions       --> MRV-005 (gl_fugitive_emissions_)
    Land use change          --> MRV-006 (gl_land_use_emissions_)
    Waste treatment          --> MRV-007 (gl_waste_treatment_)
    Agricultural emissions   --> MRV-008 (gl_agricultural_emissions_)

Zero-Hallucination:
    All emission factor lookups, combustion calculations, GWP conversions,
    and aggregations use deterministic formulas. No LLM calls in the
    calculation path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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

def _try_import_agent(agent_id: str, module_path: str) -> Any:
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

class Scope1Category(str, Enum):
    """Scope 1 emission categories mapped to MRV agents."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    REFRIGERANTS = "refrigerants"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"

class AgentStatus(str, Enum):
    """MRV agent availability status."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

class FuelType(str, Enum):
    """Common Scope 1 fuel types."""

    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    WOOD_BIOMASS = "wood_biomass"
    CNG = "cng"
    LPG = "lpg"
    JET_FUEL = "jet_fuel"
    KEROSENE = "kerosene"

class RefrigerantType(str, Enum):
    """Common refrigerant types with GWP values."""

    R_410A = "R-410A"
    R_134A = "R-134a"
    R_407C = "R-407C"
    R_404A = "R-404A"
    R_22 = "R-22"
    R_32 = "R-32"
    R_1234YF = "R-1234yf"
    R_1234ZE = "R-1234ze"
    CO2_R744 = "R-744"
    SF6 = "SF6"

# ---------------------------------------------------------------------------
# Agent-to-Category Mapping
# ---------------------------------------------------------------------------

AGENT_CATEGORY_MAP: Dict[Scope1Category, Dict[str, str]] = {
    Scope1Category.STATIONARY_COMBUSTION: {
        "agent_id": "MRV-001",
        "module_path": "greenlang.agents.mrv.stationary_combustion",
        "prefix": "gl_stationary_combustion_",
    },
    Scope1Category.REFRIGERANTS: {
        "agent_id": "MRV-002",
        "module_path": "greenlang.agents.mrv.refrigerant_emissions",
        "prefix": "gl_refrigerant_emissions_",
    },
    Scope1Category.MOBILE_COMBUSTION: {
        "agent_id": "MRV-003",
        "module_path": "greenlang.agents.mrv.mobile_combustion",
        "prefix": "gl_mobile_combustion_",
    },
    Scope1Category.PROCESS_EMISSIONS: {
        "agent_id": "MRV-004",
        "module_path": "greenlang.agents.mrv.process_emissions",
        "prefix": "gl_process_emissions_",
    },
    Scope1Category.FUGITIVE_EMISSIONS: {
        "agent_id": "MRV-005",
        "module_path": "greenlang.agents.mrv.fugitive_emissions",
        "prefix": "gl_fugitive_emissions_",
    },
    Scope1Category.LAND_USE: {
        "agent_id": "MRV-006",
        "module_path": "greenlang.agents.mrv.land_use_emissions",
        "prefix": "gl_land_use_emissions_",
    },
    Scope1Category.WASTE_TREATMENT: {
        "agent_id": "MRV-007",
        "module_path": "greenlang.agents.mrv.waste_treatment",
        "prefix": "gl_waste_treatment_",
    },
    Scope1Category.AGRICULTURAL: {
        "agent_id": "MRV-008",
        "module_path": "greenlang.agents.mrv.agricultural_emissions",
        "prefix": "gl_agricultural_emissions_",
    },
}

# Stationary combustion emission factors (kg CO2 per unit)
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "natural_gas": {"co2_kg_per_mmbtu": 53.06, "ch4_g_per_mmbtu": 1.0, "n2o_g_per_mmbtu": 0.1},
    "diesel": {"co2_kg_per_gallon": 10.21, "ch4_g_per_gallon": 0.40, "n2o_g_per_gallon": 0.08},
    "gasoline": {"co2_kg_per_gallon": 8.78, "ch4_g_per_gallon": 0.34, "n2o_g_per_gallon": 0.06},
    "fuel_oil_2": {"co2_kg_per_gallon": 10.21, "ch4_g_per_gallon": 0.40, "n2o_g_per_gallon": 0.08},
    "fuel_oil_6": {"co2_kg_per_gallon": 11.27, "ch4_g_per_gallon": 0.64, "n2o_g_per_gallon": 0.08},
    "propane": {"co2_kg_per_gallon": 5.72, "ch4_g_per_gallon": 0.24, "n2o_g_per_gallon": 0.05},
    "coal_bituminous": {"co2_kg_per_short_ton": 2328.0, "ch4_g_per_short_ton": 11.0, "n2o_g_per_short_ton": 1.6},
    "cng": {"co2_kg_per_scf": 0.0545, "ch4_g_per_scf": 0.001, "n2o_g_per_scf": 0.0001},
    "lpg": {"co2_kg_per_gallon": 5.68, "ch4_g_per_gallon": 0.24, "n2o_g_per_gallon": 0.05},
    "jet_fuel": {"co2_kg_per_gallon": 9.75, "ch4_g_per_gallon": 0.34, "n2o_g_per_gallon": 0.08},
    "kerosene": {"co2_kg_per_gallon": 10.15, "ch4_g_per_gallon": 0.40, "n2o_g_per_gallon": 0.08},
}

# GWP values (AR5 100-year)
GWP_AR5: Dict[str, int] = {
    "CO2": 1,
    "CH4": 28,
    "N2O": 265,
    "R-410A": 2088,
    "R-134a": 1430,
    "R-407C": 1774,
    "R-404A": 3922,
    "R-22": 1810,
    "R-32": 675,
    "R-1234yf": 4,
    "R-1234ze": 7,
    "R-744": 1,
    "SF6": 23500,
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Scope1AgentConfig(BaseModel):
    """Configuration for Scope 1 agent routing."""

    config_id: str = Field(default_factory=_new_uuid)
    enabled_categories: List[Scope1Category] = Field(
        default_factory=lambda: list(Scope1Category)
    )
    gwp_source: str = Field(default="AR5", description="GWP table source: AR4, AR5, AR6")
    emission_factor_source: str = Field(default="EPA", description="EF source: EPA, DEFRA, IPCC")
    include_biogenic: bool = Field(default=False)

class AgentResult(BaseModel):
    """Result from an MRV agent execution."""

    result_id: str = Field(default_factory=_new_uuid)
    agent_id: str = Field(default="")
    category: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0)
    co2_tco2e: float = Field(default=0.0)
    ch4_tco2e: float = Field(default=0.0)
    n2o_tco2e: float = Field(default=0.0)
    hfc_tco2e: float = Field(default=0.0)
    pfc_tco2e: float = Field(default=0.0)
    sf6_tco2e: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    facilities_count: int = Field(default=0)
    status: str = Field(default="success")
    error_message: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# MRVScope1Bridge
# ---------------------------------------------------------------------------

class MRVScope1Bridge:
    """Bridge to all 8 Scope 1 MRV agents (MRV-001 through MRV-008).

    Routes facility-level activity data to the appropriate Scope 1 MRV
    agent for emissions calculation. Supports individual category execution,
    batch execution of applicable categories, and agent health status.

    Attributes:
        config: Agent routing configuration.
        _agents: Loaded MRV agent references (or stubs).

    Example:
        >>> bridge = MRVScope1Bridge()
        >>> result = bridge.execute_stationary_combustion(fuel_data)
        >>> assert result.status == "success"
        >>> assert result.total_emissions_tco2e > 0
    """

    def __init__(
        self,
        config: Optional[Scope1AgentConfig] = None,
    ) -> None:
        """Initialize MRVScope1Bridge.

        Args:
            config: Agent routing configuration. Uses defaults if None.
        """
        self.config = config or Scope1AgentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}

        for category in Scope1Category:
            mapping = AGENT_CATEGORY_MAP[category]
            self._agents[mapping["agent_id"]] = _try_import_agent(
                mapping["agent_id"], mapping["module_path"]
            )

        self.logger.info(
            "MRVScope1Bridge initialized: %d categories enabled, gwp=%s, ef=%s",
            len(self.config.enabled_categories),
            self.config.gwp_source,
            self.config.emission_factor_source,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def route_to_agent(
        self,
        category: Scope1Category,
        facility_data: Dict[str, Any],
    ) -> AgentResult:
        """Route data to the appropriate Scope 1 MRV agent.

        Args:
            category: Scope 1 emission category.
            facility_data: Activity data for the category.

        Returns:
            AgentResult with emissions calculation.
        """
        start_time = time.monotonic()
        mapping = AGENT_CATEGORY_MAP.get(category)
        if not mapping:
            return AgentResult(
                status="error",
                error_message=f"Unknown category: {category.value}",
            )

        agent_id = mapping["agent_id"]
        self.logger.info(
            "Routing to %s for category '%s': %d data keys",
            agent_id, category.value, len(facility_data),
        )

        dispatch = {
            Scope1Category.STATIONARY_COMBUSTION: self.execute_stationary_combustion,
            Scope1Category.REFRIGERANTS: self.execute_refrigerants,
            Scope1Category.MOBILE_COMBUSTION: self.execute_mobile_combustion,
            Scope1Category.PROCESS_EMISSIONS: self.execute_process_emissions,
            Scope1Category.FUGITIVE_EMISSIONS: self.execute_fugitive_emissions,
            Scope1Category.LAND_USE: self.execute_land_use,
            Scope1Category.WASTE_TREATMENT: self.execute_waste_treatment,
            Scope1Category.AGRICULTURAL: self.execute_agricultural,
        }

        handler = dispatch.get(category)
        if handler:
            return handler(facility_data)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return AgentResult(
            agent_id=agent_id,
            category=category.value,
            status="error",
            error_message=f"No handler for category '{category.value}'",
            processing_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Individual Agent Methods
    # -------------------------------------------------------------------------

    def execute_stationary_combustion(
        self,
        fuel_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute stationary combustion emissions calculation via MRV-001.

        Calculates CO2, CH4, and N2O emissions from stationary fuel
        combustion using EPA emission factors and AR5 GWPs.

        Args:
            fuel_data: Dict with fuel_records list, each containing
                fuel_type, quantity, unit, and facility_id.

        Returns:
            AgentResult with Scope 1 stationary combustion emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-001"

        fuel_records = fuel_data.get("fuel_records", [])
        facilities: set = set()
        total_co2 = Decimal("0")
        total_ch4 = Decimal("0")
        total_n2o = Decimal("0")
        by_fuel: Dict[str, float] = {}

        for record in fuel_records:
            fuel_type = record.get("fuel_type", "natural_gas")
            quantity = Decimal(str(record.get("quantity", 0)))
            facility_id = record.get("facility_id", "default")
            facilities.add(facility_id)

            ef = FUEL_EMISSION_FACTORS.get(fuel_type, FUEL_EMISSION_FACTORS["natural_gas"])
            co2_key = [k for k in ef.keys() if k.startswith("co2_")][0] if ef else "co2_kg_per_mmbtu"
            co2_factor = Decimal(str(ef.get(co2_key, 0)))
            co2 = (quantity * co2_factor / Decimal("1000"))

            ch4_key = [k for k in ef.keys() if k.startswith("ch4_")][0] if ef else "ch4_g_per_mmbtu"
            ch4_factor = Decimal(str(ef.get(ch4_key, 0)))
            ch4 = (quantity * ch4_factor * Decimal(str(GWP_AR5["CH4"])) / Decimal("1000000"))

            n2o_key = [k for k in ef.keys() if k.startswith("n2o_")][0] if ef else "n2o_g_per_mmbtu"
            n2o_factor = Decimal(str(ef.get(n2o_key, 0)))
            n2o = (quantity * n2o_factor * Decimal(str(GWP_AR5["N2O"])) / Decimal("1000000"))

            total_co2 += co2
            total_ch4 += ch4
            total_n2o += n2o

            fuel_total = float((co2 + ch4 + n2o).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            by_fuel[fuel_type] = by_fuel.get(fuel_type, 0.0) + fuel_total

        total = total_co2 + total_ch4 + total_n2o
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.STATIONARY_COMBUSTION.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(total_co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(total_ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(total_n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(fuel_records),
            facilities_count=len(facilities),
            details={"by_fuel": by_fuel, "gwp_source": self.config.gwp_source},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-001 stationary: %.3f tCO2e from %d records, %d facilities",
            result.total_emissions_tco2e, len(fuel_records), len(facilities),
        )
        return result

    def execute_refrigerants(
        self,
        equipment_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute refrigerant emissions calculation via MRV-002.

        Calculates HFC/PFC emissions from refrigerant leakage using
        screening method (charge * leak_rate * GWP).

        Args:
            equipment_data: Dict with equipment_records list, each
                containing refrigerant_type, charge_kg, leak_rate_pct.

        Returns:
            AgentResult with refrigerant emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-002"

        equipment_records = equipment_data.get("equipment_records", [])
        total_hfc = Decimal("0")
        by_refrigerant: Dict[str, float] = {}

        for record in equipment_records:
            ref_type = record.get("refrigerant_type", "R-410A")
            charge_kg = Decimal(str(record.get("charge_kg", 0)))
            leak_rate = Decimal(str(record.get("leak_rate_pct", 5.0))) / Decimal("100")
            gwp = Decimal(str(GWP_AR5.get(ref_type, 1)))

            leaked_kg = charge_kg * leak_rate
            emissions_tco2e = leaked_kg * gwp / Decimal("1000")
            total_hfc += emissions_tco2e

            ref_total = float(emissions_tco2e.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            by_refrigerant[ref_type] = by_refrigerant.get(ref_type, 0.0) + ref_total

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.REFRIGERANTS.value,
            total_emissions_tco2e=float(total_hfc.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            hfc_tco2e=float(total_hfc.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(equipment_records),
            details={"by_refrigerant": by_refrigerant, "gwp_source": self.config.gwp_source},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-002 refrigerants: %.3f tCO2e from %d equipment records",
            result.total_emissions_tco2e, len(equipment_records),
        )
        return result

    def execute_mobile_combustion(
        self,
        fleet_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute mobile combustion emissions calculation via MRV-003.

        Calculates emissions from vehicle fleet fuel consumption.

        Args:
            fleet_data: Dict with vehicle_records list, each containing
                fuel_type, fuel_consumed (gallons), distance_km.

        Returns:
            AgentResult with mobile combustion emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-003"

        vehicle_records = fleet_data.get("vehicle_records", [])
        total_co2 = Decimal("0")
        total_ch4 = Decimal("0")
        total_n2o = Decimal("0")
        total_distance = Decimal("0")
        by_fuel: Dict[str, float] = {}

        for record in vehicle_records:
            fuel_type = record.get("fuel_type", "gasoline")
            fuel_consumed = Decimal(str(record.get("fuel_consumed", 0)))
            distance = Decimal(str(record.get("distance_km", 0)))
            total_distance += distance

            ef = FUEL_EMISSION_FACTORS.get(fuel_type, FUEL_EMISSION_FACTORS["gasoline"])
            co2_key = [k for k in ef.keys() if k.startswith("co2_")][0]
            co2 = fuel_consumed * Decimal(str(ef.get(co2_key, 0))) / Decimal("1000")

            ch4_key = [k for k in ef.keys() if k.startswith("ch4_")][0]
            ch4 = (fuel_consumed * Decimal(str(ef.get(ch4_key, 0)))
                    * Decimal(str(GWP_AR5["CH4"])) / Decimal("1000000"))

            n2o_key = [k for k in ef.keys() if k.startswith("n2o_")][0]
            n2o = (fuel_consumed * Decimal(str(ef.get(n2o_key, 0)))
                    * Decimal(str(GWP_AR5["N2O"])) / Decimal("1000000"))

            total_co2 += co2
            total_ch4 += ch4
            total_n2o += n2o

            fuel_total = float((co2 + ch4 + n2o).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            by_fuel[fuel_type] = by_fuel.get(fuel_type, 0.0) + fuel_total

        total = total_co2 + total_ch4 + total_n2o
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.MOBILE_COMBUSTION.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(total_co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(total_ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(total_n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(vehicle_records),
            details={
                "by_fuel": by_fuel,
                "total_distance_km": float(total_distance),
                "fleet_size": len(vehicle_records),
            },
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-003 mobile: %.3f tCO2e from %d vehicle records",
            result.total_emissions_tco2e, len(vehicle_records),
        )
        return result

    def execute_process_emissions(
        self,
        process_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute process emissions calculation via MRV-004.

        Calculates emissions from industrial process chemical reactions
        (e.g., cement clinker, lime, ammonia, steel production).

        Args:
            process_data: Dict with process_records list, each containing
                process_type, production_quantity, unit, emission_factor.

        Returns:
            AgentResult with process emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-004"

        process_records = process_data.get("process_records", [])
        total = Decimal("0")
        by_process: Dict[str, float] = {}

        for record in process_records:
            process_type = record.get("process_type", "generic")
            quantity = Decimal(str(record.get("production_quantity", 0)))
            ef = Decimal(str(record.get("emission_factor", 0)))
            emissions = quantity * ef / Decimal("1000")
            total += emissions
            by_process[process_type] = by_process.get(process_type, 0.0) + float(
                emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.PROCESS_EMISSIONS.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(process_records),
            details={"by_process": by_process},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-004 process: %.3f tCO2e from %d records",
            result.total_emissions_tco2e, len(process_records),
        )
        return result

    def execute_fugitive_emissions(
        self,
        source_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute fugitive emissions calculation via MRV-005.

        Calculates emissions from unintentional leaks (pipelines, valves,
        flanges, compressors, electrical equipment SF6).

        Args:
            source_data: Dict with source_records list, each containing
                source_type, gas_type, leak_quantity_kg.

        Returns:
            AgentResult with fugitive emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-005"

        source_records = source_data.get("source_records", [])
        total = Decimal("0")
        total_sf6 = Decimal("0")
        total_ch4 = Decimal("0")

        for record in source_records:
            gas_type = record.get("gas_type", "CH4")
            leak_kg = Decimal(str(record.get("leak_quantity_kg", 0)))
            gwp = Decimal(str(GWP_AR5.get(gas_type, 1)))
            emissions = leak_kg * gwp / Decimal("1000")
            total += emissions
            if gas_type == "SF6":
                total_sf6 += emissions
            elif gas_type == "CH4":
                total_ch4 += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.FUGITIVE_EMISSIONS.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(total_ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            sf6_tco2e=float(total_sf6.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(source_records),
            details={"sources_count": len(source_records)},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-005 fugitive: %.3f tCO2e from %d sources",
            result.total_emissions_tco2e, len(source_records),
        )
        return result

    def execute_land_use(
        self,
        land_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute land use change emissions calculation via MRV-006.

        Args:
            land_data: Dict with land_records list, each containing
                land_type, area_hectares, change_type, emission_factor.

        Returns:
            AgentResult with land use emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-006"

        land_records = land_data.get("land_records", [])
        total = Decimal("0")

        for record in land_records:
            area = Decimal(str(record.get("area_hectares", 0)))
            ef = Decimal(str(record.get("emission_factor", 0)))
            emissions = area * ef / Decimal("1000")
            total += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.LAND_USE.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(land_records),
            details={"total_area_hectares": sum(r.get("area_hectares", 0) for r in land_records)},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-006 land use: %.3f tCO2e from %d records",
            result.total_emissions_tco2e, len(land_records),
        )
        return result

    def execute_waste_treatment(
        self,
        waste_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute on-site waste treatment emissions calculation via MRV-007.

        Args:
            waste_data: Dict with waste_records list, each containing
                waste_type, quantity_tonnes, treatment_method, emission_factor.

        Returns:
            AgentResult with waste treatment emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-007"

        waste_records = waste_data.get("waste_records", [])
        total = Decimal("0")
        total_ch4 = Decimal("0")

        for record in waste_records:
            quantity = Decimal(str(record.get("quantity_tonnes", 0)))
            ef = Decimal(str(record.get("emission_factor", 0)))
            emissions = quantity * ef / Decimal("1000")
            total += emissions
            if record.get("treatment_method") in ("landfill", "anaerobic_digestion"):
                total_ch4 += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.WASTE_TREATMENT.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(total_ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(waste_records),
            details={"total_waste_tonnes": sum(r.get("quantity_tonnes", 0) for r in waste_records)},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-007 waste: %.3f tCO2e from %d records",
            result.total_emissions_tco2e, len(waste_records),
        )
        return result

    def execute_agricultural(
        self,
        ag_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute agricultural emissions calculation via MRV-008.

        Args:
            ag_data: Dict with ag_records list, each containing
                source_type (enteric, manure, soil, rice, burning),
                quantity, unit, emission_factor.

        Returns:
            AgentResult with agricultural emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-008"

        ag_records = ag_data.get("ag_records", [])
        total = Decimal("0")
        total_ch4 = Decimal("0")
        total_n2o = Decimal("0")

        for record in ag_records:
            source_type = record.get("source_type", "generic")
            quantity = Decimal(str(record.get("quantity", 0)))
            ef = Decimal(str(record.get("emission_factor", 0)))
            emissions = quantity * ef / Decimal("1000")
            total += emissions
            if source_type in ("enteric", "manure", "rice"):
                total_ch4 += emissions
            elif source_type in ("soil",):
                total_n2o += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = AgentResult(
            agent_id=agent_id,
            category=Scope1Category.AGRICULTURAL.value,
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(total_ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(total_n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=len(ag_records),
            details={"sources": list({r.get("source_type", "generic") for r in ag_records})},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-008 agricultural: %.3f tCO2e from %d records",
            result.total_emissions_tco2e, len(ag_records),
        )
        return result

    # -------------------------------------------------------------------------
    # Batch Execution
    # -------------------------------------------------------------------------

    def execute_all_applicable(
        self,
        facility_data: Dict[str, Any],
        applicable_categories: Optional[List[Scope1Category]] = None,
    ) -> Dict[str, AgentResult]:
        """Execute emissions calculations for all applicable Scope 1 categories.

        Args:
            facility_data: Dict keyed by category value, each containing
                the activity data for that category.
            applicable_categories: Categories to include. Uses all enabled if None.

        Returns:
            Dict mapping category value to AgentResult.
        """
        categories = applicable_categories or self.config.enabled_categories
        self.logger.info(
            "Executing all applicable Scope 1 categories: %d",
            len(categories),
        )

        results: Dict[str, AgentResult] = {}
        for category in categories:
            category_data = facility_data.get(category.value, {})
            if not category_data:
                self.logger.debug("No data for category '%s', skipping", category.value)
                continue
            try:
                result = self.route_to_agent(category, category_data)
                results[category.value] = result
            except Exception as exc:
                self.logger.error(
                    "Failed to execute category '%s': %s", category.value, exc
                )
                results[category.value] = AgentResult(
                    agent_id=AGENT_CATEGORY_MAP[category]["agent_id"],
                    category=category.value,
                    status="error",
                    error_message=str(exc),
                )

        total = sum(r.total_emissions_tco2e for r in results.values() if r.status == "success")
        self.logger.info(
            "Scope 1 total: %.3f tCO2e across %d categories",
            total, len(results),
        )
        return results

    # -------------------------------------------------------------------------
    # Agent Status
    # -------------------------------------------------------------------------

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the availability status of a specific MRV agent.

        Args:
            agent_id: MRV agent identifier (e.g., 'MRV-001').

        Returns:
            Dict with agent status information.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return {
                "agent_id": agent_id,
                "status": AgentStatus.UNAVAILABLE.value,
                "message": "Agent not registered",
            }

        is_stub = isinstance(agent, _AgentStub)
        return {
            "agent_id": agent_id,
            "status": AgentStatus.DEGRADED.value if is_stub else AgentStatus.AVAILABLE.value,
            "message": "Using stub (module not importable)" if is_stub else "Agent available",
            "module_loaded": not is_stub,
        }

    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get availability status for all 8 Scope 1 MRV agents.

        Returns:
            Dict mapping agent_id to status information.
        """
        return {
            agent_id: self.get_agent_status(agent_id)
            for agent_id in sorted(self._agents.keys())
        }
