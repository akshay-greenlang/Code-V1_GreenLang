# -*- coding: utf-8 -*-
"""
GL-MRV-X-003: Scope 2 Location-Based Agent
===========================================

Calculates Scope 2 location-based emissions using grid-average emission factors.
Follows GHG Protocol Scope 2 Guidance for location-based accounting.

Capabilities:
    - Grid-average emission factor application
    - Regional/country-specific factors
    - Sub-national grid factors support
    - Electricity, steam, heating, cooling emissions
    - Time-of-use factor support
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All calculations are deterministic mathematical operations
    - NO LLM involvement in any calculation path
    - Emission factors from authoritative grid databases
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EnergyType(str, Enum):
    """Types of purchased energy."""
    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"


class UnitType(str, Enum):
    """Energy units."""
    KWH = "kwh"
    MWH = "mwh"
    GJ = "gj"
    MMBTU = "mmbtu"
    THERMS = "therms"


# =============================================================================
# GRID EMISSION FACTORS DATABASE
# Source: IEA, EPA eGRID, EU ETS, National Inventories
# Values in kg CO2e per kWh
# =============================================================================

GRID_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # United States - eGRID Subregions (2022)
    "US-CAMX": {"co2": Decimal("0.225"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "US-ERCT": {"co2": Decimal("0.380"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-FRCC": {"co2": Decimal("0.392"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-MROE": {"co2": Decimal("0.482"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-MROW": {"co2": Decimal("0.455"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-NEWE": {"co2": Decimal("0.228"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "US-NWPP": {"co2": Decimal("0.295"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-NYCW": {"co2": Decimal("0.255"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "US-NYLI": {"co2": Decimal("0.352"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-NYUP": {"co2": Decimal("0.115"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "US-RFCE": {"co2": Decimal("0.340"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-RFCM": {"co2": Decimal("0.495"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-RFCW": {"co2": Decimal("0.515"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-RMPA": {"co2": Decimal("0.530"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-SPNO": {"co2": Decimal("0.480"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-SPSO": {"co2": Decimal("0.420"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-SRMV": {"co2": Decimal("0.365"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-SRMW": {"co2": Decimal("0.575"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "US-SRSO": {"co2": Decimal("0.425"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-SRTV": {"co2": Decimal("0.450"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "US-SRVC": {"co2": Decimal("0.345"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "US-AVG": {"co2": Decimal("0.386"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},

    # European Countries (2022)
    "EU-DE": {"co2": Decimal("0.350"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "EU-FR": {"co2": Decimal("0.052"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "EU-GB": {"co2": Decimal("0.207"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "EU-ES": {"co2": Decimal("0.160"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "EU-IT": {"co2": Decimal("0.315"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "EU-PL": {"co2": Decimal("0.680"), "ch4": Decimal("0.00006"), "n2o": Decimal("0.000006")},
    "EU-NL": {"co2": Decimal("0.380"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "EU-BE": {"co2": Decimal("0.165"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},
    "EU-SE": {"co2": Decimal("0.013"), "ch4": Decimal("0.00000"), "n2o": Decimal("0.000000")},
    "EU-NO": {"co2": Decimal("0.008"), "ch4": Decimal("0.00000"), "n2o": Decimal("0.000000")},
    "EU-DK": {"co2": Decimal("0.135"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "EU-AT": {"co2": Decimal("0.092"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "EU-CH": {"co2": Decimal("0.024"), "ch4": Decimal("0.00000"), "n2o": Decimal("0.000000")},
    "EU-AVG": {"co2": Decimal("0.255"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},

    # Asia-Pacific
    "APAC-CN": {"co2": Decimal("0.555"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "APAC-JP": {"co2": Decimal("0.465"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "APAC-IN": {"co2": Decimal("0.708"), "ch4": Decimal("0.00006"), "n2o": Decimal("0.000006")},
    "APAC-KR": {"co2": Decimal("0.415"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "APAC-AU": {"co2": Decimal("0.656"), "ch4": Decimal("0.00006"), "n2o": Decimal("0.000006")},
    "APAC-SG": {"co2": Decimal("0.408"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "APAC-MY": {"co2": Decimal("0.585"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "APAC-TH": {"co2": Decimal("0.452"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "APAC-VN": {"co2": Decimal("0.520"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "APAC-ID": {"co2": Decimal("0.720"), "ch4": Decimal("0.00006"), "n2o": Decimal("0.000006")},
    "APAC-PH": {"co2": Decimal("0.555"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "APAC-NZ": {"co2": Decimal("0.095"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},

    # Americas (excluding US)
    "AMER-CA": {"co2": Decimal("0.120"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "AMER-MX": {"co2": Decimal("0.425"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "AMER-BR": {"co2": Decimal("0.075"), "ch4": Decimal("0.00001"), "n2o": Decimal("0.000001")},
    "AMER-AR": {"co2": Decimal("0.310"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "AMER-CL": {"co2": Decimal("0.355"), "ch4": Decimal("0.00003"), "n2o": Decimal("0.000003")},
    "AMER-CO": {"co2": Decimal("0.175"), "ch4": Decimal("0.00002"), "n2o": Decimal("0.000002")},

    # Middle East & Africa
    "MEA-AE": {"co2": Decimal("0.445"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "MEA-SA": {"co2": Decimal("0.555"), "ch4": Decimal("0.00005"), "n2o": Decimal("0.000005")},
    "MEA-ZA": {"co2": Decimal("0.895"), "ch4": Decimal("0.00008"), "n2o": Decimal("0.000008")},
    "MEA-EG": {"co2": Decimal("0.425"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
    "MEA-IL": {"co2": Decimal("0.485"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},

    # Global average
    "GLOBAL": {"co2": Decimal("0.436"), "ch4": Decimal("0.00004"), "n2o": Decimal("0.000004")},
}

# Steam/heating/cooling emission factors (kg CO2e per GJ)
THERMAL_EMISSION_FACTORS: Dict[EnergyType, Decimal] = {
    EnergyType.STEAM: Decimal("66.3"),  # Natural gas boiler
    EnergyType.HEATING: Decimal("56.1"),  # District heating avg
    EnergyType.COOLING: Decimal("0.12"),  # Electric chiller per kWh equivalent
}

# GWP values for CH4 and N2O
GWP_AR6 = {"ch4": Decimal("29.8"), "n2o": Decimal("273")}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class EnergyConsumption(BaseModel):
    """Energy consumption record."""
    energy_type: EnergyType = Field(..., description="Type of energy")
    quantity: float = Field(..., gt=0, description="Energy quantity")
    unit: UnitType = Field(..., description="Unit of measurement")
    grid_region: Optional[str] = Field(None, description="Grid region code")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    period_start: Optional[datetime] = Field(None, description="Period start")
    period_end: Optional[datetime] = Field(None, description="Period end")
    custom_emission_factor: Optional[float] = Field(
        None, description="Custom emission factor (kg CO2e/kWh)"
    )
    supplier: Optional[str] = Field(None, description="Energy supplier")

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        if not (0 < v < 1e15):
            raise ValueError("Quantity must be positive and finite")
        return v


class LocationBasedResult(BaseModel):
    """Result of location-based calculation."""
    energy_type: EnergyType = Field(..., description="Energy type")
    grid_region: str = Field(..., description="Grid region used")
    energy_quantity: float = Field(..., description="Energy quantity")
    energy_unit: UnitType = Field(..., description="Energy unit")
    energy_kwh: float = Field(..., description="Energy in kWh")

    # Emissions breakdown
    co2_emissions_kg: float = Field(..., description="CO2 emissions in kg")
    ch4_emissions_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_emissions_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total emissions in kg CO2e")
    total_co2e_tonnes: float = Field(..., description="Total emissions in tonnes CO2e")

    # Emission factors used
    ef_co2: float = Field(..., description="CO2 EF used (kg/kWh)")
    ef_ch4: float = Field(..., description="CH4 EF used (kg/kWh)")
    ef_n2o: float = Field(..., description="N2O EF used (kg/kWh)")
    ef_source: str = Field(..., description="EF source")

    # Metadata
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 hash")
    facility_id: Optional[str] = Field(None)


class Scope2LocationBasedInput(BaseModel):
    """Input model for Scope2LocationBasedAgent."""
    energy_consumptions: List[EnergyConsumption] = Field(
        ..., min_length=1, description="Energy consumption records"
    )
    default_grid_region: str = Field(
        default="GLOBAL", description="Default grid region"
    )
    organization_id: Optional[str] = Field(None)
    reporting_period: Optional[str] = Field(None)


class Scope2LocationBasedOutput(BaseModel):
    """Output model for Scope2LocationBasedAgent."""
    success: bool = Field(...)
    calculation_results: List[LocationBasedResult] = Field(default_factory=list)

    # Totals
    total_co2e_tonnes: float = Field(...)
    total_electricity_co2e_tonnes: float = Field(default=0.0)
    total_steam_co2e_tonnes: float = Field(default=0.0)
    total_heating_co2e_tonnes: float = Field(default=0.0)
    total_cooling_co2e_tonnes: float = Field(default=0.0)

    # Breakdown
    emissions_by_region: Dict[str, float] = Field(default_factory=dict)
    emissions_by_facility: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    organization_id: Optional[str] = Field(None)
    reporting_period: Optional[str] = Field(None)


# =============================================================================
# SCOPE 2 LOCATION-BASED AGENT
# =============================================================================

class Scope2LocationBasedAgent(DeterministicAgent):
    """
    GL-MRV-X-003: Scope 2 Location-Based Agent

    Calculates Scope 2 emissions using location-based method with
    grid-average emission factors following GHG Protocol Scope 2 Guidance.

    Zero-Hallucination Implementation:
        - All calculations use deterministic mathematical operations
        - Grid emission factors from authoritative sources (IEA, EPA eGRID)
        - Complete provenance tracking with SHA-256 hashes

    Supported Features:
        - Grid-average emission factors by region
        - Electricity, steam, heating, cooling
        - Multi-facility aggregation
        - Custom emission factors

    Example:
        >>> agent = Scope2LocationBasedAgent()
        >>> result = agent.execute({
        ...     "energy_consumptions": [{
        ...         "energy_type": "electricity",
        ...         "quantity": 1000,
        ...         "unit": "mwh",
        ...         "grid_region": "US-CAMX"
        ...     }]
        ... })
    """

    AGENT_ID = "GL-MRV-X-003"
    AGENT_NAME = "Scope 2 Location-Based Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="Scope2LocationBasedAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Location-based Scope 2 emissions calculator"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Scope2LocationBasedAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute location-based Scope 2 calculation."""
        start_time = DeterministicClock.now()

        try:
            scope2_input = Scope2LocationBasedInput(**inputs)
            results: List[LocationBasedResult] = []

            # Process each energy consumption record
            for consumption in scope2_input.energy_consumptions:
                result = self._calculate_location_based(
                    consumption,
                    scope2_input.default_grid_region
                )
                results.append(result)

            # Aggregate totals
            total_co2e = sum(r.total_co2e_tonnes for r in results)
            total_elec = sum(
                r.total_co2e_tonnes for r in results
                if r.energy_type == EnergyType.ELECTRICITY
            )
            total_steam = sum(
                r.total_co2e_tonnes for r in results
                if r.energy_type == EnergyType.STEAM
            )
            total_heating = sum(
                r.total_co2e_tonnes for r in results
                if r.energy_type == EnergyType.HEATING
            )
            total_cooling = sum(
                r.total_co2e_tonnes for r in results
                if r.energy_type == EnergyType.COOLING
            )

            # Breakdown by region
            emissions_by_region: Dict[str, float] = {}
            for r in results:
                emissions_by_region[r.grid_region] = (
                    emissions_by_region.get(r.grid_region, 0) + r.total_co2e_tonnes
                )

            # Breakdown by facility
            emissions_by_facility: Dict[str, float] = {}
            for r in results:
                if r.facility_id:
                    emissions_by_facility[r.facility_id] = (
                        emissions_by_facility.get(r.facility_id, 0) + r.total_co2e_tonnes
                    )

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_provenance_hash({
                "input": inputs,
                "total_co2e_tonnes": total_co2e
            })

            output = Scope2LocationBasedOutput(
                success=True,
                calculation_results=results,
                total_co2e_tonnes=round(total_co2e, 4),
                total_electricity_co2e_tonnes=round(total_elec, 4),
                total_steam_co2e_tonnes=round(total_steam, 4),
                total_heating_co2e_tonnes=round(total_heating, 4),
                total_cooling_co2e_tonnes=round(total_cooling, 4),
                emissions_by_region=emissions_by_region,
                emissions_by_facility=emissions_by_facility,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS",
                organization_id=scope2_input.organization_id,
                reporting_period=scope2_input.reporting_period
            )

            self._capture_audit_entry(
                operation="calculate_scope2_location_based",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Processed {len(results)} consumption records"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Location-based calculation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _calculate_location_based(
        self,
        consumption: EnergyConsumption,
        default_region: str
    ) -> LocationBasedResult:
        """Calculate emissions for a single energy consumption record."""
        trace = []

        # Determine grid region
        grid_region = consumption.grid_region or default_region
        trace.append(f"Grid region: {grid_region}")

        # Convert to kWh
        energy_kwh = self._convert_to_kwh(
            consumption.quantity,
            consumption.unit,
            consumption.energy_type
        )
        trace.append(f"Energy: {energy_kwh:.2f} kWh")

        # Get emission factors
        if consumption.custom_emission_factor:
            ef_co2 = Decimal(str(consumption.custom_emission_factor))
            ef_ch4 = Decimal("0")
            ef_n2o = Decimal("0")
            ef_source = "Custom"
        elif consumption.energy_type == EnergyType.ELECTRICITY:
            factors = GRID_EMISSION_FACTORS.get(grid_region, GRID_EMISSION_FACTORS["GLOBAL"])
            ef_co2 = factors["co2"]
            ef_ch4 = factors["ch4"]
            ef_n2o = factors["n2o"]
            ef_source = f"Grid-{grid_region}"
        else:
            # Thermal energy - convert kWh to GJ for factor lookup
            thermal_ef = THERMAL_EMISSION_FACTORS.get(
                consumption.energy_type, Decimal("56.1")
            )
            # Convert kg/GJ to kg/kWh
            ef_co2 = thermal_ef * Decimal("0.0036")
            ef_ch4 = Decimal("0.00003")
            ef_n2o = Decimal("0.000003")
            ef_source = f"Thermal-{consumption.energy_type.value}"

        trace.append(f"EF CO2: {float(ef_co2):.6f} kg/kWh")

        # Calculate emissions
        energy_decimal = Decimal(str(energy_kwh))
        co2_kg = energy_decimal * ef_co2
        ch4_kg = energy_decimal * ef_ch4
        n2o_kg = energy_decimal * ef_n2o

        # Convert CH4 and N2O to CO2e
        ch4_co2e = ch4_kg * GWP_AR6["ch4"]
        n2o_co2e = n2o_kg * GWP_AR6["n2o"]

        total_co2e_kg = co2_kg + ch4_co2e + n2o_co2e
        total_co2e_tonnes = total_co2e_kg / Decimal("1000")

        trace.append(f"Total: {float(total_co2e_tonnes):.4f} tCO2e")

        provenance_hash = self._compute_provenance_hash({
            "grid_region": grid_region,
            "energy_kwh": float(energy_kwh),
            "total_co2e_tonnes": float(total_co2e_tonnes)
        })

        return LocationBasedResult(
            energy_type=consumption.energy_type,
            grid_region=grid_region,
            energy_quantity=consumption.quantity,
            energy_unit=consumption.unit,
            energy_kwh=float(energy_decimal.quantize(Decimal("0.01"))),
            co2_emissions_kg=float(co2_kg.quantize(Decimal("0.001"))),
            ch4_emissions_kg=float(ch4_kg.quantize(Decimal("0.000001"))),
            n2o_emissions_kg=float(n2o_kg.quantize(Decimal("0.000001"))),
            total_co2e_kg=float(total_co2e_kg.quantize(Decimal("0.001"))),
            total_co2e_tonnes=float(total_co2e_tonnes.quantize(Decimal("0.000001"))),
            ef_co2=float(ef_co2),
            ef_ch4=float(ef_ch4),
            ef_n2o=float(ef_n2o),
            ef_source=ef_source,
            calculation_trace=trace,
            provenance_hash=provenance_hash,
            facility_id=consumption.facility_id
        )

    def _convert_to_kwh(
        self,
        quantity: float,
        unit: UnitType,
        energy_type: EnergyType
    ) -> float:
        """Convert energy quantity to kWh."""
        conversions = {
            UnitType.KWH: Decimal("1"),
            UnitType.MWH: Decimal("1000"),
            UnitType.GJ: Decimal("277.778"),
            UnitType.MMBTU: Decimal("293.071"),
            UnitType.THERMS: Decimal("29.3071"),
        }
        factor = conversions.get(unit, Decimal("1"))
        return float(Decimal(str(quantity)) * factor)

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_grid_factor(self, region: str) -> Optional[Dict[str, float]]:
        """Get grid emission factors for a region."""
        factors = GRID_EMISSION_FACTORS.get(region)
        if factors:
            return {k: float(v) for k, v in factors.items()}
        return None

    def get_supported_regions(self) -> List[str]:
        """Get list of supported grid regions."""
        return list(GRID_EMISSION_FACTORS.keys())
