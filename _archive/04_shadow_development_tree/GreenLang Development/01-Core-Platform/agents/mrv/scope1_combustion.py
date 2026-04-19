# -*- coding: utf-8 -*-
"""
GL-MRV-X-001: Scope 1 Combustion Calculator
============================================

Calculates Scope 1 emissions from stationary and mobile combustion sources.
Applies emission factors, oxidation factors, and fuel-specific parameters
following GHG Protocol Corporate Standard methodology.

Capabilities:
    - Stationary combustion emissions (boilers, furnaces, heaters)
    - Mobile combustion emissions (fleet vehicles, equipment)
    - Emission factor application (CO2, CH4, N2O)
    - Oxidation factor adjustment
    - Fuel-specific heating value conversions
    - Multi-fuel facility aggregation
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All calculations are deterministic mathematical operations
    - NO LLM involvement in any calculation path
    - All emission factors are traceable to authoritative sources
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CombustionType(str, Enum):
    """Types of combustion sources."""
    STATIONARY = "stationary"
    MOBILE = "mobile"


class FuelType(str, Enum):
    """Standard fuel types for combustion."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    LPG = "lpg"
    PROPANE = "propane"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    WOOD = "wood"
    BIOMASS = "biomass"
    KEROSENE = "kerosene"
    JET_FUEL = "jet_fuel"
    LANDFILL_GAS = "landfill_gas"
    BIOGAS = "biogas"


class EmissionGas(str, Enum):
    """Greenhouse gases from combustion."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"


class UnitType(str, Enum):
    """Units for fuel consumption."""
    # Volume units
    LITERS = "liters"
    GALLONS = "gallons"
    CUBIC_METERS = "m3"
    CUBIC_FEET = "ft3"
    BARRELS = "barrels"
    # Mass units
    KG = "kg"
    TONNES = "tonnes"
    LBS = "lbs"
    SHORT_TONS = "short_tons"
    # Energy units
    KWH = "kwh"
    MWH = "mwh"
    GJ = "gj"
    MMBTU = "mmbtu"
    THERMS = "therms"


# =============================================================================
# EMISSION FACTORS DATABASE
# GHG Protocol Emission Factors (2024 Values)
# Source: EPA GHG Emission Factors Hub, IPCC AR6
# =============================================================================

# Stationary combustion emission factors (kg/GJ)
STATIONARY_EMISSION_FACTORS: Dict[FuelType, Dict[EmissionGas, Decimal]] = {
    FuelType.NATURAL_GAS: {
        EmissionGas.CO2: Decimal("56.1"),
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0001"),
    },
    FuelType.DIESEL: {
        EmissionGas.CO2: Decimal("74.1"),
        EmissionGas.CH4: Decimal("0.003"),
        EmissionGas.N2O: Decimal("0.0006"),
    },
    FuelType.FUEL_OIL_NO2: {
        EmissionGas.CO2: Decimal("73.96"),
        EmissionGas.CH4: Decimal("0.003"),
        EmissionGas.N2O: Decimal("0.0006"),
    },
    FuelType.FUEL_OIL_NO6: {
        EmissionGas.CO2: Decimal("77.37"),
        EmissionGas.CH4: Decimal("0.003"),
        EmissionGas.N2O: Decimal("0.0006"),
    },
    FuelType.LPG: {
        EmissionGas.CO2: Decimal("63.1"),
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0001"),
    },
    FuelType.PROPANE: {
        EmissionGas.CO2: Decimal("63.1"),
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0001"),
    },
    FuelType.COAL_BITUMINOUS: {
        EmissionGas.CO2: Decimal("94.6"),
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0015"),
    },
    FuelType.COAL_ANTHRACITE: {
        EmissionGas.CO2: Decimal("98.3"),
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0015"),
    },
    FuelType.WOOD: {
        EmissionGas.CO2: Decimal("0.0"),  # Biogenic - reported separately
        EmissionGas.CH4: Decimal("0.03"),
        EmissionGas.N2O: Decimal("0.004"),
    },
    FuelType.BIOMASS: {
        EmissionGas.CO2: Decimal("0.0"),  # Biogenic - reported separately
        EmissionGas.CH4: Decimal("0.03"),
        EmissionGas.N2O: Decimal("0.004"),
    },
    FuelType.KEROSENE: {
        EmissionGas.CO2: Decimal("71.5"),
        EmissionGas.CH4: Decimal("0.003"),
        EmissionGas.N2O: Decimal("0.0006"),
    },
    FuelType.LANDFILL_GAS: {
        EmissionGas.CO2: Decimal("0.0"),  # Biogenic
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0001"),
    },
    FuelType.BIOGAS: {
        EmissionGas.CO2: Decimal("0.0"),  # Biogenic
        EmissionGas.CH4: Decimal("0.001"),
        EmissionGas.N2O: Decimal("0.0001"),
    },
}

# Mobile combustion emission factors (kg CO2/liter for liquid fuels)
MOBILE_EMISSION_FACTORS: Dict[FuelType, Dict[EmissionGas, Decimal]] = {
    FuelType.GASOLINE: {
        EmissionGas.CO2: Decimal("2.31"),  # kg/liter
        EmissionGas.CH4: Decimal("0.00019"),
        EmissionGas.N2O: Decimal("0.00022"),
    },
    FuelType.DIESEL: {
        EmissionGas.CO2: Decimal("2.68"),  # kg/liter
        EmissionGas.CH4: Decimal("0.00013"),
        EmissionGas.N2O: Decimal("0.00044"),
    },
    FuelType.LPG: {
        EmissionGas.CO2: Decimal("1.51"),  # kg/liter
        EmissionGas.CH4: Decimal("0.00062"),
        EmissionGas.N2O: Decimal("0.00002"),
    },
    FuelType.NATURAL_GAS: {
        EmissionGas.CO2: Decimal("2.02"),  # kg/m3
        EmissionGas.CH4: Decimal("0.00092"),
        EmissionGas.N2O: Decimal("0.00002"),
    },
    FuelType.JET_FUEL: {
        EmissionGas.CO2: Decimal("2.52"),  # kg/liter
        EmissionGas.CH4: Decimal("0.00002"),
        EmissionGas.N2O: Decimal("0.00002"),
    },
}

# Fuel heating values (GJ per unit)
FUEL_HEATING_VALUES: Dict[FuelType, Dict[str, Decimal]] = {
    FuelType.NATURAL_GAS: {
        "value": Decimal("0.0373"),  # GJ/m3
        "unit": "m3",
    },
    FuelType.DIESEL: {
        "value": Decimal("0.0381"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.GASOLINE: {
        "value": Decimal("0.0341"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.LPG: {
        "value": Decimal("0.0255"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.PROPANE: {
        "value": Decimal("0.0255"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.FUEL_OIL_NO2: {
        "value": Decimal("0.0388"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.FUEL_OIL_NO6: {
        "value": Decimal("0.0422"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.COAL_BITUMINOUS: {
        "value": Decimal("26.2"),  # GJ/tonne
        "unit": "tonne",
    },
    FuelType.COAL_ANTHRACITE: {
        "value": Decimal("26.7"),  # GJ/tonne
        "unit": "tonne",
    },
    FuelType.WOOD: {
        "value": Decimal("15.4"),  # GJ/tonne (dry)
        "unit": "tonne",
    },
    FuelType.KEROSENE: {
        "value": Decimal("0.0370"),  # GJ/liter
        "unit": "liter",
    },
    FuelType.JET_FUEL: {
        "value": Decimal("0.0372"),  # GJ/liter
        "unit": "liter",
    },
}

# Default oxidation factors by fuel type
OXIDATION_FACTORS: Dict[FuelType, Decimal] = {
    FuelType.NATURAL_GAS: Decimal("0.995"),
    FuelType.DIESEL: Decimal("0.99"),
    FuelType.GASOLINE: Decimal("0.99"),
    FuelType.LPG: Decimal("0.995"),
    FuelType.PROPANE: Decimal("0.995"),
    FuelType.FUEL_OIL_NO2: Decimal("0.99"),
    FuelType.FUEL_OIL_NO6: Decimal("0.99"),
    FuelType.COAL_BITUMINOUS: Decimal("0.98"),
    FuelType.COAL_ANTHRACITE: Decimal("0.98"),
    FuelType.WOOD: Decimal("0.99"),
    FuelType.BIOMASS: Decimal("0.99"),
    FuelType.KEROSENE: Decimal("0.99"),
    FuelType.JET_FUEL: Decimal("0.99"),
    FuelType.LANDFILL_GAS: Decimal("0.995"),
    FuelType.BIOGAS: Decimal("0.995"),
}

# GWP values (AR6 100-year)
GWP_AR6: Dict[EmissionGas, Decimal] = {
    EmissionGas.CO2: Decimal("1"),
    EmissionGas.CH4: Decimal("29.8"),
    EmissionGas.N2O: Decimal("273"),
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FuelConsumption(BaseModel):
    """Fuel consumption record for a combustion source."""
    fuel_type: FuelType = Field(..., description="Type of fuel")
    quantity: float = Field(..., gt=0, description="Fuel quantity consumed")
    unit: UnitType = Field(..., description="Unit of measurement")
    combustion_type: CombustionType = Field(
        default=CombustionType.STATIONARY,
        description="Type of combustion"
    )
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    source_id: Optional[str] = Field(None, description="Source identifier")
    period_start: Optional[datetime] = Field(None, description="Period start date")
    period_end: Optional[datetime] = Field(None, description="Period end date")
    custom_heating_value: Optional[float] = Field(
        None, description="Custom heating value in GJ/unit"
    )
    custom_oxidation_factor: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Custom oxidation factor"
    )

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Ensure quantity is finite and positive."""
        if not (0 < v < 1e15):
            raise ValueError("Quantity must be positive and finite")
        return v


class EmissionResult(BaseModel):
    """Result of emissions calculation for a single gas."""
    gas: EmissionGas = Field(..., description="Greenhouse gas type")
    emissions_kg: float = Field(..., description="Emissions in kg")
    emissions_tco2e: float = Field(..., description="Emissions in tCO2e")
    emission_factor: float = Field(..., description="Applied emission factor")
    emission_factor_unit: str = Field(..., description="Unit of emission factor")
    gwp_applied: float = Field(..., description="GWP value applied")


class CombustionCalculationResult(BaseModel):
    """Complete result of a combustion emissions calculation."""
    calculation_id: str = Field(..., description="Unique calculation ID")
    fuel_type: FuelType = Field(..., description="Fuel type")
    combustion_type: CombustionType = Field(..., description="Combustion type")

    # Input data
    fuel_quantity: float = Field(..., description="Original fuel quantity")
    fuel_unit: UnitType = Field(..., description="Original fuel unit")
    energy_gj: float = Field(..., description="Converted energy in GJ")

    # Emission results
    emissions_by_gas: List[EmissionResult] = Field(
        default_factory=list, description="Emissions by gas type"
    )
    total_co2e_kg: float = Field(..., description="Total emissions in kg CO2e")
    total_co2e_tonnes: float = Field(..., description="Total emissions in tonnes CO2e")

    # Biogenic emissions (if applicable)
    biogenic_co2_kg: float = Field(default=0.0, description="Biogenic CO2 in kg")

    # Calculation metadata
    heating_value_used: float = Field(..., description="Heating value used (GJ/unit)")
    oxidation_factor_used: float = Field(..., description="Oxidation factor used")
    gwp_source: str = Field(default="AR6", description="GWP source standard")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_trace: List[str] = Field(
        default_factory=list, description="Step-by-step calculation trace"
    )
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    # Optional metadata
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    source_id: Optional[str] = Field(None, description="Source identifier")
    period_start: Optional[datetime] = Field(None, description="Period start")
    period_end: Optional[datetime] = Field(None, description="Period end")


class Scope1CombustionInput(BaseModel):
    """Input model for Scope1CombustionAgent."""
    fuel_consumptions: List[FuelConsumption] = Field(
        ..., min_length=1, description="List of fuel consumption records"
    )
    gwp_source: str = Field(default="AR6", description="GWP source (AR4, AR5, AR6)")
    include_biogenic: bool = Field(
        default=False, description="Include biogenic CO2 in total"
    )
    organization_id: Optional[str] = Field(None, description="Organization ID")
    reporting_period: Optional[str] = Field(None, description="Reporting period")


class Scope1CombustionOutput(BaseModel):
    """Output model for Scope1CombustionAgent."""
    success: bool = Field(..., description="Calculation success status")
    calculation_results: List[CombustionCalculationResult] = Field(
        default_factory=list, description="Individual calculation results"
    )

    # Aggregated totals
    total_co2e_tonnes: float = Field(..., description="Total Scope 1 emissions (tCO2e)")
    total_co2_tonnes: float = Field(..., description="Total CO2 emissions (tonnes)")
    total_ch4_tonnes: float = Field(..., description="Total CH4 emissions (tonnes)")
    total_n2o_tonnes: float = Field(..., description="Total N2O emissions (tonnes)")
    total_biogenic_co2_tonnes: float = Field(
        default=0.0, description="Total biogenic CO2 (tonnes)"
    )

    # Breakdown by fuel type
    emissions_by_fuel: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by fuel type (tCO2e)"
    )

    # Breakdown by combustion type
    stationary_emissions_tco2e: float = Field(
        default=0.0, description="Stationary combustion emissions"
    )
    mobile_emissions_tco2e: float = Field(
        default=0.0, description="Mobile combustion emissions"
    )

    # Metadata
    gwp_source: str = Field(..., description="GWP source used")
    processing_time_ms: float = Field(..., description="Processing duration")
    provenance_hash: str = Field(..., description="Combined provenance hash")
    validation_status: str = Field(..., description="PASS or FAIL")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    # Organization context
    organization_id: Optional[str] = Field(None, description="Organization ID")
    reporting_period: Optional[str] = Field(None, description="Reporting period")


# =============================================================================
# SCOPE 1 COMBUSTION CALCULATOR AGENT
# =============================================================================

class Scope1CombustionAgent(DeterministicAgent):
    """
    GL-MRV-X-001: Scope 1 Combustion Calculator Agent

    Calculates Scope 1 GHG emissions from combustion sources following
    GHG Protocol Corporate Standard methodology. Supports both stationary
    and mobile combustion with complete audit trail.

    Zero-Hallucination Implementation:
        - All calculations use deterministic mathematical operations
        - Emission factors from EPA/IPCC authoritative sources
        - Complete provenance tracking with SHA-256 hashes
        - Full calculation trace for regulatory audit

    Supported Calculations:
        - Stationary combustion (boilers, furnaces, heaters)
        - Mobile combustion (fleet vehicles, equipment)
        - Multi-fuel facility aggregation
        - Biogenic emissions reporting

    Attributes:
        AGENT_ID: Unique agent identifier
        AGENT_NAME: Human-readable agent name
        VERSION: Agent version
        category: Agent category (CRITICAL)

    Example:
        >>> agent = Scope1CombustionAgent()
        >>> result = agent.execute({
        ...     "fuel_consumptions": [{
        ...         "fuel_type": "natural_gas",
        ...         "quantity": 1000,
        ...         "unit": "m3",
        ...         "combustion_type": "stationary"
        ...     }]
        ... })
        >>> print(result["total_co2e_tonnes"])
    """

    AGENT_ID = "GL-MRV-X-001"
    AGENT_NAME = "Scope 1 Combustion Calculator"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="Scope1CombustionAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="GHG Protocol Scope 1 combustion emissions calculator"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """
        Initialize Scope1CombustionAgent.

        Args:
            enable_audit_trail: Whether to capture full audit trail
        """
        super().__init__(enable_audit_trail=enable_audit_trail)
        self._calculation_counter = 0
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Scope 1 combustion emissions calculation.

        Args:
            inputs: Dictionary containing fuel consumption data

        Returns:
            Dictionary with calculation results and emissions totals

        Raises:
            ValueError: If input validation fails
        """
        start_time = DeterministicClock.now()
        calculation_trace: List[str] = []

        try:
            # Parse and validate input
            calculation_trace.append("Step 1: Validating input data")
            scope1_input = Scope1CombustionInput(**inputs)

            # Initialize results
            calculation_results: List[CombustionCalculationResult] = []
            total_co2e_kg = Decimal("0")
            total_co2_kg = Decimal("0")
            total_ch4_kg = Decimal("0")
            total_n2o_kg = Decimal("0")
            total_biogenic_kg = Decimal("0")
            emissions_by_fuel: Dict[str, Decimal] = {}
            stationary_emissions = Decimal("0")
            mobile_emissions = Decimal("0")

            calculation_trace.append(
                f"Step 2: Processing {len(scope1_input.fuel_consumptions)} "
                f"fuel consumption records"
            )

            # Process each fuel consumption record
            for fuel_record in scope1_input.fuel_consumptions:
                result = self._calculate_combustion_emissions(
                    fuel_record,
                    scope1_input.gwp_source,
                    scope1_input.include_biogenic
                )
                calculation_results.append(result)

                # Aggregate totals
                result_co2e = Decimal(str(result.total_co2e_kg))
                total_co2e_kg += result_co2e
                total_biogenic_kg += Decimal(str(result.biogenic_co2_kg))

                # Track by gas
                for gas_result in result.emissions_by_gas:
                    if gas_result.gas == EmissionGas.CO2:
                        total_co2_kg += Decimal(str(gas_result.emissions_kg))
                    elif gas_result.gas == EmissionGas.CH4:
                        total_ch4_kg += Decimal(str(gas_result.emissions_kg))
                    elif gas_result.gas == EmissionGas.N2O:
                        total_n2o_kg += Decimal(str(gas_result.emissions_kg))

                # Track by fuel type
                fuel_key = fuel_record.fuel_type.value
                if fuel_key not in emissions_by_fuel:
                    emissions_by_fuel[fuel_key] = Decimal("0")
                emissions_by_fuel[fuel_key] += result_co2e

                # Track by combustion type
                if fuel_record.combustion_type == CombustionType.STATIONARY:
                    stationary_emissions += result_co2e
                else:
                    mobile_emissions += result_co2e

            calculation_trace.append("Step 3: Aggregating results")

            # Convert to tonnes
            total_co2e_tonnes = total_co2e_kg / Decimal("1000")
            total_co2_tonnes = total_co2_kg / Decimal("1000")
            total_ch4_tonnes = total_ch4_kg / Decimal("1000")
            total_n2o_tonnes = total_n2o_kg / Decimal("1000")
            total_biogenic_tonnes = total_biogenic_kg / Decimal("1000")

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Compute provenance hash
            provenance_data = {
                "input": inputs,
                "total_co2e_tonnes": float(total_co2e_tonnes),
                "calculation_count": len(calculation_results)
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)

            calculation_trace.append(
                f"Step 4: Calculation complete. Total: {float(total_co2e_tonnes):.4f} tCO2e"
            )

            # Build output
            output = Scope1CombustionOutput(
                success=True,
                calculation_results=calculation_results,
                total_co2e_tonnes=float(total_co2e_tonnes.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                total_co2_tonnes=float(total_co2_tonnes.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                total_ch4_tonnes=float(total_ch4_tonnes.quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )),
                total_n2o_tonnes=float(total_n2o_tonnes.quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )),
                total_biogenic_co2_tonnes=float(total_biogenic_tonnes.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                emissions_by_fuel={
                    k: float(v / Decimal("1000")) for k, v in emissions_by_fuel.items()
                },
                stationary_emissions_tco2e=float(stationary_emissions / Decimal("1000")),
                mobile_emissions_tco2e=float(mobile_emissions / Decimal("1000")),
                gwp_source=scope1_input.gwp_source,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS",
                organization_id=scope1_input.organization_id,
                reporting_period=scope1_input.reporting_period
            )

            # Capture audit entry
            self._capture_audit_entry(
                operation="calculate_scope1_combustion",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Scope 1 combustion calculation failed: {str(e)}", exc_info=True)

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _calculate_combustion_emissions(
        self,
        fuel_record: FuelConsumption,
        gwp_source: str,
        include_biogenic: bool
    ) -> CombustionCalculationResult:
        """
        Calculate emissions for a single fuel consumption record.

        Args:
            fuel_record: Fuel consumption data
            gwp_source: GWP source standard
            include_biogenic: Whether to include biogenic in total

        Returns:
            CombustionCalculationResult with detailed emissions
        """
        calculation_trace: List[str] = []
        self._calculation_counter += 1
        calc_id = f"CALC-{self._calculation_counter:06d}"

        fuel_type = fuel_record.fuel_type
        quantity = Decimal(str(fuel_record.quantity))

        calculation_trace.append(
            f"Calculating emissions for {fuel_type.value}: "
            f"{fuel_record.quantity} {fuel_record.unit.value}"
        )

        # Step 1: Get heating value and convert to energy
        if fuel_record.custom_heating_value:
            heating_value = Decimal(str(fuel_record.custom_heating_value))
        elif fuel_type in FUEL_HEATING_VALUES:
            heating_value = FUEL_HEATING_VALUES[fuel_type]["value"]
        else:
            heating_value = Decimal("0.035")  # Default

        # Convert fuel quantity to energy (GJ)
        energy_gj = self._convert_to_energy_gj(
            fuel_type, quantity, fuel_record.unit, heating_value
        )
        calculation_trace.append(f"Energy content: {float(energy_gj):.4f} GJ")

        # Step 2: Get oxidation factor
        if fuel_record.custom_oxidation_factor:
            oxidation_factor = Decimal(str(fuel_record.custom_oxidation_factor))
        else:
            oxidation_factor = OXIDATION_FACTORS.get(fuel_type, Decimal("0.99"))

        calculation_trace.append(f"Oxidation factor: {float(oxidation_factor):.4f}")

        # Step 3: Get emission factors
        if fuel_record.combustion_type == CombustionType.STATIONARY:
            emission_factors = STATIONARY_EMISSION_FACTORS.get(fuel_type, {})
            ef_unit = "kg/GJ"
        else:
            emission_factors = MOBILE_EMISSION_FACTORS.get(fuel_type, {})
            ef_unit = "kg/unit"

        # Step 4: Get GWP values
        gwp_values = GWP_AR6  # Currently only AR6 supported

        # Step 5: Calculate emissions for each gas
        emission_results: List[EmissionResult] = []
        total_co2e_kg = Decimal("0")
        biogenic_co2_kg = Decimal("0")

        for gas in [EmissionGas.CO2, EmissionGas.CH4, EmissionGas.N2O]:
            ef = emission_factors.get(gas, Decimal("0"))
            gwp = gwp_values.get(gas, Decimal("1"))

            if fuel_record.combustion_type == CombustionType.STATIONARY:
                # Stationary: emissions = energy * EF * oxidation
                emissions_kg = energy_gj * ef * oxidation_factor
            else:
                # Mobile: emissions = quantity * EF
                emissions_kg = quantity * ef

            emissions_co2e_kg = emissions_kg * gwp
            emissions_tco2e = emissions_co2e_kg / Decimal("1000")

            # Track biogenic CO2 separately
            is_biogenic_fuel = fuel_type in [
                FuelType.WOOD, FuelType.BIOMASS,
                FuelType.LANDFILL_GAS, FuelType.BIOGAS
            ]
            if gas == EmissionGas.CO2 and is_biogenic_fuel:
                biogenic_co2_kg = emissions_kg
                if not include_biogenic:
                    emissions_kg = Decimal("0")
                    emissions_co2e_kg = Decimal("0")
            else:
                total_co2e_kg += emissions_co2e_kg

            emission_results.append(EmissionResult(
                gas=gas,
                emissions_kg=float(emissions_kg.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )),
                emissions_tco2e=float(emissions_tco2e.quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )),
                emission_factor=float(ef),
                emission_factor_unit=ef_unit,
                gwp_applied=float(gwp)
            ))

            if ef > 0:
                calculation_trace.append(
                    f"  {gas.value}: {float(ef)} {ef_unit} x GWP {float(gwp)} = "
                    f"{float(emissions_co2e_kg):.4f} kg CO2e"
                )

        # Compute provenance hash for this calculation
        provenance_data = {
            "fuel_type": fuel_type.value,
            "quantity": float(quantity),
            "unit": fuel_record.unit.value,
            "energy_gj": float(energy_gj),
            "total_co2e_kg": float(total_co2e_kg)
        }
        provenance_hash = self._compute_provenance_hash(provenance_data)

        calculation_trace.append(
            f"Total: {float(total_co2e_kg):.4f} kg CO2e "
            f"({float(total_co2e_kg / Decimal('1000')):.6f} tCO2e)"
        )

        return CombustionCalculationResult(
            calculation_id=calc_id,
            fuel_type=fuel_type,
            combustion_type=fuel_record.combustion_type,
            fuel_quantity=float(quantity),
            fuel_unit=fuel_record.unit,
            energy_gj=float(energy_gj.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )),
            emissions_by_gas=emission_results,
            total_co2e_kg=float(total_co2e_kg.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            total_co2e_tonnes=float((total_co2e_kg / Decimal("1000")).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )),
            biogenic_co2_kg=float(biogenic_co2_kg.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            heating_value_used=float(heating_value),
            oxidation_factor_used=float(oxidation_factor),
            gwp_source=gwp_source,
            provenance_hash=provenance_hash,
            calculation_trace=calculation_trace,
            facility_id=fuel_record.facility_id,
            source_id=fuel_record.source_id,
            period_start=fuel_record.period_start,
            period_end=fuel_record.period_end
        )

    def _convert_to_energy_gj(
        self,
        fuel_type: FuelType,
        quantity: Decimal,
        unit: UnitType,
        heating_value: Decimal
    ) -> Decimal:
        """
        Convert fuel quantity to energy in GJ.

        Args:
            fuel_type: Type of fuel
            quantity: Fuel quantity
            unit: Unit of measurement
            heating_value: Heating value in GJ/base unit

        Returns:
            Energy in GJ
        """
        # Unit conversion factors to base unit
        volume_to_liters = {
            UnitType.LITERS: Decimal("1"),
            UnitType.GALLONS: Decimal("3.78541"),
            UnitType.CUBIC_METERS: Decimal("1000"),
            UnitType.CUBIC_FEET: Decimal("28.3168"),
            UnitType.BARRELS: Decimal("158.987"),
        }

        mass_to_tonnes = {
            UnitType.KG: Decimal("0.001"),
            UnitType.TONNES: Decimal("1"),
            UnitType.LBS: Decimal("0.000453592"),
            UnitType.SHORT_TONS: Decimal("0.907185"),
        }

        energy_to_gj = {
            UnitType.KWH: Decimal("0.0036"),
            UnitType.MWH: Decimal("3.6"),
            UnitType.GJ: Decimal("1"),
            UnitType.MMBTU: Decimal("1.055056"),
            UnitType.THERMS: Decimal("0.105506"),
        }

        # If unit is already energy, convert directly
        if unit in energy_to_gj:
            return quantity * energy_to_gj[unit]

        # Determine base unit from heating value
        hv_info = FUEL_HEATING_VALUES.get(fuel_type, {"unit": "liter"})
        base_unit = hv_info.get("unit", "liter")

        # Convert quantity to base unit
        if base_unit == "liter" and unit in volume_to_liters:
            base_quantity = quantity * volume_to_liters[unit]
        elif base_unit == "m3" and unit in volume_to_liters:
            base_quantity = quantity * volume_to_liters[unit] / Decimal("1000")
        elif base_unit == "tonne" and unit in mass_to_tonnes:
            base_quantity = quantity * mass_to_tonnes[unit]
        else:
            # Assume quantity is already in correct base unit
            base_quantity = quantity

        # Calculate energy
        return base_quantity * heating_value

    def _compute_provenance_hash(self, data: Any) -> str:
        """
        Compute SHA-256 provenance hash.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash string
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def calculate_stationary(
        self,
        fuel_type: str,
        quantity: float,
        unit: str,
        facility_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate stationary combustion emissions.

        Args:
            fuel_type: Fuel type string
            quantity: Fuel quantity
            unit: Unit of measurement
            facility_id: Optional facility identifier

        Returns:
            Calculation result dictionary
        """
        return self.execute({
            "fuel_consumptions": [{
                "fuel_type": fuel_type,
                "quantity": quantity,
                "unit": unit,
                "combustion_type": "stationary",
                "facility_id": facility_id
            }]
        })

    def calculate_mobile(
        self,
        fuel_type: str,
        quantity: float,
        unit: str,
        source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate mobile combustion emissions.

        Args:
            fuel_type: Fuel type string
            quantity: Fuel quantity
            unit: Unit of measurement
            source_id: Optional source identifier

        Returns:
            Calculation result dictionary
        """
        return self.execute({
            "fuel_consumptions": [{
                "fuel_type": fuel_type,
                "quantity": quantity,
                "unit": unit,
                "combustion_type": "mobile",
                "source_id": source_id
            }]
        })

    def get_supported_fuels(self) -> List[str]:
        """Get list of supported fuel types."""
        return [ft.value for ft in FuelType]

    def get_emission_factor(
        self,
        fuel_type: str,
        combustion_type: str = "stationary"
    ) -> Dict[str, float]:
        """
        Get emission factors for a fuel type.

        Args:
            fuel_type: Fuel type string
            combustion_type: "stationary" or "mobile"

        Returns:
            Dictionary of emission factors by gas
        """
        try:
            ft = FuelType(fuel_type)
            if combustion_type == "stationary":
                factors = STATIONARY_EMISSION_FACTORS.get(ft, {})
            else:
                factors = MOBILE_EMISSION_FACTORS.get(ft, {})

            return {gas.value: float(ef) for gas, ef in factors.items()}
        except ValueError:
            return {}
