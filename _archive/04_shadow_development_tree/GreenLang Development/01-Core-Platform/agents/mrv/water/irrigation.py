# -*- coding: utf-8 -*-
"""
GL-MRV-WAT-004: Irrigation MRV Agent
====================================

MRV agent for agricultural irrigation emissions measurement.
Calculates GHG emissions from irrigation systems including:
- Pumping energy for groundwater and surface water
- Energy for pressurized systems (drip, sprinkler)
- Fertilizer application through irrigation (fertigation)
- Water losses and efficiency impacts

Methodologies:
    - GHG Protocol Agricultural Guidance
    - IPCC Guidelines for Agriculture
    - FAO water footprint methodology

Zero-Hallucination Guarantees:
    - All emissions calculated deterministically from activity data
    - NO LLM involvement in any emission calculations
    - All emission factors traceable to authoritative sources
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class IrrigationType(str, Enum):
    """Irrigation system types."""
    SURFACE_FLOOD = "surface_flood"
    FURROW = "furrow"
    SPRINKLER = "sprinkler"
    CENTER_PIVOT = "center_pivot"
    DRIP = "drip"
    SUBSURFACE_DRIP = "subsurface_drip"
    MICRO_SPRINKLER = "micro_sprinkler"


class WaterSourceType(str, Enum):
    """Irrigation water source."""
    GROUNDWATER = "groundwater"
    SURFACE_WATER = "surface_water"
    CANAL = "canal"
    RECYCLED = "recycled"
    RAINWATER = "rainwater"


class PumpPowerSource(str, Enum):
    """Pump power source."""
    GRID_ELECTRICITY = "grid_electricity"
    DIESEL = "diesel"
    SOLAR = "solar"
    NATURAL_GAS = "natural_gas"
    GRAVITY = "gravity"


class CropType(str, Enum):
    """Crop categories."""
    CEREALS = "cereals"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    LEGUMES = "legumes"
    OILSEEDS = "oilseeds"
    COTTON = "cotton"
    SUGARCANE = "sugarcane"
    RICE = "rice"
    OTHER = "other"


# Irrigation efficiency benchmarks (%)
IRRIGATION_EFFICIENCY = {
    IrrigationType.SURFACE_FLOOD: Decimal("45"),
    IrrigationType.FURROW: Decimal("55"),
    IrrigationType.SPRINKLER: Decimal("75"),
    IrrigationType.CENTER_PIVOT: Decimal("85"),
    IrrigationType.DRIP: Decimal("90"),
    IrrigationType.SUBSURFACE_DRIP: Decimal("95"),
    IrrigationType.MICRO_SPRINKLER: Decimal("85"),
}

# Energy intensity (kWh per 1000 m3 pumped per meter lift)
PUMPING_ENERGY_FACTOR = Decimal("2.72")  # kWh per 1000m3 per m lift

# Default emission factors
DEFAULT_EMISSION_FACTORS = {
    "electricity_kwh": Decimal("0.417"),     # kgCO2e/kWh
    "diesel_l": Decimal("2.68"),             # kgCO2e/L
    "natural_gas_m3": Decimal("2.0"),        # kgCO2e/m3
    "solar_kwh": Decimal("0.041"),           # kgCO2e/kWh (lifecycle)
    # Fertilizers (fertigation)
    "nitrogen_kg": Decimal("5.88"),          # kgCO2e/kg N (production + N2O)
    "phosphorus_kg": Decimal("1.0"),         # kgCO2e/kg P2O5
    "potassium_kg": Decimal("0.5"),          # kgCO2e/kg K2O
}


# =============================================================================
# PYDANTIC MODELS - INPUT
# =============================================================================

class IrrigationSystemRecord(BaseModel):
    """Record of irrigation system operations."""
    record_id: str = Field(..., description="Unique record identifier")
    farm_id: str = Field(..., description="Farm/field identifier")
    farm_name: Optional[str] = Field(None, description="Farm name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")

    # Field characteristics
    irrigated_area_ha: float = Field(..., ge=0, description="Irrigated area (hectares)")
    crop_type: CropType = Field(default=CropType.OTHER, description="Primary crop")

    # Irrigation system
    irrigation_type: IrrigationType = Field(..., description="Irrigation system type")
    water_source: WaterSourceType = Field(..., description="Water source")
    pump_power_source: PumpPowerSource = Field(..., description="Pump power source")

    # Water volumes
    water_applied_m3: float = Field(..., ge=0, description="Total water applied (m3)")
    water_sourced_m3: Optional[float] = Field(None, ge=0, description="Water sourced/pumped (m3)")

    # Pumping parameters
    average_lift_m: float = Field(default=0, ge=0, description="Average pumping lift (m)")
    average_pressure_bar: float = Field(default=0, ge=0, description="System pressure (bar)")
    pump_efficiency_percent: float = Field(default=60, ge=0, le=100, description="Pump efficiency")

    # Energy consumption (if metered)
    electricity_kwh: Optional[float] = Field(None, ge=0, description="Electricity consumed (kWh)")
    diesel_l: Optional[float] = Field(None, ge=0, description="Diesel consumed (L)")
    natural_gas_m3: Optional[float] = Field(None, ge=0, description="Natural gas (m3)")

    # Fertigation
    nitrogen_applied_kg: float = Field(default=0, ge=0, description="N applied via fertigation (kg)")
    phosphorus_applied_kg: float = Field(default=0, ge=0, description="P2O5 applied (kg)")
    potassium_applied_kg: float = Field(default=0, ge=0, description="K2O applied (kg)")

    # Custom factors
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        None, description="Custom emission factors"
    )
    custom_efficiency: Optional[float] = Field(
        None, ge=0, le=100, description="Custom irrigation efficiency %"
    )


class IrrigationInput(BaseModel):
    """Input data for Irrigation MRV Agent."""
    irrigation_records: List[IrrigationSystemRecord] = Field(
        ..., description="Irrigation system records"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    include_scope3: bool = Field(default=True, description="Include Scope 3 (fertilizers)")
    grid_emission_factor: Optional[float] = Field(
        None, description="Custom grid emission factor (kgCO2e/kWh)"
    )


# =============================================================================
# PYDANTIC MODELS - OUTPUT
# =============================================================================

class IrrigationEmissionBreakdown(BaseModel):
    """Emission breakdown for irrigation system."""
    pumping_emissions_kgco2e: float = Field(..., description="Pumping energy emissions")
    fertigation_emissions_kgco2e: float = Field(..., description="Fertigation emissions")
    total_emissions_kgco2e: float = Field(..., description="Total emissions")


class IrrigationSystemResult(BaseModel):
    """Emissions result for an irrigation system."""
    record_id: str = Field(..., description="Source record ID")
    farm_id: str = Field(..., description="Farm identifier")

    # Area and volumes
    irrigated_area_ha: float = Field(..., description="Irrigated area")
    water_applied_m3: float = Field(..., description="Water applied")
    water_per_ha_m3: float = Field(..., description="Water per hectare")

    # Efficiency
    irrigation_efficiency_percent: float = Field(..., description="Irrigation efficiency")
    effective_water_m3: float = Field(..., description="Effective water reaching crop")

    # Energy
    estimated_energy_kwh: float = Field(..., description="Estimated/actual energy")
    energy_per_m3_kwh: float = Field(..., description="Energy per m3")

    # Scope 1: On-site fuel (diesel, natural gas)
    scope1_emissions_kgco2e: float = Field(..., description="Scope 1 emissions")

    # Scope 2: Purchased electricity
    scope2_emissions_kgco2e: float = Field(..., description="Scope 2 emissions")

    # Scope 3: Fertilizer production
    scope3_emissions_kgco2e: float = Field(..., description="Scope 3 emissions")

    # Totals
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    emission_breakdown: IrrigationEmissionBreakdown = Field(..., description="Breakdown")

    # Intensities
    emissions_per_ha_kgco2e: float = Field(..., description="Emissions per hectare")
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions per m3 water")

    # Provenance
    calculation_method: str = Field(..., description="Calculation method")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class IrrigationOutput(BaseModel):
    """Output from Irrigation MRV Agent."""
    # Summary
    reporting_year: int = Field(..., description="Reporting year")
    total_irrigated_area_ha: float = Field(..., description="Total irrigated area")
    total_water_applied_m3: float = Field(..., description="Total water applied")

    # Emissions
    scope1_total_kgco2e: float = Field(..., description="Total Scope 1")
    scope2_total_kgco2e: float = Field(..., description="Total Scope 2")
    scope3_total_kgco2e: float = Field(..., description="Total Scope 3")
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    total_emissions_tco2e: float = Field(..., description="Total (tCO2e)")

    # Intensities
    average_emissions_per_ha_kgco2e: float = Field(..., description="Average per hectare")
    average_emissions_per_m3_kgco2e: float = Field(..., description="Average per m3")
    average_water_per_ha_m3: float = Field(..., description="Average water per ha")

    # Detailed results
    system_results: List[IrrigationSystemResult] = Field(..., description="Per-system results")

    # Provenance
    provenance_hash: str = Field(..., description="Overall provenance hash")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    methodology_version: str = Field(..., description="Methodology version")
    processing_time_ms: float = Field(..., description="Processing time")


# =============================================================================
# IRRIGATION MRV AGENT IMPLEMENTATION
# =============================================================================

class IrrigationMRVAgent(BaseAgent):
    """
    GL-MRV-WAT-004: Irrigation MRV Agent

    Calculates GHG emissions from agricultural irrigation systems.
    Supports various irrigation types from flood to precision drip.

    Zero-Hallucination Guarantees:
        - All calculations are deterministic mathematical operations
        - NO LLM involvement in any emission calculations
        - All emission factors traceable to authoritative sources
        - Complete provenance hash for every calculation

    Usage:
        agent = IrrigationMRVAgent()
        result = agent.run({
            "irrigation_records": [...],
            "reporting_year": 2024
        })
    """

    AGENT_ID = "GL-MRV-WAT-004"
    AGENT_NAME = "Irrigation MRV Agent"
    VERSION = "1.0.0"
    METHODOLOGY_VERSION = "GHG-Protocol-Ag-v2024.1"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Irrigation MRV Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="MRV agent for irrigation system emissions",
                version=self.VERSION,
                parameters={
                    "default_grid_factor": 0.417,
                }
            )
        super().__init__(config)

        self._emission_factors = DEFAULT_EMISSION_FACTORS.copy()
        self._calculations_performed = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute irrigation emissions calculation."""
        start_time = time.time()

        try:
            irr_input = IrrigationInput(**input_data)

            # Apply custom grid factor
            if irr_input.grid_emission_factor is not None:
                self._emission_factors["electricity_kwh"] = Decimal(
                    str(irr_input.grid_emission_factor)
                )

            # Process each system
            system_results = []
            for record in irr_input.irrigation_records:
                result = self._calculate_system_emissions(record, irr_input.include_scope3)
                system_results.append(result)

            # Aggregate totals
            scope1_total = sum(r.scope1_emissions_kgco2e for r in system_results)
            scope2_total = sum(r.scope2_emissions_kgco2e for r in system_results)
            scope3_total = sum(r.scope3_emissions_kgco2e for r in system_results)
            total_emissions = scope1_total + scope2_total + scope3_total

            total_area = sum(r.irrigated_area_ha for r in system_results)
            total_water = sum(r.water_applied_m3 for r in system_results)

            # Averages
            avg_per_ha = total_emissions / total_area if total_area > 0 else 0.0
            avg_per_m3 = total_emissions / total_water if total_water > 0 else 0.0
            avg_water_per_ha = total_water / total_area if total_area > 0 else 0.0

            # Provenance
            provenance_hash = self._compute_provenance_hash(irr_input, system_results)

            processing_time = (time.time() - start_time) * 1000

            output = IrrigationOutput(
                reporting_year=irr_input.reporting_year,
                total_irrigated_area_ha=round(total_area, 2),
                total_water_applied_m3=round(total_water, 2),
                scope1_total_kgco2e=round(scope1_total, 2),
                scope2_total_kgco2e=round(scope2_total, 2),
                scope3_total_kgco2e=round(scope3_total, 2),
                total_emissions_kgco2e=round(total_emissions, 2),
                total_emissions_tco2e=round(total_emissions / 1000, 4),
                average_emissions_per_ha_kgco2e=round(avg_per_ha, 2),
                average_emissions_per_m3_kgco2e=round(avg_per_m3, 6),
                average_water_per_ha_m3=round(avg_water_per_ha, 2),
                system_results=system_results,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                methodology_version=self.METHODOLOGY_VERSION,
                processing_time_ms=processing_time,
            )

            self._calculations_performed += 1

            self.logger.info(
                f"Calculated irrigation emissions: {total_emissions:.2f} kgCO2e "
                f"({total_area:.0f} ha, {total_water:.0f} m3)"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "methodology": self.METHODOLOGY_VERSION,
                    "systems_processed": len(system_results),
                }
            )

        except Exception as e:
            self.logger.error(f"Irrigation MRV calculation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID}
            )

    def _calculate_system_emissions(
        self,
        record: IrrigationSystemRecord,
        include_scope3: bool
    ) -> IrrigationSystemResult:
        """
        Calculate emissions for an irrigation system.

        ZERO-HALLUCINATION: All calculations are deterministic.
        """
        factors = self._get_emission_factors(record.custom_emission_factors)

        # Get irrigation efficiency
        efficiency = Decimal(str(record.custom_efficiency)) if record.custom_efficiency else (
            IRRIGATION_EFFICIENCY.get(record.irrigation_type, Decimal("75"))
        )
        effective_water = float(Decimal(str(record.water_applied_m3)) * efficiency / Decimal("100"))

        # Water per hectare
        water_per_ha = record.water_applied_m3 / record.irrigated_area_ha if record.irrigated_area_ha > 0 else 0.0

        # Calculate or use metered energy
        if record.electricity_kwh is not None:
            electricity_kwh = record.electricity_kwh
            calculation_method = "metered"
        else:
            # Estimate energy based on pumping lift and pressure
            # E = (Q * H * rho * g) / (3.6e6 * eta)
            # Simplified: E (kWh) = V (m3) * H (m) * 0.00272 / pump_efficiency
            total_head = record.average_lift_m + (record.average_pressure_bar * 10.2)  # Convert bar to m head
            water_sourced = record.water_sourced_m3 or record.water_applied_m3
            pump_eff = record.pump_efficiency_percent / 100

            electricity_kwh = float(
                Decimal(str(water_sourced)) *
                Decimal(str(total_head)) *
                PUMPING_ENERGY_FACTOR /
                Decimal("1000") /
                Decimal(str(max(0.3, pump_eff)))
            )
            calculation_method = "estimated"

        energy_per_m3 = electricity_kwh / record.water_applied_m3 if record.water_applied_m3 > 0 else 0.0

        # Scope 1: On-site fuel
        scope1_diesel = 0.0
        scope1_gas = 0.0
        if record.diesel_l is not None:
            scope1_diesel = float(Decimal(str(record.diesel_l)) * factors["diesel_l"])
        if record.natural_gas_m3 is not None:
            scope1_gas = float(Decimal(str(record.natural_gas_m3)) * factors["natural_gas_m3"])
        scope1 = scope1_diesel + scope1_gas

        # Scope 2: Purchased electricity
        scope2 = 0.0
        if record.pump_power_source == PumpPowerSource.GRID_ELECTRICITY:
            scope2 = float(Decimal(str(electricity_kwh)) * factors["electricity_kwh"])
        elif record.pump_power_source == PumpPowerSource.SOLAR:
            scope2 = float(Decimal(str(electricity_kwh)) * factors["solar_kwh"])

        # Scope 3: Fertilizer production
        scope3 = 0.0
        if include_scope3:
            scope3_n = float(Decimal(str(record.nitrogen_applied_kg)) * factors["nitrogen_kg"])
            scope3_p = float(Decimal(str(record.phosphorus_applied_kg)) * factors["phosphorus_kg"])
            scope3_k = float(Decimal(str(record.potassium_applied_kg)) * factors["potassium_kg"])
            scope3 = scope3_n + scope3_p + scope3_k

        total_emissions = scope1 + scope2 + scope3

        # Breakdown
        pumping_emissions = scope1 + scope2
        breakdown = IrrigationEmissionBreakdown(
            pumping_emissions_kgco2e=round(pumping_emissions, 2),
            fertigation_emissions_kgco2e=round(scope3, 2),
            total_emissions_kgco2e=round(total_emissions, 2),
        )

        # Intensities
        emissions_per_ha = total_emissions / record.irrigated_area_ha if record.irrigated_area_ha > 0 else 0.0
        emissions_per_m3 = total_emissions / record.water_applied_m3 if record.water_applied_m3 > 0 else 0.0

        # Provenance
        provenance_hash = self._compute_record_provenance(record, total_emissions)

        return IrrigationSystemResult(
            record_id=record.record_id,
            farm_id=record.farm_id,
            irrigated_area_ha=record.irrigated_area_ha,
            water_applied_m3=record.water_applied_m3,
            water_per_ha_m3=round(water_per_ha, 2),
            irrigation_efficiency_percent=float(efficiency),
            effective_water_m3=round(effective_water, 2),
            estimated_energy_kwh=round(electricity_kwh, 2),
            energy_per_m3_kwh=round(energy_per_m3, 4),
            scope1_emissions_kgco2e=round(scope1, 2),
            scope2_emissions_kgco2e=round(scope2, 2),
            scope3_emissions_kgco2e=round(scope3, 2),
            total_emissions_kgco2e=round(total_emissions, 2),
            emission_breakdown=breakdown,
            emissions_per_ha_kgco2e=round(emissions_per_ha, 2),
            emissions_per_m3_kgco2e=round(emissions_per_m3, 6),
            calculation_method=calculation_method,
            provenance_hash=provenance_hash,
        )

    def _get_emission_factors(
        self,
        custom_factors: Optional[Dict[str, float]]
    ) -> Dict[str, Decimal]:
        """Get emission factors with custom overrides."""
        factors = self._emission_factors.copy()
        if custom_factors:
            for key, value in custom_factors.items():
                if key in factors:
                    factors[key] = Decimal(str(value))
        return factors

    def _compute_record_provenance(
        self,
        record: IrrigationSystemRecord,
        total_emissions: float
    ) -> str:
        """Compute SHA-256 provenance hash for a record."""
        provenance_data = {
            "record_id": record.record_id,
            "farm_id": record.farm_id,
            "irrigated_area_ha": record.irrigated_area_ha,
            "water_applied_m3": record.water_applied_m3,
            "irrigation_type": record.irrigation_type.value,
            "total_emissions": round(total_emissions, 2),
            "methodology": self.METHODOLOGY_VERSION,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

    def _compute_provenance_hash(
        self,
        input_data: IrrigationInput,
        results: List[IrrigationSystemResult]
    ) -> str:
        """Compute overall provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "methodology": self.METHODOLOGY_VERSION,
            "reporting_year": input_data.reporting_year,
            "systems_count": len(results),
            "system_hashes": [r.provenance_hash for r in results],
            "total_emissions": sum(r.total_emissions_kgco2e for r in results),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()
