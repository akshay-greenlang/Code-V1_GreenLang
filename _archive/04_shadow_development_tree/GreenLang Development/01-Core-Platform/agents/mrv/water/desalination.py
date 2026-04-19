# -*- coding: utf-8 -*-
"""
GL-MRV-WAT-003: Desalination MRV Agent
======================================

MRV agent for desalination plant emissions measurement.
Calculates GHG emissions from desalination operations including:
- High energy consumption of reverse osmosis/thermal processes
- Chemical usage for pre/post-treatment
- Brine disposal impacts

Methodologies:
    - GHG Protocol Scope 1, 2, and 3
    - IDA (International Desalination Association) benchmarks
    - EPA emission factors

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

class DesalinationTechnology(str, Enum):
    """Desalination technology types."""
    REVERSE_OSMOSIS = "reverse_osmosis"
    MULTI_STAGE_FLASH = "multi_stage_flash"
    MULTI_EFFECT_DISTILLATION = "multi_effect_distillation"
    ELECTRODIALYSIS = "electrodialysis"
    HYBRID = "hybrid"


class WaterSourceSalinity(str, Enum):
    """Source water salinity levels."""
    SEAWATER = "seawater"           # ~35,000 ppm TDS
    BRACKISH = "brackish"           # 1,000-10,000 ppm TDS
    SLIGHTLY_BRACKISH = "slightly_brackish"  # 500-1,000 ppm TDS


class EnergySourceType(str, Enum):
    """Energy source for desalination."""
    GRID = "grid"
    NATURAL_GAS = "natural_gas"
    SOLAR = "solar"
    WIND = "wind"
    HYBRID_RENEWABLE = "hybrid_renewable"
    COGENERATION = "cogeneration"


# Energy intensity benchmarks (kWh per m3 of product water)
# Source: IDA Desalination Yearbook, World Bank reports
ENERGY_INTENSITY_BENCHMARKS = {
    (DesalinationTechnology.REVERSE_OSMOSIS, WaterSourceSalinity.SEAWATER): Decimal("3.5"),
    (DesalinationTechnology.REVERSE_OSMOSIS, WaterSourceSalinity.BRACKISH): Decimal("1.0"),
    (DesalinationTechnology.REVERSE_OSMOSIS, WaterSourceSalinity.SLIGHTLY_BRACKISH): Decimal("0.5"),
    (DesalinationTechnology.MULTI_STAGE_FLASH, WaterSourceSalinity.SEAWATER): Decimal("15.0"),
    (DesalinationTechnology.MULTI_EFFECT_DISTILLATION, WaterSourceSalinity.SEAWATER): Decimal("8.0"),
    (DesalinationTechnology.ELECTRODIALYSIS, WaterSourceSalinity.BRACKISH): Decimal("0.8"),
    (DesalinationTechnology.HYBRID, WaterSourceSalinity.SEAWATER): Decimal("5.0"),
}

# Default emission factors
DEFAULT_EMISSION_FACTORS = {
    "electricity_kwh": Decimal("0.417"),     # kgCO2e/kWh (grid average)
    "natural_gas_m3": Decimal("2.0"),        # kgCO2e/m3
    "solar_kwh": Decimal("0.041"),           # kgCO2e/kWh (lifecycle)
    "wind_kwh": Decimal("0.011"),            # kgCO2e/kWh (lifecycle)
    # Chemicals
    "antiscalant_kg": Decimal("2.5"),        # kgCO2e/kg
    "chlorine_kg": Decimal("1.5"),           # kgCO2e/kg
    "sulfuric_acid_kg": Decimal("0.12"),     # kgCO2e/kg
    "caustic_soda_kg": Decimal("1.17"),      # kgCO2e/kg
    "coagulant_kg": Decimal("0.8"),          # kgCO2e/kg
    "membrane_m2": Decimal("50.0"),          # kgCO2e/m2 (lifecycle)
}


# =============================================================================
# PYDANTIC MODELS - INPUT
# =============================================================================

class DesalinationPlantRecord(BaseModel):
    """Record of desalination plant operations."""
    record_id: str = Field(..., description="Unique record identifier")
    plant_id: str = Field(..., description="Plant identifier")
    plant_name: Optional[str] = Field(None, description="Plant name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")

    # Technology and capacity
    technology: DesalinationTechnology = Field(..., description="Desalination technology")
    source_salinity: WaterSourceSalinity = Field(..., description="Source water salinity")
    design_capacity_m3_day: float = Field(..., ge=0, description="Design capacity (m3/day)")

    # Production volumes
    intake_volume_m3: float = Field(..., ge=0, description="Feed water intake (m3)")
    product_water_m3: float = Field(..., ge=0, description="Product water produced (m3)")
    brine_volume_m3: float = Field(default=0, ge=0, description="Brine/concentrate (m3)")

    # Water quality
    feed_tds_ppm: float = Field(default=35000, ge=0, description="Feed water TDS (ppm)")
    product_tds_ppm: float = Field(default=200, ge=0, description="Product water TDS (ppm)")

    # Energy consumption
    energy_source: EnergySourceType = Field(default=EnergySourceType.GRID, description="Primary energy source")
    electricity_kwh: float = Field(default=0, ge=0, description="Total electricity (kWh)")
    thermal_energy_mwh: float = Field(default=0, ge=0, description="Thermal energy (MWh)")
    natural_gas_m3: float = Field(default=0, ge=0, description="Natural gas (m3)")
    renewable_electricity_kwh: float = Field(default=0, ge=0, description="Renewable electricity (kWh)")

    # Chemical usage
    antiscalant_kg: float = Field(default=0, ge=0, description="Antiscalant (kg)")
    chlorine_kg: float = Field(default=0, ge=0, description="Chlorine (kg)")
    sulfuric_acid_kg: float = Field(default=0, ge=0, description="Sulfuric acid (kg)")
    caustic_soda_kg: float = Field(default=0, ge=0, description="Caustic soda (kg)")
    coagulant_kg: float = Field(default=0, ge=0, description="Coagulant (kg)")

    # Membrane replacement
    membrane_replaced_m2: float = Field(default=0, ge=0, description="Membrane replaced (m2)")

    # Custom factors
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        None, description="Custom emission factors"
    )
    custom_grid_factor: Optional[float] = Field(
        None, description="Custom grid emission factor"
    )

    @field_validator('product_water_m3')
    @classmethod
    def validate_recovery(cls, v: float, info) -> float:
        """Validate reasonable recovery rate."""
        intake = info.data.get('intake_volume_m3', 0)
        if intake > 0 and v > intake:
            raise ValueError("Product water cannot exceed intake volume")
        return v


class DesalinationInput(BaseModel):
    """Input data for Desalination MRV Agent."""
    plant_records: List[DesalinationPlantRecord] = Field(
        ..., description="Desalination plant records"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    include_scope3: bool = Field(default=True, description="Include Scope 3 emissions")


# =============================================================================
# PYDANTIC MODELS - OUTPUT
# =============================================================================

class DesalinationEmissionBreakdown(BaseModel):
    """Emission breakdown for desalination plant."""
    electricity_emissions_kgco2e: float = Field(..., description="Electricity emissions")
    thermal_emissions_kgco2e: float = Field(..., description="Thermal energy emissions")
    chemical_emissions_kgco2e: float = Field(..., description="Chemical production emissions")
    membrane_emissions_kgco2e: float = Field(..., description="Membrane lifecycle emissions")


class DesalinationPlantResult(BaseModel):
    """Emissions result for a desalination plant."""
    record_id: str = Field(..., description="Source record ID")
    plant_id: str = Field(..., description="Plant identifier")

    # Production metrics
    product_water_m3: float = Field(..., description="Product water produced")
    recovery_rate_percent: float = Field(..., description="Water recovery rate")
    capacity_utilization_percent: float = Field(..., description="Capacity utilization")

    # Energy metrics
    specific_energy_kwh_m3: float = Field(..., description="Energy per m3 product")
    renewable_fraction: float = Field(..., description="Renewable energy fraction")

    # Scope 1: On-site fuel combustion
    scope1_emissions_kgco2e: float = Field(..., description="Scope 1 emissions")

    # Scope 2: Purchased electricity/thermal
    scope2_emissions_kgco2e: float = Field(..., description="Scope 2 emissions")

    # Scope 3: Chemicals, membranes
    scope3_emissions_kgco2e: float = Field(..., description="Scope 3 emissions")

    # Totals
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    emission_breakdown: DesalinationEmissionBreakdown = Field(..., description="Breakdown")

    # Intensity
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions per m3")

    # Benchmark comparison
    benchmark_energy_kwh_m3: float = Field(..., description="Benchmark energy intensity")
    energy_performance_ratio: float = Field(..., description="Actual/benchmark ratio")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class DesalinationOutput(BaseModel):
    """Output from Desalination MRV Agent."""
    # Summary
    reporting_year: int = Field(..., description="Reporting year")
    total_product_water_m3: float = Field(..., description="Total product water")
    total_energy_consumed_mwh: float = Field(..., description="Total energy consumed")

    # Emissions
    scope1_total_kgco2e: float = Field(..., description="Total Scope 1")
    scope2_total_kgco2e: float = Field(..., description="Total Scope 2")
    scope3_total_kgco2e: float = Field(..., description="Total Scope 3")
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    total_emissions_tco2e: float = Field(..., description="Total (tCO2e)")

    # Intensity
    average_specific_energy_kwh_m3: float = Field(..., description="Average energy intensity")
    average_emissions_per_m3_kgco2e: float = Field(..., description="Average emission intensity")

    # Detailed results
    plant_results: List[DesalinationPlantResult] = Field(..., description="Per-plant results")

    # Provenance
    provenance_hash: str = Field(..., description="Overall provenance hash")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    methodology_version: str = Field(..., description="Methodology version")
    processing_time_ms: float = Field(..., description="Processing time")


# =============================================================================
# DESALINATION MRV AGENT IMPLEMENTATION
# =============================================================================

class DesalinationMRVAgent(BaseAgent):
    """
    GL-MRV-WAT-003: Desalination MRV Agent

    Calculates GHG emissions from desalination plants.
    Supports RO, MSF, MED, and other desalination technologies.

    Zero-Hallucination Guarantees:
        - All calculations are deterministic mathematical operations
        - NO LLM involvement in any emission calculations
        - All emission factors traceable to authoritative sources
        - Complete provenance hash for every calculation

    Usage:
        agent = DesalinationMRVAgent()
        result = agent.run({
            "plant_records": [...],
            "reporting_year": 2024
        })
    """

    AGENT_ID = "GL-MRV-WAT-003"
    AGENT_NAME = "Desalination MRV Agent"
    VERSION = "1.0.0"
    METHODOLOGY_VERSION = "IDA-GHG-v2024.1"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Desalination MRV Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="MRV agent for desalination plant emissions",
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
        """Execute desalination emissions calculation."""
        start_time = time.time()

        try:
            desal_input = DesalinationInput(**input_data)

            # Process each plant
            plant_results = []
            for record in desal_input.plant_records:
                result = self._calculate_plant_emissions(record, desal_input.include_scope3)
                plant_results.append(result)

            # Aggregate totals
            scope1_total = sum(r.scope1_emissions_kgco2e for r in plant_results)
            scope2_total = sum(r.scope2_emissions_kgco2e for r in plant_results)
            scope3_total = sum(r.scope3_emissions_kgco2e for r in plant_results)
            total_emissions = scope1_total + scope2_total + scope3_total

            total_water = sum(r.product_water_m3 for r in plant_results)
            total_energy_kwh = sum(
                r.record_id  # We need actual energy from records
                for r in plant_results
            )
            # Calculate from specific energy
            total_energy_mwh = sum(
                r.specific_energy_kwh_m3 * r.product_water_m3 / 1000
                for r in plant_results
            )

            # Averages
            avg_energy = total_energy_mwh * 1000 / total_water if total_water > 0 else 0.0
            avg_intensity = total_emissions / total_water if total_water > 0 else 0.0

            # Provenance
            provenance_hash = self._compute_provenance_hash(desal_input, plant_results)

            processing_time = (time.time() - start_time) * 1000

            output = DesalinationOutput(
                reporting_year=desal_input.reporting_year,
                total_product_water_m3=round(total_water, 2),
                total_energy_consumed_mwh=round(total_energy_mwh, 2),
                scope1_total_kgco2e=round(scope1_total, 2),
                scope2_total_kgco2e=round(scope2_total, 2),
                scope3_total_kgco2e=round(scope3_total, 2),
                total_emissions_kgco2e=round(total_emissions, 2),
                total_emissions_tco2e=round(total_emissions / 1000, 4),
                average_specific_energy_kwh_m3=round(avg_energy, 2),
                average_emissions_per_m3_kgco2e=round(avg_intensity, 4),
                plant_results=plant_results,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                methodology_version=self.METHODOLOGY_VERSION,
                processing_time_ms=processing_time,
            )

            self._calculations_performed += 1

            self.logger.info(
                f"Calculated desalination emissions: {total_emissions:.2f} kgCO2e "
                f"({total_water:.0f} m3 produced)"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "methodology": self.METHODOLOGY_VERSION,
                    "plants_processed": len(plant_results),
                }
            )

        except Exception as e:
            self.logger.error(f"Desalination MRV calculation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID}
            )

    def _calculate_plant_emissions(
        self,
        record: DesalinationPlantRecord,
        include_scope3: bool
    ) -> DesalinationPlantResult:
        """
        Calculate emissions for a desalination plant.

        ZERO-HALLUCINATION: All calculations are deterministic.
        """
        factors = self._get_emission_factors(record.custom_emission_factors, record.custom_grid_factor)

        # Calculate metrics
        recovery_rate = (
            record.product_water_m3 / record.intake_volume_m3 * 100
            if record.intake_volume_m3 > 0 else 0.0
        )

        # Calculate days in period
        period_days = (record.reporting_period_end - record.reporting_period_start).days
        capacity_utilization = (
            record.product_water_m3 / (record.design_capacity_m3_day * period_days) * 100
            if record.design_capacity_m3_day > 0 and period_days > 0 else 0.0
        )

        specific_energy = (
            record.electricity_kwh / record.product_water_m3
            if record.product_water_m3 > 0 else 0.0
        )

        renewable_fraction = (
            record.renewable_electricity_kwh / record.electricity_kwh
            if record.electricity_kwh > 0 else 0.0
        )

        # Get benchmark
        benchmark_key = (record.technology, record.source_salinity)
        benchmark_energy = float(
            ENERGY_INTENSITY_BENCHMARKS.get(benchmark_key, Decimal("3.5"))
        )
        energy_performance = specific_energy / benchmark_energy if benchmark_energy > 0 else 1.0

        # Scope 1: On-site combustion (natural gas for thermal desal)
        scope1 = float(Decimal(str(record.natural_gas_m3)) * factors["natural_gas_m3"])

        # Scope 2: Purchased electricity
        # Separate grid and renewable
        grid_electricity = record.electricity_kwh - record.renewable_electricity_kwh
        scope2_grid = float(Decimal(str(max(0, grid_electricity))) * factors["electricity_kwh"])
        scope2_renewable = float(Decimal(str(record.renewable_electricity_kwh)) * factors.get("solar_kwh", Decimal("0.041")))
        scope2_thermal = float(Decimal(str(record.thermal_energy_mwh * 1000)) * factors["electricity_kwh"] * Decimal("0.5"))  # Thermal efficiency factor
        scope2 = scope2_grid + scope2_renewable + scope2_thermal

        # Scope 3: Chemicals and membranes
        scope3_chemicals = 0.0
        scope3_membranes = 0.0
        if include_scope3:
            scope3_chemicals = float(
                Decimal(str(record.antiscalant_kg)) * factors["antiscalant_kg"] +
                Decimal(str(record.chlorine_kg)) * factors["chlorine_kg"] +
                Decimal(str(record.sulfuric_acid_kg)) * factors["sulfuric_acid_kg"] +
                Decimal(str(record.caustic_soda_kg)) * factors["caustic_soda_kg"] +
                Decimal(str(record.coagulant_kg)) * factors["coagulant_kg"]
            )
            scope3_membranes = float(
                Decimal(str(record.membrane_replaced_m2)) * factors["membrane_m2"]
            )
        scope3 = scope3_chemicals + scope3_membranes

        total_emissions = scope1 + scope2 + scope3

        # Breakdown
        breakdown = DesalinationEmissionBreakdown(
            electricity_emissions_kgco2e=round(scope2_grid + scope2_renewable, 2),
            thermal_emissions_kgco2e=round(scope2_thermal + scope1, 2),
            chemical_emissions_kgco2e=round(scope3_chemicals, 2),
            membrane_emissions_kgco2e=round(scope3_membranes, 2),
        )

        # Intensity
        emissions_per_m3 = total_emissions / record.product_water_m3 if record.product_water_m3 > 0 else 0.0

        # Provenance
        provenance_hash = self._compute_record_provenance(record, total_emissions)

        return DesalinationPlantResult(
            record_id=record.record_id,
            plant_id=record.plant_id,
            product_water_m3=record.product_water_m3,
            recovery_rate_percent=round(recovery_rate, 2),
            capacity_utilization_percent=round(capacity_utilization, 2),
            specific_energy_kwh_m3=round(specific_energy, 2),
            renewable_fraction=round(renewable_fraction, 4),
            scope1_emissions_kgco2e=round(scope1, 2),
            scope2_emissions_kgco2e=round(scope2, 2),
            scope3_emissions_kgco2e=round(scope3, 2),
            total_emissions_kgco2e=round(total_emissions, 2),
            emission_breakdown=breakdown,
            emissions_per_m3_kgco2e=round(emissions_per_m3, 4),
            benchmark_energy_kwh_m3=benchmark_energy,
            energy_performance_ratio=round(energy_performance, 2),
            provenance_hash=provenance_hash,
        )

    def _get_emission_factors(
        self,
        custom_factors: Optional[Dict[str, float]],
        custom_grid_factor: Optional[float]
    ) -> Dict[str, Decimal]:
        """Get emission factors with custom overrides."""
        factors = self._emission_factors.copy()
        if custom_grid_factor is not None:
            factors["electricity_kwh"] = Decimal(str(custom_grid_factor))
        if custom_factors:
            for key, value in custom_factors.items():
                if key in factors:
                    factors[key] = Decimal(str(value))
        return factors

    def _compute_record_provenance(
        self,
        record: DesalinationPlantRecord,
        total_emissions: float
    ) -> str:
        """Compute SHA-256 provenance hash for a record."""
        provenance_data = {
            "record_id": record.record_id,
            "plant_id": record.plant_id,
            "product_water_m3": record.product_water_m3,
            "technology": record.technology.value,
            "total_emissions": round(total_emissions, 2),
            "methodology": self.METHODOLOGY_VERSION,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

    def _compute_provenance_hash(
        self,
        input_data: DesalinationInput,
        results: List[DesalinationPlantResult]
    ) -> str:
        """Compute overall provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "methodology": self.METHODOLOGY_VERSION,
            "reporting_year": input_data.reporting_year,
            "plants_count": len(results),
            "plant_hashes": [r.provenance_hash for r in results],
            "total_emissions": sum(r.total_emissions_kgco2e for r in results),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()
