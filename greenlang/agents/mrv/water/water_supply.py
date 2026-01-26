# -*- coding: utf-8 -*-
"""
GL-MRV-WAT-001: Water Supply MRV Agent
======================================

MRV agent for water treatment and distribution emissions measurement.
Calculates GHG emissions from water supply systems including:
- Water treatment plant energy consumption
- Pumping and distribution energy
- Chemical usage in treatment processes
- Fugitive emissions from infrastructure

Methodologies:
    - GHG Protocol Scope 1, 2, and 3
    - EPA emission factors for water utilities
    - IPCC Guidelines for National Greenhouse Gas Inventories

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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class WaterSourceType(str, Enum):
    """Types of water sources."""
    SURFACE_WATER = "surface_water"
    GROUNDWATER = "groundwater"
    RECYCLED = "recycled"
    DESALINATED = "desalinated"
    MIXED = "mixed"


class TreatmentLevel(str, Enum):
    """Water treatment levels."""
    NONE = "none"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    ADVANCED = "advanced"


class DistributionType(str, Enum):
    """Water distribution system types."""
    GRAVITY = "gravity"
    PUMPED = "pumped"
    MIXED = "mixed"


# Default emission factors (kgCO2e per unit)
# Source: EPA emission factors, IPCC Guidelines
DEFAULT_EMISSION_FACTORS = {
    "electricity_kwh": Decimal("0.417"),  # US grid average kgCO2e/kWh
    "natural_gas_m3": Decimal("2.0"),     # kgCO2e/m3 natural gas
    "diesel_l": Decimal("2.68"),          # kgCO2e/L diesel
    "chlorine_kg": Decimal("1.5"),        # kgCO2e/kg chlorine production
    "alum_kg": Decimal("0.53"),           # kgCO2e/kg aluminum sulfate
    "lime_kg": Decimal("0.79"),           # kgCO2e/kg lime
    "polymer_kg": Decimal("2.0"),         # kgCO2e/kg polymer
    "activated_carbon_kg": Decimal("3.0"),  # kgCO2e/kg activated carbon
}

# Energy intensity benchmarks (kWh per m3 of water)
TREATMENT_ENERGY_BENCHMARKS = {
    TreatmentLevel.NONE: Decimal("0.02"),
    TreatmentLevel.PRIMARY: Decimal("0.15"),
    TreatmentLevel.SECONDARY: Decimal("0.30"),
    TreatmentLevel.TERTIARY: Decimal("0.50"),
    TreatmentLevel.ADVANCED: Decimal("0.80"),
}

# Source-specific energy factors (kWh per m3)
SOURCE_ENERGY_FACTORS = {
    WaterSourceType.SURFACE_WATER: Decimal("0.10"),
    WaterSourceType.GROUNDWATER: Decimal("0.30"),  # Higher pumping energy
    WaterSourceType.RECYCLED: Decimal("0.50"),
    WaterSourceType.DESALINATED: Decimal("3.50"),  # Very energy intensive
    WaterSourceType.MIXED: Decimal("0.25"),
}


# =============================================================================
# PYDANTIC MODELS - INPUT
# =============================================================================

class WaterTreatmentRecord(BaseModel):
    """Record of water treatment operations."""
    record_id: str = Field(..., description="Unique record identifier")
    facility_id: str = Field(..., description="Treatment facility identifier")
    facility_name: Optional[str] = Field(None, description="Facility name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")

    # Water volumes
    water_intake_m3: float = Field(..., ge=0, description="Raw water intake (m3)")
    water_treated_m3: float = Field(..., ge=0, description="Treated water output (m3)")
    water_losses_m3: float = Field(default=0, ge=0, description="Treatment losses (m3)")

    # Source and treatment
    water_source: WaterSourceType = Field(..., description="Water source type")
    treatment_level: TreatmentLevel = Field(..., description="Treatment level")

    # Energy consumption
    electricity_kwh: float = Field(default=0, ge=0, description="Electricity consumed (kWh)")
    natural_gas_m3: float = Field(default=0, ge=0, description="Natural gas consumed (m3)")
    diesel_l: float = Field(default=0, ge=0, description="Diesel consumed (L)")

    # Chemical usage
    chlorine_kg: float = Field(default=0, ge=0, description="Chlorine used (kg)")
    alum_kg: float = Field(default=0, ge=0, description="Aluminum sulfate used (kg)")
    lime_kg: float = Field(default=0, ge=0, description="Lime used (kg)")
    polymer_kg: float = Field(default=0, ge=0, description="Polymer used (kg)")
    activated_carbon_kg: float = Field(default=0, ge=0, description="Activated carbon (kg)")

    # Optional custom factors
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        None, description="Custom emission factors to override defaults"
    )

    @field_validator('water_treated_m3')
    @classmethod
    def validate_treated_volume(cls, v: float, info) -> float:
        """Ensure treated volume does not exceed intake."""
        intake = info.data.get('water_intake_m3', 0)
        if intake > 0 and v > intake * 1.1:  # Allow 10% measurement tolerance
            raise ValueError("Treated volume cannot significantly exceed intake")
        return v


class DistributionRecord(BaseModel):
    """Record of water distribution operations."""
    record_id: str = Field(..., description="Unique record identifier")
    network_id: str = Field(..., description="Distribution network identifier")
    network_name: Optional[str] = Field(None, description="Network name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")

    # Distribution volumes
    water_distributed_m3: float = Field(..., ge=0, description="Water distributed (m3)")
    water_delivered_m3: float = Field(..., ge=0, description="Water delivered to customers (m3)")
    non_revenue_water_m3: float = Field(default=0, ge=0, description="Non-revenue water/losses (m3)")

    # Network characteristics
    distribution_type: DistributionType = Field(..., description="Distribution system type")
    network_length_km: float = Field(default=0, ge=0, description="Network pipe length (km)")
    average_pressure_bar: float = Field(default=0, ge=0, description="Average system pressure (bar)")

    # Energy consumption
    pumping_electricity_kwh: float = Field(default=0, ge=0, description="Pumping electricity (kWh)")
    booster_stations_count: int = Field(default=0, ge=0, description="Number of booster stations")

    # Custom factors
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        None, description="Custom emission factors"
    )


class WaterSupplyInput(BaseModel):
    """Input data for Water Supply MRV Agent."""
    treatment_records: List[WaterTreatmentRecord] = Field(
        default_factory=list, description="Water treatment records"
    )
    distribution_records: List[DistributionRecord] = Field(
        default_factory=list, description="Distribution records"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    include_scope3: bool = Field(default=True, description="Include Scope 3 emissions")
    grid_emission_factor: Optional[float] = Field(
        None, description="Custom grid emission factor (kgCO2e/kWh)"
    )


# =============================================================================
# PYDANTIC MODELS - OUTPUT
# =============================================================================

class EmissionBreakdown(BaseModel):
    """Breakdown of emissions by source."""
    electricity_emissions_kgco2e: float = Field(..., description="Electricity emissions")
    natural_gas_emissions_kgco2e: float = Field(..., description="Natural gas emissions")
    diesel_emissions_kgco2e: float = Field(..., description="Diesel emissions")
    chemical_emissions_kgco2e: float = Field(..., description="Chemical production emissions")
    total_emissions_kgco2e: float = Field(..., description="Total emissions")

    # Intensity metrics
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions per m3 water")
    energy_intensity_kwh_per_m3: float = Field(..., description="Energy per m3 water")


class TreatmentEmissionResult(BaseModel):
    """Emissions result for a treatment facility."""
    record_id: str = Field(..., description="Source record ID")
    facility_id: str = Field(..., description="Facility identifier")

    # Volumes processed
    water_intake_m3: float = Field(..., description="Water intake")
    water_treated_m3: float = Field(..., description="Water treated")

    # Scope 1 emissions (on-site fuel combustion)
    scope1_emissions_kgco2e: float = Field(..., description="Scope 1 emissions")

    # Scope 2 emissions (purchased electricity)
    scope2_emissions_kgco2e: float = Field(..., description="Scope 2 emissions")

    # Scope 3 emissions (chemicals, upstream)
    scope3_emissions_kgco2e: float = Field(..., description="Scope 3 emissions")

    # Total and breakdown
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    emission_breakdown: EmissionBreakdown = Field(..., description="Detailed breakdown")

    # Provenance
    calculation_method: str = Field(..., description="Calculation method used")
    emission_factors_used: Dict[str, float] = Field(..., description="Emission factors applied")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class DistributionEmissionResult(BaseModel):
    """Emissions result for distribution network."""
    record_id: str = Field(..., description="Source record ID")
    network_id: str = Field(..., description="Network identifier")

    # Volumes
    water_distributed_m3: float = Field(..., description="Water distributed")
    water_delivered_m3: float = Field(..., description="Water delivered")
    non_revenue_water_percent: float = Field(..., description="NRW percentage")

    # Emissions
    scope2_emissions_kgco2e: float = Field(..., description="Scope 2 emissions (pumping)")

    # Intensity
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions per m3")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class WaterSupplyOutput(BaseModel):
    """Output from Water Supply MRV Agent."""
    # Summary
    reporting_year: int = Field(..., description="Reporting year")
    total_water_treated_m3: float = Field(..., description="Total water treated")
    total_water_distributed_m3: float = Field(..., description="Total water distributed")

    # Emissions by scope
    scope1_total_kgco2e: float = Field(..., description="Total Scope 1 emissions")
    scope2_total_kgco2e: float = Field(..., description="Total Scope 2 emissions")
    scope3_total_kgco2e: float = Field(..., description="Total Scope 3 emissions")
    total_emissions_kgco2e: float = Field(..., description="Total emissions all scopes")

    # Converted to tonnes
    total_emissions_tco2e: float = Field(..., description="Total emissions (tCO2e)")

    # Intensity metrics
    overall_emissions_per_m3_kgco2e: float = Field(..., description="Overall intensity")
    treatment_intensity_kgco2e_per_m3: float = Field(..., description="Treatment intensity")
    distribution_intensity_kgco2e_per_m3: float = Field(..., description="Distribution intensity")

    # Detailed results
    treatment_results: List[TreatmentEmissionResult] = Field(
        ..., description="Per-facility treatment results"
    )
    distribution_results: List[DistributionEmissionResult] = Field(
        ..., description="Per-network distribution results"
    )

    # Provenance
    provenance_hash: str = Field(..., description="Overall provenance hash")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    methodology_version: str = Field(..., description="Methodology version")
    processing_time_ms: float = Field(..., description="Processing time")


# =============================================================================
# WATER SUPPLY MRV AGENT IMPLEMENTATION
# =============================================================================

class WaterSupplyMRVAgent(BaseAgent):
    """
    GL-MRV-WAT-001: Water Supply MRV Agent

    Calculates GHG emissions from water treatment and distribution systems.
    Supports Scope 1, 2, and 3 emissions with complete provenance tracking.

    Zero-Hallucination Guarantees:
        - All calculations are deterministic mathematical operations
        - NO LLM involvement in any emission calculations
        - All emission factors are traceable to authoritative sources
        - Complete provenance hash for every calculation

    Usage:
        agent = WaterSupplyMRVAgent()
        result = agent.run({
            "treatment_records": [...],
            "distribution_records": [...],
            "reporting_year": 2024
        })
    """

    AGENT_ID = "GL-MRV-WAT-001"
    AGENT_NAME = "Water Supply MRV Agent"
    VERSION = "1.0.0"
    METHODOLOGY_VERSION = "GHG-Protocol-v2024.1"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Water Supply MRV Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="MRV agent for water treatment and distribution emissions",
                version=self.VERSION,
                parameters={
                    "default_grid_factor": 0.417,
                    "include_uncertainty": False,
                    "decimal_precision": 6,
                }
            )
        super().__init__(config)

        # Load emission factors
        self._emission_factors = DEFAULT_EMISSION_FACTORS.copy()

        # Statistics
        self._calculations_performed = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute water supply emissions calculation.

        Args:
            input_data: Dictionary containing water supply input data

        Returns:
            AgentResult with calculated emissions
        """
        start_time = time.time()

        try:
            # Parse and validate input
            ws_input = WaterSupplyInput(**input_data)

            # Apply custom grid factor if provided
            if ws_input.grid_emission_factor is not None:
                self._emission_factors["electricity_kwh"] = Decimal(
                    str(ws_input.grid_emission_factor)
                )

            # Process treatment records
            treatment_results = []
            for record in ws_input.treatment_records:
                result = self._calculate_treatment_emissions(
                    record, ws_input.include_scope3
                )
                treatment_results.append(result)

            # Process distribution records
            distribution_results = []
            for record in ws_input.distribution_records:
                result = self._calculate_distribution_emissions(record)
                distribution_results.append(result)

            # Calculate totals
            scope1_total = sum(r.scope1_emissions_kgco2e for r in treatment_results)
            scope2_total = (
                sum(r.scope2_emissions_kgco2e for r in treatment_results) +
                sum(r.scope2_emissions_kgco2e for r in distribution_results)
            )
            scope3_total = sum(r.scope3_emissions_kgco2e for r in treatment_results)
            total_emissions = scope1_total + scope2_total + scope3_total

            # Calculate volumes
            total_treated = sum(r.water_treated_m3 for r in treatment_results)
            total_distributed = sum(r.water_distributed_m3 for r in distribution_results)

            # Calculate intensities
            treatment_emissions = sum(r.total_emissions_kgco2e for r in treatment_results)
            distribution_emissions = sum(r.scope2_emissions_kgco2e for r in distribution_results)

            treatment_intensity = (
                treatment_emissions / total_treated if total_treated > 0 else 0.0
            )
            distribution_intensity = (
                distribution_emissions / total_distributed if total_distributed > 0 else 0.0
            )
            overall_intensity = (
                total_emissions / max(total_treated, total_distributed, 1)
            )

            # Generate provenance hash
            provenance_hash = self._compute_provenance_hash(
                ws_input, treatment_results, distribution_results
            )

            processing_time = (time.time() - start_time) * 1000

            output = WaterSupplyOutput(
                reporting_year=ws_input.reporting_year,
                total_water_treated_m3=total_treated,
                total_water_distributed_m3=total_distributed,
                scope1_total_kgco2e=round(scope1_total, 2),
                scope2_total_kgco2e=round(scope2_total, 2),
                scope3_total_kgco2e=round(scope3_total, 2),
                total_emissions_kgco2e=round(total_emissions, 2),
                total_emissions_tco2e=round(total_emissions / 1000, 4),
                overall_emissions_per_m3_kgco2e=round(overall_intensity, 6),
                treatment_intensity_kgco2e_per_m3=round(treatment_intensity, 6),
                distribution_intensity_kgco2e_per_m3=round(distribution_intensity, 6),
                treatment_results=treatment_results,
                distribution_results=distribution_results,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                methodology_version=self.METHODOLOGY_VERSION,
                processing_time_ms=processing_time,
            )

            self._calculations_performed += 1

            self.logger.info(
                f"Calculated water supply emissions: {total_emissions:.2f} kgCO2e "
                f"({total_treated:.0f} m3 treated, {total_distributed:.0f} m3 distributed)"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "methodology": self.METHODOLOGY_VERSION,
                    "records_processed": len(ws_input.treatment_records) + len(ws_input.distribution_records),
                }
            )

        except Exception as e:
            self.logger.error(f"Water supply MRV calculation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID}
            )

    def _calculate_treatment_emissions(
        self,
        record: WaterTreatmentRecord,
        include_scope3: bool
    ) -> TreatmentEmissionResult:
        """
        Calculate emissions for a water treatment facility.

        ZERO-HALLUCINATION: All calculations are deterministic.
        """
        # Get emission factors (use custom if provided)
        factors = self._get_emission_factors(record.custom_emission_factors)

        # Scope 1: On-site fuel combustion
        scope1_gas = float(Decimal(str(record.natural_gas_m3)) * factors["natural_gas_m3"])
        scope1_diesel = float(Decimal(str(record.diesel_l)) * factors["diesel_l"])
        scope1_total = scope1_gas + scope1_diesel

        # Scope 2: Purchased electricity
        scope2_total = float(
            Decimal(str(record.electricity_kwh)) * factors["electricity_kwh"]
        )

        # Scope 3: Embodied emissions in chemicals
        scope3_total = 0.0
        if include_scope3:
            scope3_chlorine = float(Decimal(str(record.chlorine_kg)) * factors["chlorine_kg"])
            scope3_alum = float(Decimal(str(record.alum_kg)) * factors["alum_kg"])
            scope3_lime = float(Decimal(str(record.lime_kg)) * factors["lime_kg"])
            scope3_polymer = float(Decimal(str(record.polymer_kg)) * factors["polymer_kg"])
            scope3_carbon = float(
                Decimal(str(record.activated_carbon_kg)) * factors["activated_carbon_kg"]
            )
            scope3_total = scope3_chlorine + scope3_alum + scope3_lime + scope3_polymer + scope3_carbon

        total_emissions = scope1_total + scope2_total + scope3_total

        # Calculate intensities
        emissions_per_m3 = (
            total_emissions / record.water_treated_m3
            if record.water_treated_m3 > 0 else 0.0
        )
        energy_per_m3 = (
            record.electricity_kwh / record.water_treated_m3
            if record.water_treated_m3 > 0 else 0.0
        )

        # Create emission breakdown
        breakdown = EmissionBreakdown(
            electricity_emissions_kgco2e=round(scope2_total, 2),
            natural_gas_emissions_kgco2e=round(scope1_gas, 2),
            diesel_emissions_kgco2e=round(scope1_diesel, 2),
            chemical_emissions_kgco2e=round(scope3_total, 2),
            total_emissions_kgco2e=round(total_emissions, 2),
            emissions_per_m3_kgco2e=round(emissions_per_m3, 6),
            energy_intensity_kwh_per_m3=round(energy_per_m3, 4),
        )

        # Generate provenance hash for this record
        provenance_hash = self._compute_record_provenance(record, breakdown)

        return TreatmentEmissionResult(
            record_id=record.record_id,
            facility_id=record.facility_id,
            water_intake_m3=record.water_intake_m3,
            water_treated_m3=record.water_treated_m3,
            scope1_emissions_kgco2e=round(scope1_total, 2),
            scope2_emissions_kgco2e=round(scope2_total, 2),
            scope3_emissions_kgco2e=round(scope3_total, 2),
            total_emissions_kgco2e=round(total_emissions, 2),
            emission_breakdown=breakdown,
            calculation_method="activity-based",
            emission_factors_used={k: float(v) for k, v in factors.items()},
            provenance_hash=provenance_hash,
        )

    def _calculate_distribution_emissions(
        self,
        record: DistributionRecord
    ) -> DistributionEmissionResult:
        """
        Calculate emissions for water distribution.

        ZERO-HALLUCINATION: All calculations are deterministic.
        """
        factors = self._get_emission_factors(record.custom_emission_factors)

        # Scope 2: Pumping electricity
        scope2_total = float(
            Decimal(str(record.pumping_electricity_kwh)) * factors["electricity_kwh"]
        )

        # Calculate NRW percentage
        nrw_percent = (
            (record.non_revenue_water_m3 / record.water_distributed_m3 * 100)
            if record.water_distributed_m3 > 0 else 0.0
        )

        # Calculate intensity
        emissions_per_m3 = (
            scope2_total / record.water_delivered_m3
            if record.water_delivered_m3 > 0 else 0.0
        )

        # Generate provenance hash
        provenance_str = json.dumps({
            "record_id": record.record_id,
            "electricity_kwh": record.pumping_electricity_kwh,
            "water_distributed_m3": record.water_distributed_m3,
            "emissions": scope2_total,
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

        return DistributionEmissionResult(
            record_id=record.record_id,
            network_id=record.network_id,
            water_distributed_m3=record.water_distributed_m3,
            water_delivered_m3=record.water_delivered_m3,
            non_revenue_water_percent=round(nrw_percent, 2),
            scope2_emissions_kgco2e=round(scope2_total, 2),
            emissions_per_m3_kgco2e=round(emissions_per_m3, 6),
            provenance_hash=provenance_hash,
        )

    def _get_emission_factors(
        self,
        custom_factors: Optional[Dict[str, float]]
    ) -> Dict[str, Decimal]:
        """Get emission factors, applying custom overrides if provided."""
        factors = self._emission_factors.copy()
        if custom_factors:
            for key, value in custom_factors.items():
                if key in factors:
                    factors[key] = Decimal(str(value))
        return factors

    def _compute_record_provenance(
        self,
        record: WaterTreatmentRecord,
        breakdown: EmissionBreakdown
    ) -> str:
        """Compute SHA-256 provenance hash for a treatment record."""
        provenance_data = {
            "record_id": record.record_id,
            "facility_id": record.facility_id,
            "water_treated_m3": record.water_treated_m3,
            "electricity_kwh": record.electricity_kwh,
            "total_emissions": breakdown.total_emissions_kgco2e,
            "methodology": self.METHODOLOGY_VERSION,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

    def _compute_provenance_hash(
        self,
        input_data: WaterSupplyInput,
        treatment_results: List[TreatmentEmissionResult],
        distribution_results: List[DistributionEmissionResult]
    ) -> str:
        """Compute overall provenance hash for the calculation."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "methodology": self.METHODOLOGY_VERSION,
            "reporting_year": input_data.reporting_year,
            "treatment_records": len(treatment_results),
            "distribution_records": len(distribution_results),
            "treatment_hashes": [r.provenance_hash for r in treatment_results],
            "distribution_hashes": [r.provenance_hash for r in distribution_results],
            "total_emissions": sum(r.total_emissions_kgco2e for r in treatment_results) +
                             sum(r.scope2_emissions_kgco2e for r in distribution_results),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        base_stats = super().get_stats()
        base_stats["calculations_performed"] = self._calculations_performed
        return base_stats
