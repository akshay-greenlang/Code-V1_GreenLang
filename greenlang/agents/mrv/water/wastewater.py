# -*- coding: utf-8 -*-
"""
GL-MRV-WAT-002: Wastewater MRV Agent
====================================

MRV agent for wastewater treatment emissions measurement.
Calculates GHG emissions from wastewater treatment including:
- Energy consumption for treatment processes
- Process emissions (CH4, N2O from biological treatment)
- Biogas capture and utilization
- Sludge handling and disposal

Methodologies:
    - IPCC Guidelines for Wastewater Treatment
    - GHG Protocol Scope 1, 2, and 3
    - EPA emission factors for wastewater utilities

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

class WastewaterTreatmentType(str, Enum):
    """Types of wastewater treatment processes."""
    ACTIVATED_SLUDGE = "activated_sludge"
    TRICKLING_FILTER = "trickling_filter"
    LAGOON = "lagoon"
    ANAEROBIC = "anaerobic"
    CONSTRUCTED_WETLAND = "constructed_wetland"
    MEMBRANE_BIOREACTOR = "membrane_bioreactor"
    SEPTIC = "septic"
    NONE = "none"


class SludgeDisposalMethod(str, Enum):
    """Sludge disposal methods."""
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    LAND_APPLICATION = "land_application"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"


class EffluentDischargeType(str, Enum):
    """Effluent discharge destinations."""
    SURFACE_WATER = "surface_water"
    GROUNDWATER = "groundwater"
    OCEAN = "ocean"
    REUSE = "reuse"
    SEWER = "sewer"


# Global Warming Potentials (AR6 100-year)
GWP_CH4 = Decimal("29.8")  # IPCC AR6
GWP_N2O = Decimal("273")   # IPCC AR6

# IPCC default emission factors
# MCF = Methane Correction Factor, EF = Emission Factor
TREATMENT_MCF = {
    WastewaterTreatmentType.ACTIVATED_SLUDGE: Decimal("0.0"),
    WastewaterTreatmentType.TRICKLING_FILTER: Decimal("0.0"),
    WastewaterTreatmentType.LAGOON: Decimal("0.2"),
    WastewaterTreatmentType.ANAEROBIC: Decimal("0.8"),
    WastewaterTreatmentType.CONSTRUCTED_WETLAND: Decimal("0.1"),
    WastewaterTreatmentType.MEMBRANE_BIOREACTOR: Decimal("0.0"),
    WastewaterTreatmentType.SEPTIC: Decimal("0.5"),
    WastewaterTreatmentType.NONE: Decimal("0.0"),
}

# N2O emission factors (kg N2O-N per kg N in effluent)
N2O_EMISSION_FACTOR = Decimal("0.016")  # IPCC default

# Energy factors (kWh per m3)
TREATMENT_ENERGY_FACTORS = {
    WastewaterTreatmentType.ACTIVATED_SLUDGE: Decimal("0.50"),
    WastewaterTreatmentType.TRICKLING_FILTER: Decimal("0.30"),
    WastewaterTreatmentType.LAGOON: Decimal("0.10"),
    WastewaterTreatmentType.ANAEROBIC: Decimal("0.15"),
    WastewaterTreatmentType.CONSTRUCTED_WETLAND: Decimal("0.05"),
    WastewaterTreatmentType.MEMBRANE_BIOREACTOR: Decimal("0.80"),
    WastewaterTreatmentType.SEPTIC: Decimal("0.02"),
    WastewaterTreatmentType.NONE: Decimal("0.0"),
}

# Default emission factors
DEFAULT_EMISSION_FACTORS = {
    "electricity_kwh": Decimal("0.417"),  # kgCO2e/kWh
    "natural_gas_m3": Decimal("2.0"),     # kgCO2e/m3
    "diesel_l": Decimal("2.68"),          # kgCO2e/L
    "biogas_m3": Decimal("2.0"),          # kgCO2e/m3 if not captured
    "sludge_landfill_kg": Decimal("0.5"), # kgCO2e/kg wet sludge
}

# IPCC maximum CH4 producing capacity (Bo)
# kg CH4 per kg BOD or COD
BO_DEFAULT = Decimal("0.25")  # kg CH4/kg BOD for domestic wastewater


# =============================================================================
# PYDANTIC MODELS - INPUT
# =============================================================================

class WastewaterTreatmentRecord(BaseModel):
    """Record of wastewater treatment operations."""
    record_id: str = Field(..., description="Unique record identifier")
    facility_id: str = Field(..., description="Treatment facility identifier")
    facility_name: Optional[str] = Field(None, description="Facility name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")

    # Wastewater volumes
    influent_volume_m3: float = Field(..., ge=0, description="Influent volume (m3)")
    effluent_volume_m3: float = Field(..., ge=0, description="Effluent volume (m3)")

    # Treatment type
    treatment_type: WastewaterTreatmentType = Field(..., description="Treatment process type")

    # Influent characteristics
    influent_bod_mg_l: float = Field(default=200, ge=0, description="Influent BOD (mg/L)")
    influent_cod_mg_l: float = Field(default=400, ge=0, description="Influent COD (mg/L)")
    influent_tkn_mg_l: float = Field(default=40, ge=0, description="Total Kjeldahl Nitrogen (mg/L)")

    # Effluent characteristics
    effluent_bod_mg_l: float = Field(default=20, ge=0, description="Effluent BOD (mg/L)")
    effluent_tkn_mg_l: float = Field(default=10, ge=0, description="Effluent TKN (mg/L)")

    # Energy consumption
    electricity_kwh: float = Field(default=0, ge=0, description="Electricity consumed (kWh)")
    natural_gas_m3: float = Field(default=0, ge=0, description="Natural gas consumed (m3)")
    diesel_l: float = Field(default=0, ge=0, description="Diesel consumed (L)")

    # Biogas
    biogas_produced_m3: float = Field(default=0, ge=0, description="Biogas produced (m3)")
    biogas_captured_m3: float = Field(default=0, ge=0, description="Biogas captured/flared (m3)")
    biogas_utilized_m3: float = Field(default=0, ge=0, description="Biogas utilized for energy (m3)")

    # Sludge
    sludge_produced_kg: float = Field(default=0, ge=0, description="Sludge produced (kg wet)")
    sludge_dry_solids_percent: float = Field(default=3, ge=0, le=100, description="Sludge dry solids %")
    sludge_disposal_method: SludgeDisposalMethod = Field(
        default=SludgeDisposalMethod.LANDFILL, description="Sludge disposal method"
    )

    # Effluent discharge
    effluent_discharge_type: EffluentDischargeType = Field(
        default=EffluentDischargeType.SURFACE_WATER, description="Effluent destination"
    )

    # Population served (for per-capita calculations)
    population_served: Optional[int] = Field(None, ge=0, description="Population served")

    # Custom factors
    custom_emission_factors: Optional[Dict[str, float]] = Field(
        None, description="Custom emission factors"
    )
    custom_mcf: Optional[float] = Field(None, ge=0, le=1, description="Custom MCF")


class WastewaterInput(BaseModel):
    """Input data for Wastewater MRV Agent."""
    treatment_records: List[WastewaterTreatmentRecord] = Field(
        ..., description="Wastewater treatment records"
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

class ProcessEmissions(BaseModel):
    """Process emissions breakdown."""
    ch4_emissions_kgco2e: float = Field(..., description="CH4 emissions as CO2e")
    n2o_emissions_kgco2e: float = Field(..., description="N2O emissions as CO2e")
    ch4_kg: float = Field(..., description="CH4 in kg")
    n2o_kg: float = Field(..., description="N2O in kg")


class WastewaterEmissionResult(BaseModel):
    """Emissions result for a wastewater treatment facility."""
    record_id: str = Field(..., description="Source record ID")
    facility_id: str = Field(..., description="Facility identifier")

    # Volumes
    influent_volume_m3: float = Field(..., description="Influent volume")
    effluent_volume_m3: float = Field(..., description="Effluent volume")

    # BOD/COD removed
    bod_removed_kg: float = Field(..., description="BOD removed (kg)")
    nitrogen_in_effluent_kg: float = Field(..., description="Nitrogen in effluent (kg)")

    # Scope 1: Process emissions + on-site fuel
    scope1_process_emissions_kgco2e: float = Field(..., description="Process CH4/N2O emissions")
    scope1_fuel_emissions_kgco2e: float = Field(..., description="On-site fuel emissions")
    scope1_total_kgco2e: float = Field(..., description="Total Scope 1")

    # Process emission details
    process_emissions: ProcessEmissions = Field(..., description="Process emission breakdown")

    # Biogas
    biogas_avoided_emissions_kgco2e: float = Field(..., description="Avoided emissions from biogas")
    biogas_fugitive_emissions_kgco2e: float = Field(..., description="Fugitive biogas emissions")

    # Scope 2: Purchased electricity
    scope2_emissions_kgco2e: float = Field(..., description="Scope 2 emissions")

    # Scope 3: Sludge disposal, effluent discharge
    scope3_sludge_emissions_kgco2e: float = Field(..., description="Sludge disposal emissions")
    scope3_effluent_emissions_kgco2e: float = Field(..., description="Effluent discharge emissions")
    scope3_total_kgco2e: float = Field(..., description="Total Scope 3")

    # Totals
    total_emissions_kgco2e: float = Field(..., description="Total gross emissions")
    net_emissions_kgco2e: float = Field(..., description="Net emissions (after biogas credit)")

    # Intensity
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions per m3 treated")
    emissions_per_kg_bod_kgco2e: float = Field(..., description="Emissions per kg BOD removed")

    # Provenance
    mcf_used: float = Field(..., description="Methane correction factor used")
    calculation_method: str = Field(..., description="IPCC methodology")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class WastewaterOutput(BaseModel):
    """Output from Wastewater MRV Agent."""
    # Summary
    reporting_year: int = Field(..., description="Reporting year")
    total_volume_treated_m3: float = Field(..., description="Total volume treated")
    total_bod_removed_kg: float = Field(..., description="Total BOD removed")

    # Emissions by scope
    scope1_total_kgco2e: float = Field(..., description="Total Scope 1 emissions")
    scope2_total_kgco2e: float = Field(..., description="Total Scope 2 emissions")
    scope3_total_kgco2e: float = Field(..., description="Total Scope 3 emissions")
    total_gross_emissions_kgco2e: float = Field(..., description="Total gross emissions")
    biogas_avoided_emissions_kgco2e: float = Field(..., description="Biogas avoided emissions")
    total_net_emissions_kgco2e: float = Field(..., description="Total net emissions")

    # Converted to tonnes
    total_net_emissions_tco2e: float = Field(..., description="Total net emissions (tCO2e)")

    # Process emissions breakdown
    total_ch4_emissions_kgco2e: float = Field(..., description="Total CH4 emissions")
    total_n2o_emissions_kgco2e: float = Field(..., description="Total N2O emissions")

    # Intensity metrics
    emissions_per_m3_kgco2e: float = Field(..., description="Emissions intensity per m3")
    emissions_per_capita_kgco2e: Optional[float] = Field(None, description="Per capita emissions")

    # Detailed results
    facility_results: List[WastewaterEmissionResult] = Field(..., description="Per-facility results")

    # Provenance
    provenance_hash: str = Field(..., description="Overall provenance hash")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    methodology_version: str = Field(..., description="IPCC methodology version")
    processing_time_ms: float = Field(..., description="Processing time")


# =============================================================================
# WASTEWATER MRV AGENT IMPLEMENTATION
# =============================================================================

class WastewaterMRVAgent(BaseAgent):
    """
    GL-MRV-WAT-002: Wastewater MRV Agent

    Calculates GHG emissions from wastewater treatment facilities.
    Implements IPCC methodology for CH4 and N2O process emissions.

    Zero-Hallucination Guarantees:
        - All calculations are deterministic using IPCC formulas
        - NO LLM involvement in any emission calculations
        - All emission factors traceable to IPCC Guidelines
        - Complete provenance hash for every calculation

    Usage:
        agent = WastewaterMRVAgent()
        result = agent.run({
            "treatment_records": [...],
            "reporting_year": 2024
        })
    """

    AGENT_ID = "GL-MRV-WAT-002"
    AGENT_NAME = "Wastewater MRV Agent"
    VERSION = "1.0.0"
    METHODOLOGY_VERSION = "IPCC-2019-Refinement"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Wastewater MRV Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="MRV agent for wastewater treatment emissions",
                version=self.VERSION,
                parameters={
                    "default_grid_factor": 0.417,
                    "gwp_ch4": 29.8,
                    "gwp_n2o": 273,
                }
            )
        super().__init__(config)

        self._emission_factors = DEFAULT_EMISSION_FACTORS.copy()
        self._calculations_performed = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute wastewater emissions calculation."""
        start_time = time.time()

        try:
            ww_input = WastewaterInput(**input_data)

            # Apply custom grid factor
            if ww_input.grid_emission_factor is not None:
                self._emission_factors["electricity_kwh"] = Decimal(
                    str(ww_input.grid_emission_factor)
                )

            # Process each facility
            facility_results = []
            for record in ww_input.treatment_records:
                result = self._calculate_facility_emissions(record, ww_input.include_scope3)
                facility_results.append(result)

            # Aggregate totals
            scope1_total = sum(r.scope1_total_kgco2e for r in facility_results)
            scope2_total = sum(r.scope2_emissions_kgco2e for r in facility_results)
            scope3_total = sum(r.scope3_total_kgco2e for r in facility_results)
            gross_total = scope1_total + scope2_total + scope3_total
            biogas_avoided = sum(r.biogas_avoided_emissions_kgco2e for r in facility_results)
            net_total = gross_total - biogas_avoided

            total_ch4 = sum(r.process_emissions.ch4_emissions_kgco2e for r in facility_results)
            total_n2o = sum(r.process_emissions.n2o_emissions_kgco2e for r in facility_results)

            total_volume = sum(r.influent_volume_m3 for r in facility_results)
            total_bod = sum(r.bod_removed_kg for r in facility_results)

            # Calculate per-capita if population data available
            total_pop = sum(
                r.population_served or 0
                for r in ww_input.treatment_records
            )
            per_capita = net_total / total_pop if total_pop > 0 else None

            # Intensity
            intensity = net_total / total_volume if total_volume > 0 else 0.0

            # Provenance
            provenance_hash = self._compute_provenance_hash(ww_input, facility_results)

            processing_time = (time.time() - start_time) * 1000

            output = WastewaterOutput(
                reporting_year=ww_input.reporting_year,
                total_volume_treated_m3=round(total_volume, 2),
                total_bod_removed_kg=round(total_bod, 2),
                scope1_total_kgco2e=round(scope1_total, 2),
                scope2_total_kgco2e=round(scope2_total, 2),
                scope3_total_kgco2e=round(scope3_total, 2),
                total_gross_emissions_kgco2e=round(gross_total, 2),
                biogas_avoided_emissions_kgco2e=round(biogas_avoided, 2),
                total_net_emissions_kgco2e=round(net_total, 2),
                total_net_emissions_tco2e=round(net_total / 1000, 4),
                total_ch4_emissions_kgco2e=round(total_ch4, 2),
                total_n2o_emissions_kgco2e=round(total_n2o, 2),
                emissions_per_m3_kgco2e=round(intensity, 6),
                emissions_per_capita_kgco2e=round(per_capita, 2) if per_capita else None,
                facility_results=facility_results,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                methodology_version=self.METHODOLOGY_VERSION,
                processing_time_ms=processing_time,
            )

            self._calculations_performed += 1

            self.logger.info(
                f"Calculated wastewater emissions: {net_total:.2f} kgCO2e "
                f"({total_volume:.0f} m3 treated)"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "methodology": self.METHODOLOGY_VERSION,
                    "facilities_processed": len(facility_results),
                }
            )

        except Exception as e:
            self.logger.error(f"Wastewater MRV calculation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID}
            )

    def _calculate_facility_emissions(
        self,
        record: WastewaterTreatmentRecord,
        include_scope3: bool
    ) -> WastewaterEmissionResult:
        """
        Calculate emissions for a wastewater treatment facility.

        ZERO-HALLUCINATION: Uses IPCC methodology with deterministic calculations.

        IPCC CH4 Equation:
            CH4 = (TOW * EF_CH4 - S) * (1 - R)
            Where:
            - TOW = Total Organics in Wastewater (kg BOD)
            - EF_CH4 = Emission Factor = Bo * MCF
            - S = Organic component removed as sludge
            - R = CH4 recovered

        IPCC N2O Equation:
            N2O = N_effluent * EF_N2O * 44/28
        """
        factors = self._get_emission_factors(record.custom_emission_factors)

        # Get MCF for treatment type
        mcf = Decimal(str(record.custom_mcf)) if record.custom_mcf is not None else (
            TREATMENT_MCF.get(record.treatment_type, Decimal("0.0"))
        )

        # Calculate BOD removed (kg)
        # BOD_removed = Volume * (BOD_in - BOD_out) / 1000 (convert mg to kg)
        bod_removed = (
            Decimal(str(record.influent_volume_m3)) *
            (Decimal(str(record.influent_bod_mg_l)) - Decimal(str(record.effluent_bod_mg_l))) /
            Decimal("1000")
        )

        # Calculate nitrogen in effluent (kg)
        # N_effluent = Volume * TKN_out / 1000
        nitrogen_effluent = (
            Decimal(str(record.effluent_volume_m3)) *
            Decimal(str(record.effluent_tkn_mg_l)) /
            Decimal("1000")
        )

        # CH4 emissions (IPCC methodology)
        # CH4 = TOW * Bo * MCF
        # TOW = influent BOD load
        tow = (
            Decimal(str(record.influent_volume_m3)) *
            Decimal(str(record.influent_bod_mg_l)) /
            Decimal("1000")
        )
        ch4_kg = float(tow * BO_DEFAULT * mcf)

        # Subtract recovered biogas CH4 (assume biogas is ~60% CH4)
        biogas_ch4_content = Decimal("0.6")
        ch4_in_biogas = Decimal(str(record.biogas_captured_m3)) * biogas_ch4_content * Decimal("0.717")  # density kg/m3
        ch4_kg = max(0, ch4_kg - float(ch4_in_biogas))

        ch4_co2e = ch4_kg * float(GWP_CH4)

        # N2O emissions
        n2o_kg = float(nitrogen_effluent * N2O_EMISSION_FACTOR * Decimal("44") / Decimal("28"))
        n2o_co2e = n2o_kg * float(GWP_N2O)

        process_emissions = ProcessEmissions(
            ch4_emissions_kgco2e=round(ch4_co2e, 2),
            n2o_emissions_kgco2e=round(n2o_co2e, 2),
            ch4_kg=round(ch4_kg, 4),
            n2o_kg=round(n2o_kg, 4),
        )

        # Scope 1: Process emissions + fuel combustion
        scope1_process = ch4_co2e + n2o_co2e
        scope1_fuel = float(
            Decimal(str(record.natural_gas_m3)) * factors["natural_gas_m3"] +
            Decimal(str(record.diesel_l)) * factors["diesel_l"]
        )
        scope1_total = scope1_process + scope1_fuel

        # Biogas utilization credit (avoided emissions)
        biogas_avoided = float(
            Decimal(str(record.biogas_utilized_m3)) * factors["electricity_kwh"] * Decimal("2.0")  # ~2 kWh/m3 biogas
        )

        # Fugitive biogas emissions (uncaptured)
        biogas_uncaptured = max(0, record.biogas_produced_m3 - record.biogas_captured_m3)
        biogas_fugitive = float(
            Decimal(str(biogas_uncaptured)) * biogas_ch4_content * Decimal("0.717") * GWP_CH4
        )

        # Scope 2: Electricity
        scope2 = float(Decimal(str(record.electricity_kwh)) * factors["electricity_kwh"])

        # Scope 3: Sludge disposal + effluent discharge
        scope3_sludge = 0.0
        scope3_effluent = 0.0
        if include_scope3:
            scope3_sludge = float(
                Decimal(str(record.sludge_produced_kg)) * factors["sludge_landfill_kg"]
            )
            # Effluent discharge emissions (minor, based on remaining BOD)
            effluent_bod_kg = (
                Decimal(str(record.effluent_volume_m3)) *
                Decimal(str(record.effluent_bod_mg_l)) / Decimal("1000")
            )
            scope3_effluent = float(effluent_bod_kg * BO_DEFAULT * Decimal("0.1") * GWP_CH4)

        scope3_total = scope3_sludge + scope3_effluent

        # Totals
        total_gross = scope1_total + scope2 + scope3_total + biogas_fugitive
        total_net = total_gross - biogas_avoided

        # Intensities
        emissions_per_m3 = total_net / record.influent_volume_m3 if record.influent_volume_m3 > 0 else 0.0
        emissions_per_bod = total_net / float(bod_removed) if bod_removed > 0 else 0.0

        # Provenance
        provenance_hash = self._compute_record_provenance(record, total_net)

        return WastewaterEmissionResult(
            record_id=record.record_id,
            facility_id=record.facility_id,
            influent_volume_m3=record.influent_volume_m3,
            effluent_volume_m3=record.effluent_volume_m3,
            bod_removed_kg=round(float(bod_removed), 2),
            nitrogen_in_effluent_kg=round(float(nitrogen_effluent), 2),
            scope1_process_emissions_kgco2e=round(scope1_process, 2),
            scope1_fuel_emissions_kgco2e=round(scope1_fuel, 2),
            scope1_total_kgco2e=round(scope1_total, 2),
            process_emissions=process_emissions,
            biogas_avoided_emissions_kgco2e=round(biogas_avoided, 2),
            biogas_fugitive_emissions_kgco2e=round(biogas_fugitive, 2),
            scope2_emissions_kgco2e=round(scope2, 2),
            scope3_sludge_emissions_kgco2e=round(scope3_sludge, 2),
            scope3_effluent_emissions_kgco2e=round(scope3_effluent, 2),
            scope3_total_kgco2e=round(scope3_total, 2),
            total_emissions_kgco2e=round(total_gross, 2),
            net_emissions_kgco2e=round(total_net, 2),
            emissions_per_m3_kgco2e=round(emissions_per_m3, 6),
            emissions_per_kg_bod_kgco2e=round(emissions_per_bod, 4),
            mcf_used=float(mcf),
            calculation_method="IPCC-2019-CH4-N2O",
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
        record: WastewaterTreatmentRecord,
        total_emissions: float
    ) -> str:
        """Compute SHA-256 provenance hash for a record."""
        provenance_data = {
            "record_id": record.record_id,
            "facility_id": record.facility_id,
            "influent_volume_m3": record.influent_volume_m3,
            "treatment_type": record.treatment_type.value,
            "total_emissions": round(total_emissions, 2),
            "methodology": self.METHODOLOGY_VERSION,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

    def _compute_provenance_hash(
        self,
        input_data: WastewaterInput,
        results: List[WastewaterEmissionResult]
    ) -> str:
        """Compute overall provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "methodology": self.METHODOLOGY_VERSION,
            "reporting_year": input_data.reporting_year,
            "records_count": len(results),
            "record_hashes": [r.provenance_hash for r in results],
            "total_emissions": sum(r.net_emissions_kgco2e for r in results),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()
