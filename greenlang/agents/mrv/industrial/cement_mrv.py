# -*- coding: utf-8 -*-
"""
GL-MRV-IND-002: Cement Production MRV Agent
============================================

Industrial MRV agent for cement sector emissions measurement, reporting, and verification.
Implements CBAM-compliant calculations following IPCC and GNR methodologies.

Cement Types (EN 197-1):
    - CEM I: Portland cement (95-100% clinker)
    - CEM II: Portland composite (65-94% clinker)
    - CEM III: Blast furnace cement (5-64% clinker)
    - CEM IV: Pozzolanic cement (45-89% clinker)
    - CEM V: Composite cement (20-64% clinker)

Sources:
    - IPCC 2006 Guidelines, Volume 3, Chapter 2
    - GCCA/CSI Getting the Numbers Right (GNR) Database
    - EU ETS Monitoring and Reporting Regulation (EU) 2018/2066
    - EN 197-1 Cement Composition Standard
    - CBAM Implementing Regulation (EU) 2023/956 Annex III

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

from .base import (
    IndustrialMRVBaseAgent,
    IndustrialMRVInput,
    IndustrialMRVOutput,
    CalculationStep,
    EmissionFactor,
    DataQuality,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CementType(str, Enum):
    """Cement types per EN 197-1."""
    CEM_I = "CEM_I"
    CEM_II_A = "CEM_II_A"
    CEM_II_B = "CEM_II_B"
    CEM_III_A = "CEM_III_A"
    CEM_III_B = "CEM_III_B"
    CEM_III_C = "CEM_III_C"
    CEM_IV_A = "CEM_IV_A"
    CEM_IV_B = "CEM_IV_B"
    CEM_V_A = "CEM_V_A"
    CEM_V_B = "CEM_V_B"


class KilnFuelType(str, Enum):
    """Kiln fuel types."""
    COAL = "COAL"
    PETCOKE = "PETCOKE"
    NATURAL_GAS = "NATURAL_GAS"
    ALTERNATIVE = "ALTERNATIVE"  # RDF, biomass
    MIXED = "MIXED"


class SCMType(str, Enum):
    """Supplementary Cementitious Materials."""
    GGBS = "GGBS"  # Ground Granulated Blast-furnace Slag
    FLY_ASH = "FLY_ASH"
    NATURAL_POZZOLAN = "NATURAL_POZZOLAN"
    LIMESTONE = "LIMESTONE"


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class SCMInput(BaseModel):
    """SCM input with quantity."""
    scm_type: SCMType
    quantity_tonnes: Decimal = Field(ge=0)


class CementMRVInput(IndustrialMRVInput):
    """Input model for Cement Production MRV."""

    cement_type: CementType = Field(
        default=CementType.CEM_I,
        description="Cement type per EN 197-1"
    )

    clinker_production_tonnes: Optional[Decimal] = Field(
        None, ge=0,
        description="Clinker production (if measured separately)"
    )

    clinker_ratio: Optional[Decimal] = Field(
        None, ge=Decimal("0.05"), le=Decimal("1.00"),
        description="Custom clinker-to-cement ratio"
    )

    kiln_fuel_type: KilnFuelType = Field(
        default=KilnFuelType.MIXED,
        description="Primary kiln fuel type"
    )

    kiln_fuel_mix: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom fuel mix percentages"
    )

    scm_inputs: List[SCMInput] = Field(
        default_factory=list,
        description="Supplementary cementitious materials"
    )


class CementMRVOutput(IndustrialMRVOutput):
    """Output model for Cement Production MRV."""

    # Cement-specific fields
    cement_type: str = Field(default="")
    clinker_production_tonnes: Decimal = Field(default=Decimal("0"))
    clinker_ratio: Decimal = Field(default=Decimal("0.95"))

    # Emissions breakdown
    calcination_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    kiln_fuel_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    scm_credits_tco2e: Decimal = Field(default=Decimal("0"))


# =============================================================================
# CEMENT MRV AGENT
# =============================================================================

class CementProductionMRVAgent(IndustrialMRVBaseAgent[CementMRVInput, CementMRVOutput]):
    """
    GL-MRV-IND-002: Cement Production MRV Agent

    Implements zero-hallucination emissions calculation for cement production
    with CBAM compliance and complete audit trail.

    Emission Sources:
        - Calcination: CaCO3 decomposition (0.525 tCO2/t clinker)
        - Kiln fuel combustion: 0.20-0.35 tCO2/t clinker depending on fuel
        - Electricity: Grinding and auxiliary (~110 kWh/t cement)

    Example:
        >>> agent = CementProductionMRVAgent()
        >>> input_data = CementMRVInput(
        ...     facility_id="CEMENT_001",
        ...     reporting_period="2024-Q1",
        ...     production_tonnes=Decimal("50000"),
        ...     cement_type=CementType.CEM_I,
        ...     kiln_fuel_type=KilnFuelType.COAL
        ... )
        >>> result = agent.process(input_data)
        >>> print(f"Emissions: {result.total_emissions_tco2e} tCO2e")
    """

    AGENT_ID = "GL-MRV-IND-002"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Cement"
    CBAM_CN_CODE = "2523"
    CBAM_PRODUCT_CATEGORY = "Cement"

    # Clinker calcination (tCO2/t clinker) - IPCC 2006
    EF_CLINKER_CALCINATION = Decimal("0.525")

    # Kiln fuel factors (tCO2/t clinker)
    EF_KILN_COAL = Decimal("0.350")
    EF_KILN_PETCOKE = Decimal("0.340")
    EF_KILN_NATURAL_GAS = Decimal("0.200")
    EF_KILN_ALTERNATIVE = Decimal("0.150")
    EF_KILN_MIXED = Decimal("0.310")  # Global average

    # Electricity consumption (kWh/t cement)
    GRINDING_ELECTRICITY_KWH_PER_T = Decimal("110")

    # SCM credit factors (tCO2/t SCM)
    SCM_CREDITS = {
        SCMType.GGBS: Decimal("0.070"),
        SCMType.FLY_ASH: Decimal("0.020"),
        SCMType.NATURAL_POZZOLAN: Decimal("0.010"),
        SCMType.LIMESTONE: Decimal("0.030"),
    }

    # Clinker ratios by cement type (EN 197-1)
    CLINKER_RATIOS = {
        CementType.CEM_I: Decimal("0.95"),
        CementType.CEM_II_A: Decimal("0.85"),
        CementType.CEM_II_B: Decimal("0.70"),
        CementType.CEM_III_A: Decimal("0.50"),
        CementType.CEM_III_B: Decimal("0.25"),
        CementType.CEM_III_C: Decimal("0.10"),
        CementType.CEM_IV_A: Decimal("0.75"),
        CementType.CEM_IV_B: Decimal("0.55"),
        CementType.CEM_V_A: Decimal("0.50"),
        CementType.CEM_V_B: Decimal("0.30"),
    }

    def _load_emission_factors(self) -> None:
        """Load cement sector emission factors."""
        self._emission_factors = {
            "calcination": EmissionFactor(
                factor_id="cement_calcination",
                value=self.EF_CLINKER_CALCINATION,
                unit="tCO2/t_clinker",
                source="IPCC 2006 Vol 3 Ch 2 Table 2.1",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=5.0
            ),
            "kiln_coal": EmissionFactor(
                factor_id="cement_kiln_coal",
                value=self.EF_KILN_COAL,
                unit="tCO2/t_clinker",
                source="GCCA GNR Database 2020",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=10.0
            ),
            "kiln_petcoke": EmissionFactor(
                factor_id="cement_kiln_petcoke",
                value=self.EF_KILN_PETCOKE,
                unit="tCO2/t_clinker",
                source="GCCA GNR Database 2020",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=10.0
            ),
            "kiln_gas": EmissionFactor(
                factor_id="cement_kiln_gas",
                value=self.EF_KILN_NATURAL_GAS,
                unit="tCO2/t_clinker",
                source="GCCA GNR Database 2020",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=10.0
            ),
            "kiln_alternative": EmissionFactor(
                factor_id="cement_kiln_alternative",
                value=self.EF_KILN_ALTERNATIVE,
                unit="tCO2/t_clinker",
                source="GCCA GNR Database 2020",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
        }

    def calculate_emissions(self, input_data: CementMRVInput) -> CementMRVOutput:
        """
        Calculate cement production emissions - ZERO HALLUCINATION.

        Args:
            input_data: Validated cement production input

        Returns:
            Complete MRV output with emissions breakdown
        """
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Step 1: Determine clinker production
        step_num += 1
        clinker_ratio = self._get_clinker_ratio(input_data)
        clinker_tonnes = input_data.clinker_production_tonnes or (
            input_data.production_tonnes * clinker_ratio
        )
        clinker_tonnes = self._round_emissions(clinker_tonnes)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate clinker production from cement type",
            formula="cement_tonnes * clinker_ratio",
            inputs={
                "cement_tonnes": str(input_data.production_tonnes),
                "cement_type": input_data.cement_type.value,
                "clinker_ratio": str(clinker_ratio)
            },
            output_value=clinker_tonnes,
            output_unit="tonnes_clinker",
            source="EN 197-1:2011"
        ))

        # Step 2: Calcination emissions
        step_num += 1
        calcination_ef = self._emission_factors["calcination"]
        calcination_emissions = clinker_tonnes * calcination_ef.value
        calcination_emissions = self._round_emissions(calcination_emissions)
        factors_used.append(calcination_ef)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Calcination emissions (CaCO3 decomposition)",
            formula="clinker_tonnes * 0.525",
            inputs={
                "clinker_tonnes": str(clinker_tonnes),
                "calcination_factor": str(calcination_ef.value)
            },
            output_value=calcination_emissions,
            output_unit="tCO2e",
            source=calcination_ef.source
        ))

        # Step 3: Kiln fuel emissions
        step_num += 1
        kiln_ef = self._get_kiln_fuel_factor(input_data.kiln_fuel_type)
        kiln_fuel_emissions = clinker_tonnes * kiln_ef.value
        kiln_fuel_emissions = self._round_emissions(kiln_fuel_emissions)
        factors_used.append(kiln_ef)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Kiln fuel combustion emissions",
            formula="clinker_tonnes * kiln_fuel_factor",
            inputs={
                "clinker_tonnes": str(clinker_tonnes),
                "kiln_fuel_type": input_data.kiln_fuel_type.value,
                "kiln_fuel_factor": str(kiln_ef.value)
            },
            output_value=kiln_fuel_emissions,
            output_unit="tCO2e",
            source=kiln_ef.source
        ))

        # Step 4: Electricity emissions
        step_num += 1
        electricity_emissions = self._calculate_electricity_emissions(input_data)
        electricity_emissions = self._round_emissions(electricity_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Electricity emissions (grinding, etc.)",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(
                    input_data.electricity_kwh or
                    input_data.production_tonnes * self.GRINDING_ELECTRICITY_KWH_PER_T
                ),
                "grid_factor": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=electricity_emissions,
            output_unit="tCO2e",
            source="Grid emission factor"
        ))

        # Step 5: SCM credits
        step_num += 1
        scm_credits = self._calculate_scm_credits(input_data.scm_inputs)
        scm_credits = self._round_emissions(scm_credits)

        steps.append(CalculationStep(
            step_number=step_num,
            description="SCM credits (avoided clinker production)",
            formula="sum(scm_tonnes * scm_credit_factor)",
            inputs={
                "scm_inputs": str([
                    {"type": s.scm_type.value, "tonnes": str(s.quantity_tonnes)}
                    for s in input_data.scm_inputs
                ])
            },
            output_value=scm_credits,
            output_unit="tCO2e",
            source="GCCA GNR Allocation Methodology"
        ))

        # Step 6: Total emissions
        step_num += 1
        scope_1 = calcination_emissions + kiln_fuel_emissions + scm_credits
        scope_2 = electricity_emissions
        total_emissions = scope_1 + scope_2
        total_emissions = self._round_emissions(total_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions calculation",
            formula="calcination + kiln_fuel + electricity + scm_credits",
            inputs={
                "calcination": str(calcination_emissions),
                "kiln_fuel": str(kiln_fuel_emissions),
                "electricity": str(electricity_emissions),
                "scm_credits": str(scm_credits)
            },
            output_value=total_emissions,
            output_unit="tCO2e",
            source="Summation"
        ))

        # Calculate intensity
        emission_intensity = (
            total_emissions / input_data.production_tonnes
            if input_data.production_tonnes > 0 else Decimal("0")
        )
        emission_intensity = self._round_intensity(emission_intensity)

        # Create CBAM output
        cbam_output = self._create_cbam_output(
            production_tonnes=input_data.production_tonnes,
            direct_emissions=scope_1,
            indirect_emissions=scope_2
        )

        return CementMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            cement_type=input_data.cement_type.value,
            clinker_production_tonnes=clinker_tonnes,
            clinker_ratio=clinker_ratio,
            calcination_emissions_tco2e=calcination_emissions,
            kiln_fuel_emissions_tco2e=kiln_fuel_emissions,
            scm_credits_tco2e=scm_credits,
            scope_1_emissions_tco2e=self._round_emissions(scope_1),
            scope_2_emissions_tco2e=self._round_emissions(scope_2),
            total_emissions_tco2e=total_emissions,
            emission_intensity_tco2e_per_t=emission_intensity,
            cbam_output=cbam_output,
            calculation_steps=steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True
        )

    def _get_clinker_ratio(self, input_data: CementMRVInput) -> Decimal:
        """Get clinker ratio from input or cement type."""
        if input_data.clinker_ratio is not None:
            return input_data.clinker_ratio
        return self.CLINKER_RATIOS.get(input_data.cement_type, Decimal("0.95"))

    def _get_kiln_fuel_factor(self, fuel_type: KilnFuelType) -> EmissionFactor:
        """Get kiln fuel emission factor - DETERMINISTIC."""
        fuel_map = {
            KilnFuelType.COAL: "kiln_coal",
            KilnFuelType.PETCOKE: "kiln_petcoke",
            KilnFuelType.NATURAL_GAS: "kiln_gas",
            KilnFuelType.ALTERNATIVE: "kiln_alternative",
            KilnFuelType.MIXED: None,
        }
        factor_id = fuel_map.get(fuel_type)
        if factor_id:
            return self._emission_factors[factor_id]
        # Return mixed/average factor
        return EmissionFactor(
            factor_id="cement_kiln_mixed",
            value=self.EF_KILN_MIXED,
            unit="tCO2/t_clinker",
            source="GCCA GNR Database 2020 (weighted average)",
            region="global",
            valid_from="2020-01-01",
            uncertainty_percent=15.0
        )

    def _calculate_electricity_emissions(self, input_data: CementMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh or (
            input_data.production_tonnes * self.GRINDING_ELECTRICITY_KWH_PER_T
        )
        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")

    def _calculate_scm_credits(self, scm_inputs: List[SCMInput]) -> Decimal:
        """Calculate SCM credits (negative emissions)."""
        if not scm_inputs:
            return Decimal("0")

        total_credit = Decimal("0")
        for scm in scm_inputs:
            factor = self.SCM_CREDITS.get(scm.scm_type, Decimal("0"))
            total_credit -= scm.quantity_tonnes * factor
        return total_credit
