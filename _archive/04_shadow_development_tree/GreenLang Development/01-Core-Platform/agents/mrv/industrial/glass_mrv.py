# -*- coding: utf-8 -*-
"""
GL-MRV-IND-006: Glass Production MRV Agent
===========================================

Industrial MRV agent for glass manufacturing emissions measurement, reporting,
and verification. Covers container glass, flat glass, and specialty glass.

Glass Types:
    - Container Glass: Bottles, jars
    - Flat Glass: Windows, automotive
    - Fiber Glass: Insulation, reinforcement
    - Specialty Glass: Laboratory, optical

Sources:
    - IPCC 2006 Guidelines, Volume 3, Chapter 2.4
    - European Container Glass Federation (FEVE)
    - Glass Alliance Europe
    - EPA AP-42 Chapter 11.15

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import Field

from .base import (
    IndustrialMRVBaseAgent,
    IndustrialMRVInput,
    IndustrialMRVOutput,
    CalculationStep,
    EmissionFactor,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class GlassType(str, Enum):
    """Glass product types."""
    CONTAINER = "CONTAINER"
    FLAT = "FLAT"
    FIBER = "FIBER"
    SPECIALTY = "SPECIALTY"


class FurnaceType(str, Enum):
    """Glass furnace types."""
    REGENERATIVE = "REGENERATIVE"
    RECUPERATIVE = "RECUPERATIVE"
    OXY_FUEL = "OXY_FUEL"
    ELECTRIC = "ELECTRIC"
    HYBRID = "HYBRID"


class GlassMRVInput(IndustrialMRVInput):
    """Input model for Glass Production MRV."""

    glass_type: GlassType = Field(
        default=GlassType.CONTAINER,
        description="Type of glass product"
    )

    furnace_type: FurnaceType = Field(
        default=FurnaceType.REGENERATIVE,
        description="Furnace technology"
    )

    # Raw materials
    cullet_ratio: Decimal = Field(
        default=Decimal("0.50"),
        ge=0, le=1,
        description="Recycled glass (cullet) ratio"
    )
    soda_ash_tonnes: Optional[Decimal] = Field(None, ge=0)
    limestone_tonnes: Optional[Decimal] = Field(None, ge=0)


class GlassMRVOutput(IndustrialMRVOutput):
    """Output model for Glass Production MRV."""

    glass_type: str = Field(default="")
    furnace_type: str = Field(default="")
    cullet_ratio: Decimal = Field(default=Decimal("0"))

    # Emissions breakdown
    combustion_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    process_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class GlassProductionMRVAgent(IndustrialMRVBaseAgent[GlassMRVInput, GlassMRVOutput]):
    """
    GL-MRV-IND-006: Glass Production MRV Agent

    Emission Sources:
        - Combustion: Melting furnace (natural gas, fuel oil)
        - Process: Carbonate decomposition (soda ash, limestone)

    Emission Factors (tCO2/t glass):
        - Container glass: 0.45-0.65
        - Flat glass: 0.50-0.70
        - Fiber glass: 0.60-0.80

    Cullet Impact: Each 10% increase in cullet reduces energy ~2.5%
    """

    AGENT_ID = "GL-MRV-IND-006"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Glass"
    CBAM_CN_CODE = "7001-7020"
    CBAM_PRODUCT_CATEGORY = "Glass"

    # Base emission factors (tCO2/t glass) at 0% cullet
    EF_CONTAINER = Decimal("0.65")
    EF_FLAT = Decimal("0.70")
    EF_FIBER = Decimal("0.80")
    EF_SPECIALTY = Decimal("0.90")

    # Process emission factors (tCO2/t raw material)
    EF_SODA_ASH = Decimal("0.415")  # Na2CO3 decomposition
    EF_LIMESTONE = Decimal("0.440")  # CaCO3 decomposition

    # Cullet energy reduction factor
    CULLET_REDUCTION_PER_10PCT = Decimal("0.025")

    # Electricity consumption (kWh/t glass)
    ELECTRICITY_CONSUMPTION = Decimal("250")

    def _load_emission_factors(self) -> None:
        """Load glass sector emission factors."""
        self._emission_factors = {
            "container": EmissionFactor(
                factor_id="glass_container",
                value=self.EF_CONTAINER,
                unit="tCO2/t_glass",
                source="FEVE Container Glass Report",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
            "flat": EmissionFactor(
                factor_id="glass_flat",
                value=self.EF_FLAT,
                unit="tCO2/t_glass",
                source="Glass Alliance Europe",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
            "soda_ash": EmissionFactor(
                factor_id="glass_soda_ash",
                value=self.EF_SODA_ASH,
                unit="tCO2/t_Na2CO3",
                source="IPCC 2006 Vol 3",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=5.0
            ),
            "limestone": EmissionFactor(
                factor_id="glass_limestone",
                value=self.EF_LIMESTONE,
                unit="tCO2/t_CaCO3",
                source="IPCC 2006 Vol 3",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=5.0
            ),
        }

    def calculate_emissions(self, input_data: GlassMRVInput) -> GlassMRVOutput:
        """Calculate glass production emissions."""
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Step 1: Combustion emissions (adjusted for cullet)
        step_num += 1
        base_ef = self._get_base_emission_factor(input_data.glass_type)
        factors_used.append(base_ef)

        # Cullet reduces energy requirements
        cullet_reduction = (
            input_data.cullet_ratio / Decimal("0.10") * self.CULLET_REDUCTION_PER_10PCT
        )
        adjusted_ef = base_ef.value * (Decimal("1") - cullet_reduction)

        combustion_emissions = input_data.production_tonnes * adjusted_ef
        combustion_emissions = self._round_emissions(combustion_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Combustion emissions (cullet-adjusted)",
            formula="production_tonnes * base_ef * (1 - cullet_reduction)",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "base_emission_factor": str(base_ef.value),
                "cullet_ratio": str(input_data.cullet_ratio),
                "cullet_reduction": str(cullet_reduction)
            },
            output_value=combustion_emissions,
            output_unit="tCO2e",
            source=base_ef.source
        ))

        # Step 2: Process emissions (carbonate decomposition)
        step_num += 1
        process_emissions = self._calculate_process_emissions(input_data)
        process_emissions = self._round_emissions(process_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Process emissions (carbonate decomposition)",
            formula="soda_ash * 0.415 + limestone * 0.440",
            inputs={
                "soda_ash_tonnes": str(input_data.soda_ash_tonnes or Decimal("0")),
                "limestone_tonnes": str(input_data.limestone_tonnes or Decimal("0"))
            },
            output_value=process_emissions,
            output_unit="tCO2e",
            source="IPCC 2006 stoichiometry"
        ))

        # Step 3: Electricity emissions
        step_num += 1
        electricity_emissions = self._calculate_electricity_emissions(input_data)
        electricity_emissions = self._round_emissions(electricity_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Electricity emissions",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh or Decimal("0")),
                "grid_factor": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=electricity_emissions,
            output_unit="tCO2e",
            source="Grid emission factor"
        ))

        # Totals
        scope_1 = combustion_emissions + process_emissions
        scope_2 = electricity_emissions
        total_emissions = scope_1 + scope_2

        emission_intensity = (
            total_emissions / input_data.production_tonnes
            if input_data.production_tonnes > 0 else Decimal("0")
        )

        cbam_output = self._create_cbam_output(
            production_tonnes=input_data.production_tonnes,
            direct_emissions=scope_1,
            indirect_emissions=scope_2
        )

        return GlassMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            glass_type=input_data.glass_type.value,
            furnace_type=input_data.furnace_type.value,
            cullet_ratio=input_data.cullet_ratio,
            combustion_emissions_tco2e=combustion_emissions,
            process_emissions_tco2e=process_emissions,
            scope_1_emissions_tco2e=self._round_emissions(scope_1),
            scope_2_emissions_tco2e=self._round_emissions(scope_2),
            total_emissions_tco2e=self._round_emissions(total_emissions),
            emission_intensity_tco2e_per_t=self._round_intensity(emission_intensity),
            cbam_output=cbam_output,
            calculation_steps=steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True
        )

    def _get_base_emission_factor(self, glass_type: GlassType) -> EmissionFactor:
        """Get base emission factor for glass type."""
        if glass_type == GlassType.CONTAINER:
            return self._emission_factors["container"]
        elif glass_type == GlassType.FLAT:
            return self._emission_factors["flat"]
        else:
            return EmissionFactor(
                factor_id=f"glass_{glass_type.value.lower()}",
                value=self.EF_FIBER if glass_type == GlassType.FIBER else self.EF_SPECIALTY,
                unit="tCO2/t_glass",
                source="Glass Alliance Europe",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=20.0
            )

    def _calculate_process_emissions(self, input_data: GlassMRVInput) -> Decimal:
        """Calculate process emissions from carbonate decomposition."""
        emissions = Decimal("0")

        if input_data.soda_ash_tonnes:
            emissions += input_data.soda_ash_tonnes * self.EF_SODA_ASH

        if input_data.limestone_tonnes:
            emissions += input_data.limestone_tonnes * self.EF_LIMESTONE

        return emissions

    def _calculate_electricity_emissions(self, input_data: GlassMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh or (
            input_data.production_tonnes * self.ELECTRICITY_CONSUMPTION
        )
        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")
