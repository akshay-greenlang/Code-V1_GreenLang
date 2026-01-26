# -*- coding: utf-8 -*-
"""
GL-MRV-IND-005: Pulp & Paper MRV Agent
=======================================

Industrial MRV agent for pulp and paper sector emissions measurement, reporting,
and verification. Covers chemical pulping, mechanical pulping, and paper production.

Production Types:
    - Chemical Pulp: Kraft process, Sulfite process
    - Mechanical Pulp: TMP, CTMP, SGW
    - Paper Products: Printing, Packaging, Tissue

Sources:
    - IPCC 2006 Guidelines, Volume 3, Chapter 2.4 (Pulp and Paper)
    - CEPI Carbon Footprint Guidelines
    - NCASI Environmental Performance
    - EPA AP-42 Chapter 10.2

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


class PulpType(str, Enum):
    """Pulp production types."""
    KRAFT = "KRAFT"  # Chemical - Kraft process
    SULFITE = "SULFITE"  # Chemical - Sulfite process
    TMP = "TMP"  # Thermo-mechanical pulp
    CTMP = "CTMP"  # Chemi-thermo-mechanical pulp
    SGW = "SGW"  # Stone groundwood
    RECYCLED = "RECYCLED"  # Recycled fiber


class PaperType(str, Enum):
    """Paper product types."""
    PRINTING_WRITING = "PRINTING_WRITING"
    NEWSPRINT = "NEWSPRINT"
    PACKAGING = "PACKAGING"
    TISSUE = "TISSUE"
    SPECIALTY = "SPECIALTY"


class PulpPaperMRVInput(IndustrialMRVInput):
    """Input model for Pulp & Paper MRV."""

    pulp_type: Optional[PulpType] = Field(None)
    paper_type: Optional[PaperType] = Field(None)

    # Process inputs
    wood_input_tonnes: Optional[Decimal] = Field(None, ge=0)
    recycled_fiber_tonnes: Optional[Decimal] = Field(None, ge=0)
    black_liquor_gj: Optional[Decimal] = Field(None, ge=0)  # Biogenic fuel
    steam_gj: Optional[Decimal] = Field(None, ge=0)

    # Biogenic vs fossil fuel split
    fossil_fuel_share: Decimal = Field(default=Decimal("0.30"), ge=0, le=1)


class PulpPaperMRVOutput(IndustrialMRVOutput):
    """Output model for Pulp & Paper MRV."""

    pulp_type: Optional[str] = Field(default=None)
    paper_type: Optional[str] = Field(default=None)

    # Emissions breakdown
    combustion_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    process_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    biogenic_emissions_tco2: Decimal = Field(default=Decimal("0"))  # Reported separately


class PulpPaperMRVAgent(IndustrialMRVBaseAgent[PulpPaperMRVInput, PulpPaperMRVOutput]):
    """
    GL-MRV-IND-005: Pulp & Paper MRV Agent

    Emission Factors (tCO2/t product):
        - Kraft Pulp: 0.40-0.80 (fossil portion)
        - Mechanical Pulp: 0.10-0.30
        - Paper: 0.30-0.60

    Note: Paper industry has significant biogenic emissions from black liquor
    combustion. These are reported separately per GHG Protocol.
    """

    AGENT_ID = "GL-MRV-IND-005"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Pulp & Paper"
    CBAM_CN_CODE = "4701-4813"
    CBAM_PRODUCT_CATEGORY = "Pulp and Paper"

    # Emission factors (tCO2/t product) - fossil portion
    EF_KRAFT_PULP = Decimal("0.60")
    EF_SULFITE_PULP = Decimal("0.70")
    EF_MECHANICAL_PULP = Decimal("0.20")
    EF_RECYCLED_PULP = Decimal("0.15")
    EF_PAPER = Decimal("0.45")

    # Electricity consumption (kWh/t)
    ELEC_KRAFT = Decimal("700")
    ELEC_MECHANICAL = Decimal("2000")
    ELEC_PAPER = Decimal("500")

    def _load_emission_factors(self) -> None:
        """Load pulp & paper emission factors."""
        self._emission_factors = {
            "kraft": EmissionFactor(
                factor_id="pp_kraft",
                value=self.EF_KRAFT_PULP,
                unit="tCO2/t_pulp",
                source="CEPI Carbon Footprint Guidelines",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=20.0
            ),
            "mechanical": EmissionFactor(
                factor_id="pp_mechanical",
                value=self.EF_MECHANICAL_PULP,
                unit="tCO2/t_pulp",
                source="CEPI Carbon Footprint Guidelines",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=20.0
            ),
            "paper": EmissionFactor(
                factor_id="pp_paper",
                value=self.EF_PAPER,
                unit="tCO2/t_paper",
                source="CEPI Carbon Footprint Guidelines",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=20.0
            ),
        }

    def calculate_emissions(self, input_data: PulpPaperMRVInput) -> PulpPaperMRVOutput:
        """Calculate pulp & paper emissions."""
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Determine emission factor based on product type
        ef = self._get_emission_factor(input_data)
        factors_used.append(ef)

        # Step 1: Process/combustion emissions
        step_num += 1
        process_emissions = (
            input_data.production_tonnes * ef.value * input_data.fossil_fuel_share
        )
        process_emissions = self._round_emissions(process_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Process/combustion emissions (fossil fuel portion)",
            formula="production_tonnes * emission_factor * fossil_share",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "emission_factor": str(ef.value),
                "fossil_fuel_share": str(input_data.fossil_fuel_share)
            },
            output_value=process_emissions,
            output_unit="tCO2e",
            source=ef.source
        ))

        # Step 2: Electricity emissions
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

        # Step 3: Biogenic emissions (reported separately)
        step_num += 1
        biogenic_share = Decimal("1") - input_data.fossil_fuel_share
        biogenic_emissions = input_data.production_tonnes * ef.value * biogenic_share
        biogenic_emissions = self._round_emissions(biogenic_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Biogenic emissions (reported separately)",
            formula="production_tonnes * emission_factor * biogenic_share",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "emission_factor": str(ef.value),
                "biogenic_share": str(biogenic_share)
            },
            output_value=biogenic_emissions,
            output_unit="tCO2",
            source="Biogenic carbon accounting"
        ))

        # Totals
        scope_1 = process_emissions
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

        return PulpPaperMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            pulp_type=input_data.pulp_type.value if input_data.pulp_type else None,
            paper_type=input_data.paper_type.value if input_data.paper_type else None,
            combustion_emissions_tco2e=process_emissions,
            process_emissions_tco2e=Decimal("0"),
            biogenic_emissions_tco2=biogenic_emissions,
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

    def _get_emission_factor(self, input_data: PulpPaperMRVInput) -> EmissionFactor:
        """Get appropriate emission factor."""
        if input_data.pulp_type in [PulpType.KRAFT, PulpType.SULFITE]:
            return self._emission_factors["kraft"]
        elif input_data.pulp_type in [PulpType.TMP, PulpType.CTMP, PulpType.SGW]:
            return self._emission_factors["mechanical"]
        else:
            return self._emission_factors["paper"]

    def _calculate_electricity_emissions(self, input_data: PulpPaperMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh
        if electricity_kwh is None:
            if input_data.pulp_type in [PulpType.TMP, PulpType.CTMP, PulpType.SGW]:
                electricity_kwh = input_data.production_tonnes * self.ELEC_MECHANICAL
            elif input_data.pulp_type in [PulpType.KRAFT, PulpType.SULFITE]:
                electricity_kwh = input_data.production_tonnes * self.ELEC_KRAFT
            else:
                electricity_kwh = input_data.production_tonnes * self.ELEC_PAPER

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")
