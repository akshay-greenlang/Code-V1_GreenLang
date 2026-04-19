# -*- coding: utf-8 -*-
"""
GL-MRV-IND-001: Steel Production MRV Agent
===========================================

Industrial MRV agent for steel sector emissions measurement, reporting, and verification.
Implements CBAM-compliant calculations for all major steel production routes.

Production Routes:
    - BF-BOF: Blast Furnace - Basic Oxygen Furnace (integrated route)
    - EAF: Electric Arc Furnace (scrap-based)
    - DRI-EAF: Direct Reduced Iron + Electric Arc Furnace
    - H2-DRI: Hydrogen-based Direct Reduction + EAF

Sources:
    - IPCC 2006 Guidelines for National GHG Inventories, Volume 3
    - World Steel Association CO2 Emissions Data Collection Guidelines (2021)
    - IEA Iron and Steel Technology Roadmap (2020)
    - EU ETS Monitoring and Reporting Regulation (EU) 2018/2066
    - CBAM Implementing Regulation (EU) 2023/956 Annex III

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .base import (
    IndustrialMRVBaseAgent,
    IndustrialMRVInput,
    IndustrialMRVOutput,
    CalculationStep,
    EmissionFactor,
    DataQuality,
    VerificationStatus,
    NATURAL_GAS_EF_KG_CO2_PER_M3,
    COAL_EF_KG_CO2_PER_KG,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SteelProductionRoute(str, Enum):
    """Steel production routes."""
    BF_BOF = "BF_BOF"       # Blast Furnace - Basic Oxygen Furnace
    EAF = "EAF"             # Electric Arc Furnace
    DRI_EAF = "DRI_EAF"     # Direct Reduced Iron + EAF
    H2_DRI = "H2_DRI"       # Hydrogen DRI + EAF


class HydrogenSource(str, Enum):
    """Hydrogen production source for H2-DRI route."""
    GREEN = "GREEN"   # Electrolysis with renewable electricity
    BLUE = "BLUE"     # SMR with CCS
    GRAY = "GRAY"     # SMR without CCS


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class SteelMRVInput(IndustrialMRVInput):
    """Input model for Steel Production MRV."""

    production_route: SteelProductionRoute = Field(
        ..., description="Steel production technology route"
    )

    # Scrap input
    scrap_input_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Scrap steel input in metric tonnes"
    )

    # Route-specific inputs
    iron_ore_tonnes: Optional[Decimal] = Field(None, ge=0)
    coke_tonnes: Optional[Decimal] = Field(None, ge=0)
    limestone_tonnes: Optional[Decimal] = Field(None, ge=0)

    # H2-DRI specific
    hydrogen_source: Optional[HydrogenSource] = Field(None)
    hydrogen_kg: Optional[Decimal] = Field(None, ge=0)

    @field_validator('hydrogen_source')
    @classmethod
    def validate_h2_source(cls, v, info):
        """Validate hydrogen source for H2-DRI route."""
        route = info.data.get('production_route')
        if route == SteelProductionRoute.H2_DRI and v is None:
            raise ValueError("hydrogen_source required for H2-DRI route")
        return v


class SteelMRVOutput(IndustrialMRVOutput):
    """Output model for Steel Production MRV."""

    # Steel-specific fields
    production_route: str = Field(default="")
    scrap_input_tonnes: Decimal = Field(default=Decimal("0"))
    scrap_credit_tco2e: Decimal = Field(default=Decimal("0"))

    # Emissions breakdown
    process_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    combustion_emissions_tco2e: Decimal = Field(default=Decimal("0"))


# =============================================================================
# STEEL MRV AGENT
# =============================================================================

class SteelProductionMRVAgent(IndustrialMRVBaseAgent[SteelMRVInput, SteelMRVOutput]):
    """
    GL-MRV-IND-001: Steel Production MRV Agent

    Implements zero-hallucination emissions calculation for steel production
    with CBAM compliance and complete audit trail.

    Emission Factors (tCO2/t crude steel):
        - BF-BOF: 1.85 (World Steel Association, 2021)
        - EAF: 0.40 (IPCC 2006, Table 4.1)
        - DRI-EAF: 1.10 (IEA Technology Roadmap, 2020)
        - H2-DRI: 0.05-0.30 (HYBRIT project, 2022)

    Example:
        >>> agent = SteelProductionMRVAgent()
        >>> input_data = SteelMRVInput(
        ...     facility_id="STEEL_001",
        ...     reporting_period="2024-Q1",
        ...     production_tonnes=Decimal("10000"),
        ...     production_route=SteelProductionRoute.BF_BOF,
        ...     scrap_input_tonnes=Decimal("1000")
        ... )
        >>> result = agent.process(input_data)
        >>> print(f"Emissions: {result.total_emissions_tco2e} tCO2e")
    """

    AGENT_ID = "GL-MRV-IND-001"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Steel"
    CBAM_CN_CODE = "7206-7229"
    CBAM_PRODUCT_CATEGORY = "Iron and Steel"

    # Emission factors (tCO2/t crude steel) - DETERMINISTIC CONSTANTS
    EF_BF_BOF = Decimal("1.85")
    EF_EAF = Decimal("0.40")
    EF_DRI_EAF = Decimal("1.10")
    EF_H2_DRI_GREEN = Decimal("0.05")
    EF_H2_DRI_BLUE = Decimal("0.175")  # Midpoint
    EF_H2_DRI_GRAY = Decimal("0.30")

    # Scrap credit (tCO2/t scrap)
    SCRAP_CREDIT = Decimal("-1.50")

    # EAF electricity consumption (kWh/t)
    EAF_ELECTRICITY_KWH_PER_T = Decimal("400")

    def _load_emission_factors(self) -> None:
        """Load steel sector emission factors."""
        self._emission_factors = {
            "BF_BOF": EmissionFactor(
                factor_id="steel_bf_bof",
                value=self.EF_BF_BOF,
                unit="tCO2/t_steel",
                source="World Steel Association CO2 Data Collection 2021",
                region="global",
                valid_from="2021-01-01",
                uncertainty_percent=10.0
            ),
            "EAF": EmissionFactor(
                factor_id="steel_eaf",
                value=self.EF_EAF,
                unit="tCO2/t_steel",
                source="IPCC 2006 Vol 3 Ch 4 Table 4.1",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=15.0
            ),
            "DRI_EAF": EmissionFactor(
                factor_id="steel_dri_eaf",
                value=self.EF_DRI_EAF,
                unit="tCO2/t_steel",
                source="IEA Iron and Steel Technology Roadmap 2020",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
            "H2_DRI_GREEN": EmissionFactor(
                factor_id="steel_h2dri_green",
                value=self.EF_H2_DRI_GREEN,
                unit="tCO2/t_steel",
                source="HYBRIT Fossil-Free Steel Project 2022",
                region="global",
                valid_from="2022-01-01",
                uncertainty_percent=25.0
            ),
            "SCRAP_CREDIT": EmissionFactor(
                factor_id="steel_scrap_credit",
                value=self.SCRAP_CREDIT,
                unit="tCO2/t_scrap",
                source="World Steel LCA Methodology Report 2017",
                region="global",
                valid_from="2017-01-01",
                uncertainty_percent=20.0
            ),
        }

    def calculate_emissions(self, input_data: SteelMRVInput) -> SteelMRVOutput:
        """
        Calculate steel production emissions - ZERO HALLUCINATION.

        Args:
            input_data: Validated steel production input

        Returns:
            Complete MRV output with emissions breakdown
        """
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        # Generate calculation ID
        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Step 1: Get route emission factor
        step_num += 1
        route_ef = self._get_route_emission_factor(
            input_data.production_route,
            input_data.hydrogen_source
        )
        factors_used.append(route_ef)

        # Step 2: Calculate direct emissions
        step_num += 1
        direct_emissions = input_data.production_tonnes * route_ef.value
        direct_emissions = self._round_emissions(direct_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description=f"Direct emissions from {input_data.production_route.value} route",
            formula="production_tonnes * emission_factor",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "emission_factor_tCO2_per_t": str(route_ef.value)
            },
            output_value=direct_emissions,
            output_unit="tCO2e",
            source=route_ef.source
        ))

        # Step 3: Calculate indirect emissions (electricity)
        step_num += 1
        indirect_emissions = self._calculate_indirect_emissions(input_data)
        indirect_emissions = self._round_emissions(indirect_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Indirect emissions from electricity consumption",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh or Decimal("0")),
                "grid_factor_kg_co2_per_kwh": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=indirect_emissions,
            output_unit="tCO2e",
            source="Grid emission factor"
        ))

        # Step 4: Calculate scrap credit
        step_num += 1
        scrap_credit = self._calculate_scrap_credit(input_data.scrap_input_tonnes)
        factors_used.append(self._emission_factors["SCRAP_CREDIT"])

        steps.append(CalculationStep(
            step_number=step_num,
            description="Scrap credit (avoided primary production)",
            formula="scrap_tonnes * scrap_credit_factor",
            inputs={
                "scrap_tonnes": str(input_data.scrap_input_tonnes),
                "scrap_credit_factor": str(self.SCRAP_CREDIT)
            },
            output_value=scrap_credit,
            output_unit="tCO2e",
            source="World Steel LCA Methodology 2017"
        ))

        # Step 5: Calculate totals
        step_num += 1
        scope_1 = direct_emissions + scrap_credit
        scope_2 = indirect_emissions
        total_emissions = scope_1 + scope_2
        total_emissions = self._round_emissions(total_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions calculation",
            formula="direct + indirect + scrap_credit",
            inputs={
                "direct_emissions": str(direct_emissions),
                "indirect_emissions": str(indirect_emissions),
                "scrap_credit": str(scrap_credit)
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

        return SteelMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            production_route=input_data.production_route.value,
            scrap_input_tonnes=input_data.scrap_input_tonnes,
            scrap_credit_tco2e=scrap_credit,
            process_emissions_tco2e=direct_emissions,
            combustion_emissions_tco2e=Decimal("0"),
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

    def _get_route_emission_factor(
        self,
        route: SteelProductionRoute,
        h2_source: Optional[HydrogenSource]
    ) -> EmissionFactor:
        """Get emission factor for production route - DETERMINISTIC."""
        if route == SteelProductionRoute.BF_BOF:
            return self._emission_factors["BF_BOF"]
        elif route == SteelProductionRoute.EAF:
            return self._emission_factors["EAF"]
        elif route == SteelProductionRoute.DRI_EAF:
            return self._emission_factors["DRI_EAF"]
        elif route == SteelProductionRoute.H2_DRI:
            if h2_source == HydrogenSource.GREEN:
                return self._emission_factors["H2_DRI_GREEN"]
            elif h2_source == HydrogenSource.BLUE:
                return EmissionFactor(
                    factor_id="steel_h2dri_blue",
                    value=self.EF_H2_DRI_BLUE,
                    unit="tCO2/t_steel",
                    source="HYBRIT Project 2022",
                    region="global",
                    valid_from="2022-01-01",
                    uncertainty_percent=25.0
                )
            else:
                return EmissionFactor(
                    factor_id="steel_h2dri_gray",
                    value=self.EF_H2_DRI_GRAY,
                    unit="tCO2/t_steel",
                    source="HYBRIT Project 2022",
                    region="global",
                    valid_from="2022-01-01",
                    uncertainty_percent=25.0
                )
        raise ValueError(f"Unknown production route: {route}")

    def _calculate_indirect_emissions(self, input_data: SteelMRVInput) -> Decimal:
        """Calculate indirect emissions from electricity."""
        if (input_data.grid_emission_factor_kg_co2_per_kwh is None or
                input_data.grid_emission_factor_kg_co2_per_kwh == 0):
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh
        if electricity_kwh is None:
            # Estimate for EAF routes
            if input_data.production_route in [
                SteelProductionRoute.EAF, SteelProductionRoute.H2_DRI
            ]:
                electricity_kwh = (
                    input_data.production_tonnes * self.EAF_ELECTRICITY_KWH_PER_T
                )
            else:
                return Decimal("0")

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")  # Convert to tonnes

    def _calculate_scrap_credit(self, scrap_tonnes: Decimal) -> Decimal:
        """Calculate scrap credit (negative emissions)."""
        if scrap_tonnes <= 0:
            return Decimal("0")
        return self._round_emissions(scrap_tonnes * self.SCRAP_CREDIT)
