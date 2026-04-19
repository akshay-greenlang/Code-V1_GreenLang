# -*- coding: utf-8 -*-
"""
GL-MRV-IND-004: Aluminum Production MRV Agent
==============================================

Industrial MRV agent for aluminum sector emissions measurement, reporting, and verification.
Covers primary aluminum smelting (Hall-Heroult process) and secondary aluminum recycling.

Production Routes:
    - Primary: Bauxite -> Alumina -> Aluminum (Hall-Heroult electrolysis)
    - Secondary: Scrap recycling (remelting)

Emission Sources:
    - Direct: Anode consumption (PFC emissions), fuel combustion
    - Indirect: Electricity for electrolysis (very high)

Sources:
    - IPCC 2006 Guidelines, Volume 3, Chapter 4 (Metal Industry)
    - International Aluminium Institute (IAI) GHG Protocol
    - World Aluminium - Life Cycle Inventory Data
    - CBAM Implementing Regulation (EU) 2023/956 Annex III

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
    DataQuality,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AluminumProductionRoute(str, Enum):
    """Aluminum production routes."""
    PRIMARY_PREBAKE = "PRIMARY_PREBAKE"  # Prebake anode technology
    PRIMARY_SODERBERG = "PRIMARY_SODERBERG"  # Soderberg anode (older)
    SECONDARY = "SECONDARY"  # Scrap recycling


class AnodeTechnology(str, Enum):
    """Anode technology types."""
    PREBAKE = "PREBAKE"
    CWPB = "CWPB"  # Center-Worked Prebake
    SWPB = "SWPB"  # Side-Worked Prebake
    VSS = "VSS"    # Vertical Stud Soderberg
    HSS = "HSS"    # Horizontal Stud Soderberg


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class AluminumMRVInput(IndustrialMRVInput):
    """Input model for Aluminum Production MRV."""

    production_route: AluminumProductionRoute = Field(
        ..., description="Aluminum production route"
    )

    anode_technology: Optional[AnodeTechnology] = Field(
        None, description="Anode technology (for primary production)"
    )

    # Alumina input (for primary)
    alumina_tonnes: Optional[Decimal] = Field(None, ge=0)

    # Scrap input (for secondary)
    scrap_input_tonnes: Optional[Decimal] = Field(None, ge=0)

    # Anode consumption
    anode_consumption_kg_per_t: Optional[Decimal] = Field(
        None, ge=0,
        description="Net anode consumption kg/t Al"
    )

    # PFC emissions data (if measured)
    cf4_emissions_kg: Optional[Decimal] = Field(None, ge=0)
    c2f6_emissions_kg: Optional[Decimal] = Field(None, ge=0)


class AluminumMRVOutput(IndustrialMRVOutput):
    """Output model for Aluminum Production MRV."""

    production_route: str = Field(default="")
    anode_technology: Optional[str] = Field(default=None)

    # Emissions breakdown
    electrolysis_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    anode_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    pfc_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    alumina_emissions_tco2e: Decimal = Field(default=Decimal("0"))


# =============================================================================
# ALUMINUM MRV AGENT
# =============================================================================

class AluminumProductionMRVAgent(IndustrialMRVBaseAgent[AluminumMRVInput, AluminumMRVOutput]):
    """
    GL-MRV-IND-004: Aluminum Production MRV Agent

    Implements zero-hallucination emissions calculation for aluminum production.

    Key Characteristics:
        - Primary aluminum is extremely electricity-intensive (~15 MWh/t)
        - PFC emissions (CF4, C2F6) have very high GWP
        - Secondary aluminum uses ~5% of primary energy

    Emission Factors:
        - Primary (direct): ~1.5 tCO2e/t Al (anode consumption)
        - Primary (indirect): ~8-15 tCO2e/t Al (electricity, depends on grid)
        - Secondary: ~0.3 tCO2e/t Al

    Example:
        >>> agent = AluminumProductionMRVAgent()
        >>> input_data = AluminumMRVInput(
        ...     facility_id="ALUM_001",
        ...     reporting_period="2024-Q1",
        ...     production_tonnes=Decimal("50000"),
        ...     production_route=AluminumProductionRoute.PRIMARY_PREBAKE,
        ...     electricity_kwh=Decimal("750000000"),
        ...     grid_emission_factor_kg_co2_per_kwh=Decimal("0.5")
        ... )
        >>> result = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-IND-004"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Aluminum"
    CBAM_CN_CODE = "7601"
    CBAM_PRODUCT_CATEGORY = "Aluminum"

    # Electricity consumption (kWh/t aluminum)
    ELECTRICITY_PRIMARY = Decimal("15000")  # 15 MWh/t
    ELECTRICITY_SECONDARY = Decimal("700")   # 0.7 MWh/t

    # Anode consumption factors (tCO2/t Al) - IPCC 2006
    EF_ANODE_PREBAKE = Decimal("1.50")
    EF_ANODE_SODERBERG = Decimal("1.80")

    # PFC emission factors (slope method)
    # CF4: ~6.5 kg/t Al (Prebake), ~17 kg/t Al (Soderberg)
    # C2F6: ~0.6 kg/t Al (Prebake), ~1.7 kg/t Al (Soderberg)
    EF_CF4_PREBAKE = Decimal("6.5")   # kg/t Al
    EF_CF4_SODERBERG = Decimal("17.0")
    EF_C2F6_PREBAKE = Decimal("0.6")
    EF_C2F6_SODERBERG = Decimal("1.7")

    # GWP values (AR5)
    GWP_CF4 = Decimal("6630")
    GWP_C2F6 = Decimal("11100")

    # Secondary (recycling) emission factor
    EF_SECONDARY = Decimal("0.30")

    # Alumina refining factor
    EF_ALUMINA = Decimal("0.50")  # tCO2/t alumina

    def _load_emission_factors(self) -> None:
        """Load aluminum sector emission factors."""
        self._emission_factors = {
            "anode_prebake": EmissionFactor(
                factor_id="al_anode_prebake",
                value=self.EF_ANODE_PREBAKE,
                unit="tCO2/t_Al",
                source="IPCC 2006 Vol 3 Ch 4",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=10.0
            ),
            "anode_soderberg": EmissionFactor(
                factor_id="al_anode_soderberg",
                value=self.EF_ANODE_SODERBERG,
                unit="tCO2/t_Al",
                source="IPCC 2006 Vol 3 Ch 4",
                region="global",
                valid_from="2006-01-01",
                uncertainty_percent=10.0
            ),
            "secondary": EmissionFactor(
                factor_id="al_secondary",
                value=self.EF_SECONDARY,
                unit="tCO2/t_Al",
                source="IAI GHG Protocol",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
            "alumina": EmissionFactor(
                factor_id="al_alumina",
                value=self.EF_ALUMINA,
                unit="tCO2/t_alumina",
                source="World Aluminium LCI",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            ),
        }

    def calculate_emissions(self, input_data: AluminumMRVInput) -> AluminumMRVOutput:
        """
        Calculate aluminum production emissions - ZERO HALLUCINATION.

        Args:
            input_data: Validated aluminum production input

        Returns:
            Complete MRV output with emissions breakdown
        """
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        if input_data.production_route == AluminumProductionRoute.SECONDARY:
            return self._calculate_secondary(input_data, calc_id)

        # Primary production calculations
        # Step 1: Anode consumption emissions
        step_num += 1
        anode_ef = self._get_anode_emission_factor(input_data)
        anode_emissions = input_data.production_tonnes * anode_ef.value
        anode_emissions = self._round_emissions(anode_emissions)
        factors_used.append(anode_ef)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Anode consumption emissions",
            formula="production_tonnes * anode_emission_factor",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "anode_technology": str(input_data.anode_technology or "PREBAKE"),
                "emission_factor": str(anode_ef.value)
            },
            output_value=anode_emissions,
            output_unit="tCO2e",
            source=anode_ef.source
        ))

        # Step 2: PFC emissions
        step_num += 1
        pfc_emissions = self._calculate_pfc_emissions(input_data)
        pfc_emissions = self._round_emissions(pfc_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="PFC emissions (CF4 + C2F6)",
            formula="(CF4_kg * GWP_CF4 + C2F6_kg * GWP_C2F6) / 1000",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "cf4_factor_kg_per_t": str(self.EF_CF4_PREBAKE),
                "c2f6_factor_kg_per_t": str(self.EF_C2F6_PREBAKE),
                "gwp_cf4": str(self.GWP_CF4),
                "gwp_c2f6": str(self.GWP_C2F6)
            },
            output_value=pfc_emissions,
            output_unit="tCO2e",
            source="IPCC 2006 + AR5 GWP"
        ))

        # Step 3: Electricity emissions
        step_num += 1
        electricity_emissions = self._calculate_electricity_emissions(input_data)
        electricity_emissions = self._round_emissions(electricity_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Electricity emissions (electrolysis)",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(
                    input_data.electricity_kwh or
                    input_data.production_tonnes * self.ELECTRICITY_PRIMARY
                ),
                "grid_factor": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=electricity_emissions,
            output_unit="tCO2e",
            source="Grid emission factor"
        ))

        # Step 4: Alumina emissions (if reported)
        step_num += 1
        alumina_emissions = Decimal("0")
        if input_data.alumina_tonnes:
            alumina_ef = self._emission_factors["alumina"]
            alumina_emissions = input_data.alumina_tonnes * alumina_ef.value
            alumina_emissions = self._round_emissions(alumina_emissions)
            factors_used.append(alumina_ef)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Alumina refining emissions",
            formula="alumina_tonnes * alumina_factor",
            inputs={
                "alumina_tonnes": str(input_data.alumina_tonnes or Decimal("0")),
                "alumina_factor": str(self.EF_ALUMINA)
            },
            output_value=alumina_emissions,
            output_unit="tCO2e",
            source="World Aluminium LCI"
        ))

        # Step 5: Total emissions
        step_num += 1
        scope_1 = anode_emissions + pfc_emissions + alumina_emissions
        scope_2 = electricity_emissions
        total_emissions = scope_1 + scope_2
        total_emissions = self._round_emissions(total_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions calculation",
            formula="anode + pfc + alumina + electricity",
            inputs={
                "anode": str(anode_emissions),
                "pfc": str(pfc_emissions),
                "alumina": str(alumina_emissions),
                "electricity": str(electricity_emissions)
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

        return AluminumMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            production_route=input_data.production_route.value,
            anode_technology=(
                input_data.anode_technology.value
                if input_data.anode_technology else None
            ),
            electrolysis_emissions_tco2e=electricity_emissions,
            anode_emissions_tco2e=anode_emissions,
            pfc_emissions_tco2e=pfc_emissions,
            alumina_emissions_tco2e=alumina_emissions,
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

    def _calculate_secondary(
        self,
        input_data: AluminumMRVInput,
        calc_id: str
    ) -> AluminumMRVOutput:
        """Calculate emissions for secondary (recycled) aluminum."""
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []

        secondary_ef = self._emission_factors["secondary"]
        factors_used.append(secondary_ef)

        # Direct emissions from remelting
        direct_emissions = input_data.production_tonnes * secondary_ef.value
        direct_emissions = self._round_emissions(direct_emissions)

        steps.append(CalculationStep(
            step_number=1,
            description="Secondary aluminum (remelting) emissions",
            formula="production_tonnes * secondary_factor",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "emission_factor": str(secondary_ef.value)
            },
            output_value=direct_emissions,
            output_unit="tCO2e",
            source=secondary_ef.source
        ))

        # Electricity emissions
        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh:
            electricity_kwh = input_data.electricity_kwh or (
                input_data.production_tonnes * self.ELECTRICITY_SECONDARY
            )
            electricity_emissions = (
                electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
            ) / Decimal("1000")
            electricity_emissions = self._round_emissions(electricity_emissions)

        steps.append(CalculationStep(
            step_number=2,
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

        total_emissions = direct_emissions + electricity_emissions
        emission_intensity = (
            total_emissions / input_data.production_tonnes
            if input_data.production_tonnes > 0 else Decimal("0")
        )

        cbam_output = self._create_cbam_output(
            production_tonnes=input_data.production_tonnes,
            direct_emissions=direct_emissions,
            indirect_emissions=electricity_emissions
        )

        return AluminumMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            production_route=input_data.production_route.value,
            scope_1_emissions_tco2e=direct_emissions,
            scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total_emissions),
            emission_intensity_tco2e_per_t=self._round_intensity(emission_intensity),
            cbam_output=cbam_output,
            calculation_steps=steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True
        )

    def _get_anode_emission_factor(self, input_data: AluminumMRVInput) -> EmissionFactor:
        """Get anode emission factor based on technology."""
        if input_data.production_route == AluminumProductionRoute.PRIMARY_SODERBERG:
            return self._emission_factors["anode_soderberg"]
        return self._emission_factors["anode_prebake"]

    def _calculate_pfc_emissions(self, input_data: AluminumMRVInput) -> Decimal:
        """Calculate PFC emissions (CF4 + C2F6)."""
        # Use measured values if available
        if input_data.cf4_emissions_kg is not None:
            cf4_kg = input_data.cf4_emissions_kg
            c2f6_kg = input_data.c2f6_emissions_kg or Decimal("0")
        else:
            # Use default factors
            if input_data.production_route == AluminumProductionRoute.PRIMARY_SODERBERG:
                cf4_kg = input_data.production_tonnes * self.EF_CF4_SODERBERG
                c2f6_kg = input_data.production_tonnes * self.EF_C2F6_SODERBERG
            else:
                cf4_kg = input_data.production_tonnes * self.EF_CF4_PREBAKE
                c2f6_kg = input_data.production_tonnes * self.EF_C2F6_PREBAKE

        # Convert to CO2e using GWP
        cf4_co2e = cf4_kg * self.GWP_CF4 / Decimal("1000")
        c2f6_co2e = c2f6_kg * self.GWP_C2F6 / Decimal("1000")

        return cf4_co2e + c2f6_co2e

    def _calculate_electricity_emissions(self, input_data: AluminumMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh
        if electricity_kwh is None:
            electricity_kwh = input_data.production_tonnes * self.ELECTRICITY_PRIMARY

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")
