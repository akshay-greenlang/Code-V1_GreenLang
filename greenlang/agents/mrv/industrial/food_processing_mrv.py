# -*- coding: utf-8 -*-
"""
GL-MRV-IND-007: Food Processing MRV Agent
==========================================

Industrial MRV agent for food processing sector emissions measurement, reporting,
and verification. Covers major food processing subsectors.

Subsectors:
    - Meat Processing
    - Dairy Processing
    - Beverages (non-alcoholic)
    - Grain Milling
    - Sugar Production
    - Fruit & Vegetable Processing
    - Frozen Foods

Sources:
    - IPCC 2006 Guidelines, Volume 3
    - EPA Food Processing Industry Guidelines
    - European Food and Drink Industry (FoodDrinkEurope)
    - FAO Food Industry Energy Guide

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


class FoodSubsector(str, Enum):
    """Food processing subsectors."""
    MEAT = "MEAT"
    DAIRY = "DAIRY"
    BEVERAGES = "BEVERAGES"
    GRAIN_MILLING = "GRAIN_MILLING"
    SUGAR = "SUGAR"
    FRUIT_VEGETABLE = "FRUIT_VEGETABLE"
    FROZEN_FOODS = "FROZEN_FOODS"
    BAKERY = "BAKERY"
    CONFECTIONERY = "CONFECTIONERY"


class FoodProcessingMRVInput(IndustrialMRVInput):
    """Input model for Food Processing MRV."""

    subsector: FoodSubsector = Field(
        ..., description="Food processing subsector"
    )

    # Process inputs
    steam_gj: Optional[Decimal] = Field(None, ge=0)
    refrigerant_kg: Optional[Decimal] = Field(None, ge=0)
    refrigerant_gwp: Optional[Decimal] = Field(
        default=Decimal("1430"),  # R-134a default
        ge=0,
        description="GWP of refrigerant used"
    )
    wastewater_m3: Optional[Decimal] = Field(None, ge=0)


class FoodProcessingMRVOutput(IndustrialMRVOutput):
    """Output model for Food Processing MRV."""

    subsector: str = Field(default="")

    # Emissions breakdown
    thermal_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    refrigerant_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    wastewater_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class FoodProcessingMRVAgent(IndustrialMRVBaseAgent[FoodProcessingMRVInput, FoodProcessingMRVOutput]):
    """
    GL-MRV-IND-007: Food Processing MRV Agent

    Emission Sources:
        - Thermal: Steam, heating, drying
        - Refrigeration: Cooling, freezing (refrigerant leakage)
        - Electricity: Motors, lighting, HVAC
        - Wastewater: Methane from treatment

    Emission Factors (tCO2e/t product):
        - Meat processing: 0.15-0.40
        - Dairy processing: 0.10-0.30
        - Beverages: 0.05-0.15
        - Frozen foods: 0.20-0.50
    """

    AGENT_ID = "GL-MRV-IND-007"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Food Processing"
    CBAM_CN_CODE = "0201-2106"
    CBAM_PRODUCT_CATEGORY = "Food Products"

    # Emission factors (tCO2e/t product)
    EMISSION_FACTORS = {
        FoodSubsector.MEAT: Decimal("0.25"),
        FoodSubsector.DAIRY: Decimal("0.20"),
        FoodSubsector.BEVERAGES: Decimal("0.10"),
        FoodSubsector.GRAIN_MILLING: Decimal("0.08"),
        FoodSubsector.SUGAR: Decimal("0.30"),
        FoodSubsector.FRUIT_VEGETABLE: Decimal("0.12"),
        FoodSubsector.FROZEN_FOODS: Decimal("0.35"),
        FoodSubsector.BAKERY: Decimal("0.15"),
        FoodSubsector.CONFECTIONERY: Decimal("0.18"),
    }

    # Electricity consumption (kWh/t product)
    ELECTRICITY_CONSUMPTION = {
        FoodSubsector.MEAT: Decimal("300"),
        FoodSubsector.DAIRY: Decimal("250"),
        FoodSubsector.BEVERAGES: Decimal("150"),
        FoodSubsector.FROZEN_FOODS: Decimal("500"),
    }

    # Refrigerant leakage rate (annual % of charge)
    REFRIGERANT_LEAKAGE_RATE = Decimal("0.10")  # 10%

    # Wastewater methane factor (kg CH4/m3)
    EF_WASTEWATER_CH4 = Decimal("0.5")
    GWP_CH4 = Decimal("28")

    def _load_emission_factors(self) -> None:
        """Load food processing emission factors."""
        self._emission_factors = {}
        for subsector, value in self.EMISSION_FACTORS.items():
            self._emission_factors[subsector.value] = EmissionFactor(
                factor_id=f"food_{subsector.value.lower()}",
                value=value,
                unit="tCO2e/t_product",
                source="FoodDrinkEurope / EPA Guidelines",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=25.0
            )

    def calculate_emissions(self, input_data: FoodProcessingMRVInput) -> FoodProcessingMRVOutput:
        """Calculate food processing emissions."""
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Get emission factor for subsector
        ef = self._emission_factors[input_data.subsector.value]
        factors_used.append(ef)

        # Step 1: Thermal/process emissions
        step_num += 1
        thermal_emissions = input_data.production_tonnes * ef.value
        thermal_emissions = self._round_emissions(thermal_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description=f"Thermal/process emissions for {input_data.subsector.value}",
            formula="production_tonnes * emission_factor",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "subsector": input_data.subsector.value,
                "emission_factor": str(ef.value)
            },
            output_value=thermal_emissions,
            output_unit="tCO2e",
            source=ef.source
        ))

        # Step 2: Refrigerant emissions
        step_num += 1
        refrigerant_emissions = self._calculate_refrigerant_emissions(input_data)
        refrigerant_emissions = self._round_emissions(refrigerant_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Refrigerant leakage emissions",
            formula="refrigerant_kg * leakage_rate * GWP / 1000",
            inputs={
                "refrigerant_kg": str(input_data.refrigerant_kg or Decimal("0")),
                "leakage_rate": str(self.REFRIGERANT_LEAKAGE_RATE),
                "gwp": str(input_data.refrigerant_gwp or Decimal("1430"))
            },
            output_value=refrigerant_emissions,
            output_unit="tCO2e",
            source="GHG Protocol refrigerant methodology"
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

        # Step 4: Wastewater emissions
        step_num += 1
        wastewater_emissions = self._calculate_wastewater_emissions(input_data)
        wastewater_emissions = self._round_emissions(wastewater_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Wastewater treatment emissions (CH4)",
            formula="wastewater_m3 * CH4_factor * GWP_CH4 / 1000",
            inputs={
                "wastewater_m3": str(input_data.wastewater_m3 or Decimal("0")),
                "ch4_factor": str(self.EF_WASTEWATER_CH4),
                "gwp_ch4": str(self.GWP_CH4)
            },
            output_value=wastewater_emissions,
            output_unit="tCO2e",
            source="IPCC wastewater guidelines"
        ))

        # Totals
        scope_1 = thermal_emissions + refrigerant_emissions + wastewater_emissions
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

        return FoodProcessingMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            subsector=input_data.subsector.value,
            thermal_emissions_tco2e=thermal_emissions,
            refrigerant_emissions_tco2e=refrigerant_emissions,
            wastewater_emissions_tco2e=wastewater_emissions,
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

    def _calculate_refrigerant_emissions(self, input_data: FoodProcessingMRVInput) -> Decimal:
        """Calculate refrigerant leakage emissions."""
        if input_data.refrigerant_kg is None:
            return Decimal("0")

        leakage_kg = input_data.refrigerant_kg * self.REFRIGERANT_LEAKAGE_RATE
        gwp = input_data.refrigerant_gwp or Decimal("1430")
        return leakage_kg * gwp / Decimal("1000")

    def _calculate_electricity_emissions(self, input_data: FoodProcessingMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh
        if electricity_kwh is None:
            default_consumption = self.ELECTRICITY_CONSUMPTION.get(
                input_data.subsector, Decimal("200")
            )
            electricity_kwh = input_data.production_tonnes * default_consumption

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")

    def _calculate_wastewater_emissions(self, input_data: FoodProcessingMRVInput) -> Decimal:
        """Calculate wastewater treatment emissions."""
        if input_data.wastewater_m3 is None:
            return Decimal("0")

        ch4_kg = input_data.wastewater_m3 * self.EF_WASTEWATER_CH4
        return ch4_kg * self.GWP_CH4 / Decimal("1000")
