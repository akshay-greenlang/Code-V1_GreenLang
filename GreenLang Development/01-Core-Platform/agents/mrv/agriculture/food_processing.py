# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-008: Food Processing MRV Agent
=========================================

Calculates emissions from food processing operations.

Reference: GHG Protocol, DEFRA Conversion Factors
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from greenlang.agents.mrv.agriculture.base import (
    BaseAgricultureMRVAgent,
    AgricultureMRVInput,
    AgricultureMRVOutput,
    AgricultureSector,
    EmissionScope,
    CalculationStep,
)

logger = logging.getLogger(__name__)


class ProcessingType(str, Enum):
    """Food processing types."""
    GRAIN_MILLING = "grain_milling"
    FRUIT_VEGETABLE = "fruit_vegetable"
    MEAT_PROCESSING = "meat_processing"
    DAIRY_PROCESSING = "dairy_processing"
    BEVERAGE = "beverage"
    BAKING = "baking"
    SUGAR_PROCESSING = "sugar_processing"
    OIL_EXTRACTION = "oil_extraction"
    CANNING = "canning"
    FREEZING = "freezing"
    DRYING = "drying"
    OTHER = "other"


class EnergyType(str, Enum):
    """Energy types for processing."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    COAL = "coal"
    BIOMASS = "biomass"
    STEAM = "steam"


# Emission factors
ENERGY_FACTORS: Dict[str, Decimal] = {
    EnergyType.ELECTRICITY.value: Decimal("0.207"),  # kg CO2e/kWh
    EnergyType.NATURAL_GAS.value: Decimal("2.02"),  # kg CO2e/m3
    EnergyType.DIESEL.value: Decimal("2.688"),  # kg CO2e/liter
    EnergyType.COAL.value: Decimal("2.42"),  # kg CO2e/kg
    EnergyType.BIOMASS.value: Decimal("0.015"),  # kg CO2e/kg (low biogenic)
    EnergyType.STEAM.value: Decimal("0.20"),  # kg CO2e/kg steam
}


class ProcessingRecord(BaseModel):
    """Processing operation record."""

    facility_id: Optional[str] = Field(None, description="Facility ID")
    processing_type: ProcessingType = Field(..., description="Processing type")

    # Energy consumption
    electricity_kwh: Optional[Decimal] = Field(None, ge=0, description="Electricity")
    natural_gas_m3: Optional[Decimal] = Field(None, ge=0, description="Natural gas")
    diesel_liters: Optional[Decimal] = Field(None, ge=0, description="Diesel")
    coal_kg: Optional[Decimal] = Field(None, ge=0, description="Coal")
    biomass_kg: Optional[Decimal] = Field(None, ge=0, description="Biomass")
    steam_kg: Optional[Decimal] = Field(None, ge=0, description="Steam purchased")

    # Production
    production_tonnes: Optional[Decimal] = Field(None, ge=0, description="Production")

    class Config:
        use_enum_values = True


class FoodProcessingInput(AgricultureMRVInput):
    """Input for Food Processing MRV Agent."""

    processing: List[ProcessingRecord] = Field(
        default_factory=list, description="Processing records"
    )
    grid_emission_factor: Decimal = Field(
        Decimal("0.207"), description="Grid emission factor"
    )


class FoodProcessingOutput(AgricultureMRVOutput):
    """Output for Food Processing MRV Agent."""

    total_production_tonnes: Decimal = Field(Decimal("0"), description="Total production")
    total_electricity_kwh: Decimal = Field(Decimal("0"), description="Electricity")
    total_natural_gas_m3: Decimal = Field(Decimal("0"), description="Natural gas")
    emissions_per_tonne: Optional[Decimal] = Field(None, description="kg CO2e/tonne")
    emissions_by_process: Dict[str, Decimal] = Field(default_factory=dict)


class FoodProcessingMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-008: Food Processing MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-008"
    AGENT_NAME = "Food Processing MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.FOOD_PROCESSING
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: FoodProcessingInput) -> FoodProcessingOutput:
        """Calculate food processing emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        total_production = Decimal("0")
        total_elec = Decimal("0")
        total_gas = Decimal("0")
        total_co2 = Decimal("0")
        scope_1 = Decimal("0")
        scope_2 = Decimal("0")
        emissions_by_proc: Dict[str, Decimal] = {}

        for proc in input_data.processing:
            ptype = proc.processing_type.value if hasattr(proc.processing_type, 'value') else str(proc.processing_type)

            proc_co2 = Decimal("0")

            # Electricity (Scope 2)
            if proc.electricity_kwh:
                elec_co2 = (proc.electricity_kwh * input_data.grid_emission_factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                proc_co2 += elec_co2
                scope_2 += elec_co2
                total_elec += proc.electricity_kwh

            # Natural gas (Scope 1)
            if proc.natural_gas_m3:
                gas_co2 = (proc.natural_gas_m3 * ENERGY_FACTORS[EnergyType.NATURAL_GAS.value]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                proc_co2 += gas_co2
                scope_1 += gas_co2
                total_gas += proc.natural_gas_m3

            # Diesel (Scope 1)
            if proc.diesel_liters:
                diesel_co2 = (proc.diesel_liters * ENERGY_FACTORS[EnergyType.DIESEL.value]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                proc_co2 += diesel_co2
                scope_1 += diesel_co2

            # Coal (Scope 1)
            if proc.coal_kg:
                coal_co2 = (proc.coal_kg * ENERGY_FACTORS[EnergyType.COAL.value]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                proc_co2 += coal_co2
                scope_1 += coal_co2

            # Purchased steam (Scope 2)
            if proc.steam_kg:
                steam_co2 = (proc.steam_kg * ENERGY_FACTORS[EnergyType.STEAM.value]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                proc_co2 += steam_co2
                scope_2 += steam_co2

            if proc.production_tonnes:
                total_production += proc.production_tonnes

            total_co2 += proc_co2
            emissions_by_proc[ptype] = emissions_by_proc.get(
                ptype, Decimal("0")
            ) + proc_co2

        # Calculate intensity
        emissions_per_tonne = None
        if total_production > 0:
            emissions_per_tonne = (total_co2 / total_production).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Determine primary scope
        scope = EmissionScope.SCOPE_1 if scope_1 > scope_2 else EmissionScope.SCOPE_2

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_production_tonnes": str(total_production),
        }

        base_output = self._create_output(
            co2_kg=total_co2,
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=[],
            activity_summary=activity_summary,
            start_time=start_time,
            scope=scope,
            warnings=warnings,
        )

        return FoodProcessingOutput(
            **base_output.dict(),
            total_production_tonnes=total_production,
            total_electricity_kwh=total_elec,
            total_natural_gas_m3=total_gas,
            emissions_per_tonne=emissions_per_tonne,
            emissions_by_process=emissions_by_proc,
        )
