# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-006: Agricultural Machinery MRV Agent
================================================

Calculates CO2 emissions from farm equipment and machinery.

Reference: DEFRA Conversion Factors, EPA Mobile Source Guidance
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


class MachineryType(str, Enum):
    """Agricultural machinery types."""
    TRACTOR_SMALL = "tractor_small"  # <50 hp
    TRACTOR_MEDIUM = "tractor_medium"  # 50-100 hp
    TRACTOR_LARGE = "tractor_large"  # >100 hp
    COMBINE_HARVESTER = "combine_harvester"
    SPRAYER = "sprayer"
    IRRIGATION_PUMP = "irrigation_pump"
    GRAIN_DRYER = "grain_dryer"
    TRUCK_FARM = "truck_farm"
    ATV = "atv"
    OTHER = "other"


class FuelType(str, Enum):
    """Fuel types for farm machinery."""
    DIESEL = "diesel"
    PETROL = "petrol"
    LPG = "lpg"
    BIODIESEL = "biodiesel"
    ELECTRICITY = "electricity"


# Emission factors (kg CO2e / liter)
FUEL_FACTORS: Dict[str, Decimal] = {
    FuelType.DIESEL.value: Decimal("2.68787"),
    FuelType.PETROL.value: Decimal("2.31463"),
    FuelType.LPG.value: Decimal("1.55377"),
    FuelType.BIODIESEL.value: Decimal("0.134"),  # Biogenic CO2 excluded
    FuelType.ELECTRICITY.value: Decimal("0.207"),  # kg CO2e / kWh
}


class MachineryRecord(BaseModel):
    """Machinery record."""

    machinery_id: Optional[str] = Field(None, description="Machinery ID")
    machinery_type: MachineryType = Field(..., description="Machinery type")
    fuel_type: FuelType = Field(FuelType.DIESEL, description="Fuel type")
    fuel_consumed_liters: Optional[Decimal] = Field(None, ge=0, description="Fuel in liters")
    electricity_kwh: Optional[Decimal] = Field(None, ge=0, description="Electricity in kWh")
    operating_hours: Optional[Decimal] = Field(None, ge=0, description="Operating hours")

    class Config:
        use_enum_values = True


class AgriculturalMachineryInput(AgricultureMRVInput):
    """Input for Agricultural Machinery MRV Agent."""

    machinery: List[MachineryRecord] = Field(
        default_factory=list, description="Machinery records"
    )


class AgriculturalMachineryOutput(AgricultureMRVOutput):
    """Output for Agricultural Machinery MRV Agent."""

    total_fuel_liters: Decimal = Field(Decimal("0"), description="Total fuel")
    total_electricity_kwh: Decimal = Field(Decimal("0"), description="Total electricity")
    emissions_by_machinery: Dict[str, Decimal] = Field(default_factory=dict)


class AgriculturalMachineryMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-006: Agricultural Machinery MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-006"
    AGENT_NAME = "Agricultural Machinery MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.AGRICULTURAL_MACHINERY
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: AgriculturalMachineryInput) -> AgriculturalMachineryOutput:
        """Calculate machinery emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        total_fuel = Decimal("0")
        total_elec = Decimal("0")
        total_co2 = Decimal("0")
        emissions_by_mach: Dict[str, Decimal] = {}

        for mach in input_data.machinery:
            mtype = mach.machinery_type.value if hasattr(mach.machinery_type, 'value') else str(mach.machinery_type)
            ftype = mach.fuel_type.value if hasattr(mach.fuel_type, 'value') else str(mach.fuel_type)

            mach_co2 = Decimal("0")

            if mach.fuel_consumed_liters and ftype != FuelType.ELECTRICITY.value:
                ef = FUEL_FACTORS.get(ftype, FUEL_FACTORS[FuelType.DIESEL.value])
                mach_co2 = (mach.fuel_consumed_liters * ef).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_fuel += mach.fuel_consumed_liters

            if mach.electricity_kwh or ftype == FuelType.ELECTRICITY.value:
                kwh = mach.electricity_kwh or Decimal("0")
                mach_co2 += (kwh * FUEL_FACTORS[FuelType.ELECTRICITY.value]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_elec += kwh

            total_co2 += mach_co2
            emissions_by_mach[mtype] = emissions_by_mach.get(
                mtype, Decimal("0")
            ) + mach_co2

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_fuel_liters": str(total_fuel),
            "total_electricity_kwh": str(total_elec),
        }

        base_output = self._create_output(
            co2_kg=total_co2,
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=[],
            activity_summary=activity_summary,
            start_time=start_time,
            warnings=warnings,
        )

        return AgriculturalMachineryOutput(
            **base_output.dict(),
            total_fuel_liters=total_fuel,
            total_electricity_kwh=total_elec,
            emissions_by_machinery=emissions_by_mach,
        )
