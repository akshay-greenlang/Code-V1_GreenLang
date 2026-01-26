# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-007: Irrigation MRV Agent
====================================

Calculates CO2 emissions from irrigation energy consumption.

Reference: DEFRA, FAO Guidelines
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


class IrrigationType(str, Enum):
    """Irrigation system types."""
    FLOOD = "flood"
    FURROW = "furrow"
    SPRINKLER = "sprinkler"
    CENTER_PIVOT = "center_pivot"
    DRIP = "drip"
    SUBSURFACE_DRIP = "subsurface_drip"
    MANUAL = "manual"


class WaterSource(str, Enum):
    """Water source for irrigation."""
    GROUNDWATER = "groundwater"
    SURFACE_WATER = "surface_water"
    RECYCLED = "recycled"
    MUNICIPAL = "municipal"


class EnergySource(str, Enum):
    """Energy source for pumping."""
    ELECTRICITY = "electricity"
    DIESEL = "diesel"
    SOLAR = "solar"
    GRAVITY = "gravity"


# Energy intensity factors (kWh / m3 water)
PUMPING_ENERGY: Dict[str, Decimal] = {
    WaterSource.GROUNDWATER.value: Decimal("0.50"),  # Varies with depth
    WaterSource.SURFACE_WATER.value: Decimal("0.20"),
    WaterSource.RECYCLED.value: Decimal("0.30"),
    WaterSource.MUNICIPAL.value: Decimal("0.10"),
}

# Grid emission factor (kg CO2e / kWh) - default
GRID_EF = Decimal("0.207")

# Diesel emission factor for pumps (kg CO2e / liter)
DIESEL_EF = Decimal("2.68787")


class IrrigationRecord(BaseModel):
    """Irrigation record."""

    field_id: Optional[str] = Field(None, description="Field ID")
    irrigation_type: IrrigationType = Field(
        IrrigationType.SPRINKLER, description="Irrigation type"
    )
    water_source: WaterSource = Field(
        WaterSource.GROUNDWATER, description="Water source"
    )
    energy_source: EnergySource = Field(
        EnergySource.ELECTRICITY, description="Energy source"
    )
    water_volume_m3: Optional[Decimal] = Field(None, ge=0, description="Water volume")
    electricity_kwh: Optional[Decimal] = Field(None, ge=0, description="Electricity")
    diesel_liters: Optional[Decimal] = Field(None, ge=0, description="Diesel")
    area_hectares: Optional[Decimal] = Field(None, ge=0, description="Irrigated area")

    class Config:
        use_enum_values = True


class IrrigationInput(AgricultureMRVInput):
    """Input for Irrigation MRV Agent."""

    irrigation: List[IrrigationRecord] = Field(
        default_factory=list, description="Irrigation records"
    )
    grid_emission_factor: Decimal = Field(
        GRID_EF, description="Grid emission factor"
    )


class IrrigationOutput(AgricultureMRVOutput):
    """Output for Irrigation MRV Agent."""

    total_water_m3: Decimal = Field(Decimal("0"), description="Total water")
    total_electricity_kwh: Decimal = Field(Decimal("0"), description="Electricity")
    total_diesel_liters: Decimal = Field(Decimal("0"), description="Diesel")
    emissions_by_source: Dict[str, Decimal] = Field(default_factory=dict)


class IrrigationMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-007: Irrigation MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-007"
    AGENT_NAME = "Irrigation MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.IRRIGATION
    DEFAULT_SCOPE = EmissionScope.SCOPE_2

    def calculate(self, input_data: IrrigationInput) -> IrrigationOutput:
        """Calculate irrigation emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        total_water = Decimal("0")
        total_elec = Decimal("0")
        total_diesel = Decimal("0")
        total_co2 = Decimal("0")
        emissions_by_src: Dict[str, Decimal] = {}

        for irr in input_data.irrigation:
            esrc = irr.energy_source.value if hasattr(irr.energy_source, 'value') else str(irr.energy_source)
            wsrc = irr.water_source.value if hasattr(irr.water_source, 'value') else str(irr.water_source)

            irr_co2 = Decimal("0")

            # Calculate from electricity
            if irr.electricity_kwh:
                irr_co2 = (irr.electricity_kwh * input_data.grid_emission_factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_elec += irr.electricity_kwh

            # Calculate from diesel
            elif irr.diesel_liters:
                irr_co2 = (irr.diesel_liters * DIESEL_EF).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_diesel += irr.diesel_liters

            # Estimate from water volume
            elif irr.water_volume_m3:
                pump_energy = PUMPING_ENERGY.get(wsrc, Decimal("0.35"))
                energy_kwh = irr.water_volume_m3 * pump_energy

                if esrc == EnergySource.ELECTRICITY.value:
                    irr_co2 = (energy_kwh * input_data.grid_emission_factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_elec += energy_kwh
                elif esrc == EnergySource.DIESEL.value:
                    # Convert kWh to liters (approx 0.25 L/kWh for diesel pump)
                    liters = energy_kwh * Decimal("0.25")
                    irr_co2 = (liters * DIESEL_EF).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_diesel += liters

            if irr.water_volume_m3:
                total_water += irr.water_volume_m3

            total_co2 += irr_co2
            emissions_by_src[esrc] = emissions_by_src.get(
                esrc, Decimal("0")
            ) + irr_co2

        # Scope depends on energy source
        scope = EmissionScope.SCOPE_1 if total_diesel > 0 else EmissionScope.SCOPE_2

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_water_m3": str(total_water),
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

        return IrrigationOutput(
            **base_output.dict(),
            total_water_m3=total_water,
            total_electricity_kwh=total_elec,
            total_diesel_liters=total_diesel,
            emissions_by_source=emissions_by_src,
        )
