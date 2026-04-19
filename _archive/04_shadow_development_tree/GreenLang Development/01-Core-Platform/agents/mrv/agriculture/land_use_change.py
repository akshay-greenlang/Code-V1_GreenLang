# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-004: Land Use Change MRV Agent
=========================================

Calculates CO2 emissions from land use change and land management.

Reference: IPCC 2006 Guidelines, Volume 4, Chapter 2
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
    ClimateZone,
)

logger = logging.getLogger(__name__)


class LandUseCategory(str, Enum):
    """Land use categories per IPCC."""
    FOREST = "forest"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER = "other"


# Carbon stock change factors (tonnes C/ha) - simplified IPCC defaults
CARBON_STOCK: Dict[str, Dict[str, Decimal]] = {
    "tropical": {
        LandUseCategory.FOREST.value: Decimal("120"),
        LandUseCategory.CROPLAND.value: Decimal("45"),
        LandUseCategory.GRASSLAND.value: Decimal("65"),
    },
    "temperate": {
        LandUseCategory.FOREST.value: Decimal("80"),
        LandUseCategory.CROPLAND.value: Decimal("35"),
        LandUseCategory.GRASSLAND.value: Decimal("50"),
    },
    "default": {
        LandUseCategory.FOREST.value: Decimal("100"),
        LandUseCategory.CROPLAND.value: Decimal("40"),
        LandUseCategory.GRASSLAND.value: Decimal("55"),
        LandUseCategory.WETLAND.value: Decimal("200"),
        LandUseCategory.SETTLEMENT.value: Decimal("20"),
        LandUseCategory.OTHER.value: Decimal("30"),
    },
}


class LandUseRecord(BaseModel):
    """Land use change record."""

    record_id: Optional[str] = Field(None, description="Record ID")
    from_land_use: LandUseCategory = Field(..., description="Original land use")
    to_land_use: LandUseCategory = Field(..., description="New land use")
    area_hectares: Decimal = Field(..., ge=0, description="Area converted")
    conversion_year: Optional[int] = Field(None, description="Year of conversion")

    class Config:
        use_enum_values = True


class LandUseChangeInput(AgricultureMRVInput):
    """Input for Land Use Change MRV Agent."""

    land_changes: List[LandUseRecord] = Field(
        default_factory=list, description="Land use change records"
    )
    transition_period_years: int = Field(20, description="Carbon transition period")


class LandUseChangeOutput(AgricultureMRVOutput):
    """Output for Land Use Change MRV Agent."""

    total_area_changed_ha: Decimal = Field(Decimal("0"), description="Total area changed")
    carbon_stock_change_tonnes_c: Decimal = Field(Decimal("0"), description="Carbon stock change")
    emissions_by_transition: Dict[str, Decimal] = Field(default_factory=dict)


class LandUseChangeMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-004: Land Use Change MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-004"
    AGENT_NAME = "Land Use Change MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.LAND_USE_CHANGE
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: LandUseChangeInput) -> LandUseChangeOutput:
        """Calculate land use change emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        total_area = Decimal("0")
        total_carbon_change = Decimal("0")
        emissions_by_trans: Dict[str, Decimal] = {}

        # Determine climate
        climate = "default"
        if hasattr(input_data, 'climate_zone'):
            cz = input_data.climate_zone.value if hasattr(input_data.climate_zone, 'value') else str(input_data.climate_zone)
            if "tropical" in cz:
                climate = "tropical"
            elif "temperate" in cz:
                climate = "temperate"

        stocks = CARBON_STOCK.get(climate, CARBON_STOCK["default"])

        for record in input_data.land_changes:
            from_use = record.from_land_use.value if hasattr(record.from_land_use, 'value') else str(record.from_land_use)
            to_use = record.to_land_use.value if hasattr(record.to_land_use, 'value') else str(record.to_land_use)

            from_stock = stocks.get(from_use, Decimal("40"))
            to_stock = stocks.get(to_use, Decimal("40"))

            # Carbon change (negative = emission)
            delta_c = (to_stock - from_stock) * record.area_hectares

            # Annualize over transition period
            annual_delta_c = delta_c / Decimal(str(input_data.transition_period_years))

            total_area += record.area_hectares
            total_carbon_change += annual_delta_c

            # Track by transition
            trans_key = f"{from_use}_to_{to_use}"
            # Convert C to CO2 (44/12)
            co2_emissions = -annual_delta_c * Decimal("44") / Decimal("12")
            if co2_emissions > 0:
                emissions_by_trans[trans_key] = emissions_by_trans.get(
                    trans_key, Decimal("0")
                ) + co2_emissions

        # Total CO2 (emissions are positive when carbon is lost)
        total_co2 = (-total_carbon_change * Decimal("44") / Decimal("12")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        if total_co2 < 0:
            total_co2 = Decimal("0")  # Net sequestration - report as zero emissions

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_area_ha": str(total_area),
        }

        base_output = self._create_output(
            co2_kg=total_co2 * Decimal("1000"),  # Convert tonnes to kg
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=[],
            activity_summary=activity_summary,
            start_time=start_time,
            warnings=warnings,
        )

        return LandUseChangeOutput(
            **base_output.dict(),
            total_area_changed_ha=total_area,
            carbon_stock_change_tonnes_c=total_carbon_change,
            emissions_by_transition=emissions_by_trans,
        )
