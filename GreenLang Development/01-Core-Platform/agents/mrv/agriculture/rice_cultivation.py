# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-005: Rice Cultivation MRV Agent
==========================================

Calculates CH4 emissions from flooded rice paddies.

Reference: IPCC 2006 Guidelines, Volume 4, Chapter 5
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


class WaterRegime(str, Enum):
    """Rice field water management regimes."""
    CONTINUOUSLY_FLOODED = "continuously_flooded"
    INTERMITTENT_SINGLE = "intermittent_single"
    INTERMITTENT_MULTIPLE = "intermittent_multiple"
    RAINFED_REGULAR = "rainfed_regular"
    RAINFED_DROUGHT = "rainfed_drought"
    DEEPWATER = "deepwater"
    UPLAND = "upland"


class OrganicAmendment(str, Enum):
    """Organic amendment types."""
    NONE = "none"
    STRAW_SHORT = "straw_short"  # <30 days before cultivation
    STRAW_LONG = "straw_long"  # >30 days before cultivation
    COMPOST = "compost"
    FARMYARD_MANURE = "farmyard_manure"
    GREEN_MANURE = "green_manure"


# IPCC 2006 baseline emission factor (kg CH4/ha/day)
EF_BASELINE = Decimal("1.30")

# Scaling factors for water regime
SF_WATER: Dict[str, Decimal] = {
    WaterRegime.CONTINUOUSLY_FLOODED.value: Decimal("1.0"),
    WaterRegime.INTERMITTENT_SINGLE.value: Decimal("0.60"),
    WaterRegime.INTERMITTENT_MULTIPLE.value: Decimal("0.52"),
    WaterRegime.RAINFED_REGULAR.value: Decimal("0.28"),
    WaterRegime.RAINFED_DROUGHT.value: Decimal("0.25"),
    WaterRegime.DEEPWATER.value: Decimal("0.31"),
    WaterRegime.UPLAND.value: Decimal("0"),
}

# Scaling factors for organic amendments
SF_ORGANIC: Dict[str, Decimal] = {
    OrganicAmendment.NONE.value: Decimal("1.0"),
    OrganicAmendment.STRAW_SHORT.value: Decimal("2.0"),
    OrganicAmendment.STRAW_LONG.value: Decimal("1.0"),
    OrganicAmendment.COMPOST.value: Decimal("1.0"),
    OrganicAmendment.FARMYARD_MANURE.value: Decimal("1.6"),
    OrganicAmendment.GREEN_MANURE.value: Decimal("1.2"),
}


class RiceFieldRecord(BaseModel):
    """Rice field record."""

    field_id: Optional[str] = Field(None, description="Field ID")
    area_hectares: Decimal = Field(..., ge=0, description="Field area")
    cultivation_days: int = Field(120, ge=1, le=365, description="Cultivation period")
    water_regime: WaterRegime = Field(
        WaterRegime.CONTINUOUSLY_FLOODED, description="Water regime"
    )
    organic_amendment: OrganicAmendment = Field(
        OrganicAmendment.NONE, description="Organic amendment"
    )
    amendment_rate_tonnes_ha: Decimal = Field(
        Decimal("0"), ge=0, description="Amendment rate"
    )

    class Config:
        use_enum_values = True


class RiceCultivationInput(AgricultureMRVInput):
    """Input for Rice Cultivation MRV Agent."""

    rice_fields: List[RiceFieldRecord] = Field(
        default_factory=list, description="Rice field records"
    )


class RiceCultivationOutput(AgricultureMRVOutput):
    """Output for Rice Cultivation MRV Agent."""

    total_area_hectares: Decimal = Field(Decimal("0"), description="Total rice area")
    total_cultivation_ha_days: Decimal = Field(Decimal("0"), description="Total ha-days")
    rice_ch4_kg: Decimal = Field(Decimal("0"), description="Rice CH4 emissions")


class RiceCultivationMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-005: Rice Cultivation MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-005"
    AGENT_NAME = "Rice Cultivation MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.RICE_CULTIVATION
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: RiceCultivationInput) -> RiceCultivationOutput:
        """Calculate rice cultivation CH4 emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        total_area = Decimal("0")
        total_ha_days = Decimal("0")
        total_ch4 = Decimal("0")

        for field in input_data.rice_fields:
            water = field.water_regime.value if hasattr(field.water_regime, 'value') else str(field.water_regime)
            organic = field.organic_amendment.value if hasattr(field.organic_amendment, 'value') else str(field.organic_amendment)

            sf_w = SF_WATER.get(water, Decimal("1.0"))
            sf_o = SF_ORGANIC.get(organic, Decimal("1.0"))

            # IPCC formula: CH4 = A x t x EF x SF_w x SF_o
            ha_days = field.area_hectares * Decimal(str(field.cultivation_days))
            ch4 = (field.area_hectares * Decimal(str(field.cultivation_days)) *
                   EF_BASELINE * sf_w * sf_o).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            total_area += field.area_hectares
            total_ha_days += ha_days
            total_ch4 += ch4

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_area_ha": str(total_area),
        }

        base_output = self._create_output(
            co2_kg=Decimal("0"),
            ch4_kg=total_ch4,
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=[],
            activity_summary=activity_summary,
            start_time=start_time,
            warnings=warnings,
        )

        return RiceCultivationOutput(
            **base_output.dict(),
            total_area_hectares=total_area,
            total_cultivation_ha_days=total_ha_days,
            rice_ch4_kg=total_ch4,
        )
