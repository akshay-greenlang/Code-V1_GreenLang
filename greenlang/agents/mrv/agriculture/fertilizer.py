# -*- coding: utf-8 -*-
"""
GL-MRV-AGR-003: Fertilizer MRV Agent
====================================

Calculates N2O emissions from synthetic and organic fertilizer application.

Reference: IPCC 2006 Guidelines, Volume 4, Chapter 11
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
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


class FertilizerType(str, Enum):
    """Fertilizer types."""
    UREA = "urea"
    AMMONIUM_NITRATE = "ammonium_nitrate"
    AMMONIUM_SULFATE = "ammonium_sulfate"
    CALCIUM_AMMONIUM_NITRATE = "calcium_ammonium_nitrate"
    DAP = "dap"
    MAP = "map"
    NPK = "npk"
    ORGANIC_MANURE = "organic_manure"
    ORGANIC_COMPOST = "organic_compost"
    ORGANIC_SLURRY = "organic_slurry"


# N content by fertilizer type (kg N / kg fertilizer)
FERTILIZER_N_CONTENT: Dict[str, Decimal] = {
    FertilizerType.UREA.value: Decimal("0.46"),
    FertilizerType.AMMONIUM_NITRATE.value: Decimal("0.34"),
    FertilizerType.AMMONIUM_SULFATE.value: Decimal("0.21"),
    FertilizerType.CALCIUM_AMMONIUM_NITRATE.value: Decimal("0.27"),
    FertilizerType.DAP.value: Decimal("0.18"),
    FertilizerType.MAP.value: Decimal("0.11"),
    FertilizerType.NPK.value: Decimal("0.15"),
    FertilizerType.ORGANIC_MANURE.value: Decimal("0.02"),
    FertilizerType.ORGANIC_COMPOST.value: Decimal("0.015"),
    FertilizerType.ORGANIC_SLURRY.value: Decimal("0.005"),
}

# IPCC default emission factors
EF_DIRECT_N2O = Decimal("0.01")  # kg N2O-N / kg N applied
EF_INDIRECT_VOLATILIZATION = Decimal("0.01")  # kg N2O-N / kg N volatilized
EF_INDIRECT_LEACHING = Decimal("0.0075")  # kg N2O-N / kg N leached
FRAC_VOLATILIZATION = Decimal("0.10")  # fraction N volatilized
FRAC_LEACHING = Decimal("0.30")  # fraction N leached


class FertilizerRecord(BaseModel):
    """Fertilizer application record."""

    record_id: Optional[str] = Field(None, description="Record ID")
    fertilizer_type: FertilizerType = Field(..., description="Fertilizer type")
    amount_kg: Decimal = Field(..., ge=0, description="Amount applied in kg")
    n_content_fraction: Optional[Decimal] = Field(
        None, ge=0, le=1, description="N content if known"
    )
    area_hectares: Optional[Decimal] = Field(
        None, ge=0, description="Application area"
    )

    class Config:
        use_enum_values = True


class FertilizerInput(AgricultureMRVInput):
    """Input for Fertilizer MRV Agent."""

    fertilizers: List[FertilizerRecord] = Field(
        default_factory=list, description="Fertilizer applications"
    )
    include_direct: bool = Field(True, description="Include direct N2O")
    include_indirect: bool = Field(True, description="Include indirect N2O")


class FertilizerOutput(AgricultureMRVOutput):
    """Output for Fertilizer MRV Agent."""

    total_n_applied_kg: Decimal = Field(Decimal("0"), description="Total N applied")
    direct_n2o_kg: Decimal = Field(Decimal("0"), description="Direct N2O emissions")
    indirect_n2o_kg: Decimal = Field(Decimal("0"), description="Indirect N2O emissions")
    emissions_by_fertilizer: Dict[str, Decimal] = Field(default_factory=dict)


class FertilizerMRVAgent(BaseAgricultureMRVAgent):
    """GL-MRV-AGR-003: Fertilizer MRV Agent"""

    AGENT_ID = "GL-MRV-AGR-003"
    AGENT_NAME = "Fertilizer MRV Agent"
    AGENT_VERSION = "1.0.0"
    SECTOR = AgricultureSector.FERTILIZER
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: FertilizerInput) -> FertilizerOutput:
        """Calculate fertilizer N2O emissions."""
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        total_n = Decimal("0")
        direct_n2o = Decimal("0")
        indirect_n2o = Decimal("0")
        emissions_by_fert: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize fertilizer emissions calculation",
            inputs={"num_records": len(input_data.fertilizers)},
        ))

        for fert in input_data.fertilizers:
            ftype = fert.fertilizer_type.value if hasattr(fert.fertilizer_type, 'value') else str(fert.fertilizer_type)

            # Get N content
            n_content = fert.n_content_fraction or FERTILIZER_N_CONTENT.get(
                ftype, Decimal("0.15")
            )
            n_applied = fert.amount_kg * n_content
            total_n += n_applied

            fert_direct = Decimal("0")
            fert_indirect = Decimal("0")

            # Direct N2O
            if input_data.include_direct:
                n2o_n_direct = n_applied * EF_DIRECT_N2O
                fert_direct = (n2o_n_direct * Decimal("44") / Decimal("28")).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                direct_n2o += fert_direct

            # Indirect N2O
            if input_data.include_indirect:
                # Volatilization
                n_vol = n_applied * FRAC_VOLATILIZATION
                n2o_n_vol = n_vol * EF_INDIRECT_VOLATILIZATION

                # Leaching
                n_leach = n_applied * FRAC_LEACHING
                n2o_n_leach = n_leach * EF_INDIRECT_LEACHING

                fert_indirect = ((n2o_n_vol + n2o_n_leach) * Decimal("44") / Decimal("28")).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                indirect_n2o += fert_indirect

            # Track by type
            fert_co2e = (fert_direct + fert_indirect) * self.gwp["N2O"]
            emissions_by_fert[ftype] = emissions_by_fert.get(
                ftype, Decimal("0")
            ) + fert_co2e

        total_n2o = direct_n2o + indirect_n2o

        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_n_applied_kg": str(total_n),
        }

        base_output = self._create_output(
            co2_kg=Decimal("0"),
            ch4_kg=Decimal("0"),
            n2o_kg=total_n2o,
            steps=steps,
            emission_factors=emission_factors,
            activity_summary=activity_summary,
            start_time=start_time,
            warnings=warnings,
        )

        return FertilizerOutput(
            **base_output.dict(),
            total_n_applied_kg=total_n,
            direct_n2o_kg=direct_n2o,
            indirect_n2o_kg=indirect_n2o,
            emissions_by_fertilizer=emissions_by_fert,
        )
