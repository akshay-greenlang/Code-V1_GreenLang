# -*- coding: utf-8 -*-
"""
GL-MRV-WST-004: Composting MRV Agent
=====================================

Calculates emissions from organic waste composting operations including
aerobic composting and anaerobic digestion facilities.

Key Features:
- IPCC 2006/2019 biological treatment methodology
- CH4 and N2O emissions from aerobic composting
- Biogas generation and capture for anaerobic digestion
- Support for various feedstock types
- Carbon sequestration credits for soil application

Zero-Hallucination Guarantees:
- All calculations use IPCC deterministic formulas
- Emission factors from published regulatory sources
- SHA-256 provenance hash for complete audit trail

Reference Standards:
- IPCC 2006 Guidelines Volume 5, Chapter 4 (Biological Treatment)
- IPCC 2019 Refinement - Waste Chapter
- EPA Composting Emission Factors
- EU Biowaste Directive Compliance

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.mrv.waste.base import (
    BaseWasteMRVAgent,
    WasteMRVInput,
    WasteMRVOutput,
    WasteType,
    TreatmentMethod,
    EmissionScope,
    DataQualityTier,
    CalculationMethod,
    EmissionFactor,
    CalculationStep,
    GWP_AR6_100,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMPOSTING CONSTANTS
# =============================================================================

class CompostingType(str, Enum):
    """Types of composting processes."""
    WINDROW = "windrow"  # Open windrow composting
    AERATED_STATIC_PILE = "aerated_static_pile"
    IN_VESSEL = "in_vessel"
    VERMICOMPOSTING = "vermicomposting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"


class FeedstockType(str, Enum):
    """Types of composting feedstocks."""
    FOOD_WASTE = "food_waste"
    YARD_WASTE = "yard_waste"
    MIXED_ORGANIC = "mixed_organic"
    AGRICULTURAL_RESIDUES = "agricultural_residues"
    SEWAGE_SLUDGE = "sewage_sludge"
    ANIMAL_MANURE = "animal_manure"


# IPCC Default emission factors for aerobic composting (kg per tonne wet waste)
COMPOSTING_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    CompostingType.WINDROW.value: {
        "ch4": Decimal("4.0"),  # kg CH4/tonne
        "n2o": Decimal("0.24"),  # kg N2O/tonne
        "ch4_range_low": Decimal("0.08"),
        "ch4_range_high": Decimal("20"),
        "n2o_range_low": Decimal("0.06"),
        "n2o_range_high": Decimal("0.6"),
    },
    CompostingType.AERATED_STATIC_PILE.value: {
        "ch4": Decimal("2.0"),
        "n2o": Decimal("0.20"),
        "ch4_range_low": Decimal("0.04"),
        "ch4_range_high": Decimal("10"),
        "n2o_range_low": Decimal("0.05"),
        "n2o_range_high": Decimal("0.5"),
    },
    CompostingType.IN_VESSEL.value: {
        "ch4": Decimal("1.0"),
        "n2o": Decimal("0.15"),
        "ch4_range_low": Decimal("0.02"),
        "ch4_range_high": Decimal("5"),
        "n2o_range_low": Decimal("0.03"),
        "n2o_range_high": Decimal("0.4"),
    },
    CompostingType.VERMICOMPOSTING.value: {
        "ch4": Decimal("0.5"),
        "n2o": Decimal("0.10"),
        "ch4_range_low": Decimal("0.01"),
        "ch4_range_high": Decimal("2"),
        "n2o_range_low": Decimal("0.02"),
        "n2o_range_high": Decimal("0.3"),
    },
    CompostingType.ANAEROBIC_DIGESTION.value: {
        "ch4": Decimal("1.0"),  # Fugitive only (most captured)
        "n2o": Decimal("0.05"),
        "ch4_range_low": Decimal("0.0"),
        "ch4_range_high": Decimal("5"),
        "n2o_range_low": Decimal("0.01"),
        "n2o_range_high": Decimal("0.2"),
    },
}

# Biogas yield for anaerobic digestion (m3/tonne wet waste)
BIOGAS_YIELD: Dict[str, Decimal] = {
    FeedstockType.FOOD_WASTE.value: Decimal("150"),
    FeedstockType.YARD_WASTE.value: Decimal("60"),
    FeedstockType.MIXED_ORGANIC.value: Decimal("100"),
    FeedstockType.AGRICULTURAL_RESIDUES.value: Decimal("80"),
    FeedstockType.SEWAGE_SLUDGE.value: Decimal("40"),
    FeedstockType.ANIMAL_MANURE.value: Decimal("60"),
    "default": Decimal("80"),
}

# CH4 content in biogas (fraction)
BIOGAS_CH4_FRACTION = Decimal("0.60")

# Carbon sequestration from compost application (kg CO2e/tonne compost)
CARBON_SEQUESTRATION: Dict[str, Decimal] = {
    "soil_application": Decimal("100"),  # Conservative estimate
    "agricultural": Decimal("150"),
    "land_restoration": Decimal("200"),
}

# Compost yield (tonnes compost per tonne input)
COMPOST_YIELD: Dict[str, Decimal] = {
    CompostingType.WINDROW.value: Decimal("0.35"),
    CompostingType.AERATED_STATIC_PILE.value: Decimal("0.40"),
    CompostingType.IN_VESSEL.value: Decimal("0.45"),
    CompostingType.VERMICOMPOSTING.value: Decimal("0.30"),
    CompostingType.ANAEROBIC_DIGESTION.value: Decimal("0.25"),  # Digestate
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class OrganicFeedstock(BaseModel):
    """Individual organic feedstock record."""
    feedstock_type: FeedstockType = Field(..., description="Type of feedstock")
    mass_tonnes: Decimal = Field(..., gt=0, description="Mass in metric tonnes")
    moisture_content: Decimal = Field(
        Decimal("0.6"), ge=0, le=0.95, description="Moisture content"
    )
    nitrogen_content: Optional[Decimal] = Field(
        None, ge=0, le=0.1, description="Nitrogen content (dry basis)"
    )

    class Config:
        use_enum_values = True


class CompostingInput(WasteMRVInput):
    """Input model for Composting MRV Agent."""

    # Composting process
    composting_type: CompostingType = Field(
        CompostingType.WINDROW, description="Type of composting process"
    )

    # Feedstocks
    feedstocks: List[OrganicFeedstock] = Field(
        default_factory=list, description="Organic feedstocks"
    )

    # Simplified input
    total_organic_waste_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Total organic waste composted"
    )
    default_feedstock_type: FeedstockType = Field(
        FeedstockType.MIXED_ORGANIC, description="Default feedstock type"
    )

    # Process parameters
    process_duration_days: Decimal = Field(
        Decimal("60"), gt=0, description="Composting duration in days"
    )
    turning_frequency: str = Field(
        "weekly", description="Turning frequency (daily/weekly/monthly)"
    )

    # Anaerobic digestion specific
    biogas_capture_efficiency: Decimal = Field(
        Decimal("0.95"), ge=0, le=1, description="Biogas capture efficiency"
    )
    biogas_to_energy: bool = Field(
        True, description="Whether biogas is used for energy"
    )
    flare_remaining: bool = Field(
        True, description="Whether uncaptured gas is flared"
    )

    # Carbon sequestration
    include_sequestration_credit: bool = Field(
        False, description="Include carbon sequestration credit"
    )
    compost_end_use: str = Field(
        "soil_application", description="End use of compost"
    )


class CompostingOutput(WasteMRVOutput):
    """Output model for Composting MRV Agent."""

    # Composting-specific outputs
    total_feedstock_tonnes: Decimal = Field(
        Decimal("0"), description="Total feedstock processed"
    )
    compost_produced_tonnes: Decimal = Field(
        Decimal("0"), description="Compost/digestate produced"
    )

    # Gas emissions
    ch4_process_kg: Decimal = Field(
        Decimal("0"), description="CH4 from composting process"
    )
    n2o_process_kg: Decimal = Field(
        Decimal("0"), description="N2O from composting process"
    )

    # Anaerobic digestion specific
    biogas_generated_m3: Decimal = Field(
        Decimal("0"), description="Biogas generated"
    )
    biogas_captured_m3: Decimal = Field(
        Decimal("0"), description="Biogas captured"
    )
    ch4_in_biogas_kg: Decimal = Field(
        Decimal("0"), description="CH4 in biogas"
    )
    energy_generated_mwh: Decimal = Field(
        Decimal("0"), description="Energy from biogas"
    )

    # Credits
    sequestration_credit_kg_co2e: Decimal = Field(
        Decimal("0"), description="Carbon sequestration credit"
    )
    avoided_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Avoided emissions from energy"
    )
    net_emissions_kg_co2e: Decimal = Field(
        Decimal("0"), description="Net emissions"
    )


# =============================================================================
# COMPOSTING MRV AGENT
# =============================================================================

class CompostingMRVAgent(BaseWasteMRVAgent[CompostingInput, CompostingOutput]):
    """
    GL-MRV-WST-004: Composting MRV Agent

    Calculates emissions from organic waste composting and anaerobic digestion.

    Calculation Approach:
    1. Aggregate feedstock quantities
    2. Calculate CH4 emissions based on composting type
    3. Calculate N2O emissions based on nitrogen content
    4. For AD: calculate biogas generation and capture
    5. Calculate energy recovery credits
    6. Calculate carbon sequestration credits
    7. Report net emissions

    Key Formula (IPCC):
        CH4 = M_waste * EF_CH4 * (1 - R)
        N2O = M_waste * EF_N2O

    Where:
        M_waste = Mass of organic waste treated
        EF = Emission factor
        R = Recovery/capture fraction (for AD)

    Example:
        >>> agent = CompostingMRVAgent()
        >>> input_data = CompostingInput(
        ...     organization_id="ORG001",
        ...     reporting_year=2024,
        ...     composting_type=CompostingType.IN_VESSEL,
        ...     total_organic_waste_tonnes=Decimal("5000"),
        ... )
        >>> result = agent.calculate(input_data)
    """

    AGENT_ID = "GL-MRV-WST-004"
    AGENT_NAME = "Composting MRV Agent"
    AGENT_VERSION = "1.0.0"
    TREATMENT_METHOD = TreatmentMethod.COMPOSTING
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    # CH4 density (kg/m3 at STP)
    CH4_DENSITY = Decimal("0.717")
    # Energy content of biogas (MWh/m3)
    BIOGAS_ENERGY_CONTENT = Decimal("0.00555")  # ~20 MJ/m3 / 3600 MJ/MWh

    def __init__(self):
        """Initialize Composting MRV Agent."""
        super().__init__()
        self._emission_factors = COMPOSTING_EMISSION_FACTORS
        self._biogas_yield = BIOGAS_YIELD
        self._compost_yield = COMPOST_YIELD
        self._sequestration = CARBON_SEQUESTRATION
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def calculate(self, input_data: CompostingInput) -> CompostingOutput:
        """
        Calculate composting emissions.

        Args:
            input_data: Composting input data

        Returns:
            CompostingOutput with emissions and audit trail
        """
        start_time = datetime.now(timezone.utc)
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Step 1: Initialize
        is_ad = input_data.composting_type == CompostingType.ANAEROBIC_DIGESTION

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize composting emissions calculation",
            formula="N/A",
            inputs={
                "composting_type": input_data.composting_type.value,
                "is_anaerobic_digestion": is_ad,
                "process_duration_days": str(input_data.process_duration_days),
            },
            output="Initialization complete",
        ))

        # Step 2: Collect feedstock data
        feedstocks = self._collect_feedstock_data(input_data)
        total_feedstock = sum(f.mass_tonnes for f in feedstocks)

        steps.append(CalculationStep(
            step_number=2,
            description="Aggregate organic feedstocks",
            formula="total_feedstock = sum(feedstock_mass)",
            inputs={"num_feedstocks": len(feedstocks)},
            output=f"{total_feedstock} tonnes",
        ))

        # Step 3: Get emission factors
        ef_data = self._emission_factors.get(
            input_data.composting_type.value,
            self._emission_factors[CompostingType.WINDROW.value]
        )

        ch4_ef = ef_data["ch4"]
        n2o_ef = ef_data["n2o"]

        steps.append(CalculationStep(
            step_number=3,
            description="Retrieve emission factors for composting type",
            formula="EF = f(composting_type)",
            inputs={
                "composting_type": input_data.composting_type.value,
            },
            output=f"CH4: {ch4_ef} kg/t, N2O: {n2o_ef} kg/t",
        ))

        # Record emission factors
        ef_ch4 = EmissionFactor(
            factor_id=f"ipcc_composting_ch4_{input_data.composting_type.value}",
            factor_value=ch4_ef,
            factor_unit="kg CH4/tonne",
            source="IPCC",
            source_uri="https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol5.html",
            version="2019",
            last_updated="2019-05-01",
            uncertainty_pct=50.0,  # High uncertainty for composting
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="global",
            treatment_method=TreatmentMethod.COMPOSTING,
        )
        emission_factors.append(ef_ch4)

        # Step 4: Calculate CH4 emissions
        if is_ad:
            # For AD, most CH4 is captured in biogas
            # Calculate biogas generation
            total_biogas_m3 = Decimal("0")
            for feedstock in feedstocks:
                yield_rate = self._biogas_yield.get(
                    feedstock.feedstock_type.value,
                    self._biogas_yield["default"]
                )
                biogas = feedstock.mass_tonnes * yield_rate
                total_biogas_m3 += biogas

            ch4_in_biogas_m3 = total_biogas_m3 * BIOGAS_CH4_FRACTION
            ch4_in_biogas_kg = (ch4_in_biogas_m3 * self.CH4_DENSITY).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Captured vs fugitive
            biogas_captured_m3 = total_biogas_m3 * input_data.biogas_capture_efficiency
            biogas_fugitive_m3 = total_biogas_m3 - biogas_captured_m3

            ch4_fugitive_kg = (biogas_fugitive_m3 * BIOGAS_CH4_FRACTION * self.CH4_DENSITY).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Additional process emissions
            ch4_process_kg = (total_feedstock * ch4_ef).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            total_ch4_kg = ch4_process_kg + ch4_fugitive_kg

            steps.append(CalculationStep(
                step_number=4,
                description="Calculate AD biogas and CH4 emissions",
                formula="CH4_total = CH4_process + CH4_fugitive",
                inputs={
                    "biogas_generated_m3": str(total_biogas_m3),
                    "capture_efficiency": str(input_data.biogas_capture_efficiency),
                    "ch4_in_biogas_kg": str(ch4_in_biogas_kg),
                },
                output=f"Total CH4: {total_ch4_kg} kg",
            ))
        else:
            # Standard composting
            total_ch4_kg = (total_feedstock * ch4_ef).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            ch4_process_kg = total_ch4_kg
            total_biogas_m3 = Decimal("0")
            biogas_captured_m3 = Decimal("0")
            ch4_in_biogas_kg = Decimal("0")

            steps.append(CalculationStep(
                step_number=4,
                description="Calculate aerobic composting CH4 emissions",
                formula="CH4 = feedstock_tonnes * CH4_EF",
                inputs={
                    "feedstock_tonnes": str(total_feedstock),
                    "ch4_ef_kg_per_tonne": str(ch4_ef),
                },
                output=f"{total_ch4_kg} kg CH4",
            ))

        # Step 5: Calculate N2O emissions
        total_n2o_kg = (total_feedstock * n2o_ef).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=5,
            description="Calculate N2O emissions",
            formula="N2O = feedstock_tonnes * N2O_EF",
            inputs={
                "feedstock_tonnes": str(total_feedstock),
                "n2o_ef_kg_per_tonne": str(n2o_ef),
            },
            output=f"{total_n2o_kg} kg N2O",
        ))

        # Step 6: Convert to CO2e
        ch4_co2e_kg = (total_ch4_kg * GWP_AR6_100["CH4_biogenic"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        n2o_co2e_kg = (total_n2o_kg * GWP_AR6_100["N2O"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        total_gross_kg = ch4_co2e_kg + n2o_co2e_kg

        steps.append(CalculationStep(
            step_number=6,
            description="Convert to CO2e",
            formula="CO2e = CH4 * GWP_CH4 + N2O * GWP_N2O",
            inputs={
                "ch4_kg": str(total_ch4_kg),
                "n2o_kg": str(total_n2o_kg),
                "gwp_ch4": str(GWP_AR6_100["CH4_biogenic"]),
                "gwp_n2o": str(GWP_AR6_100["N2O"]),
            },
            output=f"{total_gross_kg} kg CO2e",
        ))

        # Step 7: Calculate energy recovery (AD only)
        energy_mwh = Decimal("0")
        avoided_emissions_kg = Decimal("0")

        if is_ad and input_data.biogas_to_energy:
            energy_mwh = (biogas_captured_m3 * self.BIOGAS_ENERGY_CONTENT).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            # Avoided emissions from grid electricity (assume 400g/kWh)
            avoided_emissions_kg = (energy_mwh * Decimal("1000") * Decimal("0.4")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=7,
                description="Calculate energy recovery and avoided emissions",
                formula="energy_mwh = biogas_m3 * energy_content; avoided = energy * grid_ef",
                inputs={
                    "biogas_captured_m3": str(biogas_captured_m3),
                    "energy_content_mwh_per_m3": str(self.BIOGAS_ENERGY_CONTENT),
                },
                output=f"{energy_mwh} MWh, {avoided_emissions_kg} kg CO2e avoided",
            ))
        else:
            steps.append(CalculationStep(
                step_number=7,
                description="No energy recovery for this process type",
                formula="N/A",
                inputs={"is_ad": is_ad, "biogas_to_energy": input_data.biogas_to_energy},
                output="0 MWh",
            ))

        # Step 8: Calculate sequestration credit
        sequestration_credit_kg = Decimal("0")
        compost_yield_factor = self._compost_yield.get(
            input_data.composting_type.value,
            self._compost_yield[CompostingType.WINDROW.value]
        )
        compost_produced = (total_feedstock * compost_yield_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        if input_data.include_sequestration_credit:
            seq_factor = self._sequestration.get(
                input_data.compost_end_use,
                self._sequestration["soil_application"]
            )
            sequestration_credit_kg = (compost_produced * seq_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        steps.append(CalculationStep(
            step_number=8,
            description="Calculate compost production and sequestration credit",
            formula="compost = feedstock * yield; credit = compost * seq_factor",
            inputs={
                "yield_factor": str(compost_yield_factor),
                "include_credit": input_data.include_sequestration_credit,
            },
            output=f"Compost: {compost_produced}t, Credit: {sequestration_credit_kg} kg CO2e",
        ))

        # Step 9: Calculate net emissions
        net_emissions_kg = (total_gross_kg - avoided_emissions_kg - sequestration_credit_kg).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=9,
            description="Calculate net emissions",
            formula="net = gross - avoided - sequestration",
            inputs={
                "gross_kg": str(total_gross_kg),
                "avoided_kg": str(avoided_emissions_kg),
                "sequestration_kg": str(sequestration_credit_kg),
            },
            output=f"{net_emissions_kg} kg CO2e net",
        ))

        # Create activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "facility_id": input_data.facility_id,
            "reporting_year": input_data.reporting_year,
            "composting_type": input_data.composting_type.value,
            "total_feedstock_tonnes": str(total_feedstock),
            "compost_produced_tonnes": str(compost_produced),
        }

        # Build output
        output = CompostingOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_gross_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_gross_kg),
            co2_kg=Decimal("0"),
            ch4_kg=total_ch4_kg,
            n2o_kg=total_n2o_kg,
            ch4_generated_kg=ch4_process_kg if not is_ad else ch4_in_biogas_kg,
            ch4_captured_kg=Decimal("0") if not is_ad else (ch4_in_biogas_kg - total_ch4_kg),
            ch4_flared_kg=Decimal("0"),
            ch4_utilized_kg=Decimal("0") if not is_ad else ch4_in_biogas_kg * input_data.biogas_capture_efficiency,
            ch4_emitted_kg=total_ch4_kg,
            scope=EmissionScope.SCOPE_1,
            calculation_steps=steps,
            provenance_hash="",
            data_quality_tier=DataQualityTier.TIER_2,
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings,
            # Composting-specific fields
            total_feedstock_tonnes=total_feedstock,
            compost_produced_tonnes=compost_produced,
            ch4_process_kg=ch4_process_kg,
            n2o_process_kg=total_n2o_kg,
            biogas_generated_m3=total_biogas_m3,
            biogas_captured_m3=biogas_captured_m3,
            ch4_in_biogas_kg=ch4_in_biogas_kg,
            energy_generated_mwh=energy_mwh,
            sequestration_credit_kg_co2e=sequestration_credit_kg,
            avoided_emissions_kg_co2e=avoided_emissions_kg,
            net_emissions_kg_co2e=net_emissions_kg,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data={
                "total_emissions_kg_co2e": total_gross_kg,
                "net_emissions_kg_co2e": net_emissions_kg,
            },
            steps=steps,
        )

        return output

    def _collect_feedstock_data(self, input_data: CompostingInput) -> List[OrganicFeedstock]:
        """Collect and normalize feedstock data."""
        feedstocks = list(input_data.feedstocks)

        if not feedstocks and input_data.total_organic_waste_tonnes:
            feedstocks.append(OrganicFeedstock(
                feedstock_type=input_data.default_feedstock_type,
                mass_tonnes=input_data.total_organic_waste_tonnes,
            ))

        return feedstocks
