# -*- coding: utf-8 -*-
"""
GL-MRV-WST-001: Landfill MRV Agent
===================================

Calculates methane emissions from landfill disposal using IPCC First Order
Decay (FOD) methodology. Supports gas capture and flaring calculations.

Key Features:
- IPCC 2006/2019 First Order Decay (FOD) model
- Multi-year decay modeling for historical waste deposits
- Landfill gas capture and flaring accounting
- Support for managed and unmanaged landfills
- Climate zone adjustments for decay rates

Zero-Hallucination Guarantees:
- All calculations use IPCC deterministic formulas
- Emission factors from published regulatory sources
- SHA-256 provenance hash for complete audit trail

Reference Standards:
- IPCC 2006 Guidelines Volume 5, Chapter 3 (Solid Waste Disposal)
- IPCC 2019 Refinement - Waste Chapter
- EPA Landfill Methane Outreach Program (LMOP)

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
import math
import logging

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.mrv.waste.base import (
    BaseWasteMRVAgent,
    WasteMRVInput,
    WasteMRVOutput,
    WasteType,
    TreatmentMethod,
    LandfillType,
    EmissionScope,
    DataQualityTier,
    CalculationMethod,
    EmissionFactor,
    CalculationStep,
    GWP_AR6_100,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class WasteDeposit(BaseModel):
    """Individual waste deposit record for FOD calculation."""
    deposit_year: int = Field(..., ge=1950, le=2100, description="Year waste was deposited")
    waste_type: WasteType = Field(..., description="Type of waste")
    waste_tonnes: Decimal = Field(..., gt=0, description="Mass in metric tonnes")
    moisture_content: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Moisture content as fraction"
    )

    class Config:
        use_enum_values = True


class LandfillInput(WasteMRVInput):
    """Input model for Landfill MRV Agent."""

    # Landfill characteristics
    landfill_id: Optional[str] = Field(None, description="Landfill site identifier")
    landfill_type: LandfillType = Field(
        LandfillType.MANAGED_ANAEROBIC, description="Type of landfill"
    )
    opening_year: Optional[int] = Field(None, ge=1900, le=2100, description="Year landfill opened")

    # Waste deposits (for multi-year FOD)
    waste_deposits: List[WasteDeposit] = Field(
        default_factory=list, description="Historical waste deposits"
    )

    # Current year waste (if not using deposits list)
    current_year_waste_tonnes: Optional[Decimal] = Field(
        None, gt=0, description="Waste deposited in reporting year"
    )
    current_waste_type: WasteType = Field(
        WasteType.MUNICIPAL_SOLID_WASTE, description="Waste type for current year"
    )

    # Landfill gas collection
    has_gas_collection: bool = Field(False, description="Whether site has LFG collection")
    gas_collection_efficiency: Decimal = Field(
        Decimal("0.75"), ge=0, le=1, description="LFG collection efficiency (0-1)"
    )
    gas_flaring: bool = Field(False, description="Whether collected gas is flared")
    gas_energy_recovery: bool = Field(False, description="Whether collected gas is used for energy")
    flare_efficiency: Decimal = Field(
        Decimal("0.98"), ge=0, le=1, description="Flare destruction efficiency"
    )

    # IPCC parameters (override defaults if known)
    doc_f: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Fraction DOC that decomposes"
    )
    oxidation_factor: Decimal = Field(
        Decimal("0.1"), ge=0, le=1, description="Oxidation factor for covered landfills"
    )

    @field_validator("gas_collection_efficiency", "flare_efficiency")
    @classmethod
    def validate_efficiency(cls, v: Decimal) -> Decimal:
        """Ensure efficiency is within valid range."""
        if v < Decimal("0") or v > Decimal("1"):
            raise ValueError("Efficiency must be between 0 and 1")
        return v


class LandfillOutput(WasteMRVOutput):
    """Output model for Landfill MRV Agent."""

    # Landfill-specific outputs
    landfill_id: Optional[str] = Field(None, description="Landfill site identifier")
    total_waste_deposited_tonnes: Decimal = Field(
        Decimal("0"), description="Total waste included in calculation"
    )

    # Landfill gas quantities
    landfill_gas_generated_m3: Decimal = Field(
        Decimal("0"), description="Total LFG generated (m3)"
    )
    ch4_in_lfg_m3: Decimal = Field(
        Decimal("0"), description="CH4 component of LFG (m3)"
    )

    # Breakdown by disposition
    ch4_to_atmosphere_kg: Decimal = Field(
        Decimal("0"), description="CH4 released to atmosphere"
    )
    co2_from_flaring_kg: Decimal = Field(
        Decimal("0"), description="CO2 from CH4 flaring"
    )
    energy_recovered_mwh: Decimal = Field(
        Decimal("0"), description="Energy recovered from LFG"
    )

    # Model parameters used
    mcf_applied: Decimal = Field(Decimal("1.0"), description="MCF value used")
    doc_f_applied: Decimal = Field(Decimal("0.5"), description="DOC_f value used")
    decay_rate_applied: Decimal = Field(Decimal("0.05"), description="Decay rate k used")


# =============================================================================
# LANDFILL MRV AGENT
# =============================================================================

class LandfillMRVAgent(BaseWasteMRVAgent[LandfillInput, LandfillOutput]):
    """
    GL-MRV-WST-001: Landfill Methane Emissions MRV Agent

    Calculates methane emissions from solid waste disposal in landfills
    using IPCC First Order Decay (FOD) methodology.

    Calculation Approach:
    1. Determine DDOC (Degradable Decomposable Organic Carbon) from waste composition
    2. Apply First Order Decay model for multi-year emissions
    3. Account for landfill gas collection and destruction
    4. Apply oxidation factor for surface cover
    5. Convert CH4 to CO2e using GWP values

    Key Formula (IPCC FOD):
        CH4_emitted = DDOCm_decomposed * F * (16/12) * (1-R) * (1-OX)

    Where:
        DDOCm_decomposed = DDOCm * (1 - e^(-k*t))
        DDOCm = W * DOC * DOC_f * MCF
        F = Fraction CH4 in LFG (default 0.5)
        R = Recovery fraction
        OX = Oxidation factor

    Example:
        >>> agent = LandfillMRVAgent()
        >>> input_data = LandfillInput(
        ...     organization_id="ORG001",
        ...     reporting_year=2024,
        ...     landfill_type=LandfillType.MANAGED_ANAEROBIC,
        ...     current_year_waste_tonnes=Decimal("10000"),
        ...     current_waste_type=WasteType.MUNICIPAL_SOLID_WASTE,
        ...     has_gas_collection=True,
        ...     gas_collection_efficiency=Decimal("0.75"),
        ... )
        >>> result = agent.calculate(input_data)
        >>> print(f"Emissions: {result.total_emissions_mt_co2e} MT CO2e")
    """

    AGENT_ID = "GL-MRV-WST-001"
    AGENT_NAME = "Landfill MRV Agent"
    AGENT_VERSION = "1.0.0"
    TREATMENT_METHOD = TreatmentMethod.LANDFILL
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    # CH4 density at STP (kg/m3)
    CH4_DENSITY = Decimal("0.717")
    # Fraction CH4 in landfill gas (default)
    F_CH4 = Decimal("0.5")
    # Energy content of CH4 (MWh/tonne)
    CH4_ENERGY_CONTENT = Decimal("13.9")

    def __init__(self):
        """Initialize Landfill MRV Agent."""
        super().__init__()
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def calculate(self, input_data: LandfillInput) -> LandfillOutput:
        """
        Calculate landfill methane emissions.

        Args:
            input_data: Landfill input data with waste deposits

        Returns:
            LandfillOutput with emissions and audit trail
        """
        start_time = datetime.now(timezone.utc)
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Step 1: Initialize and validate inputs
        steps.append(CalculationStep(
            step_number=1,
            description="Initialize landfill emissions calculation",
            formula="N/A",
            inputs={
                "landfill_type": input_data.landfill_type.value,
                "reporting_year": input_data.reporting_year,
                "has_gas_collection": input_data.has_gas_collection,
                "climate_zone": input_data.climate_zone or "temperate",
            },
            output="Initialization complete",
        ))

        # Step 2: Get MCF value
        mcf = self._get_mcf_value(input_data.landfill_type)
        steps.append(CalculationStep(
            step_number=2,
            description="Determine Methane Correction Factor (MCF)",
            formula="MCF = f(landfill_type)",
            inputs={"landfill_type": input_data.landfill_type.value},
            output=str(mcf),
        ))

        # Step 3: Collect waste data
        waste_deposits = self._collect_waste_data(input_data)
        total_waste = sum(wd.waste_tonnes for wd in waste_deposits)

        steps.append(CalculationStep(
            step_number=3,
            description="Aggregate waste deposit data",
            formula="total_waste = sum(waste_deposits)",
            inputs={
                "num_deposits": len(waste_deposits),
                "reporting_year": input_data.reporting_year,
            },
            output=f"{total_waste} tonnes",
        ))

        # Step 4: Calculate DDOC and CH4 generation for each deposit
        doc_f = input_data.doc_f or Decimal("0.5")
        climate_zone = input_data.climate_zone or "temperate"

        total_ch4_generated = Decimal("0")
        total_ddoc = Decimal("0")

        for deposit in waste_deposits:
            doc = self._get_doc_value(deposit.waste_type)
            half_life = self._get_half_life(deposit.waste_type, climate_zone)
            decay_rate = self._calculate_decay_rate(half_life)

            # Years since deposit
            years = input_data.reporting_year - deposit.deposit_year
            if years < 0:
                continue

            # Calculate DDOCm
            ddoc_m = deposit.waste_tonnes * doc * doc_f * mcf

            # Apply FOD decay
            if years == 0:
                # Current year: use half decay for partial year
                ddoc_decomposed = ddoc_m * (Decimal("1") - Decimal(str(
                    math.exp(-float(decay_rate) * 0.5)
                )))
            else:
                # Previous years: full decay
                ddoc_decomposed = ddoc_m * (Decimal("1") - Decimal(str(
                    math.exp(-float(decay_rate) * float(years))
                )))

            # CH4 generated from decomposed DOC
            ch4_from_deposit = ddoc_decomposed * self.F_CH4 * (Decimal("16") / Decimal("12"))
            total_ch4_generated += ch4_from_deposit
            total_ddoc += ddoc_m

        total_ch4_generated = total_ch4_generated.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=4,
            description="Calculate CH4 generation from FOD model",
            formula="CH4_gen = DDOCm * (1 - e^(-k*t)) * F * (16/12)",
            inputs={
                "total_ddoc": str(total_ddoc),
                "doc_f": str(doc_f),
                "mcf": str(mcf),
                "decay_rate": str(decay_rate if waste_deposits else "0"),
            },
            output=f"{total_ch4_generated} tonnes CH4",
        ))

        # Record emission factor
        ef_landfill = EmissionFactor(
            factor_id=f"ipcc_landfill_mcf_{input_data.landfill_type.value}",
            factor_value=mcf,
            factor_unit="fraction",
            source="IPCC",
            source_uri="https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol5.html",
            version="2019",
            last_updated="2019-05-01",
            uncertainty_pct=15.0,
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="global",
            treatment_method=TreatmentMethod.LANDFILL,
        )
        emission_factors.append(ef_landfill)

        # Step 5: Apply gas collection and flaring
        ch4_collected = Decimal("0")
        ch4_flared = Decimal("0")
        ch4_utilized = Decimal("0")
        co2_from_flaring = Decimal("0")

        if input_data.has_gas_collection:
            ch4_collected = total_ch4_generated * input_data.gas_collection_efficiency

            if input_data.gas_flaring:
                ch4_flared = ch4_collected * input_data.flare_efficiency
                # CO2 from combustion: CH4 + 2O2 -> CO2 + 2H2O
                # Molecular weight ratio: 44/16 = 2.75
                co2_from_flaring = ch4_flared * Decimal("2.75")

            if input_data.gas_energy_recovery:
                ch4_utilized = ch4_collected
                if input_data.gas_flaring:
                    # If both, split 50/50 or use remaining after flaring
                    ch4_utilized = ch4_collected - ch4_flared

        ch4_collected = ch4_collected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        ch4_flared = ch4_flared.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        ch4_utilized = ch4_utilized.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        co2_from_flaring = co2_from_flaring.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=5,
            description="Apply landfill gas collection and destruction",
            formula="CH4_collected = CH4_gen * collection_eff; CH4_flared = CH4_collected * flare_eff",
            inputs={
                "collection_efficiency": str(input_data.gas_collection_efficiency),
                "flare_efficiency": str(input_data.flare_efficiency),
                "gas_flaring": input_data.gas_flaring,
                "gas_energy_recovery": input_data.gas_energy_recovery,
            },
            output=f"Collected: {ch4_collected}t, Flared: {ch4_flared}t, Utilized: {ch4_utilized}t",
        ))

        # Step 6: Apply oxidation factor
        ch4_before_oxidation = total_ch4_generated - ch4_collected
        ox_factor = input_data.oxidation_factor
        ch4_oxidized = ch4_before_oxidation * ox_factor
        ch4_emitted = (ch4_before_oxidation - ch4_oxidized).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=6,
            description="Apply oxidation factor for surface cover",
            formula="CH4_emitted = (CH4_gen - CH4_collected) * (1 - OX)",
            inputs={
                "ch4_before_oxidation": str(ch4_before_oxidation),
                "oxidation_factor": str(ox_factor),
            },
            output=f"{ch4_emitted} tonnes CH4 emitted",
        ))

        # Step 7: Convert to CO2e
        ch4_emitted_kg = self._tonnes_to_kg(ch4_emitted)
        ch4_co2e_kg = ch4_emitted_kg * GWP_AR6_100["CH4_biogenic"]
        ch4_co2e_kg = ch4_co2e_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # CO2 from flaring also counted
        co2_from_flaring_kg = self._tonnes_to_kg(co2_from_flaring)

        total_emissions_kg = ch4_co2e_kg + co2_from_flaring_kg

        steps.append(CalculationStep(
            step_number=7,
            description="Convert CH4 to CO2e using GWP",
            formula="CO2e = CH4_emitted * GWP_CH4 + CO2_flaring",
            inputs={
                "ch4_emitted_kg": str(ch4_emitted_kg),
                "gwp_ch4_biogenic": str(GWP_AR6_100["CH4_biogenic"]),
                "co2_from_flaring_kg": str(co2_from_flaring_kg),
            },
            output=f"{total_emissions_kg} kg CO2e",
        ))

        # Step 8: Calculate energy recovered
        energy_recovered_mwh = (ch4_utilized * self.CH4_ENERGY_CONTENT).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Calculate LFG volume (m3)
        lfg_generated_m3 = (total_ch4_generated / self.CH4_DENSITY / self.F_CH4).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        ch4_in_lfg_m3 = (total_ch4_generated / self.CH4_DENSITY).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Create activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "facility_id": input_data.facility_id,
            "landfill_id": input_data.landfill_id,
            "reporting_year": input_data.reporting_year,
            "landfill_type": input_data.landfill_type.value,
            "total_waste_tonnes": str(total_waste),
            "num_deposits": len(waste_deposits),
            "has_gas_collection": input_data.has_gas_collection,
        }

        # Build output
        output = LandfillOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_emissions_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_emissions_kg),
            co2_kg=co2_from_flaring_kg,
            ch4_kg=ch4_emitted_kg,
            n2o_kg=Decimal("0"),
            ch4_generated_kg=self._tonnes_to_kg(total_ch4_generated),
            ch4_captured_kg=self._tonnes_to_kg(ch4_collected),
            ch4_flared_kg=self._tonnes_to_kg(ch4_flared),
            ch4_utilized_kg=self._tonnes_to_kg(ch4_utilized),
            ch4_emitted_kg=ch4_emitted_kg,
            scope=EmissionScope.SCOPE_1,
            calculation_steps=steps,
            provenance_hash="",  # Will be set below
            data_quality_tier=DataQualityTier.TIER_2,
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings,
            # Landfill-specific fields
            landfill_id=input_data.landfill_id,
            total_waste_deposited_tonnes=total_waste,
            landfill_gas_generated_m3=lfg_generated_m3,
            ch4_in_lfg_m3=ch4_in_lfg_m3,
            ch4_to_atmosphere_kg=ch4_emitted_kg,
            co2_from_flaring_kg=co2_from_flaring_kg,
            energy_recovered_mwh=energy_recovered_mwh,
            mcf_applied=mcf,
            doc_f_applied=doc_f,
            decay_rate_applied=decay_rate if waste_deposits else Decimal("0.05"),
        )

        # Generate provenance hash
        output.provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data={
                "total_emissions_kg_co2e": total_emissions_kg,
                "ch4_emitted_kg": ch4_emitted_kg,
            },
            steps=steps,
        )

        return output

    def _collect_waste_data(self, input_data: LandfillInput) -> List[WasteDeposit]:
        """Collect and normalize waste deposit data."""
        deposits = list(input_data.waste_deposits)

        # Add current year waste if specified
        if input_data.current_year_waste_tonnes:
            deposits.append(WasteDeposit(
                deposit_year=input_data.reporting_year,
                waste_type=input_data.current_waste_type,
                waste_tonnes=input_data.current_year_waste_tonnes,
            ))

        return deposits

    def _get_half_life(self, waste_type: WasteType, climate_zone: str) -> Decimal:
        """Get half-life for waste type and climate zone."""
        zone_data = self._half_life.get(climate_zone, self._half_life["temperate"])
        return zone_data.get(waste_type.value, zone_data["default"])

    def _calculate_decay_rate(self, half_life: Decimal) -> Decimal:
        """Calculate decay rate k from half-life."""
        # k = ln(2) / t_1/2
        ln2 = Decimal(str(math.log(2)))
        return (ln2 / half_life).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
