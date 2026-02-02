# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-003: Landfill Gas Capture Agent
==============================================

Plans landfill gas capture and utilization systems for methane emission reduction.

Key Features:
- LFG capture system sizing
- Flaring vs energy recovery analysis
- Collection efficiency optimization
- Phased implementation planning
- Financial analysis with carbon credits

Author: GreenLang Framework Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from pydantic import BaseModel, Field

from greenlang.agents.decarbonization.waste.base import (
    BaseWasteDecarbAgent,
    WasteDecarbInput,
    WasteDecarbOutput,
    DecarbonizationStrategy,
    DecarbonizationIntervention,
    DecarbonizationPathway,
    ImplementationTimeline,
    CostCategory,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class LFGUtilizationType(str, Enum):
    """LFG utilization options."""
    FLARING = "flaring"
    ELECTRICITY = "electricity"
    DIRECT_USE = "direct_use"
    RNG = "rng"  # Renewable Natural Gas
    CNG_VEHICLE = "cng_vehicle"


class LandfillGasCaptureInput(WasteDecarbInput):
    """Input model for Landfill Gas Capture Agent."""
    landfill_id: str = Field(..., description="Landfill identifier")
    waste_in_place_tonnes: Decimal = Field(..., gt=0, description="Total waste in place")
    annual_waste_acceptance_tonnes: Decimal = Field(
        Decimal("0"), ge=0, description="Annual waste acceptance"
    )
    current_ch4_emissions_kg: Decimal = Field(..., gt=0, description="Current annual CH4")
    current_collection_efficiency: Decimal = Field(
        Decimal("0"), ge=0, le=1, description="Current LFG collection efficiency"
    )
    target_collection_efficiency: Decimal = Field(
        Decimal("0.85"), ge=0, le=1, description="Target collection efficiency"
    )
    preferred_utilization: LFGUtilizationType = Field(
        LFGUtilizationType.ELECTRICITY, description="Preferred utilization method"
    )
    electricity_price_usd_per_kwh: Decimal = Field(
        Decimal("0.08"), ge=0, description="Electricity sale price"
    )
    carbon_credit_usd_per_tco2e: Decimal = Field(
        Decimal("30"), ge=0, description="Carbon credit price"
    )


class LFGCaptureProjection(BaseModel):
    """LFG capture projections."""
    year: int
    ch4_captured_kg: Decimal = Field(Decimal("0"))
    ch4_destroyed_kg: Decimal = Field(Decimal("0"))
    emissions_avoided_kg_co2e: Decimal = Field(Decimal("0"))
    electricity_generated_mwh: Decimal = Field(Decimal("0"))
    revenue_usd: Decimal = Field(Decimal("0"))


class LandfillGasCaptureOutput(WasteDecarbOutput):
    """Output model for Landfill Gas Capture Agent."""
    lfg_generation_rate_m3_per_year: Decimal = Field(Decimal("0"))
    ch4_reduction_potential_kg: Decimal = Field(Decimal("0"))
    recommended_utilization: str = Field("")
    system_capacity_m3_per_hour: Decimal = Field(Decimal("0"))
    annual_electricity_potential_mwh: Decimal = Field(Decimal("0"))
    annual_revenue_potential_usd: Decimal = Field(Decimal("0"))
    projections: List[LFGCaptureProjection] = Field(default_factory=list)


class LandfillGasCaptureAgent(BaseWasteDecarbAgent[LandfillGasCaptureInput, LandfillGasCaptureOutput]):
    """
    GL-DECARB-WST-003: Landfill Gas Capture Agent

    Plans landfill gas capture systems for methane emission reduction
    with economic analysis of utilization options.

    LFG Capture Hierarchy:
    1. Beneficial use (electricity, RNG, direct use)
    2. High-efficiency enclosed flare
    3. Open flare (last resort)

    Example:
        >>> agent = LandfillGasCaptureAgent()
        >>> input_data = LandfillGasCaptureInput(
        ...     organization_id="ORG001",
        ...     landfill_id="LF001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("5000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     waste_in_place_tonnes=Decimal("500000"),
        ...     current_ch4_emissions_kg=Decimal("1000000"),
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-003"
    AGENT_NAME = "Landfill Gas Capture Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.LANDFILL_GAS_CAPTURE

    # CH4 energy content (kWh/m3)
    CH4_ENERGY_CONTENT = Decimal("10")
    # CH4 density (kg/m3)
    CH4_DENSITY = Decimal("0.717")
    # GWP for CH4
    GWP_CH4 = Decimal("27.9")
    # Electricity generation efficiency
    ELECTRICITY_EFFICIENCY = Decimal("0.35")
    # Flare efficiency
    FLARE_EFFICIENCY = Decimal("0.98")

    def plan(self, input_data: LandfillGasCaptureInput) -> LandfillGasCaptureOutput:
        """Generate LFG capture plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []

        # Calculate LFG generation rate
        ch4_m3_per_year = (input_data.current_ch4_emissions_kg / self.CH4_DENSITY).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        lfg_m3_per_year = ch4_m3_per_year * Decimal("2")  # 50% CH4 in LFG

        # Calculate collection improvement
        collection_improvement = input_data.target_collection_efficiency - input_data.current_collection_efficiency
        ch4_captured_additional = (input_data.current_ch4_emissions_kg * collection_improvement).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Calculate emissions reduction
        if input_data.preferred_utilization == LFGUtilizationType.FLARING:
            destruction_efficiency = self.FLARE_EFFICIENCY
        else:
            destruction_efficiency = Decimal("0.99")  # Engine/turbine

        ch4_destroyed = (ch4_captured_additional * destruction_efficiency).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        emissions_avoided = (ch4_destroyed * self.GWP_CH4).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Calculate electricity potential
        electricity_potential = Decimal("0")
        if input_data.preferred_utilization in [LFGUtilizationType.ELECTRICITY, LFGUtilizationType.RNG]:
            ch4_captured_m3 = ch4_captured_additional / self.CH4_DENSITY
            electricity_potential = (
                ch4_captured_m3 * self.CH4_ENERGY_CONTENT * self.ELECTRICITY_EFFICIENCY / Decimal("1000")
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Calculate revenue
        electricity_revenue = electricity_potential * Decimal("1000") * input_data.electricity_price_usd_per_kwh
        carbon_revenue = emissions_avoided / Decimal("1000") * input_data.carbon_credit_usd_per_tco2e
        total_revenue = (electricity_revenue + carbon_revenue).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # System sizing
        system_capacity = (lfg_m3_per_year * input_data.target_collection_efficiency / Decimal("8760")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        # Create interventions
        if input_data.current_collection_efficiency == Decimal("0"):
            # New system installation
            interventions.append(DecarbonizationIntervention(
                intervention_id="LFG-INSTALL-001",
                strategy=DecarbonizationStrategy.LANDFILL_GAS_CAPTURE,
                description="Install new LFG collection system",
                timeline=ImplementationTimeline.MEDIUM_TERM,
                cost_category=CostCategory.HIGH_COST,
                reduction_potential_kg_co2e=emissions_avoided * Decimal("0.7"),
                reduction_potential_pct=Decimal("60"),
                capex_usd=system_capacity * Decimal("50000"),
                annual_opex_usd=system_capacity * Decimal("5000"),
                confidence=ConfidenceLevel.MEDIUM,
                co_benefits=["Energy generation", "Odor control", "Carbon credits"],
            ))
        else:
            # System upgrade
            interventions.append(DecarbonizationIntervention(
                intervention_id="LFG-UPGRADE-001",
                strategy=DecarbonizationStrategy.LANDFILL_GAS_CAPTURE,
                description="Upgrade LFG collection efficiency",
                timeline=ImplementationTimeline.SHORT_TERM,
                cost_category=CostCategory.MODERATE_COST,
                reduction_potential_kg_co2e=emissions_avoided,
                reduction_potential_pct=(collection_improvement * Decimal("100")),
                capex_usd=system_capacity * Decimal("20000"),
                annual_opex_usd=system_capacity * Decimal("3000"),
                confidence=ConfidenceLevel.HIGH,
            ))

        # Add utilization intervention
        if input_data.preferred_utilization == LFGUtilizationType.ELECTRICITY:
            interventions.append(DecarbonizationIntervention(
                intervention_id="LFG-ELEC-001",
                strategy=DecarbonizationStrategy.LANDFILL_GAS_CAPTURE,
                description="Install electricity generation equipment",
                timeline=ImplementationTimeline.MEDIUM_TERM,
                cost_category=CostCategory.HIGH_COST,
                reduction_potential_kg_co2e=Decimal("0"),  # Counted in capture
                reduction_potential_pct=Decimal("0"),
                capex_usd=electricity_potential * Decimal("2000"),  # $/kW installed
                annual_opex_usd=electricity_potential * Decimal("50"),
                confidence=ConfidenceLevel.MEDIUM,
                co_benefits=["Renewable energy", "Revenue generation"],
            ))

        # Create projections
        projections: List[LFGCaptureProjection] = []
        for year in range(input_data.current_year, input_data.target_year + 1):
            years_elapsed = year - input_data.current_year
            ramp_factor = min(Decimal("1"), Decimal(str(years_elapsed)) / Decimal("3"))

            proj = LFGCaptureProjection(
                year=year,
                ch4_captured_kg=ch4_captured_additional * ramp_factor,
                ch4_destroyed_kg=ch4_destroyed * ramp_factor,
                emissions_avoided_kg_co2e=emissions_avoided * ramp_factor,
                electricity_generated_mwh=electricity_potential * ramp_factor,
                revenue_usd=total_revenue * ramp_factor,
            )
            projections.append(proj)

        # Create pathway
        total_capex = sum(i.capex_usd for i in interventions)
        total_opex = sum(i.annual_opex_usd for i in interventions)

        pathway = DecarbonizationPathway(
            pathway_id="LFG-001",
            name="Landfill Gas Capture and Utilization",
            description=f"LFG capture with {input_data.preferred_utilization.value}",
            interventions=interventions,
            total_reduction_kg_co2e=emissions_avoided,
            total_capex_usd=total_capex,
            total_annual_opex_usd=total_opex,
            start_year=input_data.current_year,
            target_year=input_data.target_year,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            target_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e - emissions_avoided,
            reduction_pct=(emissions_avoided / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
        )

        output = LandfillGasCaptureOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=emissions_avoided,
            achievable_reduction_pct=(emissions_avoided / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
            total_capex_usd=total_capex,
            total_annual_opex_usd=total_opex,
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            lfg_generation_rate_m3_per_year=lfg_m3_per_year,
            ch4_reduction_potential_kg=ch4_destroyed,
            recommended_utilization=input_data.preferred_utilization.value,
            system_capacity_m3_per_hour=system_capacity,
            annual_electricity_potential_mwh=electricity_potential,
            annual_revenue_potential_usd=total_revenue,
            projections=projections,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data={"landfill_id": input_data.landfill_id},
            output_data={"emissions_avoided": str(emissions_avoided)},
        )

        return output
