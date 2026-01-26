# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-004: Waste-to-Energy Optimizer Agent
===================================================

Optimizes waste-to-energy facility operations and planning for emission
reduction while maximizing energy recovery.

Key Features:
- WtE technology comparison
- Feedstock optimization
- Energy recovery maximization
- Emission control optimization
- Carbon capture readiness assessment

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


class WtETechnology(str, Enum):
    """Waste-to-Energy technologies."""
    MASS_BURN = "mass_burn"
    RDF = "rdf"
    GASIFICATION = "gasification"
    PLASMA_ARC = "plasma_arc"
    PYROLYSIS = "pyrolysis"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"


class OptimizationStrategy(str, Enum):
    """WtE optimization strategies."""
    FEEDSTOCK_OPTIMIZATION = "feedstock_optimization"
    COMBUSTION_EFFICIENCY = "combustion_efficiency"
    ENERGY_RECOVERY = "energy_recovery"
    CHP_IMPLEMENTATION = "chp_implementation"
    CARBON_CAPTURE = "carbon_capture"
    EMISSION_CONTROLS = "emission_controls"


class WasteToEnergyInput(WasteDecarbInput):
    """Input model for Waste-to-Energy Optimizer."""
    facility_id: str = Field(..., description="WtE facility identifier")
    technology: WtETechnology = Field(WtETechnology.MASS_BURN, description="Current technology")
    annual_throughput_tonnes: Decimal = Field(..., gt=0, description="Annual waste throughput")
    current_efficiency: Decimal = Field(
        Decimal("0.25"), ge=0, le=0.5, description="Current electrical efficiency"
    )
    target_efficiency: Decimal = Field(
        Decimal("0.35"), ge=0, le=0.5, description="Target efficiency"
    )
    has_ccs: bool = Field(False, description="Has carbon capture")
    feedstock_composition: Dict[str, Decimal] = Field(
        default_factory=dict, description="Feedstock composition by type"
    )


class WtEOptimizationScenario(BaseModel):
    """WtE optimization scenario."""
    scenario_id: str
    description: str
    efficiency_gain_pct: Decimal = Field(Decimal("0"))
    emission_reduction_pct: Decimal = Field(Decimal("0"))
    energy_increase_mwh: Decimal = Field(Decimal("0"))
    capex_usd: Decimal = Field(Decimal("0"))
    payback_years: Decimal = Field(Decimal("0"))


class WasteToEnergyOutput(WasteDecarbOutput):
    """Output model for Waste-to-Energy Optimizer."""
    current_electricity_generation_mwh: Decimal = Field(Decimal("0"))
    optimized_electricity_generation_mwh: Decimal = Field(Decimal("0"))
    efficiency_improvement_pct: Decimal = Field(Decimal("0"))
    optimization_scenarios: List[WtEOptimizationScenario] = Field(default_factory=list)
    recommended_technology_upgrade: Optional[str] = None
    ccs_feasibility_score: Decimal = Field(Decimal("0"))


class WasteToEnergyOptimizerAgent(BaseWasteDecarbAgent[WasteToEnergyInput, WasteToEnergyOutput]):
    """
    GL-DECARB-WST-004: Waste-to-Energy Optimizer Agent

    Optimizes WtE operations to maximize energy recovery and minimize emissions.

    Optimization Priorities:
    1. Improve combustion efficiency
    2. Maximize energy recovery (CHP where possible)
    3. Optimize feedstock composition
    4. Upgrade emission controls
    5. Assess CCS readiness

    Example:
        >>> agent = WasteToEnergyOptimizerAgent()
        >>> input_data = WasteToEnergyInput(
        ...     organization_id="ORG001",
        ...     facility_id="WTE001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("10000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     annual_throughput_tonnes=Decimal("100000"),
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-004"
    AGENT_NAME = "Waste-to-Energy Optimizer Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.WASTE_TO_ENERGY

    # Energy content of MSW (MJ/kg)
    MSW_ENERGY_CONTENT = Decimal("10")
    # Biogenic fraction of MSW
    BIOGENIC_FRACTION = Decimal("0.5")

    def plan(self, input_data: WasteToEnergyInput) -> WasteToEnergyOutput:
        """Generate WtE optimization plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []
        scenarios: List[WtEOptimizationScenario] = []

        # Calculate current electricity generation
        energy_mj = input_data.annual_throughput_tonnes * Decimal("1000") * self.MSW_ENERGY_CONTENT
        current_electricity = (energy_mj * input_data.current_efficiency / Decimal("3.6")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        target_electricity = (energy_mj * input_data.target_efficiency / Decimal("3.6")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        electricity_gain = target_electricity - current_electricity

        # Calculate emission reduction from efficiency improvement
        # More electricity = more grid displacement
        grid_ef = Decimal("400")  # kg CO2e/MWh
        avoided_emissions = (electricity_gain * grid_ef).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Scenario 1: Combustion Efficiency
        scenarios.append(WtEOptimizationScenario(
            scenario_id="WTE-EFF-001",
            description="Combustion and steam cycle optimization",
            efficiency_gain_pct=Decimal("3"),
            emission_reduction_pct=Decimal("5"),
            energy_increase_mwh=current_electricity * Decimal("0.12"),
            capex_usd=Decimal("5000000"),
            payback_years=Decimal("4"),
        ))

        interventions.append(DecarbonizationIntervention(
            intervention_id="WTE-EFF-001",
            strategy=DecarbonizationStrategy.WASTE_TO_ENERGY,
            description="Optimize combustion and steam cycle efficiency",
            timeline=ImplementationTimeline.SHORT_TERM,
            cost_category=CostCategory.MODERATE_COST,
            reduction_potential_kg_co2e=avoided_emissions * Decimal("0.3"),
            reduction_potential_pct=Decimal("5"),
            capex_usd=Decimal("5000000"),
            confidence=ConfidenceLevel.HIGH,
        ))

        # Scenario 2: CHP Implementation
        if not input_data.has_ccs:
            scenarios.append(WtEOptimizationScenario(
                scenario_id="WTE-CHP-001",
                description="Combined Heat and Power implementation",
                efficiency_gain_pct=Decimal("15"),
                emission_reduction_pct=Decimal("15"),
                energy_increase_mwh=current_electricity * Decimal("0.5"),  # Heat as equivalent
                capex_usd=Decimal("15000000"),
                payback_years=Decimal("6"),
            ))

            interventions.append(DecarbonizationIntervention(
                intervention_id="WTE-CHP-001",
                strategy=DecarbonizationStrategy.WASTE_TO_ENERGY,
                description="Implement combined heat and power",
                timeline=ImplementationTimeline.MEDIUM_TERM,
                cost_category=CostCategory.HIGH_COST,
                reduction_potential_kg_co2e=avoided_emissions * Decimal("0.5"),
                reduction_potential_pct=Decimal("15"),
                capex_usd=Decimal("15000000"),
                confidence=ConfidenceLevel.MEDIUM,
                co_benefits=["District heating", "Industrial process heat"],
            ))

        # Scenario 3: Carbon Capture (if applicable)
        ccs_feasibility = Decimal("0")
        if input_data.annual_throughput_tonnes > Decimal("50000"):
            ccs_feasibility = Decimal("0.7")  # Good candidate for CCS

            # Calculate CCS potential
            # Fossil CO2 from waste = throughput * carbon_content * fossil_fraction * 44/12
            fossil_co2 = (
                input_data.annual_throughput_tonnes * Decimal("0.28") *
                (Decimal("1") - self.BIOGENIC_FRACTION) * Decimal("3.667") * Decimal("1000")
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            ccs_capture = fossil_co2 * Decimal("0.90")  # 90% capture

            scenarios.append(WtEOptimizationScenario(
                scenario_id="WTE-CCS-001",
                description="Carbon capture and storage",
                efficiency_gain_pct=Decimal("-3"),  # Efficiency penalty
                emission_reduction_pct=Decimal("45"),
                energy_increase_mwh=Decimal("0"),
                capex_usd=Decimal("100000000"),
                payback_years=Decimal("15"),
            ))

            interventions.append(DecarbonizationIntervention(
                intervention_id="WTE-CCS-001",
                strategy=DecarbonizationStrategy.WASTE_TO_ENERGY,
                description="Install carbon capture on flue gas",
                timeline=ImplementationTimeline.LONG_TERM,
                cost_category=CostCategory.VERY_HIGH_COST,
                reduction_potential_kg_co2e=ccs_capture,
                reduction_potential_pct=Decimal("45"),
                capex_usd=Decimal("100000000"),
                confidence=ConfidenceLevel.LOW,
                co_benefits=["BECCS potential (biogenic CO2)", "Carbon credits"],
            ))

        # Total reduction
        total_reduction = sum(i.reduction_potential_kg_co2e for i in interventions)
        total_capex = sum(i.capex_usd for i in interventions)

        # Create pathway
        pathway = DecarbonizationPathway(
            pathway_id="WTE-001",
            name="WtE Optimization Pathway",
            description="Comprehensive WtE efficiency and emission optimization",
            interventions=interventions,
            total_reduction_kg_co2e=total_reduction,
            total_capex_usd=total_capex,
            start_year=input_data.current_year,
            target_year=input_data.target_year,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            target_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e - total_reduction,
            reduction_pct=(total_reduction / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
        )

        efficiency_improvement = input_data.target_efficiency - input_data.current_efficiency

        output = WasteToEnergyOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=total_reduction,
            achievable_reduction_pct=(total_reduction / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
            total_capex_usd=total_capex,
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            current_electricity_generation_mwh=current_electricity,
            optimized_electricity_generation_mwh=target_electricity,
            efficiency_improvement_pct=efficiency_improvement * Decimal("100"),
            optimization_scenarios=scenarios,
            recommended_technology_upgrade="CHP" if not input_data.has_ccs else "CCS",
            ccs_feasibility_score=ccs_feasibility,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data={"facility_id": input_data.facility_id},
            output_data={"total_reduction": str(total_reduction)},
        )

        return output
