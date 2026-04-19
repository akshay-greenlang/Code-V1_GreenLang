# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-002: Circular Economy Agent
==========================================

Plans circular economy strategies to eliminate waste and maintain
material value in closed loops.

Key Features:
- Material flow analysis
- Circularity metrics calculation
- Design for circularity recommendations
- Business model innovation
- Extended producer responsibility planning

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


class CircularStrategy(str, Enum):
    """Circular economy strategies."""
    MAINTAIN = "maintain"  # Maintain/prolong product use
    REPAIR = "repair"
    REUSE = "reuse"
    REFURBISH = "refurbish"
    REMANUFACTURE = "remanufacture"
    REPURPOSE = "repurpose"
    RECYCLE = "recycle"
    RECOVER = "recover"


class CircularEconomyInput(WasteDecarbInput):
    """Input model for Circular Economy Agent."""
    material_flows_tonnes: Dict[str, Decimal] = Field(
        default_factory=dict, description="Material flows by type"
    )
    current_circularity_rate: Decimal = Field(
        Decimal("0.1"), ge=0, le=1, description="Current circularity rate"
    )
    target_circularity_rate: Decimal = Field(
        Decimal("0.5"), ge=0, le=1, description="Target circularity rate"
    )
    product_categories: List[str] = Field(default_factory=list, description="Product categories")


class CircularityMetrics(BaseModel):
    """Circularity metrics output."""
    material_circularity_indicator: Decimal = Field(Decimal("0"), description="MCI (0-1)")
    recycled_input_rate: Decimal = Field(Decimal("0"))
    end_of_life_recovery_rate: Decimal = Field(Decimal("0"))
    utility_factor: Decimal = Field(Decimal("1"))


class CircularEconomyOutput(WasteDecarbOutput):
    """Output model for Circular Economy Agent."""
    current_circularity_metrics: Optional[CircularityMetrics] = None
    projected_circularity_metrics: Optional[CircularityMetrics] = None
    circular_strategies: List[str] = Field(default_factory=list)
    material_savings_tonnes: Decimal = Field(Decimal("0"))
    value_retention_usd: Decimal = Field(Decimal("0"))


class CircularEconomyAgent(BaseWasteDecarbAgent[CircularEconomyInput, CircularEconomyOutput]):
    """
    GL-DECARB-WST-002: Circular Economy Agent

    Plans comprehensive circular economy strategies based on Ellen MacArthur
    Foundation and EU Circular Economy frameworks.

    Circularity Loops (in priority order):
    1. Maintain - Keep products in use longer
    2. Repair - Fix products to extend life
    3. Reuse - Use products again for same purpose
    4. Refurbish - Restore products to good condition
    5. Remanufacture - Rebuild products using original components
    6. Repurpose - Use products for different purpose
    7. Recycle - Process materials for new products
    8. Recover - Extract energy from materials

    Example:
        >>> agent = CircularEconomyAgent()
        >>> input_data = CircularEconomyInput(
        ...     organization_id="ORG001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("2000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     current_circularity_rate=Decimal("0.15"),
        ...     target_circularity_rate=Decimal("0.50"),
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-002"
    AGENT_NAME = "Circular Economy Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.CIRCULAR_DESIGN

    # Emission reduction per % circularity improvement
    CIRCULARITY_EMISSION_FACTOR = Decimal("0.8")  # 0.8% emission reduction per 1% circularity

    # Strategy potentials (% of remaining linear flow captured)
    STRATEGY_POTENTIALS = {
        CircularStrategy.MAINTAIN.value: Decimal("10"),
        CircularStrategy.REPAIR.value: Decimal("15"),
        CircularStrategy.REUSE.value: Decimal("20"),
        CircularStrategy.REFURBISH.value: Decimal("12"),
        CircularStrategy.REMANUFACTURE.value: Decimal("8"),
        CircularStrategy.REPURPOSE.value: Decimal("5"),
        CircularStrategy.RECYCLE.value: Decimal("35"),
        CircularStrategy.RECOVER.value: Decimal("10"),
    }

    def plan(self, input_data: CircularEconomyInput) -> CircularEconomyOutput:
        """Generate circular economy plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []
        circular_strategies: List[str] = []

        # Calculate circularity gap
        circularity_gap = input_data.target_circularity_rate - input_data.current_circularity_rate
        linear_flow = Decimal("1") - input_data.current_circularity_rate

        # Calculate potential from each strategy
        total_reduction_kg = Decimal("0")
        cumulative_circularity = input_data.current_circularity_rate

        for strategy in CircularStrategy:
            if cumulative_circularity >= input_data.target_circularity_rate:
                break

            potential = self.STRATEGY_POTENTIALS.get(strategy.value, Decimal("10"))
            circularity_gain = (linear_flow * potential / Decimal("100")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Don't exceed target
            if cumulative_circularity + circularity_gain > input_data.target_circularity_rate:
                circularity_gain = input_data.target_circularity_rate - cumulative_circularity

            # Calculate emission reduction
            emission_reduction_pct = circularity_gain * self.CIRCULARITY_EMISSION_FACTOR * Decimal("100")
            reduction_kg = (input_data.baseline_emissions_kg_co2e * emission_reduction_pct / Decimal("100")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            if circularity_gain > Decimal("0.001"):
                intervention = DecarbonizationIntervention(
                    intervention_id=f"CE-{strategy.value.upper()[:4]}-001",
                    strategy=DecarbonizationStrategy.CIRCULAR_DESIGN,
                    description=f"Implement {strategy.value} circular strategy",
                    timeline=ImplementationTimeline.MEDIUM_TERM,
                    cost_category=CostCategory.MODERATE_COST,
                    reduction_potential_kg_co2e=reduction_kg,
                    reduction_potential_pct=emission_reduction_pct,
                    confidence=ConfidenceLevel.MEDIUM,
                )
                interventions.append(intervention)
                circular_strategies.append(strategy.value)
                total_reduction_kg += reduction_kg
                cumulative_circularity += circularity_gain

        # Create pathway
        pathway = DecarbonizationPathway(
            pathway_id="CE-001",
            name="Circular Economy Transition Pathway",
            description="Comprehensive circular economy implementation",
            interventions=interventions,
            total_reduction_kg_co2e=total_reduction_kg,
            start_year=input_data.current_year,
            target_year=input_data.target_year,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            target_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e - total_reduction_kg,
            reduction_pct=(total_reduction_kg / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
        )

        # Calculate circularity metrics
        current_metrics = CircularityMetrics(
            material_circularity_indicator=input_data.current_circularity_rate,
            recycled_input_rate=input_data.current_circularity_rate * Decimal("0.6"),
            end_of_life_recovery_rate=input_data.current_circularity_rate * Decimal("0.8"),
        )

        projected_metrics = CircularityMetrics(
            material_circularity_indicator=cumulative_circularity,
            recycled_input_rate=cumulative_circularity * Decimal("0.6"),
            end_of_life_recovery_rate=cumulative_circularity * Decimal("0.8"),
        )

        # Calculate material savings (rough estimate)
        material_savings = Decimal("0")
        for material, tonnes in input_data.material_flows_tonnes.items():
            savings = (tonnes * (cumulative_circularity - input_data.current_circularity_rate)).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            material_savings += savings

        achievable_pct = (total_reduction_kg / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0")

        output = CircularEconomyOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=total_reduction_kg,
            achievable_reduction_pct=achievable_pct,
            gap_to_target_kg_co2e=Decimal("0"),
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            current_circularity_metrics=current_metrics,
            projected_circularity_metrics=projected_metrics,
            circular_strategies=circular_strategies,
            material_savings_tonnes=material_savings,
            value_retention_usd=material_savings * Decimal("500"),  # Rough value
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data={"circularity_gap": str(circularity_gap)},
            output_data={"total_reduction_kg": str(total_reduction_kg)},
        )

        return output
