# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-001: Waste Reduction Planner Agent
=================================================

Plans waste minimization strategies at source to reduce emissions
from waste generation and disposal.

Key Features:
- Source reduction opportunity assessment
- Waste hierarchy optimization (reduce > reuse > recycle)
- Process efficiency improvements
- Material substitution analysis
- Packaging optimization

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


class WasteReductionCategory(str, Enum):
    """Categories of waste reduction opportunities."""
    PROCESS_OPTIMIZATION = "process_optimization"
    MATERIAL_EFFICIENCY = "material_efficiency"
    PACKAGING_REDUCTION = "packaging_reduction"
    PRODUCT_REDESIGN = "product_redesign"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    BEHAVIORAL_CHANGE = "behavioral_change"


class WasteReductionInput(WasteDecarbInput):
    """Input model for Waste Reduction Planner."""
    current_waste_tonnes: Decimal = Field(..., gt=0, description="Current annual waste")
    waste_intensity_kg_per_unit: Optional[Decimal] = Field(None, description="Waste per unit output")
    production_units: Optional[Decimal] = Field(None, description="Annual production units")
    categories_to_assess: List[WasteReductionCategory] = Field(
        default_factory=lambda: list(WasteReductionCategory),
        description="Categories to include"
    )


class WasteReductionOutput(WasteDecarbOutput):
    """Output model for Waste Reduction Planner."""
    waste_reduction_potential_tonnes: Decimal = Field(Decimal("0"))
    intensity_improvement_pct: Decimal = Field(Decimal("0"))
    reduction_by_category: Dict[str, Decimal] = Field(default_factory=dict)


class WasteReductionPlannerAgent(BaseWasteDecarbAgent[WasteReductionInput, WasteReductionOutput]):
    """
    GL-DECARB-WST-001: Waste Reduction Planner Agent

    Plans waste minimization strategies following the waste hierarchy:
    Prevent > Reduce > Reuse > Recycle > Recover > Dispose

    Example:
        >>> agent = WasteReductionPlannerAgent()
        >>> input_data = WasteReductionInput(
        ...     organization_id="ORG001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("1000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     current_waste_tonnes=Decimal("5000"),
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-001"
    AGENT_NAME = "Waste Reduction Planner Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.WASTE_REDUCTION

    # Reduction potential by category (% of waste in that category)
    CATEGORY_POTENTIALS = {
        WasteReductionCategory.PROCESS_OPTIMIZATION.value: Decimal("15"),
        WasteReductionCategory.MATERIAL_EFFICIENCY.value: Decimal("12"),
        WasteReductionCategory.PACKAGING_REDUCTION.value: Decimal("25"),
        WasteReductionCategory.PRODUCT_REDESIGN.value: Decimal("20"),
        WasteReductionCategory.DIGITAL_TRANSFORMATION.value: Decimal("10"),
        WasteReductionCategory.BEHAVIORAL_CHANGE.value: Decimal("8"),
    }

    def plan(self, input_data: WasteReductionInput) -> WasteReductionOutput:
        """Generate waste reduction plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []
        reduction_by_category: Dict[str, Decimal] = {}
        total_reduction_kg = Decimal("0")

        # Create interventions for each category
        for category in input_data.categories_to_assess:
            potential_pct = self.CATEGORY_POTENTIALS.get(category.value, Decimal("10"))

            # Assume waste is distributed across categories
            category_share = Decimal("1") / Decimal(str(len(input_data.categories_to_assess)))
            category_waste = input_data.current_waste_tonnes * category_share

            reduction_tonnes = (category_waste * potential_pct / Decimal("100")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Convert to emissions (use average 500 kg CO2e/tonne waste)
            reduction_kg = reduction_tonnes * Decimal("500") * Decimal("1000")

            intervention = self._create_intervention(
                intervention_id=f"WRED-{category.value.upper()[:4]}-001",
                strategy=DecarbonizationStrategy.WASTE_REDUCTION,
                description=f"Implement {category.value.replace('_', ' ')} measures",
                baseline_kg=input_data.baseline_emissions_kg_co2e * category_share,
                reduction_pct=potential_pct,
            )
            intervention.reduction_potential_kg_co2e = reduction_kg
            interventions.append(intervention)

            reduction_by_category[category.value] = reduction_tonnes
            total_reduction_kg += reduction_kg

        # Create pathway
        pathway = DecarbonizationPathway(
            pathway_id="WRP-001",
            name="Comprehensive Waste Reduction Pathway",
            description="Multi-category waste reduction strategy",
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

        # Calculate summary
        total_waste_reduction = sum(reduction_by_category.values())
        achievable_pct = (total_reduction_kg / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0")

        gap = (
            input_data.baseline_emissions_kg_co2e * input_data.target_reduction_pct / Decimal("100")
        ) - total_reduction_kg

        activity_summary = {
            "organization_id": input_data.organization_id,
            "current_waste_tonnes": str(input_data.current_waste_tonnes),
            "categories_assessed": len(input_data.categories_to_assess),
        }

        output = WasteReductionOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=total_reduction_kg,
            achievable_reduction_pct=achievable_pct,
            gap_to_target_kg_co2e=gap if gap > 0 else Decimal("0"),
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            waste_reduction_potential_tonnes=total_waste_reduction,
            intensity_improvement_pct=achievable_pct,
            reduction_by_category=reduction_by_category,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data={"total_reduction_kg": str(total_reduction_kg)},
        )

        return output
