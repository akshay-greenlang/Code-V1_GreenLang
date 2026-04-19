# -*- coding: utf-8 -*-
"""
GL-DECARB-WST-005: Extended Producer Responsibility (EPR) Planner Agent
=========================================================================

Plans Extended Producer Responsibility programs to shift waste management
costs and responsibilities to producers.

Key Features:
- EPR scheme design and evaluation
- Producer fee structure optimization
- Collection and recycling target setting
- Eco-modulation of fees
- Compliance pathway planning

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


class EPRProductCategory(str, Enum):
    """Product categories covered by EPR."""
    PACKAGING = "packaging"
    ELECTRONICS = "electronics"
    BATTERIES = "batteries"
    TEXTILES = "textiles"
    VEHICLES = "vehicles"
    TIRES = "tires"
    FURNITURE = "furniture"
    CONSTRUCTION = "construction"


class EPRSchemeType(str, Enum):
    """Types of EPR schemes."""
    COLLECTIVE = "collective"  # Producer Responsibility Organization
    INDIVIDUAL = "individual"  # Each producer manages own
    HYBRID = "hybrid"


class EPRPlannerInput(WasteDecarbInput):
    """Input model for EPR Planner."""
    product_category: EPRProductCategory = Field(..., description="Product category")
    scheme_type: EPRSchemeType = Field(
        EPRSchemeType.COLLECTIVE, description="EPR scheme type"
    )
    current_recycling_rate: Decimal = Field(
        Decimal("0.3"), ge=0, le=1, description="Current recycling rate"
    )
    target_recycling_rate: Decimal = Field(
        Decimal("0.7"), ge=0, le=1, description="Target recycling rate"
    )
    annual_products_tonnes: Decimal = Field(..., gt=0, description="Annual products placed on market")
    current_collection_coverage: Decimal = Field(
        Decimal("0.5"), ge=0, le=1, description="Current collection coverage"
    )
    apply_eco_modulation: bool = Field(True, description="Apply eco-modulation to fees")


class EPRFeeStructure(BaseModel):
    """EPR fee structure output."""
    base_fee_usd_per_tonne: Decimal = Field(Decimal("0"))
    eco_modulated_fee_usd_per_tonne: Decimal = Field(Decimal("0"))
    recyclability_bonus_pct: Decimal = Field(Decimal("0"))
    recycled_content_bonus_pct: Decimal = Field(Decimal("0"))
    hazardous_penalty_pct: Decimal = Field(Decimal("0"))


class EPRTarget(BaseModel):
    """EPR recycling/recovery targets."""
    year: int
    collection_target_pct: Decimal = Field(Decimal("0"))
    recycling_target_pct: Decimal = Field(Decimal("0"))
    recovery_target_pct: Decimal = Field(Decimal("0"))


class EPRPlannerOutput(WasteDecarbOutput):
    """Output model for EPR Planner."""
    recommended_scheme_type: str = Field("")
    fee_structure: Optional[EPRFeeStructure] = None
    annual_fee_revenue_usd: Decimal = Field(Decimal("0"))
    targets_by_year: List[EPRTarget] = Field(default_factory=list)
    collection_infrastructure_cost_usd: Decimal = Field(Decimal("0"))
    recycling_capacity_needed_tonnes: Decimal = Field(Decimal("0"))


class EPRPlannerAgent(BaseWasteDecarbAgent[EPRPlannerInput, EPRPlannerOutput]):
    """
    GL-DECARB-WST-005: Extended Producer Responsibility Planner Agent

    Plans EPR programs based on EU Packaging Directive and global best practices.

    EPR Design Principles:
    1. Full cost coverage - Producers pay full end-of-life costs
    2. Eco-modulation - Fees reflect environmental impact
    3. Collection targets - Progressive increase
    4. Recycling targets - Material-specific goals
    5. Reporting - Transparent data collection

    Example:
        >>> agent = EPRPlannerAgent()
        >>> input_data = EPRPlannerInput(
        ...     organization_id="ORG001",
        ...     baseline_year=2020,
        ...     baseline_emissions_kg_co2e=Decimal("3000000"),
        ...     current_year=2024,
        ...     target_year=2030,
        ...     product_category=EPRProductCategory.PACKAGING,
        ...     annual_products_tonnes=Decimal("50000"),
        ... )
        >>> result = agent.plan(input_data)
    """

    AGENT_ID = "GL-DECARB-WST-005"
    AGENT_NAME = "Extended Producer Responsibility Planner Agent"
    AGENT_VERSION = "1.0.0"
    STRATEGY = DecarbonizationStrategy.EXTENDED_PRODUCER_RESPONSIBILITY

    # Base fees by category (USD/tonne)
    BASE_FEES = {
        EPRProductCategory.PACKAGING.value: Decimal("150"),
        EPRProductCategory.ELECTRONICS.value: Decimal("500"),
        EPRProductCategory.BATTERIES.value: Decimal("1000"),
        EPRProductCategory.TEXTILES.value: Decimal("200"),
        EPRProductCategory.VEHICLES.value: Decimal("100"),
        EPRProductCategory.TIRES.value: Decimal("300"),
        EPRProductCategory.FURNITURE.value: Decimal("180"),
        EPRProductCategory.CONSTRUCTION.value: Decimal("80"),
    }

    # Emission reduction per % recycling improvement
    RECYCLING_EMISSION_FACTOR = Decimal("15")  # kg CO2e per tonne per % improvement

    def plan(self, input_data: EPRPlannerInput) -> EPRPlannerOutput:
        """Generate EPR plan."""
        start_time = datetime.now(timezone.utc)

        interventions: List[DecarbonizationIntervention] = []

        # Calculate recycling improvement
        recycling_improvement = input_data.target_recycling_rate - input_data.current_recycling_rate
        collection_improvement = Decimal("0.95") - input_data.current_collection_coverage

        # Calculate emission reduction
        emission_reduction_kg = (
            input_data.annual_products_tonnes *
            recycling_improvement * Decimal("100") *
            self.RECYCLING_EMISSION_FACTOR
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Calculate fee structure
        base_fee = self.BASE_FEES.get(
            input_data.product_category.value,
            Decimal("150")
        )

        fee_structure = EPRFeeStructure(
            base_fee_usd_per_tonne=base_fee,
            eco_modulated_fee_usd_per_tonne=base_fee * Decimal("1.2") if input_data.apply_eco_modulation else base_fee,
            recyclability_bonus_pct=Decimal("-20") if input_data.apply_eco_modulation else Decimal("0"),
            recycled_content_bonus_pct=Decimal("-15") if input_data.apply_eco_modulation else Decimal("0"),
            hazardous_penalty_pct=Decimal("50") if input_data.apply_eco_modulation else Decimal("0"),
        )

        annual_revenue = (input_data.annual_products_tonnes * fee_structure.eco_modulated_fee_usd_per_tonne).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Create targets timeline
        targets: List[EPRTarget] = []
        years_to_target = input_data.target_year - input_data.current_year

        for i in range(years_to_target + 1):
            year = input_data.current_year + i
            progress = Decimal(str(i)) / Decimal(str(years_to_target)) if years_to_target > 0 else Decimal("1")

            targets.append(EPRTarget(
                year=year,
                collection_target_pct=input_data.current_collection_coverage + (collection_improvement * progress),
                recycling_target_pct=input_data.current_recycling_rate + (recycling_improvement * progress),
                recovery_target_pct=min(Decimal("0.95"), input_data.current_recycling_rate + (recycling_improvement * progress) + Decimal("0.15")),
            ))

        # Calculate infrastructure costs
        collection_cost = (
            input_data.annual_products_tonnes * collection_improvement * Decimal("50")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        recycling_capacity_needed = (input_data.annual_products_tonnes * input_data.target_recycling_rate).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Create interventions
        interventions.append(DecarbonizationIntervention(
            intervention_id="EPR-SCHEME-001",
            strategy=DecarbonizationStrategy.EXTENDED_PRODUCER_RESPONSIBILITY,
            description=f"Establish {input_data.scheme_type.value} EPR scheme for {input_data.product_category.value}",
            timeline=ImplementationTimeline.MEDIUM_TERM,
            cost_category=CostCategory.MODERATE_COST,
            reduction_potential_kg_co2e=emission_reduction_kg * Decimal("0.4"),
            reduction_potential_pct=Decimal("15"),
            capex_usd=Decimal("2000000"),
            confidence=ConfidenceLevel.MEDIUM,
        ))

        if collection_improvement > Decimal("0.1"):
            interventions.append(DecarbonizationIntervention(
                intervention_id="EPR-COLLECT-001",
                strategy=DecarbonizationStrategy.EXTENDED_PRODUCER_RESPONSIBILITY,
                description="Expand collection infrastructure",
                timeline=ImplementationTimeline.MEDIUM_TERM,
                cost_category=CostCategory.HIGH_COST,
                reduction_potential_kg_co2e=emission_reduction_kg * Decimal("0.3"),
                reduction_potential_pct=Decimal("10"),
                capex_usd=collection_cost,
                confidence=ConfidenceLevel.MEDIUM,
            ))

        interventions.append(DecarbonizationIntervention(
            intervention_id="EPR-RECYCLE-001",
            strategy=DecarbonizationStrategy.EXTENDED_PRODUCER_RESPONSIBILITY,
            description="Increase recycling capacity and technology",
            timeline=ImplementationTimeline.MEDIUM_TERM,
            cost_category=CostCategory.HIGH_COST,
            reduction_potential_kg_co2e=emission_reduction_kg * Decimal("0.3"),
            reduction_potential_pct=Decimal("12"),
            capex_usd=recycling_capacity_needed * Decimal("200"),
            confidence=ConfidenceLevel.MEDIUM,
        ))

        # Create pathway
        total_capex = sum(i.capex_usd for i in interventions)

        pathway = DecarbonizationPathway(
            pathway_id="EPR-001",
            name=f"EPR Implementation for {input_data.product_category.value}",
            description="Extended Producer Responsibility program implementation",
            interventions=interventions,
            total_reduction_kg_co2e=emission_reduction_kg,
            total_capex_usd=total_capex,
            start_year=input_data.current_year,
            target_year=input_data.target_year,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            target_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e - emission_reduction_kg,
            reduction_pct=(emission_reduction_kg / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
        )

        output = EPRPlannerOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            recommended_pathway=pathway,
            baseline_emissions_kg_co2e=input_data.baseline_emissions_kg_co2e,
            achievable_reduction_kg_co2e=emission_reduction_kg,
            achievable_reduction_pct=(emission_reduction_kg / input_data.baseline_emissions_kg_co2e * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ) if input_data.baseline_emissions_kg_co2e > 0 else Decimal("0"),
            total_capex_usd=total_capex,
            provenance_hash="",
            calculation_timestamp=datetime.now(timezone.utc),
            calculation_duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            recommended_scheme_type=input_data.scheme_type.value,
            fee_structure=fee_structure,
            annual_fee_revenue_usd=annual_revenue,
            targets_by_year=targets,
            collection_infrastructure_cost_usd=collection_cost,
            recycling_capacity_needed_tonnes=recycling_capacity_needed,
        )

        output.provenance_hash = self._generate_provenance_hash(
            input_data={"product_category": input_data.product_category.value},
            output_data={"emission_reduction_kg": str(emission_reduction_kg)},
        )

        return output
