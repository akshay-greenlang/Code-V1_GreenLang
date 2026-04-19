# -*- coding: utf-8 -*-
"""
GL-DECARB-X-017: Scope 3 Reduction Planner Agent
=================================================

Plans comprehensive Scope 3 emission reduction strategies
across all 15 categories.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class Scope3Category(str, Enum):
    CAT_1_PURCHASED_GOODS = "cat_1_purchased_goods"
    CAT_2_CAPITAL_GOODS = "cat_2_capital_goods"
    CAT_3_FUEL_ENERGY = "cat_3_fuel_energy"
    CAT_4_UPSTREAM_TRANSPORT = "cat_4_upstream_transport"
    CAT_5_WASTE = "cat_5_waste"
    CAT_6_BUSINESS_TRAVEL = "cat_6_business_travel"
    CAT_7_COMMUTING = "cat_7_commuting"
    CAT_8_LEASED_ASSETS = "cat_8_leased_assets"
    CAT_9_DOWNSTREAM_TRANSPORT = "cat_9_downstream_transport"
    CAT_10_PROCESSING = "cat_10_processing"
    CAT_11_PRODUCT_USE = "cat_11_product_use"
    CAT_12_END_OF_LIFE = "cat_12_end_of_life"
    CAT_13_DOWNSTREAM_LEASED = "cat_13_downstream_leased"
    CAT_14_FRANCHISES = "cat_14_franchises"
    CAT_15_INVESTMENTS = "cat_15_investments"


class Scope3Intervention(BaseModel):
    intervention_id: str = Field(...)
    name: str = Field(...)
    category: Scope3Category = Field(...)
    description: str = Field(default="")

    # Impact
    reduction_potential_tco2e: float = Field(..., ge=0)
    reduction_percent: float = Field(default=0, ge=0, le=100)

    # Implementation
    implementation_type: str = Field(default="operational")  # operational, procurement, design
    time_to_impact_months: int = Field(default=12, ge=0)
    complexity: str = Field(default="medium")

    # Levers
    levers: List[str] = Field(default_factory=list)


class Scope3ReductionPlan(BaseModel):
    plan_id: str = Field(...)
    total_scope3_tco2e: float = Field(..., ge=0)
    target_reduction_percent: float = Field(..., ge=0, le=100)

    # By category
    category_emissions: Dict[str, float] = Field(default_factory=dict)
    category_interventions: Dict[str, List[Scope3Intervention]] = Field(default_factory=dict)

    # Summary
    interventions: List[Scope3Intervention] = Field(default_factory=list)
    total_reduction_potential_tco2e: float = Field(default=0, ge=0)
    achievable_reduction_percent: float = Field(default=0, ge=0, le=100)

    provenance_hash: str = Field(default="")


class Scope3ReductionInput(BaseModel):
    operation: str = Field(default="plan")
    total_scope3_tco2e: float = Field(default=500000, ge=0)
    target_reduction_percent: float = Field(default=25, ge=0, le=100)
    category_breakdown: Dict[str, float] = Field(default_factory=dict)


class Scope3ReductionOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    plan: Optional[Scope3ReductionPlan] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# Default interventions by category
DEFAULT_INTERVENTIONS = {
    Scope3Category.CAT_1_PURCHASED_GOODS: [
        {"name": "Low-carbon material specifications", "reduction_pct": 15, "levers": ["Material substitution", "Recycled content"]},
        {"name": "Supplier engagement program", "reduction_pct": 10, "levers": ["SBTi alignment", "Data sharing"]},
    ],
    Scope3Category.CAT_4_UPSTREAM_TRANSPORT: [
        {"name": "Modal shift to rail/sea", "reduction_pct": 20, "levers": ["Route optimization", "Carrier selection"]},
        {"name": "Local sourcing", "reduction_pct": 15, "levers": ["Nearshoring", "Regional suppliers"]},
    ],
    Scope3Category.CAT_6_BUSINESS_TRAVEL: [
        {"name": "Virtual meeting policy", "reduction_pct": 30, "levers": ["Video conferencing", "Travel approval"]},
        {"name": "Sustainable aviation fuel", "reduction_pct": 10, "levers": ["SAF mandates", "Airline selection"]},
    ],
    Scope3Category.CAT_11_PRODUCT_USE: [
        {"name": "Product efficiency improvements", "reduction_pct": 25, "levers": ["Design optimization", "Energy efficiency"]},
        {"name": "Renewable energy integration", "reduction_pct": 15, "levers": ["Product features", "User guidance"]},
    ],
}


class Scope3ReductionPlanner(DeterministicAgent):
    """GL-DECARB-X-017: Scope 3 Reduction Planner Agent"""

    AGENT_ID = "GL-DECARB-X-017"
    AGENT_NAME = "Scope 3 Reduction Planner"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="Scope3ReductionPlanner",
        category=AgentCategory.CRITICAL,
        description="Plans Scope 3 reduction strategies"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Plans Scope 3 reductions", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            s3_input = Scope3ReductionInput(**inputs)
            calculation_trace.append(f"Operation: {s3_input.operation}")

            if s3_input.operation == "plan":
                # Default category breakdown if not provided
                if not s3_input.category_breakdown:
                    s3_input.category_breakdown = {
                        Scope3Category.CAT_1_PURCHASED_GOODS.value: 0.60,
                        Scope3Category.CAT_4_UPSTREAM_TRANSPORT.value: 0.15,
                        Scope3Category.CAT_6_BUSINESS_TRAVEL.value: 0.05,
                        Scope3Category.CAT_11_PRODUCT_USE.value: 0.15,
                        "other": 0.05,
                    }

                category_emissions = {}
                category_interventions = {}
                all_interventions = []

                for cat_key, fraction in s3_input.category_breakdown.items():
                    if cat_key == "other":
                        continue

                    cat_emissions = s3_input.total_scope3_tco2e * fraction
                    category_emissions[cat_key] = cat_emissions

                    # Get interventions for this category
                    try:
                        cat_enum = Scope3Category(cat_key)
                        cat_interventions_data = DEFAULT_INTERVENTIONS.get(cat_enum, [])
                    except ValueError:
                        cat_interventions_data = []

                    interventions = []
                    for idx, intv_data in enumerate(cat_interventions_data):
                        reduction = cat_emissions * intv_data["reduction_pct"] / 100
                        intervention = Scope3Intervention(
                            intervention_id=deterministic_id({"cat": cat_key, "idx": idx}, "s3intv_"),
                            name=intv_data["name"],
                            category=cat_enum if isinstance(cat_key, str) else Scope3Category.CAT_1_PURCHASED_GOODS,
                            reduction_potential_tco2e=reduction,
                            reduction_percent=intv_data["reduction_pct"],
                            levers=intv_data.get("levers", [])
                        )
                        interventions.append(intervention)
                        all_interventions.append(intervention)

                    category_interventions[cat_key] = interventions

                total_reduction = sum(i.reduction_potential_tco2e for i in all_interventions)
                achievable_pct = (total_reduction / s3_input.total_scope3_tco2e * 100) if s3_input.total_scope3_tco2e > 0 else 0

                plan = Scope3ReductionPlan(
                    plan_id=deterministic_id({"target": s3_input.target_reduction_percent}, "s3plan_"),
                    total_scope3_tco2e=s3_input.total_scope3_tco2e,
                    target_reduction_percent=s3_input.target_reduction_percent,
                    category_emissions=category_emissions,
                    category_interventions=category_interventions,
                    interventions=all_interventions,
                    total_reduction_potential_tco2e=total_reduction,
                    achievable_reduction_percent=achievable_pct
                )
                plan.provenance_hash = content_hash(plan.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Created plan with {len(all_interventions)} interventions")

                self._capture_audit_entry(
                    operation="plan",
                    inputs=inputs,
                    outputs={"interventions": len(all_interventions)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "plan",
                    "success": True,
                    "plan": plan.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {s3_input.operation}")

        except Exception as e:
            self.logger.error(f"Planning failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
