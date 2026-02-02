# -*- coding: utf-8 -*-
"""
GL-DECARB-X-020: Cost-Benefit Analyzer Agent
=============================================

Analyzes costs and benefits of decarbonization investments.

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


class BenefitType(str, Enum):
    ENERGY_SAVINGS = "energy_savings"
    CARBON_SAVINGS = "carbon_savings"
    MAINTENANCE_REDUCTION = "maintenance_reduction"
    RISK_MITIGATION = "risk_mitigation"
    REPUTATION = "reputation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class CostCategory(str, Enum):
    CAPITAL = "capital"
    OPERATING = "operating"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    OPPORTUNITY = "opportunity"


class CostItem(BaseModel):
    name: str = Field(...)
    category: CostCategory = Field(...)
    amount_usd: float = Field(...)
    is_recurring: bool = Field(default=False)
    years: int = Field(default=1, ge=1)


class BenefitItem(BaseModel):
    name: str = Field(...)
    benefit_type: BenefitType = Field(...)
    annual_value_usd: float = Field(default=0, ge=0)
    is_quantified: bool = Field(default=True)
    description: str = Field(default="")


class CostBenefitAnalysis(BaseModel):
    analysis_id: str = Field(...)
    project_name: str = Field(...)
    analysis_period_years: int = Field(default=15, ge=1)
    discount_rate: float = Field(default=0.08, ge=0)

    # Costs
    costs: List[CostItem] = Field(default_factory=list)
    total_cost_pv: float = Field(default=0)

    # Benefits
    benefits: List[BenefitItem] = Field(default_factory=list)
    total_benefit_pv: float = Field(default=0)

    # Results
    net_benefit_pv: float = Field(default=0)
    benefit_cost_ratio: float = Field(default=0)
    simple_payback_years: Optional[float] = Field(None)
    discounted_payback_years: Optional[float] = Field(None)
    irr_percent: Optional[float] = Field(None)

    provenance_hash: str = Field(default="")


class CostBenefitInput(BaseModel):
    operation: str = Field(default="analyze")
    project_name: str = Field(default="Decarbonization Project")
    analysis_period_years: int = Field(default=15, ge=1)
    discount_rate: float = Field(default=0.08, ge=0, le=0.3)
    costs: List[Dict[str, Any]] = Field(default_factory=list)
    benefits: List[Dict[str, Any]] = Field(default_factory=list)
    carbon_price_usd: float = Field(default=50, ge=0)
    emission_reduction_tco2e: float = Field(default=1000, ge=0)


class CostBenefitOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    analysis: Optional[CostBenefitAnalysis] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class CostBenefitAnalyzer(DeterministicAgent):
    """GL-DECARB-X-020: Cost-Benefit Analyzer Agent"""

    AGENT_ID = "GL-DECARB-X-020"
    AGENT_NAME = "Cost-Benefit Analyzer"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="CostBenefitAnalyzer",
        category=AgentCategory.CRITICAL,
        description="Analyzes costs and benefits"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Analyzes costs and benefits", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            cba_input = CostBenefitInput(**inputs)
            calculation_trace.append(f"Operation: {cba_input.operation}")

            if cba_input.operation == "analyze":
                # Parse costs
                costs = [CostItem(**c) for c in cba_input.costs] if cba_input.costs else [
                    CostItem(name="Capital Investment", category=CostCategory.CAPITAL, amount_usd=500000),
                    CostItem(name="Implementation", category=CostCategory.OPERATING, amount_usd=50000),
                ]

                # Parse benefits or generate from carbon savings
                benefits = [BenefitItem(**b) for b in cba_input.benefits] if cba_input.benefits else []

                # Add carbon savings benefit
                carbon_benefit = BenefitItem(
                    name="Carbon Savings (at shadow price)",
                    benefit_type=BenefitType.CARBON_SAVINGS,
                    annual_value_usd=cba_input.emission_reduction_tco2e * cba_input.carbon_price_usd,
                    is_quantified=True
                )
                benefits.append(carbon_benefit)

                # Calculate PV of costs
                total_cost_pv = 0
                for cost in costs:
                    if cost.is_recurring:
                        # Annuity PV
                        for year in range(1, cost.years + 1):
                            total_cost_pv += cost.amount_usd / ((1 + cba_input.discount_rate) ** year)
                    else:
                        total_cost_pv += cost.amount_usd

                # Calculate PV of benefits
                total_benefit_pv = 0
                for benefit in benefits:
                    for year in range(1, cba_input.analysis_period_years + 1):
                        total_benefit_pv += benefit.annual_value_usd / ((1 + cba_input.discount_rate) ** year)

                # Calculate metrics
                net_benefit = total_benefit_pv - total_cost_pv
                bcr = total_benefit_pv / total_cost_pv if total_cost_pv > 0 else 0

                # Simple payback
                annual_benefit = sum(b.annual_value_usd for b in benefits)
                initial_cost = sum(c.amount_usd for c in costs if not c.is_recurring)
                simple_payback = initial_cost / annual_benefit if annual_benefit > 0 else None

                calculation_trace.append(f"Total cost PV: ${total_cost_pv:,.0f}")
                calculation_trace.append(f"Total benefit PV: ${total_benefit_pv:,.0f}")
                calculation_trace.append(f"Net benefit: ${net_benefit:,.0f}")
                calculation_trace.append(f"BCR: {bcr:.2f}")

                analysis = CostBenefitAnalysis(
                    analysis_id=deterministic_id({"project": cba_input.project_name}, "cba_"),
                    project_name=cba_input.project_name,
                    analysis_period_years=cba_input.analysis_period_years,
                    discount_rate=cba_input.discount_rate,
                    costs=costs,
                    total_cost_pv=total_cost_pv,
                    benefits=benefits,
                    total_benefit_pv=total_benefit_pv,
                    net_benefit_pv=net_benefit,
                    benefit_cost_ratio=bcr,
                    simple_payback_years=simple_payback
                )
                analysis.provenance_hash = content_hash(analysis.model_dump(exclude={"provenance_hash"}))

                self._capture_audit_entry(
                    operation="analyze",
                    inputs=inputs,
                    outputs={"net_benefit": net_benefit, "bcr": bcr},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "analyze",
                    "success": True,
                    "analysis": analysis.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {cba_input.operation}")

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
