# -*- coding: utf-8 -*-
"""
GL-DECARB-X-021: Transition Risk Assessor Agent
================================================

Assesses climate transition risks for decarbonization planning.

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


class TransitionRiskType(str, Enum):
    POLICY = "policy"  # Carbon pricing, regulations
    TECHNOLOGY = "technology"  # Stranded assets, new tech
    MARKET = "market"  # Customer preferences, commodities
    REPUTATION = "reputation"  # Stakeholder perception
    LEGAL = "legal"  # Litigation, liability


class RiskLikelihood(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskImpact(str, Enum):
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"


class TransitionRisk(BaseModel):
    risk_id: str = Field(...)
    name: str = Field(...)
    risk_type: TransitionRiskType = Field(...)
    description: str = Field(default="")

    # Assessment
    likelihood: RiskLikelihood = Field(...)
    impact: RiskImpact = Field(...)
    risk_score: float = Field(default=0, ge=0, le=25)
    time_horizon_years: int = Field(default=5, ge=1)

    # Financial exposure
    potential_exposure_usd: Optional[float] = Field(None, ge=0)

    # Mitigation
    mitigation_strategies: List[str] = Field(default_factory=list)
    mitigation_status: str = Field(default="not_started")
    residual_risk_score: float = Field(default=0, ge=0, le=25)


class TransitionRiskAssessment(BaseModel):
    assessment_id: str = Field(...)
    organization_name: str = Field(...)
    assessment_date: datetime = Field(default_factory=DeterministicClock.now)

    # Risks
    risks: List[TransitionRisk] = Field(default_factory=list)

    # Summary
    total_risks: int = Field(default=0, ge=0)
    high_priority_risks: int = Field(default=0, ge=0)
    total_exposure_usd: float = Field(default=0, ge=0)
    average_risk_score: float = Field(default=0, ge=0, le=25)

    # By category
    risks_by_type: Dict[str, int] = Field(default_factory=dict)

    provenance_hash: str = Field(default="")


class TransitionRiskInput(BaseModel):
    operation: str = Field(default="assess")
    organization_name: str = Field(default="Organization")
    sector: str = Field(default="general")
    current_emissions_tco2e: float = Field(default=100000, ge=0)
    revenue_musd: float = Field(default=100, ge=0)
    carbon_intensity: float = Field(default=100, ge=0, description="tCO2e/MUSD")


class TransitionRiskOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    assessment: Optional[TransitionRiskAssessment] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# Risk score matrix
LIKELIHOOD_SCORE = {
    RiskLikelihood.LOW: 1,
    RiskLikelihood.MEDIUM: 2,
    RiskLikelihood.HIGH: 3,
    RiskLikelihood.VERY_HIGH: 4,
}

IMPACT_SCORE = {
    RiskImpact.NEGLIGIBLE: 1,
    RiskImpact.MINOR: 2,
    RiskImpact.MODERATE: 3,
    RiskImpact.MAJOR: 4,
    RiskImpact.SEVERE: 5,
}


class TransitionRiskAssessor(DeterministicAgent):
    """GL-DECARB-X-021: Transition Risk Assessor Agent"""

    AGENT_ID = "GL-DECARB-X-021"
    AGENT_NAME = "Transition Risk Assessor"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="TransitionRiskAssessor",
        category=AgentCategory.CRITICAL,
        description="Assesses climate transition risks"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Assesses transition risks", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            tr_input = TransitionRiskInput(**inputs)
            calculation_trace.append(f"Operation: {tr_input.operation}")

            if tr_input.operation == "assess":
                risks = []

                # Policy risk: Carbon pricing
                carbon_price_exposure = tr_input.current_emissions_tco2e * 100  # $100/tCO2e scenario
                risks.append(TransitionRisk(
                    risk_id=deterministic_id({"type": "policy_carbon"}, "trisk_"),
                    name="Carbon Pricing Risk",
                    risk_type=TransitionRiskType.POLICY,
                    description="Exposure to rising carbon prices through taxes or ETS",
                    likelihood=RiskLikelihood.HIGH,
                    impact=RiskImpact.MAJOR if carbon_price_exposure > 1000000 else RiskImpact.MODERATE,
                    risk_score=LIKELIHOOD_SCORE[RiskLikelihood.HIGH] * IMPACT_SCORE[RiskImpact.MAJOR],
                    time_horizon_years=3,
                    potential_exposure_usd=carbon_price_exposure,
                    mitigation_strategies=[
                        "Accelerate decarbonization investments",
                        "Hedge with carbon offsets",
                        "Implement internal carbon pricing"
                    ]
                ))

                # Technology risk: Stranded assets
                risks.append(TransitionRisk(
                    risk_id=deterministic_id({"type": "tech_stranded"}, "trisk_"),
                    name="Stranded Asset Risk",
                    risk_type=TransitionRiskType.TECHNOLOGY,
                    description="Risk of fossil fuel assets becoming obsolete",
                    likelihood=RiskLikelihood.MEDIUM,
                    impact=RiskImpact.MODERATE,
                    risk_score=LIKELIHOOD_SCORE[RiskLikelihood.MEDIUM] * IMPACT_SCORE[RiskImpact.MODERATE],
                    time_horizon_years=7,
                    mitigation_strategies=[
                        "Review asset depreciation schedules",
                        "Plan early retirement of high-carbon assets",
                        "Transition to low-carbon alternatives"
                    ]
                ))

                # Market risk: Customer preferences
                risks.append(TransitionRisk(
                    risk_id=deterministic_id({"type": "market_customer"}, "trisk_"),
                    name="Customer Preference Shift",
                    risk_type=TransitionRiskType.MARKET,
                    description="Risk of losing customers to low-carbon competitors",
                    likelihood=RiskLikelihood.HIGH,
                    impact=RiskImpact.MAJOR,
                    risk_score=LIKELIHOOD_SCORE[RiskLikelihood.HIGH] * IMPACT_SCORE[RiskImpact.MAJOR],
                    time_horizon_years=5,
                    potential_exposure_usd=tr_input.revenue_musd * 1000000 * 0.15,
                    mitigation_strategies=[
                        "Develop low-carbon product offerings",
                        "Communicate sustainability progress",
                        "Obtain environmental certifications"
                    ]
                ))

                # Reputation risk
                risks.append(TransitionRisk(
                    risk_id=deterministic_id({"type": "reputation"}, "trisk_"),
                    name="Reputation Risk",
                    risk_type=TransitionRiskType.REPUTATION,
                    description="Risk of reputational damage from climate inaction",
                    likelihood=RiskLikelihood.MEDIUM,
                    impact=RiskImpact.MODERATE,
                    risk_score=LIKELIHOOD_SCORE[RiskLikelihood.MEDIUM] * IMPACT_SCORE[RiskImpact.MODERATE],
                    time_horizon_years=2,
                    mitigation_strategies=[
                        "Set and communicate science-based targets",
                        "Increase climate disclosure (TCFD/CDP)",
                        "Engage with stakeholders proactively"
                    ]
                ))

                # Calculate summary
                total_exposure = sum(r.potential_exposure_usd or 0 for r in risks)
                high_priority = sum(1 for r in risks if r.risk_score >= 12)
                avg_score = sum(r.risk_score for r in risks) / len(risks) if risks else 0

                risks_by_type = {}
                for r in risks:
                    rt = r.risk_type.value
                    risks_by_type[rt] = risks_by_type.get(rt, 0) + 1

                assessment = TransitionRiskAssessment(
                    assessment_id=deterministic_id({"org": tr_input.organization_name}, "trassess_"),
                    organization_name=tr_input.organization_name,
                    risks=risks,
                    total_risks=len(risks),
                    high_priority_risks=high_priority,
                    total_exposure_usd=total_exposure,
                    average_risk_score=avg_score,
                    risks_by_type=risks_by_type
                )
                assessment.provenance_hash = content_hash(assessment.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Identified {len(risks)} transition risks")
                calculation_trace.append(f"High priority risks: {high_priority}")

                self._capture_audit_entry(
                    operation="assess",
                    inputs=inputs,
                    outputs={"risks": len(risks), "high_priority": high_priority},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "assess",
                    "success": True,
                    "assessment": assessment.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {tr_input.operation}")

        except Exception as e:
            self.logger.error(f"Assessment failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
