# -*- coding: utf-8 -*-
"""
GL-ADAPT-WAT-004: Water Security Planner Agent
=============================================

Adaptation agent for comprehensive water security planning.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class SecurityDimension(str, Enum):
    AVAILABILITY = "availability"
    ACCESSIBILITY = "accessibility"
    QUALITY = "quality"
    AFFORDABILITY = "affordability"
    GOVERNANCE = "governance"


class SecurityLevel(str, Enum):
    SECURE = "secure"
    MODERATELY_SECURE = "moderately_secure"
    INSECURE = "insecure"
    HIGHLY_INSECURE = "highly_insecure"


class DimensionScore(BaseModel):
    """Score for a security dimension."""
    dimension: SecurityDimension
    score: float  # 0-100
    level: SecurityLevel
    key_indicators: List[str]
    gaps_identified: List[str]


class SecurityScorecard(BaseModel):
    """Overall security scorecard."""
    overall_score: float
    overall_level: SecurityLevel
    dimension_scores: List[DimensionScore]
    strengths: List[str]
    weaknesses: List[str]


class WaterSecurityInput(BaseModel):
    """Input for water security analysis."""
    region_id: str
    # Availability indicators
    per_capita_availability_m3_year: float
    supply_reliability_percent: float
    drought_vulnerability: float  # 0-1
    # Accessibility indicators
    population_with_access_percent: float
    hours_of_supply_per_day: float
    # Quality indicators
    drinking_water_compliance_percent: float
    wastewater_treatment_percent: float
    # Affordability indicators
    water_cost_as_income_percent: float
    # Governance indicators
    regulation_effectiveness_score: float  # 0-10
    stakeholder_participation_score: float  # 0-10


class StrategicAction(BaseModel):
    """Strategic action for water security."""
    action_id: str
    dimension: SecurityDimension
    action_title: str
    description: str
    impact_score: float
    investment_required: float
    timeframe: str


class WaterSecurityOutput(BaseModel):
    """Output from water security analysis."""
    region_id: str
    scorecard: SecurityScorecard
    strategic_actions: List[StrategicAction]
    investment_priorities: List[str]
    policy_recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class WaterSecurityPlannerAgent(BaseAgent):
    """
    GL-ADAPT-WAT-004: Water Security Planner Agent

    Comprehensive water security assessment and planning.
    """

    AGENT_ID = "GL-ADAPT-WAT-004"
    AGENT_NAME = "Water Security Planner Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water security planning and assessment",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            ws_input = WaterSecurityInput(**input_data)
            dimension_scores = []
            actions = []
            action_id = 1

            # Availability score
            avail_score = min(100, (
                (ws_input.per_capita_availability_m3_year / 1700 * 40) +
                (ws_input.supply_reliability_percent * 0.4) +
                ((1 - ws_input.drought_vulnerability) * 20)
            ))
            avail_level = self._score_to_level(avail_score)
            avail_gaps = []
            if ws_input.per_capita_availability_m3_year < 1700:
                avail_gaps.append("Below water stress threshold")
            if ws_input.drought_vulnerability > 0.5:
                avail_gaps.append("High drought vulnerability")

            dimension_scores.append(DimensionScore(
                dimension=SecurityDimension.AVAILABILITY,
                score=round(avail_score, 1),
                level=avail_level,
                key_indicators=["per_capita_availability", "supply_reliability", "drought_vulnerability"],
                gaps_identified=avail_gaps,
            ))

            if avail_gaps:
                actions.append(StrategicAction(
                    action_id=f"SA-{action_id}",
                    dimension=SecurityDimension.AVAILABILITY,
                    action_title="Diversify water sources",
                    description="Develop alternative water sources to improve supply reliability",
                    impact_score=8.0,
                    investment_required=50000000,
                    timeframe="5-10 years",
                ))
                action_id += 1

            # Accessibility score
            access_score = (
                ws_input.population_with_access_percent * 0.6 +
                (ws_input.hours_of_supply_per_day / 24 * 100) * 0.4
            )
            access_level = self._score_to_level(access_score)
            access_gaps = []
            if ws_input.population_with_access_percent < 95:
                access_gaps.append("Service coverage below target")
            if ws_input.hours_of_supply_per_day < 20:
                access_gaps.append("Intermittent water supply")

            dimension_scores.append(DimensionScore(
                dimension=SecurityDimension.ACCESSIBILITY,
                score=round(access_score, 1),
                level=access_level,
                key_indicators=["population_with_access", "hours_of_supply"],
                gaps_identified=access_gaps,
            ))

            # Quality score
            quality_score = (
                ws_input.drinking_water_compliance_percent * 0.6 +
                ws_input.wastewater_treatment_percent * 0.4
            )
            quality_level = self._score_to_level(quality_score)
            quality_gaps = []
            if ws_input.drinking_water_compliance_percent < 95:
                quality_gaps.append("Drinking water quality compliance issues")
            if ws_input.wastewater_treatment_percent < 80:
                quality_gaps.append("Wastewater treatment coverage insufficient")

            dimension_scores.append(DimensionScore(
                dimension=SecurityDimension.QUALITY,
                score=round(quality_score, 1),
                level=quality_level,
                key_indicators=["drinking_water_compliance", "wastewater_treatment"],
                gaps_identified=quality_gaps,
            ))

            # Affordability score
            afford_score = max(0, 100 - ws_input.water_cost_as_income_percent * 20)
            afford_level = self._score_to_level(afford_score)
            afford_gaps = []
            if ws_input.water_cost_as_income_percent > 3:
                afford_gaps.append("Water costs exceed affordability threshold")

            dimension_scores.append(DimensionScore(
                dimension=SecurityDimension.AFFORDABILITY,
                score=round(afford_score, 1),
                level=afford_level,
                key_indicators=["water_cost_as_income"],
                gaps_identified=afford_gaps,
            ))

            # Governance score
            gov_score = (
                ws_input.regulation_effectiveness_score * 5 +
                ws_input.stakeholder_participation_score * 5
            )
            gov_level = self._score_to_level(gov_score)
            gov_gaps = []
            if ws_input.regulation_effectiveness_score < 6:
                gov_gaps.append("Regulatory effectiveness needs improvement")
            if ws_input.stakeholder_participation_score < 6:
                gov_gaps.append("Limited stakeholder engagement")

            dimension_scores.append(DimensionScore(
                dimension=SecurityDimension.GOVERNANCE,
                score=round(gov_score, 1),
                level=gov_level,
                key_indicators=["regulation_effectiveness", "stakeholder_participation"],
                gaps_identified=gov_gaps,
            ))

            # Overall score
            overall_score = sum(d.score for d in dimension_scores) / len(dimension_scores)
            overall_level = self._score_to_level(overall_score)

            # Identify strengths and weaknesses
            sorted_dims = sorted(dimension_scores, key=lambda d: d.score, reverse=True)
            strengths = [f"{sorted_dims[0].dimension.value}: {sorted_dims[0].score:.0f}/100"]
            weaknesses = [f"{sorted_dims[-1].dimension.value}: {sorted_dims[-1].score:.0f}/100"]

            scorecard = SecurityScorecard(
                overall_score=round(overall_score, 1),
                overall_level=overall_level,
                dimension_scores=dimension_scores,
                strengths=strengths,
                weaknesses=weaknesses,
            )

            # Investment priorities
            priorities = [
                d.dimension.value for d in sorted(dimension_scores, key=lambda d: d.score)[:2]
            ]

            # Policy recommendations
            policies = [
                "Integrate water security into national development planning",
                "Strengthen regulatory frameworks for water resource management",
                "Increase investment in water infrastructure resilience",
            ]

            provenance_hash = hashlib.sha256(
                json.dumps({"region": ws_input.region_id, "score": overall_score}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = WaterSecurityOutput(
                region_id=ws_input.region_id,
                scorecard=scorecard,
                strategic_actions=actions,
                investment_priorities=priorities,
                policy_recommendations=policies,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Water security analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _score_to_level(self, score: float) -> SecurityLevel:
        if score >= 80:
            return SecurityLevel.SECURE
        elif score >= 60:
            return SecurityLevel.MODERATELY_SECURE
        elif score >= 40:
            return SecurityLevel.INSECURE
        else:
            return SecurityLevel.HIGHLY_INSECURE
