# -*- coding: utf-8 -*-
"""
GL-ADAPT-WAT-003: Water Infrastructure Resilience Agent
======================================================

Adaptation agent for water infrastructure climate resilience.

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


class AssetType(str, Enum):
    TREATMENT_PLANT = "treatment_plant"
    PUMP_STATION = "pump_station"
    RESERVOIR = "reservoir"
    PIPELINE = "pipeline"
    WELL = "well"


class ClimateHazard(str, Enum):
    FLOODING = "flooding"
    DROUGHT = "drought"
    EXTREME_HEAT = "extreme_heat"
    SEA_LEVEL_RISE = "sea_level_rise"
    WILDFIRE = "wildfire"


class ResilienceRating(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class InfrastructureAsset(BaseModel):
    """Infrastructure asset definition."""
    asset_id: str
    asset_name: str
    asset_type: AssetType
    construction_year: int
    design_life_years: int
    replacement_value: float
    criticality_score: float  # 1-10
    current_condition: float  # 1-10
    exposed_hazards: List[ClimateHazard]


class ResilienceAssessment(BaseModel):
    """Resilience assessment for asset."""
    asset_id: str
    overall_resilience_score: float
    resilience_rating: ResilienceRating
    hazard_vulnerabilities: Dict[str, float]
    remaining_useful_life_years: float
    adaptation_priority: str


class AdaptationMeasure(BaseModel):
    """Adaptation measure for asset."""
    measure_id: str
    asset_id: str
    measure_type: str
    description: str
    resilience_improvement: float
    estimated_cost: float
    implementation_priority: str


class InfraResilienceInput(BaseModel):
    """Input for infrastructure resilience analysis."""
    system_id: str
    assets: List[InfrastructureAsset]
    climate_scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    planning_horizon_years: int = Field(default=30)


class InfraResilienceOutput(BaseModel):
    """Output from infrastructure resilience analysis."""
    system_id: str
    overall_system_resilience: float
    system_rating: ResilienceRating
    total_assets_assessed: int
    critical_assets_at_risk: int
    total_adaptation_investment_needed: float
    assessments: List[ResilienceAssessment]
    adaptation_measures: List[AdaptationMeasure]
    priority_actions: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class WaterInfrastructureResilienceAgent(BaseAgent):
    """
    GL-ADAPT-WAT-003: Water Infrastructure Resilience Agent

    Assesses water infrastructure climate resilience.
    """

    AGENT_ID = "GL-ADAPT-WAT-003"
    AGENT_NAME = "Water Infrastructure Resilience Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water infrastructure climate resilience assessment",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            ir_input = InfraResilienceInput(**input_data)
            assessments = []
            measures = []
            critical_at_risk = 0
            total_investment = 0.0

            current_year = datetime.now().year

            for asset in ir_input.assets:
                # Calculate remaining useful life
                age = current_year - asset.construction_year
                rul = max(0, asset.design_life_years - age)

                # Calculate hazard vulnerabilities
                hazard_vulns = {}
                for hazard in asset.exposed_hazards:
                    # Base vulnerability from condition
                    base_vuln = 1 - (asset.current_condition / 10)
                    # Increase for older assets
                    age_factor = min(1, age / asset.design_life_years)
                    vuln = min(1.0, base_vuln + age_factor * 0.3)
                    hazard_vulns[hazard.value] = round(vuln, 2)

                # Overall resilience score
                avg_vuln = sum(hazard_vulns.values()) / max(1, len(hazard_vulns))
                resilience_score = (1 - avg_vuln) * 10

                # Rating
                if resilience_score >= 8:
                    rating = ResilienceRating.EXCELLENT
                elif resilience_score >= 6:
                    rating = ResilienceRating.GOOD
                elif resilience_score >= 4:
                    rating = ResilienceRating.FAIR
                elif resilience_score >= 2:
                    rating = ResilienceRating.POOR
                else:
                    rating = ResilienceRating.CRITICAL

                # Priority
                if asset.criticality_score > 7 and resilience_score < 5:
                    priority = "immediate"
                    critical_at_risk += 1
                elif resilience_score < 4:
                    priority = "high"
                elif resilience_score < 6:
                    priority = "medium"
                else:
                    priority = "low"

                assessment = ResilienceAssessment(
                    asset_id=asset.asset_id,
                    overall_resilience_score=round(resilience_score, 1),
                    resilience_rating=rating,
                    hazard_vulnerabilities=hazard_vulns,
                    remaining_useful_life_years=rul,
                    adaptation_priority=priority,
                )
                assessments.append(assessment)

                # Generate adaptation measures for at-risk assets
                if priority in ["immediate", "high"]:
                    measure_cost = asset.replacement_value * 0.15
                    total_investment += measure_cost

                    measure = AdaptationMeasure(
                        measure_id=f"AM-{asset.asset_id}",
                        asset_id=asset.asset_id,
                        measure_type="hardening",
                        description=f"Climate-proof {asset.asset_type.value} against identified hazards",
                        resilience_improvement=3.0,
                        estimated_cost=round(measure_cost, 0),
                        implementation_priority=priority,
                    )
                    measures.append(measure)

            # System-level metrics
            avg_resilience = sum(a.overall_resilience_score for a in assessments) / max(1, len(assessments))

            if avg_resilience >= 7:
                system_rating = ResilienceRating.GOOD
            elif avg_resilience >= 5:
                system_rating = ResilienceRating.FAIR
            else:
                system_rating = ResilienceRating.POOR

            # Priority actions
            priority_actions = []
            if critical_at_risk > 0:
                priority_actions.append(f"Immediately address {critical_at_risk} critical assets at risk")
            priority_actions.append("Develop long-term infrastructure adaptation plan")
            priority_actions.append("Integrate climate projections into capital planning")

            provenance_hash = hashlib.sha256(
                json.dumps({"system": ir_input.system_id, "assets": len(ir_input.assets)}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = InfraResilienceOutput(
                system_id=ir_input.system_id,
                overall_system_resilience=round(avg_resilience, 1),
                system_rating=system_rating,
                total_assets_assessed=len(ir_input.assets),
                critical_assets_at_risk=critical_at_risk,
                total_adaptation_investment_needed=round(total_investment, 0),
                assessments=assessments,
                adaptation_measures=measures,
                priority_actions=priority_actions,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Infrastructure resilience analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
