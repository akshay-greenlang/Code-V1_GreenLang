# -*- coding: utf-8 -*-
"""
GL-ADAPT-WAT-001: Water Scarcity Risk Agent
==========================================

Adaptation agent for drought and water scarcity risk assessment.

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


class ScarcityRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"


class DroughtIndicator(BaseModel):
    """Drought indicator measurement."""
    indicator_name: str
    value: float
    threshold_moderate: float
    threshold_severe: float
    unit: str
    assessment_date: datetime


class WaterBalance(BaseModel):
    """Water balance for region."""
    region_id: str
    available_supply_m3_year: float
    total_demand_m3_year: float
    renewable_resources_m3_year: float
    storage_capacity_m3: float
    current_storage_m3: float


class ClimateProjection(BaseModel):
    """Climate projection data."""
    scenario: str  # e.g., RCP4.5, RCP8.5, SSP2
    time_horizon: str  # e.g., 2030, 2050
    precipitation_change_percent: float
    temperature_change_c: float
    drought_frequency_change: float


class WaterScarcityInput(BaseModel):
    """Input for water scarcity analysis."""
    region_id: str
    water_balance: WaterBalance
    drought_indicators: List[DroughtIndicator] = Field(default_factory=list)
    climate_projections: List[ClimateProjection] = Field(default_factory=list)
    historical_drought_years: List[int] = Field(default_factory=list)
    population: int = Field(default=0, ge=0)
    economic_activity_index: float = Field(default=1.0)


class ScarcityProjection(BaseModel):
    """Scarcity projection for time horizon."""
    time_horizon: str
    scenario: str
    projected_supply_m3_year: float
    projected_demand_m3_year: float
    supply_demand_gap_m3_year: float
    risk_level: ScarcityRiskLevel
    confidence_percent: float


class AdaptationRecommendation(BaseModel):
    """Adaptation recommendation."""
    recommendation_id: str
    category: str
    description: str
    estimated_water_savings_m3_year: float
    estimated_cost: float
    implementation_timeframe: str
    priority: str


class WaterScarcityOutput(BaseModel):
    """Output from water scarcity analysis."""
    region_id: str
    current_risk_level: ScarcityRiskLevel
    water_stress_index: float
    per_capita_availability_m3_year: float
    storage_days_remaining: float
    drought_indicators_summary: Dict[str, Any]
    future_projections: List[ScarcityProjection]
    adaptation_recommendations: List[AdaptationRecommendation]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class WaterScarcityRiskAgent(BaseAgent):
    """
    GL-ADAPT-WAT-001: Water Scarcity Risk Agent

    Assesses drought and water scarcity risks with climate projections.
    """

    AGENT_ID = "GL-ADAPT-WAT-001"
    AGENT_NAME = "Water Scarcity Risk Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Drought and water scarcity risk assessment",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            ws_input = WaterScarcityInput(**input_data)
            wb = ws_input.water_balance

            # Calculate water stress index
            # Falkenmark: < 1700 m3/capita/year = stress
            per_capita = wb.renewable_resources_m3_year / max(1, ws_input.population)
            stress_index = (wb.total_demand_m3_year / wb.available_supply_m3_year) if wb.available_supply_m3_year > 0 else 999

            # Determine current risk level
            if stress_index > 1.0:
                current_risk = ScarcityRiskLevel.CRITICAL
            elif stress_index > 0.8:
                current_risk = ScarcityRiskLevel.SEVERE
            elif stress_index > 0.6:
                current_risk = ScarcityRiskLevel.HIGH
            elif stress_index > 0.4:
                current_risk = ScarcityRiskLevel.MODERATE
            else:
                current_risk = ScarcityRiskLevel.LOW

            # Storage days
            daily_demand = wb.total_demand_m3_year / 365
            storage_days = wb.current_storage_m3 / daily_demand if daily_demand > 0 else 999

            # Drought indicators summary
            indicators_in_drought = sum(
                1 for d in ws_input.drought_indicators
                if d.value > d.threshold_moderate
            )
            indicators_summary = {
                "total_indicators": len(ws_input.drought_indicators),
                "indicators_in_drought": indicators_in_drought,
                "drought_status": "active" if indicators_in_drought > len(ws_input.drought_indicators) / 2 else "normal",
            }

            # Future projections
            projections = []
            for cp in ws_input.climate_projections:
                # Adjust supply based on precipitation change
                future_supply = wb.available_supply_m3_year * (1 + cp.precipitation_change_percent / 100)
                # Adjust demand based on temperature (higher temp = higher demand)
                future_demand = wb.total_demand_m3_year * (1 + cp.temperature_change_c * 0.02)

                gap = future_demand - future_supply

                if gap > future_supply * 0.3:
                    risk = ScarcityRiskLevel.CRITICAL
                elif gap > future_supply * 0.15:
                    risk = ScarcityRiskLevel.SEVERE
                elif gap > 0:
                    risk = ScarcityRiskLevel.HIGH
                else:
                    risk = ScarcityRiskLevel.MODERATE if cp.precipitation_change_percent < -10 else ScarcityRiskLevel.LOW

                proj = ScarcityProjection(
                    time_horizon=cp.time_horizon,
                    scenario=cp.scenario,
                    projected_supply_m3_year=round(future_supply, 0),
                    projected_demand_m3_year=round(future_demand, 0),
                    supply_demand_gap_m3_year=round(gap, 0),
                    risk_level=risk,
                    confidence_percent=75 if "SSP" in cp.scenario else 70,
                )
                projections.append(proj)

            # Adaptation recommendations
            recommendations = []
            rec_id = 1

            if stress_index > 0.5:
                recommendations.append(AdaptationRecommendation(
                    recommendation_id=f"REC-{rec_id}",
                    category="demand_management",
                    description="Implement water conservation programs and pricing reform",
                    estimated_water_savings_m3_year=wb.total_demand_m3_year * 0.1,
                    estimated_cost=1000000,
                    implementation_timeframe="1-2 years",
                    priority="high",
                ))
                rec_id += 1

            if storage_days < 90:
                recommendations.append(AdaptationRecommendation(
                    recommendation_id=f"REC-{rec_id}",
                    category="storage",
                    description="Increase storage capacity through new reservoirs or aquifer recharge",
                    estimated_water_savings_m3_year=wb.total_demand_m3_year * 0.05,
                    estimated_cost=10000000,
                    implementation_timeframe="3-5 years",
                    priority="high",
                ))
                rec_id += 1

            recommendations.append(AdaptationRecommendation(
                recommendation_id=f"REC-{rec_id}",
                category="diversification",
                description="Develop alternative water sources (recycled, desalination)",
                estimated_water_savings_m3_year=wb.total_demand_m3_year * 0.15,
                estimated_cost=50000000,
                implementation_timeframe="5-10 years",
                priority="medium",
            ))

            provenance_hash = hashlib.sha256(
                json.dumps({"region": ws_input.region_id, "stress": stress_index}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = WaterScarcityOutput(
                region_id=ws_input.region_id,
                current_risk_level=current_risk,
                water_stress_index=round(stress_index, 3),
                per_capita_availability_m3_year=round(per_capita, 0),
                storage_days_remaining=round(storage_days, 1),
                drought_indicators_summary=indicators_summary,
                future_projections=projections,
                adaptation_recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Water scarcity analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
