# -*- coding: utf-8 -*-
"""
GL-ADAPT-WAT-002: Flood Risk Agent
==================================

Adaptation agent for flood risk assessment and planning.

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


class FloodType(str, Enum):
    RIVERINE = "riverine"
    COASTAL = "coastal"
    PLUVIAL = "pluvial"
    GROUNDWATER = "groundwater"


class RiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FloodHazard(BaseModel):
    """Flood hazard definition."""
    hazard_id: str
    flood_type: FloodType
    return_period_years: float
    flood_depth_m: float
    flood_extent_km2: float
    flow_velocity_m_s: Optional[float] = None


class FloodVulnerability(BaseModel):
    """Vulnerability assessment for area."""
    area_id: str
    area_name: str
    population: int
    residential_buildings: int
    commercial_buildings: int
    critical_infrastructure: int
    property_value: float
    vulnerability_score: float  # 0-1


class ExposureData(BaseModel):
    """Exposure data for flood zone."""
    zone_id: str
    flood_zone_type: str  # e.g., AE, VE, X
    area_km2: float
    population_exposed: int
    properties_exposed: int
    economic_value_exposed: float


class FloodRiskInput(BaseModel):
    """Input for flood risk analysis."""
    region_id: str
    hazards: List[FloodHazard]
    vulnerabilities: List[FloodVulnerability]
    exposure_data: List[ExposureData] = Field(default_factory=list)
    climate_factor: float = Field(default=1.0, description="Climate change multiplier")
    historical_flood_events: List[Dict[str, Any]] = Field(default_factory=list)


class RiskScore(BaseModel):
    """Risk score for area."""
    area_id: str
    hazard_score: float
    vulnerability_score: float
    exposure_score: float
    total_risk_score: float
    risk_level: RiskLevel
    expected_annual_damage: float


class MitigationMeasure(BaseModel):
    """Flood mitigation measure."""
    measure_id: str
    measure_type: str
    description: str
    risk_reduction_percent: float
    estimated_cost: float
    benefit_cost_ratio: float
    implementation_time_years: float


class FloodRiskOutput(BaseModel):
    """Output from flood risk analysis."""
    region_id: str
    overall_risk_level: RiskLevel
    total_population_at_risk: int
    total_property_at_risk: float
    expected_annual_damage: float
    risk_scores: List[RiskScore]
    high_risk_areas: List[str]
    mitigation_measures: List[MitigationMeasure]
    climate_adjusted_risk_increase_percent: float
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class FloodRiskAgent(BaseAgent):
    """
    GL-ADAPT-WAT-002: Flood Risk Agent

    Assesses flood risks and recommends mitigation measures.
    """

    AGENT_ID = "GL-ADAPT-WAT-002"
    AGENT_NAME = "Flood Risk Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Flood risk assessment and mitigation planning",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            fr_input = FloodRiskInput(**input_data)
            risk_scores = []
            high_risk_areas = []
            total_ead = 0.0

            for vuln in fr_input.vulnerabilities:
                # Calculate hazard score based on flood depth and extent
                max_depth = max((h.flood_depth_m for h in fr_input.hazards), default=0)
                hazard_score = min(1.0, max_depth / 3.0)  # Normalize to 3m max

                # Vulnerability score from input
                vulnerability_score = vuln.vulnerability_score

                # Exposure score based on economic value
                max_value = max((v.property_value for v in fr_input.vulnerabilities), default=1)
                exposure_score = vuln.property_value / max_value if max_value > 0 else 0

                # Total risk score (geometric mean)
                total_score = (hazard_score * vulnerability_score * exposure_score) ** (1/3)

                # Apply climate factor
                total_score *= fr_input.climate_factor

                # Determine risk level
                if total_score > 0.8:
                    risk_level = RiskLevel.VERY_HIGH
                elif total_score > 0.6:
                    risk_level = RiskLevel.HIGH
                elif total_score > 0.4:
                    risk_level = RiskLevel.MEDIUM
                elif total_score > 0.2:
                    risk_level = RiskLevel.LOW
                else:
                    risk_level = RiskLevel.VERY_LOW

                # Expected annual damage (simplified)
                # EAD = probability * consequence
                probability = 1 / 100  # Assume 100-year base event
                ead = probability * vuln.property_value * total_score

                score = RiskScore(
                    area_id=vuln.area_id,
                    hazard_score=round(hazard_score, 3),
                    vulnerability_score=round(vulnerability_score, 3),
                    exposure_score=round(exposure_score, 3),
                    total_risk_score=round(total_score, 3),
                    risk_level=risk_level,
                    expected_annual_damage=round(ead, 0),
                )
                risk_scores.append(score)
                total_ead += ead

                if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                    high_risk_areas.append(vuln.area_id)

            # Determine overall risk
            avg_score = sum(s.total_risk_score for s in risk_scores) / max(1, len(risk_scores))
            if avg_score > 0.6:
                overall_risk = RiskLevel.HIGH
            elif avg_score > 0.4:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW

            # Total exposure
            total_pop = sum(v.population for v in fr_input.vulnerabilities)
            total_property = sum(v.property_value for v in fr_input.vulnerabilities)

            # Climate adjustment
            climate_increase = (fr_input.climate_factor - 1) * 100

            # Mitigation measures
            measures = []
            if high_risk_areas:
                measures.append(MitigationMeasure(
                    measure_id="MIT-001",
                    measure_type="structural",
                    description="Construct flood walls and levees in high-risk areas",
                    risk_reduction_percent=60,
                    estimated_cost=total_ead * 10,
                    benefit_cost_ratio=2.5,
                    implementation_time_years=3,
                ))
                measures.append(MitigationMeasure(
                    measure_id="MIT-002",
                    measure_type="nature_based",
                    description="Restore wetlands and create flood storage areas",
                    risk_reduction_percent=30,
                    estimated_cost=total_ead * 5,
                    benefit_cost_ratio=4.0,
                    implementation_time_years=5,
                ))
            measures.append(MitigationMeasure(
                measure_id="MIT-003",
                measure_type="non_structural",
                description="Implement early warning systems and evacuation planning",
                risk_reduction_percent=20,
                estimated_cost=total_ead * 0.5,
                benefit_cost_ratio=8.0,
                implementation_time_years=1,
            ))

            provenance_hash = hashlib.sha256(
                json.dumps({"region": fr_input.region_id, "ead": total_ead}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = FloodRiskOutput(
                region_id=fr_input.region_id,
                overall_risk_level=overall_risk,
                total_population_at_risk=total_pop,
                total_property_at_risk=round(total_property, 0),
                expected_annual_damage=round(total_ead, 0),
                risk_scores=risk_scores,
                high_risk_areas=high_risk_areas,
                mitigation_measures=measures,
                climate_adjusted_risk_increase_percent=round(climate_increase, 1),
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Flood risk analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
