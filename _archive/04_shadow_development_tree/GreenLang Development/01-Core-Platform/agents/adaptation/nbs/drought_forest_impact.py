# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-004: Drought Impact on Forests Agent
==================================================

Assesses drought impacts on forest ecosystems and recommends adaptation measures.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class DroughtSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class ForestType(str, Enum):
    BOREAL = "boreal"
    TEMPERATE = "temperate"
    TROPICAL = "tropical"
    MEDITERRANEAN = "mediterranean"
    MONTANE = "montane"


class ForestSite(BaseModel):
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    forest_type: ForestType = Field(...)
    avg_annual_precipitation_mm: float = Field(default=800, ge=0)
    soil_water_holding_capacity: float = Field(default=0.5, ge=0, le=1)
    tree_species_diversity: int = Field(default=10, ge=1)
    avg_tree_age_years: int = Field(default=50, ge=1)


class DroughtImpactAssessment(BaseModel):
    site_id: str = Field(...)
    drought_vulnerability_score: float = Field(..., ge=0, le=100)
    mortality_risk_pct: float = Field(..., ge=0, le=100)
    growth_reduction_pct: float = Field(..., ge=0, le=100)
    recovery_time_years: int = Field(...)
    adaptation_measures: List[str] = Field(...)
    priority: str = Field(...)


class DroughtImpactInput(BaseModel):
    project_id: str = Field(...)
    sites: List[ForestSite] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")
    projection_year: int = Field(default=2050, ge=2024, le=2100)


class DroughtImpactOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    high_risk_area_ha: float = Field(...)
    assessments: List[DroughtImpactAssessment] = Field(...)
    overall_vulnerability: DroughtSeverity = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class DroughtForestImpactAgent(BaseAgent):
    """GL-ADAPT-NBS-004: Drought Impact on Forests Agent"""

    AGENT_ID = "GL-ADAPT-NBS-004"
    AGENT_NAME = "Drought Impact on Forests Agent"
    VERSION = "1.0.0"

    FOREST_VULNERABILITY = {
        ForestType.BOREAL: 0.6,
        ForestType.TEMPERATE: 0.5,
        ForestType.TROPICAL: 0.7,
        ForestType.MEDITERRANEAN: 0.8,
        ForestType.MONTANE: 0.55,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Forest drought impact assessment",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = DroughtImpactInput(**input_data)
            assessments = []
            high_risk_area = 0.0

            for site in agent_input.sites:
                assessment = self._assess_site(site, agent_input.climate_scenario)
                assessments.append(assessment)
                if assessment.mortality_risk_pct > 30:
                    high_risk_area += site.area_ha

            total_area = sum(s.area_ha for s in agent_input.sites)
            avg_vulnerability = sum(a.drought_vulnerability_score for a in assessments) / len(assessments)

            if avg_vulnerability >= 80:
                overall = DroughtSeverity.EXTREME
            elif avg_vulnerability >= 60:
                overall = DroughtSeverity.SEVERE
            elif avg_vulnerability >= 40:
                overall = DroughtSeverity.MODERATE
            elif avg_vulnerability >= 20:
                overall = DroughtSeverity.MILD
            else:
                overall = DroughtSeverity.NONE

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = DroughtImpactOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                high_risk_area_ha=high_risk_area,
                assessments=assessments,
                overall_vulnerability=overall,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _assess_site(self, site: ForestSite, scenario: str) -> DroughtImpactAssessment:
        base_vulnerability = self.FOREST_VULNERABILITY.get(site.forest_type, 0.5)
        precipitation_factor = max(0, 1 - (site.avg_annual_precipitation_mm / 1500))
        soil_factor = 1 - site.soil_water_holding_capacity
        diversity_factor = max(0, 1 - (site.tree_species_diversity / 30))
        age_factor = min(1, site.avg_tree_age_years / 150)

        vulnerability_score = (
            base_vulnerability * 0.3 +
            precipitation_factor * 0.25 +
            soil_factor * 0.2 +
            diversity_factor * 0.15 +
            age_factor * 0.1
        ) * 100

        climate_multiplier = 1.3 if scenario == "RCP4.5" else 1.5
        vulnerability_score = min(100, vulnerability_score * climate_multiplier)

        mortality_risk = vulnerability_score * 0.4
        growth_reduction = vulnerability_score * 0.6
        recovery_time = int(5 + vulnerability_score / 10)

        if vulnerability_score >= 70:
            priority = "critical"
            measures = [
                "Species diversification",
                "Thinning for water stress reduction",
                "Drought-resistant genetic material",
                "Soil moisture monitoring",
                "Emergency irrigation infrastructure",
            ]
        elif vulnerability_score >= 50:
            priority = "high"
            measures = [
                "Selective thinning",
                "Mulching and soil improvement",
                "Understory management",
                "Climate monitoring",
            ]
        else:
            priority = "moderate"
            measures = [
                "Regular monitoring",
                "Gradual species transition",
                "Soil health maintenance",
            ]

        return DroughtImpactAssessment(
            site_id=site.site_id,
            drought_vulnerability_score=round(vulnerability_score, 1),
            mortality_risk_pct=round(mortality_risk, 1),
            growth_reduction_pct=round(growth_reduction, 1),
            recovery_time_years=recovery_time,
            adaptation_measures=measures,
            priority=priority,
        )
