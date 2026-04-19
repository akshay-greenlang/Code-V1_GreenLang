"""GL-098: Innovation Scout Agent (INNOVATION-SCOUT).

Scouts emerging technologies for energy innovation.

Standards: TRL Framework, Technology Roadmapping
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TechnologyCategory(str, Enum):
    GENERATION = "GENERATION"
    STORAGE = "STORAGE"
    EFFICIENCY = "EFFICIENCY"
    DIGITALIZATION = "DIGITALIZATION"
    MATERIALS = "MATERIALS"


class Technology(BaseModel):
    tech_id: str
    name: str
    category: TechnologyCategory
    trl: int = Field(ge=1, le=9)
    market_potential_usd: float = Field(default=0, ge=0)
    time_to_market_years: int = Field(default=5, ge=0)
    cost_reduction_potential_pct: float = Field(default=0, ge=0, le=100)


class InnovationScoutInput(BaseModel):
    scout_id: str
    focus_areas: List[TechnologyCategory] = Field(default_factory=list)
    technologies: List[Technology] = Field(default_factory=list)
    min_trl: int = Field(default=5, ge=1, le=9)
    investment_horizon_years: int = Field(default=5, ge=1)
    risk_appetite: str = Field(default="MODERATE")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TechnologyAssessment(BaseModel):
    tech_name: str
    category: str
    readiness_score: float
    market_score: float
    strategic_fit_score: float
    overall_score: float
    recommendation: str


class InnovationScoutOutput(BaseModel):
    scout_id: str
    technologies_assessed: int
    high_potential: int
    watch_list: int
    assessments: List[TechnologyAssessment]
    top_opportunities: List[str]
    emerging_trends: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class InnovationScoutAgent:
    AGENT_ID = "GL-098"
    AGENT_NAME = "INNOVATION-SCOUT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"InnovationScoutAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = InnovationScoutInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: InnovationScoutInput) -> InnovationScoutOutput:
        recommendations = []
        assessments = []
        high_potential = 0
        watch_list = 0
        opportunities = []
        trends = set()

        for tech in inp.technologies:
            # Readiness score (TRL-based)
            readiness = tech.trl / 9 * 100

            # Market score
            if tech.market_potential_usd >= 10e9:
                market = 100
            elif tech.market_potential_usd >= 1e9:
                market = 80
            elif tech.market_potential_usd >= 100e6:
                market = 60
            else:
                market = 40

            # Strategic fit
            fit = 100 if tech.category in inp.focus_areas else 50
            if tech.time_to_market_years <= inp.investment_horizon_years:
                fit += 20

            # Risk adjustment
            risk_mult = {"LOW": 0.8, "MODERATE": 1.0, "HIGH": 1.2}.get(inp.risk_appetite, 1.0)
            if tech.trl < 5:
                fit *= risk_mult

            fit = min(100, fit)

            # Overall score
            overall = (readiness * 0.3 + market * 0.3 + fit * 0.4)

            # Recommendation
            if overall >= 75 and tech.trl >= inp.min_trl:
                rec = "PURSUE"
                high_potential += 1
                opportunities.append(tech.name)
            elif overall >= 50:
                rec = "MONITOR"
                watch_list += 1
            else:
                rec = "DEPRIORITIZE"

            assessments.append(TechnologyAssessment(
                tech_name=tech.name,
                category=tech.category.value,
                readiness_score=round(readiness, 1),
                market_score=round(market, 1),
                strategic_fit_score=round(fit, 1),
                overall_score=round(overall, 1),
                recommendation=rec
            ))

            trends.add(tech.category.value)

        # Sort by overall score
        assessments.sort(key=lambda x: -x.overall_score)

        # Emerging trends
        emerging = list(trends)[:3]

        # Recommendations
        if high_potential == 0:
            recommendations.append("No high-potential technologies - expand search criteria")
        elif high_potential > 5:
            recommendations.append(f"{high_potential} high-potential technologies - prioritize portfolio")

        if watch_list > 10:
            recommendations.append(f"{watch_list} technologies on watch list - establish monitoring cadence")

        low_trl = [t for t in inp.technologies if t.trl < 5]
        if low_trl and inp.risk_appetite == "HIGH":
            recommendations.append(f"{len(low_trl)} early-stage (TRL<5) opportunities for risk-tolerant investment")

        if not inp.focus_areas:
            recommendations.append("Define focus areas to improve strategic alignment scoring")

        calc_hash = hashlib.sha256(json.dumps({
            "scout": inp.scout_id,
            "assessed": len(inp.technologies),
            "high_potential": high_potential
        }).encode()).hexdigest()

        return InnovationScoutOutput(
            scout_id=inp.scout_id,
            technologies_assessed=len(inp.technologies),
            high_potential=high_potential,
            watch_list=watch_list,
            assessments=assessments,
            top_opportunities=opportunities[:5],
            emerging_trends=emerging,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-098", "name": "INNOVATION-SCOUT", "version": "1.0.0",
    "summary": "Emerging technology scouting for energy",
    "standards": [{"ref": "TRL Framework"}, {"ref": "Technology Roadmapping"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
