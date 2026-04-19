"""GL-085: Innovation Scout Agent (INNOVATION-SCOUT).

Identifies emerging technologies and innovation opportunities.

Standards: TRL Framework, ISO 56002
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TechnologyDomain(str, Enum):
    ELECTRIFICATION = "ELECTRIFICATION"
    HYDROGEN = "HYDROGEN"
    CARBON_CAPTURE = "CARBON_CAPTURE"
    HEAT_PUMPS = "HEAT_PUMPS"
    THERMAL_STORAGE = "THERMAL_STORAGE"
    DIGITAL_TWIN = "DIGITAL_TWIN"
    AI_OPTIMIZATION = "AI_OPTIMIZATION"


class EmergingTechnology(BaseModel):
    technology_id: str
    name: str
    domain: TechnologyDomain
    trl_level: int = Field(ge=1, le=9)
    potential_reduction_pct: float = Field(ge=0, le=100)
    implementation_cost_range: str
    timeline_years: int = Field(ge=1)
    risk_level: str


class InnovationInput(BaseModel):
    facility_id: str
    current_technologies: List[str] = Field(default_factory=list)
    innovation_budget_usd: float = Field(default=100000, ge=0)
    risk_appetite: str = Field(default="MODERATE")
    priority_domains: List[TechnologyDomain] = Field(default_factory=list)
    emission_reduction_target_pct: float = Field(default=50, ge=0, le=100)
    emerging_technologies: List[EmergingTechnology] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TechnologyRecommendation(BaseModel):
    technology_name: str
    domain: str
    fit_score: float
    strategic_value: str
    next_steps: List[str]
    estimated_roi_pct: float
    implementation_timeline: str


class InnovationOutput(BaseModel):
    facility_id: str
    technologies_evaluated: int
    high_potential_count: int
    recommendations: List[TechnologyRecommendation]
    innovation_readiness_score: float
    technology_gap_areas: List[str]
    pilot_candidates: List[str]
    watch_list: List[str]
    strategic_recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class InnovationScoutAgent:
    AGENT_ID = "GL-085B"
    AGENT_NAME = "INNOVATION-SCOUT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"InnovationScoutAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = InnovationInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _score_technology(self, tech: EmergingTechnology, inp: InnovationInput) -> float:
        """Score a technology based on fit criteria."""
        score = 0

        # TRL scoring (higher TRL = more ready)
        score += tech.trl_level * 5

        # Priority domain match
        if tech.domain in inp.priority_domains:
            score += 20

        # Reduction potential vs target
        if tech.potential_reduction_pct >= inp.emission_reduction_target_pct * 0.5:
            score += 15

        # Risk alignment
        risk_scores = {"LOW": 3, "MEDIUM": 2, "HIGH": 1}
        appetite_mult = {"LOW": 0.5, "MODERATE": 1.0, "HIGH": 1.5}
        base_risk = risk_scores.get(tech.risk_level, 2)
        mult = appetite_mult.get(inp.risk_appetite, 1.0)
        score += base_risk * 5 * mult

        # Timeline (shorter = better)
        score += max(0, 10 - tech.timeline_years)

        return min(100, score)

    def _process(self, inp: InnovationInput) -> InnovationOutput:
        strategic_recs = []

        # Score all technologies
        scored = []
        for tech in inp.emerging_technologies:
            score = self._score_technology(tech, inp)
            scored.append((tech, score))

        # Sort by score
        scored.sort(key=lambda x: -x[1])

        # Generate recommendations
        recommendations = []
        pilot_candidates = []
        watch_list = []

        for tech, score in scored:
            if score >= 70:
                value = "HIGH"
                if tech.trl_level >= 7:
                    pilot_candidates.append(tech.name)
            elif score >= 50:
                value = "MEDIUM"
                watch_list.append(tech.name)
            else:
                value = "LOW"
                continue

            # Estimate ROI (simplified)
            roi = tech.potential_reduction_pct * 2 - tech.timeline_years * 5

            # Next steps
            steps = []
            if tech.trl_level < 5:
                steps.append("Monitor development progress")
            elif tech.trl_level < 7:
                steps.append("Engage with technology providers")
                steps.append("Evaluate pilot opportunity")
            else:
                steps.append("Request detailed proposal")
                steps.append("Conduct site assessment")
                steps.append("Develop business case")

            recommendations.append(TechnologyRecommendation(
                technology_name=tech.name,
                domain=tech.domain.value,
                fit_score=round(score, 1),
                strategic_value=value,
                next_steps=steps,
                estimated_roi_pct=round(max(0, roi), 1),
                implementation_timeline=f"{tech.timeline_years} years"
            ))

        # High potential count
        high_potential = len([s for s in scored if s[1] >= 70])

        # Technology gaps
        current_domains = set()
        for tech_name in inp.current_technologies:
            for domain in TechnologyDomain:
                if domain.value.lower() in tech_name.lower():
                    current_domains.add(domain)

        gaps = [d.value for d in inp.priority_domains if d not in current_domains]

        # Innovation readiness score
        readiness = 50
        if inp.innovation_budget_usd > 50000:
            readiness += 10
        if len(inp.current_technologies) > 3:
            readiness += 10
        if inp.risk_appetite == "HIGH":
            readiness += 10
        if pilot_candidates:
            readiness += 20

        # Strategic recommendations
        if not inp.emerging_technologies:
            strategic_recs.append("Expand technology scouting - no emerging technologies evaluated")
        if high_potential == 0:
            strategic_recs.append("No high-fit technologies found - broaden search criteria")
        if pilot_candidates:
            strategic_recs.append(f"Launch pilots for: {', '.join(pilot_candidates[:3])}")
        if gaps:
            strategic_recs.append(f"Technology gaps in priority areas: {', '.join(gaps)}")
        if inp.innovation_budget_usd < 50000:
            strategic_recs.append("Limited innovation budget - focus on low-cost pilots")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "evaluated": len(inp.emerging_technologies),
            "high_potential": high_potential
        }).encode()).hexdigest()

        return InnovationOutput(
            facility_id=inp.facility_id,
            technologies_evaluated=len(inp.emerging_technologies),
            high_potential_count=high_potential,
            recommendations=recommendations[:5],
            innovation_readiness_score=round(readiness, 1),
            technology_gap_areas=gaps,
            pilot_candidates=pilot_candidates[:3],
            watch_list=watch_list[:5],
            strategic_recommendations=strategic_recs,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-085B", "name": "INNOVATION-SCOUT", "version": "1.0.0",
    "summary": "Emerging technology identification and evaluation",
    "standards": [{"ref": "TRL Framework"}, {"ref": "ISO 56002"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
