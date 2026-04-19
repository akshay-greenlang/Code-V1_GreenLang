"""GL-075: Contractor Performance Agent (CONTRACTOR).

Evaluates and manages contractor performance for energy projects.

Standards: ISO 9001, OSHA 1910
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContractorType(str, Enum):
    MECHANICAL = "MECHANICAL"
    ELECTRICAL = "ELECTRICAL"
    CONTROLS = "CONTROLS"
    GENERAL = "GENERAL"
    SPECIALIZED = "SPECIALIZED"


class PerformanceRating(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    SATISFACTORY = "SATISFACTORY"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    UNACCEPTABLE = "UNACCEPTABLE"


class ProjectHistory(BaseModel):
    project_id: str
    project_name: str
    contract_value_usd: float
    actual_cost_usd: float
    scheduled_days: int
    actual_days: int
    quality_score: float = Field(ge=0, le=100)
    safety_incidents: int = Field(ge=0)


class ContractorInput(BaseModel):
    contractor_id: str
    contractor_name: str
    contractor_type: ContractorType = Field(default=ContractorType.MECHANICAL)
    project_history: List[ProjectHistory] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    insurance_coverage_usd: float = Field(default=1000000, ge=0)
    years_in_business: int = Field(default=5, ge=0)
    employee_count: int = Field(default=20, ge=1)
    safety_program_certified: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContractorOutput(BaseModel):
    contractor_id: str
    contractor_name: str
    overall_rating: PerformanceRating
    overall_score: float
    cost_performance_score: float
    schedule_performance_score: float
    quality_score: float
    safety_score: float
    total_projects: int
    total_contract_value_usd: float
    average_cost_variance_pct: float
    average_schedule_variance_pct: float
    prequalified: bool
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class ContractorAgent:
    AGENT_ID = "GL-075"
    AGENT_NAME = "CONTRACTOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"ContractorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ContractorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_rating(self, score: float) -> PerformanceRating:
        """Convert numeric score to rating."""
        if score >= 90:
            return PerformanceRating.EXCELLENT
        elif score >= 75:
            return PerformanceRating.GOOD
        elif score >= 60:
            return PerformanceRating.SATISFACTORY
        elif score >= 40:
            return PerformanceRating.NEEDS_IMPROVEMENT
        else:
            return PerformanceRating.UNACCEPTABLE

    def _process(self, inp: ContractorInput) -> ContractorOutput:
        recommendations = []

        if not inp.project_history:
            # No history - base score on qualifications
            cost_score = 70
            schedule_score = 70
            quality_avg = 70
            safety_score = 80 if inp.safety_program_certified else 60
            total_value = 0
            avg_cost_var = 0
            avg_schedule_var = 0
        else:
            # Calculate from history
            cost_variances = []
            schedule_variances = []
            quality_scores = []
            total_incidents = 0
            total_value = 0

            for proj in inp.project_history:
                # Cost variance
                if proj.contract_value_usd > 0:
                    cost_var = (proj.actual_cost_usd - proj.contract_value_usd) / proj.contract_value_usd * 100
                    cost_variances.append(cost_var)

                # Schedule variance
                if proj.scheduled_days > 0:
                    schedule_var = (proj.actual_days - proj.scheduled_days) / proj.scheduled_days * 100
                    schedule_variances.append(schedule_var)

                quality_scores.append(proj.quality_score)
                total_incidents += proj.safety_incidents
                total_value += proj.contract_value_usd

            # Averages
            avg_cost_var = sum(cost_variances) / len(cost_variances) if cost_variances else 0
            avg_schedule_var = sum(schedule_variances) / len(schedule_variances) if schedule_variances else 0
            quality_avg = sum(quality_scores) / len(quality_scores) if quality_scores else 70

            # Score calculations (100 = perfect, penalize overruns)
            cost_score = max(0, 100 - abs(avg_cost_var) * 2)
            schedule_score = max(0, 100 - abs(avg_schedule_var) * 2)

            # Safety score (penalize incidents)
            incidents_per_project = total_incidents / len(inp.project_history)
            safety_score = max(0, 100 - incidents_per_project * 25)

        # Overall score (weighted)
        overall = (cost_score * 0.25 + schedule_score * 0.25 + quality_avg * 0.30 + safety_score * 0.20)

        # Prequalification check
        prequalified = (
            overall >= 60 and
            inp.years_in_business >= 3 and
            inp.insurance_coverage_usd >= 500000 and
            inp.safety_program_certified
        )

        # Recommendations
        if cost_score < 70:
            recommendations.append(f"Cost performance below standard ({avg_cost_var:.1f}% average overrun)")
        if schedule_score < 70:
            recommendations.append(f"Schedule performance needs improvement ({avg_schedule_var:.1f}% average delay)")
        if quality_avg < 75:
            recommendations.append("Quality scores below target - require detailed QA plan")
        if safety_score < 80:
            recommendations.append("Safety record requires attention - verify training program")
        if not inp.safety_program_certified:
            recommendations.append("Safety program certification required for prequalification")
        if inp.years_in_business < 3:
            recommendations.append("Limited experience - consider for smaller projects only")
        if len(inp.certifications) < 2:
            recommendations.append("Limited certifications - verify technical capabilities")
        if not prequalified:
            recommendations.append("Does not meet prequalification criteria")

        rating = self._calculate_rating(overall)

        calc_hash = hashlib.sha256(json.dumps({
            "contractor": inp.contractor_id,
            "overall_score": round(overall, 1),
            "rating": rating.value
        }).encode()).hexdigest()

        return ContractorOutput(
            contractor_id=inp.contractor_id,
            contractor_name=inp.contractor_name,
            overall_rating=rating,
            overall_score=round(overall, 1),
            cost_performance_score=round(cost_score, 1),
            schedule_performance_score=round(schedule_score, 1),
            quality_score=round(quality_avg, 1),
            safety_score=round(safety_score, 1),
            total_projects=len(inp.project_history),
            total_contract_value_usd=round(total_value, 2),
            average_cost_variance_pct=round(avg_cost_var, 1),
            average_schedule_variance_pct=round(avg_schedule_var, 1),
            prequalified=prequalified,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-075", "name": "CONTRACTOR", "version": "1.0.0",
    "summary": "Contractor performance evaluation and management",
    "standards": [{"ref": "ISO 9001"}, {"ref": "OSHA 1910"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
