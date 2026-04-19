"""GL-100: Kaizen Driver Agent (KAIZEN-DRIVER).

Drives continuous improvement using Kaizen methodology.

Standards: Lean Manufacturing, PDCA
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImprovementType(str, Enum):
    ENERGY = "ENERGY"
    QUALITY = "QUALITY"
    SAFETY = "SAFETY"
    PRODUCTIVITY = "PRODUCTIVITY"
    COST = "COST"


class PDCAPhase(str, Enum):
    PLAN = "PLAN"
    DO = "DO"
    CHECK = "CHECK"
    ACT = "ACT"


class KaizenIdea(BaseModel):
    idea_id: str
    description: str
    improvement_type: ImprovementType
    submitter: str = Field(default="Anonymous")
    estimated_savings_usd: float = Field(default=0, ge=0)
    implementation_effort: str = Field(default="MEDIUM")
    current_phase: PDCAPhase = Field(default=PDCAPhase.PLAN)


class KaizenDriverInput(BaseModel):
    program_id: str
    program_name: str = Field(default="Kaizen Program")
    ideas: List[KaizenIdea] = Field(default_factory=list)
    active_events: int = Field(default=0, ge=0)
    completed_events_ytd: int = Field(default=0, ge=0)
    participation_rate_pct: float = Field(default=50, ge=0, le=100)
    target_ideas_per_employee: float = Field(default=2, ge=0)
    employee_count: int = Field(default=100, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdeaPipeline(BaseModel):
    phase: str
    count: int
    total_value_usd: float


class KaizenDriverOutput(BaseModel):
    program_id: str
    total_ideas: int
    ideas_by_phase: List[IdeaPipeline]
    total_potential_savings_usd: float
    realized_savings_usd: float
    ideas_per_employee: float
    participation_gap_pct: float
    top_improvement_areas: List[str]
    quick_wins: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class KaizenDriverAgent:
    AGENT_ID = "GL-100"
    AGENT_NAME = "KAIZEN-DRIVER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"KaizenDriverAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = KaizenDriverInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: KaizenDriverInput) -> KaizenDriverOutput:
        recommendations = []

        # Group by phase
        by_phase = {}
        for idea in inp.ideas:
            phase = idea.current_phase.value
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(idea)

        # Pipeline summary
        pipeline = []
        total_potential = 0
        realized = 0

        for phase in PDCAPhase:
            ideas = by_phase.get(phase.value, [])
            value = sum(i.estimated_savings_usd for i in ideas)
            pipeline.append(IdeaPipeline(
                phase=phase.value,
                count=len(ideas),
                total_value_usd=round(value, 2)
            ))
            total_potential += value
            if phase == PDCAPhase.ACT:
                realized += value

        # Group by improvement type
        by_type = {}
        for idea in inp.ideas:
            it = idea.improvement_type.value
            if it not in by_type:
                by_type[it] = 0
            by_type[it] += 1

        top_areas = sorted(by_type.keys(), key=lambda x: -by_type[x])[:3]

        # Quick wins (low effort, in PLAN phase)
        quick_wins = [
            i.description[:50] for i in inp.ideas
            if i.implementation_effort == "LOW" and i.current_phase == PDCAPhase.PLAN
        ][:5]

        # Ideas per employee
        ideas_per_emp = len(inp.ideas) / inp.employee_count if inp.employee_count > 0 else 0

        # Participation gap
        participation_gap = 100 - inp.participation_rate_pct

        # Recommendations
        if ideas_per_emp < inp.target_ideas_per_employee:
            recommendations.append(f"Ideas/employee {ideas_per_emp:.1f} below target {inp.target_ideas_per_employee}")
        if participation_gap > 30:
            recommendations.append(f"Participation gap {participation_gap:.0f}% - engage more employees")
        if quick_wins:
            recommendations.append(f"{len(quick_wins)} quick wins ready for implementation")

        plan_ideas = len(by_phase.get("PLAN", []))
        if plan_ideas > len(inp.ideas) * 0.5:
            recommendations.append(f"{plan_ideas} ideas stuck in PLAN phase - accelerate execution")

        if inp.completed_events_ytd < 4:
            recommendations.append("Limited Kaizen events YTD - schedule more formal events")

        if total_potential > 100000:
            recommendations.append(f"${total_potential:,.0f} savings potential - ensure resource allocation")

        calc_hash = hashlib.sha256(json.dumps({
            "program": inp.program_id,
            "ideas": len(inp.ideas),
            "potential": round(total_potential, 2)
        }).encode()).hexdigest()

        return KaizenDriverOutput(
            program_id=inp.program_id,
            total_ideas=len(inp.ideas),
            ideas_by_phase=pipeline,
            total_potential_savings_usd=round(total_potential, 2),
            realized_savings_usd=round(realized, 2),
            ideas_per_employee=round(ideas_per_emp, 2),
            participation_gap_pct=round(participation_gap, 1),
            top_improvement_areas=top_areas,
            quick_wins=quick_wins,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-100", "name": "KAIZEN-DRIVER", "version": "1.0.0",
    "summary": "Continuous improvement using Kaizen methodology",
    "standards": [{"ref": "Lean Manufacturing"}, {"ref": "PDCA"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
