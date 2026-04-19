"""GL-100: Kaizen Agent (KAIZEN).

Drives continuous improvement using PDCA cycles.

Standards: ISO 9001, Lean Six Sigma
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImprovementCategory(str, Enum):
    ENERGY = "ENERGY"
    EMISSIONS = "EMISSIONS"
    EFFICIENCY = "EFFICIENCY"
    COST = "COST"
    SAFETY = "SAFETY"
    QUALITY = "QUALITY"


class PDCAPhase(str, Enum):
    PLAN = "PLAN"
    DO = "DO"
    CHECK = "CHECK"
    ACT = "ACT"


class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ImprovementItem(BaseModel):
    """Improvement opportunity."""
    item_id: str
    title: str
    category: ImprovementCategory
    description: str
    current_value: float
    target_value: float
    unit: str
    estimated_savings_usd: float = Field(default=0, ge=0)
    effort_hours: float = Field(default=0, ge=0)
    owner: Optional[str] = None


class KaizenInput(BaseModel):
    """Input for Kaizen analysis."""
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(default="Facility")
    improvement_items: List[ImprovementItem] = Field(default_factory=list)
    current_phase: PDCAPhase = Field(default=PDCAPhase.PLAN)
    cycle_number: int = Field(default=1, ge=1)
    baseline_metrics: Dict[str, float] = Field(default_factory=dict)
    target_metrics: Dict[str, float] = Field(default_factory=dict)
    completed_actions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PrioritizedItem(BaseModel):
    """Prioritized improvement with score."""
    item_id: str
    title: str
    category: ImprovementCategory
    priority: Priority
    impact_score: float
    effort_score: float
    roi_score: float
    overall_score: float
    recommended_action: str


class PDCARecommendation(BaseModel):
    """PDCA phase recommendation."""
    phase: PDCAPhase
    actions: List[str]
    success_criteria: List[str]
    estimated_duration_days: int


class KaizenOutput(BaseModel):
    """Output from Kaizen analysis."""
    facility_id: str
    facility_name: str
    current_phase: PDCAPhase
    cycle_number: int

    # Prioritization
    prioritized_items: List[PrioritizedItem]
    quick_wins: List[str]
    strategic_initiatives: List[str]

    # Impact analysis
    total_potential_savings_usd: float
    total_effort_hours: float
    average_roi: float
    improvement_velocity: float

    # PDCA guidance
    pdca_recommendations: List[PDCARecommendation]
    next_phase: PDCAPhase

    # Metrics
    items_by_category: Dict[str, int]
    completion_rate_pct: float

    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class KaizenAgent:
    """GL-100: Kaizen Agent - Continuous improvement driver."""

    AGENT_ID = "GL-100"
    AGENT_NAME = "KAIZEN"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"KaizenAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = KaizenInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_scores(self, item: ImprovementItem) -> tuple:
        """Calculate impact, effort, and ROI scores."""
        # Impact score (0-100)
        if item.current_value > 0:
            improvement_pct = abs(item.target_value - item.current_value) / item.current_value * 100
        else:
            improvement_pct = 50
        impact_score = min(100, improvement_pct + item.estimated_savings_usd / 1000)

        # Effort score (inverse - lower effort = higher score)
        if item.effort_hours <= 8:
            effort_score = 100
        elif item.effort_hours <= 40:
            effort_score = 80
        elif item.effort_hours <= 160:
            effort_score = 50
        else:
            effort_score = 20

        # ROI score
        if item.effort_hours > 0:
            hourly_value = item.estimated_savings_usd / item.effort_hours
            roi_score = min(100, hourly_value * 2)
        else:
            roi_score = 50

        return round(impact_score, 1), round(effort_score, 1), round(roi_score, 1)

    def _determine_priority(self, impact: float, effort: float, roi: float) -> Priority:
        """Determine priority based on scores."""
        overall = (impact * 0.4 + effort * 0.3 + roi * 0.3)
        if overall >= 70:
            return Priority.HIGH
        elif overall >= 40:
            return Priority.MEDIUM
        return Priority.LOW

    def _get_next_phase(self, current: PDCAPhase) -> PDCAPhase:
        """Get next PDCA phase."""
        sequence = [PDCAPhase.PLAN, PDCAPhase.DO, PDCAPhase.CHECK, PDCAPhase.ACT]
        idx = sequence.index(current)
        return sequence[(idx + 1) % 4]

    def _generate_pdca_recommendations(self, phase: PDCAPhase, items: List[PrioritizedItem]) -> PDCARecommendation:
        """Generate recommendations for PDCA phase."""
        if phase == PDCAPhase.PLAN:
            return PDCARecommendation(
                phase=PDCAPhase.PLAN,
                actions=[
                    "Define improvement objectives and scope",
                    "Analyze root causes of current performance gaps",
                    "Develop action plans with owners and timelines",
                    "Establish baseline measurements"
                ],
                success_criteria=[
                    "All improvements have documented plans",
                    "Resources allocated and owners assigned",
                    "Success metrics defined"
                ],
                estimated_duration_days=14
            )
        elif phase == PDCAPhase.DO:
            return PDCARecommendation(
                phase=PDCAPhase.DO,
                actions=[
                    "Execute improvement actions per plan",
                    "Document changes and observations",
                    "Collect data during implementation",
                    "Address obstacles and risks"
                ],
                success_criteria=[
                    "All planned actions completed",
                    "Implementation documented",
                    "Data collected for analysis"
                ],
                estimated_duration_days=30
            )
        elif phase == PDCAPhase.CHECK:
            return PDCARecommendation(
                phase=PDCAPhase.CHECK,
                actions=[
                    "Analyze results vs targets",
                    "Identify gaps and root causes",
                    "Document lessons learned",
                    "Validate measurement accuracy"
                ],
                success_criteria=[
                    "Performance vs target documented",
                    "Lessons learned captured",
                    "Recommendations prepared"
                ],
                estimated_duration_days=7
            )
        else:  # ACT
            return PDCARecommendation(
                phase=PDCAPhase.ACT,
                actions=[
                    "Standardize successful improvements",
                    "Update procedures and training",
                    "Plan next improvement cycle",
                    "Celebrate wins and share learnings"
                ],
                success_criteria=[
                    "Standards updated",
                    "Training completed",
                    "Next cycle initiated"
                ],
                estimated_duration_days=7
            )

    def _process(self, inp: KaizenInput) -> KaizenOutput:
        recommendations = []
        prioritized = []
        quick_wins = []
        strategic = []

        # Prioritize items
        for item in inp.improvement_items:
            impact, effort, roi = self._calculate_scores(item)
            overall = round(impact * 0.4 + effort * 0.3 + roi * 0.3, 1)
            priority = self._determine_priority(impact, effort, roi)

            action = "Implement immediately" if priority == Priority.HIGH else \
                     "Schedule for next sprint" if priority == Priority.MEDIUM else \
                     "Add to backlog"

            prioritized.append(PrioritizedItem(
                item_id=item.item_id,
                title=item.title,
                category=item.category,
                priority=priority,
                impact_score=impact,
                effort_score=effort,
                roi_score=roi,
                overall_score=overall,
                recommended_action=action
            ))

            # Classify
            if effort >= 70 and impact >= 50:  # High effort score = low effort
                quick_wins.append(item.item_id)
            if impact >= 70 and effort < 50:  # Low effort score = high effort, high impact
                strategic.append(item.item_id)

        # Sort by score
        prioritized.sort(key=lambda x: x.overall_score, reverse=True)

        # Totals
        total_savings = sum(i.estimated_savings_usd for i in inp.improvement_items)
        total_effort = sum(i.effort_hours for i in inp.improvement_items)
        avg_roi = total_savings / total_effort if total_effort > 0 else 0

        # Category counts
        by_category = {}
        for item in inp.improvement_items:
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Completion rate
        total_items = len(inp.improvement_items)
        completed = len(inp.completed_actions)
        completion_rate = (completed / total_items * 100) if total_items > 0 else 0

        # Velocity (items per cycle)
        velocity = completed / inp.cycle_number if inp.cycle_number > 0 else 0

        # PDCA recommendations
        pdca_recs = [
            self._generate_pdca_recommendations(inp.current_phase, prioritized),
            self._generate_pdca_recommendations(self._get_next_phase(inp.current_phase), prioritized)
        ]

        # Generate recommendations
        if len(quick_wins) > 0:
            recommendations.append(f"{len(quick_wins)} quick wins identified - implement within 1 week")
        if len(strategic) > 0:
            recommendations.append(f"{len(strategic)} strategic initiatives require dedicated project resources")

        high_priority = [p for p in prioritized if p.priority == Priority.HIGH]
        if high_priority:
            recommendations.append(f"Focus on {len(high_priority)} high-priority items first")

        if completion_rate < 50:
            recommendations.append("Completion rate is low - review resource allocation")
        if velocity < 2:
            recommendations.append("Consider smaller improvement cycles to increase velocity")

        # Energy/emissions focus
        energy_items = [i for i in inp.improvement_items if i.category in [ImprovementCategory.ENERGY, ImprovementCategory.EMISSIONS]]
        if energy_items:
            energy_savings = sum(i.estimated_savings_usd for i in energy_items)
            recommendations.append(f"Energy/emissions improvements: ${energy_savings:,.0f} potential savings")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "cycle": inp.cycle_number,
            "items": len(inp.improvement_items),
            "savings": round(total_savings, 2)
        }).encode()).hexdigest()

        return KaizenOutput(
            facility_id=inp.facility_id,
            facility_name=inp.facility_name,
            current_phase=inp.current_phase,
            cycle_number=inp.cycle_number,
            prioritized_items=prioritized,
            quick_wins=quick_wins,
            strategic_initiatives=strategic,
            total_potential_savings_usd=round(total_savings, 2),
            total_effort_hours=round(total_effort, 1),
            average_roi=round(avg_roi, 2),
            improvement_velocity=round(velocity, 2),
            pdca_recommendations=pdca_recs,
            next_phase=self._get_next_phase(inp.current_phase),
            items_by_category=by_category,
            completion_rate_pct=round(completion_rate, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Continuous Improvement",
            "type": "Management",
            "standards": ["ISO 9001", "Lean Six Sigma"]
        }


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-100",
    "name": "KAIZEN",
    "version": "1.0.0",
    "summary": "Continuous improvement driver using PDCA cycles",
    "tags": ["kaizen", "pdca", "continuous-improvement", "lean", "six-sigma"],
    "standards": [
        {"ref": "ISO 9001", "description": "Quality Management Systems"},
        {"ref": "Lean Six Sigma", "description": "Process Improvement Methodology"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
