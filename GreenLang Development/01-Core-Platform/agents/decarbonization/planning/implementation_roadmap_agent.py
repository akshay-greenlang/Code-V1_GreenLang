# -*- coding: utf-8 -*-
"""
GL-DECARB-X-007: Implementation Roadmap Agent
==============================================

Creates phased implementation plans for decarbonization initiatives
with detailed timelines, milestones, and resource requirements.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class PhaseType(str, Enum):
    PLANNING = "planning"
    PILOT = "pilot"
    SCALE_UP = "scale_up"
    FULL_DEPLOYMENT = "full_deployment"
    OPTIMIZATION = "optimization"


class MilestoneStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    AT_RISK = "at_risk"


class RoadmapMilestone(BaseModel):
    milestone_id: str = Field(...)
    name: str = Field(...)
    phase: PhaseType = Field(...)
    start_date: datetime = Field(...)
    end_date: datetime = Field(...)
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)
    deliverables: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    resources_required: Dict[str, Any] = Field(default_factory=dict)
    budget_usd: float = Field(default=0, ge=0)
    emission_reduction_tco2e: float = Field(default=0, ge=0)


class RoadmapPhase(BaseModel):
    phase_id: str = Field(...)
    phase_type: PhaseType = Field(...)
    name: str = Field(...)
    start_year: int = Field(...)
    end_year: int = Field(...)
    objectives: List[str] = Field(default_factory=list)
    milestones: List[RoadmapMilestone] = Field(default_factory=list)
    total_budget_usd: float = Field(default=0, ge=0)
    expected_reduction_tco2e: float = Field(default=0, ge=0)


class ImplementationRoadmap(BaseModel):
    roadmap_id: str = Field(...)
    name: str = Field(...)
    description: str = Field(default="")
    start_year: int = Field(...)
    end_year: int = Field(...)
    target_reduction_percent: float = Field(..., ge=0, le=100)
    phases: List[RoadmapPhase] = Field(default_factory=list)
    total_budget_usd: float = Field(default=0, ge=0)
    total_reduction_tco2e: float = Field(default=0, ge=0)
    critical_path: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=DeterministicClock.now)


class ImplementationRoadmapInput(BaseModel):
    operation: str = Field(default="generate")
    scenario_data: Optional[Dict[str, Any]] = Field(None)
    target_year: int = Field(default=2030)
    target_reduction_percent: float = Field(default=50, ge=0, le=100)
    abatement_options: List[Dict[str, Any]] = Field(default_factory=list)


class ImplementationRoadmapOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    roadmap: Optional[ImplementationRoadmap] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class ImplementationRoadmapAgent(DeterministicAgent):
    """
    GL-DECARB-X-007: Implementation Roadmap Agent

    Creates detailed implementation roadmaps with phases, milestones,
    and resource requirements.
    """

    AGENT_ID = "GL-DECARB-X-007"
    AGENT_NAME = "Implementation Roadmap Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="ImplementationRoadmapAgent",
        category=AgentCategory.CRITICAL,
        description="Creates phased implementation roadmaps"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Creates implementation roadmaps", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            roadmap_input = ImplementationRoadmapInput(**inputs)
            calculation_trace.append(f"Operation: {roadmap_input.operation}")

            if roadmap_input.operation == "generate":
                result = self._generate_roadmap(roadmap_input, calculation_trace)
            else:
                raise ValueError(f"Unknown operation: {roadmap_input.operation}")

            result["processing_time_ms"] = (time.time() - start_time) * 1000

            self._capture_audit_entry(
                operation=roadmap_input.operation,
                inputs=inputs,
                outputs={"success": result["success"]},
                calculation_trace=calculation_trace
            )
            return result

        except Exception as e:
            self.logger.error(f"Roadmap generation failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }

    def _generate_roadmap(
        self,
        roadmap_input: ImplementationRoadmapInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Generate implementation roadmap from scenario data."""
        current_year = DeterministicClock.now().year
        target_year = roadmap_input.target_year
        years = target_year - current_year

        # Create phases
        phases = []

        # Phase 1: Planning (Year 1)
        planning_phase = RoadmapPhase(
            phase_id="phase_1",
            phase_type=PhaseType.PLANNING,
            name="Planning & Assessment",
            start_year=current_year,
            end_year=current_year + 1,
            objectives=[
                "Complete baseline emissions inventory",
                "Finalize abatement option selection",
                "Secure initial funding",
                "Establish governance structure"
            ],
            total_budget_usd=100000,
            expected_reduction_tco2e=0
        )
        phases.append(planning_phase)

        # Phase 2: Pilot (Years 2-3)
        pilot_phase = RoadmapPhase(
            phase_id="phase_2",
            phase_type=PhaseType.PILOT,
            name="Pilot Projects",
            start_year=current_year + 1,
            end_year=current_year + 2,
            objectives=[
                "Implement quick-win efficiency projects",
                "Pilot high-TRL technologies",
                "Build internal capabilities",
                "Validate cost and savings assumptions"
            ],
            total_budget_usd=500000,
            expected_reduction_tco2e=5000
        )
        phases.append(pilot_phase)

        # Phase 3: Scale-up (Years 3-5)
        scaleup_phase = RoadmapPhase(
            phase_id="phase_3",
            phase_type=PhaseType.SCALE_UP,
            name="Scale-Up Deployment",
            start_year=current_year + 2,
            end_year=current_year + 4,
            objectives=[
                "Scale successful pilots",
                "Deploy renewable energy systems",
                "Electrify suitable processes",
                "Engage key suppliers"
            ],
            total_budget_usd=2000000,
            expected_reduction_tco2e=20000
        )
        phases.append(scaleup_phase)

        # Phase 4: Full Deployment
        if years > 4:
            full_phase = RoadmapPhase(
                phase_id="phase_4",
                phase_type=PhaseType.FULL_DEPLOYMENT,
                name="Full Deployment",
                start_year=current_year + 4,
                end_year=target_year,
                objectives=[
                    "Complete major infrastructure changes",
                    "Achieve target emission reductions",
                    "Implement advanced technologies",
                    "Realize net-zero operations"
                ],
                total_budget_usd=5000000,
                expected_reduction_tco2e=50000
            )
            phases.append(full_phase)

        # Calculate totals
        total_budget = sum(p.total_budget_usd for p in phases)
        total_reduction = sum(p.expected_reduction_tco2e for p in phases)

        roadmap = ImplementationRoadmap(
            roadmap_id=deterministic_id({"target_year": target_year}, "roadmap_"),
            name=f"Decarbonization Roadmap to {target_year}",
            description=f"Phased implementation plan for {roadmap_input.target_reduction_percent}% reduction by {target_year}",
            start_year=current_year,
            end_year=target_year,
            target_reduction_percent=roadmap_input.target_reduction_percent,
            phases=phases,
            total_budget_usd=total_budget,
            total_reduction_tco2e=total_reduction,
            critical_path=["Baseline assessment", "Funding approval", "Technology selection", "Deployment"],
            risks=["Budget overruns", "Technology delays", "Supply chain constraints", "Regulatory changes"]
        )

        roadmap.provenance_hash = content_hash(roadmap.model_dump(exclude={"provenance_hash"}))
        calculation_trace.append(f"Generated roadmap with {len(phases)} phases")

        return {
            "operation": "generate",
            "success": True,
            "roadmap": roadmap.model_dump(),
            "timestamp": DeterministicClock.now().isoformat()
        }
