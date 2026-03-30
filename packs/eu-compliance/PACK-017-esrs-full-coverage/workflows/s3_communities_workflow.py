# -*- coding: utf-8 -*-
"""
S3 Affected Communities Workflow
================================

5-phase workflow for ESRS S3 Affected Communities disclosure covering
community mapping, engagement assessment with FPIC, material impact actions,
target tracking, and report generation with full provenance tracking.

Phases:
    1. CommunityMapping       -- Identify affected communities
    2. EngagementAssessment   -- S3-2 engagement processes, FPIC
    3. ImpactAssessment       -- S3-4 material impact actions
    4. TargetTracking         -- S3-5 target progress
    5. ReportGeneration       -- Compile S3 disclosure

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the S3 affected communities workflow."""
    COMMUNITY_MAPPING = "community_mapping"
    ENGAGEMENT_ASSESSMENT = "engagement_assessment"
    IMPACT_ASSESSMENT = "impact_assessment"
    TARGET_TRACKING = "target_tracking"
    REPORT_GENERATION = "report_generation"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class CommunityType(str, Enum):
    """Affected community type."""
    LOCAL_COMMUNITY = "local_community"
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    RURAL_COMMUNITY = "rural_community"
    URBAN_COMMUNITY = "urban_community"
    DISPLACED_COMMUNITY = "displaced_community"

class ImpactType(str, Enum):
    """Community impact type classification."""
    ENVIRONMENTAL = "environmental"
    HEALTH = "health"
    ECONOMIC = "economic"
    CULTURAL = "cultural"
    LAND_RIGHTS = "land_rights"
    DISPLACEMENT = "displacement"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CommunityRecord(BaseModel):
    """Affected community record."""
    community_id: str = Field(default_factory=lambda: f"com-{_new_uuid()[:8]}")
    community_name: str = Field(default="")
    community_type: CommunityType = Field(default=CommunityType.LOCAL_COMMUNITY)
    location: str = Field(default="")
    population_estimate: int = Field(default=0, ge=0)
    is_indigenous: bool = Field(default=False)
    fpic_required: bool = Field(default=False)
    fpic_obtained: bool = Field(default=False)
    impacts: List[ImpactType] = Field(default_factory=list)
    engagement_active: bool = Field(default=False)

class EngagementProcess(BaseModel):
    """Community engagement process record."""
    process_id: str = Field(default_factory=lambda: f"cep-{_new_uuid()[:8]}")
    process_name: str = Field(default="")
    communities_covered: int = Field(default=0, ge=0)
    includes_fpic: bool = Field(default=False)
    grievance_mechanism: bool = Field(default=False)
    frequency: str = Field(default="annual")
    outcomes_documented: bool = Field(default=False)

class CommunityImpactAction(BaseModel):
    """Action taken to address community impacts."""
    action_id: str = Field(default_factory=lambda: f"cia-{_new_uuid()[:8]}")
    action_description: str = Field(default="")
    impact_type: ImpactType = Field(default=ImpactType.ENVIRONMENTAL)
    communities_benefited: int = Field(default=0, ge=0)
    investment_eur: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="planned", description="planned, in_progress, completed")

class CommunityTarget(BaseModel):
    """Community-related target."""
    target_id: str = Field(default_factory=lambda: f"ct-{_new_uuid()[:8]}")
    target_description: str = Field(default="")
    metric: str = Field(default="")
    baseline_year: int = Field(default=2019)
    baseline_value: float = Field(default=0.0)
    target_year: int = Field(default=2030)
    target_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)

class S3CommunitiesInput(BaseModel):
    """Input data model for S3CommunitiesWorkflow."""
    communities: List[CommunityRecord] = Field(
        default_factory=list, description="Affected community records"
    )
    engagement_processes: List[EngagementProcess] = Field(
        default_factory=list, description="S3-2 engagement process records"
    )
    impact_actions: List[CommunityImpactAction] = Field(
        default_factory=list, description="S3-4 impact actions"
    )
    targets: List[CommunityTarget] = Field(
        default_factory=list, description="S3-5 targets"
    )
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="S3-1 policies"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class S3CommunitiesWorkflowResult(BaseModel):
    """Complete result from S3 affected communities workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="s3_affected_communities")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    communities_mapped: int = Field(default=0)
    indigenous_communities: int = Field(default=0)
    fpic_required_count: int = Field(default=0)
    fpic_obtained_count: int = Field(default=0)
    engagement_coverage_pct: float = Field(default=0.0)
    impact_actions_count: int = Field(default=0)
    total_investment_eur: float = Field(default=0.0)
    target_progress: Dict[str, float] = Field(default_factory=dict)
    quality_issues: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class S3CommunitiesWorkflow:
    """
    5-phase S3 Affected Communities workflow.

    Implements end-to-end community impact assessment covering community
    mapping and identification, engagement process evaluation including FPIC,
    material impact action assessment, target tracking, and S3 disclosure
    report generation per ESRS S3 requirements.

    Zero-hallucination: all coverage and investment calculations use
    deterministic arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = S3CommunitiesWorkflow()
        >>> inp = S3CommunitiesInput(communities=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.communities_mapped >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize S3CommunitiesWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._communities: List[CommunityRecord] = []
        self._target_progress: Dict[str, float] = {}
        self._quality_issues: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.COMMUNITY_MAPPING.value, "description": "Identify affected communities"},
            {"name": WorkflowPhase.ENGAGEMENT_ASSESSMENT.value, "description": "S3-2 engagement processes, FPIC"},
            {"name": WorkflowPhase.IMPACT_ASSESSMENT.value, "description": "S3-4 material impact actions"},
            {"name": WorkflowPhase.TARGET_TRACKING.value, "description": "S3-5 target progress"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Compile S3 disclosure"},
        ]

    def validate_inputs(self, input_data: S3CommunitiesInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.communities:
            issues.append("No community data provided")
        fpic_needed = [c for c in input_data.communities if c.fpic_required and not c.fpic_obtained]
        if fpic_needed:
            issues.append(f"{len(fpic_needed)} communities require FPIC but have not obtained it")
        return issues

    async def execute(
        self,
        input_data: Optional[S3CommunitiesInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> S3CommunitiesWorkflowResult:
        """
        Execute the 5-phase S3 affected communities workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            S3CommunitiesWorkflowResult with community mapping and engagement data.
        """
        if input_data is None:
            input_data = S3CommunitiesInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting S3 communities workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_community_mapping(input_data))
            phases_done += 1
            phase_results.append(await self._phase_engagement_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_impact_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_target_tracking(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("S3 communities workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        self._communities = list(input_data.communities)
        indigenous = sum(1 for c in self._communities if c.is_indigenous)
        fpic_req = sum(1 for c in self._communities if c.fpic_required)
        fpic_obt = sum(1 for c in self._communities if c.fpic_obtained)
        engaged = sum(1 for c in self._communities if c.engagement_active)
        eng_pct = round((engaged / len(self._communities) * 100) if self._communities else 0.0, 2)
        total_invest = sum(a.investment_eur for a in input_data.impact_actions)

        result = S3CommunitiesWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            communities_mapped=len(self._communities),
            indigenous_communities=indigenous,
            fpic_required_count=fpic_req,
            fpic_obtained_count=fpic_obt,
            engagement_coverage_pct=eng_pct,
            impact_actions_count=len(input_data.impact_actions),
            total_investment_eur=round(total_invest, 2),
            target_progress=self._target_progress,
            quality_issues=self._quality_issues,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "S3 communities %s completed in %.2fs: %d communities, %.1f%% engaged",
            self.workflow_id, elapsed, len(self._communities), eng_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Community Mapping
    # -------------------------------------------------------------------------

    async def _phase_community_mapping(self, input_data: S3CommunitiesInput) -> PhaseResult:
        """Identify and map affected communities."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        communities = input_data.communities
        type_dist: Dict[str, int] = {}
        for c in communities:
            type_dist[c.community_type.value] = type_dist.get(c.community_type.value, 0) + 1

        outputs["communities_count"] = len(communities)
        outputs["type_distribution"] = type_dist
        outputs["indigenous_count"] = sum(1 for c in communities if c.is_indigenous)
        outputs["total_population_estimate"] = sum(c.population_estimate for c in communities)
        outputs["fpic_required"] = sum(1 for c in communities if c.fpic_required)

        if not communities:
            warnings.append("No affected communities identified")
        indigenous_no_fpic = [c for c in communities if c.is_indigenous and c.fpic_required and not c.fpic_obtained]
        if indigenous_no_fpic:
            warnings.append(f"{len(indigenous_no_fpic)} indigenous communities without FPIC")
            self._quality_issues.append("CRITICAL: FPIC not obtained for indigenous communities")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 CommunityMapping: %d communities", len(communities))
        return PhaseResult(
            phase_name=WorkflowPhase.COMMUNITY_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Engagement Assessment (S3-2)
    # -------------------------------------------------------------------------

    async def _phase_engagement_assessment(self, input_data: S3CommunitiesInput) -> PhaseResult:
        """Assess engagement processes and FPIC compliance."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        processes = input_data.engagement_processes
        total_communities = len(input_data.communities)
        covered = sum(p.communities_covered for p in processes)
        coverage_pct = round((covered / total_communities * 100) if total_communities > 0 else 0.0, 2)

        outputs["processes_count"] = len(processes)
        outputs["communities_coverage_pct"] = coverage_pct
        outputs["includes_fpic"] = sum(1 for p in processes if p.includes_fpic)
        outputs["has_grievance_mechanism"] = sum(1 for p in processes if p.grievance_mechanism)
        outputs["outcomes_documented"] = sum(1 for p in processes if p.outcomes_documented)

        if not processes:
            warnings.append("No engagement processes documented (S3-2)")
        if not any(p.grievance_mechanism for p in processes):
            warnings.append("No grievance mechanism in place")
        if coverage_pct < 50.0 and total_communities > 0:
            warnings.append(f"Low community engagement coverage: {coverage_pct}%")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 EngagementAssessment: %.1f%% coverage", coverage_pct)
        return PhaseResult(
            phase_name=WorkflowPhase.ENGAGEMENT_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Impact Assessment (S3-4)
    # -------------------------------------------------------------------------

    async def _phase_impact_assessment(self, input_data: S3CommunitiesInput) -> PhaseResult:
        """Assess material impact actions for affected communities."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        actions = input_data.impact_actions
        by_type: Dict[str, int] = {}
        for a in actions:
            by_type[a.impact_type.value] = by_type.get(a.impact_type.value, 0) + 1

        by_status: Dict[str, int] = {}
        for a in actions:
            by_status[a.status] = by_status.get(a.status, 0) + 1

        total_invest = sum(a.investment_eur for a in actions)

        outputs["actions_count"] = len(actions)
        outputs["by_impact_type"] = by_type
        outputs["by_status"] = by_status
        outputs["total_investment_eur"] = round(total_invest, 2)
        outputs["communities_benefited"] = sum(a.communities_benefited for a in actions)

        if not actions:
            warnings.append("No impact actions defined (S3-4)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ImpactAssessment: %d actions, EUR %.0f invested", len(actions), total_invest)
        return PhaseResult(
            phase_name=WorkflowPhase.IMPACT_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Tracking (S3-5)
    # -------------------------------------------------------------------------

    async def _phase_target_tracking(self, input_data: S3CommunitiesInput) -> PhaseResult:
        """Track progress against community-related targets."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._target_progress = {}

        for target in input_data.targets:
            if target.target_value != target.baseline_value:
                change_needed = target.target_value - target.baseline_value
                change_achieved = target.current_value - target.baseline_value
                progress_pct = round(
                    (change_achieved / change_needed * 100) if change_needed != 0 else 100.0, 2
                )
            else:
                progress_pct = 100.0 if target.current_value >= target.target_value else 0.0
            self._target_progress[target.target_id] = progress_pct

        outputs["targets_assessed"] = len(self._target_progress)
        outputs["target_progress"] = self._target_progress
        outputs["on_track_count"] = sum(1 for p in self._target_progress.values() if p >= 50.0)

        if not input_data.targets:
            warnings.append("No community targets defined (S3-5)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TargetTracking: %d targets", len(self._target_progress))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(self, input_data: S3CommunitiesInput) -> PhaseResult:
        """Compile S3 affected communities disclosure."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        communities = input_data.communities
        outputs["s3_disclosure"] = {
            "s3_1_policies_count": len(input_data.policies),
            "s3_2_engagement_processes": len(input_data.engagement_processes),
            "s3_3_communities_mapped": len(communities),
            "s3_3_indigenous_count": sum(1 for c in communities if c.is_indigenous),
            "s3_4_impact_actions": len(input_data.impact_actions),
            "s3_5_targets": self._target_progress,
            "reporting_year": input_data.reporting_year,
            "entity_name": input_data.entity_name,
        }
        outputs["report_ready"] = True

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 ReportGeneration: S3 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: S3CommunitiesWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
