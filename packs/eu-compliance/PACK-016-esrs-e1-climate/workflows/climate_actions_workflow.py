# -*- coding: utf-8 -*-
"""
Climate Actions Workflow
==============================

5-phase workflow for climate actions and resources tracking per ESRS E1-3.
Implements policy review, action registration, resource allocation,
taxonomy alignment check, and report generation.

Phases:
    1. PolicyReview           -- Review climate-related policies
    2. ActionRegistration     -- Register mitigation and adaptation actions
    3. ResourceAllocation     -- Track CAPEX/OPEX resource allocation
    4. TaxonomyCheck          -- Check EU Taxonomy alignment of actions
    5. ReportGeneration       -- Produce E1-3 disclosure data

Author: GreenLang Team
Version: 16.0.0
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
    """Phases of the climate actions workflow."""
    POLICY_REVIEW = "policy_review"
    ACTION_REGISTRATION = "action_registration"
    RESOURCE_ALLOCATION = "resource_allocation"
    TAXONOMY_CHECK = "taxonomy_check"
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

class ActionType(str, Enum):
    """Climate action type."""
    MITIGATION = "mitigation"
    ADAPTATION = "adaptation"
    MITIGATION_AND_ADAPTATION = "mitigation_and_adaptation"

class ActionPhase(str, Enum):
    """Action implementation phase."""
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    OPERATIONAL = "operational"
    COMPLETED = "completed"
    DEFERRED = "deferred"

class ResourceType(str, Enum):
    """Resource allocation type."""
    CAPEX = "capex"
    OPEX = "opex"
    R_AND_D = "r_and_d"
    HUMAN_RESOURCES = "human_resources"

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

class ClimatePolicy(BaseModel):
    """A climate-related policy."""
    policy_id: str = Field(default_factory=lambda: f"cp-{_new_uuid()[:8]}")
    name: str = Field(..., description="Policy name")
    scope: str = Field(default="organization", description="own_operations, upstream, downstream")
    policy_type: str = Field(default="mitigation", description="mitigation, adaptation, both")
    approved_date: str = Field(default="")
    review_date: str = Field(default="")
    description: str = Field(default="")
    is_active: bool = Field(default=True)

class ClimateAction(BaseModel):
    """A climate action with tracking details."""
    action_id: str = Field(default_factory=lambda: f"ca-{_new_uuid()[:8]}")
    name: str = Field(..., description="Action name")
    description: str = Field(default="")
    action_type: ActionType = Field(default=ActionType.MITIGATION)
    action_phase: ActionPhase = Field(default=ActionPhase.PLANNING)
    expected_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    actual_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    start_year: int = Field(default=2025)
    end_year: int = Field(default=2030)
    responsible_unit: str = Field(default="")
    is_taxonomy_aligned: bool = Field(default=False)
    taxonomy_activity_code: str = Field(default="")
    co_benefits: List[str] = Field(default_factory=list)

class ResourceAllocation(BaseModel):
    """Resource allocation for a climate action."""
    allocation_id: str = Field(default_factory=lambda: f"ra-{_new_uuid()[:8]}")
    action_id: str = Field(default="", description="Associated action ID")
    resource_type: ResourceType = Field(default=ResourceType.CAPEX)
    amount_eur: float = Field(default=0.0, ge=0.0)
    year: int = Field(default=2025)
    description: str = Field(default="")
    is_committed: bool = Field(default=False)

class ClimateActionsInput(BaseModel):
    """Input data model for ClimateActionsWorkflow."""
    policies: List[ClimatePolicy] = Field(default_factory=list)
    actions: List[ClimateAction] = Field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class ClimateActionsResult(BaseModel):
    """Complete result from climate actions workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="climate_actions")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    policies: List[ClimatePolicy] = Field(default_factory=list)
    actions: List[ClimateAction] = Field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = Field(default_factory=list)
    total_actions: int = Field(default=0)
    mitigation_actions: int = Field(default=0)
    adaptation_actions: int = Field(default=0)
    total_expected_reduction_tco2e: float = Field(default=0.0)
    total_capex_eur: float = Field(default=0.0)
    total_opex_eur: float = Field(default=0.0)
    taxonomy_aligned_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ClimateActionsWorkflow:
    """
    5-phase climate actions and resources tracking workflow for ESRS E1-3.

    Implements policy review, action registration with expected reductions,
    resource allocation tracking (CAPEX/OPEX), EU Taxonomy alignment
    checking, and disclosure-ready output generation.

    Zero-hallucination: all aggregations use deterministic arithmetic.

    Example:
        >>> wf = ClimateActionsWorkflow()
        >>> inp = ClimateActionsInput(actions=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_actions >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ClimateActionsWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._policies: List[ClimatePolicy] = []
        self._actions: List[ClimateAction] = []
        self._allocations: List[ResourceAllocation] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review climate-related policies"},
            {"name": WorkflowPhase.ACTION_REGISTRATION.value, "description": "Register mitigation/adaptation actions"},
            {"name": WorkflowPhase.RESOURCE_ALLOCATION.value, "description": "Track CAPEX/OPEX allocation"},
            {"name": WorkflowPhase.TAXONOMY_CHECK.value, "description": "Check EU Taxonomy alignment"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-3 disclosure data"},
        ]

    def validate_inputs(self, input_data: ClimateActionsInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.actions:
            issues.append("No climate actions provided")
        for a in input_data.actions:
            if a.end_year < a.start_year:
                issues.append(f"Action {a.action_id}: end year before start year")
        return issues

    async def execute(
        self,
        input_data: Optional[ClimateActionsInput] = None,
        actions: Optional[List[ClimateAction]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ClimateActionsResult:
        """
        Execute the 5-phase climate actions workflow.

        Args:
            input_data: Full input model (preferred).
            actions: Climate actions (fallback).
            config: Configuration overrides.

        Returns:
            ClimateActionsResult with policies, actions, resources, and taxonomy status.
        """
        if input_data is None:
            input_data = ClimateActionsInput(
                actions=actions or [],
                config=config or {},
            )

        started_at = utcnow()
        self.logger.info("Starting climate actions workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_action_registration(input_data))
            phase_results.append(await self._phase_resource_allocation(input_data))
            phase_results.append(await self._phase_taxonomy_check(input_data))
            phase_results.append(await self._phase_report_generation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Climate actions workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        mitigation = sum(1 for a in self._actions if a.action_type in (ActionType.MITIGATION, ActionType.MITIGATION_AND_ADAPTATION))
        adaptation = sum(1 for a in self._actions if a.action_type in (ActionType.ADAPTATION, ActionType.MITIGATION_AND_ADAPTATION))
        total_reduction = sum(a.expected_reduction_tco2e for a in self._actions)
        total_capex = sum(r.amount_eur for r in self._allocations if r.resource_type == ResourceType.CAPEX)
        total_opex = sum(r.amount_eur for r in self._allocations if r.resource_type == ResourceType.OPEX)
        taxonomy_aligned = sum(1 for a in self._actions if a.is_taxonomy_aligned)
        taxonomy_pct = round((taxonomy_aligned / len(self._actions) * 100) if self._actions else 0.0, 1)

        result = ClimateActionsResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            policies=self._policies,
            actions=self._actions,
            resource_allocations=self._allocations,
            total_actions=len(self._actions),
            mitigation_actions=mitigation,
            adaptation_actions=adaptation,
            total_expected_reduction_tco2e=round(total_reduction, 2),
            total_capex_eur=round(total_capex, 2),
            total_opex_eur=round(total_opex, 2),
            taxonomy_aligned_pct=taxonomy_pct,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Climate actions %s completed in %.2fs: %d actions, %.0f tCO2e reduction",
            self.workflow_id, elapsed, len(self._actions), total_reduction,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review
    # -------------------------------------------------------------------------

    async def _phase_policy_review(
        self, input_data: ClimateActionsInput,
    ) -> PhaseResult:
        """Review climate-related policies."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._policies = list(input_data.policies)
        active = sum(1 for p in self._policies if p.is_active)

        outputs["policies_reviewed"] = len(self._policies)
        outputs["active_policies"] = active
        outputs["policy_types"] = dict(self._count_by_field(self._policies, "policy_type"))

        if not self._policies:
            warnings.append("No climate policies provided; E1-2 disclosure may be incomplete")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies, %d active", len(self._policies), active)
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Action Registration
    # -------------------------------------------------------------------------

    async def _phase_action_registration(
        self, input_data: ClimateActionsInput,
    ) -> PhaseResult:
        """Register all climate actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._actions = list(input_data.actions)

        type_counts: Dict[str, int] = {}
        phase_counts: Dict[str, int] = {}
        for a in self._actions:
            type_counts[a.action_type.value] = type_counts.get(a.action_type.value, 0) + 1
            phase_counts[a.action_phase.value] = phase_counts.get(a.action_phase.value, 0) + 1

        outputs["actions_registered"] = len(self._actions)
        outputs["type_distribution"] = type_counts
        outputs["phase_distribution"] = phase_counts
        outputs["total_expected_reduction_tco2e"] = round(
            sum(a.expected_reduction_tco2e for a in self._actions), 2
        )

        if not self._actions:
            warnings.append("No climate actions registered")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 ActionRegistration: %d actions registered", len(self._actions))
        return PhaseResult(
            phase_name=WorkflowPhase.ACTION_REGISTRATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Resource Allocation
    # -------------------------------------------------------------------------

    async def _phase_resource_allocation(
        self, input_data: ClimateActionsInput,
    ) -> PhaseResult:
        """Track resource allocation for climate actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._allocations = list(input_data.resource_allocations)

        type_totals: Dict[str, float] = {}
        for r in self._allocations:
            type_totals[r.resource_type.value] = type_totals.get(r.resource_type.value, 0.0) + r.amount_eur

        committed = sum(r.amount_eur for r in self._allocations if r.is_committed)
        total = sum(r.amount_eur for r in self._allocations)

        outputs["allocations_tracked"] = len(self._allocations)
        outputs["type_totals_eur"] = {k: round(v, 2) for k, v in type_totals.items()}
        outputs["total_allocated_eur"] = round(total, 2)
        outputs["committed_eur"] = round(committed, 2)
        outputs["committed_pct"] = round((committed / total * 100) if total > 0 else 0.0, 1)

        # Check for unlinked allocations
        action_ids = set(a.action_id for a in self._actions)
        unlinked = [r for r in self._allocations if r.action_id and r.action_id not in action_ids]
        if unlinked:
            warnings.append(f"{len(unlinked)} resource allocations reference unknown actions")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ResourceAllocation: %.0f EUR total allocated", total)
        return PhaseResult(
            phase_name=WorkflowPhase.RESOURCE_ALLOCATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Taxonomy Check
    # -------------------------------------------------------------------------

    async def _phase_taxonomy_check(
        self, input_data: ClimateActionsInput,
    ) -> PhaseResult:
        """Check EU Taxonomy alignment of climate actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        aligned_count = sum(1 for a in self._actions if a.is_taxonomy_aligned)
        aligned_capex = sum(
            r.amount_eur for r in self._allocations
            if r.resource_type == ResourceType.CAPEX
            and any(a.action_id == r.action_id and a.is_taxonomy_aligned for a in self._actions)
        )
        total_capex = sum(
            r.amount_eur for r in self._allocations
            if r.resource_type == ResourceType.CAPEX
        )

        outputs["taxonomy_aligned_actions"] = aligned_count
        outputs["taxonomy_aligned_pct"] = round(
            (aligned_count / len(self._actions) * 100) if self._actions else 0.0, 1
        )
        outputs["taxonomy_aligned_capex_eur"] = round(aligned_capex, 2)
        outputs["taxonomy_aligned_capex_pct"] = round(
            (aligned_capex / total_capex * 100) if total_capex > 0 else 0.0, 1
        )

        if aligned_count == 0 and self._actions:
            warnings.append("No actions are EU Taxonomy aligned")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 TaxonomyCheck: %d aligned of %d actions",
            aligned_count, len(self._actions),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.TAXONOMY_CHECK.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: ClimateActionsInput,
    ) -> PhaseResult:
        """Generate E1-3 disclosure-ready output."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["e1_3_disclosure"] = {
            "policies_count": len(self._policies),
            "active_policies": sum(1 for p in self._policies if p.is_active),
            "actions_count": len(self._actions),
            "mitigation_actions": sum(
                1 for a in self._actions
                if a.action_type in (ActionType.MITIGATION, ActionType.MITIGATION_AND_ADAPTATION)
            ),
            "adaptation_actions": sum(
                1 for a in self._actions
                if a.action_type in (ActionType.ADAPTATION, ActionType.MITIGATION_AND_ADAPTATION)
            ),
            "total_expected_reduction_tco2e": round(
                sum(a.expected_reduction_tco2e for a in self._actions), 2
            ),
            "total_capex_eur": round(
                sum(r.amount_eur for r in self._allocations if r.resource_type == ResourceType.CAPEX), 2
            ),
            "total_opex_eur": round(
                sum(r.amount_eur for r in self._allocations if r.resource_type == ResourceType.OPEX), 2
            ),
            "taxonomy_aligned_pct": round(
                (sum(1 for a in self._actions if a.is_taxonomy_aligned)
                 / len(self._actions) * 100) if self._actions else 0.0, 1
            ),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 ReportGeneration: E1-3 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ClimateActionsResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)

    def _count_by_field(self, items: List[Any], field: str) -> Dict[str, int]:
        """Count items by a field value."""
        counts: Dict[str, int] = {}
        for item in items:
            val = getattr(item, field, None)
            key = val.value if hasattr(val, "value") else str(val)
            counts[key] = counts.get(key, 0) + 1
        return counts
