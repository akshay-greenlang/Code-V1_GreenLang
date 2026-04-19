# -*- coding: utf-8 -*-
"""
Recalculation Assessment Workflow
=====================================

4-phase workflow for trigger identification and significance testing
within PACK-045 Base Year Management Pack.

Phases:
    1. TriggerDetection         -- Scan for structural, methodological, and
                                   data events that may require base year
                                   recalculation per GHG Protocol Chapter 5.
    2. SignificanceTesting      -- Quantify each trigger's impact and apply
                                   the organization's significance threshold
                                   to determine recalculation necessity.
    3. PolicyCompliance         -- Verify triggers against the organization's
                                   base year recalculation policy, check
                                   mandatory vs. optional recalculation rules.
    4. RecommendationGeneration -- Produce actionable recommendations with
                                   priority ranking, estimated effort, and
                                   GHG Protocol references.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 5 (Tracking Emissions Over Time)
    ISO 14064-1:2018 Clause 9.3 (Base year recalculation)
    ESRS E1-6 (GHG base year and recalculation requirements)

Schedule: Triggered upon change event detection or annually during review
Estimated duration: 1-3 weeks

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AssessmentPhase(str, Enum):
    """Recalculation assessment workflow phases."""

    TRIGGER_DETECTION = "trigger_detection"
    SIGNIFICANCE_TESTING = "significance_testing"
    POLICY_COMPLIANCE = "policy_compliance"
    RECOMMENDATION_GENERATION = "recommendation_generation"


class TriggerType(str, Enum):
    """Type of recalculation trigger per GHG Protocol."""

    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"
    METHODOLOGY_CHANGE = "methodology_change"
    EMISSION_FACTOR_UPDATE = "emission_factor_update"
    BOUNDARY_CHANGE = "boundary_change"
    DATA_ERROR_CORRECTION = "data_error_correction"
    FACILITY_CLOSURE = "facility_closure"
    FACILITY_OPENING = "facility_opening"
    REGULATORY_CHANGE = "regulatory_change"


class TriggerCategory(str, Enum):
    """GHG Protocol trigger categorization."""

    STRUCTURAL = "structural"
    METHODOLOGICAL = "methodological"
    DATA_ERROR = "data_error"
    EXTERNAL = "external"


class SignificanceLevel(str, Enum):
    """Significance classification for recalculation triggers."""

    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    NEGLIGIBLE = "negligible"


class ComplianceStatus(str, Enum):
    """Policy compliance status for a trigger."""

    MANDATORY_RECALCULATION = "mandatory_recalculation"
    OPTIONAL_RECALCULATION = "optional_recalculation"
    NOT_REQUIRED = "not_required"
    REQUIRES_REVIEW = "requires_review"


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ExternalEvent(BaseModel):
    """External event that may trigger recalculation."""

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:8]}")
    event_type: TriggerType = Field(default=TriggerType.DATA_ERROR_CORRECTION)
    description: str = Field(default="")
    effective_date: str = Field(default="", description="ISO date")
    source: str = Field(default="", description="Event source system or reporter")
    entity_ids_affected: List[str] = Field(default_factory=list)
    estimated_impact_tco2e: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmissionsInventory(BaseModel):
    """Simplified emissions inventory for comparison."""

    year: int = Field(..., ge=2010, le=2050)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    facilities_count: int = Field(default=0, ge=0)
    methodology_version: str = Field(default="")
    categories: Dict[str, float] = Field(default_factory=dict)


class RecalculationPolicy(BaseModel):
    """Organization's base year recalculation policy."""

    significance_threshold_pct: float = Field(
        default=5.0, ge=0.1, le=50.0,
        description="Percentage threshold for significant change",
    )
    mandatory_structural: bool = Field(
        default=True, description="Structural changes always require recalculation",
    )
    mandatory_methodology: bool = Field(
        default=True, description="Methodology changes always require recalculation",
    )
    data_error_threshold_pct: float = Field(
        default=5.0, ge=0.1, le=50.0,
        description="Data error threshold for recalculation",
    )
    cumulative_threshold_pct: float = Field(
        default=10.0, ge=0.1, le=100.0,
        description="Cumulative small-change threshold",
    )
    review_period_months: int = Field(default=12, ge=1, le=60)
    policy_version: str = Field(default="v1.0")


class DetectedTrigger(BaseModel):
    """A detected recalculation trigger with quantification."""

    trigger_id: str = Field(default_factory=lambda: f"trg-{uuid.uuid4().hex[:8]}")
    trigger_type: TriggerType = Field(...)
    category: TriggerCategory = Field(default=TriggerCategory.STRUCTURAL)
    source_event_id: str = Field(default="")
    description: str = Field(default="")
    detected_at: str = Field(default="")
    impact_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    affected_scopes: List[str] = Field(default_factory=list)
    affected_entities: List[str] = Field(default_factory=list)


class SignificanceResult(BaseModel):
    """Significance test result for a single trigger."""

    trigger_id: str = Field(default="")
    significance_level: SignificanceLevel = Field(default=SignificanceLevel.NEGLIGIBLE)
    impact_pct: float = Field(default=0.0, description="Impact as % of base year")
    exceeds_threshold: bool = Field(default=False)
    threshold_applied_pct: float = Field(default=5.0)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class PolicyComplianceResult(BaseModel):
    """Policy compliance check result for a trigger."""

    trigger_id: str = Field(default="")
    compliance_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_REQUIRED)
    policy_rule_applied: str = Field(default="")
    policy_version: str = Field(default="")
    mandatory: bool = Field(default=False)
    justification: str = Field(default="")


class Recommendation(BaseModel):
    """Actionable recommendation for base year management."""

    recommendation_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:8]}")
    trigger_id: str = Field(default="")
    priority: RecommendationPriority = Field(default=RecommendationPriority.MEDIUM)
    action: str = Field(default="")
    rationale: str = Field(default="")
    estimated_effort_days: int = Field(default=0, ge=0)
    ghg_protocol_reference: str = Field(default="")
    deadline_suggestion: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class RecalculationAssessmentInput(BaseModel):
    """Input data model for RecalculationAssessmentWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organization identifier")
    base_year: int = Field(..., ge=2010, le=2050, description="Current base year")
    current_inventory: EmissionsInventory = Field(
        ..., description="Current reporting year inventory",
    )
    previous_inventory: Optional[EmissionsInventory] = Field(
        default=None, description="Previous reporting year inventory for comparison",
    )
    base_year_inventory: Optional[EmissionsInventory] = Field(
        default=None, description="Original base year inventory snapshot",
    )
    external_events: List[ExternalEvent] = Field(
        default_factory=list, description="Events to evaluate as triggers",
    )
    policy: RecalculationPolicy = Field(
        default_factory=RecalculationPolicy,
        description="Organization recalculation policy",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RecalculationAssessmentResult(BaseModel):
    """Complete result from recalculation assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="recalculation_assessment")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    base_year: int = Field(default=0)
    detected_triggers: List[DetectedTrigger] = Field(default_factory=list)
    significance_results: List[SignificanceResult] = Field(default_factory=list)
    policy_compliance: List[PolicyComplianceResult] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    recalculation_required: bool = Field(default=False)
    provenance_hash: str = Field(default="")


# =============================================================================
# TRIGGER CATEGORIZATION MAP (Zero-Hallucination)
# =============================================================================

TRIGGER_CATEGORY_MAP: Dict[TriggerType, TriggerCategory] = {
    TriggerType.ACQUISITION: TriggerCategory.STRUCTURAL,
    TriggerType.DIVESTITURE: TriggerCategory.STRUCTURAL,
    TriggerType.MERGER: TriggerCategory.STRUCTURAL,
    TriggerType.OUTSOURCING: TriggerCategory.STRUCTURAL,
    TriggerType.INSOURCING: TriggerCategory.STRUCTURAL,
    TriggerType.FACILITY_CLOSURE: TriggerCategory.STRUCTURAL,
    TriggerType.FACILITY_OPENING: TriggerCategory.STRUCTURAL,
    TriggerType.METHODOLOGY_CHANGE: TriggerCategory.METHODOLOGICAL,
    TriggerType.EMISSION_FACTOR_UPDATE: TriggerCategory.METHODOLOGICAL,
    TriggerType.BOUNDARY_CHANGE: TriggerCategory.STRUCTURAL,
    TriggerType.DATA_ERROR_CORRECTION: TriggerCategory.DATA_ERROR,
    TriggerType.REGULATORY_CHANGE: TriggerCategory.EXTERNAL,
}

# GHG Protocol recommended references by trigger category
GHG_PROTOCOL_REFS: Dict[str, str] = {
    "structural": "GHG Protocol Corporate Standard, Chapter 5, Section 5.4",
    "methodological": "GHG Protocol Corporate Standard, Chapter 5, Section 5.5",
    "data_error": "GHG Protocol Corporate Standard, Chapter 5, Section 5.6",
    "external": "GHG Protocol Corporate Standard, Chapter 5",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RecalculationAssessmentWorkflow:
    """
    4-phase workflow for trigger identification and significance testing.

    Scans for events that may require base year recalculation, quantifies
    impact, verifies policy compliance, and generates prioritized
    recommendations aligned with GHG Protocol Chapter 5 guidance.

    Zero-hallucination: significance thresholds use deterministic
    percentage-of-base-year calculation, no LLM calls in quantification.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _triggers: Detected recalculation triggers.
        _significance: Significance test results.
        _compliance: Policy compliance results.
        _recommendations: Prioritized recommendations.

    Example:
        >>> wf = RecalculationAssessmentWorkflow()
        >>> event = ExternalEvent(event_type=TriggerType.ACQUISITION)
        >>> inv = EmissionsInventory(year=2025, total_tco2e=50000.0)
        >>> inp = RecalculationAssessmentInput(
        ...     organization_id="org-001", base_year=2022,
        ...     current_inventory=inv, external_events=[event],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[AssessmentPhase] = [
        AssessmentPhase.TRIGGER_DETECTION,
        AssessmentPhase.SIGNIFICANCE_TESTING,
        AssessmentPhase.POLICY_COMPLIANCE,
        AssessmentPhase.RECOMMENDATION_GENERATION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RecalculationAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._triggers: List[DetectedTrigger] = []
        self._significance: List[SignificanceResult] = []
        self._compliance: List[PolicyComplianceResult] = []
        self._recommendations: List[Recommendation] = []
        self._recalculation_required: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: RecalculationAssessmentInput,
    ) -> RecalculationAssessmentResult:
        """
        Execute the 4-phase recalculation assessment workflow.

        Args:
            input_data: Base year context, events, and policy configuration.

        Returns:
            RecalculationAssessmentResult with triggers, significance, recommendations.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting recalculation assessment %s org=%s base_year=%d events=%d",
            self.workflow_id, input_data.organization_id,
            input_data.base_year, len(input_data.external_events),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_trigger_detection,
            self._phase_significance_testing,
            self._phase_policy_compliance,
            self._phase_recommendation_generation,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._execute_with_retry(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Recalculation assessment failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = RecalculationAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            base_year=input_data.base_year,
            detected_triggers=self._triggers,
            significance_results=self._significance,
            policy_compliance=self._compliance,
            recommendations=self._recommendations,
            recalculation_required=self._recalculation_required,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Recalculation assessment %s completed in %.2fs status=%s triggers=%d recalc=%s",
            self.workflow_id, elapsed, overall_status.value,
            len(self._triggers), self._recalculation_required,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: RecalculationAssessmentInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Trigger Detection
    # -------------------------------------------------------------------------

    async def _phase_trigger_detection(
        self, input_data: RecalculationAssessmentInput,
    ) -> PhaseResult:
        """Scan for structural, methodological, and data events."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._triggers = []
        now_iso = datetime.utcnow().isoformat()

        # Process external events into triggers
        for event in input_data.external_events:
            category = TRIGGER_CATEGORY_MAP.get(
                event.event_type, TriggerCategory.EXTERNAL,
            )

            # Estimate impact from event or use inventory comparison
            impact_tco2e = event.estimated_impact_tco2e
            base_total = (
                input_data.base_year_inventory.total_tco2e
                if input_data.base_year_inventory
                else input_data.current_inventory.total_tco2e
            )
            impact_pct = (
                (abs(impact_tco2e) / max(base_total, 1.0)) * 100.0
                if impact_tco2e != 0.0
                else 0.0
            )

            self._triggers.append(DetectedTrigger(
                trigger_type=event.event_type,
                category=category,
                source_event_id=event.event_id,
                description=event.description or f"{event.event_type.value} event detected",
                detected_at=now_iso,
                impact_tco2e=round(impact_tco2e, 2),
                impact_pct=round(impact_pct, 2),
                affected_scopes=["scope1", "scope2"],
                affected_entities=event.entity_ids_affected,
            ))

        # Auto-detect triggers from inventory comparison
        if input_data.previous_inventory and input_data.base_year_inventory:
            variance = self._detect_inventory_variance(
                input_data.current_inventory,
                input_data.previous_inventory,
                input_data.base_year_inventory,
            )
            if variance:
                self._triggers.extend(variance)
                warnings.append(
                    f"Auto-detected {len(variance)} variance-based triggers"
                )

        # Categorize triggers
        category_counts: Dict[str, int] = {}
        for trg in self._triggers:
            cat = trg.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        outputs["triggers_detected"] = len(self._triggers)
        outputs["category_counts"] = category_counts
        outputs["event_sources"] = len(input_data.external_events)
        outputs["auto_detected"] = len(self._triggers) - len(input_data.external_events)

        if not self._triggers:
            warnings.append("No recalculation triggers detected")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 TriggerDetection: %d triggers detected (%s)",
            len(self._triggers), category_counts,
        )
        return PhaseResult(
            phase_name="trigger_detection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _detect_inventory_variance(
        self,
        current: EmissionsInventory,
        previous: EmissionsInventory,
        base_year: EmissionsInventory,
    ) -> List[DetectedTrigger]:
        """Detect variance-based triggers from inventory comparison."""
        triggers: List[DetectedTrigger] = []
        now_iso = datetime.utcnow().isoformat()

        # Year-over-year variance check
        if previous.total_tco2e > 0:
            yoy_change = current.total_tco2e - previous.total_tco2e
            yoy_pct = (abs(yoy_change) / previous.total_tco2e) * 100.0

            if yoy_pct > 20.0:
                triggers.append(DetectedTrigger(
                    trigger_type=TriggerType.DATA_ERROR_CORRECTION,
                    category=TriggerCategory.DATA_ERROR,
                    description=(
                        f"Unusual year-over-year variance: {yoy_pct:.1f}% "
                        f"({yoy_change:+.2f} tCO2e)"
                    ),
                    detected_at=now_iso,
                    impact_tco2e=round(yoy_change, 2),
                    impact_pct=round(
                        (abs(yoy_change) / max(base_year.total_tco2e, 1.0)) * 100.0, 2,
                    ),
                    affected_scopes=["scope1", "scope2", "scope3"],
                ))

        # Scope-level variance checks
        for scope_attr in ["scope1_tco2e", "scope2_tco2e", "scope3_tco2e"]:
            curr_val = getattr(current, scope_attr, 0.0)
            prev_val = getattr(previous, scope_attr, 0.0)
            if prev_val > 0:
                scope_pct = (abs(curr_val - prev_val) / prev_val) * 100.0
                if scope_pct > 30.0:
                    scope_name = scope_attr.replace("_tco2e", "")
                    triggers.append(DetectedTrigger(
                        trigger_type=TriggerType.BOUNDARY_CHANGE,
                        category=TriggerCategory.STRUCTURAL,
                        description=(
                            f"Significant {scope_name} variance: {scope_pct:.1f}%"
                        ),
                        detected_at=now_iso,
                        impact_tco2e=round(curr_val - prev_val, 2),
                        impact_pct=round(
                            (abs(curr_val - prev_val) / max(base_year.total_tco2e, 1.0)) * 100.0,
                            2,
                        ),
                        affected_scopes=[scope_name],
                    ))

        return triggers

    # -------------------------------------------------------------------------
    # Phase 2: Significance Testing
    # -------------------------------------------------------------------------

    async def _phase_significance_testing(
        self, input_data: RecalculationAssessmentInput,
    ) -> PhaseResult:
        """Quantify trigger impact and apply significance threshold."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._significance = []
        threshold = input_data.policy.significance_threshold_pct
        base_total = (
            input_data.base_year_inventory.total_tco2e
            if input_data.base_year_inventory
            else input_data.current_inventory.total_tco2e
        )

        cumulative_impact_pct = 0.0

        for trigger in self._triggers:
            # Recalculate impact as percentage of base year
            impact_pct = (
                (abs(trigger.impact_tco2e) / max(base_total, 1.0)) * 100.0
            )
            exceeds = impact_pct >= threshold

            # Determine significance level
            if impact_pct >= threshold:
                level = SignificanceLevel.SIGNIFICANT
            elif impact_pct >= threshold * 0.5:
                level = SignificanceLevel.MODERATE
            elif impact_pct >= 1.0:
                level = SignificanceLevel.MINOR
            else:
                level = SignificanceLevel.NEGLIGIBLE

            calc_details = {
                "impact_tco2e": round(trigger.impact_tco2e, 2),
                "base_year_total_tco2e": round(base_total, 2),
                "impact_pct": round(impact_pct, 4),
                "threshold_pct": threshold,
                "formula": "abs(impact_tco2e) / base_year_total * 100",
            }

            sig_hash = hashlib.sha256(
                json.dumps(calc_details, sort_keys=True).encode("utf-8")
            ).hexdigest()

            self._significance.append(SignificanceResult(
                trigger_id=trigger.trigger_id,
                significance_level=level,
                impact_pct=round(impact_pct, 4),
                exceeds_threshold=exceeds,
                threshold_applied_pct=threshold,
                calculation_details=calc_details,
                provenance_hash=sig_hash,
            ))

            cumulative_impact_pct += impact_pct

        # Check cumulative threshold
        cumulative_exceeds = cumulative_impact_pct >= input_data.policy.cumulative_threshold_pct
        if cumulative_exceeds:
            warnings.append(
                f"Cumulative impact {cumulative_impact_pct:.2f}% exceeds "
                f"threshold {input_data.policy.cumulative_threshold_pct:.1f}%"
            )

        significant_count = sum(
            1 for s in self._significance
            if s.significance_level == SignificanceLevel.SIGNIFICANT
        )

        outputs["triggers_tested"] = len(self._significance)
        outputs["significant"] = significant_count
        outputs["moderate"] = sum(
            1 for s in self._significance
            if s.significance_level == SignificanceLevel.MODERATE
        )
        outputs["minor"] = sum(
            1 for s in self._significance
            if s.significance_level == SignificanceLevel.MINOR
        )
        outputs["negligible"] = sum(
            1 for s in self._significance
            if s.significance_level == SignificanceLevel.NEGLIGIBLE
        )
        outputs["threshold_applied_pct"] = threshold
        outputs["cumulative_impact_pct"] = round(cumulative_impact_pct, 4)
        outputs["cumulative_exceeds_threshold"] = cumulative_exceeds

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 SignificanceTesting: %d significant, cumulative=%.2f%%",
            significant_count, cumulative_impact_pct,
        )
        return PhaseResult(
            phase_name="significance_testing", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Policy Compliance
    # -------------------------------------------------------------------------

    async def _phase_policy_compliance(
        self, input_data: RecalculationAssessmentInput,
    ) -> PhaseResult:
        """Verify triggers against organization recalculation policy."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._compliance = []
        self._recalculation_required = False
        policy = input_data.policy

        for trigger, sig in zip(self._triggers, self._significance):
            compliance = self._evaluate_policy_compliance(trigger, sig, policy)
            self._compliance.append(compliance)

            if compliance.mandatory:
                self._recalculation_required = True

        # Check if optional recalculations should be recommended
        optional_count = sum(
            1 for c in self._compliance
            if c.compliance_status == ComplianceStatus.OPTIONAL_RECALCULATION
        )

        mandatory_count = sum(
            1 for c in self._compliance
            if c.compliance_status == ComplianceStatus.MANDATORY_RECALCULATION
        )

        outputs["mandatory_recalculations"] = mandatory_count
        outputs["optional_recalculations"] = optional_count
        outputs["not_required"] = sum(
            1 for c in self._compliance
            if c.compliance_status == ComplianceStatus.NOT_REQUIRED
        )
        outputs["requires_review"] = sum(
            1 for c in self._compliance
            if c.compliance_status == ComplianceStatus.REQUIRES_REVIEW
        )
        outputs["recalculation_required"] = self._recalculation_required
        outputs["policy_version"] = policy.policy_version

        if mandatory_count > 0:
            warnings.append(
                f"{mandatory_count} trigger(s) require mandatory base year recalculation"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PolicyCompliance: mandatory=%d optional=%d recalc_required=%s",
            mandatory_count, optional_count, self._recalculation_required,
        )
        return PhaseResult(
            phase_name="policy_compliance", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _evaluate_policy_compliance(
        self,
        trigger: DetectedTrigger,
        sig: SignificanceResult,
        policy: RecalculationPolicy,
    ) -> PolicyComplianceResult:
        """Evaluate a single trigger against recalculation policy."""
        # Structural changes with mandatory policy
        if (
            trigger.category == TriggerCategory.STRUCTURAL
            and policy.mandatory_structural
            and sig.exceeds_threshold
        ):
            return PolicyComplianceResult(
                trigger_id=trigger.trigger_id,
                compliance_status=ComplianceStatus.MANDATORY_RECALCULATION,
                policy_rule_applied="mandatory_structural",
                policy_version=policy.policy_version,
                mandatory=True,
                justification=(
                    f"Structural change exceeds {policy.significance_threshold_pct}% "
                    f"threshold ({sig.impact_pct:.2f}%); mandatory per policy."
                ),
            )

        # Methodology changes with mandatory policy
        if (
            trigger.category == TriggerCategory.METHODOLOGICAL
            and policy.mandatory_methodology
        ):
            return PolicyComplianceResult(
                trigger_id=trigger.trigger_id,
                compliance_status=ComplianceStatus.MANDATORY_RECALCULATION,
                policy_rule_applied="mandatory_methodology",
                policy_version=policy.policy_version,
                mandatory=True,
                justification=(
                    f"Methodology change triggers mandatory recalculation per policy."
                ),
            )

        # Data errors exceeding threshold
        if (
            trigger.category == TriggerCategory.DATA_ERROR
            and sig.impact_pct >= policy.data_error_threshold_pct
        ):
            return PolicyComplianceResult(
                trigger_id=trigger.trigger_id,
                compliance_status=ComplianceStatus.MANDATORY_RECALCULATION,
                policy_rule_applied="data_error_threshold",
                policy_version=policy.policy_version,
                mandatory=True,
                justification=(
                    f"Data error impact {sig.impact_pct:.2f}% exceeds "
                    f"{policy.data_error_threshold_pct}% threshold."
                ),
            )

        # Significant but not mandatory
        if sig.exceeds_threshold:
            return PolicyComplianceResult(
                trigger_id=trigger.trigger_id,
                compliance_status=ComplianceStatus.OPTIONAL_RECALCULATION,
                policy_rule_applied="significance_threshold",
                policy_version=policy.policy_version,
                mandatory=False,
                justification=(
                    f"Impact {sig.impact_pct:.2f}% exceeds threshold but "
                    f"category does not require mandatory recalculation."
                ),
            )

        # Below threshold
        if sig.significance_level in (SignificanceLevel.MODERATE,):
            return PolicyComplianceResult(
                trigger_id=trigger.trigger_id,
                compliance_status=ComplianceStatus.REQUIRES_REVIEW,
                policy_rule_applied="moderate_review",
                policy_version=policy.policy_version,
                mandatory=False,
                justification=(
                    f"Moderate impact ({sig.impact_pct:.2f}%); recommend review."
                ),
            )

        return PolicyComplianceResult(
            trigger_id=trigger.trigger_id,
            compliance_status=ComplianceStatus.NOT_REQUIRED,
            policy_rule_applied="below_threshold",
            policy_version=policy.policy_version,
            mandatory=False,
            justification=(
                f"Impact {sig.impact_pct:.2f}% below significance threshold; "
                f"no recalculation required."
            ),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Recommendation Generation
    # -------------------------------------------------------------------------

    async def _phase_recommendation_generation(
        self, input_data: RecalculationAssessmentInput,
    ) -> PhaseResult:
        """Produce actionable recommendations with priority ranking."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._recommendations = []

        for trigger, sig, comp in zip(
            self._triggers, self._significance, self._compliance,
        ):
            priority = self._determine_priority(sig, comp)
            action = self._build_action(trigger, sig, comp)
            effort = self._estimate_effort(trigger, sig)
            ref = GHG_PROTOCOL_REFS.get(trigger.category.value, "GHG Protocol Chapter 5")

            self._recommendations.append(Recommendation(
                trigger_id=trigger.trigger_id,
                priority=priority,
                action=action,
                rationale=comp.justification,
                estimated_effort_days=effort,
                ghg_protocol_reference=ref,
                deadline_suggestion=self._suggest_deadline(priority),
            ))

        # Sort by priority
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
            RecommendationPriority.INFORMATIONAL: 4,
        }
        self._recommendations.sort(
            key=lambda r: priority_order.get(r.priority, 99),
        )

        outputs["recommendations_count"] = len(self._recommendations)
        outputs["priority_distribution"] = {
            p.value: sum(1 for r in self._recommendations if r.priority == p)
            for p in RecommendationPriority
        }
        outputs["total_estimated_effort_days"] = sum(
            r.estimated_effort_days for r in self._recommendations
        )
        outputs["recalculation_required"] = self._recalculation_required

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RecommendationGeneration: %d recommendations, effort=%d days",
            len(self._recommendations), outputs["total_estimated_effort_days"],
        )
        return PhaseResult(
            phase_name="recommendation_generation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _determine_priority(
        self, sig: SignificanceResult, comp: PolicyComplianceResult,
    ) -> RecommendationPriority:
        """Determine recommendation priority from significance and compliance."""
        if comp.compliance_status == ComplianceStatus.MANDATORY_RECALCULATION:
            if sig.significance_level == SignificanceLevel.SIGNIFICANT:
                return RecommendationPriority.CRITICAL
            return RecommendationPriority.HIGH
        if comp.compliance_status == ComplianceStatus.OPTIONAL_RECALCULATION:
            return RecommendationPriority.MEDIUM
        if comp.compliance_status == ComplianceStatus.REQUIRES_REVIEW:
            return RecommendationPriority.LOW
        return RecommendationPriority.INFORMATIONAL

    def _build_action(
        self,
        trigger: DetectedTrigger,
        sig: SignificanceResult,
        comp: PolicyComplianceResult,
    ) -> str:
        """Build actionable recommendation text."""
        if comp.mandatory:
            return (
                f"Execute base year recalculation for {trigger.trigger_type.value} "
                f"(impact: {sig.impact_pct:.2f}%). Update base year inventory, "
                f"restate historical time series, and update all downstream reports."
            )
        if comp.compliance_status == ComplianceStatus.OPTIONAL_RECALCULATION:
            return (
                f"Consider base year recalculation for {trigger.trigger_type.value} "
                f"(impact: {sig.impact_pct:.2f}%). Document decision rationale "
                f"regardless of recalculation outcome."
            )
        if comp.compliance_status == ComplianceStatus.REQUIRES_REVIEW:
            return (
                f"Review {trigger.trigger_type.value} impact ({sig.impact_pct:.2f}%) "
                f"with GHG inventory team. Document review outcome in audit trail."
            )
        return (
            f"Log {trigger.trigger_type.value} event for audit trail. "
            f"No recalculation action required (impact: {sig.impact_pct:.2f}%)."
        )

    def _estimate_effort(
        self, trigger: DetectedTrigger, sig: SignificanceResult,
    ) -> int:
        """Estimate implementation effort in working days."""
        base_effort: Dict[TriggerCategory, int] = {
            TriggerCategory.STRUCTURAL: 10,
            TriggerCategory.METHODOLOGICAL: 7,
            TriggerCategory.DATA_ERROR: 3,
            TriggerCategory.EXTERNAL: 5,
        }
        effort = base_effort.get(trigger.category, 5)
        if sig.significance_level == SignificanceLevel.SIGNIFICANT:
            effort = int(effort * 1.5)
        elif sig.significance_level == SignificanceLevel.NEGLIGIBLE:
            effort = max(1, effort // 3)
        return effort

    def _suggest_deadline(self, priority: RecommendationPriority) -> str:
        """Suggest a deadline based on priority level."""
        deadline_days: Dict[RecommendationPriority, int] = {
            RecommendationPriority.CRITICAL: 14,
            RecommendationPriority.HIGH: 30,
            RecommendationPriority.MEDIUM: 60,
            RecommendationPriority.LOW: 90,
            RecommendationPriority.INFORMATIONAL: 180,
        }
        days = deadline_days.get(priority, 60)
        return f"{days} days from assessment date"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._triggers = []
        self._significance = []
        self._compliance = []
        self._recommendations = []
        self._recalculation_required = False

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: RecalculationAssessmentResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{result.base_year}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
