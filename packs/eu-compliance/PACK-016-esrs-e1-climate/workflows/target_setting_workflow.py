# -*- coding: utf-8 -*-
"""
Target Setting Workflow
==============================

5-phase workflow for climate target setting and validation per ESRS E1-4.
Implements baseline determination, target definition, SBTi validation,
progress assessment, and report generation with full provenance tracking.

Phases:
    1. BaselineDetermination  -- Establish base year emissions
    2. TargetDefinition       -- Define reduction targets per scope
    3. SBTiValidation         -- Validate against SBTi criteria
    4. ProgressAssessment     -- Assess progress vs. targets
    5. ReportGeneration       -- Produce E1-4 disclosure data

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
    """Phases of the target setting workflow."""
    BASELINE_DETERMINATION = "baseline_determination"
    TARGET_DEFINITION = "target_definition"
    SBTI_VALIDATION = "sbti_validation"
    PROGRESS_ASSESSMENT = "progress_assessment"
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

class TargetType(str, Enum):
    """Climate target type classification."""
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    NET_ZERO = "net_zero"

class TargetScope(str, Enum):
    """Scope to which a target applies."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"

class SBTiStatus(str, Enum):
    """SBTi target validation status."""
    VALIDATED = "validated"
    COMMITTED = "committed"
    PENDING = "pending"
    NOT_SUBMITTED = "not_submitted"
    NON_COMPLIANT = "non_compliant"

class ProgressStatus(str, Enum):
    """Progress towards target."""
    ON_TRACK = "on_track"
    SLIGHTLY_BEHIND = "slightly_behind"
    SIGNIFICANTLY_BEHIND = "significantly_behind"
    EXCEEDED = "exceeded"
    NOT_STARTED = "not_started"

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

class ClimateTarget(BaseModel):
    """A defined climate target."""
    target_id: str = Field(default_factory=lambda: f"ct-{_new_uuid()[:8]}")
    name: str = Field(..., description="Target name")
    target_type: TargetType = Field(..., description="Type of target")
    target_scope: TargetScope = Field(..., description="Scope applicability")
    base_year: int = Field(default=2019, ge=1990, le=2050)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    target_value: float = Field(default=0.0, description="Target value (% reduction or absolute)")
    target_unit: str = Field(default="pct_reduction", description="pct_reduction, tco2e, mwh_per_eur")
    interim_milestones: Dict[int, float] = Field(
        default_factory=dict, description="Year -> target value milestones"
    )
    current_value: float = Field(default=0.0, description="Current performance value")
    sbti_status: SBTiStatus = Field(default=SBTiStatus.NOT_SUBMITTED)
    is_science_based: bool = Field(default=False)
    description: str = Field(default="")

class SBTiValidationResult(BaseModel):
    """SBTi validation assessment for a target."""
    target_id: str = Field(default="")
    is_compliant: bool = Field(default=False)
    ambition_level: str = Field(default="", description="1.5C or WB2C")
    required_annual_reduction_pct: float = Field(default=0.0)
    actual_annual_reduction_pct: float = Field(default=0.0)
    scope_coverage_adequate: bool = Field(default=False)
    base_year_valid: bool = Field(default=False)
    time_frame_valid: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)

class TargetSettingInput(BaseModel):
    """Input data model for TargetSettingWorkflow."""
    targets: List[ClimateTarget] = Field(
        default_factory=list, description="Defined climate targets"
    )
    scope_1_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope_2_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope_3_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    sbti_sector: str = Field(default="", description="SBTi sector classification")
    config: Dict[str, Any] = Field(default_factory=dict)

class TargetSettingResult(BaseModel):
    """Complete result from target setting workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="target_setting")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    targets: List[ClimateTarget] = Field(default_factory=list)
    sbti_validations: List[SBTiValidationResult] = Field(default_factory=list)
    targets_on_track: int = Field(default=0)
    targets_behind: int = Field(default=0)
    overall_progress_pct: float = Field(default=0.0)
    has_scope_1_2_target: bool = Field(default=False)
    has_scope_3_target: bool = Field(default=False)
    all_sbti_compliant: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# SBTI REQUIREMENTS
# =============================================================================

SBTI_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "1.5c": {
        "annual_reduction_scope_1_2_pct": 4.2,
        "scope_1_2_coverage_pct": 95.0,
        "scope_3_coverage_pct": 67.0,
        "max_target_year_offset": 10,
        "min_base_year": 2015,
    },
    "wb2c": {
        "annual_reduction_scope_1_2_pct": 2.5,
        "scope_1_2_coverage_pct": 95.0,
        "scope_3_coverage_pct": 67.0,
        "max_target_year_offset": 15,
        "min_base_year": 2015,
    },
}

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class TargetSettingWorkflow:
    """
    5-phase climate target setting and validation workflow for ESRS E1-4.

    Implements target definition, SBTi validation, progress tracking,
    and disclosure-ready output generation. Validates targets against
    Science Based Targets initiative criteria for 1.5C and WB2C pathways.

    Zero-hallucination: all progress calculations and SBTi checks use
    deterministic arithmetic with documented criteria.

    Example:
        >>> wf = TargetSettingWorkflow()
        >>> inp = TargetSettingInput(targets=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.has_scope_1_2_target is True
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TargetSettingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._targets: List[ClimateTarget] = []
        self._sbti_results: List[SBTiValidationResult] = []
        self._progress_map: Dict[str, ProgressStatus] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.BASELINE_DETERMINATION.value, "description": "Establish base year emissions"},
            {"name": WorkflowPhase.TARGET_DEFINITION.value, "description": "Define reduction targets per scope"},
            {"name": WorkflowPhase.SBTI_VALIDATION.value, "description": "Validate against SBTi criteria"},
            {"name": WorkflowPhase.PROGRESS_ASSESSMENT.value, "description": "Assess progress vs targets"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-4 disclosure data"},
        ]

    def validate_inputs(self, input_data: TargetSettingInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.targets:
            issues.append("No climate targets defined")
        for t in input_data.targets:
            if t.target_year <= t.base_year:
                issues.append(f"Target {t.target_id}: target year must be after base year")
            if t.target_type == TargetType.ABSOLUTE and t.target_value < 0:
                issues.append(f"Target {t.target_id}: absolute target value cannot be negative")
        return issues

    async def execute(
        self,
        input_data: Optional[TargetSettingInput] = None,
        targets: Optional[List[ClimateTarget]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TargetSettingResult:
        """
        Execute the 5-phase target setting workflow.

        Args:
            input_data: Full input model (preferred).
            targets: Climate targets (fallback).
            config: Configuration overrides.

        Returns:
            TargetSettingResult with SBTi validations and progress assessment.
        """
        if input_data is None:
            input_data = TargetSettingInput(
                targets=targets or [],
                config=config or {},
            )

        started_at = utcnow()
        self.logger.info("Starting target setting workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_baseline_determination(input_data))
            phase_results.append(await self._phase_target_definition(input_data))
            phase_results.append(await self._phase_sbti_validation(input_data))
            phase_results.append(await self._phase_progress_assessment(input_data))
            phase_results.append(await self._phase_report_generation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Target setting workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        on_track = sum(1 for s in self._progress_map.values() if s in (ProgressStatus.ON_TRACK, ProgressStatus.EXCEEDED))
        behind = sum(1 for s in self._progress_map.values() if s in (ProgressStatus.SLIGHTLY_BEHIND, ProgressStatus.SIGNIFICANTLY_BEHIND))
        has_s12 = any(t.target_scope in (TargetScope.SCOPE_1, TargetScope.SCOPE_2, TargetScope.SCOPE_1_2) for t in self._targets)
        has_s3 = any(t.target_scope in (TargetScope.SCOPE_3, TargetScope.ALL_SCOPES) for t in self._targets)
        all_sbti = all(v.is_compliant for v in self._sbti_results) if self._sbti_results else False

        result = TargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            targets=self._targets,
            sbti_validations=self._sbti_results,
            targets_on_track=on_track,
            targets_behind=behind,
            overall_progress_pct=round(
                (on_track / len(self._targets) * 100) if self._targets else 0.0, 1
            ),
            has_scope_1_2_target=has_s12,
            has_scope_3_target=has_s3,
            all_sbti_compliant=all_sbti,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Target setting %s completed in %.2fs: %d on track, %d behind",
            self.workflow_id, elapsed, on_track, behind,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Determination
    # -------------------------------------------------------------------------

    async def _phase_baseline_determination(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Establish base year emissions for target tracking."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_current = (
            input_data.scope_1_emissions_tco2e
            + input_data.scope_2_emissions_tco2e
            + input_data.scope_3_emissions_tco2e
        )

        outputs["scope_1_current_tco2e"] = input_data.scope_1_emissions_tco2e
        outputs["scope_2_current_tco2e"] = input_data.scope_2_emissions_tco2e
        outputs["scope_3_current_tco2e"] = input_data.scope_3_emissions_tco2e
        outputs["total_current_tco2e"] = round(total_current, 2)

        # Validate base year consistency across targets
        base_years = set(t.base_year for t in input_data.targets)
        if len(base_years) > 1:
            warnings.append(f"Multiple base years used across targets: {sorted(base_years)}")

        outputs["base_years_used"] = sorted(list(base_years))

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 BaselineDetermination: total=%.0f tCO2e", total_current)
        return PhaseResult(
            phase_name=WorkflowPhase.BASELINE_DETERMINATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Target Definition
    # -------------------------------------------------------------------------

    async def _phase_target_definition(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Catalog and validate target definitions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._targets = list(input_data.targets)

        type_counts: Dict[str, int] = {}
        scope_counts: Dict[str, int] = {}
        for t in self._targets:
            type_counts[t.target_type.value] = type_counts.get(t.target_type.value, 0) + 1
            scope_counts[t.target_scope.value] = scope_counts.get(t.target_scope.value, 0) + 1

        outputs["targets_defined"] = len(self._targets)
        outputs["type_distribution"] = type_counts
        outputs["scope_distribution"] = scope_counts
        outputs["target_years"] = sorted(set(t.target_year for t in self._targets))

        if not self._targets:
            warnings.append("No climate targets defined")

        # Check for near-term and long-term targets
        near_term = [t for t in self._targets if t.target_year <= input_data.reporting_year + 5]
        long_term = [t for t in self._targets if t.target_year > input_data.reporting_year + 10]
        if not near_term:
            warnings.append("No near-term targets (within 5 years)")
        if not long_term:
            warnings.append("No long-term targets (beyond 10 years)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 TargetDefinition: %d targets defined", len(self._targets))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_DEFINITION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: SBTi Validation
    # -------------------------------------------------------------------------

    async def _phase_sbti_validation(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Validate targets against SBTi criteria."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._sbti_results = []

        for target in self._targets:
            validation = self._validate_sbti_target(target, input_data)
            self._sbti_results.append(validation)

        compliant_count = sum(1 for v in self._sbti_results if v.is_compliant)
        outputs["targets_validated"] = len(self._sbti_results)
        outputs["sbti_compliant"] = compliant_count
        outputs["sbti_non_compliant"] = len(self._sbti_results) - compliant_count

        if compliant_count < len(self._sbti_results):
            warnings.append(
                f"{len(self._sbti_results) - compliant_count} targets do not meet SBTi criteria"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SBTiValidation: %d compliant of %d",
            compliant_count, len(self._sbti_results),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SBTI_VALIDATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _validate_sbti_target(
        self, target: ClimateTarget, input_data: TargetSettingInput,
    ) -> SBTiValidationResult:
        """Validate a single target against SBTi criteria."""
        issues: List[str] = []
        ambition = "1.5c"  # Default to most ambitious

        reqs = SBTI_REQUIREMENTS[ambition]

        # Base year check
        base_year_valid = target.base_year >= reqs["min_base_year"]
        if not base_year_valid:
            issues.append(f"Base year {target.base_year} is before {reqs['min_base_year']}")

        # Time frame check
        years = target.target_year - target.base_year
        time_frame_valid = 5 <= years <= reqs["max_target_year_offset"]
        if not time_frame_valid:
            issues.append(f"Target time frame ({years} years) outside SBTi range")

        # Annual reduction rate
        actual_annual = (target.target_value / years) if years > 0 else 0.0
        required_annual = reqs["annual_reduction_scope_1_2_pct"]
        reduction_adequate = actual_annual >= required_annual

        if not reduction_adequate:
            issues.append(
                f"Annual reduction ({actual_annual:.1f}%) below SBTi "
                f"minimum ({required_annual}%)"
            )
            # Check WB2C as fallback
            wb2c_rate = SBTI_REQUIREMENTS["wb2c"]["annual_reduction_scope_1_2_pct"]
            if actual_annual >= wb2c_rate:
                ambition = "wb2c"
                reduction_adequate = True
                issues.pop()

        scope_coverage = target.target_scope in (
            TargetScope.SCOPE_1_2, TargetScope.ALL_SCOPES,
            TargetScope.SCOPE_1, TargetScope.SCOPE_2,
        )
        if not scope_coverage:
            issues.append("Target does not cover Scope 1+2 emissions")

        is_compliant = base_year_valid and time_frame_valid and reduction_adequate and scope_coverage

        return SBTiValidationResult(
            target_id=target.target_id,
            is_compliant=is_compliant,
            ambition_level=ambition,
            required_annual_reduction_pct=required_annual,
            actual_annual_reduction_pct=round(actual_annual, 2),
            scope_coverage_adequate=scope_coverage,
            base_year_valid=base_year_valid,
            time_frame_valid=time_frame_valid,
            issues=issues,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Progress Assessment
    # -------------------------------------------------------------------------

    async def _phase_progress_assessment(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Assess progress toward each target."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._progress_map = {}

        for target in self._targets:
            progress_status = self._assess_progress(target, input_data)
            self._progress_map[target.target_id] = progress_status

        status_counts: Dict[str, int] = {}
        for status in self._progress_map.values():
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

        outputs["progress_distribution"] = status_counts
        outputs["on_track_count"] = status_counts.get("on_track", 0) + status_counts.get("exceeded", 0)
        outputs["behind_count"] = status_counts.get("slightly_behind", 0) + status_counts.get("significantly_behind", 0)

        behind_targets = [
            tid for tid, s in self._progress_map.items()
            if s in (ProgressStatus.SLIGHTLY_BEHIND, ProgressStatus.SIGNIFICANTLY_BEHIND)
        ]
        if behind_targets:
            warnings.append(f"{len(behind_targets)} targets are behind schedule")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ProgressAssessment: %s", status_counts)
        return PhaseResult(
            phase_name=WorkflowPhase.PROGRESS_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_progress(
        self, target: ClimateTarget, input_data: TargetSettingInput,
    ) -> ProgressStatus:
        """Assess progress for a single target."""
        if target.base_year_emissions_tco2e <= 0:
            return ProgressStatus.NOT_STARTED

        total_years = max(1, target.target_year - target.base_year)
        elapsed_years = max(0, input_data.reporting_year - target.base_year)
        expected_progress_pct = min(100.0, (elapsed_years / total_years) * target.target_value)

        actual_reduction = target.base_year_emissions_tco2e - target.current_value
        actual_progress_pct = (
            (actual_reduction / target.base_year_emissions_tco2e * 100)
            if target.base_year_emissions_tco2e > 0 else 0.0
        )

        if actual_progress_pct >= expected_progress_pct * 1.1:
            return ProgressStatus.EXCEEDED
        elif actual_progress_pct >= expected_progress_pct * 0.9:
            return ProgressStatus.ON_TRACK
        elif actual_progress_pct >= expected_progress_pct * 0.7:
            return ProgressStatus.SLIGHTLY_BEHIND
        else:
            return ProgressStatus.SIGNIFICANTLY_BEHIND

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Generate E1-4 disclosure-ready output."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        target_summaries = []
        for t in self._targets:
            progress = self._progress_map.get(t.target_id, ProgressStatus.NOT_STARTED)
            sbti = next((v for v in self._sbti_results if v.target_id == t.target_id), None)
            target_summaries.append({
                "target_id": t.target_id,
                "name": t.name,
                "type": t.target_type.value,
                "scope": t.target_scope.value,
                "target_year": t.target_year,
                "target_value": t.target_value,
                "progress_status": progress.value,
                "sbti_compliant": sbti.is_compliant if sbti else False,
            })

        outputs["e1_4_disclosure"] = {
            "targets_count": len(self._targets),
            "targets": target_summaries,
            "has_scope_1_2_target": any(
                t.target_scope in (TargetScope.SCOPE_1_2, TargetScope.ALL_SCOPES)
                for t in self._targets
            ),
            "has_scope_3_target": any(
                t.target_scope in (TargetScope.SCOPE_3, TargetScope.ALL_SCOPES)
                for t in self._targets
            ),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 ReportGeneration: E1-4 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: TargetSettingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
