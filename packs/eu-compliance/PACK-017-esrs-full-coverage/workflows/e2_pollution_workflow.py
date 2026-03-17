# -*- coding: utf-8 -*-
"""
ESRS E2 Pollution Workflow
===============================================

5-phase workflow for ESRS E2 Pollution disclosure covering policy review,
emissions to air/water/soil calculation, substances of concern assessment,
target evaluation, and financial effects analysis with full provenance tracking.

Phases:
    1. PolicyReview           -- Review pollution policies and actions (E2-1, E2-2)
    2. EmissionsCalculation   -- Calculate pollutant emissions to air, water, soil (E2-4)
    3. SubstancesAssessment   -- Assess substances of concern and SVHCs (E2-5)
    4. TargetEvaluation       -- Evaluate pollution reduction targets (E2-3)
    5. FinancialEffects       -- Assess financial effects from pollution (E2-6)

ESRS E2 Disclosure Requirements (6 DRs):
    E2-1: Policies related to pollution
    E2-2: Actions and resources related to pollution
    E2-3: Targets related to pollution
    E2-4: Pollution of air, water and soil
    E2-5: Substances of concern and substances of very high concern
    E2-6: Anticipated financial effects from pollution-related impacts

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

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


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
    """Phases of the E2 pollution workflow."""
    POLICY_REVIEW = "policy_review"
    EMISSIONS_CALCULATION = "emissions_calculation"
    SUBSTANCES_ASSESSMENT = "substances_assessment"
    TARGET_EVALUATION = "target_evaluation"
    FINANCIAL_EFFECTS = "financial_effects"


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


class PollutantMedium(str, Enum):
    """Pollution medium categories."""
    AIR = "air"
    WATER = "water"
    SOIL = "soil"


class SubstanceConcernLevel(str, Enum):
    """Substance concern classification."""
    STANDARD = "standard"
    CONCERN = "substance_of_concern"
    SVHC = "substance_of_very_high_concern"


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


class PollutantRecord(BaseModel):
    """Individual pollutant emission record per E2-4."""
    record_id: str = Field(default_factory=lambda: f"pol-{_new_uuid()[:8]}")
    pollutant_name: str = Field(default="", description="Name of pollutant")
    cas_number: str = Field(default="", description="CAS registry number if applicable")
    medium: PollutantMedium = Field(default=PollutantMedium.AIR)
    quantity_tonnes: float = Field(default=0.0, ge=0.0, description="Annual emission in tonnes")
    reporting_year: int = Field(default=2025)
    source_facility: str = Field(default="")
    measurement_method: str = Field(default="calculated", description="measured, calculated, estimated")


class SubstanceRecord(BaseModel):
    """Substance of concern record per E2-5."""
    substance_id: str = Field(default_factory=lambda: f"sub-{_new_uuid()[:8]}")
    substance_name: str = Field(default="")
    cas_number: str = Field(default="")
    concern_level: SubstanceConcernLevel = Field(default=SubstanceConcernLevel.CONCERN)
    quantity_tonnes: float = Field(default=0.0, ge=0.0)
    product_category: str = Field(default="")
    phase_out_plan: bool = Field(default=False)


class PollutionTarget(BaseModel):
    """Pollution reduction target per E2-3."""
    target_id: str = Field(default_factory=lambda: f"pt-{_new_uuid()[:8]}")
    target_name: str = Field(default="")
    pollutant: str = Field(default="")
    medium: PollutantMedium = Field(default=PollutantMedium.AIR)
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    current_progress_pct: float = Field(default=0.0, ge=0.0, le=200.0)
    on_track: bool = Field(default=False)


class E2PollutionInput(BaseModel):
    """Input data model for E2PollutionWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    e2_is_material: bool = Field(default=True, description="Whether E2 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="E2-1 pollution policies"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="E2-2 actions and resources"
    )
    pollutant_records: List[PollutantRecord] = Field(
        default_factory=list, description="E2-4 pollutant emissions"
    )
    substance_records: List[SubstanceRecord] = Field(
        default_factory=list, description="E2-5 substances of concern"
    )
    targets: List[PollutionTarget] = Field(
        default_factory=list, description="E2-3 pollution targets"
    )
    financial_effects_data: Dict[str, Any] = Field(
        default_factory=dict, description="E2-6 financial effects"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class E2PollutionWorkflowResult(BaseModel):
    """Complete result from E2 pollution workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="e2_pollution")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    e2_is_material: bool = Field(default=True)
    total_air_emissions_tonnes: float = Field(default=0.0)
    total_water_emissions_tonnes: float = Field(default=0.0)
    total_soil_emissions_tonnes: float = Field(default=0.0)
    substances_of_concern_count: int = Field(default=0)
    svhc_count: int = Field(default=0)
    targets_on_track: int = Field(default=0)
    targets_total: int = Field(default=0)
    policies_count: int = Field(default=0)
    has_financial_effects: bool = Field(default=False)
    overall_completeness_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class E2PollutionWorkflow:
    """
    5-phase ESRS E2 Pollution workflow.

    Orchestrates policy review, emissions calculation, substances assessment,
    target evaluation, and financial effects analysis for complete E2
    disclosure covering E2-1 through E2-6.

    Zero-hallucination: all pollutant aggregations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = E2PollutionWorkflow()
        >>> inp = E2PollutionInput(pollutant_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_air_emissions_tonnes >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize E2PollutionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review pollution policies and actions"},
            {"name": WorkflowPhase.EMISSIONS_CALCULATION.value, "description": "Calculate pollutant emissions"},
            {"name": WorkflowPhase.SUBSTANCES_ASSESSMENT.value, "description": "Assess substances of concern"},
            {"name": WorkflowPhase.TARGET_EVALUATION.value, "description": "Evaluate pollution targets"},
            {"name": WorkflowPhase.FINANCIAL_EFFECTS.value, "description": "Assess financial effects"},
        ]

    def validate_inputs(self, input_data: E2PollutionInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.e2_is_material:
            issues.append("E2 is not material; full disclosure not required")
        if not input_data.pollutant_records:
            issues.append("No pollutant emission records provided")
        for rec in input_data.pollutant_records:
            if rec.quantity_tonnes < 0:
                issues.append(f"Negative emission in record {rec.record_id}")
        return issues

    async def execute(
        self,
        input_data: Optional[E2PollutionInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> E2PollutionWorkflowResult:
        """
        Execute the 5-phase E2 pollution workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            E2PollutionWorkflowResult with pollutant totals and compliance status.
        """
        if input_data is None:
            input_data = E2PollutionInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting E2 pollution workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_emissions_calculation(input_data))
            phase_results.append(await self._phase_substances_assessment(input_data))
            phase_results.append(await self._phase_target_evaluation(input_data))
            phase_results.append(await self._phase_financial_effects(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("E2 pollution workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        air_total = sum(r.quantity_tonnes for r in input_data.pollutant_records if r.medium == PollutantMedium.AIR)
        water_total = sum(r.quantity_tonnes for r in input_data.pollutant_records if r.medium == PollutantMedium.WATER)
        soil_total = sum(r.quantity_tonnes for r in input_data.pollutant_records if r.medium == PollutantMedium.SOIL)
        soc_count = sum(1 for s in input_data.substance_records if s.concern_level == SubstanceConcernLevel.CONCERN)
        svhc_count = sum(1 for s in input_data.substance_records if s.concern_level == SubstanceConcernLevel.SVHC)
        on_track = sum(1 for t in input_data.targets if t.on_track)

        completeness = self._calculate_completeness(input_data)

        result = E2PollutionWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            e2_is_material=input_data.e2_is_material,
            total_air_emissions_tonnes=round(air_total, 4),
            total_water_emissions_tonnes=round(water_total, 4),
            total_soil_emissions_tonnes=round(soil_total, 4),
            substances_of_concern_count=soc_count,
            svhc_count=svhc_count,
            targets_on_track=on_track,
            targets_total=len(input_data.targets),
            policies_count=len(input_data.policies),
            has_financial_effects=bool(input_data.financial_effects_data),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "E2 pollution %s completed in %.2fs: air=%.2ft, water=%.2ft, soil=%.2ft",
            self.workflow_id, elapsed, air_total, water_total, soil_total,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review (E2-1, E2-2)
    # -------------------------------------------------------------------------

    async def _phase_policy_review(
        self, input_data: E2PollutionInput,
    ) -> PhaseResult:
        """Review pollution policies and actions."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["policies_count"] = len(input_data.policies)
        outputs["actions_count"] = len(input_data.actions)
        outputs["has_prevention_policy"] = any(
            p.get("type") == "prevention" for p in input_data.policies
        )
        outputs["has_remediation_actions"] = any(
            a.get("type") == "remediation" for a in input_data.actions
        )

        if not input_data.policies:
            warnings.append("No pollution policies defined (E2-1)")
        if not input_data.actions:
            warnings.append("No pollution actions defined (E2-2)")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies, %d actions",
                         len(input_data.policies), len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Emissions Calculation (E2-4)
    # -------------------------------------------------------------------------

    async def _phase_emissions_calculation(
        self, input_data: E2PollutionInput,
    ) -> PhaseResult:
        """Calculate pollutant emissions to air, water, and soil."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        records = input_data.pollutant_records
        by_medium: Dict[str, float] = {}
        by_pollutant: Dict[str, float] = {}

        for rec in records:
            by_medium[rec.medium.value] = by_medium.get(rec.medium.value, 0.0) + rec.quantity_tonnes
            by_pollutant[rec.pollutant_name] = by_pollutant.get(rec.pollutant_name, 0.0) + rec.quantity_tonnes

        outputs["records_count"] = len(records)
        outputs["emissions_by_medium"] = {k: round(v, 4) for k, v in by_medium.items()}
        outputs["emissions_by_pollutant"] = {k: round(v, 4) for k, v in by_pollutant.items()}
        outputs["total_emissions_tonnes"] = round(sum(by_medium.values()), 4)
        outputs["unique_pollutants"] = len(by_pollutant)

        if not records:
            warnings.append("No pollutant emission records provided (E2-4)")
        zero_records = [r for r in records if r.quantity_tonnes == 0.0]
        if zero_records:
            warnings.append(f"{len(zero_records)} records with zero emissions")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 2 EmissionsCalculation: %d records, %.2f tonnes total",
                         len(records), outputs["total_emissions_tonnes"])
        return PhaseResult(
            phase_name=WorkflowPhase.EMISSIONS_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Substances Assessment (E2-5)
    # -------------------------------------------------------------------------

    async def _phase_substances_assessment(
        self, input_data: E2PollutionInput,
    ) -> PhaseResult:
        """Assess substances of concern and SVHCs."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        substances = input_data.substance_records
        soc = [s for s in substances if s.concern_level == SubstanceConcernLevel.CONCERN]
        svhc = [s for s in substances if s.concern_level == SubstanceConcernLevel.SVHC]
        phase_out = [s for s in substances if s.phase_out_plan]

        outputs["total_substances"] = len(substances)
        outputs["substances_of_concern"] = len(soc)
        outputs["svhc_count"] = len(svhc)
        outputs["with_phase_out_plan"] = len(phase_out)
        outputs["total_quantity_tonnes"] = round(sum(s.quantity_tonnes for s in substances), 4)

        if svhc and not phase_out:
            warnings.append("SVHCs identified but no phase-out plans in place")
        if not substances:
            warnings.append("No substance records provided (E2-5)")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 3 SubstancesAssessment: %d total, %d SVHC",
                         len(substances), len(svhc))
        return PhaseResult(
            phase_name=WorkflowPhase.SUBSTANCES_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Evaluation (E2-3)
    # -------------------------------------------------------------------------

    async def _phase_target_evaluation(
        self, input_data: E2PollutionInput,
    ) -> PhaseResult:
        """Evaluate pollution reduction targets and progress."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets
        on_track = [t for t in targets if t.on_track]
        off_track = [t for t in targets if not t.on_track]

        outputs["targets_count"] = len(targets)
        outputs["on_track_count"] = len(on_track)
        outputs["off_track_count"] = len(off_track)
        outputs["avg_progress_pct"] = round(
            sum(t.current_progress_pct for t in targets) / len(targets)
            if targets else 0.0, 1
        )
        outputs["by_medium"] = {
            medium.value: sum(1 for t in targets if t.medium == medium)
            for medium in PollutantMedium
        }

        if not targets:
            warnings.append("No pollution reduction targets defined (E2-3)")
        if off_track:
            warnings.append(f"{len(off_track)} targets are off track")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TargetEvaluation: %d targets, %d on track",
                         len(targets), len(on_track))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_EVALUATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Financial Effects (E2-6)
    # -------------------------------------------------------------------------

    async def _phase_financial_effects(
        self, input_data: E2PollutionInput,
    ) -> PhaseResult:
        """Assess anticipated financial effects from pollution."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.financial_effects_data
        outputs["has_financial_effects"] = bool(data)
        outputs["remediation_costs_eur"] = data.get("remediation_costs_eur", 0)
        outputs["fines_provisions_eur"] = data.get("fines_provisions_eur", 0)
        outputs["insurance_coverage_eur"] = data.get("insurance_coverage_eur", 0)
        outputs["stranded_asset_risk"] = data.get("stranded_asset_risk", False)

        total_exposure = (
            data.get("remediation_costs_eur", 0)
            + data.get("fines_provisions_eur", 0)
        )
        outputs["total_financial_exposure_eur"] = total_exposure

        if not data:
            warnings.append("No financial effects data provided (E2-6)")
        if total_exposure > 0 and not data.get("insurance_coverage_eur"):
            warnings.append("Financial exposure exists but no insurance coverage reported")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 5 FinancialEffects: exposure=%d EUR", total_exposure)
        return PhaseResult(
            phase_name=WorkflowPhase.FINANCIAL_EFFECTS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_completeness(self, input_data: E2PollutionInput) -> float:
        """Calculate overall E2 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        scores.append(100.0 if input_data.pollutant_records else 0.0)
        scores.append(100.0 if input_data.substance_records else 0.0)
        scores.append(100.0 if input_data.financial_effects_data else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: E2PollutionWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
