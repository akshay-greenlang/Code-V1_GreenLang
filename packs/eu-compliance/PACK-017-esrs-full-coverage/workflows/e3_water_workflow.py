# -*- coding: utf-8 -*-
"""
ESRS E3 Water and Marine Resources Workflow
=============================================

5-phase workflow for ESRS E3 Water and Marine Resources disclosure covering
policy review, water balance calculation, water stress assessment, target
evaluation, and financial effects analysis with full provenance tracking.

Phases:
    1. PolicyReview        -- Review water/marine policies and actions (E3-1, E3-2)
    2. WaterBalance        -- Calculate water consumption, withdrawal, discharge (E3-4)
    3. StressAssessment    -- Assess operations in water-stressed areas (E3-3)
    4. TargetEvaluation    -- Evaluate water reduction targets (E3-3)
    5. FinancialEffects    -- Assess financial effects from water risks (E3-5)

ESRS E3 Disclosure Requirements (5 DRs):
    E3-1: Policies related to water and marine resources
    E3-2: Actions and resources related to water and marine resources
    E3-3: Targets related to water and marine resources
    E3-4: Water consumption
    E3-5: Anticipated financial effects from water and marine resources

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
    """Phases of the E3 water workflow."""
    POLICY_REVIEW = "policy_review"
    WATER_BALANCE = "water_balance"
    STRESS_ASSESSMENT = "stress_assessment"
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

class WaterSourceType(str, Enum):
    """Water source classification."""
    SURFACE = "surface_water"
    GROUNDWATER = "groundwater"
    SEAWATER = "seawater"
    PRODUCED = "produced_water"
    THIRD_PARTY = "third_party_water"
    RAINWATER = "rainwater"

class WaterStressLevel(str, Enum):
    """WRI Aqueduct water stress classification."""
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"

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

class WaterRecord(BaseModel):
    """Water withdrawal, discharge, or consumption record per E3-4."""
    record_id: str = Field(default_factory=lambda: f"wr-{_new_uuid()[:8]}")
    flow_type: str = Field(default="withdrawal", description="withdrawal, discharge, or consumption")
    source_type: WaterSourceType = Field(default=WaterSourceType.SURFACE)
    volume_m3: float = Field(default=0.0, ge=0.0, description="Volume in cubic metres")
    facility_name: str = Field(default="")
    basin_name: str = Field(default="")
    stress_level: WaterStressLevel = Field(default=WaterStressLevel.LOW)
    is_freshwater: bool = Field(default=True, description="TDS <= 1000 mg/L")
    reporting_year: int = Field(default=2025)

class WaterTarget(BaseModel):
    """Water target per E3-3."""
    target_id: str = Field(default_factory=lambda: f"wt-{_new_uuid()[:8]}")
    target_name: str = Field(default="")
    metric: str = Field(default="water_consumption", description="Target metric")
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    current_progress_pct: float = Field(default=0.0, ge=0.0, le=200.0)
    on_track: bool = Field(default=False)
    applies_to_stressed_areas: bool = Field(default=False)

class E3WaterInput(BaseModel):
    """Input data model for E3WaterWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    e3_is_material: bool = Field(default=True, description="Whether E3 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="E3-1 water policies"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="E3-2 actions and resources"
    )
    water_records: List[WaterRecord] = Field(
        default_factory=list, description="E3-4 water flow records"
    )
    targets: List[WaterTarget] = Field(
        default_factory=list, description="E3-3 water targets"
    )
    marine_resources_data: Dict[str, Any] = Field(
        default_factory=dict, description="Marine resources impact data"
    )
    financial_effects_data: Dict[str, Any] = Field(
        default_factory=dict, description="E3-5 financial effects"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class E3WaterWorkflowResult(BaseModel):
    """Complete result from E3 water workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="e3_water")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    e3_is_material: bool = Field(default=True)
    total_withdrawal_m3: float = Field(default=0.0)
    total_discharge_m3: float = Field(default=0.0)
    total_consumption_m3: float = Field(default=0.0)
    withdrawal_stressed_areas_m3: float = Field(default=0.0)
    facilities_in_stressed_areas: int = Field(default=0)
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

class E3WaterWorkflow:
    """
    5-phase ESRS E3 Water and Marine Resources workflow.

    Orchestrates policy review, water balance calculation, stress assessment,
    target evaluation, and financial effects for complete E3 disclosure
    covering E3-1 through E3-5.

    Zero-hallucination: all water balance calculations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = E3WaterWorkflow()
        >>> inp = E3WaterInput(water_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_consumption_m3 >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize E3WaterWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review water/marine policies"},
            {"name": WorkflowPhase.WATER_BALANCE.value, "description": "Calculate water balance"},
            {"name": WorkflowPhase.STRESS_ASSESSMENT.value, "description": "Assess water stress areas"},
            {"name": WorkflowPhase.TARGET_EVALUATION.value, "description": "Evaluate water targets"},
            {"name": WorkflowPhase.FINANCIAL_EFFECTS.value, "description": "Assess financial effects"},
        ]

    def validate_inputs(self, input_data: E3WaterInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.e3_is_material:
            issues.append("E3 is not material; full disclosure not required")
        if not input_data.water_records:
            issues.append("No water records provided")
        return issues

    async def execute(
        self,
        input_data: Optional[E3WaterInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> E3WaterWorkflowResult:
        """
        Execute the 5-phase E3 water workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            E3WaterWorkflowResult with water balance and stress assessment.
        """
        if input_data is None:
            input_data = E3WaterInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting E3 water workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_water_balance(input_data))
            phase_results.append(await self._phase_stress_assessment(input_data))
            phase_results.append(await self._phase_target_evaluation(input_data))
            phase_results.append(await self._phase_financial_effects(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("E3 water workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        withdrawal = sum(r.volume_m3 for r in input_data.water_records if r.flow_type == "withdrawal")
        discharge = sum(r.volume_m3 for r in input_data.water_records if r.flow_type == "discharge")
        consumption = sum(r.volume_m3 for r in input_data.water_records if r.flow_type == "consumption")
        stressed_withdrawal = sum(
            r.volume_m3 for r in input_data.water_records
            if r.flow_type == "withdrawal" and r.stress_level in (
                WaterStressLevel.HIGH, WaterStressLevel.EXTREMELY_HIGH
            )
        )
        stressed_facilities = len(set(
            r.facility_name for r in input_data.water_records
            if r.stress_level in (WaterStressLevel.HIGH, WaterStressLevel.EXTREMELY_HIGH)
            and r.facility_name
        ))
        on_track = sum(1 for t in input_data.targets if t.on_track)
        completeness = self._calculate_completeness(input_data)

        result = E3WaterWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            e3_is_material=input_data.e3_is_material,
            total_withdrawal_m3=round(withdrawal, 2),
            total_discharge_m3=round(discharge, 2),
            total_consumption_m3=round(consumption, 2),
            withdrawal_stressed_areas_m3=round(stressed_withdrawal, 2),
            facilities_in_stressed_areas=stressed_facilities,
            targets_on_track=on_track,
            targets_total=len(input_data.targets),
            policies_count=len(input_data.policies),
            has_financial_effects=bool(input_data.financial_effects_data),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "E3 water %s completed in %.2fs: withdrawal=%.0fm3, consumption=%.0fm3",
            self.workflow_id, elapsed, withdrawal, consumption,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review (E3-1, E3-2)
    # -------------------------------------------------------------------------

    async def _phase_policy_review(self, input_data: E3WaterInput) -> PhaseResult:
        """Review water and marine resource policies and actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["policies_count"] = len(input_data.policies)
        outputs["actions_count"] = len(input_data.actions)
        outputs["has_water_policy"] = any(p.get("scope") == "water" for p in input_data.policies)
        outputs["has_marine_policy"] = any(p.get("scope") == "marine" for p in input_data.policies)

        if not input_data.policies:
            warnings.append("No water/marine policies defined (E3-1)")
        if not input_data.actions:
            warnings.append("No water/marine actions defined (E3-2)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies, %d actions",
                         len(input_data.policies), len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Water Balance (E3-4)
    # -------------------------------------------------------------------------

    async def _phase_water_balance(self, input_data: E3WaterInput) -> PhaseResult:
        """Calculate water withdrawal, discharge, and consumption balance."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        records = input_data.water_records
        withdrawal = sum(r.volume_m3 for r in records if r.flow_type == "withdrawal")
        discharge = sum(r.volume_m3 for r in records if r.flow_type == "discharge")
        consumption = sum(r.volume_m3 for r in records if r.flow_type == "consumption")

        # Breakdown by source type
        by_source: Dict[str, float] = {}
        for r in records:
            if r.flow_type == "withdrawal":
                by_source[r.source_type.value] = by_source.get(r.source_type.value, 0.0) + r.volume_m3

        freshwater = sum(r.volume_m3 for r in records if r.is_freshwater and r.flow_type == "withdrawal")

        outputs["total_withdrawal_m3"] = round(withdrawal, 2)
        outputs["total_discharge_m3"] = round(discharge, 2)
        outputs["total_consumption_m3"] = round(consumption, 2)
        outputs["calculated_consumption_m3"] = round(withdrawal - discharge, 2)
        outputs["withdrawal_by_source"] = {k: round(v, 2) for k, v in by_source.items()}
        outputs["freshwater_withdrawal_m3"] = round(freshwater, 2)
        outputs["records_count"] = len(records)

        if not records:
            warnings.append("No water records provided (E3-4)")
        if consumption == 0 and withdrawal > 0:
            warnings.append("No direct consumption records; using withdrawal minus discharge")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 WaterBalance: withdrawal=%.0f, discharge=%.0f, consumption=%.0f m3",
                         withdrawal, discharge, consumption)
        return PhaseResult(
            phase_name=WorkflowPhase.WATER_BALANCE.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Stress Assessment
    # -------------------------------------------------------------------------

    async def _phase_stress_assessment(self, input_data: E3WaterInput) -> PhaseResult:
        """Assess operations in water-stressed areas."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        records = input_data.water_records
        high_stress = [
            r for r in records
            if r.stress_level in (WaterStressLevel.HIGH, WaterStressLevel.EXTREMELY_HIGH)
        ]
        stressed_facilities = set(r.facility_name for r in high_stress if r.facility_name)
        stressed_basins = set(r.basin_name for r in high_stress if r.basin_name)

        outputs["high_stress_records"] = len(high_stress)
        outputs["facilities_in_stressed_areas"] = len(stressed_facilities)
        outputs["stressed_basins"] = list(stressed_basins)
        outputs["stressed_withdrawal_m3"] = round(
            sum(r.volume_m3 for r in high_stress if r.flow_type == "withdrawal"), 2
        )
        outputs["stress_distribution"] = {
            level.value: sum(1 for r in records if r.stress_level == level)
            for level in WaterStressLevel
        }

        if high_stress:
            warnings.append(
                f"{len(stressed_facilities)} facilities in high/extremely-high water stress areas"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 StressAssessment: %d facilities in stressed areas",
                         len(stressed_facilities))
        return PhaseResult(
            phase_name=WorkflowPhase.STRESS_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Evaluation (E3-3)
    # -------------------------------------------------------------------------

    async def _phase_target_evaluation(self, input_data: E3WaterInput) -> PhaseResult:
        """Evaluate water reduction targets and progress."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets
        on_track = [t for t in targets if t.on_track]

        outputs["targets_count"] = len(targets)
        outputs["on_track_count"] = len(on_track)
        outputs["off_track_count"] = len(targets) - len(on_track)
        outputs["avg_progress_pct"] = round(
            sum(t.current_progress_pct for t in targets) / len(targets)
            if targets else 0.0, 1
        )
        outputs["stressed_area_targets"] = sum(1 for t in targets if t.applies_to_stressed_areas)

        if not targets:
            warnings.append("No water targets defined (E3-3)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TargetEvaluation: %d targets, %d on track",
                         len(targets), len(on_track))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_EVALUATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Financial Effects (E3-5)
    # -------------------------------------------------------------------------

    async def _phase_financial_effects(self, input_data: E3WaterInput) -> PhaseResult:
        """Assess anticipated financial effects from water/marine risks."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.financial_effects_data
        outputs["has_financial_effects"] = bool(data)
        outputs["physical_risk_eur"] = data.get("physical_risk_eur", 0)
        outputs["regulatory_risk_eur"] = data.get("regulatory_risk_eur", 0)
        outputs["opportunity_eur"] = data.get("opportunity_eur", 0)
        outputs["total_exposure_eur"] = (
            data.get("physical_risk_eur", 0) + data.get("regulatory_risk_eur", 0)
        )

        if not data:
            warnings.append("No financial effects data provided (E3-5)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 FinancialEffects: exposure=%d EUR", outputs["total_exposure_eur"])
        return PhaseResult(
            phase_name=WorkflowPhase.FINANCIAL_EFFECTS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_completeness(self, input_data: E3WaterInput) -> float:
        """Calculate overall E3 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        scores.append(100.0 if input_data.water_records else 0.0)
        scores.append(100.0 if input_data.financial_effects_data else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: E3WaterWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
