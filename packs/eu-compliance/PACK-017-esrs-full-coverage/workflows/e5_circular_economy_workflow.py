# -*- coding: utf-8 -*-
"""
ESRS E5 Resource Use and Circular Economy Workflow
====================================================

5-phase workflow for ESRS E5 Resource Use and Circular Economy disclosure
covering policy review, resource inflows analysis, resource outflows analysis,
target evaluation, and financial effects with full provenance tracking.

Phases:
    1. PolicyReview        -- Review circular economy policies and actions (E5-1, E5-2)
    2. ResourceInflows     -- Analyse material inflows and circularity (E5-4)
    3. ResourceOutflows    -- Analyse waste and secondary materials outflows (E5-5)
    4. TargetEvaluation    -- Evaluate resource use and circularity targets (E5-3)
    5. FinancialEffects    -- Assess financial effects from resource use (E5-6)

ESRS E5 Disclosure Requirements (6 DRs):
    E5-1: Policies related to resource use and circular economy
    E5-2: Actions and resources related to resource use and circular economy
    E5-3: Targets related to resource use and circular economy
    E5-4: Resource inflows
    E5-5: Resource outflows
    E5-6: Anticipated financial effects from resource use and circular economy

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
    """Phases of the E5 circular economy workflow."""
    POLICY_REVIEW = "policy_review"
    RESOURCE_INFLOWS = "resource_inflows"
    RESOURCE_OUTFLOWS = "resource_outflows"
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

class MaterialOrigin(str, Enum):
    """Material origin classification for inflows."""
    VIRGIN = "virgin"
    RECYCLED = "recycled"
    RENEWABLE = "renewable"
    SECONDARY = "secondary"

class WasteCategory(str, Enum):
    """Waste category classification for outflows."""
    HAZARDOUS = "hazardous"
    NON_HAZARDOUS = "non_hazardous"
    RADIOACTIVE = "radioactive"

class WasteDisposal(str, Enum):
    """Waste disposal method."""
    RECYCLED = "recycled"
    REUSED = "reused"
    COMPOSTED = "composted"
    INCINERATED_ENERGY = "incinerated_with_energy_recovery"
    INCINERATED_NO_ENERGY = "incinerated_without_energy_recovery"
    LANDFILL = "landfill"
    OTHER = "other"

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

class InflowRecord(BaseModel):
    """Resource inflow record per E5-4."""
    record_id: str = Field(default_factory=lambda: f"in-{_new_uuid()[:8]}")
    material_name: str = Field(default="")
    origin: MaterialOrigin = Field(default=MaterialOrigin.VIRGIN)
    weight_tonnes: float = Field(default=0.0, ge=0.0)
    is_renewable: bool = Field(default=False)
    reporting_year: int = Field(default=2025)

class OutflowRecord(BaseModel):
    """Resource outflow / waste record per E5-5."""
    record_id: str = Field(default_factory=lambda: f"out-{_new_uuid()[:8]}")
    waste_name: str = Field(default="")
    category: WasteCategory = Field(default=WasteCategory.NON_HAZARDOUS)
    disposal_method: WasteDisposal = Field(default=WasteDisposal.LANDFILL)
    weight_tonnes: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025)

class CircularTarget(BaseModel):
    """Circular economy target per E5-3."""
    target_id: str = Field(default_factory=lambda: f"ct-{_new_uuid()[:8]}")
    target_name: str = Field(default="")
    metric: str = Field(default="", description="recycled_content_pct, waste_reduction, etc.")
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    target_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)
    on_track: bool = Field(default=False)

class E5CircularInput(BaseModel):
    """Input data model for E5CircularWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    e5_is_material: bool = Field(default=True, description="Whether E5 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="E5-1 circular economy policies"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="E5-2 actions and resources"
    )
    inflow_records: List[InflowRecord] = Field(
        default_factory=list, description="E5-4 resource inflows"
    )
    outflow_records: List[OutflowRecord] = Field(
        default_factory=list, description="E5-5 resource outflows"
    )
    targets: List[CircularTarget] = Field(
        default_factory=list, description="E5-3 circularity targets"
    )
    financial_effects_data: Dict[str, Any] = Field(
        default_factory=dict, description="E5-6 financial effects"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class E5CircularWorkflowResult(BaseModel):
    """Complete result from E5 circular economy workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="e5_circular_economy")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    e5_is_material: bool = Field(default=True)
    total_inflow_tonnes: float = Field(default=0.0)
    recycled_inflow_tonnes: float = Field(default=0.0)
    recycled_content_pct: float = Field(default=0.0)
    total_outflow_tonnes: float = Field(default=0.0)
    waste_recycled_tonnes: float = Field(default=0.0)
    waste_landfill_tonnes: float = Field(default=0.0)
    hazardous_waste_tonnes: float = Field(default=0.0)
    circularity_rate_pct: float = Field(default=0.0)
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

class E5CircularWorkflow:
    """
    5-phase ESRS E5 Resource Use and Circular Economy workflow.

    Orchestrates policy review, resource inflows analysis, outflows analysis,
    target evaluation, and financial effects for complete E5 disclosure
    covering E5-1 through E5-6.

    Zero-hallucination: all material flow and circularity calculations use
    deterministic arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = E5CircularWorkflow()
        >>> inp = E5CircularInput(inflow_records=[...], outflow_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.circularity_rate_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize E5CircularWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review circular economy policies"},
            {"name": WorkflowPhase.RESOURCE_INFLOWS.value, "description": "Analyse resource inflows"},
            {"name": WorkflowPhase.RESOURCE_OUTFLOWS.value, "description": "Analyse resource outflows"},
            {"name": WorkflowPhase.TARGET_EVALUATION.value, "description": "Evaluate circularity targets"},
            {"name": WorkflowPhase.FINANCIAL_EFFECTS.value, "description": "Assess financial effects"},
        ]

    def validate_inputs(self, input_data: E5CircularInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.e5_is_material:
            issues.append("E5 is not material; full disclosure not required")
        if not input_data.inflow_records and not input_data.outflow_records:
            issues.append("No resource flow records provided")
        return issues

    async def execute(
        self,
        input_data: Optional[E5CircularInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> E5CircularWorkflowResult:
        """
        Execute the 5-phase E5 circular economy workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            E5CircularWorkflowResult with circularity metrics and compliance status.
        """
        if input_data is None:
            input_data = E5CircularInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting E5 circular economy workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_resource_inflows(input_data))
            phase_results.append(await self._phase_resource_outflows(input_data))
            phase_results.append(await self._phase_target_evaluation(input_data))
            phase_results.append(await self._phase_financial_effects(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("E5 circular economy workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        total_inflow = sum(r.weight_tonnes for r in input_data.inflow_records)
        recycled_inflow = sum(
            r.weight_tonnes for r in input_data.inflow_records
            if r.origin in (MaterialOrigin.RECYCLED, MaterialOrigin.SECONDARY)
        )
        recycled_pct = round((recycled_inflow / total_inflow * 100) if total_inflow > 0 else 0.0, 1)

        total_outflow = sum(r.weight_tonnes for r in input_data.outflow_records)
        recycled_outflow = sum(
            r.weight_tonnes for r in input_data.outflow_records
            if r.disposal_method == WasteDisposal.RECYCLED
        )
        landfill_outflow = sum(
            r.weight_tonnes for r in input_data.outflow_records
            if r.disposal_method == WasteDisposal.LANDFILL
        )
        hazardous = sum(
            r.weight_tonnes for r in input_data.outflow_records
            if r.category == WasteCategory.HAZARDOUS
        )
        circularity = round(
            ((recycled_inflow + recycled_outflow) / (total_inflow + total_outflow) * 100)
            if (total_inflow + total_outflow) > 0 else 0.0, 1
        )
        on_track = sum(1 for t in input_data.targets if t.on_track)
        completeness = self._calculate_completeness(input_data)

        result = E5CircularWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            e5_is_material=input_data.e5_is_material,
            total_inflow_tonnes=round(total_inflow, 2),
            recycled_inflow_tonnes=round(recycled_inflow, 2),
            recycled_content_pct=recycled_pct,
            total_outflow_tonnes=round(total_outflow, 2),
            waste_recycled_tonnes=round(recycled_outflow, 2),
            waste_landfill_tonnes=round(landfill_outflow, 2),
            hazardous_waste_tonnes=round(hazardous, 2),
            circularity_rate_pct=circularity,
            targets_on_track=on_track,
            targets_total=len(input_data.targets),
            policies_count=len(input_data.policies),
            has_financial_effects=bool(input_data.financial_effects_data),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "E5 circular %s completed in %.2fs: inflow=%.0ft, outflow=%.0ft, circularity=%.1f%%",
            self.workflow_id, elapsed, total_inflow, total_outflow, circularity,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review (E5-1, E5-2)
    # -------------------------------------------------------------------------

    async def _phase_policy_review(self, input_data: E5CircularInput) -> PhaseResult:
        """Review circular economy policies and actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["policies_count"] = len(input_data.policies)
        outputs["actions_count"] = len(input_data.actions)
        outputs["has_waste_prevention_policy"] = any(
            p.get("type") == "waste_prevention" for p in input_data.policies
        )
        outputs["has_eco_design_policy"] = any(
            p.get("type") == "eco_design" for p in input_data.policies
        )

        if not input_data.policies:
            warnings.append("No circular economy policies defined (E5-1)")
        if not input_data.actions:
            warnings.append("No circular economy actions defined (E5-2)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies, %d actions",
                         len(input_data.policies), len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Resource Inflows (E5-4)
    # -------------------------------------------------------------------------

    async def _phase_resource_inflows(self, input_data: E5CircularInput) -> PhaseResult:
        """Analyse resource inflows and circularity of inputs."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        records = input_data.inflow_records
        total = sum(r.weight_tonnes for r in records)
        by_origin: Dict[str, float] = {}
        for r in records:
            by_origin[r.origin.value] = by_origin.get(r.origin.value, 0.0) + r.weight_tonnes

        recycled = by_origin.get(MaterialOrigin.RECYCLED.value, 0.0) + by_origin.get(MaterialOrigin.SECONDARY.value, 0.0)
        renewable = sum(r.weight_tonnes for r in records if r.is_renewable)

        outputs["records_count"] = len(records)
        outputs["total_inflow_tonnes"] = round(total, 2)
        outputs["by_origin"] = {k: round(v, 2) for k, v in by_origin.items()}
        outputs["recycled_content_pct"] = round((recycled / total * 100) if total > 0 else 0.0, 1)
        outputs["renewable_content_pct"] = round((renewable / total * 100) if total > 0 else 0.0, 1)

        if not records:
            warnings.append("No resource inflow records provided (E5-4)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 ResourceInflows: %.0f tonnes, %.1f%% recycled",
                         total, outputs["recycled_content_pct"])
        return PhaseResult(
            phase_name=WorkflowPhase.RESOURCE_INFLOWS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Resource Outflows (E5-5)
    # -------------------------------------------------------------------------

    async def _phase_resource_outflows(self, input_data: E5CircularInput) -> PhaseResult:
        """Analyse resource outflows and waste management."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        records = input_data.outflow_records
        total = sum(r.weight_tonnes for r in records)
        by_disposal: Dict[str, float] = {}
        by_category: Dict[str, float] = {}
        for r in records:
            by_disposal[r.disposal_method.value] = by_disposal.get(r.disposal_method.value, 0.0) + r.weight_tonnes
            by_category[r.category.value] = by_category.get(r.category.value, 0.0) + r.weight_tonnes

        diverted = sum(
            r.weight_tonnes for r in records
            if r.disposal_method in (WasteDisposal.RECYCLED, WasteDisposal.REUSED, WasteDisposal.COMPOSTED)
        )

        outputs["records_count"] = len(records)
        outputs["total_outflow_tonnes"] = round(total, 2)
        outputs["by_disposal_method"] = {k: round(v, 2) for k, v in by_disposal.items()}
        outputs["by_category"] = {k: round(v, 2) for k, v in by_category.items()}
        outputs["diversion_rate_pct"] = round((diverted / total * 100) if total > 0 else 0.0, 1)
        outputs["hazardous_pct"] = round(
            (by_category.get(WasteCategory.HAZARDOUS.value, 0.0) / total * 100) if total > 0 else 0.0, 1
        )

        if not records:
            warnings.append("No resource outflow records provided (E5-5)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ResourceOutflows: %.0f tonnes, %.1f%% diverted",
                         total, outputs["diversion_rate_pct"])
        return PhaseResult(
            phase_name=WorkflowPhase.RESOURCE_OUTFLOWS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Evaluation (E5-3)
    # -------------------------------------------------------------------------

    async def _phase_target_evaluation(self, input_data: E5CircularInput) -> PhaseResult:
        """Evaluate circular economy targets and progress."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets
        on_track = [t for t in targets if t.on_track]

        outputs["targets_count"] = len(targets)
        outputs["on_track_count"] = len(on_track)
        outputs["off_track_count"] = len(targets) - len(on_track)
        outputs["metrics_tracked"] = list(set(t.metric for t in targets if t.metric))

        if not targets:
            warnings.append("No circular economy targets defined (E5-3)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TargetEvaluation: %d targets, %d on track",
                         len(targets), len(on_track))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_EVALUATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Financial Effects (E5-6)
    # -------------------------------------------------------------------------

    async def _phase_financial_effects(self, input_data: E5CircularInput) -> PhaseResult:
        """Assess anticipated financial effects from resource use."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.financial_effects_data
        outputs["has_financial_effects"] = bool(data)
        outputs["waste_management_costs_eur"] = data.get("waste_management_costs_eur", 0)
        outputs["material_cost_savings_eur"] = data.get("material_cost_savings_eur", 0)
        outputs["circular_revenue_eur"] = data.get("circular_revenue_eur", 0)
        outputs["epr_costs_eur"] = data.get("epr_costs_eur", 0)

        if not data:
            warnings.append("No financial effects data provided (E5-6)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 FinancialEffects: has_data=%s", bool(data))
        return PhaseResult(
            phase_name=WorkflowPhase.FINANCIAL_EFFECTS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_completeness(self, input_data: E5CircularInput) -> float:
        """Calculate overall E5 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        scores.append(100.0 if input_data.inflow_records else 0.0)
        scores.append(100.0 if input_data.outflow_records else 0.0)
        scores.append(100.0 if input_data.financial_effects_data else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: E5CircularWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
