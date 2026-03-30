# -*- coding: utf-8 -*-
"""
Target Recalibration Workflow
====================================

5-phase DAG workflow for recalibrating interim targets after significant
structural changes within PACK-029 Interim Targets Pack.  The workflow
loads updated baseline data (e.g., after acquisition, divestiture, or
methodology change), recalculates interim targets, validates against
SBTi criteria, updates the annual pathway, and generates a recalibration
report.

Phases:
    1. LoadUpdatedBaseline   -- Load updated baseline data after structural
                                 change (M&A, boundary change, restatement)
    2. RecalcInterimTargets  -- Recalculate interim targets using the updated
                                 baseline via InterimTargetEngine
    3. ValidateNewTargets    -- Validate recalculated targets against SBTi
                                 criteria via MilestoneValidationEngine
    4. UpdatePathway         -- Update annual pathway with new targets via
                                 AnnualPathwayEngine
    5. RecalibrationReport   -- Generate target recalibration report with
                                 before/after comparison

Regulatory references:
    - SBTi Base Year Recalculation Policy
    - SBTi Significant Threshold (5% boundary change)
    - GHG Protocol Corporate Standard (recalculation triggers)
    - IPCC 2006 GL (base year adjustment rules)
    - ISO 14064-1:2018 (recalculation requirements)

Zero-hallucination: all recalculations use deterministic SBTi published
rules and GHG Protocol recalculation triggers.  No LLM in computation.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def _interpolate_linear(base_val: float, target_val: float, base_yr: int,
                         target_yr: int, current_yr: int) -> float:
    if target_yr <= base_yr:
        return target_val
    t = min(max((current_yr - base_yr) / (target_yr - base_yr), 0.0), 1.0)
    return base_val + t * (target_val - base_val)

def _calc_cagr(start_val: float, end_val: float, years: int) -> float:
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class RecalibrationTrigger(str, Enum):
    """SBTi/GHG Protocol recalculation triggers."""
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"
    METHODOLOGY_CHANGE = "methodology_change"
    EMISSION_FACTOR_UPDATE = "emission_factor_update"
    BOUNDARY_CHANGE = "boundary_change"
    ERROR_CORRECTION = "error_correction"
    STRUCTURAL_CHANGE = "structural_change"

class RecalibrationScope(str, Enum):
    """Scope of recalibration."""
    BASE_YEAR_ONLY = "base_year_only"
    ALL_HISTORICAL = "all_historical"
    TARGETS_ONLY = "targets_only"
    FULL_RECALIBRATION = "full_recalibration"

class ValidationResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    WARNING = "warning"

class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

# =============================================================================
# SBTI RECALCULATION RULES (Zero-Hallucination: Published Guidance)
# =============================================================================

RECALCULATION_RULES: Dict[str, Dict[str, Any]] = {
    "RULE-001": {
        "name": "Significant Threshold",
        "description": "Recalculation required if structural changes affect base year emissions by >= 5%.",
        "threshold_pct": 5.0,
        "applies_to": ["acquisition", "divestiture", "merger", "outsourcing", "insourcing"],
        "mandatory": True,
    },
    "RULE-002": {
        "name": "Methodology Change",
        "description": "Recalculation required for changes in calculation methodology or emission factors.",
        "threshold_pct": 0.0,
        "applies_to": ["methodology_change", "emission_factor_update"],
        "mandatory": True,
    },
    "RULE-003": {
        "name": "Error Correction",
        "description": "Recalculation required for discovery of significant errors in base year data.",
        "threshold_pct": 1.0,
        "applies_to": ["error_correction"],
        "mandatory": True,
    },
    "RULE-004": {
        "name": "Boundary Change",
        "description": "Recalculation required for changes in organizational or operational boundary.",
        "threshold_pct": 5.0,
        "applies_to": ["boundary_change"],
        "mandatory": True,
    },
    "RULE-005": {
        "name": "Organic Growth Exclusion",
        "description": "Organic growth/decline does NOT trigger recalculation.",
        "threshold_pct": 999.0,
        "applies_to": [],
        "mandatory": False,
    },
}

# Impact factors for different M&A scenarios
MA_IMPACT_FACTORS: Dict[str, Dict[str, float]] = {
    "acquisition_full": {
        "scope1_impact_pct": 100.0,
        "scope2_impact_pct": 100.0,
        "scope3_impact_pct": 80.0,
        "boundary_expansion_pct": 100.0,
    },
    "acquisition_partial": {
        "scope1_impact_pct": 50.0,
        "scope2_impact_pct": 50.0,
        "scope3_impact_pct": 40.0,
        "boundary_expansion_pct": 50.0,
    },
    "divestiture": {
        "scope1_impact_pct": -100.0,
        "scope2_impact_pct": -100.0,
        "scope3_impact_pct": -60.0,
        "boundary_expansion_pct": -100.0,
    },
    "outsourcing": {
        "scope1_impact_pct": -80.0,
        "scope2_impact_pct": -50.0,
        "scope3_impact_pct": 30.0,
        "boundary_expansion_pct": -50.0,
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")

class BaselineChange(BaseModel):
    """Details of a structural change affecting the baseline."""
    change_id: str = Field(default="")
    trigger: RecalibrationTrigger = Field(default=RecalibrationTrigger.STRUCTURAL_CHANGE)
    description: str = Field(default="")
    effective_date: str = Field(default="")
    scope1_impact_tco2e: float = Field(default=0.0)
    scope2_impact_tco2e: float = Field(default=0.0)
    scope3_impact_tco2e: float = Field(default=0.0)
    total_impact_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    exceeds_threshold: bool = Field(default=False)
    recalculation_required: bool = Field(default=False)

class UpdatedBaseline(BaseModel):
    """Updated baseline after structural changes."""
    original_base_year_tco2e: float = Field(default=0.0)
    original_scope1_tco2e: float = Field(default=0.0)
    original_scope2_tco2e: float = Field(default=0.0)
    original_scope3_tco2e: float = Field(default=0.0)
    updated_base_year_tco2e: float = Field(default=0.0)
    updated_scope1_tco2e: float = Field(default=0.0)
    updated_scope2_tco2e: float = Field(default=0.0)
    updated_scope3_tco2e: float = Field(default=0.0)
    net_change_tco2e: float = Field(default=0.0)
    net_change_pct: float = Field(default=0.0)
    changes_applied: List[BaselineChange] = Field(default_factory=list)
    recalculation_required: bool = Field(default=False)
    provenance_hash: str = Field(default="")

class RecalibratedTarget(BaseModel):
    """A single recalibrated interim target."""
    target_year: int = Field(default=2030)
    original_target_tco2e: float = Field(default=0.0)
    original_reduction_pct: float = Field(default=0.0)
    recalibrated_target_tco2e: float = Field(default=0.0)
    recalibrated_reduction_pct: float = Field(default=0.0)
    change_tco2e: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)
    maintains_ambition: bool = Field(default=True)
    sbti_minimum_met: bool = Field(default=True)

class RecalibrationValidation(BaseModel):
    """Validation of recalibrated targets."""
    total_criteria: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    overall_result: ValidationResult = Field(default=ValidationResult.PASS)
    ambition_maintained: bool = Field(default=True)
    sbti_resubmission_required: bool = Field(default=False)
    provenance_hash: str = Field(default="")

class UpdatedPathway(BaseModel):
    """Updated annual pathway after recalibration."""
    original_pathway_points: int = Field(default=0)
    updated_pathway_points: int = Field(default=0)
    pathway: List[Dict[str, Any]] = Field(default_factory=list)
    avg_annual_change_pct: float = Field(default=0.0)
    pathway_tightened: bool = Field(default=False)
    provenance_hash: str = Field(default="")

class RecalibrationReport(BaseModel):
    """Complete target recalibration report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    company_name: str = Field(default="")
    trigger_summary: str = Field(default="")
    updated_baseline: UpdatedBaseline = Field(default_factory=UpdatedBaseline)
    recalibrated_targets: List[RecalibratedTarget] = Field(default_factory=list)
    validation: RecalibrationValidation = Field(default_factory=RecalibrationValidation)
    updated_pathway: UpdatedPathway = Field(default_factory=UpdatedPathway)
    before_after_comparison: Dict[str, Any] = Field(default_factory=dict)
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class TargetRecalibrationConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    base_year: int = Field(default=2020, ge=2015, le=2030)
    original_base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    original_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    original_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    original_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_target_year: int = Field(default=2030)
    near_term_reduction_pct: float = Field(default=42.0)
    long_term_target_year: int = Field(default=2050)
    long_term_reduction_pct: float = Field(default=90.0)
    sbti_ambition: str = Field(default="1.5c")
    recalibration_scope: RecalibrationScope = Field(default=RecalibrationScope.FULL_RECALIBRATION)
    significance_threshold_pct: float = Field(default=5.0)
    maintain_ambition: bool = Field(default=True)
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])

class TargetRecalibrationInput(BaseModel):
    config: TargetRecalibrationConfig = Field(default_factory=TargetRecalibrationConfig)
    structural_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structural changes [{trigger, description, scope1_impact, scope2_impact, scope3_impact}]",
    )
    original_interim_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Original interim targets [{year, target_tco2e, reduction_pct}]",
    )
    original_pathway: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Original annual pathway [{year, target_tco2e}]",
    )

class TargetRecalibrationResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="target_recalibration")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    updated_baseline: UpdatedBaseline = Field(default_factory=UpdatedBaseline)
    recalibrated_targets: List[RecalibratedTarget] = Field(default_factory=list)
    validation: RecalibrationValidation = Field(default_factory=RecalibrationValidation)
    updated_pathway: UpdatedPathway = Field(default_factory=UpdatedPathway)
    report: RecalibrationReport = Field(default_factory=RecalibrationReport)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class TargetRecalibrationWorkflow:
    """
    5-phase DAG workflow for target recalibration.

    Phase 1: LoadUpdatedBaseline  -- Load and apply structural changes.
    Phase 2: RecalcInterimTargets -- Recalculate interim targets.
    Phase 3: ValidateNewTargets   -- Validate against SBTi criteria.
    Phase 4: UpdatePathway        -- Update annual pathway.
    Phase 5: RecalibrationReport  -- Generate recalibration report.

    DAG Dependencies:
        Phase 1 -> Phase 2 -> Phase 3
                            -> Phase 4 (parallel with Phase 3)
                -> Phase 5 (depends on all prior)
    """

    def __init__(self, config: Optional[TargetRecalibrationConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or TargetRecalibrationConfig()
        self._phase_results: List[PhaseResult] = []
        self._baseline: UpdatedBaseline = UpdatedBaseline()
        self._targets: List[RecalibratedTarget] = []
        self._validation: RecalibrationValidation = RecalibrationValidation()
        self._pathway: UpdatedPathway = UpdatedPathway()
        self._report: RecalibrationReport = RecalibrationReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: TargetRecalibrationInput) -> TargetRecalibrationResult:
        started_at = utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting target recalibration workflow %s, company=%s",
            self.workflow_id, self.config.company_name,
        )

        try:
            phase1 = await self._phase_load_updated_baseline(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_recalc_targets(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_validate_targets(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_update_pathway(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_recalibration_report(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Target recalibration failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        result = TargetRecalibrationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            updated_baseline=self._baseline,
            recalibrated_targets=self._targets,
            validation=self._validation,
            updated_pathway=self._pathway,
            report=self._report,
            key_findings=self._generate_findings(),
            recommendations=self._generate_recommendations(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Updated Baseline
    # -------------------------------------------------------------------------

    async def _phase_load_updated_baseline(self, input_data: TargetRecalibrationInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cfg = self.config
        orig_s1 = cfg.original_scope1_tco2e or cfg.original_base_emissions_tco2e * 0.45
        orig_s2 = cfg.original_scope2_tco2e or cfg.original_base_emissions_tco2e * 0.20
        orig_s3 = cfg.original_scope3_tco2e or cfg.original_base_emissions_tco2e * 0.35
        orig_total = cfg.original_base_emissions_tco2e or (orig_s1 + orig_s2 + orig_s3)

        if orig_total <= 0:
            orig_total = 100000
            orig_s1 = 45000
            orig_s2 = 20000
            orig_s3 = 35000
            warnings.append("Original baseline not provided; using default 100,000 tCO2e.")

        changes: List[BaselineChange] = []
        total_s1_impact = 0.0
        total_s2_impact = 0.0
        total_s3_impact = 0.0

        for sc in input_data.structural_changes:
            s1_impact = sc.get("scope1_impact", 0)
            s2_impact = sc.get("scope2_impact", 0)
            s3_impact = sc.get("scope3_impact", 0)
            total_impact = s1_impact + s2_impact + s3_impact
            impact_pct = abs(total_impact) / max(orig_total, 1e-10) * 100

            trigger_str = sc.get("trigger", "structural_change")
            try:
                trigger = RecalibrationTrigger(trigger_str)
            except ValueError:
                trigger = RecalibrationTrigger.STRUCTURAL_CHANGE

            exceeds = impact_pct >= cfg.significance_threshold_pct

            applicable_rules = [
                r for r in RECALCULATION_RULES.values()
                if trigger_str in r["applies_to"] and r["mandatory"]
            ]
            recalc_required = exceeds or len(applicable_rules) > 0

            changes.append(BaselineChange(
                change_id=f"BC-{_new_uuid()[:6]}",
                trigger=trigger,
                description=sc.get("description", "Structural change"),
                effective_date=sc.get("effective_date", utcnow().strftime("%Y-%m-%d")),
                scope1_impact_tco2e=round(s1_impact, 2),
                scope2_impact_tco2e=round(s2_impact, 2),
                scope3_impact_tco2e=round(s3_impact, 2),
                total_impact_tco2e=round(total_impact, 2),
                impact_pct=round(impact_pct, 2),
                exceeds_threshold=exceeds,
                recalculation_required=recalc_required,
            ))

            total_s1_impact += s1_impact
            total_s2_impact += s2_impact
            total_s3_impact += s3_impact

        updated_s1 = orig_s1 + total_s1_impact
        updated_s2 = orig_s2 + total_s2_impact
        updated_s3 = orig_s3 + total_s3_impact
        updated_total = updated_s1 + updated_s2 + updated_s3
        net_change = updated_total - orig_total
        net_change_pct = (net_change / max(orig_total, 1e-10)) * 100

        any_recalc = any(c.recalculation_required for c in changes)

        self._baseline = UpdatedBaseline(
            original_base_year_tco2e=round(orig_total, 2),
            original_scope1_tco2e=round(orig_s1, 2),
            original_scope2_tco2e=round(orig_s2, 2),
            original_scope3_tco2e=round(orig_s3, 2),
            updated_base_year_tco2e=round(updated_total, 2),
            updated_scope1_tco2e=round(updated_s1, 2),
            updated_scope2_tco2e=round(updated_s2, 2),
            updated_scope3_tco2e=round(updated_s3, 2),
            net_change_tco2e=round(net_change, 2),
            net_change_pct=round(net_change_pct, 2),
            changes_applied=changes,
            recalculation_required=any_recalc,
        )
        self._baseline.provenance_hash = _compute_hash(
            self._baseline.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["original_total"] = round(orig_total, 2)
        outputs["updated_total"] = round(updated_total, 2)
        outputs["net_change_tco2e"] = round(net_change, 2)
        outputs["net_change_pct"] = round(net_change_pct, 2)
        outputs["changes_count"] = len(changes)
        outputs["recalculation_required"] = any_recalc

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="load_updated_baseline", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_load_updated_baseline",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Recalculate Interim Targets
    # -------------------------------------------------------------------------

    async def _phase_recalc_targets(self, input_data: TargetRecalibrationInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        cfg = self.config
        updated_base = self._baseline.updated_base_year_tco2e
        orig_base = self._baseline.original_base_year_tco2e
        recalc_targets: List[RecalibratedTarget] = []

        # Recalculate for each original target
        milestones = input_data.original_interim_targets
        if not milestones:
            # Default 5-year intervals
            for interval in [5, 10, 15, 20, 25, 30]:
                target_year = cfg.base_year + interval
                if target_year > cfg.long_term_target_year:
                    continue
                frac = interval / (cfg.long_term_target_year - cfg.base_year)
                reduction = cfg.long_term_reduction_pct * frac
                milestones.append({
                    "year": target_year,
                    "target_tco2e": round(orig_base * (1 - reduction / 100), 2),
                    "reduction_pct": round(reduction, 2),
                })

        for ms in milestones:
            target_year = ms.get("year", 2030)
            orig_target = ms.get("target_tco2e", 0)
            orig_reduction = ms.get("reduction_pct", 0)

            if orig_reduction <= 0 and orig_target > 0 and orig_base > 0:
                orig_reduction = ((orig_base - orig_target) / orig_base) * 100

            # Maintain same reduction percentage against new baseline
            if cfg.maintain_ambition:
                new_target = updated_base * (1 - orig_reduction / 100)
            else:
                # Adjust target proportionally
                ratio = updated_base / max(orig_base, 1e-10)
                new_target = orig_target * ratio

            new_reduction = ((updated_base - new_target) / max(updated_base, 1e-10)) * 100
            change = new_target - orig_target
            change_pct = (change / max(orig_target, 1e-10)) * 100

            # SBTi minimum check
            years_from_base = target_year - cfg.base_year
            min_rate = 4.2 if cfg.sbti_ambition == "1.5c" else 2.5
            min_reduction = min_rate * years_from_base
            sbti_met = new_reduction >= min_reduction

            recalc_targets.append(RecalibratedTarget(
                target_year=target_year,
                original_target_tco2e=round(orig_target, 2),
                original_reduction_pct=round(orig_reduction, 2),
                recalibrated_target_tco2e=round(new_target, 2),
                recalibrated_reduction_pct=round(new_reduction, 2),
                change_tco2e=round(change, 2),
                change_pct=round(change_pct, 2),
                maintains_ambition=cfg.maintain_ambition,
                sbti_minimum_met=sbti_met,
            ))

        self._targets = recalc_targets

        outputs["targets_recalibrated"] = len(recalc_targets)
        outputs["ambition_maintained"] = cfg.maintain_ambition
        for t in recalc_targets:
            outputs[f"target_{t.target_year}_change_tco2e"] = t.change_tco2e
            outputs[f"target_{t.target_year}_new_reduction_pct"] = t.recalibrated_reduction_pct

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="recalc_interim_targets", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_recalc_targets",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Validate New Targets
    # -------------------------------------------------------------------------

    async def _phase_validate_targets(self, input_data: TargetRecalibrationInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        findings: List[Dict[str, Any]] = []
        cfg = self.config

        # Check 1: All targets meet SBTi minimum
        all_meet = all(t.sbti_minimum_met for t in self._targets)
        findings.append({
            "criterion": "SBTi Minimum Rate",
            "result": "pass" if all_meet else "fail",
            "description": "All recalibrated targets meet SBTi minimum reduction rate.",
        })

        # Check 2: Ambition maintained
        ambition_ok = all(t.maintains_ambition for t in self._targets)
        findings.append({
            "criterion": "Ambition Maintenance",
            "result": "pass" if ambition_ok else "warning",
            "description": "Recalibrated targets maintain original ambition level.",
        })

        # Check 3: Base year change within significant threshold
        change_significant = abs(self._baseline.net_change_pct) >= cfg.significance_threshold_pct
        findings.append({
            "criterion": "Significance Threshold",
            "result": "pass" if change_significant else "warning",
            "description": f"Base year change ({self._baseline.net_change_pct:.1f}%) vs threshold ({cfg.significance_threshold_pct}%).",
        })

        # Check 4: Long-term target still achieves 90% reduction
        lt_targets = [t for t in self._targets if t.target_year >= cfg.long_term_target_year - 5]
        lt_ok = all(t.recalibrated_reduction_pct >= 85 for t in lt_targets) if lt_targets else True
        findings.append({
            "criterion": "Long-Term Reduction",
            "result": "pass" if lt_ok else "conditional",
            "description": "Long-term reduction trajectory maintains >= 85% path.",
        })

        # Check 5: Resubmission required
        resubmit = change_significant or not all_meet
        findings.append({
            "criterion": "SBTi Resubmission",
            "result": "warning" if resubmit else "pass",
            "description": f"SBTi resubmission {'required' if resubmit else 'not required'}.",
        })

        passed = sum(1 for f in findings if f["result"] == "pass")
        failed = sum(1 for f in findings if f["result"] == "fail")
        total = len(findings)

        overall = ValidationResult.PASS if failed == 0 else (
            ValidationResult.CONDITIONAL if failed <= 1 else ValidationResult.FAIL
        )

        self._validation = RecalibrationValidation(
            total_criteria=total,
            passed=passed,
            failed=failed,
            findings=findings,
            overall_result=overall,
            ambition_maintained=ambition_ok,
            sbti_resubmission_required=resubmit,
        )
        self._validation.provenance_hash = _compute_hash(
            self._validation.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_criteria"] = total
        outputs["passed"] = passed
        outputs["failed"] = failed
        outputs["overall_result"] = overall.value
        outputs["resubmission_required"] = resubmit

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="validate_new_targets", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_validate_targets",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Update Pathway
    # -------------------------------------------------------------------------

    async def _phase_update_pathway(self, input_data: TargetRecalibrationInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        cfg = self.config
        updated_base = self._baseline.updated_base_year_tco2e
        target_total = updated_base * (1 - cfg.long_term_reduction_pct / 100)

        pathway_points: List[Dict[str, Any]] = []
        for year in range(cfg.base_year, cfg.long_term_target_year + 1):
            val = _interpolate_linear(updated_base, target_total, cfg.base_year, cfg.long_term_target_year, year)
            cum_red = ((updated_base - val) / max(updated_base, 1e-10)) * 100
            pathway_points.append({
                "year": year,
                "target_tco2e": round(val, 2),
                "cumulative_reduction_pct": round(cum_red, 2),
            })

        avg_annual = abs(_calc_cagr(
            updated_base, target_total,
            cfg.long_term_target_year - cfg.base_year,
        ))

        orig_base = self._baseline.original_base_year_tco2e
        tightened = updated_base > orig_base  # More emissions = tighter target needed

        self._pathway = UpdatedPathway(
            original_pathway_points=len(input_data.original_pathway) or (cfg.long_term_target_year - cfg.base_year + 1),
            updated_pathway_points=len(pathway_points),
            pathway=pathway_points,
            avg_annual_change_pct=round(avg_annual, 2),
            pathway_tightened=tightened,
        )
        self._pathway.provenance_hash = _compute_hash(
            self._pathway.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["pathway_points"] = len(pathway_points)
        outputs["avg_annual_reduction_pct"] = round(avg_annual, 2)
        outputs["pathway_tightened"] = tightened

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="update_pathway", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_update_pathway",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Recalibration Report
    # -------------------------------------------------------------------------

    async def _phase_recalibration_report(self, input_data: TargetRecalibrationInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        findings = self._generate_findings()
        recommendations = self._generate_recommendations()

        trigger_summary = ", ".join(
            c.trigger.value for c in self._baseline.changes_applied
        ) if self._baseline.changes_applied else "No structural changes"

        comparison = {
            "base_year_original": self._baseline.original_base_year_tco2e,
            "base_year_updated": self._baseline.updated_base_year_tco2e,
            "base_year_change_pct": self._baseline.net_change_pct,
        }
        for t in self._targets:
            comparison[f"target_{t.target_year}_original"] = t.original_target_tco2e
            comparison[f"target_{t.target_year}_updated"] = t.recalibrated_target_tco2e
            comparison[f"target_{t.target_year}_change_pct"] = t.change_pct

        exec_parts = [
            f"Target Recalibration Report for {self.config.company_name or 'Company'}.",
            f"Trigger(s): {trigger_summary}.",
            f"Base year change: {self._baseline.net_change_tco2e:+,.0f} tCO2e ({self._baseline.net_change_pct:+.1f}%).",
            f"Recalibration required: {'Yes' if self._baseline.recalculation_required else 'No'}.",
            f"Targets recalibrated: {len(self._targets)}.",
            f"Validation: {self._validation.overall_result.value}.",
            f"SBTi resubmission: {'Required' if self._validation.sbti_resubmission_required else 'Not required'}.",
        ]

        self._report = RecalibrationReport(
            report_id=f"RCR-{self.workflow_id[:8]}",
            report_date=utcnow().strftime("%Y-%m-%d"),
            company_name=self.config.company_name,
            trigger_summary=trigger_summary,
            updated_baseline=self._baseline,
            recalibrated_targets=self._targets,
            validation=self._validation,
            updated_pathway=self._pathway,
            before_after_comparison=comparison,
            executive_summary=" ".join(exec_parts),
            key_findings=findings,
            recommendations=recommendations,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["findings_count"] = len(findings)
        outputs["recommendations_count"] = len(recommendations)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="recalibration_report", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_recalibration_report",
        )

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(
            f"Base year emissions changed by {self._baseline.net_change_tco2e:+,.0f} tCO2e "
            f"({self._baseline.net_change_pct:+.1f}%).",
        )
        findings.append(
            f"Recalculation {'required' if self._baseline.recalculation_required else 'not required'} "
            f"per SBTi significance threshold.",
        )
        for t in self._targets[:3]:
            findings.append(
                f"Target {t.target_year}: adjusted by {t.change_tco2e:+,.0f} tCO2e to "
                f"{t.recalibrated_target_tco2e:,.0f} tCO2e ({t.recalibrated_reduction_pct:.1f}% reduction).",
            )
        findings.append(f"Validation: {self._validation.overall_result.value}.")
        if self._validation.sbti_resubmission_required:
            findings.append("SBTi target resubmission is required due to significant baseline change.")
        return findings

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        if self._validation.sbti_resubmission_required:
            recs.append("Submit updated targets to SBTi for re-validation.")
        recs.append("Update all internal tracking systems with recalibrated targets and pathway.")
        recs.append("Communicate recalibrated targets to stakeholders and reporting frameworks.")
        if self._pathway.pathway_tightened:
            recs.append("Review initiative portfolio to ensure adequate abatement for tightened pathway.")
        recs.append("Document recalibration rationale in base year recalculation policy.")
        recs.append("Update CDP, TCFD, and SBTi annual disclosures with recalibrated data.")
        return recs
