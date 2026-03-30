# -*- coding: utf-8 -*-
"""
ESRS S1 Own Workforce Workflow
================================

8-phase workflow for ESRS S1 Own Workforce disclosure covering policy review,
demographics analysis, health and safety metrics, training metrics, diversity
and pay gap analysis, work-life balance assessment, human rights due diligence,
and report assembly with full provenance tracking.

Phases:
    1. PolicyReview       -- Review workforce policies (S1-1, S1-2, S1-3, S1-4)
    2. Demographics       -- Analyse workforce demographics (S1-6, S1-7, S1-8)
    3. HealthSafety       -- Assess health and safety metrics (S1-14)
    4. Training           -- Evaluate training and development (S1-13)
    5. DiversityPay       -- Analyse diversity and pay equity (S1-9, S1-16, S1-17)
    6. WorkLifeBalance    -- Assess work-life balance and leave (S1-15)
    7. HumanRights        -- Evaluate human rights due diligence (S1-10, S1-11, S1-12)
    8. ReportAssembly     -- Assemble complete S1 disclosure (S1-5)

ESRS S1 Disclosure Requirements (17 DRs):
    S1-1:  Policies related to own workforce
    S1-2:  Processes for engaging with own workforce
    S1-3:  Processes to remediate negative impacts and channels
    S1-4:  Taking action on material impacts
    S1-5:  Targets related to managing impacts
    S1-6:  Characteristics of the undertaking's employees
    S1-7:  Characteristics of non-employee workers
    S1-8:  Collective bargaining coverage and social dialogue
    S1-9:  Diversity metrics
    S1-10: Adequate wages
    S1-11: Social protection
    S1-12: Persons with disabilities
    S1-13: Training and skills development metrics
    S1-14: Health and safety metrics
    S1-15: Work-life balance metrics
    S1-16: Remuneration metrics (pay gap indicators)
    S1-17: Incidents, complaints and severe human rights impacts

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
    """Phases of the S1 own workforce workflow."""
    POLICY_REVIEW = "policy_review"
    DEMOGRAPHICS = "demographics"
    HEALTH_SAFETY = "health_safety"
    TRAINING = "training"
    DIVERSITY_PAY = "diversity_pay"
    WORK_LIFE_BALANCE = "work_life_balance"
    HUMAN_RIGHTS = "human_rights"
    REPORT_ASSEMBLY = "report_assembly"

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

class ContractType(str, Enum):
    """Employment contract type."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    NON_GUARANTEED_HOURS = "non_guaranteed_hours"

class GenderCategory(str, Enum):
    """Gender category for diversity reporting."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    NOT_DISCLOSED = "not_disclosed"

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

class WorkforceDemographics(BaseModel):
    """Workforce demographics data per S1-6/S1-7."""
    total_employees: int = Field(default=0, ge=0)
    total_fte: float = Field(default=0.0, ge=0.0)
    by_gender: Dict[str, int] = Field(default_factory=dict)
    by_country: Dict[str, int] = Field(default_factory=dict)
    by_contract_type: Dict[str, int] = Field(default_factory=dict)
    non_employee_workers: int = Field(default=0, ge=0)
    turnover_rate_pct: float = Field(default=0.0, ge=0.0)
    collective_bargaining_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class HealthSafetyData(BaseModel):
    """Health and safety metrics per S1-14."""
    fatalities: int = Field(default=0, ge=0)
    recordable_incidents: int = Field(default=0, ge=0)
    lost_time_incidents: int = Field(default=0, ge=0)
    total_hours_worked: float = Field(default=0.0, ge=0.0)
    ltir: float = Field(default=0.0, ge=0.0, description="Lost Time Injury Rate")
    trir: float = Field(default=0.0, ge=0.0, description="Total Recordable Incident Rate")
    near_misses: int = Field(default=0, ge=0)
    occupational_diseases: int = Field(default=0, ge=0)

class TrainingData(BaseModel):
    """Training and development metrics per S1-13."""
    total_training_hours: float = Field(default=0.0, ge=0.0)
    avg_training_hours_per_employee: float = Field(default=0.0, ge=0.0)
    by_gender: Dict[str, float] = Field(default_factory=dict, description="Avg hours by gender")
    by_category: Dict[str, float] = Field(default_factory=dict, description="Avg hours by job category")
    skills_development_programs: int = Field(default=0, ge=0)

class DiversityPayData(BaseModel):
    """Diversity and pay equity metrics per S1-9, S1-16, S1-17."""
    gender_pay_gap_pct: float = Field(default=0.0, description="Unadjusted mean gender pay gap")
    gender_pay_gap_median_pct: float = Field(default=0.0, description="Median gender pay gap")
    management_gender_ratio: Dict[str, float] = Field(default_factory=dict)
    board_gender_ratio: Dict[str, float] = Field(default_factory=dict)
    adequate_wage_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    disability_inclusion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    incidents_reported: int = Field(default=0, ge=0)
    incidents_resolved: int = Field(default=0, ge=0)

class S1WorkforceInput(BaseModel):
    """Input data model for S1WorkforceWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    s1_is_material: bool = Field(default=True, description="Whether S1 is material")
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="S1-1 workforce policies"
    )
    engagement_processes: List[Dict[str, Any]] = Field(
        default_factory=list, description="S1-2 engagement processes"
    )
    remediation_processes: List[Dict[str, Any]] = Field(
        default_factory=list, description="S1-3 remediation channels"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="S1-4 actions on impacts"
    )
    demographics: Optional[WorkforceDemographics] = Field(
        default=None, description="S1-6/S1-7/S1-8 workforce demographics"
    )
    health_safety: Optional[HealthSafetyData] = Field(
        default=None, description="S1-14 health and safety"
    )
    training: Optional[TrainingData] = Field(
        default=None, description="S1-13 training data"
    )
    diversity_pay: Optional[DiversityPayData] = Field(
        default=None, description="S1-9/S1-16/S1-17 diversity and pay"
    )
    work_life_balance_data: Dict[str, Any] = Field(
        default_factory=dict, description="S1-15 work-life balance"
    )
    social_protection_data: Dict[str, Any] = Field(
        default_factory=dict, description="S1-11 social protection"
    )
    targets: List[Dict[str, Any]] = Field(
        default_factory=list, description="S1-5 targets"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class S1WorkforceWorkflowResult(BaseModel):
    """Complete result from S1 workforce workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="s1_workforce")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    s1_is_material: bool = Field(default=True)
    total_employees: int = Field(default=0)
    total_fte: float = Field(default=0.0)
    turnover_rate_pct: float = Field(default=0.0)
    collective_bargaining_pct: float = Field(default=0.0)
    fatalities: int = Field(default=0)
    ltir: float = Field(default=0.0)
    trir: float = Field(default=0.0)
    avg_training_hours: float = Field(default=0.0)
    gender_pay_gap_pct: float = Field(default=0.0)
    incidents_reported: int = Field(default=0)
    policies_count: int = Field(default=0)
    targets_count: int = Field(default=0)
    overall_completeness_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class S1WorkforceWorkflow:
    """
    8-phase ESRS S1 Own Workforce workflow.

    Orchestrates policy review, demographics, health and safety, training,
    diversity and pay, work-life balance, human rights, and report assembly
    for complete S1 disclosure covering S1-1 through S1-17.

    Zero-hallucination: all workforce metric aggregations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = S1WorkforceWorkflow()
        >>> inp = S1WorkforceInput(demographics=WorkforceDemographics(...))
        >>> result = await wf.execute(inp)
        >>> assert result.total_employees >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize S1WorkforceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review workforce policies"},
            {"name": WorkflowPhase.DEMOGRAPHICS.value, "description": "Analyse workforce demographics"},
            {"name": WorkflowPhase.HEALTH_SAFETY.value, "description": "Assess health and safety"},
            {"name": WorkflowPhase.TRAINING.value, "description": "Evaluate training metrics"},
            {"name": WorkflowPhase.DIVERSITY_PAY.value, "description": "Analyse diversity and pay equity"},
            {"name": WorkflowPhase.WORK_LIFE_BALANCE.value, "description": "Assess work-life balance"},
            {"name": WorkflowPhase.HUMAN_RIGHTS.value, "description": "Evaluate human rights due diligence"},
            {"name": WorkflowPhase.REPORT_ASSEMBLY.value, "description": "Assemble S1 disclosure"},
        ]

    def validate_inputs(self, input_data: S1WorkforceInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.s1_is_material:
            issues.append("S1 is not material; full disclosure not required")
        if input_data.demographics is None:
            issues.append("No demographics data provided (S1-6)")
        if input_data.health_safety is None:
            issues.append("No health and safety data provided (S1-14)")
        return issues

    async def execute(
        self,
        input_data: Optional[S1WorkforceInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> S1WorkforceWorkflowResult:
        """
        Execute the 8-phase S1 workforce workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            S1WorkforceWorkflowResult with workforce metrics and compliance status.
        """
        if input_data is None:
            input_data = S1WorkforceInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting S1 workforce workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_demographics(input_data))
            phase_results.append(await self._phase_health_safety(input_data))
            phase_results.append(await self._phase_training(input_data))
            phase_results.append(await self._phase_diversity_pay(input_data))
            phase_results.append(await self._phase_work_life_balance(input_data))
            phase_results.append(await self._phase_human_rights(input_data))
            phase_results.append(await self._phase_report_assembly(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("S1 workforce workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        demo = input_data.demographics
        hs = input_data.health_safety
        train = input_data.training
        dp = input_data.diversity_pay
        completeness = self._calculate_completeness(input_data)

        result = S1WorkforceWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            s1_is_material=input_data.s1_is_material,
            total_employees=demo.total_employees if demo else 0,
            total_fte=demo.total_fte if demo else 0.0,
            turnover_rate_pct=demo.turnover_rate_pct if demo else 0.0,
            collective_bargaining_pct=demo.collective_bargaining_pct if demo else 0.0,
            fatalities=hs.fatalities if hs else 0,
            ltir=hs.ltir if hs else 0.0,
            trir=hs.trir if hs else 0.0,
            avg_training_hours=train.avg_training_hours_per_employee if train else 0.0,
            gender_pay_gap_pct=dp.gender_pay_gap_pct if dp else 0.0,
            incidents_reported=dp.incidents_reported if dp else 0,
            policies_count=len(input_data.policies),
            targets_count=len(input_data.targets),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "S1 workforce %s completed in %.2fs: %d employees, LTIR=%.2f",
            self.workflow_id, elapsed, result.total_employees, result.ltir,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review (S1-1 to S1-4)
    # -------------------------------------------------------------------------

    async def _phase_policy_review(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Review workforce policies, engagement, remediation, and actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["policies_count"] = len(input_data.policies)
        outputs["engagement_processes_count"] = len(input_data.engagement_processes)
        outputs["remediation_channels_count"] = len(input_data.remediation_processes)
        outputs["actions_count"] = len(input_data.actions)
        outputs["targets_count"] = len(input_data.targets)

        if not input_data.policies:
            warnings.append("No workforce policies defined (S1-1)")
        if not input_data.engagement_processes:
            warnings.append("No engagement processes defined (S1-2)")
        if not input_data.remediation_processes:
            warnings.append("No remediation channels defined (S1-3)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 PolicyReview: %d policies, %d actions",
                         len(input_data.policies), len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Demographics (S1-6, S1-7, S1-8)
    # -------------------------------------------------------------------------

    async def _phase_demographics(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Analyse workforce demographics and composition."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        demo = input_data.demographics
        if demo:
            outputs["total_employees"] = demo.total_employees
            outputs["total_fte"] = demo.total_fte
            outputs["by_gender"] = demo.by_gender
            outputs["by_country"] = demo.by_country
            outputs["by_contract_type"] = demo.by_contract_type
            outputs["non_employee_workers"] = demo.non_employee_workers
            outputs["turnover_rate_pct"] = demo.turnover_rate_pct
            outputs["collective_bargaining_pct"] = demo.collective_bargaining_pct

            if demo.collective_bargaining_pct < 50:
                warnings.append("Collective bargaining coverage below 50% (S1-8)")
            if not demo.by_gender:
                warnings.append("Gender breakdown not provided (S1-6)")
        else:
            outputs["total_employees"] = 0
            warnings.append("No demographics data provided (S1-6)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 Demographics: %d employees", outputs["total_employees"])
        return PhaseResult(
            phase_name=WorkflowPhase.DEMOGRAPHICS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Health & Safety (S1-14)
    # -------------------------------------------------------------------------

    async def _phase_health_safety(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Assess health and safety metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        hs = input_data.health_safety
        if hs:
            outputs["fatalities"] = hs.fatalities
            outputs["recordable_incidents"] = hs.recordable_incidents
            outputs["lost_time_incidents"] = hs.lost_time_incidents
            outputs["total_hours_worked"] = hs.total_hours_worked
            outputs["ltir"] = hs.ltir
            outputs["trir"] = hs.trir
            outputs["near_misses"] = hs.near_misses
            outputs["occupational_diseases"] = hs.occupational_diseases

            if hs.fatalities > 0:
                warnings.append(f"CRITICAL: {hs.fatalities} workplace fatalities reported")
            if hs.ltir == 0 and hs.total_hours_worked == 0:
                warnings.append("No hours worked data; LTIR cannot be validated")
        else:
            outputs["fatalities"] = 0
            warnings.append("No health and safety data provided (S1-14)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 HealthSafety: fatalities=%d, LTIR=%.2f",
                         outputs["fatalities"], outputs.get("ltir", 0))
        return PhaseResult(
            phase_name=WorkflowPhase.HEALTH_SAFETY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Training (S1-13)
    # -------------------------------------------------------------------------

    async def _phase_training(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Evaluate training and development metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        train = input_data.training
        if train:
            outputs["total_training_hours"] = train.total_training_hours
            outputs["avg_hours_per_employee"] = train.avg_training_hours_per_employee
            outputs["by_gender"] = train.by_gender
            outputs["by_category"] = train.by_category
            outputs["skills_programs"] = train.skills_development_programs

            if train.avg_training_hours_per_employee < 8:
                warnings.append("Average training hours below 8 per employee")
        else:
            outputs["total_training_hours"] = 0
            warnings.append("No training data provided (S1-13)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 Training: %.0f total hours",
                         outputs["total_training_hours"])
        return PhaseResult(
            phase_name=WorkflowPhase.TRAINING.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Diversity & Pay (S1-9, S1-16, S1-17)
    # -------------------------------------------------------------------------

    async def _phase_diversity_pay(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Analyse diversity metrics and pay equity."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        dp = input_data.diversity_pay
        if dp:
            outputs["gender_pay_gap_pct"] = dp.gender_pay_gap_pct
            outputs["gender_pay_gap_median_pct"] = dp.gender_pay_gap_median_pct
            outputs["management_gender_ratio"] = dp.management_gender_ratio
            outputs["board_gender_ratio"] = dp.board_gender_ratio
            outputs["adequate_wage_compliance_pct"] = dp.adequate_wage_compliance_pct
            outputs["disability_inclusion_pct"] = dp.disability_inclusion_pct
            outputs["incidents_reported"] = dp.incidents_reported
            outputs["incidents_resolved"] = dp.incidents_resolved

            if dp.gender_pay_gap_pct > 5.0:
                warnings.append(f"Gender pay gap of {dp.gender_pay_gap_pct}% exceeds 5% threshold")
            if dp.adequate_wage_compliance_pct < 100:
                warnings.append("Not all workers receiving adequate wages (S1-10)")
        else:
            outputs["gender_pay_gap_pct"] = 0
            warnings.append("No diversity/pay data provided (S1-9/S1-16)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 DiversityPay: pay_gap=%.1f%%",
                         outputs["gender_pay_gap_pct"])
        return PhaseResult(
            phase_name=WorkflowPhase.DIVERSITY_PAY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Work-Life Balance (S1-15)
    # -------------------------------------------------------------------------

    async def _phase_work_life_balance(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Assess work-life balance and leave metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.work_life_balance_data
        outputs["has_data"] = bool(data)
        outputs["family_leave_entitled_pct"] = data.get("family_leave_entitled_pct", 0)
        outputs["family_leave_taken_pct"] = data.get("family_leave_taken_pct", 0)
        outputs["flexible_working_pct"] = data.get("flexible_working_pct", 0)
        outputs["avg_working_hours"] = data.get("avg_working_hours", 0)

        if not data:
            warnings.append("No work-life balance data provided (S1-15)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 6 WorkLifeBalance: has_data=%s", bool(data))
        return PhaseResult(
            phase_name=WorkflowPhase.WORK_LIFE_BALANCE.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Human Rights (S1-10, S1-11, S1-12)
    # -------------------------------------------------------------------------

    async def _phase_human_rights(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Evaluate human rights due diligence."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["has_social_protection"] = bool(input_data.social_protection_data)
        outputs["social_protection_coverage_pct"] = input_data.social_protection_data.get("coverage_pct", 0)
        outputs["has_disability_data"] = bool(
            input_data.diversity_pay and input_data.diversity_pay.disability_inclusion_pct > 0
        )

        dp = input_data.diversity_pay
        if dp:
            outputs["severe_incidents"] = dp.incidents_reported
            outputs["incidents_resolved_pct"] = round(
                (dp.incidents_resolved / dp.incidents_reported * 100)
                if dp.incidents_reported > 0 else 100.0, 1
            )
            if dp.incidents_reported > 0 and dp.incidents_resolved < dp.incidents_reported:
                warnings.append("Unresolved human rights incidents exist (S1-17)")
        else:
            outputs["severe_incidents"] = 0

        if not input_data.social_protection_data:
            warnings.append("No social protection data provided (S1-11)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 7 HumanRights: incidents=%d", outputs["severe_incidents"])
        return PhaseResult(
            phase_name=WorkflowPhase.HUMAN_RIGHTS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Report Assembly
    # -------------------------------------------------------------------------

    async def _phase_report_assembly(self, input_data: S1WorkforceInput) -> PhaseResult:
        """Assemble complete S1 disclosure from all phase results."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Count data availability across all 17 DRs
        dr_available = sum([
            bool(input_data.policies),                    # S1-1
            bool(input_data.engagement_processes),        # S1-2
            bool(input_data.remediation_processes),       # S1-3
            bool(input_data.actions),                     # S1-4
            bool(input_data.targets),                     # S1-5
            input_data.demographics is not None,          # S1-6
            input_data.demographics is not None and input_data.demographics.non_employee_workers >= 0,  # S1-7
            input_data.demographics is not None and input_data.demographics.collective_bargaining_pct >= 0,  # S1-8
            input_data.diversity_pay is not None,         # S1-9
            input_data.diversity_pay is not None and input_data.diversity_pay.adequate_wage_compliance_pct >= 0,  # S1-10
            bool(input_data.social_protection_data),      # S1-11
            input_data.diversity_pay is not None,         # S1-12
            input_data.training is not None,              # S1-13
            input_data.health_safety is not None,         # S1-14
            bool(input_data.work_life_balance_data),      # S1-15
            input_data.diversity_pay is not None,         # S1-16
            input_data.diversity_pay is not None,         # S1-17
        ])

        outputs["drs_with_data"] = dr_available
        outputs["drs_total"] = 17
        outputs["completeness_pct"] = round((dr_available / 17 * 100), 1)
        outputs["disclosure_ready"] = dr_available >= 14  # 80% threshold

        if dr_available < 17:
            warnings.append(f"{17 - dr_available} S1 disclosure requirements missing data")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 8 ReportAssembly: %d/17 DRs with data (%.1f%%)",
                         dr_available, outputs["completeness_pct"])
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_ASSEMBLY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_completeness(self, input_data: S1WorkforceInput) -> float:
        """Calculate overall S1 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.engagement_processes else 0.0)
        scores.append(100.0 if input_data.remediation_processes else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        scores.append(100.0 if input_data.demographics else 0.0)
        scores.append(100.0 if input_data.health_safety else 0.0)
        scores.append(100.0 if input_data.training else 0.0)
        scores.append(100.0 if input_data.diversity_pay else 0.0)
        scores.append(100.0 if input_data.work_life_balance_data else 0.0)
        scores.append(100.0 if input_data.social_protection_data else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: S1WorkforceWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
