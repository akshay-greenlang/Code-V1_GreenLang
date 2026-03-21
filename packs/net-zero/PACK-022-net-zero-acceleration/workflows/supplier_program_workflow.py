# -*- coding: utf-8 -*-
"""
Supplier Engagement Program Workflow
==========================================

5-phase workflow for designing and tracking a Scope 3 supplier engagement
program within PACK-022 Net-Zero Acceleration Pack.  The workflow assesses
suppliers by Scope 3 contribution, tiers them by emission significance,
designs engagement programs per tier, tracks execution progress, and
generates an impact report with estimated Scope 3 reductions.

Phases:
    1. Assessment        -- Assess suppliers by Scope 3 contribution,
                             assign maturity scores
    2. Tiering           -- Tier suppliers into 4 levels by emission
                             contribution (Pareto analysis)
    3. ProgramDesign     -- Design engagement programs per tier
                             (milestones, timeline, resources)
    4. Execution         -- Track execution progress, supplier RAG status
    5. Reporting         -- Generate impact report with Scope 3 reduction
                             estimation

Regulatory references:
    - SBTi Net-Zero Standard v1.2 - Scope 3 requirements
    - SBTi Supplier Engagement guidance
    - CDP Supply Chain Program
    - GHG Protocol Scope 3 Standard (Category 1)

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


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


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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


class SupplierTier(str, Enum):
    """Supplier engagement tiers."""

    TIER_1 = "tier_1"   # Top emitters, deep engagement
    TIER_2 = "tier_2"   # Significant emitters, standard engagement
    TIER_3 = "tier_3"   # Moderate emitters, awareness program
    TIER_4 = "tier_4"   # Low emitters, monitoring only


class RAGStatus(str, Enum):
    """Red/Amber/Green progress status."""

    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    NOT_STARTED = "not_started"


class SupplierMaturity(str, Enum):
    """Supplier climate maturity levels."""

    LEADER = "leader"           # Has SBTi targets, reports CDP
    ADVANCED = "advanced"       # Has targets, some reporting
    DEVELOPING = "developing"   # Awareness, no formal targets
    NASCENT = "nascent"         # No climate action


# =============================================================================
# TIER THRESHOLDS (Zero-Hallucination)
# =============================================================================

# Cumulative emission contribution thresholds for tiering (Pareto-based)
TIER_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "tier_1": {
        "cumulative_pct_max": 50.0,
        "description": "Top emitters covering ~50% of Scope 3",
        "engagement_level": "deep",
        "target": "Set SBTi targets within 24 months",
        "review_frequency": "quarterly",
    },
    "tier_2": {
        "cumulative_pct_max": 75.0,
        "description": "Significant emitters covering 50-75% cumulative",
        "engagement_level": "standard",
        "target": "GHG inventory and reduction plan within 18 months",
        "review_frequency": "semi_annual",
    },
    "tier_3": {
        "cumulative_pct_max": 95.0,
        "description": "Moderate emitters covering 75-95% cumulative",
        "engagement_level": "awareness",
        "target": "Complete climate questionnaire within 12 months",
        "review_frequency": "annual",
    },
    "tier_4": {
        "cumulative_pct_max": 100.0,
        "description": "Low emitters covering remaining 5%",
        "engagement_level": "monitoring",
        "target": "Acknowledge supplier code of conduct",
        "review_frequency": "annual",
    },
}

# Expected reduction potential by tier (conservative estimates)
TIER_REDUCTION_POTENTIAL: Dict[str, float] = {
    "tier_1": 15.0,   # 15% reduction from engaged suppliers
    "tier_2": 10.0,   # 10% reduction
    "tier_3": 5.0,    # 5% reduction
    "tier_4": 2.0,    # 2% reduction
}


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


class SupplierRecord(BaseModel):
    """A single supplier assessment record."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    scope3_contribution_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_contribution_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    spend_usd: float = Field(default=0.0, ge=0.0)
    category: str = Field(default="", description="Scope 3 category (e.g., 'purchased_goods')")
    country: str = Field(default="")
    has_sbti_target: bool = Field(default=False)
    has_ghg_inventory: bool = Field(default=False)
    reports_cdp: bool = Field(default=False)
    maturity: SupplierMaturity = Field(default=SupplierMaturity.NASCENT)


class SupplierAssessment(BaseModel):
    """Assessment result for a single supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    scope3_contribution_tco2e: float = Field(default=0.0)
    scope3_contribution_pct: float = Field(default=0.0)
    cumulative_contribution_pct: float = Field(default=0.0)
    maturity: SupplierMaturity = Field(default=SupplierMaturity.NASCENT)
    maturity_score: int = Field(default=0, ge=0, le=100)
    tier: SupplierTier = Field(default=SupplierTier.TIER_4)
    rank: int = Field(default=0)


class TierSummary(BaseModel):
    """Summary statistics for a supplier tier."""

    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    supplier_count: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    emissions_share_pct: float = Field(default=0.0)
    engagement_level: str = Field(default="")
    target_description: str = Field(default="")
    review_frequency: str = Field(default="")
    expected_reduction_pct: float = Field(default=0.0)


class EngagementMilestone(BaseModel):
    """A milestone in the supplier engagement program."""

    milestone_id: str = Field(default="")
    description: str = Field(default="")
    target_month: int = Field(default=0, ge=0, le=36)
    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    kpi: str = Field(default="")
    success_criteria: str = Field(default="")


class TierProgram(BaseModel):
    """Engagement program design for a single tier."""

    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    program_name: str = Field(default="")
    duration_months: int = Field(default=24)
    milestones: List[EngagementMilestone] = Field(default_factory=list)
    budget_per_supplier_usd: float = Field(default=0.0)
    total_budget_usd: float = Field(default=0.0)
    supplier_count: int = Field(default=0)
    resources_required: List[str] = Field(default_factory=list)


class SupplierProgress(BaseModel):
    """Execution progress for a single supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    rag_status: RAGStatus = Field(default=RAGStatus.NOT_STARTED)
    milestones_completed: int = Field(default=0)
    milestones_total: int = Field(default=0)
    completion_pct: float = Field(default=0.0)
    estimated_reduction_tco2e: float = Field(default=0.0)
    notes: str = Field(default="")


class ImpactReport(BaseModel):
    """Scope 3 supplier program impact report."""

    total_suppliers_engaged: int = Field(default=0)
    total_scope3_covered_tco2e: float = Field(default=0.0)
    coverage_pct: float = Field(default=0.0)
    estimated_reduction_tco2e: float = Field(default=0.0)
    estimated_reduction_pct: float = Field(default=0.0)
    suppliers_with_targets: int = Field(default=0)
    suppliers_with_targets_pct: float = Field(default=0.0)
    green_count: int = Field(default=0)
    amber_count: int = Field(default=0)
    red_count: int = Field(default=0)
    not_started_count: int = Field(default=0)
    total_program_cost_usd: float = Field(default=0.0)
    cost_per_tco2e_reduced: float = Field(default=0.0)
    recommendations: List[str] = Field(default_factory=list)


class SupplierProgramConfig(BaseModel):
    """Configuration for the supplier engagement program workflow."""

    suppliers: List[SupplierRecord] = Field(default_factory=list)
    scope3_baseline_tco2e: float = Field(default=50000.0, ge=0.0)
    coverage_target_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    engagement_budget_usd: float = Field(default=500000.0, ge=0.0)
    program_duration_months: int = Field(default=24, ge=6, le=60)
    base_year: int = Field(default=2024, ge=2015, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class SupplierProgramResult(BaseModel):
    """Complete result from the supplier engagement program workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="supplier_program")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    assessments: List[SupplierAssessment] = Field(default_factory=list)
    tier_summaries: List[TierSummary] = Field(default_factory=list)
    programs: List[TierProgram] = Field(default_factory=list)
    supplier_progress: List[SupplierProgress] = Field(default_factory=list)
    impact_report: ImpactReport = Field(default_factory=ImpactReport)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SupplierProgramWorkflow:
    """
    5-phase supplier engagement program workflow.

    Assesses suppliers by Scope 3 contribution, tiers them by emission
    significance, designs engagement programs, tracks execution, and
    generates an impact report with estimated Scope 3 reductions.

    Zero-hallucination: all tiering thresholds, reduction estimates,
    and cost calculations use deterministic formulas.  No LLM calls
    in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = SupplierProgramWorkflow()
        >>> config = SupplierProgramConfig(suppliers=[...], scope3_baseline=50000)
        >>> result = await wf.execute(config)
        >>> assert result.impact_report.estimated_reduction_tco2e > 0
    """

    def __init__(self) -> None:
        """Initialise SupplierProgramWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._assessments: List[SupplierAssessment] = []
        self._tier_summaries: List[TierSummary] = []
        self._programs: List[TierProgram] = []
        self._progress: List[SupplierProgress] = []
        self._impact: ImpactReport = ImpactReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: SupplierProgramConfig) -> SupplierProgramResult:
        """
        Execute the 5-phase supplier engagement program workflow.

        Args:
            config: Supplier program configuration with supplier records,
                Scope 3 baseline, coverage target, and budget.

        Returns:
            SupplierProgramResult with assessments, programs, progress,
            and impact report.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting supplier program workflow %s, suppliers=%d",
            self.workflow_id, len(config.suppliers),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_assessment(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_tiering(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_program_design(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_execution(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_reporting(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Supplier program workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = SupplierProgramResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            assessments=self._assessments,
            tier_summaries=self._tier_summaries,
            programs=self._programs,
            supplier_progress=self._progress,
            impact_report=self._impact,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Supplier program workflow %s completed in %.2fs",
            self.workflow_id, elapsed,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Assessment
    # -------------------------------------------------------------------------

    async def _phase_assessment(self, config: SupplierProgramConfig) -> PhaseResult:
        """Assess suppliers by Scope 3 contribution and maturity."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        suppliers = config.suppliers
        if not suppliers:
            suppliers = self._generate_sample_suppliers(config)
            warnings.append(
                f"No suppliers provided; generated {len(suppliers)} sample suppliers"
            )

        total_scope3 = config.scope3_baseline_tco2e
        if total_scope3 <= 0:
            total_scope3 = sum(s.scope3_contribution_tco2e for s in suppliers)

        # Calculate contribution percentages and sort descending
        assessed: List[SupplierAssessment] = []
        for supplier in suppliers:
            pct = (supplier.scope3_contribution_tco2e / total_scope3 * 100.0) if total_scope3 > 0 else 0.0
            maturity_score = self._calculate_maturity_score(supplier)
            assessed.append(SupplierAssessment(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.supplier_name,
                scope3_contribution_tco2e=supplier.scope3_contribution_tco2e,
                scope3_contribution_pct=round(pct, 2),
                maturity=supplier.maturity,
                maturity_score=maturity_score,
            ))

        # Sort by contribution descending
        assessed.sort(key=lambda a: a.scope3_contribution_tco2e, reverse=True)

        # Calculate cumulative contribution
        cumulative = 0.0
        for idx, a in enumerate(assessed):
            cumulative += a.scope3_contribution_pct
            a.cumulative_contribution_pct = round(cumulative, 2)
            a.rank = idx + 1

        self._assessments = assessed

        outputs["supplier_count"] = len(assessed)
        outputs["total_scope3_tco2e"] = round(total_scope3, 2)
        outputs["top_5_coverage_pct"] = round(
            assessed[4].cumulative_contribution_pct if len(assessed) >= 5 else cumulative, 2
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Assessment: %d suppliers assessed", len(assessed))
        return PhaseResult(
            phase_name="assessment",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calculate_maturity_score(self, supplier: SupplierRecord) -> int:
        """Calculate supplier climate maturity score (0-100)."""
        score = 0
        if supplier.has_sbti_target:
            score += 40
        if supplier.has_ghg_inventory:
            score += 30
        if supplier.reports_cdp:
            score += 20
        if supplier.maturity == SupplierMaturity.LEADER:
            score += 10
        elif supplier.maturity == SupplierMaturity.ADVANCED:
            score += 5
        return min(score, 100)

    def _generate_sample_suppliers(self, config: SupplierProgramConfig) -> List[SupplierRecord]:
        """Generate sample supplier data when none provided."""
        total = config.scope3_baseline_tco2e
        # Pareto-like distribution: few suppliers dominate
        shares = [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04,
                  0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03]
        suppliers: List[SupplierRecord] = []
        for idx, share in enumerate(shares):
            suppliers.append(SupplierRecord(
                supplier_id=f"SUP-{idx + 1:03d}",
                supplier_name=f"Supplier {idx + 1}",
                scope3_contribution_tco2e=round(total * share, 2),
                scope3_contribution_pct=round(share * 100, 2),
                spend_usd=round(total * share * 50, 2),
                category="purchased_goods",
                has_sbti_target=idx < 2,
                has_ghg_inventory=idx < 5,
                reports_cdp=idx < 3,
                maturity=(
                    SupplierMaturity.LEADER if idx < 2
                    else SupplierMaturity.ADVANCED if idx < 5
                    else SupplierMaturity.DEVELOPING if idx < 10
                    else SupplierMaturity.NASCENT
                ),
            ))
        return suppliers

    # -------------------------------------------------------------------------
    # Phase 2: Tiering
    # -------------------------------------------------------------------------

    async def _phase_tiering(self, config: SupplierProgramConfig) -> PhaseResult:
        """Tier suppliers into 4 levels by emission contribution."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        tier_assignments: Dict[str, List[SupplierAssessment]] = {
            "tier_1": [], "tier_2": [], "tier_3": [], "tier_4": [],
        }

        for assessment in self._assessments:
            cum = assessment.cumulative_contribution_pct
            if cum <= TIER_THRESHOLDS["tier_1"]["cumulative_pct_max"]:
                assessment.tier = SupplierTier.TIER_1
                tier_assignments["tier_1"].append(assessment)
            elif cum <= TIER_THRESHOLDS["tier_2"]["cumulative_pct_max"]:
                assessment.tier = SupplierTier.TIER_2
                tier_assignments["tier_2"].append(assessment)
            elif cum <= TIER_THRESHOLDS["tier_3"]["cumulative_pct_max"]:
                assessment.tier = SupplierTier.TIER_3
                tier_assignments["tier_3"].append(assessment)
            else:
                assessment.tier = SupplierTier.TIER_4
                tier_assignments["tier_4"].append(assessment)

        # Build tier summaries
        total_scope3 = config.scope3_baseline_tco2e
        self._tier_summaries = []
        for tier_key, tier_enum in [
            ("tier_1", SupplierTier.TIER_1),
            ("tier_2", SupplierTier.TIER_2),
            ("tier_3", SupplierTier.TIER_3),
            ("tier_4", SupplierTier.TIER_4),
        ]:
            suppliers_in_tier = tier_assignments[tier_key]
            tier_emissions = sum(s.scope3_contribution_tco2e for s in suppliers_in_tier)
            tier_pct = (tier_emissions / total_scope3 * 100.0) if total_scope3 > 0 else 0.0
            info = TIER_THRESHOLDS[tier_key]
            self._tier_summaries.append(TierSummary(
                tier=tier_enum,
                supplier_count=len(suppliers_in_tier),
                total_emissions_tco2e=round(tier_emissions, 2),
                emissions_share_pct=round(tier_pct, 2),
                engagement_level=info["engagement_level"],
                target_description=info["target"],
                review_frequency=info["review_frequency"],
                expected_reduction_pct=TIER_REDUCTION_POTENTIAL[tier_key],
            ))

        for ts in self._tier_summaries:
            outputs[f"{ts.tier.value}_count"] = ts.supplier_count
            outputs[f"{ts.tier.value}_emissions_pct"] = ts.emissions_share_pct

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Tiering: T1=%d, T2=%d, T3=%d, T4=%d",
                         len(tier_assignments["tier_1"]), len(tier_assignments["tier_2"]),
                         len(tier_assignments["tier_3"]), len(tier_assignments["tier_4"]))
        return PhaseResult(
            phase_name="tiering",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Program Design
    # -------------------------------------------------------------------------

    async def _phase_program_design(self, config: SupplierProgramConfig) -> PhaseResult:
        """Design engagement programs per tier."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_budget = config.engagement_budget_usd
        # Budget allocation: 50% T1, 25% T2, 15% T3, 10% T4
        budget_shares = {"tier_1": 0.50, "tier_2": 0.25, "tier_3": 0.15, "tier_4": 0.10}

        self._programs = []
        for ts in self._tier_summaries:
            tier_key = ts.tier.value
            tier_budget = total_budget * budget_shares.get(tier_key, 0.10)
            per_supplier = (tier_budget / ts.supplier_count) if ts.supplier_count > 0 else 0.0
            milestones = self._design_milestones(ts.tier, config.program_duration_months)
            resources = self._determine_resources(ts.tier)

            self._programs.append(TierProgram(
                tier=ts.tier,
                program_name=f"{ts.tier.value.replace('_', ' ').title()} Engagement Program",
                duration_months=config.program_duration_months,
                milestones=milestones,
                budget_per_supplier_usd=round(per_supplier, 2),
                total_budget_usd=round(tier_budget, 2),
                supplier_count=ts.supplier_count,
                resources_required=resources,
            ))

        outputs["programs_designed"] = len(self._programs)
        outputs["total_budget_allocated"] = round(total_budget, 2)
        for prog in self._programs:
            outputs[f"{prog.tier.value}_budget_usd"] = prog.total_budget_usd
            outputs[f"{prog.tier.value}_milestones"] = len(prog.milestones)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Program design: %d tier programs created", len(self._programs))
        return PhaseResult(
            phase_name="program_design",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _design_milestones(self, tier: SupplierTier, duration_months: int) -> List[EngagementMilestone]:
        """Design milestones for a given tier."""
        milestones: List[EngagementMilestone] = []

        if tier == SupplierTier.TIER_1:
            milestones = [
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m1", description="Kick-off meeting and data request",
                    target_month=1, tier=tier, kpi="meeting_completed",
                    success_criteria="Data sharing agreement signed",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m2", description="GHG inventory baseline established",
                    target_month=6, tier=tier, kpi="ghg_inventory_complete",
                    success_criteria="Scope 1+2 inventory validated",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m3", description="Reduction target set",
                    target_month=12, tier=tier, kpi="target_set",
                    success_criteria="SBTi-aligned target committed",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m4", description="Reduction plan implementation",
                    target_month=18, tier=tier, kpi="plan_implemented",
                    success_criteria="Key reduction actions initiated",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m5", description="Progress review and validation",
                    target_month=min(24, duration_months), tier=tier, kpi="progress_validated",
                    success_criteria="Measurable reduction demonstrated",
                ),
            ]
        elif tier == SupplierTier.TIER_2:
            milestones = [
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m1", description="Climate questionnaire sent",
                    target_month=1, tier=tier, kpi="questionnaire_sent",
                    success_criteria="Questionnaire delivered",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m2", description="Questionnaire response received",
                    target_month=3, tier=tier, kpi="response_received",
                    success_criteria="Complete response with emissions data",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m3", description="GHG inventory guidance provided",
                    target_month=9, tier=tier, kpi="guidance_provided",
                    success_criteria="Supplier has GHG methodology",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m4", description="Reduction commitment received",
                    target_month=min(18, duration_months), tier=tier, kpi="commitment_received",
                    success_criteria="Formal reduction commitment documented",
                ),
            ]
        elif tier == SupplierTier.TIER_3:
            milestones = [
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m1", description="Awareness communication sent",
                    target_month=1, tier=tier, kpi="communication_sent",
                    success_criteria="Supplier acknowledges climate expectations",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m2", description="Basic climate questionnaire completed",
                    target_month=6, tier=tier, kpi="questionnaire_complete",
                    success_criteria="Basic emissions data collected",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m3", description="Annual review",
                    target_month=min(12, duration_months), tier=tier, kpi="annual_review",
                    success_criteria="Annual progress documented",
                ),
            ]
        else:  # TIER_4
            milestones = [
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m1", description="Code of conduct acknowledgement",
                    target_month=3, tier=tier, kpi="code_acknowledged",
                    success_criteria="Supplier code of conduct signed",
                ),
                EngagementMilestone(
                    milestone_id=f"{tier.value}_m2", description="Annual monitoring check",
                    target_month=min(12, duration_months), tier=tier, kpi="monitoring_check",
                    success_criteria="No adverse climate events",
                ),
            ]

        return milestones

    def _determine_resources(self, tier: SupplierTier) -> List[str]:
        """Determine resources required for a tier program."""
        if tier == SupplierTier.TIER_1:
            return [
                "Dedicated sustainability manager per 5 suppliers",
                "GHG accounting training materials",
                "Quarterly review meeting calendar",
                "SBTi target-setting toolkit",
                "On-site assessment capability",
            ]
        elif tier == SupplierTier.TIER_2:
            return [
                "Sustainability analyst per 15 suppliers",
                "Climate questionnaire platform (CDP Supply Chain or equivalent)",
                "Emissions calculation guidance documents",
                "Semi-annual review schedule",
            ]
        elif tier == SupplierTier.TIER_3:
            return [
                "Automated questionnaire platform",
                "Climate awareness training webinars",
                "Standardised communication templates",
            ]
        else:
            return [
                "Automated monitoring system",
                "Supplier code of conduct template",
            ]

    # -------------------------------------------------------------------------
    # Phase 4: Execution Tracking
    # -------------------------------------------------------------------------

    async def _phase_execution(self, config: SupplierProgramConfig) -> PhaseResult:
        """Track execution progress with supplier-by-supplier RAG status."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._progress = []
        program_map = {p.tier: p for p in self._programs}

        for assessment in self._assessments:
            program = program_map.get(assessment.tier)
            total_milestones = len(program.milestones) if program else 0

            # Determine RAG based on maturity (simulation of current state)
            rag, completed = self._determine_rag_status(assessment, total_milestones)
            completion_pct = (completed / total_milestones * 100.0) if total_milestones > 0 else 0.0

            # Estimate reduction based on tier potential and completion
            tier_key = assessment.tier.value
            reduction_potential = TIER_REDUCTION_POTENTIAL.get(tier_key, 0.0)
            estimated_reduction = assessment.scope3_contribution_tco2e * (
                reduction_potential / 100.0
            ) * (completion_pct / 100.0)

            self._progress.append(SupplierProgress(
                supplier_id=assessment.supplier_id,
                supplier_name=assessment.supplier_name,
                tier=assessment.tier,
                rag_status=rag,
                milestones_completed=completed,
                milestones_total=total_milestones,
                completion_pct=round(completion_pct, 1),
                estimated_reduction_tco2e=round(estimated_reduction, 2),
            ))

        green_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.GREEN)
        amber_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.AMBER)
        red_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.RED)
        not_started = sum(1 for p in self._progress if p.rag_status == RAGStatus.NOT_STARTED)

        outputs["green"] = green_count
        outputs["amber"] = amber_count
        outputs["red"] = red_count
        outputs["not_started"] = not_started
        outputs["total_suppliers"] = len(self._progress)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Execution: G=%d, A=%d, R=%d, NS=%d",
            green_count, amber_count, red_count, not_started,
        )
        return PhaseResult(
            phase_name="execution",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _determine_rag_status(
        self, assessment: SupplierAssessment, total_milestones: int
    ) -> tuple:
        """Determine RAG status and milestones completed based on maturity."""
        if assessment.maturity == SupplierMaturity.LEADER:
            completed = total_milestones
            rag = RAGStatus.GREEN
        elif assessment.maturity == SupplierMaturity.ADVANCED:
            completed = max(1, int(total_milestones * 0.6))
            rag = RAGStatus.GREEN
        elif assessment.maturity == SupplierMaturity.DEVELOPING:
            completed = max(0, int(total_milestones * 0.3))
            rag = RAGStatus.AMBER
        else:
            completed = 0
            rag = RAGStatus.RED if total_milestones > 0 else RAGStatus.NOT_STARTED
        return rag, completed

    # -------------------------------------------------------------------------
    # Phase 5: Reporting
    # -------------------------------------------------------------------------

    async def _phase_reporting(self, config: SupplierProgramConfig) -> PhaseResult:
        """Generate impact report with Scope 3 reduction estimation."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_engaged = len(self._progress)
        total_covered = sum(a.scope3_contribution_tco2e for a in self._assessments)
        coverage_pct = (total_covered / config.scope3_baseline_tco2e * 100.0) if config.scope3_baseline_tco2e > 0 else 0.0

        total_reduction = sum(p.estimated_reduction_tco2e for p in self._progress)
        reduction_pct = (total_reduction / config.scope3_baseline_tco2e * 100.0) if config.scope3_baseline_tco2e > 0 else 0.0

        suppliers_with_targets = sum(
            1 for a in self._assessments
            if a.maturity in (SupplierMaturity.LEADER, SupplierMaturity.ADVANCED)
        )
        targets_pct = (suppliers_with_targets / total_engaged * 100.0) if total_engaged > 0 else 0.0

        green_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.GREEN)
        amber_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.AMBER)
        red_count = sum(1 for p in self._progress if p.rag_status == RAGStatus.RED)
        not_started = sum(1 for p in self._progress if p.rag_status == RAGStatus.NOT_STARTED)

        total_cost = sum(p.total_budget_usd for p in self._programs)
        cost_per_tco2e = (total_cost / total_reduction) if total_reduction > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            config, coverage_pct, reduction_pct, red_count, not_started
        )

        self._impact = ImpactReport(
            total_suppliers_engaged=total_engaged,
            total_scope3_covered_tco2e=round(total_covered, 2),
            coverage_pct=round(coverage_pct, 2),
            estimated_reduction_tco2e=round(total_reduction, 2),
            estimated_reduction_pct=round(reduction_pct, 2),
            suppliers_with_targets=suppliers_with_targets,
            suppliers_with_targets_pct=round(targets_pct, 2),
            green_count=green_count,
            amber_count=amber_count,
            red_count=red_count,
            not_started_count=not_started,
            total_program_cost_usd=round(total_cost, 2),
            cost_per_tco2e_reduced=round(cost_per_tco2e, 2),
            recommendations=recommendations,
        )

        outputs["coverage_pct"] = self._impact.coverage_pct
        outputs["reduction_tco2e"] = self._impact.estimated_reduction_tco2e
        outputs["reduction_pct"] = self._impact.estimated_reduction_pct
        outputs["cost_per_tco2e"] = self._impact.cost_per_tco2e_reduced
        outputs["recommendation_count"] = len(recommendations)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Reporting: coverage=%.1f%%, reduction=%.1f tCO2e (%.1f%%)",
            coverage_pct, total_reduction, reduction_pct,
        )
        return PhaseResult(
            phase_name="reporting",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_recommendations(
        self,
        config: SupplierProgramConfig,
        coverage_pct: float,
        reduction_pct: float,
        red_count: int,
        not_started: int,
    ) -> List[str]:
        """Generate actionable recommendations based on program status."""
        recs: List[str] = []

        if coverage_pct < config.coverage_target_pct:
            gap = config.coverage_target_pct - coverage_pct
            recs.append(
                f"Scope 3 coverage is {coverage_pct:.1f}%, below target of {config.coverage_target_pct:.1f}%. "
                f"Expand program to additional suppliers to close the {gap:.1f}% gap."
            )

        if red_count > 0:
            recs.append(
                f"{red_count} supplier(s) have RED status. Escalate engagement "
                "with additional resources or executive-level outreach."
            )

        if not_started > 0:
            recs.append(
                f"{not_started} supplier(s) have not started engagement. "
                "Initiate kick-off communications within 30 days."
            )

        if reduction_pct < 5.0:
            recs.append(
                "Estimated Scope 3 reduction is below 5%. Consider increasing "
                "Tier 1 supplier engagement intensity and providing financial "
                "incentives for early action."
            )

        if coverage_pct >= config.coverage_target_pct and red_count == 0:
            recs.append(
                "Program is on track. Consider expanding to additional Scope 3 "
                "categories (e.g., upstream transportation, business travel)."
            )

        if not recs:
            recs.append("Continue monitoring program progress quarterly.")

        return recs
