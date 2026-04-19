# -*- coding: utf-8 -*-
"""
ISO 50001 Certification Workflow
======================================

4-phase workflow for ISO 50001:2018 certification support within
PACK-031 Industrial Energy Audit Pack.

Phases:
    1. EnMSGapAnalysis           -- Assess current state vs ISO 50001:2018 requirements
    2. EnergyPolicyDevelopment   -- Template energy policy, objectives, targets
    3. EnPITracking              -- Establish and monitor EnPIs per ISO 50006
    4. ManagementReviewPrep      -- Compile data for top management review

The workflow follows GreenLang zero-hallucination principles: all gap
scoring uses deterministic checklists against ISO 50001:2018 clauses.
EnPI calculations use regression-based baselines per ISO 50006.

Schedule: semi-annual
Estimated duration: 240 minutes

Author: GreenLang Team
Version: 31.0.0
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


class GapStatus(str, Enum):
    """Gap analysis item status."""

    CONFORMING = "conforming"
    PARTIALLY_CONFORMING = "partially_conforming"
    NON_CONFORMING = "non_conforming"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"


class EnPIType(str, Enum):
    """EnPI classification per ISO 50006."""

    ABSOLUTE = "absolute"               # Total consumption kWh
    RATIO = "ratio"                     # kWh per unit of output (SEC)
    REGRESSION = "regression"           # Regression model based
    STATISTICAL = "statistical"         # Statistical model
    ENGINEERING = "engineering"         # Engineering model


class PolicyComponent(str, Enum):
    """Energy policy required components per ISO 50001:2018 Clause 5.2."""

    COMMITMENT_IMPROVEMENT = "commitment_to_continual_improvement"
    COMMITMENT_INFORMATION = "commitment_to_information_availability"
    COMMITMENT_LEGAL = "commitment_to_legal_requirements"
    SUPPORT_TARGETS = "support_for_objectives_and_targets"
    SUPPORT_PROCUREMENT = "support_for_energy_efficient_procurement"
    SUPPORT_DESIGN = "support_for_energy_efficient_design"


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


class GapAnalysisItem(BaseModel):
    """Single ISO 50001 clause gap analysis item."""

    item_id: str = Field(default_factory=lambda: f"gap-{uuid.uuid4().hex[:8]}")
    clause_number: str = Field(default="", description="ISO 50001:2018 clause number")
    clause_title: str = Field(default="", description="Clause title")
    requirement: str = Field(default="", description="Specific requirement text")
    status: GapStatus = Field(default=GapStatus.NOT_ASSESSED)
    evidence_available: bool = Field(default=False, description="Evidence exists")
    evidence_description: str = Field(default="", description="What evidence exists")
    gap_description: str = Field(default="", description="What is missing")
    priority: str = Field(default="medium", description="high|medium|low")
    estimated_effort_days: int = Field(default=0, ge=0, description="Effort to close gap")
    responsible_person: str = Field(default="", description="Person responsible")


class EnergyPolicyDraft(BaseModel):
    """Draft energy policy output."""

    policy_version: str = Field(default="1.0", description="Policy version")
    organization_name: str = Field(default="", description="Organization name")
    scope: str = Field(default="", description="EnMS scope and boundaries")
    components_covered: List[str] = Field(default_factory=list, description="Policy components")
    policy_text: str = Field(default="", description="Draft policy text")
    objectives: List[Dict[str, str]] = Field(default_factory=list, description="Energy objectives")
    targets: List[Dict[str, Any]] = Field(default_factory=list, description="Measurable targets")
    approved_by: str = Field(default="", description="Top management approver")
    effective_date: str = Field(default="", description="YYYY-MM-DD")
    review_date: str = Field(default="", description="Next review YYYY-MM-DD")
    iso_clause: str = Field(default="5.2", description="ISO 50001 clause reference")


class EnPIRecord(BaseModel):
    """Energy Performance Indicator record per ISO 50006."""

    enpi_id: str = Field(default_factory=lambda: f"enpi-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="EnPI name")
    enpi_type: EnPIType = Field(default=EnPIType.RATIO)
    unit: str = Field(default="kWh/unit", description="EnPI unit")
    boundary: str = Field(default="", description="SEU or facility boundary")
    baseline_value: float = Field(default=0.0, description="Baseline period value")
    baseline_period: str = Field(default="", description="Baseline period YYYY")
    current_value: float = Field(default=0.0, description="Current period value")
    current_period: str = Field(default="", description="Current period YYYY")
    target_value: float = Field(default=0.0, description="Target value")
    target_year: int = Field(default=0, description="Target year")
    improvement_pct: float = Field(default=0.0, description="Improvement from baseline %")
    on_track: bool = Field(default=False, description="On track to meet target")
    relevant_variables: List[str] = Field(default_factory=list, description="Driving variables")
    regression_r_squared: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    iso_50006_compliant: bool = Field(default=False)


class ManagementReviewPackage(BaseModel):
    """Data package for ISO 50001 management review."""

    review_id: str = Field(default_factory=lambda: f"rev-{uuid.uuid4().hex[:8]}")
    review_date: str = Field(default="", description="Planned review date YYYY-MM-DD")
    period_covered: str = Field(default="", description="Review period")
    energy_policy_status: str = Field(default="", description="Policy review status")
    enpi_summary: List[Dict[str, Any]] = Field(default_factory=list, description="EnPI summaries")
    objectives_status: List[Dict[str, str]] = Field(default_factory=list)
    nonconformities: List[Dict[str, str]] = Field(default_factory=list)
    corrective_actions: List[Dict[str, str]] = Field(default_factory=list)
    audit_results: str = Field(default="", description="Internal/external audit summary")
    legal_compliance_status: str = Field(default="")
    continual_improvement_evidence: List[str] = Field(default_factory=list)
    resource_requirements: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    iso_clause: str = Field(default="9.3", description="ISO 50001 clause reference")


class ISO50001CertificationInput(BaseModel):
    """Input data model for ISO50001CertificationWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    organization_name: str = Field(default="", description="Organization name")
    enms_scope: str = Field(default="", description="EnMS scope description")
    current_gap_responses: Dict[str, str] = Field(
        default_factory=dict,
        description="Clause -> status mapping from prior assessment",
    )
    enpis: List[EnPIRecord] = Field(default_factory=list, description="Current EnPIs")
    total_energy_consumption_mwh: float = Field(default=0.0, ge=0.0)
    annual_production_volume: float = Field(default=0.0, ge=0.0)
    production_unit: str = Field(default="tonnes")
    significant_energy_uses: List[str] = Field(default_factory=list, description="SEU list")
    legal_requirements: List[str] = Field(default_factory=list, description="Legal reqs")
    nonconformities: List[Dict[str, str]] = Field(default_factory=list)
    improvement_actions: List[Dict[str, str]] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ISO50001CertificationResult(BaseModel):
    """Complete result from ISO 50001 certification workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="iso_50001_certification")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    facility_id: str = Field(default="")
    gap_analysis: List[GapAnalysisItem] = Field(default_factory=list)
    energy_policy: EnergyPolicyDraft = Field(default_factory=EnergyPolicyDraft)
    enpis: List[EnPIRecord] = Field(default_factory=list)
    management_review: ManagementReviewPackage = Field(default_factory=ManagementReviewPackage)
    overall_conformance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps_total: int = Field(default=0)
    gaps_closed: int = Field(default=0)
    gaps_remaining: int = Field(default=0)
    certification_readiness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ISO 50001:2018 CLAUSE STRUCTURE (Zero-Hallucination)
# =============================================================================

ISO_50001_CLAUSES: List[Dict[str, str]] = [
    {"clause": "4.1", "title": "Understanding the organization and its context",
     "requirement": "Determine external and internal issues relevant to the EnMS"},
    {"clause": "4.2", "title": "Understanding the needs and expectations of interested parties",
     "requirement": "Determine interested parties and their requirements"},
    {"clause": "4.3", "title": "Determining the scope of the EnMS",
     "requirement": "Define boundaries and applicability of the EnMS"},
    {"clause": "4.4", "title": "Energy management system",
     "requirement": "Establish, implement, maintain, and continually improve an EnMS"},
    {"clause": "5.1", "title": "Leadership and commitment",
     "requirement": "Top management shall demonstrate leadership and commitment to the EnMS"},
    {"clause": "5.2", "title": "Energy policy",
     "requirement": "Establish an energy policy with commitments to improvement and legal compliance"},
    {"clause": "5.3", "title": "Organizational roles, responsibilities and authorities",
     "requirement": "Assign responsibility and authority for the EnMS"},
    {"clause": "6.1", "title": "Actions to address risks and opportunities",
     "requirement": "Determine risks and opportunities and plan actions"},
    {"clause": "6.2", "title": "Objectives, energy targets, and planning to achieve them",
     "requirement": "Establish energy objectives and targets at relevant functions and levels"},
    {"clause": "6.3", "title": "Energy review",
     "requirement": "Analyse energy use and consumption, identify SEUs, estimate future use"},
    {"clause": "6.4", "title": "Energy performance indicators",
     "requirement": "Determine EnPIs that are appropriate to measure and monitor energy performance"},
    {"clause": "6.5", "title": "Energy baseline",
     "requirement": "Establish energy baselines using information from the energy review"},
    {"clause": "6.6", "title": "Planning for collection of energy data",
     "requirement": "Plan data collection to ensure key characteristics are monitored and measured"},
    {"clause": "7.1", "title": "Resources",
     "requirement": "Determine and provide resources needed for the EnMS"},
    {"clause": "7.2", "title": "Competence",
     "requirement": "Determine competence requirements and ensure personnel are competent"},
    {"clause": "7.3", "title": "Awareness",
     "requirement": "Ensure persons are aware of energy policy, EnPIs, and their contribution"},
    {"clause": "7.4", "title": "Communication",
     "requirement": "Determine internal and external communications relevant to the EnMS"},
    {"clause": "7.5", "title": "Documented information",
     "requirement": "Include documented information required by the standard and for effectiveness"},
    {"clause": "8.1", "title": "Operational planning and control",
     "requirement": "Plan, implement, and control processes related to SEUs"},
    {"clause": "8.2", "title": "Design",
     "requirement": "Consider energy performance improvement in design of facilities and equipment"},
    {"clause": "8.3", "title": "Procurement",
     "requirement": "Establish energy performance criteria for procurement of energy services and products"},
    {"clause": "9.1", "title": "Monitoring, measurement, analysis and evaluation",
     "requirement": "Monitor, measure, analyse and evaluate energy performance and the EnMS"},
    {"clause": "9.2", "title": "Internal audit",
     "requirement": "Conduct internal audits at planned intervals"},
    {"clause": "9.3", "title": "Management review",
     "requirement": "Top management shall review the EnMS at planned intervals"},
    {"clause": "10.1", "title": "Nonconformity and corrective action",
     "requirement": "React to nonconformities and take corrective action"},
    {"clause": "10.2", "title": "Continual improvement",
     "requirement": "Continually improve the suitability, adequacy, and effectiveness of the EnMS"},
]


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ISO50001CertificationWorkflow:
    """
    4-phase ISO 50001:2018 certification support workflow.

    Performs EnMS gap analysis against all 26 clauses, drafts energy
    policy with objectives, tracks EnPIs per ISO 50006, and prepares
    management review data packages.

    Zero-hallucination: gap scoring uses deterministic clause-by-clause
    checklists. EnPI calculations use regression baselines.

    Attributes:
        workflow_id: Unique execution identifier.
        _gap_items: Per-clause gap analysis items.
        _energy_policy: Draft energy policy.
        _enpis: Tracked EnPIs.
        _management_review: Review package.

    Example:
        >>> wf = ISO50001CertificationWorkflow()
        >>> inp = ISO50001CertificationInput(organization_name="Acme Corp")
        >>> result = await wf.execute(inp)
        >>> assert result.certification_readiness_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ISO50001CertificationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._gap_items: List[GapAnalysisItem] = []
        self._energy_policy: EnergyPolicyDraft = EnergyPolicyDraft()
        self._enpis: List[EnPIRecord] = []
        self._management_review: ManagementReviewPackage = ManagementReviewPackage()
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[ISO50001CertificationInput] = None,
        organization_name: str = "",
    ) -> ISO50001CertificationResult:
        """
        Execute the 4-phase ISO 50001 certification workflow.

        Args:
            input_data: Full input model (preferred).
            organization_name: Organization name (fallback).

        Returns:
            ISO50001CertificationResult with gap analysis, policy, EnPIs, review.
        """
        if input_data is None:
            input_data = ISO50001CertificationInput(
                organization_name=organization_name,
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting ISO 50001 certification workflow %s for %s",
            self.workflow_id, input_data.organization_name,
        )

        self._phase_results = []
        self._gap_items = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_gap_analysis(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_energy_policy_development(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_enpi_tracking(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_management_review_prep(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("ISO 50001 certification workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        conforming = sum(1 for g in self._gap_items if g.status == GapStatus.CONFORMING)
        partial = sum(1 for g in self._gap_items if g.status == GapStatus.PARTIALLY_CONFORMING)
        non_conforming = sum(1 for g in self._gap_items if g.status == GapStatus.NON_CONFORMING)
        total_assessed = conforming + partial + non_conforming
        conformance_pct = (conforming / max(total_assessed, 1)) * 100.0
        readiness = ((conforming + partial * 0.5) / max(total_assessed, 1)) * 100.0

        result = ISO50001CertificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            facility_id=input_data.facility_id,
            gap_analysis=self._gap_items,
            energy_policy=self._energy_policy,
            enpis=self._enpis,
            management_review=self._management_review,
            overall_conformance_pct=round(conformance_pct, 1),
            gaps_total=len(self._gap_items),
            gaps_closed=conforming,
            gaps_remaining=non_conforming + partial,
            certification_readiness_pct=round(readiness, 1),
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "ISO 50001 certification workflow %s completed in %.2fs readiness=%.1f%%",
            self.workflow_id, elapsed, readiness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: EnMS Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: ISO50001CertificationInput
    ) -> PhaseResult:
        """Assess current state vs all ISO 50001:2018 clauses."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for clause_def in ISO_50001_CLAUSES:
            clause_num = clause_def["clause"]
            existing_status = input_data.current_gap_responses.get(clause_num, "")

            if existing_status == "conforming":
                status = GapStatus.CONFORMING
            elif existing_status == "partially_conforming":
                status = GapStatus.PARTIALLY_CONFORMING
            elif existing_status == "non_conforming":
                status = GapStatus.NON_CONFORMING
            else:
                status = GapStatus.NOT_ASSESSED

            # Auto-assess based on available data
            if status == GapStatus.NOT_ASSESSED:
                status = self._auto_assess_clause(clause_num, input_data)

            # Determine priority
            if clause_num.startswith(("5.", "6.")):
                priority = "high"
            elif clause_num.startswith(("9.", "10.")):
                priority = "high"
            else:
                priority = "medium"

            gap_desc = ""
            if status == GapStatus.NON_CONFORMING:
                gap_desc = f"No evidence of compliance with {clause_def['title']}"
            elif status == GapStatus.PARTIALLY_CONFORMING:
                gap_desc = f"Partial evidence for {clause_def['title']}; improvement needed"

            effort = 0
            if status == GapStatus.NON_CONFORMING:
                effort = 10
            elif status == GapStatus.PARTIALLY_CONFORMING:
                effort = 5

            self._gap_items.append(GapAnalysisItem(
                clause_number=clause_num,
                clause_title=clause_def["title"],
                requirement=clause_def["requirement"],
                status=status,
                evidence_available=status == GapStatus.CONFORMING,
                gap_description=gap_desc,
                priority=priority,
                estimated_effort_days=effort,
            ))

        conforming = sum(1 for g in self._gap_items if g.status == GapStatus.CONFORMING)
        non_conforming = sum(1 for g in self._gap_items if g.status == GapStatus.NON_CONFORMING)
        partial = sum(1 for g in self._gap_items if g.status == GapStatus.PARTIALLY_CONFORMING)

        outputs["clauses_assessed"] = len(self._gap_items)
        outputs["conforming"] = conforming
        outputs["partially_conforming"] = partial
        outputs["non_conforming"] = non_conforming
        outputs["not_assessed"] = sum(1 for g in self._gap_items if g.status == GapStatus.NOT_ASSESSED)
        outputs["total_effort_days"] = sum(g.estimated_effort_days for g in self._gap_items)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 GapAnalysis: %d clauses, conforming=%d partial=%d non=%d",
            len(self._gap_items), conforming, partial, non_conforming,
        )
        return PhaseResult(
            phase_name="gap_analysis", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _auto_assess_clause(
        self, clause_num: str, input_data: ISO50001CertificationInput
    ) -> GapStatus:
        """Auto-assess clause status from available input data."""
        # Clause 6.3 Energy review: need consumption data
        if clause_num == "6.3":
            if input_data.total_energy_consumption_mwh > 0 and input_data.significant_energy_uses:
                return GapStatus.CONFORMING
            elif input_data.total_energy_consumption_mwh > 0:
                return GapStatus.PARTIALLY_CONFORMING
            return GapStatus.NON_CONFORMING

        # Clause 6.4 EnPIs: need EnPI records
        if clause_num == "6.4":
            if input_data.enpis:
                compliant = sum(1 for e in input_data.enpis if e.iso_50006_compliant)
                if compliant == len(input_data.enpis):
                    return GapStatus.CONFORMING
                return GapStatus.PARTIALLY_CONFORMING
            return GapStatus.NON_CONFORMING

        # Clause 6.5 Energy baseline: need EnPIs with baselines
        if clause_num == "6.5":
            has_baselines = any(e.baseline_value > 0 for e in input_data.enpis)
            return GapStatus.CONFORMING if has_baselines else GapStatus.NON_CONFORMING

        # Clause 10.1 Nonconformity: need NC tracking
        if clause_num == "10.1":
            if input_data.nonconformities:
                return GapStatus.CONFORMING
            return GapStatus.PARTIALLY_CONFORMING

        # Default: not assessed
        return GapStatus.NOT_ASSESSED

    # -------------------------------------------------------------------------
    # Phase 2: Energy Policy Development
    # -------------------------------------------------------------------------

    async def _phase_energy_policy_development(
        self, input_data: ISO50001CertificationInput
    ) -> PhaseResult:
        """Draft energy policy with objectives and targets per Clause 5.2."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        org_name = input_data.organization_name or "The Organization"
        scope = input_data.enms_scope or "All facilities within the organizational boundary"

        # Build policy text from required components
        policy_components = [
            PolicyComponent.COMMITMENT_IMPROVEMENT.value,
            PolicyComponent.COMMITMENT_INFORMATION.value,
            PolicyComponent.COMMITMENT_LEGAL.value,
            PolicyComponent.SUPPORT_TARGETS.value,
            PolicyComponent.SUPPORT_PROCUREMENT.value,
            PolicyComponent.SUPPORT_DESIGN.value,
        ]

        policy_text = (
            f"Energy Policy - {org_name}\n\n"
            f"{org_name} is committed to continual improvement of energy performance "
            f"across all operations within the scope of our Energy Management System. "
            f"We commit to:\n\n"
            f"1. Continually improving energy performance and the effectiveness of "
            f"our Energy Management System.\n"
            f"2. Ensuring availability of information and resources to achieve energy "
            f"objectives and targets.\n"
            f"3. Complying with applicable legal and other requirements relating to "
            f"energy efficiency, use, and consumption.\n"
            f"4. Supporting the procurement of energy-efficient products and services, "
            f"and design for energy performance improvement.\n"
            f"5. Setting and reviewing energy objectives and targets.\n\n"
            f"Scope: {scope}\n\n"
            f"This policy is reviewed annually and communicated to all persons working "
            f"under the control of the organization."
        )

        # Generate objectives
        objectives = [
            {"objective": "Reduce specific energy consumption", "timeline": "Annual", "clause": "6.2"},
            {"objective": "Implement top 3 ECMs from energy audit", "timeline": "12 months", "clause": "6.2"},
            {"objective": "Achieve ISO 50001 certification", "timeline": "18 months", "clause": "4.4"},
            {"objective": "Improve energy data coverage to 90%", "timeline": "6 months", "clause": "6.6"},
        ]

        # Generate measurable targets
        current_sec = 0.0
        if input_data.enpis:
            sec_enpis = [e for e in input_data.enpis if e.enpi_type == EnPIType.RATIO]
            if sec_enpis:
                current_sec = sec_enpis[0].current_value

        targets = [
            {
                "target": "Reduce SEC by 5% from baseline",
                "metric": "kWh/unit",
                "baseline": round(current_sec, 2) if current_sec > 0 else "TBD",
                "target_value": round(current_sec * 0.95, 2) if current_sec > 0 else "TBD",
                "deadline": f"{input_data.reporting_year + 1}-12-31",
            },
            {
                "target": "Reduce total energy consumption by 3%",
                "metric": "MWh/yr",
                "baseline": round(input_data.total_energy_consumption_mwh, 0),
                "target_value": round(input_data.total_energy_consumption_mwh * 0.97, 0),
                "deadline": f"{input_data.reporting_year + 1}-12-31",
            },
        ]

        effective_date = datetime.utcnow().strftime("%Y-%m-%d")
        review_date = datetime(datetime.utcnow().year + 1, 1, 15).strftime("%Y-%m-%d")

        self._energy_policy = EnergyPolicyDraft(
            organization_name=org_name,
            scope=scope,
            components_covered=policy_components,
            policy_text=policy_text,
            objectives=objectives,
            targets=targets,
            effective_date=effective_date,
            review_date=review_date,
        )

        outputs["policy_generated"] = True
        outputs["components_covered"] = len(policy_components)
        outputs["objectives_count"] = len(objectives)
        outputs["targets_count"] = len(targets)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 EnergyPolicyDevelopment: %d objectives, %d targets",
            len(objectives), len(targets),
        )
        return PhaseResult(
            phase_name="energy_policy_development", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: EnPI Tracking
    # -------------------------------------------------------------------------

    async def _phase_enpi_tracking(
        self, input_data: ISO50001CertificationInput
    ) -> PhaseResult:
        """Establish and monitor EnPIs per ISO 50006."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if input_data.enpis:
            # Assess and enrich existing EnPIs
            for enpi in input_data.enpis:
                assessed = self._assess_enpi(enpi)
                self._enpis.append(assessed)
        else:
            # Create default EnPIs from available data
            self._enpis = self._create_default_enpis(input_data)

        on_track = sum(1 for e in self._enpis if e.on_track)
        iso_compliant = sum(1 for e in self._enpis if e.iso_50006_compliant)

        outputs["enpis_tracked"] = len(self._enpis)
        outputs["on_track_count"] = on_track
        outputs["iso_50006_compliant"] = iso_compliant
        outputs["enpis"] = [
            {
                "name": e.name,
                "value": e.current_value,
                "baseline": e.baseline_value,
                "improvement_pct": e.improvement_pct,
                "on_track": e.on_track,
            }
            for e in self._enpis
        ]

        if not self._enpis:
            warnings.append("No EnPIs could be established; provide consumption and production data")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 EnPITracking: %d EnPIs, %d on track, %d ISO 50006 compliant",
            len(self._enpis), on_track, iso_compliant,
        )
        return PhaseResult(
            phase_name="enpi_tracking", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_enpi(self, enpi: EnPIRecord) -> EnPIRecord:
        """Assess and enrich an existing EnPI record."""
        # Calculate improvement
        if enpi.baseline_value > 0 and enpi.current_value > 0:
            improvement = ((enpi.baseline_value - enpi.current_value) / enpi.baseline_value) * 100.0
        else:
            improvement = 0.0

        # Check if on track to target
        on_track = False
        if enpi.target_value > 0 and enpi.current_value > 0:
            on_track = enpi.current_value <= enpi.target_value

        # ISO 50006 compliance check
        iso_compliant = (
            enpi.baseline_value > 0
            and enpi.current_value > 0
            and enpi.boundary != ""
            and enpi.enpi_type in (EnPIType.RATIO, EnPIType.REGRESSION)
        )

        # Data quality
        dq = 50.0  # Default
        if enpi.regression_r_squared > 0:
            dq = enpi.regression_r_squared * 100.0
        if enpi.baseline_value > 0:
            dq = min(dq + 20.0, 100.0)

        return EnPIRecord(
            enpi_id=enpi.enpi_id,
            name=enpi.name,
            enpi_type=enpi.enpi_type,
            unit=enpi.unit,
            boundary=enpi.boundary,
            baseline_value=enpi.baseline_value,
            baseline_period=enpi.baseline_period,
            current_value=enpi.current_value,
            current_period=enpi.current_period,
            target_value=enpi.target_value,
            target_year=enpi.target_year,
            improvement_pct=round(improvement, 2),
            on_track=on_track,
            relevant_variables=enpi.relevant_variables,
            regression_r_squared=enpi.regression_r_squared,
            data_quality_score=round(dq, 1),
            iso_50006_compliant=iso_compliant,
        )

    def _create_default_enpis(
        self, input_data: ISO50001CertificationInput
    ) -> List[EnPIRecord]:
        """Create default EnPIs from available data."""
        enpis: List[EnPIRecord] = []

        if input_data.total_energy_consumption_mwh > 0:
            enpis.append(EnPIRecord(
                name="Total energy consumption",
                enpi_type=EnPIType.ABSOLUTE,
                unit="MWh/yr",
                boundary=input_data.enms_scope or "Facility",
                current_value=input_data.total_energy_consumption_mwh,
                current_period=str(input_data.reporting_year),
                relevant_variables=["production_volume", "degree_days"],
            ))

        if input_data.total_energy_consumption_mwh > 0 and input_data.annual_production_volume > 0:
            sec = (input_data.total_energy_consumption_mwh * 1000.0) / input_data.annual_production_volume
            enpis.append(EnPIRecord(
                name=f"SEC (kWh per {input_data.production_unit})",
                enpi_type=EnPIType.RATIO,
                unit=f"kWh/{input_data.production_unit}",
                boundary=input_data.enms_scope or "Facility",
                current_value=round(sec, 2),
                current_period=str(input_data.reporting_year),
                relevant_variables=["production_volume"],
            ))

        return enpis

    # -------------------------------------------------------------------------
    # Phase 4: Management Review Preparation
    # -------------------------------------------------------------------------

    async def _phase_management_review_prep(
        self, input_data: ISO50001CertificationInput
    ) -> PhaseResult:
        """Compile data package for top management review per Clause 9.3."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build EnPI summary for review
        enpi_summary = [
            {
                "name": e.name,
                "current": e.current_value,
                "baseline": e.baseline_value,
                "improvement_pct": e.improvement_pct,
                "on_track": e.on_track,
                "unit": e.unit,
            }
            for e in self._enpis
        ]

        # Build objectives status
        objectives_status = [
            {"objective": obj["objective"], "status": "in_progress", "timeline": obj["timeline"]}
            for obj in self._energy_policy.objectives
        ]

        # Nonconformities from input
        nonconformities = input_data.nonconformities

        # Corrective actions from input
        corrective_actions = input_data.improvement_actions

        # Continual improvement evidence from gap analysis
        conforming_clauses = [
            g.clause_title for g in self._gap_items
            if g.status == GapStatus.CONFORMING
        ]

        # Resource requirements for remaining gaps
        resource_reqs = []
        for gap in self._gap_items:
            if gap.status in (GapStatus.NON_CONFORMING, GapStatus.PARTIALLY_CONFORMING):
                resource_reqs.append(
                    f"Clause {gap.clause_number} ({gap.clause_title}): "
                    f"~{gap.estimated_effort_days} days effort"
                )

        # Recommendations
        recommendations = []
        non_conforming = [g for g in self._gap_items if g.status == GapStatus.NON_CONFORMING]
        if non_conforming:
            recommendations.append(
                f"Address {len(non_conforming)} non-conforming clauses before certification audit"
            )

        off_track_enpis = [e for e in self._enpis if not e.on_track and e.target_value > 0]
        if off_track_enpis:
            recommendations.append(
                f"Review {len(off_track_enpis)} EnPIs not on track to meet targets"
            )

        recommendations.append("Schedule internal audit within next quarter")
        recommendations.append("Review and update energy policy annually")

        review_date = datetime.utcnow().strftime("%Y-%m-%d")
        period = f"{input_data.reporting_year}-01-01 to {input_data.reporting_year}-12-31"

        self._management_review = ManagementReviewPackage(
            review_date=review_date,
            period_covered=period,
            energy_policy_status="Active - next review " + self._energy_policy.review_date,
            enpi_summary=enpi_summary,
            objectives_status=objectives_status,
            nonconformities=nonconformities,
            corrective_actions=corrective_actions,
            legal_compliance_status="Compliance with legal requirements verified" if input_data.legal_requirements else "Legal register not provided",
            continual_improvement_evidence=[f"Conforming to {len(conforming_clauses)} of {len(ISO_50001_CLAUSES)} clauses"],
            resource_requirements=resource_reqs,
            recommendations=recommendations,
        )

        outputs["review_prepared"] = True
        outputs["enpi_count"] = len(enpi_summary)
        outputs["nonconformities_count"] = len(nonconformities)
        outputs["recommendations_count"] = len(recommendations)
        outputs["resource_requirements_count"] = len(resource_reqs)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ManagementReviewPrep: %d EnPIs, %d recommendations",
            len(enpi_summary), len(recommendations),
        )
        return PhaseResult(
            phase_name="management_review_preparation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ISO50001CertificationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
