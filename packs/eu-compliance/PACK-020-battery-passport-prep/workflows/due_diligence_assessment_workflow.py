# -*- coding: utf-8 -*-
"""
Due Diligence Assessment Workflow
=======================================

4-phase workflow for supply chain due diligence assessment per EU Battery
Regulation 2023/1542, Article 48 and Annex X. Implements supplier mapping,
risk assessment, mitigation planning, and audit verification for battery
raw material supply chains.

Phases:
    1. SupplierMapping       -- Map upstream supply chain and identify tiers
    2. RiskAssessment        -- Assess ESG and regulatory risks per supplier
    3. MitigationPlanning    -- Plan risk mitigation actions
    4. AuditVerification     -- Verify through third-party audits

Regulatory references:
    - EU Regulation 2023/1542 Art. 48 (due diligence policies)
    - EU Regulation 2023/1542 Annex X (due diligence requirements)
    - OECD Due Diligence Guidance for Responsible Supply Chains
    - EU Conflict Minerals Regulation 2017/821
    - UN Guiding Principles on Business and Human Rights

Author: GreenLang Team
Version: 1.0.0
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

_MODULE_VERSION = "1.0.0"


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
    """Phases of the due diligence assessment workflow."""
    SUPPLIER_MAPPING = "supplier_mapping"
    RISK_ASSESSMENT = "risk_assessment"
    MITIGATION_PLANNING = "mitigation_planning"
    AUDIT_VERIFICATION = "audit_verification"


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


class SupplierTier(str, Enum):
    """Supply chain tier classification."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4_PLUS = "tier_4_plus"
    UNKNOWN = "unknown"


class RiskCategory(str, Enum):
    """Due diligence risk categories per Annex X."""
    HUMAN_RIGHTS = "human_rights"
    CHILD_LABOUR = "child_labour"
    FORCED_LABOUR = "forced_labour"
    ENVIRONMENTAL = "environmental"
    CONFLICT_MINERALS = "conflict_minerals"
    CORRUPTION = "corruption"
    HEALTH_SAFETY = "health_safety"
    INDIGENOUS_RIGHTS = "indigenous_rights"
    WATER_POLLUTION = "water_pollution"
    DEFORESTATION = "deforestation"


class RiskLevel(str, Enum):
    """Risk severity level."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"
    NOT_ASSESSED = "not_assessed"


class MitigationStatus(str, Enum):
    """Status of risk mitigation action."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


class AuditOutcome(str, Enum):
    """Third-party audit outcome."""
    CONFORMANT = "conformant"
    MINOR_NON_CONFORMANCE = "minor_non_conformance"
    MAJOR_NON_CONFORMANCE = "major_non_conformance"
    NOT_AUDITED = "not_audited"


# =============================================================================
# HIGH-RISK REGIONS AND MATERIALS
# =============================================================================

HIGH_RISK_REGIONS: List[str] = [
    "COD", "COG", "RWA", "UGA", "TZA", "BDI", "ZMB",  # Great Lakes
    "MMR", "CHN_XJ",  # Human rights risk regions
]

CONFLICT_MINERALS: List[str] = [
    "cobalt", "tantalum", "tin", "tungsten", "gold", "lithium",
]

# Risk score weights for composite calculation
RISK_WEIGHTS: Dict[str, float] = {
    RiskCategory.HUMAN_RIGHTS.value: 0.20,
    RiskCategory.CHILD_LABOUR.value: 0.20,
    RiskCategory.FORCED_LABOUR.value: 0.15,
    RiskCategory.ENVIRONMENTAL.value: 0.15,
    RiskCategory.CONFLICT_MINERALS.value: 0.10,
    RiskCategory.CORRUPTION.value: 0.05,
    RiskCategory.HEALTH_SAFETY.value: 0.05,
    RiskCategory.INDIGENOUS_RIGHTS.value: 0.05,
    RiskCategory.WATER_POLLUTION.value: 0.03,
    RiskCategory.DEFORESTATION.value: 0.02,
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
    """Supply chain participant record."""
    supplier_id: str = Field(default_factory=lambda: f"sup-{_new_uuid()[:8]}")
    supplier_name: str = Field(..., description="Supplier legal name")
    country_code: str = Field(default="", description="ISO 3166-1 alpha-3")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    materials_supplied: List[str] = Field(
        default_factory=list, description="Raw materials supplied"
    )
    certification: str = Field(
        default="", description="e.g., RMI, IRMA, SA8000, ISO 14001"
    )
    audit_date: str = Field(default="", description="Last audit ISO date")
    has_due_diligence_policy: bool = Field(default=False)
    annual_volume_kg: float = Field(default=0.0, ge=0.0)
    is_critical: bool = Field(default=False, description="Single-source critical supplier")


class RiskAssessmentResult(BaseModel):
    """Risk assessment for a specific supplier."""
    assessment_id: str = Field(default_factory=lambda: f"ra-{_new_uuid()[:8]}")
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    risk_scores: Dict[str, float] = Field(
        default_factory=dict, description="Risk score 0-5 per category"
    )
    composite_risk_score: float = Field(default=0.0, ge=0.0, le=5.0)
    risk_level: RiskLevel = Field(default=RiskLevel.NOT_ASSESSED)
    high_risk_flags: List[str] = Field(default_factory=list)
    in_high_risk_region: bool = Field(default=False)
    conflict_mineral_exposure: bool = Field(default=False)


class MitigationAction(BaseModel):
    """Risk mitigation action plan."""
    action_id: str = Field(default_factory=lambda: f"mit-{_new_uuid()[:8]}")
    supplier_id: str = Field(default="")
    risk_category: str = Field(default="")
    description: str = Field(default="")
    status: MitigationStatus = Field(default=MitigationStatus.PLANNED)
    priority: str = Field(default="medium")
    deadline: str = Field(default="")
    responsible_party: str = Field(default="")
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)


class AuditRecord(BaseModel):
    """Third-party audit record."""
    audit_id: str = Field(default_factory=lambda: f"aud-{_new_uuid()[:8]}")
    supplier_id: str = Field(default="")
    auditor_name: str = Field(default="")
    audit_standard: str = Field(default="")
    audit_date: str = Field(default="")
    outcome: AuditOutcome = Field(default=AuditOutcome.NOT_AUDITED)
    findings_count: int = Field(default=0, ge=0)
    critical_findings: int = Field(default=0, ge=0)
    corrective_actions_required: int = Field(default=0, ge=0)


class DueDiligenceInput(BaseModel):
    """Input data model for DueDiligenceAssessmentWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    suppliers: List[SupplierRecord] = Field(default_factory=list)
    existing_audits: List[AuditRecord] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    risk_threshold: float = Field(
        default=3.0, ge=0.0, le=5.0,
        description="Risk score threshold for mitigation requirement"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class DueDiligenceResult(BaseModel):
    """Complete result from due diligence assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="due_diligence_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    suppliers_mapped: int = Field(default=0, ge=0)
    risk_assessments: List[RiskAssessmentResult] = Field(default_factory=list)
    mitigation_actions: List[MitigationAction] = Field(default_factory=list)
    audit_records: List[AuditRecord] = Field(default_factory=list)
    high_risk_supplier_count: int = Field(default=0, ge=0)
    avg_risk_score: float = Field(default=0.0, ge=0.0, le=5.0)
    supply_chain_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    due_diligence_compliant: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DueDiligenceAssessmentWorkflow:
    """
    4-phase supply chain due diligence assessment workflow per EU Battery Regulation.

    Implements due diligence per EU Regulation 2023/1542 Art. 48 and Annex X,
    aligned with OECD Due Diligence Guidance. Maps supplier tiers, assesses
    ESG and regulatory risks with weighted scoring, creates mitigation action
    plans, and tracks third-party audit verification.

    Zero-hallucination: all risk scores use deterministic weighted averaging
    with documented risk factor weights. No LLM in risk calculation paths.

    Example:
        >>> wf = DueDiligenceAssessmentWorkflow()
        >>> inp = DueDiligenceInput(suppliers=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.due_diligence_compliant in (True, False)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DueDiligenceAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._suppliers: List[SupplierRecord] = []
        self._assessments: List[RiskAssessmentResult] = []
        self._mitigations: List[MitigationAction] = []
        self._audits: List[AuditRecord] = []
        self._compliant: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.SUPPLIER_MAPPING.value, "description": "Map upstream supply chain and identify tiers"},
            {"name": WorkflowPhase.RISK_ASSESSMENT.value, "description": "Assess ESG and regulatory risks per supplier"},
            {"name": WorkflowPhase.MITIGATION_PLANNING.value, "description": "Plan risk mitigation actions"},
            {"name": WorkflowPhase.AUDIT_VERIFICATION.value, "description": "Verify through third-party audits"},
        ]

    def validate_inputs(self, input_data: DueDiligenceInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.suppliers:
            issues.append("No suppliers provided")
        for sup in input_data.suppliers:
            if not sup.supplier_name:
                issues.append(f"Supplier {sup.supplier_id}: missing name")
            if not sup.country_code:
                issues.append(f"Supplier {sup.supplier_id}: missing country code")
        return issues

    async def execute(
        self,
        input_data: Optional[DueDiligenceInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DueDiligenceResult:
        """
        Execute the 4-phase due diligence assessment workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            DueDiligenceResult with risk assessments, mitigation plans, and audit status.
        """
        if input_data is None:
            input_data = DueDiligenceInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting due diligence workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_supplier_mapping(input_data))
            phases_done += 1
            phase_results.append(await self._phase_risk_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_mitigation_planning(input_data))
            phases_done += 1
            phase_results.append(await self._phase_audit_verification(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Due diligence workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        high_risk_count = sum(
            1 for a in self._assessments
            if a.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )
        avg_risk = round(
            sum(a.composite_risk_score for a in self._assessments)
            / len(self._assessments) if self._assessments else 0.0, 2
        )

        # Supply chain coverage: percentage of suppliers with risk assessment
        coverage = round(
            (len(self._assessments) / len(self._suppliers) * 100)
            if self._suppliers else 0.0, 1
        )

        result = DueDiligenceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            suppliers_mapped=len(self._suppliers),
            risk_assessments=self._assessments,
            mitigation_actions=self._mitigations,
            audit_records=self._audits,
            high_risk_supplier_count=high_risk_count,
            avg_risk_score=avg_risk,
            supply_chain_coverage_pct=coverage,
            due_diligence_compliant=self._compliant,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Due diligence %s completed in %.2fs: %d suppliers, %d high-risk, compliant=%s",
            self.workflow_id, elapsed, len(self._suppliers), high_risk_count, self._compliant,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Mapping
    # -------------------------------------------------------------------------

    async def _phase_supplier_mapping(
        self, input_data: DueDiligenceInput,
    ) -> PhaseResult:
        """Map upstream supply chain and classify tiers."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._suppliers = list(input_data.suppliers)

        tier_counts: Dict[str, int] = {}
        country_counts: Dict[str, int] = {}
        material_counts: Dict[str, int] = {}
        critical_count = 0

        for sup in self._suppliers:
            tier_counts[sup.tier.value] = tier_counts.get(sup.tier.value, 0) + 1
            if sup.country_code:
                country_counts[sup.country_code] = country_counts.get(sup.country_code, 0) + 1
            for mat in sup.materials_supplied:
                material_counts[mat] = material_counts.get(mat, 0) + 1
            if sup.is_critical:
                critical_count += 1

        outputs["suppliers_mapped"] = len(self._suppliers)
        outputs["tier_distribution"] = tier_counts
        outputs["country_distribution"] = country_counts
        outputs["material_distribution"] = material_counts
        outputs["critical_suppliers"] = critical_count
        outputs["unique_countries"] = len(country_counts)

        # High-risk region detection
        hr_suppliers = [
            s for s in self._suppliers if s.country_code in HIGH_RISK_REGIONS
        ]
        if hr_suppliers:
            warnings.append(
                f"{len(hr_suppliers)} suppliers in high-risk regions: "
                f"{', '.join(set(s.country_code for s in hr_suppliers))}"
            )

        # Conflict mineral detection
        conflict_exposure = [
            s for s in self._suppliers
            if any(m.lower() in CONFLICT_MINERALS for m in s.materials_supplied)
        ]
        if conflict_exposure:
            warnings.append(
                f"{len(conflict_exposure)} suppliers supply conflict minerals"
            )

        # Check for tier gaps
        if SupplierTier.TIER_1.value not in tier_counts:
            warnings.append("No Tier 1 suppliers mapped")

        if not self._suppliers:
            warnings.append("No suppliers provided; due diligence cannot proceed")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 SupplierMapping: %d suppliers, %d countries",
            len(self._suppliers), len(country_counts),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SUPPLIER_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Risk Assessment
    # -------------------------------------------------------------------------

    async def _phase_risk_assessment(
        self, input_data: DueDiligenceInput,
    ) -> PhaseResult:
        """Assess ESG and regulatory risks per supplier."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._assessments = []

        for supplier in self._suppliers:
            assessment = self._assess_supplier_risk(supplier)
            self._assessments.append(assessment)

        level_counts: Dict[str, int] = {}
        for a in self._assessments:
            level_counts[a.risk_level.value] = level_counts.get(a.risk_level.value, 0) + 1

        avg_score = round(
            sum(a.composite_risk_score for a in self._assessments)
            / len(self._assessments) if self._assessments else 0.0, 2
        )

        outputs["assessments_completed"] = len(self._assessments)
        outputs["risk_level_distribution"] = level_counts
        outputs["avg_composite_score"] = avg_score
        outputs["high_risk_regions_flagged"] = sum(
            1 for a in self._assessments if a.in_high_risk_region
        )
        outputs["conflict_mineral_exposure_count"] = sum(
            1 for a in self._assessments if a.conflict_mineral_exposure
        )

        critical_or_high = [
            a for a in self._assessments
            if a.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]
        if critical_or_high:
            warnings.append(
                f"{len(critical_or_high)} suppliers classified as high/critical risk"
            )
            for a in critical_or_high:
                warnings.append(
                    f"  - {a.supplier_name}: score {a.composite_risk_score}, "
                    f"flags: {', '.join(a.high_risk_flags)}"
                )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 RiskAssessment: %d assessed, avg score %.2f",
            len(self._assessments), avg_score,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_supplier_risk(self, supplier: SupplierRecord) -> RiskAssessmentResult:
        """Deterministic risk assessment for a supplier."""
        in_hr_region = supplier.country_code in HIGH_RISK_REGIONS
        has_conflict = any(
            m.lower() in CONFLICT_MINERALS for m in supplier.materials_supplied
        )

        risk_scores: Dict[str, float] = {}
        flags: List[str] = []

        # Base risk by tier (further from manufacturer = higher base risk)
        tier_base: Dict[str, float] = {
            SupplierTier.TIER_1.value: 1.0,
            SupplierTier.TIER_2.value: 2.0,
            SupplierTier.TIER_3.value: 3.0,
            SupplierTier.TIER_4_PLUS.value: 3.5,
            SupplierTier.UNKNOWN.value: 3.0,
        }
        base = tier_base.get(supplier.tier.value, 2.5)

        # Human rights risk
        hr_score = base + (1.5 if in_hr_region else 0.0)
        hr_score -= 0.5 if supplier.has_due_diligence_policy else 0.0
        hr_score -= 0.5 if supplier.certification else 0.0
        risk_scores[RiskCategory.HUMAN_RIGHTS.value] = min(5.0, max(0.0, hr_score))

        # Child labour risk
        cl_score = base + (2.0 if in_hr_region else 0.0)
        cl_score -= 0.5 if supplier.certification else 0.0
        risk_scores[RiskCategory.CHILD_LABOUR.value] = min(5.0, max(0.0, cl_score))
        if cl_score >= 3.5:
            flags.append("child_labour_risk")

        # Forced labour risk
        fl_score = base + (1.5 if in_hr_region else 0.0)
        fl_score -= 0.5 if supplier.has_due_diligence_policy else 0.0
        risk_scores[RiskCategory.FORCED_LABOUR.value] = min(5.0, max(0.0, fl_score))
        if fl_score >= 3.5:
            flags.append("forced_labour_risk")

        # Environmental risk
        env_score = base + (0.5 if "mining" in " ".join(supplier.materials_supplied).lower() else 0.0)
        env_score -= 0.5 if "ISO 14001" in supplier.certification else 0.0
        risk_scores[RiskCategory.ENVIRONMENTAL.value] = min(5.0, max(0.0, env_score))

        # Conflict minerals risk
        cm_score = 1.0 + (3.0 if has_conflict and in_hr_region else 0.0)
        cm_score += 1.0 if has_conflict and not in_hr_region else 0.0
        risk_scores[RiskCategory.CONFLICT_MINERALS.value] = min(5.0, max(0.0, cm_score))
        if has_conflict and in_hr_region:
            flags.append("conflict_mineral_high_risk_region")

        # Corruption
        corruption_score = base + (1.0 if in_hr_region else 0.0)
        risk_scores[RiskCategory.CORRUPTION.value] = min(5.0, max(0.0, corruption_score))

        # Health and safety
        hs_score = base
        risk_scores[RiskCategory.HEALTH_SAFETY.value] = min(5.0, max(0.0, hs_score))

        # Indigenous rights
        ir_score = base + (1.0 if in_hr_region else 0.0)
        risk_scores[RiskCategory.INDIGENOUS_RIGHTS.value] = min(5.0, max(0.0, ir_score))

        # Water pollution
        wp_score = base * 0.8
        risk_scores[RiskCategory.WATER_POLLUTION.value] = min(5.0, max(0.0, wp_score))

        # Deforestation
        df_score = base * 0.6
        risk_scores[RiskCategory.DEFORESTATION.value] = min(5.0, max(0.0, df_score))

        # Composite score using weights
        composite = sum(
            risk_scores.get(cat, 0.0) * weight
            for cat, weight in RISK_WEIGHTS.items()
        )
        composite = round(min(5.0, max(0.0, composite)), 2)

        # Determine risk level
        if composite >= 4.0:
            level = RiskLevel.CRITICAL
        elif composite >= 3.0:
            level = RiskLevel.HIGH
        elif composite >= 2.0:
            level = RiskLevel.MEDIUM
        elif composite >= 1.0:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NEGLIGIBLE

        if in_hr_region:
            flags.append("high_risk_region")
        if has_conflict:
            flags.append("conflict_mineral_supplier")
        if not supplier.has_due_diligence_policy:
            flags.append("no_due_diligence_policy")
        if not supplier.certification:
            flags.append("no_certification")

        return RiskAssessmentResult(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.supplier_name,
            risk_scores={k: round(v, 2) for k, v in risk_scores.items()},
            composite_risk_score=composite,
            risk_level=level,
            high_risk_flags=flags,
            in_high_risk_region=in_hr_region,
            conflict_mineral_exposure=has_conflict,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Mitigation Planning
    # -------------------------------------------------------------------------

    async def _phase_mitigation_planning(
        self, input_data: DueDiligenceInput,
    ) -> PhaseResult:
        """Plan risk mitigation actions for high-risk suppliers."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._mitigations = []

        threshold = input_data.risk_threshold

        for assessment in self._assessments:
            if assessment.composite_risk_score < threshold:
                continue

            # Generate mitigation actions for each high-scoring risk category
            for cat, score in assessment.risk_scores.items():
                if score < threshold:
                    continue

                priority = "critical" if score >= 4.0 else "high" if score >= 3.0 else "medium"
                action_desc = self._generate_mitigation_description(cat, assessment)

                self._mitigations.append(MitigationAction(
                    supplier_id=assessment.supplier_id,
                    risk_category=cat,
                    description=action_desc,
                    status=MitigationStatus.PLANNED,
                    priority=priority,
                    deadline=f"{input_data.reporting_year + 1}-06-30",
                    responsible_party="Supply Chain Compliance Team",
                    estimated_cost_eur=self._estimate_mitigation_cost(cat, score),
                ))

        priority_counts: Dict[str, int] = {}
        for m in self._mitigations:
            priority_counts[m.priority] = priority_counts.get(m.priority, 0) + 1

        total_cost = sum(m.estimated_cost_eur for m in self._mitigations)

        outputs["mitigation_actions_created"] = len(self._mitigations)
        outputs["priority_distribution"] = priority_counts
        outputs["total_estimated_cost_eur"] = round(total_cost, 2)
        outputs["suppliers_requiring_mitigation"] = len(set(
            m.supplier_id for m in self._mitigations
        ))
        outputs["risk_threshold_applied"] = threshold

        if not self._mitigations:
            outputs["note"] = "No suppliers exceed risk threshold"
        else:
            critical = priority_counts.get("critical", 0)
            if critical > 0:
                warnings.append(
                    f"{critical} critical-priority mitigation actions required"
                )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 MitigationPlanning: %d actions, cost EUR %.0f",
            len(self._mitigations), total_cost,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MITIGATION_PLANNING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_mitigation_description(
        self, category: str, assessment: RiskAssessmentResult,
    ) -> str:
        """Generate deterministic mitigation action description."""
        descriptions: Dict[str, str] = {
            RiskCategory.HUMAN_RIGHTS.value: (
                f"Conduct human rights impact assessment for {assessment.supplier_name}. "
                f"Require adherence to UNGPs and provide evidence of compliance."
            ),
            RiskCategory.CHILD_LABOUR.value: (
                f"Require {assessment.supplier_name} to implement child labour monitoring "
                f"programme and provide third-party verification."
            ),
            RiskCategory.FORCED_LABOUR.value: (
                f"Require {assessment.supplier_name} to implement forced labour due diligence "
                f"per ILO conventions and provide worker interview evidence."
            ),
            RiskCategory.ENVIRONMENTAL.value: (
                f"Require {assessment.supplier_name} to obtain ISO 14001 certification "
                f"and submit environmental impact assessment."
            ),
            RiskCategory.CONFLICT_MINERALS.value: (
                f"Require {assessment.supplier_name} to participate in RMI RMAP audit "
                f"and provide chain of custody documentation."
            ),
            RiskCategory.CORRUPTION.value: (
                f"Require {assessment.supplier_name} to implement anti-corruption policy "
                f"aligned with OECD guidelines."
            ),
            RiskCategory.HEALTH_SAFETY.value: (
                f"Require {assessment.supplier_name} to implement OHS management system "
                f"per ISO 45001 and provide incident reports."
            ),
            RiskCategory.INDIGENOUS_RIGHTS.value: (
                f"Require FPIC documentation from {assessment.supplier_name} for operations "
                f"affecting indigenous communities."
            ),
            RiskCategory.WATER_POLLUTION.value: (
                f"Require {assessment.supplier_name} to conduct water quality monitoring "
                f"and submit discharge compliance reports."
            ),
            RiskCategory.DEFORESTATION.value: (
                f"Require {assessment.supplier_name} to provide satellite-verified "
                f"zero-deforestation commitment."
            ),
        }
        return descriptions.get(category, f"Address {category} risk for {assessment.supplier_name}.")

    def _estimate_mitigation_cost(self, category: str, score: float) -> float:
        """Estimate mitigation cost based on category and severity."""
        base_costs: Dict[str, float] = {
            RiskCategory.HUMAN_RIGHTS.value: 15000.0,
            RiskCategory.CHILD_LABOUR.value: 20000.0,
            RiskCategory.FORCED_LABOUR.value: 18000.0,
            RiskCategory.ENVIRONMENTAL.value: 25000.0,
            RiskCategory.CONFLICT_MINERALS.value: 30000.0,
            RiskCategory.CORRUPTION.value: 10000.0,
            RiskCategory.HEALTH_SAFETY.value: 12000.0,
            RiskCategory.INDIGENOUS_RIGHTS.value: 15000.0,
            RiskCategory.WATER_POLLUTION.value: 20000.0,
            RiskCategory.DEFORESTATION.value: 18000.0,
        }
        base = base_costs.get(category, 10000.0)
        severity_multiplier = score / 3.0
        return round(base * severity_multiplier, 2)

    # -------------------------------------------------------------------------
    # Phase 4: Audit Verification
    # -------------------------------------------------------------------------

    async def _phase_audit_verification(
        self, input_data: DueDiligenceInput,
    ) -> PhaseResult:
        """Verify due diligence through third-party audits."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._audits = list(input_data.existing_audits)

        # Determine which suppliers have audits
        audited_supplier_ids = set(a.supplier_id for a in self._audits)
        supplier_ids = set(s.supplier_id for s in self._suppliers)
        unaudited = supplier_ids - audited_supplier_ids
        audit_coverage_pct = round(
            (len(audited_supplier_ids & supplier_ids) / len(supplier_ids) * 100)
            if supplier_ids else 0.0, 1
        )

        # Audit outcome summary
        outcome_counts: Dict[str, int] = {}
        for audit in self._audits:
            outcome_counts[audit.outcome.value] = (
                outcome_counts.get(audit.outcome.value, 0) + 1
            )

        # Check conformance for high-risk suppliers
        high_risk_ids = set(
            a.supplier_id for a in self._assessments
            if a.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )
        high_risk_audited = high_risk_ids & audited_supplier_ids
        high_risk_unaudited = high_risk_ids - audited_supplier_ids

        # Determine overall compliance
        has_policy = len(self._suppliers) > 0
        has_risk_assessment = len(self._assessments) > 0
        has_mitigation = len(self._mitigations) > 0 or not high_risk_ids
        has_audit_coverage = audit_coverage_pct >= 50.0 or not self._suppliers
        no_unmitigated_critical = not high_risk_unaudited or len(high_risk_unaudited) == 0

        self._compliant = all([
            has_policy,
            has_risk_assessment,
            has_mitigation,
            has_audit_coverage or not high_risk_ids,
        ])

        outputs["total_audits"] = len(self._audits)
        outputs["audit_coverage_pct"] = audit_coverage_pct
        outputs["outcome_distribution"] = outcome_counts
        outputs["unaudited_suppliers"] = len(unaudited)
        outputs["high_risk_audited"] = len(high_risk_audited)
        outputs["high_risk_unaudited"] = len(high_risk_unaudited)
        outputs["due_diligence_compliant"] = self._compliant

        if high_risk_unaudited:
            warnings.append(
                f"{len(high_risk_unaudited)} high-risk suppliers have not been audited"
            )

        if audit_coverage_pct < 50:
            warnings.append(
                f"Audit coverage is {audit_coverage_pct}% (target: 50%+)"
            )

        major_nc = sum(
            1 for a in self._audits
            if a.outcome == AuditOutcome.MAJOR_NON_CONFORMANCE
        )
        if major_nc > 0:
            warnings.append(
                f"{major_nc} audits resulted in major non-conformance"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 AuditVerification: coverage %.1f%%, compliant=%s",
            audit_coverage_pct, self._compliant,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.AUDIT_VERIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: DueDiligenceResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
