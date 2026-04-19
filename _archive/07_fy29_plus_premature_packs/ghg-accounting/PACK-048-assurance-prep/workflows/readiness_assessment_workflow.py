# -*- coding: utf-8 -*-
"""
Readiness Assessment Workflow
====================================

5-phase workflow for GHG assurance readiness assessment covering standard
selection, checklist generation, evidence checking, score calculation, and
gap reporting within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. StandardSelection          -- User selects the target assurance standard
                                     (ISAE 3410 / ISO 14064-3 / AA1000AS v3),
                                     and the workflow loads the corresponding
                                     requirement set, evidence expectations,
                                     and scoring criteria from the standard
                                     register.
    2. ChecklistGeneration        -- Generates a standard-specific checklist
                                     covering governance, data management,
                                     methodology documentation, internal
                                     controls, evidence availability, and
                                     disclosure requirements for the selected
                                     assurance standard.
    3. EvidenceCheck              -- Checks existing evidence items against
                                     checklist requirements, classifying each
                                     as AVAILABLE / PARTIAL / MISSING, and
                                     recording the evidence source, timestamp,
                                     and quality grade.
    4. ScoreCalculation           -- Calculates weighted readiness scores per
                                     category and an overall composite readiness
                                     score using Decimal arithmetic with
                                     ROUND_HALF_UP, based on category weights
                                     defined by the assurance standard.
    5. GapReporting               -- Produces a gap report with prioritised
                                     remediation recommendations, estimated
                                     effort per gap, owner assignment guidance,
                                     and a timeline to assurance-readiness.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Assurance Engagements on GHG Statements
    ISO 14064-3:2019 - Specification for validation/verification of GHG
    AA1000AS v3 (2020) - AccountAbility Assurance Standard
    ESRS E1 (2024) - Climate change disclosure assurance
    SEC Climate Disclosure Rules (2024) - Attestation requirements
    CSRD (2022/2464) - Mandatory assurance of sustainability reporting
    PCAF Global Standard (2022) - Financed emissions data quality

Schedule: Annually or upon assurance engagement initiation
Estimated duration: 2-4 weeks depending on evidence availability

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

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

class ReadinessPhase(str, Enum):
    """Readiness assessment workflow phases."""

    STANDARD_SELECTION = "standard_selection"
    CHECKLIST_GENERATION = "checklist_generation"
    EVIDENCE_CHECK = "evidence_check"
    SCORE_CALCULATION = "score_calculation"
    GAP_REPORTING = "gap_reporting"

class AssuranceStandard(str, Enum):
    """Supported assurance standards."""

    ISAE_3410 = "isae_3410"
    ISO_14064_3 = "iso_14064_3"
    AA1000AS_V3 = "aa1000as_v3"

class AssuranceLevel(str, Enum):
    """Assurance engagement level."""

    LIMITED = "limited"
    REASONABLE = "reasonable"

class EvidenceStatus(str, Enum):
    """Status of evidence against a checklist requirement."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"

class ReadinessBand(str, Enum):
    """Readiness classification band."""

    READY = "ready"
    NEARLY_READY = "nearly_ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"

class GapPriority(str, Enum):
    """Priority classification for remediation gaps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ChecklistCategory(str, Enum):
    """Categories within an assurance readiness checklist."""

    GOVERNANCE = "governance"
    DATA_MANAGEMENT = "data_management"
    METHODOLOGY = "methodology"
    INTERNAL_CONTROLS = "internal_controls"
    EVIDENCE_AVAILABILITY = "evidence_availability"
    DISCLOSURE_REQUIREMENTS = "disclosure_requirements"
    SCOPE_COMPLETENESS = "scope_completeness"
    UNCERTAINTY_MANAGEMENT = "uncertainty_management"

# =============================================================================
# STANDARD-SPECIFIC REFERENCE DATA (Zero-Hallucination)
# =============================================================================

STANDARD_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "isae_3410": {
        "full_name": "International Standard on Assurance Engagements 3410",
        "issuer": "IAASB",
        "year": 2012,
        "categories": [
            "governance", "data_management", "methodology",
            "internal_controls", "evidence_availability",
            "disclosure_requirements", "scope_completeness",
            "uncertainty_management",
        ],
        "category_weights": {
            "governance": Decimal("0.10"),
            "data_management": Decimal("0.20"),
            "methodology": Decimal("0.15"),
            "internal_controls": Decimal("0.15"),
            "evidence_availability": Decimal("0.15"),
            "disclosure_requirements": Decimal("0.10"),
            "scope_completeness": Decimal("0.10"),
            "uncertainty_management": Decimal("0.05"),
        },
        "checklist_items_per_category": {
            "governance": 5,
            "data_management": 8,
            "methodology": 6,
            "internal_controls": 7,
            "evidence_availability": 6,
            "disclosure_requirements": 5,
            "scope_completeness": 4,
            "uncertainty_management": 4,
        },
    },
    "iso_14064_3": {
        "full_name": "ISO 14064-3:2019 Verification and Validation of GHG Statements",
        "issuer": "ISO",
        "year": 2019,
        "categories": [
            "governance", "data_management", "methodology",
            "internal_controls", "evidence_availability",
            "disclosure_requirements", "scope_completeness",
            "uncertainty_management",
        ],
        "category_weights": {
            "governance": Decimal("0.10"),
            "data_management": Decimal("0.18"),
            "methodology": Decimal("0.18"),
            "internal_controls": Decimal("0.12"),
            "evidence_availability": Decimal("0.14"),
            "disclosure_requirements": Decimal("0.10"),
            "scope_completeness": Decimal("0.12"),
            "uncertainty_management": Decimal("0.06"),
        },
        "checklist_items_per_category": {
            "governance": 4,
            "data_management": 7,
            "methodology": 7,
            "internal_controls": 5,
            "evidence_availability": 6,
            "disclosure_requirements": 4,
            "scope_completeness": 5,
            "uncertainty_management": 5,
        },
    },
    "aa1000as_v3": {
        "full_name": "AA1000 Assurance Standard v3",
        "issuer": "AccountAbility",
        "year": 2020,
        "categories": [
            "governance", "data_management", "methodology",
            "internal_controls", "evidence_availability",
            "disclosure_requirements", "scope_completeness",
            "uncertainty_management",
        ],
        "category_weights": {
            "governance": Decimal("0.15"),
            "data_management": Decimal("0.15"),
            "methodology": Decimal("0.12"),
            "internal_controls": Decimal("0.12"),
            "evidence_availability": Decimal("0.14"),
            "disclosure_requirements": Decimal("0.14"),
            "scope_completeness": Decimal("0.10"),
            "uncertainty_management": Decimal("0.08"),
        },
        "checklist_items_per_category": {
            "governance": 6,
            "data_management": 6,
            "methodology": 5,
            "internal_controls": 5,
            "evidence_availability": 6,
            "disclosure_requirements": 6,
            "scope_completeness": 4,
            "uncertainty_management": 4,
        },
    },
}

READINESS_BAND_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "ready": (80.0, 100.1),
    "nearly_ready": (60.0, 80.0),
    "partially_ready": (40.0, 60.0),
    "not_ready": (0.0, 40.0),
}

GAP_EFFORT_HOURS: Dict[str, Tuple[int, int]] = {
    "critical": (40, 120),
    "high": (20, 60),
    "medium": (8, 30),
    "low": (2, 12),
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class ChecklistItem(BaseModel):
    """A single checklist requirement within a category."""

    item_id: str = Field(default_factory=lambda: f"ci-{_new_uuid()[:8]}")
    category: ChecklistCategory = Field(...)
    requirement: str = Field(default="")
    description: str = Field(default="")
    evidence_status: EvidenceStatus = Field(default=EvidenceStatus.MISSING)
    evidence_source: str = Field(default="")
    evidence_timestamp: str = Field(default="")
    evidence_quality_grade: str = Field(default="")
    is_mandatory: bool = Field(default=True)
    standard_reference: str = Field(default="")
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class CategoryScore(BaseModel):
    """Readiness score for a single category."""

    category: ChecklistCategory = Field(...)
    total_items: int = Field(default=0, ge=0)
    available_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    raw_score: str = Field(default="0.00", description="Decimal string 0-100")
    weighted_score: str = Field(default="0.00", description="Decimal string weighted")
    weight: str = Field(default="0.00", description="Category weight")
    band: ReadinessBand = Field(default=ReadinessBand.NOT_READY)
    provenance_hash: str = Field(default="")

class GapItem(BaseModel):
    """A gap identified in readiness assessment."""

    gap_id: str = Field(default_factory=lambda: f"gap-{_new_uuid()[:8]}")
    category: ChecklistCategory = Field(...)
    checklist_item_id: str = Field(default="")
    requirement: str = Field(default="")
    current_status: EvidenceStatus = Field(default=EvidenceStatus.MISSING)
    priority: GapPriority = Field(default=GapPriority.MEDIUM)
    remediation_action: str = Field(default="")
    estimated_effort_hours_min: int = Field(default=0, ge=0)
    estimated_effort_hours_max: int = Field(default=0, ge=0)
    suggested_owner_role: str = Field(default="")
    target_completion_weeks: int = Field(default=4, ge=1)
    provenance_hash: str = Field(default="")

class StandardConfig(BaseModel):
    """Configuration details of the selected assurance standard."""

    standard: AssuranceStandard = Field(...)
    full_name: str = Field(default="")
    issuer: str = Field(default="")
    year: int = Field(default=2012)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    categories: List[str] = Field(default_factory=list)
    total_checklist_items: int = Field(default=0)
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class ReadinessAssessmentInput(BaseModel):
    """Input data model for ReadinessAssessmentWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    target_standard: AssuranceStandard = Field(
        default=AssuranceStandard.ISAE_3410,
        description="Target assurance standard",
    )
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED,
        description="Target assurance level",
    )
    existing_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-existing evidence items with category, source, timestamp",
    )
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
        description="Emission scopes in scope of assurance",
    )
    reporting_period: str = Field(default="2025", description="Reporting period")
    include_scope_3: bool = Field(default=False, description="Include Scope 3 in checklist")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class ReadinessAssessmentResult(BaseModel):
    """Complete result from readiness assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="readiness_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    standard_config: Optional[StandardConfig] = Field(default=None)
    checklist_items: List[ChecklistItem] = Field(default_factory=list)
    category_scores: List[CategoryScore] = Field(default_factory=list)
    overall_score: str = Field(default="0.00", description="Decimal string 0-100")
    overall_band: ReadinessBand = Field(default=ReadinessBand.NOT_READY)
    gaps: List[GapItem] = Field(default_factory=list)
    total_gaps: int = Field(default=0)
    critical_gaps: int = Field(default=0)
    estimated_remediation_weeks: int = Field(default=0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ReadinessAssessmentWorkflow:
    """
    5-phase workflow for GHG assurance readiness assessment.

    Selects the target assurance standard, generates a standard-specific
    checklist, checks existing evidence against requirements, calculates
    weighted readiness scores per category, and produces a gap report with
    prioritised remediation recommendations.

    Zero-hallucination: all scoring uses Decimal arithmetic with ROUND_HALF_UP;
    category weights and checklist structures are drawn from deterministic
    standard-specific reference tables; no LLM calls in scoring path;
    SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _standard_config: Selected standard configuration.
        _checklist_items: Generated checklist items.
        _category_scores: Per-category readiness scores.
        _gaps: Identified gaps.
        _overall_score: Overall readiness score (Decimal).
        _overall_band: Overall readiness band.

    Example:
        >>> wf = ReadinessAssessmentWorkflow()
        >>> inp = ReadinessAssessmentInput(
        ...     organization_id="org-001",
        ...     target_standard=AssuranceStandard.ISAE_3410,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[ReadinessPhase] = [
        ReadinessPhase.STANDARD_SELECTION,
        ReadinessPhase.CHECKLIST_GENERATION,
        ReadinessPhase.EVIDENCE_CHECK,
        ReadinessPhase.SCORE_CALCULATION,
        ReadinessPhase.GAP_REPORTING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    # Checklist requirement templates per category
    CHECKLIST_TEMPLATES: Dict[str, List[str]] = {
        "governance": [
            "Board/management oversight of GHG reporting documented",
            "Roles and responsibilities for GHG reporting assigned",
            "GHG reporting policy approved and current",
            "Management review process documented and evidenced",
            "Assurance engagement governance structure defined",
            "Escalation procedures for data quality issues defined",
        ],
        "data_management": [
            "Activity data collection procedures documented",
            "Emission factor sources identified and referenced",
            "Data flow from source to report is traceable",
            "Data reconciliation with financial records performed",
            "Electronic data systems validated and access-controlled",
            "Manual data entry controls and review processes in place",
            "Data retention policy meets assurance requirements",
            "Change management for data corrections documented",
        ],
        "methodology": [
            "Organisational boundary approach documented (equity/control)",
            "Operational boundary for each scope documented",
            "Emission calculation methodology documented per source",
            "GWP values and AR vintage explicitly stated",
            "Base year and recalculation policy documented",
            "Exclusions justified and quantified",
            "Methodology consistent with GHG Protocol / ISO 14064-1",
        ],
        "internal_controls": [
            "Internal audit of GHG data performed",
            "Segregation of duties in data collection and reporting",
            "Automated validation checks on activity data",
            "Management sign-off on reported emissions",
            "Error correction and restatement procedures documented",
            "Access controls on GHG data systems",
            "Periodic control effectiveness testing performed",
        ],
        "evidence_availability": [
            "Source documents for activity data accessible",
            "Emission factor documentation and provenance available",
            "Calculation spreadsheets / system outputs available",
            "Internal review and sign-off records available",
            "Prior period comparatives available",
            "Third-party data confirmations available",
        ],
        "disclosure_requirements": [
            "GHG statement format meets standard requirements",
            "Scope 1, 2 (location and market) separately disclosed",
            "Intensity metrics disclosed with denominators explained",
            "Significant changes from prior period disclosed",
            "Uncertainty qualitative/quantitative disclosure present",
            "Assurance statement placement and format agreed",
        ],
        "scope_completeness": [
            "All Scope 1 sources identified and included",
            "All Scope 2 instruments identified and documented",
            "Scope 3 categories screened and relevance assessed",
            "Completeness check against sector guidance performed",
        ],
        "uncertainty_management": [
            "Quantitative uncertainty analysis performed",
            "Data quality indicators assigned to each source",
            "Sensitivity analysis on key assumptions documented",
            "Uncertainty reporting threshold defined and applied",
        ],
    }

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ReadinessAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._standard_config: Optional[StandardConfig] = None
        self._checklist_items: List[ChecklistItem] = []
        self._category_scores: List[CategoryScore] = []
        self._gaps: List[GapItem] = []
        self._overall_score: Decimal = Decimal("0.00")
        self._overall_band: ReadinessBand = ReadinessBand.NOT_READY
        self._estimated_weeks: int = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: ReadinessAssessmentInput,
    ) -> ReadinessAssessmentResult:
        """
        Execute the 5-phase readiness assessment workflow.

        Args:
            input_data: Organisation details, target standard, existing evidence.

        Returns:
            ReadinessAssessmentResult with scores, checklist, and gap report.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting readiness assessment %s org=%s standard=%s level=%s",
            self.workflow_id, input_data.organization_id,
            input_data.target_standard.value, input_data.assurance_level.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_standard_selection,
            self._phase_2_checklist_generation,
            self._phase_3_evidence_check,
            self._phase_4_score_calculation,
            self._phase_5_gap_reporting,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Readiness assessment failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        critical_count = sum(1 for g in self._gaps if g.priority == GapPriority.CRITICAL)

        result = ReadinessAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            standard_config=self._standard_config,
            checklist_items=self._checklist_items,
            category_scores=self._category_scores,
            overall_score=str(self._overall_score),
            overall_band=self._overall_band,
            gaps=self._gaps,
            total_gaps=len(self._gaps),
            critical_gaps=critical_count,
            estimated_remediation_weeks=self._estimated_weeks,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Readiness assessment %s completed in %.2fs status=%s score=%s band=%s gaps=%d",
            self.workflow_id, elapsed, overall_status.value,
            str(self._overall_score), self._overall_band.value, len(self._gaps),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Standard Selection
    # -------------------------------------------------------------------------

    async def _phase_1_standard_selection(
        self, input_data: ReadinessAssessmentInput,
    ) -> PhaseResult:
        """Select target assurance standard and load requirement set."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        std_key = input_data.target_standard.value
        std_data = STANDARD_REQUIREMENTS.get(std_key)

        if not std_data:
            return PhaseResult(
                phase_name="standard_selection", phase_number=1,
                status=PhaseStatus.FAILED,
                errors=[f"Unknown standard: {std_key}"],
                duration_seconds=time.monotonic() - started,
            )

        total_items = sum(std_data["checklist_items_per_category"].values())

        self._standard_config = StandardConfig(
            standard=input_data.target_standard,
            full_name=std_data["full_name"],
            issuer=std_data["issuer"],
            year=std_data["year"],
            assurance_level=input_data.assurance_level,
            categories=std_data["categories"],
            total_checklist_items=total_items,
            provenance_hash=_compute_hash({
                "standard": std_key, "items": total_items,
            }),
        )

        if input_data.assurance_level == AssuranceLevel.REASONABLE:
            warnings.append(
                "Reasonable assurance requires more rigorous evidence; "
                "checklist will include enhanced requirements"
            )

        outputs["standard"] = std_key
        outputs["full_name"] = std_data["full_name"]
        outputs["issuer"] = std_data["issuer"]
        outputs["total_checklist_items"] = total_items
        outputs["assurance_level"] = input_data.assurance_level.value
        outputs["categories"] = std_data["categories"]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 StandardSelection: %s (%s) %d items, level=%s",
            std_key, std_data["full_name"], total_items,
            input_data.assurance_level.value,
        )
        return PhaseResult(
            phase_name="standard_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Checklist Generation
    # -------------------------------------------------------------------------

    async def _phase_2_checklist_generation(
        self, input_data: ReadinessAssessmentInput,
    ) -> PhaseResult:
        """Generate standard-specific checklist from requirement templates."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._standard_config:
            return PhaseResult(
                phase_name="checklist_generation", phase_number=2,
                status=PhaseStatus.FAILED,
                errors=["Standard not selected in Phase 1"],
                duration_seconds=time.monotonic() - started,
            )

        std_key = self._standard_config.standard.value
        std_data = STANDARD_REQUIREMENTS[std_key]
        items_per_cat = std_data["checklist_items_per_category"]

        self._checklist_items = []
        for cat_name, item_count in items_per_cat.items():
            try:
                category = ChecklistCategory(cat_name)
            except ValueError:
                warnings.append(f"Unknown category: {cat_name}")
                continue

            templates = self.CHECKLIST_TEMPLATES.get(cat_name, [])
            for idx in range(item_count):
                requirement_text = (
                    templates[idx] if idx < len(templates)
                    else f"{cat_name} requirement {idx + 1}"
                )

                # Reasonable assurance adds enhanced requirement suffix
                if input_data.assurance_level == AssuranceLevel.REASONABLE:
                    requirement_text += " (enhanced for reasonable assurance)"

                item = ChecklistItem(
                    category=category,
                    requirement=requirement_text,
                    description=f"{std_key} {cat_name} item {idx + 1}",
                    is_mandatory=True,
                    standard_reference=f"{std_key.upper()} {cat_name}.{idx + 1}",
                )
                self._checklist_items.append(item)

        # If scope 3 included, add scope completeness extras
        if input_data.include_scope_3:
            scope3_extras = [
                "Scope 3 category-level calculations documented",
                "Scope 3 supplier data quality assessed",
                "Scope 3 estimation methodology justified",
            ]
            for extra_text in scope3_extras:
                self._checklist_items.append(ChecklistItem(
                    category=ChecklistCategory.SCOPE_COMPLETENESS,
                    requirement=extra_text,
                    description="Additional Scope 3 checklist item",
                    is_mandatory=True,
                    standard_reference=f"{std_key.upper()} scope3.extra",
                ))

        outputs["checklist_items_generated"] = len(self._checklist_items)
        outputs["categories_covered"] = len(items_per_cat)
        outputs["scope_3_extras"] = len(self._checklist_items) - sum(items_per_cat.values())

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 ChecklistGeneration: %d items across %d categories",
            len(self._checklist_items), len(items_per_cat),
        )
        return PhaseResult(
            phase_name="checklist_generation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Evidence Check
    # -------------------------------------------------------------------------

    async def _phase_3_evidence_check(
        self, input_data: ReadinessAssessmentInput,
    ) -> PhaseResult:
        """Check existing evidence against checklist requirements."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build evidence lookup by category
        evidence_by_category: Dict[str, List[Dict[str, Any]]] = {}
        for ev in input_data.existing_evidence:
            cat = ev.get("category", "")
            if cat not in evidence_by_category:
                evidence_by_category[cat] = []
            evidence_by_category[cat].append(ev)

        available_count = 0
        partial_count = 0
        missing_count = 0

        for item in self._checklist_items:
            cat_key = item.category.value
            cat_evidence = evidence_by_category.get(cat_key, [])

            if cat_evidence:
                # Pop the first matching evidence item
                ev = cat_evidence.pop(0)
                completeness = ev.get("completeness", 0.0)

                if completeness >= 80.0:
                    item.evidence_status = EvidenceStatus.AVAILABLE
                    available_count += 1
                elif completeness >= 40.0:
                    item.evidence_status = EvidenceStatus.PARTIAL
                    partial_count += 1
                else:
                    item.evidence_status = EvidenceStatus.MISSING
                    missing_count += 1

                item.evidence_source = ev.get("source", "")
                item.evidence_timestamp = ev.get("timestamp", "")
                item.evidence_quality_grade = ev.get("quality_grade", "")
            else:
                item.evidence_status = EvidenceStatus.MISSING
                missing_count += 1

            item_data = {
                "id": item.item_id,
                "status": item.evidence_status.value,
                "source": item.evidence_source,
            }
            item.provenance_hash = _compute_hash(item_data)

        total_items = len(self._checklist_items)
        outputs["total_items_checked"] = total_items
        outputs["available"] = available_count
        outputs["partial"] = partial_count
        outputs["missing"] = missing_count
        outputs["evidence_items_provided"] = len(input_data.existing_evidence)

        if missing_count > total_items * 0.5:
            warnings.append(
                f"More than 50% of checklist items have missing evidence "
                f"({missing_count}/{total_items})"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 EvidenceCheck: available=%d partial=%d missing=%d (total=%d)",
            available_count, partial_count, missing_count, total_items,
        )
        return PhaseResult(
            phase_name="evidence_check", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Score Calculation
    # -------------------------------------------------------------------------

    async def _phase_4_score_calculation(
        self, input_data: ReadinessAssessmentInput,
    ) -> PhaseResult:
        """Calculate weighted readiness scores per category and overall."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._standard_config:
            return PhaseResult(
                phase_name="score_calculation", phase_number=4,
                status=PhaseStatus.FAILED,
                errors=["Standard not configured"],
                duration_seconds=time.monotonic() - started,
            )

        std_key = self._standard_config.standard.value
        std_data = STANDARD_REQUIREMENTS[std_key]
        category_weights = std_data["category_weights"]

        self._category_scores = []
        total_weighted = Decimal("0.00")

        # Group checklist items by category
        items_by_cat: Dict[str, List[ChecklistItem]] = {}
        for item in self._checklist_items:
            cat_val = item.category.value
            if cat_val not in items_by_cat:
                items_by_cat[cat_val] = []
            items_by_cat[cat_val].append(item)

        for cat_name, items in items_by_cat.items():
            total = len(items)
            avail = sum(1 for i in items if i.evidence_status == EvidenceStatus.AVAILABLE)
            partial = sum(1 for i in items if i.evidence_status == EvidenceStatus.PARTIAL)
            missing = sum(1 for i in items if i.evidence_status == EvidenceStatus.MISSING)

            # Score: available = 1.0, partial = 0.5, missing = 0.0
            if total > 0:
                numerator = Decimal(str(avail)) + Decimal("0.5") * Decimal(str(partial))
                raw = (numerator / Decimal(str(total))) * Decimal("100")
                raw = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                raw = Decimal("0.00")

            weight = category_weights.get(cat_name, Decimal("0.10"))
            weighted = (raw * weight).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            total_weighted += weighted

            band = self._classify_readiness_band(float(raw))

            try:
                category_enum = ChecklistCategory(cat_name)
            except ValueError:
                category_enum = ChecklistCategory.GOVERNANCE

            score_data = {
                "category": cat_name, "raw": str(raw),
                "weight": str(weight), "weighted": str(weighted),
            }
            self._category_scores.append(CategoryScore(
                category=category_enum,
                total_items=total,
                available_count=avail,
                partial_count=partial,
                missing_count=missing,
                raw_score=str(raw),
                weighted_score=str(weighted),
                weight=str(weight),
                band=band,
                provenance_hash=_compute_hash(score_data),
            ))

        self._overall_score = total_weighted.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        self._overall_band = self._classify_readiness_band(float(self._overall_score))

        outputs["category_count"] = len(self._category_scores)
        outputs["overall_score"] = str(self._overall_score)
        outputs["overall_band"] = self._overall_band.value
        outputs["scores_by_category"] = {
            cs.category.value: str(cs.raw_score) for cs in self._category_scores
        }

        if self._overall_band == ReadinessBand.NOT_READY:
            warnings.append(
                "Organisation is not ready for assurance; "
                "significant remediation required"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 ScoreCalculation: overall=%s band=%s categories=%d",
            str(self._overall_score), self._overall_band.value,
            len(self._category_scores),
        )
        return PhaseResult(
            phase_name="score_calculation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Gap Reporting
    # -------------------------------------------------------------------------

    async def _phase_5_gap_reporting(
        self, input_data: ReadinessAssessmentInput,
    ) -> PhaseResult:
        """Produce gap report with prioritised remediation recommendations."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []
        owner_map: Dict[str, str] = {
            "governance": "Sustainability Director",
            "data_management": "Data Management Lead",
            "methodology": "GHG Technical Specialist",
            "internal_controls": "Internal Audit Manager",
            "evidence_availability": "Documentation Coordinator",
            "disclosure_requirements": "Reporting Manager",
            "scope_completeness": "GHG Technical Specialist",
            "uncertainty_management": "GHG Technical Specialist",
        }

        for item in self._checklist_items:
            if item.evidence_status == EvidenceStatus.AVAILABLE:
                continue

            # Determine priority based on category weight and evidence status
            priority = self._determine_gap_priority(item)
            effort_range = GAP_EFFORT_HOURS.get(priority.value, (8, 30))

            remediation = self._generate_remediation_action(item)
            target_weeks = self._estimate_target_weeks(priority)

            gap_data = {
                "item": item.item_id, "priority": priority.value,
                "category": item.category.value,
            }
            gap = GapItem(
                category=item.category,
                checklist_item_id=item.item_id,
                requirement=item.requirement,
                current_status=item.evidence_status,
                priority=priority,
                remediation_action=remediation,
                estimated_effort_hours_min=effort_range[0],
                estimated_effort_hours_max=effort_range[1],
                suggested_owner_role=owner_map.get(item.category.value, "Sustainability Team"),
                target_completion_weeks=target_weeks,
                provenance_hash=_compute_hash(gap_data),
            )
            self._gaps.append(gap)

        # Sort gaps by priority (critical first)
        priority_order = {
            GapPriority.CRITICAL: 0, GapPriority.HIGH: 1,
            GapPriority.MEDIUM: 2, GapPriority.LOW: 3,
        }
        self._gaps.sort(key=lambda g: priority_order.get(g.priority, 99))

        # Estimate overall remediation timeline
        if self._gaps:
            max_target = max(g.target_completion_weeks for g in self._gaps)
            self._estimated_weeks = max_target
        else:
            self._estimated_weeks = 0

        critical_count = sum(1 for g in self._gaps if g.priority == GapPriority.CRITICAL)
        high_count = sum(1 for g in self._gaps if g.priority == GapPriority.HIGH)

        outputs["total_gaps"] = len(self._gaps)
        outputs["critical_gaps"] = critical_count
        outputs["high_gaps"] = high_count
        outputs["medium_gaps"] = sum(1 for g in self._gaps if g.priority == GapPriority.MEDIUM)
        outputs["low_gaps"] = sum(1 for g in self._gaps if g.priority == GapPriority.LOW)
        outputs["estimated_remediation_weeks"] = self._estimated_weeks
        outputs["gaps_by_category"] = {}
        for g in self._gaps:
            cat = g.category.value
            outputs["gaps_by_category"][cat] = outputs["gaps_by_category"].get(cat, 0) + 1

        if critical_count > 0:
            warnings.append(
                f"{critical_count} critical gaps require immediate attention "
                f"before assurance engagement"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 GapReporting: %d gaps (critical=%d high=%d) est_weeks=%d",
            len(self._gaps), critical_count, high_count, self._estimated_weeks,
        )
        return PhaseResult(
            phase_name="gap_reporting", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: ReadinessAssessmentInput,
        phase_number: int,
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
    # Calculation Helpers
    # -------------------------------------------------------------------------

    def _classify_readiness_band(self, score: float) -> ReadinessBand:
        """Classify a readiness score into a band."""
        for band_name, (lower, upper) in READINESS_BAND_THRESHOLDS.items():
            if lower <= score < upper:
                return ReadinessBand(band_name)
        return ReadinessBand.NOT_READY

    def _determine_gap_priority(self, item: ChecklistItem) -> GapPriority:
        """Determine gap priority based on category and evidence status."""
        high_priority_categories = {
            ChecklistCategory.DATA_MANAGEMENT,
            ChecklistCategory.METHODOLOGY,
            ChecklistCategory.INTERNAL_CONTROLS,
        }

        if item.evidence_status == EvidenceStatus.MISSING:
            if item.category in high_priority_categories:
                return GapPriority.CRITICAL
            if item.is_mandatory:
                return GapPriority.HIGH
            return GapPriority.MEDIUM
        else:
            # PARTIAL evidence
            if item.category in high_priority_categories:
                return GapPriority.HIGH
            return GapPriority.MEDIUM

    def _generate_remediation_action(self, item: ChecklistItem) -> str:
        """Generate a remediation action description for a gap item."""
        if item.evidence_status == EvidenceStatus.MISSING:
            return (
                f"Create and document evidence for: {item.requirement}. "
                f"Ensure alignment with {item.standard_reference}."
            )
        return (
            f"Complete partial evidence for: {item.requirement}. "
            f"Address identified gaps to meet {item.standard_reference} requirements."
        )

    def _estimate_target_weeks(self, priority: GapPriority) -> int:
        """Estimate target completion weeks based on gap priority."""
        week_map: Dict[GapPriority, int] = {
            GapPriority.CRITICAL: 4,
            GapPriority.HIGH: 8,
            GapPriority.MEDIUM: 12,
            GapPriority.LOW: 16,
        }
        return week_map.get(priority, 12)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._standard_config = None
        self._checklist_items = []
        self._category_scores = []
        self._gaps = []
        self._overall_score = Decimal("0.00")
        self._overall_band = ReadinessBand.NOT_READY
        self._estimated_weeks = 0

    def _compute_provenance(self, result: ReadinessAssessmentResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.overall_score}|{result.overall_band.value}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
