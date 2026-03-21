# -*- coding: utf-8 -*-
"""
Pledge Onboarding Workflow
================================

5-phase workflow guiding new participants through the complete Race to Zero
pledge onboarding process within PACK-025 Race to Zero Pack.  Covers
organization profiling, eligibility assessment, baseline validation, target
proposal formulation, and commitment documentation packaging.

Phases:
    1. OrganizationProfiling   -- Collect org profile, actor type, sector, scope
    2. EligibilityAssessment   -- Validate pledge eligibility against 8 criteria
    3. BaselineValidation      -- Validate baseline emissions and base year
    4. TargetProposal          -- Propose interim and long-term targets
    5. CommitmentDocumentation -- Generate pledge commitment package

Regulatory references:
    - Race to Zero Campaign (UNFCCC Climate Champions, 2020/2022)
    - Race to Zero Interpretation Guide (June 2022 update)
    - HLEG "Integrity Matters" Report (November 2022)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - Paris Agreement (2015) -- 1.5C temperature goal
    - GHG Protocol Corporate Standard (2015)
    - GHG Protocol Scope 3 Standard (2011)

Zero-hallucination: all eligibility checks, target calculations, and
pathway validations use deterministic formulas and reference tables.
No LLM calls in the numeric computation path.

Author: GreenLang Team
Version: 25.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest of *data*."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


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
    CANCELLED = "cancelled"


class OnboardingPhase(str, Enum):
    """The 5 phases of the pledge onboarding workflow."""
    ORGANIZATION_PROFILING = "organization_profiling"
    ELIGIBILITY_ASSESSMENT = "eligibility_assessment"
    BASELINE_VALIDATION = "baseline_validation"
    TARGET_PROPOSAL = "target_proposal"
    COMMITMENT_DOCUMENTATION = "commitment_documentation"


class ActorType(str, Enum):
    """Race to Zero actor types."""
    CORPORATE = "corporate"
    FINANCIAL_INSTITUTION = "financial_institution"
    CITY = "city"
    REGION = "region"
    UNIVERSITY = "university"
    HEALTHCARE = "healthcare"
    SME = "sme"


class PartnerInitiative(str, Enum):
    """Race to Zero partner initiatives (accelerators)."""
    SBTI = "sbti"
    CDP = "cdp"
    C40 = "c40"
    ICLEI = "iclei"
    GFANZ = "gfanz"
    WE_MEAN_BUSINESS = "we_mean_business"
    THE_CLIMATE_PLEDGE = "the_climate_pledge"
    SECOND_NATURE = "second_nature"
    SME_CLIMATE_HUB = "sme_climate_hub"
    HEALTH_CARE_WITHOUT_HARM = "health_care_without_harm"


class PledgeQuality(str, Enum):
    """Pledge quality assessment level."""
    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    INELIGIBLE = "ineligible"


class EligibilityStatus(str, Enum):
    """Eligibility determination status."""
    ELIGIBLE = "eligible"
    CONDITIONAL = "conditional"
    INELIGIBLE = "ineligible"


class TargetAmbition(str, Enum):
    """Target ambition level relative to 1.5C pathway."""
    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_WB2C = "aligned_well_below_2c"
    BELOW_2C = "below_2c"
    INSUFFICIENT = "insufficient"


class ScopeType(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# Pledge eligibility criteria per Race to Zero Interpretation Guide
PLEDGE_ELIGIBILITY_CRITERIA: List[Dict[str, Any]] = [
    {
        "id": "PC-01",
        "name": "Net-zero commitment by 2050",
        "description": "Pledge to reach net zero no later than 2050",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-02",
        "name": "Partner initiative membership",
        "description": "Join through a recognized Race to Zero partner initiative",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-03",
        "name": "Interim 2030 target",
        "description": "Commit to an interim target aligned with halving emissions by 2030",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-04",
        "name": "Action plan commitment",
        "description": "Commit to publish a climate action plan within 12 months",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-05",
        "name": "Annual reporting commitment",
        "description": "Commit to report progress annually through partner channels",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-06",
        "name": "Scope coverage",
        "description": "Pledge covers all material emission scopes",
        "required": True,
        "actor_types": ["corporate", "financial_institution"],
    },
    {
        "id": "PC-07",
        "name": "Governance endorsement",
        "description": "Board or senior leadership endorsement of pledge",
        "required": True,
        "actor_types": ["all"],
    },
    {
        "id": "PC-08",
        "name": "Public disclosure",
        "description": "Commitment to publicly disclose pledge and progress",
        "required": True,
        "actor_types": ["all"],
    },
]

# Partner initiative to actor type mapping
PARTNER_ACTOR_MAP: Dict[str, List[str]] = {
    "sbti": ["corporate", "financial_institution"],
    "cdp": ["corporate", "financial_institution"],
    "c40": ["city"],
    "iclei": ["city", "region"],
    "gfanz": ["financial_institution"],
    "we_mean_business": ["corporate"],
    "the_climate_pledge": ["corporate"],
    "second_nature": ["university"],
    "sme_climate_hub": ["sme"],
    "health_care_without_harm": ["healthcare"],
}

# 1.5C pathway annual reduction rates
PATHWAY_REDUCTION_RATES: Dict[str, float] = {
    "aligned_1_5c": 4.2,         # % per year (SBTi 1.5C)
    "aligned_well_below_2c": 2.5,  # % per year (SBTi WB2C)
    "below_2c": 1.8,             # % per year
    "insufficient": 0.0,
}

# Minimum baseline requirements
MIN_BASELINE_YEAR = 2015
PREFERRED_BASELINE_YEAR = 2019
MAX_BASELINE_AGE_YEARS = 5
IPCC_2030_REDUCTION_PCT = 43.0  # IPCC AR6: 43% CO2 reduction by 2030 vs 2019
R2Z_2030_TARGET_PCT = 50.0      # Race to Zero: ~50% by 2030
MIN_SCOPE3_COVERAGE_PCT = 67.0  # SBTi minimum Scope 3 coverage


# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[OnboardingPhase, List[OnboardingPhase]] = {
    OnboardingPhase.ORGANIZATION_PROFILING: [],
    OnboardingPhase.ELIGIBILITY_ASSESSMENT: [OnboardingPhase.ORGANIZATION_PROFILING],
    OnboardingPhase.BASELINE_VALIDATION: [OnboardingPhase.ELIGIBILITY_ASSESSMENT],
    OnboardingPhase.TARGET_PROPOSAL: [OnboardingPhase.BASELINE_VALIDATION],
    OnboardingPhase.COMMITMENT_DOCUMENTATION: [OnboardingPhase.TARGET_PROPOSAL],
}

# Phase execution order (topological sort)
PHASE_EXECUTION_ORDER: List[OnboardingPhase] = [
    OnboardingPhase.ORGANIZATION_PROFILING,
    OnboardingPhase.ELIGIBILITY_ASSESSMENT,
    OnboardingPhase.BASELINE_VALIDATION,
    OnboardingPhase.TARGET_PROPOSAL,
    OnboardingPhase.COMMITMENT_DOCUMENTATION,
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: OnboardingPhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class OrganizationProfile(BaseModel):
    """Organization profile for Race to Zero onboarding."""
    org_name: str = Field(default="")
    actor_type: ActorType = Field(default=ActorType.CORPORATE)
    sector: str = Field(default="")
    sub_sector: str = Field(default="")
    country: str = Field(default="")
    region: str = Field(default="")
    employee_count: int = Field(default=0, ge=0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    existing_commitments: List[str] = Field(default_factory=list)
    partner_initiatives: List[PartnerInitiative] = Field(default_factory=list)


class EligibilityAssessment(BaseModel):
    """Eligibility assessment for Race to Zero participation."""
    status: EligibilityStatus = Field(default=EligibilityStatus.INELIGIBLE)
    criteria_met: int = Field(default=0)
    criteria_total: int = Field(default=8)
    criteria_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pledge_quality: PledgeQuality = Field(default=PledgeQuality.INELIGIBLE)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    partner_alignment: Dict[str, bool] = Field(default_factory=dict)


class BaselineEmissions(BaseModel):
    """Baseline emissions for target setting."""
    base_year: int = Field(default=2019, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    scope3_categories_included: List[int] = Field(default_factory=list)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    verification_status: str = Field(default="unverified")
    is_valid: bool = Field(default=False)
    validation_issues: List[str] = Field(default_factory=list)


class TargetProposal(BaseModel):
    """Proposed targets for Race to Zero commitment."""
    interim_target_year: int = Field(default=2030)
    interim_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    long_term_target_year: int = Field(default=2050)
    long_term_target: str = Field(default="net_zero")
    annual_reduction_rate: float = Field(default=0.0, ge=0.0)
    ambition_level: TargetAmbition = Field(default=TargetAmbition.INSUFFICIENT)
    scope_coverage: Dict[str, float] = Field(default_factory=dict)
    methodology: str = Field(default="absolute_contraction")
    fair_share_assessment: str = Field(default="")
    pathway_aligned: bool = Field(default=False)


class CommitmentPackage(BaseModel):
    """Complete commitment package for Race to Zero submission."""
    package_id: str = Field(default="")
    org_name: str = Field(default="")
    actor_type: str = Field(default="")
    partner_initiative: str = Field(default="")
    pledge_statement: str = Field(default="")
    interim_target_summary: str = Field(default="")
    long_term_target_summary: str = Field(default="")
    action_plan_commitment: str = Field(default="")
    reporting_commitment: str = Field(default="")
    governance_endorsement: str = Field(default="")
    documents_generated: List[str] = Field(default_factory=list)
    submission_ready: bool = Field(default=False)
    submission_deadline: str = Field(default="")


class PledgeOnboardingConfig(BaseModel):
    """Configuration for the pledge onboarding workflow."""
    pack_id: str = Field(default="PACK-025")
    pack_version: str = Field(default="1.0.0")
    org_name: str = Field(default="")
    actor_type: ActorType = Field(default=ActorType.CORPORATE)
    sector: str = Field(default="")
    country: str = Field(default="")
    employee_count: int = Field(default=0, ge=0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2019, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    scope3_categories: List[int] = Field(default_factory=lambda: list(range(1, 16)))
    partner_initiative: PartnerInitiative = Field(default=PartnerInitiative.SBTI)
    has_net_zero_commitment: bool = Field(default=True)
    has_interim_target: bool = Field(default=True)
    has_action_plan: bool = Field(default=False)
    has_governance_endorsement: bool = Field(default=False)
    has_annual_reporting: bool = Field(default=False)
    has_public_disclosure: bool = Field(default=False)
    existing_commitments: List[str] = Field(default_factory=list)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    enable_provenance: bool = Field(default=True)
    timeout_per_phase_seconds: int = Field(default=600, ge=30)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("scope3_categories")
    @classmethod
    def _validate_categories(cls, v: List[int]) -> List[int]:
        for cat_id in v:
            if cat_id < 1 or cat_id > 15:
                raise ValueError(f"Scope 3 category must be 1-15, got {cat_id}")
        return v


class PledgeOnboardingResult(BaseModel):
    """Complete result from the pledge onboarding workflow."""
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="pledge_onboarding")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    profile: Optional[OrganizationProfile] = Field(None)
    eligibility: Optional[EligibilityAssessment] = Field(None)
    baseline: Optional[BaselineEmissions] = Field(None)
    targets: Optional[TargetProposal] = Field(None)
    commitment: Optional[CommitmentPackage] = Field(None)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PledgeOnboardingWorkflow:
    """
    5-phase pledge onboarding workflow for PACK-025 Race to Zero Pack.

    Guides new participants through the complete Race to Zero pledge
    onboarding process: organization profiling, eligibility assessment,
    baseline validation, target proposal formulation, and commitment
    documentation packaging.

    Zero-hallucination: all eligibility checks, target calculations, and
    pathway validations use deterministic formulas and reference tables.

    Engines used:
        - pledge_commitment_engine (eligibility + pledge quality)
        - starting_line_engine (criteria pre-check)
        - interim_target_engine (target proposal)

    Attributes:
        config: Workflow configuration.

    Example:
        >>> wf = PledgeOnboardingWorkflow()
        >>> config = PledgeOnboardingConfig(
        ...     org_name="Acme Corp",
        ...     actor_type=ActorType.CORPORATE,
        ...     scope1_tco2e=5000.0,
        ...     scope2_tco2e=3000.0,
        ...     scope3_tco2e=12000.0,
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[PledgeOnboardingConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialise PledgeOnboardingWorkflow."""
        self.config = config or PledgeOnboardingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, PledgeOnboardingResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> PledgeOnboardingResult:
        """
        Execute the 5-phase pledge onboarding workflow.

        Args:
            input_data: Optional additional input data to merge into context.

        Returns:
            PledgeOnboardingResult with pledge commitment package.
        """
        input_data = input_data or {}

        result = PledgeOnboardingResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.execution_id] = result

        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER
        total_phases = len(phases)

        self.logger.info(
            "Starting pledge onboarding: execution_id=%s, org=%s, actor=%s",
            result.execution_id, self.config.org_name, self.config.actor_type.value,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["org_name"] = self.config.org_name
        shared_context["actor_type"] = self.config.actor_type.value
        shared_context["sector"] = self.config.sector
        shared_context["country"] = self.config.country
        shared_context["reporting_year"] = self.config.reporting_year
        shared_context["base_year"] = self.config.base_year
        shared_context["scope1_tco2e"] = self.config.scope1_tco2e
        shared_context["scope2_tco2e"] = self.config.scope2_tco2e
        shared_context["scope3_tco2e"] = self.config.scope3_tco2e

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    result.errors.append("Onboarding cancelled by user")
                    break

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=PhaseStatus.FAILED,
                        errors=["Dependencies not met"],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                # Progress callback
                progress_pct = (phase_idx / total_phases) * 100.0
                if self._progress_callback:
                    await self._progress_callback(
                        phase.value, progress_pct, f"Executing {phase.value}"
                    )

                # Execute phase
                phase_result = await self._execute_phase(phase, shared_context)
                result.phase_results[phase.value] = phase_result

                if phase_result.status == PhaseStatus.FAILED:
                    # Check if this is a hard failure
                    if phase == OnboardingPhase.ELIGIBILITY_ASSESSMENT:
                        result.status = WorkflowStatus.FAILED
                        result.errors.append(f"Phase '{phase.value}' failed -- entity ineligible")
                        break
                    elif phase in (
                        OnboardingPhase.ORGANIZATION_PROFILING,
                        OnboardingPhase.BASELINE_VALIDATION,
                    ):
                        result.status = WorkflowStatus.FAILED
                        result.errors.append(f"Phase '{phase.value}' failed")
                        break

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Pledge onboarding failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = _utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.quality_score = self._compute_quality_score(result)
            result.profile = self._extract_profile(shared_context)
            result.eligibility = self._extract_eligibility(shared_context)
            result.baseline = self._extract_baseline(shared_context)
            result.targets = self._extract_targets(shared_context)
            result.commitment = self._extract_commitment(shared_context)
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        self.logger.info(
            "Pledge onboarding %s: status=%s, phases=%d/%d, duration=%.1fms",
            result.execution_id, result.status.value,
            len(result.phases_completed), total_phases, result.total_duration_ms,
        )
        return result

    def cancel(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a running onboarding execution."""
        self._cancelled.add(execution_id)
        return {"cancelled": True, "execution_id": execution_id}

    def get_result(self, execution_id: str) -> Optional[PledgeOnboardingResult]:
        """Retrieve result for a given execution."""
        return self._results.get(execution_id)

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase: OnboardingPhase, context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute a single phase of the onboarding workflow."""
        started = _utcnow()
        start_time = time.monotonic()

        handler = self._get_phase_handler(phase)
        try:
            outputs, warnings, errors, records = await handler(context)
            status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        except Exception as exc:
            outputs = {}
            warnings = []
            errors = [str(exc)]
            records = 0
            status = PhaseStatus.FAILED

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return PhaseResult(
            phase=phase,
            status=status,
            started_at=started,
            completed_at=_utcnow(),
            duration_ms=round(elapsed_ms, 2),
            records_processed=records,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(outputs) if self.config.enable_provenance else "",
        )

    def _get_phase_handler(self, phase: OnboardingPhase):
        """Return the handler coroutine for a given phase."""
        handlers = {
            OnboardingPhase.ORGANIZATION_PROFILING: self._handle_organization_profiling,
            OnboardingPhase.ELIGIBILITY_ASSESSMENT: self._handle_eligibility_assessment,
            OnboardingPhase.BASELINE_VALIDATION: self._handle_baseline_validation,
            OnboardingPhase.TARGET_PROPOSAL: self._handle_target_proposal,
            OnboardingPhase.COMMITMENT_DOCUMENTATION: self._handle_commitment_documentation,
        }
        return handlers[phase]

    # -------------------------------------------------------------------------
    # Phase 1: Organization Profiling
    # -------------------------------------------------------------------------

    async def _handle_organization_profiling(
        self, ctx: Dict[str, Any]
    ) -> tuple:
        """Collect and validate organization profile."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        org_name = ctx.get("org_name", self.config.org_name)
        actor_type = ctx.get("actor_type", self.config.actor_type.value)
        sector = ctx.get("sector", self.config.sector)
        country = ctx.get("country", self.config.country)

        # Validate organization name
        if not org_name:
            errors.append("Organization name is required for Race to Zero onboarding")

        # Validate actor type
        valid_actor_types = [at.value for at in ActorType]
        if actor_type not in valid_actor_types:
            errors.append(f"Invalid actor type '{actor_type}'; must be one of {valid_actor_types}")

        # Check partner initiative compatibility
        partner = self.config.partner_initiative.value
        compatible_actors = PARTNER_ACTOR_MAP.get(partner, [])
        if compatible_actors and actor_type not in compatible_actors:
            warnings.append(
                f"Partner initiative '{partner}' typically serves {compatible_actors}, "
                f"not '{actor_type}'. Consider alternative partner."
            )

        # Sector validation for corporates
        if actor_type == ActorType.CORPORATE.value and not sector:
            warnings.append("Sector not specified; sector pathway alignment will be limited")

        # Employee count validation
        employee_count = self.config.employee_count
        if actor_type == ActorType.SME.value and employee_count > 250:
            warnings.append(
                f"Employee count ({employee_count}) exceeds SME threshold (250). "
                f"Consider 'corporate' actor type."
            )
        elif actor_type == ActorType.CORPORATE.value and 0 < employee_count <= 250:
            warnings.append(
                f"Employee count ({employee_count}) is within SME range. "
                f"SME Climate Hub may offer simplified pathway."
            )

        # Build organization profile output
        total_emissions = (
            self.config.scope1_tco2e + self.config.scope2_tco2e + self.config.scope3_tco2e
        )

        outputs["org_name"] = org_name
        outputs["actor_type"] = actor_type
        outputs["sector"] = sector
        outputs["sub_sector"] = self.config.sector  # Map to sub-sector if available
        outputs["country"] = country
        outputs["employee_count"] = employee_count
        outputs["revenue_usd"] = self.config.revenue_usd
        outputs["reporting_year"] = self.config.reporting_year
        outputs["partner_initiative"] = partner
        outputs["existing_commitments"] = self.config.existing_commitments
        outputs["total_emissions_tco2e"] = round(total_emissions, 2)
        outputs["emission_intensity"] = round(
            total_emissions / max(self.config.revenue_usd / 1_000_000, 1.0), 4
        ) if self.config.revenue_usd > 0 else 0.0
        outputs["profile_complete"] = not errors

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 2: Eligibility Assessment
    # -------------------------------------------------------------------------

    async def _handle_eligibility_assessment(
        self, ctx: Dict[str, Any]
    ) -> tuple:
        """Validate pledge eligibility against 8 Race to Zero criteria."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        actor_type = ctx.get("actor_type", self.config.actor_type.value)
        criteria_results: Dict[str, Dict[str, Any]] = {}
        criteria_met = 0
        criteria_total = 0
        gaps: List[str] = []
        recommendations: List[str] = []

        for criterion in PLEDGE_ELIGIBILITY_CRITERIA:
            cid = criterion["id"]
            applies = (
                "all" in criterion["actor_types"]
                or actor_type in criterion["actor_types"]
            )

            if not applies:
                criteria_results[cid] = {
                    "name": criterion["name"],
                    "status": "not_applicable",
                    "applies": False,
                }
                continue

            criteria_total += 1
            passed = False

            # Evaluate each criterion
            if cid == "PC-01":  # Net-zero commitment
                passed = self.config.has_net_zero_commitment
                if not passed:
                    gaps.append("Missing net-zero by 2050 commitment")
                    recommendations.append("Formalize net-zero by 2050 pledge statement")

            elif cid == "PC-02":  # Partner initiative
                partner = self.config.partner_initiative.value
                compatible = PARTNER_ACTOR_MAP.get(partner, [])
                passed = bool(partner) and (
                    not compatible or actor_type in compatible or "all" in compatible
                )
                if not passed:
                    gaps.append(f"Partner initiative '{partner}' not compatible with '{actor_type}'")
                    recommendations.append("Select a compatible Race to Zero partner initiative")

            elif cid == "PC-03":  # Interim 2030 target
                passed = self.config.has_interim_target
                if not passed:
                    gaps.append("No interim 2030 target set")
                    recommendations.append(
                        "Set an interim target of approximately 50% absolute reduction by 2030"
                    )

            elif cid == "PC-04":  # Action plan commitment
                passed = self.config.has_action_plan
                if not passed:
                    gaps.append("No action plan published or committed")
                    recommendations.append(
                        "Commit to publishing a climate action plan within 12 months of joining"
                    )

            elif cid == "PC-05":  # Annual reporting
                passed = self.config.has_annual_reporting
                if not passed:
                    gaps.append("No annual reporting commitment")
                    recommendations.append(
                        "Commit to annual progress reporting through partner initiative"
                    )

            elif cid == "PC-06":  # Scope coverage
                total = (
                    self.config.scope1_tco2e
                    + self.config.scope2_tco2e
                    + self.config.scope3_tco2e
                )
                s12 = self.config.scope1_tco2e + self.config.scope2_tco2e
                scope_coverage = (s12 / max(total, 1.0)) * 100.0 if total > 0 else 0.0
                scope3_ok = (
                    self.config.scope3_coverage_pct >= MIN_SCOPE3_COVERAGE_PCT
                    or self.config.scope3_tco2e > 0
                )
                passed = scope_coverage > 0 and scope3_ok
                if not passed:
                    gaps.append("Insufficient scope coverage for pledge")
                    recommendations.append(
                        f"Ensure Scope 3 coverage >= {MIN_SCOPE3_COVERAGE_PCT}% "
                        "of total value chain emissions"
                    )

            elif cid == "PC-07":  # Governance endorsement
                passed = self.config.has_governance_endorsement
                if not passed:
                    gaps.append("No board/senior leadership endorsement")
                    recommendations.append(
                        "Obtain formal board or C-suite endorsement of the net-zero pledge"
                    )

            elif cid == "PC-08":  # Public disclosure
                passed = self.config.has_public_disclosure
                if not passed:
                    gaps.append("No public disclosure commitment")
                    recommendations.append(
                        "Commit to publicly disclosing pledge and annual progress"
                    )

            if passed:
                criteria_met += 1

            criteria_results[cid] = {
                "name": criterion["name"],
                "status": "pass" if passed else "fail",
                "applies": True,
                "required": criterion["required"],
            }
            records += 1

        # Determine eligibility status
        all_required_met = all(
            criteria_results.get(c["id"], {}).get("status") in ("pass", "not_applicable")
            for c in PLEDGE_ELIGIBILITY_CRITERIA
            if c["required"]
        )

        if criteria_met == criteria_total:
            eligibility_status = EligibilityStatus.ELIGIBLE.value
            pledge_quality = PledgeQuality.STRONG.value
        elif criteria_met >= criteria_total - 2 and all_required_met:
            eligibility_status = EligibilityStatus.ELIGIBLE.value
            pledge_quality = PledgeQuality.ADEQUATE.value
        elif criteria_met >= criteria_total - 4:
            eligibility_status = EligibilityStatus.CONDITIONAL.value
            pledge_quality = PledgeQuality.WEAK.value
        else:
            eligibility_status = EligibilityStatus.INELIGIBLE.value
            pledge_quality = PledgeQuality.INELIGIBLE.value
            errors.append(
                f"Entity does not meet minimum eligibility: {criteria_met}/{criteria_total} criteria met"
            )

        # Partner alignment check
        partner_alignment: Dict[str, bool] = {}
        for pi in PartnerInitiative:
            compatible = PARTNER_ACTOR_MAP.get(pi.value, [])
            partner_alignment[pi.value] = (
                not compatible or actor_type in compatible
            )

        outputs["eligibility_status"] = eligibility_status
        outputs["pledge_quality"] = pledge_quality
        outputs["criteria_met"] = criteria_met
        outputs["criteria_total"] = criteria_total
        outputs["criteria_results"] = criteria_results
        outputs["gaps"] = gaps
        outputs["gaps_count"] = len(gaps)
        outputs["recommendations"] = recommendations
        outputs["partner_alignment"] = partner_alignment

        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 3: Baseline Validation
    # -------------------------------------------------------------------------

    async def _handle_baseline_validation(
        self, ctx: Dict[str, Any]
    ) -> tuple:
        """Validate baseline emissions and base year for target setting."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        base_year = self.config.base_year
        reporting_year = self.config.reporting_year
        scope1 = self.config.scope1_tco2e
        scope2 = self.config.scope2_tco2e
        scope3 = self.config.scope3_tco2e
        total = scope1 + scope2 + scope3
        scope3_coverage = self.config.scope3_coverage_pct

        validation_issues: List[str] = []
        is_valid = True

        # Base year validation
        if base_year < MIN_BASELINE_YEAR:
            validation_issues.append(
                f"Base year {base_year} is before minimum allowed ({MIN_BASELINE_YEAR})"
            )
            is_valid = False

        baseline_age = reporting_year - base_year
        if baseline_age > MAX_BASELINE_AGE_YEARS:
            warnings.append(
                f"Baseline is {baseline_age} years old (preferred max {MAX_BASELINE_AGE_YEARS}). "
                "Consider updating to a more recent baseline."
            )

        if base_year < PREFERRED_BASELINE_YEAR:
            warnings.append(
                f"Base year {base_year} is before preferred year ({PREFERRED_BASELINE_YEAR}). "
                "Race to Zero recommends 2019 or later."
            )

        # Emissions completeness validation
        if total <= 0:
            validation_issues.append("Total baseline emissions must be > 0")
            is_valid = False

        if scope1 <= 0 and scope2 <= 0:
            validation_issues.append("Scope 1 and Scope 2 cannot both be zero")
            is_valid = False

        # Scope 3 coverage for corporates and FIs
        actor_type = ctx.get("actor_type", self.config.actor_type.value)
        if actor_type in (ActorType.CORPORATE.value, ActorType.FINANCIAL_INSTITUTION.value):
            if scope3 <= 0:
                warnings.append(
                    "Scope 3 emissions are zero. Race to Zero requires Scope 3 coverage."
                )
            if scope3_coverage < MIN_SCOPE3_COVERAGE_PCT:
                warnings.append(
                    f"Scope 3 coverage ({scope3_coverage:.1f}%) below minimum "
                    f"({MIN_SCOPE3_COVERAGE_PCT}%). Increase Scope 3 screening."
                )

        # Data quality assessment
        data_quality = 0.0
        if scope1 > 0:
            data_quality += 30.0  # Primary data for Scope 1
        if scope2 > 0:
            data_quality += 25.0  # Scope 2 data
        if scope3 > 0:
            data_quality += 25.0  # Scope 3 data
        if scope3_coverage >= MIN_SCOPE3_COVERAGE_PCT:
            data_quality += 10.0
        if base_year >= PREFERRED_BASELINE_YEAR:
            data_quality += 10.0

        # Scope breakdown
        scope_breakdown = {
            "scope_1_pct": round((scope1 / max(total, 1.0)) * 100.0, 1),
            "scope_2_pct": round((scope2 / max(total, 1.0)) * 100.0, 1),
            "scope_3_pct": round((scope3 / max(total, 1.0)) * 100.0, 1),
        }

        # Scope 3 category coverage
        categories_included = self.config.scope3_categories
        categories_coverage_pct = (len(categories_included) / 15.0) * 100.0

        outputs["base_year"] = base_year
        outputs["reporting_year"] = reporting_year
        outputs["baseline_age_years"] = baseline_age
        outputs["scope1_tco2e"] = round(scope1, 2)
        outputs["scope2_tco2e"] = round(scope2, 2)
        outputs["scope3_tco2e"] = round(scope3, 2)
        outputs["total_tco2e"] = round(total, 2)
        outputs["scope3_coverage_pct"] = round(scope3_coverage, 1)
        outputs["scope3_categories_included"] = categories_included
        outputs["scope3_categories_coverage_pct"] = round(categories_coverage_pct, 1)
        outputs["scope_breakdown"] = scope_breakdown
        outputs["data_quality_score"] = round(data_quality, 1)
        outputs["validation_issues"] = validation_issues
        outputs["is_valid"] = is_valid

        if not is_valid:
            errors.extend(validation_issues)

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 4: Target Proposal
    # -------------------------------------------------------------------------

    async def _handle_target_proposal(
        self, ctx: Dict[str, Any]
    ) -> tuple:
        """Propose interim and long-term targets for Race to Zero commitment."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        baseline_data = ctx.get("baseline_validation", {})
        total_baseline = baseline_data.get("total_tco2e", 0)
        base_year = baseline_data.get("base_year", self.config.base_year)
        scope3_coverage = baseline_data.get("scope3_coverage_pct", self.config.scope3_coverage_pct)

        # Calculate target emissions for 2030
        target_reduction_pct = self.config.target_reduction_pct
        interim_year = 2030
        long_term_year = 2050

        # Annual reduction rate calculation
        years_to_interim = max(interim_year - base_year, 1)
        annual_reduction_rate = target_reduction_pct / years_to_interim

        # Target emissions
        interim_target_tco2e = total_baseline * (1 - target_reduction_pct / 100.0)
        residual_2050_pct = 10.0  # Assume 10% residual for neutralization
        long_term_target_tco2e = total_baseline * (residual_2050_pct / 100.0)

        # Assess ambition level
        if annual_reduction_rate >= PATHWAY_REDUCTION_RATES["aligned_1_5c"]:
            ambition = TargetAmbition.ALIGNED_1_5C.value
        elif annual_reduction_rate >= PATHWAY_REDUCTION_RATES["aligned_well_below_2c"]:
            ambition = TargetAmbition.ALIGNED_WB2C.value
        elif annual_reduction_rate >= PATHWAY_REDUCTION_RATES["below_2c"]:
            ambition = TargetAmbition.BELOW_2C.value
        else:
            ambition = TargetAmbition.INSUFFICIENT.value
            warnings.append(
                f"Annual reduction rate ({annual_reduction_rate:.1f}%) is insufficient "
                f"for 1.5C alignment (need >= {PATHWAY_REDUCTION_RATES['aligned_1_5c']}%/yr)"
            )

        # Check Race to Zero minimum (50% by 2030)
        if target_reduction_pct < R2Z_2030_TARGET_PCT:
            warnings.append(
                f"Target reduction ({target_reduction_pct:.0f}%) is below Race to Zero "
                f"minimum (~{R2Z_2030_TARGET_PCT:.0f}% by 2030)"
            )

        # Check IPCC minimum (43% by 2030)
        if target_reduction_pct < IPCC_2030_REDUCTION_PCT:
            warnings.append(
                f"Target reduction ({target_reduction_pct:.0f}%) is below IPCC AR6 "
                f"minimum ({IPCC_2030_REDUCTION_PCT:.0f}% CO2 reduction by 2030)"
            )

        # Scope coverage assessment
        scope_coverage = {
            "scope_1": 100.0,
            "scope_2": 100.0,
            "scope_3": round(scope3_coverage, 1),
        }

        pathway_aligned = (
            target_reduction_pct >= IPCC_2030_REDUCTION_PCT
            and annual_reduction_rate >= PATHWAY_REDUCTION_RATES["aligned_well_below_2c"]
        )

        # Fair share assessment
        fair_share = "adequate"
        if target_reduction_pct >= R2Z_2030_TARGET_PCT:
            fair_share = "strong"
        elif target_reduction_pct < IPCC_2030_REDUCTION_PCT:
            fair_share = "insufficient"

        outputs["interim_target_year"] = interim_year
        outputs["interim_reduction_pct"] = round(target_reduction_pct, 1)
        outputs["interim_target_tco2e"] = round(interim_target_tco2e, 2)
        outputs["long_term_target_year"] = long_term_year
        outputs["long_term_target"] = "net_zero"
        outputs["long_term_target_tco2e"] = round(long_term_target_tco2e, 2)
        outputs["annual_reduction_rate"] = round(annual_reduction_rate, 2)
        outputs["ambition_level"] = ambition
        outputs["scope_coverage"] = scope_coverage
        outputs["methodology"] = "absolute_contraction"
        outputs["fair_share_assessment"] = fair_share
        outputs["pathway_aligned"] = pathway_aligned
        outputs["ipcc_gap_pct"] = round(
            max(IPCC_2030_REDUCTION_PCT - target_reduction_pct, 0), 1
        )
        outputs["r2z_gap_pct"] = round(
            max(R2Z_2030_TARGET_PCT - target_reduction_pct, 0), 1
        )
        outputs["baseline_tco2e"] = round(total_baseline, 2)
        outputs["reduction_tco2e"] = round(total_baseline - interim_target_tco2e, 2)

        records = 1
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Phase 5: Commitment Documentation
    # -------------------------------------------------------------------------

    async def _handle_commitment_documentation(
        self, ctx: Dict[str, Any]
    ) -> tuple:
        """Generate pledge commitment package for Race to Zero submission."""
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []
        records = 0

        profile = ctx.get("organization_profiling", {})
        eligibility = ctx.get("eligibility_assessment", {})
        baseline = ctx.get("baseline_validation", {})
        targets = ctx.get("target_proposal", {})

        org_name = profile.get("org_name", self.config.org_name)
        actor_type = profile.get("actor_type", self.config.actor_type.value)
        partner = profile.get("partner_initiative", self.config.partner_initiative.value)

        # Check eligibility status
        elig_status = eligibility.get("eligibility_status", "ineligible")
        if elig_status == EligibilityStatus.INELIGIBLE.value:
            errors.append("Cannot generate commitment package: entity is ineligible")
            return outputs, warnings, errors, records

        # Generate pledge statement
        interim_pct = targets.get("interim_reduction_pct", self.config.target_reduction_pct)
        ambition = targets.get("ambition_level", "insufficient")

        pledge_statement = (
            f"{org_name} hereby commits to the Race to Zero campaign, pledging to reach "
            f"net-zero greenhouse gas emissions no later than 2050. We commit to an interim "
            f"target of {interim_pct:.0f}% absolute emission reduction by 2030 from our "
            f"{baseline.get('base_year', self.config.base_year)} baseline. This commitment "
            f"covers all material Scope 1, Scope 2, and Scope 3 emissions."
        )

        interim_summary = (
            f"{interim_pct:.0f}% absolute reduction by 2030 from "
            f"{baseline.get('base_year', self.config.base_year)} baseline "
            f"({baseline.get('total_tco2e', 0):,.0f} tCO2e). "
            f"Ambition level: {ambition}. "
            f"Annual reduction rate: {targets.get('annual_reduction_rate', 0):.1f}%/yr."
        )

        long_term_summary = (
            f"Net-zero emissions by 2050 with maximum {targets.get('long_term_target_tco2e', 0):,.0f} "
            f"tCO2e residual emissions to be neutralized through high-quality carbon removal."
        )

        action_plan_commitment = (
            f"{org_name} commits to publishing a comprehensive climate action plan within "
            f"12 months of joining the Race to Zero campaign, including specific decarbonization "
            f"actions, timelines, milestones, and resource allocation."
        )

        reporting_commitment = (
            f"{org_name} commits to annual progress reporting through the {partner.upper()} "
            f"partner initiative, including emissions inventory, target progress, and "
            f"action plan implementation updates."
        )

        governance_endorsement = (
            f"This pledge has been endorsed by the Board of Directors / Senior Leadership "
            f"of {org_name} and reflects our organization's commitment to contributing to "
            f"the global effort to limit warming to 1.5 degrees Celsius."
        )

        # Documents to generate
        documents = [
            "pledge_commitment_letter",
            "starting_line_self_assessment",
            "interim_target_summary",
            "baseline_emissions_report",
            "governance_endorsement_record",
            "partner_initiative_application",
        ]

        # Submission readiness
        gaps_count = eligibility.get("gaps_count", 0)
        submission_ready = (
            elig_status in (EligibilityStatus.ELIGIBLE.value, EligibilityStatus.CONDITIONAL.value)
            and baseline.get("is_valid", False)
            and targets.get("pathway_aligned", False)
        )

        if not submission_ready:
            reasons = []
            if elig_status == EligibilityStatus.CONDITIONAL.value:
                reasons.append(f"{gaps_count} eligibility gap(s) remain")
            if not baseline.get("is_valid", False):
                reasons.append("Baseline validation failed")
            if not targets.get("pathway_aligned", False):
                reasons.append("Target not aligned with 1.5C pathway")
            warnings.append(
                f"Submission not fully ready: {'; '.join(reasons)}"
            )

        # Submission deadline (12 months from current date)
        from datetime import timedelta
        submission_deadline = (_utcnow() + timedelta(days=365)).strftime("%Y-%m-%d")

        package_id = _new_uuid()
        outputs["package_id"] = package_id
        outputs["org_name"] = org_name
        outputs["actor_type"] = actor_type
        outputs["partner_initiative"] = partner
        outputs["pledge_statement"] = pledge_statement
        outputs["interim_target_summary"] = interim_summary
        outputs["long_term_target_summary"] = long_term_summary
        outputs["action_plan_commitment"] = action_plan_commitment
        outputs["reporting_commitment"] = reporting_commitment
        outputs["governance_endorsement"] = governance_endorsement
        outputs["documents_generated"] = documents
        outputs["documents_count"] = len(documents)
        outputs["submission_ready"] = submission_ready
        outputs["submission_deadline"] = submission_deadline
        outputs["eligibility_status"] = elig_status
        outputs["pledge_quality"] = eligibility.get("pledge_quality", "unknown")
        outputs["ambition_level"] = ambition

        records = len(documents)
        return outputs, warnings, errors, records

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _dependencies_met(
        self, phase: OnboardingPhase, result: PledgeOnboardingResult
    ) -> bool:
        """Check if all dependencies for a phase are met."""
        deps = PHASE_DEPENDENCIES.get(phase, [])
        for dep in deps:
            dep_result = result.phase_results.get(dep.value)
            if not dep_result or dep_result.status not in (
                PhaseStatus.COMPLETED, PhaseStatus.SKIPPED
            ):
                return False
        return True

    def _compute_quality_score(self, result: PledgeOnboardingResult) -> float:
        """Compute overall quality score for the onboarding."""
        total = len(PHASE_EXECUTION_ORDER)
        completed = len(result.phases_completed)
        skipped = len(result.phases_skipped)
        effective = completed + skipped * 0.5
        return round((effective / max(total, 1)) * 100.0, 1)

    def _extract_profile(self, ctx: Dict[str, Any]) -> Optional[OrganizationProfile]:
        """Extract organization profile from context."""
        data = ctx.get("organization_profiling", {})
        if not data:
            return None
        return OrganizationProfile(
            org_name=data.get("org_name", ""),
            actor_type=ActorType(data.get("actor_type", "corporate")),
            sector=data.get("sector", ""),
            country=data.get("country", ""),
            employee_count=data.get("employee_count", 0),
            revenue_usd=data.get("revenue_usd", 0.0),
            reporting_year=data.get("reporting_year", 2025),
            partner_initiatives=[self.config.partner_initiative],
        )

    def _extract_eligibility(self, ctx: Dict[str, Any]) -> Optional[EligibilityAssessment]:
        """Extract eligibility assessment from context."""
        data = ctx.get("eligibility_assessment", {})
        if not data:
            return None
        return EligibilityAssessment(
            status=EligibilityStatus(data.get("eligibility_status", "ineligible")),
            criteria_met=data.get("criteria_met", 0),
            criteria_total=data.get("criteria_total", 8),
            criteria_results=data.get("criteria_results", {}),
            pledge_quality=PledgeQuality(data.get("pledge_quality", "ineligible")),
            gaps=data.get("gaps", []),
            recommendations=data.get("recommendations", []),
            partner_alignment=data.get("partner_alignment", {}),
        )

    def _extract_baseline(self, ctx: Dict[str, Any]) -> Optional[BaselineEmissions]:
        """Extract baseline emissions from context."""
        data = ctx.get("baseline_validation", {})
        if not data:
            return None
        return BaselineEmissions(
            base_year=data.get("base_year", 2019),
            scope1_tco2e=data.get("scope1_tco2e", 0.0),
            scope2_tco2e=data.get("scope2_tco2e", 0.0),
            scope3_tco2e=data.get("scope3_tco2e", 0.0),
            total_tco2e=data.get("total_tco2e", 0.0),
            scope3_coverage_pct=data.get("scope3_coverage_pct", 0.0),
            scope3_categories_included=data.get("scope3_categories_included", []),
            data_quality_score=data.get("data_quality_score", 0.0),
            is_valid=data.get("is_valid", False),
            validation_issues=data.get("validation_issues", []),
        )

    def _extract_targets(self, ctx: Dict[str, Any]) -> Optional[TargetProposal]:
        """Extract target proposal from context."""
        data = ctx.get("target_proposal", {})
        if not data:
            return None
        return TargetProposal(
            interim_target_year=data.get("interim_target_year", 2030),
            interim_reduction_pct=data.get("interim_reduction_pct", 50.0),
            long_term_target_year=data.get("long_term_target_year", 2050),
            long_term_target=data.get("long_term_target", "net_zero"),
            annual_reduction_rate=data.get("annual_reduction_rate", 0.0),
            ambition_level=TargetAmbition(data.get("ambition_level", "insufficient")),
            scope_coverage=data.get("scope_coverage", {}),
            methodology=data.get("methodology", "absolute_contraction"),
            fair_share_assessment=data.get("fair_share_assessment", ""),
            pathway_aligned=data.get("pathway_aligned", False),
        )

    def _extract_commitment(self, ctx: Dict[str, Any]) -> Optional[CommitmentPackage]:
        """Extract commitment package from context."""
        data = ctx.get("commitment_documentation", {})
        if not data:
            return None
        return CommitmentPackage(
            package_id=data.get("package_id", ""),
            org_name=data.get("org_name", ""),
            actor_type=data.get("actor_type", ""),
            partner_initiative=data.get("partner_initiative", ""),
            pledge_statement=data.get("pledge_statement", ""),
            interim_target_summary=data.get("interim_target_summary", ""),
            long_term_target_summary=data.get("long_term_target_summary", ""),
            action_plan_commitment=data.get("action_plan_commitment", ""),
            reporting_commitment=data.get("reporting_commitment", ""),
            governance_endorsement=data.get("governance_endorsement", ""),
            documents_generated=data.get("documents_generated", []),
            submission_ready=data.get("submission_ready", False),
            submission_deadline=data.get("submission_deadline", ""),
        )
