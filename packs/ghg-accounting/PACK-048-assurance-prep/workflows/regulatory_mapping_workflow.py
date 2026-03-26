# -*- coding: utf-8 -*-
"""
Regulatory Mapping Workflow
====================================

4-phase workflow for GHG assurance regulatory obligation mapping covering
jurisdiction identification, requirement mapping, gap analysis, and
compliance planning within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. JurisdictionIdentification  -- Identify all jurisdictions where the
                                      organisation has GHG assurance obligations,
                                      based on operating locations, listing
                                      status, revenue thresholds, and sector
                                      classification.
    2. RequirementMapping          -- Map specific assurance requirements per
                                      jurisdiction, including standard, level,
                                      scope, timeline, and penalty provisions.
    3. GapAnalysis                 -- Analyse gaps between the organisation's
                                      current assurance readiness and each
                                      jurisdiction's requirements.
    4. CompliancePlanning          -- Produce a compliance plan with timelines,
                                      prioritised actions, resource requirements,
                                      and milestone tracking per jurisdiction.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    CSRD (2022/2464) - EU mandatory assurance requirements
    SEC Climate Disclosure Rules (2024) - US assurance requirements
    ISSB IFRS S2 (2023) - Global baseline assurance expectations
    UK Sustainability Disclosure Standards (2024) - UK requirements
    ESRS E1 (2024) - EU climate disclosure assurance
    Singapore SGX Sustainability Reporting (2024) - APAC requirements
    Australia ASRS (2024) - Australian sustainability assurance

Schedule: Annually or upon regulatory change / corporate restructuring
Estimated duration: 2-3 weeks

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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


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


class RegulatoryMappingPhase(str, Enum):
    """Regulatory mapping workflow phases."""

    JURISDICTION_IDENTIFICATION = "jurisdiction_identification"
    REQUIREMENT_MAPPING = "requirement_mapping"
    GAP_ANALYSIS = "gap_analysis"
    COMPLIANCE_PLANNING = "compliance_planning"


class Jurisdiction(str, Enum):
    """Supported jurisdictions."""

    EU = "eu"
    US = "us"
    UK = "uk"
    SINGAPORE = "singapore"
    AUSTRALIA = "australia"
    JAPAN = "japan"
    CANADA = "canada"
    HONG_KONG = "hong_kong"
    GLOBAL_VOLUNTARY = "global_voluntary"


class AssuranceRequirementLevel(str, Enum):
    """Required assurance level per jurisdiction."""

    MANDATORY_REASONABLE = "mandatory_reasonable"
    MANDATORY_LIMITED = "mandatory_limited"
    RECOMMENDED = "recommended"
    VOLUNTARY = "voluntary"
    NOT_REQUIRED = "not_required"


class ComplianceStatus(str, Enum):
    """Organisation's compliance status for a jurisdiction."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class ActionPriority(str, Enum):
    """Priority for compliance actions."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# REGULATORY REGISTER (Zero-Hallucination Reference Data)
# =============================================================================

REGULATORY_REGISTER: Dict[str, Dict[str, Any]] = {
    "eu": {
        "jurisdiction_name": "European Union",
        "regulation": "CSRD / ESRS E1",
        "standard": "ISAE 3000 / ISAE 3410",
        "current_level": "mandatory_limited",
        "future_level": "mandatory_reasonable",
        "future_level_date": "2028-01-01",
        "scope_requirements": ["scope_1", "scope_2", "scope_3"],
        "first_reporting_year": 2024,
        "penalties": "Administrative fines up to EUR 10M or 5% of turnover",
        "thresholds": {
            "employees": 250,
            "revenue_eur_m": 50.0,
            "balance_sheet_eur_m": 25.0,
        },
    },
    "us": {
        "jurisdiction_name": "United States",
        "regulation": "SEC Climate Disclosure Rules",
        "standard": "PCAOB / AICPA attestation standards",
        "current_level": "mandatory_limited",
        "future_level": "mandatory_reasonable",
        "future_level_date": "2029-01-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2025,
        "penalties": "SEC enforcement actions; civil penalties",
        "thresholds": {
            "sec_filer_category": "large_accelerated",
            "market_cap_usd_m": 700.0,
        },
    },
    "uk": {
        "jurisdiction_name": "United Kingdom",
        "regulation": "UK Sustainability Disclosure Standards",
        "standard": "ISAE 3410",
        "current_level": "recommended",
        "future_level": "mandatory_limited",
        "future_level_date": "2026-01-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2026,
        "penalties": "FCA enforcement; premium listing requirements",
        "thresholds": {
            "employees": 500,
            "revenue_gbp_m": 500.0,
        },
    },
    "singapore": {
        "jurisdiction_name": "Singapore",
        "regulation": "SGX Sustainability Reporting",
        "standard": "ISAE 3410 / AA1000AS",
        "current_level": "recommended",
        "future_level": "mandatory_limited",
        "future_level_date": "2027-01-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2027,
        "penalties": "SGX listing rule enforcement",
        "thresholds": {
            "market_cap_sgd_m": 500.0,
        },
    },
    "australia": {
        "jurisdiction_name": "Australia",
        "regulation": "Australian Sustainability Reporting Standards",
        "standard": "ASAE 3410 / ISAE 3410",
        "current_level": "mandatory_limited",
        "future_level": "mandatory_reasonable",
        "future_level_date": "2030-01-01",
        "scope_requirements": ["scope_1", "scope_2", "scope_3"],
        "first_reporting_year": 2025,
        "penalties": "ASIC enforcement; civil penalties",
        "thresholds": {
            "employees": 500,
            "revenue_aud_m": 500.0,
        },
    },
    "japan": {
        "jurisdiction_name": "Japan",
        "regulation": "SSBJ Sustainability Disclosure Standards",
        "standard": "JICPA attestation standards",
        "current_level": "voluntary",
        "future_level": "mandatory_limited",
        "future_level_date": "2027-04-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2027,
        "penalties": "FSA enforcement",
        "thresholds": {
            "listed": True,
            "prime_market": True,
        },
    },
    "canada": {
        "jurisdiction_name": "Canada",
        "regulation": "CSA Climate Disclosure Requirements",
        "standard": "CPA Canada / ISAE 3410",
        "current_level": "voluntary",
        "future_level": "mandatory_limited",
        "future_level_date": "2027-01-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2027,
        "penalties": "CSA enforcement actions",
        "thresholds": {
            "reporting_issuer": True,
        },
    },
    "hong_kong": {
        "jurisdiction_name": "Hong Kong",
        "regulation": "HKEX ESG Listing Rules",
        "standard": "ISAE 3000 / ISAE 3410",
        "current_level": "recommended",
        "future_level": "mandatory_limited",
        "future_level_date": "2026-01-01",
        "scope_requirements": ["scope_1", "scope_2"],
        "first_reporting_year": 2026,
        "penalties": "HKEX listing rule enforcement",
        "thresholds": {
            "listed": True,
        },
    },
    "global_voluntary": {
        "jurisdiction_name": "Global Voluntary Frameworks",
        "regulation": "CDP / SBTi / TCFD",
        "standard": "ISAE 3410 / AA1000AS / ISO 14064-3",
        "current_level": "voluntary",
        "future_level": "voluntary",
        "future_level_date": "",
        "scope_requirements": ["scope_1", "scope_2", "scope_3"],
        "first_reporting_year": 2020,
        "penalties": "Reputational; scoring impact",
        "thresholds": {},
    },
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


class JurisdictionRecord(BaseModel):
    """Record of an identified jurisdiction with assurance obligations."""

    jurisdiction: Jurisdiction = Field(...)
    jurisdiction_name: str = Field(default="")
    regulation: str = Field(default="")
    applicable_standard: str = Field(default="")
    current_requirement: AssuranceRequirementLevel = Field(
        default=AssuranceRequirementLevel.NOT_REQUIRED,
    )
    future_requirement: AssuranceRequirementLevel = Field(
        default=AssuranceRequirementLevel.NOT_REQUIRED,
    )
    future_requirement_date: str = Field(default="")
    scope_requirements: List[str] = Field(default_factory=list)
    first_reporting_year: int = Field(default=2025)
    penalties: str = Field(default="")
    meets_thresholds: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class RequirementGap(BaseModel):
    """Gap between current state and jurisdictional requirement."""

    gap_id: str = Field(default_factory=lambda: f"rgap-{_new_uuid()[:8]}")
    jurisdiction: Jurisdiction = Field(...)
    requirement_area: str = Field(default="")
    current_state: str = Field(default="")
    required_state: str = Field(default="")
    gap_description: str = Field(default="")
    compliance_status: ComplianceStatus = Field(default=ComplianceStatus.NON_COMPLIANT)
    effort_weeks: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")


class ComplianceAction(BaseModel):
    """A compliance action to close a regulatory gap."""

    action_id: str = Field(default_factory=lambda: f"act-{_new_uuid()[:8]}")
    jurisdiction: Jurisdiction = Field(...)
    gap_id: str = Field(default="")
    action_description: str = Field(default="")
    priority: ActionPriority = Field(default=ActionPriority.MEDIUM)
    owner_role: str = Field(default="")
    target_completion_date: str = Field(default="")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class RegulatoryMappingInput(BaseModel):
    """Input data model for RegulatoryMappingWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    operating_jurisdictions: List[str] = Field(
        default_factory=lambda: ["eu"],
        description="Jurisdictions where organisation operates",
    )
    listing_jurisdictions: List[str] = Field(
        default_factory=list,
        description="Jurisdictions where organisation is listed",
    )
    employee_count: int = Field(default=0, ge=0)
    revenue_usd_m: float = Field(default=0.0, ge=0.0)
    market_cap_usd_m: float = Field(default=0.0, ge=0.0)
    current_assurance_level: str = Field(default="none")
    current_assurance_standard: str = Field(default="")
    current_scope_coverage: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
    )
    reporting_period: str = Field(default="2025")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RegulatoryMappingResult(BaseModel):
    """Complete result from regulatory mapping workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_mapping")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    jurisdictions: List[JurisdictionRecord] = Field(default_factory=list)
    gaps: List[RequirementGap] = Field(default_factory=list)
    actions: List[ComplianceAction] = Field(default_factory=list)
    total_jurisdictions: int = Field(default=0)
    mandatory_jurisdictions: int = Field(default=0)
    total_gaps: int = Field(default=0)
    total_actions: int = Field(default=0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatoryMappingWorkflow:
    """
    4-phase workflow for regulatory obligation mapping.

    Identifies jurisdictions with assurance obligations, maps specific
    requirements, analyses gaps against current state, and produces a
    compliance plan with prioritised actions.

    Zero-hallucination: all regulatory requirements are drawn from a
    deterministic register; gap assessments use structured criteria;
    no LLM calls in compliance logic; SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _jurisdictions: Identified jurisdictions.
        _gaps: Requirement gaps.
        _actions: Compliance actions.

    Example:
        >>> wf = RegulatoryMappingWorkflow()
        >>> inp = RegulatoryMappingInput(
        ...     organization_id="org-001",
        ...     operating_jurisdictions=["eu", "us"],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[RegulatoryMappingPhase] = [
        RegulatoryMappingPhase.JURISDICTION_IDENTIFICATION,
        RegulatoryMappingPhase.REQUIREMENT_MAPPING,
        RegulatoryMappingPhase.GAP_ANALYSIS,
        RegulatoryMappingPhase.COMPLIANCE_PLANNING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RegulatoryMappingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._jurisdictions: List[JurisdictionRecord] = []
        self._gaps: List[RequirementGap] = []
        self._actions: List[ComplianceAction] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: RegulatoryMappingInput,
    ) -> RegulatoryMappingResult:
        """
        Execute the 4-phase regulatory mapping workflow.

        Args:
            input_data: Organisation jurisdiction, size, and current compliance.

        Returns:
            RegulatoryMappingResult with jurisdictions, gaps, and actions.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting regulatory mapping %s org=%s jurisdictions=%s",
            self.workflow_id, input_data.organization_id,
            input_data.operating_jurisdictions,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_jurisdiction_identification,
            self._phase_2_requirement_mapping,
            self._phase_3_gap_analysis,
            self._phase_4_compliance_planning,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Regulatory mapping failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        mandatory = sum(
            1 for j in self._jurisdictions
            if j.current_requirement in (
                AssuranceRequirementLevel.MANDATORY_LIMITED,
                AssuranceRequirementLevel.MANDATORY_REASONABLE,
            )
        )

        result = RegulatoryMappingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            jurisdictions=self._jurisdictions,
            gaps=self._gaps,
            actions=self._actions,
            total_jurisdictions=len(self._jurisdictions),
            mandatory_jurisdictions=mandatory,
            total_gaps=len(self._gaps),
            total_actions=len(self._actions),
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Regulatory mapping %s completed in %.2fs status=%s "
            "jurisdictions=%d mandatory=%d gaps=%d actions=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._jurisdictions), mandatory,
            len(self._gaps), len(self._actions),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Jurisdiction Identification
    # -------------------------------------------------------------------------

    async def _phase_1_jurisdiction_identification(
        self, input_data: RegulatoryMappingInput,
    ) -> PhaseResult:
        """Identify jurisdictions with assurance obligations."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        all_jurisdictions = set(input_data.operating_jurisdictions)
        all_jurisdictions.update(input_data.listing_jurisdictions)
        all_jurisdictions.add("global_voluntary")

        self._jurisdictions = []
        for j_str in all_jurisdictions:
            reg_data = REGULATORY_REGISTER.get(j_str)
            if not reg_data:
                warnings.append(f"Unknown jurisdiction: {j_str}")
                continue

            try:
                jurisdiction = Jurisdiction(j_str)
            except ValueError:
                continue

            # Check if organisation meets thresholds
            thresholds = reg_data.get("thresholds", {})
            meets = True
            if "employees" in thresholds:
                if input_data.employee_count < thresholds["employees"]:
                    meets = False
            if "revenue_eur_m" in thresholds:
                # Approximate EUR from USD
                rev_eur = input_data.revenue_usd_m * 0.92
                if rev_eur < thresholds["revenue_eur_m"]:
                    meets = False
            if "market_cap_usd_m" in thresholds:
                if input_data.market_cap_usd_m < thresholds["market_cap_usd_m"]:
                    meets = False

            try:
                current_req = AssuranceRequirementLevel(reg_data["current_level"])
            except ValueError:
                current_req = AssuranceRequirementLevel.NOT_REQUIRED

            try:
                future_req = AssuranceRequirementLevel(reg_data["future_level"])
            except ValueError:
                future_req = AssuranceRequirementLevel.NOT_REQUIRED

            j_hash = {"jurisdiction": j_str, "meets": meets, "level": current_req.value}
            record = JurisdictionRecord(
                jurisdiction=jurisdiction,
                jurisdiction_name=reg_data["jurisdiction_name"],
                regulation=reg_data["regulation"],
                applicable_standard=reg_data["standard"],
                current_requirement=current_req,
                future_requirement=future_req,
                future_requirement_date=reg_data.get("future_level_date", ""),
                scope_requirements=reg_data.get("scope_requirements", []),
                first_reporting_year=reg_data.get("first_reporting_year", 2025),
                penalties=reg_data.get("penalties", ""),
                meets_thresholds=meets,
                provenance_hash=_compute_hash(j_hash),
            )
            self._jurisdictions.append(record)

        outputs["jurisdictions_identified"] = len(self._jurisdictions)
        outputs["meeting_thresholds"] = sum(
            1 for j in self._jurisdictions if j.meets_thresholds
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 JurisdictionIdentification: %d jurisdictions",
            len(self._jurisdictions),
        )
        return PhaseResult(
            phase_name="jurisdiction_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Requirement Mapping
    # -------------------------------------------------------------------------

    async def _phase_2_requirement_mapping(
        self, input_data: RegulatoryMappingInput,
    ) -> PhaseResult:
        """Map specific requirements per jurisdiction."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        mandatory_count = 0
        for j_rec in self._jurisdictions:
            if j_rec.current_requirement in (
                AssuranceRequirementLevel.MANDATORY_LIMITED,
                AssuranceRequirementLevel.MANDATORY_REASONABLE,
            ) and j_rec.meets_thresholds:
                mandatory_count += 1

        outputs["mandatory_jurisdictions"] = mandatory_count
        outputs["recommended_jurisdictions"] = sum(
            1 for j in self._jurisdictions
            if j.current_requirement == AssuranceRequirementLevel.RECOMMENDED
        )
        outputs["voluntary_jurisdictions"] = sum(
            1 for j in self._jurisdictions
            if j.current_requirement == AssuranceRequirementLevel.VOLUNTARY
        )

        # Scope coverage analysis
        all_required_scopes: set = set()
        for j_rec in self._jurisdictions:
            if j_rec.meets_thresholds:
                all_required_scopes.update(j_rec.scope_requirements)

        outputs["required_scopes"] = sorted(all_required_scopes)
        outputs["scope_3_required"] = "scope_3" in all_required_scopes

        if "scope_3" in all_required_scopes and "scope_3" not in input_data.current_scope_coverage:
            warnings.append(
                "Scope 3 assurance required by at least one jurisdiction "
                "but not currently covered"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 RequirementMapping: mandatory=%d scopes=%s",
            mandatory_count, sorted(all_required_scopes),
        )
        return PhaseResult(
            phase_name="requirement_mapping", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_3_gap_analysis(
        self, input_data: RegulatoryMappingInput,
    ) -> PhaseResult:
        """Analyse gaps between current state and requirements."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []

        for j_rec in self._jurisdictions:
            if not j_rec.meets_thresholds:
                continue
            if j_rec.current_requirement == AssuranceRequirementLevel.NOT_REQUIRED:
                continue

            # Gap 1: Assurance level
            current_level = input_data.current_assurance_level
            required_level = j_rec.current_requirement.value

            if current_level == "none" or (
                "reasonable" in required_level and current_level == "limited"
            ):
                gap_data = {
                    "jurisdiction": j_rec.jurisdiction.value,
                    "area": "assurance_level",
                }
                self._gaps.append(RequirementGap(
                    jurisdiction=j_rec.jurisdiction,
                    requirement_area="Assurance Level",
                    current_state=current_level,
                    required_state=required_level,
                    gap_description=(
                        f"{j_rec.jurisdiction_name} requires {required_level} "
                        f"assurance; current state is '{current_level}'"
                    ),
                    compliance_status=ComplianceStatus.NON_COMPLIANT,
                    effort_weeks=12 if current_level == "none" else 8,
                    provenance_hash=_compute_hash(gap_data),
                ))

            # Gap 2: Scope coverage
            for req_scope in j_rec.scope_requirements:
                if req_scope not in input_data.current_scope_coverage:
                    scope_gap_data = {
                        "jurisdiction": j_rec.jurisdiction.value,
                        "area": f"scope_{req_scope}",
                    }
                    self._gaps.append(RequirementGap(
                        jurisdiction=j_rec.jurisdiction,
                        requirement_area=f"Scope Coverage ({req_scope})",
                        current_state="Not covered",
                        required_state=f"{req_scope} assurance required",
                        gap_description=(
                            f"{j_rec.jurisdiction_name} requires {req_scope} "
                            f"in assurance scope"
                        ),
                        compliance_status=ComplianceStatus.NON_COMPLIANT,
                        effort_weeks=8 if req_scope == "scope_3" else 4,
                        provenance_hash=_compute_hash(scope_gap_data),
                    ))

            # Gap 3: Standard alignment
            if (input_data.current_assurance_standard and
                    input_data.current_assurance_standard not in j_rec.applicable_standard):
                std_gap_data = {
                    "jurisdiction": j_rec.jurisdiction.value,
                    "area": "standard_alignment",
                }
                self._gaps.append(RequirementGap(
                    jurisdiction=j_rec.jurisdiction,
                    requirement_area="Assurance Standard",
                    current_state=input_data.current_assurance_standard,
                    required_state=j_rec.applicable_standard,
                    gap_description=(
                        f"{j_rec.jurisdiction_name} requires {j_rec.applicable_standard}; "
                        f"current standard is {input_data.current_assurance_standard}"
                    ),
                    compliance_status=ComplianceStatus.PARTIALLY_COMPLIANT,
                    effort_weeks=4,
                    provenance_hash=_compute_hash(std_gap_data),
                ))

        outputs["total_gaps"] = len(self._gaps)
        outputs["non_compliant_gaps"] = sum(
            1 for g in self._gaps if g.compliance_status == ComplianceStatus.NON_COMPLIANT
        )
        outputs["partially_compliant_gaps"] = sum(
            1 for g in self._gaps if g.compliance_status == ComplianceStatus.PARTIALLY_COMPLIANT
        )

        elapsed = time.monotonic() - started
        self.logger.info("Phase 3 GapAnalysis: %d gaps identified", len(self._gaps))
        return PhaseResult(
            phase_name="gap_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Planning
    # -------------------------------------------------------------------------

    async def _phase_4_compliance_planning(
        self, input_data: RegulatoryMappingInput,
    ) -> PhaseResult:
        """Produce compliance plan with timelines and actions."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._actions = []
        cost_map: Dict[str, float] = {
            "assurance_level": 75000.0,
            "scope_coverage": 40000.0,
            "standard_alignment": 20000.0,
        }
        owner_map: Dict[str, str] = {
            "assurance_level": "Sustainability Director",
            "scope_coverage": "GHG Technical Specialist",
            "standard_alignment": "Reporting Manager",
        }

        for gap in self._gaps:
            area_key = gap.requirement_area.lower().replace(" ", "_")
            if "assurance" in area_key and "level" in area_key:
                area_key = "assurance_level"
            elif "scope" in area_key:
                area_key = "scope_coverage"
            elif "standard" in area_key:
                area_key = "standard_alignment"

            # Priority based on compliance status and jurisdiction importance
            if gap.compliance_status == ComplianceStatus.NON_COMPLIANT:
                priority = ActionPriority.HIGH
                if gap.jurisdiction in (Jurisdiction.EU, Jurisdiction.US):
                    priority = ActionPriority.URGENT
            else:
                priority = ActionPriority.MEDIUM

            estimated_cost = cost_map.get(area_key, 30000.0)

            action_data = {
                "gap": gap.gap_id, "priority": priority.value,
                "jurisdiction": gap.jurisdiction.value,
            }
            action = ComplianceAction(
                jurisdiction=gap.jurisdiction,
                gap_id=gap.gap_id,
                action_description=(
                    f"Close gap: {gap.gap_description}. "
                    f"Move from '{gap.current_state}' to '{gap.required_state}'."
                ),
                priority=priority,
                owner_role=owner_map.get(area_key, "Sustainability Team"),
                estimated_cost_usd=estimated_cost,
                provenance_hash=_compute_hash(action_data),
            )
            self._actions.append(action)

        # Sort by priority
        priority_order = {
            ActionPriority.URGENT: 0, ActionPriority.HIGH: 1,
            ActionPriority.MEDIUM: 2, ActionPriority.LOW: 3,
        }
        self._actions.sort(key=lambda a: priority_order.get(a.priority, 99))

        total_cost = sum(a.estimated_cost_usd for a in self._actions)

        outputs["total_actions"] = len(self._actions)
        outputs["urgent_actions"] = sum(
            1 for a in self._actions if a.priority == ActionPriority.URGENT
        )
        outputs["estimated_total_cost_usd"] = round(total_cost, 2)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 CompliancePlanning: %d actions, cost=%.0f USD",
            len(self._actions), total_cost,
        )
        return PhaseResult(
            phase_name="compliance_planning", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: RegulatoryMappingInput,
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
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._jurisdictions = []
        self._gaps = []
        self._actions = []

    def _compute_provenance(self, result: RegulatoryMappingResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.total_jurisdictions}|{result.total_gaps}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
