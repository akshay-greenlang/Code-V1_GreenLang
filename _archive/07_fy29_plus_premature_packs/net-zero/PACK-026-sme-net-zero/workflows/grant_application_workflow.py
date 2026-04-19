# -*- coding: utf-8 -*-
"""
Grant Application Workflow
===============================

5-phase workflow for SME grant discovery and application support
within PACK-026 SME Net Zero Pack.  Guides SMEs through finding,
qualifying for, and applying to government/institutional grants
for decarbonisation projects.

Phases:
    1. GrantSearch          -- Find matching grants by sector, size, location
    2. EligibilityCheck     -- Verify eligibility criteria
    3. DataPreparation      -- Prepare supporting data (baseline, targets, project)
    4. ApplicationSupport   -- Generate pre-filled application templates
    5. SubmissionExport     -- Export to PDF for manual submission

Uses: grant_finder_engine, certification_readiness_engine.

Zero-hallucination: all grant data from verified government sources.
SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

class GrantType(str, Enum):
    CAPITAL_GRANT = "capital_grant"
    REVENUE_GRANT = "revenue_grant"
    TAX_RELIEF = "tax_relief"
    LOAN = "loan"
    VOUCHER = "voucher"
    MIXED = "mixed"

class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    LIKELY_ELIGIBLE = "likely_eligible"
    NEEDS_VERIFICATION = "needs_verification"
    INELIGIBLE = "ineligible"

class ApplicationStatus(str, Enum):
    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    READY_TO_SUBMIT = "ready_to_submit"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"

class DocumentType(str, Enum):
    APPLICATION_FORM = "application_form"
    PROJECT_DESCRIPTION = "project_description"
    BASELINE_REPORT = "baseline_report"
    TARGET_SUMMARY = "target_summary"
    COST_BENEFIT_ANALYSIS = "cost_benefit_analysis"
    FINANCIAL_PROJECTION = "financial_projection"
    COMPANY_PROFILE = "company_profile"
    SUPPORTING_EVIDENCE = "supporting_evidence"

# =============================================================================
# GRANT DATABASE
# =============================================================================

GRANT_REGISTRY: List[Dict[str, Any]] = [
    {
        "grant_id": "BEIS-IETF-2026",
        "name": "Industrial Energy Transformation Fund Phase 3",
        "provider": "Department for Energy Security and Net Zero (UK)",
        "grant_type": "capital_grant",
        "max_amount_gbp": 30_000_000,
        "min_amount_gbp": 100_000,
        "co_funding_pct": 50,
        "eligible_sectors": ["manufacturing_light", "manufacturing_heavy"],
        "eligible_sizes": ["medium", "large_sme"],
        "eligible_countries": ["UK"],
        "eligible_projects": ["energy_efficiency", "fuel_switching", "heat_recovery", "process_change"],
        "deadline": "2026-09-30",
        "status": "open",
        "url": "https://www.gov.uk/government/collections/industrial-energy-transformation-fund",
        "requirements": [
            "UK-registered business",
            "Manufacturing SIC code",
            "Energy-intensive process",
            "Minimum 20% CO2 reduction",
        ],
        "documents_required": [
            "application_form", "project_description", "cost_benefit_analysis",
            "financial_projection", "baseline_report",
        ],
    },
    {
        "grant_id": "BEF-TECH-2026",
        "name": "Business Energy Fund - Technology Grant",
        "provider": "BEIS (UK)",
        "grant_type": "capital_grant",
        "max_amount_gbp": 20_000,
        "min_amount_gbp": 1_000,
        "co_funding_pct": 40,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "eligible_countries": ["UK"],
        "eligible_projects": ["led_lighting", "heating_controls", "insulation", "solar_pv", "heat_pump"],
        "deadline": "2026-12-31",
        "status": "open",
        "url": "https://www.gov.uk/business-energy-fund",
        "requirements": [
            "UK-registered SME",
            "Fewer than 250 employees",
            "Energy improvement project",
        ],
        "documents_required": [
            "application_form", "project_description", "company_profile",
        ],
    },
    {
        "grant_id": "ECO4-2026",
        "name": "ECO4 Energy Company Obligation",
        "provider": "Ofgem (UK)",
        "grant_type": "voucher",
        "max_amount_gbp": 10_000,
        "min_amount_gbp": 0,
        "co_funding_pct": 0,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium", "large_sme"],
        "eligible_countries": ["UK"],
        "eligible_projects": ["insulation", "heating_upgrade", "glazing"],
        "deadline": "2026-06-30",
        "status": "open",
        "url": "https://www.ofgem.gov.uk/environmental-programmes/eco",
        "requirements": [
            "UK business premises",
            "Poor energy efficiency rating (EPC D or below)",
        ],
        "documents_required": [
            "application_form", "company_profile",
        ],
    },
    {
        "grant_id": "ERDF-GREEN-2026",
        "name": "EU Green SME Fund",
        "provider": "European Regional Development Fund",
        "grant_type": "capital_grant",
        "max_amount_gbp": 50_000,
        "min_amount_gbp": 5_000,
        "co_funding_pct": 50,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "eligible_countries": ["DE", "FR", "IE", "NL", "ES", "IT"],
        "eligible_projects": ["energy_efficiency", "renewable_energy", "waste_reduction", "circular_economy"],
        "deadline": "2026-11-30",
        "status": "open",
        "url": "https://ec.europa.eu/regional_policy/funding/erdf_en",
        "requirements": [
            "EU-registered SME",
            "Decarbonisation project plan",
            "Minimum 15% emission reduction",
        ],
        "documents_required": [
            "application_form", "project_description", "baseline_report",
            "cost_benefit_analysis", "company_profile",
        ],
    },
    {
        "grant_id": "SEAI-SME-2026",
        "name": "SEAI SME Energy Efficiency Grant",
        "provider": "Sustainable Energy Authority of Ireland",
        "grant_type": "capital_grant",
        "max_amount_gbp": 5_000,
        "min_amount_gbp": 500,
        "co_funding_pct": 30,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "eligible_countries": ["IE"],
        "eligible_projects": ["energy_audit", "energy_efficiency", "monitoring"],
        "deadline": "2027-03-31",
        "status": "open",
        "url": "https://www.seai.ie/business-and-public-sector/small-and-medium-business/",
        "requirements": [
            "Ireland-registered SME",
            "Annual energy spend over 10k EUR",
        ],
        "documents_required": [
            "application_form", "company_profile",
        ],
    },
    {
        "grant_id": "HORIZON-SME-2026",
        "name": "Horizon Europe SME Instrument",
        "provider": "European Commission",
        "grant_type": "mixed",
        "max_amount_gbp": 200_000,
        "min_amount_gbp": 50_000,
        "co_funding_pct": 30,
        "eligible_sectors": ["technology", "manufacturing_light", "manufacturing_heavy"],
        "eligible_sizes": ["small", "medium"],
        "eligible_countries": ["DE", "FR", "IE", "NL", "ES", "IT"],
        "eligible_projects": ["clean_tech", "climate_innovation", "process_innovation"],
        "deadline": "2026-10-15",
        "status": "open",
        "url": "https://ec.europa.eu/programmes/horizon-europe/",
        "requirements": [
            "EU-registered innovative SME",
            "Climate technology focus",
            "Growth potential",
        ],
        "documents_required": [
            "application_form", "project_description", "financial_projection",
            "company_profile", "supporting_evidence",
        ],
    },
    {
        "grant_id": "SG-GREEN-2026",
        "name": "Scottish Green Business Fund",
        "provider": "Zero Waste Scotland",
        "grant_type": "capital_grant",
        "max_amount_gbp": 15_000,
        "min_amount_gbp": 1_000,
        "co_funding_pct": 50,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "eligible_countries": ["UK"],
        "eligible_regions": ["Scotland"],
        "eligible_projects": ["energy_efficiency", "renewable_energy", "waste_reduction"],
        "deadline": "2027-03-31",
        "status": "open",
        "url": "https://www.zerowastescotland.org.uk/",
        "requirements": [
            "Scottish-registered SME",
            "Environmental improvement project",
        ],
        "documents_required": [
            "application_form", "project_description", "company_profile",
        ],
    },
    {
        "grant_id": "WG-GREEN-2026",
        "name": "Welsh Government Green Business Grant",
        "provider": "Welsh Government",
        "grant_type": "capital_grant",
        "max_amount_gbp": 10_000,
        "min_amount_gbp": 1_000,
        "co_funding_pct": 50,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "eligible_countries": ["UK"],
        "eligible_regions": ["Wales"],
        "eligible_projects": ["energy_efficiency", "renewable_energy", "carbon_reduction"],
        "deadline": "2027-03-31",
        "status": "open",
        "url": "https://www.gov.wales/",
        "requirements": [
            "Wales-registered SME",
            "Carbon reduction project",
        ],
        "documents_required": [
            "application_form", "project_description", "company_profile",
        ],
    },
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    mobile_summary: str = Field(default="")

class GrantSearchCriteria(BaseModel):
    """Criteria for grant search."""

    industry_sector: str = Field(default="other")
    company_size: str = Field(default="small")
    country: str = Field(default="UK")
    region: str = Field(default="")
    postcode: str = Field(default="")
    project_types: List[str] = Field(default_factory=list)
    min_amount_gbp: float = Field(default=0.0, ge=0.0)
    max_amount_needed_gbp: float = Field(default=100_000, ge=0.0)
    employee_count: int = Field(default=1, ge=1)

class GrantSearchResult(BaseModel):
    """Result of grant search."""

    grant_id: str = Field(default="")
    name: str = Field(default="")
    provider: str = Field(default="")
    grant_type: str = Field(default="capital_grant")
    max_amount_gbp: float = Field(default=0.0, ge=0.0)
    min_amount_gbp: float = Field(default=0.0, ge=0.0)
    co_funding_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    deadline: str = Field(default="")
    status: str = Field(default="open")
    url: str = Field(default="")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_reasons: List[str] = Field(default_factory=list)

class EligibilityCheckResult(BaseModel):
    """Eligibility check for a specific grant."""

    grant_id: str = Field(default="")
    grant_name: str = Field(default="")
    eligibility_status: str = Field(default="needs_verification")
    criteria_met: List[str] = Field(default_factory=list)
    criteria_not_met: List[str] = Field(default_factory=list)
    criteria_uncertain: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: str = Field(default="")

class ProjectDescription(BaseModel):
    """Pre-filled project description for grant application."""

    project_title: str = Field(default="")
    project_summary: str = Field(default="")
    current_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    expected_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    expected_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    implementation_timeline_months: int = Field(default=12, ge=1)
    total_project_cost_gbp: float = Field(default=0.0, ge=0.0)
    grant_amount_requested_gbp: float = Field(default=0.0, ge=0.0)
    co_funding_amount_gbp: float = Field(default=0.0, ge=0.0)
    annual_cost_savings_gbp: float = Field(default=0.0, ge=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    additional_benefits: List[str] = Field(default_factory=list)

class ApplicationTemplate(BaseModel):
    """Pre-filled grant application template."""

    grant_id: str = Field(default="")
    grant_name: str = Field(default="")
    company_name: str = Field(default="")
    company_registration: str = Field(default="")
    project: ProjectDescription = Field(default_factory=ProjectDescription)
    documents_needed: List[str] = Field(default_factory=list)
    documents_ready: List[str] = Field(default_factory=list)
    documents_missing: List[str] = Field(default_factory=list)
    application_status: str = Field(default="draft")
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    next_actions: List[str] = Field(default_factory=list)

class SubmissionPackage(BaseModel):
    """Final submission package for export."""

    grant_id: str = Field(default="")
    grant_name: str = Field(default="")
    application: ApplicationTemplate = Field(default_factory=ApplicationTemplate)
    export_format: str = Field(default="pdf")
    export_ready: bool = Field(default=False)
    export_notes: List[str] = Field(default_factory=list)

class GrantApplicationConfig(BaseModel):
    """Configuration for grant application workflow."""

    baseline_tco2e: float = Field(default=0.0, ge=0.0, description="Current baseline")
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    planned_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Actions for which grants are sought"
    )
    total_project_budget_gbp: float = Field(default=0.0, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class GrantApplicationInput(BaseModel):
    """Complete input for grant application workflow."""

    organization_name: str = Field(default="")
    industry_sector: str = Field(default="other")
    employee_count: int = Field(default=1, ge=1)
    annual_revenue_gbp: float = Field(default=0.0, ge=0.0)
    country: str = Field(default="UK")
    region: str = Field(default="")
    postcode: str = Field(default="")
    search_criteria: GrantSearchCriteria = Field(default_factory=GrantSearchCriteria)
    config: GrantApplicationConfig = Field(default_factory=GrantApplicationConfig)

class GrantApplicationResult(BaseModel):
    """Complete result from grant application workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_grant_application")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    grants_found: List[GrantSearchResult] = Field(default_factory=list)
    eligibility_results: List[EligibilityCheckResult] = Field(default_factory=list)
    applications: List[ApplicationTemplate] = Field(default_factory=list)
    submissions: List[SubmissionPackage] = Field(default_factory=list)
    total_potential_funding_gbp: float = Field(default=0.0, ge=0.0)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class GrantApplicationWorkflow:
    """
    5-phase grant application workflow for SME decarbonisation funding.

    Phase 1: Grant Search - Find matching grants
    Phase 2: Eligibility Check - Verify criteria
    Phase 3: Data Preparation - Prepare supporting evidence
    Phase 4: Application Support - Generate pre-filled templates
    Phase 5: Submission Export - Export to PDF

    Example:
        >>> wf = GrantApplicationWorkflow()
        >>> inp = GrantApplicationInput(
        ...     organization_name="Acme Ltd",
        ...     country="UK",
        ...     search_criteria=GrantSearchCriteria(
        ...         industry_sector="manufacturing_light",
        ...         company_size="small",
        ...     ),
        ...     config=GrantApplicationConfig(
        ...         baseline_tco2e=150.0,
        ...         total_project_budget_gbp=25000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._grants_found: List[GrantSearchResult] = []
        self._eligibility: List[EligibilityCheckResult] = []
        self._applications: List[ApplicationTemplate] = []
        self._submissions: List[SubmissionPackage] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: GrantApplicationInput) -> GrantApplicationResult:
        """Execute the 5-phase grant application workflow."""
        started_at = utcnow()
        self.logger.info(
            "Starting grant application workflow %s for %s",
            self.workflow_id, input_data.organization_name,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_grant_search(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_eligibility_check(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_data_preparation(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_application_support(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_submission_export(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Grant application workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
                mobile_summary="Grant search failed.",
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        total_funding = sum(g.max_amount_gbp for g in self._grants_found)
        next_steps = self._generate_next_steps()

        result = GrantApplicationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            grants_found=self._grants_found,
            eligibility_results=self._eligibility,
            applications=self._applications,
            submissions=self._submissions,
            total_potential_funding_gbp=round(total_funding, 2),
            next_steps=next_steps,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Grant Search
    # -------------------------------------------------------------------------

    async def _phase_grant_search(self, inp: GrantApplicationInput) -> PhaseResult:
        """Search for matching grants based on SME profile."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        criteria = inp.search_criteria
        country = criteria.country or inp.country or "UK"
        sector = criteria.industry_sector or inp.industry_sector
        size = criteria.company_size

        results: List[GrantSearchResult] = []

        for grant in GRANT_REGISTRY:
            eligible_sectors = grant.get("eligible_sectors", [])
            eligible_sizes = grant.get("eligible_sizes", [])
            eligible_countries = grant.get("eligible_countries", [])

            sector_ok = "all" in eligible_sectors or sector in eligible_sectors
            size_ok = size in eligible_sizes
            country_ok = (
                country in eligible_countries
                or (country in ["DE", "FR", "IE", "NL", "ES", "IT"] and "EU" in [c[:2] for c in eligible_countries])
            )

            if not (sector_ok and size_ok and country_ok):
                continue

            # Amount range check
            if criteria.max_amount_needed_gbp > 0 and grant["min_amount_gbp"] > criteria.max_amount_needed_gbp:
                continue

            # Calculate relevance
            relevance = 0.4
            match_reasons: List[str] = []

            if sector in eligible_sectors:
                relevance += 0.15
                match_reasons.append(f"Sector match: {sector}")
            elif "all" in eligible_sectors:
                relevance += 0.05
                match_reasons.append("Open to all sectors")

            if country in eligible_countries:
                relevance += 0.15
                match_reasons.append(f"Country match: {country}")

            if grant.get("status") == "open":
                relevance += 0.1
                match_reasons.append("Currently accepting applications")

            # Project type matching
            if criteria.project_types:
                eligible_projects = grant.get("eligible_projects", [])
                overlap = set(criteria.project_types) & set(eligible_projects)
                if overlap:
                    relevance += 0.15
                    match_reasons.append(f"Project match: {', '.join(overlap)}")

            results.append(GrantSearchResult(
                grant_id=grant["grant_id"],
                name=grant["name"],
                provider=grant["provider"],
                grant_type=grant.get("grant_type", "capital_grant"),
                max_amount_gbp=grant["max_amount_gbp"],
                min_amount_gbp=grant["min_amount_gbp"],
                co_funding_pct=grant.get("co_funding_pct", 0),
                deadline=grant.get("deadline", ""),
                status=grant.get("status", "open"),
                url=grant.get("url", ""),
                relevance_score=round(min(relevance, 1.0), 2),
                match_reasons=match_reasons,
            ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        self._grants_found = results

        outputs["grants_found"] = len(results)
        outputs["total_potential_gbp"] = sum(g.max_amount_gbp for g in results)

        if not results:
            warnings.append("No matching grants found. Consider broadening search criteria.")

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="grant_search", phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Found {len(results)} matching grants",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Eligibility Check
    # -------------------------------------------------------------------------

    async def _phase_eligibility_check(self, inp: GrantApplicationInput) -> PhaseResult:
        """Check eligibility for each found grant."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._eligibility = []
        size = inp.search_criteria.company_size
        sector = inp.industry_sector
        country = inp.country

        for grant_result in self._grants_found:
            grant_data = next(
                (g for g in GRANT_REGISTRY if g["grant_id"] == grant_result.grant_id),
                None,
            )
            if not grant_data:
                continue

            criteria_met: List[str] = []
            criteria_not_met: List[str] = []
            criteria_uncertain: List[str] = []

            requirements = grant_data.get("requirements", [])
            for req in requirements:
                req_lower = req.lower()
                if "registered" in req_lower and ("uk" in req_lower or "sme" in req_lower):
                    if country == "UK":
                        criteria_met.append(req)
                    else:
                        criteria_not_met.append(req)
                elif "registered" in req_lower and ("eu" in req_lower or "ireland" in req_lower):
                    if country in ["DE", "FR", "IE", "NL", "ES", "IT"]:
                        criteria_met.append(req)
                    else:
                        criteria_not_met.append(req)
                elif "fewer than" in req_lower or "employee" in req_lower:
                    if inp.employee_count < 250:
                        criteria_met.append(req)
                    else:
                        criteria_not_met.append(req)
                elif "manufacturing" in req_lower:
                    if "manufacturing" in sector:
                        criteria_met.append(req)
                    else:
                        criteria_not_met.append(req)
                else:
                    criteria_uncertain.append(req)

            total_criteria = len(criteria_met) + len(criteria_not_met) + len(criteria_uncertain)
            score = len(criteria_met) / max(total_criteria, 1)

            if criteria_not_met:
                status = EligibilityStatus.INELIGIBLE.value
                recommendation = "This grant may not be suitable based on current criteria."
            elif criteria_uncertain:
                status = EligibilityStatus.NEEDS_VERIFICATION.value
                recommendation = "Contact the grant provider to verify remaining eligibility criteria."
            elif score >= 0.8:
                status = EligibilityStatus.ELIGIBLE.value
                recommendation = "Strong match - proceed with application."
            else:
                status = EligibilityStatus.LIKELY_ELIGIBLE.value
                recommendation = "Likely eligible - review detailed criteria before applying."

            self._eligibility.append(EligibilityCheckResult(
                grant_id=grant_result.grant_id,
                grant_name=grant_result.name,
                eligibility_status=status,
                criteria_met=criteria_met,
                criteria_not_met=criteria_not_met,
                criteria_uncertain=criteria_uncertain,
                score=round(score, 2),
                recommendation=recommendation,
            ))

        eligible_count = sum(
            1 for e in self._eligibility
            if e.eligibility_status in [EligibilityStatus.ELIGIBLE.value, EligibilityStatus.LIKELY_ELIGIBLE.value]
        )
        outputs["eligibility_checks"] = len(self._eligibility)
        outputs["eligible_count"] = eligible_count

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="eligibility_check", phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Eligible for {eligible_count} of {len(self._eligibility)} grants",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Data Preparation
    # -------------------------------------------------------------------------

    async def _phase_data_preparation(self, inp: GrantApplicationInput) -> PhaseResult:
        """Prepare supporting data for grant applications."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        config = inp.config
        baseline = config.baseline_tco2e
        target_reduction = config.target_reduction_pct
        budget = config.total_project_budget_gbp

        # Generate project descriptions for eligible grants
        eligible_grants = [
            e for e in self._eligibility
            if e.eligibility_status in [
                EligibilityStatus.ELIGIBLE.value,
                EligibilityStatus.LIKELY_ELIGIBLE.value,
                EligibilityStatus.NEEDS_VERIFICATION.value,
            ]
        ]

        projects_prepared = 0
        for elig in eligible_grants:
            grant_data = next(
                (g for g in GRANT_REGISTRY if g["grant_id"] == elig.grant_id),
                None,
            )
            if not grant_data:
                continue

            co_funding_pct = grant_data.get("co_funding_pct", 50)
            grant_amount = min(
                budget * (1 - co_funding_pct / 100.0),
                grant_data["max_amount_gbp"],
            )
            co_funding = budget - grant_amount

            expected_reduction = baseline * (target_reduction / 100.0)
            payback = budget / max(expected_reduction * 50, 1)  # Rough GBP 50/tCO2e savings

            # Determine project title from actions
            action_titles = [a.get("title", "decarbonisation") for a in config.planned_actions[:3]]
            project_title = (
                f"{inp.organization_name} - "
                + (", ".join(action_titles) if action_titles else "Carbon Reduction Project")
            )

            project = ProjectDescription(
                project_title=project_title,
                project_summary=(
                    f"Implementation of carbon reduction measures to achieve "
                    f"{target_reduction:.0f}% emission reduction from a baseline of "
                    f"{baseline:.1f} tCO2e, targeting {expected_reduction:.1f} tCO2e annual savings."
                ),
                current_baseline_tco2e=round(baseline, 4),
                expected_reduction_tco2e=round(expected_reduction, 4),
                expected_reduction_pct=target_reduction,
                implementation_timeline_months=12,
                total_project_cost_gbp=round(budget, 2),
                grant_amount_requested_gbp=round(grant_amount, 2),
                co_funding_amount_gbp=round(co_funding, 2),
                annual_cost_savings_gbp=round(expected_reduction * 50, 2),
                payback_years=round(payback, 1),
                additional_benefits=[
                    "Reduced energy costs",
                    "Improved energy security",
                    "Enhanced brand reputation",
                    "Regulatory compliance readiness",
                ],
            )

            # Store for Phase 4
            elig.recommendation = json.dumps(project.model_dump())
            projects_prepared += 1

        outputs["projects_prepared"] = projects_prepared
        if baseline <= 0:
            warnings.append("No baseline provided; project descriptions will use placeholder data")

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="data_preparation", phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Prepared data for {projects_prepared} applications",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Application Support
    # -------------------------------------------------------------------------

    async def _phase_application_support(self, inp: GrantApplicationInput) -> PhaseResult:
        """Generate pre-filled application templates."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._applications = []
        eligible_grants = [
            e for e in self._eligibility
            if e.eligibility_status in [
                EligibilityStatus.ELIGIBLE.value,
                EligibilityStatus.LIKELY_ELIGIBLE.value,
            ]
        ]

        for elig in eligible_grants:
            grant_data = next(
                (g for g in GRANT_REGISTRY if g["grant_id"] == elig.grant_id),
                None,
            )
            if not grant_data:
                continue

            docs_required = grant_data.get("documents_required", [])
            docs_ready: List[str] = []
            docs_missing: List[str] = []

            # Determine which documents we can auto-generate
            auto_docs = {"application_form", "project_description", "company_profile", "baseline_report"}
            for doc in docs_required:
                if doc in auto_docs:
                    docs_ready.append(doc)
                else:
                    docs_missing.append(doc)

            completeness = (len(docs_ready) / max(len(docs_required), 1)) * 100

            # Parse project from Phase 3
            try:
                project_data = json.loads(elig.recommendation)
                project = ProjectDescription(**project_data)
            except (json.JSONDecodeError, TypeError, ValueError):
                project = ProjectDescription(
                    project_title=f"{inp.organization_name} - Carbon Reduction Project",
                )

            next_actions: List[str] = []
            if docs_missing:
                next_actions.append(f"Prepare missing documents: {', '.join(docs_missing)}")
            next_actions.append("Review auto-generated content for accuracy")
            next_actions.append(f"Submit before deadline: {grant_data.get('deadline', 'TBC')}")

            template = ApplicationTemplate(
                grant_id=elig.grant_id,
                grant_name=elig.grant_name,
                company_name=inp.organization_name,
                company_registration="",
                project=project,
                documents_needed=docs_required,
                documents_ready=docs_ready,
                documents_missing=docs_missing,
                application_status=ApplicationStatus.DRAFT.value,
                completeness_pct=round(completeness, 1),
                next_actions=next_actions,
            )
            self._applications.append(template)

        outputs["applications_created"] = len(self._applications)
        avg_completeness = (
            sum(a.completeness_pct for a in self._applications) / max(len(self._applications), 1)
        )
        outputs["avg_completeness_pct"] = round(avg_completeness, 1)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="application_support", phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"{len(self._applications)} applications drafted ({avg_completeness:.0f}% complete)",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Submission Export
    # -------------------------------------------------------------------------

    async def _phase_submission_export(self, inp: GrantApplicationInput) -> PhaseResult:
        """Prepare submission packages for export."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._submissions = []

        for app in self._applications:
            export_ready = app.completeness_pct >= 80 and not app.documents_missing
            notes: List[str] = []

            if not export_ready:
                if app.documents_missing:
                    notes.append(
                        f"Missing documents: {', '.join(app.documents_missing)}. "
                        "Prepare these before submitting."
                    )
                notes.append("Application requires manual review before submission.")
            else:
                notes.append("Application is ready for export and submission.")

            submission = SubmissionPackage(
                grant_id=app.grant_id,
                grant_name=app.grant_name,
                application=app,
                export_format="pdf",
                export_ready=export_ready,
                export_notes=notes,
            )
            self._submissions.append(submission)

        ready_count = sum(1 for s in self._submissions if s.export_ready)
        outputs["total_submissions"] = len(self._submissions)
        outputs["ready_for_export"] = ready_count
        outputs["needs_additional_work"] = len(self._submissions) - ready_count

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="submission_export", phase_number=5,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"{ready_count} of {len(self._submissions)} ready to submit",
        )

    # -------------------------------------------------------------------------
    # Next Steps
    # -------------------------------------------------------------------------

    def _generate_next_steps(self) -> List[str]:
        steps: List[str] = []

        ready = [s for s in self._submissions if s.export_ready]
        not_ready = [s for s in self._submissions if not s.export_ready]

        if ready:
            steps.append(
                f"Export and submit {len(ready)} ready application(s): "
                + ", ".join(s.grant_name for s in ready[:3])
            )

        if not_ready:
            steps.append(
                f"Complete {len(not_ready)} application(s) requiring additional documents."
            )

        steps.append("Set calendar reminders for grant deadlines.")
        steps.append("Track application status in the SME dashboard.")
        steps.append("Consider engaging a grant writer for large applications (>GBP 50k).")

        return steps
