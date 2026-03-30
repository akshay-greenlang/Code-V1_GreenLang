# -*- coding: utf-8 -*-
"""
DueDiligencePolicyEngine - PACK-019 CSDDD Due Diligence Policy Engine
======================================================================

Assesses corporate due diligence policy completeness and compliance
against the EU Corporate Sustainability Due Diligence Directive
(CSDDD / CS3D), focusing on Article 5 requirements for adopting and
implementing a due diligence policy.

The CSDDD (Directive 2024/1760) requires in-scope companies to embed
responsible business conduct into their policies, identify and address
adverse human rights and environmental impacts throughout their value
chains, and establish adequate governance structures.

CSDDD Scope Determination (Art 2):
    - Phase 1 (26 July 2027): >5000 employees AND >EUR 1500M worldwide
      net turnover (or EU net turnover for third-country companies)
    - Phase 2 (26 July 2028): >3000 employees AND >EUR 900M worldwide
      net turnover
    - Phase 3 (26 July 2029): >1000 employees AND >EUR 450M worldwide
      net turnover

CSDDD Art 5 - Due Diligence Policy:
    - Para 1: Companies shall integrate due diligence into all their
      relevant policies and risk management systems.
    - Para 2: The due diligence policy shall contain:
      (a) a description of the company's approach to due diligence
      (b) a code of conduct describing the rules and principles to be
          followed by employees and subsidiaries
      (c) a description of the processes put in place to implement
          due diligence, including measures to verify compliance
          with the code of conduct and to extend its application to
          business partners
    - Para 3: Companies shall update their policy annually and
      whenever a significant change occurs.

Additional Article Assessments (Art 6-29):
    - Art 6: Identifying adverse impacts
    - Art 7: Prioritising adverse impacts
    - Art 8: Preventing potential adverse impacts
    - Art 9: Bringing actual adverse impacts to an end
    - Art 10: Remediation
    - Art 11: Meaningful stakeholder engagement
    - Art 12: Notification mechanism / complaints procedure
    - Art 13: Monitoring
    - Art 14: Reporting / communication
    - Art 15: Climate transition plan (Art 22 of the Directive)
    - Art 22: Climate transition plan alignment with Paris Agreement
    - Art 29: Civil liability

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - UN Guiding Principles on Business and Human Rights (UNGPs)
    - OECD Guidelines for Multinational Enterprises (2023 update)
    - OECD Due Diligence Guidance for Responsible Business Conduct
    - ILO Declaration on Fundamental Principles and Rights at Work
    - International Bill of Human Rights

Zero-Hallucination:
    - Scope determination uses fixed numeric thresholds
    - Policy assessment uses boolean and count-based scoring
    - Article compliance scores are deterministic weighted averages
    - Overall scoring uses Decimal arithmetic throughout
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-019 CSDDD Readiness
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )

def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompanyScope(str, Enum):
    """Phased applicability scope under CSDDD Art 2.

    The CSDDD applies in three phases based on employee count and
    worldwide net turnover thresholds.  Companies may also choose
    to apply the directive voluntarily before they fall in scope.

    - PHASE_1: >5000 employees AND >EUR 1500M turnover (from 26 July 2027)
    - PHASE_2: >3000 employees AND >EUR 900M turnover (from 26 July 2028)
    - PHASE_3: >1000 employees AND >EUR 450M turnover (from 26 July 2029)
    - NOT_IN_SCOPE: Does not meet any threshold
    - VOLUNTARY: Company voluntarily adopts CSDDD framework
    """
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    NOT_IN_SCOPE = "not_in_scope"
    VOLUNTARY = "voluntary"

class ComplianceStatus(str, Enum):
    """Compliance status of a company against a CSDDD requirement.

    Used to classify the outcome of each article-level assessment.
    """
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

class ArticleReference(str, Enum):
    """CSDDD article references for assessment tracking.

    Each value corresponds to a substantive obligation article
    in Directive (EU) 2024/1760 that requires company action.
    """
    ART_5 = "art_5"
    ART_6 = "art_6"
    ART_7 = "art_7"
    ART_8 = "art_8"
    ART_9 = "art_9"
    ART_10 = "art_10"
    ART_11 = "art_11"
    ART_12 = "art_12"
    ART_13 = "art_13"
    ART_14 = "art_14"
    ART_15 = "art_15"
    ART_22 = "art_22"
    ART_29 = "art_29"

class PolicyArea(str, Enum):
    """Policy areas that must be addressed under CSDDD Art 5.

    The due diligence policy must integrate responsible business
    conduct across these key operational areas.
    """
    CODE_OF_CONDUCT = "code_of_conduct"
    RISK_MANAGEMENT = "risk_management"
    GOVERNANCE = "governance"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CSDDD scope thresholds by phase (employee_count, turnover_eur)
SCOPE_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    CompanyScope.PHASE_1.value: {
        "min_employees": Decimal("5000"),
        "min_turnover_eur": Decimal("1500000000"),
        "effective_date": Decimal("20270726"),
    },
    CompanyScope.PHASE_2.value: {
        "min_employees": Decimal("3000"),
        "min_turnover_eur": Decimal("900000000"),
        "effective_date": Decimal("20280726"),
    },
    CompanyScope.PHASE_3.value: {
        "min_employees": Decimal("1000"),
        "min_turnover_eur": Decimal("450000000"),
        "effective_date": Decimal("20290726"),
    },
}

# Article weights for overall score calculation (out of 100)
ARTICLE_WEIGHTS: Dict[str, Decimal] = {
    ArticleReference.ART_5.value: Decimal("15"),    # DD policy
    ArticleReference.ART_6.value: Decimal("10"),    # Identifying impacts
    ArticleReference.ART_7.value: Decimal("8"),     # Prioritising impacts
    ArticleReference.ART_8.value: Decimal("10"),    # Prevention
    ArticleReference.ART_9.value: Decimal("10"),    # Bringing to an end
    ArticleReference.ART_10.value: Decimal("8"),    # Remediation
    ArticleReference.ART_11.value: Decimal("7"),    # Stakeholder engagement
    ArticleReference.ART_12.value: Decimal("6"),    # Complaints procedure
    ArticleReference.ART_13.value: Decimal("7"),    # Monitoring
    ArticleReference.ART_14.value: Decimal("5"),    # Reporting
    ArticleReference.ART_15.value: Decimal("6"),    # Climate transition plan
    ArticleReference.ART_22.value: Decimal("5"),    # Paris alignment
    ArticleReference.ART_29.value: Decimal("3"),    # Civil liability
}

# Minimum criteria for each policy area
POLICY_AREA_CRITERIA: Dict[str, List[str]] = {
    PolicyArea.CODE_OF_CONDUCT.value: [
        "has_code_of_conduct",
        "code_covers_human_rights",
        "code_covers_environment",
        "code_covers_subsidiaries",
        "code_covers_business_partners",
        "code_reviewed_annually",
    ],
    PolicyArea.RISK_MANAGEMENT.value: [
        "has_risk_management_framework",
        "risk_framework_covers_hr",
        "risk_framework_covers_env",
        "risk_assessment_frequency_adequate",
        "risk_mitigation_measures_defined",
    ],
    PolicyArea.GOVERNANCE.value: [
        "board_oversight_defined",
        "dedicated_dd_officer",
        "governance_body_receives_reports",
        "accountability_mechanisms_exist",
    ],
    PolicyArea.MONITORING.value: [
        "monitoring_process_defined",
        "periodic_assessments_conducted",
        "kpis_defined",
        "third_party_verification",
    ],
    PolicyArea.REPORTING.value: [
        "annual_reporting_committed",
        "reporting_covers_impacts",
        "reporting_covers_measures",
        "reporting_publicly_available",
    ],
    PolicyArea.STAKEHOLDER_ENGAGEMENT.value: [
        "stakeholder_mapping_conducted",
        "engagement_process_defined",
        "affected_communities_consulted",
        "engagement_outcomes_documented",
    ],
}

# Art 5 specific sub-criteria
ART_5_CRITERIA: List[str] = [
    "has_dd_policy",
    "dd_policy_describes_approach",
    "dd_policy_has_code_of_conduct",
    "code_rules_for_employees",
    "code_rules_for_subsidiaries",
    "dd_policy_describes_processes",
    "dd_policy_verification_measures",
    "dd_policy_extended_to_partners",
    "dd_policy_updated_annually",
    "dd_policy_approved_by_board",
]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """Company profile for CSDDD scope determination and assessment.

    Contains all company-level attributes needed to determine
    whether the company falls within CSDDD scope (Phase 1/2/3)
    and to assess the completeness of its due diligence policy.

    Attributes:
        company_name: Legal name of the company.
        country: ISO 3166-1 alpha-2 country code of registration.
        sector: Industry sector or NACE code.
        employee_count: Average number of employees in reporting year.
        worldwide_turnover_eur: Worldwide net turnover in EUR.
        eu_turnover_eur: EU-generated net turnover in EUR.
        reporting_year: Fiscal year being assessed.
        value_chain_tiers: Number of value chain tiers monitored.
        has_dd_policy: Whether the company has a DD policy.
        has_code_of_conduct: Whether a code of conduct exists.
        has_grievance_mechanism: Whether a complaints procedure exists.
        has_climate_transition_plan: Whether a CTP exists (Art 22).
    """
    company_name: str = Field(
        ...,
        description="Legal name of the company",
        max_length=500,
    )
    country: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code of registration",
        max_length=3,
    )
    sector: str = Field(
        default="",
        description="Industry sector or NACE code",
        max_length=200,
    )
    employee_count: int = Field(
        ...,
        description="Average number of employees in the reporting year",
        ge=0,
    )
    worldwide_turnover_eur: Decimal = Field(
        ...,
        description="Worldwide net turnover in EUR",
        ge=Decimal("0"),
    )
    eu_turnover_eur: Decimal = Field(
        default=Decimal("0"),
        description="EU-generated net turnover in EUR",
        ge=Decimal("0"),
    )
    reporting_year: int = Field(
        ...,
        description="Fiscal year being assessed",
        ge=2024,
    )
    value_chain_tiers: int = Field(
        default=1,
        description="Number of value chain tiers actively monitored",
        ge=0,
        le=10,
    )
    is_eu_company: bool = Field(
        default=True,
        description="Whether the company is formed under EU member state law",
    )
    is_third_country_company: bool = Field(
        default=False,
        description="Whether the company is formed under non-EU law",
    )
    has_dd_policy: bool = Field(
        default=False,
        description="Whether the company has adopted a DD policy",
    )
    dd_policy_describes_approach: bool = Field(
        default=False,
        description="Whether the DD policy describes the company's approach",
    )
    dd_policy_has_code_of_conduct: bool = Field(
        default=False,
        description="Whether the DD policy includes a code of conduct",
    )
    code_rules_for_employees: bool = Field(
        default=False,
        description="Whether the code of conduct has rules for employees",
    )
    code_rules_for_subsidiaries: bool = Field(
        default=False,
        description="Whether the code of conduct covers subsidiaries",
    )
    dd_policy_describes_processes: bool = Field(
        default=False,
        description="Whether the DD policy describes implementation processes",
    )
    dd_policy_verification_measures: bool = Field(
        default=False,
        description="Whether verification measures are described",
    )
    dd_policy_extended_to_partners: bool = Field(
        default=False,
        description="Whether the DD policy extends to business partners",
    )
    dd_policy_updated_annually: bool = Field(
        default=False,
        description="Whether the DD policy is updated at least annually",
    )
    dd_policy_approved_by_board: bool = Field(
        default=False,
        description="Whether the DD policy is approved by a governing body",
    )
    has_code_of_conduct: bool = Field(
        default=False,
        description="Whether the company has a code of conduct",
    )
    code_covers_human_rights: bool = Field(
        default=False,
        description="Whether the code of conduct covers human rights",
    )
    code_covers_environment: bool = Field(
        default=False,
        description="Whether the code of conduct covers environmental issues",
    )
    code_covers_subsidiaries: bool = Field(
        default=False,
        description="Whether the code of conduct applies to subsidiaries",
    )
    code_covers_business_partners: bool = Field(
        default=False,
        description="Whether the code of conduct is extended to business partners",
    )
    code_reviewed_annually: bool = Field(
        default=False,
        description="Whether the code of conduct is reviewed annually",
    )
    has_risk_management_framework: bool = Field(
        default=False,
        description="Whether a risk management framework exists",
    )
    risk_framework_covers_hr: bool = Field(
        default=False,
        description="Whether the risk framework covers human rights",
    )
    risk_framework_covers_env: bool = Field(
        default=False,
        description="Whether the risk framework covers environment",
    )
    risk_assessment_frequency_adequate: bool = Field(
        default=False,
        description="Whether risk assessments are conducted at adequate frequency",
    )
    risk_mitigation_measures_defined: bool = Field(
        default=False,
        description="Whether risk mitigation measures are defined",
    )
    board_oversight_defined: bool = Field(
        default=False,
        description="Whether board-level DD oversight is defined",
    )
    dedicated_dd_officer: bool = Field(
        default=False,
        description="Whether a dedicated DD officer or function exists",
    )
    governance_body_receives_reports: bool = Field(
        default=False,
        description="Whether the governance body receives DD reports",
    )
    accountability_mechanisms_exist: bool = Field(
        default=False,
        description="Whether accountability mechanisms are in place",
    )
    monitoring_process_defined: bool = Field(
        default=False,
        description="Whether a monitoring process for DD is defined",
    )
    periodic_assessments_conducted: bool = Field(
        default=False,
        description="Whether periodic DD assessments are conducted",
    )
    kpis_defined: bool = Field(
        default=False,
        description="Whether KPIs for DD effectiveness are defined",
    )
    third_party_verification: bool = Field(
        default=False,
        description="Whether third-party verification is used",
    )
    annual_reporting_committed: bool = Field(
        default=False,
        description="Whether the company commits to annual DD reporting",
    )
    reporting_covers_impacts: bool = Field(
        default=False,
        description="Whether reporting covers identified impacts",
    )
    reporting_covers_measures: bool = Field(
        default=False,
        description="Whether reporting covers measures taken",
    )
    reporting_publicly_available: bool = Field(
        default=False,
        description="Whether the DD report is publicly available",
    )
    stakeholder_mapping_conducted: bool = Field(
        default=False,
        description="Whether stakeholder mapping has been conducted",
    )
    engagement_process_defined: bool = Field(
        default=False,
        description="Whether a stakeholder engagement process is defined",
    )
    affected_communities_consulted: bool = Field(
        default=False,
        description="Whether affected communities have been consulted",
    )
    engagement_outcomes_documented: bool = Field(
        default=False,
        description="Whether engagement outcomes are documented",
    )
    has_grievance_mechanism: bool = Field(
        default=False,
        description="Whether a complaints procedure / grievance mechanism exists",
    )
    grievance_mechanism_accessible: bool = Field(
        default=False,
        description="Whether the grievance mechanism is accessible to stakeholders",
    )
    grievance_mechanism_documented: bool = Field(
        default=False,
        description="Whether the grievance mechanism procedures are documented",
    )
    has_climate_transition_plan: bool = Field(
        default=False,
        description="Whether a climate transition plan exists (Art 22)",
    )
    climate_plan_paris_aligned: bool = Field(
        default=False,
        description="Whether the climate plan is Paris Agreement aligned",
    )
    climate_plan_has_targets: bool = Field(
        default=False,
        description="Whether the climate plan includes time-bound targets",
    )
    climate_plan_has_actions: bool = Field(
        default=False,
        description="Whether the climate plan includes implementation actions",
    )
    climate_plan_has_investments: bool = Field(
        default=False,
        description="Whether the climate plan describes investment needs",
    )
    has_impact_identification_process: bool = Field(
        default=False,
        description="Whether a process for identifying adverse impacts exists",
    )
    impact_identification_covers_own_ops: bool = Field(
        default=False,
        description="Whether impact identification covers own operations",
    )
    impact_identification_covers_subsidiaries: bool = Field(
        default=False,
        description="Whether impact identification covers subsidiaries",
    )
    impact_identification_covers_value_chain: bool = Field(
        default=False,
        description="Whether impact identification covers the value chain",
    )
    has_prioritisation_methodology: bool = Field(
        default=False,
        description="Whether a methodology for prioritising impacts exists",
    )
    prioritisation_considers_severity: bool = Field(
        default=False,
        description="Whether prioritisation considers severity",
    )
    prioritisation_considers_likelihood: bool = Field(
        default=False,
        description="Whether prioritisation considers likelihood",
    )
    has_prevention_measures: bool = Field(
        default=False,
        description="Whether prevention measures for potential impacts exist",
    )
    prevention_includes_cap: bool = Field(
        default=False,
        description="Whether a corrective action plan is part of prevention",
    )
    prevention_includes_contractual: bool = Field(
        default=False,
        description="Whether contractual assurances are used for prevention",
    )
    has_cessation_measures: bool = Field(
        default=False,
        description="Whether measures exist to bring actual impacts to an end",
    )
    cessation_includes_corrective_action: bool = Field(
        default=False,
        description="Whether corrective action is part of cessation approach",
    )
    has_remediation_process: bool = Field(
        default=False,
        description="Whether a remediation process exists (Art 10)",
    )
    remediation_includes_financial: bool = Field(
        default=False,
        description="Whether financial remediation provisions exist",
    )
    has_civil_liability_awareness: bool = Field(
        default=False,
        description="Whether the company is aware of civil liability provisions",
    )
    civil_liability_insurance: bool = Field(
        default=False,
        description="Whether liability insurance covers DD obligations",
    )

    @field_validator("country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alphabetic."""
        if v and not v.isalpha():
            raise ValueError("Country code must be alphabetic")
        return v.upper()

class PolicyAssessment(BaseModel):
    """Assessment result for a single policy area under CSDDD Art 5.

    Captures the compliance status, numeric score, identified gaps,
    and actionable recommendations for one of the six policy areas.
    """
    policy_area: PolicyArea = Field(
        ...,
        description="Policy area being assessed",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Compliance status for this policy area",
    )
    score: Decimal = Field(
        ...,
        description="Score from 0 to 100 for this policy area",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    criteria_met: int = Field(
        default=0,
        description="Number of criteria met in this policy area",
        ge=0,
    )
    criteria_total: int = Field(
        default=0,
        description="Total number of criteria for this policy area",
        ge=0,
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps in this policy area",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to close gaps",
    )

class ArticleAssessment(BaseModel):
    """Assessment result for a single CSDDD article.

    Captures the compliance determination for one article of
    the Directive, including the article reference, status,
    score, and supporting details.
    """
    article: ArticleReference = Field(
        ...,
        description="CSDDD article reference",
    )
    article_title: str = Field(
        default="",
        description="Human-readable article title",
        max_length=500,
    )
    status: ComplianceStatus = Field(
        ...,
        description="Compliance status for this article",
    )
    score: Decimal = Field(
        ...,
        description="Score from 0 to 100 for this article",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    weight: Decimal = Field(
        default=Decimal("0"),
        description="Weight of this article in overall score",
    )
    criteria_met: int = Field(
        default=0,
        description="Number of criteria satisfied",
        ge=0,
    )
    criteria_total: int = Field(
        default=0,
        description="Total criteria for this article",
        ge=0,
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps for this article",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for this article",
    )

class ScopeAssessment(BaseModel):
    """Scope determination result under CSDDD Art 2.

    Determines which phase (if any) the company falls into
    based on its employee count and worldwide turnover.
    """
    scope_phase: CompanyScope = Field(
        ...,
        description="Determined scope phase",
    )
    in_scope: bool = Field(
        ...,
        description="Whether the company is in scope for CSDDD",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of scope determination",
        max_length=2000,
    )
    employee_count: int = Field(
        default=0,
        description="Employee count used for determination",
        ge=0,
    )
    turnover_eur: Decimal = Field(
        default=Decimal("0"),
        description="Turnover figure used for determination",
    )
    effective_date: str = Field(
        default="",
        description="Date from which obligations apply",
    )
    meets_employee_threshold: bool = Field(
        default=False,
        description="Whether employee threshold is met",
    )
    meets_turnover_threshold: bool = Field(
        default=False,
        description="Whether turnover threshold is met",
    )

class DueDiligencePolicyResult(BaseModel):
    """Complete due diligence policy assessment result.

    Aggregates scope determination, article-level assessments,
    policy area assessments, and overall scoring into a single
    auditable result with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    company: str = Field(
        default="",
        description="Company name assessed",
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year assessed",
    )
    scope_assessment: ScopeAssessment = Field(
        ...,
        description="Scope determination result",
    )
    article_assessments: List[ArticleAssessment] = Field(
        default_factory=list,
        description="Per-article compliance assessments",
    )
    policy_assessments: List[PolicyAssessment] = Field(
        default_factory=list,
        description="Per-policy-area assessments",
    )
    overall_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall weighted compliance score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    overall_status: ComplianceStatus = Field(
        default=ComplianceStatus.NON_COMPLIANT,
        description="Overall compliance status",
    )
    total_criteria_met: int = Field(
        default=0,
        description="Total criteria met across all articles",
        ge=0,
    )
    total_criteria: int = Field(
        default=0,
        description="Total criteria across all articles",
        ge=0,
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritised recommendations",
    )
    gaps_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary of gaps by article",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow,
        description="Assessment timestamp (UTC)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail provenance",
    )

# ---------------------------------------------------------------------------
# Article metadata
# ---------------------------------------------------------------------------

ARTICLE_TITLES: Dict[str, str] = {
    ArticleReference.ART_5.value: "Due diligence policy (Art 5)",
    ArticleReference.ART_6.value: "Identifying adverse impacts (Art 6)",
    ArticleReference.ART_7.value: "Prioritising adverse impacts (Art 7)",
    ArticleReference.ART_8.value: "Preventing potential adverse impacts (Art 8)",
    ArticleReference.ART_9.value: "Bringing actual adverse impacts to an end (Art 9)",
    ArticleReference.ART_10.value: "Remediation (Art 10)",
    ArticleReference.ART_11.value: "Meaningful stakeholder engagement (Art 11)",
    ArticleReference.ART_12.value: "Notification mechanism and complaints procedure (Art 12)",
    ArticleReference.ART_13.value: "Monitoring (Art 13)",
    ArticleReference.ART_14.value: "Reporting and communication (Art 14)",
    ArticleReference.ART_15.value: "Climate transition plan (Art 15/22)",
    ArticleReference.ART_22.value: "Paris Agreement alignment (Art 22)",
    ArticleReference.ART_29.value: "Civil liability (Art 29)",
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DueDiligencePolicyEngine:
    """CSDDD Due Diligence Policy assessment engine.

    Provides deterministic, zero-hallucination assessments of corporate
    due diligence policies against the requirements of EU Directive
    2024/1760 (CSDDD / CS3D).

    The engine evaluates:
    1. **Scope determination** - whether the company falls under Phase 1,
       Phase 2, Phase 3, or is out of scope based on employee count and
       worldwide turnover thresholds.
    2. **Article-level compliance** - per-article assessment of compliance
       with each substantive obligation (Art 5 through Art 29).
    3. **Policy area completeness** - assessment of six key policy areas
       (code of conduct, risk management, governance, monitoring,
       reporting, stakeholder engagement).
    4. **Overall scoring** - weighted aggregate score using Decimal
       arithmetic for reproducibility.

    All calculations use deterministic formulas with no LLM involvement.
    Every result includes a SHA-256 provenance hash for audit trails.

    Usage::

        engine = DueDiligencePolicyEngine()
        profile = CompanyProfile(
            company_name="Acme Corp",
            country="DE",
            sector="manufacturing",
            employee_count=6000,
            worldwide_turnover_eur=Decimal("2000000000"),
            reporting_year=2027,
            has_dd_policy=True,
            has_code_of_conduct=True,
        )
        result = engine.assess_policy(profile)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Scope Determination                                                  #
    # ------------------------------------------------------------------ #

    def assess_scope(self, profile: CompanyProfile) -> ScopeAssessment:
        """Determine CSDDD scope phase for a company.

        Evaluates the company's employee count and worldwide turnover
        against the three-phase thresholds defined in CSDDD Art 2.
        For third-country companies, EU turnover is used instead of
        worldwide turnover.

        Args:
            profile: CompanyProfile with employee and turnover data.

        Returns:
            ScopeAssessment with phase determination and reasoning.
        """
        logger.info(
            "Assessing CSDDD scope for %s (employees=%d, turnover=%.2f EUR)",
            profile.company_name, profile.employee_count,
            float(profile.worldwide_turnover_eur),
        )

        emp = _decimal(profile.employee_count)
        turnover = (
            profile.eu_turnover_eur
            if profile.is_third_country_company
            else profile.worldwide_turnover_eur
        )
        turnover_label = "EU" if profile.is_third_country_company else "worldwide"

        # Check phases in order (most restrictive first)
        for phase_key in [
            CompanyScope.PHASE_1.value,
            CompanyScope.PHASE_2.value,
            CompanyScope.PHASE_3.value,
        ]:
            thresholds = SCOPE_THRESHOLDS[phase_key]
            min_emp = thresholds["min_employees"]
            min_turn = thresholds["min_turnover_eur"]
            eff_date_raw = str(int(thresholds["effective_date"]))
            eff_date = f"{eff_date_raw[:4]}-{eff_date_raw[4:6]}-{eff_date_raw[6:]}"

            meets_emp = emp > min_emp
            meets_turn = turnover > min_turn

            if meets_emp and meets_turn:
                phase = CompanyScope(phase_key)
                reasoning = (
                    f"{profile.company_name} meets {phase.value} thresholds: "
                    f"employees {profile.employee_count} > {int(min_emp)} "
                    f"AND {turnover_label} turnover EUR {float(turnover):,.2f} "
                    f"> EUR {float(min_turn):,.2f}. "
                    f"Obligations apply from {eff_date}."
                )
                logger.info("Scope: %s - %s", phase.value, reasoning)
                return ScopeAssessment(
                    scope_phase=phase,
                    in_scope=True,
                    reasoning=reasoning,
                    employee_count=profile.employee_count,
                    turnover_eur=turnover,
                    effective_date=eff_date,
                    meets_employee_threshold=True,
                    meets_turnover_threshold=True,
                )

        # Not in scope
        reasoning = (
            f"{profile.company_name} does not meet any CSDDD threshold: "
            f"employees {profile.employee_count}, "
            f"{turnover_label} turnover EUR {float(turnover):,.2f}. "
            f"Lowest threshold is Phase 3: >1000 employees AND >EUR 450M."
        )
        logger.info("Scope: NOT_IN_SCOPE - %s", reasoning)
        return ScopeAssessment(
            scope_phase=CompanyScope.NOT_IN_SCOPE,
            in_scope=False,
            reasoning=reasoning,
            employee_count=profile.employee_count,
            turnover_eur=turnover,
            effective_date="",
            meets_employee_threshold=False,
            meets_turnover_threshold=False,
        )

    # ------------------------------------------------------------------ #
    # Article-Level Assessment                                             #
    # ------------------------------------------------------------------ #

    def _assess_art_5(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 5 - Due diligence policy.

        Evaluates whether the company's DD policy meets the ten
        sub-criteria derived from Art 5 Para 1-3.

        Args:
            profile: CompanyProfile with DD policy attributes.

        Returns:
            ArticleAssessment for Art 5.
        """
        criteria_map = {
            "has_dd_policy": profile.has_dd_policy,
            "dd_policy_describes_approach": profile.dd_policy_describes_approach,
            "dd_policy_has_code_of_conduct": profile.dd_policy_has_code_of_conduct,
            "code_rules_for_employees": profile.code_rules_for_employees,
            "code_rules_for_subsidiaries": profile.code_rules_for_subsidiaries,
            "dd_policy_describes_processes": profile.dd_policy_describes_processes,
            "dd_policy_verification_measures": profile.dd_policy_verification_measures,
            "dd_policy_extended_to_partners": profile.dd_policy_extended_to_partners,
            "dd_policy_updated_annually": profile.dd_policy_updated_annually,
            "dd_policy_approved_by_board": profile.dd_policy_approved_by_board,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = self._generate_art_5_recommendations(gaps)

        status = self._score_to_status(score)

        return ArticleAssessment(
            article=ArticleReference.ART_5,
            article_title=ARTICLE_TITLES[ArticleReference.ART_5.value],
            status=status,
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_5.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_6(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 6 - Identifying adverse impacts.

        Evaluates whether the company has processes to identify actual
        and potential adverse impacts on human rights and environment
        in its own operations, subsidiaries, and value chain.

        Args:
            profile: CompanyProfile with impact identification attributes.

        Returns:
            ArticleAssessment for Art 6.
        """
        criteria_map = {
            "has_impact_identification_process": profile.has_impact_identification_process,
            "covers_own_operations": profile.impact_identification_covers_own_ops,
            "covers_subsidiaries": profile.impact_identification_covers_subsidiaries,
            "covers_value_chain": profile.impact_identification_covers_value_chain,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_impact_identification_process" in gaps:
            recommendations.append(
                "Establish a formal process for identifying actual and potential "
                "adverse impacts across own operations and value chain (Art 6)."
            )
        if "covers_own_operations" in gaps:
            recommendations.append(
                "Extend impact identification to cover own operations."
            )
        if "covers_subsidiaries" in gaps:
            recommendations.append(
                "Ensure impact identification covers all subsidiaries."
            )
        if "covers_value_chain" in gaps:
            recommendations.append(
                "Map and assess adverse impacts across the value chain, "
                "including upstream and downstream business partners."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_6,
            article_title=ARTICLE_TITLES[ArticleReference.ART_6.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_6.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_7(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 7 - Prioritising adverse impacts.

        Evaluates whether the company has a methodology to prioritise
        identified adverse impacts based on severity and likelihood.

        Args:
            profile: CompanyProfile with prioritisation attributes.

        Returns:
            ArticleAssessment for Art 7.
        """
        criteria_map = {
            "has_prioritisation_methodology": profile.has_prioritisation_methodology,
            "considers_severity": profile.prioritisation_considers_severity,
            "considers_likelihood": profile.prioritisation_considers_likelihood,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_prioritisation_methodology" in gaps:
            recommendations.append(
                "Develop a methodology to prioritise adverse impacts where "
                "it is not possible to address all simultaneously (Art 7)."
            )
        if "considers_severity" in gaps:
            recommendations.append(
                "Incorporate severity (scale, scope, irremediable character) "
                "into the prioritisation methodology."
            )
        if "considers_likelihood" in gaps:
            recommendations.append(
                "Include likelihood of occurrence in the impact "
                "prioritisation framework."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_7,
            article_title=ARTICLE_TITLES[ArticleReference.ART_7.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_7.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_8(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 8 - Preventing potential adverse impacts.

        Evaluates whether the company has appropriate measures to prevent
        potential adverse impacts, including corrective action plans and
        contractual assurances with business partners.

        Args:
            profile: CompanyProfile with prevention attributes.

        Returns:
            ArticleAssessment for Art 8.
        """
        criteria_map = {
            "has_prevention_measures": profile.has_prevention_measures,
            "includes_corrective_action_plan": profile.prevention_includes_cap,
            "includes_contractual_assurances": profile.prevention_includes_contractual,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_prevention_measures" in gaps:
            recommendations.append(
                "Develop and implement appropriate measures to prevent "
                "potential adverse human rights and environmental impacts (Art 8)."
            )
        if "includes_corrective_action_plan" in gaps:
            recommendations.append(
                "Create a corrective action plan with timelines and "
                "qualitative and quantitative indicators for measuring progress."
            )
        if "includes_contractual_assurances" in gaps:
            recommendations.append(
                "Seek contractual assurances from direct business partners "
                "to ensure compliance with the company's code of conduct."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_8,
            article_title=ARTICLE_TITLES[ArticleReference.ART_8.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_8.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_9(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 9 - Bringing actual adverse impacts to an end.

        Evaluates whether the company has measures to cease, minimise,
        or bring to an end actual adverse impacts that have occurred.

        Args:
            profile: CompanyProfile with cessation attributes.

        Returns:
            ArticleAssessment for Art 9.
        """
        criteria_map = {
            "has_cessation_measures": profile.has_cessation_measures,
            "includes_corrective_action": profile.cessation_includes_corrective_action,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_cessation_measures" in gaps:
            recommendations.append(
                "Establish measures to bring actual adverse impacts to "
                "an end and minimise their extent (Art 9)."
            )
        if "includes_corrective_action" in gaps:
            recommendations.append(
                "Develop corrective action procedures including "
                "neutralisation of impacts through financial or "
                "non-financial remediation."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_9,
            article_title=ARTICLE_TITLES[ArticleReference.ART_9.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_9.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_10(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 10 - Remediation.

        Evaluates whether the company has adequate remediation processes
        including financial provision and engagement with affected persons.

        Args:
            profile: CompanyProfile with remediation attributes.

        Returns:
            ArticleAssessment for Art 10.
        """
        criteria_map = {
            "has_remediation_process": profile.has_remediation_process,
            "includes_financial_provision": profile.remediation_includes_financial,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_remediation_process" in gaps:
            recommendations.append(
                "Establish a remediation process to provide or cooperate "
                "in the provision of remediation where the company has "
                "caused or contributed to an actual adverse impact (Art 10)."
            )
        if "includes_financial_provision" in gaps:
            recommendations.append(
                "Provide for financial remediation including compensation, "
                "restitution, or rehabilitation as appropriate."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_10,
            article_title=ARTICLE_TITLES[ArticleReference.ART_10.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_10.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_11(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 11 - Meaningful stakeholder engagement.

        Evaluates whether the company conducts meaningful engagement
        with stakeholders including affected communities, workers,
        and their representatives.

        Args:
            profile: CompanyProfile with stakeholder engagement attributes.

        Returns:
            ArticleAssessment for Art 11.
        """
        criteria_map = {
            "stakeholder_mapping_conducted": profile.stakeholder_mapping_conducted,
            "engagement_process_defined": profile.engagement_process_defined,
            "affected_communities_consulted": profile.affected_communities_consulted,
            "engagement_outcomes_documented": profile.engagement_outcomes_documented,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "stakeholder_mapping_conducted" in gaps:
            recommendations.append(
                "Conduct stakeholder mapping to identify relevant "
                "affected stakeholders and their representatives (Art 11)."
            )
        if "engagement_process_defined" in gaps:
            recommendations.append(
                "Define a meaningful engagement process that is timely, "
                "culturally sensitive, and accessible."
            )
        if "affected_communities_consulted" in gaps:
            recommendations.append(
                "Consult affected communities and workers' representatives "
                "as part of due diligence steps."
            )
        if "engagement_outcomes_documented" in gaps:
            recommendations.append(
                "Document engagement outcomes and how they have informed "
                "due diligence measures."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_11,
            article_title=ARTICLE_TITLES[ArticleReference.ART_11.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_11.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_12(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 12 - Notification mechanism and complaints procedure.

        Evaluates whether the company has an accessible and documented
        complaints procedure / grievance mechanism.

        Args:
            profile: CompanyProfile with grievance mechanism attributes.

        Returns:
            ArticleAssessment for Art 12.
        """
        criteria_map = {
            "has_grievance_mechanism": profile.has_grievance_mechanism,
            "grievance_mechanism_accessible": profile.grievance_mechanism_accessible,
            "grievance_mechanism_documented": profile.grievance_mechanism_documented,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_grievance_mechanism" in gaps:
            recommendations.append(
                "Establish a notification mechanism and complaints procedure "
                "for persons and organisations to submit concerns about "
                "adverse impacts (Art 12)."
            )
        if "grievance_mechanism_accessible" in gaps:
            recommendations.append(
                "Ensure the complaints procedure is accessible to affected "
                "stakeholders, including in relevant languages."
            )
        if "grievance_mechanism_documented" in gaps:
            recommendations.append(
                "Document the complaints procedure, including the process "
                "for handling and responding to complaints."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_12,
            article_title=ARTICLE_TITLES[ArticleReference.ART_12.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_12.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_13(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 13 - Monitoring.

        Evaluates whether the company has adequate monitoring processes
        including periodic assessments, KPIs, and third-party verification.

        Args:
            profile: CompanyProfile with monitoring attributes.

        Returns:
            ArticleAssessment for Art 13.
        """
        criteria_map = {
            "monitoring_process_defined": profile.monitoring_process_defined,
            "periodic_assessments_conducted": profile.periodic_assessments_conducted,
            "kpis_defined": profile.kpis_defined,
            "third_party_verification": profile.third_party_verification,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "monitoring_process_defined" in gaps:
            recommendations.append(
                "Define a monitoring process to assess the implementation "
                "and effectiveness of due diligence measures (Art 13)."
            )
        if "periodic_assessments_conducted" in gaps:
            recommendations.append(
                "Conduct periodic assessments of own operations, "
                "subsidiaries, and business partners."
            )
        if "kpis_defined" in gaps:
            recommendations.append(
                "Define qualitative and quantitative KPIs to measure "
                "due diligence effectiveness."
            )
        if "third_party_verification" in gaps:
            recommendations.append(
                "Consider engaging independent third-party verification "
                "of due diligence measures and outcomes."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_13,
            article_title=ARTICLE_TITLES[ArticleReference.ART_13.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_13.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_14(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 14 - Reporting and communication.

        Evaluates whether the company reports on due diligence annually,
        covering impacts, measures, and making reports publicly available.

        Args:
            profile: CompanyProfile with reporting attributes.

        Returns:
            ArticleAssessment for Art 14.
        """
        criteria_map = {
            "annual_reporting_committed": profile.annual_reporting_committed,
            "reporting_covers_impacts": profile.reporting_covers_impacts,
            "reporting_covers_measures": profile.reporting_covers_measures,
            "reporting_publicly_available": profile.reporting_publicly_available,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "annual_reporting_committed" in gaps:
            recommendations.append(
                "Commit to annual reporting on due diligence policies, "
                "actions taken, and their outcomes (Art 14)."
            )
        if "reporting_covers_impacts" in gaps:
            recommendations.append(
                "Ensure reporting covers identified adverse impacts."
            )
        if "reporting_covers_measures" in gaps:
            recommendations.append(
                "Include prevention, mitigation, and remediation measures "
                "in the DD report."
            )
        if "reporting_publicly_available" in gaps:
            recommendations.append(
                "Make the due diligence report publicly available "
                "on the company's website."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_14,
            article_title=ARTICLE_TITLES[ArticleReference.ART_14.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_14.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_15(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 15 / Art 22 - Climate transition plan.

        Evaluates whether the company has adopted and is implementing
        a climate transition plan aligned with the Paris Agreement.

        Args:
            profile: CompanyProfile with climate transition plan attributes.

        Returns:
            ArticleAssessment for Art 15.
        """
        criteria_map = {
            "has_climate_transition_plan": profile.has_climate_transition_plan,
            "climate_plan_paris_aligned": profile.climate_plan_paris_aligned,
            "climate_plan_has_targets": profile.climate_plan_has_targets,
            "climate_plan_has_actions": profile.climate_plan_has_actions,
            "climate_plan_has_investments": profile.climate_plan_has_investments,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_climate_transition_plan" in gaps:
            recommendations.append(
                "Adopt and put into effect a transition plan for climate "
                "change mitigation aimed at limiting global warming to "
                "1.5 C (Art 22)."
            )
        if "climate_plan_paris_aligned" in gaps:
            recommendations.append(
                "Ensure the transition plan is aligned with the objective "
                "of the Paris Agreement to limit global warming to 1.5 C."
            )
        if "climate_plan_has_targets" in gaps:
            recommendations.append(
                "Include time-bound, science-based GHG reduction targets "
                "for 2030 and 2050 in the transition plan."
            )
        if "climate_plan_has_actions" in gaps:
            recommendations.append(
                "Define concrete implementation actions including "
                "decarbonisation levers and their expected impact."
            )
        if "climate_plan_has_investments" in gaps:
            recommendations.append(
                "Describe investment needs and the financial resources "
                "allocated to implementing the transition plan."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_15,
            article_title=ARTICLE_TITLES[ArticleReference.ART_15.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_15.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_22(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 22 - Paris Agreement alignment (detailed).

        Provides a focused assessment on Paris Agreement alignment
        aspects of the climate transition plan, complementing Art 15.

        Args:
            profile: CompanyProfile with climate plan attributes.

        Returns:
            ArticleAssessment for Art 22.
        """
        criteria_map = {
            "has_climate_transition_plan": profile.has_climate_transition_plan,
            "paris_aligned": profile.climate_plan_paris_aligned,
            "has_time_bound_targets": profile.climate_plan_has_targets,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_climate_transition_plan" in gaps:
            recommendations.append(
                "A transition plan for climate change mitigation is "
                "mandatory under Art 22 of the CSDDD."
            )
        if "paris_aligned" in gaps:
            recommendations.append(
                "Align the transition plan with 1.5 C pathways based on "
                "the latest IPCC scientific evidence."
            )
        if "has_time_bound_targets" in gaps:
            recommendations.append(
                "Set absolute GHG emission reduction targets for 2030 "
                "and climate neutrality by 2050."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_22,
            article_title=ARTICLE_TITLES[ArticleReference.ART_22.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_22.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_art_29(self, profile: CompanyProfile) -> ArticleAssessment:
        """Assess Art 29 - Civil liability.

        Evaluates whether the company is aware of and prepared for
        civil liability obligations under the CSDDD.

        Args:
            profile: CompanyProfile with civil liability attributes.

        Returns:
            ArticleAssessment for Art 29.
        """
        criteria_map = {
            "has_civil_liability_awareness": profile.has_civil_liability_awareness,
            "civil_liability_insurance": profile.civil_liability_insurance,
        }

        met = sum(1 for v in criteria_map.values() if v)
        total = len(criteria_map)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )

        gaps = [k for k, v in criteria_map.items() if not v]
        recommendations = []
        if "has_civil_liability_awareness" in gaps:
            recommendations.append(
                "Ensure the board and legal teams are aware of civil "
                "liability provisions under Art 29, which allow affected "
                "persons to claim damages for harm resulting from "
                "non-compliance with DD obligations."
            )
        if "civil_liability_insurance" in gaps:
            recommendations.append(
                "Review and extend liability insurance to cover potential "
                "claims arising under CSDDD Art 29 civil liability."
            )

        return ArticleAssessment(
            article=ArticleReference.ART_29,
            article_title=ARTICLE_TITLES[ArticleReference.ART_29.value],
            status=self._score_to_status(score),
            score=score,
            weight=ARTICLE_WEIGHTS[ArticleReference.ART_29.value],
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _assess_article(
        self,
        article: ArticleReference,
        profile: CompanyProfile,
    ) -> ArticleAssessment:
        """Dispatch to the appropriate per-article assessment method.

        Args:
            article: The CSDDD article to assess.
            profile: Company profile with relevant attributes.

        Returns:
            ArticleAssessment for the specified article.

        Raises:
            ValueError: If the article is not supported.
        """
        dispatch: Dict[ArticleReference, Any] = {
            ArticleReference.ART_5: self._assess_art_5,
            ArticleReference.ART_6: self._assess_art_6,
            ArticleReference.ART_7: self._assess_art_7,
            ArticleReference.ART_8: self._assess_art_8,
            ArticleReference.ART_9: self._assess_art_9,
            ArticleReference.ART_10: self._assess_art_10,
            ArticleReference.ART_11: self._assess_art_11,
            ArticleReference.ART_12: self._assess_art_12,
            ArticleReference.ART_13: self._assess_art_13,
            ArticleReference.ART_14: self._assess_art_14,
            ArticleReference.ART_15: self._assess_art_15,
            ArticleReference.ART_22: self._assess_art_22,
            ArticleReference.ART_29: self._assess_art_29,
        }

        handler = dispatch.get(article)
        if handler is None:
            raise ValueError(f"Unsupported article: {article}")
        return handler(profile)

    # ------------------------------------------------------------------ #
    # Policy Area Assessment                                               #
    # ------------------------------------------------------------------ #

    def _assess_policy_area(
        self,
        area: PolicyArea,
        profile: CompanyProfile,
    ) -> PolicyAssessment:
        """Assess a single policy area against its criteria.

        Counts how many of the area's criteria are satisfied by the
        company profile and computes a score from 0 to 100.

        Args:
            area: The policy area to assess.
            profile: CompanyProfile with relevant attributes.

        Returns:
            PolicyAssessment for the specified area.
        """
        criteria_list = POLICY_AREA_CRITERIA.get(area.value, [])
        if not criteria_list:
            return PolicyAssessment(
                policy_area=area,
                status=ComplianceStatus.NOT_APPLICABLE,
                score=Decimal("0"),
                criteria_met=0,
                criteria_total=0,
                gaps=[],
                recommendations=[],
            )

        met = 0
        gaps: List[str] = []
        for criterion in criteria_list:
            value = getattr(profile, criterion, False)
            if value:
                met += 1
            else:
                gaps.append(criterion)

        total = len(criteria_list)
        score = _round_val(
            _decimal(met) / _decimal(total) * Decimal("100"), 1
        )
        status = self._score_to_status(score)

        recommendations = self._generate_policy_area_recommendations(area, gaps)

        return PolicyAssessment(
            policy_area=area,
            status=status,
            score=score,
            criteria_met=met,
            criteria_total=total,
            gaps=gaps,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------ #
    # Scoring Helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_to_status(score: Decimal) -> ComplianceStatus:
        """Convert a numeric score to a ComplianceStatus.

        Thresholds:
        - >= 80: COMPLIANT
        - >= 40: PARTIALLY_COMPLIANT
        - < 40:  NON_COMPLIANT

        Args:
            score: Score from 0 to 100.

        Returns:
            ComplianceStatus enum value.
        """
        if score >= Decimal("80"):
            return ComplianceStatus.COMPLIANT
        elif score >= Decimal("40"):
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _calculate_overall_score(
        self,
        assessments: List[ArticleAssessment],
    ) -> Tuple[Decimal, ComplianceStatus]:
        """Calculate weighted overall compliance score.

        Uses the weights defined in ARTICLE_WEIGHTS to compute
        a weighted average score across all article assessments.

        Args:
            assessments: List of ArticleAssessment results.

        Returns:
            Tuple of (overall_score, overall_status).
        """
        if not assessments:
            return Decimal("0"), ComplianceStatus.NON_COMPLIANT

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for assessment in assessments:
            weight = ARTICLE_WEIGHTS.get(
                assessment.article.value, Decimal("0")
            )
            weighted_sum += assessment.score * weight
            total_weight += weight

        overall_score = _safe_divide(weighted_sum, total_weight)
        overall_score = _round_val(overall_score, 1)
        overall_status = self._score_to_status(overall_score)

        return overall_score, overall_status

    # ------------------------------------------------------------------ #
    # Recommendation Generation                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_art_5_recommendations(gaps: List[str]) -> List[str]:
        """Generate recommendations for Art 5 gaps.

        Maps each missing Art 5 criterion to a specific, actionable
        recommendation text.

        Args:
            gaps: List of unmet Art 5 criterion keys.

        Returns:
            List of recommendation strings.
        """
        rec_map: Dict[str, str] = {
            "has_dd_policy": (
                "Adopt a formal due diligence policy that integrates "
                "responsible business conduct into all relevant policies "
                "and risk management systems (Art 5 Para 1)."
            ),
            "dd_policy_describes_approach": (
                "Include a description of the company's approach to due "
                "diligence, including the long-term and short-term strategy "
                "(Art 5 Para 2(a))."
            ),
            "dd_policy_has_code_of_conduct": (
                "Include a code of conduct describing rules and principles "
                "for employees and subsidiaries (Art 5 Para 2(b))."
            ),
            "code_rules_for_employees": (
                "Ensure the code of conduct specifies rules and principles "
                "that employees at all levels must follow."
            ),
            "code_rules_for_subsidiaries": (
                "Extend the code of conduct to apply to all subsidiaries."
            ),
            "dd_policy_describes_processes": (
                "Describe the processes put in place to implement due "
                "diligence, including measures to verify compliance "
                "(Art 5 Para 2(c))."
            ),
            "dd_policy_verification_measures": (
                "Include measures to verify compliance with the code of "
                "conduct, including audit and monitoring mechanisms."
            ),
            "dd_policy_extended_to_partners": (
                "Describe how the due diligence policy and code of conduct "
                "are extended to business partners in the value chain."
            ),
            "dd_policy_updated_annually": (
                "Update the due diligence policy at least annually and "
                "whenever a significant change occurs (Art 5 Para 3)."
            ),
            "dd_policy_approved_by_board": (
                "Ensure the due diligence policy is approved at the "
                "appropriate governance body level."
            ),
        }

        recommendations = []
        for gap in gaps:
            rec = rec_map.get(gap)
            if rec:
                recommendations.append(rec)
        return recommendations

    @staticmethod
    def _generate_policy_area_recommendations(
        area: PolicyArea,
        gaps: List[str],
    ) -> List[str]:
        """Generate recommendations for policy area gaps.

        Maps each missing criterion within a policy area to an
        actionable recommendation.

        Args:
            area: The policy area being assessed.
            gaps: List of unmet criteria keys.

        Returns:
            List of recommendation strings.
        """
        rec_map: Dict[str, Dict[str, str]] = {
            PolicyArea.CODE_OF_CONDUCT.value: {
                "has_code_of_conduct": (
                    "Adopt a code of conduct covering responsible business "
                    "conduct expectations."
                ),
                "code_covers_human_rights": (
                    "Extend the code of conduct to explicitly cover "
                    "human rights standards."
                ),
                "code_covers_environment": (
                    "Include environmental standards and expectations "
                    "in the code of conduct."
                ),
                "code_covers_subsidiaries": (
                    "Apply the code of conduct to all subsidiaries."
                ),
                "code_covers_business_partners": (
                    "Extend the code of conduct to business partners "
                    "in the value chain."
                ),
                "code_reviewed_annually": (
                    "Review and update the code of conduct at least annually."
                ),
            },
            PolicyArea.RISK_MANAGEMENT.value: {
                "has_risk_management_framework": (
                    "Establish a risk management framework that integrates "
                    "human rights and environmental risks."
                ),
                "risk_framework_covers_hr": (
                    "Ensure the risk framework explicitly covers human "
                    "rights risks across the value chain."
                ),
                "risk_framework_covers_env": (
                    "Include environmental risks in the risk management "
                    "framework."
                ),
                "risk_assessment_frequency_adequate": (
                    "Conduct risk assessments at least annually and upon "
                    "significant changes."
                ),
                "risk_mitigation_measures_defined": (
                    "Define specific risk mitigation measures for identified "
                    "high-risk areas."
                ),
            },
            PolicyArea.GOVERNANCE.value: {
                "board_oversight_defined": (
                    "Define board-level oversight responsibilities for "
                    "due diligence."
                ),
                "dedicated_dd_officer": (
                    "Appoint a dedicated due diligence officer or function."
                ),
                "governance_body_receives_reports": (
                    "Ensure the governing body receives regular reports "
                    "on due diligence progress."
                ),
                "accountability_mechanisms_exist": (
                    "Implement accountability mechanisms linking DD "
                    "performance to management responsibilities."
                ),
            },
            PolicyArea.MONITORING.value: {
                "monitoring_process_defined": (
                    "Define a monitoring process for due diligence measures."
                ),
                "periodic_assessments_conducted": (
                    "Conduct periodic assessments of DD implementation."
                ),
                "kpis_defined": (
                    "Define quantitative and qualitative KPIs for DD."
                ),
                "third_party_verification": (
                    "Engage independent third parties for verification."
                ),
            },
            PolicyArea.REPORTING.value: {
                "annual_reporting_committed": (
                    "Commit to annual reporting on due diligence."
                ),
                "reporting_covers_impacts": (
                    "Cover identified adverse impacts in reporting."
                ),
                "reporting_covers_measures": (
                    "Include measures taken in the DD report."
                ),
                "reporting_publicly_available": (
                    "Make the DD report publicly available online."
                ),
            },
            PolicyArea.STAKEHOLDER_ENGAGEMENT.value: {
                "stakeholder_mapping_conducted": (
                    "Conduct stakeholder mapping to identify affected groups."
                ),
                "engagement_process_defined": (
                    "Define a meaningful stakeholder engagement process."
                ),
                "affected_communities_consulted": (
                    "Consult affected communities as part of DD."
                ),
                "engagement_outcomes_documented": (
                    "Document stakeholder engagement outcomes."
                ),
            },
        }

        area_recs = rec_map.get(area.value, {})
        recommendations = []
        for gap in gaps:
            rec = area_recs.get(gap)
            if rec:
                recommendations.append(rec)
        return recommendations

    # ------------------------------------------------------------------ #
    # Main Assessment Entry Point                                          #
    # ------------------------------------------------------------------ #

    def assess_policy(
        self, profile: CompanyProfile
    ) -> DueDiligencePolicyResult:
        """Run a complete CSDDD due diligence policy assessment.

        Performs scope determination, per-article compliance assessments,
        policy area assessments, and calculates an overall weighted score.

        This is the primary entry point for the engine.

        Args:
            profile: CompanyProfile with all relevant attributes.

        Returns:
            DueDiligencePolicyResult with complete assessment data
            and provenance hash.

        Raises:
            ValueError: If profile data is invalid.
        """
        start_time = time.time()
        logger.info(
            "Starting CSDDD DD policy assessment for %s (year=%d)",
            profile.company_name, profile.reporting_year,
        )

        # Step 1: Scope determination
        scope_assessment = self.assess_scope(profile)

        # Step 2: Assess all articles
        article_assessments: List[ArticleAssessment] = []
        for article in ArticleReference:
            assessment = self._assess_article(article, profile)
            article_assessments.append(assessment)

        # Step 3: Assess policy areas
        policy_assessments: List[PolicyAssessment] = []
        for area in PolicyArea:
            pa = self._assess_policy_area(area, profile)
            policy_assessments.append(pa)

        # Step 4: Calculate overall score
        overall_score, overall_status = self._calculate_overall_score(
            article_assessments
        )

        # Step 5: Aggregate totals
        total_met = sum(a.criteria_met for a in article_assessments)
        total_criteria = sum(a.criteria_total for a in article_assessments)

        # Step 6: Aggregate gaps
        gaps_summary: Dict[str, int] = {}
        for a in article_assessments:
            gaps_summary[a.article.value] = len(a.gaps)

        # Step 7: Collect top recommendations (prioritised by score)
        sorted_articles = sorted(
            article_assessments, key=lambda a: a.score
        )
        all_recommendations: List[str] = []
        for a in sorted_articles:
            for rec in a.recommendations:
                if rec not in all_recommendations:
                    all_recommendations.append(rec)

        processing_time_ms = (time.time() - start_time) * 1000

        # Step 8: Build result
        result = DueDiligencePolicyResult(
            company=profile.company_name,
            reporting_year=profile.reporting_year,
            scope_assessment=scope_assessment,
            article_assessments=article_assessments,
            policy_assessments=policy_assessments,
            overall_score=overall_score,
            overall_status=overall_status,
            total_criteria_met=total_met,
            total_criteria=total_criteria,
            recommendations=all_recommendations,
            gaps_summary=gaps_summary,
            processing_time_ms=_round2(processing_time_ms),
            assessed_at=utcnow(),
        )

        # Step 9: Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "CSDDD DD policy assessment complete for %s: "
            "score=%.1f, status=%s, criteria=%d/%d, time=%.2fms",
            profile.company_name, float(overall_score),
            overall_status.value, total_met, total_criteria,
            processing_time_ms,
        )

        return result
