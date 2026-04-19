"""
PACK-019 CSDDD Readiness Pack - Pack Configuration

This module implements the complete PackConfig for the Corporate Sustainability
Due Diligence Directive (CSDDD / CS3D) Readiness Pack. It provides Pydantic v2
models covering every aspect of CSDDD compliance: company scoping, adverse impact
identification, prevention/mitigation measures, grievance mechanisms, climate
transition planning, civil liability provisions, and stakeholder engagement.

Regulatory Context:
    - CSDDD: Directive (EU) 2024/1760 (13 June 2024, published OJ 5 July 2024)
    - OECD Due Diligence Guidance for Responsible Business Conduct (2018)
    - UN Guiding Principles on Business and Human Rights (2011)
    - ILO Declaration on Fundamental Principles and Rights at Work (1998)
    - Paris Agreement (2015) - Climate transition plan alignment
    - EU Taxonomy Regulation 2020/852 - Activity classification
    - CSRD Directive (EU) 2022/2464 - Reporting interoperability

Phase-In Schedule (per CSDDD Art. 37-38):
    - Phase 1 (26 July 2027): >5000 employees AND >EUR 1.5bn net turnover
    - Phase 2 (26 July 2028): >3000 employees AND >EUR 900m net turnover
    - Phase 3 (26 July 2029): >1000 employees AND >EUR 450m net turnover
    - Non-EU: Same thresholds on EU-generated turnover

OECD Due Diligence Six-Step Framework:
    - Step 1: Embed responsible business conduct into policies
    - Step 2: Identify and assess adverse impacts
    - Step 3: Cease, prevent, or mitigate adverse impacts
    - Step 4: Track implementation and results
    - Step 5: Communicate how impacts are addressed
    - Step 6: Provide for or cooperate in remediation

Configuration Merge Order (later overrides earlier):
    1. Base pack defaults (this module)
    2. Sector preset YAML (manufacturing / extractives / financial_services /
       retail / technology / agriculture)
    3. Environment overrides (CSDDD_PACK_* environment variables)
    4. Explicit runtime overrides

Example:
    >>> config = PackConfig()
    >>> warnings = config.validate_thresholds()
    >>> print(config.config_hash)
    >>> mfg_config = PackConfig.from_preset("manufacturing")
    >>> articles = mfg_config.get_article_requirements()
"""

import hashlib
import json
import logging
import os
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"

ALL_ENGINES: List[str] = [
    "scope_assessment",
    "impact_identification",
    "prevention_planning",
    "grievance_management",
    "climate_transition",
    "liability_assessment",
    "stakeholder_engagement",
    "remediation_tracking",
]

ALL_WORKFLOWS: List[str] = [
    "company_intake",
    "scope_determination",
    "value_chain_mapping",
    "impact_assessment",
    "prevention_measures",
    "grievance_setup",
    "climate_plan_review",
    "liability_screening",
    "stakeholder_consultation",
    "remediation_workflow",
    "periodic_monitoring",
    "annual_reporting",
]

AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing": "Manufacturing - supply chain labour risks, process pollution, raw material sourcing",
    "extractives": "Extractives - indigenous rights, environmental degradation, conflict minerals",
    "financial_services": "Financial services - financed impacts, portfolio screening, ESG integration",
    "retail": "Retail - apparel supply chains, food sourcing, consumer product safety",
    "technology": "Technology - mineral supply chains, data privacy, electronics labour",
    "agriculture": "Agriculture - land rights, deforestation, labour exploitation, water stress",
}


# =============================================================================
# Enums - CSDDD Readiness Pack enumeration types (13 enums)
# =============================================================================


class CompanyScope(str, Enum):
    """Company scope classification under CSDDD Art. 2 phase-in schedule.

    Phase 1 (2027): >5000 employees AND >EUR 1.5bn turnover
    Phase 2 (2028): >3000 employees AND >EUR 900m turnover
    Phase 3 (2029): >1000 employees AND >EUR 450m turnover
    """

    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"
    PHASE_3 = "PHASE_3"
    NOT_IN_SCOPE = "NOT_IN_SCOPE"
    VOLUNTARY = "VOLUNTARY"


class SectorType(str, Enum):
    """Business sector classification for sector-specific due diligence guidance."""

    MANUFACTURING = "MANUFACTURING"
    EXTRACTIVES = "EXTRACTIVES"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    RETAIL = "RETAIL"
    TECHNOLOGY = "TECHNOLOGY"
    AGRICULTURE = "AGRICULTURE"
    ENERGY = "ENERGY"
    CONSTRUCTION = "CONSTRUCTION"
    TRANSPORT = "TRANSPORT"
    SERVICES = "SERVICES"
    OTHER = "OTHER"


class AdverseImpactType(str, Enum):
    """Adverse impact categories per CSDDD Art. 3(1)(b)-(c) and Annex Part I/II.

    Human Rights: Annex Part I - international instruments
    Environmental: Annex Part II - environmental conventions
    """

    HUMAN_RIGHTS = "HUMAN_RIGHTS"
    ENVIRONMENTAL = "ENVIRONMENTAL"


class ImpactSeverity(str, Enum):
    """Impact severity classification per OECD Due Diligence Guidance (2018).

    Severity is assessed based on scale, scope, and irremediability.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ComplianceStatus(str, Enum):
    """Compliance status for individual CSDDD articles and requirements."""

    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MeasureType(str, Enum):
    """Types of appropriate measures per CSDDD Art. 10-12.

    Prevention (Art. 10): Prevent potential adverse impacts
    Mitigation (Art. 11): Mitigate actual adverse impacts
    Cessation (Art. 8(3)(b)): Cease causing adverse impacts
    Remediation (Art. 12): Provide or cooperate in remediation
    """

    PREVENTION = "PREVENTION"
    MITIGATION = "MITIGATION"
    CESSATION = "CESSATION"
    REMEDIATION = "REMEDIATION"


class StakeholderGroup(str, Enum):
    """Stakeholder groups for meaningful engagement per CSDDD Art. 13.

    The Directive requires engagement with affected stakeholders including
    workers, trade unions, communities, indigenous peoples, and civil society.
    """

    WORKERS = "WORKERS"
    TRADE_UNIONS = "TRADE_UNIONS"
    COMMUNITIES = "COMMUNITIES"
    INDIGENOUS_PEOPLES = "INDIGENOUS_PEOPLES"
    NGOS = "NGOS"
    INVESTORS = "INVESTORS"
    CONSUMERS = "CONSUMERS"
    REGULATORS = "REGULATORS"


class GrievanceChannel(str, Enum):
    """Grievance mechanism channels per CSDDD Art. 14.

    Companies must establish or participate in grievance mechanisms that
    enable any person to submit complaints about adverse impacts.
    """

    HOTLINE = "HOTLINE"
    EMAIL = "EMAIL"
    WEB_PORTAL = "WEB_PORTAL"
    IN_PERSON = "IN_PERSON"
    MOBILE_APP = "MOBILE_APP"
    POSTAL = "POSTAL"
    TRADE_UNION_REP = "TRADE_UNION_REP"


class TransitionPlanStatus(str, Enum):
    """Climate transition plan status per CSDDD Art. 22.

    Companies must adopt and put into effect a transition plan for climate
    change mitigation aligned with the Paris Agreement 1.5C target.
    """

    DRAFTED = "DRAFTED"
    APPROVED = "APPROVED"
    IMPLEMENTING = "IMPLEMENTING"
    ON_TRACK = "ON_TRACK"
    BEHIND_SCHEDULE = "BEHIND_SCHEDULE"
    ACHIEVED = "ACHIEVED"


class CacheBackend(str, Enum):
    """Backend for configuration and lookup caching."""

    MEMORY = "MEMORY"
    REDIS = "REDIS"
    DISABLED = "DISABLED"


class ArticleReference(str, Enum):
    """CSDDD article references for mapping requirements to legal provisions.

    Art. 5:  Integration of due diligence into policies
    Art. 6:  Identifying actual and potential adverse impacts
    Art. 7:  Prioritisation of identified adverse impacts
    Art. 8:  Prevention of potential adverse impacts
    Art. 9:  Prevention measures in contractual assurances
    Art. 10: Verification measures
    Art. 11: Mitigation of actual adverse impacts
    Art. 12: Remediation of actual adverse impacts
    Art. 13: Meaningful engagement with stakeholders
    Art. 14: Grievance mechanisms (notification mechanism)
    Art. 15: Monitoring of due diligence policy and measures
    Art. 16: Communication and reporting
    Art. 17: Administrative supervision - Member State authorities
    Art. 18: Supervisory authority designation
    Art. 19: Powers of supervisory authorities
    Art. 20: Penalties - pecuniary penalties and injunctive relief
    Art. 21: European Network of Supervisory Authorities
    Art. 22: Climate transition plan
    Art. 23: Maximum harmonisation - relationship with national law
    Art. 24: Transposition by Member States
    Art. 25: Exercise of the delegation
    Art. 26: Civil liability
    Art. 27: Reporting and review
    Art. 28: Amendment to Directive (EU) 2019/1937
    Art. 29: Entry into force and application dates
    """

    ART_5 = "ART_5"
    ART_6 = "ART_6"
    ART_7 = "ART_7"
    ART_8 = "ART_8"
    ART_9 = "ART_9"
    ART_10 = "ART_10"
    ART_11 = "ART_11"
    ART_12 = "ART_12"
    ART_13 = "ART_13"
    ART_14 = "ART_14"
    ART_15 = "ART_15"
    ART_16 = "ART_16"
    ART_17 = "ART_17"
    ART_18 = "ART_18"
    ART_19 = "ART_19"
    ART_20 = "ART_20"
    ART_21 = "ART_21"
    ART_22 = "ART_22"
    ART_23 = "ART_23"
    ART_24 = "ART_24"
    ART_25 = "ART_25"
    ART_26 = "ART_26"
    ART_27 = "ART_27"
    ART_28 = "ART_28"
    ART_29 = "ART_29"


class OECDStep(str, Enum):
    """OECD Due Diligence six-step framework (OECD 2018 Guidance).

    The CSDDD is explicitly aligned with OECD due diligence steps.
    """

    STEP_1_EMBED = "STEP_1_EMBED"
    STEP_2_IDENTIFY = "STEP_2_IDENTIFY"
    STEP_3_PREVENT = "STEP_3_PREVENT"
    STEP_4_TRACK = "STEP_4_TRACK"
    STEP_5_COMMUNICATE = "STEP_5_COMMUNICATE"
    STEP_6_REMEDIATE = "STEP_6_REMEDIATE"


# =============================================================================
# Article-to-OECD Step Mapping
# =============================================================================

ARTICLE_OECD_MAPPING: Dict[ArticleReference, OECDStep] = {
    ArticleReference.ART_5: OECDStep.STEP_1_EMBED,
    ArticleReference.ART_6: OECDStep.STEP_2_IDENTIFY,
    ArticleReference.ART_7: OECDStep.STEP_2_IDENTIFY,
    ArticleReference.ART_8: OECDStep.STEP_3_PREVENT,
    ArticleReference.ART_9: OECDStep.STEP_3_PREVENT,
    ArticleReference.ART_10: OECDStep.STEP_3_PREVENT,
    ArticleReference.ART_11: OECDStep.STEP_3_PREVENT,
    ArticleReference.ART_12: OECDStep.STEP_6_REMEDIATE,
    ArticleReference.ART_13: OECDStep.STEP_2_IDENTIFY,
    ArticleReference.ART_14: OECDStep.STEP_6_REMEDIATE,
    ArticleReference.ART_15: OECDStep.STEP_4_TRACK,
    ArticleReference.ART_16: OECDStep.STEP_5_COMMUNICATE,
    ArticleReference.ART_22: OECDStep.STEP_1_EMBED,
}

# =============================================================================
# Article Descriptions for Reporting
# =============================================================================

ARTICLE_DESCRIPTIONS: Dict[ArticleReference, str] = {
    ArticleReference.ART_5: "Integration of due diligence into company policies and risk management systems",
    ArticleReference.ART_6: "Identification and assessment of actual and potential adverse impacts",
    ArticleReference.ART_7: "Prioritisation of identified adverse impacts based on severity and likelihood",
    ArticleReference.ART_8: "Prevention of potential adverse human rights and environmental impacts",
    ArticleReference.ART_9: "Contractual assurances from business partners for prevention measures",
    ArticleReference.ART_10: "Verification of compliance by business partners (audits, monitoring)",
    ArticleReference.ART_11: "Mitigation (bringing to an end) of actual adverse impacts",
    ArticleReference.ART_12: "Remediation of actual adverse impacts including financial compensation",
    ArticleReference.ART_13: "Meaningful engagement with affected stakeholders throughout due diligence",
    ArticleReference.ART_14: "Notification mechanism (grievance) for submitting complaints",
    ArticleReference.ART_15: "Periodic assessment and monitoring of due diligence effectiveness",
    ArticleReference.ART_16: "Annual public reporting on due diligence policies and outcomes",
    ArticleReference.ART_17: "Administrative supervision by designated Member State authorities",
    ArticleReference.ART_18: "Designation of national supervisory authorities",
    ArticleReference.ART_19: "Investigative and enforcement powers of supervisory authorities",
    ArticleReference.ART_20: "Pecuniary penalties based on turnover and injunctive relief",
    ArticleReference.ART_21: "European Network of Supervisory Authorities for coordination",
    ArticleReference.ART_22: "Climate transition plan aligned with Paris Agreement 1.5C pathway",
    ArticleReference.ART_23: "Maximum harmonisation provisions for uniform application",
    ArticleReference.ART_24: "Transposition by Member States into national law",
    ArticleReference.ART_25: "Exercise of the delegation for supplementary acts",
    ArticleReference.ART_26: "Civil liability for failure to comply with due diligence obligations",
    ArticleReference.ART_27: "Commission review and reporting on Directive effectiveness",
    ArticleReference.ART_28: "Amendment of Whistleblower Protection Directive (EU) 2019/1937",
    ArticleReference.ART_29: "Entry into force, application dates, and phase-in schedule",
}


# =============================================================================
# Sub-Configurations
# =============================================================================


class ScopeConfig(BaseModel):
    """Company scope determination thresholds per CSDDD Art. 2.

    The CSDDD applies in three phases based on employee count and net
    worldwide turnover. Non-EU companies are scoped on EU-generated turnover.

    Attributes:
        phase_1_employee_threshold: Phase 1 employee threshold (default 5000).
        phase_1_turnover_threshold: Phase 1 net turnover in EUR (default 1.5bn).
        phase_2_employee_threshold: Phase 2 employee threshold (default 3000).
        phase_2_turnover_threshold: Phase 2 net turnover in EUR (default 900m).
        phase_3_employee_threshold: Phase 3 employee threshold (default 1000).
        phase_3_turnover_threshold: Phase 3 net turnover in EUR (default 450m).
        include_non_eu: Whether to include non-EU companies based on EU turnover.
        franchise_threshold: Franchise/licensing turnover threshold in EUR.
        group_consolidation: Whether to apply group-level consolidation.
        headcount_method: Method for counting employees (FTE or headcount).
    """

    phase_1_employee_threshold: int = Field(
        5000, ge=1,
        description="Phase 1 (2027): minimum employee headcount",
    )
    phase_1_turnover_threshold: Decimal = Field(
        Decimal("1500000000"),
        description="Phase 1 (2027): minimum net worldwide turnover in EUR",
    )
    phase_2_employee_threshold: int = Field(
        3000, ge=1,
        description="Phase 2 (2028): minimum employee headcount",
    )
    phase_2_turnover_threshold: Decimal = Field(
        Decimal("900000000"),
        description="Phase 2 (2028): minimum net worldwide turnover in EUR",
    )
    phase_3_employee_threshold: int = Field(
        1000, ge=1,
        description="Phase 3 (2029): minimum employee headcount",
    )
    phase_3_turnover_threshold: Decimal = Field(
        Decimal("450000000"),
        description="Phase 3 (2029): minimum net worldwide turnover in EUR",
    )
    include_non_eu: bool = Field(
        True,
        description="Include non-EU companies scoped on EU-generated turnover (Art. 2(2))",
    )
    franchise_threshold: Decimal = Field(
        Decimal("80000000"),
        description="Franchise/licensing royalty threshold in EUR (Art. 2(1)(b))",
    )
    group_consolidation: bool = Field(
        True,
        description="Apply group-level employee/turnover consolidation (Art. 2(3))",
    )
    headcount_method: str = Field(
        "HEADCOUNT",
        description="Employee counting method: HEADCOUNT (default) or FTE",
    )

    @field_validator("headcount_method")
    @classmethod
    def validate_headcount_method(cls, v: str) -> str:
        """Ensure headcount method is valid."""
        allowed = {"HEADCOUNT", "FTE"}
        if v.upper() not in allowed:
            raise ValueError(f"headcount_method must be one of {sorted(allowed)}, got: {v}")
        return v.upper()

    def determine_scope(
        self,
        employee_count: int,
        net_turnover: Decimal,
    ) -> CompanyScope:
        """Determine the company scope phase based on employee count and turnover.

        Both thresholds must be met (AND logic) per CSDDD Art. 2(1).

        Args:
            employee_count: Number of employees (headcount or FTE).
            net_turnover: Net worldwide turnover in EUR.

        Returns:
            CompanyScope enum value.
        """
        if (
            employee_count >= self.phase_1_employee_threshold
            and net_turnover >= self.phase_1_turnover_threshold
        ):
            return CompanyScope.PHASE_1
        if (
            employee_count >= self.phase_2_employee_threshold
            and net_turnover >= self.phase_2_turnover_threshold
        ):
            return CompanyScope.PHASE_2
        if (
            employee_count >= self.phase_3_employee_threshold
            and net_turnover >= self.phase_3_turnover_threshold
        ):
            return CompanyScope.PHASE_3
        return CompanyScope.NOT_IN_SCOPE

    def get_application_date(self, scope: CompanyScope) -> Optional[date]:
        """Return the application date for a given scope phase.

        Args:
            scope: CompanyScope value.

        Returns:
            Date when obligations begin, or None if not in scope.
        """
        application_dates = {
            CompanyScope.PHASE_1: date(2027, 7, 26),
            CompanyScope.PHASE_2: date(2028, 7, 26),
            CompanyScope.PHASE_3: date(2029, 7, 26),
        }
        return application_dates.get(scope)


class ImpactConfig(BaseModel):
    """Adverse impact identification and prioritisation per CSDDD Art. 6-7.

    Severity is assessed based on three dimensions from the OECD framework:
    scale (gravity of the impact), scope (number of individuals affected),
    and irremediability (whether the impact can be restored).

    Attributes:
        severity_weights: Weight for each severity dimension (scale, scope, irremediability).
        likelihood_weights: Likelihood level weights for risk matrix calculation.
        risk_matrix_thresholds: Thresholds for mapping risk score to severity level.
        max_impacts: Maximum number of impacts to track per assessment cycle.
        include_potential: Whether to include potential (not yet realised) impacts.
        include_actual: Whether to include actual (already occurring) impacts.
        value_chain_depth: Number of tiers in the value chain to assess.
        prioritisation_method: Method for prioritising impacts (SEVERITY_FIRST or LIKELIHOOD_FIRST).
    """

    severity_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "scale": Decimal("0.40"),
            "scope": Decimal("0.35"),
            "irremediability": Decimal("0.25"),
        },
        description="Weights for severity dimensions (must sum to 1.0)",
    )
    likelihood_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "very_likely": Decimal("1.0"),
            "likely": Decimal("0.75"),
            "possible": Decimal("0.50"),
            "unlikely": Decimal("0.25"),
            "rare": Decimal("0.10"),
        },
        description="Likelihood level weights for risk matrix calculation",
    )
    risk_matrix_thresholds: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "critical": Decimal("0.80"),
            "high": Decimal("0.60"),
            "medium": Decimal("0.35"),
            "low": Decimal("0.0"),
        },
        description="Score thresholds for severity classification",
    )
    max_impacts: int = Field(
        500, ge=1, le=10000,
        description="Maximum number of adverse impacts to track per assessment",
    )
    include_potential: bool = Field(
        True, description="Include potential (not yet realised) adverse impacts",
    )
    include_actual: bool = Field(
        True, description="Include actual (already occurring) adverse impacts",
    )
    value_chain_depth: int = Field(
        3, ge=1, le=10,
        description="Number of value chain tiers to assess (Art. 6(1))",
    )
    prioritisation_method: str = Field(
        "SEVERITY_FIRST",
        description="Prioritisation: SEVERITY_FIRST (default) or LIKELIHOOD_FIRST",
    )

    @field_validator("prioritisation_method")
    @classmethod
    def validate_prioritisation(cls, v: str) -> str:
        """Ensure prioritisation method is valid."""
        allowed = {"SEVERITY_FIRST", "LIKELIHOOD_FIRST"}
        if v.upper() not in allowed:
            raise ValueError(f"prioritisation_method must be one of {sorted(allowed)}, got: {v}")
        return v.upper()

    @model_validator(mode="after")
    def validate_severity_weights_sum(self) -> "ImpactConfig":
        """Ensure severity weights sum to 1.0."""
        total = sum(self.severity_weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(
                f"severity_weights must sum to 1.0, got {total}. "
                f"Current weights: {self.severity_weights}"
            )
        return self


class PreventionConfig(BaseModel):
    """Prevention and mitigation measure configuration per CSDDD Art. 8-11.

    Attributes:
        effectiveness_threshold: Minimum effectiveness score (0-100) for a
            measure to be considered adequate (default 60).
        budget_warning_threshold: Budget utilisation percentage above which
            a warning is raised (default 85%).
        overdue_tolerance_days: Days past deadline before a measure is flagged
            as overdue (default 30).
        require_contractual_assurances: Whether contractual assurances are
            mandatory for direct business partners (Art. 9).
        verification_required: Whether third-party verification is required (Art. 10).
        industry_initiative_credit: Allow credit for verified industry initiatives.
        corrective_action_plan_days: Days to produce a corrective action plan
            when measures fail (default 90).
        cascading_requirements: Whether to cascade due diligence requirements
            to sub-contractors (Art. 9(3)).
    """

    effectiveness_threshold: Decimal = Field(
        Decimal("60"), ge=Decimal("0"), le=Decimal("100"),
        description="Minimum measure effectiveness score (0-100)",
    )
    budget_warning_threshold: Decimal = Field(
        Decimal("85"), ge=Decimal("0"), le=Decimal("100"),
        description="Budget utilisation percentage triggering warning",
    )
    overdue_tolerance_days: int = Field(
        30, ge=0, le=365,
        description="Days past deadline before measure is flagged overdue",
    )
    require_contractual_assurances: bool = Field(
        True,
        description="Require contractual assurances from direct partners (Art. 9)",
    )
    verification_required: bool = Field(
        True,
        description="Require third-party verification of measures (Art. 10)",
    )
    industry_initiative_credit: bool = Field(
        True,
        description="Allow credit for verified industry/multi-stakeholder initiatives",
    )
    corrective_action_plan_days: int = Field(
        90, ge=1, le=365,
        description="Days allowed to produce corrective action plan on failure",
    )
    cascading_requirements: bool = Field(
        True,
        description="Cascade due diligence requirements to sub-contractors (Art. 9(3))",
    )


class GrievanceConfig(BaseModel):
    """Grievance mechanism (notification mechanism) per CSDDD Art. 14.

    Companies must provide or participate in effective notification mechanisms
    enabling any person to submit complaints regarding adverse impacts.

    Attributes:
        channels: Active grievance channels.
        response_time_target_days: Target days for initial acknowledgement.
        resolution_target_days: Target days for full resolution.
        anonymous_allowed: Whether anonymous complaints are accepted.
        languages: Languages supported by the grievance mechanism.
        escalation_tiers: Number of internal escalation levels.
        external_mechanism_accepted: Whether external mechanisms (e.g. NCP)
            satisfy the requirement.
        whistleblower_protection: Enable Directive (EU) 2019/1937 integration.
        max_open_complaints: Maximum open complaints before capacity warning.
    """

    channels: List[GrievanceChannel] = Field(
        default_factory=lambda: [
            GrievanceChannel.HOTLINE,
            GrievanceChannel.EMAIL,
            GrievanceChannel.WEB_PORTAL,
        ],
        description="Active grievance channels (Art. 14(1))",
    )
    response_time_target_days: int = Field(
        30, ge=1, le=180,
        description="Target days for initial response/acknowledgement",
    )
    resolution_target_days: int = Field(
        90, ge=1, le=365,
        description="Target days for complaint resolution",
    )
    anonymous_allowed: bool = Field(
        True,
        description="Accept anonymous complaints (recommended per UNGP 31)",
    )
    languages: List[str] = Field(
        default_factory=lambda: ["en", "de", "fr", "es"],
        description="ISO 639-1 language codes for grievance mechanism",
    )
    escalation_tiers: int = Field(
        3, ge=1, le=5,
        description="Number of internal escalation levels",
    )
    external_mechanism_accepted: bool = Field(
        True,
        description="Accept external mechanisms (OECD NCP, industry bodies) per Art. 14(2)",
    )
    whistleblower_protection: bool = Field(
        True,
        description="Enable Whistleblower Protection Directive integration (Art. 28)",
    )
    max_open_complaints: int = Field(
        1000, ge=1, le=100000,
        description="Maximum open complaints before capacity warning",
    )

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Ensure all language codes are 2-letter ISO 639-1."""
        for lang in v:
            if len(lang) != 2 or not lang.isalpha():
                raise ValueError(f"Invalid ISO 639-1 language code: {lang}")
        return [lang.lower() for lang in v]


class ClimateConfig(BaseModel):
    """Climate transition plan configuration per CSDDD Art. 22.

    Companies must adopt and put into effect a transition plan for climate
    change mitigation which aims to ensure that the business model and
    strategy are compatible with the Paris Agreement 1.5C target.

    Attributes:
        sbti_15c_annual_reduction: SBTi 1.5C annual linear reduction rate.
        sbti_2c_annual_reduction: SBTi well-below 2C annual reduction rate.
        target_years: Target years for emission reduction milestones.
        scope_coverage: GHG Protocol scopes included in transition plan.
        base_year: Base year for emission reduction targets.
        interim_targets_required: Whether interim targets are mandatory.
        paris_alignment_method: Method for assessing Paris alignment.
        include_scope_3: Whether Scope 3 emissions are included.
        decarbonisation_levers: Categories of decarbonisation actions.
        financial_planning_required: Whether financial resource allocation
            for transition is required in the plan.
    """

    sbti_15c_annual_reduction: Decimal = Field(
        Decimal("4.2"),
        description="SBTi 1.5C pathway annual linear reduction rate (%)",
    )
    sbti_2c_annual_reduction: Decimal = Field(
        Decimal("2.5"),
        description="SBTi well-below 2C pathway annual reduction rate (%)",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2045, 2050],
        description="Milestone years for emission reduction targets",
    )
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
        description="GHG Protocol scopes included in transition plan",
    )
    base_year: int = Field(
        2019, ge=2015, le=2025,
        description="Base year for emission reduction targets",
    )
    interim_targets_required: bool = Field(
        True,
        description="Whether interim (e.g. 2030) targets must be set",
    )
    paris_alignment_method: str = Field(
        "SDA",
        description="Paris alignment assessment method: SDA, ACA, or GEVA",
    )
    include_scope_3: bool = Field(
        True,
        description="Include Scope 3 value chain emissions in plan",
    )
    decarbonisation_levers: List[str] = Field(
        default_factory=lambda: [
            "energy_efficiency",
            "renewable_energy",
            "electrification",
            "process_innovation",
            "supply_chain_engagement",
            "circular_economy",
        ],
        description="Categories of decarbonisation actions in transition plan",
    )
    financial_planning_required: bool = Field(
        True,
        description="Require financial resource allocation for transition (Art. 22(1)(d))",
    )

    @field_validator("paris_alignment_method")
    @classmethod
    def validate_alignment_method(cls, v: str) -> str:
        """Ensure Paris alignment method is valid."""
        allowed = {"SDA", "ACA", "GEVA"}
        if v.upper() not in allowed:
            raise ValueError(f"paris_alignment_method must be one of {sorted(allowed)}, got: {v}")
        return v.upper()

    @field_validator("scope_coverage")
    @classmethod
    def validate_scope_coverage(cls, v: List[str]) -> List[str]:
        """Ensure all scope values are valid GHG Protocol scopes."""
        allowed = {"scope_1", "scope_2", "scope_3"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid scope(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")
        return v


class LiabilityConfig(BaseModel):
    """Civil liability configuration per CSDDD Art. 26.

    Member States must ensure that companies are liable for damages caused
    by failure to comply with their due diligence obligations.

    Attributes:
        limitation_period_default: Default limitation period in years.
        turnover_penalty_cap_pct: Maximum penalty as percentage of net turnover.
        insurance_coverage_target: Target D&O/liability insurance coverage in EUR.
        joint_liability_enabled: Whether joint liability with subsidiaries applies.
        burden_of_proof: Who bears the burden of proof (CLAIMANT or COMPANY).
        injunctive_relief_enabled: Whether injunctive measures are available.
        legal_aid_provisions: Whether legal aid is available to claimants.
        representative_actions: Whether representative (class) actions are enabled.
    """

    limitation_period_default: int = Field(
        5, ge=1, le=30,
        description="Limitation period for civil claims in years (Art. 26(5))",
    )
    turnover_penalty_cap_pct: Decimal = Field(
        Decimal("5.0"), ge=Decimal("0"), le=Decimal("100"),
        description="Maximum administrative penalty as % of net worldwide turnover (Art. 20(3))",
    )
    insurance_coverage_target: Decimal = Field(
        Decimal("50000000"),
        description="Target D&O/liability insurance coverage in EUR",
    )
    joint_liability_enabled: bool = Field(
        True,
        description="Joint liability with subsidiaries for failure to prevent (Art. 26(1))",
    )
    burden_of_proof: str = Field(
        "COMPANY",
        description="Burden of proof allocation: CLAIMANT or COMPANY (Art. 26(3))",
    )
    injunctive_relief_enabled: bool = Field(
        True,
        description="Injunctive relief (interim measures) available (Art. 26(2))",
    )
    legal_aid_provisions: bool = Field(
        True,
        description="Legal aid available to claimants (Art. 26(6))",
    )
    representative_actions: bool = Field(
        True,
        description="Representative (collective) actions enabled (Art. 26(7))",
    )

    @field_validator("burden_of_proof")
    @classmethod
    def validate_burden_of_proof(cls, v: str) -> str:
        """Ensure burden of proof setting is valid."""
        allowed = {"CLAIMANT", "COMPANY"}
        if v.upper() not in allowed:
            raise ValueError(f"burden_of_proof must be one of {sorted(allowed)}, got: {v}")
        return v.upper()


class StakeholderConfig(BaseModel):
    """Stakeholder engagement configuration per CSDDD Art. 13.

    Companies must carry out meaningful engagement with stakeholders where
    relevant to the identification and assessment of adverse impacts and
    to the design and implementation of appropriate measures.

    Attributes:
        required_groups: Stakeholder groups that must be engaged.
        min_engagement_frequency: Minimum engagements per year per group.
        meaningfulness_threshold: Score (0-100) for engagement to be deemed meaningful.
        documentation_required: Whether engagement activities must be documented.
        free_prior_informed_consent: Whether FPIC is required for indigenous peoples.
        accessibility_standards: Whether accessible formats must be provided.
        feedback_loop_required: Whether feedback on outcomes must be communicated.
        digital_engagement_enabled: Whether digital/online engagement is permitted.
    """

    required_groups: List[StakeholderGroup] = Field(
        default_factory=lambda: [
            StakeholderGroup.WORKERS,
            StakeholderGroup.TRADE_UNIONS,
            StakeholderGroup.COMMUNITIES,
        ],
        description="Stakeholder groups that must be engaged (Art. 13(1))",
    )
    min_engagement_frequency: int = Field(
        2, ge=1, le=12,
        description="Minimum engagement sessions per year per stakeholder group",
    )
    meaningfulness_threshold: Decimal = Field(
        Decimal("60"), ge=Decimal("0"), le=Decimal("100"),
        description="Score (0-100) above which engagement is deemed meaningful",
    )
    documentation_required: bool = Field(
        True,
        description="Require documentation of all engagement activities",
    )
    free_prior_informed_consent: bool = Field(
        True,
        description="Require FPIC for indigenous peoples (ILO Convention 169)",
    )
    accessibility_standards: bool = Field(
        True,
        description="Provide accessible formats for persons with disabilities",
    )
    feedback_loop_required: bool = Field(
        True,
        description="Communicate outcomes of engagement back to stakeholders",
    )
    digital_engagement_enabled: bool = Field(
        True,
        description="Permit digital/online engagement alongside in-person",
    )


class ReportingConfig(BaseModel):
    """Reporting and communication configuration per CSDDD Art. 16.

    Companies must publish an annual statement on their due diligence
    policies, identified impacts, measures taken, and their effectiveness.

    Attributes:
        format: Default output report format.
        include_executive_summary: Whether to include an executive summary.
        include_recommendations: Whether to include corrective action recommendations.
        max_report_pages: Maximum page count for PDF reports.
        annual_statement_required: Whether an annual public statement is required.
        csrd_cross_reference: Whether to cross-reference CSRD/ESRS disclosures.
        reporting_language: ISO 639-1 language code for reports.
        data_retention_years: Years to retain reporting data.
        xbrl_tagging_enabled: Whether XBRL digital tagging is enabled.
        assurance_level: Level of assurance on due diligence reporting.
    """

    format: ReportFormat = Field(
        ReportFormat.PDF,
        description="Default output format for compliance reports",
    )
    include_executive_summary: bool = Field(
        True, description="Include executive summary in reports",
    )
    include_recommendations: bool = Field(
        True, description="Include corrective action recommendations",
    )
    max_report_pages: int = Field(
        200, ge=10, le=1000,
        description="Maximum page count for PDF reports",
    )
    annual_statement_required: bool = Field(
        True,
        description="Require annual public statement per Art. 16",
    )
    csrd_cross_reference: bool = Field(
        True,
        description="Cross-reference CSRD/ESRS disclosures (ESRS S1-S4, G1)",
    )
    reporting_language: str = Field(
        "en", description="ISO 639-1 language code for reports",
    )
    data_retention_years: int = Field(
        7, ge=1, le=20,
        description="Years to retain due diligence and reporting data",
    )
    xbrl_tagging_enabled: bool = Field(
        False,
        description="Enable XBRL digital tagging for machine-readable reports",
    )
    assurance_level: str = Field(
        "LIMITED",
        description="Assurance level: NONE, LIMITED, or REASONABLE",
    )

    @field_validator("reporting_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure reporting language is a 2-letter ISO 639-1 code."""
        if len(v) != 2 or not v.isalpha():
            raise ValueError(f"reporting_language must be 2-letter ISO 639-1 code, got: {v}")
        return v.lower()

    @field_validator("assurance_level")
    @classmethod
    def validate_assurance(cls, v: str) -> str:
        """Ensure assurance level is valid."""
        allowed = {"NONE", "LIMITED", "REASONABLE"}
        if v.upper() not in allowed:
            raise ValueError(f"assurance_level must be one of {sorted(allowed)}, got: {v}")
        return v.upper()


class CacheConfig(BaseModel):
    """Cache configuration for engine lookups and computation results.

    Attributes:
        backend: Cache backend type (MEMORY, REDIS, or DISABLED).
        ttl_seconds: Time-to-live for cache entries in seconds.
        max_entries: Maximum number of cache entries before eviction.
        redis_url: Redis connection URL (only used when backend is REDIS).
        key_prefix: Prefix for all cache keys to avoid collisions.
    """

    backend: CacheBackend = Field(
        CacheBackend.MEMORY,
        description="Cache backend: MEMORY (default), REDIS, or DISABLED",
    )
    ttl_seconds: int = Field(
        3600, ge=0, le=86400,
        description="Cache entry TTL in seconds (0 = no expiry within session)",
    )
    max_entries: int = Field(
        10000, ge=100, le=1000000,
        description="Maximum cache entries before LRU eviction",
    )
    redis_url: Optional[str] = Field(
        None,
        description="Redis connection URL (redis://host:port/db)",
    )
    key_prefix: str = Field(
        "csddd_pack_019",
        description="Cache key prefix for namespace isolation",
    )


# =============================================================================
# Core Article Requirements Mapping
# =============================================================================

CORE_ARTICLE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "ART_5": {
        "title": "Due Diligence Policy Integration",
        "article": ArticleReference.ART_5,
        "oecd_step": OECDStep.STEP_1_EMBED,
        "mandatory": True,
        "requires": [
            "Written due diligence policy approved by board",
            "Description of approach to human rights and environmental due diligence",
            "Code of conduct applicable to employees and subsidiaries",
            "Description of processes to implement due diligence",
            "Annual policy update process",
        ],
        "cross_reference_esrs": ["ESRS S1-1", "ESRS S2-1", "ESRS S3-1", "ESRS S4-1", "ESRS G1-1"],
    },
    "ART_6": {
        "title": "Impact Identification and Assessment",
        "article": ArticleReference.ART_6,
        "oecd_step": OECDStep.STEP_2_IDENTIFY,
        "mandatory": True,
        "requires": [
            "Map own operations for actual and potential adverse impacts",
            "Map subsidiaries' operations for adverse impacts",
            "Map business partners' operations in chains of activities",
            "Identify and assess severity based on scale, scope, irremediability",
            "Engage with stakeholders in identification process",
        ],
        "cross_reference_esrs": ["ESRS 2 IRO-1", "ESRS S1-2", "ESRS S2-2"],
    },
    "ART_7": {
        "title": "Impact Prioritisation",
        "article": ArticleReference.ART_7,
        "oecd_step": OECDStep.STEP_2_IDENTIFY,
        "mandatory": True,
        "requires": [
            "Prioritise impacts based on severity and likelihood",
            "Address most severe and likely impacts first",
            "Document prioritisation methodology",
            "Review and update priorities periodically",
        ],
        "cross_reference_esrs": ["ESRS 2 IRO-1", "ESRS 1 para 38-40"],
    },
    "ART_8": {
        "title": "Prevention of Potential Impacts",
        "article": ArticleReference.ART_8,
        "oecd_step": OECDStep.STEP_3_PREVENT,
        "mandatory": True,
        "requires": [
            "Develop and implement prevention action plan with timelines",
            "Seek contractual assurances from direct business partners",
            "Make necessary investments for prevention",
            "Provide targeted and proportionate support to SME partners",
            "Collaborate with other entities where appropriate",
        ],
        "cross_reference_esrs": ["ESRS S1-4", "ESRS S2-4", "ESRS S3-4", "ESRS S4-4"],
    },
    "ART_9": {
        "title": "Contractual Assurances",
        "article": ArticleReference.ART_9,
        "oecd_step": OECDStep.STEP_3_PREVENT,
        "mandatory": True,
        "requires": [
            "Obtain contractual assurances from direct business partners",
            "Include cascading clauses for indirect partners",
            "Partner compliance with company code of conduct",
            "Prevention action plan as contract requirement",
            "Verification rights in contractual clauses",
        ],
        "cross_reference_esrs": ["ESRS S2-1", "ESRS G1-2"],
    },
    "ART_10": {
        "title": "Verification Measures",
        "article": ArticleReference.ART_10,
        "oecd_step": OECDStep.STEP_3_PREVENT,
        "mandatory": True,
        "requires": [
            "Verify partner compliance through audits or assessments",
            "Engage independent third-party verification where appropriate",
            "Use industry schemes for verification where available",
            "Document verification findings and follow-up actions",
        ],
        "cross_reference_esrs": ["ESRS S2-4", "ESRS G1-4"],
    },
    "ART_11": {
        "title": "Mitigation of Actual Impacts",
        "article": ArticleReference.ART_11,
        "oecd_step": OECDStep.STEP_3_PREVENT,
        "mandatory": True,
        "requires": [
            "Take appropriate measures to bring actual impacts to an end",
            "Develop corrective action plan with timelines and benchmarks",
            "Seek contractual assurances for mitigation",
            "Suspend or terminate business relationship as last resort",
            "Provide support for mitigation to SME partners",
        ],
        "cross_reference_esrs": ["ESRS S1-4", "ESRS S2-4", "ESRS S3-4"],
    },
    "ART_12": {
        "title": "Remediation",
        "article": ArticleReference.ART_12,
        "oecd_step": OECDStep.STEP_6_REMEDIATE,
        "mandatory": True,
        "requires": [
            "Provide remediation where company caused or contributed to impact",
            "Enable remediation through legitimate processes",
            "Cooperate with state-based judicial/non-judicial mechanisms",
            "Remediation proportionate to significance of adverse impact",
            "Financial compensation where appropriate",
        ],
        "cross_reference_esrs": ["ESRS S1-4", "ESRS S2-4", "ESRS S3-4"],
    },
    "ART_13": {
        "title": "Meaningful Stakeholder Engagement",
        "article": ArticleReference.ART_13,
        "oecd_step": OECDStep.STEP_2_IDENTIFY,
        "mandatory": True,
        "requires": [
            "Engage with affected stakeholders during impact identification",
            "Consult during prevention and mitigation measure design",
            "Provide relevant information to stakeholders",
            "Ensure engagement is timely, accessible, and culturally appropriate",
            "Ensure free prior and informed consent for indigenous peoples",
        ],
        "cross_reference_esrs": ["ESRS S1-2", "ESRS S2-2", "ESRS S3-2", "ESRS S4-2"],
    },
    "ART_14": {
        "title": "Notification Mechanism (Grievance)",
        "article": ArticleReference.ART_14,
        "oecd_step": OECDStep.STEP_6_REMEDIATE,
        "mandatory": True,
        "requires": [
            "Establish or participate in a notification mechanism",
            "Accessible to persons and organisations with legitimate concern",
            "Fair, transparent, and rights-compatible procedures",
            "Protection against retaliation for complainants",
            "Inform about process and expected timelines",
        ],
        "cross_reference_esrs": ["ESRS S1-3", "ESRS S2-3", "ESRS S3-3", "ESRS S4-3"],
    },
    "ART_15": {
        "title": "Monitoring",
        "article": ArticleReference.ART_15,
        "oecd_step": OECDStep.STEP_4_TRACK,
        "mandatory": True,
        "requires": [
            "Carry out periodic assessments of own operations",
            "Assess subsidiaries and business partner operations",
            "Monitor effectiveness of prevention and mitigation measures",
            "Conduct assessments at least every 12 months",
            "Update due diligence measures based on assessment results",
        ],
        "cross_reference_esrs": ["ESRS 2 IRO-1", "ESRS S1-5", "ESRS S2-5"],
    },
    "ART_16": {
        "title": "Reporting and Communication",
        "article": ArticleReference.ART_16,
        "oecd_step": OECDStep.STEP_5_COMMUNICATE,
        "mandatory": True,
        "requires": [
            "Publish annual statement on due diligence on company website",
            "Report identified actual and potential adverse impacts",
            "Report measures taken and their effectiveness",
            "Companies subject to CSRD may fulfil through sustainability statement",
            "Plain language, accessible to affected stakeholders",
        ],
        "cross_reference_esrs": ["ESRS 2 SBM-3", "ESRS S1-1 to S1-17"],
    },
    "ART_22": {
        "title": "Climate Transition Plan",
        "article": ArticleReference.ART_22,
        "oecd_step": OECDStep.STEP_1_EMBED,
        "mandatory": True,
        "requires": [
            "Adopt transition plan for climate change mitigation",
            "Time-bound targets for 2030 and 5-year steps to 2050",
            "Ensure compatibility with Paris Agreement 1.5C target",
            "Include Scope 1, 2, and where relevant Scope 3 emissions",
            "Describe decarbonisation levers and investment plan",
            "Report on implementation progress annually",
        ],
        "cross_reference_esrs": ["ESRS E1-1", "ESRS E1-4", "ESRS E1-6"],
    },
    "ART_26": {
        "title": "Civil Liability",
        "article": ArticleReference.ART_26,
        "oecd_step": OECDStep.STEP_6_REMEDIATE,
        "mandatory": True,
        "requires": [
            "Company liable for damages from failure to comply with Art. 8 and 11",
            "Full compensation for damage including lost profits",
            "Limitation period not less than 5 years from date of knowledge",
            "Access to evidence available to company",
            "Injunctive relief (interim measures) available to claimants",
            "Representative actions permitted under Directive (EU) 2020/1828",
        ],
        "cross_reference_esrs": [],
    },
}


# =============================================================================
# PackConfig - Top-Level Configuration
# =============================================================================


class PackConfig(BaseModel):
    """Top-level configuration for PACK-019 CSDDD Readiness Pack.

    Aggregates all sub-configurations and provides factory methods for
    loading from presets, YAML files, and environment overrides.

    Attributes:
        pack_name: Immutable pack identifier.
        version: Configuration schema version.
        company_scope: Phase classification for the company.
        sector: Target business sector.
        enabled_engines: Active engine list (subset of ALL_ENGINES).
        enabled_workflows: Active workflow list (subset of ALL_WORKFLOWS).
        scope_config: Company scope determination thresholds.
        impact_config: Adverse impact identification settings.
        prevention_config: Prevention and mitigation measure settings.
        grievance_config: Grievance mechanism settings.
        climate_config: Climate transition plan settings.
        liability_config: Civil liability settings.
        stakeholder_config: Stakeholder engagement settings.
        reporting_config: Reporting and communication settings.
        cache_config: Engine caching settings.

    Example:
        >>> config = PackConfig()
        >>> warnings = config.validate_thresholds()
        >>> print(len(warnings))
        0
        >>> print(config.config_hash[:16])
    """

    pack_name: str = Field(
        "PACK-019-csddd-readiness",
        description="Pack identifier",
    )
    version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    company_scope: CompanyScope = Field(
        CompanyScope.PHASE_1,
        description="Company scope phase classification (Art. 2)",
    )
    sector: SectorType = Field(
        SectorType.MANUFACTURING,
        description="Target business sector for sector-specific guidance",
    )
    enabled_engines: List[str] = Field(
        default_factory=lambda: list(ALL_ENGINES),
        description="Engines to activate (subset of ALL_ENGINES)",
    )
    enabled_workflows: List[str] = Field(
        default_factory=lambda: list(ALL_WORKFLOWS),
        description="Workflows to activate (subset of ALL_WORKFLOWS)",
    )
    scope_config: ScopeConfig = Field(
        default_factory=ScopeConfig,
        description="Company scope determination thresholds",
    )
    impact_config: ImpactConfig = Field(
        default_factory=ImpactConfig,
        description="Adverse impact identification and prioritisation settings",
    )
    prevention_config: PreventionConfig = Field(
        default_factory=PreventionConfig,
        description="Prevention and mitigation measure settings",
    )
    grievance_config: GrievanceConfig = Field(
        default_factory=GrievanceConfig,
        description="Grievance mechanism settings (Art. 14)",
    )
    climate_config: ClimateConfig = Field(
        default_factory=ClimateConfig,
        description="Climate transition plan settings (Art. 22)",
    )
    liability_config: LiabilityConfig = Field(
        default_factory=LiabilityConfig,
        description="Civil liability settings (Art. 26)",
    )
    stakeholder_config: StakeholderConfig = Field(
        default_factory=StakeholderConfig,
        description="Stakeholder engagement settings (Art. 13)",
    )
    reporting_config: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Reporting and communication settings (Art. 16)",
    )
    cache_config: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Engine caching configuration",
    )

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------

    @property
    def config_hash(self) -> str:
        """SHA-256 hash of serialised configuration for provenance tracking."""
        serialised = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, default=str,
        )
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    @property
    def application_date(self) -> Optional[date]:
        """Return the CSDDD application date for the configured scope phase."""
        return self.scope_config.get_application_date(self.company_scope)

    @property
    def is_in_scope(self) -> bool:
        """Return whether the company is in scope for CSDDD obligations."""
        return self.company_scope not in (
            CompanyScope.NOT_IN_SCOPE,
            CompanyScope.VOLUNTARY,
        )

    @property
    def core_articles(self) -> List[ArticleReference]:
        """Return the list of core CSDDD articles applicable to all in-scope companies."""
        return [
            ArticleReference.ART_5,
            ArticleReference.ART_6,
            ArticleReference.ART_7,
            ArticleReference.ART_8,
            ArticleReference.ART_9,
            ArticleReference.ART_10,
            ArticleReference.ART_11,
            ArticleReference.ART_12,
            ArticleReference.ART_13,
            ArticleReference.ART_14,
            ArticleReference.ART_15,
            ArticleReference.ART_16,
            ArticleReference.ART_22,
            ArticleReference.ART_26,
        ]

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("enabled_engines")
    @classmethod
    def validate_engines(cls, v: List[str]) -> List[str]:
        """Ensure every enabled engine is a recognised engine name."""
        invalid = set(v) - set(ALL_ENGINES)
        if invalid:
            raise ValueError(f"Unknown engine(s): {sorted(invalid)}. Valid: {ALL_ENGINES}")
        return v

    @field_validator("enabled_workflows")
    @classmethod
    def validate_workflows(cls, v: List[str]) -> List[str]:
        """Ensure every enabled workflow is a recognised workflow name."""
        invalid = set(v) - set(ALL_WORKFLOWS)
        if invalid:
            raise ValueError(f"Unknown workflow(s): {sorted(invalid)}. Valid: {ALL_WORKFLOWS}")
        return v

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def validate_thresholds(self) -> List[str]:
        """Run comprehensive validation across all sub-configs.

        Checks internal consistency, regulatory minimums, and configuration
        coherence. Returns a list of warning strings; empty means fully valid.

        Returns:
            List of warning strings. Empty list indicates no issues.
        """
        warnings: List[str] = []

        # Engine validation
        if not self.enabled_engines:
            warnings.append("No engines enabled. At minimum scope_assessment and impact_identification are required.")
        required_engines = {"scope_assessment", "impact_identification"}
        missing_engines = required_engines - set(self.enabled_engines)
        if missing_engines:
            warnings.append(
                f"Missing required engine(s): {sorted(missing_engines)}. "
                "Scope assessment and impact identification are mandatory (Art. 2, Art. 6)."
            )
        if "climate_transition" not in self.enabled_engines:
            warnings.append("Climate transition engine disabled. Art. 22 requires a transition plan.")
        if "grievance_management" not in self.enabled_engines:
            warnings.append("Grievance management engine disabled. Art. 14 requires a notification mechanism.")

        # Scope validation
        if self.company_scope == CompanyScope.NOT_IN_SCOPE:
            warnings.append(
                "Company is NOT_IN_SCOPE. CSDDD obligations do not apply unless VOLUNTARY."
            )

        # Impact configuration validation
        if self.impact_config.max_impacts < 10:
            warnings.append(
                f"max_impacts={self.impact_config.max_impacts} is very low. "
                "Consider at least 50 for comprehensive due diligence."
            )
        if not self.impact_config.include_potential:
            warnings.append(
                "Potential impacts excluded. Art. 6 requires identifying both actual and potential impacts."
            )
        if not self.impact_config.include_actual:
            warnings.append(
                "Actual impacts excluded. Art. 6 requires identifying both actual and potential impacts."
            )

        # Prevention configuration validation
        if self.prevention_config.effectiveness_threshold < Decimal("40"):
            warnings.append(
                f"Prevention effectiveness_threshold={self.prevention_config.effectiveness_threshold} "
                "is below recommended minimum of 40."
            )
        if not self.prevention_config.require_contractual_assurances:
            warnings.append(
                "Contractual assurances disabled. Art. 9 requires them for direct business partners."
            )

        # Grievance configuration validation
        if not self.grievance_config.channels:
            warnings.append("No grievance channels configured. Art. 14 requires at least one channel.")
        if self.grievance_config.response_time_target_days > 60:
            warnings.append(
                f"Grievance response target {self.grievance_config.response_time_target_days} days "
                "exceeds recommended 30-60 day range."
            )
        if self.grievance_config.resolution_target_days > 180:
            warnings.append(
                f"Grievance resolution target {self.grievance_config.resolution_target_days} days "
                "exceeds recommended 90-180 day range."
            )

        # Climate configuration validation
        if not self.climate_config.include_scope_3:
            warnings.append(
                "Scope 3 excluded from climate transition plan. Art. 22 requires 'where relevant' "
                "inclusion of value chain emissions."
            )
        if 2030 not in self.climate_config.target_years:
            warnings.append("2030 not in target years. CSDDD Art. 22 requires 2030 interim targets.")
        if 2050 not in self.climate_config.target_years:
            warnings.append("2050 not in target years. Paris Agreement requires net-zero by 2050.")

        # Liability configuration validation
        if self.liability_config.limitation_period_default < 5:
            warnings.append(
                f"Limitation period {self.liability_config.limitation_period_default} years "
                "is below the Art. 26(5) minimum of 5 years."
            )

        # Stakeholder configuration validation
        required_groups = {StakeholderGroup.WORKERS, StakeholderGroup.TRADE_UNIONS}
        missing_groups = required_groups - set(self.stakeholder_config.required_groups)
        if missing_groups:
            warnings.append(
                f"Missing essential stakeholder groups: {[g.value for g in missing_groups]}. "
                "Workers and trade unions must be engaged (Art. 13)."
            )

        # Reporting validation
        if not self.reporting_config.annual_statement_required:
            warnings.append(
                "Annual statement disabled. Art. 16 requires annual public due diligence reporting."
            )

        return warnings

    def get_article_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Return all CSDDD article requirements applicable to this configuration.

        Returns:
            Dictionary mapping article keys to their requirement definitions,
            filtered by company scope and sector relevance.
        """
        if not self.is_in_scope and self.company_scope != CompanyScope.VOLUNTARY:
            logger.info("Company is not in scope. Returning empty article requirements.")
            return {}

        requirements = dict(CORE_ARTICLE_REQUIREMENTS)

        # Add scope-phase-specific notes
        for key, req in requirements.items():
            req["company_scope"] = self.company_scope.value
            req["sector"] = self.sector.value
            req["application_date"] = str(self.application_date) if self.application_date else None

        return requirements

    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Return configuration scoped to a specific engine.

        Args:
            engine_name: Must be in ALL_ENGINES.

        Returns:
            Dict with engine name, enabled status, and relevant pack parameters.

        Raises:
            ValueError: If engine_name is not recognised.
        """
        if engine_name not in ALL_ENGINES:
            raise ValueError(f"Unknown engine: {engine_name}. Valid: {ALL_ENGINES}")

        base = {
            "engine": engine_name,
            "enabled": engine_name in self.enabled_engines,
            "sector": self.sector.value,
            "company_scope": self.company_scope.value,
            "config_hash": self.config_hash,
        }

        engine_specific: Dict[str, Dict[str, Any]] = {
            "scope_assessment": {
                "scope_config": self.scope_config.model_dump(mode="json"),
            },
            "impact_identification": {
                "impact_config": self.impact_config.model_dump(mode="json"),
                "value_chain_depth": self.impact_config.value_chain_depth,
            },
            "prevention_planning": {
                "prevention_config": self.prevention_config.model_dump(mode="json"),
            },
            "grievance_management": {
                "grievance_config": self.grievance_config.model_dump(mode="json"),
            },
            "climate_transition": {
                "climate_config": self.climate_config.model_dump(mode="json"),
            },
            "liability_assessment": {
                "liability_config": self.liability_config.model_dump(mode="json"),
            },
            "stakeholder_engagement": {
                "stakeholder_config": self.stakeholder_config.model_dump(mode="json"),
            },
            "remediation_tracking": {
                "prevention_config": self.prevention_config.model_dump(mode="json"),
                "grievance_config": self.grievance_config.model_dump(mode="json"),
            },
        }

        base.update(engine_specific.get(engine_name, {}))
        return base

    def get_workflow_config(self) -> Dict[str, Any]:
        """Return workflow orchestration configuration.

        Returns:
            Dict mapping each workflow to enabled status plus shared parameters.
        """
        return {
            "workflows": {wf: (wf in self.enabled_workflows) for wf in ALL_WORKFLOWS},
            "company_scope": self.company_scope.value,
            "sector": self.sector.value,
            "application_date": str(self.application_date) if self.application_date else None,
            "reporting_language": self.reporting_config.reporting_language,
            "report_format": self.reporting_config.format.value,
            "config_hash": self.config_hash,
        }

    def get_oecd_step_mapping(self) -> Dict[str, List[str]]:
        """Map each OECD step to its corresponding CSDDD articles.

        Returns:
            Dict mapping OECDStep values to lists of ArticleReference values.
        """
        mapping: Dict[str, List[str]] = {step.value: [] for step in OECDStep}
        for article, step in ARTICLE_OECD_MAPPING.items():
            mapping[step.value].append(article.value)
        return mapping

    def get_compliance_checklist(self) -> List[Dict[str, Any]]:
        """Generate a compliance checklist based on current configuration.

        Returns:
            List of checklist items with article, requirement, and readiness status.
        """
        checklist: List[Dict[str, Any]] = []
        requirements = self.get_article_requirements()

        for key, req in requirements.items():
            article = req["article"]
            for i, requirement_text in enumerate(req["requires"]):
                checklist.append({
                    "checklist_id": f"{key}-{i + 1:02d}",
                    "article": article.value,
                    "article_title": req["title"],
                    "requirement": requirement_text,
                    "oecd_step": req["oecd_step"].value,
                    "mandatory": req["mandatory"],
                    "status": ComplianceStatus.NON_COMPLIANT.value,
                    "esrs_cross_reference": req.get("cross_reference_esrs", []),
                })

        return checklist

    def to_dict(self) -> Dict[str, Any]:
        """Serialise full configuration to a plain dict with config_hash."""
        data = self.model_dump(mode="json")
        data["config_hash"] = self.config_hash
        data["application_date"] = str(self.application_date) if self.application_date else None
        data["is_in_scope"] = self.is_in_scope
        return data

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load PackConfig from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            raw = _deep_merge(raw, env_overrides)
        return cls(**raw)

    @classmethod
    def from_preset(cls, preset_name: str) -> "PackConfig":
        """Load PackConfig from a bundled sector preset.

        Loads the YAML preset file for the given sector and constructs
        a PackConfig with sector-appropriate defaults.

        Args:
            preset_name: One of AVAILABLE_PRESETS keys.

        Returns:
            PackConfig instance configured for the specified sector.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        return cls.from_yaml(preset_path)

    @classmethod
    def from_demo(cls) -> "PackConfig":
        """Load the demo configuration for testing and walkthroughs.

        Returns:
            PackConfig instance with demo company data.

        Raises:
            FileNotFoundError: If demo config YAML is missing.
        """
        demo_path = DEMO_DIR / "demo_config.yaml"
        if not demo_path.exists():
            raise FileNotFoundError(f"Demo config not found: {demo_path}")
        return cls.from_yaml(demo_path)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load overrides from CSDDD_PACK_* environment variables.

        Environment variables are mapped to config keys by stripping the
        CSDDD_PACK_ prefix and lowercasing. Boolean and integer values
        are auto-converted.

        Returns:
            Dict of overrides to merge into configuration.
        """
        overrides: Dict[str, Any] = {}
        prefix = "CSDDD_PACK_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            config_key = key[len(prefix):].lower()
            if value.lower() in ("true", "false"):
                overrides[config_key] = value.lower() == "true"
            elif value.isdigit():
                overrides[config_key] = int(value)
            else:
                overrides[config_key] = value
        return overrides

    def __repr__(self) -> str:
        """Return concise string representation of PackConfig."""
        return (
            f"PackConfig(pack={self.pack_name!r}, version={self.version!r}, "
            f"scope={self.company_scope.value}, sector={self.sector.value}, "
            f"engines={len(self.enabled_engines)}, workflows={len(self.enabled_workflows)})"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_all_articles() -> List[ArticleReference]:
    """Return all CSDDD article references.

    Returns:
        List of all ArticleReference enum members.
    """
    return list(ArticleReference)


def get_article_description(article: ArticleReference) -> str:
    """Return the description for a specific CSDDD article.

    Args:
        article: ArticleReference enum value.

    Returns:
        Human-readable description of the article.
    """
    return ARTICLE_DESCRIPTIONS.get(article, f"No description available for {article.value}")


def get_oecd_step_for_article(article: ArticleReference) -> Optional[OECDStep]:
    """Return the OECD step corresponding to a CSDDD article.

    Args:
        article: ArticleReference enum value.

    Returns:
        OECDStep enum value, or None if article has no OECD mapping.
    """
    return ARTICLE_OECD_MAPPING.get(article)


def get_phase_in_schedule() -> Dict[str, Dict[str, Any]]:
    """Return the complete CSDDD phase-in schedule.

    Returns:
        Dict mapping phase names to application criteria and dates.
    """
    return {
        "phase_1": {
            "application_date": "2027-07-26",
            "employee_threshold": 5000,
            "turnover_threshold_eur": 1_500_000_000,
            "description": "Largest companies: >5000 employees AND >EUR 1.5bn turnover",
        },
        "phase_2": {
            "application_date": "2028-07-26",
            "employee_threshold": 3000,
            "turnover_threshold_eur": 900_000_000,
            "description": "Large companies: >3000 employees AND >EUR 900m turnover",
        },
        "phase_3": {
            "application_date": "2029-07-26",
            "employee_threshold": 1000,
            "turnover_threshold_eur": 450_000_000,
            "description": "Medium-large companies: >1000 employees AND >EUR 450m turnover",
        },
    }


def get_annex_instruments() -> Dict[str, List[str]]:
    """Return the CSDDD Annex Part I and Part II international instruments.

    Part I covers human rights instruments; Part II covers environmental
    conventions referenced by the CSDDD for identifying adverse impacts.

    Returns:
        Dict with keys 'human_rights' and 'environmental', each a list of
        international instrument names.
    """
    return {
        "human_rights": [
            "Universal Declaration of Human Rights (1948)",
            "International Covenant on Civil and Political Rights (1966)",
            "International Covenant on Economic, Social and Cultural Rights (1966)",
            "Convention on the Prevention of Genocide (1948)",
            "Convention against Torture (1984)",
            "Convention on the Elimination of Racial Discrimination (1965)",
            "Convention on the Elimination of Discrimination against Women (1979)",
            "Convention on the Rights of the Child (1989)",
            "Convention on the Rights of Persons with Disabilities (2006)",
            "UN Declaration on the Rights of Indigenous Peoples (2007)",
            "ILO Convention 29 - Forced Labour (1930)",
            "ILO Convention 87 - Freedom of Association (1948)",
            "ILO Convention 98 - Right to Organise (1949)",
            "ILO Convention 100 - Equal Remuneration (1951)",
            "ILO Convention 105 - Abolition of Forced Labour (1957)",
            "ILO Convention 111 - Discrimination in Employment (1958)",
            "ILO Convention 138 - Minimum Age (1973)",
            "ILO Convention 182 - Worst Forms of Child Labour (1999)",
        ],
        "environmental": [
            "Stockholm Convention on Persistent Organic Pollutants (2001)",
            "Minamata Convention on Mercury (2013)",
            "Basel Convention on Hazardous Wastes (1989)",
            "Rotterdam Convention on Prior Informed Consent (1998)",
            "Convention on Biological Diversity (1992)",
            "Cartagena Protocol on Biosafety (2000)",
            "Nagoya Protocol on Access and Benefit Sharing (2010)",
            "CITES - Endangered Species Trade (1973)",
            "Paris Agreement on Climate Change (2015)",
            "UNFCCC - Climate Change (1992)",
            "London Convention on Dumping of Wastes at Sea (1972)",
            "MARPOL Convention on Ship Pollution (1973/1978)",
        ],
    }
