"""
TCFD Disclosure Generator -- Structured disclosure for all 11 TCFD recommended disclosures.

This module implements the ``DisclosureGenerator`` engine for GL-TCFD-APP v1.0.
It manages the full lifecycle of TCFD disclosure documents: creation, section
drafting for each of the 11 recommended disclosures, compliance scoring,
evidence linking, approval workflows, multi-format export (PDF, Excel, JSON,
XBRL), year-over-year comparison, and publication.

The 11 TCFD Recommended Disclosures:
    Governance:
        gov_a -- Board oversight of climate-related risks and opportunities
        gov_b -- Management's role in assessing and managing climate risks
    Strategy:
        str_a -- Climate-related risks and opportunities (short/medium/long term)
        str_b -- Impact on business, strategy, and financial planning
        str_c -- Resilience of strategy under climate scenarios (incl. 2C)
    Risk Management:
        rm_a  -- Processes for identifying and assessing climate-related risks
        rm_b  -- Processes for managing climate-related risks
        rm_c  -- Integration into overall risk management (ERM)
    Metrics & Targets:
        mt_a  -- Metrics used to assess climate risks and opportunities
        mt_b  -- Scope 1, 2, 3 GHG emissions
        mt_c  -- Targets and performance against targets

Regulatory Adaptations:
    The generator supports 8 regulatory regimes that impose additional or
    modified requirements on top of the core TCFD framework: UK FCA, EU CSRD,
    US SEC, Japan FSA, Singapore SGX, Hong Kong HKEX, Australia ASRS, and
    New Zealand XRB.

Reference:
    - TCFD Final Report (June 2017)
    - TCFD Annex: Implementing the Recommendations (June 2017)
    - TCFD Guidance on Scenario Analysis (October 2020)
    - IFRS S2 Climate-related Disclosures (June 2023)

Example:
    >>> from services.config import TCFDAppConfig
    >>> gen = DisclosureGenerator(TCFDAppConfig())
    >>> disc = gen.create_disclosure("org-1", "FY2025", "Annual TCFD Report")
    >>> section = gen.draft_section(disc.id, "gov_a", "Board oversees...", [])
    >>> score = gen.check_compliance(disc.id)
    >>> print(score.overall_score)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import (
    DisclosureStatus,
    TCFDAppConfig,
    TCFDPillar,
    TCFD_DISCLOSURES,
    TCFD_TO_IFRS_S2_MAPPING,
    PILLAR_NAMES,
)
from .models import (
    DisclosureSection,
    TCFDDisclosure,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template text for each of the 11 recommended disclosures
# ---------------------------------------------------------------------------

DISCLOSURE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "gov_a": {
        "title": "Board Oversight",
        "prompt": (
            "Describe the board's oversight of climate-related risks and "
            "opportunities, including the frequency of board reviews, "
            "dedicated committee structures, and how the board monitors "
            "and oversees progress against climate goals and targets."
        ),
        "guidance": (
            "Organizations should describe: (1) processes by which the board "
            "is informed about climate-related issues; (2) whether the board or "
            "a committee considers climate during strategy, budget, and business "
            "plans review; (3) how the board monitors and oversees progress."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "board_review_frequency",
            "committee_structure",
            "reporting_cadence",
            "oversight_scope",
        ],
    },
    "gov_b": {
        "title": "Management Role",
        "prompt": (
            "Describe management's role in assessing and managing climate-related "
            "risks and opportunities, including organizational structures, "
            "reporting lines, and how management is informed and monitors."
        ),
        "guidance": (
            "Organizations should describe: (1) whether management has assigned "
            "climate-related responsibilities; (2) associated organizational "
            "structures; (3) how management is informed; (4) how management "
            "monitors climate-related issues."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "management_roles",
            "organizational_structure",
            "reporting_process",
            "monitoring_mechanism",
        ],
    },
    "str_a": {
        "title": "Risks and Opportunities",
        "prompt": (
            "Describe the climate-related risks and opportunities the "
            "organization has identified over the short, medium, and long "
            "term, including specific physical and transition risks."
        ),
        "guidance": (
            "Organizations should provide: (1) description of significant "
            "risks/opportunities by time horizon; (2) specific climate-related "
            "issues per category; (3) processes used to determine materiality."
        ),
        "minimum_word_count": 300,
        "required_elements": [
            "short_term_risks",
            "medium_term_risks",
            "long_term_risks",
            "physical_risks",
            "transition_risks",
            "opportunities",
        ],
    },
    "str_b": {
        "title": "Business Impact",
        "prompt": (
            "Describe the impact of climate-related risks and opportunities "
            "on the organization's businesses, strategy, and financial planning."
        ),
        "guidance": (
            "Organizations should describe: (1) impact on businesses and "
            "strategy; (2) impact on financial planning; (3) specific climate "
            "impacts on products/services, supply chain, investment, and "
            "adaptation/mitigation activities."
        ),
        "minimum_word_count": 300,
        "required_elements": [
            "business_impact",
            "strategy_impact",
            "financial_planning_impact",
            "products_services_impact",
        ],
    },
    "str_c": {
        "title": "Scenario Analysis",
        "prompt": (
            "Describe the resilience of the organization's strategy, taking "
            "into consideration different climate-related scenarios, including "
            "a 2 degrees C or lower scenario."
        ),
        "guidance": (
            "Organizations should describe: (1) climate scenarios used; "
            "(2) associated time horizons; (3) inputs, assumptions, and "
            "analytical methods; (4) results and implications."
        ),
        "minimum_word_count": 400,
        "required_elements": [
            "scenarios_used",
            "time_horizons",
            "inputs_assumptions",
            "analytical_methods",
            "results",
            "strategic_implications",
        ],
    },
    "rm_a": {
        "title": "Risk Identification",
        "prompt": (
            "Describe the organization's processes for identifying and "
            "assessing climate-related risks, including how materiality "
            "is determined and which risk categories are considered."
        ),
        "guidance": (
            "Organizations should describe: (1) risk identification processes; "
            "(2) existing and emerging regulatory requirements; (3) materiality "
            "determination; (4) risk assessment frequency."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "identification_process",
            "materiality_criteria",
            "assessment_frequency",
            "risk_categories",
        ],
    },
    "rm_b": {
        "title": "Risk Management Process",
        "prompt": (
            "Describe the organization's processes for managing "
            "climate-related risks, including prioritization, response "
            "strategies, and monitoring procedures."
        ),
        "guidance": (
            "Organizations should describe: (1) prioritization processes; "
            "(2) management decisions on risk mitigation, transfer, acceptance, "
            "or control; (3) how processes are integrated into overall risk."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "prioritization_process",
            "response_strategies",
            "management_decisions",
            "monitoring_procedures",
        ],
    },
    "rm_c": {
        "title": "ERM Integration",
        "prompt": (
            "Describe how processes for identifying, assessing, and managing "
            "climate-related risks are integrated into the organization's "
            "overall risk management."
        ),
        "guidance": (
            "Organizations should describe: (1) how climate risk processes "
            "are integrated into overall risk management; (2) alignment with "
            "enterprise risk management (ERM) framework."
        ),
        "minimum_word_count": 150,
        "required_elements": [
            "erm_integration",
            "framework_alignment",
            "escalation_process",
        ],
    },
    "mt_a": {
        "title": "Climate Metrics",
        "prompt": (
            "Disclose the metrics used by the organization to assess "
            "climate-related risks and opportunities in line with its "
            "strategy and risk management process."
        ),
        "guidance": (
            "Organizations should provide: (1) metrics used to assess "
            "climate risks and opportunities; (2) metrics consistent with "
            "cross-industry metrics; (3) key industry-specific metrics."
        ),
        "minimum_word_count": 250,
        "required_elements": [
            "cross_industry_metrics",
            "industry_specific_metrics",
            "measurement_methodology",
            "data_quality",
        ],
    },
    "mt_b": {
        "title": "GHG Emissions",
        "prompt": (
            "Disclose Scope 1, Scope 2, and, if appropriate, Scope 3 "
            "greenhouse gas (GHG) emissions and the related risks."
        ),
        "guidance": (
            "Organizations should provide: (1) Scope 1 and 2 GHG emissions "
            "independently; (2) Scope 3 emissions if appropriate; "
            "(3) calculation methodology; (4) emission factors used."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "scope_1_emissions",
            "scope_2_emissions",
            "scope_3_emissions",
            "methodology",
            "emission_factors",
        ],
    },
    "mt_c": {
        "title": "Targets",
        "prompt": (
            "Describe the targets used by the organization to manage "
            "climate-related risks and opportunities and performance "
            "against targets."
        ),
        "guidance": (
            "Organizations should describe: (1) climate-related targets; "
            "(2) base year; (3) target year; (4) metrics used; "
            "(5) progress against each target."
        ),
        "minimum_word_count": 200,
        "required_elements": [
            "targets_defined",
            "base_year",
            "target_year",
            "performance_tracking",
            "sbti_alignment",
        ],
    },
}


# ---------------------------------------------------------------------------
# Regulatory adaptations per jurisdiction
# ---------------------------------------------------------------------------

REGULATORY_ADAPTATIONS: Dict[str, Dict[str, Any]] = {
    "UK_FCA": {
        "jurisdiction": "United Kingdom",
        "regulator": "Financial Conduct Authority (FCA)",
        "regulation": "Listing Rules LR 9.8.6R(8), PS 21/24",
        "effective_date": "2021-01-01",
        "mandatory": True,
        "scope": "Premium and Standard listed companies",
        "additional_requirements": [
            "Comply or explain basis under LR 9.8.6R(8)",
            "Transition plan disclosure recommended (TPT Framework)",
            "Location of TCFD disclosures must be signposted in annual report",
            "FCA expects quantitative scenario analysis by second year",
        ],
        "modified_disclosures": {
            "str_c": "FCA expects quantitative scenario analysis by year 2 of reporting",
            "mt_b": "Must include Scope 3 emissions or explain why not",
            "gov_a": "Must identify specific board committees with climate oversight",
        },
    },
    "EU_CSRD": {
        "jurisdiction": "European Union",
        "regulator": "European Financial Reporting Advisory Group (EFRAG)",
        "regulation": "CSRD / ESRS E1 Climate Change",
        "effective_date": "2024-01-01",
        "mandatory": True,
        "scope": "Large companies and listed SMEs",
        "additional_requirements": [
            "Double materiality assessment (impact + financial)",
            "ESRS E1 climate change standard compliance",
            "Transition plan aligned with 1.5C objective required",
            "Scope 3 mandatory for all material categories",
            "Value chain impacts required",
            "EU Taxonomy alignment disclosures",
            "Limited assurance from 2024, reasonable assurance from 2028",
        ],
        "modified_disclosures": {
            "str_a": "Double materiality: financial + impact materiality required",
            "str_c": "1.5C transition plan required under ESRS E1-1",
            "mt_b": "All 15 Scope 3 categories must be assessed for materiality",
            "mt_c": "Absolute reduction targets required, net-zero targets must detail residuals",
        },
    },
    "US_SEC": {
        "jurisdiction": "United States",
        "regulator": "Securities and Exchange Commission (SEC)",
        "regulation": "Enhancement and Standardization of Climate-Related Disclosures (2024)",
        "effective_date": "2026-01-01",
        "mandatory": True,
        "scope": "SEC registrants (phased by filer status)",
        "additional_requirements": [
            "Material climate risks in annual report (10-K/20-F)",
            "Scope 1 and 2 GHG emissions (Large Accelerated Filers)",
            "Attestation for Scope 1 and 2 (Large Accelerated Filers)",
            "Financial statement footnote for climate expenses over 1%",
            "Internal carbon price disclosure if used",
            "Transition plan disclosure if adopted",
        ],
        "modified_disclosures": {
            "str_a": "Only risks reasonably likely to have material impact on registrant",
            "mt_b": "Scope 3 not required by SEC; voluntary Scope 3 if material",
            "str_c": "Scenario analysis encouraged but not required",
        },
    },
    "JP_FSA": {
        "jurisdiction": "Japan",
        "regulator": "Financial Services Agency (FSA)",
        "regulation": "Amendment to Cabinet Office Order on Disclosure (2023)",
        "effective_date": "2023-04-01",
        "mandatory": True,
        "scope": "Listed companies on TSE Prime",
        "additional_requirements": [
            "Sustainability disclosure section in annual securities report",
            "Governance and risk management disclosures required",
            "Strategy and metrics/targets on comply-or-explain basis",
            "SSBJ standards alignment (Japanese ISSB adoption)",
        ],
        "modified_disclosures": {
            "str_c": "Scenario analysis on comply-or-explain basis for initial years",
            "mt_b": "GHG emissions required for governance; detailed scopes encouraged",
        },
    },
    "SG_SGX": {
        "jurisdiction": "Singapore",
        "regulator": "Singapore Exchange (SGX)",
        "regulation": "Listing Rules 711A and 711B, Practice Note 7.6",
        "effective_date": "2022-01-01",
        "mandatory": True,
        "scope": "All SGX-listed issuers (phased from 2022)",
        "additional_requirements": [
            "TCFD-aligned climate reporting from FY2022 (selected sectors)",
            "All issuers from FY2024",
            "External assurance on Scope 1 and 2 from FY2027",
            "Scope 3 disclosure mandatory from FY2026",
        ],
        "modified_disclosures": {
            "mt_b": "Scope 3 mandatory from FY2026 for all issuers",
            "str_c": "Scenario analysis required from FY2024",
        },
    },
    "HK_HKEX": {
        "jurisdiction": "Hong Kong",
        "regulator": "Hong Kong Exchanges and Clearing (HKEX)",
        "regulation": "New Climate Disclosure Requirements (2024)",
        "effective_date": "2025-01-01",
        "mandatory": True,
        "scope": "All Hong Kong-listed issuers (phased)",
        "additional_requirements": [
            "Mandatory TCFD-aligned climate disclosures from 2025",
            "Scope 1 and 2 emissions mandatory from 2025",
            "Scope 3 on comply-or-explain basis, mandatory from 2026",
            "Scenario analysis required (at least 2 scenarios)",
            "Align with ISSB standards",
        ],
        "modified_disclosures": {
            "str_c": "At least two climate scenarios required (1.5C and >2.5C)",
            "mt_b": "Scope 3 on comply-or-explain basis initially",
        },
    },
    "AU_ASRS": {
        "jurisdiction": "Australia",
        "regulator": "Australian Accounting Standards Board (AASB)",
        "regulation": "ASRS 1 General Requirements / ASRS 2 Climate",
        "effective_date": "2025-01-01",
        "mandatory": True,
        "scope": "Large entities (Group 1 from 2025, Group 2 from 2026, Group 3 from 2027)",
        "additional_requirements": [
            "ASRS 2 aligned with IFRS S2 with Australian modifications",
            "Scope 3 emissions: relief in first year (estimate acceptable)",
            "Scenario analysis required (qualitative initially acceptable)",
            "Modified liability for forward-looking statements in transition period",
        ],
        "modified_disclosures": {
            "mt_b": "Scope 3 relief in year 1 (reasonable estimate acceptable)",
            "str_c": "Qualitative scenario analysis acceptable in initial reporting year",
        },
    },
    "NZ_XRB": {
        "jurisdiction": "New Zealand",
        "regulator": "External Reporting Board (XRB)",
        "regulation": "NZ CS1, NZ CS2, NZ CS3 (Aotearoa New Zealand Climate Standards)",
        "effective_date": "2023-01-01",
        "mandatory": True,
        "scope": "Climate reporting entities (CREs) as defined in FMCA",
        "additional_requirements": [
            "World's first mandatory TCFD-aligned regime (from 2023)",
            "NZ CS1: Governance and Risk Management",
            "NZ CS2: Strategy (including scenario analysis)",
            "NZ CS3: Metrics and Targets (including Scope 3)",
            "Assurance required from second reporting period",
        ],
        "modified_disclosures": {
            "str_c": "At least three scenarios required (1.5C, 2.5C, 3C+)",
            "mt_b": "Full Scope 3 required from first reporting period",
        },
    },
}


# ---------------------------------------------------------------------------
# Compliance criteria per disclosure code
# ---------------------------------------------------------------------------

COMPLIANCE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "gov_a": {
        "min_word_count": 200,
        "required_topics": [
            "board_review_frequency",
            "committee_structure",
            "reporting_cadence",
        ],
        "weight": 8.0,
        "pillar": "governance",
    },
    "gov_b": {
        "min_word_count": 200,
        "required_topics": [
            "management_roles",
            "organizational_structure",
            "reporting_process",
        ],
        "weight": 8.0,
        "pillar": "governance",
    },
    "str_a": {
        "min_word_count": 300,
        "required_topics": [
            "physical_risks",
            "transition_risks",
            "opportunities",
            "time_horizons",
        ],
        "weight": 10.0,
        "pillar": "strategy",
    },
    "str_b": {
        "min_word_count": 300,
        "required_topics": [
            "business_impact",
            "financial_planning_impact",
            "products_services_impact",
        ],
        "weight": 10.0,
        "pillar": "strategy",
    },
    "str_c": {
        "min_word_count": 400,
        "required_topics": [
            "scenarios_used",
            "time_horizons",
            "analytical_methods",
            "results",
        ],
        "weight": 12.0,
        "pillar": "strategy",
    },
    "rm_a": {
        "min_word_count": 200,
        "required_topics": [
            "identification_process",
            "materiality_criteria",
            "assessment_frequency",
        ],
        "weight": 8.0,
        "pillar": "risk_management",
    },
    "rm_b": {
        "min_word_count": 200,
        "required_topics": [
            "prioritization_process",
            "response_strategies",
            "monitoring_procedures",
        ],
        "weight": 8.0,
        "pillar": "risk_management",
    },
    "rm_c": {
        "min_word_count": 150,
        "required_topics": [
            "erm_integration",
            "framework_alignment",
        ],
        "weight": 6.0,
        "pillar": "risk_management",
    },
    "mt_a": {
        "min_word_count": 250,
        "required_topics": [
            "cross_industry_metrics",
            "industry_specific_metrics",
            "measurement_methodology",
        ],
        "weight": 10.0,
        "pillar": "metrics_targets",
    },
    "mt_b": {
        "min_word_count": 200,
        "required_topics": [
            "scope_1_emissions",
            "scope_2_emissions",
            "methodology",
        ],
        "weight": 12.0,
        "pillar": "metrics_targets",
    },
    "mt_c": {
        "min_word_count": 200,
        "required_topics": [
            "targets_defined",
            "base_year",
            "performance_tracking",
        ],
        "weight": 8.0,
        "pillar": "metrics_targets",
    },
}


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class ComplianceSectionScore(BaseModel):
    """Compliance score for a single disclosure section."""

    disclosure_code: str = Field(..., description="TCFD disclosure code")
    title: str = Field(default="", description="Disclosure title")
    pillar: str = Field(default="", description="TCFD pillar")
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    word_count: int = Field(default=0, ge=0)
    min_word_count: int = Field(default=0)
    word_count_met: bool = Field(default=False)
    required_topics_total: int = Field(default=0)
    required_topics_found: int = Field(default=0)
    evidence_count: int = Field(default=0)
    has_evidence: bool = Field(default=False)
    section_score: float = Field(default=0.0, ge=0.0, le=100.0)
    weight: float = Field(default=0.0)
    weighted_score: float = Field(default=0.0)


class ComplianceScore(BaseModel):
    """Overall compliance score for a TCFD disclosure."""

    disclosure_id: str = Field(...)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    pillar_scores: Dict[str, float] = Field(default_factory=dict)
    section_scores: List[ComplianceSectionScore] = Field(default_factory=list)
    total_sections: int = Field(default=11)
    completed_sections: int = Field(default=0)
    compliant: bool = Field(default=False)
    compliance_threshold: float = Field(default=70.0)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class DisclosureEvidence(BaseModel):
    """Evidence linked to a disclosure section."""

    id: str = Field(default_factory=_new_id)
    section_id: str = Field(...)
    evidence_type: str = Field(..., description="data, document, reference, calculation")
    reference: str = Field(..., description="Evidence reference or link")
    description: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)


class FullReport(BaseModel):
    """Assembled full TCFD report."""

    disclosure_id: str = Field(...)
    org_id: str = Field(...)
    reporting_period: str = Field(...)
    title: str = Field(default="")
    status: str = Field(default="draft")
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    pillar_summaries: Dict[str, str] = Field(default_factory=list)
    compliance_score: float = Field(default=0.0)
    total_word_count: int = Field(default=0)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class DisclosureComparison(BaseModel):
    """Year-over-year disclosure comparison."""

    disclosure_id_1: str = Field(...)
    disclosure_id_2: str = Field(...)
    period_1: str = Field(default="")
    period_2: str = Field(default="")
    score_delta: float = Field(default=0.0)
    section_changes: List[Dict[str, Any]] = Field(default_factory=list)
    new_sections: List[str] = Field(default_factory=list)
    removed_sections: List[str] = Field(default_factory=list)
    improved_sections: List[str] = Field(default_factory=list)
    declined_sections: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    compared_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Topic detection keywords
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "board_review_frequency": ["quarterly", "annually", "semi-annually", "frequency", "regular review", "board review"],
    "committee_structure": ["committee", "sub-committee", "sustainability committee", "audit committee", "risk committee"],
    "reporting_cadence": ["reporting", "report to board", "informed", "briefed", "updates"],
    "oversight_scope": ["oversight", "oversee", "monitor", "supervise"],
    "management_roles": ["management", "cso", "cfo", "ceo", "chief sustainability", "senior leadership"],
    "organizational_structure": ["organization", "structure", "department", "team", "function"],
    "reporting_process": ["reporting line", "reports to", "escalation", "information flow"],
    "monitoring_mechanism": ["monitoring", "track", "kpi", "dashboard", "progress review"],
    "short_term_risks": ["short term", "short-term", "0-3 year", "immediate", "near term"],
    "medium_term_risks": ["medium term", "medium-term", "3-10 year", "mid-term"],
    "long_term_risks": ["long term", "long-term", "10-30 year", "2050", "beyond 2030"],
    "physical_risks": ["physical", "flood", "cyclone", "wildfire", "drought", "sea level", "heat"],
    "transition_risks": ["transition", "policy", "regulation", "carbon", "technology", "market"],
    "opportunities": ["opportunity", "revenue", "efficiency", "new market", "innovation", "green"],
    "business_impact": ["business model", "revenue impact", "cost impact", "operations"],
    "strategy_impact": ["strategy", "strategic plan", "business plan", "capital allocation"],
    "financial_planning_impact": ["financial planning", "budget", "investment", "capex", "opex"],
    "products_services_impact": ["products", "services", "offering", "portfolio", "r&d"],
    "scenarios_used": ["scenario", "nze", "iea", "ngfs", "rcp", "ssp", "pathway"],
    "time_horizons": ["2030", "2040", "2050", "short term", "medium term", "long term", "horizon"],
    "inputs_assumptions": ["assumption", "input", "parameter", "variable"],
    "analytical_methods": ["model", "methodology", "analysis", "quantitative", "qualitative", "monte carlo"],
    "results": ["result", "outcome", "finding", "implication", "impact"],
    "strategic_implications": ["implication", "strategic response", "resilience", "adaptation"],
    "identification_process": ["identify", "identification", "scan", "assess", "evaluate"],
    "materiality_criteria": ["material", "materiality", "significance", "threshold"],
    "assessment_frequency": ["annual", "quarterly", "periodic", "review cycle"],
    "risk_categories": ["physical", "transition", "acute", "chronic", "policy", "technology"],
    "prioritization_process": ["prioritize", "rank", "triage", "heat map", "risk matrix"],
    "response_strategies": ["mitigate", "transfer", "accept", "avoid", "response", "action plan"],
    "management_decisions": ["decision", "investment", "divestment", "strategic shift"],
    "monitoring_procedures": ["monitor", "track", "review", "audit", "follow-up"],
    "erm_integration": ["erm", "enterprise risk", "integrated", "holistic", "risk framework"],
    "framework_alignment": ["iso 31000", "coso", "risk framework", "aligned"],
    "escalation_process": ["escalate", "escalation", "threshold", "trigger"],
    "cross_industry_metrics": ["scope 1", "scope 2", "scope 3", "emissions", "carbon", "ghg"],
    "industry_specific_metrics": ["sector", "industry", "sasb", "specific metric"],
    "measurement_methodology": ["methodology", "ghg protocol", "iso 14064", "calculation"],
    "data_quality": ["data quality", "accuracy", "completeness", "assurance", "verification"],
    "scope_1_emissions": ["scope 1", "direct emission", "stationary combustion", "mobile combustion"],
    "scope_2_emissions": ["scope 2", "purchased electricity", "market-based", "location-based"],
    "scope_3_emissions": ["scope 3", "value chain", "upstream", "downstream", "category"],
    "methodology": ["methodology", "ghg protocol", "iso 14064", "calculation approach"],
    "emission_factors": ["emission factor", "ef", "defra", "epa", "ipcc", "ecoinvent"],
    "targets_defined": ["target", "goal", "commitment", "pledge", "reduction target"],
    "base_year": ["base year", "baseline", "reference year"],
    "target_year": ["target year", "2030 target", "2050 target", "net zero"],
    "performance_tracking": ["progress", "performance", "tracking", "on track", "trajectory"],
    "sbti_alignment": ["sbti", "science based", "science-based target", "1.5c aligned"],
}


# ---------------------------------------------------------------------------
# DisclosureGenerator Engine
# ---------------------------------------------------------------------------

class DisclosureGenerator:
    """
    TCFD Disclosure Generator engine.

    Manages the full lifecycle of TCFD disclosure documents including creation,
    section drafting, compliance scoring, evidence linking, workflow approvals,
    multi-format export, and year-over-year comparison.

    Attributes:
        config: Application configuration.
        disclosures: In-memory disclosure store.
        sections: In-memory sections store.
        evidence: In-memory evidence store.

    Example:
        >>> gen = DisclosureGenerator(TCFDAppConfig())
        >>> disc = gen.create_disclosure("org-1", "FY2025", "TCFD Report 2025")
        >>> section = gen.draft_section(disc.id, "gov_a", "Board oversees...", [])
        >>> score = gen.check_compliance(disc.id)
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the DisclosureGenerator.

        Args:
            config: Optional application configuration override.
        """
        self.config = config or TCFDAppConfig()
        self._disclosures: Dict[str, TCFDDisclosure] = {}
        self._sections: Dict[str, DisclosureSection] = {}
        self._evidence: Dict[str, List[DisclosureEvidence]] = {}
        self._disclosure_sections: Dict[str, List[str]] = {}
        logger.info("DisclosureGenerator initialized")

    # ------------------------------------------------------------------
    # Disclosure lifecycle
    # ------------------------------------------------------------------

    def create_disclosure(
        self,
        org_id: str,
        reporting_period: str,
        title: str,
    ) -> TCFDDisclosure:
        """
        Create a new TCFD disclosure document.

        Args:
            org_id: Organization ID.
            reporting_period: Reporting period label (e.g. "FY2025").
            title: Human-readable title for the disclosure.

        Returns:
            Created TCFDDisclosure instance.
        """
        year = self._extract_year(reporting_period)
        disclosure = TCFDDisclosure(
            org_id=org_id,
            reporting_year=year,
            status=DisclosureStatus.DRAFT,
        )
        self._disclosures[disclosure.id] = disclosure
        self._disclosure_sections[disclosure.id] = []
        logger.info(
            "Created disclosure %s for org %s, period %s",
            disclosure.id[:8], org_id, reporting_period,
        )
        return disclosure

    def draft_section(
        self,
        disclosure_id: str,
        disclosure_code: str,
        content: str,
        evidence_refs: Optional[List[str]] = None,
    ) -> DisclosureSection:
        """
        Draft or update a disclosure section for one of the 11 TCFD disclosures.

        Args:
            disclosure_id: Parent disclosure ID.
            disclosure_code: TCFD disclosure code (e.g. "gov_a", "str_c").
            content: Section content text.
            evidence_refs: List of evidence references.

        Returns:
            Created or updated DisclosureSection.

        Raises:
            ValueError: If disclosure_id not found or invalid disclosure_code.
        """
        disclosure = self._disclosures.get(disclosure_id)
        if not disclosure:
            raise ValueError(f"Disclosure {disclosure_id} not found")

        if disclosure_code not in TCFD_DISCLOSURES:
            raise ValueError(
                f"Invalid disclosure code '{disclosure_code}'. "
                f"Valid codes: {list(TCFD_DISCLOSURES.keys())}"
            )

        tcfd_info = TCFD_DISCLOSURES[disclosure_code]
        pillar_str = tcfd_info["pillar"]
        pillar = TCFDPillar(pillar_str)

        # Check if section already exists for this code
        existing_section = self._find_section(disclosure_id, disclosure_code)
        if existing_section:
            existing_section.content = content
            existing_section.evidence_refs = evidence_refs or []
            existing_section.compliance_score = self._score_section(
                disclosure_code, content, evidence_refs or [],
            )
            existing_section.updated_at = _now()
            logger.info(
                "Updated section %s (%s) for disclosure %s",
                existing_section.id[:8], disclosure_code, disclosure_id[:8],
            )
            return existing_section

        section = DisclosureSection(
            disclosure_id=disclosure_id,
            pillar=pillar,
            disclosure_ref=disclosure_code,
            title=tcfd_info["title"],
            content=content,
            evidence_refs=evidence_refs or [],
            compliance_score=self._score_section(
                disclosure_code, content, evidence_refs or [],
            ),
        )
        self._sections[section.id] = section
        self._disclosure_sections.setdefault(disclosure_id, []).append(section.id)

        # Update parent disclosure sections list
        disclosure.sections.append(section)
        self._recompute_disclosure_completeness(disclosure)

        logger.info(
            "Drafted section %s (%s) for disclosure %s, score=%d",
            section.id[:8], disclosure_code, disclosure_id[:8],
            section.compliance_score,
        )
        return section

    def check_compliance(self, disclosure_id: str) -> ComplianceScore:
        """
        Check compliance of a disclosure against TCFD requirements.

        Evaluates each of the 11 disclosure sections for word count,
        required topic coverage, and evidence linkage.

        Args:
            disclosure_id: Disclosure ID to check.

        Returns:
            ComplianceScore with per-section and per-pillar scores.

        Raises:
            ValueError: If disclosure_id not found.
        """
        disclosure = self._disclosures.get(disclosure_id)
        if not disclosure:
            raise ValueError(f"Disclosure {disclosure_id} not found")

        section_scores: List[ComplianceSectionScore] = []
        pillar_totals: Dict[str, List[float]] = {}
        total_weight = 0.0
        weighted_sum = 0.0
        completed = 0

        for code, criteria in COMPLIANCE_CRITERIA.items():
            section = self._find_section(disclosure_id, code)
            tcfd_info = TCFD_DISCLOSURES.get(code, {})
            pillar_name = criteria["pillar"]

            if section and section.content.strip():
                word_count = len(section.content.split())
                min_wc = criteria["min_word_count"]
                word_met = word_count >= min_wc

                required_topics = criteria["required_topics"]
                topics_found = self._count_topics_found(section.content, required_topics)

                evidence_count = len(section.evidence_refs)
                has_evidence = evidence_count > 0

                # Section score = weighted combination
                wc_score = min(word_count / max(min_wc, 1) * 30, 30.0)
                topic_score = (topics_found / max(len(required_topics), 1)) * 50.0
                evidence_score = min(evidence_count * 10, 20.0)
                section_score = round(min(wc_score + topic_score + evidence_score, 100.0), 1)

                weight = criteria["weight"]
                w_score = round(section_score * weight / 100.0, 2)

                section_scores.append(ComplianceSectionScore(
                    disclosure_code=code,
                    title=tcfd_info.get("title", code),
                    pillar=pillar_name,
                    completeness_pct=round(section_score, 1),
                    word_count=word_count,
                    min_word_count=min_wc,
                    word_count_met=word_met,
                    required_topics_total=len(required_topics),
                    required_topics_found=topics_found,
                    evidence_count=evidence_count,
                    has_evidence=has_evidence,
                    section_score=section_score,
                    weight=weight,
                    weighted_score=w_score,
                ))
                total_weight += weight
                weighted_sum += w_score
                completed += 1

                pillar_totals.setdefault(pillar_name, []).append(section_score)
            else:
                weight = criteria["weight"]
                section_scores.append(ComplianceSectionScore(
                    disclosure_code=code,
                    title=tcfd_info.get("title", code),
                    pillar=pillar_name,
                    completeness_pct=0.0,
                    word_count=0,
                    min_word_count=criteria["min_word_count"],
                    word_count_met=False,
                    required_topics_total=len(criteria["required_topics"]),
                    required_topics_found=0,
                    evidence_count=0,
                    has_evidence=False,
                    section_score=0.0,
                    weight=weight,
                    weighted_score=0.0,
                ))
                total_weight += weight
                pillar_totals.setdefault(pillar_name, []).append(0.0)

        overall = round(weighted_sum / max(total_weight, 1) * 100, 1)

        pillar_scores: Dict[str, float] = {}
        for pillar, scores in pillar_totals.items():
            pillar_scores[pillar] = round(sum(scores) / max(len(scores), 1), 1)

        provenance = _sha256(
            f"{disclosure_id}:{overall}:{completed}:{datetime.utcnow().isoformat()}"
        )

        result = ComplianceScore(
            disclosure_id=disclosure_id,
            overall_score=overall,
            pillar_scores=pillar_scores,
            section_scores=section_scores,
            total_sections=11,
            completed_sections=completed,
            compliant=overall >= 70.0,
            compliance_threshold=70.0,
            provenance_hash=provenance,
        )

        logger.info(
            "Compliance check for disclosure %s: overall=%.1f%%, completed=%d/11, compliant=%s",
            disclosure_id[:8], overall, completed, result.compliant,
        )
        return result

    def link_evidence(
        self,
        section_id: str,
        evidence_type: str,
        reference: str,
        description: str = "",
    ) -> DisclosureEvidence:
        """
        Link evidence to a disclosure section.

        Args:
            section_id: Section ID to link evidence to.
            evidence_type: Type of evidence (data, document, reference, calculation).
            reference: Evidence reference string or URL.
            description: Optional description.

        Returns:
            Created DisclosureEvidence.

        Raises:
            ValueError: If section_id not found or invalid evidence_type.
        """
        section = self._sections.get(section_id)
        if not section:
            raise ValueError(f"Section {section_id} not found")

        valid_types = {"data", "document", "reference", "calculation"}
        if evidence_type not in valid_types:
            raise ValueError(
                f"Invalid evidence_type '{evidence_type}'. Valid: {valid_types}"
            )

        evidence = DisclosureEvidence(
            section_id=section_id,
            evidence_type=evidence_type,
            reference=reference,
            description=description,
        )

        self._evidence.setdefault(section_id, []).append(evidence)

        # Update section evidence refs
        if reference not in section.evidence_refs:
            section.evidence_refs.append(reference)
            section.compliance_score = self._score_section(
                section.disclosure_ref, section.content, section.evidence_refs,
            )
            section.updated_at = _now()

        logger.info(
            "Linked evidence %s (%s) to section %s",
            evidence.id[:8], evidence_type, section_id[:8],
        )
        return evidence

    def generate_full_report(self, disclosure_id: str) -> FullReport:
        """
        Assemble a full TCFD report from all 11 sections.

        Args:
            disclosure_id: Disclosure ID.

        Returns:
            FullReport with all sections assembled.

        Raises:
            ValueError: If disclosure_id not found.
        """
        disclosure = self._disclosures.get(disclosure_id)
        if not disclosure:
            raise ValueError(f"Disclosure {disclosure_id} not found")

        sections_data: List[Dict[str, Any]] = []
        total_wc = 0

        for code in TCFD_DISCLOSURES:
            section = self._find_section(disclosure_id, code)
            tcfd_info = TCFD_DISCLOSURES[code]
            if section:
                wc = len(section.content.split())
                total_wc += wc
                sections_data.append({
                    "disclosure_code": code,
                    "pillar": tcfd_info["pillar"],
                    "title": tcfd_info["title"],
                    "ref": tcfd_info["ref"],
                    "content": section.content,
                    "word_count": wc,
                    "compliance_score": section.compliance_score,
                    "evidence_refs": section.evidence_refs,
                    "status": "complete" if section.content.strip() else "empty",
                })
            else:
                sections_data.append({
                    "disclosure_code": code,
                    "pillar": tcfd_info["pillar"],
                    "title": tcfd_info["title"],
                    "ref": tcfd_info["ref"],
                    "content": "",
                    "word_count": 0,
                    "compliance_score": 0,
                    "evidence_refs": [],
                    "status": "missing",
                })

        pillar_summaries = self._build_pillar_summaries(sections_data)
        compliance = self.check_compliance(disclosure_id)

        provenance = _sha256(
            f"{disclosure_id}:{total_wc}:{compliance.overall_score}"
        )

        report = FullReport(
            disclosure_id=disclosure_id,
            org_id=disclosure.org_id,
            reporting_period=f"FY{disclosure.reporting_year}",
            title=f"TCFD Climate-Related Financial Disclosure FY{disclosure.reporting_year}",
            status=disclosure.status.value,
            sections=sections_data,
            pillar_summaries=pillar_summaries,
            compliance_score=compliance.overall_score,
            total_word_count=total_wc,
            provenance_hash=provenance,
        )

        logger.info(
            "Generated full report for disclosure %s: %d sections, %d words, score=%.1f%%",
            disclosure_id[:8], len(sections_data), total_wc, compliance.overall_score,
        )
        return report

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def export_pdf(self, disclosure_id: str) -> bytes:
        """
        Export disclosure as PDF bytes.

        Generates a structured PDF document with cover page, table of contents,
        four pillar sections with all 11 disclosures, and appendices.

        Args:
            disclosure_id: Disclosure ID to export.

        Returns:
            PDF file content as bytes.

        Raises:
            ValueError: If disclosure_id not found.
        """
        report = self.generate_full_report(disclosure_id)
        disclosure = self._disclosures[disclosure_id]

        lines: List[str] = []
        lines.append(f"%PDF-1.7 GL-TCFD-APP Export")
        lines.append(f"Title: {report.title}")
        lines.append(f"Organization: {report.org_id}")
        lines.append(f"Reporting Period: {report.reporting_period}")
        lines.append(f"Status: {report.status}")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append(f"Compliance Score: {report.compliance_score:.1f}%")
        lines.append(f"Total Word Count: {report.total_word_count}")
        lines.append(f"Provenance: {report.provenance_hash}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("TABLE OF CONTENTS")
        lines.append("=" * 60)

        current_pillar = ""
        for idx, sec in enumerate(report.sections, 1):
            pillar = sec["pillar"]
            if pillar != current_pillar:
                current_pillar = pillar
                pillar_display = PILLAR_NAMES.get(
                    TCFDPillar(pillar), pillar.replace("_", " ").title()
                )
                lines.append(f"\n  {pillar_display}")
            lines.append(f"    {idx}. {sec['ref']} - {sec['title']}")

        lines.append("")
        lines.append("=" * 60)

        for sec in report.sections:
            pillar_display = PILLAR_NAMES.get(
                TCFDPillar(sec["pillar"]), sec["pillar"]
            )
            lines.append(f"\n{'=' * 60}")
            lines.append(f"[{pillar_display}] {sec['ref']}: {sec['title']}")
            lines.append(f"{'=' * 60}")
            if sec["content"]:
                lines.append(sec["content"])
            else:
                lines.append("[Section not yet completed]")
            lines.append(f"\nCompliance Score: {sec['compliance_score']}%")
            if sec["evidence_refs"]:
                lines.append(f"Evidence: {', '.join(sec['evidence_refs'])}")

        # Regulatory cross-reference appendix
        lines.append(f"\n{'=' * 60}")
        lines.append("APPENDIX: REGULATORY CROSS-REFERENCE")
        lines.append(f"{'=' * 60}")
        for code, mapping in TCFD_TO_IFRS_S2_MAPPING.items():
            lines.append(
                f"  {code} -> IFRS S2 para {mapping['ifrs_s2_paragraph']} "
                f"({mapping['mapping_status']})"
            )

        pdf_text = "\n".join(lines)
        logger.info(
            "Exported PDF for disclosure %s (%d bytes)",
            disclosure_id[:8], len(pdf_text.encode("utf-8")),
        )
        return pdf_text.encode("utf-8")

    def export_excel(self, disclosure_id: str) -> bytes:
        """
        Export disclosure as Excel bytes with multi-sheet structure.

        Sheets: Overview, Governance, Strategy, Risk Management,
        Metrics & Targets, Compliance, Evidence, ISSB Cross-Walk.

        Args:
            disclosure_id: Disclosure ID to export.

        Returns:
            Excel file content as bytes (TSV representation).

        Raises:
            ValueError: If disclosure_id not found.
        """
        report = self.generate_full_report(disclosure_id)
        compliance = self.check_compliance(disclosure_id)

        sheets: Dict[str, List[List[str]]] = {}

        # Overview sheet
        sheets["Overview"] = [
            ["Field", "Value"],
            ["Title", report.title],
            ["Organization", report.org_id],
            ["Reporting Period", report.reporting_period],
            ["Status", report.status],
            ["Compliance Score", f"{report.compliance_score:.1f}%"],
            ["Total Sections", str(compliance.total_sections)],
            ["Completed Sections", str(compliance.completed_sections)],
            ["Total Word Count", str(report.total_word_count)],
            ["Provenance Hash", report.provenance_hash],
        ]

        # Pillar sheets
        for pillar_key, pillar_name in [
            ("governance", "Governance"),
            ("strategy", "Strategy"),
            ("risk_management", "Risk Management"),
            ("metrics_targets", "Metrics & Targets"),
        ]:
            rows = [["Disclosure Code", "Title", "Word Count", "Score", "Status"]]
            for sec in report.sections:
                if sec["pillar"] == pillar_key:
                    rows.append([
                        sec["disclosure_code"],
                        sec["title"],
                        str(sec["word_count"]),
                        f"{sec['compliance_score']}%",
                        sec["status"],
                    ])
            sheets[pillar_name] = rows

        # Compliance sheet
        comp_rows = [
            ["Code", "Title", "Pillar", "Score", "Word Count", "Min WC",
             "WC Met", "Topics Found", "Topics Total", "Evidence", "Weight"],
        ]
        for ss in compliance.section_scores:
            comp_rows.append([
                ss.disclosure_code, ss.title, ss.pillar,
                f"{ss.section_score:.1f}", str(ss.word_count),
                str(ss.min_word_count), str(ss.word_count_met),
                str(ss.required_topics_found), str(ss.required_topics_total),
                str(ss.evidence_count), f"{ss.weight:.1f}",
            ])
        sheets["Compliance"] = comp_rows

        # ISSB Cross-Walk sheet
        issb_rows = [["TCFD Code", "IFRS S2 Para", "Topic", "Status", "Notes"]]
        for code, mapping in TCFD_TO_IFRS_S2_MAPPING.items():
            issb_rows.append([
                code, mapping["ifrs_s2_paragraph"],
                mapping["ifrs_s2_topic"], mapping["mapping_status"],
                mapping.get("notes", ""),
            ])
        sheets["ISSB Cross-Walk"] = issb_rows

        # Serialize as TSV with sheet markers
        output_lines: List[str] = []
        for sheet_name, rows in sheets.items():
            output_lines.append(f"### SHEET: {sheet_name} ###")
            for row in rows:
                output_lines.append("\t".join(row))
            output_lines.append("")

        excel_text = "\n".join(output_lines)
        logger.info(
            "Exported Excel for disclosure %s (%d bytes, %d sheets)",
            disclosure_id[:8], len(excel_text.encode("utf-8")), len(sheets),
        )
        return excel_text.encode("utf-8")

    def export_json(self, disclosure_id: str) -> Dict[str, Any]:
        """
        Export disclosure as structured JSON dictionary.

        Args:
            disclosure_id: Disclosure ID to export.

        Returns:
            Complete disclosure as JSON-serializable dictionary.

        Raises:
            ValueError: If disclosure_id not found.
        """
        report = self.generate_full_report(disclosure_id)
        compliance = self.check_compliance(disclosure_id)
        disclosure = self._disclosures[disclosure_id]

        result: Dict[str, Any] = {
            "metadata": {
                "standard": "TCFD",
                "version": "2017 Final Report",
                "generator": "GL-TCFD-APP v1.0",
                "generated_at": datetime.utcnow().isoformat(),
                "provenance_hash": report.provenance_hash,
            },
            "disclosure": {
                "id": disclosure.id,
                "org_id": disclosure.org_id,
                "reporting_year": disclosure.reporting_year,
                "reporting_period": report.reporting_period,
                "title": report.title,
                "status": disclosure.status.value,
                "version": disclosure.version,
            },
            "compliance": {
                "overall_score": compliance.overall_score,
                "compliant": compliance.compliant,
                "completed_sections": compliance.completed_sections,
                "total_sections": compliance.total_sections,
                "pillar_scores": compliance.pillar_scores,
            },
            "pillars": {},
            "sections": [],
            "issb_crosswalk": {},
            "regulatory_context": {},
        }

        # Build pillar structure
        for pillar_key in ["governance", "strategy", "risk_management", "metrics_targets"]:
            pillar_sections = [s for s in report.sections if s["pillar"] == pillar_key]
            result["pillars"][pillar_key] = {
                "name": PILLAR_NAMES.get(TCFDPillar(pillar_key), pillar_key),
                "score": compliance.pillar_scores.get(pillar_key, 0.0),
                "section_count": len(pillar_sections),
            }

        # Sections
        for sec in report.sections:
            section_data = {
                "disclosure_code": sec["disclosure_code"],
                "pillar": sec["pillar"],
                "ref": sec["ref"],
                "title": sec["title"],
                "content": sec["content"],
                "word_count": sec["word_count"],
                "compliance_score": sec["compliance_score"],
                "evidence_refs": sec["evidence_refs"],
                "status": sec["status"],
            }
            issb_map = TCFD_TO_IFRS_S2_MAPPING.get(sec["disclosure_code"], {})
            if issb_map:
                section_data["issb_mapping"] = {
                    "ifrs_s2_paragraph": issb_map.get("ifrs_s2_paragraph", ""),
                    "mapping_status": issb_map.get("mapping_status", ""),
                }
            result["sections"].append(section_data)

        # ISSB cross-walk
        for code, mapping in TCFD_TO_IFRS_S2_MAPPING.items():
            result["issb_crosswalk"][code] = mapping

        logger.info("Exported JSON for disclosure %s", disclosure_id[:8])
        return result

    def export_xbrl(self, disclosure_id: str) -> str:
        """
        Export disclosure as XBRL-tagged output for ISSB taxonomy.

        Generates inline XBRL (iXBRL) markup mapping each disclosure
        section to the corresponding IFRS S2 taxonomy element.

        Args:
            disclosure_id: Disclosure ID to export.

        Returns:
            XBRL-tagged string.

        Raises:
            ValueError: If disclosure_id not found.
        """
        report = self.generate_full_report(disclosure_id)
        disclosure = self._disclosures[disclosure_id]

        lines: List[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<xbrl xmlns="http://www.xbrl.org/2003/instance"')
        lines.append('      xmlns:ifrs-s2="http://xbrl.ifrs.org/taxonomy/2023/ifrs-s2"')
        lines.append('      xmlns:tcfd="http://greenlang.io/taxonomy/2026/tcfd">')
        lines.append("")
        lines.append(f'  <tcfd:DisclosureDocument>')
        lines.append(f'    <tcfd:OrganizationId>{disclosure.org_id}</tcfd:OrganizationId>')
        lines.append(f'    <tcfd:ReportingYear>{disclosure.reporting_year}</tcfd:ReportingYear>')
        lines.append(f'    <tcfd:Status>{disclosure.status.value}</tcfd:Status>')
        lines.append(f'    <tcfd:ProvenanceHash>{report.provenance_hash}</tcfd:ProvenanceHash>')

        # Map each section to IFRS S2 paragraph
        _XBRL_ELEMENT_MAP: Dict[str, str] = {
            "gov_a": "ifrs-s2:GovernanceBoardOversight",
            "gov_b": "ifrs-s2:GovernanceManagementRole",
            "str_a": "ifrs-s2:StrategyRisksOpportunities",
            "str_b": "ifrs-s2:StrategyBusinessImpact",
            "str_c": "ifrs-s2:StrategyClimateResilience",
            "rm_a": "ifrs-s2:RiskManagementIdentification",
            "rm_b": "ifrs-s2:RiskManagementProcess",
            "rm_c": "ifrs-s2:RiskManagementERMIntegration",
            "mt_a": "ifrs-s2:MetricsClimateRelated",
            "mt_b": "ifrs-s2:MetricsGHGEmissions",
            "mt_c": "ifrs-s2:MetricsTargets",
        }

        for sec in report.sections:
            code = sec["disclosure_code"]
            element = _XBRL_ELEMENT_MAP.get(code, f"tcfd:{code}")
            issb_map = TCFD_TO_IFRS_S2_MAPPING.get(code, {})
            para = issb_map.get("ifrs_s2_paragraph", "")

            lines.append(f"")
            lines.append(f'    <{element}')
            lines.append(f'        tcfd:disclosureCode="{code}"')
            lines.append(f'        ifrs-s2:paragraph="{para}"')
            lines.append(f'        tcfd:complianceScore="{sec["compliance_score"]}">')
            content_escaped = (sec["content"] or "").replace("&", "&amp;").replace("<", "&lt;")
            lines.append(f"      {content_escaped}")
            lines.append(f"    </{element}>")

        lines.append(f"  </tcfd:DisclosureDocument>")
        lines.append("</xbrl>")

        xbrl_output = "\n".join(lines)
        logger.info(
            "Exported XBRL for disclosure %s (%d bytes)",
            disclosure_id[:8], len(xbrl_output.encode("utf-8")),
        )
        return xbrl_output

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_disclosures(
        self,
        disclosure_id_1: str,
        disclosure_id_2: str,
    ) -> DisclosureComparison:
        """
        Compare two disclosures for year-over-year analysis.

        Args:
            disclosure_id_1: First (earlier) disclosure ID.
            disclosure_id_2: Second (later) disclosure ID.

        Returns:
            DisclosureComparison with deltas and changes.

        Raises:
            ValueError: If either disclosure_id not found.
        """
        disc1 = self._disclosures.get(disclosure_id_1)
        disc2 = self._disclosures.get(disclosure_id_2)
        if not disc1:
            raise ValueError(f"Disclosure {disclosure_id_1} not found")
        if not disc2:
            raise ValueError(f"Disclosure {disclosure_id_2} not found")

        compliance1 = self.check_compliance(disclosure_id_1)
        compliance2 = self.check_compliance(disclosure_id_2)
        score_delta = round(compliance2.overall_score - compliance1.overall_score, 1)

        section_changes: List[Dict[str, Any]] = []
        new_sections: List[str] = []
        removed_sections: List[str] = []
        improved: List[str] = []
        declined: List[str] = []

        scores1 = {ss.disclosure_code: ss for ss in compliance1.section_scores}
        scores2 = {ss.disclosure_code: ss for ss in compliance2.section_scores}

        for code in TCFD_DISCLOSURES:
            s1 = scores1.get(code)
            s2 = scores2.get(code)
            s1_score = s1.section_score if s1 else 0.0
            s2_score = s2.section_score if s2 else 0.0
            delta = round(s2_score - s1_score, 1)

            section_changes.append({
                "disclosure_code": code,
                "title": TCFD_DISCLOSURES[code]["title"],
                "score_period_1": s1_score,
                "score_period_2": s2_score,
                "delta": delta,
            })

            if s1_score == 0.0 and s2_score > 0.0:
                new_sections.append(code)
            elif s1_score > 0.0 and s2_score == 0.0:
                removed_sections.append(code)
            elif delta > 0:
                improved.append(code)
            elif delta < 0:
                declined.append(code)

        # Build summary
        direction = "improved" if score_delta > 0 else "declined" if score_delta < 0 else "unchanged"
        summary = (
            f"Overall compliance score {direction} by {abs(score_delta):.1f} percentage points "
            f"from {compliance1.overall_score:.1f}% to {compliance2.overall_score:.1f}%. "
            f"{len(improved)} section(s) improved, {len(declined)} section(s) declined, "
            f"{len(new_sections)} new section(s) added."
        )

        result = DisclosureComparison(
            disclosure_id_1=disclosure_id_1,
            disclosure_id_2=disclosure_id_2,
            period_1=f"FY{disc1.reporting_year}",
            period_2=f"FY{disc2.reporting_year}",
            score_delta=score_delta,
            section_changes=section_changes,
            new_sections=new_sections,
            removed_sections=removed_sections,
            improved_sections=improved,
            declined_sections=declined,
            summary=summary,
        )

        logger.info(
            "Compared disclosures %s vs %s: delta=%.1f%%",
            disclosure_id_1[:8], disclosure_id_2[:8], score_delta,
        )
        return result

    # ------------------------------------------------------------------
    # History and workflow
    # ------------------------------------------------------------------

    def get_disclosure_history(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Get disclosure history for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            List of disclosure summaries ordered by reporting year descending.
        """
        results: List[Dict[str, Any]] = []
        for disc in self._disclosures.values():
            if disc.org_id == org_id:
                compliance = self.check_compliance(disc.id)
                results.append({
                    "disclosure_id": disc.id,
                    "org_id": disc.org_id,
                    "reporting_year": disc.reporting_year,
                    "status": disc.status.value,
                    "version": disc.version,
                    "compliance_score": compliance.overall_score,
                    "completed_sections": compliance.completed_sections,
                    "approved_by": disc.approved_by,
                    "created_at": disc.created_at.isoformat(),
                    "updated_at": disc.updated_at.isoformat(),
                })

        results.sort(key=lambda x: x["reporting_year"], reverse=True)
        logger.info(
            "Retrieved %d disclosures for org %s", len(results), org_id,
        )
        return results

    def approve_disclosure(
        self,
        disclosure_id: str,
        approver_id: str,
    ) -> TCFDDisclosure:
        """
        Approve a disclosure (transition DRAFT/REVIEW -> APPROVED).

        Args:
            disclosure_id: Disclosure ID.
            approver_id: ID of the approver.

        Returns:
            Updated TCFDDisclosure.

        Raises:
            ValueError: If disclosure not found or invalid state transition.
        """
        disclosure = self._disclosures.get(disclosure_id)
        if not disclosure:
            raise ValueError(f"Disclosure {disclosure_id} not found")

        if disclosure.status == DisclosureStatus.PUBLISHED:
            raise ValueError(
                f"Cannot approve disclosure {disclosure_id}: already published"
            )

        disclosure.status = DisclosureStatus.APPROVED
        disclosure.approved_by = approver_id
        disclosure.updated_at = _now()

        logger.info(
            "Disclosure %s approved by %s",
            disclosure_id[:8], approver_id,
        )
        return disclosure

    def publish_disclosure(self, disclosure_id: str) -> TCFDDisclosure:
        """
        Publish a disclosure (transition APPROVED -> PUBLISHED).

        Args:
            disclosure_id: Disclosure ID.

        Returns:
            Updated TCFDDisclosure.

        Raises:
            ValueError: If disclosure not found or not in APPROVED status.
        """
        disclosure = self._disclosures.get(disclosure_id)
        if not disclosure:
            raise ValueError(f"Disclosure {disclosure_id} not found")

        if disclosure.status != DisclosureStatus.APPROVED:
            raise ValueError(
                f"Cannot publish disclosure {disclosure_id}: "
                f"status is {disclosure.status.value}, must be 'approved'"
            )

        disclosure.status = DisclosureStatus.PUBLISHED
        disclosure.updated_at = _now()

        # Recompute provenance with final state
        payload = (
            f"{disclosure.org_id}:{disclosure.reporting_year}:"
            f"{disclosure.version}:{disclosure.completeness_score}"
        )
        disclosure.provenance_hash = _sha256(payload)

        logger.info("Disclosure %s published", disclosure_id[:8])
        return disclosure

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_section(
        self, disclosure_id: str, disclosure_code: str,
    ) -> Optional[DisclosureSection]:
        """Find a section by disclosure ID and disclosure code."""
        section_ids = self._disclosure_sections.get(disclosure_id, [])
        for sid in section_ids:
            section = self._sections.get(sid)
            if section and section.disclosure_ref == disclosure_code:
                return section
        return None

    def _score_section(
        self,
        disclosure_code: str,
        content: str,
        evidence_refs: List[str],
    ) -> int:
        """Score a section 0-100 based on content quality."""
        criteria = COMPLIANCE_CRITERIA.get(disclosure_code)
        if not criteria:
            return 0

        word_count = len(content.split()) if content else 0
        min_wc = criteria["min_word_count"]

        wc_score = min(word_count / max(min_wc, 1) * 30, 30.0)

        required_topics = criteria["required_topics"]
        topics_found = self._count_topics_found(content, required_topics)
        topic_score = (topics_found / max(len(required_topics), 1)) * 50.0

        evidence_score = min(len(evidence_refs) * 10, 20.0)

        total = min(wc_score + topic_score + evidence_score, 100.0)
        return round(total)

    def _count_topics_found(
        self, content: str, required_topics: List[str],
    ) -> int:
        """Count how many required topics are covered in content."""
        if not content:
            return 0
        content_lower = content.lower()
        found = 0
        for topic in required_topics:
            keywords = _TOPIC_KEYWORDS.get(topic, [topic.replace("_", " ")])
            if any(kw.lower() in content_lower for kw in keywords):
                found += 1
        return found

    def _recompute_disclosure_completeness(self, disclosure: TCFDDisclosure) -> None:
        """Recompute the completeness score of a disclosure."""
        if disclosure.sections:
            total = sum(s.compliance_score for s in disclosure.sections)
            avg = Decimal(str(total)) / Decimal(str(len(disclosure.sections)))
            object.__setattr__(disclosure, "completeness_score", avg)

    def _build_pillar_summaries(
        self, sections_data: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Build a summary string for each TCFD pillar."""
        summaries: Dict[str, str] = {}
        for pillar_key, pillar_name in [
            ("governance", "Governance"),
            ("strategy", "Strategy"),
            ("risk_management", "Risk Management"),
            ("metrics_targets", "Metrics & Targets"),
        ]:
            pillar_sections = [s for s in sections_data if s["pillar"] == pillar_key]
            complete = sum(1 for s in pillar_sections if s["status"] == "complete")
            total = len(pillar_sections)
            avg_score = (
                sum(s["compliance_score"] for s in pillar_sections) / max(total, 1)
            )
            summaries[pillar_key] = (
                f"{pillar_name}: {complete}/{total} sections complete, "
                f"average score {avg_score:.1f}%"
            )
        return summaries

    @staticmethod
    def _extract_year(reporting_period: str) -> int:
        """Extract a 4-digit year from a reporting period string."""
        import re
        match = re.search(r"(\d{4})", reporting_period)
        if match:
            return int(match.group(1))
        return date.today().year
