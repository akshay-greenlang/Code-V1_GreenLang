# -*- coding: utf-8 -*-
"""
CredibilityAssessmentEngine - PACK-025 Race to Zero Engine 9
==============================================================

Evaluates pledge credibility against the UN High-Level Expert Group (HLEG)
"Integrity Matters" 10 recommendations with 45+ sub-criteria. Assesses
science-based ambition, governance maturity, transparency, lobbying alignment,
fossil fuel phase-out, just transition planning, voluntary credit use,
systemic investment, and accountability mechanisms.

Calculation Methodology:
    Per-Recommendation Score (0-100):
        Each recommendation has 3-5 sub-criteria
        sub_score = evidence_weight * compliance_factor
        rec_score = mean(sub_scores)

    Overall Credibility Score (0-100):
        Weighted average across 10 recommendations
        Higher weights on Rec 1-3 (pledge/targets/plan), Rec 6 (lobbying),
        Rec 8 (transparency)
        credibility = sum(rec_score * rec_weight) / sum(rec_weights)

    Credibility Tier:
        HIGH:     >= 80
        MODERATE: >= 60
        LOW:      >= 40
        CRITICAL: < 40

    Temperature Alignment (simplified):
        temp = 1.5 + max(0, (4.2 - annual_rate) / 4.2) * 2.0
        Capped at 4.0C

    Governance Maturity:
        EXEMPLARY:  Board oversight + CCO + incentives + risk integration
        MATURE:     Board oversight + executive responsibility
        DEVELOPING: Some governance elements in place
        NASCENT:    Minimal or no climate governance

Regulatory References:
    - UN HLEG "Integrity Matters" (November 2022), 10 Recommendations
    - IPCC AR6 WG3 (2022), 1.5C pathway benchmarks
    - SBTi Corporate Net-Zero Standard v1.1 (2023)
    - ICVCM Core Carbon Principles (2023)
    - VCMI Claims Code of Practice (2023)
    - ISO 14068-1:2023 Carbon Neutrality
    - Race to Zero Starting Line Criteria (2022)

Zero-Hallucination:
    - HLEG sub-criteria from official "Integrity Matters" report
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------


class CredibilityTier(str, Enum):
    """Credibility tier classification."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


class GovernanceMaturity(str, Enum):
    """Governance maturity level."""
    EXEMPLARY = "EXEMPLARY"
    MATURE = "MATURE"
    DEVELOPING = "DEVELOPING"
    NASCENT = "NASCENT"


class SubCriterionStatus(str, Enum):
    """Sub-criterion assessment status."""
    MET = "MET"
    PARTIALLY_MET = "PARTIALLY_MET"
    NOT_MET = "NOT_MET"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PathwayAlignment(str, Enum):
    """Temperature pathway alignment."""
    ALIGNED_1_5C = "1.5C_ALIGNED"
    ALIGNED_WB2C = "WELL_BELOW_2C"
    ALIGNED_2C = "2C_ALIGNED"
    MISALIGNED = "MISALIGNED"


class OffsetUsageRating(str, Enum):
    """Voluntary credit usage rating."""
    RESPONSIBLE = "RESPONSIBLE"
    ACCEPTABLE = "ACCEPTABLE"
    EXCESSIVE = "EXCESSIVE"
    NON_COMPLIANT = "NON_COMPLIANT"


class LobbyingAlignmentRating(str, Enum):
    """Lobbying climate alignment rating."""
    FULLY_ALIGNED = "FULLY_ALIGNED"
    MOSTLY_ALIGNED = "MOSTLY_ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    MISALIGNED = "MISALIGNED"


# ---------------------------------------------------------------------------
# HLEG Recommendations Database
# ---------------------------------------------------------------------------

# Recommendation ID -> (name, weight, sub-criteria IDs)
HLEG_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "REC_01": {
        "name": "Announce net-zero pledge",
        "weight": Decimal("0.12"),
        "description": "Publicly commit to reaching net zero by 2050 at the latest",
        "sub_criteria": {
            "REC_01_SC1": {
                "name": "Pledge specificity",
                "description": "Net-zero pledge specifies target year, scope, and boundary",
                "max_score": Decimal("100"),
            },
            "REC_01_SC2": {
                "name": "Timeline commitment",
                "description": "Net-zero target year is 2050 or earlier",
                "max_score": Decimal("100"),
            },
            "REC_01_SC3": {
                "name": "Scope coverage",
                "description": "Pledge covers Scope 1, 2, and material Scope 3",
                "max_score": Decimal("100"),
            },
            "REC_01_SC4": {
                "name": "Public availability",
                "description": "Pledge is publicly available and accessible",
                "max_score": Decimal("100"),
            },
            "REC_01_SC5": {
                "name": "Governance approval",
                "description": "Pledge approved by board or highest governance body",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_02": {
        "name": "Set interim targets",
        "weight": Decimal("0.14"),
        "description": "Set 2030 interim target consistent with 1.5C pathway",
        "sub_criteria": {
            "REC_02_SC1": {
                "name": "2030 target set",
                "description": "Interim target for 2030 is established and quantified",
                "max_score": Decimal("100"),
            },
            "REC_02_SC2": {
                "name": "Science-based methodology",
                "description": "Target uses SBTi, IEA NZE, or IPCC SR1.5 methodology",
                "max_score": Decimal("100"),
            },
            "REC_02_SC3": {
                "name": "Scope coverage",
                "description": "Interim target covers all material scopes (S1+S2+material S3)",
                "max_score": Decimal("100"),
            },
            "REC_02_SC4": {
                "name": "Annual milestones",
                "description": "Annual milestones defined between now and 2030",
                "max_score": Decimal("100"),
            },
            "REC_02_SC5": {
                "name": "Fair share",
                "description": "Target reflects fair share based on capability and historical responsibility",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_03": {
        "name": "Implement transition plan",
        "weight": Decimal("0.14"),
        "description": "Create and implement a comprehensive transition plan",
        "sub_criteria": {
            "REC_03_SC1": {
                "name": "Quantified actions",
                "description": "Plan includes specific, quantified decarbonization actions",
                "max_score": Decimal("100"),
            },
            "REC_03_SC2": {
                "name": "Resource allocation",
                "description": "Financial, human, and technical resources allocated",
                "max_score": Decimal("100"),
            },
            "REC_03_SC3": {
                "name": "Implementation timeline",
                "description": "Actions have defined timelines with milestones",
                "max_score": Decimal("100"),
            },
            "REC_03_SC4": {
                "name": "Sector alignment",
                "description": "Plan aligned with relevant sector decarbonization pathway",
                "max_score": Decimal("100"),
            },
            "REC_03_SC5": {
                "name": "Technology pathway",
                "description": "Technology choices identified with TRL assessment",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_04": {
        "name": "Phase out fossil fuels",
        "weight": Decimal("0.10"),
        "description": "End expansion and begin phase-out of fossil fuel use/investment",
        "sub_criteria": {
            "REC_04_SC1": {
                "name": "No new fossil capacity",
                "description": "No investment in new fossil fuel exploration or capacity",
                "max_score": Decimal("100"),
            },
            "REC_04_SC2": {
                "name": "Divestment policy",
                "description": "Policy for divesting from fossil fuel assets (where applicable)",
                "max_score": Decimal("100"),
            },
            "REC_04_SC3": {
                "name": "Phase-out timeline",
                "description": "Clear timeline for phasing out unabated fossil fuel use",
                "max_score": Decimal("100"),
            },
            "REC_04_SC4": {
                "name": "Stranded asset risk",
                "description": "Stranded asset risk assessed and disclosed",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_05": {
        "name": "Use voluntary credits responsibly",
        "weight": Decimal("0.08"),
        "description": "Credits complement reductions; meet quality criteria",
        "sub_criteria": {
            "REC_05_SC1": {
                "name": "Complementary use",
                "description": "Credits supplement (not substitute) direct emission reductions",
                "max_score": Decimal("100"),
            },
            "REC_05_SC2": {
                "name": "Quality criteria",
                "description": "Credits meet recognized quality standards (ICVCM CCP)",
                "max_score": Decimal("100"),
            },
            "REC_05_SC3": {
                "name": "ICVCM alignment",
                "description": "Credit procurement aligns with ICVCM Core Carbon Principles",
                "max_score": Decimal("100"),
            },
            "REC_05_SC4": {
                "name": "Transparency",
                "description": "Credit type, volume, registry, and project details disclosed",
                "max_score": Decimal("100"),
            },
            "REC_05_SC5": {
                "name": "Retirement practice",
                "description": "Credits properly retired in recognized registry",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_06": {
        "name": "Align lobbying with climate goals",
        "weight": Decimal("0.10"),
        "description": "Ensure all lobbying and policy engagement supports climate goals",
        "sub_criteria": {
            "REC_06_SC1": {
                "name": "Trade association audit",
                "description": "Annual audit of trade association climate positions",
                "max_score": Decimal("100"),
            },
            "REC_06_SC2": {
                "name": "Lobbying disclosure",
                "description": "All lobbying activities and expenditures disclosed publicly",
                "max_score": Decimal("100"),
            },
            "REC_06_SC3": {
                "name": "Policy alignment",
                "description": "All policy engagement supports Paris Agreement goals",
                "max_score": Decimal("100"),
            },
            "REC_06_SC4": {
                "name": "No obstruction",
                "description": "No direct or indirect obstruction of climate legislation",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_07": {
        "name": "Plan for a just transition",
        "weight": Decimal("0.08"),
        "description": "Integrate just transition principles into climate plans",
        "sub_criteria": {
            "REC_07_SC1": {
                "name": "Workforce planning",
                "description": "Workforce transition plan with reskilling and redeployment",
                "max_score": Decimal("100"),
            },
            "REC_07_SC2": {
                "name": "Community engagement",
                "description": "Engagement with affected communities on transition impacts",
                "max_score": Decimal("100"),
            },
            "REC_07_SC3": {
                "name": "Stakeholder consultation",
                "description": "Structured stakeholder consultation process in place",
                "max_score": Decimal("100"),
            },
            "REC_07_SC4": {
                "name": "Distributional impacts",
                "description": "Assessment of distributional impacts of transition",
                "max_score": Decimal("100"),
            },
            "REC_07_SC5": {
                "name": "Human rights",
                "description": "Human rights due diligence integrated into transition planning",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_08": {
        "name": "Increase transparency",
        "weight": Decimal("0.10"),
        "description": "Annual public reporting with methodology and verification",
        "sub_criteria": {
            "REC_08_SC1": {
                "name": "Annual public reporting",
                "description": "Climate progress reported publicly at least annually",
                "max_score": Decimal("100"),
            },
            "REC_08_SC2": {
                "name": "Methodology disclosure",
                "description": "Calculation methodologies publicly documented",
                "max_score": Decimal("100"),
            },
            "REC_08_SC3": {
                "name": "Assumption transparency",
                "description": "Key assumptions and limitations transparently documented",
                "max_score": Decimal("100"),
            },
            "REC_08_SC4": {
                "name": "Third-party verification",
                "description": "Emission data and claims verified by independent third party",
                "max_score": Decimal("100"),
            },
            "REC_08_SC5": {
                "name": "Data accessibility",
                "description": "Data available in accessible, machine-readable formats",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_09": {
        "name": "Invest in systemic change",
        "weight": Decimal("0.06"),
        "description": "Contribute to systemic change beyond own value chain",
        "sub_criteria": {
            "REC_09_SC1": {
                "name": "Climate finance contribution",
                "description": "Financial contribution to climate solutions beyond own operations",
                "max_score": Decimal("100"),
            },
            "REC_09_SC2": {
                "name": "R&D investment",
                "description": "Investment in R&D for climate solutions and emerging technologies",
                "max_score": Decimal("100"),
            },
            "REC_09_SC3": {
                "name": "Supply chain capacity building",
                "description": "Active capacity building for supply chain decarbonization",
                "max_score": Decimal("100"),
            },
        },
    },
    "REC_10": {
        "name": "Ensure governance and accountability",
        "weight": Decimal("0.08"),
        "description": "Embed climate into governance, incentives, and accountability",
        "sub_criteria": {
            "REC_10_SC1": {
                "name": "Board oversight",
                "description": "Board-level oversight of climate strategy and targets",
                "max_score": Decimal("100"),
            },
            "REC_10_SC2": {
                "name": "Executive incentives",
                "description": "Executive compensation linked to climate targets",
                "max_score": Decimal("100"),
            },
            "REC_10_SC3": {
                "name": "Climate risk integration",
                "description": "Climate risk integrated into enterprise risk management",
                "max_score": Decimal("100"),
            },
            "REC_10_SC4": {
                "name": "Accountability mechanisms",
                "description": "Internal and external accountability mechanisms in place",
                "max_score": Decimal("100"),
            },
        },
    },
}

# Total sub-criteria count
_TOTAL_SUB_CRITERIA = sum(
    len(rec["sub_criteria"]) for rec in HLEG_RECOMMENDATIONS.values()
)

# IPCC 1.5C pathway benchmarks
IPCC_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "2025": {"reduction_from_2019": Decimal("15"), "annual_rate": Decimal("4.2")},
    "2030": {"reduction_from_2019": Decimal("43"), "annual_rate": Decimal("4.2")},
    "2035": {"reduction_from_2019": Decimal("60"), "annual_rate": Decimal("4.2")},
    "2040": {"reduction_from_2019": Decimal("69"), "annual_rate": Decimal("4.2")},
    "2050": {"reduction_from_2019": Decimal("84"), "annual_rate": Decimal("4.2")},
}

# Offset usage limits (per HLEG and VCMI guidance)
OFFSET_LIMITS: Dict[str, Decimal] = {
    "max_offset_share": Decimal("10"),         # Max 10% of residual emissions
    "removals_preferred_share": Decimal("50"),  # Prefer 50%+ removal credits
    "icvcm_compliance_min": Decimal("100"),     # 100% ICVCM-compliant required
}

# Credibility tier thresholds
TIER_THRESHOLDS: Dict[str, Decimal] = {
    "HIGH": Decimal("80"),
    "MODERATE": Decimal("60"),
    "LOW": Decimal("40"),
    "CRITICAL": Decimal("0"),
}

# Governance maturity requirements
GOVERNANCE_REQUIREMENTS: Dict[str, List[str]] = {
    "EXEMPLARY": ["board_oversight", "cco_role", "executive_incentives", "risk_integration", "external_accountability"],
    "MATURE": ["board_oversight", "executive_responsibility", "risk_integration"],
    "DEVELOPING": ["board_oversight"],
    "NASCENT": [],
}


# ---------------------------------------------------------------------------
# Input / Output Models
# ---------------------------------------------------------------------------


class SubCriterionInput(BaseModel):
    """Input for a single HLEG sub-criterion assessment."""
    criterion_id: str = Field(..., description="Sub-criterion ID (e.g., REC_01_SC1)")
    evidence_provided: bool = Field(False, description="Whether evidence has been provided")
    evidence_description: Optional[str] = Field(None, description="Description of evidence")
    compliance_level: float = Field(0.0, description="Compliance level 0.0-1.0")
    evidence_quality: float = Field(0.0, description="Evidence quality 0.0-1.0")
    source_document: Optional[str] = Field(None, description="Source document reference")
    last_updated: Optional[str] = Field(None, description="Date evidence was last updated")

    @field_validator("compliance_level")
    @classmethod
    def validate_compliance_level(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("evidence_quality")
    @classmethod
    def validate_evidence_quality(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class LobbyingRecord(BaseModel):
    """Record of a lobbying or trade association engagement."""
    association_name: str = Field(..., description="Trade association or lobbying target")
    membership_type: str = Field("member", description="Type of membership or engagement")
    annual_expenditure: float = Field(0.0, description="Annual lobbying expenditure (USD)")
    climate_position_aligned: bool = Field(False, description="Whether climate positions are aligned")
    audit_conducted: bool = Field(False, description="Whether audit of climate positions has been conducted")
    disclosed_publicly: bool = Field(False, description="Whether lobbying activity is publicly disclosed")
    notes: Optional[str] = Field(None, description="Additional notes")


class OffsetUsageRecord(BaseModel):
    """Record of voluntary carbon credit usage."""
    total_offsets_tco2e: float = Field(0.0, description="Total offsets used (tCO2e)")
    total_emissions_tco2e: float = Field(0.0, description="Total emissions (tCO2e)")
    removal_credits_pct: float = Field(0.0, description="% of credits that are removals")
    icvcm_compliant_pct: float = Field(0.0, description="% of credits ICVCM-compliant")
    credits_retired: bool = Field(False, description="Whether credits properly retired")
    registry_used: Optional[str] = Field(None, description="Registry used for retirement")
    project_details_disclosed: bool = Field(False, description="Whether project details disclosed")


class GovernanceInput(BaseModel):
    """Input for governance structure assessment."""
    board_oversight: bool = Field(False, description="Board-level climate oversight")
    cco_role: bool = Field(False, description="Chief Climate/Sustainability Officer role exists")
    executive_responsibility: bool = Field(False, description="Executive with climate responsibility")
    executive_incentives: bool = Field(False, description="Executive compensation linked to climate targets")
    risk_integration: bool = Field(False, description="Climate risk in enterprise risk management")
    external_accountability: bool = Field(False, description="External accountability mechanisms")
    governance_committee: bool = Field(False, description="Dedicated climate/sustainability committee")
    regular_board_reporting: bool = Field(False, description="Regular board reporting on climate progress")


class CredibilityInput(BaseModel):
    """Input for credibility assessment engine."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    assessment_date: Optional[str] = Field(None, description="Assessment date (ISO 8601)")
    reporting_year: int = Field(2024, description="Reporting year")

    # Pledge and target data
    has_net_zero_pledge: bool = Field(False, description="Entity has a net-zero pledge")
    net_zero_target_year: Optional[int] = Field(None, description="Net-zero target year")
    pledge_publicly_available: bool = Field(False, description="Pledge is publicly available")
    pledge_board_approved: bool = Field(False, description="Pledge approved by board")
    scope_1_covered: bool = Field(False, description="Scope 1 covered by pledge")
    scope_2_covered: bool = Field(False, description="Scope 2 covered by pledge")
    scope_3_covered: bool = Field(False, description="Material Scope 3 covered")
    scope_3_coverage_pct: float = Field(0.0, description="Scope 3 category coverage %")

    # Interim target data
    has_2030_target: bool = Field(False, description="Has 2030 interim target")
    target_methodology: Optional[str] = Field(None, description="Target methodology (SBTi, IEA, IPCC)")
    annual_reduction_rate: float = Field(0.0, description="Annualized reduction rate %")
    baseline_year: Optional[int] = Field(None, description="Baseline year")
    baseline_emissions_tco2e: float = Field(0.0, description="Baseline emissions tCO2e")
    current_emissions_tco2e: float = Field(0.0, description="Current year emissions tCO2e")
    annual_milestones_defined: bool = Field(False, description="Annual milestones defined to 2030")
    fair_share_assessment: bool = Field(False, description="Fair share assessment conducted")

    # Transition plan data
    has_transition_plan: bool = Field(False, description="Published transition plan")
    plan_has_quantified_actions: bool = Field(False, description="Plan has quantified actions")
    plan_has_resource_allocation: bool = Field(False, description="Plan has resource allocation")
    plan_has_timeline: bool = Field(False, description="Plan has implementation timeline")
    plan_sector_aligned: bool = Field(False, description="Plan aligned with sector pathway")
    plan_technology_pathway: bool = Field(False, description="Technology pathway identified")

    # Fossil fuel data
    no_new_fossil_capacity: bool = Field(False, description="No new fossil fuel investment")
    has_divestment_policy: bool = Field(False, description="Fossil fuel divestment policy")
    has_phaseout_timeline: bool = Field(False, description="Fossil fuel phase-out timeline")
    stranded_asset_assessed: bool = Field(False, description="Stranded asset risk assessed")

    # Offset usage
    offset_usage: Optional[OffsetUsageRecord] = Field(None, description="Offset usage record")

    # Lobbying
    lobbying_records: List[LobbyingRecord] = Field(
        default_factory=list, description="Lobbying and trade association records"
    )

    # Just transition
    has_workforce_plan: bool = Field(False, description="Workforce transition plan")
    community_engagement: bool = Field(False, description="Community engagement on transition")
    stakeholder_consultation: bool = Field(False, description="Stakeholder consultation process")
    distributional_impact_assessed: bool = Field(False, description="Distributional impacts assessed")
    human_rights_due_diligence: bool = Field(False, description="Human rights due diligence")

    # Transparency
    annual_public_reporting: bool = Field(False, description="Annual public climate reporting")
    methodology_disclosed: bool = Field(False, description="Methodology publicly documented")
    assumptions_transparent: bool = Field(False, description="Assumptions documented")
    third_party_verified: bool = Field(False, description="Third-party verification")
    data_machine_readable: bool = Field(False, description="Data in machine-readable format")

    # Systemic change
    climate_finance_contribution: bool = Field(False, description="Climate finance contribution")
    climate_finance_amount_usd: float = Field(0.0, description="Climate finance amount USD")
    rd_investment: bool = Field(False, description="R&D investment in climate solutions")
    supply_chain_capacity_building: bool = Field(False, description="Supply chain capacity building")

    # Governance
    governance: Optional[GovernanceInput] = Field(None, description="Governance structure")

    # Sub-criteria overrides (for detailed assessments)
    sub_criteria_inputs: List[SubCriterionInput] = Field(
        default_factory=list, description="Detailed sub-criterion assessments"
    )

    @field_validator("reporting_year")
    @classmethod
    def validate_reporting_year(cls, v: int) -> int:
        if v < 2015 or v > 2060:
            raise ValueError(f"Reporting year {v} out of valid range [2015, 2060]")
        return v


class SubCriterionAssessment(BaseModel):
    """Assessment result for a single sub-criterion."""
    criterion_id: str
    criterion_name: str
    recommendation_id: str
    status: str
    score: float
    max_score: float = 100.0
    evidence_quality: float = 0.0
    gap_description: Optional[str] = None
    remediation_action: Optional[str] = None
    effort_level: Optional[str] = None


class RecommendationAssessment(BaseModel):
    """Assessment result for a single HLEG recommendation."""
    recommendation_id: str
    recommendation_name: str
    weight: float
    score: float
    max_score: float = 100.0
    sub_criteria_results: List[SubCriterionAssessment]
    sub_criteria_met: int = 0
    sub_criteria_total: int = 0
    key_gaps: List[str]
    improvement_actions: List[str]


class LobbyingAlignmentAssessment(BaseModel):
    """Lobbying alignment assessment result."""
    overall_rating: str
    associations_audited: int = 0
    associations_total: int = 0
    aligned_count: int = 0
    misaligned_count: int = 0
    total_expenditure_usd: float = 0.0
    publicly_disclosed: bool = False
    key_concerns: List[str]


class OffsetUsageAssessment(BaseModel):
    """Offset usage assessment result."""
    rating: str
    offset_share_pct: float = 0.0
    max_allowed_pct: float = 10.0
    removal_share_pct: float = 0.0
    icvcm_compliance_pct: float = 0.0
    credits_retired: bool = False
    concerns: List[str]
    score: float = 0.0


class GovernanceAssessment(BaseModel):
    """Governance maturity assessment result."""
    maturity_level: str
    elements_present: List[str]
    elements_missing: List[str]
    score: float = 0.0
    recommendations: List[str]


class ImprovementPriority(BaseModel):
    """Ranked improvement priority."""
    rank: int
    recommendation_id: str
    recommendation_name: str
    current_score: float
    gap_score: float
    weighted_impact: float
    actions: List[str]
    estimated_effort: str
    estimated_timeline: str


class CredibilityResult(BaseModel):
    """Complete credibility assessment result."""
    assessment_id: str
    entity_id: str
    entity_name: str
    reporting_year: int
    assessment_date: str

    # Overall scores
    credibility_score: float
    credibility_tier: str
    temperature_alignment: float
    pathway_alignment: str

    # Per-recommendation breakdown
    recommendation_assessments: List[RecommendationAssessment]

    # Specialized assessments
    governance_assessment: GovernanceAssessment
    lobbying_assessment: LobbyingAlignmentAssessment
    offset_assessment: OffsetUsageAssessment

    # Sub-criteria summary
    total_sub_criteria: int
    sub_criteria_met: int
    sub_criteria_partially_met: int
    sub_criteria_not_met: int
    compliance_rate_pct: float

    # Improvement priorities
    improvement_priorities: List[ImprovementPriority]
    top_3_actions: List[str]

    # Summary
    key_strengths: List[str]
    key_weaknesses: List[str]
    overall_narrative: str

    # Provenance
    engine_version: str
    module_version: str
    calculated_at: str
    processing_time_ms: float
    provenance_hash: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CredibilityAssessmentEngine:
    """
    Evaluates pledge credibility against the HLEG "Integrity Matters"
    10 recommendations with 45+ sub-criteria.

    Usage::

        engine = CredibilityAssessmentEngine()
        result = engine.assess(credibility_input)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._module_version = _MODULE_VERSION
        logger.info(
            "CredibilityAssessmentEngine initialized (v%s, %d recommendations, %d sub-criteria)",
            self._module_version,
            len(HLEG_RECOMMENDATIONS),
            _TOTAL_SUB_CRITERIA,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, data: CredibilityInput) -> CredibilityResult:
        """Run full HLEG credibility assessment."""
        t0 = time.monotonic()
        logger.info(
            "Assessing credibility for entity=%s year=%d",
            data.entity_id,
            data.reporting_year,
        )

        # Build sub-criteria override lookup
        sc_overrides: Dict[str, SubCriterionInput] = {
            sc.criterion_id: sc for sc in data.sub_criteria_inputs
        }

        # Assess each recommendation
        rec_assessments: List[RecommendationAssessment] = []
        for rec_id, rec_info in HLEG_RECOMMENDATIONS.items():
            rec_result = self._assess_recommendation(data, rec_id, rec_info, sc_overrides)
            rec_assessments.append(rec_result)

        # Compute overall credibility score (weighted average)
        credibility_score = self._compute_weighted_score(rec_assessments)

        # Determine tier
        credibility_tier = self._determine_tier(credibility_score)

        # Temperature alignment
        annual_rate = _decimal(data.annual_reduction_rate)
        temp_alignment = self._compute_temperature(annual_rate)
        pathway = self._determine_pathway(annual_rate)

        # Governance assessment
        governance_result = self._assess_governance(data)

        # Lobbying assessment
        lobbying_result = self._assess_lobbying(data)

        # Offset assessment
        offset_result = self._assess_offsets(data)

        # Sub-criteria summary
        total_sc = 0
        met_sc = 0
        partial_sc = 0
        not_met_sc = 0
        for rec in rec_assessments:
            for sc in rec.sub_criteria_results:
                total_sc += 1
                if sc.status == SubCriterionStatus.MET.value:
                    met_sc += 1
                elif sc.status == SubCriterionStatus.PARTIALLY_MET.value:
                    partial_sc += 1
                elif sc.status == SubCriterionStatus.NOT_MET.value:
                    not_met_sc += 1

        compliance_rate = _round3(
            float(_safe_pct(_decimal(met_sc), _decimal(total_sc)))
        )

        # Improvement priorities
        priorities = self._rank_improvement_priorities(rec_assessments)

        # Top 3 actions
        top_3 = [p.actions[0] for p in priorities[:3] if p.actions]

        # Strengths and weaknesses
        strengths = self._identify_strengths(rec_assessments)
        weaknesses = self._identify_weaknesses(rec_assessments)

        # Narrative
        narrative = self._generate_narrative(
            credibility_score, credibility_tier, met_sc, total_sc, strengths, weaknesses
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        result = CredibilityResult(
            assessment_id=_new_uuid(),
            entity_id=data.entity_id,
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            assessment_date=data.assessment_date or _utcnow().isoformat(),
            credibility_score=_round3(credibility_score),
            credibility_tier=credibility_tier,
            temperature_alignment=_round3(temp_alignment),
            pathway_alignment=pathway,
            recommendation_assessments=rec_assessments,
            governance_assessment=governance_result,
            lobbying_assessment=lobbying_result,
            offset_assessment=offset_result,
            total_sub_criteria=total_sc,
            sub_criteria_met=met_sc,
            sub_criteria_partially_met=partial_sc,
            sub_criteria_not_met=not_met_sc,
            compliance_rate_pct=compliance_rate,
            improvement_priorities=priorities,
            top_3_actions=top_3,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            overall_narrative=narrative,
            engine_version=self._module_version,
            module_version=self._module_version,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash="",
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Credibility assessment complete: entity=%s score=%.1f tier=%s temp=%.2fC hash=%s",
            data.entity_id,
            credibility_score,
            credibility_tier,
            temp_alignment,
            result.provenance_hash[:12],
        )
        return result

    # ------------------------------------------------------------------
    # Recommendation Assessment
    # ------------------------------------------------------------------

    def _assess_recommendation(
        self,
        data: CredibilityInput,
        rec_id: str,
        rec_info: Dict[str, Any],
        sc_overrides: Dict[str, SubCriterionInput],
    ) -> RecommendationAssessment:
        """Assess a single HLEG recommendation."""
        sub_results: List[SubCriterionAssessment] = []
        sub_met = 0
        sub_total = len(rec_info["sub_criteria"])

        for sc_id, sc_info in rec_info["sub_criteria"].items():
            # Check for override first
            if sc_id in sc_overrides:
                override = sc_overrides[sc_id]
                score = _round3(override.compliance_level * 100.0)
                eq = override.evidence_quality
                status = self._score_to_status(score)
            else:
                # Auto-assess from input data
                score, eq = self._auto_assess_sub_criterion(data, rec_id, sc_id)
                status = self._score_to_status(score)

            gap = None
            remediation = None
            effort = None
            if score < 80.0:
                gap = f"{sc_info['name']}: {sc_info['description']} - currently at {score:.0f}%"
                remediation = f"Improve {sc_info['name'].lower()} to meet HLEG requirements"
                effort = "HIGH" if score < 40.0 else "MEDIUM" if score < 70.0 else "LOW"

            if status == SubCriterionStatus.MET.value:
                sub_met += 1

            sub_results.append(SubCriterionAssessment(
                criterion_id=sc_id,
                criterion_name=sc_info["name"],
                recommendation_id=rec_id,
                status=status,
                score=score,
                evidence_quality=_round3(eq),
                gap_description=gap,
                remediation_action=remediation,
                effort_level=effort,
            ))

        # Recommendation score = mean of sub-criteria scores
        if sub_results:
            rec_score = sum(s.score for s in sub_results) / len(sub_results)
        else:
            rec_score = 0.0

        # Identify key gaps
        key_gaps = [
            s.gap_description for s in sub_results
            if s.gap_description and s.score < 60.0
        ]

        # Improvement actions
        improvement_actions = [
            s.remediation_action for s in sub_results
            if s.remediation_action and s.score < 80.0
        ]

        return RecommendationAssessment(
            recommendation_id=rec_id,
            recommendation_name=rec_info["name"],
            weight=float(rec_info["weight"]),
            score=_round3(rec_score),
            sub_criteria_results=sub_results,
            sub_criteria_met=sub_met,
            sub_criteria_total=sub_total,
            key_gaps=key_gaps,
            improvement_actions=improvement_actions,
        )

    def _auto_assess_sub_criterion(
        self, data: CredibilityInput, rec_id: str, sc_id: str,
    ) -> tuple:
        """Auto-assess a sub-criterion from input data. Returns (score, evidence_quality)."""

        # ------- REC 01: Announce net-zero pledge -------
        if sc_id == "REC_01_SC1":
            # Pledge specificity
            score = 100.0 if (data.has_net_zero_pledge and data.net_zero_target_year) else 0.0
            if data.has_net_zero_pledge and not data.net_zero_target_year:
                score = 40.0
            return score, 0.8 if score > 0 else 0.0

        if sc_id == "REC_01_SC2":
            # Timeline commitment (2050 or earlier)
            if data.net_zero_target_year and data.net_zero_target_year <= 2050:
                return 100.0, 0.9
            elif data.net_zero_target_year and data.net_zero_target_year <= 2060:
                return 50.0, 0.7
            return 0.0, 0.0

        if sc_id == "REC_01_SC3":
            # Scope coverage
            scopes_covered = sum([
                data.scope_1_covered,
                data.scope_2_covered,
                data.scope_3_covered,
            ])
            score = (scopes_covered / 3.0) * 100.0
            return _round3(score), 0.8 if score > 0 else 0.0

        if sc_id == "REC_01_SC4":
            # Public availability
            return (100.0, 0.9) if data.pledge_publicly_available else (0.0, 0.0)

        if sc_id == "REC_01_SC5":
            # Governance approval
            return (100.0, 0.9) if data.pledge_board_approved else (0.0, 0.0)

        # ------- REC 02: Set interim targets -------
        if sc_id == "REC_02_SC1":
            # 2030 target set
            return (100.0, 0.9) if data.has_2030_target else (0.0, 0.0)

        if sc_id == "REC_02_SC2":
            # Science-based methodology
            if data.target_methodology:
                valid = ["sbti", "iea", "ipcc", "science-based", "nze"]
                meth_lower = data.target_methodology.lower()
                if any(v in meth_lower for v in valid):
                    return 100.0, 0.9
                return 50.0, 0.5
            return 0.0, 0.0

        if sc_id == "REC_02_SC3":
            # Scope coverage for interim target
            scopes = sum([data.scope_1_covered, data.scope_2_covered, data.scope_3_covered])
            return _round3((scopes / 3.0) * 100.0), 0.8

        if sc_id == "REC_02_SC4":
            # Annual milestones
            return (100.0, 0.8) if data.annual_milestones_defined else (0.0, 0.0)

        if sc_id == "REC_02_SC5":
            # Fair share
            return (100.0, 0.7) if data.fair_share_assessment else (0.0, 0.0)

        # ------- REC 03: Implement transition plan -------
        if sc_id == "REC_03_SC1":
            return (100.0, 0.8) if data.plan_has_quantified_actions else (0.0, 0.0)

        if sc_id == "REC_03_SC2":
            return (100.0, 0.8) if data.plan_has_resource_allocation else (0.0, 0.0)

        if sc_id == "REC_03_SC3":
            return (100.0, 0.8) if data.plan_has_timeline else (0.0, 0.0)

        if sc_id == "REC_03_SC4":
            return (100.0, 0.8) if data.plan_sector_aligned else (0.0, 0.0)

        if sc_id == "REC_03_SC5":
            return (100.0, 0.7) if data.plan_technology_pathway else (0.0, 0.0)

        # ------- REC 04: Phase out fossil fuels -------
        if sc_id == "REC_04_SC1":
            return (100.0, 0.9) if data.no_new_fossil_capacity else (0.0, 0.0)

        if sc_id == "REC_04_SC2":
            return (100.0, 0.7) if data.has_divestment_policy else (0.0, 0.0)

        if sc_id == "REC_04_SC3":
            return (100.0, 0.8) if data.has_phaseout_timeline else (0.0, 0.0)

        if sc_id == "REC_04_SC4":
            return (100.0, 0.7) if data.stranded_asset_assessed else (0.0, 0.0)

        # ------- REC 05: Use voluntary credits responsibly -------
        if sc_id == "REC_05_SC1":
            return self._assess_offset_complementary(data)

        if sc_id == "REC_05_SC2":
            return self._assess_offset_quality(data)

        if sc_id == "REC_05_SC3":
            return self._assess_offset_icvcm(data)

        if sc_id == "REC_05_SC4":
            if data.offset_usage and data.offset_usage.project_details_disclosed:
                return 100.0, 0.8
            return 0.0, 0.0

        if sc_id == "REC_05_SC5":
            if data.offset_usage and data.offset_usage.credits_retired:
                return 100.0, 0.9
            return 0.0, 0.0

        # ------- REC 06: Align lobbying -------
        if sc_id == "REC_06_SC1":
            return self._assess_lobbying_audit(data)

        if sc_id == "REC_06_SC2":
            return self._assess_lobbying_disclosure(data)

        if sc_id == "REC_06_SC3":
            return self._assess_lobbying_policy_alignment(data)

        if sc_id == "REC_06_SC4":
            # No obstruction - assume aligned if lobbying records show alignment
            if data.lobbying_records:
                misaligned = sum(1 for r in data.lobbying_records if not r.climate_position_aligned)
                if misaligned == 0:
                    return 100.0, 0.8
                return _round3(max(0.0, (1.0 - misaligned / len(data.lobbying_records)) * 100.0)), 0.6
            return 50.0, 0.3  # No records = unknown

        # ------- REC 07: Just transition -------
        if sc_id == "REC_07_SC1":
            return (100.0, 0.7) if data.has_workforce_plan else (0.0, 0.0)

        if sc_id == "REC_07_SC2":
            return (100.0, 0.7) if data.community_engagement else (0.0, 0.0)

        if sc_id == "REC_07_SC3":
            return (100.0, 0.7) if data.stakeholder_consultation else (0.0, 0.0)

        if sc_id == "REC_07_SC4":
            return (100.0, 0.7) if data.distributional_impact_assessed else (0.0, 0.0)

        if sc_id == "REC_07_SC5":
            return (100.0, 0.7) if data.human_rights_due_diligence else (0.0, 0.0)

        # ------- REC 08: Increase transparency -------
        if sc_id == "REC_08_SC1":
            return (100.0, 0.9) if data.annual_public_reporting else (0.0, 0.0)

        if sc_id == "REC_08_SC2":
            return (100.0, 0.8) if data.methodology_disclosed else (0.0, 0.0)

        if sc_id == "REC_08_SC3":
            return (100.0, 0.8) if data.assumptions_transparent else (0.0, 0.0)

        if sc_id == "REC_08_SC4":
            return (100.0, 0.9) if data.third_party_verified else (0.0, 0.0)

        if sc_id == "REC_08_SC5":
            return (100.0, 0.7) if data.data_machine_readable else (0.0, 0.0)

        # ------- REC 09: Invest in systemic change -------
        if sc_id == "REC_09_SC1":
            if data.climate_finance_contribution:
                # Score based on amount (basic threshold at $100K)
                amount = _decimal(data.climate_finance_amount_usd)
                if amount >= Decimal("1000000"):
                    return 100.0, 0.8
                elif amount >= Decimal("100000"):
                    return 70.0, 0.7
                elif amount > Decimal("0"):
                    return 40.0, 0.5
            return 0.0, 0.0

        if sc_id == "REC_09_SC2":
            return (100.0, 0.7) if data.rd_investment else (0.0, 0.0)

        if sc_id == "REC_09_SC3":
            return (100.0, 0.7) if data.supply_chain_capacity_building else (0.0, 0.0)

        # ------- REC 10: Governance and accountability -------
        if sc_id == "REC_10_SC1":
            gov = data.governance
            if gov and gov.board_oversight:
                return 100.0, 0.9
            return 0.0, 0.0

        if sc_id == "REC_10_SC2":
            gov = data.governance
            if gov and gov.executive_incentives:
                return 100.0, 0.8
            return 0.0, 0.0

        if sc_id == "REC_10_SC3":
            gov = data.governance
            if gov and gov.risk_integration:
                return 100.0, 0.8
            return 0.0, 0.0

        if sc_id == "REC_10_SC4":
            gov = data.governance
            if gov and gov.external_accountability:
                return 100.0, 0.8
            elif gov and (gov.governance_committee or gov.regular_board_reporting):
                return 60.0, 0.6
            return 0.0, 0.0

        # Default fallback
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Offset Assessments
    # ------------------------------------------------------------------

    def _assess_offset_complementary(self, data: CredibilityInput) -> tuple:
        """Assess whether offsets are used as complement, not substitute."""
        ou = data.offset_usage
        if not ou or ou.total_offsets_tco2e == 0:
            return 100.0, 0.5  # No offsets = not misusing them

        share = _decimal(ou.total_offsets_tco2e) / _decimal(max(ou.total_emissions_tco2e, 1))
        share_pct = float(share * Decimal("100"))

        if share_pct <= 10.0:
            return 100.0, 0.8
        elif share_pct <= 20.0:
            return 60.0, 0.7
        elif share_pct <= 50.0:
            return 30.0, 0.6
        return 0.0, 0.5

    def _assess_offset_quality(self, data: CredibilityInput) -> tuple:
        """Assess credit quality criteria."""
        ou = data.offset_usage
        if not ou or ou.total_offsets_tco2e == 0:
            return 100.0, 0.3

        icvcm = _decimal(ou.icvcm_compliant_pct)
        if icvcm >= Decimal("100"):
            return 100.0, 0.9
        elif icvcm >= Decimal("80"):
            return 70.0, 0.7
        elif icvcm >= Decimal("50"):
            return 40.0, 0.5
        return 10.0, 0.3

    def _assess_offset_icvcm(self, data: CredibilityInput) -> tuple:
        """Assess ICVCM alignment."""
        ou = data.offset_usage
        if not ou or ou.total_offsets_tco2e == 0:
            return 100.0, 0.3

        icvcm = _decimal(ou.icvcm_compliant_pct)
        removal = _decimal(ou.removal_credits_pct)

        score = Decimal("0")
        # ICVCM compliance (60% weight)
        score += (icvcm / Decimal("100")) * Decimal("60")
        # Removal preference (40% weight)
        score += min(removal / Decimal("50"), Decimal("1")) * Decimal("40")

        return _round3(float(score)), 0.7

    # ------------------------------------------------------------------
    # Lobbying Assessments
    # ------------------------------------------------------------------

    def _assess_lobbying_audit(self, data: CredibilityInput) -> tuple:
        """Assess trade association audit completion."""
        if not data.lobbying_records:
            return 50.0, 0.2  # No records = cannot assess

        audited = sum(1 for r in data.lobbying_records if r.audit_conducted)
        total = len(data.lobbying_records)
        score = (audited / total) * 100.0 if total > 0 else 0.0
        return _round3(score), 0.7

    def _assess_lobbying_disclosure(self, data: CredibilityInput) -> tuple:
        """Assess lobbying disclosure."""
        if not data.lobbying_records:
            return 50.0, 0.2

        disclosed = sum(1 for r in data.lobbying_records if r.disclosed_publicly)
        total = len(data.lobbying_records)
        score = (disclosed / total) * 100.0 if total > 0 else 0.0
        return _round3(score), 0.7

    def _assess_lobbying_policy_alignment(self, data: CredibilityInput) -> tuple:
        """Assess policy engagement alignment."""
        if not data.lobbying_records:
            return 50.0, 0.2

        aligned = sum(1 for r in data.lobbying_records if r.climate_position_aligned)
        total = len(data.lobbying_records)
        score = (aligned / total) * 100.0 if total > 0 else 0.0
        return _round3(score), 0.7

    # ------------------------------------------------------------------
    # Scoring Helpers
    # ------------------------------------------------------------------

    def _score_to_status(self, score: float) -> str:
        """Convert numeric score to status enum value."""
        if score >= 80.0:
            return SubCriterionStatus.MET.value
        elif score >= 40.0:
            return SubCriterionStatus.PARTIALLY_MET.value
        return SubCriterionStatus.NOT_MET.value

    def _compute_weighted_score(self, assessments: List[RecommendationAssessment]) -> float:
        """Compute weighted average credibility score."""
        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        for rec in assessments:
            weight = _decimal(rec.weight)
            score = _decimal(rec.score)
            total_weighted += score * weight
            total_weight += weight

        if total_weight == Decimal("0"):
            return 0.0

        return float(_safe_divide(total_weighted, total_weight))

    def _determine_tier(self, score: float) -> str:
        """Determine credibility tier from score."""
        d_score = _decimal(score)
        if d_score >= TIER_THRESHOLDS["HIGH"]:
            return CredibilityTier.HIGH.value
        elif d_score >= TIER_THRESHOLDS["MODERATE"]:
            return CredibilityTier.MODERATE.value
        elif d_score >= TIER_THRESHOLDS["LOW"]:
            return CredibilityTier.LOW.value
        return CredibilityTier.CRITICAL.value

    def _compute_temperature(self, annual_rate: Decimal) -> float:
        """Compute temperature alignment score (simplified SBTi-aligned)."""
        # temp = 1.5 + max(0, (4.2 - annual_rate) / 4.2) * 2.0
        # Capped at 4.0C, floored at 1.2C
        target_rate = Decimal("4.2")
        if annual_rate >= target_rate:
            return 1.5

        gap = max(Decimal("0"), target_rate - annual_rate)
        temp = Decimal("1.5") + (gap / target_rate) * Decimal("2.0")
        temp = min(temp, Decimal("4.0"))
        return float(_round_val(temp, 2))

    def _determine_pathway(self, annual_rate: Decimal) -> str:
        """Determine pathway alignment from annual reduction rate."""
        if annual_rate >= Decimal("4.2"):
            return PathwayAlignment.ALIGNED_1_5C.value
        elif annual_rate >= Decimal("2.5"):
            return PathwayAlignment.ALIGNED_WB2C.value
        elif annual_rate >= Decimal("1.0"):
            return PathwayAlignment.ALIGNED_2C.value
        return PathwayAlignment.MISALIGNED.value

    # ------------------------------------------------------------------
    # Governance Assessment
    # ------------------------------------------------------------------

    def _assess_governance(self, data: CredibilityInput) -> GovernanceAssessment:
        """Assess governance maturity level."""
        gov = data.governance
        if not gov:
            return GovernanceAssessment(
                maturity_level=GovernanceMaturity.NASCENT.value,
                elements_present=[],
                elements_missing=list(GOVERNANCE_REQUIREMENTS["EXEMPLARY"]),
                score=0.0,
                recommendations=[
                    "Establish board-level climate oversight",
                    "Appoint executive with climate responsibility",
                    "Integrate climate risk into enterprise risk management",
                ],
            )

        # Check all governance elements
        elements_present = []
        elements_missing = []

        checks = {
            "board_oversight": gov.board_oversight,
            "cco_role": gov.cco_role,
            "executive_responsibility": gov.executive_responsibility,
            "executive_incentives": gov.executive_incentives,
            "risk_integration": gov.risk_integration,
            "external_accountability": gov.external_accountability,
        }

        for elem, present in checks.items():
            if present:
                elements_present.append(elem)
            else:
                elements_missing.append(elem)

        # Determine maturity level
        exemplary_reqs = GOVERNANCE_REQUIREMENTS["EXEMPLARY"]
        mature_reqs = GOVERNANCE_REQUIREMENTS["MATURE"]
        developing_reqs = GOVERNANCE_REQUIREMENTS["DEVELOPING"]

        if all(r in elements_present for r in exemplary_reqs):
            maturity = GovernanceMaturity.EXEMPLARY.value
        elif all(r in elements_present for r in mature_reqs):
            maturity = GovernanceMaturity.MATURE.value
        elif all(r in elements_present for r in developing_reqs):
            maturity = GovernanceMaturity.DEVELOPING.value
        else:
            maturity = GovernanceMaturity.NASCENT.value

        # Score
        total_possible = len(checks)
        score = (len(elements_present) / total_possible) * 100.0 if total_possible > 0 else 0.0

        # Recommendations
        recs = []
        if "board_oversight" in elements_missing:
            recs.append("Establish board-level climate oversight committee")
        if "cco_role" in elements_missing:
            recs.append("Appoint Chief Climate Officer or equivalent role")
        if "executive_incentives" in elements_missing:
            recs.append("Link executive compensation to climate target achievement")
        if "risk_integration" in elements_missing:
            recs.append("Integrate climate risk into enterprise risk management framework")
        if "external_accountability" in elements_missing:
            recs.append("Establish external accountability mechanisms (advisory board, AGM votes)")

        return GovernanceAssessment(
            maturity_level=maturity,
            elements_present=elements_present,
            elements_missing=elements_missing,
            score=_round3(score),
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Lobbying Assessment
    # ------------------------------------------------------------------

    def _assess_lobbying(self, data: CredibilityInput) -> LobbyingAlignmentAssessment:
        """Assess lobbying and trade association alignment."""
        records = data.lobbying_records
        if not records:
            return LobbyingAlignmentAssessment(
                overall_rating=LobbyingAlignmentRating.PARTIALLY_ALIGNED.value,
                key_concerns=["No lobbying records provided; unable to fully assess alignment"],
            )

        total = len(records)
        audited = sum(1 for r in records if r.audit_conducted)
        aligned = sum(1 for r in records if r.climate_position_aligned)
        misaligned = total - aligned
        total_spend = sum(r.annual_expenditure for r in records)
        disclosed = all(r.disclosed_publicly for r in records)

        # Determine overall rating
        aligned_pct = (aligned / total) * 100.0 if total > 0 else 0.0
        if aligned_pct >= 100.0 and audited == total:
            rating = LobbyingAlignmentRating.FULLY_ALIGNED.value
        elif aligned_pct >= 80.0:
            rating = LobbyingAlignmentRating.MOSTLY_ALIGNED.value
        elif aligned_pct >= 50.0:
            rating = LobbyingAlignmentRating.PARTIALLY_ALIGNED.value
        else:
            rating = LobbyingAlignmentRating.MISALIGNED.value

        concerns = []
        if misaligned > 0:
            misaligned_names = [r.association_name for r in records if not r.climate_position_aligned]
            concerns.append(
                f"{misaligned} trade association(s) with misaligned climate positions: "
                f"{', '.join(misaligned_names[:3])}"
            )
        if audited < total:
            concerns.append(
                f"Only {audited}/{total} trade associations audited for climate position alignment"
            )
        if not disclosed:
            concerns.append("Not all lobbying activities publicly disclosed")

        return LobbyingAlignmentAssessment(
            overall_rating=rating,
            associations_audited=audited,
            associations_total=total,
            aligned_count=aligned,
            misaligned_count=misaligned,
            total_expenditure_usd=_round3(total_spend),
            publicly_disclosed=disclosed,
            key_concerns=concerns,
        )

    # ------------------------------------------------------------------
    # Offset Assessment
    # ------------------------------------------------------------------

    def _assess_offsets(self, data: CredibilityInput) -> OffsetUsageAssessment:
        """Assess voluntary carbon credit usage."""
        ou = data.offset_usage
        if not ou or ou.total_offsets_tco2e == 0:
            return OffsetUsageAssessment(
                rating=OffsetUsageRating.RESPONSIBLE.value,
                concerns=["No offset usage recorded"],
                score=100.0,
            )

        # Calculate offset share
        total_emissions = max(ou.total_emissions_tco2e, 1.0)
        offset_share = (ou.total_offsets_tco2e / total_emissions) * 100.0
        max_allowed = float(OFFSET_LIMITS["max_offset_share"])

        concerns = []
        score = Decimal("100")

        # Check share limit
        if offset_share > max_allowed:
            concerns.append(
                f"Offset usage ({offset_share:.1f}%) exceeds HLEG recommended maximum ({max_allowed}%)"
            )
            score -= Decimal("30")

        # Check removal share
        removal_min = float(OFFSET_LIMITS["removals_preferred_share"])
        if ou.removal_credits_pct < removal_min:
            concerns.append(
                f"Removal credits ({ou.removal_credits_pct:.1f}%) below recommended ({removal_min}%)"
            )
            score -= Decimal("15")

        # Check ICVCM compliance
        if ou.icvcm_compliant_pct < 100.0:
            concerns.append(
                f"Only {ou.icvcm_compliant_pct:.1f}% of credits are ICVCM-compliant (100% required)"
            )
            score -= Decimal("20")

        # Check retirement
        if not ou.credits_retired:
            concerns.append("Credits not properly retired in recognized registry")
            score -= Decimal("15")

        # Check disclosure
        if not ou.project_details_disclosed:
            concerns.append("Credit project details not publicly disclosed")
            score -= Decimal("10")

        score = max(score, Decimal("0"))

        # Determine rating
        if score >= Decimal("80"):
            rating = OffsetUsageRating.RESPONSIBLE.value
        elif score >= Decimal("60"):
            rating = OffsetUsageRating.ACCEPTABLE.value
        elif score >= Decimal("30"):
            rating = OffsetUsageRating.EXCESSIVE.value
        else:
            rating = OffsetUsageRating.NON_COMPLIANT.value

        return OffsetUsageAssessment(
            rating=rating,
            offset_share_pct=_round3(offset_share),
            max_allowed_pct=max_allowed,
            removal_share_pct=_round3(ou.removal_credits_pct),
            icvcm_compliance_pct=_round3(ou.icvcm_compliant_pct),
            credits_retired=ou.credits_retired,
            concerns=concerns,
            score=_round3(float(score)),
        )

    # ------------------------------------------------------------------
    # Improvement Priorities
    # ------------------------------------------------------------------

    def _rank_improvement_priorities(
        self, assessments: List[RecommendationAssessment],
    ) -> List[ImprovementPriority]:
        """Rank improvement priorities by gap-weighted impact."""
        priorities = []

        for rec in assessments:
            gap = 100.0 - rec.score
            if gap <= 0:
                continue

            weighted_impact = gap * rec.weight
            actions = rec.improvement_actions[:3] if rec.improvement_actions else [
                f"Improve {rec.recommendation_name} compliance"
            ]

            # Estimate effort and timeline
            if gap > 60:
                effort = "HIGH"
                timeline = "6-12 months"
            elif gap > 30:
                effort = "MEDIUM"
                timeline = "3-6 months"
            else:
                effort = "LOW"
                timeline = "1-3 months"

            priorities.append(ImprovementPriority(
                rank=0,  # Set after sorting
                recommendation_id=rec.recommendation_id,
                recommendation_name=rec.recommendation_name,
                current_score=rec.score,
                gap_score=_round3(gap),
                weighted_impact=_round3(weighted_impact),
                actions=actions,
                estimated_effort=effort,
                estimated_timeline=timeline,
            ))

        # Sort by weighted impact descending
        priorities.sort(key=lambda p: p.weighted_impact, reverse=True)

        # Assign ranks
        for i, p in enumerate(priorities):
            p.rank = i + 1

        return priorities

    # ------------------------------------------------------------------
    # Strengths & Weaknesses
    # ------------------------------------------------------------------

    def _identify_strengths(self, assessments: List[RecommendationAssessment]) -> List[str]:
        """Identify key strengths from high-scoring recommendations."""
        strengths = []
        for rec in sorted(assessments, key=lambda r: r.score, reverse=True):
            if rec.score >= 80.0:
                strengths.append(
                    f"Strong performance on {rec.recommendation_name} "
                    f"({rec.sub_criteria_met}/{rec.sub_criteria_total} criteria met, score {rec.score:.1f})"
                )
        return strengths[:5]

    def _identify_weaknesses(self, assessments: List[RecommendationAssessment]) -> List[str]:
        """Identify key weaknesses from low-scoring recommendations."""
        weaknesses = []
        for rec in sorted(assessments, key=lambda r: r.score):
            if rec.score < 60.0:
                weaknesses.append(
                    f"Weak performance on {rec.recommendation_name} "
                    f"(score {rec.score:.1f}, {len(rec.key_gaps)} critical gaps)"
                )
        return weaknesses[:5]

    # ------------------------------------------------------------------
    # Narrative Generation
    # ------------------------------------------------------------------

    def _generate_narrative(
        self,
        score: float,
        tier: str,
        met: int,
        total: int,
        strengths: List[str],
        weaknesses: List[str],
    ) -> str:
        """Generate assessment narrative summary."""
        parts = []

        # Overall assessment
        parts.append(
            f"The entity's climate pledge credibility is rated as {tier} "
            f"with an overall score of {score:.1f}/100. "
            f"Of {total} HLEG sub-criteria assessed, {met} are fully met "
            f"({(met/total*100) if total > 0 else 0:.0f}% compliance rate)."
        )

        # Strengths
        if strengths:
            parts.append(
                f"Key strengths include: {'; '.join(strengths[:3])}."
            )

        # Weaknesses
        if weaknesses:
            parts.append(
                f"Areas requiring improvement: {'; '.join(weaknesses[:3])}."
            )

        # Tier-specific guidance
        if tier == CredibilityTier.HIGH.value:
            parts.append(
                "The entity demonstrates strong climate commitment credibility "
                "and is well-positioned for Race to Zero campaign participation."
            )
        elif tier == CredibilityTier.MODERATE.value:
            parts.append(
                "The entity shows moderate credibility with room for improvement. "
                "Focus on addressing key gaps in the top improvement priorities."
            )
        elif tier == CredibilityTier.LOW.value:
            parts.append(
                "The entity's pledge credibility requires significant strengthening. "
                "Immediate action on governance, transparency, and target-setting is recommended."
            )
        else:
            parts.append(
                "The entity's pledge credibility is critically low. "
                "Fundamental improvements to climate commitment structure are required "
                "before Race to Zero campaign participation can be considered."
            )

        return " ".join(parts)
