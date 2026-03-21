# -*- coding: utf-8 -*-
"""
VCMIValidationEngine - PACK-022 Net Zero Acceleration Engine 9
================================================================

Validates corporate carbon credit claims against the Voluntary Carbon
Markets Integrity Initiative (VCMI) Claims Code of Practice (2023),
assessing eligibility for Silver, Gold, and Platinum claim tiers.

VCMI Claims Code Framework (2023):
    The VCMI Claims Code provides a rulebook for companies to credibly
    use carbon credits as part of their climate commitments.  It defines
    four Foundational Criteria that must be met, plus three claim tiers
    (Silver, Gold, Platinum) based on the volume of carbon credits
    purchased and retired relative to unabated emissions.

    Foundational Criteria:
        1. Science-aligned near-term emissions reduction target
           (SBTi-validated or equivalent, covering Scope 1+2 at minimum)
        2. Demonstrated progress toward the near-term target
           (on track to meet target as measured by annual progress)
        3. Public disclosure of GHG inventory
           (through CDP, ESRS, annual report, or equivalent platform)
        4. Carbon credit quality requirements
           (credits must be eligible under the ICVCM Assessment Framework)

    Claim Tiers:
        - Silver:   Meet all 4 foundational criteria + purchase and retire
                    credits >= 20% of unabated Scope 1, 2, and 3 emissions
        - Gold:     Meet all 4 foundational criteria + purchase and retire
                    credits >= 60% of unabated Scope 1, 2, and 3 emissions
        - Platinum:  Meet all 4 foundational criteria + purchase and retire
                    credits >= 100% of unabated Scope 1, 2, and 3 emissions

ICVCM Core Carbon Principles (CCPs):
    - Additionality: Emission reductions would not have occurred without
      the carbon credit project
    - Permanence: Emission reductions are permanent or have adequate
      safeguards for reversal risk
    - Robust quantification: Emission reductions are conservatively estimated
    - No double counting: Credits are not counted by multiple parties
    - Sustainable development benefits: Projects contribute to SDGs
    - No net harm: Projects do not cause environmental/social harm
    - Transition toward net zero: Credits support the global transition
    - Effective governance: Program governance is transparent and accountable
    - Registry systems: Credits are tracked in credible registries

ISO 14068-1:2023 Carbon Neutrality:
    Additionally, the engine compares VCMI compliance with ISO 14068-1
    requirements for carbon neutrality claims, identifying overlaps and
    gaps between the two frameworks.

Features:
    - 4 foundational criteria validation with 0-100 scoring
    - 3-tier claim eligibility (Silver/Gold/Platinum)
    - ICVCM Core Carbon Principles compliance check
    - Evidence scoring per criterion
    - Gap analysis for tier upgrades
    - Greenwashing risk assessment
    - Annual re-validation tracking
    - ISO 14068-1 comparison

Regulatory References:
    - VCMI Claims Code of Practice (June 2023)
    - ICVCM Assessment Framework (2023)
    - ICVCM Core Carbon Principles (March 2023)
    - ISO 14068-1:2023 Carbon neutrality
    - SBTi Corporate Net-Zero Standard v1.1 (2023)
    - GHG Protocol Corporate Standard (2004, revised 2015)

Zero-Hallucination:
    - All scoring uses deterministic rule-based comparison
    - Tier eligibility uses fixed threshold comparison (20/60/100%)
    - Evidence scores use weighted average of validated inputs
    - Greenwashing risk uses rule-based flag conditions
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

# VCMI tier credit coverage thresholds (% of unabated emissions)
VCMI_TIER_THRESHOLDS: Dict[str, Decimal] = {
    "silver": Decimal("20"),
    "gold": Decimal("60"),
    "platinum": Decimal("100"),
}

# ICVCM Core Carbon Principles (CCP) checklist
ICVCM_CORE_CARBON_PRINCIPLES: List[Dict[str, str]] = [
    {"id": "CCP-1", "name": "Additionality", "description": "Emission reductions would not have occurred without the project"},
    {"id": "CCP-2", "name": "Permanence", "description": "Reductions are permanent or have safeguards for reversal risk"},
    {"id": "CCP-3", "name": "Robust Quantification", "description": "Reductions are conservatively estimated using best methods"},
    {"id": "CCP-4", "name": "No Double Counting", "description": "Credits are not counted by multiple parties"},
    {"id": "CCP-5", "name": "Sustainable Development", "description": "Projects contribute to UN Sustainable Development Goals"},
    {"id": "CCP-6", "name": "No Net Harm", "description": "Projects do not cause environmental or social harm"},
    {"id": "CCP-7", "name": "Transition Toward Net Zero", "description": "Credits support the global transition to net zero"},
    {"id": "CCP-8", "name": "Effective Governance", "description": "Program governance is transparent and accountable"},
    {"id": "CCP-9", "name": "Registry Systems", "description": "Credits are tracked in credible, transparent registries"},
    {"id": "CCP-10", "name": "Third-Party Validation", "description": "Projects are validated and verified by independent bodies"},
]

# ISO 14068-1 comparison points
ISO_14068_REQUIREMENTS: List[Dict[str, str]] = [
    {"id": "ISO-1", "name": "GHG Inventory", "description": "Complete GHG inventory per ISO 14064-1"},
    {"id": "ISO-2", "name": "Reduction Plan", "description": "Documented reduction plan with interim targets"},
    {"id": "ISO-3", "name": "Offsetting", "description": "Use offsets only for residual emissions after maximum reduction"},
    {"id": "ISO-4", "name": "Credit Quality", "description": "Credits from programs meeting ISO 14064-2/3"},
    {"id": "ISO-5", "name": "Time-bound Commitment", "description": "Time-bound carbon neutrality commitment"},
    {"id": "ISO-6", "name": "Public Disclosure", "description": "Publicly disclose inventory, reductions, and offsets"},
    {"id": "ISO-7", "name": "Third-Party Verification", "description": "Independent third-party verification of claim"},
]

# Greenwashing risk indicators
GREENWASHING_INDICATORS: List[Dict[str, str]] = [
    {"id": "GW-1", "name": "No Reduction Target", "description": "Using credits without a science-based reduction target"},
    {"id": "GW-2", "name": "Credits Over Reduction", "description": "Credits exceed own reduction efforts significantly"},
    {"id": "GW-3", "name": "Low Quality Credits", "description": "Using credits that fail ICVCM CCP assessment"},
    {"id": "GW-4", "name": "Selective Scope", "description": "Excluding material Scope 3 from claim boundary"},
    {"id": "GW-5", "name": "No Progress", "description": "No demonstrated progress on own emissions reduction"},
    {"id": "GW-6", "name": "Misleading Language", "description": "Using 'carbon neutral' without meeting ISO 14068-1"},
    {"id": "GW-7", "name": "Stale Inventory", "description": "GHG inventory more than 2 years old"},
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VCMITier(str, Enum):
    """VCMI claim tier."""
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    NOT_ELIGIBLE = "not_eligible"


class CriterionStatus(str, Enum):
    """Status of a foundational criterion."""
    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    NOT_ASSESSED = "not_assessed"


class EvidenceStrength(str, Enum):
    """Strength of evidence provided."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ABSENT = "absent"


class GreenwashingRiskLevel(str, Enum):
    """Greenwashing risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CreditQualityLevel(str, Enum):
    """Carbon credit quality level."""
    CCP_APPROVED = "ccp_approved"
    HIGH_QUALITY = "high_quality"
    STANDARD = "standard"
    LOW_QUALITY = "low_quality"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CarbonCreditPortfolio(BaseModel):
    """Carbon credit portfolio for VCMI assessment."""
    portfolio_id: str = Field(default_factory=_new_uuid, description="Portfolio ID")
    total_credits_retired: Decimal = Field(description="Total credits retired (tCO2e)")
    ccp_approved_credits: Decimal = Field(default=Decimal("0"), description="CCP-approved credits (tCO2e)")
    non_ccp_credits: Decimal = Field(default=Decimal("0"), description="Non-CCP credits (tCO2e)")
    credit_vintage_year: int = Field(default=2025, description="Average credit vintage year")
    registries: List[str] = Field(default_factory=list, description="Credit registry names")
    project_types: List[str] = Field(default_factory=list, description="Project type categories")
    ccp_compliance: Dict[str, bool] = Field(
        default_factory=dict, description="CCP compliance per principle ID"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_credits_retired", "ccp_approved_credits",
                     "non_ccp_credits", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class EmissionsData(BaseModel):
    """Emissions data for VCMI assessment."""
    reporting_year: int = Field(description="Reporting year")
    scope1_emissions: Decimal = Field(description="Scope 1 tCO2e")
    scope2_emissions: Decimal = Field(description="Scope 2 tCO2e")
    scope3_emissions: Decimal = Field(default=Decimal("0"), description="Scope 3 tCO2e")
    total_emissions: Decimal = Field(default=Decimal("0"), description="Total tCO2e")
    unabated_emissions: Decimal = Field(default=Decimal("0"), description="Unabated emissions tCO2e")
    reductions_achieved: Decimal = Field(default=Decimal("0"), description="Reductions vs base year tCO2e")
    base_year: int = Field(default=2019, description="Base year for reductions")
    base_year_emissions: Decimal = Field(default=Decimal("0"), description="Base year total tCO2e")
    has_sbti_target: bool = Field(default=False, description="Has SBTi-validated target")
    target_reduction_pct: Decimal = Field(default=Decimal("0"), description="Target reduction %")
    has_public_disclosure: bool = Field(default=False, description="GHG inventory publicly disclosed")
    disclosure_platform: str = Field(default="", description="Disclosure platform (CDP, ESRS, etc.)")
    inventory_year: int = Field(default=2025, description="Year of most recent GHG inventory")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("scope1_emissions", "scope2_emissions", "scope3_emissions",
                     "total_emissions", "unabated_emissions",
                     "reductions_achieved", "base_year_emissions",
                     "target_reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class FoundationalCriterionResult(BaseModel):
    """Result for a single VCMI foundational criterion."""
    criterion_id: str = Field(description="Criterion identifier (FC-1 to FC-4)")
    criterion_name: str = Field(description="Criterion name")
    status: CriterionStatus = Field(description="Assessment status")
    score: Decimal = Field(description="Score (0-100)")
    evidence_strength: EvidenceStrength = Field(description="Evidence strength")
    findings: List[str] = Field(default_factory=list, description="Detailed findings")
    requirements: List[str] = Field(default_factory=list, description="What is needed to meet")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class TierEligibility(BaseModel):
    """Eligibility assessment for a specific VCMI tier."""
    tier: VCMITier = Field(description="Claim tier")
    eligible: bool = Field(description="Whether eligible for this tier")
    foundational_criteria_met: bool = Field(description="All 4 foundational criteria met")
    credit_threshold_pct: Decimal = Field(description="Required credit coverage (%)")
    actual_coverage_pct: Decimal = Field(description="Actual credit coverage (%)")
    coverage_gap_pct: Decimal = Field(default=Decimal("0"), description="Gap to threshold (%)")
    credits_needed: Decimal = Field(default=Decimal("0"), description="Additional credits needed (tCO2e)")
    reasons: List[str] = Field(default_factory=list, description="Eligibility reasons")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("credit_threshold_pct", "actual_coverage_pct",
                     "coverage_gap_pct", "credits_needed", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class GapToNextTier(BaseModel):
    """Gap analysis for upgrading to the next VCMI tier."""
    current_tier: VCMITier = Field(description="Current tier (or not_eligible)")
    next_tier: VCMITier = Field(description="Next available tier")
    gaps: List[str] = Field(default_factory=list, description="Gaps to address")
    additional_credits_needed: Decimal = Field(description="Additional credits needed (tCO2e)")
    criteria_gaps: List[str] = Field(default_factory=list, description="Foundational criteria gaps")
    estimated_cost_usd: Decimal = Field(default=Decimal("0"), description="Estimated cost (USD)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("additional_credits_needed", "estimated_cost_usd", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class GreenwashingFlag(BaseModel):
    """A greenwashing risk flag."""
    flag_id: str = Field(description="Indicator ID (GW-1 to GW-7)")
    flag_name: str = Field(description="Indicator name")
    triggered: bool = Field(description="Whether the flag is triggered")
    severity: GreenwashingRiskLevel = Field(description="Risk severity")
    description: str = Field(default="", description="Detailed description")
    recommendation: str = Field(default="", description="Recommended action")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ICVCMAssessment(BaseModel):
    """Assessment against ICVCM Core Carbon Principles."""
    principle_id: str = Field(description="CCP identifier")
    principle_name: str = Field(description="Principle name")
    compliant: bool = Field(description="Whether compliant")
    score: Decimal = Field(default=Decimal("0"), description="Score (0-100)")
    evidence: str = Field(default="", description="Evidence summary")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ISOComparison(BaseModel):
    """Comparison with ISO 14068-1 requirements."""
    requirement_id: str = Field(description="ISO requirement ID")
    requirement_name: str = Field(description="Requirement name")
    vcmi_overlap: bool = Field(description="Whether VCMI covers this requirement")
    met_by_vcmi_compliance: bool = Field(description="Whether VCMI compliance meets this")
    gap: str = Field(default="", description="Gap description if not covered")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class VCMIResult(BaseModel):
    """Complete VCMI validation result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    entity_name: str = Field(default="", description="Entity name")
    reporting_year: int = Field(description="Assessment year")
    foundational_criteria_results: List[FoundationalCriterionResult] = Field(
        default_factory=list, description="Per-criterion results"
    )
    all_foundational_met: bool = Field(default=False, description="All 4 criteria met")
    foundational_overall_score: Decimal = Field(
        default=Decimal("0"), description="Average foundational score"
    )
    tier_eligibility: List[TierEligibility] = Field(
        default_factory=list, description="Per-tier eligibility"
    )
    highest_eligible_tier: VCMITier = Field(
        default=VCMITier.NOT_ELIGIBLE, description="Highest eligible tier"
    )
    evidence_scores: Dict[str, str] = Field(
        default_factory=dict, description="Evidence scores by criterion"
    )
    gaps_to_next_tier: Optional[GapToNextTier] = Field(
        default=None, description="Gap analysis for next tier"
    )
    greenwashing_flags: List[GreenwashingFlag] = Field(
        default_factory=list, description="Greenwashing risk flags"
    )
    greenwashing_risk_level: GreenwashingRiskLevel = Field(
        default=GreenwashingRiskLevel.LOW, description="Overall greenwashing risk"
    )
    icvcm_assessment: List[ICVCMAssessment] = Field(
        default_factory=list, description="ICVCM CCP assessment"
    )
    iso_comparison: List[ISOComparison] = Field(
        default_factory=list, description="ISO 14068-1 comparison"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("foundational_overall_score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class VCMIValidationConfig(BaseModel):
    """Configuration for the VCMIValidationEngine."""
    silver_threshold_pct: Decimal = Field(
        default=Decimal("20"), description="Silver tier credit coverage threshold (%)"
    )
    gold_threshold_pct: Decimal = Field(
        default=Decimal("60"), description="Gold tier credit coverage threshold (%)"
    )
    platinum_threshold_pct: Decimal = Field(
        default=Decimal("100"), description="Platinum tier credit coverage threshold (%)"
    )
    min_foundational_score: Decimal = Field(
        default=Decimal("70"), description="Min score per criterion to be 'met'"
    )
    partially_met_threshold: Decimal = Field(
        default=Decimal("40"), description="Score threshold for 'partially met'"
    )
    credit_price_usd_per_tonne: Decimal = Field(
        default=Decimal("15"), description="Assumed credit price for cost estimates"
    )
    max_inventory_age_years: int = Field(
        default=2, description="Max age of GHG inventory for validity"
    )
    decimal_precision: int = Field(
        default=4, description="Decimal places for results"
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

CarbonCreditPortfolio.model_rebuild()
EmissionsData.model_rebuild()
FoundationalCriterionResult.model_rebuild()
TierEligibility.model_rebuild()
GapToNextTier.model_rebuild()
GreenwashingFlag.model_rebuild()
ICVCMAssessment.model_rebuild()
ISOComparison.model_rebuild()
VCMIResult.model_rebuild()
VCMIValidationConfig.model_rebuild()


# ---------------------------------------------------------------------------
# VCMIValidationEngine
# ---------------------------------------------------------------------------


class VCMIValidationEngine:
    """
    VCMI Claims Code validation engine.

    Validates corporate carbon credit claims against the VCMI Claims
    Code of Practice, assessing foundational criteria, tier eligibility,
    ICVCM CCP compliance, greenwashing risk, and ISO 14068-1 comparison.

    Attributes:
        config: Engine configuration.

    Example:
        >>> engine = VCMIValidationEngine()
        >>> result = engine.validate(emissions_data, credit_portfolio)
        >>> print(result.highest_eligible_tier)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VCMIValidationEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = VCMIValidationConfig(**config)
        elif config and isinstance(config, VCMIValidationConfig):
            self.config = config
        else:
            self.config = VCMIValidationConfig()

        logger.info("VCMIValidationEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Foundational Criteria Assessment
    # -------------------------------------------------------------------

    def _assess_criterion_1(
        self, emissions: EmissionsData
    ) -> FoundationalCriterionResult:
        """FC-1: Science-aligned near-term emissions reduction target.

        Assesses whether the entity has an SBTi-validated or equivalent
        science-aligned near-term target covering at minimum Scope 1+2.

        Args:
            emissions: Emissions data.

        Returns:
            FoundationalCriterionResult for FC-1.
        """
        score = Decimal("0")
        findings: List[str] = []
        requirements: List[str] = []

        # SBTi validation (50 points)
        if emissions.has_sbti_target:
            score += Decimal("50")
            findings.append("SBTi-validated target confirmed")
        else:
            requirements.append("Obtain SBTi target validation")

        # Target ambitiousness (30 points)
        if emissions.target_reduction_pct >= Decimal("42"):
            score += Decimal("30")
            findings.append(f"Target reduction of {emissions.target_reduction_pct}% exceeds 42% 1.5C-aligned threshold")
        elif emissions.target_reduction_pct >= Decimal("25"):
            score += Decimal("20")
            findings.append(f"Target reduction of {emissions.target_reduction_pct}% meets well-below-2C threshold")
        elif emissions.target_reduction_pct > Decimal("0"):
            score += Decimal("10")
            findings.append(f"Target reduction of {emissions.target_reduction_pct}% is below science-aligned levels")
            requirements.append("Increase target ambition to at least 42% (1.5C-aligned)")
        else:
            requirements.append("Set a quantified emissions reduction target")

        # Base year defined (10 points)
        if emissions.base_year > 0 and emissions.base_year_emissions > Decimal("0"):
            score += Decimal("10")
            findings.append(f"Base year {emissions.base_year} established with {emissions.base_year_emissions} tCO2e")
        else:
            requirements.append("Define a base year with documented emissions")

        # Coverage of scopes (10 points)
        if emissions.scope3_emissions > Decimal("0"):
            score += Decimal("10")
            findings.append("Scope 3 included in target boundary")
        else:
            score += Decimal("5")
            findings.append("Only Scope 1+2 in target boundary; Scope 3 recommended")
            requirements.append("Include material Scope 3 categories in target boundary")

        score = min(score, Decimal("100"))
        status = self._score_to_status(score)
        evidence = self._score_to_evidence(score)

        result = FoundationalCriterionResult(
            criterion_id="FC-1",
            criterion_name="Science-aligned near-term emissions reduction target",
            status=status,
            score=score,
            evidence_strength=evidence,
            findings=findings,
            requirements=requirements,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_criterion_2(
        self, emissions: EmissionsData
    ) -> FoundationalCriterionResult:
        """FC-2: Demonstrated progress toward near-term target.

        Args:
            emissions: Emissions data.

        Returns:
            FoundationalCriterionResult for FC-2.
        """
        score = Decimal("0")
        findings: List[str] = []
        requirements: List[str] = []

        if emissions.base_year_emissions <= Decimal("0"):
            requirements.append("Establish base year emissions for progress measurement")
            return FoundationalCriterionResult(
                criterion_id="FC-2",
                criterion_name="Demonstrated progress toward near-term target",
                status=CriterionStatus.NOT_MET,
                score=Decimal("0"),
                evidence_strength=EvidenceStrength.ABSENT,
                findings=["No base year emissions data available"],
                requirements=requirements,
            )

        # Calculate progress
        reduction_achieved = emissions.base_year_emissions - emissions.total_emissions
        reduction_pct = _safe_pct(reduction_achieved, emissions.base_year_emissions)

        # Progress scoring (60 points)
        if reduction_pct > Decimal("0"):
            if emissions.target_reduction_pct > Decimal("0"):
                years_elapsed = emissions.reporting_year - emissions.base_year
                total_duration = max(years_elapsed, 1)
                expected_pct = (emissions.target_reduction_pct * _decimal(years_elapsed)
                                / _decimal(max(total_duration, 5)))
                progress_ratio = _safe_divide(reduction_pct, expected_pct) if expected_pct > 0 else Decimal("0")

                if progress_ratio >= Decimal("1"):
                    score += Decimal("60")
                    findings.append(f"On track: {reduction_pct}% reduction achieved vs {expected_pct}% expected")
                elif progress_ratio >= Decimal("0.7"):
                    score += Decimal("40")
                    findings.append(f"Slightly behind: {reduction_pct}% achieved vs {expected_pct}% expected")
                    requirements.append("Accelerate reduction efforts to meet interim milestones")
                else:
                    score += Decimal("20")
                    findings.append(f"Significantly behind: {reduction_pct}% achieved vs {expected_pct}% expected")
                    requirements.append("Major acceleration needed to meet near-term target")
            else:
                score += Decimal("30")
                findings.append(f"Reduction of {reduction_pct}% achieved but no quantified target for comparison")
                requirements.append("Set a quantified reduction target for progress tracking")
        else:
            findings.append("No emissions reduction achieved since base year")
            requirements.append("Demonstrate measurable emissions reductions")

        # Consecutive year improvement (20 points)
        if emissions.reductions_achieved > Decimal("0"):
            score += Decimal("20")
            findings.append(f"Year-on-year reductions of {emissions.reductions_achieved} tCO2e")
        else:
            requirements.append("Demonstrate consecutive year-on-year improvements")

        # Intensity improvement (20 points)
        if reduction_pct > Decimal("5"):
            score += Decimal("20")
            findings.append("Meaningful absolute reduction exceeds 5%")
        elif reduction_pct > Decimal("0"):
            score += Decimal("10")
            findings.append("Some absolute reduction but below 5% threshold")

        score = min(score, Decimal("100"))
        status = self._score_to_status(score)
        evidence = self._score_to_evidence(score)

        result = FoundationalCriterionResult(
            criterion_id="FC-2",
            criterion_name="Demonstrated progress toward near-term target",
            status=status,
            score=score,
            evidence_strength=evidence,
            findings=findings,
            requirements=requirements,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_criterion_3(
        self, emissions: EmissionsData
    ) -> FoundationalCriterionResult:
        """FC-3: Public disclosure of GHG inventory.

        Args:
            emissions: Emissions data.

        Returns:
            FoundationalCriterionResult for FC-3.
        """
        score = Decimal("0")
        findings: List[str] = []
        requirements: List[str] = []

        # Public disclosure (40 points)
        if emissions.has_public_disclosure:
            score += Decimal("40")
            findings.append("GHG inventory is publicly disclosed")
            if emissions.disclosure_platform:
                findings.append(f"Disclosure platform: {emissions.disclosure_platform}")
        else:
            requirements.append("Publicly disclose GHG inventory through CDP, ESRS, or equivalent")

        # Disclosure platform quality (20 points)
        premium_platforms = {"cdp", "esrs", "csrd", "annual_report", "sec"}
        if emissions.disclosure_platform.lower() in premium_platforms:
            score += Decimal("20")
            findings.append("Uses recognized disclosure platform")
        elif emissions.disclosure_platform:
            score += Decimal("10")
            findings.append("Disclosure platform not among top-tier frameworks")
            requirements.append("Consider disclosing through CDP or ESRS for stronger credibility")

        # Inventory freshness (20 points)
        current_year = emissions.reporting_year
        inventory_age = current_year - emissions.inventory_year
        if inventory_age <= 1:
            score += Decimal("20")
            findings.append(f"GHG inventory is current (age: {inventory_age} year(s))")
        elif inventory_age <= self.config.max_inventory_age_years:
            score += Decimal("10")
            findings.append(f"GHG inventory is {inventory_age} years old")
            requirements.append("Update GHG inventory to most recent reporting year")
        else:
            findings.append(f"GHG inventory is {inventory_age} years old (stale)")
            requirements.append(f"GHG inventory must be within {self.config.max_inventory_age_years} years")

        # Scope completeness (20 points)
        has_s1 = emissions.scope1_emissions > Decimal("0")
        has_s2 = emissions.scope2_emissions > Decimal("0")
        has_s3 = emissions.scope3_emissions > Decimal("0")

        if has_s1 and has_s2 and has_s3:
            score += Decimal("20")
            findings.append("All three scopes disclosed")
        elif has_s1 and has_s2:
            score += Decimal("12")
            findings.append("Scope 1 and 2 disclosed; Scope 3 missing")
            requirements.append("Disclose material Scope 3 categories")
        elif has_s1 or has_s2:
            score += Decimal("5")
            findings.append("Incomplete scope coverage in disclosure")
            requirements.append("Disclose both Scope 1 and Scope 2 emissions")

        score = min(score, Decimal("100"))
        status = self._score_to_status(score)
        evidence = self._score_to_evidence(score)

        result = FoundationalCriterionResult(
            criterion_id="FC-3",
            criterion_name="Public disclosure of GHG inventory",
            status=status,
            score=score,
            evidence_strength=evidence,
            findings=findings,
            requirements=requirements,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_criterion_4(
        self, credits: CarbonCreditPortfolio
    ) -> FoundationalCriterionResult:
        """FC-4: Carbon credit quality (ICVCM Assessment Framework).

        Args:
            credits: Carbon credit portfolio data.

        Returns:
            FoundationalCriterionResult for FC-4.
        """
        score = Decimal("0")
        findings: List[str] = []
        requirements: List[str] = []

        total = credits.total_credits_retired

        if total <= Decimal("0"):
            return FoundationalCriterionResult(
                criterion_id="FC-4",
                criterion_name="Carbon credit quality (ICVCM CCP eligible)",
                status=CriterionStatus.NOT_MET,
                score=Decimal("0"),
                evidence_strength=EvidenceStrength.ABSENT,
                findings=["No carbon credits retired"],
                requirements=["Purchase and retire carbon credits from ICVCM-approved programs"],
            )

        # CCP-approved proportion (40 points)
        ccp_pct = _safe_pct(credits.ccp_approved_credits, total)
        if ccp_pct >= Decimal("100"):
            score += Decimal("40")
            findings.append("100% of credits are CCP-approved")
        elif ccp_pct >= Decimal("75"):
            score += Decimal("30")
            findings.append(f"{ccp_pct}% of credits are CCP-approved")
            requirements.append("Increase CCP-approved share to 100%")
        elif ccp_pct >= Decimal("50"):
            score += Decimal("20")
            findings.append(f"{ccp_pct}% of credits are CCP-approved")
            requirements.append("Majority of credits should be CCP-approved")
        else:
            score += Decimal("5")
            findings.append(f"Only {ccp_pct}% of credits are CCP-approved")
            requirements.append("Replace non-CCP credits with CCP-approved credits")

        # CCP principle compliance (30 points)
        principles_met = sum(1 for v in credits.ccp_compliance.values() if v)
        total_principles = len(ICVCM_CORE_CARBON_PRINCIPLES)
        if credits.ccp_compliance:
            compliance_pct = _safe_pct(_decimal(principles_met), _decimal(total_principles))
            if compliance_pct >= Decimal("90"):
                score += Decimal("30")
                findings.append(f"{principles_met}/{total_principles} CCP principles met")
            elif compliance_pct >= Decimal("70"):
                score += Decimal("20")
                findings.append(f"{principles_met}/{total_principles} CCP principles met")
            else:
                score += Decimal("10")
                findings.append(f"Only {principles_met}/{total_principles} CCP principles met")
            requirements.extend(
                f"Address CCP: {p['name']}" for p in ICVCM_CORE_CARBON_PRINCIPLES
                if not credits.ccp_compliance.get(p["id"], False)
            )
        else:
            requirements.append("Assess credits against all 10 ICVCM Core Carbon Principles")

        # Registry quality (15 points)
        recognized_registries = {"verra", "gold_standard", "american_carbon_registry", "climate_action_reserve", "puro_earth"}
        if credits.registries:
            recognized = sum(1 for r in credits.registries if r.lower() in recognized_registries)
            if recognized > 0:
                score += Decimal("15")
                findings.append(f"{recognized} recognized registry(ies) used")
            else:
                score += Decimal("5")
                findings.append("Credits from unrecognized registries")
                requirements.append("Use credits from recognized registries (Verra, Gold Standard, etc.)")
        else:
            requirements.append("Document credit registry sources")

        # Vintage (15 points)
        current_year = 2026
        vintage_age = current_year - credits.credit_vintage_year
        if vintage_age <= 3:
            score += Decimal("15")
            findings.append(f"Credit vintage is recent ({credits.credit_vintage_year})")
        elif vintage_age <= 5:
            score += Decimal("10")
            findings.append(f"Credit vintage is {vintage_age} years old")
        else:
            score += Decimal("3")
            findings.append(f"Credit vintage is {vintage_age} years old (outdated)")
            requirements.append("Use more recent vintage credits (within 3 years)")

        score = min(score, Decimal("100"))
        status = self._score_to_status(score)
        evidence = self._score_to_evidence(score)

        result = FoundationalCriterionResult(
            criterion_id="FC-4",
            criterion_name="Carbon credit quality (ICVCM CCP eligible)",
            status=status,
            score=score,
            evidence_strength=evidence,
            findings=findings,
            requirements=requirements,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Helpers for scoring
    # -------------------------------------------------------------------

    def _score_to_status(self, score: Decimal) -> CriterionStatus:
        """Convert a numeric score to a criterion status.

        Args:
            score: Score (0-100).

        Returns:
            CriterionStatus.
        """
        if score >= self.config.min_foundational_score:
            return CriterionStatus.MET
        elif score >= self.config.partially_met_threshold:
            return CriterionStatus.PARTIALLY_MET
        else:
            return CriterionStatus.NOT_MET

    def _score_to_evidence(self, score: Decimal) -> EvidenceStrength:
        """Convert a numeric score to evidence strength.

        Args:
            score: Score (0-100).

        Returns:
            EvidenceStrength.
        """
        if score >= Decimal("80"):
            return EvidenceStrength.STRONG
        elif score >= Decimal("50"):
            return EvidenceStrength.MODERATE
        elif score > Decimal("0"):
            return EvidenceStrength.WEAK
        else:
            return EvidenceStrength.ABSENT

    # -------------------------------------------------------------------
    # Tier Eligibility
    # -------------------------------------------------------------------

    def _assess_tier_eligibility(
        self,
        tier: VCMITier,
        foundational_met: bool,
        coverage_pct: Decimal,
        unabated: Decimal,
        credits_retired: Decimal,
    ) -> TierEligibility:
        """Assess eligibility for a specific VCMI tier.

        Args:
            tier: Target tier.
            foundational_met: Whether all foundational criteria are met.
            coverage_pct: Actual credit coverage percentage.
            unabated: Unabated emissions (tCO2e).
            credits_retired: Credits retired (tCO2e).

        Returns:
            TierEligibility result.
        """
        threshold = VCMI_TIER_THRESHOLDS[tier.value]
        eligible = foundational_met and coverage_pct >= threshold
        gap_pct = max(threshold - coverage_pct, Decimal("0"))
        credits_needed = Decimal("0")

        reasons: List[str] = []
        if not foundational_met:
            reasons.append("Foundational criteria not fully met")
        if coverage_pct < threshold:
            credits_needed = _round_val(
                (threshold - coverage_pct) / Decimal("100") * unabated, 2
            )
            reasons.append(
                f"Credit coverage {coverage_pct}% below {threshold}% threshold "
                f"(need {credits_needed} additional tCO2e)"
            )
        if eligible:
            reasons.append(f"All requirements met for {tier.value} tier")

        result = TierEligibility(
            tier=tier,
            eligible=eligible,
            foundational_criteria_met=foundational_met,
            credit_threshold_pct=threshold,
            actual_coverage_pct=coverage_pct,
            coverage_gap_pct=_round_val(gap_pct, 2),
            credits_needed=credits_needed,
            reasons=reasons,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Greenwashing Risk Assessment
    # -------------------------------------------------------------------

    def _assess_greenwashing_risk(
        self,
        emissions: EmissionsData,
        credits: CarbonCreditPortfolio,
        fc_results: List[FoundationalCriterionResult],
    ) -> Tuple[List[GreenwashingFlag], GreenwashingRiskLevel]:
        """Assess greenwashing risk.

        Args:
            emissions: Emissions data.
            credits: Credit portfolio.
            fc_results: Foundational criteria results.

        Returns:
            Tuple of (flags, overall_risk_level).
        """
        flags: List[GreenwashingFlag] = []
        triggered_count = 0
        critical_count = 0

        # GW-1: No reduction target
        has_target = emissions.has_sbti_target or emissions.target_reduction_pct > Decimal("0")
        gw1_triggered = not has_target and credits.total_credits_retired > Decimal("0")
        if gw1_triggered:
            triggered_count += 1
            critical_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-1", flag_name="No Reduction Target",
            triggered=gw1_triggered,
            severity=GreenwashingRiskLevel.CRITICAL if gw1_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[0]["description"],
            recommendation="Establish a science-aligned reduction target before making credit-based claims",
        ))

        # GW-2: Credits over reduction
        reduction = emissions.base_year_emissions - emissions.total_emissions
        gw2_triggered = (
            credits.total_credits_retired > Decimal("0")
            and reduction <= Decimal("0")
            and credits.total_credits_retired > emissions.total_emissions * Decimal("0.5")
        )
        if gw2_triggered:
            triggered_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-2", flag_name="Credits Over Reduction",
            triggered=gw2_triggered,
            severity=GreenwashingRiskLevel.HIGH if gw2_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[1]["description"],
            recommendation="Prioritize own-operations emission reductions over credit purchases",
        ))

        # GW-3: Low quality credits
        ccp_pct = _safe_pct(credits.ccp_approved_credits, credits.total_credits_retired) if credits.total_credits_retired > 0 else Decimal("100")
        gw3_triggered = credits.total_credits_retired > 0 and ccp_pct < Decimal("50")
        if gw3_triggered:
            triggered_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-3", flag_name="Low Quality Credits",
            triggered=gw3_triggered,
            severity=GreenwashingRiskLevel.HIGH if gw3_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[2]["description"],
            recommendation="Replace with ICVCM CCP-approved credits",
        ))

        # GW-4: Selective scope
        gw4_triggered = (
            emissions.scope3_emissions <= Decimal("0")
            and credits.total_credits_retired > Decimal("0")
        )
        if gw4_triggered:
            triggered_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-4", flag_name="Selective Scope",
            triggered=gw4_triggered,
            severity=GreenwashingRiskLevel.MEDIUM if gw4_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[3]["description"],
            recommendation="Include material Scope 3 categories in claim boundary",
        ))

        # GW-5: No progress
        gw5_triggered = reduction <= Decimal("0") and emissions.base_year_emissions > Decimal("0")
        if gw5_triggered:
            triggered_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-5", flag_name="No Progress",
            triggered=gw5_triggered,
            severity=GreenwashingRiskLevel.HIGH if gw5_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[4]["description"],
            recommendation="Demonstrate measurable year-on-year emission reductions",
        ))

        # GW-6: Misleading language (no ISO 14068-1 compliance)
        gw6_triggered = False  # Would need claim language analysis
        flags.append(GreenwashingFlag(
            flag_id="GW-6", flag_name="Misleading Language",
            triggered=gw6_triggered,
            severity=GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[5]["description"],
            recommendation="Ensure claim language aligns with VCMI terminology",
        ))

        # GW-7: Stale inventory
        inventory_age = emissions.reporting_year - emissions.inventory_year
        gw7_triggered = inventory_age > self.config.max_inventory_age_years
        if gw7_triggered:
            triggered_count += 1
        flags.append(GreenwashingFlag(
            flag_id="GW-7", flag_name="Stale Inventory",
            triggered=gw7_triggered,
            severity=GreenwashingRiskLevel.MEDIUM if gw7_triggered else GreenwashingRiskLevel.LOW,
            description=GREENWASHING_INDICATORS[6]["description"],
            recommendation=f"Update GHG inventory to within {self.config.max_inventory_age_years} years",
        ))

        # Set provenance hashes
        for flag in flags:
            flag.provenance_hash = _compute_hash(flag)

        # Overall risk level
        if critical_count > 0:
            overall = GreenwashingRiskLevel.CRITICAL
        elif triggered_count >= 3:
            overall = GreenwashingRiskLevel.HIGH
        elif triggered_count >= 1:
            overall = GreenwashingRiskLevel.MEDIUM
        else:
            overall = GreenwashingRiskLevel.LOW

        return flags, overall

    # -------------------------------------------------------------------
    # ICVCM Assessment
    # -------------------------------------------------------------------

    def assess_icvcm_compliance(
        self, credits: CarbonCreditPortfolio
    ) -> List[ICVCMAssessment]:
        """Assess carbon credits against ICVCM Core Carbon Principles.

        Args:
            credits: Carbon credit portfolio.

        Returns:
            List of ICVCMAssessment.
        """
        assessments: List[ICVCMAssessment] = []

        for principle in ICVCM_CORE_CARBON_PRINCIPLES:
            pid = principle["id"]
            compliant = credits.ccp_compliance.get(pid, False)
            score = Decimal("100") if compliant else Decimal("0")
            evidence = "Confirmed compliant" if compliant else "Not assessed or non-compliant"

            assessment = ICVCMAssessment(
                principle_id=pid,
                principle_name=principle["name"],
                compliant=compliant,
                score=score,
                evidence=evidence,
            )
            assessment.provenance_hash = _compute_hash(assessment)
            assessments.append(assessment)

        return assessments

    # -------------------------------------------------------------------
    # ISO 14068-1 Comparison
    # -------------------------------------------------------------------

    def compare_iso_14068(
        self,
        emissions: EmissionsData,
        fc_results: List[FoundationalCriterionResult],
    ) -> List[ISOComparison]:
        """Compare VCMI compliance against ISO 14068-1 requirements.

        Args:
            emissions: Emissions data.
            fc_results: Foundational criteria results.

        Returns:
            List of ISOComparison.
        """
        fc_scores: Dict[str, Decimal] = {}
        for fc in fc_results:
            fc_scores[fc.criterion_id] = fc.score

        comparisons: List[ISOComparison] = []

        # ISO-1: GHG Inventory -> Covered by FC-3
        fc3_met = fc_scores.get("FC-3", Decimal("0")) >= self.config.min_foundational_score
        comparisons.append(ISOComparison(
            requirement_id="ISO-1", requirement_name="GHG Inventory",
            vcmi_overlap=True, met_by_vcmi_compliance=fc3_met,
            gap="" if fc3_met else "Complete GHG inventory per ISO 14064-1",
        ))

        # ISO-2: Reduction Plan -> Covered by FC-1 + FC-2
        fc1_met = fc_scores.get("FC-1", Decimal("0")) >= self.config.min_foundational_score
        fc2_met = fc_scores.get("FC-2", Decimal("0")) >= self.config.min_foundational_score
        comparisons.append(ISOComparison(
            requirement_id="ISO-2", requirement_name="Reduction Plan",
            vcmi_overlap=True, met_by_vcmi_compliance=fc1_met and fc2_met,
            gap="" if (fc1_met and fc2_met) else "Document reduction plan with interim targets",
        ))

        # ISO-3: Offsetting -> Partially covered by VCMI
        comparisons.append(ISOComparison(
            requirement_id="ISO-3", requirement_name="Offsetting",
            vcmi_overlap=True, met_by_vcmi_compliance=True,
            gap="VCMI requires credits for unabated emissions; ISO requires maximum reduction first",
        ))

        # ISO-4: Credit Quality -> Covered by FC-4
        fc4_met = fc_scores.get("FC-4", Decimal("0")) >= self.config.min_foundational_score
        comparisons.append(ISOComparison(
            requirement_id="ISO-4", requirement_name="Credit Quality",
            vcmi_overlap=True, met_by_vcmi_compliance=fc4_met,
            gap="" if fc4_met else "Ensure credits meet ISO 14064-2/3 requirements",
        ))

        # ISO-5: Time-bound Commitment -> Not directly in VCMI
        comparisons.append(ISOComparison(
            requirement_id="ISO-5", requirement_name="Time-bound Commitment",
            vcmi_overlap=False, met_by_vcmi_compliance=False,
            gap="VCMI does not require explicit time-bound carbon neutrality commitment; add separately for ISO compliance",
        ))

        # ISO-6: Public Disclosure -> Covered by FC-3
        comparisons.append(ISOComparison(
            requirement_id="ISO-6", requirement_name="Public Disclosure",
            vcmi_overlap=True, met_by_vcmi_compliance=fc3_met,
            gap="" if fc3_met else "Publicly disclose inventory, reductions, and offsets",
        ))

        # ISO-7: Third-Party Verification -> Not directly in VCMI
        comparisons.append(ISOComparison(
            requirement_id="ISO-7", requirement_name="Third-Party Verification",
            vcmi_overlap=False, met_by_vcmi_compliance=False,
            gap="VCMI does not mandate third-party verification of the overall claim; obtain independent verification for ISO compliance",
        ))

        for comp in comparisons:
            comp.provenance_hash = _compute_hash(comp)

        return comparisons

    # -------------------------------------------------------------------
    # Gap Analysis
    # -------------------------------------------------------------------

    def _build_gap_analysis(
        self,
        current_tier: VCMITier,
        tier_results: List[TierEligibility],
        fc_results: List[FoundationalCriterionResult],
        unabated: Decimal,
    ) -> Optional[GapToNextTier]:
        """Build gap analysis for the next tier upgrade.

        Args:
            current_tier: Current highest eligible tier.
            tier_results: All tier eligibility results.
            fc_results: Foundational criteria results.
            unabated: Unabated emissions.

        Returns:
            GapToNextTier or None if already Platinum.
        """
        tier_order = [VCMITier.NOT_ELIGIBLE, VCMITier.SILVER, VCMITier.GOLD, VCMITier.PLATINUM]
        current_idx = tier_order.index(current_tier)
        if current_idx >= len(tier_order) - 1:
            return None  # Already Platinum

        next_tier = tier_order[current_idx + 1]
        gaps: List[str] = []
        criteria_gaps: List[str] = []
        additional_credits = Decimal("0")

        # Check foundational criteria gaps
        for fc in fc_results:
            if fc.status != CriterionStatus.MET:
                criteria_gaps.extend(fc.requirements)
                gaps.append(f"{fc.criterion_id}: {fc.criterion_name} - {fc.status.value}")

        # Check credit coverage gap
        for tr in tier_results:
            if tr.tier == next_tier and not tr.eligible:
                additional_credits = tr.credits_needed
                if tr.coverage_gap_pct > Decimal("0"):
                    gaps.append(f"Need {tr.coverage_gap_pct}% more credit coverage ({additional_credits} tCO2e)")

        estimated_cost = _round_val(
            additional_credits * self.config.credit_price_usd_per_tonne, 2
        )

        result = GapToNextTier(
            current_tier=current_tier,
            next_tier=next_tier,
            gaps=gaps,
            additional_credits_needed=additional_credits,
            criteria_gaps=criteria_gaps,
            estimated_cost_usd=estimated_cost,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Full Validation Pipeline
    # -------------------------------------------------------------------

    def validate(
        self,
        emissions: EmissionsData,
        credits: CarbonCreditPortfolio,
        entity_name: str = "",
    ) -> VCMIResult:
        """Run complete VCMI validation.

        Assesses all four foundational criteria, determines tier eligibility,
        checks ICVCM compliance, evaluates greenwashing risk, and compares
        with ISO 14068-1.

        Args:
            emissions: Emissions data.
            credits: Carbon credit portfolio.
            entity_name: Optional entity name.

        Returns:
            Complete VCMIResult.
        """
        logger.info("Running VCMI validation for %s, year %d", entity_name, emissions.reporting_year)

        # Auto-calculate total and unabated if not set
        if emissions.total_emissions <= Decimal("0"):
            emissions.total_emissions = (
                emissions.scope1_emissions + emissions.scope2_emissions + emissions.scope3_emissions
            )
        if emissions.unabated_emissions <= Decimal("0"):
            emissions.unabated_emissions = emissions.total_emissions

        emissions.provenance_hash = _compute_hash(emissions)
        credits.provenance_hash = _compute_hash(credits)

        # Step 1: Assess foundational criteria
        fc1 = self._assess_criterion_1(emissions)
        fc2 = self._assess_criterion_2(emissions)
        fc3 = self._assess_criterion_3(emissions)
        fc4 = self._assess_criterion_4(credits)
        fc_results = [fc1, fc2, fc3, fc4]

        all_met = all(fc.status == CriterionStatus.MET for fc in fc_results)
        overall_score = _round_val(
            sum(fc.score for fc in fc_results) / Decimal("4"), 2
        )

        # Step 2: Calculate credit coverage
        coverage_pct = _safe_pct(
            credits.total_credits_retired, emissions.unabated_emissions
        )

        # Step 3: Assess tier eligibility
        tier_results: List[TierEligibility] = []
        highest_tier = VCMITier.NOT_ELIGIBLE

        for tier in [VCMITier.SILVER, VCMITier.GOLD, VCMITier.PLATINUM]:
            te = self._assess_tier_eligibility(
                tier, all_met, coverage_pct,
                emissions.unabated_emissions, credits.total_credits_retired,
            )
            tier_results.append(te)
            if te.eligible:
                highest_tier = tier

        # Step 4: Evidence scores
        evidence_scores = {fc.criterion_id: str(fc.score) for fc in fc_results}

        # Step 5: Gap analysis
        gap = self._build_gap_analysis(highest_tier, tier_results, fc_results, emissions.unabated_emissions)

        # Step 6: Greenwashing risk
        gw_flags, gw_level = self._assess_greenwashing_risk(emissions, credits, fc_results)

        # Step 7: ICVCM assessment
        icvcm = self.assess_icvcm_compliance(credits)

        # Step 8: ISO 14068-1 comparison
        iso_comp = self.compare_iso_14068(emissions, fc_results)

        # Step 9: Build recommendations
        recommendations = self._build_recommendations(
            fc_results, highest_tier, gw_flags, coverage_pct
        )

        result = VCMIResult(
            entity_name=entity_name,
            reporting_year=emissions.reporting_year,
            foundational_criteria_results=fc_results,
            all_foundational_met=all_met,
            foundational_overall_score=overall_score,
            tier_eligibility=tier_results,
            highest_eligible_tier=highest_tier,
            evidence_scores=evidence_scores,
            gaps_to_next_tier=gap,
            greenwashing_flags=gw_flags,
            greenwashing_risk_level=gw_level,
            icvcm_assessment=icvcm,
            iso_comparison=iso_comp,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "VCMI validation complete for %s: tier=%s, foundational=%s, GW risk=%s",
            entity_name, highest_tier.value, all_met, gw_level.value,
        )
        return result

    def _build_recommendations(
        self,
        fc_results: List[FoundationalCriterionResult],
        highest_tier: VCMITier,
        gw_flags: List[GreenwashingFlag],
        coverage_pct: Decimal,
    ) -> List[str]:
        """Build prioritized recommendations.

        Args:
            fc_results: Foundational criteria results.
            highest_tier: Current highest tier.
            gw_flags: Greenwashing flags.
            coverage_pct: Credit coverage percentage.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # Priority 1: Address failing criteria
        for fc in fc_results:
            if fc.status == CriterionStatus.NOT_MET:
                recs.append(f"[CRITICAL] {fc.criterion_id}: {', '.join(fc.requirements[:2])}")
            elif fc.status == CriterionStatus.PARTIALLY_MET:
                recs.append(f"[HIGH] {fc.criterion_id}: {', '.join(fc.requirements[:2])}")

        # Priority 2: Address greenwashing risks
        for flag in gw_flags:
            if flag.triggered:
                recs.append(f"[{flag.severity.value.upper()}] {flag.flag_name}: {flag.recommendation}")

        # Priority 3: Tier advancement
        if highest_tier == VCMITier.NOT_ELIGIBLE:
            recs.append("[INFO] Focus on meeting all 4 foundational criteria before pursuing any claim tier")
        elif highest_tier == VCMITier.SILVER:
            recs.append("[INFO] To reach Gold: increase credit coverage from "
                        f"{coverage_pct}% to 60% of unabated emissions")
        elif highest_tier == VCMITier.GOLD:
            recs.append("[INFO] To reach Platinum: increase credit coverage from "
                        f"{coverage_pct}% to 100% of unabated emissions")

        return recs

    # -------------------------------------------------------------------
    # Annual Re-validation
    # -------------------------------------------------------------------

    def revalidate(
        self,
        previous_result: VCMIResult,
        new_emissions: EmissionsData,
        new_credits: CarbonCreditPortfolio,
        entity_name: str = "",
    ) -> Dict[str, Any]:
        """Perform annual re-validation comparing with previous results.

        Args:
            previous_result: Previous year's VCMIResult.
            new_emissions: New year's emissions data.
            new_credits: New year's credit portfolio.
            entity_name: Entity name.

        Returns:
            Dictionary with re-validation results and year-over-year changes.
        """
        new_result = self.validate(new_emissions, new_credits, entity_name)

        tier_change = "no_change"
        if new_result.highest_eligible_tier != previous_result.highest_eligible_tier:
            tier_order = [VCMITier.NOT_ELIGIBLE, VCMITier.SILVER, VCMITier.GOLD, VCMITier.PLATINUM]
            old_idx = tier_order.index(previous_result.highest_eligible_tier)
            new_idx = tier_order.index(new_result.highest_eligible_tier)
            tier_change = "upgraded" if new_idx > old_idx else "downgraded"

        score_change = new_result.foundational_overall_score - previous_result.foundational_overall_score

        revalidation = {
            "new_result": new_result.model_dump(mode="json"),
            "year_over_year": {
                "previous_tier": previous_result.highest_eligible_tier.value,
                "current_tier": new_result.highest_eligible_tier.value,
                "tier_change": tier_change,
                "previous_score": str(previous_result.foundational_overall_score),
                "current_score": str(new_result.foundational_overall_score),
                "score_change": str(score_change),
                "previous_gw_risk": previous_result.greenwashing_risk_level.value,
                "current_gw_risk": new_result.greenwashing_risk_level.value,
            },
            "provenance_hash": _compute_hash({
                "prev": previous_result.provenance_hash,
                "new": new_result.provenance_hash,
            }),
        }

        logger.info(
            "Re-validation complete: tier %s -> %s (%s), score delta=%.1f",
            previous_result.highest_eligible_tier.value,
            new_result.highest_eligible_tier.value,
            tier_change, float(score_change),
        )
        return revalidation

    def clear(self) -> None:
        """No persistent state to clear; engine is stateless."""
        logger.info("VCMIValidationEngine cleared (stateless)")
