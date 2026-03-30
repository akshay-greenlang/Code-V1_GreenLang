# -*- coding: utf-8 -*-
"""
ComparativeClaimsEngine - PACK-018 EU Green Claims Prep Engine 5
=================================================================

Validates comparative environmental claims and improvement claims per the
EU Green Claims Directive (Directive 2024/825 amending Directive 2005/29/EC
and the proposed Directive on substantiation and communication of explicit
environmental claims COM/2023/166).

Under Articles 3(4) and 5, comparative environmental claims must meet
stringent requirements to prevent misleading comparisons. This engine
validates year-over-year improvements, product-versus-product comparisons,
industry benchmark comparisons, and regulatory baseline comparisons.

Key Regulatory Requirements:
    - Article 3(4): Comparative environmental claims shall use equivalent
      information and data for the assessment of the environmental impact,
      processes, or social aspects being compared.
    - Article 5(1): Environmental claims about future environmental
      performance shall only be made where they include clear, objective,
      publicly available and verifiable commitments, set out in a detailed
      and realistic implementation plan.
    - Article 5(2): The implementation plan shall include measurable and
      time-bound targets, allocation of resources, and be subject to
      independent third-party monitoring.
    - Article 5(3): Where the achievement of the targets depends on
      carbon offsets, these shall be clearly disclosed and the claim
      shall not be communicated as if the future performance is certain.

Methodology Validation Rules:
    - Comparison boundaries must be equivalent (same scope, same functional unit)
    - Baseline and current data must use the same methodology
    - Improvement percentages must be calculated from auditable data
    - Future claims must have binding targets and independent monitoring
    - Time horizons must be realistic and specific

Zero-Hallucination:
    - Improvement percentages use deterministic Decimal arithmetic
    - Methodology validation uses predefined rule sets
    - Compliance scoring uses deterministic weighted sums
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Safely divide two Decimal numbers, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComparisonType(str, Enum):
    """Type of comparative environmental claim per Article 3(4).

    Each type represents a different basis for comparison, with distinct
    validation requirements and evidentiary standards.
    """
    YEAR_OVER_YEAR = "year_over_year"
    PRODUCT_VS_PRODUCT = "product_vs_product"
    VS_INDUSTRY_AVERAGE = "vs_industry_average"
    VS_REGULATORY_BASELINE = "vs_regulatory_baseline"
    IMPROVEMENT_OVER_TIME = "improvement_over_time"

class FutureClaimStatus(str, Enum):
    """Validation status for future environmental performance claims.

    Per Article 5, future claims must meet specific substantiation
    requirements including binding targets and independent monitoring.
    """
    VALIDATED = "validated"
    CONDITIONAL = "conditional"
    UNSUBSTANTIATED = "unsubstantiated"
    PROHIBITED = "prohibited"

class MethodologyStatus(str, Enum):
    """Methodology validation outcome for comparative claims.

    Reflects whether the methodology used for comparison meets
    the equivalence requirements of Article 3(4).
    """
    EQUIVALENT = "equivalent"
    PARTIALLY_EQUIVALENT = "partially_equivalent"
    NON_EQUIVALENT = "non_equivalent"
    INSUFFICIENT_DATA = "insufficient_data"

class ClaimCompliance(str, Enum):
    """Overall compliance assessment for a comparative claim.

    Summarizes the regulatory compliance status after all
    validation checks have been applied.
    """
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_AMENDMENT = "requires_amendment"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Validation rules for each comparison type, defining mandatory requirements.
COMPARISON_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    ComparisonType.YEAR_OVER_YEAR: {
        "name": "Year-over-Year Improvement",
        "description": "Comparison of environmental performance between two "
                       "or more reporting periods for the same entity",
        "mandatory_fields": [
            "baseline_value", "current_value", "baseline_year",
            "current_year", "unit", "methodology",
        ],
        "rules": [
            "Baseline and current periods must use identical methodology",
            "Organizational boundaries must be consistent across periods",
            "Scope of measurement must be identical",
            "Restatement required if methodology changes",
            "Minimum 12-month reporting periods required",
        ],
        "minimum_data_years": 2,
        "requires_restatement_policy": True,
    },
    ComparisonType.PRODUCT_VS_PRODUCT: {
        "name": "Product-versus-Product Comparison",
        "description": "Comparison of environmental performance between "
                       "two or more products using equivalent functional units",
        "mandatory_fields": [
            "baseline_value", "current_value", "unit", "methodology",
        ],
        "rules": [
            "Products must serve the same function (functional unit equivalence)",
            "Life-cycle stages included must be identical",
            "System boundaries must be equivalent",
            "Data quality requirements must be consistent",
            "Geographic scope must be specified and comparable",
        ],
        "minimum_data_years": 1,
        "requires_restatement_policy": False,
    },
    ComparisonType.VS_INDUSTRY_AVERAGE: {
        "name": "Comparison Against Industry Average",
        "description": "Comparison of entity or product performance against "
                       "an industry benchmark or sector average",
        "mandatory_fields": [
            "baseline_value", "current_value", "unit", "methodology",
        ],
        "rules": [
            "Industry average source must be publicly available",
            "Industry average must be from a recognized body (ISO, EU JRC, etc.)",
            "Comparison scope must match industry average scope",
            "Industry average vintage must be within 3 years",
            "Sample size and representativeness must be disclosed",
        ],
        "minimum_data_years": 1,
        "requires_restatement_policy": False,
    },
    ComparisonType.VS_REGULATORY_BASELINE: {
        "name": "Comparison Against Regulatory Baseline",
        "description": "Comparison of performance against a regulatory "
                       "requirement, standard, or legal minimum",
        "mandatory_fields": [
            "baseline_value", "current_value", "unit", "methodology",
        ],
        "rules": [
            "Regulatory baseline must be currently in force",
            "The claim must not present legal compliance as distinction",
            "Exceeding the baseline must be quantified objectively",
            "The specific regulation/standard must be cited",
            "Measurement methodology must follow the regulatory standard",
        ],
        "minimum_data_years": 1,
        "requires_restatement_policy": False,
    },
    ComparisonType.IMPROVEMENT_OVER_TIME: {
        "name": "Improvement Over Time (Trend)",
        "description": "Claim of progressive improvement across multiple "
                       "reporting periods, demonstrating a trend",
        "mandatory_fields": [
            "baseline_value", "current_value", "baseline_year",
            "current_year", "unit", "methodology",
        ],
        "rules": [
            "Minimum three data points required to establish a trend",
            "Consistent methodology across all data points",
            "Outliers and anomalies must be disclosed",
            "Base year must be justified and representative",
            "Rate of improvement must be calculated transparently",
        ],
        "minimum_data_years": 3,
        "requires_restatement_policy": True,
    },
}

# Future claim validation criteria per Article 5.
FUTURE_CLAIM_CRITERIA: Dict[str, Dict[str, Any]] = {
    "binding_targets": {
        "description": "Clear, objective, publicly available and verifiable "
                       "commitments with measurable targets",
        "weight": Decimal("0.30"),
        "required_for_validation": True,
    },
    "implementation_plan": {
        "description": "Detailed and realistic implementation plan with "
                       "time-bound milestones and resource allocation",
        "weight": Decimal("0.25"),
        "required_for_validation": True,
    },
    "independent_monitoring": {
        "description": "Independent third-party monitoring of progress "
                       "toward the stated environmental targets",
        "weight": Decimal("0.25"),
        "required_for_validation": True,
    },
    "offset_disclosure": {
        "description": "Clear disclosure of reliance on carbon offsets, "
                       "with offsets not presented as certainties",
        "weight": Decimal("0.10"),
        "required_for_validation": False,
    },
    "progress_reporting": {
        "description": "Regular public reporting on progress toward "
                       "the claimed future environmental performance",
        "weight": Decimal("0.10"),
        "required_for_validation": False,
    },
}

# Methodology equivalence scoring weights.
METHODOLOGY_WEIGHTS: Dict[str, Decimal] = {
    "scope_equivalence": Decimal("0.25"),
    "boundary_equivalence": Decimal("0.20"),
    "data_quality_equivalence": Decimal("0.20"),
    "temporal_equivalence": Decimal("0.15"),
    "functional_unit_equivalence": Decimal("0.20"),
}

# Minimum improvement thresholds to qualify as a meaningful claim.
MINIMUM_IMPROVEMENT_THRESHOLDS: Dict[str, Decimal] = {
    "ghg_emissions": Decimal("5.0"),
    "energy_consumption": Decimal("3.0"),
    "water_usage": Decimal("3.0"),
    "waste_generation": Decimal("5.0"),
    "recycled_content": Decimal("5.0"),
    "renewable_energy": Decimal("5.0"),
    "default": Decimal("3.0"),
}

# Maximum credible annual improvement rates (as percentage).
MAX_CREDIBLE_ANNUAL_IMPROVEMENT: Dict[str, Decimal] = {
    "ghg_emissions": Decimal("15.0"),
    "energy_consumption": Decimal("10.0"),
    "water_usage": Decimal("10.0"),
    "waste_generation": Decimal("12.0"),
    "default": Decimal("10.0"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ComparativeClaim(BaseModel):
    """A comparative environmental claim subject to validation.

    Represents a single comparative or improvement claim as defined
    under Article 3(4) of the Green Claims Directive, including all
    data necessary for substantiation assessment.
    """
    claim_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the claim",
    )
    claim_text: str = Field(
        ...,
        min_length=10,
        description="Full text of the comparative claim as communicated",
    )
    comparison_type: ComparisonType = Field(
        ...,
        description="Type of comparison being made",
    )
    baseline_value: Decimal = Field(
        ...,
        description="Baseline/reference value for comparison",
    )
    current_value: Decimal = Field(
        ...,
        description="Current or compared value",
    )
    baseline_year: Optional[int] = Field(
        None,
        ge=1990,
        le=2030,
        description="Year of the baseline measurement",
    )
    current_year: Optional[int] = Field(
        None,
        ge=1990,
        le=2030,
        description="Year of the current measurement",
    )
    unit: str = Field(
        ...,
        min_length=1,
        description="Unit of measurement (e.g., tCO2e, kWh, m3)",
    )
    methodology: str = Field(
        ...,
        min_length=5,
        description="Methodology used for measurement",
    )
    improvement_pct: Decimal = Field(
        Decimal("0"),
        description="Claimed improvement percentage",
    )
    has_binding_target: bool = Field(
        False,
        description="Whether the claim is backed by a binding target",
    )
    independent_monitoring: bool = Field(
        False,
        description="Whether independent third-party monitoring is in place",
    )
    metric_category: Optional[str] = Field(
        None,
        description="Category of environmental metric (ghg_emissions, "
                    "energy_consumption, water_usage, waste_generation)",
    )
    scope_description: Optional[str] = Field(
        None,
        description="Description of the measurement scope and boundaries",
    )
    data_source: Optional[str] = Field(
        None,
        description="Source of the data used for the claim",
    )

    @field_validator("current_year")
    @classmethod
    def current_year_after_baseline(cls, v: Optional[int], info: Any) -> Optional[int]:
        """Validate that current_year is not before baseline_year."""
        baseline = info.data.get("baseline_year")
        if v is not None and baseline is not None and v < baseline:
            raise ValueError(
                f"current_year ({v}) cannot be before baseline_year ({baseline})"
            )
        return v

class FutureClaimInput(BaseModel):
    """Input for future environmental claim assessment per Article 5.

    Captures the details of a forward-looking environmental claim
    and its supporting substantiation elements.
    """
    claim_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the future claim",
    )
    claim_text: str = Field(
        ...,
        min_length=10,
        description="Full text of the future environmental claim",
    )
    target_year: Optional[int] = Field(
        None,
        ge=2025,
        le=2060,
        description="Target year for the claimed future performance",
    )
    has_binding_targets: bool = Field(
        False,
        description="Whether binding commitments with measurable targets exist",
    )
    has_implementation_plan: bool = Field(
        False,
        description="Whether a detailed implementation plan is in place",
    )
    has_independent_monitoring: bool = Field(
        False,
        description="Whether independent third-party monitoring is engaged",
    )
    has_offset_disclosure: bool = Field(
        False,
        description="Whether reliance on offsets is disclosed",
    )
    has_progress_reporting: bool = Field(
        False,
        description="Whether regular progress reporting is committed to",
    )
    relies_on_offsets: bool = Field(
        False,
        description="Whether the claim relies on carbon offsets",
    )
    offset_percentage: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of the claim relying on carbon offsets",
    )
    targets: List[str] = Field(
        default_factory=list,
        description="List of specific measurable targets supporting the claim",
    )
    monitoring_details: Optional[str] = Field(
        None,
        description="Details of the independent monitoring arrangement",
    )

class MethodologyAssessment(BaseModel):
    """Input for methodology equivalence assessment.

    Captures the specific attributes needed to determine whether
    two methodologies are equivalent for comparison purposes.
    """
    scope_match: bool = Field(
        False,
        description="Whether the scope of measurement is equivalent",
    )
    boundary_match: bool = Field(
        False,
        description="Whether system boundaries are equivalent",
    )
    data_quality_match: bool = Field(
        False,
        description="Whether data quality requirements are equivalent",
    )
    temporal_match: bool = Field(
        False,
        description="Whether time periods are equivalent",
    )
    functional_unit_match: bool = Field(
        False,
        description="Whether functional units are equivalent",
    )
    methodology_name: Optional[str] = Field(
        None,
        description="Name of the methodology or standard used",
    )
    methodology_version: Optional[str] = Field(
        None,
        description="Version of the methodology or standard",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ComparativeClaimsEngine:
    """Engine for validating comparative and improvement claims.

    Validates comparative environmental claims per Articles 3(4) and 5
    of the EU Green Claims Directive. Applies deterministic validation
    rules to assess whether a comparative claim meets the requirements
    for equivalent methodology, meaningful improvement, credible
    improvement rates, and proper future claim substantiation.

    Attributes:
        engine_id: Unique identifier for this engine instance.
        version: Module version string.

    Example:
        >>> engine = ComparativeClaimsEngine()
        >>> claim = ComparativeClaim(
        ...     claim_text="We reduced CO2 emissions by 20% since 2020",
        ...     comparison_type=ComparisonType.YEAR_OVER_YEAR,
        ...     baseline_value=Decimal("1000"),
        ...     current_value=Decimal("800"),
        ...     baseline_year=2020,
        ...     current_year=2024,
        ...     unit="tCO2e",
        ...     methodology="GHG Protocol Corporate Standard",
        ...     improvement_pct=Decimal("20.0"),
        ... )
        >>> result = engine.validate_comparative_claim(claim)
        >>> assert "provenance_hash" in result
    """

    def __init__(self) -> None:
        """Initialize ComparativeClaimsEngine."""
        self.engine_id: str = _new_uuid()
        self.version: str = _MODULE_VERSION
        logger.info(
            "ComparativeClaimsEngine initialized | engine_id=%s version=%s",
            self.engine_id,
            self.version,
        )

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def validate_comparative_claim(self, claim: ComparativeClaim) -> Dict[str, Any]:
        """Validate a comparative environmental claim per Article 3(4).

        Applies a comprehensive set of checks including mandatory field
        presence, methodology equivalence, improvement magnitude, and
        credibility of improvement rate.

        Args:
            claim: The comparative claim to validate.

        Returns:
            Dict with validation results, findings, compliance status,
            overall score, and provenance_hash.
        """
        logger.info(
            "Validating comparative claim | claim_id=%s type=%s",
            claim.claim_id,
            claim.comparison_type.value,
        )
        timestamp = utcnow()
        findings: List[Dict[str, Any]] = []
        scores: Dict[str, Decimal] = {}

        # Step 1: Check mandatory fields
        mandatory_result = self._check_mandatory_fields(claim)
        findings.append(mandatory_result)
        scores["mandatory_fields"] = mandatory_result["score"]

        # Step 2: Calculate actual improvement
        improvement = self._calculate_improvement_internal(claim)
        findings.append(improvement)
        scores["improvement_calculation"] = improvement["score"]

        # Step 3: Check improvement threshold
        threshold_result = self._check_improvement_threshold(claim, improvement)
        findings.append(threshold_result)
        scores["improvement_threshold"] = threshold_result["score"]

        # Step 4: Check credibility of improvement rate
        credibility_result = self._check_credibility(claim, improvement)
        findings.append(credibility_result)
        scores["credibility"] = credibility_result["score"]

        # Step 5: Check year requirements
        year_result = self._check_year_requirements(claim)
        findings.append(year_result)
        scores["year_requirements"] = year_result["score"]

        # Calculate overall score
        overall_score = self._calculate_overall_score(scores)

        # Determine compliance status
        compliance = self._determine_compliance(overall_score, findings)

        result = {
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "comparison_type": claim.comparison_type.value,
            "timestamp": str(timestamp),
            "overall_score": str(_round_val(overall_score, 2)),
            "compliance_status": compliance,
            "dimension_scores": {k: str(_round_val(v, 2)) for k, v in scores.items()},
            "findings": findings,
            "requirements": COMPARISON_REQUIREMENTS.get(
                claim.comparison_type, {}
            ).get("rules", []),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Comparative claim validated | claim_id=%s compliance=%s score=%s",
            claim.claim_id,
            compliance,
            str(_round_val(overall_score, 2)),
        )
        return result

    def assess_future_claim(
        self,
        claim_text: str,
        targets: List[str],
        monitoring: Optional[str] = None,
        *,
        has_binding_targets: bool = False,
        has_implementation_plan: bool = False,
        has_independent_monitoring: bool = False,
        has_offset_disclosure: bool = False,
        has_progress_reporting: bool = False,
        relies_on_offsets: bool = False,
        offset_percentage: Decimal = Decimal("0"),
        target_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Assess a future environmental performance claim per Article 5.

        Evaluates whether a future claim meets the substantiation
        requirements, including binding targets, implementation plans,
        independent monitoring, and offset disclosure.

        Args:
            claim_text: Full text of the future claim.
            targets: List of specific measurable targets.
            monitoring: Details of independent monitoring arrangement.
            has_binding_targets: Whether binding commitments exist.
            has_implementation_plan: Whether implementation plan exists.
            has_independent_monitoring: Whether monitoring is in place.
            has_offset_disclosure: Whether offset reliance is disclosed.
            has_progress_reporting: Whether progress reporting is committed.
            relies_on_offsets: Whether the claim relies on offsets.
            offset_percentage: Percentage relying on offsets.
            target_year: Target year for claimed performance.

        Returns:
            Dict with future claim status, criteria scores, findings,
            and provenance_hash.
        """
        logger.info("Assessing future claim | text_length=%d", len(claim_text))
        timestamp = utcnow()
        claim_id = _new_uuid()

        criteria_scores: Dict[str, Decimal] = {}
        findings: List[Dict[str, str]] = []

        # Evaluate each criterion
        if has_binding_targets and len(targets) > 0:
            criteria_scores["binding_targets"] = Decimal("1.0")
            findings.append({
                "criterion": "binding_targets",
                "status": "met",
                "detail": f"{len(targets)} binding target(s) identified",
            })
        elif len(targets) > 0:
            criteria_scores["binding_targets"] = Decimal("0.5")
            findings.append({
                "criterion": "binding_targets",
                "status": "partial",
                "detail": "Targets exist but are not confirmed as binding",
            })
        else:
            criteria_scores["binding_targets"] = Decimal("0")
            findings.append({
                "criterion": "binding_targets",
                "status": "not_met",
                "detail": "No measurable targets provided for future claim",
            })

        if has_implementation_plan:
            criteria_scores["implementation_plan"] = Decimal("1.0")
            findings.append({
                "criterion": "implementation_plan",
                "status": "met",
                "detail": "Detailed implementation plan is in place",
            })
        else:
            criteria_scores["implementation_plan"] = Decimal("0")
            findings.append({
                "criterion": "implementation_plan",
                "status": "not_met",
                "detail": "No implementation plan provided (required by Art. 5(1))",
            })

        if has_independent_monitoring:
            criteria_scores["independent_monitoring"] = Decimal("1.0")
            detail = f"Independent monitoring: {monitoring}" if monitoring else \
                "Independent monitoring confirmed"
            findings.append({
                "criterion": "independent_monitoring",
                "status": "met",
                "detail": detail,
            })
        else:
            criteria_scores["independent_monitoring"] = Decimal("0")
            findings.append({
                "criterion": "independent_monitoring",
                "status": "not_met",
                "detail": "No independent third-party monitoring (required by Art. 5(2))",
            })

        if relies_on_offsets:
            if has_offset_disclosure:
                criteria_scores["offset_disclosure"] = Decimal("1.0")
                findings.append({
                    "criterion": "offset_disclosure",
                    "status": "met",
                    "detail": f"Offset reliance ({offset_percentage}%) is disclosed",
                })
            else:
                criteria_scores["offset_disclosure"] = Decimal("0")
                findings.append({
                    "criterion": "offset_disclosure",
                    "status": "not_met",
                    "detail": "Claim relies on offsets but does not disclose this",
                })
        else:
            criteria_scores["offset_disclosure"] = Decimal("1.0")
            findings.append({
                "criterion": "offset_disclosure",
                "status": "not_applicable",
                "detail": "Claim does not rely on carbon offsets",
            })

        if has_progress_reporting:
            criteria_scores["progress_reporting"] = Decimal("1.0")
            findings.append({
                "criterion": "progress_reporting",
                "status": "met",
                "detail": "Regular progress reporting is committed",
            })
        else:
            criteria_scores["progress_reporting"] = Decimal("0.0")
            findings.append({
                "criterion": "progress_reporting",
                "status": "not_met",
                "detail": "No commitment to regular progress reporting",
            })

        # Calculate weighted score
        weighted_score = Decimal("0")
        for criterion_key, score in criteria_scores.items():
            weight = FUTURE_CLAIM_CRITERIA.get(
                criterion_key, {}
            ).get("weight", Decimal("0.10"))
            weighted_score += score * weight

        weighted_score = _round_val(weighted_score * Decimal("100"), 2)

        # Determine status
        status = self._determine_future_claim_status(
            criteria_scores, relies_on_offsets, offset_percentage,
        )

        # Check for prohibited conditions
        prohibited_reasons: List[str] = []
        if relies_on_offsets and offset_percentage > Decimal("50"):
            prohibited_reasons.append(
                "Claim relies >50% on offsets and cannot be presented as certain"
            )
        if target_year is not None and target_year > 2050:
            prohibited_reasons.append(
                "Target year beyond 2050 lacks credibility without interim targets"
            )

        if prohibited_reasons:
            status = FutureClaimStatus.PROHIBITED.value

        result = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "timestamp": str(timestamp),
            "future_claim_status": status,
            "weighted_score": str(weighted_score),
            "criteria_scores": {
                k: str(_round_val(v, 2)) for k, v in criteria_scores.items()
            },
            "findings": findings,
            "targets_count": len(targets),
            "targets": targets,
            "relies_on_offsets": relies_on_offsets,
            "offset_percentage": str(offset_percentage),
            "prohibited_reasons": prohibited_reasons,
            "target_year": target_year,
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Future claim assessed | claim_id=%s status=%s score=%s",
            claim_id,
            status,
            str(weighted_score),
        )
        return result

    def calculate_improvement(
        self,
        baseline: Decimal,
        current: Decimal,
        *,
        metric_category: Optional[str] = None,
        baseline_year: Optional[int] = None,
        current_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate improvement between baseline and current values.

        Computes absolute change, percentage change, annualized rate,
        and validates against minimum improvement thresholds and maximum
        credible improvement rates.

        Args:
            baseline: Baseline measurement value.
            current: Current measurement value.
            metric_category: Category for threshold lookup.
            baseline_year: Year of baseline measurement.
            current_year: Year of current measurement.

        Returns:
            Dict with improvement metrics, threshold checks,
            and provenance_hash.
        """
        logger.info(
            "Calculating improvement | baseline=%s current=%s",
            str(baseline),
            str(current),
        )
        timestamp = utcnow()
        calc_id = _new_uuid()

        baseline_d = _decimal(baseline)
        current_d = _decimal(current)

        # Absolute change (positive = improvement for reduction metrics)
        absolute_change = baseline_d - current_d
        percentage_change = _safe_divide(
            absolute_change * Decimal("100"),
            baseline_d,
            default=Decimal("0"),
        )
        percentage_change = _round_val(percentage_change, 3)

        # Direction assessment
        is_improvement = absolute_change > Decimal("0")

        # Annualized rate
        annualized_rate = Decimal("0")
        years_elapsed = Decimal("0")
        if baseline_year is not None and current_year is not None:
            years_elapsed = _decimal(current_year - baseline_year)
            if years_elapsed > Decimal("0"):
                annualized_rate = _safe_divide(
                    percentage_change, years_elapsed,
                )
                annualized_rate = _round_val(annualized_rate, 3)

        # Threshold check
        category = metric_category or "default"
        min_threshold = MINIMUM_IMPROVEMENT_THRESHOLDS.get(
            category,
            MINIMUM_IMPROVEMENT_THRESHOLDS["default"],
        )
        meets_threshold = abs(percentage_change) >= min_threshold

        # Credibility check
        max_rate = MAX_CREDIBLE_ANNUAL_IMPROVEMENT.get(
            category,
            MAX_CREDIBLE_ANNUAL_IMPROVEMENT["default"],
        )
        is_credible = True
        credibility_note = "Within credible range"
        if years_elapsed > Decimal("0") and abs(annualized_rate) > max_rate:
            is_credible = False
            credibility_note = (
                f"Annualized rate ({annualized_rate}%) exceeds maximum "
                f"credible rate ({max_rate}%) for {category}"
            )

        result = {
            "calculation_id": calc_id,
            "timestamp": str(timestamp),
            "baseline_value": str(_round_val(baseline_d, 3)),
            "current_value": str(_round_val(current_d, 3)),
            "absolute_change": str(_round_val(absolute_change, 3)),
            "percentage_change": str(percentage_change),
            "is_improvement": is_improvement,
            "annualized_rate": str(annualized_rate),
            "years_elapsed": str(years_elapsed),
            "metric_category": category,
            "minimum_threshold": str(min_threshold),
            "meets_threshold": meets_threshold,
            "is_credible": is_credible,
            "credibility_note": credibility_note,
            "max_credible_rate": str(max_rate),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Improvement calculated | calc_id=%s change=%s%% credible=%s",
            calc_id,
            str(percentage_change),
            is_credible,
        )
        return result

    def validate_methodology(
        self,
        claim: ComparativeClaim,
        assessment: Optional[MethodologyAssessment] = None,
    ) -> Dict[str, Any]:
        """Validate the methodology used for a comparative claim.

        Assesses whether the methodologies used for baseline and
        current measurements are equivalent per Article 3(4)
        requirements.

        Args:
            claim: The comparative claim to validate methodology for.
            assessment: Optional detailed methodology assessment input.

        Returns:
            Dict with methodology validation results, dimension scores,
            recommendations, and provenance_hash.
        """
        logger.info(
            "Validating methodology | claim_id=%s type=%s",
            claim.claim_id,
            claim.comparison_type.value,
        )
        timestamp = utcnow()
        validation_id = _new_uuid()
        dimension_scores: Dict[str, Decimal] = {}
        findings: List[Dict[str, str]] = []
        recommendations: List[str] = []

        if assessment is not None:
            # Score each dimension based on the assessment
            if assessment.scope_match:
                dimension_scores["scope_equivalence"] = Decimal("1.0")
                findings.append({
                    "dimension": "scope_equivalence",
                    "status": "equivalent",
                    "detail": "Measurement scope is equivalent",
                })
            else:
                dimension_scores["scope_equivalence"] = Decimal("0")
                findings.append({
                    "dimension": "scope_equivalence",
                    "status": "non_equivalent",
                    "detail": "Measurement scope differs between baseline and current",
                })
                recommendations.append(
                    "Align measurement scope between baseline and current periods"
                )

            if assessment.boundary_match:
                dimension_scores["boundary_equivalence"] = Decimal("1.0")
                findings.append({
                    "dimension": "boundary_equivalence",
                    "status": "equivalent",
                    "detail": "System boundaries are equivalent",
                })
            else:
                dimension_scores["boundary_equivalence"] = Decimal("0")
                findings.append({
                    "dimension": "boundary_equivalence",
                    "status": "non_equivalent",
                    "detail": "System boundaries differ",
                })
                recommendations.append(
                    "Ensure system boundaries are identical for fair comparison"
                )

            if assessment.data_quality_match:
                dimension_scores["data_quality_equivalence"] = Decimal("1.0")
                findings.append({
                    "dimension": "data_quality_equivalence",
                    "status": "equivalent",
                    "detail": "Data quality requirements are equivalent",
                })
            else:
                dimension_scores["data_quality_equivalence"] = Decimal("0")
                findings.append({
                    "dimension": "data_quality_equivalence",
                    "status": "non_equivalent",
                    "detail": "Data quality requirements differ",
                })
                recommendations.append(
                    "Apply consistent data quality requirements across periods"
                )

            if assessment.temporal_match:
                dimension_scores["temporal_equivalence"] = Decimal("1.0")
                findings.append({
                    "dimension": "temporal_equivalence",
                    "status": "equivalent",
                    "detail": "Time periods are equivalent",
                })
            else:
                dimension_scores["temporal_equivalence"] = Decimal("0")
                findings.append({
                    "dimension": "temporal_equivalence",
                    "status": "non_equivalent",
                    "detail": "Time periods are not equivalent",
                })
                recommendations.append(
                    "Use equivalent time periods for comparison"
                )

            if assessment.functional_unit_match:
                dimension_scores["functional_unit_equivalence"] = Decimal("1.0")
                findings.append({
                    "dimension": "functional_unit_equivalence",
                    "status": "equivalent",
                    "detail": "Functional units are equivalent",
                })
            else:
                dimension_scores["functional_unit_equivalence"] = Decimal("0")
                findings.append({
                    "dimension": "functional_unit_equivalence",
                    "status": "non_equivalent",
                    "detail": "Functional units differ",
                })
                recommendations.append(
                    "Ensure functional units are identical for product comparisons"
                )
        else:
            # Without explicit assessment, evaluate based on claim data
            has_methodology = bool(claim.methodology and len(claim.methodology) >= 5)
            base_score = Decimal("0.5") if has_methodology else Decimal("0")
            for dim_key in METHODOLOGY_WEIGHTS:
                dimension_scores[dim_key] = base_score
                findings.append({
                    "dimension": dim_key,
                    "status": "insufficient_data",
                    "detail": f"No explicit assessment provided for {dim_key}",
                })
            recommendations.append(
                "Provide a detailed methodology assessment for comprehensive validation"
            )

        # Calculate weighted score
        weighted_score = Decimal("0")
        for dim_key, score in dimension_scores.items():
            weight = METHODOLOGY_WEIGHTS.get(dim_key, Decimal("0.20"))
            weighted_score += score * weight

        weighted_score_pct = _round_val(weighted_score * Decimal("100"), 2)

        # Determine methodology status
        if weighted_score >= Decimal("0.90"):
            methodology_status = MethodologyStatus.EQUIVALENT.value
        elif weighted_score >= Decimal("0.60"):
            methodology_status = MethodologyStatus.PARTIALLY_EQUIVALENT.value
        elif weighted_score > Decimal("0"):
            methodology_status = MethodologyStatus.NON_EQUIVALENT.value
        else:
            methodology_status = MethodologyStatus.INSUFFICIENT_DATA.value

        result = {
            "validation_id": validation_id,
            "claim_id": claim.claim_id,
            "timestamp": str(timestamp),
            "methodology_status": methodology_status,
            "weighted_score": str(weighted_score_pct),
            "dimension_scores": {
                k: str(_round_val(v, 2)) for k, v in dimension_scores.items()
            },
            "findings": findings,
            "recommendations": recommendations,
            "methodology_name": claim.methodology,
            "comparison_type": claim.comparison_type.value,
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Methodology validated | validation_id=%s status=%s score=%s",
            validation_id,
            methodology_status,
            str(weighted_score_pct),
        )
        return result

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _check_mandatory_fields(
        self, claim: ComparativeClaim,
    ) -> Dict[str, Any]:
        """Check whether all mandatory fields are present for the claim type.

        Args:
            claim: Comparative claim to check.

        Returns:
            Finding dict with check results and score.
        """
        requirements = COMPARISON_REQUIREMENTS.get(claim.comparison_type, {})
        mandatory = requirements.get("mandatory_fields", [])
        missing: List[str] = []

        for field_name in mandatory:
            value = getattr(claim, field_name, None)
            if value is None or (isinstance(value, str) and len(value.strip()) == 0):
                missing.append(field_name)

        total = len(mandatory) if mandatory else 1
        present = total - len(missing)
        score = _safe_divide(_decimal(present), _decimal(total))

        return {
            "check": "mandatory_fields",
            "mandatory_count": total,
            "present_count": present,
            "missing_fields": missing,
            "score": _round_val(score, 2),
            "passed": len(missing) == 0,
        }

    def _calculate_improvement_internal(
        self, claim: ComparativeClaim,
    ) -> Dict[str, Any]:
        """Calculate the actual improvement from claim data.

        Args:
            claim: Comparative claim with baseline and current values.

        Returns:
            Finding dict with calculated improvement metrics.
        """
        baseline_d = _decimal(claim.baseline_value)
        current_d = _decimal(claim.current_value)

        absolute_change = baseline_d - current_d
        actual_pct = _safe_divide(
            absolute_change * Decimal("100"), baseline_d,
        )
        actual_pct = _round_val(actual_pct, 3)
        claimed_pct = _round_val(_decimal(claim.improvement_pct), 3)

        # Check consistency between claimed and actual
        discrepancy = abs(actual_pct - claimed_pct)
        is_consistent = discrepancy <= Decimal("1.0")

        score = Decimal("1.0") if is_consistent else Decimal("0.5")
        if discrepancy > Decimal("5.0"):
            score = Decimal("0")

        return {
            "check": "improvement_calculation",
            "baseline_value": str(baseline_d),
            "current_value": str(current_d),
            "absolute_change": str(_round_val(absolute_change, 3)),
            "actual_percentage": str(actual_pct),
            "claimed_percentage": str(claimed_pct),
            "discrepancy": str(_round_val(discrepancy, 3)),
            "is_consistent": is_consistent,
            "score": score,
        }

    def _check_improvement_threshold(
        self,
        claim: ComparativeClaim,
        improvement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check whether improvement meets minimum threshold.

        Args:
            claim: The comparative claim.
            improvement: Improvement calculation result.

        Returns:
            Finding dict with threshold check results.
        """
        category = claim.metric_category or "default"
        threshold = MINIMUM_IMPROVEMENT_THRESHOLDS.get(
            category,
            MINIMUM_IMPROVEMENT_THRESHOLDS["default"],
        )
        actual_pct = _decimal(improvement["actual_percentage"])
        meets = abs(actual_pct) >= threshold
        score = Decimal("1.0") if meets else Decimal("0.5")

        return {
            "check": "improvement_threshold",
            "metric_category": category,
            "minimum_threshold": str(threshold),
            "actual_improvement": str(actual_pct),
            "meets_threshold": meets,
            "score": score,
            "note": "Improvement meets minimum threshold for meaningful claim"
            if meets else
            f"Improvement below {threshold}% threshold for {category}",
        }

    def _check_credibility(
        self,
        claim: ComparativeClaim,
        improvement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check credibility of the claimed improvement rate.

        Args:
            claim: The comparative claim.
            improvement: Improvement calculation result.

        Returns:
            Finding dict with credibility assessment.
        """
        category = claim.metric_category or "default"
        max_rate = MAX_CREDIBLE_ANNUAL_IMPROVEMENT.get(
            category,
            MAX_CREDIBLE_ANNUAL_IMPROVEMENT["default"],
        )
        actual_pct = abs(_decimal(improvement["actual_percentage"]))

        annualized_rate = Decimal("0")
        is_credible = True
        note = "Within credible improvement range"

        if claim.baseline_year and claim.current_year:
            years = _decimal(claim.current_year - claim.baseline_year)
            if years > Decimal("0"):
                annualized_rate = _round_val(
                    _safe_divide(actual_pct, years), 3
                )
                if annualized_rate > max_rate:
                    is_credible = False
                    note = (
                        f"Annualized rate {annualized_rate}%/yr exceeds "
                        f"maximum credible rate {max_rate}%/yr for {category}"
                    )

        score = Decimal("1.0") if is_credible else Decimal("0.25")

        return {
            "check": "credibility",
            "metric_category": category,
            "max_credible_annual_rate": str(max_rate),
            "annualized_rate": str(annualized_rate),
            "is_credible": is_credible,
            "score": score,
            "note": note,
        }

    def _check_year_requirements(
        self, claim: ComparativeClaim,
    ) -> Dict[str, Any]:
        """Check year-related requirements for the comparison type.

        Args:
            claim: The comparative claim.

        Returns:
            Finding dict with year requirement check results.
        """
        requirements = COMPARISON_REQUIREMENTS.get(claim.comparison_type, {})
        min_years = requirements.get("minimum_data_years", 1)

        if claim.baseline_year is not None and claim.current_year is not None:
            actual_years = claim.current_year - claim.baseline_year
            meets_minimum = actual_years >= min_years
            score = Decimal("1.0") if meets_minimum else Decimal("0.5")
            return {
                "check": "year_requirements",
                "minimum_years": min_years,
                "actual_years": actual_years,
                "meets_minimum": meets_minimum,
                "score": score,
                "note": f"Data spans {actual_years} year(s), minimum {min_years}",
            }

        # Year fields not required for some comparison types
        needs_years = "baseline_year" in requirements.get("mandatory_fields", [])
        if needs_years:
            return {
                "check": "year_requirements",
                "minimum_years": min_years,
                "actual_years": None,
                "meets_minimum": False,
                "score": Decimal("0"),
                "note": "Year fields are required but not provided",
            }

        return {
            "check": "year_requirements",
            "minimum_years": min_years,
            "actual_years": None,
            "meets_minimum": True,
            "score": Decimal("1.0"),
            "note": "Year fields not required for this comparison type",
        }

    def _calculate_overall_score(
        self, scores: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate overall validation score from dimension scores.

        Args:
            scores: Dict of dimension names to scores (0-1).

        Returns:
            Overall score as Decimal (0-100).
        """
        if not scores:
            return Decimal("0")

        total = Decimal("0")
        for score_val in scores.values():
            total += score_val

        average = _safe_divide(total, _decimal(len(scores)))
        return _round_val(average * Decimal("100"), 2)

    def _determine_compliance(
        self,
        overall_score: Decimal,
        findings: List[Dict[str, Any]],
    ) -> str:
        """Determine compliance status based on score and findings.

        Args:
            overall_score: Overall validation score (0-100).
            findings: List of individual finding results.

        Returns:
            Compliance status string.
        """
        # Check for any critical failures
        has_critical_failure = any(
            f.get("score") == Decimal("0") and f.get("check") == "mandatory_fields"
            for f in findings
        )

        if has_critical_failure:
            return ClaimCompliance.NON_COMPLIANT.value

        if overall_score >= Decimal("80"):
            return ClaimCompliance.COMPLIANT.value
        elif overall_score >= Decimal("60"):
            return ClaimCompliance.PARTIALLY_COMPLIANT.value
        elif overall_score >= Decimal("40"):
            return ClaimCompliance.REQUIRES_AMENDMENT.value
        else:
            return ClaimCompliance.NON_COMPLIANT.value

    def _determine_future_claim_status(
        self,
        criteria_scores: Dict[str, Decimal],
        relies_on_offsets: bool,
        offset_percentage: Decimal,
    ) -> str:
        """Determine the status of a future environmental claim.

        Args:
            criteria_scores: Scores for each assessment criterion.
            relies_on_offsets: Whether the claim relies on offsets.
            offset_percentage: Percentage relying on offsets.

        Returns:
            Future claim status string.
        """
        # Check mandatory criteria
        mandatory_met = all(
            criteria_scores.get(k, Decimal("0")) >= Decimal("1.0")
            for k, v in FUTURE_CLAIM_CRITERIA.items()
            if v.get("required_for_validation", False)
        )

        if mandatory_met:
            if relies_on_offsets and offset_percentage > Decimal("50"):
                return FutureClaimStatus.CONDITIONAL.value
            return FutureClaimStatus.VALIDATED.value

        # Check if any mandatory criteria are partially met
        any_partial = any(
            Decimal("0") < criteria_scores.get(k, Decimal("0")) < Decimal("1.0")
            for k, v in FUTURE_CLAIM_CRITERIA.items()
            if v.get("required_for_validation", False)
        )

        if any_partial:
            return FutureClaimStatus.CONDITIONAL.value

        return FutureClaimStatus.UNSUBSTANTIATED.value
