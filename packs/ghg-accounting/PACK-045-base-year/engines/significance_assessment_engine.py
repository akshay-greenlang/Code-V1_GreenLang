# -*- coding: utf-8 -*-
"""
SignificanceAssessmentEngine - PACK-045 Base Year Management Engine 5
======================================================================

Quantitative significance testing engine per GHG Protocol Corporate
Standard Chapter 5 (Base Year Recalculation Policy). This engine takes
confirmed recalculation triggers and applies individual, cumulative,
and sensitivity-based significance tests to determine whether base year
recalculation is required.

The GHG Protocol recommends that organisations establish a "significance
threshold" (typically 5-10% of base year emissions) above which structural
and methodological changes necessitate recalculation of the base year
inventory. This engine implements that guidance with deterministic,
bit-perfect arithmetic.

Significance Assessment Methodology:

    Individual Assessment:
        For each trigger:
            significance_pct = abs(emission_impact_tco2e) / base_year_total_tco2e * 100
            is_significant = significance_pct >= threshold_pct

        The outcome is SIGNIFICANT if the percentage meets or exceeds the
        configured threshold, NOT_SIGNIFICANT if below, or BORDERLINE if
        within a configurable margin (default +/- 1 percentage point).

    Cumulative Assessment:
        Aggregates all individual trigger impacts:
            cumulative_impact = sum(abs(emission_impact_i) for all triggers)
            cumulative_pct = cumulative_impact / base_year_total * 100
            cumulative_significant = cumulative_pct >= cumulative_threshold

        Even when individual triggers are below the threshold, their
        combined effect may exceed it. The GHG Protocol does not mandate
        a specific cumulative test, but good practice recommends one
        (typically using the same or a lower threshold).

    Sensitivity Analysis:
        Tests whether the significance outcome is robust to uncertainty
        in the emission impact estimates:
            For BASE_CASE:     impact as estimated
            For LOW_IMPACT:    impact * (1 - sensitivity_range)
            For HIGH_IMPACT:   impact * (1 + sensitivity_range)

        If the outcome changes between scenarios (e.g. SIGNIFICANT in
        HIGH_IMPACT but NOT_SIGNIFICANT in LOW_IMPACT), this indicates
        the assessment is sensitive to estimation uncertainty and may
        warrant additional scrutiny.

Threshold Guidance (GHG Protocol Ch 5, p.35):
    - "A significance threshold is typically between 5% and 10%"
    - SBTi Corporate Manual requires 5% threshold for target tracking
    - ESRS E1 requires disclosure of recalculation policy and thresholds
    - CDP C5.1 asks for the significance threshold used

Outcome Classification:
    SIGNIFICANT:        Impact >= threshold (recalculation required)
    NOT_SIGNIFICANT:    Impact < threshold (recalculation not required)
    BORDERLINE:         Impact within margin of threshold (manual review)

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Corporate Value Chain (Scope 3) Standard, Chapter 5
    - ISO 14064-1:2018, Clause 5.2 (Base year selection)
    - ESRS E1-6 (Climate change - base year recalculation disclosures)
    - CDP Climate Change Questionnaire C5.1-C5.2 (2026)
    - SBTi Corporate Net-Zero Standard v1.1, Section 7 (Recalculation)
    - US SEC Climate Disclosure Rule (2024), Item 1504
    - California SB 253 Climate Corporate Data Accountability Act (2026)

Zero-Hallucination Guarantee:
    - All calculations use deterministic Python Decimal arithmetic
    - Threshold comparisons use exact Decimal comparison operators
    - Sensitivity scenarios use deterministic multiplier arithmetic
    - No LLM involvement in any assessment or recommendation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash regardless of when or how fast it was computed.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Any numeric or string value to convert.

    Returns:
        Decimal representation; Decimal('0') on conversion failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of numerator / denominator, or default.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely: (part / whole) * 100.

    Args:
        part: Numerator value.
        whole: Denominator value (base year total).

    Returns:
        Percentage as Decimal; Decimal('0') when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: The Decimal value to round.
        places: Number of decimal places (default 4).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _abs_decimal(value: Decimal) -> Decimal:
    """Return the absolute value of a Decimal.

    Args:
        value: The input Decimal.

    Returns:
        Absolute value.
    """
    return value if value >= Decimal("0") else -value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignificanceMethod(str, Enum):
    """Method used for significance assessment.

    INDIVIDUAL:   Each trigger is assessed independently against the
                  significance threshold.
    CUMULATIVE:   All trigger impacts are summed and assessed against
                  a cumulative threshold.
    COMBINED:     Both individual and cumulative assessments are performed
                  and the stricter outcome is used.
    """
    INDIVIDUAL = "individual"
    CUMULATIVE = "cumulative"
    COMBINED = "combined"


class AssessmentOutcome(str, Enum):
    """Result of a significance assessment.

    SIGNIFICANT:        Impact meets or exceeds the significance threshold.
                        Base year recalculation is required.
    NOT_SIGNIFICANT:    Impact is below the significance threshold.
                        Base year recalculation is not required.
    BORDERLINE:         Impact is within a configurable margin of the
                        threshold. Manual review is recommended.
    """
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    BORDERLINE = "borderline"


class SensitivityScenario(str, Enum):
    """Sensitivity analysis scenario for impact estimation uncertainty.

    BASE_CASE:    Impact as originally estimated (no adjustment).
    LOW_IMPACT:   Impact reduced by the sensitivity range factor,
                  representing the lower bound of the estimate:
                  adjusted = impact * (1 - range).
    HIGH_IMPACT:  Impact increased by the sensitivity range factor,
                  representing the upper bound of the estimate:
                  adjusted = impact * (1 + range).
    """
    BASE_CASE = "base_case"
    LOW_IMPACT = "low_impact"
    HIGH_IMPACT = "high_impact"


class TriggerType(str, Enum):
    """Types of events that may trigger base year recalculation.

    Mirrors the TriggerType from RecalculationTriggerEngine for input
    compatibility.

    ACQUISITION:              Purchase of operations or business units.
    DIVESTITURE:              Sale or closure of operations.
    MERGER:                   Full organisational merger.
    METHODOLOGY_CHANGE:       Change in calculation methodology or factors.
    ERROR_CORRECTION:         Correction of significant historical errors.
    SOURCE_BOUNDARY_CHANGE:   Addition or removal of source categories.
    OUTSOURCING_INSOURCING:   Transfer of activities across boundary.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    SOURCE_BOUNDARY_CHANGE = "source_boundary_change"
    OUTSOURCING_INSOURCING = "outsourcing_insourcing"


class EvidenceCategory(str, Enum):
    """Category of evidence in an evidence package.

    CALCULATION:      Calculation details and provenance.
    THRESHOLD:        Threshold configuration and source reference.
    TRIGGER_DETAIL:   Detail of the trigger event.
    SENSITIVITY:      Sensitivity analysis results.
    RECOMMENDATION:   Recommendation and rationale.
    REGULATORY_REF:   Regulatory reference supporting the assessment.
    """
    CALCULATION = "calculation"
    THRESHOLD = "threshold"
    TRIGGER_DETAIL = "trigger_detail"
    SENSITIVITY = "sensitivity"
    RECOMMENDATION = "recommendation"
    REGULATORY_REF = "regulatory_ref"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default significance threshold as percentage of base year emissions.
# Source: GHG Protocol Corporate Standard, Chapter 5, p.35.
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# Default cumulative significance threshold.
# Good practice: same as individual threshold or slightly lower.
DEFAULT_CUMULATIVE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# Default borderline margin (percentage points from threshold).
# Triggers within +/- this margin are classified as BORDERLINE.
DEFAULT_BORDERLINE_MARGIN_PCT: Decimal = Decimal("1.0")

# Default sensitivity range for sensitivity analysis (as a fraction).
# A range of 0.20 means +/- 20% of estimated impact.
DEFAULT_SENSITIVITY_RANGE: Decimal = Decimal("0.20")

# SBTi significance threshold.
SBTI_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# Maximum triggers per assessment run.
MAX_TRIGGERS_PER_ASSESSMENT: int = 200


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class TriggerInput(BaseModel):
    """Input representation of a confirmed trigger for significance assessment.

    Attributes:
        trigger_id: Unique identifier of the trigger being assessed.
        trigger_type: Classification of the trigger event.
        description: Human-readable description of the trigger.
        emission_impact_tco2e: Estimated absolute emission impact in tCO2e.
        effective_date: Date when the triggering event takes effect.
        scope: Which GHG scope is primarily affected.
        category: Emission source category affected.
        metadata: Additional metadata from the trigger detection phase.
    """
    trigger_id: str = Field(
        default_factory=_new_uuid, description="Trigger identifier"
    )
    trigger_type: TriggerType = Field(
        ..., description="Trigger type classification"
    )
    description: str = Field(
        default="", description="Trigger description"
    )
    emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Absolute emission impact (tCO2e)"
    )
    effective_date: Optional[date] = Field(
        default=None, description="Effective date of trigger event"
    )
    scope: str = Field(
        default="all", description="Affected GHG scope"
    )
    category: str = Field(
        default="", description="Affected source category"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("emission_impact_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission impact to Decimal."""
        return _decimal(v)


class AssessmentPolicy(BaseModel):
    """Policy configuration for significance assessment.

    Controls the thresholds, margins, and methods used in the
    significance assessment process. Typically configured per
    organisation and aligned with the organisation's recalculation
    policy (as required by GHG Protocol Chapter 5).

    Attributes:
        individual_threshold_pct: Significance threshold for individual
            trigger assessment (default 5.0%).
        cumulative_threshold_pct: Significance threshold for cumulative
            assessment across all triggers (default 5.0%).
        borderline_margin_pct: Margin around the threshold within which
            a trigger is classified as BORDERLINE (default 1.0%).
        sensitivity_range: Range for sensitivity analysis as a fraction
            (default 0.20 = +/- 20%).
        assessment_method: Which assessment method to use (default COMBINED).
        merger_always_significant: Whether MERGER triggers are always
            classified as SIGNIFICANT regardless of threshold (default True).
        include_sensitivity: Whether to run sensitivity analysis (default True).
        sbti_mode: Use SBTi-specific thresholds and rules (default False).
        organisation_name: Name of the reporting organisation (for evidence).
        base_year: The base year being assessed (for evidence).
    """
    individual_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        ge=0, le=100,
        description="Individual significance threshold (%)"
    )
    cumulative_threshold_pct: Decimal = Field(
        default=DEFAULT_CUMULATIVE_THRESHOLD_PCT,
        ge=0, le=100,
        description="Cumulative significance threshold (%)"
    )
    borderline_margin_pct: Decimal = Field(
        default=DEFAULT_BORDERLINE_MARGIN_PCT,
        ge=0, le=50,
        description="Borderline margin (percentage points)"
    )
    sensitivity_range: Decimal = Field(
        default=DEFAULT_SENSITIVITY_RANGE,
        ge=0, le=1,
        description="Sensitivity range as fraction (0.20 = +/- 20%)"
    )
    assessment_method: SignificanceMethod = Field(
        default=SignificanceMethod.COMBINED,
        description="Assessment method"
    )
    merger_always_significant: bool = Field(
        default=True,
        description="MERGER triggers always significant"
    )
    include_sensitivity: bool = Field(
        default=True,
        description="Include sensitivity analysis"
    )
    sbti_mode: bool = Field(
        default=False,
        description="Use SBTi thresholds and rules"
    )
    organisation_name: str = Field(
        default="", description="Reporting organisation name"
    )
    base_year: Optional[int] = Field(
        default=None, ge=1990, le=2030,
        description="Base year being assessed"
    )

    @field_validator(
        "individual_threshold_pct", "cumulative_threshold_pct",
        "borderline_margin_pct", "sensitivity_range",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce thresholds and margins to Decimal."""
        return _decimal(v)

    @model_validator(mode="after")
    def apply_sbti_overrides(self) -> "AssessmentPolicy":
        """When sbti_mode is enabled, enforce SBTi thresholds.

        SBTi Corporate Manual (2023), Section 7 mandates a 5%
        significance threshold for target tracking.
        """
        if self.sbti_mode:
            self.individual_threshold_pct = SBTI_SIGNIFICANCE_THRESHOLD_PCT
            self.cumulative_threshold_pct = SBTI_SIGNIFICANCE_THRESHOLD_PCT
            self.merger_always_significant = True
        return self


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class TriggerAssessment(BaseModel):
    """Individual significance assessment result for a single trigger.

    Documents the complete assessment of one trigger against the
    significance threshold, including the calculated significance
    percentage, the threshold used, the outcome, and the margin
    between the calculated percentage and the threshold.

    Attributes:
        trigger_id: Identifier of the assessed trigger.
        trigger_type: Type classification of the trigger.
        emission_impact_tco2e: Absolute emission impact used (tCO2e).
        base_year_total_tco2e: Base year total used as denominator (tCO2e).
        significance_pct: Calculated significance percentage.
        threshold_pct: Threshold percentage applied.
        outcome: Assessment outcome (SIGNIFICANT, NOT_SIGNIFICANT, BORDERLINE).
        margin_pct: Distance from threshold in percentage points:
            margin = significance_pct - threshold_pct
            Positive = above threshold, Negative = below threshold.
    """
    trigger_id: str = Field(default="", description="Trigger ID")
    trigger_type: TriggerType = Field(
        default=TriggerType.METHODOLOGY_CHANGE,
        description="Trigger type"
    )
    emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Emission impact (tCO2e)"
    )
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Base year total (tCO2e)"
    )
    significance_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Significance percentage"
    )
    threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        ge=0,
        description="Applied threshold (%)"
    )
    outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.NOT_SIGNIFICANT,
        description="Assessment outcome"
    )
    margin_pct: Decimal = Field(
        default=Decimal("0"),
        description="Distance from threshold (pp)"
    )

    @field_validator(
        "emission_impact_tco2e", "base_year_total_tco2e",
        "significance_pct", "threshold_pct", "margin_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class CumulativeAssessment(BaseModel):
    """Cumulative significance assessment across all triggers.

    Aggregates individual trigger impacts and assesses the total
    against the cumulative significance threshold.

    Attributes:
        triggers: List of individual trigger assessments included.
        cumulative_impact_tco2e: Sum of absolute emission impacts (tCO2e).
        cumulative_significance_pct: Cumulative impact as % of base year.
        cumulative_threshold_pct: Cumulative threshold percentage applied.
        cumulative_outcome: Overall cumulative assessment outcome.
        trigger_count: Number of triggers included in the cumulative test.
        significant_trigger_count: Number of individually significant triggers.
    """
    triggers: List[TriggerAssessment] = Field(
        default_factory=list,
        description="Individual assessments"
    )
    cumulative_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Cumulative impact (tCO2e)"
    )
    cumulative_significance_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Cumulative significance (%)"
    )
    cumulative_threshold_pct: Decimal = Field(
        default=DEFAULT_CUMULATIVE_THRESHOLD_PCT,
        ge=0,
        description="Cumulative threshold (%)"
    )
    cumulative_outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.NOT_SIGNIFICANT,
        description="Cumulative assessment outcome"
    )
    trigger_count: int = Field(
        default=0, ge=0, description="Total triggers assessed"
    )
    significant_trigger_count: int = Field(
        default=0, ge=0, description="Individually significant triggers"
    )

    @field_validator(
        "cumulative_impact_tco2e", "cumulative_significance_pct",
        "cumulative_threshold_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class SensitivityResult(BaseModel):
    """Result of a sensitivity scenario analysis.

    Tests whether the significance outcome is robust to uncertainty
    in the emission impact estimates by applying high and low bounds.

    Attributes:
        scenario: The sensitivity scenario applied.
        adjusted_impact_tco2e: Impact after applying the scenario multiplier.
        adjusted_significance_pct: Recalculated significance percentage.
        outcome: Assessment outcome under this scenario.
        outcome_changes: Whether the outcome differs from the BASE_CASE.
    """
    scenario: SensitivityScenario = Field(
        ..., description="Sensitivity scenario"
    )
    adjusted_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Adjusted impact (tCO2e)"
    )
    adjusted_significance_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Adjusted significance (%)"
    )
    outcome: AssessmentOutcome = Field(
        default=AssessmentOutcome.NOT_SIGNIFICANT,
        description="Outcome under scenario"
    )
    outcome_changes: bool = Field(
        default=False,
        description="Whether outcome differs from base case"
    )

    @field_validator(
        "adjusted_impact_tco2e", "adjusted_significance_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class EvidenceItem(BaseModel):
    """A single item in an evidence package for audit and verification.

    Attributes:
        item_id: Unique evidence item identifier.
        category: Category of evidence (calculation, threshold, etc.).
        title: Short title of the evidence item.
        description: Detailed description of the evidence.
        value: The evidence value (numeric, string, or JSON).
        source_reference: Regulatory or standard reference supporting this.
    """
    item_id: str = Field(
        default_factory=_new_uuid, description="Evidence item ID"
    )
    category: EvidenceCategory = Field(
        ..., description="Evidence category"
    )
    title: str = Field(default="", description="Evidence title")
    description: str = Field(default="", description="Evidence description")
    value: str = Field(default="", description="Evidence value")
    source_reference: str = Field(
        default="", description="Regulatory/standard reference"
    )


class EvidencePackage(BaseModel):
    """Complete evidence package for audit and third-party verification.

    Assembled from assessment results for ISAE 3410 / ISAE 3000 limited
    or reasonable assurance engagements.

    Attributes:
        package_id: Unique package identifier.
        organisation: Reporting organisation name.
        base_year: Base year being assessed.
        assessment_date: Date the assessment was performed.
        items: List of evidence items in the package.
        summary: Executive summary of the assessment.
        provenance_hash: SHA-256 hash of the evidence package.
    """
    package_id: str = Field(
        default_factory=_new_uuid, description="Package ID"
    )
    organisation: str = Field(default="", description="Organisation name")
    base_year: Optional[int] = Field(
        default=None, description="Base year"
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow, description="Assessment date"
    )
    items: List[EvidenceItem] = Field(
        default_factory=list, description="Evidence items"
    )
    summary: str = Field(default="", description="Executive summary")
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class SignificanceResult(BaseModel):
    """Complete significance assessment result with full provenance.

    Aggregates individual assessments, cumulative assessment, sensitivity
    analysis, and recommendations into a single auditable result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version that produced this result.
        calculated_at: Timestamp of the assessment.
        processing_time_ms: Total processing time in milliseconds.
        individual_assessments: List of individual trigger assessments.
        cumulative_assessment: Cumulative assessment across all triggers.
        sensitivity_results: Sensitivity analysis results for each scenario.
        overall_recommendation: Final recommendation text.
        recalculation_required: Whether base year recalculation is required
            based on the assessment.
        rationale: Detailed rationale supporting the recommendation.
        assessment_method: Method used for the assessment.
        policy_applied: Policy configuration used.
        base_year_total_tco2e: Base year total used in calculations.
        provenance_hash: SHA-256 hash of the complete result.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0, description="Processing time (ms)"
    )
    individual_assessments: List[TriggerAssessment] = Field(
        default_factory=list, description="Individual assessments"
    )
    cumulative_assessment: Optional[CumulativeAssessment] = Field(
        default=None, description="Cumulative assessment"
    )
    sensitivity_results: List[SensitivityResult] = Field(
        default_factory=list, description="Sensitivity analysis results"
    )
    overall_recommendation: str = Field(
        default="", description="Final recommendation"
    )
    recalculation_required: bool = Field(
        default=False,
        description="Whether recalculation is required"
    )
    rationale: str = Field(
        default="", description="Assessment rationale"
    )
    assessment_method: SignificanceMethod = Field(
        default=SignificanceMethod.COMBINED,
        description="Assessment method used"
    )
    policy_applied: Optional[AssessmentPolicy] = Field(
        default=None, description="Policy configuration used"
    )
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Base year total (tCO2e)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator("base_year_total_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce base year total to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Model Rebuild (Pydantic v2 deferred annotations resolution)
# ---------------------------------------------------------------------------

TriggerInput.model_rebuild()
AssessmentPolicy.model_rebuild()
TriggerAssessment.model_rebuild()
CumulativeAssessment.model_rebuild()
SensitivityResult.model_rebuild()
EvidenceItem.model_rebuild()
EvidencePackage.model_rebuild()
SignificanceResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SignificanceAssessmentEngine:
    """Quantitative significance testing engine per GHG Protocol Ch 5.

    Takes confirmed recalculation triggers and applies individual,
    cumulative, and sensitivity-based significance tests to determine
    whether base year recalculation is required.

    Assessment Workflow:
        1. Assess each trigger individually against the threshold.
        2. Compute cumulative impact and assess against cumulative threshold.
        3. Run sensitivity analysis to test outcome robustness.
        4. Generate overall recommendation and rationale.
        5. Produce SHA-256 provenance hash for audit.

    Usage:
        >>> engine = SignificanceAssessmentEngine()
        >>> triggers = [TriggerInput(trigger_type=TriggerType.ACQUISITION, ...)]
        >>> policy = AssessmentPolicy(individual_threshold_pct=Decimal("5.0"))
        >>> result = engine.assess_significance(triggers, Decimal("50000"), policy)
        >>> if result.recalculation_required:
        ...     # Route to BaseYearAdjustmentEngine
        ...     pass

    All calculations use Python Decimal arithmetic with ROUND_HALF_UP.
    No LLM is used in any assessment or recommendation path.
    """

    def __init__(self) -> None:
        """Initialise the SignificanceAssessmentEngine."""
        self._version: str = _MODULE_VERSION
        logger.info(
            "SignificanceAssessmentEngine v%s initialised", self._version
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def assess_significance(
        self,
        triggers: List[TriggerInput],
        base_year_total_tco2e: Decimal,
        policy: Optional[AssessmentPolicy] = None,
    ) -> SignificanceResult:
        """Execute complete significance assessment.

        Performs individual, cumulative, and sensitivity assessments
        based on the configured policy and returns a comprehensive
        result with provenance.

        Args:
            triggers: List of confirmed triggers to assess.
            base_year_total_tco2e: Total base year emissions (tCO2e).
            policy: Assessment policy configuration. If None, defaults used.

        Returns:
            SignificanceResult with all assessments, recommendation, and hash.

        Raises:
            ValueError: If base_year_total_tco2e is negative.
        """
        start_ns = time.perf_counter_ns()
        base_year_total = _decimal(base_year_total_tco2e)

        if base_year_total < Decimal("0"):
            raise ValueError("base_year_total_tco2e must be non-negative")

        if policy is None:
            policy = AssessmentPolicy()

        # Step 1: Individual assessments
        individual_assessments: List[TriggerAssessment] = []
        for trigger in triggers:
            assessment = self.assess_individual(
                trigger, base_year_total, policy.individual_threshold_pct,
                policy.borderline_margin_pct, policy.merger_always_significant,
            )
            individual_assessments.append(assessment)

        # Step 2: Cumulative assessment
        cumulative_assessment: Optional[CumulativeAssessment] = None
        if policy.assessment_method in (
            SignificanceMethod.CUMULATIVE, SignificanceMethod.COMBINED
        ):
            cumulative_assessment = self.assess_cumulative(
                triggers, base_year_total,
                policy.cumulative_threshold_pct,
                policy.borderline_margin_pct,
            )
            cumulative_assessment.triggers = individual_assessments

        # Step 3: Sensitivity analysis
        sensitivity_results: List[SensitivityResult] = []
        if policy.include_sensitivity and triggers:
            sensitivity_results = self.run_sensitivity(
                triggers, base_year_total,
                policy.sensitivity_range,
                policy.individual_threshold_pct,
                policy.borderline_margin_pct,
            )

        # Step 4: Determine overall outcome
        recalculation_required, rationale = self._determine_overall_outcome(
            individual_assessments, cumulative_assessment,
            sensitivity_results, policy,
        )

        # Step 5: Generate recommendation
        recommendation = self.recommend_action(
            individual_assessments, cumulative_assessment,
            sensitivity_results, recalculation_required, policy,
        )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        result = SignificanceResult(
            individual_assessments=individual_assessments,
            cumulative_assessment=cumulative_assessment,
            sensitivity_results=sensitivity_results,
            overall_recommendation=recommendation,
            recalculation_required=recalculation_required,
            rationale=rationale,
            assessment_method=policy.assessment_method,
            policy_applied=policy,
            base_year_total_tco2e=base_year_total,
            processing_time_ms=round(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Significance assessment complete: %d triggers, recalculation=%s, "
            "processing=%.2fms",
            len(triggers), recalculation_required, elapsed_ms,
        )

        return result

    def assess_individual(
        self,
        trigger: TriggerInput,
        base_year_total: Decimal,
        threshold_pct: Decimal = DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        borderline_margin_pct: Decimal = DEFAULT_BORDERLINE_MARGIN_PCT,
        merger_always_significant: bool = True,
    ) -> TriggerAssessment:
        """Assess a single trigger for individual significance.

        Formula:
            significance_pct = abs(emission_impact_tco2e) / base_year_total * 100

        Outcome determination:
            if MERGER and merger_always_significant: SIGNIFICANT
            elif significance_pct >= threshold_pct: SIGNIFICANT
            elif significance_pct >= (threshold_pct - margin): BORDERLINE
            else: NOT_SIGNIFICANT

        Margin calculation:
            margin_pct = significance_pct - threshold_pct
            Positive means above threshold, negative means below.

        Args:
            trigger: The trigger to assess.
            base_year_total: Base year total emissions (tCO2e).
            threshold_pct: Significance threshold percentage.
            borderline_margin_pct: Margin for borderline classification.
            merger_always_significant: Force SIGNIFICANT for mergers.

        Returns:
            TriggerAssessment with calculated significance and outcome.
        """
        impact = _abs_decimal(_decimal(trigger.emission_impact_tco2e))
        significance = _safe_pct(impact, base_year_total)
        margin = significance - threshold_pct

        # Determine outcome
        if trigger.trigger_type == TriggerType.MERGER and merger_always_significant:
            outcome = AssessmentOutcome.SIGNIFICANT
        elif significance >= threshold_pct:
            outcome = AssessmentOutcome.SIGNIFICANT
        elif significance >= (threshold_pct - borderline_margin_pct):
            outcome = AssessmentOutcome.BORDERLINE
        else:
            outcome = AssessmentOutcome.NOT_SIGNIFICANT

        return TriggerAssessment(
            trigger_id=trigger.trigger_id,
            trigger_type=trigger.trigger_type,
            emission_impact_tco2e=_round_val(impact, 3),
            base_year_total_tco2e=_round_val(base_year_total, 3),
            significance_pct=_round_val(significance, 4),
            threshold_pct=_round_val(threshold_pct, 4),
            outcome=outcome,
            margin_pct=_round_val(margin, 4),
        )

    def assess_cumulative(
        self,
        triggers: List[TriggerInput],
        base_year_total: Decimal,
        cumulative_threshold_pct: Decimal = DEFAULT_CUMULATIVE_THRESHOLD_PCT,
        borderline_margin_pct: Decimal = DEFAULT_BORDERLINE_MARGIN_PCT,
    ) -> CumulativeAssessment:
        """Assess cumulative significance across all triggers.

        Formula:
            cumulative_impact = sum(abs(trigger_i.emission_impact_tco2e) for all i)
            cumulative_pct = cumulative_impact / base_year_total * 100

        Outcome:
            SIGNIFICANT if cumulative_pct >= cumulative_threshold_pct
            BORDERLINE if cumulative_pct >= (cumulative_threshold_pct - margin)
            NOT_SIGNIFICANT otherwise

        Args:
            triggers: All triggers to include in cumulative assessment.
            base_year_total: Base year total emissions (tCO2e).
            cumulative_threshold_pct: Cumulative significance threshold.
            borderline_margin_pct: Margin for borderline classification.

        Returns:
            CumulativeAssessment with aggregated impact and outcome.
        """
        cumulative_impact = sum(
            (_abs_decimal(_decimal(t.emission_impact_tco2e)) for t in triggers),
            Decimal("0"),
        )
        cumulative_pct = _safe_pct(cumulative_impact, base_year_total)

        # Count individually significant triggers
        significant_count = 0
        for trigger in triggers:
            ind_pct = _safe_pct(
                _abs_decimal(_decimal(trigger.emission_impact_tco2e)),
                base_year_total,
            )
            if ind_pct >= cumulative_threshold_pct:
                significant_count += 1

        # Determine cumulative outcome
        if cumulative_pct >= cumulative_threshold_pct:
            outcome = AssessmentOutcome.SIGNIFICANT
        elif cumulative_pct >= (cumulative_threshold_pct - borderline_margin_pct):
            outcome = AssessmentOutcome.BORDERLINE
        else:
            outcome = AssessmentOutcome.NOT_SIGNIFICANT

        return CumulativeAssessment(
            cumulative_impact_tco2e=_round_val(cumulative_impact, 3),
            cumulative_significance_pct=_round_val(cumulative_pct, 4),
            cumulative_threshold_pct=_round_val(cumulative_threshold_pct, 4),
            cumulative_outcome=outcome,
            trigger_count=len(triggers),
            significant_trigger_count=significant_count,
        )

    def run_sensitivity(
        self,
        triggers: List[TriggerInput],
        base_year_total: Decimal,
        range_pct: Decimal = DEFAULT_SENSITIVITY_RANGE,
        threshold_pct: Decimal = DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        borderline_margin_pct: Decimal = DEFAULT_BORDERLINE_MARGIN_PCT,
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis across three scenarios.

        Tests whether the significance outcome is robust to uncertainty
        in the estimated emission impacts.

        Scenarios:
            BASE_CASE:    cumulative_impact (as estimated)
            LOW_IMPACT:   cumulative_impact * (1 - range_pct)
            HIGH_IMPACT:  cumulative_impact * (1 + range_pct)

        For each scenario:
            adjusted_pct = adjusted_impact / base_year_total * 100
            outcome = SIGNIFICANT / BORDERLINE / NOT_SIGNIFICANT

        outcome_changes is True when the scenario outcome differs from
        the BASE_CASE outcome.

        Args:
            triggers: All triggers to include.
            base_year_total: Base year total emissions (tCO2e).
            range_pct: Sensitivity range as fraction (e.g. 0.20 = 20%).
            threshold_pct: Significance threshold for outcome determination.
            borderline_margin_pct: Margin for borderline classification.

        Returns:
            List of SensitivityResult for each scenario.
        """
        cumulative_impact = sum(
            (_abs_decimal(_decimal(t.emission_impact_tco2e)) for t in triggers),
            Decimal("0"),
        )

        scenarios: List[Tuple[SensitivityScenario, Decimal]] = [
            (SensitivityScenario.BASE_CASE, Decimal("1")),
            (SensitivityScenario.LOW_IMPACT, Decimal("1") - _decimal(range_pct)),
            (SensitivityScenario.HIGH_IMPACT, Decimal("1") + _decimal(range_pct)),
        ]

        # Compute base case outcome first
        base_pct = _safe_pct(cumulative_impact, base_year_total)
        base_outcome = self._classify_outcome(
            base_pct, threshold_pct, borderline_margin_pct
        )

        results: List[SensitivityResult] = []
        for scenario, multiplier in scenarios:
            adjusted_impact = cumulative_impact * multiplier
            adjusted_pct = _safe_pct(adjusted_impact, base_year_total)
            scenario_outcome = self._classify_outcome(
                adjusted_pct, threshold_pct, borderline_margin_pct
            )
            outcome_changes = scenario_outcome != base_outcome

            results.append(SensitivityResult(
                scenario=scenario,
                adjusted_impact_tco2e=_round_val(adjusted_impact, 3),
                adjusted_significance_pct=_round_val(adjusted_pct, 4),
                outcome=scenario_outcome,
                outcome_changes=outcome_changes,
            ))

        return results

    def recommend_action(
        self,
        individual_assessments: List[TriggerAssessment],
        cumulative_assessment: Optional[CumulativeAssessment],
        sensitivity_results: List[SensitivityResult],
        recalculation_required: bool,
        policy: AssessmentPolicy,
    ) -> str:
        """Generate a human-readable recommendation based on assessments.

        This is a deterministic string-construction function (no LLM).
        The recommendation is built from the assessment outcomes and
        policy configuration using conditional logic.

        Args:
            individual_assessments: Individual trigger assessments.
            cumulative_assessment: Cumulative assessment result.
            sensitivity_results: Sensitivity analysis results.
            recalculation_required: Whether recalculation is needed.
            policy: Assessment policy used.

        Returns:
            Recommendation text string.
        """
        parts: List[str] = []

        if recalculation_required:
            parts.append(
                "BASE YEAR RECALCULATION REQUIRED. "
                "One or more significance tests indicate that the base year "
                "must be recalculated to maintain comparability."
            )
        else:
            parts.append(
                "Base year recalculation is NOT required based on the "
                "significance assessment."
            )

        # Individual results summary
        sig_count = sum(
            1 for a in individual_assessments
            if a.outcome == AssessmentOutcome.SIGNIFICANT
        )
        borderline_count = sum(
            1 for a in individual_assessments
            if a.outcome == AssessmentOutcome.BORDERLINE
        )
        total = len(individual_assessments)

        if sig_count > 0:
            parts.append(
                f"{sig_count} of {total} trigger(s) individually exceed the "
                f"{policy.individual_threshold_pct}% significance threshold."
            )
        if borderline_count > 0:
            parts.append(
                f"{borderline_count} trigger(s) are BORDERLINE (within "
                f"{policy.borderline_margin_pct} pp of threshold). "
                "Manual review recommended for borderline triggers."
            )

        # Cumulative summary
        if cumulative_assessment is not None:
            if cumulative_assessment.cumulative_outcome == AssessmentOutcome.SIGNIFICANT:
                parts.append(
                    f"Cumulative impact ({cumulative_assessment.cumulative_significance_pct}%) "
                    f"exceeds the cumulative threshold ({policy.cumulative_threshold_pct}%). "
                    "Even if individual triggers are below threshold, the combined "
                    "effect requires recalculation."
                )
            elif cumulative_assessment.cumulative_outcome == AssessmentOutcome.BORDERLINE:
                parts.append(
                    f"Cumulative impact ({cumulative_assessment.cumulative_significance_pct}%) "
                    f"is BORDERLINE relative to the cumulative threshold "
                    f"({policy.cumulative_threshold_pct}%). "
                    "Consider recalculation as a precautionary measure."
                )

        # Sensitivity summary
        if sensitivity_results:
            outcome_changes = any(r.outcome_changes for r in sensitivity_results)
            if outcome_changes:
                parts.append(
                    "SENSITIVITY WARNING: The significance outcome changes "
                    "under sensitivity analysis. The assessment is sensitive to "
                    "uncertainty in the estimated emission impacts. Additional "
                    "data or more precise impact estimates are recommended."
                )
            else:
                parts.append(
                    "Sensitivity analysis confirms the outcome is robust across "
                    "all tested scenarios."
                )

        # SBTi note
        if policy.sbti_mode:
            parts.append(
                "Assessment performed under SBTi mode. Per SBTi Corporate Manual "
                "Section 7, a 5% significance threshold is applied for target tracking."
            )

        return " ".join(parts)

    def generate_evidence_package(
        self,
        result: SignificanceResult,
    ) -> EvidencePackage:
        """Generate an evidence package for audit and verification.

        Assembles assessment results, calculation details, threshold
        references, and regulatory citations into a structured evidence
        package suitable for ISAE 3410 assurance engagements.

        Args:
            result: The complete significance assessment result.

        Returns:
            EvidencePackage with categorised evidence items.
        """
        items: List[EvidenceItem] = []
        policy = result.policy_applied or AssessmentPolicy()

        # 1. Threshold configuration evidence
        items.append(EvidenceItem(
            category=EvidenceCategory.THRESHOLD,
            title="Individual Significance Threshold",
            description=(
                f"Significance threshold set at {policy.individual_threshold_pct}% "
                f"of base year emissions per organisation recalculation policy."
            ),
            value=str(policy.individual_threshold_pct),
            source_reference=(
                "GHG Protocol Corporate Standard (2015), Chapter 5, p.35: "
                "'A significance threshold is typically between 5% and 10%.'"
            ),
        ))

        items.append(EvidenceItem(
            category=EvidenceCategory.THRESHOLD,
            title="Cumulative Significance Threshold",
            description=(
                f"Cumulative threshold set at {policy.cumulative_threshold_pct}% "
                f"for assessment of combined trigger impacts."
            ),
            value=str(policy.cumulative_threshold_pct),
            source_reference=(
                "GHG Protocol Corporate Standard (2015), Chapter 5. "
                "Good practice recommendation for cumulative assessment."
            ),
        ))

        items.append(EvidenceItem(
            category=EvidenceCategory.THRESHOLD,
            title="Base Year Total Emissions",
            description=(
                f"Base year total emissions used as denominator: "
                f"{result.base_year_total_tco2e} tCO2e."
            ),
            value=str(result.base_year_total_tco2e),
            source_reference="Organisation's verified base year inventory.",
        ))

        # 2. Individual assessment evidence
        for assessment in result.individual_assessments:
            items.append(EvidenceItem(
                category=EvidenceCategory.CALCULATION,
                title=f"Individual Assessment: {assessment.trigger_id}",
                description=(
                    f"Trigger type: {assessment.trigger_type.value}. "
                    f"Impact: {assessment.emission_impact_tco2e} tCO2e. "
                    f"Significance: {assessment.significance_pct}%. "
                    f"Threshold: {assessment.threshold_pct}%. "
                    f"Margin: {assessment.margin_pct} pp. "
                    f"Outcome: {assessment.outcome.value}."
                ),
                value=str(assessment.significance_pct),
                source_reference=(
                    "GHG Protocol Corporate Standard (2015), Chapter 5: "
                    "significance_pct = abs(impact) / base_year_total * 100"
                ),
            ))

        # 3. Cumulative assessment evidence
        if result.cumulative_assessment is not None:
            cum = result.cumulative_assessment
            items.append(EvidenceItem(
                category=EvidenceCategory.CALCULATION,
                title="Cumulative Significance Assessment",
                description=(
                    f"Cumulative impact: {cum.cumulative_impact_tco2e} tCO2e. "
                    f"Cumulative significance: {cum.cumulative_significance_pct}%. "
                    f"Threshold: {cum.cumulative_threshold_pct}%. "
                    f"Outcome: {cum.cumulative_outcome.value}. "
                    f"Triggers: {cum.trigger_count}, "
                    f"individually significant: {cum.significant_trigger_count}."
                ),
                value=str(cum.cumulative_significance_pct),
                source_reference=(
                    "GHG Protocol Corporate Standard (2015), Chapter 5. "
                    "Cumulative assessment is a good practice extension."
                ),
            ))

        # 4. Sensitivity evidence
        for sens in result.sensitivity_results:
            items.append(EvidenceItem(
                category=EvidenceCategory.SENSITIVITY,
                title=f"Sensitivity: {sens.scenario.value}",
                description=(
                    f"Adjusted impact: {sens.adjusted_impact_tco2e} tCO2e. "
                    f"Adjusted significance: {sens.adjusted_significance_pct}%. "
                    f"Outcome: {sens.outcome.value}. "
                    f"Outcome changes from base: {sens.outcome_changes}."
                ),
                value=str(sens.adjusted_significance_pct),
                source_reference="Sensitivity analysis per assessment policy.",
            ))

        # 5. Overall recommendation evidence
        items.append(EvidenceItem(
            category=EvidenceCategory.RECOMMENDATION,
            title="Overall Recommendation",
            description=result.overall_recommendation,
            value="RECALCULATION_REQUIRED" if result.recalculation_required else "NO_RECALCULATION",
            source_reference=result.rationale,
        ))

        # 6. Regulatory references
        regulatory_refs = [
            ("GHG Protocol Corporate Standard", "Chapter 5 (2015 revision)"),
            ("ISO 14064-1:2018", "Clause 5.2"),
            ("ESRS E1-6", "Base year recalculation disclosures"),
            ("CDP Climate Change C5.1-C5.2", "2026 questionnaire"),
            ("SBTi Corporate Manual v1.1", "Section 7 - Recalculation"),
            ("US SEC Climate Disclosure Rule", "Item 1504 (2024)"),
        ]
        for ref_name, ref_detail in regulatory_refs:
            items.append(EvidenceItem(
                category=EvidenceCategory.REGULATORY_REF,
                title=ref_name,
                description=ref_detail,
                value=ref_name,
                source_reference=ref_detail,
            ))

        # Build summary
        summary = (
            f"Significance assessment for {len(result.individual_assessments)} "
            f"trigger(s) against base year total of {result.base_year_total_tco2e} tCO2e. "
            f"Recalculation required: {result.recalculation_required}. "
            f"Method: {result.assessment_method.value}."
        )

        package = EvidencePackage(
            organisation=policy.organisation_name,
            base_year=policy.base_year,
            items=items,
            summary=summary,
        )
        package.provenance_hash = _compute_hash(package)

        return package

    # -----------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------

    def _classify_outcome(
        self,
        significance_pct: Decimal,
        threshold_pct: Decimal,
        borderline_margin_pct: Decimal,
    ) -> AssessmentOutcome:
        """Classify a significance percentage into an outcome.

        Deterministic classification using exact Decimal comparisons.

        Args:
            significance_pct: Calculated significance percentage.
            threshold_pct: Significance threshold.
            borderline_margin_pct: Margin for borderline band.

        Returns:
            AssessmentOutcome classification.
        """
        if significance_pct >= threshold_pct:
            return AssessmentOutcome.SIGNIFICANT
        elif significance_pct >= (threshold_pct - borderline_margin_pct):
            return AssessmentOutcome.BORDERLINE
        else:
            return AssessmentOutcome.NOT_SIGNIFICANT

    def _determine_overall_outcome(
        self,
        individual_assessments: List[TriggerAssessment],
        cumulative_assessment: Optional[CumulativeAssessment],
        sensitivity_results: List[SensitivityResult],
        policy: AssessmentPolicy,
    ) -> Tuple[bool, str]:
        """Determine the overall recalculation decision and rationale.

        Decision Logic:
            1. If any individual trigger is SIGNIFICANT => recalculation required.
            2. If cumulative assessment is SIGNIFICANT => recalculation required.
            3. If sensitivity shows HIGH_IMPACT as SIGNIFICANT and base case is
               BORDERLINE => recommend recalculation (precautionary).
            4. Otherwise, recalculation not required.

        Args:
            individual_assessments: Individual trigger assessments.
            cumulative_assessment: Cumulative assessment.
            sensitivity_results: Sensitivity analysis results.
            policy: Assessment policy.

        Returns:
            Tuple of (recalculation_required, rationale string).
        """
        reasons: List[str] = []

        # Check 1: Any individual trigger significant
        sig_individuals = [
            a for a in individual_assessments
            if a.outcome == AssessmentOutcome.SIGNIFICANT
        ]
        if sig_individuals:
            trigger_ids = ", ".join(a.trigger_id for a in sig_individuals)
            reasons.append(
                f"{len(sig_individuals)} individual trigger(s) exceed the "
                f"{policy.individual_threshold_pct}% threshold "
                f"(trigger IDs: {trigger_ids})."
            )

        # Check 2: Cumulative assessment significant
        cumulative_significant = False
        if cumulative_assessment is not None:
            if cumulative_assessment.cumulative_outcome == AssessmentOutcome.SIGNIFICANT:
                cumulative_significant = True
                reasons.append(
                    f"Cumulative impact ({cumulative_assessment.cumulative_significance_pct}%) "
                    f"exceeds the {policy.cumulative_threshold_pct}% cumulative threshold."
                )

        # Check 3: Sensitivity-driven precautionary recalculation
        sensitivity_precaution = False
        if sensitivity_results:
            base_result = next(
                (r for r in sensitivity_results
                 if r.scenario == SensitivityScenario.BASE_CASE),
                None,
            )
            high_result = next(
                (r for r in sensitivity_results
                 if r.scenario == SensitivityScenario.HIGH_IMPACT),
                None,
            )
            if (
                base_result is not None
                and high_result is not None
                and base_result.outcome == AssessmentOutcome.BORDERLINE
                and high_result.outcome == AssessmentOutcome.SIGNIFICANT
            ):
                sensitivity_precaution = True
                reasons.append(
                    "Base case is BORDERLINE but HIGH_IMPACT scenario is SIGNIFICANT. "
                    "Precautionary recalculation recommended due to estimation uncertainty."
                )

        # Decision
        recalculation_required = (
            len(sig_individuals) > 0
            or cumulative_significant
            or sensitivity_precaution
        )

        if not recalculation_required:
            reasons.append(
                "All significance tests are below threshold. "
                "Base year recalculation is not required at this time."
            )

        rationale = " ".join(reasons)
        return recalculation_required, rationale
