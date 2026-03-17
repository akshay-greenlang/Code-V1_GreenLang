# -*- coding: utf-8 -*-
"""
ImpactMaterialityEngine - PACK-015 Double Materiality Engine 1
================================================================

Scores impact materiality per ESRS 1 Paragraphs 43-48.

Under the European Sustainability Reporting Standards (ESRS), impact
materiality assesses whether a sustainability matter is material based
on the severity of actual or potential impacts on people or the
environment.  For actual negative impacts, severity is determined by
scale, scope, and irremediability.  For potential negative impacts,
likelihood is also considered.  Positive impacts are assessed by scale
and scope only.

ESRS 1 Impact Materiality Assessment Framework:
    - Para 43: A sustainability matter is material from an impact
      perspective when it pertains to the undertaking's material actual
      or potential, positive or negative impacts on people or the
      environment over the short-, medium- or long-term.
    - Para 44: Severity is determined by scale (how grave), scope (how
      widespread), and irremediable character (whether and to what
      extent the negative impacts could be remediated).
    - Para 45: Each characteristic has equal weight in determining
      severity.
    - Para 46: For potential negative impacts, likelihood is an
      additional factor.
    - Para 47: For positive impacts, severity is determined by scale
      and scope.
    - Para 48: Materiality of impacts is assessed based on appropriate
      quantitative and/or qualitative thresholds.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS 1 General Requirements, Chapter 3 Double Materiality
    - EFRAG Implementation Guidance IG 1 (Materiality Assessment)
    - GRI 3: Material Topics 2021 (cross-reference)

Zero-Hallucination:
    - Severity calculation uses geometric mean of scale, scope,
      irremediability (deterministic)
    - Impact score combines severity and likelihood via multiplication
    - Threshold comparison is a simple numeric check
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-015 Double Materiality
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
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


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


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TimeHorizon(str, Enum):
    """Time horizon for sustainability impact assessment.

    Per ESRS 1 Para 77, undertakings shall consider short-, medium-,
    and long-term time horizons when assessing impacts.
    """
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class ValueChainStage(str, Enum):
    """Stage of the value chain where the impact occurs.

    Per ESRS 1 Para 63, the materiality assessment shall cover the
    undertaking's own operations and its upstream and downstream
    value chain.
    """
    UPSTREAM = "upstream"
    OWN_OPERATIONS = "own_operations"
    DOWNSTREAM = "downstream"


class ESRSTopic(str, Enum):
    """ESRS sustainability topics as defined in ESRS 1 Appendix A.

    Each topic corresponds to a topical standard (E1-E5, S1-S4, G1)
    and contains multiple sub-topics for detailed assessment.
    """
    E1_CLIMATE = "e1_climate"
    E2_POLLUTION = "e2_pollution"
    E3_WATER = "e3_water"
    E4_BIODIVERSITY = "e4_biodiversity"
    E5_CIRCULAR_ECONOMY = "e5_circular_economy"
    S1_OWN_WORKFORCE = "s1_own_workforce"
    S2_VALUE_CHAIN_WORKERS = "s2_value_chain_workers"
    S3_AFFECTED_COMMUNITIES = "s3_affected_communities"
    S4_CONSUMERS = "s4_consumers"
    G1_BUSINESS_CONDUCT = "g1_business_conduct"


class ImpactType(str, Enum):
    """Type of sustainability impact per ESRS 1 Para 43.

    Impacts can be actual or potential, and positive or negative.
    The assessment methodology differs:
    - Actual negative: severity = f(scale, scope, irremediability)
    - Potential negative: materiality = f(severity, likelihood)
    - Actual positive: severity = f(scale, scope)
    - Potential positive: materiality = f(severity, likelihood)
    """
    ACTUAL_NEGATIVE = "actual_negative"
    POTENTIAL_NEGATIVE = "potential_negative"
    ACTUAL_POSITIVE = "actual_positive"
    POTENTIAL_POSITIVE = "potential_positive"


class ScaleLevel(int, Enum):
    """Scale level for impact assessment criteria (1-5).

    Scale measures how grave or beneficial the impact is.
    Per ESRS 1 Para 44, scale is one of three characteristics
    that determine severity.
    """
    NEGLIGIBLE = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ESRS sustainability matters mapping: each topic maps to its sub-topics
# as defined in ESRS 1 Appendix A and the topical standards.
ESRS_SUSTAINABILITY_MATTERS: Dict[str, List[str]] = {
    "e1_climate": [
        "climate_change_mitigation",
        "climate_change_adaptation",
        "energy",
        "ghg_emissions_scope_1",
        "ghg_emissions_scope_2",
        "ghg_emissions_scope_3",
        "ghg_removal_storage",
        "carbon_credits",
        "internal_carbon_pricing",
    ],
    "e2_pollution": [
        "pollution_of_air",
        "pollution_of_water",
        "pollution_of_soil",
        "pollution_of_living_organisms_food",
        "substances_of_concern",
        "substances_of_very_high_concern",
        "microplastics",
    ],
    "e3_water": [
        "water_consumption",
        "water_withdrawals",
        "water_discharges",
        "water_discharges_in_oceans",
        "water_stress_areas",
    ],
    "e4_biodiversity": [
        "direct_impact_drivers_land_use",
        "direct_impact_drivers_exploitation",
        "direct_impact_drivers_climate_change",
        "direct_impact_drivers_pollution",
        "direct_impact_drivers_invasive_species",
        "impacts_on_species",
        "impacts_on_ecosystems",
        "impacts_on_ecosystem_services",
    ],
    "e5_circular_economy": [
        "resource_inflows",
        "resource_outflows_waste",
        "resource_outflows_products_services",
        "waste_management",
    ],
    "s1_own_workforce": [
        "working_conditions_secure_employment",
        "working_conditions_working_time",
        "working_conditions_adequate_wages",
        "working_conditions_social_dialogue",
        "working_conditions_freedom_of_association",
        "working_conditions_collective_bargaining",
        "working_conditions_work_life_balance",
        "working_conditions_health_and_safety",
        "equal_treatment_gender_equality",
        "equal_treatment_training_skills",
        "equal_treatment_employment_inclusion_of_disabled",
        "equal_treatment_measures_against_violence",
        "equal_treatment_diversity",
        "other_work_related_rights_child_labour",
        "other_work_related_rights_forced_labour",
        "other_work_related_rights_privacy",
    ],
    "s2_value_chain_workers": [
        "working_conditions_in_value_chain",
        "equal_treatment_in_value_chain",
        "other_work_related_rights_in_value_chain",
    ],
    "s3_affected_communities": [
        "communities_economic_social_cultural_rights",
        "communities_civil_political_rights",
        "communities_rights_of_indigenous_peoples",
    ],
    "s4_consumers": [
        "information_related_impacts_on_consumers",
        "personal_safety_of_consumers",
        "social_inclusion_of_consumers",
    ],
    "g1_business_conduct": [
        "corporate_culture",
        "protection_of_whistleblowers",
        "animal_welfare",
        "political_engagement_lobbying",
        "management_of_relationships_with_suppliers",
        "corruption_and_bribery",
    ],
}


# Weights for scale, scope, and irremediability in severity calculation.
# Per ESRS 1 Para 45, each characteristic has equal weight.  We use
# geometric mean (cube root of product), which naturally gives equal
# weight while penalising extreme imbalances more than arithmetic mean.
SCALE_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

SCOPE_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

IRREMEDIABILITY_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

# Likelihood weights for potential impacts (1-5 scale).
LIKELIHOOD_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

# Time horizon discount factors for impact assessment.
# Nearer-term impacts are weighted more heavily in materiality.
TIME_HORIZON_WEIGHTS: Dict[str, Decimal] = {
    "short_term": Decimal("1.00"),
    "medium_term": Decimal("0.85"),
    "long_term": Decimal("0.70"),
}

# Default materiality threshold (on 0-1 normalised scale).
# Matters scoring above this threshold are considered material.
DEFAULT_MATERIALITY_THRESHOLD: Decimal = Decimal("0.40")

# Scale level descriptions for documentation and reporting.
SCALE_DESCRIPTIONS: Dict[int, str] = {
    1: "Negligible: Minimal or no discernible impact",
    2: "Low: Minor impact, easily absorbed or addressed",
    3: "Moderate: Noticeable impact requiring management attention",
    4: "High: Significant impact requiring urgent action",
    5: "Very High: Severe or catastrophic impact, existential concern",
}

SCOPE_DESCRIPTIONS: Dict[int, str] = {
    1: "Negligible: Isolated to single location or small group",
    2: "Low: Limited to a few locations or stakeholder groups",
    3: "Moderate: Affects multiple locations or significant groups",
    4: "High: Widespread across many locations or stakeholder groups",
    5: "Very High: Pervasive across entire value chain or society",
}

IRREMEDIABILITY_DESCRIPTIONS: Dict[int, str] = {
    1: "Fully Remediable: Impact easily reversed at low cost",
    2: "Mostly Remediable: Impact largely reversible with effort",
    3: "Partially Remediable: Impact partly reversible with significant effort",
    4: "Mostly Irremediable: Impact very difficult to reverse",
    5: "Fully Irremediable: Impact permanent and irreversible",
}

LIKELIHOOD_DESCRIPTIONS: Dict[int, str] = {
    1: "Remote: Highly unlikely to occur (< 5% probability)",
    2: "Unlikely: Low probability of occurrence (5-20%)",
    3: "Possible: Moderate probability of occurrence (20-50%)",
    4: "Likely: High probability of occurrence (50-80%)",
    5: "Very Likely: Near certain to occur (> 80% probability)",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class SustainabilityMatter(BaseModel):
    """A sustainability matter to be assessed for impact materiality.

    Represents a specific topic or sub-topic from the ESRS framework
    that the undertaking evaluates as part of its double materiality
    assessment.  Each matter is linked to an ESRS topic, a sub-topic,
    and a value chain stage.
    """
    id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this sustainability matter",
    )
    name: str = Field(
        ...,
        description="Human-readable name of the sustainability matter",
        min_length=1,
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed description of the sustainability matter",
        max_length=5000,
    )
    esrs_topic: ESRSTopic = Field(
        ...,
        description="ESRS topic this matter belongs to (E1-E5, S1-S4, G1)",
    )
    sub_topic: str = Field(
        default="",
        description="Specific sub-topic within the ESRS topic",
        max_length=500,
    )
    value_chain_stage: ValueChainStage = Field(
        default=ValueChainStage.OWN_OPERATIONS,
        description="Stage of the value chain where this matter applies",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Sustainability matter name cannot be empty")
        return v.strip()


class ImpactAssessment(BaseModel):
    """Assessment inputs for a single sustainability matter's impact.

    Contains the scoring criteria used to evaluate impact materiality
    per ESRS 1 Para 43-48.  Scale, scope, and irremediability
    determine severity.  Likelihood is used for potential impacts.
    """
    matter_id: str = Field(
        ...,
        description="ID of the sustainability matter being assessed",
        min_length=1,
    )
    scale: int = Field(
        ...,
        description="Scale of the impact (1=Negligible to 5=Very High)",
        ge=1,
        le=5,
    )
    scope: int = Field(
        ...,
        description="Scope of the impact (1=Negligible to 5=Very High)",
        ge=1,
        le=5,
    )
    irremediability: int = Field(
        ...,
        description="Irremediability of the impact (1=Fully Remediable to 5=Fully Irremediable)",
        ge=1,
        le=5,
    )
    likelihood: int = Field(
        default=3,
        description="Likelihood of occurrence (1=Remote to 5=Very Likely). "
                    "Used only for potential impacts; ignored for actual impacts.",
        ge=1,
        le=5,
    )
    is_actual: bool = Field(
        default=True,
        description="True if this is an actual impact, False if potential",
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.SHORT_TERM,
        description="Time horizon over which the impact is assessed",
    )
    impact_type: ImpactType = Field(
        default=ImpactType.ACTUAL_NEGATIVE,
        description="Type of impact (actual/potential, positive/negative)",
    )
    rationale: str = Field(
        default="",
        description="Rationale or justification for the scores assigned",
        max_length=5000,
    )

    @field_validator("scale", "scope", "irremediability", "likelihood")
    @classmethod
    def validate_score_range(cls, v: int) -> int:
        """Validate score is within 1-5 range."""
        if v < 1 or v > 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v


class ImpactMaterialityResult(BaseModel):
    """Result of impact materiality assessment for a single matter.

    Contains the calculated severity score, likelihood score, and
    combined impact materiality score, along with the materiality
    determination and ranking.  Full provenance is tracked via
    SHA-256 hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of calculation (UTC)",
    )
    matter_id: str = Field(
        ...,
        description="ID of the sustainability matter assessed",
    )
    matter_name: str = Field(
        default="",
        description="Name of the sustainability matter",
    )
    esrs_topic: str = Field(
        default="",
        description="ESRS topic of the assessed matter",
    )
    impact_type: str = Field(
        default="",
        description="Type of impact assessed",
    )
    severity_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Calculated severity score (0-1 scale, 3 decimal places)",
    )
    likelihood_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Likelihood score (0-1 scale). 1.0 for actual impacts.",
    )
    time_horizon_weight: Decimal = Field(
        default=Decimal("1.000"),
        description="Time horizon weighting factor applied",
    )
    impact_materiality_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Combined impact materiality score (0-1 scale)",
    )
    is_material: bool = Field(
        default=False,
        description="Whether the matter is material based on threshold",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD,
        description="Materiality threshold used for determination",
    )
    ranking: int = Field(
        default=0,
        description="Ranking among all assessed matters (1 = highest score)",
    )
    rationale: str = Field(
        default="",
        description="Explanation of the materiality determination",
    )
    scale_input: int = Field(default=0, description="Scale input value (1-5)")
    scope_input: int = Field(default=0, description="Scope input value (1-5)")
    irremediability_input: int = Field(
        default=0, description="Irremediability input value (1-5)"
    )
    likelihood_input: int = Field(
        default=0, description="Likelihood input value (1-5)"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )


class BatchImpactResult(BaseModel):
    """Result of batch impact materiality assessment.

    Contains results for all assessed matters, summary statistics,
    and full provenance tracking.
    """
    batch_id: str = Field(
        default_factory=_new_uuid,
        description="Unique batch identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this batch",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of batch calculation (UTC)",
    )
    total_matters: int = Field(
        default=0,
        description="Total number of matters assessed",
    )
    material_count: int = Field(
        default=0,
        description="Number of matters determined as material",
    )
    not_material_count: int = Field(
        default=0,
        description="Number of matters determined as not material",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD,
        description="Materiality threshold used",
    )
    results: List[ImpactMaterialityResult] = Field(
        default_factory=list,
        description="Individual results for each matter, ranked by score",
    )
    by_topic: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material matters by ESRS topic",
    )
    by_impact_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material matters by impact type",
    )
    avg_severity: Decimal = Field(
        default=Decimal("0.000"),
        description="Average severity score across all matters",
    )
    avg_materiality_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Average materiality score across all matters",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire batch result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ImpactMaterialityEngine:
    """Impact materiality scoring engine per ESRS 1 Para 43-48.

    Provides deterministic, zero-hallucination calculations for:
    - Severity scoring (geometric mean of scale, scope, irremediability)
    - Likelihood scoring for potential impacts
    - Combined impact materiality scoring
    - Materiality threshold determination
    - Ranking of sustainability matters by impact score
    - Batch assessment with summary statistics

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Calculation Methodology:
        1. Severity = geometric_mean(scale_weight, scope_weight,
           irremediability_weight)
           = (scale_w * scope_w * irremediability_w) ^ (1/3)
        2. Likelihood Score = likelihood_weight (for potential impacts)
           or 1.0 (for actual impacts)
        3. Time Horizon Weight = factor from TIME_HORIZON_WEIGHTS
        4. Impact Materiality Score = severity * likelihood_score
           * time_horizon_weight
        5. is_material = score >= threshold

    Usage::

        engine = ImpactMaterialityEngine()
        matter = SustainabilityMatter(
            name="GHG Emissions Scope 1",
            esrs_topic=ESRSTopic.E1_CLIMATE,
            sub_topic="ghg_emissions_scope_1",
        )
        assessment = ImpactAssessment(
            matter_id=matter.id,
            scale=4,
            scope=4,
            irremediability=3,
            is_actual=True,
            time_horizon=TimeHorizon.SHORT_TERM,
        )
        result = engine.assess_impact(matter, assessment)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Core Calculation Methods                                             #
    # ------------------------------------------------------------------ #

    def calculate_severity(
        self,
        scale: int,
        scope: int,
        irremediability: int,
    ) -> Decimal:
        """Calculate severity score using geometric mean.

        Per ESRS 1 Para 44-45, severity is determined by three
        characteristics with equal weight: scale, scope, and
        irremediable character.  Geometric mean is used because it:
        - Gives equal weight to each characteristic
        - Penalises extreme imbalances more than arithmetic mean
        - Returns a value on the same 0-1 scale as inputs

        Formula:
            severity = (scale_w * scope_w * irremediability_w) ^ (1/3)

        Args:
            scale: Scale of impact (1-5).
            scope: Scope of impact (1-5).
            irremediability: Irremediability of impact (1-5).

        Returns:
            Severity score as Decimal (0-1 scale, 4 decimal places).

        Raises:
            ValueError: If any input is outside 1-5 range.
        """
        if not (1 <= scale <= 5):
            raise ValueError(f"Scale must be 1-5, got {scale}")
        if not (1 <= scope <= 5):
            raise ValueError(f"Scope must be 1-5, got {scope}")
        if not (1 <= irremediability <= 5):
            raise ValueError(
                f"Irremediability must be 1-5, got {irremediability}"
            )

        scale_w = SCALE_WEIGHTS[scale]
        scope_w = SCOPE_WEIGHTS[scope]
        irrem_w = IRREMEDIABILITY_WEIGHTS[irremediability]

        # Geometric mean: (a * b * c) ^ (1/3)
        product = scale_w * scope_w * irrem_w
        # Use Decimal exponentiation for deterministic cube root
        # Decimal.ln() and Decimal.exp() are not available in all
        # implementations, so we use float conversion for the cube root
        # and then convert back to Decimal for precision.
        product_float = float(product)
        if product_float <= 0.0:
            return Decimal("0.0000")

        cube_root = product_float ** (1.0 / 3.0)
        severity = _round_val(_decimal(cube_root), 4)
        return severity

    def calculate_impact_score(
        self,
        severity: Decimal,
        likelihood: int,
        is_actual: bool,
        time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
    ) -> Decimal:
        """Calculate combined impact materiality score.

        For actual impacts:
            score = severity * 1.0 * time_horizon_weight
        For potential impacts:
            score = severity * likelihood_weight * time_horizon_weight

        Args:
            severity: Severity score (0-1 Decimal).
            likelihood: Likelihood rating (1-5). Ignored for actual
                impacts.
            is_actual: True if actual impact, False if potential.
            time_horizon: Time horizon for the impact.

        Returns:
            Impact materiality score as Decimal (0-1 scale, 4 decimal
            places).

        Raises:
            ValueError: If likelihood is outside 1-5 range.
        """
        if not (1 <= likelihood <= 5):
            raise ValueError(f"Likelihood must be 1-5, got {likelihood}")

        if is_actual:
            likelihood_score = Decimal("1.0000")
        else:
            likelihood_score = _round_val(
                LIKELIHOOD_WEIGHTS[likelihood], 4
            )

        time_weight = TIME_HORIZON_WEIGHTS.get(
            time_horizon.value, Decimal("1.00")
        )

        score = severity * likelihood_score * time_weight
        return _round_val(score, 4)

    # ------------------------------------------------------------------ #
    # Single Assessment                                                    #
    # ------------------------------------------------------------------ #

    def assess_impact(
        self,
        matter: SustainabilityMatter,
        assessment: ImpactAssessment,
        threshold: Optional[Decimal] = None,
    ) -> ImpactMaterialityResult:
        """Assess impact materiality for a single sustainability matter.

        Executes the full impact materiality assessment workflow:
        1. Calculate severity from scale, scope, irremediability
        2. Determine likelihood score (actual vs potential)
        3. Apply time horizon weighting
        4. Calculate combined impact materiality score
        5. Compare against threshold for materiality determination
        6. Generate provenance hash

        Args:
            matter: The sustainability matter to assess.
            assessment: Assessment inputs (scores and metadata).
            threshold: Materiality threshold (default 0.40).

        Returns:
            ImpactMaterialityResult with complete provenance.

        Raises:
            ValueError: If assessment.matter_id does not match matter.id.
        """
        t0 = time.perf_counter()

        if assessment.matter_id != matter.id:
            raise ValueError(
                f"Assessment matter_id '{assessment.matter_id}' does not "
                f"match matter.id '{matter.id}'"
            )

        if threshold is None:
            threshold = DEFAULT_MATERIALITY_THRESHOLD

        # Step 1: Calculate severity
        severity = self.calculate_severity(
            assessment.scale,
            assessment.scope,
            assessment.irremediability,
        )

        # Step 2: Determine likelihood score
        if assessment.is_actual:
            likelihood_score = Decimal("1.0000")
        else:
            likelihood_score = _round_val(
                LIKELIHOOD_WEIGHTS[assessment.likelihood], 4
            )

        # Step 3: Time horizon weight
        time_weight = TIME_HORIZON_WEIGHTS.get(
            assessment.time_horizon.value, Decimal("1.00")
        )

        # Step 4: Combined score
        impact_score = self.calculate_impact_score(
            severity,
            assessment.likelihood,
            assessment.is_actual,
            assessment.time_horizon,
        )

        # Step 5: Materiality determination
        is_material = impact_score >= threshold

        # Step 6: Build rationale
        rationale = self._build_rationale(
            matter, assessment, severity, likelihood_score,
            time_weight, impact_score, is_material, threshold,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ImpactMaterialityResult(
            matter_id=matter.id,
            matter_name=matter.name,
            esrs_topic=matter.esrs_topic.value,
            impact_type=assessment.impact_type.value,
            severity_score=_round_val(severity, 3),
            likelihood_score=_round_val(likelihood_score, 3),
            time_horizon_weight=_round_val(time_weight, 3),
            impact_materiality_score=_round_val(impact_score, 3),
            is_material=is_material,
            threshold_used=threshold,
            rationale=rationale,
            scale_input=assessment.scale,
            scope_input=assessment.scope,
            irremediability_input=assessment.irremediability,
            likelihood_input=assessment.likelihood,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Batch Assessment                                                     #
    # ------------------------------------------------------------------ #

    def batch_assess(
        self,
        matters: List[SustainabilityMatter],
        assessments: List[ImpactAssessment],
        threshold: Optional[Decimal] = None,
    ) -> BatchImpactResult:
        """Assess impact materiality for multiple sustainability matters.

        Processes all matter-assessment pairs, ranks results by score,
        and generates summary statistics.

        Args:
            matters: List of sustainability matters to assess.
            assessments: List of assessments corresponding to matters.
                Each assessment's matter_id must match a matter's id.
            threshold: Materiality threshold (default 0.40).

        Returns:
            BatchImpactResult with all individual results and summaries.

        Raises:
            ValueError: If matters or assessments lists are empty, or
                if assessment matter_ids do not match any matter.
        """
        t0 = time.perf_counter()

        if not matters:
            raise ValueError("At least one SustainabilityMatter is required")
        if not assessments:
            raise ValueError("At least one ImpactAssessment is required")

        if threshold is None:
            threshold = DEFAULT_MATERIALITY_THRESHOLD

        # Build matter lookup
        matter_lookup: Dict[str, SustainabilityMatter] = {
            m.id: m for m in matters
        }

        # Assess each pair
        results: List[ImpactMaterialityResult] = []
        for asmt in assessments:
            matter = matter_lookup.get(asmt.matter_id)
            if matter is None:
                logger.warning(
                    "No matter found for assessment matter_id=%s, skipping",
                    asmt.matter_id,
                )
                continue
            result = self.assess_impact(matter, asmt, threshold)
            results.append(result)

        # Rank by score descending
        results = self.rank_impacts(results)

        # Summary statistics
        material_results = [r for r in results if r.is_material]
        material_count = len(material_results)
        not_material_count = len(results) - material_count

        # By topic counts (material only)
        by_topic: Dict[str, int] = {}
        for r in material_results:
            topic = r.esrs_topic
            by_topic[topic] = by_topic.get(topic, 0) + 1

        # By impact type counts (material only)
        by_type: Dict[str, int] = {}
        for r in material_results:
            itype = r.impact_type
            by_type[itype] = by_type.get(itype, 0) + 1

        # Average scores
        if results:
            avg_sev = _round_val(
                sum(r.severity_score for r in results) / _decimal(len(results)),
                3,
            )
            avg_mat = _round_val(
                sum(r.impact_materiality_score for r in results)
                / _decimal(len(results)),
                3,
            )
        else:
            avg_sev = Decimal("0.000")
            avg_mat = Decimal("0.000")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        batch_result = BatchImpactResult(
            total_matters=len(results),
            material_count=material_count,
            not_material_count=not_material_count,
            threshold_used=threshold,
            results=results,
            by_topic=by_topic,
            by_impact_type=by_type,
            avg_severity=avg_sev,
            avg_materiality_score=avg_mat,
            processing_time_ms=elapsed_ms,
        )

        batch_result.provenance_hash = _compute_hash(batch_result)
        return batch_result

    # ------------------------------------------------------------------ #
    # Ranking and Filtering                                                #
    # ------------------------------------------------------------------ #

    def rank_impacts(
        self,
        results: List[ImpactMaterialityResult],
    ) -> List[ImpactMaterialityResult]:
        """Rank impact materiality results by score descending.

        Assigns a ranking number to each result (1 = highest score).
        Ties receive the same ranking.

        Args:
            results: List of ImpactMaterialityResult to rank.

        Returns:
            Sorted list with ranking assigned.
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.impact_materiality_score,
            reverse=True,
        )
        rank = 1
        for i, result in enumerate(sorted_results):
            if i > 0 and (
                result.impact_materiality_score
                < sorted_results[i - 1].impact_materiality_score
            ):
                rank = i + 1
            result.ranking = rank
        return sorted_results

    def apply_threshold(
        self,
        results: List[ImpactMaterialityResult],
        threshold: Decimal,
    ) -> List[ImpactMaterialityResult]:
        """Filter results to only those meeting the materiality threshold.

        Args:
            results: List of ImpactMaterialityResult.
            threshold: Materiality threshold (Decimal, 0-1 scale).

        Returns:
            Filtered list containing only material results.
        """
        filtered = []
        for r in results:
            if r.impact_materiality_score >= threshold:
                r.is_material = True
                r.threshold_used = threshold
                filtered.append(r)
        return filtered

    # ------------------------------------------------------------------ #
    # Utility: Get ESRS Sub-Topics                                         #
    # ------------------------------------------------------------------ #

    def get_sustainability_matters(
        self,
        topic: ESRSTopic,
    ) -> List[str]:
        """Return the list of sub-topics for a given ESRS topic.

        Args:
            topic: ESRS topic enum value.

        Returns:
            List of sub-topic strings.
        """
        return ESRS_SUSTAINABILITY_MATTERS.get(topic.value, [])

    def get_all_topics(self) -> Dict[str, List[str]]:
        """Return the complete ESRS sustainability matters mapping.

        Returns:
            Dict mapping topic string to list of sub-topic strings.
        """
        return dict(ESRS_SUSTAINABILITY_MATTERS)

    # ------------------------------------------------------------------ #
    # Positive Impact Assessment                                           #
    # ------------------------------------------------------------------ #

    def calculate_positive_severity(
        self,
        scale: int,
        scope: int,
    ) -> Decimal:
        """Calculate severity for positive impacts.

        Per ESRS 1 Para 47, for positive impacts severity is
        determined by scale and scope only (irremediability is not
        applicable).  We use the geometric mean of two factors.

        Formula:
            severity = (scale_w * scope_w) ^ (1/2)

        Args:
            scale: Scale of positive impact (1-5).
            scope: Scope of positive impact (1-5).

        Returns:
            Severity score as Decimal (0-1 scale, 4 decimal places).

        Raises:
            ValueError: If inputs are outside 1-5 range.
        """
        if not (1 <= scale <= 5):
            raise ValueError(f"Scale must be 1-5, got {scale}")
        if not (1 <= scope <= 5):
            raise ValueError(f"Scope must be 1-5, got {scope}")

        scale_w = SCALE_WEIGHTS[scale]
        scope_w = SCOPE_WEIGHTS[scope]

        product_float = float(scale_w * scope_w)
        if product_float <= 0.0:
            return Decimal("0.0000")

        sqrt_val = product_float ** 0.5
        return _round_val(_decimal(sqrt_val), 4)

    def assess_positive_impact(
        self,
        matter: SustainabilityMatter,
        scale: int,
        scope: int,
        likelihood: int = 5,
        is_actual: bool = True,
        time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
        threshold: Optional[Decimal] = None,
    ) -> ImpactMaterialityResult:
        """Assess a positive impact's materiality.

        Uses the two-factor severity calculation (scale, scope) and
        applies likelihood and time horizon adjustments.

        Args:
            matter: The sustainability matter.
            scale: Scale of positive impact (1-5).
            scope: Scope of positive impact (1-5).
            likelihood: Likelihood (1-5), used only if not actual.
            is_actual: True if actual positive impact.
            time_horizon: Time horizon for the impact.
            threshold: Materiality threshold (default 0.40).

        Returns:
            ImpactMaterialityResult with full provenance.
        """
        t0 = time.perf_counter()

        if threshold is None:
            threshold = DEFAULT_MATERIALITY_THRESHOLD

        severity = self.calculate_positive_severity(scale, scope)

        if is_actual:
            likelihood_score = Decimal("1.0000")
            impact_type = ImpactType.ACTUAL_POSITIVE
        else:
            likelihood_score = _round_val(
                LIKELIHOOD_WEIGHTS[likelihood], 4
            )
            impact_type = ImpactType.POTENTIAL_POSITIVE

        time_weight = TIME_HORIZON_WEIGHTS.get(
            time_horizon.value, Decimal("1.00")
        )

        impact_score = _round_val(
            severity * likelihood_score * time_weight, 4
        )
        is_material = impact_score >= threshold

        rationale = (
            f"Positive impact assessment for '{matter.name}': "
            f"severity={severity} (scale={scale}, scope={scope}), "
            f"likelihood_score={likelihood_score}, "
            f"time_horizon_weight={time_weight}, "
            f"impact_score={impact_score}. "
            f"{'MATERIAL' if is_material else 'NOT MATERIAL'} "
            f"(threshold={threshold})."
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ImpactMaterialityResult(
            matter_id=matter.id,
            matter_name=matter.name,
            esrs_topic=matter.esrs_topic.value,
            impact_type=impact_type.value,
            severity_score=_round_val(severity, 3),
            likelihood_score=_round_val(likelihood_score, 3),
            time_horizon_weight=_round_val(time_weight, 3),
            impact_materiality_score=_round_val(impact_score, 3),
            is_material=is_material,
            threshold_used=threshold,
            rationale=rationale,
            scale_input=scale,
            scope_input=scope,
            irremediability_input=0,
            likelihood_input=likelihood,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Score Interpretation                                                 #
    # ------------------------------------------------------------------ #

    def interpret_score(self, score: Decimal) -> str:
        """Return a human-readable interpretation of a materiality score.

        This is a deterministic mapping, not an LLM interpretation.

        Args:
            score: Impact materiality score (0-1 Decimal).

        Returns:
            Interpretation string.
        """
        score_float = float(score)
        if score_float >= 0.80:
            return "Critical: Highest priority sustainability matter"
        elif score_float >= 0.60:
            return "High: Significant sustainability matter requiring action"
        elif score_float >= 0.40:
            return "Moderate: Material sustainability matter for disclosure"
        elif score_float >= 0.20:
            return "Low: Below materiality threshold, monitor for changes"
        else:
            return "Negligible: Not material, minimal impact identified"

    def get_score_breakdown(
        self,
        result: ImpactMaterialityResult,
    ) -> Dict[str, Any]:
        """Return a structured breakdown of a result's scoring.

        Useful for audit documentation and stakeholder communication.

        Args:
            result: An ImpactMaterialityResult.

        Returns:
            Dict with all scoring components and descriptions.
        """
        return {
            "matter_id": result.matter_id,
            "matter_name": result.matter_name,
            "esrs_topic": result.esrs_topic,
            "impact_type": result.impact_type,
            "inputs": {
                "scale": {
                    "value": result.scale_input,
                    "description": SCALE_DESCRIPTIONS.get(
                        result.scale_input, ""
                    ),
                },
                "scope": {
                    "value": result.scope_input,
                    "description": SCOPE_DESCRIPTIONS.get(
                        result.scope_input, ""
                    ),
                },
                "irremediability": {
                    "value": result.irremediability_input,
                    "description": IRREMEDIABILITY_DESCRIPTIONS.get(
                        result.irremediability_input, ""
                    ),
                },
                "likelihood": {
                    "value": result.likelihood_input,
                    "description": LIKELIHOOD_DESCRIPTIONS.get(
                        result.likelihood_input, ""
                    ),
                },
            },
            "scores": {
                "severity_score": str(result.severity_score),
                "likelihood_score": str(result.likelihood_score),
                "time_horizon_weight": str(result.time_horizon_weight),
                "impact_materiality_score": str(
                    result.impact_materiality_score
                ),
            },
            "determination": {
                "is_material": result.is_material,
                "threshold": str(result.threshold_used),
                "ranking": result.ranking,
                "interpretation": self.interpret_score(
                    result.impact_materiality_score
                ),
            },
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private: Rationale Builder                                           #
    # ------------------------------------------------------------------ #

    def _build_rationale(
        self,
        matter: SustainabilityMatter,
        assessment: ImpactAssessment,
        severity: Decimal,
        likelihood_score: Decimal,
        time_weight: Decimal,
        impact_score: Decimal,
        is_material: bool,
        threshold: Decimal,
    ) -> str:
        """Build a deterministic rationale string for the assessment.

        This is a template-based rationale, not generated by an LLM.

        Args:
            matter: The sustainability matter.
            assessment: The assessment inputs.
            severity: Calculated severity score.
            likelihood_score: Calculated likelihood score.
            time_weight: Time horizon weight applied.
            impact_score: Final impact materiality score.
            is_material: Materiality determination.
            threshold: Threshold used.

        Returns:
            Rationale string.
        """
        impact_type_label = assessment.impact_type.value.replace("_", " ")
        actual_label = "actual" if assessment.is_actual else "potential"

        parts = [
            f"Impact materiality assessment for '{matter.name}' "
            f"({matter.esrs_topic.value}): ",
            f"Impact type: {impact_type_label}. ",
            f"Severity = geometric_mean(scale={assessment.scale}, "
            f"scope={assessment.scope}, "
            f"irremediability={assessment.irremediability}) = {severity}. ",
        ]

        if not assessment.is_actual:
            parts.append(
                f"Likelihood = {assessment.likelihood} "
                f"(score: {likelihood_score}). "
            )
        else:
            parts.append(
                f"Actual impact, likelihood score = {likelihood_score}. "
            )

        parts.append(
            f"Time horizon: {assessment.time_horizon.value} "
            f"(weight: {time_weight}). "
        )
        parts.append(
            f"Impact materiality score = {severity} * {likelihood_score} "
            f"* {time_weight} = {impact_score}. "
        )

        if is_material:
            parts.append(
                f"MATERIAL: Score {impact_score} >= threshold {threshold}."
            )
        else:
            parts.append(
                f"NOT MATERIAL: Score {impact_score} < threshold {threshold}."
            )

        if assessment.rationale:
            parts.append(f" Assessor rationale: {assessment.rationale}")

        return "".join(parts)
