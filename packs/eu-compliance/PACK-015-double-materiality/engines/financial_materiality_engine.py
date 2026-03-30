# -*- coding: utf-8 -*-
"""
FinancialMaterialityEngine - PACK-015 Double Materiality Engine 2
===================================================================

Scores financial materiality per ESRS 1 Paragraphs 49-51.

Under the European Sustainability Reporting Standards (ESRS), financial
materiality assesses whether a sustainability matter generates or may
generate risks or opportunities that have or may have a material
influence on the undertaking's financial position, financial performance,
cash flows, access to finance, or cost of capital over the short-,
medium-, or long-term.

ESRS 1 Financial Materiality Assessment Framework:
    - Para 49: A sustainability matter is material from a financial
      perspective if it triggers or may trigger material financial
      effects on the undertaking.  This includes effects deriving
      from risks or opportunities.
    - Para 50: Financial effects include those on the undertaking's
      development, financial position, financial performance, cash
      flows, access to finance, or cost of capital.
    - Para 51: Dependencies on natural, human, and social resources
      can be sources of financial risks or opportunities.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS 1 General Requirements, Chapter 3 Double Materiality
    - EFRAG Implementation Guidance IG 1 (Materiality Assessment)
    - IFRS S1 / ISSB (cross-reference for financial materiality)
    - IAS 1 (definition of material information)

Zero-Hallucination:
    - Financial score = magnitude_weight * likelihood_weight
      * time_horizon_weight (deterministic arithmetic)
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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TimeHorizon(str, Enum):
    """Time horizon for financial materiality assessment.

    Per ESRS 1 Para 77, undertakings shall consider short-, medium-,
    and long-term time horizons.
    """
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class FinancialMagnitude(int, Enum):
    """Magnitude of potential financial effect (1-5 scale).

    Measures the size of the financial impact relative to the
    undertaking's financial position, performance, and cash flows.
    """
    NEGLIGIBLE = 1
    LOW = 2
    MODERATE = 3
    SIGNIFICANT = 4
    CRITICAL = 5

class FinancialLikelihood(int, Enum):
    """Likelihood of the financial effect materialising (1-5 scale).

    Assesses the probability that the risk or opportunity will
    result in a material financial effect.
    """
    REMOTE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5

class AffectedResource(str, Enum):
    """Financial resource or metric affected by the sustainability matter.

    Per ESRS 1 Para 50, financial effects include impacts on the
    undertaking's development, financial position, financial performance,
    cash flows, access to finance, or cost of capital.
    """
    REVENUE = "revenue"
    COST = "cost"
    ASSETS = "assets"
    LIABILITIES = "liabilities"
    CAPITAL = "capital"
    ACCESS_TO_FINANCE = "access_to_finance"

class RiskOrOpportunity(str, Enum):
    """Whether the matter represents a risk, opportunity, or both.

    Per ESRS 1 Para 49, financial materiality arises from risks and/or
    opportunities that have or may have a material influence on the
    undertaking's finances.
    """
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    BOTH = "both"

class ESRSTopic(str, Enum):
    """ESRS sustainability topics (duplicated for self-contained engine)."""
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Time horizon weights for financial materiality.
# Nearer-term financial effects carry more weight because they are
# more certain and more directly actionable.
TIME_HORIZON_WEIGHTS: Dict[str, Decimal] = {
    "short_term": Decimal("1.00"),
    "medium_term": Decimal("0.80"),
    "long_term": Decimal("0.60"),
}

# Magnitude weights mapping integer score to 0-1 scale.
MAGNITUDE_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

# Likelihood weights mapping integer score to 0-1 scale.
LIKELIHOOD_WEIGHTS: Dict[int, Decimal] = {
    1: Decimal("0.20"),
    2: Decimal("0.40"),
    3: Decimal("0.60"),
    4: Decimal("0.80"),
    5: Decimal("1.00"),
}

# Magnitude descriptions for reporting and documentation.
MAGNITUDE_DESCRIPTIONS: Dict[int, str] = {
    1: "Negligible: Immaterial financial effect (< 1% of relevant metric)",
    2: "Low: Minor financial effect (1-3% of relevant metric)",
    3: "Moderate: Noticeable financial effect (3-5% of relevant metric)",
    4: "Significant: Substantial financial effect (5-10% of relevant metric)",
    5: "Critical: Major financial effect (> 10% of relevant metric)",
}

# Likelihood descriptions for reporting and documentation.
LIKELIHOOD_DESCRIPTIONS: Dict[int, str] = {
    1: "Remote: Very unlikely to materialise (< 5% probability)",
    2: "Unlikely: Low probability of materialising (5-20%)",
    3: "Possible: Moderate probability of materialising (20-50%)",
    4: "Likely: High probability of materialising (50-80%)",
    5: "Very Likely: Near certain to materialise (> 80% probability)",
}

# Default financial materiality threshold (0-1 normalised scale).
DEFAULT_FINANCIAL_THRESHOLD: Decimal = Decimal("0.40")

# Financial KPI map: maps each ESRS topic to the financial KPIs
# most likely to be affected, per sector-agnostic ESRS guidance.
FINANCIAL_KPI_MAP: Dict[str, List[str]] = {
    "e1_climate": [
        "carbon_pricing_costs",
        "stranded_asset_risk",
        "energy_cost_savings",
        "green_revenue_share",
        "climate_capex",
        "insurance_premium_changes",
        "cost_of_capital_impact",
    ],
    "e2_pollution": [
        "remediation_costs",
        "regulatory_fines",
        "pollution_abatement_capex",
        "health_liability_provisions",
        "product_reformulation_costs",
    ],
    "e3_water": [
        "water_procurement_costs",
        "water_treatment_costs",
        "operational_disruption_risk",
        "water_efficiency_savings",
        "regulatory_compliance_costs",
    ],
    "e4_biodiversity": [
        "land_use_remediation_costs",
        "ecosystem_service_dependency_value",
        "biodiversity_offset_costs",
        "permit_licence_risk",
        "supply_chain_disruption_risk",
    ],
    "e5_circular_economy": [
        "raw_material_cost_savings",
        "waste_management_costs",
        "epr_fee_obligations",
        "circular_revenue_streams",
        "resource_efficiency_gains",
    ],
    "s1_own_workforce": [
        "employee_turnover_costs",
        "recruitment_costs",
        "training_investment",
        "health_safety_costs",
        "litigation_liability",
        "productivity_impact",
    ],
    "s2_value_chain_workers": [
        "supply_chain_disruption_costs",
        "due_diligence_compliance_costs",
        "supplier_switching_costs",
        "reputational_impact_on_revenue",
    ],
    "s3_affected_communities": [
        "social_licence_to_operate_risk",
        "community_investment_costs",
        "legal_claims_provisions",
        "project_delay_costs",
    ],
    "s4_consumers": [
        "product_liability_costs",
        "consumer_trust_revenue_impact",
        "data_protection_compliance_costs",
        "product_recall_provisions",
    ],
    "g1_business_conduct": [
        "anti_corruption_compliance_costs",
        "regulatory_fine_provisions",
        "governance_restructuring_costs",
        "reputation_impact_on_valuation",
        "whistleblower_claim_costs",
    ],
}

# Resource descriptions for reporting.
RESOURCE_DESCRIPTIONS: Dict[str, str] = {
    "revenue": "Impact on the undertaking's revenue streams",
    "cost": "Impact on operating or capital costs",
    "assets": "Impact on asset values (impairment, revaluation)",
    "liabilities": "Impact on provisions, contingent liabilities",
    "capital": "Impact on cost of capital or equity valuation",
    "access_to_finance": "Impact on ability to raise debt or equity",
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class FinancialImpact(BaseModel):
    """Input data for financial materiality assessment of a matter.

    Contains the magnitude and likelihood scores, time horizon,
    and metadata about the affected financial resources and trigger
    events that could cause the financial effect to materialise.
    """
    matter_id: str = Field(
        ...,
        description="ID of the sustainability matter being assessed",
        min_length=1,
    )
    matter_name: str = Field(
        default="",
        description="Name of the sustainability matter",
        max_length=500,
    )
    esrs_topic: ESRSTopic = Field(
        default=ESRSTopic.E1_CLIMATE,
        description="ESRS topic this matter belongs to",
    )
    magnitude: int = Field(
        ...,
        description="Magnitude of financial effect (1=Negligible to 5=Critical)",
        ge=1,
        le=5,
    )
    likelihood: int = Field(
        ...,
        description="Likelihood of financial effect materialising "
                    "(1=Remote to 5=Very Likely)",
        ge=1,
        le=5,
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.SHORT_TERM,
        description="Time horizon for the financial effect",
    )
    risk_or_opportunity: RiskOrOpportunity = Field(
        default=RiskOrOpportunity.RISK,
        description="Whether this represents a risk, opportunity, or both",
    )
    affected_resources: List[AffectedResource] = Field(
        default_factory=list,
        description="Financial resources affected by this matter",
    )
    trigger_events: List[str] = Field(
        default_factory=list,
        description="Events or conditions that could trigger the financial "
                    "effect (e.g., 'carbon price increase', 'regulation change')",
    )
    estimated_financial_range_low_eur: Optional[float] = Field(
        default=None,
        description="Lower bound of estimated financial effect (EUR)",
    )
    estimated_financial_range_high_eur: Optional[float] = Field(
        default=None,
        description="Upper bound of estimated financial effect (EUR)",
    )
    rationale: str = Field(
        default="",
        description="Rationale for the magnitude and likelihood scores",
        max_length=5000,
    )

    @field_validator("magnitude", "likelihood")
    @classmethod
    def validate_score_range(cls, v: int) -> int:
        """Validate score is within 1-5 range."""
        if v < 1 or v > 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v

class FinancialMaterialityResult(BaseModel):
    """Result of financial materiality assessment for a single matter.

    Contains the calculated financial score, magnitude and likelihood
    scores, time horizon weight, materiality determination, and
    full provenance tracking.
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
        default_factory=utcnow,
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
    financial_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Combined financial materiality score (0-1 scale)",
    )
    magnitude_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Normalised magnitude score (0-1 scale)",
    )
    likelihood_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Normalised likelihood score (0-1 scale)",
    )
    time_horizon_weight: Decimal = Field(
        default=Decimal("1.000"),
        description="Time horizon weighting factor applied",
    )
    is_material: bool = Field(
        default=False,
        description="Whether the matter is financially material",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_FINANCIAL_THRESHOLD,
        description="Financial materiality threshold used",
    )
    risk_or_opportunity: str = Field(
        default="risk",
        description="Whether this is a risk, opportunity, or both",
    )
    affected_kpis: List[str] = Field(
        default_factory=list,
        description="Financial KPIs potentially affected",
    )
    affected_resources: List[str] = Field(
        default_factory=list,
        description="Financial resources affected",
    )
    magnitude_input: int = Field(
        default=0,
        description="Magnitude input value (1-5)",
    )
    likelihood_input: int = Field(
        default=0,
        description="Likelihood input value (1-5)",
    )
    ranking: int = Field(
        default=0,
        description="Ranking among all assessed matters (1 = highest score)",
    )
    rationale: str = Field(
        default="",
        description="Explanation of the financial materiality determination",
    )
    estimated_financial_range_low_eur: Optional[float] = Field(
        default=None,
        description="Lower bound of estimated financial effect (EUR)",
    )
    estimated_financial_range_high_eur: Optional[float] = Field(
        default=None,
        description="Upper bound of estimated financial effect (EUR)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

class BatchFinancialResult(BaseModel):
    """Result of batch financial materiality assessment.

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
        default_factory=utcnow,
        description="Timestamp of batch calculation (UTC)",
    )
    total_matters: int = Field(
        default=0,
        description="Total number of matters assessed",
    )
    material_count: int = Field(
        default=0,
        description="Number of matters determined as financially material",
    )
    not_material_count: int = Field(
        default=0,
        description="Number of matters determined as not financially material",
    )
    risk_count: int = Field(
        default=0,
        description="Number of material risks identified",
    )
    opportunity_count: int = Field(
        default=0,
        description="Number of material opportunities identified",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_FINANCIAL_THRESHOLD,
        description="Financial materiality threshold used",
    )
    results: List[FinancialMaterialityResult] = Field(
        default_factory=list,
        description="Individual results for each matter, ranked by score",
    )
    by_topic: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material matters by ESRS topic",
    )
    by_risk_opportunity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material matters by risk/opportunity type",
    )
    avg_financial_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Average financial score across all matters",
    )
    avg_magnitude: Decimal = Field(
        default=Decimal("0.000"),
        description="Average magnitude score across all matters",
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

class FinancialMaterialityEngine:
    """Financial materiality scoring engine per ESRS 1 Para 49-51.

    Provides deterministic, zero-hallucination calculations for:
    - Financial score from magnitude and likelihood
    - Time horizon weighting
    - Financial materiality threshold determination
    - Ranking of matters by financial score
    - Batch assessment with summary statistics
    - Financial KPI impact mapping

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Calculation Methodology:
        1. Magnitude Score = MAGNITUDE_WEIGHTS[magnitude]
        2. Likelihood Score = LIKELIHOOD_WEIGHTS[likelihood]
        3. Time Horizon Weight = TIME_HORIZON_WEIGHTS[time_horizon]
        4. Financial Score = magnitude_score * likelihood_score
           * time_horizon_weight
        5. is_material = financial_score >= threshold

    Usage::

        engine = FinancialMaterialityEngine()
        impact = FinancialImpact(
            matter_id="matter-001",
            matter_name="Carbon pricing risk",
            esrs_topic=ESRSTopic.E1_CLIMATE,
            magnitude=4,
            likelihood=4,
            time_horizon=TimeHorizon.SHORT_TERM,
            risk_or_opportunity=RiskOrOpportunity.RISK,
            affected_resources=[AffectedResource.COST],
        )
        result = engine.assess_financial_impact(impact)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Core Calculation Methods                                             #
    # ------------------------------------------------------------------ #

    def calculate_financial_score(
        self,
        magnitude: int,
        likelihood: int,
        time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
    ) -> Decimal:
        """Calculate the combined financial materiality score.

        Formula:
            score = magnitude_weight * likelihood_weight
                    * time_horizon_weight

        Args:
            magnitude: Magnitude of financial effect (1-5).
            likelihood: Likelihood of effect materialising (1-5).
            time_horizon: Time horizon for the effect.

        Returns:
            Financial materiality score as Decimal (0-1 scale).

        Raises:
            ValueError: If magnitude or likelihood is outside 1-5.
        """
        if not (1 <= magnitude <= 5):
            raise ValueError(f"Magnitude must be 1-5, got {magnitude}")
        if not (1 <= likelihood <= 5):
            raise ValueError(f"Likelihood must be 1-5, got {likelihood}")

        mag_w = MAGNITUDE_WEIGHTS[magnitude]
        lik_w = LIKELIHOOD_WEIGHTS[likelihood]
        time_w = TIME_HORIZON_WEIGHTS.get(
            time_horizon.value, Decimal("1.00")
        )

        score = mag_w * lik_w * time_w
        return _round_val(score, 4)

    def calculate_time_horizon_weight(
        self,
        horizon: TimeHorizon,
    ) -> Decimal:
        """Return the time horizon weight for a given horizon.

        Time horizon weights reflect the decreasing certainty and
        actionability of financial effects over longer time periods:
            SHORT_TERM = 1.00 (most certain, most actionable)
            MEDIUM_TERM = 0.80
            LONG_TERM = 0.60 (least certain)

        Args:
            horizon: TimeHorizon enum value.

        Returns:
            Time horizon weight as Decimal.
        """
        return TIME_HORIZON_WEIGHTS.get(horizon.value, Decimal("1.00"))

    # ------------------------------------------------------------------ #
    # Single Assessment                                                    #
    # ------------------------------------------------------------------ #

    def assess_financial_impact(
        self,
        impact: FinancialImpact,
        threshold: Optional[Decimal] = None,
    ) -> FinancialMaterialityResult:
        """Assess financial materiality for a single sustainability matter.

        Executes the full financial materiality assessment workflow:
        1. Calculate magnitude score
        2. Calculate likelihood score
        3. Apply time horizon weight
        4. Calculate combined financial score
        5. Look up affected financial KPIs
        6. Compare against threshold for materiality determination
        7. Generate provenance hash

        Args:
            impact: Financial impact inputs.
            threshold: Materiality threshold (default 0.40).

        Returns:
            FinancialMaterialityResult with complete provenance.
        """
        t0 = time.perf_counter()

        if threshold is None:
            threshold = DEFAULT_FINANCIAL_THRESHOLD

        # Step 1: Magnitude score
        mag_score = _round_val(MAGNITUDE_WEIGHTS[impact.magnitude], 4)

        # Step 2: Likelihood score
        lik_score = _round_val(LIKELIHOOD_WEIGHTS[impact.likelihood], 4)

        # Step 3: Time horizon weight
        time_weight = self.calculate_time_horizon_weight(impact.time_horizon)

        # Step 4: Combined financial score
        financial_score = self.calculate_financial_score(
            impact.magnitude,
            impact.likelihood,
            impact.time_horizon,
        )

        # Step 5: Affected KPIs
        affected_kpis = FINANCIAL_KPI_MAP.get(impact.esrs_topic.value, [])

        # Step 6: Materiality determination
        is_material = financial_score >= threshold

        # Step 7: Build rationale
        rationale = self._build_rationale(
            impact, mag_score, lik_score, time_weight,
            financial_score, is_material, threshold,
        )

        # Affected resources as strings
        affected_resource_strs = [r.value for r in impact.affected_resources]

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = FinancialMaterialityResult(
            matter_id=impact.matter_id,
            matter_name=impact.matter_name,
            esrs_topic=impact.esrs_topic.value,
            financial_score=_round_val(financial_score, 3),
            magnitude_score=_round_val(mag_score, 3),
            likelihood_score=_round_val(lik_score, 3),
            time_horizon_weight=_round_val(time_weight, 3),
            is_material=is_material,
            threshold_used=threshold,
            risk_or_opportunity=impact.risk_or_opportunity.value,
            affected_kpis=affected_kpis,
            affected_resources=affected_resource_strs,
            magnitude_input=impact.magnitude,
            likelihood_input=impact.likelihood,
            rationale=rationale,
            estimated_financial_range_low_eur=impact.estimated_financial_range_low_eur,
            estimated_financial_range_high_eur=impact.estimated_financial_range_high_eur,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Batch Assessment                                                     #
    # ------------------------------------------------------------------ #

    def batch_assess(
        self,
        impacts: List[FinancialImpact],
        threshold: Optional[Decimal] = None,
    ) -> BatchFinancialResult:
        """Assess financial materiality for multiple sustainability matters.

        Processes all financial impacts, ranks results by score,
        and generates summary statistics.

        Args:
            impacts: List of FinancialImpact inputs.
            threshold: Materiality threshold (default 0.40).

        Returns:
            BatchFinancialResult with all individual results and summaries.

        Raises:
            ValueError: If impacts list is empty.
        """
        t0 = time.perf_counter()

        if not impacts:
            raise ValueError("At least one FinancialImpact is required")

        if threshold is None:
            threshold = DEFAULT_FINANCIAL_THRESHOLD

        # Assess each impact
        results: List[FinancialMaterialityResult] = []
        for impact in impacts:
            result = self.assess_financial_impact(impact, threshold)
            results.append(result)

        # Rank by score descending
        results = self.rank_financial(results)

        # Summary statistics
        material_results = [r for r in results if r.is_material]
        material_count = len(material_results)
        not_material_count = len(results) - material_count

        risk_count = sum(
            1 for r in material_results
            if r.risk_or_opportunity in ("risk", "both")
        )
        opportunity_count = sum(
            1 for r in material_results
            if r.risk_or_opportunity in ("opportunity", "both")
        )

        # By topic counts (material only)
        by_topic: Dict[str, int] = {}
        for r in material_results:
            topic = r.esrs_topic
            by_topic[topic] = by_topic.get(topic, 0) + 1

        # By risk/opportunity (material only)
        by_ro: Dict[str, int] = {}
        for r in material_results:
            ro = r.risk_or_opportunity
            by_ro[ro] = by_ro.get(ro, 0) + 1

        # Average scores
        if results:
            avg_fin = _round_val(
                sum(r.financial_score for r in results)
                / _decimal(len(results)),
                3,
            )
            avg_mag = _round_val(
                sum(r.magnitude_score for r in results)
                / _decimal(len(results)),
                3,
            )
        else:
            avg_fin = Decimal("0.000")
            avg_mag = Decimal("0.000")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        batch_result = BatchFinancialResult(
            total_matters=len(results),
            material_count=material_count,
            not_material_count=not_material_count,
            risk_count=risk_count,
            opportunity_count=opportunity_count,
            threshold_used=threshold,
            results=results,
            by_topic=by_topic,
            by_risk_opportunity=by_ro,
            avg_financial_score=avg_fin,
            avg_magnitude=avg_mag,
            processing_time_ms=elapsed_ms,
        )

        batch_result.provenance_hash = _compute_hash(batch_result)
        return batch_result

    # ------------------------------------------------------------------ #
    # Ranking and Filtering                                                #
    # ------------------------------------------------------------------ #

    def rank_financial(
        self,
        results: List[FinancialMaterialityResult],
    ) -> List[FinancialMaterialityResult]:
        """Rank financial materiality results by score descending.

        Assigns a ranking number to each result (1 = highest score).
        Ties receive the same ranking.

        Args:
            results: List of FinancialMaterialityResult to rank.

        Returns:
            Sorted list with ranking assigned.
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.financial_score,
            reverse=True,
        )
        rank = 1
        for i, result in enumerate(sorted_results):
            if i > 0 and (
                result.financial_score
                < sorted_results[i - 1].financial_score
            ):
                rank = i + 1
            result.ranking = rank
        return sorted_results

    def apply_threshold(
        self,
        results: List[FinancialMaterialityResult],
        threshold: Decimal,
    ) -> List[FinancialMaterialityResult]:
        """Filter results to only those meeting the financial threshold.

        Args:
            results: List of FinancialMaterialityResult.
            threshold: Financial materiality threshold (Decimal, 0-1).

        Returns:
            Filtered list containing only financially material results.
        """
        filtered = []
        for r in results:
            if r.financial_score >= threshold:
                r.is_material = True
                r.threshold_used = threshold
                filtered.append(r)
        return filtered

    # ------------------------------------------------------------------ #
    # KPI and Resource Lookups                                             #
    # ------------------------------------------------------------------ #

    def get_affected_kpis(self, topic: ESRSTopic) -> List[str]:
        """Return the list of financial KPIs affected by an ESRS topic.

        Args:
            topic: ESRS topic enum value.

        Returns:
            List of KPI name strings.
        """
        return FINANCIAL_KPI_MAP.get(topic.value, [])

    def get_resource_description(self, resource: AffectedResource) -> str:
        """Return a description of a financial resource.

        Args:
            resource: AffectedResource enum value.

        Returns:
            Description string.
        """
        return RESOURCE_DESCRIPTIONS.get(resource.value, "")

    def get_all_kpi_mappings(self) -> Dict[str, List[str]]:
        """Return the complete financial KPI mapping by ESRS topic.

        Returns:
            Dict mapping topic string to list of KPI strings.
        """
        return dict(FINANCIAL_KPI_MAP)

    # ------------------------------------------------------------------ #
    # Score Interpretation                                                 #
    # ------------------------------------------------------------------ #

    def interpret_score(self, score: Decimal) -> str:
        """Return a human-readable interpretation of a financial score.

        This is a deterministic mapping, not an LLM interpretation.

        Args:
            score: Financial materiality score (0-1 Decimal).

        Returns:
            Interpretation string.
        """
        score_float = float(score)
        if score_float >= 0.80:
            return "Critical: Highest priority financial risk or opportunity"
        elif score_float >= 0.60:
            return "High: Significant financial implications requiring action"
        elif score_float >= 0.40:
            return "Moderate: Material financial effect for disclosure"
        elif score_float >= 0.20:
            return "Low: Below threshold, monitor for changes in conditions"
        else:
            return "Negligible: Immaterial financial effect"

    def get_magnitude_description(self, magnitude: int) -> str:
        """Return description for a magnitude level.

        Args:
            magnitude: Magnitude level (1-5).

        Returns:
            Description string.
        """
        return MAGNITUDE_DESCRIPTIONS.get(magnitude, "")

    def get_likelihood_description(self, likelihood: int) -> str:
        """Return description for a likelihood level.

        Args:
            likelihood: Likelihood level (1-5).

        Returns:
            Description string.
        """
        return LIKELIHOOD_DESCRIPTIONS.get(likelihood, "")

    def get_score_breakdown(
        self,
        result: FinancialMaterialityResult,
    ) -> Dict[str, Any]:
        """Return a structured breakdown of a result's scoring.

        Useful for audit documentation and stakeholder communication.

        Args:
            result: A FinancialMaterialityResult.

        Returns:
            Dict with all scoring components and descriptions.
        """
        return {
            "matter_id": result.matter_id,
            "matter_name": result.matter_name,
            "esrs_topic": result.esrs_topic,
            "risk_or_opportunity": result.risk_or_opportunity,
            "inputs": {
                "magnitude": {
                    "value": result.magnitude_input,
                    "description": MAGNITUDE_DESCRIPTIONS.get(
                        result.magnitude_input, ""
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
                "magnitude_score": str(result.magnitude_score),
                "likelihood_score": str(result.likelihood_score),
                "time_horizon_weight": str(result.time_horizon_weight),
                "financial_score": str(result.financial_score),
            },
            "determination": {
                "is_material": result.is_material,
                "threshold": str(result.threshold_used),
                "ranking": result.ranking,
                "interpretation": self.interpret_score(result.financial_score),
            },
            "affected_kpis": result.affected_kpis,
            "affected_resources": result.affected_resources,
            "estimated_range": {
                "low_eur": result.estimated_financial_range_low_eur,
                "high_eur": result.estimated_financial_range_high_eur,
            },
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Risk-Opportunity Analysis                                            #
    # ------------------------------------------------------------------ #

    def classify_risk_opportunity(
        self,
        results: List[FinancialMaterialityResult],
    ) -> Dict[str, List[FinancialMaterialityResult]]:
        """Classify material results into risks and opportunities.

        Args:
            results: List of FinancialMaterialityResult.

        Returns:
            Dict with keys 'risks', 'opportunities', 'both',
            each mapping to a list of results.
        """
        classified: Dict[str, List[FinancialMaterialityResult]] = {
            "risks": [],
            "opportunities": [],
            "both": [],
        }
        for r in results:
            if not r.is_material:
                continue
            if r.risk_or_opportunity == "risk":
                classified["risks"].append(r)
            elif r.risk_or_opportunity == "opportunity":
                classified["opportunities"].append(r)
            elif r.risk_or_opportunity == "both":
                classified["both"].append(r)
        return classified

    def calculate_aggregate_exposure(
        self,
        results: List[FinancialMaterialityResult],
    ) -> Dict[str, Any]:
        """Calculate aggregate financial exposure from material matters.

        Sums estimated financial ranges across all material results
        to give a total potential financial exposure.

        Args:
            results: List of FinancialMaterialityResult.

        Returns:
            Dict with aggregate exposure metrics.
        """
        material = [r for r in results if r.is_material]
        total_low = Decimal("0.00")
        total_high = Decimal("0.00")
        counted = 0

        for r in material:
            if (
                r.estimated_financial_range_low_eur is not None
                and r.estimated_financial_range_high_eur is not None
            ):
                total_low += _decimal(r.estimated_financial_range_low_eur)
                total_high += _decimal(r.estimated_financial_range_high_eur)
                counted += 1

        return {
            "material_matters_count": len(material),
            "matters_with_estimates": counted,
            "aggregate_low_eur": float(_round_val(total_low, 2)),
            "aggregate_high_eur": float(_round_val(total_high, 2)),
            "aggregate_midpoint_eur": float(
                _round_val((total_low + total_high) / Decimal("2"), 2)
            ),
            "provenance_hash": _compute_hash({
                "material_count": len(material),
                "total_low": str(total_low),
                "total_high": str(total_high),
            }),
        }

    # ------------------------------------------------------------------ #
    # Private: Rationale Builder                                           #
    # ------------------------------------------------------------------ #

    def _build_rationale(
        self,
        impact: FinancialImpact,
        mag_score: Decimal,
        lik_score: Decimal,
        time_weight: Decimal,
        financial_score: Decimal,
        is_material: bool,
        threshold: Decimal,
    ) -> str:
        """Build a deterministic rationale string for the assessment.

        This is a template-based rationale, not generated by an LLM.

        Args:
            impact: The financial impact inputs.
            mag_score: Calculated magnitude score.
            lik_score: Calculated likelihood score.
            time_weight: Time horizon weight.
            financial_score: Final financial materiality score.
            is_material: Financial materiality determination.
            threshold: Threshold used.

        Returns:
            Rationale string.
        """
        ro_label = impact.risk_or_opportunity.value.replace("_", " ")
        resources = ", ".join(r.value for r in impact.affected_resources) or "none specified"

        parts = [
            f"Financial materiality assessment for '{impact.matter_name}' "
            f"({impact.esrs_topic.value}): ",
            f"Type: {ro_label}. ",
            f"Magnitude = {impact.magnitude} (score: {mag_score}). ",
            f"Likelihood = {impact.likelihood} (score: {lik_score}). ",
            f"Time horizon: {impact.time_horizon.value} "
            f"(weight: {time_weight}). ",
            f"Financial score = {mag_score} * {lik_score} * {time_weight} "
            f"= {financial_score}. ",
            f"Affected resources: {resources}. ",
        ]

        if impact.trigger_events:
            triggers = ", ".join(impact.trigger_events)
            parts.append(f"Trigger events: {triggers}. ")

        if (
            impact.estimated_financial_range_low_eur is not None
            and impact.estimated_financial_range_high_eur is not None
        ):
            parts.append(
                f"Estimated financial range: EUR "
                f"{impact.estimated_financial_range_low_eur:,.0f} - "
                f"{impact.estimated_financial_range_high_eur:,.0f}. "
            )

        if is_material:
            parts.append(
                f"FINANCIALLY MATERIAL: Score {financial_score} "
                f">= threshold {threshold}."
            )
        else:
            parts.append(
                f"NOT FINANCIALLY MATERIAL: Score {financial_score} "
                f"< threshold {threshold}."
            )

        if impact.rationale:
            parts.append(f" Assessor rationale: {impact.rationale}")

        return "".join(parts)
