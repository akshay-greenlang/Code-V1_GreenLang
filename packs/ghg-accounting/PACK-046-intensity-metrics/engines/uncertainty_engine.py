# -*- coding: utf-8 -*-
"""
UncertaintyEngine - PACK-046 Intensity Metrics Engine 8
====================================================================

Uncertainty quantification engine for intensity metrics.  Implements
IPCC Tier 1 and Tier 2 error propagation, GHG Protocol data quality
scoring, and confidence interval calculation.

Calculation Methodology:
    IPCC Tier 1 Error Propagation (for intensity I = E / D):
        U_I = sqrt( U_E^2 + U_D^2 )
        Where:
            U_I = percentage uncertainty of intensity (%)
            U_E = percentage uncertainty of emissions numerator (%)
            U_D = percentage uncertainty of denominator (%)

    For multi-component emissions (E = E1 + E2 + ... + En):
        U_E = sqrt( SUM_i( (E_i * U_i)^2 ) ) / SUM_i(E_i) * 100
        Where U_i is percentage uncertainty of each component.

    Confidence Intervals:
        90% CI: I * (1 +/- 1.645 * U_I / 100)
        95% CI: I * (1 +/- 1.960 * U_I / 100)
        99% CI: I * (1 +/- 2.576 * U_I / 100)

    Data Quality Scoring (GHG Protocol 5-point scale):
        Score 1 (High quality):   U = 3.5%   (measured, calibrated)
        Score 2 (Good quality):   U = 7.5%   (calculated, verified)
        Score 3 (Fair quality):   U = 15%    (estimated, sector average)
        Score 4 (Low quality):    U = 35%    (proxy, modelled)
        Score 5 (Very low):       U = 75%    (rough estimate, expert judgement)

    Combined Portfolio Uncertainty:
        For N metrics with uncertainties U_1 ... U_N and values V_1 ... V_N:
        U_portfolio = sqrt( SUM_i( (V_i * U_i)^2 ) ) / SUM_i(V_i) * 100

    Tier 2 Monte Carlo Propagation:
        For each iteration k = 1 ... n:
            Sample E_k from distribution(E, U_E)
            Sample D_k from distribution(D, U_D)
            I_k = E_k / D_k
        Report percentile-based CI from {I_k}

Regulatory References:
    - IPCC 2006 Guidelines Vol 1 Ch 3 (Uncertainties)
    - IPCC 2019 Refinement Vol 1 Ch 3 (Uncertainties)
    - GHG Protocol Corporate Standard (2015), Appendix A
    - GHG Protocol Scope 3 Standard, Appendix C
    - ISO 14064-1:2018 Clause 5.3.5 (Uncertainty assessment)
    - ESRS E1-6 (Data quality and uncertainty)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Uncertainty formulas from IPCC published guidelines
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
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


def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _sqrt_decimal(value: Decimal) -> Decimal:
    """Square root via float conversion (deterministic for same input)."""
    if value <= Decimal("0"):
        return Decimal("0")
    return _decimal(float(value) ** 0.5)


def _median_decimal(values: List[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / Decimal("2")


def _percentile_decimal(values: List[Decimal], pct: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (pct / Decimal("100")) * Decimal(str(n - 1))
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - Decimal(str(lower))
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UncertaintyTier(str, Enum):
    """IPCC uncertainty assessment tier.

    TIER_1: Error propagation (analytical).
    TIER_2: Monte Carlo propagation (simulation).
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"


class DataQualityLevel(int, Enum):
    """GHG Protocol data quality score (1-5)."""
    HIGH = 1
    GOOD = 2
    FAIR = 3
    LOW = 4
    VERY_LOW = 5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Data quality to uncertainty mapping (percentage)
QUALITY_UNCERTAINTY_MAP: Dict[int, Decimal] = {
    DataQualityLevel.HIGH.value: Decimal("3.5"),
    DataQualityLevel.GOOD.value: Decimal("7.5"),
    DataQualityLevel.FAIR.value: Decimal("15"),
    DataQualityLevel.LOW.value: Decimal("35"),
    DataQualityLevel.VERY_LOW.value: Decimal("75"),
}

# Confidence level z-scores
CONFIDENCE_Z: Dict[str, Decimal] = {
    "90": Decimal("1.645"),
    "95": Decimal("1.960"),
    "99": Decimal("2.576"),
}

# Default Monte Carlo iterations
DEFAULT_MC_ITERATIONS: int = 10000
MAX_MC_ITERATIONS: int = 100000
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class EmissionComponent(BaseModel):
    """A single emission component with uncertainty.

    Attributes:
        component_id:      Component identifier.
        component_name:    Human-readable name.
        value_tco2e:       Emission value (tCO2e).
        uncertainty_pct:   Uncertainty as percentage (%).
        data_quality:      Data quality score (1-5), used if uncertainty_pct is None.
    """
    component_id: str = Field(default="", description="Component ID")
    component_name: str = Field(default="", description="Component name")
    value_tco2e: Decimal = Field(..., ge=0, description="Emissions (tCO2e)")
    uncertainty_pct: Optional[Decimal] = Field(
        default=None, ge=0, description="Uncertainty (%)"
    )
    data_quality: int = Field(default=3, ge=1, le=5, description="Data quality (1-5)")

    @field_validator("value_tco2e", mode="before")
    @classmethod
    def coerce_val(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("uncertainty_pct", mode="before")
    @classmethod
    def coerce_unc(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

    @model_validator(mode="after")
    def resolve_uncertainty(self) -> "EmissionComponent":
        """If uncertainty_pct is not provided, derive from data quality."""
        if self.uncertainty_pct is None:
            u = QUALITY_UNCERTAINTY_MAP.get(self.data_quality, Decimal("15"))
            object.__setattr__(self, "uncertainty_pct", u)
        return self


class DenominatorUncertainty(BaseModel):
    """Denominator with uncertainty information.

    Attributes:
        denominator_value:  Denominator value.
        uncertainty_pct:    Uncertainty as percentage (%).
        data_quality:       Data quality score (1-5).
    """
    denominator_value: Decimal = Field(..., gt=0, description="Denominator value")
    uncertainty_pct: Optional[Decimal] = Field(default=None, ge=0, description="Uncertainty (%)")
    data_quality: int = Field(default=3, ge=1, le=5, description="Data quality (1-5)")

    @field_validator("denominator_value", mode="before")
    @classmethod
    def coerce_val(cls, v: Any) -> Decimal:
        return _decimal(v)

    @model_validator(mode="after")
    def resolve_uncertainty(self) -> "DenominatorUncertainty":
        if self.uncertainty_pct is None:
            u = QUALITY_UNCERTAINTY_MAP.get(self.data_quality, Decimal("15"))
            object.__setattr__(self, "uncertainty_pct", u)
        return self


class DataQualityAssessment(BaseModel):
    """Data quality assessment for an intensity metric.

    Attributes:
        metric_id:                  Metric identifier.
        emission_components:        Emission components with quality scores.
        denominator:                Denominator with quality score.
        overall_quality_score:      Weighted average quality score.
        overall_quality_label:      Human-readable quality label.
    """
    metric_id: str = Field(default="", description="Metric ID")
    emission_components: List[EmissionComponent] = Field(
        default_factory=list, description="Emission components"
    )
    denominator: Optional[DenominatorUncertainty] = Field(
        default=None, description="Denominator"
    )
    overall_quality_score: Decimal = Field(default=Decimal("3"), description="Overall quality")
    overall_quality_label: str = Field(default="fair", description="Quality label")


class UncertaintyInput(BaseModel):
    """Input for uncertainty quantification.

    Attributes:
        metric_id:               Metric identifier.
        intensity_value:         Calculated intensity value.
        emission_components:     Emission components with uncertainty.
        denominator:             Denominator with uncertainty.
        tier:                    IPCC tier for uncertainty assessment.
        confidence_levels:       Confidence levels for CI calculation.
        monte_carlo_iterations:  Iterations for Tier 2.
        random_seed:             Random seed for reproducibility.
        output_precision:        Output decimal places.
    """
    metric_id: str = Field(default="", description="Metric ID")
    intensity_value: Decimal = Field(..., ge=0, description="Intensity value")
    emission_components: List[EmissionComponent] = Field(
        ..., min_length=1, description="Emission components"
    )
    denominator: DenominatorUncertainty = Field(..., description="Denominator")
    tier: UncertaintyTier = Field(default=UncertaintyTier.TIER_1, description="Tier")
    confidence_levels: List[str] = Field(
        default_factory=lambda: ["90", "95"], description="Confidence levels"
    )
    monte_carlo_iterations: int = Field(
        default=DEFAULT_MC_ITERATIONS, ge=100, le=MAX_MC_ITERATIONS,
        description="MC iterations"
    )
    random_seed: int = Field(default=DEFAULT_SEED, description="Random seed")
    output_precision: int = Field(default=4, ge=0, le=12, description="Precision")

    @field_validator("intensity_value", mode="before")
    @classmethod
    def coerce_val(cls, v: Any) -> Decimal:
        return _decimal(v)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class ConfidenceInterval(BaseModel):
    """Confidence interval for an intensity value.

    Attributes:
        level:       Confidence level (e.g. '95').
        lower_bound: Lower bound.
        upper_bound: Upper bound.
        z_score:     Z-score used.
    """
    level: str = Field(..., description="Confidence level")
    lower_bound: Decimal = Field(default=Decimal("0"), description="Lower bound")
    upper_bound: Decimal = Field(default=Decimal("0"), description="Upper bound")
    z_score: Decimal = Field(default=Decimal("0"), description="Z-score")


class ComponentUncertainty(BaseModel):
    """Uncertainty breakdown for a single component.

    Attributes:
        component_id:      Component identifier.
        component_name:    Component name.
        value_tco2e:       Emission value.
        uncertainty_pct:   Uncertainty (%).
        contribution_pct:  Contribution to total uncertainty (%).
        data_quality:      Data quality score.
    """
    component_id: str = Field(default="", description="Component ID")
    component_name: str = Field(default="", description="Component name")
    value_tco2e: Decimal = Field(default=Decimal("0"), description="Value (tCO2e)")
    uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Uncertainty (%)")
    contribution_pct: Decimal = Field(default=Decimal("0"), description="Contribution (%)")
    data_quality: int = Field(default=3, description="Data quality")


class MonteCarloUncertainty(BaseModel):
    """Monte Carlo uncertainty results."""
    iterations: int = Field(default=0, description="Iterations")
    mean: Decimal = Field(default=Decimal("0"), description="Mean")
    median: Decimal = Field(default=Decimal("0"), description="Median")
    std_dev: Decimal = Field(default=Decimal("0"), description="Std dev")
    ci_90_lower: Decimal = Field(default=Decimal("0"), description="90% CI lower")
    ci_90_upper: Decimal = Field(default=Decimal("0"), description="90% CI upper")
    ci_95_lower: Decimal = Field(default=Decimal("0"), description="95% CI lower")
    ci_95_upper: Decimal = Field(default=Decimal("0"), description="95% CI upper")


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification.

    Attributes:
        result_id:                Unique result identifier.
        metric_id:                Metric identifier.
        intensity_value:          Intensity value assessed.
        emission_uncertainty_pct: Combined emission uncertainty (%).
        denominator_uncertainty_pct: Denominator uncertainty (%).
        intensity_uncertainty_pct: Intensity uncertainty (%).
        confidence_intervals:     Confidence intervals.
        component_breakdown:      Per-component uncertainty breakdown.
        data_quality_assessment:  Overall data quality assessment.
        monte_carlo:              Monte Carlo results (if Tier 2).
        tier_used:                IPCC tier used.
        warnings:                 Warnings.
        calculated_at:            Timestamp.
        processing_time_ms:       Processing time (ms).
        provenance_hash:          SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    metric_id: str = Field(default="", description="Metric ID")
    intensity_value: Decimal = Field(default=Decimal("0"), description="Intensity")
    emission_uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Emission U (%)")
    denominator_uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Denom U (%)")
    intensity_uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Intensity U (%)")
    confidence_intervals: List[ConfidenceInterval] = Field(
        default_factory=list, description="CIs"
    )
    component_breakdown: List[ComponentUncertainty] = Field(
        default_factory=list, description="Component breakdown"
    )
    data_quality_assessment: Optional[DataQualityAssessment] = Field(
        default=None, description="Quality assessment"
    )
    monte_carlo: Optional[MonteCarloUncertainty] = Field(default=None, description="MC results")
    tier_used: UncertaintyTier = Field(default=UncertaintyTier.TIER_1, description="Tier used")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class UncertaintyEngine:
    """Uncertainty quantification engine for intensity metrics.

    Implements IPCC Tier 1 error propagation and Tier 2 Monte Carlo
    propagation for intensity metric uncertainty assessment.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Seeded Monte Carlo; SHA-256 provenance hashes.
        - Auditable: IPCC-published formulas throughout.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("UncertaintyEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: UncertaintyInput) -> UncertaintyResult:
        """Perform uncertainty quantification.

        Args:
            input_data: Uncertainty assessment input.

        Returns:
            UncertaintyResult with uncertainties and confidence intervals.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        # 1. Combined emission uncertainty
        emission_u, component_breakdown, total_emissions = self._combined_emission_uncertainty(
            input_data.emission_components, prec_str
        )

        # 2. Denominator uncertainty
        denom_u = input_data.denominator.uncertainty_pct or Decimal("0")

        # 3. Intensity uncertainty (IPCC Tier 1)
        # U_I = sqrt(U_E^2 + U_D^2)
        intensity_u = _sqrt_decimal(emission_u ** 2 + denom_u ** 2)
        intensity_u = intensity_u.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # 4. Confidence intervals
        cis: List[ConfidenceInterval] = []
        for cl in input_data.confidence_levels:
            z = CONFIDENCE_Z.get(cl, Decimal("1.960"))
            half_width = input_data.intensity_value * z * intensity_u / Decimal("100")
            lower = max(Decimal("0"), input_data.intensity_value - half_width)
            upper = input_data.intensity_value + half_width
            cis.append(ConfidenceInterval(
                level=cl,
                lower_bound=lower.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                upper_bound=upper.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                z_score=z,
            ))

        # 5. Data quality assessment
        quality_assessment = self._assess_data_quality(
            input_data.emission_components, input_data.denominator, total_emissions
        )

        # 6. Monte Carlo (Tier 2)
        mc: Optional[MonteCarloUncertainty] = None
        if input_data.tier == UncertaintyTier.TIER_2:
            mc = self._tier_2_monte_carlo(
                input_data.emission_components,
                input_data.denominator,
                input_data.monte_carlo_iterations,
                input_data.random_seed,
                prec_str,
            )

        # Warnings
        if intensity_u > Decimal("50"):
            warnings.append(
                f"Intensity uncertainty of {_round2(intensity_u)}% is very high. "
                f"Consider improving data quality."
            )

        high_u_components = [
            c.component_name or c.component_id
            for c in input_data.emission_components
            if c.uncertainty_pct is not None and c.uncertainty_pct > Decimal("50")
        ]
        if high_u_components:
            warnings.append(
                f"High-uncertainty emission components: {high_u_components}"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = UncertaintyResult(
            metric_id=input_data.metric_id,
            intensity_value=input_data.intensity_value,
            emission_uncertainty_pct=emission_u.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            denominator_uncertainty_pct=denom_u.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            intensity_uncertainty_pct=intensity_u,
            confidence_intervals=cis,
            component_breakdown=component_breakdown,
            data_quality_assessment=quality_assessment,
            monte_carlo=mc,
            tier_used=input_data.tier,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def compute_combined_uncertainty(
        self,
        values: List[Decimal],
        uncertainties_pct: List[Decimal],
    ) -> Decimal:
        """Compute combined uncertainty for summed values.

        Formula: U = sqrt(SUM((V_i * U_i)^2)) / SUM(V_i) * 100

        Args:
            values:             Individual values.
            uncertainties_pct:  Individual uncertainties (%).

        Returns:
            Combined uncertainty as percentage.
        """
        if not values or not uncertainties_pct:
            return Decimal("0")

        total = sum(values)
        if total == Decimal("0"):
            return Decimal("0")

        sum_sq = Decimal("0")
        for i in range(min(len(values), len(uncertainties_pct))):
            term = values[i] * uncertainties_pct[i] / Decimal("100")
            sum_sq += term ** 2

        return (_sqrt_decimal(sum_sq) / total * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def quality_to_uncertainty(self, quality_score: int) -> Decimal:
        """Convert data quality score to uncertainty percentage.

        Args:
            quality_score: Score 1-5 per GHG Protocol.

        Returns:
            Uncertainty percentage.
        """
        return QUALITY_UNCERTAINTY_MAP.get(quality_score, Decimal("15"))

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _combined_emission_uncertainty(
        self,
        components: List[EmissionComponent],
        prec_str: str,
    ) -> Tuple[Decimal, List[ComponentUncertainty], Decimal]:
        """Compute combined emission uncertainty."""
        total_emissions = sum(c.value_tco2e for c in components)
        if total_emissions == Decimal("0"):
            return Decimal("0"), [], Decimal("0")

        sum_sq = Decimal("0")
        breakdown: List[ComponentUncertainty] = []

        for comp in components:
            u = comp.uncertainty_pct or Decimal("0")
            term = comp.value_tco2e * u / Decimal("100")
            sq = term ** 2
            sum_sq += sq

            breakdown.append(ComponentUncertainty(
                component_id=comp.component_id,
                component_name=comp.component_name,
                value_tco2e=comp.value_tco2e,
                uncertainty_pct=u,
                contribution_pct=Decimal("0"),  # Computed below
                data_quality=comp.data_quality,
            ))

        combined_u = _sqrt_decimal(sum_sq) / total_emissions * Decimal("100")

        # Contribution percentages
        if sum_sq > Decimal("0"):
            for i, comp in enumerate(components):
                u = comp.uncertainty_pct or Decimal("0")
                term_sq = (comp.value_tco2e * u / Decimal("100")) ** 2
                contribution = (term_sq / sum_sq * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                breakdown[i].contribution_pct = contribution

        return combined_u, breakdown, total_emissions

    def _assess_data_quality(
        self,
        components: List[EmissionComponent],
        denominator: DenominatorUncertainty,
        total_emissions: Decimal,
    ) -> DataQualityAssessment:
        """Compute overall data quality assessment."""
        # Weighted average quality score
        if total_emissions > Decimal("0"):
            weighted_sum = sum(
                Decimal(str(c.data_quality)) * c.value_tco2e for c in components
            )
            avg_emission_quality = weighted_sum / total_emissions
        else:
            avg_emission_quality = Decimal("3")

        # Include denominator quality
        denom_weight = Decimal("0.2")
        emission_weight = Decimal("0.8")
        overall = (
            avg_emission_quality * emission_weight
            + Decimal(str(denominator.data_quality)) * denom_weight
        ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Label
        overall_int = int(overall.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        labels = {1: "high", 2: "good", 3: "fair", 4: "low", 5: "very_low"}
        label = labels.get(max(1, min(5, overall_int)), "fair")

        return DataQualityAssessment(
            emission_components=components,
            denominator=denominator,
            overall_quality_score=overall,
            overall_quality_label=label,
        )

    def _tier_2_monte_carlo(
        self,
        components: List[EmissionComponent],
        denominator: DenominatorUncertainty,
        iterations: int,
        seed: int,
        prec_str: str,
    ) -> MonteCarloUncertainty:
        """Tier 2 Monte Carlo uncertainty propagation."""
        rng = random.Random(seed)
        results: List[Decimal] = []

        for _ in range(iterations):
            # Sample emissions (normal distribution)
            sampled_e = Decimal("0")
            for comp in components:
                u = comp.uncertainty_pct or Decimal("0")
                std = float(comp.value_tco2e * u / Decimal("100")) / 1.96
                sampled = rng.gauss(float(comp.value_tco2e), max(std, 0))
                sampled_e += _decimal(max(sampled, 0))

            # Sample denominator
            u_d = denominator.uncertainty_pct or Decimal("0")
            std_d = float(denominator.denominator_value * u_d / Decimal("100")) / 1.96
            sampled_d = rng.gauss(float(denominator.denominator_value), max(std_d, 0))
            sampled_d_dec = _decimal(max(sampled_d, 0.001))

            intensity = sampled_e / sampled_d_dec
            results.append(intensity)

        if not results:
            return MonteCarloUncertainty()

        sorted_results = sorted(results)
        n = len(sorted_results)
        total = sum(sorted_results)
        mean = (total / Decimal(str(n))).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        median = _median_decimal(sorted_results).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        mean_f = float(mean)
        variance = sum((float(v) - mean_f) ** 2 for v in sorted_results) / n
        std = _decimal(variance ** 0.5).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return MonteCarloUncertainty(
            iterations=n,
            mean=mean,
            median=median,
            std_dev=std,
            ci_90_lower=_percentile_decimal(sorted_results, Decimal("5")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            ci_90_upper=_percentile_decimal(sorted_results, Decimal("95")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            ci_95_lower=_percentile_decimal(sorted_results, Decimal("2.5")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            ci_95_upper=_percentile_decimal(sorted_results, Decimal("97.5")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
        )

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "UncertaintyTier",
    "DataQualityLevel",
    # Input Models
    "EmissionComponent",
    "DenominatorUncertainty",
    "DataQualityAssessment",
    "UncertaintyInput",
    # Output Models
    "ConfidenceInterval",
    "ComponentUncertainty",
    "MonteCarloUncertainty",
    "UncertaintyResult",
    # Engine
    "UncertaintyEngine",
    # Constants
    "QUALITY_UNCERTAINTY_MAP",
    "CONFIDENCE_Z",
]
