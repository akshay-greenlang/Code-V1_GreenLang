# -*- coding: utf-8 -*-
"""
ScenarioEngine - PACK-046 Intensity Metrics Engine 7
====================================================================

Scenario modelling engine for intensity metrics.  Models numerator
(emissions) and denominator effects independently across efficiency,
growth, structural, methodology, and combined scenarios.  Includes
Monte Carlo simulation for probabilistic target achievement.

Calculation Methodology:
    Scenario Types:
        EFFICIENCY:    Emission reduction with constant denominator.
            I_new = (E * (1 - efficiency_factor)) / D
        GROWTH:        Denominator growth with proportional emission change.
            I_new = (E * (1 + emission_growth)) / (D * (1 + denom_growth))
        STRUCTURAL:    Mix shift between high/low intensity segments.
            I_new = SUM(segment_weight_new_i * I_i)
        METHODOLOGY:   Emission factor or boundary change.
            I_new = (E * methodology_factor) / D
        COMBINED:      Multiple scenarios applied sequentially.

    Monte Carlo Simulation:
        For n iterations (default 10,000):
            1. Sample each input parameter from its distribution
            2. Calculate scenario intensity
            3. Store result
        Output distribution: mean, median, p10, p25, p75, p90, p95, p99
        Target probability = count(I < target) / n

    Input Distributions:
        NORMAL:      mu, sigma -> N(mu, sigma)
        LOGNORMAL:   mu, sigma -> exp(N(mu, sigma))
        TRIANGULAR:  min, mode, max -> Tri(min, mode, max)
        UNIFORM:     min, max -> U(min, max)
        FIXED:       value (deterministic, no sampling)

Regulatory References:
    - TCFD Scenario Analysis guidance
    - ESRS E1-9: Anticipated financial effects from transition risks
    - CDP C3: Business strategy and scenario analysis
    - SBTi target ambition scenarios

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic (except
      Monte Carlo sampling which uses seeded pseudo-random numbers)
    - Monte Carlo uses fixed seed for reproducibility
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  7 of 10
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


def _round6(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


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


class ScenarioType(str, Enum):
    """Type of scenario being modelled."""
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    STRUCTURAL = "structural"
    METHODOLOGY = "methodology"
    COMBINED = "combined"


class DistributionType(str, Enum):
    """Probability distribution for input parameters."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    UNIFORM = "uniform"
    FIXED = "fixed"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ITERATIONS: int = 10000
MAX_ITERATIONS: int = 100000
DEFAULT_SEED: int = 42
MAX_SCENARIOS: int = 50
MAX_SEGMENTS: int = 100


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class DistributionParams(BaseModel):
    """Parameters for a probability distribution.

    Attributes:
        distribution:   Distribution type.
        mean:           Mean (for normal/lognormal).
        std_dev:        Standard deviation (for normal/lognormal).
        min_val:        Minimum (for triangular/uniform).
        mode_val:       Mode (for triangular).
        max_val:        Maximum (for triangular/uniform).
        fixed_value:    Fixed value (for deterministic).
    """
    distribution: DistributionType = Field(
        default=DistributionType.FIXED, description="Distribution type"
    )
    mean: Decimal = Field(default=Decimal("0"), description="Mean")
    std_dev: Decimal = Field(default=Decimal("0"), ge=0, description="Std dev")
    min_val: Decimal = Field(default=Decimal("0"), description="Minimum")
    mode_val: Decimal = Field(default=Decimal("0"), description="Mode")
    max_val: Decimal = Field(default=Decimal("0"), description="Maximum")
    fixed_value: Decimal = Field(default=Decimal("0"), description="Fixed value")

    @field_validator(
        "mean", "std_dev", "min_val", "mode_val", "max_val", "fixed_value",
        mode="before",
    )
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)


class SegmentInput(BaseModel):
    """A segment for structural scenario modelling.

    Attributes:
        segment_id:      Segment identifier.
        segment_name:    Segment name.
        current_weight:  Current weight (0-1).
        new_weight:      New weight (0-1) under scenario.
        intensity:       Segment intensity.
    """
    segment_id: str = Field(..., description="Segment ID")
    segment_name: str = Field(default="", description="Segment name")
    current_weight: Decimal = Field(..., ge=0, le=1, description="Current weight")
    new_weight: Decimal = Field(..., ge=0, le=1, description="New weight")
    intensity: Decimal = Field(..., ge=0, description="Segment intensity")

    @field_validator("current_weight", "new_weight", "intensity", mode="before")
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)


class ScenarioDefinition(BaseModel):
    """Definition of a single scenario.

    Attributes:
        scenario_id:            Scenario identifier.
        scenario_name:          Human-readable name.
        scenario_type:          Scenario type.
        efficiency_factor:      Emission reduction fraction (for EFFICIENCY).
        emission_growth:        Emission growth fraction (for GROWTH).
        denominator_growth:     Denominator growth fraction (for GROWTH).
        methodology_factor:     Methodology adjustment factor (for METHODOLOGY).
        segments:               Segments (for STRUCTURAL).
        sub_scenarios:          Sub-scenarios (for COMBINED).
        efficiency_distribution:  Distribution for efficiency factor (Monte Carlo).
        growth_distribution:      Distribution for emission growth (Monte Carlo).
        denom_distribution:       Distribution for denominator growth (Monte Carlo).
    """
    scenario_id: str = Field(default_factory=_new_uuid, description="Scenario ID")
    scenario_name: str = Field(default="", description="Scenario name")
    scenario_type: ScenarioType = Field(..., description="Scenario type")
    efficiency_factor: Decimal = Field(default=Decimal("0"), ge=0, le=1, description="Efficiency factor")
    emission_growth: Decimal = Field(default=Decimal("0"), description="Emission growth")
    denominator_growth: Decimal = Field(default=Decimal("0"), description="Denominator growth")
    methodology_factor: Decimal = Field(default=Decimal("1"), ge=0, description="Methodology factor")
    segments: List[SegmentInput] = Field(default_factory=list, description="Segments")
    sub_scenarios: List[str] = Field(default_factory=list, description="Sub-scenario IDs")
    efficiency_distribution: Optional[DistributionParams] = Field(
        default=None, description="Efficiency distribution"
    )
    growth_distribution: Optional[DistributionParams] = Field(
        default=None, description="Growth distribution"
    )
    denom_distribution: Optional[DistributionParams] = Field(
        default=None, description="Denominator distribution"
    )

    @field_validator(
        "efficiency_factor", "emission_growth", "denominator_growth", "methodology_factor",
        mode="before",
    )
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)


class ScenarioInput(BaseModel):
    """Input for scenario analysis.

    Attributes:
        organisation_id:    Organisation identifier.
        current_emissions:  Current emissions (tCO2e).
        current_denominator: Current denominator value.
        current_intensity:  Current intensity (auto-computed if omitted).
        target_intensity:   Target intensity for probability calculation.
        scenarios:          List of scenario definitions.
        monte_carlo_iterations: Number of Monte Carlo iterations.
        random_seed:        Random seed for reproducibility.
        output_precision:   Output decimal places.
    """
    organisation_id: str = Field(default="", description="Organisation ID")
    current_emissions: Decimal = Field(..., gt=0, description="Current emissions (tCO2e)")
    current_denominator: Decimal = Field(..., gt=0, description="Current denominator")
    current_intensity: Optional[Decimal] = Field(default=None, description="Current intensity")
    target_intensity: Optional[Decimal] = Field(default=None, ge=0, description="Target intensity")
    scenarios: List[ScenarioDefinition] = Field(..., min_length=1, description="Scenarios")
    monte_carlo_iterations: int = Field(
        default=DEFAULT_ITERATIONS, ge=100, le=MAX_ITERATIONS,
        description="Monte Carlo iterations"
    )
    random_seed: int = Field(default=DEFAULT_SEED, description="Random seed")
    output_precision: int = Field(default=6, ge=0, le=12, description="Precision")

    @field_validator("current_emissions", "current_denominator", mode="before")
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)

    @model_validator(mode="after")
    def compute_intensity(self) -> "ScenarioInput":
        if self.current_intensity is None:
            intensity = self.current_emissions / self.current_denominator
            object.__setattr__(self, "current_intensity", intensity)
        return self


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class MonteCarloDistribution(BaseModel):
    """Distribution of Monte Carlo simulation results.

    Attributes:
        iterations:  Number of iterations.
        mean:        Mean of results.
        median:      Median.
        std_dev:     Standard deviation.
        p10:         10th percentile.
        p25:         25th percentile.
        p75:         75th percentile.
        p90:         90th percentile.
        p95:         95th percentile.
        p99:         99th percentile.
        min_val:     Minimum.
        max_val:     Maximum.
    """
    iterations: int = Field(default=0, description="Iterations")
    mean: Decimal = Field(default=Decimal("0"), description="Mean")
    median: Decimal = Field(default=Decimal("0"), description="Median")
    std_dev: Decimal = Field(default=Decimal("0"), description="Std dev")
    p10: Decimal = Field(default=Decimal("0"), description="P10")
    p25: Decimal = Field(default=Decimal("0"), description="P25")
    p75: Decimal = Field(default=Decimal("0"), description="P75")
    p90: Decimal = Field(default=Decimal("0"), description="P90")
    p95: Decimal = Field(default=Decimal("0"), description="P95")
    p99: Decimal = Field(default=Decimal("0"), description="P99")
    min_val: Decimal = Field(default=Decimal("0"), description="Min")
    max_val: Decimal = Field(default=Decimal("0"), description="Max")


class ScenarioOutcome(BaseModel):
    """Outcome of a single scenario.

    Attributes:
        scenario_id:             Scenario identifier.
        scenario_name:           Scenario name.
        scenario_type:           Scenario type.
        resulting_intensity:     Deterministic resulting intensity.
        intensity_change_pct:    Change from current (%).
        resulting_emissions:     Resulting emissions.
        resulting_denominator:   Resulting denominator.
        monte_carlo:             Monte Carlo distribution (if ran).
        target_probability:      Probability of meeting target (%).
    """
    scenario_id: str = Field(default="", description="Scenario ID")
    scenario_name: str = Field(default="", description="Scenario name")
    scenario_type: ScenarioType = Field(default=ScenarioType.EFFICIENCY, description="Type")
    resulting_intensity: Decimal = Field(default=Decimal("0"), description="Resulting intensity")
    intensity_change_pct: Decimal = Field(default=Decimal("0"), description="Change (%)")
    resulting_emissions: Decimal = Field(default=Decimal("0"), description="Resulting emissions")
    resulting_denominator: Decimal = Field(default=Decimal("0"), description="Resulting denominator")
    monte_carlo: Optional[MonteCarloDistribution] = Field(default=None, description="Monte Carlo")
    target_probability: Optional[Decimal] = Field(default=None, description="Target probability (%)")


class ScenarioResult(BaseModel):
    """Result of scenario analysis.

    Attributes:
        result_id:          Unique result identifier.
        organisation_id:    Organisation identifier.
        current_intensity:  Current intensity.
        target_intensity:   Target intensity (if provided).
        outcomes:           Per-scenario outcomes.
        best_scenario:      ID of best-performing scenario.
        worst_scenario:     ID of worst-performing scenario.
        warnings:           Warnings.
        calculated_at:      Timestamp.
        processing_time_ms: Processing time (ms).
        provenance_hash:    SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    current_intensity: Decimal = Field(default=Decimal("0"), description="Current intensity")
    target_intensity: Optional[Decimal] = Field(default=None, description="Target")
    outcomes: List[ScenarioOutcome] = Field(default_factory=list, description="Outcomes")
    best_scenario: str = Field(default="", description="Best scenario ID")
    worst_scenario: str = Field(default="", description="Worst scenario ID")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ScenarioEngine:
    """Scenario modelling engine for intensity metrics.

    Models numerator and denominator effects independently, with
    Monte Carlo simulation for probabilistic outcomes.

    Guarantees:
        - Deterministic: Seeded Monte Carlo ensures reproducibility.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every scenario documented with methodology.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("ScenarioEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: ScenarioInput) -> ScenarioResult:
        """Run scenario analysis.

        Args:
            input_data: Scenario analysis input.

        Returns:
            ScenarioResult with outcomes and Monte Carlo distributions.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        if len(input_data.scenarios) > MAX_SCENARIOS:
            raise ValueError(f"Maximum {MAX_SCENARIOS} scenarios allowed.")

        current_e = input_data.current_emissions
        current_d = input_data.current_denominator
        current_i = input_data.current_intensity or (current_e / current_d)
        target = input_data.target_intensity

        outcomes: List[ScenarioOutcome] = []
        scenario_map = {s.scenario_id: s for s in input_data.scenarios}

        for scenario in input_data.scenarios:
            # Deterministic calculation
            det_e, det_d = self._apply_scenario_deterministic(
                current_e, current_d, scenario, scenario_map
            )
            det_i = _safe_divide(det_e, det_d)
            det_i = det_i.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            change_pct = Decimal("0")
            if current_i > Decimal("0"):
                change_pct = (
                    (det_i - current_i) / current_i * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Monte Carlo simulation
            mc: Optional[MonteCarloDistribution] = None
            target_prob: Optional[Decimal] = None

            if self._has_distributions(scenario):
                mc_results = self._monte_carlo(
                    current_e, current_d, scenario, scenario_map,
                    input_data.monte_carlo_iterations, input_data.random_seed,
                )
                mc = self._compute_mc_distribution(mc_results, prec_str)

                if target is not None and mc_results:
                    below_target = sum(1 for v in mc_results if v <= target)
                    target_prob = (
                        Decimal(str(below_target)) / Decimal(str(len(mc_results))) * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            outcomes.append(ScenarioOutcome(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.scenario_name,
                scenario_type=scenario.scenario_type,
                resulting_intensity=det_i,
                intensity_change_pct=change_pct,
                resulting_emissions=det_e.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                resulting_denominator=det_d.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                monte_carlo=mc,
                target_probability=target_prob,
            ))

        # Best / worst scenario
        best_id = ""
        worst_id = ""
        if outcomes:
            sorted_outcomes = sorted(outcomes, key=lambda o: o.resulting_intensity)
            best_id = sorted_outcomes[0].scenario_id
            worst_id = sorted_outcomes[-1].scenario_id

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ScenarioResult(
            organisation_id=input_data.organisation_id,
            current_intensity=current_i.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            target_intensity=target,
            outcomes=outcomes,
            best_scenario=best_id,
            worst_scenario=worst_id,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Deterministic Scenario
    # ------------------------------------------------------------------

    def _apply_scenario_deterministic(
        self,
        emissions: Decimal,
        denominator: Decimal,
        scenario: ScenarioDefinition,
        scenario_map: Dict[str, ScenarioDefinition],
    ) -> Tuple[Decimal, Decimal]:
        """Apply scenario to emissions and denominator (deterministic).

        Returns (new_emissions, new_denominator).
        """
        if scenario.scenario_type == ScenarioType.EFFICIENCY:
            new_e = emissions * (Decimal("1") - scenario.efficiency_factor)
            return new_e, denominator

        elif scenario.scenario_type == ScenarioType.GROWTH:
            new_e = emissions * (Decimal("1") + scenario.emission_growth)
            new_d = denominator * (Decimal("1") + scenario.denominator_growth)
            return new_e, new_d

        elif scenario.scenario_type == ScenarioType.STRUCTURAL:
            # Weighted-sum intensity from segments
            if not scenario.segments:
                return emissions, denominator
            new_intensity = sum(
                s.new_weight * s.intensity for s in scenario.segments
            )
            new_e = new_intensity * denominator
            return new_e, denominator

        elif scenario.scenario_type == ScenarioType.METHODOLOGY:
            new_e = emissions * scenario.methodology_factor
            return new_e, denominator

        elif scenario.scenario_type == ScenarioType.COMBINED:
            e, d = emissions, denominator
            for sub_id in scenario.sub_scenarios:
                sub = scenario_map.get(sub_id)
                if sub is not None:
                    e, d = self._apply_scenario_deterministic(e, d, sub, scenario_map)
            return e, d

        return emissions, denominator

    # ------------------------------------------------------------------
    # Internal: Monte Carlo
    # ------------------------------------------------------------------

    def _has_distributions(self, scenario: ScenarioDefinition) -> bool:
        """Check if scenario has any stochastic distributions."""
        for dist in [
            scenario.efficiency_distribution,
            scenario.growth_distribution,
            scenario.denom_distribution,
        ]:
            if dist is not None and dist.distribution != DistributionType.FIXED:
                return True
        return False

    def _monte_carlo(
        self,
        emissions: Decimal,
        denominator: Decimal,
        scenario: ScenarioDefinition,
        scenario_map: Dict[str, ScenarioDefinition],
        iterations: int,
        seed: int,
    ) -> List[Decimal]:
        """Run Monte Carlo simulation for a scenario."""
        rng = random.Random(seed)
        results: List[Decimal] = []

        for _ in range(iterations):
            # Sample parameters
            if scenario.scenario_type == ScenarioType.EFFICIENCY:
                eff = self._sample(scenario.efficiency_distribution, scenario.efficiency_factor, rng)
                eff = max(Decimal("0"), min(Decimal("1"), eff))
                new_e = emissions * (Decimal("1") - eff)
                new_d = denominator
            elif scenario.scenario_type == ScenarioType.GROWTH:
                eg = self._sample(scenario.growth_distribution, scenario.emission_growth, rng)
                dg = self._sample(scenario.denom_distribution, scenario.denominator_growth, rng)
                new_e = emissions * (Decimal("1") + eg)
                new_d = denominator * (Decimal("1") + dg)
            elif scenario.scenario_type == ScenarioType.METHODOLOGY:
                mf = self._sample(scenario.efficiency_distribution, scenario.methodology_factor, rng)
                mf = max(Decimal("0"), mf)
                new_e = emissions * mf
                new_d = denominator
            else:
                new_e = emissions
                new_d = denominator

            if new_d > Decimal("0"):
                intensity = new_e / new_d
                results.append(intensity)

        return results

    def _sample(
        self,
        dist_params: Optional[DistributionParams],
        default_value: Decimal,
        rng: random.Random,
    ) -> Decimal:
        """Sample from a distribution."""
        if dist_params is None or dist_params.distribution == DistributionType.FIXED:
            if dist_params is not None and dist_params.fixed_value != Decimal("0"):
                return dist_params.fixed_value
            return default_value

        if dist_params.distribution == DistributionType.NORMAL:
            val = rng.gauss(float(dist_params.mean), float(dist_params.std_dev))
            return _decimal(val)

        elif dist_params.distribution == DistributionType.LOGNORMAL:
            val = rng.lognormvariate(float(dist_params.mean), float(dist_params.std_dev))
            return _decimal(val)

        elif dist_params.distribution == DistributionType.TRIANGULAR:
            low = float(dist_params.min_val)
            high = float(dist_params.max_val)
            mode = float(dist_params.mode_val)
            if low >= high:
                return _decimal(mode)
            mode = max(low, min(high, mode))
            val = rng.triangular(low, high, mode)
            return _decimal(val)

        elif dist_params.distribution == DistributionType.UNIFORM:
            low = float(dist_params.min_val)
            high = float(dist_params.max_val)
            if low >= high:
                return _decimal(low)
            val = rng.uniform(low, high)
            return _decimal(val)

        return default_value

    def _compute_mc_distribution(
        self,
        results: List[Decimal],
        prec_str: str,
    ) -> MonteCarloDistribution:
        """Compute distribution statistics from Monte Carlo results."""
        if not results:
            return MonteCarloDistribution()

        sorted_vals = sorted(results)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        mean = (total / Decimal(str(n))).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        median = _median_decimal(sorted_vals).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Std dev
        mean_f = float(mean)
        variance = sum((float(v) - mean_f) ** 2 for v in sorted_vals) / n
        std = _decimal(variance ** 0.5).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return MonteCarloDistribution(
            iterations=n,
            mean=mean,
            median=median,
            std_dev=std,
            p10=_percentile_decimal(sorted_vals, Decimal("10")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p25=_percentile_decimal(sorted_vals, Decimal("25")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p75=_percentile_decimal(sorted_vals, Decimal("75")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p90=_percentile_decimal(sorted_vals, Decimal("90")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p95=_percentile_decimal(sorted_vals, Decimal("95")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            p99=_percentile_decimal(sorted_vals, Decimal("99")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            ),
            min_val=sorted_vals[0].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            max_val=sorted_vals[-1].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
        )

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ScenarioType",
    "DistributionType",
    # Input Models
    "DistributionParams",
    "SegmentInput",
    "ScenarioDefinition",
    "ScenarioInput",
    # Output Models
    "MonteCarloDistribution",
    "ScenarioOutcome",
    "ScenarioResult",
    # Engine
    "ScenarioEngine",
    # Constants
    "DEFAULT_ITERATIONS",
    "DEFAULT_SEED",
]
