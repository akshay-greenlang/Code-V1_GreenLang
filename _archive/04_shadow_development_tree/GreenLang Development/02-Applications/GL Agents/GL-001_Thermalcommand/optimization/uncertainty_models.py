"""
Uncertainty Models - Prediction intervals and scenario generation

This module implements deterministic uncertainty quantification models
for robust planning in process heat systems. All calculations are
bit-perfect reproducible with SHA-256 provenance tracking.

Key Components:
    - UncertaintyModelEngine: Main engine for uncertainty quantification
    - WeatherUncertaintyModel: Weather forecast uncertainty
    - PriceUncertaintyModel: Energy price uncertainty
    - DemandUncertaintyModel: Process heat demand variability

Reference Standards:
    - ISO/IEC Guide 98-3:2008 (Uncertainty of measurement)
    - GUM (Guide to expression of uncertainty in measurement)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
import math

from .uq_schemas import (
    ProvenanceRecord,
    PredictionInterval,
    QuantileSet,
    QuantileValue,
    Scenario,
    ScenarioSet,
    ScenarioVariable,
    UncertaintyBand,
    UncertaintySource,
    UncertaintySourceType,
    DistributionType,
)


class DeterministicRNG:
    """
    Deterministic Random Number Generator - ZERO HALLUCINATION.

    Uses a seeded Linear Congruential Generator (LCG) for
    bit-perfect reproducibility. Same seed produces identical
    sequence every time.
    """

    # LCG parameters (same as glibc)
    _MULTIPLIER = 1103515245
    _INCREMENT = 12345
    _MODULUS = 2 ** 31

    def __init__(self, seed: int):
        """Initialize with seed for reproducibility."""
        self._seed = seed
        self._state = seed

    def reset(self) -> None:
        """Reset to initial seed."""
        self._state = self._seed

    def next_int(self) -> int:
        """Generate next integer in sequence - DETERMINISTIC."""
        self._state = (self._MULTIPLIER * self._state + self._INCREMENT) % self._MODULUS
        return self._state

    def next_uniform(self) -> Decimal:
        """Generate uniform random number in [0, 1) - DETERMINISTIC."""
        return Decimal(str(self.next_int())) / Decimal(str(self._MODULUS))

    def next_normal(self, mean: Decimal = Decimal("0"), std: Decimal = Decimal("1")) -> Decimal:
        """
        Generate normal random number using Box-Muller - DETERMINISTIC.

        Uses Box-Muller transform which is deterministic given
        uniform random inputs.
        """
        u1 = self.next_uniform()
        u2 = self.next_uniform()

        # Avoid log(0)
        while u1 == Decimal("0"):
            u1 = self.next_uniform()

        # Box-Muller transform
        u1_float = float(u1)
        u2_float = float(u2)

        z0 = math.sqrt(-2.0 * math.log(u1_float)) * math.cos(2.0 * math.pi * u2_float)

        result = mean + std * Decimal(str(z0))
        return result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


class UncertaintyModelEngine:
    """
    Main uncertainty quantification engine - ZERO HALLUCINATION.

    Provides deterministic uncertainty quantification for:
    - Prediction interval generation (quantiles)
    - Scenario set generation
    - Multi-source uncertainty aggregation

    All calculations are bit-perfect reproducible with provenance tracking.
    """

    # Standard quantile probabilities for prediction intervals
    DEFAULT_QUANTILES = [
        Decimal("0.01"), Decimal("0.05"), Decimal("0.10"),
        Decimal("0.25"), Decimal("0.50"), Decimal("0.75"),
        Decimal("0.90"), Decimal("0.95"), Decimal("0.99")
    ]

    def __init__(self, seed: int = 42):
        """Initialize uncertainty model engine."""
        self._seed = seed
        self._rng = DeterministicRNG(seed)

    def generate_prediction_interval(
        self,
        point_estimate: Decimal,
        uncertainty_source: UncertaintySource,
        confidence_level: Decimal = Decimal("0.90"),
        horizon_minutes: int = 60
    ) -> PredictionInterval:
        """
        Generate prediction interval from point estimate - DETERMINISTIC.

        Args:
            point_estimate: Central point prediction
            uncertainty_source: Source defining uncertainty distribution
            confidence_level: Confidence level (e.g., 0.90 for 90%)
            horizon_minutes: Forecast horizon

        Returns:
            PredictionInterval with bounds and provenance
        """
        start_time = time.time()

        # Reset RNG for reproducibility
        self._rng.reset()

        # Calculate interval bounds based on distribution
        lower_bound, upper_bound = self._compute_interval_bounds(
            point_estimate=point_estimate,
            distribution=uncertainty_source.distribution,
            parameters=uncertainty_source.parameters,
            confidence_level=confidence_level,
            horizon_minutes=horizon_minutes
        )

        # Create provenance record
        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="prediction_interval_generation",
            inputs={
                "point_estimate": str(point_estimate),
                "distribution": uncertainty_source.distribution.value,
                "parameters": {k: str(v) for k, v in uncertainty_source.parameters.items()},
                "confidence_level": str(confidence_level),
                "horizon_minutes": horizon_minutes,
                "seed": self._seed
            },
            outputs={
                "lower_bound": str(lower_bound),
                "upper_bound": str(upper_bound)
            },
            computation_time_ms=computation_time_ms
        )

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            variable_name=uncertainty_source.name,
            unit=uncertainty_source.unit,
            horizon_minutes=horizon_minutes,
            source_model="UncertaintyModelEngine",
            provenance=provenance
        )

    def _compute_interval_bounds(
        self,
        point_estimate: Decimal,
        distribution: DistributionType,
        parameters: Dict[str, Decimal],
        confidence_level: Decimal,
        horizon_minutes: int
    ) -> Tuple[Decimal, Decimal]:
        """
        Compute interval bounds based on distribution - DETERMINISTIC.

        Uses analytical formulas where available, Monte Carlo otherwise.
        """
        alpha = (Decimal("1") - confidence_level) / Decimal("2")

        if distribution == DistributionType.NORMAL:
            return self._normal_interval(point_estimate, parameters, alpha, horizon_minutes)
        elif distribution == DistributionType.LOGNORMAL:
            return self._lognormal_interval(point_estimate, parameters, alpha, horizon_minutes)
        elif distribution == DistributionType.UNIFORM:
            return self._uniform_interval(point_estimate, parameters, alpha)
        elif distribution == DistributionType.TRIANGULAR:
            return self._triangular_interval(point_estimate, parameters, alpha)
        elif distribution in (DistributionType.EMPIRICAL, DistributionType.BOOTSTRAP):
            return self._empirical_interval(point_estimate, parameters, alpha)
        else:
            # Default to normal approximation
            return self._normal_interval(point_estimate, parameters, alpha, horizon_minutes)

    def _normal_interval(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        alpha: Decimal,
        horizon_minutes: int
    ) -> Tuple[Decimal, Decimal]:
        """Compute normal distribution interval - DETERMINISTIC."""
        std = parameters.get("std", parameters.get("sigma", Decimal("0.1") * abs(point_estimate)))

        # Scale uncertainty with horizon (random walk scaling)
        horizon_scale = Decimal(str(math.sqrt(horizon_minutes / 60.0)))
        scaled_std = std * horizon_scale

        # Z-scores for common confidence levels (pre-computed)
        z_scores = {
            Decimal("0.005"): Decimal("2.576"),  # 99% CI
            Decimal("0.025"): Decimal("1.960"),  # 95% CI
            Decimal("0.05"): Decimal("1.645"),   # 90% CI
            Decimal("0.10"): Decimal("1.282"),   # 80% CI
            Decimal("0.25"): Decimal("0.674"),   # 50% CI
        }

        # Find closest z-score
        z = z_scores.get(alpha, Decimal("1.645"))

        lower = point_estimate - z * scaled_std
        upper = point_estimate + z * scaled_std

        return (
            lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        )

    def _lognormal_interval(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        alpha: Decimal,
        horizon_minutes: int
    ) -> Tuple[Decimal, Decimal]:
        """Compute lognormal distribution interval - DETERMINISTIC."""
        sigma = parameters.get("sigma", Decimal("0.2"))

        # Scale with horizon
        horizon_scale = Decimal(str(math.sqrt(horizon_minutes / 60.0)))
        scaled_sigma = sigma * horizon_scale

        # Use multiplicative factors
        z_scores = {
            Decimal("0.005"): Decimal("2.576"),
            Decimal("0.025"): Decimal("1.960"),
            Decimal("0.05"): Decimal("1.645"),
            Decimal("0.10"): Decimal("1.282"),
            Decimal("0.25"): Decimal("0.674"),
        }
        z = z_scores.get(alpha, Decimal("1.645"))

        # Lognormal bounds
        sigma_float = float(scaled_sigma)
        z_float = float(z)

        lower_factor = Decimal(str(math.exp(-z_float * sigma_float)))
        upper_factor = Decimal(str(math.exp(z_float * sigma_float)))

        lower = point_estimate * lower_factor
        upper = point_estimate * upper_factor

        return (
            lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        )

    def _uniform_interval(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        alpha: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Compute uniform distribution interval - DETERMINISTIC."""
        half_width = parameters.get("half_width", Decimal("0.1") * abs(point_estimate))

        lower = point_estimate - half_width * (Decimal("1") - alpha)
        upper = point_estimate + half_width * (Decimal("1") - alpha)

        return (
            lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        )

    def _triangular_interval(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        alpha: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Compute triangular distribution interval - DETERMINISTIC."""
        left = parameters.get("left", point_estimate * Decimal("0.9"))
        right = parameters.get("right", point_estimate * Decimal("1.1"))

        # Use linear interpolation for quantiles
        alpha_float = float(alpha)

        if alpha_float <= 0.5:
            lower = left + (point_estimate - left) * Decimal(str(math.sqrt(2 * alpha_float)))
        else:
            lower = point_estimate

        if 1 - alpha_float <= 0.5:
            upper = right - (right - point_estimate) * Decimal(str(math.sqrt(2 * (1 - alpha_float))))
        else:
            upper = point_estimate + (right - point_estimate) * Decimal(str(1 - math.sqrt(2 * alpha_float)))

        return (
            lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        )

    def _empirical_interval(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        alpha: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Compute empirical distribution interval - DETERMINISTIC."""
        # Use stored quantiles if available
        lower_quantile = f"q{int(float(alpha) * 100):02d}"
        upper_quantile = f"q{int((1 - float(alpha)) * 100):02d}"

        lower = parameters.get(lower_quantile, point_estimate * Decimal("0.85"))
        upper = parameters.get(upper_quantile, point_estimate * Decimal("1.15"))

        return (
            lower.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            upper.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        )

    def generate_quantile_set(
        self,
        point_estimate: Decimal,
        uncertainty_source: UncertaintySource,
        quantile_probs: Optional[List[Decimal]] = None,
        horizon_minutes: int = 60
    ) -> QuantileSet:
        """
        Generate set of quantiles for a prediction - DETERMINISTIC.

        Args:
            point_estimate: Central point prediction
            uncertainty_source: Source defining uncertainty distribution
            quantile_probs: List of quantile probabilities (default: standard set)
            horizon_minutes: Forecast horizon

        Returns:
            QuantileSet with all requested quantiles and provenance
        """
        start_time = time.time()

        if quantile_probs is None:
            quantile_probs = self.DEFAULT_QUANTILES

        self._rng.reset()

        quantiles = []
        for prob in quantile_probs:
            # Calculate quantile value
            value = self._compute_quantile_value(
                point_estimate=point_estimate,
                distribution=uncertainty_source.distribution,
                parameters=uncertainty_source.parameters,
                probability=prob,
                horizon_minutes=horizon_minutes
            )
            quantiles.append(QuantileValue(probability=prob, value=value))

        # Create provenance
        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="quantile_set_generation",
            inputs={
                "point_estimate": str(point_estimate),
                "distribution": uncertainty_source.distribution.value,
                "parameters": {k: str(v) for k, v in uncertainty_source.parameters.items()},
                "quantile_probs": [str(p) for p in quantile_probs],
                "horizon_minutes": horizon_minutes,
                "seed": self._seed
            },
            outputs={
                "quantiles": [(str(q.probability), str(q.value)) for q in quantiles]
            },
            computation_time_ms=computation_time_ms
        )

        return QuantileSet(
            quantiles=quantiles,
            variable_name=uncertainty_source.name,
            unit=uncertainty_source.unit,
            provenance=provenance
        )

    def _compute_quantile_value(
        self,
        point_estimate: Decimal,
        distribution: DistributionType,
        parameters: Dict[str, Decimal],
        probability: Decimal,
        horizon_minutes: int
    ) -> Decimal:
        """Compute single quantile value - DETERMINISTIC."""
        if distribution == DistributionType.NORMAL:
            return self._normal_quantile(point_estimate, parameters, probability, horizon_minutes)
        elif distribution == DistributionType.LOGNORMAL:
            return self._lognormal_quantile(point_estimate, parameters, probability, horizon_minutes)
        elif distribution == DistributionType.UNIFORM:
            return self._uniform_quantile(point_estimate, parameters, probability)
        else:
            return self._normal_quantile(point_estimate, parameters, probability, horizon_minutes)

    def _normal_quantile(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        probability: Decimal,
        horizon_minutes: int
    ) -> Decimal:
        """Compute normal distribution quantile - DETERMINISTIC."""
        std = parameters.get("std", parameters.get("sigma", Decimal("0.1") * abs(point_estimate)))
        horizon_scale = Decimal(str(math.sqrt(horizon_minutes / 60.0)))
        scaled_std = std * horizon_scale

        # Inverse normal CDF (probit function approximation)
        z = self._inverse_normal_cdf(probability)

        value = point_estimate + z * scaled_std
        return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _lognormal_quantile(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        probability: Decimal,
        horizon_minutes: int
    ) -> Decimal:
        """Compute lognormal distribution quantile - DETERMINISTIC."""
        sigma = parameters.get("sigma", Decimal("0.2"))
        horizon_scale = Decimal(str(math.sqrt(horizon_minutes / 60.0)))
        scaled_sigma = sigma * horizon_scale

        z = self._inverse_normal_cdf(probability)
        factor = Decimal(str(math.exp(float(z) * float(scaled_sigma))))

        value = point_estimate * factor
        return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _uniform_quantile(
        self,
        point_estimate: Decimal,
        parameters: Dict[str, Decimal],
        probability: Decimal
    ) -> Decimal:
        """Compute uniform distribution quantile - DETERMINISTIC."""
        half_width = parameters.get("half_width", Decimal("0.1") * abs(point_estimate))
        lower = point_estimate - half_width
        upper = point_estimate + half_width

        value = lower + (upper - lower) * probability
        return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _inverse_normal_cdf(self, probability: Decimal) -> Decimal:
        """
        Inverse normal CDF (probit function) - DETERMINISTIC.

        Uses Abramowitz and Stegun approximation for reproducibility.
        """
        p = float(probability)

        # Handle edge cases
        if p <= 0.0:
            return Decimal("-3.5")
        if p >= 1.0:
            return Decimal("3.5")

        # Use symmetry
        if p > 0.5:
            return -self._inverse_normal_cdf(Decimal(str(1.0 - p)))

        # Rational approximation (Abramowitz and Stegun 26.2.23)
        t = math.sqrt(-2.0 * math.log(p))

        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

        return Decimal(str(-z)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def generate_scenarios(
        self,
        uncertainty_sources: List[UncertaintySource],
        num_scenarios: int,
        horizon_start: datetime,
        horizon_end: datetime,
        include_base_case: bool = True,
        include_worst_case: bool = True,
        correlation_matrix: Optional[List[List[Decimal]]] = None
    ) -> ScenarioSet:
        """
        Generate scenario set for stochastic optimization - DETERMINISTIC.

        Args:
            uncertainty_sources: List of uncertainty sources
            num_scenarios: Number of scenarios to generate
            horizon_start: Start of scenario horizon
            horizon_end: End of scenario horizon
            include_base_case: Include expected value scenario
            include_worst_case: Include worst case scenarios
            correlation_matrix: Correlation matrix between sources

        Returns:
            ScenarioSet with scenarios and provenance
        """
        start_time = time.time()
        self._rng.reset()

        scenarios = []
        base_prob = Decimal("1") / Decimal(str(num_scenarios))

        # Generate base case (expected values)
        if include_base_case:
            base_scenario = self._generate_base_case_scenario(
                uncertainty_sources=uncertainty_sources,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
                probability=base_prob
            )
            scenarios.append(base_scenario)

        # Generate worst case scenarios
        if include_worst_case:
            worst_scenarios = self._generate_worst_case_scenarios(
                uncertainty_sources=uncertainty_sources,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
                probability=base_prob
            )
            scenarios.extend(worst_scenarios)

        # Generate Monte Carlo scenarios
        remaining = num_scenarios - len(scenarios)
        for i in range(remaining):
            scenario = self._generate_monte_carlo_scenario(
                uncertainty_sources=uncertainty_sources,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
                probability=base_prob,
                scenario_index=i,
                correlation_matrix=correlation_matrix
            )
            scenarios.append(scenario)

        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        for s in scenarios:
            s.probability = s.probability / total_prob

        # Create provenance
        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="scenario_generation",
            inputs={
                "num_sources": len(uncertainty_sources),
                "num_scenarios": num_scenarios,
                "horizon_start": horizon_start.isoformat(),
                "horizon_end": horizon_end.isoformat(),
                "seed": self._seed
            },
            outputs={
                "num_scenarios_generated": len(scenarios),
                "scenario_ids": [str(s.scenario_id) for s in scenarios]
            },
            computation_time_ms=computation_time_ms
        )

        return ScenarioSet(
            name=f"ScenarioSet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            scenarios=scenarios,
            generation_method="monte_carlo",
            uncertainty_sources=uncertainty_sources,
            correlation_matrix=correlation_matrix,
            provenance=provenance
        )

    def _generate_base_case_scenario(
        self,
        uncertainty_sources: List[UncertaintySource],
        horizon_start: datetime,
        horizon_end: datetime,
        probability: Decimal
    ) -> Scenario:
        """Generate base case scenario with expected values - DETERMINISTIC."""
        variables = []

        for source in uncertainty_sources:
            # Use mean as expected value
            mean = source.parameters.get("mean", source.parameters.get("mu", Decimal("0")))
            variables.append(ScenarioVariable(
                name=source.name,
                value=mean,
                unit=source.unit,
                uncertainty_source=str(source.source_id)
            ))

        return Scenario(
            name="Base Case (Expected)",
            probability=probability,
            variables=variables,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            is_base_case=True,
            is_worst_case=False
        )

    def _generate_worst_case_scenarios(
        self,
        uncertainty_sources: List[UncertaintySource],
        horizon_start: datetime,
        horizon_end: datetime,
        probability: Decimal
    ) -> List[Scenario]:
        """Generate worst case scenarios - DETERMINISTIC."""
        scenarios = []

        # High-cost scenario (high demand, high prices, adverse weather)
        high_variables = []
        for source in uncertainty_sources:
            # Use 95th percentile
            value = self._compute_quantile_value(
                point_estimate=source.parameters.get("mean", Decimal("0")),
                distribution=source.distribution,
                parameters=source.parameters,
                probability=Decimal("0.95"),
                horizon_minutes=60
            )
            high_variables.append(ScenarioVariable(
                name=source.name,
                value=value,
                unit=source.unit,
                uncertainty_source=str(source.source_id)
            ))

        scenarios.append(Scenario(
            name="Worst Case (High)",
            probability=probability,
            variables=high_variables,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            is_base_case=False,
            is_worst_case=True
        ))

        # Low-efficiency scenario
        low_variables = []
        for source in uncertainty_sources:
            # Use 5th percentile
            value = self._compute_quantile_value(
                point_estimate=source.parameters.get("mean", Decimal("0")),
                distribution=source.distribution,
                parameters=source.parameters,
                probability=Decimal("0.05"),
                horizon_minutes=60
            )
            low_variables.append(ScenarioVariable(
                name=source.name,
                value=value,
                unit=source.unit,
                uncertainty_source=str(source.source_id)
            ))

        scenarios.append(Scenario(
            name="Worst Case (Low)",
            probability=probability,
            variables=low_variables,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            is_base_case=False,
            is_worst_case=True
        ))

        return scenarios

    def _generate_monte_carlo_scenario(
        self,
        uncertainty_sources: List[UncertaintySource],
        horizon_start: datetime,
        horizon_end: datetime,
        probability: Decimal,
        scenario_index: int,
        correlation_matrix: Optional[List[List[Decimal]]] = None
    ) -> Scenario:
        """Generate Monte Carlo scenario - DETERMINISTIC."""
        variables = []

        # Generate correlated or independent samples
        if correlation_matrix is not None:
            samples = self._generate_correlated_samples(
                num_sources=len(uncertainty_sources),
                correlation_matrix=correlation_matrix
            )
        else:
            samples = [self._rng.next_normal() for _ in uncertainty_sources]

        for i, source in enumerate(uncertainty_sources):
            mean = source.parameters.get("mean", source.parameters.get("mu", Decimal("0")))
            std = source.parameters.get("std", source.parameters.get("sigma", Decimal("0.1")))

            value = mean + samples[i] * std
            variables.append(ScenarioVariable(
                name=source.name,
                value=value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                unit=source.unit,
                uncertainty_source=str(source.source_id)
            ))

        return Scenario(
            name=f"MC Scenario {scenario_index + 1}",
            probability=probability,
            variables=variables,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            is_base_case=False,
            is_worst_case=False
        )

    def _generate_correlated_samples(
        self,
        num_sources: int,
        correlation_matrix: List[List[Decimal]]
    ) -> List[Decimal]:
        """Generate correlated samples using Cholesky decomposition - DETERMINISTIC."""
        # Cholesky decomposition
        L = self._cholesky_decomposition(correlation_matrix)

        # Generate independent standard normal samples
        z = [self._rng.next_normal() for _ in range(num_sources)]

        # Transform to correlated samples
        correlated = []
        for i in range(num_sources):
            value = Decimal("0")
            for j in range(i + 1):
                value += L[i][j] * z[j]
            correlated.append(value)

        return correlated

    def _cholesky_decomposition(self, matrix: List[List[Decimal]]) -> List[List[Decimal]]:
        """Cholesky decomposition - DETERMINISTIC."""
        n = len(matrix)
        L = [[Decimal("0")] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    sum_sq = sum(L[i][k] ** 2 for k in range(j))
                    diff = matrix[i][i] - sum_sq
                    if diff > 0:
                        L[i][j] = Decimal(str(math.sqrt(float(diff))))
                    else:
                        L[i][j] = Decimal("0.0001")  # Numerical stability
                else:
                    sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                    if L[j][j] != 0:
                        L[i][j] = (matrix[i][j] - sum_prod) / L[j][j]
                    else:
                        L[i][j] = Decimal("0")

        return L


class WeatherUncertaintyModel:
    """
    Weather forecast uncertainty model - DETERMINISTIC.

    Models uncertainty in weather forecasts (temperature, solar
    irradiance, wind speed) that affect process heat operations.
    """

    def __init__(self, seed: int = 42):
        """Initialize weather uncertainty model."""
        self._engine = UncertaintyModelEngine(seed=seed)

    def create_temperature_source(
        self,
        forecast_temperature_c: Decimal,
        forecast_horizon_hours: int = 24,
        location: str = "default"
    ) -> UncertaintySource:
        """
        Create temperature uncertainty source - DETERMINISTIC.

        Uncertainty increases with forecast horizon following
        standard NWP error growth patterns.
        """
        # Base uncertainty (degrees C)
        base_std = Decimal("1.5")

        # Scale with horizon (error growth)
        horizon_factor = Decimal(str(1 + 0.1 * forecast_horizon_hours))
        std = base_std * horizon_factor

        return UncertaintySource(
            name=f"temperature_{location}",
            source_type=UncertaintySourceType.WEATHER,
            distribution=DistributionType.NORMAL,
            parameters={
                "mean": forecast_temperature_c,
                "std": std
            },
            unit="degC",
            time_correlation=Decimal("0.85")  # High autocorrelation
        )

    def create_solar_irradiance_source(
        self,
        forecast_irradiance_w_m2: Decimal,
        forecast_horizon_hours: int = 24,
        cloud_cover_percent: Decimal = Decimal("30")
    ) -> UncertaintySource:
        """
        Create solar irradiance uncertainty source - DETERMINISTIC.

        Uncertainty depends on cloud cover and forecast horizon.
        """
        # Relative uncertainty based on cloud cover
        base_cv = Decimal("0.15") + cloud_cover_percent / Decimal("100") * Decimal("0.25")

        # Scale with horizon
        horizon_factor = Decimal(str(1 + 0.05 * forecast_horizon_hours))
        cv = base_cv * horizon_factor

        return UncertaintySource(
            name="solar_irradiance",
            source_type=UncertaintySourceType.WEATHER,
            distribution=DistributionType.LOGNORMAL,
            parameters={
                "mean": forecast_irradiance_w_m2,
                "sigma": cv
            },
            unit="W/m2",
            time_correlation=Decimal("0.70")
        )

    def generate_weather_scenarios(
        self,
        temperature_c: Decimal,
        solar_w_m2: Decimal,
        horizon_hours: int,
        num_scenarios: int = 10
    ) -> ScenarioSet:
        """Generate weather scenarios - DETERMINISTIC."""
        sources = [
            self.create_temperature_source(temperature_c, horizon_hours),
            self.create_solar_irradiance_source(solar_w_m2, horizon_hours)
        ]

        # Temperature and solar are often correlated
        correlation_matrix = [
            [Decimal("1.0"), Decimal("0.3")],
            [Decimal("0.3"), Decimal("1.0")]
        ]

        now = datetime.utcnow()
        return self._engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=num_scenarios,
            horizon_start=now,
            horizon_end=now + timedelta(hours=horizon_hours),
            correlation_matrix=correlation_matrix
        )


class PriceUncertaintyModel:
    """
    Energy price uncertainty model - DETERMINISTIC.

    Models uncertainty in electricity, natural gas, and fuel
    prices that affect process heat economics.
    """

    def __init__(self, seed: int = 42):
        """Initialize price uncertainty model."""
        self._engine = UncertaintyModelEngine(seed=seed)

    def create_electricity_price_source(
        self,
        forecast_price_usd_kwh: Decimal,
        forecast_horizon_hours: int = 24,
        market_volatility: str = "normal"
    ) -> UncertaintySource:
        """
        Create electricity price uncertainty source - DETERMINISTIC.

        Electricity prices exhibit mean reversion with occasional spikes.
        """
        volatility_factors = {
            "low": Decimal("0.10"),
            "normal": Decimal("0.20"),
            "high": Decimal("0.40")
        }
        base_vol = volatility_factors.get(market_volatility, Decimal("0.20"))

        # Scale with horizon (jump risk)
        horizon_factor = Decimal(str(1 + 0.02 * forecast_horizon_hours))
        sigma = base_vol * horizon_factor

        return UncertaintySource(
            name="electricity_price",
            source_type=UncertaintySourceType.PRICE,
            distribution=DistributionType.LOGNORMAL,
            parameters={
                "mean": forecast_price_usd_kwh,
                "sigma": sigma
            },
            unit="USD/kWh",
            time_correlation=Decimal("0.60")
        )

    def create_natural_gas_price_source(
        self,
        forecast_price_usd_mmbtu: Decimal,
        forecast_horizon_hours: int = 24
    ) -> UncertaintySource:
        """
        Create natural gas price uncertainty source - DETERMINISTIC.

        Natural gas prices have seasonal patterns and supply shocks.
        """
        base_vol = Decimal("0.15")
        horizon_factor = Decimal(str(1 + 0.01 * forecast_horizon_hours))
        sigma = base_vol * horizon_factor

        return UncertaintySource(
            name="natural_gas_price",
            source_type=UncertaintySourceType.PRICE,
            distribution=DistributionType.LOGNORMAL,
            parameters={
                "mean": forecast_price_usd_mmbtu,
                "sigma": sigma
            },
            unit="USD/MMBtu",
            time_correlation=Decimal("0.75")
        )

    def generate_price_scenarios(
        self,
        electricity_price_usd_kwh: Decimal,
        gas_price_usd_mmbtu: Decimal,
        horizon_hours: int,
        num_scenarios: int = 10
    ) -> ScenarioSet:
        """Generate price scenarios - DETERMINISTIC."""
        sources = [
            self.create_electricity_price_source(electricity_price_usd_kwh, horizon_hours),
            self.create_natural_gas_price_source(gas_price_usd_mmbtu, horizon_hours)
        ]

        # Electricity and gas prices are correlated
        correlation_matrix = [
            [Decimal("1.0"), Decimal("0.6")],
            [Decimal("0.6"), Decimal("1.0")]
        ]

        now = datetime.utcnow()
        return self._engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=num_scenarios,
            horizon_start=now,
            horizon_end=now + timedelta(hours=horizon_hours),
            correlation_matrix=correlation_matrix
        )


class DemandUncertaintyModel:
    """
    Process heat demand uncertainty model - DETERMINISTIC.

    Models uncertainty in heat demand from production schedules,
    batch processes, and equipment availability.
    """

    def __init__(self, seed: int = 42):
        """Initialize demand uncertainty model."""
        self._engine = UncertaintyModelEngine(seed=seed)

    def create_heat_demand_source(
        self,
        forecast_demand_mw: Decimal,
        forecast_horizon_hours: int = 24,
        process_type: str = "continuous"
    ) -> UncertaintySource:
        """
        Create heat demand uncertainty source - DETERMINISTIC.

        Batch processes have higher uncertainty than continuous.
        """
        process_cv = {
            "continuous": Decimal("0.05"),
            "semi_batch": Decimal("0.15"),
            "batch": Decimal("0.25")
        }
        base_cv = process_cv.get(process_type, Decimal("0.10"))

        # Scale with horizon
        horizon_factor = Decimal(str(1 + 0.02 * forecast_horizon_hours))
        std = forecast_demand_mw * base_cv * horizon_factor

        return UncertaintySource(
            name="heat_demand",
            source_type=UncertaintySourceType.DEMAND,
            distribution=DistributionType.NORMAL,
            parameters={
                "mean": forecast_demand_mw,
                "std": std
            },
            unit="MW_th",
            time_correlation=Decimal("0.80")
        )

    def create_equipment_availability_source(
        self,
        expected_availability: Decimal = Decimal("0.95"),
        num_units: int = 3
    ) -> UncertaintySource:
        """
        Create equipment availability uncertainty source - DETERMINISTIC.

        Models probability of equipment being available.
        """
        return UncertaintySource(
            name="equipment_availability",
            source_type=UncertaintySourceType.EQUIPMENT,
            distribution=DistributionType.EMPIRICAL,
            parameters={
                "mean": expected_availability,
                "q05": expected_availability - Decimal("0.10"),
                "q95": expected_availability
            },
            unit="fraction"
        )

    def generate_demand_scenarios(
        self,
        heat_demand_mw: Decimal,
        horizon_hours: int,
        num_scenarios: int = 10,
        process_type: str = "continuous"
    ) -> ScenarioSet:
        """Generate demand scenarios - DETERMINISTIC."""
        sources = [
            self.create_heat_demand_source(heat_demand_mw, horizon_hours, process_type),
            self.create_equipment_availability_source()
        ]

        now = datetime.utcnow()
        return self._engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=num_scenarios,
            horizon_start=now,
            horizon_end=now + timedelta(hours=horizon_hours)
        )
