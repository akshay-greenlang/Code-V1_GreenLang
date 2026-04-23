"""
GL-006 HEATRECLAIM - Uncertainty Quantifier

Implements uncertainty quantification for heat recovery systems using
Monte Carlo simulation and Latin Hypercube Sampling to propagate
input uncertainties to output KPIs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
import hashlib
import json
import logging
import math
import random
from copy import deepcopy

from ..core.schemas import (
    HeatStream,
    HENDesign,
    OptimizationResult,
)
from ..core.config import UncertaintyParameters
from ..calculators.pinch_analysis import PinchAnalysisCalculator
from ..calculators.hen_synthesis import HENSynthesizer
from ..calculators.economic_calculator import EconomicCalculator

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyInput:
    """Definition of an uncertain input parameter."""

    name: str
    distribution: str  # "normal", "uniform", "triangular"
    nominal_value: float
    std_dev: Optional[float] = None  # For normal distribution
    min_value: Optional[float] = None  # For uniform/triangular
    max_value: Optional[float] = None  # For uniform/triangular
    mode_value: Optional[float] = None  # For triangular


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification."""

    n_samples: int
    confidence_level: float

    # Output statistics
    output_means: Dict[str, float] = field(default_factory=dict)
    output_stds: Dict[str, float] = field(default_factory=dict)
    output_medians: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Robustness
    feasibility_probability: float = 1.0
    robustness_score: float = 1.0

    # Sensitivity
    sensitivity_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw samples (optional)
    samples: Optional[List[Dict[str, float]]] = None

    # Provenance
    input_hash: str = ""
    seed: int = 42


class UncertaintyQuantifier:
    """
    Uncertainty quantification engine for heat recovery optimization.

    Supports:
    - Monte Carlo simulation
    - Latin Hypercube Sampling (LHS)
    - Sensitivity analysis
    - Robustness scoring

    Example:
        >>> uq = UncertaintyQuantifier()
        >>> uncertain_inputs = [
        ...     UncertaintyInput("flow_rate", "normal", 10.0, std_dev=0.5),
        ...     UncertaintyInput("T_supply", "uniform", 100.0, min_value=95, max_value=105),
        ... ]
        >>> result = uq.analyze(design, uncertain_inputs)
        >>> print(f"TAC 95% CI: {result.confidence_intervals['TAC']}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        params: Optional[UncertaintyParameters] = None,
        delta_t_min: float = 10.0,
    ) -> None:
        """
        Initialize uncertainty quantifier.

        Args:
            params: Uncertainty parameters
            delta_t_min: Minimum approach temperature
        """
        self.params = params or UncertaintyParameters()
        self.delta_t_min = delta_t_min

        self.pinch_calc = PinchAnalysisCalculator(delta_t_min=delta_t_min)
        self.econ_calc = EconomicCalculator()

    def analyze(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        design: Optional[HENDesign] = None,
        uncertain_inputs: Optional[List[UncertaintyInput]] = None,
        n_samples: Optional[int] = None,
        method: str = "monte_carlo",
    ) -> UncertaintyResult:
        """
        Perform uncertainty analysis.

        Args:
            hot_streams: Hot process streams (nominal)
            cold_streams: Cold process streams (nominal)
            design: HEN design to evaluate (optional)
            uncertain_inputs: List of uncertain inputs
            n_samples: Number of samples
            method: Sampling method ("monte_carlo" or "lhs")

        Returns:
            UncertaintyResult with statistics and bounds
        """
        n = n_samples or self.params.n_samples
        random.seed(self.params.random_seed)

        logger.info(
            f"Starting uncertainty analysis: {n} samples, method={method}"
        )

        # Generate default uncertain inputs if not provided
        if uncertain_inputs is None:
            uncertain_inputs = self._generate_default_uncertainties(
                hot_streams, cold_streams
            )

        # Generate samples
        if method == "lhs":
            samples = self._latin_hypercube_samples(uncertain_inputs, n)
        else:
            samples = self._monte_carlo_samples(uncertain_inputs, n)

        # Evaluate each sample
        output_samples = []
        feasible_count = 0

        for i, sample in enumerate(samples):
            try:
                # Apply sample to streams
                hot_perturbed = self._apply_perturbations(
                    hot_streams, sample, "hot"
                )
                cold_perturbed = self._apply_perturbations(
                    cold_streams, sample, "cold"
                )

                # Evaluate
                outputs = self._evaluate_sample(
                    hot_perturbed, cold_perturbed, design
                )

                if outputs.get("is_feasible", True):
                    feasible_count += 1

                output_samples.append(outputs)

            except Exception as e:
                logger.debug(f"Sample {i} failed: {e}")
                output_samples.append({"is_feasible": False})

            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{n} samples")

        # Calculate statistics
        result = self._calculate_statistics(
            output_samples, self.params.confidence_level
        )

        result.n_samples = n
        result.confidence_level = self.params.confidence_level
        result.feasibility_probability = feasible_count / n
        result.seed = self.params.random_seed

        # Calculate sensitivity indices
        if len(samples) >= 100:
            result.sensitivity_indices = self._calculate_sensitivity(
                samples, output_samples, uncertain_inputs
            )

        # Calculate robustness score
        result.robustness_score = self._calculate_robustness(
            output_samples, result.confidence_intervals
        )

        # Compute input hash
        result.input_hash = self._compute_hash({
            "n_samples": n,
            "uncertain_inputs": [u.name for u in uncertain_inputs],
            "seed": self.params.random_seed,
        })

        logger.info(
            f"UQ complete: feasibility={result.feasibility_probability:.1%}, "
            f"robustness={result.robustness_score:.2f}"
        )

        return result

    def _generate_default_uncertainties(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> List[UncertaintyInput]:
        """Generate default uncertainty specifications."""
        uncertainties = []

        # Flow rate uncertainties
        for s in hot_streams + cold_streams:
            uncertainties.append(UncertaintyInput(
                name=f"{s.stream_id}_m_dot",
                distribution="normal",
                nominal_value=s.m_dot_kg_s,
                std_dev=s.m_dot_kg_s * self.params.flow_rate_uncertainty / 100,
            ))

        # Temperature uncertainties
        for s in hot_streams + cold_streams:
            uncertainties.append(UncertaintyInput(
                name=f"{s.stream_id}_T_supply",
                distribution="normal",
                nominal_value=s.T_supply_C,
                std_dev=self.params.temperature_uncertainty,
            ))

        # Cp uncertainties
        for s in hot_streams + cold_streams:
            uncertainties.append(UncertaintyInput(
                name=f"{s.stream_id}_Cp",
                distribution="normal",
                nominal_value=s.Cp_kJ_kgK,
                std_dev=s.Cp_kJ_kgK * self.params.cp_uncertainty / 100,
            ))

        return uncertainties

    def _monte_carlo_samples(
        self,
        uncertain_inputs: List[UncertaintyInput],
        n_samples: int,
    ) -> List[Dict[str, float]]:
        """Generate Monte Carlo samples."""
        samples = []

        for _ in range(n_samples):
            sample = {}
            for ui in uncertain_inputs:
                sample[ui.name] = self._sample_distribution(ui)
            samples.append(sample)

        return samples

    def _latin_hypercube_samples(
        self,
        uncertain_inputs: List[UncertaintyInput],
        n_samples: int,
    ) -> List[Dict[str, float]]:
        """Generate Latin Hypercube samples."""
        n_vars = len(uncertain_inputs)
        samples = []

        # Generate stratified samples for each variable
        strata = []
        for _ in range(n_vars):
            perm = list(range(n_samples))
            random.shuffle(perm)
            strata.append(perm)

        # Convert to actual values
        for i in range(n_samples):
            sample = {}
            for j, ui in enumerate(uncertain_inputs):
                # Get stratum for this variable
                stratum = strata[j][i]
                # Random value within stratum
                u = (stratum + random.random()) / n_samples
                sample[ui.name] = self._inverse_cdf(ui, u)
            samples.append(sample)

        return samples

    def _sample_distribution(self, ui: UncertaintyInput) -> float:
        """Sample from specified distribution."""
        if ui.distribution == "normal":
            std = ui.std_dev or ui.nominal_value * 0.05
            return random.gauss(ui.nominal_value, std)

        elif ui.distribution == "uniform":
            low = ui.min_value if ui.min_value is not None else ui.nominal_value * 0.9
            high = ui.max_value if ui.max_value is not None else ui.nominal_value * 1.1
            return random.uniform(low, high)

        elif ui.distribution == "triangular":
            low = ui.min_value if ui.min_value is not None else ui.nominal_value * 0.9
            high = ui.max_value if ui.max_value is not None else ui.nominal_value * 1.1
            mode = ui.mode_value if ui.mode_value is not None else ui.nominal_value
            return random.triangular(low, high, mode)

        else:
            return ui.nominal_value

    def _inverse_cdf(self, ui: UncertaintyInput, u: float) -> float:
        """Inverse CDF for Latin Hypercube sampling."""
        if ui.distribution == "normal":
            std = ui.std_dev or ui.nominal_value * 0.05
            # Approximate inverse normal CDF
            return ui.nominal_value + std * self._norm_inv(u)

        elif ui.distribution == "uniform":
            low = ui.min_value if ui.min_value is not None else ui.nominal_value * 0.9
            high = ui.max_value if ui.max_value is not None else ui.nominal_value * 1.1
            return low + u * (high - low)

        else:
            return self._sample_distribution(ui)

    def _norm_inv(self, p: float) -> float:
        """Approximate inverse of standard normal CDF."""
        # Rational approximation
        if p <= 0:
            return -4.0
        if p >= 1:
            return 4.0

        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
            return -(2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                   (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
        else:
            t = math.sqrt(-2 * math.log(1 - p))
            return (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                   (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)

    def _apply_perturbations(
        self,
        streams: List[HeatStream],
        sample: Dict[str, float],
        stream_type: str,
    ) -> List[HeatStream]:
        """Apply sampled perturbations to streams."""
        perturbed = []

        for s in streams:
            s_copy = deepcopy(s)

            # Apply flow rate perturbation
            key = f"{s.stream_id}_m_dot"
            if key in sample:
                s_copy.m_dot_kg_s = max(0.1, sample[key])

            # Apply temperature perturbation
            key = f"{s.stream_id}_T_supply"
            if key in sample:
                s_copy.T_supply_C = sample[key]

            # Apply Cp perturbation
            key = f"{s.stream_id}_Cp"
            if key in sample:
                s_copy.Cp_kJ_kgK = max(0.1, sample[key])

            perturbed.append(s_copy)

        return perturbed

    def _evaluate_sample(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        design: Optional[HENDesign],
    ) -> Dict[str, float]:
        """Evaluate outputs for a single sample."""
        outputs = {}

        # Pinch analysis
        pinch_result = self.pinch_calc.calculate(hot_streams, cold_streams)

        outputs["QH_min_kW"] = pinch_result.minimum_hot_utility_kW
        outputs["QC_min_kW"] = pinch_result.minimum_cold_utility_kW
        outputs["pinch_T_C"] = pinch_result.pinch_temperature_C
        outputs["max_recovery_kW"] = pinch_result.maximum_heat_recovery_kW

        # Economic (simplified)
        hours = 8000
        steam_cost = 15.0  # $/GJ
        cw_cost = 0.5  # $/GJ

        utility_cost = (
            pinch_result.minimum_hot_utility_kW * hours * 0.0036 * steam_cost +
            pinch_result.minimum_cold_utility_kW * hours * 0.0036 * cw_cost
        )
        outputs["annual_utility_cost_usd"] = utility_cost

        # Check feasibility (approach temperatures positive)
        outputs["is_feasible"] = pinch_result.is_valid

        return outputs

    def _calculate_statistics(
        self,
        samples: List[Dict[str, float]],
        confidence_level: float,
    ) -> UncertaintyResult:
        """Calculate statistics from samples."""
        result = UncertaintyResult(
            n_samples=len(samples),
            confidence_level=confidence_level,
        )

        # Get output names from first valid sample
        output_names = set()
        for s in samples:
            for key in s:
                if key != "is_feasible":
                    output_names.add(key)

        for name in output_names:
            values = [
                s[name] for s in samples
                if name in s and s.get("is_feasible", True)
            ]

            if not values:
                continue

            # Mean
            mean = sum(values) / len(values)
            result.output_means[name] = round(mean, 4)

            # Standard deviation
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            result.output_stds[name] = round(std, 4)

            # Median
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                median = sorted_values[n // 2]
            result.output_medians[name] = round(median, 4)

            # Confidence interval
            alpha = 1 - confidence_level
            lower_idx = int(n * alpha / 2)
            upper_idx = int(n * (1 - alpha / 2))
            result.confidence_intervals[name] = (
                round(sorted_values[max(0, lower_idx)], 4),
                round(sorted_values[min(n - 1, upper_idx)], 4),
            )

        return result

    def _calculate_sensitivity(
        self,
        input_samples: List[Dict[str, float]],
        output_samples: List[Dict[str, float]],
        uncertain_inputs: List[UncertaintyInput],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate sensitivity indices (correlation-based)."""
        sensitivity = {}

        for out_name in ["QH_min_kW", "annual_utility_cost_usd"]:
            sensitivity[out_name] = {}

            out_values = [
                s.get(out_name, 0) for s in output_samples
                if s.get("is_feasible", True)
            ]

            if len(out_values) < 10:
                continue

            for ui in uncertain_inputs:
                in_values = [
                    s.get(ui.name, ui.nominal_value)
                    for i, s in enumerate(input_samples)
                    if output_samples[i].get("is_feasible", True)
                ]

                if len(in_values) != len(out_values):
                    continue

                # Correlation coefficient
                corr = self._correlation(in_values, out_values)
                sensitivity[out_name][ui.name] = round(corr, 4)

        return sensitivity

    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        if var_x * var_y <= 0:
            return 0.0

        return cov / math.sqrt(var_x * var_y)

    def _calculate_robustness(
        self,
        samples: List[Dict[str, float]],
        confidence_intervals: Dict[str, Tuple[float, float]],
    ) -> float:
        """Calculate overall robustness score."""
        feasible = [s for s in samples if s.get("is_feasible", True)]

        if not feasible:
            return 0.0

        # Robustness based on coefficient of variation
        cv_scores = []
        for name in ["QH_min_kW", "annual_utility_cost_usd"]:
            values = [s.get(name, 0) for s in feasible if name in s]
            if values:
                mean = sum(values) / len(values)
                if mean > 0:
                    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                    cv = std / mean
                    # Lower CV = more robust
                    cv_scores.append(max(0, 1 - cv))

        if not cv_scores:
            return 1.0

        return sum(cv_scores) / len(cv_scores)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
