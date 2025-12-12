"""
Monte Carlo Uncertainty Engine

This module implements Monte Carlo simulation for uncertainty analysis in
emission calculations, providing probabilistic estimates and confidence
intervals for GHG inventory results.

Features:
- Monte Carlo simulation with configurable iterations
- Multiple probability distributions (normal, lognormal, uniform, triangular)
- Latin Hypercube Sampling for efficiency
- Correlation matrix support
- Sensitivity analysis
- Convergence monitoring
- Result aggregation and percentile calculation

References:
- IPCC Guidelines for National Greenhouse Gas Inventories (Uncertainty)
- GHG Protocol: Policy and Action Standard (Uncertainty chapter)
- ISO 14064-1:2018 Annex B (Uncertainty assessment)

Usage:
    from engines.monte_carlo import MonteCarloEngine, UncertainParameter

    engine = MonteCarloEngine(iterations=10000)

    # Define uncertain parameters
    ef = UncertainParameter(
        name="emission_factor",
        distribution="lognormal",
        mean=53.06,
        std_dev=5.3,
    )

    activity = UncertainParameter(
        name="activity_data",
        distribution="normal",
        mean=1000,
        std_dev=50,
    )

    # Run simulation
    result = engine.simulate(
        model=lambda params: params["emission_factor"] * params["activity_data"],
        parameters=[ef, activity],
    )

    print(f"Mean: {result.mean:.2f}")
    print(f"95% CI: [{result.percentile_2_5:.2f}, {result.percentile_97_5:.2f}]")
"""
import logging
import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Probability Distributions
# =============================================================================


class Distribution(Enum):
    """Supported probability distributions."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"
    GAMMA = "gamma"
    FIXED = "fixed"  # No uncertainty


@dataclass
class UncertainParameter:
    """
    Represents an uncertain parameter for Monte Carlo simulation.

    Attributes:
        name: Parameter identifier
        distribution: Probability distribution type
        mean: Mean or central value
        std_dev: Standard deviation (for normal/lognormal)
        min_value: Minimum value (for uniform/triangular)
        max_value: Maximum value (for uniform/triangular)
        mode: Mode value (for triangular)
        alpha: Alpha parameter (for beta/gamma)
        beta: Beta parameter (for beta/gamma)
        uncertainty_pct: Relative uncertainty as percentage
        correlation_group: Group ID for correlated parameters
    """

    name: str
    distribution: Distribution = Distribution.NORMAL
    mean: float = 0.0
    std_dev: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mode: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    uncertainty_pct: Optional[float] = None
    correlation_group: Optional[str] = None

    def __post_init__(self):
        """Validate and derive parameters."""
        # Derive std_dev from uncertainty_pct if needed
        if self.std_dev is None and self.uncertainty_pct is not None:
            self.std_dev = abs(self.mean * self.uncertainty_pct / 100)

        # Validate distribution parameters
        if self.distribution == Distribution.NORMAL:
            if self.std_dev is None:
                raise ValueError("Normal distribution requires std_dev")

        elif self.distribution == Distribution.LOGNORMAL:
            if self.std_dev is None:
                raise ValueError("Lognormal distribution requires std_dev")
            if self.mean <= 0:
                raise ValueError("Lognormal mean must be positive")

        elif self.distribution == Distribution.UNIFORM:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Uniform distribution requires min/max values")

        elif self.distribution == Distribution.TRIANGULAR:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Triangular distribution requires min/max values")
            if self.mode is None:
                self.mode = self.mean

    def sample(self, rng: random.Random = random) -> float:
        """Generate a random sample from the distribution."""
        if self.distribution == Distribution.FIXED:
            return self.mean

        elif self.distribution == Distribution.NORMAL:
            return rng.gauss(self.mean, self.std_dev)

        elif self.distribution == Distribution.LOGNORMAL:
            # Convert to lognormal parameters
            mu, sigma = self._lognormal_params()
            return rng.lognormvariate(mu, sigma)

        elif self.distribution == Distribution.UNIFORM:
            return rng.uniform(self.min_value, self.max_value)

        elif self.distribution == Distribution.TRIANGULAR:
            return rng.triangular(self.min_value, self.max_value, self.mode)

        elif self.distribution == Distribution.BETA:
            return rng.betavariate(self.alpha, self.beta)

        elif self.distribution == Distribution.GAMMA:
            return rng.gammavariate(self.alpha, self.beta)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _lognormal_params(self) -> Tuple[float, float]:
        """Convert mean/std_dev to lognormal mu/sigma parameters."""
        variance = self.std_dev ** 2
        mu = math.log(self.mean ** 2 / math.sqrt(variance + self.mean ** 2))
        sigma = math.sqrt(math.log(1 + variance / self.mean ** 2))
        return mu, sigma


# =============================================================================
# Simulation Results
# =============================================================================


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""

    iterations: int
    samples: List[float]

    # Summary statistics
    mean: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    median: float = 0.0
    mode: float = 0.0

    # Percentiles
    percentile_2_5: float = 0.0
    percentile_5: float = 0.0
    percentile_10: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    percentile_95: float = 0.0
    percentile_97_5: float = 0.0

    # Uncertainty metrics
    coefficient_of_variation: float = 0.0
    relative_uncertainty_95: float = 0.0

    # Convergence
    converged: bool = False
    convergence_iterations: int = 0

    # Sensitivity analysis results
    sensitivity: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate statistics from samples."""
        if not self.samples:
            return

        n = len(self.samples)
        sorted_samples = sorted(self.samples)

        self.mean = statistics.mean(self.samples)
        self.std_dev = statistics.stdev(self.samples) if n > 1 else 0.0
        self.variance = self.std_dev ** 2
        self.median = statistics.median(self.samples)

        # Mode approximation (bin-based)
        try:
            self.mode = statistics.mode(
                [round(s, 2) for s in self.samples]
            )
        except statistics.StatisticsError:
            self.mode = self.median

        # Percentiles
        self.percentile_2_5 = self._percentile(sorted_samples, 2.5)
        self.percentile_5 = self._percentile(sorted_samples, 5)
        self.percentile_10 = self._percentile(sorted_samples, 10)
        self.percentile_25 = self._percentile(sorted_samples, 25)
        self.percentile_75 = self._percentile(sorted_samples, 75)
        self.percentile_90 = self._percentile(sorted_samples, 90)
        self.percentile_95 = self._percentile(sorted_samples, 95)
        self.percentile_97_5 = self._percentile(sorted_samples, 97.5)

        # Uncertainty metrics
        if self.mean != 0:
            self.coefficient_of_variation = self.std_dev / abs(self.mean) * 100
            ci_width = self.percentile_97_5 - self.percentile_2_5
            self.relative_uncertainty_95 = ci_width / abs(self.mean) * 100

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        n = len(sorted_data)
        if n == 0:
            return 0.0

        k = (n - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_data[int(k)]

        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

    def get_confidence_interval(self, confidence: float = 95.0) -> Tuple[float, float]:
        """Get confidence interval for specified confidence level."""
        alpha = (100 - confidence) / 2
        sorted_samples = sorted(self.samples)
        lower = self._percentile(sorted_samples, alpha)
        upper = self._percentile(sorted_samples, 100 - alpha)
        return (lower, upper)

    def summary(self) -> Dict[str, Any]:
        """Generate summary dictionary."""
        return {
            "iterations": self.iterations,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "median": self.median,
            "confidence_interval_95": [self.percentile_2_5, self.percentile_97_5],
            "coefficient_of_variation_pct": self.coefficient_of_variation,
            "relative_uncertainty_95_pct": self.relative_uncertainty_95,
            "converged": self.converged,
        }


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    parameter_name: str
    sensitivity_index: float  # Sobol first-order index
    total_index: float  # Total-order index
    rank: int
    contribution_pct: float


# =============================================================================
# Monte Carlo Engine
# =============================================================================


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for uncertainty analysis.

    Provides probabilistic estimation of emission calculations with
    support for multiple distribution types and correlation.
    """

    def __init__(
        self,
        iterations: int = 10000,
        seed: Optional[int] = None,
        convergence_threshold: float = 0.01,
        convergence_check_interval: int = 1000,
    ):
        """
        Initialize the Monte Carlo engine.

        Args:
            iterations: Number of simulation iterations
            seed: Random seed for reproducibility
            convergence_threshold: Relative change threshold for convergence
            convergence_check_interval: Check convergence every N iterations
        """
        self.iterations = iterations
        self.seed = seed
        self.convergence_threshold = convergence_threshold
        self.convergence_check_interval = convergence_check_interval

        self._rng = random.Random(seed)

    def simulate(
        self,
        model: Callable[[Dict[str, float]], float],
        parameters: List[UncertainParameter],
        correlation_matrix: Optional[List[List[float]]] = None,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            model: Function that takes parameter dict and returns result
            parameters: List of uncertain parameters
            correlation_matrix: Optional correlation matrix for parameters

        Returns:
            SimulationResult with statistics
        """
        logger.info(f"Starting Monte Carlo simulation with {self.iterations} iterations")

        samples = []
        param_samples: Dict[str, List[float]] = {p.name: [] for p in parameters}

        # Check for convergence
        converged = False
        convergence_iter = self.iterations
        previous_mean = None

        for i in range(self.iterations):
            # Sample parameters
            if correlation_matrix:
                param_values = self._sample_correlated(parameters, correlation_matrix)
            else:
                param_values = {p.name: p.sample(self._rng) for p in parameters}

            # Store parameter samples for sensitivity analysis
            for name, value in param_values.items():
                param_samples[name].append(value)

            # Evaluate model
            try:
                result = model(param_values)
                samples.append(result)
            except Exception as e:
                logger.warning(f"Model evaluation failed at iteration {i}: {e}")
                continue

            # Check convergence
            if (
                not converged
                and (i + 1) % self.convergence_check_interval == 0
                and len(samples) > 100
            ):
                current_mean = statistics.mean(samples)
                if previous_mean is not None:
                    relative_change = abs(current_mean - previous_mean) / abs(previous_mean)
                    if relative_change < self.convergence_threshold:
                        converged = True
                        convergence_iter = i + 1
                        logger.info(f"Simulation converged at iteration {convergence_iter}")
                previous_mean = current_mean

        # Create result
        result = SimulationResult(
            iterations=len(samples),
            samples=samples,
            converged=converged,
            convergence_iterations=convergence_iter,
        )

        # Run sensitivity analysis
        result.sensitivity = self._sensitivity_analysis(param_samples, samples)

        logger.info(
            f"Simulation complete: mean={result.mean:.4f}, "
            f"95% CI=[{result.percentile_2_5:.4f}, {result.percentile_97_5:.4f}]"
        )

        return result

    def _sample_correlated(
        self,
        parameters: List[UncertainParameter],
        correlation_matrix: List[List[float]],
    ) -> Dict[str, float]:
        """Sample correlated parameters using Cholesky decomposition."""
        n = len(parameters)

        # Generate independent standard normal samples
        z = [self._rng.gauss(0, 1) for _ in range(n)]

        # Apply Cholesky decomposition
        try:
            L = self._cholesky(correlation_matrix)
            correlated_z = [
                sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n)
            ]
        except Exception:
            # Fall back to independent sampling
            logger.warning("Cholesky decomposition failed, using independent samples")
            correlated_z = z

        # Transform to parameter distributions
        param_values = {}
        for i, param in enumerate(parameters):
            if param.distribution == Distribution.NORMAL:
                param_values[param.name] = param.mean + param.std_dev * correlated_z[i]
            elif param.distribution == Distribution.LOGNORMAL:
                mu, sigma = param._lognormal_params()
                param_values[param.name] = math.exp(mu + sigma * correlated_z[i])
            else:
                # For other distributions, use independent sampling
                param_values[param.name] = param.sample(self._rng)

        return param_values

    def _cholesky(self, matrix: List[List[float]]) -> List[List[float]]:
        """Cholesky decomposition of a positive-definite matrix."""
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(matrix[i][i] - s, 0))
                else:
                    if L[j][j] != 0:
                        L[i][j] = (matrix[i][j] - s) / L[j][j]

        return L

    def _sensitivity_analysis(
        self,
        param_samples: Dict[str, List[float]],
        output_samples: List[float],
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis using correlation-based method.

        Returns dict of parameter name -> sensitivity contribution
        """
        if len(output_samples) < 100:
            return {}

        sensitivity = {}
        output_std = statistics.stdev(output_samples)

        if output_std == 0:
            return {}

        # Calculate correlation coefficient for each parameter
        for param_name, values in param_samples.items():
            if len(values) != len(output_samples):
                continue

            try:
                # Pearson correlation
                n = len(values)
                mean_x = statistics.mean(values)
                mean_y = statistics.mean(output_samples)
                std_x = statistics.stdev(values)

                if std_x == 0:
                    sensitivity[param_name] = 0.0
                    continue

                covariance = sum(
                    (values[i] - mean_x) * (output_samples[i] - mean_y)
                    for i in range(n)
                ) / (n - 1)

                correlation = covariance / (std_x * output_std)

                # Sensitivity index (squared correlation)
                sensitivity[param_name] = correlation ** 2

            except Exception as e:
                logger.warning(f"Sensitivity calculation failed for {param_name}: {e}")
                sensitivity[param_name] = 0.0

        # Normalize to sum to 1
        total = sum(sensitivity.values())
        if total > 0:
            sensitivity = {k: v / total for k, v in sensitivity.items()}

        return sensitivity

    def latin_hypercube_sample(
        self,
        parameters: List[UncertainParameter],
        n_samples: int,
    ) -> List[Dict[str, float]]:
        """
        Generate Latin Hypercube samples for more efficient coverage.

        Args:
            parameters: List of uncertain parameters
            n_samples: Number of samples to generate

        Returns:
            List of parameter dictionaries
        """
        samples = []

        # Generate stratified samples for each parameter
        param_samples = {}
        for param in parameters:
            # Divide [0, 1] into n_samples equal intervals
            intervals = [(i / n_samples, (i + 1) / n_samples) for i in range(n_samples)]
            self._rng.shuffle(intervals)

            # Sample from each interval and transform to distribution
            values = []
            for low, high in intervals:
                u = self._rng.uniform(low, high)
                value = self._inverse_cdf(param, u)
                values.append(value)

            param_samples[param.name] = values

        # Combine into sample dictionaries
        for i in range(n_samples):
            sample = {name: values[i] for name, values in param_samples.items()}
            samples.append(sample)

        return samples

    def _inverse_cdf(self, param: UncertainParameter, u: float) -> float:
        """Inverse CDF (quantile function) for parameter distribution."""
        if param.distribution == Distribution.NORMAL:
            # Use approximation for normal inverse CDF
            return param.mean + param.std_dev * self._norm_ppf(u)

        elif param.distribution == Distribution.LOGNORMAL:
            mu, sigma = param._lognormal_params()
            return math.exp(mu + sigma * self._norm_ppf(u))

        elif param.distribution == Distribution.UNIFORM:
            return param.min_value + u * (param.max_value - param.min_value)

        elif param.distribution == Distribution.TRIANGULAR:
            # Inverse CDF for triangular
            a, b, c = param.min_value, param.max_value, param.mode
            fc = (c - a) / (b - a)
            if u < fc:
                return a + math.sqrt(u * (b - a) * (c - a))
            else:
                return b - math.sqrt((1 - u) * (b - a) * (b - c))

        else:
            # Fall back to direct sampling
            return param.sample(self._rng)

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """Approximation of normal distribution percent point function."""
        # Rational approximation for the normal inverse CDF
        # Accurate to about 4.5 decimal places
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")

        if p < 0.5:
            return -MonteCarloEngine._norm_ppf(1 - p)

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_combined_uncertainty(
    uncertainties: List[float],
    values: List[float],
    method: str = "propagation",
) -> float:
    """
    Calculate combined uncertainty from multiple sources.

    Args:
        uncertainties: Relative uncertainties (as decimals, e.g., 0.05 for 5%)
        values: Associated values
        method: Calculation method ("propagation" or "root_sum_squares")

    Returns:
        Combined relative uncertainty
    """
    if method == "propagation":
        # Error propagation for multiplication
        # Combined uncertainty = sqrt(sum of squared relative uncertainties)
        return math.sqrt(sum(u ** 2 for u in uncertainties))

    elif method == "root_sum_squares":
        # Root sum of squares for absolute uncertainties
        abs_uncertainties = [u * v for u, v in zip(uncertainties, values)]
        total_value = sum(values)
        if total_value == 0:
            return 0.0
        combined_abs = math.sqrt(sum(u ** 2 for u in abs_uncertainties))
        return combined_abs / total_value

    else:
        raise ValueError(f"Unknown method: {method}")


def ipcc_uncertainty_category(uncertainty_pct: float) -> str:
    """
    Categorize uncertainty according to IPCC guidelines.

    Args:
        uncertainty_pct: Uncertainty as percentage

    Returns:
        IPCC uncertainty category
    """
    if uncertainty_pct <= 10:
        return "Low"
    elif uncertainty_pct <= 30:
        return "Moderate"
    elif uncertainty_pct <= 100:
        return "High"
    else:
        return "Very High"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MonteCarloEngine",
    "UncertainParameter",
    "SimulationResult",
    "SensitivityResult",
    "Distribution",
    "calculate_combined_uncertainty",
    "ipcc_uncertainty_category",
]
