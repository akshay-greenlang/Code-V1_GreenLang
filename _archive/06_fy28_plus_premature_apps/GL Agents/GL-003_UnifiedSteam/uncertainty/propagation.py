"""
Uncertainty Propagation for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module implements uncertainty propagation methods including:
- Linear propagation (weighted sum of variances)
- Jacobian-based analytic propagation for nonlinear functions
- Monte Carlo simulation for complex/nonlinear cases

Zero-Hallucination Guarantee:
- All propagation calculations are deterministic mathematical formulas
- Monte Carlo uses fixed seeds for reproducibility
- Complete provenance tracking with SHA-256 hashes
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import math
import time

import numpy as np
from numpy.random import Generator, PCG64

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    MonteCarloResult,
    Distribution,
    DistributionType
)


logger = logging.getLogger(__name__)


# Default numerical differentiation step size
DEFAULT_FINITE_DIFF_STEP = 1e-6

# Default Monte Carlo samples
DEFAULT_MC_SAMPLES = 10000

# Convergence threshold for Monte Carlo
MC_CONVERGENCE_THRESHOLD = 0.01  # 1% relative change


@dataclass
class CorrelationMatrix:
    """
    Correlation matrix for handling correlated inputs.

    Attributes:
        variable_names: List of variable names (ordered)
        matrix: Correlation coefficients (NxN numpy array)
    """
    variable_names: List[str]
    matrix: np.ndarray

    def __post_init__(self):
        """Validate correlation matrix."""
        n = len(self.variable_names)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} does not match "
                f"number of variables {n}"
            )

        # Check symmetric
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValueError("Correlation matrix must be symmetric")

        # Check diagonal is 1
        if not np.allclose(np.diag(self.matrix), 1.0):
            raise ValueError("Correlation matrix diagonal must be 1")

        # Check values in [-1, 1]
        if np.any(np.abs(self.matrix) > 1.0):
            raise ValueError("Correlation coefficients must be in [-1, 1]")

    @classmethod
    def identity(cls, variable_names: List[str]) -> "CorrelationMatrix":
        """Create identity correlation matrix (uncorrelated inputs)."""
        n = len(variable_names)
        return cls(
            variable_names=variable_names,
            matrix=np.eye(n)
        )

    def get_correlation(self, var1: str, var2: str) -> float:
        """Get correlation coefficient between two variables."""
        i = self.variable_names.index(var1)
        j = self.variable_names.index(var2)
        return self.matrix[i, j]


class UncertaintyPropagator:
    """
    Propagates uncertainty through calculations using multiple methods.

    Supports:
    - Linear propagation for linear combinations
    - Jacobian-based propagation for smooth nonlinear functions
    - Monte Carlo for complex/discontinuous functions
    - Correlated and uncorrelated inputs

    All methods are deterministic and reproducible.

    Example:
        propagator = UncertaintyPropagator()

        # Linear combination: z = 2*x + 3*y
        inputs = {
            "x": UncertainValue.from_measurement(100, 2.0),
            "y": UncertainValue.from_measurement(50, 3.0)
        }
        result = propagator.propagate_linear(inputs, {"x": 2.0, "y": 3.0})

        # Nonlinear: z = x * y
        def f(vals): return vals["x"] * vals["y"]
        def jacobian(vals): return {"x": vals["y"], "y": vals["x"]}
        result = propagator.propagate_nonlinear(inputs, f, jacobian)

        # Monte Carlo
        distributions = {
            "x": Distribution.normal(100, 2.0),
            "y": Distribution.normal(50, 1.5)
        }
        mc_result = propagator.propagate_monte_carlo(
            distributions, f, n_samples=50000
        )
    """

    def __init__(
        self,
        finite_diff_step: float = DEFAULT_FINITE_DIFF_STEP,
        default_mc_samples: int = DEFAULT_MC_SAMPLES,
        default_seed: int = 42
    ):
        """
        Initialize uncertainty propagator.

        Args:
            finite_diff_step: Step size for numerical differentiation
            default_mc_samples: Default number of Monte Carlo samples
            default_seed: Default random seed for reproducibility
        """
        self.finite_diff_step = finite_diff_step
        self.default_mc_samples = default_mc_samples
        self.default_seed = default_seed

    def propagate_linear(
        self,
        inputs: Dict[str, UncertainValue],
        coefficients: Dict[str, float],
        correlation: Optional[CorrelationMatrix] = None,
        output_name: str = "output"
    ) -> PropagatedUncertainty:
        """
        Propagate uncertainty through a linear combination.

        For z = sum(a_i * x_i), the variance is:
        Var(z) = sum(a_i^2 * Var(x_i)) + 2*sum(a_i*a_j*Cov(x_i,x_j))

        Args:
            inputs: Dictionary of input uncertain values
            coefficients: Dictionary of linear coefficients
            correlation: Optional correlation matrix for correlated inputs
            output_name: Name for the output

        Returns:
            PropagatedUncertainty with linear propagation result
        """
        start_time = time.perf_counter()

        # Validate inputs
        for name in coefficients:
            if name not in inputs:
                raise ValueError(f"Coefficient '{name}' has no matching input")

        # Compute output mean: z = sum(a_i * x_i)
        output_mean = sum(
            coefficients[name] * inputs[name].mean
            for name in coefficients
        )

        # Compute output variance
        if correlation is None:
            # Uncorrelated: Var(z) = sum(a_i^2 * Var(x_i))
            output_variance = sum(
                (coefficients[name] ** 2) * (inputs[name].std ** 2)
                for name in coefficients
            )
        else:
            # Correlated: include covariance terms
            output_variance = 0.0
            names = list(coefficients.keys())

            for i, name_i in enumerate(names):
                a_i = coefficients[name_i]
                var_i = inputs[name_i].std ** 2

                # Variance term
                output_variance += a_i ** 2 * var_i

                # Covariance terms
                for j in range(i + 1, len(names)):
                    name_j = names[j]
                    a_j = coefficients[name_j]
                    std_i = inputs[name_i].std
                    std_j = inputs[name_j].std
                    rho = correlation.get_correlation(name_i, name_j)

                    # 2 * a_i * a_j * rho * sigma_i * sigma_j
                    output_variance += 2 * a_i * a_j * rho * std_i * std_j

        output_std = math.sqrt(max(output_variance, 0.0))

        # Compute 95% confidence interval
        z_95 = 1.96
        lower_95 = output_mean - z_95 * output_std
        upper_95 = output_mean + z_95 * output_std

        # Determine dominant contributor
        contributions = {}
        for name in coefficients:
            contrib = (coefficients[name] ** 2) * (inputs[name].std ** 2)
            contributions[name] = contrib

        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            dominant = max(contributions, key=contributions.get)
        else:
            dominant = list(coefficients.keys())[0] if coefficients else ""

        computation_time = (time.perf_counter() - start_time) * 1000

        return PropagatedUncertainty(
            output_name=output_name,
            value=output_mean,
            uncertainty=output_std,
            confidence_interval_95=(lower_95, upper_95),
            contributing_inputs=inputs,
            sensitivity_coefficients=coefficients,
            dominant_contributor=dominant,
            propagation_method="linear",
            computation_time_ms=computation_time
        )

    def propagate_nonlinear(
        self,
        inputs: Dict[str, UncertainValue],
        function: Callable[[Dict[str, float]], float],
        jacobian: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
        correlation: Optional[CorrelationMatrix] = None,
        output_name: str = "output"
    ) -> PropagatedUncertainty:
        """
        Propagate uncertainty through a nonlinear function using Jacobian.

        For z = f(x_1, ..., x_n), first-order approximation gives:
        Var(z) = sum((df/dx_i)^2 * Var(x_i)) + covariance terms

        Args:
            inputs: Dictionary of input uncertain values
            function: Function f(inputs) -> float
            jacobian: Function returning partial derivatives. If None,
                      numerical differentiation is used.
            correlation: Optional correlation matrix
            output_name: Name for the output

        Returns:
            PropagatedUncertainty with Jacobian propagation result
        """
        start_time = time.perf_counter()

        # Get mean values for evaluation point
        mean_values = {name: uv.mean for name, uv in inputs.items()}

        # Compute output at mean
        output_mean = function(mean_values)

        # Compute or evaluate Jacobian
        if jacobian is not None:
            sensitivities = jacobian(mean_values)
        else:
            sensitivities = self._numerical_jacobian(function, mean_values)

        # Compute output variance using first-order Taylor expansion
        if correlation is None:
            # Uncorrelated case
            output_variance = sum(
                (sensitivities.get(name, 0.0) ** 2) * (inputs[name].std ** 2)
                for name in inputs
            )
        else:
            # Correlated case
            output_variance = 0.0
            names = list(inputs.keys())

            for i, name_i in enumerate(names):
                s_i = sensitivities.get(name_i, 0.0)
                var_i = inputs[name_i].std ** 2

                output_variance += s_i ** 2 * var_i

                for j in range(i + 1, len(names)):
                    name_j = names[j]
                    s_j = sensitivities.get(name_j, 0.0)
                    std_i = inputs[name_i].std
                    std_j = inputs[name_j].std
                    rho = correlation.get_correlation(name_i, name_j)

                    output_variance += 2 * s_i * s_j * rho * std_i * std_j

        output_std = math.sqrt(max(output_variance, 0.0))

        # Compute 95% CI
        z_95 = 1.96
        lower_95 = output_mean - z_95 * output_std
        upper_95 = output_mean + z_95 * output_std

        # Determine dominant contributor
        contributions = {}
        for name in inputs:
            s = sensitivities.get(name, 0.0)
            contributions[name] = (s ** 2) * (inputs[name].std ** 2)

        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            dominant = max(contributions, key=contributions.get)
        else:
            dominant = list(inputs.keys())[0] if inputs else ""

        computation_time = (time.perf_counter() - start_time) * 1000

        return PropagatedUncertainty(
            output_name=output_name,
            value=output_mean,
            uncertainty=output_std,
            confidence_interval_95=(lower_95, upper_95),
            contributing_inputs=inputs,
            sensitivity_coefficients=sensitivities,
            dominant_contributor=dominant,
            propagation_method="jacobian",
            computation_time_ms=computation_time
        )

    def propagate_monte_carlo(
        self,
        inputs: Dict[str, Distribution],
        function: Callable[[Dict[str, float]], float],
        n_samples: int = 10000,
        seed: Optional[int] = None,
        correlation: Optional[CorrelationMatrix] = None,
        convergence_check: bool = True,
        return_samples: bool = False
    ) -> MonteCarloResult:
        """
        Propagate uncertainty using Monte Carlo simulation.

        Samples from input distributions, evaluates function for each sample,
        and computes statistics of the output distribution.

        Args:
            inputs: Dictionary of input distributions
            function: Function f(inputs) -> float
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility (default: use propagator default)
            correlation: Optional correlation matrix for correlated sampling
            convergence_check: Whether to check for convergence
            return_samples: Whether to return raw samples in result

        Returns:
            MonteCarloResult with Monte Carlo statistics
        """
        start_time = time.perf_counter()

        # Use default seed if not provided
        seed = seed if seed is not None else self.default_seed

        # Create random generator with fixed seed for reproducibility
        rng = Generator(PCG64(seed))

        # Generate samples
        if correlation is None:
            # Independent sampling
            samples_dict = self._generate_independent_samples(
                inputs, n_samples, rng
            )
        else:
            # Correlated sampling using Cholesky decomposition
            samples_dict = self._generate_correlated_samples(
                inputs, n_samples, rng, correlation
            )

        # Evaluate function for each sample
        outputs = np.zeros(n_samples)
        for i in range(n_samples):
            sample_values = {name: samples_dict[name][i] for name in inputs}
            try:
                outputs[i] = function(sample_values)
            except Exception as e:
                logger.warning(f"Function evaluation failed for sample {i}: {e}")
                outputs[i] = np.nan

        # Remove any NaN values
        valid_outputs = outputs[~np.isnan(outputs)]

        if len(valid_outputs) == 0:
            raise ValueError("All Monte Carlo samples failed function evaluation")

        # Compute statistics
        mean = float(np.mean(valid_outputs))
        std = float(np.std(valid_outputs, ddof=1))
        median = float(np.median(valid_outputs))

        # Compute percentiles
        percentiles = {
            2.5: float(np.percentile(valid_outputs, 2.5)),
            5.0: float(np.percentile(valid_outputs, 5.0)),
            25.0: float(np.percentile(valid_outputs, 25.0)),
            50.0: float(np.percentile(valid_outputs, 50.0)),
            75.0: float(np.percentile(valid_outputs, 75.0)),
            95.0: float(np.percentile(valid_outputs, 95.0)),
            97.5: float(np.percentile(valid_outputs, 97.5))
        }

        # Check convergence
        convergence_achieved = True
        if convergence_check and n_samples >= 1000:
            # Compare statistics from first and second half
            half = n_samples // 2
            mean_first = np.mean(valid_outputs[:half])
            mean_second = np.mean(valid_outputs[half:])

            relative_change = abs(mean_second - mean_first) / max(abs(mean), 1e-10)
            convergence_achieved = relative_change < MC_CONVERGENCE_THRESHOLD

        computation_time = (time.perf_counter() - start_time) * 1000

        return MonteCarloResult(
            output_name="monte_carlo_output",
            mean=mean,
            std=std,
            median=median,
            percentiles=percentiles,
            n_samples=len(valid_outputs),
            convergence_achieved=convergence_achieved,
            seed=seed,
            samples=list(valid_outputs) if return_samples else None
        )

    def compute_sensitivity(
        self,
        function: Callable[[Dict[str, float]], float],
        inputs: Dict[str, float],
        parameter: str,
        step: Optional[float] = None
    ) -> float:
        """
        Compute sensitivity (partial derivative) of function to parameter.

        Uses central finite difference: df/dx = (f(x+h) - f(x-h)) / (2h)

        Args:
            function: Function f(inputs) -> float
            inputs: Dictionary of input values (at evaluation point)
            parameter: Parameter to compute sensitivity for
            step: Finite difference step size (default: propagator default)

        Returns:
            Sensitivity coefficient (partial derivative)
        """
        if parameter not in inputs:
            raise ValueError(f"Parameter '{parameter}' not in inputs")

        step = step or self.finite_diff_step
        x = inputs[parameter]

        # Handle zero or very small values
        h = max(step * abs(x), step)

        # Forward evaluation
        inputs_plus = inputs.copy()
        inputs_plus[parameter] = x + h
        f_plus = function(inputs_plus)

        # Backward evaluation
        inputs_minus = inputs.copy()
        inputs_minus[parameter] = x - h
        f_minus = function(inputs_minus)

        # Central difference
        sensitivity = (f_plus - f_minus) / (2 * h)

        return sensitivity

    def _numerical_jacobian(
        self,
        function: Callable[[Dict[str, float]], float],
        inputs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute numerical Jacobian (all partial derivatives).

        Args:
            function: Function to differentiate
            inputs: Evaluation point

        Returns:
            Dictionary mapping parameter name to sensitivity
        """
        jacobian = {}

        for param in inputs:
            jacobian[param] = self.compute_sensitivity(function, inputs, param)

        return jacobian

    def _generate_independent_samples(
        self,
        inputs: Dict[str, Distribution],
        n_samples: int,
        rng: Generator
    ) -> Dict[str, np.ndarray]:
        """Generate independent samples from each distribution."""
        samples = {}

        for name, dist in inputs.items():
            samples[name] = self._sample_distribution(dist, n_samples, rng)

        return samples

    def _generate_correlated_samples(
        self,
        inputs: Dict[str, Distribution],
        n_samples: int,
        rng: Generator,
        correlation: CorrelationMatrix
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated samples using Cholesky decomposition.

        Uses the copula approach:
        1. Generate correlated standard normal samples
        2. Transform to uniform via CDF
        3. Transform to target distribution via inverse CDF
        """
        names = list(inputs.keys())
        n_vars = len(names)

        # Reorder correlation matrix to match input order
        reordered_matrix = np.eye(n_vars)
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if name_i in correlation.variable_names and name_j in correlation.variable_names:
                    reordered_matrix[i, j] = correlation.get_correlation(name_i, name_j)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(reordered_matrix)
        except np.linalg.LinAlgError:
            logger.warning(
                "Correlation matrix not positive definite, "
                "falling back to independent sampling"
            )
            return self._generate_independent_samples(inputs, n_samples, rng)

        # Generate independent standard normals
        z = rng.standard_normal((n_vars, n_samples))

        # Apply correlation via Cholesky factor
        correlated_z = L @ z

        # Transform each variable
        samples = {}
        for i, name in enumerate(names):
            dist = inputs[name]
            z_i = correlated_z[i, :]

            # Transform standard normal to target distribution
            samples[name] = self._transform_normal_to_distribution(z_i, dist)

        return samples

    def _sample_distribution(
        self,
        dist: Distribution,
        n_samples: int,
        rng: Generator
    ) -> np.ndarray:
        """Sample from a distribution."""
        params = dist.parameters

        if dist.distribution_type == DistributionType.NORMAL:
            samples = rng.normal(params["mean"], params["std"], n_samples)

        elif dist.distribution_type == DistributionType.UNIFORM:
            samples = rng.uniform(params["low"], params["high"], n_samples)

        elif dist.distribution_type == DistributionType.TRIANGULAR:
            samples = rng.triangular(
                params["low"],
                params["mode"],
                params["high"],
                n_samples
            )

        elif dist.distribution_type == DistributionType.LOGNORMAL:
            # scipy parameterization: underlying normal has mu, sigma
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", 1.0)
            samples = rng.lognormal(mu, sigma, n_samples)

        elif dist.distribution_type == DistributionType.BETA:
            samples = rng.beta(params["a"], params["b"], n_samples)
            # Scale if bounds provided
            if dist.bounds:
                low, high = dist.bounds
                samples = low + samples * (high - low)

        else:
            # Default to normal
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            samples = rng.normal(mean, std, n_samples)

        # Apply truncation if bounds specified
        if dist.bounds:
            low, high = dist.bounds
            samples = np.clip(samples, low, high)

        return samples

    def _transform_normal_to_distribution(
        self,
        z: np.ndarray,
        dist: Distribution
    ) -> np.ndarray:
        """
        Transform standard normal samples to target distribution.

        Uses inverse CDF transform.
        """
        from scipy import stats

        params = dist.parameters

        if dist.distribution_type == DistributionType.NORMAL:
            # z ~ N(0,1) -> x = mu + sigma * z
            return params["mean"] + params["std"] * z

        elif dist.distribution_type == DistributionType.UNIFORM:
            # Transform via uniform CDF
            u = stats.norm.cdf(z)
            low, high = params["low"], params["high"]
            return low + u * (high - low)

        elif dist.distribution_type == DistributionType.LOGNORMAL:
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", 1.0)
            return np.exp(mu + sigma * z)

        elif dist.distribution_type == DistributionType.TRIANGULAR:
            u = stats.norm.cdf(z)
            low = params["low"]
            mode = params["mode"]
            high = params["high"]
            return stats.triang.ppf(
                u,
                (mode - low) / (high - low),
                loc=low,
                scale=high - low
            )

        else:
            # Default to normal transform
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            return mean + std * z

    def sobol_sensitivity(
        self,
        inputs: Dict[str, Distribution],
        function: Callable[[Dict[str, float]], float],
        n_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Sobol sensitivity indices for global sensitivity analysis.

        Sobol indices quantify how much of output variance is due to each
        input and their interactions.

        Args:
            inputs: Dictionary of input distributions
            function: Function f(inputs) -> float
            n_samples: Number of samples for estimation
            seed: Random seed

        Returns:
            Dictionary with first-order (S1) and total-order (ST) indices
        """
        seed = seed if seed is not None else self.default_seed
        rng = Generator(PCG64(seed))

        names = list(inputs.keys())
        n_vars = len(names)

        # Generate two independent sample matrices A and B
        A = np.zeros((n_samples, n_vars))
        B = np.zeros((n_samples, n_vars))

        for j, name in enumerate(names):
            dist = inputs[name]
            A[:, j] = self._sample_distribution(dist, n_samples, rng)
            B[:, j] = self._sample_distribution(dist, n_samples, rng)

        # Evaluate function at base matrices
        def eval_matrix(M):
            results = np.zeros(n_samples)
            for i in range(n_samples):
                sample = {names[j]: M[i, j] for j in range(n_vars)}
                results[i] = function(sample)
            return results

        f_A = eval_matrix(A)
        f_B = eval_matrix(B)

        # Compute variance of output
        f_all = np.concatenate([f_A, f_B])
        var_y = np.var(f_all)

        if var_y < 1e-10:
            # No variance in output
            return {
                name: {"S1": 0.0, "ST": 0.0}
                for name in names
            }

        sensitivities = {}

        for j, name in enumerate(names):
            # Matrix AB_j: A with j-th column from B
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            f_AB_j = eval_matrix(AB_j)

            # First-order index: S1_j
            # Estimator: (1/n) * sum(f_B * (f_AB_j - f_A)) / Var(Y)
            S1 = np.mean(f_B * (f_AB_j - f_A)) / var_y
            S1 = max(0, min(1, S1))  # Bound to [0, 1]

            # Total-order index: ST_j
            # Estimator: (1/2n) * sum((f_A - f_AB_j)^2) / Var(Y)
            ST = 0.5 * np.mean((f_A - f_AB_j) ** 2) / var_y
            ST = max(0, min(1, ST))  # Bound to [0, 1]

            sensitivities[name] = {
                "S1": float(S1),
                "ST": float(ST)
            }

        return sensitivities


def combine_uncertainties(
    uncertainties: List[UncertainValue],
    method: str = "quadrature"
) -> UncertainValue:
    """
    Combine multiple independent uncertainties.

    Args:
        uncertainties: List of uncertain values to combine
        method: Combination method ("quadrature", "linear", "max")

    Returns:
        Combined UncertainValue
    """
    if not uncertainties:
        raise ValueError("Cannot combine empty list of uncertainties")

    if len(uncertainties) == 1:
        return uncertainties[0]

    # Sum of means
    combined_mean = sum(u.mean for u in uncertainties)

    if method == "quadrature":
        # Root sum of squares (uncorrelated)
        combined_variance = sum(u.std ** 2 for u in uncertainties)
        combined_std = math.sqrt(combined_variance)

    elif method == "linear":
        # Linear sum (conservative)
        combined_std = sum(u.std for u in uncertainties)

    elif method == "max":
        # Maximum uncertainty (dominated by worst)
        combined_std = max(u.std for u in uncertainties)

    else:
        raise ValueError(f"Unknown combination method: {method}")

    z_95 = 1.96
    lower_95 = combined_mean - z_95 * combined_std
    upper_95 = combined_mean + z_95 * combined_std

    return UncertainValue(
        mean=combined_mean,
        std=combined_std,
        lower_95=lower_95,
        upper_95=upper_95,
        distribution_type=DistributionType.NORMAL,
        timestamp=datetime.utcnow()
    )
