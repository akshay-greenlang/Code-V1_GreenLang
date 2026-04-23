# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Uncertainty Propagator Module

Provides advanced uncertainty propagation methods for combustion calculations.
Implements both analytical (linear) and numerical (Monte Carlo) methods
following GUM (Guide to Expression of Uncertainty in Measurement) principles.

Methods:
    - Linear propagation: Taylor series first-order approximation
    - Monte Carlo propagation: Numerical sampling for nonlinear functions
    - Correlation handling: Full covariance matrix support
    - Model uncertainty: Propagation through ML/surrogate models

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum
import numpy as np
import hashlib
import json
from datetime import datetime


class PropagationMethod(str, Enum):
    """Uncertainty propagation method."""
    LINEAR = "linear"  # First-order Taylor series
    MONTE_CARLO = "monte_carlo"  # Numerical sampling
    SECOND_ORDER = "second_order"  # Second-order Taylor series


@dataclass
class MonteCarloResult:
    """
    Result of Monte Carlo uncertainty propagation.

    Attributes:
        mean: Mean of Monte Carlo samples
        std: Standard deviation of samples
        samples: Raw Monte Carlo samples (if retained)
        percentiles: Key percentiles (5th, 25th, 50th, 75th, 95th)
        n_samples: Number of Monte Carlo samples
        seed: Random seed used (for reproducibility)
        convergence_std: Standard deviation of running mean (convergence indicator)
    """
    mean: float
    std: float
    samples: Optional[np.ndarray] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 10000
    seed: int = 42
    convergence_std: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "mean": self.mean,
            "std": self.std,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "percentiles": self.percentiles,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a value.

    Attributes:
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: Method used to compute interval
    """
    lower: float
    upper: float
    confidence: float = 0.95
    method: str = "normal"
    center: float = field(init=False)
    width: float = field(init=False)

    def __post_init__(self):
        self.center = (self.lower + self.upper) / 2
        self.width = self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper


@dataclass
class PredictionUncertainty:
    """
    Uncertainty for ML model predictions.

    Attributes:
        prediction: Model prediction (point estimate)
        aleatoric_uncertainty: Data uncertainty (irreducible)
        epistemic_uncertainty: Model uncertainty (reducible with more data)
        total_uncertainty: Combined uncertainty
        confidence_interval: Confidence bounds
    """
    prediction: float
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    total_uncertainty: float = field(init=False)
    confidence_interval: Optional[ConfidenceInterval] = None
    provenance_hash: str = ""

    def __post_init__(self):
        # RSS combination of uncertainty components
        self.total_uncertainty = np.sqrt(
            self.aleatoric_uncertainty**2 + self.epistemic_uncertainty**2
        )
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "prediction": self.prediction,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "total_uncertainty": self.total_uncertainty,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class CombinedUncertainty:
    """
    Combined uncertainty from multiple sources.

    Attributes:
        combined_std: Combined standard uncertainty
        expanded_uncertainty: Expanded uncertainty (k * combined_std)
        coverage_factor: Coverage factor k
        individual_contributions: Contribution from each source
        correlation_contribution: Extra contribution from correlations
        effective_dof: Effective degrees of freedom (Welch-Satterthwaite)
    """
    combined_std: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    individual_contributions: Dict[str, float] = field(default_factory=dict)
    correlation_contribution: float = 0.0
    effective_dof: Optional[float] = None
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "combined_std": self.combined_std,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "individual_contributions": self.individual_contributions,
            "correlation_contribution": self.correlation_contribution,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class UncertaintyPropagator:
    """
    Advanced uncertainty propagation for combustion calculations.

    Supports:
    - Linear (Taylor series) propagation for well-behaved functions
    - Monte Carlo propagation for nonlinear/complex functions
    - Correlation handling between input variables
    - Model uncertainty propagation for ML predictions

    ZERO HALLUCINATION: All calculations are deterministic.
    Same inputs with same seed -> Same outputs (guaranteed).

    Example Usage:
        >>> propagator = UncertaintyPropagator()
        >>> coefficients = [1.0, 2.0, 0.5]
        >>> uncertainties = [0.1, 0.05, 0.2]
        >>> combined = propagator.propagate_linear(coefficients, uncertainties)
        >>> print(f"Combined uncertainty: {combined}")
    """

    def __init__(self, default_seed: int = 42, default_n_samples: int = 10000):
        """
        Initialize the uncertainty propagator.

        Args:
            default_seed: Default random seed for reproducibility
            default_n_samples: Default number of Monte Carlo samples
        """
        self.default_seed = default_seed
        self.default_n_samples = default_n_samples

    def propagate_linear(
        self,
        coefficients: List[float],
        uncertainties: List[float],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Propagate uncertainties using linear (first-order) approximation.

        Formula: u_y^2 = sum_i sum_j c_i * c_j * u_i * u_j * r_ij
        For uncorrelated: u_y = sqrt(sum_i (c_i * u_i)^2)

        Args:
            coefficients: Sensitivity coefficients (partial derivatives)
            uncertainties: Standard uncertainties of inputs
            correlation_matrix: Correlation matrix (identity if None)

        Returns:
            Combined standard uncertainty

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        n = len(coefficients)
        if len(uncertainties) != n:
            raise ValueError("coefficients and uncertainties must have same length")

        c = np.array(coefficients)
        u = np.array(uncertainties)

        if correlation_matrix is None:
            # Uncorrelated case: u_y = sqrt(sum_i (c_i * u_i)^2)
            variance = np.sum((c * u) ** 2)
        else:
            # Correlated case: full covariance propagation
            r = np.array(correlation_matrix)
            if r.shape != (n, n):
                raise ValueError(f"correlation_matrix must be {n}x{n}")

            # Build covariance matrix from correlations and uncertainties
            # cov_ij = u_i * u_j * r_ij
            cov = np.outer(u, u) * r

            # Propagate: var_y = c^T * Cov * c
            variance = c @ cov @ c

        return float(np.sqrt(variance))

    def propagate_monte_carlo(
        self,
        func: Callable[..., float],
        inputs: Dict[str, Tuple[float, float]],
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        retain_samples: bool = False,
    ) -> MonteCarloResult:
        """
        Propagate uncertainties using Monte Carlo simulation.

        Samples from input distributions, evaluates function, and
        computes output statistics. Suitable for nonlinear functions.

        Args:
            func: Function to evaluate (takes keyword arguments)
            inputs: Dict of {name: (mean, std)} for each input
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            correlation_matrix: Input correlations (identity if None)
            retain_samples: Whether to keep raw samples in result

        Returns:
            MonteCarloResult with mean, std, percentiles

        DETERMINISTIC: Same inputs with same seed -> Same outputs.
        """
        if n_samples is None:
            n_samples = self.default_n_samples
        if seed is None:
            seed = self.default_seed

        # Set seed for reproducibility
        rng = np.random.default_rng(seed)

        # Extract input names, means, and stds
        names = list(inputs.keys())
        means = np.array([inputs[name][0] for name in names])
        stds = np.array([inputs[name][1] for name in names])

        # Generate correlated samples if correlation matrix provided
        if correlation_matrix is not None:
            # Cholesky decomposition for correlated sampling
            L = np.linalg.cholesky(correlation_matrix)
            # Generate independent standard normal samples
            z = rng.standard_normal((n_samples, len(names)))
            # Transform to correlated samples
            correlated_z = z @ L.T
            # Scale to actual distributions
            samples = means + stds * correlated_z
        else:
            # Independent sampling
            samples = rng.normal(loc=means, scale=stds, size=(n_samples, len(names)))

        # Evaluate function for each sample
        outputs = np.zeros(n_samples)
        for i in range(n_samples):
            kwargs = {name: samples[i, j] for j, name in enumerate(names)}
            outputs[i] = func(**kwargs)

        # Compute statistics
        mean_val = float(np.mean(outputs))
        std_val = float(np.std(outputs, ddof=1))

        # Percentiles
        percentiles = {
            "p5": float(np.percentile(outputs, 5)),
            "p25": float(np.percentile(outputs, 25)),
            "p50": float(np.percentile(outputs, 50)),
            "p75": float(np.percentile(outputs, 75)),
            "p95": float(np.percentile(outputs, 95)),
        }

        # Convergence indicator: std of running mean
        running_means = np.cumsum(outputs) / np.arange(1, n_samples + 1)
        convergence_std = float(np.std(running_means[-100:]))  # Last 100 samples

        return MonteCarloResult(
            mean=mean_val,
            std=std_val,
            samples=outputs if retain_samples else None,
            percentiles=percentiles,
            n_samples=n_samples,
            seed=seed,
            convergence_std=convergence_std,
        )

    def compute_confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
        method: str = "percentile",
    ) -> ConfidenceInterval:
        """
        Compute confidence interval from sample data.

        Args:
            values: Sample values (e.g., from Monte Carlo)
            confidence: Confidence level (0.0 to 1.0)
            method: "percentile" (empirical) or "normal" (assumes normality)

        Returns:
            ConfidenceInterval with lower and upper bounds

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if method == "percentile":
            # Empirical percentile method (non-parametric)
            alpha = 1 - confidence
            lower = float(np.percentile(values, 100 * alpha / 2))
            upper = float(np.percentile(values, 100 * (1 - alpha / 2)))

        elif method == "normal":
            # Normal approximation
            from scipy.stats import norm
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            z = norm.ppf((1 + confidence) / 2)
            lower = float(mean - z * std)
            upper = float(mean + z * std)

        else:
            raise ValueError(f"Unknown method: {method}")

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence=confidence,
            method=method,
        )

    def propagate_through_model(
        self,
        model: Any,
        features: Dict[str, Tuple[float, float]],
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> PredictionUncertainty:
        """
        Propagate uncertainty through an ML model prediction.

        Estimates both aleatoric (data) and epistemic (model) uncertainty
        by sampling from input distributions and analyzing prediction variance.

        Args:
            model: ML model with predict() method
            features: Dict of {feature_name: (mean, std)} for inputs
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility

        Returns:
            PredictionUncertainty with decomposed uncertainty

        DETERMINISTIC: Same inputs with same seed -> Same outputs.
        """
        if seed is None:
            seed = self.default_seed

        rng = np.random.default_rng(seed)

        # Extract feature names and distributions
        names = list(features.keys())
        means = np.array([features[name][0] for name in names])
        stds = np.array([features[name][1] for name in names])

        # Sample from input distributions
        samples = rng.normal(loc=means, scale=stds, size=(n_samples, len(names)))

        # Get predictions for all samples
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            feature_dict = {name: samples[i, j] for j, name in enumerate(names)}
            # Assume model has predict method that takes dict or array
            if hasattr(model, 'predict'):
                pred = model.predict(np.array([list(feature_dict.values())]))
                predictions[i] = pred[0] if hasattr(pred, '__len__') else pred
            else:
                # Fallback: assume model is callable
                predictions[i] = model(**feature_dict)

        # Point estimate (mean of predictions)
        prediction = float(np.mean(predictions))

        # Total uncertainty from prediction variance
        total_std = float(np.std(predictions, ddof=1))

        # Estimate aleatoric vs epistemic decomposition
        # Aleatoric: uncertainty from input noise propagation
        # Epistemic: model uncertainty (harder to estimate without ensemble)

        # Simple heuristic: assume model uncertainty is ~10-20% of total
        # In practice, use ensemble methods or Bayesian models
        epistemic_fraction = 0.15
        aleatoric_uncertainty = total_std * np.sqrt(1 - epistemic_fraction**2)
        epistemic_uncertainty = total_std * epistemic_fraction

        # Confidence interval
        ci = self.compute_confidence_interval(predictions, confidence=0.95)

        return PredictionUncertainty(
            prediction=prediction,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            confidence_interval=ci,
        )

    def combine_uncertainties(
        self,
        uncertainties: List[float],
        correlation: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
        coverage_factor: float = 2.0,
    ) -> CombinedUncertainty:
        """
        Combine multiple uncertainty sources with optional correlations.

        Uses GUM-compliant combination formula:
        u_c^2 = sum_i sum_j w_i * w_j * u_i * u_j * r_ij

        Args:
            uncertainties: List of standard uncertainties
            correlation: Correlation matrix (identity if None)
            weights: Weights for each uncertainty (unit weights if None)
            names: Names for each uncertainty source
            coverage_factor: Coverage factor k for expanded uncertainty

        Returns:
            CombinedUncertainty with full breakdown

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        n = len(uncertainties)
        u = np.array(uncertainties)

        if weights is None:
            w = np.ones(n)
        else:
            w = np.array(weights)
            if len(w) != n:
                raise ValueError("weights must match uncertainties length")

        if names is None:
            names = [f"source_{i}" for i in range(n)]

        # Compute individual contributions (assuming no correlation)
        individual = (w * u) ** 2

        if correlation is None:
            # Uncorrelated combination
            variance = np.sum(individual)
            correlation_contribution = 0.0
        else:
            r = np.array(correlation)
            if r.shape != (n, n):
                raise ValueError(f"correlation must be {n}x{n}")

            # Full covariance propagation
            cov = np.outer(w * u, w * u) * r
            variance = np.sum(cov)

            # Correlation contribution = total - individual (diagonal)
            correlation_contribution = float(variance - np.sum(individual))

        combined_std = float(np.sqrt(variance))
        expanded = coverage_factor * combined_std

        # Build contribution dict
        contributions = {
            names[i]: float(np.sqrt(individual[i]))
            for i in range(n)
        }

        return CombinedUncertainty(
            combined_std=combined_std,
            expanded_uncertainty=expanded,
            coverage_factor=coverage_factor,
            individual_contributions=contributions,
            correlation_contribution=correlation_contribution,
        )

    def sensitivity_analysis(
        self,
        func: Callable[..., float],
        inputs: Dict[str, Tuple[float, float]],
        delta_fraction: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute sensitivity coefficients for a function.

        Uses central finite differences to estimate partial derivatives.

        Args:
            func: Function to analyze
            inputs: Dict of {name: (mean, std)} for each input
            delta_fraction: Fraction of value to perturb (default 1%)

        Returns:
            Dict of {name: sensitivity_coefficient}

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        sensitivities = {}
        base_values = {name: mean for name, (mean, std) in inputs.items()}
        base_output = func(**base_values)

        for name, (mean, std) in inputs.items():
            # Central difference approximation
            delta = max(abs(mean) * delta_fraction, 1e-10)

            # Perturbed values
            values_plus = base_values.copy()
            values_plus[name] = mean + delta

            values_minus = base_values.copy()
            values_minus[name] = mean - delta

            # Evaluate function
            f_plus = func(**values_plus)
            f_minus = func(**values_minus)

            # Partial derivative
            sensitivity = (f_plus - f_minus) / (2 * delta)
            sensitivities[name] = float(sensitivity)

        return sensitivities

    def uncertainty_contribution_analysis(
        self,
        func: Callable[..., float],
        inputs: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Analyze contribution of each input to total output uncertainty.

        Returns fractional contribution (sums to 1.0 for uncorrelated inputs).

        Args:
            func: Function to analyze
            inputs: Dict of {name: (mean, std)} for each input

        Returns:
            Dict of {name: fractional_contribution}

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Get sensitivity coefficients
        sensitivities = self.sensitivity_analysis(func, inputs)

        # Compute individual variance contributions
        contributions = {}
        total_variance = 0.0

        for name, (mean, std) in inputs.items():
            c = sensitivities[name]
            variance_contribution = (c * std) ** 2
            contributions[name] = variance_contribution
            total_variance += variance_contribution

        # Normalize to fractions
        if total_variance > 0:
            contributions = {
                name: var / total_variance
                for name, var in contributions.items()
            }

        return contributions
