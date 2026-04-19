# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Weibull Analysis Module

This module implements Weibull distribution analysis for equipment Remaining
Useful Life (RUL) estimation. Weibull analysis is the industry standard for
reliability engineering and maintenance optimization.

The Weibull distribution is characterized by:
- Beta (shape parameter): Indicates failure pattern
  - Beta < 1: Infant mortality (decreasing failure rate)
  - Beta = 1: Random failures (constant failure rate, exponential)
  - Beta > 1: Wear-out failures (increasing failure rate)
- Eta (scale parameter): Characteristic life (63.2% failure point)
- Gamma (location parameter): Failure-free period

This module provides ZERO-HALLUCINATION calculations using deterministic
statistical formulas with full provenance tracking.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.weibull import (
    ...     WeibullAnalyzer
    ... )
    >>> analyzer = WeibullAnalyzer(config)
    >>> result = analyzer.analyze(failure_times, current_age=5000)
    >>> print(f"RUL P50: {result.rul_p50_hours:.0f} hours")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    WeibullConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    WeibullAnalysisResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FailureData:
    """Container for failure/survival data."""
    time: float  # Time to failure or censoring
    is_failure: bool = True  # True if failure, False if censored (still running)
    failure_mode: Optional[str] = None  # Failure mode if known


@dataclass
class WeibullParameters:
    """Weibull distribution parameters."""
    beta: float  # Shape parameter
    eta: float  # Scale parameter (characteristic life)
    gamma: float = 0.0  # Location parameter (failure-free life)
    r_squared: Optional[float] = None
    method: str = "mle"


# =============================================================================
# WEIBULL ANALYZER CLASS
# =============================================================================

class WeibullAnalyzer:
    """
    Weibull Distribution Analyzer for RUL estimation.

    This class implements Weibull analysis using both Maximum Likelihood
    Estimation (MLE) and Rank Regression methods. It provides P10, P50, P90
    confidence intervals for remaining useful life predictions.

    All calculations are DETERMINISTIC with full provenance tracking.
    No ML/LLM in calculation path - ZERO HALLUCINATION.

    Attributes:
        config: Weibull analysis configuration
        _cached_parameters: Cached Weibull parameters

    Example:
        >>> config = WeibullConfig(method="mle", confidence_level=0.90)
        >>> analyzer = WeibullAnalyzer(config)
        >>> result = analyzer.analyze(
        ...     failure_data=[5000, 6200, 4800, 7100],
        ...     current_age=3500
        ... )
        >>> print(f"Beta: {result.beta:.2f}, Eta: {result.eta_hours:.0f}")
    """

    def __init__(self, config: Optional[WeibullConfig] = None) -> None:
        """
        Initialize Weibull analyzer.

        Args:
            config: Weibull analysis configuration
        """
        self.config = config or WeibullConfig()
        self._cached_parameters: Optional[WeibullParameters] = None

        logger.info(
            f"WeibullAnalyzer initialized: method={self.config.method}, "
            f"confidence={self.config.confidence_level}"
        )

    def analyze(
        self,
        failure_data: List[FailureData],
        current_age: float,
        historical_parameters: Optional[WeibullParameters] = None,
    ) -> WeibullAnalysisResult:
        """
        Perform Weibull analysis and estimate RUL.

        Args:
            failure_data: List of failure/survival data points
            current_age: Current equipment age in hours
            historical_parameters: Optional historical Weibull parameters

        Returns:
            WeibullAnalysisResult with parameters and RUL estimates

        Raises:
            ValueError: If insufficient data for analysis
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Starting Weibull analysis: {len(failure_data)} data points, "
            f"current_age={current_age}"
        )

        # Count failures and censored observations
        failures = [d for d in failure_data if d.is_failure]
        censored = [d for d in failure_data if not d.is_failure]
        n_failures = len(failures)
        n_censored = len(censored)

        # Check minimum data requirements
        if n_failures < self.config.minimum_failures:
            logger.warning(
                f"Insufficient failures ({n_failures}) for analysis. "
                f"Using historical parameters if available."
            )
            if historical_parameters:
                parameters = historical_parameters
            else:
                # Use industry default for rotating equipment
                parameters = WeibullParameters(
                    beta=2.5,  # Typical wear-out
                    eta=50000.0,  # 50,000 hours typical life
                    method="default"
                )
        else:
            # Estimate parameters
            if self.config.method == "mle":
                parameters = self._estimate_mle(failure_data)
            elif self.config.method == "rank_regression":
                parameters = self._estimate_rank_regression(failure_data)
            else:
                parameters = self._estimate_median_ranks(failure_data)

        # Calculate confidence bounds for parameters
        beta_ci, eta_ci = self._calculate_parameter_confidence(
            parameters,
            n_failures,
            self.config.confidence_level
        )

        # Calculate current failure probability
        current_cdf = self._cdf(current_age, parameters)

        # Calculate RUL at different probability levels
        rul_p10 = self._inverse_cdf(0.10, parameters) - current_age
        rul_p50 = self._inverse_cdf(0.50, parameters) - current_age
        rul_p90 = self._inverse_cdf(0.90, parameters) - current_age

        # Ensure non-negative RUL
        rul_p10 = max(0, rul_p10)
        rul_p50 = max(0, rul_p50)
        rul_p90 = max(0, rul_p90)

        # Calculate conditional failure probability for next 30 days (720 hours)
        future_age = current_age + 720
        cond_prob_30d = self._conditional_failure_probability(
            current_age,
            future_age,
            parameters
        )

        # Interpret beta value
        interpretation = self._interpret_beta(parameters.beta)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            failure_data,
            current_age,
            parameters
        )

        processing_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Weibull analysis complete: beta={parameters.beta:.2f}, "
            f"eta={parameters.eta:.0f}, RUL_P50={rul_p50:.0f}h"
        )

        return WeibullAnalysisResult(
            beta=parameters.beta,
            eta_hours=parameters.eta,
            gamma_hours=parameters.gamma,
            rul_p10_hours=rul_p10,
            rul_p50_hours=rul_p50,
            rul_p90_hours=rul_p90,
            current_age_hours=current_age,
            current_failure_probability=current_cdf,
            conditional_failure_probability_30d=cond_prob_30d,
            beta_ci_lower=beta_ci[0],
            beta_ci_upper=beta_ci[1],
            eta_ci_lower_hours=eta_ci[0],
            eta_ci_upper_hours=eta_ci[1],
            confidence_level=self.config.confidence_level,
            failure_mode_interpretation=interpretation,
            r_squared=parameters.r_squared,
            n_failures=n_failures,
            n_censored=n_censored,
            provenance_hash=provenance_hash,
        )

    def analyze_from_times(
        self,
        failure_times: List[float],
        censored_times: Optional[List[float]] = None,
        current_age: float = 0,
    ) -> WeibullAnalysisResult:
        """
        Convenience method to analyze from simple time lists.

        Args:
            failure_times: List of times to failure (hours)
            censored_times: List of censored (survival) times
            current_age: Current equipment age

        Returns:
            WeibullAnalysisResult
        """
        failure_data = []

        for t in failure_times:
            failure_data.append(FailureData(time=t, is_failure=True))

        if censored_times:
            for t in censored_times:
                failure_data.append(FailureData(time=t, is_failure=False))

        return self.analyze(failure_data, current_age)

    def _estimate_mle(
        self,
        data: List[FailureData]
    ) -> WeibullParameters:
        """
        Estimate Weibull parameters using Maximum Likelihood Estimation.

        MLE provides the most statistically efficient estimates when
        sample sizes are reasonable (>10 failures).

        Args:
            data: Failure data

        Returns:
            WeibullParameters
        """
        times = [d.time for d in data]
        is_failure = [d.is_failure for d in data]
        n = len(times)
        r = sum(is_failure)  # Number of failures

        if r == 0:
            raise ValueError("No failures in dataset for MLE")

        # Newton-Raphson iteration for beta
        beta = 2.0  # Initial guess
        max_iterations = 100
        tolerance = 1e-6

        for _ in range(max_iterations):
            # Calculate terms for Newton-Raphson
            sum_t_beta = sum(t ** beta for t in times)
            sum_t_beta_ln = sum((t ** beta) * math.log(t) for t in times if t > 0)
            sum_ln_f = sum(
                math.log(t) for t, f in zip(times, is_failure)
                if f and t > 0
            )

            if sum_t_beta <= 0:
                break

            # First derivative of log-likelihood w.r.t. beta
            f_beta = (
                r / beta +
                sum_ln_f -
                (r / sum_t_beta) * sum_t_beta_ln
            )

            # Second derivative (for Newton-Raphson)
            sum_t_beta_ln2 = sum(
                (t ** beta) * (math.log(t) ** 2) for t in times if t > 0
            )

            f_prime = (
                -r / (beta ** 2) -
                (r / sum_t_beta) * sum_t_beta_ln2 +
                (r / (sum_t_beta ** 2)) * (sum_t_beta_ln ** 2)
            )

            if abs(f_prime) < 1e-10:
                break

            # Newton-Raphson update
            beta_new = beta - f_beta / f_prime

            if beta_new <= 0:
                beta_new = beta / 2  # Ensure positive

            if abs(beta_new - beta) < tolerance:
                beta = beta_new
                break

            beta = beta_new

        # Estimate eta given beta
        eta = (sum(t ** beta for t in times) / r) ** (1 / beta)

        # Calculate R-squared approximation
        r_squared = self._calculate_r_squared(data, beta, eta)

        return WeibullParameters(
            beta=beta,
            eta=eta,
            gamma=0.0,
            r_squared=r_squared,
            method="mle"
        )

    def _estimate_rank_regression(
        self,
        data: List[FailureData]
    ) -> WeibullParameters:
        """
        Estimate Weibull parameters using Rank Regression (Least Squares).

        Rank regression is more robust with small sample sizes but
        less efficient than MLE for larger samples.

        Args:
            data: Failure data

        Returns:
            WeibullParameters
        """
        # Filter to failures only and sort
        failures = sorted([d.time for d in data if d.is_failure])
        n = len(failures)

        if n < 2:
            raise ValueError("Need at least 2 failures for rank regression")

        # Calculate median ranks using Bernard's approximation
        # F(i) = (i - 0.3) / (n + 0.4)
        x = []  # ln(t)
        y = []  # ln(ln(1/(1-F)))

        for i, t in enumerate(failures, 1):
            if t <= 0:
                continue
            rank = (i - 0.3) / (n + 0.4)
            if rank <= 0 or rank >= 1:
                continue

            x.append(math.log(t))
            y.append(math.log(math.log(1 / (1 - rank))))

        if len(x) < 2:
            raise ValueError("Insufficient valid data points")

        # Linear regression: y = beta * x - beta * ln(eta)
        n_pts = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        # Calculate slope (beta) and intercept
        denominator = n_pts * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            raise ValueError("Singular matrix in regression")

        beta = (n_pts * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - beta * sum_x) / n_pts

        # Solve for eta: intercept = -beta * ln(eta)
        eta = math.exp(-intercept / beta) if beta > 0 else 50000

        # Calculate R-squared
        y_mean = sum_y / n_pts
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum(
            (yi - (beta * xi + intercept)) ** 2
            for xi, yi in zip(x, y)
        )
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return WeibullParameters(
            beta=max(0.1, beta),  # Ensure positive
            eta=max(1, eta),
            gamma=0.0,
            r_squared=r_squared,
            method="rank_regression"
        )

    def _estimate_median_ranks(
        self,
        data: List[FailureData]
    ) -> WeibullParameters:
        """
        Estimate parameters using median rank regression.

        This method is similar to rank regression but uses median ranks
        which can be more robust for small samples.

        Args:
            data: Failure data

        Returns:
            WeibullParameters
        """
        # For simplicity, delegate to rank regression with Bernard's approximation
        # which approximates median ranks
        return self._estimate_rank_regression(data)

    def _cdf(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate Weibull cumulative distribution function (failure probability).

        F(t) = 1 - exp(-((t - gamma) / eta)^beta)

        Args:
            t: Time
            params: Weibull parameters

        Returns:
            Cumulative failure probability
        """
        if t <= params.gamma:
            return 0.0

        t_adj = (t - params.gamma) / params.eta
        return 1 - math.exp(-(t_adj ** params.beta))

    def _pdf(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate Weibull probability density function.

        f(t) = (beta/eta) * ((t-gamma)/eta)^(beta-1) * exp(-((t-gamma)/eta)^beta)

        Args:
            t: Time
            params: Weibull parameters

        Returns:
            Probability density
        """
        if t <= params.gamma:
            return 0.0

        t_adj = (t - params.gamma) / params.eta
        term1 = (params.beta / params.eta) * (t_adj ** (params.beta - 1))
        term2 = math.exp(-(t_adj ** params.beta))
        return term1 * term2

    def _inverse_cdf(
        self,
        p: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate inverse CDF (quantile function) for Weibull.

        t = gamma + eta * (-ln(1-p))^(1/beta)

        Args:
            p: Probability (0 < p < 1)
            params: Weibull parameters

        Returns:
            Time at which probability p is reached
        """
        if p <= 0:
            return params.gamma
        if p >= 1:
            return float('inf')

        return params.gamma + params.eta * ((-math.log(1 - p)) ** (1 / params.beta))

    def _hazard_rate(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate instantaneous hazard rate (failure rate).

        h(t) = (beta/eta) * ((t-gamma)/eta)^(beta-1)

        Args:
            t: Time
            params: Weibull parameters

        Returns:
            Hazard rate
        """
        if t <= params.gamma:
            return 0.0

        t_adj = (t - params.gamma) / params.eta
        return (params.beta / params.eta) * (t_adj ** (params.beta - 1))

    def _conditional_failure_probability(
        self,
        t_current: float,
        t_future: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate conditional probability of failure.

        P(T <= t_future | T > t_current) = 1 - R(t_future)/R(t_current)

        Args:
            t_current: Current age
            t_future: Future time point
            params: Weibull parameters

        Returns:
            Conditional failure probability
        """
        if t_future <= t_current:
            return 0.0

        r_current = 1 - self._cdf(t_current, params)
        r_future = 1 - self._cdf(t_future, params)

        if r_current <= 0:
            return 1.0

        return 1 - (r_future / r_current)

    def _calculate_parameter_confidence(
        self,
        params: WeibullParameters,
        n_failures: int,
        confidence: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate confidence intervals for Weibull parameters.

        Uses approximate Fisher information matrix approach.

        Args:
            params: Estimated parameters
            n_failures: Number of failures
            confidence: Confidence level

        Returns:
            Tuple of (beta_ci, eta_ci) confidence intervals
        """
        # Z-score for confidence level
        alpha = 1 - confidence
        # Approximation for large sample normal quantile
        z = 1.645 if confidence == 0.90 else 1.96 if confidence == 0.95 else 2.576

        # Approximate standard errors (simplified)
        # For MLE, SE(beta) ~ beta / sqrt(n), SE(eta) ~ eta / sqrt(n)
        if n_failures > 0:
            se_beta = params.beta / math.sqrt(n_failures)
            se_eta = params.eta / math.sqrt(n_failures)
        else:
            se_beta = params.beta * 0.3
            se_eta = params.eta * 0.3

        beta_lower = max(0.1, params.beta - z * se_beta)
        beta_upper = params.beta + z * se_beta

        eta_lower = max(1, params.eta - z * se_eta)
        eta_upper = params.eta + z * se_eta

        return (beta_lower, beta_upper), (eta_lower, eta_upper)

    def _interpret_beta(self, beta: float) -> str:
        """
        Provide interpretation of Weibull shape parameter.

        Args:
            beta: Shape parameter

        Returns:
            Human-readable interpretation
        """
        if beta < 0.5:
            return (
                "Very early life failures (infant mortality). "
                "Check for manufacturing defects, installation issues, "
                "or break-in period problems."
            )
        elif beta < 1.0:
            return (
                "Decreasing failure rate (infant mortality pattern). "
                "Equipment likely suffering from early life issues. "
                "Consider burn-in testing or installation review."
            )
        elif abs(beta - 1.0) < 0.1:
            return (
                "Constant failure rate (random failures). "
                "Failures are independent of age. "
                "Condition monitoring may not effectively predict failures."
            )
        elif beta < 2.0:
            return (
                "Slightly increasing failure rate (early wear-out). "
                "Equipment showing gradual degradation. "
                "Predictive maintenance is moderately effective."
            )
        elif beta < 4.0:
            return (
                "Increasing failure rate (wear-out pattern). "
                "Equipment following typical wear degradation. "
                "Predictive maintenance highly effective."
            )
        else:
            return (
                "Rapidly increasing failure rate (severe wear-out). "
                "Equipment experiencing accelerated degradation. "
                "Consider proactive replacement before failure."
            )

    def _calculate_r_squared(
        self,
        data: List[FailureData],
        beta: float,
        eta: float
    ) -> float:
        """
        Calculate R-squared for fit quality assessment.

        Args:
            data: Failure data
            beta: Estimated shape parameter
            eta: Estimated scale parameter

        Returns:
            R-squared value (0-1)
        """
        failures = sorted([d.time for d in data if d.is_failure])
        n = len(failures)

        if n < 2:
            return 0.0

        params = WeibullParameters(beta=beta, eta=eta)

        # Calculate empirical vs theoretical
        y_emp = []
        y_theo = []

        for i, t in enumerate(failures, 1):
            if t <= 0:
                continue
            rank = (i - 0.3) / (n + 0.4)
            y_emp.append(math.log(math.log(1 / (1 - rank))))
            y_theo.append(
                params.beta * (math.log(t) - math.log(params.eta))
            )

        if len(y_emp) < 2:
            return 0.0

        # R-squared calculation
        y_mean = sum(y_emp) / len(y_emp)
        ss_tot = sum((y - y_mean) ** 2 for y in y_emp)
        ss_res = sum((ye - yt) ** 2 for ye, yt in zip(y_emp, y_theo))

        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def _calculate_provenance(
        self,
        data: List[FailureData],
        current_age: float,
        params: WeibullParameters
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            data: Input failure data
            current_age: Current equipment age
            params: Estimated parameters

        Returns:
            SHA-256 hash string
        """
        # Create deterministic string representation
        data_str = "|".join(
            f"{d.time:.6f},{d.is_failure}"
            for d in sorted(data, key=lambda x: x.time)
        )
        provenance_str = (
            f"weibull|{self.config.method}|{current_age:.6f}|"
            f"{params.beta:.8f}|{params.eta:.8f}|{params.gamma:.8f}|"
            f"{data_str}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_mtbf(self, params: WeibullParameters) -> float:
        """
        Calculate Mean Time Between Failures (MTBF).

        MTBF = eta * Gamma(1 + 1/beta) + gamma

        Args:
            params: Weibull parameters

        Returns:
            MTBF in hours
        """
        # Gamma function approximation using Stirling's approximation
        # for Gamma(1 + 1/beta)
        x = 1 + 1 / params.beta

        # Use math.gamma if available (Python 3.2+)
        gamma_val = math.gamma(x)

        return params.eta * gamma_val + params.gamma

    def calculate_b_life(
        self,
        params: WeibullParameters,
        percentage: float = 10.0
    ) -> float:
        """
        Calculate B-life (time by which percentage of units fail).

        B10 life is time by which 10% of units fail, commonly used
        for maintenance planning.

        Args:
            params: Weibull parameters
            percentage: Failure percentage (default 10 for B10)

        Returns:
            B-life in hours
        """
        p = percentage / 100.0
        return self._inverse_cdf(p, params)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_weibull_analysis(
    failure_times: List[float],
    current_age: float = 0,
    confidence_level: float = 0.90
) -> WeibullAnalysisResult:
    """
    Quick Weibull analysis with default configuration.

    Args:
        failure_times: List of times to failure
        current_age: Current equipment age
        confidence_level: Confidence level for intervals

    Returns:
        WeibullAnalysisResult

    Example:
        >>> result = quick_weibull_analysis([5000, 6200, 4800, 7100], current_age=3000)
        >>> print(f"RUL: {result.rul_p50_hours:.0f} hours")
    """
    config = WeibullConfig(confidence_level=confidence_level)
    analyzer = WeibullAnalyzer(config)
    return analyzer.analyze_from_times(failure_times, current_age=current_age)
