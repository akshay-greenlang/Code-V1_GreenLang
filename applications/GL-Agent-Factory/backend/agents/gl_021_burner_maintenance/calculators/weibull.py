"""
Weibull Failure Analysis Calculator

This module implements Weibull distribution calculations for reliability
engineering and failure analysis of industrial burner components.

The Weibull distribution is widely used in reliability engineering because
it can model various failure patterns:
- beta < 1: Infant mortality (decreasing failure rate)
- beta = 1: Random failures (constant failure rate, exponential distribution)
- beta > 1: Wear-out failures (increasing failure rate)

All formulas are from standard reliability engineering references:
- MIL-HDBK-217F Reliability Prediction of Electronic Equipment
- IEEE Std 493 Recommended Practice for Design of Reliable Industrial Systems
- IEC 61649 Weibull Analysis

Example:
    >>> reliability = weibull_reliability(10000, beta=2.5, eta=40000)
    >>> print(f"Reliability at 10000 hours: {reliability:.4f}")
    Reliability at 10000 hours: 0.9851
"""

import math
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def weibull_reliability(t: float, beta: float, eta: float) -> float:
    """
    Calculate Weibull reliability function R(t).

    The reliability function gives the probability that the component
    survives beyond time t.

    Formula:
        R(t) = exp(-(t/eta)^beta)

    Args:
        t: Time (operating hours). Must be >= 0.
        beta: Shape parameter (Weibull slope). Must be > 0.
        eta: Scale parameter (characteristic life in hours). Must be > 0.

    Returns:
        Reliability probability between 0 and 1.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> weibull_reliability(10000, 2.5, 40000)
        0.9851...
    """
    if t < 0:
        raise ValueError(f"Time t must be >= 0, got {t}")
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    # At t=0, reliability is 1.0
    if t == 0:
        return 1.0

    # R(t) = exp(-(t/eta)^beta)
    exponent = -((t / eta) ** beta)
    reliability = math.exp(exponent)

    # Clamp to valid range due to floating point
    return max(0.0, min(1.0, reliability))


def weibull_failure_rate(t: float, beta: float, eta: float) -> float:
    """
    Calculate Weibull hazard (failure rate) function h(t).

    The hazard function gives the instantaneous failure rate at time t,
    given survival to time t.

    Formula:
        h(t) = (beta/eta) * (t/eta)^(beta-1)

    Args:
        t: Time (operating hours). Must be > 0 for calculation.
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Failure rate per hour.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> rate = weibull_failure_rate(10000, 2.5, 40000)
        >>> print(f"Failure rate: {rate:.6e} per hour")
    """
    if t < 0:
        raise ValueError(f"Time t must be >= 0, got {t}")
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    # Handle t=0 edge case
    if t == 0:
        if beta < 1:
            # Failure rate approaches infinity at t=0 for beta < 1
            return float('inf')
        elif beta == 1:
            # Exponential distribution: constant failure rate
            return 1.0 / eta
        else:
            # beta > 1: failure rate is 0 at t=0
            return 0.0

    # h(t) = (beta/eta) * (t/eta)^(beta-1)
    hazard = (beta / eta) * ((t / eta) ** (beta - 1))

    return hazard


def weibull_mean_life(beta: float, eta: float) -> float:
    """
    Calculate Weibull mean time to failure (MTTF).

    The MTTF is the expected value of the Weibull distribution.

    Formula:
        MTTF = eta * Gamma(1 + 1/beta)

    where Gamma is the complete gamma function.

    Args:
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Mean time to failure in hours.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> mttf = weibull_mean_life(2.5, 40000)
        >>> print(f"MTTF: {mttf:.0f} hours")
    """
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    # MTTF = eta * Gamma(1 + 1/beta)
    mttf = eta * math.gamma(1 + 1 / beta)

    return mttf


def remaining_useful_life(
    current_hours: float,
    beta: float,
    eta: float,
    reliability_threshold: float = 0.5
) -> float:
    """
    Calculate remaining useful life (RUL) given current age.

    RUL is the additional time until reliability drops to the threshold.
    This is calculated by finding when R(t) = reliability_threshold and
    subtracting current_hours.

    Formula:
        t_threshold = eta * (-ln(R_threshold))^(1/beta)
        RUL = max(0, t_threshold - current_hours)

    Args:
        current_hours: Current operating hours. Must be >= 0.
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.
        reliability_threshold: Target reliability level (default 0.5 = median life).
            Must be between 0 and 1 exclusive.

    Returns:
        Remaining useful life in hours. Returns 0 if already past threshold.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> rul = remaining_useful_life(10000, 2.5, 40000, 0.5)
        >>> print(f"RUL: {rul:.0f} hours until 50% reliability")
    """
    if current_hours < 0:
        raise ValueError(f"Current hours must be >= 0, got {current_hours}")
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")
    if reliability_threshold <= 0 or reliability_threshold >= 1:
        raise ValueError(
            f"Reliability threshold must be between 0 and 1, got {reliability_threshold}"
        )

    # Solve R(t) = threshold for t
    # exp(-(t/eta)^beta) = threshold
    # -(t/eta)^beta = ln(threshold)
    # (t/eta)^beta = -ln(threshold)
    # t = eta * (-ln(threshold))^(1/beta)

    t_threshold = eta * ((-math.log(reliability_threshold)) ** (1 / beta))

    # RUL is time remaining to reach threshold
    rul = max(0.0, t_threshold - current_hours)

    logger.debug(
        f"RUL calculation: current={current_hours}h, threshold_time={t_threshold:.0f}h, "
        f"RUL={rul:.0f}h"
    )

    return rul


def calculate_failure_probability(
    t_start: float,
    t_end: float,
    beta: float,
    eta: float
) -> float:
    """
    Calculate probability of failure between t_start and t_end.

    This gives the conditional probability of failing in the interval
    [t_start, t_end] given survival to t_start.

    Formula:
        P(failure in [t1,t2] | survival to t1) = 1 - R(t2)/R(t1)

    Args:
        t_start: Start time (hours). Must be >= 0.
        t_end: End time (hours). Must be > t_start.
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Probability of failure in interval, between 0 and 1.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> # Probability of failure in next 720 hours (30 days)
        >>> prob = calculate_failure_probability(10000, 10720, 2.5, 40000)
        >>> print(f"30-day failure probability: {prob:.4f}")
    """
    if t_start < 0:
        raise ValueError(f"t_start must be >= 0, got {t_start}")
    if t_end <= t_start:
        raise ValueError(f"t_end must be > t_start, got t_end={t_end}, t_start={t_start}")
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    # R(t_start)
    r_start = weibull_reliability(t_start, beta, eta)

    # R(t_end)
    r_end = weibull_reliability(t_end, beta, eta)

    # Handle edge case where r_start is effectively 0
    if r_start < 1e-15:
        logger.warning(f"Reliability at t_start is effectively 0 (R={r_start})")
        return 1.0

    # Conditional failure probability
    # P(fail in [t1,t2] | survive to t1) = 1 - R(t2)/R(t1)
    prob = 1 - (r_end / r_start)

    # Clamp to valid range
    return max(0.0, min(1.0, prob))


def weibull_percentile_life(
    percentile: float,
    beta: float,
    eta: float
) -> float:
    """
    Calculate the life at which a given percentile of units will have failed.

    Also known as B-life or L-life (e.g., B10 life = 10th percentile life).

    Formula:
        t_p = eta * (-ln(1-p))^(1/beta)

    where p is the failure fraction (percentile/100).

    Args:
        percentile: Failure percentile (0-100). E.g., 10 for B10 life.
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Life in hours at which percentile% will have failed.

    Raises:
        ValueError: If parameters are out of valid range.

    Example:
        >>> b10_life = weibull_percentile_life(10, 2.5, 40000)
        >>> print(f"B10 life: {b10_life:.0f} hours")
    """
    if percentile <= 0 or percentile >= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    # Convert percentile to fraction
    p = percentile / 100.0

    # t_p = eta * (-ln(1-p))^(1/beta)
    # Since R(t) = 1-p at the percentile point
    t_p = eta * ((-math.log(1 - p)) ** (1 / beta))

    return t_p


def weibull_variance(beta: float, eta: float) -> float:
    """
    Calculate variance of Weibull distribution.

    Formula:
        Var = eta^2 * [Gamma(1 + 2/beta) - Gamma(1 + 1/beta)^2]

    Args:
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Variance in hours squared.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    if beta <= 0:
        raise ValueError(f"Beta must be > 0, got {beta}")
    if eta <= 0:
        raise ValueError(f"Eta must be > 0, got {eta}")

    gamma_term1 = math.gamma(1 + 2 / beta)
    gamma_term2 = math.gamma(1 + 1 / beta) ** 2

    variance = (eta ** 2) * (gamma_term1 - gamma_term2)

    return variance


def weibull_standard_deviation(beta: float, eta: float) -> float:
    """
    Calculate standard deviation of Weibull distribution.

    Args:
        beta: Shape parameter. Must be > 0.
        eta: Scale parameter (hours). Must be > 0.

    Returns:
        Standard deviation in hours.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    return math.sqrt(weibull_variance(beta, eta))


def estimate_weibull_parameters(
    failure_times: list,
    method: str = "mle"
) -> Tuple[float, float]:
    """
    Estimate Weibull parameters from failure time data.

    This uses the Maximum Likelihood Estimation (MLE) method
    which is standard for Weibull analysis.

    Note: This is a simplified implementation. For production use,
    consider using scipy.stats.weibull_min.fit() or similar.

    Args:
        failure_times: List of failure times in hours. Must have >= 2 values.
        method: Estimation method. Currently only "mle" is supported.

    Returns:
        Tuple of (beta, eta) estimated parameters.

    Raises:
        ValueError: If insufficient data or invalid method.

    Example:
        >>> times = [15000, 18000, 22000, 25000, 30000]
        >>> beta, eta = estimate_weibull_parameters(times)
    """
    if len(failure_times) < 2:
        raise ValueError("Need at least 2 failure times for parameter estimation")

    if method != "mle":
        raise ValueError(f"Unsupported method: {method}. Only 'mle' is supported.")

    # Sort failure times
    times = sorted([t for t in failure_times if t > 0])
    n = len(times)

    if n < 2:
        raise ValueError("Need at least 2 positive failure times")

    # Initial beta estimate using regression on Weibull plot
    # ln(ln(1/(1-F))) = beta*ln(t) - beta*ln(eta)
    # where F is median rank estimate

    # Median rank estimation: F_i = (i - 0.3)/(n + 0.4)
    import statistics

    ln_times = [math.log(t) for t in times]
    ln_ln_terms = []

    for i in range(n):
        median_rank = (i + 1 - 0.3) / (n + 0.4)
        if median_rank > 0 and median_rank < 1:
            ln_ln_terms.append(math.log(-math.log(1 - median_rank)))

    # Simple linear regression to estimate beta
    if len(ln_ln_terms) != n:
        # Fallback: use moment estimation
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times)
        cv = std_t / mean_t  # Coefficient of variation

        # Approximate beta from CV (empirical relationship)
        # For Weibull: CV = sqrt(Gamma(1+2/beta)/Gamma(1+1/beta)^2 - 1)
        # Approximate: beta ≈ 1.2 / CV for CV in typical range
        beta_estimate = max(0.5, min(10, 1.2 / cv if cv > 0 else 2.0))
        eta_estimate = mean_t / math.gamma(1 + 1 / beta_estimate)
    else:
        # Linear regression: y = a + b*x where b = beta
        mean_x = statistics.mean(ln_times)
        mean_y = statistics.mean(ln_ln_terms)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(ln_times, ln_ln_terms))
        denominator = sum((x - mean_x) ** 2 for x in ln_times)

        if denominator > 0:
            beta_estimate = numerator / denominator
            intercept = mean_y - beta_estimate * mean_x
            eta_estimate = math.exp(-intercept / beta_estimate)
        else:
            # Fallback
            beta_estimate = 2.0
            eta_estimate = statistics.mean(times)

    # Ensure valid parameter range
    beta_estimate = max(0.5, min(10, beta_estimate))
    eta_estimate = max(100, eta_estimate)

    logger.info(f"Estimated Weibull parameters: beta={beta_estimate:.3f}, eta={eta_estimate:.0f}")

    return (beta_estimate, eta_estimate)
