"""
greenlang/utils/uncertainty.py

Uncertainty propagation for emission calculations

Provides:
- Uncertainty propagation from emission factors to calculated emissions
- Combined uncertainty calculation (multiple sources)
- Confidence interval estimation
- Monte Carlo simulation support (for complex scenarios)

Based on:
- IPCC Guidelines for National Greenhouse Gas Inventories (2006) - Chapter 3: Uncertainties
- ISO Guide to the Expression of Uncertainty in Measurement (GUM)
- GHGP Corporate Standard - Uncertainty Assessment

Example:
    >>> from greenlang.utils.uncertainty import propagate_uncertainty
    >>>
    >>> # Factor has ±5% uncertainty, amount = 1000
    >>> result = propagate_uncertainty(
    ...     emission_value=10210.0,  # kg CO2e
    ...     factor_uncertainty_pct=5.0,  # ±5%
    ...     amount=1000.0,
    ...     amount_uncertainty_pct=2.0  # Optional: ±2% on amount measurement
    ... )
    >>> print(result)
    {
        'value': 10210.0,
        'uncertainty_pct': 5.39,  # Combined uncertainty
        'uncertainty_absolute': 550.0,
        'confidence_interval_95': (9110.0, 11310.0),
        'lower_bound': 9110.0,
        'upper_bound': 11310.0
    }
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """
    Uncertainty analysis result.

    Attributes:
        value: Central estimate (mean value)
        uncertainty_pct: Combined relative uncertainty as percentage (±X%)
        uncertainty_absolute: Absolute uncertainty (± kg CO2e)
        lower_bound_95: Lower bound of 95% confidence interval
        upper_bound_95: Upper bound of 95% confidence interval
        sources: List of uncertainty sources and their contributions
    """

    value: float
    uncertainty_pct: float
    uncertainty_absolute: float
    lower_bound_95: float
    upper_bound_95: float
    sources: List[Dict[str, float]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "value": self.value,
            "uncertainty_pct": self.uncertainty_pct,
            "uncertainty_absolute": self.uncertainty_absolute,
            "confidence_interval_95": (self.lower_bound_95, self.upper_bound_95),
            "lower_bound": self.lower_bound_95,
            "upper_bound": self.upper_bound_95,
            "sources": self.sources,
        }


def propagate_uncertainty(
    emission_value: float,
    factor_uncertainty_pct: float,
    amount: Optional[float] = None,
    amount_uncertainty_pct: float = 0.0,
    additional_uncertainties: Optional[List[float]] = None,
) -> UncertaintyResult:
    """
    Propagate uncertainty from emission factor to calculated emissions.

    Uses error propagation formula for multiplication:
    If Z = A × B, then U_Z = Z × sqrt((U_A/A)^2 + (U_B/B)^2)

    Args:
        emission_value: Calculated emission value (kg CO2e)
        factor_uncertainty_pct: Emission factor uncertainty (±%)
        amount: Consumption amount (optional, for detailed breakdown)
        amount_uncertainty_pct: Measurement uncertainty on amount (±%)
        additional_uncertainties: List of additional uncertainty sources (±% each)

    Returns:
        UncertaintyResult with propagated uncertainty and confidence intervals
    """
    # Convert percentage uncertainties to relative (0-1)
    factor_u_rel = factor_uncertainty_pct / 100.0
    amount_u_rel = amount_uncertainty_pct / 100.0

    # Collect all uncertainty sources
    sources = []

    # Source 1: Emission factor uncertainty (always present)
    sources.append({
        "source": "emission_factor",
        "uncertainty_pct": factor_uncertainty_pct,
        "contribution_pct": factor_uncertainty_pct,  # Will be recalculated
    })

    # Source 2: Amount measurement uncertainty (if provided)
    if amount_uncertainty_pct > 0:
        sources.append({
            "source": "amount_measurement",
            "uncertainty_pct": amount_uncertainty_pct,
            "contribution_pct": amount_uncertainty_pct,
        })

    # Source 3: Additional uncertainties (e.g., efficiency, renewable %, etc.)
    if additional_uncertainties:
        for i, unc_pct in enumerate(additional_uncertainties):
            sources.append({
                "source": f"additional_{i+1}",
                "uncertainty_pct": unc_pct,
                "contribution_pct": unc_pct,
            })

    # Combine uncertainties using root-sum-of-squares (RSS)
    # U_combined = sqrt(U1^2 + U2^2 + U3^2 + ...)
    combined_u_rel_squared = factor_u_rel ** 2

    if amount_uncertainty_pct > 0:
        combined_u_rel_squared += amount_u_rel ** 2

    if additional_uncertainties:
        for unc_pct in additional_uncertainties:
            unc_rel = unc_pct / 100.0
            combined_u_rel_squared += unc_rel ** 2

    combined_u_rel = math.sqrt(combined_u_rel_squared)
    combined_u_pct = combined_u_rel * 100.0

    # Calculate absolute uncertainty
    uncertainty_absolute = emission_value * combined_u_rel

    # Calculate 95% confidence interval (±1.96 standard deviations)
    # Assumes normal distribution
    ci_multiplier = 1.96
    ci_half_width = uncertainty_absolute * ci_multiplier

    lower_bound_95 = emission_value - ci_half_width
    upper_bound_95 = emission_value + ci_half_width

    # Ensure lower bound is not negative (emissions can't be negative)
    lower_bound_95 = max(0.0, lower_bound_95)

    # Calculate contribution of each source to total uncertainty
    # Contribution = (U_i^2 / U_total^2) * 100
    for source in sources:
        u_i_rel = source["uncertainty_pct"] / 100.0
        contribution = (u_i_rel ** 2 / combined_u_rel_squared) * 100
        source["contribution_pct"] = contribution

    return UncertaintyResult(
        value=emission_value,
        uncertainty_pct=combined_u_pct,
        uncertainty_absolute=uncertainty_absolute,
        lower_bound_95=lower_bound_95,
        upper_bound_95=upper_bound_95,
        sources=sources,
    )


def combine_uncertainties(uncertainties: List[float]) -> float:
    """
    Combine multiple independent uncertainty sources using root-sum-of-squares.

    Args:
        uncertainties: List of uncertainty percentages (±%)

    Returns:
        Combined uncertainty (±%)

    Example:
        >>> combine_uncertainties([5.0, 3.0, 2.0])
        6.16  # ±6.16%
    """
    if not uncertainties:
        return 0.0

    # Convert to relative uncertainties (0-1)
    rel_uncertainties = [u / 100.0 for u in uncertainties]

    # Root-sum-of-squares
    combined_squared = sum(u ** 2 for u in rel_uncertainties)
    combined_rel = math.sqrt(combined_squared)

    return combined_rel * 100.0


def monte_carlo_simulation(
    amount: float,
    factor_mean: float,
    factor_uncertainty_pct: float,
    n_iterations: int = 10000,
    distribution: str = "normal",
) -> Dict:
    """
    Monte Carlo simulation for uncertainty analysis (advanced).

    Samples from probability distributions for input parameters and
    calculates distribution of output emissions.

    Args:
        amount: Consumption amount (assumed exact for simplicity)
        factor_mean: Mean emission factor
        factor_uncertainty_pct: Uncertainty on factor (±%)
        n_iterations: Number of Monte Carlo iterations
        distribution: Probability distribution ("normal", "lognormal", "uniform")

    Returns:
        Dict with:
        - mean: Mean of simulated emissions
        - std: Standard deviation
        - percentile_5: 5th percentile (lower bound 90% CI)
        - percentile_50: Median
        - percentile_95: 95th percentile (upper bound 90% CI)
        - percentile_2_5: 2.5th percentile (lower bound 95% CI)
        - percentile_97_5: 97.5th percentile (upper bound 95% CI)

    Note:
        Requires numpy for performance. Falls back to analytical if not available.
    """
    try:
        import numpy as np

        # Convert uncertainty to standard deviation
        # For normal distribution: 95% CI = ±1.96σ
        # So σ = uncertainty / 1.96
        factor_std = (factor_uncertainty_pct / 100.0) * factor_mean / 1.96

        # Generate samples
        if distribution == "normal":
            factor_samples = np.random.normal(
                loc=factor_mean, scale=factor_std, size=n_iterations
            )
        elif distribution == "lognormal":
            # Log-normal distribution (ensures positive values)
            mu = math.log(factor_mean)
            sigma = factor_std / factor_mean  # Coefficient of variation
            factor_samples = np.random.lognormal(
                mean=mu, sigma=sigma, size=n_iterations
            )
        elif distribution == "uniform":
            # Uniform distribution within ±uncertainty
            half_width = (factor_uncertainty_pct / 100.0) * factor_mean
            factor_samples = np.random.uniform(
                low=factor_mean - half_width,
                high=factor_mean + half_width,
                size=n_iterations,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Calculate emissions for each sample
        emission_samples = amount * factor_samples

        # Calculate statistics
        return {
            "mean": float(np.mean(emission_samples)),
            "std": float(np.std(emission_samples)),
            "percentile_2_5": float(np.percentile(emission_samples, 2.5)),
            "percentile_5": float(np.percentile(emission_samples, 5)),
            "percentile_50": float(np.percentile(emission_samples, 50)),
            "percentile_95": float(np.percentile(emission_samples, 95)),
            "percentile_97_5": float(np.percentile(emission_samples, 97.5)),
            "samples": emission_samples.tolist()
            if n_iterations <= 1000
            else None,  # Only return samples if small
        }

    except ImportError:
        logger.warning(
            "NumPy not available for Monte Carlo simulation. Using analytical approximation."
        )

        # Fallback to analytical approximation
        emission_mean = amount * factor_mean
        emission_std = amount * (
            (factor_uncertainty_pct / 100.0) * factor_mean / 1.96
        )

        return {
            "mean": emission_mean,
            "std": emission_std,
            "percentile_2_5": emission_mean - 1.96 * emission_std,
            "percentile_5": emission_mean - 1.645 * emission_std,
            "percentile_50": emission_mean,
            "percentile_95": emission_mean + 1.645 * emission_std,
            "percentile_97_5": emission_mean + 1.96 * emission_std,
            "samples": None,
            "note": "Analytical approximation (NumPy not available)",
        }


def calculate_relative_uncertainty(
    absolute_uncertainty: float, value: float
) -> float:
    """
    Calculate relative uncertainty from absolute uncertainty.

    Args:
        absolute_uncertainty: Absolute uncertainty (same units as value)
        value: Central estimate value

    Returns:
        Relative uncertainty as percentage (±%)

    Example:
        >>> calculate_relative_uncertainty(500, 10000)
        5.0  # ±5%
    """
    if value == 0:
        return 0.0

    return (absolute_uncertainty / abs(value)) * 100.0


def calculate_absolute_uncertainty(uncertainty_pct: float, value: float) -> float:
    """
    Calculate absolute uncertainty from relative uncertainty.

    Args:
        uncertainty_pct: Relative uncertainty (±%)
        value: Central estimate value

    Returns:
        Absolute uncertainty (same units as value)

    Example:
        >>> calculate_absolute_uncertainty(5.0, 10000)
        500.0  # ±500 kg
    """
    return (uncertainty_pct / 100.0) * abs(value)


def uncertainty_contribution_analysis(
    uncertainties: Dict[str, float], emission_value: float
) -> Dict[str, Dict]:
    """
    Analyze contribution of each uncertainty source to total uncertainty.

    Args:
        uncertainties: Dict mapping source name to uncertainty percentage
        emission_value: Total emission value

    Returns:
        Dict with analysis results including:
        - total_uncertainty_pct: Combined uncertainty
        - contributions: Dict of each source's contribution

    Example:
        >>> analysis = uncertainty_contribution_analysis(
        ...     {
        ...         "emission_factor": 5.0,
        ...         "amount_measurement": 2.0,
        ...         "efficiency": 3.0
        ...     },
        ...     emission_value=10000
        ... )
        >>> print(analysis["total_uncertainty_pct"])
        6.16  # Combined
        >>> print(analysis["contributions"]["emission_factor"]["contribution_pct"])
        65.9  # Emission factor contributes ~66% of total uncertainty
    """
    # Calculate combined uncertainty
    unc_list = list(uncertainties.values())
    total_u_pct = combine_uncertainties(unc_list)

    total_u_rel_squared = (total_u_pct / 100.0) ** 2

    # Calculate contribution of each source
    contributions = {}
    for source, unc_pct in uncertainties.items():
        u_rel = unc_pct / 100.0
        contribution_pct = (u_rel ** 2 / total_u_rel_squared) * 100

        contributions[source] = {
            "uncertainty_pct": unc_pct,
            "contribution_pct": contribution_pct,
            "absolute_uncertainty": calculate_absolute_uncertainty(
                unc_pct, emission_value
            ),
        }

    return {
        "total_uncertainty_pct": total_u_pct,
        "total_uncertainty_absolute": calculate_absolute_uncertainty(
            total_u_pct, emission_value
        ),
        "contributions": contributions,
    }


# ==================== GHGP-Specific Uncertainty Tiers ====================


def categorize_uncertainty_tier(uncertainty_pct: float) -> Tuple[str, str]:
    """
    Categorize uncertainty level according to GHGP guidance.

    Based on GHG Protocol Corporate Standard uncertainty guidance:
    - Low: < 10%
    - Medium: 10-30%
    - High: > 30%

    Args:
        uncertainty_pct: Uncertainty percentage

    Returns:
        Tuple of (tier, description)

    Example:
        >>> categorize_uncertainty_tier(5.0)
        ('low', 'High confidence in data quality')
        >>> categorize_uncertainty_tier(25.0)
        ('medium', 'Moderate confidence, acceptable for most reporting')
        >>> categorize_uncertainty_tier(50.0)
        ('high', 'Low confidence, improvement recommended')
    """
    if uncertainty_pct < 10:
        return (
            "low",
            "High confidence in data quality (suitable for all reporting frameworks)",
        )
    elif uncertainty_pct < 30:
        return (
            "medium",
            "Moderate confidence, acceptable for most reporting (GHGP compliant)",
        )
    else:
        return (
            "high",
            "Low confidence, data quality improvement recommended (may not meet some regulatory requirements)",
        )


# ==================== TESTING & EXAMPLES ====================

if __name__ == "__main__":
    # Example 1: Simple uncertainty propagation
    print("=" * 60)
    print("Example 1: Simple Uncertainty Propagation")
    print("=" * 60)

    result = propagate_uncertainty(
        emission_value=10210.0,  # 1000 gallons diesel × 10.21 kgCO2e/gallon
        factor_uncertainty_pct=5.0,  # EPA factor has ±5% uncertainty
    )

    print(f"Emission value: {result.value:.2f} kg CO2e")
    print(f"Uncertainty: ±{result.uncertainty_pct:.2f}%")
    print(f"Absolute uncertainty: ±{result.uncertainty_absolute:.2f} kg CO2e")
    print(
        f"95% CI: ({result.lower_bound_95:.2f}, {result.upper_bound_95:.2f}) kg CO2e"
    )
    print()

    # Example 2: Combined uncertainties
    print("=" * 60)
    print("Example 2: Combined Uncertainties (Factor + Measurement)")
    print("=" * 60)

    result2 = propagate_uncertainty(
        emission_value=10210.0,
        factor_uncertainty_pct=5.0,  # Factor uncertainty
        amount=1000.0,
        amount_uncertainty_pct=2.0,  # Measurement uncertainty on fuel amount
    )

    print(f"Emission value: {result2.value:.2f} kg CO2e")
    print(f"Combined uncertainty: ±{result2.uncertainty_pct:.2f}%")
    print("\nUncertainty sources:")
    for source in result2.sources:
        print(f"  - {source['source']}: ±{source['uncertainty_pct']:.2f}% "
              f"(contributes {source['contribution_pct']:.1f}% to total)")
    print()

    # Example 3: Uncertainty tier categorization
    print("=" * 60)
    print("Example 3: GHGP Uncertainty Tier Categorization")
    print("=" * 60)

    for unc in [5.0, 15.0, 40.0]:
        tier, desc = categorize_uncertainty_tier(unc)
        print(f"Uncertainty: ±{unc:.1f}% → Tier: {tier.upper()} ({desc})")
    print()

    # Example 4: Contribution analysis
    print("=" * 60)
    print("Example 4: Uncertainty Contribution Analysis")
    print("=" * 60)

    analysis = uncertainty_contribution_analysis(
        {
            "emission_factor": 5.0,
            "amount_measurement": 2.0,
            "efficiency_estimate": 3.0,
        },
        emission_value=10210.0,
    )

    print(
        f"Total combined uncertainty: ±{analysis['total_uncertainty_pct']:.2f}%"
    )
    print("\nContribution by source:")
    for source, data in analysis["contributions"].items():
        print(
            f"  - {source}: ±{data['uncertainty_pct']:.1f}% "
            f"(contributes {data['contribution_pct']:.1f}% to total)"
        )
