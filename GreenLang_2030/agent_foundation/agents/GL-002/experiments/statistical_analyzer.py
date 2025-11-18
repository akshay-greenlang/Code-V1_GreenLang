"""
Statistical Analyzer for A/B Testing

This module provides statistical analysis tools for comparing experiment variants,
calculating significance, effect sizes, and power analysis.

Example:
    >>> analyzer = StatisticalAnalyzer(significance_level=0.05)
    >>> result = analyzer.compare_variants(
    ...     control_mean=100, control_std=10, control_n=1000,
    ...     treatment_mean=105, treatment_std=12, treatment_n=1000
    ... )
    >>> print(f"P-value: {result['p_value']}, Significant: {result['is_significant']}")
"""

from typing import Dict, Tuple, Optional
import math
import logging
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analysis engine for A/B testing.

    Provides methods for comparing variants, calculating effect sizes,
    confidence intervals, and power analysis.

    Attributes:
        significance_level: Alpha threshold for statistical significance (default: 0.05)
        power_target: Target statistical power (default: 0.8)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        power_target: float = 0.8
    ):
        """
        Initialize StatisticalAnalyzer.

        Args:
            significance_level: Alpha threshold (typically 0.05)
            power_target: Desired statistical power (typically 0.8)
        """
        self.significance_level = significance_level
        self.power_target = power_target

        logger.info(
            f"StatisticalAnalyzer initialized: alpha={significance_level}, "
            f"power={power_target}"
        )

    def compare_variants(
        self,
        control_mean: float,
        control_std: float,
        control_n: int,
        treatment_mean: float,
        treatment_std: float,
        treatment_n: int
    ) -> Dict[str, any]:
        """
        Compare two variants using Welch's t-test.

        Uses Welch's t-test which doesn't assume equal variances.

        Args:
            control_mean: Mean of control variant
            control_std: Standard deviation of control
            control_n: Sample size of control
            treatment_mean: Mean of treatment variant
            treatment_std: Standard deviation of treatment
            treatment_n: Sample size of treatment

        Returns:
            Dictionary with test results including p-value, significance,
            effect size, confidence intervals, and recommendation
        """
        try:
            # Perform Welch's t-test
            t_statistic, p_value = self._welch_t_test(
                mean1=control_mean,
                std1=control_std,
                n1=control_n,
                mean2=treatment_mean,
                std2=treatment_std,
                n2=treatment_n
            )

            # Determine if statistically significant
            is_significant = p_value < self.significance_level

            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(
                mean1=control_mean,
                std1=control_std,
                n1=control_n,
                mean2=treatment_mean,
                std2=treatment_std,
                n2=treatment_n
            )

            # Calculate relative improvement
            relative_improvement = (
                ((treatment_mean - control_mean) / control_mean) * 100
                if control_mean != 0 else 0
            )

            # Calculate confidence interval for difference
            ci_lower, ci_upper = self._calculate_difference_ci(
                mean1=control_mean,
                std1=control_std,
                n1=control_n,
                mean2=treatment_mean,
                std2=treatment_std,
                n2=treatment_n
            )

            # Calculate statistical power
            statistical_power = self._calculate_power(
                effect_size=effect_size,
                n1=control_n,
                n2=treatment_n,
                alpha=self.significance_level
            )

            # Calculate required sample size for target power
            required_sample_size = self._calculate_required_sample_size(
                effect_size=effect_size,
                alpha=self.significance_level,
                power=self.power_target
            )

            # Determine recommendation and confidence
            recommendation, confidence = self._determine_recommendation(
                is_significant=is_significant,
                relative_improvement=relative_improvement,
                effect_size=effect_size,
                power=statistical_power,
                n1=control_n,
                n2=treatment_n,
                required_n=required_sample_size
            )

            result = {
                "p_value": round(p_value, 6),
                "is_significant": is_significant,
                "significance_level": self.significance_level,
                "effect_size": round(effect_size, 4),
                "relative_improvement": round(relative_improvement, 2),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "statistical_power": round(statistical_power, 4),
                "required_sample_size": int(required_sample_size),
                "recommendation": recommendation,
                "confidence": confidence
            }

            logger.info(
                f"Variant comparison: p={result['p_value']:.4f}, "
                f"significant={is_significant}, improvement={relative_improvement:.2f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to compare variants: {e}", exc_info=True)
            raise

    def _welch_t_test(
        self,
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int
    ) -> Tuple[float, float]:
        """
        Perform Welch's t-test (unequal variances t-test).

        Returns:
            Tuple of (t-statistic, p-value)
        """
        # Standard errors
        se1 = std1 / math.sqrt(n1)
        se2 = std2 / math.sqrt(n2)

        # Standard error of difference
        se_diff = math.sqrt(se1**2 + se2**2)

        # t-statistic
        t_statistic = (mean2 - mean1) / se_diff if se_diff > 0 else 0

        # Degrees of freedom (Welch-Satterthwaite equation)
        numerator = (std1**2 / n1 + std2**2 / n2) ** 2
        denominator = (
            (std1**2 / n1)**2 / (n1 - 1) +
            (std2**2 / n2)**2 / (n2 - 1)
        )
        df = numerator / denominator if denominator > 0 else n1 + n2 - 2

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

        return t_statistic, p_value

    def _calculate_cohens_d(
        self,
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Uses pooled standard deviation.

        Returns:
            Cohen's d (effect size)
        """
        # Pooled standard deviation
        pooled_std = math.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )

        # Cohen's d
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

        return cohens_d

    def _calculate_difference_ci(
        self,
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference between means.

        Args:
            confidence_level: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Difference in means
        diff = mean2 - mean1

        # Standard error of difference
        se_diff = math.sqrt((std1**2 / n1) + (std2**2 / n2))

        # Degrees of freedom
        numerator = (std1**2 / n1 + std2**2 / n2) ** 2
        denominator = (
            (std1**2 / n1)**2 / (n1 - 1) +
            (std2**2 / n2)**2 / (n2 - 1)
        )
        df = numerator / denominator if denominator > 0 else n1 + n2 - 2

        # Critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Confidence interval
        margin = t_critical * se_diff
        ci_lower = diff - margin
        ci_upper = diff + margin

        return ci_lower, ci_upper

    def calculate_confidence_interval(
        self,
        mean: float,
        std: float,
        n: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a single mean.

        Args:
            mean: Sample mean
            std: Sample standard deviation
            n: Sample size
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n <= 1:
            return mean, mean

        # Standard error
        se = std / math.sqrt(n)

        # Degrees of freedom
        df = n - 1

        # Critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Confidence interval
        margin = t_critical * se
        ci_lower = mean - margin
        ci_upper = mean + margin

        return ci_lower, ci_upper

    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: float
    ) -> float:
        """
        Calculate statistical power of test.

        Args:
            effect_size: Cohen's d
            n1: Sample size of group 1
            n2: Sample size of group 2
            alpha: Significance level

        Returns:
            Statistical power (probability of detecting effect if it exists)
        """
        # Non-centrality parameter
        ncp = effect_size * math.sqrt((n1 * n2) / (n1 + n2))

        # Degrees of freedom
        df = n1 + n2 - 2

        # Critical value
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Power calculation using non-central t-distribution
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)

        return max(0.0, min(1.0, power))  # Clamp to [0, 1]

    def _calculate_required_sample_size(
        self,
        effect_size: float,
        alpha: float,
        power: float,
        ratio: float = 1.0
    ) -> int:
        """
        Calculate required sample size per group.

        Args:
            effect_size: Expected Cohen's d
            alpha: Significance level
            power: Desired power
            ratio: Allocation ratio (treatment:control)

        Returns:
            Required sample size per group
        """
        if effect_size == 0:
            return 10000  # Very large if no effect expected

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n = (
            (z_alpha + z_beta)**2 * (1 + 1/ratio) / (effect_size**2)
        )

        return max(10, int(math.ceil(n)))  # Minimum of 10

    def _determine_recommendation(
        self,
        is_significant: bool,
        relative_improvement: float,
        effect_size: float,
        power: float,
        n1: int,
        n2: int,
        required_n: int
    ) -> Tuple[str, str]:
        """
        Determine recommendation and confidence level.

        Args:
            is_significant: Whether result is statistically significant
            relative_improvement: Percentage improvement
            effect_size: Cohen's d
            power: Statistical power
            n1: Sample size group 1
            n2: Sample size group 2
            required_n: Required sample size for target power

        Returns:
            Tuple of (recommendation, confidence_level)
        """
        min_sample = min(n1, n2)

        # Check if we have enough samples
        enough_samples = min_sample >= required_n

        # Determine confidence level
        if power >= 0.8 and enough_samples:
            confidence = "high"
        elif power >= 0.6 or min_sample >= required_n * 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine recommendation
        if not enough_samples:
            recommendation = "continue"  # Need more data
        elif is_significant and relative_improvement > 5:
            recommendation = "ship"  # Clear winner
        elif is_significant and relative_improvement < -5:
            recommendation = "stop"  # Significant regression
        elif not is_significant and abs(effect_size) < 0.2:
            recommendation = "stop"  # No meaningful difference
        else:
            recommendation = "iterate"  # Inconclusive, try different approach

        return recommendation, confidence

    def calculate_minimum_detectable_effect(
        self,
        n1: int,
        n2: int,
        alpha: float = 0.05,
        power: float = 0.8,
        std: float = 1.0
    ) -> float:
        """
        Calculate minimum detectable effect size.

        Args:
            n1: Sample size group 1
            n2: Sample size group 2
            alpha: Significance level
            power: Desired power
            std: Standard deviation

        Returns:
            Minimum detectable effect (Cohen's d)
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # MDE formula
        mde = (z_alpha + z_beta) * std * math.sqrt((1/n1) + (1/n2))

        return mde

    def bayesian_probability_of_superiority(
        self,
        control_mean: float,
        control_std: float,
        control_n: int,
        treatment_mean: float,
        treatment_std: float,
        treatment_n: int,
        samples: int = 10000
    ) -> float:
        """
        Calculate Bayesian probability that treatment is superior to control.

        Uses Monte Carlo sampling from posterior distributions.

        Args:
            samples: Number of Monte Carlo samples

        Returns:
            Probability that treatment > control (0-1)
        """
        # Sample from posterior distributions (assuming normal priors)
        control_samples = np.random.normal(
            control_mean,
            control_std / math.sqrt(control_n),
            samples
        )

        treatment_samples = np.random.normal(
            treatment_mean,
            treatment_std / math.sqrt(treatment_n),
            samples
        )

        # Calculate probability of superiority
        prob_superior = np.mean(treatment_samples > control_samples)

        logger.info(
            f"Bayesian probability of superiority: {prob_superior:.4f}"
        )

        return float(prob_superior)
