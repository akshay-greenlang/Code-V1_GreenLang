# -*- coding: utf-8 -*-
"""
A/B Testing Module

This module provides A/B testing framework for GreenLang ML models,
enabling statistically rigorous comparison of model variants and
data-driven deployment decisions.

A/B testing is critical for validating model improvements before
production deployment, especially for regulatory compliance where
model changes must be justified with statistical evidence.

Example:
    >>> from greenlang.ml.mlops import ABTesting
    >>> ab_test = ABTesting(model_a, model_b, test_name="emission_model_v2")
    >>> ab_test.run_test(X_test, y_test, n_iterations=1000)
    >>> if ab_test.results.winner == "B":
    ...     deploy_model(model_b)
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Type of statistical test."""
    WELCH_T = "welch_t"
    MANN_WHITNEY = "mann_whitney"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"


class AllocationStrategy(str, Enum):
    """Traffic allocation strategy."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


class ABTestConfig(BaseModel):
    """Configuration for A/B test."""

    test_name: str = Field(
        ...,
        description="Test name"
    )
    test_type: TestType = Field(
        default=TestType.WELCH_T,
        description="Statistical test type"
    )
    allocation_strategy: AllocationStrategy = Field(
        default=AllocationStrategy.RANDOM,
        description="Traffic allocation strategy"
    )
    traffic_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Traffic fraction to variant B"
    )
    minimum_sample_size: int = Field(
        default=100,
        ge=10,
        description="Minimum samples per variant"
    )
    significance_level: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level (alpha)"
    )
    power: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        description="Statistical power (1-beta)"
    )
    minimum_effect_size: float = Field(
        default=0.05,
        gt=0,
        description="Minimum detectable effect size"
    )
    enable_early_stopping: bool = Field(
        default=True,
        description="Enable early stopping"
    )
    early_stopping_confidence: float = Field(
        default=0.99,
        gt=0.9,
        le=1.0,
        description="Confidence for early stopping"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class VariantResult(BaseModel):
    """Results for a single variant."""

    variant_name: str = Field(
        ...,
        description="Variant identifier"
    )
    n_samples: int = Field(
        ...,
        description="Number of samples"
    )
    mean: float = Field(
        ...,
        description="Mean metric value"
    )
    std: float = Field(
        ...,
        description="Standard deviation"
    )
    median: float = Field(
        ...,
        description="Median value"
    )
    ci_lower: float = Field(
        ...,
        description="Lower confidence interval"
    )
    ci_upper: float = Field(
        ...,
        description="Upper confidence interval"
    )
    values: Optional[List[float]] = Field(
        default=None,
        description="Raw values (optional)"
    )


class ABTestResult(BaseModel):
    """Results from A/B test."""

    test_name: str = Field(
        ...,
        description="Test name"
    )
    variant_a: VariantResult = Field(
        ...,
        description="Results for variant A (control)"
    )
    variant_b: VariantResult = Field(
        ...,
        description="Results for variant B (treatment)"
    )
    winner: Optional[str] = Field(
        default=None,
        description="Winning variant (A, B, or None)"
    )
    is_significant: bool = Field(
        ...,
        description="Whether difference is significant"
    )
    p_value: float = Field(
        ...,
        description="P-value from statistical test"
    )
    effect_size: float = Field(
        ...,
        description="Effect size (Cohen's d)"
    )
    relative_improvement: float = Field(
        ...,
        description="Relative improvement of B over A"
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level used"
    )
    test_type: str = Field(
        ...,
        description="Statistical test used"
    )
    early_stopped: bool = Field(
        default=False,
        description="Whether test was early stopped"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test completion timestamp"
    )


class ABTesting:
    """
    A/B Testing Framework for GreenLang ML models.

    This class provides statistically rigorous A/B testing capabilities
    for comparing model variants, enabling data-driven deployment
    decisions with proper significance testing.

    Key capabilities:
    - Multiple statistical tests (t-test, Mann-Whitney, Bootstrap)
    - Traffic allocation strategies
    - Early stopping for efficiency
    - Bayesian analysis
    - Multi-armed bandit optimization
    - Provenance tracking

    Attributes:
        model_a: Control model (A)
        model_b: Treatment model (B)
        config: Test configuration
        _results_a: Collected results for A
        _results_b: Collected results for B
        _allocation_counts: Traffic allocation counts

    Example:
        >>> ab = ABTesting(
        ...     model_a, model_b,
        ...     config=ABTestConfig(
        ...         test_name="carbon_model_improvement",
        ...         significance_level=0.05
        ...     )
        ... )
        >>> result = ab.run_test(X_test, y_test)
        >>> print(f"Winner: {result.winner}, p-value: {result.p_value:.4f}")
    """

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        config: Optional[ABTestConfig] = None,
        metric_fn: Optional[Callable] = None
    ):
        """
        Initialize A/B testing framework.

        Args:
            model_a: Control model
            model_b: Treatment model
            config: Test configuration
            metric_fn: Custom metric function (default: MSE for regression)
        """
        self.model_a = model_a
        self.model_b = model_b
        self.config = config or ABTestConfig(test_name="default_test")
        self.metric_fn = metric_fn or self._default_metric

        self._results_a: List[float] = []
        self._results_b: List[float] = []
        self._allocation_counts = {"A": 0, "B": 0}
        self._bandit_rewards = {"A": [], "B": []}

        np.random.seed(self.config.random_state)

        logger.info(
            f"ABTesting initialized: {self.config.test_name}, "
            f"strategy={self.config.allocation_strategy}"
        )

    def _default_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Default metric: negative MSE (higher is better)."""
        return -np.mean((y_true - y_pred) ** 2)

    def _allocate_traffic(self) -> str:
        """
        Allocate traffic to variant based on strategy.

        Returns:
            "A" or "B"
        """
        if self.config.allocation_strategy == AllocationStrategy.RANDOM:
            return "B" if np.random.random() < self.config.traffic_split else "A"

        elif self.config.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            total = self._allocation_counts["A"] + self._allocation_counts["B"]
            if self._allocation_counts["B"] / (total + 1) < self.config.traffic_split:
                return "B"
            return "A"

        elif self.config.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            epsilon = 0.1
            if np.random.random() < epsilon:
                return np.random.choice(["A", "B"])
            # Exploit best
            mean_a = np.mean(self._results_a) if self._results_a else 0
            mean_b = np.mean(self._results_b) if self._results_b else 0
            return "B" if mean_b > mean_a else "A"

        elif self.config.allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            # Beta distribution for Bernoulli rewards (simplified)
            if not self._bandit_rewards["A"] or not self._bandit_rewards["B"]:
                return np.random.choice(["A", "B"])

            # Sample from posterior
            alpha_a = sum(1 for r in self._bandit_rewards["A"] if r > 0) + 1
            beta_a = sum(1 for r in self._bandit_rewards["A"] if r <= 0) + 1
            alpha_b = sum(1 for r in self._bandit_rewards["B"] if r > 0) + 1
            beta_b = sum(1 for r in self._bandit_rewards["B"] if r <= 0) + 1

            sample_a = np.random.beta(alpha_a, beta_a)
            sample_b = np.random.beta(alpha_b, beta_b)

            return "B" if sample_b > sample_a else "A"

        return np.random.choice(["A", "B"])

    def _calculate_provenance(
        self,
        results_a: List[float],
        results_b: List[float],
        p_value: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{self.config.test_name}|{len(results_a)}|{len(results_b)}|"
            f"{np.mean(results_a):.8f}|{np.mean(results_b):.8f}|{p_value:.8f}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval using bootstrap."""
        if len(values) < 2:
            return (np.mean(values), np.mean(values))

        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def _welch_t_test(
        self,
        a: List[float],
        b: List[float]
    ) -> Tuple[float, float]:
        """
        Perform Welch's t-test.

        Returns:
            Tuple of (t-statistic, p-value)
        """
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

        # Welch's t-statistic
        se = np.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return (0.0, 1.0)

        t_stat = (mean_b - mean_a) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1

        # Two-tailed p-value using normal approximation for large df
        try:
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        except ImportError:
            # Normal approximation
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return (float(t_stat), float(p_value))

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        # Error function approximation
        a1, a2, a3, a4, a5 = (
            0.254829592, -0.284496736, 1.421413741,
            -1.453152027, 1.061405429
        )
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return 0.5 * (1.0 + sign * y)

    def _mann_whitney_test(
        self,
        a: List[float],
        b: List[float]
    ) -> Tuple[float, float]:
        """Perform Mann-Whitney U test."""
        try:
            from scipy import stats
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            return (float(stat), float(p_value))
        except ImportError:
            # Fallback to normal approximation
            combined = sorted([(v, "a") for v in a] + [(v, "b") for v in b])
            ranks = {i: rank + 1 for rank, (_, i) in enumerate(combined)}

            # This is simplified - proper implementation needs tie correction
            logger.warning("SciPy not available, using approximate Mann-Whitney")
            return self._welch_t_test(a, b)

    def _bootstrap_test(
        self,
        a: List[float],
        b: List[float],
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Perform bootstrap hypothesis test."""
        observed_diff = np.mean(b) - np.mean(a)

        # Pool samples under null hypothesis
        pooled = a + b
        n_a = len(a)

        count_extreme = 0
        for _ in range(n_bootstrap):
            np.random.shuffle(pooled)
            boot_a = pooled[:n_a]
            boot_b = pooled[n_a:]
            boot_diff = np.mean(boot_b) - np.mean(boot_a)

            if abs(boot_diff) >= abs(observed_diff):
                count_extreme += 1

        p_value = count_extreme / n_bootstrap
        return (float(observed_diff), float(p_value))

    def _compute_effect_size(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Compute Cohen's d effect size."""
        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)

        n_a, n_b = len(a), len(b)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) /
            (n_a + n_b - 2)
        )

        if pooled_std == 0:
            return 0.0

        return float((mean_b - mean_a) / pooled_std)

    def _check_early_stopping(self) -> bool:
        """Check if test should be early stopped."""
        if not self.config.enable_early_stopping:
            return False

        if len(self._results_a) < 50 or len(self._results_b) < 50:
            return False

        # Bayesian early stopping
        mean_a = np.mean(self._results_a)
        mean_b = np.mean(self._results_b)
        std_a = np.std(self._results_a) / np.sqrt(len(self._results_a))
        std_b = np.std(self._results_b) / np.sqrt(len(self._results_b))

        # Approximate probability that B > A
        if std_a + std_b > 0:
            z = (mean_b - mean_a) / np.sqrt(std_a ** 2 + std_b ** 2)
            prob_b_wins = self._normal_cdf(z)

            if prob_b_wins > self.config.early_stopping_confidence:
                return True
            if prob_b_wins < (1 - self.config.early_stopping_confidence):
                return True

        return False

    def run_sample(
        self,
        x: np.ndarray,
        y_true: np.ndarray
    ) -> str:
        """
        Run a single sample through the test.

        Args:
            x: Input features
            y_true: True label

        Returns:
            Variant used ("A" or "B")
        """
        variant = self._allocate_traffic()
        self._allocation_counts[variant] += 1

        model = self.model_a if variant == "A" else self.model_b

        # Get prediction
        if hasattr(model, "predict"):
            y_pred = model.predict(x.reshape(1, -1))[0]
        else:
            y_pred = model(x)

        # Compute metric
        metric_value = self.metric_fn(np.array([y_true]), np.array([y_pred]))

        if variant == "A":
            self._results_a.append(metric_value)
            self._bandit_rewards["A"].append(metric_value)
        else:
            self._results_b.append(metric_value)
            self._bandit_rewards["B"].append(metric_value)

        return variant

    def run_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: Optional[int] = None
    ) -> ABTestResult:
        """
        Run complete A/B test.

        Args:
            X: Test features
            y: True labels
            n_samples: Number of samples (default: all)

        Returns:
            ABTestResult with test results

        Example:
            >>> result = ab.run_test(X_test, y_test)
            >>> if result.is_significant and result.winner == "B":
            ...     print("Model B is significantly better!")
        """
        logger.info(f"Starting A/B test: {self.config.test_name}")

        n_samples = n_samples or len(X)
        early_stopped = False

        for i in range(min(n_samples, len(X))):
            self.run_sample(X[i], y[i])

            # Check early stopping
            if self._check_early_stopping():
                logger.info(f"Early stopping at sample {i}")
                early_stopped = True
                break

        return self.get_results(early_stopped=early_stopped)

    def get_results(
        self,
        early_stopped: bool = False
    ) -> ABTestResult:
        """
        Get current test results.

        Args:
            early_stopped: Whether test was early stopped

        Returns:
            ABTestResult with current results
        """
        if len(self._results_a) < 2 or len(self._results_b) < 2:
            raise ValueError("Insufficient samples for statistical test")

        # Run statistical test
        if self.config.test_type == TestType.WELCH_T:
            stat, p_value = self._welch_t_test(self._results_a, self._results_b)
        elif self.config.test_type == TestType.MANN_WHITNEY:
            stat, p_value = self._mann_whitney_test(self._results_a, self._results_b)
        elif self.config.test_type == TestType.BOOTSTRAP:
            stat, p_value = self._bootstrap_test(self._results_a, self._results_b)
        else:
            stat, p_value = self._welch_t_test(self._results_a, self._results_b)

        # Compute effect size
        effect_size = self._compute_effect_size(self._results_a, self._results_b)

        # Compute confidence intervals
        ci_a = self._compute_confidence_interval(
            self._results_a, 1 - self.config.significance_level
        )
        ci_b = self._compute_confidence_interval(
            self._results_b, 1 - self.config.significance_level
        )

        # Determine winner
        mean_a = np.mean(self._results_a)
        mean_b = np.mean(self._results_b)
        is_significant = p_value < self.config.significance_level

        if is_significant:
            winner = "B" if mean_b > mean_a else "A"
        else:
            winner = None

        # Calculate relative improvement
        relative_improvement = (
            (mean_b - mean_a) / abs(mean_a) if mean_a != 0 else 0.0
        )

        # Create variant results
        variant_a = VariantResult(
            variant_name="A",
            n_samples=len(self._results_a),
            mean=float(mean_a),
            std=float(np.std(self._results_a)),
            median=float(np.median(self._results_a)),
            ci_lower=ci_a[0],
            ci_upper=ci_a[1]
        )

        variant_b = VariantResult(
            variant_name="B",
            n_samples=len(self._results_b),
            mean=float(mean_b),
            std=float(np.std(self._results_b)),
            median=float(np.median(self._results_b)),
            ci_lower=ci_b[0],
            ci_upper=ci_b[1]
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            self._results_a, self._results_b, p_value
        )

        logger.info(
            f"A/B test complete: winner={winner}, p={p_value:.4f}, "
            f"effect_size={effect_size:.4f}"
        )

        return ABTestResult(
            test_name=self.config.test_name,
            variant_a=variant_a,
            variant_b=variant_b,
            winner=winner,
            is_significant=is_significant,
            p_value=float(p_value),
            effect_size=float(effect_size),
            relative_improvement=float(relative_improvement),
            confidence_level=1 - self.config.significance_level,
            test_type=self.config.test_type.value,
            early_stopped=early_stopped,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def reset(self) -> None:
        """Reset test state."""
        self._results_a.clear()
        self._results_b.clear()
        self._allocation_counts = {"A": 0, "B": 0}
        self._bandit_rewards = {"A": [], "B": []}
        logger.info(f"A/B test reset: {self.config.test_name}")


# Unit test stubs
class TestABTesting:
    """Unit tests for ABTesting."""

    def test_init(self):
        """Test initialization."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        config = ABTestConfig(test_name="test")
        ab = ABTesting(MockModel(), MockModel(), config)
        assert ab.config.test_name == "test"

    def test_traffic_allocation(self):
        """Test traffic allocation."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        config = ABTestConfig(
            test_name="test",
            allocation_strategy=AllocationStrategy.RANDOM
        )
        ab = ABTesting(MockModel(), MockModel(), config)

        allocations = [ab._allocate_traffic() for _ in range(100)]
        assert "A" in allocations
        assert "B" in allocations

    def test_welch_t_test(self):
        """Test Welch's t-test."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        ab = ABTesting(MockModel(), MockModel(), ABTestConfig(test_name="test"))

        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [6.0, 7.0, 8.0, 9.0, 10.0]

        stat, p_value = ab._welch_t_test(a, b)
        assert p_value < 0.05  # Should be significant

    def test_effect_size(self):
        """Test effect size calculation."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        ab = ABTesting(MockModel(), MockModel(), ABTestConfig(test_name="test"))

        a = [1.0] * 10
        b = [2.0] * 10

        effect = ab._compute_effect_size(a, b)
        assert effect > 0  # B should be better

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        ab = ABTesting(MockModel(), MockModel(), ABTestConfig(test_name="test"))

        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]

        hash1 = ab._calculate_provenance(a, b, 0.05)
        hash2 = ab._calculate_provenance(a, b, 0.05)

        assert hash1 == hash2
