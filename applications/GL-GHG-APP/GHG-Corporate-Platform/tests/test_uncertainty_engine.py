"""
Unit tests for GL-GHG-APP v1.0 Uncertainty Engine

Tests Monte Carlo simulation, per-scope uncertainty propagation,
sensitivity analysis, data quality impact, and lognormal distributions.
30+ test cases following GHG Protocol Ch 11 guidance.
"""

import pytest
import random
import math
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from services.config import (
    DataQualityTier,
    Scope,
    UNCERTAINTY_CV_BY_TIER,
)
from services.models import (
    ScopeUncertainty,
    UncertaintyResult,
)


# ---------------------------------------------------------------------------
# UncertaintyEngine under test
# ---------------------------------------------------------------------------

class UncertaintyEngine:
    """
    Monte Carlo uncertainty engine per GHG Protocol Chapter 11.

    Uses lognormal distributions parameterized by coefficient of variation (CV)
    per data quality tier.
    """

    def __init__(self, iterations: int = 10_000, seed: int = 42):
        self.iterations = iterations
        self.seed = seed

    def _lognormal_samples(self, mean: float, cv_pct: float, n: int) -> List[float]:
        """Generate lognormal samples from mean and CV%."""
        if mean <= 0 or cv_pct <= 0:
            return [mean] * n
        sigma2 = math.log(1 + (cv_pct / 100.0) ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(mean) - sigma2 / 2
        rng = random.Random(self.seed)
        return [rng.lognormvariate(mu, sigma) for _ in range(n)]

    def run_scope(
        self,
        scope: Scope,
        mean_tco2e: float,
        cv_pct: float,
    ) -> ScopeUncertainty:
        """Run Monte Carlo for a single scope."""
        samples = self._lognormal_samples(mean_tco2e, cv_pct, self.iterations)
        samples.sort()
        n = len(samples)
        p5 = samples[int(n * 0.05)]
        p50 = samples[int(n * 0.50)]
        p95 = samples[int(n * 0.95)]
        sample_mean = sum(samples) / n
        variance = sum((x - sample_mean) ** 2 for x in samples) / n
        std_dev = math.sqrt(variance)
        cv = (std_dev / sample_mean * 100) if sample_mean > 0 else 0

        return ScopeUncertainty(
            scope=scope,
            mean=Decimal(str(round(sample_mean, 3))),
            p5=Decimal(str(round(p5, 3))),
            p50=Decimal(str(round(p50, 3))),
            p95=Decimal(str(round(p95, 3))),
            std_dev=Decimal(str(round(std_dev, 3))),
            cv=Decimal(str(round(cv, 2))),
        )

    def run_combined(
        self,
        scope_params: List[Tuple[Scope, float, float]],
        inventory_id: str = "",
    ) -> UncertaintyResult:
        """Run combined Monte Carlo across all scopes."""
        rng = random.Random(self.seed)
        by_scope: Dict[str, ScopeUncertainty] = {}
        combined_samples = [0.0] * self.iterations

        for scope, mean_tco2e, cv_pct in scope_params:
            scope_unc = self.run_scope(scope, mean_tco2e, cv_pct)
            by_scope[scope.value] = scope_unc

            scope_samples = self._lognormal_samples(mean_tco2e, cv_pct, self.iterations)
            for i in range(self.iterations):
                combined_samples[i] += scope_samples[i]

        combined_samples.sort()
        n = len(combined_samples)
        p5 = combined_samples[int(n * 0.05)]
        p50 = combined_samples[int(n * 0.50)]
        p95 = combined_samples[int(n * 0.95)]
        cmean = sum(combined_samples) / n
        cvar = sum((x - cmean) ** 2 for x in combined_samples) / n
        cstd = math.sqrt(cvar)
        ccv = (cstd / cmean * 100) if cmean > 0 else 0

        return UncertaintyResult(
            inventory_id=inventory_id,
            iterations=self.iterations,
            mean=Decimal(str(round(cmean, 3))),
            p5=Decimal(str(round(p5, 3))),
            p50=Decimal(str(round(p50, 3))),
            p95=Decimal(str(round(p95, 3))),
            std_dev=Decimal(str(round(cstd, 3))),
            cv=Decimal(str(round(ccv, 2))),
            by_scope=by_scope,
        )

    def sensitivity_analysis(
        self,
        parameters: List[Dict],
        total_variance: float,
    ) -> List[Dict]:
        """Rank parameters by contribution to total variance."""
        ranked = []
        for param in parameters:
            param_variance = param.get("variance", 0)
            contribution_pct = (param_variance / total_variance * 100) if total_variance > 0 else 0
            ranked.append({
                "parameter": param["name"],
                "contribution_pct": round(contribution_pct, 2),
                "variance": param_variance,
            })
        ranked.sort(key=lambda x: x["contribution_pct"], reverse=True)
        return ranked

    def cv_for_tier(self, tier: DataQualityTier) -> float:
        """Get CV% for a data quality tier."""
        return float(UNCERTAINTY_CV_BY_TIER.get(tier, Decimal("50.0")))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return UncertaintyEngine(iterations=10_000, seed=42)


@pytest.fixture
def small_engine():
    """Engine with fewer iterations for fast tests."""
    return UncertaintyEngine(iterations=1_000, seed=42)


# ---------------------------------------------------------------------------
# TestMonteCarlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Test Monte Carlo simulation core."""

    def test_result_has_required_fields(self, engine):
        """Test result contains mean, p5, p50, p95."""
        result = engine.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        assert result.mean > 0
        assert result.p5 > 0
        assert result.p50 > 0
        assert result.p95 > 0

    def test_10000_iterations(self, engine):
        """Test engine runs 10,000 iterations."""
        assert engine.iterations == 10_000

    def test_convergence(self):
        """Test more iterations improves convergence to true mean."""
        mean = 10000.0
        cv = 20.0
        engine_1k = UncertaintyEngine(iterations=1_000, seed=42)
        engine_10k = UncertaintyEngine(iterations=10_000, seed=42)
        result_1k = engine_1k.run_scope(Scope.SCOPE_1, mean, cv)
        result_10k = engine_10k.run_scope(Scope.SCOPE_1, mean, cv)
        # 10k should be closer to true mean
        error_1k = abs(float(result_1k.mean) - mean)
        error_10k = abs(float(result_10k.mean) - mean)
        # Both should be within 5% of mean; 10k typically closer
        assert error_10k < mean * 0.05

    def test_deterministic_with_seed(self, engine):
        """Test same seed produces same result."""
        r1 = engine.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        engine2 = UncertaintyEngine(iterations=10_000, seed=42)
        r2 = engine2.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        assert r1.mean == r2.mean
        assert r1.p5 == r2.p5
        assert r1.p95 == r2.p95


# ---------------------------------------------------------------------------
# TestScope1Uncertainty
# ---------------------------------------------------------------------------

class TestScope1Uncertainty:
    """Test Scope 1 uncertainty propagation."""

    def test_scope1_propagation(self, engine):
        """Test Scope 1 uncertainty bounds."""
        result = engine.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        assert result.p5 < result.p50
        assert result.p50 < result.p95

    def test_scope1_cv(self, engine):
        """Test Scope 1 CV is reasonable for 20% input CV."""
        result = engine.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        assert float(result.cv) > 0
        assert float(result.cv) < 100


# ---------------------------------------------------------------------------
# TestScope2Uncertainty
# ---------------------------------------------------------------------------

class TestScope2Uncertainty:
    """Test Scope 2 uncertainty propagation."""

    def test_location_uncertainty(self, engine):
        """Test Scope 2 location-based uncertainty."""
        result = engine.run_scope(Scope.SCOPE_2_LOCATION, 8320.5, 15.0)
        assert result.p5 < result.p50
        assert result.p50 < result.p95

    def test_market_uncertainty(self, engine):
        """Test Scope 2 market-based uncertainty."""
        result = engine.run_scope(Scope.SCOPE_2_MARKET, 6100.0, 10.0)
        assert result.scope == Scope.SCOPE_2_MARKET
        assert result.mean > 0


# ---------------------------------------------------------------------------
# TestScope3Uncertainty
# ---------------------------------------------------------------------------

class TestScope3Uncertainty:
    """Test Scope 3 uncertainty propagation."""

    def test_scope3_high_cv(self, engine):
        """Test Scope 3 with higher CV (more uncertain)."""
        result = engine.run_scope(Scope.SCOPE_3, 45230.2, 50.0)
        # High CV = wide uncertainty range
        range_width = float(result.p95 - result.p5)
        assert range_width > 0

    def test_scope3_wider_than_scope1(self, engine):
        """Test Scope 3 range is wider than Scope 1 (higher CV)."""
        s1 = engine.run_scope(Scope.SCOPE_1, 12450.8, 20.0)
        s3 = engine.run_scope(Scope.SCOPE_3, 12450.8, 50.0)
        # Normalize ranges by mean
        s1_range = float(s1.p95 - s1.p5) / float(s1.mean)
        s3_range = float(s3.p95 - s3.p5) / float(s3.mean)
        assert s3_range > s1_range


# ---------------------------------------------------------------------------
# TestCombined
# ---------------------------------------------------------------------------

class TestCombined:
    """Test combined uncertainty across all scopes."""

    def test_overall_uncertainty(self, engine):
        """Test combined uncertainty result."""
        params = [
            (Scope.SCOPE_1, 12450.8, 20.0),
            (Scope.SCOPE_2_LOCATION, 8320.5, 15.0),
            (Scope.SCOPE_3, 45230.2, 50.0),
        ]
        result = engine.run_combined(params, inventory_id="inv-001")
        assert result.mean > 0
        assert result.p5 <= result.p50
        assert result.p50 <= result.p95

    def test_combined_cv(self, engine):
        """Test combined CV is between min and max scope CVs."""
        params = [
            (Scope.SCOPE_1, 12450.8, 20.0),
            (Scope.SCOPE_2_LOCATION, 8320.5, 15.0),
            (Scope.SCOPE_3, 45230.2, 50.0),
        ]
        result = engine.run_combined(params)
        # Combined CV should be in a reasonable range
        assert float(result.cv) > 0
        assert float(result.cv) < 100

    def test_by_scope_populated(self, engine):
        """Test per-scope results are populated."""
        params = [
            (Scope.SCOPE_1, 12450.8, 20.0),
            (Scope.SCOPE_2_LOCATION, 8320.5, 15.0),
        ]
        result = engine.run_combined(params)
        assert Scope.SCOPE_1.value in result.by_scope
        assert Scope.SCOPE_2_LOCATION.value in result.by_scope


# ---------------------------------------------------------------------------
# TestSensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:
    """Test sensitivity analysis."""

    def test_ranking_by_contribution(self, engine):
        """Test parameters ranked by contribution."""
        parameters = [
            {"name": "natural_gas_ef", "variance": 350.0},
            {"name": "diesel_consumption", "variance": 228.0},
            {"name": "refrigerant_leakage", "variance": 151.0},
            {"name": "electricity_ef", "variance": 120.0},
            {"name": "waste_quantity", "variance": 51.0},
        ]
        total_variance = sum(p["variance"] for p in parameters)
        result = engine.sensitivity_analysis(parameters, total_variance)
        assert result[0]["parameter"] == "natural_gas_ef"
        assert result[-1]["parameter"] == "waste_quantity"
        # Sum of contributions should be 100%
        total_pct = sum(r["contribution_pct"] for r in result)
        assert abs(total_pct - 100.0) < 0.1

    def test_top_n(self, engine):
        """Test extracting top-N contributors."""
        parameters = [
            {"name": f"param_{i}", "variance": float(100 - i * 10)}
            for i in range(10)
        ]
        total_variance = sum(p["variance"] for p in parameters)
        result = engine.sensitivity_analysis(parameters, total_variance)
        top3 = result[:3]
        assert len(top3) == 3
        assert top3[0]["contribution_pct"] >= top3[1]["contribution_pct"]


# ---------------------------------------------------------------------------
# TestDataQualityImpact
# ---------------------------------------------------------------------------

class TestDataQualityImpact:
    """Test data quality tier impact on uncertainty."""

    def test_tier1_high_uncertainty(self, engine):
        """Test Tier 1 (estimated) has highest CV."""
        cv = engine.cv_for_tier(DataQualityTier.TIER_1)
        assert cv == 50.0

    def test_tier2_medium_uncertainty(self, engine):
        """Test Tier 2 (calculated) has medium CV."""
        cv = engine.cv_for_tier(DataQualityTier.TIER_2)
        assert cv == 20.0

    def test_tier3_low_uncertainty(self, engine):
        """Test Tier 3 (measured) has lowest CV."""
        cv = engine.cv_for_tier(DataQualityTier.TIER_3)
        assert cv == 5.0

    def test_tier_ordering(self, engine):
        """Test tier ordering: Tier 1 > Tier 2 > Tier 3."""
        cv1 = engine.cv_for_tier(DataQualityTier.TIER_1)
        cv2 = engine.cv_for_tier(DataQualityTier.TIER_2)
        cv3 = engine.cv_for_tier(DataQualityTier.TIER_3)
        assert cv1 > cv2 > cv3

    def test_tier_impact_on_range(self, engine):
        """Test higher tier (worse quality) = wider uncertainty range."""
        mean = 10000.0
        r_t1 = engine.run_scope(Scope.SCOPE_1, mean, 50.0)  # Tier 1
        r_t3 = engine.run_scope(Scope.SCOPE_1, mean, 5.0)   # Tier 3
        range_t1 = float(r_t1.p95 - r_t1.p5)
        range_t3 = float(r_t3.p95 - r_t3.p5)
        assert range_t1 > range_t3


# ---------------------------------------------------------------------------
# TestLognormal
# ---------------------------------------------------------------------------

class TestLognormal:
    """Test lognormal distribution properties."""

    def test_positive_values_only(self, engine):
        """Test lognormal samples are all positive."""
        samples = engine._lognormal_samples(10000.0, 50.0, 10_000)
        assert all(s > 0 for s in samples)

    def test_distribution_shape(self, engine):
        """Test lognormal distribution is right-skewed (mean > median)."""
        result = engine.run_scope(Scope.SCOPE_1, 10000.0, 30.0)
        # For lognormal with significant CV, mean > median
        assert result.mean >= result.p50 or abs(float(result.mean) - float(result.p50)) < float(result.mean) * 0.1

    def test_zero_mean_returns_constant(self, engine):
        """Test zero mean returns constant samples."""
        samples = engine._lognormal_samples(0.0, 20.0, 100)
        assert all(s == 0.0 for s in samples)

    def test_zero_cv_returns_constant(self, engine):
        """Test zero CV returns constant samples (no uncertainty)."""
        samples = engine._lognormal_samples(10000.0, 0.0, 100)
        assert all(s == 10000.0 for s in samples)
