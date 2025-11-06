"""
Tests for Uncertainty Quantification Module

Tests cover:
- Category-based uncertainty estimation
- Tier multipliers
- Uncertainty quantification
- Simple propagation
- Chain propagation
- Confidence interval calculation
- Bounds application

Version: 1.0.0
Date: 2025-10-30
"""

import pytest
import numpy as np
from services.methodologies import (
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    UncertaintyResult,
    PedigreeScore,
    quantify_uncertainty,
    propagate_uncertainty,
    DistributionType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def quantifier():
    """Create an UncertaintyQuantifier instance."""
    return UncertaintyQuantifier()


@pytest.fixture
def good_pedigree():
    """Create good quality pedigree score."""
    return PedigreeScore(
        reliability=1, completeness=2, temporal=1,
        geographical=2, technological=1
    )


# ============================================================================
# CATEGORY UNCERTAINTY TESTS
# ============================================================================

def test_get_category_uncertainty_known(quantifier):
    """Test getting uncertainty for known categories."""
    assert quantifier.get_category_uncertainty("metals") == 0.15
    assert quantifier.get_category_uncertainty("electricity") == 0.10
    assert quantifier.get_category_uncertainty("plastics") == 0.20
    assert quantifier.get_category_uncertainty("textiles") == 0.30


def test_get_category_uncertainty_case_insensitive(quantifier):
    """Test case-insensitive category matching."""
    assert quantifier.get_category_uncertainty("METALS") == 0.15
    assert quantifier.get_category_uncertainty("Metals") == 0.15
    assert quantifier.get_category_uncertainty("metals") == 0.15


def test_get_category_uncertainty_partial_match(quantifier):
    """Test partial category matching."""
    # Should match "electricity"
    uncertainty = quantifier.get_category_uncertainty("electricity_grid")
    assert uncertainty == 0.10


def test_get_category_uncertainty_unknown(quantifier):
    """Test unknown category returns default."""
    uncertainty = quantifier.get_category_uncertainty("unknown_category")
    assert uncertainty == 0.50  # Default uncertainty


# ============================================================================
# TIER MULTIPLIER TESTS
# ============================================================================

def test_apply_tier_multiplier(quantifier):
    """Test tier multiplier application."""
    base = 0.10

    # Tier 1: no increase
    tier1 = quantifier.apply_tier_multiplier(base, 1)
    assert tier1 == 0.10

    # Tier 2: 50% increase
    tier2 = quantifier.apply_tier_multiplier(base, 2)
    assert tier2 == 0.15

    # Tier 3: 100% increase
    tier3 = quantifier.apply_tier_multiplier(base, 3)
    assert tier3 == 0.20


def test_apply_tier_multiplier_invalid(quantifier):
    """Test invalid tier uses tier 3 multiplier."""
    base = 0.10
    result = quantifier.apply_tier_multiplier(base, 5)
    assert result == 0.20  # Same as tier 3


# ============================================================================
# BOUNDS APPLICATION TESTS
# ============================================================================

def test_apply_bounds_within_range(quantifier):
    """Test bounds application for values within range."""
    uncertainty = quantifier.apply_bounds(0.25)
    assert uncertainty == 0.25  # Unchanged


def test_apply_bounds_below_floor(quantifier):
    """Test bounds application for values below floor."""
    uncertainty = quantifier.apply_bounds(0.005)  # Below min (0.01)
    assert uncertainty == 0.01  # Floor applied


def test_apply_bounds_above_ceiling(quantifier):
    """Test bounds application for values above ceiling."""
    uncertainty = quantifier.apply_bounds(10.0)  # Above max (5.0)
    assert uncertainty == 5.0  # Ceiling applied


# ============================================================================
# UNCERTAINTY ESTIMATION TESTS
# ============================================================================

def test_estimate_uncertainty_custom(quantifier):
    """Test uncertainty estimation with custom value."""
    uncertainty = quantifier.estimate_uncertainty(custom_uncertainty=0.25)
    assert uncertainty == 0.25


def test_estimate_uncertainty_pedigree(quantifier, good_pedigree):
    """Test uncertainty estimation from pedigree score."""
    uncertainty = quantifier.estimate_uncertainty(pedigree_score=good_pedigree)
    assert uncertainty > 0
    assert uncertainty < 0.5  # Good quality should have low uncertainty


def test_estimate_uncertainty_category(quantifier):
    """Test uncertainty estimation from category."""
    uncertainty = quantifier.estimate_uncertainty(category="electricity")
    assert uncertainty == 0.10


def test_estimate_uncertainty_category_tier(quantifier):
    """Test uncertainty estimation from category and tier."""
    # Electricity base: 0.10, Tier 2 multiplier: 1.5
    uncertainty = quantifier.estimate_uncertainty(
        category="electricity",
        tier=2
    )
    assert uncertainty == 0.15


def test_estimate_uncertainty_default(quantifier):
    """Test uncertainty estimation with no inputs."""
    uncertainty = quantifier.estimate_uncertainty()
    assert uncertainty == 0.50  # Default


def test_estimate_uncertainty_priority(quantifier, good_pedigree):
    """Test priority order in uncertainty estimation."""
    # Custom should override everything
    uncertainty = quantifier.estimate_uncertainty(
        custom_uncertainty=0.25,
        pedigree_score=good_pedigree,
        category="electricity",
    )
    assert uncertainty == 0.25


# ============================================================================
# CONFIDENCE INTERVAL TESTS
# ============================================================================

def test_calculate_confidence_interval_normal(quantifier):
    """Test confidence interval for normal distribution."""
    lower, upper = quantifier.calculate_confidence_interval(
        mean=100,
        std_dev=10,
        confidence_level=0.95,
        distribution=DistributionType.NORMAL,
    )

    # 95% CI: mean ± 1.96 * std_dev
    assert lower == pytest.approx(100 - 1.96 * 10, rel=0.01)
    assert upper == pytest.approx(100 + 1.96 * 10, rel=0.01)


def test_calculate_confidence_interval_lognormal(quantifier):
    """Test confidence interval for lognormal distribution."""
    lower, upper = quantifier.calculate_confidence_interval(
        mean=100,
        std_dev=15,
        confidence_level=0.95,
        distribution=DistributionType.LOGNORMAL,
    )

    # Should be asymmetric around mean
    assert lower < 100 < upper
    assert (100 - lower) < (upper - 100)  # More spread above


def test_calculate_confidence_interval_different_levels(quantifier):
    """Test different confidence levels."""
    lower_95, upper_95 = quantifier.calculate_confidence_interval(
        mean=100, std_dev=10, confidence_level=0.95
    )
    lower_90, upper_90 = quantifier.calculate_confidence_interval(
        mean=100, std_dev=10, confidence_level=0.90
    )

    # 95% CI should be wider than 90% CI
    assert (upper_95 - lower_95) > (upper_90 - lower_90)


def test_confidence_interval_non_negative(quantifier):
    """Test that confidence intervals are non-negative."""
    lower, upper = quantifier.calculate_confidence_interval(
        mean=10,
        std_dev=15,  # Large uncertainty
        confidence_level=0.95,
        distribution=DistributionType.NORMAL,
    )

    # Lower bound should be clamped to 0 for emissions
    assert lower >= 0


# ============================================================================
# UNCERTAINTY QUANTIFICATION TESTS
# ============================================================================

def test_quantify_uncertainty_basic(quantifier):
    """Test basic uncertainty quantification."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        category="electricity",
        tier=1,
    )

    assert isinstance(result, UncertaintyResult)
    assert result.mean == 1000.0
    assert result.std_dev > 0
    assert result.relative_std_dev == 0.10  # Electricity uncertainty
    assert result.confidence_95_lower < result.mean < result.confidence_95_upper


def test_quantify_uncertainty_with_pedigree(quantifier, good_pedigree):
    """Test uncertainty quantification with pedigree score."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        pedigree_score=good_pedigree,
    )

    assert result.pedigree_score == good_pedigree
    assert "pedigree_matrix" in result.uncertainty_sources


def test_quantify_uncertainty_all_parameters(quantifier, good_pedigree):
    """Test uncertainty quantification with all parameters."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        category="electricity",
        tier=2,
        pedigree_score=good_pedigree,
        custom_uncertainty=0.12,
        distribution=DistributionType.LOGNORMAL,
    )

    assert result.mean == 1000.0
    assert result.relative_std_dev == 0.12  # Custom has priority
    assert result.distribution_type == DistributionType.LOGNORMAL


def test_quantify_uncertainty_sources(quantifier):
    """Test uncertainty source tracking."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        category="electricity",
        tier=2,
    )

    assert "category_electricity" in result.uncertainty_sources
    assert "tier_2" in result.uncertainty_sources


# ============================================================================
# SIMPLE PROPAGATION TESTS
# ============================================================================

def test_propagate_simple_analytical(quantifier):
    """Test simple propagation with analytical method."""
    result = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="analytical",
    )

    # Expected mean: 1000 × 2.5 = 2500
    assert result.mean == 2500

    # Check uncertainty propagated correctly
    # CV² = (0.1)² + (0.15)² = 0.0325
    # CV = sqrt(0.0325) ≈ 0.18
    expected_cv = np.sqrt(0.1**2 + 0.15**2)
    assert result.relative_std_dev == pytest.approx(expected_cv, rel=0.01)

    assert result.methodology == "analytical_propagation"


def test_propagate_simple_monte_carlo(quantifier):
    """Test simple propagation with Monte Carlo method."""
    result = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="monte_carlo",
    )

    # Expected mean: 1000 × 2.5 = 2500
    assert result.mean == pytest.approx(2500, rel=0.05)

    # Check uncertainty
    expected_cv = np.sqrt(0.1**2 + 0.15**2)
    assert result.relative_std_dev == pytest.approx(expected_cv, rel=0.1)

    assert result.methodology == "monte_carlo_propagation"


def test_propagate_simple_consistency(quantifier):
    """Test analytical and Monte Carlo give similar results."""
    analytical = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="analytical",
    )

    monte_carlo = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="monte_carlo",
    )

    # Results should be similar
    assert analytical.mean == pytest.approx(monte_carlo.mean, rel=0.05)
    assert analytical.std_dev == pytest.approx(monte_carlo.std_dev, rel=0.1)


# ============================================================================
# CHAIN PROPAGATION TESTS
# ============================================================================

def test_propagate_chain_multiply(quantifier):
    """Test chain propagation with multiplication."""
    # Calculate: 100 × 2.5 = 250
    result = quantifier.propagate_chain(
        values=[100, 2.5],
        uncertainties=[0.1, 0.15],
        operations=["*"],
    )

    assert result.mean == 250
    assert result.std_dev > 0


def test_propagate_chain_add(quantifier):
    """Test chain propagation with addition."""
    # Calculate: 100 + 50 = 150
    result = quantifier.propagate_chain(
        values=[100, 50],
        uncertainties=[0.1, 0.2],
        operations=["+"],
    )

    assert result.mean == 150
    assert result.std_dev > 0


def test_propagate_chain_combined(quantifier):
    """Test chain propagation with mixed operations."""
    # Calculate: (100 × 2.5) + 50 = 300
    result = quantifier.propagate_chain(
        values=[100, 2.5, 50],
        uncertainties=[0.1, 0.15, 0.2],
        operations=["*", "+"],
    )

    assert result.mean == 300
    assert result.std_dev > 0


def test_propagate_chain_validation(quantifier):
    """Test chain propagation input validation."""
    # Mismatched lengths
    with pytest.raises(ValueError):
        quantifier.propagate_chain(
            values=[100, 2.5],
            uncertainties=[0.1],  # Too short
            operations=["*"],
        )

    # Wrong number of operations
    with pytest.raises(ValueError):
        quantifier.propagate_chain(
            values=[100, 2.5, 50],
            uncertainties=[0.1, 0.15, 0.2],
            operations=["*"],  # Should be 2 operations
        )

    # Invalid operation
    with pytest.raises(ValueError):
        quantifier.propagate_chain(
            values=[100, 2.5],
            uncertainties=[0.1, 0.15],
            operations=["/"],  # Not supported
        )


# ============================================================================
# SENSITIVITY ANALYZER TESTS
# ============================================================================

def test_sensitivity_analyzer_init():
    """Test sensitivity analyzer initialization."""
    analyzer = SensitivityAnalyzer()
    assert analyzer is not None


def test_analyze_contribution():
    """Test contribution analysis."""
    analyzer = SensitivityAnalyzer()

    parameters = {
        "activity": 1000.0,
        "factor": 2.5,
    }
    result = 2500.0

    contributions = analyzer.analyze_contribution(parameters, result)

    assert "activity" in contributions
    assert "factor" in contributions

    # Contributions should sum to 1.0
    total = sum(contributions.values())
    assert total == pytest.approx(1.0, rel=0.01)

    # Larger value should have larger contribution
    assert contributions["activity"] > contributions["factor"]


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_quantify_uncertainty_convenience():
    """Test convenience function for uncertainty quantification."""
    result = quantify_uncertainty(
        mean=1000.0,
        category="electricity",
        tier=1,
    )

    assert isinstance(result, UncertaintyResult)
    assert result.mean == 1000.0


def test_propagate_uncertainty_convenience():
    """Test convenience function for uncertainty propagation."""
    result = propagate_uncertainty(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
    )

    assert isinstance(result, UncertaintyResult)
    assert result.mean == 2500


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_zero_uncertainty(quantifier):
    """Test with zero uncertainty."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        custom_uncertainty=0.0,
    )

    # Should apply minimum floor
    assert result.relative_std_dev >= 0.01


def test_very_high_uncertainty(quantifier):
    """Test with very high uncertainty."""
    result = quantifier.quantify_uncertainty(
        mean=1000.0,
        custom_uncertainty=10.0,  # 1000%
    )

    # Should apply maximum ceiling
    assert result.relative_std_dev <= 5.0


def test_zero_mean(quantifier):
    """Test uncertainty with zero mean."""
    result = quantifier.quantify_uncertainty(
        mean=0.0,
        category="electricity",
    )

    assert result.mean == 0.0
    assert result.std_dev >= 0


def test_negative_mean(quantifier):
    """Test uncertainty with negative mean (edge case)."""
    # Emissions shouldn't be negative, but test robustness
    result = quantifier.quantify_uncertainty(
        mean=-100.0,
        category="electricity",
    )

    # Should handle gracefully
    assert result.mean == -100.0


def test_propagate_zero_uncertainty(quantifier):
    """Test propagation with zero uncertainties."""
    result = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.0,
        factor_mean=2.5,
        factor_uncertainty=0.0,
        method="analytical",
    )

    assert result.mean == 2500
    # With no uncertainty, std_dev should be minimal
    assert result.std_dev < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
