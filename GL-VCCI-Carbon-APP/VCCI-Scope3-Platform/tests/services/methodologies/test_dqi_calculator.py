"""
Tests for Data Quality Index (DQI) Calculator

Tests cover:
- Source quality scoring
- Tier scoring
- Pedigree to DQI conversion
- Composite DQI calculation
- Quality label assignment
- DQI report generation
- DQI comparison

Version: 1.0.0
Date: 2025-10-30
"""

import pytest
from services.methodologies import (
    DQICalculator,
    DQIScore,
    PedigreeScore,
    calculate_dqi,
    assess_factor_quality,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator():
    """Create a DQICalculator instance."""
    return DQICalculator()


@pytest.fixture
def excellent_pedigree():
    """Create excellent quality pedigree score."""
    return PedigreeScore(
        reliability=1, completeness=1, temporal=1,
        geographical=1, technological=1
    )


@pytest.fixture
def good_pedigree():
    """Create good quality pedigree score."""
    return PedigreeScore(
        reliability=1, completeness=2, temporal=1,
        geographical=2, technological=1
    )


@pytest.fixture
def fair_pedigree():
    """Create fair quality pedigree score."""
    return PedigreeScore(
        reliability=3, completeness=3, temporal=3,
        geographical=3, technological=3
    )


# ============================================================================
# SOURCE QUALITY TESTS
# ============================================================================

def test_get_source_quality_score_primary(calculator):
    """Test source quality scores for primary data."""
    assert calculator.get_source_quality_score("primary_measured") == 100.0
    assert calculator.get_source_quality_score("primary_calculated") == 95.0


def test_get_source_quality_score_databases(calculator):
    """Test source quality scores for databases."""
    assert calculator.get_source_quality_score("ecoinvent") == 90.0
    assert calculator.get_source_quality_score("gabi") == 90.0
    assert calculator.get_source_quality_score("idemat") == 85.0
    assert calculator.get_source_quality_score("defra") == 80.0


def test_get_source_quality_score_estimates(calculator):
    """Test source quality scores for estimates."""
    assert calculator.get_source_quality_score("expert_estimate") == 60.0
    assert calculator.get_source_quality_score("proxy") == 50.0
    assert calculator.get_source_quality_score("unknown") == 30.0


def test_get_source_quality_score_case_insensitive(calculator):
    """Test case-insensitive source matching."""
    assert calculator.get_source_quality_score("ECOINVENT") == 90.0
    assert calculator.get_source_quality_score("EcoInvent") == 90.0
    assert calculator.get_source_quality_score("ecoinvent") == 90.0


def test_get_source_quality_score_partial_match(calculator):
    """Test partial matching for sources."""
    # Should match "ecoinvent"
    score = calculator.get_source_quality_score("ecoinvent v3.9")
    assert score == 90.0


def test_get_source_quality_score_unknown(calculator):
    """Test unknown source returns default."""
    score = calculator.get_source_quality_score("some_random_database")
    assert score == 30.0  # Default unknown score


# ============================================================================
# TIER SCORING TESTS
# ============================================================================

def test_get_tier_score(calculator):
    """Test tier scoring."""
    assert calculator.get_tier_score(1) == 100.0  # Primary
    assert calculator.get_tier_score(2) == 80.0   # Secondary
    assert calculator.get_tier_score(3) == 50.0   # Estimated


def test_get_tier_score_invalid(calculator):
    """Test invalid tier returns tier 3 score."""
    score = calculator.get_tier_score(4)
    assert score == 50.0  # Same as tier 3


# ============================================================================
# PEDIGREE TO DQI TESTS
# ============================================================================

def test_pedigree_to_dqi_score_excellent(calculator, excellent_pedigree):
    """Test pedigree to DQI conversion for excellent data."""
    dqi = calculator.pedigree_to_dqi_score(excellent_pedigree)
    assert dqi == 100.0


def test_pedigree_to_dqi_score_good(calculator, good_pedigree):
    """Test pedigree to DQI conversion for good data."""
    dqi = calculator.pedigree_to_dqi_score(good_pedigree)
    # Average score = 1.4 → DQI ≈ 92.5
    assert 90 <= dqi <= 95


def test_pedigree_to_dqi_score_fair(calculator, fair_pedigree):
    """Test pedigree to DQI conversion for fair data."""
    dqi = calculator.pedigree_to_dqi_score(fair_pedigree)
    # Average score = 3.0 → DQI = 75
    assert 70 <= dqi <= 80


def test_pedigree_to_dqi_interpolation(calculator):
    """Test interpolation in pedigree to DQI conversion."""
    # Test a score between mapping points
    pedigree = PedigreeScore(
        reliability=2, completeness=2, temporal=2,
        geographical=2, technological=2
    )
    # Average = 2.0
    dqi_interp = calculator.pedigree_to_dqi_score(pedigree, use_interpolation=True)
    dqi_direct = calculator.pedigree_to_dqi_score(pedigree, use_interpolation=False)

    # Both should be similar
    assert abs(dqi_interp - dqi_direct) < 5.0


# ============================================================================
# COMPOSITE DQI TESTS
# ============================================================================

def test_calculate_composite_dqi_all_components(calculator, good_pedigree):
    """Test composite DQI with all components."""
    dqi = calculator.calculate_composite_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
    )

    # Should be high quality (all components are good)
    assert dqi >= 85.0
    assert dqi <= 100.0


def test_calculate_composite_dqi_pedigree_only(calculator, good_pedigree):
    """Test composite DQI with pedigree only."""
    dqi = calculator.calculate_composite_dqi(pedigree_score=good_pedigree)

    # Should be normalized to pedigree score
    pedigree_dqi = calculator.pedigree_to_dqi_score(good_pedigree)
    assert dqi == pytest.approx(pedigree_dqi, rel=0.01)


def test_calculate_composite_dqi_source_and_tier(calculator):
    """Test composite DQI with source and tier only."""
    dqi = calculator.calculate_composite_dqi(
        factor_source="ecoinvent",
        data_tier=1,
    )

    # Should be weighted average of source (90) and tier (100)
    # Default weights: source=0.3, tier=0.2
    # Normalized: source=0.6, tier=0.4
    expected = 0.6 * 90 + 0.4 * 100
    assert dqi == pytest.approx(expected, rel=0.01)


def test_calculate_composite_dqi_custom_weights(calculator, good_pedigree):
    """Test composite DQI with custom weights."""
    custom_weights = {
        "pedigree": 0.6,
        "source": 0.3,
        "tier": 0.1,
    }

    dqi = calculator.calculate_composite_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=2,
        custom_weights=custom_weights,
    )

    assert 80 <= dqi <= 95


def test_calculate_composite_dqi_no_components(calculator):
    """Test composite DQI with no components."""
    dqi = calculator.calculate_composite_dqi()
    assert dqi == 0.0


# ============================================================================
# QUALITY LABEL TESTS
# ============================================================================

def test_get_quality_label(calculator):
    """Test quality label assignment."""
    assert calculator.get_quality_label(95) == "Excellent"
    assert calculator.get_quality_label(85) == "Good"
    assert calculator.get_quality_label(65) == "Fair"
    assert calculator.get_quality_label(40) == "Poor"


def test_get_quality_label_boundaries(calculator):
    """Test quality label boundaries."""
    assert calculator.get_quality_label(90.0) == "Excellent"  # Exactly 90
    assert calculator.get_quality_label(89.9) == "Good"       # Just below 90
    assert calculator.get_quality_label(70.0) == "Good"       # Exactly 70
    assert calculator.get_quality_label(69.9) == "Fair"       # Just below 70
    assert calculator.get_quality_label(50.0) == "Fair"       # Exactly 50
    assert calculator.get_quality_label(49.9) == "Poor"       # Just below 50


# ============================================================================
# DQI CALCULATION TESTS
# ============================================================================

def test_calculate_dqi_complete(calculator, good_pedigree):
    """Test complete DQI calculation."""
    dqi = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
        assessed_by="test_user",
        notes="Test assessment",
    )

    assert isinstance(dqi, DQIScore)
    assert dqi.score >= 85.0
    assert dqi.quality_label in ["Excellent", "Good"]
    assert dqi.pedigree_score == good_pedigree
    assert dqi.factor_source == "ecoinvent"
    assert dqi.data_tier == 1
    assert dqi.assessed_by == "test_user"
    assert dqi.notes == "Test assessment"


def test_calculate_dqi_components(calculator, good_pedigree):
    """Test DQI component contributions."""
    dqi = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
    )

    assert dqi.pedigree_contribution > 0
    assert dqi.source_contribution > 0
    assert dqi.tier_contribution > 0

    # Source contribution should be 90 (ecoinvent)
    assert dqi.source_contribution == 90.0

    # Tier contribution should be 100 (tier 1)
    assert dqi.tier_contribution == 100.0


def test_calculate_dqi_tier_penalty(calculator, good_pedigree):
    """Test tier penalty application."""
    # Tier 1 (no penalty)
    dqi_tier1 = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
    )

    # Tier 2 (10 point penalty)
    dqi_tier2 = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=2,
    )

    # Tier 3 (20 point penalty)
    dqi_tier3 = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=3,
    )

    # DQI should decrease with lower tiers
    assert dqi_tier1.score > dqi_tier2.score > dqi_tier3.score


# ============================================================================
# DQI REPORT TESTS
# ============================================================================

def test_generate_dqi_report(calculator, good_pedigree):
    """Test DQI report generation."""
    dqi = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
    )

    report = calculator.generate_dqi_report(dqi)

    assert "dqi_score" in report
    assert "quality_label" in report
    assert "components" in report
    assert "weights" in report
    assert "metadata" in report
    assert "pedigree_details" in report
    assert "recommendations" in report


def test_dqi_report_recommendations_high_quality(calculator, excellent_pedigree):
    """Test recommendations for high quality data."""
    dqi = calculator.calculate_dqi(
        pedigree_score=excellent_pedigree,
        factor_source="primary_measured",
        data_tier=1,
    )

    report = calculator.generate_dqi_report(dqi)

    # Should have few or no recommendations
    assert len(report["recommendations"]) == 0


def test_dqi_report_recommendations_low_quality(calculator, fair_pedigree):
    """Test recommendations for low quality data."""
    dqi = calculator.calculate_dqi(
        pedigree_score=fair_pedigree,
        factor_source="proxy",
        data_tier=3,
    )

    report = calculator.generate_dqi_report(dqi)

    # Should have multiple recommendations
    assert len(report["recommendations"]) > 0


# ============================================================================
# DQI COMPARISON TESTS
# ============================================================================

def test_compare_dqi(calculator, good_pedigree, excellent_pedigree):
    """Test DQI comparison."""
    dqi1 = calculator.calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="defra",
        data_tier=2,
    )

    dqi2 = calculator.calculate_dqi(
        pedigree_score=excellent_pedigree,
        factor_source="primary_measured",
        data_tier=1,
    )

    comparison = calculator.compare_dqi(dqi1, dqi2)

    assert "scores" in comparison
    assert "quality_labels" in comparison
    assert "components" in comparison

    # DQI2 should be higher than DQI1
    assert comparison["scores"]["dqi2"] > comparison["scores"]["dqi1"]
    assert comparison["scores"]["difference"] > 0

    # Quality should have improved
    if comparison["quality_labels"]["dqi1"] != comparison["quality_labels"]["dqi2"]:
        assert comparison["quality_labels"]["improved"] is True


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_calculate_dqi_convenience(good_pedigree):
    """Test convenience function for DQI calculation."""
    dqi = calculate_dqi(
        pedigree_score=good_pedigree,
        factor_source="ecoinvent",
        data_tier=1,
    )

    assert isinstance(dqi, DQIScore)
    assert dqi.score > 80


def test_assess_factor_quality_convenience():
    """Test convenience function for factor quality assessment."""
    score = assess_factor_quality("ecoinvent", tier=1)

    assert 80 <= score <= 100


def test_assess_factor_quality_different_tiers():
    """Test factor quality with different tiers."""
    score_tier1 = assess_factor_quality("ecoinvent", tier=1)
    score_tier2 = assess_factor_quality("ecoinvent", tier=2)
    score_tier3 = assess_factor_quality("ecoinvent", tier=3)

    # Scores should decrease with lower tiers
    assert score_tier1 > score_tier2 > score_tier3


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_dqi_score_bounds(calculator, excellent_pedigree):
    """Test DQI scores are within valid bounds."""
    dqi = calculator.calculate_dqi(
        pedigree_score=excellent_pedigree,
        factor_source="primary_measured",
        data_tier=1,
    )

    assert 0 <= dqi.score <= 100
    assert 0 <= dqi.pedigree_contribution <= 100
    assert 0 <= dqi.source_contribution <= 100
    assert 0 <= dqi.tier_contribution <= 100


def test_dqi_with_missing_pedigree(calculator):
    """Test DQI calculation without pedigree score."""
    dqi = calculator.calculate_dqi(
        factor_source="ecoinvent",
        data_tier=2,
    )

    assert dqi.pedigree_contribution == 0.0
    assert dqi.pedigree_score is None
    assert dqi.score > 0  # Should still have source and tier


def test_dqi_with_only_tier(calculator):
    """Test DQI calculation with only tier."""
    dqi = calculator.calculate_dqi(data_tier=1)

    assert dqi.tier_contribution == 100.0
    assert dqi.pedigree_contribution == 0.0
    assert dqi.source_contribution == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
