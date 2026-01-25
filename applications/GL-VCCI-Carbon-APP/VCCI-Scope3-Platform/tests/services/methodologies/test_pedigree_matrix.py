# -*- coding: utf-8 -*-
"""
Tests for ILCD Pedigree Matrix Implementation

Tests cover:
- Pedigree score validation
- Uncertainty factor calculation
- Combined uncertainty calculation
- DQI conversion
- Temporal score assessment
- Quality report generation

Version: 1.0.0
Date: 2025-10-30
"""

import pytest
import math
from services.methodologies import (
    PedigreeScore,
    PedigreeMatrixEvaluator,
    create_pedigree_score,
    assess_data_quality,
    PedigreeIndicator,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def evaluator():
    """Create a PedigreeMatrixEvaluator instance."""
    return PedigreeMatrixEvaluator()


@pytest.fixture
def excellent_pedigree():
    """Create an excellent quality pedigree score."""
    return PedigreeScore(
        reliability=1,
        completeness=1,
        temporal=1,
        geographical=1,
        technological=1,
    )


@pytest.fixture
def good_pedigree():
    """Create a good quality pedigree score."""
    return PedigreeScore(
        reliability=1,
        completeness=2,
        temporal=1,
        geographical=2,
        technological=1,
    )


@pytest.fixture
def fair_pedigree():
    """Create a fair quality pedigree score."""
    return PedigreeScore(
        reliability=3,
        completeness=3,
        temporal=3,
        geographical=3,
        technological=3,
    )


@pytest.fixture
def poor_pedigree():
    """Create a poor quality pedigree score."""
    return PedigreeScore(
        reliability=4,
        completeness=4,
        temporal=5,
        geographical=4,
        technological=5,
    )


# ============================================================================
# PEDIGREE SCORE TESTS
# ============================================================================

def test_pedigree_score_creation(good_pedigree):
    """Test creating a valid pedigree score."""
    assert good_pedigree.reliability == 1
    assert good_pedigree.completeness == 2
    assert good_pedigree.temporal == 1
    assert good_pedigree.geographical == 2
    assert good_pedigree.technological == 1


def test_pedigree_score_validation():
    """Test pedigree score validation."""
    # Valid scores
    PedigreeScore(
        reliability=1,
        completeness=2,
        temporal=3,
        geographical=4,
        technological=5,
    )

    # Invalid scores (should raise validation error)
    with pytest.raises(Exception):
        PedigreeScore(
            reliability=0,  # Too low
            completeness=2,
            temporal=1,
            geographical=2,
            technological=1,
        )

    with pytest.raises(Exception):
        PedigreeScore(
            reliability=1,
            completeness=6,  # Too high
            temporal=1,
            geographical=2,
            technological=1,
        )


def test_pedigree_average_score(good_pedigree):
    """Test average score calculation."""
    # Good pedigree: (1 + 2 + 1 + 2 + 1) / 5 = 1.4
    assert good_pedigree.average_score == 1.4


def test_pedigree_quality_label(excellent_pedigree, good_pedigree, fair_pedigree, poor_pedigree):
    """Test quality label assignment."""
    assert excellent_pedigree.quality_label == "Excellent"
    assert good_pedigree.quality_label == "Good"
    assert fair_pedigree.quality_label == "Fair"
    assert poor_pedigree.quality_label == "Poor"


def test_pedigree_to_dict(good_pedigree):
    """Test conversion to dictionary."""
    d = good_pedigree.to_dict()
    assert d["reliability"] == 1
    assert d["completeness"] == 2
    assert d["temporal"] == 1
    assert d["geographical"] == 2
    assert d["technological"] == 1


# ============================================================================
# UNCERTAINTY FACTOR TESTS
# ============================================================================

def test_get_uncertainty_factor(evaluator):
    """Test getting uncertainty factors."""
    # Reliability dimension
    assert evaluator.get_uncertainty_factor(PedigreeIndicator.RELIABILITY, 1) == 1.00
    assert evaluator.get_uncertainty_factor(PedigreeIndicator.RELIABILITY, 2) == 1.05
    assert evaluator.get_uncertainty_factor(PedigreeIndicator.RELIABILITY, 5) == 1.50

    # Temporal dimension
    assert evaluator.get_uncertainty_factor(PedigreeIndicator.TEMPORAL, 1) == 1.00
    assert evaluator.get_uncertainty_factor(PedigreeIndicator.TEMPORAL, 3) == 1.10


def test_calculate_combined_uncertainty_excellent(evaluator, excellent_pedigree):
    """Test combined uncertainty for excellent quality data."""
    # All scores = 1, all factors = 1.0
    # Combined factor = 1.0 × 1.0 × 1.0 × 1.0 × 1.0 = 1.0
    # With base_uncertainty = 0, result should be minimal
    uncertainty = evaluator.calculate_combined_uncertainty(excellent_pedigree, 0.1)
    assert uncertainty == pytest.approx(0.1, rel=0.01)


def test_calculate_combined_uncertainty_good(evaluator, good_pedigree):
    """Test combined uncertainty for good quality data."""
    # Scores: 1, 2, 1, 2, 1
    # Factors: 1.0, 1.02, 1.0, 1.01, 1.0
    # Combined factor ≈ 1.0303
    uncertainty = evaluator.calculate_combined_uncertainty(good_pedigree, 0.1)
    assert uncertainty > 0.1  # Should be higher than base
    assert uncertainty < 0.15  # But not too high


def test_calculate_dimension_uncertainties(evaluator, good_pedigree):
    """Test dimension-specific uncertainty calculation."""
    uncertainties = evaluator.calculate_dimension_uncertainties(good_pedigree)

    assert "reliability" in uncertainties
    assert "completeness" in uncertainties
    assert "temporal" in uncertainties
    assert "geographical" in uncertainties
    assert "technological" in uncertainties

    assert uncertainties["reliability"] == 1.00
    assert uncertainties["completeness"] == 1.02
    assert uncertainties["geographical"] == 1.01


# ============================================================================
# TEMPORAL ASSESSMENT TESTS
# ============================================================================

def test_assess_temporal_score(evaluator):
    """Test automatic temporal score assessment."""
    # Recent data (<3 years)
    assert evaluator.assess_temporal_score(2023, 2024) == 1

    # 4 years old
    assert evaluator.assess_temporal_score(2020, 2024) == 2

    # 8 years old
    assert evaluator.assess_temporal_score(2016, 2024) == 3

    # 12 years old
    assert evaluator.assess_temporal_score(2012, 2024) == 4

    # 20 years old
    assert evaluator.assess_temporal_score(2004, 2024) == 5


# ============================================================================
# DQI CONVERSION TESTS
# ============================================================================

def test_pedigree_to_dqi(evaluator, excellent_pedigree, good_pedigree, fair_pedigree, poor_pedigree):
    """Test pedigree to DQI conversion."""
    # Excellent: avg_score = 1.0 → DQI = 100
    dqi_excellent = evaluator.pedigree_to_dqi(excellent_pedigree)
    assert dqi_excellent == 100.0

    # Good: avg_score = 1.4 → DQI ≈ 92.5
    dqi_good = evaluator.pedigree_to_dqi(good_pedigree)
    assert 90 <= dqi_good <= 95

    # Fair: avg_score = 3.0 → DQI = 75
    dqi_fair = evaluator.pedigree_to_dqi(fair_pedigree)
    assert 70 <= dqi_fair <= 80

    # Poor: avg_score = 4.4 → DQI ≈ 40
    dqi_poor = evaluator.pedigree_to_dqi(poor_pedigree)
    assert 30 <= dqi_poor <= 45


# ============================================================================
# UNCERTAINTY RESULT TESTS
# ============================================================================

def test_pedigree_to_uncertainty_result(evaluator, good_pedigree):
    """Test conversion to uncertainty result."""
    result = evaluator.pedigree_to_uncertainty_result(
        mean=1000.0, pedigree=good_pedigree, base_uncertainty=0.1
    )

    assert result.mean == 1000.0
    assert result.std_dev > 0
    assert result.relative_std_dev > 0
    assert result.confidence_95_lower < result.mean
    assert result.confidence_95_upper > result.mean
    assert result.pedigree_score == good_pedigree


def test_uncertainty_result_confidence_intervals(evaluator, good_pedigree):
    """Test confidence interval calculation."""
    result = evaluator.pedigree_to_uncertainty_result(
        mean=1000.0, pedigree=good_pedigree, base_uncertainty=0.15
    )

    # 95% CI should be wider than 90% CI
    ci_95_range = result.confidence_95_upper - result.confidence_95_lower
    ci_90_range = result.confidence_90_upper - result.confidence_90_lower

    assert ci_95_range > ci_90_range

    # Confidence intervals should be non-negative for emissions
    assert result.confidence_95_lower >= 0
    assert result.confidence_90_lower >= 0


# ============================================================================
# QUALITY REPORT TESTS
# ============================================================================

def test_generate_quality_report(evaluator, good_pedigree):
    """Test quality report generation."""
    report = evaluator.generate_quality_report(good_pedigree)

    assert "pedigree_scores" in report
    assert "average_score" in report
    assert "quality_label" in report
    assert "dqi_score" in report
    assert "combined_uncertainty" in report
    assert "dimension_uncertainties" in report
    assert "dimension_descriptions" in report
    assert "improvement_opportunities" in report

    assert report["quality_label"] == "Good"
    assert report["average_score"] == 1.4


def test_quality_report_improvement_opportunities(evaluator, poor_pedigree):
    """Test improvement opportunity identification."""
    report = evaluator.generate_quality_report(poor_pedigree)

    # Poor scores should have improvement opportunities
    assert len(report["improvement_opportunities"]) > 0

    # Should identify dimensions with scores >= 3
    opportunities = report["improvement_opportunities"]
    assert "temporal" in opportunities  # Score = 5
    assert "technological" in opportunities  # Score = 5


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_create_pedigree_score():
    """Test convenience function for creating pedigree scores."""
    score = create_pedigree_score(
        reliability=1,
        completeness=2,
        temporal=1,
        geographical=2,
        technological=1,
        reference_year=2024,
        data_year=2023,
    )

    assert score.reliability == 1
    assert score.reference_year == 2024
    assert score.data_year == 2023


def test_assess_data_quality():
    """Test quick data quality assessment function."""
    report = assess_data_quality(
        reliability=1,
        completeness=2,
        temporal=1,
        geographical=2,
        technological=1,
    )

    assert "quality_label" in report
    assert "dqi_score" in report
    assert report["quality_label"] == "Good"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_zero_mean_uncertainty(evaluator, good_pedigree):
    """Test uncertainty calculation with zero mean."""
    result = evaluator.pedigree_to_uncertainty_result(
        mean=0.0, pedigree=good_pedigree, base_uncertainty=0.1
    )

    assert result.mean == 0.0
    assert result.std_dev >= 0


def test_negative_mean_uncertainty(evaluator, good_pedigree):
    """Test uncertainty calculation with negative mean (edge case)."""
    # Emissions shouldn't be negative, but test robustness
    result = evaluator.pedigree_to_uncertainty_result(
        mean=-100.0, pedigree=good_pedigree, base_uncertainty=0.1
    )

    # Should handle gracefully
    assert result.mean == -100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
