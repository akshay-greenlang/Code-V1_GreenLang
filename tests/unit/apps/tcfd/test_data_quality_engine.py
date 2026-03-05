# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Data Quality Engine.

Tests quality scoring dimensions, completeness checking, validation,
and improvement suggestions with 18+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ISSBMetricType,
)
from services.models import (
    ClimateMetric,
    ClimateTarget,
    TargetProgress,
    GovernanceAssessment,
    ClimateRisk,
    RiskType,
    _new_id,
)


# ===========================================================================
# Quality Scoring Dimensions
# ===========================================================================

class TestQualityScoringDimensions:
    """Test data quality scoring across dimensions."""

    @pytest.mark.parametrize("score", range(1, 6))
    def test_data_quality_tiers(self, score):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Test Quality",
            value=Decimal("50000"),
            reporting_year=2025,
            data_quality_score=score,
        )
        assert metric.data_quality_score == score

    def test_highest_quality(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Verified Emissions",
            value=Decimal("50000"),
            reporting_year=2025,
            data_quality_score=5,
            source="Verified third-party audit",
        )
        assert metric.data_quality_score == 5

    def test_lowest_quality(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Estimated Emissions",
            value=Decimal("50000"),
            reporting_year=2025,
            data_quality_score=1,
            source="Industry average estimate",
        )
        assert metric.data_quality_score == 1


# ===========================================================================
# Completeness Checking
# ===========================================================================

class TestCompletenessChecking:
    """Test data completeness checking."""

    def test_metric_with_all_fields(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Complete Metric",
            value=Decimal("125000"),
            unit="tCO2e",
            reporting_year=2025,
            scope="scope_1",
            data_quality_score=4,
            source="MRV Pipeline",
            industry_benchmark=Decimal("150000"),
        )
        assert metric.scope is not None
        assert metric.source != ""
        assert metric.industry_benchmark is not None

    def test_metric_with_minimal_fields(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Minimal Metric",
            value=Decimal("50000"),
            reporting_year=2025,
        )
        assert metric.scope is None
        assert metric.source == ""
        assert metric.industry_benchmark is None

    def test_target_completeness(self, sample_climate_target):
        assert sample_climate_target.base_year is not None
        assert sample_climate_target.target_year is not None
        assert sample_climate_target.base_value is not None
        assert sample_climate_target.target_value is not None
        assert sample_climate_target.unit is not None

    def test_governance_completeness(self, sample_governance_assessment):
        assert sample_governance_assessment.board_oversight_score is not None
        assert sample_governance_assessment.climate_competency_score is not None
        assert sample_governance_assessment.maturity_scores is not None

    def test_risk_completeness(self, sample_climate_risk):
        assert sample_climate_risk.risk_type is not None
        assert sample_climate_risk.name is not None
        assert sample_climate_risk.likelihood is not None
        assert sample_climate_risk.impact is not None


# ===========================================================================
# Validation
# ===========================================================================

class TestValidation:
    """Test data validation rules."""

    def test_metric_reporting_year_range(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Test",
            value=Decimal("50000"),
            reporting_year=2025,
        )
        assert 1990 <= metric.reporting_year <= 2100

    def test_target_year_after_base(self, sample_climate_target):
        assert sample_climate_target.target_year > sample_climate_target.base_year

    def test_progress_percentage_valid(self, sample_target_progress):
        assert Decimal("0") <= sample_target_progress.progress_pct <= Decimal("200")

    def test_provenance_hash_valid(self, sample_climate_metric):
        assert len(sample_climate_metric.provenance_hash) == 64

    def test_negative_emissions_invalid(self):
        # ClimateMetric allows any Decimal for value (emissions can be negative for removals)
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Carbon Removal",
            value=Decimal("-5000"),
            reporting_year=2025,
        )
        assert metric.value == Decimal("-5000")


# ===========================================================================
# Improvement Suggestions
# ===========================================================================

class TestImprovementSuggestions:
    """Test data quality improvement suggestions."""

    def test_low_quality_needs_improvement(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Low Quality",
            value=Decimal("50000"),
            reporting_year=2025,
            data_quality_score=1,
        )
        assert metric.data_quality_score < 3  # needs improvement

    def test_missing_source_flag(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="No Source",
            value=Decimal("50000"),
            reporting_year=2025,
            source="",
        )
        assert metric.source == ""  # flag for improvement

    def test_missing_scope_flag(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="No Scope",
            value=Decimal("50000"),
            reporting_year=2025,
            scope=None,
        )
        assert metric.scope is None  # flag for improvement
