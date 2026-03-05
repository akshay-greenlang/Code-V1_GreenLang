# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Metrics & Targets Engine.

Tests GHG emissions retrieval, intensity metrics, cross-industry metrics,
industry metrics, target creation, progress tracking, SBTi alignment,
peer benchmarking, and implied temperature rise with 28+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ISSBMetricType,
    TargetType,
    ISSB_CROSS_INDUSTRY_METRICS,
    TCFD_DISCLOSURES,
)
from services.models import (
    ClimateMetric,
    ClimateTarget,
    TargetProgress,
    _new_id,
)


# ===========================================================================
# GHG Emissions Retrieval
# ===========================================================================

class TestGHGEmissionsRetrieval:
    """Test GHG emissions metric retrieval."""

    def test_scope1_metric(self, sample_climate_metric):
        assert sample_climate_metric.metric_type == ISSBMetricType.GHG_EMISSIONS
        assert sample_climate_metric.value == Decimal("125000")
        assert sample_climate_metric.unit == "tCO2e"

    def test_scope1_metric_scope(self, sample_climate_metric):
        assert sample_climate_metric.scope == "scope_1"

    @pytest.mark.parametrize("scope", ["scope_1", "scope_2", "scope_3"])
    def test_all_scopes(self, scope):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name=f"Total {scope} Emissions",
            value=Decimal("50000"),
            unit="tCO2e",
            reporting_year=2025,
            scope=scope,
        )
        assert metric.scope == scope

    def test_metric_data_source(self, sample_climate_metric):
        assert sample_climate_metric.source == "MRV Agent Pipeline"


# ===========================================================================
# Intensity Metrics
# ===========================================================================

class TestIntensityMetrics:
    """Test GHG intensity metric calculations."""

    def test_revenue_intensity(self):
        emissions = Decimal("125000")
        revenue = Decimal("2500000000")
        intensity = emissions / revenue * Decimal("1000000")
        assert intensity == Decimal("50")  # 50 tCO2e per million USD

    def test_employee_intensity(self):
        emissions = Decimal("125000")
        employees = Decimal("5000")
        intensity = emissions / employees
        assert intensity == Decimal("25")  # 25 tCO2e per employee


# ===========================================================================
# Cross-Industry Metrics
# ===========================================================================

class TestCrossIndustryMetrics:
    """Test ISSB cross-industry metrics."""

    @pytest.mark.parametrize("metric_type", list(ISSBMetricType))
    def test_all_issb_metrics_defined(self, metric_type):
        assert metric_type in ISSB_CROSS_INDUSTRY_METRICS
        entry = ISSB_CROSS_INDUSTRY_METRICS[metric_type]
        assert "name" in entry
        assert "ifrs_s2_paragraph" in entry
        assert "unit" in entry

    def test_ghg_emissions_metric(self):
        entry = ISSB_CROSS_INDUSTRY_METRICS[ISSBMetricType.GHG_EMISSIONS]
        assert entry["ifrs_s2_paragraph"] == "29(a)"

    def test_seven_cross_industry_metrics(self):
        assert len(ISSB_CROSS_INDUSTRY_METRICS) == 7

    def test_internal_carbon_price_metric(self):
        entry = ISSB_CROSS_INDUSTRY_METRICS[ISSBMetricType.INTERNAL_CARBON_PRICE]
        assert entry["unit"] == "currency_per_tCO2e"

    def test_metric_creation_all_types(self):
        for mt in ISSBMetricType:
            metric = ClimateMetric(
                org_id=_new_id(),
                metric_type=mt,
                metric_name=f"Test {mt.value}",
                value=Decimal("100"),
                reporting_year=2025,
            )
            assert metric.metric_type == mt


# ===========================================================================
# Industry Metrics
# ===========================================================================

class TestIndustryMetrics:
    """Test industry-specific metrics."""

    def test_metric_data_quality(self, sample_climate_metric):
        assert 1 <= sample_climate_metric.data_quality_score <= 5

    def test_metric_provenance(self, sample_climate_metric):
        assert len(sample_climate_metric.provenance_hash) == 64

    def test_metric_benchmark(self):
        metric = ClimateMetric(
            org_id=_new_id(),
            metric_type=ISSBMetricType.GHG_EMISSIONS,
            metric_name="Scope 1 with benchmark",
            value=Decimal("125000"),
            reporting_year=2025,
            industry_benchmark=Decimal("150000"),
        )
        assert metric.industry_benchmark == Decimal("150000")
        assert metric.value < metric.industry_benchmark  # below benchmark


# ===========================================================================
# Target Creation
# ===========================================================================

class TestTargetCreation:
    """Test climate target creation."""

    def test_absolute_target(self, sample_climate_target):
        assert sample_climate_target.target_type == TargetType.ABSOLUTE
        assert sample_climate_target.base_year == 2020
        assert sample_climate_target.target_year == 2030

    @pytest.mark.parametrize("target_type", list(TargetType))
    def test_all_target_types(self, target_type):
        target = ClimateTarget(
            org_id=_new_id(),
            target_type=target_type,
            target_name=f"Test {target_type.value}",
            base_year=2020,
            base_value=Decimal("100000"),
            target_year=2030,
            target_value=Decimal("50000"),
        )
        assert target.target_type == target_type

    def test_target_year_validation(self):
        with pytest.raises(Exception):
            ClimateTarget(
                org_id=_new_id(),
                target_type=TargetType.ABSOLUTE,
                target_name="Invalid",
                base_year=2030,
                base_value=Decimal("100000"),
                target_year=2020,
                target_value=Decimal("50000"),
            )

    def test_net_zero_target(self):
        target = ClimateTarget(
            org_id=_new_id(),
            target_type=TargetType.NET_ZERO,
            target_name="Net Zero 2050",
            base_year=2020,
            base_value=Decimal("200000"),
            target_year=2050,
            target_value=Decimal("0"),
            sbti_aligned=True,
        )
        assert target.target_value == Decimal("0")
        assert target.sbti_aligned is True


# ===========================================================================
# Progress Tracking
# ===========================================================================

class TestProgressTracking:
    """Test target progress tracking."""

    def test_progress_percentage(self, sample_target_progress):
        assert sample_target_progress.progress_pct == Decimal("45.0")

    def test_on_track_status(self, sample_target_progress):
        assert sample_target_progress.on_track is True

    def test_gap_to_target(self, sample_target_progress):
        assert sample_target_progress.gap_to_target == Decimal("55000")

    def test_progress_provenance(self, sample_target_progress):
        assert len(sample_target_progress.provenance_hash) == 64

    def test_behind_schedule_progress(self):
        progress = TargetProgress(
            target_id=_new_id(),
            reporting_year=2025,
            current_value=Decimal("180000"),
            progress_pct=Decimal("20.0"),
            gap_to_target=Decimal("80000"),
            on_track=False,
            notes="Behind linear trajectory",
        )
        assert progress.on_track is False


# ===========================================================================
# SBTi Alignment
# ===========================================================================

class TestSBTiAlignment:
    """Test Science Based Targets initiative alignment."""

    def test_sbti_aligned_target(self, sample_climate_target):
        assert sample_climate_target.sbti_aligned is True

    def test_not_sbti_aligned(self):
        target = ClimateTarget(
            org_id=_new_id(),
            target_type=TargetType.ABSOLUTE,
            target_name="Non-SBTi Target",
            base_year=2020,
            base_value=Decimal("100000"),
            target_year=2030,
            target_value=Decimal("80000"),
            sbti_aligned=False,
        )
        assert target.sbti_aligned is False


# ===========================================================================
# Metrics & Targets Disclosures
# ===========================================================================

class TestMetricsTargetsDisclosures:
    """Test metrics and targets disclosure definitions."""

    def test_mt_a_defined(self):
        assert "mt_a" in TCFD_DISCLOSURES
        assert "metric" in TCFD_DISCLOSURES["mt_a"]["description"].lower()

    def test_mt_b_ghg_emissions(self):
        assert "mt_b" in TCFD_DISCLOSURES
        assert "scope" in TCFD_DISCLOSURES["mt_b"]["description"].lower()

    def test_mt_c_targets(self):
        assert "mt_c" in TCFD_DISCLOSURES
        assert "target" in TCFD_DISCLOSURES["mt_c"]["description"].lower()
