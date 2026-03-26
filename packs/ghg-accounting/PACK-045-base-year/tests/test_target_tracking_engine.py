# -*- coding: utf-8 -*-
"""
Tests for TargetTrackingEngine (Engine 8).

Covers target progress tracking, SBTi pathway calculation, rebasing,
attribution, and reporting.
Target: ~50 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.target_tracking_engine import (
    TargetTrackingEngine,
    EmissionsTarget,
    YearlyActual,
    TargetTrackingResult,
    PathwayPoint,
    ProgressPoint,
    RebaseResult,
    ReductionAttribution,
    TargetType,
    TargetStatus,
    SBTiAmbition,
    ScopeType,
    IntensityMetric,
)


# ============================================================================
# Fixtures (use correct field names: name, base_year_tco2e, scopes)
# ============================================================================

@pytest.fixture
def sample_target():
    return EmissionsTarget(
        target_id="TGT-001",
        name="Scope 1 Reduction Target",
        target_type=TargetType.ABSOLUTE,
        scopes=[ScopeType.SCOPE_1],
        base_year=2019,
        base_year_tco2e=Decimal("100000"),
        target_year=2030,
        target_reduction_pct=Decimal("42.0"),
    )


@pytest.fixture
def intensity_target():
    return EmissionsTarget(
        target_id="TGT-002",
        name="Intensity Reduction Target",
        target_type=TargetType.INTENSITY,
        scopes=[ScopeType.SCOPE_1],
        base_year=2019,
        base_year_tco2e=Decimal("100000"),
        target_year=2030,
        target_reduction_pct=Decimal("50.0"),
        intensity_metric=IntensityMetric.PER_REVENUE,
        base_year_intensity_denominator=Decimal("100"),
    )


@pytest.fixture
def sbti_target():
    return EmissionsTarget(
        target_id="TGT-003",
        name="SBTi 1.5C Target",
        target_type=TargetType.ABSOLUTE,
        scopes=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],
        base_year=2019,
        base_year_tco2e=Decimal("100000"),
        target_year=2030,
        target_reduction_pct=Decimal("46.2"),
        sbti_ambition=SBTiAmbition.ONE_POINT_FIVE_C,
    )


@pytest.fixture
def yearly_actuals():
    return [
        YearlyActual(year=2019, actual_tco2e=Decimal("100000")),
        YearlyActual(year=2020, actual_tco2e=Decimal("95000")),
        YearlyActual(year=2021, actual_tco2e=Decimal("90000")),
        YearlyActual(year=2022, actual_tco2e=Decimal("85000")),
        YearlyActual(year=2023, actual_tco2e=Decimal("82000")),
    ]


# ============================================================================
# Engine Init
# ============================================================================

class TestTargetTrackingEngineInit:
    def test_engine_creation(self, target_engine):
        assert target_engine is not None

    def test_engine_is_instance(self, target_engine):
        assert isinstance(target_engine, TargetTrackingEngine)


# ============================================================================
# Track Progress
# ============================================================================

class TestTrackProgress:
    def test_track_absolute_progress(self, target_engine, sample_target, yearly_actuals):
        result = target_engine.track_progress(sample_target, yearly_actuals)
        assert isinstance(result, TargetTrackingResult)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_track_progress_points(self, target_engine, sample_target, yearly_actuals):
        result = target_engine.track_progress(sample_target, yearly_actuals)
        assert len(result.progress_points) >= 1

    def test_progress_current_status(self, target_engine, sample_target, yearly_actuals):
        result = target_engine.track_progress(sample_target, yearly_actuals)
        assert result.current_status in list(TargetStatus)

    def test_progress_years_remaining(self, target_engine, sample_target, yearly_actuals):
        result = target_engine.track_progress(sample_target, yearly_actuals)
        assert result.years_remaining >= 0

    def test_progress_has_recommendations(self, target_engine, sample_target, yearly_actuals):
        result = target_engine.track_progress(sample_target, yearly_actuals)
        assert isinstance(result.recommendations, list)


# ============================================================================
# Calculate Linear Pathway
# ============================================================================

class TestCalculateLinearPathway:
    def test_linear_pathway(self, target_engine, sample_target):
        pathway = target_engine.calculate_linear_pathway(sample_target)
        assert isinstance(pathway, list)
        assert len(pathway) >= 2

    def test_pathway_starts_at_base(self, target_engine, sample_target):
        pathway = target_engine.calculate_linear_pathway(sample_target)
        first = pathway[0]
        assert isinstance(first, PathwayPoint)
        assert first.year == sample_target.base_year

    def test_pathway_ends_at_target(self, target_engine, sample_target):
        pathway = target_engine.calculate_linear_pathway(sample_target)
        last = pathway[-1]
        assert last.year == sample_target.target_year

    def test_pathway_monotonically_decreasing(self, target_engine, sample_target):
        pathway = target_engine.calculate_linear_pathway(sample_target)
        for i in range(1, len(pathway)):
            assert pathway[i].expected_tco2e <= pathway[i - 1].expected_tco2e

    def test_pathway_point_has_cumulative_reduction(self, target_engine, sample_target):
        pathway = target_engine.calculate_linear_pathway(sample_target)
        for point in pathway:
            assert hasattr(point, "cumulative_reduction_pct")


# ============================================================================
# Calculate SBTi Pathway (no ambition param)
# ============================================================================

class TestCalculateSBTiPathway:
    def test_sbti_pathway(self, target_engine, sbti_target):
        pathway = target_engine.calculate_sbti_pathway(sbti_target)
        assert isinstance(pathway, list)
        assert len(pathway) >= 2

    def test_sbti_pathway_from_regular_target(self, target_engine, sample_target):
        pathway = target_engine.calculate_sbti_pathway(sample_target)
        assert isinstance(pathway, list)


# ============================================================================
# Calculate Required Rate
# ============================================================================

class TestCalculateRequiredRate:
    def test_required_rate(self, target_engine, sample_target):
        rate = target_engine.calculate_required_rate(
            sample_target,
            current_actual=Decimal("85000"),
            current_year=2023,
        )
        assert isinstance(rate, Decimal)
        assert rate > Decimal("0")


# ============================================================================
# Assess Status
# ============================================================================

class TestAssessStatus:
    def test_assess_on_track(self, target_engine, sample_target):
        # With base 100000, target 42% reduction by 2030, if 82000 in 2023
        status = target_engine.assess_status(
            sample_target,
            current_actual=Decimal("82000"),
        )
        assert isinstance(status, TargetStatus)

    def test_assess_with_expected(self, target_engine, sample_target):
        status = target_engine.assess_status(
            sample_target,
            current_actual=Decimal("85000"),
            expected=Decimal("82000"),
        )
        assert isinstance(status, TargetStatus)


# ============================================================================
# Rebase Target
# ============================================================================

class TestRebaseTarget:
    def test_rebase_target(self, target_engine, sample_target):
        result = target_engine.rebase_target(
            sample_target,
            new_base_year_tco2e=Decimal("105000"),
            adjustment_reason="Acquisition of subsidiary",
        )
        assert isinstance(result, RebaseResult)

    def test_rebase_preserves_target_year(self, target_engine, sample_target):
        result = target_engine.rebase_target(
            sample_target,
            new_base_year_tco2e=Decimal("105000"),
            adjustment_reason="Acquisition",
        )
        assert result.rebased_target.target_year == sample_target.target_year

    def test_rebase_updates_base_tco2e(self, target_engine, sample_target):
        result = target_engine.rebase_target(
            sample_target,
            new_base_year_tco2e=Decimal("105000"),
            adjustment_reason="Acquisition",
        )
        assert result.rebased_target.base_year_tco2e == Decimal("105000")

    def test_rebase_has_provenance(self, target_engine, sample_target):
        result = target_engine.rebase_target(
            sample_target,
            new_base_year_tco2e=Decimal("105000"),
            adjustment_reason="Acquisition",
        )
        assert result.provenance_hash != ""


# ============================================================================
# Attribute Reductions
# ============================================================================

class TestAttributeReductions:
    def test_attribute_reductions(self, target_engine):
        base_sources = {"stationary": Decimal("30000"), "mobile": Decimal("20000")}
        current_sources = {"stationary": Decimal("25000"), "mobile": Decimal("18000")}
        attributions = target_engine.attribute_reductions(base_sources, current_sources)
        assert isinstance(attributions, list)
        for attr in attributions:
            assert isinstance(attr, ReductionAttribution)


# ============================================================================
# Validate SBTi Alignment
# ============================================================================

class TestValidateSBTiAlignment:
    def test_validate_alignment(self, target_engine, sbti_target):
        result = target_engine.validate_sbti_alignment(sbti_target)
        assert isinstance(result, dict)

    def test_validate_regular_target(self, target_engine, sample_target):
        result = target_engine.validate_sbti_alignment(sample_target)
        assert isinstance(result, dict)


# ============================================================================
# Generate Progress Report
# ============================================================================

class TestGenerateProgressReport:
    def test_generate_report(self, target_engine, sample_target, yearly_actuals):
        tracking_result = target_engine.track_progress(sample_target, yearly_actuals)
        report = target_engine.generate_progress_report(tracking_result)
        assert isinstance(report, str)
        assert len(report) > 0


# ============================================================================
# Get Pathway Summary (takes EmissionsTarget)
# ============================================================================

class TestGetPathwaySummary:
    def test_get_pathway_summary(self, target_engine, sample_target):
        summary = target_engine.get_pathway_summary(sample_target)
        assert isinstance(summary, dict)


# ============================================================================
# Model Tests
# ============================================================================

class TestEmissionsTargetModel:
    def test_create_absolute_target(self, sample_target):
        assert sample_target.target_type == TargetType.ABSOLUTE
        assert sample_target.base_year == 2019
        assert sample_target.target_reduction_pct == Decimal("42.0")

    def test_create_intensity_target(self, intensity_target):
        assert intensity_target.target_type == TargetType.INTENSITY
        assert intensity_target.intensity_metric is not None

    def test_target_id_set(self, sample_target):
        assert sample_target.target_id == "TGT-001"

    def test_target_name(self, sample_target):
        assert sample_target.name == "Scope 1 Reduction Target"

    def test_target_scopes(self, sample_target):
        assert ScopeType.SCOPE_1 in sample_target.scopes


class TestYearlyActualModel:
    def test_create_actual(self):
        actual = YearlyActual(year=2023, actual_tco2e=Decimal("85000"))
        assert actual.year == 2023
        assert actual.actual_tco2e == Decimal("85000")

    def test_actual_with_intensity(self):
        actual = YearlyActual(
            year=2023,
            actual_tco2e=Decimal("85000"),
            intensity_denominator=Decimal("110"),
        )
        assert actual.intensity_denominator == Decimal("110")

    def test_actual_is_verified_default(self):
        actual = YearlyActual(year=2023, actual_tco2e=Decimal("85000"))
        assert actual.is_verified is False


# ============================================================================
# Enums
# ============================================================================

class TestEnums:
    def test_target_type(self):
        assert TargetType.ABSOLUTE is not None
        assert TargetType.INTENSITY is not None
        assert len(TargetType) == 2

    def test_target_status(self):
        assert TargetStatus.ON_TRACK is not None
        assert TargetStatus.BEHIND is not None
        assert TargetStatus.AHEAD is not None
        assert len(TargetStatus) >= 3

    def test_sbti_ambition(self):
        assert SBTiAmbition.WELL_BELOW_2C is not None
        assert SBTiAmbition.ONE_POINT_FIVE_C is not None
        assert SBTiAmbition.NET_ZERO is not None
        assert len(SBTiAmbition) == 3

    def test_scope_type(self):
        assert ScopeType.SCOPE_1 is not None
        assert ScopeType.SCOPE_2 is not None
        assert ScopeType.SCOPE_3 is not None
        assert len(ScopeType) == 3
