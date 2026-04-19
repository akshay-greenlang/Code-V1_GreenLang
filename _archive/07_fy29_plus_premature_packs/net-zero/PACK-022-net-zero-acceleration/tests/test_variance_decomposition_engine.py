# -*- coding: utf-8 -*-
"""
Unit tests for VarianceDecompositionEngine (PACK-022 Engine 7).

Tests LMDI-I decomposition, driver attribution, rolling forecast,
early warning alerts, cumulative effects, and the full pipeline.
"""

import pytest
from decimal import Decimal

from engines.variance_decomposition_engine import (
    VarianceDecompositionEngine,
    VarianceDecompositionConfig,
    SegmentData,
    YearDecomposition,
    DriverAttribution,
    ForecastPoint,
    EarlyWarningAlert,
    CumulativeEffect,
    VarianceResult,
    DecompositionMethod,
    DecompositionEffect,
    ScopeFilter,
    ForecastHorizon,
    AlertSeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(seg_id, name, year, emissions, activity, scope=ScopeFilter.ALL_SCOPES):
    return SegmentData(
        segment_id=seg_id,
        segment_name=name,
        year=year,
        emissions=Decimal(str(emissions)),
        activity=Decimal(str(activity)),
        scope=scope,
    )


def _build_two_year_engine():
    """Engine with two segments across two years for basic decomposition."""
    eng = VarianceDecompositionEngine()
    segments = [
        _make_segment("seg-a", "Division A", 2023, 500, 1000),
        _make_segment("seg-b", "Division B", 2023, 300, 800),
        _make_segment("seg-a", "Division A", 2024, 450, 1100),
        _make_segment("seg-b", "Division B", 2024, 280, 850),
    ]
    eng.add_segment_data(segments)
    return eng


def _build_three_year_engine():
    """Engine with three years for cumulative and forecast tests."""
    eng = VarianceDecompositionEngine()
    segments = [
        _make_segment("seg-a", "Div A", 2022, 600, 1000),
        _make_segment("seg-b", "Div B", 2022, 400, 900),
        _make_segment("seg-a", "Div A", 2023, 550, 1050),
        _make_segment("seg-b", "Div B", 2023, 370, 920),
        _make_segment("seg-a", "Div A", 2024, 500, 1100),
        _make_segment("seg-b", "Div B", 2024, 340, 950),
    ]
    eng.add_segment_data(segments)
    return eng


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return VarianceDecompositionEngine()


@pytest.fixture
def two_year_engine():
    return _build_two_year_engine()


@pytest.fixture
def three_year_engine():
    return _build_three_year_engine()


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestVarianceDecompositionInit:

    def test_default_init(self, engine):
        assert isinstance(engine.config, VarianceDecompositionConfig)
        assert engine.config.method == DecompositionMethod.LMDI_I

    def test_custom_config(self):
        eng = VarianceDecompositionEngine({"alert_threshold_yellow_pct": "8"})
        assert eng.config.alert_threshold_yellow_pct == Decimal("8")

    def test_config_object(self):
        cfg = VarianceDecompositionConfig(decimal_precision=6)
        eng = VarianceDecompositionEngine(cfg)
        assert eng.config.decimal_precision == 6


# ---------------------------------------------------------------------------
# Data Management Tests
# ---------------------------------------------------------------------------


class TestDataManagement:

    def test_add_segment_data(self, engine):
        segments = [_make_segment("s1", "S1", 2024, 100, 500)]
        count = engine.add_segment_data(segments)
        assert count == 1

    def test_set_planned_emissions(self, engine):
        engine.set_planned_emissions({2024: Decimal("800"), 2025: Decimal("700")})
        assert engine._planned_emissions[2024] == Decimal("800")

    def test_get_available_years(self, two_year_engine):
        years = two_year_engine._get_available_years()
        assert years == [2023, 2024]

    def test_get_segments_for_year(self, two_year_engine):
        segs = two_year_engine._get_segments_for_year(2023)
        assert len(segs) == 2

    def test_clear(self, two_year_engine):
        two_year_engine.clear()
        assert len(two_year_engine._segment_data) == 0
        assert len(two_year_engine._planned_emissions) == 0


# ---------------------------------------------------------------------------
# LMDI-I Decomposition Tests
# ---------------------------------------------------------------------------


class TestDecomposition:

    def test_basic_decomposition(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        assert isinstance(decomp, YearDecomposition)
        assert decomp.year_from == 2023
        assert decomp.year_to == 2024
        assert decomp.segment_count == 2
        assert len(decomp.provenance_hash) == 64

    def test_zero_residual_property(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        effects_sum = decomp.activity_effect + decomp.intensity_effect + decomp.structural_effect
        total = decomp.total_change
        residual_check = float(total - effects_sum)
        assert abs(residual_check) == pytest.approx(float(decomp.residual), abs=0.01)

    def test_total_change_matches(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        expected_total = (Decimal("450") + Decimal("280")) - (Decimal("500") + Decimal("300"))
        assert float(decomp.total_change) == pytest.approx(float(expected_total), rel=1e-3)

    def test_no_data_for_year_raises(self, engine):
        engine.add_segment_data([_make_segment("s1", "S1", 2024, 100, 500)])
        with pytest.raises(ValueError, match="No segment data for year"):
            engine.decompose_year(2023, 2024)

    def test_no_common_segments_raises(self, engine):
        engine.add_segment_data([
            _make_segment("s1", "S1", 2023, 100, 500),
            _make_segment("s2", "S2", 2024, 100, 500),
        ])
        with pytest.raises(ValueError, match="No common segments"):
            engine.decompose_year(2023, 2024)

    def test_activity_effect_positive_with_growth(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        # Activity grew from 1800 to 1950, so activity effect should be positive
        assert float(decomp.activity_effect) > 0

    def test_intensity_effect_negative_with_improvement(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        # Emissions fell while activity grew, so intensity should be negative
        assert float(decomp.intensity_effect) < 0

    def test_percentage_effects_populated(self, two_year_engine):
        decomp = two_year_engine.decompose_year(2023, 2024)
        assert decomp.total_change_pct != Decimal("0")

    def test_scope_filter(self, engine):
        engine.add_segment_data([
            _make_segment("s1", "S1", 2023, 100, 500, ScopeFilter.SCOPE_1),
            _make_segment("s1", "S1", 2024, 90, 520, ScopeFilter.SCOPE_1),
            _make_segment("s2", "S2", 2023, 50, 300, ScopeFilter.SCOPE_2),
            _make_segment("s2", "S2", 2024, 48, 310, ScopeFilter.SCOPE_2),
        ])
        decomp = engine.decompose_year(2023, 2024, scope=ScopeFilter.SCOPE_1)
        assert decomp.segment_count == 1


# ---------------------------------------------------------------------------
# Driver Attribution Tests
# ---------------------------------------------------------------------------


class TestDriverAttribution:

    def test_attribution_count(self, two_year_engine):
        drivers = two_year_engine.attribute_drivers(2023, 2024)
        assert len(drivers) == 2

    def test_attribution_sorted_by_absolute(self, two_year_engine):
        drivers = two_year_engine.attribute_drivers(2023, 2024)
        for i in range(len(drivers) - 1):
            assert abs(drivers[i].total_contribution) >= abs(drivers[i + 1].total_contribution)

    def test_attribution_provenance_hash(self, two_year_engine):
        drivers = two_year_engine.attribute_drivers(2023, 2024)
        for d in drivers:
            assert len(d.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Rolling Forecast Tests
# ---------------------------------------------------------------------------


class TestRollingForecast:

    def test_forecast_returns_points(self, three_year_engine):
        forecasts = three_year_engine.rolling_forecast(ForecastHorizon.THREE_YEAR)
        assert len(forecasts) == 3

    def test_forecast_years_sequential(self, three_year_engine):
        forecasts = three_year_engine.rolling_forecast(ForecastHorizon.THREE_YEAR)
        assert forecasts[0].year == 2025
        assert forecasts[1].year == 2026
        assert forecasts[2].year == 2027

    def test_forecast_confidence_widens(self, three_year_engine):
        forecasts = three_year_engine.rolling_forecast(ForecastHorizon.THREE_YEAR)
        widths = [float(f.confidence_upper - f.confidence_lower) for f in forecasts]
        for i in range(len(widths) - 1):
            assert widths[i + 1] >= widths[i]

    def test_forecast_one_year_horizon(self, three_year_engine):
        forecasts = three_year_engine.rolling_forecast(ForecastHorizon.ONE_YEAR)
        assert len(forecasts) == 1

    def test_forecast_insufficient_data(self, engine):
        engine.add_segment_data([_make_segment("s1", "S1", 2024, 100, 500)])
        forecasts = engine.rolling_forecast()
        assert forecasts == []

    def test_forecast_with_planned_emissions(self, three_year_engine):
        three_year_engine.set_planned_emissions({2025: Decimal("750")})
        forecasts = three_year_engine.rolling_forecast(ForecastHorizon.ONE_YEAR)
        assert forecasts[0].planned_emissions is not None


# ---------------------------------------------------------------------------
# Early Warning Tests
# ---------------------------------------------------------------------------


class TestEarlyWarning:

    def test_no_alerts_when_no_plan(self, two_year_engine):
        alerts = two_year_engine.check_early_warnings()
        assert len(alerts) == 0

    def test_yellow_alert(self, two_year_engine):
        two_year_engine.set_planned_emissions({2024: Decimal("690")})
        alerts = two_year_engine.check_early_warnings()
        assert len(alerts) >= 1
        severities = [a.severity for a in alerts]
        assert AlertSeverity.YELLOW in severities or AlertSeverity.ORANGE in severities or AlertSeverity.RED in severities

    def test_red_alert(self, two_year_engine):
        two_year_engine.set_planned_emissions({2024: Decimal("500")})
        alerts = two_year_engine.check_early_warnings()
        assert any(a.severity == AlertSeverity.RED for a in alerts)

    def test_green_no_alert(self, two_year_engine):
        actual_2024 = Decimal("730")
        two_year_engine.set_planned_emissions({2024: actual_2024})
        alerts = two_year_engine.check_early_warnings()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Cumulative Effects Tests
# ---------------------------------------------------------------------------


class TestCumulativeEffects:

    def test_cumulative_returns_four_effects(self, three_year_engine):
        effects = three_year_engine.calculate_cumulative_effects()
        assert len(effects) == 4

    def test_cumulative_effect_types(self, three_year_engine):
        effects = three_year_engine.calculate_cumulative_effects()
        types = {e.effect_type for e in effects}
        assert types == {
            DecompositionEffect.ACTIVITY,
            DecompositionEffect.INTENSITY,
            DecompositionEffect.STRUCTURAL,
            DecompositionEffect.TOTAL,
        }

    def test_cumulative_year_range(self, three_year_engine):
        effects = three_year_engine.calculate_cumulative_effects()
        for e in effects:
            assert e.year_from == 2022
            assert e.year_to == 2024

    def test_cumulative_insufficient_data(self, engine):
        engine.add_segment_data([_make_segment("s1", "S1", 2024, 100, 500)])
        assert engine.calculate_cumulative_effects() == []


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


class TestFullDecomposition:

    def test_full_decomposition_structure(self, three_year_engine):
        result = three_year_engine.run_full_decomposition()
        assert isinstance(result, VarianceResult)
        assert result.years_analyzed == 3
        assert result.segments_analyzed == 2
        assert len(result.decomposition_by_year) == 2
        assert len(result.driver_attribution) == 2
        assert len(result.cumulative_effects) == 4
        assert len(result.provenance_hash) == 64

    def test_full_with_scope_filter(self, engine):
        engine.add_segment_data([
            _make_segment("s1", "S1", 2023, 100, 500, ScopeFilter.SCOPE_1),
            _make_segment("s1", "S1", 2024, 90, 520, ScopeFilter.SCOPE_1),
        ])
        result = engine.run_full_decomposition(scope=ScopeFilter.SCOPE_1)
        assert result.years_analyzed == 2

    def test_full_decomposition_method_in_result(self, three_year_engine):
        result = three_year_engine.run_full_decomposition()
        assert result.method == DecompositionMethod.LMDI_I


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_zero_emissions_segment_skipped(self, engine):
        """Zero-emission segments are silently skipped by LMDI-I to avoid log(0)."""
        engine.add_segment_data([
            _make_segment("s1", "S1", 2023, 0, 500),
            _make_segment("s1", "S1", 2024, 0, 500),
        ])
        decomp = engine.decompose_year(2023, 2024)
        # All effects should be zero since the only segment was skipped
        assert float(decomp.total_change) == 0.0
        assert float(decomp.activity_effect) == 0.0
        assert float(decomp.intensity_effect) == 0.0

    def test_single_segment(self, engine):
        engine.add_segment_data([
            _make_segment("s1", "S1", 2023, 100, 500),
            _make_segment("s1", "S1", 2024, 90, 520),
        ])
        decomp = engine.decompose_year(2023, 2024)
        assert decomp.segment_count == 1

    def test_enum_values(self):
        assert DecompositionMethod.LMDI_I.value == "lmdi_i"
        assert ForecastHorizon.TWO_YEAR.value == "2_year"
        assert AlertSeverity.GREEN.value == "green"
        assert ScopeFilter.ALL_SCOPES.value == "all_scopes"
