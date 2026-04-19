# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Trend Extrapolation Engine (Engine 5)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.trend_extrapolation_engine import (
    TrendExtrapolationEngine, TrendExtrapolationInput, TrendExtrapolationResult,
    HistoricalDataPoint, TargetTrajectoryPoint, ForecastMethod, ConfidenceLevel,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_historical(data):
    return [HistoricalDataPoint(year=d[0], emissions_tco2e=Decimal(str(d[1]))) for d in data]


def _default_historical():
    return _make_historical([
        (2015, 230000), (2016, 225000), (2017, 220000), (2018, 215000),
        (2019, 203000), (2020, 194880), (2021, 198940), (2022, 190820),
        (2023, 182700), (2024, 174580),
    ])


def _make_target_trajectory():
    return [
        TargetTrajectoryPoint(year=2025, target_tco2e=Decimal("166000")),
        TargetTrajectoryPoint(year=2030, target_tco2e=Decimal("117740")),
        TargetTrajectoryPoint(year=2035, target_tco2e=Decimal("77000")),
    ]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        historical_data=_default_historical(),
    )
    defaults.update(kwargs)
    return TrendExtrapolationInput(**defaults)


class TestInstantiation:
    def test_creates(self):
        assert TrendExtrapolationEngine() is not None

    def test_version(self):
        assert TrendExtrapolationEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(TrendExtrapolationEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(TrendExtrapolationEngine(), "calculate_batch")

    def test_supported_methods(self):
        methods = TrendExtrapolationEngine().get_supported_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0


class TestBasicForecast:
    def test_basic_result(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_forecast_points(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert len(r.forecast_points) > 0

    def test_forecast_has_year(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        for fp in r.forecast_points:
            assert fp.year > 0

    def test_forecast_has_tco2e(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        for fp in r.forecast_points:
            assert isinstance(fp.forecast_tco2e, Decimal)

    def test_provenance(self):
        assert_provenance_hash(_run(TrendExtrapolationEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(TrendExtrapolationEngine().calculate(_make_input())))


class TestRegressionStats:
    def test_regression_exists(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r.regression_stats is not None

    def test_slope(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert isinstance(r.regression_stats.slope, Decimal)

    def test_r_squared(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert isinstance(r.regression_stats.r_squared, Decimal)

    def test_trend_direction(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r.regression_stats.trend_direction in ("decreasing", "increasing", "flat")


class TestScenarioProjections:
    def test_scenarios_exist(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert isinstance(r.scenario_projections, list)

    def test_scenario_has_points(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        for sp in r.scenario_projections:
            assert sp.scenario != ""


class TestTargetMissPredictions:
    def test_target_miss_with_trajectory(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input(
            target_trajectory=_make_target_trajectory(),
        )))
        assert isinstance(r.target_miss_predictions, list)

    def test_target_miss_fields(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input(
            target_trajectory=_make_target_trajectory(),
        )))
        for tm in r.target_miss_predictions:
            assert isinstance(tm.will_miss, bool)
            assert isinstance(tm.gap_tco2e, Decimal)


class TestOverallTrend:
    def test_overall_trend(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r.overall_trend in ("decreasing", "increasing", "flat", "accelerating_decrease",
                                   "decelerating_decrease", "volatile")

    def test_historical_years(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r.historical_years > 0


class TestForecastMethods:
    @pytest.mark.parametrize("method", [ForecastMethod.LINEAR_REGRESSION])
    def test_method(self, method):
        r = _run(TrendExtrapolationEngine().calculate(_make_input(forecast_methods=[method])))
        assert r is not None

    def test_default_methods(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert len(r.forecast_points) > 0


class TestScales:
    @pytest.mark.parametrize("scale", [0.1, 0.5, 1.0, 5.0, 100.0])
    def test_emission_scales(self, scale):
        data = _make_historical([
            (2015 + i, int(230000 * scale - i * 5000 * scale))
            for i in range(10)
        ])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C"])
    def test_entities(self, entity):
        r = _run(TrendExtrapolationEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity


class TestDecimalPrecision:
    def test_forecast_decimal(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        for fp in r.forecast_points:
            assert isinstance(fp.forecast_tco2e, Decimal)

    @pytest.mark.parametrize("val", ["123456.789", "999999.999", "1000000.001"])
    def test_precision(self, val):
        data = _make_historical([(2019 + i, float(val) * (1 - 0.03 * i)) for i in range(6)])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r is not None


class TestRecommendations:
    def test_recommendations(self):
        assert isinstance(_run(TrendExtrapolationEngine().calculate(_make_input())).recommendations, list)

    def test_warnings(self):
        assert isinstance(_run(TrendExtrapolationEngine().calculate(_make_input())).warnings, list)

    def test_data_quality(self):
        r = _run(TrendExtrapolationEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(TrendExtrapolationEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = TrendExtrapolationEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(50):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(TrendExtrapolationEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_minimum_data(self):
        data = _make_historical([(2022, 200000), (2023, 190000)])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r is not None

    def test_flat_data(self):
        data = _make_historical([(2019 + i, 200000) for i in range(6)])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r is not None

    def test_increasing_data(self):
        data = _make_historical([(2019 + i, 200000 + i * 5000) for i in range(6)])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r.overall_trend in ("increasing", "flat", "volatile")

    def test_model_dump(self):
        d = _run(TrendExtrapolationEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(TrendExtrapolationEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    @pytest.mark.parametrize("horizon", [2030, 2035, 2040, 2050])
    def test_horizons(self, horizon):
        r = _run(TrendExtrapolationEngine().calculate(_make_input(forecast_horizon_year=horizon)))
        assert r.forecast_horizon_year == horizon

    @pytest.mark.parametrize("n_years", [3, 5, 10, 14])
    def test_various_history_lengths(self, n_years):
        data = _make_historical([(2024 - n_years + i, 230000 - i * 5000) for i in range(n_years)])
        r = _run(TrendExtrapolationEngine().calculate(_make_input(historical_data=data)))
        assert r is not None
