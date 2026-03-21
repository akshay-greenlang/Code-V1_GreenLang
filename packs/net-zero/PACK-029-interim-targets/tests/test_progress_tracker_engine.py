# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Progress Tracker Engine (Engine 3)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.progress_tracker_engine import (
    ProgressTrackerEngine, ProgressTrackerInput, ProgressTrackerResult,
    ActualEmissionsPoint, TargetPoint, RAGStatus, ScopeType,
)
from .conftest import (
    assert_provenance_hash, assert_processing_time, timed_block,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_actual(data):
    return [ActualEmissionsPoint(year=d[0], emissions_tco2e=Decimal(str(d[1]))) for d in data]


def _make_targets(data):
    return [TargetPoint(year=d[0], target_tco2e=Decimal(str(d[1]))) for d in data]


def _default_actuals():
    return _make_actual([
        (2020, 194880), (2021, 198940), (2022, 190820),
        (2023, 182700), (2024, 174580), (2025, 166460),
    ])


def _default_targets():
    return _make_targets([
        (2020, 184527), (2021, 166054), (2022, 147582),
        (2023, 129109), (2024, 110636), (2025, 92164),
    ])


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        baseline_year=2019,
        baseline_tco2e=Decimal("203000"),
        actual_emissions=_default_actuals(),
        targets=_default_targets(),
    )
    defaults.update(kwargs)
    return ProgressTrackerInput(**defaults)


class TestInstantiation:
    def test_engine_instantiates(self):
        assert ProgressTrackerEngine() is not None

    def test_engine_version(self):
        assert ProgressTrackerEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(ProgressTrackerEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(ProgressTrackerEngine(), "calculate_batch")


class TestBasicCalculation:
    def test_basic_calculation(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_result_has_variance_points(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert len(r.variance_points) > 0

    def test_variance_points_have_rag(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        for vp in r.variance_points:
            assert vp.rag_status in ("green", "amber", "red", "not_assessed")

    def test_result_has_overall_assessment(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.overall_assessment is not None

    def test_result_has_progress_rate(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.progress_rate is not None

    def test_result_has_milestone_assessments(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.milestone_assessments, list)

    def test_provenance_hash(self):
        assert_provenance_hash(_run(ProgressTrackerEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(ProgressTrackerEngine().calculate(_make_input())))


class TestVarianceCalculation:
    def test_variance_tco2e(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        for vp in r.variance_points:
            assert isinstance(vp.variance_tco2e, Decimal)

    def test_variance_pct(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        for vp in r.variance_points:
            assert isinstance(vp.variance_pct, Decimal)

    def test_positive_variance_when_above_target(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        for vp in r.variance_points:
            if vp.actual_tco2e > vp.target_tco2e:
                assert vp.variance_tco2e > Decimal("0")

    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024, 2025])
    def test_variance_by_year(self, year):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert any(vp.year == year for vp in r.variance_points)


class TestRAGStatus:
    def test_overall_rag(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.overall_assessment.rag_status in ("green", "amber", "red", "not_assessed")

    def test_counts(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        oa = r.overall_assessment
        assert (oa.green_count + oa.amber_count + oa.red_count) > 0

    def test_on_track(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.overall_assessment.on_track_for_net_zero, bool)

    def test_total_periods(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.overall_assessment.total_periods_assessed > 0


class TestProgressRate:
    def test_actual_rate(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.progress_rate.actual_annual_rate_pct, Decimal)

    def test_required_rate(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.progress_rate.required_annual_rate_pct, Decimal)

    def test_gap(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.progress_rate.rate_gap_pct, Decimal)

    def test_years_data(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.progress_rate.years_of_data > 0


class TestScales:
    @pytest.mark.parametrize("baseline", [
        Decimal("50000"), Decimal("200000"), Decimal("1000000"),
        Decimal("5000000"), Decimal("50000000"),
    ])
    def test_various_baselines(self, baseline):
        f = baseline / Decimal("203000")
        a = _make_actual([(y, int(float(Decimal(str(e)) * f)))
                          for y, e in [(2020, 194880), (2021, 190820), (2022, 182700)]])
        t = _make_targets([(y, int(float(Decimal(str(e)) * f)))
                           for y, e in [(2020, 184527), (2021, 147582), (2022, 129109)]])
        r = _run(ProgressTrackerEngine().calculate(_make_input(
            baseline_tco2e=baseline, actual_emissions=a, targets=t)))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C", "Corp D"])
    def test_entities(self, entity):
        r = _run(ProgressTrackerEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(ProgressTrackerEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = ProgressTrackerEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(100):
                _run(e.calculate(inp))


class TestDecimalPrecision:
    def test_variance_decimal(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        for vp in r.variance_points:
            assert isinstance(vp.variance_tco2e, Decimal)

    def test_baseline_decimal(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.baseline_tco2e, Decimal)

    @pytest.mark.parametrize("val", ["123456.789", "999999.999", "1000000.001"])
    def test_precision(self, val):
        r = _run(ProgressTrackerEngine().calculate(_make_input(baseline_tco2e=Decimal(val))))
        assert isinstance(r.baseline_tco2e, Decimal)


class TestRecommendations:
    def test_recommendations_list(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.recommendations, list)

    def test_warnings_list(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert isinstance(r.warnings, list)

    def test_data_quality(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(ProgressTrackerEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_single_year(self):
        a = _make_actual([(2020, 190000)])
        t = _make_targets([(2020, 180000)])
        r = _run(ProgressTrackerEngine().calculate(_make_input(actual_emissions=a, targets=t)))
        assert r is not None

    def test_model_dump(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        d = r.model_dump()
        assert isinstance(d, dict)

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_year_counts(self, n):
        a = _make_actual([(2019 + i, 200000 - i * 5000) for i in range(1, n + 1)])
        t = _make_targets([(2019 + i, 200000 - i * 8000) for i in range(1, n + 1)])
        r = _run(ProgressTrackerEngine().calculate(_make_input(actual_emissions=a, targets=t)))
        assert r is not None

    def test_sha256_valid(self):
        h = _run(ProgressTrackerEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_scope_field(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.scope != ""

    def test_net_zero_year(self):
        r = _run(ProgressTrackerEngine().calculate(_make_input()))
        assert r.net_zero_year == 2050
