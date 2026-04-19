# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Variance Analysis Engine (Engine 4)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.variance_analysis_engine import (
    VarianceAnalysisEngine, VarianceAnalysisInput, VarianceAnalysisResult,
    PeriodData, SegmentData, ScopeType,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_period(year, total, activity=Decimal("0"), segments=None):
    return PeriodData(
        year=year,
        total_emissions_tco2e=Decimal(str(total)),
        total_activity_value=Decimal(str(activity)) if activity else Decimal("0"),
        segments=segments or [],
    )


def _make_segments(data):
    return [SegmentData(segment_id=d[0], emissions_tco2e=Decimal(str(d[1])),
                        activity_value=Decimal(str(d[2])) if len(d) > 2 else Decimal("0"))
            for d in data]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        base_period=_make_period(2022, 203000, 2500),
        current_period=_make_period(2023, 183000, 2650),
    )
    defaults.update(kwargs)
    return VarianceAnalysisInput(**defaults)


def _make_input_with_segments():
    base_segs = _make_segments([
        ("facility_a", 80000, 1000), ("facility_b", 60000, 800),
        ("facility_c", 40000, 500), ("fleet", 23000, 200),
    ])
    curr_segs = _make_segments([
        ("facility_a", 72000, 1050), ("facility_b", 55000, 780),
        ("facility_c", 38000, 530), ("fleet", 18000, 190),
    ])
    return VarianceAnalysisInput(
        entity_name="GreenCorp Industries",
        base_period=_make_period(2022, 203000, 2500, base_segs),
        current_period=_make_period(2023, 183000, 2650, curr_segs),
    )


class TestInstantiation:
    def test_creates(self):
        assert VarianceAnalysisEngine() is not None

    def test_version(self):
        assert VarianceAnalysisEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(VarianceAnalysisEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(VarianceAnalysisEngine(), "calculate_batch")


class TestBasicVariance:
    def test_basic_result(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_total_change(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.total_change_tco2e, Decimal)
        assert r.total_change_tco2e < Decimal("0")  # emissions decreased

    def test_total_change_pct(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.total_change_pct, Decimal)

    def test_base_and_current_years(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert r.base_year == 2022
        assert r.current_year == 2023

    def test_provenance(self):
        assert_provenance_hash(_run(VarianceAnalysisEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(VarianceAnalysisEngine().calculate(_make_input())))


class TestLMDIDecomposition:
    def test_lmdi_components_exist(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.lmdi_components, list)

    def test_lmdi_has_driver(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        for c in r.lmdi_components:
            assert c.driver != ""

    def test_lmdi_has_delta(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        for c in r.lmdi_components:
            assert isinstance(c.delta_tco2e, Decimal)


class TestWaterfall:
    def test_waterfall_steps(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.waterfall_steps, list)
        assert len(r.waterfall_steps) > 0

    def test_waterfall_has_labels(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        for step in r.waterfall_steps:
            assert step.label != ""


class TestKayaDecomposition:
    def test_kaya_exists(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert r.kaya_decomposition is not None

    def test_kaya_fields(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        k = r.kaya_decomposition
        assert isinstance(k.activity_effect_tco2e, Decimal)
        assert isinstance(k.energy_intensity_effect_tco2e, Decimal)
        assert isinstance(k.carbon_intensity_effect_tco2e, Decimal)


class TestSegmentAttribution:
    def test_segments(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input_with_segments()))
        assert isinstance(r.segment_attributions, list)


class TestScopeVariances:
    def test_scope_variances(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.scope_variances, list)


class TestTargetVariance:
    def test_target_variance(self):
        inp = _make_input(target_emissions_tco2e=Decimal("171000"))
        r = _run(VarianceAnalysisEngine().calculate(inp))
        assert isinstance(r.target_variance_tco2e, Decimal)


class TestSeverity:
    def test_overall_severity(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert r.overall_severity in ("critical", "significant", "moderate", "minor", "neutral", "positive", "favorable")


class TestScales:
    @pytest.mark.parametrize("base,curr", [
        (50000, 45000), (200000, 183000), (1000000, 920000),
        (5000000, 4600000), (50000000, 46000000),
    ])
    def test_various_scales(self, base, curr):
        r = _run(VarianceAnalysisEngine().calculate(VarianceAnalysisInput(
            entity_name="Test Corp",
            base_period=_make_period(2022, base, base // 100),
            current_period=_make_period(2023, curr, curr // 100 + 50),
        )))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C"])
    def test_entities(self, entity):
        r = _run(VarianceAnalysisEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity


class TestDecimalPrecision:
    def test_change_decimal(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.total_change_tco2e, Decimal)

    @pytest.mark.parametrize("val", ["123456.789", "999999.999", "1000000.001"])
    def test_precision(self, val):
        r = _run(VarianceAnalysisEngine().calculate(VarianceAnalysisInput(
            entity_name="Test",
            base_period=_make_period(2022, val),
            current_period=_make_period(2023, float(val) * 0.9),
        )))
        assert isinstance(r.total_change_tco2e, Decimal)


class TestRecommendationsWarnings:
    def test_recommendations(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.recommendations, list)

    def test_warnings(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert isinstance(r.warnings, list)

    def test_data_quality(self):
        r = _run(VarianceAnalysisEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(VarianceAnalysisEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = VarianceAnalysisEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(100):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(VarianceAnalysisEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_no_change(self):
        r = _run(VarianceAnalysisEngine().calculate(VarianceAnalysisInput(
            entity_name="NoChange",
            base_period=_make_period(2022, 100000),
            current_period=_make_period(2023, 100000),
        )))
        assert r.total_change_tco2e == Decimal("0")

    def test_model_dump(self):
        d = _run(VarianceAnalysisEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(VarianceAnalysisEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_increase(self):
        r = _run(VarianceAnalysisEngine().calculate(VarianceAnalysisInput(
            entity_name="Increase",
            base_period=_make_period(2022, 100000),
            current_period=_make_period(2023, 120000),
        )))
        assert r.total_change_tco2e > Decimal("0")

    def test_intermediate_periods(self):
        r = _run(VarianceAnalysisEngine().calculate(VarianceAnalysisInput(
            entity_name="Multi",
            base_period=_make_period(2020, 200000),
            current_period=_make_period(2023, 170000),
            intermediate_periods=[
                _make_period(2021, 190000),
                _make_period(2022, 180000),
            ],
        )))
        assert r is not None
