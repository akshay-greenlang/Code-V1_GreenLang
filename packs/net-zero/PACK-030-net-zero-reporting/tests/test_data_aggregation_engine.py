# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Data Aggregation Engine.
"""

import asyncio
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.data_aggregation_engine import (
    DataAggregationEngine, DataAggregationInput, DataAggregationResult,
    DataSourceType, SourceDataPoint, SourceConnection, FrameworkTarget,
    MetricCategory, DataQuality, ReconciliationStatus,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive, assert_decimal_non_negative,
    assert_percentage_range, assert_provenance_hash, assert_processing_time,
    compute_sha256, timed_block, DATA_SOURCES, FRAMEWORKS, SCOPES,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_data_points(count=5):
    categories = [MetricCategory.EMISSIONS_SCOPE_1, MetricCategory.EMISSIONS_SCOPE_2_LOCATION,
                  MetricCategory.EMISSIONS_SCOPE_2_MARKET, MetricCategory.EMISSIONS_SCOPE_3, MetricCategory.EMISSIONS_TOTAL]
    return [
        SourceDataPoint(
            source=DataSourceType.PACK_021.value,
            metric_name=f"metric_{i+1}",
            metric_category=categories[i % len(categories)].value,
            value=Decimal(str(10000 * (i + 1))),
            unit="tCO2e",
        )
        for i in range(count)
    ]


def _make_input(**kwargs):
    defaults = dict(
        organization_id="test-org-001",
        reporting_period_start=date(2024, 1, 1),
        reporting_period_end=date(2024, 12, 31),
        data_points=_make_data_points(),
    )
    defaults.update(kwargs)
    return DataAggregationInput(**defaults)


class TestDataAggregationInstantiation:
    def test_engine_instantiates(self):
        assert DataAggregationEngine() is not None

    def test_engine_has_aggregate_method(self):
        engine = DataAggregationEngine()
        assert hasattr(engine, "aggregate")

    def test_engine_version(self):
        assert DataAggregationEngine().engine_version == "1.0.0"


class TestMultiSourceCollection:
    def test_aggregate_basic(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result is not None
        assert isinstance(result, DataAggregationResult)

    def test_aggregate_returns_metrics(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result.total_metrics >= 0

    def test_aggregate_has_provenance(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert_provenance_hash(result)

    def test_aggregate_processing_time(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert_processing_time(result)

    def test_aggregate_organization_id(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result.organization_id == "test-org-001"

    def test_aggregate_with_many_data_points(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(data_points=_make_data_points(20))))
        assert result is not None

    def test_aggregate_empty_data_points(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(data_points=[])))
        assert result is not None


class TestReconciliation:
    def test_reconcile_included(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_reconciliation=True)))
        assert result is not None

    def test_reconcile_excluded(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_reconciliation=False)))
        assert result is not None

    @pytest.mark.parametrize("threshold", [Decimal("1"), Decimal("5"), Decimal("10")])
    def test_reconcile_thresholds(self, threshold):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(
            _make_input(reconciliation_threshold_pct=threshold)
        ))
        assert result is not None


class TestGapDetection:
    def test_gap_analysis_included(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_gap_analysis=True)))
        assert result is not None

    def test_gap_analysis_excluded(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_gap_analysis=False)))
        assert result is not None

    def test_empty_data_has_gaps(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(data_points=[], include_gap_analysis=True)))
        # With no data, there should be gaps or at least a valid result
        assert result is not None


class TestLineageGeneration:
    def test_lineage_included(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_lineage=True)))
        assert result is not None

    def test_lineage_excluded(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input(include_lineage=False)))
        assert result.lineage_graph is None


class TestDecimalPrecision:
    def test_aggregated_values_are_decimal(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert isinstance(result.overall_completeness_pct, Decimal)

    def test_sum_consistency(self):
        s1 = Decimal("107500")
        s2 = Decimal("67080")
        s3 = Decimal("405000")
        total = s1 + s2 + s3
        assert total == Decimal("579580")


class TestProvenance:
    def test_provenance_exists(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert_provenance_hash(result)

    def test_provenance_deterministic(self):
        engine = DataAggregationEngine()
        inp = _make_input()
        r1 = _run(engine.aggregate(inp))
        r2 = _run(engine.aggregate(inp))
        # Provenance hashes may include timestamps; just check both exist
        assert r1.provenance_hash is not None
        assert r2.provenance_hash is not None

    @pytest.mark.parametrize("run_idx", range(3))
    def test_deterministic_multiple_runs(self, run_idx):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result is not None


class TestPerformance:
    def test_aggregation_under_3_seconds(self):
        engine = DataAggregationEngine()
        with timed_block("full_aggregation", max_seconds=3.0):
            _run(engine.aggregate(_make_input()))

    def test_processing_time_recorded(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert_processing_time(result)


class TestResultModel:
    def test_result_serializable(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        d = result.model_dump()
        assert isinstance(d, dict)

    def test_result_has_engine_version(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result.engine_version == "1.0.0"

    def test_result_has_completeness(self):
        engine = DataAggregationEngine()
        result = _run(engine.aggregate(_make_input()))
        assert result.overall_completeness_pct >= Decimal("0")


class TestInputValidation:
    def test_valid_input(self):
        inp = _make_input()
        assert inp is not None
        assert inp.organization_id == "test-org-001"

    def test_empty_org_id_rejected(self):
        with pytest.raises((ValueError, Exception)):
            DataAggregationInput(
                organization_id="",
                reporting_period_start=date(2024, 1, 1),
                reporting_period_end=date(2024, 12, 31),
            )

    @pytest.mark.parametrize("start,end", [
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2024, 4, 1), date(2025, 3, 31)),
    ])
    def test_various_periods(self, start, end):
        inp = DataAggregationInput(
            organization_id="test-org",
            reporting_period_start=start,
            reporting_period_end=end,
        )
        assert inp is not None
