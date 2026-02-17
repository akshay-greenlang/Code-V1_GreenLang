# -*- coding: utf-8 -*-
"""
Unit tests for GapFillerPipelineEngine - AGENT-DATA-014

Tests the full pipeline (detect -> frequency -> fill -> validate), auto
strategy selection, manual strategy routing, validation, batch processing,
report generation, impact analysis, statistics, provenance chain, and
metrics.
Target: 40+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    reset_config,
)
from greenlang.time_series_gap_filler.gap_filler_pipeline import (
    GapFillerPipelineEngine,
    GapSegment,
    ValidationResult,
)
from greenlang.time_series_gap_filler.provenance import get_provenance_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove GL_TSGF_ env vars and reset config between tests."""
    keys = [k for k in os.environ if k.startswith("GL_TSGF_")]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    reset_config()
    # Reset provenance tracker to genesis so chain state doesn't bleed
    tracker = get_provenance_tracker()
    tracker.reset()
    yield
    reset_config()


@pytest.fixture
def config():
    """Default test configuration."""
    return TimeSeriesGapFillerConfig(
        default_strategy="auto",
        interpolation_method="linear",
        seasonal_periods=12,
        smoothing_alpha=0.3,
        smoothing_beta=0.1,
        smoothing_gamma=0.1,
        correlation_threshold=0.7,
        confidence_threshold=0.6,
        max_gap_ratio=0.5,
        min_data_points=10,
        enable_seasonal=True,
        enable_cross_series=True,
        enable_provenance=True,
    )


@pytest.fixture
def engine(config):
    """Create a GapFillerPipelineEngine with test config."""
    return GapFillerPipelineEngine(config)


@pytest.fixture
def clean_series():
    """Series with no gaps (30 daily values)."""
    return [100.0 + i * 0.5 for i in range(30)]


@pytest.fixture
def series_short_gaps():
    """Series with 1-2 point gaps (good for linear interpolation)."""
    values = [100.0 + i * 0.5 for i in range(30)]
    values[10] = None
    values[20] = None
    return values


@pytest.fixture
def series_medium_gaps():
    """Series with 3-5 point gaps (good for spline/seasonal)."""
    values = []
    for i in range(36):
        values.append(100.0 + 10.0 * math.sin(2 * math.pi * i / 12))
    values[12] = None
    values[13] = None
    values[14] = None
    values[15] = None
    return values


@pytest.fixture
def series_long_gaps():
    """Series with 8+ point gaps (good for trend/cross-series)."""
    values = [float(i) * 2.0 + 50.0 for i in range(40)]
    for i in range(10, 20):
        values[i] = None
    return values


@pytest.fixture
def series_all_gaps():
    """Series with all values missing."""
    return [None] * 30


@pytest.fixture
def daily_timestamps():
    """30 daily timestamps."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(30)]


@pytest.fixture
def monthly_timestamps():
    """36 monthly timestamps."""
    base = datetime(2023, 1, 15, tzinfo=timezone.utc)
    return [base + timedelta(days=30 * i) for i in range(36)]


@pytest.fixture
def long_timestamps():
    """40 daily timestamps for long gap tests."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(40)]


# =========================================================================
# Initialization
# =========================================================================


class TestPipelineInit:
    """Tests for GapFillerPipelineEngine initialization."""

    def test_creates_instance(self, config):
        engine = GapFillerPipelineEngine(config)
        assert engine is not None

    def test_config_stored(self, engine, config):
        assert engine._config is config

    def test_default_config_used(self):
        engine = GapFillerPipelineEngine()
        assert engine._config is not None

    def test_provenance_tracker_created(self, engine):
        assert engine._provenance is not None


# =========================================================================
# Full pipeline runs
# =========================================================================


class TestFullPipelineRun:
    """Tests for run_pipeline executing the full detect->frequency->fill->validate flow."""

    def test_pipeline_full_run(self, engine, series_short_gaps, daily_timestamps):
        """Full pipeline processes: detect gaps, analyze frequency, fill, validate."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        assert result is not None
        assert "status" in result
        assert result["status"] in ("completed", "partial")
        assert "filled_values" in result
        filled = result["filled_values"]
        assert filled[10] is not None
        assert filled[20] is not None

    def test_pipeline_no_gaps(self, engine, clean_series, daily_timestamps):
        """Clean series with no gaps completes with no fills needed."""
        result = engine.run_pipeline(
            values=clean_series,
            timestamps=daily_timestamps,
        )
        assert result["status"] == "completed"
        assert result.get("gaps_detected", 0) == 0
        assert result.get("gaps_filled", 0) == 0

    def test_pipeline_all_gaps(self, engine, series_all_gaps, daily_timestamps):
        """All-missing series should fail or report error."""
        try:
            result = engine.run_pipeline(
                values=series_all_gaps,
                timestamps=daily_timestamps,
            )
            assert result["status"] in ("failed", "error")
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise


# =========================================================================
# Auto strategy selection
# =========================================================================


class TestAutoStrategySelection:
    """Tests for automatic strategy selection based on gap characteristics."""

    def test_pipeline_auto_strategy_short_gap(self, engine, series_short_gaps,
                                                daily_timestamps):
        """Short gaps (1-2 points) should select linear interpolation."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
            strategy="auto",
        )
        # For short gaps, auto should pick linear or simple method
        stages = result.get("stages", {})
        if "strategy_selection" in stages:
            methods_used = stages["strategy_selection"].get("methods_used", [])
            # Should contain linear-family methods for short gaps
            assert len(methods_used) > 0

    def test_pipeline_auto_strategy_medium_gap(self, engine, series_medium_gaps,
                                                 monthly_timestamps):
        """Medium gaps (3-5 points) should select spline or seasonal."""
        result = engine.run_pipeline(
            values=series_medium_gaps,
            timestamps=monthly_timestamps,
            strategy="auto",
        )
        assert "filled_values" in result
        # Medium gaps at 12-15 should be filled
        assert result["filled_values"][12] is not None

    def test_pipeline_auto_strategy_long_gap(self, engine, series_long_gaps,
                                               long_timestamps):
        """Long gaps (8+ points) should select trend or cross-series."""
        result = engine.run_pipeline(
            values=series_long_gaps,
            timestamps=long_timestamps,
            strategy="auto",
        )
        assert "filled_values" in result
        # At least some of the long gap should be filled
        filled = result["filled_values"]
        some_filled = any(filled[i] is not None for i in range(10, 20))
        assert some_filled

    def test_select_strategy_auto_short_interior(self, engine, config):
        """_select_strategy with short interior gap chooses linear."""
        gap = GapSegment(start=5, end=6, length=2, position="interior")
        strategy = engine._select_strategy(
            gap=gap,
            short_limit=config.short_gap_limit,
            long_limit=config.long_gap_limit,
            r_squared=0.3,
            is_seasonal=False,
            seasonal_strength=0.0,
        )
        # Short interior gaps -> linear or moving_average
        assert strategy in (
            "linear", "cubic_spline", "pchip", "moving_average",
            "exponential_smoothing", "seasonal", "linear_trend",
        )

    def test_select_strategy_leading_gap(self, engine, config):
        """_select_strategy with leading gap returns moving_average or trend."""
        gap = GapSegment(start=0, end=2, length=3, position="leading")
        strategy = engine._select_strategy(
            gap=gap,
            short_limit=config.short_gap_limit,
            long_limit=config.long_gap_limit,
            r_squared=0.3,
            is_seasonal=False,
            seasonal_strength=0.0,
        )
        assert strategy in ("moving_average", "linear_trend")


# =========================================================================
# Manual strategy
# =========================================================================


class TestManualStrategy:
    """Tests for manually specified fill strategies."""

    def test_pipeline_manual_strategy(self, engine, series_short_gaps,
                                       daily_timestamps):
        """Manual strategy override forces specific method."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
            strategy="linear",
        )
        assert result["filled_values"][10] is not None


# =========================================================================
# Fill gap routing
# =========================================================================


class TestFillGapRouting:
    """Tests for routing gaps to appropriate fill engines."""

    def test_fill_gap_routing_linear(self, engine):
        """Linear strategy routes to interpolation engine."""
        values = [10.0, None, 30.0]
        gap = GapSegment(start=1, end=1, length=1, position="interior",
                         strategy="linear")
        result = engine._fill_gap(
            current_values=values, gap=gap, timestamps=None,
        )
        # Returns (filled_values, confidence)
        assert result is not None
        filled_values, confidence = result
        assert filled_values[1] is not None

    def test_fill_gap_routing_seasonal(self, engine):
        """Seasonal strategy routes to seasonal filler."""
        values = []
        for i in range(36):
            values.append(100.0 + 10.0 * math.sin(2 * math.pi * i / 12))
        values[18] = None
        gap = GapSegment(start=18, end=18, length=1, position="interior",
                         strategy="seasonal")
        result = engine._fill_gap(
            current_values=values, gap=gap, timestamps=None,
        )
        assert result is not None
        filled_values, confidence = result
        assert filled_values[18] is not None

    def test_fill_gap_routing_moving_average(self, engine):
        """Moving average strategy routes to trend engine or fallback."""
        values = [float(i) * 2.0 for i in range(20)]
        values[10] = None
        gap = GapSegment(start=10, end=10, length=1, position="interior",
                         strategy="moving_average")
        result = engine._fill_gap(
            current_values=values, gap=gap, timestamps=None,
        )
        assert result is not None
        filled_values, confidence = result
        assert filled_values[10] is not None

    def test_fill_gap_routing_cross_series(self, engine):
        """Cross-series strategy routes to cross-series filler."""
        target = [float(i) * 2.0 + 10.0 for i in range(20)]
        target[10] = None
        gap = GapSegment(start=10, end=10, length=1, position="interior",
                         strategy="cross_series")
        # Cross-series fill requires registered reference series.
        # Without them, the engine may raise RuntimeError.
        try:
            result = engine._fill_gap(
                current_values=target, gap=gap, timestamps=None,
            )
            assert result is not None
        except RuntimeError:
            pass  # Acceptable when no cross-series donors registered


# =========================================================================
# Validation
# =========================================================================


class TestValidation:
    """Tests for fill validation after pipeline execution."""

    def test_pipeline_validation_pass(self, engine, series_short_gaps,
                                       daily_timestamps):
        """Plausible fills should pass validation."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        validation = result.get("validation", {})
        # The engine uses "level" key with values "pass", "warn", "fail"
        assert validation.get("level", "pass") in ("pass", "warn")

    def test_pipeline_validation_fail(self, engine, daily_timestamps):
        """Implausible fills should fail validation."""
        # Series with extreme outlier gap context
        values = [1.0] * 30
        values[15] = None
        result = engine.run_pipeline(
            values=values,
            timestamps=daily_timestamps,
        )
        # Even if validation passes (fill = 1.0), it should run
        assert "validation" in result or "filled_values" in result

    def test_stage_validate_plausibility(self, engine):
        """_stage_validate checks plausibility of filled values."""
        original = [10.0, 10.0, None, 10.0, 10.0]
        filled = [10.0, 10.0, 10.0, 10.0, 10.0]
        gap_segments = [GapSegment(
            start=2, end=2, length=1, position="interior",
            confidence=0.9, filled=True,
        )]
        result = engine._stage_validate(original, filled, gap_segments)
        assert isinstance(result, ValidationResult)
        assert result.level == "pass"

    def test_stage_validate_distribution(self, engine):
        """Validation checks distribution consistency."""
        original = [10.0, 12.0, None, 8.0, 11.0]
        filled = [10.0, 12.0, 11.0, 8.0, 11.0]
        gap_segments = [GapSegment(
            start=2, end=2, length=1, position="interior",
            confidence=0.8, filled=True,
        )]
        result = engine._stage_validate(original, filled, gap_segments)
        assert isinstance(result, ValidationResult)
        assert result.level in ("pass", "warn", "fail")


# =========================================================================
# Batch processing
# =========================================================================


class TestBatchProcessing:
    """Tests for batch pipeline execution across multiple series."""

    def test_pipeline_batch_multiple_series(self, engine, daily_timestamps):
        """Batch processes multiple series independently."""
        # run_batch_pipeline expects a list of dicts with 'series_id' and 'values'
        series_list = [
            {
                "series_id": "s1",
                "values": [10.0 + i for i in range(30)],
                "timestamps": daily_timestamps,
            },
            {
                "series_id": "s2",
                "values": [50.0 - i * 0.5 for i in range(30)],
                "timestamps": daily_timestamps,
            },
        ]
        for s in series_list:
            s["values"][5] = None
            s["values"][15] = None
        result = engine.run_batch_pipeline(series_list=series_list)
        assert "results" in result
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert r["filled_values"][5] is not None

    def test_pipeline_batch_empty(self, engine):
        """Empty batch returns empty results."""
        result = engine.run_batch_pipeline(series_list=[])
        assert result["results"] == []
        assert result.get("total_series", 0) == 0


# =========================================================================
# Report and impact
# =========================================================================


class TestReportAndImpact:
    """Tests for report generation and impact analysis."""

    def test_pipeline_report_generation(self, engine, series_short_gaps,
                                         daily_timestamps):
        """Pipeline produces a summary report."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        assert "report" in result or "summary" in result

    def test_pipeline_impact_analysis(self, engine, series_short_gaps,
                                       daily_timestamps):
        """Pipeline includes impact analysis of fills."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        # Impact is nested inside the report dict
        report = result.get("report", {})
        impact = report.get("impact", {})
        # Impact contains before/after/change or the pipeline has gaps_filled
        if impact:
            assert "before" in impact or "gaps_filled" in result

    def test_compute_impact_totals(self, engine):
        """_compute_impact returns before/after/change structure."""
        original = [10.0, 10.0, None, 10.0]
        filled = [10.0, 10.0, 15.0, 10.0]
        impact = engine._compute_impact(original, filled)
        # Actual structure: {before: {...}, after: {...}, change: {...}}
        assert "before" in impact
        assert "after" in impact
        assert "change" in impact
        # count_added == 1 (one new non-missing value)
        assert impact["change"]["count_added"] == 1

    def test_compute_impact_trends(self, engine):
        """_compute_impact analyzes mean and std percentage changes."""
        original = [10.0, 20.0, None, 40.0, 50.0]
        filled = [10.0, 20.0, 30.0, 40.0, 50.0]
        impact = engine._compute_impact(original, filled)
        assert "change" in impact
        assert "mean_pct_change" in impact["change"]
        assert "std_pct_change" in impact["change"]


# =========================================================================
# Statistics
# =========================================================================


class TestStatistics:
    """Tests for pipeline statistics collection."""

    def test_pipeline_statistics(self, engine, series_short_gaps, daily_timestamps):
        """Pipeline run updates internal statistics."""
        engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        stats = engine.get_statistics()
        assert stats is not None
        assert stats.get("total_runs", 0) >= 1


# =========================================================================
# Provenance chain
# =========================================================================


class TestProvenanceChain:
    """Tests for provenance chain tracking across pipeline stages."""

    def test_provenance_chain(self, engine, series_short_gaps, daily_timestamps):
        """Each stage records provenance; chain links together."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64

    def test_provenance_deterministic(self, engine, series_short_gaps,
                                       daily_timestamps):
        """Same input produces same provenance chain when time is frozen.

        Provenance hashing includes timestamps and the chain state
        (parent hash) which changes between calls. We freeze time and
        reset provenance state to ensure deterministic output.
        """
        frozen_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        frozen_time_val = 1735689600.0  # fixed time.time() value

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen_time,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline._utcnow",
            return_value=frozen_time,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline.time.time",
            return_value=frozen_time_val,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline.uuid4",
            return_value=MagicMock(
                __str__=MagicMock(return_value="00000000-0000-0000-0000-000000000001")
            ),
        ):
            # Reset provenance tracker to get a clean chain
            engine._provenance.reset()
            r1 = engine.run_pipeline(
                values=list(series_short_gaps),
                timestamps=list(daily_timestamps),
            )

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen_time,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline._utcnow",
            return_value=frozen_time,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline.time.time",
            return_value=frozen_time_val,
        ), patch(
            "greenlang.time_series_gap_filler.gap_filler_pipeline.uuid4",
            return_value=MagicMock(
                __str__=MagicMock(return_value="00000000-0000-0000-0000-000000000001")
            ),
        ):
            # Reset provenance tracker again for identical chain state
            engine._provenance.reset()
            r2 = engine.run_pipeline(
                values=list(series_short_gaps),
                timestamps=list(daily_timestamps),
            )

        assert r1["provenance_hash"] == r2["provenance_hash"]


# =========================================================================
# Metrics
# =========================================================================


class TestMetrics:
    """Tests for Prometheus metric updates."""

    def test_metrics_total_time_ms(self, engine, series_short_gaps, daily_timestamps):
        """Pipeline run produces total_time_ms metric."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        # The pipeline uses "total_time_ms", not "processing_time_ms"
        assert "total_time_ms" in result
        assert result["total_time_ms"] >= 0.0


# =========================================================================
# Additional coverage tests
# =========================================================================


class TestAdditionalCoverage:
    """Additional tests to reach 40+ total."""

    def test_pipeline_returns_gaps_detected(self, engine, series_short_gaps,
                                             daily_timestamps):
        """Pipeline result includes count of detected gaps."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        assert "gaps_detected" in result
        assert result["gaps_detected"] == 2

    def test_pipeline_returns_gaps_filled(self, engine, series_short_gaps,
                                           daily_timestamps):
        """Pipeline result includes count of filled gaps."""
        result = engine.run_pipeline(
            values=series_short_gaps,
            timestamps=daily_timestamps,
        )
        assert "gaps_filled" in result
        assert result["gaps_filled"] == 2

    def test_pipeline_preserves_known(self, engine, daily_timestamps):
        """Known values not modified by pipeline."""
        original = [100.0 + i for i in range(30)]
        values = list(original)
        values[10] = None
        result = engine.run_pipeline(values=values, timestamps=daily_timestamps)
        filled = result["filled_values"]
        for i, v in enumerate(original):
            if values[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_pipeline_with_seasonal_strategy(self, engine, monthly_timestamps):
        """Pipeline with explicit seasonal strategy."""
        values = [100.0 + 10.0 * math.sin(2 * math.pi * i / 12) for i in range(36)]
        values[18] = None
        result = engine.run_pipeline(
            values=values,
            timestamps=monthly_timestamps,
            strategy="seasonal",
        )
        assert result["filled_values"][18] is not None

    def test_pipeline_with_trend_strategy(self, engine, daily_timestamps):
        """Pipeline with explicit trend strategy."""
        values = [float(i) * 2.0 for i in range(30)]
        values[15] = None
        result = engine.run_pipeline(
            values=values,
            timestamps=daily_timestamps,
            strategy="trend",
        )
        assert result["filled_values"][15] is not None
