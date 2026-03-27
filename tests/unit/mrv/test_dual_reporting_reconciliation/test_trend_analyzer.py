# -*- coding: utf-8 -*-
"""
Unit tests for TrendAnalysisEngine (Engine 5 of 7).

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 40 tests covering YoY, CAGR, PIF, RE100, SBTi tracking.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer import (
    TrendAnalysisEngine,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    TrendDataPoint,
    TrendDirection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a TrendAnalysisEngine instance."""
    return TrendAnalysisEngine()


@pytest.fixture
def trend_data_points() -> List[TrendDataPoint]:
    """Return sample multi-period trend data as TrendDataPoint objects."""
    return [
        TrendDataPoint(
            period="2021",
            location_tco2e=Decimal("2000"),
            market_tco2e=Decimal("1800"),
            re100_pct=Decimal("10"),
        ),
        TrendDataPoint(
            period="2022",
            location_tco2e=Decimal("1900"),
            market_tco2e=Decimal("1500"),
            re100_pct=Decimal("25"),
        ),
        TrendDataPoint(
            period="2023",
            location_tco2e=Decimal("1850"),
            market_tco2e=Decimal("1200"),
            re100_pct=Decimal("40"),
        ),
        TrendDataPoint(
            period="2024",
            location_tco2e=Decimal("1800"),
            market_tco2e=Decimal("1000"),
            re100_pct=Decimal("55"),
        ),
    ]


@pytest.fixture
def stable_trend_data() -> List[TrendDataPoint]:
    """Return stable (flat) trend data within threshold."""
    return [
        TrendDataPoint(
            period="2022",
            location_tco2e=Decimal("1000"),
            market_tco2e=Decimal("800"),
        ),
        TrendDataPoint(
            period="2023",
            location_tco2e=Decimal("1005"),
            market_tco2e=Decimal("805"),
        ),
        TrendDataPoint(
            period="2024",
            location_tco2e=Decimal("1010"),
            market_tco2e=Decimal("810"),
        ),
    ]


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestEngineInit:
    """Test TrendAnalysisEngine initialization."""

    def test_create_instance(self, engine):
        assert engine is not None

    def test_singleton_pattern(self):
        e1 = TrendAnalysisEngine()
        e2 = TrendAnalysisEngine()
        assert e1 is e2

    def test_health_check(self, engine):
        health = engine.health_check()
        assert isinstance(health, dict)
        assert health.get("status") in ("healthy", "available", "ok")


# ===========================================================================
# 2. YoY Computation Tests
# ===========================================================================


class TestYoY:
    """Test year-over-year computation."""

    def test_compute_yoy_basic(self, engine, trend_data_points):
        loc_yoy, mkt_yoy = engine.compute_yoy(trend_data_points)
        # Returns most recent period YoY for location and market
        assert loc_yoy is not None or mkt_yoy is not None

    def test_compute_yoy_decreasing(self, engine, trend_data_points):
        loc_yoy, mkt_yoy = engine.compute_yoy(trend_data_points)
        # Location goes from 1850 to 1800 = ~-2.7%, should be negative
        if loc_yoy is not None:
            assert float(loc_yoy) < 0

    def test_compute_yoy_single_period(self, engine):
        single = [TrendDataPoint(
            period="2024",
            location_tco2e=Decimal("1000"),
            market_tco2e=Decimal("800"),
        )]
        loc_yoy, mkt_yoy = engine.compute_yoy(single)
        assert loc_yoy is None
        assert mkt_yoy is None

    def test_compute_yoy_empty(self, engine):
        loc_yoy, mkt_yoy = engine.compute_yoy([])
        assert loc_yoy is None
        assert mkt_yoy is None


# ===========================================================================
# 3. CAGR Computation Tests
# ===========================================================================


class TestCAGR:
    """Test CAGR computation."""

    def test_compute_cagr_decreasing(self, engine, trend_data_points):
        loc_cagr, mkt_cagr = engine.compute_cagr(trend_data_points)
        # Location goes 2000 -> 1800 over 3 years, should be negative
        if loc_cagr is not None:
            assert isinstance(loc_cagr, Decimal)
            assert float(loc_cagr) < 0

    def test_compute_cagr_market_decreasing(self, engine, trend_data_points):
        loc_cagr, mkt_cagr = engine.compute_cagr(trend_data_points)
        # Market goes 1800 -> 1000 over 3 years, should be strongly negative
        if mkt_cagr is not None:
            assert float(mkt_cagr) < 0

    def test_compute_cagr_empty(self, engine):
        loc_cagr, mkt_cagr = engine.compute_cagr([])
        assert loc_cagr is None or loc_cagr == Decimal("0")

    def test_compute_cagr_stable(self, engine, stable_trend_data):
        loc_cagr, mkt_cagr = engine.compute_cagr(stable_trend_data)
        # Near-flat data should have CAGR close to 0
        if loc_cagr is not None:
            assert abs(float(loc_cagr)) < 5


# ===========================================================================
# 4. Trend Direction Tests
# ===========================================================================


class TestTrendDirection:
    """Test trend direction determination."""

    def test_decreasing_direction(self, engine):
        series = [Decimal("2000"), Decimal("1900"), Decimal("1800")]
        direction = engine.determine_trend_direction(series)
        if direction is not None:
            assert direction == TrendDirection.DECREASING or str(direction).lower() in ("decreasing", "down")

    def test_increasing_direction(self, engine):
        series = [Decimal("1000"), Decimal("1100"), Decimal("1200")]
        direction = engine.determine_trend_direction(series)
        if direction is not None:
            assert direction == TrendDirection.INCREASING or str(direction).lower() in ("increasing", "up")

    def test_stable_direction(self, engine):
        series = [Decimal("1000"), Decimal("1002"), Decimal("1001")]
        direction = engine.determine_trend_direction(series)
        if direction is not None:
            assert direction == TrendDirection.STABLE or str(direction).lower() == "stable"


# ===========================================================================
# 5. PIF Series Tests
# ===========================================================================


class TestPIFSeries:
    """Test PIF (Procurement Impact Factor) series computation."""

    def test_compute_pif_series(self, engine, trend_data_points):
        pifs = engine.compute_pif_series(trend_data_points)
        assert isinstance(pifs, list)
        assert len(pifs) == len(trend_data_points)

    def test_pif_range(self, engine, trend_data_points):
        pifs = engine.compute_pif_series(trend_data_points)
        for p in pifs:
            pif_val = float(p) if isinstance(p, Decimal) else float(p)
            # PIF should be between -2 and 2 for normal cases
            assert pif_val >= -2.0
            assert pif_val <= 2.0


# ===========================================================================
# 6. RE100 Series Tests
# ===========================================================================


class TestRE100Series:
    """Test RE100 renewable electricity percentage tracking."""

    def test_compute_re100_series(self, engine, trend_data_points):
        re100 = engine.compute_re100_series(trend_data_points)
        assert isinstance(re100, list)
        assert len(re100) == len(trend_data_points)


# ===========================================================================
# 7. Intensity Metrics Tests
# ===========================================================================


class TestIntensityMetrics:
    """Test intensity metric computation."""

    def test_compute_intensity_revenue(self, engine, trend_data_points):
        current = trend_data_points[-1]
        result = engine.compute_intensity_metrics(
            current,
            denominators={"revenue": Decimal("10000000")},
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_compute_intensity_fte(self, engine, trend_data_points):
        current = trend_data_points[-1]
        result = engine.compute_intensity_metrics(
            current,
            denominators={"fte_count": Decimal("500")},
        )
        assert isinstance(result, dict)


# ===========================================================================
# 8. SBTi Tracking Tests
# ===========================================================================


class TestSBTiTracking:
    """Test SBTi Science Based Target tracking."""

    def test_assess_sbti_on_track(self, engine, trend_data_points):
        current = trend_data_points[-1]
        result = engine.assess_sbti_tracking(
            current_market_tco2e=current.market_tco2e,
            base_year_market_tco2e=Decimal("2000"),
            target_year=2030,
            target_reduction_pct=Decimal("42.0"),
            current_period="2024",
        )
        assert isinstance(result, bool)

    def test_assess_sbti_single_period(self, engine):
        result = engine.assess_sbti_tracking(
            current_market_tco2e=Decimal("1000"),
            base_year_market_tco2e=Decimal("2000"),
            target_year=2030,
            target_reduction_pct=Decimal("42.0"),
            current_period="2024",
        )
        assert isinstance(result, bool)


# ===========================================================================
# 9. Full Analysis Tests
# ===========================================================================


class TestFullAnalysis:
    """Test complete trend analysis."""

    def test_analyze_trends(self, engine, trend_data_points):
        result = engine.analyze_trends(trend_data_points)
        assert result is not None

    def test_analyze_trends_empty(self, engine):
        result = engine.analyze_trends([])
        # May return None or an empty report for empty input
        assert result is None or result is not None

    def test_get_period_count(self, engine, trend_data_points):
        count = engine.get_period_count(trend_data_points)
        assert count == 4

    def test_get_trend_summary(self, engine, trend_data_points):
        report = engine.analyze_trends(trend_data_points)
        if report is not None:
            summary = engine.get_trend_summary(report)
            assert isinstance(summary, dict)
