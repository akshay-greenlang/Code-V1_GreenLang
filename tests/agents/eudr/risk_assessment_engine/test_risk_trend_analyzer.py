# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 RiskTrendAnalyzer.

Tests data point addition, trend direction detection (IMPROVING, DEGRADING,
STABLE, INSUFFICIENT_DATA), window-based score change calculations, and
risk regime change detection.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    RiskLevel,
    TrendDirection,
)


def _make_analyzer():
    """Instantiate RiskTrendAnalyzer with mocked dependencies."""
    from greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer import (
        RiskTrendAnalyzer,
    )
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.trend_window_days = 365
    cfg.min_data_points = 3
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer.record_trend_analysis"
    ):
        return RiskTrendAnalyzer(config=cfg)


class TestAddDataPoint:
    """Test adding data points to the trend history."""

    def test_add_data_point(self):
        analyzer = _make_analyzer()
        point = analyzer.add_data_point(
            operator_id="OP-001",
            commodity="cocoa",
            score=Decimal("45"),
            level=RiskLevel.STANDARD,
        )
        assert point is not None
        assert point.score == Decimal("45.00")
        assert point.level == RiskLevel.STANDARD

    def test_add_multiple_data_points(self):
        analyzer = _make_analyzer()
        for score in [Decimal("40"), Decimal("42"), Decimal("38")]:
            analyzer.add_data_point("OP-001", "cocoa", score, RiskLevel.STANDARD)

        data = analyzer.get_trend_data("OP-001", "cocoa")
        assert len(data) == 3


class TestAnalyzeTrend:
    """Test trend direction detection."""

    def test_analyze_trend_improving(self):
        """Monotonically decreasing scores -> IMPROVING."""
        analyzer = _make_analyzer()
        scores = [Decimal("60"), Decimal("50"), Decimal("40")]
        for s in scores:
            analyzer.add_data_point("OP-001", "cocoa", s, RiskLevel.STANDARD)

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer.record_trend_analysis"
        ):
            trend = analyzer.analyze_trend("OP-001", "cocoa")

        assert trend.direction == TrendDirection.IMPROVING

    def test_analyze_trend_degrading(self):
        """Monotonically increasing scores -> DEGRADING."""
        analyzer = _make_analyzer()
        scores = [Decimal("30"), Decimal("45"), Decimal("60")]
        for s in scores:
            analyzer.add_data_point("OP-001", "cocoa", s, RiskLevel.STANDARD)

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer.record_trend_analysis"
        ):
            trend = analyzer.analyze_trend("OP-001", "cocoa")

        assert trend.direction == TrendDirection.DEGRADING

    def test_analyze_trend_stable(self):
        """Flat scores within threshold -> STABLE."""
        analyzer = _make_analyzer()
        scores = [Decimal("50"), Decimal("51"), Decimal("50")]
        for s in scores:
            analyzer.add_data_point("OP-001", "cocoa", s, RiskLevel.STANDARD)

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer.record_trend_analysis"
        ):
            trend = analyzer.analyze_trend("OP-001", "cocoa")

        assert trend.direction == TrendDirection.STABLE

    def test_analyze_trend_insufficient_data(self):
        """Fewer than 3 data points -> INSUFFICIENT_DATA."""
        analyzer = _make_analyzer()
        analyzer.add_data_point("OP-001", "cocoa", Decimal("50"), RiskLevel.STANDARD)

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer.record_trend_analysis"
        ):
            trend = analyzer.analyze_trend("OP-001", "cocoa")

        assert trend.direction == TrendDirection.INSUFFICIENT_DATA


class TestGetTrendData:
    """Test trend data retrieval within time window."""

    def test_get_trend_data_within_window(self):
        analyzer = _make_analyzer()
        for i in range(5):
            analyzer.add_data_point(
                "OP-001", "cocoa",
                Decimal(str(40 + i)),
                RiskLevel.STANDARD,
            )

        data = analyzer.get_trend_data("OP-001", "cocoa", days=365)
        assert len(data) == 5

    def test_get_trend_data_empty(self):
        analyzer = _make_analyzer()
        data = analyzer.get_trend_data("OP-MISSING", "cocoa")
        assert data == []


class TestRegimeChange:
    """Test risk regime change detection."""

    def test_detect_risk_regime_change(self):
        """Level change between last two points -> detected."""
        analyzer = _make_analyzer()
        analyzer.add_data_point("OP-001", "cocoa", Decimal("45"), RiskLevel.STANDARD)
        analyzer.add_data_point("OP-001", "cocoa", Decimal("70"), RiskLevel.HIGH)

        change = analyzer.detect_risk_regime_change("OP-001", "cocoa")
        assert change is not None
        assert change["detected"] is True
        assert change["change_type"] == "escalation"
        assert change["previous_level"] == "standard"
        assert change["current_level"] == "high"

    def test_detect_no_regime_change(self):
        """Same level on last two points -> None."""
        analyzer = _make_analyzer()
        analyzer.add_data_point("OP-001", "cocoa", Decimal("45"), RiskLevel.STANDARD)
        analyzer.add_data_point("OP-001", "cocoa", Decimal("50"), RiskLevel.STANDARD)

        change = analyzer.detect_risk_regime_change("OP-001", "cocoa")
        assert change is None

    def test_detect_insufficient_points(self):
        """Fewer than 2 points -> None."""
        analyzer = _make_analyzer()
        analyzer.add_data_point("OP-001", "cocoa", Decimal("45"), RiskLevel.STANDARD)

        change = analyzer.detect_risk_regime_change("OP-001", "cocoa")
        assert change is None


class TestTrendStats:
    """Test trend analyzer statistics."""

    def test_trend_stats(self):
        analyzer = _make_analyzer()
        analyzer.add_data_point("OP-001", "cocoa", Decimal("50"), RiskLevel.STANDARD)

        stats = analyzer.get_trend_stats()
        assert stats["total_data_points"] >= 1
        assert stats["tracked_pairs"] >= 1
        assert "total_analyses" in stats
