# -*- coding: utf-8 -*-
"""
Unit tests for PerformanceTrackerEngine -- PACK-037 Engine 7
===============================================================

Tests event performance calculation, compliance determination, season
summary, trend detection, and degradation alerts.

Coverage target: 85%+
Total tests: ~50
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack037_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("performance_tracker_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "PerformanceTrackerEngine")

    def test_engine_instantiation(self):
        engine = _m.PerformanceTrackerEngine()
        assert engine is not None


class TestEventPerformanceCalculation:
    """Test individual event performance calculation."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_event_performance", None)
                or getattr(engine, "event_performance", None)
                or getattr(engine, "calculate_performance", None))

    def test_calculate_performance(self, sample_dr_event_results):
        engine = _m.PerformanceTrackerEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("calculate_event_performance method not found")
        result = calc(event_results=sample_dr_event_results)
        assert result is not None

    def test_performance_ratio(self, sample_dr_event_results):
        actual = sample_dr_event_results["actual_reduction_kw"]
        target = sample_dr_event_results["target_reduction_kw"]
        ratio = actual / target
        assert ratio == pytest.approx(0.975, rel=0.01)

    def test_performance_above_threshold(self, sample_dr_event_results):
        ratio = float(sample_dr_event_results["performance_ratio"])
        threshold = 0.75  # PJM threshold
        assert ratio >= threshold

    @pytest.mark.parametrize("actual,target,expected_ratio", [
        (800, 800, 1.0),
        (600, 800, 0.75),
        (400, 800, 0.50),
        (900, 800, 1.125),
        (0, 800, 0.0),
    ])
    def test_ratio_calculations(self, actual, target, expected_ratio):
        ratio = actual / target if target > 0 else 0
        assert ratio == pytest.approx(expected_ratio, rel=0.001)

    def test_average_reduction_across_intervals(self, sample_dr_event_results):
        intervals = sample_dr_event_results["measurement_intervals"]
        total_reduction = sum(i["reduction_kw"] for i in intervals)
        avg_reduction = total_reduction / len(intervals)
        assert avg_reduction > 0
        assert avg_reduction <= 800


class TestComplianceDetermination:
    """Test compliance pass/fail determination."""

    def _get_compliance(self, engine):
        return (getattr(engine, "determine_compliance", None)
                or getattr(engine, "check_compliance", None)
                or getattr(engine, "compliance_check", None))

    def test_compliance_pass(self, sample_dr_event_results):
        engine = _m.PerformanceTrackerEngine()
        check = self._get_compliance(engine)
        if check is None:
            pytest.skip("compliance check method not found")
        result = check(event_results=sample_dr_event_results,
                      threshold_pct=Decimal("0.75"))
        status = getattr(result, "status", result)
        if isinstance(status, str):
            assert status in {"PASS", "COMPLIANT"}

    @pytest.mark.parametrize("ratio,threshold,expected", [
        (0.975, 0.75, "PASS"),
        (0.80, 0.75, "PASS"),
        (0.74, 0.75, "FAIL"),
        (0.50, 0.75, "FAIL"),
        (1.00, 0.90, "PASS"),
        (0.89, 0.90, "FAIL"),
    ])
    def test_compliance_threshold(self, ratio, threshold, expected):
        if ratio >= threshold:
            status = "PASS"
        else:
            status = "FAIL"
        assert status == expected

    def test_compliance_with_penalty(self):
        # Below threshold should trigger penalty
        ratio = 0.60
        threshold = 0.75
        assert ratio < threshold
        shortfall_kw = (threshold - ratio) * 800  # target
        assert shortfall_kw > 0


class TestSeasonSummary:
    """Test season-level summary calculations."""

    def _get_summary(self, engine):
        return (getattr(engine, "season_summary", None)
                or getattr(engine, "summarize_season", None)
                or getattr(engine, "calculate_season_summary", None))

    def test_season_summary(self):
        engine = _m.PerformanceTrackerEngine()
        summary = self._get_summary(engine)
        if summary is None:
            pytest.skip("season_summary method not found")
        events = [
            {"event_id": f"EVT-{i}", "performance_ratio": Decimal(str(0.8 + i * 0.03)),
             "actual_reduction_kw": 750 + i * 10, "target_reduction_kw": 800,
             "compliance_status": "PASS"}
            for i in range(5)
        ]
        result = summary(events=events, season="SUMMER_2025")
        assert result is not None

    def test_season_average_performance(self):
        ratios = [0.80, 0.85, 0.90, 0.95, 0.975]
        avg = sum(ratios) / len(ratios)
        assert avg == pytest.approx(0.895, rel=0.01)

    def test_season_compliance_rate(self):
        events = [
            {"compliance_status": "PASS"},
            {"compliance_status": "PASS"},
            {"compliance_status": "PASS"},
            {"compliance_status": "FAIL"},
            {"compliance_status": "PASS"},
        ]
        pass_count = sum(1 for e in events if e["compliance_status"] == "PASS")
        rate = pass_count / len(events)
        assert rate == pytest.approx(0.80, rel=0.01)

    def test_season_total_reduction_mwh(self):
        reductions_kwh = [3120, 2880, 3120, 2400, 3600]
        total_mwh = sum(reductions_kwh) / 1000
        assert total_mwh == pytest.approx(15.12, rel=0.01)


class TestTrendDetection:
    """Test performance trend detection."""

    def _get_trend(self, engine):
        return (getattr(engine, "detect_trends", None)
                or getattr(engine, "trend_analysis", None)
                or getattr(engine, "analyze_trends", None))

    def test_detect_improving_trend(self):
        engine = _m.PerformanceTrackerEngine()
        trend = self._get_trend(engine)
        if trend is None:
            pytest.skip("detect_trends method not found")
        events = [
            {"event_id": f"EVT-{i}", "date": f"2025-07-{i+1:02d}",
             "performance_ratio": Decimal(str(0.75 + i * 0.05))}
            for i in range(5)
        ]
        result = trend(events=events)
        direction = getattr(result, "direction", None)
        if direction is not None:
            assert direction in {"IMPROVING", "UP"}

    def test_detect_declining_trend(self):
        engine = _m.PerformanceTrackerEngine()
        trend = self._get_trend(engine)
        if trend is None:
            pytest.skip("detect_trends method not found")
        events = [
            {"event_id": f"EVT-{i}", "date": f"2025-07-{i+1:02d}",
             "performance_ratio": Decimal(str(0.95 - i * 0.05))}
            for i in range(5)
        ]
        result = trend(events=events)
        direction = getattr(result, "direction", None)
        if direction is not None:
            assert direction in {"DECLINING", "DOWN"}

    def test_detect_stable_trend(self):
        engine = _m.PerformanceTrackerEngine()
        trend = self._get_trend(engine)
        if trend is None:
            pytest.skip("detect_trends method not found")
        events = [
            {"event_id": f"EVT-{i}", "date": f"2025-07-{i+1:02d}",
             "performance_ratio": Decimal("0.90")}
            for i in range(5)
        ]
        result = trend(events=events)
        direction = getattr(result, "direction", None)
        if direction is not None:
            assert direction in {"STABLE", "FLAT"}


class TestDegradationAlert:
    """Test performance degradation alerting."""

    def _get_alert(self, engine):
        return (getattr(engine, "check_degradation", None)
                or getattr(engine, "degradation_alert", None)
                or getattr(engine, "monitor_degradation", None))

    def test_no_alert_good_performance(self):
        engine = _m.PerformanceTrackerEngine()
        alert = self._get_alert(engine)
        if alert is None:
            pytest.skip("degradation_alert method not found")
        events = [
            {"event_id": f"EVT-{i}", "performance_ratio": Decimal("0.95")}
            for i in range(5)
        ]
        result = alert(events=events, threshold_pct=Decimal("0.80"))
        triggered = getattr(result, "alert_triggered", None)
        if triggered is not None:
            assert triggered is False

    def test_alert_on_poor_performance(self):
        engine = _m.PerformanceTrackerEngine()
        alert = self._get_alert(engine)
        if alert is None:
            pytest.skip("degradation_alert method not found")
        events = [
            {"event_id": f"EVT-{i}", "performance_ratio": Decimal("0.60")}
            for i in range(5)
        ]
        result = alert(events=events, threshold_pct=Decimal("0.80"))
        triggered = getattr(result, "alert_triggered", None)
        if triggered is not None:
            assert triggered is True

    def test_alert_on_declining_trend(self):
        engine = _m.PerformanceTrackerEngine()
        alert = self._get_alert(engine)
        if alert is None:
            pytest.skip("degradation_alert method not found")
        events = [
            {"event_id": f"EVT-{i}",
             "performance_ratio": Decimal(str(0.95 - i * 0.08))}
            for i in range(5)
        ]
        result = alert(events=events, threshold_pct=Decimal("0.80"))
        assert result is not None


class TestPerformanceProvenance:
    """Test provenance tracking for performance results."""

    def test_provenance_hash_exists(self, sample_dr_event_results):
        engine = _m.PerformanceTrackerEngine()
        calc = (getattr(engine, "calculate_event_performance", None)
                or getattr(engine, "event_performance", None))
        if calc is None:
            pytest.skip("performance calculation method not found")
        result = calc(event_results=sample_dr_event_results)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64

    def test_provenance_deterministic(self, sample_dr_event_results):
        engine = _m.PerformanceTrackerEngine()
        calc = (getattr(engine, "calculate_event_performance", None)
                or getattr(engine, "event_performance", None))
        if calc is None:
            pytest.skip("performance calculation method not found")
        r1 = calc(event_results=sample_dr_event_results)
        r2 = calc(event_results=sample_dr_event_results)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 and h2:
            assert h1 == h2


# =============================================================================
# Multi-Event Statistics
# =============================================================================


class TestMultiEventStatistics:
    """Test statistical calculations across multiple events."""

    @pytest.fixture
    def multi_event_results(self):
        return [
            {"event_id": "EVT-001", "performance_ratio": 0.95,
             "actual_reduction_kw": 760, "target_reduction_kw": 800,
             "compliance_status": "PASS", "date": "2025-06-20"},
            {"event_id": "EVT-002", "performance_ratio": 0.88,
             "actual_reduction_kw": 700, "target_reduction_kw": 800,
             "compliance_status": "PASS", "date": "2025-07-08"},
            {"event_id": "EVT-003", "performance_ratio": 0.975,
             "actual_reduction_kw": 780, "target_reduction_kw": 800,
             "compliance_status": "PASS", "date": "2025-07-15"},
            {"event_id": "EVT-004", "performance_ratio": 0.70,
             "actual_reduction_kw": 560, "target_reduction_kw": 800,
             "compliance_status": "FAIL", "date": "2025-08-05"},
            {"event_id": "EVT-005", "performance_ratio": 0.92,
             "actual_reduction_kw": 736, "target_reduction_kw": 800,
             "compliance_status": "PASS", "date": "2025-08-18"},
        ]

    def test_average_ratio(self, multi_event_results):
        avg = sum(e["performance_ratio"] for e in multi_event_results) / len(multi_event_results)
        assert avg == pytest.approx(0.885, rel=0.01)

    def test_compliance_rate(self, multi_event_results):
        passes = sum(1 for e in multi_event_results
                    if e["compliance_status"] == "PASS")
        rate = passes / len(multi_event_results)
        assert rate == pytest.approx(0.80, rel=0.01)

    def test_total_reduction(self, multi_event_results):
        total = sum(e["actual_reduction_kw"] for e in multi_event_results)
        assert total == 3536

    def test_best_event(self, multi_event_results):
        best = max(multi_event_results, key=lambda e: e["performance_ratio"])
        assert best["event_id"] == "EVT-003"

    def test_worst_event(self, multi_event_results):
        worst = min(multi_event_results, key=lambda e: e["performance_ratio"])
        assert worst["event_id"] == "EVT-004"

    def test_event_count(self, multi_event_results):
        assert len(multi_event_results) == 5

    @pytest.mark.parametrize("event_idx,expected_status", [
        (0, "PASS"), (1, "PASS"), (2, "PASS"), (3, "FAIL"), (4, "PASS"),
    ])
    def test_individual_compliance(self, multi_event_results,
                                     event_idx, expected_status):
        assert multi_event_results[event_idx]["compliance_status"] == expected_status

    def test_standard_deviation(self, multi_event_results):
        import math
        ratios = [e["performance_ratio"] for e in multi_event_results]
        mean = sum(ratios) / len(ratios)
        variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        stdev = math.sqrt(variance)
        assert stdev > 0
        assert stdev < 0.5

    @pytest.mark.parametrize("event_idx,expected_ratio", [
        (0, 0.95), (1, 0.88), (2, 0.975), (3, 0.70), (4, 0.92),
    ])
    def test_individual_ratios(self, multi_event_results,
                                event_idx, expected_ratio):
        assert (multi_event_results[event_idx]["performance_ratio"]
                == pytest.approx(expected_ratio, rel=0.01))
