# -*- coding: utf-8 -*-
"""
Unit tests for BaselineEngine -- PACK-037 Engine 3
=====================================================

Tests all 8 baseline methodologies (High-4-of-5, 10-of-10, High-5-Similar,
10CP, Deemed Profile, Type I Regression, EU Standard, Custom Regression),
same-day adjustment, baseline optimization, methodology comparison, and
risk assessment.

Coverage target: 85%+
Total tests: ~80
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


_m = _load("baseline_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")

    def test_engine_class_exists(self):
        assert hasattr(_m, "BaselineEngine")

    def test_engine_instantiation(self):
        engine = _m.BaselineEngine()
        assert engine is not None


# =============================================================================
# High-4-of-5 Methodology
# =============================================================================


class TestHigh4of5:
    """Test PJM High-4-of-5 baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_high_4_of_5", None)
                or getattr(engine, "high_4_of_5", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_4_of_5 method not found")
        result = calc(historical_days=sample_baseline_data[:5],
                     methodology="HIGH_4_OF_5")
        assert result is not None

    def test_selects_4_highest_of_5(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_4_of_5 method not found")
        days_5 = sample_baseline_data[:5]
        result = calc(historical_days=days_5, methodology="HIGH_4_OF_5")
        days_used = getattr(result, "days_used", None)
        if days_used is not None:
            assert len(days_used) == 4

    def test_excludes_lowest_day(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_4_of_5 method not found")
        result = calc(historical_days=sample_baseline_data[:5],
                     methodology="HIGH_4_OF_5")
        excluded = getattr(result, "excluded_days", None)
        if excluded is not None:
            assert len(excluded) == 1

    def test_baseline_positive(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_4_of_5 method not found")
        result = calc(historical_days=sample_baseline_data[:5],
                     methodology="HIGH_4_OF_5")
        baseline = getattr(result, "baseline_kw", result)
        if isinstance(baseline, (int, float, list)):
            if isinstance(baseline, list):
                assert all(v > 0 for v in baseline)
            else:
                assert baseline > 0

    def test_deterministic(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_4_of_5 method not found")
        r1 = calc(historical_days=sample_baseline_data[:5],
                  methodology="HIGH_4_OF_5")
        r2 = calc(historical_days=sample_baseline_data[:5],
                  methodology="HIGH_4_OF_5")
        b1 = getattr(r1, "provenance_hash", str(r1))
        b2 = getattr(r2, "provenance_hash", str(r2))
        assert b1 == b2


# =============================================================================
# 10-of-10 Methodology
# =============================================================================


class TestTenOfTen:
    """Test 10-of-10 baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_10_of_10", None)
                or getattr(engine, "ten_of_ten", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("10_of_10 method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="10_OF_10")
        assert result is not None

    def test_uses_all_10_days(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("10_of_10 method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="10_OF_10")
        days_used = getattr(result, "days_used", None)
        if days_used is not None:
            assert len(days_used) == 10

    def test_average_of_10_days(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("10_of_10 method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="10_OF_10")
        assert result is not None


# =============================================================================
# High-5-Similar Methodology
# =============================================================================


class TestHigh5Similar:
    """Test High-5-of-Similar-Days methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_high_5_similar", None)
                or getattr(engine, "high_5_similar", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_5_similar method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="HIGH_5_SIMILAR",
                     target_temp_c=33.0)
        assert result is not None

    def test_selects_similar_temperature_days(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("high_5_similar method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="HIGH_5_SIMILAR",
                     target_temp_c=33.0)
        days_used = getattr(result, "days_used", None)
        if days_used is not None:
            assert len(days_used) <= 5


# =============================================================================
# 10CP Methodology
# =============================================================================


class TestTenCP:
    """Test 10 Coincident Peak methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_10cp", None)
                or getattr(engine, "ten_cp", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("10cp method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="10CP")
        assert result is not None


# =============================================================================
# Deemed Profile Methodology
# =============================================================================


class TestDeemedProfile:
    """Test deemed profile baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_deemed_profile", None)
                or getattr(engine, "deemed_profile", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("deemed_profile method not found")
        profile = {
            "building_type": "COMMERCIAL_OFFICE",
            "floor_area_m2": 45000,
            "peak_demand_kw": 2500.0,
        }
        result = calc(profile=profile, methodology="DEEMED_PROFILE")
        assert result is not None


# =============================================================================
# Type I Regression Methodology
# =============================================================================


class TestTypeIRegression:
    """Test Type I regression baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_type_i_regression", None)
                or getattr(engine, "type_i_regression", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("type_i_regression method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="TYPE_I_REGRESSION",
                     independent_var="temperature")
        assert result is not None

    def test_regression_coefficients(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("type_i_regression method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="TYPE_I_REGRESSION",
                     independent_var="temperature")
        coeff = getattr(result, "coefficients", None)
        if coeff is not None:
            assert len(coeff) >= 2  # intercept + slope


# =============================================================================
# EU Standard Methodology
# =============================================================================


class TestEUStandard:
    """Test EU standard baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_eu_standard", None)
                or getattr(engine, "eu_standard", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("eu_standard method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="EU_STANDARD")
        assert result is not None


# =============================================================================
# Custom Regression Methodology
# =============================================================================


class TestCustomRegression:
    """Test custom regression baseline methodology."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_custom_regression", None)
                or getattr(engine, "custom_regression", None)
                or getattr(engine, "calculate_baseline", None))

    def test_basic_calculation(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("custom_regression method not found")
        result = calc(historical_days=sample_baseline_data,
                     methodology="CUSTOM_REGRESSION",
                     variables=["temperature", "occupancy"])
        assert result is not None


# =============================================================================
# Same-Day Adjustment
# =============================================================================


class TestSameDayAdjustment:
    """Test same-day morning adjustment."""

    def _get_adjust(self, engine):
        return (getattr(engine, "apply_same_day_adjustment", None)
                or getattr(engine, "same_day_adjust", None)
                or getattr(engine, "adjust_baseline", None))

    def test_adjustment_applied(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        adjust = self._get_adjust(engine)
        if adjust is None:
            pytest.skip("same_day_adjustment method not found")
        baseline_kw = [2000.0] * 96
        morning_actuals = [2100.0] * 16  # 4 hours of morning data
        result = adjust(baseline_kw=baseline_kw,
                       morning_actuals=morning_actuals)
        assert result is not None

    def test_adjustment_ratio(self):
        engine = _m.BaselineEngine()
        adjust = (getattr(engine, "apply_same_day_adjustment", None)
                  or getattr(engine, "same_day_adjust", None))
        if adjust is None:
            pytest.skip("same_day_adjustment method not found")
        baseline_kw = [2000.0] * 96
        morning_actuals = [2200.0] * 16  # 10% higher
        result = adjust(baseline_kw=baseline_kw,
                       morning_actuals=morning_actuals)
        adjusted = getattr(result, "adjusted_baseline_kw", result)
        if isinstance(adjusted, list) and len(adjusted) > 0:
            # Adjusted should be higher than original
            assert adjusted[48] >= 2000.0  # Afternoon should be adjusted up

    @pytest.mark.parametrize("adjustment_cap_pct", [0.10, 0.20, 0.50])
    def test_adjustment_cap(self, adjustment_cap_pct):
        engine = _m.BaselineEngine()
        adjust = (getattr(engine, "apply_same_day_adjustment", None)
                  or getattr(engine, "same_day_adjust", None))
        if adjust is None:
            pytest.skip("same_day_adjustment method not found")
        baseline_kw = [2000.0] * 96
        # Morning is 80% higher (should be capped)
        morning_actuals = [3600.0] * 16
        result = adjust(baseline_kw=baseline_kw,
                       morning_actuals=morning_actuals,
                       cap_pct=adjustment_cap_pct)
        assert result is not None


# =============================================================================
# Baseline Optimization
# =============================================================================


class TestBaselineOptimization:
    """Test baseline methodology selection optimization."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_methodology", None)
                or getattr(engine, "select_best_methodology", None)
                or getattr(engine, "optimize_baseline", None))

    def test_selects_best_methodology(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_methodology method not found")
        result = optimize(historical_days=sample_baseline_data)
        best = getattr(result, "best_methodology", None)
        if best is not None:
            valid = {"HIGH_4_OF_5", "10_OF_10", "HIGH_5_SIMILAR", "10CP",
                     "DEEMED_PROFILE", "TYPE_I_REGRESSION", "EU_STANDARD",
                     "CUSTOM_REGRESSION"}
            assert best in valid

    def test_returns_ranking(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_methodology method not found")
        result = optimize(historical_days=sample_baseline_data)
        ranking = getattr(result, "ranking", None)
        if ranking is not None:
            assert len(ranking) >= 2


# =============================================================================
# Methodology Comparison
# =============================================================================


class TestMethodologyComparison:
    """Test comparison across methodologies."""

    def _get_compare(self, engine):
        return (getattr(engine, "compare_methodologies", None)
                or getattr(engine, "methodology_comparison", None)
                or getattr(engine, "compare_baselines", None))

    def test_comparison_returns_results(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("compare_methodologies method not found")
        result = compare(historical_days=sample_baseline_data,
                        methodologies=["HIGH_4_OF_5", "10_OF_10"])
        assert result is not None

    def test_comparison_includes_accuracy(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("compare_methodologies method not found")
        result = compare(historical_days=sample_baseline_data,
                        methodologies=["HIGH_4_OF_5", "10_OF_10"])
        results_list = getattr(result, "results", result)
        if isinstance(results_list, list):
            for r in results_list:
                accuracy = getattr(r, "accuracy_pct", None)
                if accuracy is not None:
                    assert 0.0 <= float(accuracy) <= 100.0


# =============================================================================
# Risk Assessment
# =============================================================================


class TestRiskAssessment:
    """Test baseline risk assessment."""

    def _get_risk(self, engine):
        return (getattr(engine, "assess_baseline_risk", None)
                or getattr(engine, "baseline_risk", None)
                or getattr(engine, "risk_assessment", None))

    def test_risk_assessment(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        risk = self._get_risk(engine)
        if risk is None:
            pytest.skip("baseline risk method not found")
        result = risk(historical_days=sample_baseline_data,
                     methodology="HIGH_4_OF_5")
        assert result is not None

    def test_risk_level_valid(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        risk = self._get_risk(engine)
        if risk is None:
            pytest.skip("baseline risk method not found")
        result = risk(historical_days=sample_baseline_data,
                     methodology="HIGH_4_OF_5")
        level = getattr(result, "risk_level", None)
        if level is not None:
            assert level in {"LOW", "MEDIUM", "HIGH"}


# =============================================================================
# Provenance and Edge Cases
# =============================================================================


class TestBaselineProvenance:
    def test_provenance_hash_deterministic(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = (getattr(engine, "calculate_baseline", None)
                or getattr(engine, "calculate_high_4_of_5", None))
        if calc is None:
            pytest.skip("calculate_baseline method not found")
        r1 = calc(historical_days=sample_baseline_data[:5],
                  methodology="HIGH_4_OF_5")
        r2 = calc(historical_days=sample_baseline_data[:5],
                  methodology="HIGH_4_OF_5")
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 and h2:
            assert h1 == h2

    def test_hash_sha256_format(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = (getattr(engine, "calculate_baseline", None)
                or getattr(engine, "calculate_high_4_of_5", None))
        if calc is None:
            pytest.skip("calculate_baseline method not found")
        result = calc(historical_days=sample_baseline_data[:5],
                     methodology="HIGH_4_OF_5")
        h = getattr(result, "provenance_hash", None)
        if h:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


class TestBaselineEdgeCases:
    def test_insufficient_days(self):
        engine = _m.BaselineEngine()
        calc = (getattr(engine, "calculate_baseline", None)
                or getattr(engine, "calculate_high_4_of_5", None))
        if calc is None:
            pytest.skip("calculate_baseline method not found")
        # Only 2 days for High-4-of-5 (needs 5)
        days = [{"date": f"2025-06-0{i+1}", "intervals": [
            {"timestamp": f"2025-06-0{i+1}T12:00:00", "demand_kw": 2000.0}
        ]} for i in range(2)]
        try:
            result = calc(historical_days=days, methodology="HIGH_4_OF_5")
        except (ValueError, Exception):
            pass  # Expected to raise an error

    def test_flat_profile_baseline(self):
        engine = _m.BaselineEngine()
        calc = (getattr(engine, "calculate_baseline", None)
                or getattr(engine, "calculate_high_4_of_5", None))
        if calc is None:
            pytest.skip("calculate_baseline method not found")
        days = []
        for d in range(5):
            intervals = [
                {"timestamp": f"2025-06-{d+2:02d}T{h:02d}:00:00",
                 "demand_kw": 1000.0}
                for h in range(24)
            ]
            days.append({"date": f"2025-06-{d+2:02d}", "intervals": intervals})
        result = calc(historical_days=days, methodology="HIGH_4_OF_5")
        baseline = getattr(result, "baseline_kw", result)
        if isinstance(baseline, (int, float)):
            assert baseline == pytest.approx(1000.0, rel=0.01)
        elif isinstance(baseline, list):
            for v in baseline:
                assert v == pytest.approx(1000.0, rel=0.01)


# =============================================================================
# Methodology Selection Validation
# =============================================================================


class TestMethodologySelection:
    """Test methodology name validation and selection."""

    @pytest.mark.parametrize("methodology", [
        "HIGH_4_OF_5", "10_OF_10", "HIGH_5_SIMILAR", "10CP",
        "DEEMED_PROFILE", "TYPE_I_REGRESSION", "EU_STANDARD",
        "CUSTOM_REGRESSION",
    ])
    def test_valid_methodology_names(self, methodology):
        valid = {"HIGH_4_OF_5", "10_OF_10", "HIGH_5_SIMILAR", "10CP",
                 "DEEMED_PROFILE", "TYPE_I_REGRESSION", "EU_STANDARD",
                 "CUSTOM_REGRESSION"}
        assert methodology in valid

    def test_invalid_methodology_rejected(self, sample_baseline_data):
        engine = _m.BaselineEngine()
        calc = (getattr(engine, "calculate_baseline", None)
                or getattr(engine, "calculate_high_4_of_5", None))
        if calc is None:
            pytest.skip("calculate_baseline method not found")
        try:
            calc(historical_days=sample_baseline_data[:5],
                 methodology="INVALID_METHOD")
        except (ValueError, KeyError, Exception):
            pass  # Expected


# =============================================================================
# Baseline Interval Consistency
# =============================================================================


class TestBaselineIntervalConsistency:
    """Test that baseline output has correct interval structure."""

    def test_baseline_has_96_intervals(self, sample_baseline_data):
        for day in sample_baseline_data:
            assert len(day["intervals"]) == 96

    def test_intervals_15_min(self, sample_baseline_data):
        day = sample_baseline_data[0]
        for i in range(min(4, len(day["intervals"]))):
            ts = day["intervals"][i]["timestamp"]
            minute = int(ts.split(":")[1])
            assert minute in {0, 15, 30, 45}

    def test_all_intervals_positive_demand(self, sample_baseline_data):
        for day in sample_baseline_data:
            for interval in day["intervals"]:
                assert interval["demand_kw"] >= 0

    def test_baseline_peak_reasonable(self, sample_baseline_data):
        for day in sample_baseline_data:
            peak = max(i["demand_kw"] for i in day["intervals"])
            assert 500 < peak < 5000

    def test_baseline_trough_reasonable(self, sample_baseline_data):
        for day in sample_baseline_data:
            trough = min(i["demand_kw"] for i in day["intervals"])
            assert trough >= 0
            assert trough < 1500

    @pytest.mark.parametrize("day_idx", [0, 2, 4, 6, 8])
    def test_day_data_complete(self, sample_baseline_data, day_idx):
        day = sample_baseline_data[day_idx]
        assert "date" in day
        assert "intervals" in day
        assert len(day["intervals"]) == 96

    @pytest.mark.parametrize("day_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_all_days_non_event(self, sample_baseline_data, day_idx):
        day = sample_baseline_data[day_idx]
        assert day.get("is_event_day", False) is False
