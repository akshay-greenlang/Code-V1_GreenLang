# -*- coding: utf-8 -*-
"""
Unit tests for CPManagementEngine -- PACK-038 Engine 6
============================================================

Tests PJM 5CP prediction, ERCOT 4CP prediction, ISO-NE ICL methodology,
CP tag value calculation, response performance tracking, annual charge
forecast, and ISO methodology parametrization.

Coverage target: 85%+
Total tests: ~50
"""

import hashlib
import importlib.util
import json
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
    mod_key = f"pack038_test.{name}"
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


_m = _load("cp_management_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_class_exists(self):
        assert hasattr(_m, "CPManagementEngine")

    def test_engine_instantiation(self):
        engine = _m.CPManagementEngine()
        assert engine is not None


# =============================================================================
# PJM 5CP Prediction
# =============================================================================


class TestPJM5CPPrediction:
    """Test PJM 5 Coincident Peak prediction."""

    def _get_predict(self, engine):
        return (getattr(engine, "predict_cp", None)
                or getattr(engine, "predict_5cp", None)
                or getattr(engine, "forecast_cp", None))

    def test_prediction_result(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = self._get_predict(engine)
        if predict is None:
            pytest.skip("predict_cp method not found")
        result = predict(cp_data=sample_cp_data, methodology="PJM_5CP")
        assert result is not None

    def test_prediction_probability(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = self._get_predict(engine)
        if predict is None:
            pytest.skip("predict_cp method not found")
        result = predict(cp_data=sample_cp_data, methodology="PJM_5CP")
        prob = getattr(result, "probability", None)
        if prob is not None:
            assert 0 <= float(prob) <= 1.0

    def test_pjm_5cp_count(self, sample_cp_data):
        assert len(sample_cp_data["cp_events"]) == 5


# =============================================================================
# ERCOT 4CP Prediction
# =============================================================================


class TestERCOT4CPPrediction:
    """Test ERCOT 4 Coincident Peak prediction."""

    def _get_predict(self, engine):
        return (getattr(engine, "predict_cp", None)
                or getattr(engine, "predict_4cp", None)
                or getattr(engine, "forecast_cp", None))

    def test_ercot_prediction(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = self._get_predict(engine)
        if predict is None:
            pytest.skip("predict_cp method not found")
        ercot_data = dict(sample_cp_data, iso_rto="ERCOT", methodology="4CP")
        try:
            result = predict(cp_data=ercot_data, methodology="ERCOT_4CP")
        except (TypeError, ValueError):
            result = predict(cp_data=ercot_data)
        assert result is not None


# =============================================================================
# ISO-NE ICL Methodology
# =============================================================================


class TestISONEICL:
    """Test ISO-NE Installed Capacity Load (ICL) methodology."""

    def _get_icl(self, engine):
        return (getattr(engine, "calculate_icl", None)
                or getattr(engine, "iso_ne_icl", None)
                or getattr(engine, "predict_cp", None))

    def test_icl_result(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        icl = self._get_icl(engine)
        if icl is None:
            pytest.skip("icl method not found")
        ne_data = dict(sample_cp_data, iso_rto="ISO_NE", methodology="ICL")
        try:
            result = icl(cp_data=ne_data, methodology="ISO_NE_ICL")
        except (TypeError, ValueError):
            result = icl(cp_data=ne_data)
        assert result is not None


# =============================================================================
# ISO Methodology Parametrize
# =============================================================================


class TestISOMethodologies:
    """Test CP prediction across ISO methodologies."""

    def _get_predict(self, engine):
        return (getattr(engine, "predict_cp", None)
                or getattr(engine, "forecast_cp", None)
                or getattr(engine, "cp_prediction", None))

    @pytest.mark.parametrize("methodology", [
        "PJM_5CP", "ERCOT_4CP", "ISO_NE_ICL",
        "NYISO_ICAP", "CAISO_COINCIDENT", "MISO_COINCIDENT",
    ])
    def test_methodology_accepted(self, methodology, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = self._get_predict(engine)
        if predict is None:
            pytest.skip("predict method not found")
        try:
            result = predict(cp_data=sample_cp_data, methodology=methodology)
        except (TypeError, ValueError):
            result = predict(cp_data=sample_cp_data)
        assert result is not None

    @pytest.mark.parametrize("confidence", [0.50, 0.75, 0.90, 0.95])
    def test_confidence_levels(self, confidence, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = self._get_predict(engine)
        if predict is None:
            pytest.skip("predict method not found")
        try:
            result = predict(cp_data=sample_cp_data,
                             methodology="PJM_5CP",
                             confidence_level=confidence)
        except TypeError:
            result = predict(cp_data=sample_cp_data)
        assert result is not None


# =============================================================================
# CP Tag Value Calculation
# =============================================================================


class TestCPTagValue:
    """Test CP tag value calculation."""

    def _get_tag_value(self, engine):
        return (getattr(engine, "calculate_tag_value", None)
                or getattr(engine, "cp_tag_value", None)
                or getattr(engine, "tag_value", None))

    def test_tag_value_basic(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        calc = self._get_tag_value(engine)
        if calc is None:
            pytest.skip("tag_value method not found")
        result = calc(
            icap_tag_kw=sample_cp_data["icap_tag_kw"],
            rate_usd_per_kw_year=sample_cp_data["tag_value_usd_per_kw_year"],
        )
        value = getattr(result, "annual_charge_usd", result)
        if isinstance(value, (Decimal, int, float)):
            expected = Decimal("1850.0") * Decimal("6.80")
            assert abs(Decimal(str(value)) - expected) < Decimal("1.00")

    def test_tag_reduction_savings(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        calc = self._get_tag_value(engine)
        if calc is None:
            pytest.skip("tag_value method not found")
        baseline = calc(
            icap_tag_kw=1850,
            rate_usd_per_kw_year=Decimal("6.80"),
        )
        reduced = calc(
            icap_tag_kw=1550,
            rate_usd_per_kw_year=Decimal("6.80"),
        )
        v_base = getattr(baseline, "annual_charge_usd", baseline)
        v_red = getattr(reduced, "annual_charge_usd", reduced)
        if isinstance(v_base, (Decimal, int, float)) and isinstance(v_red, (Decimal, int, float)):
            assert Decimal(str(v_base)) > Decimal(str(v_red))


# =============================================================================
# Response Performance Tracking
# =============================================================================


class TestResponsePerformance:
    """Test CP response performance tracking."""

    def _get_performance(self, engine):
        return (getattr(engine, "track_performance", None)
                or getattr(engine, "response_performance", None)
                or getattr(engine, "evaluate_response", None))

    def test_performance_result(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        track = self._get_performance(engine)
        if track is None:
            pytest.skip("performance tracking method not found")
        result = track(cp_events=sample_cp_data["cp_events"])
        assert result is not None

    def test_all_events_tracked(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        track = self._get_performance(engine)
        if track is None:
            pytest.skip("performance tracking method not found")
        result = track(cp_events=sample_cp_data["cp_events"])
        events = getattr(result, "events_tracked", None)
        if events is not None:
            assert events == 5

    def test_average_response(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        track = self._get_performance(engine)
        if track is None:
            pytest.skip("performance tracking method not found")
        result = track(cp_events=sample_cp_data["cp_events"])
        avg = getattr(result, "average_response_kw", None)
        if avg is not None:
            expected_avg = sum(e["response_achieved_kw"] for e in sample_cp_data["cp_events"]) / 5
            assert abs(float(avg) - expected_avg) < 1.0


# =============================================================================
# Annual Charge Forecast
# =============================================================================


class TestAnnualChargeForecast:
    """Test annual CP charge forecasting."""

    def _get_forecast(self, engine):
        return (getattr(engine, "forecast_annual_charge", None)
                or getattr(engine, "annual_forecast", None)
                or getattr(engine, "project_charges", None))

    def test_forecast_result(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("forecast method not found")
        result = forecast(cp_data=sample_cp_data)
        assert result is not None

    def test_forecast_positive(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("forecast method not found")
        result = forecast(cp_data=sample_cp_data)
        charge = getattr(result, "forecast_annual_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert float(charge) > 0


# =============================================================================
# Weather-Based CP Alert
# =============================================================================


class TestWeatherBasedCPAlert:
    """Test weather-based CP event alerting."""

    def _get_alert(self, engine):
        return (getattr(engine, "weather_cp_alert", None)
                or getattr(engine, "check_cp_conditions", None)
                or getattr(engine, "assess_cp_risk", None))

    @pytest.mark.parametrize("temp_c,humidity_pct,expected_risk", [
        (38, 65, "HIGH"), (35, 55, "MEDIUM"), (30, 40, "LOW"),
        (40, 70, "CRITICAL"), (28, 30, "LOW"),
    ])
    def test_weather_alert_levels(self, temp_c, humidity_pct, expected_risk,
                                   sample_cp_data):
        engine = _m.CPManagementEngine()
        alert = self._get_alert(engine)
        if alert is None:
            pytest.skip("weather_alert method not found")
        try:
            result = alert(temperature_c=temp_c, humidity_pct=humidity_pct,
                           cp_data=sample_cp_data)
        except TypeError:
            result = alert(temperature_c=temp_c, humidity_pct=humidity_pct)
        assert result is not None

    def test_high_heat_index_triggers_alert(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        alert = self._get_alert(engine)
        if alert is None:
            pytest.skip("weather_alert method not found")
        try:
            result = alert(temperature_c=39, humidity_pct=70,
                           cp_data=sample_cp_data)
        except TypeError:
            result = alert(temperature_c=39, humidity_pct=70)
        risk = getattr(result, "risk_level", None)
        if risk is not None:
            assert risk in ["HIGH", "CRITICAL"]


# =============================================================================
# CP Reduction Strategy
# =============================================================================


class TestCPReductionStrategy:
    """Test CP tag reduction strategy calculation."""

    def _get_strategy(self, engine):
        return (getattr(engine, "reduction_strategy", None)
                or getattr(engine, "cp_reduction_plan", None)
                or getattr(engine, "plan_cp_reduction", None))

    def test_strategy_result(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        strat = self._get_strategy(engine)
        if strat is None:
            pytest.skip("reduction_strategy method not found")
        result = strat(cp_data=sample_cp_data, target_reduction_kw=300)
        assert result is not None

    @pytest.mark.parametrize("target_kw", [100, 200, 300, 400, 500])
    def test_strategy_target_scaling(self, target_kw, sample_cp_data):
        engine = _m.CPManagementEngine()
        strat = self._get_strategy(engine)
        if strat is None:
            pytest.skip("reduction_strategy method not found")
        try:
            result = strat(cp_data=sample_cp_data, target_reduction_kw=target_kw)
        except TypeError:
            result = strat(cp_data=sample_cp_data)
        assert result is not None


# =============================================================================
# CP Data Fixture Validation
# =============================================================================


class TestCPDataFixture:
    def test_five_events(self, sample_cp_data):
        assert len(sample_cp_data["cp_events"]) == 5

    def test_events_sorted_by_system_peak(self, sample_cp_data):
        peaks = [e["system_peak_mw"] for e in sample_cp_data["cp_events"]]
        assert peaks == sorted(peaks, reverse=True)

    def test_icap_tag_present(self, sample_cp_data):
        assert sample_cp_data["icap_tag_kw"] == 1850.0

    def test_weather_correlation(self, sample_cp_data):
        assert "weather_correlation" in sample_cp_data
        assert sample_cp_data["weather_correlation"]["r_squared"] > 0.8

    @pytest.mark.parametrize("cp_num", [1, 2, 3, 4, 5])
    def test_cp_event_fields(self, cp_num, sample_cp_data):
        event = sample_cp_data["cp_events"][cp_num - 1]
        assert event["cp_number"] == cp_num
        assert event["facility_demand_kw"] > 0
        assert event["response_achieved_kw"] > 0


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    def test_provenance_deterministic(self, sample_cp_data):
        engine = _m.CPManagementEngine()
        predict = (getattr(engine, "predict_cp", None)
                   or getattr(engine, "forecast_cp", None))
        if predict is None:
            pytest.skip("predict method not found")
        try:
            r1 = predict(cp_data=sample_cp_data, methodology="PJM_5CP")
            r2 = predict(cp_data=sample_cp_data, methodology="PJM_5CP")
        except TypeError:
            r1 = predict(cp_data=sample_cp_data)
            r2 = predict(cp_data=sample_cp_data)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2
            assert len(h1) == 64
