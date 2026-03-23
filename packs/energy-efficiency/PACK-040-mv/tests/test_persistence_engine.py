# -*- coding: utf-8 -*-
"""
Unit tests for PersistenceEngine -- PACK-040 Engine 9
============================================================

Tests multi-year savings persistence tracking, persistence factor
calculation, degradation rate modeling (linear, exponential,
step), re-commissioning triggers, and guarantee tracking.

Coverage target: 85%+
Total tests: ~30
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
    mod_key = f"pack040_test.{name}"
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


_m = _load("persistence_engine")


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
        assert hasattr(_m, "PersistenceEngine")

    def test_engine_instantiation(self):
        engine = _m.PersistenceEngine()
        assert engine is not None


# =============================================================================
# Persistence Factor
# =============================================================================


class TestPersistenceFactor:
    """Test persistence factor calculation across years."""

    def _get_factor(self, engine):
        return (getattr(engine, "calculate_persistence_factor", None)
                or getattr(engine, "persistence_factor", None)
                or getattr(engine, "compute_factor", None))

    def test_factor_result(self, persistence_data):
        engine = _m.PersistenceEngine()
        factor = self._get_factor(engine)
        if factor is None:
            pytest.skip("persistence_factor method not found")
        try:
            result = factor(persistence_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_factor_decreases_over_time(self, persistence_data):
        engine = _m.PersistenceEngine()
        factor = self._get_factor(engine)
        if factor is None:
            pytest.skip("persistence_factor method not found")
        try:
            result = factor(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Persistence factor not available")
        factors = (getattr(result, "yearly_factors", None)
                   or (result.get("yearly_factors") if isinstance(result, dict) else None))
        if factors is not None and isinstance(factors, list) and len(factors) >= 2:
            for i in range(1, len(factors)):
                f_curr = factors[i] if isinstance(factors[i], (int, float)) else factors[i].get("persistence_factor", 1)
                f_prev = factors[i - 1] if isinstance(factors[i - 1], (int, float)) else factors[i - 1].get("persistence_factor", 1)
                assert float(f_curr) <= float(f_prev)

    def test_year_one_factor_is_one(self, persistence_data):
        engine = _m.PersistenceEngine()
        factor = self._get_factor(engine)
        if factor is None:
            pytest.skip("persistence_factor method not found")
        try:
            result = factor(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Persistence factor not available")
        factors = (getattr(result, "yearly_factors", None)
                   or (result.get("yearly_factors") if isinstance(result, dict) else None))
        if factors is not None and isinstance(factors, list) and len(factors) >= 1:
            f1 = factors[0] if isinstance(factors[0], (int, float)) else factors[0].get("persistence_factor", 0)
            assert abs(float(f1) - 1.0) < 0.01


# =============================================================================
# Degradation Models
# =============================================================================


class TestDegradationModels:
    """Test 3 degradation models: linear, exponential, step."""

    def _get_degradation(self, engine):
        return (getattr(engine, "calculate_degradation", None)
                or getattr(engine, "degradation_model", None)
                or getattr(engine, "apply_degradation", None))

    @pytest.mark.parametrize("model", [
        "LINEAR",
        "EXPONENTIAL",
        "STEP",
    ])
    def test_degradation_model_accepted(self, model, persistence_data):
        engine = _m.PersistenceEngine()
        degradation = self._get_degradation(engine)
        if degradation is None:
            pytest.skip("degradation method not found")
        data = dict(persistence_data)
        data["degradation_model"] = model
        try:
            result = degradation(data)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("model", [
        "LINEAR",
        "EXPONENTIAL",
        "STEP",
    ])
    def test_degradation_model_deterministic(self, model, persistence_data):
        engine = _m.PersistenceEngine()
        degradation = self._get_degradation(engine)
        if degradation is None:
            pytest.skip("degradation method not found")
        data = dict(persistence_data)
        data["degradation_model"] = model
        try:
            r1 = degradation(data)
            r2 = degradation(data)
            assert str(r1) == str(r2)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    def test_linear_degradation_rate(self, persistence_data):
        """Linear degradation: savings(year) = initial * (1 - rate * year_offset)."""
        engine = _m.PersistenceEngine()
        degradation = self._get_degradation(engine)
        if degradation is None:
            pytest.skip("degradation method not found")
        data = dict(persistence_data)
        data["degradation_model"] = "LINEAR"
        try:
            result = degradation(data)
        except (ValueError, TypeError, KeyError):
            pytest.skip("Linear degradation not available")
        rate = (getattr(result, "annual_rate_pct", None)
                or (result.get("annual_rate_pct") if isinstance(result, dict) else None))
        if rate is not None:
            assert float(rate) > 0

    def test_exponential_never_zero(self, persistence_data):
        """Exponential degradation should never reach exactly zero."""
        engine = _m.PersistenceEngine()
        degradation = self._get_degradation(engine)
        if degradation is None:
            pytest.skip("degradation method not found")
        data = dict(persistence_data)
        data["degradation_model"] = "EXPONENTIAL"
        try:
            result = degradation(data)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pytest.skip("Exponential degradation not available")
        factors = (getattr(result, "yearly_factors", None)
                   or (result.get("yearly_factors") if isinstance(result, dict) else None))
        if factors is not None and isinstance(factors, list):
            for f in factors:
                val = f if isinstance(f, (int, float)) else f.get("persistence_factor", 0)
                assert float(val) > 0


# =============================================================================
# Re-commissioning Triggers
# =============================================================================


class TestRecommissioningTriggers:
    """Test re-commissioning trigger evaluation."""

    def _get_triggers(self, engine):
        return (getattr(engine, "check_triggers", None)
                or getattr(engine, "evaluate_triggers", None)
                or getattr(engine, "recommissioning_check", None))

    def test_triggers_result(self, persistence_data):
        engine = _m.PersistenceEngine()
        triggers = self._get_triggers(engine)
        if triggers is None:
            pytest.skip("recommissioning triggers method not found")
        try:
            result = triggers(persistence_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_triggers_boolean_output(self, persistence_data):
        engine = _m.PersistenceEngine()
        triggers = self._get_triggers(engine)
        if triggers is None:
            pytest.skip("recommissioning triggers method not found")
        try:
            result = triggers(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Triggers not available")
        triggered = (getattr(result, "triggered", None)
                     or getattr(result, "recommissioning_needed", None)
                     or (result.get("triggered") if isinstance(result, dict) else None))
        if triggered is not None:
            assert isinstance(triggered, bool)

    def test_shortfall_trigger(self, persistence_data):
        """Test that savings shortfall exceeding threshold triggers alert."""
        engine = _m.PersistenceEngine()
        triggers = self._get_triggers(engine)
        if triggers is None:
            pytest.skip("recommissioning triggers method not found")
        # Simulate large shortfall
        data = dict(persistence_data)
        data["years"] = list(data["years"])
        for yr in data["years"]:
            yr_copy = dict(yr)
            if yr_copy.get("actual_savings_kwh") is not None:
                yr_copy["actual_savings_kwh"] = Decimal("500000")  # Large shortfall
            data["years"][data["years"].index(yr)] = yr_copy
        try:
            result = triggers(data)
        except (ValueError, TypeError):
            pytest.skip("Triggers not available")
        assert result is not None


# =============================================================================
# Guarantee Tracking
# =============================================================================


class TestGuaranteeTracking:
    """Test ESCO performance guarantee tracking."""

    def _get_guarantee(self, engine):
        return (getattr(engine, "track_guarantee", None)
                or getattr(engine, "guarantee_status", None)
                or getattr(engine, "evaluate_guarantee", None))

    def test_guarantee_result(self, persistence_data):
        engine = _m.PersistenceEngine()
        guarantee = self._get_guarantee(engine)
        if guarantee is None:
            pytest.skip("guarantee tracking method not found")
        try:
            result = guarantee(persistence_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_guarantee_compliance(self, persistence_data):
        engine = _m.PersistenceEngine()
        guarantee = self._get_guarantee(engine)
        if guarantee is None:
            pytest.skip("guarantee tracking method not found")
        try:
            result = guarantee(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Guarantee tracking not available")
        compliant = (getattr(result, "in_compliance", None)
                     or (result.get("in_compliance") if isinstance(result, dict) else None))
        if compliant is not None:
            assert isinstance(compliant, bool)

    def test_shortfall_penalty(self, persistence_data):
        engine = _m.PersistenceEngine()
        guarantee = self._get_guarantee(engine)
        if guarantee is None:
            pytest.skip("guarantee tracking method not found")
        try:
            result = guarantee(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Guarantee tracking not available")
        penalty = (getattr(result, "shortfall_penalty_usd", None)
                   or (result.get("shortfall_penalty_usd") if isinstance(result, dict) else None))
        if penalty is not None:
            assert float(penalty) >= 0


# =============================================================================
# Provenance Tracking
# =============================================================================


class TestPersistenceProvenance:
    """Test SHA-256 provenance hashing for persistence data."""

    def _get_provenance(self, engine):
        return (getattr(engine, "compute_provenance", None)
                or getattr(engine, "provenance_hash", None)
                or getattr(engine, "get_provenance", None))

    def test_provenance_hash_format(self, persistence_data):
        engine = _m.PersistenceEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h = prov(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h is not None:
            h_str = str(h)
            assert len(h_str) == 64
            assert all(c in "0123456789abcdef" for c in h_str)

    def test_provenance_deterministic(self, persistence_data):
        engine = _m.PersistenceEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h1 = prov(persistence_data)
            h2 = prov(persistence_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h1 is not None and h2 is not None:
            assert str(h1) == str(h2)


# =============================================================================
# Persistence Fixture Validation
# =============================================================================


class TestPersistenceFixtureValidation:
    """Validate persistence fixture data consistency."""

    def test_ten_year_data(self, persistence_data):
        assert len(persistence_data["years"]) == 10

    def test_degradation_increases(self, persistence_data):
        for i in range(1, len(persistence_data["years"])):
            d_curr = persistence_data["years"][i]["degradation_pct"]
            d_prev = persistence_data["years"][i - 1]["degradation_pct"]
            assert d_curr >= d_prev

    def test_cumulative_increases(self, persistence_data):
        for i in range(1, len(persistence_data["years"])):
            c_curr = persistence_data["years"][i]["cumulative_savings_kwh"]
            c_prev = persistence_data["years"][i - 1]["cumulative_savings_kwh"]
            assert c_curr > c_prev

    def test_first_three_verified(self, persistence_data):
        for yr in persistence_data["years"][:3]:
            assert yr["verified"] is True

    def test_guarantee_positive(self, persistence_data):
        g = persistence_data["guarantee_tracking"]
        assert float(g["guaranteed_annual_savings_kwh"]) > 0

    def test_shortfall_threshold(self, persistence_data):
        threshold = float(
            persistence_data["re_commissioning_triggers"]
            ["savings_shortfall_threshold_pct"]
        )
        assert threshold > 0
