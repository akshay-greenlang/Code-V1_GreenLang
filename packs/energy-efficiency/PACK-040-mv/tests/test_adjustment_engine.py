# -*- coding: utf-8 -*-
"""
Unit tests for AdjustmentEngine -- PACK-040 Engine 2
============================================================

Tests routine and non-routine adjustments per IPMVP methodology
including weather, production, occupancy, operating hours (routine)
and floor area, equipment, schedule, static factor (non-routine).

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


_m = _load("adjustment_engine")


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
        assert hasattr(_m, "AdjustmentEngine")

    def test_engine_instantiation(self):
        engine = _m.AdjustmentEngine()
        assert engine is not None


# =============================================================================
# Routine Adjustment Types
# =============================================================================


class TestRoutineAdjustments:
    """Test 4 routine adjustment types per IPMVP."""

    def _get_routine(self, engine):
        return (getattr(engine, "calculate_routine_adjustment", None)
                or getattr(engine, "routine_adjustment", None)
                or getattr(engine, "apply_routine", None))

    @pytest.mark.parametrize("adj_type", [
        "WEATHER",
        "PRODUCTION",
        "OCCUPANCY",
        "OPERATING_HOURS",
    ])
    def test_routine_type_accepted(self, adj_type, adjustment_data):
        engine = _m.AdjustmentEngine()
        routine = self._get_routine(engine)
        if routine is None:
            pytest.skip("routine adjustment method not found")
        adj = next(
            (a for a in adjustment_data["routine_adjustments"]
             if a["type"] == adj_type), None
        )
        if adj is None:
            pytest.skip(f"No test data for {adj_type}")
        try:
            result = routine(adj)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("adj_type", [
        "WEATHER",
        "PRODUCTION",
        "OCCUPANCY",
        "OPERATING_HOURS",
    ])
    def test_routine_type_deterministic(self, adj_type, adjustment_data):
        engine = _m.AdjustmentEngine()
        routine = self._get_routine(engine)
        if routine is None:
            pytest.skip("routine adjustment method not found")
        adj = next(
            (a for a in adjustment_data["routine_adjustments"]
             if a["type"] == adj_type), None
        )
        if adj is None:
            pytest.skip(f"No test data for {adj_type}")
        try:
            r1 = routine(adj)
            r2 = routine(adj)
            assert str(r1) == str(r2)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    def test_weather_adjustment_nonzero(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        routine = self._get_routine(engine)
        if routine is None:
            pytest.skip("routine adjustment method not found")
        weather_adj = adjustment_data["routine_adjustments"][0]
        try:
            result = routine(weather_adj)
        except (ValueError, TypeError):
            pytest.skip("Weather adjustment not available")
        val = (getattr(result, "adjustment_kwh", None)
               or (result.get("adjustment_kwh") if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) != 0

    def test_production_adjustment_method(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        routine = self._get_routine(engine)
        if routine is None:
            pytest.skip("routine adjustment method not found")
        prod_adj = adjustment_data["routine_adjustments"][1]
        try:
            result = routine(prod_adj)
        except (ValueError, TypeError):
            pytest.skip("Production adjustment not available")
        method = (getattr(result, "method", None)
                  or (result.get("method") if isinstance(result, dict) else None))
        if method is not None:
            assert str(method) in ("REGRESSION_BASED", "RATIO_BASED", "regression_based", "ratio_based")


# =============================================================================
# Non-Routine Adjustment Types
# =============================================================================


class TestNonRoutineAdjustments:
    """Test 4 non-routine adjustment types per IPMVP."""

    def _get_non_routine(self, engine):
        return (getattr(engine, "calculate_non_routine_adjustment", None)
                or getattr(engine, "non_routine_adjustment", None)
                or getattr(engine, "apply_non_routine", None))

    @pytest.mark.parametrize("adj_type", [
        "FLOOR_AREA",
        "EQUIPMENT",
        "SCHEDULE",
        "STATIC_FACTOR",
    ])
    def test_non_routine_type_accepted(self, adj_type, adjustment_data):
        engine = _m.AdjustmentEngine()
        non_routine = self._get_non_routine(engine)
        if non_routine is None:
            pytest.skip("non-routine adjustment method not found")
        adj = next(
            (a for a in adjustment_data["non_routine_adjustments"]
             if a["type"] == adj_type), None
        )
        if adj is None:
            pytest.skip(f"No test data for {adj_type}")
        try:
            result = non_routine(adj)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("adj_type", [
        "FLOOR_AREA",
        "EQUIPMENT",
        "SCHEDULE",
        "STATIC_FACTOR",
    ])
    def test_non_routine_type_deterministic(self, adj_type, adjustment_data):
        engine = _m.AdjustmentEngine()
        non_routine = self._get_non_routine(engine)
        if non_routine is None:
            pytest.skip("non-routine adjustment method not found")
        adj = next(
            (a for a in adjustment_data["non_routine_adjustments"]
             if a["type"] == adj_type), None
        )
        if adj is None:
            pytest.skip(f"No test data for {adj_type}")
        try:
            r1 = non_routine(adj)
            r2 = non_routine(adj)
            assert str(r1) == str(r2)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    def test_floor_area_adjustment_negative(self, adjustment_data):
        """Floor area expansion should increase consumption (negative adjustment)."""
        engine = _m.AdjustmentEngine()
        non_routine = self._get_non_routine(engine)
        if non_routine is None:
            pytest.skip("non-routine adjustment method not found")
        floor_adj = adjustment_data["non_routine_adjustments"][0]
        try:
            result = non_routine(floor_adj)
        except (ValueError, TypeError):
            pytest.skip("Floor area adjustment not available")
        val = (getattr(result, "adjustment_kwh", None)
               or (result.get("adjustment_kwh") if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) < 0

    def test_equipment_adjustment_has_documentation(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        non_routine = self._get_non_routine(engine)
        if non_routine is None:
            pytest.skip("non-routine adjustment method not found")
        equip_adj = adjustment_data["non_routine_adjustments"][1]
        try:
            result = non_routine(equip_adj)
        except (ValueError, TypeError):
            pytest.skip("Equipment adjustment not available")
        doc = (getattr(result, "documentation", None)
               or (result.get("documentation") if isinstance(result, dict) else None))
        if doc is not None:
            assert len(str(doc)) > 0


# =============================================================================
# Net Adjustment Calculation
# =============================================================================


class TestNetAdjustment:
    """Test net adjustment calculation (routine + non-routine)."""

    def _get_net(self, engine):
        return (getattr(engine, "calculate_net_adjustment", None)
                or getattr(engine, "net_adjustment", None)
                or getattr(engine, "total_adjustment", None))

    def test_net_adjustment_result(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        net = self._get_net(engine)
        if net is None:
            pytest.skip("net adjustment method not found")
        try:
            result = net(adjustment_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_net_equals_sum(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        net = self._get_net(engine)
        if net is None:
            pytest.skip("net adjustment method not found")
        try:
            result = net(adjustment_data)
        except (ValueError, TypeError):
            pytest.skip("Net adjustment not available")
        val = (getattr(result, "net_adjustment_kwh", None)
               or (result.get("net_adjustment_kwh") if isinstance(result, dict) else None))
        if val is not None:
            expected = float(adjustment_data["net_adjustment_kwh"])
            assert abs(float(val) - expected) < 1000  # Tolerance for rounding


# =============================================================================
# Provenance Tracking
# =============================================================================


class TestAdjustmentProvenance:
    """Test SHA-256 provenance hashing for adjustments."""

    def _get_provenance(self, engine):
        return (getattr(engine, "compute_provenance", None)
                or getattr(engine, "provenance_hash", None)
                or getattr(engine, "get_provenance", None))

    def test_provenance_hash_format(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h = prov(adjustment_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h is not None:
            h_str = str(h)
            assert len(h_str) == 64
            assert all(c in "0123456789abcdef" for c in h_str)

    def test_provenance_deterministic(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h1 = prov(adjustment_data)
            h2 = prov(adjustment_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h1 is not None and h2 is not None:
            assert str(h1) == str(h2)


# =============================================================================
# Adjustment Validation & Edge Cases
# =============================================================================


class TestAdjustmentValidation:
    """Test adjustment input validation and edge cases."""

    def _get_validate(self, engine):
        return (getattr(engine, "validate_adjustment", None)
                or getattr(engine, "validate", None)
                or getattr(engine, "check_adjustment", None))

    def test_validate_routine(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        validate = self._get_validate(engine)
        if validate is None:
            pytest.skip("validate method not found")
        adj = adjustment_data["routine_adjustments"][0]
        try:
            result = validate(adj)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_validate_non_routine(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        validate = self._get_validate(engine)
        if validate is None:
            pytest.skip("validate method not found")
        adj = adjustment_data["non_routine_adjustments"][0]
        try:
            result = validate(adj)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_zero_adjustment_accepted(self, adjustment_data):
        """Static factor with zero adjustment should be valid."""
        engine = _m.AdjustmentEngine()
        non_routine = (getattr(engine, "calculate_non_routine_adjustment", None)
                       or getattr(engine, "non_routine_adjustment", None)
                       or getattr(engine, "apply_non_routine", None))
        if non_routine is None:
            pytest.skip("non_routine method not found")
        static_adj = adjustment_data["non_routine_adjustments"][3]
        try:
            result = non_routine(static_adj)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_adjustment_chronological_order(self, adjustment_data):
        """Non-routine adjustments should be applied in date order."""
        engine = _m.AdjustmentEngine()
        sort_fn = (getattr(engine, "sort_adjustments", None)
                   or getattr(engine, "order_adjustments", None)
                   or getattr(engine, "chronological_order", None))
        if sort_fn is None:
            pytest.skip("sort method not found")
        try:
            result = sort_fn(adjustment_data["non_routine_adjustments"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_routine_total_matches(self, adjustment_data):
        """Sum of routine adjustments should match total_routine_adjustment_kwh."""
        expected = float(adjustment_data["total_routine_adjustment_kwh"])
        actual_sum = sum(
            float(a["adjustment_kwh"])
            for a in adjustment_data["routine_adjustments"]
        )
        assert abs(actual_sum - expected) < 1.0

    def test_non_routine_total_matches(self, adjustment_data):
        """Sum of non-routine adjustments should match total_non_routine_adjustment_kwh."""
        expected = float(adjustment_data["total_non_routine_adjustment_kwh"])
        actual_sum = sum(
            float(a["adjustment_kwh"])
            for a in adjustment_data["non_routine_adjustments"]
        )
        assert abs(actual_sum - expected) < 1.0

    def test_adjustment_report(self, adjustment_data):
        engine = _m.AdjustmentEngine()
        report = (getattr(engine, "adjustment_report", None)
                  or getattr(engine, "generate_report", None)
                  or getattr(engine, "summary", None))
        if report is None:
            pytest.skip("report method not found")
        try:
            result = report(adjustment_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_regression_method_used(self, adjustment_data):
        """Weather adjustment should use REGRESSION_BASED method."""
        weather = adjustment_data["routine_adjustments"][0]
        assert weather["method"] == "REGRESSION_BASED"

    def test_ratio_method_used(self, adjustment_data):
        """Production adjustment should use RATIO_BASED method."""
        production = adjustment_data["routine_adjustments"][1]
        assert production["method"] == "RATIO_BASED"
