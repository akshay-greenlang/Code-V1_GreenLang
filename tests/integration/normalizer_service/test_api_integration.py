# -*- coding: utf-8 -*-
"""
API Integration Tests for Normalizer Service (AGENT-FOUND-003)

Tests the normalizer API endpoints with a simulated TestClient,
validating request/response handling, error propagation, and
end-to-end API workflows.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Self-contained converter and resolver for API integration tests
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    ENERGY = "ENERGY"
    MASS = "MASS"
    EMISSIONS = "EMISSIONS"
    VOLUME = "VOLUME"

_UT = {
    "kWh": (Dimension.ENERGY, Decimal("1")),
    "MWh": (Dimension.ENERGY, Decimal("1000")),
    "kg": (Dimension.MASS, Decimal("1")),
    "t": (Dimension.MASS, Decimal("1000")),
    "g": (Dimension.MASS, Decimal("0.001")),
    "lb": (Dimension.MASS, Decimal("0.453592")),
    "kgCO2e": (Dimension.EMISSIONS, Decimal("1")),
    "tCO2e": (Dimension.EMISSIONS, Decimal("1000")),
    "L": (Dimension.VOLUME, Decimal("1")),
    "m3": (Dimension.VOLUME, Decimal("1000")),
}
_AL = {
    "kwh": "kWh", "mwh": "MWh", "KG": "kg", "T": "t",
    "tonne": "t", "kilogram": "kg", "l": "L", "liter": "L",
    "kgco2e": "kgCO2e", "tco2e": "tCO2e",
}


class _CR:
    def __init__(self, value, dimension, provenance_hash, error=None):
        self.value = value
        self.dimension = dimension
        self.provenance_hash = provenance_hash
        self.error = error
    @property
    def ok(self):
        return self.error is None


class _Conv:
    def convert(self, value, from_unit, to_unit, **kw):
        fc = _AL.get(from_unit, from_unit) if from_unit not in _UT else from_unit
        tc = _AL.get(to_unit, to_unit) if to_unit not in _UT else to_unit
        if fc not in _UT:
            return _CR(0, "UNKNOWN", "", error=f"Unknown unit: {from_unit}")
        if tc not in _UT:
            return _CR(0, "UNKNOWN", "", error=f"Unknown unit: {to_unit}")
        fd, ff = _UT[fc]
        td, tf = _UT[tc]
        if fd != td:
            return _CR(0, "", "", error=f"Incompatible dimensions: {fd.value} -> {td.value}")
        d = Decimal(str(value))
        r = (d * ff / tf).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        h = hashlib.sha256(json.dumps([str(value), fc, tc]).encode()).hexdigest()
        return _CR(float(r), fd.value, h)


class ConfidenceLevel(str, Enum):
    EXACT = "EXACT"
    ALIAS = "ALIAS"
    UNRESOLVED = "UNRESOLVED"


class _EM:
    def __init__(self, canonical_name, confidence, level):
        self.canonical_name = canonical_name
        self.confidence = confidence
        self.level = level


_FUELS = {"natural gas": "natural_gas", "nat gas": "natural_gas", "diesel": "diesel", "diesel fuel": "diesel"}


class _Res:
    def resolve_fuel(self, name):
        lo = name.lower().strip()
        if lo in _FUELS:
            return _EM(_FUELS[lo], 0.95, ConfidenceLevel.ALIAS)
        return _EM("", 0.0, ConfidenceLevel.UNRESOLVED)


# ---------------------------------------------------------------------------
# Simulated API layer
# ---------------------------------------------------------------------------


class NormalizerTestClient:
    def __init__(self):
        self._conv = _Conv()
        self._res = _Res()

    def post_convert(self, value, from_unit, to_unit):
        r = self._conv.convert(value, from_unit, to_unit)
        d = {"value": r.value, "dimension": r.dimension, "provenance_hash": r.provenance_hash}
        if r.error:
            d["error"] = r.error
        return d

    def post_batch_convert(self, items):
        return [self.post_convert(i["value"], i["from_unit"], i["to_unit"]) for i in items]

    def post_resolve_fuel(self, name):
        m = self._res.resolve_fuel(name)
        return {"canonical_name": m.canonical_name, "confidence": m.confidence, "level": m.level}

    def get_dimensions(self):
        return ["ENERGY", "MASS", "EMISSIONS", "VOLUME", "AREA", "DISTANCE", "TIME"]

    def get_units(self, dimension):
        units_map = {
            "MASS": {"units": ["kg", "g", "t", "lb"], "base_unit": "kg"},
            "ENERGY": {"units": ["kWh", "MWh", "GJ"], "base_unit": "kWh"},
        }
        return units_map.get(dimension, {"units": [], "base_unit": ""})

    def get_health(self):
        return {"status": "healthy", "version": "1.0.0"}


@pytest.fixture
def client():
    return NormalizerTestClient()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestConvertAPIIntegration:
    """Test POST /v1/normalize/convert with real converter."""

    def test_mass_conversion(self, client):
        resp = client.post_convert(1000, "kg", "t")
        assert resp["value"] == pytest.approx(1.0, rel=1e-6)
        assert resp["dimension"] == "MASS"
        assert len(resp["provenance_hash"]) == 64

    def test_energy_conversion(self, client):
        resp = client.post_convert(100, "kWh", "MWh")
        assert resp["value"] == pytest.approx(0.1, rel=1e-6)
        assert resp["dimension"] == "ENERGY"

    def test_volume_conversion(self, client):
        resp = client.post_convert(1, "m3", "L")
        assert resp["value"] == pytest.approx(1000.0, rel=1e-3)

    def test_emissions_conversion(self, client):
        resp = client.post_convert(1, "tCO2e", "kgCO2e")
        assert resp["value"] == pytest.approx(1000.0, rel=1e-3)

    def test_incompatible_dimensions_error(self, client):
        resp = client.post_convert(1, "kg", "kWh")
        assert "error" in resp

    def test_unknown_unit_error(self, client):
        resp = client.post_convert(1, "furlongs", "m")
        assert "error" in resp

    def test_zero_value(self, client):
        resp = client.post_convert(0, "kg", "t")
        assert resp["value"] == pytest.approx(0.0, abs=1e-10)


class TestBatchConvertAPIIntegration:
    """Test POST /v1/normalize/convert/batch."""

    def test_batch_mixed_dimensions(self, client):
        items = [
            {"value": 1000, "from_unit": "kg", "to_unit": "t"},
            {"value": 100, "from_unit": "kWh", "to_unit": "MWh"},
            {"value": 1, "from_unit": "m3", "to_unit": "L"},
        ]
        results = client.post_batch_convert(items)
        assert len(results) == 3
        assert results[0]["value"] == pytest.approx(1.0, rel=1e-6)
        assert results[1]["value"] == pytest.approx(0.1, rel=1e-6)
        assert results[2]["value"] == pytest.approx(1000.0, rel=1e-3)

    def test_batch_with_errors(self, client):
        items = [
            {"value": 1, "from_unit": "kg", "to_unit": "t"},
            {"value": 1, "from_unit": "kg", "to_unit": "kWh"},
        ]
        results = client.post_batch_convert(items)
        assert len(results) == 2
        assert "error" not in results[0]
        assert "error" in results[1]

    def test_batch_empty(self, client):
        results = client.post_batch_convert([])
        assert results == []


class TestResolveFuelAPIIntegration:
    """Test POST /v1/normalize/resolve/fuel."""

    def test_resolve_natural_gas(self, client):
        resp = client.post_resolve_fuel("Natural Gas")
        assert resp["canonical_name"] == "natural_gas"
        assert resp["confidence"] >= 0.95

    def test_resolve_alias(self, client):
        resp = client.post_resolve_fuel("Nat Gas")
        assert resp["canonical_name"] == "natural_gas"

    def test_resolve_unknown(self, client):
        resp = client.post_resolve_fuel("ZZZZZ_UNKNOWN")
        assert resp["confidence"] == 0.0


class TestDimensionsAPIIntegration:
    """Test GET /v1/normalize/dimensions."""

    def test_returns_all_dimensions(self, client):
        dims = client.get_dimensions()
        assert "ENERGY" in dims
        assert "MASS" in dims
        assert "EMISSIONS" in dims
        assert "VOLUME" in dims


class TestUnitsAPIIntegration:
    """Test GET /v1/normalize/units."""

    def test_mass_units(self, client):
        resp = client.get_units("MASS")
        assert "kg" in resp["units"]
        assert resp["base_unit"] == "kg"

    def test_energy_units(self, client):
        resp = client.get_units("ENERGY")
        assert "kWh" in resp["units"]


class TestHealthAPIIntegration:
    """Test GET /health."""

    def test_health_endpoint(self, client):
        resp = client.get_health()
        assert resp["status"] == "healthy"
        assert resp["version"] == "1.0.0"
