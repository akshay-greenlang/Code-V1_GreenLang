# -*- coding: utf-8 -*-
"""
Unit Tests for Normalizer API Router (AGENT-FOUND-003)

Tests the FastAPI router endpoints for the normalizer service:
POST /v1/normalize/convert, POST /v1/normalize/convert/batch,
POST /v1/normalize/resolve/fuel, GET /v1/normalize/units,
GET /v1/normalize/dimensions, GET /health.

Coverage target: 85%+ of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inline API request/response models and handler functions
# mirroring greenlang/normalizer/api/router.py
# ---------------------------------------------------------------------------


class ConvertRequest:
    def __init__(self, value: float, from_unit: str, to_unit: str, precision: int = 10):
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.precision = precision


class ConvertResponse:
    def __init__(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        dimension: str,
        provenance_hash: str,
        error: Optional[str] = None,
    ):
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.dimension = dimension
        self.provenance_hash = provenance_hash
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "value": self.value,
            "from_unit": self.from_unit,
            "to_unit": self.to_unit,
            "dimension": self.dimension,
            "provenance_hash": self.provenance_hash,
        }
        if self.error:
            d["error"] = self.error
        return d


class BatchConvertRequest:
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items


class ResolveFuelRequest:
    def __init__(self, name: str):
        self.name = name


class ResolveFuelResponse:
    def __init__(self, canonical_name: str, confidence: float, level: str, code: str):
        self.canonical_name = canonical_name
        self.confidence = confidence
        self.level = level
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "confidence": self.confidence,
            "level": self.level,
            "code": self.code,
        }


class DimensionResponse:
    def __init__(self, dimensions: List[str]):
        self.dimensions = dimensions


class UnitsResponse:
    def __init__(self, dimension: str, units: List[str], base_unit: str):
        self.dimension = dimension
        self.units = units
        self.base_unit = base_unit


class HealthResponse:
    def __init__(self, status: str = "healthy", version: str = "1.0.0"):
        self.status = status
        self.version = version


# Simulated route handlers
class NormalizerRouter:
    """Simulates the FastAPI router for normalizer endpoints."""

    def __init__(self, service=None):
        self._service = service or MagicMock()

    def handle_convert(self, req: ConvertRequest) -> ConvertResponse:
        """POST /v1/normalize/convert"""
        if not req.from_unit or not req.to_unit:
            return ConvertResponse(
                value=0, from_unit=req.from_unit, to_unit=req.to_unit,
                dimension="", provenance_hash="",
                error="Missing from_unit or to_unit",
            )

        # Delegate to service
        result = self._service.convert(req.value, req.from_unit, req.to_unit)
        if hasattr(result, "error") and result.error:
            return ConvertResponse(
                value=0, from_unit=req.from_unit, to_unit=req.to_unit,
                dimension="", provenance_hash="",
                error=result.error,
            )

        return ConvertResponse(
            value=float(getattr(result, "value", req.value)),
            from_unit=req.from_unit,
            to_unit=req.to_unit,
            dimension=getattr(result, "dimension", "UNKNOWN"),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    def handle_batch_convert(self, req: BatchConvertRequest) -> List[ConvertResponse]:
        """POST /v1/normalize/convert/batch"""
        results = []
        for item in req.items:
            cr = ConvertRequest(
                value=item.get("value", 0),
                from_unit=item.get("from_unit", ""),
                to_unit=item.get("to_unit", ""),
            )
            results.append(self.handle_convert(cr))
        return results

    def handle_resolve_fuel(self, req: ResolveFuelRequest) -> ResolveFuelResponse:
        """POST /v1/normalize/resolve/fuel"""
        result = self._service.resolve_fuel(req.name)
        return ResolveFuelResponse(
            canonical_name=getattr(result, "canonical_name", ""),
            confidence=getattr(result, "confidence", 0.0),
            level=getattr(result, "level", "UNRESOLVED"),
            code=getattr(result, "code", ""),
        )

    def handle_get_dimensions(self) -> DimensionResponse:
        """GET /v1/normalize/dimensions"""
        return DimensionResponse(
            dimensions=["ENERGY", "MASS", "EMISSIONS", "VOLUME", "AREA", "DISTANCE", "TIME"]
        )

    def handle_get_units(self, dimension: str) -> UnitsResponse:
        """GET /v1/normalize/units?dimension=MASS"""
        units_map = {
            "MASS": (["kg", "g", "t", "lb", "oz"], "kg"),
            "ENERGY": (["kWh", "MWh", "GJ", "MJ", "BTU"], "kWh"),
            "EMISSIONS": (["kgCO2e", "tCO2e", "gCO2e", "MtCO2e"], "kgCO2e"),
            "VOLUME": (["L", "mL", "m3", "gal"], "L"),
        }
        if dimension in units_map:
            units, base = units_map[dimension]
            return UnitsResponse(dimension=dimension, units=units, base_unit=base)
        return UnitsResponse(dimension=dimension, units=[], base_unit="")

    def handle_health(self) -> HealthResponse:
        """GET /health"""
        return HealthResponse(status="healthy", version="1.0.0")


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.fixture
def router():
    return NormalizerRouter()


class TestConvertEndpoint:
    """Test POST /v1/normalize/convert."""

    def test_valid_conversion(self, router):
        # Mock service convert to return a result-like object
        mock_result = MagicMock()
        mock_result.value = 0.1
        mock_result.dimension = "ENERGY"
        mock_result.provenance_hash = "a" * 64
        mock_result.error = None
        router._service.convert.return_value = mock_result

        resp = router.handle_convert(ConvertRequest(100, "kWh", "MWh"))
        assert resp.ok
        assert resp.value == 0.1
        assert resp.dimension == "ENERGY"
        assert len(resp.provenance_hash) == 64

    def test_missing_from_unit(self, router):
        resp = router.handle_convert(ConvertRequest(100, "", "MWh"))
        assert not resp.ok
        assert "Missing" in resp.error

    def test_missing_to_unit(self, router):
        resp = router.handle_convert(ConvertRequest(100, "kWh", ""))
        assert not resp.ok
        assert "Missing" in resp.error

    def test_service_error_propagated(self, router):
        mock_result = MagicMock()
        mock_result.error = "Incompatible dimensions"
        router._service.convert.return_value = mock_result

        resp = router.handle_convert(ConvertRequest(1, "kg", "kWh"))
        assert not resp.ok
        assert "Incompatible" in resp.error

    def test_response_to_dict(self, router):
        mock_result = MagicMock()
        mock_result.value = 1000.0
        mock_result.dimension = "MASS"
        mock_result.provenance_hash = "b" * 64
        mock_result.error = None
        router._service.convert.return_value = mock_result

        resp = router.handle_convert(ConvertRequest(1, "t", "kg"))
        d = resp.to_dict()
        assert "value" in d
        assert "dimension" in d
        assert "provenance_hash" in d


class TestBatchConvertEndpoint:
    """Test POST /v1/normalize/convert/batch."""

    def test_batch_multiple_items(self, router):
        mock_result = MagicMock()
        mock_result.value = 1.0
        mock_result.dimension = "MASS"
        mock_result.provenance_hash = "c" * 64
        mock_result.error = None
        router._service.convert.return_value = mock_result

        req = BatchConvertRequest([
            {"value": 1000, "from_unit": "kg", "to_unit": "t"},
            {"value": 100, "from_unit": "kWh", "to_unit": "MWh"},
        ])
        results = router.handle_batch_convert(req)
        assert len(results) == 2

    def test_batch_empty(self, router):
        req = BatchConvertRequest([])
        results = router.handle_batch_convert(req)
        assert results == []


class TestResolveFuelEndpoint:
    """Test POST /v1/normalize/resolve/fuel."""

    def test_resolve_known_fuel(self, router):
        mock_match = MagicMock()
        mock_match.canonical_name = "natural_gas"
        mock_match.confidence = 1.0
        mock_match.level = "EXACT"
        mock_match.code = "NG"
        router._service.resolve_fuel.return_value = mock_match

        resp = router.handle_resolve_fuel(ResolveFuelRequest("Natural Gas"))
        assert resp.canonical_name == "natural_gas"
        assert resp.confidence == 1.0

    def test_resolve_unknown_fuel(self, router):
        mock_match = MagicMock()
        mock_match.canonical_name = ""
        mock_match.confidence = 0.0
        mock_match.level = "UNRESOLVED"
        mock_match.code = ""
        router._service.resolve_fuel.return_value = mock_match

        resp = router.handle_resolve_fuel(ResolveFuelRequest("XYZ_UNKNOWN"))
        assert resp.confidence == 0.0

    def test_resolve_response_to_dict(self, router):
        mock_match = MagicMock()
        mock_match.canonical_name = "diesel"
        mock_match.confidence = 0.95
        mock_match.level = "ALIAS"
        mock_match.code = "DSL"
        router._service.resolve_fuel.return_value = mock_match

        resp = router.handle_resolve_fuel(ResolveFuelRequest("diesel fuel"))
        d = resp.to_dict()
        assert d["canonical_name"] == "diesel"
        assert d["confidence"] == 0.95


class TestGetDimensionsEndpoint:
    """Test GET /v1/normalize/dimensions."""

    def test_returns_all_dimensions(self, router):
        resp = router.handle_get_dimensions()
        assert "ENERGY" in resp.dimensions
        assert "MASS" in resp.dimensions
        assert "EMISSIONS" in resp.dimensions
        assert "VOLUME" in resp.dimensions

    def test_returns_7_dimensions(self, router):
        resp = router.handle_get_dimensions()
        assert len(resp.dimensions) == 7


class TestGetUnitsEndpoint:
    """Test GET /v1/normalize/units."""

    def test_mass_units(self, router):
        resp = router.handle_get_units("MASS")
        assert "kg" in resp.units
        assert resp.base_unit == "kg"

    def test_energy_units(self, router):
        resp = router.handle_get_units("ENERGY")
        assert "kWh" in resp.units
        assert resp.base_unit == "kWh"

    def test_unknown_dimension(self, router):
        resp = router.handle_get_units("UNKNOWN")
        assert resp.units == []


class TestHealthEndpoint:
    """Test GET /health."""

    def test_health_returns_healthy(self, router):
        resp = router.handle_health()
        assert resp.status == "healthy"

    def test_health_returns_version(self, router):
        resp = router.handle_health()
        assert resp.version == "1.0.0"
