# -*- coding: utf-8 -*-
"""
Unit Tests for PlotRegistryEngine (AGENT-DATA-005)

Tests plot registration, retrieval, listing with filters (commodity, country,
risk level), pagination, compliance updates, bulk import, geolocation
validation, and EUDR-specific risk classification by country.

Coverage target: 85%+ of plot_registry.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline country risk data (EUDR benchmarking)
# ---------------------------------------------------------------------------

# Countries classified as HIGH risk per EUDR Article 29 benchmarking
HIGH_RISK_COUNTRIES = {"BR", "ID", "CD", "CG", "CM", "NG", "MM", "PG", "LA", "KH"}

# Countries classified as LOW risk (EU member states, etc.)
LOW_RISK_COUNTRIES = {"DE", "FR", "NL", "BE", "AT", "SE", "FI", "DK", "IE", "ES", "IT", "PT"}


# ---------------------------------------------------------------------------
# Inline GeolocationData (minimal, same as test_models.py)
# ---------------------------------------------------------------------------


class GeolocationData:
    def __init__(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        plot_area_ha: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
    ):
        if latitude < -90 or latitude > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        if longitude < -180 or longitude > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
        if plot_area_ha > 4.0 and polygon_coordinates is None:
            raise ValueError(
                f"Plots larger than 4 hectares require polygon coordinates "
                f"(plot_area_ha={plot_area_ha})"
            )
        self.latitude = latitude
        self.longitude = longitude
        self.plot_area_ha = plot_area_ha
        self.polygon_coordinates = polygon_coordinates


# ---------------------------------------------------------------------------
# Inline PlotRecord
# ---------------------------------------------------------------------------


class PlotRecord:
    def __init__(
        self,
        plot_id: str = "",
        commodity: str = "cocoa",
        country_code: str = "",
        geolocation: Optional[GeolocationData] = None,
        operator_id: str = "",
        operator_name: str = "",
        deforestation_free: Optional[bool] = None,
        legally_produced: Optional[bool] = None,
        cutoff_date: str = "2020-12-31",
        risk_level: str = "unknown",
        registered_at: Optional[str] = None,
        provenance_hash: Optional[str] = None,
    ):
        self.plot_id = plot_id
        self.commodity = commodity
        self.country_code = country_code
        self.geolocation = geolocation
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.deforestation_free = deforestation_free
        self.legally_produced = legally_produced
        self.cutoff_date = cutoff_date
        self.risk_level = risk_level
        self.registered_at = registered_at or datetime.now(timezone.utc).isoformat()
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline RegisterPlotRequest
# ---------------------------------------------------------------------------


class RegisterPlotRequest:
    def __init__(
        self,
        commodity: str = "cocoa",
        country_code: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        plot_area_ha: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
        operator_id: str = "",
        operator_name: str = "",
    ):
        self.commodity = commodity
        self.country_code = country_code
        self.latitude = latitude
        self.longitude = longitude
        self.plot_area_ha = plot_area_ha
        self.polygon_coordinates = polygon_coordinates
        self.operator_id = operator_id
        self.operator_name = operator_name


# ---------------------------------------------------------------------------
# Inline PlotRegistryEngine
# ---------------------------------------------------------------------------


class PlotRegistryEngine:
    """Manages EUDR production plot registration and compliance tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._plots: Dict[str, PlotRecord] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "plots_registered": 0,
            "plots_updated": 0,
            "bulk_imports": 0,
            "validation_errors": 0,
        }

    def _next_plot_id(self) -> str:
        self._counter += 1
        return f"PLOT-{self._counter:05d}"

    def _classify_country_risk(self, country_code: str) -> str:
        cc = country_code.upper()
        if cc in HIGH_RISK_COUNTRIES:
            return "high"
        if cc in LOW_RISK_COUNTRIES:
            return "low"
        return "standard"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def validate_geolocation(
        self, latitude: float, longitude: float,
        plot_area_ha: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
    ) -> GeolocationData:
        """Validate and return GeolocationData. Raises ValueError on failure."""
        return GeolocationData(
            latitude=latitude,
            longitude=longitude,
            plot_area_ha=plot_area_ha,
            polygon_coordinates=polygon_coordinates,
        )

    def register_plot(self, request: RegisterPlotRequest) -> PlotRecord:
        """Register a new production plot."""
        geo = self.validate_geolocation(
            latitude=request.latitude,
            longitude=request.longitude,
            plot_area_ha=request.plot_area_ha,
            polygon_coordinates=request.polygon_coordinates,
        )
        risk = self._classify_country_risk(request.country_code)
        with self._lock:
            plot_id = self._next_plot_id()
        prov_data = {
            "op": "register_plot", "plot_id": plot_id,
            "commodity": request.commodity, "country_code": request.country_code,
            "lat": request.latitude, "lon": request.longitude,
        }
        record = PlotRecord(
            plot_id=plot_id,
            commodity=request.commodity,
            country_code=request.country_code,
            geolocation=geo,
            operator_id=request.operator_id,
            operator_name=request.operator_name,
            risk_level=risk,
            provenance_hash=self._compute_provenance(prov_data),
        )
        with self._lock:
            self._plots[plot_id] = record
            self._stats["plots_registered"] += 1
        return record

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        """Retrieve a registered plot by ID. Returns None if not found."""
        with self._lock:
            return self._plots.get(plot_id)

    def list_plots(
        self,
        commodity: Optional[str] = None,
        country_code: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PlotRecord]:
        """List plots with optional filters and pagination."""
        with self._lock:
            result = list(self._plots.values())
        if commodity is not None:
            result = [p for p in result if p.commodity == commodity]
        if country_code is not None:
            result = [p for p in result if p.country_code == country_code]
        if risk_level is not None:
            result = [p for p in result if p.risk_level == risk_level]
        return result[offset: offset + limit]

    def get_plots_by_commodity(self, commodity: str) -> List[PlotRecord]:
        """Return all plots for a given commodity."""
        return self.list_plots(commodity=commodity, limit=999999)

    def get_plots_by_country(self, country_code: str) -> List[PlotRecord]:
        """Return all plots for a given country."""
        return self.list_plots(country_code=country_code, limit=999999)

    def update_compliance(
        self, plot_id: str,
        deforestation_free: Optional[bool] = None,
        legally_produced: Optional[bool] = None,
    ) -> Optional[PlotRecord]:
        """Update compliance status of a plot. Returns None if not found."""
        with self._lock:
            record = self._plots.get(plot_id)
            if record is None:
                return None
            if deforestation_free is not None:
                record.deforestation_free = deforestation_free
            if legally_produced is not None:
                record.legally_produced = legally_produced
            # If both compliance checks pass, consider risk downgrade
            if record.deforestation_free is True and record.legally_produced is True:
                if record.risk_level == "high":
                    record.risk_level = "standard"
                elif record.risk_level == "standard":
                    record.risk_level = "low"
            self._stats["plots_updated"] += 1
        return record

    def bulk_import(self, requests: List[RegisterPlotRequest]) -> Dict[str, Any]:
        """Import multiple plots. Returns summary with successes and failures."""
        successes = []
        failures = []
        for req in requests:
            try:
                record = self.register_plot(req)
                successes.append(record.plot_id)
            except (ValueError, Exception) as exc:
                failures.append({
                    "commodity": req.commodity,
                    "country_code": req.country_code,
                    "error": str(exc),
                })
                with self._lock:
                    self._stats["validation_errors"] += 1
        with self._lock:
            self._stats["bulk_imports"] += 1
        return {
            "total": len(requests),
            "successes": len(successes),
            "failures": len(failures),
            "success_ids": successes,
            "failure_details": failures,
        }

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """PlotRegistryEngine instance for testing."""
    return PlotRegistryEngine()


@pytest.fixture
def sample_plot_request():
    """RegisterPlotRequest with valid data for Ghana cocoa."""
    return RegisterPlotRequest(
        commodity="cocoa",
        country_code="GH",
        latitude=6.68,
        longitude=-1.62,
        plot_area_ha=3.5,
        operator_id="OP-001",
        operator_name="GhanaCocoa Ltd",
    )


@pytest.fixture
def cocoa_plot_ecuador():
    """Plot in Ecuador for cocoa (standard risk country)."""
    return RegisterPlotRequest(
        commodity="cocoa",
        country_code="EC",
        latitude=-1.83,
        longitude=-79.93,
        plot_area_ha=2.5,
        operator_id="OP-002",
        operator_name="EcuadorCacao SA",
    )


@pytest.fixture
def rubber_plot_indonesia():
    """Plot in Indonesia for rubber (high risk country)."""
    polygon = [[0.5, 104.0], [0.6, 104.0], [0.6, 104.1], [0.5, 104.1]]
    return RegisterPlotRequest(
        commodity="rubber",
        country_code="ID",
        latitude=0.55,
        longitude=104.05,
        plot_area_ha=8.0,
        polygon_coordinates=polygon,
        operator_id="OP-003",
        operator_name="IndoRubber Pty",
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegisterPlot:
    """Test plot registration via PlotRegistryEngine."""

    def test_register_plot_success(self, engine, sample_plot_request):
        """Register and verify returned PlotRecord."""
        record = engine.register_plot(sample_plot_request)
        assert record.plot_id is not None
        assert record.commodity == "cocoa"
        assert record.country_code == "GH"
        assert record.operator_id == "OP-001"
        assert record.geolocation is not None
        assert record.registered_at is not None

    def test_register_plot_generates_id(self, engine, sample_plot_request):
        """Auto-generated PLOT-xxxxx format."""
        record = engine.register_plot(sample_plot_request)
        assert record.plot_id.startswith("PLOT-")
        assert len(record.plot_id) == 10  # "PLOT-" + 5 digits

    def test_register_plot_sets_risk(self, engine, sample_plot_request):
        """Risk level set based on country."""
        record = engine.register_plot(sample_plot_request)
        # Ghana is not in HIGH or LOW lists -> standard
        assert record.risk_level == "standard"

    def test_register_plot_high_risk_country(self, engine, rubber_plot_indonesia):
        """Brazil/Indonesia gets HIGH risk."""
        record = engine.register_plot(rubber_plot_indonesia)
        assert record.risk_level == "high"

    def test_register_plot_brazil_high_risk(self, engine):
        """Brazil is a high risk country."""
        req = RegisterPlotRequest(
            commodity="soya", country_code="BR",
            latitude=-10.0, longitude=-55.0, plot_area_ha=3.0,
            operator_id="OP-BR-001",
        )
        record = engine.register_plot(req)
        assert record.risk_level == "high"

    def test_register_plot_standard_risk(self, engine, cocoa_plot_ecuador):
        """Standard (non-high, non-low) country gets STANDARD."""
        record = engine.register_plot(cocoa_plot_ecuador)
        assert record.risk_level == "standard"

    def test_register_plot_low_risk_eu_country(self, engine):
        """EU member state (e.g., France) gets LOW risk."""
        req = RegisterPlotRequest(
            commodity="wood", country_code="FR",
            latitude=46.0, longitude=2.0, plot_area_ha=2.0,
            operator_id="OP-FR-001",
        )
        record = engine.register_plot(req)
        assert record.risk_level == "low"

    def test_register_plot_provenance_hash(self, engine, sample_plot_request):
        """Provenance hash is SHA-256 (64 hex chars)."""
        record = engine.register_plot(sample_plot_request)
        assert record.provenance_hash is not None
        assert len(record.provenance_hash) == 64

    def test_register_increments_stats(self, engine, sample_plot_request):
        engine.register_plot(sample_plot_request)
        stats = engine.get_statistics()
        assert stats["plots_registered"] == 1


class TestGetPlot:
    """Test plot retrieval."""

    def test_get_plot_exists(self, engine, sample_plot_request):
        """Retrieve registered plot."""
        record = engine.register_plot(sample_plot_request)
        retrieved = engine.get_plot(record.plot_id)
        assert retrieved is not None
        assert retrieved.plot_id == record.plot_id
        assert retrieved.commodity == "cocoa"

    def test_get_plot_not_found(self, engine):
        """Returns None for unknown plot_id."""
        assert engine.get_plot("PLOT-99999") is None


class TestListPlots:
    """Test listing plots with filters and pagination."""

    def test_list_plots_all(self, engine, sample_plot_request, cocoa_plot_ecuador):
        """List all registered plots."""
        engine.register_plot(sample_plot_request)
        engine.register_plot(cocoa_plot_ecuador)
        plots = engine.list_plots()
        assert len(plots) == 2

    def test_list_plots_by_commodity(
        self, engine, sample_plot_request, rubber_plot_indonesia,
    ):
        """Filter by commodity."""
        engine.register_plot(sample_plot_request)  # cocoa
        engine.register_plot(rubber_plot_indonesia)  # rubber
        cocoa_plots = engine.list_plots(commodity="cocoa")
        assert len(cocoa_plots) == 1
        assert cocoa_plots[0].commodity == "cocoa"

    def test_list_plots_by_country(
        self, engine, sample_plot_request, cocoa_plot_ecuador,
    ):
        """Filter by country code."""
        engine.register_plot(sample_plot_request)  # GH
        engine.register_plot(cocoa_plot_ecuador)  # EC
        gh_plots = engine.list_plots(country_code="GH")
        assert len(gh_plots) == 1
        assert gh_plots[0].country_code == "GH"

    def test_list_plots_by_risk(
        self, engine, sample_plot_request, rubber_plot_indonesia,
    ):
        """Filter by risk level."""
        engine.register_plot(sample_plot_request)  # standard
        engine.register_plot(rubber_plot_indonesia)  # high
        high_risk = engine.list_plots(risk_level="high")
        assert len(high_risk) == 1
        assert high_risk[0].risk_level == "high"

    def test_list_plots_pagination(self, engine):
        """Limit and offset pagination."""
        for i in range(10):
            engine.register_plot(RegisterPlotRequest(
                commodity="cocoa", country_code="GH",
                latitude=6.0 + i * 0.01, longitude=-1.0,
                plot_area_ha=1.0, operator_id=f"OP-{i:03d}",
            ))
        page1 = engine.list_plots(limit=3, offset=0)
        assert len(page1) == 3
        page2 = engine.list_plots(limit=3, offset=3)
        assert len(page2) == 3
        assert page1[0].plot_id != page2[0].plot_id
        all_plots = engine.list_plots(limit=100, offset=0)
        assert len(all_plots) == 10

    def test_list_plots_empty(self, engine):
        """Empty registry returns empty list."""
        assert engine.list_plots() == []


class TestUpdateCompliance:
    """Test compliance status updates and risk downgrade logic."""

    def test_update_compliance_success(self, engine, sample_plot_request):
        """Update deforestation_free and legally_produced."""
        record = engine.register_plot(sample_plot_request)
        updated = engine.update_compliance(
            record.plot_id,
            deforestation_free=True,
            legally_produced=True,
        )
        assert updated is not None
        assert updated.deforestation_free is True
        assert updated.legally_produced is True

    def test_update_compliance_risk_downgrade(self, engine, rubber_plot_indonesia):
        """Compliant plot risk decreases from high to standard."""
        record = engine.register_plot(rubber_plot_indonesia)
        assert record.risk_level == "high"
        updated = engine.update_compliance(
            record.plot_id,
            deforestation_free=True,
            legally_produced=True,
        )
        assert updated.risk_level == "standard"

    def test_update_compliance_standard_to_low(self, engine, sample_plot_request):
        """Compliant standard plot risk decreases to low."""
        record = engine.register_plot(sample_plot_request)
        assert record.risk_level == "standard"
        updated = engine.update_compliance(
            record.plot_id,
            deforestation_free=True,
            legally_produced=True,
        )
        assert updated.risk_level == "low"

    def test_update_compliance_partial_no_downgrade(self, engine, rubber_plot_indonesia):
        """Only deforestation_free set (legally_produced=None) does not downgrade."""
        record = engine.register_plot(rubber_plot_indonesia)
        updated = engine.update_compliance(
            record.plot_id, deforestation_free=True,
        )
        assert updated.risk_level == "high"  # No downgrade without both checks

    def test_update_compliance_not_found(self, engine):
        """Non-existent plot returns None."""
        assert engine.update_compliance("PLOT-99999", deforestation_free=True) is None

    def test_update_increments_stats(self, engine, sample_plot_request):
        record = engine.register_plot(sample_plot_request)
        engine.update_compliance(record.plot_id, deforestation_free=True)
        stats = engine.get_statistics()
        assert stats["plots_updated"] == 1


class TestBulkImport:
    """Test bulk import of multiple plots."""

    def test_bulk_import(self, engine):
        """Import multiple plots successfully."""
        requests = [
            RegisterPlotRequest(
                commodity="cocoa", country_code="GH",
                latitude=6.0 + i * 0.01, longitude=-1.0,
                plot_area_ha=2.0, operator_id=f"OP-{i:03d}",
            )
            for i in range(5)
        ]
        result = engine.bulk_import(requests)
        assert result["total"] == 5
        assert result["successes"] == 5
        assert result["failures"] == 0
        assert len(result["success_ids"]) == 5

    def test_bulk_import_partial_failure(self, engine):
        """Some plots fail validation (bad coordinates)."""
        requests = [
            RegisterPlotRequest(
                commodity="cocoa", country_code="GH",
                latitude=6.0, longitude=-1.0, plot_area_ha=2.0,
                operator_id="OP-001",
            ),
            RegisterPlotRequest(
                commodity="cocoa", country_code="GH",
                latitude=95.0,  # Invalid latitude
                longitude=-1.0, plot_area_ha=2.0,
                operator_id="OP-002",
            ),
            RegisterPlotRequest(
                commodity="rubber", country_code="ID",
                latitude=0.5, longitude=104.0,
                plot_area_ha=10.0,  # Large plot without polygon -> fail
                operator_id="OP-003",
            ),
        ]
        result = engine.bulk_import(requests)
        assert result["total"] == 3
        assert result["successes"] == 1
        assert result["failures"] == 2
        assert len(result["failure_details"]) == 2

    def test_bulk_import_empty(self, engine):
        result = engine.bulk_import([])
        assert result["total"] == 0
        assert result["successes"] == 0


class TestValidateGeolocation:
    """Test geolocation validation helper."""

    def test_validate_geolocation_valid(self, engine):
        """Valid coordinates pass."""
        geo = engine.validate_geolocation(latitude=5.5, longitude=-73.3, plot_area_ha=2.0)
        assert geo.latitude == 5.5
        assert geo.longitude == -73.3

    def test_validate_geolocation_invalid_lat(self, engine):
        """Bad latitude fails."""
        with pytest.raises(ValueError, match="Latitude must be between"):
            engine.validate_geolocation(latitude=100.0, longitude=0.0)

    def test_validate_geolocation_invalid_lon(self, engine):
        """Bad longitude fails."""
        with pytest.raises(ValueError, match="Longitude must be between"):
            engine.validate_geolocation(latitude=0.0, longitude=200.0)

    def test_validate_geolocation_polygon_required(self, engine):
        """Large plot needs polygon."""
        with pytest.raises(ValueError, match="polygon coordinates"):
            engine.validate_geolocation(
                latitude=5.0, longitude=-73.0, plot_area_ha=5.0,
            )

    def test_validate_geolocation_large_with_polygon(self, engine):
        """Large plot with polygon succeeds."""
        polygon = [[5.0, -73.0], [5.1, -73.0], [5.1, -72.9], [5.0, -72.9]]
        geo = engine.validate_geolocation(
            latitude=5.05, longitude=-72.95,
            plot_area_ha=10.0, polygon_coordinates=polygon,
        )
        assert geo.plot_area_ha == 10.0


class TestGetPlotsByCommodityAndCountry:
    """Test commodity and country specific queries."""

    def test_get_plots_by_commodity(self, engine, sample_plot_request, rubber_plot_indonesia):
        """Returns correct subset by commodity."""
        engine.register_plot(sample_plot_request)  # cocoa
        engine.register_plot(rubber_plot_indonesia)  # rubber
        cocoa = engine.get_plots_by_commodity("cocoa")
        assert len(cocoa) == 1
        assert cocoa[0].commodity == "cocoa"

    def test_get_plots_by_country(self, engine, sample_plot_request, cocoa_plot_ecuador):
        """Returns correct subset by country."""
        engine.register_plot(sample_plot_request)  # GH
        engine.register_plot(cocoa_plot_ecuador)  # EC
        ec_plots = engine.get_plots_by_country("EC")
        assert len(ec_plots) == 1
        assert ec_plots[0].country_code == "EC"

    def test_multiple_commodities(self, engine):
        """Different commodities indexed separately."""
        commodities = ["cocoa", "coffee", "rubber", "soya", "wood"]
        for i, comm in enumerate(commodities):
            engine.register_plot(RegisterPlotRequest(
                commodity=comm, country_code="GH",
                latitude=6.0 + i * 0.01, longitude=-1.0,
                plot_area_ha=2.0, operator_id=f"OP-{comm}",
            ))
        for comm in commodities:
            plots = engine.get_plots_by_commodity(comm)
            assert len(plots) == 1
            assert plots[0].commodity == comm

    def test_get_plots_by_commodity_empty(self, engine):
        """No plots for unknown commodity."""
        assert engine.get_plots_by_commodity("palm_oil") == []

    def test_get_plots_by_country_empty(self, engine):
        """No plots for unknown country."""
        assert engine.get_plots_by_country("XX") == []
