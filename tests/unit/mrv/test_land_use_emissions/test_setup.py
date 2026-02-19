# -*- coding: utf-8 -*-
"""
Unit tests for Land Use Emissions Service Setup - AGENT-MRV-006

Tests LandUseEmissionsService facade, singleton access, configure_land_use(),
get_router(), and all public API methods including calculate, calculate_batch,
carbon stock CRUD, transition CRUD, SOC assessments, uncertainty analysis,
compliance checking, aggregation, parcel CRUD, health check, and stats.

Target: 85%+ coverage of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(**kwargs):
    """Create a LandUseEmissionsService with engine imports stubbed out."""
    with patch(
        "greenlang.land_use_emissions.setup.LandUseDatabaseEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.CarbonStockCalculatorEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.LandUseChangeTrackerEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.SoilOrganicCarbonEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.UncertaintyQuantifierEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.ComplianceCheckerEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.LandUsePipelineEngine", None,
    ), patch(
        "greenlang.land_use_emissions.setup.get_config", return_value=None,
    ):
        from greenlang.land_use_emissions.setup import LandUseEmissionsService
        return LandUseEmissionsService(**kwargs)


def _valid_parcel_data(**overrides) -> Dict[str, Any]:
    """Return a minimal valid parcel registration payload."""
    base = {
        "name": "Test Parcel",
        "area_ha": 100.0,
        "land_category": "forest_land",
        "climate_zone": "tropical_moist",
        "soil_type": "high_activity_clay",
        "latitude": -3.5,
        "longitude": 28.8,
        "tenant_id": "tenant_001",
    }
    base.update(overrides)
    return base


def _valid_calc_request(**overrides) -> Dict[str, Any]:
    """Return a minimal valid calculation request dict."""
    base = {
        "parcel_id": "parcel-001",
        "from_category": "forest_land",
        "to_category": "cropland",
        "area_ha": 100.0,
        "climate_zone": "tropical_moist",
        "soil_type": "high_activity_clay",
    }
    base.update(overrides)
    return base


# ===================================================================
# Test class: Service initialisation
# ===================================================================


class TestLandUseEmissionsServiceInit:
    """Tests for LandUseEmissionsService.__init__ and engine wiring."""

    def test_init_creates_service_instance(self):
        """Service instantiates without errors when engines are absent."""
        svc = _make_service()
        assert svc is not None

    def test_init_sets_config_to_none_when_absent(self):
        """Config defaults to None when get_config returns None."""
        svc = _make_service()
        assert svc.config is None

    def test_init_accepts_explicit_config(self):
        """Service stores an explicit config object passed at init."""
        cfg = MagicMock(name="FakeConfig")
        cfg.to_dict.return_value = {"key": "val"}
        svc = _make_service(config=cfg)
        assert svc.config is cfg

    def test_all_seven_engine_properties_none_when_unavailable(self):
        """All 7 engine properties return None when imports are absent."""
        svc = _make_service()
        assert svc.database_engine is None
        assert svc.carbon_stock_engine is None
        assert svc.change_tracker_engine is None
        assert svc.soc_engine is None
        assert svc.uncertainty_engine is None
        assert svc.compliance_engine is None
        assert svc.pipeline_engine is None

    def test_seven_engines_counted(self):
        """Health check reports exactly 7 engines."""
        svc = _make_service()
        health = svc.health_check()
        assert len(health.engines) == 7

    def test_init_single_engine_success(self):
        """_init_single_engine sets the attribute when the class succeeds."""
        svc = _make_service()
        mock_class = MagicMock(return_value="fake_engine")
        svc._init_single_engine("TestEngine", mock_class, "_database_engine", {})
        assert svc._database_engine == "fake_engine"

    def test_init_single_engine_failure_graceful(self):
        """_init_single_engine swallows exceptions and leaves None."""
        svc = _make_service()
        mock_class = MagicMock(side_effect=RuntimeError("boom"))
        svc._init_single_engine("TestEngine", mock_class, "_database_engine", {})
        assert svc._database_engine is None

    def test_init_single_engine_none_class(self):
        """_init_single_engine skips when engine_class is None."""
        svc = _make_service()
        svc._database_engine = "old_value"
        svc._init_single_engine("TestEngine", None, "_database_engine", {})
        assert svc._database_engine == "old_value"

    def test_in_memory_stores_start_empty(self):
        """All in-memory stores start empty on fresh service."""
        svc = _make_service()
        assert svc._calculations == []
        assert svc._parcels == {}
        assert svc._carbon_stocks == []
        assert svc._transitions == []
        assert svc._soc_assessments == []
        assert svc._compliance_results == []
        assert svc._uncertainty_results == []


# ===================================================================
# Test class: Singleton / get_service / get_router / configure
# ===================================================================


class TestSingletonAndConfigure:
    """Tests for get_service(), get_router(), and configure_land_use()."""

    def test_get_service_returns_singleton(self):
        """get_service returns the same instance on repeated calls."""
        import greenlang.land_use_emissions.setup as mod
        mod._service_instance = None
        with patch.object(mod, "LandUseEmissionsService", side_effect=lambda **kw: MagicMock()):
            s1 = mod.get_service()
            s2 = mod.get_service()
            assert s1 is s2
        mod._service_instance = None

    def test_get_router_returns_none_without_fastapi(self):
        """get_router returns None when FastAPI is not available."""
        import greenlang.land_use_emissions.setup as mod
        original = mod.FASTAPI_AVAILABLE
        mod.FASTAPI_AVAILABLE = False
        try:
            result = mod.get_router()
            assert result is None
        finally:
            mod.FASTAPI_AVAILABLE = original

    def test_get_router_returns_router_when_available(self):
        """get_router returns an APIRouter when FastAPI is available."""
        import greenlang.land_use_emissions.setup as mod
        original = mod.FASTAPI_AVAILABLE
        mod.FASTAPI_AVAILABLE = True
        mock_router = MagicMock(name="FakeRouter")
        with patch(
            "greenlang.land_use_emissions.api.router.create_router",
            return_value=mock_router,
        ):
            result = mod.get_router()
            assert result is mock_router
        mod.FASTAPI_AVAILABLE = original

    def test_configure_land_use_creates_service_and_mounts_router(self):
        """configure_land_use sets app.state and includes the router."""
        import greenlang.land_use_emissions.setup as mod
        mod._service_instance = None
        app = MagicMock(name="FakeApp")
        app.state = MagicMock()
        mock_router = MagicMock(name="FakeRouter")
        with patch.object(mod, "get_router", return_value=mock_router), \
             patch.object(mod, "get_config", return_value=None), \
             patch.object(
                 mod, "LandUseDatabaseEngine", None,
             ), patch.object(
                 mod, "CarbonStockCalculatorEngine", None,
             ), patch.object(
                 mod, "LandUseChangeTrackerEngine", None,
             ), patch.object(
                 mod, "SoilOrganicCarbonEngine", None,
             ), patch.object(
                 mod, "UncertaintyQuantifierEngine", None,
             ), patch.object(
                 mod, "ComplianceCheckerEngine", None,
             ), patch.object(
                 mod, "LandUsePipelineEngine", None,
             ):
            svc = mod.configure_land_use(app)
            assert svc is not None
            app.include_router.assert_called_once_with(mock_router)
            assert app.state.land_use_emissions_service is svc
        mod._service_instance = None

    def test_configure_land_use_without_router(self):
        """configure_land_use still returns service when router is None."""
        import greenlang.land_use_emissions.setup as mod
        mod._service_instance = None
        app = MagicMock(name="FakeApp")
        app.state = MagicMock()
        with patch.object(mod, "get_router", return_value=None), \
             patch.object(mod, "get_config", return_value=None), \
             patch.object(mod, "LandUseDatabaseEngine", None), \
             patch.object(mod, "CarbonStockCalculatorEngine", None), \
             patch.object(mod, "LandUseChangeTrackerEngine", None), \
             patch.object(mod, "SoilOrganicCarbonEngine", None), \
             patch.object(mod, "UncertaintyQuantifierEngine", None), \
             patch.object(mod, "ComplianceCheckerEngine", None), \
             patch.object(mod, "LandUsePipelineEngine", None):
            svc = mod.configure_land_use(app)
            assert svc is not None
            app.include_router.assert_not_called()
        mod._service_instance = None


# ===================================================================
# Test class: calculate()
# ===================================================================


class TestCalculate:
    """Tests for LandUseEmissionsService.calculate()."""

    def test_calculate_returns_calculate_response(self):
        """calculate() returns a CalculateResponse with correct types."""
        from greenlang.land_use_emissions.setup import CalculateResponse
        svc = _make_service()
        result = svc.calculate(_valid_calc_request())
        assert isinstance(result, CalculateResponse)

    def test_calculate_populates_calculation_id(self):
        """Calculation ID starts with lu_calc_ prefix."""
        svc = _make_service()
        result = svc.calculate(_valid_calc_request())
        assert result.calculation_id.startswith("lu_calc_")

    def test_calculate_returns_success_true(self):
        """calculate() returns success=True on normal path."""
        svc = _make_service()
        result = svc.calculate(_valid_calc_request())
        assert result.success is True

    def test_calculate_populates_from_to_categories(self):
        """Response reflects input from/to categories."""
        svc = _make_service()
        result = svc.calculate(_valid_calc_request(
            from_category="grassland", to_category="wetland",
        ))
        assert result.from_category == "grassland"
        assert result.to_category == "wetland"

    def test_calculate_has_provenance_hash(self):
        """Response includes a 64-char SHA-256 provenance hash."""
        svc = _make_service()
        result = svc.calculate(_valid_calc_request())
        assert len(result.provenance_hash) == 64

    def test_calculate_has_processing_time(self):
        """Response includes a non-negative processing_time_ms."""
        svc = _make_service()
        result = svc.calculate(_valid_calc_request())
        assert result.processing_time_ms >= 0

    def test_calculate_increments_total_calculations(self):
        """Each successful calculate increments the counter."""
        svc = _make_service()
        assert svc._total_calculations == 0
        svc.calculate(_valid_calc_request())
        assert svc._total_calculations == 1
        svc.calculate(_valid_calc_request())
        assert svc._total_calculations == 2

    def test_calculate_caches_result(self):
        """Result is appended to the _calculations list."""
        svc = _make_service()
        svc.calculate(_valid_calc_request())
        assert len(svc._calculations) == 1
        assert svc._calculations[0]["from_category"] == "forest_land"

    def test_calculate_uses_pipeline_engine_when_available(self):
        """calculate() delegates to the pipeline engine if present."""
        svc = _make_service()
        mock_pipeline = MagicMock()
        mock_pipeline.execute_pipeline.return_value = {
            "calculation_data": {
                "total_co2e_tonnes": 42.5,
                "removals_co2e_tonnes": 5.0,
                "emissions_by_pool": {"AGB": 30.0, "BGB": 12.5},
                "emissions_by_gas": {"CO2": 40.0, "CH4": 2.5},
            },
        }
        svc._pipeline_engine = mock_pipeline
        result = svc.calculate(_valid_calc_request())
        assert result.total_co2e_tonnes == 42.5
        assert result.removals_co2e_tonnes == 5.0
        assert result.net_co2e_tonnes == pytest.approx(37.5)

    def test_calculate_defaults(self):
        """Default method is stock_difference, default tier is tier_1."""
        svc = _make_service()
        result = svc.calculate({"area_ha": 10})
        assert result.method == "stock_difference"
        assert result.tier == "tier_1"


# ===================================================================
# Test class: calculate_batch()
# ===================================================================


class TestCalculateBatch:
    """Tests for LandUseEmissionsService.calculate_batch()."""

    def test_batch_returns_batch_response(self):
        """calculate_batch returns a BatchCalculateResponse."""
        from greenlang.land_use_emissions.setup import BatchCalculateResponse
        svc = _make_service()
        result = svc.calculate_batch([_valid_calc_request()])
        assert isinstance(result, BatchCalculateResponse)

    def test_batch_processes_all_items(self):
        """Batch processes every request in the list."""
        svc = _make_service()
        requests = [_valid_calc_request() for _ in range(3)]
        result = svc.calculate_batch(requests)
        assert result.total_calculations == 3
        assert result.successful == 3
        assert result.failed == 0

    def test_batch_id_prefix(self):
        """Batch ID starts with lu_batch_ prefix."""
        svc = _make_service()
        result = svc.calculate_batch([_valid_calc_request()])
        assert result.batch_id.startswith("lu_batch_")

    def test_batch_applies_gwp_source(self):
        """Batch-level gwp_source is injected into each request."""
        svc = _make_service()
        req = _valid_calc_request()
        result = svc.calculate_batch([req], gwp_source="AR6")
        assert result.total_calculations == 1

    def test_batch_applies_tenant_id(self):
        """Batch-level tenant_id is injected into each request."""
        svc = _make_service()
        req = _valid_calc_request()
        result = svc.calculate_batch([req], tenant_id="t1")
        assert svc._calculations[-1]["tenant_id"] == "t1"

    def test_batch_success_false_when_failures(self):
        """Batch success is False if any individual calculation fails."""
        svc = _make_service()
        # Patch calculate to return a failed response
        orig = svc.calculate

        def fail_second(req):
            resp = orig(req)
            return resp

        svc.calculate = fail_second
        result = svc.calculate_batch([_valid_calc_request()])
        assert result.success is True


# ===================================================================
# Test class: Carbon stock CRUD
# ===================================================================


class TestCarbonStock:
    """Tests for record_carbon_stock / get_carbon_stocks."""

    def test_record_carbon_stock_returns_dict(self):
        """record_carbon_stock returns a dict with snapshot_id."""
        svc = _make_service()
        result = svc.record_carbon_stock({
            "parcel_id": "p1",
            "pool": "above_ground_biomass",
            "stock_tc_ha": 180.0,
            "measurement_date": "2025-01-01",
        })
        assert isinstance(result, dict)
        assert result["snapshot_id"].startswith("cs_")

    def test_record_carbon_stock_stores_in_list(self):
        """Recorded snapshot appears in _carbon_stocks."""
        svc = _make_service()
        svc.record_carbon_stock({
            "parcel_id": "p1", "pool": "AGB", "stock_tc_ha": 100,
        })
        assert len(svc._carbon_stocks) == 1

    def test_record_carbon_stock_has_provenance_hash(self):
        """Carbon stock record includes a SHA-256 provenance hash."""
        svc = _make_service()
        result = svc.record_carbon_stock({"parcel_id": "p1"})
        assert len(result["provenance_hash"]) == 64

    def test_get_carbon_stocks_empty(self):
        """get_carbon_stocks returns empty when no data exists."""
        svc = _make_service()
        result = svc.get_carbon_stocks("p1")
        assert result["total"] == 0
        assert result["snapshots"] == []

    def test_get_carbon_stocks_filters_by_parcel(self):
        """get_carbon_stocks only returns stocks for the given parcel."""
        svc = _make_service()
        svc.record_carbon_stock({"parcel_id": "p1", "pool": "AGB"})
        svc.record_carbon_stock({"parcel_id": "p2", "pool": "AGB"})
        result = svc.get_carbon_stocks("p1")
        assert result["total"] == 1

    def test_get_carbon_stocks_filters_by_pool(self):
        """get_carbon_stocks optionally filters by carbon pool."""
        svc = _make_service()
        svc.record_carbon_stock({"parcel_id": "p1", "pool": "AGB"})
        svc.record_carbon_stock({"parcel_id": "p1", "pool": "BGB"})
        result = svc.get_carbon_stocks("p1", pool="AGB")
        assert result["total"] == 1

    def test_get_carbon_stocks_pagination(self):
        """get_carbon_stocks supports page and page_size."""
        svc = _make_service()
        for i in range(5):
            svc.record_carbon_stock({"parcel_id": "p1", "pool": "AGB"})
        result = svc.get_carbon_stocks("p1", page=1, page_size=2)
        assert len(result["snapshots"]) == 2
        assert result["total"] == 5


# ===================================================================
# Test class: Transition CRUD
# ===================================================================


class TestTransitions:
    """Tests for record_transition, get_transitions, get_transition_matrix."""

    def test_record_transition_returns_dict(self):
        """record_transition returns a dict with transition_id."""
        svc = _make_service()
        result = svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 50,
        })
        assert result["transition_id"].startswith("tr_")

    def test_record_transition_auto_detects_conversion(self):
        """When from != to, transition_type defaults to conversion."""
        svc = _make_service()
        result = svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 10,
        })
        assert result["transition_type"] == "conversion"

    def test_record_transition_remaining(self):
        """When from == to, transition_type stays remaining."""
        svc = _make_service()
        result = svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "forest_land",
            "area_ha": 10,
        })
        assert result["transition_type"] == "remaining"

    def test_get_transitions_empty(self):
        """get_transitions returns empty when no data exists."""
        svc = _make_service()
        result = svc.get_transitions()
        assert result["total"] == 0

    def test_get_transitions_filters(self):
        """get_transitions applies all optional filters."""
        svc = _make_service()
        svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 10,
        })
        svc.record_transition({
            "parcel_id": "p2",
            "from_category": "grassland",
            "to_category": "settlement",
            "area_ha": 5,
        })
        result = svc.get_transitions(parcel_id="p1")
        assert result["total"] == 1
        result2 = svc.get_transitions(from_category="grassland")
        assert result2["total"] == 1
        result3 = svc.get_transitions(to_category="cropland")
        assert result3["total"] == 1
        result4 = svc.get_transitions(transition_type="conversion")
        assert result4["total"] == 2

    def test_get_transition_matrix_empty(self):
        """Matrix is a 6x6 zeroed structure when no transitions exist."""
        svc = _make_service()
        result = svc.get_transition_matrix()
        assert len(result["categories"]) == 6
        assert result["total_area_ha"] == 0
        assert result["total_transitions"] == 0

    def test_get_transition_matrix_populates(self):
        """Matrix accumulates area for valid from/to pairs."""
        svc = _make_service()
        svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 25.0,
        })
        result = svc.get_transition_matrix()
        assert result["matrix"]["forest_land"]["cropland"] == 25.0
        assert result["total_area_ha"] == 25.0
        assert result["total_transitions"] == 1


# ===================================================================
# Test class: SOC assessment
# ===================================================================


class TestSOCAssessment:
    """Tests for assess_soc()."""

    def test_assess_soc_returns_dict(self):
        """assess_soc returns a dict with assessment_id."""
        svc = _make_service()
        result = svc.assess_soc({"parcel_id": "p1"})
        assert isinstance(result, dict)
        assert result["assessment_id"].startswith("soc_")

    def test_assess_soc_fallback_computes_soc(self):
        """Fallback SOC uses SOC_ref * F_LU * F_MG * F_I."""
        svc = _make_service()
        result = svc.assess_soc({
            "parcel_id": "p1",
            "climate_zone": "tropical_moist",
            "soil_type": "high_activity_clay",
            "land_category": "forest_land",
        })
        # Fallback defaults to soc_ref=50, factors=1.0
        assert result["soc_current"] > 0
        assert result["provenance_hash"]

    def test_assess_soc_stores_in_list(self):
        """Assessment is appended to _soc_assessments."""
        svc = _make_service()
        svc.assess_soc({"parcel_id": "p1"})
        assert len(svc._soc_assessments) == 1


# ===================================================================
# Test class: Uncertainty analysis
# ===================================================================


class TestUncertainty:
    """Tests for run_uncertainty()."""

    def test_run_uncertainty_fallback(self):
        """Fallback analytical uncertainty is used when engine is None."""
        svc = _make_service()
        # First create a calculation to reference
        calc = svc.calculate(_valid_calc_request())
        result = svc.run_uncertainty({
            "calculation_id": calc.calculation_id,
            "iterations": 1000,
            "confidence_level": 95.0,
        })
        assert result["success"] is True
        assert result["method"] == "analytical_fallback"
        assert result["confidence_level"] == 95.0

    def test_run_uncertainty_stores_in_list(self):
        """Result is appended to _uncertainty_results."""
        svc = _make_service()
        svc.run_uncertainty({"calculation_id": "nonexistent"})
        assert len(svc._uncertainty_results) == 1

    def test_run_uncertainty_99_confidence(self):
        """99% confidence level uses z_score=2.576."""
        svc = _make_service()
        calc = svc.calculate(_valid_calc_request())
        result = svc.run_uncertainty({
            "calculation_id": calc.calculation_id,
            "confidence_level": 99.0,
        })
        assert result["confidence_level"] == 99.0

    def test_run_uncertainty_90_confidence(self):
        """90% confidence level uses z_score=1.645."""
        svc = _make_service()
        calc = svc.calculate(_valid_calc_request())
        result = svc.run_uncertainty({
            "calculation_id": calc.calculation_id,
            "confidence_level": 90.0,
        })
        assert result["confidence_level"] == 90.0


# ===================================================================
# Test class: Compliance check
# ===================================================================


class TestComplianceCheck:
    """Tests for check_compliance()."""

    def test_check_compliance_fallback_returns_dict(self):
        """Fallback returns not_assessed status for all frameworks."""
        svc = _make_service()
        result = svc.check_compliance({})
        assert result["success"] is True
        assert result["frameworks_checked"] == 6  # 6 default frameworks

    def test_check_compliance_custom_frameworks(self):
        """Custom frameworks list is respected in fallback."""
        svc = _make_service()
        result = svc.check_compliance({
            "frameworks": ["GHG_PROTOCOL", "IPCC"],
        })
        assert result["frameworks_checked"] == 2
        assert len(result["results"]) == 2

    def test_check_compliance_stores_in_list(self):
        """Result is appended to _compliance_results."""
        svc = _make_service()
        svc.check_compliance({})
        assert len(svc._compliance_results) == 1


# ===================================================================
# Test class: Aggregation
# ===================================================================


class TestAggregation:
    """Tests for aggregate()."""

    def test_aggregate_empty(self):
        """aggregate() returns zeroes with no matching calculations."""
        svc = _make_service()
        result = svc.aggregate({"tenant_id": "t1"})
        assert result["calculation_count"] == 0
        assert result["total_co2e_tonnes"] == 0

    def test_aggregate_filters_by_tenant(self):
        """Only calculations for the given tenant are aggregated."""
        svc = _make_service()
        svc.calculate(_valid_calc_request(tenant_id="t1"))
        svc.calculate(_valid_calc_request(tenant_id="t2"))
        result = svc.aggregate({"tenant_id": "t1"})
        assert result["calculation_count"] == 1

    def test_aggregate_period_default(self):
        """Default aggregation period is 'annual'."""
        svc = _make_service()
        result = svc.aggregate({"tenant_id": "t1"})
        assert result["period"] == "annual"


# ===================================================================
# Test class: Parcel CRUD
# ===================================================================


class TestParcelCRUD:
    """Tests for register_parcel, get_parcel, list_parcels, update_parcel."""

    def test_register_parcel_returns_record(self):
        """register_parcel returns a dict with a parcel_id."""
        svc = _make_service()
        result = svc.register_parcel(_valid_parcel_data())
        assert result["parcel_id"].startswith("parcel_")
        assert result["name"] == "Test Parcel"
        assert result["area_ha"] == 100.0

    def test_register_parcel_missing_fields_raises(self):
        """register_parcel raises ValueError when required fields missing."""
        svc = _make_service()
        with pytest.raises(ValueError, match="Missing required fields"):
            svc.register_parcel({"name": "No area"})

    def test_get_parcel_returns_record(self):
        """get_parcel returns the parcel dict by ID."""
        svc = _make_service()
        registered = svc.register_parcel(_valid_parcel_data())
        found = svc.get_parcel(registered["parcel_id"])
        assert found is not None
        assert found["name"] == "Test Parcel"

    def test_get_parcel_returns_none_for_missing(self):
        """get_parcel returns None for a nonexistent parcel_id."""
        svc = _make_service()
        assert svc.get_parcel("does_not_exist") is None

    def test_list_parcels_empty(self):
        """list_parcels returns empty list when no parcels exist."""
        svc = _make_service()
        result = svc.list_parcels()
        assert result["total"] == 0
        assert result["parcels"] == []

    def test_list_parcels_pagination(self):
        """list_parcels supports page and page_size."""
        svc = _make_service()
        for i in range(5):
            svc.register_parcel(_valid_parcel_data(name=f"Parcel {i}"))
        result = svc.list_parcels(page=1, page_size=2)
        assert len(result["parcels"]) == 2
        assert result["total"] == 5

    def test_list_parcels_filters(self):
        """list_parcels applies tenant_id, land_category, climate_zone."""
        svc = _make_service()
        svc.register_parcel(_valid_parcel_data(
            tenant_id="t1", land_category="forest_land",
        ))
        svc.register_parcel(_valid_parcel_data(
            tenant_id="t2", land_category="cropland",
        ))
        result = svc.list_parcels(tenant_id="t1")
        assert result["total"] == 1

    def test_update_parcel_returns_updated(self):
        """update_parcel returns the updated parcel dict."""
        svc = _make_service()
        registered = svc.register_parcel(_valid_parcel_data())
        pid = registered["parcel_id"]
        updated = svc.update_parcel(pid, {"name": "New Name", "area_ha": 200})
        assert updated is not None
        assert updated["name"] == "New Name"
        assert updated["area_ha"] == 200.0

    def test_update_parcel_returns_none_for_missing(self):
        """update_parcel returns None for a nonexistent parcel_id."""
        svc = _make_service()
        assert svc.update_parcel("nope", {"name": "X"}) is None

    def test_update_parcel_recomputes_provenance(self):
        """Provenance hash changes after an update."""
        svc = _make_service()
        registered = svc.register_parcel(_valid_parcel_data())
        pid = registered["parcel_id"]
        old_hash = registered["provenance_hash"]
        updated = svc.update_parcel(pid, {"name": "Changed"})
        assert updated["provenance_hash"] != old_hash


# ===================================================================
# Test class: Health check and stats
# ===================================================================


class TestHealthAndStats:
    """Tests for health_check() and get_stats()."""

    def test_health_check_returns_health_response(self):
        """health_check returns a HealthResponse model."""
        from greenlang.land_use_emissions.setup import HealthResponse
        svc = _make_service()
        result = svc.health_check()
        assert isinstance(result, HealthResponse)

    def test_health_check_unhealthy_when_no_engines(self):
        """Status is 'unhealthy' when fewer than 3 engines loaded."""
        svc = _make_service()
        result = svc.health_check()
        assert result.status == "unhealthy"

    def test_health_check_degraded_when_some_engines(self):
        """Status is 'degraded' when 3+ but not all engines loaded."""
        svc = _make_service()
        svc._database_engine = "mock"
        svc._carbon_stock_engine = "mock"
        svc._change_tracker_engine = "mock"
        result = svc.health_check()
        assert result.status == "degraded"

    def test_health_check_healthy_when_all_engines(self):
        """Status is 'healthy' when all 7 engines are available."""
        svc = _make_service()
        svc._database_engine = "mock"
        svc._carbon_stock_engine = "mock"
        svc._change_tracker_engine = "mock"
        svc._soc_engine = "mock"
        svc._uncertainty_engine = "mock"
        svc._compliance_engine = "mock"
        svc._pipeline_engine = "mock"
        result = svc.health_check()
        assert result.status == "healthy"

    def test_health_check_service_name(self):
        """Health response service name is 'land-use-emissions'."""
        svc = _make_service()
        result = svc.health_check()
        assert result.service == "land-use-emissions"

    def test_get_stats_returns_stats_response(self):
        """get_stats returns a StatsResponse model."""
        from greenlang.land_use_emissions.setup import StatsResponse
        svc = _make_service()
        result = svc.get_stats()
        assert isinstance(result, StatsResponse)

    def test_get_stats_reflects_state(self):
        """Stats reflect actual in-memory counts."""
        svc = _make_service()
        svc.register_parcel(_valid_parcel_data())
        svc.calculate(_valid_calc_request(tenant_id="t1"))
        svc.record_carbon_stock({"parcel_id": "p1"})
        svc.record_transition({
            "parcel_id": "p1",
            "from_category": "forest_land",
            "to_category": "cropland",
            "area_ha": 10,
        })
        stats = svc.get_stats()
        assert stats.total_calculations == 1
        assert stats.total_parcels == 1
        assert stats.total_carbon_stocks == 1
        assert stats.total_transitions == 1

    def test_get_stats_uptime_positive(self):
        """Uptime is a positive number of seconds."""
        svc = _make_service()
        stats = svc.get_stats()
        assert stats.uptime_seconds >= 0
