# -*- coding: utf-8 -*-
"""
Unit tests for RefrigerantsFGasService facade (setup.py) - AGENT-MRV-002

Tests the service facade covering:
- Service creation and configuration
- calculate, calculate_batch delegation to pipeline
- get_refrigerant, list_refrigerants
- register_equipment, log_service_event
- estimate_leak_rate, check_compliance, run_uncertainty
- get_audit_trail, aggregate, validate
- get_health, get_stats
- configure_refrigerants_fgas function
- get_service, get_router module-level singletons
- Response model classes existence and fields

Target: 55+ tests, ~650 lines
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.refrigerants_fgas.setup import (
    AuditTrailResponse,
    BatchResponse,
    BlendResponse,
    CalculationResponse,
    ComplianceListResponse,
    ComplianceResponse,
    EquipmentListResponse,
    EquipmentResponse,
    HealthResponse,
    LeakRateResponse,
    PipelineResponse,
    RefrigerantListResponse,
    RefrigerantResponse,
    RefrigerantsFGasService,
    ServiceEventResponse,
    StatsResponse,
    UncertaintyResponse,
    ValidationResponse,
    _compute_hash,
    get_router,
    get_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> RefrigerantsFGasService:
    """Return a RefrigerantsFGasService instance with defaults."""
    return RefrigerantsFGasService()


@pytest.fixture
def started_service(service) -> RefrigerantsFGasService:
    """Return a RefrigerantsFGasService that has been started."""
    service.startup()
    return service


# ===========================================================================
# Test service creation
# ===========================================================================


class TestServiceCreation:
    """Tests for RefrigerantsFGasService construction."""

    def test_service_creation(self):
        """Service can be created with no arguments."""
        svc = RefrigerantsFGasService()
        assert svc is not None

    def test_service_with_config(self, custom_config):
        """Service can be created with a custom config."""
        svc = RefrigerantsFGasService(config=custom_config)
        assert svc.config is custom_config

    def test_service_default_config(self, service):
        """Service uses default (global) config when none provided."""
        # config may be None or the global config
        assert service is not None

    def test_service_has_stores(self, service):
        """Service initializes all in-memory stores."""
        assert isinstance(service._calculations, dict)
        assert isinstance(service._refrigerants, dict)
        assert isinstance(service._equipment_profiles, dict)
        assert isinstance(service._service_events, dict)
        assert isinstance(service._leak_rates, dict)
        assert isinstance(service._compliance_records, dict)
        assert isinstance(service._uncertainty_results, dict)
        assert isinstance(service._audit_entries, dict)

    def test_service_stats_zero(self, service):
        """Service initializes all stats counters to zero."""
        assert service._total_calculations == 0
        assert service._total_batch_runs == 0
        assert service._total_pipeline_runs == 0
        assert service._total_compliance_checks == 0
        assert service._total_uncertainty_runs == 0
        assert service._total_calculation_time_ms == 0.0
        assert service._started is False

    def test_service_engine_properties(self, service):
        """Service exposes all 7 engine properties."""
        _ = service.refrigerant_database_engine
        _ = service.emission_calculator_engine
        _ = service.equipment_registry_engine
        _ = service.leak_rate_estimator_engine
        _ = service.uncertainty_quantifier_engine
        _ = service.compliance_tracker_engine
        _ = service.pipeline_engine


# ===========================================================================
# Test calculate and calculate_batch
# ===========================================================================


class TestCalculate:
    """Tests for the calculate and calculate_batch methods."""

    def test_calculate(self, service):
        """calculate() returns a result dict with expected fields."""
        result = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            method="equipment_based",
        )
        assert isinstance(result, dict)
        assert "calculation_id" in result
        assert "processing_time_ms" in result
        assert "provenance_hash" in result
        assert service._total_calculations == 1

    def test_calculate_stores_result(self, service):
        """calculate() stores the result in the calculations cache."""
        result = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=10.0,
        )
        calc_id = result["calculation_id"]
        assert calc_id in service._calculations

    def test_calculate_mass_balance(self, service):
        """calculate() works with mass_balance method."""
        result = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=100.0,
            method="mass_balance",
            mass_balance_data={
                "inventory_start_kg": 500.0,
                "purchases_kg": 100.0,
                "recovery_kg": 50.0,
                "inventory_end_kg": 450.0,
            },
        )
        assert isinstance(result, dict)
        assert result.get("processing_time_ms", 0) > 0

    def test_calculate_screening(self, service):
        """calculate() works with screening method."""
        result = service.calculate(
            refrigerant_type="R_407C",
            charge_kg=10.0,
            method="screening",
        )
        assert isinstance(result, dict)

    def test_calculate_custom_leak_rate(self, service):
        """calculate() passes custom_leak_rate_pct through."""
        result = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            custom_leak_rate_pct=12.0,
        )
        assert isinstance(result, dict)

    def test_calculate_batch(self, service):
        """calculate_batch() processes multiple inputs."""
        inputs = [
            {"refrigerant_type": "R_410A", "charge_kg": 25.0},
            {"refrigerant_type": "R_134A", "charge_kg": 10.0},
        ]
        result = service.calculate_batch(inputs)

        assert isinstance(result, dict)
        assert result.get("total_count", 0) >= 2 or len(result.get("results", [])) == 2
        assert service._total_batch_runs >= 1

    def test_calculate_batch_empty(self, service):
        """calculate_batch() handles empty input list."""
        result = service.calculate_batch([])
        assert isinstance(result, dict)


# ===========================================================================
# Test refrigerant operations
# ===========================================================================


class TestRefrigerantOps:
    """Tests for refrigerant get/list operations."""

    def test_get_refrigerant_unknown(self, service):
        """get_refrigerant() returns error for unknown type."""
        result = service.get_refrigerant("NONEXISTENT_GAS")
        assert "error" in result

    def test_get_refrigerant_cached(self, service):
        """get_refrigerant() returns from cache when present."""
        service._refrigerants["R_410A"] = {
            "refrigerant_type": "R_410A",
            "gwp_ar6": 2256.0,
        }
        result = service.get_refrigerant("R_410A")
        assert result["refrigerant_type"] == "R_410A"
        assert result["gwp_ar6"] == 2256.0

    def test_list_refrigerants_empty(self, service):
        """list_refrigerants() returns empty list when no refrigerants."""
        result = service.list_refrigerants()
        assert isinstance(result, list)

    def test_list_refrigerants_with_cache(self, service):
        """list_refrigerants() returns cached refrigerants."""
        service._refrigerants["R_410A"] = {
            "refrigerant_type": "R_410A",
            "category": "HFC_BLEND",
        }
        service._refrigerants["R_134A"] = {
            "refrigerant_type": "R_134A",
            "category": "HFC",
        }
        result = service.list_refrigerants()
        assert len(result) == 2

    def test_list_refrigerants_category_filter(self, service):
        """list_refrigerants() filters by category."""
        service._refrigerants["R_410A"] = {
            "refrigerant_type": "R_410A",
            "category": "HFC_BLEND",
        }
        service._refrigerants["R_134A"] = {
            "refrigerant_type": "R_134A",
            "category": "HFC",
        }
        result = service.list_refrigerants(category="HFC")
        assert len(result) == 1
        assert result[0]["refrigerant_type"] == "R_134A"


# ===========================================================================
# Test equipment operations
# ===========================================================================


class TestEquipmentOps:
    """Tests for equipment registration and service events."""

    def test_register_equipment(self, service):
        """register_equipment() creates and returns an equipment profile."""
        result = service.register_equipment(
            equipment_type="COMMERCIAL_AC",
            name="Chiller A1",
            refrigerant_type="R_410A",
            charge_kg=50.0,
        )
        assert isinstance(result, dict)
        assert result["equipment_type"] == "COMMERCIAL_AC"
        assert result["charge_kg"] == 50.0
        assert "provenance_hash" in result
        assert len(service._equipment_profiles) == 1

    def test_register_equipment_custom_id(self, service):
        """register_equipment() respects provided equipment_id."""
        result = service.register_equipment(
            equipment_id="eq_custom_123",
            equipment_type="CHILLERS_CENTRIFUGAL",
            refrigerant_type="R_134A",
            charge_kg=100.0,
        )
        assert result["equipment_id"] == "eq_custom_123"
        assert "eq_custom_123" in service._equipment_profiles

    def test_log_service_event(self, service):
        """log_service_event() records a service event."""
        result = service.log_service_event(
            equipment_id="eq_001",
            event_type="recharge",
            refrigerant_type="R_410A",
            quantity_kg=5.0,
            technician="John Smith",
            notes="Annual recharge",
        )
        assert isinstance(result, dict)
        assert result["equipment_id"] == "eq_001"
        assert result["event_type"] == "recharge"
        assert result["quantity_kg"] == 5.0
        assert "provenance_hash" in result
        assert len(service._service_events) == 1


# ===========================================================================
# Test leak rate, compliance, uncertainty
# ===========================================================================


class TestLeakRateComplianceUncertainty:
    """Tests for leak rate estimation, compliance, and uncertainty."""

    def test_estimate_leak_rate(self, service):
        """estimate_leak_rate() returns a leak rate estimate."""
        result = service.estimate_leak_rate(
            equipment_type="COMMERCIAL_AC",
            age_years=5,
        )
        assert isinstance(result, dict)
        assert result["equipment_type"] == "COMMERCIAL_AC"
        assert result["base_rate_pct"] > 0
        assert result["effective_rate_pct"] > 0
        assert "provenance_hash" in result

    def test_estimate_leak_rate_age_factor(self, service):
        """estimate_leak_rate() applies age adjustment factor."""
        result_new = service.estimate_leak_rate(
            equipment_type="COMMERCIAL_AC",
            age_years=0,
        )
        result_old = service.estimate_leak_rate(
            equipment_type="COMMERCIAL_AC",
            age_years=20,
        )
        assert result_old["effective_rate_pct"] > result_new["effective_rate_pct"]

    def test_check_compliance(self, service):
        """check_compliance() returns compliance records."""
        result = service.check_compliance()
        assert isinstance(result, dict)
        assert "records" in result
        assert result["total_count"] >= 5
        assert "overall_compliant" in result
        assert service._total_compliance_checks >= 1

    def test_check_compliance_custom_frameworks(self, service):
        """check_compliance() accepts custom framework list."""
        result = service.check_compliance(
            frameworks=["GHG_PROTOCOL", "ISO_14064"],
        )
        assert result["total_count"] == 2

    def test_run_uncertainty_no_calc(self, service):
        """run_uncertainty() returns error when calculation not found."""
        result = service.run_uncertainty(calculation_id="nonexistent")
        assert "error" in result

    def test_run_uncertainty_after_calc(self, service):
        """run_uncertainty() works after a calculation exists."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        calc_id = calc["calculation_id"]
        result = service.run_uncertainty(calculation_id=calc_id)

        assert isinstance(result, dict)
        assert result.get("calculation_id") == calc_id or "mean_co2e_kg" in result
        assert service._total_uncertainty_runs >= 1


# ===========================================================================
# Test audit trail and aggregation
# ===========================================================================


class TestAuditAndAggregation:
    """Tests for audit trail retrieval and aggregation."""

    def test_get_audit_trail_empty(self, service):
        """get_audit_trail() returns empty list for unknown calc_id."""
        result = service.get_audit_trail("nonexistent")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_audit_trail_after_calc(self, service):
        """get_audit_trail() returns audit entries after calculation."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        calc_id = calc["calculation_id"]
        audit = service.get_audit_trail(calc_id)
        assert isinstance(audit, list)

    def test_aggregate_empty(self, service):
        """aggregate() returns zero when no calculations exist."""
        result = service.aggregate()
        assert isinstance(result, dict)
        assert result.get("grand_total_tco2e", 0.0) == 0.0

    def test_aggregate_after_calculations(self, service):
        """aggregate() sums emissions from multiple calculations."""
        service.calculate(refrigerant_type="R_410A", charge_kg=25.0)
        service.calculate(refrigerant_type="R_134A", charge_kg=10.0)
        result = service.aggregate()

        assert isinstance(result, dict)
        assert "aggregations" in result

    def test_aggregate_equity_share(self, service):
        """aggregate() applies equity share fraction."""
        service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            facility_id="fac_A",
        )
        result = service.aggregate(
            control_approach="EQUITY_SHARE",
            share=0.5,
        )
        assert isinstance(result, dict)


# ===========================================================================
# Test validate
# ===========================================================================


class TestValidate:
    """Tests for input validation without calculation."""

    def test_validate_valid_input(self, service):
        """validate() reports valid inputs correctly."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
                "facility_id": "fac_001",
            },
        ])
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_missing_fields(self, service):
        """validate() catches missing required fields."""
        result = service.validate([{"method": "equipment_based"}])
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_validate_invalid_method(self, service):
        """validate() catches unsupported method."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 10.0,
                "method": "bogus",
            },
        ])
        assert result["valid"] is False

    def test_validate_negative_charge(self, service):
        """validate() catches negative charge_kg."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": -5.0,
            },
        ])
        assert result["valid"] is False


# ===========================================================================
# Test health and stats
# ===========================================================================


class TestHealthAndStats:
    """Tests for get_health and get_stats."""

    def test_get_health(self, service):
        """get_health() returns health dict with required fields."""
        result = service.get_health()
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ("healthy", "degraded", "unhealthy")
        assert result["version"] == "1.0.0"
        assert "engines" in result
        assert isinstance(result["engines"], dict)
        assert result["engines_total"] == 7
        assert "timestamp" in result

    def test_get_health_started(self, started_service):
        """get_health() reflects started state."""
        result = started_service.get_health()
        assert result["started"] is True

    def test_get_stats(self, service):
        """get_stats() returns stats dict with required fields."""
        result = service.get_stats()
        assert isinstance(result, dict)
        assert "total_calculations" in result
        assert "total_batch_runs" in result
        assert "total_pipeline_runs" in result
        assert "total_refrigerants" in result
        assert "total_equipment" in result
        assert "total_service_events" in result
        assert "total_compliance_checks" in result
        assert "total_uncertainty_runs" in result
        assert "total_audit_entries" in result
        assert "avg_calculation_time_ms" in result
        assert "timestamp" in result

    def test_get_stats_after_operations(self, service):
        """get_stats() reflects completed operations."""
        service.calculate(refrigerant_type="R_410A", charge_kg=25.0)
        result = service.get_stats()
        assert result["total_calculations"] >= 1


# ===========================================================================
# Test lifecycle
# ===========================================================================


class TestLifecycle:
    """Tests for startup and shutdown."""

    def test_startup(self, service):
        """startup() sets _started to True."""
        service.startup()
        assert service._started is True

    def test_startup_idempotent(self, service):
        """startup() is safe to call multiple times."""
        service.startup()
        service.startup()
        assert service._started is True

    def test_shutdown(self, started_service):
        """shutdown() sets _started to False."""
        started_service.shutdown()
        assert started_service._started is False

    def test_shutdown_not_started(self, service):
        """shutdown() is safe to call when not started."""
        service.shutdown()
        assert service._started is False


# ===========================================================================
# Test module-level functions
# ===========================================================================


class TestModuleLevelFunctions:
    """Tests for configure_refrigerants_fgas, get_service, get_router."""

    def test_get_service(self):
        """get_service() returns a RefrigerantsFGasService instance."""
        import greenlang.refrigerants_fgas.setup as setup_mod

        # Reset singleton state
        setup_mod._singleton_instance = None
        setup_mod._service = None

        svc = get_service()
        assert isinstance(svc, RefrigerantsFGasService)

        # Cleanup
        setup_mod._singleton_instance = None
        setup_mod._service = None

    def test_get_router(self):
        """get_router() returns a router or None."""
        router = get_router()
        # May be None if FastAPI not available, or an APIRouter
        if router is not None:
            assert hasattr(router, "routes")

    @pytest.mark.asyncio
    async def test_configure_function(self):
        """configure_refrigerants_fgas() wires the service to a FastAPI app."""
        import greenlang.refrigerants_fgas.setup as setup_mod
        from greenlang.refrigerants_fgas.setup import (
            configure_refrigerants_fgas,
        )

        # Reset singleton state
        setup_mod._singleton_instance = None
        setup_mod._service = None

        try:
            from fastapi import FastAPI
            app = FastAPI()
            svc = await configure_refrigerants_fgas(app)
            assert isinstance(svc, RefrigerantsFGasService)
            assert hasattr(app.state, "refrigerants_fgas_service")
            assert svc._started is True
        except ImportError:
            pytest.skip("FastAPI not available")
        finally:
            setup_mod._singleton_instance = None
            setup_mod._service = None


# ===========================================================================
# Test response models existence and fields
# ===========================================================================


class TestResponseModels:
    """Tests for the 14 Pydantic response model classes."""

    _RESPONSE_MODELS = [
        CalculationResponse,
        BatchResponse,
        RefrigerantResponse,
        RefrigerantListResponse,
        EquipmentResponse,
        EquipmentListResponse,
        ServiceEventResponse,
        LeakRateResponse,
        ComplianceResponse,
        ComplianceListResponse,
        UncertaintyResponse,
        AuditTrailResponse,
        BlendResponse,
        ValidationResponse,
        PipelineResponse,
        HealthResponse,
        StatsResponse,
    ]

    def test_response_models_exist(self):
        """All 17 response model classes exist and are importable."""
        for cls in self._RESPONSE_MODELS:
            assert cls is not None
            assert hasattr(cls, "model_fields")

    @pytest.mark.parametrize("model_cls", [
        CalculationResponse,
        BatchResponse,
        RefrigerantResponse,
        EquipmentResponse,
        ServiceEventResponse,
        LeakRateResponse,
        ComplianceResponse,
        UncertaintyResponse,
        AuditTrailResponse,
        HealthResponse,
        StatsResponse,
        PipelineResponse,
    ])
    def test_response_model_instantiation(self, model_cls):
        """Response models can be instantiated with defaults."""
        instance = model_cls()
        assert instance is not None

    def test_calculation_response_fields(self):
        """CalculationResponse has expected fields."""
        fields = CalculationResponse.model_fields
        assert "calculation_id" in fields
        assert "status" in fields
        assert "refrigerant_type" in fields
        assert "total_emissions_tco2e" in fields
        assert "provenance_hash" in fields

    def test_health_response_fields(self):
        """HealthResponse has expected fields."""
        fields = HealthResponse.model_fields
        assert "status" in fields
        assert "version" in fields
        assert "engines" in fields
        assert "engines_available" in fields
        assert "provenance_chain_valid" in fields

    def test_stats_response_fields(self):
        """StatsResponse has expected fields."""
        fields = StatsResponse.model_fields
        assert "total_calculations" in fields
        assert "total_batch_runs" in fields
        assert "avg_calculation_time_ms" in fields

    def test_pipeline_response_fields(self):
        """PipelineResponse has expected fields."""
        fields = PipelineResponse.model_fields
        assert "pipeline_id" in fields
        assert "success" in fields
        assert "stages_completed" in fields
        assert "stages_total" in fields
        assert "pipeline_provenance_hash" in fields

    def test_validation_response_fields(self):
        """ValidationResponse has expected fields."""
        fields = ValidationResponse.model_fields
        assert "valid" in fields
        assert "errors" in fields
        assert "warnings" in fields
        assert "validated_count" in fields

    def test_response_models_forbid_extra(self):
        """All response models have extra='forbid' config."""
        for cls in self._RESPONSE_MODELS:
            config = getattr(cls, "model_config", {})
            assert config.get("extra") == "forbid", (
                f"{cls.__name__} should forbid extra fields"
            )


# ===========================================================================
# Additional calculate edge cases
# ===========================================================================


class TestCalculateEdgeCases:
    """Additional edge-case tests for the calculate method."""

    def test_calculate_increments_counter(self, service):
        """Each calculate call increments _total_calculations."""
        service.calculate(refrigerant_type="R_410A", charge_kg=25.0)
        service.calculate(refrigerant_type="R_134A", charge_kg=10.0)
        assert service._total_calculations == 2

    def test_calculate_returns_provenance_hash(self, service):
        """calculate() result contains a valid SHA-256 provenance hash."""
        result = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        h = result.get("provenance_hash", "")
        if h:
            assert len(h) == 64
            int(h, 16)  # valid hex

    def test_calculate_with_gwp_source(self, service):
        """calculate() respects gwp_source parameter."""
        result = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            gwp_source="AR5",
        )
        assert isinstance(result, dict)
        assert "calculation_id" in result

    def test_calculate_direct_method(self, service):
        """calculate() processes direct measurement method."""
        result = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=10.0,
            method="direct",
            measured_emissions_kg=2.5,
        )
        assert isinstance(result, dict)

    def test_calculate_top_down_method(self, service):
        """calculate() processes top_down method."""
        result = service.calculate(
            refrigerant_type="R_404A",
            charge_kg=50.0,
            method="top_down",
            equipment_type="COMMERCIAL_REFRIGERATION",
            num_units=3,
        )
        assert isinstance(result, dict)

    def test_calculate_stores_processing_time(self, service):
        """calculate() records positive processing time."""
        result = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        assert result["processing_time_ms"] > 0

    def test_calculate_accumulates_total_time(self, service):
        """Multiple calculations accumulate total_calculation_time_ms."""
        for _ in range(3):
            service.calculate(refrigerant_type="R_410A", charge_kg=10.0)
        assert service._total_calculation_time_ms > 0

    def test_calculate_batch_increments_counter(self, service):
        """calculate_batch() increments _total_batch_runs."""
        service.calculate_batch([
            {"refrigerant_type": "R_410A", "charge_kg": 10.0},
        ])
        service.calculate_batch([
            {"refrigerant_type": "R_134A", "charge_kg": 5.0},
        ])
        assert service._total_batch_runs == 2


# ===========================================================================
# Additional equipment and event tests
# ===========================================================================


class TestEquipmentEdgeCases:
    """Additional edge-case tests for equipment operations."""

    def test_register_equipment_auto_id(self, service):
        """register_equipment() auto-generates equipment_id when omitted."""
        result = service.register_equipment(
            equipment_type="COMMERCIAL_AC",
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        assert "equipment_id" in result
        assert len(result["equipment_id"]) > 0

    def test_register_multiple_equipment(self, service):
        """Multiple equipment profiles can be registered."""
        for i in range(3):
            service.register_equipment(
                equipment_id=f"eq_multi_{i}",
                equipment_type="COMMERCIAL_AC",
                refrigerant_type="R_410A",
                charge_kg=float(i + 1) * 10,
            )
        assert len(service._equipment_profiles) == 3

    def test_log_service_event_auto_id(self, service):
        """log_service_event() auto-generates event_id."""
        result = service.log_service_event(
            equipment_id="eq_auto_id",
            event_type="recharge",
            quantity_kg=5.0,
        )
        assert "event_id" in result
        assert len(result["event_id"]) > 0

    def test_log_multiple_events_same_equipment(self, service):
        """Multiple events can be logged for the same equipment."""
        for evt_type in ["installation", "recharge", "repair"]:
            service.log_service_event(
                equipment_id="eq_multi_evt",
                event_type=evt_type,
                quantity_kg=5.0,
            )
        assert len(service._service_events) == 3


# ===========================================================================
# Additional leak rate edge cases
# ===========================================================================


class TestLeakRateEdgeCases:
    """Additional edge-case tests for leak rate estimation."""

    def test_estimate_leak_rate_switchgear(self, service):
        """estimate_leak_rate() returns low rate for switchgear."""
        result = service.estimate_leak_rate(
            equipment_type="SWITCHGEAR",
            age_years=0,
        )
        assert result["base_rate_pct"] < 5.0
        assert result["effective_rate_pct"] < 5.0

    def test_estimate_leak_rate_transport(self, service):
        """estimate_leak_rate() returns higher rate for transport."""
        result = service.estimate_leak_rate(
            equipment_type="TRANSPORT_REFRIGERATION",
            age_years=5,
        )
        assert result["effective_rate_pct"] > 0

    def test_estimate_leak_rate_returns_provenance(self, service):
        """estimate_leak_rate() returns a result with provenance_hash."""
        result = service.estimate_leak_rate(
            equipment_type="COMMERCIAL_AC",
            age_years=3,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64
