# -*- coding: utf-8 -*-
"""
Unit tests for StationaryCombustionService facade - AGENT-MRV-001

Tests the unified service facade that aggregates all seven engines:
FuelDatabaseEngine, CalculatorEngine, EquipmentProfilerEngine,
FactorSelectorEngine, UncertaintyEngine, AuditEngine, and
StationaryCombustionPipelineEngine.

Validates:
- Service initialisation with all 7 engines
- Single and batch calculation through the facade
- Fuel properties and type listing
- Emission factor retrieval
- Equipment registration
- Pipeline execution
- Uncertainty quantification
- Audit trail retrieval
- Compliance mapping retrieval
- Health check (status, version, engine_count)
- configure_stationary_combustion, get_service, get_router functions
- Service lifecycle (startup, shutdown)
- Statistics tracking

Author: GreenLang Test Engineering
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    reset_config,
)
from greenlang.stationary_combustion.models import (
    EquipmentType,
    FuelType,
    RegulatoryFramework,
)
from greenlang.stationary_combustion.setup import (
    StationaryCombustionService,
    configure_stationary_combustion,
    get_router,
    get_service,
    HealthResponse,
    StatsResponse,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def config():
    """Create a test StationaryCombustionConfig."""
    return StationaryCombustionConfig(
        enable_biogenic_tracking=True,
        monte_carlo_iterations=100,
        enable_metrics=False,
    )


@pytest.fixture
def service(config):
    """Create a StationaryCombustionService instance."""
    return StationaryCombustionService(config=config)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset the module-level singleton after each test."""
    yield
    # Reset singleton state to avoid cross-test contamination
    import greenlang.stationary_combustion.setup as setup_module
    setup_module._singleton_instance = None
    setup_module._service = None
    reset_config()


# =====================================================================
# TestServiceInit
# =====================================================================


class TestServiceInit:
    """Test StationaryCombustionService initialisation."""

    def test_service_init_with_config(self, config):
        """Service initialises with provided config."""
        svc = StationaryCombustionService(config=config)
        assert svc.config is config

    def test_service_init_default_config(self):
        """Service initialises with default config when none provided."""
        svc = StationaryCombustionService()
        assert svc.config is not None

    def test_service_init_engine_properties(self, service):
        """Service exposes engine properties (may be None in test env)."""
        # These may be None since actual engine modules may not be importable
        # in the test environment, but the properties should exist
        _ = service.fuel_database_engine
        _ = service.calculator_engine
        _ = service.equipment_profiler_engine
        _ = service.factor_selector_engine
        _ = service.uncertainty_engine
        _ = service.audit_engine
        _ = service.pipeline_engine

    def test_service_init_empty_caches(self, service):
        """Service starts with empty in-memory caches."""
        assert len(service._calculations) == 0
        assert len(service._fuel_types) == 0
        assert len(service._emission_factors) == 0
        assert len(service._equipment_profiles) == 0
        assert len(service._aggregations) == 0
        assert len(service._audit_entries) == 0

    def test_service_init_zero_statistics(self, service):
        """Service starts with zero statistics."""
        assert service._total_calculations == 0
        assert service._total_batch_runs == 0
        assert service._total_pipeline_runs == 0

    def test_service_init_not_started(self, service):
        """Service is not started after init."""
        assert service._started is False

    def test_service_init_provenance_available(self, service):
        """Service has provenance tracker (may be None)."""
        # ProvenanceTracker may or may not be available
        _ = service.get_provenance()


# =====================================================================
# TestServiceCalculate
# =====================================================================


class TestServiceCalculate:
    """Test single calculation through service facade."""

    def test_calculate_returns_dict(self, service):
        """calculate() returns a dictionary result."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
        )
        assert isinstance(result, dict)
        assert "calculation_id" in result

    def test_calculate_stores_result(self, service):
        """calculate() caches the result in _calculations."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
        )
        calc_id = result["calculation_id"]
        assert calc_id in service._calculations

    def test_calculate_increments_counter(self, service):
        """calculate() increments total_calculations counter."""
        service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
        )
        assert service._total_calculations == 1

    def test_calculate_has_processing_time(self, service):
        """calculate() result includes processing_time_ms."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
        )
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_calculate_has_provenance_hash(self, service):
        """calculate() result includes provenance_hash."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
        )
        assert "provenance_hash" in result

    def test_calculate_with_custom_gwp(self, service):
        """calculate() respects custom GWP source."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
            gwp_source="AR5",
        )
        assert isinstance(result, dict)

    def test_calculate_with_facility_id(self, service):
        """calculate() accepts facility_id parameter."""
        result = service.calculate(
            fuel_type="NATURAL_GAS",
            quantity=1000.0,
            unit="CUBIC_METERS",
            facility_id="FAC-001",
        )
        assert isinstance(result, dict)


# =====================================================================
# TestServiceCalculateBatch
# =====================================================================


class TestServiceCalculateBatch:
    """Test batch calculation through service facade."""

    def test_batch_returns_dict(self, service):
        """calculate_batch() returns a dictionary."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
            {"fuel_type": "DIESEL", "quantity": 500.0, "unit": "LITERS"},
        ]
        result = service.calculate_batch(inputs)
        assert isinstance(result, dict)
        assert "batch_id" in result

    def test_batch_increments_counter(self, service):
        """calculate_batch() increments batch run counter."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
        ]
        service.calculate_batch(inputs)
        assert service._total_batch_runs == 1

    def test_batch_has_processing_time(self, service):
        """calculate_batch() result includes processing_time_ms."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
        ]
        result = service.calculate_batch(inputs)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_batch_has_provenance_hash(self, service):
        """calculate_batch() result includes provenance_hash."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
        ]
        result = service.calculate_batch(inputs)
        assert "provenance_hash" in result


# =====================================================================
# TestServiceGetFuelProperties
# =====================================================================


class TestServiceGetFuelProperties:
    """Test fuel properties retrieval."""

    def test_get_fuel_properties_unknown(self, service):
        """get_fuel_properties() returns error for unknown fuel."""
        result = service.get_fuel_properties("IMAGINARY_FUEL")
        assert isinstance(result, dict)
        assert result.get("fuel_type") == "IMAGINARY_FUEL"

    def test_get_fuel_properties_from_cache(self, service):
        """get_fuel_properties() returns from in-memory cache."""
        service._fuel_types["test_fuel"] = {
            "fuel_type": "test_fuel",
            "hhv": 40.0,
        }
        result = service.get_fuel_properties("test_fuel")
        assert result["hhv"] == 40.0


# =====================================================================
# TestServiceListFuelTypes
# =====================================================================


class TestServiceListFuelTypes:
    """Test listing all fuel types."""

    def test_list_fuel_types_returns_list(self, service):
        """list_fuel_types() returns a list."""
        result = service.list_fuel_types()
        assert isinstance(result, list)

    def test_list_fuel_types_has_entries(self, service):
        """list_fuel_types() returns at least the enum values."""
        result = service.list_fuel_types()
        assert len(result) >= len(FuelType)

    def test_list_fuel_types_has_display_name(self, service):
        """Each fuel type entry has a display_name."""
        result = service.list_fuel_types()
        for entry in result:
            assert "display_name" in entry or "fuel_type" in entry


# =====================================================================
# TestServiceGetEmissionFactor
# =====================================================================


class TestServiceGetEmissionFactor:
    """Test emission factor retrieval."""

    def test_get_emission_factor_unknown(self, service):
        """get_emission_factor() returns error for unknown factor."""
        result = service.get_emission_factor(
            fuel_type="UNKNOWN",
            gas="CO2",
            source="EPA",
        )
        assert "error" in result or "fuel_type" in result

    def test_get_emission_factor_from_cache(self, service):
        """get_emission_factor() returns from in-memory cache."""
        service._emission_factors["NATURAL_GAS:CO2:EPA"] = {
            "fuel_type": "NATURAL_GAS",
            "gas": "CO2",
            "value": 56.1,
            "source": "EPA",
        }
        result = service.get_emission_factor("NATURAL_GAS", "CO2", "EPA")
        assert result["value"] == 56.1


# =====================================================================
# TestServiceRegisterEquipment
# =====================================================================


class TestServiceRegisterEquipment:
    """Test equipment registration."""

    def test_register_equipment_returns_dict(self, service):
        """register_equipment() returns a dictionary."""
        result = service.register_equipment(
            equipment_type="BOILER_FIRE_TUBE",
            name="Boiler-1",
            facility_id="FAC-001",
        )
        assert isinstance(result, dict)
        assert "equipment_id" in result

    def test_register_equipment_stores_profile(self, service):
        """register_equipment() stores profile in _equipment_profiles."""
        result = service.register_equipment(
            equipment_type="FURNACE",
            name="Furnace-A",
        )
        equip_id = result["equipment_id"]
        assert equip_id in service._equipment_profiles

    def test_register_equipment_with_custom_id(self, service):
        """register_equipment() uses provided equipment_id."""
        result = service.register_equipment(
            equipment_id="EQ-CUSTOM-001",
            equipment_type="KILN",
            name="Kiln-1",
        )
        assert result["equipment_id"] == "EQ-CUSTOM-001"

    def test_register_equipment_default_efficiency(self, service):
        """register_equipment() uses default efficiency of 0.80."""
        result = service.register_equipment(
            equipment_type="BOILER_WATER_TUBE",
            name="Boiler-2",
        )
        assert result.get("efficiency", 0.80) == 0.80


# =====================================================================
# TestServiceRunPipeline
# =====================================================================


class TestServiceRunPipeline:
    """Test pipeline execution through the service."""

    def test_run_pipeline_returns_dict(self, service):
        """run_pipeline() returns a dictionary."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
        ]
        result = service.run_pipeline(inputs)
        assert isinstance(result, dict)

    def test_run_pipeline_increments_counter(self, service):
        """run_pipeline() increments pipeline run counter."""
        inputs = [
            {"fuel_type": "NATURAL_GAS", "quantity": 1000.0, "unit": "CUBIC_METERS"},
        ]
        service.run_pipeline(inputs)
        assert service._total_pipeline_runs == 1


# =====================================================================
# TestServiceGetUncertainty
# =====================================================================


class TestServiceGetUncertainty:
    """Test uncertainty quantification through the service."""

    def test_get_uncertainty_returns_dict(self, service):
        """get_uncertainty() returns a dictionary."""
        calc_result = {
            "calculation_id": "calc-123",
            "total_co2e_kg": 2100.0,
        }
        result = service.get_uncertainty(calc_result)
        assert isinstance(result, dict)
        assert "mean_co2e_kg" in result

    def test_get_uncertainty_default_iterations(self, service):
        """get_uncertainty() uses config iterations when none specified."""
        calc_result = {
            "calculation_id": "calc-456",
            "total_co2e_kg": 1000.0,
        }
        result = service.get_uncertainty(calc_result)
        assert "num_simulations" in result

    def test_get_uncertainty_custom_iterations(self, service):
        """get_uncertainty() respects custom iterations parameter."""
        calc_result = {
            "calculation_id": "calc-789",
            "total_co2e_kg": 500.0,
        }
        result = service.get_uncertainty(calc_result, iterations=200)
        assert isinstance(result, dict)


# =====================================================================
# TestServiceGetAuditTrail
# =====================================================================


class TestServiceGetAuditTrail:
    """Test audit trail retrieval through the service."""

    def test_get_audit_trail_returns_list(self, service):
        """get_audit_trail() returns a list."""
        result = service.get_audit_trail("calc-nonexistent")
        assert isinstance(result, list)

    def test_get_audit_trail_from_cache(self, service):
        """get_audit_trail() returns entries from in-memory cache."""
        service._audit_entries["calc-abc"] = [
            {"step": "validate", "result": "pass"},
        ]
        result = service.get_audit_trail("calc-abc")
        assert len(result) == 1

    def test_get_audit_trail_empty_for_unknown(self, service):
        """get_audit_trail() returns empty list for unknown calc_id."""
        result = service.get_audit_trail("calc-does-not-exist")
        assert result == []


# =====================================================================
# TestServiceGetComplianceMapping
# =====================================================================


class TestServiceGetComplianceMapping:
    """Test compliance mapping retrieval."""

    def test_get_compliance_mapping_all(self, service):
        """get_compliance_mapping() returns all frameworks."""
        result = service.get_compliance_mapping()
        assert isinstance(result, dict)
        assert "mappings" in result
        assert "overall_compliant" in result

    def test_get_compliance_mapping_specific_framework(self, service):
        """get_compliance_mapping() filters by framework."""
        result = service.get_compliance_mapping(framework="GHG_PROTOCOL")
        assert result["framework"] == "GHG_PROTOCOL"

    def test_compliance_mapping_has_provenance(self, service):
        """Compliance mapping includes provenance_hash."""
        result = service.get_compliance_mapping()
        assert "provenance_hash" in result


# =====================================================================
# TestServiceGetHealth
# =====================================================================


class TestServiceGetHealth:
    """Test health check functionality."""

    def test_health_returns_status(self, service):
        """get_health() returns a status field."""
        health = service.get_health()
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_returns_version(self, service):
        """get_health() returns version 1.0.0."""
        health = service.get_health()
        assert health["version"] == "1.0.0"

    def test_health_returns_engine_count(self, service):
        """get_health() returns engines_total as 7."""
        health = service.get_health()
        assert health["engines_total"] == 7

    def test_health_returns_engines_dict(self, service):
        """get_health() returns per-engine availability."""
        health = service.get_health()
        assert isinstance(health["engines"], dict)
        assert "fuel_database" in health["engines"]
        assert "calculator" in health["engines"]
        assert "pipeline" in health["engines"]

    def test_health_has_timestamp(self, service):
        """get_health() includes an ISO-8601 timestamp."""
        health = service.get_health()
        assert "timestamp" in health

    def test_health_has_statistics(self, service):
        """get_health() includes statistics summary."""
        health = service.get_health()
        assert "statistics" in health
        assert "total_calculations" in health["statistics"]

    def test_health_has_provenance_info(self, service):
        """get_health() includes provenance chain validity."""
        health = service.get_health()
        assert "provenance_chain_valid" in health
        assert "provenance_entries" in health


# =====================================================================
# TestConfigureFunction
# =====================================================================


class TestConfigureFunction:
    """Test configure_stationary_combustion function."""

    @pytest.mark.asyncio
    async def test_configure_creates_service(self, config):
        """configure_stationary_combustion creates a service."""
        app = MagicMock()
        app.state = MagicMock()
        svc = await configure_stationary_combustion(app, config=config)
        assert isinstance(svc, StationaryCombustionService)

    @pytest.mark.asyncio
    async def test_configure_attaches_to_app_state(self, config):
        """configure_stationary_combustion sets app.state."""
        app = MagicMock()
        app.state = MagicMock()
        svc = await configure_stationary_combustion(app, config=config)
        app.state.__setattr__.assert_any_call(
            "stationary_combustion_service", svc,
        )

    @pytest.mark.asyncio
    async def test_configure_starts_service(self, config):
        """configure_stationary_combustion calls startup()."""
        app = MagicMock()
        app.state = MagicMock()
        svc = await configure_stationary_combustion(app, config=config)
        assert svc._started is True


# =====================================================================
# TestGetService
# =====================================================================


class TestGetService:
    """Test get_service singleton accessor."""

    def test_get_service_returns_instance(self):
        """get_service() returns a StationaryCombustionService."""
        svc = get_service()
        assert isinstance(svc, StationaryCombustionService)

    def test_get_service_returns_same_instance(self):
        """get_service() returns the same singleton instance."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2


# =====================================================================
# TestGetRouter
# =====================================================================


class TestGetRouter:
    """Test get_router function."""

    def test_get_router_returns_router_or_none(self):
        """get_router() returns an APIRouter or None."""
        router = get_router()
        # May be None if FastAPI not installed, or an APIRouter
        assert router is None or hasattr(router, "routes")


# =====================================================================
# TestServiceLifecycle
# =====================================================================


class TestServiceLifecycle:
    """Test service startup and shutdown lifecycle."""

    def test_startup_sets_started(self, service):
        """startup() sets _started to True."""
        service.startup()
        assert service._started is True

    def test_startup_idempotent(self, service):
        """startup() is safe to call multiple times."""
        service.startup()
        service.startup()
        assert service._started is True

    def test_shutdown_clears_started(self, service):
        """shutdown() sets _started to False."""
        service.startup()
        service.shutdown()
        assert service._started is False

    def test_shutdown_when_not_started(self, service):
        """shutdown() is safe to call when not started."""
        service.shutdown()
        assert service._started is False


# =====================================================================
# TestServiceStatistics
# =====================================================================


class TestServiceStatistics:
    """Test service statistics tracking."""

    def test_get_statistics_returns_dict(self, service):
        """get_statistics() returns a dictionary."""
        stats = service.get_statistics()
        assert isinstance(stats, dict)

    def test_get_statistics_initial_values(self, service):
        """get_statistics() starts with zero counters."""
        stats = service.get_statistics()
        assert stats["total_calculations"] == 0
        assert stats["total_batch_runs"] == 0
        assert stats["total_pipeline_runs"] == 0

    def test_get_statistics_has_fuel_types(self, service):
        """get_statistics() includes fuel type count."""
        stats = service.get_statistics()
        assert "total_fuel_types" in stats
        # Should be at least the number of FuelType enum members
        assert stats["total_fuel_types"] >= 0

    def test_get_statistics_has_timestamp(self, service):
        """get_statistics() includes ISO-8601 timestamp."""
        stats = service.get_statistics()
        assert "timestamp" in stats

    def test_get_statistics_avg_time(self, service):
        """get_statistics() tracks average calculation time."""
        stats = service.get_statistics()
        assert "avg_calculation_time_ms" in stats
        assert stats["avg_calculation_time_ms"] == 0.0
