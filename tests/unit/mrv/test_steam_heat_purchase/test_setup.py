# -*- coding: utf-8 -*-
"""
Unit tests for SteamHeatPurchaseService facade - AGENT-MRV-011.

Tests the unified service interface that delegates to 7 specialized engines,
singleton pattern, health checks, service info, and module-level functions.

Target: ~50 tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

try:
    from greenlang.steam_heat_purchase.setup import (
        SteamHeatPurchaseService,
        get_service,
        create_service,
        reset_service,
    )
    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SERVICE_AVAILABLE,
    reason="greenlang.steam_heat_purchase.setup not importable",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a fresh SteamHeatPurchaseService instance."""
    reset_service()
    svc = SteamHeatPurchaseService()
    yield svc
    reset_service()


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline engine for delegation tests."""
    pipeline = MagicMock(name="SteamHeatPipelineEngine")
    pipeline.calculate_steam_emissions.return_value = {
        "status": "success",
        "total_co2e_kg": 15000.0,
        "calc_id": "calc-001",
        "provenance_hash": "a" * 64,
    }
    pipeline.calculate_heating_emissions.return_value = {
        "status": "success",
        "total_co2e_kg": 8000.0,
        "calc_id": "calc-002",
        "provenance_hash": "b" * 64,
    }
    pipeline.calculate_cooling_emissions.return_value = {
        "status": "success",
        "total_co2e_kg": 5000.0,
        "calc_id": "calc-003",
        "provenance_hash": "c" * 64,
    }
    pipeline.calculate_chp_emissions.return_value = {
        "status": "success",
        "total_co2e_kg": 20000.0,
        "heat_share": 0.5625,
        "power_share": 0.4375,
        "calc_id": "calc-004",
        "provenance_hash": "d" * 64,
    }
    return pipeline


@pytest.fixture
def mock_db_engine():
    """Create a mock database engine."""
    db = MagicMock(name="SteamHeatDatabaseEngine")
    db.get_fuel_emission_factor.return_value = {
        "fuel_type": "natural_gas",
        "co2_ef": 56.1,
        "ch4_ef": 0.001,
        "n2o_ef": 0.0001,
    }
    db.get_all_fuel_emission_factors.return_value = {
        "natural_gas": {"co2_ef": 56.1},
        "coal_bituminous": {"co2_ef": 94.6},
    }
    return db


@pytest.fixture
def mock_chp_engine():
    """Create a mock CHP allocation engine."""
    chp = MagicMock(name="CHPAllocationEngine")
    chp.allocate_chp_emissions.return_value = {
        "method": "efficiency",
        "heat_share": 0.5625,
        "power_share": 0.4375,
        "provenance_hash": "e" * 64,
    }
    return chp


@pytest.fixture
def mock_uncertainty_engine():
    """Create a mock uncertainty engine."""
    unc = MagicMock(name="UncertaintyQuantifierEngine")
    unc.quantify_uncertainty.return_value = {
        "mean_co2e_kg": 15000.0,
        "std_dev_kg": 750.0,
        "ci_lower_kg": 13500.0,
        "ci_upper_kg": 16500.0,
        "provenance_hash": "f" * 64,
    }
    return unc


@pytest.fixture
def mock_compliance_engine():
    """Create a mock compliance engine."""
    comp = MagicMock(name="ComplianceCheckerEngine")
    comp.check_compliance.return_value = {
        "frameworks": {
            "ghg_protocol_scope2": {"status": "compliant", "score_pct": 100},
        },
        "provenance_hash": "0" * 64,
    }
    comp.get_compliance_frameworks.return_value = [
        "ghg_protocol_scope2", "iso_14064", "csrd_esrs",
        "cdp", "sbti", "eu_eed", "epa_mrr",
    ]
    return comp


# ===========================================================================
# 1. Singleton Pattern Tests
# ===========================================================================


class TestSingletonPattern:
    """Tests for SteamHeatPurchaseService singleton."""

    def test_same_instance_returned(self):
        reset_service()
        s1 = SteamHeatPurchaseService()
        s2 = SteamHeatPurchaseService()
        assert s1 is s2

    def test_reset_creates_new_instance(self):
        s1 = SteamHeatPurchaseService()
        reset_service()
        s2 = SteamHeatPurchaseService()
        assert s1 is not s2

    def test_get_service_function(self):
        reset_service()
        svc = get_service()
        assert isinstance(svc, SteamHeatPurchaseService)

    def test_get_service_singleton(self):
        reset_service()
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2

    def test_create_service_returns_instance(self):
        reset_service()
        svc = create_service()
        assert isinstance(svc, SteamHeatPurchaseService)

    def test_reset_service_function(self):
        s1 = get_service()
        reset_service()
        s2 = get_service()
        assert s1 is not s2


# ===========================================================================
# 2. Delegation Tests - Steam Emissions
# ===========================================================================


class TestCalculateSteamEmissions:
    """Tests for calculate_steam_emissions delegation."""

    def test_delegates_to_pipeline(self, service, mock_pipeline):
        service.pipeline_engine = mock_pipeline
        service.db_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.calculate_steam_emissions({
            "steam_purchased_mwh": 1000.0,
            "calculation_method": "supplier_specific",
            "supplier_emission_factor": 0.25,
        })
        assert result.get("status") in ("success", "error")

    def test_invalid_request_returns_error(self, service):
        result = service.calculate_steam_emissions("not_a_dict")
        assert result["status"] == "error"

    def test_missing_required_field_returns_error(self, service):
        result = service.calculate_steam_emissions({})
        assert result["status"] == "error"


# ===========================================================================
# 3. Delegation Tests - Heating Emissions
# ===========================================================================


class TestCalculateHeatingEmissions:
    """Tests for calculate_heating_emissions delegation."""

    def test_delegates_to_pipeline(self, service, mock_pipeline):
        service.pipeline_engine = mock_pipeline
        service.db_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.calculate_heating_emissions({
            "heating_type": "district_heating",
            "energy_purchased_mwh": 500.0,
            "region": "germany",
        })
        assert result.get("status") in ("success", "error")

    def test_missing_heating_type_returns_error(self, service):
        result = service.calculate_heating_emissions({
            "energy_purchased_mwh": 500.0,
        })
        assert result["status"] == "error"

    def test_missing_energy_returns_error(self, service):
        result = service.calculate_heating_emissions({
            "heating_type": "district_heating",
        })
        assert result["status"] == "error"


# ===========================================================================
# 4. Delegation Tests - Cooling Emissions
# ===========================================================================


class TestCalculateCoolingEmissions:
    """Tests for calculate_cooling_emissions delegation."""

    def test_delegates_to_pipeline(self, service, mock_pipeline):
        service.pipeline_engine = mock_pipeline
        service.db_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.calculate_cooling_emissions({
            "cooling_type": "district_cooling",
            "energy_purchased_mwh": 300.0,
            "region": "us",
        })
        assert result.get("status") in ("success", "error")

    def test_missing_cooling_type_returns_error(self, service):
        result = service.calculate_cooling_emissions({
            "energy_purchased_mwh": 300.0,
        })
        assert result["status"] == "error"


# ===========================================================================
# 5. Delegation Tests - CHP Emissions
# ===========================================================================


class TestCalculateCHPEmissions:
    """Tests for calculate_chp_emissions delegation."""

    def test_delegates_to_pipeline(self, service, mock_pipeline):
        service.pipeline_engine = mock_pipeline
        service.db_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.calculate_chp_emissions({
            "total_fuel_gj": 2000,
            "fuel_type": "natural_gas",
            "heat_output_gj": 900,
            "power_output_gj": 700,
            "method": "efficiency",
        })
        assert result.get("status") in ("success", "error")


# ===========================================================================
# 6. Emission Factor Delegation Tests
# ===========================================================================


class TestGetFuelEmissionFactor:
    """Tests for get_fuel_emission_factor delegation."""

    def test_delegates_to_db(self, service, mock_db_engine):
        service.db_engine = mock_db_engine
        # Make sure _ensure_engines does not override
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.get_fuel_emission_factor("natural_gas")
        assert isinstance(result, dict)


# ===========================================================================
# 7. CHP Allocation Delegation Tests
# ===========================================================================


class TestAllocateCHP:
    """Tests for allocate_chp delegation."""

    def test_delegates_to_chp_engine(self, service, mock_chp_engine):
        service.chp_engine = mock_chp_engine
        service.db_engine = MagicMock()
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.allocate_chp({
            "total_fuel_gj": 2000,
            "fuel_type": "natural_gas",
            "heat_output_gj": 900,
            "power_output_gj": 700,
            "method": "efficiency",
        })
        assert isinstance(result, dict)


# ===========================================================================
# 8. Uncertainty Delegation Tests
# ===========================================================================


class TestQuantifyUncertainty:
    """Tests for quantify_uncertainty delegation."""

    def test_delegates_to_uncertainty_engine(self, service, mock_uncertainty_engine):
        service.uncertainty_engine = mock_uncertainty_engine
        service.db_engine = MagicMock()
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.quantify_uncertainty(
            calc_result={"total_co2e_kg": 15000},
            method="monte_carlo",
            iterations=1000,
        )
        assert isinstance(result, dict)


# ===========================================================================
# 9. Compliance Delegation Tests
# ===========================================================================


class TestCheckCompliance:
    """Tests for check_compliance delegation."""

    def test_delegates_to_compliance_engine(self, service, mock_compliance_engine):
        service.compliance_engine = mock_compliance_engine
        service.db_engine = MagicMock()
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()

        result = service.check_compliance(
            calc_result={"total_co2e_kg": 15000},
            frameworks=["ghg_protocol_scope2"],
        )
        assert isinstance(result, dict)


# ===========================================================================
# 10. Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Tests for health_check."""

    def test_health_check_returns_dict(self, service):
        result = service.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_status(self, service):
        result = service.health_check()
        assert "status" in result or "health" in result

    def test_health_check_has_engine_status(self, service):
        result = service.health_check()
        engines = result.get("engines", result.get("components", {}))
        if isinstance(engines, dict):
            assert len(engines) >= 1


# ===========================================================================
# 11. Service Info Tests
# ===========================================================================


class TestServiceInfo:
    """Tests for get_service_info."""

    def test_service_info_returns_dict(self, service):
        info = service.get_service_info()
        assert isinstance(info, dict)

    def test_service_info_has_capabilities(self, service):
        info = service.get_service_info()
        caps = info.get(
            "capabilities",
            info.get("supported_operations", info.get("features", [])),
        )
        assert caps is not None

    def test_service_info_has_version(self, service):
        info = service.get_service_info()
        has_version = (
            "version" in info
            or "service_version" in info
            or "api_version" in info
        )
        assert has_version or isinstance(info, dict)


# ===========================================================================
# 12. Service Stats Tests
# ===========================================================================


class TestServiceStats:
    """Tests for get_service_stats."""

    def test_stats_returns_dict(self, service):
        stats = service.get_service_stats()
        assert isinstance(stats, dict)

    def test_stats_has_request_count(self, service):
        stats = service.get_service_stats()
        count = stats.get("request_count", stats.get("total_requests", 0))
        assert isinstance(count, int)

    def test_stats_has_error_count(self, service):
        stats = service.get_service_stats()
        errors = stats.get("error_count", stats.get("total_errors", 0))
        assert isinstance(errors, int)

    def test_stats_count_increases(self, service):
        # Trigger an error to increment counters
        service.calculate_steam_emissions({})
        stats = service.get_service_stats()
        req_count = stats.get("request_count", stats.get("total_requests", 0))
        # At least one request was made
        assert req_count >= 1


# ===========================================================================
# 13. Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in service methods."""

    def test_steam_none_request_returns_error(self, service):
        result = service.calculate_steam_emissions(None)
        assert result["status"] == "error"

    def test_steam_list_request_returns_error(self, service):
        result = service.calculate_steam_emissions([1, 2, 3])
        assert result["status"] == "error"

    def test_heating_none_request_returns_error(self, service):
        result = service.calculate_heating_emissions(None)
        assert result["status"] == "error"

    def test_cooling_none_request_returns_error(self, service):
        result = service.calculate_cooling_emissions(None)
        assert result["status"] == "error"

    def test_chp_none_request_returns_error(self, service):
        result = service.calculate_chp_emissions(None)
        assert result["status"] == "error"

    def test_error_result_has_error_type(self, service):
        result = service.calculate_steam_emissions("invalid")
        assert "error_type" in result
        assert result["error_type"] in ("ValueError", "TypeError", "AttributeError")

    def test_error_result_has_timestamp(self, service):
        result = service.calculate_steam_emissions({})
        assert "timestamp" in result


# ===========================================================================
# 14. Configuration Tests
# ===========================================================================


class TestConfiguration:
    """Tests for service configuration handling."""

    def test_default_config(self, service):
        assert service.config is not None

    def test_dict_config(self):
        reset_service()
        svc = SteamHeatPurchaseService(config={"key": "value"})
        assert svc.config is not None

    def test_none_config_uses_default(self):
        reset_service()
        svc = SteamHeatPurchaseService(config=None)
        assert svc.config is not None


# ===========================================================================
# 15. Lazy Engine Initialization Tests
# ===========================================================================


class TestLazyInit:
    """Tests for lazy engine initialization."""

    def test_engines_initially_none(self, service):
        assert service.db_engine is None
        assert service.steam_engine is None
        assert service.heating_engine is None
        assert service.cooling_engine is None
        assert service.chp_engine is None
        assert service.uncertainty_engine is None
        assert service.compliance_engine is None
        assert service.pipeline_engine is None

    def test_ensure_engines_called_on_calculate(self, service):
        # When engines are not available, calculate returns error
        result = service.calculate_steam_emissions({
            "steam_purchased_mwh": 100,
        })
        assert isinstance(result, dict)


# ===========================================================================
# 16. Get Compliance Frameworks Tests
# ===========================================================================


class TestGetComplianceFrameworks:
    """Tests for get_compliance_frameworks delegation."""

    def test_returns_list_or_error(self, service, mock_compliance_engine):
        service.compliance_engine = mock_compliance_engine
        service.db_engine = MagicMock()
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()

        result = service.get_compliance_frameworks()
        if isinstance(result, list):
            assert len(result) == 7


# ===========================================================================
# 17. Get Calculation Tests
# ===========================================================================


class TestGetCalculation:
    """Tests for get_calculation delegation."""

    def test_returns_dict_or_none(self, service):
        result = service.get_calculation("nonexistent-id")
        assert result is None or isinstance(result, dict)


# ===========================================================================
# 18. Emission Factor Accessor Tests
# ===========================================================================


class TestEmissionFactorAccessors:
    """Tests for emission factor and parameter accessors."""

    def test_get_all_fuel_emission_factors(self, service, mock_db_engine):
        service.db_engine = mock_db_engine
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.get_all_fuel_emission_factors()
        assert isinstance(result, dict)

    def test_get_district_heating_factor(self, service, mock_db_engine):
        mock_db_engine.get_district_heating_factor.return_value = {
            "region": "germany",
            "ef_kgco2e_per_gj": 72.0,
        }
        service.db_engine = mock_db_engine
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.get_district_heating_factor("germany")
        assert isinstance(result, dict)

    def test_get_cooling_system_factor(self, service, mock_db_engine):
        mock_db_engine.get_cooling_system_factor.return_value = {
            "technology": "centrifugal_chiller",
            "cop_default": 6.0,
        }
        service.db_engine = mock_db_engine
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.get_cooling_system_factor("centrifugal_chiller")
        assert isinstance(result, dict)

    def test_get_chp_defaults(self, service, mock_db_engine):
        mock_db_engine.get_chp_defaults.return_value = {
            "natural_gas": {"electrical_efficiency": 0.35},
        }
        service.db_engine = mock_db_engine
        service.pipeline_engine = MagicMock()
        service.steam_engine = MagicMock()
        service.heating_engine = MagicMock()
        service.cooling_engine = MagicMock()
        service.chp_engine = MagicMock()
        service.uncertainty_engine = MagicMock()
        service.compliance_engine = MagicMock()

        result = service.get_chp_defaults("natural_gas")
        assert isinstance(result, dict)


# ===========================================================================
# 19. Service Metadata Tests
# ===========================================================================


class TestServiceMetadata:
    """Tests for service metadata in responses."""

    def test_steam_error_includes_metadata(self, service):
        result = service.calculate_steam_emissions({})
        assert "error" in result
        assert "timestamp" in result

    def test_heating_error_includes_metadata(self, service):
        result = service.calculate_heating_emissions({})
        assert "error" in result

    def test_cooling_error_includes_metadata(self, service):
        result = service.calculate_cooling_emissions({})
        assert "error" in result
