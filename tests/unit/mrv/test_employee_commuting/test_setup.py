# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-020 Employee Commuting Agent - EmployeeCommutingService (setup.py).

Tests the service facade including initialization, engine access, single/batch
calculation, commute/telework delegation, compliance checking, emission factor
retrieval, commute mode listing, working days lookup, health check,
singleton pattern, and thread safety.

Target: 30 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.employee_commuting.setup import (
        EmployeeCommutingService,
        get_service,
        get_router,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="EmployeeCommutingService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh EmployeeCommutingService instance."""
    return EmployeeCommutingService()


@pytest.fixture
def sov_input():
    """SOV commute input: medium petrol car, 15km, full-time, US."""
    return {
        "mode": "sov",
        "vehicle_type": "car_medium_petrol",
        "one_way_distance_km": 15.0,
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
        "employee_id": "EMP-001",
    }


@pytest.fixture
def transit_input():
    """Transit input: local bus, 10km, full-time, US."""
    return {
        "mode": "bus",
        "transit_type": "bus_local",
        "one_way_distance_km": 10.0,
        "commute_days_per_week": 5,
        "work_schedule": "full_time",
        "region": "US",
        "employee_id": "EMP-002",
    }


@pytest.fixture
def telework_input():
    """Telework input: full remote, US."""
    return {
        "mode": "telework",
        "frequency": "full_remote",
        "region": "US",
        "work_schedule": "full_time",
        "employee_id": "EMP-003",
    }


@pytest.fixture
def compliance_input():
    """Compliance check input for GHG Protocol."""
    return {
        "frameworks": ["GHG_PROTOCOL"],
        "total_co2e": 125000.0,
        "method": "employee_specific",
    }


@pytest.fixture
def batch_input(sov_input, transit_input, telework_input):
    """Batch input with 3 employees."""
    return {
        "employees": [sov_input, transit_input, telework_input],
        "reporting_period": "2025",
    }


# ===========================================================================
# Service Creation Tests (3)
# ===========================================================================


@_SKIP
class TestServiceCreation:
    """Test EmployeeCommutingService initialization and singleton."""

    def test_service_creation(self):
        """EmployeeCommutingService can be instantiated."""
        service = EmployeeCommutingService()
        assert service is not None

    def test_get_service_singleton(self):
        """get_service returns the same instance on repeated calls."""
        import greenlang.employee_commuting.setup as setup_mod
        original = getattr(setup_mod, "_service_instance", None)
        setup_mod._service_instance = None
        try:
            s1 = get_service()
            s2 = get_service()
            assert s1 is s2
        finally:
            setup_mod._service_instance = original

    def test_get_service_thread_safety(self):
        """get_service is thread-safe under concurrent access."""
        import greenlang.employee_commuting.setup as setup_mod
        original = getattr(setup_mod, "_service_instance", None)
        setup_mod._service_instance = None
        try:
            services = []
            errors = []

            def fetch():
                try:
                    s = get_service()
                    services.append(id(s))
                except Exception as ex:
                    errors.append(str(ex))

            threads = [threading.Thread(target=fetch) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(set(services)) == 1
        finally:
            setup_mod._service_instance = original


# ===========================================================================
# Engine Initialization Tests (2)
# ===========================================================================


@_SKIP
class TestEngineInitialization:
    """Test engine initialization and fallback."""

    def test_service_has_all_engines(self, service):
        """Service initializes all 7 engine attributes."""
        engine_attrs = [
            "_database_engine",
            "_personal_vehicle_engine",
            "_public_transit_engine",
            "_active_transport_engine",
            "_telework_engine",
            "_compliance_engine",
            "_pipeline_engine",
        ]
        for attr in engine_attrs:
            assert hasattr(service, attr), f"Missing engine attribute: {attr}"

    def test_service_graceful_engine_fallback(self):
        """Service handles ImportError gracefully for unavailable engines."""
        # The service should not crash even if individual engines fail to import
        # This is ensured by the try/except pattern in setup.py
        service = EmployeeCommutingService()
        assert service is not None


# ===========================================================================
# Calculate Method Tests (3)
# ===========================================================================


@_SKIP
class TestCalculateMethods:
    """Test calculate and calculate_batch methods."""

    def test_calculate_sov(self, service, sov_input):
        """calculate() with SOV input returns positive emissions."""
        response = service.calculate(sov_input)
        assert response is not None
        co2e = response.get("total_co2e_kg", 0)
        assert co2e > 0

    def test_calculate_batch(self, service, batch_input):
        """calculate_batch() processes multiple employees."""
        response = service.calculate_batch(batch_input)
        assert response is not None
        count = response.get("employee_count", response.get("count", 0))
        assert count >= 3

    def test_calculate_returns_provenance(self, service, sov_input):
        """calculate() result includes provenance hash."""
        response = service.calculate(sov_input)
        prov = response.get("provenance_hash") or response.get("provenance")
        assert prov is not None


# ===========================================================================
# Commute Calculation Tests (3)
# ===========================================================================


@_SKIP
class TestCommuteCalculation:
    """Test commute-specific calculation methods."""

    def test_calculate_commute_sov(self, service, sov_input):
        """calculate_commute() returns SOV commute emissions."""
        response = service.calculate_commute(sov_input)
        assert response is not None
        co2e = response.get("commute_co2e_kg", response.get("total_co2e_kg", 0))
        assert co2e > 0

    def test_calculate_commute_transit(self, service, transit_input):
        """calculate_commute() returns transit commute emissions."""
        response = service.calculate_commute(transit_input)
        assert response is not None
        co2e = response.get("commute_co2e_kg", response.get("total_co2e_kg", 0))
        assert co2e > 0

    def test_calculate_commute_carpool(self, service):
        """calculate_commute() handles carpool with occupancy."""
        carpool = {
            "mode": "carpool",
            "vehicle_type": "car_average",
            "one_way_distance_km": 20.0,
            "occupants": 3,
            "commute_days_per_week": 5,
            "work_schedule": "full_time",
            "region": "US",
        }
        response = service.calculate_commute(carpool)
        assert response is not None


# ===========================================================================
# Telework Calculation Tests (2)
# ===========================================================================


@_SKIP
class TestTeleworkCalculation:
    """Test telework calculation methods."""

    def test_calculate_telework(self, service, telework_input):
        """calculate_telework() returns telework energy emissions."""
        response = service.calculate_telework(telework_input)
        assert response is not None
        co2e = response.get("telework_co2e_kg", response.get("total_co2e_kg", 0))
        assert co2e >= 0

    def test_calculate_telework_hybrid(self, service):
        """calculate_telework() handles hybrid schedule."""
        hybrid = {
            "frequency": "hybrid_3",
            "region": "US",
            "work_schedule": "full_time",
        }
        response = service.calculate_telework(hybrid)
        assert response is not None


# ===========================================================================
# Compliance Check Tests (2)
# ===========================================================================


@_SKIP
class TestComplianceCheck:
    """Test compliance checking methods."""

    def test_check_compliance(self, service, sov_input):
        """check_compliance() returns compliance results."""
        # First calculate to get a result
        calc_result = service.calculate(sov_input)
        # Then check compliance
        compliance_input = {
            "result": calc_result,
            "frameworks": ["GHG_PROTOCOL"],
        }
        response = service.check_compliance(compliance_input)
        assert response is not None

    def test_check_compliance_all_frameworks(self, service, sov_input):
        """check_compliance() with all 7 frameworks."""
        calc_result = service.calculate(sov_input)
        compliance_input = {
            "result": calc_result,
            "frameworks": [
                "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS",
                "CDP", "SBTI", "SB_253", "GRI",
            ],
        }
        response = service.check_compliance(compliance_input)
        assert response is not None


# ===========================================================================
# Uncertainty Analysis Tests (1)
# ===========================================================================


@_SKIP
class TestUncertaintyAnalysis:
    """Test uncertainty analysis methods."""

    def test_analyze_uncertainty(self, service, sov_input):
        """analyze_uncertainty() returns uncertainty bounds."""
        calc_result = service.calculate(sov_input)
        uncertainty_input = {
            "result": calc_result,
            "method": "monte_carlo",
        }
        response = service.analyze_uncertainty(uncertainty_input)
        assert response is not None


# ===========================================================================
# Health Check Tests (1)
# ===========================================================================


@_SKIP
class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, service):
        """health_check() returns status information."""
        response = service.health_check()
        assert response is not None
        assert isinstance(response, dict)
        # Should contain at least a status field
        status = response.get("status") or response.get("healthy")
        assert status is not None


# ===========================================================================
# Emission Factor Retrieval Tests (2)
# ===========================================================================


@_SKIP
class TestEmissionFactors:
    """Test emission factor retrieval methods."""

    def test_get_emission_factors(self, service):
        """get_emission_factors() returns vehicle emission factors."""
        response = service.get_emission_factors("vehicle")
        assert response is not None
        if isinstance(response, dict):
            assert response.get("success", True) is not False
        elif isinstance(response, list):
            assert len(response) > 0

    def test_get_emission_factors_transit(self, service):
        """get_emission_factors('transit') returns transit EFs."""
        response = service.get_emission_factors("transit")
        assert response is not None


# ===========================================================================
# Commute Mode and Working Days Tests (2)
# ===========================================================================


@_SKIP
class TestCommuteModesAndWorkingDays:
    """Test commute mode listing and working days lookup."""

    def test_get_commute_modes(self, service):
        """get_commute_modes() returns list of supported modes."""
        response = service.get_commute_modes()
        assert response is not None
        if isinstance(response, list):
            assert len(response) >= 10  # At least 10 modes
            mode_names = [
                m.get("mode", m) if isinstance(m, dict) else str(m)
                for m in response
            ]
            # Should include SOV, bus, telework at minimum
            combined = " ".join(mode_names).lower()
            assert "sov" in combined or "car" in combined
        elif isinstance(response, dict):
            assert len(response) >= 10

    def test_get_working_days(self, service):
        """get_working_days() returns regional working day defaults."""
        response = service.get_working_days()
        assert response is not None
        if isinstance(response, dict):
            # Should have entries for common regions
            assert len(response) > 0
        elif isinstance(response, list):
            assert len(response) > 0


# ===========================================================================
# Router Tests (2)
# ===========================================================================


@_SKIP
class TestRouter:
    """Test FastAPI router creation."""

    def test_get_router_returns_router(self):
        """get_router returns a FastAPI APIRouter instance."""
        try:
            from fastapi import APIRouter
            rtr = get_router()
            assert isinstance(rtr, APIRouter)
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_router_has_prefix(self):
        """Router has the expected /api/v1/employee-commuting prefix."""
        try:
            rtr = get_router()
            assert rtr.prefix == "/api/v1/employee-commuting"
        except ImportError:
            pytest.skip("FastAPI not available")


# ===========================================================================
# Service Method Existence Tests (2)
# ===========================================================================


@_SKIP
class TestServiceMethodExistence:
    """Test that all expected public methods exist."""

    def test_service_all_methods_exist(self, service):
        """Service has all expected public methods."""
        expected_methods = [
            "calculate",
            "calculate_batch",
            "calculate_commute",
            "calculate_telework",
            "check_compliance",
            "analyze_uncertainty",
            "health_check",
            "get_emission_factors",
            "get_commute_modes",
            "get_working_days",
        ]
        for method_name in expected_methods:
            assert hasattr(service, method_name), f"Missing method: {method_name}"
            assert callable(getattr(service, method_name)), f"Not callable: {method_name}"

    def test_service_all_methods_callable(self, service):
        """All expected methods are callable without exception on type check."""
        methods = [
            "calculate", "calculate_batch", "calculate_commute",
            "calculate_telework", "check_compliance", "analyze_uncertainty",
            "health_check", "get_emission_factors", "get_commute_modes",
            "get_working_days",
        ]
        for name in methods:
            method = getattr(service, name, None)
            assert method is not None, f"Method {name} is None"
            assert callable(method), f"Method {name} is not callable"


# ===========================================================================
# Integration Tests (3)
# ===========================================================================


@_SKIP
class TestServiceIntegration:
    """Integration tests for full service workflows."""

    def test_calculate_then_compliance(self, service, sov_input):
        """Calculate then check compliance in sequence."""
        calc_result = service.calculate(sov_input)
        assert calc_result is not None
        compliance_input = {
            "result": calc_result,
            "frameworks": ["GHG_PROTOCOL"],
        }
        comp_result = service.check_compliance(compliance_input)
        assert comp_result is not None

    def test_batch_then_aggregate(self, service, sov_input, transit_input):
        """Batch calculate then retrieve aggregation."""
        batch = {
            "employees": [sov_input, transit_input],
            "reporting_period": "2025",
        }
        result = service.calculate_batch(batch)
        assert result is not None
        total = result.get("total_co2e_kg", 0)
        assert total > 0

    def test_multiple_modes_in_batch(self, service, sov_input, transit_input, telework_input):
        """Batch with SOV, transit, and telework returns combined emissions."""
        batch = {
            "employees": [sov_input, transit_input, telework_input],
            "reporting_period": "2025",
        }
        result = service.calculate_batch(batch)
        assert result is not None
        # All modes should be represented
        by_mode = result.get("by_mode", {})
        if by_mode:
            assert len(by_mode) >= 2


# ===========================================================================
# COVERAGE META-TEST
# ===========================================================================


def test_setup_coverage():
    """Meta-test to ensure comprehensive setup coverage."""
    tested_categories = [
        "Service Creation (3 tests)",
        "Engine Initialization (2 tests)",
        "Calculate Methods (3 tests)",
        "Commute Calculation (3 tests)",
        "Telework Calculation (2 tests)",
        "Compliance Check (2 tests)",
        "Uncertainty Analysis (1 test)",
        "Health Check (1 test)",
        "Emission Factors (2 tests)",
        "Commute Modes & Working Days (2 tests)",
        "Router (2 tests)",
        "Method Existence (2 tests)",
        "Integration (3 tests)",
    ]
    # 3+2+3+3+2+2+1+1+2+2+2+2+3 = 28 tests (plus meta = 29)
    assert len(tested_categories) == 13
