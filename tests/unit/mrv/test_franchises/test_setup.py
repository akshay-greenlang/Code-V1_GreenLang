# -*- coding: utf-8 -*-
"""
Test suite for franchises.setup - AGENT-MRV-027.

Tests the FranchisesService facade including initialization, engine access,
service methods delegation, get_service() singleton, get_router() returns
APIRouter, create_app() returns FastAPI, and health check.

Target: 35+ tests, 85%+ coverage.

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
    from greenlang.franchises.setup import (
        FranchisesService,
        get_service,
        get_router,
        create_app,
        FranchiseCalculationRequest,
        FranchiseSpecificRequest,
        AverageDataRequest,
        SpendBasedRequest,
        HybridRequest,
        NetworkCalculationRequest,
        BatchCalculationRequest,
        ComplianceCheckRequest,
        CalculationFilterRequest,
        AggregationRequest,
        FranchiseCalculationResponse,
        BatchCalculationResponse,
        ComplianceCheckResponse,
        AggregationResponse,
        HealthResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="FranchisesService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh FranchisesService instance."""
    return FranchisesService()


@pytest.fixture
def franchise_specific_request():
    """FranchiseSpecificRequest for QSR unit."""
    return FranchiseSpecificRequest(
        units=[{
            "unit_id": "FRN-QSR-001",
            "franchise_type": "qsr_restaurant",
            "ownership_type": "franchised",
            "floor_area_m2": 250.0,
            "country_code": "US",
            "climate_zone": "temperate",
            "electricity_kwh": 180000.0,
            "natural_gas_therms": 12000.0,
            "operating_months": 12,
        }],
        reporting_year=2025,
    )


@pytest.fixture
def average_data_request():
    """AverageDataRequest for QSR network."""
    return AverageDataRequest(
        franchise_type="qsr_restaurant",
        unit_count=150,
        avg_floor_area_m2=220.0,
        climate_zone="temperate",
        country_code="US",
        reporting_year=2025,
    )


@pytest.fixture
def spend_based_request():
    """SpendBasedRequest for QSR network."""
    return SpendBasedRequest(
        franchise_type="qsr_restaurant",
        naics_code="722513",
        total_revenue_usd=450000000.0,
        total_units=500,
        reporting_year=2025,
        currency="USD",
    )


@pytest.fixture
def compliance_request():
    """ComplianceCheckRequest for GHG Protocol."""
    return ComplianceCheckRequest(
        calculation_id="calc-001",
        frameworks=["ghg_protocol", "cdp"],
    )


# ===========================================================================
# Tests
# ===========================================================================


@_SKIP
class TestFranchisesService:
    """Test suite for FranchisesService facade."""

    def test_service_creation(self):
        """FranchisesService can be instantiated."""
        service = FranchisesService()
        assert service is not None
        assert service._initialized is True

    def test_service_has_all_engines(self, service):
        """Service exposes all 7 engine attributes."""
        engine_attrs = [
            "_database_engine",
            "_franchise_specific_engine",
            "_average_data_engine",
            "_spend_based_engine",
            "_hybrid_engine",
            "_compliance_engine",
            "_pipeline_engine",
        ]
        for attr in engine_attrs:
            assert hasattr(service, attr), f"Missing engine attribute: {attr}"

    def test_service_calculate_franchise_specific(self, service, franchise_specific_request):
        """calculate_franchise_specific delegates to franchise-specific engine."""
        response = service.calculate_franchise_specific(franchise_specific_request)
        assert isinstance(response, FranchiseCalculationResponse)
        assert response.total_co2e_kg > 0

    def test_service_calculate_average_data(self, service, average_data_request):
        """calculate_average_data delegates to average-data engine."""
        response = service.calculate_average_data(average_data_request)
        assert isinstance(response, FranchiseCalculationResponse)
        assert response.total_co2e_kg > 0

    def test_service_calculate_spend_based(self, service, spend_based_request):
        """calculate_spend_based delegates to spend-based engine."""
        response = service.calculate_spend_based(spend_based_request)
        assert isinstance(response, FranchiseCalculationResponse)
        assert response.total_co2e_kg > 0

    def test_service_check_compliance(self, service, compliance_request):
        """check_compliance delegates to compliance engine."""
        response = service.check_compliance(compliance_request)
        assert isinstance(response, ComplianceCheckResponse)

    def test_service_list_emission_factors(self, service):
        """list_emission_factors returns emission factor data."""
        factors = service.list_emission_factors("qsr_restaurant")
        assert factors is not None

    def test_service_get_benchmarks(self, service):
        """get_benchmarks returns EUI benchmark data."""
        benchmarks = service.get_benchmarks()
        assert benchmarks is not None

    def test_service_get_grid_factors(self, service):
        """get_grid_factors returns grid emission factors."""
        grid_factors = service.get_grid_factors()
        assert grid_factors is not None

    def test_service_get_franchise_types(self, service):
        """get_franchise_types returns list of franchise types."""
        types = service.get_franchise_types()
        assert isinstance(types, list)
        assert len(types) == 10

    def test_service_health_check(self, service):
        """health_check returns healthy status."""
        health = service.health_check()
        assert health is not None
        assert health.get("status") == "healthy" or isinstance(health, HealthResponse)

    def test_service_get_aggregations(self, service):
        """get_aggregations returns aggregation data."""
        agg = service.get_aggregations(reporting_period="2025")
        assert agg is not None


@_SKIP
class TestGetService:
    """Test get_service() singleton."""

    def test_get_service_returns_instance(self):
        """get_service returns a FranchisesService instance."""
        service = get_service()
        assert isinstance(service, FranchisesService)

    def test_get_service_singleton(self):
        """get_service returns the same instance."""
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2

    def test_get_service_thread_safety(self):
        """get_service is thread-safe."""
        results = []
        errors = []

        def get_svc():
            try:
                svc = get_service()
                results.append(id(svc))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_svc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(set(results)) == 1


@_SKIP
class TestGetRouter:
    """Test get_router() returns APIRouter."""

    def test_get_router_returns_router(self):
        """get_router returns a FastAPI APIRouter."""
        try:
            from fastapi import APIRouter
            router = get_router()
            assert isinstance(router, APIRouter)
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_router_has_routes(self):
        """Router has routes defined."""
        try:
            router = get_router()
            assert len(router.routes) > 0
        except ImportError:
            pytest.skip("FastAPI not available")


@_SKIP
class TestCreateApp:
    """Test create_app() returns FastAPI app."""

    def test_create_app_returns_fastapi(self):
        """create_app returns a FastAPI application."""
        try:
            from fastapi import FastAPI
            app = create_app()
            assert isinstance(app, FastAPI)
        except ImportError:
            pytest.skip("FastAPI not available")


@_SKIP
class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_status(self):
        """Health check returns healthy status."""
        service = FranchisesService()
        health = service.health_check()
        status = health.get("status") if isinstance(health, dict) else getattr(health, "status", None)
        assert status == "healthy"

    def test_health_check_has_agent_id(self):
        """Health check includes agent ID."""
        service = FranchisesService()
        health = service.health_check()
        agent_id = health.get("agent_id") if isinstance(health, dict) else getattr(health, "agent_id", None)
        assert agent_id == "GL-MRV-S3-014"

    def test_health_check_has_version(self):
        """Health check includes version."""
        service = FranchisesService()
        health = service.health_check()
        version = health.get("version") if isinstance(health, dict) else getattr(health, "version", None)
        assert version == "1.0.0"

    def test_health_check_has_engines(self):
        """Health check reports engine status."""
        service = FranchisesService()
        health = service.health_check()
        engines = health.get("engines") if isinstance(health, dict) else getattr(health, "engines", None)
        if engines:
            assert isinstance(engines, (dict, list))
