# -*- coding: utf-8 -*-
"""
Test suite for investments.setup - AGENT-MRV-028.

Tests the InvestmentsService facade including initialization, engine
access, single/batch/portfolio calculation, compliance checking,
emission factor retrieval, asset class lookup, singleton pattern,
and thread safety.

Coverage:
- InvestmentsService initialization
- Engine access methods
- Single investment calculation (equity, bond, CRE, sovereign)
- Batch investment calculation
- Portfolio calculation
- Compliance check delegation
- Emission factor retrieval
- get_service() singleton pattern
- get_router() returns FastAPI router
- Thread safety
- Request/Response Pydantic models

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
    from greenlang.agents.mrv.investments.setup import (
        InvestmentsService,
        get_service,
        get_router,
        EquityCalculationRequest,
        CorporateBondCalculationRequest,
        CRECalculationRequest,
        MortgageCalculationRequest,
        MotorVehicleCalculationRequest,
        SovereignBondCalculationRequest,
        PortfolioCalculationRequest,
        ComplianceCheckRequest,
        InvestmentCalculationResponse,
        PortfolioCalculationResponse,
        ComplianceCheckResponse,
        HealthResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="InvestmentsService not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service():
    """Create a fresh InvestmentsService instance."""
    return InvestmentsService()


@pytest.fixture
def equity_request():
    """EquityCalculationRequest for Apple listed equity."""
    return EquityCalculationRequest(
        asset_class="listed_equity",
        investee_name="Apple Inc.",
        outstanding_amount=100000000,
        evic=3000000000000,
        investee_scope1=22400,
        investee_scope2=9100,
        sector="information_technology",
        country="US",
    )


@pytest.fixture
def bond_request():
    """CorporateBondCalculationRequest for Tesla bond."""
    return CorporateBondCalculationRequest(
        asset_class="corporate_bond",
        investee_name="Tesla Inc.",
        outstanding_amount=75000000,
        evic=500000000000,
        investee_scope1=30000,
        investee_scope2=12000,
        sector="consumer_discretionary",
        country="US",
    )


@pytest.fixture
def cre_request():
    """CRECalculationRequest for office building."""
    return CRECalculationRequest(
        asset_class="commercial_real_estate",
        property_name="Downtown Office Tower",
        outstanding_amount=25000000,
        property_value=50000000,
        floor_area_m2=10000,
        property_type="office",
        epc_rating="B",
        climate_zone="temperate",
        country="US",
    )


@pytest.fixture
def sovereign_request():
    """SovereignBondCalculationRequest for US Treasury."""
    return SovereignBondCalculationRequest(
        asset_class="sovereign_bond",
        country="US",
        outstanding_amount=500000000,
        gdp_ppp=25460000000000,
        country_emissions=5222000000,
    )


# ===========================================================================
# Service Initialization Tests
# ===========================================================================


@_SKIP
class TestServiceInitialization:
    """Test InvestmentsService initialization."""

    def test_service_creation(self, service):
        """Test InvestmentsService can be created."""
        assert service is not None

    def test_service_has_database_engine(self, service):
        """Test service has database engine."""
        assert service.database_engine is not None

    def test_service_has_equity_engine(self, service):
        """Test service has equity calculator engine."""
        assert service.equity_engine is not None

    def test_service_has_debt_engine(self, service):
        """Test service has debt calculator engine."""
        assert service.debt_engine is not None

    def test_service_has_real_asset_engine(self, service):
        """Test service has real asset calculator engine."""
        assert service.real_asset_engine is not None

    def test_service_has_sovereign_engine(self, service):
        """Test service has sovereign bond calculator engine."""
        assert service.sovereign_engine is not None

    def test_service_has_compliance_engine(self, service):
        """Test service has compliance checker engine."""
        assert service.compliance_engine is not None

    def test_service_has_pipeline_engine(self, service):
        """Test service has pipeline engine."""
        assert service.pipeline_engine is not None


# ===========================================================================
# Calculation Method Tests
# ===========================================================================


@_SKIP
class TestServiceCalculations:
    """Test service calculation methods."""

    def test_calculate_equity(self, service, equity_request):
        """Test equity calculation via service."""
        result = service.calculate_equity(equity_request)
        assert result is not None
        assert result.financed_emissions > 0

    def test_calculate_bond(self, service, bond_request):
        """Test corporate bond calculation via service."""
        result = service.calculate_corporate_bond(bond_request)
        assert result is not None
        assert result.financed_emissions > 0

    def test_calculate_cre(self, service, cre_request):
        """Test CRE calculation via service."""
        result = service.calculate_cre(cre_request)
        assert result is not None
        assert result.financed_emissions > 0

    def test_calculate_sovereign(self, service, sovereign_request):
        """Test sovereign bond calculation via service."""
        result = service.calculate_sovereign(sovereign_request)
        assert result is not None
        assert result.financed_emissions > 0

    def test_calculate_portfolio(self, service):
        """Test portfolio calculation via service."""
        portfolio = PortfolioCalculationRequest(
            portfolio_name="Test Portfolio",
            reporting_year=2024,
            investments=[],
        )
        result = service.calculate_portfolio(portfolio)
        assert result is not None

    def test_check_compliance(self, service):
        """Test compliance check via service."""
        request = ComplianceCheckRequest(
            frameworks=["ghg_protocol", "pcaf"],
            calculation_results=[{"total_co2e": 1000}],
        )
        result = service.check_compliance(request)
        assert result is not None


# ===========================================================================
# Service Method Tests
# ===========================================================================


@_SKIP
class TestServiceMethods:
    """Test service utility methods."""

    def test_get_sector_factors(self, service):
        """Test sector factors retrieval."""
        factors = service.get_sector_factors()
        assert factors is not None
        assert len(factors) >= 12

    def test_get_country_emissions(self, service):
        """Test country emissions retrieval."""
        emissions = service.get_country_emissions()
        assert emissions is not None
        assert len(emissions) >= 50

    def test_get_pcaf_quality_criteria(self, service):
        """Test PCAF quality criteria retrieval."""
        criteria = service.get_pcaf_quality_criteria()
        assert criteria is not None

    def test_get_asset_classes(self, service):
        """Test asset classes listing."""
        classes = service.get_asset_classes()
        assert len(classes) == 8

    def test_health_check(self, service):
        """Test health check."""
        health = service.health_check()
        assert health is not None
        assert health.status == "healthy"

    def test_get_agent_info(self, service):
        """Test agent info retrieval."""
        info = service.get_agent_info()
        assert info["agent_id"] == "GL-MRV-S3-015"
        assert info["component"] == "AGENT-MRV-028"


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


@_SKIP
class TestSingletonPattern:
    """Test get_service() singleton pattern."""

    def test_get_service_singleton(self):
        """Test get_service returns the same instance."""
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2

    def test_get_service_thread_safe(self):
        """Test get_service is thread-safe."""
        instances = []

        def get_instance():
            instances.append(get_service())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)


# ===========================================================================
# Router Tests
# ===========================================================================


@_SKIP
class TestRouter:
    """Test get_router() factory."""

    def test_get_router_returns_router(self):
        """Test get_router returns a FastAPI APIRouter."""
        router = get_router()
        assert router is not None

    def test_get_router_has_routes(self):
        """Test router has registered routes."""
        router = get_router()
        assert len(router.routes) > 0


# ===========================================================================
# Request/Response Model Tests
# ===========================================================================


@_SKIP
class TestRequestResponseModels:
    """Test request and response Pydantic models."""

    def test_equity_request_creation(self):
        """Test EquityCalculationRequest creation."""
        req = EquityCalculationRequest(
            asset_class="listed_equity",
            investee_name="Test",
            outstanding_amount=100,
            evic=1000,
            investee_scope1=10,
            investee_scope2=5,
            sector="energy",
            country="US",
        )
        assert req.investee_name == "Test"

    def test_health_response_creation(self):
        """Test HealthResponse creation."""
        resp = HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-015",
            version="1.0.0",
        )
        assert resp.status == "healthy"

    def test_investment_calculation_response(self):
        """Test InvestmentCalculationResponse creation."""
        resp = InvestmentCalculationResponse(
            calculation_id="calc-001",
            asset_class="listed_equity",
            financed_emissions=1050.0,
            pcaf_quality_score=1,
            provenance_hash="a" * 64,
        )
        assert resp.financed_emissions == 1050.0

    def test_portfolio_response(self):
        """Test PortfolioCalculationResponse creation."""
        resp = PortfolioCalculationResponse(
            portfolio_name="Test",
            total_financed_emissions=103000.0,
            weighted_pcaf_score=2.5,
            provenance_hash="b" * 64,
        )
        assert resp.total_financed_emissions == 103000.0

    def test_compliance_response(self):
        """Test ComplianceCheckResponse creation."""
        resp = ComplianceCheckResponse(
            framework="ghg_protocol",
            status="pass",
            score=95.0,
            findings=[],
            recommendations=[],
        )
        assert resp.status == "pass"
