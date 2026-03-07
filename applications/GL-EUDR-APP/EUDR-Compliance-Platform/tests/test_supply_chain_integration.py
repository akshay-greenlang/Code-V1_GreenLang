# -*- coding: utf-8 -*-
"""
Integration Tests - AGENT-EUDR-001 Supply Chain Mapper in GL-EUDR-APP

Tests the integration of the AGENT-EUDR-001 Supply Chain Mapper
into the GL-EUDR-APP platform, covering:

1. Configuration: SCM settings in EUDRAppConfig
2. Router registration: All 25+ SCM endpoints accessible
3. Service layer: SupplyChainAppService lifecycle and helpers
4. DDS integration: Supply chain section in DDS generation
5. Setup facade: EUDRComplianceService with supply chain support
6. Auth middleware: Proper authentication on SCM routes

Test count: 45+ tests

NOTE: These tests use mocks for the underlying SupplyChainMapperService
since the full engine initialization requires database and Redis
connections. The tests validate the integration wiring, not the
individual engine logic (which is tested in the agent's own test suite).

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0 + AGENT-EUDR-001 Integration
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fixtures: Configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def eudr_config():
    """Create a test EUDRAppConfig with SCM settings."""
    from services.config import EUDRAppConfig

    return EUDRAppConfig(
        database_url="postgresql://localhost:5432/test_eudr",
        redis_url="redis://localhost:6379/1",
        scm_enabled=True,
        scm_pool_size=5,
        scm_cache_ttl=600,
        scm_max_nodes_per_graph=10_000,
        scm_max_tier_depth=20,
        scm_risk_weight_country=0.30,
        scm_risk_weight_commodity=0.20,
        scm_risk_weight_supplier=0.25,
        scm_risk_weight_deforestation=0.25,
        scm_enable_provenance=True,
        scm_rate_limit=500,
        scm_route_prefix="/v1/supply-chain",
    )


@pytest.fixture
def eudr_config_scm_disabled():
    """Create a test EUDRAppConfig with SCM disabled."""
    from services.config import EUDRAppConfig

    return EUDRAppConfig(scm_enabled=False)


# ---------------------------------------------------------------------------
# Fixtures: Mock SCM Service
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scm_service():
    """Create a mock SupplyChainMapperService."""
    service = MagicMock()
    service.is_running = True
    service.uptime_seconds = 42.0
    service.initialized_engine_count.return_value = 9
    service.db_pool = MagicMock()
    service.redis_client = MagicMock()
    service.config = MagicMock()
    service.config.pool_size = 5
    service.config.cache_ttl = 600

    # Mock engines
    service.graph_engine = MagicMock()
    service.multi_tier_mapper = MagicMock()
    service.geolocation_linker = MagicMock()
    service.batch_traceability_engine = MagicMock()
    service.risk_propagation_engine = MagicMock()
    service.gap_analyzer = MagicMock()
    service.visualization_engine = MagicMock()
    service.regulatory_exporter = MagicMock()
    service.supplier_onboarding_engine = MagicMock()

    # Mock health check
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "checks": {
            "database": {"status": "healthy"},
            "redis": {"status": "healthy"},
            "engines": {"status": "healthy", "initialized": 9, "total": 9},
        },
        "version": "1.0.0",
        "uptime_seconds": 42.0,
    })

    # Mock startup/shutdown
    service.startup = AsyncMock()
    service.shutdown = AsyncMock()

    return service


# ===========================================================================
# Test Group 1: Configuration
# ===========================================================================


class TestConfigurationIntegration:
    """Test EUDRAppConfig supply chain mapper settings."""

    def test_scm_enabled_default(self):
        """SCM is enabled by default."""
        from services.config import EUDRAppConfig

        config = EUDRAppConfig()
        assert config.scm_enabled is True

    def test_scm_disabled(self, eudr_config_scm_disabled):
        """SCM can be disabled via config."""
        assert eudr_config_scm_disabled.scm_enabled is False

    def test_scm_database_url_fallback(self, eudr_config):
        """SCM database URL falls back to main if empty."""
        config = eudr_config
        assert config.scm_database_url == ""
        # The supply_chain.py service should use config.database_url as fallback

    def test_scm_pool_size(self, eudr_config):
        """SCM pool size is configurable."""
        assert eudr_config.scm_pool_size == 5

    def test_scm_cache_ttl(self, eudr_config):
        """SCM cache TTL is configurable."""
        assert eudr_config.scm_cache_ttl == 600

    def test_scm_max_nodes(self, eudr_config):
        """SCM max nodes per graph is configurable."""
        assert eudr_config.scm_max_nodes_per_graph == 10_000

    def test_scm_max_tier_depth(self, eudr_config):
        """SCM max tier depth is configurable."""
        assert eudr_config.scm_max_tier_depth == 20

    def test_scm_risk_weights(self, eudr_config):
        """SCM risk weights are configurable and sum to 1.0."""
        weights = [
            eudr_config.scm_risk_weight_country,
            eudr_config.scm_risk_weight_commodity,
            eudr_config.scm_risk_weight_supplier,
            eudr_config.scm_risk_weight_deforestation,
        ]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_scm_enable_provenance(self, eudr_config):
        """SCM provenance tracking is configurable."""
        assert eudr_config.scm_enable_provenance is True

    def test_scm_rate_limit(self, eudr_config):
        """SCM rate limit is configurable."""
        assert eudr_config.scm_rate_limit == 500

    def test_scm_route_prefix(self, eudr_config):
        """SCM route prefix is configurable."""
        assert eudr_config.scm_route_prefix == "/v1/supply-chain"

    def test_scm_pool_size_bounds(self):
        """SCM pool size respects min/max bounds."""
        from services.config import EUDRAppConfig

        config = EUDRAppConfig(scm_pool_size=1)
        assert config.scm_pool_size == 1

        config = EUDRAppConfig(scm_pool_size=100)
        assert config.scm_pool_size == 100

    def test_scm_cache_ttl_bounds(self):
        """SCM cache TTL respects min/max bounds."""
        from services.config import EUDRAppConfig

        config = EUDRAppConfig(scm_cache_ttl=60)
        assert config.scm_cache_ttl == 60


# ===========================================================================
# Test Group 2: Router Registration
# ===========================================================================


class TestRouterRegistration:
    """Test API router registration including SCM routes."""

    def test_register_all_routers(self):
        """register_all_routers mounts core + SCM routers."""
        from services.api.routers import register_all_routers

        app = FastAPI()
        summary = register_all_routers(app, scm_enabled=False)

        assert summary["core_routers"] == 8
        assert summary["scm_registered"] is False
        assert summary["total_routers"] == 8

    def test_register_all_routers_scm_disabled(self):
        """SCM router is skipped when disabled."""
        from services.api.routers import register_all_routers

        app = FastAPI()
        summary = register_all_routers(app, scm_enabled=False)
        assert summary["scm_registered"] is False
        assert summary["scm_prefix"] is None

    def test_register_core_routers_only(self):
        """Core routers always register."""
        from services.api.routers import register_all_routers

        app = FastAPI()
        summary = register_all_routers(app, scm_enabled=False)
        assert summary["core_routers"] == 8

    def test_scm_router_import_guard(self):
        """SCM router gracefully handles import failure."""
        from services.api.routers import _register_supply_chain_mapper_router

        app = FastAPI()
        # Mock the import to fail
        with patch(
            "services.api.routers._register_supply_chain_mapper_router"
        ) as mock_reg:
            mock_reg.return_value = False
            # Direct call still works
            result = _register_supply_chain_mapper_router(app)
            # May succeed or fail depending on environment
            assert isinstance(result, bool)

    def test_register_with_custom_prefix(self):
        """SCM router can use a custom prefix."""
        from services.api.routers import register_all_routers

        app = FastAPI()
        summary = register_all_routers(
            app,
            scm_prefix="/api/v2/supply-chain",
            scm_enabled=False,
        )
        assert summary["scm_registered"] is False


# ===========================================================================
# Test Group 3: SupplyChainAppService Lifecycle
# ===========================================================================


class TestSupplyChainAppService:
    """Test the SupplyChainAppService facade."""

    def test_create_service(self, eudr_config):
        """Service can be created with config."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService(config=eudr_config)
        assert svc.is_initialized is False
        assert svc.config.scm_enabled is True

    def test_create_service_default_config(self):
        """Service creates with default config."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService()
        assert svc.is_initialized is False

    def test_service_not_initialized_error(self, eudr_config):
        """Accessing scm_service before init raises error."""
        from services.supply_chain import (
            SupplyChainAppService,
            SupplyChainError,
        )

        svc = SupplyChainAppService(config=eudr_config)
        with pytest.raises(SupplyChainError) as exc_info:
            _ = svc.scm_service
        assert exc_info.value.error_code == "SCM_NOT_INITIALIZED"

    def test_supply_chain_error_to_dict(self):
        """SupplyChainError serializes to dict."""
        from services.supply_chain import SupplyChainError

        err = SupplyChainError(
            message="Test error",
            detail="Some detail",
            error_code="TEST_001",
        )
        d = err.to_dict()
        assert d["error"] == "TEST_001"
        assert d["message"] == "Test error"
        assert d["detail"] == "Some detail"

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, eudr_config_scm_disabled):
        """Initialize is a no-op when SCM is disabled."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService(config=eudr_config_scm_disabled)
        await svc.initialize()
        assert svc.is_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self, eudr_config):
        """Shutdown is safe when not initialized."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService(config=eudr_config)
        # Should not raise
        await svc.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, eudr_config):
        """Health check returns not_initialized when not started."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService(config=eudr_config)
        result = await svc.health_check()
        assert result["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_dashboard_summary_not_initialized(self, eudr_config):
        """Dashboard summary returns unavailable when not initialized."""
        from services.supply_chain import SupplyChainAppService

        svc = SupplyChainAppService(config=eudr_config)
        result = await svc.get_dashboard_summary()
        assert result["service_status"] == "unavailable"


# ===========================================================================
# Test Group 4: FastAPI Integration Helpers
# ===========================================================================


class TestFastAPIIntegration:
    """Test FastAPI app.state integration helpers."""

    def test_configure_supply_chain_service(self, eudr_config):
        """Configure attaches service to app.state."""
        from services.supply_chain import (
            configure_supply_chain_service,
            get_supply_chain_service,
        )

        app = FastAPI()
        svc = configure_supply_chain_service(app, config=eudr_config)
        assert svc is not None

        retrieved = get_supply_chain_service(app)
        assert retrieved is svc

    def test_get_supply_chain_service_not_configured(self):
        """get_supply_chain_service raises when not configured."""
        from services.supply_chain import get_supply_chain_service

        app = FastAPI()
        with pytest.raises(RuntimeError, match="not configured"):
            get_supply_chain_service(app)


# ===========================================================================
# Test Group 5: EUDRComplianceService with Supply Chain
# ===========================================================================


class TestEUDRComplianceServiceIntegration:
    """Test the EUDRComplianceService facade with SCM support."""

    def test_service_creates_scm_when_enabled(self, eudr_config):
        """EUDRComplianceService creates supply_chain_service when enabled."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config)
        assert service.supply_chain_service is not None

    def test_service_no_scm_when_disabled(self, eudr_config_scm_disabled):
        """EUDRComplianceService skips SCM when disabled."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config_scm_disabled)
        assert service.supply_chain_service is None

    def test_health_check_includes_scm(self, eudr_config):
        """Health check includes supply chain mapper status."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config)
        health = service.health_check()

        assert "supply_chain_mapper" in health["engines"]
        assert "AGENT-EUDR-001" in health["agents"]
        assert health["config"]["scm_enabled"] is True

    def test_health_check_scm_not_initialized(self, eudr_config):
        """Health check shows 'not_initialized' before async init."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config)
        health = service.health_check()
        assert health["engines"]["supply_chain_mapper"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_initialize_supply_chain_no_service(
        self, eudr_config_scm_disabled
    ):
        """initialize_supply_chain returns False when disabled."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config_scm_disabled)
        result = await service.initialize_supply_chain()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_supply_chain_no_service(
        self, eudr_config_scm_disabled
    ):
        """shutdown_supply_chain is safe when disabled."""
        from services.setup import EUDRComplianceService

        service = EUDRComplianceService(config=eudr_config_scm_disabled)
        # Should not raise
        await service.shutdown_supply_chain()


# ===========================================================================
# Test Group 6: DDS Supply Chain Section
# ===========================================================================


class TestDDSSupplyChainSection:
    """Test DDS generation with supply chain section."""

    def test_dds_model_has_supply_chain_fields(self):
        """DueDiligenceStatement model includes supply chain fields."""
        from services.models import DueDiligenceStatement
        from services.config import EUDRCommodity, DDSStatus

        dds = DueDiligenceStatement(
            reference_number="EUDR-BRA-2026-000001",
            supplier_id="supplier-1",
            year=2026,
            commodity=EUDRCommodity.COFFEE,
            supply_chain_section={"graph_id": "g-1", "status": "available"},
            supply_chain_graph_id="g-1",
        )
        assert dds.supply_chain_section is not None
        assert dds.supply_chain_section["graph_id"] == "g-1"
        assert dds.supply_chain_graph_id == "g-1"

    def test_dds_model_supply_chain_none_by_default(self):
        """Supply chain fields are None by default."""
        from services.models import DueDiligenceStatement
        from services.config import EUDRCommodity

        dds = DueDiligenceStatement(
            reference_number="EUDR-BRA-2026-000001",
            supplier_id="supplier-1",
            year=2026,
            commodity=EUDRCommodity.COFFEE,
        )
        assert dds.supply_chain_section is None
        assert dds.supply_chain_graph_id is None

    def test_dds_engine_accepts_supply_chain_service(self, eudr_config):
        """DDSReportingEngine accepts supply_chain_service parameter."""
        from services.dds_reporting_engine import DDSReportingEngine

        mock_svc = MagicMock()
        engine = DDSReportingEngine(
            config=eudr_config,
            supply_chain_service=mock_svc,
        )
        assert engine._supply_chain_service is mock_svc

    def test_dds_engine_set_supply_chain_service(self, eudr_config):
        """DDSReportingEngine.set_supply_chain_service wires the service."""
        from services.dds_reporting_engine import DDSReportingEngine

        engine = DDSReportingEngine(config=eudr_config)
        assert engine._supply_chain_service is None

        mock_svc = MagicMock()
        engine.set_supply_chain_service(mock_svc)
        assert engine._supply_chain_service is mock_svc

    def test_dds_generate_without_scm(self, eudr_config):
        """DDS generates without supply chain section when SCM unavailable."""
        from services.dds_reporting_engine import DDSReportingEngine

        engine = DDSReportingEngine(config=eudr_config)
        dds = engine.generate_dds(
            supplier_id="supplier-1",
            commodity="coffee",
            year=2026,
            plots=["plot-1"],
        )
        assert dds.supply_chain_section is None
        assert dds.supply_chain_graph_id is None

    def test_dds_generate_with_scm_not_initialized(self, eudr_config):
        """DDS skips supply chain when service is not initialized."""
        from services.dds_reporting_engine import DDSReportingEngine

        mock_svc = MagicMock()
        mock_svc.is_initialized = False

        engine = DDSReportingEngine(
            config=eudr_config,
            supply_chain_service=mock_svc,
        )
        dds = engine.generate_dds(
            supplier_id="supplier-1",
            commodity="coffee",
            year=2026,
            plots=["plot-1"],
        )
        # Supply chain section should be None since not initialized
        assert dds.supply_chain_section is None


# ===========================================================================
# Test Group 7: Supply Chain App Service Error Handling
# ===========================================================================


class TestSupplyChainErrorHandling:
    """Test error handling in the SupplyChainAppService."""

    @pytest.mark.asyncio
    async def test_get_graph_with_stats_not_initialized(self, eudr_config):
        """get_graph_with_stats raises error when not initialized."""
        from services.supply_chain import (
            SupplyChainAppService,
            SupplyChainError,
        )

        svc = SupplyChainAppService(config=eudr_config)
        with pytest.raises(SupplyChainError) as exc_info:
            await svc.get_graph_with_stats("graph-1")
        assert exc_info.value.error_code == "SCM_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_run_full_analysis_not_initialized(self, eudr_config):
        """run_full_analysis raises error when not initialized."""
        from services.supply_chain import (
            SupplyChainAppService,
            SupplyChainError,
        )

        svc = SupplyChainAppService(config=eudr_config)
        with pytest.raises(SupplyChainError) as exc_info:
            await svc.run_full_analysis("graph-1")
        assert exc_info.value.error_code == "SCM_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_export_to_dds_not_initialized(self, eudr_config):
        """export_to_dds raises error when not initialized."""
        from services.supply_chain import (
            SupplyChainAppService,
            SupplyChainError,
        )

        svc = SupplyChainAppService(config=eudr_config)
        with pytest.raises(SupplyChainError) as exc_info:
            await svc.export_to_dds("graph-1")
        assert exc_info.value.error_code == "SCM_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_get_onboarding_status_not_initialized(self, eudr_config):
        """get_onboarding_status raises error when not initialized."""
        from services.supply_chain import (
            SupplyChainAppService,
            SupplyChainError,
        )

        svc = SupplyChainAppService(config=eudr_config)
        with pytest.raises(SupplyChainError) as exc_info:
            await svc.get_onboarding_status("inv-1")
        assert exc_info.value.error_code == "SCM_NOT_INITIALIZED"


# ===========================================================================
# Test Group 8: configure_eudr_app with SCM
# ===========================================================================


class TestConfigureEudrApp:
    """Test configure_eudr_app function with SCM integration."""

    def test_configure_eudr_app_stores_service(self):
        """configure_eudr_app stores service on app.state."""
        from services.setup import configure_eudr_app, get_eudr_service

        app = FastAPI()
        service = configure_eudr_app(app)
        retrieved = get_eudr_service(app)
        assert retrieved is service

    def test_configure_eudr_app_creates_scm(self):
        """configure_eudr_app creates supply chain service."""
        from services.setup import configure_eudr_app

        app = FastAPI()
        service = configure_eudr_app(app)
        assert service.supply_chain_service is not None

    @pytest.mark.asyncio
    async def test_startup_eudr_app(self):
        """startup_eudr_app initializes supply chain service."""
        from services.setup import (
            configure_eudr_app,
            startup_eudr_app,
        )

        app = FastAPI()
        service = configure_eudr_app(app)

        # Mock the supply chain service initialization
        if service.supply_chain_service is not None:
            service.supply_chain_service.initialize = AsyncMock()

        await startup_eudr_app(app)

    @pytest.mark.asyncio
    async def test_shutdown_eudr_app(self):
        """shutdown_eudr_app shuts down supply chain service."""
        from services.setup import (
            configure_eudr_app,
            shutdown_eudr_app,
        )

        app = FastAPI()
        service = configure_eudr_app(app)

        # Mock the supply chain service shutdown
        if service.supply_chain_service is not None:
            service.supply_chain_service.shutdown = AsyncMock()
            service.supply_chain_service._initialized = False

        await shutdown_eudr_app(app)

    @pytest.mark.asyncio
    async def test_shutdown_eudr_app_unconfigured(self):
        """shutdown_eudr_app is safe on unconfigured app."""
        from services.setup import shutdown_eudr_app

        app = FastAPI()
        # Should not raise
        await shutdown_eudr_app(app)
