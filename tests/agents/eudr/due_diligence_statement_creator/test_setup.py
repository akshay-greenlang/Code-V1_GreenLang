# -*- coding: utf-8 -*-
"""
Unit tests for DDSCreatorService (setup.py) - AGENT-EUDR-037

Tests service facade initialization, engine loading, startup, shutdown,
singleton pattern, health check, and lifespan context manager.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.setup import (
    DDSCreatorService,
    get_service,
    reset_service,
)


@pytest.fixture(autouse=True)
def _reset_service_singleton():
    """Reset the service singleton before/after each test."""
    reset_service()
    yield
    reset_service()


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def service(config):
    return DDSCreatorService(config=config)


# ====================================================================
# Initialization Tests
# ====================================================================


class TestDDSCreatorServiceInit:
    """Test service facade initialization."""

    def test_creates_instance(self, config):
        svc = DDSCreatorService(config=config)
        assert svc is not None

    def test_config_set(self, service, config):
        assert service.config is config

    def test_not_initialized_by_default(self, service):
        assert service.is_initialized is False

    def test_default_config_used_if_none(self):
        svc = DDSCreatorService()
        assert svc.config is not None

    def test_engine_count_zero_before_startup(self, service):
        assert service.engine_count == 0

    def test_db_pool_none_before_startup(self, service):
        assert service._db_pool is None

    def test_redis_none_before_startup(self, service):
        assert service._redis is None


# ====================================================================
# Engine Loading Tests
# ====================================================================


class TestEngineLoading:
    """Test engine initialization via _init_engines."""

    def test_init_engines_loads_all(self, service):
        service._init_engines()
        assert service.engine_count == 7

    def test_assembler_loaded(self, service):
        service._init_engines()
        assert service._assembler is not None

    def test_geolocation_formatter_loaded(self, service):
        service._init_engines()
        assert service._geolocation_formatter is not None

    def test_risk_integrator_loaded(self, service):
        service._init_engines()
        assert service._risk_integrator is not None

    def test_supply_chain_compiler_loaded(self, service):
        service._init_engines()
        assert service._supply_chain_compiler is not None

    def test_compliance_validator_loaded(self, service):
        service._init_engines()
        assert service._compliance_validator is not None

    def test_document_packager_loaded(self, service):
        service._init_engines()
        assert service._document_packager is not None

    def test_version_controller_loaded(self, service):
        service._init_engines()
        assert service._version_controller is not None

    def test_engines_dict_populated(self, service):
        service._init_engines()
        expected_engines = [
            "statement_assembler",
            "geolocation_formatter",
            "risk_data_integrator",
            "supply_chain_compiler",
            "compliance_validator",
            "document_packager",
            "version_controller",
        ]
        for name in expected_engines:
            assert name in service._engines


# ====================================================================
# Startup / Shutdown Tests
# ====================================================================


class TestStartupShutdown:
    """Test startup and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_startup_initializes(self, service):
        await service.startup()
        assert service.is_initialized is True
        assert service.engine_count == 7

    @pytest.mark.asyncio
    async def test_shutdown_deinitializes(self, service):
        await service.startup()
        await service.shutdown()
        assert service.is_initialized is False

    @pytest.mark.asyncio
    async def test_startup_without_db(self, service):
        """Startup works even without DB."""
        await service.startup()
        assert service._db_pool is None  # no psycopg_pool in test
        assert service.is_initialized is True


# ====================================================================
# Singleton Tests
# ====================================================================


class TestSingleton:
    """Test thread-safe singleton pattern."""

    def test_get_service_returns_instance(self):
        svc = get_service()
        assert isinstance(svc, DDSCreatorService)

    def test_get_service_returns_same_instance(self):
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_reset_service_clears(self):
        svc1 = get_service()
        reset_service()
        svc2 = get_service()
        assert svc1 is not svc2

    def test_get_service_with_config(self):
        cfg = DDSCreatorConfig()
        svc = get_service(config=cfg)
        assert svc.config is cfg


# ====================================================================
# Health Check Tests
# ====================================================================


class TestHealthCheck:
    """Test comprehensive health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, service):
        await service.startup()
        health = await service.health_check()
        assert isinstance(health, dict)

    @pytest.mark.asyncio
    async def test_health_check_agent_id(self, service):
        await service.startup()
        health = await service.health_check()
        assert health["agent_id"] == "GL-EUDR-DDSC-037"

    @pytest.mark.asyncio
    async def test_health_check_version(self, service):
        await service.startup()
        health = await service.health_check()
        assert health["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_engines_section(self, service):
        await service.startup()
        health = await service.health_check()
        assert "engines" in health
        assert len(health["engines"]) == 7

    @pytest.mark.asyncio
    async def test_health_check_connections_section(self, service):
        await service.startup()
        health = await service.health_check()
        assert "connections" in health
        assert "postgresql" in health["connections"]
        assert "redis" in health["connections"]

    @pytest.mark.asyncio
    async def test_health_check_timestamp(self, service):
        await service.startup()
        health = await service.health_check()
        assert health["timestamp"] is not None

    @pytest.mark.asyncio
    async def test_health_check_initialized_flag(self, service):
        await service.startup()
        health = await service.health_check()
        assert health["initialized"] is True

    @pytest.mark.asyncio
    async def test_health_check_before_startup(self, service):
        health = await service.health_check()
        assert health["initialized"] is False


# ====================================================================
# Engine Delegate Tests (no engine loaded raises RuntimeError)
# ====================================================================


class TestEngineDelegation:
    """Test service methods raise RuntimeError when engine missing."""

    @pytest.mark.asyncio
    async def test_create_statement_no_engine(self, service):
        with pytest.raises(RuntimeError, match="StatementAssembler"):
            await service.create_statement(
                operator_id="OP-001",
                operator_name="Test",
                commodities=["cocoa"],
            )

    @pytest.mark.asyncio
    async def test_format_geolocation_no_engine(self, service):
        with pytest.raises(RuntimeError, match="GeolocationFormatter"):
            await service.format_geolocation(
                statement_id="DDS-001",
                plot_id="PLT-001",
                latitude=5.0,
                longitude=-3.0,
            )

    @pytest.mark.asyncio
    async def test_integrate_risk_no_engine(self, service):
        with pytest.raises(RuntimeError, match="RiskDataIntegrator"):
            await service.integrate_risk(
                statement_id="DDS-001",
                risk_id="R-001",
                source_agent="EUDR-016",
                risk_category="country",
            )

    @pytest.mark.asyncio
    async def test_compile_supply_chain_no_engine(self, service):
        with pytest.raises(RuntimeError, match="SupplyChainCompiler"):
            await service.compile_supply_chain(
                statement_id="DDS-001",
                supply_chain_id="SC-001",
                commodity="cocoa",
            )

    @pytest.mark.asyncio
    async def test_validate_statement_no_engine(self, service):
        with pytest.raises(RuntimeError, match="ComplianceValidator"):
            await service.validate_statement("DDS-001")

    @pytest.mark.asyncio
    async def test_add_document_no_engine(self, service):
        with pytest.raises(RuntimeError, match="DocumentPackager"):
            await service.add_document(
                statement_id="DDS-001",
                document_type="certificate_of_origin",
                filename="cert.pdf",
            )

    @pytest.mark.asyncio
    async def test_apply_signature_no_engine(self, service):
        with pytest.raises(RuntimeError, match="VersionController"):
            await service.apply_signature(
                statement_id="DDS-001",
                signer_name="John Smith",
            )

    @pytest.mark.asyncio
    async def test_get_statement_no_engine(self, service):
        result = await service.get_statement("DDS-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_statements_no_engine(self, service):
        result = await service.list_statements()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_versions_no_engine(self, service):
        result = await service.get_versions("DDS-001")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latest_version_no_engine(self, service):
        result = await service.get_latest_version("DDS-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_amendments_no_engine(self, service):
        result = await service.get_amendments("DDS-001")
        assert result == []
