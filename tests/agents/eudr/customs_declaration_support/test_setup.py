# -*- coding: utf-8 -*-
"""
Unit tests for CustomsDeclarationService (setup.py) - AGENT-EUDR-039

Tests service facade initialization, engine loading, startup, shutdown,
singleton pattern, health check, and lifespan context manager.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.setup import (
    CustomsDeclarationService,
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
    return CustomsDeclarationConfig()


@pytest.fixture
def service(config):
    return CustomsDeclarationService(config=config)


# ====================================================================
# Initialization Tests
# ====================================================================


class TestCustomsDeclarationServiceInit:
    """Test service facade initialization."""

    def test_creates_instance(self, config):
        svc = CustomsDeclarationService(config=config)
        assert svc is not None

    def test_config_set(self, service, config):
        assert service.config is config

    def test_not_initialized_by_default(self, service):
        assert service.is_initialized is False

    def test_default_config_used_if_none(self):
        svc = CustomsDeclarationService()
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

    def test_cn_code_mapper_loaded(self, service):
        service._init_engines()
        assert service._cn_code_mapper is not None

    def test_hs_code_validator_loaded(self, service):
        service._init_engines()
        assert service._hs_code_validator is not None

    def test_declaration_generator_loaded(self, service):
        service._init_engines()
        assert service._declaration_generator is not None

    def test_origin_validator_loaded(self, service):
        service._init_engines()
        assert service._origin_validator is not None

    def test_value_calculator_loaded(self, service):
        service._init_engines()
        assert service._value_calculator is not None

    def test_compliance_checker_loaded(self, service):
        service._init_engines()
        assert service._compliance_checker is not None

    def test_customs_interface_loaded(self, service):
        service._init_engines()
        assert service._customs_interface is not None

    def test_engines_dict_populated(self, service):
        service._init_engines()
        expected_engines = [
            "cn_code_mapper",
            "hs_code_validator",
            "declaration_generator",
            "origin_validator",
            "value_calculator",
            "compliance_checker",
            "customs_interface",
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
        assert service._db_pool is None
        assert service.is_initialized is True


# ====================================================================
# Singleton Tests
# ====================================================================


class TestSingleton:
    """Test thread-safe singleton pattern."""

    def test_get_service_returns_instance(self):
        svc = get_service()
        assert isinstance(svc, CustomsDeclarationService)

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
        cfg = CustomsDeclarationConfig()
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
        assert health["agent_id"] == "GL-EUDR-CDS-039"

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
    async def test_create_declaration_no_engine(self, service):
        with pytest.raises(RuntimeError, match="DeclarationGenerator"):
            await service.create_declaration(
                operator_id="OP-001",
                commodities=["cocoa"],
                country_of_origin="CI",
            )

    @pytest.mark.asyncio
    async def test_map_cn_codes_no_engine(self, service):
        with pytest.raises(RuntimeError, match="CNCodeMapper"):
            await service.map_cn_codes(commodity="cocoa")

    @pytest.mark.asyncio
    async def test_validate_hs_code_no_engine(self, service):
        with pytest.raises(RuntimeError, match="HSCodeValidator"):
            await service.validate_hs_code(hs_code="180100")

    @pytest.mark.asyncio
    async def test_calculate_tariff_no_engine(self, service):
        with pytest.raises(RuntimeError, match="ValueCalculator"):
            await service.calculate_tariff(
                declaration_id="DECL-001",
                cn_code="18010000",
                customs_value=25000.00,
                quantity=10000.00,
                origin_country="CI",
            )

    @pytest.mark.asyncio
    async def test_verify_origin_no_engine(self, service):
        with pytest.raises(RuntimeError, match="OriginValidator"):
            await service.verify_origin(
                declaration_id="DECL-001",
                declared_origin="CI",
                supply_chain_origins=["CI"],
                dds_reference="REF-001",
            )

    @pytest.mark.asyncio
    async def test_run_compliance_check_no_engine(self, service):
        with pytest.raises(RuntimeError, match="ComplianceChecker"):
            await service.run_compliance_check(
                declaration_id="DECL-001",
                dds_reference="REF-001",
                cn_codes=["18010000"],
                declared_origin="CI",
            )

    @pytest.mark.asyncio
    async def test_submit_declaration_no_engine(self, service):
        with pytest.raises(RuntimeError, match="CustomsInterface"):
            await service.submit_declaration(
                declaration_id="DECL-001",
                system="ncts",
            )

    @pytest.mark.asyncio
    async def test_get_declaration_no_engine(self, service):
        result = await service.get_declaration("DECL-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_declarations_no_engine(self, service):
        result = await service.list_declarations()
        assert result == []
