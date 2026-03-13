# -*- coding: utf-8 -*-
"""
Unit tests for AuthorityCommunicationManagerService (setup.py) - AGENT-EUDR-040

Tests service facade initialization, engine loading, startup, shutdown,
singleton pattern, health check, lifespan context manager, communication
management, deadline calculation, and engine delegation.

25+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.setup import (
    AuthorityCommunicationManagerService,
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
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def service(config):
    return AuthorityCommunicationManagerService(config=config)


# ====================================================================
# Initialization Tests
# ====================================================================


class TestServiceInit:
    """Test service facade initialization."""

    def test_creates_instance(self, config):
        svc = AuthorityCommunicationManagerService(config=config)
        assert svc is not None

    def test_config_set(self, service, config):
        assert service.config is config

    def test_not_initialized_by_default(self, service):
        assert service.is_initialized is False

    def test_default_config_used_if_none(self):
        svc = AuthorityCommunicationManagerService()
        assert svc.config is not None

    def test_engine_count_zero_before_startup(self, service):
        assert service.engine_count == 0

    def test_db_pool_none_before_startup(self, service):
        assert service._db_pool is None

    def test_redis_none_before_startup(self, service):
        assert service._redis is None

    def test_communications_empty(self, service):
        assert service.communication_count == 0

    def test_provenance_initialized(self, service):
        assert service._provenance is not None


# ====================================================================
# Engine Loading Tests
# ====================================================================


class TestEngineLoading:
    """Test engine initialization via _init_engines."""

    def test_init_engines_loads_all(self, service):
        service._init_engines()
        assert service.engine_count == 7

    def test_request_handler_loaded(self, service):
        service._init_engines()
        assert service._request_handler is not None

    def test_inspection_coordinator_loaded(self, service):
        service._init_engines()
        assert service._inspection_coordinator is not None

    def test_non_compliance_manager_loaded(self, service):
        service._init_engines()
        assert service._non_compliance_manager is not None

    def test_appeal_processor_loaded(self, service):
        service._init_engines()
        assert service._appeal_processor is not None

    def test_document_exchange_loaded(self, service):
        service._init_engines()
        assert service._document_exchange is not None

    def test_notification_router_loaded(self, service):
        service._init_engines()
        assert service._notification_router is not None

    def test_template_engine_loaded(self, service):
        service._init_engines()
        assert service._template_engine is not None

    def test_engines_dict_populated(self, service):
        service._init_engines()
        expected_engines = [
            "request_handler",
            "inspection_coordinator",
            "non_compliance_manager",
            "appeal_processor",
            "document_exchange",
            "notification_router",
            "template_engine",
        ]
        for name in expected_engines:
            assert name in service._engines

    def test_get_engine_by_name(self, service):
        service._init_engines()
        engine = service.get_engine("request_handler")
        assert engine is not None

    def test_get_engine_not_found(self, service):
        service._init_engines()
        engine = service.get_engine("nonexistent")
        assert engine is None


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

    @pytest.mark.asyncio
    async def test_startup_without_redis(self, service):
        """Startup works even without Redis."""
        await service.startup()
        assert service._redis is None
        assert service.is_initialized is True


# ====================================================================
# Singleton Tests
# ====================================================================


class TestSingleton:
    """Test thread-safe singleton pattern."""

    def test_get_service_returns_instance(self):
        svc = get_service()
        assert isinstance(svc, AuthorityCommunicationManagerService)

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
        cfg = AuthorityCommunicationManagerConfig()
        svc = get_service(config=cfg)
        assert svc.config is cfg


# ====================================================================
# Communication Management Tests
# ====================================================================


class TestCommunicationManagement:
    """Test in-memory communication management."""

    @pytest.mark.asyncio
    async def test_create_communication(self, service):
        await service.startup()
        result = await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="DDS Verification Request",
        )
        assert "communication_id" in result
        assert result["status"] == "pending"
        assert result["member_state"] == "DE"
        assert service.communication_count == 1

    @pytest.mark.asyncio
    async def test_get_communication(self, service):
        await service.startup()
        created = await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test",
        )
        result = await service.get_communication(created["communication_id"])
        assert result is not None
        assert result["communication_id"] == created["communication_id"]

    @pytest.mark.asyncio
    async def test_get_communication_not_found(self, service):
        result = await service.get_communication("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_respond_to_communication(self, service):
        await service.startup()
        created = await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test",
        )
        result = await service.respond_to_communication(
            communication_id=created["communication_id"],
            responder_id="OP-001",
            body="Here is our response.",
        )
        assert result["status"] == "responded"

    @pytest.mark.asyncio
    async def test_respond_not_found_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            await service.respond_to_communication(
                communication_id="nonexistent",
                responder_id="OP-001",
                body="Test response body.",
            )

    @pytest.mark.asyncio
    async def test_list_pending_communications(self, service):
        await service.startup()
        await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test 1",
        )
        await service.create_communication(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            member_state="FR",
            communication_type="general_correspondence",
            subject="Test 2",
        )
        result = await service.list_pending_communications()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_pending_with_operator_filter(self, service):
        await service.startup()
        await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test 1",
        )
        await service.create_communication(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            member_state="FR",
            communication_type="general_correspondence",
            subject="Test 2",
        )
        result = await service.list_pending_communications(operator_id="OP-001")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_communication_has_provenance_hash(self, service):
        await service.startup()
        result = await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_communication_has_deadline(self, service):
        await service.startup()
        result = await service.create_communication(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="Test",
            priority="urgent",
        )
        assert result["deadline"] is not None


# ====================================================================
# Deadline Calculation Tests
# ====================================================================


class TestDeadlineCalculation:
    """Test deadline calculation based on priority."""

    def test_urgent_deadline(self, service):
        deadline = service._calculate_deadline("urgent")
        assert deadline is not None

    def test_normal_deadline(self, service):
        deadline = service._calculate_deadline("normal")
        assert deadline is not None

    def test_routine_deadline(self, service):
        deadline = service._calculate_deadline("routine")
        assert deadline is not None

    def test_high_deadline(self, service):
        deadline = service._calculate_deadline("high")
        assert deadline is not None

    def test_low_deadline(self, service):
        deadline = service._calculate_deadline("low")
        assert deadline is not None

    def test_unknown_priority_fallback(self, service):
        deadline = service._calculate_deadline("unknown")
        assert deadline is not None


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
        assert health["agent_id"] == "GL-EUDR-ACM-040"

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
    async def test_health_check_stores_section(self, service):
        await service.startup()
        health = await service.health_check()
        assert "stores" in health
        assert "communications" in health["stores"]

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

    @pytest.mark.asyncio
    async def test_health_check_member_states(self, service):
        await service.startup()
        health = await service.health_check()
        assert health["member_states"] == 27


# ====================================================================
# Engine Delegation Tests
# ====================================================================


class TestEngineDelegation:
    """Test service methods raise RuntimeError when engine not loaded."""

    @pytest.mark.asyncio
    async def test_handle_information_request_no_engine(self, service):
        with pytest.raises(RuntimeError, match="RequestHandler"):
            await service.handle_information_request(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                request_type="dds_verification",
                items_requested=["DDS Statement"],
            )

    @pytest.mark.asyncio
    async def test_schedule_inspection_no_engine(self, service):
        with pytest.raises(RuntimeError, match="InspectionCoordinator"):
            await service.schedule_inspection(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                inspection_type="on_the_spot",
                scheduled_date="2026-04-01T09:00:00Z",
            )

    @pytest.mark.asyncio
    async def test_record_violation_no_engine(self, service):
        with pytest.raises(RuntimeError, match="NonComplianceManager"):
            await service.record_violation(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                violation_type="missing_dds",
                severity="minor",
                description="Test violation",
            )

    @pytest.mark.asyncio
    async def test_file_appeal_no_engine(self, service):
        with pytest.raises(RuntimeError, match="AppealProcessor"):
            await service.file_appeal(
                non_compliance_id="NC-001",
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                grounds="Test grounds for the appeal submission.",
            )

    @pytest.mark.asyncio
    async def test_upload_document_no_engine(self, service):
        with pytest.raises(RuntimeError, match="DocumentExchange"):
            await service.upload_document(
                communication_id="COMM-001",
                document_type="certificate",
                title="Test",
                content=b"test",
                uploaded_by="OP-001",
            )

    @pytest.mark.asyncio
    async def test_download_document_no_engine(self, service):
        with pytest.raises(RuntimeError, match="DocumentExchange"):
            await service.download_document(document_id="DOC-001")

    @pytest.mark.asyncio
    async def test_send_notification_no_engine(self, service):
        with pytest.raises(RuntimeError, match="NotificationRouter"):
            await service.send_notification(
                communication_id="COMM-001",
                channel="email",
                recipient_type="operator",
                recipient_id="OP-001",
            )

    @pytest.mark.asyncio
    async def test_render_template_no_engine(self, service):
        with pytest.raises(RuntimeError, match="TemplateEngine"):
            await service.render_template(
                template_name="test",
                language="en",
                variables={},
            )

    @pytest.mark.asyncio
    async def test_list_templates_no_engine_returns_empty(self, service):
        result = await service.list_templates()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_authorities(self, service):
        result = await service.get_authorities()
        assert len(result) == 27

    @pytest.mark.asyncio
    async def test_get_authorities_filtered(self, service):
        result = await service.get_authorities(member_state="DE")
        assert len(result) == 1
        assert result[0]["member_state"] == "DE"

    @pytest.mark.asyncio
    async def test_get_authorities_filtered_not_found(self, service):
        result = await service.get_authorities(member_state="XX")
        assert len(result) == 0
