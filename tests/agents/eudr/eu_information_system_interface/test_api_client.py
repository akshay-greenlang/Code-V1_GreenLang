# -*- coding: utf-8 -*-
"""
Unit tests for APIClient engine - AGENT-EUDR-036

Tests circuit breaker logic, retry handling, API call recording,
health checking, and stub mode operation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import time

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.api_client import (
    APIClient,
    CircuitState,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def client() -> APIClient:
    """Create an APIClient instance in stub mode."""
    config = EUInformationSystemInterfaceConfig()
    return APIClient(config=config, provenance=ProvenanceTracker())


class TestInitialization:
    """Test APIClient initialization."""

    def test_initial_state(self, client):
        assert client.circuit_state == "closed"
        assert client.total_calls == 0
        assert client.total_errors == 0


class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_initial_state_closed(self, client):
        assert client._circuit_state == CircuitState.CLOSED

    def test_on_success_resets_failure_count(self, client):
        client._failure_count = 3
        client._on_success()
        assert client._failure_count == 0

    def test_on_failure_increments(self, client):
        client._on_failure()
        assert client._failure_count == 1

    def test_failure_threshold_opens_circuit(self, client):
        for _ in range(5):
            client._on_failure()
        assert client._circuit_state == CircuitState.OPEN

    def test_open_circuit_blocks_requests(self, client):
        for _ in range(5):
            client._on_failure()
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            client._check_circuit_breaker()

    def test_half_open_after_reset_timeout(self, client):
        for _ in range(5):
            client._on_failure()
        # Simulate time passing by setting last_failure_time far back
        client._last_failure_time = time.monotonic() - 120
        client._check_circuit_breaker()
        assert client._circuit_state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self, client):
        client._circuit_state = CircuitState.HALF_OPEN
        client._on_success()
        assert client._circuit_state == CircuitState.CLOSED

    def test_circuit_state_property(self, client):
        assert client.circuit_state == "closed"


class TestStubMode:
    """Test APIClient in stub mode (no httpx)."""

    @pytest.mark.asyncio
    async def test_submit_dds_stub(self, client):
        result = await client.submit_dds({"dds_id": "dds-001"})
        assert result["status_code"] == 200
        assert result["success"] is True
        assert "stub" in result

    @pytest.mark.asyncio
    async def test_check_status_stub(self, client):
        result = await client.check_dds_status("EUDR-REF-001")
        assert result["status_code"] == 200
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_register_operator_stub(self, client):
        result = await client.register_operator({"operator_id": "op-001"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_withdraw_dds_stub(self, client):
        result = await client.withdraw_dds("EUDR-REF-001", "Test reason")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_health_ping_stub(self, client):
        result = await client.health_ping()
        assert result["status_code"] == 200


class TestAPICallRecording:
    """Test API call recording for audit trail."""

    @pytest.mark.asyncio
    async def test_call_count_increments(self, client):
        await client.submit_dds({"dds_id": "dds-001"})
        assert client.total_calls == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_tracked(self, client):
        await client.submit_dds({"dds_id": "dds-001"})
        await client.check_dds_status("EUDR-REF-001")
        await client.register_operator({"operator_id": "op-001"})
        assert client.total_calls == 3

    @pytest.mark.asyncio
    async def test_call_history_maintained(self, client):
        await client.submit_dds({"dds_id": "dds-001"})
        assert len(client._call_history) >= 1


class TestInitializeAndClose:
    """Test HTTP client lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize(self, client):
        await client.initialize()
        # Whether httpx is available or not, this should not error
        # In test environment httpx may or may not be installed

    @pytest.mark.asyncio
    async def test_close(self, client):
        await client.close()
        # Should be safe to call even without initialization

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client):
        await client.close()
        await client.close()


class TestHealthCheck:
    """Test APIClient.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        health = await client.health_check()
        assert health["engine"] == "APIClient"
        assert health["status"] == "available"
        assert health["circuit_breaker"] == "closed"
        assert "total_calls" in health
        assert "config" in health
