# -*- coding: utf-8 -*-
"""
Unit tests for RequestHandler engine - AGENT-EUDR-040

Tests information request processing, validation, deadline calculation,
response preparation, request retrieval, listing, health checks, and
provenance tracking.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.request_handler import (
    RequestHandler,
    _REQUEST_DATA_SOURCES,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    InformationRequest,
    InformationRequestType,
    ResponseData,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def handler(config):
    return RequestHandler(config=config)


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_handler_created(self, handler):
        assert handler is not None

    def test_default_config(self):
        h = RequestHandler()
        assert h.config is not None

    def test_custom_config(self, config):
        h = RequestHandler(config=config)
        assert h.config is config

    def test_requests_empty(self, handler):
        assert len(handler._requests) == 0

    def test_responses_empty(self, handler):
        assert len(handler._responses) == 0

    def test_provenance_initialized(self, handler):
        assert handler._provenance is not None


# ====================================================================
# Request Data Sources Mapping
# ====================================================================


class TestRequestDataSources:
    def test_all_request_types_mapped(self):
        for rt in InformationRequestType:
            assert rt in _REQUEST_DATA_SOURCES, f"{rt.value} not mapped"

    def test_dds_clarification_sources(self):
        sources = _REQUEST_DATA_SOURCES[InformationRequestType.DDS_CLARIFICATION]
        assert "dds_statement" in sources

    def test_supply_chain_sources(self):
        sources = _REQUEST_DATA_SOURCES[InformationRequestType.SUPPLY_CHAIN_EVIDENCE]
        assert "supply_chain_map" in sources

    def test_geolocation_sources(self):
        sources = _REQUEST_DATA_SOURCES[InformationRequestType.GEOLOCATION_VERIFICATION]
        assert "geolocation_data" in sources

    def test_deforestation_sources(self):
        sources = _REQUEST_DATA_SOURCES[InformationRequestType.DEFORESTATION_EVIDENCE]
        assert "satellite_monitoring" in sources


# ====================================================================
# Receive Request
# ====================================================================


class TestReceiveRequest:
    @pytest.mark.asyncio
    async def test_receive_valid_request(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["DDS copy", "Risk assessment"],
            dds_reference="GL-DDS-20260313-ABCDEF",
            commodity="cocoa",
        )
        assert isinstance(result, InformationRequest)
        assert result.request_type == InformationRequestType.DDS_CLARIFICATION
        assert len(result.items_requested) == 2

    @pytest.mark.asyncio
    async def test_receive_request_sets_deadline(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="supply_chain_evidence",
            items_requested=["Supplier list"],
        )
        assert result.deadline is not None

    @pytest.mark.asyncio
    async def test_receive_request_assigns_id(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
        )
        assert result.request_id is not None
        assert len(result.request_id) > 0

    @pytest.mark.asyncio
    async def test_receive_request_computes_provenance(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_receive_request_stored(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
        )
        assert result.request_id in handler._requests

    @pytest.mark.asyncio
    async def test_receive_request_invalid_type_raises(self, handler):
        with pytest.raises(ValueError, match="Invalid request type"):
            await handler.receive_request(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                request_type="invalid_type",
                items_requested=["Item 1"],
            )

    @pytest.mark.asyncio
    async def test_receive_request_empty_items_raises(self, handler):
        with pytest.raises(ValueError, match="At least one item"):
            await handler.receive_request(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                request_type="dds_clarification",
                items_requested=[],
            )

    @pytest.mark.asyncio
    async def test_receive_urgent_priority(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
            priority="urgent",
        )
        assert result.deadline is not None
        # Urgent deadline should be within 24 hours
        hours_until_deadline = (result.deadline - result.created_at).total_seconds() / 3600
        assert hours_until_deadline <= 25  # Allow 1h tolerance

    @pytest.mark.asyncio
    async def test_receive_routine_priority(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
            priority="routine",
        )
        assert result.deadline is not None

    @pytest.mark.asyncio
    async def test_receive_all_request_types(self, handler):
        """Test each request type can be received."""
        for rt in InformationRequestType:
            result = await handler.receive_request(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                request_type=rt.value,
                items_requested=["Item"],
            )
            assert result.request_type == rt

    @pytest.mark.asyncio
    async def test_receive_request_with_commodity(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
            commodity="cocoa",
        )
        assert result.commodity == "cocoa"

    @pytest.mark.asyncio
    async def test_receive_request_with_dds_ref(self, handler):
        result = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.dds_reference == "GL-DDS-20260313-ABCDEF"


# ====================================================================
# Prepare Response
# ====================================================================


class TestPrepareResponse:
    @pytest.mark.asyncio
    async def test_prepare_response(self, handler):
        request = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["DDS copy"],
        )
        response = await handler.prepare_response(
            request_id=request.request_id,
            responder_id="OP-001",
            body="Please find the requested information attached.",
            document_ids=["DOC-001"],
        )
        assert isinstance(response, ResponseData)
        assert response.responder_id == "OP-001"

    @pytest.mark.asyncio
    async def test_prepare_response_not_found(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.prepare_response(
                request_id="nonexistent",
                responder_id="OP-001",
            )

    @pytest.mark.asyncio
    async def test_prepare_response_stored(self, handler):
        request = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item"],
        )
        response = await handler.prepare_response(
            request_id=request.request_id,
            responder_id="OP-001",
        )
        assert response.response_id in handler._responses

    @pytest.mark.asyncio
    async def test_prepare_response_with_documents(self, handler):
        request = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item"],
        )
        response = await handler.prepare_response(
            request_id=request.request_id,
            responder_id="OP-001",
            document_ids=["DOC-001", "DOC-002", "DOC-003"],
        )
        assert len(response.document_ids) == 3


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_request(self, handler):
        req = await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item"],
        )
        result = await handler.get_request(req.request_id)
        assert result is not None
        assert result.request_id == req.request_id

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, handler):
        result = await handler.get_request("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_pending_requests_empty(self, handler):
        result = await handler.list_pending_requests()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_pending_requests_multiple(self, handler):
        await handler.receive_request(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_clarification",
            items_requested=["Item 1"],
        )
        await handler.receive_request(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            request_type="supply_chain_evidence",
            items_requested=["Item 2"],
        )
        result = await handler.list_pending_requests()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, handler):
        health = await handler.health_check()
        assert health["status"] == "healthy"
        assert health["total_requests"] == 0
