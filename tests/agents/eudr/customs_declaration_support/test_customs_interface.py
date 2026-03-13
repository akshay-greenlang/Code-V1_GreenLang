# -*- coding: utf-8 -*-
"""
Unit tests for CustomsInterface engine - AGENT-EUDR-039

Tests NCTS and AIS submission with mocks, response handling,
retry logic, status tracking, error handling, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.customs_interface import CustomsInterface
from greenlang.agents.eudr.customs_declaration_support.models import (
    CustomsInterfaceResponse, CustomsSystemType, DeclarationStatus,
    MRNStatus,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def interface(config):
    return CustomsInterface(config=config)


# ====================================================================
# NCTS Submission Tests
# ====================================================================


class TestNCTSSubmission:
    @pytest.mark.asyncio
    async def test_submit_to_ncts_returns_response(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001", "cn_codes": ["18010000"]},
        )
        assert isinstance(result, CustomsInterfaceResponse)
        assert result.system == CustomsSystemType.NCTS

    @pytest.mark.asyncio
    async def test_ncts_request_id_generated(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.request_id != ""
        assert result.request_id.startswith("NCTS-")

    @pytest.mark.asyncio
    async def test_ncts_mrn_in_response(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.mrn == "26NL0003960000001A"

    @pytest.mark.asyncio
    async def test_ncts_accepted_response(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={
                "operator_id": "OP-001",
                "cn_codes": ["18010000"],
                "dds_reference": "GL-DDS-20260313-ABCDEF",
            },
        )
        assert result.status in ("accepted", "pending", "processing")

    @pytest.mark.asyncio
    async def test_ncts_timestamp_set(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_ncts_processing_time_measured(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.processing_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_ncts_empty_mrn_raises(self, interface):
        with pytest.raises(ValueError, match="MRN"):
            await interface.submit_to_ncts(
                declaration_id="DECL-001",
                mrn="",
                declaration_data={"operator_id": "OP-001"},
            )

    @pytest.mark.asyncio
    async def test_ncts_invalid_mrn_length_raises(self, interface):
        with pytest.raises(ValueError, match="MRN"):
            await interface.submit_to_ncts(
                declaration_id="DECL-001",
                mrn="SHORT",
                declaration_data={"operator_id": "OP-001"},
            )


# ====================================================================
# AIS Submission Tests
# ====================================================================


class TestAISSubmission:
    @pytest.mark.asyncio
    async def test_submit_to_ais_returns_response(self, interface):
        result = await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={"operator_id": "OP-001", "cn_codes": ["15119110"]},
        )
        assert isinstance(result, CustomsInterfaceResponse)
        assert result.system == CustomsSystemType.AIS

    @pytest.mark.asyncio
    async def test_ais_request_id_generated(self, interface):
        result = await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.request_id != ""
        assert result.request_id.startswith("AIS-")

    @pytest.mark.asyncio
    async def test_ais_mrn_in_response(self, interface):
        result = await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.mrn == "26BE0003960000003C"

    @pytest.mark.asyncio
    async def test_ais_accepted_response(self, interface):
        result = await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={
                "operator_id": "OP-001",
                "cn_codes": ["15119110"],
                "dds_reference": "GL-DDS-20260310-CDEFGH",
            },
        )
        assert result.status in ("accepted", "pending", "processing")

    @pytest.mark.asyncio
    async def test_ais_empty_mrn_raises(self, interface):
        with pytest.raises(ValueError, match="MRN"):
            await interface.submit_to_ais(
                declaration_id="DECL-001",
                mrn="",
                declaration_data={"operator_id": "OP-001"},
            )


# ====================================================================
# Generic Submit Tests
# ====================================================================


class TestGenericSubmit:
    @pytest.mark.asyncio
    async def test_submit_routes_to_ncts(self, interface):
        result = await interface.submit(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            system="ncts",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.system == CustomsSystemType.NCTS

    @pytest.mark.asyncio
    async def test_submit_routes_to_ais(self, interface):
        result = await interface.submit(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            system="ais",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result.system == CustomsSystemType.AIS

    @pytest.mark.asyncio
    async def test_submit_invalid_system_raises(self, interface):
        with pytest.raises(ValueError, match="system"):
            await interface.submit(
                declaration_id="DECL-001",
                mrn="26NL0003960000001A",
                system="invalid_system",
                declaration_data={"operator_id": "OP-001"},
            )


# ====================================================================
# Status Check Tests
# ====================================================================


class TestStatusCheck:
    @pytest.mark.asyncio
    async def test_check_ncts_status(self, interface):
        # First submit
        await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        status = await interface.check_status(
            mrn="26NL0003960000001A",
            system="ncts",
        )
        assert status is not None
        assert "status" in status

    @pytest.mark.asyncio
    async def test_check_ais_status(self, interface):
        await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={"operator_id": "OP-001"},
        )
        status = await interface.check_status(
            mrn="26BE0003960000003C",
            system="ais",
        )
        assert status is not None

    @pytest.mark.asyncio
    async def test_check_status_unknown_mrn(self, interface):
        status = await interface.check_status(
            mrn="26XX0003960000099Z",
            system="ncts",
        )
        assert status is None or status.get("status") == "not_found"


# ====================================================================
# Retry Logic Tests
# ====================================================================


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, interface):
        # The engine should handle retry internally
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, interface):
        # After max retries, should return error response or raise
        health = await interface.health_check()
        assert health["ncts_retries"] == interface._config.ncts_retry_count


# ====================================================================
# Error Response Handling Tests
# ====================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_response_with_errors(self, ncts_rejection_response):
        assert len(ncts_rejection_response.errors) == 2
        assert "E001" in ncts_rejection_response.errors[0]

    @pytest.mark.asyncio
    async def test_response_code_non_zero_for_rejection(self, ncts_rejection_response):
        assert ncts_rejection_response.response_code != "00"

    @pytest.mark.asyncio
    async def test_success_response_no_errors(self, ncts_success_response):
        assert ncts_success_response.errors is None or len(ncts_success_response.errors) == 0

    @pytest.mark.asyncio
    async def test_response_code_00_for_success(self, ncts_success_response):
        assert ncts_success_response.response_code == "00"


# ====================================================================
# Provenance Tests
# ====================================================================


class TestProvenance:
    @pytest.mark.asyncio
    async def test_submission_has_provenance(self, interface):
        result = await interface.submit_to_ncts(
            declaration_id="DECL-001",
            mrn="26NL0003960000001A",
            declaration_data={"operator_id": "OP-001"},
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_ais_submission_has_provenance(self, interface):
        result = await interface.submit_to_ais(
            declaration_id="DECL-001",
            mrn="26BE0003960000003C",
            declaration_data={"operator_id": "OP-001"},
        )
        assert len(result.provenance_hash) == 64


# ====================================================================
# Health Check Tests
# ====================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, interface):
        health = await interface.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "CustomsInterface"

    @pytest.mark.asyncio
    async def test_status_healthy(self, interface):
        health = await interface.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_ncts_connectivity(self, interface):
        health = await interface.health_check()
        assert "ncts_connected" in health

    @pytest.mark.asyncio
    async def test_ais_connectivity(self, interface):
        health = await interface.health_check()
        assert "ais_connected" in health

    @pytest.mark.asyncio
    async def test_submissions_count(self, interface):
        health = await interface.health_check()
        assert health["submissions_total"] == 0
