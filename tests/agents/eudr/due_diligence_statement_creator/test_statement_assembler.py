# -*- coding: utf-8 -*-
"""
Unit tests for StatementAssembler - AGENT-EUDR-037

Tests DDS assembly, retrieval, filtering, status updates, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.statement_assembler import StatementAssembler
from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    CommodityType, DDSStatement, DDSStatus, StatementType,
)


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def assembler(config):
    return StatementAssembler(config=config)


class TestAssembleStatement:
    @pytest.mark.asyncio
    async def test_returns_dds_statement(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001", operator_name="Test Corp")
        assert isinstance(stmt, DDSStatement)

    @pytest.mark.asyncio
    async def test_statement_id_prefix(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert stmt.statement_id.startswith("DDS-")

    @pytest.mark.asyncio
    async def test_reference_number_prefix(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert stmt.reference_number.startswith("GL-DDS-")

    @pytest.mark.asyncio
    async def test_draft_status(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert stmt.status == DDSStatus.DRAFT

    @pytest.mark.asyncio
    async def test_operator_id_set(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-TEST")
        assert stmt.operator_id == "OP-TEST"

    @pytest.mark.asyncio
    async def test_operator_name_set(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001", operator_name="ACME")
        assert stmt.operator_name == "ACME"

    @pytest.mark.asyncio
    async def test_commodities_parsed(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", commodities=["cocoa", "coffee"])
        assert CommodityType.COCOA in stmt.commodities
        assert CommodityType.COFFEE in stmt.commodities

    @pytest.mark.asyncio
    async def test_unknown_commodity_skipped(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", commodities=["cocoa", "unknown_xyz"])
        assert CommodityType.COCOA in stmt.commodities
        assert len(stmt.commodities) == 1

    @pytest.mark.asyncio
    async def test_no_commodities_defaults_to_wood(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert CommodityType.WOOD in stmt.commodities

    @pytest.mark.asyncio
    async def test_statement_type_placing(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", statement_type="placing")
        assert stmt.statement_type == StatementType.PLACING

    @pytest.mark.asyncio
    async def test_statement_type_export(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", statement_type="export")
        assert stmt.statement_type == StatementType.EXPORT

    @pytest.mark.asyncio
    async def test_invalid_statement_type_defaults_placing(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", statement_type="invalid_xyz")
        assert stmt.statement_type == StatementType.PLACING

    @pytest.mark.asyncio
    async def test_language_set(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", language="fr")
        assert stmt.language == "fr"

    @pytest.mark.asyncio
    async def test_unsupported_language_defaults(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", language="xx")
        assert stmt.language == "en"

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert len(stmt.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_version_number_is_one(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert stmt.version_number == 1

    @pytest.mark.asyncio
    async def test_initial_version_record_created(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        assert len(stmt.versions) == 1
        assert stmt.versions[0].version_number == 1

    @pytest.mark.asyncio
    async def test_empty_operator_id_raises(self, assembler):
        with pytest.raises(ValueError, match="operator_id"):
            await assembler.assemble_statement(operator_id="")

    @pytest.mark.asyncio
    async def test_assembly_count_increments(self, assembler):
        await assembler.assemble_statement(operator_id="OP-001")
        await assembler.assemble_statement(operator_id="OP-002")
        health = await assembler.health_check()
        assert health["statements_assembled"] == 2


class TestGetStatement:
    @pytest.mark.asyncio
    async def test_get_existing_statement(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        result = await assembler.get_statement(stmt.statement_id)
        assert result is not None
        assert result.statement_id == stmt.statement_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, assembler):
        result = await assembler.get_statement("DDS-NONEXISTENT")
        assert result is None


class TestListStatements:
    @pytest.mark.asyncio
    async def test_list_all(self, assembler):
        await assembler.assemble_statement(operator_id="OP-001")
        await assembler.assemble_statement(operator_id="OP-002")
        results = await assembler.list_statements()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_by_operator(self, assembler):
        await assembler.assemble_statement(operator_id="OP-001")
        await assembler.assemble_statement(operator_id="OP-002")
        results = await assembler.list_statements(operator_id="OP-001")
        assert len(results) == 1
        assert results[0].operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        results = await assembler.list_statements(status="draft")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_filter_empty_result(self, assembler):
        await assembler.assemble_statement(operator_id="OP-001")
        results = await assembler.list_statements(operator_id="OP-NOBODY")
        assert len(results) == 0


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_to_validated(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        updated = await assembler.update_status(stmt.statement_id, "validated")
        assert updated is not None
        assert updated.status == DDSStatus.VALIDATED

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, assembler):
        result = await assembler.update_status("DDS-NONEXISTENT", "draft")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_status_raises(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        with pytest.raises(ValueError):
            await assembler.update_status(stmt.statement_id, "invalid_xyz")

    @pytest.mark.asyncio
    async def test_provenance_hash_updated(self, assembler):
        stmt = await assembler.assemble_statement(operator_id="OP-001")
        old_hash = stmt.provenance_hash
        await assembler.update_status(stmt.statement_id, "validated")
        assert stmt.provenance_hash != old_hash


class TestGetSummary:
    @pytest.mark.asyncio
    async def test_get_summary_returns_summary(self, assembler):
        stmt = await assembler.assemble_statement(
            operator_id="OP-001", operator_name="Test Corp",
            commodities=["cocoa"])
        summary = await assembler.get_summary(stmt.statement_id)
        assert summary is not None
        assert summary.operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_get_summary_nonexistent_returns_none(self, assembler):
        result = await assembler.get_summary("DDS-NONEXISTENT")
        assert result is None


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, assembler):
        health = await assembler.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "StatementAssembler"

    @pytest.mark.asyncio
    async def test_status_healthy(self, assembler):
        health = await assembler.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_initial_count_zero(self, assembler):
        health = await assembler.health_check()
        assert health["statements_assembled"] == 0
        assert health["active_statements"] == 0
