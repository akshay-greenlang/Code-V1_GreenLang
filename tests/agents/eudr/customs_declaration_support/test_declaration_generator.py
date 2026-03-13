# -*- coding: utf-8 -*-
"""
Unit tests for DeclarationGenerator engine - AGENT-EUDR-039

Tests customs declaration creation, SAD form generation, MRN assignment,
status transitions, batch operations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.declaration_generator import DeclarationGenerator
from greenlang.agents.eudr.customs_declaration_support.models import (
    CommodityType, CustomsDeclaration, DeclarationStatus, DeclarationType,
    IncotermsType, SADForm, MRN_FORMAT_REGEX,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def generator(config):
    return DeclarationGenerator(config=config)


# ====================================================================
# Declaration Creation Tests
# ====================================================================


class TestCreateDeclaration:
    @pytest.mark.asyncio
    async def test_returns_customs_declaration(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001",
            operator_name="Acme Trading Ltd",
            commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert isinstance(decl, CustomsDeclaration)

    @pytest.mark.asyncio
    async def test_declaration_id_prefix(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert decl.declaration_id.startswith("DECL-")

    @pytest.mark.asyncio
    async def test_draft_status(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert decl.status == DeclarationStatus.DRAFT

    @pytest.mark.asyncio
    async def test_operator_id_set(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-TEST", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert decl.operator_id == "OP-TEST"

    @pytest.mark.asyncio
    async def test_operator_name_set(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", operator_name="ACME",
            commodities=["cocoa"], country_of_origin="CI",
        )
        assert decl.operator_name == "ACME"

    @pytest.mark.asyncio
    async def test_commodities_parsed(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001",
            commodities=["cocoa", "coffee"],
            country_of_origin="CI",
        )
        assert CommodityType.COCOA in decl.commodities
        assert CommodityType.COFFEE in decl.commodities

    @pytest.mark.asyncio
    async def test_unknown_commodity_skipped(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001",
            commodities=["cocoa", "unknown_xyz"],
            country_of_origin="CI",
        )
        assert CommodityType.COCOA in decl.commodities
        assert len(decl.commodities) == 1

    @pytest.mark.asyncio
    async def test_country_of_origin_set(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="GH",
        )
        assert decl.country_of_origin == "GH"

    @pytest.mark.asyncio
    async def test_declaration_type_import(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI", declaration_type="import",
        )
        assert decl.declaration_type == DeclarationType.IMPORT

    @pytest.mark.asyncio
    async def test_declaration_type_export(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI", declaration_type="export",
        )
        assert decl.declaration_type == DeclarationType.EXPORT

    @pytest.mark.asyncio
    async def test_incoterms_cif(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI", incoterms="CIF",
        )
        assert decl.incoterms == IncotermsType.CIF

    @pytest.mark.asyncio
    async def test_incoterms_fob(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI", incoterms="FOB",
        )
        assert decl.incoterms == IncotermsType.FOB

    @pytest.mark.asyncio
    async def test_dds_reference_set(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert decl.dds_reference == "GL-DDS-20260313-ABCDEF"

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert len(decl.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_empty_operator_id_raises(self, generator):
        with pytest.raises(ValueError, match="operator_id"):
            await generator.create_declaration(
                operator_id="", commodities=["cocoa"],
                country_of_origin="CI",
            )

    @pytest.mark.asyncio
    async def test_no_commodities_raises(self, generator):
        with pytest.raises(ValueError, match="commodit"):
            await generator.create_declaration(
                operator_id="OP-001", commodities=[],
                country_of_origin="CI",
            )


# ====================================================================
# MRN Generation Tests
# ====================================================================


class TestMRNGeneration:
    @pytest.mark.asyncio
    async def test_mrn_generated_on_creation(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert decl.mrn is not None
        assert len(decl.mrn) == 18

    @pytest.mark.asyncio
    async def test_mrn_format_compliance(self, generator):
        import re
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert re.match(MRN_FORMAT_REGEX, decl.mrn) is not None

    @pytest.mark.asyncio
    async def test_mrn_uniqueness(self, generator):
        d1 = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        d2 = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert d1.mrn != d2.mrn

    @pytest.mark.asyncio
    async def test_mrn_starts_with_year(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        year_prefix = decl.mrn[:2]
        assert year_prefix.isdigit()

    @pytest.mark.asyncio
    async def test_mrn_contains_country_code(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        # Country code is at positions 2-3
        country = decl.mrn[2:4]
        assert country.isalpha()

    @pytest.mark.asyncio
    async def test_mrn_alphanumeric(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert decl.mrn.isalnum()


# ====================================================================
# SAD Form Generation Tests
# ====================================================================


class TestSADFormGeneration:
    @pytest.mark.asyncio
    async def test_generate_sad_form(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", operator_name="Acme Trading",
            commodities=["cocoa"], country_of_origin="CI",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert isinstance(sad, SADForm)

    @pytest.mark.asyncio
    async def test_sad_form_has_declaration_type(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert sad.box1_declaration_type in ("IM", "EX", "TR")

    @pytest.mark.asyncio
    async def test_sad_form_has_commodity_code(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI", cn_codes=["18010000"],
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert len(sad.box33_commodity_code) == 8

    @pytest.mark.asyncio
    async def test_sad_form_has_origin_country(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert sad.box34_country_of_origin == "CI"

    @pytest.mark.asyncio
    async def test_sad_form_has_eudr_dds_reference(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert sad.eudr_dds_reference == "GL-DDS-20260313-ABCDEF"

    @pytest.mark.asyncio
    async def test_sad_form_has_eori(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", operator_eori="BE1234567890",
            commodities=["cocoa"], country_of_origin="CI",
        )
        sad = await generator.generate_sad_form(decl.declaration_id)
        assert sad.box8_eori == "BE1234567890" or sad.box14_eori == "BE1234567890"

    @pytest.mark.asyncio
    async def test_sad_form_nonexistent_declaration_raises(self, generator):
        with pytest.raises(ValueError, match="not found"):
            await generator.generate_sad_form("DECL-NONEXISTENT")


# ====================================================================
# Status Update Tests
# ====================================================================


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_to_submitted(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        updated = await generator.update_status(decl.declaration_id, "submitted")
        assert updated is not None
        assert updated.status == DeclarationStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_update_to_cleared(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        await generator.update_status(decl.declaration_id, "submitted")
        updated = await generator.update_status(decl.declaration_id, "cleared")
        assert updated.status == DeclarationStatus.CLEARED

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, generator):
        result = await generator.update_status("DECL-NONEXISTENT", "submitted")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_status_raises(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        with pytest.raises(ValueError):
            await generator.update_status(decl.declaration_id, "invalid_xyz")


# ====================================================================
# Retrieval Tests
# ====================================================================


class TestGetDeclaration:
    @pytest.mark.asyncio
    async def test_get_existing(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        result = await generator.get_declaration(decl.declaration_id)
        assert result is not None
        assert result.declaration_id == decl.declaration_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, generator):
        result = await generator.get_declaration("DECL-NONEXISTENT")
        assert result is None


class TestListDeclarations:
    @pytest.mark.asyncio
    async def test_list_all(self, generator):
        await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        await generator.create_declaration(
            operator_id="OP-002", commodities=["coffee"],
            country_of_origin="BR",
        )
        results = await generator.list_declarations()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_by_operator(self, generator):
        await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        await generator.create_declaration(
            operator_id="OP-002", commodities=["coffee"],
            country_of_origin="BR",
        )
        results = await generator.list_declarations(operator_id="OP-001")
        assert len(results) == 1
        assert results[0].operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, generator):
        decl = await generator.create_declaration(
            operator_id="OP-001", commodities=["cocoa"],
            country_of_origin="CI",
        )
        results = await generator.list_declarations(status="draft")
        assert len(results) == 1


# ====================================================================
# Health Check Tests
# ====================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, generator):
        health = await generator.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "DeclarationGenerator"

    @pytest.mark.asyncio
    async def test_status_healthy(self, generator):
        health = await generator.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_initial_count_zero(self, generator):
        health = await generator.health_check()
        assert health["declarations_created"] == 0
        assert health["active_declarations"] == 0
