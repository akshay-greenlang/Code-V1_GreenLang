# -*- coding: utf-8 -*-
"""
Unit tests for SupplyChainCompiler - AGENT-EUDR-037

Tests supply chain compilation, completeness validation, country summaries,
and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.supply_chain_compiler import SupplyChainCompiler
from greenlang.agents.eudr.due_diligence_statement_creator.models import CommodityType, SupplyChainData


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def compiler(config):
    return SupplyChainCompiler(config=config)


@pytest.fixture
def sample_suppliers():
    return [
        {"name": "Farm A", "tier": 1, "country_code": "CI", "plot_count": 5},
        {"name": "Farm B", "tier": 1, "country_code": "GH", "plot_count": 3},
        {"name": "Cooperative C", "tier": 2, "country_code": "CI", "plot_count": 0},
    ]


class TestCompileSupplyChain:
    @pytest.mark.asyncio
    async def test_returns_supply_chain_data(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        assert isinstance(sc, SupplyChainData)

    @pytest.mark.asyncio
    async def test_supply_chain_id_set(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-XYZ", operator_id="OP-001", commodity="cocoa")
        assert sc.supply_chain_id == "SC-XYZ"

    @pytest.mark.asyncio
    async def test_operator_id_set(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-TEST", commodity="cocoa")
        assert sc.operator_id == "OP-TEST"

    @pytest.mark.asyncio
    async def test_commodity_parsed(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        assert sc.commodity == CommodityType.COCOA

    @pytest.mark.asyncio
    async def test_invalid_commodity_defaults_wood(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="invalid_xyz")
        assert sc.commodity == CommodityType.WOOD

    @pytest.mark.asyncio
    async def test_supplier_count(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        assert sc.supplier_count == 3

    @pytest.mark.asyncio
    async def test_tier_count(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        assert sc.tier_count == 2

    @pytest.mark.asyncio
    async def test_plot_count_summed(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        assert sc.plot_count == 8

    @pytest.mark.asyncio
    async def test_countries_from_param(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers,
            countries_of_production=["CI", "GH"])
        assert "CI" in sc.countries_of_production
        assert "GH" in sc.countries_of_production

    @pytest.mark.asyncio
    async def test_countries_extracted_from_suppliers(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        assert len(sc.countries_of_production) >= 1

    @pytest.mark.asyncio
    async def test_chain_of_custody_model(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", chain_of_custody_model="mass_balance")
        assert sc.chain_of_custody_model == "mass_balance"

    @pytest.mark.asyncio
    async def test_default_chain_of_custody(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        assert sc.chain_of_custody_model == "segregation"

    @pytest.mark.asyncio
    async def test_traceability_score_from_param(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", traceability_score=85.0)
        assert sc.traceability_score == Decimal("85.00")

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        assert len(sc.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_no_suppliers(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        assert sc.supplier_count == 0
        assert sc.tier_count == 0

    @pytest.mark.asyncio
    async def test_compilation_count_increments(self, compiler):
        await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        await compiler.compile_supply_chain(
            supply_chain_id="SC-002", operator_id="OP-001", commodity="coffee")
        health = await compiler.health_check()
        assert health["compilations_completed"] == 2


class TestValidateCompleteness:
    @pytest.mark.asyncio
    async def test_complete_supply_chain(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers,
            countries_of_production=["CI", "GH"],
            traceability_score=80.0)
        result = await compiler.validate_completeness(sc)
        assert result["complete"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_no_suppliers_fails(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        result = await compiler.validate_completeness(sc)
        assert result["complete"] is False
        assert any("supplier" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_no_countries_fails(self, compiler):
        sc = SupplyChainData(
            supply_chain_id="SC-X", operator_id="OP-X",
            commodity=CommodityType.COCOA,
            supplier_count=1, suppliers=[{"name": "A"}],
            countries_of_production=[],
            traceability_score=Decimal("80"))
        result = await compiler.validate_completeness(sc)
        assert any("countries" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_low_traceability_score_fails(self, compiler):
        sc = SupplyChainData(
            supply_chain_id="SC-X", operator_id="OP-X",
            commodity=CommodityType.COCOA,
            supplier_count=1, suppliers=[{"name": "A"}],
            countries_of_production=["CI"],
            traceability_score=Decimal("30"))
        result = await compiler.validate_completeness(sc)
        assert any("traceability" in i.lower() for i in result["issues"])


class TestGetCountriesSummary:
    @pytest.mark.asyncio
    async def test_country_counts(self, compiler, sample_suppliers):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001",
            commodity="cocoa", suppliers=sample_suppliers)
        counts = await compiler.get_countries_summary(sc)
        assert isinstance(counts, dict)
        assert counts.get("CI", 0) >= 1

    @pytest.mark.asyncio
    async def test_empty_suppliers(self, compiler):
        sc = await compiler.compile_supply_chain(
            supply_chain_id="SC-001", operator_id="OP-001", commodity="cocoa")
        counts = await compiler.get_countries_summary(sc)
        assert counts == {}


class TestSupplyChainHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, compiler):
        health = await compiler.health_check()
        assert health["engine"] == "SupplyChainCompiler"
        assert health["status"] == "healthy"
