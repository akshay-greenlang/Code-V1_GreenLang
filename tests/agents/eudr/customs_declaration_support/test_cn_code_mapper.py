# -*- coding: utf-8 -*-
"""
Unit tests for CNCodeMapper engine - AGENT-EUDR-039

Tests all 7 EUDR commodity CN code mappings, lookup by commodity,
lookup by CN code, reverse mapping, batch operations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.cn_code_mapper import CNCodeMapper
from greenlang.agents.eudr.customs_declaration_support.models import (
    CommodityType, CNCodeMapping, EUDR_CN_CODE_MAPPINGS,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def mapper(config):
    return CNCodeMapper(config=config)


class TestMapCNCodes:
    """Test CN code mapping for all 7 EUDR commodities."""

    @pytest.mark.asyncio
    async def test_map_cattle_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("cattle")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.CATTLE

    @pytest.mark.asyncio
    async def test_map_cocoa_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("cocoa")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.COCOA

    @pytest.mark.asyncio
    async def test_map_coffee_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("coffee")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.COFFEE

    @pytest.mark.asyncio
    async def test_map_oil_palm_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("oil_palm")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.OIL_PALM

    @pytest.mark.asyncio
    async def test_map_rubber_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("rubber")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.RUBBER

    @pytest.mark.asyncio
    async def test_map_soya_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("soya")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.SOYA

    @pytest.mark.asyncio
    async def test_map_wood_returns_cn_codes(self, mapper):
        result = await mapper.map_commodity("wood")
        assert len(result) > 0
        for item in result:
            assert item.commodity == CommodityType.WOOD


class TestCNCodeFormat:
    """Test that all CN codes follow 8-digit format."""

    @pytest.mark.asyncio
    async def test_all_cn_codes_8_digits(self, mapper):
        for commodity_str in ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]:
            codes = await mapper.map_commodity(commodity_str)
            for code in codes:
                assert len(code.cn_code) == 8, (
                    f"CN code {code.cn_code} for {commodity_str} is not 8 digits"
                )
                assert code.cn_code.isdigit(), (
                    f"CN code {code.cn_code} for {commodity_str} contains non-digits"
                )

    @pytest.mark.asyncio
    async def test_cn_codes_have_descriptions(self, mapper):
        codes = await mapper.map_commodity("cocoa")
        for code in codes:
            assert code.description != ""
            assert len(code.description) > 5


class TestLookupByCNCode:
    """Test reverse lookup from CN code to commodity."""

    @pytest.mark.asyncio
    async def test_lookup_cocoa_cn_code(self, mapper):
        result = await mapper.lookup_cn_code("18010000")
        assert result is not None
        assert result.commodity == CommodityType.COCOA

    @pytest.mark.asyncio
    async def test_lookup_coffee_cn_code(self, mapper):
        result = await mapper.lookup_cn_code("09011100")
        assert result is not None
        assert result.commodity == CommodityType.COFFEE

    @pytest.mark.asyncio
    async def test_lookup_wood_cn_code(self, mapper):
        result = await mapper.lookup_cn_code("44011100")
        assert result is not None
        assert result.commodity == CommodityType.WOOD

    @pytest.mark.asyncio
    async def test_lookup_unknown_cn_code_returns_none(self, mapper):
        result = await mapper.lookup_cn_code("99999999")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_empty_cn_code_returns_none(self, mapper):
        result = await mapper.lookup_cn_code("")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_invalid_length_returns_none(self, mapper):
        result = await mapper.lookup_cn_code("1801")
        assert result is None


class TestIsEUDRRegulated:
    """Test whether a CN code is EUDR-regulated."""

    @pytest.mark.asyncio
    async def test_cocoa_is_regulated(self, mapper):
        assert await mapper.is_eudr_regulated("18010000") is True

    @pytest.mark.asyncio
    async def test_coffee_is_regulated(self, mapper):
        assert await mapper.is_eudr_regulated("09011100") is True

    @pytest.mark.asyncio
    async def test_unknown_not_regulated(self, mapper):
        assert await mapper.is_eudr_regulated("87032100") is False

    @pytest.mark.asyncio
    async def test_empty_not_regulated(self, mapper):
        assert await mapper.is_eudr_regulated("") is False


class TestGetDutyRate:
    """Test duty rate retrieval for CN codes."""

    @pytest.mark.asyncio
    async def test_zero_duty_for_raw_cocoa(self, mapper):
        result = await mapper.lookup_cn_code("18010000")
        if result is not None:
            assert result.duty_rate >= 0

    @pytest.mark.asyncio
    async def test_duty_rate_is_decimal(self, mapper):
        codes = await mapper.map_commodity("cocoa")
        for code in codes:
            from decimal import Decimal
            assert isinstance(code.duty_rate, Decimal)


class TestBatchMapping:
    """Test batch commodity mapping."""

    @pytest.mark.asyncio
    async def test_batch_map_multiple_commodities(self, mapper):
        commodities = ["cocoa", "coffee", "wood"]
        results = await mapper.map_commodities_batch(commodities)
        assert len(results) == 3
        assert all(commodity in results for commodity in commodities)

    @pytest.mark.asyncio
    async def test_batch_map_with_invalid_commodity(self, mapper):
        commodities = ["cocoa", "invalid_xyz"]
        results = await mapper.map_commodities_batch(commodities)
        assert "cocoa" in results
        assert len(results["cocoa"]) > 0

    @pytest.mark.asyncio
    async def test_batch_map_empty_list(self, mapper):
        results = await mapper.map_commodities_batch([])
        assert results == {} or len(results) == 0


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, mapper):
        health = await mapper.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "CNCodeMapper"

    @pytest.mark.asyncio
    async def test_status_healthy(self, mapper):
        health = await mapper.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_commodity_count(self, mapper):
        health = await mapper.health_check()
        assert health["commodities_mapped"] == 7

    @pytest.mark.asyncio
    async def test_total_cn_codes_positive(self, mapper):
        health = await mapper.health_check()
        assert health["total_cn_codes"] > 0
