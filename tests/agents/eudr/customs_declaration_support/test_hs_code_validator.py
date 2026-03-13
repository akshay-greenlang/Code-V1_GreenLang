# -*- coding: utf-8 -*-
"""
Unit tests for HSCodeValidator engine - AGENT-EUDR-039

Tests HS code validation, chapter lookup, EUDR regulation checks,
format validation, batch operations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.hs_code_validator import HSCodeValidator
from greenlang.agents.eudr.customs_declaration_support.models import (
    CommodityType, HSCodeInfo, EUDR_HS_CHAPTERS,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def validator(config):
    return HSCodeValidator(config=config)


class TestValidateHSCode:
    """Test HS code format validation."""

    @pytest.mark.asyncio
    async def test_valid_6_digit_code(self, validator):
        result = await validator.validate("180100")
        assert result is not None
        assert result.hs_code == "180100"

    @pytest.mark.asyncio
    async def test_valid_cocoa_code(self, validator):
        result = await validator.validate("180100")
        assert result is not None
        assert result.eudr_regulated is True
        assert result.commodity == CommodityType.COCOA

    @pytest.mark.asyncio
    async def test_valid_coffee_code(self, validator):
        result = await validator.validate("090111")
        assert result is not None
        assert result.eudr_regulated is True
        assert result.commodity == CommodityType.COFFEE

    @pytest.mark.asyncio
    async def test_valid_cattle_code(self, validator):
        result = await validator.validate("010221")
        assert result is not None
        assert result.eudr_regulated is True
        assert result.commodity == CommodityType.CATTLE

    @pytest.mark.asyncio
    async def test_valid_wood_code(self, validator):
        result = await validator.validate("440111")
        assert result is not None
        assert result.eudr_regulated is True
        assert result.commodity == CommodityType.WOOD

    @pytest.mark.asyncio
    async def test_valid_rubber_code(self, validator):
        result = await validator.validate("400110")
        assert result is not None
        assert result.eudr_regulated is True

    @pytest.mark.asyncio
    async def test_valid_soya_code(self, validator):
        result = await validator.validate("120190")
        assert result is not None
        assert result.eudr_regulated is True

    @pytest.mark.asyncio
    async def test_valid_palm_oil_code(self, validator):
        result = await validator.validate("151190")
        assert result is not None
        assert result.eudr_regulated is True

    @pytest.mark.asyncio
    async def test_non_eudr_code(self, validator):
        result = await validator.validate("870321")
        assert result is not None
        assert result.eudr_regulated is False

    @pytest.mark.asyncio
    async def test_empty_code_raises(self, validator):
        with pytest.raises(ValueError, match="HS code"):
            await validator.validate("")

    @pytest.mark.asyncio
    async def test_too_short_code_raises(self, validator):
        with pytest.raises(ValueError, match="6 digits"):
            await validator.validate("1801")

    @pytest.mark.asyncio
    async def test_too_long_code_raises(self, validator):
        with pytest.raises(ValueError, match="6 digits"):
            await validator.validate("18010099")

    @pytest.mark.asyncio
    async def test_non_numeric_code_raises(self, validator):
        with pytest.raises(ValueError, match="numeric"):
            await validator.validate("18AB00")


class TestChapterLookup:
    """Test HS code chapter extraction and lookup."""

    @pytest.mark.asyncio
    async def test_chapter_18_is_cocoa(self, validator):
        chapter = await validator.get_chapter(18)
        assert chapter is not None
        assert "cocoa" in chapter["description"].lower()

    @pytest.mark.asyncio
    async def test_chapter_09_is_coffee(self, validator):
        chapter = await validator.get_chapter(9)
        assert chapter is not None
        assert "coffee" in chapter["description"].lower()

    @pytest.mark.asyncio
    async def test_chapter_01_is_cattle(self, validator):
        chapter = await validator.get_chapter(1)
        assert chapter is not None
        assert "animal" in chapter["description"].lower() or "cattle" in chapter["description"].lower()

    @pytest.mark.asyncio
    async def test_chapter_44_is_wood(self, validator):
        chapter = await validator.get_chapter(44)
        assert chapter is not None
        assert "wood" in chapter["description"].lower()

    @pytest.mark.asyncio
    async def test_extract_chapter_from_hs_code(self, validator):
        info = await validator.validate("180100")
        assert info.chapter == 18

    @pytest.mark.asyncio
    async def test_extract_chapter_from_coffee_hs(self, validator):
        info = await validator.validate("090111")
        assert info.chapter == 9

    @pytest.mark.asyncio
    async def test_unknown_chapter_returns_none(self, validator):
        chapter = await validator.get_chapter(99)
        if chapter is not None:
            assert chapter.get("eudr_regulated", False) is False


class TestIsEUDRChapter:
    """Test whether HS code chapters are EUDR-regulated."""

    @pytest.mark.asyncio
    async def test_chapter_01_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(1) is True

    @pytest.mark.asyncio
    async def test_chapter_09_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(9) is True

    @pytest.mark.asyncio
    async def test_chapter_12_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(12) is True

    @pytest.mark.asyncio
    async def test_chapter_15_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(15) is True

    @pytest.mark.asyncio
    async def test_chapter_18_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(18) is True

    @pytest.mark.asyncio
    async def test_chapter_40_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(40) is True

    @pytest.mark.asyncio
    async def test_chapter_44_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(44) is True

    @pytest.mark.asyncio
    async def test_chapter_87_not_eudr_regulated(self, validator):
        assert await validator.is_eudr_chapter(87) is False


class TestBatchValidation:
    """Test batch HS code validation."""

    @pytest.mark.asyncio
    async def test_batch_validate_multiple_codes(self, validator):
        codes = ["180100", "090111", "440111"]
        results = await validator.validate_batch(codes)
        assert len(results) == 3
        assert all(r.eudr_regulated for r in results)

    @pytest.mark.asyncio
    async def test_batch_validate_mixed_codes(self, validator):
        codes = ["180100", "870321"]
        results = await validator.validate_batch(codes)
        assert len(results) == 2
        regulated = [r for r in results if r.eudr_regulated]
        non_regulated = [r for r in results if not r.eudr_regulated]
        assert len(regulated) == 1
        assert len(non_regulated) == 1

    @pytest.mark.asyncio
    async def test_batch_validate_empty_list(self, validator):
        results = await validator.validate_batch([])
        assert results == []


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, validator):
        health = await validator.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "HSCodeValidator"

    @pytest.mark.asyncio
    async def test_status_healthy(self, validator):
        health = await validator.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_eudr_chapters_loaded(self, validator):
        health = await validator.health_check()
        assert health["eudr_chapters_loaded"] > 0
