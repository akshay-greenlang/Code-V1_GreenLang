# -*- coding: utf-8 -*-
"""
Unit tests for PublicDataMiningEngine - AGENT-EUDR-027

Tests engine initialization, single-source harvest, harvest-all concurrent
operations, latest data retrieval, freshness status monitoring, stale
source detection, incremental harvest mode, and harvest statistics.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 3: Public Data Mining)
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.information_gathering.config import InformationGatheringConfig
from greenlang.agents.eudr.information_gathering.models import (
    ExternalDatabaseSource,
    FreshnessStatus,
)
from greenlang.agents.eudr.information_gathering.public_data_mining_engine import (
    PublicDataMiningEngine,
)


class TestPublicDataMiningEngineInit:
    """Test engine initialization."""

    def test_engine_initialization(self, config):
        engine = PublicDataMiningEngine(config)
        stats = engine.get_harvest_stats()
        assert stats["harvesters_registered"] == 8
        assert stats["total_harvests"] == 0

    def test_engine_with_default_config(self):
        engine = PublicDataMiningEngine()
        stats = engine.get_harvest_stats()
        assert stats["harvesters_registered"] == 8


class TestHarvestSource:
    """Test single-source harvest operations."""

    @pytest.mark.asyncio
    async def test_harvest_source_fao(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
            commodity="coffee",
        )
        assert result.source == ExternalDatabaseSource.FAO_STAT
        assert result.records_harvested > 0
        assert result.freshness_status == FreshnessStatus.FRESH
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_harvest_source_gfw(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.GLOBAL_FOREST_WATCH,
            country_code="IDN",
        )
        assert result.source == ExternalDatabaseSource.GLOBAL_FOREST_WATCH
        assert result.records_harvested > 0

    @pytest.mark.asyncio
    async def test_harvest_source_comtrade(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.UN_COMTRADE,
            country_code="DE",
            commodity="cocoa",
        )
        assert result.records_harvested > 0

    @pytest.mark.asyncio
    async def test_harvest_source_wgi(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.WORLD_BANK_WGI,
            country_code="BRA",
        )
        assert result.records_harvested == 6  # 6 indicators

    @pytest.mark.asyncio
    async def test_harvest_source_cpi(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.TRANSPARENCY_CPI,
            country_code="GH",
        )
        assert result.records_harvested >= 1

    @pytest.mark.asyncio
    async def test_harvest_source_sanctions(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.EU_SANCTIONS,
            country_code="RU",
        )
        assert result.records_harvested > 0

    @pytest.mark.asyncio
    async def test_harvest_source_land_registry_no_country(self, config):
        engine = PublicDataMiningEngine(config)
        result = await engine.harvest_source(
            ExternalDatabaseSource.NATIONAL_LAND_REGISTRY,
        )
        # Without country_code, land registry returns 0 records
        assert result.records_harvested == 0


class TestHarvestAll:
    """Test harvest-all concurrent operations."""

    @pytest.mark.asyncio
    async def test_harvest_all(self, config):
        engine = PublicDataMiningEngine(config)
        results = await engine.harvest_all(country_code="BRA")
        # Should harvest from multiple unique sources
        assert len(results) >= 5
        for result in results:
            assert result.freshness_status in (FreshnessStatus.FRESH, FreshnessStatus.UNKNOWN)

    @pytest.mark.asyncio
    async def test_harvest_all_with_commodity(self, config):
        engine = PublicDataMiningEngine(config)
        results = await engine.harvest_all(country_code="GH", commodity="cocoa")
        assert len(results) >= 5


class TestGetLatestData:
    """Test latest data retrieval."""

    @pytest.mark.asyncio
    async def test_get_latest_data_after_harvest(self, config):
        engine = PublicDataMiningEngine(config)
        await engine.harvest_source(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
            commodity="coffee",
        )
        data = engine.get_latest_data(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
            commodity="coffee",
        )
        assert data != {}
        assert data["source"] == "fao_stat"
        assert data["country_code"] == "BRA"
        assert "provenance_hash" in data

    def test_get_latest_data_no_harvest(self, config):
        engine = PublicDataMiningEngine(config)
        data = engine.get_latest_data(ExternalDatabaseSource.FAO_STAT)
        assert data == {}


class TestFreshnessStatus:
    """Test freshness status monitoring."""

    def test_get_freshness_status(self, config):
        engine = PublicDataMiningEngine(config)
        records = engine.get_freshness_status()
        assert len(records) == 8
        for record in records:
            assert record.freshness_status in (
                FreshnessStatus.FRESH,
                FreshnessStatus.STALE,
                FreshnessStatus.EXPIRED,
            )

    def test_check_stale_sources_initial(self, config):
        engine = PublicDataMiningEngine(config)
        # Initially all sources are stale/expired (never harvested)
        stale = engine.check_stale_sources()
        assert len(stale) == 8


class TestIncrementalHarvest:
    """Test incremental harvest mode."""

    @pytest.mark.asyncio
    async def test_incremental_harvest(self, config):
        engine = PublicDataMiningEngine(config)
        # First harvest (full)
        result1 = await engine.harvest_source(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
        )
        # Second harvest (incremental)
        result2 = await engine.harvest_source(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
        )
        assert result2.is_incremental is True
        assert result2.records_harvested <= result1.records_harvested


class TestHarvestStats:
    """Test harvest statistics."""

    @pytest.mark.asyncio
    async def test_harvest_stats(self, config):
        engine = PublicDataMiningEngine(config)
        await engine.harvest_source(
            ExternalDatabaseSource.FAO_STAT,
            country_code="BRA",
        )
        await engine.harvest_source(
            ExternalDatabaseSource.UN_COMTRADE,
            country_code="DE",
        )
        stats = engine.get_harvest_stats()
        assert stats["total_harvests"] == 2
        assert stats["latest_data_entries"] == 2
        assert stats["harvesters_registered"] == 8

    @pytest.mark.asyncio
    async def test_stale_count_decreases_after_harvest(self, config):
        engine = PublicDataMiningEngine(config)
        stale_before = len(engine.check_stale_sources())
        await engine.harvest_source(
            ExternalDatabaseSource.EU_SANCTIONS,
        )
        stale_after = len(engine.check_stale_sources())
        assert stale_after < stale_before
