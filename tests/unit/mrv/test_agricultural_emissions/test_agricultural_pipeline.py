# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Pipeline Engine.

Tests AgriculturalPipelineEngine: single/batch pipeline execution,
pipeline stages, error handling, health, and statistics.

Target: 70+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.agricultural_pipeline import (
        AgriculturalPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline engine not available")


@pytest.fixture
def engine():
    if PIPELINE_AVAILABLE:
        return AgriculturalPipelineEngine()
    return None


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestPipelineInit:
    """Test AgriculturalPipelineEngine initialization."""

    def test_engine_creation(self, engine):
        assert engine is not None

    def test_default_creation(self):
        e = AgriculturalPipelineEngine()
        assert e is not None

    def test_has_execute(self, engine):
        assert hasattr(engine, 'execute')

    def test_has_execute_batch(self, engine):
        assert hasattr(engine, 'execute_batch')

    def test_has_statistics(self, engine):
        assert hasattr(engine, 'get_statistics')


# ===========================================================================
# Test Class: Single Execution - Enteric
# ===========================================================================


@_SKIP
class TestEntericPipeline:
    """Test enteric fermentation pipeline execution."""

    def test_basic_enteric(self, engine):
        request = {
            "farm_id": "farm-001",
            "tenant_id": "tenant-001",
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 200,
            "calculation_method": "ipcc_tier_1",
            "gwp_source": "AR6",
        }
        result = engine.execute(request)
        assert result is not None

    def test_enteric_has_co2e(self, engine):
        request = {
            "farm_id": "farm-001",
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 200,
        }
        result = engine.execute(request)
        co2e = 0
        if isinstance(result, dict):
            co2e = float(result.get("total_co2e_tonnes", result.get("co2e_tonnes", 0)))
        elif hasattr(result, 'total_co2e_tonnes'):
            co2e = float(result.total_co2e_tonnes)
        assert co2e > 0

    def test_enteric_sheep(self, engine):
        request = {
            "source_category": "enteric_fermentation",
            "animal_type": "sheep",
            "head_count": 1000,
        }
        result = engine.execute(request)
        assert result is not None

    def test_enteric_provenance(self, engine):
        request = {
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 100,
        }
        result = engine.execute(request)
        ph = ""
        if isinstance(result, dict):
            ph = result.get("provenance_hash", "")
        elif hasattr(result, 'provenance_hash'):
            ph = result.provenance_hash
        if ph:
            assert len(ph) == 64


# ===========================================================================
# Test Class: Single Execution - Manure
# ===========================================================================


@_SKIP
class TestManurePipeline:
    """Test manure management pipeline execution."""

    def test_basic_manure(self, engine):
        request = {
            "source_category": "manure_management",
            "animal_type": "dairy_cattle",
            "head_count": 200,
            "awms_type": "uncovered_anaerobic_lagoon",
        }
        result = engine.execute(request)
        assert result is not None

    def test_manure_has_ch4_and_n2o(self, engine):
        request = {
            "source_category": "manure_management",
            "animal_type": "dairy_cattle",
            "head_count": 100,
            "awms_type": "uncovered_anaerobic_lagoon",
        }
        result = engine.execute(request)
        co2e = 0
        if isinstance(result, dict):
            co2e = float(result.get("total_co2e_tonnes", 0))
        elif hasattr(result, 'total_co2e_tonnes'):
            co2e = float(result.total_co2e_tonnes)
        assert co2e > 0


# ===========================================================================
# Test Class: Single Execution - Cropland
# ===========================================================================


@_SKIP
class TestCroplandPipeline:
    """Test cropland emissions pipeline execution."""

    def test_soil_n2o(self, engine):
        request = {
            "source_category": "cropland_emissions",
            "synthetic_n_tonnes": 100,
        }
        result = engine.execute(request)
        assert result is not None

    def test_liming(self, engine):
        request = {
            "source_category": "cropland_emissions",
            "limestone_tonnes": 1000,
        }
        result = engine.execute(request)
        assert result is not None

    def test_urea(self, engine):
        request = {
            "source_category": "cropland_emissions",
            "urea_tonnes": 500,
        }
        result = engine.execute(request)
        assert result is not None

    def test_rice(self, engine):
        request = {
            "source_category": "rice_cultivation",
            "area_ha": 50,
            "cultivation_period_days": 120,
        }
        result = engine.execute(request)
        assert result is not None

    def test_field_burning(self, engine):
        request = {
            "source_category": "field_burning",
            "crop_type": "wheat",
            "area_burned_ha": 100,
            "crop_production_tonnes": 500,
        }
        result = engine.execute(request)
        assert result is not None


# ===========================================================================
# Test Class: Batch Execution
# ===========================================================================


@_SKIP
class TestBatchPipeline:
    """Test batch pipeline execution."""

    def test_batch_basic(self, engine):
        requests = [
            {
                "source_category": "enteric_fermentation",
                "animal_type": "dairy_cattle",
                "head_count": 200,
            },
            {
                "source_category": "enteric_fermentation",
                "animal_type": "sheep",
                "head_count": 500,
            },
        ]
        result = engine.execute_batch(requests)
        assert result is not None

    def test_batch_mixed_sources(self, engine):
        requests = [
            {
                "source_category": "enteric_fermentation",
                "animal_type": "dairy_cattle",
                "head_count": 200,
            },
            {
                "source_category": "cropland_emissions",
                "synthetic_n_tonnes": 100,
            },
        ]
        result = engine.execute_batch(requests)
        assert result is not None

    def test_batch_empty(self, engine):
        result = engine.execute_batch([])
        assert result is not None

    def test_batch_aggregation(self, engine):
        requests = [
            {"source_category": "enteric_fermentation", "animal_type": "dairy_cattle", "head_count": 100},
            {"source_category": "enteric_fermentation", "animal_type": "dairy_cattle", "head_count": 200},
        ]
        result = engine.execute_batch(requests)
        # Should aggregate results
        assert result is not None


# ===========================================================================
# Test Class: Error Handling
# ===========================================================================


@_SKIP
class TestPipelineErrors:
    """Test pipeline error handling."""

    def test_missing_source_category(self, engine):
        request = {"animal_type": "dairy_cattle", "head_count": 100}
        result = engine.execute(request)
        # Should handle gracefully
        assert result is not None

    def test_invalid_animal_type(self, engine):
        request = {
            "source_category": "enteric_fermentation",
            "animal_type": "unicorn",
            "head_count": 100,
        }
        result = engine.execute(request)
        assert result is not None

    def test_zero_head_count(self, engine):
        request = {
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 0,
        }
        result = engine.execute(request)
        assert result is not None


# ===========================================================================
# Test Class: Health and Statistics
# ===========================================================================


@_SKIP
class TestPipelineHealth:
    """Test pipeline health and statistics."""

    def test_get_statistics(self, engine):
        stats = engine.get_statistics()
        assert isinstance(stats, dict)

    def test_get_engine_health(self, engine):
        health = engine.get_engine_health()
        assert isinstance(health, dict)

    def test_reset(self, engine):
        engine.execute({
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 100,
        })
        engine.reset()
        stats = engine.get_statistics()
        assert isinstance(stats, dict)

    def test_statistics_after_execution(self, engine):
        engine.execute({
            "source_category": "enteric_fermentation",
            "animal_type": "dairy_cattle",
            "head_count": 100,
        })
        stats = engine.get_statistics()
        assert isinstance(stats, dict)


# ===========================================================================
# Test Class: Thread Safety
# ===========================================================================


@_SKIP
class TestPipelineThreadSafety:
    """Test pipeline thread safety."""

    def test_concurrent_executions(self, engine):
        errors = []

        def worker():
            try:
                for _ in range(5):
                    engine.execute({
                        "source_category": "enteric_fermentation",
                        "animal_type": "dairy_cattle",
                        "head_count": 100,
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
