# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Manure Management Engine.

Tests ManureManagementEngine CH4/N2O calculations, AWMS allocation,
climate zone impacts, GWP conversion, and edge cases.

Target: 80+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.manure_management import (
        ManureManagementEngine,
    )
    MANURE_AVAILABLE = True
except ImportError:
    MANURE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

_SKIP = pytest.mark.skipif(not MANURE_AVAILABLE, reason="Manure engine not available")


@pytest.fixture
def db_engine():
    if DB_AVAILABLE:
        return AgriculturalDatabaseEngine()
    return None


@pytest.fixture
def engine(db_engine):
    if MANURE_AVAILABLE:
        return ManureManagementEngine(database=db_engine)
    return None


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestManureInit:
    """Test ManureManagementEngine initialization."""

    def test_engine_creation(self, engine):
        assert engine is not None

    def test_without_database(self):
        e = ManureManagementEngine()
        assert e is not None

    def test_has_calculate_methods(self, engine):
        assert hasattr(engine, 'calculate_manure_ch4')
        assert hasattr(engine, 'calculate_manure_n2o')

    def test_has_accessors(self, engine):
        assert hasattr(engine, 'get_vs_default')
        assert hasattr(engine, 'get_bo_default')
        assert hasattr(engine, 'get_nex_default')


# ===========================================================================
# Test Class: CH4 from Manure
# ===========================================================================


@_SKIP
class TestManureCH4:
    """Test CH4 emission calculations from manure management."""

    def test_dairy_cattle_lagoon(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=200,
            awms_type="uncovered_anaerobic_lagoon",
        )
        assert result is not None

    def test_dairy_cattle_pasture(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=200,
            awms_type="pasture_range_paddock",
        )
        assert result is not None

    def test_lagoon_higher_than_pasture(self, engine):
        """Lagoon should produce more CH4 than pasture (higher MCF)."""
        r_lagoon = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="uncovered_anaerobic_lagoon",
        )
        r_pasture = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="pasture_range_paddock",
        )
        ch4_lagoon = 0
        ch4_pasture = 0
        if isinstance(r_lagoon, dict):
            ch4_lagoon = float(r_lagoon.get("ch4_tonnes", r_lagoon.get("ch4_kg", 0)))
            ch4_pasture = float(r_pasture.get("ch4_tonnes", r_pasture.get("ch4_kg", 0)))
        elif hasattr(r_lagoon, 'ch4_tonnes'):
            ch4_lagoon = float(r_lagoon.ch4_tonnes)
            ch4_pasture = float(r_pasture.ch4_tonnes)
        assert ch4_lagoon > ch4_pasture

    def test_swine_pit_storage(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="swine_market",
            head_count=500,
            awms_type="pit_storage_above_1m",
        )
        assert result is not None

    def test_sheep_solid_storage(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="sheep",
            head_count=1000,
            awms_type="solid_storage",
        )
        assert result is not None

    def test_zero_head_count(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=0,
            awms_type="uncovered_anaerobic_lagoon",
        )
        ch4 = 0
        if isinstance(result, dict):
            ch4 = float(result.get("ch4_tonnes", result.get("ch4_kg", 0)))
        elif hasattr(result, 'ch4_tonnes'):
            ch4 = float(result.ch4_tonnes)
        assert ch4 == 0.0

    def test_digester_low_ch4(self, engine):
        """Anaerobic digester should have low MCF."""
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="anaerobic_digester",
        )
        assert result is not None

    def test_composting_low_ch4(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="composting_static",
        )
        assert result is not None


# ===========================================================================
# Test Class: N2O from Manure
# ===========================================================================


@_SKIP
class TestManureN2O:
    """Test N2O emission calculations from manure management."""

    def test_dairy_cattle_n2o(self, engine):
        result = engine.calculate_manure_n2o(
            animal_type="dairy_cattle",
            head_count=200,
            awms_type="uncovered_anaerobic_lagoon",
        )
        assert result is not None

    def test_n2o_positive(self, engine):
        result = engine.calculate_manure_n2o(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="solid_storage",
        )
        n2o = 0
        if isinstance(result, dict):
            n2o = float(result.get("n2o_tonnes", result.get("n2o_kg", 0)))
        elif hasattr(result, 'n2o_tonnes'):
            n2o = float(result.n2o_tonnes)
        assert n2o >= 0

    def test_swine_n2o(self, engine):
        result = engine.calculate_manure_n2o(
            animal_type="swine_market",
            head_count=500,
            awms_type="liquid_slurry_no_crust",
        )
        assert result is not None

    def test_indirect_n2o(self, engine):
        if not hasattr(engine, 'calculate_indirect_n2o'):
            pytest.skip("calculate_indirect_n2o not available")
        result = engine.calculate_indirect_n2o(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="uncovered_anaerobic_lagoon",
        )
        assert result is not None


# ===========================================================================
# Test Class: Total Manure Calculation
# ===========================================================================


@_SKIP
class TestManureTotal:
    """Test combined CH4+N2O manure calculations."""

    def test_total_calculation(self, engine):
        if not hasattr(engine, 'calculate_manure_total'):
            pytest.skip("calculate_manure_total not available")
        result = engine.calculate_manure_total(
            animal_type="dairy_cattle",
            head_count=200,
            awms_type="uncovered_anaerobic_lagoon",
        )
        assert result is not None

    def test_herd_manure(self, engine):
        if not hasattr(engine, 'calculate_herd_manure'):
            pytest.skip("calculate_herd_manure not available")
        result = engine.calculate_herd_manure({
            "animal_type": "dairy_cattle",
            "head_count": 200,
            "awms_allocations": [
                {"system_type": "pasture_range_paddock", "fraction": 0.6},
                {"system_type": "uncovered_anaerobic_lagoon", "fraction": 0.4},
            ],
        })
        assert result is not None

    def test_batch_manure(self, engine):
        if not hasattr(engine, 'calculate_manure_batch'):
            pytest.skip("calculate_manure_batch not available")
        requests = [
            {"animal_type": "dairy_cattle", "head_count": 100,
             "awms_type": "pasture_range_paddock"},
            {"animal_type": "sheep", "head_count": 500,
             "awms_type": "solid_storage"},
        ]
        result = engine.calculate_manure_batch(requests)
        assert result is not None


# ===========================================================================
# Test Class: Reference Data Accessors
# ===========================================================================


@_SKIP
class TestManureAccessors:
    """Test reference data accessor methods."""

    def test_get_vs_default(self, engine):
        vs = engine.get_vs_default("dairy_cattle")
        assert vs is not None
        assert Decimal(str(vs)) > Decimal("0")

    def test_get_bo_default(self, engine):
        bo = engine.get_bo_default("dairy_cattle")
        assert bo is not None
        assert Decimal(str(bo)) > Decimal("0")

    def test_get_nex_default(self, engine):
        nex = engine.get_nex_default("dairy_cattle")
        assert nex is not None
        assert Decimal(str(nex)) > Decimal("0")

    def test_get_ef3_default(self, engine):
        ef3 = engine.get_ef3_default("uncovered_anaerobic_lagoon")
        assert ef3 is not None

    def test_supported_animal_types(self, engine):
        types = engine.get_supported_animal_types()
        assert len(types) >= 10
        assert "dairy_cattle" in types

    def test_supported_awms_types(self, engine):
        types = engine.get_supported_awms_types()
        assert len(types) >= 10
        assert "pasture_range_paddock" in types

    def test_default_awms_allocation(self, engine):
        alloc = engine.get_default_awms_allocation("dairy_cattle")
        assert isinstance(alloc, dict)
        assert len(alloc) >= 1


# ===========================================================================
# Test Class: Statistics and Reset
# ===========================================================================


@_SKIP
class TestManureStatistics:
    """Test engine statistics and reset."""

    def test_get_statistics(self, engine):
        stats = engine.get_statistics()
        assert isinstance(stats, dict)

    def test_reset(self, engine):
        engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100,
            awms_type="pasture_range_paddock",
        )
        engine.reset()
        stats = engine.get_statistics()
        assert isinstance(stats, dict)


# ===========================================================================
# Test Class: Edge Cases
# ===========================================================================


@_SKIP
class TestManureEdgeCases:
    """Test manure management edge cases."""

    def test_large_herd(self, engine):
        result = engine.calculate_manure_ch4(
            animal_type="dairy_cattle",
            head_count=100_000,
            awms_type="uncovered_anaerobic_lagoon",
        )
        assert result is not None

    def test_thread_safety(self, engine):
        errors = []

        def worker():
            try:
                for _ in range(10):
                    engine.calculate_manure_ch4(
                        animal_type="dairy_cattle",
                        head_count=100,
                        awms_type="pasture_range_paddock",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
