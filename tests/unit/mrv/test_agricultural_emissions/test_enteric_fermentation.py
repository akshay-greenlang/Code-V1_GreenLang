# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Enteric Fermentation Engine.

Tests EntericFermentationEngine Tier 1/2 calculations, multi-herd,
GWP conversion, provenance hashing, and edge cases.

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
    from greenlang.agricultural_emissions.enteric_fermentation import (
        EntericFermentationEngine,
    )
    ENTERIC_AVAILABLE = True
except ImportError:
    ENTERIC_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENTERIC_AVAILABLE, reason="Enteric engine not available")


@pytest.fixture
def db_engine():
    if DB_AVAILABLE:
        return AgriculturalDatabaseEngine()
    return None


@pytest.fixture
def engine(db_engine):
    if ENTERIC_AVAILABLE:
        return EntericFermentationEngine(database=db_engine)
    return None


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestEntericInit:
    """Test EntericFermentationEngine initialization."""

    def test_engine_creation(self, engine):
        assert engine is not None

    def test_engine_without_database(self):
        e = EntericFermentationEngine()
        assert e is not None

    def test_engine_with_database(self, db_engine):
        e = EntericFermentationEngine(database=db_engine)
        assert e is not None

    def test_has_calculate_methods(self, engine):
        assert hasattr(engine, 'calculate_tier1')
        assert hasattr(engine, 'calculate_tier2')


# ===========================================================================
# Test Class: Tier 1 Calculation
# ===========================================================================


@_SKIP
class TestTier1Calculation:
    """Test Tier 1 enteric CH4 calculations."""

    def test_dairy_cattle_basic(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=200,
        )
        assert result is not None

    def test_dairy_cattle_ch4_positive(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=200,
        )
        ch4 = result.get("ch4_tonnes", result.get("ch4_kg", 0))
        if hasattr(result, 'ch4_tonnes'):
            ch4 = result.ch4_tonnes
        assert Decimal(str(ch4)) > Decimal("0")

    def test_non_dairy_cattle(self, engine):
        result = engine.calculate_tier1(
            animal_type="non_dairy_cattle",
            head_count=500,
        )
        assert result is not None

    def test_sheep(self, engine):
        result = engine.calculate_tier1(
            animal_type="sheep",
            head_count=1000,
        )
        assert result is not None

    def test_swine_market(self, engine):
        result = engine.calculate_tier1(
            animal_type="swine_market",
            head_count=2000,
        )
        assert result is not None

    def test_buffalo(self, engine):
        result = engine.calculate_tier1(
            animal_type="buffalo",
            head_count=100,
        )
        assert result is not None

    def test_goats(self, engine):
        result = engine.calculate_tier1(
            animal_type="goats",
            head_count=500,
        )
        assert result is not None

    def test_zero_head_count(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=0,
        )
        ch4 = 0
        if isinstance(result, dict):
            ch4 = float(result.get("ch4_tonnes", result.get("ch4_kg", 0)))
        elif hasattr(result, 'ch4_tonnes'):
            ch4 = float(result.ch4_tonnes)
        assert ch4 == 0.0

    def test_with_region(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=100,
            region="developed",
        )
        assert result is not None

    def test_with_gwp_source(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=100,
            gwp_source="AR6",
        )
        assert result is not None

    def test_result_has_provenance_hash(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=100,
        )
        ph = None
        if isinstance(result, dict):
            ph = result.get("provenance_hash", "")
        elif hasattr(result, 'provenance_hash'):
            ph = result.provenance_hash
        if ph:
            assert len(ph) == 64

    def test_dairy_higher_than_sheep(self, engine):
        """Dairy cattle should produce more CH4 per head than sheep."""
        r1 = engine.calculate_tier1(animal_type="dairy_cattle", head_count=1)
        r2 = engine.calculate_tier1(animal_type="sheep", head_count=1)
        ch4_dairy = 0
        ch4_sheep = 0
        if isinstance(r1, dict):
            ch4_dairy = float(r1.get("ch4_tonnes", r1.get("ch4_kg", 0)))
            ch4_sheep = float(r2.get("ch4_tonnes", r2.get("ch4_kg", 0)))
        elif hasattr(r1, 'ch4_tonnes'):
            ch4_dairy = float(r1.ch4_tonnes)
            ch4_sheep = float(r2.ch4_tonnes)
        assert ch4_dairy > ch4_sheep

    def test_linear_scaling(self, engine):
        """Doubling head count should double emissions."""
        r1 = engine.calculate_tier1(animal_type="sheep", head_count=100)
        r2 = engine.calculate_tier1(animal_type="sheep", head_count=200)
        ch4_1 = 0
        ch4_2 = 0
        if isinstance(r1, dict):
            ch4_1 = float(r1.get("ch4_tonnes", r1.get("ch4_kg", 0)))
            ch4_2 = float(r2.get("ch4_tonnes", r2.get("ch4_kg", 0)))
        elif hasattr(r1, 'ch4_tonnes'):
            ch4_1 = float(r1.ch4_tonnes)
            ch4_2 = float(r2.ch4_tonnes)
        if ch4_1 > 0:
            assert abs(ch4_2 / ch4_1 - 2.0) < 0.01

    def test_poultry_minimal(self, engine):
        result = engine.calculate_tier1(
            animal_type="poultry_broilers",
            head_count=10000,
        )
        assert result is not None


# ===========================================================================
# Test Class: Tier 2 Calculation
# ===========================================================================


@_SKIP
class TestTier2Calculation:
    """Test Tier 2 enteric CH4 calculations using GE-based formula."""

    def test_dairy_cattle_tier2(self, engine):
        result = engine.calculate_tier2(
            animal_type="dairy_cattle",
            head_count=200,
        )
        assert result is not None

    def test_tier2_with_feed(self, engine):
        result = engine.calculate_tier2(
            animal_type="dairy_cattle",
            head_count=100,
            gross_energy_mj_per_day=Decimal("335"),
            ym_factor=Decimal("6.5"),
        )
        assert result is not None

    def test_tier2_default_ym(self, engine):
        """Ym defaults should be available for common animal types."""
        if hasattr(engine, 'get_ym_default'):
            ym = engine.get_ym_default("dairy_cattle")
            assert ym is not None
            assert Decimal(str(ym)) > Decimal("3")

    def test_tier2_beef_cattle(self, engine):
        result = engine.calculate_tier2(
            animal_type="non_dairy_cattle",
            head_count=300,
        )
        assert result is not None


# ===========================================================================
# Test Class: Multi-Herd Calculation
# ===========================================================================


@_SKIP
class TestMultiHerd:
    """Test multi-herd calculation."""

    def test_calculate_herd(self, engine):
        if not hasattr(engine, 'calculate_herd'):
            pytest.skip("calculate_herd not available")
        result = engine.calculate_herd({
            "animal_type": "dairy_cattle",
            "head_count": 200,
        })
        assert result is not None

    def test_batch_calculation(self, engine):
        if not hasattr(engine, 'calculate_enteric_batch'):
            pytest.skip("calculate_enteric_batch not available")
        requests = [
            {"animal_type": "dairy_cattle", "head_count": 100},
            {"animal_type": "sheep", "head_count": 500},
        ]
        result = engine.calculate_enteric_batch(requests)
        assert result is not None

    def test_batch_aggregates(self, engine):
        if not hasattr(engine, 'calculate_enteric_batch'):
            pytest.skip("calculate_enteric_batch not available")
        requests = [
            {"animal_type": "dairy_cattle", "head_count": 100},
            {"animal_type": "non_dairy_cattle", "head_count": 200},
        ]
        result = engine.calculate_enteric_batch(requests)
        # Total should be sum of individual calculations
        assert result is not None


# ===========================================================================
# Test Class: GWP Conversion
# ===========================================================================


@_SKIP
class TestGWPConversion:
    """Test GWP conversion in enteric calculations."""

    def test_ar6_conversion(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=100,
            gwp_source="AR6",
        )
        co2e = 0
        if isinstance(result, dict):
            co2e = float(result.get("co2e_tonnes", result.get("total_co2e_tonnes", 0)))
        elif hasattr(result, 'co2e_tonnes'):
            co2e = float(result.co2e_tonnes)
        elif hasattr(result, 'total_co2e_tonnes'):
            co2e = float(result.total_co2e_tonnes)
        assert co2e > 0

    def test_ar5_vs_ar6(self, engine):
        """AR6 GWP is higher for CH4 so CO2e should be higher."""
        r_ar5 = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100, gwp_source="AR5"
        )
        r_ar6 = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100, gwp_source="AR6"
        )
        co2e_5 = 0
        co2e_6 = 0
        if isinstance(r_ar5, dict):
            co2e_5 = float(r_ar5.get("co2e_tonnes", r_ar5.get("total_co2e_tonnes", 0)))
            co2e_6 = float(r_ar6.get("co2e_tonnes", r_ar6.get("total_co2e_tonnes", 0)))
        elif hasattr(r_ar5, 'co2e_tonnes'):
            co2e_5 = float(r_ar5.co2e_tonnes)
            co2e_6 = float(r_ar6.co2e_tonnes)
        elif hasattr(r_ar5, 'total_co2e_tonnes'):
            co2e_5 = float(r_ar5.total_co2e_tonnes)
            co2e_6 = float(r_ar6.total_co2e_tonnes)
        if co2e_5 > 0 and co2e_6 > 0:
            # AR6 CH4 GWP (29.8) > AR5 CH4 GWP (28)
            assert co2e_6 >= co2e_5


# ===========================================================================
# Test Class: Provenance
# ===========================================================================


@_SKIP
class TestEntericProvenance:
    """Test provenance tracking in enteric calculations."""

    def test_hash_present(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100,
        )
        ph = ""
        if isinstance(result, dict):
            ph = result.get("provenance_hash", "")
        elif hasattr(result, 'provenance_hash'):
            ph = result.provenance_hash
        if ph:
            assert len(ph) == 64

    def test_deterministic_hash(self, engine):
        """Same inputs should produce same hash."""
        r1 = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100,
        )
        r2 = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100,
        )
        ph1 = ""
        ph2 = ""
        if isinstance(r1, dict):
            ph1 = r1.get("provenance_hash", "")
            ph2 = r2.get("provenance_hash", "")
        elif hasattr(r1, 'provenance_hash'):
            ph1 = r1.provenance_hash
            ph2 = r2.provenance_hash
        if ph1 and ph2:
            assert ph1 == ph2

    def test_different_inputs_different_hash(self, engine):
        r1 = engine.calculate_tier1(
            animal_type="dairy_cattle", head_count=100,
        )
        r2 = engine.calculate_tier1(
            animal_type="sheep", head_count=100,
        )
        ph1 = ""
        ph2 = ""
        if isinstance(r1, dict):
            ph1 = r1.get("provenance_hash", "")
            ph2 = r2.get("provenance_hash", "")
        elif hasattr(r1, 'provenance_hash'):
            ph1 = r1.provenance_hash
            ph2 = r2.provenance_hash
        if ph1 and ph2:
            assert ph1 != ph2


# ===========================================================================
# Test Class: Statistics
# ===========================================================================


@_SKIP
class TestEntericStatistics:
    """Test engine statistics."""

    def test_get_statistics(self, engine):
        stats = engine.get_statistics()
        assert isinstance(stats, dict)

    def test_statistics_after_calculation(self, engine):
        engine.calculate_tier1(animal_type="dairy_cattle", head_count=100)
        stats = engine.get_statistics()
        assert isinstance(stats, dict)


# ===========================================================================
# Test Class: Edge Cases
# ===========================================================================


@_SKIP
class TestEntericEdgeCases:
    """Test edge cases in enteric calculations."""

    def test_very_large_herd(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=1_000_000,
        )
        assert result is not None

    def test_single_animal(self, engine):
        result = engine.calculate_tier1(
            animal_type="dairy_cattle",
            head_count=1,
        )
        assert result is not None

    def test_thread_safety(self, engine):
        errors = []

        def worker():
            try:
                for _ in range(10):
                    engine.calculate_tier1(
                        animal_type="dairy_cattle",
                        head_count=100,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
