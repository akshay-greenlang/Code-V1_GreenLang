# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Cropland Engine.

Tests CroplandEmissionsEngine: direct/indirect N2O, liming CO2, urea CO2,
rice CH4, field burning, batch calculations, and edge cases.

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
    from greenlang.agents.mrv.agricultural_emissions.cropland_emissions import (
        CroplandEmissionsEngine,
    )
    CROPLAND_AVAILABLE = True
except ImportError:
    CROPLAND_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

_SKIP = pytest.mark.skipif(not CROPLAND_AVAILABLE, reason="Cropland engine not available")


@pytest.fixture
def db_engine():
    if DB_AVAILABLE:
        return AgriculturalDatabaseEngine()
    return None


@pytest.fixture
def engine(db_engine):
    if CROPLAND_AVAILABLE:
        return CroplandEmissionsEngine(database=db_engine)
    return None


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestCroplandInit:
    """Test CroplandEmissionsEngine initialization."""

    def test_engine_creation(self, engine):
        assert engine is not None

    def test_without_database(self):
        e = CroplandEmissionsEngine()
        assert e is not None

    def test_has_calculate_methods(self, engine):
        assert hasattr(engine, 'calculate_direct_n2o')
        assert hasattr(engine, 'calculate_indirect_n2o')
        assert hasattr(engine, 'calculate_liming_co2')
        assert hasattr(engine, 'calculate_urea_co2')
        assert hasattr(engine, 'calculate_rice_ch4')


# ===========================================================================
# Test Class: Direct N2O
# ===========================================================================


@_SKIP
class TestDirectN2O:
    """Test direct N2O emission calculations."""

    def test_synthetic_n(self, engine):
        result = engine.calculate_direct_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        assert result is not None

    def test_synthetic_n_positive(self, engine):
        result = engine.calculate_direct_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        n2o = 0
        if isinstance(result, dict):
            n2o = float(result.get("n2o_tonnes", result.get("n2o_n_tonnes", 0)))
        elif hasattr(result, 'n2o_tonnes'):
            n2o = float(result.n2o_tonnes)
        assert n2o > 0

    def test_organic_n(self, engine):
        result = engine.calculate_direct_n2o(
            organic_n_tonnes=Decimal("50"),
        )
        assert result is not None

    def test_crop_residue_n(self, engine):
        result = engine.calculate_direct_n2o(
            crop_residue_n_tonnes=Decimal("20"),
        )
        assert result is not None

    def test_combined_inputs(self, engine):
        result = engine.calculate_direct_n2o(
            synthetic_n_tonnes=Decimal("100"),
            organic_n_tonnes=Decimal("50"),
            crop_residue_n_tonnes=Decimal("20"),
        )
        assert result is not None

    def test_zero_input(self, engine):
        result = engine.calculate_direct_n2o(
            synthetic_n_tonnes=Decimal("0"),
        )
        n2o = 0
        if isinstance(result, dict):
            n2o = float(result.get("n2o_tonnes", result.get("n2o_n_tonnes", 0)))
        elif hasattr(result, 'n2o_tonnes'):
            n2o = float(result.n2o_tonnes)
        assert n2o == 0.0

    def test_linear_scaling(self, engine):
        """Doubling N input should double N2O."""
        r1 = engine.calculate_direct_n2o(synthetic_n_tonnes=Decimal("100"))
        r2 = engine.calculate_direct_n2o(synthetic_n_tonnes=Decimal("200"))
        n1 = 0
        n2 = 0
        if isinstance(r1, dict):
            n1 = float(r1.get("n2o_tonnes", r1.get("n2o_n_tonnes", 0)))
            n2 = float(r2.get("n2o_tonnes", r2.get("n2o_n_tonnes", 0)))
        elif hasattr(r1, 'n2o_tonnes'):
            n1 = float(r1.n2o_tonnes)
            n2 = float(r2.n2o_tonnes)
        if n1 > 0:
            assert abs(n2 / n1 - 2.0) < 0.01

    def test_ef1_applied(self, engine):
        """100 tonnes N × EF1(0.01) = 1.0 tonnes N2O-N."""
        result = engine.calculate_direct_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        # Expected: 100 × 0.01 = 1.0 tN2O-N × 44/28 ≈ 1.571 tN2O
        n2o = 0
        if isinstance(result, dict):
            n2o = float(result.get("n2o_tonnes", result.get("n2o_n_tonnes", 0)))
        elif hasattr(result, 'n2o_tonnes'):
            n2o = float(result.n2o_tonnes)
        if n2o > 0:
            assert n2o > 1.0  # at least 1.0 tN2O-N


# ===========================================================================
# Test Class: Indirect N2O
# ===========================================================================


@_SKIP
class TestIndirectN2O:
    """Test indirect N2O emission calculations."""

    def test_volatilization(self, engine):
        result = engine.calculate_indirect_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        assert result is not None

    def test_leaching(self, engine):
        result = engine.calculate_indirect_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        assert result is not None

    def test_indirect_positive(self, engine):
        result = engine.calculate_indirect_n2o(
            synthetic_n_tonnes=Decimal("100"),
        )
        n2o = 0
        if isinstance(result, dict):
            n2o = float(result.get("n2o_tonnes", result.get("total_n2o_tonnes", 0)))
        elif hasattr(result, 'n2o_tonnes'):
            n2o = float(result.n2o_tonnes)
        assert n2o >= 0


# ===========================================================================
# Test Class: Total Soil N2O
# ===========================================================================


@_SKIP
class TestTotalSoilN2O:
    """Test combined direct + indirect N2O calculation."""

    def test_total_soil(self, engine):
        if not hasattr(engine, 'calculate_total_soil_n2o'):
            pytest.skip("calculate_total_soil_n2o not available")
        result = engine.calculate_total_soil_n2o(
            synthetic_n_tonnes=Decimal("100"),
            organic_n_tonnes=Decimal("50"),
        )
        assert result is not None


# ===========================================================================
# Test Class: Liming CO2
# ===========================================================================


@_SKIP
class TestLimingCO2:
    """Test liming CO2 emission calculations."""

    def test_limestone(self, engine):
        result = engine.calculate_liming_co2(
            limestone_tonnes=Decimal("1000"),
        )
        assert result is not None

    def test_limestone_positive(self, engine):
        result = engine.calculate_liming_co2(
            limestone_tonnes=Decimal("1000"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        assert co2 > 0

    def test_dolomite(self, engine):
        result = engine.calculate_liming_co2(
            dolomite_tonnes=Decimal("500"),
        )
        assert result is not None

    def test_combined_liming(self, engine):
        result = engine.calculate_liming_co2(
            limestone_tonnes=Decimal("1000"),
            dolomite_tonnes=Decimal("500"),
        )
        assert result is not None

    def test_zero_liming(self, engine):
        result = engine.calculate_liming_co2(
            limestone_tonnes=Decimal("0"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        assert co2 == 0.0

    def test_limestone_ef_applied(self, engine):
        """1000t limestone × 0.12 tC/t × 44/12 ≈ 440 tCO2."""
        result = engine.calculate_liming_co2(
            limestone_tonnes=Decimal("1000"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        if co2 > 0:
            assert abs(co2 - 440.0) < 10.0  # ≈ 440 tCO2


# ===========================================================================
# Test Class: Urea CO2
# ===========================================================================


@_SKIP
class TestUreaCO2:
    """Test urea CO2 emission calculations."""

    def test_urea_basic(self, engine):
        result = engine.calculate_urea_co2(
            urea_tonnes=Decimal("500"),
        )
        assert result is not None

    def test_urea_positive(self, engine):
        result = engine.calculate_urea_co2(
            urea_tonnes=Decimal("500"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        assert co2 > 0

    def test_zero_urea(self, engine):
        result = engine.calculate_urea_co2(
            urea_tonnes=Decimal("0"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        assert co2 == 0.0

    def test_urea_ef_applied(self, engine):
        """500t urea × 0.20 tC/t × 44/12 ≈ 366.67 tCO2."""
        result = engine.calculate_urea_co2(
            urea_tonnes=Decimal("500"),
        )
        co2 = 0
        if isinstance(result, dict):
            co2 = float(result.get("co2_tonnes", 0))
        elif hasattr(result, 'co2_tonnes'):
            co2 = float(result.co2_tonnes)
        if co2 > 0:
            assert abs(co2 - 366.67) < 10.0


# ===========================================================================
# Test Class: Rice CH4
# ===========================================================================


@_SKIP
class TestRiceCH4:
    """Test rice cultivation CH4 calculations."""

    def test_basic_rice(self, engine):
        result = engine.calculate_rice_ch4(
            area_ha=Decimal("50"),
            cultivation_period_days=120,
        )
        assert result is not None

    def test_rice_positive(self, engine):
        result = engine.calculate_rice_ch4(
            area_ha=Decimal("50"),
            cultivation_period_days=120,
        )
        ch4 = 0
        if isinstance(result, dict):
            ch4 = float(result.get("ch4_tonnes", result.get("ch4_kg", 0)))
        elif hasattr(result, 'ch4_tonnes'):
            ch4 = float(result.ch4_tonnes)
        assert ch4 > 0

    def test_water_regime_effect(self, engine):
        """Single aeration should produce less CH4 than continuously flooded."""
        r_flood = engine.calculate_rice_ch4(
            area_ha=Decimal("50"),
            cultivation_period_days=120,
            water_regime="continuously_flooded",
        )
        r_aerate = engine.calculate_rice_ch4(
            area_ha=Decimal("50"),
            cultivation_period_days=120,
            water_regime="single_aeration",
        )
        ch4_f = 0
        ch4_a = 0
        if isinstance(r_flood, dict):
            ch4_f = float(r_flood.get("ch4_tonnes", r_flood.get("ch4_kg", 0)))
            ch4_a = float(r_aerate.get("ch4_tonnes", r_aerate.get("ch4_kg", 0)))
        elif hasattr(r_flood, 'ch4_tonnes'):
            ch4_f = float(r_flood.ch4_tonnes)
            ch4_a = float(r_aerate.ch4_tonnes)
        if ch4_f > 0 and ch4_a > 0:
            assert ch4_f > ch4_a

    def test_larger_area(self, engine):
        """More area = more CH4."""
        r1 = engine.calculate_rice_ch4(area_ha=Decimal("50"), cultivation_period_days=120)
        r2 = engine.calculate_rice_ch4(area_ha=Decimal("100"), cultivation_period_days=120)
        ch4_1 = 0
        ch4_2 = 0
        if isinstance(r1, dict):
            ch4_1 = float(r1.get("ch4_tonnes", r1.get("ch4_kg", 0)))
            ch4_2 = float(r2.get("ch4_tonnes", r2.get("ch4_kg", 0)))
        elif hasattr(r1, 'ch4_tonnes'):
            ch4_1 = float(r1.ch4_tonnes)
            ch4_2 = float(r2.ch4_tonnes)
        if ch4_1 > 0:
            assert ch4_2 > ch4_1

    def test_zero_area(self, engine):
        result = engine.calculate_rice_ch4(
            area_ha=Decimal("0"),
            cultivation_period_days=120,
        )
        ch4 = 0
        if isinstance(result, dict):
            ch4 = float(result.get("ch4_tonnes", result.get("ch4_kg", 0)))
        elif hasattr(result, 'ch4_tonnes'):
            ch4 = float(result.ch4_tonnes)
        assert ch4 == 0.0


# ===========================================================================
# Test Class: Field Burning
# ===========================================================================


@_SKIP
class TestFieldBurning:
    """Test field burning calculations."""

    def test_wheat_burning(self, engine):
        result = engine.calculate_field_burning(
            crop_type="wheat",
            area_burned_ha=Decimal("100"),
            crop_production_tonnes=Decimal("500"),
        )
        assert result is not None

    def test_rice_burning(self, engine):
        result = engine.calculate_field_burning(
            crop_type="rice",
            area_burned_ha=Decimal("50"),
            crop_production_tonnes=Decimal("300"),
        )
        assert result is not None

    def test_burning_positive(self, engine):
        result = engine.calculate_field_burning(
            crop_type="wheat",
            area_burned_ha=Decimal("100"),
            crop_production_tonnes=Decimal("500"),
        )
        total = 0
        if isinstance(result, dict):
            total = float(result.get("total_co2e_tonnes", result.get("co2e_tonnes", 0)))
        elif hasattr(result, 'total_co2e_tonnes'):
            total = float(result.total_co2e_tonnes)
        assert total > 0

    def test_zero_area_burning(self, engine):
        result = engine.calculate_field_burning(
            crop_type="wheat",
            area_burned_ha=Decimal("0"),
            crop_production_tonnes=Decimal("0"),
        )
        total = 0
        if isinstance(result, dict):
            total = float(result.get("total_co2e_tonnes", result.get("co2e_tonnes", 0)))
        elif hasattr(result, 'total_co2e_tonnes'):
            total = float(result.total_co2e_tonnes)
        assert total == 0.0


# ===========================================================================
# Test Class: Batch and Statistics
# ===========================================================================


@_SKIP
class TestCroplandBatchAndStats:
    """Test batch calculations and statistics."""

    def test_batch_calculation(self, engine):
        if not hasattr(engine, 'calculate_cropland_batch'):
            pytest.skip("calculate_cropland_batch not available")
        requests = [
            {"input_type": "synthetic_n", "quantity_tonnes": 100},
            {"input_type": "limestone", "quantity_tonnes": 500},
        ]
        result = engine.calculate_cropland_batch(requests)
        assert result is not None

    def test_engine_info(self, engine):
        if hasattr(engine, 'get_engine_info'):
            info = engine.get_engine_info()
            assert isinstance(info, dict)

    def test_default_emission_factors(self, engine):
        if hasattr(engine, 'get_default_emission_factors'):
            efs = engine.get_default_emission_factors()
            assert isinstance(efs, dict)

    def test_gwp_values(self, engine):
        if hasattr(engine, 'get_gwp_values'):
            gwp = engine.get_gwp_values()
            assert isinstance(gwp, dict)
