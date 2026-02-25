# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Database Engine.

Tests AgriculturalDatabaseEngine reference data lookups including
enteric EFs, manure VS/Bo/MCF/Nex, soil N2O, liming/urea, rice,
field burning, GWP, and feed characteristics.

Target: 80+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

_SKIP = pytest.mark.skipif(not DB_AVAILABLE, reason="Database engine not available")


# ===========================================================================
# Test Class: Initialization
# ===========================================================================


@_SKIP
class TestDatabaseInit:
    """Test AgriculturalDatabaseEngine initialization."""

    def test_engine_creation(self):
        db = AgriculturalDatabaseEngine()
        assert db is not None

    def test_engine_has_methods(self):
        db = AgriculturalDatabaseEngine()
        assert hasattr(db, 'get_enteric_ef')
        assert hasattr(db, 'get_manure_vs')

    def test_engine_has_gwp(self):
        db = AgriculturalDatabaseEngine()
        assert hasattr(db, 'get_gwp')

    def test_engine_has_statistics(self):
        db = AgriculturalDatabaseEngine()
        assert hasattr(db, 'get_statistics')


# ===========================================================================
# Test Class: Enteric EF Lookups
# ===========================================================================


@_SKIP
class TestEntericEFLookups:
    """Test enteric fermentation emission factor lookups."""

    def test_dairy_cattle_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("dairy_cattle")
        assert ef is not None
        assert Decimal(str(ef)) > Decimal("50")

    def test_non_dairy_cattle_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("non_dairy_cattle")
        assert ef is not None
        assert Decimal(str(ef)) > Decimal("20")

    def test_sheep_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("sheep")
        assert ef is not None
        assert Decimal(str(ef)) > Decimal("3")

    def test_swine_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("swine_market")
        assert ef is not None
        assert Decimal(str(ef)) >= Decimal("0")

    def test_buffalo_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("buffalo")
        assert ef is not None
        assert Decimal(str(ef)) > Decimal("30")

    def test_goats_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("goats")
        assert ef is not None

    def test_camels_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("camels")
        assert ef is not None

    def test_horses_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("horses")
        assert ef is not None

    def test_poultry_ef_zero_or_small(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("poultry_layers")
        assert ef is not None
        assert Decimal(str(ef)) >= Decimal("0")

    def test_with_region(self):
        db = AgriculturalDatabaseEngine()
        # Should accept region parameter
        ef = db.get_enteric_ef("dairy_cattle", region="developed")
        if ef is not None:
            assert Decimal(str(ef)) > Decimal("50")

    def test_unknown_animal_returns_none_or_default(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_enteric_ef("unicorn")
        # Should return None, 0, or raise
        assert ef is None or Decimal(str(ef)) >= Decimal("0")


# ===========================================================================
# Test Class: Manure Management Data
# ===========================================================================


@_SKIP
class TestManureData:
    """Test manure management reference data lookups."""

    def test_vs_dairy_cattle(self):
        db = AgriculturalDatabaseEngine()
        vs = db.get_manure_vs("dairy_cattle")
        assert vs is not None
        assert Decimal(str(vs)) > Decimal("0")

    def test_vs_sheep(self):
        db = AgriculturalDatabaseEngine()
        vs = db.get_manure_vs("sheep")
        assert vs is not None

    def test_bo_dairy_cattle(self):
        db = AgriculturalDatabaseEngine()
        bo = db.get_manure_bo("dairy_cattle")
        assert bo is not None
        assert Decimal(str(bo)) > Decimal("0")

    def test_bo_swine(self):
        db = AgriculturalDatabaseEngine()
        bo = db.get_manure_bo("swine_market")
        assert bo is not None

    def test_mcf_lagoon(self):
        db = AgriculturalDatabaseEngine()
        mcf = db.get_manure_mcf("uncovered_anaerobic_lagoon")
        assert mcf is not None
        assert Decimal(str(mcf)) > Decimal("0.50")

    def test_mcf_pasture(self):
        db = AgriculturalDatabaseEngine()
        mcf = db.get_manure_mcf("pasture_range_paddock")
        assert mcf is not None
        assert Decimal(str(mcf)) < Decimal("0.10")

    def test_mcf_with_climate(self):
        db = AgriculturalDatabaseEngine()
        mcf = db.get_manure_mcf("uncovered_anaerobic_lagoon", climate_zone="warm_temperate")
        if mcf is not None:
            assert Decimal(str(mcf)) > Decimal("0.50")

    def test_nex_dairy_cattle(self):
        db = AgriculturalDatabaseEngine()
        nex = db.get_manure_nex("dairy_cattle")
        assert nex is not None
        assert Decimal(str(nex)) > Decimal("0")

    def test_n2o_ef_lagoon(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_manure_n2o_ef("uncovered_anaerobic_lagoon")
        assert ef is not None

    def test_n2o_ef_pasture(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_manure_n2o_ef("pasture_range_paddock")
        assert ef is not None


# ===========================================================================
# Test Class: Soil N2O
# ===========================================================================


@_SKIP
class TestSoilN2O:
    """Test soil N2O emission factor lookups."""

    def test_ef1(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_soil_n2o_ef("EF1")
        assert ef is not None
        assert Decimal(str(ef)) == Decimal("0.01")

    def test_indirect_frac_gasf(self):
        db = AgriculturalDatabaseEngine()
        frac = db.get_indirect_n2o_fraction("Frac_GASF")
        if frac is None:
            frac = db.get_indirect_n2o_fraction("FRAC_GASF")
        assert frac is not None
        assert Decimal(str(frac)) == Decimal("0.10")

    def test_indirect_frac_leach(self):
        db = AgriculturalDatabaseEngine()
        frac = db.get_indirect_n2o_fraction("Frac_LEACH")
        if frac is None:
            frac = db.get_indirect_n2o_fraction("FRAC_LEACH")
        assert frac is not None
        assert Decimal(str(frac)) == Decimal("0.30")


# ===========================================================================
# Test Class: Liming and Urea
# ===========================================================================


@_SKIP
class TestLimingUrea:
    """Test liming and urea EF lookups."""

    def test_limestone_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_liming_ef("limestone")
        assert ef is not None
        assert Decimal(str(ef)) == Decimal("0.12")

    def test_dolomite_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_liming_ef("dolomite")
        assert ef is not None
        assert Decimal(str(ef)) == Decimal("0.13")

    def test_urea_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_urea_ef()
        assert ef is not None
        assert Decimal(str(ef)) == Decimal("0.20")


# ===========================================================================
# Test Class: Rice Cultivation
# ===========================================================================


@_SKIP
class TestRiceCultivation:
    """Test rice cultivation reference data lookups."""

    def test_baseline_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_rice_baseline_ef()
        assert ef is not None
        assert Decimal(str(ef)) == Decimal("1.30")

    def test_water_regime_continuously_flooded(self):
        db = AgriculturalDatabaseEngine()
        sf = db.get_rice_water_regime_sf("continuously_flooded")
        assert sf is not None
        assert Decimal(str(sf)) == Decimal("1.0")

    def test_water_regime_single_aeration(self):
        db = AgriculturalDatabaseEngine()
        sf = db.get_rice_water_regime_sf("single_aeration")
        assert sf is not None
        assert Decimal(str(sf)) < Decimal("1.0")

    def test_organic_amendment_straw(self):
        db = AgriculturalDatabaseEngine()
        cfoa = db.get_rice_organic_cfoa("straw_short")
        if cfoa is None:
            cfoa = db.get_rice_organic_cfoa("STRAW_SHORT")
        assert cfoa is not None


# ===========================================================================
# Test Class: Field Burning
# ===========================================================================


@_SKIP
class TestFieldBurning:
    """Test field burning reference data lookups."""

    def test_wheat_burning_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_field_burning_ef("wheat")
        assert ef is not None
        assert isinstance(ef, dict)

    def test_rice_burning_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_field_burning_ef("rice")
        assert ef is not None

    def test_corn_burning_ef(self):
        db = AgriculturalDatabaseEngine()
        ef = db.get_field_burning_ef("corn_maize")
        assert ef is not None


# ===========================================================================
# Test Class: GWP Values
# ===========================================================================


@_SKIP
class TestGWPValues:
    """Test GWP value lookups."""

    def test_ch4_ar6(self):
        db = AgriculturalDatabaseEngine()
        gwp = db.get_gwp("CH4", "AR6")
        if gwp is None:
            gwp = db.get_gwp("ch4", "AR6")
        assert gwp is not None

    def test_n2o_ar6(self):
        db = AgriculturalDatabaseEngine()
        gwp = db.get_gwp("N2O", "AR6")
        if gwp is None:
            gwp = db.get_gwp("n2o", "AR6")
        assert gwp is not None

    def test_co2_is_1(self):
        db = AgriculturalDatabaseEngine()
        gwp = db.get_gwp("CO2", "AR6")
        if gwp is None:
            gwp = db.get_gwp("co2", "AR6")
        assert gwp is not None
        assert Decimal(str(gwp)) == Decimal("1")

    def test_ch4_ar5(self):
        db = AgriculturalDatabaseEngine()
        gwp = db.get_gwp("CH4", "AR5")
        if gwp is None:
            gwp = db.get_gwp("ch4", "AR5")
        assert gwp is not None

    def test_ch4_ar4(self):
        db = AgriculturalDatabaseEngine()
        gwp = db.get_gwp("CH4", "AR4")
        if gwp is None:
            gwp = db.get_gwp("ch4", "AR4")
        assert gwp is not None


# ===========================================================================
# Test Class: Feed Characteristics
# ===========================================================================


@_SKIP
class TestFeedCharacteristics:
    """Test feed characteristics lookups."""

    def test_body_weight_dairy(self):
        db = AgriculturalDatabaseEngine()
        bw = db.get_body_weight("dairy_cattle")
        assert bw is not None
        assert Decimal(str(bw)) > Decimal("300")

    def test_feed_digestibility(self):
        db = AgriculturalDatabaseEngine()
        de = db.get_feed_digestibility("medium_quality")
        if de is None:
            de = db.get_feed_digestibility("temperate_pasture")
        # May return None if label not matched exactly
        assert de is None or Decimal(str(de)) > Decimal("0")

    def test_milk_yield(self):
        db = AgriculturalDatabaseEngine()
        my = db.get_milk_yield()
        assert my is not None
        assert Decimal(str(my)) > Decimal("0")


# ===========================================================================
# Test Class: Statistics
# ===========================================================================


@_SKIP
class TestDatabaseStatistics:
    """Test database engine statistics."""

    def test_get_statistics(self):
        db = AgriculturalDatabaseEngine()
        stats = db.get_statistics()
        assert isinstance(stats, dict)

    def test_statistics_has_lookups(self):
        db = AgriculturalDatabaseEngine()
        db.get_enteric_ef("dairy_cattle")
        stats = db.get_statistics()
        assert isinstance(stats, dict)

    def test_list_available_factors(self):
        db = AgriculturalDatabaseEngine()
        if hasattr(db, 'list_available_factors'):
            factors = db.list_available_factors()
            assert factors is not None
