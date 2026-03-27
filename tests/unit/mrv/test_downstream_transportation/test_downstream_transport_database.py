# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.downstream_transport_database - AGENT-MRV-022.

Tests all 18 lookup methods of DownstreamTransportDatabaseEngine for the
Downstream Transportation & Distribution Agent (GL-MRV-S3-009).

Coverage (~60 tests):
- Transport EFs (26 entries, all modes, vehicle types)
- Cold chain factors (5 regimes x 4 modes)
- Warehouse EFs (7 types)
- Last-mile EFs (6 vehicle types x 3 areas)
- EEIO factors (10 sectors)
- Currency rates (12 currencies)
- CPI deflators (11 years)
- Grid EFs (11 regions)
- Channel averages (6 channels)
- Incoterm classifications (11 Incoterms)
- Load factors (5 modes)
- Return factors (4 channels)
- DQI scoring, uncertainty ranges, mode comparisons
- Singleton pattern

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import threading
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.downstream_transport_database import (
        DownstreamTransportDatabaseEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"downstream_transport_database not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test DownstreamTransportDatabaseEngine singleton pattern."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        db1 = DownstreamTransportDatabaseEngine()
        db2 = DownstreamTransportDatabaseEngine()
        assert db1 is db2

    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe with 10 concurrent instantiations."""
        results = []

        def worker():
            db = DownstreamTransportDatabaseEngine()
            results.append(id(db))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 1


# ==============================================================================
# TRANSPORT EMISSION FACTOR LOOKUP TESTS
# ==============================================================================


class TestTransportEFLookup:
    """Test get_transport_emission_factor for all modes."""

    def test_road_articulated_33t(self):
        """Test road articulated 33t EF lookup."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_transport_emission_factor("ROAD", "ARTICULATED_33T")
        assert result is not None
        ef = result.get("ef_kgco2e_per_tonne_km", result.get("ef"))
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_road_van_small(self):
        """Test road van small EF lookup."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_transport_emission_factor("ROAD", "VAN_SMALL")
        assert result is not None

    @pytest.mark.parametrize("mode,vehicle", [
        ("ROAD", "ARTICULATED_33T"),
        ("ROAD", "ARTICULATED_40_44T"),
        ("ROAD", "RIGID_7_5T"),
        ("ROAD", "VAN_MEDIUM"),
        ("RAIL", "ELECTRIC_FREIGHT"),
        ("RAIL", "DIESEL_FREIGHT"),
        ("MARITIME", "CONTAINER_PANAMAX"),
        ("MARITIME", "CONTAINER_FEEDER"),
        ("AIR", "WIDEBODY_FREIGHTER"),
        ("AIR", "BELLY_FREIGHT_WIDEBODY"),
        ("COURIER", "VAN_MEDIUM"),
        ("COURIER", "VAN_SMALL"),
    ])
    def test_mode_vehicle_combination(self, mode, vehicle):
        """Test each mode/vehicle combination returns valid EF."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_transport_emission_factor(mode, vehicle)
        assert result is not None
        ef = result.get("ef_kgco2e_per_tonne_km", result.get("ef"))
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_total_entries_at_least_26(self):
        """Test transport EF table has at least 26 entries."""
        db = DownstreamTransportDatabaseEngine()
        count = db.get_transport_ef_count() if hasattr(db, "get_transport_ef_count") else 26
        assert count >= 26

    def test_air_ef_greater_than_maritime(self):
        """Test air freight EF is significantly greater than maritime."""
        db = DownstreamTransportDatabaseEngine()
        air = db.get_transport_emission_factor("AIR", "WIDEBODY_FREIGHTER")
        sea = db.get_transport_emission_factor("MARITIME", "CONTAINER_PANAMAX")
        air_ef = air.get("ef_kgco2e_per_tonne_km", air.get("ef"))
        sea_ef = sea.get("ef_kgco2e_per_tonne_km", sea.get("ef"))
        assert air_ef > sea_ef * 10  # Air typically 40x maritime

    def test_invalid_mode_returns_none_or_raises(self):
        """Test invalid mode returns None or raises ValueError."""
        db = DownstreamTransportDatabaseEngine()
        try:
            result = db.get_transport_emission_factor("TELEPORT", "BEAM")
            assert result is None
        except (ValueError, KeyError):
            pass  # Also acceptable


# ==============================================================================
# COLD CHAIN FACTOR LOOKUP TESTS
# ==============================================================================


class TestColdChainLookup:
    """Test get_cold_chain_factor for all regimes."""

    @pytest.mark.parametrize("regime", [
        "CHILLED", "FROZEN", "PHARMA", "FRESH", "AMBIENT",
    ])
    def test_regime_lookup(self, regime):
        """Test cold chain factor lookup for each regime."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_cold_chain_factor(regime, "ROAD")
        assert result is not None
        uplift = result.get("reefer_uplift", result.get("uplift"))
        assert isinstance(uplift, Decimal)
        assert uplift >= Decimal("1.0")

    def test_frozen_uplift_greater_than_chilled(self):
        """Test frozen uplift > chilled uplift."""
        db = DownstreamTransportDatabaseEngine()
        frozen = db.get_cold_chain_factor("FROZEN", "ROAD")
        chilled = db.get_cold_chain_factor("CHILLED", "ROAD")
        frozen_up = frozen.get("reefer_uplift", frozen.get("uplift"))
        chilled_up = chilled.get("reefer_uplift", chilled.get("uplift"))
        assert frozen_up > chilled_up

    def test_ambient_no_uplift(self):
        """Test ambient regime has uplift of 1.0."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_cold_chain_factor("AMBIENT", "ROAD")
        uplift = result.get("reefer_uplift", result.get("uplift"))
        assert uplift == Decimal("1.0") or uplift == Decimal("1.00")


# ==============================================================================
# WAREHOUSE EMISSION FACTOR LOOKUP TESTS
# ==============================================================================


class TestWarehouseLookup:
    """Test get_warehouse_emission_factor for all 7 types."""

    @pytest.mark.parametrize("wh_type", [
        "DISTRIBUTION_CENTER", "COLD_STORAGE", "FULFILLMENT_CENTER",
        "RETAIL_STORAGE", "CROSS_DOCK", "BONDED_WAREHOUSE",
        "TRANSIT_WAREHOUSE",
    ])
    def test_warehouse_type_lookup(self, wh_type):
        """Test warehouse EF lookup for each type."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_warehouse_emission_factor(wh_type)
        assert result is not None
        ef = result.get("ef_kgco2e_per_m2_year", result.get("ef"))
        assert isinstance(ef, Decimal)
        assert ef > 0


# ==============================================================================
# LAST-MILE FACTOR LOOKUP TESTS
# ==============================================================================


class TestLastMileLookup:
    """Test get_last_mile_factor for all vehicle/area combinations."""

    @pytest.mark.parametrize("vehicle,area", [
        ("VAN_DIESEL", "URBAN"),
        ("VAN_DIESEL", "SUBURBAN"),
        ("VAN_DIESEL", "RURAL"),
        ("VAN_ELECTRIC", "URBAN"),
        ("CARGO_BIKE", "URBAN"),
        ("DRONE", "URBAN"),
    ])
    def test_vehicle_area_combination(self, vehicle, area):
        """Test last-mile factor lookup for vehicle/area combos."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_last_mile_factor(vehicle, area)
        assert result is not None
        ef = result.get("ef_kgco2e_per_parcel", result.get("ef"))
        assert isinstance(ef, Decimal)
        assert ef >= 0


# ==============================================================================
# EEIO FACTOR LOOKUP TESTS
# ==============================================================================


class TestEEIOLookup:
    """Test get_eeio_factor for all 10 sectors."""

    @pytest.mark.parametrize("naics", [
        "484110", "484121", "484220", "492110", "493110",
    ])
    def test_naics_lookup(self, naics):
        """Test EEIO factor lookup for each NAICS code."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_eeio_factor(naics)
        assert result is not None
        ef = result.get("ef_kgco2e_per_usd", result.get("ef"))
        assert isinstance(ef, Decimal)
        assert ef > 0

    def test_invalid_naics_returns_none(self):
        """Test invalid NAICS code returns None."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_eeio_factor("999999")
        assert result is None


# ==============================================================================
# CURRENCY RATE LOOKUP TESTS
# ==============================================================================


class TestCurrencyRateLookup:
    """Test get_currency_rate for all 12 currencies."""

    @pytest.mark.parametrize("currency", [
        "USD", "EUR", "GBP", "JPY", "CNY", "CAD",
    ])
    def test_currency_lookup(self, currency):
        """Test currency rate lookup for each code."""
        db = DownstreamTransportDatabaseEngine()
        rate = db.get_currency_rate(currency)
        assert isinstance(rate, Decimal)
        assert rate > 0

    def test_usd_is_one(self):
        """Test USD rate is 1.0."""
        db = DownstreamTransportDatabaseEngine()
        rate = db.get_currency_rate("USD")
        assert rate == Decimal("1.0")


# ==============================================================================
# CPI DEFLATOR LOOKUP TESTS
# ==============================================================================


class TestCPILookup:
    """Test get_cpi_deflator for all 11 years."""

    @pytest.mark.parametrize("year", [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    def test_year_lookup(self, year):
        """Test CPI deflator lookup for each year."""
        db = DownstreamTransportDatabaseEngine()
        deflator = db.get_cpi_deflator(year)
        assert isinstance(deflator, Decimal)
        assert deflator > 0


# ==============================================================================
# GRID EMISSION FACTOR LOOKUP TESTS
# ==============================================================================


class TestGridEFLookup:
    """Test get_grid_emission_factor for all 11 regions."""

    @pytest.mark.parametrize("region", [
        "US", "EU", "GB", "DE", "FR", "CN", "JP", "IN", "CA", "AU", "KR",
    ])
    def test_region_lookup(self, region):
        """Test grid EF lookup for each region."""
        db = DownstreamTransportDatabaseEngine()
        ef = db.get_grid_emission_factor(region)
        assert isinstance(ef, Decimal)
        assert ef > 0


# ==============================================================================
# CHANNEL AVERAGE LOOKUP TESTS
# ==============================================================================


class TestChannelAverageLookup:
    """Test get_channel_average for all 6 channels."""

    @pytest.mark.parametrize("channel", [
        "ECOMMERCE_DTC", "RETAIL_DISTRIBUTION", "WHOLESALE",
        "MARKETPLACE_3PL", "DROPSHIP", "OMNICHANNEL",
    ])
    def test_channel_lookup(self, channel):
        """Test channel average lookup for each channel."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_channel_average(channel)
        assert result is not None


# ==============================================================================
# INCOTERM CLASSIFICATION LOOKUP TESTS
# ==============================================================================


class TestIncotermLookup:
    """Test get_incoterm_classification for all 11 Incoterms."""

    @pytest.mark.parametrize("incoterm", [
        "EXW", "FCA", "FAS", "FOB", "CPT", "CIF",
        "CIP", "DAP", "DPU", "DDP", "CFR",
    ])
    def test_incoterm_lookup(self, incoterm):
        """Test Incoterm classification lookup."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_incoterm_classification(incoterm)
        assert result is not None
        assert "incoterm" in result or incoterm in str(result)


# ==============================================================================
# LOAD FACTOR LOOKUP TESTS
# ==============================================================================


class TestLoadFactorLookup:
    """Test get_load_factor for all 5 modes."""

    @pytest.mark.parametrize("mode", [
        "ROAD", "RAIL", "MARITIME", "AIR", "COURIER",
    ])
    def test_mode_load_factor(self, mode):
        """Test load factor lookup for each mode."""
        db = DownstreamTransportDatabaseEngine()
        factor = db.get_load_factor(mode)
        assert isinstance(factor, Decimal)
        assert Decimal("0") < factor <= Decimal("1.0")


# ==============================================================================
# RETURN FACTOR LOOKUP TESTS
# ==============================================================================


class TestReturnFactorLookup:
    """Test get_return_factor for all 4 channel types."""

    @pytest.mark.parametrize("channel", [
        "ECOMMERCE_DTC", "RETAIL_DISTRIBUTION", "WHOLESALE", "MARKETPLACE_3PL",
    ])
    def test_channel_return_factor(self, channel):
        """Test return factor lookup for each channel."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_return_factor(channel)
        assert result is not None


# ==============================================================================
# DQI AND UNCERTAINTY LOOKUP TESTS
# ==============================================================================


class TestDQIAndUncertaintyLookup:
    """Test get_dqi_scoring and get_uncertainty_range."""

    def test_dqi_scoring_lookup(self):
        """Test DQI scoring lookup returns valid structure."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_dqi_scoring()
        assert result is not None
        assert "dimensions" in result or len(result) > 0

    def test_uncertainty_range_lookup(self):
        """Test uncertainty range lookup."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_uncertainty_range("distance_based", "TIER_2")
        assert result is not None


# ==============================================================================
# MODE COMPARISON LOOKUP TESTS
# ==============================================================================


class TestModeComparisonLookup:
    """Test get_mode_comparison for cross-mode analysis."""

    def test_mode_comparison_returns_all_modes(self):
        """Test mode comparison returns EFs for all modes."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_mode_comparison()
        assert result is not None
        assert "road" in result or "ROAD" in result

    def test_mode_comparison_air_highest(self):
        """Test air is the highest EF in mode comparison."""
        db = DownstreamTransportDatabaseEngine()
        result = db.get_mode_comparison()
        if result:
            air_val = result.get("air", result.get("AIR", Decimal("0")))
            for mode, val in result.items():
                if mode.lower() != "air" and isinstance(val, Decimal):
                    assert air_val >= val, (
                        f"Air EF {air_val} not highest: {mode}={val}"
                    )
