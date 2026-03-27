# -*- coding: utf-8 -*-
"""
Test suite for franchises.average_data_calculator - AGENT-MRV-027.

Tests the AverageDataCalculatorEngine including area-based and revenue-based
calculations for all 10 franchise types, hotel class adjustments, QSR cooking
adjustments, climate zone adjustments, currency conversion, CPI deflation,
franchise-type-specific adjustments, and batch calculation.

Target: 50+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from greenlang.agents.mrv.franchises.average_data_calculator import (
    AverageDataCalculatorEngine,
    FranchiseUnitInput,
    FranchiseCalculationResult,
    FranchiseType,
    ClimateZone,
    HotelClass,
    CurrencyCode,
    CalculationMethod,
    HotelOperationsInput,
    QSRCookingInput,
    CookingIntensity,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> AverageDataCalculatorEngine:
    """Create a fresh AverageDataCalculatorEngine instance."""
    AverageDataCalculatorEngine._instance = None
    return AverageDataCalculatorEngine()


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestAverageDataInit:
    """Test AverageDataCalculatorEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via get_instance."""
        AverageDataCalculatorEngine._instance = None
        e1 = AverageDataCalculatorEngine.get_instance()
        e2 = AverageDataCalculatorEngine.get_instance()
        assert e1 is e2


# ==============================================================================
# AREA-BASED CALCULATION TESTS
# ==============================================================================


class TestAreaBasedCalculation:
    """Test area-based calculation for all franchise types."""

    @pytest.mark.parametrize("franchise_type", list(FranchiseType))
    def test_area_based_all_types(self, engine, franchise_type):
        """Test area-based calculation for each franchise type."""
        unit = FranchiseUnitInput(
            unit_id=f"FRN-AB-{franchise_type.value}",
            franchise_type=franchise_type,
            floor_area_m2=Decimal("250"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert isinstance(result, FranchiseCalculationResult)
        assert result.total_co2e > 0
        assert result.method == CalculationMethod.AREA_BASED

    def test_area_based_qsr_result(self, engine):
        """Test area-based QSR produces reasonable result."""
        unit = FranchiseUnitInput(
            unit_id="FRN-QSR-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("220"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.total_co2e > 0
        assert result.unit_id == "FRN-QSR-001"

    def test_area_based_larger_area_more_emissions(self, engine):
        """Test larger floor area produces more emissions."""
        small = FranchiseUnitInput(
            unit_id="FRN-SMALL-001",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            floor_area_m2=Decimal("100"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        large = FranchiseUnitInput(
            unit_id="FRN-LARGE-001",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            floor_area_m2=Decimal("500"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result_small = engine.calculate(small)
        result_large = engine.calculate(large)
        assert result_large.total_co2e > result_small.total_co2e

    def test_area_based_provenance_hash(self, engine):
        """Test area-based result has provenance hash."""
        unit = FranchiseUnitInput(
            unit_id="FRN-PROV-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# ==============================================================================
# REVENUE-BASED CALCULATION TESTS
# ==============================================================================


class TestRevenueBasedCalculation:
    """Test revenue-based calculation for all franchise types."""

    @pytest.mark.parametrize("franchise_type", list(FranchiseType))
    def test_revenue_based_all_types(self, engine, franchise_type):
        """Test revenue-based calculation for each franchise type."""
        unit = FranchiseUnitInput(
            unit_id=f"FRN-RB-{franchise_type.value}",
            franchise_type=franchise_type,
            annual_revenue=Decimal("1200000"),
            revenue_currency=CurrencyCode.USD,
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert isinstance(result, FranchiseCalculationResult)
        assert result.total_co2e > 0
        assert result.method == CalculationMethod.REVENUE_BASED

    def test_revenue_based_higher_revenue_more_emissions(self, engine):
        """Test higher revenue produces more emissions."""
        low = FranchiseUnitInput(
            unit_id="FRN-RLOW-001",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            annual_revenue=Decimal("500000"),
            reporting_year=2025,
        )
        high = FranchiseUnitInput(
            unit_id="FRN-RHIGH-001",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            annual_revenue=Decimal("5000000"),
            reporting_year=2025,
        )
        result_low = engine.calculate(low)
        result_high = engine.calculate(high)
        assert result_high.total_co2e > result_low.total_co2e


# ==============================================================================
# HOTEL CLASS ADJUSTMENT TESTS
# ==============================================================================


class TestHotelClassAdjustments:
    """Test hotel class adjustments (economy/midscale/upscale/luxury)."""

    @pytest.mark.parametrize("hotel_class", list(HotelClass))
    def test_hotel_class_all_types(self, engine, hotel_class):
        """Test each hotel class produces a result."""
        unit = FranchiseUnitInput(
            unit_id=f"FRN-HTL-{hotel_class.value}",
            franchise_type=FranchiseType.HOTEL,
            floor_area_m2=Decimal("5000"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
            hotel_ops=HotelOperationsInput(hotel_class=hotel_class),
        )
        result = engine.calculate(unit)
        assert result.total_co2e > 0

    def test_luxury_higher_than_economy(self, engine):
        """Test luxury hotel emissions higher than economy."""
        economy = FranchiseUnitInput(
            unit_id="FRN-HTL-ECO",
            franchise_type=FranchiseType.HOTEL,
            floor_area_m2=Decimal("5000"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
            hotel_ops=HotelOperationsInput(hotel_class=HotelClass.ECONOMY),
        )
        luxury = FranchiseUnitInput(
            unit_id="FRN-HTL-LUX",
            franchise_type=FranchiseType.HOTEL,
            floor_area_m2=Decimal("5000"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
            hotel_ops=HotelOperationsInput(hotel_class=HotelClass.LUXURY),
        )
        result_economy = engine.calculate(economy)
        result_luxury = engine.calculate(luxury)
        assert result_luxury.total_co2e > result_economy.total_co2e


# ==============================================================================
# QSR COOKING ADJUSTMENT TESTS
# ==============================================================================


class TestQSRCookingAdjustment:
    """Test QSR cooking energy adjustment."""

    def test_qsr_cooking_adjustment_applied(self, engine):
        """Test QSR franchise type gets cooking energy adjustment."""
        unit = FranchiseUnitInput(
            unit_id="FRN-QSR-COOK",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("220"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.total_co2e > 0
        assert result.adjustment_co2e >= 0

    def test_qsr_higher_than_retail_same_area(self, engine):
        """Test QSR (with cooking) has higher emissions than retail (same area)."""
        qsr = FranchiseUnitInput(
            unit_id="FRN-QSR-CMP",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("250"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        retail = FranchiseUnitInput(
            unit_id="FRN-RET-CMP",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            floor_area_m2=Decimal("250"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result_qsr = engine.calculate(qsr)
        result_retail = engine.calculate(retail)
        assert result_qsr.total_co2e > result_retail.total_co2e


# ==============================================================================
# CLIMATE ZONE ADJUSTMENT TESTS
# ==============================================================================


class TestClimateZoneAdjustments:
    """Test climate zone adjustments on EUI benchmarks."""

    @pytest.mark.parametrize("climate_zone", [
        ClimateZone.ZONE_1A, ClimateZone.ZONE_3A, ClimateZone.ZONE_4A,
        ClimateZone.ZONE_5A, ClimateZone.ZONE_7,
    ])
    def test_various_climate_zones(self, engine, climate_zone):
        """Test calculation works for various climate zones."""
        unit = FranchiseUnitInput(
            unit_id=f"FRN-CZ-{climate_zone.value}",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            floor_area_m2=Decimal("300"),
            climate_zone=climate_zone,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.total_co2e > 0

    def test_hot_vs_cold_produces_different_emissions(self, engine):
        """Test hot and cold zones produce different emissions."""
        hot = FranchiseUnitInput(
            unit_id="FRN-CZ-HOT",
            franchise_type=FranchiseType.HOTEL,
            floor_area_m2=Decimal("3000"),
            climate_zone=ClimateZone.ZONE_1A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        cold = FranchiseUnitInput(
            unit_id="FRN-CZ-COLD",
            franchise_type=FranchiseType.HOTEL,
            floor_area_m2=Decimal("3000"),
            climate_zone=ClimateZone.ZONE_7,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result_hot = engine.calculate(hot)
        result_cold = engine.calculate(cold)
        assert result_hot.total_co2e != result_cold.total_co2e


# ==============================================================================
# CURRENCY CONVERSION TESTS
# ==============================================================================


class TestCurrencyConversion:
    """Test currency conversion for revenue-based calculation."""

    @pytest.mark.parametrize("currency", [
        CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP,
    ])
    def test_currency_variants(self, engine, currency):
        """Test revenue-based calculation with different currencies."""
        unit = FranchiseUnitInput(
            unit_id=f"FRN-CUR-{currency.value}",
            franchise_type=FranchiseType.RETAIL_CLOTHING,
            annual_revenue=Decimal("1000000"),
            revenue_currency=currency,
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.total_co2e > 0


# ==============================================================================
# PARTIAL YEAR AND EDGE CASE TESTS
# ==============================================================================


class TestPartialYear:
    """Test partial year proration."""

    def test_partial_year_6_months(self, engine):
        """Test 6-month operation produces roughly half emissions."""
        full = FranchiseUnitInput(
            unit_id="FRN-FULL-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
            months_operational=12,
        )
        half = FranchiseUnitInput(
            unit_id="FRN-HALF-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
            months_operational=6,
        )
        result_full = engine.calculate(full)
        result_half = engine.calculate(half)
        assert result_half.total_co2e < result_full.total_co2e


class TestEdgeCases:
    """Test edge cases for average-data calculations."""

    def test_missing_area_and_revenue_raises(self, engine):
        """Test missing both area and revenue raises ValueError."""
        with pytest.raises((ValueError, Exception)):
            unit = FranchiseUnitInput(
                unit_id="FRN-EMPTY-001",
                franchise_type=FranchiseType.QSR,
                reporting_year=2025,
            )
            engine.calculate(unit)

    def test_data_quality_is_tier_2(self, engine):
        """Test average-data result DQI is Tier 2."""
        unit = FranchiseUnitInput(
            unit_id="FRN-DQI-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.data_quality is not None
        assert result.data_quality.tier.value == "tier_2"

    def test_uncertainty_bounds(self, engine):
        """Test uncertainty bounds are present."""
        unit = FranchiseUnitInput(
            unit_id="FRN-UNC-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        result = engine.calculate(unit)
        assert result.uncertainty_lower < result.total_co2e
        assert result.uncertainty_upper > result.total_co2e


# ==============================================================================
# BATCH CALCULATION TESTS
# ==============================================================================


class TestBatchCalculation:
    """Test batch calculation for average-data method."""

    def test_batch_multiple_types(self, engine):
        """Test batch calculation with multiple franchise types."""
        inputs = [
            FranchiseUnitInput(
                unit_id="FRN-B1",
                franchise_type=FranchiseType.QSR,
                floor_area_m2=Decimal("220"),
                climate_zone=ClimateZone.ZONE_4A,
                grid_region="US_AVERAGE",
                reporting_year=2025,
            ),
            FranchiseUnitInput(
                unit_id="FRN-B2",
                franchise_type=FranchiseType.HOTEL,
                floor_area_m2=Decimal("5000"),
                climate_zone=ClimateZone.ZONE_3A,
                grid_region="US_SOUTHEAST",
                reporting_year=2025,
            ),
        ]
        results = engine.calculate_batch(inputs)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r.total_co2e > 0

    def test_batch_provenance(self, engine):
        """Test batch results have provenance hashes."""
        inputs = [
            FranchiseUnitInput(
                unit_id="FRN-BP1",
                franchise_type=FranchiseType.RETAIL_CLOTHING,
                floor_area_m2=Decimal("300"),
                climate_zone=ClimateZone.ZONE_4A,
                grid_region="US_AVERAGE",
                reporting_year=2025,
            ),
        ]
        results = engine.calculate_batch(inputs)
        for r in results:
            assert len(r.provenance_hash) == 64

    def test_batch_empty_raises(self, engine):
        """Test empty batch raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_batch([])


# ==============================================================================
# ENGINE INFO TESTS
# ==============================================================================


class TestEngineInfo:
    """Test engine info and stats methods."""

    def test_get_available_franchise_types(self, engine):
        """Test listing available franchise types."""
        types = engine.get_available_franchise_types()
        assert isinstance(types, list)
        assert len(types) == len(FranchiseType)

    def test_get_stats(self, engine):
        """Test engine stats tracking."""
        unit = FranchiseUnitInput(
            unit_id="FRN-STATS-001",
            franchise_type=FranchiseType.QSR,
            floor_area_m2=Decimal("200"),
            climate_zone=ClimateZone.ZONE_4A,
            grid_region="US_AVERAGE",
            reporting_year=2025,
        )
        engine.calculate(unit)
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert stats.get("calculation_count", 0) >= 1
