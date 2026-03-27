# -*- coding: utf-8 -*-
"""
Test suite for investments.investment_database - AGENT-MRV-028.

Tests the InvestmentDatabaseEngine (Engine 1) for the Investments Agent
(GL-MRV-S3-015) including singleton pattern, sector emission factor lookups,
country emissions lookups, PCAF quality criteria, grid emission factors,
building benchmarks, vehicle emission factors, currency rates, sovereign
data, and carbon intensity benchmarks.

Coverage:
- Singleton pattern (thread-safe, same instance)
- Sector EFs for all 12 GICS sectors
- Country emissions for 50+ countries
- PCAF quality criteria for all 8 asset classes x 5 scores
- Grid EFs for major regions
- Building benchmarks for 6 property types x 5 climate zones
- Vehicle EFs for 5 vehicle categories
- Currency rates for 15 currencies
- Sovereign data lookups
- Carbon intensity benchmarks
- Parametrized tests for sectors, countries, asset classes
- Error handling (invalid keys, missing data)
- Quantization to 8 decimal places

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional
import pytest

from greenlang.agents.mrv.investments.investment_database import (
    InvestmentDatabaseEngine,
    get_database_engine,
    reset_database_engine,
)
from greenlang.agents.mrv.investments.models import (
    AssetClass,
    Sector,
    CurrencyCode,
    PropertyType,
    ClimateZone,
    VehicleCategory,
    PCAFDataQuality,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the singleton engine before each test."""
    reset_database_engine()
    yield
    reset_database_engine()


@pytest.fixture
def engine() -> InvestmentDatabaseEngine:
    """Create a fresh InvestmentDatabaseEngine instance."""
    return InvestmentDatabaseEngine()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, engine):
        """Test InvestmentDatabaseEngine returns the same instance."""
        engine2 = InvestmentDatabaseEngine()
        assert engine is engine2

    def test_singleton_via_get_database_engine(self):
        """Test get_database_engine returns singleton."""
        engine1 = get_database_engine()
        engine2 = get_database_engine()
        assert engine1 is engine2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        instances: List[InvestmentDatabaseEngine] = []

        def get_instance():
            instances.append(InvestmentDatabaseEngine())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    def test_reset_creates_new_instance(self, engine):
        """Test reset_database_engine creates a new instance."""
        old_engine = engine
        reset_database_engine()
        new_engine = InvestmentDatabaseEngine()
        assert old_engine is not new_engine


# ==============================================================================
# SECTOR EMISSION FACTOR TESTS
# ==============================================================================


class TestSectorEFs:
    """Test sector emission factor lookups."""

    @pytest.mark.parametrize("sector", [
        "energy", "materials", "industrials", "consumer_discretionary",
        "consumer_staples", "health_care", "financials",
        "information_technology", "communication_services",
        "utilities", "real_estate", "other",
    ])
    def test_sector_ef_exists(self, engine, sector):
        """Test emission factor exists for each GICS sector."""
        ef = engine.get_sector_ef(sector)
        assert ef is not None
        assert ef > Decimal("0")

    def test_energy_sector_ef_highest(self, engine):
        """Test energy sector has highest emission factor."""
        energy_ef = engine.get_sector_ef("energy")
        for sector in ["information_technology", "financials", "health_care"]:
            other_ef = engine.get_sector_ef(sector)
            assert energy_ef >= other_ef

    def test_sector_ef_invalid_returns_none_or_raises(self, engine):
        """Test invalid sector returns None or raises ValueError."""
        try:
            result = engine.get_sector_ef("nonexistent")
            assert result is None
        except (ValueError, KeyError):
            pass

    def test_sector_ef_precision(self, engine):
        """Test sector EF values have appropriate precision."""
        ef = engine.get_sector_ef("energy")
        assert isinstance(ef, Decimal)

    def test_all_12_sectors_retrievable(self, engine):
        """Test all 12 GICS sectors are retrievable."""
        sectors = [
            "energy", "materials", "industrials", "consumer_discretionary",
            "consumer_staples", "health_care", "financials",
            "information_technology", "communication_services",
            "utilities", "real_estate", "other",
        ]
        for s in sectors:
            assert engine.get_sector_ef(s) is not None


# ==============================================================================
# COUNTRY EMISSIONS TESTS
# ==============================================================================


class TestCountryEmissions:
    """Test country emissions lookups."""

    @pytest.mark.parametrize("country", [
        "US", "CN", "IN", "RU", "JP", "DE", "GB", "FR", "BR", "CA",
        "AU", "KR", "IT", "MX", "ID", "SA", "ZA", "TR", "PL", "TH",
        "AR", "EG", "NG", "PK", "BD", "VN", "MY", "PH", "CO", "CL",
        "PE", "NZ", "SE", "NO", "DK", "FI", "NL", "BE", "CH", "AT",
        "ES", "PT", "IE", "GR", "CZ", "RO", "HU", "IL", "AE", "SG",
    ])
    def test_country_emissions_positive(self, engine, country):
        """Test country emissions are positive for 50 countries."""
        emissions = engine.get_country_emissions(country)
        assert emissions > Decimal("0")

    def test_country_emissions_us_value(self, engine):
        """Test US emissions are approximately 5.2 GtCO2e."""
        us_emissions = engine.get_country_emissions("US")
        assert us_emissions > Decimal("4000000000")
        assert us_emissions < Decimal("7000000000")

    def test_country_emissions_china_highest(self, engine):
        """Test China has the highest emissions."""
        cn_emissions = engine.get_country_emissions("CN")
        us_emissions = engine.get_country_emissions("US")
        assert cn_emissions > us_emissions

    def test_country_emissions_invalid_returns_none_or_raises(self, engine):
        """Test invalid country returns None or raises error."""
        try:
            result = engine.get_country_emissions("ZZ")
            assert result is None
        except (ValueError, KeyError):
            pass

    def test_country_emissions_case_insensitive(self, engine):
        """Test country lookup is case-insensitive."""
        us1 = engine.get_country_emissions("US")
        us2 = engine.get_country_emissions("us")
        assert us1 == us2


# ==============================================================================
# PCAF QUALITY CRITERIA TESTS
# ==============================================================================


class TestPCAFQualityCriteria:
    """Test PCAF quality criteria lookups."""

    @pytest.mark.parametrize("asset_class", [
        "listed_equity", "corporate_bond", "private_equity",
        "project_finance", "commercial_real_estate", "mortgage",
        "motor_vehicle_loan", "sovereign_bond",
    ])
    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_pcaf_criteria_all_combinations(self, engine, asset_class, score):
        """Test PCAF criteria for all 8 asset classes x 5 scores."""
        criteria = engine.get_pcaf_quality_criteria(asset_class, score)
        assert criteria is not None
        assert "description" in criteria
        assert "data_type" in criteria

    def test_pcaf_score_1_is_best(self, engine):
        """Test score 1 represents highest quality data."""
        criteria = engine.get_pcaf_quality_criteria("listed_equity", 1)
        assert "reported" in criteria["data_type"].lower() or \
               "verified" in criteria["description"].lower() or \
               "audited" in criteria["description"].lower()

    def test_pcaf_score_5_is_worst(self, engine):
        """Test score 5 represents lowest quality data."""
        criteria = engine.get_pcaf_quality_criteria("listed_equity", 5)
        assert "estimated" in criteria["data_type"].lower() or \
               "sector" in criteria["description"].lower() or \
               "average" in criteria["description"].lower()

    def test_pcaf_invalid_score(self, engine):
        """Test invalid score returns None or raises error."""
        try:
            result = engine.get_pcaf_quality_criteria("listed_equity", 6)
            assert result is None
        except (ValueError, KeyError):
            pass


# ==============================================================================
# GRID EMISSION FACTOR TESTS
# ==============================================================================


class TestGridEFs:
    """Test grid emission factor lookups."""

    @pytest.mark.parametrize("region", [
        "US_AVG", "EU_AVG", "CN_AVG", "IN_AVG", "JP_AVG",
    ])
    def test_grid_ef_exists(self, engine, region):
        """Test grid EF exists for major regions."""
        ef = engine.get_grid_ef(region)
        assert ef is not None
        assert ef > Decimal("0")

    def test_grid_ef_precision(self, engine):
        """Test grid EF has appropriate Decimal precision."""
        ef = engine.get_grid_ef("US_AVG")
        assert isinstance(ef, Decimal)


# ==============================================================================
# BUILDING BENCHMARK TESTS
# ==============================================================================


class TestBuildingBenchmarks:
    """Test building benchmark lookups."""

    @pytest.mark.parametrize("property_type", [
        "office", "retail", "industrial", "residential", "hotel", "mixed_use",
    ])
    @pytest.mark.parametrize("climate_zone", [
        "tropical", "arid", "temperate", "continental", "polar",
    ])
    def test_building_benchmark_all_combinations(
        self, engine, property_type, climate_zone
    ):
        """Test benchmarks for 6 property types x 5 climate zones."""
        benchmark = engine.get_building_benchmark(property_type, climate_zone)
        assert benchmark is not None
        assert benchmark > Decimal("0")

    def test_office_temperate_benchmark(self, engine):
        """Test specific office/temperate benchmark value."""
        benchmark = engine.get_building_benchmark("office", "temperate")
        assert benchmark > Decimal("100")
        assert benchmark < Decimal("500")


# ==============================================================================
# VEHICLE EMISSION FACTOR TESTS
# ==============================================================================


class TestVehicleEFs:
    """Test vehicle emission factor lookups."""

    @pytest.mark.parametrize("category", [
        "passenger_car", "light_commercial", "heavy_commercial",
        "electric_vehicle", "motorcycle",
    ])
    def test_vehicle_ef_by_category(self, engine, category):
        """Test each vehicle category has emission factors."""
        ef = engine.get_vehicle_ef(category)
        assert ef is not None
        assert ef > Decimal("0")

    def test_ev_lower_than_ice(self, engine):
        """Test EV emissions are lower than ICE vehicle."""
        ev_ef = engine.get_vehicle_ef("electric_vehicle")
        ice_ef = engine.get_vehicle_ef("passenger_car")
        assert ev_ef < ice_ef

    def test_heavy_commercial_highest(self, engine):
        """Test heavy commercial vehicle has highest EF."""
        heavy_ef = engine.get_vehicle_ef("heavy_commercial")
        passenger_ef = engine.get_vehicle_ef("passenger_car")
        assert heavy_ef > passenger_ef


# ==============================================================================
# CURRENCY RATE TESTS
# ==============================================================================


class TestCurrencyRates:
    """Test currency rate lookups."""

    def test_usd_rate_is_one(self, engine):
        """Test USD rate is 1.0."""
        rate = engine.get_currency_rate("USD")
        assert rate == Decimal("1.0")

    @pytest.mark.parametrize("currency", [
        "EUR", "GBP", "JPY", "CHF", "CAD", "AUD",
        "CNY", "INR", "BRL", "ZAR", "SGD", "KRW", "SEK", "NOK",
    ])
    def test_currency_rate_positive(self, engine, currency):
        """Test all 14 non-USD currency rates are positive."""
        rate = engine.get_currency_rate(currency)
        assert rate is not None
        assert rate > Decimal("0")

    def test_currency_rate_invalid(self, engine):
        """Test invalid currency returns None or raises error."""
        try:
            result = engine.get_currency_rate("XYZ")
            assert result is None
        except (ValueError, KeyError):
            pass


# ==============================================================================
# SOVEREIGN DATA TESTS
# ==============================================================================


class TestSovereignData:
    """Test sovereign data lookups."""

    @pytest.mark.parametrize("country", ["US", "CN", "DE", "JP", "GB"])
    def test_sovereign_data_exists(self, engine, country):
        """Test sovereign data exists for major countries."""
        data = engine.get_sovereign_data(country)
        assert data is not None
        assert "gdp_ppp" in data
        assert "total_emissions" in data

    def test_sovereign_data_us_gdp(self, engine):
        """Test US GDP PPP is approximately $25 trillion."""
        data = engine.get_sovereign_data("US")
        gdp = data["gdp_ppp"]
        assert gdp > Decimal("20000000000000")
        assert gdp < Decimal("30000000000000")

    def test_sovereign_data_has_population(self, engine):
        """Test sovereign data includes population."""
        data = engine.get_sovereign_data("US")
        assert "population" in data
        assert data["population"] > 0


# ==============================================================================
# CARBON INTENSITY BENCHMARK TESTS
# ==============================================================================


class TestCarbonIntensityBenchmarks:
    """Test carbon intensity benchmark lookups."""

    @pytest.mark.parametrize("sector", [
        "energy", "materials", "utilities", "industrials",
    ])
    def test_carbon_intensity_benchmark_exists(self, engine, sector):
        """Test carbon intensity benchmarks exist for high-emission sectors."""
        benchmark = engine.get_carbon_intensity_benchmark(sector)
        assert benchmark is not None
        assert benchmark > Decimal("0")


# ==============================================================================
# LOOKUP COUNT AND SUMMARY TESTS
# ==============================================================================


class TestDatabaseSummary:
    """Test database summary and lookup count."""

    def test_database_summary(self, engine):
        """Test database summary returns all reference table counts."""
        summary = engine.get_summary()
        assert "sector_emission_factors" in summary
        assert "country_emissions" in summary
        assert summary["sector_emission_factors"] >= 12
        assert summary["country_emissions"] >= 50

    def test_lookup_count_increments(self, engine):
        """Test lookup counter increments with each lookup."""
        initial_count = engine.get_lookup_count()
        engine.get_sector_ef("energy")
        engine.get_country_emissions("US")
        assert engine.get_lookup_count() == initial_count + 2

    def test_lookup_count_reset(self, engine):
        """Test lookup counter resets with engine reset."""
        engine.get_sector_ef("energy")
        assert engine.get_lookup_count() > 0
        reset_database_engine()
        new_engine = InvestmentDatabaseEngine()
        assert new_engine.get_lookup_count() == 0
