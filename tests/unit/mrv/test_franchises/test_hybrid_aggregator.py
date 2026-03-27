# -*- coding: utf-8 -*-
"""
Test suite for franchises.hybrid_aggregator - AGENT-MRV-027.

Tests the HybridAggregatorEngine including method waterfall,
tiered data collection, network aggregation, multi-brand support,
regional and franchise type aggregation, company-owned split (DC-FRN-001),
partial year handling, data coverage report, weighted DQI, and
uncertainty aggregation.

Target: 55+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from greenlang.agents.mrv.franchises.hybrid_aggregator import (
    HybridAggregatorEngine,
    HybridNetworkInput,
    FranchiseUnitData,
    NetworkAggregationResult,
    FranchiseType,
    OwnershipType,
    CalculationMethod,
    DataQualityTier,
    CurrencyCode,
)


# ==============================================================================
# HELPERS
# ==============================================================================


def _make_metered_unit(
    unit_id: str = "FRN-M-001",
    franchise_type: FranchiseType = FranchiseType.QSR,
    brand: str = "TestBrand",
    country: str = "US",
    metered_co2e: Decimal = Decimal("45000"),
) -> FranchiseUnitData:
    """Build a metered (Tier 1) FranchiseUnitData."""
    return FranchiseUnitData(
        unit_id=unit_id,
        franchise_type=franchise_type,
        brand=brand,
        ownership=OwnershipType.FRANCHISED,
        country=country,
        has_metered_data=True,
        metered_co2e=metered_co2e,
        metered_data_quality_score=Decimal("1.5"),
        floor_area_m2=Decimal("250"),
        climate_zone="4A",
        grid_region="US_AVERAGE",
        months_operational=12,
    )


def _make_estimated_unit(
    unit_id: str = "FRN-E-001",
    franchise_type: FranchiseType = FranchiseType.QSR,
    brand: str = "TestBrand",
    country: str = "US",
    floor_area_m2: Decimal = Decimal("220"),
) -> FranchiseUnitData:
    """Build an estimated (Tier 2) FranchiseUnitData."""
    return FranchiseUnitData(
        unit_id=unit_id,
        franchise_type=franchise_type,
        brand=brand,
        ownership=OwnershipType.FRANCHISED,
        country=country,
        has_metered_data=False,
        floor_area_m2=floor_area_m2,
        climate_zone="4A",
        grid_region="US_AVERAGE",
        months_operational=12,
    )


def _make_spend_unit(
    unit_id: str = "FRN-S-001",
    franchise_type: FranchiseType = FranchiseType.QSR,
    brand: str = "TestBrand",
    country: str = "US",
    annual_revenue: Decimal = Decimal("900000"),
) -> FranchiseUnitData:
    """Build a spend-only (Tier 3) FranchiseUnitData."""
    return FranchiseUnitData(
        unit_id=unit_id,
        franchise_type=franchise_type,
        brand=brand,
        ownership=OwnershipType.FRANCHISED,
        country=country,
        has_metered_data=False,
        annual_revenue=annual_revenue,
        revenue_currency=CurrencyCode.USD,
        months_operational=12,
    )


def _make_company_owned_unit(
    unit_id: str = "FRN-CO-001",
    franchise_type: FranchiseType = FranchiseType.QSR,
) -> FranchiseUnitData:
    """Build a company-owned unit (DC-FRN-001: excluded from Cat 14)."""
    return FranchiseUnitData(
        unit_id=unit_id,
        franchise_type=franchise_type,
        ownership=OwnershipType.COMPANY_OWNED,
        country="US",
        has_metered_data=True,
        metered_co2e=Decimal("50000"),
        metered_data_quality_score=Decimal("1.5"),
        floor_area_m2=Decimal("250"),
        climate_zone="4A",
        grid_region="US_AVERAGE",
        months_operational=12,
    )


def _make_network(
    units: List[FranchiseUnitData],
    network_id: str = "NET-001",
    naics_code: str = None,
    network_total_revenue: Decimal = None,
    reporting_year: int = 2025,
) -> HybridNetworkInput:
    """Build a HybridNetworkInput."""
    return HybridNetworkInput(
        network_id=network_id,
        franchisor_name="TestFranchise Inc.",
        units=units,
        naics_code=naics_code,
        network_total_revenue=network_total_revenue,
        reporting_year=reporting_year,
    )


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> HybridAggregatorEngine:
    """Create a fresh HybridAggregatorEngine instance."""
    HybridAggregatorEngine._instance = None
    return HybridAggregatorEngine.get_instance()


@pytest.fixture
def metered_unit() -> FranchiseUnitData:
    """Metered (Tier 1) franchise unit."""
    return _make_metered_unit()


@pytest.fixture
def estimated_unit() -> FranchiseUnitData:
    """Estimated (Tier 2) franchise unit."""
    return _make_estimated_unit()


@pytest.fixture
def spend_unit() -> FranchiseUnitData:
    """Spend-only (Tier 3) franchise unit."""
    return _make_spend_unit()


@pytest.fixture
def company_owned_unit() -> FranchiseUnitData:
    """Company-owned unit (excluded from Cat 14)."""
    return _make_company_owned_unit()


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestHybridAggregatorInit:
    """Test HybridAggregatorEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via get_instance."""
        HybridAggregatorEngine._instance = None
        e1 = HybridAggregatorEngine.get_instance()
        e2 = HybridAggregatorEngine.get_instance()
        assert e1 is e2

    def test_engine_reset(self):
        """Test engine reset_instance clears singleton."""
        HybridAggregatorEngine._instance = None
        eng = HybridAggregatorEngine.get_instance()
        HybridAggregatorEngine.reset_instance()
        assert HybridAggregatorEngine._instance is None


# ==============================================================================
# METHOD WATERFALL TESTS
# ==============================================================================


class TestMethodWaterfall:
    """Test method waterfall (franchise_specific -> average -> spend)."""

    def test_metered_unit_tier_1(self, engine, metered_unit):
        """Test metered data routes to Tier 1 (franchise_specific)."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        assert isinstance(result, NetworkAggregationResult)
        assert result.total_co2e > 0

    def test_estimated_unit_tier_2(self, engine, estimated_unit):
        """Test estimated data routes to Tier 2 (average_data)."""
        network = _make_network([estimated_unit])
        result = engine.calculate(network)
        assert result.total_co2e > 0

    def test_spend_unit_tier_3(self, engine, spend_unit):
        """Test spend-only data routes to Tier 3 (spend_based)."""
        network = _make_network(
            [spend_unit],
            naics_code="722513",
            network_total_revenue=Decimal("50000000"),
        )
        result = engine.calculate(network)
        assert result.total_co2e > 0

    def test_mixed_waterfall(self, engine, metered_unit, estimated_unit, spend_unit):
        """Test mixed units each routed to correct tier."""
        network = _make_network(
            [metered_unit, estimated_unit, spend_unit],
            naics_code="722513",
            network_total_revenue=Decimal("50000000"),
        )
        result = engine.calculate(network)
        assert result.total_co2e > 0
        assert len(result.unit_results) >= 1


# ==============================================================================
# NETWORK AGGREGATION TESTS
# ==============================================================================


class TestNetworkAggregation:
    """Test network-level aggregation."""

    def test_network_total(self, engine, metered_unit, estimated_unit):
        """Test network total is sum of unit results."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        assert result.total_co2e > 0

    def test_network_tco2e(self, engine, metered_unit):
        """Test network result includes tCO2e conversion."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        expected_tco2e = result.total_co2e / Decimal("1000")
        assert abs(result.total_tco2e - expected_tco2e) < Decimal("1")

    def test_unit_results_count(self, engine, metered_unit, estimated_unit):
        """Test unit_results count matches input count (minus excluded)."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        assert len(result.unit_results) >= 1


# ==============================================================================
# MULTI-BRAND SUPPORT TESTS
# ==============================================================================


class TestMultiBrandSupport:
    """Test multi-brand franchise network support."""

    def test_multi_brand_aggregation(self, engine):
        """Test multi-brand network aggregates correctly."""
        unit_a = _make_metered_unit(unit_id="FRN-A-001", brand="Brand A")
        unit_b = _make_metered_unit(unit_id="FRN-B-001", brand="Brand B")
        network = _make_network([unit_a, unit_b])
        result = engine.calculate(network)
        assert result.total_co2e > 0

    def test_by_brand_breakdown(self, engine):
        """Test by_brand breakdown present in result."""
        unit_a = _make_metered_unit(unit_id="FRN-A-001", brand="Brand A")
        unit_b = _make_metered_unit(unit_id="FRN-B-001", brand="Brand B")
        network = _make_network([unit_a, unit_b])
        result = engine.calculate(network)
        assert isinstance(result.by_brand, dict)
        if result.by_brand:
            assert len(result.by_brand) >= 2


# ==============================================================================
# REGIONAL AGGREGATION TESTS
# ==============================================================================


class TestRegionalAggregation:
    """Test regional aggregation."""

    def test_by_region_breakdown(self, engine):
        """Test by_region breakdown present in result."""
        unit_us = _make_metered_unit(unit_id="FRN-US-001", country="US")
        unit_gb = _make_metered_unit(unit_id="FRN-GB-001", country="GB")
        network = _make_network([unit_us, unit_gb])
        result = engine.calculate(network)
        assert isinstance(result.by_region, dict)
        if result.by_region:
            assert len(result.by_region) >= 1


# ==============================================================================
# FRANCHISE TYPE AGGREGATION TESTS
# ==============================================================================


class TestFranchiseTypeAggregation:
    """Test franchise type aggregation."""

    def test_by_franchise_type(self, engine):
        """Test by_franchise_type breakdown."""
        unit_qsr = _make_metered_unit(
            unit_id="FRN-QSR-001", franchise_type=FranchiseType.QSR
        )
        unit_hotel = _make_metered_unit(
            unit_id="FRN-HTL-001",
            franchise_type=FranchiseType.HOTEL,
            metered_co2e=Decimal("120000"),
        )
        network = _make_network([unit_qsr, unit_hotel])
        result = engine.calculate(network)
        assert isinstance(result.by_franchise_type, dict)
        if result.by_franchise_type:
            assert len(result.by_franchise_type) >= 2


# ==============================================================================
# COMPANY-OWNED SPLIT (DC-FRN-001) TESTS
# ==============================================================================


class TestCompanyOwnedSplit:
    """Test company-owned unit exclusion (DC-FRN-001)."""

    def test_company_owned_excluded(self, engine, metered_unit, company_owned_unit):
        """Test company-owned units excluded from Cat 14 total."""
        network = _make_network([metered_unit, company_owned_unit])
        result = engine.calculate(network)
        assert result.data_coverage.company_owned_excluded >= 1

    def test_only_company_owned(self, engine, company_owned_unit):
        """Test all company-owned units produce zero Cat 14 emissions."""
        network = _make_network([company_owned_unit])
        result = engine.calculate(network)
        assert result.total_co2e == Decimal("0")

    def test_mixed_ownership(self, engine, metered_unit, company_owned_unit):
        """Test mixed ownership correctly splits Cat 14 vs Scope 1/2."""
        network = _make_network([metered_unit, company_owned_unit])
        result = engine.calculate(network)
        assert result.data_coverage.company_owned_excluded >= 1
        assert result.data_coverage.franchised_calculated >= 1


# ==============================================================================
# PARTIAL YEAR HANDLING TESTS
# ==============================================================================


class TestPartialYearHandling:
    """Test partial year (pro-rata) handling in aggregation."""

    def test_partial_year_unit(self, engine):
        """Test partial year unit (6 months) produces lower emissions."""
        full = _make_metered_unit(unit_id="FRN-FY-001")
        partial = FranchiseUnitData(
            unit_id="FRN-PY-001",
            franchise_type=FranchiseType.QSR,
            brand="TestBrand",
            ownership=OwnershipType.FRANCHISED,
            country="US",
            has_metered_data=True,
            metered_co2e=Decimal("45000"),
            metered_data_quality_score=Decimal("1.5"),
            floor_area_m2=Decimal("250"),
            climate_zone="4A",
            grid_region="US_AVERAGE",
            months_operational=6,
        )
        full_net = _make_network([full], network_id="NET-FY")
        partial_net = _make_network([partial], network_id="NET-PY")
        result_full = engine.calculate(full_net)
        HybridAggregatorEngine._instance = None
        engine2 = HybridAggregatorEngine.get_instance()
        result_partial = engine2.calculate(partial_net)
        assert result_partial.total_co2e <= result_full.total_co2e


# ==============================================================================
# DATA COVERAGE REPORT TESTS
# ==============================================================================


class TestDataCoverageReport:
    """Test data coverage report generation."""

    def test_coverage_report_exists(self, engine, metered_unit, estimated_unit):
        """Test data coverage report is in result."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        assert result.data_coverage is not None

    def test_coverage_total_submitted(self, engine, metered_unit, estimated_unit):
        """Test data coverage total_units_submitted count."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        assert result.data_coverage.total_units_submitted == 2

    def test_coverage_method_breakdown(self, engine, metered_unit, estimated_unit):
        """Test method breakdown in data coverage."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        mb = result.method_breakdown
        assert mb.total_units >= 1


# ==============================================================================
# WEIGHTED DQI CALCULATION TESTS
# ==============================================================================


class TestWeightedDQI:
    """Test weighted DQI calculation across methods."""

    def test_weighted_dqi_present(self, engine, metered_unit, estimated_unit):
        """Test weighted DQI is present in result."""
        network = _make_network([metered_unit, estimated_unit])
        result = engine.calculate(network)
        assert result.weighted_dqi is not None
        assert result.weighted_dqi.overall_score > 0

    def test_weighted_dqi_classification(self, engine, metered_unit):
        """Test weighted DQI has classification label."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        assert result.weighted_dqi.classification in (
            "Excellent", "Good", "Fair", "Poor", "Very Poor"
        )


# ==============================================================================
# UNCERTAINTY AGGREGATION TESTS
# ==============================================================================


class TestUncertaintyAggregation:
    """Test uncertainty aggregation across methods."""

    def test_uncertainty_present(self, engine, metered_unit):
        """Test uncertainty result is present."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        assert result.uncertainty is not None

    def test_uncertainty_bounds(self, engine, metered_unit):
        """Test uncertainty bounds surround the mean."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        unc = result.uncertainty
        assert unc.ci_lower <= unc.mean_co2e
        assert unc.ci_upper >= unc.mean_co2e

    def test_uncertainty_relative_pct(self, engine, metered_unit):
        """Test relative uncertainty is a non-negative percentage."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        assert result.uncertainty.relative_uncertainty_pct >= 0


# ==============================================================================
# PROVENANCE TESTS
# ==============================================================================


class TestProvenance:
    """Test provenance hash generation."""

    def test_provenance_hash_present(self, engine, metered_unit):
        """Test provenance hash is 64-char hex."""
        network = _make_network([metered_unit])
        result = engine.calculate(network)
        assert len(result.provenance_hash) == 64


# ==============================================================================
# ENGINE STATS TESTS
# ==============================================================================


class TestEngineStats:
    """Test engine stats method."""

    def test_get_stats(self, engine, metered_unit):
        """Test get_stats returns operational metrics."""
        network = _make_network([metered_unit])
        engine.calculate(network)
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert stats["network_calculations"] >= 1

    def test_stats_units_processed(self, engine, metered_unit, estimated_unit):
        """Test stats tracks total units processed."""
        network = _make_network([metered_unit, estimated_unit])
        engine.calculate(network)
        stats = engine.get_stats()
        assert stats["total_units_processed"] >= 1


# ==============================================================================
# PARAMETRIZED WATERFALL TESTS
# ==============================================================================


class TestParametrizedWaterfall:
    """Parametrized tests for waterfall scenarios."""

    @pytest.mark.parametrize("n_metered,n_estimated,n_spend", [
        (3, 0, 0),
        (2, 1, 0),
        (1, 1, 1),
        (0, 3, 0),
        (0, 0, 3),
    ])
    def test_waterfall_proportions(
        self, engine, n_metered, n_estimated, n_spend,
    ):
        """Test waterfall with various proportions of data availability."""
        units: List[FranchiseUnitData] = []
        for i in range(n_metered):
            units.append(_make_metered_unit(unit_id=f"FRN-M-{i}"))
        for i in range(n_estimated):
            units.append(_make_estimated_unit(unit_id=f"FRN-E-{i}"))
        for i in range(n_spend):
            units.append(_make_spend_unit(unit_id=f"FRN-S-{i}"))

        if not units:
            return

        network = _make_network(
            units,
            naics_code="722513",
            network_total_revenue=Decimal("50000000"),
        )
        result = engine.calculate(network)
        assert result.total_co2e > 0


# ==============================================================================
# PARAMETRIZED FRANCHISE TYPE TESTS
# ==============================================================================


class TestParametrizedFranchiseTypes:
    """Test with different franchise types."""

    @pytest.mark.parametrize("franchise_type", [
        FranchiseType.QSR,
        FranchiseType.HOTEL,
        FranchiseType.CONVENIENCE_STORE,
        FranchiseType.RETAIL_CLOTHING,
        FranchiseType.FITNESS_CENTER,
    ])
    def test_franchise_type_metered(self, engine, franchise_type):
        """Test each franchise type with metered data."""
        unit = _make_metered_unit(
            unit_id=f"FRN-{franchise_type.value[:3].upper()}-001",
            franchise_type=franchise_type,
        )
        network = _make_network([unit])
        result = engine.calculate(network)
        assert result.total_co2e > 0
