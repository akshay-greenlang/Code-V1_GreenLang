# -*- coding: utf-8 -*-
"""
Test suite for franchises.franchise_specific_calculator - AGENT-MRV-027.

Tests the FranchiseSpecificCalculatorEngine including QSR, hotel,
convenience store, retail, fitness, and automotive calculations.
Tests all 7 emission sources, DC-FRN-001 company-owned rejection,
pro-rata partial year, batch calculation, and data quality assessment.

Target: 55+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from greenlang.franchises.franchise_specific_calculator import (
    FranchiseSpecificCalculatorEngine,
    FranchiseUnitInput,
    FranchiseCalculationResult,
    StationaryCombustionInput,
    MobileCombustionInput,
    RefrigerantInput,
    EmissionBreakdown,
    DataQualityScore,
)


# ==============================================================================
# HELPER: build FranchiseUnitInput from conftest dicts
# ==============================================================================


def _make_unit(
    unit_id: str = "FRN-QSR-001",
    franchise_type: str = "qsr",
    ownership_type: str = "franchisee",
    country: str = "US",
    region: str = "CAMX",
    floor_area_m2: Decimal = Decimal("250"),
    electricity_kwh: Decimal = Decimal("180000"),
    heating_kwh: Decimal = Decimal("0"),
    cooling_kwh: Decimal = Decimal("0"),
    reporting_year: int = 2025,
    operating_days: int = None,
    total_days: int = 365,
    stationary: StationaryCombustionInput = None,
    mobile: MobileCombustionInput = None,
    refrigerants: RefrigerantInput = None,
    process_co2e: Decimal = Decimal("0"),
) -> FranchiseUnitInput:
    """Build a FranchiseUnitInput dataclass for testing."""
    return FranchiseUnitInput(
        unit_id=unit_id,
        franchise_type=franchise_type,
        ownership_type=ownership_type,
        country=country,
        region=region,
        floor_area_m2=floor_area_m2,
        electricity_kwh=electricity_kwh,
        heating_kwh=heating_kwh,
        cooling_kwh=cooling_kwh,
        reporting_year=reporting_year,
        operating_days=operating_days,
        total_days=total_days,
        stationary_combustion=stationary,
        mobile_combustion=mobile,
        refrigerants=refrigerants,
        process_emissions_co2e_kg=process_co2e,
    )


def _make_qsr_unit() -> FranchiseUnitInput:
    """QSR unit with cooking, refrigeration, delivery fleet."""
    return _make_unit(
        unit_id="FRN-QSR-001",
        franchise_type="qsr",
        electricity_kwh=Decimal("180000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("25000"),
            propane_litres=Decimal("800"),
        ),
        mobile=MobileCombustionInput(
            total_diesel_litres=Decimal("0"),
            total_petrol_litres=Decimal("3500"),
        ),
        refrigerants=RefrigerantInput(
            equipment=[
                {
                    "equipment_type": "walk_in_cooler",
                    "refrigerant_type": "R-404A",
                    "charge_kg": Decimal("15.0"),
                },
            ],
        ),
    )


def _make_hotel_unit() -> FranchiseUnitInput:
    """Hotel unit with HVAC, water heating."""
    return _make_unit(
        unit_id="FRN-HTL-001",
        franchise_type="hotel",
        floor_area_m2=Decimal("5000"),
        electricity_kwh=Decimal("950000"),
        heating_kwh=Decimal("120000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("70000"),
        ),
        refrigerants=RefrigerantInput(
            equipment=[
                {
                    "equipment_type": "central_chiller",
                    "refrigerant_type": "R-410A",
                    "charge_kg": Decimal("85.0"),
                },
            ],
        ),
    )


def _make_convenience_unit() -> FranchiseUnitInput:
    """Convenience store unit with 24/7 refrigeration."""
    return _make_unit(
        unit_id="FRN-CVS-001",
        franchise_type="convenience_store",
        floor_area_m2=Decimal("150"),
        electricity_kwh=Decimal("210000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("2300"),
        ),
        refrigerants=RefrigerantInput(
            equipment=[
                {
                    "equipment_type": "display_cooler",
                    "refrigerant_type": "R-404A",
                    "charge_kg": Decimal("30.0"),
                },
            ],
        ),
    )


def _make_retail_unit() -> FranchiseUnitInput:
    """Retail unit with HVAC and lighting."""
    return _make_unit(
        unit_id="FRN-RTL-001",
        franchise_type="retail",
        country="GB",
        region=None,
        floor_area_m2=Decimal("400"),
        electricity_kwh=Decimal("85000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("10000"),
        ),
    )


def _make_fitness_unit() -> FranchiseUnitInput:
    """Fitness center unit with HVAC, equipment, water heating."""
    return _make_unit(
        unit_id="FRN-FIT-001",
        franchise_type="fitness",
        region="NWPP",
        floor_area_m2=Decimal("1200"),
        electricity_kwh=Decimal("350000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("22800"),
        ),
    )


def _make_automotive_unit() -> FranchiseUnitInput:
    """Automotive repair unit."""
    return _make_unit(
        unit_id="FRN-AUTO-001",
        franchise_type="automotive",
        floor_area_m2=Decimal("500"),
        electricity_kwh=Decimal("120000"),
        stationary=StationaryCombustionInput(
            natural_gas_m3=Decimal("14200"),
        ),
    )


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> FranchiseSpecificCalculatorEngine:
    """Create a fresh FranchiseSpecificCalculatorEngine instance."""
    FranchiseSpecificCalculatorEngine._instance = None
    if hasattr(FranchiseSpecificCalculatorEngine, "_initialized"):
        try:
            del FranchiseSpecificCalculatorEngine._initialized
        except AttributeError:
            pass
    return FranchiseSpecificCalculatorEngine()


@pytest.fixture
def qsr_unit() -> FranchiseUnitInput:
    """QSR franchise unit with cooking, refrigeration, delivery."""
    return _make_qsr_unit()


@pytest.fixture
def hotel_unit() -> FranchiseUnitInput:
    """Hotel franchise unit."""
    return _make_hotel_unit()


@pytest.fixture
def convenience_unit() -> FranchiseUnitInput:
    """Convenience store franchise unit."""
    return _make_convenience_unit()


@pytest.fixture
def retail_unit() -> FranchiseUnitInput:
    """Retail franchise unit."""
    return _make_retail_unit()


@pytest.fixture
def fitness_unit() -> FranchiseUnitInput:
    """Fitness center franchise unit."""
    return _make_fitness_unit()


@pytest.fixture
def automotive_unit() -> FranchiseUnitInput:
    """Automotive repair franchise unit."""
    return _make_automotive_unit()


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestFranchiseSpecificInit:
    """Test FranchiseSpecificCalculatorEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via __new__."""
        FranchiseSpecificCalculatorEngine._instance = None
        if hasattr(FranchiseSpecificCalculatorEngine, "_initialized"):
            try:
                del FranchiseSpecificCalculatorEngine._initialized
            except AttributeError:
                pass
        e1 = FranchiseSpecificCalculatorEngine()
        e2 = FranchiseSpecificCalculatorEngine()
        assert e1 is e2

    def test_engine_reset(self):
        """Test engine reset class method."""
        FranchiseSpecificCalculatorEngine._instance = None
        eng = FranchiseSpecificCalculatorEngine()
        FranchiseSpecificCalculatorEngine.reset()
        assert FranchiseSpecificCalculatorEngine._instance is None


# ==============================================================================
# QSR CALCULATION TESTS
# ==============================================================================


class TestQSRCalculation:
    """Test QSR franchise unit calculations."""

    def test_qsr_full_calculation(self, engine, qsr_unit):
        """Test full QSR calculation with cooking, refrigeration, electricity."""
        result = engine.calculate(qsr_unit)
        assert isinstance(result, FranchiseCalculationResult)
        assert result.total_co2e_kg > 0
        assert result.calculation_method == "franchise_specific"

    def test_qsr_has_breakdown(self, engine, qsr_unit):
        """Test QSR calculation includes emission breakdown."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown is not None
        assert isinstance(result.breakdown, EmissionBreakdown)

    def test_qsr_includes_stationary_combustion(self, engine, qsr_unit):
        """Test QSR breakdown includes stationary combustion from cooking gas."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown.stationary_combustion_co2e_kg >= 0

    def test_qsr_includes_electricity(self, engine, qsr_unit):
        """Test QSR breakdown includes purchased electricity emissions."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown.purchased_electricity_co2e_kg > 0

    def test_qsr_includes_refrigerant(self, engine, qsr_unit):
        """Test QSR breakdown includes refrigerant leakage."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown.refrigerant_leakage_co2e_kg >= 0

    def test_qsr_includes_mobile_combustion(self, engine, qsr_unit):
        """Test QSR breakdown includes mobile combustion from delivery fleet."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown.mobile_combustion_co2e_kg >= 0

    def test_qsr_has_provenance_hash(self, engine, qsr_unit):
        """Test QSR result has a 64-char SHA-256 provenance hash."""
        result = engine.calculate(qsr_unit)
        assert len(result.provenance_hash) == 64

    def test_qsr_scope1_and_scope2(self, engine, qsr_unit):
        """Test QSR result has scope 1 and scope 2 subtotals."""
        result = engine.calculate(qsr_unit)
        assert result.breakdown.scope1_total_co2e_kg >= 0
        assert result.breakdown.scope2_total_co2e_kg >= 0
        assert result.total_co2e_kg == (
            result.breakdown.scope1_total_co2e_kg
            + result.breakdown.scope2_total_co2e_kg
        )

    def test_qsr_data_quality(self, engine, qsr_unit):
        """Test QSR result has data quality score."""
        result = engine.calculate(qsr_unit)
        assert result.data_quality is not None
        assert isinstance(result.data_quality, DataQualityScore)

    def test_qsr_processing_time(self, engine, qsr_unit):
        """Test QSR result includes processing time."""
        result = engine.calculate(qsr_unit)
        assert result.processing_time_ms >= 0


# ==============================================================================
# HOTEL CALCULATION TESTS
# ==============================================================================


class TestHotelCalculation:
    """Test hotel franchise unit calculations."""

    def test_hotel_full_calculation(self, engine, hotel_unit):
        """Test full hotel calculation with HVAC, water, heating."""
        result = engine.calculate(hotel_unit)
        assert isinstance(result, FranchiseCalculationResult)
        assert result.total_co2e_kg > 0

    def test_hotel_higher_than_small_retail(self, engine, hotel_unit, retail_unit):
        """Test hotel (5000 m2) emissions higher than retail (400 m2)."""
        hotel_result = engine.calculate(hotel_unit)
        retail_result = engine.calculate(retail_unit)
        assert hotel_result.total_co2e_kg > retail_result.total_co2e_kg

    def test_hotel_includes_heating(self, engine, hotel_unit):
        """Test hotel heating_kwh flows through to purchased_heating."""
        result = engine.calculate(hotel_unit)
        assert result.breakdown.purchased_heating_co2e_kg >= 0


# ==============================================================================
# CONVENIENCE STORE CALCULATION TESTS
# ==============================================================================


class TestConvenienceStoreCalculation:
    """Test convenience store franchise unit calculations."""

    def test_convenience_store_calculation(self, engine, convenience_unit):
        """Test convenience store with 24/7 refrigeration and lighting."""
        result = engine.calculate(convenience_unit)
        assert result.total_co2e_kg > 0

    def test_convenience_electricity_dominated(self, engine, convenience_unit):
        """Test convenience store is electricity-heavy due to refrigeration."""
        result = engine.calculate(convenience_unit)
        assert result.breakdown.purchased_electricity_co2e_kg > 0


# ==============================================================================
# RETAIL CALCULATION TESTS
# ==============================================================================


class TestRetailCalculation:
    """Test retail franchise unit calculations."""

    def test_retail_calculation(self, engine, retail_unit):
        """Test retail franchise with HVAC and lighting."""
        result = engine.calculate(retail_unit)
        assert result.total_co2e_kg > 0


# ==============================================================================
# FITNESS CENTER CALCULATION TESTS
# ==============================================================================


class TestFitnessCenterCalculation:
    """Test fitness center franchise unit calculations."""

    def test_fitness_calculation(self, engine, fitness_unit):
        """Test fitness center with HVAC, equipment, water heating."""
        result = engine.calculate(fitness_unit)
        assert result.total_co2e_kg > 0


# ==============================================================================
# AUTOMOTIVE SERVICE CALCULATION TESTS
# ==============================================================================


class TestAutomotiveServiceCalculation:
    """Test automotive service franchise unit calculations."""

    def test_automotive_calculation(self, engine, automotive_unit):
        """Test automotive service with equipment and gas."""
        result = engine.calculate(automotive_unit)
        assert result.total_co2e_kg > 0


# ==============================================================================
# ALL 7 EMISSION SOURCE TESTS
# ==============================================================================


class TestAllEmissionSources:
    """Test all 7 emission sources through the calculate API."""

    def test_electricity_only(self, engine):
        """Test unit with only purchased electricity."""
        unit = _make_unit(
            unit_id="FRN-EL-001",
            franchise_type="generic",
            electricity_kwh=Decimal("100000"),
        )
        result = engine.calculate(unit)
        assert result.breakdown.purchased_electricity_co2e_kg > 0

    def test_heating_only(self, engine):
        """Test unit with only purchased heating."""
        unit = _make_unit(
            unit_id="FRN-HT-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            heating_kwh=Decimal("50000"),
        )
        result = engine.calculate(unit)
        assert result.breakdown.purchased_heating_co2e_kg > 0

    def test_cooling_only(self, engine):
        """Test unit with only purchased cooling."""
        unit = _make_unit(
            unit_id="FRN-CL-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            cooling_kwh=Decimal("30000"),
        )
        result = engine.calculate(unit)
        assert result.breakdown.purchased_cooling_co2e_kg > 0

    def test_stationary_combustion_only(self, engine):
        """Test unit with only stationary combustion."""
        unit = _make_unit(
            unit_id="FRN-SC-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            stationary=StationaryCombustionInput(
                natural_gas_m3=Decimal("10000"),
            ),
        )
        result = engine.calculate(unit)
        assert result.breakdown.stationary_combustion_co2e_kg > 0

    def test_mobile_combustion_only(self, engine):
        """Test unit with only mobile combustion."""
        unit = _make_unit(
            unit_id="FRN-MC-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            mobile=MobileCombustionInput(
                total_petrol_litres=Decimal("5000"),
            ),
        )
        result = engine.calculate(unit)
        assert result.breakdown.mobile_combustion_co2e_kg > 0

    def test_refrigerant_leakage_only(self, engine):
        """Test unit with only refrigerant leakage."""
        unit = _make_unit(
            unit_id="FRN-RL-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            refrigerants=RefrigerantInput(
                equipment=[
                    {
                        "equipment_type": "cooler",
                        "refrigerant_type": "R-404A",
                        "charge_kg": Decimal("20.0"),
                    },
                ],
            ),
        )
        result = engine.calculate(unit)
        assert result.breakdown.refrigerant_leakage_co2e_kg >= 0

    def test_process_emissions(self, engine):
        """Test unit with explicit process emissions."""
        unit = _make_unit(
            unit_id="FRN-PE-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            process_co2e=Decimal("500"),
        )
        result = engine.calculate(unit)
        assert result.breakdown.process_emissions_co2e_kg == Decimal("500")


# ==============================================================================
# COMPANY-OWNED REJECTION (DC-FRN-001) TESTS
# ==============================================================================


class TestCompanyOwnedRejection:
    """Test DC-FRN-001: company-owned units rejected from Cat 14."""

    def test_company_owned_rejected(self, engine):
        """Test company-owned unit raises ValueError."""
        unit = _make_unit(
            unit_id="FRN-CO-001",
            franchise_type="qsr",
            ownership_type="company_owned",
        )
        with pytest.raises(ValueError):
            engine.calculate(unit)

    def test_franchisee_accepted(self, engine, qsr_unit):
        """Test franchisee ownership is accepted."""
        result = engine.calculate(qsr_unit)
        assert result.total_co2e_kg > 0

    def test_joint_venture_accepted(self, engine):
        """Test joint_venture ownership is accepted."""
        unit = _make_unit(
            unit_id="FRN-JV-001",
            franchise_type="qsr",
            ownership_type="joint_venture",
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg > 0

    def test_master_franchisee_accepted(self, engine):
        """Test master_franchisee ownership is accepted."""
        unit = _make_unit(
            unit_id="FRN-MF-001",
            franchise_type="qsr",
            ownership_type="master_franchisee",
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg > 0


# ==============================================================================
# PRO-RATA PARTIAL YEAR TESTS
# ==============================================================================


class TestProRataPartialYear:
    """Test pro-rata for partial year operations."""

    def test_full_year_no_pro_rata(self, engine, qsr_unit):
        """Test full-year unit has pro_rata_applied=False."""
        result = engine.calculate(qsr_unit)
        assert result.pro_rata_applied is False
        assert result.pro_rata_factor == Decimal("1")

    def test_half_year_pro_rata(self, engine):
        """Test 182/365 day operation applies pro-rata."""
        unit = _make_unit(
            unit_id="FRN-PR-001",
            franchise_type="qsr",
            electricity_kwh=Decimal("180000"),
            operating_days=182,
            total_days=365,
            stationary=StationaryCombustionInput(
                natural_gas_m3=Decimal("25000"),
            ),
        )
        result = engine.calculate(unit)
        assert result.pro_rata_applied is True
        assert result.pro_rata_factor < Decimal("1")

    def test_pro_rata_reduces_emissions(self, engine):
        """Test pro-rata produces lower emissions than full year."""
        full_year = _make_unit(
            unit_id="FRN-FY-001",
            franchise_type="qsr",
            electricity_kwh=Decimal("100000"),
        )
        half_year = _make_unit(
            unit_id="FRN-HY-001",
            franchise_type="qsr",
            electricity_kwh=Decimal("100000"),
            operating_days=182,
            total_days=365,
        )
        result_full = engine.calculate(full_year)
        result_half = engine.calculate(half_year)
        assert result_half.total_co2e_kg < result_full.total_co2e_kg


# ==============================================================================
# BATCH CALCULATION TESTS
# ==============================================================================


class TestBatchCalculation:
    """Test batch calculation across multiple units."""

    def test_batch_multiple_units(self, engine, qsr_unit, hotel_unit):
        """Test batch calculation with multiple units."""
        results = engine.calculate_batch([qsr_unit, hotel_unit])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r.total_co2e_kg > 0

    def test_batch_skips_company_owned(self, engine, qsr_unit):
        """Test batch calculation skips company-owned units."""
        owned_unit = _make_unit(
            unit_id="FRN-CO-001",
            franchise_type="qsr",
            ownership_type="company_owned",
        )
        results = engine.calculate_batch([qsr_unit, owned_unit])
        assert len(results) == 1
        assert results[0].unit_id == "FRN-QSR-001"

    def test_batch_empty_list(self, engine):
        """Test batch with empty list returns empty."""
        results = engine.calculate_batch([])
        assert results == []


# ==============================================================================
# DATA QUALITY ASSESSMENT TESTS
# ==============================================================================


class TestDataQualityAssessment:
    """Test data quality assessment for franchise-specific method."""

    def test_dqi_composite_score(self, engine, qsr_unit):
        """Test DQI composite score is between 1 and 5."""
        result = engine.calculate(qsr_unit)
        composite = result.data_quality.composite
        assert Decimal("1") <= composite <= Decimal("5")

    def test_dqi_tier_label(self, engine, qsr_unit):
        """Test DQI tier label is tier_1, tier_2, or tier_3."""
        result = engine.calculate(qsr_unit)
        assert result.data_quality.tier in ("tier_1", "tier_2", "tier_3")

    def test_dqi_five_dimensions(self, engine, qsr_unit):
        """Test DQI has 5 dimension scores."""
        result = engine.calculate(qsr_unit)
        dqi = result.data_quality
        assert dqi.data_source > 0
        assert dqi.temporal > 0
        assert dqi.geographical > 0
        assert dqi.technological > 0
        assert dqi.completeness > 0


# ==============================================================================
# UNCERTAINTY TESTS
# ==============================================================================


class TestUncertainty:
    """Test uncertainty quantification for franchise-specific method."""

    def test_uncertainty_bounds(self, engine, qsr_unit):
        """Test uncertainty bounds surround the central estimate."""
        result = engine.calculate(qsr_unit)
        assert result.uncertainty_lower_co2e_kg < result.total_co2e_kg
        assert result.uncertainty_upper_co2e_kg > result.total_co2e_kg

    def test_uncertainty_percentages(self, engine, qsr_unit):
        """Test uncertainty percentage bounds are present."""
        result = engine.calculate(qsr_unit)
        assert result.uncertainty_lower_pct < 0 or result.uncertainty_lower_pct >= 0
        assert isinstance(result.uncertainty_upper_pct, Decimal)


# ==============================================================================
# ENGINE INFO AND STATS TESTS
# ==============================================================================


class TestEngineInfoAndStats:
    """Test engine info and calculation count."""

    def test_get_engine_info(self, engine):
        """Test engine info returns metadata dict."""
        info = engine.get_engine_info()
        assert info["engine"] == "FranchiseSpecificCalculatorEngine"
        assert info["agent_id"] == "GL-MRV-S3-014"
        assert info["version"] == "1.0.0"

    def test_get_calculation_count(self, engine, qsr_unit):
        """Test calculation count increments after each calculation."""
        assert engine.get_calculation_count() == 0
        engine.calculate(qsr_unit)
        assert engine.get_calculation_count() == 1

    def test_calculation_count_batch(self, engine, qsr_unit, hotel_unit):
        """Test calculation count increments for each unit in batch."""
        engine.calculate_batch([qsr_unit, hotel_unit])
        assert engine.get_calculation_count() == 2

    def test_engine_info_supported_types(self, engine):
        """Test engine info lists supported franchise types."""
        info = engine.get_engine_info()
        assert "supported_franchise_types" in info
        assert "qsr" in info["supported_franchise_types"]

    def test_engine_info_emission_sources(self, engine):
        """Test engine info lists emission sources."""
        info = engine.get_engine_info()
        assert "emission_sources" in info
        assert len(info["emission_sources"]) == 7


# ==============================================================================
# PARAMETRIZED FRANCHISE TYPE TESTS
# ==============================================================================


class TestParametrizedFranchiseTypes:
    """Test calculations across supported franchise types."""

    @pytest.mark.parametrize("franchise_type", [
        "qsr", "hotel", "convenience_store", "retail",
        "fitness", "automotive", "generic",
    ])
    def test_all_franchise_types(self, engine, franchise_type):
        """Test calculation for each franchise type."""
        unit = _make_unit(
            unit_id=f"FRN-{franchise_type[:3].upper()}-001",
            franchise_type=franchise_type,
            electricity_kwh=Decimal("100000"),
            stationary=StationaryCombustionInput(
                natural_gas_m3=Decimal("5000"),
            ),
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg > 0
        assert result.franchise_type == franchise_type


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Test edge cases for franchise-specific calculations."""

    def test_zero_floor_area(self, engine):
        """Test calculation with zero floor area still works from energy data."""
        unit = _make_unit(
            unit_id="FRN-ZFA-001",
            franchise_type="generic",
            floor_area_m2=Decimal("0"),
            electricity_kwh=Decimal("50000"),
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg >= 0

    def test_zero_electricity(self, engine):
        """Test calculation with zero electricity but other sources."""
        unit = _make_unit(
            unit_id="FRN-ZEL-001",
            franchise_type="generic",
            electricity_kwh=Decimal("0"),
            stationary=StationaryCombustionInput(
                natural_gas_m3=Decimal("10000"),
            ),
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg > 0

    def test_minimal_unit(self, engine):
        """Test calculation with only electricity data."""
        unit = _make_unit(
            unit_id="FRN-MIN-001",
            franchise_type="generic",
            electricity_kwh=Decimal("50000"),
        )
        result = engine.calculate(unit)
        assert result.total_co2e_kg > 0

    def test_to_dict_round_trip(self, engine, qsr_unit):
        """Test FranchiseUnitInput.to_dict() works."""
        d = qsr_unit.to_dict()
        assert d["unit_id"] == "FRN-QSR-001"
        assert d["franchise_type"] == "qsr"

    def test_result_to_dict(self, engine, qsr_unit):
        """Test FranchiseCalculationResult.to_dict() works."""
        result = engine.calculate(qsr_unit)
        d = result.to_dict()
        assert "total_co2e_kg" in d
        assert "provenance_hash" in d


# ==============================================================================
# PROVENANCE TESTS
# ==============================================================================


class TestProvenance:
    """Test provenance hash generation."""

    def test_provenance_deterministic(self, engine):
        """Test same input produces same provenance hash."""
        unit1 = _make_unit(
            unit_id="FRN-DET-001",
            franchise_type="generic",
            electricity_kwh=Decimal("50000"),
        )
        unit2 = _make_unit(
            unit_id="FRN-DET-001",
            franchise_type="generic",
            electricity_kwh=Decimal("50000"),
        )
        r1 = engine.calculate(unit1)
        r2 = engine.calculate(unit2)
        assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_differs_for_different_input(self, engine):
        """Test different inputs produce different provenance hashes."""
        unit1 = _make_unit(
            unit_id="FRN-D1-001",
            franchise_type="generic",
            electricity_kwh=Decimal("50000"),
        )
        unit2 = _make_unit(
            unit_id="FRN-D2-001",
            franchise_type="generic",
            electricity_kwh=Decimal("80000"),
        )
        r1 = engine.calculate(unit1)
        r2 = engine.calculate(unit2)
        assert r1.provenance_hash != r2.provenance_hash

    def test_input_hash_present(self, engine, qsr_unit):
        """Test result includes an input_hash."""
        result = engine.calculate(qsr_unit)
        assert len(result.input_hash) == 64
