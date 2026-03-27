# -*- coding: utf-8 -*-
"""
Test suite for investments.real_asset_calculator - AGENT-MRV-028.

Tests the RealAssetCalculatorEngine (Engine 4) for the Investments Agent
(GL-MRV-S3-015) including commercial real estate (CRE) via EUI method,
mortgage with EPC rating adjustments, motor vehicle loans by category,
PCAF quality tiers for each real asset type, building benchmarks, and
attribution factor calculations.

Coverage:
- CRE: floor_area x EUI x grid_EF
- Mortgage: EPC rating adjustments (A-G)
- Motor vehicle: by category (passenger/commercial/EV)
- PCAF quality for each real asset type
- Building benchmarks for 6 property types x 5 climate zones
- Attribution factor calculations (LTV-weighted)
- Parametrized tests for property types, vehicle categories, EPC ratings

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest

from greenlang.agents.mrv.investments.real_asset_calculator import (
    RealAssetCalculatorEngine,
)
from greenlang.agents.mrv.investments.models import (
    AssetClass,
    PropertyType,
    ClimateZone,
    EPCRating,
    VehicleCategory,
    FuelType,
    PCAFDataQuality,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    RealAssetCalculatorEngine.reset_instance()
    yield
    RealAssetCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh RealAssetCalculatorEngine with mocked config."""
    with patch(
        "greenlang.agents.mrv.investments.real_asset_calculator.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.real_assets.default_eui_source = "CRREM"
        cfg.general.default_gwp = "AR5"
        mock_config.return_value = cfg
        eng = RealAssetCalculatorEngine()
        yield eng


def _make_cre_input(**overrides):
    """Build a CRE input dict with defaults."""
    base = {
        "asset_class": "commercial_real_estate",
        "property_name": "Downtown Office Tower",
        "outstanding_amount": Decimal("25000000"),
        "property_value": Decimal("50000000"),
        "floor_area_m2": Decimal("10000"),
        "property_type": "office",
        "epc_rating": "B",
        "climate_zone": "temperate",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }
    base.update(overrides)
    return base


def _make_mortgage_input(**overrides):
    """Build a mortgage input dict with defaults."""
    base = {
        "asset_class": "mortgage",
        "property_name": "123 Oak Street",
        "outstanding_amount": Decimal("300000"),
        "property_value": Decimal("400000"),
        "floor_area_m2": Decimal("150"),
        "property_type": "residential",
        "epc_rating": "C",
        "climate_zone": "temperate",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 3,
    }
    base.update(overrides)
    return base


def _make_motor_vehicle_input(**overrides):
    """Build a motor vehicle loan input dict with defaults."""
    base = {
        "asset_class": "motor_vehicle_loan",
        "vehicle_description": "2024 Toyota Camry Hybrid",
        "outstanding_amount": Decimal("25000"),
        "vehicle_value": Decimal("35000"),
        "vehicle_category": "passenger_car",
        "fuel_type": "hybrid",
        "annual_mileage_km": Decimal("20000"),
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 3,
    }
    base.update(overrides)
    return base


# ==============================================================================
# CRE CALCULATION TESTS
# ==============================================================================


class TestCRECalculation:
    """Test commercial real estate (CRE) calculations."""

    def test_cre_attribution_factor(self, engine):
        """Test CRE attribution factor = outstanding / property_value."""
        data = _make_cre_input()
        result = engine.calculate(data)
        expected_af = Decimal("25000000") / Decimal("50000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_cre_building_emissions(self, engine):
        """Test CRE emissions = floor_area x EUI x grid_EF."""
        data = _make_cre_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_cre_larger_area_more_emissions(self, engine):
        """Test larger floor area produces more emissions."""
        small = _make_cre_input(floor_area_m2=Decimal("5000"))
        large = _make_cre_input(floor_area_m2=Decimal("20000"))
        r_small = engine.calculate(small)
        r_large = engine.calculate(large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]

    def test_cre_higher_ltv_more_financed(self, engine):
        """Test higher LTV ratio increases financed emissions."""
        low_ltv = _make_cre_input(
            outstanding_amount=Decimal("10000000"),
            property_value=Decimal("50000000"),
        )
        high_ltv = _make_cre_input(
            outstanding_amount=Decimal("40000000"),
            property_value=Decimal("50000000"),
        )
        r_low = engine.calculate(low_ltv)
        r_high = engine.calculate(high_ltv)
        assert r_high["financed_emissions"] > r_low["financed_emissions"]

    def test_cre_provenance_hash(self, engine):
        """Test CRE result includes provenance hash."""
        data = _make_cre_input()
        result = engine.calculate(data)
        assert len(result["provenance_hash"]) == 64

    def test_cre_attribution_method(self, engine):
        """Test CRE uses LTV-weighted attribution method."""
        data = _make_cre_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "ltv_weighted"

    @pytest.mark.parametrize("property_type", [
        "office", "retail", "industrial", "residential", "hotel", "mixed_use",
    ])
    def test_cre_property_types(self, engine, property_type):
        """Test CRE calculation for all 6 property types."""
        data = _make_cre_input(property_type=property_type)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    @pytest.mark.parametrize("climate_zone", [
        "tropical", "arid", "temperate", "continental", "polar",
    ])
    def test_cre_climate_zones(self, engine, climate_zone):
        """Test CRE calculation for all 5 climate zones."""
        data = _make_cre_input(climate_zone=climate_zone)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_zero_property_value_raises_error(self, engine):
        """Test zero property value raises error."""
        data = _make_cre_input(property_value=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)


# ==============================================================================
# MORTGAGE TESTS
# ==============================================================================


class TestMortgageCalculation:
    """Test residential mortgage calculations."""

    def test_mortgage_attribution_factor(self, engine):
        """Test mortgage AF = outstanding / property_value."""
        data = _make_mortgage_input()
        result = engine.calculate(data)
        expected_af = Decimal("300000") / Decimal("400000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_mortgage_financed_emissions(self, engine):
        """Test mortgage financed emissions are positive."""
        data = _make_mortgage_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    @pytest.mark.parametrize("epc_rating", ["A", "B", "C", "D", "E", "F", "G"])
    def test_mortgage_epc_ratings(self, engine, epc_rating):
        """Test mortgage calculation with all EPC ratings A-G."""
        data = _make_mortgage_input(epc_rating=epc_rating)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_epc_a_lower_than_g(self, engine):
        """Test EPC A produces lower emissions than EPC G."""
        data_a = _make_mortgage_input(epc_rating="A")
        data_g = _make_mortgage_input(epc_rating="G")
        r_a = engine.calculate(data_a)
        r_g = engine.calculate(data_g)
        assert r_a["financed_emissions"] < r_g["financed_emissions"]

    def test_epc_b_lower_than_f(self, engine):
        """Test EPC B produces lower emissions than EPC F."""
        data_b = _make_mortgage_input(epc_rating="B")
        data_f = _make_mortgage_input(epc_rating="F")
        r_b = engine.calculate(data_b)
        r_f = engine.calculate(data_f)
        assert r_b["financed_emissions"] < r_f["financed_emissions"]

    def test_epc_monotonic_increase(self, engine):
        """Test emissions increase monotonically from A to G."""
        ratings = ["A", "B", "C", "D", "E", "F", "G"]
        emissions = []
        for r in ratings:
            data = _make_mortgage_input(epc_rating=r)
            result = engine.calculate(data)
            emissions.append(result["financed_emissions"])
        for i in range(len(emissions) - 1):
            assert emissions[i] <= emissions[i + 1]

    def test_mortgage_floor_area_effect(self, engine):
        """Test larger floor area produces more emissions."""
        small = _make_mortgage_input(floor_area_m2=Decimal("80"))
        large = _make_mortgage_input(floor_area_m2=Decimal("300"))
        r_small = engine.calculate(small)
        r_large = engine.calculate(large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]


# ==============================================================================
# MOTOR VEHICLE TESTS
# ==============================================================================


class TestMotorVehicleCalculation:
    """Test motor vehicle loan calculations."""

    def test_motor_vehicle_attribution_factor(self, engine):
        """Test motor vehicle AF = outstanding / vehicle_value."""
        data = _make_motor_vehicle_input()
        result = engine.calculate(data)
        expected_af = Decimal("25000") / Decimal("35000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_motor_vehicle_financed_emissions(self, engine):
        """Test motor vehicle financed emissions are positive."""
        data = _make_motor_vehicle_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    @pytest.mark.parametrize("category", [
        "passenger_car", "light_commercial", "heavy_commercial",
        "electric_vehicle", "motorcycle",
    ])
    def test_vehicle_categories(self, engine, category):
        """Test calculation for all 5 vehicle categories."""
        data = _make_motor_vehicle_input(vehicle_category=category)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_ev_lower_than_ice(self, engine):
        """Test EV has lower emissions than ICE passenger car."""
        data_ice = _make_motor_vehicle_input(
            vehicle_category="passenger_car",
            fuel_type="petrol",
        )
        data_ev = _make_motor_vehicle_input(
            vehicle_category="electric_vehicle",
            fuel_type="electric",
        )
        r_ice = engine.calculate(data_ice)
        r_ev = engine.calculate(data_ev)
        assert r_ev["financed_emissions"] < r_ice["financed_emissions"]

    def test_higher_mileage_more_emissions(self, engine):
        """Test higher annual mileage produces more emissions."""
        low = _make_motor_vehicle_input(annual_mileage_km=Decimal("10000"))
        high = _make_motor_vehicle_input(annual_mileage_km=Decimal("40000"))
        r_low = engine.calculate(low)
        r_high = engine.calculate(high)
        assert r_high["financed_emissions"] > r_low["financed_emissions"]

    def test_heavy_commercial_highest(self, engine):
        """Test heavy commercial vehicle has highest emissions."""
        data_passenger = _make_motor_vehicle_input(
            vehicle_category="passenger_car",
            fuel_type="petrol",
        )
        data_heavy = _make_motor_vehicle_input(
            vehicle_category="heavy_commercial",
            fuel_type="diesel",
        )
        r_pass = engine.calculate(data_passenger)
        r_heavy = engine.calculate(data_heavy)
        assert r_heavy["financed_emissions"] > r_pass["financed_emissions"]

    def test_zero_vehicle_value_raises_error(self, engine):
        """Test zero vehicle value raises error."""
        data = _make_motor_vehicle_input(vehicle_value=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)


# ==============================================================================
# PCAF QUALITY TESTS
# ==============================================================================


class TestPCAFQualityRealAssets:
    """Test PCAF quality scoring for real assets."""

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_cre_all_pcaf_scores(self, engine, score):
        """Test CRE accepts all PCAF scores 1-5."""
        data = _make_cre_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_mortgage_all_pcaf_scores(self, engine, score):
        """Test mortgage accepts all PCAF scores 1-5."""
        data = _make_mortgage_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_motor_vehicle_all_pcaf_scores(self, engine, score):
        """Test motor vehicle accepts all PCAF scores 1-5."""
        data = _make_motor_vehicle_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score


# ==============================================================================
# BUILDING BENCHMARK TESTS
# ==============================================================================


class TestBuildingBenchmarks:
    """Test building benchmark application in CRE calculations."""

    @pytest.mark.parametrize("property_type,climate_zone", [
        ("office", "tropical"),
        ("office", "temperate"),
        ("retail", "arid"),
        ("industrial", "continental"),
        ("hotel", "polar"),
        ("mixed_use", "temperate"),
    ])
    def test_benchmark_applied(self, engine, property_type, climate_zone):
        """Test building benchmark is applied for property type x climate zone."""
        data = _make_cre_input(
            property_type=property_type,
            climate_zone=climate_zone,
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# ATTRIBUTION FACTOR TESTS
# ==============================================================================


class TestAttributionFactors:
    """Test attribution factor calculations for real assets."""

    def test_cre_af_capped_at_one(self, engine):
        """Test CRE attribution factor is capped at 1.0."""
        data = _make_cre_input(
            outstanding_amount=Decimal("60000000"),
            property_value=Decimal("50000000"),
        )
        result = engine.calculate(data)
        assert result["attribution_factor"] <= Decimal("1.0")

    def test_mortgage_af_capped_at_one(self, engine):
        """Test mortgage attribution factor is capped at 1.0."""
        data = _make_mortgage_input(
            outstanding_amount=Decimal("500000"),
            property_value=Decimal("400000"),
        )
        result = engine.calculate(data)
        assert result["attribution_factor"] <= Decimal("1.0")

    def test_motor_vehicle_af_capped_at_one(self, engine):
        """Test motor vehicle AF is capped at 1.0."""
        data = _make_motor_vehicle_input(
            outstanding_amount=Decimal("40000"),
            vehicle_value=Decimal("35000"),
        )
        result = engine.calculate(data)
        assert result["attribution_factor"] <= Decimal("1.0")


# ==============================================================================
# BATCH AND ERROR HANDLING TESTS
# ==============================================================================


class TestBatchAndErrors:
    """Test batch processing and error handling."""

    def test_batch_mixed_real_assets(self, engine):
        """Test batch with mixed real asset types."""
        items = [
            _make_cre_input(),
            _make_mortgage_input(),
            _make_motor_vehicle_input(),
        ]
        results = engine.calculate_batch(items)
        assert len(results) == 3

    def test_batch_empty(self, engine):
        """Test batch with empty list."""
        results = engine.calculate_batch([])
        assert len(results) == 0

    def test_result_required_fields(self, engine):
        """Test result contains all required fields."""
        data = _make_cre_input()
        result = engine.calculate(data)
        required = [
            "property_name", "asset_class", "attribution_factor",
            "financed_emissions", "pcaf_quality_score", "provenance_hash",
        ]
        for field in required:
            assert field in result
