# -*- coding: utf-8 -*-
"""
Unit Tests for GL-006: Scope 3 Emissions Agent

Comprehensive test suite with 75 test cases covering:
- All 15 Scope 3 categories (15 tests)
- Spend-based calculations (15 tests)
- Activity-based calculations (15 tests)
- Transport/travel calculations (15 tests)
- Data quality and error handling (15 tests)

Target: 85%+ coverage for Scope 3 Emissions Agent
Run with: pytest tests/unit/test_gl006_scope3_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

The Scope 3 Emissions Agent calculates value chain GHG emissions across all
15 GHG Protocol categories using spend-based, activity-based, and supplier-specific methods.
"""

import pytest
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GL-Agent-Factory" / "backend" / "agents"))

# Import agent components
from gl_006_scope3_emissions.agent import (
    Scope3EmissionsAgent,
    Scope3Input,
    Scope3Output,
    Scope3Category,
    CalculationMethod,
    DataQualityScore,
    SpendData,
    ActivityData,
    TransportData,
    TravelData,
    SpendEmissionFactor,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create Scope3EmissionsAgent instance."""
    return Scope3EmissionsAgent()


@pytest.fixture
def valid_purchased_goods_input():
    """Create valid Category 1 (Purchased Goods) input."""
    return Scope3Input(
        category=Scope3Category.CAT_1_PURCHASED_GOODS,
        reporting_year=2024,
        spend_data=[
            SpendData(category="steel", spend_usd=1000000.0, supplier_name="SteelCorp"),
            SpendData(category="aluminum", spend_usd=500000.0, supplier_name="AlumCo"),
            SpendData(category="plastics", spend_usd=250000.0),
        ],
        revenue_usd=50000000.0,
        employees=500,
    )


@pytest.fixture
def valid_capital_goods_input():
    """Create valid Category 2 (Capital Goods) input."""
    return Scope3Input(
        category=Scope3Category.CAT_2_CAPITAL_GOODS,
        reporting_year=2024,
        spend_data=[
            SpendData(category="machinery", spend_usd=5000000.0),
            SpendData(category="electronics", spend_usd=2000000.0),
        ],
    )


@pytest.fixture
def valid_upstream_transport_input():
    """Create valid Category 4 (Upstream Transport) input."""
    return Scope3Input(
        category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
        reporting_year=2024,
        transport_data=[
            TransportData(mode="road_truck", distance_km=1000.0, weight_tonnes=50.0),
            TransportData(mode="rail_freight", distance_km=2000.0, weight_tonnes=100.0),
            TransportData(mode="sea_container", distance_km=10000.0, weight_tonnes=200.0),
        ],
    )


@pytest.fixture
def valid_business_travel_input():
    """Create valid Category 6 (Business Travel) input."""
    return Scope3Input(
        category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
        reporting_year=2024,
        travel_data=[
            TravelData(mode="air", distance_km=5000.0, trip_type="round_trip"),
            TravelData(mode="rail_average", distance_km=500.0, trip_type="one_way"),
            TravelData(mode="car_average", distance_km=200.0, trip_type="one_way"),
        ],
    )


@pytest.fixture
def valid_waste_input():
    """Create valid Category 5 (Waste) input."""
    return Scope3Input(
        category=Scope3Category.CAT_5_WASTE,
        reporting_year=2024,
        activity_data=[
            ActivityData(activity_type="landfill_mixed", quantity=100.0, unit="tonnes"),
            ActivityData(activity_type="recycling_paper", quantity=50.0, unit="tonnes"),
            ActivityData(activity_type="incineration", quantity=25.0, unit="tonnes"),
        ],
    )


@pytest.fixture
def supplier_specific_input():
    """Create input with supplier-specific data."""
    return Scope3Input(
        category=Scope3Category.CAT_1_PURCHASED_GOODS,
        reporting_year=2024,
        calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        supplier_emissions={"SupplierA": 50000.0, "SupplierB": 30000.0},
    )


# =============================================================================
# All 15 Scope 3 Categories Tests (15 tests)
# =============================================================================

class TestScope3Categories:
    """Test suite for all 15 Scope 3 categories - 15 test cases."""

    @pytest.mark.unit
    def test_category_1_purchased_goods(self):
        """UT-GL006-001: Test Category 1 enum value."""
        assert Scope3Category.CAT_1_PURCHASED_GOODS.value == "purchased_goods"

    @pytest.mark.unit
    def test_category_2_capital_goods(self):
        """UT-GL006-002: Test Category 2 enum value."""
        assert Scope3Category.CAT_2_CAPITAL_GOODS.value == "capital_goods"

    @pytest.mark.unit
    def test_category_3_fuel_energy(self):
        """UT-GL006-003: Test Category 3 enum value."""
        assert Scope3Category.CAT_3_FUEL_ENERGY.value == "fuel_energy_activities"

    @pytest.mark.unit
    def test_category_4_upstream_transport(self):
        """UT-GL006-004: Test Category 4 enum value."""
        assert Scope3Category.CAT_4_UPSTREAM_TRANSPORT.value == "upstream_transport"

    @pytest.mark.unit
    def test_category_5_waste(self):
        """UT-GL006-005: Test Category 5 enum value."""
        assert Scope3Category.CAT_5_WASTE.value == "waste_generated"

    @pytest.mark.unit
    def test_category_6_business_travel(self):
        """UT-GL006-006: Test Category 6 enum value."""
        assert Scope3Category.CAT_6_BUSINESS_TRAVEL.value == "business_travel"

    @pytest.mark.unit
    def test_category_7_commuting(self):
        """UT-GL006-007: Test Category 7 enum value."""
        assert Scope3Category.CAT_7_COMMUTING.value == "employee_commuting"

    @pytest.mark.unit
    def test_category_8_upstream_leased(self):
        """UT-GL006-008: Test Category 8 enum value."""
        assert Scope3Category.CAT_8_UPSTREAM_LEASED.value == "upstream_leased_assets"

    @pytest.mark.unit
    def test_category_9_downstream_transport(self):
        """UT-GL006-009: Test Category 9 enum value."""
        assert Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT.value == "downstream_transport"

    @pytest.mark.unit
    def test_category_10_processing(self):
        """UT-GL006-010: Test Category 10 enum value."""
        assert Scope3Category.CAT_10_PROCESSING.value == "processing_of_products"

    @pytest.mark.unit
    def test_category_11_use_of_products(self):
        """UT-GL006-011: Test Category 11 enum value."""
        assert Scope3Category.CAT_11_USE_OF_PRODUCTS.value == "use_of_sold_products"

    @pytest.mark.unit
    def test_category_12_end_of_life(self):
        """UT-GL006-012: Test Category 12 enum value."""
        assert Scope3Category.CAT_12_END_OF_LIFE.value == "end_of_life_treatment"

    @pytest.mark.unit
    def test_category_13_downstream_leased(self):
        """UT-GL006-013: Test Category 13 enum value."""
        assert Scope3Category.CAT_13_DOWNSTREAM_LEASED.value == "downstream_leased_assets"

    @pytest.mark.unit
    def test_category_14_franchises(self):
        """UT-GL006-014: Test Category 14 enum value."""
        assert Scope3Category.CAT_14_FRANCHISES.value == "franchises"

    @pytest.mark.unit
    def test_category_15_investments(self):
        """UT-GL006-015: Test Category 15 enum value."""
        assert Scope3Category.CAT_15_INVESTMENTS.value == "investments"


# =============================================================================
# Spend-Based Calculations Tests (15 tests)
# =============================================================================

class TestSpendBasedCalculations:
    """Test suite for spend-based calculations - 15 test cases."""

    @pytest.mark.unit
    def test_steel_spend_calculation(self, agent, valid_purchased_goods_input):
        """UT-GL006-016: Test steel spend-based calculation."""
        result = agent.run(valid_purchased_goods_input)

        # Steel: 1,000,000 USD * 0.85 kgCO2e/USD = 850,000 kgCO2e
        assert result.total_emissions_kgco2e > 0
        assert "SteelCorp" in result.emissions_by_source

    @pytest.mark.unit
    def test_aluminum_spend_calculation(self, agent, valid_purchased_goods_input):
        """UT-GL006-017: Test aluminum spend-based calculation."""
        result = agent.run(valid_purchased_goods_input)

        # Aluminum has higher factor (1.20)
        assert "AlumCo" in result.emissions_by_source

    @pytest.mark.unit
    def test_spend_formula_verification(self, agent):
        """UT-GL006-018: Test formula: emissions = spend * EEIO_factor."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1000000.0)],
        )
        result = agent.run(input_data)

        # Steel factor = 0.85 kgCO2e/USD
        expected = 1000000.0 * 0.85
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_multiple_spend_categories_summed(self, agent, valid_purchased_goods_input):
        """UT-GL006-019: Test multiple spend categories are summed."""
        result = agent.run(valid_purchased_goods_input)

        # Should sum all sources
        total_from_sources = sum(result.emissions_by_source.values())
        assert result.total_emissions_kgco2e == pytest.approx(total_from_sources, rel=0.01)

    @pytest.mark.unit
    def test_spend_factor_steel(self, agent):
        """UT-GL006-020: Test steel EEIO factor value."""
        assert agent.SPEND_FACTORS["steel"].factor_kgco2e_per_usd == 0.85

    @pytest.mark.unit
    def test_spend_factor_aluminum(self, agent):
        """UT-GL006-021: Test aluminum EEIO factor value."""
        assert agent.SPEND_FACTORS["aluminum"].factor_kgco2e_per_usd == 1.20

    @pytest.mark.unit
    def test_spend_factor_services(self, agent):
        """UT-GL006-022: Test services EEIO factor (lowest)."""
        assert agent.SPEND_FACTORS["services"].factor_kgco2e_per_usd == 0.15

    @pytest.mark.unit
    def test_default_spend_factor(self, agent):
        """UT-GL006-023: Test default EEIO factor for unknown categories."""
        assert agent.SPEND_FACTORS["default"].factor_kgco2e_per_usd == 0.40

    @pytest.mark.unit
    def test_unknown_spend_category_uses_default(self, agent):
        """UT-GL006-024: Test unknown spend category uses default factor."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="unknown_category", spend_usd=100000.0)],
        )
        result = agent.run(input_data)

        expected = 100000.0 * 0.40  # Default factor
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_capital_goods_spend_calculation(self, agent, valid_capital_goods_input):
        """UT-GL006-025: Test Category 2 capital goods calculation."""
        result = agent.run(valid_capital_goods_input)
        assert result.category_number == 2
        assert result.total_emissions_kgco2e > 0

    @pytest.mark.unit
    def test_zero_spend_returns_zero(self, agent):
        """UT-GL006-026: Test zero spend returns zero emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=0.0)],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0

    @pytest.mark.unit
    def test_emission_factors_used_in_output(self, agent, valid_purchased_goods_input):
        """UT-GL006-027: Test emission factors used are recorded."""
        result = agent.run(valid_purchased_goods_input)
        assert len(result.emission_factors_used) > 0

    @pytest.mark.unit
    def test_factor_source_recorded(self, agent, valid_purchased_goods_input):
        """UT-GL006-028: Test emission factor source is recorded."""
        result = agent.run(valid_purchased_goods_input)
        assert any("EPA EEIO" in f.get("source", "") for f in result.emission_factors_used)

    @pytest.mark.unit
    def test_supplier_name_used_as_source_key(self, agent, valid_purchased_goods_input):
        """UT-GL006-029: Test supplier name is used as source key when provided."""
        result = agent.run(valid_purchased_goods_input)
        assert "SteelCorp" in result.emissions_by_source

    @pytest.mark.unit
    def test_category_used_as_source_key_without_supplier(self, agent):
        """UT-GL006-030: Test category used as source key without supplier."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="plastics", spend_usd=100000.0)],
        )
        result = agent.run(input_data)
        assert "plastics" in result.emissions_by_source


# =============================================================================
# Activity-Based Calculations Tests (15 tests)
# =============================================================================

class TestActivityBasedCalculations:
    """Test suite for activity-based calculations - 15 test cases."""

    @pytest.mark.unit
    def test_waste_landfill_calculation(self, agent, valid_waste_input):
        """UT-GL006-031: Test landfill waste emissions calculation."""
        result = agent.run(valid_waste_input)
        assert result.total_emissions_kgco2e > 0
        assert "landfill_mixed" in result.emissions_by_source

    @pytest.mark.unit
    def test_waste_recycling_negative_emissions(self, agent):
        """UT-GL006-032: Test recycling has negative (avoided) emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[
                ActivityData(activity_type="recycling_paper", quantity=100000.0, unit="kg"),
            ],
        )
        result = agent.run(input_data)
        # Paper recycling factor is negative (-0.900)
        assert result.emissions_by_source.get("recycling_paper", 0) < 0

    @pytest.mark.unit
    def test_waste_factors_landfill(self, agent):
        """UT-GL006-033: Test landfill waste factor."""
        assert agent.WASTE_FACTORS["landfill_mixed"] == 0.586

    @pytest.mark.unit
    def test_waste_factors_incineration(self, agent):
        """UT-GL006-034: Test incineration waste factor (lowest)."""
        assert agent.WASTE_FACTORS["incineration"] == 0.021

    @pytest.mark.unit
    def test_waste_unit_conversion_tonnes(self, agent):
        """UT-GL006-035: Test waste unit conversion from tonnes to kg."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[
                ActivityData(activity_type="landfill_mixed", quantity=10.0, unit="tonnes"),
            ],
        )
        result = agent.run(input_data)

        # 10 tonnes = 10000 kg * 0.586 = 5860 kgCO2e
        expected = 10000 * 0.586
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_waste_formula_verification(self, agent):
        """UT-GL006-036: Test waste formula: emissions = quantity * factor."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[
                ActivityData(activity_type="landfill_mixed", quantity=1000.0, unit="kg"),
            ],
        )
        result = agent.run(input_data)

        expected = 1000 * 0.586
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_commuting_factors_defined(self, agent):
        """UT-GL006-037: Test employee commuting factors are defined."""
        assert "car_alone" in agent.COMMUTING_FACTORS
        assert "public_transit" in agent.COMMUTING_FACTORS
        assert "bicycle" in agent.COMMUTING_FACTORS
        assert agent.COMMUTING_FACTORS["bicycle"] == 0.0

    @pytest.mark.unit
    def test_remote_work_factor(self, agent):
        """UT-GL006-038: Test remote work has very low factor."""
        assert agent.COMMUTING_FACTORS["remote"] == 0.002

    @pytest.mark.unit
    def test_calculate_waste_method(self, agent, valid_waste_input):
        """UT-GL006-039: Test _calculate_waste method."""
        emissions, factors = agent._calculate_waste(valid_waste_input.activity_data)
        assert len(emissions) > 0
        assert len(factors) > 0

    @pytest.mark.unit
    def test_multiple_waste_streams(self, agent, valid_waste_input):
        """UT-GL006-040: Test multiple waste streams are calculated."""
        result = agent.run(valid_waste_input)
        assert len(result.emissions_by_source) >= 3

    @pytest.mark.unit
    def test_activity_data_model(self):
        """UT-GL006-041: Test ActivityData model."""
        activity = ActivityData(
            activity_type="landfill_mixed",
            quantity=100.0,
            unit="kg"
        )
        assert activity.activity_type == "landfill_mixed"
        assert activity.quantity == 100.0

    @pytest.mark.unit
    def test_zero_activity_returns_zero(self, agent):
        """UT-GL006-042: Test zero activity returns zero emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[
                ActivityData(activity_type="landfill_mixed", quantity=0.0, unit="kg"),
            ],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0

    @pytest.mark.unit
    def test_recycling_metal_highest_avoided(self, agent):
        """UT-GL006-043: Test metal recycling has highest avoided emissions."""
        assert agent.WASTE_FACTORS["recycling_metal"] == -1.800

    @pytest.mark.unit
    def test_composting_low_emissions(self, agent):
        """UT-GL006-044: Test composting has low emissions."""
        assert agent.WASTE_FACTORS["composting"] == 0.010

    @pytest.mark.unit
    def test_category_5_output_structure(self, agent, valid_waste_input):
        """UT-GL006-045: Test Category 5 output has correct structure."""
        result = agent.run(valid_waste_input)
        assert result.category_number == 5
        assert result.category_name == "Waste Generated in Operations"


# =============================================================================
# Transport/Travel Calculations Tests (15 tests)
# =============================================================================

class TestTransportTravelCalculations:
    """Test suite for transport and travel calculations - 15 test cases."""

    @pytest.mark.unit
    def test_road_truck_calculation(self, agent, valid_upstream_transport_input):
        """UT-GL006-046: Test road truck transport calculation."""
        result = agent.run(valid_upstream_transport_input)
        assert "road_truck" in result.emissions_by_source

    @pytest.mark.unit
    def test_transport_formula(self, agent):
        """UT-GL006-047: Test formula: emissions = distance * weight * factor."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[
                TransportData(mode="road_truck", distance_km=1000.0, weight_tonnes=10.0),
            ],
        )
        result = agent.run(input_data)

        # tonne-km = 1000 * 10 = 10000
        # emissions = 10000 * 0.089 = 890 kgCO2e
        expected = 10000 * 0.089
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_transport_factor_road(self, agent):
        """UT-GL006-048: Test road truck transport factor."""
        assert agent.TRANSPORT_FACTORS["road_truck"] == 0.089

    @pytest.mark.unit
    def test_transport_factor_rail(self, agent):
        """UT-GL006-049: Test rail freight factor (lowest land)."""
        assert agent.TRANSPORT_FACTORS["rail_freight"] == 0.028

    @pytest.mark.unit
    def test_transport_factor_sea(self, agent):
        """UT-GL006-050: Test sea container factor (very low)."""
        assert agent.TRANSPORT_FACTORS["sea_container"] == 0.016

    @pytest.mark.unit
    def test_transport_factor_air(self, agent):
        """UT-GL006-051: Test air freight factor (highest)."""
        assert agent.TRANSPORT_FACTORS["air_freight"] == 0.602

    @pytest.mark.unit
    def test_business_travel_air_calculation(self, agent, valid_business_travel_input):
        """UT-GL006-052: Test business travel air calculation."""
        result = agent.run(valid_business_travel_input)
        # Long haul air should be in results
        assert any("air" in k for k in result.emissions_by_source)

    @pytest.mark.unit
    def test_travel_haul_type_classification(self, agent):
        """UT-GL006-053: Test air travel haul type classification."""
        # Short haul: <1500km
        # Medium haul: 1500-4000km
        # Long haul: >4000km
        assert agent.TRAVEL_FACTORS["air_short_haul"] == 0.255
        assert agent.TRAVEL_FACTORS["air_medium_haul"] == 0.156
        assert agent.TRAVEL_FACTORS["air_long_haul"] == 0.195

    @pytest.mark.unit
    def test_travel_round_trip_doubles_distance(self, agent):
        """UT-GL006-054: Test round trip doubles the distance."""
        one_way = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="car_average", distance_km=100.0, trip_type="one_way")],
        )
        round_trip = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="car_average", distance_km=100.0, trip_type="round_trip")],
        )

        one_way_result = agent.run(one_way)
        round_trip_result = agent.run(round_trip)

        assert round_trip_result.total_emissions_kgco2e == pytest.approx(
            one_way_result.total_emissions_kgco2e * 2, rel=0.01
        )

    @pytest.mark.unit
    def test_travel_factor_rail(self, agent):
        """UT-GL006-055: Test rail travel factor."""
        assert agent.TRAVEL_FACTORS["rail_average"] == 0.041

    @pytest.mark.unit
    def test_travel_factor_car(self, agent):
        """UT-GL006-056: Test car travel factor."""
        assert agent.TRAVEL_FACTORS["car_average"] == 0.171

    @pytest.mark.unit
    def test_travel_factor_electric_car(self, agent):
        """UT-GL006-057: Test electric car has lower factor."""
        assert agent.TRAVEL_FACTORS["car_electric"] == 0.053
        assert agent.TRAVEL_FACTORS["car_electric"] < agent.TRAVEL_FACTORS["car_average"]

    @pytest.mark.unit
    def test_category_4_output_structure(self, agent, valid_upstream_transport_input):
        """UT-GL006-058: Test Category 4 output structure."""
        result = agent.run(valid_upstream_transport_input)
        assert result.category_number == 4
        assert result.category_name == "Upstream Transportation and Distribution"

    @pytest.mark.unit
    def test_category_6_output_structure(self, agent, valid_business_travel_input):
        """UT-GL006-059: Test Category 6 output structure."""
        result = agent.run(valid_business_travel_input)
        assert result.category_number == 6
        assert result.category_name == "Business Travel"

    @pytest.mark.unit
    def test_downstream_transport_category_9(self, agent):
        """UT-GL006-060: Test Category 9 (Downstream Transport) works."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[
                TransportData(mode="road_truck", distance_km=500.0, weight_tonnes=20.0),
            ],
        )
        result = agent.run(input_data)
        assert result.category_number == 9


# =============================================================================
# Data Quality and Error Handling Tests (15 tests)
# =============================================================================

class TestDataQualityAndErrorHandling:
    """Test suite for data quality and error handling - 15 test cases."""

    @pytest.mark.unit
    def test_data_quality_score_generated(self, agent, valid_purchased_goods_input):
        """UT-GL006-061: Test data quality score is generated."""
        result = agent.run(valid_purchased_goods_input)
        assert result.data_quality_score in ["very_good", "good", "fair", "poor", "very_poor"]

    @pytest.mark.unit
    def test_data_quality_enum_values(self):
        """UT-GL006-062: Test DataQualityScore enum values."""
        assert DataQualityScore.VERY_GOOD.value == "very_good"
        assert DataQualityScore.GOOD.value == "good"
        assert DataQualityScore.FAIR.value == "fair"
        assert DataQualityScore.POOR.value == "poor"
        assert DataQualityScore.VERY_POOR.value == "very_poor"

    @pytest.mark.unit
    def test_supplier_specific_best_quality(self, agent, supplier_specific_input):
        """UT-GL006-063: Test supplier-specific method has best quality."""
        result = agent.run(supplier_specific_input)
        assert result.data_quality_score == "very_good"

    @pytest.mark.unit
    def test_spend_based_fair_quality(self, agent, valid_purchased_goods_input):
        """UT-GL006-064: Test spend-based method has fair quality."""
        result = agent.run(valid_purchased_goods_input)
        assert result.data_quality_score == "fair"

    @pytest.mark.unit
    def test_data_coverage_percentage(self, agent, valid_purchased_goods_input):
        """UT-GL006-065: Test data coverage percentage is calculated."""
        result = agent.run(valid_purchased_goods_input)
        assert 0 <= result.data_coverage_pct <= 100

    @pytest.mark.unit
    def test_uncertainty_percentage(self, agent, valid_purchased_goods_input):
        """UT-GL006-066: Test uncertainty percentage is estimated."""
        result = agent.run(valid_purchased_goods_input)
        assert result.uncertainty_pct is not None
        assert result.uncertainty_pct > 0

    @pytest.mark.unit
    def test_uncertainty_lower_for_better_quality(self, agent):
        """UT-GL006-067: Test uncertainty is lower for better quality."""
        uncertainty_vg = agent._estimate_uncertainty(DataQualityScore.VERY_GOOD)
        uncertainty_poor = agent._estimate_uncertainty(DataQualityScore.POOR)
        assert uncertainty_vg < uncertainty_poor

    @pytest.mark.unit
    def test_emissions_per_revenue_calculated(self, agent, valid_purchased_goods_input):
        """UT-GL006-068: Test emissions per revenue intensity is calculated."""
        result = agent.run(valid_purchased_goods_input)
        assert result.emissions_per_revenue is not None
        assert result.emissions_per_revenue > 0

    @pytest.mark.unit
    def test_emissions_per_employee_calculated(self, agent, valid_purchased_goods_input):
        """UT-GL006-069: Test emissions per employee intensity is calculated."""
        result = agent.run(valid_purchased_goods_input)
        assert result.emissions_per_employee is not None
        assert result.emissions_per_employee > 0

    @pytest.mark.unit
    def test_improvement_opportunities_generated(self, agent, valid_purchased_goods_input):
        """UT-GL006-070: Test improvement opportunities are generated."""
        result = agent.run(valid_purchased_goods_input)
        assert isinstance(result.improvement_opportunities, list)
        assert len(result.improvement_opportunities) > 0

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_purchased_goods_input):
        """UT-GL006-071: Test provenance hash is generated."""
        result = agent.run(valid_purchased_goods_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_output_timestamp(self, agent, valid_purchased_goods_input):
        """UT-GL006-072: Test output includes timestamp."""
        result = agent.run(valid_purchased_goods_input)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    @pytest.mark.unit
    def test_get_categories_method(self, agent):
        """UT-GL006-073: Test get_categories utility method."""
        categories = agent.get_categories()
        assert len(categories) == 15
        assert all("category" in c and "number" in c and "name" in c for c in categories)

    @pytest.mark.unit
    def test_calculation_method_enum_values(self):
        """UT-GL006-074: Test CalculationMethod enum values."""
        assert CalculationMethod.SPEND_BASED.value == "spend_based"
        assert CalculationMethod.AVERAGE_DATA.value == "average_data"
        assert CalculationMethod.SUPPLIER_SPECIFIC.value == "supplier_specific"
        assert CalculationMethod.HYBRID.value == "hybrid"

    @pytest.mark.unit
    def test_deterministic_calculation(self, agent, valid_purchased_goods_input):
        """UT-GL006-075: Test calculation is deterministic."""
        result1 = agent.run(valid_purchased_goods_input)
        result2 = agent.run(valid_purchased_goods_input)
        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = Scope3EmissionsAgent()
        assert agent is not None
        assert agent.AGENT_ID == "emissions/scope3_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_category_names_loaded(self):
        """Test category names are loaded."""
        agent = Scope3EmissionsAgent()
        assert len(agent.CATEGORY_NAMES) == 15

    @pytest.mark.unit
    def test_spend_factors_loaded(self):
        """Test spend factors are loaded."""
        agent = Scope3EmissionsAgent()
        assert len(agent.SPEND_FACTORS) > 0


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedScope3:
    """Parametrized tests for Scope 3 scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize("category,expected_number", [
        (Scope3Category.CAT_1_PURCHASED_GOODS, 1),
        (Scope3Category.CAT_2_CAPITAL_GOODS, 2),
        (Scope3Category.CAT_3_FUEL_ENERGY, 3),
        (Scope3Category.CAT_6_BUSINESS_TRAVEL, 6),
        (Scope3Category.CAT_15_INVESTMENTS, 15),
    ])
    def test_category_numbers(self, agent, category, expected_number):
        """Test category numbers are correct."""
        cat_info = agent.CATEGORY_NAMES[category]
        assert cat_info[0] == expected_number

    @pytest.mark.unit
    @pytest.mark.parametrize("spend_category,expected_factor", [
        ("steel", 0.85),
        ("aluminum", 1.20),
        ("plastics", 0.75),
        ("services", 0.15),
        ("default", 0.40),
    ])
    def test_spend_emission_factors(self, agent, spend_category, expected_factor):
        """Test spend emission factors."""
        assert agent.SPEND_FACTORS[spend_category].factor_kgco2e_per_usd == expected_factor


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
