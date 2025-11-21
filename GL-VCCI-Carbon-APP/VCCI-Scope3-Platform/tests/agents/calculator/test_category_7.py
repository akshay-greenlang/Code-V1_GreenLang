# -*- coding: utf-8 -*-
"""
Tests for Category 7: Employee Commuting Calculator
GL-VCCI Scope 3 Platform

Comprehensive test suite with 30+ tests covering:
- All commute modes (car, bus, train, bike, walk, motorcycle)
- LLM survey analysis
- WFH calculations
- Tier fallback logic
- Edge cases and error handling

Version: 1.0.0
Date: 2025-11-08
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from services.agents.calculator.categories.category_7 import (
    Category7Calculator,
    Category7Input,
    CommuteMode,
)
from services.agents.calculator.config import TierType
from services.agents.calculator.exceptions import (
    DataValidationError,
    CalculationError,
)


@pytest.fixture
def mock_factor_broker():
    """Mock FactorBroker for testing."""
    broker = Mock()
    broker.resolve = AsyncMock()
    return broker


@pytest.fixture
def mock_uncertainty_engine():
    """Mock UncertaintyEngine for testing."""
    engine = Mock()
    engine.propagate_commute = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def mock_provenance_builder():
    """Mock ProvenanceChainBuilder for testing."""
    from services.agents.calculator.models import ProvenanceChain, DataQualityInfo

    builder = Mock()
    builder.build = AsyncMock(return_value=ProvenanceChain(
        calculation_id="test-calc-123",
        category=7,
        tier=TierType.TIER_2,
        input_data_hash="abc123",
        calculation={},
        data_quality=DataQualityInfo(
            dqi_score=70.0,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.5
        ),
        provenance_chain=[]
    ))
    return builder


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient for testing."""
    client = Mock()
    return client


@pytest.fixture
def calculator(mock_factor_broker, mock_uncertainty_engine, mock_provenance_builder, mock_llm_client):
    """Create Category7Calculator with mocks."""
    return Category7Calculator(
        factor_broker=mock_factor_broker,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        llm_client=mock_llm_client
    )


# =============================================================================
# TIER 2 TESTS: Detailed Commute Data
# =============================================================================

@pytest.mark.asyncio
async def test_car_petrol_commute_tier2(calculator):
    """Test car petrol commute calculation."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=20.0,
        days_per_week=5.0,
        weeks_per_year=48,
        num_employees=1,
        car_occupancy=1.0
    )

    result = await calculator.calculate(input_data)

    # 20 km × 5 days × 48 weeks × 0.192 kgCO2e/km = 921.6 kgCO2e
    expected = 20.0 * 5.0 * 48 * 0.192
    assert abs(result.emissions_kgco2e - expected) < 0.01
    assert result.category == 7
    assert result.tier == TierType.TIER_2
    assert result.calculation_method == "detailed_commute_data"


@pytest.mark.asyncio
async def test_electric_car_commute_low_emissions(calculator):
    """Test electric car has lower emissions than petrol."""
    input_petrol = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=15.0,
        days_per_week=5.0,
        num_employees=1
    )

    input_electric = Category7Input(
        commute_mode=CommuteMode.CAR_ELECTRIC,
        distance_km=15.0,
        days_per_week=5.0,
        num_employees=1
    )

    result_petrol = await calculator.calculate(input_petrol)
    result_electric = await calculator.calculate(input_electric)

    assert result_electric.emissions_kgco2e < result_petrol.emissions_kgco2e
    assert result_electric.emissions_kgco2e > 0  # Not zero (grid emissions)


@pytest.mark.asyncio
async def test_bus_commute_tier2(calculator):
    """Test bus commute calculation."""
    input_data = Category7Input(
        commute_mode=CommuteMode.BUS,
        distance_km=12.0,
        days_per_week=5.0,
        weeks_per_year=48,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # 12 km × 5 days × 48 weeks × 0.103 kgCO2e/km
    expected = 12.0 * 5.0 * 48 * 0.103
    assert abs(result.emissions_kgco2e - expected) < 0.01
    assert result.tier == TierType.TIER_2


@pytest.mark.asyncio
async def test_train_commute_tier2(calculator):
    """Test train commute calculation."""
    input_data = Category7Input(
        commute_mode=CommuteMode.TRAIN,
        distance_km=30.0,
        days_per_week=5.0,
        weeks_per_year=48,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # 30 km × 5 days × 48 weeks × 0.041 kgCO2e/km
    expected = 30.0 * 5.0 * 48 * 0.041
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_bike_commute_zero_emissions(calculator):
    """Test bike commute has zero emissions."""
    input_data = Category7Input(
        commute_mode=CommuteMode.BIKE,
        distance_km=5.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 0.0
    assert result.emissions_tco2e == 0.0


@pytest.mark.asyncio
async def test_walk_commute_zero_emissions(calculator):
    """Test walk commute has zero emissions."""
    input_data = Category7Input(
        commute_mode=CommuteMode.WALK,
        distance_km=2.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 0.0


@pytest.mark.asyncio
async def test_carpool_reduces_emissions(calculator):
    """Test carpooling reduces per-person emissions."""
    input_solo = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=15.0,
        days_per_week=5.0,
        car_occupancy=1.0,
        num_employees=1
    )

    input_carpool = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=15.0,
        days_per_week=5.0,
        car_occupancy=2.0,  # 2 people in car
        num_employees=1
    )

    result_solo = await calculator.calculate(input_solo)
    result_carpool = await calculator.calculate(input_carpool)

    # Carpool should be half the emissions
    assert result_carpool.emissions_kgco2e == pytest.approx(result_solo.emissions_kgco2e / 2, rel=0.01)


@pytest.mark.asyncio
async def test_multiple_employees(calculator):
    """Test calculation scales with number of employees."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        num_employees=50
    )

    result = await calculator.calculate(input_data)

    # Should be 50x the single employee emissions
    expected_per_employee = 10.0 * 5.0 * 48 * 0.192
    expected_total = expected_per_employee * 50
    assert abs(result.emissions_kgco2e - expected_total) < 1.0


@pytest.mark.asyncio
async def test_partial_week_commute(calculator):
    """Test hybrid work (3 days per week in office)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=20.0,
        days_per_week=3.0,  # Hybrid: 3 days office, 2 days WFH
        weeks_per_year=48,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # 20 km × 3 days × 48 weeks × 0.192 kgCO2e/km
    expected = 20.0 * 3.0 * 48 * 0.192
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_custom_emission_factor(calculator):
    """Test using custom emission factor."""
    custom_ef = 0.250

    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        emission_factor=custom_ef,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # Should use custom EF instead of default
    expected = 10.0 * 5.0 * 48 * custom_ef
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_custom_weeks_per_year(calculator):
    """Test custom working weeks (accounting for vacation, etc.)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=15.0,
        days_per_week=5.0,
        weeks_per_year=45,  # 7 weeks vacation/holidays
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    expected = 15.0 * 5.0 * 45 * 0.192
    assert abs(result.emissions_kgco2e - expected) < 0.01


# =============================================================================
# TIER 3 TESTS: LLM Survey Analysis
# =============================================================================

@pytest.mark.asyncio
async def test_llm_survey_analysis_tier3(calculator):
    """Test LLM analysis of survey response."""
    input_data = Category7Input(
        survey_response="I drive my car to work about 15km each way, 4 days a week since we have hybrid work.",
        num_employees=1
    )

    with patch.object(calculator, '_analyze_commute_survey') as mock_analyze:
        mock_analyze.return_value = {
            "mode": "car_petrol",
            "distance_km": 15.0,
            "days_per_week": 4.0,
            "car_occupancy": 1.0,
            "confidence": 0.85
        }

        result = await calculator.calculate(input_data)

        assert result.tier == TierType.TIER_3
        assert result.calculation_method == "llm_survey_analysis"
        assert "LLM" in str(result.warnings)
        assert result.metadata["llm_analyzed"] is True
        mock_analyze.assert_called_once()


@pytest.mark.asyncio
async def test_llm_extracts_mode_from_text(calculator):
    """Test LLM correctly identifies transport mode from text."""
    input_data = Category7Input(
        survey_response="I take the train every day, about 25km journey",
        num_employees=1
    )

    with patch.object(calculator, '_analyze_commute_survey') as mock_analyze:
        mock_analyze.return_value = {
            "mode": "train",
            "distance_km": 25.0,
            "days_per_week": 5.0,
            "car_occupancy": 1.0,
            "confidence": 0.90
        }

        result = await calculator.calculate(input_data)

        assert "train" in result.metadata.get("commute_mode", "")


@pytest.mark.asyncio
async def test_llm_survey_without_llm_client(calculator):
    """Test survey analysis fails gracefully without LLM client."""
    calculator_no_llm = Category7Calculator(
        factor_broker=calculator.factor_broker,
        uncertainty_engine=calculator.uncertainty_engine,
        provenance_builder=calculator.provenance_builder,
        llm_client=None  # No LLM client
    )

    input_data = Category7Input(
        survey_response="I bike to work 5km each day",
        num_employees=1
    )

    # Should fall through to error (no aggregate data either)
    with pytest.raises(DataValidationError):
        await calculator_no_llm.calculate(input_data)


# =============================================================================
# TIER 3 TESTS: Aggregate Data
# =============================================================================

@pytest.mark.asyncio
async def test_aggregate_commute_tier3(calculator):
    """Test aggregate employee commute calculation."""
    input_data = Category7Input(
        total_employees=100,
        average_commute_km=12.0,
        wfh_percentage=None  # Full-time office
    )

    result = await calculator.calculate(input_data)

    # 100 employees × 12 km × 240 days/year × 0.150 avg EF
    expected = 100 * 12.0 * 240 * 0.150
    assert abs(result.emissions_kgco2e - expected) < 10.0
    assert result.tier == TierType.TIER_3
    assert result.calculation_method == "aggregate_average"


@pytest.mark.asyncio
async def test_aggregate_with_wfh_percentage(calculator):
    """Test aggregate calculation with WFH adjustment."""
    input_data = Category7Input(
        total_employees=100,
        average_commute_km=15.0,
        wfh_percentage=40.0  # 40% WFH = 60% office days
    )

    result = await calculator.calculate(input_data)

    # Should reduce working days by 40%
    # 100 × 15 × (240 * 0.6) × 0.150
    expected = 100 * 15.0 * (240 * 0.6) * 0.150
    assert abs(result.emissions_kgco2e - expected) < 10.0
    assert "WFH" in str(result.warnings)


@pytest.mark.asyncio
async def test_aggregate_low_data_quality(calculator):
    """Test aggregate method has lower data quality score."""
    input_data = Category7Input(
        total_employees=50,
        average_commute_km=10.0
    )

    result = await calculator.calculate(input_data)

    assert result.data_quality.dqi_score < 50.0
    assert result.data_quality.rating in ["fair", "poor"]


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_negative_distance_validation(calculator):
    """Test negative distance is rejected."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=-10.0,
        days_per_week=5.0,
        num_employees=1
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "distance" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_days_per_week(calculator):
    """Test invalid days per week is rejected."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=8.0,  # More than 7 days!
        num_employees=1
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "days_per_week" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_car_occupancy(calculator):
    """Test negative car occupancy is rejected."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        car_occupancy=-1.0,
        num_employees=1
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "car_occupancy" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zero_employees(calculator):
    """Test zero employees is rejected."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        num_employees=0
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "num_employees" in str(exc_info.value)


@pytest.mark.asyncio
async def test_no_valid_input_data(calculator):
    """Test completely insufficient input data."""
    input_data = Category7Input(
        num_employees=1
        # No mode, no distance, no survey, no aggregate data
    )

    with pytest.raises(DataValidationError) as exc_info:
        await calculator.calculate(input_data)

    assert "insufficient" in str(exc_info.value).lower()


# =============================================================================
# EDGE CASES & BOUNDARY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_very_short_commute(calculator):
    """Test very short commute (1km)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.WALK,
        distance_km=1.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_kgco2e == 0.0  # Walking


@pytest.mark.asyncio
async def test_very_long_commute(calculator):
    """Test very long commute (100km)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.TRAIN,
        distance_km=100.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # Should complete without error
    assert result.emissions_kgco2e > 0


@pytest.mark.asyncio
async def test_one_day_per_week(calculator):
    """Test minimal office attendance (1 day/week)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=20.0,
        days_per_week=1.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    expected = 20.0 * 1.0 * 48 * 0.192
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_fractional_days_per_week(calculator):
    """Test fractional days per week (e.g., 2.5 days)."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=15.0,
        days_per_week=2.5,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    expected = 15.0 * 2.5 * 48 * 0.192
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_motorcycle_commute(calculator):
    """Test motorcycle commute."""
    input_data = Category7Input(
        commute_mode=CommuteMode.MOTORCYCLE,
        distance_km=10.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    expected = 10.0 * 5.0 * 48 * 0.113
    assert abs(result.emissions_kgco2e - expected) < 0.01


@pytest.mark.asyncio
async def test_subway_commute(calculator):
    """Test subway/metro commute."""
    input_data = Category7Input(
        commute_mode=CommuteMode.SUBWAY,
        distance_km=8.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    expected = 8.0 * 5.0 * 48 * 0.028
    assert abs(result.emissions_kgco2e - expected) < 0.01


# =============================================================================
# METADATA & WARNINGS TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_low_car_occupancy_warning(calculator):
    """Test warning for suspiciously low car occupancy."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        car_occupancy=0.5,  # Less than 1 person?
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    # Should have warning about occupancy
    warnings_text = " ".join(result.warnings)
    assert "occupancy" in warnings_text.lower()


@pytest.mark.asyncio
async def test_excessive_days_warning(calculator):
    """Test warning for more than 5 days per week."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=6.0,  # 6-day work week
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    warnings_text = " ".join(result.warnings)
    assert "days" in warnings_text.lower() or "week" in warnings_text.lower()


@pytest.mark.asyncio
async def test_metadata_includes_mode(calculator):
    """Test result metadata includes commute mode."""
    input_data = Category7Input(
        commute_mode=CommuteMode.BUS,
        distance_km=10.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert "commute_mode" in result.metadata
    assert result.metadata["commute_mode"] == "bus"


@pytest.mark.asyncio
async def test_metadata_includes_annual_distance(calculator):
    """Test result metadata includes calculated annual distance."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=10.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert "annual_distance_km" in result.metadata
    expected_distance = 10.0 * 5.0 * 48
    assert abs(result.metadata["annual_distance_km"] - expected_distance) < 0.01


# =============================================================================
# EMISSIONS CONVERSION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_emissions_kg_to_tonnes_conversion(calculator):
    """Test emissions are correctly converted to tonnes."""
    input_data = Category7Input(
        commute_mode=CommuteMode.CAR_PETROL,
        distance_km=20.0,
        days_per_week=5.0,
        num_employees=1
    )

    result = await calculator.calculate(input_data)

    assert result.emissions_tco2e == result.emissions_kgco2e / 1000
