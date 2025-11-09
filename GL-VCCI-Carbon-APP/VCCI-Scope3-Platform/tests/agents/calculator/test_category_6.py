"""
Comprehensive Test Suite for Category 6: Business Travel
GL-VCCI Scope 3 Platform

Tests business travel calculations, flight classes, hotel stays, and car rentals.

Total: 25 tests

Version: 1.0.0
Date: 2025-11-09
"""

import pytest
from unittest.mock import Mock, AsyncMock
from services.agents.calculator.categories.category_6 import Category6Calculator
from services.agents.calculator.models import Category6Input, CalculationResult, TierType
from services.agents.calculator.config import get_config
from services.agents.calculator.exceptions import DataValidationError


@pytest.fixture
def calculator(mock_factor_broker, mock_uncertainty_engine, mock_provenance_builder):
    """Create Category6Calculator instance."""
    return Category6Calculator(
        factor_broker=mock_factor_broker,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=get_config()
    )


class TestFlightEmissions:
    """Test flight emissions calculations (10 tests)."""

    @pytest.mark.asyncio
    async def test_economy_short_haul(self, calculator, mock_factor_broker):
        """Test economy class short haul flight."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=500.0,
            passenger_count=1,
            flight_class="economy",
            region="US"
        )
        mock_factor = Mock(value=0.18, unit="kgCO2e/pkm", source="defra", uncertainty=0.20,
                          data_quality_score=75, factor_id="flight_economy_short",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(90.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_business_long_haul(self, calculator, mock_factor_broker):
        """Test business class long haul flight."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=8000.0,
            passenger_count=1,
            flight_class="business",
            region="GLOBAL"
        )
        mock_factor = Mock(value=0.52, unit="kgCO2e/pkm", source="defra", uncertainty=0.25,
                          data_quality_score=70, factor_id="flight_business_long",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="GLOBAL"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(4160.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_first_class_premium(self, calculator, mock_factor_broker):
        """Test first class flight (highest emissions)."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=5000.0,
            passenger_count=1,
            flight_class="first",
            region="GLOBAL"
        )
        mock_factor = Mock(value=0.95, unit="kgCO2e/pkm", source="defra", uncertainty=0.30,
                          data_quality_score=65, factor_id="flight_first_class",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="GLOBAL"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(4750.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_multiple_passengers(self, calculator, mock_factor_broker):
        """Test flight with multiple passengers."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=2000.0,
            passenger_count=5,
            flight_class="economy",
            region="EU"
        )
        mock_factor = Mock(value=0.15, unit="kgCO2e/pkm", source="ecoinvent", uncertainty=0.20,
                          data_quality_score=75, factor_id="flight_economy_medium",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(1500.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_round_trip(self, calculator, mock_factor_broker):
        """Test round trip flight."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=3000.0,
            passenger_count=1,
            flight_class="economy",
            is_round_trip=True,
            region="US"
        )
        mock_factor = Mock(value=0.16, unit="kgCO2e/pkm", source="defra", uncertainty=0.20,
                          data_quality_score=75, factor_id="flight_economy_long",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        # 3000 km * 2 (round trip) * 1 pax * 0.16 = 960 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(960.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_radiative_forcing(self, calculator, mock_factor_broker):
        """Test flight with radiative forcing factor."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=5000.0,
            passenger_count=1,
            flight_class="economy",
            include_radiative_forcing=True,
            region="GLOBAL"
        )
        mock_factor = Mock(value=0.15, unit="kgCO2e/pkm", source="defra", uncertainty=0.25,
                          data_quality_score=70, factor_id="flight_economy_rf",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="GLOBAL"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        # With RF, emissions typically 1.9x to 2.0x higher
        assert result.emissions_kgco2e > 750.0

    @pytest.mark.asyncio
    async def test_domestic_flight(self, calculator, mock_factor_broker):
        """Test domestic flight."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=800.0,
            passenger_count=1,
            flight_class="economy",
            region="US"
        )
        mock_factor = Mock(value=0.19, unit="kgCO2e/pkm", source="defra", uncertainty=0.20,
                          data_quality_score=75, factor_id="flight_domestic",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(152.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_international_flight(self, calculator, mock_factor_broker):
        """Test international flight."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=10000.0,
            passenger_count=1,
            flight_class="premium_economy",
            region="GLOBAL"
        )
        mock_factor = Mock(value=0.25, unit="kgCO2e/pkm", source="defra", uncertainty=0.22,
                          data_quality_score=72, factor_id="flight_premium_economy",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="GLOBAL"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(2500.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_layover_adjustment(self, calculator, mock_factor_broker):
        """Test flight with layover distance adjustment."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=6000.0,
            passenger_count=1,
            flight_class="economy",
            has_layover=True,
            region="GLOBAL"
        )
        mock_factor = Mock(value=0.14, unit="kgCO2e/pkm", source="defra", uncertainty=0.20,
                          data_quality_score=75, factor_id="flight_economy_long",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="GLOBAL"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        # Layovers typically add 10-15% to distance
        assert result.emissions_kgco2e >= 840.0

    @pytest.mark.asyncio
    async def test_flight_validation_negative_distance(self, calculator):
        """Test validation for negative flight distance."""
        input_data = Category6Input(
            travel_type="flight",
            distance_km=-1000.0,
            passenger_count=1,
            flight_class="economy",
            region="US"
        )
        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)


class TestCarRental:
    """Test car rental emissions calculations (8 tests)."""

    @pytest.mark.asyncio
    async def test_small_car(self, calculator, mock_factor_broker):
        """Test small car rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=200.0,
            vehicle_type="small_car",
            region="EU"
        )
        mock_factor = Mock(value=0.11, unit="kgCO2e/km", source="defra", uncertainty=0.15,
                          data_quality_score=80, factor_id="car_small",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(22.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_medium_car(self, calculator, mock_factor_broker):
        """Test medium car rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=500.0,
            vehicle_type="medium_car",
            region="US"
        )
        mock_factor = Mock(value=0.19, unit="kgCO2e/km", source="defra", uncertainty=0.15,
                          data_quality_score=80, factor_id="car_medium",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(95.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_large_car_suv(self, calculator, mock_factor_broker):
        """Test large car/SUV rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=300.0,
            vehicle_type="large_car",
            region="US"
        )
        mock_factor = Mock(value=0.28, unit="kgCO2e/km", source="defra", uncertainty=0.18,
                          data_quality_score=78, factor_id="car_large",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(84.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_electric_vehicle(self, calculator, mock_factor_broker):
        """Test electric vehicle rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=400.0,
            vehicle_type="electric",
            region="EU"
        )
        mock_factor = Mock(value=0.04, unit="kgCO2e/km", source="defra", uncertainty=0.25,
                          data_quality_score=70, factor_id="car_electric",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(16.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hybrid_vehicle(self, calculator, mock_factor_broker):
        """Test hybrid vehicle rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=350.0,
            vehicle_type="hybrid",
            region="US"
        )
        mock_factor = Mock(value=0.10, unit="kgCO2e/km", source="defra", uncertainty=0.18,
                          data_quality_score=75, factor_id="car_hybrid",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(35.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_van_rental(self, calculator, mock_factor_broker):
        """Test van rental."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=250.0,
            vehicle_type="van",
            region="EU"
        )
        mock_factor = Mock(value=0.35, unit="kgCO2e/km", source="defra", uncertainty=0.20,
                          data_quality_score=75, factor_id="van",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(87.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_car_occupancy_multiple(self, calculator, mock_factor_broker):
        """Test car rental with multiple occupants."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=300.0,
            vehicle_type="medium_car",
            passenger_count=3,
            region="US"
        )
        mock_factor = Mock(value=0.19, unit="kgCO2e/km", source="defra", uncertainty=0.15,
                          data_quality_score=80, factor_id="car_medium",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        # Total emissions divided by occupancy
        assert result.emissions_kgco2e == pytest.approx(19.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_car_validation_zero_distance(self, calculator):
        """Test validation for zero car distance."""
        input_data = Category6Input(
            travel_type="car_rental",
            distance_km=0.0,
            vehicle_type="medium_car",
            region="US"
        )
        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)


class TestHotelStays:
    """Test hotel stay emissions calculations (7 tests)."""

    @pytest.mark.asyncio
    async def test_hotel_basic_calculation(self, calculator, mock_factor_broker):
        """Test basic hotel stay calculation."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=3,
            hotel_type="standard",
            region="US"
        )
        mock_factor = Mock(value=12.0, unit="kgCO2e/night", source="defra", uncertainty=0.30,
                          data_quality_score=65, factor_id="hotel_standard",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(36.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_luxury(self, calculator, mock_factor_broker):
        """Test luxury hotel stay."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=5,
            hotel_type="luxury",
            region="EU"
        )
        mock_factor = Mock(value=25.0, unit="kgCO2e/night", source="defra", uncertainty=0.35,
                          data_quality_score=60, factor_id="hotel_luxury",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(125.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_budget(self, calculator, mock_factor_broker):
        """Test budget hotel stay."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=2,
            hotel_type="budget",
            region="US"
        )
        mock_factor = Mock(value=8.0, unit="kgCO2e/night", source="defra", uncertainty=0.28,
                          data_quality_score=68, factor_id="hotel_budget",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(16.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_long_stay(self, calculator, mock_factor_broker):
        """Test long hotel stay."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=14,
            hotel_type="standard",
            region="US"
        )
        mock_factor = Mock(value=12.0, unit="kgCO2e/night", source="defra", uncertainty=0.30,
                          data_quality_score=65, factor_id="hotel_standard",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="US"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(168.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_eco_certified(self, calculator, mock_factor_broker):
        """Test eco-certified hotel stay."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=4,
            hotel_type="eco",
            region="EU"
        )
        mock_factor = Mock(value=6.0, unit="kgCO2e/night", source="defra", uncertainty=0.32,
                          data_quality_score=62, factor_id="hotel_eco",
                          metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                      reference_year=2024, geographic_scope="EU"),
                          provenance=Mock(calculation_hash="hash"))
        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
        result = await calculator.calculate(input_data)
        assert result.emissions_kgco2e == pytest.approx(24.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_regional_differences(self, calculator, mock_factor_broker):
        """Test hotel emissions in different regions."""
        regions = {"US": 12.0, "EU": 10.0, "CN": 15.0}
        for region, factor_value in regions.items():
            input_data = Category6Input(
                travel_type="hotel",
                nights=3,
                hotel_type="standard",
                region=region
            )
            mock_factor = Mock(value=factor_value, unit="kgCO2e/night", source="defra", uncertainty=0.30,
                              data_quality_score=65, factor_id=f"hotel_{region}",
                              metadata=Mock(source_version="2024", gwp_standard=Mock(value="AR6"),
                                          reference_year=2024, geographic_scope=region),
                              provenance=Mock(calculation_hash="hash"))
            mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
            result = await calculator.calculate(input_data)
            assert result.emissions_kgco2e == pytest.approx(3 * factor_value, rel=0.01)

    @pytest.mark.asyncio
    async def test_hotel_validation_negative_nights(self, calculator):
        """Test validation for negative hotel nights."""
        input_data = Category6Input(
            travel_type="hotel",
            nights=-2,
            hotel_type="standard",
            region="US"
        )
        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
