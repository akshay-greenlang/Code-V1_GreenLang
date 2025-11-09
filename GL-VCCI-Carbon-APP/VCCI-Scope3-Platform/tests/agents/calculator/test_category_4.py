"""
Comprehensive Test Suite for Category 4: Upstream Transportation & Distribution
GL-VCCI Scope 3 Platform

Tests transport calculations, modal shifts, distance calculations, and edge cases.

Total: 30 tests

Version: 1.0.0
Date: 2025-11-09
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.agents.calculator.categories.category_4 import Category4Calculator
from services.agents.calculator.models import (
    Category4Input,
    CalculationResult,
    DataQualityInfo,
    TierType,
)
from services.agents.calculator.config import get_config
from services.agents.calculator.exceptions import (
    DataValidationError,
    TierFallbackError,
    CalculationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator(mock_factor_broker, mock_uncertainty_engine, mock_provenance_builder):
    """Create Category4Calculator instance."""
    return Category4Calculator(
        factor_broker=mock_factor_broker,
        uncertainty_engine=mock_uncertainty_engine,
        provenance_builder=mock_provenance_builder,
        config=get_config()
    )


@pytest.fixture
def truck_transport_input():
    """Truck transport input."""
    return Category4Input(
        transport_mode="truck",
        distance_km=500.0,
        mass_kg=10000.0,
        region="US"
    )


@pytest.fixture
def sea_transport_input():
    """Sea freight transport input."""
    return Category4Input(
        transport_mode="sea_freight",
        distance_km=5000.0,
        mass_kg=50000.0,
        region="GLOBAL"
    )


@pytest.fixture
def air_transport_input():
    """Air freight transport input."""
    return Category4Input(
        transport_mode="air_freight",
        distance_km=2000.0,
        mass_kg=1000.0,
        region="US"
    )


@pytest.fixture
def rail_transport_input():
    """Rail transport input."""
    return Category4Input(
        transport_mode="rail",
        distance_km=800.0,
        mass_kg=25000.0,
        region="EU"
    )


# ============================================================================
# TRUCK TRANSPORT TESTS (6 tests)
# ============================================================================

class TestTruckTransport:
    """Test truck transport calculations."""

    @pytest.mark.asyncio
    async def test_truck_basic_calculation(self, calculator, truck_transport_input, mock_factor_broker):
        """Test basic truck transport calculation."""
        mock_factor = Mock()
        mock_factor.value = 0.062  # kgCO2e/tkm
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash_truck")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(truck_transport_input)

        # Calculation: 10 tonnes * 500 km * 0.062 kgCO2e/tkm = 310 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(310.0, rel=0.01)
        assert result.category == 4

    @pytest.mark.asyncio
    async def test_truck_short_distance(self, calculator, mock_factor_broker):
        """Test truck transport with short distance."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=50.0,  # Short haul
            mass_kg=5000.0,
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 5 tonnes * 50 km * 0.062 = 15.5 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(15.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_truck_long_distance(self, calculator, mock_factor_broker):
        """Test truck transport with long distance."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=2000.0,  # Long haul
            mass_kg=20000.0,
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 20 tonnes * 2000 km * 0.062 = 2480 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(2480.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_truck_heavy_load(self, calculator, mock_factor_broker):
        """Test truck transport with heavy load."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=500.0,
            mass_kg=30000.0,  # 30 tonnes (heavy)
            region="EU"
        )

        mock_factor = Mock()
        mock_factor.value = 0.055  # EU trucks slightly more efficient
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "ecoinvent"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_eu"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="EU"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 30 tonnes * 500 km * 0.055 = 825 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(825.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_truck_light_load(self, calculator, mock_factor_broker):
        """Test truck transport with light load."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=300.0,
            mass_kg=500.0,  # 0.5 tonnes (light)
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 0.5 tonnes * 300 km * 0.062 = 9.3 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(9.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_truck_regional_factors(self, calculator, mock_factor_broker):
        """Test truck transport with different regional factors."""
        regions = {"US": 0.062, "EU": 0.055, "CN": 0.070}

        for region, factor_value in regions.items():
            input_data = Category4Input(
                transport_mode="truck",
                distance_km=500.0,
                mass_kg=10000.0,
                region=region
            )

            mock_factor = Mock()
            mock_factor.value = factor_value
            mock_factor.unit = "kgCO2e/tkm"
            mock_factor.source = "defra"
            mock_factor.uncertainty = 0.15
            mock_factor.data_quality_score = 75
            mock_factor.factor_id = f"truck_ef_{region}"
            mock_factor.metadata = Mock(
                source_version="2024",
                gwp_standard=Mock(value="AR6"),
                reference_year=2024,
                geographic_scope=region
            )
            mock_factor.provenance = Mock(calculation_hash="hash")

            mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

            result = await calculator.calculate(input_data)

            expected = 10.0 * 500.0 * factor_value
            assert result.emissions_kgco2e == pytest.approx(expected, rel=0.01)


# ============================================================================
# SEA FREIGHT TESTS (6 tests)
# ============================================================================

class TestSeaFreight:
    """Test sea freight transport calculations."""

    @pytest.mark.asyncio
    async def test_sea_freight_basic_calculation(self, calculator, sea_transport_input, mock_factor_broker):
        """Test basic sea freight calculation."""
        mock_factor = Mock()
        mock_factor.value = 0.0042  # kgCO2e/tkm (much lower than truck)
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "sea_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash_sea")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(sea_transport_input)

        # 50 tonnes * 5000 km * 0.0042 = 1050 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(1050.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_sea_freight_intercontinental(self, calculator, mock_factor_broker):
        """Test sea freight for intercontinental shipment."""
        input_data = Category4Input(
            transport_mode="sea_freight",
            distance_km=15000.0,  # e.g., Asia to US
            mass_kg=100000.0,  # 100 tonnes
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.0042
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "sea_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 100 tonnes * 15000 km * 0.0042 = 6300 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(6300.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_sea_freight_container_ship(self, calculator, mock_factor_broker):
        """Test sea freight with container ship."""
        input_data = Category4Input(
            transport_mode="sea_freight",
            distance_km=8000.0,
            mass_kg=50000.0,
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.0042
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "container_ship_ef"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e > 0

    @pytest.mark.asyncio
    async def test_sea_freight_bulk_carrier(self, calculator, mock_factor_broker):
        """Test sea freight with bulk carrier."""
        input_data = Category4Input(
            transport_mode="sea_freight",
            distance_km=10000.0,
            mass_kg=200000.0,  # 200 tonnes (bulk)
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.0035  # Bulk carriers more efficient
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "bulk_carrier_ef"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 200 tonnes * 10000 km * 0.0035 = 7000 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(7000.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_sea_freight_very_long_distance(self, calculator, mock_factor_broker):
        """Test sea freight with very long distance."""
        input_data = Category4Input(
            transport_mode="sea_freight",
            distance_km=25000.0,  # Around the world
            mass_kg=50000.0,
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.0042
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "sea_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 50 tonnes * 25000 km * 0.0042 = 5250 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(5250.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_sea_freight_vs_air_freight(self, calculator, mock_factor_broker):
        """Test that sea freight has lower emissions than air freight."""
        # Sea freight
        sea_input = Category4Input(
            transport_mode="sea_freight",
            distance_km=5000.0,
            mass_kg=10000.0,
            region="GLOBAL"
        )

        mock_sea_factor = Mock()
        mock_sea_factor.value = 0.0042
        mock_sea_factor.unit = "kgCO2e/tkm"
        mock_sea_factor.source = "imo"
        mock_sea_factor.uncertainty = 0.20
        mock_sea_factor.data_quality_score = 70
        mock_sea_factor.factor_id = "sea_freight_ef"
        mock_sea_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_sea_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_sea_factor)

        sea_result = await calculator.calculate(sea_input)

        # Air freight
        air_input = Category4Input(
            transport_mode="air_freight",
            distance_km=5000.0,
            mass_kg=10000.0,
            region="GLOBAL"
        )

        mock_air_factor = Mock()
        mock_air_factor.value = 0.602
        mock_air_factor.unit = "kgCO2e/tkm"
        mock_air_factor.source = "defra"
        mock_air_factor.uncertainty = 0.25
        mock_air_factor.data_quality_score = 70
        mock_air_factor.factor_id = "air_freight_ef"
        mock_air_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_air_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_air_factor)

        air_result = await calculator.calculate(air_input)

        # Air should have much higher emissions
        assert air_result.emissions_kgco2e > sea_result.emissions_kgco2e * 50


# ============================================================================
# AIR FREIGHT TESTS (5 tests)
# ============================================================================

class TestAirFreight:
    """Test air freight transport calculations."""

    @pytest.mark.asyncio
    async def test_air_freight_basic_calculation(self, calculator, air_transport_input, mock_factor_broker):
        """Test basic air freight calculation."""
        mock_factor = Mock()
        mock_factor.value = 0.602  # kgCO2e/tkm (highest emissions)
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "air_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash_air")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(air_transport_input)

        # 1 tonne * 2000 km * 0.602 = 1204 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(1204.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_air_freight_international(self, calculator, mock_factor_broker):
        """Test air freight for international shipment."""
        input_data = Category4Input(
            transport_mode="air_freight",
            distance_km=8000.0,  # Transcontinental
            mass_kg=500.0,
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.602
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "air_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 0.5 tonnes * 8000 km * 0.602 = 2408 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(2408.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_air_freight_short_haul(self, calculator, mock_factor_broker):
        """Test air freight for short haul."""
        input_data = Category4Input(
            transport_mode="air_freight",
            distance_km=500.0,  # Short domestic flight
            mass_kg=200.0,
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.700  # Short haul higher per km
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "air_freight_short_ef"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 0.2 tonnes * 500 km * 0.700 = 70 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(70.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_air_freight_express(self, calculator, mock_factor_broker):
        """Test air freight express delivery."""
        input_data = Category4Input(
            transport_mode="air_freight",
            distance_km=3000.0,
            mass_kg=100.0,  # Small express package
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.650
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.25
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "air_freight_express_ef"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 0.1 tonnes * 3000 km * 0.650 = 195 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(195.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_air_freight_high_uncertainty(self, calculator, air_transport_input, mock_factor_broker, mock_uncertainty_engine):
        """Test air freight uncertainty propagation."""
        mock_factor = Mock()
        mock_factor.value = 0.602
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.30  # Higher uncertainty
        mock_factor.data_quality_score = 65
        mock_factor.factor_id = "air_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        mock_uncertainty_engine.propagate = AsyncMock(return_value=Mock(
            p50=1204.0,
            p95=1500.0,
            std_dev=150.0
        ))

        result = await calculator.calculate(air_transport_input)

        assert result.uncertainty is not None


# ============================================================================
# RAIL TRANSPORT TESTS (4 tests)
# ============================================================================

class TestRailTransport:
    """Test rail transport calculations."""

    @pytest.mark.asyncio
    async def test_rail_basic_calculation(self, calculator, rail_transport_input, mock_factor_broker):
        """Test basic rail transport calculation."""
        mock_factor = Mock()
        mock_factor.value = 0.022  # kgCO2e/tkm (efficient)
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "ecoinvent"
        mock_factor.uncertainty = 0.18
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "rail_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="EU"
        )
        mock_factor.provenance = Mock(calculation_hash="hash_rail")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(rail_transport_input)

        # 25 tonnes * 800 km * 0.022 = 440 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(440.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_rail_electric_vs_diesel(self, calculator, mock_factor_broker):
        """Test rail transport electric vs diesel."""
        # Electric rail
        input_electric = Category4Input(
            transport_mode="rail",
            distance_km=1000.0,
            mass_kg=50000.0,
            region="EU"  # EU has more electric rail
        )

        mock_electric_factor = Mock()
        mock_electric_factor.value = 0.015  # Electric rail lower
        mock_electric_factor.unit = "kgCO2e/tkm"
        mock_electric_factor.source = "ecoinvent"
        mock_electric_factor.uncertainty = 0.18
        mock_electric_factor.data_quality_score = 75
        mock_electric_factor.factor_id = "rail_electric_ef"
        mock_electric_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="EU"
        )
        mock_electric_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_electric_factor)

        result = await calculator.calculate(input_electric)

        # 50 tonnes * 1000 km * 0.015 = 750 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(750.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_rail_long_distance(self, calculator, mock_factor_broker):
        """Test rail transport long distance."""
        input_data = Category4Input(
            transport_mode="rail",
            distance_km=3000.0,  # Cross-country
            mass_kg=100000.0,  # 100 tonnes
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.025
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.18
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "rail_ef_us"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 100 tonnes * 3000 km * 0.025 = 7500 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(7500.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_rail_freight_efficiency(self, calculator, mock_factor_broker):
        """Test rail freight efficiency for heavy loads."""
        input_data = Category4Input(
            transport_mode="rail",
            distance_km=1500.0,
            mass_kg=200000.0,  # 200 tonnes (very heavy)
            region="EU"
        )

        mock_factor = Mock()
        mock_factor.value = 0.020  # Efficient for heavy loads
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "ecoinvent"
        mock_factor.uncertainty = 0.18
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "rail_freight_ef"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="EU"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 200 tonnes * 1500 km * 0.020 = 6000 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(6000.0, rel=0.01)


# ============================================================================
# MULTIMODAL & EDGE CASES (9 tests)
# ============================================================================

class TestMultimodalAndEdgeCases:
    """Test multimodal transport and edge cases."""

    @pytest.mark.asyncio
    async def test_validation_negative_distance(self, calculator):
        """Test validation error for negative distance."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=-500.0,
            mass_kg=10000.0,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_zero_distance(self, calculator):
        """Test validation error for zero distance."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=0.0,
            mass_kg=10000.0,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_negative_mass(self, calculator):
        """Test validation error for negative mass."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=500.0,
            mass_kg=-10000.0,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_zero_mass(self, calculator):
        """Test validation error for zero mass."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=500.0,
            mass_kg=0.0,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_validation_invalid_transport_mode(self, calculator):
        """Test validation error for invalid transport mode."""
        input_data = Category4Input(
            transport_mode="teleportation",  # Invalid
            distance_km=500.0,
            mass_kg=10000.0,
            region="US"
        )

        with pytest.raises(DataValidationError):
            await calculator.calculate(input_data)

    @pytest.mark.asyncio
    async def test_very_small_mass(self, calculator, mock_factor_broker):
        """Test calculation with very small mass."""
        input_data = Category4Input(
            transport_mode="truck",
            distance_km=100.0,
            mass_kg=0.001,  # 1 gram
            region="US"
        )

        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        assert result.emissions_kgco2e == pytest.approx(0.0000062, rel=0.01)

    @pytest.mark.asyncio
    async def test_very_long_distance(self, calculator, mock_factor_broker):
        """Test calculation with very long distance."""
        input_data = Category4Input(
            transport_mode="sea_freight",
            distance_km=40000.0,  # Around the world
            mass_kg=50000.0,
            region="GLOBAL"
        )

        mock_factor = Mock()
        mock_factor.value = 0.0042
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "imo"
        mock_factor.uncertainty = 0.20
        mock_factor.data_quality_score = 70
        mock_factor.factor_id = "sea_freight_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="GLOBAL"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(input_data)

        # 50 tonnes * 40000 km * 0.0042 = 8400 kgCO2e
        assert result.emissions_kgco2e == pytest.approx(8400.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_provenance_tracking(self, calculator, truck_transport_input, mock_factor_broker, mock_provenance_builder):
        """Test provenance tracking for transport calculations."""
        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(truck_transport_input)

        mock_provenance_builder.build.assert_called()
        call_kwargs = mock_provenance_builder.build.call_args[1]
        assert call_kwargs['category'] == 4

    @pytest.mark.asyncio
    async def test_data_quality_scoring(self, calculator, truck_transport_input, mock_factor_broker):
        """Test data quality scoring for transport calculations."""
        mock_factor = Mock()
        mock_factor.value = 0.062
        mock_factor.unit = "kgCO2e/tkm"
        mock_factor.source = "defra"
        mock_factor.uncertainty = 0.15
        mock_factor.data_quality_score = 75
        mock_factor.factor_id = "truck_ef_001"
        mock_factor.metadata = Mock(
            source_version="2024",
            gwp_standard=Mock(value="AR6"),
            reference_year=2024,
            geographic_scope="US"
        )
        mock_factor.provenance = Mock(calculation_hash="hash")

        mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)

        result = await calculator.calculate(truck_transport_input)

        assert result.data_quality is not None
        assert result.data_quality.dqi_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
