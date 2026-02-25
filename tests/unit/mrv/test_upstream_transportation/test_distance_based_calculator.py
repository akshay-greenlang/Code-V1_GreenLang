"""
Unit tests for DistanceBasedCalculatorEngine - AGENT-MRV-017 Upstream Transportation & Distribution.

Tests distance-based emission calculations across all transport modes.
"""

import pytest
from decimal import Decimal
from typing import List

from greenlang.mrv.upstream_transportation.engines.distance_based_calculator import (
    DistanceBasedCalculatorEngine,
)
from greenlang.mrv.upstream_transportation.engines.transport_database import (
    TransportMode,
    VehicleType,
    VesselType,
    AircraftType,
    RailType,
    FuelType,
    LoadState,
    Region,
)
from greenlang.mrv.upstream_transportation.models import (
    DistanceBasedInput,
    DistanceBasedOutput,
    TransportLeg,
    GasBreakdown,
)


@pytest.fixture
def calculator():
    """Create DistanceBasedCalculatorEngine instance."""
    return DistanceBasedCalculatorEngine()


class TestRoadCalculations:
    """Test road transport distance-based calculations."""

    def test_calculate_road_basic(self, calculator):
        """Test basic road transport calculation (20t × 500km × articulated HGV)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        output = calculator.calculate(input_data)

        # 20 tonnes × 500 km × 0.0794 kgCO2e/tonne-km = 794 kgCO2e
        expected_emissions = Decimal("20") * Decimal("500") * Decimal("0.0794")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.mode == TransportMode.ROAD
        assert output.data_quality_score >= Decimal("0.8")  # High quality (actual distance)

    def test_calculate_road_laden_full(self, calculator):
        """Test road calculation with laden state = full (100% load)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("26.5"),  # Full payload
            distance_km=Decimal("1000"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            laden_state=LoadState.FULL,
        )

        output = calculator.calculate(input_data)

        # Laden adjustment should reduce per tonne-km factor
        base_emissions = Decimal("26.5") * Decimal("1000") * Decimal("0.0794")
        assert output.total_emissions_kgco2e < base_emissions

    def test_calculate_road_laden_half(self, calculator):
        """Test road calculation with laden state = half (50% load)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("13.25"),  # Half payload
            distance_km=Decimal("1000"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            laden_state=LoadState.HALF,
        )

        output = calculator.calculate(input_data)

        # Laden adjustment should increase per tonne-km factor
        base_emissions = Decimal("13.25") * Decimal("1000") * Decimal("0.0794")
        assert output.total_emissions_kgco2e > base_emissions

    def test_calculate_road_region_eu(self, calculator):
        """Test road calculation for EU region."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.EU,
        )

        output = calculator.calculate(input_data)

        assert output.total_emissions_kgco2e > Decimal("0")
        assert output.region == Region.EU

    def test_calculate_road_region_us(self, calculator):
        """Test road calculation for US region (SmartWay data)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.US,
        )

        output = calculator.calculate(input_data)

        assert output.total_emissions_kgco2e > Decimal("0")
        assert output.region == Region.US


class TestRailCalculations:
    """Test rail transport distance-based calculations."""

    def test_calculate_rail_diesel(self, calculator):
        """Test diesel rail freight calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.RAIL,
            mass_tonnes=Decimal("100"),
            distance_km=Decimal("800"),
            rail_type=RailType.FREIGHT_DIESEL,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        output = calculator.calculate(input_data)

        # 100 tonnes × 800 km × 0.0278 kgCO2e/tonne-km = 2224 kgCO2e
        expected_emissions = Decimal("100") * Decimal("800") * Decimal("0.0278")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.mode == TransportMode.RAIL

    def test_calculate_rail_electric_eu(self, calculator):
        """Test electric rail freight calculation (EU grid)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.RAIL,
            mass_tonnes=Decimal("100"),
            distance_km=Decimal("800"),
            rail_type=RailType.FREIGHT_ELECTRIC,
            fuel_type=FuelType.ELECTRICITY,
            region=Region.EU,
        )

        output = calculator.calculate(input_data)

        # 100 tonnes × 800 km × 0.0095 kgCO2e/tonne-km = 760 kgCO2e
        expected_emissions = Decimal("100") * Decimal("800") * Decimal("0.0095")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.total_emissions_kgco2e < Decimal("2224")  # Lower than diesel


class TestMaritimeCalculations:
    """Test maritime transport distance-based calculations."""

    def test_calculate_maritime_container_panamax(self, calculator):
        """Test container ship Panamax calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("5000"),
            distance_km=Decimal("8000"),
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
        )

        output = calculator.calculate(input_data)

        # 5000 tonnes × 8000 km × 0.0105 kgCO2e/tonne-km = 420000 kgCO2e
        expected_emissions = Decimal("5000") * Decimal("8000") * Decimal("0.0105")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_maritime_bulk_capesize(self, calculator):
        """Test bulk carrier Capesize calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("150000"),  # Large bulk cargo
            distance_km=Decimal("12000"),
            vessel_type=VesselType.BULK_CAPESIZE,
            fuel_type=FuelType.HFO,
        )

        output = calculator.calculate(input_data)

        # 150000 tonnes × 12000 km × 0.0051 kgCO2e/tonne-km = 9180000 kgCO2e
        expected_emissions = Decimal("150000") * Decimal("12000") * Decimal("0.0051")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_maritime_tanker_vlcc(self, calculator):
        """Test tanker VLCC calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("200000"),  # VLCC crude oil cargo
            distance_km=Decimal("10000"),
            vessel_type=VesselType.TANKER_VLCC,
            fuel_type=FuelType.HFO,
        )

        output = calculator.calculate(input_data)

        # 200000 tonnes × 10000 km × 0.0048 kgCO2e/tonne-km = 9600000 kgCO2e
        expected_emissions = Decimal("200000") * Decimal("10000") * Decimal("0.0048")
        assert output.total_emissions_kgco2e == expected_emissions


class TestAirCalculations:
    """Test air freight distance-based calculations."""

    def test_calculate_air_widebody_freighter(self, calculator):
        """Test widebody dedicated freighter calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.AIR,
            mass_tonnes=Decimal("50"),
            distance_km=Decimal("5000"),
            aircraft_type=AircraftType.WIDEBODY_FREIGHTER,
            fuel_type=FuelType.JET_KEROSENE,
        )

        output = calculator.calculate(input_data)

        # 50 tonnes × 5000 km × 0.5984 kgCO2e/tonne-km = 149600 kgCO2e
        expected_emissions = Decimal("50") * Decimal("5000") * Decimal("0.5984")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_air_belly_cargo_low_emissions(self, calculator):
        """Test belly cargo (lower allocated emissions)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.AIR,
            mass_tonnes=Decimal("10"),
            distance_km=Decimal("5000"),
            aircraft_type=AircraftType.BELLY_CARGO,
            fuel_type=FuelType.JET_KEROSENE,
        )

        output = calculator.calculate(input_data)

        # Belly cargo has lower allocated emissions than dedicated freighter
        freighter_emissions = Decimal("10") * Decimal("5000") * Decimal("0.5984")
        assert output.total_emissions_kgco2e < freighter_emissions

    def test_calculate_air_narrowbody_high_emissions(self, calculator):
        """Test narrowbody aircraft (higher per km for short-haul)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.AIR,
            mass_tonnes=Decimal("5"),
            distance_km=Decimal("500"),  # Short-haul
            aircraft_type=AircraftType.NARROWBODY_FREIGHTER,
            fuel_type=FuelType.JET_KEROSENE,
        )

        output = calculator.calculate(input_data)

        # Short-haul has higher emissions per tonne-km
        assert output.total_emissions_kgco2e > Decimal("0")


class TestPipelineCalculations:
    """Test pipeline transport calculations."""

    def test_calculate_pipeline_crude_oil(self, calculator):
        """Test crude oil pipeline calculation."""
        input_data = DistanceBasedInput(
            mode=TransportMode.PIPELINE,
            mass_tonnes=Decimal("10000"),
            distance_km=Decimal("500"),
            product_type="crude_oil",
            fuel_type=FuelType.NATURAL_GAS,
        )

        output = calculator.calculate(input_data)

        # 10000 tonnes × 500 km × 0.0021 kgCO2e/tonne-km = 10500 kgCO2e
        expected_emissions = Decimal("10000") * Decimal("500") * Decimal("0.0021")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.mode == TransportMode.PIPELINE


class TestIntermodalCalculations:
    """Test intermodal transport calculations."""

    def test_calculate_intermodal_truck_ship_rail_truck(self, calculator):
        """Test intermodal calculation (truck → ship → rail → truck)."""
        legs = [
            TransportLeg(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("100"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            TransportLeg(
                mode=TransportMode.MARITIME,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("8000"),
                vessel_type=VesselType.CONTAINER_PANAMAX,
                fuel_type=FuelType.HFO,
            ),
            TransportLeg(
                mode=TransportMode.RAIL,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("500"),
                rail_type=RailType.FREIGHT_DIESEL,
                fuel_type=FuelType.DIESEL,
                region=Region.EU,
            ),
            TransportLeg(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("50"),
                vehicle_type=VehicleType.RIGID_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.EU,
            ),
        ]

        output = calculator.calculate_intermodal(legs)

        # Total emissions = sum of all legs
        leg1_emissions = Decimal("20") * Decimal("100") * Decimal("0.0794")
        leg2_emissions = Decimal("20") * Decimal("8000") * Decimal("0.0105")
        leg3_emissions = Decimal("20") * Decimal("500") * Decimal("0.0278")
        leg4_emissions = Decimal("20") * Decimal("50") * Decimal("0.2065")

        expected_total = leg1_emissions + leg2_emissions + leg3_emissions + leg4_emissions
        assert output.total_emissions_kgco2e == expected_total
        assert len(output.legs) == 4


class TestLoadFactorAdjustments:
    """Test load factor and empty running adjustments."""

    def test_apply_load_factor_50_percent(self, calculator):
        """Test load factor adjustment (50% capacity utilization)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            load_factor=Decimal("0.50"),  # 50% utilization
        )

        output = calculator.calculate(input_data)

        # With 50% load factor, emissions are doubled per tonne
        base_emissions = Decimal("20") * Decimal("500") * Decimal("0.0794")
        adjusted_emissions = base_emissions / Decimal("0.50")
        assert output.total_emissions_kgco2e == adjusted_emissions

    def test_apply_load_factor_100_percent(self, calculator):
        """Test load factor adjustment (100% capacity utilization)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            load_factor=Decimal("1.0"),  # 100% utilization (no adjustment)
        )

        output = calculator.calculate(input_data)

        # With 100% load factor, no adjustment
        base_emissions = Decimal("20") * Decimal("500") * Decimal("0.0794")
        assert output.total_emissions_kgco2e == base_emissions

    def test_apply_empty_running(self, calculator):
        """Test empty running adjustment (return voyage)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("5000"),
            distance_km=Decimal("8000"),
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
            apply_empty_running=True,  # Account for return voyage
        )

        output = calculator.calculate(input_data)

        # Empty running adds 50% (1.5x multiplier)
        base_emissions = Decimal("5000") * Decimal("8000") * Decimal("0.0105")
        adjusted_emissions = base_emissions * Decimal("1.5")
        assert output.total_emissions_kgco2e == adjusted_emissions

    def test_apply_laden_adjustment(self, calculator):
        """Test laden adjustment for partially loaded vehicle."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("13.25"),  # Half payload
            distance_km=Decimal("1000"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            laden_state=LoadState.HALF,
        )

        output = calculator.calculate(input_data)

        # Laden adjustment should increase emissions
        base_emissions = Decimal("13.25") * Decimal("1000") * Decimal("0.0794")
        assert output.total_emissions_kgco2e > base_emissions


class TestReeferUplifts:
    """Test refrigerated transport uplift factors."""

    def test_apply_reefer_uplift_road(self, calculator):
        """Test reefer uplift for road transport (15%)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            is_refrigerated=True,
        )

        output = calculator.calculate(input_data)

        # Reefer adds 15% (1.15x multiplier)
        base_emissions = Decimal("20") * Decimal("500") * Decimal("0.0794")
        reefer_emissions = base_emissions * Decimal("1.15")
        assert output.total_emissions_kgco2e == reefer_emissions

    def test_apply_reefer_uplift_maritime(self, calculator):
        """Test reefer uplift for maritime transport (25%)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("1000"),
            distance_km=Decimal("8000"),
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
            is_refrigerated=True,
        )

        output = calculator.calculate(input_data)

        # Reefer adds 25% (1.25x multiplier)
        base_emissions = Decimal("1000") * Decimal("8000") * Decimal("0.0105")
        reefer_emissions = base_emissions * Decimal("1.25")
        assert output.total_emissions_kgco2e == reefer_emissions


class TestDistanceCalculations:
    """Test great circle distance and corrections."""

    def test_calculate_great_circle_distance_london_to_new_york(self, calculator):
        """Test great circle distance calculation (London to New York)."""
        distance = calculator.calculate_great_circle_distance(
            origin_lat=Decimal("51.5074"),  # London
            origin_lon=Decimal("-0.1278"),
            dest_lat=Decimal("40.7128"),  # New York
            dest_lon=Decimal("-74.0060"),
        )

        assert Decimal("5550") < distance < Decimal("5600")  # ~5570 km

    def test_calculate_great_circle_distance_shanghai_to_rotterdam(self, calculator):
        """Test great circle distance (Shanghai to Rotterdam)."""
        distance = calculator.calculate_great_circle_distance(
            origin_lat=Decimal("31.2304"),  # Shanghai
            origin_lon=Decimal("121.4737"),
            dest_lat=Decimal("51.9225"),  # Rotterdam
            dest_lon=Decimal("4.4792"),
        )

        assert Decimal("18000") < distance < Decimal("19000")  # ~18500 km (via Suez)

    def test_apply_gcd_correction_1_09(self, calculator):
        """Test great circle distance correction factor (1.09 for actual routing)."""
        gcd_distance = Decimal("5570")  # London to New York GCD
        actual_distance = calculator.apply_gcd_correction(
            gcd_distance,
            correction_factor=Decimal("1.09"),  # Typical sea route correction
        )

        assert actual_distance == gcd_distance * Decimal("1.09")
        assert actual_distance == Decimal("6071.30")


class TestUnitConversions:
    """Test unit conversion methods."""

    def test_convert_distance_km_to_miles(self, calculator):
        """Test kilometer to miles conversion."""
        miles = calculator.convert_distance(
            distance=Decimal("100"),
            from_unit="km",
            to_unit="miles",
        )

        assert miles == Decimal("62.1371")

    def test_convert_distance_km_to_nautical_miles(self, calculator):
        """Test kilometer to nautical miles conversion."""
        nm = calculator.convert_distance(
            distance=Decimal("100"),
            from_unit="km",
            to_unit="nautical_miles",
        )

        assert nm == Decimal("53.9957")

    def test_convert_mass_tonnes_to_kg(self, calculator):
        """Test tonnes to kilograms conversion."""
        kg = calculator.convert_mass(
            mass=Decimal("5"),
            from_unit="tonnes",
            to_unit="kg",
        )

        assert kg == Decimal("5000")

    def test_convert_mass_tonnes_to_lbs(self, calculator):
        """Test tonnes to pounds conversion."""
        lbs = calculator.convert_mass(
            mass=Decimal("1"),
            from_unit="tonnes",
            to_unit="lbs",
        )

        assert lbs == Decimal("2204.62")


class TestGasSplitting:
    """Test gas breakdown calculations."""

    def test_split_by_gas_road(self, calculator):
        """Test gas splitting for road transport (CO2, CH4, N2O)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        output = calculator.calculate(input_data)

        gas_breakdown = output.gas_breakdown
        assert gas_breakdown.co2_kgco2e > Decimal("0")
        assert gas_breakdown.ch4_kgco2e > Decimal("0")
        assert gas_breakdown.n2o_kgco2e > Decimal("0")

        # Sum equals total
        total_from_gases = (
            gas_breakdown.co2_kgco2e +
            gas_breakdown.ch4_kgco2e +
            gas_breakdown.n2o_kgco2e
        )
        assert total_from_gases == output.total_emissions_kgco2e

    def test_split_by_gas_maritime(self, calculator):
        """Test gas splitting for maritime transport."""
        input_data = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("5000"),
            distance_km=Decimal("8000"),
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
        )

        output = calculator.calculate(input_data)

        gas_breakdown = output.gas_breakdown
        # CO2 is dominant for HFO combustion
        assert gas_breakdown.co2_kgco2e > Decimal("0.95") * output.total_emissions_kgco2e


class TestBiogenicSplit:
    """Test biogenic vs fossil CO2 splitting."""

    def test_calculate_biogenic_split_diesel_zero(self, calculator):
        """Test biogenic split for 100% fossil diesel (zero biogenic)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            biofuel_blend_percent=Decimal("0"),  # 100% fossil
        )

        output = calculator.calculate(input_data)

        assert output.biogenic_co2_kgco2 == Decimal("0")
        assert output.fossil_co2_kgco2e == output.gas_breakdown.co2_kgco2e

    def test_calculate_biogenic_split_biodiesel_b100(self, calculator):
        """Test biogenic split for B100 biodiesel (100% biogenic)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.BIODIESEL,
            region=Region.UK,
            biofuel_blend_percent=Decimal("100"),  # 100% biogenic
        )

        output = calculator.calculate(input_data)

        # All CO2 is biogenic
        assert output.biogenic_co2_kgco2 > Decimal("0")
        assert output.fossil_co2_kgco2e == Decimal("0")


class TestWTWBreakdown:
    """Test well-to-wheel (WTW) emission breakdowns."""

    def test_calculate_wtw_breakdown(self, calculator):
        """Test WTW breakdown (WTT + TTW)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            include_wtw=True,
        )

        output = calculator.calculate(input_data)

        # WTW = WTT (upstream) + TTW (combustion)
        assert output.wtt_emissions_kgco2e > Decimal("0")
        assert output.ttw_emissions_kgco2e > Decimal("0")
        assert output.total_emissions_kgco2e == (
            output.wtt_emissions_kgco2e + output.ttw_emissions_kgco2e
        )


class TestBatchCalculations:
    """Test batch processing capabilities."""

    def test_batch_calculate(self, calculator):
        """Test batch calculation of multiple shipments."""
        inputs = [
            DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("500"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            DistanceBasedInput(
                mode=TransportMode.RAIL,
                mass_tonnes=Decimal("100"),
                distance_km=Decimal("800"),
                rail_type=RailType.FREIGHT_DIESEL,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            DistanceBasedInput(
                mode=TransportMode.AIR,
                mass_tonnes=Decimal("5"),
                distance_km=Decimal("3000"),
                aircraft_type=AircraftType.WIDEBODY_FREIGHTER,
                fuel_type=FuelType.JET_KEROSENE,
            ),
        ]

        outputs = calculator.batch_calculate(inputs)

        assert len(outputs) == 3
        assert all(o.total_emissions_kgco2e > Decimal("0") for o in outputs)

    def test_aggregate_by_mode(self, calculator):
        """Test aggregation of emissions by transport mode."""
        inputs = [
            DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("500"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("15"),
                distance_km=Decimal("300"),
                vehicle_type=VehicleType.RIGID_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            DistanceBasedInput(
                mode=TransportMode.RAIL,
                mass_tonnes=Decimal("100"),
                distance_km=Decimal("800"),
                rail_type=RailType.FREIGHT_DIESEL,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
        ]

        outputs = calculator.batch_calculate(inputs)
        aggregated = calculator.aggregate_by_mode(outputs)

        assert TransportMode.ROAD in aggregated
        assert TransportMode.RAIL in aggregated
        assert aggregated[TransportMode.ROAD] > aggregated[TransportMode.RAIL]


class TestDataQualityScoring:
    """Test data quality scoring."""

    def test_get_data_quality_score_actual_distance(self, calculator):
        """Test data quality score for actual distance (high quality)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            distance_source="actual",  # High quality
        )

        output = calculator.calculate(input_data)

        assert output.data_quality_score >= Decimal("0.8")

    def test_get_data_quality_score_estimated(self, calculator):
        """Test data quality score for estimated distance (medium quality)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            distance_source="estimated",  # Medium quality
        )

        output = calculator.calculate(input_data)

        assert Decimal("0.5") <= output.data_quality_score < Decimal("0.8")


class TestDecimalPrecision:
    """Test Decimal arithmetic precision."""

    def test_decimal_precision(self, calculator):
        """Test that all calculations use Decimal (not float)."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20.123456789"),
            distance_km=Decimal("500.987654321"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        output = calculator.calculate(input_data)

        assert isinstance(output.total_emissions_kgco2e, Decimal)
        assert isinstance(output.gas_breakdown.co2_kgco2e, Decimal)


class TestValidation:
    """Test input validation and error handling."""

    def test_zero_mass_raises(self, calculator):
        """Test that zero mass raises ValueError."""
        with pytest.raises(ValueError, match="Mass must be greater than zero"):
            input_data = DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("0"),
                distance_km=Decimal("500"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )
            calculator.calculate(input_data)

    def test_zero_distance_raises(self, calculator):
        """Test that zero distance raises ValueError."""
        with pytest.raises(ValueError, match="Distance must be greater than zero"):
            input_data = DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("0"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )
            calculator.calculate(input_data)

    def test_negative_values_raises(self, calculator):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            input_data = DistanceBasedInput(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("-20"),
                distance_km=Decimal("500"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )
            calculator.calculate(input_data)


class TestSanityChecks:
    """Test sanity checks and expected relationships."""

    def test_emissions_are_positive(self, calculator):
        """Test that emissions are always positive."""
        input_data = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        output = calculator.calculate(input_data)

        assert output.total_emissions_kgco2e > Decimal("0")

    def test_air_higher_than_rail(self, calculator):
        """Test that air freight has higher emissions than rail (per tonne-km)."""
        air_input = DistanceBasedInput(
            mode=TransportMode.AIR,
            mass_tonnes=Decimal("10"),
            distance_km=Decimal("1000"),
            aircraft_type=AircraftType.WIDEBODY_FREIGHTER,
            fuel_type=FuelType.JET_KEROSENE,
        )

        rail_input = DistanceBasedInput(
            mode=TransportMode.RAIL,
            mass_tonnes=Decimal("10"),
            distance_km=Decimal("1000"),
            rail_type=RailType.FREIGHT_DIESEL,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        air_output = calculator.calculate(air_input)
        rail_output = calculator.calculate(rail_input)

        assert air_output.total_emissions_kgco2e > rail_output.total_emissions_kgco2e

    def test_maritime_lower_than_road(self, calculator):
        """Test that maritime has lower emissions than road (per tonne-km)."""
        maritime_input = DistanceBasedInput(
            mode=TransportMode.MARITIME,
            mass_tonnes=Decimal("1000"),
            distance_km=Decimal("1000"),
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
        )

        road_input = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("1000"),
            distance_km=Decimal("1000"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        maritime_output = calculator.calculate(maritime_input)
        road_output = calculator.calculate(road_input)

        assert maritime_output.total_emissions_kgco2e < road_output.total_emissions_kgco2e

    def test_pipeline_lowest(self, calculator):
        """Test that pipeline has lowest emissions (per tonne-km)."""
        pipeline_input = DistanceBasedInput(
            mode=TransportMode.PIPELINE,
            mass_tonnes=Decimal("1000"),
            distance_km=Decimal("1000"),
            product_type="crude_oil",
            fuel_type=FuelType.NATURAL_GAS,
        )

        road_input = DistanceBasedInput(
            mode=TransportMode.ROAD,
            mass_tonnes=Decimal("1000"),
            distance_km=Decimal("1000"),
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )

        pipeline_output = calculator.calculate(pipeline_input)
        road_output = calculator.calculate(road_input)

        assert pipeline_output.total_emissions_kgco2e < road_output.total_emissions_kgco2e

    def test_intermodal_sum_equals_legs(self, calculator):
        """Test that intermodal total equals sum of individual legs."""
        legs = [
            TransportLeg(
                mode=TransportMode.ROAD,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("100"),
                vehicle_type=VehicleType.ARTICULATED_HGV,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
            TransportLeg(
                mode=TransportMode.RAIL,
                mass_tonnes=Decimal("20"),
                distance_km=Decimal("500"),
                rail_type=RailType.FREIGHT_DIESEL,
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            ),
        ]

        intermodal_output = calculator.calculate_intermodal(legs)

        # Calculate legs individually
        leg1_output = calculator.calculate(legs[0])
        leg2_output = calculator.calculate(legs[1])

        sum_of_legs = leg1_output.total_emissions_kgco2e + leg2_output.total_emissions_kgco2e

        assert intermodal_output.total_emissions_kgco2e == sum_of_legs
