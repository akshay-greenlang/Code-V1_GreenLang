"""
Unit tests for TransportDatabaseEngine - AGENT-MRV-017 Upstream Transportation & Distribution.

Tests database lookup methods for emission factors across all transport modes.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
import threading

from greenlang.mrv.upstream_transportation.engines.transport_database import (
    TransportDatabaseEngine,
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
    EmissionFactor,
    VehicleClassification,
)


@pytest.fixture
def db_engine():
    """Create TransportDatabaseEngine instance."""
    return TransportDatabaseEngine()


class TestRoadEmissionFactors:
    """Test road transport emission factor lookups."""

    def test_get_road_emission_factor_articulated_40_44t(self, db_engine):
        """Test articulated HGV 40-44t emission factor."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )
        assert ef is not None
        assert ef.value > Decimal("0")
        assert ef.value == Decimal("0.0794")  # kgCO2e/tonne-km (DEFRA 2024)
        assert ef.unit == "kgCO2e/tonne-km"
        assert ef.source == "DEFRA 2024"

    def test_get_road_emission_factor_rigid_7_5_17t(self, db_engine):
        """Test rigid HGV 7.5-17t emission factor."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.RIGID_HGV,
            gvw_tonnes=Decimal("12"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )
        assert ef is not None
        assert ef.value == Decimal("0.2065")  # kgCO2e/tonne-km
        assert ef.unit == "kgCO2e/tonne-km"

    def test_get_road_emission_factor_lcv_diesel(self, db_engine):
        """Test light commercial vehicle (LCV) diesel emission factor."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.LCV,
            gvw_tonnes=Decimal("3.0"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )
        assert ef is not None
        assert ef.value == Decimal("0.5439")  # kgCO2e/tonne-km
        assert ef.gas_splits["CO2"] > Decimal("0.95")

    def test_get_road_emission_factor_with_laden_state_full(self, db_engine):
        """Test road EF with laden state = full (100% load)."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            laden_state=LoadState.FULL,
        )
        # Laden adjustment should be applied
        assert ef is not None
        assert ef.value < Decimal("0.0794")  # Lower per tonne-km when fully loaded

    def test_get_road_emission_factor_with_laden_state_empty(self, db_engine):
        """Test road EF with laden state = empty (0% load)."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            laden_state=LoadState.EMPTY,
        )
        # Empty running should have infinite EF per tonne-km (no cargo)
        # Engine should return base vehicle EF without cargo allocation
        assert ef is not None

    def test_get_road_emission_factor_with_region_eu(self, db_engine):
        """Test road EF for EU region (different from UK)."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
            fuel_type=FuelType.DIESEL,
            region=Region.EU,
        )
        assert ef is not None
        assert ef.region == Region.EU

    def test_get_road_emission_factor_with_region_us(self, db_engine):
        """Test road EF for US region (SmartWay data)."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("18"),  # US uses different weight classes
            fuel_type=FuelType.DIESEL,
            region=Region.US,
        )
        assert ef is not None
        assert ef.source == "EPA SmartWay 2024"


class TestRailEmissionFactors:
    """Test rail transport emission factor lookups."""

    def test_get_rail_emission_factor_diesel(self, db_engine):
        """Test diesel rail freight emission factor."""
        ef = db_engine.get_rail_emission_factor(
            rail_type=RailType.FREIGHT_DIESEL,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0278")  # kgCO2e/tonne-km (DEFRA 2024)
        assert ef.unit == "kgCO2e/tonne-km"

    def test_get_rail_emission_factor_electric_eu(self, db_engine):
        """Test electric rail freight emission factor (EU grid)."""
        ef = db_engine.get_rail_emission_factor(
            rail_type=RailType.FREIGHT_ELECTRIC,
            fuel_type=FuelType.ELECTRICITY,
            region=Region.EU,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0095")  # kgCO2e/tonne-km
        assert ef.value < Decimal("0.0278")  # Lower than diesel

    def test_get_rail_emission_factor_average(self, db_engine):
        """Test average rail freight emission factor (mixed diesel/electric)."""
        ef = db_engine.get_rail_emission_factor(
            rail_type=RailType.FREIGHT_AVERAGE,
            fuel_type=None,  # Mixed
            region=Region.UK,
        )
        assert ef is not None
        assert Decimal("0.0095") < ef.value < Decimal("0.0278")


class TestMaritimeEmissionFactors:
    """Test maritime transport emission factor lookups."""

    def test_get_maritime_emission_factor_container_panamax(self, db_engine):
        """Test container ship Panamax emission factor."""
        ef = db_engine.get_maritime_emission_factor(
            vessel_type=VesselType.CONTAINER_PANAMAX,
            fuel_type=FuelType.HFO,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0105")  # kgCO2e/tonne-km (GLEC 2.0)
        assert ef.unit == "kgCO2e/tonne-km"

    def test_get_maritime_emission_factor_container_ulcv(self, db_engine):
        """Test container ship ULCV (Ultra Large) emission factor."""
        ef = db_engine.get_maritime_emission_factor(
            vessel_type=VesselType.CONTAINER_ULCV,
            fuel_type=FuelType.HFO,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0038")  # Lower per tonne-km (economies of scale)
        assert ef.value < Decimal("0.0105")

    def test_get_maritime_emission_factor_bulk_capesize(self, db_engine):
        """Test bulk carrier Capesize emission factor."""
        ef = db_engine.get_maritime_emission_factor(
            vessel_type=VesselType.BULK_CAPESIZE,
            fuel_type=FuelType.HFO,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0051")  # kgCO2e/tonne-km

    def test_get_maritime_emission_factor_tanker_vlcc(self, db_engine):
        """Test tanker VLCC (Very Large Crude Carrier) emission factor."""
        ef = db_engine.get_maritime_emission_factor(
            vessel_type=VesselType.TANKER_VLCC,
            fuel_type=FuelType.HFO,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0048")  # kgCO2e/tonne-km

    def test_get_maritime_emission_factor_inland_barge(self, db_engine):
        """Test inland waterway barge emission factor."""
        ef = db_engine.get_maritime_emission_factor(
            vessel_type=VesselType.BARGE_INLAND,
            fuel_type=FuelType.DIESEL,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0310")  # kgCO2e/tonne-km (DEFRA 2024)


class TestAirEmissionFactors:
    """Test air freight emission factor lookups."""

    def test_get_air_emission_factor_widebody_freighter(self, db_engine):
        """Test widebody dedicated freighter emission factor."""
        ef = db_engine.get_air_emission_factor(
            aircraft_type=AircraftType.WIDEBODY_FREIGHTER,
            fuel_type=FuelType.JET_KEROSENE,
            distance_km=Decimal("5000"),  # Long-haul
        )
        assert ef is not None
        assert ef.value == Decimal("0.5984")  # kgCO2e/tonne-km (DEFRA 2024)
        assert ef.unit == "kgCO2e/tonne-km"

    def test_get_air_emission_factor_belly_cargo(self, db_engine):
        """Test belly cargo (passenger aircraft) low emission factor."""
        ef = db_engine.get_air_emission_factor(
            aircraft_type=AircraftType.BELLY_CARGO,
            fuel_type=FuelType.JET_KEROSENE,
            distance_km=Decimal("5000"),
        )
        # Belly cargo has lower allocated emissions (split with passengers)
        assert ef is not None
        assert ef.value < Decimal("0.5984")  # Lower than dedicated freighter

    def test_get_air_emission_factor_narrowbody_high_emissions(self, db_engine):
        """Test narrowbody aircraft higher emission factor (short-haul)."""
        ef = db_engine.get_air_emission_factor(
            aircraft_type=AircraftType.NARROWBODY_FREIGHTER,
            fuel_type=FuelType.JET_KEROSENE,
            distance_km=Decimal("500"),  # Short-haul (higher per km)
        )
        assert ef is not None
        assert ef.value > Decimal("0.5984")  # Higher for short distances


class TestPipelineEmissionFactors:
    """Test pipeline transport emission factor lookups."""

    def test_get_pipeline_emission_factor_crude_oil(self, db_engine):
        """Test crude oil pipeline emission factor."""
        ef = db_engine.get_pipeline_emission_factor(
            product_type="crude_oil",
            fuel_type=FuelType.NATURAL_GAS,  # Pumps powered by gas
        )
        assert ef is not None
        assert ef.value == Decimal("0.0021")  # kgCO2e/tonne-km (very low)
        assert ef.unit == "kgCO2e/tonne-km"

    def test_get_pipeline_emission_factor_natural_gas(self, db_engine):
        """Test natural gas pipeline emission factor."""
        ef = db_engine.get_pipeline_emission_factor(
            product_type="natural_gas",
            fuel_type=FuelType.ELECTRICITY,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0019")  # kgCO2e/tonne-km
        assert ef.value < Decimal("0.0021")  # Slightly lower


class TestFuelEmissionFactors:
    """Test fuel-based emission factor lookups."""

    def test_get_fuel_emission_factor_diesel_ttw(self, db_engine):
        """Test diesel TTW (tank-to-wheel) emission factor."""
        ef = db_engine.get_fuel_emission_factor(
            fuel_type=FuelType.DIESEL,
            scope="ttw",
        )
        assert ef is not None
        assert ef.value == Decimal("2.6868")  # kgCO2e/litre (DEFRA 2024)
        assert ef.unit == "kgCO2e/litre"

    def test_get_fuel_emission_factor_diesel_wtw(self, db_engine):
        """Test diesel WTW (well-to-wheel) emission factor."""
        ef = db_engine.get_fuel_emission_factor(
            fuel_type=FuelType.DIESEL,
            scope="wtw",
        )
        assert ef is not None
        assert ef.value == Decimal("3.1869")  # kgCO2e/litre (TTW + WTT)
        assert ef.value > Decimal("2.6868")

    def test_get_fuel_emission_factor_jet_kerosene(self, db_engine):
        """Test jet kerosene emission factor."""
        ef = db_engine.get_fuel_emission_factor(
            fuel_type=FuelType.JET_KEROSENE,
            scope="ttw",
        )
        assert ef is not None
        assert ef.value == Decimal("2.5392")  # kgCO2e/litre

    def test_get_fuel_emission_factor_hfo(self, db_engine):
        """Test heavy fuel oil (HFO) emission factor."""
        ef = db_engine.get_fuel_emission_factor(
            fuel_type=FuelType.HFO,
            scope="ttw",
        )
        assert ef is not None
        assert ef.value == Decimal("3.1144")  # kgCO2e/litre


class TestEEIOFactors:
    """Test EEIO (Environmentally-Extended Input-Output) factors."""

    def test_get_eeio_factor_trucking(self, db_engine):
        """Test trucking EEIO factor."""
        ef = db_engine.get_eeio_factor(
            service_type="trucking",
            region=Region.US,
        )
        assert ef is not None
        assert ef.value == Decimal("0.5821")  # kgCO2e/USD (EPA 2024)
        assert ef.unit == "kgCO2e/USD"

    def test_get_eeio_factor_air_freight(self, db_engine):
        """Test air freight EEIO factor."""
        ef = db_engine.get_eeio_factor(
            service_type="air_freight",
            region=Region.US,
        )
        assert ef is not None
        assert ef.value > Decimal("0.5821")  # Higher than trucking

    def test_get_eeio_factor_rail(self, db_engine):
        """Test rail EEIO factor."""
        ef = db_engine.get_eeio_factor(
            service_type="rail",
            region=Region.US,
        )
        assert ef is not None
        assert ef.value < Decimal("0.5821")  # Lower than trucking


class TestHubEmissionFactors:
    """Test hub/warehouse emission factor lookups."""

    def test_get_hub_emission_factor_logistics_hub(self, db_engine):
        """Test logistics hub emission factor."""
        ef = db_engine.get_hub_emission_factor(
            hub_type="logistics_hub",
            region=Region.EU,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0185")  # kgCO2e/tonne-handled (GLEC 2.0)
        assert ef.unit == "kgCO2e/tonne"

    def test_get_hub_emission_factor_cold_storage(self, db_engine):
        """Test cold storage warehouse emission factor (higher due to refrigeration)."""
        ef = db_engine.get_hub_emission_factor(
            hub_type="cold_storage",
            region=Region.EU,
        )
        assert ef is not None
        assert ef.value == Decimal("0.0421")  # kgCO2e/tonne-handled
        assert ef.value > Decimal("0.0185")  # Higher than standard warehouse


class TestReeferUplifts:
    """Test refrigerated transport uplift factors."""

    def test_get_reefer_uplift_road(self, db_engine):
        """Test reefer uplift for road transport."""
        uplift = db_engine.get_reefer_uplift(
            mode=TransportMode.ROAD,
        )
        assert uplift == Decimal("1.15")  # 15% uplift for refrigeration

    def test_get_reefer_uplift_maritime(self, db_engine):
        """Test reefer uplift for maritime transport."""
        uplift = db_engine.get_reefer_uplift(
            mode=TransportMode.MARITIME,
        )
        assert uplift == Decimal("1.25")  # 25% uplift for refrigerated containers


class TestLoadFactors:
    """Test load factor lookups."""

    def test_get_load_factor_road(self, db_engine):
        """Test default road transport load factor."""
        load_factor = db_engine.get_load_factor(
            mode=TransportMode.ROAD,
            region=Region.UK,
        )
        assert load_factor == Decimal("0.57")  # DEFRA 2024 default

    def test_get_empty_running_rate_maritime(self, db_engine):
        """Test empty running rate for maritime (return voyage)."""
        empty_rate = db_engine.get_empty_running_rate(
            mode=TransportMode.MARITIME,
        )
        assert empty_rate == Decimal("0.50")  # 50% of voyages are empty


class TestVehicleClassification:
    """Test vehicle classification helpers."""

    def test_classify_vehicle_by_gvw_small(self, db_engine):
        """Test vehicle classification for small GVW."""
        classification = db_engine.classify_vehicle_by_gvw(
            gvw_tonnes=Decimal("3.0"),
        )
        assert classification.vehicle_type == VehicleType.LCV
        assert classification.weight_class == "up to 3.5t"

    def test_classify_vehicle_by_gvw_large(self, db_engine):
        """Test vehicle classification for large GVW."""
        classification = db_engine.classify_vehicle_by_gvw(
            gvw_tonnes=Decimal("42"),
        )
        assert classification.vehicle_type == VehicleType.ARTICULATED_HGV
        assert classification.weight_class == "40-44t"

    def test_classify_vessel_container(self, db_engine):
        """Test vessel classification for container ship."""
        classification = db_engine.classify_vessel(
            dwt=Decimal("80000"),  # 80k DWT
            vessel_category="container",
        )
        assert classification.vessel_type == VesselType.CONTAINER_PANAMAX

    def test_classify_vessel_bulk(self, db_engine):
        """Test vessel classification for bulk carrier."""
        classification = db_engine.classify_vessel(
            dwt=Decimal("180000"),  # 180k DWT
            vessel_category="bulk",
        )
        assert classification.vessel_type == VesselType.BULK_CAPESIZE

    def test_classify_aircraft_short_haul(self, db_engine):
        """Test aircraft classification for short-haul."""
        classification = db_engine.classify_aircraft(
            distance_km=Decimal("500"),
        )
        assert "short" in classification.range_category.lower()

    def test_classify_aircraft_long_haul(self, db_engine):
        """Test aircraft classification for long-haul."""
        classification = db_engine.classify_aircraft(
            distance_km=Decimal("8000"),
        )
        assert "long" in classification.range_category.lower()


class TestPayloadAdjustments:
    """Test payload and laden state adjustments."""

    def test_get_vehicle_payload(self, db_engine):
        """Test typical vehicle payload capacity."""
        payload = db_engine.get_vehicle_payload(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("44"),
        )
        assert payload == Decimal("26.5")  # Typical payload for 44t articulated

    def test_get_laden_adjustment_full(self, db_engine):
        """Test laden adjustment for fully loaded vehicle."""
        adjustment = db_engine.get_laden_adjustment(
            load_state=LoadState.FULL,
            vehicle_type=VehicleType.ARTICULATED_HGV,
        )
        assert adjustment < Decimal("1.0")  # Lower per tonne-km when full

    def test_get_laden_adjustment_half(self, db_engine):
        """Test laden adjustment for half-loaded vehicle."""
        adjustment = db_engine.get_laden_adjustment(
            load_state=LoadState.HALF,
            vehicle_type=VehicleType.ARTICULATED_HGV,
        )
        assert adjustment > Decimal("1.0")  # Higher per tonne-km when half-loaded


class TestEFHierarchy:
    """Test emission factor resolution hierarchy."""

    def test_resolve_ef_hierarchy(self, db_engine):
        """Test EF hierarchy resolution (specific → generic)."""
        # Try to get specific EF, fall back to generic if not available
        ef = db_engine.resolve_ef_hierarchy(
            mode=TransportMode.ROAD,
            vehicle_type=VehicleType.ARTICULATED_HGV,
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
            gvw_tonnes=Decimal("42"),
        )
        assert ef is not None
        assert ef.value > Decimal("0")


class TestUnitConversions:
    """Test unit conversion helpers."""

    def test_convert_units_km_to_miles(self, db_engine):
        """Test kilometer to miles conversion."""
        miles = db_engine.convert_distance(
            value=Decimal("100"),
            from_unit="km",
            to_unit="miles",
        )
        assert miles == Decimal("62.1371")

    def test_convert_units_tonnes_to_kg(self, db_engine):
        """Test tonnes to kilograms conversion."""
        kg = db_engine.convert_mass(
            value=Decimal("5"),
            from_unit="tonnes",
            to_unit="kg",
        )
        assert kg == Decimal("5000")

    def test_convert_units_litres_to_gallons(self, db_engine):
        """Test litres to gallons conversion."""
        gallons = db_engine.convert_volume(
            value=Decimal("100"),
            from_unit="litres",
            to_unit="gallons_us",
        )
        assert gallons == Decimal("26.4172")


class TestSourceSpecificFactors:
    """Test source-specific emission factor lookups."""

    def test_get_defra_factor(self, db_engine):
        """Test DEFRA-specific factor lookup."""
        ef = db_engine.get_defra_factor(
            mode=TransportMode.ROAD,
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
        )
        assert ef is not None
        assert ef.source == "DEFRA 2024"

    def test_get_smartway_factor(self, db_engine):
        """Test EPA SmartWay factor lookup (US)."""
        ef = db_engine.get_smartway_factor(
            vehicle_type="combination_truck",
        )
        assert ef is not None
        assert ef.source == "EPA SmartWay 2024"
        assert ef.region == Region.US

    def test_get_glec_factor(self, db_engine):
        """Test GLEC Framework factor lookup."""
        ef = db_engine.get_glec_factor(
            mode=TransportMode.MARITIME,
            vessel_type=VesselType.CONTAINER_PANAMAX,
        )
        assert ef is not None
        assert "GLEC" in ef.source


class TestFactorListing:
    """Test listing available emission factors."""

    def test_list_available_factors_road(self, db_engine):
        """Test listing all available road transport factors."""
        factors = db_engine.list_available_factors(
            mode=TransportMode.ROAD,
        )
        assert len(factors) > 10  # Multiple vehicle types, weight classes
        assert all(f.mode == TransportMode.ROAD for f in factors)

    def test_list_available_factors_all(self, db_engine):
        """Test listing all available factors (all modes)."""
        factors = db_engine.list_available_factors()
        assert len(factors) > 50  # Road + rail + maritime + air + pipeline


class TestSingletonAndThreadSafety:
    """Test singleton pattern and thread safety."""

    def test_singleton_pattern(self):
        """Test that TransportDatabaseEngine is a singleton."""
        engine1 = TransportDatabaseEngine()
        engine2 = TransportDatabaseEngine()
        assert engine1 is engine2

    def test_thread_safety(self, db_engine):
        """Test thread-safe concurrent access."""
        results = []

        def lookup_ef():
            ef = db_engine.get_road_emission_factor(
                vehicle_type=VehicleType.ARTICULATED_HGV,
                gvw_tonnes=Decimal("42"),
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )
            results.append(ef.value)

        threads = [threading.Thread(target=lookup_ef) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r == results[0] for r in results)  # All same result


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_vehicle_type(self, db_engine):
        """Test error on invalid vehicle type."""
        with pytest.raises(ValueError, match="Invalid vehicle type"):
            db_engine.get_road_emission_factor(
                vehicle_type="invalid_vehicle",
                gvw_tonnes=Decimal("42"),
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )

    def test_invalid_fuel_type(self, db_engine):
        """Test error on invalid fuel type."""
        with pytest.raises(ValueError, match="Invalid fuel type"):
            db_engine.get_fuel_emission_factor(
                fuel_type="invalid_fuel",
                scope="ttw",
            )


class TestDecimalArithmetic:
    """Test Decimal arithmetic precision."""

    def test_decimal_arithmetic(self, db_engine):
        """Test that all EFs use Decimal (not float)."""
        ef = db_engine.get_road_emission_factor(
            vehicle_type=VehicleType.ARTICULATED_HGV,
            gvw_tonnes=Decimal("42"),
            fuel_type=FuelType.DIESEL,
            region=Region.UK,
        )
        assert isinstance(ef.value, Decimal)
        assert isinstance(ef.gas_splits["CO2"], Decimal)
        assert isinstance(ef.gas_splits["CH4"], Decimal)
        assert isinstance(ef.gas_splits["N2O"], Decimal)


class TestDataCompleteness:
    """Test that all expected emission factors are present."""

    def test_all_road_factors_present(self, db_engine):
        """Test that all DEFRA road vehicle types have EFs."""
        vehicle_types = [
            VehicleType.LCV,
            VehicleType.RIGID_HGV,
            VehicleType.ARTICULATED_HGV,
        ]
        for vtype in vehicle_types:
            ef = db_engine.get_road_emission_factor(
                vehicle_type=vtype,
                gvw_tonnes=Decimal("12"),  # Mid-range
                fuel_type=FuelType.DIESEL,
                region=Region.UK,
            )
            assert ef is not None, f"Missing EF for {vtype}"

    def test_all_maritime_factors_present(self, db_engine):
        """Test that all major vessel types have EFs."""
        vessel_types = [
            VesselType.CONTAINER_PANAMAX,
            VesselType.CONTAINER_ULCV,
            VesselType.BULK_CAPESIZE,
            VesselType.TANKER_VLCC,
        ]
        for vtype in vessel_types:
            ef = db_engine.get_maritime_emission_factor(
                vessel_type=vtype,
                fuel_type=FuelType.HFO,
            )
            assert ef is not None, f"Missing EF for {vtype}"
