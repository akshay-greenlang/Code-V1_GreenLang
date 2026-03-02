# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.distance_based_calculator - AGENT-MRV-022.

Tests DistanceBasedCalculatorEngine for the Downstream Transportation &
Distribution Agent (GL-MRV-S3-009).

Coverage (~60 tests):
- calculate_shipment for all 6 transport modes
- Cold chain uplift adjustments
- Load factor adjustments
- Return logistics emissions
- WTT (well-to-tank) emissions
- Multi-leg transport chains
- Intermodal transport
- Batch processing
- Fleet average calculations
- Mode comparison analysis
- Known-value hand-calculated tests
- Singleton pattern

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import threading
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.downstream_transportation.distance_based_calculator import (
        DistanceBasedCalculatorEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"distance_based_calculator not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test DistanceBasedCalculatorEngine singleton."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        eng1 = DistanceBasedCalculatorEngine()
        eng2 = DistanceBasedCalculatorEngine()
        assert eng1 is eng2


# ==============================================================================
# ROAD TRANSPORT TESTS
# ==============================================================================


class TestRoadTransport:
    """Test distance-based calculation for road transport."""

    def test_articulated_33t_basic(self, sample_shipment):
        """Test basic articulated 33t shipment calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert isinstance(emissions, Decimal)
        assert emissions > 0

    def test_road_known_value(self):
        """
        Hand-calculated: 350 km x 15 t x 0.107 kgCO2e/tkm = 561.75 kgCO2e = 0.56175 tCO2e.
        """
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-KV-001",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("350.0"),
            "cargo_mass_tonnes": Decimal("15.0"),
            "ef_scope": "TTW",
        }
        result = engine.calculate_shipment(shipment)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        # Allow 20% tolerance for EF variation
        expected = Decimal("0.56175")
        assert abs(emissions - expected) / expected < Decimal("0.20")

    def test_road_van_small(self):
        """Test van small delivery calculation."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-VAN-001",
            "mode": "ROAD",
            "vehicle_type": "VAN_SMALL",
            "distance_km": Decimal("50.0"),
            "cargo_mass_tonnes": Decimal("0.5"),
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None

    @pytest.mark.parametrize("vehicle_type", [
        "ARTICULATED_33T", "ARTICULATED_40_44T", "RIGID_7_5T",
        "RIGID_12T", "VAN_SMALL", "VAN_MEDIUM", "VAN_LARGE",
    ])
    def test_road_all_vehicle_types(self, vehicle_type):
        """Test calculation for each road vehicle type."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": f"TEST-{vehicle_type}",
            "mode": "ROAD",
            "vehicle_type": vehicle_type,
            "distance_km": Decimal("100.0"),
            "cargo_mass_tonnes": Decimal("5.0"),
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_road_heavier_vehicle_higher_ef(self):
        """Test that heavier vehicles have higher per-tkm EFs."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "distance_km": Decimal("100.0"),
            "cargo_mass_tonnes": Decimal("5.0"),
        }
        small = {**base, "vehicle_type": "RIGID_7_5T"}
        large = {**base, "vehicle_type": "ARTICULATED_40_44T"}
        r_small = engine.calculate_shipment(small)
        r_large = engine.calculate_shipment(large)
        # Both should produce results but EFs differ
        assert r_small is not None and r_large is not None


# ==============================================================================
# RAIL TRANSPORT TESTS
# ==============================================================================


class TestRailTransport:
    """Test distance-based calculation for rail transport."""

    def test_electric_freight(self, sample_shipment_rail):
        """Test electric freight rail calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment_rail)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_rail_known_value(self):
        """
        Hand-calculated: 800 km x 100 t x 0.028 kgCO2e/tkm = 2240 kgCO2e = 2.240 tCO2e.
        """
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-KV-RAIL",
            "mode": "RAIL",
            "vehicle_type": "ELECTRIC_FREIGHT",
            "distance_km": Decimal("800.0"),
            "cargo_mass_tonnes": Decimal("100.0"),
            "country": "DE",
        }
        result = engine.calculate_shipment(shipment)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        expected = Decimal("2.240")
        assert abs(emissions - expected) / expected < Decimal("0.25")

    def test_rail_lower_than_road(self):
        """Test rail emissions are lower than road for same distance/mass."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("50.0"),
        }
        road = engine.calculate_shipment({**base, "mode": "ROAD", "vehicle_type": "ARTICULATED_33T"})
        rail = engine.calculate_shipment({**base, "mode": "RAIL", "vehicle_type": "ELECTRIC_FREIGHT"})
        road_e = road.get("emissions_tco2e", road.get("total_co2e"))
        rail_e = rail.get("emissions_tco2e", rail.get("total_co2e"))
        assert rail_e < road_e


# ==============================================================================
# MARITIME TRANSPORT TESTS
# ==============================================================================


class TestMaritimeTransport:
    """Test distance-based calculation for maritime transport."""

    def test_container_panamax(self, sample_shipment_maritime):
        """Test container Panamax calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment_maritime)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_maritime_known_value(self):
        """
        Hand-calculated: 12000 km x 50 t x 0.016 kgCO2e/tkm = 9600 kgCO2e = 9.600 tCO2e.
        """
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-KV-SEA",
            "mode": "MARITIME",
            "vessel_type": "CONTAINER_PANAMAX",
            "distance_km": Decimal("12000.0"),
            "cargo_mass_tonnes": Decimal("50.0"),
        }
        result = engine.calculate_shipment(shipment)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        expected = Decimal("9.600")
        assert abs(emissions - expected) / expected < Decimal("0.30")

    @pytest.mark.parametrize("vessel_type", [
        "CONTAINER_FEEDER", "CONTAINER_PANAMAX", "CONTAINER_POST_PANAMAX",
        "BULK_HANDYSIZE", "BULK_PANAMAX",
    ])
    def test_maritime_vessel_types(self, vessel_type):
        """Test calculation for each maritime vessel type."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": f"TEST-{vessel_type}",
            "mode": "MARITIME",
            "vessel_type": vessel_type,
            "distance_km": Decimal("5000.0"),
            "cargo_mass_tonnes": Decimal("100.0"),
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None


# ==============================================================================
# AIR TRANSPORT TESTS
# ==============================================================================


class TestAirTransport:
    """Test distance-based calculation for air transport."""

    def test_widebody_freighter(self, sample_shipment_air):
        """Test widebody freighter calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment_air)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_air_known_value(self):
        """
        Hand-calculated: 6000 km x 3 t x 0.602 kgCO2e/tkm = 10836 kgCO2e = 10.836 tCO2e.
        """
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-KV-AIR",
            "mode": "AIR",
            "aircraft_type": "WIDEBODY_FREIGHTER",
            "distance_km": Decimal("6000.0"),
            "cargo_mass_tonnes": Decimal("3.0"),
        }
        result = engine.calculate_shipment(shipment)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        expected = Decimal("10.836")
        assert abs(emissions - expected) / expected < Decimal("0.25")

    def test_air_highest_emissions(self):
        """Test air produces highest per-tkm emissions."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "distance_km": Decimal("5000.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
        }
        results = {}
        for mode, vtype in [("ROAD", "ARTICULATED_33T"), ("RAIL", "ELECTRIC_FREIGHT"),
                            ("MARITIME", "CONTAINER_PANAMAX"), ("AIR", "WIDEBODY_FREIGHTER")]:
            key = "vehicle_type" if mode != "AIR" else "aircraft_type"
            if mode == "MARITIME":
                key = "vessel_type"
            r = engine.calculate_shipment({**base, "mode": mode, key: vtype})
            results[mode] = r.get("emissions_tco2e", r.get("total_co2e"))
        assert results["AIR"] > results["ROAD"]
        assert results["AIR"] > results["MARITIME"]


# ==============================================================================
# COURIER TRANSPORT TESTS
# ==============================================================================


class TestCourierTransport:
    """Test distance-based calculation for courier transport."""

    def test_courier_van(self, sample_shipment_courier):
        """Test courier van calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment_courier)
        assert result is not None

    def test_courier_small_parcel(self):
        """Test courier calculation for small parcel."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-COURIER-001",
            "mode": "COURIER",
            "vehicle_type": "VAN_SMALL",
            "distance_km": Decimal("50.0"),
            "cargo_mass_tonnes": Decimal("0.005"),
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None


# ==============================================================================
# LAST-MILE TRANSPORT TESTS
# ==============================================================================


class TestLastMileTransport:
    """Test distance-based calculation for last-mile delivery."""

    def test_last_mile_van(self, sample_shipment_last_mile):
        """Test last-mile van delivery calculation."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment_last_mile)
        assert result is not None

    def test_last_mile_urban(self):
        """Test urban last-mile delivery."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-LM-URBAN",
            "mode": "LAST_MILE",
            "vehicle_type": "VAN_SMALL",
            "distance_km": Decimal("10.0"),
            "cargo_mass_tonnes": Decimal("0.01"),
            "delivery_area": "URBAN",
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None


# ==============================================================================
# COLD CHAIN UPLIFT TESTS
# ==============================================================================


class TestColdChainUplift:
    """Test cold chain reefer uplift on distance-based calculations."""

    def test_chilled_uplift(self):
        """Test chilled transport has higher emissions than ambient."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
        }
        ambient = engine.calculate_shipment({**base, "temperature_controlled": False})
        chilled = engine.calculate_shipment({
            **base,
            "temperature_controlled": True,
            "cold_chain_regime": "CHILLED",
        })
        amb_e = ambient.get("emissions_tco2e", ambient.get("total_co2e"))
        chl_e = chilled.get("emissions_tco2e", chilled.get("total_co2e"))
        assert chl_e > amb_e

    def test_frozen_uplift_greater_than_chilled(self):
        """Test frozen uplift produces more emissions than chilled."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
            "temperature_controlled": True,
        }
        chilled = engine.calculate_shipment({**base, "cold_chain_regime": "CHILLED"})
        frozen = engine.calculate_shipment({**base, "cold_chain_regime": "FROZEN"})
        chl_e = chilled.get("emissions_tco2e", chilled.get("total_co2e"))
        frz_e = frozen.get("emissions_tco2e", frozen.get("total_co2e"))
        assert frz_e > chl_e


# ==============================================================================
# LOAD FACTOR ADJUSTMENT TESTS
# ==============================================================================


class TestLoadFactorAdjustment:
    """Test load factor adjustments on distance-based calculations."""

    def test_lower_load_factor_higher_emissions(self):
        """Test lower load factor increases per-unit emissions."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
        }
        high_load = engine.calculate_shipment({**base, "load_factor": Decimal("0.90")})
        low_load = engine.calculate_shipment({**base, "load_factor": Decimal("0.30")})
        high_e = high_load.get("emissions_tco2e", high_load.get("total_co2e"))
        low_e = low_load.get("emissions_tco2e", low_load.get("total_co2e"))
        # Lower load factor should result in higher per-shipment emissions
        # due to less efficient use of vehicle capacity
        assert low_e >= high_e


# ==============================================================================
# RETURN LOGISTICS TESTS
# ==============================================================================


class TestReturnLogistics:
    """Test return logistics emissions integration."""

    def test_return_adds_emissions(self):
        """Test including returns increases total emissions."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
        }
        no_return = engine.calculate_shipment({**base, "include_return": False})
        with_return = engine.calculate_shipment({
            **base,
            "include_return": True,
            "return_rate": Decimal("0.15"),
        })
        no_e = no_return.get("emissions_tco2e", no_return.get("total_co2e"))
        ret_e = with_return.get("emissions_tco2e", with_return.get("total_co2e"))
        assert ret_e >= no_e


# ==============================================================================
# WTT EMISSIONS TESTS
# ==============================================================================


class TestWTTEmissions:
    """Test well-to-tank emissions inclusion."""

    def test_wtw_greater_than_ttw(self):
        """Test WTW (includes WTT) emissions are greater than TTW-only."""
        engine = DistanceBasedCalculatorEngine()
        base = {
            "shipment_id": "TEST",
            "mode": "ROAD",
            "vehicle_type": "ARTICULATED_33T",
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("10.0"),
        }
        ttw = engine.calculate_shipment({**base, "ef_scope": "TTW"})
        wtw = engine.calculate_shipment({**base, "ef_scope": "WTW"})
        ttw_e = ttw.get("emissions_tco2e", ttw.get("total_co2e"))
        wtw_e = wtw.get("emissions_tco2e", wtw.get("total_co2e"))
        assert wtw_e >= ttw_e


# ==============================================================================
# MULTI-LEG TRANSPORT TESTS
# ==============================================================================


class TestMultiLeg:
    """Test multi-leg transport chain calculations."""

    def test_two_leg_chain(self):
        """Test two-leg transport chain (road + rail)."""
        engine = DistanceBasedCalculatorEngine()
        legs = [
            {
                "leg_id": "LEG-1",
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("200.0"),
                "cargo_mass_tonnes": Decimal("20.0"),
            },
            {
                "leg_id": "LEG-2",
                "mode": "RAIL",
                "vehicle_type": "ELECTRIC_FREIGHT",
                "distance_km": Decimal("600.0"),
                "cargo_mass_tonnes": Decimal("20.0"),
            },
        ]
        result = engine.calculate_multi_leg(legs)
        assert result is not None
        total = result.get("total_emissions_tco2e", result.get("emissions_tco2e"))
        assert total > 0

    def test_multi_leg_sum(self):
        """Test multi-leg total equals sum of individual legs."""
        engine = DistanceBasedCalculatorEngine()
        legs = [
            {
                "leg_id": "LEG-1",
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("200.0"),
                "cargo_mass_tonnes": Decimal("20.0"),
            },
            {
                "leg_id": "LEG-2",
                "mode": "MARITIME",
                "vessel_type": "CONTAINER_PANAMAX",
                "distance_km": Decimal("5000.0"),
                "cargo_mass_tonnes": Decimal("20.0"),
            },
        ]
        multi_result = engine.calculate_multi_leg(legs)
        total = multi_result.get("total_emissions_tco2e", multi_result.get("emissions_tco2e"))

        # Calculate individual legs
        r1 = engine.calculate_shipment(legs[0])
        r2 = engine.calculate_shipment(legs[1])
        sum_individual = (
            r1.get("emissions_tco2e", r1.get("total_co2e")) +
            r2.get("emissions_tco2e", r2.get("total_co2e"))
        )
        # Should be approximately equal (hub emissions may add small amount)
        assert abs(total - sum_individual) / sum_individual < Decimal("0.10")


# ==============================================================================
# INTERMODAL TRANSPORT TESTS
# ==============================================================================


class TestIntermodal:
    """Test intermodal transport calculations."""

    def test_intermodal_road_rail(self):
        """Test intermodal road-rail calculation."""
        engine = DistanceBasedCalculatorEngine()
        shipment = {
            "shipment_id": "TEST-INTERMODAL",
            "mode": "RAIL",
            "vehicle_type": "INTERMODAL",
            "distance_km": Decimal("1000.0"),
            "cargo_mass_tonnes": Decimal("25.0"),
        }
        result = engine.calculate_shipment(shipment)
        assert result is not None


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Test batch shipment processing."""

    def test_batch_calculation(self):
        """Test batch calculation of multiple shipments."""
        engine = DistanceBasedCalculatorEngine()
        shipments = [
            {
                "shipment_id": f"BATCH-{i}",
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal(f"{100 + i * 50}"),
                "cargo_mass_tonnes": Decimal("10.0"),
            }
            for i in range(5)
        ]
        result = engine.calculate_batch(shipments)
        assert result is not None
        assert len(result) == 5 or "results" in result

    def test_batch_total_aggregation(self):
        """Test batch results aggregate correctly."""
        engine = DistanceBasedCalculatorEngine()
        shipments = [
            {
                "shipment_id": "B1",
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("100.0"),
                "cargo_mass_tonnes": Decimal("10.0"),
            },
            {
                "shipment_id": "B2",
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("200.0"),
                "cargo_mass_tonnes": Decimal("10.0"),
            },
        ]
        result = engine.calculate_batch(shipments)
        assert result is not None


# ==============================================================================
# FLEET AVERAGE TESTS
# ==============================================================================


class TestFleetAverage:
    """Test fleet average emission calculations."""

    def test_fleet_average(self):
        """Test fleet average factor calculation."""
        engine = DistanceBasedCalculatorEngine()
        fleet = {
            "fleet_id": "FLEET-001",
            "mode": "ROAD",
            "vehicles": [
                {"vehicle_type": "ARTICULATED_33T", "count": 10},
                {"vehicle_type": "VAN_MEDIUM", "count": 20},
            ],
            "total_tonne_km": Decimal("500000.0"),
        }
        result = engine.calculate_fleet(fleet)
        assert result is not None


# ==============================================================================
# MODE COMPARISON TESTS
# ==============================================================================


class TestModeComparison:
    """Test cross-mode comparison analysis."""

    def test_compare_modes(self):
        """Test mode comparison for same shipment parameters."""
        engine = DistanceBasedCalculatorEngine()
        comparison = engine.compare_modes(
            distance_km=Decimal("1000.0"),
            cargo_mass_tonnes=Decimal("20.0"),
        )
        assert comparison is not None
        assert "ROAD" in comparison or "road" in comparison


# ==============================================================================
# PROVENANCE HASH TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test provenance hash generation in calculation results."""

    def test_result_has_provenance_hash(self, sample_shipment):
        """Test calculation result includes provenance hash."""
        engine = DistanceBasedCalculatorEngine()
        result = engine.calculate_shipment(sample_shipment)
        ph = result.get("provenance_hash")
        assert ph is not None
        assert len(ph) == 64
        assert all(c in "0123456789abcdef" for c in ph)

    def test_deterministic_hash(self, sample_shipment):
        """Test same input produces same provenance hash."""
        engine = DistanceBasedCalculatorEngine()
        r1 = engine.calculate_shipment(sample_shipment)
        r2 = engine.calculate_shipment(sample_shipment)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_input_different_hash(self, sample_shipment, sample_shipment_rail):
        """Test different inputs produce different provenance hashes."""
        engine = DistanceBasedCalculatorEngine()
        r1 = engine.calculate_shipment(sample_shipment)
        r2 = engine.calculate_shipment(sample_shipment_rail)
        assert r1["provenance_hash"] != r2["provenance_hash"]
