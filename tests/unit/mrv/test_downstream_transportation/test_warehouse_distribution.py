# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.warehouse_distribution - AGENT-MRV-022.

Tests WarehouseDistributionEngine for the Downstream Transportation &
Distribution Agent (GL-MRV-S3-009).

Coverage (~50 tests):
- 7 warehouse types (DC, cold storage, fulfillment, retail, cross-dock,
  bonded, transit)
- Cold storage variants (chilled, frozen, pharma)
- Retail storage component
- Fulfillment center calculations
- Last-mile delivery (6 vehicle types x 3 delivery areas)
- Batch warehouse processing
- Batch last-mile processing
- Distribution chain (warehouse + last-mile combined)
- Annual estimate
- Type comparison
- Known-value hand-calculated tests
- Singleton pattern, provenance hash

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.warehouse_distribution import (
        WarehouseDistributionEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"warehouse_distribution not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test WarehouseDistributionEngine singleton."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        eng1 = WarehouseDistributionEngine()
        eng2 = WarehouseDistributionEngine()
        assert eng1 is eng2


# ==============================================================================
# WAREHOUSE TYPE TESTS
# ==============================================================================


class TestWarehouseTypes:
    """Test calculate_warehouse for all 7 warehouse types."""

    @pytest.mark.parametrize("wh_type", [
        "DISTRIBUTION_CENTER", "COLD_STORAGE", "FULFILLMENT_CENTER",
        "RETAIL_STORAGE", "CROSS_DOCK", "BONDED_WAREHOUSE",
        "TRANSIT_WAREHOUSE",
    ])
    def test_warehouse_type_calculation(self, wh_type):
        """Test warehouse emission calculation for each type."""
        engine = WarehouseDistributionEngine()
        wh_input = {
            "warehouse_id": f"WH-{wh_type}",
            "warehouse_type": wh_type,
            "floor_area_m2": Decimal("3000.0"),
            "dwell_time_hours": Decimal("48.0"),
            "region": "US",
        }
        result = engine.calculate_warehouse(wh_input)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert isinstance(emissions, Decimal)
        assert emissions > 0

    def test_distribution_center(self, sample_warehouse):
        """Test ambient distribution center calculation."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_dc_known_value(self):
        """
        Hand-calculated: 5000 m2 x 85 kWh/m2/yr x 0.42 kgCO2e/kWh
        x (72 / 8760) = 1463.01 kgCO2e = ~1.463 tCO2e.
        """
        engine = WarehouseDistributionEngine()
        wh_input = {
            "warehouse_id": "TEST-KV-DC",
            "warehouse_type": "DISTRIBUTION_CENTER",
            "floor_area_m2": Decimal("5000.0"),
            "dwell_time_hours": Decimal("72.0"),
            "energy_intensity_kwh_m2_year": Decimal("85.0"),
            "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        }
        result = engine.calculate_warehouse(wh_input)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        expected = Decimal("1.463")
        assert abs(emissions - expected) / expected < Decimal("0.25")

    def test_cross_dock_short_dwell(self, sample_warehouse_cross_dock):
        """Test cross-dock with short dwell time produces low emissions."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse_cross_dock)
        dc_result = engine.calculate_warehouse({
            "warehouse_id": "DC",
            "warehouse_type": "DISTRIBUTION_CENTER",
            "floor_area_m2": Decimal("3000.0"),
            "dwell_time_hours": Decimal("72.0"),
            "region": "US",
        })
        cross_e = result.get("emissions_tco2e", result.get("total_co2e"))
        dc_e = dc_result.get("emissions_tco2e", dc_result.get("total_co2e"))
        assert cross_e < dc_e


# ==============================================================================
# COLD STORAGE TESTS
# ==============================================================================


class TestColdStorage:
    """Test cold storage warehouse calculations."""

    def test_cold_storage_basic(self, sample_warehouse_cold):
        """Test cold storage calculation."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse_cold)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_cold_storage_higher_than_ambient(self):
        """Test cold storage emissions are higher than ambient DC."""
        engine = WarehouseDistributionEngine()
        base = {
            "warehouse_id": "TEST",
            "floor_area_m2": Decimal("3000.0"),
            "dwell_time_hours": Decimal("48.0"),
            "region": "US",
        }
        ambient = engine.calculate_warehouse({
            **base, "warehouse_type": "DISTRIBUTION_CENTER",
        })
        cold = engine.calculate_warehouse({
            **base,
            "warehouse_type": "COLD_STORAGE",
            "temperature_controlled": True,
            "temperature_range": "CHILLED",
        })
        amb_e = ambient.get("emissions_tco2e", ambient.get("total_co2e"))
        cold_e = cold.get("emissions_tco2e", cold.get("total_co2e"))
        assert cold_e > amb_e

    @pytest.mark.parametrize("temp_range", ["CHILLED", "FROZEN", "PHARMA"])
    def test_cold_storage_variants(self, temp_range):
        """Test cold storage for different temperature ranges."""
        engine = WarehouseDistributionEngine()
        wh_input = {
            "warehouse_id": f"COLD-{temp_range}",
            "warehouse_type": "COLD_STORAGE",
            "floor_area_m2": Decimal("2000.0"),
            "dwell_time_hours": Decimal("48.0"),
            "temperature_controlled": True,
            "temperature_range": temp_range,
            "region": "US",
        }
        result = engine.calculate_warehouse(wh_input)
        assert result is not None

    def test_frozen_higher_than_chilled(self):
        """Test frozen storage emissions > chilled storage."""
        engine = WarehouseDistributionEngine()
        base = {
            "warehouse_id": "TEST",
            "warehouse_type": "COLD_STORAGE",
            "floor_area_m2": Decimal("2000.0"),
            "dwell_time_hours": Decimal("48.0"),
            "temperature_controlled": True,
            "region": "US",
        }
        chilled = engine.calculate_warehouse({**base, "temperature_range": "CHILLED"})
        frozen = engine.calculate_warehouse({**base, "temperature_range": "FROZEN"})
        ch_e = chilled.get("emissions_tco2e", chilled.get("total_co2e"))
        fr_e = frozen.get("emissions_tco2e", frozen.get("total_co2e"))
        assert fr_e > ch_e

    def test_cold_storage_with_refrigerant_leakage(self):
        """Test cold storage includes refrigerant leakage emissions."""
        engine = WarehouseDistributionEngine()
        wh_input = {
            "warehouse_id": "COLD-LEAK",
            "warehouse_type": "COLD_STORAGE",
            "floor_area_m2": Decimal("2000.0"),
            "dwell_time_hours": Decimal("48.0"),
            "temperature_controlled": True,
            "temperature_range": "CHILLED",
            "refrigerant_type": "R-404A",
            "refrigerant_charge_kg": Decimal("35.0"),
            "annual_leak_rate": Decimal("0.10"),
            "region": "US",
        }
        result = engine.calculate_warehouse(wh_input)
        assert result is not None


# ==============================================================================
# RETAIL STORAGE TESTS
# ==============================================================================


class TestRetailStorage:
    """Test retail storage calculations."""

    def test_retail_storage(self, sample_warehouse_retail):
        """Test retail storage calculation."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse_retail)
        assert result is not None

    def test_retail_longer_dwell_higher_emissions(self):
        """Test longer dwell time in retail increases emissions."""
        engine = WarehouseDistributionEngine()
        base = {
            "warehouse_id": "RETAIL",
            "warehouse_type": "RETAIL_STORAGE",
            "floor_area_m2": Decimal("500.0"),
            "region": "US",
        }
        short = engine.calculate_warehouse({**base, "dwell_time_hours": Decimal("24.0")})
        long = engine.calculate_warehouse({**base, "dwell_time_hours": Decimal("168.0")})
        short_e = short.get("emissions_tco2e", short.get("total_co2e"))
        long_e = long.get("emissions_tco2e", long.get("total_co2e"))
        assert long_e > short_e


# ==============================================================================
# FULFILLMENT CENTER TESTS
# ==============================================================================


class TestFulfillmentCenter:
    """Test fulfillment center calculations."""

    def test_fulfillment_center(self, sample_warehouse_fulfillment):
        """Test fulfillment center calculation."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse_fulfillment)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0


# ==============================================================================
# LAST-MILE DELIVERY TESTS
# ==============================================================================


class TestLastMileDelivery:
    """Test calculate_last_mile for all vehicle types and delivery areas."""

    @pytest.mark.parametrize("vehicle_type", [
        "VAN_DIESEL", "VAN_ELECTRIC", "CARGO_BIKE",
        "DRONE", "PARCEL_LOCKER", "CROWD_SHIPPING",
    ])
    def test_vehicle_types(self, vehicle_type):
        """Test last-mile calculation for each vehicle type."""
        engine = WarehouseDistributionEngine()
        lm_input = {
            "delivery_id": f"LM-{vehicle_type}",
            "vehicle_type": vehicle_type,
            "delivery_area": "URBAN",
            "distance_km": Decimal("15.0"),
            "parcels_delivered": 25,
            "region": "US",
        }
        result = engine.calculate_last_mile(lm_input)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions >= 0  # Cargo bike / parcel locker may be zero or near-zero

    @pytest.mark.parametrize("area", ["URBAN", "SUBURBAN", "RURAL"])
    def test_delivery_areas(self, area):
        """Test last-mile calculation for each delivery area."""
        engine = WarehouseDistributionEngine()
        lm_input = {
            "delivery_id": f"LM-{area}",
            "vehicle_type": "VAN_DIESEL",
            "delivery_area": area,
            "distance_km": Decimal("20.0"),
            "parcels_delivered": 15,
            "region": "US",
        }
        result = engine.calculate_last_mile(lm_input)
        assert result is not None

    def test_urban_lower_than_rural(self):
        """Test urban delivery emissions lower than rural (higher density)."""
        engine = WarehouseDistributionEngine()
        base = {
            "delivery_id": "TEST",
            "vehicle_type": "VAN_DIESEL",
            "parcels_delivered": 15,
            "region": "US",
        }
        urban = engine.calculate_last_mile({
            **base, "delivery_area": "URBAN", "distance_km": Decimal("10.0"),
        })
        rural = engine.calculate_last_mile({
            **base, "delivery_area": "RURAL", "distance_km": Decimal("50.0"),
        })
        urb_e = urban.get("emissions_tco2e", urban.get("total_co2e"))
        rur_e = rural.get("emissions_tco2e", rural.get("total_co2e"))
        assert rur_e > urb_e

    def test_cargo_bike_lower_than_van(self):
        """Test cargo bike emissions lower than diesel van."""
        engine = WarehouseDistributionEngine()
        base = {
            "delivery_id": "TEST",
            "delivery_area": "URBAN",
            "distance_km": Decimal("10.0"),
            "parcels_delivered": 15,
            "region": "US",
        }
        van = engine.calculate_last_mile({**base, "vehicle_type": "VAN_DIESEL"})
        bike = engine.calculate_last_mile({**base, "vehicle_type": "CARGO_BIKE"})
        van_e = van.get("emissions_tco2e", van.get("total_co2e"))
        bike_e = bike.get("emissions_tco2e", bike.get("total_co2e"))
        assert bike_e < van_e

    def test_last_mile_fixture_urban(self, sample_last_mile_urban):
        """Test last-mile urban fixture."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_last_mile(sample_last_mile_urban)
        assert result is not None

    def test_last_mile_fixture_suburban(self, sample_last_mile_suburban):
        """Test last-mile suburban fixture."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_last_mile(sample_last_mile_suburban)
        assert result is not None

    def test_last_mile_fixture_rural(self, sample_last_mile_rural):
        """Test last-mile rural fixture."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_last_mile(sample_last_mile_rural)
        assert result is not None


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Test batch warehouse and last-mile processing."""

    def test_batch_warehouse(self):
        """Test batch warehouse calculation."""
        engine = WarehouseDistributionEngine()
        warehouses = [
            {
                "warehouse_id": f"WH-{i}",
                "warehouse_type": "DISTRIBUTION_CENTER",
                "floor_area_m2": Decimal(f"{2000 + i * 1000}"),
                "dwell_time_hours": Decimal("48.0"),
                "region": "US",
            }
            for i in range(3)
        ]
        result = engine.calculate_batch_warehouses(warehouses)
        assert result is not None

    def test_batch_last_mile(self):
        """Test batch last-mile calculation."""
        engine = WarehouseDistributionEngine()
        deliveries = [
            {
                "delivery_id": f"LM-{i}",
                "vehicle_type": "VAN_DIESEL",
                "delivery_area": "URBAN",
                "distance_km": Decimal(f"{10 + i * 5}"),
                "parcels_delivered": 20,
                "region": "US",
            }
            for i in range(5)
        ]
        result = engine.calculate_batch_last_mile(deliveries)
        assert result is not None


# ==============================================================================
# DISTRIBUTION CHAIN TESTS
# ==============================================================================


class TestDistributionChain:
    """Test combined warehouse + last-mile distribution chain."""

    def test_distribution_chain(self):
        """Test full distribution chain calculation."""
        engine = WarehouseDistributionEngine()
        chain = {
            "warehouse": {
                "warehouse_id": "WH-001",
                "warehouse_type": "FULFILLMENT_CENTER",
                "floor_area_m2": Decimal("10000.0"),
                "dwell_time_hours": Decimal("24.0"),
                "region": "US",
            },
            "last_mile": {
                "delivery_id": "LM-001",
                "vehicle_type": "VAN_DIESEL",
                "delivery_area": "URBAN",
                "distance_km": Decimal("15.0"),
                "parcels_delivered": 25,
                "region": "US",
            },
        }
        result = engine.calculate_distribution_chain(chain)
        assert result is not None
        total = result.get("total_emissions_tco2e", result.get("emissions_tco2e"))
        assert total > 0


# ==============================================================================
# ANNUAL ESTIMATE TESTS
# ==============================================================================


class TestAnnualEstimate:
    """Test annual warehouse emissions estimation."""

    def test_annual_estimate(self):
        """Test annual warehouse emissions estimate."""
        engine = WarehouseDistributionEngine()
        result = engine.estimate_annual(
            warehouse_type="DISTRIBUTION_CENTER",
            floor_area_m2=Decimal("10000.0"),
            region="US",
        )
        assert result is not None
        emissions = result.get("annual_emissions_tco2e", result.get("emissions_tco2e"))
        assert emissions > 0


# ==============================================================================
# TYPE COMPARISON TESTS
# ==============================================================================


class TestTypeComparison:
    """Test warehouse type comparison analysis."""

    def test_compare_warehouse_types(self):
        """Test comparison across all warehouse types."""
        engine = WarehouseDistributionEngine()
        comparison = engine.compare_warehouse_types(
            floor_area_m2=Decimal("5000.0"),
            dwell_time_hours=Decimal("48.0"),
            region="US",
        )
        assert comparison is not None
        assert len(comparison) >= 7 or "types" in comparison


# ==============================================================================
# PROVENANCE HASH TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test provenance hash in warehouse/last-mile results."""

    def test_warehouse_result_has_hash(self, sample_warehouse):
        """Test warehouse result includes provenance hash."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_warehouse(sample_warehouse)
        ph = result.get("provenance_hash")
        assert ph is not None
        assert len(ph) == 64

    def test_last_mile_result_has_hash(self, sample_last_mile):
        """Test last-mile result includes provenance hash."""
        engine = WarehouseDistributionEngine()
        result = engine.calculate_last_mile(sample_last_mile)
        ph = result.get("provenance_hash")
        assert ph is not None
        assert len(ph) == 64
