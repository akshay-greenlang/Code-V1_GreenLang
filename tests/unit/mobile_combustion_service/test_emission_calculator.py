# -*- coding: utf-8 -*-
"""
Unit tests for EmissionCalculatorEngine (Engine 2) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 48+ test functions covering:
- Fuel-based, distance-based, spend-based calculation methods
- Batch processing, biofuel handling, GWP application
- Vehicle age degradation, load factor adjustments
- Unit conversions, provenance tracking, error handling

Author: GreenLang QA Team
"""

from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch

import pytest

from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine
from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def db():
    """Create a VehicleDatabaseEngine instance."""
    return VehicleDatabaseEngine()


@pytest.fixture
def calc(db):
    """Create an EmissionCalculatorEngine with default config."""
    return EmissionCalculatorEngine(vehicle_database=db)


@pytest.fixture
def calc_ar5(db):
    """Create an EmissionCalculatorEngine using AR5 GWP."""
    return EmissionCalculatorEngine(
        vehicle_database=db,
        config={"default_gwp_source": "AR5"},
    )


# ===========================================================================
# TestFuelBased
# ===========================================================================


class TestFuelBased:
    """Test fuel-based calculation method."""

    def test_basic_diesel_calculation(self, calc):
        """500L diesel in heavy-duty truck produces expected CO2e."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("500"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "fuel_based"
        assert result["total_co2e_kg"] > Decimal("0")
        assert result["total_co2e_tonnes"] > Decimal("0")
        assert result["co2_fossil_kg"] > Decimal("0")
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_co2_calculation_accuracy(self, calc):
        """Verify CO2 calculation: 100L diesel * 2.68 kg/L = 268 kg CO2."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        # CO2 = 100 * 2.68 * 1.0 (oxidation) = 268 kg
        # With zero biofuel fraction, all is fossil
        assert result["co2_fossil_kg"] == Decimal("268.00000000")

    def test_gallon_unit_conversion(self, calc):
        """Fuel in gallons is properly converted to liters."""
        result = calc.calculate_fuel_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            fuel_consumed=Decimal("10"),
            fuel_unit="gallons",
        )
        assert result["status"] == "SUCCESS"
        # 10 gallons * 3.78541 = 37.8541 liters
        assert result["fuel_consumed_liters"] == pytest.approx(
            Decimal("37.85410000"), abs=Decimal("0.001")
        )

    def test_kg_unit_conversion(self, calc):
        """Fuel in kg is properly converted to liters."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("83.2"),
            fuel_unit="kg",
        )
        assert result["status"] == "SUCCESS"
        # 83.2 kg / 0.832 kg/L = 100 L
        assert result["fuel_consumed_liters"] == Decimal("100.00000000")

    def test_energy_unit_kwh(self, calc):
        """Fuel in kWh is converted via heating value."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="kwh",
        )
        assert result["status"] == "SUCCESS"
        assert result["fuel_consumed_liters"] > Decimal("0")

    def test_unsupported_unit_returns_failed(self, calc):
        """Unsupported fuel unit returns FAILED status."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="bushels",
        )
        assert result["status"] == "FAILED"
        assert "error_message" in result

    def test_invalid_vehicle_returns_failed(self, calc):
        """Invalid vehicle type returns FAILED status."""
        result = calc.calculate_fuel_based(
            vehicle_type="FLYING_CAR",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert result["status"] == "FAILED"

    def test_biofuel_separation(self, calc):
        """Biofuel blend separates fossil and biogenic CO2."""
        result = calc.calculate_fuel_based(
            vehicle_type="PASSENGER_CAR_DIESEL",
            fuel_type="BIODIESEL_B20",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        assert result["co2_biogenic_kg"] > Decimal("0")
        assert result["co2_fossil_kg"] > Decimal("0")
        # Biogenic should be 20% of total
        total_co2 = result["co2_fossil_kg"] + result["co2_biogenic_kg"]
        biogenic_ratio = result["co2_biogenic_kg"] / total_co2
        assert biogenic_ratio == pytest.approx(Decimal("0.20"), abs=Decimal("0.01"))

    def test_b100_all_biogenic(self, calc):
        """BIODIESEL_B100 has 100% biogenic CO2."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="BIODIESEL_B100",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        assert result["co2_fossil_kg"] == Decimal("0E-8") or result["co2_fossil_kg"] == Decimal("0.00000000")

    def test_gas_emissions_breakdown(self, calc):
        """Result includes per-gas emission breakdown."""
        result = calc.calculate_fuel_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            fuel_consumed=Decimal("50"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        gases = result["gas_emissions"]
        assert len(gases) >= 3
        gas_names = {g["gas"] for g in gases}
        assert "CO2" in gas_names
        assert "CH4" in gas_names
        assert "N2O" in gas_names

    def test_calculation_trace_present(self, calc):
        """Result includes non-empty calculation trace."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert len(result["calculation_trace"]) > 0

    def test_processing_time_present(self, calc):
        """Result includes processing time."""
        result = calc.calculate_fuel_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_consumed=Decimal("100"),
            fuel_unit="liters",
        )
        assert result["processing_time_ms"] >= 0


# ===========================================================================
# TestDistanceBased
# ===========================================================================


class TestDistanceBased:
    """Test distance-based calculation method."""

    def test_basic_distance_calculation(self, calc):
        """100km in gasoline car produces expected emissions."""
        result = calc.calculate_distance_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            distance_km=Decimal("100"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "distance_based"
        assert result["total_co2e_kg"] > Decimal("0")
        assert "emission_intensity_g_co2e_per_km" in result

    def test_direct_distance_factor(self, calc):
        """use_distance_factor=True uses g CO2e/km directly."""
        result = calc.calculate_distance_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            distance_km=Decimal("1000"),
            use_distance_factor=True,
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "distance_based_direct"
        # 1000 km * 192 g/km = 192000 g = 192 kg
        assert result["total_co2e_kg"] == Decimal("192.00000000")

    def test_vehicle_age_increases_emissions(self, calc):
        """Older vehicle produces more emissions than newer one."""
        young = calc.calculate_distance_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            distance_km=Decimal("10000"),
            vehicle_age_years=0,
        )
        old = calc.calculate_distance_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            distance_km=Decimal("10000"),
            vehicle_age_years=20,
        )
        assert young["status"] == "SUCCESS"
        assert old["status"] == "SUCCESS"
        assert old["total_co2e_kg"] > young["total_co2e_kg"]

    def test_load_factor_affects_emissions(self, calc):
        """Higher load factor increases emissions for freight vehicles."""
        light = calc.calculate_distance_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            distance_km=Decimal("100"),
            load_factor=Decimal("0.3"),
        )
        heavy = calc.calculate_distance_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            distance_km=Decimal("100"),
            load_factor=Decimal("0.9"),
        )
        assert light["status"] == "SUCCESS"
        assert heavy["status"] == "SUCCESS"
        assert heavy["total_co2e_kg"] > light["total_co2e_kg"]

    def test_custom_fuel_economy_override(self, calc):
        """Custom fuel economy overrides default value."""
        result = calc.calculate_distance_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            distance_km=Decimal("100"),
            fuel_economy_km_per_l=Decimal("25.0"),
        )
        assert result["status"] == "SUCCESS"
        assert result["fuel_economy_km_per_l"] == Decimal("25.0")

    def test_no_fuel_economy_for_offroad_returns_failed(self, calc):
        """Off-road vehicles without fuel economy data return error."""
        result = calc.calculate_distance_based(
            vehicle_type="CONSTRUCTION_EQUIPMENT",
            fuel_type="DIESEL",
            distance_km=Decimal("100"),
        )
        assert result["status"] == "FAILED"


# ===========================================================================
# TestSpendBased
# ===========================================================================


class TestSpendBased:
    """Test spend-based calculation method."""

    def test_basic_spend_calculation(self, calc):
        """$200 at $2/L diesel produces expected emissions."""
        result = calc.calculate_spend_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_expenditure=Decimal("200"),
            fuel_price_per_unit=Decimal("2.0"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "spend_based"
        # $200 / $2/L = 100L diesel
        assert result["total_co2e_kg"] > Decimal("0")

    def test_zero_expenditure(self, calc):
        """Zero expenditure produces zero emissions."""
        result = calc.calculate_spend_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_expenditure=Decimal("0"),
            fuel_price_per_unit=Decimal("2.0"),
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_kg"] == Decimal("0E-8") or result["total_co2e_kg"] == Decimal("0.00000000")

    def test_zero_price_returns_failed(self, calc):
        """Zero fuel price returns FAILED."""
        result = calc.calculate_spend_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_expenditure=Decimal("100"),
            fuel_price_per_unit=Decimal("0"),
        )
        assert result["status"] == "FAILED"

    def test_cost_intensity_metric(self, calc):
        """Spend-based result includes cost intensity metric."""
        result = calc.calculate_spend_based(
            vehicle_type="HEAVY_DUTY_TRUCK",
            fuel_type="DIESEL",
            fuel_expenditure=Decimal("500"),
            fuel_price_per_unit=Decimal("1.50"),
        )
        assert result["status"] == "SUCCESS"
        assert "emission_intensity_kg_co2e_per_currency" in result


# ===========================================================================
# TestBatch
# ===========================================================================


class TestBatch:
    """Test batch calculation processing."""

    def test_batch_two_items(self, calc):
        """Batch processes two items and aggregates."""
        inputs = [
            {
                "method": "fuel_based",
                "vehicle_type": "HEAVY_DUTY_TRUCK",
                "fuel_type": "DIESEL",
                "fuel_consumed": Decimal("100"),
                "fuel_unit": "liters",
            },
            {
                "method": "fuel_based",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_consumed": Decimal("50"),
                "fuel_unit": "liters",
            },
        ]
        result = calc.calculate_batch(inputs)
        assert result["input_count"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert result["status"] == "SUCCESS"
        assert result["totals"]["total_co2e_kg"] > Decimal("0")
        assert len(result["results"]) == 2

    def test_batch_partial_failure(self, calc):
        """Batch with one invalid input reports PARTIAL status."""
        inputs = [
            {
                "method": "fuel_based",
                "vehicle_type": "HEAVY_DUTY_TRUCK",
                "fuel_type": "DIESEL",
                "fuel_consumed": Decimal("100"),
                "fuel_unit": "liters",
            },
            {
                "method": "fuel_based",
                "vehicle_type": "FLYING_CAR",
                "fuel_type": "DIESEL",
                "fuel_consumed": Decimal("100"),
                "fuel_unit": "liters",
            },
        ]
        result = calc.calculate_batch(inputs)
        assert result["status"] == "PARTIAL"
        assert result["success_count"] == 1
        assert result["failure_count"] == 1

    def test_batch_empty(self, calc):
        """Empty batch returns zero totals."""
        result = calc.calculate_batch([])
        assert result["input_count"] == 0
        assert result["success_count"] == 0
        assert result["totals"]["total_co2e_kg"] == Decimal("0.00000000")

    def test_batch_provenance_hash(self, calc):
        """Batch result includes provenance hash."""
        inputs = [
            {
                "method": "fuel_based",
                "vehicle_type": "MOTORCYCLE",
                "fuel_type": "GASOLINE",
                "fuel_consumed": Decimal("10"),
                "fuel_unit": "liters",
            },
        ]
        result = calc.calculate_batch(inputs)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestGWP
# ===========================================================================


class TestGWP:
    """Test GWP application and configuration."""

    def test_apply_gwp_ch4_ar6(self, calc):
        """apply_gwp for CH4 with AR6 returns mass * 29.8."""
        result = calc.apply_gwp("CH4", Decimal("1.0"), "AR6")
        assert result == Decimal("29.80000000")

    def test_apply_gwp_n2o_ar5(self, calc):
        """apply_gwp for N2O with AR5 returns mass * 265."""
        result = calc.apply_gwp("N2O", Decimal("1.0"), "AR5")
        assert result == Decimal("265.00000000")

    def test_apply_gwp_invalid_gas_raises(self, calc):
        """apply_gwp with invalid gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            calc.apply_gwp("SF6", Decimal("1.0"))

    def test_default_gwp_ar5(self, calc_ar5):
        """Calculator configured with AR5 uses AR5 GWP."""
        result = calc_ar5.calculate_fuel_based(
            vehicle_type="PASSENGER_CAR_GASOLINE",
            fuel_type="GASOLINE",
            fuel_consumed=Decimal("50"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        assert result["gwp_source"] == "AR5"


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test provenance tracking and calculation metadata."""

    def test_provenance_hash_is_sha256(self, calc):
        """Provenance hash is a valid 64-char hex SHA-256."""
        result = calc.calculate_fuel_based(
            vehicle_type="MOTORCYCLE",
            fuel_type="GASOLINE",
            fuel_consumed=Decimal("5"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        h = result["provenance_hash"]
        assert len(h) == 64
        int(h, 16)  # Should not raise if valid hex

    def test_dispatch_method_fuel_based(self, calc):
        """calculate() dispatches to fuel_based correctly."""
        result = calc.calculate(
            method="fuel_based",
            vehicle_type="MOTORCYCLE",
            fuel_type="GASOLINE",
            fuel_consumed=Decimal("5"),
            fuel_unit="liters",
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "fuel_based"

    def test_dispatch_method_invalid_raises(self, calc):
        """calculate() with invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calculation method"):
            calc.calculate(method="magic")

    def test_biogenic_fraction_lookup(self, calc):
        """get_biogenic_fraction delegates to vehicle database."""
        frac = calc.get_biogenic_fraction("BIODIESEL_B20")
        assert frac == Decimal("0.20")

    def test_adjust_for_vehicle_age_public(self, calc):
        """Public adjust_for_vehicle_age matches internal behavior."""
        base = Decimal("12.5")
        result = calc.adjust_for_vehicle_age(base, 10)
        # Age 10 -> bracket 8_12 -> degradation 1.07
        expected = (base / Decimal("1.07")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_adjust_for_load_factor_public(self, calc):
        """Public adjust_for_load_factor applies correctly."""
        base = Decimal("2.8")
        result = calc.adjust_for_load_factor(base, "HEAVY_DUTY_TRUCK", Decimal("0.9"))
        # For heavy truck: default load = 0.65, sensitivity = 0.8
        # delta = (0.9 - 0.65) * 0.8 = 0.2, adj = 1 + 0.2 = 1.2
        # result = 2.8 / 1.2
        expected = (Decimal("2.8") / Decimal("1.2")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_statistics(self, calc):
        """get_statistics returns expected structure."""
        stats = calc.get_statistics()
        assert stats["default_gwp_source"] == "AR6"
        assert "fuel_based" in stats["supported_methods"]
        assert "distance_based" in stats["supported_methods"]
        assert "spend_based" in stats["supported_methods"]
        assert stats["decimal_precision"] == 8
