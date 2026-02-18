# -*- coding: utf-8 -*-
"""
Unit tests for Mobile Combustion Data Models - AGENT-MRV-003

Tests all 16 enumerations, 16 data models, and 2 constant dictionaries
defined in greenlang.mobile_combustion.models.

Target: 142+ tests
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from greenlang.mobile_combustion.models import (
    # Constants
    BIOFUEL_FRACTIONS,
    GWP_VALUES,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_TRIPS_PER_BATCH,
    MAX_VEHICLES_PER_REGISTRATION,
    VERSION,
    # Enums
    VehicleCategory,
    VehicleType,
    FuelType,
    EmissionGas,
    CalculationMethod,
    CalculationTier,
    EmissionFactorSource,
    GWPSource,
    DistanceUnit,
    FuelEconomyUnit,
    EmissionControlTechnology,
    VehicleStatus,
    TripStatus,
    ComplianceStatus,
    ReportingPeriod,
    UnitType,
    # Data models
    VehicleTypeInfo,
    FuelTypeInfo,
    EmissionFactorRecord,
    VehicleRegistration,
    TripRecord,
    CalculationInput,
    GasEmission,
    CalculationResult,
    BatchCalculationInput,
    BatchCalculationResponse,
    FleetAggregation,
    UncertaintyResult,
    ComplianceCheckResult,
    MobileCombustionInput,
    MobileCombustionOutput,
    AuditEntry,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# =========================================================================
# Enum Tests
# =========================================================================


class TestVehicleCategoryEnum:
    """VehicleCategory has exactly 5 members."""

    @pytest.mark.parametrize("member", ["ON_ROAD", "OFF_ROAD", "MARINE", "AVIATION", "RAIL"])
    def test_member_exists(self, member: str) -> None:
        assert VehicleCategory(member).value == member

    def test_member_count(self) -> None:
        assert len(VehicleCategory) == 5

    def test_invalid_member_raises(self) -> None:
        with pytest.raises(ValueError):
            VehicleCategory("SPACE")


class TestVehicleTypeEnum:
    """VehicleType has exactly 24 members."""

    ALL_MEMBERS = [
        "PASSENGER_CAR_GASOLINE", "PASSENGER_CAR_DIESEL",
        "PASSENGER_CAR_HYBRID", "PASSENGER_CAR_PHEV",
        "LIGHT_DUTY_TRUCK_GASOLINE", "LIGHT_DUTY_TRUCK_DIESEL",
        "MEDIUM_DUTY_TRUCK_GASOLINE", "MEDIUM_DUTY_TRUCK_DIESEL",
        "HEAVY_DUTY_TRUCK", "BUS_DIESEL", "BUS_CNG",
        "MOTORCYCLE", "VAN_LCV",
        "CONSTRUCTION_EQUIPMENT", "AGRICULTURAL_EQUIPMENT",
        "INDUSTRIAL_EQUIPMENT", "MINING_EQUIPMENT", "FORKLIFT",
        "INLAND_VESSEL", "COASTAL_VESSEL", "OCEAN_VESSEL",
        "CORPORATE_JET", "HELICOPTER", "TURBOPROP",
        "DIESEL_LOCOMOTIVE",
    ]

    @pytest.mark.parametrize("member", ALL_MEMBERS)
    def test_member_exists(self, member: str) -> None:
        assert VehicleType(member).value == member

    def test_member_count(self) -> None:
        assert len(VehicleType) == 25

    def test_str_enum_subclass(self) -> None:
        assert isinstance(VehicleType.PASSENGER_CAR_GASOLINE, str)


class TestFuelTypeEnum:
    """FuelType has exactly 16 members."""

    ALL_MEMBERS = [
        "GASOLINE", "DIESEL", "BIODIESEL_B5", "BIODIESEL_B20",
        "BIODIESEL_B100", "ETHANOL_E10", "ETHANOL_E85",
        "CNG", "LNG", "LPG", "PROPANE",
        "JET_FUEL_A", "AVGAS", "MARINE_DIESEL_OIL",
        "HEAVY_FUEL_OIL", "SUSTAINABLE_AVIATION_FUEL",
    ]

    @pytest.mark.parametrize("member", ALL_MEMBERS)
    def test_member_exists(self, member: str) -> None:
        assert FuelType(member).value == member

    def test_member_count(self) -> None:
        assert len(FuelType) == 16


class TestEmissionGasEnum:
    """EmissionGas has exactly 3 members."""

    @pytest.mark.parametrize("member", ["CO2", "CH4", "N2O"])
    def test_member_exists(self, member: str) -> None:
        assert EmissionGas(member).value == member

    def test_member_count(self) -> None:
        assert len(EmissionGas) == 3


class TestCalculationMethodEnum:
    """CalculationMethod has exactly 3 members."""

    @pytest.mark.parametrize("member", ["FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED"])
    def test_member_exists(self, member: str) -> None:
        assert CalculationMethod(member).value == member

    def test_member_count(self) -> None:
        assert len(CalculationMethod) == 3


class TestCalculationTierEnum:
    """CalculationTier has exactly 3 members."""

    @pytest.mark.parametrize("member", ["TIER_1", "TIER_2", "TIER_3"])
    def test_member_exists(self, member: str) -> None:
        assert CalculationTier(member).value == member

    def test_member_count(self) -> None:
        assert len(CalculationTier) == 3


class TestEmissionFactorSourceEnum:
    """EmissionFactorSource has exactly 5 members."""

    @pytest.mark.parametrize("member", ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"])
    def test_member_exists(self, member: str) -> None:
        assert EmissionFactorSource(member).value == member

    def test_member_count(self) -> None:
        assert len(EmissionFactorSource) == 5


class TestGWPSourceEnum:
    """GWPSource has exactly 4 members."""

    @pytest.mark.parametrize("member", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_member_exists(self, member: str) -> None:
        assert GWPSource(member).value == member

    def test_member_count(self) -> None:
        assert len(GWPSource) == 4


class TestDistanceUnitEnum:
    """DistanceUnit has exactly 3 members."""

    @pytest.mark.parametrize("member", ["KM", "MILES", "NAUTICAL_MILES"])
    def test_member_exists(self, member: str) -> None:
        assert DistanceUnit(member).value == member

    def test_member_count(self) -> None:
        assert len(DistanceUnit) == 3


class TestFuelEconomyUnitEnum:
    """FuelEconomyUnit has exactly 4 members."""

    @pytest.mark.parametrize("member", ["L_PER_100KM", "MPG_US", "MPG_UK", "KM_PER_L"])
    def test_member_exists(self, member: str) -> None:
        assert FuelEconomyUnit(member).value == member

    def test_member_count(self) -> None:
        assert len(FuelEconomyUnit) == 4


class TestEmissionControlTechnologyEnum:
    """EmissionControlTechnology has exactly 14 members."""

    ALL_MEMBERS = [
        "UNCONTROLLED", "OXIDATION_CATALYST", "THREE_WAY_CATALYST",
        "ADVANCED_CATALYST", "EURO_1", "EURO_2", "EURO_3",
        "EURO_4", "EURO_5", "EURO_6",
        "TIER_1_EPA", "TIER_2_EPA", "TIER_3_EPA", "TIER_4_EPA",
    ]

    @pytest.mark.parametrize("member", ALL_MEMBERS)
    def test_member_exists(self, member: str) -> None:
        assert EmissionControlTechnology(member).value == member

    def test_member_count(self) -> None:
        assert len(EmissionControlTechnology) == 14


class TestVehicleStatusEnum:
    """VehicleStatus has exactly 4 members."""

    @pytest.mark.parametrize("member", ["ACTIVE", "INACTIVE", "DISPOSED", "MAINTENANCE"])
    def test_member_exists(self, member: str) -> None:
        assert VehicleStatus(member).value == member

    def test_member_count(self) -> None:
        assert len(VehicleStatus) == 4


class TestTripStatusEnum:
    """TripStatus has exactly 4 members."""

    @pytest.mark.parametrize("member", ["PLANNED", "IN_PROGRESS", "COMPLETED", "CANCELLED"])
    def test_member_exists(self, member: str) -> None:
        assert TripStatus(member).value == member

    def test_member_count(self) -> None:
        assert len(TripStatus) == 4


class TestComplianceStatusEnum:
    """ComplianceStatus has exactly 4 members."""

    @pytest.mark.parametrize("member", ["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW", "EXEMPT"])
    def test_member_exists(self, member: str) -> None:
        assert ComplianceStatus(member).value == member

    def test_member_count(self) -> None:
        assert len(ComplianceStatus) == 4


class TestReportingPeriodEnum:
    """ReportingPeriod has exactly 3 members."""

    @pytest.mark.parametrize("member", ["MONTHLY", "QUARTERLY", "ANNUAL"])
    def test_member_exists(self, member: str) -> None:
        assert ReportingPeriod(member).value == member

    def test_member_count(self) -> None:
        assert len(ReportingPeriod) == 3


class TestUnitTypeEnum:
    """UnitType has exactly 7 members."""

    @pytest.mark.parametrize("member", [
        "LITERS", "GALLONS", "CUBIC_METERS", "KG", "TONNES", "KWH", "GJ",
    ])
    def test_member_exists(self, member: str) -> None:
        assert UnitType(member).value == member

    def test_member_count(self) -> None:
        assert len(UnitType) == 7


# =========================================================================
# Constants Tests
# =========================================================================


class TestGWPValuesConstant:
    """GWP_VALUES covers all 4 AR editions with CO2, CH4, N2O."""

    @pytest.mark.parametrize("ar", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_ar_edition_present(self, ar: str) -> None:
        assert ar in GWP_VALUES

    @pytest.mark.parametrize("ar", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_ar_edition_has_three_gases(self, ar: str) -> None:
        gases = GWP_VALUES[ar]
        assert set(gases.keys()) == {"CO2", "CH4", "N2O"}

    def test_co2_gwp_always_one(self) -> None:
        for ar_key, gases in GWP_VALUES.items():
            assert gases["CO2"] == 1.0, f"{ar_key} CO2 GWP should be 1.0"

    def test_ar6_ch4_value(self) -> None:
        assert GWP_VALUES["AR6"]["CH4"] == pytest.approx(27.3)

    def test_ar6_n2o_value(self) -> None:
        assert GWP_VALUES["AR6"]["N2O"] == pytest.approx(273.0)

    def test_ar6_20yr_ch4_value(self) -> None:
        assert GWP_VALUES["AR6_20YR"]["CH4"] == pytest.approx(81.2)

    def test_ar4_ch4_value(self) -> None:
        assert GWP_VALUES["AR4"]["CH4"] == pytest.approx(25.0)

    def test_ar5_n2o_value(self) -> None:
        assert GWP_VALUES["AR5"]["N2O"] == pytest.approx(265.0)


class TestBiofuelFractionsConstant:
    """BIOFUEL_FRACTIONS covers 6 biofuel types."""

    @pytest.mark.parametrize("fuel,expected_fraction", [
        ("ETHANOL_E10", 0.10),
        ("ETHANOL_E85", 0.85),
        ("BIODIESEL_B5", 0.05),
        ("BIODIESEL_B20", 0.20),
        ("BIODIESEL_B100", 1.00),
        ("SUSTAINABLE_AVIATION_FUEL", 0.50),
    ])
    def test_fuel_fraction(self, fuel: str, expected_fraction: float) -> None:
        assert BIOFUEL_FRACTIONS[fuel] == pytest.approx(expected_fraction)

    def test_fraction_count(self) -> None:
        assert len(BIOFUEL_FRACTIONS) == 6

    def test_all_fractions_between_zero_and_one(self) -> None:
        for fuel, fraction in BIOFUEL_FRACTIONS.items():
            assert 0.0 <= fraction <= 1.0, f"{fuel} fraction out of range"


# =========================================================================
# Data Model Tests
# =========================================================================


class TestVehicleTypeInfoModel:
    """VehicleTypeInfo Pydantic model tests."""

    def test_valid_construction(self) -> None:
        info = VehicleTypeInfo(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            category=VehicleCategory.ON_ROAD,
            description="Gasoline-powered passenger car",
            default_fuel_type=FuelType.GASOLINE,
        )
        assert info.vehicle_type == VehicleType.PASSENGER_CAR_GASOLINE
        assert info.category == VehicleCategory.ON_ROAD

    def test_default_ef_source(self) -> None:
        info = VehicleTypeInfo(
            vehicle_type=VehicleType.HEAVY_DUTY_TRUCK,
            category=VehicleCategory.ON_ROAD,
            description="Heavy-duty truck",
            default_fuel_type=FuelType.DIESEL,
        )
        assert info.default_ef_source == EmissionFactorSource.EPA

    def test_optional_fuel_economy(self) -> None:
        info = VehicleTypeInfo(
            vehicle_type=VehicleType.MOTORCYCLE,
            category=VehicleCategory.ON_ROAD,
            description="Motorcycle",
            default_fuel_type=FuelType.GASOLINE,
            typical_fuel_economy=4.5,
        )
        assert info.typical_fuel_economy == pytest.approx(4.5)

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValidationError):
            VehicleTypeInfo(
                vehicle_type=VehicleType.MOTORCYCLE,
                category=VehicleCategory.ON_ROAD,
                description="",
                default_fuel_type=FuelType.GASOLINE,
            )


class TestFuelTypeInfoModel:
    """FuelTypeInfo Pydantic model tests."""

    def test_valid_construction(self) -> None:
        info = FuelTypeInfo(
            fuel_type=FuelType.GASOLINE,
            description="Motor gasoline",
            density_kg_per_l=0.755,
            heating_value_gj_per_l=0.0342,
        )
        assert info.fuel_type == FuelType.GASOLINE
        assert info.biofuel_fraction == 0.0

    def test_biofuel_flag_auto_corrected(self) -> None:
        info = FuelTypeInfo(
            fuel_type=FuelType.ETHANOL_E10,
            description="E10 gasoline blend",
            biofuel_fraction=0.10,
            is_biofuel_blend=False,
        )
        assert info.is_biofuel_blend is True

    def test_zero_biofuel_fraction_not_blend(self) -> None:
        info = FuelTypeInfo(
            fuel_type=FuelType.DIESEL,
            description="Petroleum diesel",
            biofuel_fraction=0.0,
            is_biofuel_blend=False,
        )
        assert info.is_biofuel_blend is False

    def test_negative_density_raises(self) -> None:
        with pytest.raises(ValidationError):
            FuelTypeInfo(
                fuel_type=FuelType.GASOLINE,
                description="Bad density",
                density_kg_per_l=-0.5,
            )


class TestEmissionFactorRecordModel:
    """EmissionFactorRecord Pydantic model tests."""

    def test_valid_construction(self) -> None:
        rec = EmissionFactorRecord(
            fuel_type=FuelType.GASOLINE,
            gas=EmissionGas.CO2,
            value=2.31,
            unit="kg CO2/litre",
        )
        assert rec.value == pytest.approx(2.31)
        assert rec.factor_id.startswith("ef_")

    def test_model_year_range_validation(self) -> None:
        with pytest.raises(ValidationError, match="model_year_end"):
            EmissionFactorRecord(
                fuel_type=FuelType.DIESEL,
                gas=EmissionGas.CO2,
                value=2.68,
                unit="kg CO2/litre",
                model_year_start=2020,
                model_year_end=2015,
            )

    def test_expiry_before_effective_raises(self) -> None:
        now = _utcnow()
        with pytest.raises(ValidationError, match="expiry_date"):
            EmissionFactorRecord(
                fuel_type=FuelType.DIESEL,
                gas=EmissionGas.CO2,
                value=2.68,
                unit="kg CO2/litre",
                effective_date=now,
                expiry_date=now - timedelta(days=1),
            )

    def test_zero_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmissionFactorRecord(
                fuel_type=FuelType.GASOLINE,
                gas=EmissionGas.CO2,
                value=0,
                unit="kg CO2/litre",
            )

    def test_optional_vehicle_type(self) -> None:
        rec = EmissionFactorRecord(
            fuel_type=FuelType.GASOLINE,
            gas=EmissionGas.CO2,
            value=2.31,
            unit="kg CO2/litre",
        )
        assert rec.vehicle_type is None


class TestVehicleRegistrationModel:
    """VehicleRegistration Pydantic model tests."""

    def test_valid_construction(self, sample_vehicle_registration: Dict[str, Any]) -> None:
        reg = VehicleRegistration(**sample_vehicle_registration)
        assert reg.vehicle_id == "veh_test_001"
        assert reg.make == "Toyota"
        assert reg.model_year == 2024

    def test_default_vehicle_id_generated(self) -> None:
        reg = VehicleRegistration(
            make="Honda",
            model="Civic",
            model_year=2023,
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
        )
        assert reg.vehicle_id.startswith("veh_")

    def test_default_status_active(self) -> None:
        reg = VehicleRegistration(
            make="Ford",
            model="F-150",
            model_year=2025,
            vehicle_type=VehicleType.LIGHT_DUTY_TRUCK_GASOLINE,
            fuel_type=FuelType.GASOLINE,
        )
        assert reg.status == VehicleStatus.ACTIVE

    def test_disposal_before_registration_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError, match="disposal_date"):
            VehicleRegistration(
                make="Tesla",
                model="Model 3",
                model_year=2024,
                vehicle_type=VehicleType.PASSENGER_CAR_HYBRID,
                fuel_type=FuelType.GASOLINE,
                registration_date=utcnow,
                disposal_date=utcnow - timedelta(days=1),
            )

    def test_vin_max_length_17(self) -> None:
        with pytest.raises(ValidationError):
            VehicleRegistration(
                make="Toyota",
                model="Camry",
                model_year=2024,
                vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
                fuel_type=FuelType.GASOLINE,
                vin="A" * 18,
            )

    def test_model_year_range(self) -> None:
        with pytest.raises(ValidationError):
            VehicleRegistration(
                make="Vintage",
                model="Antique",
                model_year=1800,
                vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
                fuel_type=FuelType.GASOLINE,
            )


class TestTripRecordModel:
    """TripRecord Pydantic model tests."""

    def test_valid_construction(self, sample_trip_record: Dict[str, Any]) -> None:
        trip = TripRecord(**sample_trip_record)
        assert trip.trip_id == "trip_test_001"
        assert trip.distance_value == pytest.approx(245.7)

    def test_default_trip_id_generated(self, utcnow: datetime) -> None:
        trip = TripRecord(
            vehicle_id="veh_001",
            start_time=utcnow,
            end_time=utcnow + timedelta(hours=1),
        )
        assert trip.trip_id.startswith("trip_")

    def test_end_before_start_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError, match="end_time"):
            TripRecord(
                vehicle_id="veh_001",
                start_time=utcnow,
                end_time=utcnow - timedelta(hours=1),
            )

    def test_default_status_completed(self, utcnow: datetime) -> None:
        trip = TripRecord(
            vehicle_id="veh_001",
            start_time=utcnow,
            end_time=utcnow + timedelta(hours=1),
        )
        assert trip.status == TripStatus.COMPLETED

    def test_negative_distance_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError):
            TripRecord(
                vehicle_id="veh_001",
                distance_value=-10.0,
                start_time=utcnow,
                end_time=utcnow + timedelta(hours=1),
            )

    def test_negative_cargo_weight_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError):
            TripRecord(
                vehicle_id="veh_001",
                start_time=utcnow,
                end_time=utcnow + timedelta(hours=1),
                cargo_weight_kg=-50.0,
            )


class TestCalculationInputModel:
    """CalculationInput Pydantic model tests."""

    def test_valid_construction(self, sample_calculation_input: Dict[str, Any]) -> None:
        ci = CalculationInput(**sample_calculation_input)
        assert ci.vehicle_type == VehicleType.PASSENGER_CAR_GASOLINE
        assert ci.quantity == pytest.approx(50.0)

    def test_default_method_fuel_based(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.HEAVY_DUTY_TRUCK,
            fuel_type=FuelType.DIESEL,
            quantity=100.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=30),
        )
        assert ci.calculation_method == CalculationMethod.FUEL_BASED

    def test_period_end_before_start_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError, match="period_end"):
            CalculationInput(
                vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
                fuel_type=FuelType.GASOLINE,
                quantity=50.0,
                unit=UnitType.LITERS,
                period_start=utcnow,
                period_end=utcnow - timedelta(days=1),
            )

    def test_zero_quantity_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError):
            CalculationInput(
                vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
                fuel_type=FuelType.GASOLINE,
                quantity=0,
                unit=UnitType.LITERS,
                period_start=utcnow,
                period_end=utcnow + timedelta(days=1),
            )

    def test_custom_emission_factors(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
            quantity=100.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=30),
            custom_emission_factor_co2=2.35,
            custom_emission_factor_ch4=0.01,
            custom_emission_factor_n2o=0.005,
        )
        assert ci.custom_emission_factor_co2 == pytest.approx(2.35)
        assert ci.custom_emission_factor_ch4 == pytest.approx(0.01)
        assert ci.custom_emission_factor_n2o == pytest.approx(0.005)


class TestGasEmissionModel:
    """GasEmission Pydantic model tests."""

    def test_valid_construction(self) -> None:
        ge = GasEmission(
            gas=EmissionGas.CO2,
            emissions_kg=1500.0,
            emissions_tco2e=1.5,
            emission_factor_value=2.31,
            emission_factor_unit="kg CO2/litre",
            emission_factor_source="EPA",
            gwp_applied=1.0,
        )
        assert ge.emissions_kg == pytest.approx(1500.0)

    def test_zero_emissions_allowed(self) -> None:
        ge = GasEmission(
            gas=EmissionGas.CH4,
            emissions_kg=0.0,
            emissions_tco2e=0.0,
            emission_factor_value=0.001,
            emission_factor_unit="g CH4/km",
            emission_factor_source="IPCC",
            gwp_applied=27.3,
        )
        assert ge.emissions_kg == 0.0

    def test_negative_emissions_raises(self) -> None:
        with pytest.raises(ValidationError):
            GasEmission(
                gas=EmissionGas.N2O,
                emissions_kg=-10.0,
                emissions_tco2e=0.0,
                emission_factor_value=0.01,
                emission_factor_unit="g N2O/km",
                emission_factor_source="DEFRA",
                gwp_applied=273.0,
            )


class TestCalculationResultModel:
    """CalculationResult Pydantic model tests."""

    def test_valid_construction(self) -> None:
        cr = CalculationResult(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
            calculation_method=CalculationMethod.FUEL_BASED,
            tier_used=CalculationTier.TIER_1,
            total_co2e_kg=1500.0,
            total_co2e_tonnes=1.5,
        )
        assert cr.calculation_id.startswith("calc_")
        assert cr.total_co2e_tonnes == pytest.approx(1.5)

    def test_default_biogenic_zero(self) -> None:
        cr = CalculationResult(
            vehicle_type=VehicleType.HEAVY_DUTY_TRUCK,
            fuel_type=FuelType.DIESEL,
            calculation_method=CalculationMethod.FUEL_BASED,
            tier_used=CalculationTier.TIER_1,
            total_co2e_kg=5000.0,
            total_co2e_tonnes=5.0,
        )
        assert cr.biogenic_co2_kg == 0.0
        assert cr.biogenic_co2_tonnes == 0.0

    def test_default_gwp_source_ar6(self) -> None:
        cr = CalculationResult(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
            calculation_method=CalculationMethod.FUEL_BASED,
            tier_used=CalculationTier.TIER_1,
            total_co2e_kg=100.0,
            total_co2e_tonnes=0.1,
        )
        assert cr.gwp_source_used == GWPSource.AR6

    def test_serialization_round_trip(self) -> None:
        cr = CalculationResult(
            vehicle_type=VehicleType.MOTORCYCLE,
            fuel_type=FuelType.GASOLINE,
            calculation_method=CalculationMethod.DISTANCE_BASED,
            tier_used=CalculationTier.TIER_2,
            total_co2e_kg=50.0,
            total_co2e_tonnes=0.05,
        )
        d = cr.model_dump()
        cr2 = CalculationResult(**d)
        assert cr2.total_co2e_kg == cr.total_co2e_kg


class TestBatchCalculationInputModel:
    """BatchCalculationInput Pydantic model tests."""

    def test_valid_construction(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
            quantity=50.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=30),
        )
        batch = BatchCalculationInput(calculations=[ci])
        assert len(batch.calculations) == 1
        assert batch.gwp_source == GWPSource.AR6

    def test_empty_calculations_raises(self) -> None:
        with pytest.raises(ValidationError):
            BatchCalculationInput(calculations=[])

    def test_default_include_biogenic(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.HEAVY_DUTY_TRUCK,
            fuel_type=FuelType.DIESEL,
            quantity=200.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=1),
        )
        batch = BatchCalculationInput(calculations=[ci])
        assert batch.include_biogenic is True


class TestBatchCalculationResponseModel:
    """BatchCalculationResponse Pydantic model tests."""

    def test_valid_construction(self) -> None:
        resp = BatchCalculationResponse(success=True)
        assert resp.success is True
        assert resp.calculation_count == 0
        assert resp.failed_count == 0

    def test_default_totals_zero(self) -> None:
        resp = BatchCalculationResponse(success=True)
        assert resp.total_co2e_tonnes == 0.0
        assert resp.total_co2_tonnes == 0.0
        assert resp.total_ch4_tco2e == 0.0
        assert resp.total_n2o_tco2e == 0.0
        assert resp.total_biogenic_co2_tonnes == 0.0


class TestFleetAggregationModel:
    """FleetAggregation Pydantic model tests."""

    def test_valid_construction(self, utcnow: datetime) -> None:
        agg = FleetAggregation(
            fleet_id="fleet_001",
            reporting_period_type=ReportingPeriod.ANNUAL,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=365),
        )
        assert agg.fleet_id == "fleet_001"
        assert agg.total_co2e_tonnes == 0.0

    def test_period_end_before_start_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError, match="period_end"):
            FleetAggregation(
                fleet_id="fleet_001",
                reporting_period_type=ReportingPeriod.MONTHLY,
                period_start=utcnow,
                period_end=utcnow - timedelta(days=1),
            )

    def test_empty_fleet_id_raises(self, utcnow: datetime) -> None:
        with pytest.raises(ValidationError):
            FleetAggregation(
                fleet_id="",
                reporting_period_type=ReportingPeriod.ANNUAL,
                period_start=utcnow,
                period_end=utcnow + timedelta(days=365),
            )


class TestUncertaintyResultModel:
    """UncertaintyResult Pydantic model tests."""

    def test_valid_construction(self) -> None:
        ur = UncertaintyResult(
            mean_co2e=10.5,
            std_dev=1.2,
            coefficient_of_variation=0.114,
            iterations=5000,
            tier=CalculationTier.TIER_1,
        )
        assert ur.mean_co2e == pytest.approx(10.5)
        assert ur.iterations == 5000

    def test_confidence_intervals(self) -> None:
        ur = UncertaintyResult(
            mean_co2e=10.0,
            std_dev=1.0,
            coefficient_of_variation=0.1,
            iterations=10000,
            tier=CalculationTier.TIER_2,
            confidence_interval_90=(8.35, 11.65),
            confidence_interval_95=(8.04, 11.96),
            confidence_interval_99=(7.42, 12.58),
        )
        assert ur.confidence_interval_90 is not None
        assert ur.confidence_interval_90[0] < ur.confidence_interval_90[1]

    def test_data_quality_score_range(self) -> None:
        ur = UncertaintyResult(
            mean_co2e=5.0,
            std_dev=0.5,
            coefficient_of_variation=0.1,
            iterations=1000,
            tier=CalculationTier.TIER_1,
            data_quality_score=3.5,
        )
        assert 1.0 <= ur.data_quality_score <= 5.0

    def test_data_quality_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            UncertaintyResult(
                mean_co2e=5.0,
                std_dev=0.5,
                coefficient_of_variation=0.1,
                iterations=1000,
                tier=CalculationTier.TIER_1,
                data_quality_score=0.5,
            )


class TestComplianceCheckResultModel:
    """ComplianceCheckResult Pydantic model tests."""

    def test_valid_construction(self) -> None:
        ccr = ComplianceCheckResult(
            framework="GHG_PROTOCOL",
            status=ComplianceStatus.COMPLIANT,
            requirements_met=15,
            requirements_total=15,
        )
        assert ccr.status == ComplianceStatus.COMPLIANT
        assert ccr.checked_by == "GL-MRV-SCOPE1-003"

    def test_default_empty_lists(self) -> None:
        ccr = ComplianceCheckResult(
            framework="ISO_14064",
            status=ComplianceStatus.NEEDS_REVIEW,
        )
        assert ccr.findings == []
        assert ccr.recommendations == []
        assert ccr.evidence_references == []

    def test_empty_framework_raises(self) -> None:
        with pytest.raises(ValidationError):
            ComplianceCheckResult(
                framework="",
                status=ComplianceStatus.COMPLIANT,
            )


class TestMobileCombustionInputModel:
    """MobileCombustionInput Pydantic model tests."""

    def test_valid_construction(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.PASSENGER_CAR_GASOLINE,
            fuel_type=FuelType.GASOLINE,
            quantity=50.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=30),
        )
        mci = MobileCombustionInput(inputs=[ci])
        assert len(mci.inputs) == 1
        assert mci.include_biogenic is True
        assert mci.include_uncertainty is True

    def test_empty_inputs_raises(self) -> None:
        with pytest.raises(ValidationError):
            MobileCombustionInput(inputs=[])

    def test_default_gwp_source(self, utcnow: datetime) -> None:
        ci = CalculationInput(
            vehicle_type=VehicleType.BUS_DIESEL,
            fuel_type=FuelType.DIESEL,
            quantity=300.0,
            unit=UnitType.LITERS,
            period_start=utcnow,
            period_end=utcnow + timedelta(days=1),
        )
        mci = MobileCombustionInput(inputs=[ci])
        assert mci.gwp_source == GWPSource.AR6


class TestMobileCombustionOutputModel:
    """MobileCombustionOutput Pydantic model tests."""

    def test_valid_construction(self) -> None:
        mco = MobileCombustionOutput(success=True)
        assert mco.success is True
        assert mco.agent_id == "GL-MRV-SCOPE1-003"
        assert mco.agent_version == VERSION

    def test_default_empty_collections(self) -> None:
        mco = MobileCombustionOutput(success=False)
        assert mco.results == []
        assert mco.compliance_checks == []
        assert mco.errors == []

    def test_version_matches_module_constant(self) -> None:
        mco = MobileCombustionOutput(success=True)
        assert mco.agent_version == "1.0.0"


class TestAuditEntryModel:
    """AuditEntry Pydantic model tests."""

    def test_valid_construction(self) -> None:
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=0,
            step_name="fuel_quantity_normalization",
        )
        assert ae.entry_id.startswith("audit_")
        assert ae.calculation_id == "calc_001"

    def test_default_empty_hashes(self) -> None:
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=1,
            step_name="emission_factor_lookup",
        )
        assert ae.input_hash == ""
        assert ae.output_hash == ""
        assert ae.provenance_hash == ""

    def test_empty_step_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            AuditEntry(
                calculation_id="calc_001",
                step_number=0,
                step_name="",
            )

    def test_negative_step_number_raises(self) -> None:
        with pytest.raises(ValidationError):
            AuditEntry(
                calculation_id="calc_001",
                step_number=-1,
                step_name="test_step",
            )


# =========================================================================
# Module-level constants tests
# =========================================================================


class TestModuleConstants:
    """Test module-level constant values."""

    def test_version(self) -> None:
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self) -> None:
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self) -> None:
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self) -> None:
        assert MAX_TRACE_STEPS == 200

    def test_max_vehicles_per_registration(self) -> None:
        assert MAX_VEHICLES_PER_REGISTRATION == 10_000

    def test_max_trips_per_batch(self) -> None:
        assert MAX_TRIPS_PER_BATCH == 5_000
