# -*- coding: utf-8 -*-
"""
Unit tests for Stationary Combustion Agent data models - AGENT-MRV-001.

Tests all 13 enumerations and 12 Pydantic data models including field
validation, defaults, edge cases, and serialization. 80+ tests total.

AGENT-MRV-001: Stationary Combustion Agent (GL-MRV-SCOPE1-001)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.stationary_combustion.models import (
    # Constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_EFFICIENCY_COEFFICIENTS,
    DEFAULT_OXIDATION_FACTOR,
    GWP_VALUES,
    # Enumerations
    FuelCategory,
    FuelType,
    EmissionGas,
    GWPSource,
    EFSource,
    CalculationTier,
    EquipmentType,
    HeatingValueBasis,
    ControlApproach,
    CalculationStatus,
    ReportingPeriod,
    RegulatoryFramework,
    UnitType,
    # Data models
    EmissionFactor,
    FuelProperties,
    EquipmentProfile,
    CombustionInput,
    GasEmission,
    CalculationResult,
    BatchCalculationRequest,
    BatchCalculationResponse,
    UncertaintyResult,
    FacilityAggregation,
    AuditEntry,
    ComplianceMapping,
)


# =============================================================================
# Test Enumerations
# =============================================================================


class TestFuelTypeEnum:
    """Verify all 24 FuelType enum members exist with correct values."""

    EXPECTED_FUEL_TYPES = [
        ("NATURAL_GAS", "natural_gas"),
        ("DIESEL", "diesel"),
        ("GASOLINE", "gasoline"),
        ("LPG", "lpg"),
        ("PROPANE", "propane"),
        ("KEROSENE", "kerosene"),
        ("JET_FUEL", "jet_fuel"),
        ("FUEL_OIL_2", "fuel_oil_2"),
        ("FUEL_OIL_6", "fuel_oil_6"),
        ("COAL_BITUMINOUS", "coal_bituminous"),
        ("COAL_ANTHRACITE", "coal_anthracite"),
        ("COAL_SUB_BITUMINOUS", "coal_sub_bituminous"),
        ("COAL_LIGNITE", "coal_lignite"),
        ("PETROLEUM_COKE", "petroleum_coke"),
        ("WOOD", "wood"),
        ("BIOMASS_SOLID", "biomass_solid"),
        ("BIOMASS_LIQUID", "biomass_liquid"),
        ("BIOGAS", "biogas"),
        ("LANDFILL_GAS", "landfill_gas"),
        ("COKE_OVEN_GAS", "coke_oven_gas"),
        ("BLAST_FURNACE_GAS", "blast_furnace_gas"),
        ("PEAT", "peat"),
        ("WASTE_OIL", "waste_oil"),
        ("MSW", "msw"),
    ]

    def test_fuel_type_count(self):
        assert len(FuelType) == 24

    @pytest.mark.parametrize("member_name,expected_value", EXPECTED_FUEL_TYPES)
    def test_fuel_type_member(self, member_name: str, expected_value: str):
        member = FuelType[member_name]
        assert member.value == expected_value

    def test_fuel_type_is_str_enum(self):
        assert isinstance(FuelType.NATURAL_GAS, str)
        assert FuelType.NATURAL_GAS == "natural_gas"


class TestEmissionGasEnum:
    """Verify EmissionGas enum members."""

    def test_co2_value(self):
        assert EmissionGas.CO2.value == "CO2"

    def test_ch4_value(self):
        assert EmissionGas.CH4.value == "CH4"

    def test_n2o_value(self):
        assert EmissionGas.N2O.value == "N2O"

    def test_count(self):
        assert len(EmissionGas) == 3


class TestGWPSourceEnum:
    """Verify GWPSource enum members."""

    def test_ar4_value(self):
        assert GWPSource.AR4.value == "AR4"

    def test_ar5_value(self):
        assert GWPSource.AR5.value == "AR5"

    def test_ar6_value(self):
        assert GWPSource.AR6.value == "AR6"

    def test_count(self):
        assert len(GWPSource) == 3


class TestEFSourceEnum:
    """Verify EFSource enum members."""

    def test_epa_value(self):
        assert EFSource.EPA.value == "EPA"

    def test_ipcc_value(self):
        assert EFSource.IPCC.value == "IPCC"

    def test_defra_value(self):
        assert EFSource.DEFRA.value == "DEFRA"

    def test_eu_ets_value(self):
        assert EFSource.EU_ETS.value == "EU_ETS"

    def test_custom_value(self):
        assert EFSource.CUSTOM.value == "CUSTOM"

    def test_count(self):
        assert len(EFSource) == 5


class TestCalculationTierEnum:
    """Verify CalculationTier enum members."""

    def test_tier_1_value(self):
        assert CalculationTier.TIER_1.value == "TIER_1"

    def test_tier_2_value(self):
        assert CalculationTier.TIER_2.value == "TIER_2"

    def test_tier_3_value(self):
        assert CalculationTier.TIER_3.value == "TIER_3"

    def test_count(self):
        assert len(CalculationTier) == 3


class TestEquipmentTypeEnum:
    """Verify all 13 EquipmentType enum members."""

    EXPECTED_EQUIPMENT_TYPES = [
        ("BOILER_FIRE_TUBE", "boiler_fire_tube"),
        ("BOILER_WATER_TUBE", "boiler_water_tube"),
        ("FURNACE", "furnace"),
        ("PROCESS_HEATER", "process_heater"),
        ("GAS_TURBINE_SIMPLE", "gas_turbine_simple"),
        ("GAS_TURBINE_COMBINED", "gas_turbine_combined"),
        ("RECIPROCATING_ENGINE", "reciprocating_engine"),
        ("KILN", "kiln"),
        ("OVEN", "oven"),
        ("DRYER", "dryer"),
        ("FLARE", "flare"),
        ("INCINERATOR", "incinerator"),
        ("THERMAL_OXIDIZER", "thermal_oxidizer"),
    ]

    def test_count(self):
        assert len(EquipmentType) == 13

    @pytest.mark.parametrize("member_name,expected_value", EXPECTED_EQUIPMENT_TYPES)
    def test_equipment_type_member(self, member_name: str, expected_value: str):
        member = EquipmentType[member_name]
        assert member.value == expected_value


class TestUnitTypeEnum:
    """Verify all 16 UnitType enum members."""

    EXPECTED_UNIT_TYPES = [
        ("LITERS", "liters"),
        ("GALLONS", "gallons"),
        ("CUBIC_METERS", "cubic_meters"),
        ("CUBIC_FEET", "cubic_feet"),
        ("BARRELS", "barrels"),
        ("KG", "kg"),
        ("TONNES", "tonnes"),
        ("LBS", "lbs"),
        ("SHORT_TONS", "short_tons"),
        ("KWH", "kwh"),
        ("MWH", "mwh"),
        ("GJ", "gj"),
        ("MMBTU", "mmbtu"),
        ("THERMS", "therms"),
        ("MCF", "mcf"),
        ("SCF", "scf"),
    ]

    def test_count(self):
        assert len(UnitType) == 16

    @pytest.mark.parametrize("member_name,expected_value", EXPECTED_UNIT_TYPES)
    def test_unit_type_member(self, member_name: str, expected_value: str):
        member = UnitType[member_name]
        assert member.value == expected_value


class TestFuelCategoryEnum:
    """Verify FuelCategory enum members."""

    def test_gaseous(self):
        assert FuelCategory.GASEOUS.value == "gaseous"

    def test_liquid(self):
        assert FuelCategory.LIQUID.value == "liquid"

    def test_solid(self):
        assert FuelCategory.SOLID.value == "solid"

    def test_biomass(self):
        assert FuelCategory.BIOMASS.value == "biomass"

    def test_count(self):
        assert len(FuelCategory) == 4


class TestHeatingValueBasisEnum:
    """Verify HeatingValueBasis enum members."""

    def test_hhv(self):
        assert HeatingValueBasis.HHV.value == "HHV"

    def test_ncv(self):
        assert HeatingValueBasis.NCV.value == "NCV"


class TestControlApproachEnum:
    """Verify ControlApproach enum members."""

    def test_operational(self):
        assert ControlApproach.OPERATIONAL.value == "operational"

    def test_financial(self):
        assert ControlApproach.FINANCIAL.value == "financial"

    def test_equity_share(self):
        assert ControlApproach.EQUITY_SHARE.value == "equity_share"


class TestCalculationStatusEnum:
    """Verify CalculationStatus enum members."""

    def test_pending(self):
        assert CalculationStatus.PENDING.value == "pending"

    def test_running(self):
        assert CalculationStatus.RUNNING.value == "running"

    def test_completed(self):
        assert CalculationStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert CalculationStatus.FAILED.value == "failed"


class TestReportingPeriodEnum:
    """Verify ReportingPeriod enum members."""

    def test_monthly(self):
        assert ReportingPeriod.MONTHLY.value == "monthly"

    def test_quarterly(self):
        assert ReportingPeriod.QUARTERLY.value == "quarterly"

    def test_annual(self):
        assert ReportingPeriod.ANNUAL.value == "annual"


class TestRegulatoryFrameworkEnum:
    """Verify RegulatoryFramework enum members."""

    def test_ghg_protocol(self):
        assert RegulatoryFramework.GHG_PROTOCOL.value == "ghg_protocol"

    def test_iso_14064(self):
        assert RegulatoryFramework.ISO_14064.value == "iso_14064"

    def test_csrd_esrs_e1(self):
        assert RegulatoryFramework.CSRD_ESRS_E1.value == "csrd_esrs_e1"

    def test_epa_40cfr98(self):
        assert RegulatoryFramework.EPA_40CFR98.value == "epa_40cfr98"

    def test_uk_secr(self):
        assert RegulatoryFramework.UK_SECR.value == "uk_secr"

    def test_eu_ets(self):
        assert RegulatoryFramework.EU_ETS.value == "eu_ets"

    def test_count(self):
        assert len(RegulatoryFramework) == 6


# =============================================================================
# Test Constants
# =============================================================================


class TestModelConstants:
    """Verify module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        assert MAX_TRACE_STEPS == 200

    def test_max_efficiency_coefficients(self):
        assert MAX_EFFICIENCY_COEFFICIENTS == 10

    def test_default_oxidation_factor(self):
        assert DEFAULT_OXIDATION_FACTOR == 1.0

    def test_gwp_values_ar4(self):
        assert GWP_VALUES["AR4"]["CO2"] == 1.0
        assert GWP_VALUES["AR4"]["CH4"] == 25.0
        assert GWP_VALUES["AR4"]["N2O"] == 298.0

    def test_gwp_values_ar5(self):
        assert GWP_VALUES["AR5"]["CO2"] == 1.0
        assert GWP_VALUES["AR5"]["CH4"] == 28.0
        assert GWP_VALUES["AR5"]["N2O"] == 265.0

    def test_gwp_values_ar6(self):
        assert GWP_VALUES["AR6"]["CO2"] == 1.0
        assert GWP_VALUES["AR6"]["CH4"] == 27.3
        assert GWP_VALUES["AR6"]["N2O"] == 273.0


# =============================================================================
# Test Data Models
# =============================================================================


class TestEmissionFactor:
    """Test EmissionFactor Pydantic model."""

    def test_create_valid(self):
        ef = EmissionFactor(
            fuel_type=FuelType.NATURAL_GAS,
            gas=EmissionGas.CO2,
            value=53.06,
            unit="kg CO2/mmBtu",
        )
        assert ef.fuel_type == FuelType.NATURAL_GAS
        assert ef.gas == EmissionGas.CO2
        assert ef.value == 53.06
        assert ef.unit == "kg CO2/mmBtu"

    def test_default_source(self):
        ef = EmissionFactor(
            fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
            value=73.96, unit="kg CO2/mmBtu",
        )
        assert ef.source == EFSource.EPA

    def test_default_tier(self):
        ef = EmissionFactor(
            fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
            value=73.96, unit="kg CO2/mmBtu",
        )
        assert ef.tier == CalculationTier.TIER_1

    def test_default_geography(self):
        ef = EmissionFactor(
            fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
            value=73.96, unit="kg CO2/mmBtu",
        )
        assert ef.geography == "GLOBAL"

    def test_factor_id_auto_generated(self):
        ef = EmissionFactor(
            fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
            value=73.96, unit="kg CO2/mmBtu",
        )
        assert ef.factor_id.startswith("ef_")
        assert len(ef.factor_id) > 3

    def test_value_must_be_positive(self):
        with pytest.raises(ValidationError):
            EmissionFactor(
                fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
                value=0.0, unit="kg CO2/mmBtu",
            )

    def test_value_negative_raises(self):
        with pytest.raises(ValidationError):
            EmissionFactor(
                fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
                value=-1.0, unit="kg CO2/mmBtu",
            )

    def test_unit_empty_raises(self):
        with pytest.raises(ValidationError):
            EmissionFactor(
                fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
                value=73.96, unit="",
            )

    def test_expiry_after_effective_validation(self):
        with pytest.raises(ValidationError, match="expiry_date must be after effective_date"):
            EmissionFactor(
                fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
                value=73.96, unit="kg CO2/mmBtu",
                effective_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                expiry_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_optional_fields_default_none(self):
        ef = EmissionFactor(
            fuel_type=FuelType.DIESEL, gas=EmissionGas.CO2,
            value=73.96, unit="kg CO2/mmBtu",
        )
        assert ef.effective_date is None
        assert ef.expiry_date is None
        assert ef.reference is None
        assert ef.notes is None


class TestFuelProperties:
    """Test FuelProperties Pydantic model."""

    def test_create_valid(self):
        fp = FuelProperties(
            fuel_type=FuelType.NATURAL_GAS,
            category=FuelCategory.GASEOUS,
            hhv=1.028,
            hhv_unit="mmBtu/MCF",
            ncv=0.926,
            ncv_unit="mmBtu/MCF",
        )
        assert fp.fuel_type == FuelType.NATURAL_GAS
        assert fp.category == FuelCategory.GASEOUS
        assert fp.hhv == 1.028

    def test_biogenic_default_false(self):
        fp = FuelProperties(
            fuel_type=FuelType.NATURAL_GAS,
            category=FuelCategory.GASEOUS,
            hhv=1.028, hhv_unit="mmBtu/MCF",
            ncv=0.926, ncv_unit="mmBtu/MCF",
        )
        assert fp.is_biogenic is False

    def test_biogenic_true(self):
        fp = FuelProperties(
            fuel_type=FuelType.WOOD,
            category=FuelCategory.BIOMASS,
            hhv=15.4, hhv_unit="GJ/tonne",
            ncv=13.9, ncv_unit="GJ/tonne",
            is_biogenic=True,
        )
        assert fp.is_biogenic is True

    def test_oxidation_factor_default(self):
        fp = FuelProperties(
            fuel_type=FuelType.DIESEL,
            category=FuelCategory.LIQUID,
            hhv=36.2, hhv_unit="GJ/kL",
            ncv=34.0, ncv_unit="GJ/kL",
        )
        assert fp.oxidation_factor == 1.0

    def test_carbon_content_range_validated(self):
        with pytest.raises(ValidationError):
            FuelProperties(
                fuel_type=FuelType.DIESEL,
                category=FuelCategory.LIQUID,
                hhv=36.2, hhv_unit="GJ/kL",
                ncv=34.0, ncv_unit="GJ/kL",
                carbon_content=1.5,  # must be <= 1.0
            )

    def test_density_must_be_positive(self):
        with pytest.raises(ValidationError):
            FuelProperties(
                fuel_type=FuelType.DIESEL,
                category=FuelCategory.LIQUID,
                hhv=36.2, hhv_unit="GJ/kL",
                ncv=34.0, ncv_unit="GJ/kL",
                density=-0.5,
            )


class TestEquipmentProfile:
    """Test EquipmentProfile Pydantic model."""

    def test_create_valid(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.BOILER_WATER_TUBE,
            name="Main Steam Boiler",
        )
        assert ep.equipment_type == EquipmentType.BOILER_WATER_TUBE
        assert ep.name == "Main Steam Boiler"

    def test_equipment_id_auto_generated(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.FURNACE,
            name="Furnace A",
        )
        assert ep.equipment_id.startswith("eq_")

    def test_efficiency_curve_default_empty(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.KILN,
            name="Cement Kiln",
        )
        assert ep.efficiency_curve_coefficients == []

    def test_load_factor_range_valid(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.BOILER_FIRE_TUBE,
            name="Boiler 1",
            load_factor_range=(0.2, 0.9),
        )
        assert ep.load_factor_range == (0.2, 0.9)

    def test_load_factor_range_invalid_low(self):
        with pytest.raises(ValidationError, match="Load factor minimum"):
            EquipmentProfile(
                equipment_type=EquipmentType.BOILER_FIRE_TUBE,
                name="Boiler 1",
                load_factor_range=(-0.1, 0.9),
            )

    def test_load_factor_range_invalid_high(self):
        with pytest.raises(ValidationError, match="Load factor maximum"):
            EquipmentProfile(
                equipment_type=EquipmentType.BOILER_FIRE_TUBE,
                name="Boiler 1",
                load_factor_range=(0.2, 1.5),
            )

    def test_load_factor_range_inverted(self):
        with pytest.raises(ValidationError, match="must be <= maximum"):
            EquipmentProfile(
                equipment_type=EquipmentType.BOILER_FIRE_TUBE,
                name="Boiler 1",
                load_factor_range=(0.9, 0.2),
            )

    def test_maintenance_status_valid_values(self):
        for status in ("good", "fair", "poor"):
            ep = EquipmentProfile(
                equipment_type=EquipmentType.FURNACE,
                name="Test",
                maintenance_status=status,
            )
            assert ep.maintenance_status == status

    def test_maintenance_status_normalized(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.FURNACE,
            name="Test",
            maintenance_status="Good",
        )
        assert ep.maintenance_status == "good"

    def test_maintenance_status_invalid(self):
        with pytest.raises(ValidationError, match="maintenance_status"):
            EquipmentProfile(
                equipment_type=EquipmentType.FURNACE,
                name="Test",
                maintenance_status="excellent",
            )

    def test_name_empty_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.FURNACE,
                name="",
            )

    def test_rated_capacity_must_be_positive(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.FURNACE,
                name="Furnace",
                rated_capacity_mmbtu_hr=-10.0,
            )


class TestCombustionInput:
    """Test CombustionInput Pydantic model."""

    def _make_input(self, **kwargs):
        defaults = dict(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit=UnitType.CUBIC_METERS,
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        defaults.update(kwargs)
        return CombustionInput(**defaults)

    def test_create_valid(self):
        ci = self._make_input()
        assert ci.fuel_type == FuelType.NATURAL_GAS
        assert ci.quantity == 1000.0
        assert ci.unit == UnitType.CUBIC_METERS

    def test_quantity_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._make_input(quantity=0.0)

    def test_quantity_negative_raises(self):
        with pytest.raises(ValidationError):
            self._make_input(quantity=-100.0)

    def test_optional_fields_default_none(self):
        ci = self._make_input()
        assert ci.equipment_id is None
        assert ci.facility_id is None
        assert ci.source_id is None
        assert ci.custom_heating_value is None
        assert ci.custom_emission_factor is None
        assert ci.custom_oxidation_factor is None
        assert ci.tier is None
        assert ci.geography is None

    def test_custom_heating_value_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._make_input(custom_heating_value=-1.0)

    def test_custom_oxidation_factor_range(self):
        ci = self._make_input(custom_oxidation_factor=0.5)
        assert ci.custom_oxidation_factor == 0.5

    def test_custom_oxidation_factor_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._make_input(custom_oxidation_factor=1.1)

    def test_custom_oxidation_factor_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._make_input(custom_oxidation_factor=-0.1)

    def test_period_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="period_end must be after period_start"):
            self._make_input(
                period_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
                period_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_period_end_equal_start_raises(self):
        same_dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        with pytest.raises(ValidationError, match="period_end must be after period_start"):
            self._make_input(period_start=same_dt, period_end=same_dt)

    def test_heating_value_basis_default(self):
        ci = self._make_input()
        assert ci.heating_value_basis == HeatingValueBasis.HHV

    def test_custom_emission_factor_positive(self):
        ci = self._make_input(custom_emission_factor=50.0)
        assert ci.custom_emission_factor == 50.0

    def test_custom_emission_factor_zero_raises(self):
        with pytest.raises(ValidationError):
            self._make_input(custom_emission_factor=0.0)

    def test_tier_override(self):
        ci = self._make_input(tier=CalculationTier.TIER_3)
        assert ci.tier == CalculationTier.TIER_3


class TestGasEmission:
    """Test GasEmission Pydantic model."""

    def test_create_valid(self):
        ge = GasEmission(
            gas=EmissionGas.CO2,
            emissions_kg=1500.0,
            emissions_tco2e=1.5,
            emission_factor_value=53.06,
            emission_factor_unit="kg CO2/mmBtu",
            emission_factor_source="EPA",
            gwp_applied=1.0,
        )
        assert ge.gas == EmissionGas.CO2
        assert ge.emissions_kg == 1500.0
        assert ge.emissions_tco2e == 1.5

    def test_emissions_kg_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            GasEmission(
                gas=EmissionGas.CO2,
                emissions_kg=-1.0,
                emissions_tco2e=0.0,
                emission_factor_value=53.06,
                emission_factor_unit="kg CO2/mmBtu",
                emission_factor_source="EPA",
                gwp_applied=1.0,
            )

    def test_emission_factor_value_must_be_positive(self):
        with pytest.raises(ValidationError):
            GasEmission(
                gas=EmissionGas.CH4,
                emissions_kg=0.0,
                emissions_tco2e=0.0,
                emission_factor_value=0.0,
                emission_factor_unit="kg CH4/mmBtu",
                emission_factor_source="EPA",
                gwp_applied=27.3,
            )

    def test_all_fields_present(self):
        ge = GasEmission(
            gas=EmissionGas.N2O,
            emissions_kg=0.5,
            emissions_tco2e=0.000137,
            emission_factor_value=0.0001,
            emission_factor_unit="kg N2O/mmBtu",
            emission_factor_source="IPCC",
            gwp_applied=273.0,
        )
        assert ge.gas == EmissionGas.N2O
        assert ge.gwp_applied == 273.0
        assert ge.emission_factor_source == "IPCC"


class TestCalculationResult:
    """Test CalculationResult Pydantic model."""

    def _make_result(self, **kwargs):
        defaults = dict(
            fuel_type=FuelType.NATURAL_GAS,
            fuel_quantity=1000.0,
            fuel_unit=UnitType.CUBIC_METERS,
            energy_gj=38.0,
            heating_value_used=0.038,
            heating_value_basis=HeatingValueBasis.HHV,
            oxidation_factor_used=1.0,
            tier_used=CalculationTier.TIER_1,
            total_co2e_kg=2000.0,
            total_co2e_tonnes=2.0,
        )
        defaults.update(kwargs)
        return CalculationResult(**defaults)

    def test_create_valid(self):
        cr = self._make_result()
        assert cr.fuel_type == FuelType.NATURAL_GAS
        assert cr.total_co2e_tonnes == 2.0

    def test_calculation_id_auto_generated(self):
        cr = self._make_result()
        assert cr.calculation_id.startswith("calc_")

    def test_biogenic_defaults_zero(self):
        cr = self._make_result()
        assert cr.biogenic_co2_kg == 0.0
        assert cr.biogenic_co2_tonnes == 0.0

    def test_provenance_hash_default_empty(self):
        cr = self._make_result()
        assert cr.provenance_hash == ""

    def test_calculation_trace_default_empty(self):
        cr = self._make_result()
        assert cr.calculation_trace == []

    def test_timestamp_auto_generated(self):
        cr = self._make_result()
        assert cr.timestamp is not None
        assert cr.timestamp.tzinfo == timezone.utc

    def test_optional_fields_default_none(self):
        cr = self._make_result()
        assert cr.equipment_type is None
        assert cr.regulatory_framework is None
        assert cr.facility_id is None
        assert cr.source_id is None
        assert cr.period_start is None
        assert cr.period_end is None

    def test_all_fields_populated(self):
        cr = self._make_result(
            equipment_type=EquipmentType.BOILER_WATER_TUBE,
            emissions_by_gas=[
                GasEmission(
                    gas=EmissionGas.CO2,
                    emissions_kg=1950.0,
                    emissions_tco2e=1.95,
                    emission_factor_value=53.06,
                    emission_factor_unit="kg CO2/mmBtu",
                    emission_factor_source="EPA",
                    gwp_applied=1.0,
                ),
            ],
            biogenic_co2_kg=100.0,
            biogenic_co2_tonnes=0.1,
            regulatory_framework=RegulatoryFramework.GHG_PROTOCOL,
            provenance_hash="a" * 64,
            calculation_trace=["step1", "step2"],
            facility_id="facility_001",
            source_id="source_001",
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        assert cr.equipment_type == EquipmentType.BOILER_WATER_TUBE
        assert len(cr.emissions_by_gas) == 1
        assert cr.biogenic_co2_kg == 100.0
        assert len(cr.calculation_trace) == 2


class TestBatchCalculationRequest:
    """Test BatchCalculationRequest Pydantic model."""

    def _make_input(self):
        return CombustionInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=500.0,
            unit=UnitType.CUBIC_METERS,
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )

    def test_create_valid(self):
        bcr = BatchCalculationRequest(calculations=[self._make_input()])
        assert len(bcr.calculations) == 1

    def test_min_length_validation(self):
        with pytest.raises(ValidationError):
            BatchCalculationRequest(calculations=[])

    def test_defaults(self):
        bcr = BatchCalculationRequest(calculations=[self._make_input()])
        assert bcr.gwp_source == GWPSource.AR6
        assert bcr.include_biogenic is True
        assert bcr.control_approach == ControlApproach.OPERATIONAL
        assert bcr.organization_id is None
        assert bcr.reporting_period is None


class TestBatchCalculationResponse:
    """Test BatchCalculationResponse Pydantic model."""

    def test_create_valid(self):
        bcr = BatchCalculationResponse(
            success=True,
            total_co2e_tonnes=10.5,
            calculation_count=5,
        )
        assert bcr.success is True
        assert bcr.total_co2e_tonnes == 10.5

    def test_defaults(self):
        bcr = BatchCalculationResponse(success=True)
        assert bcr.results == []
        assert bcr.total_co2e_tonnes == 0.0
        assert bcr.total_co2_tonnes == 0.0
        assert bcr.total_ch4_tonnes == 0.0
        assert bcr.total_n2o_tonnes == 0.0
        assert bcr.total_biogenic_co2_tonnes == 0.0
        assert bcr.emissions_by_fuel == {}
        assert bcr.calculation_count == 0
        assert bcr.failed_count == 0
        assert bcr.processing_time_ms == 0.0
        assert bcr.provenance_hash == ""
        assert bcr.gwp_source == GWPSource.AR6

    def test_aggregated_totals(self):
        bcr = BatchCalculationResponse(
            success=True,
            total_co2e_tonnes=15.0,
            total_co2_tonnes=14.0,
            total_ch4_tonnes=0.5,
            total_n2o_tonnes=0.5,
            total_biogenic_co2_tonnes=2.0,
            emissions_by_fuel={"natural_gas": 10.0, "diesel": 5.0},
            calculation_count=3,
            failed_count=1,
        )
        assert bcr.emissions_by_fuel["natural_gas"] == 10.0
        assert bcr.calculation_count == 3
        assert bcr.failed_count == 1


class TestUncertaintyResult:
    """Test UncertaintyResult Pydantic model."""

    def test_create_valid(self):
        ur = UncertaintyResult(
            mean_co2e=5.0,
            std_dev=0.5,
            coefficient_of_variation=0.1,
            iterations=5000,
            tier=CalculationTier.TIER_1,
        )
        assert ur.mean_co2e == 5.0
        assert ur.std_dev == 0.5
        assert ur.iterations == 5000

    def test_confidence_intervals(self):
        ur = UncertaintyResult(
            mean_co2e=5.0,
            std_dev=0.5,
            coefficient_of_variation=0.1,
            confidence_intervals={
                "90": (4.18, 5.82),
                "95": (4.02, 5.98),
                "99": (3.71, 6.29),
            },
            iterations=5000,
            tier=CalculationTier.TIER_1,
        )
        assert "95" in ur.confidence_intervals
        lower, upper = ur.confidence_intervals["95"]
        assert lower < ur.mean_co2e < upper

    def test_data_quality_score_range(self):
        ur = UncertaintyResult(
            mean_co2e=5.0,
            std_dev=0.5,
            coefficient_of_variation=0.1,
            iterations=5000,
            tier=CalculationTier.TIER_1,
            data_quality_score=3.5,
        )
        assert ur.data_quality_score == 3.5

    def test_data_quality_score_below_1_raises(self):
        with pytest.raises(ValidationError):
            UncertaintyResult(
                mean_co2e=5.0, std_dev=0.5, coefficient_of_variation=0.1,
                iterations=5000, tier=CalculationTier.TIER_1,
                data_quality_score=0.5,
            )

    def test_data_quality_score_above_5_raises(self):
        with pytest.raises(ValidationError):
            UncertaintyResult(
                mean_co2e=5.0, std_dev=0.5, coefficient_of_variation=0.1,
                iterations=5000, tier=CalculationTier.TIER_1,
                data_quality_score=5.5,
            )

    def test_contributions_dict(self):
        ur = UncertaintyResult(
            mean_co2e=5.0, std_dev=0.5, coefficient_of_variation=0.1,
            iterations=5000, tier=CalculationTier.TIER_1,
            contributions={"emission_factor": 0.45, "activity_data": 0.55},
        )
        assert ur.contributions["emission_factor"] == 0.45

    def test_iterations_must_be_positive(self):
        with pytest.raises(ValidationError):
            UncertaintyResult(
                mean_co2e=5.0, std_dev=0.5, coefficient_of_variation=0.1,
                iterations=0, tier=CalculationTier.TIER_1,
            )


class TestFacilityAggregation:
    """Test FacilityAggregation Pydantic model."""

    def _make_aggregation(self, **kwargs):
        defaults = dict(
            facility_id="facility_001",
            reporting_period_type=ReportingPeriod.ANNUAL,
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        defaults.update(kwargs)
        return FacilityAggregation(**defaults)

    def test_create_valid(self):
        fa = self._make_aggregation()
        assert fa.facility_id == "facility_001"

    def test_control_approach_default(self):
        fa = self._make_aggregation()
        assert fa.control_approach == ControlApproach.OPERATIONAL

    def test_all_control_approaches(self):
        for approach in ControlApproach:
            fa = self._make_aggregation(control_approach=approach)
            assert fa.control_approach == approach

    def test_period_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="period_end must be after period_start"):
            self._make_aggregation(
                period_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
                period_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_defaults_zero(self):
        fa = self._make_aggregation()
        assert fa.total_co2e_tonnes == 0.0
        assert fa.total_co2_tonnes == 0.0
        assert fa.total_ch4_tonnes == 0.0
        assert fa.total_n2o_tonnes == 0.0
        assert fa.biogenic_co2_tonnes == 0.0
        assert fa.calculation_count == 0
        assert fa.equipment_count == 0
        assert fa.fuel_types_used == []
        assert fa.provenance_hash == ""


class TestAuditEntry:
    """Test AuditEntry Pydantic model."""

    def test_create_valid(self):
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=1,
            step_name="input_validation",
        )
        assert ae.calculation_id == "calc_001"
        assert ae.step_number == 1
        assert ae.step_name == "input_validation"

    def test_entry_id_auto_generated(self):
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=0,
            step_name="validate",
        )
        assert ae.entry_id.startswith("audit_")

    def test_defaults(self):
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=0,
            step_name="validate",
        )
        assert ae.input_data == {}
        assert ae.output_data == {}
        assert ae.emission_factor_used is None
        assert ae.methodology_reference is None
        assert ae.provenance_hash == ""

    def test_step_fields(self):
        ae = AuditEntry(
            calculation_id="calc_002",
            step_number=3,
            step_name="calculate_gas_emissions",
            input_data={"energy_gj": 38.0, "ef": 53.06},
            output_data={"co2_kg": 2016.28},
            emission_factor_used=53.06,
            methodology_reference="GHG Protocol Ch. 3, Eq. 3.1",
        )
        assert ae.emission_factor_used == 53.06
        assert ae.methodology_reference == "GHG Protocol Ch. 3, Eq. 3.1"
        assert ae.input_data["energy_gj"] == 38.0

    def test_timestamp_auto_generated(self):
        ae = AuditEntry(
            calculation_id="calc_001",
            step_number=0,
            step_name="validate",
        )
        assert ae.timestamp is not None
        assert ae.timestamp.tzinfo == timezone.utc


class TestComplianceMapping:
    """Test ComplianceMapping Pydantic model."""

    def test_create_valid(self):
        cm = ComplianceMapping(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            requirement_id="GHG-3.1",
            requirement_description="Report Scope 1 emissions using approved methodology",
            how_met="Uses EPA 40 CFR Part 98 factors with Decimal-precision arithmetic",
        )
        assert cm.framework == RegulatoryFramework.GHG_PROTOCOL
        assert cm.requirement_id == "GHG-3.1"

    def test_default_status_met(self):
        cm = ComplianceMapping(
            framework=RegulatoryFramework.ISO_14064,
            requirement_id="ISO-4.1",
            requirement_description="Quantify GHG emissions",
            how_met="Multi-tier calculation engine",
        )
        assert cm.status == "met"

    def test_all_valid_statuses(self):
        for status in ("met", "partially_met", "not_met", "not_applicable"):
            cm = ComplianceMapping(
                framework=RegulatoryFramework.GHG_PROTOCOL,
                requirement_id="REQ-1",
                requirement_description="Test",
                how_met="Test",
                status=status,
            )
            assert cm.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError, match="status must be one of"):
            ComplianceMapping(
                framework=RegulatoryFramework.GHG_PROTOCOL,
                requirement_id="REQ-1",
                requirement_description="Test",
                how_met="Test",
                status="unknown",
            )

    def test_status_normalized(self):
        cm = ComplianceMapping(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            requirement_id="REQ-1",
            requirement_description="Test",
            how_met="Test",
            status="  Met  ",
        )
        assert cm.status == "met"

    def test_all_frameworks(self):
        for fw in RegulatoryFramework:
            cm = ComplianceMapping(
                framework=fw,
                requirement_id="REQ-1",
                requirement_description="Test requirement",
                how_met="Test implementation",
            )
            assert cm.framework == fw

    def test_evidence_reference_default_none(self):
        cm = ComplianceMapping(
            framework=RegulatoryFramework.EU_ETS,
            requirement_id="ETS-1",
            requirement_description="Track emissions",
            how_met="Monitoring",
        )
        assert cm.evidence_reference is None
