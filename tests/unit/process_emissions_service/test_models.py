# -*- coding: utf-8 -*-
"""Unit tests for Process Emissions Agent data models - AGENT-MRV-004.

Tests all 16 enumerations, constant lookup tables (GWP_VALUES,
CARBONATE_EMISSION_FACTORS, PROCESS_CATEGORY_MAP, PROCESS_DEFAULT_GASES),
and 17 Pydantic v2 data models with validation, edge cases, and
parametrize coverage.  Target: 150+ tests.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pydantic import ValidationError

from greenlang.process_emissions.models import (
    # Constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_MATERIAL_INPUTS_PER_CALC,
    GWP_VALUES,
    CARBONATE_EMISSION_FACTORS,
    PROCESS_CATEGORY_MAP,
    PROCESS_DEFAULT_GASES,
    # Enums
    ProcessCategory,
    ProcessType,
    EmissionGas,
    CalculationMethod,
    CalculationTier,
    EmissionFactorSource,
    GWPSource,
    MaterialType,
    AbatementType,
    ProcessUnitType,
    ProcessMode,
    ComplianceStatus,
    ReportingPeriod,
    UnitType,
    ProductionRoute,
    CarbonateType,
    # Data models
    ProcessTypeInfo,
    RawMaterialInfo,
    EmissionFactorRecord,
    ProcessUnitRecord,
    MaterialInputRecord,
    CalculationRequest,
    GasEmissionResult,
    CalculationResult,
    CalculationDetailResult,
    AbatementRecord,
    ComplianceCheckResult,
    BatchCalculationRequest,
    BatchCalculationResult,
    UncertaintyRequest,
    UncertaintyResult,
    AggregationRequest,
    AggregationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc).replace(microsecond=0)
_YEAR_AGO = _NOW - timedelta(days=365)
_YEAR_AHEAD = _NOW + timedelta(days=365)


# ============================================================================
# TestProcessCategory - 5 tests
# ============================================================================


class TestProcessCategory:
    """Test ProcessCategory enum."""

    @pytest.mark.parametrize("member,value", [
        (ProcessCategory.MINERAL, "mineral"),
        (ProcessCategory.CHEMICAL, "chemical"),
        (ProcessCategory.METAL, "metal"),
        (ProcessCategory.ELECTRONICS, "electronics"),
        (ProcessCategory.PULP_PAPER, "pulp_paper"),
        (ProcessCategory.OTHER, "other"),
    ])
    def test_enum_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ProcessCategory) == 6

    def test_str_conversion(self):
        assert str(ProcessCategory.MINERAL) == "ProcessCategory.MINERAL"

    def test_from_value(self):
        assert ProcessCategory("mineral") == ProcessCategory.MINERAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ProcessCategory("nonexistent")


# ============================================================================
# TestProcessType - 5 tests
# ============================================================================


class TestProcessType:
    """Test ProcessType enum with all 25 values."""

    _ALL_VALUES = [
        "cement_production", "lime_production", "glass_production",
        "ceramics", "soda_ash", "ammonia_production", "nitric_acid",
        "adipic_acid", "carbide_production", "petrochemical",
        "hydrogen_production", "phosphoric_acid", "titanium_dioxide",
        "iron_steel", "aluminum_smelting", "ferroalloy",
        "lead_production", "zinc_production", "magnesium_production",
        "copper_smelting", "semiconductor", "pulp_paper",
        "mineral_wool", "carbon_anode", "food_drink",
    ]

    def test_enum_count(self):
        assert len(ProcessType) == 25

    @pytest.mark.parametrize("value", _ALL_VALUES)
    def test_from_value(self, value):
        pt = ProcessType(value)
        assert pt.value == value

    def test_cement_production_value(self):
        assert ProcessType.CEMENT_PRODUCTION.value == "cement_production"

    def test_string_enum(self):
        """ProcessType is a str enum; comparing to string works."""
        assert ProcessType.IRON_STEEL == "iron_steel"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ProcessType("not_a_process")


# ============================================================================
# TestEmissionGas - 5 tests
# ============================================================================


class TestEmissionGas:
    """Test EmissionGas enum with all 8 greenhouse gas types."""

    @pytest.mark.parametrize("member,value", [
        (EmissionGas.CO2, "CO2"),
        (EmissionGas.CH4, "CH4"),
        (EmissionGas.N2O, "N2O"),
        (EmissionGas.CF4, "CF4"),
        (EmissionGas.C2F6, "C2F6"),
        (EmissionGas.SF6, "SF6"),
        (EmissionGas.NF3, "NF3"),
        (EmissionGas.HFC, "HFC"),
    ])
    def test_gas_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(EmissionGas) == 8

    def test_co2_is_primary(self):
        assert EmissionGas.CO2.value == "CO2"

    def test_from_value(self):
        assert EmissionGas("SF6") == EmissionGas.SF6

    def test_invalid_gas_raises(self):
        with pytest.raises(ValueError):
            EmissionGas("O3")


# ============================================================================
# TestCalculationMethod - 5 tests
# ============================================================================


class TestCalculationMethod:
    """Test CalculationMethod enum."""

    @pytest.mark.parametrize("member,value", [
        (CalculationMethod.EMISSION_FACTOR, "EMISSION_FACTOR"),
        (CalculationMethod.MASS_BALANCE, "MASS_BALANCE"),
        (CalculationMethod.STOICHIOMETRIC, "STOICHIOMETRIC"),
        (CalculationMethod.DIRECT_MEASUREMENT, "DIRECT_MEASUREMENT"),
    ])
    def test_method_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(CalculationMethod) == 4

    def test_default_is_emission_factor(self):
        assert CalculationMethod.EMISSION_FACTOR.value == "EMISSION_FACTOR"

    def test_from_value(self):
        assert CalculationMethod("MASS_BALANCE") == CalculationMethod.MASS_BALANCE

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            CalculationMethod("INTERPOLATION")


# ============================================================================
# TestCalculationTier - 3 tests
# ============================================================================


class TestCalculationTier:
    """Test CalculationTier enum."""

    @pytest.mark.parametrize("member,value", [
        (CalculationTier.TIER_1, "TIER_1"),
        (CalculationTier.TIER_2, "TIER_2"),
        (CalculationTier.TIER_3, "TIER_3"),
    ])
    def test_tier_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(CalculationTier) == 3

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            CalculationTier("TIER_0")


# ============================================================================
# TestEmissionFactorSource - 5 tests
# ============================================================================


class TestEmissionFactorSource:
    """Test EmissionFactorSource enum."""

    @pytest.mark.parametrize("member,value", [
        (EmissionFactorSource.EPA, "EPA"),
        (EmissionFactorSource.IPCC, "IPCC"),
        (EmissionFactorSource.DEFRA, "DEFRA"),
        (EmissionFactorSource.EU_ETS, "EU_ETS"),
        (EmissionFactorSource.CUSTOM, "CUSTOM"),
    ])
    def test_source_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(EmissionFactorSource) == 5

    def test_default_is_epa(self):
        assert EmissionFactorSource.EPA.value == "EPA"

    def test_from_value(self):
        assert EmissionFactorSource("DEFRA") == EmissionFactorSource.DEFRA

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            EmissionFactorSource("NOAA")


# ============================================================================
# TestGWPSource - 5 tests
# ============================================================================


class TestGWPSource:
    """Test GWPSource enum."""

    @pytest.mark.parametrize("member,value", [
        (GWPSource.AR4, "AR4"),
        (GWPSource.AR5, "AR5"),
        (GWPSource.AR6, "AR6"),
        (GWPSource.AR6_20YR, "AR6_20YR"),
    ])
    def test_gwp_source_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(GWPSource) == 4

    def test_default_is_ar6(self):
        assert GWPSource.AR6.value == "AR6"

    def test_from_value(self):
        assert GWPSource("AR6_20YR") == GWPSource.AR6_20YR

    def test_invalid_gwp_source_raises(self):
        with pytest.raises(ValueError):
            GWPSource("AR3")


# ============================================================================
# TestMaterialType - 5 tests
# ============================================================================


class TestMaterialType:
    """Test MaterialType enum."""

    def test_enum_has_calcium_carbonate(self):
        assert MaterialType.CALCIUM_CARBONATE.value == "calcium_carbonate"

    def test_enum_has_coke(self):
        assert MaterialType.COKE.value == "coke"

    def test_enum_has_other(self):
        assert MaterialType.OTHER.value == "other"

    def test_enum_count_gte_20(self):
        assert len(MaterialType) >= 20

    def test_from_value(self):
        assert MaterialType("clinker") == MaterialType.CLINKER


# ============================================================================
# TestAbatementType - 5 tests
# ============================================================================


class TestAbatementType:
    """Test AbatementType enum."""

    @pytest.mark.parametrize("member,value", [
        (AbatementType.CATALYTIC_REDUCTION, "catalytic_reduction"),
        (AbatementType.THERMAL_DESTRUCTION, "thermal_destruction"),
        (AbatementType.SCRUBBING, "scrubbing"),
        (AbatementType.CARBON_CAPTURE, "carbon_capture"),
        (AbatementType.PFC_ANODE_CONTROL, "pfc_anode_control"),
        (AbatementType.SF6_RECOVERY, "sf6_recovery"),
        (AbatementType.NSCR, "nscr"),
        (AbatementType.SCR, "scr"),
        (AbatementType.EXTENDED_ABSORPTION, "extended_absorption"),
        (AbatementType.OTHER, "other"),
    ])
    def test_abatement_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(AbatementType) == 10

    def test_from_value(self):
        assert AbatementType("scr") == AbatementType.SCR

    def test_invalid_abatement_raises(self):
        with pytest.raises(ValueError):
            AbatementType("magic_filter")

    def test_is_str_enum(self):
        assert isinstance(AbatementType.SCRUBBING, str)


# ============================================================================
# TestProcessUnitType - 5 tests
# ============================================================================


class TestProcessUnitType:
    """Test ProcessUnitType enum."""

    @pytest.mark.parametrize("member,value", [
        (ProcessUnitType.KILN, "kiln"),
        (ProcessUnitType.FURNACE, "furnace"),
        (ProcessUnitType.SMELTER, "smelter"),
        (ProcessUnitType.REACTOR, "reactor"),
        (ProcessUnitType.ELECTROLYSIS_CELL, "electrolysis_cell"),
        (ProcessUnitType.REFORMER, "reformer"),
        (ProcessUnitType.CONVERTER, "converter"),
        (ProcessUnitType.CALCINER, "calciner"),
        (ProcessUnitType.DRYER, "dryer"),
        (ProcessUnitType.OTHER, "other"),
    ])
    def test_unit_type_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ProcessUnitType) == 10

    def test_from_value(self):
        assert ProcessUnitType("kiln") == ProcessUnitType.KILN

    def test_invalid_unit_type_raises(self):
        with pytest.raises(ValueError):
            ProcessUnitType("turbine")

    def test_is_str_enum(self):
        assert isinstance(ProcessUnitType.KILN, str)


# ============================================================================
# TestProcessMode - 3 tests
# ============================================================================


class TestProcessMode:
    """Test ProcessMode enum."""

    @pytest.mark.parametrize("member,value", [
        (ProcessMode.BATCH, "batch"),
        (ProcessMode.CONTINUOUS, "continuous"),
        (ProcessMode.SEMI_CONTINUOUS, "semi_continuous"),
    ])
    def test_mode_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ProcessMode) == 3

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ProcessMode("manual")


# ============================================================================
# TestCarbonateType - 5 tests
# ============================================================================


class TestCarbonateType:
    """Test CarbonateType enum."""

    @pytest.mark.parametrize("member,value", [
        (CarbonateType.CALCITE, "calcite"),
        (CarbonateType.DOLOMITE, "dolomite"),
        (CarbonateType.MAGNESITE, "magnesite"),
        (CarbonateType.SIDERITE, "siderite"),
        (CarbonateType.ANKERITE, "ankerite"),
        (CarbonateType.OTHER, "other"),
    ])
    def test_carbonate_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(CarbonateType) == 6

    def test_from_value(self):
        assert CarbonateType("dolomite") == CarbonateType.DOLOMITE

    def test_invalid_carbonate_raises(self):
        with pytest.raises(ValueError):
            CarbonateType("marble")

    def test_is_str_enum(self):
        assert isinstance(CarbonateType.CALCITE, str)


# ============================================================================
# TestGWPValues - 15 tests
# ============================================================================


class TestGWPValues:
    """Test GWP_VALUES constant lookup table for correctness."""

    @pytest.mark.parametrize("ar_source", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_gwp_source_exists(self, ar_source):
        assert ar_source in GWP_VALUES

    @pytest.mark.parametrize("ar_source", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_all_8_gases_present(self, ar_source):
        gases = GWP_VALUES[ar_source]
        for gas in ["CO2", "CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC"]:
            assert gas in gases, f"{gas} missing from {ar_source}"

    @pytest.mark.parametrize("ar_source", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_co2_gwp_is_always_one(self, ar_source):
        assert GWP_VALUES[ar_source]["CO2"] == 1.0

    def test_ar4_ch4_gwp(self):
        assert GWP_VALUES["AR4"]["CH4"] == 25.0

    def test_ar5_ch4_gwp(self):
        assert GWP_VALUES["AR5"]["CH4"] == 28.0

    def test_ar6_ch4_gwp(self):
        assert GWP_VALUES["AR6"]["CH4"] == 29.8

    def test_ar6_20yr_ch4_gwp(self):
        assert GWP_VALUES["AR6_20YR"]["CH4"] == 82.5

    def test_ar6_n2o_gwp(self):
        assert GWP_VALUES["AR6"]["N2O"] == 273.0

    def test_ar6_sf6_gwp(self):
        assert GWP_VALUES["AR6"]["SF6"] == 25200.0

    @pytest.mark.parametrize("gas", ["CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC"])
    def test_gwp_values_positive(self, gas):
        for ar_source in GWP_VALUES:
            assert GWP_VALUES[ar_source][gas] > 0

    @pytest.mark.parametrize("gas", ["CF4", "C2F6", "SF6", "NF3"])
    def test_fluorinated_gases_high_gwp(self, gas):
        """Fluorinated gases always have GWP > 1000."""
        for ar_source in GWP_VALUES:
            assert GWP_VALUES[ar_source][gas] > 1000

    def test_gwp_values_is_dict(self):
        assert isinstance(GWP_VALUES, dict)

    def test_no_extra_ar_sources(self):
        assert len(GWP_VALUES) == 4

    def test_ar6_hfc_gwp(self):
        assert GWP_VALUES["AR6"]["HFC"] == 1530.0


# ============================================================================
# TestCarbonateEmissionFactors - 10 tests
# ============================================================================


class TestCarbonateEmissionFactors:
    """Test CARBONATE_EMISSION_FACTORS constant lookup table."""

    @pytest.mark.parametrize("carbonate,expected_ef", [
        ("calcite", Decimal("0.440")),
        ("dolomite", Decimal("0.477")),
        ("magnesite", Decimal("0.522")),
        ("siderite", Decimal("0.380")),
        ("ankerite", Decimal("0.407")),
    ])
    def test_carbonate_factor_value(self, carbonate, expected_ef):
        assert CARBONATE_EMISSION_FACTORS[carbonate] == expected_ef

    def test_factors_count(self):
        assert len(CARBONATE_EMISSION_FACTORS) == 5

    def test_factors_are_decimal(self):
        for val in CARBONATE_EMISSION_FACTORS.values():
            assert isinstance(val, Decimal)

    def test_all_factors_between_zero_and_one(self):
        for val in CARBONATE_EMISSION_FACTORS.values():
            assert Decimal("0") < val < Decimal("1")

    def test_magnesite_highest_factor(self):
        """MgCO3 has the highest CO2 emission factor (0.522)."""
        assert CARBONATE_EMISSION_FACTORS["magnesite"] == max(
            CARBONATE_EMISSION_FACTORS.values()
        )

    def test_siderite_lowest_factor(self):
        """FeCO3 has the lowest CO2 emission factor (0.380)."""
        assert CARBONATE_EMISSION_FACTORS["siderite"] == min(
            CARBONATE_EMISSION_FACTORS.values()
        )


# ============================================================================
# TestProcessCategoryMap - 5 tests
# ============================================================================


class TestProcessCategoryMap:
    """Test PROCESS_CATEGORY_MAP constant."""

    def test_all_six_categories_present(self):
        expected = {"mineral", "chemical", "metal", "electronics", "pulp_paper", "other"}
        assert set(PROCESS_CATEGORY_MAP.keys()) == expected

    def test_mineral_has_five_processes(self):
        assert len(PROCESS_CATEGORY_MAP["mineral"]) == 5

    def test_chemical_has_eight_processes(self):
        assert len(PROCESS_CATEGORY_MAP["chemical"]) == 8

    def test_metal_has_seven_processes(self):
        assert len(PROCESS_CATEGORY_MAP["metal"]) == 7

    def test_total_process_count(self):
        total = sum(len(v) for v in PROCESS_CATEGORY_MAP.values())
        assert total == 25


# ============================================================================
# TestProcessDefaultGases - 5 tests
# ============================================================================


class TestProcessDefaultGases:
    """Test PROCESS_DEFAULT_GASES constant."""

    def test_cement_emits_co2(self):
        assert "CO2" in PROCESS_DEFAULT_GASES["cement_production"]

    def test_nitric_acid_emits_n2o(self):
        assert "N2O" in PROCESS_DEFAULT_GASES["nitric_acid"]

    def test_semiconductor_emits_fluorinated(self):
        gases = PROCESS_DEFAULT_GASES["semiconductor"]
        assert "CF4" in gases
        assert "SF6" in gases

    def test_aluminum_emits_pfcs(self):
        gases = PROCESS_DEFAULT_GASES["aluminum_smelting"]
        assert "CF4" in gases
        assert "C2F6" in gases

    def test_all_25_process_types_present(self):
        assert len(PROCESS_DEFAULT_GASES) == 25


# ============================================================================
# TestConstants - 5 tests
# ============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        assert MAX_GASES_PER_RESULT == 20

    def test_max_trace_steps(self):
        assert MAX_TRACE_STEPS == 200

    def test_max_material_inputs_per_calc(self):
        assert MAX_MATERIAL_INPUTS_PER_CALC == 50


# ============================================================================
# TestCalculationRequest - 15 tests
# ============================================================================


class TestCalculationRequest:
    """Test CalculationRequest Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "process_type": "cement_production",
            "production_quantity_tonnes": 1000.0,
            "period_start": _YEAR_AGO,
            "period_end": _NOW,
        }
        defaults.update(overrides)
        return CalculationRequest(**defaults)

    def test_minimal_creation(self):
        req = self._base()
        assert req.process_type == ProcessType.CEMENT_PRODUCTION
        assert req.production_quantity_tonnes == 1000.0

    def test_defaults_for_method_tier_source(self):
        req = self._base()
        assert req.calculation_method == CalculationMethod.EMISSION_FACTOR
        assert req.tier == CalculationTier.TIER_1
        assert req.emission_factor_source == EmissionFactorSource.EPA
        assert req.gwp_source == GWPSource.AR6

    def test_custom_method(self):
        req = self._base(calculation_method="MASS_BALANCE")
        assert req.calculation_method == CalculationMethod.MASS_BALANCE

    def test_custom_tier(self):
        req = self._base(tier="TIER_3")
        assert req.tier == CalculationTier.TIER_3

    def test_custom_gwp_source(self):
        req = self._base(gwp_source="AR5")
        assert req.gwp_source == GWPSource.AR5

    def test_missing_process_type_raises(self):
        with pytest.raises(ValidationError):
            CalculationRequest(
                production_quantity_tonnes=100,
                period_start=_YEAR_AGO,
                period_end=_NOW,
            )

    def test_missing_production_quantity_raises(self):
        with pytest.raises(ValidationError):
            CalculationRequest(
                process_type="cement_production",
                period_start=_YEAR_AGO,
                period_end=_NOW,
            )

    def test_zero_production_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._base(production_quantity_tonnes=0)

    def test_negative_production_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._base(production_quantity_tonnes=-100)

    def test_period_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="period_end"):
            self._base(period_start=_NOW, period_end=_YEAR_AGO)

    def test_period_end_equal_start_raises(self):
        with pytest.raises(ValidationError, match="period_end"):
            self._base(period_start=_NOW, period_end=_NOW)

    def test_optional_facility_id(self):
        req = self._base(facility_id="FAC-001")
        assert req.facility_id == "FAC-001"

    def test_optional_unit_id(self):
        req = self._base(unit_id="PU-001")
        assert req.unit_id == "PU-001"

    def test_material_inputs_default_empty(self):
        req = self._base()
        assert req.material_inputs == []

    def test_custom_emission_factors(self):
        req = self._base(custom_emission_factors={"CO2": 0.525})
        assert req.custom_emission_factors["CO2"] == 0.525


# ============================================================================
# TestCalculationResult - 10 tests
# ============================================================================


class TestCalculationResult:
    """Test CalculationResult Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "process_type": "cement_production",
            "process_category": "mineral",
            "production_quantity_tonnes": 1000.0,
            "calculation_method": "EMISSION_FACTOR",
            "tier_used": "TIER_1",
            "total_co2e_kg": 440_000.0,
            "total_co2e_tonnes": 440.0,
        }
        defaults.update(overrides)
        return CalculationResult(**defaults)

    def test_minimal_creation(self):
        result = self._base()
        assert result.total_co2e_tonnes == 440.0

    def test_calculation_id_generated(self):
        result = self._base()
        assert result.calculation_id.startswith("pecalc_")

    def test_timestamp_set(self):
        result = self._base()
        assert result.timestamp is not None

    def test_default_gross_zero(self):
        result = self._base()
        assert result.gross_co2e_tonnes == 0.0

    def test_default_abatement_zero(self):
        result = self._base()
        assert result.abatement_co2e_tonnes == 0.0

    def test_default_net_zero(self):
        result = self._base()
        assert result.net_co2e_tonnes == 0.0

    def test_default_provenance_empty(self):
        result = self._base()
        assert result.provenance_hash == ""

    def test_default_trace_empty(self):
        result = self._base()
        assert result.calculation_trace == []

    def test_emissions_by_gas_default_empty(self):
        result = self._base()
        assert result.emissions_by_gas == []

    def test_negative_total_raises(self):
        with pytest.raises(ValidationError):
            self._base(total_co2e_tonnes=-1.0)


# ============================================================================
# TestGasEmissionResult - 5 tests
# ============================================================================


class TestGasEmissionResult:
    """Test GasEmissionResult Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "gas": "CO2",
            "emissions_kg": 440_000.0,
            "emissions_tonnes": 440.0,
            "emissions_tco2e": 440.0,
            "emission_factor_value": 0.525,
            "emission_factor_unit": "tCO2/t clinker",
            "emission_factor_source": "IPCC",
            "gwp_applied": 1.0,
        }
        defaults.update(overrides)
        return GasEmissionResult(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.gas == EmissionGas.CO2
        assert r.emissions_tonnes == 440.0

    def test_negative_emissions_raises(self):
        with pytest.raises(ValidationError):
            self._base(emissions_kg=-1.0)

    def test_zero_emissions_valid(self):
        r = self._base(emissions_kg=0.0, emissions_tonnes=0.0, emissions_tco2e=0.0)
        assert r.emissions_kg == 0.0

    def test_gwp_source_default(self):
        r = self._base()
        assert r.gwp_source == "AR6"

    def test_custom_gwp_source(self):
        r = self._base(gwp_source="AR5")
        assert r.gwp_source == "AR5"


# ============================================================================
# TestBatchCalculationRequest - 5 tests
# ============================================================================


class TestBatchCalculationRequest:
    """Test BatchCalculationRequest Pydantic model."""

    def _make_calc_req(self, process_type="cement_production"):
        return CalculationRequest(
            process_type=process_type,
            production_quantity_tonnes=1000.0,
            period_start=_YEAR_AGO,
            period_end=_NOW,
        )

    def test_creation_with_one_calc(self):
        batch = BatchCalculationRequest(
            calculations=[self._make_calc_req()]
        )
        assert len(batch.calculations) == 1

    def test_creation_with_multiple_calcs(self):
        batch = BatchCalculationRequest(
            calculations=[
                self._make_calc_req("cement_production"),
                self._make_calc_req("iron_steel"),
                self._make_calc_req("nitric_acid"),
            ]
        )
        assert len(batch.calculations) == 3

    def test_empty_calculations_raises(self):
        with pytest.raises(ValidationError):
            BatchCalculationRequest(calculations=[])

    def test_default_gwp_source(self):
        batch = BatchCalculationRequest(
            calculations=[self._make_calc_req()]
        )
        assert batch.gwp_source == GWPSource.AR6

    def test_enable_compliance_default_false(self):
        batch = BatchCalculationRequest(
            calculations=[self._make_calc_req()]
        )
        assert batch.enable_compliance is False


# ============================================================================
# TestMaterialInputRecord - 10 tests
# ============================================================================


class TestMaterialInputRecord:
    """Test MaterialInputRecord Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "material_type": "limestone",
            "quantity_tonnes": 100.0,
        }
        defaults.update(overrides)
        return MaterialInputRecord(**defaults)

    def test_minimal_creation(self):
        r = self._base()
        assert r.quantity_tonnes == 100.0

    def test_default_purity(self):
        r = self._base()
        assert r.purity_fraction == 1.0

    def test_default_moisture(self):
        r = self._base()
        assert r.moisture_fraction == 0.0

    def test_zero_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._base(quantity_tonnes=0.0)

    def test_negative_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._base(quantity_tonnes=-10.0)

    def test_carbon_content_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self._base(carbon_content_fraction=1.5)

    def test_purity_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self._base(purity_fraction=1.1)

    def test_moisture_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self._base(moisture_fraction=-0.1)

    def test_optional_carbonate_type(self):
        r = self._base(carbonate_type="calcite")
        assert r.carbonate_type == CarbonateType.CALCITE

    def test_optional_source_description(self):
        r = self._base(source_description="Local quarry, batch 2025-03")
        assert "quarry" in r.source_description


# ============================================================================
# TestAbatementRecord - 5 tests
# ============================================================================


class TestAbatementRecord:
    """Test AbatementRecord Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "unit_id": "PU-001",
            "abatement_type": "catalytic_reduction",
            "efficiency": 0.90,
            "target_gas": "N2O",
        }
        defaults.update(overrides)
        return AbatementRecord(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.efficiency == 0.90
        assert r.target_gas == EmissionGas.N2O

    def test_abatement_id_generated(self):
        r = self._base()
        assert r.abatement_id.startswith("abate_")

    def test_efficiency_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self._base(efficiency=1.5)

    def test_invalid_operational_status_raises(self):
        with pytest.raises(ValidationError):
            self._base(operational_status="broken")

    def test_valid_operational_statuses(self):
        for status in ["active", "inactive", "maintenance"]:
            r = self._base(operational_status=status)
            assert r.operational_status == status


# ============================================================================
# TestEmissionFactorRecord - 8 tests
# ============================================================================


class TestEmissionFactorRecord:
    """Test EmissionFactorRecord Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "process_type": "cement_production",
            "gas": "CO2",
            "value": 0.525,
            "unit": "tCO2/t clinker",
        }
        defaults.update(overrides)
        return EmissionFactorRecord(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.value == 0.525

    def test_factor_id_generated(self):
        r = self._base()
        assert r.factor_id.startswith("pef_")

    def test_zero_value_raises(self):
        with pytest.raises(ValidationError):
            self._base(value=0.0)

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError):
            self._base(value=-0.1)

    def test_default_source_epa(self):
        r = self._base()
        assert r.source == EmissionFactorSource.EPA

    def test_default_tier_1(self):
        r = self._base()
        assert r.tier == CalculationTier.TIER_1

    def test_default_geography_global(self):
        r = self._base()
        assert r.geography == "GLOBAL"

    def test_expiry_before_effective_raises(self):
        with pytest.raises(ValidationError, match="expiry_date"):
            self._base(
                effective_date=_NOW,
                expiry_date=_YEAR_AGO,
            )


# ============================================================================
# TestProcessUnitRecord - 5 tests
# ============================================================================


class TestProcessUnitRecord:
    """Test ProcessUnitRecord Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "name": "Kiln A",
            "process_type": "cement_production",
            "unit_type": "kiln",
        }
        defaults.update(overrides)
        return ProcessUnitRecord(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.name == "Kiln A"

    def test_unit_id_generated(self):
        r = self._base()
        assert r.unit_id.startswith("pu_")

    def test_default_mode_continuous(self):
        r = self._base()
        assert r.process_mode == ProcessMode.CONTINUOUS

    def test_installation_year_valid(self):
        r = self._base(installation_year=2020)
        assert r.installation_year == 2020

    def test_installation_year_too_old_raises(self):
        with pytest.raises(ValidationError):
            self._base(installation_year=1899)


# ============================================================================
# TestProcessTypeInfo - 5 tests
# ============================================================================


class TestProcessTypeInfo:
    """Test ProcessTypeInfo Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "process_type": "cement_production",
            "category": "mineral",
            "display_name": "Cement Production",
        }
        defaults.update(overrides)
        return ProcessTypeInfo(**defaults)

    def test_creation(self):
        info = self._base()
        assert info.process_type == ProcessType.CEMENT_PRODUCTION

    def test_frozen_model(self):
        info = self._base()
        with pytest.raises(ValidationError):
            info.display_name = "Changed"

    def test_default_tier_1(self):
        info = self._base()
        assert info.default_tier == CalculationTier.TIER_1

    def test_default_method_emission_factor(self):
        info = self._base()
        assert info.default_method == CalculationMethod.EMISSION_FACTOR

    def test_empty_display_name_raises(self):
        with pytest.raises(ValidationError):
            self._base(display_name="")


# ============================================================================
# TestComplianceCheckResult - 5 tests
# ============================================================================


class TestComplianceCheckResult:
    """Test ComplianceCheckResult Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "framework": "GHG_PROTOCOL",
            "status": "compliant",
        }
        defaults.update(overrides)
        return ComplianceCheckResult(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.status == ComplianceStatus.COMPLIANT

    def test_all_statuses(self):
        for status in ["compliant", "non_compliant", "partial", "not_checked"]:
            r = self._base(status=status)
            assert r.status.value == status

    def test_default_counts_zero(self):
        r = self._base()
        assert r.requirement_count == 0
        assert r.met_count == 0
        assert r.not_met_count == 0

    def test_checked_at_auto_set(self):
        r = self._base()
        assert r.checked_at is not None

    def test_empty_framework_raises(self):
        with pytest.raises(ValidationError):
            self._base(framework="")


# ============================================================================
# TestUncertaintyRequest - 5 tests
# ============================================================================


class TestUncertaintyRequest:
    """Test UncertaintyRequest Pydantic model."""

    def _make_calc_req(self):
        return CalculationRequest(
            process_type="cement_production",
            production_quantity_tonnes=1000.0,
            period_start=_YEAR_AGO,
            period_end=_NOW,
        )

    def test_creation(self):
        req = UncertaintyRequest(calculation_request=self._make_calc_req())
        assert req.iterations == 5000

    def test_custom_iterations(self):
        req = UncertaintyRequest(
            calculation_request=self._make_calc_req(),
            iterations=10000,
        )
        assert req.iterations == 10000

    def test_custom_seed(self):
        req = UncertaintyRequest(
            calculation_request=self._make_calc_req(),
            seed=99,
        )
        assert req.seed == 99

    def test_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="confidence level"):
            UncertaintyRequest(
                calculation_request=self._make_calc_req(),
                confidence_levels=[0.0, 50.0],
            )

    def test_default_confidence_levels(self):
        req = UncertaintyRequest(calculation_request=self._make_calc_req())
        assert req.confidence_levels == [90.0, 95.0, 99.0]


# ============================================================================
# TestUncertaintyResult - 5 tests
# ============================================================================


class TestUncertaintyResult:
    """Test UncertaintyResult Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "mean_co2e_tonnes": 440.0,
            "std_dev_tonnes": 22.0,
            "coefficient_of_variation": 0.05,
            "iterations": 5000,
            "tier": "TIER_1",
            "process_type": "cement_production",
            "calculation_method": "EMISSION_FACTOR",
        }
        defaults.update(overrides)
        return UncertaintyResult(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.mean_co2e_tonnes == 440.0

    def test_negative_mean_raises(self):
        with pytest.raises(ValidationError):
            self._base(mean_co2e_tonnes=-1.0)

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValidationError):
            self._base(std_dev_tonnes=-1.0)

    def test_dqi_range_valid(self):
        r = self._base(data_quality_score=3.0)
        assert r.data_quality_score == 3.0

    def test_dqi_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self._base(data_quality_score=0.5)


# ============================================================================
# TestAggregationRequest - 5 tests
# ============================================================================


class TestAggregationRequest:
    """Test AggregationRequest Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "period_start": _YEAR_AGO,
            "period_end": _NOW,
        }
        defaults.update(overrides)
        return AggregationRequest(**defaults)

    def test_creation(self):
        req = self._base()
        assert req.reporting_period == ReportingPeriod.ANNUAL

    def test_default_group_by(self):
        req = self._base()
        assert req.group_by == ["process_type"]

    def test_period_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="period_end"):
            self._base(period_start=_NOW, period_end=_YEAR_AGO)

    def test_invalid_group_by_raises(self):
        with pytest.raises(ValidationError, match="group_by"):
            self._base(group_by=["invalid_dimension"])

    def test_valid_group_by_dimensions(self):
        for dim in ["process_type", "category", "facility", "gas", "production_route"]:
            req = self._base(group_by=[dim])
            assert dim in req.group_by


# ============================================================================
# TestBatchCalculationResult - 5 tests
# ============================================================================


class TestBatchCalculationResult:
    """Test BatchCalculationResult Pydantic model."""

    def test_creation(self):
        r = BatchCalculationResult(success=True)
        assert r.success is True

    def test_defaults(self):
        r = BatchCalculationResult(success=False)
        assert r.total_co2e_tonnes == 0.0
        assert r.calculation_count == 0
        assert r.failed_count == 0
        assert r.processing_time_ms == 0.0

    def test_default_gwp_source(self):
        r = BatchCalculationResult(success=True)
        assert r.gwp_source == GWPSource.AR6

    def test_results_default_empty(self):
        r = BatchCalculationResult(success=True)
        assert r.results == []

    def test_compliance_results_default_empty(self):
        r = BatchCalculationResult(success=True)
        assert r.compliance_results == []


# ============================================================================
# TestCalculationDetailResult - 3 tests
# ============================================================================


class TestCalculationDetailResult:
    """Test CalculationDetailResult wrapper model."""

    def _make_result(self):
        return CalculationResult(
            process_type="cement_production",
            process_category="mineral",
            production_quantity_tonnes=1000.0,
            calculation_method="EMISSION_FACTOR",
            tier_used="TIER_1",
            total_co2e_kg=440_000.0,
            total_co2e_tonnes=440.0,
        )

    def test_creation(self):
        detail = CalculationDetailResult(result=self._make_result())
        assert detail.result.total_co2e_tonnes == 440.0

    def test_default_optional_fields_none(self):
        detail = CalculationDetailResult(result=self._make_result())
        assert detail.material_balance is None
        assert detail.stoichiometric_details is None
        assert detail.emission_factor_details is None
        assert detail.abatement_details is None

    def test_audit_entries_default_empty(self):
        detail = CalculationDetailResult(result=self._make_result())
        assert detail.audit_entries == []


# ============================================================================
# TestRawMaterialInfo - 5 tests
# ============================================================================


class TestRawMaterialInfo:
    """Test RawMaterialInfo Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "material_type": "calcium_carbonate",
            "display_name": "Calcium Carbonate (CaCO3)",
        }
        defaults.update(overrides)
        return RawMaterialInfo(**defaults)

    def test_creation(self):
        info = self._base()
        assert info.material_type == MaterialType.CALCIUM_CARBONATE

    def test_frozen_model(self):
        info = self._base()
        with pytest.raises(ValidationError):
            info.display_name = "Changed"

    def test_carbon_content_valid_range(self):
        info = self._base(carbon_content_fraction=0.12)
        assert info.carbon_content_fraction == 0.12

    def test_carbon_content_out_of_range(self):
        with pytest.raises(ValidationError):
            self._base(carbon_content_fraction=1.5)

    def test_empty_display_name_raises(self):
        with pytest.raises(ValidationError):
            self._base(display_name="")


# ============================================================================
# TestAggregationResult - 3 tests
# ============================================================================


class TestAggregationResult:
    """Test AggregationResult Pydantic model."""

    def _base(self, **overrides):
        defaults = {
            "period_start": _YEAR_AGO,
            "period_end": _NOW,
            "reporting_period": "annual",
        }
        defaults.update(overrides)
        return AggregationResult(**defaults)

    def test_creation(self):
        r = self._base()
        assert r.total_co2e_tonnes == 0.0

    def test_period_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="period_end"):
            self._base(period_start=_NOW, period_end=_YEAR_AGO)

    def test_defaults(self):
        r = self._base()
        assert r.calculation_count == 0
        assert r.process_types_included == []
        assert r.provenance_hash == ""


# ============================================================================
# Additional enum tests
# ============================================================================


class TestComplianceStatus:
    """Test ComplianceStatus enum."""

    @pytest.mark.parametrize("member,value", [
        (ComplianceStatus.COMPLIANT, "compliant"),
        (ComplianceStatus.NON_COMPLIANT, "non_compliant"),
        (ComplianceStatus.PARTIAL, "partial"),
        (ComplianceStatus.NOT_CHECKED, "not_checked"),
    ])
    def test_status_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ComplianceStatus) == 4


class TestReportingPeriod:
    """Test ReportingPeriod enum."""

    @pytest.mark.parametrize("member,value", [
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.ANNUAL, "annual"),
    ])
    def test_period_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ReportingPeriod) == 3


class TestUnitType:
    """Test UnitType enum."""

    @pytest.mark.parametrize("member,value", [
        (UnitType.MASS, "mass"),
        (UnitType.VOLUME, "volume"),
        (UnitType.ENERGY, "energy"),
        (UnitType.AREA, "area"),
        (UnitType.COUNT, "count"),
    ])
    def test_unit_type_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(UnitType) == 5


class TestProductionRoute:
    """Test ProductionRoute enum."""

    @pytest.mark.parametrize("member,value", [
        (ProductionRoute.BF_BOF, "bf_bof"),
        (ProductionRoute.EAF, "eaf"),
        (ProductionRoute.DRI, "dri"),
        (ProductionRoute.OHF, "ohf"),
        (ProductionRoute.PREBAKE, "prebake"),
        (ProductionRoute.SODERBERG_VSS, "soderberg_vss"),
        (ProductionRoute.SODERBERG_HSS, "soderberg_hss"),
        (ProductionRoute.CWPB, "cwpb"),
        (ProductionRoute.SWPB, "swpb"),
    ])
    def test_route_value(self, member, value):
        assert member.value == value

    def test_enum_count(self):
        assert len(ProductionRoute) == 9

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            ProductionRoute("bessemer")
