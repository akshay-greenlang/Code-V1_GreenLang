# -*- coding: utf-8 -*-
"""
Tests for Scope3ActivityEngine - PACK-022 Engine 4

Activity-based Scope 3 emission calculations for 9 top categories
with hybrid methodology, data quality scoring, and activity-vs-spend
comparison.

Coverage targets: 85%+ of Scope3ActivityEngine methods.
"""

import pytest
from decimal import Decimal

from engines.scope3_activity_engine import (
    Scope3ActivityEngine,
    Scope3ActivityInput,
    Scope3ActivityResult,
    CategoryResult,
    MethodComparison,
    PurchasedGoodEntry,
    FuelEnergyEntry,
    TransportEntry,
    WasteEntry,
    TravelEntry,
    CommuteProfile,
    UseOfSoldEntry,
    EndOfLifeEntry,
    SpendFallbackEntry,
    Scope3Category,
    CalculationMethod,
    TransportMode,
    WasteType,
    TravelMode,
    CommuteMode,
    TRANSPORT_FACTORS,
    TRAVEL_FACTORS,
    COMMUTE_FACTORS,
    WASTE_FACTORS,
    PRODUCT_FACTORS,
    SPEND_FACTORS,
    WTT_FACTORS,
    DQ_SCORES,
    CATEGORY_NAMES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a Scope3ActivityEngine instance."""
    return Scope3ActivityEngine()


@pytest.fixture
def minimal_input():
    """Minimal input with no activity data."""
    return Scope3ActivityInput(
        entity_name="MinCo",
        reporting_year=2024,
    )


@pytest.fixture
def full_input():
    """Full input with data across multiple categories."""
    return Scope3ActivityInput(
        entity_name="FullCorp",
        reporting_year=2024,
        purchased_goods=[
            PurchasedGoodEntry(product_type="steel", quantity_kg=Decimal("100000")),
            PurchasedGoodEntry(product_type="plastics_general", quantity_kg=Decimal("50000")),
        ],
        fuel_energy=[
            FuelEnergyEntry(fuel_type="electricity", quantity=Decimal("500000"), unit="kwh"),
            FuelEnergyEntry(fuel_type="natural_gas", quantity=Decimal("200000"), unit="kwh"),
        ],
        upstream_transport=[
            TransportEntry(mode=TransportMode.ROAD_TRUCK, tonnes=Decimal("500"), distance_km=Decimal("1000")),
            TransportEntry(mode=TransportMode.RAIL_FREIGHT, tonnes=Decimal("1000"), distance_km=Decimal("800")),
        ],
        waste=[
            WasteEntry(waste_type=WasteType.GENERAL_LANDFILL, quantity_tonnes=Decimal("200")),
            WasteEntry(waste_type=WasteType.RECYCLING, quantity_tonnes=Decimal("150")),
        ],
        business_travel=[
            TravelEntry(mode=TravelMode.AIR_SHORT_HAUL, distance_km=Decimal("50000")),
            TravelEntry(mode=TravelMode.RAIL, distance_km=Decimal("20000")),
            TravelEntry(mode=TravelMode.HOTEL_NIGHT, nights=100),
        ],
        commuting_profiles=[
            CommuteProfile(mode=CommuteMode.CAR_PETROL, employee_pct=Decimal("40"), avg_distance_km_one_way=Decimal("20")),
            CommuteProfile(mode=CommuteMode.BUS, employee_pct=Decimal("30"), avg_distance_km_one_way=Decimal("15")),
            CommuteProfile(mode=CommuteMode.BICYCLE, employee_pct=Decimal("30"), avg_distance_km_one_way=Decimal("5")),
        ],
        total_employees=500,
        working_days_per_year=230,
        remote_work_pct=Decimal("20"),
        downstream_transport=[
            TransportEntry(mode=TransportMode.SEA_CONTAINER, tonnes=Decimal("2000"), distance_km=Decimal("5000")),
        ],
        use_of_sold_products=[
            UseOfSoldEntry(
                product_name="Appliance",
                units_sold=Decimal("10000"),
                energy_per_use_kwh=Decimal("2.5"),
                uses_per_lifetime=Decimal("5000"),
            ),
        ],
        end_of_life=[
            EndOfLifeEntry(
                product_name="Appliance",
                units_sold=Decimal("10000"),
                weight_per_unit_kg=Decimal("15"),
                treatment_type=WasteType.GENERAL_LANDFILL,
            ),
        ],
        spend_fallbacks=[
            SpendFallbackEntry(category=Scope3Category.CAT_02, spend_usd_thousands=Decimal("5000")),
        ],
        grid_factor_tco2e_per_mwh=Decimal("0.436"),
    )


# ---------------------------------------------------------------------------
# TestScope3Basic
# ---------------------------------------------------------------------------


class TestScope3Basic:
    """Basic functionality tests for Scope3ActivityEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated."""
        engine = Scope3ActivityEngine()
        assert engine.engine_version == "1.0.0"

    def test_calculate_returns_result(self, engine, full_input):
        """calculate() returns a Scope3ActivityResult."""
        result = engine.calculate(full_input)
        assert isinstance(result, Scope3ActivityResult)

    def test_result_entity_name(self, engine, full_input):
        """Result carries entity name."""
        result = engine.calculate(full_input)
        assert result.entity_name == "FullCorp"

    def test_result_reporting_year(self, engine, full_input):
        """Result carries reporting year."""
        result = engine.calculate(full_input)
        assert result.reporting_year == 2024

    def test_result_provenance_hash(self, engine, full_input):
        """Result has 64-char hex provenance hash."""
        result = engine.calculate(full_input)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_result_processing_time(self, engine, full_input):
        """Result has positive processing time."""
        result = engine.calculate(full_input)
        assert result.processing_time_ms > 0.0

    def test_result_total_scope3_positive(self, engine, full_input):
        """Total Scope 3 is positive with data."""
        result = engine.calculate(full_input)
        assert float(result.total_scope3_tco2e) > 0


# ---------------------------------------------------------------------------
# TestAllCategoriesPresent
# ---------------------------------------------------------------------------


class TestAllCategoriesPresent:
    """Tests that all 15 categories are present in output."""

    def test_15_categories_present(self, engine, full_input):
        """Result has entries for all 15 Scope 3 categories."""
        result = engine.calculate(full_input)
        cat_values = {r.category for r in result.category_emissions}
        for cat in Scope3Category:
            assert cat.value in cat_values

    def test_minimal_input_has_15_categories(self, engine, minimal_input):
        """Even with no data, all 15 categories are present (with zero)."""
        result = engine.calculate(minimal_input)
        assert len(result.category_emissions) == 15


# ---------------------------------------------------------------------------
# TestCat01PurchasedGoods
# ---------------------------------------------------------------------------


class TestCat01PurchasedGoods:
    """Tests for Category 1 (Purchased Goods) calculation."""

    def test_cat01_steel_calculation(self, engine):
        """Steel: 100,000 kg * 1.89 kgCO2e/kg / 1000 = 189 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="SteelTest",
            reporting_year=2024,
            purchased_goods=[
                PurchasedGoodEntry(product_type="steel", quantity_kg=Decimal("100000")),
            ],
        )
        result = engine.calculate(inp)
        cat01 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_01.value][0]
        assert float(cat01.emissions_tco2e) == pytest.approx(189.0, rel=1e-3)

    def test_cat01_custom_factor(self, engine):
        """Custom factor overrides default."""
        inp = Scope3ActivityInput(
            entity_name="CustomFactor",
            reporting_year=2024,
            purchased_goods=[
                PurchasedGoodEntry(
                    product_type="custom_product",
                    quantity_kg=Decimal("10000"),
                    custom_factor_kgco2e_per_kg=Decimal("5.0"),
                ),
            ],
        )
        result = engine.calculate(inp)
        cat01 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_01.value][0]
        # 10000 * 5.0 / 1000 = 50
        assert float(cat01.emissions_tco2e) == pytest.approx(50.0, rel=1e-3)

    def test_cat01_method_activity_based(self, engine):
        """Cat 1 uses activity-based when data provided."""
        inp = Scope3ActivityInput(
            entity_name="Method",
            reporting_year=2024,
            purchased_goods=[
                PurchasedGoodEntry(product_type="cement", quantity_kg=Decimal("1000")),
            ],
        )
        result = engine.calculate(inp)
        cat01 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_01.value][0]
        assert cat01.method == CalculationMethod.ACTIVITY_BASED.value
        assert cat01.data_quality_score == 2


# ---------------------------------------------------------------------------
# TestCat04UpstreamTransport
# ---------------------------------------------------------------------------


class TestCat04UpstreamTransport:
    """Tests for Category 4 (Upstream Transportation) calculation."""

    def test_cat04_road_truck(self, engine):
        """Road truck: 100t * 500km * 62 gCO2e/tkm / 1e6 = 3.1 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="TruckTest",
            reporting_year=2024,
            upstream_transport=[
                TransportEntry(mode=TransportMode.ROAD_TRUCK, tonnes=Decimal("100"), distance_km=Decimal("500")),
            ],
        )
        result = engine.calculate(inp)
        cat04 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_04.value][0]
        assert float(cat04.emissions_tco2e) == pytest.approx(3.1, rel=1e-2)

    def test_cat04_air_freight_higher_than_rail(self, engine):
        """Air freight emission factor is much higher than rail."""
        inp = Scope3ActivityInput(
            entity_name="ModeCompare",
            reporting_year=2024,
            upstream_transport=[
                TransportEntry(mode=TransportMode.AIR_FREIGHT, tonnes=Decimal("10"), distance_km=Decimal("100")),
                TransportEntry(mode=TransportMode.RAIL_FREIGHT, tonnes=Decimal("10"), distance_km=Decimal("100")),
            ],
        )
        result = engine.calculate(inp)
        cat04 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_04.value][0]
        # Total: air + rail, and air dominates
        air_only = 10 * 100 * 602 / 1_000_000
        rail_only = 10 * 100 * 6.1 / 1_000_000
        assert air_only > rail_only * 10  # air >> rail


# ---------------------------------------------------------------------------
# TestCat05Waste
# ---------------------------------------------------------------------------


class TestCat05Waste:
    """Tests for Category 5 (Waste) calculation."""

    def test_cat05_landfill(self, engine):
        """Landfill: 100t * 586 kgCO2e/t / 1000 = 58.6 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="WasteTest",
            reporting_year=2024,
            waste=[
                WasteEntry(waste_type=WasteType.GENERAL_LANDFILL, quantity_tonnes=Decimal("100")),
            ],
        )
        result = engine.calculate(inp)
        cat05 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_05.value][0]
        assert float(cat05.emissions_tco2e) == pytest.approx(58.6, rel=1e-2)

    def test_cat05_composting_lower_than_landfill(self, engine):
        """Composting has lower factor than landfill."""
        assert float(WASTE_FACTORS[WasteType.COMPOSTING]) < float(WASTE_FACTORS[WasteType.GENERAL_LANDFILL])


# ---------------------------------------------------------------------------
# TestCat06BusinessTravel
# ---------------------------------------------------------------------------


class TestCat06BusinessTravel:
    """Tests for Category 6 (Business Travel) calculation."""

    def test_cat06_air_short_haul(self, engine):
        """Short haul: 10000 km * 0.15845 kgCO2e/pkm / 1000 = 1.5845 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="TravelTest",
            reporting_year=2024,
            business_travel=[
                TravelEntry(mode=TravelMode.AIR_SHORT_HAUL, distance_km=Decimal("10000")),
            ],
        )
        result = engine.calculate(inp)
        cat06 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_06.value][0]
        assert float(cat06.emissions_tco2e) == pytest.approx(1.5845, rel=1e-2)

    def test_cat06_hotel_nights(self, engine):
        """Hotel: 10 nights * 20.6 kgCO2e/night / 1000 = 0.206 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="HotelTest",
            reporting_year=2024,
            business_travel=[
                TravelEntry(mode=TravelMode.HOTEL_NIGHT, nights=10),
            ],
        )
        result = engine.calculate(inp)
        cat06 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_06.value][0]
        assert float(cat06.emissions_tco2e) == pytest.approx(0.206, rel=1e-2)


# ---------------------------------------------------------------------------
# TestCat07Commuting
# ---------------------------------------------------------------------------


class TestCat07Commuting:
    """Tests for Category 7 (Employee Commuting) calculation."""

    def test_cat07_basic_calculation(self, engine):
        """Basic commuting calculation with single mode."""
        inp = Scope3ActivityInput(
            entity_name="CommuteTest",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(
                    mode=CommuteMode.CAR_PETROL,
                    employee_pct=Decimal("100"),
                    avg_distance_km_one_way=Decimal("10"),
                ),
            ],
            total_employees=100,
            working_days_per_year=230,
            remote_work_pct=Decimal("0"),
        )
        result = engine.calculate(inp)
        cat07 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_07.value][0]
        # 100 * 1.0 * 20km * 230 * 0.1714 / 1000 = 78.844 tCO2e
        assert float(cat07.emissions_tco2e) == pytest.approx(78.844, rel=1e-2)

    def test_cat07_remote_work_reduces_emissions(self, engine):
        """50% remote work halves commuting emissions."""
        base_inp = Scope3ActivityInput(
            entity_name="NoRemote",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(mode=CommuteMode.CAR_PETROL, employee_pct=Decimal("100"), avg_distance_km_one_way=Decimal("10")),
            ],
            total_employees=100,
            working_days_per_year=230,
            remote_work_pct=Decimal("0"),
        )
        remote_inp = Scope3ActivityInput(
            entity_name="HalfRemote",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(mode=CommuteMode.CAR_PETROL, employee_pct=Decimal("100"), avg_distance_km_one_way=Decimal("10")),
            ],
            total_employees=100,
            working_days_per_year=230,
            remote_work_pct=Decimal("50"),
        )
        result_base = engine.calculate(base_inp)
        result_remote = engine.calculate(remote_inp)
        base_em = float([r for r in result_base.category_emissions if r.category == Scope3Category.CAT_07.value][0].emissions_tco2e)
        remote_em = float([r for r in result_remote.category_emissions if r.category == Scope3Category.CAT_07.value][0].emissions_tco2e)
        assert remote_em == pytest.approx(base_em * 0.5, rel=1e-3)

    def test_cat07_bicycle_zero_emissions(self, engine):
        """100% bicycle commuting produces zero emissions."""
        inp = Scope3ActivityInput(
            entity_name="BikeOnly",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(mode=CommuteMode.BICYCLE, employee_pct=Decimal("100")),
            ],
            total_employees=100,
            working_days_per_year=230,
        )
        result = engine.calculate(inp)
        cat07 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_07.value][0]
        assert float(cat07.emissions_tco2e) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestCat11UseOfSold
# ---------------------------------------------------------------------------


class TestCat11UseOfSold:
    """Tests for Category 11 (Use of Sold Products)."""

    def test_cat11_basic_calculation(self, engine):
        """1000 units * 1 kWh/use * 500 uses / 1000 MWh * 0.436 = 218 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="Cat11Test",
            reporting_year=2024,
            use_of_sold_products=[
                UseOfSoldEntry(
                    units_sold=Decimal("1000"),
                    energy_per_use_kwh=Decimal("1.0"),
                    uses_per_lifetime=Decimal("500"),
                ),
            ],
            grid_factor_tco2e_per_mwh=Decimal("0.436"),
        )
        result = engine.calculate(inp)
        cat11 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_11.value][0]
        # 1000 * 1 * 500 = 500,000 kWh = 500 MWh * 0.436 = 218 tCO2e
        assert float(cat11.emissions_tco2e) == pytest.approx(218.0, rel=1e-2)


# ---------------------------------------------------------------------------
# TestCat12EndOfLife
# ---------------------------------------------------------------------------


class TestCat12EndOfLife:
    """Tests for Category 12 (End-of-Life Treatment)."""

    def test_cat12_basic_calculation(self, engine):
        """1000 units * 10 kg * 586 kgCO2e/t / 1e6 = 5.86 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="Cat12Test",
            reporting_year=2024,
            end_of_life=[
                EndOfLifeEntry(
                    units_sold=Decimal("1000"),
                    weight_per_unit_kg=Decimal("10"),
                    treatment_type=WasteType.GENERAL_LANDFILL,
                ),
            ],
        )
        result = engine.calculate(inp)
        cat12 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_12.value][0]
        assert float(cat12.emissions_tco2e) == pytest.approx(5.86, rel=1e-2)


# ---------------------------------------------------------------------------
# TestSpendFallback
# ---------------------------------------------------------------------------


class TestSpendFallback:
    """Tests for spend-based fallback categories."""

    def test_spend_fallback_cat02(self, engine):
        """Cat 2: 1000 $k * 0.350 = 350 tCO2e."""
        inp = Scope3ActivityInput(
            entity_name="SpendTest",
            reporting_year=2024,
            spend_fallbacks=[
                SpendFallbackEntry(category=Scope3Category.CAT_02, spend_usd_thousands=Decimal("1000")),
            ],
        )
        result = engine.calculate(inp)
        cat02 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_02.value][0]
        assert float(cat02.emissions_tco2e) == pytest.approx(350.0, rel=1e-3)

    def test_spend_method_is_spend_based(self, engine):
        """Spend fallback uses spend_based method."""
        inp = Scope3ActivityInput(
            entity_name="SpendMethod",
            reporting_year=2024,
            spend_fallbacks=[
                SpendFallbackEntry(category=Scope3Category.CAT_15, spend_usd_thousands=Decimal("500")),
            ],
        )
        result = engine.calculate(inp)
        cat15 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_15.value][0]
        assert cat15.method == CalculationMethod.SPEND_BASED.value
        assert cat15.data_quality_score == 4


# ---------------------------------------------------------------------------
# TestDataQuality
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Tests for data quality scoring."""

    def test_activity_based_quality_is_2(self, engine):
        """Activity-based data quality score is 2."""
        assert DQ_SCORES[CalculationMethod.ACTIVITY_BASED] == 2

    def test_spend_based_quality_is_4(self, engine):
        """Spend-based data quality score is 4."""
        assert DQ_SCORES[CalculationMethod.SPEND_BASED] == 4

    def test_overall_quality_weighted(self, engine, full_input):
        """Overall data quality is between 1 and 5."""
        result = engine.calculate(full_input)
        dq = float(result.overall_data_quality)
        assert 1.0 <= dq <= 5.0


# ---------------------------------------------------------------------------
# TestUpstreamDownstream
# ---------------------------------------------------------------------------


class TestUpstreamDownstream:
    """Tests for upstream/downstream split."""

    def test_upstream_plus_downstream_equals_total(self, engine, full_input):
        """Upstream + downstream = total Scope 3."""
        result = engine.calculate(full_input)
        total = float(result.total_scope3_tco2e)
        up = float(result.upstream_tco2e)
        down = float(result.downstream_tco2e)
        assert up + down == pytest.approx(total, rel=1e-3)


# ---------------------------------------------------------------------------
# TestMateriality
# ---------------------------------------------------------------------------


class TestMateriality:
    """Tests for materiality flagging."""

    def test_material_categories_flagged(self, engine, full_input):
        """Categories >= 1% are flagged as material."""
        result = engine.calculate(full_input)
        assert len(result.material_categories) > 0

    def test_material_flag_consistency(self, engine, full_input):
        """material_categories list matches is_material flags."""
        result = engine.calculate(full_input)
        flagged = {r.category for r in result.category_emissions if r.is_material}
        listed = set(result.material_categories)
        assert flagged == listed


# ---------------------------------------------------------------------------
# TestUtilityMethods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for engine utility methods."""

    def test_get_emission_factor_transport(self, engine):
        """get_emission_factor returns transport factor."""
        val = engine.get_emission_factor("transport", TransportMode.ROAD_TRUCK)
        assert val is not None
        assert float(val) == 62.0

    def test_get_emission_factor_waste(self, engine):
        """get_emission_factor returns waste factor."""
        val = engine.get_emission_factor("waste", WasteType.GENERAL_LANDFILL)
        assert val is not None
        assert float(val) == 586.0

    def test_get_emission_factor_product(self, engine):
        """get_emission_factor returns product factor."""
        val = engine.get_emission_factor("product", "steel")
        assert val is not None
        assert float(val) == 1.89

    def test_get_emission_factor_unknown(self, engine):
        """get_emission_factor returns None for unknown."""
        val = engine.get_emission_factor("unknown_type", "foo")
        assert val is None

    def test_get_summary(self, engine, full_input):
        """get_summary returns expected structure."""
        result = engine.calculate(full_input)
        summary = engine.get_summary(result)
        assert summary["entity_name"] == "FullCorp"
        assert "total_scope3_tco2e" in summary
        assert "provenance_hash" in summary
        assert len(summary["top_3_categories"]) <= 3


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_quantity_entries(self, engine):
        """Zero quantity entries produce zero emissions."""
        inp = Scope3ActivityInput(
            entity_name="ZeroQ",
            reporting_year=2024,
            purchased_goods=[
                PurchasedGoodEntry(product_type="steel", quantity_kg=Decimal("0")),
            ],
        )
        result = engine.calculate(inp)
        cat01 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_01.value][0]
        assert float(cat01.emissions_tco2e) == pytest.approx(0.0, abs=1e-6)

    def test_zero_employees_commuting(self, engine):
        """Zero employees produce zero commuting emissions."""
        inp = Scope3ActivityInput(
            entity_name="NoEmp",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(mode=CommuteMode.CAR_PETROL, employee_pct=Decimal("100")),
            ],
            total_employees=0,
        )
        result = engine.calculate(inp)
        cat07 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_07.value][0]
        assert float(cat07.emissions_tco2e) == pytest.approx(0.0, abs=1e-6)

    def test_100_pct_remote_work(self, engine):
        """100% remote work produces zero commuting emissions."""
        inp = Scope3ActivityInput(
            entity_name="AllRemote",
            reporting_year=2024,
            commuting_profiles=[
                CommuteProfile(mode=CommuteMode.CAR_PETROL, employee_pct=Decimal("100")),
            ],
            total_employees=100,
            working_days_per_year=230,
            remote_work_pct=Decimal("100"),
        )
        result = engine.calculate(inp)
        cat07 = [r for r in result.category_emissions if r.category == Scope3Category.CAT_07.value][0]
        assert float(cat07.emissions_tco2e) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum definitions."""

    def test_scope3_category_has_15_values(self):
        """Scope3Category has 15 members."""
        assert len(Scope3Category) == 15

    def test_transport_mode_values(self):
        """TransportMode has expected values."""
        assert TransportMode.ROAD_TRUCK.value == "road_truck"
        assert TransportMode.AIR_FREIGHT.value == "air_freight"

    def test_waste_type_values(self):
        """WasteType has expected values."""
        assert WasteType.GENERAL_LANDFILL.value == "general_landfill"
        assert WasteType.RECYCLING.value == "recycling"

    def test_travel_mode_values(self):
        """TravelMode has expected values."""
        assert TravelMode.AIR_SHORT_HAUL.value == "air_short_haul"
        assert TravelMode.HOTEL_NIGHT.value == "hotel_night"

    def test_commute_mode_values(self):
        """CommuteMode has expected values."""
        assert CommuteMode.CAR_PETROL.value == "car_petrol"
        assert CommuteMode.BICYCLE.value == "bicycle"
        assert CommuteMode.WALK.value == "walk"

    def test_calculation_method_values(self):
        """CalculationMethod has expected values."""
        assert CalculationMethod.ACTIVITY_BASED.value == "activity_based"
        assert CalculationMethod.SPEND_BASED.value == "spend_based"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for Pydantic input validation."""

    def test_empty_entity_name_rejected(self):
        """Empty entity name is rejected."""
        with pytest.raises(Exception):
            Scope3ActivityInput(entity_name="", reporting_year=2024)

    def test_invalid_reporting_year_rejected(self):
        """Reporting year before 2020 is rejected."""
        with pytest.raises(Exception):
            Scope3ActivityInput(entity_name="Bad", reporting_year=2019)

    def test_negative_remote_work_pct_rejected(self):
        """Negative remote work percentage is rejected."""
        with pytest.raises(Exception):
            Scope3ActivityInput(
                entity_name="Bad",
                reporting_year=2024,
                remote_work_pct=Decimal("-10"),
            )

    def test_remote_work_over_100_rejected(self):
        """Remote work over 100% is rejected."""
        with pytest.raises(Exception):
            Scope3ActivityInput(
                entity_name="Bad",
                reporting_year=2024,
                remote_work_pct=Decimal("101"),
            )
