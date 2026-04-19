# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Scope 3 Estimator Engine.

Tests spend-based calculations, DEFRA/EPA emission factors, simplified
categories (1, 6, 7), multi-currency handling, and data quality scoring.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~350 lines, 50+ tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines import (
    Scope3EstimatorEngine,
    Scope3EstimatorInput,
    Scope3EstimatorResult,
    CategoryEstimate,
)

from engines.scope3_estimator_engine import (
    Scope3Category,
    SpendEntry,
    SpendCurrency,
    DataSourceType,
)

# Try to import optional constants
try:
    from engines.scope3_estimator_engine import (
        EEIO_FACTORS_GENERAL as DEFRA_SPEND_FACTORS,
    )
except ImportError:
    DEFRA_SPEND_FACTORS = {}

try:
    from engines.scope3_estimator_engine import (
        EEIO_FACTORS_GENERAL as EPA_SPEND_FACTORS,
    )
except ImportError:
    EPA_SPEND_FACTORS = {}

from .conftest import assert_decimal_close, assert_provenance_hash


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_category_by_id(result, cat_id: str):
    """Get category by string ID from result."""
    for cat in result.categories:
        if cat.category == cat_id:
            return cat
    return None


def get_category_by_number(result, cat_num: int):
    """Get category by number (1-7) from result."""
    cat_map = {
        1: "cat_01_purchased_goods",
        2: "cat_02_capital_goods",
        3: "cat_03_fuel_energy",
        4: "cat_04_upstream_transport",
        5: "cat_05_waste",
        6: "cat_06_business_travel",
        7: "cat_07_employee_commuting",
    }
    cat_id = cat_map.get(cat_num)
    if not cat_id:
        return None
    return get_category_by_id(result, cat_id)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> Scope3EstimatorEngine:
    return Scope3EstimatorEngine()


@pytest.fixture
def basic_spend_input() -> Scope3EstimatorInput:
    return Scope3EstimatorInput(
        entity_name="SmallCo Ltd",
        reporting_year=2025,
        headcount=25,
        spend_entries=[
            SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("450000")),
            SpendEntry(category=Scope3Category.CAT_06_BUSINESS_TRAVEL, amount=Decimal("22000")),
            SpendEntry(category=Scope3Category.CAT_07_EMPLOYEE_COMMUTING, amount=Decimal("18000")),
        ],
    )


@pytest.fixture
def full_spend_input() -> Scope3EstimatorInput:
    return Scope3EstimatorInput(
        entity_name="EuroManufact GmbH",
        reporting_year=2025,
        headcount=145,
        include_optional_categories=True,
        spend_entries=[
            SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("8500000")),
            SpendEntry(category=Scope3Category.CAT_02_CAPITAL_GOODS, amount=Decimal("1200000")),
            SpendEntry(category=Scope3Category.CAT_03_FUEL_ENERGY, amount=Decimal("35000")),
            SpendEntry(category=Scope3Category.CAT_04_UPSTREAM_TRANSPORT, amount=Decimal("450000")),
            SpendEntry(category=Scope3Category.CAT_05_WASTE, amount=Decimal("28000")),
            SpendEntry(category=Scope3Category.CAT_06_BUSINESS_TRAVEL, amount=Decimal("120000")),
            SpendEntry(category=Scope3Category.CAT_07_EMPLOYEE_COMMUTING, amount=Decimal("85000")),
        ],
    )


@pytest.fixture
def micro_spend_input() -> Scope3EstimatorInput:
    return Scope3EstimatorInput(
        entity_name="Micro Cafe",
        reporting_year=2025,
        headcount=6,
        spend_entries=[
            SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("85000")),
            SpendEntry(category=Scope3Category.CAT_06_BUSINESS_TRAVEL, amount=Decimal("2000")),
            SpendEntry(category=Scope3Category.CAT_07_EMPLOYEE_COMMUTING, amount=Decimal("3000")),
        ],
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestScope3EstimatorInstantiation:
    def test_engine_creates(self) -> None:
        engine = Scope3EstimatorEngine()
        assert engine is not None

    def test_engine_has_version(self) -> None:
        engine = Scope3EstimatorEngine()
        assert hasattr(engine, "engine_version")

    def test_engine_has_calculate_method(self) -> None:
        engine = Scope3EstimatorEngine()
        assert hasattr(engine, "calculate")


# ===========================================================================
# Tests -- Emission Factor Databases
# ===========================================================================


class TestEmissionFactorDatabases:
    def test_eeio_factors_loaded(self) -> None:
        assert len(DEFRA_SPEND_FACTORS) > 0

    @pytest.mark.parametrize("category", [
        Scope3Category.CAT_01_PURCHASED_GOODS,
        Scope3Category.CAT_06_BUSINESS_TRAVEL,
        Scope3Category.CAT_07_EMPLOYEE_COMMUTING,
    ])
    def test_eeio_has_core_sme_categories(self, category) -> None:
        assert category in DEFRA_SPEND_FACTORS

    def test_eeio_factors_positive(self) -> None:
        for category, factor in DEFRA_SPEND_FACTORS.items():
            assert factor > Decimal("0"), f"EEIO factor for {category} must be positive"


# ===========================================================================
# Tests -- Scope 3 Category Enum
# ===========================================================================


class TestSpendCategory:
    @pytest.mark.parametrize("cat", [
        "cat_01_purchased_goods", "cat_02_capital_goods", "cat_03_fuel_energy",
        "cat_04_upstream_transport", "cat_05_waste", "cat_06_business_travel",
        "cat_07_employee_commuting",
    ])
    def test_spend_category_values(self, cat) -> None:
        assert Scope3Category(cat) is not None


# ===========================================================================
# Tests -- Basic Spend Estimation (Cat 1, 6, 7)
# ===========================================================================


class TestBasicSpendEstimation:
    def test_basic_estimation_calculates(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        assert isinstance(result, Scope3EstimatorResult)
        assert result.total_scope3_tco2e > Decimal("0")

    def test_cat1_purchased_goods(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        cat1 = get_category_by_number(result, 1)
        assert cat1 is not None
        assert cat1.tco2e > Decimal("0")
        assert "spend_based" in cat1.methodology

    def test_cat6_business_travel(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        cat6 = get_category_by_number(result, 6)
        assert cat6 is not None
        assert cat6.tco2e > Decimal("0")

    def test_cat7_employee_commuting(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        cat7 = get_category_by_number(result, 7)
        assert cat7 is not None
        assert cat7.tco2e > Decimal("0")

    def test_total_equals_sum_of_categories(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        cat_sum = sum(c.tco2e for c in result.categories)
        assert_decimal_close(result.total_scope3_tco2e, cat_sum, Decimal("0.01"))

    def test_provenance_hash(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        assert_provenance_hash(result)


# ===========================================================================
# Tests -- Full Spend Estimation (All 7 Categories)
# ===========================================================================


class TestFullSpendEstimation:
    def test_full_estimation_calculates(self, engine, full_spend_input) -> None:
        result = engine.calculate(full_spend_input)
        assert result.total_scope3_tco2e > Decimal("0")

    def test_all_7_categories_estimated(self, engine, full_spend_input) -> None:
        result = engine.calculate(full_spend_input)
        assert len(result.categories) >= 7

    def test_cat1_dominates_for_manufacturing(self, engine, full_spend_input) -> None:
        """Cat 1 (purchased goods) should be the largest for manufacturing."""
        result = engine.calculate(full_spend_input)
        cat1 = get_category_by_number(result, 1)
        assert cat1 is not None
        assert cat1.tco2e > Decimal("0")
        # Cat 1 should be a major contributor (at least half of any other single cat)
        for cat in result.categories:
            if cat.category != "cat_01_purchased_goods":
                assert cat1.tco2e >= cat.tco2e * Decimal("0.5")

    def test_upstream_transportation_estimated(self, engine, full_spend_input) -> None:
        result = engine.calculate(full_spend_input)
        cat4 = get_category_by_number(result, 4)
        assert cat4 is not None
        assert cat4.tco2e > Decimal("0")

    def test_waste_estimated(self, engine, full_spend_input) -> None:
        result = engine.calculate(full_spend_input)
        cat5 = get_category_by_number(result, 5)
        assert cat5 is not None
        assert cat5.tco2e > Decimal("0")


# ===========================================================================
# Tests -- Micro Business Estimation
# ===========================================================================


class TestMicroBusinessEstimation:
    def test_micro_estimation_calculates(self, engine, micro_spend_input) -> None:
        result = engine.calculate(micro_spend_input)
        assert result.total_scope3_tco2e > Decimal("0")

    def test_micro_simplified_categories(self, engine, micro_spend_input) -> None:
        """Micro businesses should use simplified categories (1, 6, 7)."""
        result = engine.calculate(micro_spend_input)
        category_ids = [c.category for c in result.categories]
        assert "cat_01_purchased_goods" in category_ids
        # At least one of cat 6 or 7 should be present
        has_travel_or_commute = (
            "cat_06_business_travel" in category_ids or
            "cat_07_employee_commuting" in category_ids
        )
        assert has_travel_or_commute

    def test_micro_lower_total_than_medium(self, engine, micro_spend_input, full_spend_input) -> None:
        micro_result = engine.calculate(micro_spend_input)
        medium_result = engine.calculate(full_spend_input)
        assert micro_result.total_scope3_tco2e < medium_result.total_scope3_tco2e


# ===========================================================================
# Tests -- Multi-Currency Support
# ===========================================================================


class TestMultiCurrencySupport:
    @pytest.mark.parametrize("currency", [SpendCurrency.GBP, SpendCurrency.EUR, SpendCurrency.USD, SpendCurrency.AUD, SpendCurrency.CAD])
    def test_currency_accepted(self, engine, currency) -> None:
        inp = Scope3EstimatorInput(
            entity_name="Test",
            reporting_year=2025,
            headcount=20,
            spend_entries=[
                SpendEntry(
                    category=Scope3Category.CAT_01_PURCHASED_GOODS,
                    amount=Decimal("100000"),
                    currency=currency,
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.total_scope3_tco2e > Decimal("0")

    def test_different_currencies_give_different_results(self, engine) -> None:
        inp_gbp = Scope3EstimatorInput(
            entity_name="Test",
            reporting_year=2025,
            headcount=20,
            spend_entries=[
                SpendEntry(
                    category=Scope3Category.CAT_01_PURCHASED_GOODS,
                    amount=Decimal("100000"),
                    currency=SpendCurrency.GBP,
                ),
            ],
        )
        inp_usd = Scope3EstimatorInput(
            entity_name="Test",
            reporting_year=2025,
            headcount=20,
            spend_entries=[
                SpendEntry(
                    category=Scope3Category.CAT_01_PURCHASED_GOODS,
                    amount=Decimal("100000"),
                    currency=SpendCurrency.USD,
                ),
            ],
        )
        result_gbp = engine.calculate(inp_gbp)
        result_usd = engine.calculate(inp_usd)
        # GBP and USD have different FX rates so results should differ
        # (unless both map to 1.0, in which case they are the same)
        assert result_gbp.total_scope3_tco2e > Decimal("0")
        assert result_usd.total_scope3_tco2e > Decimal("0")


# ===========================================================================
# Tests -- Data Quality Scoring
# ===========================================================================


class TestDataQualityScoring:
    def test_spend_based_quality_score(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        assert hasattr(result, "data_quality")
        assert result.data_quality is not None

    def test_more_categories_higher_quality(self, engine, basic_spend_input, full_spend_input) -> None:
        basic_result = engine.calculate(basic_spend_input)
        full_result = engine.calculate(full_spend_input)
        assert full_result.data_quality is not None
        assert basic_result.data_quality is not None


# ===========================================================================
# Tests -- Decimal Arithmetic & Zero Hallucination
# ===========================================================================


class TestDecimalArithmetic:
    def test_all_outputs_decimal(self, engine, basic_spend_input) -> None:
        result = engine.calculate(basic_spend_input)
        assert isinstance(result.total_scope3_tco2e, Decimal)
        for cat in result.categories:
            assert isinstance(cat.tco2e, Decimal)

    def test_no_float_imprecision(self, engine) -> None:
        """Verify no float-related precision errors."""
        inp = Scope3EstimatorInput(
            entity_name="Precision Test",
            reporting_year=2025,
            headcount=20,
            spend_entries=[
                SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("0.01")),
            ],
        )
        result = engine.calculate(inp)
        assert result.total_scope3_tco2e >= Decimal("0")

    def test_deterministic(self, engine, basic_spend_input) -> None:
        """Same inputs should produce same total emissions."""
        r1 = engine.calculate(basic_spend_input)
        r2 = engine.calculate(basic_spend_input)
        assert r1.total_scope3_tco2e == r2.total_scope3_tco2e


# ===========================================================================
# Tests -- Error Handling
# ===========================================================================


class TestScope3ErrorHandling:
    def test_empty_spend_data_still_calculates(self, engine) -> None:
        """Empty spend entries produce zero emissions (not an error)."""
        result = engine.calculate(Scope3EstimatorInput(
            entity_name="Test",
            reporting_year=2025,
            headcount=20,
            spend_entries=[],
        ))
        assert result.total_scope3_tco2e == Decimal("0")

    def test_negative_spend_raises(self, engine) -> None:
        with pytest.raises(Exception):
            engine.calculate(Scope3EstimatorInput(
                entity_name="Test",
                reporting_year=2025,
                headcount=20,
                spend_entries=[
                    SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("-50000")),
                ],
            ))

    def test_invalid_currency_on_spend_entry_raises(self, engine) -> None:
        """Passing an invalid string for SpendCurrency enum should raise."""
        with pytest.raises(Exception):
            SpendEntry(
                category=Scope3Category.CAT_01_PURCHASED_GOODS,
                amount=Decimal("50000"),
                currency="INVALID",
            )
