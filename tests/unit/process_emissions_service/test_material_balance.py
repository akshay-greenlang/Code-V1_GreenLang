# -*- coding: utf-8 -*-
"""
Unit tests for MaterialBalanceEngine - AGENT-MRV-004 Process Emissions Agent

Tests the carbon mass balance engine implementing IPCC 2006 Vol 3 Tier 2/3
methodology.  Validates calculate_carbon_balance(), get_carbonate_emissions(),
calculate_by_product_credits(), verify_mass_balance(), get_clinker_ratio(),
get_material_summary(), and process-specific balance implementations.

Verifies against known reference values (e.g., 1000t CaCO3 x 0.440 = 440 tCO2).

55 tests across 9 test classes.

Author: GreenLang QA Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.process_emissions.material_balance import (
    MaterialBalanceEngine,
    MaterialInput,
    CarbonBalance,
    ByProductCredit,
    MaterialSummary,
    CARBONATE_CO2_FACTORS,
    DEFAULT_CARBON_CONTENT,
    CARBON_TO_CO2_EXACT,
)


# =========================================================================
# Decimal helper
# =========================================================================

_D = Decimal


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def engine() -> MaterialBalanceEngine:
    """Create a MaterialBalanceEngine for testing."""
    return MaterialBalanceEngine(precision=8)


@pytest.fixture
def limestone_input() -> MaterialInput:
    """Create a standard limestone material input (100,000 tonnes)."""
    return MaterialInput(
        material_id="ls-001",
        material_type="LIMESTONE",
        quantity=_D("100000"),
        carbon_content=_D("0.120"),
        fraction_oxidized=_D("1.0"),
    )


@pytest.fixture
def clinker_output() -> MaterialInput:
    """Create a standard clinker output (62,000 tonnes)."""
    return MaterialInput(
        material_id="clk-001",
        material_type="CLINKER",
        quantity=_D("62000"),
        carbon_content=_D("0.000"),
        is_product=True,
    )


@pytest.fixture
def coke_input() -> MaterialInput:
    """Create a coke input for iron/steel (500 tonnes)."""
    return MaterialInput(
        material_id="coke-001",
        material_type="COKE",
        quantity=_D("500"),
        carbon_content=_D("0.850"),
        fraction_oxidized=_D("1.0"),
    )


# =========================================================================
# TestCarbonBalance (15 tests)
# =========================================================================

class TestCarbonBalance:
    """Tests for calculate_carbon_balance() with various material combinations."""

    def test_single_input_no_output(
        self, engine: MaterialBalanceEngine, limestone_input: MaterialInput
    ):
        """100,000t limestone at 12% C -> 12,000 tC -> ~44,000 tCO2."""
        balance = engine.calculate_carbon_balance([limestone_input])
        assert isinstance(balance, CarbonBalance)
        assert balance.total_carbon_input_tonnes == _D("12000.00000000")
        assert balance.total_carbon_output_tonnes == _D("0")
        # CO2 = 12000 * 44/12 = 44000
        assert balance.co2_emissions_tonnes > _D("43999")
        assert balance.co2_emissions_tonnes < _D("44001")

    def test_input_with_product_output(
        self, engine: MaterialBalanceEngine,
        limestone_input: MaterialInput, clinker_output: MaterialInput
    ):
        """Limestone input with clinker output reduces emissions."""
        balance = engine.calculate_carbon_balance(
            [limestone_input, clinker_output]
        )
        # C_in=12000, C_out=0 (clinker 0% carbon), net=12000
        assert balance.total_carbon_input_tonnes == _D("12000.00000000")
        assert balance.total_carbon_output_tonnes == _D("0")

    def test_product_with_carbon_content(self, engine: MaterialBalanceEngine):
        """Product with non-zero carbon content reduces net emissions."""
        inputs = [
            MaterialInput(
                material_id="coal-001",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.750"),
            ),
            MaterialInput(
                material_id="steel-001",
                material_type="SCRAP_METAL",
                quantity=_D("5000"),
                carbon_content=_D("0.005"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # C_in = 1000*0.75 = 750, C_out = 5000*0.005 = 25
        # net = 725, CO2 = 725 * 44/12 = ~2656
        assert balance.net_carbon_emissions_tonnes == _D("725.00000000")
        assert balance.co2_emissions_tonnes > _D("2655")

    def test_carbon_stock_change_reduces_emissions(
        self, engine: MaterialBalanceEngine
    ):
        """Positive carbon_stock_change reduces net emissions."""
        inputs = [
            MaterialInput(
                material_id="coal-001",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.750"),
            ),
        ]
        balance = engine.calculate_carbon_balance(
            inputs, carbon_stock_change=_D("100")
        )
        # C_in=750, stock_change=100, net=650
        assert balance.net_carbon_emissions_tonnes == _D("650.00000000")

    def test_negative_net_carbon_clamped_to_zero(
        self, engine: MaterialBalanceEngine
    ):
        """Negative net carbon (carbon sequestration) clamps CO2 to zero."""
        inputs = [
            MaterialInput(
                material_id="input-001",
                material_type="OTHER",
                quantity=_D("100"),
                carbon_content=_D("0.01"),
            ),
            MaterialInput(
                material_id="product-001",
                material_type="OTHER",
                quantity=_D("10000"),
                carbon_content=_D("0.10"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        assert balance.co2_emissions_tonnes == _D("0")

    def test_multiple_inputs_summed(self, engine: MaterialBalanceEngine):
        """Multiple inputs have their carbon contributions summed."""
        inputs = [
            MaterialInput(
                material_id="a",
                material_type="COKE",
                quantity=_D("300"),
                carbon_content=_D("0.850"),
            ),
            MaterialInput(
                material_id="b",
                material_type="COAL",
                quantity=_D("200"),
                carbon_content=_D("0.750"),
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # C_in = 300*0.85 + 200*0.75 = 255 + 150 = 405
        assert balance.total_carbon_input_tonnes == _D("405.00000000")

    def test_fraction_oxidized_less_than_one(self, engine: MaterialBalanceEngine):
        """Fraction oxidized < 1.0 reduces effective carbon input."""
        inputs = [
            MaterialInput(
                material_id="a",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.750"),
                fraction_oxidized=_D("0.80"),
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # C_in = 1000 * 0.75 * 0.80 = 600
        assert balance.total_carbon_input_tonnes == _D("600.00000000")

    def test_moisture_reduces_dry_mass(self, engine: MaterialBalanceEngine):
        """Moisture content reduces effective dry mass."""
        inputs = [
            MaterialInput(
                material_id="a",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.750"),
                moisture_content=_D("0.10"),
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # dry_mass = 1000 * (1-0.10) = 900
        # C_in = 900 * 0.75 * 1.0 = 675
        assert balance.total_carbon_input_tonnes == _D("675.00000000")

    def test_default_carbon_content_used(self, engine: MaterialBalanceEngine):
        """Default carbon content is applied when not specified (sentinel -1)."""
        inputs = [
            MaterialInput(
                material_id="a",
                material_type="COKE",
                quantity=_D("1000"),
                # carbon_content defaults to -1 (sentinel), will use 0.850
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # Default COKE carbon = 0.850
        assert balance.total_carbon_input_tonnes == _D("850.00000000")

    def test_provenance_hash_present(self, engine: MaterialBalanceEngine):
        """CarbonBalance result includes a provenance hash."""
        inputs = [
            MaterialInput(
                material_id="a",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.75"),
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        assert balance.provenance_hash != ""
        assert len(balance.provenance_hash) == 64

    def test_calculation_trace_non_empty(self, engine: MaterialBalanceEngine):
        """CarbonBalance result includes a non-empty trace."""
        inputs = [
            MaterialInput(material_id="a", material_type="COAL",
                          quantity=_D("100"), carbon_content=_D("0.75")),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        assert len(balance.calculation_trace) > 0

    def test_timestamp_present(self, engine: MaterialBalanceEngine):
        """CarbonBalance result includes a timestamp."""
        inputs = [
            MaterialInput(material_id="a", material_type="COAL",
                          quantity=_D("100"), carbon_content=_D("0.75")),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        assert balance.timestamp != ""

    def test_material_details_populated(self, engine: MaterialBalanceEngine):
        """material_details list has one entry per material."""
        inputs = [
            MaterialInput(material_id="a", material_type="COAL",
                          quantity=_D("100"), carbon_content=_D("0.75")),
            MaterialInput(material_id="b", material_type="IRON_ORE",
                          quantity=_D("200"), carbon_content=_D("0.00"),
                          is_product=True),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        assert len(balance.material_details) == 2

    def test_co2_c_ratio_exact(self, engine: MaterialBalanceEngine):
        """CO2 = net_carbon * 44/12 exactly."""
        inputs = [
            MaterialInput(
                material_id="pure_c",
                material_type="OTHER",
                quantity=_D("12"),
                carbon_content=_D("1.000"),
            ),
        ]
        balance = engine.calculate_carbon_balance(inputs)
        # net_C = 12, CO2 = 12 * 44/12 = 44.0
        assert balance.co2_emissions_tonnes == _D("44.00000000")

    def test_empty_materials_raises(self, engine: MaterialBalanceEngine):
        """Empty materials list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_carbon_balance([])


# =========================================================================
# TestCarbonateEmissions (10 tests)
# =========================================================================

class TestCarbonateEmissions:
    """Tests for get_carbonate_emissions() with different carbonate types."""

    def test_calcite_1000t(self, engine: MaterialBalanceEngine):
        """1000t CaCO3 (100% carbonate, calcite) x 0.440 = 440 tCO2."""
        materials = [
            MaterialInput(
                material_id="calcite-001",
                material_type="CALCIUM_CARBONATE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="CALCITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("440.00000000")

    def test_dolomite_1000t(self, engine: MaterialBalanceEngine):
        """1000t dolomite x 0.477 = 477 tCO2."""
        materials = [
            MaterialInput(
                material_id="dol-001",
                material_type="DOLOMITE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="DOLOMITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("477.00000000")

    def test_magnesite_1000t(self, engine: MaterialBalanceEngine):
        """1000t MgCO3 x 0.522 = 522 tCO2."""
        materials = [
            MaterialInput(
                material_id="mag-001",
                material_type="MAGNESIUM_CARBONATE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="MAGNESITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("522.00000000")

    def test_siderite_1000t(self, engine: MaterialBalanceEngine):
        """1000t FeCO3 x 0.380 = 380 tCO2."""
        materials = [
            MaterialInput(
                material_id="sid-001",
                material_type="IRON_CARBONATE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="SIDERITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("380.00000000")

    def test_ankerite_1000t(self, engine: MaterialBalanceEngine):
        """1000t ankerite x 0.407 = 407 tCO2."""
        materials = [
            MaterialInput(
                material_id="ank-001",
                material_type="ANKERITE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="ANKERITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("407.00000000")

    def test_partial_calcination(self, engine: MaterialBalanceEngine):
        """Partial calcination (50%) halves emissions."""
        materials = [
            MaterialInput(
                material_id="calc-partial",
                material_type="CALCIUM_CARBONATE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="CALCITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials, fraction_calcined=_D("0.5"))
        assert co2 == _D("220.00000000")

    def test_partial_carbonate_content(self, engine: MaterialBalanceEngine):
        """Material with 50% carbonate content halves CO2."""
        materials = [
            MaterialInput(
                material_id="half-carb",
                material_type="LIMESTONE_RAW",
                quantity=_D("1000"),
                carbonate_content=_D("0.50"),
                carbonate_type="CALCITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        # 1000 * 0.50 * 0.440 = 220
        assert co2 == _D("220.00000000")

    def test_no_carbonate_material_zero_emissions(
        self, engine: MaterialBalanceEngine
    ):
        """Material with zero carbonate_content produces zero emissions."""
        materials = [
            MaterialInput(
                material_id="no-carb",
                material_type="IRON_ORE",
                quantity=_D("1000"),
                carbonate_content=_D("0"),
                carbonate_type="CALCITE",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("0")

    def test_invalid_fraction_calcined_raises(
        self, engine: MaterialBalanceEngine
    ):
        """fraction_calcined outside [0,1] raises ValueError."""
        materials = [
            MaterialInput(
                material_id="bad-fc",
                material_type="CALCIUM_CARBONATE",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="CALCITE",
            ),
        ]
        with pytest.raises(ValueError, match="fraction_calcined"):
            engine.get_carbonate_emissions(materials, fraction_calcined=_D("1.5"))

    def test_unknown_carbonate_type_skipped(self, engine: MaterialBalanceEngine):
        """Unknown carbonate_type is silently skipped (with warning log)."""
        materials = [
            MaterialInput(
                material_id="mystery",
                material_type="OTHER",
                quantity=_D("1000"),
                carbonate_content=_D("1.0"),
                carbonate_type="UNOBTAINIUM",
            ),
        ]
        co2 = engine.get_carbonate_emissions(materials)
        assert co2 == _D("0")


# =========================================================================
# TestByProductCredits (8 tests)
# =========================================================================

class TestByProductCredits:
    """Tests for calculate_by_product_credits()."""

    def test_single_by_product(self, engine: MaterialBalanceEngine):
        """By-product with 5% carbon produces CO2 credit."""
        by_products = [
            MaterialInput(
                material_id="slag-001",
                material_type="SLAG",
                quantity=_D("1000"),
                carbon_content=_D("0.05"),
                is_by_product=True,
            ),
        ]
        credits = engine.calculate_by_product_credits(by_products)
        assert len(credits) == 1
        # credit = 1000 * 0.05 * 44/12 = 50 * 3.667 = ~183.3
        assert credits[0].credit_co2_tonnes > _D("183")
        assert credits[0].credit_co2_tonnes < _D("184")

    def test_zero_carbon_by_product(self, engine: MaterialBalanceEngine):
        """By-product with 0% carbon has zero credit."""
        by_products = [
            MaterialInput(
                material_id="slag-zero",
                material_type="SLAG",
                quantity=_D("1000"),
                carbon_content=_D("0"),
                is_by_product=True,
            ),
        ]
        credits = engine.calculate_by_product_credits(by_products)
        assert len(credits) == 1
        assert credits[0].credit_co2_tonnes == _D("0")

    def test_multiple_by_products(self, engine: MaterialBalanceEngine):
        """Multiple by-products each get a separate credit."""
        by_products = [
            MaterialInput(
                material_id="bp-001", material_type="SLAG",
                quantity=_D("500"), carbon_content=_D("0.03"),
                is_by_product=True,
            ),
            MaterialInput(
                material_id="bp-002", material_type="DUST",
                quantity=_D("200"), carbon_content=_D("0.01"),
                is_by_product=True,
            ),
        ]
        credits = engine.calculate_by_product_credits(by_products)
        assert len(credits) == 2
        total = sum(c.credit_co2_tonnes for c in credits)
        assert total > _D("0")

    def test_non_by_product_skipped(self, engine: MaterialBalanceEngine):
        """Materials not marked as by-product are skipped."""
        materials = [
            MaterialInput(
                material_id="not-bp",
                material_type="COAL",
                quantity=_D("1000"),
                carbon_content=_D("0.75"),
                is_by_product=False,
            ),
        ]
        credits = engine.calculate_by_product_credits(materials)
        assert len(credits) == 0

    def test_by_product_credit_type_populated(self, engine: MaterialBalanceEngine):
        """ByProductCredit.by_product_type is populated."""
        bp = MaterialInput(
            material_id="bp-type", material_type="BLAST_FURNACE_GAS",
            quantity=_D("100"), carbon_content=_D("0.10"),
            is_by_product=True,
        )
        credits = engine.calculate_by_product_credits([bp])
        assert credits[0].by_product_type == "BLAST_FURNACE_GAS"

    def test_by_product_moisture_adjusted(self, engine: MaterialBalanceEngine):
        """By-product with moisture has reduced effective mass."""
        bp = MaterialInput(
            material_id="wet-bp", material_type="SLAG",
            quantity=_D("1000"), carbon_content=_D("0.05"),
            moisture_content=_D("0.20"),
            is_by_product=True,
        )
        credits = engine.calculate_by_product_credits([bp])
        # dry_mass = 1000 * 0.80 = 800
        # credit = 800 * 0.05 * 44/12 = 40 * 3.667 = ~146.7
        assert credits[0].quantity_tonnes == _D("800.00000000")

    def test_empty_by_products_list(self, engine: MaterialBalanceEngine):
        """Empty by-products list returns empty credits."""
        credits = engine.calculate_by_product_credits([])
        assert credits == []

    def test_by_product_credit_co2_formula(self, engine: MaterialBalanceEngine):
        """Credit = quantity x carbon_content x 44/12."""
        bp = MaterialInput(
            material_id="exact-bp", material_type="TEST",
            quantity=_D("12"),  # exactly 12 tonnes
            carbon_content=_D("1.0"),  # 100% carbon
            is_by_product=True,
        )
        credits = engine.calculate_by_product_credits([bp])
        # 12 * 1.0 * 44/12 = 44.0 exactly
        assert credits[0].credit_co2_tonnes == _D("44.00000000")


# =========================================================================
# TestMassBalanceVerification (5 tests)
# =========================================================================

class TestMassBalanceVerification:
    """Tests for verify_mass_balance() pass/fail checking."""

    def test_closed_balance_passes(self, engine: MaterialBalanceEngine):
        """Mass balance within 5% tolerance passes."""
        materials = [
            MaterialInput(
                material_id="in-1", material_type="COAL",
                quantity=_D("1000"), carbon_content=_D("0.75"),
            ),
            MaterialInput(
                material_id="out-1", material_type="STEEL",
                quantity=_D("960"), carbon_content=_D("0.005"),
                is_product=True,
            ),
        ]
        passed, msg = engine.verify_mass_balance(materials, tolerance=_D("0.05"))
        assert passed is True
        assert "PASSED" in msg

    def test_open_balance_fails(self, engine: MaterialBalanceEngine):
        """Mass balance exceeding tolerance fails."""
        materials = [
            MaterialInput(
                material_id="in-1", material_type="COAL",
                quantity=_D("1000"), carbon_content=_D("0.75"),
            ),
            MaterialInput(
                material_id="out-1", material_type="PRODUCT",
                quantity=_D("500"), carbon_content=_D("0"),
                is_product=True,
            ),
        ]
        passed, msg = engine.verify_mass_balance(materials, tolerance=_D("0.05"))
        assert passed is False
        assert "FAILED" in msg

    def test_no_input_mass_fails(self, engine: MaterialBalanceEngine):
        """No input mass produces a failure."""
        materials = [
            MaterialInput(
                material_id="out-1", material_type="PRODUCT",
                quantity=_D("1000"), carbon_content=_D("0"),
                is_product=True,
            ),
        ]
        passed, msg = engine.verify_mass_balance(materials)
        assert passed is False

    def test_zero_tolerance_exact_match_required(
        self, engine: MaterialBalanceEngine
    ):
        """Zero tolerance requires exact mass balance."""
        materials = [
            MaterialInput(
                material_id="in-1", material_type="COAL",
                quantity=_D("1000"), carbon_content=_D("0.75"),
            ),
            MaterialInput(
                material_id="out-1", material_type="PRODUCT",
                quantity=_D("1000"), carbon_content=_D("0"),
                is_product=True,
            ),
        ]
        passed, msg = engine.verify_mass_balance(materials, tolerance=_D("0"))
        assert passed is True

    def test_invalid_tolerance_raises(self, engine: MaterialBalanceEngine):
        """Tolerance outside [0,1] raises ValueError."""
        materials = [
            MaterialInput(
                material_id="in-1", material_type="COAL",
                quantity=_D("100"), carbon_content=_D("0.75"),
            ),
        ]
        with pytest.raises(ValueError, match="tolerance"):
            engine.verify_mass_balance(materials, tolerance=_D("1.5"))


# =========================================================================
# TestClinkerRatio (5 tests)
# =========================================================================

class TestClinkerRatio:
    """Tests for get_clinker_ratio() cement-specific calculations."""

    def test_standard_clinker_ratio(self, engine: MaterialBalanceEngine):
        """95,000t clinker / 100,000t cement = 0.95."""
        ratio = engine.get_clinker_ratio(
            cement_tonnes=_D("100000"),
            clinker_tonnes=_D("95000"),
        )
        assert ratio == _D("0.95000000")

    def test_blended_cement_ratio(self, engine: MaterialBalanceEngine):
        """70,000t clinker / 100,000t cement = 0.70."""
        ratio = engine.get_clinker_ratio(
            cement_tonnes=_D("100000"),
            clinker_tonnes=_D("70000"),
        )
        assert ratio == _D("0.70000000")

    def test_zero_cement_raises(self, engine: MaterialBalanceEngine):
        """Zero cement_tonnes raises ValueError."""
        with pytest.raises(ValueError, match="cement_tonnes"):
            engine.get_clinker_ratio(_D("0"), _D("100"))

    def test_negative_clinker_raises(self, engine: MaterialBalanceEngine):
        """Negative clinker_tonnes raises ValueError."""
        with pytest.raises(ValueError, match="clinker_tonnes"):
            engine.get_clinker_ratio(_D("100"), _D("-10"))

    def test_ratio_out_of_range_raises(self, engine: MaterialBalanceEngine):
        """Clinker ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_clinker_ratio(
                cement_tonnes=_D("1000"),
                clinker_tonnes=_D("2000"),  # ratio = 2.0
            )


# =========================================================================
# TestCKDCorrection (3 tests)
# =========================================================================

class TestCKDCorrection:
    """Tests for cement kiln dust (CKD) correction factor."""

    def test_default_ckd_factor(self, engine: MaterialBalanceEngine):
        """Default CKD correction factor is 1.02 (2%)."""
        from greenlang.process_emissions.material_balance import (
            CKD_CORRECTION_FACTOR_DEFAULT,
        )
        assert CKD_CORRECTION_FACTOR_DEFAULT == _D("1.02")

    def test_clinker_to_cement_ratio_default(self, engine: MaterialBalanceEngine):
        """Default clinker-to-cement ratio is 0.95."""
        from greenlang.process_emissions.material_balance import (
            CLINKER_TO_CEMENT_RATIO_DEFAULT,
        )
        assert CLINKER_TO_CEMENT_RATIO_DEFAULT == _D("0.95")

    def test_clinker_ratio_bounds(self, engine: MaterialBalanceEngine):
        """Clinker ratio min=0.50, max=1.00."""
        from greenlang.process_emissions.material_balance import (
            CLINKER_TO_CEMENT_RATIO_MIN,
            CLINKER_TO_CEMENT_RATIO_MAX,
        )
        assert CLINKER_TO_CEMENT_RATIO_MIN == _D("0.50")
        assert CLINKER_TO_CEMENT_RATIO_MAX == _D("1.00")


# =========================================================================
# TestProcessSpecificBalance (10 tests)
# =========================================================================

class TestProcessSpecificBalance:
    """Tests for cement, iron/steel, aluminum, ammonia specific logic."""

    def test_cement_balance_basic(self, engine: MaterialBalanceEngine):
        """Cement balance with limestone produces CO2 from calcination."""
        materials = [
            MaterialInput(
                material_id="ls-1",
                material_type="LIMESTONE",
                quantity=_D("50000"),
                carbon_content=_D("0.120"),
            ),
            MaterialInput(
                material_id="clk-1",
                material_type="CLINKER",
                quantity=_D("30000"),
                carbon_content=_D("0.000"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 50000*0.12 = 6000, C_out = 0
        assert balance.total_carbon_input_tonnes == _D("6000.00000000")
        assert balance.co2_emissions_tonnes > _D("0")

    def test_iron_steel_bf_bof_balance(self, engine: MaterialBalanceEngine):
        """BF-BOF balance with coke and scrap steel."""
        materials = [
            MaterialInput(
                material_id="coke-bf",
                material_type="COKE",
                quantity=_D("600"),
                carbon_content=_D("0.850"),
            ),
            MaterialInput(
                material_id="iron-ore",
                material_type="IRON_ORE",
                quantity=_D("1500"),
                carbon_content=_D("0.000"),
            ),
            MaterialInput(
                material_id="steel-out",
                material_type="SCRAP_METAL",
                quantity=_D("1000"),
                carbon_content=_D("0.005"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 600*0.85 + 1500*0 = 510, C_out = 1000*0.005 = 5
        # net = 505, CO2 = 505 * 44/12 = ~1851
        assert balance.total_carbon_input_tonnes == _D("510.00000000")
        assert balance.net_carbon_emissions_tonnes == _D("505.00000000")

    def test_aluminum_anode_carbon_balance(self, engine: MaterialBalanceEngine):
        """Aluminum balance with anode carbon consumption."""
        materials = [
            MaterialInput(
                material_id="anode-001",
                material_type="ANODE_CARBON",
                quantity=_D("420"),
                carbon_content=_D("0.850"),
            ),
            MaterialInput(
                material_id="al-out",
                material_type="ALUMINA",
                quantity=_D("1000"),
                carbon_content=_D("0.000"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 420 * 0.85 = 357
        assert balance.total_carbon_input_tonnes == _D("357.00000000")

    def test_ammonia_smr_balance(self, engine: MaterialBalanceEngine):
        """Ammonia SMR natural gas feedstock carbon balance."""
        materials = [
            MaterialInput(
                material_id="ng-feed",
                material_type="NATURAL_GAS_FEEDSTOCK",
                quantity=_D("1000"),
                carbon_content=_D("0.730"),
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 1000 * 0.73 = 730
        assert balance.total_carbon_input_tonnes == _D("730.00000000")

    def test_eaf_scrap_balance(self, engine: MaterialBalanceEngine):
        """EAF scrap steel balance (low carbon intensity)."""
        materials = [
            MaterialInput(
                material_id="scrap-001",
                material_type="SCRAP_METAL",
                quantity=_D("1000"),
                carbon_content=_D("0.005"),
            ),
            MaterialInput(
                material_id="electrode",
                material_type="ELECTRODE_CARBON",
                quantity=_D("5"),
                carbon_content=_D("0.990"),
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 1000*0.005 + 5*0.99 = 5 + 4.95 = 9.95
        assert balance.total_carbon_input_tonnes == _D("9.95000000")

    def test_petrochemical_naphtha_balance(self, engine: MaterialBalanceEngine):
        """Naphtha cracker carbon balance with product carbon retention."""
        materials = [
            MaterialInput(
                material_id="naphtha-001",
                material_type="NAPHTHA",
                quantity=_D("1000"),
                carbon_content=_D("0.836"),
            ),
            MaterialInput(
                material_id="ethylene-out",
                material_type="OTHER",
                quantity=_D("300"),
                carbon_content=_D("0.857"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 1000*0.836 = 836, C_out = 300*0.857 = 257.1
        # net = 578.9
        assert balance.net_carbon_emissions_tonnes > _D("578")

    def test_dri_natural_gas_balance(self, engine: MaterialBalanceEngine):
        """DRI natural gas route balance."""
        materials = [
            MaterialInput(
                material_id="ng-dri",
                material_type="NATURAL_GAS_FEEDSTOCK",
                quantity=_D("200"),
                carbon_content=_D("0.730"),
            ),
            MaterialInput(
                material_id="dri-out",
                material_type="IRON_ORE",
                quantity=_D("1000"),
                carbon_content=_D("0.010"),
                is_product=True,
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 200*0.73 = 146, C_out = 1000*0.01 = 10, net = 136
        assert balance.net_carbon_emissions_tonnes == _D("136.00000000")

    def test_glass_mixed_carbonates(self, engine: MaterialBalanceEngine):
        """Glass production with multiple carbonate raw materials."""
        materials = [
            MaterialInput(
                material_id="soda-ash",
                material_type="OTHER",
                quantity=_D("200"),
                carbon_content=_D("0.113"),
            ),
            MaterialInput(
                material_id="limestone",
                material_type="LIMESTONE",
                quantity=_D("100"),
                carbon_content=_D("0.120"),
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 200*0.113 + 100*0.12 = 22.6 + 12 = 34.6
        assert balance.total_carbon_input_tonnes == _D("34.60000000")

    def test_carbide_production_balance(self, engine: MaterialBalanceEngine):
        """Calcium carbide production from lime and coke."""
        materials = [
            MaterialInput(
                material_id="lime-in",
                material_type="CALCIUM_OXIDE",
                quantity=_D("100"),
                carbon_content=_D("0.000"),
            ),
            MaterialInput(
                material_id="coke-in",
                material_type="COKE",
                quantity=_D("80"),
                carbon_content=_D("0.850"),
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        # C_in = 0 + 80*0.85 = 68
        assert balance.total_carbon_input_tonnes == _D("68.00000000")


# =========================================================================
# TestMaterialSummary (5 tests)
# =========================================================================

class TestMaterialSummary:
    """Tests for get_material_summary()."""

    def test_summary_counts(self, engine: MaterialBalanceEngine):
        """Summary correctly counts inputs and outputs."""
        materials = [
            MaterialInput(material_id="in-1", material_type="COAL",
                          quantity=_D("1000"), carbon_content=_D("0.75")),
            MaterialInput(material_id="in-2", material_type="COKE",
                          quantity=_D("500"), carbon_content=_D("0.85")),
            MaterialInput(material_id="out-1", material_type="PRODUCT",
                          quantity=_D("800"), carbon_content=_D("0.01"),
                          is_product=True),
            MaterialInput(material_id="bp-1", material_type="SLAG",
                          quantity=_D("200"), carbon_content=_D("0.02"),
                          is_by_product=True),
        ]
        summary = engine.get_material_summary(materials)
        assert summary.input_count == 2
        assert summary.output_count == 2  # product + by-product
        assert summary.by_product_count == 1

    def test_summary_mass_totals(self, engine: MaterialBalanceEngine):
        """Summary totals input and output mass correctly."""
        materials = [
            MaterialInput(material_id="in-1", material_type="COAL",
                          quantity=_D("1000"), carbon_content=_D("0.75")),
            MaterialInput(material_id="out-1", material_type="PRODUCT",
                          quantity=_D("950"), carbon_content=_D("0"),
                          is_product=True),
        ]
        summary = engine.get_material_summary(materials)
        assert summary.total_input_mass_tonnes == _D("1000.00000000")
        assert summary.total_output_mass_tonnes == _D("950.00000000")

    def test_summary_residual(self, engine: MaterialBalanceEngine):
        """Mass balance residual = input - output."""
        materials = [
            MaterialInput(material_id="in-1", material_type="COAL",
                          quantity=_D("1000"), carbon_content=_D("0.75")),
            MaterialInput(material_id="out-1", material_type="PRODUCT",
                          quantity=_D("900"), carbon_content=_D("0"),
                          is_product=True),
        ]
        summary = engine.get_material_summary(materials)
        assert summary.mass_balance_residual_tonnes == _D("100.00000000")

    def test_summary_dominant_types(self, engine: MaterialBalanceEngine):
        """Dominant types reflect largest mass flows."""
        materials = [
            MaterialInput(material_id="in-1", material_type="COAL",
                          quantity=_D("800"), carbon_content=_D("0.75")),
            MaterialInput(material_id="in-2", material_type="COKE",
                          quantity=_D("200"), carbon_content=_D("0.85")),
            MaterialInput(material_id="out-1", material_type="STEEL",
                          quantity=_D("900"), carbon_content=_D("0.005"),
                          is_product=True),
        ]
        summary = engine.get_material_summary(materials)
        assert summary.dominant_input_type == "COAL"
        assert summary.dominant_output_type == "STEEL"

    def test_summary_provenance_hash(self, engine: MaterialBalanceEngine):
        """Summary includes a SHA-256 provenance hash."""
        materials = [
            MaterialInput(material_id="in-1", material_type="COAL",
                          quantity=_D("100"), carbon_content=_D("0.75")),
        ]
        summary = engine.get_material_summary(materials)
        assert len(summary.provenance_hash) == 64


# =========================================================================
# TestEdgeCases (5 tests)
# =========================================================================

class TestEdgeCasesMaterialBalance:
    """Tests for edge cases: empty inputs, zero quantities."""

    def test_all_zero_quantities(self, engine: MaterialBalanceEngine):
        """Materials with all zero quantities produce zero emissions."""
        materials = [
            MaterialInput(material_id="z-1", material_type="COAL",
                          quantity=_D("0"), carbon_content=_D("0.75")),
        ]
        balance = engine.calculate_carbon_balance(materials)
        assert balance.co2_emissions_tonnes == _D("0")

    def test_very_large_quantity(self, engine: MaterialBalanceEngine):
        """Very large quantities compute without overflow."""
        materials = [
            MaterialInput(
                material_id="big-1",
                material_type="COAL",
                quantity=_D("1000000000"),
                carbon_content=_D("0.750"),
            ),
        ]
        balance = engine.calculate_carbon_balance(materials)
        assert balance.co2_emissions_tonnes > _D("0")

    def test_track_materials_basic(self, engine: MaterialBalanceEngine):
        """track_materials() registers materials and returns IDs."""
        materials = [
            MaterialInput(material_id="tm-1", material_type="COAL",
                          quantity=_D("100"), carbon_content=_D("0.75")),
            MaterialInput(material_id="tm-2", material_type="COKE",
                          quantity=_D("200"), carbon_content=_D("0.85")),
        ]
        result = engine.track_materials(materials)
        assert result["registered_count"] == 2
        assert len(result["material_ids"]) == 2

    def test_track_materials_negative_quantity_raises(
        self, engine: MaterialBalanceEngine
    ):
        """Negative material quantity raises ValueError."""
        materials = [
            MaterialInput(material_id="neg-1", material_type="COAL",
                          quantity=_D("-100"), carbon_content=_D("0.75")),
        ]
        with pytest.raises(ValueError, match="negative quantity"):
            engine.track_materials(materials)

    def test_constants_match_documentation(self, engine: MaterialBalanceEngine):
        """CARBON_TO_CO2 constant matches 44/12."""
        expected = _D("44") / _D("12")
        assert CARBON_TO_CO2_EXACT == expected
