# -*- coding: utf-8 -*-
"""
Unit tests for EmissionCalculatorEngine - AGENT-MRV-004 Process Emissions Agent

Tests all four calculation methodologies (Emission Factor, Mass Balance,
Stoichiometric, Direct Measurement), GWP application, abatement application,
batch calculation, decimal precision, and edge cases.

Validates calculations against known reference values (e.g., 1000 tonnes
clinker * 0.507 = 507 tCO2).

50 tests across 9 test classes.

Author: GreenLang QA Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.emission_calculator import EmissionCalculatorEngine
from greenlang.process_emissions.process_database import ProcessDatabaseEngine


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def db() -> ProcessDatabaseEngine:
    """Create a ProcessDatabaseEngine with provenance disabled."""
    return ProcessDatabaseEngine(config={"enable_provenance": False})


@pytest.fixture
def calc(db: ProcessDatabaseEngine) -> EmissionCalculatorEngine:
    """Create an EmissionCalculatorEngine with provenance disabled."""
    return EmissionCalculatorEngine(
        process_database=db,
        config={"enable_provenance": False},
    )


# =========================================================================
# TestEmissionFactorMethod (15 tests) - Tier 1
# =========================================================================

class TestEmissionFactorMethod:
    """Tests for Tier 1 emission factor calculations."""

    def test_cement_1000_tonnes_clinker(self, calc: EmissionCalculatorEngine):
        """1000 tonnes clinker x 0.507 tCO2/t = 507 tCO2."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        # CEMENT IPCC EF is 0.507 tCO2/t clinker
        # CO2 GWP AR6 = 1, so co2e = 507 tonnes
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        # May include CKD correction (default 2%), so: 507 * 1.02 = 517.14
        # Without CKD: 507.0
        assert co2_gas["raw_emissions_tonnes"] == Decimal("507.00000000")

    def test_cement_100000_tonnes_clinker(self, calc: EmissionCalculatorEngine):
        """100,000 tonnes clinker x 0.507 = 50,700 tCO2 (before CKD)."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("100000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["raw_emissions_tonnes"] == Decimal("50700.00000000")

    def test_lime_10000_tonnes(self, calc: EmissionCalculatorEngine):
        """10,000 tonnes CaO x 0.785 tCO2/t = 7850 tCO2."""
        result = calc.calculate(
            process_type="LIME",
            method="EMISSION_FACTOR",
            activity_data=Decimal("10000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["raw_emissions_tonnes"] == Decimal("7850.00000000")

    def test_ammonia_5000_tonnes(self, calc: EmissionCalculatorEngine):
        """5000 tonnes NH3 x 1.500 tCO2/t = 7500 tCO2."""
        result = calc.calculate(
            process_type="AMMONIA",
            method="EMISSION_FACTOR",
            activity_data=Decimal("5000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["raw_emissions_tonnes"] == Decimal("7500.00000000")

    def test_iron_steel_bf_bof_1000_tonnes(self, calc: EmissionCalculatorEngine):
        """1000t steel BF-BOF x 1.900 = 1900 tCO2 raw."""
        result = calc.calculate(
            process_type="IRON_STEEL_BF_BOF",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["raw_emissions_tonnes"] == Decimal("1900.00000000")

    def test_iron_steel_eaf_1000_tonnes(self, calc: EmissionCalculatorEngine):
        """1000t steel EAF x 0.400 = 400 tCO2."""
        result = calc.calculate(
            process_type="IRON_STEEL_EAF",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["raw_emissions_tonnes"] == Decimal("400.00000000")

    def test_aluminum_prebake_produces_pfc_emissions(
        self, calc: EmissionCalculatorEngine
    ):
        """Aluminum prebake includes CO2, CF4, and C2F6 in results."""
        result = calc.calculate(
            process_type="ALUMINUM_PREBAKE",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        gases = {g["gas"] for g in result["gas_emissions"]}
        assert "CO2" in gases
        assert "CF4" in gases
        assert "C2F6" in gases

    def test_nitric_acid_n2o_emissions(self, calc: EmissionCalculatorEngine):
        """Nitric acid: 10000 t HNO3 x 0.007 tN2O/t = 70 tN2O."""
        result = calc.calculate(
            process_type="NITRIC_ACID",
            method="EMISSION_FACTOR",
            activity_data=Decimal("10000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        n2o_gas = [g for g in result["gas_emissions"] if g["gas"] == "N2O"][0]
        assert n2o_gas["raw_emissions_tonnes"] == Decimal("70.00000000")

    def test_adipic_acid_unabated_n2o(self, calc: EmissionCalculatorEngine):
        """Adipic acid unabated: 1000t x 0.300 tN2O/t = 300 tN2O."""
        result = calc.calculate(
            process_type="ADIPIC_ACID",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="IPCC",
        )
        assert result["status"] == "SUCCESS"
        n2o_gas = [g for g in result["gas_emissions"] if g["gas"] == "N2O"][0]
        assert n2o_gas["raw_emissions_tonnes"] == Decimal("300.00000000")

    def test_epa_source_factor(self, calc: EmissionCalculatorEngine):
        """EPA source returns different factor (0.510) than IPCC (0.507)."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            ef_source="EPA",
        )
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["emission_factor"] == Decimal("0.510")

    def test_custom_factor_override(self, calc: EmissionCalculatorEngine):
        """Custom factors override database values."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            custom_factors={"CO2": Decimal("0.600")},
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["emission_factor"] == Decimal("0.600")

    def test_zero_activity_data(self, calc: EmissionCalculatorEngine):
        """Zero activity data produces zero emissions."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] == Decimal("0")

    def test_negative_activity_data_raises(self, calc: EmissionCalculatorEngine):
        """Negative activity data raises ValueError or returns FAILED."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("-1000"),
        )
        # Either raises inside or returns FAILED status
        assert result["status"] == "FAILED"

    def test_result_has_provenance_hash(self, calc: EmissionCalculatorEngine):
        """Successful calculation includes a provenance hash."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
        )
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_result_has_calculation_trace(self, calc: EmissionCalculatorEngine):
        """Calculation trace includes at least one entry."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
        )
        assert len(result["calculation_trace"]) > 0


# =========================================================================
# TestMassBalanceMethod (10 tests) - Tier 2/3
# =========================================================================

class TestMassBalanceMethod:
    """Tests for mass balance carbon accounting calculations."""

    def test_simple_mass_balance_co2(self, calc: EmissionCalculatorEngine):
        """Simple mass balance: 1000t limestone (12% C) -> CO2."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "LIMESTONE",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.12"),
                "fraction_oxidized": Decimal("1.0"),
            }],
            carbon_outputs=[],
        )
        assert result["status"] == "SUCCESS"
        # C_in = 1000 * 0.12 * 1.0 = 120 tC
        # CO2 = 120 * 3.66417 = ~439.7 tonnes
        assert result["total_co2e_tonnes"] > Decimal("0")

    def test_mass_balance_with_outputs(self, calc: EmissionCalculatorEngine):
        """Mass balance with product carbon output reduces emissions."""
        result = calc.calculate(
            process_type="IRON_STEEL_BF_BOF",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "COKE",
                "quantity_tonnes": Decimal("500"),
                "carbon_content": Decimal("0.85"),
                "fraction_oxidized": Decimal("1.0"),
            }],
            carbon_outputs=[{
                "product_type": "STEEL",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.005"),
            }],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # C_in = 500*0.85 = 425, C_out = 1000*0.005 = 5
        # Net C = 420, CO2 = 420 * 3.66417 = ~1539 tonnes
        assert co2_gas["net_carbon_tonnes"] > Decimal("0")

    def test_mass_balance_with_stock_change(self, calc: EmissionCalculatorEngine):
        """Carbon stock change reduces net emissions."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "LIMESTONE",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.12"),
            }],
            carbon_outputs=[],
            carbon_stock_change=Decimal("20"),  # 20t C stored
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # C_in = 120, stock_change = 20, net_C = 100
        assert co2_gas["net_carbon_tonnes"] == pytest.approx(
            Decimal("100"), abs=Decimal("1")
        )

    def test_mass_balance_negative_net_clamped_to_zero(
        self, calc: EmissionCalculatorEngine
    ):
        """Negative net carbon (carbon accumulation) clamps to zero CO2."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "LIMESTONE",
                "quantity_tonnes": Decimal("100"),
                "carbon_content": Decimal("0.01"),
            }],
            carbon_outputs=[{
                "product_type": "CLINKER",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.10"),
            }],
        )
        assert result["status"] == "SUCCESS"
        # C_in=1, C_out=100 => net=-99, clamped to 0
        assert result["total_co2e_tonnes"] == Decimal("0")

    def test_mass_balance_empty_inputs_zero_emissions(
        self, calc: EmissionCalculatorEngine
    ):
        """Empty material inputs produce zero emissions."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[],
            carbon_outputs=[],
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] == Decimal("0")

    def test_mass_balance_with_abatement(self, calc: EmissionCalculatorEngine):
        """Abatement reduces mass balance emissions."""
        result_no_abate = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "LIMESTONE",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.12"),
            }],
            carbon_outputs=[],
        )
        result_with_abate = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "LIMESTONE",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.12"),
            }],
            carbon_outputs=[],
            abatement_efficiency=Decimal("0.50"),
        )
        assert result_with_abate["total_co2e_tonnes"] < result_no_abate["total_co2e_tonnes"]

    def test_mass_balance_multiple_inputs(self, calc: EmissionCalculatorEngine):
        """Multiple material inputs are summed correctly."""
        result = calc.calculate(
            process_type="IRON_STEEL_BF_BOF",
            method="MASS_BALANCE",
            material_inputs=[
                {
                    "material_type": "COKE",
                    "quantity_tonnes": Decimal("300"),
                    "carbon_content": Decimal("0.85"),
                },
                {
                    "material_type": "COAL",
                    "quantity_tonnes": Decimal("200"),
                    "carbon_content": Decimal("0.75"),
                },
            ],
            carbon_outputs=[],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # C_in = 300*0.85 + 200*0.75 = 255 + 150 = 405
        expected_carbon = Decimal("405")
        assert co2_gas["total_carbon_input_tonnes"] == pytest.approx(
            expected_carbon, abs=Decimal("1")
        )

    def test_mass_balance_fraction_oxidized(self, calc: EmissionCalculatorEngine):
        """Fraction oxidized < 1.0 reduces carbon input."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "COKE",
                "quantity_tonnes": Decimal("1000"),
                "carbon_content": Decimal("0.85"),
                "fraction_oxidized": Decimal("0.90"),
            }],
            carbon_outputs=[],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # C_in = 1000 * 0.85 * 0.90 = 765
        assert co2_gas["total_carbon_input_tonnes"] == pytest.approx(
            Decimal("765"), abs=Decimal("1")
        )

    def test_mass_balance_co2_c_ratio(self, calc: EmissionCalculatorEngine):
        """CO2 = net_carbon * 44/12 (3.66417)."""
        result = calc.calculate(
            process_type="CEMENT",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "COKE",
                "quantity_tonnes": Decimal("100"),
                "carbon_content": Decimal("1.000"),
                "fraction_oxidized": Decimal("1.0"),
            }],
            carbon_outputs=[],
        )
        co2_gas = result["gas_emissions"][0]
        # net_carbon = 100, CO2 = 100 * 3.66417 = 366.417
        assert co2_gas["raw_emissions_tonnes"] > Decimal("365")
        assert co2_gas["raw_emissions_tonnes"] < Decimal("368")

    def test_mass_balance_default_carbon_content(
        self, calc: EmissionCalculatorEngine
    ):
        """Mass balance uses default carbon content when not specified."""
        result = calc.calculate(
            process_type="IRON_STEEL_BF_BOF",
            method="MASS_BALANCE",
            material_inputs=[{
                "material_type": "COKE",
                "quantity_tonnes": Decimal("100"),
            }],
            carbon_outputs=[],
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] > Decimal("0")


# =========================================================================
# TestStoichiometricMethod (8 tests) - Tier 2
# =========================================================================

class TestStoichiometricMethod:
    """Tests for carbonate-based stoichiometric calculations."""

    def test_calcite_stoichiometric(self, calc: EmissionCalculatorEngine):
        """1000t CaCO3 x 0.440 = 440 tCO2."""
        result = calc.calculate(
            process_type="CEMENT",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "CALCITE",
                "quantity_tonnes": Decimal("1000"),
                "fraction_calcined": Decimal("1.0"),
                "purity": Decimal("1.0"),
            }],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # 1000 * 1.0 * 0.440 * 1.0 = 440 (using co2_factor)
        raw = co2_gas["raw_emissions_tonnes"]
        assert raw >= Decimal("439") and raw <= Decimal("441")

    def test_magnesite_stoichiometric(self, calc: EmissionCalculatorEngine):
        """1000t MgCO3 x 0.522 = 522 tCO2."""
        result = calc.calculate(
            process_type="LIME",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "MAGNESITE",
                "quantity_tonnes": Decimal("1000"),
                "fraction_calcined": Decimal("1.0"),
                "purity": Decimal("1.0"),
            }],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        raw = co2_gas["raw_emissions_tonnes"]
        assert raw >= Decimal("521") and raw <= Decimal("523")

    def test_dolomite_stoichiometric(self, calc: EmissionCalculatorEngine):
        """1000t dolomite x 0.477 = 477 tCO2."""
        result = calc.calculate(
            process_type="GLASS",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "DOLOMITE",
                "quantity_tonnes": Decimal("1000"),
                "fraction_calcined": Decimal("1.0"),
                "purity": Decimal("1.0"),
            }],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        raw = co2_gas["raw_emissions_tonnes"]
        assert raw >= Decimal("476") and raw <= Decimal("478")

    def test_partial_calcination(self, calc: EmissionCalculatorEngine):
        """Fraction calcined < 1.0 reduces emissions proportionally."""
        result = calc.calculate(
            process_type="CEMENT",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "CALCITE",
                "quantity_tonnes": Decimal("1000"),
                "fraction_calcined": Decimal("0.50"),
                "purity": Decimal("1.0"),
            }],
        )
        co2_gas = result["gas_emissions"][0]
        raw = co2_gas["raw_emissions_tonnes"]
        # 1000 * 1.0 * 0.440 * 0.50 = 220
        assert raw >= Decimal("219") and raw <= Decimal("221")

    def test_carbonate_purity_factor(self, calc: EmissionCalculatorEngine):
        """Purity < 1.0 reduces effective carbonate quantity."""
        result = calc.calculate(
            process_type="CEMENT",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "CALCITE",
                "quantity_tonnes": Decimal("1000"),
                "fraction_calcined": Decimal("1.0"),
                "purity": Decimal("0.90"),
            }],
        )
        co2_gas = result["gas_emissions"][0]
        raw = co2_gas["raw_emissions_tonnes"]
        # 1000 * 0.90 * 0.440 * 1.0 = 396
        assert raw >= Decimal("395") and raw <= Decimal("397")

    def test_multiple_carbonate_inputs(self, calc: EmissionCalculatorEngine):
        """Multiple carbonate types are summed correctly."""
        result = calc.calculate(
            process_type="GLASS",
            method="STOICHIOMETRIC",
            carbonate_inputs=[
                {
                    "carbonate_type": "CALCITE",
                    "quantity_tonnes": Decimal("500"),
                    "fraction_calcined": Decimal("1.0"),
                },
                {
                    "carbonate_type": "DOLOMITE",
                    "quantity_tonnes": Decimal("500"),
                    "fraction_calcined": Decimal("1.0"),
                },
            ],
        )
        assert result["status"] == "SUCCESS"
        co2_gas = result["gas_emissions"][0]
        # 500*0.440 + 500*0.477 = 220 + 238.5 = 458.5
        raw = co2_gas["raw_emissions_tonnes"]
        assert raw > Decimal("457") and raw < Decimal("460")

    def test_empty_carbonate_inputs(self, calc: EmissionCalculatorEngine):
        """Empty carbonate inputs produce zero emissions."""
        result = calc.calculate(
            process_type="CEMENT",
            method="STOICHIOMETRIC",
            carbonate_inputs=[],
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] == Decimal("0")

    def test_stoichiometric_with_abatement(self, calc: EmissionCalculatorEngine):
        """Abatement reduces stoichiometric emissions."""
        result = calc.calculate(
            process_type="CEMENT",
            method="STOICHIOMETRIC",
            carbonate_inputs=[{
                "carbonate_type": "CALCITE",
                "quantity_tonnes": Decimal("1000"),
            }],
            abatement_efficiency=Decimal("0.90"),
        )
        co2_gas = result["gas_emissions"][0]
        # Raw ~440, net = 440 * (1-0.90) = 44
        net = co2_gas["net_emissions_tonnes"]
        assert net < Decimal("50")


# =========================================================================
# TestDirectMeasurement (5 tests)
# =========================================================================

class TestDirectMeasurement:
    """Tests for direct measurement pass-through calculations."""

    def test_direct_co2_measurement(self, calc: EmissionCalculatorEngine):
        """Direct CO2 measurement passes through unchanged."""
        result = calc.calculate(
            process_type="CEMENT",
            method="DIRECT_MEASUREMENT",
            measured_emissions={"CO2": Decimal("50000")},
        )
        assert result["status"] == "SUCCESS"
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        assert co2_gas["measured_mass_tonnes"] == Decimal("50000")
        assert co2_gas["net_emissions_tonnes"] == Decimal("50000")

    def test_direct_n2o_measurement_with_gwp(self, calc: EmissionCalculatorEngine):
        """Direct N2O measurement applies GWP conversion."""
        result = calc.calculate(
            process_type="NITRIC_ACID",
            method="DIRECT_MEASUREMENT",
            measured_emissions={"N2O": Decimal("3.5")},
            gwp_source="AR6",
        )
        assert result["status"] == "SUCCESS"
        n2o_gas = [g for g in result["gas_emissions"] if g["gas"] == "N2O"][0]
        # 3.5 * 273 (AR6 N2O GWP) = 955.5 tCO2e
        assert n2o_gas["co2e_tonnes"] == Decimal("955.50000000")

    def test_direct_multiple_gases(self, calc: EmissionCalculatorEngine):
        """Direct measurement with multiple gases."""
        result = calc.calculate(
            process_type="ALUMINUM_PREBAKE",
            method="DIRECT_MEASUREMENT",
            measured_emissions={
                "CO2": Decimal("1500"),
                "CF4": Decimal("0.04"),
            },
        )
        assert result["status"] == "SUCCESS"
        gases = {g["gas"] for g in result["gas_emissions"]}
        assert "CO2" in gases
        assert "CF4" in gases

    def test_direct_negative_measurement_fails(
        self, calc: EmissionCalculatorEngine
    ):
        """Negative measured emissions raise an error."""
        result = calc.calculate(
            process_type="CEMENT",
            method="DIRECT_MEASUREMENT",
            measured_emissions={"CO2": Decimal("-100")},
        )
        assert result["status"] == "FAILED"

    def test_direct_empty_measurements(self, calc: EmissionCalculatorEngine):
        """Empty measured emissions dict produces zero total."""
        result = calc.calculate(
            process_type="CEMENT",
            method="DIRECT_MEASUREMENT",
            measured_emissions={},
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] == Decimal("0")


# =========================================================================
# TestGWPApplication (5 tests)
# =========================================================================

class TestGWPApplication:
    """Tests for apply_gwp() utility method."""

    def test_co2_gwp_is_one(self, calc: EmissionCalculatorEngine):
        """CO2 GWP is 1, so co2e equals mass."""
        result = calc.apply_gwp("CO2", Decimal("100"), gwp_source="AR6")
        assert result["co2e_tonnes"] == Decimal("100.00000000")

    def test_n2o_gwp_ar6(self, calc: EmissionCalculatorEngine):
        """N2O GWP AR6 = 273. 1 tonne N2O = 273 tCO2e."""
        result = calc.apply_gwp("N2O", Decimal("1"), gwp_source="AR6")
        assert result["co2e_tonnes"] == Decimal("273.00000000")

    def test_ch4_gwp_ar6(self, calc: EmissionCalculatorEngine):
        """CH4 GWP AR6 = 29.8. 10 tonnes = 298 tCO2e."""
        result = calc.apply_gwp("CH4", Decimal("10"), gwp_source="AR6")
        assert result["co2e_tonnes"] == Decimal("298.00000000")

    def test_sf6_gwp_ar6(self, calc: EmissionCalculatorEngine):
        """SF6 GWP AR6 = 25200. 0.001 tonnes = 25.2 tCO2e."""
        result = calc.apply_gwp("SF6", Decimal("0.001"), gwp_source="AR6")
        assert result["co2e_tonnes"] == Decimal("25.20000000")

    def test_apply_gwp_returns_correct_keys(self, calc: EmissionCalculatorEngine):
        """apply_gwp returns gas, mass_tonnes, gwp, co2e_kg, co2e_tonnes."""
        result = calc.apply_gwp("CO2", Decimal("100"), gwp_source="AR6")
        assert "gas" in result
        assert "mass_tonnes" in result
        assert "gwp" in result
        assert "co2e_kg" in result
        assert "co2e_tonnes" in result


# =========================================================================
# TestAbatementApplication (5 tests)
# =========================================================================

class TestAbatementApplication:
    """Tests for apply_abatement() utility method."""

    def test_90_percent_abatement(self, calc: EmissionCalculatorEngine):
        """90% abatement: 100t gross -> 10t net."""
        result = calc.apply_abatement(Decimal("100"), Decimal("0.90"))
        assert result["net_emissions_tonnes"] == Decimal("10.00000000")

    def test_zero_abatement(self, calc: EmissionCalculatorEngine):
        """0% abatement: net equals gross."""
        result = calc.apply_abatement(Decimal("100"), Decimal("0"))
        assert result["net_emissions_tonnes"] == Decimal("100.00000000")

    def test_100_percent_abatement(self, calc: EmissionCalculatorEngine):
        """100% abatement: net is zero."""
        result = calc.apply_abatement(Decimal("100"), Decimal("1.0"))
        assert result["net_emissions_tonnes"] == Decimal("0")

    def test_50_percent_abatement(self, calc: EmissionCalculatorEngine):
        """50% abatement: 200t -> 100t."""
        result = calc.apply_abatement(Decimal("200"), Decimal("0.50"))
        assert result["net_emissions_tonnes"] == Decimal("100.00000000")

    def test_abatement_returns_all_keys(self, calc: EmissionCalculatorEngine):
        """apply_abatement returns gross, abated, and net emissions."""
        result = calc.apply_abatement(Decimal("100"), Decimal("0.50"))
        assert "gross_emissions_tonnes" in result
        assert "abated_emissions_tonnes" in result
        assert "net_emissions_tonnes" in result
        assert "abatement_efficiency" in result


# =========================================================================
# TestBatchCalculation (5 tests)
# =========================================================================

class TestBatchCalculation:
    """Tests for calculate_batch() batch processing."""

    def test_batch_two_calculations(self, calc: EmissionCalculatorEngine):
        """Batch of two produces two results with correct totals."""
        batch = calc.calculate_batch([
            {"process_type": "CEMENT", "activity_data": Decimal("1000")},
            {"process_type": "LIME", "activity_data": Decimal("1000")},
        ])
        assert batch["total_count"] == 2
        assert batch["success_count"] == 2
        assert batch["failure_count"] == 0
        assert batch["total_co2e_tonnes"] > Decimal("0")

    def test_batch_with_failure(self, calc: EmissionCalculatorEngine):
        """Batch counts failures correctly when invalid data is present."""
        batch = calc.calculate_batch([
            {"process_type": "CEMENT", "activity_data": Decimal("1000")},
            {"process_type": "CEMENT", "method": "INVALID_METHOD"},
        ])
        assert batch["success_count"] == 1
        assert batch["failure_count"] == 1

    def test_batch_empty_list(self, calc: EmissionCalculatorEngine):
        """Empty batch produces zero counts and zero total."""
        batch = calc.calculate_batch([])
        assert batch["total_count"] == 0
        assert batch["total_co2e_tonnes"] == Decimal("0")

    def test_batch_gwp_override(self, calc: EmissionCalculatorEngine):
        """GWP override applies to all calculations in batch."""
        batch = calc.calculate_batch(
            [{"process_type": "CEMENT", "activity_data": Decimal("1000")}],
            gwp_source="AR5",
        )
        assert batch["success_count"] == 1

    def test_batch_processing_time(self, calc: EmissionCalculatorEngine):
        """Batch reports processing_time_ms."""
        batch = calc.calculate_batch([
            {"process_type": "CEMENT", "activity_data": Decimal("1000")},
        ])
        assert batch["processing_time_ms"] > 0


# =========================================================================
# TestDecimalPrecision (5 tests)
# =========================================================================

class TestDecimalPrecision:
    """Tests for 8+ decimal place precision in calculations."""

    def test_8_decimal_places(self, calc: EmissionCalculatorEngine):
        """Results are quantized to 8 decimal places."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1"),
        )
        co2_gas = [g for g in result["gas_emissions"] if g["gas"] == "CO2"][0]
        # 1 * 0.507 = 0.507 -> quantized to 0.50700000
        s = str(co2_gas["raw_emissions_tonnes"])
        if "." in s:
            decimals = len(s.split(".")[1])
            assert decimals == 8

    def test_no_floating_point_drift(self, calc: EmissionCalculatorEngine):
        """Repeated identical calculations produce identical results."""
        results = []
        for _ in range(5):
            r = calc.calculate(
                process_type="CEMENT",
                method="EMISSION_FACTOR",
                activity_data=Decimal("123456.789"),
            )
            results.append(r["total_co2e_tonnes"])
        assert all(r == results[0] for r in results)

    def test_very_small_value_precision(self, calc: EmissionCalculatorEngine):
        """Very small activity data retains precision."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("0.001"),
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] > Decimal("0")

    def test_very_large_value(self, calc: EmissionCalculatorEngine):
        """Very large activity data computes without overflow."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000000000"),
        )
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_tonnes"] > Decimal("0")

    def test_all_results_are_decimal(self, calc: EmissionCalculatorEngine):
        """All numeric results are Decimal, not float."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
        )
        assert isinstance(result["total_co2e_kg"], Decimal)
        assert isinstance(result["total_co2e_tonnes"], Decimal)
        for g in result["gas_emissions"]:
            assert isinstance(g["co2e_tonnes"], Decimal)


# =========================================================================
# TestEdgeCases (5 tests)
# =========================================================================

class TestEdgeCasesCalculator:
    """Tests for edge cases and error handling."""

    def test_invalid_method_returns_failed(self, calc: EmissionCalculatorEngine):
        """Invalid method name returns FAILED status."""
        result = calc.calculate(
            process_type="CEMENT",
            method="BOGUS_METHOD",
            activity_data=Decimal("1000"),
        )
        assert result["status"] == "FAILED"
        assert "error_message" in result

    def test_result_always_has_calculation_id(self, calc: EmissionCalculatorEngine):
        """Every result includes a calculation_id."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
        )
        assert "calculation_id" in result
        assert result["calculation_id"] != ""

    def test_custom_calculation_id(self, calc: EmissionCalculatorEngine):
        """Custom calculation_id is preserved in the result."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
            calculation_id="my_custom_id_001",
        )
        assert result["calculation_id"] == "my_custom_id_001"

    def test_processing_time_positive(self, calc: EmissionCalculatorEngine):
        """Processing time is always positive."""
        result = calc.calculate(
            process_type="CEMENT",
            method="EMISSION_FACTOR",
            activity_data=Decimal("1000"),
        )
        assert result["processing_time_ms"] > 0

    def test_failed_result_has_zero_emissions(self, calc: EmissionCalculatorEngine):
        """Failed calculation has zero CO2e totals."""
        result = calc.calculate(
            process_type="CEMENT",
            method="INVALID",
            activity_data=Decimal("1000"),
        )
        assert result["total_co2e_kg"] == Decimal("0")
        assert result["total_co2e_tonnes"] == Decimal("0")
