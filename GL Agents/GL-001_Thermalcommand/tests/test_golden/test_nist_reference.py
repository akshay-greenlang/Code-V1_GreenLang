"""
GL-001 ThermalCommand - NIST Reference Validation Tests

Golden value tests using NIST-traceable reference data for validation
of all critical calculations. These tests ensure regulatory compliance
by validating against authoritative reference values.

Reference Sources:
- NIST Chemistry WebBook (https://webbook.nist.gov)
- NIST Special Publication 811 (SI Units)
- NIST Standard Reference Data Program
- ASME PTC 4.1-2013 Reference Tables
- EPA 40 CFR Part 98 Table C-1 and C-2

Test Categories:
1. Unit Conversion Validation (NIST SP 811)
2. Steam Properties Validation (NIST/ASME)
3. Emission Factor Validation (EPA 40 CFR 98)
4. GWP Value Validation (IPCC AR4/AR5)
5. Efficiency Calculation Validation (ASME PTC 4.1)

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import pytest
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# NIST REFERENCE DATA
# =============================================================================

# NIST SP 811 Unit Conversion Reference Values
# Source: https://www.nist.gov/pml/special-publication-811
NIST_UNIT_CONVERSIONS = {
    "energy": [
        # (value, from_unit, to_unit, expected_result, tolerance_percent)
        (Decimal("1.0"), "btu", "kj", Decimal("1.05505585262"), Decimal("0.0001")),
        (Decimal("1.0"), "kwh", "btu", Decimal("3412.14163312794"), Decimal("0.0001")),
        (Decimal("1.0"), "mwh", "mmbtu", Decimal("3.412141633"), Decimal("0.0001")),
        (Decimal("1000.0"), "btu", "kj", Decimal("1055.05585262"), Decimal("0.0001")),
        (Decimal("100.0"), "mmbtu", "mwh", Decimal("29.3071070172"), Decimal("0.0001")),
    ],
    "mass": [
        (Decimal("1.0"), "kg", "lb", Decimal("2.20462262185"), Decimal("0.0001")),
        (Decimal("1.0"), "lb", "kg", Decimal("0.45359237"), Decimal("0.0001")),
        (Decimal("1.0"), "mt", "lb", Decimal("2204.62262185"), Decimal("0.0001")),
        (Decimal("1000.0"), "kg", "lb", Decimal("2204.62262185"), Decimal("0.0001")),
    ],
    "temperature": [
        # Fahrenheit to Celsius conversions
        (Decimal("32.0"), "f", "c", Decimal("0.0"), Decimal("0.0001")),
        (Decimal("212.0"), "f", "c", Decimal("100.0"), Decimal("0.0001")),
        (Decimal("77.0"), "f", "c", Decimal("25.0"), Decimal("0.0001")),
        (Decimal("-40.0"), "f", "c", Decimal("-40.0"), Decimal("0.0001")),
        # Celsius to Kelvin conversions
        (Decimal("0.0"), "c", "k", Decimal("273.15"), Decimal("0.0001")),
        (Decimal("100.0"), "c", "k", Decimal("373.15"), Decimal("0.0001")),
        (Decimal("25.0"), "c", "k", Decimal("298.15"), Decimal("0.0001")),
    ],
    "pressure": [
        (Decimal("1.0"), "psi", "kpa", Decimal("6.89475729317"), Decimal("0.0001")),
        (Decimal("14.696"), "psi", "kpa", Decimal("101.325"), Decimal("0.01")),  # 1 atm
        (Decimal("1.0"), "bar", "psi", Decimal("14.5037737730"), Decimal("0.0001")),
    ],
}

# EPA 40 CFR Part 98 Table C-1 Reference Values
# Source: https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-98
EPA_EMISSION_FACTORS_REFERENCE = {
    "natural_gas": {
        "co2_kg_per_mmbtu": Decimal("53.06"),
        "ch4_kg_per_mmbtu": Decimal("0.001"),
        "n2o_kg_per_mmbtu": Decimal("0.0001"),
        "source": "40 CFR 98 Table C-1 and C-2",
    },
    "distillate_fuel_oil": {
        "co2_kg_per_mmbtu": Decimal("73.96"),
        "ch4_kg_per_mmbtu": Decimal("0.003"),
        "n2o_kg_per_mmbtu": Decimal("0.0006"),
        "source": "40 CFR 98 Table C-1 and C-2",
    },
    "coal_bituminous": {
        "co2_kg_per_mmbtu": Decimal("93.28"),
        "ch4_kg_per_mmbtu": Decimal("0.011"),
        "n2o_kg_per_mmbtu": Decimal("0.0016"),
        "source": "40 CFR 98 Table C-1 and C-2",
    },
}

# IPCC Global Warming Potentials (100-year horizon)
# AR4 values are used by EPA 40 CFR Part 98
GWP_REFERENCE = {
    "ar4": {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
        "source": "IPCC AR4 / 40 CFR 98.A",
    },
    "ar5": {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
        "source": "IPCC AR5",
    },
}

# ASME PTC 4.1 Reference Efficiency Values
# Based on standard boiler operating conditions
ASME_EFFICIENCY_REFERENCE = [
    # (fuel_input_mmbtu, useful_output_mmbtu, expected_efficiency_pct, tolerance_pct)
    (Decimal("100.0"), Decimal("82.0"), Decimal("82.0"), Decimal("0.01")),
    (Decimal("100.0"), Decimal("85.0"), Decimal("85.0"), Decimal("0.01")),
    (Decimal("150.0"), Decimal("123.0"), Decimal("82.0"), Decimal("0.01")),
    (Decimal("200.0"), Decimal("172.0"), Decimal("86.0"), Decimal("0.01")),
]

# Indirect efficiency method reference values (sum of losses)
ASME_INDIRECT_REFERENCE = [
    # (losses_dict, expected_efficiency_pct, tolerance_pct)
    (
        {
            "dry_flue_gas": Decimal("5.0"),
            "moisture_fuel": Decimal("0.5"),
            "moisture_air": Decimal("0.3"),
            "hydrogen_combustion": Decimal("3.5"),
            "radiation": Decimal("1.0"),
            "blowdown": Decimal("0.5"),
            "other": Decimal("0.2"),
        },
        Decimal("89.0"),  # 100 - 11 = 89%
        Decimal("0.01"),
    ),
    (
        {
            "dry_flue_gas": Decimal("8.0"),
            "moisture_fuel": Decimal("1.0"),
            "moisture_air": Decimal("0.5"),
            "hydrogen_combustion": Decimal("4.0"),
            "radiation": Decimal("1.5"),
            "blowdown": Decimal("1.0"),
            "other": Decimal("0.5"),
        },
        Decimal("83.5"),  # 100 - 16.5 = 83.5%
        Decimal("0.01"),
    ),
]


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def precision_calculator():
    """Create precision calculator for testing."""
    from calculators.precision_utils import PrecisionCalculator
    return PrecisionCalculator(context="engineering")


@pytest.fixture
def unit_converter():
    """Create unit converter for testing."""
    from calculators.precision_utils import PrecisionUnitConverter
    return PrecisionUnitConverter()


@pytest.fixture
def emissions_calculator():
    """Create emissions calculator for testing."""
    from calculators.precision_utils import PrecisionEmissionsCalculator
    return PrecisionEmissionsCalculator()


@pytest.fixture
def efficiency_calculator():
    """Create efficiency calculator for testing."""
    from calculators.precision_utils import PrecisionEfficiencyCalculator
    return PrecisionEfficiencyCalculator()


@pytest.fixture
def epa_mapper():
    """Create EPA compliance mapper for testing."""
    from compliance.epa_mapping import EPAComplianceMapper
    return EPAComplianceMapper()


# =============================================================================
# UNIT CONVERSION VALIDATION TESTS
# =============================================================================

class TestNISTUnitConversions:
    """
    Validate unit conversions against NIST SP 811 reference values.

    These tests ensure that all unit conversion calculations match
    the authoritative NIST reference values within specified tolerances.
    """

    @pytest.mark.parametrize("value,from_unit,to_unit,expected,tolerance", NIST_UNIT_CONVERSIONS["energy"])
    def test_energy_conversions(
        self,
        unit_converter,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        expected: Decimal,
        tolerance: Decimal,
    ):
        """
        Validate energy unit conversions against NIST reference.

        Reference: NIST SP 811 Appendix B
        """
        result = unit_converter.convert_energy(value, from_unit, to_unit)

        deviation = abs(result.converted_value - expected) / expected * Decimal("100")

        assert deviation <= tolerance, (
            f"Energy conversion {value} {from_unit} -> {to_unit} "
            f"resulted in {result.converted_value}, expected {expected} "
            f"(deviation: {deviation}%, tolerance: {tolerance}%)"
        )

        # Verify provenance hash is generated
        assert len(result.provenance_hash) == 16, "Provenance hash must be 16 characters"

    @pytest.mark.parametrize("value,from_unit,to_unit,expected,tolerance", NIST_UNIT_CONVERSIONS["mass"])
    def test_mass_conversions(
        self,
        unit_converter,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        expected: Decimal,
        tolerance: Decimal,
    ):
        """
        Validate mass unit conversions against NIST reference.

        Reference: NIST SP 811 Appendix B
        """
        result = unit_converter.convert_mass(value, from_unit, to_unit)

        deviation = abs(result.converted_value - expected) / expected * Decimal("100")

        assert deviation <= tolerance, (
            f"Mass conversion {value} {from_unit} -> {to_unit} "
            f"resulted in {result.converted_value}, expected {expected} "
            f"(deviation: {deviation}%, tolerance: {tolerance}%)"
        )

    @pytest.mark.parametrize("value,from_unit,to_unit,expected,tolerance", NIST_UNIT_CONVERSIONS["temperature"])
    def test_temperature_conversions(
        self,
        unit_converter,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        expected: Decimal,
        tolerance: Decimal,
    ):
        """
        Validate temperature unit conversions against NIST reference.

        Reference: NIST SP 811 Section 4.2.1
        """
        result = unit_converter.convert_temperature(value, from_unit, to_unit)

        if expected == Decimal("0"):
            deviation = abs(result.converted_value - expected)
            assert deviation <= Decimal("0.01"), (
                f"Temperature conversion {value} {from_unit} -> {to_unit} "
                f"resulted in {result.converted_value}, expected {expected}"
            )
        else:
            deviation = abs(result.converted_value - expected) / abs(expected) * Decimal("100")
            assert deviation <= tolerance, (
                f"Temperature conversion {value} {from_unit} -> {to_unit} "
                f"resulted in {result.converted_value}, expected {expected} "
                f"(deviation: {deviation}%, tolerance: {tolerance}%)"
            )

    def test_conversion_roundtrip_precision(self, unit_converter):
        """
        Validate that roundtrip conversions maintain precision.

        Convert value A -> B -> A and verify result equals original
        within numerical precision limits.
        """
        original = Decimal("1000.123456")

        # BTU -> kJ -> BTU
        to_kj = unit_converter.convert_energy(original, "btu", "kj")
        back_to_btu = unit_converter.convert_energy(to_kj.converted_value, "kj", "btu")

        deviation = abs(back_to_btu.converted_value - original) / original * Decimal("100")

        assert deviation < Decimal("0.001"), (
            f"Roundtrip conversion lost precision: {original} -> {to_kj.converted_value} -> {back_to_btu.converted_value}"
        )


# =============================================================================
# EPA EMISSION FACTOR VALIDATION TESTS
# =============================================================================

class TestEPAEmissionFactors:
    """
    Validate emission factors against EPA 40 CFR Part 98 Table C-1 and C-2.

    These tests ensure that all emission calculations use the correct
    EPA-mandated emission factors for regulatory compliance.
    """

    def test_natural_gas_co2_factor(self, emissions_calculator):
        """
        Validate natural gas CO2 emission factor matches EPA Table C-1.

        Reference: 40 CFR 98 Table C-1 (53.06 kg CO2/MMBtu)
        """
        result = emissions_calculator.calculate_ghg_emissions("natural_gas", Decimal("1.0"))

        expected_co2 = EPA_EMISSION_FACTORS_REFERENCE["natural_gas"]["co2_kg_per_mmbtu"]

        assert result.co2_kg == expected_co2, (
            f"Natural gas CO2 factor mismatch: got {result.co2_kg}, expected {expected_co2}"
        )

    def test_natural_gas_ch4_factor(self, emissions_calculator):
        """
        Validate natural gas CH4 emission factor matches EPA Table C-2.

        Reference: 40 CFR 98 Table C-2 (0.001 kg CH4/MMBtu)
        """
        result = emissions_calculator.calculate_ghg_emissions("natural_gas", Decimal("1.0"))

        expected_ch4 = EPA_EMISSION_FACTORS_REFERENCE["natural_gas"]["ch4_kg_per_mmbtu"]

        assert result.ch4_kg == expected_ch4, (
            f"Natural gas CH4 factor mismatch: got {result.ch4_kg}, expected {expected_ch4}"
        )

    def test_natural_gas_n2o_factor(self, emissions_calculator):
        """
        Validate natural gas N2O emission factor matches EPA Table C-2.

        Reference: 40 CFR 98 Table C-2 (0.0001 kg N2O/MMBtu)
        """
        result = emissions_calculator.calculate_ghg_emissions("natural_gas", Decimal("1.0"))

        expected_n2o = EPA_EMISSION_FACTORS_REFERENCE["natural_gas"]["n2o_kg_per_mmbtu"]

        assert result.n2o_kg == expected_n2o, (
            f"Natural gas N2O factor mismatch: got {result.n2o_kg}, expected {expected_n2o}"
        )

    def test_co2e_calculation_with_gwp(self, emissions_calculator):
        """
        Validate CO2-equivalent calculation uses correct GWP values.

        CO2e = CO2 + (CH4 * GWP_CH4) + (N2O * GWP_N2O)

        Reference: 40 CFR 98 Subpart A
        """
        result = emissions_calculator.calculate_ghg_emissions("natural_gas", Decimal("1000.0"))

        # Manual calculation
        co2 = Decimal("53.06") * Decimal("1000")
        ch4 = Decimal("0.001") * Decimal("1000") * Decimal("28")  # Using AR5 GWP
        n2o = Decimal("0.0001") * Decimal("1000") * Decimal("265")  # Using AR5 GWP

        expected_co2e = co2 + ch4 + n2o

        # Allow for rounding differences
        deviation = abs(result.co2e_kg - expected_co2e) / expected_co2e * Decimal("100")

        assert deviation < Decimal("0.1"), (
            f"CO2e calculation mismatch: got {result.co2e_kg}, expected {expected_co2e}"
        )

    def test_emissions_provenance_tracking(self, emissions_calculator):
        """
        Validate that emissions calculations include proper provenance.

        All EPA-reportable values must have SHA-256 hash for audit trail.
        """
        result = emissions_calculator.calculate_ghg_emissions("natural_gas", Decimal("100.0"))

        # Verify hashes are present and valid
        assert len(result.input_hash) == 16, "Input hash must be 16 characters"
        assert len(result.output_hash) == 16, "Output hash must be 16 characters"
        assert result.input_hash != result.output_hash, "Input and output hashes must differ"

        # Verify calculation steps are recorded
        assert len(result.calculation_steps) >= 4, "Must have at least 4 calculation steps"

        # Verify emission factors are documented
        assert "co2_kg_per_mmbtu" in result.emission_factors_used


# =============================================================================
# GWP VALIDATION TESTS
# =============================================================================

class TestGWPValues:
    """
    Validate Global Warming Potential values against IPCC references.

    EPA 40 CFR Part 98 uses IPCC AR4 values for current reporting.
    """

    def test_gwp_co2_value(self, epa_mapper):
        """
        Validate CO2 GWP is always 1 (reference gas).
        """
        from compliance.epa_mapping import PollutantType

        gwp = epa_mapper.get_gwp(PollutantType.CO2)

        assert gwp == Decimal("1"), f"CO2 GWP must be 1, got {gwp}"

    def test_gwp_ch4_value(self, epa_mapper):
        """
        Validate CH4 GWP matches EPA-required value.

        Reference: 40 CFR 98.A uses IPCC AR4 value of 25
        """
        from compliance.epa_mapping import PollutantType

        gwp = epa_mapper.get_gwp(PollutantType.CH4)

        # EPA uses AR4 value of 25
        assert gwp == GWP_REFERENCE["ar4"]["ch4"], (
            f"CH4 GWP must match EPA AR4 value ({GWP_REFERENCE['ar4']['ch4']}), got {gwp}"
        )

    def test_gwp_n2o_value(self, epa_mapper):
        """
        Validate N2O GWP matches EPA-required value.

        Reference: 40 CFR 98.A uses IPCC AR4 value of 298
        """
        from compliance.epa_mapping import PollutantType

        gwp = epa_mapper.get_gwp(PollutantType.N2O)

        # EPA uses AR4 value of 298
        assert gwp == GWP_REFERENCE["ar4"]["n2o"], (
            f"N2O GWP must match EPA AR4 value ({GWP_REFERENCE['ar4']['n2o']}), got {gwp}"
        )


# =============================================================================
# EFFICIENCY CALCULATION VALIDATION TESTS
# =============================================================================

class TestASMEEfficiencyCalculations:
    """
    Validate efficiency calculations against ASME PTC 4.1 reference values.

    These tests ensure that boiler efficiency calculations follow
    the ASME Power Test Code methodology.
    """

    @pytest.mark.parametrize("fuel_input,useful_output,expected_eff,tolerance", ASME_EFFICIENCY_REFERENCE)
    def test_direct_method_efficiency(
        self,
        efficiency_calculator,
        fuel_input: Decimal,
        useful_output: Decimal,
        expected_eff: Decimal,
        tolerance: Decimal,
    ):
        """
        Validate direct method efficiency calculation.

        Efficiency = Useful Output / Fuel Input * 100

        Reference: ASME PTC 4.1-2013 Section 5.2
        """
        result = efficiency_calculator.calculate_direct(fuel_input, useful_output)

        deviation = abs(result.efficiency_percent - expected_eff)

        assert deviation <= tolerance, (
            f"Direct efficiency calculation: {useful_output}/{fuel_input} "
            f"resulted in {result.efficiency_percent}%, expected {expected_eff}% "
            f"(deviation: {deviation}%, tolerance: {tolerance}%)"
        )

        # Verify provenance
        assert len(result.input_hash) == 16
        assert len(result.output_hash) == 16
        assert result.formula_reference == "ASME PTC 4.1-2013"

    @pytest.mark.parametrize("losses,expected_eff,tolerance", ASME_INDIRECT_REFERENCE)
    def test_indirect_method_efficiency(
        self,
        efficiency_calculator,
        losses: Dict[str, Decimal],
        expected_eff: Decimal,
        tolerance: Decimal,
    ):
        """
        Validate indirect (heat loss) method efficiency calculation.

        Efficiency = 100 - Sum of Losses (%)

        Reference: ASME PTC 4.1-2013 Section 5.3
        """
        result = efficiency_calculator.calculate_indirect(Decimal("100.0"), losses)

        deviation = abs(result.efficiency_percent - expected_eff)

        assert deviation <= tolerance, (
            f"Indirect efficiency calculation "
            f"resulted in {result.efficiency_percent}%, expected {expected_eff}% "
            f"(deviation: {deviation}%, tolerance: {tolerance}%)"
        )

        # Verify all losses are tracked
        assert len(result.losses_breakdown) == len(losses)

    def test_efficiency_uncertainty_values(self, efficiency_calculator):
        """
        Validate efficiency uncertainty values per ASME PTC 4.1.

        Direct method: ~1% uncertainty
        Indirect method: ~0.5% uncertainty
        """
        direct_result = efficiency_calculator.calculate_direct(
            Decimal("100.0"),
            Decimal("85.0")
        )

        indirect_result = efficiency_calculator.calculate_indirect(
            Decimal("100.0"),
            {"dry_flue_gas": Decimal("10.0"), "radiation": Decimal("2.0")}
        )

        # Direct method has higher uncertainty
        assert direct_result.uncertainty_percent >= indirect_result.uncertainty_percent

        # Uncertainty should be reasonable values
        assert Decimal("0.1") <= direct_result.uncertainty_percent <= Decimal("2.0")
        assert Decimal("0.1") <= indirect_result.uncertainty_percent <= Decimal("1.0")


# =============================================================================
# PRECISION ARITHMETIC VALIDATION TESTS
# =============================================================================

class TestDecimalPrecision:
    """
    Validate that all calculations use proper Decimal precision.

    These tests ensure that floating-point errors do not affect
    regulatory compliance calculations.
    """

    def test_decimal_round_half_up(self, precision_calculator):
        """
        Validate ROUND_HALF_UP behavior per NIST guidelines.

        0.5 should round up to 1 (not to nearest even as in banker's rounding)
        """
        # Test cases where ROUND_HALF_UP differs from ROUND_HALF_EVEN
        test_cases = [
            (Decimal("2.5"), 0, Decimal("3")),
            (Decimal("3.5"), 0, Decimal("4")),
            (Decimal("2.25"), 1, Decimal("2.3")),
            (Decimal("2.35"), 1, Decimal("2.4")),
        ]

        for value, places, expected in test_cases:
            result = precision_calculator.round_half_up(value, places)
            assert result == expected, (
                f"ROUND_HALF_UP({value}, {places}) = {result}, expected {expected}"
            )

    def test_no_floating_point_errors(self, precision_calculator):
        """
        Validate that classic floating-point errors do not occur.

        Example: 0.1 + 0.2 should equal exactly 0.3
        """
        # This would fail with floats: 0.1 + 0.2 != 0.3
        result = precision_calculator.add(Decimal("0.1"), Decimal("0.2"))

        assert result == Decimal("0.3"), (
            f"0.1 + 0.2 = {result}, expected 0.3 (floating-point error detected)"
        )

    def test_kahan_summation_accuracy(self, precision_calculator):
        """
        Validate Kahan summation for improved accuracy.

        Sum many small values without accumulated error.
        """
        # Sum 10000 values of 0.0001 should equal 1.0
        values = [Decimal("0.0001")] * 10000

        result = precision_calculator.sum(values)

        expected = Decimal("1.0")
        deviation = abs(result - expected)

        assert deviation < Decimal("0.0001"), (
            f"Kahan summation: sum of 10000 x 0.0001 = {result}, expected {expected}"
        )

    def test_division_precision_maintained(self, precision_calculator):
        """
        Validate that division maintains required precision.
        """
        # 1/3 should maintain precision
        result = precision_calculator.divide(Decimal("1"), Decimal("3"))

        # Verify result has proper decimal places
        assert "." in str(result)

        # Verify result is close to 0.333...
        expected = Decimal("0.33333333")
        deviation = abs(result - expected)

        assert deviation < Decimal("0.00001"), (
            f"1/3 = {result}, expected ~{expected}"
        )


# =============================================================================
# COMPLIANCE MAPPING VALIDATION TESTS
# =============================================================================

class TestEPAComplianceMapping:
    """
    Validate EPA compliance requirement mappings.
    """

    def test_ghg_requirements_exist(self, epa_mapper):
        """
        Validate that all required GHG reporting requirements are mapped.
        """
        from compliance.epa_mapping import PollutantType

        co2_reqs = epa_mapper.get_requirements_by_pollutant(PollutantType.CO2)

        # Should have Tier 1-4 calculation methods for CO2
        method_ids = [req.calculation_method for req in co2_reqs]

        assert "TIER_1_FUEL_ANALYSIS" in method_ids, "Missing Tier 1 CO2 calculation"
        assert "TIER_4_CEMS" in method_ids, "Missing Tier 4 CEMS calculation"

    def test_emission_factor_citations(self, epa_mapper):
        """
        Validate that emission factors include proper EPA citations.
        """
        from compliance.epa_mapping import FuelCategory

        factor = epa_mapper.get_co2_emission_factor(FuelCategory.NATURAL_GAS)

        assert factor is not None, "Natural gas CO2 factor not found"
        assert "40 CFR" in factor.source_document, "Missing CFR citation"
        assert "Table C-1" in factor.table_reference, "Missing table reference"
        assert factor.effective_date is not None, "Missing effective date"

    def test_compliance_validation(self, epa_mapper):
        """
        Validate compliance checking against emission limits.
        """
        # Test a value below the limit
        result = epa_mapper.validate_compliance(
            "NSPS-001",  # SO2 limit: 0.50 lb/MMBtu
            Decimal("0.30"),  # Below limit
        )

        assert result.is_compliant is True
        assert result.margin_percent > Decimal("0")
        assert len(result.provenance_hash) == 16

        # Test a value above the limit
        result_fail = epa_mapper.validate_compliance(
            "NSPS-001",
            Decimal("0.60"),  # Above limit
        )

        assert result_fail.is_compliant is False
        assert result_fail.margin_percent < Decimal("0")


# =============================================================================
# GOLDEN VALUE INTEGRATION TESTS
# =============================================================================

class TestGoldenValueIntegration:
    """
    End-to-end golden value tests validating complete calculation chains.
    """

    def test_complete_emissions_calculation(
        self,
        unit_converter,
        emissions_calculator,
    ):
        """
        Golden value test for complete emissions calculation chain.

        Scenario: 1000 MMBTU natural gas combustion
        Expected outputs: CO2, CH4, N2O, CO2e with EPA factors
        """
        heat_input_mmbtu = Decimal("1000.0")

        # Calculate emissions
        result = emissions_calculator.calculate_ghg_emissions(
            "natural_gas",
            heat_input_mmbtu,
        )

        # Golden values (pre-calculated)
        expected_co2_kg = Decimal("53060.0")  # 1000 * 53.06
        expected_ch4_kg = Decimal("1.0")       # 1000 * 0.001
        expected_n2o_kg = Decimal("0.1")       # 1000 * 0.0001

        # CO2e = CO2 + CH4*28 + N2O*265 (AR5 GWP)
        expected_co2e_kg = expected_co2_kg + (expected_ch4_kg * Decimal("28")) + (expected_n2o_kg * Decimal("265"))

        # Validate with 0.1% tolerance
        assert abs(result.co2_kg - expected_co2_kg) / expected_co2_kg < Decimal("0.001")
        assert abs(result.ch4_kg - expected_ch4_kg) / expected_ch4_kg < Decimal("0.001")
        assert abs(result.n2o_kg - expected_n2o_kg) / expected_n2o_kg < Decimal("0.001")

        # CO2e tolerance slightly higher due to compounding
        assert abs(result.co2e_kg - expected_co2e_kg) / expected_co2e_kg < Decimal("0.01")

    def test_complete_efficiency_calculation(self, efficiency_calculator):
        """
        Golden value test for complete efficiency calculation.

        Scenario: Boiler with known fuel input and steam output
        Validate both direct and indirect methods
        """
        fuel_input = Decimal("185.0")  # MMBTU/hr
        steam_output = Decimal("152.0")  # MMBTU/hr

        # Direct method
        direct_result = efficiency_calculator.calculate_direct(fuel_input, steam_output)

        # Golden value: 152/185 * 100 = 82.162...%
        expected_eff = (steam_output / fuel_input) * Decimal("100")

        assert abs(direct_result.efficiency_percent - expected_eff) < Decimal("0.01")

        # Losses calculation
        losses_mmbtu = fuel_input - steam_output
        assert abs(direct_result.total_losses_mmbtu_hr - losses_mmbtu) < Decimal("0.01")


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for golden value tests."""
    config.addinivalue_line("markers", "golden: NIST golden value validation tests")
    config.addinivalue_line("markers", "epa: EPA compliance validation tests")
    config.addinivalue_line("markers", "asme: ASME PTC validation tests")
    config.addinivalue_line("markers", "precision: Decimal precision validation tests")
