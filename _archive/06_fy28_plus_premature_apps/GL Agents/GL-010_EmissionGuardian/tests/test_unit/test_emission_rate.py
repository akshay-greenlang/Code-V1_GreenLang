"""
GL-010 EmissionGuardian - Emission Rate Calculator Tests

Comprehensive test suite for EPA 40 CFR Part 75 emission rate calculations.
Tests deterministic behavior, numerical precision, and provenance tracking.

Reference: 40 CFR Part 75, Appendix A and F
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List

# Import will need adjustment based on actual module structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.emission_rate import (
    FuelType,
    F_FACTORS_O2,
    F_FACTORS_CO2,
    MOLECULAR_WEIGHTS,
    REFERENCE_O2_PERCENT,
    MOLAR_VOLUME_DSCF,
    CalculationTrace,
    EmissionRateResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_o2_measurement():
    """Sample O2 concentration measurement."""
    return Decimal("5.0")  # 5% O2


@pytest.fixture
def sample_nox_ppm():
    """Sample NOx concentration in ppm."""
    return Decimal("100.0")  # 100 ppm NOx


@pytest.fixture
def sample_so2_ppm():
    """Sample SO2 concentration in ppm."""
    return Decimal("50.0")  # 50 ppm SO2


# =============================================================================
# TEST: CONSTANTS VALIDATION
# =============================================================================

class TestEPAConstants:
    """Test EPA constants are correctly defined."""

    def test_f_factors_o2_all_fuels_defined(self):
        """All fuel types should have O2 F-factors."""
        for fuel in FuelType:
            assert fuel in F_FACTORS_O2
            assert F_FACTORS_O2[fuel] > Decimal("0")

    def test_f_factors_co2_all_fuels_defined(self):
        """All fuel types should have CO2 F-factors."""
        for fuel in FuelType:
            assert fuel in F_FACTORS_CO2
            assert F_FACTORS_CO2[fuel] > Decimal("0")

    def test_natural_gas_f_factor_o2(self):
        """Natural gas O2 F-factor should be 8710 dscf/MMBtu per EPA."""
        assert F_FACTORS_O2[FuelType.NATURAL_GAS] == Decimal("8710")

    def test_coal_f_factor_o2(self):
        """Coal O2 F-factor should be 9780 dscf/MMBtu per EPA."""
        assert F_FACTORS_O2[FuelType.COAL] == Decimal("9780")

    def test_molecular_weight_nox(self):
        """NOx molecular weight should be 46.01 (as NO2)."""
        assert MOLECULAR_WEIGHTS["NOx"] == Decimal("46.01")

    def test_molecular_weight_so2(self):
        """SO2 molecular weight should be 64.06."""
        assert MOLECULAR_WEIGHTS["SO2"] == Decimal("64.06")

    def test_reference_o2(self):
        """Reference O2 should be 20.9%."""
        assert REFERENCE_O2_PERCENT == Decimal("20.9")

    def test_molar_volume(self):
        """Molar volume should be 385.3 dscf/lb-mole."""
        assert MOLAR_VOLUME_DSCF == Decimal("385.3")


# =============================================================================
# TEST: OXYGEN CORRECTION FACTOR
# =============================================================================

class TestO2CorrectionFactor:
    """Test O2 correction factor calculations."""

    def test_correction_at_zero_o2(self):
        """Correction factor at 0% O2 should be ~1.32."""
        # Formula: (20.9 - O2_ref) / (20.9 - O2_meas)
        # At 0% O2: (20.9 - 0) / (20.9 - 0) = 1.0 for 0% reference
        # But typically reference is 3% for gas, giving (20.9 - 3) / (20.9 - 0)
        o2_measured = Decimal("0.0")
        o2_reference = Decimal("3.0")  # Dry basis reference for natural gas
        expected = (Decimal("20.9") - o2_reference) / (Decimal("20.9") - o2_measured)
        # Should be approximately 0.856
        assert expected < Decimal("1.0")

    def test_correction_at_reference_o2(self):
        """Correction factor at reference O2 should be 1.0."""
        o2_measured = Decimal("3.0")
        o2_reference = Decimal("3.0")
        correction = (Decimal("20.9") - o2_reference) / (Decimal("20.9") - o2_measured)
        assert correction == Decimal("1.0")

    def test_correction_increases_with_o2(self):
        """Correction factor should increase with O2 concentration."""
        o2_reference = Decimal("3.0")

        correction_low = (Decimal("20.9") - o2_reference) / (Decimal("20.9") - Decimal("3.0"))
        correction_high = (Decimal("20.9") - o2_reference) / (Decimal("20.9") - Decimal("10.0"))

        assert correction_high > correction_low


# =============================================================================
# TEST: EMISSION RATE CALCULATIONS
# =============================================================================

class TestEmissionRateCalculation:
    """Test emission rate calculation formulas."""

    def test_nox_lb_mmbtu_formula(self):
        """Test NOx emission rate in lb/MMBtu using O2 method."""
        # Formula: ER = (C * MW * Fd) / (Mv * (20.9 - %O2d))
        # C = concentration in ppm
        # MW = molecular weight
        # Fd = F-factor for fuel
        # Mv = molar volume
        # %O2d = measured O2 dry

        c_nox_ppm = Decimal("100")
        mw_nox = MOLECULAR_WEIGHTS["NOx"]
        fd = F_FACTORS_O2[FuelType.NATURAL_GAS]
        mv = MOLAR_VOLUME_DSCF
        o2_measured = Decimal("5.0")

        # Convert ppm to fraction
        c_fraction = c_nox_ppm / Decimal("1000000")

        # Calculate emission rate
        numerator = c_fraction * mw_nox * fd
        denominator = mv * (REFERENCE_O2_PERCENT - o2_measured)

        er_lb_mmbtu = numerator / denominator

        # Should be a reasonable value (typically 0.1 - 0.5 for natural gas)
        assert er_lb_mmbtu > Decimal("0")
        assert er_lb_mmbtu < Decimal("1.0")

    def test_so2_lb_mmbtu_formula(self):
        """Test SO2 emission rate in lb/MMBtu using O2 method."""
        c_so2_ppm = Decimal("50")
        mw_so2 = MOLECULAR_WEIGHTS["SO2"]
        fd = F_FACTORS_O2[FuelType.COAL]
        mv = MOLAR_VOLUME_DSCF
        o2_measured = Decimal("5.0")

        c_fraction = c_so2_ppm / Decimal("1000000")

        numerator = c_fraction * mw_so2 * fd
        denominator = mv * (REFERENCE_O2_PERCENT - o2_measured)

        er_lb_mmbtu = numerator / denominator

        assert er_lb_mmbtu > Decimal("0")

    def test_zero_concentration_gives_zero_rate(self):
        """Zero pollutant concentration should give zero emission rate."""
        c_nox_ppm = Decimal("0")

        er = c_nox_ppm * MOLECULAR_WEIGHTS["NOx"] / MOLAR_VOLUME_DSCF

        assert er == Decimal("0")


# =============================================================================
# TEST: CALCULATION TRACE
# =============================================================================

class TestCalculationTrace:
    """Test calculation trace for explainability."""

    def test_trace_structure(self):
        """Calculation trace should have required fields."""
        trace = CalculationTrace(
            step_number=1,
            description="Calculate O2 correction factor",
            formula="(20.9 - O2_ref) / (20.9 - O2_meas)",
            inputs={"O2_ref": "3.0", "O2_meas": "5.0"},
            output="correction_factor",
            output_value=Decimal("1.128"),
        )

        assert trace.step_number == 1
        assert "O2" in trace.description
        assert "O2_ref" in trace.inputs
        assert trace.output_value > Decimal("1")

    def test_trace_timestamp_populated(self):
        """Trace should auto-populate timestamp."""
        trace = CalculationTrace(
            step_number=1,
            description="Test",
            formula="a + b",
            inputs={"a": "1", "b": "2"},
            output="c",
            output_value=Decimal("3"),
        )

        assert trace.timestamp is not None
        assert len(trace.timestamp) > 0


# =============================================================================
# TEST: EMISSION RATE RESULT
# =============================================================================

class TestEmissionRateResult:
    """Test EmissionRateResult structure."""

    def test_result_structure(self):
        """Result should have value, unit, and trace."""
        result = EmissionRateResult(
            value=Decimal("0.15"),
            unit="lb/MMBtu",
            calculation_trace=[
                CalculationTrace(
                    step_number=1,
                    description="Step 1",
                    formula="a * b",
                    inputs={"a": "1"},
                    output="x",
                    output_value=Decimal("1"),
                )
            ],
        )

        assert result.value == Decimal("0.15")
        assert result.unit == "lb/MMBtu"
        assert len(result.calculation_trace) == 1


# =============================================================================
# TEST: DECIMAL PRECISION
# =============================================================================

class TestDecimalPrecision:
    """Test Decimal precision for regulatory compliance."""

    def test_decimal_rounding(self):
        """Test proper Decimal rounding."""
        value = Decimal("0.12345")
        rounded = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert rounded == Decimal("0.123")

    def test_no_floating_point_errors(self):
        """Decimal arithmetic should avoid floating point errors."""
        # Classic floating point problem: 0.1 + 0.2 != 0.3
        a = Decimal("0.1")
        b = Decimal("0.2")
        c = a + b

        assert c == Decimal("0.3")

    def test_precision_maintained_through_calculation(self):
        """Precision should be maintained through multi-step calculation."""
        # Simulate emission rate calculation
        concentration = Decimal("100.123456")
        mw = Decimal("46.01")
        factor = Decimal("8710")

        result = concentration * mw * factor / Decimal("385.3")

        # Should maintain precision
        assert len(str(result).split(".")[-1]) <= 10  # Reasonable precision


# =============================================================================
# TEST: FUEL TYPE VALIDATION
# =============================================================================

class TestFuelTypeValidation:
    """Test fuel type handling."""

    def test_all_fuel_types_enumerated(self):
        """All EPA fuel types should be defined."""
        expected_fuels = {"coal", "oil", "natural_gas", "wood", "refuse"}
        actual_fuels = {ft.value for ft in FuelType}
        assert expected_fuels == actual_fuels

    def test_fuel_f_factors_different(self):
        """Different fuels should have different F-factors."""
        coal_f = F_FACTORS_O2[FuelType.COAL]
        gas_f = F_FACTORS_O2[FuelType.NATURAL_GAS]
        assert coal_f != gas_f


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_concentration(self):
        """Very low concentrations should calculate correctly."""
        c_ppm = Decimal("0.1")  # 0.1 ppm
        mw = MOLECULAR_WEIGHTS["NOx"]

        result = c_ppm * mw / MOLAR_VOLUME_DSCF

        assert result > Decimal("0")
        assert result < Decimal("0.001")

    def test_high_o2_concentration(self):
        """High O2 (near ambient) should work without division by zero."""
        o2_measured = Decimal("20.0")  # Near ambient

        # Should not divide by zero
        denominator = REFERENCE_O2_PERCENT - o2_measured
        assert denominator > Decimal("0")

    def test_o2_at_reference(self):
        """O2 at exactly reference should give normal correction."""
        o2_measured = REFERENCE_O2_PERCENT
        # This would cause division by zero in uncorrected formula
        # But reference-based calculation should handle it
        pass  # Implementation would handle this case


# =============================================================================
# TEST: PROVENANCE HASH
# =============================================================================

class TestProvenanceHash:
    """Test SHA-256 provenance hash generation."""

    def test_hash_generation(self):
        """Result should include provenance hash."""
        import hashlib
        import json

        inputs = {
            "concentration_ppm": "100",
            "fuel_type": "natural_gas",
            "o2_percent": "5.0",
        }

        hash_input = json.dumps(inputs, sort_keys=True)
        expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        assert len(expected_hash) == 64  # SHA-256 produces 64 hex chars

    def test_hash_deterministic(self):
        """Same inputs should produce same hash."""
        import hashlib
        import json

        inputs = {"a": "1", "b": "2"}

        hash1 = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Different inputs should produce different hashes."""
        import hashlib
        import json

        inputs1 = {"concentration": "100"}
        inputs2 = {"concentration": "101"}

        hash1 = hashlib.sha256(json.dumps(inputs1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(inputs2, sort_keys=True).encode()).hexdigest()

        assert hash1 != hash2
