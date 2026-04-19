# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Reproducibility and Determinism Tests

Comprehensive tests for ensuring bit-perfect reproducibility and deterministic
calculations across all condenser optimization components.

Key test areas:
- Bit-perfect reproducibility
- Provenance hash consistency
- Cross-platform determinism
- Numerical precision verification
- Random seed handling

Test coverage target: 95%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
import math
import platform
import struct
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyOutput,
    calculate_cw_temperature_rise,
    calculate_optimal_cw_flow,
    calculate_cw_pumping_power,
)
from calculators.heat_transfer_calculator import (
    HeatTransferCalculator,
    HeatTransferInput,
)
from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingInput,
)
from calculators.vacuum_calculator import (
    VacuumCalculator,
    VacuumInput,
)
from calculators.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    verify_provenance,
    compute_input_fingerprint,
    compute_output_fingerprint,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def efficiency_calculator():
    """Create EfficiencyCalculator instance."""
    return EfficiencyCalculator()


@pytest.fixture
def heat_transfer_calculator():
    """Create HeatTransferCalculator instance."""
    return HeatTransferCalculator()


@pytest.fixture
def fouling_calculator():
    """Create FoulingCalculator instance."""
    return FoulingCalculator()


@pytest.fixture
def vacuum_calculator():
    """Create VacuumCalculator instance."""
    return VacuumCalculator()


@pytest.fixture
def standard_efficiency_input():
    """Standard efficiency input for reproducibility testing."""
    return EfficiencyInput(
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        heat_duty_mw=200.0,
        turbine_output_mw=300.0,
        design_backpressure_mmhg=50.8,
        actual_backpressure_mmhg=55.0,
        design_u_value_w_m2k=3500.0,
        actual_u_value_w_m2k=3000.0,
        heat_transfer_area_m2=17500.0,
        electricity_cost_usd_mwh=50.0,
        operating_hours_per_year=8000,
    )


@pytest.fixture
def standard_heat_transfer_input():
    """Standard heat transfer input for reproducibility testing."""
    return HeatTransferInput(
        heat_duty_mw=200.0,
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        tube_od_mm=25.4,
        tube_id_mm=23.4,
        tube_length_m=12.0,
        tube_count=18500,
        tube_material="titanium",
        design_u_value_w_m2k=3500.0,
        fouling_factor_m2k_w=0.00015,
    )


@pytest.fixture
def golden_test_cases():
    """Golden test cases with known correct outputs."""
    return [
        {
            "name": "Case 1 - Standard Operation",
            "input": {
                "steam_temp_c": 40.0,
                "cw_inlet_temp_c": 25.0,
                "cw_outlet_temp_c": 35.0,
                "cw_flow_rate_m3_hr": 50000.0,
                "heat_duty_mw": 200.0,
                "turbine_output_mw": 300.0,
                "design_backpressure_mmhg": 50.8,
                "actual_backpressure_mmhg": 55.0,
                "design_u_value_w_m2k": 3500.0,
                "actual_u_value_w_m2k": 3000.0,
                "heat_transfer_area_m2": 17500.0,
            },
            "expected_ttd_c": 5.0,
            "expected_itd_c": 15.0,
            "expected_cleanliness_factor": 0.8571,
            "expected_cw_temp_rise_c": 10.0,
        },
        {
            "name": "Case 2 - Degraded Condenser",
            "input": {
                "steam_temp_c": 45.0,
                "cw_inlet_temp_c": 28.0,
                "cw_outlet_temp_c": 38.0,
                "cw_flow_rate_m3_hr": 45000.0,
                "heat_duty_mw": 180.0,
                "turbine_output_mw": 280.0,
                "design_backpressure_mmhg": 50.8,
                "actual_backpressure_mmhg": 75.0,
                "design_u_value_w_m2k": 3500.0,
                "actual_u_value_w_m2k": 2200.0,
                "heat_transfer_area_m2": 17500.0,
            },
            "expected_ttd_c": 7.0,
            "expected_itd_c": 17.0,
            "expected_cleanliness_factor": 0.6286,
            "expected_cw_temp_rise_c": 10.0,
        },
    ]


# =============================================================================
# BIT-PERFECT REPRODUCIBILITY TESTS
# =============================================================================

class TestBitPerfectReproducibility:
    """Test suite for bit-perfect reproducibility."""

    @pytest.mark.determinism
    def test_same_input_same_output(self, efficiency_calculator, standard_efficiency_input):
        """Test that identical inputs produce identical outputs."""
        result1, prov1 = efficiency_calculator.calculate(standard_efficiency_input)
        result2, prov2 = efficiency_calculator.calculate(standard_efficiency_input)

        # All numeric outputs should be exactly equal
        assert result1.thermal_efficiency_pct == result2.thermal_efficiency_pct
        assert result1.heat_recovery_efficiency_pct == result2.heat_recovery_efficiency_pct
        assert result1.cw_temp_rise_c == result2.cw_temp_rise_c
        assert result1.approach_temp_c == result2.approach_temp_c
        assert result1.ttd_c == result2.ttd_c
        assert result1.itd_c == result2.itd_c
        assert result1.cpi == result2.cpi
        assert result1.cleanliness_factor == result2.cleanliness_factor

    @pytest.mark.determinism
    def test_multiple_iterations_identical(self, efficiency_calculator, standard_efficiency_input):
        """Test that multiple iterations produce identical results."""
        results = []
        for _ in range(10):
            result, _ = efficiency_calculator.calculate(standard_efficiency_input)
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result.cpi == first_result.cpi, f"Iteration {i} CPI differs"
            assert result.ttd_c == first_result.ttd_c, f"Iteration {i} TTD differs"
            assert result.cleanliness_factor == first_result.cleanliness_factor

    @pytest.mark.determinism
    def test_order_independence(self, efficiency_calculator):
        """Test that calculation order doesn't affect results."""
        input1 = EfficiencyInput(
            steam_temp_c=40.0, cw_inlet_temp_c=25.0, cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0, heat_duty_mw=200.0, turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8, actual_backpressure_mmhg=55.0,
            design_u_value_w_m2k=3500.0, actual_u_value_w_m2k=3000.0,
            heat_transfer_area_m2=17500.0,
        )

        input2 = EfficiencyInput(
            steam_temp_c=45.0, cw_inlet_temp_c=28.0, cw_outlet_temp_c=38.0,
            cw_flow_rate_m3_hr=45000.0, heat_duty_mw=180.0, turbine_output_mw=280.0,
            design_backpressure_mmhg=50.8, actual_backpressure_mmhg=60.0,
            design_u_value_w_m2k=3500.0, actual_u_value_w_m2k=2800.0,
            heat_transfer_area_m2=17500.0,
        )

        # Calculate in order 1, 2
        result1_a, _ = efficiency_calculator.calculate(input1)
        result2_a, _ = efficiency_calculator.calculate(input2)

        # Calculate in order 2, 1
        result2_b, _ = efficiency_calculator.calculate(input2)
        result1_b, _ = efficiency_calculator.calculate(input1)

        # Results should be identical regardless of order
        assert result1_a.cpi == result1_b.cpi
        assert result2_a.cpi == result2_b.cpi

    @pytest.mark.determinism
    def test_floating_point_precision(self, efficiency_calculator, standard_efficiency_input):
        """Test floating point precision consistency."""
        result1, _ = efficiency_calculator.calculate(standard_efficiency_input)
        result2, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Convert to bytes and compare
        bytes1 = struct.pack('d', result1.cpi)
        bytes2 = struct.pack('d', result2.cpi)

        assert bytes1 == bytes2, "Floating point representation differs"

    @pytest.mark.determinism
    def test_decimal_precision_consistency(self):
        """Test Decimal type precision consistency."""
        # Test with Decimal calculations
        value1 = Decimal("3.14159265358979323846")
        value2 = Decimal("2.71828182845904523536")

        result1 = value1 * value2
        result2 = value1 * value2

        assert result1 == result2
        assert str(result1) == str(result2)


# =============================================================================
# PROVENANCE HASH CONSISTENCY TESTS
# =============================================================================

class TestProvenanceHashConsistency:
    """Test suite for provenance hash consistency."""

    @pytest.mark.determinism
    def test_input_hash_consistency(self, efficiency_calculator, standard_efficiency_input):
        """Test input hash is consistent across calculations."""
        _, prov1 = efficiency_calculator.calculate(standard_efficiency_input)
        _, prov2 = efficiency_calculator.calculate(standard_efficiency_input)

        assert prov1.input_hash == prov2.input_hash

    @pytest.mark.determinism
    def test_output_hash_consistency(self, efficiency_calculator, standard_efficiency_input):
        """Test output hash is consistent across calculations."""
        _, prov1 = efficiency_calculator.calculate(standard_efficiency_input)
        _, prov2 = efficiency_calculator.calculate(standard_efficiency_input)

        assert prov1.output_hash == prov2.output_hash

    @pytest.mark.determinism
    def test_provenance_hash_consistency(self, efficiency_calculator, standard_efficiency_input):
        """Test provenance hash is consistent (excluding timestamp)."""
        # Note: Full provenance hash includes timestamp, so we compare components
        _, prov1 = efficiency_calculator.calculate(standard_efficiency_input)
        _, prov2 = efficiency_calculator.calculate(standard_efficiency_input)

        # Input and output hashes should be identical
        assert prov1.input_hash == prov2.input_hash
        assert prov1.output_hash == prov2.output_hash

    @pytest.mark.determinism
    def test_hash_length_standard(self, efficiency_calculator, standard_efficiency_input):
        """Test hash lengths are SHA-256 standard (64 hex chars)."""
        _, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert len(provenance.input_hash) == 64
        assert len(provenance.output_hash) == 64
        assert len(provenance.provenance_hash) == 64

    @pytest.mark.determinism
    def test_hash_uniqueness(self, efficiency_calculator):
        """Test different inputs produce different hashes."""
        input1 = EfficiencyInput(
            steam_temp_c=40.0, cw_inlet_temp_c=25.0, cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0, heat_duty_mw=200.0, turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8, actual_backpressure_mmhg=55.0,
            design_u_value_w_m2k=3500.0, actual_u_value_w_m2k=3000.0,
            heat_transfer_area_m2=17500.0,
        )

        input2 = EfficiencyInput(
            steam_temp_c=45.0,  # Different
            cw_inlet_temp_c=25.0, cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0, heat_duty_mw=200.0, turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8, actual_backpressure_mmhg=55.0,
            design_u_value_w_m2k=3500.0, actual_u_value_w_m2k=3000.0,
            heat_transfer_area_m2=17500.0,
        )

        _, prov1 = efficiency_calculator.calculate(input1)
        _, prov2 = efficiency_calculator.calculate(input2)

        assert prov1.input_hash != prov2.input_hash
        assert prov1.output_hash != prov2.output_hash

    @pytest.mark.determinism
    def test_provenance_verification_passes(self, efficiency_calculator, standard_efficiency_input):
        """Test provenance verification succeeds for valid records."""
        _, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        is_valid = verify_provenance(provenance)
        assert is_valid is True

    @pytest.mark.determinism
    def test_input_fingerprint_consistency(self):
        """Test input fingerprint function consistency."""
        inputs = {
            "steam_temp_c": 40.0,
            "cw_inlet_temp_c": 25.0,
            "flow_rate": 50000.0,
        }

        fp1 = compute_input_fingerprint(inputs)
        fp2 = compute_input_fingerprint(inputs)

        assert fp1 == fp2
        assert len(fp1) == 16  # Shortened fingerprint

    @pytest.mark.determinism
    def test_output_fingerprint_consistency(self):
        """Test output fingerprint function consistency."""
        outputs = {
            "ttd_c": 5.0,
            "efficiency": 0.85,
            "cpi": 0.90,
        }

        fp1 = compute_output_fingerprint(outputs)
        fp2 = compute_output_fingerprint(outputs)

        assert fp1 == fp2


# =============================================================================
# CROSS-PLATFORM DETERMINISM TESTS
# =============================================================================

class TestCrossPlatformDeterminism:
    """Test suite for cross-platform determinism."""

    @pytest.mark.determinism
    def test_json_serialization_deterministic(self):
        """Test JSON serialization is deterministic."""
        data = {
            "z_value": 1.0,
            "a_value": 2.0,
            "m_value": 3.0,
            "nested": {"b": 1, "a": 2},
        }

        # Serialize with sorted keys
        json1 = json.dumps(data, sort_keys=True)
        json2 = json.dumps(data, sort_keys=True)

        assert json1 == json2

    @pytest.mark.determinism
    def test_sha256_consistency(self):
        """Test SHA-256 hash consistency."""
        data = "test_data_for_hashing"

        hash1 = hashlib.sha256(data.encode('utf-8')).hexdigest()
        hash2 = hashlib.sha256(data.encode('utf-8')).hexdigest()

        assert hash1 == hash2
        assert hash1 == "d404559f602eab6fd602ac7680dacbfaadd13630335e951f097af3900e9de176"

    @pytest.mark.determinism
    def test_float_representation(self):
        """Test float representation is consistent."""
        # IEEE 754 double precision
        value = 3.141592653589793

        # Pack to bytes
        packed = struct.pack('d', value)
        unpacked = struct.unpack('d', packed)[0]

        assert value == unpacked

    @pytest.mark.determinism
    def test_calculation_on_current_platform(self, efficiency_calculator, standard_efficiency_input):
        """Test calculation produces expected results on current platform."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Platform info for debugging
        platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }

        # Results should be within expected bounds
        assert 60 < result.thermal_efficiency_pct < 70
        assert result.ttd_c == 5.0
        assert result.itd_c == 15.0

    @pytest.mark.determinism
    def test_math_functions_consistency(self):
        """Test math library functions are consistent."""
        test_values = [0.5, 1.0, 2.0, math.pi, math.e]

        for val in test_values:
            # Test various math functions
            log_result = math.log(val)
            exp_result = math.exp(val) if val < 10 else 0
            sqrt_result = math.sqrt(val)

            # Results should be exactly reproducible
            assert math.log(val) == log_result
            if val < 10:
                assert math.exp(val) == exp_result
            assert math.sqrt(val) == sqrt_result

    @pytest.mark.determinism
    def test_dict_key_ordering(self):
        """Test dictionary key ordering for hashing."""
        # Python 3.7+ maintains insertion order, but we should sort for hashing
        dict1 = {"c": 3, "a": 1, "b": 2}
        dict2 = {"a": 1, "b": 2, "c": 3}

        # Without sorting, order matters
        json1_unsorted = json.dumps(dict1)
        json2_unsorted = json.dumps(dict2)

        # With sorting, order doesn't matter
        json1_sorted = json.dumps(dict1, sort_keys=True)
        json2_sorted = json.dumps(dict2, sort_keys=True)

        assert json1_sorted == json2_sorted


# =============================================================================
# GOLDEN TEST DATA VERIFICATION
# =============================================================================

class TestGoldenDataVerification:
    """Test suite for golden test data verification."""

    @pytest.mark.determinism
    def test_golden_test_case_1(self, efficiency_calculator, golden_test_cases):
        """Test against golden test case 1."""
        case = golden_test_cases[0]
        input_data = EfficiencyInput(**case["input"])

        result, _ = efficiency_calculator.calculate(input_data)

        assert result.ttd_c == case["expected_ttd_c"]
        assert result.itd_c == case["expected_itd_c"]
        assert result.cw_temp_rise_c == case["expected_cw_temp_rise_c"]
        assert abs(result.cleanliness_factor - case["expected_cleanliness_factor"]) < 0.001

    @pytest.mark.determinism
    def test_golden_test_case_2(self, efficiency_calculator, golden_test_cases):
        """Test against golden test case 2."""
        case = golden_test_cases[1]
        input_data = EfficiencyInput(**case["input"])

        result, _ = efficiency_calculator.calculate(input_data)

        assert result.ttd_c == case["expected_ttd_c"]
        assert result.itd_c == case["expected_itd_c"]
        assert abs(result.cleanliness_factor - case["expected_cleanliness_factor"]) < 0.001

    @pytest.mark.determinism
    def test_standalone_functions_golden(self):
        """Test standalone functions against golden values."""
        # CW temperature rise
        temp_rise = calculate_cw_temperature_rise(200.0, 50000.0)
        # Expected: ~3.45 C (can vary slightly based on constants)
        assert 3.0 < temp_rise < 4.0

        # Optimal CW flow
        flow = calculate_optimal_cw_flow(200.0, 10.0)
        # Expected: ~17262 m3/hr
        assert 15000 < flow < 20000

        # Pumping power
        power = calculate_cw_pumping_power(50000.0, 20.0, 0.80)
        # Expected: ~341 kW
        assert 300 < power < 400


# =============================================================================
# NUMERICAL PRECISION TESTS
# =============================================================================

class TestNumericalPrecision:
    """Test suite for numerical precision verification."""

    @pytest.mark.determinism
    def test_rounding_consistency(self):
        """Test rounding is consistent."""
        values = [2.5, 3.5, 4.5, 5.5, 2.55, 2.545]

        for val in values:
            rounded1 = round(val, 1)
            rounded2 = round(val, 1)
            assert rounded1 == rounded2

    @pytest.mark.determinism
    def test_decimal_rounding(self):
        """Test Decimal rounding for financial calculations."""
        value = Decimal("2.545")

        rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        assert rounded == Decimal("2.55")

    @pytest.mark.determinism
    def test_division_precision(self):
        """Test division precision."""
        numerator = 3000.0
        denominator = 3500.0

        result1 = numerator / denominator
        result2 = numerator / denominator

        assert result1 == result2
        assert abs(result1 - 0.857142857142857) < 1e-10

    @pytest.mark.determinism
    def test_multiplication_associativity(self):
        """Test multiplication is associative within precision."""
        a, b, c = 1.1, 2.2, 3.3

        result1 = (a * b) * c
        result2 = a * (b * c)

        # May differ in last bits due to floating point
        assert abs(result1 - result2) < 1e-14

    @pytest.mark.determinism
    def test_temperature_calculations_precision(self, efficiency_calculator, standard_efficiency_input):
        """Test temperature calculation precision."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Temperature values should be precise to 2 decimal places
        assert result.ttd_c == 5.0  # Exact
        assert result.itd_c == 15.0  # Exact
        assert result.cw_temp_rise_c == 10.0  # Exact

    @pytest.mark.determinism
    def test_efficiency_calculations_precision(self, efficiency_calculator, standard_efficiency_input):
        """Test efficiency calculation precision."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Efficiency should be rounded to 2 decimal places
        assert isinstance(result.thermal_efficiency_pct, float)
        # Check reasonable precision
        str_value = str(result.thermal_efficiency_pct)
        if '.' in str_value:
            decimal_places = len(str_value.split('.')[1])
            assert decimal_places <= 2


# =============================================================================
# PROVENANCE TRACKER TESTS
# =============================================================================

class TestProvenanceTracker:
    """Test suite for ProvenanceTracker class."""

    @pytest.mark.determinism
    def test_tracker_initialization(self):
        """Test ProvenanceTracker initialization."""
        tracker = ProvenanceTracker("TestCalculator", "1.0.0")

        assert tracker._calculator_name == "TestCalculator"
        assert tracker._calculator_version == "1.0.0"

    @pytest.mark.determinism
    def test_set_inputs(self):
        """Test setting inputs generates timestamp and ID."""
        tracker = ProvenanceTracker("TestCalculator", "1.0.0")
        tracker.set_inputs({"a": 1, "b": 2})

        assert tracker._timestamp is not None
        assert tracker._calculation_id is not None
        assert tracker._calculation_id.startswith("CALC-")

    @pytest.mark.determinism
    def test_add_step(self):
        """Test adding calculation steps."""
        tracker = ProvenanceTracker("TestCalculator", "1.0.0")
        tracker.set_inputs({"a": 1})

        tracker.add_step(
            step_number=1,
            description="Test step",
            operation="add",
            inputs={"a": 1, "b": 2},
            output_value=3,
            output_name="result"
        )

        assert len(tracker._steps) == 1
        assert tracker._steps[0].step_number == 1

    @pytest.mark.determinism
    def test_finalize(self):
        """Test finalizing provenance record."""
        tracker = ProvenanceTracker("TestCalculator", "1.0.0")
        tracker.set_inputs({"a": 1})
        tracker.add_step(1, "Test", "add", {"a": 1}, 1, "result")
        tracker.set_outputs({"result": 1})

        record = tracker.finalize()

        assert isinstance(record, ProvenanceRecord)
        assert record.calculator_name == "TestCalculator"
        assert record.provenance_hash is not None

    @pytest.mark.determinism
    def test_value_normalization(self):
        """Test value normalization for consistent hashing."""
        tracker = ProvenanceTracker("TestCalculator", "1.0.0")

        # Test float normalization
        normalized = tracker._normalize_single_value(3.14159265358979)
        assert isinstance(normalized, float)

        # Test Decimal normalization
        decimal_val = Decimal("3.14")
        normalized_decimal = tracker._normalize_single_value(decimal_val)
        assert normalized_decimal == "3.14"


# =============================================================================
# CALCULATION STEP TESTS
# =============================================================================

class TestCalculationStep:
    """Test suite for CalculationStep dataclass."""

    @pytest.mark.determinism
    def test_step_immutability(self):
        """Test CalculationStep is immutable."""
        step = CalculationStep(
            step_number=1,
            description="Test",
            operation="add",
            inputs={"a": 1},
            output_value=1,
            output_name="result"
        )

        # Attempting to modify should raise error
        with pytest.raises(Exception):  # FrozenInstanceError
            step.step_number = 2

    @pytest.mark.determinism
    def test_step_equality(self):
        """Test CalculationStep equality."""
        step1 = CalculationStep(1, "Test", "add", {"a": 1}, 1, "result")
        step2 = CalculationStep(1, "Test", "add", {"a": 1}, 1, "result")

        assert step1 == step2

    @pytest.mark.determinism
    def test_step_hash(self):
        """Test CalculationStep is hashable."""
        step = CalculationStep(1, "Test", "add", {"a": 1}, 1, "result")

        # Should be hashable (frozen dataclass)
        hash_value = hash(step)
        assert isinstance(hash_value, int)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressionPrevention:
    """Test suite for preventing regressions in determinism."""

    @pytest.mark.determinism
    def test_efficiency_output_stability(self, efficiency_calculator, standard_efficiency_input):
        """Test efficiency output values are stable."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # These values should not change without intentional modification
        assert result.ttd_c == 5.0
        assert result.itd_c == 15.0
        assert result.cw_temp_rise_c == 10.0

    @pytest.mark.determinism
    def test_cleanliness_factor_formula(self):
        """Test cleanliness factor formula is stable."""
        actual_u = 3000.0
        design_u = 3500.0

        cf = actual_u / design_u

        # Formula should produce consistent result
        assert abs(cf - 0.857142857) < 1e-6

    @pytest.mark.determinism
    def test_heat_rate_impact_constant(self):
        """Test heat rate impact constant is stable."""
        from calculators.efficiency_calculator import HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG

        # This constant should not change without versioning
        assert HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG == 25.0

    @pytest.mark.determinism
    def test_carbon_factor_constant(self):
        """Test carbon emission factor constant is stable."""
        from calculators.efficiency_calculator import CARBON_EMISSION_FACTOR_KG_CO2_MWH

        # This constant should not change without versioning
        assert CARBON_EMISSION_FACTOR_KG_CO2_MWH == 400.0


# =============================================================================
# MULTI-CALCULATOR CONSISTENCY TESTS
# =============================================================================

class TestMultiCalculatorConsistency:
    """Test consistency across multiple calculators."""

    @pytest.mark.determinism
    def test_shared_constants_consistency(self):
        """Test shared constants are consistent across modules."""
        # Water density used in multiple calculators
        from calculators.efficiency_calculator import calculate_cw_temperature_rise

        # Calculate with known values
        temp_rise_1 = calculate_cw_temperature_rise(100.0, 25000.0)
        temp_rise_2 = calculate_cw_temperature_rise(100.0, 25000.0)

        assert temp_rise_1 == temp_rise_2

    @pytest.mark.determinism
    def test_provenance_version_tracking(self, efficiency_calculator, standard_efficiency_input):
        """Test calculator version is tracked in provenance."""
        _, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert provenance.calculator_version is not None
        assert provenance.calculator_name == "EfficiencyCalculator"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
