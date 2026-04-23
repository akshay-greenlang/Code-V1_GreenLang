"""
GL-010 EmissionGuardian - RATA Calculator Tests

Comprehensive test suite for EPA 40 CFR Part 75 RATA (Relative Accuracy Test Audit)
calculations. Tests statistical methods, bias detection, and provenance tracking.

Reference: 40 CFR Part 75, Appendix A, Section 7
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from typing import List
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.rata_calculator import (
    T_VALUES_95,
    RATA_PASS_THRESHOLD_STANDARD,
    RATA_PASS_THRESHOLD_ABBREVIATED,
    CalculationTrace,
    RATAResult,
    calculate_mean_difference,
    calculate_standard_deviation,
    calculate_confidence_coefficient,
    calculate_relative_accuracy,
    calculate_bias_test,
    calculate_bias_adjustment_factor,
    perform_rata,
    calculate_provenance_hash,
    _get_t_value,
    _sqrt_decimal,
    _apply_precision,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_cems_values():
    """Sample CEMS measurements for 9-run RATA."""
    return [
        Decimal("100.5"), Decimal("102.3"), Decimal("99.8"),
        Decimal("101.2"), Decimal("100.0"), Decimal("103.1"),
        Decimal("98.9"), Decimal("101.8"), Decimal("100.4"),
    ]


@pytest.fixture
def sample_rm_values():
    """Sample Reference Method measurements for 9-run RATA."""
    return [
        Decimal("101.0"), Decimal("101.5"), Decimal("100.2"),
        Decimal("100.8"), Decimal("99.5"), Decimal("102.8"),
        Decimal("99.2"), Decimal("101.2"), Decimal("100.8"),
    ]


@pytest.fixture
def abbreviated_cems_values():
    """Sample CEMS for 3-run abbreviated RATA."""
    return [Decimal("100.0"), Decimal("102.0"), Decimal("101.0")]


@pytest.fixture
def abbreviated_rm_values():
    """Sample RM for 3-run abbreviated RATA."""
    return [Decimal("100.5"), Decimal("101.5"), Decimal("100.8")]


# =============================================================================
# TEST: EPA T-VALUE CONSTANTS
# =============================================================================

class TestTValueConstants:
    """Test EPA t-value lookup table."""

    def test_t_values_all_df_present(self):
        """T-values should be defined for df 1-30."""
        for df in range(1, 31):
            assert df in T_VALUES_95
            assert T_VALUES_95[df] > Decimal("0")

    def test_t_value_decreases_with_df(self):
        """T-value should decrease as degrees of freedom increase."""
        prev_t = T_VALUES_95[1]
        for df in range(2, 31):
            current_t = T_VALUES_95[df]
            assert current_t < prev_t
            prev_t = current_t

    def test_t_value_df_1(self):
        """t(1) should be 12.706 for 95% confidence."""
        assert T_VALUES_95[1] == Decimal("12.706")

    def test_t_value_df_9(self):
        """t(9) commonly used for 9-run RATA (df=8)."""
        # Actually df = n-1 = 8 for 9 runs
        assert T_VALUES_95[8] == Decimal("2.306")

    def test_get_t_value_normal_df(self):
        """_get_t_value should return correct values."""
        assert _get_t_value(9) == Decimal("2.262")

    def test_get_t_value_large_df(self):
        """Large df should return normal approximation 1.96."""
        assert _get_t_value(100) == Decimal("1.96")

    def test_get_t_value_invalid_df(self):
        """Invalid df should raise ValueError."""
        with pytest.raises(ValueError):
            _get_t_value(0)


# =============================================================================
# TEST: RATA THRESHOLDS
# =============================================================================

class TestRATAThresholds:
    """Test EPA RATA pass/fail thresholds."""

    def test_standard_threshold(self):
        """Standard RATA threshold should be 10%."""
        assert RATA_PASS_THRESHOLD_STANDARD == Decimal("10.0")

    def test_abbreviated_threshold(self):
        """Abbreviated RATA threshold should be 7.5%."""
        assert RATA_PASS_THRESHOLD_ABBREVIATED == Decimal("7.5")


# =============================================================================
# TEST: UTILITY FUNCTIONS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_sqrt_decimal_positive(self):
        """Square root of positive number."""
        result = _sqrt_decimal(Decimal("4"))
        assert abs(float(result) - 2.0) < 0.0001

    def test_sqrt_decimal_zero(self):
        """Square root of zero should be zero."""
        assert _sqrt_decimal(Decimal("0")) == Decimal("0")

    def test_sqrt_decimal_negative_raises(self):
        """Square root of negative should raise."""
        with pytest.raises(ValueError):
            _sqrt_decimal(Decimal("-1"))

    def test_apply_precision_2_decimals(self):
        """Test precision application with 2 decimal places."""
        result = _apply_precision(Decimal("1.2345"), 2)
        assert result == Decimal("1.23")

    def test_apply_precision_0_decimals(self):
        """Test precision application with 0 decimal places."""
        result = _apply_precision(Decimal("1.567"), 0)
        assert result == Decimal("2")

    def test_apply_precision_rounds_half_up(self):
        """Test ROUND_HALF_UP behavior."""
        result = _apply_precision(Decimal("1.235"), 2)
        assert result == Decimal("1.24")


# =============================================================================
# TEST: MEAN DIFFERENCE CALCULATION
# =============================================================================

class TestMeanDifference:
    """Test mean difference calculations."""

    def test_mean_difference_basic(self):
        """Basic mean difference calculation."""
        cems = [Decimal("100"), Decimal("102")]
        rm = [Decimal("99"), Decimal("101")]
        mean_diff, differences = calculate_mean_difference(cems, rm)

        # Differences: 1, 1 -> mean = 1
        assert mean_diff == Decimal("1")
        assert differences == [Decimal("1"), Decimal("1")]

    def test_mean_difference_negative(self):
        """Mean difference can be negative."""
        cems = [Decimal("98"), Decimal("99")]
        rm = [Decimal("100"), Decimal("101")]
        mean_diff, differences = calculate_mean_difference(cems, rm)

        assert mean_diff == Decimal("-2")

    def test_mean_difference_mismatched_length(self):
        """Mismatched list lengths should raise."""
        cems = [Decimal("100")]
        rm = [Decimal("100"), Decimal("101")]

        with pytest.raises(ValueError):
            calculate_mean_difference(cems, rm)

    def test_mean_difference_zero(self):
        """Identical values should give zero mean difference."""
        cems = [Decimal("100"), Decimal("100")]
        rm = [Decimal("100"), Decimal("100")]
        mean_diff, _ = calculate_mean_difference(cems, rm)

        assert mean_diff == Decimal("0")


# =============================================================================
# TEST: STANDARD DEVIATION CALCULATION
# =============================================================================

class TestStandardDeviation:
    """Test standard deviation calculations."""

    def test_standard_deviation_basic(self):
        """Basic standard deviation calculation."""
        differences = [Decimal("1"), Decimal("2"), Decimal("3")]
        mean_diff = Decimal("2")  # Mean of 1,2,3

        std_dev = calculate_standard_deviation(differences, mean_diff)

        # Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 2/2 = 1
        # Std dev = sqrt(1) = 1
        assert abs(float(std_dev) - 1.0) < 0.0001

    def test_standard_deviation_zero(self):
        """Identical differences should give zero std dev."""
        differences = [Decimal("5"), Decimal("5"), Decimal("5")]
        mean_diff = Decimal("5")

        std_dev = calculate_standard_deviation(differences, mean_diff)
        assert std_dev == Decimal("0")

    def test_standard_deviation_requires_two_points(self):
        """Need at least 2 points for sample std dev."""
        with pytest.raises(ValueError):
            calculate_standard_deviation([Decimal("1")], Decimal("1"))


# =============================================================================
# TEST: CONFIDENCE COEFFICIENT CALCULATION
# =============================================================================

class TestConfidenceCoefficient:
    """Test confidence coefficient calculations."""

    def test_confidence_coefficient_formula(self):
        """Test CC = t * Sd / sqrt(n) formula."""
        std_dev = Decimal("2.0")
        num_runs = 9

        # t(8) = 2.306, sqrt(9) = 3
        # CC = 2.306 * 2.0 / 3 = 1.537...
        cc = calculate_confidence_coefficient(std_dev, num_runs)

        expected = Decimal("2.306") * std_dev / Decimal("3")
        assert abs(float(cc) - float(expected)) < 0.01

    def test_confidence_coefficient_zero_std(self):
        """Zero std dev should give zero CC."""
        cc = calculate_confidence_coefficient(Decimal("0"), 9)
        assert cc == Decimal("0")


# =============================================================================
# TEST: RELATIVE ACCURACY CALCULATION
# =============================================================================

class TestRelativeAccuracy:
    """Test relative accuracy calculations."""

    def test_relative_accuracy_formula(self):
        """Test RA = (|d_bar| + CC) / RM_mean * 100."""
        mean_diff = Decimal("2.0")
        cc = Decimal("1.5")
        rm_mean = Decimal("100.0")

        # RA = (2.0 + 1.5) / 100 * 100 = 3.5%
        ra = calculate_relative_accuracy(mean_diff, cc, rm_mean)
        assert ra == Decimal("3.50")

    def test_relative_accuracy_negative_mean_diff(self):
        """Negative mean diff should use absolute value."""
        mean_diff = Decimal("-2.0")
        cc = Decimal("1.5")
        rm_mean = Decimal("100.0")

        ra = calculate_relative_accuracy(mean_diff, cc, rm_mean)
        assert ra == Decimal("3.50")

    def test_relative_accuracy_zero_rm_mean_raises(self):
        """Zero RM mean should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_relative_accuracy(Decimal("1"), Decimal("1"), Decimal("0"))


# =============================================================================
# TEST: BIAS TEST CALCULATION
# =============================================================================

class TestBiasTest:
    """Test bias test (paired t-test) calculations."""

    def test_bias_test_no_bias(self):
        """Differences with small mean should not show bias."""
        differences = [Decimal("0.1"), Decimal("-0.1"), Decimal("0.2"),
                      Decimal("-0.2"), Decimal("0.1"), Decimal("-0.1"),
                      Decimal("0.1"), Decimal("-0.1"), Decimal("0.0")]
        mean_diff = sum(differences) / 9
        std_dev = calculate_standard_deviation(differences, mean_diff)

        bias_significant, t_stat = calculate_bias_test(
            differences, mean_diff, std_dev, 9
        )

        # Should not be significant with small mean
        assert not bias_significant

    def test_bias_test_with_bias(self):
        """Consistent positive differences should show bias."""
        differences = [Decimal("5"), Decimal("6"), Decimal("5.5"),
                      Decimal("5.2"), Decimal("5.8"), Decimal("5.1"),
                      Decimal("5.3"), Decimal("5.7"), Decimal("5.4")]
        mean_diff = sum(differences) / 9
        std_dev = calculate_standard_deviation(differences, mean_diff)

        bias_significant, t_stat = calculate_bias_test(
            differences, mean_diff, std_dev, 9
        )

        # Should be significant with consistent positive differences
        assert bias_significant
        assert t_stat > 0

    def test_bias_test_zero_std_dev(self):
        """Zero std dev should return no bias."""
        differences = [Decimal("1")] * 9

        bias_significant, t_stat = calculate_bias_test(
            differences, Decimal("1"), Decimal("0"), 9
        )

        assert not bias_significant
        assert t_stat == Decimal("0")


# =============================================================================
# TEST: BIAS ADJUSTMENT FACTOR
# =============================================================================

class TestBiasAdjustmentFactor:
    """Test bias adjustment factor calculations."""

    def test_baf_basic(self):
        """BAF = RM_mean / CEMS_mean."""
        rm_mean = Decimal("102.0")
        cems_mean = Decimal("100.0")

        baf = calculate_bias_adjustment_factor(cems_mean, rm_mean)

        assert baf == Decimal("1.0200")

    def test_baf_no_bias(self):
        """Equal means should give BAF = 1.0."""
        baf = calculate_bias_adjustment_factor(
            Decimal("100.0"), Decimal("100.0")
        )
        assert baf == Decimal("1.0000")

    def test_baf_zero_cems_raises(self):
        """Zero CEMS mean should raise."""
        with pytest.raises(ValueError):
            calculate_bias_adjustment_factor(Decimal("0"), Decimal("100"))


# =============================================================================
# TEST: COMPLETE RATA CALCULATION
# =============================================================================

class TestPerformRATA:
    """Test complete RATA calculation."""

    def test_standard_rata_pass(self, sample_cems_values, sample_rm_values):
        """Standard 9-run RATA that passes."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        assert result.num_runs == 9
        assert result.test_type == "standard"
        assert result.is_valid
        # With similar values, should pass
        assert result.relative_accuracy <= Decimal("10.0")
        assert result.passed

    def test_standard_rata_has_trace(self, sample_cems_values, sample_rm_values):
        """RATA should include calculation trace."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        assert len(result.calculation_trace) >= 6
        # Check trace steps
        steps = [t.step_number for t in result.calculation_trace]
        assert 1 in steps  # CEMS mean
        assert 2 in steps  # RM mean
        assert 3 in steps  # Mean difference
        assert 4 in steps  # Std dev
        assert 5 in steps  # Confidence coefficient
        assert 6 in steps  # Relative accuracy

    def test_abbreviated_rata(self, abbreviated_cems_values, abbreviated_rm_values):
        """Abbreviated 3-run RATA."""
        result = perform_rata(
            abbreviated_cems_values,
            abbreviated_rm_values,
            test_type="abbreviated"
        )

        assert result.num_runs == 3
        assert result.test_type == "abbreviated"
        # Abbreviated threshold is 7.5%
        assert result.passed == (result.relative_accuracy <= Decimal("7.5"))

    def test_rata_fail_high_ra(self):
        """RATA with high relative accuracy should fail."""
        # Create values with significant difference
        cems = [Decimal("100")] * 9
        rm = [Decimal("120")] * 9  # 20% higher

        result = perform_rata(cems, rm)

        # RA should be high due to 20% systematic difference
        assert result.relative_accuracy > Decimal("10.0")
        assert not result.passed

    def test_rata_provenance_hash(self, sample_cems_values, sample_rm_values):
        """RATA result should have provenance hash."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        assert len(result.provenance_hash) == 64  # SHA-256 hex
        assert result.provenance_hash.isalnum()

    def test_rata_deterministic(self, sample_cems_values, sample_rm_values):
        """Same inputs should produce same results."""
        result1 = perform_rata(sample_cems_values, sample_rm_values)
        result2 = perform_rata(sample_cems_values, sample_rm_values)

        assert result1.relative_accuracy == result2.relative_accuracy
        assert result1.mean_difference == result2.mean_difference
        assert result1.passed == result2.passed

    def test_rata_validation_messages(self):
        """Validation messages for incorrect run counts."""
        cems = [Decimal("100")] * 5  # 5 runs, not 9
        rm = [Decimal("100")] * 5

        result = perform_rata(cems, rm, test_type="standard")

        assert not result.is_valid
        assert len(result.validation_messages) > 0
        assert "9+" in result.validation_messages[0]

    def test_rata_mismatched_lengths_raises(self):
        """Mismatched list lengths should raise."""
        cems = [Decimal("100")] * 9
        rm = [Decimal("100")] * 8

        with pytest.raises(ValueError):
            perform_rata(cems, rm)


# =============================================================================
# TEST: PROVENANCE HASH
# =============================================================================

class TestProvenanceHash:
    """Test SHA-256 provenance hash generation."""

    def test_provenance_hash_deterministic(self):
        """Same inputs should produce same hash."""
        inputs = {"a": "1", "b": "2"}
        outputs = {"result": "3"}
        trace = [CalculationTrace(
            step_number=1, description="Test", formula="a+b",
            inputs={"a": "1"}, output="result", output_value=Decimal("3")
        )]

        hash1 = calculate_provenance_hash("test_func", inputs, outputs, trace)
        hash2 = calculate_provenance_hash("test_func", inputs, outputs, trace)

        assert hash1 == hash2

    def test_provenance_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        trace = []

        hash1 = calculate_provenance_hash(
            "test", {"a": "1"}, {"r": "1"}, trace
        )
        hash2 = calculate_provenance_hash(
            "test", {"a": "2"}, {"r": "1"}, trace
        )

        assert hash1 != hash2

    def test_provenance_hash_length(self):
        """Provenance hash should be SHA-256 (64 hex chars)."""
        hash_val = calculate_provenance_hash("test", {}, {}, [])

        assert len(hash_val) == 64


# =============================================================================
# TEST: CALCULATION TRACE
# =============================================================================

class TestCalculationTraceStruct:
    """Test CalculationTrace structure."""

    def test_trace_structure(self):
        """Trace should have all required fields."""
        trace = CalculationTrace(
            step_number=1,
            description="Calculate mean",
            formula="sum(x) / n",
            inputs={"values": "[1,2,3]", "n": "3"},
            output="mean",
            output_value=Decimal("2"),
        )

        assert trace.step_number == 1
        assert "mean" in trace.description.lower()
        assert trace.output_value == Decimal("2")

    def test_trace_timestamp_auto(self):
        """Timestamp should auto-populate."""
        trace = CalculationTrace(
            step_number=1, description="Test", formula="x",
            inputs={}, output="y", output_value=Decimal("1")
        )

        assert trace.timestamp is not None
        assert len(trace.timestamp) > 0


# =============================================================================
# TEST: RATA RESULT STRUCTURE
# =============================================================================

class TestRATAResultStructure:
    """Test RATAResult structure."""

    def test_result_has_all_fields(self, sample_cems_values, sample_rm_values):
        """RATAResult should have all EPA-required fields."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        # Core results
        assert hasattr(result, "relative_accuracy")
        assert hasattr(result, "mean_difference")
        assert hasattr(result, "standard_deviation")
        assert hasattr(result, "confidence_coefficient")

        # Means
        assert hasattr(result, "reference_method_mean")
        assert hasattr(result, "cems_mean")

        # Pass/fail
        assert hasattr(result, "passed")
        assert hasattr(result, "test_type")
        assert hasattr(result, "num_runs")

        # Bias
        assert hasattr(result, "bias_test_passed")
        assert hasattr(result, "bias_adjustment_factor")

        # Provenance
        assert hasattr(result, "calculation_trace")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "formula_reference")

    def test_result_formula_reference(self, sample_cems_values, sample_rm_values):
        """Result should reference EPA regulation."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        assert "40 CFR Part 75" in result.formula_reference
        assert "Appendix A" in result.formula_reference


# =============================================================================
# TEST: DECIMAL PRECISION
# =============================================================================

class TestDecimalPrecision:
    """Test Decimal precision for regulatory compliance."""

    def test_result_precision(self, sample_cems_values, sample_rm_values):
        """Results should have appropriate decimal precision."""
        result = perform_rata(sample_cems_values, sample_rm_values)

        # RA should be 2 decimal places
        ra_str = str(result.relative_accuracy)
        if "." in ra_str:
            decimal_places = len(ra_str.split(".")[-1])
            assert decimal_places <= 2

    def test_no_floating_point_errors(self):
        """Decimal should avoid floating point errors."""
        # Classic 0.1 + 0.2 problem
        a = Decimal("0.1")
        b = Decimal("0.2")
        assert a + b == Decimal("0.3")


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_differences(self):
        """Very small differences should still calculate."""
        cems = [Decimal("100.001")] * 9
        rm = [Decimal("100.000")] * 9

        result = perform_rata(cems, rm)

        assert result.relative_accuracy >= Decimal("0")
        assert result.is_valid

    def test_minimum_runs_for_standard(self):
        """Exactly 9 runs should be valid for standard."""
        cems = [Decimal("100")] * 9
        rm = [Decimal("100")] * 9

        result = perform_rata(cems, rm, test_type="standard")

        assert result.is_valid
        assert result.num_runs == 9

    def test_exactly_three_runs_abbreviated(self):
        """Exactly 3 runs should be valid for abbreviated."""
        cems = [Decimal("100")] * 3
        rm = [Decimal("100")] * 3

        result = perform_rata(cems, rm, test_type="abbreviated")

        assert result.is_valid
        assert result.num_runs == 3

    def test_zero_values(self):
        """Zero values should calculate (though RA would be undefined)."""
        cems = [Decimal("0")] * 9
        rm = [Decimal("0")] * 9

        # RM mean of zero would cause division error
        with pytest.raises(ValueError):
            perform_rata(cems, rm)

