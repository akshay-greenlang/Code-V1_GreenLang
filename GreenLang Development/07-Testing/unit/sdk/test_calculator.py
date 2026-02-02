# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang BaseCalculator.

Tests cover:
- Calculator initialization and configuration
- Calculation precision and rounding
- Calculation caching
- Calculation steps tracking
- Unit conversion
- safe_divide edge cases
- Deterministic calculations
- Input validation
- Error handling
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import Mock, patch

from greenlang.agents.calculator import (
    BaseCalculator,
    CalculatorConfig,
    CalculatorResult,
    CalculationStep,
    UnitConverter,
)


# Test Calculator Implementations

class SimpleCalculator(BaseCalculator):
    """Simple calculator that adds two numbers."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Add a and b."""
        return inputs["a"] + inputs["b"]


class MultiStepCalculator(BaseCalculator):
    """Calculator that tracks multiple steps."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Perform multi-step calculation."""
        a = inputs["a"]
        b = inputs["b"]

        # Step 1: Add
        sum_val = a + b
        self.add_calculation_step(
            step_name="Addition",
            formula="a + b",
            inputs={"a": a, "b": b},
            result=sum_val
        )

        # Step 2: Multiply by 2
        doubled = sum_val * 2
        self.add_calculation_step(
            step_name="Doubling",
            formula="(a + b) * 2",
            inputs={"sum": sum_val},
            result=doubled
        )

        return doubled


class ValidationCalculator(BaseCalculator):
    """Calculator with input validation."""

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs contain required fields."""
        return "value" in inputs and isinstance(inputs["value"], (int, float))

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Calculate square of value."""
        return inputs["value"] ** 2


class PrecisionCalculator(BaseCalculator):
    """Calculator for testing precision."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Divide a by b with high precision."""
        return Decimal(str(inputs["a"])) / Decimal(str(inputs["b"]))


class SafeDivideCalculator(BaseCalculator):
    """Calculator using safe_divide."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Safely divide a by b."""
        return self.safe_divide(inputs["a"], inputs["b"])


class UnitConversionCalculator(BaseCalculator):
    """Calculator with unit conversion."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Convert value between units."""
        value = inputs["value"]
        from_unit = inputs["from_unit"]
        to_unit = inputs["to_unit"]

        return self.convert_units(value, from_unit, to_unit)


# Test Classes

@pytest.mark.unit
class TestCalculatorConfig:
    """Test CalculatorConfig model."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = CalculatorConfig(
            name="TestCalc",
            description="Test calculator"
        )

        assert config.precision == 6
        assert config.enable_caching is True
        assert config.cache_size == 128
        assert config.validate_inputs is True
        assert config.deterministic is True
        assert config.allow_division_by_zero is False
        assert config.rounding_mode == "ROUND_HALF_UP"

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = CalculatorConfig(
            name="CustomCalc",
            description="Custom calculator",
            precision=10,
            enable_caching=False,
            cache_size=256,
            validate_inputs=False,
            deterministic=False,
            allow_division_by_zero=True,
            rounding_mode="ROUND_DOWN"
        )

        assert config.precision == 10
        assert config.enable_caching is False
        assert config.cache_size == 256
        assert config.validate_inputs is False
        assert config.allow_division_by_zero is True

    def test_config_precision_validation(self):
        """Test precision validation."""
        with pytest.raises(ValueError, match="precision must be between 0 and 28"):
            CalculatorConfig(
                name="Test",
                description="Test",
                precision=-1
            )

        with pytest.raises(ValueError, match="precision must be between 0 and 28"):
            CalculatorConfig(
                name="Test",
                description="Test",
                precision=29
            )

    def test_config_cache_size_validation(self):
        """Test cache size validation."""
        with pytest.raises(ValueError, match="cache_size must be non-negative"):
            CalculatorConfig(
                name="Test",
                description="Test",
                cache_size=-1
            )


@pytest.mark.unit
class TestCalculationStep:
    """Test CalculationStep model."""

    def test_step_creation(self):
        """Test creating a calculation step."""
        step = CalculationStep(
            step_name="Addition",
            formula="a + b",
            inputs={"a": 5, "b": 3},
            result=8,
            units="kg"
        )

        assert step.step_name == "Addition"
        assert step.formula == "a + b"
        assert step.inputs == {"a": 5, "b": 3}
        assert step.result == 8
        assert step.units == "kg"
        assert step.timestamp is not None

    def test_step_without_units(self):
        """Test step without units."""
        step = CalculationStep(
            step_name="Test",
            formula="x * y",
            inputs={},
            result=10
        )

        assert step.units is None


@pytest.mark.unit
class TestCalculatorResult:
    """Test CalculatorResult model."""

    def test_result_success(self):
        """Test successful calculator result."""
        result = CalculatorResult(
            success=True,
            result_value=42.5,
            precision=6,
            cached=False
        )

        assert result.success is True
        assert result.result_value == 42.5
        assert result.precision == 6
        assert result.cached is False
        assert result.calculation_steps == []

    def test_result_with_steps(self):
        """Test result with calculation steps."""
        step1 = CalculationStep(
            step_name="Step1",
            formula="a + b",
            inputs={},
            result=10
        )
        step2 = CalculationStep(
            step_name="Step2",
            formula="x * 2",
            inputs={},
            result=20
        )

        result = CalculatorResult(
            success=True,
            result_value=20,
            calculation_steps=[step1, step2]
        )

        assert len(result.calculation_steps) == 2
        assert result.calculation_steps[0].step_name == "Step1"

    def test_result_cached(self):
        """Test cached result."""
        result = CalculatorResult(
            success=True,
            result_value=100,
            cached=True
        )

        assert result.cached is True


@pytest.mark.unit
class TestCalculatorInitialization:
    """Test calculator initialization."""

    def test_initialization_defaults(self):
        """Test calculator initializes with defaults."""
        calc = SimpleCalculator()

        assert calc.config is not None
        assert calc.config.precision == 6
        assert calc._calc_cache == {}
        assert calc._calculation_steps == []

    def test_initialization_custom_config(self):
        """Test calculator with custom config."""
        config = CalculatorConfig(
            name="CustomCalc",
            description="Custom",
            precision=10,
            cache_size=256
        )
        calc = SimpleCalculator(config)

        assert calc.config.precision == 10
        assert calc.config.cache_size == 256


@pytest.mark.unit
class TestCalculation:
    """Test basic calculation functionality."""

    def test_simple_calculation(self):
        """Test simple calculation."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 5, "b": 3}})

        assert result.success is True
        assert result.result_value == 8

    def test_calculation_with_floats(self):
        """Test calculation with floating point numbers."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 5.5, "b": 3.2}})

        assert result.success is True
        assert result.result_value == pytest.approx(8.7)

    def test_calculation_returns_data(self):
        """Test calculation returns data dict."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 10, "b": 20}})

        assert result.success is True
        assert result.data["result"] == 30
        assert result.data["inputs"] == {"a": 10, "b": 20}


@pytest.mark.unit
class TestPrecision:
    """Test calculation precision."""

    def test_round_decimal_default_precision(self):
        """Test rounding with default precision."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            precision=6
        )
        calc = SimpleCalculator(config)

        rounded = calc.round_decimal(3.123456789)
        assert rounded == Decimal("3.123457")

    def test_round_decimal_custom_precision(self):
        """Test rounding with custom precision."""
        calc = SimpleCalculator()

        rounded = calc.round_decimal(3.123456789, precision=2)
        assert rounded == Decimal("3.12")

    def test_round_decimal_zero_precision(self):
        """Test rounding with zero precision."""
        calc = SimpleCalculator()

        rounded = calc.round_decimal(3.123456789, precision=0)
        assert rounded == Decimal("4")

    def test_precision_applied_to_result(self):
        """Test precision is applied to calculation result."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            precision=2
        )
        calc = PrecisionCalculator(config)

        result = calc.run({"inputs": {"a": 10, "b": 3}})

        assert result.success is True
        assert result.result_value == pytest.approx(3.33, abs=0.01)

    def test_high_precision_calculation(self):
        """Test high precision calculation."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            precision=10
        )
        calc = PrecisionCalculator(config)

        result = calc.run({"inputs": {"a": 1, "b": 3}})

        assert result.success is True
        # Should have 10 decimal places
        assert result.result_value == pytest.approx(0.3333333333, abs=1e-10)


@pytest.mark.unit
class TestCaching:
    """Test calculation caching."""

    def test_cache_hit(self):
        """Test cache hit on repeated calculation."""
        calc = SimpleCalculator()

        # First call - cache miss
        result1 = calc.run({"inputs": {"a": 5, "b": 3}})
        assert result1.success is True
        assert result1.cached is False

        # Second call - cache hit
        result2 = calc.run({"inputs": {"a": 5, "b": 3}})
        assert result2.success is True
        assert result2.cached is True
        assert result2.result_value == result1.result_value

    def test_cache_different_inputs(self):
        """Test cache miss with different inputs."""
        calc = SimpleCalculator()

        result1 = calc.run({"inputs": {"a": 5, "b": 3}})
        result2 = calc.run({"inputs": {"a": 10, "b": 20}})

        assert result1.cached is False
        assert result2.cached is False
        assert result1.result_value != result2.result_value

    def test_cache_disabled(self):
        """Test caching disabled."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            enable_caching=False
        )
        calc = SimpleCalculator(config)

        result1 = calc.run({"inputs": {"a": 5, "b": 3}})
        result2 = calc.run({"inputs": {"a": 5, "b": 3}})

        assert result1.cached is False
        assert result2.cached is False

    def test_cache_size_limit(self):
        """Test cache respects size limit."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            cache_size=3
        )
        calc = SimpleCalculator(config)

        # Fill cache
        calc.run({"inputs": {"a": 1, "b": 1}})
        calc.run({"inputs": {"a": 2, "b": 2}})
        calc.run({"inputs": {"a": 3, "b": 3}})

        assert len(calc._calc_cache) == 3

        # Add one more - should evict oldest
        calc.run({"inputs": {"a": 4, "b": 4}})

        assert len(calc._calc_cache) == 3

    def test_get_cache_key(self):
        """Test cache key generation."""
        calc = SimpleCalculator()

        key1 = calc.get_cache_key({"a": 5, "b": 3})
        key2 = calc.get_cache_key({"a": 5, "b": 3})
        key3 = calc.get_cache_key({"a": 3, "b": 5})

        # Same inputs should produce same key
        assert key1 == key2
        # Different inputs should produce different keys
        assert key1 != key3

    def test_cache_key_order_independent(self):
        """Test cache key is order-independent."""
        calc = SimpleCalculator()

        key1 = calc.get_cache_key({"a": 5, "b": 3})
        key2 = calc.get_cache_key({"b": 3, "a": 5})

        # Different order should produce same key
        assert key1 == key2

    def test_clear_cache(self):
        """Test clearing the cache."""
        calc = SimpleCalculator()

        # Add to cache
        calc.run({"inputs": {"a": 5, "b": 3}})
        assert len(calc._calc_cache) > 0

        # Clear
        calc.clear_cache()
        assert len(calc._calc_cache) == 0

    def test_cache_tracks_hits_misses(self):
        """Test cache hit/miss tracking."""
        calc = SimpleCalculator()

        # First call - miss
        calc.run({"inputs": {"a": 5, "b": 3}})
        stats1 = calc.get_stats()
        assert stats1["custom_counters"]["cache_misses"] == 1
        assert stats1["custom_counters"].get("cache_hits", 0) == 0

        # Second call - hit
        calc.run({"inputs": {"a": 5, "b": 3}})
        stats2 = calc.get_stats()
        assert stats2["custom_counters"]["cache_hits"] == 1
        assert stats2["custom_counters"]["cache_misses"] == 1


@pytest.mark.unit
class TestCalculationSteps:
    """Test calculation steps tracking."""

    def test_add_calculation_step(self):
        """Test adding calculation steps."""
        calc = MultiStepCalculator()
        result = calc.run({"inputs": {"a": 5, "b": 3}})

        assert result.success is True
        assert len(result.calculation_steps) == 2

        step1 = result.calculation_steps[0]
        assert step1.step_name == "Addition"
        assert step1.formula == "a + b"
        assert step1.result == 8

        step2 = result.calculation_steps[1]
        assert step2.step_name == "Doubling"
        assert step2.result == 16

    def test_steps_cleared_between_runs(self):
        """Test calculation steps are cleared between runs."""
        calc = MultiStepCalculator()

        result1 = calc.run({"inputs": {"a": 5, "b": 3}})
        assert len(result1.calculation_steps) == 2

        result2 = calc.run({"inputs": {"a": 10, "b": 20}})
        assert len(result2.calculation_steps) == 2

        # Steps should be for second calculation only
        assert result2.calculation_steps[0].result == 30

    def test_steps_with_units(self):
        """Test adding steps with units."""
        calc = SimpleCalculator()
        calc.add_calculation_step(
            step_name="Energy",
            formula="power * time",
            inputs={"power": 100, "time": 5},
            result=500,
            units="kWh"
        )

        assert len(calc._calculation_steps) == 1
        assert calc._calculation_steps[0].units == "kWh"


@pytest.mark.unit
class TestSafeDivide:
    """Test safe division functionality."""

    def test_safe_divide_normal(self):
        """Test safe divide with normal values."""
        calc = SafeDivideCalculator()
        result = calc.run({"inputs": {"a": 10, "b": 2}})

        assert result.success is True
        assert result.result_value == 5.0

    def test_safe_divide_by_zero_not_allowed(self):
        """Test safe divide raises on zero when not allowed."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            allow_division_by_zero=False
        )
        calc = SafeDivideCalculator(config)

        result = calc.run({"inputs": {"a": 10, "b": 0}})

        assert result.success is False
        assert "division by zero" in result.error.lower()

    def test_safe_divide_by_zero_allowed(self):
        """Test safe divide returns None when zero allowed."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            allow_division_by_zero=True
        )
        calc = SafeDivideCalculator(config)

        result = calc.run({"inputs": {"a": 10, "b": 0}})

        assert result.success is True
        assert result.result_value is None

    def test_safe_divide_negative_values(self):
        """Test safe divide with negative values."""
        calc = SafeDivideCalculator()

        result = calc.run({"inputs": {"a": -10, "b": 2}})
        assert result.result_value == -5.0

        result = calc.run({"inputs": {"a": 10, "b": -2}})
        assert result.result_value == -5.0

    def test_safe_divide_fractional(self):
        """Test safe divide with fractional result."""
        calc = SafeDivideCalculator()
        result = calc.run({"inputs": {"a": 7, "b": 3}})

        assert result.success is True
        assert result.result_value == pytest.approx(2.333333, abs=0.01)


@pytest.mark.unit
class TestUnitConverter:
    """Test UnitConverter functionality."""

    def test_convert_same_unit(self):
        """Test conversion to same unit."""
        result = UnitConverter.convert(100, "kg", "kg")
        assert result == 100

    def test_convert_energy_units(self):
        """Test energy unit conversions."""
        # kWh to J
        result = UnitConverter.convert(1, "kWh", "J")
        assert result == 3600000

        # MWh to kWh
        result = UnitConverter.convert(1, "MWh", "kWh")
        assert result == 1000

    def test_convert_mass_units(self):
        """Test mass unit conversions."""
        # kg to g
        result = UnitConverter.convert(1, "kg", "g")
        assert result == 1000

        # t to kg
        result = UnitConverter.convert(1, "t", "kg")
        assert result == 1000

        # g to kg
        result = UnitConverter.convert(1000, "g", "kg")
        assert result == 1

    def test_convert_volume_units(self):
        """Test volume unit conversions."""
        # L to m3
        result = UnitConverter.convert(1000, "L", "m3")
        assert result == 1

        # m3 to L
        result = UnitConverter.convert(1, "m3", "L")
        assert result == 1000

    def test_convert_unknown_unit(self):
        """Test conversion with unknown unit."""
        with pytest.raises(ValueError, match="Unknown units"):
            UnitConverter.convert(100, "unknown", "kg")

    def test_convert_incompatible_units(self):
        """Test conversion between incompatible units."""
        with pytest.raises(ValueError, match="Incompatible units"):
            UnitConverter.convert(100, "kg", "kWh")


@pytest.mark.unit
class TestUnitConversion:
    """Test calculator unit conversion."""

    def test_convert_units_method(self):
        """Test convert_units method."""
        calc = UnitConversionCalculator()
        result = calc.run({
            "inputs": {
                "value": 1000,
                "from_unit": "g",
                "to_unit": "kg"
            }
        })

        assert result.success is True
        assert result.result_value == 1.0

    def test_convert_energy(self):
        """Test energy conversion."""
        calc = UnitConversionCalculator()
        result = calc.run({
            "inputs": {
                "value": 1,
                "from_unit": "kWh",
                "to_unit": "MWh"
            }
        })

        assert result.success is True
        assert result.result_value == 0.001

    def test_convert_invalid_units(self):
        """Test conversion with invalid units."""
        calc = UnitConversionCalculator()
        result = calc.run({
            "inputs": {
                "value": 100,
                "from_unit": "kg",
                "to_unit": "kWh"
            }
        })

        assert result.success is False
        assert "incompatible" in result.error.lower()


@pytest.mark.unit
class TestInputValidation:
    """Test input validation."""

    def test_validate_input_success(self):
        """Test input validation with valid input."""
        calc = SimpleCalculator()
        assert calc.validate_input({"inputs": {"a": 1, "b": 2}}) is True

    def test_validate_input_missing_inputs_key(self):
        """Test validation fails without inputs key."""
        calc = SimpleCalculator()
        assert calc.validate_input({}) is False
        assert calc.validate_input({"other": "data"}) is False

    def test_validate_input_wrong_type(self):
        """Test validation fails with wrong inputs type."""
        calc = SimpleCalculator()
        assert calc.validate_input({"inputs": "not a dict"}) is False
        assert calc.validate_input({"inputs": [1, 2, 3]}) is False

    def test_custom_validation(self):
        """Test custom validation logic."""
        calc = ValidationCalculator()

        # Valid
        result = calc.run({"inputs": {"value": 5}})
        assert result.success is True

        # Invalid - missing field
        result = calc.run({"inputs": {}})
        assert result.success is False

        # Invalid - wrong type
        result = calc.run({"inputs": {"value": "not a number"}})
        assert result.success is False

    def test_validation_disabled(self):
        """Test calculator with validation disabled."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            validate_inputs=False
        )
        calc = ValidationCalculator(config)

        # Should not validate even with invalid input
        result = calc.run({"inputs": {}})
        # Will fail in calculation, but not in validation
        assert result.success is False


@pytest.mark.unit
class TestDeterminism:
    """Test deterministic calculations."""

    def test_deterministic_results(self):
        """Test same inputs produce same results."""
        config = CalculatorConfig(
            name="Test",
            description="Test",
            deterministic=True,
            enable_caching=False  # Disable cache to test actual calculation
        )
        calc = SimpleCalculator(config)

        results = []
        for _ in range(10):
            result = calc.run({"inputs": {"a": 5.5, "b": 3.2}})
            results.append(result.result_value)

        # All results should be exactly the same
        assert all(r == results[0] for r in results)

    def test_cache_key_deterministic(self):
        """Test cache keys are deterministic."""
        calc = SimpleCalculator()

        keys = []
        for _ in range(10):
            key = calc.get_cache_key({"a": 5, "b": 3})
            keys.append(key)

        # All keys should be the same
        assert all(k == keys[0] for k in keys)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling."""

    def test_calculation_exception(self):
        """Test calculation exception is caught."""
        class FailingCalculator(BaseCalculator):
            def calculate(self, inputs):
                raise ValueError("Calculation failed")

        calc = FailingCalculator()
        result = calc.run({"inputs": {"a": 1}})

        assert result.success is False
        assert "Calculation failed" in result.error

    def test_no_inputs_provided(self):
        """Test execution without inputs."""
        calc = SimpleCalculator()
        result = calc.run({})

        assert result.success is False
        assert "no inputs" in result.error.lower()

    def test_empty_inputs(self):
        """Test execution with empty inputs dict."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {}})

        assert result.success is False


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_values(self):
        """Test calculation with zero values."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 0, "b": 0}})

        assert result.success is True
        assert result.result_value == 0

    def test_negative_values(self):
        """Test calculation with negative values."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": -5, "b": 3}})

        assert result.success is True
        assert result.result_value == -2

    def test_very_large_numbers(self):
        """Test calculation with very large numbers."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 1e15, "b": 1e15}})

        assert result.success is True
        assert result.result_value == 2e15

    def test_very_small_numbers(self):
        """Test calculation with very small numbers."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 1e-10, "b": 1e-10}})

        assert result.success is True
        assert result.result_value == pytest.approx(2e-10)

    def test_mixed_types(self):
        """Test calculation with mixed int/float types."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 5, "b": 3.5}})

        assert result.success is True
        assert result.result_value == 8.5

    def test_concurrent_calculations(self):
        """Test concurrent calculator executions."""
        import threading

        calc = SimpleCalculator()
        results = []
        errors = []

        def run_calc(a, b):
            try:
                result = calc.run({"inputs": {"a": a, "b": b}})
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_calc, args=(i, i+1))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0
        assert all(r.success for r in results)
