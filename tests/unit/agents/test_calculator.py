"""
Comprehensive tests for BaseCalculator class.
Tests calculation caching, unit conversion, deterministic computation, and tracing.
"""

import pytest
from decimal import Decimal
from greenlang.agents.calculator import (
    BaseCalculator, CalculatorConfig, CalculatorResult, UnitConverter, CalculationStep
)
from typing import Dict, Any


class SimpleCalculator(BaseCalculator):
    """Simple test calculator."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Add two numbers."""
        return inputs["a"] + inputs["b"]


class EmissionsCalculator(BaseCalculator):
    """Calculator with unit conversion and tracing."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Calculate CO2 emissions."""
        energy = inputs["energy_kwh"]
        factor = inputs["emission_factor"]

        # Add calculation step
        result = energy * factor

        self.add_calculation_step(
            step_name="emission_calculation",
            formula="energy_kwh * emission_factor",
            inputs=inputs,
            result=result,
            units="kg_co2e"
        )

        return result


class DivisionCalculator(BaseCalculator):
    """Calculator that tests division."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Divide two numbers safely."""
        return self.safe_divide(inputs["numerator"], inputs["denominator"])


class TestCalculatorConfig:
    """Test CalculatorConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = CalculatorConfig(
            name="TestCalc",
            description="Test calculator"
        )
        assert config.precision == 6
        assert config.enable_caching is True
        assert config.cache_size == 128
        assert config.deterministic is True
        assert config.allow_division_by_zero is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = CalculatorConfig(
            name="CustomCalc",
            description="Custom calculator",
            precision=10,
            cache_size=256,
            allow_division_by_zero=True
        )
        assert config.precision == 10
        assert config.cache_size == 256
        assert config.allow_division_by_zero is True

    def test_precision_validation(self):
        """Test precision bounds validation."""
        with pytest.raises(ValueError):
            CalculatorConfig(name="Bad", description="Test", precision=-1)

        with pytest.raises(ValueError):
            CalculatorConfig(name="Bad", description="Test", precision=50)

    def test_cache_size_validation(self):
        """Test cache size validation."""
        with pytest.raises(ValueError):
            CalculatorConfig(name="Bad", description="Test", cache_size=-10)


class TestUnitConverter:
    """Test UnitConverter functionality."""

    def test_same_unit_conversion(self):
        """Test converting to same unit."""
        result = UnitConverter.convert(100, "kWh", "kWh")
        assert result == 100

    def test_energy_conversion(self):
        """Test energy unit conversions."""
        # kWh to MWh
        result = UnitConverter.convert(1000, "kWh", "MWh")
        assert pytest.approx(result, rel=1e-6) == 1.0

        # kWh to J
        result = UnitConverter.convert(1, "kWh", "J")
        assert result == 3600000

    def test_mass_conversion(self):
        """Test mass unit conversions."""
        # kg to t
        result = UnitConverter.convert(1000, "kg", "t")
        assert result == 1.0

        # g to kg
        result = UnitConverter.convert(1000, "g", "kg")
        assert result == 1.0

    def test_volume_conversion(self):
        """Test volume unit conversions."""
        # L to m3
        result = UnitConverter.convert(1000, "L", "m3")
        assert result == 1.0

    def test_unknown_units(self):
        """Test conversion with unknown units."""
        with pytest.raises(ValueError, match="Unknown units"):
            UnitConverter.convert(100, "unknown", "kWh")

    def test_incompatible_units(self):
        """Test conversion between incompatible units."""
        with pytest.raises(ValueError, match="Incompatible units"):
            UnitConverter.convert(100, "kWh", "kg")


class TestBaseCalculator:
    """Test BaseCalculator functionality."""

    def test_simple_calculation(self):
        """Test basic calculation."""
        calc = SimpleCalculator()
        result = calc.run({"inputs": {"a": 10, "b": 20}})

        assert result.success is True
        assert result.result_value == 30
        assert result.data["result"] == 30

    def test_calculation_with_precision(self):
        """Test calculation with precision rounding."""
        config = CalculatorConfig(
            name="PrecisionCalc",
            description="Test",
            precision=2
        )
        calc = SimpleCalculator(config=config)

        result = calc.run({"inputs": {"a": 1.111, "b": 2.222}})

        # Result should be rounded to 2 decimal places
        assert result.success is True
        assert result.precision == 2

    def test_calculation_caching(self):
        """Test that calculations are cached."""
        calc = SimpleCalculator()

        # First calculation
        result1 = calc.run({"inputs": {"a": 10, "b": 20}})
        assert result1.cached is False

        # Second calculation with same inputs (should be cached)
        result2 = calc.run({"inputs": {"a": 10, "b": 20}})
        assert result2.cached is True
        assert result2.result_value == result1.result_value

    def test_cache_miss(self):
        """Test cache miss with different inputs."""
        calc = SimpleCalculator()

        result1 = calc.run({"inputs": {"a": 10, "b": 20}})
        result2 = calc.run({"inputs": {"a": 15, "b": 25}})

        assert result1.cached is False
        assert result2.cached is False
        assert result1.result_value != result2.result_value

    def test_caching_disabled(self):
        """Test calculation with caching disabled."""
        config = CalculatorConfig(
            name="NoCacheCalc",
            description="Test",
            enable_caching=False
        )
        calc = SimpleCalculator(config=config)

        result1 = calc.run({"inputs": {"a": 10, "b": 20}})
        result2 = calc.run({"inputs": {"a": 10, "b": 20}})

        assert result1.cached is False
        assert result2.cached is False

    def test_cache_key_generation(self):
        """Test cache key generation."""
        calc = SimpleCalculator()

        key1 = calc.get_cache_key({"a": 10, "b": 20})
        key2 = calc.get_cache_key({"a": 10, "b": 20})
        key3 = calc.get_cache_key({"a": 20, "b": 10})

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        assert key1 != key3

    def test_clear_cache(self):
        """Test clearing the cache."""
        calc = SimpleCalculator()

        # Cache a result
        calc.run({"inputs": {"a": 10, "b": 20}})
        assert len(calc._calc_cache) > 0

        # Clear cache
        calc.clear_cache()
        assert len(calc._calc_cache) == 0

    def test_cache_size_limit(self):
        """Test cache size limit and LRU eviction."""
        config = CalculatorConfig(
            name="SmallCacheCalc",
            description="Test",
            cache_size=2
        )
        calc = SimpleCalculator(config=config)

        # Fill cache beyond limit
        calc.run({"inputs": {"a": 1, "b": 1}})
        calc.run({"inputs": {"a": 2, "b": 2}})
        calc.run({"inputs": {"a": 3, "b": 3}})  # Should evict oldest

        # Cache should not exceed size limit
        assert len(calc._calc_cache) <= 2

    def test_calculation_steps(self):
        """Test calculation step tracking."""
        calc = EmissionsCalculator()

        result = calc.run({"inputs": {
            "energy_kwh": 1000,
            "emission_factor": 0.385
        }})

        assert result.success is True
        assert len(result.calculation_steps) > 0

        step = result.calculation_steps[0]
        assert step.step_name == "emission_calculation"
        assert step.formula == "energy_kwh * emission_factor"
        assert step.units == "kg_co2e"

    def test_round_decimal(self):
        """Test decimal rounding."""
        calc = SimpleCalculator()

        rounded = calc.round_decimal(3.14159, precision=2)
        assert rounded == Decimal("3.14")

        rounded = calc.round_decimal(2.5, precision=0)
        assert rounded == Decimal("3")

    def test_safe_divide(self):
        """Test safe division."""
        calc = DivisionCalculator()

        # Normal division
        result = calc.run({"inputs": {"numerator": 10, "denominator": 2}})
        assert result.success is True
        assert result.result_value == 5.0

    def test_safe_divide_by_zero_not_allowed(self):
        """Test division by zero when not allowed."""
        calc = DivisionCalculator()

        result = calc.run({"inputs": {"numerator": 10, "denominator": 0}})
        assert result.success is False
        assert "Division by zero" in result.error

    def test_safe_divide_by_zero_allowed(self):
        """Test division by zero when allowed."""
        config = CalculatorConfig(
            name="DivCalc",
            description="Test",
            allow_division_by_zero=True
        )
        calc = DivisionCalculator(config=config)

        result = calc.run({"inputs": {"numerator": 10, "denominator": 0}})
        assert result.success is True
        assert result.result_value is None

    def test_unit_conversion(self):
        """Test unit conversion in calculator."""
        calc = SimpleCalculator()

        # Convert kWh to MWh
        converted = calc.convert_units(1000, "kWh", "MWh")
        assert pytest.approx(converted, rel=1e-6) == 1.0

    def test_validation_enabled(self):
        """Test input validation when enabled."""
        config = CalculatorConfig(
            name="ValidatingCalc",
            description="Test",
            validate_inputs=True
        )
        calc = SimpleCalculator(config=config)

        # Missing inputs key
        result = calc.run({"wrong_key": "value"})
        assert result.success is False

    def test_validation_disabled(self):
        """Test with validation disabled."""
        config = CalculatorConfig(
            name="NoValidationCalc",
            description="Test",
            validate_inputs=False
        )
        calc = SimpleCalculator(config=config)

        # This would normally fail validation
        result = calc.run({"inputs": {}})
        # Will fail in calculation, not validation
        assert result.success is False

    def test_stats_tracking(self):
        """Test that calculator tracks cache statistics."""
        calc = SimpleCalculator()

        # First call - cache miss
        calc.run({"inputs": {"a": 10, "b": 20}})

        # Second call - cache hit
        calc.run({"inputs": {"a": 10, "b": 20}})

        stats = calc.get_stats()
        assert stats["custom_counters"]["cache_hits"] == 1
        assert stats["custom_counters"]["cache_misses"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
