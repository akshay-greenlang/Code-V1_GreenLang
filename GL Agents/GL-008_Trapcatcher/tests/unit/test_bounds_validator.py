# -*- coding: utf-8 -*-
"""
Unit tests for GL-008 TRAPCATCHER Bounds Validator.

Comprehensive tests for physical bounds validation including
explicit range checks for pressure (0-25 bar), temperature (273-373 K),
and acoustic levels (0-120 dB).

Author: GL-BackendDeveloper
Date: December 2025
"""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.bounds_validator import (
    BoundsValidator,
    BoundsViolation,
    BoundsValidationResult,
    PhysicalBounds,
    BoundsViolationSeverity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator():
    """Create bounds validator instance."""
    return BoundsValidator(strict_mode=False)


@pytest.fixture
def strict_validator():
    """Create strict bounds validator."""
    return BoundsValidator(strict_mode=True)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestBoundsValidatorInitialization:
    """Test validator initialization."""

    def test_create_validator(self):
        """Test basic validator creation."""
        validator = BoundsValidator()
        assert validator is not None
        assert validator.strict_mode is True

    def test_create_non_strict_validator(self):
        """Test non-strict validator creation."""
        validator = BoundsValidator(strict_mode=False)
        assert validator.strict_mode is False

    def test_bounds_defined(self, validator):
        """Test that all required bounds are defined."""
        assert "pressure_bar" in validator.BOUNDS
        assert "temperature_k" in validator.BOUNDS
        assert "acoustic_db" in validator.BOUNDS


# =============================================================================
# PRESSURE VALIDATION TESTS (0-25 bar)
# =============================================================================

class TestPressureValidation:
    """Test pressure bounds validation (0-25 bar)."""

    def test_valid_pressure(self, validator):
        """Test valid pressure within range."""
        is_valid, violation = validator.validate_value("pressure_bar", 10.0)
        assert is_valid is True
        assert violation is None

    def test_pressure_at_minimum(self, validator):
        """Test pressure at minimum boundary (0 bar)."""
        is_valid, violation = validator.validate_value("pressure_bar", 0.0)
        # Should generate warning near boundary
        assert is_valid is True

    def test_pressure_at_maximum(self, validator):
        """Test pressure at maximum boundary (25 bar)."""
        is_valid, violation = validator.validate_value("pressure_bar", 25.0)
        # Should generate warning near boundary
        assert is_valid is True

    def test_pressure_exceeds_maximum(self, validator):
        """Test pressure exceeding maximum (30 bar > 25 bar)."""
        is_valid, violation = validator.validate_value("pressure_bar", 30.0)
        assert is_valid is False
        assert violation is not None
        assert violation.severity == BoundsViolationSeverity.ERROR
        assert "exceeds maximum" in violation.message
        assert "25.0" in violation.message

    def test_negative_pressure_critical(self, validator):
        """Test negative pressure is critical violation."""
        is_valid, violation = validator.validate_value("pressure_bar", -5.0)
        assert is_valid is False
        assert violation is not None
        assert violation.severity == BoundsViolationSeverity.CRITICAL

    def test_pressure_near_boundary_warning(self, validator):
        """Test pressure near boundary generates warning."""
        # Near minimum (within 5% of range)
        is_valid, violation = validator.validate_value("pressure_bar", 0.5)
        assert is_valid is True
        if violation:
            assert violation.severity == BoundsViolationSeverity.WARNING


# =============================================================================
# TEMPERATURE VALIDATION TESTS (273-373 K / 0-100 C)
# =============================================================================

class TestTemperatureValidation:
    """Test temperature bounds validation."""

    def test_valid_temperature_kelvin(self, validator):
        """Test valid temperature in Kelvin (350 K)."""
        is_valid, violation = validator.validate_value("temperature_k", 350.0)
        assert is_valid is True
        assert violation is None

    def test_temperature_at_minimum_kelvin(self, validator):
        """Test temperature at minimum (273 K = 0 C)."""
        is_valid, violation = validator.validate_value("temperature_k", 273.0)
        assert is_valid is True

    def test_temperature_below_minimum_kelvin(self, validator):
        """Test temperature below minimum (250 K < 273 K)."""
        is_valid, violation = validator.validate_value("temperature_k", 250.0)
        assert is_valid is False
        assert violation is not None
        assert "below minimum" in violation.message

    def test_temperature_above_maximum_kelvin(self, validator):
        """Test temperature above maximum extended range."""
        # Extended range is up to 523 K (250 C)
        is_valid, violation = validator.validate_value("temperature_k", 550.0)
        assert is_valid is False
        assert violation is not None

    def test_valid_temperature_celsius(self, validator):
        """Test valid temperature in Celsius."""
        is_valid, violation = validator.validate_value("temperature_c", 100.0)
        assert is_valid is True

    def test_temperature_celsius_exceeds_max(self, validator):
        """Test Celsius temperature exceeding max (300 > 250)."""
        is_valid, violation = validator.validate_value("temperature_c", 300.0)
        assert is_valid is False


# =============================================================================
# ACOUSTIC VALIDATION TESTS (0-120 dB)
# =============================================================================

class TestAcousticValidation:
    """Test acoustic level bounds validation (0-120 dB)."""

    def test_valid_acoustic_level(self, validator):
        """Test valid acoustic level (75 dB)."""
        is_valid, violation = validator.validate_value("acoustic_db", 75.0)
        assert is_valid is True
        assert violation is None

    def test_acoustic_at_minimum(self, validator):
        """Test acoustic at minimum (0 dB)."""
        is_valid, violation = validator.validate_value("acoustic_db", 0.0)
        assert is_valid is True

    def test_acoustic_at_maximum(self, validator):
        """Test acoustic at maximum (120 dB)."""
        is_valid, violation = validator.validate_value("acoustic_db", 120.0)
        assert is_valid is True

    def test_acoustic_exceeds_maximum(self, validator):
        """Test acoustic exceeding maximum (150 dB > 120 dB)."""
        is_valid, violation = validator.validate_value("acoustic_db", 150.0)
        assert is_valid is False
        assert violation is not None
        assert "exceeds maximum" in violation.message
        assert "120.0" in violation.message

    def test_negative_acoustic_critical(self, validator):
        """Test negative acoustic level is critical."""
        is_valid, violation = validator.validate_value("acoustic_db", -10.0)
        assert is_valid is False
        assert violation is not None
        assert violation.severity == BoundsViolationSeverity.CRITICAL


# =============================================================================
# SENSOR INPUT VALIDATION TESTS
# =============================================================================

class TestSensorInputValidation:
    """Test comprehensive sensor input validation."""

    def test_validate_all_valid_inputs(self, validator):
        """Test validation with all valid inputs."""
        result = validator.validate_sensor_input(
            pressure_bar=10.0,
            temperature_c=150.0,
            acoustic_db=75.0,
        )
        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_with_one_invalid(self, validator):
        """Test validation with one invalid input."""
        result = validator.validate_sensor_input(
            pressure_bar=30.0,  # Invalid: > 25 bar
            temperature_c=150.0,
            acoustic_db=75.0,
        )
        assert result.is_valid is False
        assert result.error_count == 1

    def test_validate_with_multiple_invalid(self, validator):
        """Test validation with multiple invalid inputs."""
        result = validator.validate_sensor_input(
            pressure_bar=30.0,   # Invalid
            temperature_c=300.0, # Invalid
            acoustic_db=150.0,   # Invalid
        )
        assert result.is_valid is False
        assert result.error_count == 3

    def test_validate_optional_parameters(self, validator):
        """Test validation skips None parameters."""
        result = validator.validate_sensor_input(
            pressure_bar=10.0,
            temperature_c=None,
            acoustic_db=None,
        )
        assert result.is_valid is True
        assert "pressure_bar" in result.validated_parameters
        assert "temperature_c" not in result.validated_parameters


# =============================================================================
# TRAP PARAMETERS VALIDATION TESTS
# =============================================================================

class TestTrapParametersValidation:
    """Test trap-specific parameter validation."""

    def test_validate_trap_parameters_valid(self, validator):
        """Test valid trap parameters."""
        result = validator.validate_trap_parameters(
            pressure_bar=10.0,
            inlet_temperature_c=185.0,
            outlet_temperature_c=180.0,
        )
        assert result.is_valid is True

    def test_validate_trap_parameters_with_delta_t(self, validator):
        """Test delta temperature is calculated and validated."""
        result = validator.validate_trap_parameters(
            pressure_bar=10.0,
            inlet_temperature_c=200.0,
            outlet_temperature_c=190.0,
        )
        assert "delta_temperature" in result.validated_parameters

    def test_validate_trap_with_orifice(self, validator):
        """Test validation includes orifice diameter."""
        result = validator.validate_trap_parameters(
            pressure_bar=10.0,
            inlet_temperature_c=185.0,
            outlet_temperature_c=180.0,
            orifice_diameter_mm=6.35,
        )
        assert "orifice_diameter_mm" in result.validated_parameters

    def test_validate_trap_invalid_orifice(self, validator):
        """Test validation catches invalid orifice."""
        result = validator.validate_trap_parameters(
            pressure_bar=10.0,
            inlet_temperature_c=185.0,
            outlet_temperature_c=180.0,
            orifice_diameter_mm=100.0,  # Invalid: > 50 mm
        )
        assert result.is_valid is False


# =============================================================================
# VALIDATION RESULT TESTS
# =============================================================================

class TestBoundsValidationResult:
    """Test BoundsValidationResult class."""

    def test_result_initialization(self):
        """Test result object initialization."""
        result = BoundsValidationResult()
        assert result.is_valid is True
        assert result.has_warnings is False
        assert result.error_count == 0

    def test_add_violation(self):
        """Test adding violation to result."""
        result = BoundsValidationResult()
        violation = BoundsViolation(
            parameter="pressure_bar",
            value=30.0,
            min_bound=0.0,
            max_bound=25.0,
            unit="bar",
            severity=BoundsViolationSeverity.ERROR,
            message="Test error",
            standard_reference="ASME PTC 39",
        )
        result.add_violation(violation)

        assert result.is_valid is False
        assert result.error_count == 1

    def test_add_warning(self):
        """Test adding warning to result."""
        result = BoundsValidationResult()
        violation = BoundsViolation(
            parameter="pressure_bar",
            value=0.5,
            min_bound=0.0,
            max_bound=25.0,
            unit="bar",
            severity=BoundsViolationSeverity.WARNING,
            message="Near boundary",
            standard_reference="ASME PTC 39",
        )
        result.add_violation(violation)

        assert result.is_valid is True  # Warnings don't affect validity
        assert result.has_warnings is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BoundsValidationResult()
        result.add_validated("pressure_bar")

        d = result.to_dict()

        assert "is_valid" in d
        assert "has_warnings" in d
        assert "validated_parameters" in d
        assert "violations" in d


# =============================================================================
# UNIT CONVERSION TESTS
# =============================================================================

class TestUnitConversions:
    """Test unit conversion methods."""

    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        assert BoundsValidator.celsius_to_kelvin(0.0) == 273.15
        assert BoundsValidator.celsius_to_kelvin(100.0) == 373.15
        assert BoundsValidator.celsius_to_kelvin(-40.0) == 233.15

    def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius conversion."""
        assert BoundsValidator.kelvin_to_celsius(273.15) == 0.0
        assert BoundsValidator.kelvin_to_celsius(373.15) == 100.0

    def test_bar_to_kpa(self):
        """Test bar to kPa conversion."""
        assert BoundsValidator.bar_to_kpa(1.0) == 100.0
        assert BoundsValidator.bar_to_kpa(10.0) == 1000.0

    def test_kpa_to_bar(self):
        """Test kPa to bar conversion."""
        assert BoundsValidator.kpa_to_bar(100.0) == 1.0
        assert BoundsValidator.kpa_to_bar(1000.0) == 10.0


# =============================================================================
# PHYSICAL BOUNDS DATA CLASS TESTS
# =============================================================================

class TestPhysicalBounds:
    """Test PhysicalBounds data class."""

    def test_create_bounds(self):
        """Test creating PhysicalBounds."""
        bounds = PhysicalBounds(
            min_value=0.0,
            max_value=25.0,
            unit="bar",
            warning_margin=0.05,
            standard_reference="ASME PTC 39",
        )

        assert bounds.min_value == 0.0
        assert bounds.max_value == 25.0
        assert bounds.unit == "bar"

    def test_bounds_defaults(self):
        """Test PhysicalBounds default values."""
        bounds = PhysicalBounds(
            min_value=0.0,
            max_value=100.0,
            unit="test",
        )

        assert bounds.warning_margin == 0.1
        assert bounds.standard_reference == ""


# =============================================================================
# BOUNDS VIOLATION DATA CLASS TESTS
# =============================================================================

class TestBoundsViolation:
    """Test BoundsViolation data class."""

    def test_create_violation(self):
        """Test creating BoundsViolation."""
        violation = BoundsViolation(
            parameter="pressure_bar",
            value=30.0,
            min_bound=0.0,
            max_bound=25.0,
            unit="bar",
            severity=BoundsViolationSeverity.ERROR,
            message="Pressure exceeds maximum",
            standard_reference="ASME PTC 39",
        )

        assert violation.parameter == "pressure_bar"
        assert violation.value == 30.0
        assert violation.severity == BoundsViolationSeverity.ERROR

    def test_violation_immutable(self):
        """Test that BoundsViolation is immutable."""
        violation = BoundsViolation(
            parameter="test",
            value=1.0,
            min_bound=0.0,
            max_bound=10.0,
            unit="test",
            severity=BoundsViolationSeverity.ERROR,
            message="Test",
            standard_reference="Test",
        )

        with pytest.raises(AttributeError):
            violation.value = 2.0


# =============================================================================
# SEVERITY ENUM TESTS
# =============================================================================

class TestBoundsViolationSeverity:
    """Test BoundsViolationSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert BoundsViolationSeverity.WARNING == "warning"
        assert BoundsViolationSeverity.ERROR == "error"
        assert BoundsViolationSeverity.CRITICAL == "critical"

    def test_severity_comparison(self):
        """Test severity can be used in comparisons."""
        assert BoundsViolationSeverity.ERROR.value == "error"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_undefined_parameter(self, validator):
        """Test validation of undefined parameter."""
        is_valid, violation = validator.validate_value("unknown_param", 100.0)
        # Should return valid with no violation for undefined params
        assert is_valid is True
        assert violation is None

    def test_custom_bounds(self, validator):
        """Test validation with custom bounds."""
        custom_bounds = PhysicalBounds(
            min_value=0.0,
            max_value=50.0,
            unit="custom",
            warning_margin=0.1,
        )

        is_valid, violation = validator.validate_value("custom", 25.0, custom_bounds)
        assert is_valid is True

        is_valid, violation = validator.validate_value("custom", 60.0, custom_bounds)
        assert is_valid is False

    def test_zero_warning_margin(self, validator):
        """Test bounds with zero warning margin."""
        bounds = PhysicalBounds(
            min_value=0.0,
            max_value=100.0,
            unit="test",
            warning_margin=0.0,
        )

        # Should not generate warnings near boundaries
        is_valid, violation = validator.validate_value("test", 0.0, bounds)
        assert is_valid is True

    def test_operating_hours_max(self, validator):
        """Test operating hours maximum (8760 hours/year)."""
        is_valid, violation = validator.validate_value("operating_hours_yr", 8760.0)
        assert is_valid is True

        is_valid, violation = validator.validate_value("operating_hours_yr", 9000.0)
        assert is_valid is False


# =============================================================================
# STANDARD REFERENCE TESTS
# =============================================================================

class TestStandardReferences:
    """Test that violations include standard references."""

    def test_pressure_reference(self, validator):
        """Test pressure violation includes ASME reference."""
        is_valid, violation = validator.validate_value("pressure_bar", 30.0)
        assert "ASME PTC 39" in violation.standard_reference

    def test_acoustic_reference(self, validator):
        """Test acoustic violation includes ISO reference."""
        is_valid, violation = validator.validate_value("acoustic_db", 150.0)
        assert "ISO 7841" in violation.standard_reference
