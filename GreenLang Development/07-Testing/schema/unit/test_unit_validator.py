# -*- coding: utf-8 -*-
"""
Unit tests for GL-FOUND-X-002 Unit Validator.

This module tests the UnitValidator class which validates unit values
against compiled schema IR unit specifications.

Tests cover:
    - Unit presence validation (GLSCHEMA-E300)
    - Unit compatibility/dimension validation (GLSCHEMA-E301)
    - Non-canonical unit detection (GLSCHEMA-E302)
    - Unknown unit detection (GLSCHEMA-E303)
    - Multiple input format parsing
    - Unit conversion and normalization
    - UnitCatalog functionality

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.3
"""

from typing import Any, Dict, List, Optional

import pytest

from greenlang.schema.compiler.ir import UnitSpecIR
from greenlang.schema.errors import ErrorCode
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
)
from greenlang.schema.models.finding import Severity
from greenlang.schema.units.catalog import (
    DimensionDefinition,
    UnitCatalog,
    UnitDefinition,
)
from greenlang.schema.validator.units import (
    NormalizedUnit,
    UnitValidator,
    create_unit_finding,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def catalog() -> UnitCatalog:
    """Create a unit catalog with standard SI units."""
    return UnitCatalog()


@pytest.fixture
def standard_options() -> ValidationOptions:
    """Create standard validation options."""
    return ValidationOptions(profile=ValidationProfile.STANDARD)


@pytest.fixture
def strict_options() -> ValidationOptions:
    """Create strict validation options."""
    return ValidationOptions(profile=ValidationProfile.STRICT)


@pytest.fixture
def validator(catalog: UnitCatalog, standard_options: ValidationOptions) -> UnitValidator:
    """Create a unit validator with standard options."""
    return UnitValidator(catalog, standard_options)


@pytest.fixture
def strict_validator(catalog: UnitCatalog, strict_options: ValidationOptions) -> UnitValidator:
    """Create a unit validator with strict options."""
    return UnitValidator(catalog, strict_options)


@pytest.fixture
def energy_spec() -> UnitSpecIR:
    """Create energy unit specification."""
    return UnitSpecIR(
        path="/energy",
        dimension="energy",
        canonical="kWh",
    )


@pytest.fixture
def mass_spec() -> UnitSpecIR:
    """Create mass unit specification."""
    return UnitSpecIR(
        path="/mass",
        dimension="mass",
        canonical="kg",
    )


@pytest.fixture
def emissions_spec() -> UnitSpecIR:
    """Create emissions unit specification."""
    return UnitSpecIR(
        path="/emissions",
        dimension="emissions",
        canonical="tCO2e",
    )


# =============================================================================
# TEST: UNIT CATALOG
# =============================================================================


class TestUnitCatalog:
    """Tests for UnitCatalog functionality."""

    def test_catalog_initialization(self, catalog: UnitCatalog):
        """Test catalog initializes with SI units."""
        # Should have energy units
        assert catalog.is_known_unit("kWh")
        assert catalog.is_known_unit("MWh")
        assert catalog.is_known_unit("J")

        # Should have mass units
        assert catalog.is_known_unit("kg")
        assert catalog.is_known_unit("t")
        assert catalog.is_known_unit("lb")

        # Should have dimensions
        dimensions = catalog.list_all_dimensions()
        assert "energy" in dimensions
        assert "mass" in dimensions
        assert "volume" in dimensions
        assert "temperature" in dimensions
        assert "emissions" in dimensions

    def test_get_unit(self, catalog: UnitCatalog):
        """Test getting unit definition."""
        unit = catalog.get_unit("kWh")
        assert unit is not None
        assert unit.symbol == "kWh"
        assert unit.dimension == "energy"
        assert unit.si_factor == 3_600_000.0

    def test_get_unit_with_alias(self, catalog: UnitCatalog):
        """Test getting unit by alias."""
        unit = catalog.get_unit("kilowatt-hour")
        assert unit is not None
        assert unit.symbol == "kWh"

    def test_get_unit_dimension(self, catalog: UnitCatalog):
        """Test getting unit dimension."""
        assert catalog.get_unit_dimension("kWh") == "energy"
        assert catalog.get_unit_dimension("kg") == "mass"
        assert catalog.get_unit_dimension("L") == "volume"
        assert catalog.get_unit_dimension("unknown") is None

    def test_is_compatible(self, catalog: UnitCatalog):
        """Test unit compatibility checking."""
        # Same dimension
        assert catalog.is_compatible("kWh", "MWh") is True
        assert catalog.is_compatible("kWh", "J") is True
        assert catalog.is_compatible("kg", "lb") is True

        # Different dimensions
        assert catalog.is_compatible("kWh", "kg") is False
        assert catalog.is_compatible("L", "m2") is False

    def test_conversion_basic(self, catalog: UnitCatalog):
        """Test basic unit conversion."""
        # Energy conversions
        assert catalog.convert(1000, "Wh", "kWh") == pytest.approx(1.0)
        assert catalog.convert(1, "kWh", "Wh") == pytest.approx(1000.0)
        assert catalog.convert(1, "MWh", "kWh") == pytest.approx(1000.0)

        # Mass conversions
        assert catalog.convert(1000, "g", "kg") == pytest.approx(1.0)
        assert catalog.convert(1, "t", "kg") == pytest.approx(1000.0)

    def test_conversion_temperature(self, catalog: UnitCatalog):
        """Test temperature conversion with offset."""
        # Celsius to Kelvin (0C = 273.15K)
        assert catalog.convert(0, "C", "K") == pytest.approx(273.15)
        assert catalog.convert(100, "C", "K") == pytest.approx(373.15)

        # Fahrenheit to Celsius (32F = 0C)
        assert catalog.convert(32, "F", "C") == pytest.approx(0.0, abs=0.01)
        assert catalog.convert(212, "F", "C") == pytest.approx(100.0, abs=0.01)

    def test_conversion_incompatible_units_raises(self, catalog: UnitCatalog):
        """Test that converting incompatible units raises ValueError."""
        with pytest.raises(ValueError, match="incompatible dimensions"):
            catalog.convert(100, "kWh", "kg")

    def test_conversion_unknown_unit_raises(self, catalog: UnitCatalog):
        """Test that converting unknown units raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source unit"):
            catalog.convert(100, "foobar", "kWh")

        with pytest.raises(ValueError, match="Unknown target unit"):
            catalog.convert(100, "kWh", "foobar")

    def test_register_custom_unit(self, catalog: UnitCatalog):
        """Test registering a custom unit."""
        custom_unit = UnitDefinition(
            symbol="myunit",
            name="My Custom Unit",
            dimension="energy",
            si_factor=500.0,
        )
        catalog.register_unit(custom_unit)

        assert catalog.is_known_unit("myunit")
        assert catalog.get_unit_dimension("myunit") == "energy"
        assert catalog.is_compatible("myunit", "kWh")

    def test_register_alias(self, catalog: UnitCatalog):
        """Test registering an alias."""
        catalog.register_alias("kwh", "kWh")  # lowercase alias

        unit = catalog.get_unit("kwh")
        assert unit is not None
        assert unit.symbol == "kWh"

    def test_get_canonical_unit(self, catalog: UnitCatalog):
        """Test getting canonical unit for dimension."""
        assert catalog.get_canonical_unit("energy") == "kWh"
        assert catalog.get_canonical_unit("mass") == "kg"
        assert catalog.get_canonical_unit("emissions") == "tCO2e"

    def test_list_units_for_dimension(self, catalog: UnitCatalog):
        """Test listing units for a dimension."""
        energy_units = catalog.list_units_for_dimension("energy")
        assert "kWh" in energy_units
        assert "MWh" in energy_units
        assert "J" in energy_units
        assert "BTU" in energy_units

        # Should not include units from other dimensions
        assert "kg" not in energy_units


# =============================================================================
# TEST: UNIT VALIDATOR - INPUT PARSING
# =============================================================================


class TestUnitValidatorParsing:
    """Tests for unit value parsing."""

    def test_parse_object_form(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing object form: {"value": 10, "unit": "kWh"}."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 100.0
        assert normalized.unit == "kWh"

    def test_parse_object_form_amount(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing object form with 'amount' field."""
        findings, normalized = validator.validate(
            {"amount": 100, "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 100.0

    def test_parse_object_form_string_value(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing object form with string numeric value."""
        findings, normalized = validator.validate(
            {"value": "100.5", "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(100.5)

    def test_parse_string_form(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing string form: "100 kWh"."""
        findings, normalized = validator.validate(
            "100 kWh",
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 100.0
        assert normalized.unit == "kWh"

    def test_parse_string_form_with_decimal(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing string form with decimal: "100.5 kWh"."""
        findings, normalized = validator.validate(
            "100.5 kWh",
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(100.5)

    def test_parse_string_form_with_negative(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing string form with negative value."""
        temp_spec = UnitSpecIR(path="/temp", dimension="temperature", canonical="C")
        findings, normalized = validator.validate(
            "-5 C",
            temp_spec,
            "/temp"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(-5.0)

    def test_parse_string_form_with_scientific(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test parsing string form with scientific notation."""
        findings, normalized = validator.validate(
            "1.5e3 kWh",
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(1500.0)


# =============================================================================
# TEST: UNIT VALIDATOR - ERROR DETECTION
# =============================================================================


class TestUnitValidatorErrors:
    """Tests for unit validation error detection."""

    def test_missing_unit_error(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test GLSCHEMA-E300: Missing unit error."""
        findings, normalized = validator.validate(
            100.0,  # Raw number without unit
            energy_spec,
            "/energy"
        )

        assert len(findings) == 1
        assert findings[0].code == ErrorCode.UNIT_MISSING.value
        assert findings[0].severity == Severity.ERROR
        assert "/energy" in findings[0].path

    def test_missing_unit_in_object(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test missing unit in object form."""
        findings, normalized = validator.validate(
            {"value": 100},  # Object without unit
            energy_spec,
            "/energy"
        )

        assert len(findings) == 1
        assert findings[0].code == ErrorCode.UNIT_MISSING.value

    def test_incompatible_unit_error(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test GLSCHEMA-E301: Incompatible unit error."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "kg"},  # Mass unit for energy field
            energy_spec,
            "/energy"
        )

        assert len(findings) == 1
        assert findings[0].code == ErrorCode.UNIT_INCOMPATIBLE.value
        assert findings[0].severity == Severity.ERROR
        assert "kg" in str(findings[0].message)

    def test_unknown_unit_error(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test GLSCHEMA-E303: Unknown unit error."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "foobar"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 1
        assert findings[0].code == ErrorCode.UNIT_UNKNOWN.value
        assert findings[0].severity == Severity.ERROR
        assert "foobar" in str(findings[0].message)

    def test_noncanonical_unit_warning_strict_mode(
        self,
        strict_validator: UnitValidator,
        energy_spec: UnitSpecIR
    ):
        """Test GLSCHEMA-E302: Non-canonical unit warning in strict mode."""
        findings, normalized = strict_validator.validate(
            {"value": 1000, "unit": "Wh"},  # Wh instead of canonical kWh
            energy_spec,
            "/energy"
        )

        # Should have a warning but still normalize
        assert any(f.code == ErrorCode.UNIT_NONCANONICAL.value for f in findings)
        warning = [f for f in findings if f.code == ErrorCode.UNIT_NONCANONICAL.value][0]
        assert warning.severity == Severity.WARNING

        # Should still produce normalized output
        assert normalized is not None
        assert normalized.unit == "kWh"

    def test_noncanonical_unit_no_warning_standard_mode(
        self,
        validator: UnitValidator,
        energy_spec: UnitSpecIR
    ):
        """Test that standard mode doesn't warn for non-canonical units."""
        findings, normalized = validator.validate(
            {"value": 1000, "unit": "Wh"},
            energy_spec,
            "/energy"
        )

        # Should not have the noncanonical warning
        assert not any(f.code == ErrorCode.UNIT_NONCANONICAL.value for f in findings)
        assert normalized is not None


# =============================================================================
# TEST: UNIT VALIDATOR - NORMALIZATION
# =============================================================================


class TestUnitValidatorNormalization:
    """Tests for unit normalization/conversion."""

    def test_normalize_to_canonical(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test normalizing to canonical unit."""
        findings, normalized = validator.validate(
            {"value": 1000, "unit": "Wh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(1.0)  # 1000 Wh = 1 kWh
        assert normalized.unit == "kWh"
        assert normalized.original_value == 1000.0
        assert normalized.original_unit == "Wh"
        assert normalized.was_converted is True

    def test_no_conversion_when_canonical(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test no conversion when already canonical."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 100.0
        assert normalized.unit == "kWh"
        assert normalized.was_converted is False

    def test_normalize_emissions(self, validator: UnitValidator, emissions_spec: UnitSpecIR):
        """Test normalizing emissions units."""
        findings, normalized = validator.validate(
            {"value": 1000, "unit": "kgCO2e"},
            emissions_spec,
            "/emissions"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(1.0)  # 1000 kg = 1 tonne
        assert normalized.unit == "tCO2e"

    def test_normalize_large_conversion(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test normalizing with large conversion factor."""
        findings, normalized = validator.validate(
            {"value": 1, "unit": "GWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(1_000_000.0)  # 1 GWh = 1,000,000 kWh
        assert normalized.unit == "kWh"

    def test_normalized_unit_to_dict(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test NormalizedUnit.to_dict() method."""
        _, normalized = validator.validate(
            {"value": 1000, "unit": "Wh"},
            energy_spec,
            "/energy"
        )

        result = normalized.to_dict()
        assert result == {"value": pytest.approx(1.0), "unit": "kWh"}

    def test_normalized_unit_to_dict_with_meta(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test NormalizedUnit.to_dict_with_meta() method."""
        _, normalized = validator.validate(
            {"value": 1000, "unit": "Wh"},
            energy_spec,
            "/energy"
        )

        result = normalized.to_dict_with_meta()
        assert result["value"] == pytest.approx(1.0)
        assert result["unit"] == "kWh"
        assert "_meta" in result
        assert result["_meta"]["original_value"] == 1000.0
        assert result["_meta"]["original_unit"] == "Wh"


# =============================================================================
# TEST: UNIT VALIDATOR - EDGE CASES
# =============================================================================


class TestUnitValidatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_whitespace_in_string_form(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test handling of extra whitespace in string form."""
        findings, normalized = validator.validate(
            "  100   kWh  ",  # Extra whitespace
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 100.0

    def test_zero_value(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test handling of zero value."""
        findings, normalized = validator.validate(
            {"value": 0, "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == 0.0

    def test_very_small_value(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test handling of very small value."""
        findings, normalized = validator.validate(
            {"value": 0.0001, "unit": "kWh"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.value == pytest.approx(0.0001)

    def test_empty_unit_string(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test handling of empty unit string."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": ""},
            energy_spec,
            "/energy"
        )

        # Empty unit string is reported as missing
        assert len(findings) >= 1
        assert any(f.code == ErrorCode.UNIT_MISSING.value for f in findings)

    def test_unit_alias_resolution(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test unit alias is resolved correctly."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "kilowatt-hour"},  # Alias
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.unit == "kWh"

    def test_case_insensitive_unit(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test case-insensitive unit lookup."""
        findings, normalized = validator.validate(
            {"value": 100, "unit": "kwh"},  # Lowercase
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
        assert normalized.unit == "kWh"


# =============================================================================
# TEST: CREATE_UNIT_FINDING HELPER
# =============================================================================


class TestCreateUnitFinding:
    """Tests for create_unit_finding helper function."""

    def test_create_missing_unit_finding(self):
        """Test creating UNIT_MISSING finding."""
        finding = create_unit_finding(
            code=ErrorCode.UNIT_MISSING,
            path="/energy",
            expected_dimension="energy",
            canonical_unit="kWh",
            allowed_units=["kWh", "MWh", "GWh"],
        )

        assert finding.code == ErrorCode.UNIT_MISSING.value
        assert finding.severity == Severity.ERROR
        assert finding.path == "/energy"
        assert finding.hint is not None
        assert "kWh" in finding.hint.suggested_values

    def test_create_incompatible_unit_finding(self):
        """Test creating UNIT_INCOMPATIBLE finding."""
        finding = create_unit_finding(
            code=ErrorCode.UNIT_INCOMPATIBLE,
            path="/energy",
            unit="kg",
            dimension="mass",
            expected_dimension="energy",
            allowed_units=["kWh", "MWh"],
        )

        assert finding.code == ErrorCode.UNIT_INCOMPATIBLE.value
        assert finding.severity == Severity.ERROR
        assert finding.actual is not None
        assert finding.actual["unit"] == "kg"
        assert finding.actual["dimension"] == "mass"

    def test_create_noncanonical_unit_finding(self):
        """Test creating UNIT_NONCANONICAL finding."""
        finding = create_unit_finding(
            code=ErrorCode.UNIT_NONCANONICAL,
            path="/energy",
            unit="Wh",
            canonical_unit="kWh",
        )

        assert finding.code == ErrorCode.UNIT_NONCANONICAL.value
        assert finding.severity == Severity.WARNING  # Warning, not error


# =============================================================================
# TEST: ALLOWED UNITS RESTRICTION
# =============================================================================


class TestAllowedUnitsRestriction:
    """Tests for allowed units restriction in unit spec."""

    def test_unit_in_allowed_list(self, validator: UnitValidator):
        """Test unit in allowed list passes."""
        spec = UnitSpecIR(
            path="/energy",
            dimension="energy",
            canonical="kWh",
            allowed=frozenset(["kWh", "MWh"]),
        )

        findings, normalized = validator.validate(
            {"value": 100, "unit": "kWh"},
            spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None

    def test_unit_not_in_allowed_list(self, validator: UnitValidator):
        """Test unit not in allowed list fails."""
        spec = UnitSpecIR(
            path="/energy",
            dimension="energy",
            canonical="kWh",
            allowed=frozenset(["kWh", "MWh"]),
        )

        findings, normalized = validator.validate(
            {"value": 100, "unit": "Wh"},  # Not in allowed list
            spec,
            "/energy"
        )

        assert len(findings) == 1
        assert findings[0].code == ErrorCode.UNIT_INCOMPATIBLE.value

    def test_empty_allowed_list_allows_all(self, validator: UnitValidator, energy_spec: UnitSpecIR):
        """Test that empty allowed list allows all units."""
        # Default energy_spec has no allowed restriction
        findings, normalized = validator.validate(
            {"value": 100, "unit": "J"},
            energy_spec,
            "/energy"
        )

        assert len(findings) == 0
        assert normalized is not None
