# -*- coding: utf-8 -*-
"""
Unit tests for the Unit Canonicalizer (Task 3.2).

This module tests:
    - ConversionRecord model
    - CanonicalizedValue model
    - KeyRename model
    - UnitCanonicalizer class
    - KeyCanonicalizer class
    - Helper functions

Test Coverage:
    - Multiple input formats (object, string, raw numeric)
    - Unit conversion accuracy
    - Metadata preservation
    - Conversion record tracking
    - Error handling for incompatible units
    - Idempotency of canonicalization
    - Key alias resolution
    - Casing normalization

Author: GreenLang Framework Team
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.2
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest

from greenlang.schema.compiler.ir import SchemaIR, UnitSpecIR
from greenlang.schema.normalizer.canonicalizer import (
    CANONICAL_UNITS,
    CanonicalizedValue,
    ConversionRecord,
    KeyCanonicalizer,
    KeyRename,
    UnitCanonicalizer,
    get_canonical_unit,
    is_canonical_unit,
)
from greenlang.schema.units.catalog import UnitCatalog


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def unit_catalog() -> UnitCatalog:
    """Create a unit catalog with standard units."""
    return UnitCatalog()


@pytest.fixture
def canonicalizer(unit_catalog) -> UnitCanonicalizer:
    """Create a UnitCanonicalizer instance."""
    return UnitCanonicalizer(unit_catalog)


@pytest.fixture
def energy_unit_spec() -> UnitSpecIR:
    """Create a UnitSpecIR for energy."""
    return UnitSpecIR(
        path="/energy",
        dimension="energy",
        canonical="kWh",
        allowed=["Wh", "kWh", "MWh", "GWh", "J", "kJ", "MJ", "GJ"]
    )


@pytest.fixture
def mass_unit_spec() -> UnitSpecIR:
    """Create a UnitSpecIR for mass."""
    return UnitSpecIR(
        path="/mass",
        dimension="mass",
        canonical="kg",
        allowed=["g", "kg", "t", "lb"]
    )


@pytest.fixture
def emissions_unit_spec() -> UnitSpecIR:
    """Create a UnitSpecIR for emissions."""
    return UnitSpecIR(
        path="/emissions",
        dimension="emissions",
        canonical="tCO2e",
        allowed=["gCO2e", "kgCO2e", "tCO2e", "MTCO2e"]
    )


@pytest.fixture
def sample_schema_ir() -> SchemaIR:
    """Create a sample SchemaIR for testing KeyCanonicalizer."""
    return SchemaIR(
        schema_id="test/emissions",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        renamed_fields={
            "old_energy": "energy",
            "legacy_emissions": "emissions",
            "oldName": "new_name",
        }
    )


@pytest.fixture
def key_canonicalizer(sample_schema_ir) -> KeyCanonicalizer:
    """Create a KeyCanonicalizer instance."""
    return KeyCanonicalizer(sample_schema_ir)


# =============================================================================
# CONVERSION RECORD MODEL TESTS
# =============================================================================


class TestConversionRecord:
    """Tests for the ConversionRecord model."""

    def test_create_conversion_record(self):
        """Test creating a ConversionRecord."""
        record = ConversionRecord(
            path="/energy",
            original_value=1000.0,
            original_unit="Wh",
            canonical_value=1.0,
            canonical_unit="kWh",
            conversion_factor=0.001,
            dimension="energy"
        )

        assert record.path == "/energy"
        assert record.original_value == 1000.0
        assert record.original_unit == "Wh"
        assert record.canonical_value == 1.0
        assert record.canonical_unit == "kWh"
        assert record.conversion_factor == 0.001
        assert record.dimension == "energy"

    def test_conversion_record_to_dict(self):
        """Test converting ConversionRecord to dictionary."""
        record = ConversionRecord(
            path="/mass",
            original_value=500.0,
            original_unit="g",
            canonical_value=0.5,
            canonical_unit="kg",
            conversion_factor=0.001,
            dimension="mass"
        )

        d = record.to_dict()
        assert d["path"] == "/mass"
        assert d["original_value"] == 500.0
        assert d["original_unit"] == "g"
        assert d["canonical_value"] == 0.5
        assert d["canonical_unit"] == "kg"

    def test_conversion_record_compute_hash(self):
        """Test computing SHA-256 hash of ConversionRecord."""
        record = ConversionRecord(
            path="/energy",
            original_value=1000.0,
            original_unit="Wh",
            canonical_value=1.0,
            canonical_unit="kWh",
            conversion_factor=0.001,
            dimension="energy"
        )

        hash_value = record.compute_hash()
        assert len(hash_value) == 64  # SHA-256 produces 64 hex characters
        assert hash_value.isalnum()

    def test_conversion_record_immutable(self):
        """Test that ConversionRecord is immutable."""
        record = ConversionRecord(
            path="/energy",
            original_value=1000.0,
            original_unit="Wh",
            canonical_value=1.0,
            canonical_unit="kWh",
            conversion_factor=0.001,
            dimension="energy"
        )

        with pytest.raises(Exception):  # Pydantic frozen model
            record.path = "/new_path"


# =============================================================================
# CANONICALIZED VALUE MODEL TESTS
# =============================================================================


class TestCanonicalizedValue:
    """Tests for the CanonicalizedValue model."""

    def test_create_canonicalized_value(self):
        """Test creating a CanonicalizedValue."""
        value = CanonicalizedValue(
            value=1.0,
            unit="kWh",
            meta={
                "original_value": 1000.0,
                "original_unit": "Wh",
                "conversion_factor": 0.001
            }
        )

        assert value.value == 1.0
        assert value.unit == "kWh"
        assert value.meta["original_value"] == 1000.0

    def test_canonicalized_value_to_dict_with_meta(self):
        """Test converting CanonicalizedValue to dict with metadata."""
        value = CanonicalizedValue(
            value=1.0,
            unit="kWh",
            meta={
                "original_value": 1000.0,
                "original_unit": "Wh",
                "conversion_factor": 0.001
            }
        )

        d = value.to_dict()
        assert d["value"] == 1.0
        assert d["unit"] == "kWh"
        assert "_meta" in d
        assert d["_meta"]["original_value"] == 1000.0

    def test_canonicalized_value_to_dict_without_meta(self):
        """Test converting CanonicalizedValue to dict without metadata."""
        value = CanonicalizedValue(
            value=1.0,
            unit="kWh"
        )

        d = value.to_dict()
        assert d["value"] == 1.0
        assert d["unit"] == "kWh"
        assert "_meta" not in d

    def test_canonicalized_value_was_converted(self):
        """Test was_converted property."""
        converted = CanonicalizedValue(
            value=1.0,
            unit="kWh",
            meta={"original_unit": "Wh"}
        )
        not_converted = CanonicalizedValue(
            value=1.0,
            unit="kWh"
        )

        assert converted.was_converted is True
        assert not_converted.was_converted is False


# =============================================================================
# KEY RENAME MODEL TESTS
# =============================================================================


class TestKeyRename:
    """Tests for the KeyRename model."""

    def test_create_key_rename_alias(self):
        """Test creating a KeyRename for alias resolution."""
        rename = KeyRename(
            path="/",
            original_key="old_energy",
            canonical_key="energy",
            reason="alias"
        )

        assert rename.path == "/"
        assert rename.original_key == "old_energy"
        assert rename.canonical_key == "energy"
        assert rename.reason == "alias"

    def test_create_key_rename_casing(self):
        """Test creating a KeyRename for casing normalization."""
        rename = KeyRename(
            path="/nested",
            original_key="EnergyConsumption",
            canonical_key="energy_consumption",
            reason="casing"
        )

        assert rename.reason == "casing"

    def test_key_rename_invalid_reason(self):
        """Test that invalid reason raises error."""
        with pytest.raises(Exception):
            KeyRename(
                path="/",
                original_key="foo",
                canonical_key="bar",
                reason="invalid_reason"
            )


# =============================================================================
# UNIT CANONICALIZER TESTS - BASIC
# =============================================================================


class TestUnitCanonicalizerBasic:
    """Basic tests for UnitCanonicalizer."""

    def test_canonicalizer_init(self, unit_catalog):
        """Test UnitCanonicalizer initialization."""
        canonicalizer = UnitCanonicalizer(unit_catalog)
        assert canonicalizer.catalog == unit_catalog
        assert canonicalizer.get_records() == []

    def test_canonicalize_no_conversion_needed(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalization when value is already canonical."""
        value = {"value": 100.0, "unit": "kWh"}
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["value"] == 100.0
        assert result["unit"] == "kWh"
        assert "_meta" not in result
        assert len(records) == 0

    def test_canonicalize_object_form(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalization with object form input."""
        value = {"value": 1000.0, "unit": "Wh"}
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["value"] == 1.0
        assert result["unit"] == "kWh"
        assert "_meta" in result
        assert result["_meta"]["original_value"] == 1000.0
        assert result["_meta"]["original_unit"] == "Wh"
        assert len(records) == 1

    def test_canonicalize_string_form(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalization with string form input."""
        value = "1000 Wh"
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["value"] == 1.0
        assert result["unit"] == "kWh"
        assert len(records) == 1

    def test_canonicalize_raw_numeric(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalization with raw numeric input (uses canonical unit)."""
        value = 100.0
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        # Raw numeric uses canonical unit from spec
        assert result["value"] == 100.0
        assert result["unit"] == "kWh"
        assert "_meta" not in result
        assert len(records) == 0


# =============================================================================
# UNIT CANONICALIZER TESTS - CONVERSION ACCURACY
# =============================================================================


class TestUnitCanonicalizerConversions:
    """Tests for conversion accuracy."""

    def test_wh_to_kwh(self, canonicalizer, energy_unit_spec):
        """Test Wh to kWh conversion."""
        value = {"value": 1000.0, "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert abs(result["value"] - 1.0) < 0.0001

    def test_mwh_to_kwh(self, canonicalizer, energy_unit_spec):
        """Test MWh to kWh conversion."""
        value = {"value": 1.0, "unit": "MWh"}
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert abs(result["value"] - 1000.0) < 0.0001
        assert records[0].conversion_factor == 1000.0

    def test_gwh_to_kwh(self, canonicalizer, energy_unit_spec):
        """Test GWh to kWh conversion."""
        value = {"value": 1.0, "unit": "GWh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert abs(result["value"] - 1000000.0) < 0.0001

    def test_grams_to_kg(self, canonicalizer, mass_unit_spec):
        """Test g to kg conversion."""
        value = {"value": 500.0, "unit": "g"}
        result, records = canonicalizer.canonicalize(
            value, mass_unit_spec, "/mass"
        )
        assert abs(result["value"] - 0.5) < 0.0001
        assert records[0].conversion_factor == 0.001

    def test_tonnes_to_kg(self, canonicalizer, mass_unit_spec):
        """Test t to kg conversion."""
        value = {"value": 2.5, "unit": "t"}
        result, _ = canonicalizer.canonicalize(
            value, mass_unit_spec, "/mass"
        )
        assert abs(result["value"] - 2500.0) < 0.0001

    def test_kgco2e_to_tco2e(self, canonicalizer, emissions_unit_spec):
        """Test kgCO2e to tCO2e conversion."""
        value = {"value": 1000.0, "unit": "kgCO2e"}
        result, _ = canonicalizer.canonicalize(
            value, emissions_unit_spec, "/emissions"
        )
        assert abs(result["value"] - 1.0) < 0.0001


# =============================================================================
# UNIT CANONICALIZER TESTS - INPUT FORMATS
# =============================================================================


class TestUnitCanonicalizerInputFormats:
    """Tests for various input formats."""

    def test_object_with_amount_key(self, canonicalizer, energy_unit_spec):
        """Test object form with 'amount' instead of 'value'."""
        value = {"amount": 1000.0, "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert result["value"] == 1.0

    def test_object_with_quantity_key(self, canonicalizer, mass_unit_spec):
        """Test object form with 'quantity' instead of 'value'."""
        value = {"quantity": 500.0, "unit": "g"}
        result, _ = canonicalizer.canonicalize(
            value, mass_unit_spec, "/mass"
        )
        assert abs(result["value"] - 0.5) < 0.0001

    def test_object_with_units_key(self, canonicalizer, energy_unit_spec):
        """Test object form with 'units' instead of 'unit'."""
        value = {"value": 1000.0, "units": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert result["value"] == 1.0

    def test_string_with_scientific_notation(
        self, canonicalizer, energy_unit_spec
    ):
        """Test string form with scientific notation."""
        value = "1.5e3 Wh"
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert abs(result["value"] - 1.5) < 0.0001

    def test_string_with_negative_value(self, canonicalizer, energy_unit_spec):
        """Test string form with negative value."""
        value = "-500 Wh"
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert abs(result["value"] - (-0.5)) < 0.0001

    def test_string_value_numeric_coercion(
        self, canonicalizer, energy_unit_spec
    ):
        """Test object form with string numeric value."""
        value = {"value": "1000", "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert result["value"] == 1.0

    def test_integer_value(self, canonicalizer, energy_unit_spec):
        """Test with integer value."""
        value = {"value": 1000, "unit": "Wh"}  # int, not float
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert result["value"] == 1.0


# =============================================================================
# UNIT CANONICALIZER TESTS - ERROR HANDLING
# =============================================================================


class TestUnitCanonicalizerErrors:
    """Tests for error handling."""

    def test_incompatible_units_raises_error(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that incompatible units raise ValueError."""
        value = {"value": 100.0, "unit": "kg"}  # mass unit for energy spec
        with pytest.raises(ValueError) as exc_info:
            canonicalizer.canonicalize(value, energy_unit_spec, "/energy")
        assert "incompatible" in str(exc_info.value).lower()

    def test_unparseable_value_returns_original(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that unparseable value returns original."""
        value = {"not_a_value": "foo"}
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        assert result == value
        assert len(records) == 0

    def test_invalid_string_format_returns_original(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that invalid string format returns original."""
        value = "not a unit string"
        result, records = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )
        # Should return unchanged since it can't be parsed
        assert result == value
        assert len(records) == 0


# =============================================================================
# UNIT CANONICALIZER TESTS - METADATA
# =============================================================================


class TestUnitCanonicalizerMetadata:
    """Tests for metadata preservation."""

    def test_metadata_contains_original_value(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that metadata contains original value."""
        value = {"value": 1000.0, "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["_meta"]["original_value"] == 1000.0

    def test_metadata_contains_original_unit(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that metadata contains original unit."""
        value = {"value": 1000.0, "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["_meta"]["original_unit"] == "Wh"

    def test_metadata_contains_conversion_factor(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that metadata contains conversion factor."""
        value = {"value": 1000.0, "unit": "Wh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert result["_meta"]["conversion_factor"] == 0.001

    def test_no_metadata_when_no_conversion(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that metadata is absent when no conversion occurs."""
        value = {"value": 100.0, "unit": "kWh"}
        result, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        assert "_meta" not in result


# =============================================================================
# UNIT CANONICALIZER TESTS - RECORD MANAGEMENT
# =============================================================================


class TestUnitCanonicalizerRecords:
    """Tests for conversion record management."""

    def test_get_records_returns_all_records(
        self, canonicalizer, energy_unit_spec, mass_unit_spec
    ):
        """Test that get_records returns all conversion records."""
        canonicalizer.canonicalize(
            {"value": 1000, "unit": "Wh"}, energy_unit_spec, "/energy"
        )
        canonicalizer.canonicalize(
            {"value": 500, "unit": "g"}, mass_unit_spec, "/mass"
        )

        records = canonicalizer.get_records()
        assert len(records) == 2
        assert records[0].path == "/energy"
        assert records[1].path == "/mass"

    def test_clear_records(self, canonicalizer, energy_unit_spec):
        """Test that clear_records removes all records."""
        canonicalizer.canonicalize(
            {"value": 1000, "unit": "Wh"}, energy_unit_spec, "/energy"
        )
        assert len(canonicalizer.get_records()) == 1

        canonicalizer.clear_records()
        assert len(canonicalizer.get_records()) == 0

    def test_record_dimension(self, canonicalizer, energy_unit_spec):
        """Test that conversion record contains correct dimension."""
        canonicalizer.canonicalize(
            {"value": 1000, "unit": "Wh"}, energy_unit_spec, "/energy"
        )

        records = canonicalizer.get_records()
        assert records[0].dimension == "energy"


# =============================================================================
# UNIT CANONICALIZER TESTS - OBJECT CANONICALIZATION
# =============================================================================


class TestUnitCanonicalizerObject:
    """Tests for canonicalize_object method."""

    def test_canonicalize_object_simple(
        self, canonicalizer, energy_unit_spec, mass_unit_spec
    ):
        """Test canonicalizing a simple object."""
        payload = {
            "energy": {"value": 1000, "unit": "Wh"},
            "mass": {"value": 500, "unit": "g"},
            "name": "Test"
        }
        unit_specs = {
            "/energy": energy_unit_spec,
            "/mass": mass_unit_spec
        }

        result, records = canonicalizer.canonicalize_object(payload, unit_specs)

        assert result["energy"]["value"] == 1.0
        assert result["energy"]["unit"] == "kWh"
        assert abs(result["mass"]["value"] - 0.5) < 0.0001
        assert result["mass"]["unit"] == "kg"
        assert result["name"] == "Test"
        assert len(records) == 2

    def test_canonicalize_object_nested(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalizing nested objects."""
        payload = {
            "facility": {
                "energy": {"value": 1000, "unit": "Wh"}
            }
        }
        unit_specs = {
            "/facility/energy": energy_unit_spec
        }

        result, records = canonicalizer.canonicalize_object(payload, unit_specs)

        assert result["facility"]["energy"]["value"] == 1.0
        assert len(records) == 1

    def test_canonicalize_object_with_arrays(
        self, canonicalizer, energy_unit_spec
    ):
        """Test canonicalizing objects with arrays."""
        payload = {
            "items": [
                {"energy": {"value": 1000, "unit": "Wh"}},
                {"energy": {"value": 2000, "unit": "Wh"}}
            ]
        }
        unit_specs = {
            "/items/0/energy": energy_unit_spec,
            "/items/1/energy": UnitSpecIR(
                path="/items/1/energy",
                dimension="energy",
                canonical="kWh",
                allowed=[]
            )
        }

        result, records = canonicalizer.canonicalize_object(payload, unit_specs)

        assert result["items"][0]["energy"]["value"] == 1.0
        assert result["items"][1]["energy"]["value"] == 2.0
        assert len(records) == 2


# =============================================================================
# UNIT CANONICALIZER TESTS - IDEMPOTENCY
# =============================================================================


class TestUnitCanonicalizerIdempotency:
    """Tests for canonicalization idempotency."""

    def test_canonicalization_is_idempotent(
        self, canonicalizer, energy_unit_spec
    ):
        """Test that canonicalize(canonicalize(x)) == canonicalize(x)."""
        value = {"value": 1000, "unit": "Wh"}

        # First canonicalization
        result1, _ = canonicalizer.canonicalize(
            value, energy_unit_spec, "/energy"
        )

        # Second canonicalization of result
        result2, records2 = canonicalizer.canonicalize(
            result1, energy_unit_spec, "/energy"
        )

        # Should be identical (no further conversion)
        assert result1["value"] == result2["value"]
        assert result1["unit"] == result2["unit"]
        # Second canonicalization should not have _meta (no conversion)
        assert "_meta" not in result2
        assert len(records2) == 0


# =============================================================================
# KEY CANONICALIZER TESTS
# =============================================================================


class TestKeyCanonicalizer:
    """Tests for KeyCanonicalizer."""

    def test_key_canonicalizer_init(self, sample_schema_ir):
        """Test KeyCanonicalizer initialization."""
        canonicalizer = KeyCanonicalizer(sample_schema_ir)
        assert canonicalizer.ir == sample_schema_ir

    def test_resolve_alias(self, key_canonicalizer):
        """Test resolving key aliases."""
        payload = {"old_energy": 100, "other": "value"}
        result, renames = key_canonicalizer.canonicalize(payload)

        assert "energy" in result
        assert "old_energy" not in result
        assert result["energy"] == 100
        assert len(renames) == 1
        assert renames[0].original_key == "old_energy"
        assert renames[0].canonical_key == "energy"
        assert renames[0].reason == "alias"

    def test_casing_normalization(self, key_canonicalizer):
        """Test casing normalization (CamelCase to snake_case)."""
        payload = {"EnergyConsumption": 100}
        result, renames = key_canonicalizer.canonicalize(payload)

        assert "energy_consumption" in result
        assert "EnergyConsumption" not in result
        assert len(renames) == 1
        assert renames[0].reason == "casing"

    def test_nested_key_canonicalization(self, key_canonicalizer):
        """Test canonicalization of nested keys."""
        payload = {
            "Facility": {
                "EnergyUsage": 100
            }
        }
        result, renames = key_canonicalizer.canonicalize(payload)

        assert "facility" in result
        assert "energy_usage" in result["facility"]
        assert len(renames) == 2

    def test_stable_key_ordering(self, key_canonicalizer):
        """Test that output keys are alphabetically sorted."""
        payload = {"zebra": 1, "alpha": 2, "middle": 3}
        result, _ = key_canonicalizer.canonicalize(payload)

        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_no_change_for_canonical_keys(self, key_canonicalizer):
        """Test that canonical keys are unchanged."""
        payload = {"energy": 100, "mass": 50}
        result, renames = key_canonicalizer.canonicalize(payload)

        assert result == payload  # Keys already sorted
        assert len(renames) == 0

    def test_array_key_canonicalization(self, key_canonicalizer):
        """Test canonicalization of keys in arrays."""
        payload = {
            "Items": [
                {"Name": "Item1"},
                {"Name": "Item2"}
            ]
        }
        result, renames = key_canonicalizer.canonicalize(payload)

        assert "items" in result
        assert all("name" in item for item in result["items"])


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_canonical_unit_energy(self):
        """Test getting canonical unit for energy."""
        assert get_canonical_unit("energy") == "kWh"

    def test_get_canonical_unit_mass(self):
        """Test getting canonical unit for mass."""
        assert get_canonical_unit("mass") == "kg"

    def test_get_canonical_unit_emissions(self):
        """Test getting canonical unit for emissions."""
        assert get_canonical_unit("emissions") == "kgCO2e"

    def test_get_canonical_unit_unknown(self):
        """Test getting canonical unit for unknown dimension."""
        assert get_canonical_unit("unknown_dimension") is None

    def test_is_canonical_unit_true(self, unit_catalog):
        """Test is_canonical_unit returns True for canonical units."""
        assert is_canonical_unit("kWh", "energy", unit_catalog) is True
        assert is_canonical_unit("kg", "mass", unit_catalog) is True

    def test_is_canonical_unit_false(self, unit_catalog):
        """Test is_canonical_unit returns False for non-canonical units."""
        assert is_canonical_unit("Wh", "energy", unit_catalog) is False
        assert is_canonical_unit("g", "mass", unit_catalog) is False


# =============================================================================
# CANONICAL UNITS CONSTANT TESTS
# =============================================================================


class TestCanonicalUnitsConstant:
    """Tests for CANONICAL_UNITS constant."""

    def test_all_expected_dimensions_present(self):
        """Test that all expected dimensions have canonical units defined."""
        expected_dimensions = [
            "energy", "mass", "volume", "area", "length",
            "time", "temperature", "power", "emissions"
        ]
        for dim in expected_dimensions:
            assert dim in CANONICAL_UNITS, f"Missing dimension: {dim}"

    def test_canonical_units_are_strings(self):
        """Test that all canonical units are non-empty strings."""
        for dim, unit in CANONICAL_UNITS.items():
            assert isinstance(unit, str), f"{dim} unit is not a string"
            assert len(unit) > 0, f"{dim} unit is empty"
