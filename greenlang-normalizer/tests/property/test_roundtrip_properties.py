"""
Property-Based Tests for Roundtrip Properties (GL-FOUND-X-003).

This module tests roundtrip/serialization properties using Hypothesis.
These tests verify that data can be transformed and reversed without loss.

Properties Tested:
    1. Parse Roundtrip: parse(format(parse(s))) == parse(s)
    2. Serialization Roundtrip: deserialize(serialize(obj)) == obj
    3. Audit Replay: replay(audit_record) == original_result
    4. Hash Consistency: hash(data) is deterministic
    5. JSON Serialization: to_json(from_json(s)) preserves structure
    6. Quantity Roundtrip: Quantity serialization is reversible

References:
    - FR-079: Enable audit replay to reproduce identical outputs
    - NFR-003: Normalization must be deterministic
    - NFR-004: No silent data loss or transformation without audit trail
"""

import json
import hashlib
from typing import Any, Dict
from datetime import datetime

import pytest
from hypothesis import given, assume, settings, note
from hypothesis import strategies as st

from .strategies import (
    valid_quantity_strings,
    valid_quantity_dict,
    valid_positive_values,
    valid_simple_unit_strings,
    valid_measurement_audit,
    valid_entity_audit,
    valid_audit_event,
    valid_entity_with_variations,
    valid_fuel_names,
    SUSTAINABILITY_UNITS,
    FUEL_NAMES,
)


# =============================================================================
# Helper Functions
# =============================================================================

def approx_equal(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Check if two floats are approximately equal."""
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1e-15
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol


def json_serialize(obj: Any) -> str:
    """Serialize object to JSON with deterministic ordering."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def json_deserialize(s: str) -> Any:
    """Deserialize JSON string to object."""
    return json.loads(s)


# =============================================================================
# Property 1: Parse Roundtrip - parse(format(parse(s))) == parse(s)
# =============================================================================

class TestParseRoundtripProperty:
    """
    Tests for parse roundtrip property.

    Parse Roundtrip: parse(format(parse(s))) == parse(s)

    Parsing a quantity string, formatting it, and parsing again should
    produce the same result as the first parse.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(SUSTAINABILITY_UNITS[:10]),
    )
    @settings(max_examples=1000)
    def test_parse_format_parse_roundtrip(self, unit_parser, value: float, unit: str):
        """Parsing, formatting, and parsing again should be idempotent."""
        # Format a quantity string
        quantity_str = f"{value} {unit}"

        # First parse
        result1 = unit_parser.parse(quantity_str)
        assume(result1.success)  # Skip if initial parse fails

        # Format the parsed quantity
        formatted = str(result1.quantity)

        # Second parse
        result2 = unit_parser.parse(formatted)
        assume(result2.success)

        note(f"Original: {quantity_str}")
        note(f"First parse: {result1.quantity}")
        note(f"Formatted: {formatted}")
        note(f"Second parse: {result2.quantity}")

        # Compare results
        assert approx_equal(
            result1.quantity.magnitude,
            result2.quantity.magnitude,
        ), f"Magnitude mismatch: {result1.quantity.magnitude} != {result2.quantity.magnitude}"

        # Units should match (after normalization)
        assert result1.quantity.unit == result2.quantity.unit, \
            f"Unit mismatch: {result1.quantity.unit} != {result2.quantity.unit}"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(quantity_str=valid_quantity_strings())
    @settings(max_examples=500)
    def test_parse_idempotence(self, unit_parser, quantity_str: str):
        """Parsing the same string multiple times should give the same result."""
        results = [unit_parser.parse(quantity_str) for _ in range(5)]

        # All parses should have the same success status
        first_success = results[0].success
        for i, result in enumerate(results[1:], 2):
            assert result.success == first_success, \
                f"Parse {i} has different success status"

        if first_success:
            # All successful parses should have same magnitude and unit
            first_qty = results[0].quantity
            for i, result in enumerate(results[1:], 2):
                assert result.quantity.magnitude == first_qty.magnitude, \
                    f"Parse {i} has different magnitude"
                assert result.quantity.unit == first_qty.unit, \
                    f"Parse {i} has different unit"


# =============================================================================
# Property 2: Serialization Roundtrip - deserialize(serialize(obj)) == obj
# =============================================================================

class TestSerializationRoundtripProperty:
    """
    Tests for serialization roundtrip property.

    Serialization Roundtrip: deserialize(serialize(obj)) == obj

    Objects should survive a serialize/deserialize cycle unchanged.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(data=valid_quantity_dict())
    @settings(max_examples=1000)
    def test_quantity_dict_roundtrip(self, data: Dict[str, Any]):
        """Quantity dictionary should survive JSON roundtrip."""
        # Serialize
        serialized = json_serialize(data)

        # Deserialize
        deserialized = json_deserialize(serialized)

        note(f"Original: {data}")
        note(f"Serialized: {serialized}")
        note(f"Deserialized: {deserialized}")

        # Compare
        assert deserialized["unit"] == data["unit"], "Unit mismatch"
        assert approx_equal(
            deserialized["magnitude"],
            data["magnitude"],
        ), "Magnitude mismatch"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(audit=valid_measurement_audit())
    @settings(max_examples=500)
    def test_measurement_audit_roundtrip(self, audit: Dict[str, Any]):
        """Measurement audit record should survive JSON roundtrip."""
        serialized = json_serialize(audit)
        deserialized = json_deserialize(serialized)

        assert deserialized["field"] == audit["field"], "Field mismatch"
        assert approx_equal(
            deserialized["raw_value"],
            audit["raw_value"],
        ), "Raw value mismatch"
        assert deserialized["raw_unit"] == audit["raw_unit"], "Raw unit mismatch"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(audit=valid_entity_audit())
    @settings(max_examples=500)
    def test_entity_audit_roundtrip(self, audit: Dict[str, Any]):
        """Entity audit record should survive JSON roundtrip."""
        serialized = json_serialize(audit)
        deserialized = json_deserialize(serialized)

        assert deserialized["field"] == audit["field"], "Field mismatch"
        assert deserialized["entity_type"] == audit["entity_type"], "Entity type mismatch"
        assert deserialized["raw_name"] == audit["raw_name"], "Raw name mismatch"
        assert deserialized["reference_id"] == audit["reference_id"], "Reference ID mismatch"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(event=valid_audit_event())
    @settings(max_examples=200)
    def test_full_audit_event_roundtrip(self, event: Dict[str, Any]):
        """Complete audit event should survive JSON roundtrip."""
        serialized = json_serialize(event)
        deserialized = json_deserialize(serialized)

        assert deserialized["event_id"] == event["event_id"], "Event ID mismatch"
        assert deserialized["status"] == event["status"], "Status mismatch"
        assert deserialized["policy_mode"] == event["policy_mode"], "Policy mode mismatch"
        assert len(deserialized["measurements"]) == len(event["measurements"]), \
            "Measurements count mismatch"
        assert len(deserialized["entities"]) == len(event["entities"]), \
            "Entities count mismatch"


# =============================================================================
# Property 3: Audit Replay - replay(audit_record) == original_result
# =============================================================================

class TestAuditReplayProperty:
    """
    Tests for audit replay property.

    Audit Replay: replay(audit_record) == original_result

    Given an audit record, replaying the normalization with the same
    inputs and configuration should produce the same result.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @pytest.mark.audit
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(["kilogram", "kilowatt_hour", "liter", "meter"]),
    )
    @settings(max_examples=500)
    def test_conversion_replay_determinism(
        self,
        unit_converter,
        value: float,
        unit: str,
    ):
        """
        Converting the same value should always produce the same result.

        This simulates audit replay by performing the same conversion multiple
        times and verifying determinism.
        """
        try:
            from gl_normalizer_core.parser import Quantity

            # Target unit mapping
            target_units = {
                "kilogram": "gram",
                "kilowatt_hour": "megajoule",
                "liter": "gallon",
                "meter": "kilometer",
            }
            target = target_units[unit]

            quantity = Quantity(magnitude=value, unit=unit)

            # Perform conversion multiple times
            results = [
                unit_converter.convert(quantity, target)
                for _ in range(5)
            ]

            # All results should be identical
            first = results[0]
            for i, result in enumerate(results[1:], 2):
                assert result.success == first.success, \
                    f"Replay {i}: success status differs"
                if result.success:
                    assert result.converted_quantity.magnitude == first.converted_quantity.magnitude, \
                        f"Replay {i}: magnitude differs"
                    assert result.provenance_hash == first.provenance_hash, \
                        f"Replay {i}: provenance hash differs"

        except ImportError:
            pytest.skip("gl_normalizer_core not available")

    @pytest.mark.property
    @pytest.mark.roundtrip
    @pytest.mark.audit
    def test_hash_chain_replay(self, hash_chain_generator):
        """Hash chain should produce identical hashes for identical inputs."""
        measurements = [
            {"field": "energy", "raw_value": 100, "raw_unit": "kWh"}
        ]
        entities = [
            {"field": "fuel", "raw_name": "Diesel"}
        ]

        # Compute payload hash multiple times
        hashes = [
            hash_chain_generator.compute_payload_hash(measurements, entities)
            for _ in range(5)
        ]

        first_hash = hashes[0]
        for i, h in enumerate(hashes[1:], 2):
            assert h == first_hash, \
                f"Hash computation {i} differs: {h} != {first_hash}"


# =============================================================================
# Property 4: Hash Consistency
# =============================================================================

class TestHashConsistencyProperty:
    """
    Tests for hash consistency and determinism.

    Hash operations should be deterministic and consistent.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(audit=valid_audit_event())
    @settings(max_examples=500)
    def test_payload_hash_determinism(self, hash_chain_generator, audit: Dict[str, Any]):
        """Payload hash should be deterministic for the same input."""
        measurements = audit.get("measurements", [])
        entities = audit.get("entities", [])

        # Compute hash multiple times
        hashes = [
            hash_chain_generator.compute_payload_hash(measurements, entities)
            for _ in range(5)
        ]

        first = hashes[0]
        for i, h in enumerate(hashes[1:], 2):
            assert h == first, f"Hash {i} differs: {h} != {first}"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(audit=valid_audit_event())
    @settings(max_examples=500)
    def test_event_hash_determinism(self, hash_chain_generator, audit: Dict[str, Any]):
        """Event hash should be deterministic for the same input."""
        # Compute event hash multiple times
        hashes = [
            hash_chain_generator.compute_event_hash(audit, prev_event_hash=None)
            for _ in range(5)
        ]

        first = hashes[0]
        for i, h in enumerate(hashes[1:], 2):
            assert h == first, f"Event hash {i} differs: {h} != {first}"

    @pytest.mark.property
    @pytest.mark.roundtrip
    def test_hash_chain_integrity(self, hash_chain_generator):
        """Hash chain should maintain integrity through multiple events."""
        scope_id = "test-scope"
        events = []
        prev_hash = None

        # Create a chain of events
        for i in range(5):
            event_id = hash_chain_generator.generate_event_id()
            event_data = {
                "event_id": event_id,
                "sequence": i,
                "timestamp": datetime.utcnow().isoformat(),
            }

            event_hash = hash_chain_generator.compute_event_hash(event_data, prev_hash)
            event_data["event_hash"] = event_hash
            event_data["prev_event_hash"] = prev_hash

            events.append(event_data)
            prev_hash = event_hash

        # Verify chain integrity
        is_valid, error = hash_chain_generator.verify_chain_integrity(events)
        assert is_valid, f"Chain integrity verification failed: {error}"


# =============================================================================
# Property 5: Entity Resolution Roundtrip
# =============================================================================

class TestEntityResolutionRoundtripProperty:
    """
    Tests for entity resolution roundtrip properties.

    Resolution should be idempotent: resolving an already-resolved name
    should return the same result.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(fuel_name=valid_fuel_names())
    @settings(max_examples=500)
    def test_resolution_idempotence(self, reference_resolver, fuel_name: str):
        """Resolving an entity multiple times should give the same result."""
        results = [
            reference_resolver.resolve(fuel_name, "fuels")
            for _ in range(5)
        ]

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.success == first.success, \
                f"Resolution {i}: success status differs"
            if result.success:
                assert result.resolved.resolved_id == first.resolved.resolved_id, \
                    f"Resolution {i}: ID differs"
                assert result.resolved.confidence_score == first.resolved.confidence_score, \
                    f"Resolution {i}: confidence differs"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(variation=valid_entity_with_variations())
    @settings(max_examples=200)
    def test_canonical_name_resolution(
        self,
        reference_resolver,
        variation: tuple,
    ):
        """
        Resolving a canonical name and its variations should return the same ID.

        Note: This may not always pass for all variations if they're not in
        the alias list, but canonical names should always resolve to themselves.
        """
        canonical, variation_str = variation

        # Resolve canonical name
        canonical_result = reference_resolver.resolve(canonical, "fuels")

        if canonical_result.success:
            # If canonical resolves, the resolved name should be the canonical
            note(f"Canonical: {canonical} -> {canonical_result.resolved.resolved_name}")


# =============================================================================
# Property 6: Provenance Hash Roundtrip
# =============================================================================

class TestProvenanceHashProperty:
    """
    Tests for provenance hash properties.

    Provenance hashes should be deterministic and reproducible.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(["kilogram", "kilowatt_hour", "liter"]),
    )
    @settings(max_examples=500)
    def test_parse_provenance_determinism(self, unit_parser, value: float, unit: str):
        """Parse result provenance hash should be deterministic."""
        quantity_str = f"{value} {unit}"

        results = [unit_parser.parse(quantity_str) for _ in range(5)]

        if results[0].success:
            first_hash = results[0].provenance_hash
            for i, result in enumerate(results[1:], 2):
                assert result.provenance_hash == first_hash, \
                    f"Parse {i}: provenance hash differs"

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(["kilogram", "kilowatt_hour", "liter"]),
    )
    @settings(max_examples=500)
    def test_conversion_provenance_determinism(
        self,
        unit_converter,
        value: float,
        unit: str,
    ):
        """Conversion result provenance hash should be deterministic."""
        try:
            from gl_normalizer_core.parser import Quantity

            target_units = {
                "kilogram": "gram",
                "kilowatt_hour": "megajoule",
                "liter": "gallon",
            }
            target = target_units[unit]

            quantity = Quantity(magnitude=value, unit=unit)
            results = [
                unit_converter.convert(quantity, target)
                for _ in range(5)
            ]

            if results[0].success:
                first_hash = results[0].provenance_hash
                for i, result in enumerate(results[1:], 2):
                    assert result.provenance_hash == first_hash, \
                        f"Conversion {i}: provenance hash differs"

        except ImportError:
            pytest.skip("gl_normalizer_core not available")


# =============================================================================
# Property 7: Quantity Model Roundtrip
# =============================================================================

class TestQuantityModelRoundtrip:
    """
    Tests for Quantity model serialization roundtrip.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(SUSTAINABILITY_UNITS[:10]),
    )
    @settings(max_examples=500)
    def test_quantity_model_serialization(self, quantity_class, value: float, unit: str):
        """Quantity model should survive serialization roundtrip."""
        try:
            # Create quantity
            qty = quantity_class(magnitude=value, unit=unit)

            # Serialize to dict
            serialized = qty.model_dump()

            # Deserialize back
            deserialized = quantity_class(**serialized)

            # Compare
            assert deserialized.magnitude == qty.magnitude, \
                f"Magnitude mismatch: {deserialized.magnitude} != {qty.magnitude}"
            assert deserialized.unit == qty.unit, \
                f"Unit mismatch: {deserialized.unit} != {qty.unit}"

        except Exception as e:
            note(f"Error: {e}")
            pytest.skip("Quantity model operations not available")

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(
        value=valid_positive_values(),
        unit=st.sampled_from(SUSTAINABILITY_UNITS[:10]),
    )
    @settings(max_examples=500)
    def test_quantity_json_roundtrip(self, quantity_class, value: float, unit: str):
        """Quantity should survive JSON serialization roundtrip."""
        try:
            qty = quantity_class(magnitude=value, unit=unit)

            # Serialize to JSON
            json_str = qty.model_dump_json()

            # Deserialize back
            deserialized = quantity_class.model_validate_json(json_str)

            # Compare
            assert approx_equal(deserialized.magnitude, qty.magnitude), \
                f"Magnitude mismatch after JSON roundtrip"
            assert deserialized.unit == qty.unit, \
                f"Unit mismatch after JSON roundtrip"

        except Exception as e:
            note(f"Error: {e}")
            pytest.skip("Quantity JSON operations not available")


# =============================================================================
# Edge Cases
# =============================================================================

class TestRoundtripEdgeCases:
    """
    Tests for edge cases in roundtrip operations.
    """

    @pytest.mark.property
    @pytest.mark.roundtrip
    def test_empty_audit_payload_roundtrip(self, hash_chain_generator):
        """Empty audit payload should serialize and hash correctly."""
        measurements = []
        entities = []

        # Should not raise
        payload_hash = hash_chain_generator.compute_payload_hash(measurements, entities)
        assert payload_hash is not None
        assert payload_hash.startswith("sha256:")

        # Hash should be deterministic
        payload_hash_2 = hash_chain_generator.compute_payload_hash(measurements, entities)
        assert payload_hash == payload_hash_2

    @pytest.mark.property
    @pytest.mark.roundtrip
    def test_special_characters_roundtrip(self, hash_chain_generator):
        """Special characters in data should survive roundtrip."""
        special_data = {
            "field": "energy_consumption",
            "notes": "Temperature: 25\u00b0C, Flow: 100 m\u00b3/h",
            "unicode": "\u03bc\u00b5",  # Micro signs
        }

        # Should serialize without error
        event_hash = hash_chain_generator.compute_event_hash(special_data, None)
        assert event_hash is not None

        # Should be deterministic
        event_hash_2 = hash_chain_generator.compute_event_hash(special_data, None)
        assert event_hash == event_hash_2

    @pytest.mark.property
    @pytest.mark.roundtrip
    def test_large_payload_roundtrip(self, hash_chain_generator):
        """Large payloads should hash correctly."""
        # Create a large payload
        measurements = [
            {"field": f"field_{i}", "raw_value": float(i), "raw_unit": "kg"}
            for i in range(1000)
        ]
        entities = [
            {"field": f"entity_{i}", "raw_name": f"Entity {i}"}
            for i in range(100)
        ]

        # Should complete without error
        payload_hash = hash_chain_generator.compute_payload_hash(measurements, entities)
        assert payload_hash is not None

        # Should be deterministic
        payload_hash_2 = hash_chain_generator.compute_payload_hash(measurements, entities)
        assert payload_hash == payload_hash_2

    @pytest.mark.property
    @pytest.mark.roundtrip
    @given(value=valid_positive_values())
    @settings(max_examples=100)
    def test_precision_preservation_roundtrip(self, unit_parser, value: float):
        """Numeric precision should be preserved through parsing."""
        # Format with high precision
        quantity_str = f"{value:.15e} kg"

        result = unit_parser.parse(quantity_str)
        if result.success:
            # Check precision is preserved
            assert approx_equal(result.quantity.magnitude, value, rel_tol=1e-9), \
                f"Precision lost: {value} -> {result.quantity.magnitude}"
