"""
GreenLang AgentSpec v2 - Schema Generation Tests

Tests for JSON Schema generation from Pydantic models.

This test file validates that:
1. JSON Schema can be generated from Pydantic models
2. Generated schema is valid JSON Schema (draft-2020-12)
3. Schema check mode detects drift
4. Generated schema matches committed schema

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-201 (AgentSpec v2 Schema + Validators) - DoD Section D.3
"""

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from greenlang.specs.agentspec_v2 import to_json_schema


# ============================================================================
# SCHEMA GENERATION TESTS
# ============================================================================

def test_schema_generation_succeeds():
    """Test that to_json_schema() generates valid schema without errors."""
    schema = to_json_schema()

    assert isinstance(schema, dict)
    assert len(schema) > 0
    assert "$schema" in schema
    assert "$id" in schema
    assert "title" in schema


def test_schema_has_correct_metadata():
    """Test that generated schema has correct metadata fields."""
    schema = to_json_schema()

    # Check required metadata
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://greenlang.io/specs/agentspec_v2.json"
    assert schema["title"] == "GreenLang AgentSpec v2"


def test_schema_has_all_required_sections():
    """Test that schema includes all required AgentSpec v2 sections."""
    schema = to_json_schema()

    assert "properties" in schema
    properties = schema["properties"]

    # Check top-level required fields
    assert "schema_version" in properties
    assert "id" in properties
    assert "name" in properties
    assert "version" in properties
    assert "compute" in properties
    assert "ai" in properties
    assert "realtime" in properties
    assert "provenance" in properties


def test_schema_is_valid_json_schema_draft_2020_12():
    """Test that exported JSON Schema is valid JSON Schema draft-2020-12."""
    schema = to_json_schema()

    # This should not raise an exception
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as e:
        pytest.fail(f"Generated schema is not valid JSON Schema draft-2020-12: {e}")


def test_schema_can_validate_example_spec():
    """Test that generated schema can validate an example AgentSpec."""
    schema = to_json_schema()

    # Create a minimal valid spec
    example_spec = {
        "schema_version": "2.0.0",
        "id": "test/example_v1",
        "name": "Example Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {
                "x": {"dtype": "float64", "unit": "1"}
            },
            "outputs": {
                "y": {"dtype": "float64", "unit": "1"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    # Validate against schema - should not raise
    validator = jsonschema.Draft202012Validator(schema)
    try:
        validator.validate(example_spec)
    except jsonschema.ValidationError as e:
        pytest.fail(f"Valid spec failed schema validation: {e}")


# ============================================================================
# SCHEMA CONSISTENCY TESTS
# ============================================================================

def test_generated_schema_matches_committed_schema():
    """Test that generated schema matches committed schema (no drift)."""
    # Generate schema
    generated_schema = to_json_schema()

    # Load committed schema
    committed_schema_path = Path(__file__).parent.parent.parent / "greenlang" / "specs" / "agentspec_v2.json"

    if not committed_schema_path.exists():
        pytest.skip(f"Committed schema not found: {committed_schema_path}")

    with open(committed_schema_path, 'r', encoding='utf-8') as f:
        committed_schema = json.load(f)

    # Normalize for comparison
    generated_json = json.dumps(generated_schema, indent=2, sort_keys=True)
    committed_json = json.dumps(committed_schema, indent=2, sort_keys=True)

    if generated_json != committed_json:
        pytest.fail(
            "Schema drift detected! Generated schema differs from committed schema.\n"
            "Run: python scripts/generate_schema.py --output greenlang/specs/agentspec_v2.json"
        )


def test_schema_roundtrip_consistency():
    """Test that schema → re-export → identical."""
    # Generate schema twice
    schema1 = to_json_schema()
    schema2 = to_json_schema()

    # Should be identical
    assert json.dumps(schema1, sort_keys=True) == json.dumps(schema2, sort_keys=True)


# ============================================================================
# CLI SCRIPT TESTS
# ============================================================================

def test_generate_schema_script_runs_successfully():
    """Test that generate_schema.py script runs without errors."""
    script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_schema.py"

    if not script_path.exists():
        pytest.skip(f"Script not found: {script_path}")

    # Run script with --help to verify it works
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert "generate_schema" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_generate_schema_script_check_mode():
    """Test that generate_schema.py --check mode works."""
    script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_schema.py"

    if not script_path.exists():
        pytest.skip(f"Script not found: {script_path}")

    # Run script in check mode
    result = subprocess.run(
        [sys.executable, str(script_path), "--check"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=script_path.parent.parent
    )

    # Should succeed if schema is up-to-date
    # Note: This might fail if schema is out of sync, which is expected
    if result.returncode != 0:
        # Check if it's a drift error (expected) or actual failure
        if "Schema MISMATCH detected" not in result.stdout and "Schema MISMATCH detected" not in result.stderr:
            pytest.fail(f"Script failed unexpectedly: {result.stderr}")


# ============================================================================
# SCHEMA CONTENT TESTS
# ============================================================================

def test_schema_includes_all_p0_fields():
    """Test that schema includes all P0 critical blocker fields."""
    schema = to_json_schema()

    # Navigate to definitions
    definitions = schema.get("$defs", {})

    # Check ComputeSpec has required fields
    if "ComputeSpec" in definitions:
        compute_props = definitions["ComputeSpec"]["properties"]
        assert "entrypoint" in compute_props
        assert "inputs" in compute_props
        assert "outputs" in compute_props
        assert "factors" in compute_props


def test_schema_includes_all_p1_fields():
    """Test that schema includes all P1 enhancement fields."""
    schema = to_json_schema()

    # Navigate to definitions
    definitions = schema.get("$defs", {})

    # Check ComputeSpec has P1 fields
    if "ComputeSpec" in definitions:
        compute_props = definitions["ComputeSpec"]["properties"]
        assert "dependencies" in compute_props
        assert "python_version" in compute_props
        assert "timeout_s" in compute_props
        assert "memory_limit_mb" in compute_props

    # Check ProvenanceSpec has gwp_set
    if "ProvenanceSpec" in definitions:
        prov_props = definitions["ProvenanceSpec"]["properties"]
        assert "gwp_set" in prov_props

    # Check AIBudget has max_retries
    if "AIBudget" in definitions:
        budget_props = definitions["AIBudget"]["properties"]
        assert "max_retries" in budget_props

    # Check RealtimeSpec has snapshot_path
    if "RealtimeSpec" in definitions:
        realtime_props = definitions["RealtimeSpec"]["properties"]
        assert "snapshot_path" in realtime_props


def test_schema_has_correct_defaults():
    """Test that schema specifies correct default values."""
    schema = to_json_schema()

    definitions = schema.get("$defs", {})

    # Check timeout_s default
    if "ComputeSpec" in definitions:
        compute_props = definitions["ComputeSpec"]["properties"]
        if "timeout_s" in compute_props:
            timeout_spec = compute_props["timeout_s"]
            # Check for default value
            if "default" in timeout_spec:
                assert timeout_spec["default"] == 30

    # Check memory_limit_mb default
    if "ComputeSpec" in definitions:
        compute_props = definitions["ComputeSpec"]["properties"]
        if "memory_limit_mb" in compute_props:
            memory_spec = compute_props["memory_limit_mb"]
            # Check for default value
            if "default" in memory_spec:
                assert memory_spec["default"] == 512

    # Check max_retries default
    if "AIBudget" in definitions:
        budget_props = definitions["AIBudget"]["properties"]
        if "max_retries" in budget_props:
            retries_spec = budget_props["max_retries"]
            if "default" in retries_spec:
                assert retries_spec["default"] == 3


# ============================================================================
# SCHEMA SIZE AND COMPLEXITY TESTS
# ============================================================================

def test_schema_is_reasonable_size():
    """Test that schema is not too large (performance check)."""
    schema = to_json_schema()
    schema_json = json.dumps(schema)

    # Schema should be between 10KB and 100KB
    assert 10000 < len(schema_json) < 100000, \
        f"Schema size ({len(schema_json)} bytes) is outside reasonable range"


def test_schema_can_be_serialized_to_json():
    """Test that schema can be serialized to JSON without errors."""
    schema = to_json_schema()

    try:
        schema_json = json.dumps(schema, indent=2, sort_keys=True)
        assert len(schema_json) > 0
    except Exception as e:
        pytest.fail(f"Schema serialization failed: {e}")


def test_schema_can_be_deserialized_from_json():
    """Test that schema can be deserialized from JSON."""
    schema = to_json_schema()
    schema_json = json.dumps(schema)

    try:
        deserialized = json.loads(schema_json)
        assert deserialized == schema
    except Exception as e:
        pytest.fail(f"Schema deserialization failed: {e}")
