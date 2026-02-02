# -*- coding: utf-8 -*-
"""
Tests for ScenarioSpec v1

Tests spec validation, YAML round-trips, and error handling.
"""

import pytest
from pathlib import Path

from greenlang.specs.scenariospec_v1 import (
    ScenarioSpecV1,
    ParameterSpec,
    DistributionSpec,
    MonteCarloSpec,
    from_yaml,
    validate_spec,
    to_yaml,
)
from greenlang.specs.errors import GLValidationError, GLVErr


def test_scenario_spec_basic():
    """Test basic scenario spec creation."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test_scenario",
        seed=42,
        parameters=[
            ParameterSpec(
                id="param1",
                type="sweep",
                values=[1, 2, 3]
            )
        ]
    )

    assert spec.name == "test_scenario"
    assert spec.seed == 42
    assert len(spec.parameters) == 1
    assert spec.mode == "replay"  # default


def test_distribution_validation():
    """Test distribution parameter validation."""
    # Valid triangular distribution
    dist = DistributionSpec(
        kind="triangular",
        low=0.0,
        mode=0.5,
        high=1.0
    )
    assert dist.kind == "triangular"

    # Invalid triangular (mode out of bounds)
    with pytest.raises(GLValidationError) as exc_info:
        DistributionSpec(
            kind="triangular",
            low=0.0,
            mode=1.5,  # Invalid: mode > high
            high=1.0
        )
    assert GLVErr.CONSTRAINT in str(exc_info.value.code)


def test_monte_carlo_required():
    """Test that Monte Carlo config is required for distribution parameters."""
    with pytest.raises(GLValidationError) as exc_info:
        ScenarioSpecV1(
            schema_version="1.0.0",
            name="test",
            seed=42,
            parameters=[
                ParameterSpec(
                    id="price",
                    type="distribution",
                    distribution=DistributionSpec(
                        kind="uniform",
                        low=0.08,
                        high=0.22
                    )
                )
            ]
            # Missing monte_carlo config!
        )
    assert "Monte Carlo configuration required" in str(exc_info.value)


def test_parameter_id_unique():
    """Test that parameter IDs must be unique."""
    with pytest.raises(GLValidationError) as exc_info:
        ScenarioSpecV1(
            schema_version="1.0.0",
            name="test",
            seed=42,
            parameters=[
                ParameterSpec(id="param1", type="sweep", values=[1, 2]),
                ParameterSpec(id="param1", type="sweep", values=[3, 4]),  # Duplicate!
            ]
        )
    assert GLVErr.DUPLICATE_NAME in str(exc_info.value.code)


def test_yaml_roundtrip(tmp_path):
    """Test YAML serialization round-trip."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test_roundtrip",
        seed=123,
        parameters=[
            ParameterSpec(id="x", type="sweep", values=[1.0, 2.0, 3.0])
        ]
    )

    # Write to YAML
    yaml_path = tmp_path / "test_scenario.yaml"
    to_yaml(spec, yaml_path)

    # Read back
    loaded_spec = from_yaml(yaml_path)

    # Verify round-trip
    assert loaded_spec.name == spec.name
    assert loaded_spec.seed == spec.seed
    assert len(loaded_spec.parameters) == len(spec.parameters)
    assert loaded_spec.parameters[0].id == spec.parameters[0].id


def test_seed_range_validation():
    """Test seed must be in valid range."""
    # Valid seed
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test",
        seed=42,
        parameters=[ParameterSpec(id="x", type="sweep", values=[1])]
    )
    assert spec.seed == 42

    # Invalid: negative seed
    with pytest.raises(GLValidationError):
        ScenarioSpecV1(
            schema_version="1.0.0",
            name="test",
            seed=-1,  # Invalid!
            parameters=[ParameterSpec(id="x", type="sweep", values=[1])]
        )


def test_example_scenarios_valid():
    """Test that example scenario YAMLs are valid."""
    examples_dir = Path(__file__).parents[2] / "docs" / "scenarios" / "examples"

    for yaml_file in examples_dir.glob("*.yaml"):
        # Should load without errors
        spec = from_yaml(yaml_file)
        assert spec.schema_version == "1.0.0"
        print(f"✓ {yaml_file.name} is valid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
