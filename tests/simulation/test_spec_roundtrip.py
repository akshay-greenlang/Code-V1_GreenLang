"""
Tests for ScenarioSpec v1 - Round-trip YAML/JSON serialization

Tests spec validation, YAML round-trips, and error handling as specified in SIM-401.

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

import pytest
from pathlib import Path
import yaml
import json

from greenlang.simulation.spec import (
    ScenarioSpecV1,
    ParameterSpec,
    DistributionSpec,
    MonteCarloSpec,
    from_yaml,
    from_json,
    to_yaml,
    validate_spec,
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


def test_yaml_roundtrip(tmp_path):
    """Test YAML serialization round-trip (SIM-401 AC)."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test_roundtrip",
        description="Test round-trip serialization",
        seed=123456789,
        mode="replay",
        parameters=[
            ParameterSpec(id="x", type="sweep", values=[1.0, 2.0, 3.0]),
            ParameterSpec(id="y", type="sweep", values=["a", "b", "c"]),
        ],
        metadata={"owner": "test-squad", "tags": ["test"]}
    )

    # Write to YAML
    yaml_path = tmp_path / "test_scenario.yaml"
    to_yaml(spec, yaml_path)

    # Verify file exists
    assert yaml_path.exists()

    # Read back
    loaded_spec = from_yaml(yaml_path)

    # Verify round-trip equality
    assert loaded_spec.name == spec.name
    assert loaded_spec.seed == spec.seed
    assert loaded_spec.mode == spec.mode
    assert len(loaded_spec.parameters) == len(spec.parameters)
    assert loaded_spec.parameters[0].id == spec.parameters[0].id
    assert loaded_spec.parameters[1].id == spec.parameters[1].id
    assert loaded_spec.metadata == spec.metadata

    # Verify spec hashes match (stability test)
    assert loaded_spec.model_dump() == spec.model_dump()


def test_yaml_roundtrip_with_distributions(tmp_path):
    """Test round-trip with distribution parameters."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test_distributions",
        seed=42,
        parameters=[
            ParameterSpec(
                id="price",
                type="distribution",
                distribution=DistributionSpec(
                    kind="triangular",
                    low=0.08,
                    mode=0.12,
                    high=0.22
                )
            )
        ],
        monte_carlo=MonteCarloSpec(trials=1000)
    )

    yaml_path = tmp_path / "dist_scenario.yaml"
    to_yaml(spec, yaml_path)

    loaded_spec = from_yaml(yaml_path)

    assert loaded_spec.parameters[0].distribution.kind == "triangular"
    assert loaded_spec.parameters[0].distribution.low == 0.08
    assert loaded_spec.parameters[0].distribution.mode == 0.12
    assert loaded_spec.parameters[0].distribution.high == 0.22
    assert loaded_spec.monte_carlo.trials == 1000


def test_json_roundtrip(tmp_path):
    """Test JSON serialization round-trip."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="test_json_roundtrip",
        seed=999,
        parameters=[ParameterSpec(id="z", type="sweep", values=[10, 20, 30])]
    )

    # Manually save as JSON
    json_path = tmp_path / "test_scenario.json"
    with open(json_path, "w") as f:
        json.dump(spec.model_dump(exclude_none=True), f, indent=2)

    # Load back
    loaded_spec = from_json(json_path)

    assert loaded_spec.name == spec.name
    assert loaded_spec.seed == spec.seed
    assert loaded_spec.parameters[0].values == spec.parameters[0].values


def test_spec_stable_ordering(tmp_path):
    """Test that YAML output has stable ordering."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="stable_order_test",
        seed=123,
        parameters=[
            ParameterSpec(id="a", type="sweep", values=[1]),
            ParameterSpec(id="b", type="sweep", values=[2]),
            ParameterSpec(id="c", type="sweep", values=[3]),
        ]
    )

    yaml_path1 = tmp_path / "stable1.yaml"
    yaml_path2 = tmp_path / "stable2.yaml"

    to_yaml(spec, yaml_path1)
    to_yaml(spec, yaml_path2)

    # Read both files as text
    with open(yaml_path1) as f1, open(yaml_path2) as f2:
        content1 = f1.read()
        content2 = f2.read()

    # Should be byte-identical
    assert content1 == content2


def test_example_scenarios_valid():
    """Test that example scenario YAMLs are valid (SIM-401 AC)."""
    example_dir = Path("docs/scenarios/examples")

    # Test baseline_sweep.yaml
    baseline_path = example_dir / "baseline_sweep.yaml"
    if baseline_path.exists():
        spec = from_yaml(baseline_path)
        assert spec.name == "building_baseline_sweep"
        assert spec.seed == 42
        assert spec.mode == "replay"

    # Test monte_carlo.yaml
    mc_path = example_dir / "monte_carlo.yaml"
    if mc_path.exists():
        spec = from_yaml(mc_path)
        assert spec.name == "building_decarb_baseline"
        assert spec.seed == 123456789
        assert spec.monte_carlo is not None
        assert spec.monte_carlo.trials == 2000


def test_validate_spec_function():
    """Test validate_spec() function."""
    data = {
        "schema_version": "1.0.0",
        "name": "test_validate",
        "seed": 777,
        "parameters": [
            {"id": "x", "type": "sweep", "values": [1, 2, 3]}
        ]
    }

    spec = validate_spec(data)
    assert spec.name == "test_validate"
    assert spec.seed == 777


def test_invalid_yaml_fails(tmp_path):
    """Test that invalid YAML fails validation."""
    invalid_yaml = """
schema_version: "1.0.0"
name: "invalid_seed"
seed: -1  # Invalid: negative seed
parameters:
  - id: "x"
    type: "sweep"
    values: [1, 2, 3]
"""

    yaml_path = tmp_path / "invalid.yaml"
    with open(yaml_path, "w") as f:
        f.write(invalid_yaml)

    with pytest.raises(GLValidationError):
        from_yaml(yaml_path)


def test_spec_with_all_distribution_types(tmp_path):
    """Test round-trip with all distribution types."""
    spec = ScenarioSpecV1(
        schema_version="1.0.0",
        name="all_distributions",
        seed=42,
        parameters=[
            ParameterSpec(
                id="uniform_param",
                type="distribution",
                distribution=DistributionSpec(kind="uniform", low=0.0, high=1.0)
            ),
            ParameterSpec(
                id="normal_param",
                type="distribution",
                distribution=DistributionSpec(kind="normal", mean=100.0, std=15.0)
            ),
            ParameterSpec(
                id="lognormal_param",
                type="distribution",
                distribution=DistributionSpec(kind="lognormal", mean=0.0, sigma=1.0)
            ),
            ParameterSpec(
                id="triangular_param",
                type="distribution",
                distribution=DistributionSpec(kind="triangular", low=0.08, mode=0.12, high=0.22)
            ),
        ],
        monte_carlo=MonteCarloSpec(trials=500)
    )

    yaml_path = tmp_path / "all_dist.yaml"
    to_yaml(spec, yaml_path)
    loaded_spec = from_yaml(yaml_path)

    assert len(loaded_spec.parameters) == 4
    assert loaded_spec.parameters[0].distribution.kind == "uniform"
    assert loaded_spec.parameters[1].distribution.kind == "normal"
    assert loaded_spec.parameters[2].distribution.kind == "lognormal"
    assert loaded_spec.parameters[3].distribution.kind == "triangular"
