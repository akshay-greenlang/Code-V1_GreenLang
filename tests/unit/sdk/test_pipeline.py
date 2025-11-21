# -*- coding: utf-8 -*-
"""
Comprehensive tests for SDK Pipeline abstraction.

Tests cover:
- Pipeline initialization
- YAML loading and saving
- Pipeline validation
- Input handling
- Step configuration
"""

import pytest
import yaml
import json
from pathlib import Path
from greenlang.sdk.pipeline import Pipeline


@pytest.mark.unit
class TestPipelineInitialization:
    """Test Pipeline initialization."""

    def test_pipeline_minimal(self):
        """Test creating pipeline with minimal fields."""
        pipeline = Pipeline(name="test-pipeline")

        assert pipeline.name == "test-pipeline"
        assert pipeline.version == "1.0"
        assert pipeline.description is None
        assert pipeline.inputs == {}
        assert pipeline.steps == []
        assert pipeline.outputs == {}

    def test_pipeline_full_fields(self):
        """Test creating pipeline with all fields."""
        pipeline = Pipeline(
            name="full-pipeline",
            version="2.0",
            description="A full test pipeline",
            inputs={"param1": "value1"},
            steps=[{"name": "step1", "agent": "test-agent"}],
            outputs={"result": "value"},
        )

        assert pipeline.name == "full-pipeline"
        assert pipeline.version == "2.0"
        assert pipeline.description == "A full test pipeline"
        assert pipeline.inputs == {"param1": "value1"}
        assert len(pipeline.steps) == 1
        assert pipeline.outputs == {"result": "value"}

    def test_pipeline_with_multiple_steps(self):
        """Test pipeline with multiple steps."""
        steps = [
            {"name": "step1", "agent": "agent1"},
            {"name": "step2", "agent": "agent2"},
            {"name": "step3", "agent": "agent3"},
        ]
        pipeline = Pipeline(name="multi-step", steps=steps)

        assert len(pipeline.steps) == 3
        assert pipeline.steps[0]["name"] == "step1"
        assert pipeline.steps[2]["name"] == "step3"


@pytest.mark.unit
class TestPipelineYAMLLoading:
    """Test loading pipeline from YAML."""

    def test_from_yaml_basic(self, tmp_path):
        """Test loading basic pipeline from YAML."""
        yaml_content = """
name: test-pipeline
version: "1.0"
steps:
  - name: step1
    agent: test-agent
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert pipeline.name == "test-pipeline"
        assert pipeline.version == "1.0"
        assert len(pipeline.steps) == 1

    def test_from_yaml_with_inputs(self, tmp_path):
        """Test loading pipeline with inputs from YAML."""
        yaml_content = """
name: input-pipeline
version: "1.0"
inputs:
  param1: value1
  param2: 42
steps:
  - name: process
    agent: processor
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert pipeline.inputs["param1"] == "value1"
        assert pipeline.inputs["param2"] == 42

    def test_from_yaml_with_description(self, tmp_path):
        """Test loading pipeline with description."""
        yaml_content = """
name: described-pipeline
version: "1.0"
description: This is a test pipeline
steps:
  - name: step1
    agent: agent1
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert pipeline.description == "This is a test pipeline"

    def test_from_yaml_complex_steps(self, tmp_path):
        """Test loading pipeline with complex steps."""
        yaml_content = """
name: complex-pipeline
version: "1.0"
steps:
  - name: step1
    agent: agent1
    inputs:
      key: value
  - name: step2
    pipeline: sub-pipeline
    config:
      timeout: 300
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert len(pipeline.steps) == 2
        assert "inputs" in pipeline.steps[0]
        assert "config" in pipeline.steps[1]


@pytest.mark.unit
class TestPipelineYAMLSaving:
    """Test saving pipeline to YAML."""

    def test_to_yaml_basic(self, tmp_path):
        """Test saving basic pipeline to YAML."""
        pipeline = Pipeline(
            name="save-test",
            version="1.0",
            steps=[{"name": "step1", "agent": "agent1"}],
        )

        yaml_file = tmp_path / "output.yaml"
        pipeline.to_yaml(str(yaml_file))

        assert yaml_file.exists()

        # Verify content
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "save-test"
        assert data["version"] == "1.0"
        assert len(data["steps"]) == 1

    def test_to_yaml_with_inputs(self, tmp_path):
        """Test saving pipeline with inputs."""
        pipeline = Pipeline(
            name="input-pipeline",
            inputs={"key1": "value1", "key2": 42},
            steps=[{"name": "step1", "agent": "agent1"}],
        )

        yaml_file = tmp_path / "output.yaml"
        pipeline.to_yaml(str(yaml_file))

        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        assert data["inputs"]["key1"] == "value1"
        assert data["inputs"]["key2"] == 42

    def test_to_yaml_roundtrip(self, tmp_path):
        """Test roundtrip: save and load pipeline."""
        original = Pipeline(
            name="roundtrip",
            version="2.0",
            description="Test roundtrip",
            inputs={"param": "value"},
            steps=[
                {"name": "step1", "agent": "agent1"},
                {"name": "step2", "agent": "agent2"},
            ],
        )

        yaml_file = tmp_path / "pipeline.yaml"
        original.to_yaml(str(yaml_file))

        loaded = Pipeline.from_yaml(str(yaml_file))

        assert loaded.name == original.name
        assert loaded.version == original.version
        assert loaded.description == original.description
        assert loaded.inputs == original.inputs
        assert len(loaded.steps) == len(original.steps)


@pytest.mark.unit
class TestPipelineToDict:
    """Test converting pipeline to dictionary."""

    def test_to_dict_basic(self):
        """Test converting basic pipeline to dict."""
        pipeline = Pipeline(
            name="dict-test", steps=[{"name": "step1", "agent": "agent1"}]
        )

        data = pipeline.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "dict-test"
        assert "steps" in data

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        pipeline = Pipeline(
            name="full",
            version="1.5",
            description="Full pipeline",
            inputs={"in": 1},
            steps=[{"name": "s1", "agent": "a1"}],
            outputs={"out": 2},
        )

        data = pipeline.to_dict()

        assert data["name"] == "full"
        assert data["version"] == "1.5"
        assert data["description"] == "Full pipeline"
        assert data["inputs"] == {"in": 1}
        assert len(data["steps"]) == 1
        assert data["outputs"] == {"out": 2}


@pytest.mark.unit
class TestPipelineValidation:
    """Test pipeline validation."""

    def test_validate_valid_pipeline(self):
        """Test validating a valid pipeline."""
        pipeline = Pipeline(
            name="valid",
            steps=[
                {"name": "step1", "agent": "agent1"},
                {"name": "step2", "agent": "agent2"},
            ],
        )

        errors = pipeline.validate()

        assert errors == []

    def test_validate_missing_name(self):
        """Test validation fails when name is missing."""
        pipeline = Pipeline(name="", steps=[{"name": "step1", "agent": "agent1"}])

        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("name" in err.lower() for err in errors)

    def test_validate_no_steps(self):
        """Test validation fails when no steps."""
        pipeline = Pipeline(name="no-steps", steps=[])

        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("step" in err.lower() for err in errors)

    def test_validate_step_missing_name(self):
        """Test validation fails when step missing name."""
        pipeline = Pipeline(
            name="bad-step", steps=[{"agent": "agent1"}]  # Missing 'name'
        )

        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("name" in err.lower() for err in errors)

    def test_validate_step_missing_agent_and_pipeline(self):
        """Test validation fails when step has neither agent nor pipeline."""
        pipeline = Pipeline(
            name="bad-step", steps=[{"name": "step1"}]  # Missing agent/pipeline
        )

        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("agent" in err.lower() or "pipeline" in err.lower() for err in errors)

    def test_validate_multiple_steps_some_invalid(self):
        """Test validation catches multiple errors."""
        pipeline = Pipeline(
            name="multi-error",
            steps=[
                {"name": "step1", "agent": "agent1"},  # Valid
                {"agent": "agent2"},  # Missing name
                {"name": "step3"},  # Missing agent/pipeline
            ],
        )

        errors = pipeline.validate()

        assert len(errors) >= 2  # At least 2 errors

    def test_validate_step_with_pipeline_reference(self):
        """Test validation passes with pipeline reference."""
        pipeline = Pipeline(
            name="with-sub-pipeline",
            steps=[{"name": "step1", "pipeline": "sub-pipeline"}],
        )

        errors = pipeline.validate()

        assert errors == []


@pytest.mark.unit
class TestPipelineInputLoading:
    """Test loading inputs from files."""

    def test_load_inputs_from_json(self, tmp_path):
        """Test loading inputs from JSON file."""
        pipeline = Pipeline(name="test")

        json_file = tmp_path / "inputs.json"
        json_file.write_text(json.dumps({"param1": "value1", "param2": 42}))

        pipeline.load_inputs_file(str(json_file))

        assert pipeline.inputs["param1"] == "value1"
        assert pipeline.inputs["param2"] == 42

    def test_load_inputs_from_yaml(self, tmp_path):
        """Test loading inputs from YAML file."""
        pipeline = Pipeline(name="test")

        yaml_file = tmp_path / "inputs.yaml"
        yaml_file.write_text(
            yaml.dump({"param1": "value1", "param2": 42})
        )

        pipeline.load_inputs_file(str(yaml_file))

        assert pipeline.inputs["param1"] == "value1"
        assert pipeline.inputs["param2"] == 42

    def test_load_inputs_from_yml(self, tmp_path):
        """Test loading inputs from .yml file."""
        pipeline = Pipeline(name="test")

        yml_file = tmp_path / "inputs.yml"
        yml_file.write_text(yaml.dump({"key": "value"}))

        pipeline.load_inputs_file(str(yml_file))

        assert pipeline.inputs["key"] == "value"

    def test_load_inputs_updates_existing(self, tmp_path):
        """Test that loading inputs updates existing inputs."""
        pipeline = Pipeline(name="test", inputs={"existing": "value"})

        json_file = tmp_path / "inputs.json"
        json_file.write_text(json.dumps({"new": "data"}))

        pipeline.load_inputs_file(str(json_file))

        assert "existing" in pipeline.inputs
        assert "new" in pipeline.inputs

    def test_load_inputs_overwrites_duplicates(self, tmp_path):
        """Test that loading inputs overwrites duplicate keys."""
        pipeline = Pipeline(name="test", inputs={"key": "old"})

        json_file = tmp_path / "inputs.json"
        json_file.write_text(json.dumps({"key": "new"}))

        pipeline.load_inputs_file(str(json_file))

        assert pipeline.inputs["key"] == "new"


@pytest.mark.unit
class TestPipelineEdgeCases:
    """Test pipeline edge cases."""

    def test_empty_pipeline_name(self):
        """Test pipeline with empty name."""
        pipeline = Pipeline(name="")

        errors = pipeline.validate()
        assert len(errors) > 0

    def test_pipeline_with_outputs_no_steps(self):
        """Test pipeline with outputs but no steps."""
        pipeline = Pipeline(name="no-steps", outputs={"result": "value"})

        errors = pipeline.validate()
        assert len(errors) > 0  # Should fail due to no steps

    def test_pipeline_version_as_number(self):
        """Test pipeline version can be a number."""
        pipeline = Pipeline(name="test", version="2.5")

        assert pipeline.version == "2.5"

    def test_pipeline_complex_inputs(self):
        """Test pipeline with complex nested inputs."""
        complex_inputs = {
            "config": {"nested": {"deep": "value"}},
            "list": [1, 2, 3],
            "mixed": {"a": [1, 2], "b": {"c": 3}},
        }

        pipeline = Pipeline(name="complex", inputs=complex_inputs)

        assert pipeline.inputs == complex_inputs
