# -*- coding: utf-8 -*-
"""
Comprehensive SDK Integration Tests.

Tests cover:
- End-to-end agent execution
- Agent chaining in pipelines
- Context and artifact management
- Transform and validator composition
- Complete data processing workflows
"""

import pytest
import json
from pathlib import Path
from typing import Any, Dict
from greenlang.sdk.base import (
    Agent,
    Result,
    Metadata,
    Transform,
    Validator,
    Pipeline as BasePipeline,
)
from greenlang.sdk.context import Context
from greenlang.sdk.pipeline import Pipeline


# Test Agents
class DataLoaderAgent(Agent[Dict, Dict]):
    """Agent that loads and validates data."""

    def validate(self, input_data: Dict) -> bool:
        """Validate input has source field."""
        return isinstance(input_data, dict) and "source" in input_data

    def process(self, input_data: Dict) -> Dict:
        """Load data from source."""
        source = input_data["source"]
        return {"data": f"loaded from {source}", "records": 10, "status": "loaded"}


class TransformAgent(Agent[Dict, Dict]):
    """Agent that transforms data."""

    def validate(self, input_data: Dict) -> bool:
        """Validate input has data field."""
        return isinstance(input_data, dict) and "data" in input_data

    def process(self, input_data: Dict) -> Dict:
        """Transform the data."""
        data = input_data["data"]
        transformed = f"transformed: {data}"
        return {"data": transformed, "transform_applied": True}


class AggregatorAgent(Agent[Dict, Dict]):
    """Agent that aggregates results."""

    def validate(self, input_data: Dict) -> bool:
        """Validate input."""
        return isinstance(input_data, dict)

    def process(self, input_data: Dict) -> Dict:
        """Aggregate data."""
        return {
            "summary": "aggregated",
            "input_keys": list(input_data.keys()),
            "total": sum(v for v in input_data.values() if isinstance(v, (int, float))),
        }


# Test Transforms
class UppercaseTransform(Transform[str, str]):
    """Transform that uppercases strings."""

    def apply(self, data: str) -> str:
        """Uppercase the data."""
        return data.upper()


class MultiplyTransform(Transform[int, int]):
    """Transform that multiplies by a factor."""

    def __init__(self, factor: int):
        self.factor = factor

    def apply(self, data: int) -> int:
        """Multiply by factor."""
        return data * self.factor


# Test Validators
class RangeValidator(Validator[int]):
    """Validator for numeric ranges."""

    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, data: int) -> Result:
        """Validate range."""
        if not isinstance(data, (int, float)):
            return Result(success=False, error="Not a number")

        if self.min_val <= data <= self.max_val:
            return Result(success=True, data=data)

        return Result(
            success=False, error=f"Value {data} not in range [{self.min_val}, {self.max_val}]"
        )


@pytest.mark.integration
class TestAgentExecution:
    """Test complete agent execution workflows."""

    def test_single_agent_execution(self):
        """Test executing a single agent."""
        agent = DataLoaderAgent()
        input_data = {"source": "database"}

        result = agent.run(input_data)

        assert result.success is True
        assert "loaded from database" in result.data["data"]
        assert result.data["records"] == 10
        assert result.metadata["agent"] == "dataloaderagent"

    def test_agent_execution_with_validation_failure(self):
        """Test agent execution with invalid input."""
        agent = DataLoaderAgent()
        invalid_input = {"no_source": "data"}

        result = agent.run(invalid_input)

        assert result.success is False
        assert result.error == "Input validation failed"

    def test_agent_chaining(self):
        """Test chaining multiple agents."""
        loader = DataLoaderAgent()
        transformer = TransformAgent()

        # Execute first agent
        result1 = loader.run({"source": "file.csv"})
        assert result1.success is True

        # Execute second agent with first result
        result2 = transformer.run(result1.data)
        assert result2.success is True
        assert "transformed" in result2.data["data"]
        assert result2.data["transform_applied"] is True

    def test_three_agent_pipeline(self):
        """Test pipeline of three agents."""
        loader = DataLoaderAgent()
        transformer = TransformAgent()
        aggregator = AggregatorAgent()

        # Execute pipeline manually
        r1 = loader.run({"source": "api"})
        r2 = transformer.run(r1.data)
        r3 = aggregator.run(r2.data)

        assert r1.success and r2.success and r3.success
        assert "summary" in r3.data
        assert "aggregated" in r3.data["summary"]


@pytest.mark.integration
class TestContextIntegration:
    """Test context management in workflows."""

    def test_context_with_agent_execution(self, tmp_path):
        """Test using context to track agent execution."""
        ctx = Context(
            inputs={"source": "test.csv"}, artifacts_dir=tmp_path / "artifacts"
        )

        # Execute agent
        agent = DataLoaderAgent()
        result = agent.run(ctx.inputs)

        # Add result to context
        ctx.add_step_result("load", result)

        assert "load" in ctx.steps
        assert ctx.get_step_output("load") is not None

    def test_context_artifact_tracking(self, tmp_path):
        """Test tracking artifacts in context."""
        ctx = Context(artifacts_dir=tmp_path / "artifacts")

        # Execute agent and save result as artifact
        agent = DataLoaderAgent()
        result = agent.run({"source": "data.json"})

        # Save result as artifact
        artifact = ctx.save_artifact("load_result", result.data, "json")

        assert artifact.name == "load_result"
        assert artifact.path.exists()
        assert "load_result" in ctx.list_artifacts()

    def test_multi_step_context(self, tmp_path):
        """Test context with multiple steps."""
        ctx = Context(
            inputs={"source": "data.csv"}, artifacts_dir=tmp_path / "artifacts"
        )

        # Step 1: Load
        loader = DataLoaderAgent()
        load_result = loader.run(ctx.inputs)
        ctx.add_step_result("load", load_result)

        # Step 2: Transform
        transformer = TransformAgent()
        transform_result = transformer.run(load_result.data)
        ctx.add_step_result("transform", transform_result)

        # Step 3: Aggregate
        aggregator = AggregatorAgent()
        agg_result = aggregator.run(transform_result.data)
        ctx.add_step_result("aggregate", agg_result)

        # Verify all steps tracked
        assert len(ctx.steps) == 3
        all_outputs = ctx.get_all_step_outputs()
        assert "load" in all_outputs
        assert "transform" in all_outputs
        assert "aggregate" in all_outputs


@pytest.mark.integration
class TestTransformValidatorComposition:
    """Test composing transforms and validators."""

    def test_transform_chain(self):
        """Test chaining multiple transforms."""
        upper = UppercaseTransform()
        multiply = MultiplyTransform(2)

        # String transform
        text = "hello"
        result = upper(text)
        assert result == "HELLO"

        # Numeric transform
        number = 5
        result = multiply(number)
        assert result == 10

    def test_validator_pipeline(self):
        """Test multiple validators on same data."""
        validator1 = RangeValidator(0, 100)
        validator2 = RangeValidator(10, 50)

        value = 25

        result1 = validator1.validate(value)
        result2 = validator2.validate(value)

        assert result1.success is True
        assert result2.success is True

    def test_transform_then_validate(self):
        """Test applying transform then validating."""
        multiply = MultiplyTransform(5)
        validator = RangeValidator(0, 100)

        # Transform
        input_val = 10
        transformed = multiply(input_val)

        # Validate
        validation_result = validator.validate(transformed)

        assert transformed == 50
        assert validation_result.success is True


@pytest.mark.integration
class TestPipelineIntegration:
    """Test pipeline creation and execution."""

    def test_pipeline_yaml_creation(self, tmp_path):
        """Test creating and saving pipeline YAML."""
        pipeline = Pipeline(
            name="test-pipeline",
            version="1.0",
            description="Integration test pipeline",
            inputs={"source": "data.csv", "output": "results.json"},
            steps=[
                {"name": "load", "agent": "data-loader"},
                {"name": "transform", "agent": "transformer"},
                {"name": "aggregate", "agent": "aggregator"},
            ],
        )

        # Save pipeline
        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.to_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Load and verify
        loaded_pipeline = Pipeline.from_yaml(str(yaml_path))
        assert loaded_pipeline.name == "test-pipeline"
        assert len(loaded_pipeline.steps) == 3

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        valid_pipeline = Pipeline(
            name="valid",
            steps=[
                {"name": "step1", "agent": "agent1"},
                {"name": "step2", "agent": "agent2"},
            ],
        )

        errors = valid_pipeline.validate()
        assert len(errors) == 0

    def test_pipeline_with_inputs(self, tmp_path):
        """Test pipeline with input file loading."""
        pipeline = Pipeline(name="test")

        # Create inputs file
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps({"param1": "value1", "param2": 42}))

        # Load inputs
        pipeline.load_inputs_file(str(inputs_file))

        assert pipeline.inputs["param1"] == "value1"
        assert pipeline.inputs["param2"] == 42


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_data_pipeline(self, tmp_path):
        """Test complete data processing pipeline."""
        # Setup context
        ctx = Context(
            inputs={"source": "test.csv", "config": {"timeout": 30}},
            artifacts_dir=tmp_path / "artifacts",
            profile="test",
        )

        # Step 1: Load data
        loader = DataLoaderAgent()
        load_result = loader.run(ctx.inputs)
        assert load_result.success is True
        ctx.add_step_result("load", load_result)

        # Save load result as artifact
        ctx.save_artifact("raw_data", load_result.data, "json")

        # Step 2: Transform data
        transformer = TransformAgent()
        transform_result = transformer.run(load_result.data)
        assert transform_result.success is True
        ctx.add_step_result("transform", transform_result)

        # Save transformed data
        ctx.save_artifact("transformed_data", transform_result.data, "json")

        # Step 3: Aggregate
        aggregator = AggregatorAgent()
        agg_result = aggregator.run(transform_result.data)
        assert agg_result.success is True
        ctx.add_step_result("aggregate", agg_result)

        # Save final result
        ctx.save_artifact("final_result", agg_result.data, "json")

        # Verify complete workflow
        assert len(ctx.steps) == 3
        assert len(ctx.artifacts) == 3

        # Convert to result
        final_result = ctx.to_result()
        assert final_result.success is True
        assert "load" in final_result.data
        assert "transform" in final_result.data
        assert "aggregate" in final_result.data

    def test_workflow_with_failure_handling(self):
        """Test workflow with failure handling."""
        ctx = Context(inputs={"invalid": "data"})

        # Try to execute with invalid input
        loader = DataLoaderAgent()
        result = loader.run(ctx.inputs)

        # Should fail validation
        assert result.success is False

        # Add to context anyway
        ctx.add_step_result("load", result)

        # Convert to result - should be failure
        final_result = ctx.to_result()
        assert final_result.success is False


@pytest.mark.integration
class TestAgentMetadata:
    """Test agent metadata in integrated workflows."""

    def test_agent_with_full_metadata(self):
        """Test agent with complete metadata."""
        metadata = Metadata(
            id="test-loader",
            name="Test Data Loader",
            version="2.0.0",
            description="Loads test data",
            author="Test Team",
            tags=["loader", "test"],
        )

        agent = DataLoaderAgent(metadata=metadata)

        # Execute
        result = agent.run({"source": "test"})

        # Verify metadata in result
        assert result.metadata["agent"] == "test-loader"
        assert result.metadata["version"] == "2.0.0"

        # Verify description
        description = agent.describe()
        assert description["metadata"]["name"] == "Test Data Loader"
        assert "loader" in description["metadata"]["tags"]


@pytest.mark.integration
class TestErrorPropagation:
    """Test error handling and propagation."""

    def test_error_in_agent_chain(self):
        """Test error propagation through agent chain."""
        loader = DataLoaderAgent()
        transformer = TransformAgent()

        # First agent succeeds
        r1 = loader.run({"source": "test"})
        assert r1.success is True

        # Second agent fails (wrong input format)
        r2 = transformer.run({"wrong": "format"})
        assert r2.success is False

    def test_context_tracks_failures(self):
        """Test that context tracks failed steps."""
        ctx = Context()

        # Add successful step
        ctx.add_step_result("step1", Result(success=True, data={"a": 1}))

        # Add failed step
        ctx.add_step_result("step2", Result(success=False, error="Failed"))

        # Context result should be failure
        result = ctx.to_result()
        assert result.success is False
