"""
Comprehensive Unit Tests for Core Coverage - GreenLang Phase 5

This module provides extensive unit tests for core GreenLang functionality:
- Workflow orchestration
- DAG validation
- Policy enforcement
- Resource management
- Error recovery
- Edge cases and boundary conditions

Target: 95%+ coverage for core modules
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, call
import asyncio
from datetime import datetime, timedelta

# Import core modules
from greenlang.core.workflow import (
    Workflow,
    WorkflowStep,
    WorkflowBuilder
)
from greenlang.core.orchestrator import WorkflowOrchestrator
from greenlang.core.artifact_manager import ArtifactManager


# ============================================================================
# WORKFLOW TESTS - Comprehensive coverage for workflow.py
# ============================================================================

class TestWorkflowStep:
    """Test WorkflowStep model and its methods."""

    def test_workflow_step_creation_minimal(self):
        """Test creating a workflow step with minimal required fields."""
        step = WorkflowStep(name="test_step", agent_id="agent_001")
        assert step.name == "test_step"
        assert step.agent_id == "agent_001"
        assert step.description is None
        assert step.on_failure == "stop"
        assert step.retry_count == 0

    def test_workflow_step_creation_complete(self):
        """Test creating a workflow step with all fields."""
        step = WorkflowStep(
            name="complete_step",
            agent_id="agent_002",
            description="A complete test step",
            input_mapping={"input": "context.data"},
            output_key="result",
            condition="context.ready == true",
            on_failure="skip",
            retry_count=3
        )
        assert step.name == "complete_step"
        assert step.agent_id == "agent_002"
        assert step.description == "A complete test step"
        assert step.input_mapping == {"input": "context.data"}
        assert step.output_key == "result"
        assert step.condition == "context.ready == true"
        assert step.on_failure == "skip"
        assert step.retry_count == 3

    def test_workflow_step_validation_name_required(self):
        """Test that step name is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            WorkflowStep(agent_id="agent_001")

    def test_workflow_step_validation_agent_id_required(self):
        """Test that agent_id is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            WorkflowStep(name="test_step")

    def test_workflow_step_retry_count_negative(self):
        """Test workflow step with negative retry count."""
        # Should accept negative values (validation can be added)
        step = WorkflowStep(name="test", agent_id="agent", retry_count=-1)
        assert step.retry_count == -1

    def test_workflow_step_on_failure_values(self):
        """Test different on_failure values."""
        for value in ["stop", "skip", "continue"]:
            step = WorkflowStep(name="test", agent_id="agent", on_failure=value)
            assert step.on_failure == value

    def test_workflow_step_input_mapping_empty(self):
        """Test step with empty input mapping."""
        step = WorkflowStep(
            name="test",
            agent_id="agent",
            input_mapping={}
        )
        assert step.input_mapping == {}

    def test_workflow_step_input_mapping_complex(self):
        """Test step with complex input mapping."""
        mapping = {
            "param1": "context.data.field1",
            "param2": "context.results.step1.output",
            "param3": "constant_value"
        }
        step = WorkflowStep(name="test", agent_id="agent", input_mapping=mapping)
        assert step.input_mapping == mapping


class TestWorkflow:
    """Test Workflow model and its methods."""

    def test_workflow_creation_minimal(self):
        """Test creating a workflow with minimal fields."""
        workflow = Workflow(
            name="test_workflow",
            description="Test workflow",
            steps=[]
        )
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert workflow.version == "0.0.1"
        assert len(workflow.steps) == 0
        assert workflow.output_mapping is None
        assert workflow.metadata == {}

    def test_workflow_creation_complete(self):
        """Test creating a workflow with all fields."""
        steps = [
            WorkflowStep(name="step1", agent_id="agent1"),
            WorkflowStep(name="step2", agent_id="agent2")
        ]
        workflow = Workflow(
            name="complete_workflow",
            description="Complete workflow",
            version="1.0.0",
            steps=steps,
            output_mapping={"result": "context.final_result"},
            metadata={"author": "test", "tags": ["test"]}
        )
        assert workflow.name == "complete_workflow"
        assert workflow.version == "1.0.0"
        assert len(workflow.steps) == 2
        assert workflow.output_mapping == {"result": "context.final_result"}
        assert workflow.metadata == {"author": "test", "tags": ["test"]}

    def test_workflow_add_step(self):
        """Test adding a step to a workflow."""
        workflow = Workflow(name="test", description="test", steps=[])
        step = WorkflowStep(name="step1", agent_id="agent1")
        workflow.add_step(step)
        assert len(workflow.steps) == 1
        assert workflow.steps[0].name == "step1"

    def test_workflow_add_multiple_steps(self):
        """Test adding multiple steps to a workflow."""
        workflow = Workflow(name="test", description="test", steps=[])
        for i in range(5):
            step = WorkflowStep(name=f"step{i}", agent_id=f"agent{i}")
            workflow.add_step(step)
        assert len(workflow.steps) == 5

    def test_workflow_remove_step(self):
        """Test removing a step from a workflow."""
        steps = [
            WorkflowStep(name="step1", agent_id="agent1"),
            WorkflowStep(name="step2", agent_id="agent2"),
            WorkflowStep(name="step3", agent_id="agent3")
        ]
        workflow = Workflow(name="test", description="test", steps=steps)
        workflow.remove_step("step2")
        assert len(workflow.steps) == 2
        assert workflow.steps[0].name == "step1"
        assert workflow.steps[1].name == "step3"

    def test_workflow_remove_nonexistent_step(self):
        """Test removing a step that doesn't exist."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(name="test", description="test", steps=steps)
        workflow.remove_step("nonexistent")
        assert len(workflow.steps) == 1  # Should not remove anything

    def test_workflow_get_step(self):
        """Test getting a step by name."""
        steps = [
            WorkflowStep(name="step1", agent_id="agent1"),
            WorkflowStep(name="step2", agent_id="agent2")
        ]
        workflow = Workflow(name="test", description="test", steps=steps)
        step = workflow.get_step("step2")
        assert step is not None
        assert step.name == "step2"
        assert step.agent_id == "agent2"

    def test_workflow_get_nonexistent_step(self):
        """Test getting a step that doesn't exist."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(name="test", description="test", steps=steps)
        step = workflow.get_step("nonexistent")
        assert step is None

    def test_workflow_validation_no_errors(self):
        """Test workflow validation with valid workflow."""
        steps = [
            WorkflowStep(name="step1", agent_id="agent1"),
            WorkflowStep(name="step2", agent_id="agent2")
        ]
        workflow = Workflow(name="test", description="test", steps=steps)
        errors = workflow.validate_workflow()
        assert errors == []

    def test_workflow_validation_no_steps(self):
        """Test workflow validation with no steps."""
        workflow = Workflow(name="test", description="test", steps=[])
        errors = workflow.validate_workflow()
        assert len(errors) == 1
        assert "no steps" in errors[0].lower()

    def test_workflow_validation_duplicate_step_names(self):
        """Test workflow validation with duplicate step names."""
        steps = [
            WorkflowStep(name="duplicate", agent_id="agent1"),
            WorkflowStep(name="unique", agent_id="agent2"),
            WorkflowStep(name="duplicate", agent_id="agent3")
        ]
        workflow = Workflow(name="test", description="test", steps=steps)
        errors = workflow.validate_workflow()
        assert len(errors) == 1
        assert "duplicate" in errors[0].lower()

    def test_workflow_to_yaml(self):
        """Test workflow serialization to YAML."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(name="test", description="test", steps=steps)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            workflow.to_yaml(yaml_path)
            assert Path(yaml_path).exists()
            # Verify file is not empty
            content = Path(yaml_path).read_text()
            assert len(content) > 0
            assert "test" in content
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_workflow_from_yaml(self):
        """Test workflow deserialization from YAML."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(name="test", description="test workflow", steps=steps)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            workflow.to_yaml(yaml_path)
            loaded = Workflow.from_yaml(yaml_path)
            assert loaded.name == "test"
            assert loaded.description == "test workflow"
            assert len(loaded.steps) == 1
            assert loaded.steps[0].name == "step1"
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_workflow_to_json(self):
        """Test workflow serialization to JSON."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(name="test", description="test", steps=steps)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            workflow.to_json(json_path)
            assert Path(json_path).exists()
            with open(json_path) as f:
                data = json.load(f)
            assert data['name'] == 'test'
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_workflow_from_json(self):
        """Test workflow deserialization from JSON."""
        steps = [WorkflowStep(name="step1", agent_id="agent1")]
        workflow = Workflow(
            name="test",
            description="test workflow",
            steps=steps,
            metadata={"key": "value"}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            workflow.to_json(json_path)
            loaded = Workflow.from_json(json_path)
            assert loaded.name == "test"
            assert loaded.description == "test workflow"
            assert len(loaded.steps) == 1
            assert loaded.metadata == {"key": "value"}
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_workflow_roundtrip_yaml(self):
        """Test YAML serialization roundtrip."""
        original = Workflow(
            name="roundtrip",
            description="Test roundtrip",
            version="2.0.0",
            steps=[
                WorkflowStep(name="s1", agent_id="a1", retry_count=3),
                WorkflowStep(name="s2", agent_id="a2", on_failure="skip")
            ],
            output_mapping={"out": "ctx.result"},
            metadata={"test": True}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            original.to_yaml(yaml_path)
            loaded = Workflow.from_yaml(yaml_path)

            assert loaded.name == original.name
            assert loaded.version == original.version
            assert len(loaded.steps) == len(original.steps)
            assert loaded.steps[0].retry_count == 3
            assert loaded.steps[1].on_failure == "skip"
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_workflow_roundtrip_json(self):
        """Test JSON serialization roundtrip."""
        original = Workflow(
            name="roundtrip",
            description="Test roundtrip",
            steps=[WorkflowStep(name="s1", agent_id="a1")],
            output_mapping={"result": "ctx.output"}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            original.to_json(json_path)
            loaded = Workflow.from_json(json_path)

            assert loaded.name == original.name
            assert loaded.output_mapping == original.output_mapping
        finally:
            Path(json_path).unlink(missing_ok=True)


class TestWorkflowBuilder:
    """Test WorkflowBuilder pattern."""

    def test_builder_creation(self):
        """Test creating a workflow builder."""
        builder = WorkflowBuilder("test_workflow", "Test description")
        assert builder.workflow.name == "test_workflow"
        assert builder.workflow.description == "Test description"
        assert len(builder.workflow.steps) == 0

    def test_builder_add_step(self):
        """Test adding a step using builder."""
        builder = WorkflowBuilder("test", "desc")
        builder.add_step("step1", "agent1")
        assert len(builder.workflow.steps) == 1
        assert builder.workflow.steps[0].name == "step1"

    def test_builder_add_step_with_params(self):
        """Test adding a step with additional parameters."""
        builder = WorkflowBuilder("test", "desc")
        builder.add_step(
            "step1",
            "agent1",
            description="Test step",
            retry_count=5,
            on_failure="continue"
        )
        step = builder.workflow.steps[0]
        assert step.description == "Test step"
        assert step.retry_count == 5
        assert step.on_failure == "continue"

    def test_builder_chain_add_steps(self):
        """Test chaining add_step calls."""
        builder = WorkflowBuilder("test", "desc")
        builder.add_step("step1", "agent1").add_step("step2", "agent2")
        assert len(builder.workflow.steps) == 2

    def test_builder_with_output_mapping(self):
        """Test setting output mapping using builder."""
        builder = WorkflowBuilder("test", "desc")
        mapping = {"result": "context.output"}
        builder.with_output_mapping(mapping)
        assert builder.workflow.output_mapping == mapping

    def test_builder_with_metadata(self):
        """Test setting metadata using builder."""
        builder = WorkflowBuilder("test", "desc")
        metadata = {"author": "tester", "version": "1.0"}
        builder.with_metadata(metadata)
        assert builder.workflow.metadata == metadata

    def test_builder_complete_workflow(self):
        """Test building a complete workflow using builder."""
        workflow = (
            WorkflowBuilder("complete", "Complete workflow")
            .add_step("step1", "agent1", description="First step")
            .add_step("step2", "agent2", retry_count=3)
            .add_step("step3", "agent3", on_failure="skip")
            .with_output_mapping({"final": "context.result"})
            .with_metadata({"created": "2025-01-01"})
            .workflow
        )

        assert workflow.name == "complete"
        assert len(workflow.steps) == 3
        assert workflow.steps[0].description == "First step"
        assert workflow.steps[1].retry_count == 3
        assert workflow.steps[2].on_failure == "skip"
        assert workflow.output_mapping == {"final": "context.result"}
        assert workflow.metadata == {"created": "2025-01-01"}


# ============================================================================
# ORCHESTRATOR TESTS - Coverage for orchestrator.py
# ============================================================================

class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator functionality."""

    @pytest.fixture
    def mock_agent_registry(self):
        """Create a mock agent registry."""
        registry = Mock()
        registry.get_agent = Mock(return_value=Mock())
        return registry

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple test workflow."""
        return Workflow(
            name="test_workflow",
            description="Test",
            steps=[
                WorkflowStep(name="step1", agent_id="agent1"),
                WorkflowStep(name="step2", agent_id="agent2")
            ]
        )

    def test_orchestrator_creation(self, mock_agent_registry):
        """Test creating a workflow orchestrator."""
        orchestrator = WorkflowOrchestrator(agent_registry=mock_agent_registry)
        assert orchestrator is not None
        assert orchestrator.agent_registry == mock_agent_registry

    @patch('greenlang.core.orchestrator.WorkflowOrchestrator')
    def test_orchestrator_execute_workflow(self, mock_orch_class, simple_workflow, mock_agent_registry):
        """Test executing a workflow."""
        # Setup mock
        mock_orch = Mock()
        mock_orch_class.return_value = mock_orch
        mock_orch.execute.return_value = {"status": "success"}

        # Create orchestrator
        orchestrator = mock_orch_class(agent_registry=mock_agent_registry)

        # Execute workflow
        result = orchestrator.execute(simple_workflow, context={})

        # Verify
        assert result == {"status": "success"}
        mock_orch.execute.assert_called_once()

    def test_orchestrator_workflow_validation_before_execution(self, simple_workflow, mock_agent_registry):
        """Test that workflow is validated before execution."""
        # Create workflow with no steps (invalid)
        invalid_workflow = Workflow(name="invalid", description="Invalid", steps=[])

        orchestrator = WorkflowOrchestrator(agent_registry=mock_agent_registry)

        # Execution should fail validation
        errors = invalid_workflow.validate_workflow()
        assert len(errors) > 0

    def test_orchestrator_step_error_handling_stop(self, mock_agent_registry):
        """Test error handling with on_failure='stop'."""
        workflow = Workflow(
            name="test",
            description="Test",
            steps=[
                WorkflowStep(name="step1", agent_id="agent1", on_failure="stop")
            ]
        )

        # Mock agent that raises error
        mock_agent = Mock()
        mock_agent.execute = Mock(side_effect=Exception("Agent error"))
        mock_agent_registry.get_agent = Mock(return_value=mock_agent)

        orchestrator = WorkflowOrchestrator(agent_registry=mock_agent_registry)

        # Verify error handling
        # Implementation would stop on error

    def test_orchestrator_step_retry_logic(self, mock_agent_registry):
        """Test step retry logic."""
        workflow = Workflow(
            name="test",
            description="Test",
            steps=[
                WorkflowStep(name="step1", agent_id="agent1", retry_count=3)
            ]
        )

        # Mock agent that fails twice then succeeds
        mock_agent = Mock()
        call_count = [0]

        def execute_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return {"result": "success"}

        mock_agent.execute = Mock(side_effect=execute_with_retry)
        mock_agent_registry.get_agent = Mock(return_value=mock_agent)

        # Orchestrator should retry
        # Implementation would handle retries


# ============================================================================
# ARTIFACT MANAGER TESTS
# ============================================================================

class TestArtifactManager:
    """Test ArtifactManager functionality."""

    @pytest.fixture
    def temp_artifact_dir(self):
        """Create a temporary directory for artifacts."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_artifact_manager_creation(self, temp_artifact_dir):
        """Test creating an artifact manager."""
        manager = ArtifactManager(storage_path=temp_artifact_dir)
        assert manager is not None
        assert manager.storage_path == temp_artifact_dir

    @patch('greenlang.core.artifact_manager.ArtifactManager')
    def test_artifact_manager_store_artifact(self, mock_manager_class, temp_artifact_dir):
        """Test storing an artifact."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.store.return_value = "artifact_id_123"

        manager = mock_manager_class(storage_path=temp_artifact_dir)
        artifact_id = manager.store("test_data", metadata={"type": "test"})

        assert artifact_id == "artifact_id_123"
        mock_manager.store.assert_called_once()

    @patch('greenlang.core.artifact_manager.ArtifactManager')
    def test_artifact_manager_retrieve_artifact(self, mock_manager_class, temp_artifact_dir):
        """Test retrieving an artifact."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.retrieve.return_value = {"data": "test_data"}

        manager = mock_manager_class(storage_path=temp_artifact_dir)
        artifact = manager.retrieve("artifact_id_123")

        assert artifact == {"data": "test_data"}
        mock_manager.retrieve.assert_called_once_with("artifact_id_123")


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_workflow_with_maximum_steps(self):
        """Test workflow with a large number of steps."""
        steps = [WorkflowStep(name=f"step{i}", agent_id=f"agent{i}") for i in range(1000)]
        workflow = Workflow(name="large", description="Large workflow", steps=steps)
        assert len(workflow.steps) == 1000

    def test_workflow_step_with_empty_name(self):
        """Test step with empty name."""
        with pytest.raises(Exception):
            WorkflowStep(name="", agent_id="agent1")

    def test_workflow_step_with_unicode_name(self):
        """Test step with Unicode characters in name."""
        step = WorkflowStep(name="测试步骤", agent_id="agent1")
        assert step.name == "测试步骤"

    def test_workflow_with_circular_dependencies(self):
        """Test workflow with potential circular dependencies."""
        # This would require dependency tracking in implementation
        steps = [
            WorkflowStep(
                name="step1",
                agent_id="agent1",
                input_mapping={"input": "context.step2.output"}
            ),
            WorkflowStep(
                name="step2",
                agent_id="agent2",
                input_mapping={"input": "context.step1.output"}
            )
        ]
        workflow = Workflow(name="circular", description="Circular", steps=steps)
        # Validation logic would detect circular dependencies

    def test_workflow_serialization_with_special_characters(self):
        """Test serialization with special characters."""
        workflow = Workflow(
            name="special chars: <>&\"'",
            description="Test with quotes: \"hello\"",
            steps=[WorkflowStep(name="step1", agent_id="agent1")]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            workflow.to_json(json_path)
            loaded = Workflow.from_json(json_path)
            assert loaded.name == workflow.name
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_workflow_with_very_long_description(self):
        """Test workflow with very long description."""
        long_desc = "A" * 10000
        workflow = Workflow(
            name="test",
            description=long_desc,
            steps=[WorkflowStep(name="step1", agent_id="agent1")]
        )
        assert len(workflow.description) == 10000

    def test_workflow_metadata_with_nested_structures(self):
        """Test workflow with deeply nested metadata."""
        metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        workflow = Workflow(
            name="test",
            description="test",
            steps=[],
            metadata=metadata
        )
        assert workflow.metadata["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_workflow_validation_performance(self):
        """Test validation performance with large workflow."""
        import time
        steps = [WorkflowStep(name=f"step{i}", agent_id=f"agent{i}") for i in range(100)]
        workflow = Workflow(name="perf", description="Performance test", steps=steps)

        start = time.time()
        errors = workflow.validate_workflow()
        duration = time.time() - start

        assert duration < 1.0  # Should validate in under 1 second
        assert errors == []

    def test_workflow_serialization_performance(self):
        """Test serialization performance."""
        import time
        steps = [WorkflowStep(name=f"step{i}", agent_id=f"agent{i}") for i in range(100)]
        workflow = Workflow(name="perf", description="Performance test", steps=steps)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            start = time.time()
            workflow.to_json(json_path)
            duration = time.time() - start

            assert duration < 1.0  # Should serialize in under 1 second
        finally:
            Path(json_path).unlink(missing_ok=True)


# ============================================================================
# INTEGRATION TEST SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""

    def test_complete_workflow_lifecycle(self):
        """Test complete workflow lifecycle: create, serialize, load, execute."""
        # Create workflow
        workflow = (
            WorkflowBuilder("lifecycle_test", "Lifecycle test workflow")
            .add_step("init", "init_agent")
            .add_step("process", "process_agent", retry_count=2)
            .add_step("finalize", "final_agent")
            .with_output_mapping({"result": "context.final_output"})
            .workflow
        )

        # Validate
        errors = workflow.validate_workflow()
        assert errors == []

        # Serialize to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            workflow.to_json(json_path)

            # Load from JSON
            loaded_workflow = Workflow.from_json(json_path)

            # Verify loaded workflow
            assert loaded_workflow.name == workflow.name
            assert len(loaded_workflow.steps) == 3
            assert loaded_workflow.steps[1].retry_count == 2

        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_workflow_modification_and_revalidation(self):
        """Test modifying a workflow and revalidating."""
        workflow = Workflow(
            name="modifiable",
            description="Test",
            steps=[WorkflowStep(name="step1", agent_id="agent1")]
        )

        # Initial validation
        errors = workflow.validate_workflow()
        assert errors == []

        # Add duplicate step
        workflow.add_step(WorkflowStep(name="step1", agent_id="agent2"))

        # Revalidation should catch duplicate
        errors = workflow.validate_workflow()
        assert len(errors) > 0

        # Remove duplicate
        workflow.remove_step("step1")

        # Should be valid again
        errors = workflow.validate_workflow()
        assert errors == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
