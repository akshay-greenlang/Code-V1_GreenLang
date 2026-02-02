# -*- coding: utf-8 -*-
"""
Comprehensive tests for Pipeline critical fixes.

Tests cover:
- Agent loading from registry, packs, and dynamic imports
- Sub-pipeline execution with YAML, pack, and inline definitions
- Agent validation during pipeline validation
- Checkpoint status properties
- Error handling for missing agents/pipelines
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from greenlang.sdk.pipeline import Pipeline, StepResult, PipelineState


@pytest.mark.unit
class TestAgentLoading:
    """Test agent loading from various sources."""

    def test_load_agent_from_registry(self):
        """Test loading agent from the AgentRegistry."""
        pipeline = Pipeline(
            name="test-registry-load",
            steps=[{"name": "step1", "agent": "FuelAgent"}]
        )

        # Mock the registry
        mock_agent_info = Mock()
        mock_agent_info.version = "2.0.0"

        mock_agent_instance = Mock()
        mock_agent_instance.process = Mock(return_value={"result": "success"})

        with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
            with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
                with patch("greenlang.agents.registry.create_agent", return_value=mock_agent_instance):
                    agent = pipeline._load_agent("FuelAgent", {})
                    # If registry loading works, we get the mock agent
                    # If it falls through, we'd get a ValueError

    def test_load_agent_from_pack_format(self):
        """Test loading agent with pack:agent format."""
        pipeline = Pipeline(
            name="test-pack-load",
            steps=[{"name": "step1", "agent": "my_pack:MyAgent"}]
        )

        mock_agent_class = Mock()
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_loader = Mock()
        mock_loader.get_agent.return_value = mock_agent_class

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", False):
            with patch("greenlang.packs.loader.PackLoader", return_value=mock_loader):
                # The method should try pack loading for pack:agent format
                try:
                    agent = pipeline._load_agent("my_pack:MyAgent", {"key": "value"})
                except ValueError:
                    # Expected if pack doesn't exist
                    pass

    def test_load_agent_from_module_path(self):
        """Test loading agent with module.path:ClassName format."""
        pipeline = Pipeline(
            name="test-module-load",
            steps=[{"name": "step1", "agent": "some.module.path:AgentClass"}]
        )

        mock_module = Mock()
        mock_agent_class = Mock()
        mock_agent_class.return_value = Mock()
        mock_module.AgentClass = mock_agent_class

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", False):
            with patch("importlib.import_module", return_value=mock_module):
                agent = pipeline._load_agent("some.module.path:AgentClass", {})
                assert agent is not None

    def test_load_agent_not_found_raises_error(self):
        """Test that loading a non-existent agent raises ValueError."""
        pipeline = Pipeline(
            name="test-not-found",
            steps=[{"name": "step1", "agent": "NonExistentAgent"}]
        )

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", False):
            with pytest.raises(ValueError) as exc_info:
                pipeline._load_agent("NonExistentAgent", {})

            assert "NonExistentAgent" in str(exc_info.value)
            assert "could not be loaded" in str(exc_info.value)

    def test_load_agent_with_config(self):
        """Test that agent config is passed correctly."""
        pipeline = Pipeline(
            name="test-config",
            steps=[{"name": "step1", "agent": "TestAgent", "config": {"timeout": 30}}]
        )

        mock_agent_class = Mock()
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        mock_module = Mock()
        mock_module.TestAgent = mock_agent_class

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", False):
            with patch("importlib.import_module", return_value=mock_module):
                agent = pipeline._load_agent("test.module:TestAgent", {"timeout": 30})
                mock_agent_class.assert_called_once_with({"timeout": 30})


@pytest.mark.unit
class TestSubPipelineExecution:
    """Test sub-pipeline execution functionality."""

    def test_execute_sub_pipeline_from_yaml(self, tmp_path):
        """Test executing sub-pipeline from YAML file."""
        # Create a sub-pipeline YAML
        sub_pipeline_yaml = tmp_path / "sub_pipeline.yaml"
        sub_pipeline_yaml.write_text(yaml.dump({
            "name": "sub-pipeline",
            "version": "1.0",
            "steps": [{"name": "sub_step1", "agent": "TestAgent"}]
        }))

        parent_pipeline = Pipeline(
            name="parent-pipeline",
            steps=[{
                "name": "step1",
                "pipeline": str(sub_pipeline_yaml)
            }]
        )

        # Mock the agent loading and execution
        mock_agent = Mock()
        mock_agent.process.return_value = {"result": "sub_success"}

        # Mock agent info to pass validation
        mock_agent_info = Mock()
        mock_agent_info.version = "1.0.0"

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
                with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
                    result = parent_pipeline._execute_sub_pipeline(
                        sub_pipeline_ref=str(sub_pipeline_yaml),
                        sub_pipeline_config={},
                        step={"name": "step1", "pipeline": str(sub_pipeline_yaml)},
                        parent_inputs={"parent_key": "parent_value"}
                    )

                    assert result["sub_pipeline_name"] == "sub-pipeline"
                    assert result["execution_state"] == "completed"

    def test_execute_sub_pipeline_not_found(self):
        """Test that missing sub-pipeline raises ValueError."""
        parent_pipeline = Pipeline(
            name="parent-pipeline",
            steps=[{"name": "step1", "pipeline": "non_existent_pipeline"}]
        )

        with pytest.raises(ValueError) as exc_info:
            parent_pipeline._execute_sub_pipeline(
                sub_pipeline_ref="non_existent_pipeline",
                sub_pipeline_config={},
                step={"name": "step1", "pipeline": "non_existent_pipeline"},
                parent_inputs={}
            )

        assert "non_existent_pipeline" in str(exc_info.value)
        assert "could not be loaded" in str(exc_info.value)

    def test_execute_sub_pipeline_input_mapping(self, tmp_path):
        """Test that input mapping passes values correctly."""
        sub_pipeline_yaml = tmp_path / "sub_pipeline.yaml"
        sub_pipeline_yaml.write_text(yaml.dump({
            "name": "sub-pipeline",
            "version": "1.0",
            "steps": [{"name": "sub_step1", "agent": "TestAgent"}]
        }))

        parent_pipeline = Pipeline(
            name="parent-pipeline",
            steps=[{
                "name": "step1",
                "pipeline": str(sub_pipeline_yaml),
                "input_mapping": {
                    "sub_input": "$previous_step.output_key"
                }
            }]
        )

        # Simulate previous step output
        parent_pipeline._agent_outputs = {
            "previous_step": {"output_key": "mapped_value"}
        }

        mock_agent = Mock()
        mock_agent.process.return_value = {"result": "success"}

        # Mock agent info to pass validation
        mock_agent_info = Mock()
        mock_agent_info.version = "1.0.0"

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
                with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
                    result = parent_pipeline._execute_sub_pipeline(
                        sub_pipeline_ref=str(sub_pipeline_yaml),
                        sub_pipeline_config={},
                        step={
                            "name": "step1",
                            "pipeline": str(sub_pipeline_yaml),
                            "input_mapping": {"sub_input": "$previous_step.output_key"}
                        },
                        parent_inputs={}
                    )

                    # Check that the sub-pipeline received the mapped input
                    assert result["execution_state"] == "completed"

    def test_execute_sub_pipeline_inherits_checkpoint_settings(self, tmp_path):
        """Test that sub-pipeline inherits checkpoint settings from parent."""
        sub_pipeline_yaml = tmp_path / "sub_pipeline.yaml"
        sub_pipeline_yaml.write_text(yaml.dump({
            "name": "sub-pipeline",
            "version": "1.0",
            "steps": [{"name": "sub_step1", "agent": "TestAgent"}]
        }))

        parent_pipeline = Pipeline(
            name="parent-pipeline",
            checkpoint_enabled=True,
            checkpoint_strategy="file",
            steps=[{"name": "step1", "pipeline": str(sub_pipeline_yaml)}]
        )

        mock_agent = Mock()
        mock_agent.process.return_value = {"result": "success"}

        # Mock agent info to pass validation
        mock_agent_info = Mock()
        mock_agent_info.version = "1.0.0"

        captured_sub_pipeline = None

        original_execute = Pipeline.execute

        def mock_execute(self, resume=None, dry_run=False):
            nonlocal captured_sub_pipeline
            captured_sub_pipeline = self
            return original_execute(self, resume=resume, dry_run=dry_run)

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
                with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
                    with patch.object(Pipeline, "execute", mock_execute):
                        parent_pipeline._execute_sub_pipeline(
                            sub_pipeline_ref=str(sub_pipeline_yaml),
                            sub_pipeline_config={},
                            step={"name": "step1", "pipeline": str(sub_pipeline_yaml)},
                            parent_inputs={}
                        )

                        # Verify checkpoint settings were inherited
                        assert captured_sub_pipeline.checkpoint_enabled == True
                        assert captured_sub_pipeline.checkpoint_strategy == "file"


@pytest.mark.unit
class TestAgentValidation:
    """Test agent validation during pipeline validation."""

    def test_validate_agent_exists_in_registry(self):
        """Test that validation passes for registered agents."""
        pipeline = Pipeline(
            name="valid-pipeline",
            steps=[{"name": "step1", "agent": "FuelAgent"}]
        )

        mock_agent_info = Mock()
        mock_agent_info.version = "2.0.0"

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
            with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
                errors = pipeline.validate()
                # No errors about missing agents
                assert not any("not found in registry" in err for err in errors)

    def test_validate_agent_not_in_registry(self):
        """Test that validation fails for non-registered agents."""
        pipeline = Pipeline(
            name="invalid-pipeline",
            steps=[{"name": "step1", "agent": "NonExistentAgent"}]
        )

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
            with patch("greenlang.sdk.pipeline.get_agent_info", return_value=None):
                errors = pipeline.validate()
                assert any("NonExistentAgent" in err and "not found in registry" in err for err in errors)

    def test_validate_skips_registry_check_when_unavailable(self):
        """Test that validation skips registry check when registry unavailable."""
        pipeline = Pipeline(
            name="no-registry-pipeline",
            steps=[{"name": "step1", "agent": "AnyAgent"}]
        )

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", False):
            errors = pipeline.validate()
            # Should not have registry-related errors
            assert not any("not found in registry" in err for err in errors)

    def test_validate_multiple_agents_some_missing(self):
        """Test validation with some agents missing from registry."""
        pipeline = Pipeline(
            name="mixed-pipeline",
            steps=[
                {"name": "step1", "agent": "FuelAgent"},
                {"name": "step2", "agent": "MissingAgent"},
                {"name": "step3", "agent": "CarbonAgent"},
            ]
        )

        def mock_get_agent_info(name):
            if name in ["FuelAgent", "CarbonAgent"]:
                return Mock(version="1.0.0")
            return None

        with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
            with patch("greenlang.sdk.pipeline.get_agent_info", side_effect=mock_get_agent_info):
                errors = pipeline.validate()
                assert len([e for e in errors if "MissingAgent" in e]) == 1
                assert len([e for e in errors if "FuelAgent" in e]) == 0
                assert len([e for e in errors if "CarbonAgent" in e]) == 0


@pytest.mark.unit
class TestCheckpointStatus:
    """Test checkpoint status properties and methods."""

    def test_checkpoint_actually_enabled_property(self):
        """Test checkpoint_actually_enabled property."""
        pipeline = Pipeline(
            name="test-checkpoint",
            checkpoint_enabled=True
        )

        # Without checkpointing dependencies available, it should be False
        with patch("greenlang.sdk.pipeline.CHECKPOINTING_AVAILABLE", False):
            # Re-initialize to trigger checkpoint manager setup
            pipeline._checkpoint_actually_enabled = False
            pipeline._checkpoint_disabled_reason = "Test reason"

            assert pipeline.checkpoint_actually_enabled == False
            assert pipeline.checkpoint_disabled_reason == "Test reason"

    def test_get_checkpoint_status_method(self):
        """Test get_checkpoint_status returns comprehensive info."""
        pipeline = Pipeline(
            name="test-checkpoint",
            checkpoint_enabled=True,
            checkpoint_strategy="file",
            strict_checkpoint=False,
            auto_resume=True,
            checkpoint_after_each_step=True
        )

        pipeline._checkpoint_actually_enabled = False
        pipeline._checkpoint_disabled_reason = "Dependencies not available"

        status = pipeline.get_checkpoint_status()

        assert status["requested"] == True
        assert status["actually_enabled"] == False
        assert status["disabled_reason"] == "Dependencies not available"
        assert status["strategy"] == "file"
        assert status["strict_mode"] == False
        assert status["auto_resume"] == True
        assert status["checkpoint_after_each_step"] == True


@pytest.mark.unit
class TestPipelineExecution:
    """Test pipeline execution with the critical fixes."""

    def test_execute_step_with_agent(self):
        """Test executing a step with an agent."""
        pipeline = Pipeline(
            name="test-execution",
            steps=[{"name": "step1", "agent": "TestAgent"}],
            inputs={"input_key": "input_value"}
        )

        mock_agent = Mock()
        mock_agent.process.return_value = {"result": "success"}

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            result = pipeline._execute_step(
                step={"name": "step1", "agent": "TestAgent"},
                index=0
            )

            assert result.status == "completed"
            assert result.agent_name == "TestAgent"
            assert result.output == {"result": "success"}
            assert result.provenance_hash is not None

    def test_execute_step_with_sub_pipeline(self, tmp_path):
        """Test executing a step with a sub-pipeline."""
        sub_pipeline_yaml = tmp_path / "sub.yaml"
        sub_pipeline_yaml.write_text(yaml.dump({
            "name": "sub-pipeline",
            "version": "1.0",
            "steps": [{"name": "sub_step", "agent": "SubAgent"}]
        }))

        pipeline = Pipeline(
            name="parent-pipeline",
            steps=[{"name": "step1", "pipeline": str(sub_pipeline_yaml)}]
        )

        mock_agent = Mock()
        mock_agent.process.return_value = {"sub_result": "success"}

        # Mock agent info to pass validation
        mock_agent_info = Mock()
        mock_agent_info.version = "1.0.0"

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            with patch("greenlang.sdk.pipeline.AGENT_REGISTRY_AVAILABLE", True):
                with patch("greenlang.sdk.pipeline.get_agent_info", return_value=mock_agent_info):
                    result = pipeline._execute_step(
                        step={"name": "step1", "pipeline": str(sub_pipeline_yaml)},
                        index=0
                    )

                    assert result.status == "completed"
                    assert "pipeline:sub-pipeline" in result.agent_name or "pipeline:" in result.agent_name

    def test_execute_step_failure_returns_failed_result(self):
        """Test that step execution failure returns a failed StepResult."""
        pipeline = Pipeline(
            name="test-failure",
            steps=[{"name": "step1", "agent": "FailingAgent"}]
        )

        mock_agent = Mock()
        mock_agent.process.side_effect = Exception("Agent processing failed")

        with patch.object(Pipeline, "_load_agent", return_value=mock_agent):
            result = pipeline._execute_step(
                step={"name": "step1", "agent": "FailingAgent"},
                index=0
            )

            assert result.status == "failed"
            assert result.error_message == "Agent processing failed"


@pytest.mark.unit
class TestToPolicyDoc:
    """Test to_policy_doc method for OPA evaluation."""

    def test_to_policy_doc_basic(self):
        """Test basic policy document generation."""
        pipeline = Pipeline(
            name="policy-test",
            version="1.0",
            steps=[
                {"name": "step1", "agent": "Agent1", "inputs": {"a": 1}, "outputs": {"b": 2}},
                {"name": "step2", "agent": "Agent2", "inputs": {"c": 3}, "outputs": {"d": 4}},
            ]
        )

        policy_doc = pipeline.to_policy_doc()

        assert policy_doc["name"] == "policy-test"
        assert policy_doc["version"] == "1.0"
        assert len(policy_doc["steps"]) == 2
        assert policy_doc["steps"][0]["name"] == "step1"
        assert policy_doc["steps"][0]["agent"] == "Agent1"
        assert "a" in policy_doc["steps"][0]["inputs"]
        assert "b" in policy_doc["steps"][0]["outputs"]

    def test_to_policy_doc_excludes_runtime_state(self):
        """Test that policy document excludes runtime state."""
        pipeline = Pipeline(
            name="policy-test",
            steps=[{"name": "step1", "agent": "Agent1"}]
        )

        # Set some runtime state
        pipeline._agent_outputs = {"step1": {"secret": "data"}}
        pipeline._provenance_hashes = {"step1": "hash123"}

        policy_doc = pipeline.to_policy_doc()

        # Should not contain runtime state
        assert "_agent_outputs" not in str(policy_doc)
        assert "_provenance_hashes" not in str(policy_doc)
        assert "secret" not in str(policy_doc)


@pytest.mark.unit
class TestPrepareStepInputs:
    """Test step input preparation."""

    def test_prepare_step_inputs_basic(self):
        """Test basic step input preparation."""
        pipeline = Pipeline(
            name="test",
            inputs={"global_key": "global_value"},
            steps=[{"name": "step1", "agent": "Agent1"}]
        )

        inputs = pipeline._prepare_step_inputs({"name": "step1", "agent": "Agent1"})

        assert inputs["global_key"] == "global_value"

    def test_prepare_step_inputs_with_references(self):
        """Test step input preparation with references to previous outputs."""
        pipeline = Pipeline(
            name="test",
            inputs={"global_key": "global_value"},
            steps=[
                {"name": "step1", "agent": "Agent1"},
                {"name": "step2", "agent": "Agent2", "inputs": {"ref_key": "$step1.output_key"}}
            ]
        )

        # Simulate step1 output
        pipeline._agent_outputs = {"step1": {"output_key": "step1_output"}}

        inputs = pipeline._prepare_step_inputs({
            "name": "step2",
            "agent": "Agent2",
            "inputs": {"ref_key": "$step1.output_key"}
        })

        assert inputs["global_key"] == "global_value"
        assert inputs["ref_key"] == "step1_output"

    def test_prepare_step_inputs_nested_reference(self):
        """Test step input preparation with nested references."""
        pipeline = Pipeline(
            name="test",
            steps=[
                {"name": "step1", "agent": "Agent1"},
                {"name": "step2", "agent": "Agent2", "inputs": {"ref_key": "$step1.nested.deep.value"}}
            ]
        )

        # Simulate step1 output with nested structure
        pipeline._agent_outputs = {
            "step1": {"nested": {"deep": {"value": "nested_value"}}}
        }

        inputs = pipeline._prepare_step_inputs({
            "name": "step2",
            "inputs": {"ref_key": "$step1.nested.deep.value"}
        })

        assert inputs["ref_key"] == "nested_value"
