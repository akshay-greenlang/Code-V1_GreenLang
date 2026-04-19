# -*- coding: utf-8 -*-
"""
Tests for invalid pipeline schema validation.

Tests various invalid pipeline configurations to ensure they properly fail
validation against the JSON schema and Pydantic models.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from greenlang.sdk.pipeline import Pipeline, PipelineValidationError
from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec, RetrySpec, OnErrorObj


class TestInvalidPipelineSchemas:
    """Test invalid pipeline configurations that should fail validation."""

    def test_missing_required_name_field(self):
        """Test pipeline missing required 'name' field."""
        pipeline_data = {
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent"
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "name" in str(exc_info.value).lower()

    def test_missing_required_steps_field(self):
        """Test pipeline missing required 'steps' field."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0"
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "steps" in str(exc_info.value).lower()

    def test_empty_steps_array(self):
        """Test pipeline with empty steps array."""
        pipeline_data = {
            "name": "empty-pipeline",
            "version": "1.0.0",
            "steps": []
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Should fail due to min_length=1 constraint
        assert "at least 1" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()

    def test_step_missing_required_name(self):
        """Test step missing required 'name' field."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "agent": "test.Agent"
                    # Missing 'name' field
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "name" in str(exc_info.value).lower()

    def test_step_missing_required_agent(self):
        """Test step missing required 'agent' field."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step"
                    # Missing 'agent' field
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "agent" in str(exc_info.value).lower()

    def test_invalid_on_error_values(self):
        """Test various invalid on_error values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "on_error": None  # Will be replaced in tests
                }
            ]
        }

        invalid_on_error_values = [
            "invalid_policy",
            "STOP",  # Wrong case
            "Continue",  # Wrong case
            123,  # Number instead of string
            [],  # Array instead of string/object
            {"policy": "invalid_policy"},  # Invalid policy in object
            {"retry": {"max": 3}},  # Missing policy in object
            {"policy": "stop", "retry": "invalid"},  # Invalid retry value
            {"policy": "stop", "retry": {"max": -1}},  # Negative max retries
            {"policy": "stop", "retry": {"max": 15}},  # Too many retries (limit is 10)
            {"policy": "stop", "retry": {"backoff_seconds": -1.0}},  # Negative backoff
        ]

        for invalid_value in invalid_on_error_values:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "on_error": invalid_value
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_both_inputs_and_inputsref_present(self):
        """Test step with both 'inputs' and 'inputsRef' fields."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "setup_step",
                    "agent": "setup.Agent"
                },
                {
                    "name": "invalid_step",
                    "agent": "test.Agent",
                    "inputs": {"data": "test"},
                    "inputsRef": "${steps.setup_step.outputs}"
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "mutually exclusive" in str(exc_info.value).lower()

    def test_both_in_and_inputsref_present(self):
        """Test step with both 'in' and 'inputsRef' fields."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "setup_step",
                    "agent": "setup.Agent"
                },
                {
                    "name": "invalid_step",
                    "agent": "test.Agent",
                    "in": {"data": "test"},
                    "inputsRef": "${steps.setup_step.outputs}"
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "mutually exclusive" in str(exc_info.value).lower()

    def test_both_inputs_and_in_present(self):
        """Test step with both 'inputs' and 'in' fields."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "invalid_step",
                    "agent": "test.Agent",
                    "inputs": {"data": "test"},
                    "in": {"other_data": "test2"}
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "mutually exclusive" in str(exc_info.value).lower()

    def test_invalid_reference_syntax(self):
        """Test various invalid reference syntax patterns."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "first_step",
                    "agent": "first.Agent"
                },
                {
                    "name": "second_step",
                    "agent": "second.Agent",
                    "inputsRef": None  # Will be replaced
                }
            ]
        }

        invalid_references = [
            "steps.first_step.outputs",  # Missing $ prefix
            "${step.first_step.outputs}",  # Wrong prefix (step vs steps)
            "${steps.}",  # Empty step name
            "${steps..outputs}",  # Double dot
            "${steps.nonexistent.outputs}",  # Nonexistent step
            "${inputs.}",  # Empty input name
            "${vars.}",  # Empty var name
            "${env.}",  # Empty env name
            "${invalid.first_step.outputs}",  # Invalid prefix
            123,  # Number instead of string
            [],  # Array instead of string
        ]

        for invalid_ref in invalid_references:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": "first_step",
                    "agent": "first.Agent"
                },
                {
                    "name": "second_step",
                    "agent": "second.Agent",
                    "inputsRef": invalid_ref
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_retry_configuration(self):
        """Test invalid retry configurations."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "on_error": None  # Will be replaced
                }
            ]
        }

        invalid_retry_configs = [
            # Invalid retry object structures
            {"policy": "continue", "retry": {}},  # Missing required fields
            {"policy": "continue", "retry": {"max": "invalid"}},  # String instead of int
            {"policy": "continue", "retry": {"backoff_seconds": "invalid"}},  # String instead of float
            {"policy": "continue", "retry": {"max": 3}},  # Missing backoff_seconds
            {"policy": "continue", "retry": {"backoff_seconds": 1.0}},  # Missing max
            {"policy": "continue", "retry": {"max": -1, "backoff_seconds": 1.0}},  # Negative max
            {"policy": "continue", "retry": {"max": 11, "backoff_seconds": 1.0}},  # Max > 10
            {"policy": "continue", "retry": {"max": 3, "backoff_seconds": -1.0}},  # Negative backoff
        ]

        for invalid_config in invalid_retry_configs:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "on_error": invalid_config
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_step_names(self):
        """Test invalid step names."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": None,  # Will be replaced
                    "agent": "test.Agent"
                }
            ]
        }

        invalid_names = [
            "",  # Empty string
            "   ",  # Whitespace only
            "123invalid",  # Starting with number (depending on validation rules)
            "invalid-name-",  # Ending with hyphen
            "invalid name",  # Space in name
            "invalid@name",  # Special character
            "invalid.name",  # Dot in name
            None,  # None value
            123,  # Number instead of string
        ]

        for invalid_name in invalid_names:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": invalid_name,
                    "agent": "test.Agent"
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_duplicate_step_names(self):
        """Test pipeline with duplicate step names."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "duplicate_step",
                    "agent": "first.Agent"
                },
                {
                    "name": "unique_step",
                    "agent": "second.Agent"
                },
                {
                    "name": "duplicate_step",  # Duplicate name
                    "agent": "third.Agent"
                }
            ]
        }

        with pytest.raises(PipelineValidationError) as exc_info:
            Pipeline.from_dict(pipeline_data, validate_spec=True)

        assert "duplicate" in str(exc_info.value).lower()

    def test_invalid_timeout_values(self):
        """Test invalid timeout values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "timeout": None  # Will be replaced
                }
            ]
        }

        invalid_timeouts = [
            -1.0,  # Negative timeout
            0,  # Zero timeout
            "30",  # String instead of number
            [],  # Array instead of number
            {},  # Object instead of number
        ]

        for invalid_timeout in invalid_timeouts:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "timeout": invalid_timeout
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_parallel_values(self):
        """Test invalid parallel field values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "parallel": None  # Will be replaced
                }
            ]
        }

        invalid_parallel_values = [
            "true",  # String instead of boolean
            "false",  # String instead of boolean
            1,  # Number instead of boolean
            0,  # Number instead of boolean
            [],  # Array instead of boolean
            {},  # Object instead of boolean
        ]

        for invalid_value in invalid_parallel_values:
            pipeline_data = base_pipeline.copy()
            pipeline_data["steps"] = [
                {
                    "name": "test_step",
                    "agent": "test.Agent",
                    "parallel": invalid_value
                }
            ]

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_pipeline_name(self):
        """Test invalid pipeline names."""
        base_pipeline = {
            "name": None,  # Will be replaced
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent"
                }
            ]
        }

        invalid_names = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            123,  # Number instead of string
            [],  # Array instead of string
            {},  # Object instead of string
        ]

        for invalid_name in invalid_names:
            pipeline_data = base_pipeline.copy()
            pipeline_data["name"] = invalid_name

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_version_values(self):
        """Test invalid version values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": None,  # Will be replaced
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent"
                }
            ]
        }

        invalid_versions = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            123,  # Number instead of string
            [],  # Array instead of string
            {},  # Object instead of string
        ]

        for invalid_version in invalid_versions:
            pipeline_data = base_pipeline.copy()
            pipeline_data["version"] = invalid_version

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_max_parallel_steps(self):
        """Test invalid max_parallel_steps values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "max_parallel_steps": None,  # Will be replaced
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent"
                }
            ]
        }

        invalid_max_parallel = [
            0,  # Zero or negative values
            -1,
            "5",  # String instead of number
            [],  # Array instead of number
            {},  # Object instead of number
        ]

        for invalid_value in invalid_max_parallel:
            pipeline_data = base_pipeline.copy()
            pipeline_data["max_parallel_steps"] = invalid_value

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_stop_on_error_values(self):
        """Test invalid stop_on_error values."""
        base_pipeline = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "stop_on_error": None,  # Will be replaced
            "steps": [
                {
                    "name": "test_step",
                    "agent": "test.Agent"
                }
            ]
        }

        invalid_stop_on_error_values = [
            "true",  # String instead of boolean
            "false",  # String instead of boolean
            1,  # Number instead of boolean
            0,  # Number instead of boolean
            [],  # Array instead of boolean
            {},  # Object instead of boolean
        ]

        for invalid_value in invalid_stop_on_error_values:
            pipeline_data = base_pipeline.copy()
            pipeline_data["stop_on_error"] = invalid_value

            with pytest.raises(PipelineValidationError):
                Pipeline.from_dict(pipeline_data, validate_spec=True)

    def test_invalid_step_references_in_conditions(self):
        """Test invalid step references in conditions."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "first_step",
                    "agent": "first.Agent"
                },
                {
                    "name": "second_step",
                    "agent": "second.Agent",
                    "condition": "${steps.nonexistent_step.success}"  # References nonexistent step
                }
            ]
        }

        # This should create the pipeline but fail validation
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        errors = pipeline.validate()
        assert len(errors) > 0
        assert any("nonexistent_step" in error for error in errors)

    def test_self_referencing_step(self):
        """Test step referencing itself."""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "self_ref_step",
                    "agent": "test.Agent",
                    "inputsRef": "${steps.self_ref_step.outputs}"  # Self reference
                }
            ]
        }

        # This should create the pipeline but fail validation
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        errors = pipeline.validate()
        assert len(errors) > 0
        assert any("cannot reference itself" in error for error in errors)

    def test_non_dict_pipeline_data(self):
        """Test creating pipeline from non-dictionary data."""
        invalid_data_types = [
            "string",
            123,
            [],
            None,
            True
        ]

        for invalid_data in invalid_data_types:
            with pytest.raises(PipelineValidationError) as exc_info:
                Pipeline.from_dict(invalid_data, validate_spec=True)

            assert "dictionary" in str(exc_info.value).lower()

    def test_retry_spec_validation_directly(self):
        """Test RetrySpec validation directly."""
        # Valid retry spec
        valid_retry = RetrySpec(max=3, backoff_seconds=1.5)
        assert valid_retry.max == 3
        assert valid_retry.backoff_seconds == 1.5

        # Invalid retry specs
        with pytest.raises(ValidationError):
            RetrySpec(max=-1, backoff_seconds=1.0)  # Negative max

        with pytest.raises(ValidationError):
            RetrySpec(max=11, backoff_seconds=1.0)  # Max > 10

        with pytest.raises(ValidationError):
            RetrySpec(max=3, backoff_seconds=-1.0)  # Negative backoff

    def test_step_spec_validation_directly(self):
        """Test StepSpec validation directly."""
        # Valid step spec
        valid_step = StepSpec(name="test", agent="agent")
        assert valid_step.name == "test"

        # Invalid step specs
        with pytest.raises(ValidationError):
            StepSpec(name="", agent="agent")  # Empty name

        with pytest.raises(ValidationError):
            StepSpec(name="test", agent="")  # Empty agent

        with pytest.raises(ValidationError):
            StepSpec(
                name="test",
                agent="agent",
                inputs={"data": "test"},
                inputsRef="${steps.other.outputs}"  # Both inputs and inputsRef
            )

        with pytest.raises(ValidationError):
            StepSpec(
                name="test",
                agent="agent",
                timeout=-1.0  # Negative timeout
            )