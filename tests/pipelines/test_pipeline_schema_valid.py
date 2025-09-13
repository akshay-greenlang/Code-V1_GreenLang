"""
Tests for valid pipeline schema validation.

Tests various valid pipeline configurations to ensure they pass validation
against the JSON schema and Pydantic models.
"""

import pytest
from typing import Dict, Any

from greenlang.sdk.pipeline import Pipeline, PipelineValidationError
from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec, RetrySpec, OnErrorObj


class TestValidPipelineSchemas:
    """Test valid pipeline configurations."""

    def test_valid_minimal_pipeline(self):
        """Test minimal valid pipeline spec."""
        pipeline_data = {
            "name": "minimal-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "single_step",
                    "agent": "test.Agent",
                    "action": "run"
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "minimal-pipeline"
        assert pipeline.spec.version == "1.0.0"
        assert len(pipeline.spec.steps) == 1
        assert pipeline.spec.steps[0].name == "single_step"
        assert pipeline.spec.steps[0].agent == "test.Agent"
        assert pipeline.spec.steps[0].action == "run"

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_valid_full_featured_pipeline(self):
        """Test pipeline with all possible valid fields."""
        pipeline_data = {
            "name": "comprehensive-pipeline",
            "version": "2.1.0",
            "description": "A comprehensive test pipeline with all features",
            "author": "Test Author",
            "tags": ["test", "comprehensive", "validation"],
            "inputs": {
                "input_data": {
                    "type": "object",
                    "required": ["value"]
                }
            },
            "outputs": {
                "result": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "value": {"type": "number"}
                    }
                }
            },
            "parameters": {
                "max_iterations": 10,
                "timeout_seconds": 300,
                "enable_debug": False
            },
            "artifacts_dir": "results",
            "stop_on_error": True,
            "max_parallel_steps": 3,
            "on_error": "stop",
            "hooks": {
                "on_start": [
                    {"action": "log", "message": "Pipeline starting"}
                ],
                "on_complete": [
                    {"action": "log", "message": "Pipeline completed"}
                ],
                "on_error": [
                    {"action": "log", "message": "Pipeline failed"}
                ]
            },
            "metadata": {
                "category": "test",
                "complexity": "high"
            },
            "steps": [
                {
                    "name": "validate_input",
                    "agent": "validation.InputValidator",
                    "action": "validate",
                    "description": "Validate incoming data",
                    "inputs": {
                        "data": "${inputs.input_data}",
                        "schema": "input_schema.json"
                    },
                    "with": {
                        "strict_mode": True,
                        "allow_extra_fields": False
                    },
                    "timeout": 30.0,
                    "on_error": "stop",
                    "outputs": {
                        "validated_data": {"type": "object"}
                    }
                },
                {
                    "name": "process_data",
                    "agent": "processing.DataProcessor",
                    "action": "process",
                    "description": "Process the validated data",
                    "inputsRef": "${steps.validate_input.outputs}",
                    "with": {
                        "algorithm": "advanced",
                        "parallel": True
                    },
                    "parallel": True,
                    "timeout": 120.0,
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 3,
                            "backoff_seconds": 2.0
                        }
                    },
                    "outputs": {
                        "processed_data": {"type": "object"},
                        "metrics": {"type": "object"}
                    }
                },
                {
                    "name": "generate_report",
                    "agent": "reporting.ReportGenerator",
                    "action": "generate",
                    "description": "Generate final report",
                    "inputs": {
                        "data": "${steps.process_data.outputs.processed_data}",
                        "template": "comprehensive_report.j2"
                    },
                    "condition": "${steps.process_data.success}",
                    "timeout": 60.0,
                    "on_error": "skip",
                    "outputs": {
                        "report": {"type": "string"},
                        "artifacts": {"type": "array"}
                    }
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "comprehensive-pipeline"
        assert pipeline.spec.version == "2.1.0"
        assert pipeline.spec.description == "A comprehensive test pipeline with all features"
        assert pipeline.spec.author == "Test Author"
        assert len(pipeline.spec.tags) == 3
        assert pipeline.spec.artifacts_dir == "results"
        assert pipeline.spec.stop_on_error is True
        assert pipeline.spec.max_parallel_steps == 3
        assert len(pipeline.spec.steps) == 3

        # Check individual steps
        validate_step = pipeline.spec.steps[0]
        assert validate_step.name == "validate_input"
        assert validate_step.timeout == 30.0
        assert validate_step.on_error == "stop"

        process_step = pipeline.spec.steps[1]
        assert process_step.name == "process_data"
        assert process_step.parallel is True
        assert process_step.inputsRef == "${steps.validate_input.outputs}"
        assert hasattr(process_step.on_error, 'policy')
        assert process_step.on_error.policy == "continue"
        assert process_step.on_error.retry.max == 3
        assert process_step.on_error.retry.backoff_seconds == 2.0

        report_step = pipeline.spec.steps[2]
        assert report_step.name == "generate_report"
        assert report_step.condition == "${steps.process_data.success}"
        assert report_step.on_error == "skip"

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_valid_pipeline_with_complex_references(self):
        """Test pipeline with complex reference patterns."""
        pipeline_data = {
            "name": "reference-test-pipeline",
            "version": "1.0.0",
            "inputs": {
                "base_config": {
                    "database_url": "postgresql://localhost:5432/test",
                    "api_key": "test-key"
                }
            },
            "parameters": {
                "batch_size": 100,
                "max_retries": 5
            },
            "steps": [
                {
                    "name": "setup_environment",
                    "agent": "setup.EnvironmentSetup",
                    "action": "configure",
                    "inputs": {
                        "database_url": "${inputs.base_config.database_url}",
                        "batch_size": "${vars.batch_size}",
                        "environment": "${env.ENVIRONMENT}",
                        "timestamp": "${env.BUILD_TIMESTAMP}"
                    }
                },
                {
                    "name": "load_data",
                    "agent": "data.DataLoader",
                    "action": "load",
                    "inputs": {
                        "connection": "${steps.setup_environment.outputs.connection}",
                        "config": "${inputs.base_config}",
                        "max_retries": "${vars.max_retries}"
                    }
                },
                {
                    "name": "process_batch",
                    "agent": "processing.BatchProcessor",
                    "action": "process_batch",
                    "inputs": {
                        "data": "${steps.load_data.outputs.data}",
                        "metadata": "${steps.load_data.outputs.metadata}",
                        "settings": {
                            "batch_size": "${vars.batch_size}",
                            "connection_string": "${steps.setup_environment.outputs.connection_string}"
                        }
                    },
                    "condition": "${steps.load_data.success} and ${steps.setup_environment.outputs.ready}"
                },
                {
                    "name": "finalize",
                    "agent": "cleanup.Finalizer",
                    "action": "cleanup",
                    "inputs": {
                        "results": "${steps.process_batch.outputs}",
                        "temp_files": "${steps.load_data.outputs.temp_files}",
                        "connection": "${steps.setup_environment.outputs.connection}"
                    },
                    "condition": "${steps.process_batch.completed}"
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "reference-test-pipeline"
        assert len(pipeline.spec.steps) == 4

        # Verify reference patterns are preserved in inputs
        setup_step = pipeline.spec.steps[0]
        assert setup_step.inputs["database_url"] == "${inputs.base_config.database_url}"
        assert setup_step.inputs["batch_size"] == "${vars.batch_size}"
        assert setup_step.inputs["environment"] == "${env.ENVIRONMENT}"

        load_step = pipeline.spec.steps[1]
        assert load_step.inputs["connection"] == "${steps.setup_environment.outputs.connection}"
        assert load_step.inputs["max_retries"] == "${vars.max_retries}"

        process_step = pipeline.spec.steps[2]
        assert process_step.inputs["data"] == "${steps.load_data.outputs.data}"
        assert process_step.condition == "${steps.load_data.success} and ${steps.setup_environment.outputs.ready}"

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_valid_pipeline_with_retry_and_error_handling(self):
        """Test pipeline with comprehensive retry and error handling."""
        pipeline_data = {
            "name": "resilient-pipeline",
            "version": "1.0.0",
            "stop_on_error": False,
            "on_error": {
                "policy": "continue",
                "retry": {
                    "max": 2,
                    "backoff_seconds": 1.0
                }
            },
            "steps": [
                {
                    "name": "critical_step",
                    "agent": "critical.CriticalAgent",
                    "action": "execute",
                    "inputs": {
                        "data": "important_data"
                    },
                    "on_error": "stop"  # This step must succeed
                },
                {
                    "name": "retry_step",
                    "agent": "network.NetworkAgent",
                    "action": "fetch",
                    "inputs": {
                        "url": "https://api.example.com/data",
                        "timeout": 30
                    },
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 5,
                            "backoff_seconds": 2.5
                        }
                    }
                },
                {
                    "name": "optional_step",
                    "agent": "optional.OptionalAgent",
                    "action": "enhance",
                    "inputs": {
                        "base_data": "${steps.critical_step.outputs.data}",
                        "enhancement_data": "${steps.retry_step.outputs.data}"
                    },
                    "condition": "${steps.retry_step.success}",
                    "on_error": "skip"  # Skip if this fails
                },
                {
                    "name": "fail_fast_step",
                    "agent": "validation.StrictValidator",
                    "action": "validate",
                    "inputs": {
                        "data": "${steps.critical_step.outputs.data}"
                    },
                    "on_error": "fail"  # Fail entire pipeline if this fails
                },
                {
                    "name": "parallel_safe_step",
                    "agent": "parallel.SafeAgent",
                    "action": "process",
                    "inputs": {
                        "data": "${steps.critical_step.outputs.data}"
                    },
                    "parallel": True,
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 3,
                            "backoff_seconds": 0.5
                        }
                    }
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "resilient-pipeline"
        assert pipeline.spec.stop_on_error is False
        assert hasattr(pipeline.spec.on_error, 'policy')
        assert pipeline.spec.on_error.policy == "continue"
        assert pipeline.spec.on_error.retry.max == 2

        # Check step error configurations
        critical_step = pipeline.spec.steps[0]
        assert critical_step.on_error == "stop"

        retry_step = pipeline.spec.steps[1]
        assert hasattr(retry_step.on_error, 'policy')
        assert retry_step.on_error.policy == "continue"
        assert retry_step.on_error.retry.max == 5
        assert retry_step.on_error.retry.backoff_seconds == 2.5

        optional_step = pipeline.spec.steps[2]
        assert optional_step.on_error == "skip"
        assert optional_step.condition == "${steps.retry_step.success}"

        fail_step = pipeline.spec.steps[3]
        assert fail_step.on_error == "fail"

        parallel_step = pipeline.spec.steps[4]
        assert parallel_step.parallel is True
        assert hasattr(parallel_step.on_error, 'policy')
        assert parallel_step.on_error.retry.max == 3

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_valid_pipeline_with_reserved_keywords(self):
        """Test pipeline using reserved Python keywords correctly."""
        pipeline_data = {
            "name": "keyword-test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "test_in_keyword",
                    "agent": "test.Agent",
                    "action": "run",
                    "in": {  # Using 'in' keyword
                        "data": "test_data",
                        "config": {"key": "value"}
                    }
                },
                {
                    "name": "test_with_keyword",
                    "agent": "test.Agent",
                    "action": "configure",
                    "inputs": {
                        "source": "${steps.test_in_keyword.outputs}"
                    },
                    "with": {  # Using 'with' keyword
                        "mode": "advanced",
                        "debug": True,
                        "nested": {
                            "setting1": "value1",
                            "setting2": 42
                        }
                    }
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "keyword-test-pipeline"
        assert len(pipeline.spec.steps) == 2

        # Check that 'in' was properly handled and normalized to 'inputs'
        first_step = pipeline.spec.steps[0]
        assert first_step.inputs is not None
        assert first_step.inputs["data"] == "test_data"
        assert first_step.inputs["config"]["key"] == "value"

        # Check that 'with' was properly handled
        second_step = pipeline.spec.steps[1]
        assert second_step.with_ is not None
        assert second_step.with_["mode"] == "advanced"
        assert second_step.with_["debug"] is True
        assert second_step.with_["nested"]["setting1"] == "value1"

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_valid_parallel_pipeline(self):
        """Test pipeline with parallel execution capabilities."""
        pipeline_data = {
            "name": "parallel-pipeline",
            "version": "1.0.0",
            "max_parallel_steps": 5,
            "steps": [
                {
                    "name": "sequential_setup",
                    "agent": "setup.SetupAgent",
                    "action": "initialize",
                    "parallel": False
                },
                {
                    "name": "parallel_task_1",
                    "agent": "worker.WorkerAgent",
                    "action": "process",
                    "inputs": {
                        "task_id": 1,
                        "data": "${steps.sequential_setup.outputs.data}"
                    },
                    "parallel": True
                },
                {
                    "name": "parallel_task_2",
                    "agent": "worker.WorkerAgent",
                    "action": "process",
                    "inputs": {
                        "task_id": 2,
                        "data": "${steps.sequential_setup.outputs.data}"
                    },
                    "parallel": True
                },
                {
                    "name": "parallel_task_3",
                    "agent": "worker.WorkerAgent",
                    "action": "process",
                    "inputs": {
                        "task_id": 3,
                        "data": "${steps.sequential_setup.outputs.data}"
                    },
                    "parallel": True
                },
                {
                    "name": "sequential_aggregation",
                    "agent": "aggregator.AggregatorAgent",
                    "action": "combine",
                    "inputs": {
                        "results": [
                            "${steps.parallel_task_1.outputs}",
                            "${steps.parallel_task_2.outputs}",
                            "${steps.parallel_task_3.outputs}"
                        ]
                    },
                    "parallel": False
                }
            ]
        }

        # Should create without errors
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "parallel-pipeline"
        assert pipeline.spec.max_parallel_steps == 5
        assert len(pipeline.spec.steps) == 5

        # Check parallel settings
        assert pipeline.spec.steps[0].parallel is False
        assert pipeline.spec.steps[1].parallel is True
        assert pipeline.spec.steps[2].parallel is True
        assert pipeline.spec.steps[3].parallel is True
        assert pipeline.spec.steps[4].parallel is False

        # Get parallel steps using the method
        parallel_steps = pipeline.spec.get_parallel_steps()
        assert len(parallel_steps) == 3
        parallel_names = [step.name for step in parallel_steps]
        assert "parallel_task_1" in parallel_names
        assert "parallel_task_2" in parallel_names
        assert "parallel_task_3" in parallel_names

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_step_spec_creation_directly(self):
        """Test creating StepSpec directly with valid data."""
        # Test minimal step
        minimal_step = StepSpec(
            name="test_step",
            agent="test.Agent"
        )
        assert minimal_step.name == "test_step"
        assert minimal_step.agent == "test.Agent"
        assert minimal_step.action == "run"  # default
        assert minimal_step.on_error == "stop"  # default
        assert minimal_step.parallel is False  # default

        # Test step with retry configuration
        retry_step = StepSpec(
            name="retry_step",
            agent="retry.Agent",
            action="execute",
            on_error=OnErrorObj(
                policy="continue",
                retry=RetrySpec(max=3, backoff_seconds=1.5)
            )
        )
        assert retry_step.name == "retry_step"
        assert retry_step.on_error.policy == "continue"
        assert retry_step.on_error.retry.max == 3
        assert retry_step.on_error.retry.backoff_seconds == 1.5

        # Test step with all optional fields
        comprehensive_step = StepSpec(
            name="comprehensive_step",
            agent="comprehensive.Agent",
            action="process",
            inputs={"data": "test"},
            with_={"config": "value"},
            condition="${previous.success}",
            parallel=True,
            timeout=60.0,
            description="A comprehensive test step",
            id="step_123",
            outputs={"result": "object"}
        )
        assert comprehensive_step.description == "A comprehensive test step"
        assert comprehensive_step.id == "step_123"
        assert comprehensive_step.timeout == 60.0
        assert comprehensive_step.parallel is True
        assert comprehensive_step.condition == "${previous.success}"

    def test_pipeline_spec_creation_directly(self):
        """Test creating PipelineSpec directly with valid data."""
        steps = [
            StepSpec(name="step1", agent="agent1"),
            StepSpec(name="step2", agent="agent2", parallel=True)
        ]

        pipeline_spec = PipelineSpec(
            name="direct-test-pipeline",
            version="1.0.0",
            description="Test pipeline created directly",
            author="Test Author",
            tags=["test", "direct"],
            steps=steps,
            inputs={"test_input": "value"},
            outputs={"test_output": "result"},
            parameters={"param1": "value1"},
            artifacts_dir="output",
            stop_on_error=True,
            max_parallel_steps=2,
            on_error="continue",
            metadata={"category": "test"}
        )

        assert pipeline_spec.name == "direct-test-pipeline"
        assert pipeline_spec.version == "1.0.0"
        assert len(pipeline_spec.steps) == 2
        assert pipeline_spec.get_step("step1") is not None
        assert pipeline_spec.get_step("step2") is not None
        assert pipeline_spec.get_step("nonexistent") is None

        parallel_steps = pipeline_spec.get_parallel_steps()
        assert len(parallel_steps) == 1
        assert parallel_steps[0].name == "step2"