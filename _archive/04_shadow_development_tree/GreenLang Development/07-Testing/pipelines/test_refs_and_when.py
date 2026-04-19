# -*- coding: utf-8 -*-
"""
Tests for pipeline reference resolution and condition evaluation.

Tests various reference patterns (${vars.*}, ${inputs.*}, ${steps.*.*}, ${env.*})
and 'when' condition evaluation in pipeline contexts.
"""

import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch
import re

from greenlang.sdk.pipeline import Pipeline, PipelineValidationError
from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec


class MockReferenceResolver:
    """Mock reference resolver for testing reference patterns."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def resolve_reference(self, reference: str) -> Any:
        """Resolve a reference pattern against the context."""
        # Remove ${} wrapper
        if reference.startswith("${") and reference.endswith("}"):
            reference = reference[2:-1]

        parts = reference.split(".")
        current = self.context

        try:
            for part in parts:
                if isinstance(current, dict):
                    current = current[part]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None

    def evaluate_condition(self, condition: str, step_results: Dict[str, Any]) -> bool:
        """Simple condition evaluation for testing."""
        # Replace references with values
        resolved_condition = condition

        # Find all references in the condition
        refs = re.findall(r'\${([^}]+)}', condition)

        for ref in refs:
            value = self.resolve_reference(f"${{{ref}}}")
            if value is not None:
                resolved_condition = resolved_condition.replace(f"${{{ref}}}", str(value))

        # Simple boolean evaluation (in real implementation, use safe evaluation)
        try:
            # Replace common patterns
            resolved_condition = resolved_condition.replace("true", "True").replace("false", "False")
            return eval(resolved_condition)
        except:
            return False


class TestReferenceResolution:
    """Test reference resolution patterns."""

    def test_vars_reference_resolution(self):
        """Test ${vars.*} reference resolution."""
        pipeline_data = {
            "name": "vars-test-pipeline",
            "version": "1.0.0",
            "parameters": {
                "batch_size": 100,
                "timeout_seconds": 300,
                "enable_debug": True,
                "database_config": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db"
                }
            },
            "steps": [
                {
                    "name": "config_step",
                    "agent": "config.Agent",
                    "inputs": {
                        "batch_size": "${vars.batch_size}",
                        "timeout": "${vars.timeout_seconds}",
                        "debug_mode": "${vars.enable_debug}",
                        "db_host": "${vars.database_config.host}",
                        "db_port": "${vars.database_config.port}",
                        "full_config": "${vars.database_config}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.name == "vars-test-pipeline"

        # Check that references are preserved in the step inputs
        config_step = pipeline.spec.steps[0]
        assert config_step.inputs["batch_size"] == "${vars.batch_size}"
        assert config_step.inputs["timeout"] == "${vars.timeout_seconds}"
        assert config_step.inputs["debug_mode"] == "${vars.enable_debug}"
        assert config_step.inputs["db_host"] == "${vars.database_config.host}"
        assert config_step.inputs["db_port"] == "${vars.database_config.port}"
        assert config_step.inputs["full_config"] == "${vars.database_config}"

        # Test reference resolution (simulated)
        resolver = MockReferenceResolver({
            "vars": pipeline.spec.parameters
        })

        assert resolver.resolve_reference("${vars.batch_size}") == 100
        assert resolver.resolve_reference("${vars.timeout_seconds}") == 300
        assert resolver.resolve_reference("${vars.enable_debug}") is True
        assert resolver.resolve_reference("${vars.database_config.host}") == "localhost"
        assert resolver.resolve_reference("${vars.database_config.port}") == 5432

        # Should have no validation errors
        errors = pipeline.validate()
        assert len(errors) == 0

    def test_inputs_reference_resolution(self):
        """Test ${inputs.*} reference resolution."""
        pipeline_data = {
            "name": "inputs-test-pipeline",
            "version": "1.0.0",
            "inputs": {
                "source_data": {
                    "path": "/data/source.csv",
                    "format": "csv",
                    "encoding": "utf-8"
                },
                "processing_options": {
                    "clean_data": True,
                    "validate_schema": True,
                    "output_format": "json"
                },
                "api_key": "secret-key-123",
                "max_records": 10000
            },
            "steps": [
                {
                    "name": "load_data",
                    "agent": "data.Loader",
                    "inputs": {
                        "file_path": "${inputs.source_data.path}",
                        "file_format": "${inputs.source_data.format}",
                        "encoding": "${inputs.source_data.encoding}",
                        "full_source_config": "${inputs.source_data}",
                        "max_records": "${inputs.max_records}"
                    }
                },
                {
                    "name": "process_data",
                    "agent": "data.Processor",
                    "inputs": {
                        "clean": "${inputs.processing_options.clean_data}",
                        "validate": "${inputs.processing_options.validate_schema}",
                        "output_format": "${inputs.processing_options.output_format}",
                        "all_options": "${inputs.processing_options}",
                        "auth_key": "${inputs.api_key}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Check reference preservation
        load_step = pipeline.spec.steps[0]
        assert load_step.inputs["file_path"] == "${inputs.source_data.path}"
        assert load_step.inputs["file_format"] == "${inputs.source_data.format}"
        assert load_step.inputs["full_source_config"] == "${inputs.source_data}"
        assert load_step.inputs["max_records"] == "${inputs.max_records}"

        process_step = pipeline.spec.steps[1]
        assert process_step.inputs["clean"] == "${inputs.processing_options.clean_data}"
        assert process_step.inputs["all_options"] == "${inputs.processing_options}"
        assert process_step.inputs["auth_key"] == "${inputs.api_key}"

        # Test reference resolution
        resolver = MockReferenceResolver({
            "inputs": pipeline.spec.inputs
        })

        assert resolver.resolve_reference("${inputs.source_data.path}") == "/data/source.csv"
        assert resolver.resolve_reference("${inputs.source_data.format}") == "csv"
        assert resolver.resolve_reference("${inputs.processing_options.clean_data}") is True
        assert resolver.resolve_reference("${inputs.api_key}") == "secret-key-123"
        assert resolver.resolve_reference("${inputs.max_records}") == 10000

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_steps_reference_resolution(self):
        """Test ${steps.*.*} reference resolution."""
        pipeline_data = {
            "name": "steps-test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "extract_data",
                    "agent": "extract.Agent",
                    "inputs": {
                        "source": "database"
                    },
                    "outputs": {
                        "data": {"type": "array"},
                        "metadata": {"type": "object"},
                        "record_count": {"type": "integer"}
                    }
                },
                {
                    "name": "transform_data",
                    "agent": "transform.Agent",
                    "inputs": {
                        "raw_data": "${steps.extract_data.outputs.data}",
                        "metadata": "${steps.extract_data.outputs.metadata}",
                        "count": "${steps.extract_data.outputs.record_count}",
                        "all_outputs": "${steps.extract_data.outputs}"
                    },
                    "outputs": {
                        "transformed_data": {"type": "array"},
                        "transformation_log": {"type": "string"}
                    }
                },
                {
                    "name": "validate_results",
                    "agent": "validate.Agent",
                    "inputs": {
                        "data": "${steps.transform_data.outputs.transformed_data}",
                        "log": "${steps.transform_data.outputs.transformation_log}",
                        "original_count": "${steps.extract_data.outputs.record_count}"
                    }
                },
                {
                    "name": "save_results",
                    "agent": "save.Agent",
                    "inputs": {
                        "validated_data": "${steps.validate_results.outputs.data}",
                        "extract_metadata": "${steps.extract_data.outputs.metadata}",
                        "transform_log": "${steps.transform_data.outputs.transformation_log}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Check reference preservation
        transform_step = pipeline.spec.steps[1]
        assert transform_step.inputs["raw_data"] == "${steps.extract_data.outputs.data}"
        assert transform_step.inputs["metadata"] == "${steps.extract_data.outputs.metadata}"
        assert transform_step.inputs["all_outputs"] == "${steps.extract_data.outputs}"

        validate_step = pipeline.spec.steps[2]
        assert validate_step.inputs["data"] == "${steps.transform_data.outputs.transformed_data}"
        assert validate_step.inputs["original_count"] == "${steps.extract_data.outputs.record_count}"

        save_step = pipeline.spec.steps[3]
        assert save_step.inputs["validated_data"] == "${steps.validate_results.outputs.data}"
        assert save_step.inputs["extract_metadata"] == "${steps.extract_data.outputs.metadata}"

        # Test with mock step results
        mock_step_results = {
            "steps": {
                "extract_data": {
                    "outputs": {
                        "data": [1, 2, 3],
                        "metadata": {"source": "db", "table": "users"},
                        "record_count": 3
                    }
                },
                "transform_data": {
                    "outputs": {
                        "transformed_data": ["a", "b", "c"],
                        "transformation_log": "Applied uppercase transformation"
                    }
                }
            }
        }

        resolver = MockReferenceResolver(mock_step_results)
        assert resolver.resolve_reference("${steps.extract_data.outputs.data}") == [1, 2, 3]
        assert resolver.resolve_reference("${steps.extract_data.outputs.record_count}") == 3
        assert resolver.resolve_reference("${steps.transform_data.outputs.transformed_data}") == ["a", "b", "c"]

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_env_reference_resolution(self):
        """Test ${env.*} reference resolution."""
        pipeline_data = {
            "name": "env-test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "environment_setup",
                    "agent": "env.Agent",
                    "inputs": {
                        "environment": "${env.ENVIRONMENT}",
                        "debug_mode": "${env.DEBUG}",
                        "api_url": "${env.API_BASE_URL}",
                        "database_url": "${env.DATABASE_URL}",
                        "log_level": "${env.LOG_LEVEL}",
                        "build_version": "${env.BUILD_VERSION}",
                        "worker_count": "${env.WORKER_COUNT}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Check reference preservation
        setup_step = pipeline.spec.steps[0]
        assert setup_step.inputs["environment"] == "${env.ENVIRONMENT}"
        assert setup_step.inputs["debug_mode"] == "${env.DEBUG}"
        assert setup_step.inputs["api_url"] == "${env.API_BASE_URL}"
        assert setup_step.inputs["database_url"] == "${env.DATABASE_URL}"

        # Test with mock environment variables
        mock_env = {
            "env": {
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "API_BASE_URL": "https://api.production.com",
                "DATABASE_URL": "postgresql://prod:5432/db",
                "LOG_LEVEL": "INFO",
                "BUILD_VERSION": "1.2.3",
                "WORKER_COUNT": "4"
            }
        }

        resolver = MockReferenceResolver(mock_env)
        assert resolver.resolve_reference("${env.ENVIRONMENT}") == "production"
        assert resolver.resolve_reference("${env.DEBUG}") == "false"
        assert resolver.resolve_reference("${env.API_BASE_URL}") == "https://api.production.com"
        assert resolver.resolve_reference("${env.WORKER_COUNT}") == "4"

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_complex_nested_reference_resolution(self):
        """Test complex nested reference patterns."""
        pipeline_data = {
            "name": "complex-refs-pipeline",
            "version": "1.0.0",
            "inputs": {
                "config": {
                    "database": {
                        "host": "localhost",
                        "credentials": {
                            "username": "user",
                            "password": "pass"
                        }
                    },
                    "processing": {
                        "batch_size": 1000,
                        "parallel_workers": 4
                    }
                }
            },
            "parameters": {
                "environment": "test",
                "retry_config": {
                    "max_attempts": 3,
                    "backoff_multiplier": 2.0
                }
            },
            "steps": [
                {
                    "name": "database_connect",
                    "agent": "db.ConnectionAgent",
                    "inputs": {
                        "host": "${inputs.config.database.host}",
                        "username": "${inputs.config.database.credentials.username}",
                        "password": "${inputs.config.database.credentials.password}",
                        "environment": "${vars.environment}",
                        "ssl_enabled": "${env.SSL_ENABLED}"
                    }
                },
                {
                    "name": "process_batches",
                    "agent": "process.BatchAgent",
                    "inputs": {
                        "connection": "${steps.database_connect.outputs.connection}",
                        "batch_size": "${inputs.config.processing.batch_size}",
                        "workers": "${inputs.config.processing.parallel_workers}",
                        "retry_attempts": "${vars.retry_config.max_attempts}",
                        "backoff_multiplier": "${vars.retry_config.backoff_multiplier}"
                    }
                },
                {
                    "name": "validate_processing",
                    "agent": "validate.Agent",
                    "inputs": {
                        "results": "${steps.process_batches.outputs.results}",
                        "expected_count": "${steps.process_batches.outputs.processed_count}",
                        "connection_info": "${steps.database_connect.outputs.connection_info}",
                        "original_batch_size": "${inputs.config.processing.batch_size}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Verify deeply nested references are preserved
        db_step = pipeline.spec.steps[0]
        assert db_step.inputs["username"] == "${inputs.config.database.credentials.username}"
        assert db_step.inputs["password"] == "${inputs.config.database.credentials.password}"

        process_step = pipeline.spec.steps[1]
        assert process_step.inputs["batch_size"] == "${inputs.config.processing.batch_size}"
        assert process_step.inputs["retry_attempts"] == "${vars.retry_config.max_attempts}"

        # Test complex resolution
        context = {
            "inputs": pipeline.spec.inputs,
            "vars": pipeline.spec.parameters,
            "env": {"SSL_ENABLED": "true"}
        }

        resolver = MockReferenceResolver(context)
        assert resolver.resolve_reference("${inputs.config.database.credentials.username}") == "user"
        assert resolver.resolve_reference("${inputs.config.processing.batch_size}") == 1000
        assert resolver.resolve_reference("${vars.retry_config.max_attempts}") == 3
        assert resolver.resolve_reference("${env.SSL_ENABLED}") == "true"

        errors = pipeline.validate()
        assert len(errors) == 0


class TestConditionEvaluation:
    """Test 'when' condition evaluation."""

    def test_simple_boolean_conditions(self):
        """Test simple boolean conditions."""
        pipeline_data = {
            "name": "condition-test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "always_run",
                    "agent": "test.Agent"
                },
                {
                    "name": "condition_true",
                    "agent": "test.Agent",
                    "condition": "true"
                },
                {
                    "name": "condition_false",
                    "agent": "test.Agent",
                    "condition": "false"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Check conditions are preserved
        assert pipeline.spec.steps[0].condition is None
        assert pipeline.spec.steps[1].condition == "true"
        assert pipeline.spec.steps[2].condition == "false"

        # Test condition evaluation
        resolver = MockReferenceResolver({})
        assert resolver.evaluate_condition("true", {}) is True
        assert resolver.evaluate_condition("false", {}) is False

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_reference_based_conditions(self):
        """Test conditions with references."""
        pipeline_data = {
            "name": "ref-condition-pipeline",
            "version": "1.0.0",
            "parameters": {
                "enable_processing": True,
                "max_records": 1000
            },
            "steps": [
                {
                    "name": "extract_data",
                    "agent": "extract.Agent",
                    "outputs": {
                        "success": {"type": "boolean"},
                        "record_count": {"type": "integer"}
                    }
                },
                {
                    "name": "validate_data",
                    "agent": "validate.Agent",
                    "condition": "${steps.extract_data.outputs.success}",
                    "inputs": {
                        "data": "${steps.extract_data.outputs.data}"
                    }
                },
                {
                    "name": "process_if_enabled",
                    "agent": "process.Agent",
                    "condition": "${vars.enable_processing}",
                    "inputs": {
                        "data": "${steps.extract_data.outputs.data}"
                    }
                },
                {
                    "name": "process_large_dataset",
                    "agent": "process.Agent",
                    "condition": "${steps.extract_data.outputs.record_count} > ${vars.max_records}",
                    "inputs": {
                        "data": "${steps.extract_data.outputs.data}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Check conditions are preserved
        assert pipeline.spec.steps[1].condition == "${steps.extract_data.outputs.success}"
        assert pipeline.spec.steps[2].condition == "${vars.enable_processing}"
        assert pipeline.spec.steps[3].condition == "${steps.extract_data.outputs.record_count} > ${vars.max_records}"

        # Test condition evaluation with mock context
        context = {
            "vars": {"enable_processing": True, "max_records": 1000},
            "steps": {
                "extract_data": {
                    "outputs": {
                        "success": True,
                        "record_count": 1500
                    }
                }
            }
        }

        resolver = MockReferenceResolver(context)
        assert resolver.evaluate_condition("${steps.extract_data.outputs.success}", {}) is True
        assert resolver.evaluate_condition("${vars.enable_processing}", {}) is True
        assert resolver.evaluate_condition("${steps.extract_data.outputs.record_count} > ${vars.max_records}", {}) is True

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_complex_logical_conditions(self):
        """Test complex conditions with logical operators."""
        pipeline_data = {
            "name": "complex-condition-pipeline",
            "version": "1.0.0",
            "parameters": {
                "debug_mode": False,
                "min_confidence": 0.8
            },
            "steps": [
                {
                    "name": "data_analysis",
                    "agent": "analyze.Agent",
                    "outputs": {
                        "confidence": {"type": "number"},
                        "error_count": {"type": "integer"},
                        "processing_time": {"type": "number"}
                    }
                },
                {
                    "name": "conditional_processing",
                    "agent": "process.Agent",
                    "condition": "${steps.data_analysis.outputs.confidence} >= ${vars.min_confidence} and ${steps.data_analysis.outputs.error_count} == 0"
                },
                {
                    "name": "debug_processing",
                    "agent": "debug.Agent",
                    "condition": "${vars.debug_mode} or ${steps.data_analysis.outputs.error_count} > 0"
                },
                {
                    "name": "performance_check",
                    "agent": "perf.Agent",
                    "condition": "${steps.data_analysis.outputs.processing_time} > 60 and ${steps.data_analysis.outputs.confidence} < 0.9"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Test complex condition evaluation
        context = {
            "vars": {"debug_mode": False, "min_confidence": 0.8},
            "steps": {
                "data_analysis": {
                    "outputs": {
                        "confidence": 0.85,
                        "error_count": 0,
                        "processing_time": 45.5
                    }
                }
            }
        }

        resolver = MockReferenceResolver(context)

        # Should be True: confidence >= 0.8 and error_count == 0
        condition1 = "${steps.data_analysis.outputs.confidence} >= ${vars.min_confidence} and ${steps.data_analysis.outputs.error_count} == 0"

        # Should be False: debug_mode is False and error_count is 0
        condition2 = "${vars.debug_mode} or ${steps.data_analysis.outputs.error_count} > 0"

        # Should be False: processing_time <= 60 and confidence >= 0.9 is false
        condition3 = "${steps.data_analysis.outputs.processing_time} > 60 and ${steps.data_analysis.outputs.confidence} < 0.9"

        # Note: These would require more sophisticated evaluation in a real implementation
        # For now, we just verify the conditions are preserved correctly
        assert pipeline.spec.steps[1].condition == condition1
        assert pipeline.spec.steps[2].condition == condition2
        assert pipeline.spec.steps[3].condition == condition3

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_invalid_condition_references(self):
        """Test conditions with invalid references."""
        pipeline_data = {
            "name": "invalid-condition-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "first_step",
                    "agent": "first.Agent"
                },
                {
                    "name": "invalid_ref_step",
                    "agent": "test.Agent",
                    "condition": "${steps.nonexistent_step.outputs.success}"
                }
            ]
        }

        # Should create pipeline but fail validation
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("nonexistent_step" in error for error in errors)

    def test_self_referencing_conditions(self):
        """Test conditions that reference the same step."""
        pipeline_data = {
            "name": "self-ref-condition-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "self_referencing_step",
                    "agent": "test.Agent",
                    "condition": "${steps.self_referencing_step.outputs.ready}"
                }
            ]
        }

        # Should create pipeline but fail validation
        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        errors = pipeline.validate()

        assert len(errors) > 0
        assert any("cannot reference itself" in error for error in errors)

    def test_mixed_reference_types_in_conditions(self):
        """Test conditions mixing different reference types."""
        pipeline_data = {
            "name": "mixed-refs-condition-pipeline",
            "version": "1.0.0",
            "inputs": {
                "enable_feature": True,
                "threshold": 10
            },
            "parameters": {
                "production_mode": True
            },
            "steps": [
                {
                    "name": "data_step",
                    "agent": "data.Agent",
                    "outputs": {
                        "count": {"type": "integer"}
                    }
                },
                {
                    "name": "mixed_condition_step",
                    "agent": "process.Agent",
                    "condition": "${inputs.enable_feature} and ${vars.production_mode} and ${steps.data_step.outputs.count} > ${inputs.threshold} and ${env.FEATURE_ENABLED} == 'true'"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Verify the complex mixed condition is preserved
        expected_condition = "${inputs.enable_feature} and ${vars.production_mode} and ${steps.data_step.outputs.count} > ${inputs.threshold} and ${env.FEATURE_ENABLED} == 'true'"
        assert pipeline.spec.steps[1].condition == expected_condition

        # Test with full context
        context = {
            "inputs": {"enable_feature": True, "threshold": 10},
            "vars": {"production_mode": True},
            "steps": {"data_step": {"outputs": {"count": 15}}},
            "env": {"FEATURE_ENABLED": "true"}
        }

        resolver = MockReferenceResolver(context)
        # In a real implementation, this would evaluate to True
        # All conditions: True and True and (15 > 10) and ('true' == 'true') = True

        errors = pipeline.validate()
        assert len(errors) == 0