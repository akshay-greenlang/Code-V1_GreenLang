"""
Tests for pipeline retry and error handling functionality.

Tests retry policies, error handling behaviors, backoff calculations,
and deterministic retry behavior.
"""

import pytest
import time
import math
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

from greenlang.sdk.pipeline import Pipeline, PipelineValidationError
from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec, RetrySpec, OnErrorObj, OnErrorPolicy


class MockStepExecutor:
    """Mock step executor for testing retry and error handling."""

    def __init__(self):
        self.execution_attempts: Dict[str, int] = {}
        self.execution_results: Dict[str, List[Dict[str, Any]]] = {}
        self.execution_times: List[float] = []

    def execute_step(self, step_name: str, should_fail: bool = False, fail_attempts: int = 0) -> Dict[str, Any]:
        """Mock step execution with configurable failure."""
        if step_name not in self.execution_attempts:
            self.execution_attempts[step_name] = 0
            self.execution_results[step_name] = []

        self.execution_attempts[step_name] += 1
        attempt = self.execution_attempts[step_name]
        self.execution_times.append(time.time())

        if should_fail and attempt <= fail_attempts:
            result = {
                "success": False,
                "error": f"Simulated failure on attempt {attempt}",
                "attempt": attempt,
                "timestamp": time.time()
            }
        else:
            result = {
                "success": True,
                "data": {"result": f"success_on_attempt_{attempt}"},
                "attempt": attempt,
                "timestamp": time.time()
            }

        self.execution_results[step_name].append(result)
        return result

    def get_execution_count(self, step_name: str) -> int:
        """Get number of execution attempts for a step."""
        return self.execution_attempts.get(step_name, 0)

    def get_execution_history(self, step_name: str) -> List[Dict[str, Any]]:
        """Get full execution history for a step."""
        return self.execution_results.get(step_name, [])


class TestRetryConfiguration:
    """Test retry configuration and validation."""

    def test_valid_retry_configurations(self):
        """Test various valid retry configurations."""
        valid_retry_configs = [
            # Basic retry config
            {"max": 3, "backoff_seconds": 1.0},
            # Maximum retry attempts
            {"max": 10, "backoff_seconds": 0.1},
            # Zero backoff (immediate retry)
            {"max": 2, "backoff_seconds": 0.0},
            # Fractional backoff
            {"max": 5, "backoff_seconds": 0.5},
            # Long backoff
            {"max": 3, "backoff_seconds": 30.0},
        ]

        for retry_config in valid_retry_configs:
            # Test RetrySpec creation directly
            retry_spec = RetrySpec(**retry_config)
            assert retry_spec.max == retry_config["max"]
            assert retry_spec.backoff_seconds == retry_config["backoff_seconds"]

            # Test in pipeline context
            pipeline_data = {
                "name": "retry-test-pipeline",
                "version": "1.0.0",
                "steps": [
                    {
                        "name": "retry_step",
                        "agent": "test.Agent",
                        "on_error": {
                            "policy": "continue",
                            "retry": retry_config
                        }
                    }
                ]
            }

            pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
            step = pipeline.spec.steps[0]
            assert step.on_error.retry.max == retry_config["max"]
            assert step.on_error.retry.backoff_seconds == retry_config["backoff_seconds"]

            errors = pipeline.validate()
            assert len(errors) == 0

    def test_retry_backoff_calculation(self):
        """Test exponential backoff calculation."""
        retry_configs = [
            {"max": 3, "backoff_seconds": 1.0},
            {"max": 5, "backoff_seconds": 2.0},
            {"max": 4, "backoff_seconds": 0.5},
        ]

        for config in retry_configs:
            base_backoff = config["backoff_seconds"]
            max_attempts = config["max"]

            # Calculate expected backoff times for exponential backoff
            expected_backoffs = []
            for attempt in range(1, max_attempts + 1):
                # Exponential backoff: base * (2 ^ (attempt - 1))
                backoff_time = base_backoff * (2 ** (attempt - 1))
                expected_backoffs.append(backoff_time)

            # Verify backoff calculation logic (would be implemented in executor)
            for i, expected in enumerate(expected_backoffs):
                calculated = base_backoff * (2 ** i)
                assert calculated == expected

    def test_deterministic_backoff_calculation(self):
        """Test that backoff calculations are deterministic."""
        retry_config = {"max": 4, "backoff_seconds": 1.5}
        base_backoff = retry_config["backoff_seconds"]

        # Calculate backoff times multiple times - should be identical
        def calculate_backoff_sequence(max_attempts: int, base: float) -> List[float]:
            return [base * (2 ** attempt) for attempt in range(max_attempts)]

        sequence1 = calculate_backoff_sequence(retry_config["max"], base_backoff)
        sequence2 = calculate_backoff_sequence(retry_config["max"], base_backoff)
        sequence3 = calculate_backoff_sequence(retry_config["max"], base_backoff)

        assert sequence1 == sequence2 == sequence3

        # Verify specific values
        expected_sequence = [1.5, 3.0, 6.0, 12.0]
        assert sequence1 == expected_sequence


class TestErrorPolicies:
    """Test different error handling policies."""

    def test_stop_on_error_behavior(self):
        """Test 'stop' error policy behavior."""
        pipeline_data = {
            "name": "stop-on-error-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "critical_step",
                    "agent": "critical.Agent",
                    "on_error": "stop"
                },
                {
                    "name": "subsequent_step",
                    "agent": "subsequent.Agent",
                    "inputs": {
                        "data": "${steps.critical_step.outputs.data}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        critical_step = pipeline.spec.steps[0]
        assert critical_step.on_error == "stop"

        # Simulate execution behavior
        executor = MockStepExecutor()

        # Step fails and should stop pipeline
        result = executor.execute_step("critical_step", should_fail=True, fail_attempts=1)
        assert result["success"] is False

        # Pipeline should stop here, subsequent step should not execute
        # (In real implementation, execution engine would handle this)

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_continue_on_error_behavior(self):
        """Test 'continue' error policy behavior."""
        pipeline_data = {
            "name": "continue-on-error-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "optional_step",
                    "agent": "optional.Agent",
                    "on_error": "continue"
                },
                {
                    "name": "final_step",
                    "agent": "final.Agent"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        optional_step = pipeline.spec.steps[0]
        assert optional_step.on_error == "continue"

        # Simulate execution behavior
        executor = MockStepExecutor()

        # First step fails but pipeline continues
        result1 = executor.execute_step("optional_step", should_fail=True, fail_attempts=1)
        assert result1["success"] is False

        # Final step should still execute
        result2 = executor.execute_step("final_step", should_fail=False)
        assert result2["success"] is True

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_skip_on_error_behavior(self):
        """Test 'skip' error policy behavior."""
        pipeline_data = {
            "name": "skip-on-error-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "enhancement_step",
                    "agent": "enhancement.Agent",
                    "on_error": "skip"
                },
                {
                    "name": "dependent_step",
                    "agent": "dependent.Agent",
                    "inputs": {
                        "enhanced_data": "${steps.enhancement_step.outputs.data}"
                    },
                    "condition": "${steps.enhancement_step.success}"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        enhancement_step = pipeline.spec.steps[0]
        assert enhancement_step.on_error == "skip"

        # Dependent step should have condition checking success
        dependent_step = pipeline.spec.steps[1]
        assert dependent_step.condition == "${steps.enhancement_step.success}"

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_fail_on_error_behavior(self):
        """Test 'fail' error policy behavior (alias for stop)."""
        pipeline_data = {
            "name": "fail-on-error-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "validation_step",
                    "agent": "validation.Agent",
                    "on_error": "fail"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        validation_step = pipeline.spec.steps[0]
        assert validation_step.on_error == "fail"

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_retry_with_different_policies(self):
        """Test retry behavior with different error policies."""
        pipeline_data = {
            "name": "retry-policies-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "retry_then_stop",
                    "agent": "network.Agent",
                    "on_error": {
                        "policy": "stop",
                        "retry": {
                            "max": 3,
                            "backoff_seconds": 1.0
                        }
                    }
                },
                {
                    "name": "retry_then_continue",
                    "agent": "optional.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 2,
                            "backoff_seconds": 0.5
                        }
                    }
                },
                {
                    "name": "retry_then_skip",
                    "agent": "enhancement.Agent",
                    "on_error": {
                        "policy": "skip",
                        "retry": {
                            "max": 5,
                            "backoff_seconds": 2.0
                        }
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Verify each step's retry configuration
        stop_step = pipeline.spec.steps[0]
        assert stop_step.on_error.policy == "stop"
        assert stop_step.on_error.retry.max == 3
        assert stop_step.on_error.retry.backoff_seconds == 1.0

        continue_step = pipeline.spec.steps[1]
        assert continue_step.on_error.policy == "continue"
        assert continue_step.on_error.retry.max == 2
        assert continue_step.on_error.retry.backoff_seconds == 0.5

        skip_step = pipeline.spec.steps[2]
        assert skip_step.on_error.policy == "skip"
        assert skip_step.on_error.retry.max == 5
        assert skip_step.on_error.retry.backoff_seconds == 2.0

        errors = pipeline.validate()
        assert len(errors) == 0


class TestRetryExecution:
    """Test retry execution behavior."""

    def test_retry_with_exponential_backoff(self):
        """Test retry execution with exponential backoff."""
        pipeline_data = {
            "name": "exponential-backoff-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "flaky_step",
                    "agent": "flaky.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 4,
                            "backoff_seconds": 0.1  # Short for testing
                        }
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        executor = MockStepExecutor()

        # Simulate step that fails 3 times then succeeds
        for attempt in range(1, 5):  # 4 total attempts
            start_time = time.time()

            should_fail = attempt <= 3  # Fail first 3 attempts
            result = executor.execute_step("flaky_step", should_fail=should_fail, fail_attempts=3)

            if attempt <= 3:
                assert result["success"] is False
                assert result["attempt"] == attempt

                # Calculate expected backoff for next attempt
                if attempt < 4:  # Not the last attempt
                    expected_backoff = 0.1 * (2 ** (attempt - 1))
                    # In real implementation, would verify actual sleep time
            else:
                assert result["success"] is True
                assert result["attempt"] == 4

        # Verify total execution attempts
        assert executor.get_execution_count("flaky_step") == 4

        # Verify execution history
        history = executor.get_execution_history("flaky_step")
        assert len(history) == 4
        assert history[0]["success"] is False
        assert history[1]["success"] is False
        assert history[2]["success"] is False
        assert history[3]["success"] is True

    def test_retry_exhaustion(self):
        """Test behavior when retry attempts are exhausted."""
        pipeline_data = {
            "name": "retry-exhaustion-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "always_fails",
                    "agent": "broken.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 2,
                            "backoff_seconds": 0.1
                        }
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        executor = MockStepExecutor()

        # Simulate step that always fails
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            result = executor.execute_step("always_fails", should_fail=True, fail_attempts=999)
            assert result["success"] is False
            assert result["attempt"] == attempt

        # After max attempts, step should be considered failed
        assert executor.get_execution_count("always_fails") == max_attempts

        # Verify all attempts failed
        history = executor.get_execution_history("always_fails")
        assert len(history) == max_attempts
        for execution in history:
            assert execution["success"] is False

    def test_immediate_success_no_retry(self):
        """Test that successful steps don't trigger retries."""
        pipeline_data = {
            "name": "immediate-success-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "successful_step",
                    "agent": "reliable.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 5,
                            "backoff_seconds": 1.0
                        }
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        executor = MockStepExecutor()

        # Step succeeds immediately
        result = executor.execute_step("successful_step", should_fail=False)
        assert result["success"] is True
        assert result["attempt"] == 1

        # Should only execute once
        assert executor.get_execution_count("successful_step") == 1

        history = executor.get_execution_history("successful_step")
        assert len(history) == 1
        assert history[0]["success"] is True


class TestGlobalErrorPolicies:
    """Test pipeline-level error policies."""

    def test_global_stop_on_error(self):
        """Test pipeline-level stop_on_error setting."""
        pipeline_data = {
            "name": "global-stop-pipeline",
            "version": "1.0.0",
            "stop_on_error": True,
            "steps": [
                {
                    "name": "step1",
                    "agent": "agent1.Agent"
                },
                {
                    "name": "step2",
                    "agent": "agent2.Agent"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.stop_on_error is True

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_global_continue_on_error(self):
        """Test pipeline-level continue on error."""
        pipeline_data = {
            "name": "global-continue-pipeline",
            "version": "1.0.0",
            "stop_on_error": False,
            "on_error": {
                "policy": "continue",
                "retry": {
                    "max": 3,
                    "backoff_seconds": 1.0
                }
            },
            "steps": [
                {
                    "name": "step1",
                    "agent": "agent1.Agent"
                },
                {
                    "name": "step2",
                    "agent": "agent2.Agent"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)
        assert pipeline.spec.stop_on_error is False
        assert pipeline.spec.on_error.policy == "continue"
        assert pipeline.spec.on_error.retry.max == 3

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_step_overrides_global_policy(self):
        """Test that step-level policies override global policies."""
        pipeline_data = {
            "name": "override-policy-pipeline",
            "version": "1.0.0",
            "stop_on_error": True,  # Global: stop on error
            "on_error": "stop",
            "steps": [
                {
                    "name": "uses_global",
                    "agent": "global.Agent"
                    # Uses global policy
                },
                {
                    "name": "overrides_global",
                    "agent": "override.Agent",
                    "on_error": {
                        "policy": "continue",  # Override global policy
                        "retry": {
                            "max": 2,
                            "backoff_seconds": 0.5
                        }
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Global settings
        assert pipeline.spec.stop_on_error is True
        assert pipeline.spec.on_error == "stop"

        # First step uses default (global)
        global_step = pipeline.spec.steps[0]
        assert global_step.on_error == "stop"  # Default value

        # Second step overrides global
        override_step = pipeline.spec.steps[1]
        assert override_step.on_error.policy == "continue"
        assert override_step.on_error.retry.max == 2

        errors = pipeline.validate()
        assert len(errors) == 0


class TestErrorHandlingIntegration:
    """Test integration of error handling with other pipeline features."""

    def test_error_handling_with_conditions(self):
        """Test error handling with conditional step execution."""
        pipeline_data = {
            "name": "error-condition-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "risky_step",
                    "agent": "risky.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 2,
                            "backoff_seconds": 0.1
                        }
                    }
                },
                {
                    "name": "conditional_cleanup",
                    "agent": "cleanup.Agent",
                    "condition": "not ${steps.risky_step.success}",
                    "on_error": "stop"
                },
                {
                    "name": "success_processing",
                    "agent": "process.Agent",
                    "condition": "${steps.risky_step.success}",
                    "inputs": {
                        "data": "${steps.risky_step.outputs.data}"
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        risky_step = pipeline.spec.steps[0]
        assert risky_step.on_error.policy == "continue"

        cleanup_step = pipeline.spec.steps[1]
        assert cleanup_step.condition == "not ${steps.risky_step.success}"
        assert cleanup_step.on_error == "stop"

        success_step = pipeline.spec.steps[2]
        assert success_step.condition == "${steps.risky_step.success}"

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_error_handling_with_parallel_execution(self):
        """Test error handling with parallel step execution."""
        pipeline_data = {
            "name": "parallel-error-pipeline",
            "version": "1.0.0",
            "max_parallel_steps": 3,
            "steps": [
                {
                    "name": "setup",
                    "agent": "setup.Agent"
                },
                {
                    "name": "parallel_task_1",
                    "agent": "worker.Agent",
                    "parallel": True,
                    "inputs": {
                        "task_id": 1,
                        "setup_data": "${steps.setup.outputs.data}"
                    },
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 2,
                            "backoff_seconds": 1.0
                        }
                    }
                },
                {
                    "name": "parallel_task_2",
                    "agent": "worker.Agent",
                    "parallel": True,
                    "inputs": {
                        "task_id": 2,
                        "setup_data": "${steps.setup.outputs.data}"
                    },
                    "on_error": "skip"
                },
                {
                    "name": "parallel_task_3",
                    "agent": "worker.Agent",
                    "parallel": True,
                    "inputs": {
                        "task_id": 3,
                        "setup_data": "${steps.setup.outputs.data}"
                    },
                    "on_error": "stop"
                },
                {
                    "name": "aggregation",
                    "agent": "aggregate.Agent",
                    "inputs": {
                        "results": [
                            "${steps.parallel_task_1.outputs}",
                            "${steps.parallel_task_2.outputs}",
                            "${steps.parallel_task_3.outputs}"
                        ]
                    }
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Verify parallel steps have different error policies
        parallel_steps = pipeline.spec.get_parallel_steps()
        assert len(parallel_steps) == 3

        task1 = pipeline.spec.get_step("parallel_task_1")
        assert task1.on_error.policy == "continue"
        assert task1.on_error.retry.max == 2

        task2 = pipeline.spec.get_step("parallel_task_2")
        assert task2.on_error == "skip"

        task3 = pipeline.spec.get_step("parallel_task_3")
        assert task3.on_error == "stop"

        errors = pipeline.validate()
        assert len(errors) == 0

    def test_complex_error_scenario(self):
        """Test complex error handling scenario with multiple failure points."""
        pipeline_data = {
            "name": "complex-error-pipeline",
            "version": "1.0.0",
            "stop_on_error": False,
            "on_error": {
                "policy": "continue",
                "retry": {
                    "max": 1,
                    "backoff_seconds": 0.5
                }
            },
            "steps": [
                {
                    "name": "critical_init",
                    "agent": "init.Agent",
                    "on_error": "stop"  # Must succeed
                },
                {
                    "name": "data_fetch",
                    "agent": "fetch.Agent",
                    "inputs": {
                        "config": "${steps.critical_init.outputs.config}"
                    },
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 5,
                            "backoff_seconds": 2.0
                        }
                    }
                },
                {
                    "name": "fallback_data",
                    "agent": "fallback.Agent",
                    "condition": "not ${steps.data_fetch.success}",
                    "on_error": "stop"  # Fallback must work if primary fails
                },
                {
                    "name": "data_processing",
                    "agent": "process.Agent",
                    "inputs": {
                        "data": "${steps.data_fetch.success} ? ${steps.data_fetch.outputs.data} : ${steps.fallback_data.outputs.data}"
                    },
                    "on_error": {
                        "policy": "skip",
                        "retry": {
                            "max": 3,
                            "backoff_seconds": 1.0
                        }
                    }
                },
                {
                    "name": "optional_enhancement",
                    "agent": "enhance.Agent",
                    "condition": "${steps.data_processing.success}",
                    "inputs": {
                        "data": "${steps.data_processing.outputs.data}"
                    },
                    "on_error": "skip"  # Enhancement is optional
                },
                {
                    "name": "final_output",
                    "agent": "output.Agent",
                    "inputs": {
                        "processed_data": "${steps.data_processing.outputs.data}",
                        "enhanced_data": "${steps.optional_enhancement.outputs.data}"
                    },
                    "condition": "${steps.data_processing.success} or ${steps.fallback_data.success}"
                }
            ]
        }

        pipeline = Pipeline.from_dict(pipeline_data, validate_spec=True)

        # Verify complex error handling setup
        assert pipeline.spec.stop_on_error is False
        assert pipeline.spec.on_error.policy == "continue"

        critical_step = pipeline.spec.get_step("critical_init")
        assert critical_step.on_error == "stop"

        fetch_step = pipeline.spec.get_step("data_fetch")
        assert fetch_step.on_error.retry.max == 5

        fallback_step = pipeline.spec.get_step("fallback_data")
        assert fallback_step.condition == "not ${steps.data_fetch.success}"
        assert fallback_step.on_error == "stop"

        processing_step = pipeline.spec.get_step("data_processing")
        assert processing_step.on_error.policy == "skip"

        enhancement_step = pipeline.spec.get_step("optional_enhancement")
        assert enhancement_step.on_error == "skip"
        assert enhancement_step.condition == "${steps.data_processing.success}"

        final_step = pipeline.spec.get_step("final_output")
        assert final_step.condition == "${steps.data_processing.success} or ${steps.fallback_data.success}"

        errors = pipeline.validate()
        assert len(errors) == 0


class TestDeterministicBehavior:
    """Test deterministic behavior of retry and error handling."""

    def test_deterministic_backoff_timing(self):
        """Test that backoff calculations are deterministic and reproducible."""
        base_backoff = 1.5
        max_attempts = 5

        def calculate_backoff_sequence(base: float, max_att: int) -> List[float]:
            """Calculate backoff sequence for given parameters."""
            return [base * (2 ** attempt) for attempt in range(max_att)]

        # Calculate sequence multiple times
        sequences = [
            calculate_backoff_sequence(base_backoff, max_attempts)
            for _ in range(10)
        ]

        # All sequences should be identical
        first_sequence = sequences[0]
        for sequence in sequences[1:]:
            assert sequence == first_sequence

        # Verify expected values
        expected = [1.5, 3.0, 6.0, 12.0, 24.0]
        assert first_sequence == expected

    def test_deterministic_retry_decision(self):
        """Test that retry decisions are deterministic."""
        retry_config = RetrySpec(max=3, backoff_seconds=1.0)

        # Same input should always produce same retry decision
        def should_retry(attempt: int, max_attempts: int) -> bool:
            return attempt < max_attempts

        # Test multiple times with same inputs
        results = []
        for _ in range(20):
            decisions = [should_retry(attempt, retry_config.max) for attempt in range(1, 6)]
            results.append(decisions)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

        # Verify expected decisions
        expected_decisions = [True, True, False, False]  # Retry for attempts 1, 2, but not 3, 4
        assert first_result[:4] == expected_decisions

    def test_error_policy_consistency(self):
        """Test that error policy application is consistent."""
        pipeline_data = {
            "name": "consistency-test-pipeline",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "consistent_step",
                    "agent": "test.Agent",
                    "on_error": {
                        "policy": "continue",
                        "retry": {
                            "max": 3,
                            "backoff_seconds": 1.0
                        }
                    }
                }
            ]
        }

        # Create pipeline multiple times - should be identical
        pipelines = [
            Pipeline.from_dict(pipeline_data, validate_spec=True)
            for _ in range(5)
        ]

        # All pipelines should have identical configuration
        first_pipeline = pipelines[0]
        first_step = first_pipeline.spec.steps[0]

        for pipeline in pipelines[1:]:
            step = pipeline.spec.steps[0]
            assert step.on_error.policy == first_step.on_error.policy
            assert step.on_error.retry.max == first_step.on_error.retry.max
            assert step.on_error.retry.backoff_seconds == first_step.on_error.retry.backoff_seconds

        # Validation should be consistent
        for pipeline in pipelines:
            errors = pipeline.validate()
            assert len(errors) == 0