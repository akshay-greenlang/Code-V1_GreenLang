"""Tests for GreenLang Exception Hierarchy.

Comprehensive test suite covering:
- Base exception functionality
- AgentException hierarchy
- WorkflowException hierarchy
- DataException hierarchy
- Rich error context
- Exception serialization
- Exception utilities
- Edge cases

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

import pytest
import json
from datetime import datetime

from greenlang.exceptions import (
    # Base
    GreenLangException,
    # Agent exceptions
    AgentException,
    ValidationError,
    ExecutionError,
    TimeoutError,
    ConfigurationError,
    # Workflow exceptions
    WorkflowException,
    DAGError,
    PolicyViolation,
    ResourceError,
    OrchestrationError,
    # Data exceptions
    DataException,
    InvalidSchema,
    MissingData,
    CorruptedData,
    DataAccessError,
    # Utilities
    format_exception_chain,
    is_retriable,
)


# ==============================================================================
# Base Exception Tests
# ==============================================================================

class TestGreenLangException:
    """Tests for base GreenLangException."""

    def test_create_basic_exception(self):
        """Can create basic exception with message."""
        exc = GreenLangException("Something went wrong")

        assert exc.message == "Something went wrong"
        assert exc.error_code.startswith("GL_")
        assert exc.agent_name is None
        assert exc.context == {}
        assert isinstance(exc.timestamp, datetime)

    def test_create_exception_with_context(self):
        """Can create exception with rich context."""
        exc = GreenLangException(
            message="Test error",
            error_code="GL_TEST_001",
            agent_name="TestAgent",
            context={"key": "value", "count": 42}
        )

        assert exc.message == "Test error"
        assert exc.error_code == "GL_TEST_001"
        assert exc.agent_name == "TestAgent"
        assert exc.context == {"key": "value", "count": 42}

    def test_exception_str_representation(self):
        """Exception string includes error code and message."""
        exc = GreenLangException(
            message="Test error",
            error_code="GL_TEST_001",
            agent_name="TestAgent"
        )

        str_repr = str(exc)
        assert "GL_TEST_001" in str_repr
        assert "TestAgent" in str_repr
        assert "Test error" in str_repr

    def test_exception_to_dict(self):
        """Exception can be converted to dictionary."""
        exc = GreenLangException(
            message="Test error",
            agent_name="TestAgent",
            context={"key": "value"}
        )

        exc_dict = exc.to_dict()

        assert exc_dict["error_type"] == "GreenLangException"
        assert exc_dict["message"] == "Test error"
        assert exc_dict["agent_name"] == "TestAgent"
        assert exc_dict["context"] == {"key": "value"}
        assert "timestamp" in exc_dict
        assert "traceback" in exc_dict

    def test_exception_to_json(self):
        """Exception can be serialized to JSON."""
        exc = GreenLangException(
            message="Test error",
            context={"key": "value"}
        )

        json_str = exc.to_json()
        parsed = json.loads(json_str)

        assert parsed["message"] == "Test error"
        assert parsed["context"] == {"key": "value"}

    def test_auto_generated_error_code(self):
        """Error code is auto-generated from class name."""
        exc = GreenLangException("Test error")

        assert exc.error_code == "GL_GREEN_LANG_EXCEPTION"


# ==============================================================================
# Agent Exception Tests
# ==============================================================================

class TestAgentExceptions:
    """Tests for AgentException hierarchy."""

    def test_validation_error_basic(self):
        """ValidationError can be created."""
        exc = ValidationError(
            message="Invalid input",
            agent_name="FuelAgent",
            context={"field": "fuel_type"}
        )

        assert isinstance(exc, AgentException)
        assert isinstance(exc, GreenLangException)
        assert exc.message == "Invalid input"
        assert exc.agent_name == "FuelAgent"
        assert exc.error_code.startswith("GL_AGENT_")

    def test_validation_error_with_invalid_fields(self):
        """ValidationError tracks invalid fields."""
        exc = ValidationError(
            message="Validation failed",
            agent_name="FuelAgent",
            invalid_fields={
                "fuel_type": "Must be one of: natural_gas, coal, diesel",
                "amount": "Must be positive number"
            }
        )

        assert "invalid_fields" in exc.context
        assert exc.context["invalid_fields"]["fuel_type"] == "Must be one of: natural_gas, coal, diesel"

    def test_execution_error_with_cause(self):
        """ExecutionError can wrap another exception."""
        cause = ValueError("Division by zero")
        exc = ExecutionError(
            message="Calculation failed",
            agent_name="FuelAgent",
            step="calculate_emissions",
            cause=cause
        )

        assert exc.context["step"] == "calculate_emissions"
        assert exc.context["cause"] == "Division by zero"
        assert exc.context["cause_type"] == "ValueError"

    def test_timeout_error_with_metrics(self):
        """TimeoutError tracks timeout metrics."""
        exc = TimeoutError(
            message="Execution timed out",
            agent_name="FuelAgent",
            timeout_seconds=30.0,
            elapsed_seconds=35.5
        )

        assert exc.context["timeout_seconds"] == 30.0
        assert exc.context["elapsed_seconds"] == 35.5

    def test_configuration_error(self):
        """ConfigurationError can be raised."""
        exc = ConfigurationError(
            message="Missing API key",
            agent_name="FuelAgent",
            context={"config_key": "OPENAI_API_KEY"}
        )

        assert isinstance(exc, AgentException)
        assert exc.message == "Missing API key"


# ==============================================================================
# Workflow Exception Tests
# ==============================================================================

class TestWorkflowExceptions:
    """Tests for WorkflowException hierarchy."""

    def test_dag_error_with_cycle(self):
        """DAGError can track cycle information."""
        exc = DAGError(
            message="Cycle detected in workflow",
            workflow_id="wf_123",
            invalid_nodes=["step1", "step2", "step3", "step1"],
            context={"cycle": ["step1", "step2", "step3", "step1"]}
        )

        assert isinstance(exc, WorkflowException)
        assert exc.context["workflow_id"] == "wf_123"
        assert exc.context["invalid_nodes"] == ["step1", "step2", "step3", "step1"]

    def test_policy_violation(self):
        """PolicyViolation tracks violated policy."""
        exc = PolicyViolation(
            message="Execution time limit exceeded",
            policy_name="max_execution_time",
            violation_details={"limit": 300, "actual": 350}
        )

        assert exc.context["policy_name"] == "max_execution_time"
        assert exc.context["violation_details"]["limit"] == 300

    def test_resource_error_with_metrics(self):
        """ResourceError tracks resource usage."""
        exc = ResourceError(
            message="Memory limit exceeded",
            resource_type="memory",
            limit=1024.0,
            used=1500.0
        )

        assert exc.context["resource_type"] == "memory"
        assert exc.context["limit"] == 1024.0
        assert exc.context["used"] == 1500.0

    def test_orchestration_error(self):
        """OrchestrationError can be raised."""
        exc = OrchestrationError(
            message="Failed to coordinate agents",
            context={"failed_agents": ["agent1", "agent2"]}
        )

        assert isinstance(exc, WorkflowException)


# ==============================================================================
# Data Exception Tests
# ==============================================================================

class TestDataExceptions:
    """Tests for DataException hierarchy."""

    def test_invalid_schema_with_details(self):
        """InvalidSchema tracks schema errors."""
        exc = InvalidSchema(
            message="Schema validation failed",
            expected_schema={"fuel_type": "string", "amount": "number"},
            actual_data={"fuel_type": 123, "amount": "invalid"},
            schema_errors=["fuel_type: expected string, got int"]
        )

        assert isinstance(exc, DataException)
        assert "expected_schema" in exc.context
        assert "actual_data" in exc.context
        assert "schema_errors" in exc.context

    def test_missing_data(self):
        """MissingData tracks missing fields."""
        exc = MissingData(
            message="Required data not found",
            data_type="emission_factor",
            missing_fields=["fuel_type", "country"]
        )

        assert exc.context["data_type"] == "emission_factor"
        assert exc.context["missing_fields"] == ["fuel_type", "country"]

    def test_corrupted_data(self):
        """CorruptedData tracks corruption details."""
        exc = CorruptedData(
            message="Checksum verification failed",
            data_source="/data/emissions.json",
            corruption_details={
                "expected_checksum": "abc123",
                "actual_checksum": "def456"
            }
        )

        assert exc.context["data_source"] == "/data/emissions.json"
        assert "corruption_details" in exc.context

    def test_data_access_error_with_cause(self):
        """DataAccessError wraps original exception."""
        cause = ConnectionError("Database unreachable")
        exc = DataAccessError(
            message="Failed to access database",
            data_source="emissions_db",
            operation="SELECT",
            cause=cause
        )

        assert exc.context["data_source"] == "emissions_db"
        assert exc.context["operation"] == "SELECT"
        assert exc.context["cause"] == "Database unreachable"
        assert exc.context["cause_type"] == "ConnectionError"


# ==============================================================================
# Exception Utilities Tests
# ==============================================================================

class TestExceptionUtilities:
    """Tests for exception utility functions."""

    def test_format_exception_chain_single(self):
        """Can format single exception."""
        exc = ValidationError(
            message="Validation failed",
            agent_name="FuelAgent",
            context={"field": "fuel_type"}
        )

        formatted = format_exception_chain(exc)

        assert "Validation failed" in formatted
        assert "FuelAgent" in formatted
        assert "field" in formatted

    def test_format_exception_chain_with_cause(self):
        """Can format exception chain with cause."""
        cause = ValueError("Invalid value")
        exc = ExecutionError(
            message="Execution failed",
            agent_name="FuelAgent",
            cause=cause
        )

        formatted = format_exception_chain(exc)

        assert "Execution failed" in formatted
        assert "ValueError" in formatted

    def test_is_retriable_for_timeout(self):
        """Timeout errors are retriable."""
        exc = TimeoutError(
            message="Timed out",
            timeout_seconds=30.0
        )

        assert is_retriable(exc) is True

    def test_is_retriable_for_resource_error(self):
        """Resource errors are retriable."""
        exc = ResourceError(
            message="Out of memory",
            resource_type="memory"
        )

        assert is_retriable(exc) is True

    def test_is_not_retriable_for_validation(self):
        """Validation errors are not retriable."""
        exc = ValidationError(
            message="Invalid input"
        )

        assert is_retriable(exc) is False

    def test_is_not_retriable_for_policy_violation(self):
        """Policy violations are not retriable."""
        exc = PolicyViolation(
            message="Policy violated"
        )

        assert is_retriable(exc) is False

    def test_is_not_retriable_for_dag_error(self):
        """DAG errors are not retriable."""
        exc = DAGError(
            message="Cycle detected"
        )

        assert is_retriable(exc) is False


# ==============================================================================
# Edge Cases and Integration Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_exception_can_be_raised_and_caught(self):
        """Exceptions can be raised and caught normally."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                message="Test error",
                agent_name="TestAgent"
            )

        exc = exc_info.value
        assert exc.message == "Test error"
        assert exc.agent_name == "TestAgent"

    def test_exception_inheritance_chain(self):
        """Exception inheritance chain is correct."""
        exc = ValidationError("Test")

        assert isinstance(exc, ValidationError)
        assert isinstance(exc, AgentException)
        assert isinstance(exc, GreenLangException)
        assert isinstance(exc, Exception)

    def test_multiple_exceptions_have_unique_codes(self):
        """Different exception types have different error codes."""
        validation_exc = ValidationError("Test")
        execution_exc = ExecutionError("Test")
        timeout_exc = TimeoutError("Test")

        assert validation_exc.error_code != execution_exc.error_code
        assert execution_exc.error_code != timeout_exc.error_code

    def test_exception_context_can_be_complex(self):
        """Exception context can contain complex nested data."""
        exc = ValidationError(
            message="Complex validation failed",
            context={
                "input": {
                    "nested": {
                        "data": [1, 2, 3],
                        "dict": {"a": "b"}
                    }
                },
                "errors": [
                    {"field": "nested.data", "error": "Too many items"},
                    {"field": "nested.dict.a", "error": "Invalid value"}
                ]
            }
        )

        assert len(exc.context["errors"]) == 2
        assert exc.context["input"]["nested"]["data"] == [1, 2, 3]

    def test_exception_serialization_roundtrip(self):
        """Exception can be serialized and deserialized."""
        original = ValidationError(
            message="Test error",
            agent_name="TestAgent",
            context={"field": "value"}
        )

        # Serialize to JSON
        json_str = original.to_json()

        # Deserialize
        parsed = json.loads(json_str)

        # Verify
        assert parsed["message"] == "Test error"
        assert parsed["agent_name"] == "TestAgent"
        assert parsed["context"]["field"] == "value"
        assert parsed["error_type"] == "ValidationError"

    def test_exception_with_none_context_fields(self):
        """Exception handles None values in context gracefully."""
        exc = ExecutionError(
            message="Test error",
            agent_name=None,
            context=None,
            step=None,
            cause=None
        )

        assert exc.agent_name is None
        assert exc.context == {}

    def test_timestamp_is_recent(self):
        """Exception timestamp is recent."""
        before = datetime.now()
        exc = GreenLangException("Test")
        after = datetime.now()

        assert before <= exc.timestamp <= after
