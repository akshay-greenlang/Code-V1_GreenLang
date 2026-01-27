# -*- coding: utf-8 -*-
"""
Unit Tests for GreenLang Orchestrator Error Taxonomy

Tests for GL-FOUND-X-001 error classification system.

Author: GreenLang Team
Date: 2026-01-27
"""

from datetime import datetime
from typing import Any, Dict

import pytest

from greenlang.orchestrator.errors.taxonomy import (
    # Enums
    ErrorClass,
    FixType,
    RetryPolicy,
    # Error codes
    ErrorCode,
    # Models
    SuggestedFix,
    OrchestrationError,
    ErrorMetadata,
    # Registry
    ErrorRegistry,
    # Formatters
    format_error_cli,
    format_error_json,
    format_error_markdown,
    # Factory functions
    create_error,
    create_validation_error,
    create_resource_error,
    create_policy_error,
    create_infrastructure_error,
    # Utilities
    get_error_class_for_http_status,
    get_error_class_for_k8s_exit_code,
    serialize_error_chain,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestErrorClass:
    """Tests for ErrorClass enum."""

    def test_all_error_classes_exist(self) -> None:
        """Verify all required error classes are defined."""
        assert ErrorClass.TRANSIENT == "TRANSIENT"
        assert ErrorClass.RESOURCE == "RESOURCE"
        assert ErrorClass.USER_CONFIG == "USER_CONFIG"
        assert ErrorClass.POLICY_DENIED == "POLICY_DENIED"
        assert ErrorClass.AGENT_BUG == "AGENT_BUG"
        assert ErrorClass.INFRASTRUCTURE == "INFRASTRUCTURE"

    def test_error_class_is_string_enum(self) -> None:
        """Verify ErrorClass inherits from str for JSON serialization."""
        assert isinstance(ErrorClass.TRANSIENT.value, str)
        # str(Enum) returns "EnumName.VALUE", but .value gives the string
        assert ErrorClass.TRANSIENT.value == "TRANSIENT"


class TestFixType:
    """Tests for FixType enum."""

    def test_all_fix_types_exist(self) -> None:
        """Verify all required fix types are defined."""
        assert FixType.RESOURCE_CHANGE == "RESOURCE_CHANGE"
        assert FixType.PARAM_CHANGE == "PARAM_CHANGE"
        assert FixType.POLICY_REQUEST == "POLICY_REQUEST"
        assert FixType.CODE_FIX == "CODE_FIX"
        assert FixType.RETRY == "RETRY"
        assert FixType.DATA_FIX == "DATA_FIX"
        assert FixType.INFRASTRUCTURE_FIX == "INFRASTRUCTURE_FIX"
        assert FixType.CONTACT_SUPPORT == "CONTACT_SUPPORT"


class TestRetryPolicy:
    """Tests for RetryPolicy enum."""

    def test_all_retry_policies_exist(self) -> None:
        """Verify all required retry policies are defined."""
        assert RetryPolicy.NO_RETRY == "NO_RETRY"
        assert RetryPolicy.IMMEDIATE_RETRY == "IMMEDIATE_RETRY"
        assert RetryPolicy.EXPONENTIAL_BACKOFF == "EXPONENTIAL_BACKOFF"
        assert RetryPolicy.BOUNDED_RETRY == "BOUNDED_RETRY"
        assert RetryPolicy.CONDITIONAL_RETRY == "CONDITIONAL_RETRY"


# =============================================================================
# ERROR CODE TESTS
# =============================================================================


class TestErrorCode:
    """Tests for ErrorCode constants."""

    def test_yaml_error_codes(self) -> None:
        """Verify YAML error codes."""
        assert ErrorCode.GL_E_YAML_INVALID == "GL-E-YAML-INVALID"
        assert ErrorCode.GL_E_YAML_SCHEMA == "GL-E-YAML-SCHEMA"
        assert ErrorCode.GL_E_YAML_VERSION == "GL-E-YAML-VERSION"

    def test_dag_error_codes(self) -> None:
        """Verify DAG error codes."""
        assert ErrorCode.GL_E_DAG_CYCLE == "GL-E-DAG-CYCLE"
        assert ErrorCode.GL_E_DAG_ORPHAN == "GL-E-DAG-ORPHAN"
        assert ErrorCode.GL_E_DAG_MISSING_DEP == "GL-E-DAG-MISSING-DEP"
        assert ErrorCode.GL_E_DAG_DUPLICATE_ID == "GL-E-DAG-DUPLICATE-ID"

    def test_agent_error_codes(self) -> None:
        """Verify agent error codes."""
        assert ErrorCode.GL_E_AGENT_NOT_FOUND == "GL-E-AGENT-NOT-FOUND"
        assert ErrorCode.GL_E_AGENT_VERSION == "GL-E-AGENT-VERSION"
        assert ErrorCode.GL_E_AGENT_CRASH == "GL-E-AGENT-CRASH"
        assert ErrorCode.GL_E_AGENT_TIMEOUT == "GL-E-AGENT-TIMEOUT"

    def test_param_error_codes(self) -> None:
        """Verify parameter error codes."""
        assert ErrorCode.GL_E_PARAM_MISSING == "GL-E-PARAM-MISSING"
        assert ErrorCode.GL_E_PARAM_TYPE == "GL-E-PARAM-TYPE"
        assert ErrorCode.GL_E_PARAM_RANGE == "GL-E-PARAM-RANGE"

    def test_k8s_error_codes(self) -> None:
        """Verify Kubernetes error codes."""
        assert ErrorCode.GL_E_K8S_JOB_OOM == "GL-E-K8S-JOB-OOM"
        assert ErrorCode.GL_E_K8S_JOB_TIMEOUT == "GL-E-K8S-JOB-TIMEOUT"
        assert ErrorCode.GL_E_K8S_IMAGEPULL == "GL-E-K8S-IMAGEPULL"
        assert ErrorCode.GL_E_K8S_EVICTION == "GL-E-K8S-EVICTION"
        assert ErrorCode.GL_E_K8S_QUOTA == "GL-E-K8S-QUOTA"

    def test_s3_error_codes(self) -> None:
        """Verify S3 error codes."""
        assert ErrorCode.GL_E_S3_ACCESS_DENIED == "GL-E-S3-ACCESS-DENIED"
        assert ErrorCode.GL_E_S3_NOT_FOUND == "GL-E-S3-NOT-FOUND"
        assert ErrorCode.GL_E_S3_TIMEOUT == "GL-E-S3-TIMEOUT"

    def test_policy_error_codes(self) -> None:
        """Verify policy error codes."""
        assert ErrorCode.GL_E_OPA_DENY == "GL-E-OPA-DENY"
        assert ErrorCode.GL_E_RBAC_DENIED == "GL-E-RBAC-DENIED"
        assert ErrorCode.GL_E_TENANT_ISOLATION == "GL-E-TENANT-ISOLATION"

    def test_quota_error_codes(self) -> None:
        """Verify quota error codes."""
        assert ErrorCode.GL_E_QUOTA_EXCEEDED == "GL-E-QUOTA-EXCEEDED"
        assert ErrorCode.GL_E_QUOTA_CPU == "GL-E-QUOTA-CPU"
        assert ErrorCode.GL_E_QUOTA_MEMORY == "GL-E-QUOTA-MEMORY"
        assert ErrorCode.GL_E_QUOTA_GPU == "GL-E-QUOTA-GPU"

    def test_all_codes_follow_pattern(self) -> None:
        """Verify all error codes follow GL-E-* pattern."""
        for attr_name in dir(ErrorCode):
            if attr_name.startswith("GL_E_"):
                code = getattr(ErrorCode, attr_name)
                assert code.startswith("GL-E-"), f"Code {code} does not start with GL-E-"


# =============================================================================
# PYDANTIC MODEL TESTS
# =============================================================================


class TestSuggestedFix:
    """Tests for SuggestedFix model."""

    def test_create_minimal_fix(self) -> None:
        """Create suggested fix with minimal fields."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            description="Increase memory limit",
        )
        assert fix.fix_type == FixType.RESOURCE_CHANGE
        assert fix.description == "Increase memory limit"
        assert fix.field is None
        assert fix.recommended_value is None

    def test_create_full_fix(self) -> None:
        """Create suggested fix with all fields."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            field="steps.calculate.resources.memory",
            recommended_value="4Gi",
            description="Increase memory limit",
            cli_command="greenlang run --memory 4Gi",
            doc_link="https://docs.greenlang.io/memory",
        )
        assert fix.field == "steps.calculate.resources.memory"
        assert fix.recommended_value == "4Gi"
        assert fix.cli_command == "greenlang run --memory 4Gi"
        assert fix.doc_link == "https://docs.greenlang.io/memory"

    def test_fix_is_immutable(self) -> None:
        """Verify SuggestedFix is immutable."""
        fix = SuggestedFix(
            fix_type=FixType.RETRY,
            description="Retry the operation",
        )
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
            fix.description = "New description"


class TestOrchestrationError:
    """Tests for OrchestrationError model."""

    def test_create_minimal_error(self) -> None:
        """Create error with minimal fields."""
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="Job killed due to OOM",
        )
        assert error.code == "GL-E-K8S-JOB-OOM"
        assert error.error_class == ErrorClass.RESOURCE
        assert error.message == "Job killed due to OOM"
        assert error.details == {}
        assert error.suggested_fixes == []
        assert error.links == {}

    def test_create_full_error(self) -> None:
        """Create error with all fields."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            description="Increase memory",
        )
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="Job killed due to OOM",
            details={"exit_code": 137, "memory_limit": "2Gi"},
            suggested_fixes=[fix],
            links={"logs": "https://logs.example.com"},
            retry_policy=RetryPolicy.BOUNDED_RETRY,
            run_id="run-123",
            step_id="calculate",
            agent_name="emissions_calculator",
            trace_id="trace-456",
            correlation_id="corr-789",
            cause="Memory usage exceeded limit",
        )
        assert error.details == {"exit_code": 137, "memory_limit": "2Gi"}
        assert len(error.suggested_fixes) == 1
        assert error.run_id == "run-123"
        assert error.step_id == "calculate"
        assert error.agent_name == "emissions_calculator"
        assert error.trace_id == "trace-456"
        assert error.correlation_id == "corr-789"
        assert error.cause == "Memory usage exceeded limit"

    def test_error_code_validation(self) -> None:
        """Verify error code must start with GL-E-."""
        with pytest.raises(ValueError):
            OrchestrationError(
                code="INVALID-CODE",
                error_class=ErrorClass.USER_CONFIG,
                message="Test",
            )

    def test_error_to_dict(self) -> None:
        """Test to_dict serialization."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Missing required parameter",
            run_id="run-123",
        )
        result = error.to_dict()
        assert result["code"] == "GL-E-PARAM-MISSING"
        assert result["error_class"] == "USER_CONFIG"
        assert result["message"] == "Missing required parameter"
        assert result["run_id"] == "run-123"

    def test_error_to_json(self) -> None:
        """Test to_json serialization."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Missing required parameter",
        )
        json_str = error.to_json()
        assert "GL-E-PARAM-MISSING" in json_str
        assert "USER_CONFIG" in json_str

    def test_error_hash_generation(self) -> None:
        """Test error hash generation for deduplication."""
        error1 = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM 1",
            step_id="step1",
            agent_name="agent1",
        )
        error2 = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM 2",  # Different message
            step_id="step1",
            agent_name="agent1",
        )
        error3 = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM 3",
            step_id="step2",  # Different step
            agent_name="agent1",
        )

        # Same code/step/agent should have same hash
        assert error1.get_error_hash() == error2.get_error_hash()
        # Different step should have different hash
        assert error1.get_error_hash() != error3.get_error_hash()

    def test_timestamp_auto_generated(self) -> None:
        """Verify timestamp is automatically generated."""
        before = datetime.utcnow()
        error = OrchestrationError(
            code="GL-E-INTERNAL",
            error_class=ErrorClass.AGENT_BUG,
            message="Test",
        )
        after = datetime.utcnow()

        assert before <= error.timestamp <= after


# =============================================================================
# ERROR REGISTRY TESTS
# =============================================================================


class TestErrorRegistry:
    """Tests for ErrorRegistry."""

    def test_registry_initializes(self) -> None:
        """Verify registry initializes successfully."""
        ErrorRegistry.initialize()
        assert ErrorRegistry._initialized is True

    def test_get_metadata_for_valid_code(self) -> None:
        """Get metadata for a valid error code."""
        metadata = ErrorRegistry.get_metadata(ErrorCode.GL_E_K8S_JOB_OOM)
        assert metadata is not None
        assert metadata.code == ErrorCode.GL_E_K8S_JOB_OOM
        assert metadata.error_class == ErrorClass.RESOURCE
        assert metadata.retry_policy == RetryPolicy.BOUNDED_RETRY

    def test_get_metadata_for_invalid_code(self) -> None:
        """Get metadata for an invalid error code returns None."""
        metadata = ErrorRegistry.get_metadata("GL-E-NONEXISTENT")
        assert metadata is None

    def test_get_error_class(self) -> None:
        """Test get_error_class method."""
        assert ErrorRegistry.get_error_class(ErrorCode.GL_E_K8S_JOB_OOM) == ErrorClass.RESOURCE
        assert ErrorRegistry.get_error_class(ErrorCode.GL_E_OPA_DENY) == ErrorClass.POLICY_DENIED
        assert ErrorRegistry.get_error_class(ErrorCode.GL_E_API_5XX) == ErrorClass.TRANSIENT
        assert ErrorRegistry.get_error_class(ErrorCode.GL_E_PARAM_MISSING) == ErrorClass.USER_CONFIG
        assert ErrorRegistry.get_error_class(ErrorCode.GL_E_AGENT_CRASH) == ErrorClass.AGENT_BUG

    def test_get_retry_policy(self) -> None:
        """Test get_retry_policy method."""
        assert ErrorRegistry.get_retry_policy(ErrorCode.GL_E_K8S_JOB_OOM) == RetryPolicy.BOUNDED_RETRY
        assert ErrorRegistry.get_retry_policy(ErrorCode.GL_E_PARAM_MISSING) == RetryPolicy.NO_RETRY
        assert ErrorRegistry.get_retry_policy(ErrorCode.GL_E_API_5XX) == RetryPolicy.EXPONENTIAL_BACKOFF

    def test_get_default_fixes(self) -> None:
        """Test get_default_fixes method."""
        fixes = ErrorRegistry.get_default_fixes(ErrorCode.GL_E_K8S_JOB_OOM)
        assert len(fixes) > 0
        assert any(fix.fix_type == FixType.RESOURCE_CHANGE for fix in fixes)

    def test_get_http_error_class(self) -> None:
        """Test HTTP status code to error class mapping."""
        # Transient
        assert ErrorRegistry.get_http_error_class(500) == ErrorClass.TRANSIENT
        assert ErrorRegistry.get_http_error_class(502) == ErrorClass.TRANSIENT
        assert ErrorRegistry.get_http_error_class(503) == ErrorClass.TRANSIENT
        assert ErrorRegistry.get_http_error_class(429) == ErrorClass.TRANSIENT

        # User config
        assert ErrorRegistry.get_http_error_class(400) == ErrorClass.USER_CONFIG
        assert ErrorRegistry.get_http_error_class(404) == ErrorClass.USER_CONFIG
        assert ErrorRegistry.get_http_error_class(422) == ErrorClass.USER_CONFIG

        # Policy denied
        assert ErrorRegistry.get_http_error_class(401) == ErrorClass.POLICY_DENIED
        assert ErrorRegistry.get_http_error_class(403) == ErrorClass.POLICY_DENIED

    def test_get_k8s_error_code(self) -> None:
        """Test K8s exit code to error code mapping."""
        assert ErrorRegistry.get_k8s_error_code(137) == ErrorCode.GL_E_K8S_JOB_OOM
        assert ErrorRegistry.get_k8s_error_code(143) == ErrorCode.GL_E_K8S_JOB_TIMEOUT
        assert ErrorRegistry.get_k8s_error_code(1) == ErrorCode.GL_E_AGENT_CRASH
        assert ErrorRegistry.get_k8s_error_code(0) is None  # Success

    def test_list_codes(self) -> None:
        """Test list_codes method."""
        all_codes = ErrorRegistry.list_codes()
        assert len(all_codes) > 0
        assert ErrorCode.GL_E_K8S_JOB_OOM in all_codes

        # Filter by category
        k8s_codes = ErrorRegistry.list_codes(category="kubernetes")
        assert ErrorCode.GL_E_K8S_JOB_OOM in k8s_codes
        assert ErrorCode.GL_E_PARAM_MISSING not in k8s_codes

    def test_list_categories(self) -> None:
        """Test list_categories method."""
        categories = ErrorRegistry.list_categories()
        assert "kubernetes" in categories
        assert "parameter" in categories
        assert "policy" in categories
        assert "storage" in categories


# =============================================================================
# FORMATTER TESTS
# =============================================================================


class TestFormatErrorCli:
    """Tests for CLI formatting."""

    def test_format_basic_error(self) -> None:
        """Format a basic error for CLI."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Missing required parameter 'dataset'",
        )
        output = format_error_cli(error, color=False)
        assert "GL-E-PARAM-MISSING" in output
        assert "USER_CONFIG" in output
        assert "Missing required parameter 'dataset'" in output

    def test_format_error_with_details(self) -> None:
        """Format an error with details."""
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="Job killed due to OOM",
            details={"exit_code": 137, "memory_limit": "2Gi"},
        )
        output = format_error_cli(error, color=False)
        assert "Details:" in output
        assert "exit_code: 137" in output
        assert "memory_limit: 2Gi" in output

    def test_format_error_with_fixes(self) -> None:
        """Format an error with suggested fixes."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            field="resources.memory",
            recommended_value="4Gi",
            description="Increase memory limit",
            cli_command="greenlang run --memory 4Gi",
        )
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM",
            suggested_fixes=[fix],
        )
        output = format_error_cli(error, color=False)
        assert "Suggested Fixes:" in output
        assert "RESOURCE_CHANGE" in output
        assert "Increase memory limit" in output
        assert "greenlang run --memory 4Gi" in output

    def test_format_error_with_context(self) -> None:
        """Format an error with run context."""
        error = OrchestrationError(
            code="GL-E-AGENT-CRASH",
            error_class=ErrorClass.AGENT_BUG,
            message="Agent crashed",
            run_id="run-123",
            step_id="calculate",
            agent_name="emissions_calc",
        )
        output = format_error_cli(error, color=False)
        assert "Run: run-123" in output
        assert "Step: calculate" in output
        assert "Agent: emissions_calc" in output

    def test_format_error_with_color(self) -> None:
        """Format an error with color codes."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Test",
        )
        output = format_error_cli(error, color=True)
        # Should contain ANSI escape codes
        assert "\033[" in output


class TestFormatErrorJson:
    """Tests for JSON formatting."""

    def test_format_basic_error_json(self) -> None:
        """Format a basic error as JSON."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Missing parameter",
        )
        result = format_error_json(error)

        assert result["error"]["code"] == "GL-E-PARAM-MISSING"
        assert result["error"]["class"] == "USER_CONFIG"
        assert result["error"]["message"] == "Missing parameter"
        assert "timestamp" in result["error"]

    def test_format_error_json_with_context(self) -> None:
        """Format an error with context as JSON."""
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM",
            run_id="run-123",
            step_id="step-1",
            agent_name="agent-1",
            trace_id="trace-456",
        )
        result = format_error_json(error)

        assert result["context"]["run_id"] == "run-123"
        assert result["context"]["step_id"] == "step-1"
        assert result["context"]["agent_name"] == "agent-1"
        assert result["context"]["trace_id"] == "trace-456"

    def test_format_error_json_with_fixes(self) -> None:
        """Format an error with fixes as JSON."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            field="memory",
            recommended_value="4Gi",
            description="Increase memory",
        )
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM",
            suggested_fixes=[fix],
        )
        result = format_error_json(error)

        assert len(result["suggested_fixes"]) == 1
        assert result["suggested_fixes"][0]["type"] == "RESOURCE_CHANGE"
        assert result["suggested_fixes"][0]["field"] == "memory"
        assert result["suggested_fixes"][0]["recommended_value"] == "4Gi"

    def test_format_error_json_excludes_none_context(self) -> None:
        """Verify None values are excluded from context."""
        error = OrchestrationError(
            code="GL-E-INTERNAL",
            error_class=ErrorClass.AGENT_BUG,
            message="Test",
            run_id="run-123",
            # step_id, agent_name, etc. are None
        )
        result = format_error_json(error)

        assert "run_id" in result["context"]
        assert "step_id" not in result["context"]
        assert "agent_name" not in result["context"]

    def test_format_error_json_with_stack_trace(self) -> None:
        """Format error with stack trace included."""
        error = OrchestrationError(
            code="GL-E-INTERNAL",
            error_class=ErrorClass.AGENT_BUG,
            message="Test",
            stack_trace="Traceback...",
        )
        result = format_error_json(error, include_stack_trace=True)
        assert "debug" in result
        assert result["debug"]["stack_trace"] == "Traceback..."


class TestFormatErrorMarkdown:
    """Tests for Markdown formatting."""

    def test_format_basic_error_markdown(self) -> None:
        """Format a basic error as Markdown."""
        error = OrchestrationError(
            code="GL-E-PARAM-MISSING",
            error_class=ErrorClass.USER_CONFIG,
            message="Missing parameter 'dataset'",
        )
        output = format_error_markdown(error)

        assert "## Error: GL-E-PARAM-MISSING" in output
        assert "**Class:** `USER_CONFIG`" in output
        assert "> Missing parameter 'dataset'" in output

    def test_format_error_markdown_with_details(self) -> None:
        """Format error with details as Markdown."""
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM",
            details={"memory_limit": "2Gi", "exit_code": 137},
        )
        output = format_error_markdown(error)

        assert "### Details" in output
        assert "| Key | Value |" in output
        assert "`memory_limit`" in output
        assert "`2Gi`" in output

    def test_format_error_markdown_with_fixes(self) -> None:
        """Format error with fixes as Markdown."""
        fix = SuggestedFix(
            fix_type=FixType.RESOURCE_CHANGE,
            description="Increase memory",
            doc_link="https://docs.example.com",
        )
        error = OrchestrationError(
            code="GL-E-K8S-JOB-OOM",
            error_class=ErrorClass.RESOURCE,
            message="OOM",
            suggested_fixes=[fix],
        )
        output = format_error_markdown(error)

        assert "### Suggested Fixes" in output
        assert "#### 1. RESOURCE_CHANGE" in output
        assert "Increase memory" in output
        assert "[https://docs.example.com]" in output


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateError:
    """Tests for create_error factory function."""

    def test_create_error_basic(self) -> None:
        """Create error from code with defaults."""
        error = create_error(ErrorCode.GL_E_K8S_JOB_OOM, memory_limit="2Gi")

        assert error.code == ErrorCode.GL_E_K8S_JOB_OOM
        assert error.error_class == ErrorClass.RESOURCE
        assert error.retry_policy == RetryPolicy.BOUNDED_RETRY
        assert len(error.suggested_fixes) > 0

    def test_create_error_with_custom_message(self) -> None:
        """Create error with custom message."""
        error = create_error(
            ErrorCode.GL_E_PARAM_MISSING,
            message="Custom message here",
        )
        assert error.message == "Custom message here"

    def test_create_error_with_context(self) -> None:
        """Create error with run context."""
        error = create_error(
            ErrorCode.GL_E_AGENT_CRASH,
            run_id="run-123",
            step_id="step-1",
            agent_name="test-agent",
            error="NullPointerException",
        )
        assert error.run_id == "run-123"
        assert error.step_id == "step-1"
        assert error.agent_name == "test-agent"

    def test_create_error_with_custom_fixes(self) -> None:
        """Create error with custom suggested fixes."""
        custom_fix = SuggestedFix(
            fix_type=FixType.CONTACT_SUPPORT,
            description="Call support hotline",
        )
        error = create_error(
            ErrorCode.GL_E_INTERNAL,
            suggested_fixes=[custom_fix],
            error="Unknown",
        )
        assert len(error.suggested_fixes) == 1
        assert error.suggested_fixes[0].fix_type == FixType.CONTACT_SUPPORT

    def test_create_error_with_links(self) -> None:
        """Create error with links."""
        error = create_error(
            ErrorCode.GL_E_K8S_JOB_OOM,
            links={"logs": "https://logs.example.com"},
            memory_limit="2Gi",
        )
        assert error.links["logs"] == "https://logs.example.com"


class TestCreateValidationError:
    """Tests for create_validation_error factory."""

    def test_create_yaml_syntax_error(self) -> None:
        """Create YAML syntax error."""
        error = create_validation_error("Invalid YAML syntax at line 10")
        assert error.code == ErrorCode.GL_E_YAML_INVALID
        assert error.error_class == ErrorClass.USER_CONFIG

    def test_create_schema_error(self) -> None:
        """Create schema validation error."""
        error = create_validation_error("Schema validation failed")
        assert error.code == ErrorCode.GL_E_YAML_SCHEMA

    def test_create_param_error(self) -> None:
        """Create parameter error."""
        error = create_validation_error(
            "Required field missing",
            step_id="step-1",
            field="params.dataset",
        )
        assert error.code == ErrorCode.GL_E_PARAM_MISSING
        assert error.step_id == "step-1"
        assert error.details["field"] == "params.dataset"


class TestCreateResourceError:
    """Tests for create_resource_error factory."""

    def test_create_oom_error(self) -> None:
        """Create OOM error."""
        error = create_resource_error("memory", used="2.1Gi", limit="2Gi")
        assert error.code == ErrorCode.GL_E_K8S_JOB_OOM
        assert error.error_class == ErrorClass.RESOURCE
        assert error.details["used"] == "2.1Gi"
        assert error.details["limit"] == "2Gi"

    def test_create_timeout_error(self) -> None:
        """Create timeout error."""
        error = create_resource_error("timeout", step_id="slow-step")
        assert error.code == ErrorCode.GL_E_K8S_JOB_TIMEOUT
        assert error.step_id == "slow-step"

    def test_create_cpu_quota_error(self) -> None:
        """Create CPU quota error."""
        error = create_resource_error("cpu", used="8", limit="4")
        assert error.code == ErrorCode.GL_E_QUOTA_CPU


class TestCreatePolicyError:
    """Tests for create_policy_error factory."""

    def test_create_opa_error(self) -> None:
        """Create OPA policy error."""
        error = create_policy_error("opa.data_access", "Unauthorized data access")
        assert error.code == ErrorCode.GL_E_OPA_DENY
        assert error.error_class == ErrorClass.POLICY_DENIED

    def test_create_rbac_error(self) -> None:
        """Create RBAC error."""
        error = create_policy_error("rbac.admin", "Admin role required")
        assert error.code == ErrorCode.GL_E_RBAC_DENIED

    def test_create_tenant_isolation_error(self) -> None:
        """Create tenant isolation error."""
        error = create_policy_error("tenant.boundary", "Cross-tenant access denied")
        assert error.code == ErrorCode.GL_E_TENANT_ISOLATION


class TestCreateInfrastructureError:
    """Tests for create_infrastructure_error factory."""

    def test_create_k8s_error(self) -> None:
        """Create K8s infrastructure error."""
        error = create_infrastructure_error("kubernetes", "API server unavailable")
        assert error.code == ErrorCode.GL_E_K8S_API
        assert error.error_class == ErrorClass.INFRASTRUCTURE

    def test_create_s3_error(self) -> None:
        """Create S3 infrastructure error."""
        error = create_infrastructure_error("s3", "Connection timeout")
        assert error.code == ErrorCode.GL_E_S3_TIMEOUT

    def test_create_db_error(self) -> None:
        """Create database infrastructure error."""
        error = create_infrastructure_error("database", "Connection refused")
        assert error.code == ErrorCode.GL_E_DB_CONNECTION


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestGetErrorClassForHttpStatus:
    """Tests for get_error_class_for_http_status utility."""

    def test_transient_status_codes(self) -> None:
        """Test transient HTTP status codes."""
        assert get_error_class_for_http_status(500) == ErrorClass.TRANSIENT
        assert get_error_class_for_http_status(502) == ErrorClass.TRANSIENT
        assert get_error_class_for_http_status(503) == ErrorClass.TRANSIENT
        assert get_error_class_for_http_status(504) == ErrorClass.TRANSIENT
        assert get_error_class_for_http_status(429) == ErrorClass.TRANSIENT

    def test_user_config_status_codes(self) -> None:
        """Test user config HTTP status codes."""
        assert get_error_class_for_http_status(400) == ErrorClass.USER_CONFIG
        assert get_error_class_for_http_status(404) == ErrorClass.USER_CONFIG
        assert get_error_class_for_http_status(422) == ErrorClass.USER_CONFIG

    def test_policy_denied_status_codes(self) -> None:
        """Test policy denied HTTP status codes."""
        assert get_error_class_for_http_status(401) == ErrorClass.POLICY_DENIED
        assert get_error_class_for_http_status(403) == ErrorClass.POLICY_DENIED

    def test_unknown_status_code(self) -> None:
        """Test unknown HTTP status code defaults to AGENT_BUG."""
        assert get_error_class_for_http_status(999) == ErrorClass.AGENT_BUG


class TestGetErrorClassForK8sExitCode:
    """Tests for get_error_class_for_k8s_exit_code utility."""

    def test_oom_exit_code(self) -> None:
        """Test OOM exit code (137 = SIGKILL)."""
        code, error_class = get_error_class_for_k8s_exit_code(137)
        assert code == ErrorCode.GL_E_K8S_JOB_OOM
        assert error_class == ErrorClass.RESOURCE

    def test_timeout_exit_code(self) -> None:
        """Test timeout exit code (143 = SIGTERM)."""
        code, error_class = get_error_class_for_k8s_exit_code(143)
        assert code == ErrorCode.GL_E_K8S_JOB_TIMEOUT
        assert error_class == ErrorClass.RESOURCE

    def test_crash_exit_code(self) -> None:
        """Test crash exit code."""
        code, error_class = get_error_class_for_k8s_exit_code(1)
        assert code == ErrorCode.GL_E_AGENT_CRASH
        assert error_class == ErrorClass.AGENT_BUG

    def test_success_exit_code(self) -> None:
        """Test success exit code returns None."""
        code, error_class = get_error_class_for_k8s_exit_code(0)
        assert code is None
        assert error_class == ErrorClass.AGENT_BUG  # Default for unknown


class TestSerializeErrorChain:
    """Tests for serialize_error_chain utility."""

    def test_serialize_empty_chain(self) -> None:
        """Serialize empty error chain."""
        result = serialize_error_chain([])
        assert result["errors"] == []
        assert result["count"] == 0

    def test_serialize_single_error(self) -> None:
        """Serialize single error."""
        error = OrchestrationError(
            code="GL-E-INTERNAL",
            error_class=ErrorClass.AGENT_BUG,
            message="Root cause",
            step_id="step-1",
        )
        result = serialize_error_chain([error])

        assert result["count"] == 1
        assert result["chain"][0]["is_root_cause"] is True
        assert result["chain"][0]["code"] == "GL-E-INTERNAL"

    def test_serialize_error_chain(self) -> None:
        """Serialize chain of related errors."""
        root = OrchestrationError(
            code="GL-E-DB-CONNECTION",
            error_class=ErrorClass.INFRASTRUCTURE,
            message="Database connection failed",
            step_id="step-1",
        )
        effect = OrchestrationError(
            code="GL-E-AGENT-CRASH",
            error_class=ErrorClass.AGENT_BUG,
            message="Agent crashed due to DB failure",
            step_id="step-1",
        )
        result = serialize_error_chain([root, effect])

        assert result["count"] == 2
        assert result["chain"][0]["is_root_cause"] is True
        assert result["chain"][1]["is_root_cause"] is False
        assert "root_cause" in result
        assert "GL-E-DB-CONNECTION" in result["summary"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestErrorTaxonomyIntegration:
    """Integration tests for the error taxonomy system."""

    def test_full_error_workflow(self) -> None:
        """Test complete error creation and formatting workflow."""
        # Create error from code
        error = create_error(
            ErrorCode.GL_E_K8S_JOB_OOM,
            run_id="run-integration-test",
            step_id="calculate_emissions",
            agent_name="emissions_calculator",
            links={
                "logs": "https://logs.example.com/run-integration-test",
                "trace": "https://trace.example.com/run-integration-test",
            },
            memory_limit="2Gi",
        )

        # Verify error properties
        assert error.code == ErrorCode.GL_E_K8S_JOB_OOM
        assert error.error_class == ErrorClass.RESOURCE
        assert error.retry_policy == RetryPolicy.BOUNDED_RETRY
        assert len(error.suggested_fixes) > 0

        # Format for CLI
        cli_output = format_error_cli(error, color=False)
        assert "GL-E-K8S-JOB-OOM" in cli_output
        assert "run-integration-test" in cli_output
        assert "calculate_emissions" in cli_output

        # Format for JSON (CI/CD)
        json_output = format_error_json(error)
        assert json_output["error"]["code"] == "GL-E-K8S-JOB-OOM"
        assert json_output["context"]["run_id"] == "run-integration-test"

        # Format for Markdown (reports)
        md_output = format_error_markdown(error)
        assert "## Error: GL-E-K8S-JOB-OOM" in md_output

    def test_error_deduplication_by_hash(self) -> None:
        """Test that similar errors can be deduplicated."""
        errors = [
            create_error(
                ErrorCode.GL_E_K8S_JOB_OOM,
                step_id="step-1",
                agent_name="agent-1",
                memory_limit="2Gi",
            )
            for _ in range(5)
        ]

        # All should have the same hash
        hashes = set(e.get_error_hash() for e in errors)
        assert len(hashes) == 1

    def test_all_error_codes_have_metadata(self) -> None:
        """Verify all defined error codes have registry metadata."""
        for attr_name in dir(ErrorCode):
            if attr_name.startswith("GL_E_"):
                code = getattr(ErrorCode, attr_name)
                metadata = ErrorRegistry.get_metadata(code)
                assert metadata is not None, f"No metadata for {code}"
                assert metadata.error_class is not None
                assert metadata.retry_policy is not None
                assert metadata.default_message is not None

    def test_error_class_determines_retry_behavior(self) -> None:
        """Verify error classes have appropriate retry policies."""
        # Transient errors should be retryable
        transient_codes = [
            ErrorCode.GL_E_NETWORK_TIMEOUT,
            ErrorCode.GL_E_API_5XX,
            ErrorCode.GL_E_S3_TIMEOUT,
        ]
        for code in transient_codes:
            metadata = ErrorRegistry.get_metadata(code)
            assert metadata is not None
            assert metadata.retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF

        # User config errors should not be retryable
        user_config_codes = [
            ErrorCode.GL_E_PARAM_MISSING,
            ErrorCode.GL_E_YAML_INVALID,
            ErrorCode.GL_E_DAG_CYCLE,
        ]
        for code in user_config_codes:
            metadata = ErrorRegistry.get_metadata(code)
            assert metadata is not None
            assert metadata.retry_policy == RetryPolicy.NO_RETRY

        # Policy errors should not be retryable
        policy_codes = [
            ErrorCode.GL_E_OPA_DENY,
            ErrorCode.GL_E_RBAC_DENIED,
        ]
        for code in policy_codes:
            metadata = ErrorRegistry.get_metadata(code)
            assert metadata is not None
            assert metadata.retry_policy == RetryPolicy.NO_RETRY
