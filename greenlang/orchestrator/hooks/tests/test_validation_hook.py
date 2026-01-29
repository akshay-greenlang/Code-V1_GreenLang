# -*- coding: utf-8 -*-
"""
Unit Tests for Pre-Run Validation Hook
=======================================

GL-FOUND-X-002: Schema Compiler & Validator - Task 6.3

Comprehensive tests for the PreRunValidationHook module, including:
- InputSchemaSpec model validation
- PipelineValidationConfig model validation
- PreRunValidationHook.validate() method
- ValidationHookError exception handling
- Factory functions (create_validation_hook, extract_input_schemas)

Test Coverage Target: 85%+

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from greenlang.orchestrator.hooks.validation_hook import (
    InputSchemaSpec,
    InputValidationResult,
    PipelineValidationConfig,
    PreRunValidationHook,
    PreRunValidationResult,
    ValidationHookError,
    create_validation_hook,
    extract_input_schemas,
)
from greenlang.schema.models.config import (
    CoercionPolicy,
    UnknownFieldPolicy,
    ValidationOptions,
    ValidationProfile,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_input_schema_spec() -> InputSchemaSpec:
    """Create a sample InputSchemaSpec for testing."""
    return InputSchemaSpec(
        schema="gl://schemas/emissions/activity@1.3.0",
        required=True,
        profile=None,
        description="Activity data for emissions calculation",
    )


@pytest.fixture
def sample_validation_config() -> PipelineValidationConfig:
    """Create a sample PipelineValidationConfig for testing."""
    return PipelineValidationConfig(
        enabled=True,
        profile=ValidationProfile.STANDARD,
        fail_on_warnings=False,
        normalize_inputs=True,
        max_validation_time_seconds=60.0,
    )


@pytest.fixture
def sample_input_schemas() -> Dict[str, InputSchemaSpec]:
    """Create sample input schemas for testing."""
    return {
        "data": InputSchemaSpec(
            schema="gl://schemas/emissions/activity@1.3.0",
            required=True,
        ),
        "config": InputSchemaSpec(
            schema="gl://schemas/config/calculation@1.0.0",
            required=False,
        ),
    }


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Create a sample pipeline configuration dictionary."""
    return {
        "pipeline": {
            "id": "emissions-calculation",
            "validation": {
                "enabled": True,
                "profile": "strict",
                "fail_on_warnings": False,
                "normalize_inputs": True,
            },
            "inputs": {
                "data": {
                    "schema": "gl://schemas/emissions/activity@1.3.0",
                    "required": True,
                    "description": "Activity data input",
                },
                "config": {
                    "schema": "gl://schemas/config/calculation@1.0.0",
                    "required": False,
                    "profile": "permissive",
                },
            },
        }
    }


@pytest.fixture
def mock_validation_report() -> MagicMock:
    """Create a mock ValidationReport for testing."""
    mock_report = MagicMock()
    mock_report.valid = True
    mock_report.normalized_payload = {"energy": {"value": 100, "unit": "kWh"}}
    mock_report.findings = []
    mock_report.summary = MagicMock()
    mock_report.summary.error_count = 0
    mock_report.summary.warning_count = 0
    mock_report.summary.total_findings.return_value = 0
    return mock_report


# =============================================================================
# INPUT SCHEMA SPEC TESTS
# =============================================================================


class TestInputSchemaSpec:
    """Tests for InputSchemaSpec model."""

    def test_create_valid_schema_spec(self) -> None:
        """Test creating a valid InputSchemaSpec."""
        spec = InputSchemaSpec(
            schema="gl://schemas/emissions/activity@1.3.0",
            required=True,
        )
        assert spec.schema == "gl://schemas/emissions/activity@1.3.0"
        assert spec.required is True
        assert spec.profile is None
        assert spec.description is None

    def test_create_schema_spec_with_all_fields(self) -> None:
        """Test creating InputSchemaSpec with all optional fields."""
        spec = InputSchemaSpec(
            schema="gl://schemas/emissions/activity@1.3.0",
            required=False,
            profile=ValidationProfile.STRICT,
            description="Activity data for emissions calculation",
        )
        assert spec.schema == "gl://schemas/emissions/activity@1.3.0"
        assert spec.required is False
        assert spec.profile == ValidationProfile.STRICT
        assert spec.description == "Activity data for emissions calculation"

    def test_schema_uri_must_start_with_gl_schemas(self) -> None:
        """Test that schema URI must start with 'gl://schemas/'."""
        with pytest.raises(ValidationError) as exc_info:
            InputSchemaSpec(
                schema="https://example.com/schema",
                required=True,
            )
        assert "Must start with 'gl://schemas/'" in str(exc_info.value)

    def test_schema_uri_must_include_version(self) -> None:
        """Test that schema URI must include version after '@'."""
        with pytest.raises(ValidationError) as exc_info:
            InputSchemaSpec(
                schema="gl://schemas/emissions/activity",
                required=True,
            )
        assert "Must include version after '@'" in str(exc_info.value)

    def test_to_schema_ref_conversion(self) -> None:
        """Test converting InputSchemaSpec to SchemaRef."""
        spec = InputSchemaSpec(
            schema="gl://schemas/emissions/activity@1.3.0",
            required=True,
        )
        schema_ref = spec.to_schema_ref()
        assert schema_ref.schema_id == "emissions/activity"
        assert schema_ref.version == "1.3.0"

    def test_description_max_length(self) -> None:
        """Test that description is limited to 500 characters."""
        long_description = "a" * 501
        with pytest.raises(ValidationError):
            InputSchemaSpec(
                schema="gl://schemas/test@1.0.0",
                required=True,
                description=long_description,
            )


# =============================================================================
# PIPELINE VALIDATION CONFIG TESTS
# =============================================================================


class TestPipelineValidationConfig:
    """Tests for PipelineValidationConfig model."""

    def test_create_default_config(self) -> None:
        """Test creating PipelineValidationConfig with defaults."""
        config = PipelineValidationConfig()
        assert config.enabled is True
        assert config.profile == ValidationProfile.STANDARD
        assert config.fail_on_warnings is False
        assert config.normalize_inputs is True
        assert config.max_validation_time_seconds == 60.0

    def test_create_strict_config(self) -> None:
        """Test creating strict PipelineValidationConfig."""
        config = PipelineValidationConfig(
            enabled=True,
            profile=ValidationProfile.STRICT,
            fail_on_warnings=True,
            normalize_inputs=True,
            max_validation_time_seconds=30.0,
        )
        assert config.profile == ValidationProfile.STRICT
        assert config.fail_on_warnings is True
        assert config.max_validation_time_seconds == 30.0

    def test_max_validation_time_bounds(self) -> None:
        """Test max_validation_time_seconds bounds validation."""
        # Valid max time
        config = PipelineValidationConfig(max_validation_time_seconds=300.0)
        assert config.max_validation_time_seconds == 300.0

        # Too high
        with pytest.raises(ValidationError):
            PipelineValidationConfig(max_validation_time_seconds=400.0)

        # Too low
        with pytest.raises(ValidationError):
            PipelineValidationConfig(max_validation_time_seconds=0.0)

    def test_to_validation_options(self) -> None:
        """Test converting PipelineValidationConfig to ValidationOptions."""
        config = PipelineValidationConfig(
            profile=ValidationProfile.STRICT,
            normalize_inputs=True,
        )
        options = config.to_validation_options()
        assert options.profile == ValidationProfile.STRICT
        assert options.normalize is True
        assert options.emit_patches is True
        assert options.max_errors == 100
        assert options.fail_fast is False


# =============================================================================
# INPUT VALIDATION RESULT TESTS
# =============================================================================


class TestInputValidationResult:
    """Tests for InputValidationResult model."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid InputValidationResult."""
        result = InputValidationResult(
            input_name="data",
            valid=True,
            validation_time_ms=50.0,
        )
        assert result.input_name == "data"
        assert result.valid is True
        assert result.report is None
        assert result.normalized_value is None
        assert result.error is None

    def test_create_result_with_error(self) -> None:
        """Test creating InputValidationResult with error."""
        result = InputValidationResult(
            input_name="data",
            valid=False,
            error="Required input 'data' is missing",
            validation_time_ms=10.0,
        )
        assert result.valid is False
        assert result.error == "Required input 'data' is missing"

    def test_error_count_with_no_report(self) -> None:
        """Test error_count when no report is present."""
        result = InputValidationResult(
            input_name="data",
            valid=False,
            error="Validation failed",
        )
        assert result.error_count() == 1

        result_valid = InputValidationResult(
            input_name="data",
            valid=True,
        )
        assert result_valid.error_count() == 0

    def test_warning_count_with_no_report(self) -> None:
        """Test warning_count when no report is present."""
        result = InputValidationResult(
            input_name="data",
            valid=True,
        )
        assert result.warning_count() == 0

    def test_has_findings_without_report(self) -> None:
        """Test has_findings when no report is present."""
        result_with_error = InputValidationResult(
            input_name="data",
            valid=False,
            error="Error message",
        )
        assert result_with_error.has_findings() is True

        result_valid = InputValidationResult(
            input_name="data",
            valid=True,
        )
        assert result_valid.has_findings() is False


# =============================================================================
# PRE-RUN VALIDATION RESULT TESTS
# =============================================================================


class TestPreRunValidationResult:
    """Tests for PreRunValidationResult model."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid PreRunValidationResult."""
        result = PreRunValidationResult(
            valid=True,
            input_results={
                "data": InputValidationResult(input_name="data", valid=True),
            },
            errors=[],
            total_findings=0,
            total_validation_time_ms=100.0,
        )
        assert result.valid is True
        assert len(result.input_results) == 1
        assert len(result.errors) == 0

    def test_get_normalized_inputs(self) -> None:
        """Test getting normalized inputs from result."""
        result = PreRunValidationResult(
            valid=True,
            input_results={
                "data": InputValidationResult(
                    input_name="data",
                    valid=True,
                    normalized_value={"energy": 100},
                ),
                "config": InputValidationResult(
                    input_name="config",
                    valid=True,
                    normalized_value=None,
                ),
            },
        )
        normalized = result.get_normalized_inputs()
        assert "data" in normalized
        assert normalized["data"] == {"energy": 100}
        assert "config" not in normalized

    def test_get_failed_inputs(self) -> None:
        """Test getting failed inputs from result."""
        result = PreRunValidationResult(
            valid=False,
            input_results={
                "data": InputValidationResult(input_name="data", valid=False),
                "config": InputValidationResult(input_name="config", valid=True),
            },
            errors=["data validation failed"],
        )
        failed = result.get_failed_inputs()
        assert "data" in failed
        assert "config" not in failed

    def test_total_errors(self) -> None:
        """Test counting total errors across inputs."""
        result = PreRunValidationResult(
            valid=False,
            input_results={
                "data": InputValidationResult(
                    input_name="data",
                    valid=False,
                    error="Error 1",
                ),
                "config": InputValidationResult(
                    input_name="config",
                    valid=False,
                    error="Error 2",
                ),
            },
        )
        assert result.total_errors() == 2

    def test_format_summary(self) -> None:
        """Test formatting validation summary."""
        result = PreRunValidationResult(
            valid=True,
            input_results={
                "data": InputValidationResult(input_name="data", valid=True),
            },
            total_findings=0,
            total_validation_time_ms=50.0,
        )
        summary = result.format_summary()
        assert "PASSED" in summary
        assert "1/1 inputs valid" in summary
        assert "0 findings" in summary


# =============================================================================
# VALIDATION HOOK ERROR TESTS
# =============================================================================


class TestValidationHookError:
    """Tests for ValidationHookError exception."""

    def test_create_error_from_result(self) -> None:
        """Test creating ValidationHookError from result."""
        result = PreRunValidationResult(
            valid=False,
            input_results={
                "data": InputValidationResult(
                    input_name="data",
                    valid=False,
                    error="Required input 'data' is missing",
                ),
            },
            errors=["Required input 'data' is missing"],
        )
        error = ValidationHookError(result)
        assert error.result == result
        assert "Pre-run validation failed" in str(error)
        assert "1 error(s)" in str(error)

    def test_error_message_includes_first_error(self) -> None:
        """Test that error message includes the first error."""
        result = PreRunValidationResult(
            valid=False,
            input_results={},
            errors=["First error", "Second error"],
        )
        error = ValidationHookError(result)
        assert "First error" in str(error)


# =============================================================================
# PRE-RUN VALIDATION HOOK TESTS
# =============================================================================


class TestPreRunValidationHook:
    """Tests for PreRunValidationHook class."""

    def test_hook_initialization(
        self,
        sample_validation_config: PipelineValidationConfig,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test PreRunValidationHook initialization."""
        hook = PreRunValidationHook(sample_validation_config, sample_input_schemas)
        assert hook.config == sample_validation_config
        assert hook.input_schemas == sample_input_schemas
        assert hook._validator is None

    @pytest.mark.asyncio
    async def test_validate_disabled(
        self,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test validation when disabled."""
        config = PipelineValidationConfig(enabled=False)
        hook = PreRunValidationHook(config, sample_input_schemas)

        result = await hook.validate(
            inputs={"data": {"energy": 100}},
            context={"run_id": "test-123"},
        )

        assert result.valid is True
        assert len(result.input_results) == 0
        assert result.total_validation_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_validate_missing_required_input(
        self,
        sample_validation_config: PipelineValidationConfig,
    ) -> None:
        """Test validation with missing required input."""
        input_schemas = {
            "data": InputSchemaSpec(
                schema="gl://schemas/emissions/activity@1.3.0",
                required=True,
            ),
        }
        hook = PreRunValidationHook(sample_validation_config, input_schemas)

        result = await hook.validate(
            inputs={},  # Missing required 'data' input
            context={"run_id": "test-123"},
        )

        assert result.valid is False
        assert "data" in result.input_results
        assert result.input_results["data"].valid is False
        assert "Required input 'data' is missing" in result.input_results["data"].error

    @pytest.mark.asyncio
    async def test_validate_optional_input_not_provided(
        self,
        sample_validation_config: PipelineValidationConfig,
    ) -> None:
        """Test validation with optional input not provided."""
        input_schemas = {
            "config": InputSchemaSpec(
                schema="gl://schemas/config/calculation@1.0.0",
                required=False,
            ),
        }
        hook = PreRunValidationHook(sample_validation_config, input_schemas)

        result = await hook.validate(
            inputs={},  # Optional 'config' not provided
            context={"run_id": "test-123"},
        )

        assert result.valid is True
        assert "config" in result.input_results
        assert result.input_results["config"].valid is True

    @pytest.mark.asyncio
    async def test_validate_with_mocked_validator(
        self,
        sample_validation_config: PipelineValidationConfig,
        mock_validation_report: MagicMock,
    ) -> None:
        """Test validation with mocked SchemaValidator."""
        input_schemas = {
            "data": InputSchemaSpec(
                schema="gl://schemas/emissions/activity@1.3.0",
                required=True,
            ),
        }
        hook = PreRunValidationHook(sample_validation_config, input_schemas)

        # Mock the validator
        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_validation_report
        hook._validator = mock_validator

        result = await hook.validate(
            inputs={"data": {"energy": 100}},
            context={"run_id": "test-123"},
        )

        assert result.valid is True
        assert result.input_results["data"].valid is True
        assert result.input_results["data"].normalized_value == {
            "energy": {"value": 100, "unit": "kWh"}
        }

    @pytest.mark.asyncio
    async def test_validate_timeout(
        self,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test validation timeout handling."""
        config = PipelineValidationConfig(
            enabled=True,
            max_validation_time_seconds=0.001,  # Very short timeout
        )
        hook = PreRunValidationHook(config, sample_input_schemas)

        # Create a slow validator that will timeout
        async def slow_validate(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        with patch.object(
            hook, "_validate_input_with_timeout", side_effect=slow_validate
        ):
            result = await hook.validate(
                inputs={"data": {"energy": 100}},
                context={"run_id": "test-123"},
            )

            assert result.valid is False
            assert "timeout exceeded" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_fail_on_warnings(
        self,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test validation with fail_on_warnings enabled."""
        config = PipelineValidationConfig(
            enabled=True,
            fail_on_warnings=True,
        )
        hook = PreRunValidationHook(config, sample_input_schemas)

        # Mock validator that returns warnings
        mock_report = MagicMock()
        mock_report.valid = True
        mock_report.normalized_payload = None
        mock_report.findings = []
        mock_report.summary = MagicMock()
        mock_report.summary.error_count = 0
        mock_report.summary.warning_count = 1
        mock_report.summary.total_findings.return_value = 1

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        hook._validator = mock_validator

        result = await hook.validate(
            inputs={"data": {"energy": 100}},
            context={"run_id": "test-123"},
        )

        # With fail_on_warnings=True, warnings should cause failure
        assert result.valid is False

    def test_compute_provenance_hash(
        self,
        sample_validation_config: PipelineValidationConfig,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test provenance hash computation."""
        hook = PreRunValidationHook(sample_validation_config, sample_input_schemas)

        inputs = {"data": {"energy": 100}}
        results = {
            "data": InputValidationResult(input_name="data", valid=True),
        }
        context = {"run_id": "test-123"}

        hash1 = hook._compute_provenance_hash(inputs, results, context)
        hash2 = hook._compute_provenance_hash(inputs, results, context)

        # Same inputs should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_build_options_with_input_override(
        self,
        sample_validation_config: PipelineValidationConfig,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test building ValidationOptions with input-level profile override."""
        hook = PreRunValidationHook(sample_validation_config, sample_input_schemas)

        # Input with profile override
        spec = InputSchemaSpec(
            schema="gl://schemas/test@1.0.0",
            required=True,
            profile=ValidationProfile.STRICT,
        )

        options = hook._build_options(spec)
        assert options.profile == ValidationProfile.STRICT

    def test_build_options_without_override(
        self,
        sample_validation_config: PipelineValidationConfig,
        sample_input_schemas: Dict[str, InputSchemaSpec],
    ) -> None:
        """Test building ValidationOptions without input-level override."""
        hook = PreRunValidationHook(sample_validation_config, sample_input_schemas)

        # Input without profile override
        spec = InputSchemaSpec(
            schema="gl://schemas/test@1.0.0",
            required=True,
            profile=None,
        )

        options = hook._build_options(spec)
        # Should use pipeline-level profile
        assert options.profile == sample_validation_config.profile


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateValidationHook:
    """Tests for create_validation_hook factory function."""

    def test_create_hook_from_config(
        self, sample_pipeline_config: Dict[str, Any]
    ) -> None:
        """Test creating validation hook from pipeline config."""
        hook = create_validation_hook(sample_pipeline_config)

        assert hook is not None
        assert isinstance(hook, PreRunValidationHook)
        assert hook.config.enabled is True
        assert hook.config.profile == ValidationProfile.STRICT
        assert len(hook.input_schemas) == 2

    def test_create_hook_disabled(self) -> None:
        """Test creating hook when validation is disabled."""
        config = {
            "pipeline": {
                "validation": {"enabled": False},
                "inputs": {},
            }
        }
        hook = create_validation_hook(config)
        assert hook is None

    def test_create_hook_no_schemas(self) -> None:
        """Test creating hook with no input schemas."""
        config = {
            "pipeline": {
                "validation": {"enabled": True},
                "inputs": {},
            }
        }
        hook = create_validation_hook(config)
        assert hook is None

    def test_create_hook_invalid_profile(self) -> None:
        """Test creating hook with invalid profile falls back to standard."""
        config = {
            "pipeline": {
                "validation": {"enabled": True, "profile": "invalid_profile"},
                "inputs": {
                    "data": {"schema": "gl://schemas/test@1.0.0"},
                },
            }
        }
        hook = create_validation_hook(config)
        assert hook is not None
        assert hook.config.profile == ValidationProfile.STANDARD


class TestExtractInputSchemas:
    """Tests for extract_input_schemas function."""

    def test_extract_schemas_from_config(
        self, sample_pipeline_config: Dict[str, Any]
    ) -> None:
        """Test extracting input schemas from pipeline config."""
        schemas = extract_input_schemas(sample_pipeline_config)

        assert len(schemas) == 2
        assert "data" in schemas
        assert "config" in schemas
        assert schemas["data"].required is True
        assert schemas["config"].required is False

    def test_extract_schemas_with_profile_override(
        self, sample_pipeline_config: Dict[str, Any]
    ) -> None:
        """Test extracting schemas with profile overrides."""
        schemas = extract_input_schemas(sample_pipeline_config)

        assert schemas["config"].profile == ValidationProfile.PERMISSIVE

    def test_extract_schemas_no_inputs(self) -> None:
        """Test extracting schemas when no inputs defined."""
        config = {"pipeline": {"inputs": {}}}
        schemas = extract_input_schemas(config)
        assert len(schemas) == 0

    def test_extract_schemas_skip_non_dict_inputs(self) -> None:
        """Test that non-dict input definitions are skipped."""
        config = {
            "pipeline": {
                "inputs": {
                    "data": {"schema": "gl://schemas/test@1.0.0"},
                    "invalid": "not_a_dict",
                }
            }
        }
        schemas = extract_input_schemas(config)
        assert len(schemas) == 1
        assert "data" in schemas
        assert "invalid" not in schemas

    def test_extract_schemas_skip_inputs_without_schema(self) -> None:
        """Test that inputs without schema are skipped."""
        config = {
            "pipeline": {
                "inputs": {
                    "data": {"schema": "gl://schemas/test@1.0.0"},
                    "no_schema": {"required": True},
                }
            }
        }
        schemas = extract_input_schemas(config)
        assert len(schemas) == 1
        assert "data" in schemas
        assert "no_schema" not in schemas


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestValidationHookIntegration:
    """Integration tests for the validation hook system."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(
        self, sample_pipeline_config: Dict[str, Any]
    ) -> None:
        """Test complete validation flow from config to result."""
        # Create hook from config
        hook = create_validation_hook(sample_pipeline_config)
        assert hook is not None

        # Create mock validator
        mock_report = MagicMock()
        mock_report.valid = True
        mock_report.normalized_payload = {"normalized": True}
        mock_report.findings = []
        mock_report.summary = MagicMock()
        mock_report.summary.error_count = 0
        mock_report.summary.warning_count = 0
        mock_report.summary.total_findings.return_value = 0

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_report
        hook._validator = mock_validator

        # Run validation
        result = await hook.validate(
            inputs={
                "data": {"energy": 100, "unit": "kWh"},
            },
            context={
                "tenant_id": "tenant-123",
                "run_id": "run-456",
            },
        )

        # Verify result
        assert result.valid is True
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert result.validated_at is not None
        assert result.total_validation_time_ms > 0

    @pytest.mark.asyncio
    async def test_validation_error_propagation(self) -> None:
        """Test that validation errors are properly propagated."""
        config = PipelineValidationConfig(enabled=True)
        input_schemas = {
            "data": InputSchemaSpec(
                schema="gl://schemas/test@1.0.0",
                required=True,
            ),
        }
        hook = PreRunValidationHook(config, input_schemas)

        # Provide no inputs for required field
        result = await hook.validate(
            inputs={},
            context={"run_id": "test-123"},
        )

        assert result.valid is False
        assert len(result.errors) > 0

        # Test that ValidationHookError can be raised
        with pytest.raises(ValidationHookError) as exc_info:
            if not result.valid:
                raise ValidationHookError(result)

        assert exc_info.value.result == result


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestValidationHookEdgeCases:
    """Edge case tests for validation hook."""

    def test_empty_input_schemas(self) -> None:
        """Test hook with empty input schemas."""
        config = PipelineValidationConfig(enabled=True)
        hook = PreRunValidationHook(config, {})
        assert len(hook.input_schemas) == 0

    @pytest.mark.asyncio
    async def test_validation_with_exception_in_validator(
        self,
        sample_validation_config: PipelineValidationConfig,
    ) -> None:
        """Test handling of exceptions during validation."""
        input_schemas = {
            "data": InputSchemaSpec(
                schema="gl://schemas/test@1.0.0",
                required=True,
            ),
        }
        hook = PreRunValidationHook(sample_validation_config, input_schemas)

        # Mock validator that raises exception
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = RuntimeError("Validator crashed")
        hook._validator = mock_validator

        result = await hook.validate(
            inputs={"data": {"value": 100}},
            context={"run_id": "test-123"},
        )

        # Exception should be caught and result should indicate failure
        assert result.valid is False
        assert "data" in result.input_results
        assert result.input_results["data"].valid is False
        assert "Validator crashed" in result.input_results["data"].error

    def test_input_schema_spec_model_config(self) -> None:
        """Test InputSchemaSpec model configuration."""
        # Should reject extra fields
        with pytest.raises(ValidationError):
            InputSchemaSpec(
                schema="gl://schemas/test@1.0.0",
                required=True,
                extra_field="not_allowed",  # type: ignore
            )

    def test_pipeline_validation_config_model_config(self) -> None:
        """Test PipelineValidationConfig model configuration."""
        # Should reject extra fields
        with pytest.raises(ValidationError):
            PipelineValidationConfig(
                enabled=True,
                extra_field="not_allowed",  # type: ignore
            )
