"""
Comprehensive tests for GreenLang Validation Framework.

Tests cover:
- ValidationFramework class functionality
- ValidationResult error collection and merging
- Severity levels (ERROR, WARNING, INFO)
- Pre/post validator hooks
- Batch validation
- Validator enabling/disabling
"""

import pytest
from datetime import datetime
from typing import Any

from greenlang.validation.framework import (
    ValidationFramework,
    ValidationResult,
    ValidationError,
    ValidationSeverity,
    Validator
)


# Test Fixtures
@pytest.fixture
def framework():
    """Create a fresh validation framework."""
    return ValidationFramework()


@pytest.fixture
def sample_validator():
    """Create a sample validator function."""
    def validator(data: Any) -> ValidationResult:
        result = ValidationResult(valid=True)
        if not isinstance(data, dict):
            error = ValidationError(
                field="root",
                message="Data must be a dictionary",
                severity=ValidationSeverity.ERROR,
                validator="sample"
            )
            result.add_error(error)
        return result
    return validator


@pytest.fixture
def age_validator():
    """Create an age validation function."""
    def validator(data: Any) -> ValidationResult:
        result = ValidationResult(valid=True)
        if "age" in data and data["age"] < 0:
            error = ValidationError(
                field="age",
                message="Age cannot be negative",
                severity=ValidationSeverity.ERROR,
                validator="age_check",
                value=data["age"]
            )
            result.add_error(error)
        return result
    return validator


@pytest.fixture
def warning_validator():
    """Create a validator that generates warnings."""
    def validator(data: Any) -> ValidationResult:
        result = ValidationResult(valid=True)
        if "email" not in data:
            warning = ValidationError(
                field="email",
                message="Email is recommended",
                severity=ValidationSeverity.WARNING,
                validator="email_check"
            )
            result.add_error(warning)
        return result
    return validator


@pytest.fixture
def failing_validator():
    """Create a validator that always raises an exception."""
    def validator(data: Any) -> ValidationResult:
        raise ValueError("Validator crashed!")
    return validator


# ValidationError Tests
class TestValidationError:
    """Test ValidationError model."""

    def test_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError(
            field="username",
            message="Username is required",
            severity=ValidationSeverity.ERROR,
            validator="required_fields"
        )
        assert error.field == "username"
        assert error.message == "Username is required"
        assert error.severity == ValidationSeverity.ERROR
        assert error.validator == "required_fields"

    def test_error_with_value_and_expected(self):
        """Test error with value and expected fields."""
        error = ValidationError(
            field="age",
            message="Age must be at least 18",
            severity=ValidationSeverity.ERROR,
            validator="age_check",
            value=15,
            expected=18
        )
        assert error.value == 15
        assert error.expected == 18

    def test_error_string_representation(self):
        """Test string representation of error."""
        error = ValidationError(
            field="email",
            message="Invalid email format",
            severity=ValidationSeverity.WARNING,
            validator="format_check"
        )
        assert "[WARNING]" in str(error)
        assert "email" in str(error)
        assert "Invalid email format" in str(error)

    def test_error_severity_levels(self):
        """Test all severity levels."""
        for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            error = ValidationError(
                field="test",
                message="Test message",
                severity=severity,
                validator="test"
            )
            assert error.severity == severity


# ValidationResult Tests
class TestValidationResult:
    """Test ValidationResult functionality."""

    def test_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.info) == 0
        assert isinstance(result.timestamp, datetime)

    def test_add_error(self):
        """Test adding errors to result."""
        result = ValidationResult(valid=True)

        error = ValidationError(
            field="name",
            message="Name is required",
            severity=ValidationSeverity.ERROR,
            validator="required"
        )
        result.add_error(error)

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == error

    def test_add_warning(self):
        """Test adding warnings to result."""
        result = ValidationResult(valid=True)

        warning = ValidationError(
            field="phone",
            message="Phone number recommended",
            severity=ValidationSeverity.WARNING,
            validator="recommended"
        )
        result.add_error(warning)

        assert result.valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_add_info(self):
        """Test adding info messages to result."""
        result = ValidationResult(valid=True)

        info = ValidationError(
            field="status",
            message="Record processed successfully",
            severity=ValidationSeverity.INFO,
            validator="info"
        )
        result.add_error(info)

        assert result.valid is True
        assert len(result.info) == 1

    def test_merge_results(self):
        """Test merging two validation results."""
        result1 = ValidationResult(valid=True)
        result2 = ValidationResult(valid=True)

        # Add error to result2
        error = ValidationError(
            field="field1",
            message="Error 1",
            severity=ValidationSeverity.ERROR,
            validator="test"
        )
        result2.add_error(error)

        # Merge
        result1.merge(result2)

        assert result1.valid is False
        assert len(result1.errors) == 1

    def test_merge_multiple_results(self):
        """Test merging multiple results."""
        result = ValidationResult(valid=True)

        # Create multiple results with different severities
        result2 = ValidationResult(valid=True)
        result2.add_error(ValidationError(
            field="f1", message="Error", severity=ValidationSeverity.ERROR, validator="t"
        ))

        result3 = ValidationResult(valid=True)
        result3.add_error(ValidationError(
            field="f2", message="Warning", severity=ValidationSeverity.WARNING, validator="t"
        ))

        result.merge(result2)
        result.merge(result3)

        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.valid is False

    def test_merge_metadata(self):
        """Test merging metadata."""
        result1 = ValidationResult(valid=True)
        result1.metadata["key1"] = "value1"

        result2 = ValidationResult(valid=True)
        result2.metadata["key2"] = "value2"

        result1.merge(result2)

        assert "key1" in result1.metadata
        assert "key2" in result1.metadata

    def test_get_error_count(self):
        """Test error counting."""
        result = ValidationResult(valid=True)

        for i in range(5):
            result.add_error(ValidationError(
                field=f"field{i}",
                message=f"Error {i}",
                severity=ValidationSeverity.ERROR,
                validator="test"
            ))

        assert result.get_error_count() == 5

    def test_get_warning_count(self):
        """Test warning counting."""
        result = ValidationResult(valid=True)

        for i in range(3):
            result.add_error(ValidationError(
                field=f"field{i}",
                message=f"Warning {i}",
                severity=ValidationSeverity.WARNING,
                validator="test"
            ))

        assert result.get_warning_count() == 3

    def test_get_summary(self):
        """Test getting summary string."""
        result = ValidationResult(valid=True)

        # Add errors and warnings
        result.add_error(ValidationError(
            field="f1", message="Error", severity=ValidationSeverity.ERROR, validator="t"
        ))
        result.add_error(ValidationError(
            field="f2", message="Warning", severity=ValidationSeverity.WARNING, validator="t"
        ))

        summary = result.get_summary()
        assert "FAILED" in summary
        assert "1 errors" in summary
        assert "1 warnings" in summary

    def test_summary_passed(self):
        """Test summary for passed validation."""
        result = ValidationResult(valid=True)
        summary = result.get_summary()
        assert "PASSED" in summary

    def test_string_representation(self):
        """Test string representation."""
        result = ValidationResult(valid=True)
        assert "PASSED" in str(result)


# ValidationFramework Tests
class TestValidationFramework:
    """Test ValidationFramework functionality."""

    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = ValidationFramework()
        assert len(framework.validators) == 0
        assert len(framework.validator_configs) == 0
        assert len(framework.pre_validators) == 0
        assert len(framework.post_validators) == 0

    def test_add_validator(self, framework, sample_validator):
        """Test adding a validator."""
        framework.add_validator("sample", sample_validator)

        assert "sample" in framework.validators
        assert "sample" in framework.validator_configs
        assert framework.validators["sample"] == sample_validator

    def test_add_validator_with_config(self, framework, sample_validator):
        """Test adding validator with custom config."""
        config = Validator(
            name="custom",
            enabled=False,
            severity=ValidationSeverity.WARNING
        )

        framework.add_validator("custom", sample_validator, config)

        assert framework.validator_configs["custom"].enabled is False
        assert framework.validator_configs["custom"].severity == ValidationSeverity.WARNING

    def test_remove_validator(self, framework, sample_validator):
        """Test removing a validator."""
        framework.add_validator("sample", sample_validator)
        assert "sample" in framework.validators

        framework.remove_validator("sample")
        assert "sample" not in framework.validators
        assert "sample" not in framework.validator_configs

    def test_validate_single_validator(self, framework, sample_validator):
        """Test validation with single validator."""
        framework.add_validator("sample", sample_validator)

        result = framework.validate({"key": "value"})
        assert result.valid is True

    def test_validate_failing_data(self, framework, sample_validator):
        """Test validation with failing data."""
        framework.add_validator("sample", sample_validator)

        result = framework.validate("not a dict")
        assert result.valid is False
        assert len(result.errors) == 1

    def test_validate_multiple_validators(self, framework, sample_validator, age_validator):
        """Test validation with multiple validators."""
        framework.add_validator("sample", sample_validator)
        framework.add_validator("age", age_validator)

        result = framework.validate({"age": -5})
        assert result.valid is False
        assert len(result.errors) == 1

    def test_validate_specific_validators(self, framework, sample_validator, age_validator):
        """Test validation with specific validator list."""
        framework.add_validator("sample", sample_validator)
        framework.add_validator("age", age_validator)

        # Only run age validator
        result = framework.validate({"age": 25}, validators=["age"])
        assert result.valid is True
        assert "validators_run" in result.metadata
        assert result.metadata["validators_run"] == ["age"]

    def test_stop_on_error(self, framework, sample_validator, age_validator):
        """Test stop on error functionality."""
        framework.add_validator("sample", sample_validator)
        framework.add_validator("age", age_validator)

        result = framework.validate("not a dict", stop_on_error=True)
        assert result.valid is False
        # Should stop after first validator fails

    def test_disabled_validator(self, framework, sample_validator):
        """Test that disabled validators are skipped."""
        framework.add_validator("sample", sample_validator)
        framework.disable_validator("sample")

        result = framework.validate("not a dict")
        assert result.valid is True  # Validator was skipped

    def test_enable_validator(self, framework, sample_validator):
        """Test enabling a validator."""
        config = Validator(name="sample", enabled=False)
        framework.add_validator("sample", sample_validator, config)

        framework.enable_validator("sample")

        result = framework.validate("not a dict")
        assert result.valid is False  # Now validator runs

    def test_validator_exception_handling(self, framework, failing_validator):
        """Test handling of validator exceptions."""
        framework.add_validator("failing", failing_validator)

        result = framework.validate({"test": "data"})
        assert result.valid is False
        assert len(result.errors) == 1
        assert "failed" in result.errors[0].message.lower()

    def test_add_pre_validator(self, framework, sample_validator):
        """Test adding pre-validator hook."""
        pre_called = []

        def pre_hook(data):
            pre_called.append(True)

        framework.add_pre_validator(pre_hook)
        framework.add_validator("sample", sample_validator)

        framework.validate({"test": "data"})
        assert len(pre_called) == 1

    def test_add_post_validator(self, framework, sample_validator):
        """Test adding post-validator hook."""
        post_called = []

        def post_hook(data, result):
            post_called.append(True)
            assert isinstance(result, ValidationResult)

        framework.add_post_validator(post_hook)
        framework.add_validator("sample", sample_validator)

        framework.validate({"test": "data"})
        assert len(post_called) == 1

    def test_pre_validator_exception_handling(self, framework, sample_validator):
        """Test that pre-validator exceptions don't break validation."""
        def failing_pre(data):
            raise ValueError("Pre-validator failed")

        framework.add_pre_validator(failing_pre)
        framework.add_validator("sample", sample_validator)

        result = framework.validate({"test": "data"})
        # Should still complete validation
        assert result.valid is True

    def test_batch_validation(self, framework, age_validator):
        """Test batch validation."""
        framework.add_validator("age", age_validator)

        data_list = [
            {"age": 25},
            {"age": -5},
            {"age": 30}
        ]

        results = framework.validate_batch(data_list)

        assert len(results) == 3
        assert results[0].valid is True
        assert results[1].valid is False
        assert results[2].valid is True

    def test_batch_validation_empty_list(self, framework):
        """Test batch validation with empty list."""
        results = framework.validate_batch([])
        assert len(results) == 0

    def test_get_validator_names(self, framework, sample_validator, age_validator):
        """Test getting validator names."""
        framework.add_validator("sample", sample_validator)
        framework.add_validator("age", age_validator)

        names = framework.get_validator_names()
        assert "sample" in names
        assert "age" in names

    def test_get_validation_summary(self, framework, age_validator):
        """Test getting batch validation summary."""
        framework.add_validator("age", age_validator)

        data_list = [
            {"age": 25},
            {"age": -5},
            {"age": 30},
            {"age": -1}
        ]

        results = framework.validate_batch(data_list)
        summary = framework.get_validation_summary(results)

        assert summary["total"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 2
        assert summary["pass_rate"] == 50.0
        assert summary["total_errors"] == 2

    def test_validation_summary_all_passed(self, framework, age_validator):
        """Test summary when all validations pass."""
        framework.add_validator("age", age_validator)

        data_list = [{"age": i} for i in range(5)]
        results = framework.validate_batch(data_list)
        summary = framework.get_validation_summary(results)

        assert summary["passed"] == 5
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 100.0

    def test_validation_metadata(self, framework, sample_validator):
        """Test that metadata is set correctly."""
        framework.add_validator("sample", sample_validator)

        result = framework.validate({"test": "data"})

        assert "validators_run" in result.metadata
        assert "total_issues" in result.metadata

    def test_warnings_dont_fail_validation(self, framework, warning_validator):
        """Test that warnings don't cause validation to fail."""
        framework.add_validator("warning", warning_validator)

        result = framework.validate({"name": "test"})

        assert result.valid is True
        assert len(result.warnings) == 1

    def test_mixed_severities(self, framework, age_validator, warning_validator):
        """Test validation with mixed severity levels."""
        framework.add_validator("age", age_validator)
        framework.add_validator("warning", warning_validator)

        result = framework.validate({"age": -5})

        assert result.valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
