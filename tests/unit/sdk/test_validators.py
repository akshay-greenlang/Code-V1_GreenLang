"""
Comprehensive tests for SDK Validator abstraction.

Tests cover:
- Validator initialization
- Validate method functionality
- Callable interface
- Result structure
- Error reporting
- Complex validation rules
"""

import pytest
from typing import Dict, Any, List
from greenlang.sdk.base import Validator, Result


class RangeValidator(Validator[int]):
    """Validator that checks if number is within range."""

    def __init__(self, min_val: int, max_val: int):
        """Initialize with min and max values."""
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, data: int) -> Result:
        """Validate that data is within range."""
        if not isinstance(data, (int, float)):
            return Result(
                success=False,
                error=f"Expected number, got {type(data).__name__}",
            )

        if data < self.min_val or data > self.max_val:
            return Result(
                success=False,
                error=f"Value {data} not in range [{self.min_val}, {self.max_val}]",
            )

        return Result(success=True, data=data)


class RequiredFieldsValidator(Validator[Dict[str, Any]]):
    """Validator that checks for required dictionary fields."""

    def __init__(self, required_fields: List[str]):
        """Initialize with required field names."""
        self.required_fields = required_fields

    def validate(self, data: Dict[str, Any]) -> Result:
        """Validate that all required fields are present."""
        if not isinstance(data, dict):
            return Result(
                success=False,
                error=f"Expected dict, got {type(data).__name__}",
            )

        missing = [field for field in self.required_fields if field not in data]

        if missing:
            return Result(
                success=False,
                error=f"Missing required fields: {', '.join(missing)}",
                metadata={"missing_fields": missing},
            )

        return Result(
            success=True,
            data=data,
            metadata={"validated_fields": self.required_fields},
        )


class EmailValidator(Validator[str]):
    """Simple email validator."""

    def validate(self, data: str) -> Result:
        """Validate email format."""
        if not isinstance(data, str):
            return Result(success=False, error="Email must be a string")

        if "@" not in data or "." not in data.split("@")[-1]:
            return Result(
                success=False, error=f"Invalid email format: {data}"
            )

        return Result(success=True, data=data)


class AlwaysPassValidator(Validator[Any]):
    """Validator that always passes."""

    def validate(self, data: Any) -> Result:
        """Always return success."""
        return Result(success=True, data=data)


class AlwaysFailValidator(Validator[Any]):
    """Validator that always fails."""

    def validate(self, data: Any) -> Result:
        """Always return failure."""
        return Result(success=False, error="Validation always fails")


@pytest.mark.unit
class TestValidatorBasics:
    """Test basic validator functionality."""

    def test_range_validator_valid(self):
        """Test range validator with valid input."""
        validator = RangeValidator(0, 100)
        result = validator.validate(50)

        assert result.success is True
        assert result.data == 50
        assert result.error is None

    def test_range_validator_invalid_too_low(self):
        """Test range validator with value too low."""
        validator = RangeValidator(10, 20)
        result = validator.validate(5)

        assert result.success is False
        assert "not in range" in result.error

    def test_range_validator_invalid_too_high(self):
        """Test range validator with value too high."""
        validator = RangeValidator(10, 20)
        result = validator.validate(25)

        assert result.success is False
        assert "not in range" in result.error

    def test_range_validator_boundary_values(self):
        """Test range validator at boundaries."""
        validator = RangeValidator(0, 100)

        # Test lower boundary
        result_min = validator.validate(0)
        assert result_min.success is True

        # Test upper boundary
        result_max = validator.validate(100)
        assert result_max.success is True


@pytest.mark.unit
class TestValidatorCallable:
    """Test validator callable interface."""

    def test_validator_is_callable(self):
        """Test that validator can be called directly."""
        validator = AlwaysPassValidator()
        result = validator("any data")

        assert result is True

    def test_callable_returns_bool(self):
        """Test that calling validator returns boolean."""
        pass_validator = AlwaysPassValidator()
        fail_validator = AlwaysFailValidator()

        assert pass_validator("data") is True
        assert fail_validator("data") is False

    def test_callable_different_from_validate(self):
        """Test that callable returns bool while validate returns Result."""
        validator = RangeValidator(0, 10)

        # validate() returns Result
        validate_result = validator.validate(5)
        assert isinstance(validate_result, Result)

        # Calling directly returns bool
        call_result = validator(5)
        assert isinstance(call_result, bool)


@pytest.mark.unit
class TestRequiredFieldsValidator:
    """Test required fields validator."""

    def test_all_fields_present(self):
        """Test validation when all required fields present."""
        validator = RequiredFieldsValidator(["name", "age"])
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        result = validator.validate(data)

        assert result.success is True
        assert result.error is None
        assert "validated_fields" in result.metadata

    def test_missing_single_field(self):
        """Test validation with one missing field."""
        validator = RequiredFieldsValidator(["name", "age"])
        data = {"name": "Bob"}
        result = validator.validate(data)

        assert result.success is False
        assert "age" in result.error
        assert "missing_fields" in result.metadata
        assert "age" in result.metadata["missing_fields"]

    def test_missing_multiple_fields(self):
        """Test validation with multiple missing fields."""
        validator = RequiredFieldsValidator(["name", "age", "email"])
        data = {"name": "Charlie"}
        result = validator.validate(data)

        assert result.success is False
        assert "missing_fields" in result.metadata
        assert len(result.metadata["missing_fields"]) == 2

    def test_no_required_fields(self):
        """Test validator with no required fields."""
        validator = RequiredFieldsValidator([])
        data = {"any": "data"}
        result = validator.validate(data)

        assert result.success is True

    def test_wrong_input_type(self):
        """Test validator with wrong input type."""
        validator = RequiredFieldsValidator(["name"])
        result = validator.validate("not a dict")

        assert result.success is False
        assert "Expected dict" in result.error


@pytest.mark.unit
class TestEmailValidator:
    """Test email validator."""

    def test_valid_email(self):
        """Test validation with valid email."""
        validator = EmailValidator()
        result = validator.validate("user@example.com")

        assert result.success is True
        assert result.data == "user@example.com"

    def test_valid_email_complex(self):
        """Test validation with complex valid email."""
        validator = EmailValidator()
        emails = [
            "test.user@example.com",
            "user+tag@example.co.uk",
            "user_name@sub.example.org",
        ]

        for email in emails:
            result = validator.validate(email)
            assert result.success is True, f"Failed for {email}"

    def test_invalid_email_no_at(self):
        """Test validation with email missing @."""
        validator = EmailValidator()
        result = validator.validate("userexample.com")

        assert result.success is False
        assert "Invalid email" in result.error

    def test_invalid_email_no_domain(self):
        """Test validation with email missing domain."""
        validator = EmailValidator()
        result = validator.validate("user@")

        assert result.success is False

    def test_invalid_email_no_extension(self):
        """Test validation with email missing extension."""
        validator = EmailValidator()
        result = validator.validate("user@example")

        assert result.success is False

    def test_email_wrong_type(self):
        """Test email validator with non-string input."""
        validator = EmailValidator()
        result = validator.validate(12345)

        assert result.success is False
        assert "must be a string" in result.error


@pytest.mark.unit
class TestValidatorEdgeCases:
    """Test validator edge cases."""

    def test_validator_with_none(self):
        """Test validators with None input."""
        range_validator = RangeValidator(0, 10)
        result = range_validator.validate(None)

        assert result.success is False

    def test_validator_with_empty_dict(self):
        """Test required fields validator with empty dict."""
        validator = RequiredFieldsValidator(["name"])
        result = validator.validate({})

        assert result.success is False
        assert "missing_fields" in result.metadata

    def test_validator_with_extra_fields(self):
        """Test that extra fields don't affect validation."""
        validator = RequiredFieldsValidator(["name"])
        data = {"name": "Alice", "age": 30, "extra": "data"}
        result = validator.validate(data)

        assert result.success is True

    def test_range_validator_float(self):
        """Test range validator accepts floats."""
        validator = RangeValidator(0, 10)
        result = validator.validate(5.5)

        assert result.success is True


@pytest.mark.unit
class TestValidatorComposition:
    """Test composing multiple validators."""

    def test_sequential_validation(self):
        """Test running validators sequentially."""
        validators = [
            RangeValidator(0, 100),
            RangeValidator(10, 50),
        ]

        # Value passes first but fails second
        value = 75
        results = [v.validate(value) for v in validators]

        assert results[0].success is True
        assert results[1].success is False

    def test_all_validators_pass(self):
        """Test when all validators pass."""
        data = {"name": "Alice", "age": 30, "email": "alice@example.com"}

        validators = [
            RequiredFieldsValidator(["name", "age"]),
            RequiredFieldsValidator(["email"]),
        ]

        results = [v.validate(data) for v in validators]
        assert all(r.success for r in results)

    def test_any_validator_fails(self):
        """Test when any validator fails."""
        data = {"name": "Bob"}

        validators = [
            RequiredFieldsValidator(["name"]),
            RequiredFieldsValidator(["age"]),
        ]

        results = [v.validate(data) for v in validators]
        assert not all(r.success for r in results)


@pytest.mark.unit
class TestValidatorResult:
    """Test validator result structure."""

    def test_success_result_structure(self):
        """Test structure of successful validation result."""
        validator = AlwaysPassValidator()
        result = validator.validate({"data": "value"})

        assert hasattr(result, "success")
        assert hasattr(result, "data")
        assert hasattr(result, "error")
        assert hasattr(result, "metadata")

    def test_failure_result_structure(self):
        """Test structure of failed validation result."""
        validator = AlwaysFailValidator()
        result = validator.validate("any")

        assert result.success is False
        assert result.error is not None
        assert isinstance(result.error, str)

    def test_result_with_metadata(self):
        """Test that result can include metadata."""
        validator = RequiredFieldsValidator(["field1", "field2"])
        data = {"field1": "value1", "field2": "value2"}
        result = validator.validate(data)

        assert result.metadata is not None
        assert isinstance(result.metadata, dict)
        assert "validated_fields" in result.metadata
