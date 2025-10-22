"""
GreenLang Validation Framework
Core validation framework with multi-layer validation support.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation errors."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationError(BaseModel):
    """A single validation error or warning."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    severity: ValidationSeverity = Field(default=ValidationSeverity.ERROR, description="Severity level")
    validator: str = Field(..., description="Name of validator that failed")
    value: Any = Field(default=None, description="Value that failed validation")
    expected: Any = Field(default=None, description="Expected value or format")
    location: Optional[str] = Field(default=None, description="Location in data structure")

    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.field}: {self.message}"


class ValidationResult(BaseModel):
    """Result of validation process."""
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(default_factory=list, description="List of errors")
    warnings: List[ValidationError] = Field(default_factory=list, description="List of warnings")
    info: List[ValidationError] = Field(default_factory=list, description="List of info messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")

    def add_error(self, error: ValidationError):
        """Add an error to the result."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)

    def merge(self, other: "ValidationResult"):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.valid:
            self.valid = False
        self.metadata.update(other.metadata)

    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)

    def get_warning_count(self) -> int:
        """Get total number of warnings."""
        return len(self.warnings)

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        status = "PASSED" if self.valid else "FAILED"
        return f"Validation {status}: {len(self.errors)} errors, {len(self.warnings)} warnings"

    def __str__(self):
        return self.get_summary()


class Validator(BaseModel):
    """Base validator configuration."""
    name: str = Field(..., description="Validator name")
    enabled: bool = Field(default=True, description="Whether validator is enabled")
    severity: ValidationSeverity = Field(default=ValidationSeverity.ERROR, description="Default severity")
    stop_on_error: bool = Field(default=False, description="Stop validation on first error")

    class Config:
        arbitrary_types_allowed = True


class ValidationFramework:
    """
    Core validation framework supporting multiple validation strategies.

    Provides:
    - Multi-layer validation (schema, rules, quality)
    - Configurable severity levels
    - Validation result aggregation
    - Custom validator registration
    - Conditional validation
    - Batch validation support

    Example:
        framework = ValidationFramework()
        framework.add_validator("schema", schema_validator)
        framework.add_validator("business_rules", rules_validator)

        result = framework.validate(data)
        if not result.valid:
            print(result.get_summary())
    """

    def __init__(self):
        """Initialize validation framework."""
        self.validators: Dict[str, Callable] = {}
        self.validator_configs: Dict[str, Validator] = {}
        self.pre_validators: List[Callable] = []
        self.post_validators: List[Callable] = []

    def add_validator(
        self,
        name: str,
        validator_func: Callable[[Any], ValidationResult],
        config: Optional[Validator] = None
    ):
        """
        Register a validator function.

        Args:
            name: Unique name for the validator
            validator_func: Function that takes data and returns ValidationResult
            config: Optional validator configuration
        """
        if config is None:
            config = Validator(name=name)

        self.validators[name] = validator_func
        self.validator_configs[name] = config
        logger.debug(f"Registered validator: {name}")

    def remove_validator(self, name: str):
        """Remove a validator by name."""
        if name in self.validators:
            del self.validators[name]
            del self.validator_configs[name]
            logger.debug(f"Removed validator: {name}")

    def add_pre_validator(self, validator_func: Callable):
        """Add a pre-validation hook."""
        self.pre_validators.append(validator_func)

    def add_post_validator(self, validator_func: Callable):
        """Add a post-validation hook."""
        self.post_validators.append(validator_func)

    def validate(
        self,
        data: Any,
        validators: Optional[List[str]] = None,
        stop_on_error: bool = False
    ) -> ValidationResult:
        """
        Validate data using registered validators.

        Args:
            data: Data to validate
            validators: List of validator names to use (all if None)
            stop_on_error: Stop on first error

        Returns:
            Aggregated validation result
        """
        result = ValidationResult(valid=True)

        # Determine which validators to use
        if validators is None:
            validators = list(self.validators.keys())

        # Run pre-validators
        for pre_validator in self.pre_validators:
            try:
                pre_validator(data)
            except Exception as e:
                logger.warning(f"Pre-validator failed: {str(e)}")

        # Run validators
        for validator_name in validators:
            if validator_name not in self.validators:
                logger.warning(f"Validator not found: {validator_name}")
                continue

            config = self.validator_configs[validator_name]
            if not config.enabled:
                logger.debug(f"Skipping disabled validator: {validator_name}")
                continue

            try:
                validator_func = self.validators[validator_name]
                validator_result = validator_func(data)

                # Merge result
                result.merge(validator_result)

                # Check if should stop
                if (stop_on_error or config.stop_on_error) and not validator_result.valid:
                    logger.debug(f"Stopping validation after {validator_name} due to errors")
                    break

            except Exception as e:
                logger.error(f"Validator {validator_name} raised exception: {str(e)}", exc_info=True)
                error = ValidationError(
                    field="__framework__",
                    message=f"Validator {validator_name} failed: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    validator=validator_name
                )
                result.add_error(error)
                result.valid = False

                if stop_on_error:
                    break

        # Run post-validators
        for post_validator in self.post_validators:
            try:
                post_validator(data, result)
            except Exception as e:
                logger.warning(f"Post-validator failed: {str(e)}")

        # Set metadata
        result.metadata["validators_run"] = validators
        result.metadata["total_issues"] = len(result.errors) + len(result.warnings)

        return result

    def validate_batch(
        self,
        data_list: List[Any],
        validators: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate a batch of data items.

        Args:
            data_list: List of data items to validate
            validators: List of validator names to use

        Returns:
            List of validation results
        """
        results = []
        for data in data_list:
            result = self.validate(data, validators)
            results.append(result)

        return results

    def get_validator_names(self) -> List[str]:
        """Get list of registered validator names."""
        return list(self.validators.keys())

    def enable_validator(self, name: str):
        """Enable a validator."""
        if name in self.validator_configs:
            self.validator_configs[name].enabled = True

    def disable_validator(self, name: str):
        """Disable a validator."""
        if name in self.validator_configs:
            self.validator_configs[name].enabled = False

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get summary statistics for batch validation.

        Args:
            results: List of validation results

        Returns:
            Summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.valid)
        failed = total - passed
        total_errors = sum(r.get_error_count() for r in results)
        total_warnings = sum(r.get_warning_count() for r in results)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round((passed / total * 100), 2) if total > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "avg_errors_per_item": round(total_errors / total, 2) if total > 0 else 0
        }
