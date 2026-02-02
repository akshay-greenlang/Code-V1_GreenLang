"""
Comprehensive data validation for GreenLang pipelines.

Provides validation decorators, schema enforcement, and input sanitization
for all data entry points and agent methods.
"""

import logging
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Type, Union, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import json
from pydantic import BaseModel, ValidationError, validator, Field
import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"       # Fail on any validation error
    WARNING = "warning"     # Log warnings but continue
    LENIENT = "lenient"     # Attempt to fix/coerce data


class DataType(Enum):
    """Common data types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    LIST = "list"
    DICT = "dict"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationError(Exception):
    """Custom validation error with details."""
    def __init__(self, message: str, errors: List[str], data: Any = None):
        self.message = message
        self.errors = errors
        self.data = data
        super().__init__(message)


class DataValidator:
    """
    Core data validator with comprehensive validation rules.

    Features:
    - Type validation and coercion
    - Range and boundary checks
    - Pattern matching
    - Business rule validation
    - Schema validation (JSON Schema, Pydantic)
    """

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STRICT,
        custom_rules: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize data validator.

        Args:
            level: Validation strictness level
            custom_rules: Custom validation rules
        """
        self.level = level
        self.custom_rules = custom_rules or {}

    def validate_type(
        self,
        value: Any,
        expected_type: Union[type, DataType],
        coerce: bool = False
    ) -> ValidationResult:
        """
        Validate data type with optional coercion.

        Args:
            value: Value to validate
            expected_type: Expected type or DataType enum
            coerce: Whether to attempt type coercion

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)

        # Handle DataType enum
        if isinstance(expected_type, DataType):
            if expected_type == DataType.STRING:
                expected_type = str
            elif expected_type == DataType.INTEGER:
                expected_type = int
            elif expected_type == DataType.FLOAT:
                expected_type = float
            elif expected_type == DataType.BOOLEAN:
                expected_type = bool
            elif expected_type == DataType.LIST:
                expected_type = list
            elif expected_type == DataType.DICT:
                expected_type = dict

        # Check type
        if not isinstance(value, expected_type):
            if coerce and self.level != ValidationLevel.STRICT:
                try:
                    result.cleaned_data = expected_type(value)
                    result.warnings.append(f"Coerced {type(value).__name__} to {expected_type.__name__}")
                except (ValueError, TypeError) as e:
                    result.is_valid = False
                    result.errors.append(f"Cannot coerce {value} to {expected_type.__name__}: {e}")
            else:
                result.is_valid = False
                result.errors.append(
                    f"Expected type {expected_type.__name__}, got {type(value).__name__}"
                )
        else:
            result.cleaned_data = value

        return result

    def validate_range(
        self,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> ValidationResult:
        """
        Validate numeric value is within range.

        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, cleaned_data=value)

        if min_value is not None and value < min_value:
            result.is_valid = False
            result.errors.append(f"Value {value} is below minimum {min_value}")

        if max_value is not None and value > max_value:
            result.is_valid = False
            result.errors.append(f"Value {value} exceeds maximum {max_value}")

        return result

    def validate_pattern(
        self,
        value: str,
        pattern: str,
        error_message: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate string matches regex pattern.

        Args:
            value: String to validate
            pattern: Regex pattern
            error_message: Custom error message

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, cleaned_data=value)

        if not re.match(pattern, value):
            result.is_valid = False
            error = error_message or f"Value '{value}' does not match pattern '{pattern}'"
            result.errors.append(error)

        return result

    def validate_email(self, email: str) -> ValidationResult:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return self.validate_pattern(
            email,
            pattern,
            f"Invalid email address: {email}"
        )

    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL format."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return self.validate_pattern(
            url,
            pattern,
            f"Invalid URL: {url}"
        )

    def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Union[Dict[str, Any], Type[BaseModel]]
    ) -> ValidationResult:
        """
        Validate data against JSON Schema or Pydantic model.

        Args:
            data: Data to validate
            schema: JSON Schema dict or Pydantic model class

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)

        # Pydantic model validation
        if inspect.isclass(schema) and issubclass(schema, BaseModel):
            try:
                validated = schema(**data)
                result.cleaned_data = validated.dict()
            except ValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    result.errors.append(
                        f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
                    )

        # JSON Schema validation
        else:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(data))

            if errors:
                result.is_valid = False
                for error in errors:
                    path = '.'.join(str(p) for p in error.path)
                    result.errors.append(f"{path}: {error.message}" if path else error.message)
            else:
                result.cleaned_data = data

        return result

    def validate_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str]
    ) -> ValidationResult:
        """
        Validate required fields are present and non-empty.

        Args:
            data: Data dictionary
            required_fields: List of required field names

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, cleaned_data=data)

        for field in required_fields:
            if field not in data:
                result.is_valid = False
                result.errors.append(f"Required field missing: {field}")
            elif data[field] is None or data[field] == "":
                result.is_valid = False
                result.errors.append(f"Required field empty: {field}")

        return result

    def validate_business_rules(
        self,
        data: Any,
        rules: List[Callable[[Any], bool]],
        rule_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate data against business rules.

        Args:
            data: Data to validate
            rules: List of validation functions
            rule_names: Optional names for rules

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, cleaned_data=data)

        for i, rule in enumerate(rules):
            try:
                if not rule(data):
                    result.is_valid = False
                    rule_name = rule_names[i] if rule_names and i < len(rule_names) else f"Rule {i+1}"
                    result.errors.append(f"Business rule violation: {rule_name}")
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Error evaluating rule {i+1}: {e}")

        return result


# Validation Decorators

def validate_input(
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    required_fields: Optional[List[str]] = None,
    validation_level: ValidationLevel = ValidationLevel.STRICT,
    parameter_name: str = "data"
):
    """
    Decorator for validating function input.

    Args:
        schema: JSON Schema or Pydantic model
        required_fields: Required field names
        validation_level: Validation strictness
        parameter_name: Name of parameter to validate

    Example:
        @validate_input(
            schema=ShipmentSchema,
            required_fields=["shipment_id", "origin", "destination"]
        )
        def process_shipment(data):
            # data is guaranteed to be valid here
            return calculate_emissions(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the parameter to validate
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if parameter_name not in bound_args.arguments:
                raise ValueError(f"Parameter '{parameter_name}' not found in function signature")

            data = bound_args.arguments[parameter_name]

            # Create validator
            validator = DataValidator(level=validation_level)

            # Validate schema
            if schema:
                result = validator.validate_schema(data, schema)
                if not result.is_valid:
                    if validation_level == ValidationLevel.STRICT:
                        raise ValidationError(
                            f"Schema validation failed for {func.__name__}",
                            result.errors,
                            data
                        )
                    elif validation_level == ValidationLevel.WARNING:
                        for error in result.errors:
                            logger.warning(f"Validation warning in {func.__name__}: {error}")
                    # Update data with cleaned version if available
                    if result.cleaned_data is not None:
                        bound_args.arguments[parameter_name] = result.cleaned_data

            # Validate required fields
            if required_fields and isinstance(data, dict):
                result = validator.validate_required_fields(data, required_fields)
                if not result.is_valid:
                    if validation_level == ValidationLevel.STRICT:
                        raise ValidationError(
                            f"Required fields validation failed for {func.__name__}",
                            result.errors,
                            data
                        )
                    elif validation_level == ValidationLevel.WARNING:
                        for error in result.errors:
                            logger.warning(f"Validation warning in {func.__name__}: {error}")

            # Call function with validated data
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper
    return decorator


def validate_output(
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    validation_level: ValidationLevel = ValidationLevel.WARNING
):
    """
    Decorator for validating function output.

    Args:
        schema: JSON Schema or Pydantic model for output
        validation_level: Validation strictness

    Example:
        @validate_output(schema=EmissionsResultSchema)
        def calculate_emissions(data):
            # Calculate emissions
            return result  # Will be validated before return
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if schema:
                validator = DataValidator(level=validation_level)
                validation_result = validator.validate_schema(result, schema)

                if not validation_result.is_valid:
                    if validation_level == ValidationLevel.STRICT:
                        raise ValidationError(
                            f"Output validation failed for {func.__name__}",
                            validation_result.errors,
                            result
                        )
                    else:
                        for error in validation_result.errors:
                            logger.warning(f"Output validation warning in {func.__name__}: {error}")

                # Return cleaned data if available
                if validation_result.cleaned_data is not None:
                    return validation_result.cleaned_data

            return result

        return wrapper
    return decorator


def validate_type_hints(validation_level: ValidationLevel = ValidationLevel.STRICT):
    """
    Decorator that validates function arguments against type hints.

    Example:
        @validate_type_hints()
        def process_data(
            id: int,
            name: str,
            amount: float,
            active: bool = True
        ):
            # All parameters are validated against their type hints
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            type_hints = get_type_hints(func)
            validator = DataValidator(level=validation_level)

            for param_name, param_value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]

                    # Skip Optional types for None values
                    if param_value is None:
                        continue

                    # Basic type validation
                    if expected_type in (int, str, float, bool, list, dict):
                        result = validator.validate_type(
                            param_value,
                            expected_type,
                            coerce=(validation_level == ValidationLevel.LENIENT)
                        )

                        if not result.is_valid:
                            if validation_level == ValidationLevel.STRICT:
                                raise ValidationError(
                                    f"Type validation failed for parameter '{param_name}' in {func.__name__}",
                                    result.errors,
                                    param_value
                                )
                            else:
                                for error in result.errors:
                                    logger.warning(f"Type validation warning: {error}")

                        # Update with cleaned value if coerced
                        if result.cleaned_data is not None and result.cleaned_data != param_value:
                            bound_args.arguments[param_name] = result.cleaned_data

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper
    return decorator


# Pydantic Models for Common Validation Scenarios

class EmissionFactorSchema(BaseModel):
    """Schema for emission factor data."""
    factor_id: str = Field(..., min_length=1)
    category: str
    value: float = Field(..., ge=0)
    unit: str
    source: str
    year: int = Field(..., ge=2000, le=2100)
    confidence_level: float = Field(..., ge=0, le=1)

    @validator('unit')
    def validate_unit(cls, v):
        valid_units = ['kgCO2e/kg', 'kgCO2e/kWh', 'kgCO2e/m3', 'kgCO2e/km']
        if v not in valid_units:
            raise ValueError(f"Unit must be one of {valid_units}")
        return v


class ShipmentSchema(BaseModel):
    """Schema for shipment data."""
    shipment_id: str = Field(..., min_length=1)
    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    weight: float = Field(..., gt=0)
    volume: Optional[float] = Field(None, gt=0)
    transport_mode: str
    departure_date: datetime
    arrival_date: Optional[datetime] = None

    @validator('transport_mode')
    def validate_transport(cls, v):
        valid_modes = ['road', 'rail', 'air', 'sea', 'multimodal']
        if v.lower() not in valid_modes:
            raise ValueError(f"Transport mode must be one of {valid_modes}")
        return v.lower()

    @validator('arrival_date')
    def validate_dates(cls, v, values):
        if v and 'departure_date' in values and v < values['departure_date']:
            raise ValueError("Arrival date cannot be before departure date")
        return v


class SupplierDataSchema(BaseModel):
    """Schema for supplier data."""
    supplier_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    country: str = Field(..., regex='^[A-Z]{2}$')  # ISO country code
    emissions_data: Optional[Dict[str, float]] = None
    certifications: List[str] = Field(default_factory=list)
    data_quality_score: float = Field(..., ge=0, le=100)


# Integration helper for pipeline validation

class PipelineValidator:
    """
    Comprehensive validator for data pipeline entry points.

    Ensures all data entering the pipeline is valid and consistent.
    """

    def __init__(
        self,
        schemas: Dict[str, Union[Dict, Type[BaseModel]]],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        Initialize pipeline validator.

        Args:
            schemas: Dictionary of schemas for different data types
            validation_level: Validation strictness level
        """
        self.schemas = schemas
        self.validator = DataValidator(level=validation_level)
        self.validation_stats = {
            "total_validations": 0,
            "successful": 0,
            "failed": 0,
            "warnings": 0
        }

    def validate_entry(
        self,
        data: Any,
        data_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate data entering the pipeline.

        Args:
            data: Data to validate
            data_type: Type of data (key in schemas dict)
            context: Additional context for validation

        Returns:
            Validation result

        Example:
            validator = PipelineValidator(schemas={
                "shipment": ShipmentSchema,
                "supplier": SupplierDataSchema
            })

            result = validator.validate_entry(
                data=shipment_data,
                data_type="shipment"
            )

            if not result.is_valid:
                raise ValidationError("Invalid shipment data", result.errors)
        """
        self.validation_stats["total_validations"] += 1

        if data_type not in self.schemas:
            result = ValidationResult(
                is_valid=False,
                errors=[f"Unknown data type: {data_type}"]
            )
            self.validation_stats["failed"] += 1
            return result

        schema = self.schemas[data_type]
        result = self.validator.validate_schema(data, schema)

        # Update statistics
        if result.is_valid:
            self.validation_stats["successful"] += 1
        else:
            self.validation_stats["failed"] += 1

        if result.warnings:
            self.validation_stats["warnings"] += len(result.warnings)

        # Log validation
        logger.info(
            f"Validated {data_type} data: "
            f"valid={result.is_valid}, "
            f"errors={len(result.errors)}, "
            f"warnings={len(result.warnings)}"
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()