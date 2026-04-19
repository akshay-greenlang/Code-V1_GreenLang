"""
GreenLang Framework - Validation Engine

Comprehensive validation for agent inputs, outputs, and configurations.
Ensures data quality and compliance with GreenLang standards.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import re
import math


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"        # Must be fixed
    WARNING = "warning"    # Should be addressed
    INFO = "info"          # Informational only


class ValidationType(Enum):
    """Types of validation checks."""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    PHYSICAL = "physical"
    REFERENCE = "reference"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    validation_type: ValidationType
    actual_value: Any = None
    expected: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "validation_type": self.validation_type.value,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "expected": str(self.expected) if self.expected is not None else None,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def error_count(self) -> int:
        """Count of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


@dataclass
class FieldValidator:
    """Validator for a single field."""
    name: str
    required: bool = True
    field_type: Optional[Type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[Set[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    custom_message: Optional[str] = None
    physical_unit: Optional[str] = None
    physical_constraints: Optional[Dict[str, float]] = None


class ValidationEngine:
    """
    Comprehensive validation engine for GreenLang agents.

    Provides:
    - Type validation
    - Range validation
    - Pattern validation
    - Physical constraints (temperature, pressure, etc.)
    - Custom validation rules
    - Cross-field validation

    Usage:
        >>> engine = ValidationEngine()
        >>> engine.add_field("temperature", required=True, min_value=-273.15)
        >>> engine.add_field("pressure", required=True, min_value=0)
        >>> result = engine.validate({"temperature": 100, "pressure": 101325})
    """

    # Physical limits for common quantities
    PHYSICAL_LIMITS = {
        "temperature_K": {"min": 0, "max": 1e8},
        "temperature_C": {"min": -273.15, "max": 1e8},
        "temperature_F": {"min": -459.67, "max": 1e8},
        "pressure_Pa": {"min": 0, "max": 1e12},
        "pressure_bar": {"min": 0, "max": 1e7},
        "mass_kg": {"min": 0, "max": 1e15},
        "energy_J": {"min": 0, "max": 1e20},
        "power_W": {"min": 0, "max": 1e15},
        "efficiency": {"min": 0, "max": 1.0},
        "percentage": {"min": 0, "max": 100},
        "flow_rate_kg_s": {"min": 0, "max": 1e6},
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validation engine.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self._validators: Dict[str, FieldValidator] = {}
        self._cross_validators: List[Callable[[Dict], List[ValidationIssue]]] = []

    def add_field(
        self,
        name: str,
        required: bool = True,
        field_type: Optional[Type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[Set[Any]] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        custom_message: Optional[str] = None,
        physical_unit: Optional[str] = None,
    ) -> "ValidationEngine":
        """
        Add a field validator.

        Returns self for chaining.
        """
        physical_constraints = None
        if physical_unit and physical_unit in self.PHYSICAL_LIMITS:
            physical_constraints = self.PHYSICAL_LIMITS[physical_unit]

        self._validators[name] = FieldValidator(
            name=name,
            required=required,
            field_type=field_type,
            min_value=min_value,
            max_value=max_value,
            pattern=pattern,
            allowed_values=allowed_values,
            custom_validator=custom_validator,
            custom_message=custom_message,
            physical_unit=physical_unit,
            physical_constraints=physical_constraints,
        )
        return self

    def add_cross_validator(
        self,
        validator: Callable[[Dict], List[ValidationIssue]],
    ) -> "ValidationEngine":
        """Add a cross-field validator."""
        self._cross_validators.append(validator)
        return self

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against all registered validators.

        Args:
            data: Dictionary of field names to values

        Returns:
            ValidationResult with issues found
        """
        issues: List[ValidationIssue] = []

        # Validate each registered field
        for name, validator in self._validators.items():
            field_issues = self._validate_field(name, data.get(name), validator)
            issues.extend(field_issues)

        # Run cross-field validators
        for cross_validator in self._cross_validators:
            cross_issues = cross_validator(data)
            issues.extend(cross_issues)

        # Determine validity
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]

        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            validated_data=data if is_valid else None,
        )

    def _validate_field(
        self,
        name: str,
        value: Any,
        validator: FieldValidator,
    ) -> List[ValidationIssue]:
        """Validate a single field."""
        issues: List[ValidationIssue] = []

        # Required check
        if value is None:
            if validator.required:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Required field '{name}' is missing",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.REQUIRED,
                ))
            return issues

        # Type check
        if validator.field_type is not None:
            if not isinstance(value, validator.field_type):
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' must be {validator.field_type.__name__}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.TYPE_CHECK,
                    actual_value=type(value).__name__,
                    expected=validator.field_type.__name__,
                ))
                return issues  # Skip further checks if type is wrong

        # Range checks
        if validator.min_value is not None and isinstance(value, (int, float)):
            if value < validator.min_value:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' must be >= {validator.min_value}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    actual_value=value,
                    expected=f">= {validator.min_value}",
                ))

        if validator.max_value is not None and isinstance(value, (int, float)):
            if value > validator.max_value:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' must be <= {validator.max_value}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    actual_value=value,
                    expected=f"<= {validator.max_value}",
                ))

        # Physical constraints
        if validator.physical_constraints and isinstance(value, (int, float)):
            pc = validator.physical_constraints
            if "min" in pc and value < pc["min"]:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' violates physical limit (min: {pc['min']})",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.PHYSICAL,
                    actual_value=value,
                    expected=f">= {pc['min']} ({validator.physical_unit})",
                ))
            if "max" in pc and value > pc["max"]:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' violates physical limit (max: {pc['max']})",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.PHYSICAL,
                    actual_value=value,
                    expected=f"<= {pc['max']} ({validator.physical_unit})",
                ))

        # Pattern check
        if validator.pattern is not None and isinstance(value, str):
            if not re.match(validator.pattern, value):
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' must match pattern '{validator.pattern}'",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.PATTERN,
                    actual_value=value,
                    expected=validator.pattern,
                ))

        # Allowed values check
        if validator.allowed_values is not None:
            if value not in validator.allowed_values:
                issues.append(ValidationIssue(
                    field=name,
                    message=f"Field '{name}' must be one of {validator.allowed_values}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.REFERENCE,
                    actual_value=value,
                    expected=list(validator.allowed_values),
                ))

        # Custom validation
        if validator.custom_validator is not None:
            if not validator.custom_validator(value):
                message = validator.custom_message or f"Field '{name}' failed custom validation"
                issues.append(ValidationIssue(
                    field=name,
                    message=message,
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.CUSTOM,
                    actual_value=value,
                ))

        return issues

    def validate_numeric_array(
        self,
        name: str,
        values: List[float],
        min_length: int = 1,
        max_length: Optional[int] = None,
        element_min: Optional[float] = None,
        element_max: Optional[float] = None,
    ) -> List[ValidationIssue]:
        """Validate a numeric array."""
        issues: List[ValidationIssue] = []

        if not isinstance(values, (list, tuple)):
            issues.append(ValidationIssue(
                field=name,
                message=f"Field '{name}' must be a list or tuple",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.TYPE_CHECK,
            ))
            return issues

        if len(values) < min_length:
            issues.append(ValidationIssue(
                field=name,
                message=f"Field '{name}' must have at least {min_length} elements",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.RANGE,
                actual_value=len(values),
                expected=f">= {min_length}",
            ))

        if max_length is not None and len(values) > max_length:
            issues.append(ValidationIssue(
                field=name,
                message=f"Field '{name}' must have at most {max_length} elements",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.RANGE,
                actual_value=len(values),
                expected=f"<= {max_length}",
            ))

        for i, val in enumerate(values):
            if not isinstance(val, (int, float)):
                issues.append(ValidationIssue(
                    field=f"{name}[{i}]",
                    message=f"Element {i} of '{name}' must be numeric",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.TYPE_CHECK,
                ))
                continue

            if math.isnan(val) or math.isinf(val):
                issues.append(ValidationIssue(
                    field=f"{name}[{i}]",
                    message=f"Element {i} of '{name}' cannot be NaN or Inf",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                ))

            if element_min is not None and val < element_min:
                issues.append(ValidationIssue(
                    field=f"{name}[{i}]",
                    message=f"Element {i} of '{name}' must be >= {element_min}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    actual_value=val,
                    expected=f">= {element_min}",
                ))

            if element_max is not None and val > element_max:
                issues.append(ValidationIssue(
                    field=f"{name}[{i}]",
                    message=f"Element {i} of '{name}' must be <= {element_max}",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.RANGE,
                    actual_value=val,
                    expected=f"<= {element_max}",
                ))

        return issues


# Pre-built validators for common GreenLang patterns
class GreenLangValidators:
    """Pre-built validators for GreenLang agent patterns."""

    @staticmethod
    def temperature_validator(unit: str = "C") -> ValidationEngine:
        """Create validator for temperature fields."""
        engine = ValidationEngine()
        physical_unit = f"temperature_{unit}"
        engine.add_field(
            "temperature",
            required=True,
            field_type=(int, float),
            physical_unit=physical_unit,
        )
        return engine

    @staticmethod
    def stream_validator() -> ValidationEngine:
        """Create validator for thermal stream data."""
        engine = ValidationEngine()
        engine.add_field("name", required=True, field_type=str)
        engine.add_field("inlet_temp", required=True, field_type=(int, float),
                        min_value=-273.15)
        engine.add_field("outlet_temp", required=True, field_type=(int, float),
                        min_value=-273.15)
        engine.add_field("heat_capacity_rate", required=True, field_type=(int, float),
                        min_value=0)

        # Cross-validator: outlet must differ from inlet
        def validate_temps(data: Dict) -> List[ValidationIssue]:
            issues = []
            inlet = data.get("inlet_temp")
            outlet = data.get("outlet_temp")
            if inlet is not None and outlet is not None:
                if abs(inlet - outlet) < 0.001:
                    issues.append(ValidationIssue(
                        field="inlet_temp,outlet_temp",
                        message="Inlet and outlet temperatures must differ",
                        severity=ValidationSeverity.WARNING,
                        validation_type=ValidationType.CUSTOM,
                    ))
            return issues

        engine.add_cross_validator(validate_temps)
        return engine

    @staticmethod
    def efficiency_validator() -> ValidationEngine:
        """Create validator for efficiency values."""
        engine = ValidationEngine()
        engine.add_field(
            "efficiency",
            required=True,
            field_type=(int, float),
            min_value=0,
            max_value=1.0,
            physical_unit="efficiency",
        )
        return engine

    @staticmethod
    def emission_factor_validator() -> ValidationEngine:
        """Create validator for emission factors."""
        engine = ValidationEngine()
        engine.add_field("factor", required=True, field_type=(int, float), min_value=0)
        engine.add_field("unit", required=True, field_type=str)
        engine.add_field("source", required=True, field_type=str)
        engine.add_field("year", required=False, field_type=int, min_value=1990, max_value=2100)
        return engine
