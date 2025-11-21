"""
Validation Framework
====================

Schema validation and data integrity checks for GreenLang.

Author: Infrastructure Team
Created: 2025-11-21
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime

from greenlang.infrastructure.base import BaseInfrastructureComponent, InfrastructureConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)


@dataclass
class ValidationRule:
    """A single validation rule."""
    name: str
    rule_type: str  # 'schema', 'range', 'custom'
    config: Dict[str, Any]
    enabled: bool = True
    severity: str = 'error'  # 'error', 'warning'


class ValidationFramework(BaseInfrastructureComponent):
    """
    Framework for validating data against schemas and rules.

    Supports JSON schema validation, custom rules, and multi-rule validation.
    """

    def __init__(self, config: Optional[InfrastructureConfig] = None):
        """Initialize validation framework."""
        super().__init__(config or InfrastructureConfig(component_name="ValidationFramework"))
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_count = 0
        self.error_count = 0

    def _initialize(self) -> None:
        """Initialize validation resources."""
        logger.info("ValidationFramework initialized")

    def start(self) -> None:
        """Start the validation framework."""
        self.status = self.status.RUNNING
        logger.info("ValidationFramework started")

    def stop(self) -> None:
        """Stop the validation framework."""
        self.status = self.status.STOPPED
        logger.info("ValidationFramework stopped")

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a JSON schema for validation."""
        self.schemas[name] = schema
        logger.debug(f"Registered schema: {name}")

    def register_rule(self, rule: ValidationRule) -> None:
        """Register a validation rule."""
        self.rules[rule.name] = rule
        logger.debug(f"Registered rule: {rule.name}")

    def validate(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate data against a schema.

        Args:
            data: Data to validate
            schema: JSON schema to validate against
            schema_name: Name of registered schema to use

        Returns:
            ValidationResult with validation status and any errors
        """
        self.validation_count += 1
        self.update_activity()

        result = ValidationResult(is_valid=True)

        # Get schema
        if schema_name:
            schema = self.schemas.get(schema_name)
            if not schema:
                result.add_error(f"Schema '{schema_name}' not found")
                return result

        if not schema:
            result.add_warning("No schema provided for validation")
            return result

        # Validate against JSON schema
        try:
            jsonschema.validate(instance=data, schema=schema)
            logger.debug("Schema validation passed")
        except jsonschema.ValidationError as e:
            self.error_count += 1
            result.add_error(f"Schema validation failed: {str(e)}")
            logger.warning(f"Validation failed: {str(e)}")
        except Exception as e:
            self.error_count += 1
            result.add_error(f"Validation error: {str(e)}")
            logger.error(f"Unexpected validation error: {str(e)}")

        # Update metrics
        self._metrics.update({
            "validation_count": self.validation_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.validation_count if self.validation_count > 0 else 0
        })

        return result

    def validate_with_rules(self, data: Any, rule_names: List[str]) -> ValidationResult:
        """
        Validate data using multiple named rules.

        Args:
            data: Data to validate
            rule_names: List of rule names to apply

        Returns:
            Combined ValidationResult
        """
        result = ValidationResult(is_valid=True)

        for rule_name in rule_names:
            rule = self.rules.get(rule_name)
            if not rule:
                result.add_warning(f"Rule '{rule_name}' not found")
                continue

            if not rule.enabled:
                continue

            rule_result = self._apply_rule(data, rule)
            if not rule_result.is_valid:
                if rule.severity == 'error':
                    result.errors.extend(rule_result.errors)
                    result.is_valid = False
                else:
                    result.warnings.extend(rule_result.errors)

        return result

    def _apply_rule(self, data: Any, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule."""
        result = ValidationResult(is_valid=True)

        try:
            if rule.rule_type == 'schema':
                return self.validate(data, schema=rule.config.get('schema'))
            elif rule.rule_type == 'range':
                self._validate_range(data, rule.config, result)
            elif rule.rule_type == 'custom':
                self._validate_custom(data, rule.config, result)
            else:
                result.add_warning(f"Unknown rule type: {rule.rule_type}")
        except Exception as e:
            result.add_error(f"Rule '{rule.name}' failed: {str(e)}")

        return result

    def _validate_range(self, data: Any, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate numeric ranges."""
        if not isinstance(data, (int, float)):
            result.add_error(f"Expected numeric value, got {type(data).__name__}")
            return

        min_val = config.get('min')
        max_val = config.get('max')

        if min_val is not None and data < min_val:
            result.add_error(f"Value {data} is below minimum {min_val}")

        if max_val is not None and data > max_val:
            result.add_error(f"Value {data} exceeds maximum {max_val}")

    def _validate_custom(self, data: Any, config: Dict[str, Any], result: ValidationResult) -> None:
        """Apply custom validation logic."""
        # Custom validation can be extended here
        validator_func = config.get('validator')
        if callable(validator_func):
            try:
                if not validator_func(data):
                    result.add_error(config.get('error_message', 'Custom validation failed'))
            except Exception as e:
                result.add_error(f"Custom validator error: {str(e)}")