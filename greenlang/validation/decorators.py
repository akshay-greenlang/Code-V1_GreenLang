"""
GreenLang Validation Decorators
Decorators for easy validation integration.
"""

from typing import Callable, Dict, Any, Optional, List
from functools import wraps
import logging

from .framework import ValidationFramework, ValidationResult
from .schema import SchemaValidator
from .rules import RulesEngine

logger = logging.getLogger(__name__)


class ValidationException(Exception):
    """Exception raised when validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(result.get_summary())


def validate(
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[List[Dict[str, Any]]] = None,
    raise_on_error: bool = True
):
    """
    Decorator to validate function inputs.

    Args:
        schema: JSON Schema dictionary
        rules: List of business rules
        raise_on_error: Raise exception on validation failure

    Example:
        @validate(schema={"type": "object", "required": ["name"]})
        def process_user(data):
            return f"Hello {data['name']}"

        # Raises ValidationException if 'name' is missing
        process_user({"age": 25})
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get first argument as data
            if args:
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            else:
                logger.warning("@validate: No data argument found")
                return func(*args, **kwargs)

            # Create validation framework
            framework = ValidationFramework()

            # Add schema validator if provided
            if schema:
                schema_validator = SchemaValidator(schema)
                framework.add_validator("schema", schema_validator.validate)

            # Add rules validator if provided
            if rules:
                rules_engine = RulesEngine()
                rules_engine.load_rules_from_dict(rules)
                framework.add_validator("rules", rules_engine.validate)

            # Validate
            result = framework.validate(data)

            # Handle result
            if not result.valid:
                logger.error(f"Validation failed in {func.__name__}: {result.get_summary()}")

                if raise_on_error:
                    raise ValidationException(result)
                else:
                    # Attach result to kwargs for function to handle
                    kwargs['_validation_result'] = result

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_schema(schema: Dict[str, Any], raise_on_error: bool = True):
    """
    Decorator to validate inputs against JSON Schema.

    Args:
        schema: JSON Schema dictionary
        raise_on_error: Raise exception on validation failure

    Example:
        @validate_schema({"type": "object", "required": ["email"]})
        def send_email(data):
            return f"Sending to {data['email']}"
    """
    return validate(schema=schema, raise_on_error=raise_on_error)


def validate_rules(rules: List[Dict[str, Any]], raise_on_error: bool = True):
    """
    Decorator to validate inputs against business rules.

    Args:
        rules: List of business rule configurations
        raise_on_error: Raise exception on validation failure

    Example:
        @validate_rules([
            {
                "name": "check_age",
                "field": "age",
                "operator": ">=",
                "value": 18,
                "message": "Must be 18 or older"
            }
        ])
        def register_user(data):
            return f"Registered {data['name']}"
    """
    return validate(rules=rules, raise_on_error=raise_on_error)


def validate_output(schema: Optional[Dict[str, Any]] = None):
    """
    Decorator to validate function outputs.

    Args:
        schema: JSON Schema for output validation

    Example:
        @validate_output(schema={"type": "object", "required": ["status"]})
        def get_status():
            return {"status": "ok", "code": 200}
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if schema:
                validator = SchemaValidator(schema)
                validation_result = validator.validate(result)

                if not validation_result.valid:
                    logger.error(f"Output validation failed in {func.__name__}: {validation_result.get_summary()}")
                    raise ValidationException(validation_result)

            return result

        return wrapper

    return decorator
