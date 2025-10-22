"""
GreenLang Business Rules Engine
Flexible rule-based validation system.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
import operator
from datetime import datetime

from .framework import ValidationResult, ValidationError, ValidationSeverity

logger = logging.getLogger(__name__)


class RuleOperator(str, Enum):
    """Supported comparison operators."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    REGEX = "regex"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class Rule(BaseModel):
    """A single validation rule."""
    name: str = Field(..., description="Rule name")
    field: str = Field(..., description="Field to validate")
    operator: RuleOperator = Field(..., description="Comparison operator")
    value: Any = Field(default=None, description="Expected value")
    message: Optional[str] = Field(default=None, description="Custom error message")
    severity: ValidationSeverity = Field(default=ValidationSeverity.ERROR, description="Error severity")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    condition: Optional[str] = Field(default=None, description="Conditional expression")

    class Config:
        use_enum_values = True


class RuleSet(BaseModel):
    """A collection of related rules."""
    name: str = Field(..., description="Rule set name")
    rules: List[Rule] = Field(default_factory=list, description="List of rules")
    description: Optional[str] = Field(default=None, description="Rule set description")
    enabled: bool = Field(default=True, description="Whether rule set is enabled")


class RulesEngine:
    """
    Business rules validation engine.

    Supports:
    - Multiple comparison operators
    - Conditional rules
    - Rule sets for organization
    - Custom error messages
    - Configurable severity levels

    Example:
        engine = RulesEngine()

        # Add rule
        rule = Rule(
            name="check_age",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18,
            message="Age must be 18 or older"
        )
        engine.add_rule(rule)

        # Validate
        result = engine.validate({"age": 25})
    """

    # Operator functions
    OPERATORS = {
        RuleOperator.EQUALS: operator.eq,
        RuleOperator.NOT_EQUALS: operator.ne,
        RuleOperator.GREATER_THAN: operator.gt,
        RuleOperator.GREATER_EQUAL: operator.ge,
        RuleOperator.LESS_THAN: operator.lt,
        RuleOperator.LESS_EQUAL: operator.le,
    }

    def __init__(self):
        """Initialize rules engine."""
        self.rules: List[Rule] = []
        self.rule_sets: Dict[str, RuleSet] = {}

    def add_rule(self, rule: Rule):
        """
        Add a validation rule.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
        logger.debug(f"Added rule: {rule.name}")

    def add_rule_set(self, rule_set: RuleSet):
        """
        Add a rule set.

        Args:
            rule_set: RuleSet to add
        """
        self.rule_sets[rule_set.name] = rule_set
        logger.debug(f"Added rule set: {rule_set.name}")

    def remove_rule(self, rule_name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]

    def get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        Get value from nested field path (e.g., 'user.address.city').

        Args:
            data: Data dictionary
            field_path: Field path with dots

        Returns:
            Field value or None if not found
        """
        parts = field_path.split('.')
        value = data

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def evaluate_condition(self, data: Dict[str, Any], condition: str) -> bool:
        """
        Evaluate a conditional expression.

        Args:
            data: Data dictionary
            condition: Condition string (simplified evaluation)

        Returns:
            True if condition is met, False otherwise
        """
        # Simple condition evaluation - can be extended
        # For now, just check if field exists
        if "exists:" in condition:
            field = condition.replace("exists:", "").strip()
            return self.get_field_value(data, field) is not None

        return True

    def evaluate_rule(self, rule: Rule, data: Dict[str, Any]) -> Optional[ValidationError]:
        """
        Evaluate a single rule against data.

        Args:
            rule: Rule to evaluate
            data: Data to validate

        Returns:
            ValidationError if rule fails, None if passes
        """
        # Check if rule is enabled
        if not rule.enabled:
            return None

        # Check condition
        if rule.condition and not self.evaluate_condition(data, rule.condition):
            return None

        # Get field value
        field_value = self.get_field_value(data, rule.field)

        # Evaluate based on operator
        passed = False

        if rule.operator in self.OPERATORS:
            # Standard comparison operators
            op_func = self.OPERATORS[rule.operator]
            try:
                passed = op_func(field_value, rule.value)
            except (TypeError, ValueError) as e:
                logger.warning(f"Comparison failed for rule {rule.name}: {str(e)}")
                passed = False

        elif rule.operator == RuleOperator.IN:
            passed = field_value in rule.value if rule.value else False

        elif rule.operator == RuleOperator.NOT_IN:
            passed = field_value not in rule.value if rule.value else True

        elif rule.operator == RuleOperator.CONTAINS:
            passed = rule.value in field_value if field_value else False

        elif rule.operator == RuleOperator.IS_NULL:
            passed = field_value is None

        elif rule.operator == RuleOperator.NOT_NULL:
            passed = field_value is not None

        elif rule.operator == RuleOperator.REGEX:
            import re
            pattern = rule.value
            passed = bool(re.match(pattern, str(field_value))) if field_value else False

        # Create error if failed
        if not passed:
            message = rule.message or f"Rule '{rule.name}' failed for field '{rule.field}'"
            return ValidationError(
                field=rule.field,
                message=message,
                severity=rule.severity,
                validator="business_rules",
                value=field_value,
                expected=rule.value,
                location=f"$.{rule.field}"
            )

        return None

    def validate(self, data: Dict[str, Any], rule_set_name: Optional[str] = None) -> ValidationResult:
        """
        Validate data against rules.

        Args:
            data: Data to validate
            rule_set_name: Optional rule set name to use (uses all rules if None)

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Determine which rules to use
        if rule_set_name:
            if rule_set_name not in self.rule_sets:
                error = ValidationError(
                    field="__framework__",
                    message=f"Rule set '{rule_set_name}' not found",
                    severity=ValidationSeverity.ERROR,
                    validator="business_rules"
                )
                result.add_error(error)
                return result

            rule_set = self.rule_sets[rule_set_name]
            if not rule_set.enabled:
                result.metadata["skipped"] = f"Rule set '{rule_set_name}' is disabled"
                return result

            rules_to_check = rule_set.rules
        else:
            rules_to_check = self.rules

        # Evaluate each rule
        for rule in rules_to_check:
            try:
                error = self.evaluate_rule(rule, data)
                if error:
                    result.add_error(error)
            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.name}: {str(e)}", exc_info=True)
                error = ValidationError(
                    field=rule.field,
                    message=f"Rule evaluation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    validator="business_rules"
                )
                result.add_error(error)

        result.metadata["rules_evaluated"] = len(rules_to_check)
        return result

    def load_rules_from_dict(self, rules_config: List[Dict[str, Any]]):
        """
        Load rules from configuration dictionary.

        Args:
            rules_config: List of rule configurations
        """
        for rule_dict in rules_config:
            rule = Rule(**rule_dict)
            self.add_rule(rule)

    def get_rule_names(self) -> List[str]:
        """Get list of all rule names."""
        return [rule.name for rule in self.rules]

    def enable_rule(self, rule_name: str):
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True

    def disable_rule(self, rule_name: str):
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
