# -*- coding: utf-8 -*-
"""
Rule Engine Integration for GL-FOUND-X-002.

This module integrates with the existing PolicyEngine DSL for cross-field
rule evaluation in the GreenLang Schema Validator.

Features:
    - Expression-based rule evaluation
    - Conditional requirements (if X then Y required)
    - Consistency checks (sum of components = total)
    - Range dependencies
    - All standard operators (==, !=, >, <, >=, <=, in, not_in, contains, regex)
    - Logical operators (and, or, not)
    - Arithmetic operators (sum, +, -, *, /)
    - JSON Pointer path resolution

Error Codes:
    - GLSCHEMA-E400: RULE_VIOLATION - Cross-field rule failed
    - GLSCHEMA-E401: CONDITIONAL_REQUIRED - Conditional requirement not met
    - GLSCHEMA-E402: CONSISTENCY_ERROR - Consistency check failed

Example:
    >>> from greenlang.schema.validator.rules import RuleValidator, Rule
    >>> validator = RuleValidator(ir, options)
    >>> findings = validator.validate(payload)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.4
"""

from __future__ import annotations

import hashlib
import logging
import math
import operator
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schema.compiler.ir import RuleBindingIR, SchemaIR
from greenlang.schema.models.config import ValidationOptions
from greenlang.schema.models.finding import Finding, FindingHint, Severity

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR CODE CONSTANTS
# =============================================================================

ERROR_CODE_RULE_VIOLATION = "GLSCHEMA-E400"
ERROR_CODE_CONDITIONAL_REQUIRED = "GLSCHEMA-E401"
ERROR_CODE_CONSISTENCY_ERROR = "GLSCHEMA-E402"


# =============================================================================
# RULE EXPRESSION MODELS
# =============================================================================


class RuleExpression(BaseModel):
    """
    A rule expression to evaluate.

    Rule expressions can be simple field comparisons or complex
    nested logical expressions. They support:
    - Field access via JSON Pointer paths
    - Comparison operators (==, !=, >, <, >=, <=)
    - Membership operators (in, not_in, contains)
    - String operators (starts_with, ends_with, matches/regex)
    - Null checks (is_null, is_not_null, exists)
    - Logical operators (and, or, not)
    - Arithmetic (sum, +, -, *, /)

    Attributes:
        operator: The operation to perform (e.g., "==", "and", "exists")
        field: JSON Pointer path for field access (e.g., "/fuel_type")
        value: Literal value for comparison
        operands: List of sub-expressions for logical/aggregate operators
        fields: List of field paths for aggregate operations like sum

    Example:
        >>> # Simple comparison
        >>> expr = RuleExpression(
        ...     operator="==",
        ...     field="/fuel_type",
        ...     value="gas"
        ... )
        >>> # Logical expression
        >>> expr = RuleExpression(
        ...     operator="and",
        ...     operands=[
        ...         RuleExpression(operator=">=", field="/value", value=0),
        ...         RuleExpression(operator="<=", field="/value", value=100)
        ...     ]
        ... )
    """

    operator: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="The operation to perform"
    )
    field: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="JSON Pointer path for field access"
    )
    value: Optional[Any] = Field(
        default=None,
        description="Literal value for comparison"
    )
    operands: List["RuleExpression"] = Field(
        default_factory=list,
        description="Sub-expressions for logical/aggregate operators"
    )
    fields: List[str] = Field(
        default_factory=list,
        description="List of field paths for aggregate operations"
    )

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: Optional[str]) -> Optional[str]:
        """Validate field is a valid JSON Pointer."""
        if v is None:
            return v
        # JSON Pointer must start with / or be empty
        if v and not v.startswith("/"):
            raise ValueError(
                f"Invalid JSON Pointer '{v}'. Must start with '/' (e.g., '/field_name')"
            )
        return v

    def is_comparison(self) -> bool:
        """Check if this is a simple comparison expression."""
        return self.operator in COMPARISON_OPERATORS

    def is_logical(self) -> bool:
        """Check if this is a logical expression (and/or/not)."""
        return self.operator in ("and", "or", "not")

    def is_aggregate(self) -> bool:
        """Check if this is an aggregate expression (sum, count, etc.)."""
        return self.operator in ("sum", "count", "min", "max", "avg")

    def is_null_check(self) -> bool:
        """Check if this is a null/existence check."""
        return self.operator in ("is_null", "is_not_null", "exists", "not_exists")


class Rule(BaseModel):
    """
    A cross-field validation rule.

    Rules define validation logic that spans multiple fields. Each rule
    has an optional condition (when) and a check expression. The rule
    fires only when the condition is met (or always if no condition).

    Attributes:
        rule_id: Unique identifier for this rule
        severity: Severity level (error, warning, info)
        when: Optional condition expression (rule applies only if true)
        check: The validation expression (must evaluate to true to pass)
        message: Static error message if rule fails
        message_template: Dynamic message with {{ path }} placeholders

    Example:
        >>> rule = Rule(
        ...     rule_id="methane_slip_required",
        ...     severity="error",
        ...     when=RuleExpression(operator="==", field="/fuel_type", value="gas"),
        ...     check=RuleExpression(operator="exists", field="/methane_slip"),
        ...     message="methane_slip is required when fuel_type is 'gas'"
        ... )
    """

    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for this rule"
    )
    severity: str = Field(
        default="error",
        description="Severity level: error, warning, or info"
    )
    when: Optional[RuleExpression] = Field(
        default=None,
        description="Condition for when rule applies"
    )
    check: RuleExpression = Field(
        ...,
        description="The validation expression to check"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Error message if rule fails"
    )
    message_template: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Dynamic message template with {{ path }} placeholders"
    )

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is a known value."""
        valid = {"error", "warning", "info"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid severity '{v}'. Must be one of: {valid}")
        return v.lower()

    def is_conditional(self) -> bool:
        """Check if this rule has a condition."""
        return self.when is not None

    def get_error_code(self) -> str:
        """
        Get the appropriate error code for this rule.

        Returns error code based on rule type:
        - Conditional requirements get GLSCHEMA-E401
        - Consistency checks (sum/equals) get GLSCHEMA-E402
        - Other rules get GLSCHEMA-E400
        """
        if self.is_conditional():
            if self.check.is_null_check():
                return ERROR_CODE_CONDITIONAL_REQUIRED
        if self.check.operator in ("==", "!=") and self.check.operands:
            # Check if this looks like a consistency check (sum comparison)
            for op in self.check.operands:
                if op.is_aggregate():
                    return ERROR_CODE_CONSISTENCY_ERROR
        return ERROR_CODE_RULE_VIOLATION


# Rebuild model for forward reference resolution
RuleExpression.model_rebuild()


# =============================================================================
# OPERATOR DEFINITIONS
# =============================================================================

# Comparison operators
COMPARISON_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}

# Extended operators for rule expressions
EXTENDED_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "in": lambda a, b: a in b if b is not None else False,
    "not_in": lambda a, b: a not in b if b is not None else True,
    "contains": lambda a, b: b in a if a is not None else False,
    "starts_with": lambda a, b: str(a).startswith(str(b)) if a is not None else False,
    "ends_with": lambda a, b: str(a).endswith(str(b)) if a is not None else False,
    "matches": lambda a, b: bool(re.match(b, str(a))) if a is not None else False,
    "regex": lambda a, b: bool(re.match(b, str(a))) if a is not None else False,
}

# Null check operators (unary)
NULL_CHECK_OPERATORS: Dict[str, Callable[[Any], bool]] = {
    "is_null": lambda a: a is None,
    "is_not_null": lambda a: a is not None,
    "exists": lambda a: a is not None,
    "not_exists": lambda a: a is None,
}

# Arithmetic operators
ARITHMETIC_OPERATORS: Dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: a / b if b != 0 else float('inf'),
}


# =============================================================================
# EXPRESSION EVALUATOR
# =============================================================================


class ExpressionEvaluator:
    """
    Evaluates rule expressions against payloads.

    This evaluator processes rule expressions and returns boolean results
    for conditions or computed values for aggregate operations. It integrates
    patterns from the PolicyEngine for consistent expression evaluation.

    Supported operators:
        - Comparison: ==, !=, >, <, >=, <=
        - Membership: in, not_in, contains
        - String: starts_with, ends_with, matches, regex
        - Null checks: is_null, is_not_null, exists, not_exists
        - Logical: and, or, not
        - Aggregate: sum, count, min, max, avg
        - Arithmetic: +, -, *, /

    Attributes:
        OPERATORS: Combined operator dictionary

    Example:
        >>> evaluator = ExpressionEvaluator()
        >>> payload = {"fuel_type": "gas", "methane_slip": 0.5}
        >>> expr = RuleExpression(operator="==", field="/fuel_type", value="gas")
        >>> result = evaluator.evaluate(expr, payload)
        >>> print(result)  # True
    """

    # All comparison operators (binary)
    OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
        **COMPARISON_OPERATORS,
        **EXTENDED_OPERATORS,
    }

    def __init__(self) -> None:
        """Initialize the expression evaluator."""
        self._evaluation_depth = 0
        self._max_depth = 50  # Prevent stack overflow

    def evaluate(
        self,
        expression: RuleExpression,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Evaluate expression against payload.

        This is the main entry point for expression evaluation. It handles
        all expression types including comparisons, logical operations,
        aggregations, and null checks.

        Args:
            expression: The rule expression to evaluate
            payload: The data payload to evaluate against

        Returns:
            Boolean for condition expressions, computed value for aggregates

        Raises:
            EvaluationError: If expression evaluation fails
            RecursionError: If expression nesting exceeds max depth

        Example:
            >>> expr = RuleExpression(
            ...     operator="and",
            ...     operands=[
            ...         RuleExpression(operator=">=", field="/value", value=0),
            ...         RuleExpression(operator="<=", field="/value", value=100)
            ...     ]
            ... )
            >>> evaluator.evaluate(expr, {"value": 50})
            True
        """
        self._evaluation_depth += 1
        try:
            if self._evaluation_depth > self._max_depth:
                raise RecursionError(
                    f"Expression evaluation depth ({self._evaluation_depth}) "
                    f"exceeds maximum ({self._max_depth})"
                )

            op = expression.operator.lower()

            # Handle logical operators
            if op in ("and", "or", "not"):
                return self._evaluate_logical(op, expression.operands, payload)

            # Handle null checks
            if op in NULL_CHECK_OPERATORS:
                return self._evaluate_null_check(expression, payload)

            # Handle aggregations
            if op in ("sum", "count", "min", "max", "avg"):
                return self._evaluate_aggregate(expression, payload)

            # Handle arithmetic
            if op in ARITHMETIC_OPERATORS:
                return self._evaluate_arithmetic(expression, payload)

            # Handle comparisons
            if op in self.OPERATORS:
                return self._evaluate_comparison_expr(expression, payload)

            # Handle value/literal operator (returns field value or literal)
            if op in ("value", "literal", "field"):
                if expression.field:
                    return self._get_value_at_path(payload, expression.field)
                return expression.value

            # Unknown operator
            logger.warning(f"Unknown operator '{op}' in expression")
            return False

        finally:
            self._evaluation_depth -= 1

    def _get_value_at_path(
        self,
        payload: Dict[str, Any],
        path: str
    ) -> Any:
        """
        Get value at JSON Pointer path.

        Implements RFC 6901 JSON Pointer resolution, supporting:
        - Object property access: /property
        - Array index access: /array/0
        - Nested paths: /parent/child/grandchild
        - Escaped characters: ~0 for ~ and ~1 for /

        Args:
            payload: The data payload
            path: JSON Pointer path (e.g., "/field/subfield")

        Returns:
            Value at the path, or None if path doesn't exist

        Example:
            >>> payload = {"user": {"name": "Alice", "scores": [90, 85, 88]}}
            >>> self._get_value_at_path(payload, "/user/name")
            'Alice'
            >>> self._get_value_at_path(payload, "/user/scores/1")
            85
        """
        if not path:
            return payload

        # Handle root reference
        if path == "/":
            return payload

        # JSON Pointer must start with /
        if not path.startswith("/"):
            logger.warning(f"Invalid JSON Pointer '{path}' - must start with '/'")
            return None

        # Split path and resolve
        parts = path[1:].split("/")  # Skip leading /
        value = payload

        for part in parts:
            if value is None:
                return None

            # Handle JSON Pointer escape sequences
            # ~1 -> / and ~0 -> ~
            part = part.replace("~1", "/").replace("~0", "~")

            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                # Try to parse as array index
                try:
                    idx = int(part)
                    if 0 <= idx < len(value):
                        value = value[idx]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                return None

        return value

    def _evaluate_comparison(
        self,
        op: str,
        left: Any,
        right: Any
    ) -> bool:
        """
        Evaluate a comparison operation.

        Performs type-safe comparison with null handling.
        For numeric comparisons with None, returns False.

        Args:
            op: Operator string (==, !=, >, etc.)
            left: Left operand
            right: Right operand

        Returns:
            Boolean comparison result

        Example:
            >>> self._evaluate_comparison(">=", 10, 5)
            True
            >>> self._evaluate_comparison("in", "apple", ["apple", "banana"])
            True
        """
        # Get operator function
        op_func = self.OPERATORS.get(op)
        if op_func is None:
            logger.warning(f"Unknown comparison operator: {op}")
            return False

        # Handle None cases for numeric comparisons
        if left is None or right is None:
            if op in ("==", "!="):
                return op_func(left, right)
            # For <, >, <=, >= with None, return False
            if op in (">", "<", ">=", "<="):
                return False
            # For membership operators, let the lambda handle it
            pass

        try:
            return op_func(left, right)
        except (TypeError, ValueError) as e:
            logger.debug(f"Comparison error ({op}): {e}")
            return False

    def _evaluate_comparison_expr(
        self,
        expression: RuleExpression,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a comparison expression.

        Handles expressions with field references, literal values,
        and nested expressions as operands.

        Args:
            expression: Comparison expression
            payload: Data payload

        Returns:
            Boolean comparison result
        """
        op = expression.operator.lower()

        # Determine left operand
        if expression.field:
            left = self._get_value_at_path(payload, expression.field)
        elif expression.operands and len(expression.operands) >= 1:
            left = self.evaluate(expression.operands[0], payload)
        else:
            left = None

        # Determine right operand
        if expression.value is not None:
            right = expression.value
        elif expression.operands and len(expression.operands) >= 2:
            right = self.evaluate(expression.operands[1], payload)
        else:
            right = None

        return self._evaluate_comparison(op, left, right)

    def _evaluate_logical(
        self,
        op: str,
        operands: List[RuleExpression],
        payload: Dict[str, Any]
    ) -> bool:
        """
        Evaluate logical operations (and, or, not).

        Implements short-circuit evaluation for efficiency:
        - 'and' stops at first False
        - 'or' stops at first True
        - 'not' requires exactly one operand

        Args:
            op: Logical operator (and, or, not)
            operands: List of sub-expressions
            payload: Data payload

        Returns:
            Boolean result of logical operation

        Raises:
            ValueError: If 'not' has wrong number of operands
        """
        if op == "and":
            if not operands:
                return True  # Empty 'and' is vacuously true
            for operand in operands:
                if not self.evaluate(operand, payload):
                    return False  # Short-circuit
            return True

        elif op == "or":
            if not operands:
                return False  # Empty 'or' is vacuously false
            for operand in operands:
                if self.evaluate(operand, payload):
                    return True  # Short-circuit
            return False

        elif op == "not":
            if len(operands) != 1:
                logger.warning(f"'not' operator expects 1 operand, got {len(operands)}")
                return False
            return not self.evaluate(operands[0], payload)

        return False

    def _evaluate_null_check(
        self,
        expression: RuleExpression,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Evaluate null/existence check operations.

        Checks whether a field exists or is null/not-null.

        Args:
            expression: Null check expression
            payload: Data payload

        Returns:
            Boolean result of null check
        """
        op = expression.operator.lower()
        op_func = NULL_CHECK_OPERATORS.get(op)

        if op_func is None:
            return False

        # Get the field value
        if expression.field:
            value = self._get_value_at_path(payload, expression.field)
        else:
            value = None

        return op_func(value)

    def _evaluate_aggregate(
        self,
        expression: RuleExpression,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Evaluate aggregate operations (sum, count, min, max, avg).

        Aggregates values from multiple fields or array elements.

        Args:
            expression: Aggregate expression
            payload: Data payload

        Returns:
            Computed aggregate value

        Example:
            >>> expr = RuleExpression(
            ...     operator="sum",
            ...     fields=["/scope1", "/scope2", "/scope3"]
            ... )
            >>> evaluator._evaluate_aggregate(expr, {"scope1": 10, "scope2": 20, "scope3": 30})
            60
        """
        op = expression.operator.lower()

        # Collect values from fields
        values = []
        for field_path in expression.fields:
            val = self._get_value_at_path(payload, field_path)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)

        # Also check operands for nested expressions
        for operand in expression.operands:
            val = self.evaluate(operand, payload)
            if val is not None and isinstance(val, (int, float)):
                values.append(val)

        # Handle single field pointing to an array
        if expression.field:
            arr_val = self._get_value_at_path(payload, expression.field)
            if isinstance(arr_val, list):
                for item in arr_val:
                    if isinstance(item, (int, float)):
                        values.append(item)

        if not values:
            return 0 if op == "sum" else None

        if op == "sum":
            return sum(values)
        elif op == "count":
            return len(values)
        elif op == "min":
            return min(values)
        elif op == "max":
            return max(values)
        elif op == "avg":
            return sum(values) / len(values)

        return None

    def _evaluate_arithmetic(
        self,
        expression: RuleExpression,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Evaluate arithmetic operations (+, -, *, /).

        Performs arithmetic on two operands from fields or nested expressions.

        Args:
            expression: Arithmetic expression
            payload: Data payload

        Returns:
            Computed numeric value
        """
        op = expression.operator
        op_func = ARITHMETIC_OPERATORS.get(op)

        if op_func is None or len(expression.operands) < 2:
            return None

        left = self.evaluate(expression.operands[0], payload)
        right = self.evaluate(expression.operands[1], payload)

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None

        try:
            return op_func(left, right)
        except (ZeroDivisionError, OverflowError):
            return float('inf') if op == "/" else None


# =============================================================================
# RULE VALIDATOR
# =============================================================================


class RuleValidator:
    """
    Validates payloads against cross-field rules.

    The RuleValidator evaluates rules defined in the schema IR against
    data payloads, generating findings for any rule violations.

    Supported rule types:
        - Conditional requirements: "if field X has value Y, field Z is required"
        - Consistency checks: "sum of fields must equal total field"
        - Range dependencies: "if unit is X, value must be in range"
        - Custom expressions: Any boolean expression using supported operators

    Attributes:
        ir: Compiled schema Intermediate Representation
        options: Validation options (controls severity escalation, etc.)
        _evaluator: Expression evaluator instance
        _findings: List of accumulated findings

    Example:
        >>> validator = RuleValidator(schema_ir, options)
        >>> findings = validator.validate(payload)
        >>> for f in findings:
        ...     print(f.format_short())
    """

    def __init__(self, ir: SchemaIR, options: ValidationOptions) -> None:
        """
        Initialize the rule validator.

        Args:
            ir: Compiled schema IR containing rule bindings
            options: Validation options
        """
        self.ir = ir
        self.options = options
        self._evaluator = ExpressionEvaluator()
        self._findings: List[Finding] = []
        logger.debug(
            f"RuleValidator initialized for schema {ir.schema_id} "
            f"with {len(ir.rule_bindings)} rules"
        )

    def validate(
        self,
        payload: Dict[str, Any],
        rules: Optional[List[Rule]] = None
    ) -> List[Finding]:
        """
        Evaluate cross-field rules against payload.

        If rules are not provided, uses rules from the schema IR.
        For each rule:
        1. Check if 'when' condition is met (if present)
        2. If condition met, evaluate 'check' expression
        3. Generate finding if check fails

        Args:
            payload: The data payload to validate
            rules: Optional list of rules (uses IR rules if not provided)

        Returns:
            List of findings for failed rules

        Example:
            >>> findings = validator.validate({"fuel_type": "gas"})
            >>> print(len(findings))
            1  # Missing methane_slip
        """
        start_time = time.perf_counter()
        self._findings = []

        # Get rules to evaluate
        rules_to_eval: List[Rule] = []
        if rules is not None:
            rules_to_eval = rules
        else:
            # Convert IR rule bindings to Rule objects
            for binding in self.ir.rule_bindings:
                rule = self._parse_rule_binding(binding)
                if rule is not None:
                    rules_to_eval.append(rule)

        logger.debug(f"Evaluating {len(rules_to_eval)} rules against payload")

        # Evaluate each rule
        for rule in rules_to_eval:
            finding = self._evaluate_rule(rule, payload)
            if finding is not None:
                self._findings.append(finding)

                # Check fail-fast option
                if self.options.fail_fast:
                    break

                # Check max errors
                if self.options.max_errors > 0 and len(self._findings) >= self.options.max_errors:
                    logger.info(f"Reached max errors limit ({self.options.max_errors})")
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Rule validation completed: {len(self._findings)} findings in {elapsed_ms:.2f}ms"
        )

        return self._findings

    def _evaluate_rule(
        self,
        rule: Rule,
        payload: Dict[str, Any]
    ) -> Optional[Finding]:
        """
        Evaluate a single rule.

        Checks the 'when' condition (if present), then evaluates
        the 'check' expression. Returns a Finding if the rule fails.

        Args:
            rule: Rule to evaluate
            payload: Data payload

        Returns:
            Finding if rule fails, None if passes

        Example:
            >>> rule = Rule(
            ...     rule_id="test",
            ...     check=RuleExpression(operator="exists", field="/name"),
            ...     message="Name is required"
            ... )
            >>> finding = validator._evaluate_rule(rule, {})
            >>> print(finding.message)
            'Name is required'
        """
        try:
            # Check 'when' condition
            if rule.when is not None:
                condition_met = self._evaluator.evaluate(rule.when, payload)
                if not condition_met:
                    # Condition not met, rule doesn't apply
                    logger.debug(f"Rule {rule.rule_id} condition not met, skipping")
                    return None

            # Evaluate 'check' expression
            check_passed = self._evaluator.evaluate(rule.check, payload)

            if not check_passed:
                # Rule failed, create finding
                return self._create_finding(rule, payload)

            return None

        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}", exc_info=True)
            # Create error finding for rule evaluation failure
            return Finding(
                code=ERROR_CODE_RULE_VIOLATION,
                severity=Severity.ERROR,
                path="",
                message=f"Rule '{rule.rule_id}' evaluation failed: {str(e)}",
                expected={"rule_id": rule.rule_id},
                actual={"error": str(e)},
            )

    def _create_finding(
        self,
        rule: Rule,
        payload: Dict[str, Any]
    ) -> Finding:
        """
        Create a finding for a failed rule.

        Formats the message using the template if available,
        determines the appropriate error code, and constructs
        a complete Finding object.

        Args:
            rule: The failed rule
            payload: Data payload (for message formatting)

        Returns:
            Finding object describing the violation
        """
        # Get error code
        error_code = rule.get_error_code()

        # Get severity
        severity_map = {
            "error": Severity.ERROR,
            "warning": Severity.WARNING,
            "info": Severity.INFO,
        }
        severity = severity_map.get(rule.severity, Severity.ERROR)

        # In strict mode, escalate warnings to errors
        if self.options.profile.is_strict() and severity == Severity.WARNING:
            severity = Severity.ERROR

        # Format message
        message = rule.message
        if rule.message_template:
            message = self._format_message(rule.message_template, payload)

        # Determine affected path
        path = self._get_primary_path(rule)

        # Build expected/actual for debugging
        expected = {"rule_id": rule.rule_id, "check": "pass"}
        actual = {"check": "fail"}

        # Add field values to actual if available
        if rule.check.field:
            actual["field_value"] = self._evaluator._get_value_at_path(
                payload, rule.check.field
            )

        return Finding(
            code=error_code,
            severity=severity,
            path=path,
            message=message,
            expected=expected,
            actual=actual,
            hint=FindingHint(
                category="rule_violation",
                docs_url=f"https://docs.greenlang.dev/errors/{error_code}",
            ),
        )

    def _format_message(
        self,
        template: str,
        payload: Dict[str, Any]
    ) -> str:
        """
        Format message template with payload values.

        Supports {{ path }} placeholders where path is a JSON Pointer.
        Also supports {{ /field }} syntax for explicit paths.

        Args:
            template: Message template with placeholders
            payload: Data payload for value substitution

        Returns:
            Formatted message string

        Example:
            >>> template = "Value at {{ /fuel_type }} is invalid"
            >>> payload = {"fuel_type": "gas"}
            >>> formatted = validator._format_message(template, payload)
            >>> print(formatted)
            'Value at gas is invalid'
        """
        if not template:
            return ""

        def replace_placeholder(match: re.Match) -> str:
            """Replace a single placeholder."""
            path = match.group(1).strip()

            # Ensure path starts with /
            if not path.startswith("/"):
                path = "/" + path

            value = self._evaluator._get_value_at_path(payload, path)
            return str(value) if value is not None else "(null)"

        # Pattern: {{ path }} or {{path}}
        pattern = r"\{\{\s*([^}]+)\s*\}\}"
        return re.sub(pattern, replace_placeholder, template)

    def _get_primary_path(self, rule: Rule) -> str:
        """
        Get the primary path affected by a rule.

        For error reporting, we need a single path. This method
        determines the most relevant path from the rule's expressions.

        Args:
            rule: The rule to get path from

        Returns:
            JSON Pointer path string
        """
        # First, try the check expression's field
        if rule.check.field:
            return rule.check.field

        # Try the first operand's field
        if rule.check.operands:
            for op in rule.check.operands:
                if op.field:
                    return op.field

        # Try fields list
        if rule.check.fields:
            return rule.check.fields[0]

        # Fall back to root
        return ""

    def _parse_rule_binding(
        self,
        binding: RuleBindingIR
    ) -> Optional[Rule]:
        """
        Convert IR binding to Rule object.

        Parses the RuleBindingIR from the compiled schema into
        a Rule object that can be evaluated.

        Args:
            binding: Rule binding from schema IR

        Returns:
            Rule object, or None if parsing fails
        """
        try:
            # Parse 'when' condition if present
            when_expr = None
            if binding.when:
                when_expr = self._parse_expression_dict(binding.when)

            # Parse 'check' expression
            check_expr = self._parse_expression_dict(binding.check)
            if check_expr is None:
                logger.warning(f"Failed to parse check expression for rule {binding.rule_id}")
                return None

            return Rule(
                rule_id=binding.rule_id,
                severity=binding.severity,
                when=when_expr,
                check=check_expr,
                message=binding.message,
                message_template=binding.message_template,
            )

        except Exception as e:
            logger.error(f"Failed to parse rule binding {binding.rule_id}: {e}")
            return None

    def _parse_expression_dict(
        self,
        expr_dict: Dict[str, Any]
    ) -> Optional[RuleExpression]:
        """
        Parse an expression dictionary into RuleExpression.

        Handles various expression formats from the DSL:
        - Simple: {"field": "/path", "operator": "==", "value": 123}
        - Nested: {"operator": "and", "operands": [...]}
        - Legacy: {"$ref": "$.path"} or {"==": [...]}

        Args:
            expr_dict: Expression dictionary from rule binding

        Returns:
            RuleExpression object, or None if parsing fails
        """
        if not expr_dict:
            return None

        try:
            # Direct format: {"operator": "...", "field": "...", ...}
            if "operator" in expr_dict:
                return RuleExpression(
                    operator=expr_dict["operator"],
                    field=expr_dict.get("field"),
                    value=expr_dict.get("value"),
                    operands=[
                        self._parse_expression_dict(op)
                        for op in expr_dict.get("operands", [])
                        if op
                    ],
                    fields=expr_dict.get("fields", []),
                )

            # Legacy operator-as-key format: {"==": [...], "$ref": "..."}
            for op in list(COMPARISON_OPERATORS.keys()) + ["and", "or", "not", "sum", "exists"]:
                if op in expr_dict:
                    operand_val = expr_dict[op]

                    # Handle different operand formats
                    if isinstance(operand_val, list):
                        operands = []
                        for item in operand_val:
                            if isinstance(item, dict):
                                parsed = self._parse_expression_dict(item)
                                if parsed:
                                    operands.append(parsed)
                            else:
                                # Literal value - create simple value expression
                                operands.append(RuleExpression(
                                    operator="literal",
                                    value=item,
                                ))
                        return RuleExpression(
                            operator=op,
                            operands=operands,
                        )
                    elif isinstance(operand_val, dict):
                        # Nested expression
                        operand = self._parse_expression_dict(operand_val)
                        if operand:
                            return RuleExpression(
                                operator=op,
                                operands=[operand],
                            )
                    else:
                        # Simple value
                        return RuleExpression(
                            operator=op,
                            value=operand_val,
                        )

            # Handle $ref format (legacy path reference)
            if "$ref" in expr_dict:
                ref_path = expr_dict["$ref"]
                # Convert $.path to /path format
                if ref_path.startswith("$."):
                    ref_path = "/" + ref_path[2:].replace(".", "/")
                return RuleExpression(
                    operator="value",
                    field=ref_path,
                )

            logger.warning(f"Unknown expression format: {expr_dict}")
            return None

        except Exception as e:
            logger.error(f"Failed to parse expression: {e}")
            return None


# =============================================================================
# BUILT-IN RULE TEMPLATES
# =============================================================================

BUILTIN_RULES: Dict[str, str] = {
    "conditional_required": """
when:
  operator: "=="
  field: "/fuel_type"
  value: "gas"
check:
  operator: "exists"
  field: "/methane_slip"
message: "methane_slip is required when fuel_type is 'gas'"
""",

    "sum_equals_total": """
check:
  operator: "=="
  operands:
    - operator: "sum"
      fields: ["/scope1", "/scope2", "/scope3"]
    - operator: "value"
      field: "/total_emissions"
message: "Sum of scopes must equal total_emissions"
""",

    "range_check": """
when:
  operator: "=="
  field: "/unit"
  value: "celsius"
check:
  operator: "and"
  operands:
    - operator: ">="
      field: "/value"
      value: -273.15
    - operator: "<="
      field: "/value"
      value: 1000
message: "Temperature must be between -273.15 and 1000 celsius"
""",

    "enum_dependency": """
when:
  operator: "in"
  field: "/transport_mode"
  value: ["car", "truck", "bus"]
check:
  operator: "exists"
  field: "/fuel_type"
message: "fuel_type is required for motorized transport"
""",

    "non_negative": """
check:
  operator: ">="
  field: "/emissions"
  value: 0
message: "emissions must be non-negative"
""",

    "percentage_range": """
check:
  operator: "and"
  operands:
    - operator: ">="
      field: "/percentage"
      value: 0
    - operator: "<="
      field: "/percentage"
      value: 100
message: "percentage must be between 0 and 100"
""",
}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_conditional_required_rule(
    rule_id: str,
    condition_field: str,
    condition_value: Any,
    required_field: str,
    message: Optional[str] = None,
) -> Rule:
    """
    Create a conditional required field rule.

    Args:
        rule_id: Unique identifier for the rule
        condition_field: Field path to check (JSON Pointer)
        condition_value: Value that triggers the requirement
        required_field: Field path that becomes required
        message: Optional custom message

    Returns:
        Rule object for conditional requirement

    Example:
        >>> rule = create_conditional_required_rule(
        ...     rule_id="gas_methane_slip",
        ...     condition_field="/fuel_type",
        ...     condition_value="gas",
        ...     required_field="/methane_slip"
        ... )
    """
    if message is None:
        message = f"{required_field} is required when {condition_field} is '{condition_value}'"

    return Rule(
        rule_id=rule_id,
        severity="error",
        when=RuleExpression(
            operator="==",
            field=condition_field,
            value=condition_value,
        ),
        check=RuleExpression(
            operator="exists",
            field=required_field,
        ),
        message=message,
    )


def create_sum_consistency_rule(
    rule_id: str,
    component_fields: List[str],
    total_field: str,
    message: Optional[str] = None,
    tolerance: float = 0.0,
) -> Rule:
    """
    Create a sum consistency rule.

    Args:
        rule_id: Unique identifier for the rule
        component_fields: List of field paths to sum
        total_field: Field path that should equal the sum
        message: Optional custom message
        tolerance: Numeric tolerance for comparison (default 0)

    Returns:
        Rule object for sum consistency check

    Example:
        >>> rule = create_sum_consistency_rule(
        ...     rule_id="scope_sum",
        ...     component_fields=["/scope1", "/scope2", "/scope3"],
        ...     total_field="/total_emissions"
        ... )
    """
    if message is None:
        fields_str = ", ".join(component_fields)
        message = f"Sum of [{fields_str}] must equal {total_field}"

    return Rule(
        rule_id=rule_id,
        severity="error",
        check=RuleExpression(
            operator="==",
            operands=[
                RuleExpression(
                    operator="sum",
                    fields=component_fields,
                ),
                RuleExpression(
                    operator="value",
                    field=total_field,
                ),
            ],
        ),
        message=message,
    )


def create_range_rule(
    rule_id: str,
    field: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
    message: Optional[str] = None,
) -> Rule:
    """
    Create a range validation rule.

    Args:
        rule_id: Unique identifier for the rule
        field: Field path to check
        minimum: Minimum allowed value (None for no limit)
        maximum: Maximum allowed value (None for no limit)
        exclusive_min: Whether minimum is exclusive
        exclusive_max: Whether maximum is exclusive
        message: Optional custom message

    Returns:
        Rule object for range validation

    Example:
        >>> rule = create_range_rule(
        ...     rule_id="temperature_range",
        ...     field="/temperature",
        ...     minimum=-273.15,
        ...     maximum=1000
        ... )
    """
    conditions = []

    if minimum is not None:
        op = ">" if exclusive_min else ">="
        conditions.append(RuleExpression(operator=op, field=field, value=minimum))

    if maximum is not None:
        op = "<" if exclusive_max else "<="
        conditions.append(RuleExpression(operator=op, field=field, value=maximum))

    if not conditions:
        raise ValueError("At least one of minimum or maximum must be specified")

    if len(conditions) == 1:
        check = conditions[0]
    else:
        check = RuleExpression(operator="and", operands=conditions)

    if message is None:
        bounds = []
        if minimum is not None:
            op_str = ">" if exclusive_min else ">="
            bounds.append(f"{op_str} {minimum}")
        if maximum is not None:
            op_str = "<" if exclusive_max else "<="
            bounds.append(f"{op_str} {maximum}")
        message = f"{field} must be {' and '.join(bounds)}"

    return Rule(
        rule_id=rule_id,
        severity="error",
        check=check,
        message=message,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Error codes
    "ERROR_CODE_RULE_VIOLATION",
    "ERROR_CODE_CONDITIONAL_REQUIRED",
    "ERROR_CODE_CONSISTENCY_ERROR",
    # Models
    "RuleExpression",
    "Rule",
    # Classes
    "ExpressionEvaluator",
    "RuleValidator",
    # Constants
    "COMPARISON_OPERATORS",
    "EXTENDED_OPERATORS",
    "NULL_CHECK_OPERATORS",
    "ARITHMETIC_OPERATORS",
    "BUILTIN_RULES",
    # Factory functions
    "create_conditional_required_rule",
    "create_sum_consistency_rule",
    "create_range_rule",
]
