"""
Spec Validator Module for GreenLang Agent Generator

This module provides comprehensive validation for AgentSpec objects,
including JSON Schema validation, semantic validation, and business
rule validation.

The validator ensures:
- All required fields are present
- Cross-references are valid (tools referenced by agents exist)
- Business rules are satisfied (naming conventions, constraints)
- Regulatory compliance requirements are met

Example:
    >>> validator = SpecValidator()
    >>> result = validator.validate(spec)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"Error: {error.message}")
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .schema import (
    AgentSpec,
    AgentDef,
    AgentType,
    ToolDef,
    ToolType,
    InputDef,
    OutputDef,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Types
# =============================================================================

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """
    Represents a validation error.

    Attributes:
        code: Unique error code (e.g., 'E001')
        message: Human-readable error message
        path: JSON path to the problematic field
        severity: Error severity level
        suggestion: Optional suggestion for fixing the error
        context: Additional context information
    """

    code: str
    message: str
    path: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format error as string."""
        parts = [f"[{self.code}]"]
        if self.path:
            parts.append(f"at '{self.path}':")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"(Suggestion: {self.suggestion})")
        return " ".join(parts)


@dataclass
class ValidationWarning:
    """
    Represents a validation warning.

    Warnings are non-fatal issues that should be addressed
    but don't prevent code generation.
    """

    code: str
    message: str
    path: str = ""
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format warning as string."""
        parts = [f"[{self.code}]"]
        if self.path:
            parts.append(f"at '{self.path}':")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"(Suggestion: {self.suggestion})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """
    Complete validation result.

    Contains all errors, warnings, and metadata about the validation.

    Attributes:
        is_valid: True if no errors (warnings don't affect validity)
        errors: List of validation errors
        warnings: List of validation warnings
        validated_at: Timestamp of validation
        spec_hash: Hash of the validated spec
        validation_time_ms: Time taken for validation
    """

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    spec_hash: Optional[str] = None
    validation_time_ms: float = 0.0

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get total warning count."""
        return len(self.warnings)

    def add_error(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            code=code,
            message=message,
            path=path,
            suggestion=suggestion,
            context=context,
        ))
        self.is_valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: Optional[str] = None,
        **context: Any,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationWarning(
            code=code,
            message=message,
            path=path,
            suggestion=suggestion,
            context=context,
        ))

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.errors:
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [
                {
                    "code": e.code,
                    "message": e.message,
                    "path": e.path,
                    "suggestion": e.suggestion,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "code": w.code,
                    "message": w.message,
                    "path": w.path,
                    "suggestion": w.suggestion,
                }
                for w in self.warnings
            ],
            "validated_at": self.validated_at.isoformat(),
            "validation_time_ms": self.validation_time_ms,
        }


# =============================================================================
# Validation Rules Registry
# =============================================================================

class ValidationRuleRegistry:
    """
    Registry for validation rules.

    Allows registration of custom validation rules that are
    executed during spec validation.
    """

    def __init__(self):
        self._rules: Dict[str, Callable[[AgentSpec], ValidationResult]] = {}

    def register(self, rule_id: str):
        """
        Decorator to register a validation rule.

        Example:
            @registry.register("custom_rule")
            def validate_custom(spec: AgentSpec) -> ValidationResult:
                result = ValidationResult(is_valid=True)
                # validation logic
                return result
        """
        def decorator(func: Callable[[AgentSpec], ValidationResult]):
            self._rules[rule_id] = func
            return func
        return decorator

    def get_rule(self, rule_id: str) -> Optional[Callable]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> Dict[str, Callable]:
        """Get all registered rules."""
        return self._rules.copy()


# Global registry
validation_registry = ValidationRuleRegistry()


# =============================================================================
# Spec Validator
# =============================================================================

class SpecValidator:
    """
    Comprehensive validator for AgentSpec objects.

    This validator performs:
    1. Schema validation (required fields, types)
    2. Semantic validation (cross-references, dependencies)
    3. Business rule validation (naming, constraints)
    4. Regulatory compliance validation

    Example:
        >>> validator = SpecValidator()
        >>> result = validator.validate(spec)
        >>> if result.is_valid:
        ...     print("Spec is valid!")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    # Error codes
    ERR_MISSING_PACK = "E001"
    ERR_MISSING_PACK_ID = "E002"
    ERR_MISSING_PACK_NAME = "E003"
    ERR_INVALID_PACK_ID = "E004"
    ERR_INVALID_VERSION = "E005"
    ERR_NO_AGENTS = "E006"
    ERR_MISSING_AGENT_ID = "E007"
    ERR_DUPLICATE_AGENT_ID = "E008"
    ERR_MISSING_AGENT_TYPE = "E009"
    ERR_INVALID_AGENT_ID = "E010"
    ERR_MISSING_TOOL_REF = "E011"
    ERR_DUPLICATE_TOOL_ID = "E012"
    ERR_INVALID_TOOL_ID = "E013"
    ERR_MISSING_INPUT_NAME = "E014"
    ERR_MISSING_INPUT_TYPE = "E015"
    ERR_INVALID_INPUT_NAME = "E016"
    ERR_MISSING_OUTPUT_NAME = "E017"
    ERR_MISSING_OUTPUT_TYPE = "E018"
    ERR_INVALID_OUTPUT_NAME = "E019"
    ERR_CIRCULAR_DEPENDENCY = "E020"
    ERR_INVALID_PYTHON_TYPE = "E021"

    # Warning codes
    WARN_NO_DESCRIPTION = "W001"
    WARN_NO_TOOLS = "W002"
    WARN_NO_INPUTS = "W003"
    WARN_NO_OUTPUTS = "W004"
    WARN_NO_GOLDEN_TESTS = "W005"
    WARN_NO_DEPLOYMENT = "W006"
    WARN_SHORT_DESCRIPTION = "W007"
    WARN_NO_VALIDATION_RULES = "W008"
    WARN_MISSING_DEADLINE = "W009"
    WARN_PAST_DEADLINE = "W010"

    # Patterns
    KEBAB_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
    SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
    SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$")
    PYTHON_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    # Valid Python types and custom types
    VALID_PRIMITIVE_TYPES = {
        "str", "string", "int", "float", "bool", "boolean",
        "list", "dict", "date", "datetime", "Any", "None",
        "Optional", "List", "Dict", "Union", "Tuple",
    }

    def __init__(self, custom_rules: Optional[ValidationRuleRegistry] = None):
        """
        Initialize the spec validator.

        Args:
            custom_rules: Optional registry of custom validation rules
        """
        self.custom_rules = custom_rules or validation_registry
        logger.info("SpecValidator initialized")

    def validate(self, spec: AgentSpec) -> ValidationResult:
        """
        Perform complete validation of an AgentSpec.

        Args:
            spec: The AgentSpec to validate

        Returns:
            ValidationResult with all errors and warnings
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True, spec_hash=spec.spec_hash)

        logger.info(f"Starting validation of spec: {spec.pack.id if spec.pack else 'unknown'}")

        try:
            # 1. Schema validation (required fields)
            self._validate_schema(spec, result)

            # 2. Semantic validation (cross-references)
            self._validate_semantics(spec, result)

            # 3. Business rule validation
            self._validate_business_rules(spec, result)

            # 4. Run custom validation rules
            self._run_custom_rules(spec, result)

        except Exception as e:
            logger.error(f"Validation failed with exception: {e}", exc_info=True)
            result.add_error(
                "E999",
                f"Unexpected validation error: {str(e)}",
                suggestion="Check the spec format and try again",
            )

        # Calculate validation time
        result.validation_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Validation complete: valid={result.is_valid}, "
            f"errors={result.error_count}, warnings={result.warning_count}, "
            f"time={result.validation_time_ms:.2f}ms"
        )

        return result

    def validate_agent(
        self,
        agent: AgentDef,
        available_tools: Set[str],
    ) -> ValidationResult:
        """
        Validate a single agent definition.

        Args:
            agent: The agent to validate
            available_tools: Set of available tool IDs

        Returns:
            ValidationResult for this agent
        """
        result = ValidationResult(is_valid=True)
        path_prefix = f"agents[{agent.id}]"

        # Validate agent ID
        if not agent.id:
            result.add_error(
                self.ERR_MISSING_AGENT_ID,
                "Agent ID is required",
                path=path_prefix,
            )
        elif not self.KEBAB_CASE_PATTERN.match(agent.id):
            result.add_error(
                self.ERR_INVALID_AGENT_ID,
                f"Agent ID must be kebab-case: {agent.id}",
                path=f"{path_prefix}.id",
                suggestion="Use lowercase letters, numbers, and hyphens (e.g., 'my-agent-v1')",
            )

        # Validate agent type
        if not agent.type:
            result.add_error(
                self.ERR_MISSING_AGENT_TYPE,
                "Agent type is required",
                path=f"{path_prefix}.type",
            )

        # Validate description
        if not agent.description:
            result.add_warning(
                self.WARN_NO_DESCRIPTION,
                f"Agent '{agent.id}' has no description",
                path=f"{path_prefix}.description",
                suggestion="Add a description for documentation",
            )
        elif len(agent.description) < 20:
            result.add_warning(
                self.WARN_SHORT_DESCRIPTION,
                f"Agent '{agent.id}' description is very short",
                path=f"{path_prefix}.description",
                suggestion="Add more detail to the description",
            )

        # Validate inputs
        if not agent.inputs:
            result.add_warning(
                self.WARN_NO_INPUTS,
                f"Agent '{agent.id}' has no inputs defined",
                path=f"{path_prefix}.inputs",
            )
        else:
            input_names: Set[str] = set()
            for i, inp in enumerate(agent.inputs):
                inp_result = self._validate_input(inp, f"{path_prefix}.inputs[{i}]")
                result.merge(inp_result)

                # Check for duplicates
                if inp.name in input_names:
                    result.add_error(
                        "E022",
                        f"Duplicate input name: {inp.name}",
                        path=f"{path_prefix}.inputs[{i}].name",
                    )
                input_names.add(inp.name)

        # Validate outputs
        if not agent.outputs:
            result.add_warning(
                self.WARN_NO_OUTPUTS,
                f"Agent '{agent.id}' has no outputs defined",
                path=f"{path_prefix}.outputs",
            )
        else:
            output_names: Set[str] = set()
            for i, out in enumerate(agent.outputs):
                out_result = self._validate_output(out, f"{path_prefix}.outputs[{i}]")
                result.merge(out_result)

                # Check for duplicates
                if out.name in output_names:
                    result.add_error(
                        "E023",
                        f"Duplicate output name: {out.name}",
                        path=f"{path_prefix}.outputs[{i}].name",
                    )
                output_names.add(out.name)

        # Validate tool references
        for tool_id in agent.tools:
            if tool_id not in available_tools:
                result.add_error(
                    self.ERR_MISSING_TOOL_REF,
                    f"Agent '{agent.id}' references undefined tool: {tool_id}",
                    path=f"{path_prefix}.tools",
                    suggestion=f"Define the tool '{tool_id}' in the tools section",
                )

        return result

    def validate_tool(self, tool: ToolDef) -> ValidationResult:
        """
        Validate a single tool definition.

        Args:
            tool: The tool to validate

        Returns:
            ValidationResult for this tool
        """
        result = ValidationResult(is_valid=True)
        path_prefix = f"tools[{tool.id}]"

        # Validate tool ID
        if not tool.id:
            result.add_error(
                "E024",
                "Tool ID is required",
                path=path_prefix,
            )
        elif not self.SNAKE_CASE_PATTERN.match(tool.id):
            result.add_error(
                self.ERR_INVALID_TOOL_ID,
                f"Tool ID must be snake_case: {tool.id}",
                path=f"{path_prefix}.id",
                suggestion="Use lowercase letters, numbers, and underscores",
            )

        # Validate description
        if not tool.description:
            result.add_warning(
                self.WARN_NO_DESCRIPTION,
                f"Tool '{tool.id}' has no description",
                path=f"{path_prefix}.description",
            )

        # Validate external API tools have config
        if tool.type == ToolType.EXTERNAL_API:
            if not tool.config:
                result.add_warning(
                    "W011",
                    f"External API tool '{tool.id}' has no config",
                    path=f"{path_prefix}.config",
                    suggestion="Add base_url, auth_type, and other config",
                )
            elif tool.config and not tool.config.base_url:
                result.add_warning(
                    "W012",
                    f"External API tool '{tool.id}' has no base_url",
                    path=f"{path_prefix}.config.base_url",
                )

        return result

    def _validate_schema(self, spec: AgentSpec, result: ValidationResult) -> None:
        """Validate required fields are present."""
        # Validate pack
        if not spec.pack:
            result.add_error(
                self.ERR_MISSING_PACK,
                "Pack specification is required",
                path="pack",
            )
            return

        if not spec.pack.id:
            result.add_error(
                self.ERR_MISSING_PACK_ID,
                "Pack ID is required",
                path="pack.id",
            )
        elif not self.KEBAB_CASE_PATTERN.match(spec.pack.id):
            result.add_error(
                self.ERR_INVALID_PACK_ID,
                f"Pack ID must be kebab-case: {spec.pack.id}",
                path="pack.id",
                suggestion="Use lowercase letters, numbers, and hyphens",
            )

        if not spec.pack.name:
            result.add_error(
                self.ERR_MISSING_PACK_NAME,
                "Pack name is required",
                path="pack.name",
            )

        if not self.SEMVER_PATTERN.match(spec.pack.version):
            result.add_error(
                self.ERR_INVALID_VERSION,
                f"Pack version must be semantic version: {spec.pack.version}",
                path="pack.version",
                suggestion="Use format: x.y.z (e.g., '1.0.0')",
            )

        # Validate agents exist
        if not spec.agents:
            result.add_error(
                self.ERR_NO_AGENTS,
                "At least one agent must be defined",
                path="agents",
            )

    def _validate_semantics(self, spec: AgentSpec, result: ValidationResult) -> None:
        """Validate cross-references and dependencies."""
        # Build tool ID set
        tool_ids: Set[str] = {tool.id for tool in spec.tools}

        # Check for duplicate tool IDs
        seen_tool_ids: Set[str] = set()
        for tool in spec.tools:
            if tool.id in seen_tool_ids:
                result.add_error(
                    self.ERR_DUPLICATE_TOOL_ID,
                    f"Duplicate tool ID: {tool.id}",
                    path=f"tools[{tool.id}]",
                )
            seen_tool_ids.add(tool.id)

        # Check for duplicate agent IDs
        seen_agent_ids: Set[str] = set()
        for agent in spec.agents:
            if agent.id in seen_agent_ids:
                result.add_error(
                    self.ERR_DUPLICATE_AGENT_ID,
                    f"Duplicate agent ID: {agent.id}",
                    path=f"agents[{agent.id}]",
                )
            seen_agent_ids.add(agent.id)

        # Validate each agent
        for agent in spec.agents:
            agent_result = self.validate_agent(agent, tool_ids)
            result.merge(agent_result)

        # Validate each tool
        for tool in spec.tools:
            tool_result = self.validate_tool(tool)
            result.merge(tool_result)

        # Check for unused tools
        used_tools: Set[str] = set()
        for agent in spec.agents:
            used_tools.update(agent.tools)

        unused_tools = tool_ids - used_tools
        for tool_id in unused_tools:
            result.add_warning(
                "W013",
                f"Tool '{tool_id}' is defined but not used by any agent",
                path=f"tools[{tool_id}]",
                suggestion="Remove unused tools or assign them to an agent",
            )

    def _validate_business_rules(
        self,
        spec: AgentSpec,
        result: ValidationResult,
    ) -> None:
        """Validate business rules and best practices."""
        # Check for golden tests
        if not spec.golden_tests or spec.golden_tests.count == 0:
            result.add_warning(
                self.WARN_NO_GOLDEN_TESTS,
                "No golden tests defined",
                path="golden_tests",
                suggestion="Define golden tests for validation coverage",
            )

        # Check for deployment config
        if not spec.deployment:
            result.add_warning(
                self.WARN_NO_DEPLOYMENT,
                "No deployment configuration",
                path="deployment",
                suggestion="Add deployment config for production readiness",
            )

        # Check for validation rules
        if not spec.validation or not spec.validation.get("rules"):
            result.add_warning(
                self.WARN_NO_VALIDATION_RULES,
                "No validation rules defined",
                path="validation.rules",
                suggestion="Add validation rules for input checking",
            )

        # Check deadline for regulatory agents
        if spec.pack.category == "regulatory-compliance":
            if not spec.pack.deadline:
                result.add_warning(
                    self.WARN_MISSING_DEADLINE,
                    "Regulatory compliance agent should have a deadline",
                    path="pack.deadline",
                )
            elif spec.pack.deadline:
                try:
                    deadline = datetime.strptime(spec.pack.deadline, "%Y-%m-%d")
                    if deadline < datetime.utcnow():
                        result.add_warning(
                            self.WARN_PAST_DEADLINE,
                            f"Pack deadline has passed: {spec.pack.deadline}",
                            path="pack.deadline",
                        )
                except ValueError:
                    pass  # Invalid date format already caught

    def _validate_input(self, inp: InputDef, path: str) -> ValidationResult:
        """Validate an input definition."""
        result = ValidationResult(is_valid=True)

        if not inp.name:
            result.add_error(
                self.ERR_MISSING_INPUT_NAME,
                "Input name is required",
                path=f"{path}.name",
            )
        elif not self.PYTHON_IDENTIFIER_PATTERN.match(inp.name):
            result.add_error(
                self.ERR_INVALID_INPUT_NAME,
                f"Input name must be valid Python identifier: {inp.name}",
                path=f"{path}.name",
            )

        if not inp.type:
            result.add_error(
                self.ERR_MISSING_INPUT_TYPE,
                "Input type is required",
                path=f"{path}.type",
            )
        else:
            # Check if type looks like a Python type
            base_type = inp.type.split("[")[0]  # Handle List[str] etc.
            if (
                base_type not in self.VALID_PRIMITIVE_TYPES
                and not base_type[0].isupper()  # Custom types should be PascalCase
            ):
                result.add_warning(
                    "W014",
                    f"Input type may not be valid: {inp.type}",
                    path=f"{path}.type",
                    suggestion="Use standard Python types or PascalCase custom types",
                )

        return result

    def _validate_output(self, out: OutputDef, path: str) -> ValidationResult:
        """Validate an output definition."""
        result = ValidationResult(is_valid=True)

        if not out.name:
            result.add_error(
                self.ERR_MISSING_OUTPUT_NAME,
                "Output name is required",
                path=f"{path}.name",
            )
        elif not self.PYTHON_IDENTIFIER_PATTERN.match(out.name):
            result.add_error(
                self.ERR_INVALID_OUTPUT_NAME,
                f"Output name must be valid Python identifier: {out.name}",
                path=f"{path}.name",
            )

        if not out.type:
            result.add_error(
                self.ERR_MISSING_OUTPUT_TYPE,
                "Output type is required",
                path=f"{path}.type",
            )

        return result

    def _run_custom_rules(self, spec: AgentSpec, result: ValidationResult) -> None:
        """Run custom validation rules."""
        for rule_id, rule_func in self.custom_rules.get_all_rules().items():
            try:
                rule_result = rule_func(spec)
                result.merge(rule_result)
            except Exception as e:
                logger.error(f"Custom rule '{rule_id}' failed: {e}")
                result.add_warning(
                    "W999",
                    f"Custom rule '{rule_id}' failed: {str(e)}",
                )


# =============================================================================
# Built-in Custom Rules
# =============================================================================

@validation_registry.register("zero_hallucination_check")
def validate_zero_hallucination(spec: AgentSpec) -> ValidationResult:
    """
    Validate that deterministic calculators don't use ML tools.

    Zero-hallucination principle: Deterministic agents should not
    use ML models for numeric calculations.
    """
    result = ValidationResult(is_valid=True)

    for agent in spec.agents:
        if agent.type == AgentType.DETERMINISTIC_CALCULATOR:
            ml_tools = []
            for tool_id in agent.tools:
                tool = spec.get_tool(tool_id)
                if tool and tool.type == ToolType.ML_MODEL:
                    ml_tools.append(tool_id)

            if ml_tools:
                result.add_warning(
                    "W100",
                    f"Deterministic agent '{agent.id}' uses ML tools: {ml_tools}",
                    path=f"agents[{agent.id}].tools",
                    suggestion=(
                        "Consider using deterministic tools for numeric calculations "
                        "to maintain zero-hallucination guarantees"
                    ),
                )

    return result


@validation_registry.register("provenance_tracking_check")
def validate_provenance_tracking(spec: AgentSpec) -> ValidationResult:
    """
    Validate that agents have provenance tracking outputs.

    All GreenLang agents should output a provenance_hash for audit.
    """
    result = ValidationResult(is_valid=True)

    for agent in spec.agents:
        has_provenance = any(
            out.name in ("provenance_hash", "provenance", "audit_hash")
            for out in agent.outputs
        )

        if not has_provenance:
            result.add_warning(
                "W101",
                f"Agent '{agent.id}' should have a provenance tracking output",
                path=f"agents[{agent.id}].outputs",
                suggestion="Add 'provenance_hash' output for audit compliance",
            )

    return result


# =============================================================================
# Unit Test Stubs
# =============================================================================

def _test_spec_validator():
    """
    Unit test stub for SpecValidator.

    Run with: pytest backend/agent_generator/parser/spec_validator.py
    """
    from backend.agent_generator.parser.yaml_parser import (
        AgentSpec,
        PackSpec,
        AgentDefinition,
        InputDefinition,
        OutputDefinition,
    )

    # Test 1: Valid spec
    valid_spec = AgentSpec(
        pack=PackSpec(
            id="test-agent-v1",
            name="Test Agent",
            version="1.0.0",
            description="A test agent",
        ),
        agents=[
            AgentDefinition(
                id="test-validator",
                name="Test Validator",
                type="deterministic-calculator",
                description="A test validator that does validation",
                inputs=[
                    InputDefinition(
                        name="input_value",
                        type="float",
                        required=True,
                        description="Input value",
                    ),
                ],
                outputs=[
                    OutputDefinition(
                        name="result",
                        type="float",
                        description="Result",
                    ),
                ],
                tools=[],
            ),
        ],
        tools=[],
    )

    validator = SpecValidator()
    result = validator.validate(valid_spec)

    assert result.is_valid, f"Expected valid but got errors: {result.errors}"
    print(f"Valid spec test passed (warnings: {result.warning_count})")

    # Test 2: Invalid spec (missing pack ID)
    invalid_spec = AgentSpec(
        pack=PackSpec(
            id="",  # Invalid: empty
            name="Test Agent",
            version="1.0.0",
        ),
        agents=[],
        tools=[],
    )

    result = validator.validate(invalid_spec)
    assert not result.is_valid, "Expected invalid"
    assert any(e.code == "E002" for e in result.errors), "Expected E002 error"
    print("Invalid spec test passed")

    print("SpecValidator tests passed!")


if __name__ == "__main__":
    _test_spec_validator()
