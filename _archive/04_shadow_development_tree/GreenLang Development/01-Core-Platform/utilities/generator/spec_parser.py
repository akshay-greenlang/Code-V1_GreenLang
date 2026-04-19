"""
AgentSpec Parser - Parse and validate AgentSpec v2 YAML files.

This module parses pack.yaml files in AgentSpec v2 format and extracts
all components needed for code generation: inputs, outputs, tools,
prompts, and provenance configuration.

Example:
    >>> parser = AgentSpecParser()
    >>> spec = parser.parse("path/to/pack.yaml")
    >>> print(spec.name)
    >>> print(spec.inputs)
    >>> print(spec.tools)
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

import yaml
from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class PackKind(str, Enum):
    """Supported pack kinds."""
    PACK = "pack"
    AGENT = "agent"
    TOOL = "tool"
    PIPELINE = "pipeline"


class ToolSafetyLevel(str, Enum):
    """Tool safety levels for execution."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    RESTRICTED = "restricted"


class ProvenanceGWPSet(str, Enum):
    """Supported GWP (Global Warming Potential) sets."""
    AR6GWP100 = "AR6GWP100"
    AR5GWP100 = "AR5GWP100"
    AR4GWP100 = "AR4GWP100"
    SAR = "SAR"


# =============================================================================
# Data Models - Input/Output Fields
# =============================================================================

class FieldConstraint(BaseModel):
    """Constraint definition for field validation."""

    type: str = Field(..., description="Constraint type: min, max, pattern, enum, etc.")
    value: Any = Field(..., description="Constraint value")
    message: Optional[str] = Field(None, description="Custom error message")


class InputField(BaseModel):
    """Input field definition extracted from AgentSpec."""

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type: string, number, integer, boolean, array, object")
    description: Optional[str] = Field(None, description="Field description")
    required: bool = Field(True, description="Whether field is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[Any]] = Field(None, description="Allowed values")
    minimum: Optional[float] = Field(None, description="Minimum value for numbers")
    maximum: Optional[float] = Field(None, description="Maximum value for numbers")
    min_length: Optional[int] = Field(None, description="Minimum length for strings")
    max_length: Optional[int] = Field(None, description="Maximum length for strings")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")
    unit: Optional[str] = Field(None, description="Unit of measurement")

    @property
    def python_type(self) -> str:
        """Convert JSON Schema type to Python type hint."""
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
            "null": "None",
        }
        return type_mapping.get(self.type, "Any")

    @property
    def pydantic_field_args(self) -> str:
        """Generate Pydantic Field() arguments."""
        args = []

        if not self.required and self.default is None:
            args.append("None")
        elif self.default is not None:
            if isinstance(self.default, str):
                args.append(f'"{self.default}"')
            else:
                args.append(str(self.default))
        else:
            args.append("...")

        if self.description:
            args.append(f'description="{self.description}"')
        if self.minimum is not None:
            args.append(f"ge={self.minimum}")
        if self.maximum is not None:
            args.append(f"le={self.maximum}")
        if self.min_length is not None:
            args.append(f"min_length={self.min_length}")
        if self.max_length is not None:
            args.append(f"max_length={self.max_length}")

        return ", ".join(args)


class OutputField(BaseModel):
    """Output field definition extracted from AgentSpec."""

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type")
    description: Optional[str] = Field(None, description="Field description")
    required: bool = Field(True, description="Whether field is required")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    pattern: Optional[str] = Field(None, description="Regex pattern for validation")

    @property
    def python_type(self) -> str:
        """Convert JSON Schema type to Python type hint."""
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
            "null": "None",
        }
        return type_mapping.get(self.type, "Any")


# =============================================================================
# Data Models - Tools
# =============================================================================

class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: Optional[str] = Field(None, description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[Any]] = Field(None, description="Allowed values")


class ToolDefinition(BaseModel):
    """Tool definition extracted from AgentSpec."""

    name: str = Field(..., description="Tool name (snake_case)")
    description: str = Field(..., description="Tool description")
    impl: str = Field(..., description="Implementation path: python://module.path:function")
    safe: bool = Field(True, description="Whether tool is safe (deterministic)")
    schema_in: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    schema_out: Dict[str, Any] = Field(default_factory=dict, description="Output schema")

    @property
    def module_path(self) -> str:
        """Extract module path from impl string."""
        if self.impl.startswith("python://"):
            path = self.impl[9:]  # Remove "python://"
            return path.split(":")[0] if ":" in path else path
        return self.impl

    @property
    def function_name(self) -> str:
        """Extract function name from impl string."""
        if ":" in self.impl:
            return self.impl.split(":")[-1]
        return self.name

    @property
    def input_parameters(self) -> List[ToolParameter]:
        """Parse input parameters from schema."""
        params = []
        properties = self.schema_in.get("properties", {})
        required = self.schema_in.get("required", [])

        for name, schema in properties.items():
            params.append(ToolParameter(
                name=name,
                type=schema.get("type", "string"),
                description=schema.get("description"),
                required=name in required,
                default=schema.get("default"),
                enum=schema.get("enum"),
            ))

        return params

    @property
    def output_parameters(self) -> List[ToolParameter]:
        """Parse output parameters from schema."""
        params = []
        properties = self.schema_out.get("properties", {})
        required = self.schema_out.get("required", [])

        for name, schema in properties.items():
            params.append(ToolParameter(
                name=name,
                type=schema.get("type", "string"),
                description=schema.get("description"),
                required=name in required,
            ))

        return params

    @property
    def class_name(self) -> str:
        """Generate class name for tool wrapper."""
        # Convert snake_case to PascalCase
        parts = self.name.split("_")
        return "".join(part.capitalize() for part in parts) + "Tool"


# =============================================================================
# Data Models - AI/Prompts
# =============================================================================

class AIBudget(BaseModel):
    """AI usage budget configuration."""

    max_cost_usd: float = Field(1.0, ge=0, description="Maximum cost in USD")
    max_input_tokens: int = Field(15000, ge=100, description="Maximum input tokens")
    max_output_tokens: int = Field(2000, ge=100, description="Maximum output tokens")
    max_retries: int = Field(3, ge=1, le=10, description="Maximum retry attempts")


class AIConfig(BaseModel):
    """AI configuration extracted from AgentSpec."""

    json_mode: bool = Field(True, description="Enable JSON output mode")
    system_prompt: str = Field(..., description="System prompt for the agent")
    budget: AIBudget = Field(default_factory=AIBudget, description="AI budget configuration")
    rag_collections: List[str] = Field(default_factory=list, description="RAG collections to use")
    tools: List[ToolDefinition] = Field(default_factory=list, description="Available tools")

    @property
    def system_prompt_escaped(self) -> str:
        """Return properly escaped system prompt for code generation."""
        return self.system_prompt.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")


# =============================================================================
# Data Models - Provenance
# =============================================================================

class ProvenanceConfig(BaseModel):
    """Provenance tracking configuration."""

    pin_ef: bool = Field(True, description="Pin emission factors")
    gwp_set: ProvenanceGWPSet = Field(ProvenanceGWPSet.AR6GWP100, description="GWP set to use")
    record: List[str] = Field(
        default_factory=lambda: ["inputs", "outputs", "factors", "timestamp"],
        description="Fields to record in provenance"
    )

    @property
    def should_hash_inputs(self) -> bool:
        """Check if inputs should be hashed."""
        return "inputs" in self.record or "inputs_hash" in self.record

    @property
    def should_hash_outputs(self) -> bool:
        """Check if outputs should be hashed."""
        return "outputs" in self.record


# =============================================================================
# Data Models - Tests
# =============================================================================

class GoldenTest(BaseModel):
    """Golden test case definition."""

    name: str = Field(..., description="Test case name")
    description: Optional[str] = Field(None, description="Test description")
    input: Dict[str, Any] = Field(..., description="Test input data")
    expect: Dict[str, Any] = Field(..., description="Expected output values with tolerances")

    @property
    def test_method_name(self) -> str:
        """Generate test method name."""
        # Convert name to valid Python identifier
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
        return f"test_{clean_name}"


class PropertyTest(BaseModel):
    """Property-based test definition."""

    name: str = Field(..., description="Property name")
    rule: str = Field(..., description="Property rule expression")
    description: Optional[str] = Field(None, description="Property description")
    tolerance: Optional[float] = Field(None, description="Numeric tolerance")


class TestConfig(BaseModel):
    """Test configuration extracted from AgentSpec."""

    golden: List[GoldenTest] = Field(default_factory=list, description="Golden test cases")
    properties: List[PropertyTest] = Field(default_factory=list, description="Property-based tests")


# =============================================================================
# Data Models - Author/Metadata
# =============================================================================

class Author(BaseModel):
    """Author information."""

    name: str = Field(..., description="Author name")
    email: Optional[str] = Field(None, description="Author email")
    organization: Optional[str] = Field(None, description="Organization name")


# =============================================================================
# Main Parsed AgentSpec Model
# =============================================================================

class ParsedAgentSpec(BaseModel):
    """
    Complete parsed AgentSpec with all components extracted.

    This is the main output of the AgentSpecParser, containing all
    information needed for code generation.
    """

    # Identity
    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    id: str = Field(..., description="Agent ID (unique identifier)")
    kind: PackKind = Field(PackKind.PACK, description="Pack kind")
    pack_schema_version: str = Field("1.0", description="Pack schema version")

    # Metadata
    summary: Optional[str] = Field(None, description="Agent summary")
    author: Optional[Author] = Field(None, description="Author information")
    license: str = Field("Apache-2.0", description="License identifier")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags")
    owners: List[str] = Field(default_factory=list, description="Owner identifiers")

    # AI Configuration
    ai: Optional[AIConfig] = Field(None, description="AI configuration")

    # Provenance
    provenance: Optional[ProvenanceConfig] = Field(None, description="Provenance configuration")

    # Tests
    tests: Optional[TestConfig] = Field(None, description="Test configuration")

    # Extracted Fields (parsed from tests/tools)
    inputs: List[InputField] = Field(default_factory=list, description="Input fields")
    outputs: List[OutputField] = Field(default_factory=list, description="Output fields")

    # Source
    source_file: Optional[str] = Field(None, description="Source YAML file path")
    source_hash: Optional[str] = Field(None, description="SHA-256 hash of source file")
    parsed_at: datetime = Field(default_factory=datetime.utcnow, description="Parse timestamp")

    @property
    def agent_class_name(self) -> str:
        """Generate Python class name for the agent."""
        # Convert name to PascalCase
        words = re.sub(r"[^a-zA-Z0-9\s]", " ", self.name).split()
        return "".join(word.capitalize() for word in words) + "Agent"

    @property
    def module_name(self) -> str:
        """Generate Python module name."""
        # Convert id to valid module name
        if "/" in self.id:
            name = self.id.split("/")[-1]
        else:
            name = self.id
        return re.sub(r"[^a-z0-9_]", "_", name.lower())

    @property
    def tools(self) -> List[ToolDefinition]:
        """Get all tools from AI config."""
        if self.ai and self.ai.tools:
            return self.ai.tools
        return []

    @property
    def system_prompt(self) -> Optional[str]:
        """Get system prompt from AI config."""
        if self.ai:
            return self.ai.system_prompt
        return None

    @property
    def has_provenance(self) -> bool:
        """Check if provenance tracking is configured."""
        return self.provenance is not None

    @property
    def has_tests(self) -> bool:
        """Check if tests are defined."""
        return self.tests is not None and (
            len(self.tests.golden) > 0 or len(self.tests.properties) > 0
        )

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(Exception):
    """Raised when AgentSpec validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.errors:
            error_list = "\n  - ".join(self.errors)
            return f"{self.message}\n  - {error_list}"
        return self.message


# =============================================================================
# AgentSpec Parser
# =============================================================================

class AgentSpecParser:
    """
    Parser for AgentSpec v2 YAML files.

    Parses pack.yaml files and extracts all components needed for
    code generation: inputs, outputs, tools, prompts, and provenance.

    Example:
        >>> parser = AgentSpecParser()
        >>> spec = parser.parse("path/to/pack.yaml")
        >>> print(spec.name)
        >>> print(spec.tools)
    """

    # Required fields for validation
    REQUIRED_FIELDS = ["name", "version", "kind"]

    # Supported pack schema versions
    SUPPORTED_SCHEMA_VERSIONS = ["1.0", "1.1", "2.0"]

    def __init__(self, strict: bool = True):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors on validation failures.
                   If False, log warnings and continue.
        """
        self.strict = strict
        self._warnings: List[str] = []
        self._errors: List[str] = []

    def parse(self, yaml_path: Union[str, Path]) -> ParsedAgentSpec:
        """
        Parse an AgentSpec YAML file.

        Args:
            yaml_path: Path to pack.yaml file

        Returns:
            Parsed AgentSpec with all components extracted

        Raises:
            ValidationError: If validation fails in strict mode
            FileNotFoundError: If file does not exist
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"AgentSpec file not found: {yaml_path}")

        logger.info(f"Parsing AgentSpec from: {yaml_path}")

        # Read and parse YAML
        content = yaml_path.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(content.encode()).hexdigest()

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax: {e}")

        if not isinstance(data, dict):
            raise ValidationError("AgentSpec must be a YAML mapping")

        # Validate required fields
        self._validate_required_fields(data)

        # Parse all components
        spec = self._parse_spec(data, str(yaml_path), source_hash)

        # Extract inputs/outputs from tests if not explicitly defined
        if not spec.inputs:
            spec.inputs = self._extract_inputs_from_tests(data)
        if not spec.outputs:
            spec.outputs = self._extract_outputs_from_tests(data)

        # Final validation
        self._validate_spec(spec)

        if self.strict and self._errors:
            raise ValidationError("AgentSpec validation failed", self._errors)

        if self._warnings:
            for warning in self._warnings:
                logger.warning(warning)

        logger.info(f"Successfully parsed AgentSpec: {spec.name} v{spec.version}")
        return spec

    def parse_string(self, yaml_content: str) -> ParsedAgentSpec:
        """
        Parse AgentSpec from YAML string.

        Args:
            yaml_content: YAML content as string

        Returns:
            Parsed AgentSpec
        """
        source_hash = hashlib.sha256(yaml_content.encode()).hexdigest()

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax: {e}")

        if not isinstance(data, dict):
            raise ValidationError("AgentSpec must be a YAML mapping")

        self._validate_required_fields(data)
        spec = self._parse_spec(data, "<string>", source_hash)

        if not spec.inputs:
            spec.inputs = self._extract_inputs_from_tests(data)
        if not spec.outputs:
            spec.outputs = self._extract_outputs_from_tests(data)

        self._validate_spec(spec)

        if self.strict and self._errors:
            raise ValidationError("AgentSpec validation failed", self._errors)

        return spec

    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """Validate that required fields are present."""
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing)}"
            )

    def _parse_spec(
        self,
        data: Dict[str, Any],
        source_file: str,
        source_hash: str
    ) -> ParsedAgentSpec:
        """Parse raw YAML data into ParsedAgentSpec."""

        # Parse author
        author = None
        if "author" in data and isinstance(data["author"], dict):
            author = Author(
                name=data["author"].get("name", "Unknown"),
                email=data["author"].get("email"),
                organization=data["author"].get("organization"),
            )

        # Parse AI config
        ai_config = None
        if "ai" in data and isinstance(data["ai"], dict):
            ai_data = data["ai"]

            # Parse tools
            tools = []
            for tool_data in ai_data.get("tools", []):
                tools.append(ToolDefinition(
                    name=tool_data.get("name", "unknown"),
                    description=tool_data.get("description", ""),
                    impl=tool_data.get("impl", ""),
                    safe=tool_data.get("safe", True),
                    schema_in=tool_data.get("schema_in", {}),
                    schema_out=tool_data.get("schema_out", {}),
                ))

            # Parse budget
            budget_data = ai_data.get("budget", {})
            budget = AIBudget(
                max_cost_usd=budget_data.get("max_cost_usd", 1.0),
                max_input_tokens=budget_data.get("max_input_tokens", 15000),
                max_output_tokens=budget_data.get("max_output_tokens", 2000),
                max_retries=budget_data.get("max_retries", 3),
            )

            ai_config = AIConfig(
                json_mode=ai_data.get("json_mode", True),
                system_prompt=ai_data.get("system_prompt", ""),
                budget=budget,
                rag_collections=ai_data.get("rag_collections", []),
                tools=tools,
            )

        # Parse provenance config
        provenance = None
        if "provenance" in data and isinstance(data["provenance"], dict):
            prov_data = data["provenance"]
            gwp_set = prov_data.get("gwp_set", "AR6GWP100")

            # Map string to enum
            try:
                gwp_enum = ProvenanceGWPSet(gwp_set)
            except ValueError:
                gwp_enum = ProvenanceGWPSet.AR6GWP100
                self._warnings.append(f"Unknown GWP set '{gwp_set}', defaulting to AR6GWP100")

            provenance = ProvenanceConfig(
                pin_ef=prov_data.get("pin_ef", True),
                gwp_set=gwp_enum,
                record=prov_data.get("record", ["inputs", "outputs", "factors", "timestamp"]),
            )

        # Parse tests
        tests = None
        if "tests" in data and isinstance(data["tests"], dict):
            tests_data = data["tests"]

            golden_tests = []
            for test in tests_data.get("golden", []):
                golden_tests.append(GoldenTest(
                    name=test.get("name", "unnamed"),
                    description=test.get("description"),
                    input=test.get("input", {}),
                    expect=test.get("expect", {}),
                ))

            property_tests = []
            for prop in tests_data.get("properties", []):
                property_tests.append(PropertyTest(
                    name=prop.get("name", "unnamed"),
                    rule=prop.get("rule", ""),
                    description=prop.get("description"),
                    tolerance=prop.get("tolerance"),
                ))

            tests = TestConfig(
                golden=golden_tests,
                properties=property_tests,
            )

        # Build ParsedAgentSpec
        return ParsedAgentSpec(
            name=data.get("name", "Unknown Agent"),
            version=data.get("version", "0.0.1"),
            id=data.get("id", "unknown/agent_v1"),
            kind=PackKind(data.get("kind", "pack")),
            pack_schema_version=data.get("pack_schema_version", "1.0"),
            summary=data.get("summary"),
            author=author,
            license=data.get("license", "Apache-2.0"),
            tags=data.get("tags", []),
            owners=data.get("owners", []),
            ai=ai_config,
            provenance=provenance,
            tests=tests,
            source_file=source_file,
            source_hash=source_hash,
        )

    def _extract_inputs_from_tests(self, data: Dict[str, Any]) -> List[InputField]:
        """Extract input field definitions from test cases."""
        inputs: Dict[str, InputField] = {}

        tests = data.get("tests", {})
        golden = tests.get("golden", [])

        for test in golden:
            test_input = test.get("input", {})
            for name, value in test_input.items():
                if name not in inputs:
                    # Infer type from value
                    field_type = self._infer_type(value)
                    inputs[name] = InputField(
                        name=name,
                        type=field_type,
                        description=f"Input field: {name}",
                        required=True,
                    )

        return list(inputs.values())

    def _extract_outputs_from_tests(self, data: Dict[str, Any]) -> List[OutputField]:
        """Extract output field definitions from test expectations."""
        outputs: Dict[str, OutputField] = {}

        tests = data.get("tests", {})
        golden = tests.get("golden", [])

        for test in golden:
            expect = test.get("expect", {})
            for name, value in expect.items():
                if name not in outputs:
                    # Handle both simple values and dict with value/tol
                    if isinstance(value, dict):
                        actual_value = value.get("value", value)
                    else:
                        actual_value = value

                    field_type = self._infer_type(actual_value)
                    outputs[name] = OutputField(
                        name=name,
                        type=field_type,
                        description=f"Output field: {name}",
                        required=True,
                    )

        return list(outputs.values())

    def _infer_type(self, value: Any) -> str:
        """Infer JSON Schema type from Python value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"

    def _validate_spec(self, spec: ParsedAgentSpec) -> None:
        """Validate the parsed spec for completeness and consistency."""

        # Check for valid ID format
        if not re.match(r"^[a-z0-9_/-]+$", spec.id.lower()):
            self._warnings.append(
                f"Agent ID '{spec.id}' contains non-standard characters"
            )

        # Check for system prompt if AI is configured
        if spec.ai and not spec.ai.system_prompt:
            self._errors.append("AI configuration requires a system_prompt")

        # Check tools have valid implementations
        for tool in spec.tools:
            if not tool.impl:
                self._errors.append(f"Tool '{tool.name}' missing implementation path")
            elif not tool.impl.startswith("python://"):
                self._warnings.append(
                    f"Tool '{tool.name}' has non-standard impl format: {tool.impl}"
                )

        # Check for golden tests if testing is important
        if spec.tests and not spec.tests.golden:
            self._warnings.append("No golden tests defined")

    @property
    def warnings(self) -> List[str]:
        """Get list of warnings from last parse."""
        return self._warnings.copy()

    @property
    def errors(self) -> List[str]:
        """Get list of errors from last parse."""
        return self._errors.copy()
