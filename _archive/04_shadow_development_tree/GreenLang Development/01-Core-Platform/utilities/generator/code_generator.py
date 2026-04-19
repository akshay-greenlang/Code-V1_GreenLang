"""
Code Generator - Generate Python agent code from ParsedAgentSpec.

This module generates complete agent packages from parsed AgentSpec
specifications using Jinja2 templates.

Generated components:
- agent.py: Main agent class with lifecycle hooks
- tools.py: Tool wrapper classes for zero-hallucination
- test_agent.py: Unit test suite

Example:
    >>> from greenlang.generator import AgentSpecParser, CodeGenerator
    >>> parser = AgentSpecParser()
    >>> spec = parser.parse("path/to/pack.yaml")
    >>> generator = CodeGenerator()
    >>> result = generator.generate(spec, output_dir="./generated")
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.greenlang.generator.spec_parser import (
    ParsedAgentSpec,
    InputField,
    OutputField,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationOptions:
    """Options for code generation."""

    # Output options
    output_dir: Optional[Path] = None
    overwrite: bool = False

    # Component options
    generate_tools: bool = True
    generate_tests: bool = True
    generate_readme: bool = True
    generate_init: bool = True

    # Code style options
    line_length: int = 88
    use_black_formatting: bool = True
    docstring_style: str = "google"  # google, numpy, sphinx

    # Template options
    template_dir: Optional[Path] = None

    # SDK options
    sdk_import_path: str = "greenlang_sdk.core"
    use_async: bool = True


@dataclass
class GeneratedFile:
    """Represents a generated file."""

    filename: str
    content: str
    relative_path: str = ""

    @property
    def full_path(self) -> str:
        """Get full relative path."""
        if self.relative_path:
            return f"{self.relative_path}/{self.filename}"
        return self.filename


@dataclass
class GeneratedCode:
    """Complete generated code package."""

    agent_code: str
    tools_code: str
    tests_code: str
    readme_content: str
    init_content: str
    files: List[GeneratedFile] = field(default_factory=list)

    # Metadata
    spec_name: str = ""
    spec_version: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Statistics
    total_lines: int = 0
    num_tools: int = 0
    num_tests: int = 0


# =============================================================================
# Jinja2 Template Helpers
# =============================================================================

def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def json_type_to_python(json_type: str) -> str:
    """Convert JSON Schema type to Python type."""
    mapping = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "List[Any]",
        "object": "Dict[str, Any]",
        "null": "None",
    }
    return mapping.get(json_type, "Any")


def format_docstring(text: str, indent: int = 4) -> str:
    """Format text as a docstring with proper indentation."""
    if not text:
        return '""""""'

    lines = text.strip().split("\n")
    indent_str = " " * indent

    if len(lines) == 1:
        return f'"""{lines[0]}"""'

    formatted = ['"""']
    for line in lines:
        formatted.append(f"{indent_str}{line}" if line.strip() else "")
    formatted.append(f'{indent_str}"""')
    return "\n".join(formatted)


def escape_string(text: str) -> str:
    """Escape a string for use in Python code."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# =============================================================================
# Code Generator
# =============================================================================

class CodeGenerator:
    """
    Generate Python agent code from ParsedAgentSpec.

    Uses Jinja2 templates to generate:
    - agent.py: Main agent class
    - tools.py: Tool wrapper classes
    - test_agent.py: Unit tests

    Example:
        >>> generator = CodeGenerator()
        >>> result = generator.generate(spec, output_dir="./generated")
        >>> print(result.agent_code)
    """

    def __init__(self, options: Optional[GenerationOptions] = None):
        """
        Initialize code generator.

        Args:
            options: Generation options. Uses defaults if not provided.
        """
        self.options = options or GenerationOptions()
        self._setup_jinja_env()

    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with templates and filters."""
        # Determine template directory
        if self.options.template_dir:
            template_dir = self.options.template_dir
        else:
            # Default to templates directory next to this file
            template_dir = Path(__file__).parent / "templates"

        # Create environment
        if template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            # Fallback to inline templates
            logger.warning(f"Template directory not found: {template_dir}")
            self.env = Environment(
                trim_blocks=True,
                lstrip_blocks=True,
            )

        # Add custom filters
        self.env.filters["snake_to_pascal"] = snake_to_pascal
        self.env.filters["snake_to_camel"] = snake_to_camel
        self.env.filters["json_type_to_python"] = json_type_to_python
        self.env.filters["format_docstring"] = format_docstring
        self.env.filters["escape_string"] = escape_string

        # Add global functions
        self.env.globals["now"] = datetime.utcnow

    def generate(
        self,
        spec: ParsedAgentSpec,
        output_dir: Optional[Path] = None
    ) -> GeneratedCode:
        """
        Generate complete agent package from spec.

        Args:
            spec: Parsed AgentSpec
            output_dir: Output directory. If provided, writes files to disk.

        Returns:
            GeneratedCode with all generated components
        """
        logger.info(f"Generating agent code for: {spec.name} v{spec.version}")

        # Build template context
        context = self._build_context(spec)

        # Generate all components
        agent_code = self._generate_agent(context)
        tools_code = self._generate_tools(context)
        tests_code = self._generate_tests(context)
        readme_content = self._generate_readme(context)
        init_content = self._generate_init(context)

        # Create result
        result = GeneratedCode(
            agent_code=agent_code,
            tools_code=tools_code,
            tests_code=tests_code,
            readme_content=readme_content,
            init_content=init_content,
            spec_name=spec.name,
            spec_version=spec.version,
            num_tools=len(spec.tools),
            num_tests=len(spec.tests.golden) if spec.tests else 0,
        )

        # Calculate statistics
        result.total_lines = sum(
            code.count("\n") + 1
            for code in [agent_code, tools_code, tests_code]
        )

        # Build file list
        result.files = [
            GeneratedFile("agent.py", agent_code),
            GeneratedFile("tools.py", tools_code),
            GeneratedFile("test_agent.py", tests_code, "tests"),
            GeneratedFile("README.md", readme_content),
            GeneratedFile("__init__.py", init_content),
        ]

        # Write to disk if output_dir provided
        if output_dir:
            self._write_files(result, Path(output_dir))

        logger.info(f"Generated {result.total_lines} lines of code across {len(result.files)} files")
        return result

    def _build_context(self, spec: ParsedAgentSpec) -> Dict[str, Any]:
        """Build template context from spec."""
        return {
            # Spec data
            "spec": spec,
            "name": spec.name,
            "version": spec.version,
            "id": spec.id,
            "summary": spec.summary or f"{spec.name} agent",
            "license": spec.license,

            # Class names
            "agent_class_name": spec.agent_class_name,
            "module_name": spec.module_name,

            # Components
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "tools": spec.tools,
            "tests": spec.tests,
            "provenance": spec.provenance,
            "ai": spec.ai,

            # System prompt (escaped)
            "system_prompt": spec.system_prompt or "",

            # Options
            "use_async": self.options.use_async,
            "sdk_import_path": self.options.sdk_import_path,

            # Metadata
            "generated_at": datetime.utcnow().isoformat(),
            "generator_version": "1.0.0",
        }

    def _generate_agent(self, context: Dict[str, Any]) -> str:
        """Generate agent.py code."""
        try:
            template = self.env.get_template("agent_class.py.j2")
            return template.render(**context)
        except Exception as e:
            logger.warning(f"Template not found, using inline generation: {e}")
            return self._generate_agent_inline(context)

    def _generate_tools(self, context: Dict[str, Any]) -> str:
        """Generate tools.py code."""
        try:
            template = self.env.get_template("tools.py.j2")
            return template.render(**context)
        except Exception as e:
            logger.warning(f"Template not found, using inline generation: {e}")
            return self._generate_tools_inline(context)

    def _generate_tests(self, context: Dict[str, Any]) -> str:
        """Generate test_agent.py code."""
        try:
            template = self.env.get_template("test_agent.py.j2")
            return template.render(**context)
        except Exception as e:
            logger.warning(f"Template not found, using inline generation: {e}")
            return self._generate_tests_inline(context)

    def _generate_readme(self, context: Dict[str, Any]) -> str:
        """Generate README.md content."""
        spec = context["spec"]

        return f'''# {spec.name}

> {spec.summary or "GreenLang Agent"}

**Version:** {spec.version}
**License:** {spec.license}
**ID:** {spec.id}

## Overview

This agent was generated from an AgentSpec YAML specification using the GreenLang Agent Generator.

## Installation

```bash
pip install -e .
```

## Usage

```python
from {spec.module_name} import {spec.agent_class_name}

# Initialize agent
agent = {spec.agent_class_name}()

# Run agent
result = await agent.run({{
    # Your input data here
}})

print(result.output)
print(result.provenance)
```

## Tools

This agent includes the following tools:

{self._generate_tools_docs(spec.tools)}

## Tests

Run tests with:

```bash
pytest tests/
```

## Provenance

This agent tracks complete provenance for audit trails including:
- Input/output hashes (SHA-256)
- Tool call records
- Emission factor citations
- Timestamp tracking

---

Generated with GreenLang Agent Generator v1.0.0
'''

    def _generate_tools_docs(self, tools: List[ToolDefinition]) -> str:
        """Generate documentation for tools."""
        if not tools:
            return "No tools configured."

        docs = []
        for tool in tools:
            doc = f"### {tool.name}\n\n{tool.description}\n"
            if tool.input_parameters:
                doc += "\n**Parameters:**\n"
                for param in tool.input_parameters:
                    req = " (required)" if param.required else ""
                    doc += f"- `{param.name}`: {param.type}{req}\n"
            docs.append(doc)

        return "\n".join(docs)

    def _generate_init(self, context: Dict[str, Any]) -> str:
        """Generate __init__.py content."""
        spec = context["spec"]

        return f'''"""
{spec.name} - {spec.summary or "GreenLang Agent"}

Version: {spec.version}
License: {spec.license}
"""

from {spec.module_name}.agent import {spec.agent_class_name}
from {spec.module_name}.tools import *

__all__ = [
    "{spec.agent_class_name}",
]

__version__ = "{spec.version}"
'''

    # =========================================================================
    # Inline Template Generation (Fallback)
    # =========================================================================

    def _generate_agent_inline(self, context: Dict[str, Any]) -> str:
        """Generate agent code without templates (fallback)."""
        spec: ParsedAgentSpec = context["spec"]

        # Build input class fields
        input_fields = []
        for inp in spec.inputs:
            optional = "Optional[" if not inp.required else ""
            close = "]" if not inp.required else ""
            field_args = inp.pydantic_field_args
            input_fields.append(
                f"    {inp.name}: {optional}{inp.python_type}{close} = Field({field_args})"
            )

        # Build output class fields
        output_fields = []
        for out in spec.outputs:
            optional = "Optional[" if not out.required else ""
            close = "]" if not out.required else ""
            desc = f', description="{out.description}"' if out.description else ""
            output_fields.append(
                f"    {out.name}: {optional}{out.python_type}{close} = Field(...{desc})"
            )

        # Build tool method calls
        tool_methods = []
        for tool in spec.tools:
            tool_methods.append(f'''
    async def call_{tool.name}(self, **kwargs) -> Dict[str, Any]:
        """
        Call {tool.name} tool.

        {tool.description}
        """
        result = await self._execute_tool("{tool.name}", kwargs)
        self.record_tool_call("{tool.name}", kwargs, result)
        return result
''')

        # System prompt (escaped for multi-line string)
        system_prompt = spec.system_prompt or ""
        system_prompt_escaped = system_prompt.replace('"""', '\\"\\"\\"')

        return f'''"""
{spec.name} Agent - {spec.summary or "Generated GreenLang Agent"}

This module implements the {spec.agent_class_name} for the GreenLang platform.
Generated from AgentSpec: {spec.id}

Version: {spec.version}
License: {spec.license}
Generated: {datetime.utcnow().isoformat()}
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from {self.options.sdk_import_path}.agent_base import SDKAgentBase, AgentResult
from {self.options.sdk_import_path}.provenance import ProvenanceTracker

from .tools import *

logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class {spec.agent_class_name}Input(BaseModel):
    """Input data model for {spec.agent_class_name}."""

{chr(10).join(input_fields) if input_fields else "    pass"}


class {spec.agent_class_name}Output(BaseModel):
    """Output data model for {spec.agent_class_name}."""

{chr(10).join(output_fields) if output_fields else "    pass"}

    # Standard provenance fields
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance hash")
    processing_time_ms: Optional[float] = Field(None, description="Processing duration")


# =============================================================================
# Agent Implementation
# =============================================================================

class {spec.agent_class_name}(SDKAgentBase[{spec.agent_class_name}Input, {spec.agent_class_name}Output]):
    """
    {spec.name} Agent Implementation.

    {spec.summary or "Generated from AgentSpec."}

    This agent follows GreenLang's zero-hallucination principle:
    - All calculations use deterministic tools
    - Complete provenance tracking with SHA-256 hashes
    - Full audit trail for regulatory compliance

    Attributes:
        agent_id: Unique identifier for this agent
        agent_version: Version of this agent

    Example:
        >>> agent = {spec.agent_class_name}()
        >>> result = await agent.run({{"key": "value"}})
        >>> print(result.output)
    """

    # System prompt for AI orchestration
    SYSTEM_PROMPT = """{system_prompt_escaped}"""

    def __init__(
        self,
        agent_id: str = "{spec.id}",
        agent_version: str = "{spec.version}",
        enable_provenance: bool = True,
        enable_citations: bool = True,
    ):
        """Initialize {spec.agent_class_name}."""
        super().__init__(
            agent_id=agent_id,
            agent_version=agent_version,
            enable_provenance=enable_provenance,
            enable_citations=enable_citations,
        )

        # Initialize tool registry
        self._tools: Dict[str, Any] = {{}}
        self._register_tools()

        logger.info(f"Initialized {{self.agent_id}} v{{self.agent_version}}")

    def _register_tools(self) -> None:
        """Register available tools."""
{self._generate_tool_registrations(spec.tools)}

    async def validate_input(
        self,
        input_data: {spec.agent_class_name}Input,
        context: dict
    ) -> {spec.agent_class_name}Input:
        """
        Validate input data against schema.

        Args:
            input_data: Raw input data
            context: Execution context

        Returns:
            Validated input data

        Raises:
            ValidationError: If validation fails
        """
        # Pydantic handles basic validation via the model
        # Add custom validation logic here
        logger.debug(f"Validating input for {{self.agent_id}}")

        return input_data

    async def execute(
        self,
        validated_input: {spec.agent_class_name}Input,
        context: dict
    ) -> {spec.agent_class_name}Output:
        """
        Execute main agent logic.

        ZERO-HALLUCINATION: All calculations use deterministic tools.
        No LLM calls are made for numeric calculations.

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Processed output with provenance

        Raises:
            ExecutionError: If execution fails
        """
        start_time = datetime.utcnow()
        logger.info(f"Executing {{self.agent_id}}")

        try:
            # Execute core logic using tools
            result_data = await self._execute_core_logic(validated_input, context)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Build output
            output = {spec.agent_class_name}Output(
                **result_data,
                processing_time_ms=processing_time,
            )

            logger.info(f"{{self.agent_id}} execution completed in {{processing_time:.2f}}ms")
            return output

        except Exception as e:
            logger.error(f"{{self.agent_id}} execution failed: {{e}}", exc_info=True)
            raise

    async def _execute_core_logic(
        self,
        input_data: {spec.agent_class_name}Input,
        context: dict
    ) -> Dict[str, Any]:
        """
        Execute core business logic.

        IMPORTANT: This method must use ONLY deterministic tools.
        No LLM calls for calculations.

        Override this method to implement custom logic.

        Args:
            input_data: Validated input
            context: Execution context

        Returns:
            Dictionary of output values
        """
                # Core logic implementation
        # Example:
        # result = await self.call_calculate_emissions(
        #     fuel_type=input_data.fuel_type,
        #     quantity=input_data.quantity
        # )
        # return {{"emissions_tco2e": result["value"]}}

        return {{}}

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {{tool_name}}")

        tool = self._tools[tool_name]
        return await tool.execute(params)

    # =========================================================================
    # Tool Methods
    # =========================================================================
{"".join(tool_methods)}

# =============================================================================
# Convenience Functions
# =============================================================================

async def create_agent(**kwargs) -> {spec.agent_class_name}:
    """Factory function to create agent instance."""
    return {spec.agent_class_name}(**kwargs)
'''

    def _generate_tool_registrations(self, tools: List[ToolDefinition]) -> str:
        """Generate tool registration code."""
        if not tools:
            return "        pass  # No tools to register"

        lines = []
        for tool in tools:
            lines.append(f'        self._tools["{tool.name}"] = {tool.class_name}()')

        return "\n".join(lines)

    def _generate_tools_inline(self, context: Dict[str, Any]) -> str:
        """Generate tools code without templates (fallback)."""
        spec: ParsedAgentSpec = context["spec"]

        tool_classes = []
        for tool in spec.tools:
            # Build parameter definitions
            params = []
            for param in tool.input_parameters:
                python_type = json_type_to_python(param.type)
                default = f" = {repr(param.default)}" if param.default is not None else ""
                params.append(f"        {param.name}: {python_type}{default}")

            params_str = ",\n".join(params) if params else "        # No parameters"

            # Build return type hints
            returns = []
            for ret in tool.output_parameters:
                python_type = json_type_to_python(ret.type)
                returns.append(f'            "{ret.name}": {python_type}')

            returns_str = ",\n".join(returns) if returns else '            "result": Any'

            tool_classes.append(f'''
class {tool.class_name}:
    """
    {tool.description}

    Implementation: {tool.impl}
    Safe: {tool.safe}
    """

    def __init__(self):
        """Initialize {tool.class_name}."""
        self.name = "{tool.name}"
        self.safe = {tool.safe}

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Tool execution result
        """
            # Tool logic implementation
        # This should call the actual implementation at {tool.impl}

        # Validate required parameters
{self._generate_param_validation(tool.input_parameters)}

        # Execute tool logic (ZERO-HALLUCINATION)
        result = await self._execute_internal(params)

        return result

    async def _execute_internal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal execution logic.

        Override this method to implement actual tool functionality.
        """
        # Placeholder implementation
        return {{"status": "success", "tool": self.name}}

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
            # Schema-based validation
        return True
''')

        return f'''"""
Tool implementations for {spec.name}.

This module provides tool wrapper classes for zero-hallucination calculations.
All tools are deterministic and track provenance.

Generated from AgentSpec: {spec.id}
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Base
# =============================================================================

class BaseTool:
    """Base class for all tools."""

    name: str = "base_tool"
    safe: bool = True

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool - must be overridden."""
        raise NotImplementedError

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters."""
        return True


# =============================================================================
# Tool Implementations
# =============================================================================
{"".join(tool_classes)}

# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, type] = {{
{self._generate_tool_registry(spec.tools)}
}}


def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool instance by name."""
    tool_class = TOOL_REGISTRY.get(name)
    if tool_class:
        return tool_class()
    return None


def list_tools() -> List[str]:
    """List all available tool names."""
    return list(TOOL_REGISTRY.keys())
'''

    def _generate_param_validation(self, params: List) -> str:
        """Generate parameter validation code."""
        if not params:
            return "        pass  # No parameters to validate"

        lines = []
        for param in params:
            if param.required:
                lines.append(
                    f'        if "{param.name}" not in params:\n'
                    f'            raise ValueError("Missing required parameter: {param.name}")'
                )

        return "\n".join(lines) if lines else "        pass  # All parameters optional"

    def _generate_tool_registry(self, tools: List[ToolDefinition]) -> str:
        """Generate tool registry dictionary."""
        if not tools:
            return "    # No tools registered"

        lines = []
        for tool in tools:
            lines.append(f'    "{tool.name}": {tool.class_name},')

        return "\n".join(lines)

    def _generate_tests_inline(self, context: Dict[str, Any]) -> str:
        """Generate test code without templates (fallback)."""
        spec: ParsedAgentSpec = context["spec"]

        # Generate golden test methods
        test_methods = []
        if spec.tests and spec.tests.golden:
            for test in spec.tests.golden:
                # Format input data
                input_str = self._format_dict(test.input, indent=12)

                # Format expected output
                expect_checks = []
                for key, value in test.expect.items():
                    if isinstance(value, dict) and "value" in value:
                        tol = value.get("tol", 0.01)
                        expect_checks.append(
                            f'        assert abs(result.output.{key} - {value["value"]}) < {tol}, '
                            f'f"Expected {key}={value["value"]}, got {{result.output.{key}}}"'
                        )
                    else:
                        expect_checks.append(
                            f'        assert result.output.{key} == {repr(value)}, '
                            f'f"Expected {key}={repr(value)}, got {{result.output.{key}}}"'
                        )

                test_methods.append(f'''
    @pytest.mark.asyncio
    async def {test.test_method_name}(self, agent: {spec.agent_class_name}):
        """
        {test.description or test.name}
        """
        input_data = {spec.agent_class_name}Input(
{input_str}
        )

        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None
{chr(10).join(expect_checks)}
''')

        # Generate property test methods
        property_tests = []
        if spec.tests and spec.tests.properties:
            for prop in spec.tests.properties:
                property_tests.append(f'''
    def test_property_{prop.name.replace("-", "_")}(self, agent: {spec.agent_class_name}):
        """
        Property: {prop.description or prop.rule}
        Rule: {prop.rule}
        """
        # Property-based testing implementation
        # Rule: {prop.rule}
        pass
''')

        # Pre-compute test sections to avoid f-string nesting issues
        golden_tests_section = "".join(test_methods) if test_methods else '''
    def test_placeholder(self, agent):
        """Placeholder test - add golden tests to AgentSpec."""
        assert agent is not None
'''

        property_tests_section = "".join(property_tests) if property_tests else '''
    def test_placeholder(self, agent):
        """Placeholder test - add property tests to AgentSpec."""
        assert agent is not None
'''

        return f'''"""
Test suite for {spec.name}.

This module provides unit tests for the {spec.agent_class_name}.
Generated from AgentSpec golden tests and property tests.

Run with: pytest tests/test_agent.py -v
"""

import pytest
from typing import Dict, Any

from {spec.module_name}.agent import {spec.agent_class_name}, {spec.agent_class_name}Input, {spec.agent_class_name}Output


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent() -> {spec.agent_class_name}:
    """Create agent instance for testing."""
    return {spec.agent_class_name}(
        enable_provenance=True,
        enable_citations=True,
    )


@pytest.fixture
def sample_input() -> Dict[str, Any]:
    """Sample input data for testing."""
    return {{
{self._generate_sample_input(spec.inputs)}
    }}


# =============================================================================
# Golden Tests (from AgentSpec)
# =============================================================================

class TestGolden:
    """Golden test cases from AgentSpec."""
{golden_tests_section}


# =============================================================================
# Property Tests
# =============================================================================

class TestProperties:
    """Property-based tests from AgentSpec."""
{property_tests_section}


# =============================================================================
# Unit Tests
# =============================================================================

class TestAgent:
    """Unit tests for {spec.agent_class_name}."""

    def test_agent_initialization(self, agent: {spec.agent_class_name}):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_id == "{spec.id}"
        assert agent.agent_version == "{spec.version}"

    def test_input_validation(self, agent: {spec.agent_class_name}, sample_input: Dict[str, Any]):
        """Test input validation."""
        input_data = {spec.agent_class_name}Input(**sample_input)
        assert input_data is not None

    @pytest.mark.asyncio
    async def test_execute_returns_output(self, agent: {spec.agent_class_name}, sample_input: Dict[str, Any]):
        """Test agent execution returns valid output."""
        input_data = {spec.agent_class_name}Input(**sample_input)
        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_provenance_tracking(self, agent: {spec.agent_class_name}, sample_input: Dict[str, Any]):
        """Test provenance is tracked correctly."""
        input_data = {spec.agent_class_name}Input(**sample_input)
        result = await agent.run(input_data)

        assert result.provenance is not None
        assert result.provenance.input_hash is not None
        assert result.provenance.output_hash is not None
        assert result.provenance.provenance_chain is not None


# =============================================================================
# Tool Tests
# =============================================================================

class TestTools:
    """Tests for agent tools."""
{self._generate_tool_tests(spec.tools, spec.agent_class_name)}
'''

    def _generate_sample_input(self, inputs: List[InputField]) -> str:
        """Generate sample input dictionary for tests."""
        if not inputs:
            return "        # No inputs defined"

        lines = []
        for inp in inputs:
            if inp.default is not None:
                value = repr(inp.default)
            elif inp.type == "string":
                value = '"sample_value"'
            elif inp.type == "number":
                value = "1.0"
            elif inp.type == "integer":
                value = "1"
            elif inp.type == "boolean":
                value = "True"
            else:
                value = "None"

            lines.append(f'        "{inp.name}": {value},')

        return "\n".join(lines)

    def _generate_tool_tests(self, tools: List[ToolDefinition], agent_class: str) -> str:
        """Generate test methods for tools."""
        if not tools:
            return '''
    def test_no_tools(self):
        """No tools configured."""
        pass
'''

        tests = []
        for tool in tools:
            tests.append(f'''
    @pytest.mark.asyncio
    async def test_{tool.name}_exists(self, agent: {agent_class}):
        """Test {tool.name} tool is registered."""
        assert "{tool.name}" in agent._tools

    @pytest.mark.asyncio
    async def test_{tool.name}_execution(self, agent: {agent_class}):
        """Test {tool.name} tool executes."""
        result = await agent.call_{tool.name}()
        assert result is not None
''')

        return "\n".join(tests)

    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary as Python code."""
        lines = []
        indent_str = " " * indent

        for key, value in d.items():
            if isinstance(value, str):
                lines.append(f'{indent_str}{key}="{value}",')
            elif isinstance(value, (int, float, bool)):
                lines.append(f'{indent_str}{key}={value},')
            elif isinstance(value, dict):
                lines.append(f'{indent_str}{key}={repr(value)},')
            else:
                lines.append(f'{indent_str}{key}={repr(value)},')

        return "\n".join(lines)

    def _write_files(self, result: GeneratedCode, output_dir: Path) -> None:
        """Write generated files to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in result.files:
            file_path = output_dir / file.relative_path / file.filename if file.relative_path else output_dir / file.filename

            # Create subdirectories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            file_path.write_text(file.content, encoding="utf-8")
            logger.info(f"Wrote: {file_path}")
