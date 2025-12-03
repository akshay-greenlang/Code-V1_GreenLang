"""
GreenLang Agent Generator Package.

This package provides the Agent Generator that transforms AgentSpec YAML
specifications into production-ready Python agent code.

Components:
- spec_parser: Parse and validate AgentSpec YAML files
- code_generator: Generate Python code from parsed specs
- templates: Jinja2 templates for code generation

Example:
    >>> from greenlang.generator import AgentSpecParser, CodeGenerator
    >>> parser = AgentSpecParser()
    >>> spec = parser.parse("path/to/pack.yaml")
    >>> generator = CodeGenerator()
    >>> generator.generate(spec, output_dir="./generated")
"""

from greenlang.generator.spec_parser import (
    AgentSpecParser,
    ParsedAgentSpec,
    InputField,
    OutputField,
    ToolDefinition,
    ProvenanceConfig,
    ValidationError,
)

from greenlang.generator.code_generator import (
    CodeGenerator,
    GeneratedCode,
    GenerationOptions,
)

__all__ = [
    # Parser
    "AgentSpecParser",
    "ParsedAgentSpec",
    "InputField",
    "OutputField",
    "ToolDefinition",
    "ProvenanceConfig",
    "ValidationError",
    # Generator
    "CodeGenerator",
    "GeneratedCode",
    "GenerationOptions",
]

__version__ = "1.0.0"
