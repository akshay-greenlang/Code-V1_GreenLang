"""
GreenLang Agent Generator

A code generation system that transforms AgentSpec YAML definitions (pack.yaml)
into production-ready Python agents with zero-hallucination guarantees.

The Agent Generator ensures:
- Zero-Hallucination Compliance: All generated calculations use deterministic lookup tables
- Complete Provenance Tracking: SHA-256 hashing of all calculation steps
- GreenLang Pattern Conformance: Generated code matches existing hand-written agents
- Production Readiness: Type hints, docstrings, validation, and test generation

Architecture Overview:
    backend/agent_generator/
    |-- __init__.py                 # This file - public API exports
    |-- engine.py                   # Core generation orchestrator
    |-- config.py                   # Generator configuration
    |
    |-- parser/                     # YAML parsing and validation
    |   |-- __init__.py             # Parser exports
    |   |-- yaml_parser.py          # YAML loading with schema validation
    |   |-- spec_validator.py       # AgentSpec validation rules
    |   |-- schema.py               # Pydantic models for pack.yaml
    |
    |-- generators/                 # Code generators
    |   |-- __init__.py             # Generator exports
    |   |-- agent_gen.py            # Agent class generator
    |   |-- model_gen.py            # Pydantic model generator
    |
    |-- templates/                  # Jinja2 templates
    |   |-- __init__.py             # Template engine
    |   |-- agent/agent_class.py.j2 # Agent class template
    |   |-- tests/test_agent.py.j2  # Test suite template

Example:
    >>> from backend.agent_generator import AgentGeneratorEngine, GeneratorConfig
    >>> from pathlib import Path
    >>>
    >>> # Initialize the generator
    >>> config = GeneratorConfig()
    >>> engine = AgentGeneratorEngine(config)
    >>>
    >>> # Generate agent from pack.yaml (async)
    >>> result = await engine.generate_from_yaml(
    ...     yaml_path=Path("08-regulatory-agents/eudr/pack.yaml"),
    ...     output_path=Path("backend/agents"),
    ... )
    >>> print(f"Generated {result.file_count} files")

    >>> # Synchronous version
    >>> result = engine.generate_from_yaml_sync(Path("pack.yaml"))

Public API:
    - AgentGeneratorEngine: Main orchestrator for code generation
    - GeneratorConfig: Configuration dataclass
    - AgentSpec: Pydantic model for pack.yaml
    - YAMLParser: YAML loading and parsing
    - SpecValidator: Specification validation
    - ValidationResult: Validation results container
    - GenerationResult: Generation results container
    - AgentGenerator: Agent class code generator
    - ModelGenerator: Pydantic model generator
    - TemplateEngine: Jinja2 template engine
    - create_generator: Factory function

Version: 1.0.0
Author: GreenLang Team
License: Proprietary
"""

from typing import List

# Version information
__version__ = "1.0.0"
__author__ = "GreenLang Team"
__license__ = "Proprietary"

# Core components - lazy imports to avoid circular dependencies


def __getattr__(name: str):
    """Lazy import of generator components."""
    # Engine components
    if name == "AgentGeneratorEngine":
        from .engine import AgentGeneratorEngine
        return AgentGeneratorEngine
    elif name == "GeneratorConfig":
        from .engine import GeneratorConfig
        return GeneratorConfig
    elif name == "GenerationResult":
        from .engine import GenerationResult
        return GenerationResult
    elif name == "create_generator":
        from .engine import create_generator
        return create_generator

    # Parser components
    elif name == "AgentSpec":
        from .parser.schema import AgentSpec
        return AgentSpec
    elif name == "AgentDef":
        from .parser.schema import AgentDef
        return AgentDef
    elif name == "PackMeta":
        from .parser.schema import PackMeta
        return PackMeta
    elif name == "YAMLParser":
        from .parser.yaml_parser import YAMLParser
        return YAMLParser
    elif name == "ParseError":
        from .parser.yaml_parser import ParseError
        return ParseError
    elif name == "SpecValidator":
        from .parser.spec_validator import SpecValidator
        return SpecValidator
    elif name == "ValidationResult":
        from .parser.spec_validator import ValidationResult
        return ValidationResult

    # Generator components
    elif name == "AgentGenerator":
        from .generators.agent_gen import AgentGenerator
        return AgentGenerator
    elif name == "ModelGenerator":
        from .generators.model_gen import ModelGenerator
        return ModelGenerator

    # Template components
    elif name == "TemplateEngine":
        from .templates import TemplateEngine
        return TemplateEngine
    elif name == "get_template_engine":
        from .templates import get_template_engine
        return get_template_engine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: List[str] = [
    # Core engine
    "AgentGeneratorEngine",
    "GeneratorConfig",
    "GenerationResult",
    "create_generator",
    # Parser
    "AgentSpec",
    "AgentDef",
    "PackMeta",
    "YAMLParser",
    "ParseError",
    "SpecValidator",
    "ValidationResult",
    # Generators
    "AgentGenerator",
    "ModelGenerator",
    # Templates
    "TemplateEngine",
    "get_template_engine",
    # Version info
    "__version__",
    "__author__",
]
