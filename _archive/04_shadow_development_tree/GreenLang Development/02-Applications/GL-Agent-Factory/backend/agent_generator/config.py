"""
Generator Configuration Module

This module defines the configuration dataclass for the Agent Generator,
providing settings for code generation, validation, and output formatting.

Example:
    >>> from backend.agent_generator.config import GeneratorConfig
    >>>
    >>> # Use defaults
    >>> config = GeneratorConfig()
    >>>
    >>> # Customize settings
    >>> config = GeneratorConfig(
    ...     output_base_dir=Path("custom/agents"),
    ...     enforce_zero_hallucination=True,
    ...     min_test_coverage=0.90,
    ... )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GeneratorConfig:
    """
    Configuration for the Agent Generator.

    This dataclass contains all configurable settings for code generation,
    validation rules, output formatting, and file naming conventions.

    Attributes:
        output_base_dir: Base directory for generated agent modules
        module_prefix: Prefix for generated module names (e.g., 'gl')
        python_version: Target Python version for generated code
        line_length: Maximum line length for code formatting
        include_type_hints: Whether to include type annotations
        include_docstrings: Whether to include docstrings
        template_dir: Directory containing Jinja2 templates
        enforce_zero_hallucination: Enforce zero-hallucination rules
        require_provenance: Require provenance tracking in agents
        require_golden_tests: Require golden test definitions
        min_test_coverage: Minimum required test coverage (0.0-1.0)
        base_calculator_import: Import path for BaseCalculator
        agent_suffix: Suffix for agent class names
        input_suffix: Suffix for input model names
        output_suffix: Suffix for output model names
        agent_filename: Filename for generated agent code
        test_filename: Filename for generated tests
        init_filename: Filename for package __init__.py
        generate_tests: Whether to generate test files
        generate_tools: Whether to generate tool wrappers
        generate_models_separate: Whether to put models in separate file
        use_black: Apply Black code formatting
        use_isort: Apply isort import sorting

    Example:
        >>> config = GeneratorConfig(
        ...     output_base_dir=Path("backend/agents"),
        ...     enforce_zero_hallucination=True,
        ... )
        >>> print(config.agent_suffix)
        'Agent'
    """

    # ==========================================================================
    # Output Settings
    # ==========================================================================

    output_base_dir: Path = field(
        default_factory=lambda: Path("backend/agents"),
        metadata={"description": "Base directory for generated agent modules"}
    )

    module_prefix: str = field(
        default="gl",
        metadata={"description": "Prefix for generated module names"}
    )

    # ==========================================================================
    # Code Generation Settings
    # ==========================================================================

    python_version: str = field(
        default="3.11",
        metadata={"description": "Target Python version"}
    )

    line_length: int = field(
        default=100,
        metadata={"description": "Maximum line length for code formatting"}
    )

    include_type_hints: bool = field(
        default=True,
        metadata={"description": "Include type annotations in generated code"}
    )

    include_docstrings: bool = field(
        default=True,
        metadata={"description": "Include docstrings in generated code"}
    )

    # ==========================================================================
    # Template Settings
    # ==========================================================================

    template_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "templates",
        metadata={"description": "Directory containing Jinja2 templates"}
    )

    # ==========================================================================
    # Validation Settings
    # ==========================================================================

    enforce_zero_hallucination: bool = field(
        default=True,
        metadata={"description": "Enforce zero-hallucination rules for calculators"}
    )

    require_provenance: bool = field(
        default=True,
        metadata={"description": "Require provenance tracking in generated agents"}
    )

    require_golden_tests: bool = field(
        default=True,
        metadata={"description": "Require golden test definitions in spec"}
    )

    min_test_coverage: float = field(
        default=0.85,
        metadata={"description": "Minimum required test coverage (0.0-1.0)"}
    )

    # ==========================================================================
    # Import Settings
    # ==========================================================================

    base_calculator_import: str = field(
        default="backend.engines.base_calculator",
        metadata={"description": "Import path for BaseCalculator class"}
    )

    provenance_mixin_import: str = field(
        default="backend.engines.base_calculator.ProvenanceMixin",
        metadata={"description": "Import path for ProvenanceMixin class"}
    )

    # ==========================================================================
    # Naming Conventions
    # ==========================================================================

    agent_suffix: str = field(
        default="Agent",
        metadata={"description": "Suffix for agent class names"}
    )

    input_suffix: str = field(
        default="Input",
        metadata={"description": "Suffix for input model names"}
    )

    output_suffix: str = field(
        default="Output",
        metadata={"description": "Suffix for output model names"}
    )

    # ==========================================================================
    # File Names
    # ==========================================================================

    agent_filename: str = field(
        default="agent.py",
        metadata={"description": "Filename for generated agent code"}
    )

    test_filename: str = field(
        default="test_agent.py",
        metadata={"description": "Filename for generated tests"}
    )

    init_filename: str = field(
        default="__init__.py",
        metadata={"description": "Filename for package __init__.py"}
    )

    models_filename: str = field(
        default="models.py",
        metadata={"description": "Filename for separate models file"}
    )

    tools_filename: str = field(
        default="tools.py",
        metadata={"description": "Filename for tool wrappers"}
    )

    # ==========================================================================
    # Generation Options
    # ==========================================================================

    generate_tests: bool = field(
        default=True,
        metadata={"description": "Whether to generate test files"}
    )

    generate_tools: bool = field(
        default=True,
        metadata={"description": "Whether to generate tool wrappers"}
    )

    generate_models_separate: bool = field(
        default=False,
        metadata={"description": "Put models in separate file (models.py)"}
    )

    generate_cli: bool = field(
        default=False,
        metadata={"description": "Generate CLI entry point"}
    )

    # ==========================================================================
    # Formatting Options
    # ==========================================================================

    use_black: bool = field(
        default=True,
        metadata={"description": "Apply Black code formatting"}
    )

    use_isort: bool = field(
        default=True,
        metadata={"description": "Apply isort import sorting"}
    )

    # ==========================================================================
    # Zero-Hallucination Settings
    # ==========================================================================

    prohibited_imports: List[str] = field(
        default_factory=lambda: [
            "openai",
            "anthropic",
            "langchain",
            "llama_index",
            "transformers",
        ],
        metadata={"description": "Imports prohibited in CALCULATION path (LLM allowed for explanations)"}
    )

    allowed_ml_agent_types: List[str] = field(
        default_factory=lambda: [
            "ml-classifier",
            "report-generator",
        ],
        metadata={"description": "Agent types allowed to use ML/LLM in calculation path"}
    )

    # ==========================================================================
    # INTELLIGENCE FRAMEWORK SETTINGS (Solves Intelligence Paradox - Dec 2025)
    # ==========================================================================

    require_intelligence: bool = field(
        default=True,
        metadata={
            "description": "MANDATORY: Require all new agents to implement IntelligentAgentBase"
        }
    )

    default_intelligence_level: str = field(
        default="STANDARD",
        metadata={
            "description": "Default intelligence level for generated agents",
            "allowed_values": ["BASIC", "STANDARD", "ADVANCED", "FULL"]
        }
    )

    intelligence_level_none_allowed: bool = field(
        default=False,
        metadata={
            "description": "Allow NONE intelligence level (DEPRECATED - only for legacy)"
        }
    )

    require_intelligence_decorator: bool = field(
        default=True,
        metadata={
            "description": "Require @require_intelligence decorator on all agent classes"
        }
    )

    require_explanation_generation: bool = field(
        default=True,
        metadata={
            "description": "Require agents to call generate_explanation() in execute()"
        }
    )

    require_recommendation_generation: bool = field(
        default=True,
        metadata={
            "description": "Require agents to call generate_recommendations() in execute()"
        }
    )

    intelligence_base_class: str = field(
        default="IntelligentAgentBase",
        metadata={
            "description": "Base class for intelligent agents"
        }
    )

    intelligence_mixin_class: str = field(
        default="IntelligenceMixin",
        metadata={
            "description": "Mixin class for retrofitting existing agents"
        }
    )

    intelligence_imports: List[str] = field(
        default_factory=lambda: [
            "from greenlang.agents import IntelligentAgentBase, IntelligentAgentConfig",
            "from greenlang.agents import IntelligenceLevel, IntelligenceCapabilities",
            "from greenlang.agents import require_intelligence",
            "from greenlang.agents import Recommendation, Anomaly",
        ],
        metadata={
            "description": "Import statements for intelligence framework"
        }
    )

    default_regulatory_context: str = field(
        default="GHG Protocol, CSRD ESRS E1",
        metadata={
            "description": "Default regulatory context for generated agents"
        }
    )

    # ==========================================================================
    # Provenance Settings
    # ==========================================================================

    provenance_hash_algorithm: str = field(
        default="sha256",
        metadata={"description": "Hash algorithm for provenance tracking"}
    )

    provenance_include_timestamp: bool = field(
        default=True,
        metadata={"description": "Include timestamp in provenance data"}
    )

    # ==========================================================================
    # Methods
    # ==========================================================================

    def get_module_name(self, agent_id: str, sequence_number: int) -> str:
        """
        Generate module name from agent ID and sequence number.

        Args:
            agent_id: Agent identifier (e.g., 'eudr-validator')
            sequence_number: Numeric sequence (e.g., 14)

        Returns:
            Module name (e.g., 'gl_014_eudr_validator')

        Example:
            >>> config = GeneratorConfig()
            >>> config.get_module_name('eudr-validator', 14)
            'gl_014_eudr_validator'
        """
        # Convert agent ID to snake_case
        safe_id = agent_id.replace("-", "_").replace(" ", "_").lower()
        return f"{self.module_prefix}_{sequence_number:03d}_{safe_id}"

    def get_agent_class_name(self, agent_name: str) -> str:
        """
        Generate agent class name from agent name.

        Args:
            agent_name: Human-readable agent name

        Returns:
            PascalCase class name with Agent suffix

        Example:
            >>> config = GeneratorConfig()
            >>> config.get_agent_class_name('EUDR Compliance Validator')
            'EUDRComplianceValidatorAgent'
        """
        # Remove special characters and convert to PascalCase
        words = agent_name.replace("-", " ").replace("_", " ").split()
        pascal = "".join(word.capitalize() for word in words)

        # Ensure suffix is not duplicated
        if not pascal.endswith(self.agent_suffix):
            pascal += self.agent_suffix

        return pascal

    def get_model_class_name(self, agent_name: str, model_type: str) -> str:
        """
        Generate model class name from agent name and type.

        Args:
            agent_name: Human-readable agent name
            model_type: 'input' or 'output'

        Returns:
            PascalCase class name with appropriate suffix

        Example:
            >>> config = GeneratorConfig()
            >>> config.get_model_class_name('EUDR Compliance', 'input')
            'EUDRComplianceInput'
        """
        words = agent_name.replace("-", " ").replace("_", " ").split()
        # Remove 'Agent' if present
        words = [w for w in words if w.lower() != "agent"]
        pascal = "".join(word.capitalize() for word in words)

        suffix = self.input_suffix if model_type == "input" else self.output_suffix
        return pascal + suffix

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid)

        Example:
            >>> config = GeneratorConfig(min_test_coverage=1.5)
            >>> errors = config.validate()
            >>> print(errors)
            ['min_test_coverage must be between 0.0 and 1.0']
        """
        errors = []

        # Validate test coverage range
        if not 0.0 <= self.min_test_coverage <= 1.0:
            errors.append("min_test_coverage must be between 0.0 and 1.0")

        # Validate line length
        if self.line_length < 40 or self.line_length > 200:
            errors.append("line_length must be between 40 and 200")

        # Validate template directory exists
        if not self.template_dir.exists():
            errors.append(f"template_dir does not exist: {self.template_dir}")

        # Validate Python version format
        parts = self.python_version.split(".")
        if len(parts) < 2:
            errors.append("python_version must be in format 'X.Y' or 'X.Y.Z'")

        return errors

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "output_base_dir": str(self.output_base_dir),
            "module_prefix": self.module_prefix,
            "python_version": self.python_version,
            "line_length": self.line_length,
            "include_type_hints": self.include_type_hints,
            "include_docstrings": self.include_docstrings,
            "template_dir": str(self.template_dir),
            "enforce_zero_hallucination": self.enforce_zero_hallucination,
            "require_provenance": self.require_provenance,
            "require_golden_tests": self.require_golden_tests,
            "min_test_coverage": self.min_test_coverage,
            "generate_tests": self.generate_tests,
            "generate_tools": self.generate_tools,
            "use_black": self.use_black,
            "use_isort": self.use_isort,
        }


# Default configuration instance
DEFAULT_CONFIG = GeneratorConfig()
