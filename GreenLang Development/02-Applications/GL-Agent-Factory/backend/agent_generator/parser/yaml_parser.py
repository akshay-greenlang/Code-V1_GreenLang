"""
YAML Parser Module for Agent Generator

This module provides YAML loading and parsing capabilities for pack.yaml
AgentSpec files. It handles:
- YAML file loading with error handling
- External file reference resolution ($ref)
- Environment variable interpolation
- Schema validation against Pydantic models

Example:
    >>> from backend.agent_generator.parser.yaml_parser import YAMLParser
    >>> from pathlib import Path
    >>>
    >>> parser = YAMLParser()
    >>> spec = parser.parse(Path("08-regulatory-agents/eudr/pack.yaml"))
    >>> print(spec.pack.name)
    'EUDR Deforestation Compliance Agent'
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError as PydanticValidationError

from .schema import (
    AgentSpec,
    AgentDef,
    AgentType,
    InputDef,
    OutputDef,
    PackMeta,
    ToolDef,
    ToolType,
    ToolConfig,
    ValidationRule,
    GoldenTestSpec,
    GoldenTestCategory,
)

# Aliases for backward compatibility
AgentDefinition = AgentDef
InputDefinition = InputDef
OutputDefinition = OutputDef
ToolDefinition = ToolDef
PackSpec = PackMeta
SchemaDefinition = dict  # Placeholder for JSON Schema dicts

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """
    Exception raised when YAML parsing fails.

    Provides detailed error information including line and column numbers
    for easier debugging.

    Attributes:
        message: Error description
        line: Line number where error occurred (1-indexed)
        column: Column number where error occurred (1-indexed)
        file_path: Path to the file being parsed
    """

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        file_path: Optional[Path] = None,
    ):
        self.message = message
        self.line = line
        self.column = column
        self.file_path = file_path

        # Build detailed error message
        location = []
        if file_path:
            location.append(f"file={file_path}")
        if line is not None:
            location.append(f"line={line}")
        if column is not None:
            location.append(f"column={column}")

        full_message = message
        if location:
            full_message = f"{message} ({', '.join(location)})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.message,
            "line": self.line,
            "column": self.column,
            "file_path": str(self.file_path) if self.file_path else None,
        }


class YAMLParser:
    """
    Parses AgentSpec YAML files into Pydantic models.

    This parser handles:
    - Loading YAML files with proper error handling
    - Resolving external file references ($ref pointers)
    - Interpolating environment variables (${VAR} syntax)
    - Converting YAML data to Pydantic AgentSpec models

    Attributes:
        base_path: Base directory for resolving relative paths

    Example:
        >>> parser = YAMLParser()
        >>> spec = parser.parse(Path("pack.yaml"))
        >>> for agent in spec.agents:
        ...     print(f"Agent: {agent.name}")

        >>> # Parse from string
        >>> yaml_content = '''
        ... pack:
        ...   id: test
        ...   name: Test Agent
        ...   version: "1.0.0"
        ... agents: []
        ... '''
        >>> spec = parser.parse_string(yaml_content)
    """

    # Regex for environment variable interpolation
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    # Regex for $ref pointers
    REF_PATTERN = re.compile(r'^\$ref:\s*(.+)$')

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the YAML parser.

        Args:
            base_path: Base directory for resolving relative file paths.
                      If not provided, uses the current working directory.
        """
        self.base_path = base_path or Path.cwd()

    def parse(self, yaml_path: Path) -> AgentSpec:
        """
        Parse a pack.yaml file into an AgentSpec model.

        Args:
            yaml_path: Path to the pack.yaml file

        Returns:
            Validated AgentSpec model

        Raises:
            ParseError: If YAML syntax is invalid or file not found
            ValidationError: If spec doesn't match expected schema

        Example:
            >>> parser = YAMLParser()
            >>> spec = parser.parse(Path("08-regulatory-agents/eudr/pack.yaml"))
            >>> print(spec.pack.id)
            'gl-eudr-compliance-v1'
        """
        yaml_path = Path(yaml_path)

        # Validate file exists
        if not yaml_path.exists():
            raise ParseError(
                f"File not found: {yaml_path}",
                file_path=yaml_path,
            )

        if not yaml_path.is_file():
            raise ParseError(
                f"Path is not a file: {yaml_path}",
                file_path=yaml_path,
            )

        # Update base path to file's directory
        self.base_path = yaml_path.parent

        # Load YAML content
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except IOError as e:
            raise ParseError(
                f"Failed to read file: {e}",
                file_path=yaml_path,
            )

        logger.debug(f"Parsing YAML file: {yaml_path}")

        # Parse YAML
        try:
            data = yaml.safe_load(raw_content)
        except yaml.YAMLError as e:
            # Extract line/column from YAML error
            line = None
            column = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1  # Convert to 1-indexed
                column = e.problem_mark.column + 1

            raise ParseError(
                f"Invalid YAML syntax: {str(e)}",
                line=line,
                column=column,
                file_path=yaml_path,
            )

        if data is None:
            raise ParseError(
                "Empty YAML file",
                file_path=yaml_path,
            )

        # Process the data
        data = self._resolve_references(data, yaml_path.parent)
        data = self._interpolate_env_vars(data)

        # Convert to AgentSpec model
        return self._to_agent_spec(data, yaml_path)

    def parse_string(self, yaml_content: str, source_name: str = "<string>") -> AgentSpec:
        """
        Parse YAML content from a string.

        Args:
            yaml_content: YAML content as string
            source_name: Name to use in error messages

        Returns:
            Validated AgentSpec model

        Raises:
            ParseError: If YAML syntax is invalid
            ValidationError: If spec doesn't match expected schema

        Example:
            >>> parser = YAMLParser()
            >>> spec = parser.parse_string('''
            ... pack:
            ...   id: test
            ...   name: Test
            ...   version: "1.0.0"
            ...   description: Test agent
            ... agents: []
            ... ''')
        """
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            line = None
            column = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
                column = e.problem_mark.column + 1

            raise ParseError(
                f"Invalid YAML syntax: {str(e)}",
                line=line,
                column=column,
            )

        if data is None:
            raise ParseError("Empty YAML content")

        # Process the data
        data = self._interpolate_env_vars(data)

        # Convert to AgentSpec model
        return self._to_agent_spec(data)

    def parse_dict(self, data: Dict[str, Any]) -> AgentSpec:
        """
        Parse a dictionary directly into an AgentSpec model.

        Useful for programmatic spec creation or testing.

        Args:
            data: Dictionary matching pack.yaml structure

        Returns:
            Validated AgentSpec model

        Example:
            >>> parser = YAMLParser()
            >>> spec = parser.parse_dict({
            ...     "pack": {"id": "test", "name": "Test", "version": "1.0.0", "description": ""},
            ...     "agents": [],
            ... })
        """
        return self._to_agent_spec(data)

    def _resolve_references(
        self,
        data: Any,
        base_dir: Path,
        visited: Optional[set] = None,
    ) -> Any:
        """
        Resolve $ref pointers to external files.

        Handles:
        - File references: "$ref: ./schemas/input.yaml"
        - Nested file references
        - Circular reference detection

        Args:
            data: Data to process (dict, list, or scalar)
            base_dir: Base directory for resolving relative paths
            visited: Set of visited file paths (for cycle detection)

        Returns:
            Data with $ref pointers resolved
        """
        if visited is None:
            visited = set()

        if isinstance(data, dict):
            # Check for $ref at this level
            if "$ref" in data and len(data) == 1:
                ref_path = data["$ref"]
                return self._load_reference(ref_path, base_dir, visited)

            # Recursively process dict values
            return {
                key: self._resolve_references(value, base_dir, visited)
                for key, value in data.items()
            }

        elif isinstance(data, list):
            # Recursively process list items
            return [
                self._resolve_references(item, base_dir, visited)
                for item in data
            ]

        elif isinstance(data, str):
            # Check for inline $ref syntax
            match = self.REF_PATTERN.match(data)
            if match:
                ref_path = match.group(1).strip()
                return self._load_reference(ref_path, base_dir, visited)

            return data

        else:
            # Return scalar values unchanged
            return data

    def _load_reference(
        self,
        ref_path: str,
        base_dir: Path,
        visited: set,
    ) -> Any:
        """
        Load a referenced YAML file.

        Args:
            ref_path: Path to referenced file (relative or absolute)
            base_dir: Base directory for relative paths
            visited: Set of visited file paths

        Returns:
            Parsed content of referenced file

        Raises:
            ParseError: If reference is circular or file not found
        """
        # Resolve relative path
        if not Path(ref_path).is_absolute():
            full_path = (base_dir / ref_path).resolve()
        else:
            full_path = Path(ref_path).resolve()

        # Check for circular references
        if str(full_path) in visited:
            raise ParseError(
                f"Circular reference detected: {ref_path}",
                file_path=full_path,
            )

        # Validate file exists
        if not full_path.exists():
            raise ParseError(
                f"Referenced file not found: {ref_path}",
                file_path=full_path,
            )

        # Mark as visited
        visited.add(str(full_path))

        # Load and parse referenced file
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                ref_data = yaml.safe_load(f.read())
        except yaml.YAMLError as e:
            raise ParseError(
                f"Invalid YAML in referenced file: {e}",
                file_path=full_path,
            )

        # Recursively resolve references in the loaded data
        return self._resolve_references(ref_data, full_path.parent, visited)

    def _interpolate_env_vars(self, data: Any) -> Any:
        """
        Interpolate environment variables in string values.

        Replaces ${VAR} syntax with environment variable values.
        Supports default values: ${VAR:-default}

        Args:
            data: Data to process

        Returns:
            Data with environment variables interpolated

        Example:
            >>> os.environ["API_KEY"] = "secret123"
            >>> parser._interpolate_env_vars("${API_KEY}")
            'secret123'
            >>> parser._interpolate_env_vars("${MISSING:-default}")
            'default'
        """
        if isinstance(data, dict):
            return {
                key: self._interpolate_env_vars(value)
                for key, value in data.items()
            }

        elif isinstance(data, list):
            return [self._interpolate_env_vars(item) for item in data]

        elif isinstance(data, str):
            return self._replace_env_vars(data)

        else:
            return data

    def _replace_env_vars(self, value: str) -> str:
        """
        Replace environment variables in a string.

        Args:
            value: String potentially containing ${VAR} patterns

        Returns:
            String with environment variables replaced
        """
        def replacer(match):
            var_expr = match.group(1)

            # Check for default value syntax: ${VAR:-default}
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default)
            else:
                var_name = var_expr.strip()
                result = os.environ.get(var_name)
                if result is None:
                    logger.warning(f"Environment variable not found: {var_name}")
                    return match.group(0)  # Keep original if not found
                return result

        return self.ENV_VAR_PATTERN.sub(replacer, value)

    def _to_agent_spec(
        self,
        data: Dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> AgentSpec:
        """
        Convert parsed dictionary to AgentSpec model.

        Args:
            data: Parsed YAML data
            source_path: Original file path for error messages

        Returns:
            Validated AgentSpec model

        Raises:
            ParseError: If data doesn't match expected schema
        """
        try:
            return AgentSpec(**data)
        except PydanticValidationError as e:
            # Extract first error for simpler message
            errors = e.errors()
            if errors:
                first_error = errors[0]
                field_path = " -> ".join(str(p) for p in first_error.get("loc", []))
                message = first_error.get("msg", "Validation failed")
                raise ParseError(
                    f"Schema validation failed for '{field_path}': {message}",
                    file_path=source_path,
                )
            else:
                raise ParseError(
                    f"Schema validation failed: {str(e)}",
                    file_path=source_path,
                )

    def get_raw_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        """
        Load YAML file without converting to AgentSpec.

        Useful for debugging or custom processing.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Raw parsed YAML data as dictionary
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f.read())

        data = self._resolve_references(data, yaml_path.parent)
        data = self._interpolate_env_vars(data)

        return data

    def validate_syntax(self, yaml_path: Path) -> List[str]:
        """
        Validate YAML syntax without full schema validation.

        Useful for quick syntax checks.

        Args:
            yaml_path: Path to YAML file

        Returns:
            List of syntax errors (empty if valid)
        """
        errors = []

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f.read())
        except yaml.YAMLError as e:
            errors.append(str(e))
        except IOError as e:
            errors.append(f"Cannot read file: {e}")

        return errors
