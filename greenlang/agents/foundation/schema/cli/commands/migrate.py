# -*- coding: utf-8 -*-
"""
Migrate Command for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the 'greenlang schema migrate' command group for
migrating existing validation code to GreenLang schemas.

Features:
    - Analyze codebase for existing validation patterns
    - Generate detailed migration reports
    - Convert JSON Schema to GreenLang schema format
    - Estimate migration effort

Commands:
    - analyze: Scan codebase for validation patterns
    - report: Generate detailed migration report
    - convert: Convert JSON Schema to GreenLang format

Exit Codes:
    0 - Success (analysis/conversion completed)
    1 - Warnings found (patterns detected that may need attention)
    2 - Error (system error, missing file, etc.)

Example:
    $ greenlang schema migrate analyze --path ./src
    $ greenlang schema migrate report --path ./src --output migration.json
    $ greenlang schema migrate convert --input schema.json --output schema.yaml

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 6.4
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import click
import yaml

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_WARNINGS = 1
EXIT_ERROR = 2

# File extensions to scan
PYTHON_EXTENSIONS = {".py"}
JSON_EXTENSIONS = {".json"}
YAML_EXTENSIONS = {".yaml", ".yml"}
ALL_SCAN_EXTENSIONS = PYTHON_EXTENSIONS | JSON_EXTENSIONS | YAML_EXTENSIONS


# =============================================================================
# Data Models
# =============================================================================


class ValidationPattern(BaseModel):
    """
    Detected validation pattern in code.

    Represents a single instance of validation logic found during
    codebase analysis.

    Attributes:
        file: Path to the file containing the pattern.
        line: Line number where the pattern was found.
        pattern_type: Type of validation pattern detected.
        code_snippet: The actual code snippet.
        schema_reference: Optional reference to an external schema.
        migration_difficulty: Estimated difficulty to migrate.
        suggestion: Suggestion for migration.

    Example:
        >>> pattern = ValidationPattern(
        ...     file="src/validator.py",
        ...     line=42,
        ...     pattern_type="jsonschema",
        ...     code_snippet="jsonschema.validate(data, schema)",
        ...     migration_difficulty="easy",
        ...     suggestion="Replace with greenlang.schema.validate()"
        ... )
    """

    file: str = Field(
        ...,
        description="Path to the file containing the pattern"
    )

    line: int = Field(
        ...,
        ge=1,
        description="Line number where the pattern was found"
    )

    pattern_type: str = Field(
        ...,
        description="Type of validation pattern (jsonschema, pydantic, manual, custom)"
    )

    code_snippet: str = Field(
        ...,
        max_length=1024,
        description="The actual code snippet found"
    )

    schema_reference: Optional[str] = Field(
        default=None,
        description="Optional reference to an external schema"
    )

    migration_difficulty: str = Field(
        default="easy",
        description="Estimated migration difficulty (easy, medium, hard)"
    )

    suggestion: str = Field(
        default="",
        description="Suggestion for migration"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        """Validate pattern_type is a known type."""
        valid_types = {"jsonschema", "pydantic", "manual", "custom"}
        if v not in valid_types:
            raise ValueError(
                f"Invalid pattern_type '{v}'. Must be one of: {valid_types}"
            )
        return v

    @field_validator("migration_difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        """Validate migration_difficulty is a known level."""
        valid_levels = {"easy", "medium", "hard"}
        if v not in valid_levels:
            raise ValueError(
                f"Invalid migration_difficulty '{v}'. Must be one of: {valid_levels}"
            )
        return v


class MigrationReport(BaseModel):
    """
    Migration analysis report.

    Contains the complete results of analyzing a codebase for
    validation patterns and migration recommendations.

    Attributes:
        analyzed_files: Number of files analyzed.
        validation_patterns_found: Total count of patterns detected.
        patterns: List of detected validation patterns.
        json_schemas_found: Paths to JSON Schema files found.
        estimated_effort: Overall migration effort estimate.
        recommendations: List of migration recommendations.

    Example:
        >>> report = MigrationReport(
        ...     analyzed_files=50,
        ...     validation_patterns_found=12,
        ...     patterns=[...],
        ...     json_schemas_found=["schemas/input.json"],
        ...     estimated_effort="medium",
        ...     recommendations=["Start with Pydantic models..."]
        ... )
    """

    analyzed_files: int = Field(
        ...,
        ge=0,
        description="Number of files analyzed"
    )

    validation_patterns_found: int = Field(
        ...,
        ge=0,
        description="Total count of patterns detected"
    )

    patterns: List[ValidationPattern] = Field(
        default_factory=list,
        description="List of detected validation patterns"
    )

    json_schemas_found: List[str] = Field(
        default_factory=list,
        description="Paths to JSON Schema files found"
    )

    estimated_effort: str = Field(
        default="low",
        description="Overall migration effort estimate (low, medium, high)"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="List of migration recommendations"
    )

    analysis_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to perform analysis"
    )

    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when report was generated"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator("estimated_effort")
    @classmethod
    def validate_effort(cls, v: str) -> str:
        """Validate estimated_effort is a known level."""
        valid_levels = {"low", "medium", "high"}
        if v not in valid_levels:
            raise ValueError(
                f"Invalid estimated_effort '{v}'. Must be one of: {valid_levels}"
            )
        return v


class SchemaConversion(BaseModel):
    """
    Schema conversion result.

    Contains the result of converting a JSON Schema to GreenLang format.

    Attributes:
        original_format: The original schema format (json-schema).
        original_dialect: The JSON Schema dialect version.
        converted_schema: The converted GreenLang schema.
        warnings: Warnings generated during conversion.
        gl_extensions_added: GreenLang extensions added to the schema.

    Example:
        >>> conversion = SchemaConversion(
        ...     original_format="json-schema",
        ...     original_dialect="draft-07",
        ...     converted_schema={...},
        ...     warnings=["No unit specifications detected"],
        ...     gl_extensions_added={"$glVersion": "1.0"}
        ... )
    """

    original_format: str = Field(
        ...,
        description="The original schema format"
    )

    original_dialect: Optional[str] = Field(
        default=None,
        description="The JSON Schema dialect version if applicable"
    )

    converted_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="The converted GreenLang schema"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during conversion"
    )

    gl_extensions_added: Dict[str, Any] = Field(
        default_factory=dict,
        description="GreenLang extensions added to the schema"
    )

    conversion_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to perform conversion"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }


# =============================================================================
# Migration Analyzer
# =============================================================================


class MigrationAnalyzer:
    """
    Analyzes codebase for validation patterns.

    Scans Python files for common validation patterns including:
    - jsonschema library usage
    - Pydantic model definitions
    - Manual validation code (isinstance, assertions)
    - Custom validator implementations

    Attributes:
        path: Path to the directory to analyze.

    Example:
        >>> analyzer = MigrationAnalyzer(Path("./src"))
        >>> report = analyzer.analyze()
        >>> print(f"Found {report.validation_patterns_found} patterns")
    """

    # Patterns to detect jsonschema usage
    JSONSCHEMA_PATTERNS = [
        (r"jsonschema\.validate\s*\(", "jsonschema.validate() call"),
        (r"from\s+jsonschema\s+import", "jsonschema import"),
        (r"import\s+jsonschema", "jsonschema module import"),
        (r"Draft\d+Validator", "JSON Schema Draft Validator"),
        (r"\.validate\s*\(\s*\w+\s*,\s*\w+\s*\)", "Generic validate call"),
    ]

    # Patterns to detect Pydantic usage
    PYDANTIC_PATTERNS = [
        (r"class\s+\w+\s*\(\s*BaseModel\s*\)", "Pydantic BaseModel class"),
        (r"from\s+pydantic\s+import", "Pydantic import"),
        (r"@validator\s*\(", "Pydantic @validator decorator"),
        (r"@field_validator\s*\(", "Pydantic @field_validator decorator"),
        (r"@root_validator", "Pydantic @root_validator decorator"),
        (r"@model_validator", "Pydantic @model_validator decorator"),
        (r"Field\s*\(", "Pydantic Field() usage"),
    ]

    # Patterns to detect manual validation
    MANUAL_PATTERNS = [
        (r"if\s+\w+\s+not\s+in\s+", "Manual membership check"),
        (r"raise\s+ValueError\s*\(", "ValueError raise"),
        (r"raise\s+TypeError\s*\(", "TypeError raise"),
        (r"raise\s+ValidationError\s*\(", "ValidationError raise"),
        (r"assert\s+isinstance\s*\(", "isinstance assertion"),
        (r"if\s+type\s*\(\s*\w+\s*\)\s*[!=]=", "Type comparison"),
        (r"if\s+not\s+isinstance\s*\(", "isinstance check"),
    ]

    # Patterns for custom validators
    CUSTOM_PATTERNS = [
        (r"def\s+validate_\w+\s*\(", "Custom validate_ function"),
        (r"def\s+check_\w+\s*\(", "Custom check_ function"),
        (r"class\s+\w*Validator\s*[:\(]", "Custom Validator class"),
        (r"def\s+is_valid\s*\(", "is_valid method"),
    ]

    def __init__(self, path: Path):
        """
        Initialize MigrationAnalyzer.

        Args:
            path: Path to the directory to analyze.
        """
        self.path = path
        self._patterns: List[ValidationPattern] = []
        self._json_schemas: List[str] = []
        self._files_analyzed: int = 0

    def analyze(self) -> MigrationReport:
        """
        Run full analysis on the codebase.

        Scans all Python files in the path for validation patterns
        and generates a comprehensive migration report.

        Returns:
            MigrationReport with all findings and recommendations.

        Example:
            >>> analyzer = MigrationAnalyzer(Path("./src"))
            >>> report = analyzer.analyze()
        """
        start_time = time.perf_counter()

        # Clear previous results
        self._patterns = []
        self._json_schemas = []
        self._files_analyzed = 0

        # Scan all files
        if self.path.is_file():
            self._scan_file(self.path)
        else:
            self._scan_directory(self.path)

        # Calculate analysis time
        analysis_time_ms = (time.perf_counter() - start_time) * 1000

        # Generate recommendations
        recommendations = self._generate_recommendations(self._patterns)

        # Estimate effort
        estimated_effort = self._estimate_effort(self._patterns)

        return MigrationReport(
            analyzed_files=self._files_analyzed,
            validation_patterns_found=len(self._patterns),
            patterns=self._patterns,
            json_schemas_found=self._json_schemas,
            estimated_effort=estimated_effort,
            recommendations=recommendations,
            analysis_time_ms=analysis_time_ms,
        )

    def _scan_directory(self, directory: Path) -> None:
        """
        Recursively scan a directory for files to analyze.

        Args:
            directory: Directory path to scan.
        """
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Skip common non-source directories
                    if item.name in {
                        "__pycache__",
                        ".git",
                        ".venv",
                        "venv",
                        "node_modules",
                        ".tox",
                        ".mypy_cache",
                        ".pytest_cache",
                        "dist",
                        "build",
                        "egg-info",
                    }:
                        continue
                    self._scan_directory(item)
                elif item.is_file():
                    self._scan_file(item)
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {directory}")

    def _scan_file(self, file_path: Path) -> None:
        """
        Scan a single file for validation patterns.

        Args:
            file_path: Path to the file to scan.
        """
        suffix = file_path.suffix.lower()

        if suffix in PYTHON_EXTENSIONS:
            self._scan_python_file(file_path)
        elif suffix in JSON_EXTENSIONS:
            self._check_json_schema(file_path)
        elif suffix in YAML_EXTENSIONS:
            self._check_yaml_schema(file_path)

    def _scan_python_file(self, file_path: Path) -> None:
        """
        Scan a Python file for validation patterns.

        Args:
            file_path: Path to the Python file.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            self._files_analyzed += 1

            # Detect jsonschema patterns
            patterns = self._detect_jsonschema(content, str(file_path))
            self._patterns.extend(patterns)

            # Detect Pydantic patterns
            patterns = self._detect_pydantic(content, str(file_path))
            self._patterns.extend(patterns)

            # Detect manual validation patterns
            patterns = self._detect_manual(content, str(file_path))
            self._patterns.extend(patterns)

            # Detect custom validator patterns
            patterns = self._detect_custom(content, str(file_path))
            self._patterns.extend(patterns)

        except UnicodeDecodeError:
            logger.warning(f"Could not decode file: {file_path}")
        except IOError as e:
            logger.warning(f"Could not read file {file_path}: {e}")

    def _detect_jsonschema(
        self,
        content: str,
        file_path: str
    ) -> List[ValidationPattern]:
        """
        Detect jsonschema usage in file content.

        Args:
            content: File content to analyze.
            file_path: Path to the file (for reporting).

        Returns:
            List of detected ValidationPattern instances.
        """
        patterns = []
        lines = content.split("\n")

        for pattern, description in self.JSONSCHEMA_PATTERNS:
            regex = re.compile(pattern)
            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    patterns.append(ValidationPattern(
                        file=file_path,
                        line=line_num,
                        pattern_type="jsonschema",
                        code_snippet=line.strip()[:200],
                        migration_difficulty="easy",
                        suggestion=(
                            f"Replace {description} with "
                            "greenlang.schema.validate(). "
                            "GreenLang is compatible with JSON Schema Draft 2020-12."
                        ),
                    ))

        return patterns

    def _detect_pydantic(
        self,
        content: str,
        file_path: str
    ) -> List[ValidationPattern]:
        """
        Detect Pydantic model usage in file content.

        Args:
            content: File content to analyze.
            file_path: Path to the file (for reporting).

        Returns:
            List of detected ValidationPattern instances.
        """
        patterns = []
        lines = content.split("\n")

        for pattern, description in self.PYDANTIC_PATTERNS:
            regex = re.compile(pattern)
            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    # Pydantic models can be converted or kept as-is
                    difficulty = "easy" if "import" in description else "medium"
                    patterns.append(ValidationPattern(
                        file=file_path,
                        line=line_num,
                        pattern_type="pydantic",
                        code_snippet=line.strip()[:200],
                        migration_difficulty=difficulty,
                        suggestion=(
                            f"Found {description}. Pydantic models can be "
                            "exported to JSON Schema using model_json_schema(), "
                            "then enhanced with GreenLang extensions."
                        ),
                    ))

        return patterns

    def _detect_manual(
        self,
        content: str,
        file_path: str
    ) -> List[ValidationPattern]:
        """
        Detect manual validation code in file content.

        Args:
            content: File content to analyze.
            file_path: Path to the file (for reporting).

        Returns:
            List of detected ValidationPattern instances.
        """
        patterns = []
        lines = content.split("\n")

        for pattern, description in self.MANUAL_PATTERNS:
            regex = re.compile(pattern)
            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    patterns.append(ValidationPattern(
                        file=file_path,
                        line=line_num,
                        pattern_type="manual",
                        code_snippet=line.strip()[:200],
                        migration_difficulty="medium",
                        suggestion=(
                            f"Found {description}. Consider converting to "
                            "schema constraints (enum, type, min/max) or "
                            "cross-field rules in GreenLang schema."
                        ),
                    ))

        return patterns

    def _detect_custom(
        self,
        content: str,
        file_path: str
    ) -> List[ValidationPattern]:
        """
        Detect custom validator implementations in file content.

        Args:
            content: File content to analyze.
            file_path: Path to the file (for reporting).

        Returns:
            List of detected ValidationPattern instances.
        """
        patterns = []
        lines = content.split("\n")

        for pattern, description in self.CUSTOM_PATTERNS:
            regex = re.compile(pattern)
            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    patterns.append(ValidationPattern(
                        file=file_path,
                        line=line_num,
                        pattern_type="custom",
                        code_snippet=line.strip()[:200],
                        migration_difficulty="hard",
                        suggestion=(
                            f"Found {description}. Custom validation logic "
                            "may need to be expressed as GreenLang cross-field "
                            "rules or remain as code-based post-validation."
                        ),
                    ))

        return patterns

    def _check_json_schema(self, file_path: Path) -> None:
        """
        Check if a JSON file is a JSON Schema.

        Args:
            file_path: Path to the JSON file.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            self._files_analyzed += 1

            data = json.loads(content)

            # Check if it looks like a JSON Schema
            if isinstance(data, dict):
                schema_indicators = {
                    "$schema",
                    "type",
                    "properties",
                    "required",
                    "items",
                    "definitions",
                    "$defs",
                    "allOf",
                    "anyOf",
                    "oneOf",
                }
                if schema_indicators & set(data.keys()):
                    self._json_schemas.append(str(file_path))
                    logger.debug(f"Found potential JSON Schema: {file_path}")

        except (json.JSONDecodeError, IOError):
            pass  # Not a valid JSON file, skip

    def _check_yaml_schema(self, file_path: Path) -> None:
        """
        Check if a YAML file is a JSON Schema.

        Args:
            file_path: Path to the YAML file.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            self._files_analyzed += 1

            data = yaml.safe_load(content)

            # Check if it looks like a JSON Schema
            if isinstance(data, dict):
                schema_indicators = {
                    "$schema",
                    "type",
                    "properties",
                    "required",
                    "items",
                    "definitions",
                    "$defs",
                    "allOf",
                    "anyOf",
                    "oneOf",
                }
                if schema_indicators & set(data.keys()):
                    self._json_schemas.append(str(file_path))
                    logger.debug(f"Found potential YAML Schema: {file_path}")

        except (yaml.YAMLError, IOError):
            pass  # Not a valid YAML file, skip

    def _estimate_effort(self, patterns: List[ValidationPattern]) -> str:
        """
        Estimate overall migration effort based on patterns found.

        Args:
            patterns: List of detected validation patterns.

        Returns:
            Effort estimate: "low", "medium", or "high".
        """
        if not patterns:
            return "low"

        # Count by difficulty
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
        for pattern in patterns:
            difficulty_counts[pattern.migration_difficulty] += 1

        total = len(patterns)
        hard_ratio = difficulty_counts["hard"] / total
        medium_ratio = difficulty_counts["medium"] / total

        # Determine overall effort
        if total > 50 or hard_ratio > 0.3:
            return "high"
        elif total > 20 or hard_ratio > 0.1 or medium_ratio > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        patterns: List[ValidationPattern]
    ) -> List[str]:
        """
        Generate migration recommendations based on patterns found.

        Args:
            patterns: List of detected validation patterns.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Count pattern types
        type_counts: Dict[str, int] = {}
        for pattern in patterns:
            type_counts[pattern.pattern_type] = (
                type_counts.get(pattern.pattern_type, 0) + 1
            )

        # Generate type-specific recommendations
        if type_counts.get("jsonschema", 0) > 0:
            recommendations.append(
                f"Found {type_counts['jsonschema']} jsonschema usages. "
                "These can be directly migrated to GreenLang as it supports "
                "JSON Schema Draft 2020-12. Use 'greenlang schema migrate convert' "
                "to upgrade existing schemas."
            )

        if type_counts.get("pydantic", 0) > 0:
            recommendations.append(
                f"Found {type_counts['pydantic']} Pydantic patterns. "
                "Export Pydantic models to JSON Schema using model_json_schema(), "
                "then add GreenLang extensions ($unit, $rules, $aliases) as needed."
            )

        if type_counts.get("manual", 0) > 0:
            recommendations.append(
                f"Found {type_counts['manual']} manual validation patterns. "
                "Convert to schema constraints where possible: "
                "enum for membership checks, "
                "minimum/maximum for range checks, "
                "pattern for string validation."
            )

        if type_counts.get("custom", 0) > 0:
            recommendations.append(
                f"Found {type_counts['custom']} custom validator patterns. "
                "Evaluate if these can be expressed as GreenLang cross-field rules "
                "using the $rules extension. Complex business logic may need to "
                "remain as code-based post-validation."
            )

        # JSON Schema recommendations
        if self._json_schemas:
            recommendations.append(
                f"Found {len(self._json_schemas)} existing JSON Schema file(s). "
                "Use 'greenlang schema migrate convert' to upgrade to "
                "Draft 2020-12 and add GreenLang extensions."
            )

        # General recommendations
        if patterns:
            recommendations.append(
                "Start migration with the 'easy' difficulty patterns first. "
                "Run 'greenlang schema lint' on migrated schemas to ensure "
                "best practices are followed."
            )
            recommendations.append(
                "Consider adding unit specifications ($unit) to numeric fields "
                "that represent physical quantities (energy, mass, distance)."
            )

        if not recommendations:
            recommendations.append(
                "No validation patterns detected. Your codebase may already be "
                "using GreenLang schemas or may not have validation logic."
            )

        return recommendations


# =============================================================================
# Schema Converter
# =============================================================================


class SchemaConverter:
    """
    Converts JSON Schema to GreenLang format.

    Handles conversion from various JSON Schema drafts to
    GreenLang's extended JSON Schema Draft 2020-12 format.

    Attributes:
        add_extensions: Whether to add GreenLang extensions automatically.

    Example:
        >>> converter = SchemaConverter()
        >>> result = converter.convert(original_schema)
        >>> print(yaml.dump(result.converted_schema))
    """

    # JSON Schema draft patterns
    DRAFT_PATTERNS = {
        "draft-04": re.compile(r"draft-0?4"),
        "draft-06": re.compile(r"draft-0?6"),
        "draft-07": re.compile(r"draft-0?7"),
        "2019-09": re.compile(r"2019-09"),
        "2020-12": re.compile(r"2020-12"),
    }

    # Property names that might need unit specifications
    UNIT_PROPERTY_PATTERNS = {
        "energy": {"dimension": "energy", "canonical": "kWh"},
        "power": {"dimension": "power", "canonical": "kW"},
        "mass": {"dimension": "mass", "canonical": "kg"},
        "weight": {"dimension": "mass", "canonical": "kg"},
        "distance": {"dimension": "length", "canonical": "m"},
        "length": {"dimension": "length", "canonical": "m"},
        "area": {"dimension": "area", "canonical": "m2"},
        "volume": {"dimension": "volume", "canonical": "L"},
        "temperature": {"dimension": "temperature", "canonical": "C"},
        "pressure": {"dimension": "pressure", "canonical": "Pa"},
        "currency": {"dimension": "currency", "canonical": "USD"},
        "emission": {"dimension": "mass", "canonical": "kgCO2e"},
        "carbon": {"dimension": "mass", "canonical": "kgCO2e"},
        "co2": {"dimension": "mass", "canonical": "kgCO2e"},
        "consumption": {"dimension": "energy", "canonical": "kWh"},
    }

    def __init__(self, add_extensions: bool = True):
        """
        Initialize SchemaConverter.

        Args:
            add_extensions: Whether to add GreenLang extensions automatically.
        """
        self.add_extensions = add_extensions

    def convert(
        self,
        schema: Dict[str, Any],
        add_extensions: bool = True
    ) -> SchemaConversion:
        """
        Convert JSON Schema to GreenLang format.

        Args:
            schema: The original JSON Schema dictionary.
            add_extensions: Whether to add GreenLang extensions.

        Returns:
            SchemaConversion result with converted schema and metadata.

        Example:
            >>> converter = SchemaConverter()
            >>> result = converter.convert({"type": "object", "properties": {...}})
        """
        start_time = time.perf_counter()
        warnings: List[str] = []
        gl_extensions: Dict[str, Any] = {}

        # Deep copy to avoid modifying original
        converted = self._deep_copy(schema)

        # Detect original dialect
        original_dialect = self._detect_dialect(schema)
        if original_dialect:
            logger.info(f"Detected JSON Schema dialect: {original_dialect}")

        # Upgrade to Draft 2020-12
        converted = self._upgrade_dialect(converted)

        # Add GreenLang extensions if requested
        if add_extensions or self.add_extensions:
            converted, gl_extensions, ext_warnings = self._add_gl_extensions(converted)
            warnings.extend(ext_warnings)

        # Add GreenLang version marker
        converted["$glVersion"] = "1.0"
        gl_extensions["$glVersion"] = "1.0"

        conversion_time_ms = (time.perf_counter() - start_time) * 1000

        return SchemaConversion(
            original_format="json-schema",
            original_dialect=original_dialect,
            converted_schema=converted,
            warnings=warnings,
            gl_extensions_added=gl_extensions,
            conversion_time_ms=conversion_time_ms,
        )

    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of an object.

        Args:
            obj: Object to copy.

        Returns:
            Deep copy of the object.
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def _detect_dialect(self, schema: Dict[str, Any]) -> Optional[str]:
        """
        Detect the JSON Schema dialect version.

        Args:
            schema: Schema dictionary.

        Returns:
            Detected dialect version or None.
        """
        schema_uri = schema.get("$schema", "")

        for dialect, pattern in self.DRAFT_PATTERNS.items():
            if pattern.search(schema_uri):
                return dialect

        return None

    def _upgrade_dialect(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgrade schema to JSON Schema Draft 2020-12.

        Args:
            schema: Schema dictionary.

        Returns:
            Updated schema dictionary.
        """
        # Set Draft 2020-12 as the dialect
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

        # Convert 'definitions' to '$defs' (2019-09+ change)
        if "definitions" in schema:
            schema["$defs"] = schema.pop("definitions")

        # Recursively process nested schemas
        self._upgrade_nested(schema)

        return schema

    def _upgrade_nested(self, schema: Dict[str, Any]) -> None:
        """
        Recursively upgrade nested schema structures.

        Args:
            schema: Schema dictionary to process in place.
        """
        # Convert 'definitions' to '$defs' in nested schemas
        if isinstance(schema, dict):
            if "definitions" in schema:
                schema["$defs"] = schema.pop("definitions")

            # Process properties
            if "properties" in schema and isinstance(schema["properties"], dict):
                for prop_schema in schema["properties"].values():
                    if isinstance(prop_schema, dict):
                        self._upgrade_nested(prop_schema)

            # Process items
            if "items" in schema and isinstance(schema["items"], dict):
                self._upgrade_nested(schema["items"])

            # Process additionalProperties
            if "additionalProperties" in schema and isinstance(
                schema["additionalProperties"], dict
            ):
                self._upgrade_nested(schema["additionalProperties"])

            # Process allOf, anyOf, oneOf
            for keyword in ("allOf", "anyOf", "oneOf"):
                if keyword in schema and isinstance(schema[keyword], list):
                    for sub_schema in schema[keyword]:
                        if isinstance(sub_schema, dict):
                            self._upgrade_nested(sub_schema)

            # Process $defs
            if "$defs" in schema and isinstance(schema["$defs"], dict):
                for def_schema in schema["$defs"].values():
                    if isinstance(def_schema, dict):
                        self._upgrade_nested(def_schema)

    def _add_gl_extensions(
        self,
        schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Add GreenLang extensions to schema.

        Args:
            schema: Schema dictionary.

        Returns:
            Tuple of (updated schema, extensions added, warnings).
        """
        extensions_added: Dict[str, Any] = {}
        warnings: List[str] = []

        # Process properties for potential unit specs
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    unit_spec = self._detect_units(prop_name, prop_schema)
                    if unit_spec:
                        prop_schema["$unit"] = unit_spec
                        extensions_added[f"properties/{prop_name}/$unit"] = unit_spec
                        warnings.append(
                            f"Added unit specification to '{prop_name}': "
                            f"dimension={unit_spec['dimension']}, "
                            f"canonical={unit_spec['canonical']}. "
                            "Please review and adjust if needed."
                        )

        # Add deprecation extension placeholder
        if not warnings:
            warnings.append(
                "No automatic unit specifications were added. "
                "Consider adding $unit to numeric fields representing "
                "physical quantities."
            )

        return schema, extensions_added, warnings

    def _detect_units(
        self,
        property_name: str,
        property_schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if a property should have a unit specification.

        Args:
            property_name: Name of the property.
            property_schema: Schema definition for the property.

        Returns:
            Unit specification dict or None.
        """
        # Only add units to numeric types
        prop_type = property_schema.get("type")
        if prop_type not in ("number", "integer"):
            return None

        # Check property name against known patterns
        name_lower = property_name.lower()
        for pattern, unit_spec in self.UNIT_PROPERTY_PATTERNS.items():
            if pattern in name_lower:
                return {
                    "dimension": unit_spec["dimension"],
                    "canonical": unit_spec["canonical"],
                    "allowed": [unit_spec["canonical"]],
                }

        return None


# =============================================================================
# Output Formatting
# =============================================================================


def format_analysis_text(report: MigrationReport, verbosity: int = 0) -> str:
    """
    Format migration report as human-readable text.

    Args:
        report: MigrationReport to format.
        verbosity: Verbosity level (0=summary, 1=detailed, 2=all).

    Returns:
        Formatted text string.
    """
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 60)
    lines.append("  Migration Analysis Report")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Files analyzed:          {report.analyzed_files}")
    lines.append(f"Validation patterns:     {report.validation_patterns_found}")
    lines.append(f"JSON Schemas found:      {len(report.json_schemas_found)}")
    lines.append(f"Estimated effort:        {report.estimated_effort.upper()}")
    lines.append(f"Analysis time:           {report.analysis_time_ms:.2f}ms")
    lines.append("")

    # Pattern breakdown
    if report.patterns:
        type_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}
        for pattern in report.patterns:
            type_counts[pattern.pattern_type] = (
                type_counts.get(pattern.pattern_type, 0) + 1
            )
            difficulty_counts[pattern.migration_difficulty] = (
                difficulty_counts.get(pattern.migration_difficulty, 0) + 1
            )

        lines.append("-" * 40)
        lines.append("Pattern Breakdown:")
        lines.append("-" * 40)
        for ptype, count in sorted(type_counts.items()):
            lines.append(f"  {ptype:15s}  {count:4d}")
        lines.append("")

        lines.append("Difficulty Breakdown:")
        for diff, count in sorted(difficulty_counts.items()):
            lines.append(f"  {diff:15s}  {count:4d}")
        lines.append("")

    # JSON Schemas found
    if report.json_schemas_found:
        lines.append("-" * 40)
        lines.append("JSON Schemas Found:")
        lines.append("-" * 40)
        for schema_path in report.json_schemas_found[:10]:
            lines.append(f"  {schema_path}")
        if len(report.json_schemas_found) > 10:
            lines.append(f"  ... and {len(report.json_schemas_found) - 10} more")
        lines.append("")

    # Detailed patterns (if verbose)
    if verbosity >= 1 and report.patterns:
        lines.append("-" * 40)
        lines.append("Detected Patterns:")
        lines.append("-" * 40)

        max_patterns = 100 if verbosity >= 2 else 20
        for i, pattern in enumerate(report.patterns[:max_patterns]):
            lines.append("")
            lines.append(f"[{i+1}] {pattern.pattern_type.upper()} ({pattern.migration_difficulty})")
            lines.append(f"    File: {pattern.file}:{pattern.line}")
            lines.append(f"    Code: {pattern.code_snippet[:80]}...")
            if verbosity >= 2:
                lines.append(f"    Suggestion: {pattern.suggestion}")

        if len(report.patterns) > max_patterns:
            lines.append(f"\n  ... and {len(report.patterns) - max_patterns} more patterns")
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append("-" * 40)
        lines.append("Recommendations:")
        lines.append("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"\n{i}. {rec}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_analysis_json(report: MigrationReport) -> str:
    """
    Format migration report as JSON.

    Args:
        report: MigrationReport to format.

    Returns:
        JSON string.
    """
    return report.model_dump_json(indent=2)


def format_conversion_text(conversion: SchemaConversion) -> str:
    """
    Format schema conversion result as human-readable text.

    Args:
        conversion: SchemaConversion result.

    Returns:
        Formatted text string.
    """
    lines = []

    lines.append("")
    lines.append("Schema Conversion Complete")
    lines.append("-" * 40)
    lines.append(f"Original format:  {conversion.original_format}")
    if conversion.original_dialect:
        lines.append(f"Original dialect: {conversion.original_dialect}")
    lines.append(f"Conversion time:  {conversion.conversion_time_ms:.2f}ms")
    lines.append("")

    if conversion.warnings:
        lines.append("Warnings:")
        for warning in conversion.warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    if conversion.gl_extensions_added:
        lines.append("GreenLang Extensions Added:")
        for ext_path, ext_value in conversion.gl_extensions_added.items():
            lines.append(f"  {ext_path}: {ext_value}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
def migrate():
    """
    Schema migration tools.

    Tools for migrating existing validation code and schemas to
    GreenLang's JSON Schema-based format.

    \b
    Subcommands:
        analyze  Scan codebase for validation patterns
        report   Generate detailed migration report (JSON)
        convert  Convert JSON Schema to GreenLang format

    \b
    Examples:
        # Analyze codebase for validation patterns
        greenlang schema migrate analyze --path ./src

        # Generate JSON migration report
        greenlang schema migrate report --path ./src --output migration.json

        # Convert existing JSON Schema
        greenlang schema migrate convert --input schema.json --output schema.yaml
    """
    pass


@migrate.command()
@click.option(
    "--path", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory or file to analyze.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, prints to stdout.",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (text or json).",
)
@click.option(
    "-v", "--verbose",
    "verbosity",
    count=True,
    help="Increase verbosity. -v shows patterns, -vv shows suggestions.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except errors.",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    path: str,
    output: Optional[str],
    output_format: str,
    verbosity: int,
    quiet: bool,
) -> None:
    """
    Analyze codebase for existing validation patterns.

    Scans Python files for validation code including:
    - jsonschema library usage
    - Pydantic model definitions
    - Manual validation (isinstance, assertions)
    - Custom validator implementations

    Also detects existing JSON Schema files.

    \b
    Examples:
        # Analyze src directory
        greenlang schema migrate analyze --path ./src

        # Verbose output with pattern details
        greenlang schema migrate analyze --path ./src -v

        # Output as JSON
        greenlang schema migrate analyze --path ./src --format json

        # Save to file
        greenlang schema migrate analyze --path ./src --output analysis.txt
    """
    # Configure logging
    if verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbosity >= 1:
        logging.basicConfig(level=logging.INFO)

    try:
        # Run analysis
        analyzer = MigrationAnalyzer(Path(path))
        report = analyzer.analyze()

        # Format output
        if output_format == "json":
            result = format_analysis_json(report)
        else:
            result = format_analysis_text(report, verbosity)

        # Output results
        if output:
            Path(output).write_text(result, encoding="utf-8")
            if not quiet:
                click.echo(f"Analysis saved to: {output}")
        elif not quiet:
            click.echo(result)

        # Exit code based on findings
        if report.validation_patterns_found > 0:
            ctx.exit(EXIT_WARNINGS)
        else:
            ctx.exit(EXIT_SUCCESS)

    except Exception as e:
        logger.exception("Analysis failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)


@migrate.command()
@click.option(
    "--path", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory or file to analyze.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output file path for the JSON report.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except errors.",
)
@click.pass_context
def report(
    ctx: click.Context,
    path: str,
    output: str,
    quiet: bool,
) -> None:
    """
    Generate detailed migration report in JSON format.

    Creates a comprehensive JSON report that can be used for:
    - Tracking migration progress
    - Generating work items/tickets
    - CI/CD integration

    \b
    Examples:
        # Generate migration report
        greenlang schema migrate report --path ./src --output migration.json

        # Quiet mode for CI
        greenlang schema migrate report -p ./src -o migration.json -q
    """
    try:
        # Run analysis
        analyzer = MigrationAnalyzer(Path(path))
        migration_report = analyzer.analyze()

        # Save as JSON
        json_output = migration_report.model_dump_json(indent=2)
        Path(output).write_text(json_output, encoding="utf-8")

        if not quiet:
            click.echo(f"Migration report saved to: {output}")
            click.echo(f"  Files analyzed: {migration_report.analyzed_files}")
            click.echo(f"  Patterns found: {migration_report.validation_patterns_found}")
            click.echo(f"  Estimated effort: {migration_report.estimated_effort}")

        # Exit code based on effort
        if migration_report.estimated_effort == "high":
            ctx.exit(EXIT_WARNINGS)
        else:
            ctx.exit(EXIT_SUCCESS)

    except Exception as e:
        logger.exception("Report generation failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)


@migrate.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON Schema file path.",
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output file path (.yaml or .json).",
)
@click.option(
    "--add-units",
    is_flag=True,
    default=False,
    help="Automatically add unit specifications based on property names.",
)
@click.option(
    "--no-extensions",
    is_flag=True,
    default=False,
    help="Skip adding GreenLang extensions.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except errors.",
)
@click.pass_context
def convert(
    ctx: click.Context,
    input_path: str,
    output_path: str,
    add_units: bool,
    no_extensions: bool,
    quiet: bool,
) -> None:
    """
    Convert JSON Schema to GreenLang schema format.

    Upgrades existing JSON Schema to Draft 2020-12 and optionally
    adds GreenLang extensions ($unit, $glVersion).

    \b
    Features:
        - Upgrades older drafts to 2020-12
        - Converts 'definitions' to '$defs'
        - Optionally adds unit specifications
        - Outputs as YAML or JSON based on extension

    \b
    Examples:
        # Basic conversion
        greenlang schema migrate convert --input schema.json --output schema.yaml

        # With automatic unit detection
        greenlang schema migrate convert -i schema.json -o schema.yaml --add-units

        # Without GreenLang extensions
        greenlang schema migrate convert -i old.json -o new.json --no-extensions
    """
    try:
        # Read input schema
        input_file = Path(input_path)
        content = input_file.read_text(encoding="utf-8")

        # Parse based on extension
        if input_file.suffix.lower() in YAML_EXTENSIONS:
            schema = yaml.safe_load(content)
        else:
            schema = json.loads(content)

        if not isinstance(schema, dict):
            raise click.BadParameter(
                f"Input file does not contain a valid schema object"
            )

        # Convert schema
        converter = SchemaConverter(add_extensions=not no_extensions)
        result = converter.convert(schema, add_extensions=add_units)

        # Write output
        output_file = Path(output_path)
        if output_file.suffix.lower() in YAML_EXTENSIONS:
            output_content = yaml.dump(
                result.converted_schema,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        else:
            output_content = json.dumps(
                result.converted_schema,
                indent=2,
                ensure_ascii=False,
            )

        output_file.write_text(output_content, encoding="utf-8")

        if not quiet:
            click.echo(format_conversion_text(result))
            click.echo(f"Converted schema saved to: {output_path}")

        # Exit with warnings if there were any
        if result.warnings:
            ctx.exit(EXIT_WARNINGS)
        else:
            ctx.exit(EXIT_SUCCESS)

    except json.JSONDecodeError as e:
        if not quiet:
            click.echo(f"Error: Invalid JSON in input file: {e}", err=True)
        ctx.exit(EXIT_ERROR)
    except yaml.YAMLError as e:
        if not quiet:
            click.echo(f"Error: Invalid YAML in input file: {e}", err=True)
        ctx.exit(EXIT_ERROR)
    except Exception as e:
        logger.exception("Conversion failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)


# Export for CLI registration
__all__ = [
    "migrate",
    "analyze",
    "report",
    "convert",
    "MigrationAnalyzer",
    "SchemaConverter",
    "ValidationPattern",
    "MigrationReport",
    "SchemaConversion",
]
