# -*- coding: utf-8 -*-
"""
Template Registry - Central Management for Agent Templates
==========================================================

Provides template loading, validation, versioning, composition, and
variable interpolation for the GreenLang Agent Factory.

Features:
- Template loading from YAML files
- Schema validation against AgentSpec DSL
- Semantic versioning support
- Template composition and inheritance
- Variable interpolation with type safety
- Caching for performance

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TemplateCategory(str, Enum):
    """Categories of agent templates."""
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    EMISSIONS = "emissions"
    MAINTENANCE = "maintenance"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"


class TemplateStatus(str, Enum):
    """Template lifecycle status."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VariableType(str, Enum):
    """Types for template variables."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ENUM = "enum"
    DATE = "date"
    DATETIME = "datetime"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class TemplateVersion:
    """
    Semantic version for templates.

    Format: MAJOR.MINOR.PATCH
    - MAJOR: Breaking changes
    - MINOR: New features, backward compatible
    - PATCH: Bug fixes, backward compatible
    """
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "TemplateVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, TemplateVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    @classmethod
    def from_string(cls, version_str: str) -> "TemplateVersion":
        """Parse version from string."""
        # Handle prerelease and build metadata
        prerelease = None
        build = None

        if "+" in version_str:
            version_str, build = version_str.split("+", 1)
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)

        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")

        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            prerelease=prerelease,
            build=build,
        )

    def increment_major(self) -> "TemplateVersion":
        """Increment major version."""
        return TemplateVersion(self.major + 1, 0, 0)

    def increment_minor(self) -> "TemplateVersion":
        """Increment minor version."""
        return TemplateVersion(self.major, self.minor + 1, 0)

    def increment_patch(self) -> "TemplateVersion":
        """Increment patch version."""
        return TemplateVersion(self.major, self.minor, self.patch + 1)


@dataclass
class TemplateVariable:
    """
    Template variable definition.

    Used for variable interpolation in templates.
    """
    name: str
    description: str
    var_type: VariableType
    default: Optional[Any] = None
    required: bool = True
    validation_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this variable definition.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required and self.default is None:
                return False, f"Variable '{self.name}' is required"
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Pattern validation
        if self.validation_pattern and isinstance(value, str):
            if not re.match(self.validation_pattern, value):
                return False, f"Value '{value}' does not match pattern '{self.validation_pattern}'"

        # Allowed values validation
        if self.allowed_values and value not in self.allowed_values:
            return False, f"Value '{value}' not in allowed values: {self.allowed_values}"

        # Range validation
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value {value} is above maximum {self.max_value}"

        return True, None

    def _validate_type(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value type."""
        type_map = {
            VariableType.STRING: str,
            VariableType.INTEGER: int,
            VariableType.FLOAT: (int, float),
            VariableType.BOOLEAN: bool,
            VariableType.LIST: list,
            VariableType.DICT: dict,
        }

        expected_type = type_map.get(self.var_type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Expected {self.var_type.value}, got {type(value).__name__}"

        return True, None


class TemplateMetadata(BaseModel):
    """
    Metadata for an agent template.

    Contains all non-functional information about a template.
    """
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable template name")
    description: str = Field(..., description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    version: str = Field(..., description="Semantic version string")
    status: TemplateStatus = Field(default=TemplateStatus.DRAFT, description="Lifecycle status")

    # Authorship
    author: str = Field(default="GreenLang Team", description="Template author")
    maintainer: Optional[str] = Field(None, description="Current maintainer")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Classification
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    equipment_types: List[str] = Field(default_factory=list, description="Applicable equipment")
    industries: List[str] = Field(default_factory=list, description="Target industries")

    # Standards and compliance
    applicable_standards: List[str] = Field(default_factory=list, description="Referenced standards")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")

    # Dependencies
    parent_template: Optional[str] = Field(None, description="Parent template for inheritance")
    dependencies: List[str] = Field(default_factory=list, description="Required templates")

    # Provenance
    content_hash: Optional[str] = Field(None, description="SHA-256 hash of template content")

    @validator("version")
    def validate_version(cls, v):
        """Validate semantic version format."""
        try:
            TemplateVersion.from_string(v)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {e}")
        return v

    def get_version_object(self) -> TemplateVersion:
        """Get version as TemplateVersion object."""
        return TemplateVersion.from_string(self.version)


# =============================================================================
# Template Loader
# =============================================================================

class TemplateLoader:
    """
    Loads and parses agent templates from YAML files.

    Features:
    - YAML parsing with safe loading
    - Template validation
    - Caching for performance
    - Directory scanning
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the template loader."""
        self.templates_dir = templates_dir or Path(__file__).parent
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._file_mtimes: Dict[str, float] = {}

    def load_template(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a template from a YAML file.

        Args:
            template_path: Path to template file (relative or absolute)

        Returns:
            Parsed template as dictionary

        Raises:
            FileNotFoundError: If template file not found
            yaml.YAMLError: If YAML parsing fails
        """
        # Resolve path
        if isinstance(template_path, str):
            template_path = Path(template_path)

        if not template_path.is_absolute():
            template_path = self.templates_dir / template_path

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Check cache
        cache_key = str(template_path)
        mtime = template_path.stat().st_mtime

        if cache_key in self._cache and self._file_mtimes.get(cache_key) == mtime:
            logger.debug(f"Using cached template: {template_path.name}")
            return self._cache[cache_key]

        # Load and parse
        logger.info(f"Loading template: {template_path.name}")
        with open(template_path, "r", encoding="utf-8") as f:
            template_data = yaml.safe_load(f)

        # Calculate content hash
        with open(template_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()

        template_data["_content_hash"] = content_hash
        template_data["_source_path"] = str(template_path)

        # Update cache
        self._cache[cache_key] = template_data
        self._file_mtimes[cache_key] = mtime

        return template_data

    def load_all_templates(self, category: Optional[TemplateCategory] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load all templates, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            Dictionary of template_id -> template_data
        """
        templates = {}

        # Determine directories to scan
        if category:
            scan_dirs = [self.templates_dir / category.value]
        else:
            scan_dirs = [
                self.templates_dir / cat.value
                for cat in TemplateCategory
                if (self.templates_dir / cat.value).exists()
            ]

        # Scan for YAML files
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            for yaml_file in scan_dir.glob("*.yaml"):
                try:
                    template_data = self.load_template(yaml_file)
                    template_id = template_data.get("metadata", {}).get("template_id")
                    if template_id:
                        templates[template_id] = template_data
                except Exception as e:
                    logger.warning(f"Failed to load template {yaml_file}: {e}")

        return templates

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()
        self._file_mtimes.clear()


# =============================================================================
# Template Validator
# =============================================================================

class TemplateValidator:
    """
    Validates agent templates against the AgentSpec DSL schema.

    Checks:
    - Required fields presence
    - Field type correctness
    - Cross-reference validity
    - Formula completeness
    - Safety constraint validity
    """

    # Required top-level sections
    REQUIRED_SECTIONS = ["metadata", "input_schema", "output_schema", "processing"]

    # Required metadata fields
    REQUIRED_METADATA = ["template_id", "name", "description", "category", "version"]

    # Required processing sections
    REQUIRED_PROCESSING = ["steps"]

    def __init__(self):
        """Initialize the validator."""
        self._validation_errors: List[str] = []

    def validate(self, template_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a template against the AgentSpec DSL.

        Args:
            template_data: Parsed template dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self._validation_errors = []

        # Check required sections
        self._validate_required_sections(template_data)

        # Validate metadata
        if "metadata" in template_data:
            self._validate_metadata(template_data["metadata"])

        # Validate input schema
        if "input_schema" in template_data:
            self._validate_schema(template_data["input_schema"], "input_schema")

        # Validate output schema
        if "output_schema" in template_data:
            self._validate_schema(template_data["output_schema"], "output_schema")

        # Validate processing
        if "processing" in template_data:
            self._validate_processing(template_data["processing"])

        # Validate formulas
        if "formulas" in template_data:
            self._validate_formulas(template_data["formulas"])

        # Validate safety constraints
        if "safety_constraints" in template_data:
            self._validate_safety_constraints(template_data["safety_constraints"])

        # Validate zero-hallucination config
        if "zero_hallucination" in template_data:
            self._validate_zero_hallucination(template_data["zero_hallucination"])

        is_valid = len(self._validation_errors) == 0
        return is_valid, self._validation_errors

    def _validate_required_sections(self, template_data: Dict[str, Any]):
        """Validate required top-level sections."""
        for section in self.REQUIRED_SECTIONS:
            if section not in template_data:
                self._validation_errors.append(f"Missing required section: {section}")

    def _validate_metadata(self, metadata: Dict[str, Any]):
        """Validate metadata section."""
        for field in self.REQUIRED_METADATA:
            if field not in metadata:
                self._validation_errors.append(f"Missing required metadata field: {field}")

        # Validate version format
        if "version" in metadata:
            try:
                TemplateVersion.from_string(metadata["version"])
            except ValueError:
                self._validation_errors.append(f"Invalid version format: {metadata['version']}")

        # Validate category
        if "category" in metadata:
            try:
                TemplateCategory(metadata["category"])
            except ValueError:
                self._validation_errors.append(f"Invalid category: {metadata['category']}")

    def _validate_schema(self, schema: Dict[str, Any], schema_name: str):
        """Validate input/output schema."""
        if "fields" not in schema:
            self._validation_errors.append(f"{schema_name} must have 'fields' section")
            return

        for field_name, field_def in schema.get("fields", {}).items():
            if "type" not in field_def:
                self._validation_errors.append(f"{schema_name}.{field_name} missing 'type'")
            if "description" not in field_def:
                self._validation_errors.append(f"{schema_name}.{field_name} missing 'description'")

    def _validate_processing(self, processing: Dict[str, Any]):
        """Validate processing section."""
        if "steps" not in processing:
            self._validation_errors.append("processing section must have 'steps'")
            return

        steps = processing["steps"]
        if not isinstance(steps, list):
            self._validation_errors.append("processing.steps must be a list")
            return

        for i, step in enumerate(steps):
            if "name" not in step:
                self._validation_errors.append(f"processing.steps[{i}] missing 'name'")
            if "type" not in step:
                self._validation_errors.append(f"processing.steps[{i}] missing 'type'")

    def _validate_formulas(self, formulas: List[Dict[str, Any]]):
        """Validate formulas section."""
        for i, formula in enumerate(formulas):
            if "id" not in formula:
                self._validation_errors.append(f"formulas[{i}] missing 'id'")
            if "equation" not in formula:
                self._validation_errors.append(f"formulas[{i}] missing 'equation'")
            if "source" not in formula:
                self._validation_errors.append(f"formulas[{i}] missing 'source' (standard reference)")

    def _validate_safety_constraints(self, constraints: List[Dict[str, Any]]):
        """Validate safety constraints."""
        for i, constraint in enumerate(constraints):
            if "id" not in constraint:
                self._validation_errors.append(f"safety_constraints[{i}] missing 'id'")
            if "condition" not in constraint:
                self._validation_errors.append(f"safety_constraints[{i}] missing 'condition'")
            if "action" not in constraint:
                self._validation_errors.append(f"safety_constraints[{i}] missing 'action'")

    def _validate_zero_hallucination(self, config: Dict[str, Any]):
        """Validate zero-hallucination configuration."""
        if "enabled" not in config:
            self._validation_errors.append("zero_hallucination missing 'enabled' flag")

        if config.get("enabled", False):
            if "calculation_mode" not in config:
                self._validation_errors.append("zero_hallucination missing 'calculation_mode'")

            valid_modes = ["deterministic", "formula_only", "lookup_only"]
            if config.get("calculation_mode") not in valid_modes:
                self._validation_errors.append(
                    f"Invalid calculation_mode: {config.get('calculation_mode')}"
                )


# =============================================================================
# Variable Interpolator
# =============================================================================

class VariableInterpolator:
    """
    Interpolates variables in template content.

    Supports:
    - Simple variable substitution: ${variable}
    - Default values: ${variable:default}
    - Nested paths: ${section.subsection.value}
    - Conditional interpolation
    - Type coercion
    """

    # Pattern for variable references: ${name} or ${name:default}
    VAR_PATTERN = re.compile(r'\$\{([a-zA-Z_][a-zA-Z0-9_\.]*)(:[^}]*)?\}')

    def __init__(self, variables: Optional[Dict[str, TemplateVariable]] = None):
        """Initialize the interpolator."""
        self.variables = variables or {}

    def interpolate(
        self,
        content: Union[str, Dict, List],
        values: Dict[str, Any],
        strict: bool = True
    ) -> Union[str, Dict, List]:
        """
        Interpolate variables in content.

        Args:
            content: Content to interpolate (string, dict, or list)
            values: Variable values
            strict: If True, raise error for undefined variables

        Returns:
            Interpolated content

        Raises:
            ValueError: If strict=True and variable is undefined
        """
        if isinstance(content, str):
            return self._interpolate_string(content, values, strict)
        elif isinstance(content, dict):
            return {
                self._interpolate_string(str(k), values, strict) if isinstance(k, str) else k:
                self.interpolate(v, values, strict)
                for k, v in content.items()
            }
        elif isinstance(content, list):
            return [self.interpolate(item, values, strict) for item in content]
        else:
            return content

    def _interpolate_string(self, text: str, values: Dict[str, Any], strict: bool) -> str:
        """Interpolate variables in a string."""
        def replace_var(match):
            var_path = match.group(1)
            default = match.group(2)

            # Remove leading colon from default
            if default:
                default = default[1:]

            # Get value from nested path
            value = self._get_nested_value(values, var_path)

            if value is None:
                if default is not None:
                    return default
                if strict:
                    raise ValueError(f"Undefined variable: {var_path}")
                return match.group(0)

            # Validate if variable definition exists
            if var_path in self.variables:
                is_valid, error = self.variables[var_path].validate(value)
                if not is_valid:
                    raise ValueError(f"Variable validation failed: {error}")

            return str(value)

        return self.VAR_PATTERN.sub(replace_var, text)

    def _get_nested_value(self, values: Dict[str, Any], path: str) -> Any:
        """Get value from nested path (e.g., 'a.b.c')."""
        parts = path.split(".")
        current = values

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def get_required_variables(self, content: Union[str, Dict, List]) -> Set[str]:
        """Extract all variable names from content."""
        variables = set()

        if isinstance(content, str):
            for match in self.VAR_PATTERN.finditer(content):
                variables.add(match.group(1))
        elif isinstance(content, dict):
            for k, v in content.items():
                if isinstance(k, str):
                    variables.update(self.get_required_variables(k))
                variables.update(self.get_required_variables(v))
        elif isinstance(content, list):
            for item in content:
                variables.update(self.get_required_variables(item))

        return variables


# =============================================================================
# Template Composer
# =============================================================================

class TemplateComposer:
    """
    Composes templates through inheritance and composition.

    Features:
    - Template inheritance (parent_template)
    - Section merging
    - Override handling
    - Conflict resolution
    """

    def __init__(self, loader: TemplateLoader):
        """Initialize the composer."""
        self.loader = loader

    def compose(
        self,
        template_data: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compose a template with inheritance and overrides.

        Args:
            template_data: Base template data
            overrides: Optional overrides to apply

        Returns:
            Composed template
        """
        result = template_data.copy()

        # Handle inheritance
        parent_id = template_data.get("metadata", {}).get("parent_template")
        if parent_id:
            parent_data = self._load_parent_template(parent_id)
            result = self._merge_templates(parent_data, result)

        # Apply overrides
        if overrides:
            result = self._deep_merge(result, overrides)

        return result

    def _load_parent_template(self, parent_id: str) -> Dict[str, Any]:
        """Load parent template by ID."""
        from . import TEMPLATE_FILES

        if parent_id not in TEMPLATE_FILES:
            raise ValueError(f"Parent template not found: {parent_id}")

        return self.loader.load_template(TEMPLATE_FILES[parent_id])

    def _merge_templates(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parent and child templates."""
        result = self._deep_merge(parent.copy(), child)

        # Update metadata to reflect inheritance
        result["metadata"]["parent_template"] = parent.get("metadata", {}).get("template_id")

        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # For lists, extend rather than replace (can be customized)
                result[key] = result[key] + [v for v in value if v not in result[key]]
            else:
                result[key] = value

        return result


# =============================================================================
# Template Registry
# =============================================================================

class TemplateRegistry:
    """
    Central registry for agent templates.

    Provides:
    - Template loading and caching
    - Validation
    - Version management
    - Composition
    - Search and discovery

    Example:
        >>> registry = TemplateRegistry()
        >>> template = registry.get_template("boiler_efficiency_agent")
        >>> if registry.validate_template(template):
        ...     agent = registry.instantiate_agent(template)
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize the template registry.

        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or Path(__file__).parent
        self.loader = TemplateLoader(self.templates_dir)
        self.validator = TemplateValidator()
        self.interpolator = VariableInterpolator()
        self.composer = TemplateComposer(self.loader)

        self._templates: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[str, TemplateMetadata] = {}

        logger.info(f"Template registry initialized: {self.templates_dir}")

    def load_all(self, category: Optional[TemplateCategory] = None) -> int:
        """
        Load all templates into the registry.

        Args:
            category: Optional category filter

        Returns:
            Number of templates loaded
        """
        templates = self.loader.load_all_templates(category)
        self._templates.update(templates)

        # Build metadata cache
        for template_id, template_data in templates.items():
            try:
                metadata = self._extract_metadata(template_data)
                self._metadata_cache[template_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to extract metadata for {template_id}: {e}")

        logger.info(f"Loaded {len(templates)} templates")
        return len(templates)

    def get_template(
        self,
        template_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a template by ID.

        Args:
            template_id: Template identifier
            version: Optional specific version

        Returns:
            Template data or None if not found
        """
        # Check cache
        if template_id in self._templates:
            template = self._templates[template_id]
            if version is None or template.get("metadata", {}).get("version") == version:
                return template

        # Try loading from file
        from . import TEMPLATE_FILES

        if template_id in TEMPLATE_FILES:
            try:
                template = self.loader.load_template(TEMPLATE_FILES[template_id])
                self._templates[template_id] = template
                return template
            except FileNotFoundError:
                pass

        return None

    def get_metadata(self, template_id: str) -> Optional[TemplateMetadata]:
        """Get template metadata by ID."""
        if template_id in self._metadata_cache:
            return self._metadata_cache[template_id]

        template = self.get_template(template_id)
        if template:
            return self._extract_metadata(template)

        return None

    def validate_template(self, template_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a template.

        Args:
            template_data: Template to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.validator.validate(template_data)

    def compose_template(
        self,
        template_id: str,
        variables: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compose a template with variable interpolation and overrides.

        Args:
            template_id: Base template ID
            variables: Variable values for interpolation
            overrides: Template section overrides

        Returns:
            Composed and interpolated template
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Compose with parent if applicable
        composed = self.composer.compose(template, overrides)

        # Interpolate variables
        if variables:
            composed = self.interpolator.interpolate(composed, variables, strict=False)

        return composed

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[TemplateCategory] = None,
        equipment_type: Optional[str] = None,
        standard: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[TemplateMetadata]:
        """
        Search templates by various criteria.

        Args:
            query: Text search in name/description
            category: Filter by category
            equipment_type: Filter by equipment type
            standard: Filter by applicable standard
            tags: Filter by tags (any match)

        Returns:
            List of matching template metadata
        """
        results = []

        for template_id, metadata in self._metadata_cache.items():
            # Category filter
            if category and metadata.category != category:
                continue

            # Equipment type filter
            if equipment_type and equipment_type not in metadata.equipment_types:
                continue

            # Standard filter
            if standard and standard not in metadata.applicable_standards:
                continue

            # Tags filter
            if tags and not any(tag in metadata.tags for tag in tags):
                continue

            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower not in metadata.name.lower() and
                    query_lower not in metadata.description.lower()):
                    continue

            results.append(metadata)

        return results

    def list_templates(
        self,
        category: Optional[TemplateCategory] = None
    ) -> List[str]:
        """List all template IDs, optionally filtered by category."""
        if category:
            return [
                tid for tid, meta in self._metadata_cache.items()
                if meta.category == category
            ]
        return list(self._metadata_cache.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_templates": len(self._templates),
            "by_category": {},
            "by_status": {},
            "total_formulas": 0,
            "total_constraints": 0,
        }

        for metadata in self._metadata_cache.values():
            # Count by category
            cat = metadata.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by status
            status = metadata.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Count formulas and constraints
        for template in self._templates.values():
            stats["total_formulas"] += len(template.get("formulas", []))
            stats["total_constraints"] += len(template.get("safety_constraints", []))

        return stats

    def _extract_metadata(self, template_data: Dict[str, Any]) -> TemplateMetadata:
        """Extract metadata from template data."""
        meta_dict = template_data.get("metadata", {})
        return TemplateMetadata(
            template_id=meta_dict.get("template_id", "unknown"),
            name=meta_dict.get("name", "Unknown Template"),
            description=meta_dict.get("description", ""),
            category=TemplateCategory(meta_dict.get("category", "efficiency")),
            version=meta_dict.get("version", "0.0.1"),
            status=TemplateStatus(meta_dict.get("status", "draft")),
            author=meta_dict.get("author", "GreenLang Team"),
            tags=meta_dict.get("tags", []),
            equipment_types=meta_dict.get("equipment_types", []),
            applicable_standards=meta_dict.get("applicable_standards", []),
            content_hash=template_data.get("_content_hash"),
        )


# =============================================================================
# Module-level Singleton
# =============================================================================

_registry_instance: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """
    Get or create the global template registry instance.

    Returns:
        Global TemplateRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TemplateRegistry()
        _registry_instance.load_all()
    return _registry_instance
