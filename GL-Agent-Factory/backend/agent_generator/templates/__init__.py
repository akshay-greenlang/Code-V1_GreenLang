"""
GreenLang Agent Factory - Jinja2 Template System

This module provides the template engine and filters for generating
agent code from pack.yaml specifications.

Templates:
- agent/agent_class.py.j2  - Main agent class
- tests/test_agent.py.j2   - Test suite
- tools/tool_wrapper.py.j2 - Tool wrappers

Custom Filters:
- pascal_case: Convert to PascalCase
- snake_case: Convert to snake_case
- dtype_to_python: Convert dtype to Python type
- truncate: Truncate string with ellipsis
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


# =============================================================================
# Template Directory
# =============================================================================

TEMPLATE_DIR = Path(__file__).parent


# =============================================================================
# Custom Filters
# =============================================================================

def pascal_case(value: str) -> str:
    """
    Convert string to PascalCase.

    Examples:
        - "my-agent-name" -> "MyAgentName"
        - "gl_carbon_emissions_v1" -> "GlCarbonEmissionsV1"
        - "foo_bar" -> "FooBar"
    """
    # Split on non-alphanumeric characters
    words = re.split(r'[-_\s]+', str(value))
    # Capitalize each word
    return ''.join(word.capitalize() for word in words)


def snake_case(value: str) -> str:
    """
    Convert string to snake_case.

    Examples:
        - "MyAgentName" -> "my_agent_name"
        - "gl-carbon-emissions" -> "gl_carbon_emissions"
    """
    # Replace hyphens and spaces with underscores
    value = re.sub(r'[-\s]+', '_', str(value))
    # Insert underscore before uppercase letters
    value = re.sub(r'([a-z])([A-Z])', r'\1_\2', value)
    return value.lower()


def dtype_to_python(dtype: str) -> str:
    """
    Convert pack.yaml dtype to Python type annotation.

    Mapping:
        - string -> str
        - float64 -> float
        - float32 -> float
        - int32 -> int
        - int64 -> int
        - bool -> bool
        - object -> Dict[str, Any]
        - array -> List[Any]
        - datetime -> datetime
    """
    mapping = {
        "string": "str",
        "str": "str",
        "float64": "float",
        "float32": "float",
        "float": "float",
        "int32": "int",
        "int64": "int",
        "int": "int",
        "integer": "int",
        "bool": "bool",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "dict": "Dict[str, Any]",
        "array": "List[Any]",
        "list": "List[Any]",
        "datetime": "datetime",
        "date": "datetime",
    }
    return mapping.get(dtype.lower(), "Any")


def truncate_filter(value: str, length: int = 80, end: str = "...") -> str:
    """Truncate string to specified length with ellipsis."""
    value = str(value)
    if len(value) <= length:
        return value
    return value[:length - len(end)] + end


def wordwrap_filter(value: str, width: int = 72) -> str:
    """Wrap text to specified width."""
    import textwrap
    return textwrap.fill(str(value), width=width)


def indent_filter(value: str, width: int = 4) -> str:
    """Indent all lines by specified width."""
    indent = " " * width
    return "\n".join(indent + line for line in str(value).split("\n"))


def tojson_filter(value: Any) -> str:
    """Convert value to JSON string (Python repr for templates)."""
    import json
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


# =============================================================================
# Template Engine
# =============================================================================

class TemplateEngine:
    """
    Jinja2 template engine for agent code generation.

    This engine loads templates from the templates directory and provides
    custom filters for code generation.

    Example:
        >>> engine = TemplateEngine()
        >>> code = engine.render("agent/agent_class.py.j2", {
        ...     "pack": spec.pack,
        ...     "compute": spec.compute,
        ...     "generated_at": datetime.utcnow().isoformat(),
        ... })
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize template engine.

        Args:
            template_dir: Path to templates directory (default: module directory)
        """
        self.template_dir = template_dir or TEMPLATE_DIR

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        # Register custom filters
        self._register_filters()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self.env.filters["pascal_case"] = pascal_case
        self.env.filters["snake_case"] = snake_case
        self.env.filters["dtype_to_python"] = dtype_to_python
        self.env.filters["truncate"] = truncate_filter
        self.env.filters["wordwrap"] = wordwrap_filter
        self.env.filters["indent"] = indent_filter
        self.env.filters["tojson"] = tojson_filter

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.

        Args:
            template_name: Template file name (e.g., "agent/agent_class.py.j2")
            context: Template context variables

        Returns:
            Rendered template as string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_agent(self, spec: Dict[str, Any]) -> str:
        """
        Render agent class from pack specification.

        Args:
            spec: Parsed pack.yaml specification

        Returns:
            Generated agent.py code
        """
        import hashlib

        context = {
            "schema_version": spec.get("schema_version", "2.0.0"),
            "pack": spec.get("pack", {}),
            "metadata": spec.get("metadata", {}),
            "compute": spec.get("compute", {}),
            "ai": spec.get("ai", {}),
            "tools": spec.get("tools", []),
            "factors": spec.get("factors", []),
            "provenance": spec.get("provenance", {}),
            "tests": spec.get("tests", {}),
            "certification": spec.get("certification", {}),
            "generated_at": datetime.utcnow().isoformat(),
            "code_hash": hashlib.sha256(
                str(spec).encode()
            ).hexdigest()[:16],
        }

        return self.render("agent/agent_class.py.j2", context)

    def render_tests(self, spec: Dict[str, Any], module_path: str) -> str:
        """
        Render test suite from pack specification.

        Args:
            spec: Parsed pack.yaml specification
            module_path: Python module path for imports

        Returns:
            Generated test_agent.py code
        """
        context = {
            "pack": spec.get("pack", {}),
            "tests": spec.get("tests", {}),
            "module_path": module_path,
            "generated_at": datetime.utcnow().isoformat(),
        }

        return self.render("tests/test_agent.py.j2", context)

    def get_available_templates(self) -> Dict[str, list]:
        """Get list of available templates by category."""
        templates = {
            "agent": [],
            "tests": [],
            "tools": [],
        }

        for category in templates.keys():
            category_dir = self.template_dir / category
            if category_dir.exists():
                templates[category] = [
                    f.name for f in category_dir.glob("*.j2")
                ]

        return templates


# =============================================================================
# Module-level instance
# =============================================================================

_engine: Optional[TemplateEngine] = None


def get_template_engine() -> TemplateEngine:
    """Get or create the global template engine instance."""
    global _engine
    if _engine is None:
        _engine = TemplateEngine()
    return _engine


def render_template(template_name: str, context: Dict[str, Any]) -> str:
    """Convenience function to render a template."""
    return get_template_engine().render(template_name, context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TemplateEngine",
    "get_template_engine",
    "render_template",
    "pascal_case",
    "snake_case",
    "dtype_to_python",
    "TEMPLATE_DIR",
]
