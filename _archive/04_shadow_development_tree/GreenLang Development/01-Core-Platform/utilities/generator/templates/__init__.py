"""
Jinja2 Templates for Agent Generation.

This package contains Jinja2 templates used by the CodeGenerator:
- agent_class.py.j2: Main agent class template
- tools.py.j2: Tool wrapper classes template
- test_agent.py.j2: Test suite template

Templates use the following context variables:
- spec: ParsedAgentSpec object
- name, version, id: Agent identity
- inputs, outputs: Field definitions
- tools: Tool definitions
- tests: Test definitions
- provenance: Provenance configuration
- ai: AI configuration

Custom filters:
- snake_to_pascal: Convert snake_case to PascalCase
- snake_to_camel: Convert snake_case to camelCase
- json_type_to_python: Convert JSON Schema type to Python type
- format_docstring: Format text as docstring
- escape_string: Escape string for Python code
"""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent

TEMPLATES = {
    "agent_class": "agent_class.py.j2",
    "tools": "tools.py.j2",
    "test_agent": "test_agent.py.j2",
}


def get_template_path(name: str) -> Path:
    """Get full path to a template file."""
    if name in TEMPLATES:
        return TEMPLATE_DIR / TEMPLATES[name]
    raise ValueError(f"Unknown template: {name}")


__all__ = [
    "TEMPLATE_DIR",
    "TEMPLATES",
    "get_template_path",
]
