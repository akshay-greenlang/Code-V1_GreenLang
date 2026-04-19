"""
Model Generator Module for GreenLang Agent Factory

This module generates Pydantic models for agent input/output types.
It creates comprehensive model definitions with:

- Full type hints for all fields
- Pydantic validators and constraints
- Enum classes for constrained values
- Nested model definitions
- Documentation strings

Example:
    >>> generator = ModelGenerator(config)
    >>> models_code = generator.generate_models(spec)
    >>> print(models_code)  # Complete models.py content
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import Environment, BaseLoader

from backend.agent_generator.parser.yaml_parser import (
    AgentSpec,
    AgentDefinition,
    InputDefinition,
    OutputDefinition,
    ToolDefinition,
    SchemaDefinition,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type System
# =============================================================================

class TypeInfo:
    """Information about a type for code generation."""

    def __init__(
        self,
        python_type: str,
        is_primitive: bool = True,
        is_optional: bool = False,
        is_list: bool = False,
        is_dict: bool = False,
        is_enum: bool = False,
        is_custom: bool = False,
        inner_type: Optional[str] = None,
        enum_values: Optional[List[str]] = None,
    ):
        self.python_type = python_type
        self.is_primitive = is_primitive
        self.is_optional = is_optional
        self.is_list = is_list
        self.is_dict = is_dict
        self.is_enum = is_enum
        self.is_custom = is_custom
        self.inner_type = inner_type
        self.enum_values = enum_values or []

    def get_full_type(self) -> str:
        """Get the full Python type annotation."""
        if self.is_optional:
            return f"Optional[{self.python_type}]"
        return self.python_type


class TypeMapper:
    """
    Maps YAML/JSON types to Python types.

    Handles:
    - Primitive types (str, int, float, bool)
    - Complex types (List, Dict, Optional)
    - Custom types (Pydantic models)
    - Date/datetime types
    - Enum types
    """

    # Mapping from YAML types to Python types
    PRIMITIVE_TYPES = {
        # String types
        "string": "str",
        "str": "str",
        "text": "str",

        # Numeric types
        "integer": "int",
        "int": "int",
        "int32": "int",
        "int64": "int",
        "float": "float",
        "float32": "float",
        "float64": "float",
        "double": "float",
        "number": "float",
        "decimal": "Decimal",

        # Boolean types
        "boolean": "bool",
        "bool": "bool",

        # Date/time types
        "date": "date",
        "datetime": "datetime",
        "timestamp": "datetime",
        "time": "time",

        # Other types
        "any": "Any",
        "null": "None",
        "none": "None",
        "void": "None",
    }

    # Complex types
    COMPLEX_TYPES = {
        "object": "Dict[str, Any]",
        "dict": "Dict[str, Any]",
        "dictionary": "Dict[str, Any]",
        "map": "Dict[str, Any]",
        "array": "List[Any]",
        "list": "List[Any]",
    }

    def __init__(self):
        """Initialize the type mapper."""
        self._custom_types: Set[str] = set()

    def map_type(self, type_str: str, context: str = "") -> TypeInfo:
        """
        Map a type string to TypeInfo.

        Args:
            type_str: The type string from YAML
            context: Context for error messages

        Returns:
            TypeInfo with Python type details
        """
        if not type_str:
            return TypeInfo("Any", is_primitive=True)

        # Clean up type string
        type_str = type_str.strip()
        lower_type = type_str.lower()

        # Check for Optional prefix
        is_optional = False
        if lower_type.startswith("optional[") and type_str.endswith("]"):
            is_optional = True
            type_str = type_str[9:-1]
            lower_type = type_str.lower()

        # Check for primitive types
        if lower_type in self.PRIMITIVE_TYPES:
            python_type = self.PRIMITIVE_TYPES[lower_type]
            return TypeInfo(
                python_type=python_type,
                is_primitive=True,
                is_optional=is_optional,
            )

        # Check for complex types
        if lower_type in self.COMPLEX_TYPES:
            python_type = self.COMPLEX_TYPES[lower_type]
            return TypeInfo(
                python_type=python_type,
                is_primitive=False,
                is_optional=is_optional,
                is_dict="Dict" in python_type,
                is_list="List" in python_type,
            )

        # Check for List[X] format
        list_match = re.match(r"list\[(.+)\]", type_str, re.IGNORECASE)
        if list_match:
            inner_type = list_match.group(1)
            inner_info = self.map_type(inner_type)
            return TypeInfo(
                python_type=f"List[{inner_info.python_type}]",
                is_primitive=False,
                is_optional=is_optional,
                is_list=True,
                inner_type=inner_info.python_type,
            )

        # Check for Dict[K, V] format
        dict_match = re.match(r"dict\[(.+),\s*(.+)\]", type_str, re.IGNORECASE)
        if dict_match:
            key_type = self.map_type(dict_match.group(1)).python_type
            value_type = self.map_type(dict_match.group(2)).python_type
            return TypeInfo(
                python_type=f"Dict[{key_type}, {value_type}]",
                is_primitive=False,
                is_optional=is_optional,
                is_dict=True,
            )

        # Assume custom type (keep as-is, should be PascalCase)
        self._custom_types.add(type_str)
        return TypeInfo(
            python_type=type_str,
            is_primitive=False,
            is_optional=is_optional,
            is_custom=True,
        )

    def get_custom_types(self) -> Set[str]:
        """Get all custom types that were encountered."""
        return self._custom_types.copy()


# =============================================================================
# Templates
# =============================================================================

MODELS_TEMPLATE = '''"""
Pydantic Models for {{ spec.pack.name }}

Generated by GreenLang Agent Factory.

This module contains all input/output models and enums used by
the agents in this pack.

Pack: {{ spec.pack.name }} v{{ spec.pack.version }}
Generated: {{ timestamp }}
"""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# Enums
# =============================================================================

{% for enum in enums %}
class {{ enum.name }}(str, Enum):
    """{{ enum.description }}"""

{% for value in enum.values %}
    {{ value.name }} = "{{ value.value }}"
{% endfor %}


{% endfor %}
# =============================================================================
# Base Models
# =============================================================================

class ProvenanceInfo(BaseModel):
    """
    Provenance tracking information for audit compliance.

    This model tracks the complete audit trail for any calculation,
    enabling verification and reproducibility.
    """

    provenance_hash: str = Field(..., description="SHA-256 hash of calculation chain")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = Field(..., description="ID of the agent that performed calculation")
    agent_version: str = Field(..., description="Version of the agent")


class ValidationStatus(str, Enum):
    """Validation status values."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"


# =============================================================================
# Custom Type Models
# =============================================================================

{% for model in custom_models %}
class {{ model.name }}(BaseModel):
    """
    {{ model.description }}

{% if model.fields %}
    Attributes:
{% for field in model.fields %}
        {{ field.name }}: {{ field.description or field.name }}
{% endfor %}
{% endif %}
    """

{% if not model.fields %}
    pass
{% else %}
{% for field in model.fields %}
    {{ field.name }}: {{ field.type }}{% if field.default is not none %} = {{ field.default }}{% elif not field.required %} = None{% endif %}

{% endfor %}
{% endif %}

{% endfor %}
# =============================================================================
# Agent Input/Output Models
# =============================================================================

{% for agent in spec.agents %}
# --- {{ agent.name }} Models ---

class {{ agent.get_class_name() }}Input(BaseModel):
    """
    Input model for {{ agent.name }}.

{% if agent.inputs %}
    Attributes:
{% for inp in agent.inputs %}
        {{ inp.name }}: {{ inp.description or inp.name }}
{% endfor %}
{% endif %}
    """

{% if not agent.inputs %}
    pass
{% else %}
{% for inp in agent.inputs %}
    {{ generate_field(inp) }}
{% endfor %}
{% endif %}


class {{ agent.get_class_name() }}Output(BaseModel):
    """
    Output model for {{ agent.name }}.

    All outputs include provenance tracking for audit compliance.

{% if agent.outputs %}
    Attributes:
{% for out in agent.outputs %}
        {{ out.name }}: {{ out.description or out.name }}
{% endfor %}
{% endif %}
    """

{% if not agent.outputs %}
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    validation_status: ValidationStatus = Field(..., description="Validation status")
{% else %}
{% for out in agent.outputs %}
    {{ generate_output_field(out) }}
{% endfor %}
{% if not has_provenance(agent) %}
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
{% endif %}
{% if not has_processing_time(agent) %}
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
{% endif %}
{% endif %}


{% endfor %}
# =============================================================================
# Tool Schema Models
# =============================================================================

{% for tool in spec.tools %}
{% if tool.input_schema %}
class {{ tool.id | pascal_case }}Input(BaseModel):
    """Input model for {{ tool.name }}."""

{% for prop_name, prop_def in (tool.input_schema.properties or {}).items() %}
    {{ prop_name }}: {{ schema_to_type(prop_def) }}{% if prop_name not in (tool.input_schema.required or []) %} = None{% endif %}

{% endfor %}
{% if not tool.input_schema.properties %}
    pass
{% endif %}


{% endif %}
{% if tool.output_schema %}
class {{ tool.id | pascal_case }}Output(BaseModel):
    """Output model for {{ tool.name }}."""

{% for prop_name, prop_def in (tool.output_schema.properties or {}).items() %}
    {{ prop_name }}: {{ schema_to_type(prop_def) }}{% if prop_name not in (tool.output_schema.required or []) %} = None{% endif %}

{% endfor %}
{% if not tool.output_schema.properties %}
    pass
{% endif %}


{% endif %}
{% endfor %}
# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base models
    "ProvenanceInfo",
    "ValidationStatus",

    # Enums
{% for enum in enums %}
    "{{ enum.name }}",
{% endfor %}

    # Custom models
{% for model in custom_models %}
    "{{ model.name }}",
{% endfor %}

    # Agent models
{% for agent in spec.agents %}
    "{{ agent.get_class_name() }}Input",
    "{{ agent.get_class_name() }}Output",
{% endfor %}

    # Tool models
{% for tool in spec.tools %}
{% if tool.input_schema %}
    "{{ tool.id | pascal_case }}Input",
{% endif %}
{% if tool.output_schema %}
    "{{ tool.id | pascal_case }}Output",
{% endif %}
{% endfor %}
]
'''


# =============================================================================
# Helper Classes
# =============================================================================

class EnumDefinition:
    """Definition for an enum to generate."""

    def __init__(
        self,
        name: str,
        description: str = "",
        values: Optional[List[Tuple[str, str]]] = None,
    ):
        self.name = name
        self.description = description
        self.values = [
            {"name": v[0], "value": v[1]}
            for v in (values or [])
        ]


class CustomModelDefinition:
    """Definition for a custom model to generate."""

    def __init__(
        self,
        name: str,
        description: str = "",
        fields: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.description = description
        self.fields = fields or []


# =============================================================================
# Model Generator
# =============================================================================

class ModelGenerator:
    """
    Generator for Pydantic models.

    This generator creates models.py files containing all input/output
    models and enums for a pack's agents.

    Features:
    - Full type hints for all fields
    - Pydantic validators and constraints
    - Enum classes for constrained values
    - Nested model definitions
    - Documentation strings

    Example:
        >>> generator = ModelGenerator(config)
        >>> code = generator.generate_models(spec)
    """

    def __init__(self, config: Any = None):
        """
        Initialize the model generator.

        Args:
            config: Generator configuration
        """
        self.config = config
        self.type_mapper = TypeMapper()
        self._env = Environment(loader=BaseLoader())

        # Add custom filters
        self._env.filters["pascal_case"] = self._to_pascal_case
        self._env.filters["snake_case"] = self._to_snake_case

        logger.info("ModelGenerator initialized")

    def generate_models(self, spec: AgentSpec) -> str:
        """
        Generate complete models.py code for a spec.

        Args:
            spec: The AgentSpec to generate models for

        Returns:
            Complete Python module code as string
        """
        logger.info(f"Generating models for pack: {spec.pack.id}")

        # Collect enums
        enums = self._collect_enums(spec)

        # Collect custom models
        custom_models = self._collect_custom_models(spec)

        # Create template functions
        def generate_field(inp: InputDefinition) -> str:
            return self._generate_input_field(inp)

        def generate_output_field(out: OutputDefinition) -> str:
            return self._generate_output_field(out)

        def has_provenance(agent: AgentDefinition) -> bool:
            return any(out.name == "provenance_hash" for out in agent.outputs)

        def has_processing_time(agent: AgentDefinition) -> bool:
            return any(out.name == "processing_time_ms" for out in agent.outputs)

        def schema_to_type(prop_def: Dict[str, Any]) -> str:
            return self._schema_property_to_type(prop_def)

        # Render template
        template = self._env.from_string(MODELS_TEMPLATE)
        code = template.render(
            spec=spec,
            enums=enums,
            custom_models=custom_models,
            generate_field=generate_field,
            generate_output_field=generate_output_field,
            has_provenance=has_provenance,
            has_processing_time=has_processing_time,
            schema_to_type=schema_to_type,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(f"Generated {len(code)} characters of model code")
        return code

    def generate_input_model(
        self,
        agent: AgentDefinition,
        class_name: str,
    ) -> str:
        """
        Generate a single input model.

        Args:
            agent: Agent definition
            class_name: Name for the model class

        Returns:
            Python class definition as string
        """
        lines = [
            f"class {class_name}(BaseModel):",
            f'    """Input model for {agent.name}."""',
            "",
        ]

        if not agent.inputs:
            lines.append("    pass")
        else:
            for inp in agent.inputs:
                lines.append(f"    {self._generate_input_field(inp)}")

        return "\n".join(lines)

    def generate_output_model(
        self,
        agent: AgentDefinition,
        class_name: str,
    ) -> str:
        """
        Generate a single output model.

        Args:
            agent: Agent definition
            class_name: Name for the model class

        Returns:
            Python class definition as string
        """
        lines = [
            f"class {class_name}(BaseModel):",
            f'    """Output model for {agent.name}."""',
            "",
        ]

        if not agent.outputs:
            lines.append('    provenance_hash: str = Field(..., description="SHA-256 provenance hash")')
            lines.append('    processing_time_ms: float = Field(..., description="Processing time")')
        else:
            for out in agent.outputs:
                lines.append(f"    {self._generate_output_field(out)}")

            # Add standard provenance fields if missing
            output_names = {out.name for out in agent.outputs}
            if "provenance_hash" not in output_names:
                lines.append('    provenance_hash: str = Field(..., description="SHA-256 provenance hash")')
            if "processing_time_ms" not in output_names:
                lines.append('    processing_time_ms: float = Field(..., description="Processing time")')

        return "\n".join(lines)

    def _generate_input_field(self, inp: InputDefinition) -> str:
        """Generate a Pydantic field for an input."""
        type_info = self.type_mapper.map_type(inp.type)

        # Determine Python type
        if inp.required:
            python_type = type_info.python_type
        else:
            python_type = f"Optional[{type_info.python_type}]"

        # Build Field arguments
        field_args = []

        if inp.required:
            field_args.append("...")
        elif inp.default is not None:
            field_args.append(self._format_default(inp.default, type_info))
        else:
            field_args.append("None")

        if inp.description:
            # Escape quotes in description
            desc = inp.description.replace('"', '\\"')
            field_args.append(f'description="{desc}"')

        # Add constraints from the constraints dict
        for key, value in inp.constraints.items():
            if key in ("ge", "gt", "le", "lt"):
                field_args.append(f"{key}={value}")
            elif key in ("min_length", "max_length"):
                field_args.append(f"{key}={value}")
            elif key == "regex":
                field_args.append(f'regex=r"{value}"')
            elif key == "enum":
                # Will be handled as enum type
                pass

        return f"{inp.name}: {python_type} = Field({', '.join(field_args)})"

    def _generate_output_field(self, out: OutputDefinition) -> str:
        """Generate a Pydantic field for an output."""
        type_info = self.type_mapper.map_type(out.type)
        python_type = type_info.python_type

        desc = out.description.replace('"', '\\"') if out.description else out.name
        return f'{out.name}: {python_type} = Field(..., description="{desc}")'

    def _format_default(self, value: Any, type_info: TypeInfo) -> str:
        """Format a default value for code generation."""
        if value is None:
            return "None"
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return repr(value)
        if isinstance(value, dict):
            return repr(value)
        return repr(value)

    def _collect_enums(self, spec: AgentSpec) -> List[EnumDefinition]:
        """Collect all enums that need to be generated."""
        enums = []

        # Check for enum constraints in inputs
        for agent in spec.agents:
            for inp in agent.inputs:
                if "enum" in inp.constraints:
                    enum_values = inp.constraints["enum"]
                    enum_name = self._to_pascal_case(inp.name) + "Type"
                    values = [
                        (self._to_snake_case(v).upper(), v)
                        for v in enum_values
                    ]
                    enums.append(EnumDefinition(
                        name=enum_name,
                        description=f"Enum for {inp.name}",
                        values=values,
                    ))

        # Check for enum types in tool schemas
        for tool in spec.tools:
            if tool.output_schema:
                # output_schema is a dict, access properties via get()
                properties = tool.output_schema.get('properties', {}) if isinstance(tool.output_schema, dict) else {}
                for prop_name, prop_def in properties.items():
                    if isinstance(prop_def, dict) and "enum" in prop_def:
                        enum_values = prop_def["enum"]
                        enum_name = self._to_pascal_case(prop_name) + "Type"
                        values = [
                            (self._to_snake_case(v).upper(), v)
                            for v in enum_values
                        ]
                        enums.append(EnumDefinition(
                            name=enum_name,
                            description=f"Enum for {prop_name}",
                            values=values,
                        ))

        return enums

    def _collect_custom_models(self, spec: AgentSpec) -> List[CustomModelDefinition]:
        """Collect custom models that need to be generated."""
        models = []

        # Get all custom types used
        custom_types = set()
        for agent in spec.agents:
            for inp in agent.inputs:
                type_info = self.type_mapper.map_type(inp.type)
                if type_info.is_custom:
                    custom_types.add(inp.type)
            for out in agent.outputs:
                type_info = self.type_mapper.map_type(out.type)
                if type_info.is_custom:
                    custom_types.add(out.type)

        # Generate stub models for custom types
        for type_name in sorted(custom_types):
            models.append(CustomModelDefinition(
                name=type_name,
                description=f"Custom model for {type_name}. TODO: Define fields.",
                fields=[],
            ))

        return models

    def _schema_property_to_type(self, prop_def: Dict[str, Any]) -> str:
        """Convert a JSON Schema property to Python type."""
        if not isinstance(prop_def, dict):
            return "Any"

        schema_type = prop_def.get("type", "any")

        # Handle enum
        if "enum" in prop_def:
            return "str"  # Could be improved to use actual enum

        # Handle const
        if "const" in prop_def:
            return "str"

        # Handle type mapping
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "object": "Dict[str, Any]",
            "array": "List[Any]",
            "null": "None",
        }

        if schema_type in type_mapping:
            return type_mapping[schema_type]

        return "Any"

    @staticmethod
    def _to_pascal_case(s: str) -> str:
        """Convert string to PascalCase."""
        parts = re.split(r"[-_\s]", s)
        return "".join(word.capitalize() for word in parts)

    @staticmethod
    def _to_snake_case(s: Any) -> str:
        """Convert string to snake_case."""
        # Handle non-string values
        s = str(s)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.replace("-", "_").lower()


# =============================================================================
# Unit Test Stubs
# =============================================================================

def _test_model_generator():
    """
    Unit test stub for ModelGenerator.

    Run with: pytest backend/agent_generator/generators/model_gen.py
    """
    from backend.agent_generator.parser.yaml_parser import (
        AgentSpec,
        PackSpec,
        AgentDefinition,
        InputDefinition,
        OutputDefinition,
    )

    # Create test spec
    spec = AgentSpec(
        pack=PackSpec(
            id="test-models-v1",
            name="Test Models Pack",
            version="1.0.0",
            description="Test pack for model generation",
        ),
        agents=[
            AgentDefinition(
                id="test-agent",
                name="Test Agent",
                type="deterministic-calculator",
                description="Test agent",
                inputs=[
                    InputDefinition(
                        name="value",
                        type="float",
                        required=True,
                        description="Input value",
                    ),
                    InputDefinition(
                        name="category",
                        type="str",
                        required=False,
                        default="default",
                        description="Category",
                    ),
                ],
                outputs=[
                    OutputDefinition(
                        name="result",
                        type="float",
                        description="Result value",
                    ),
                ],
            ),
        ],
        tools=[],
    )

    # Generate models
    generator = ModelGenerator()
    code = generator.generate_models(spec)

    # Verify code
    assert "class TestAgentInput(BaseModel):" in code
    assert "class TestAgentOutput(BaseModel):" in code
    assert "value: float" in code
    assert "provenance_hash" in code

    print(f"Generated model code length: {len(code)} characters")
    print("ModelGenerator tests passed!")


if __name__ == "__main__":
    _test_model_generator()
