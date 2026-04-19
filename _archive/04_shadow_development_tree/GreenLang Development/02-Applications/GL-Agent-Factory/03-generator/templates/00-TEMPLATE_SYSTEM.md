# Agent Generator Template System

**Version**: 1.0.0
**Status**: Design
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Executive Summary

The Template System is the code generation engine that transforms AgentSpec v2 YAML into production-ready Python code. It uses Jinja2 templates with custom filters and a hierarchical template inheritance system to generate type-safe, tested, documented agent packs.

**Key Features**:
- **Template Engine**: Jinja2 with custom filters for GreenLang patterns
- **Template Hierarchy**: Base templates with specialization for agent types
- **Type Safety**: All templates generate type-hinted code
- **Zero Hallucination**: Templates enforce deterministic calculator usage
- **Production Quality**: Generated code passes Ruff, Mypy, 85%+ test coverage

---

## 1. Template System Architecture

### 1.1 Template Directory Structure

```
templates/
├── _macros/                          # Reusable template macros
│   ├── imports.jinja2                # Import statement macros
│   ├── pydantic_models.jinja2        # Pydantic model generation
│   ├── validators.jinja2             # Validator function macros
│   ├── docstrings.jinja2             # Docstring formatting
│   └── type_hints.jinja2             # Type hint formatting
│
├── agent_class/                      # Agent class templates
│   ├── base_agent.py.jinja2          # Base agent template
│   ├── calculator_agent.py.jinja2   # Pure calculator agent
│   ├── llm_agent.py.jinja2           # LLM-powered agent
│   ├── orchestrator_agent.py.jinja2 # Multi-agent orchestrator
│   └── hybrid_agent.py.jinja2        # LLM + calculator hybrid
│
├── tools/                            # Tool wrapper templates
│   ├── calculator_tool.py.jinja2     # Deterministic calculator tool
│   ├── llm_tool.py.jinja2            # LLM-based tool
│   ├── validation_tool.py.jinja2    # Validation tool
│   └── integration_tool.py.jinja2   # External integration tool
│
├── tests/                            # Test suite templates
│   ├── test_agent.py.jinja2          # Agent unit tests
│   ├── test_tools.py.jinja2          # Tool unit tests
│   ├── test_integration.py.jinja2   # Integration tests
│   ├── test_determinism.py.jinja2   # Determinism tests
│   ├── test_performance.py.jinja2   # Performance benchmarks
│   └── fixtures.yaml.jinja2          # Test fixtures
│
├── graph/                            # LangGraph configuration
│   ├── agent_graph.yaml.jinja2       # Main graph config
│   ├── simple_graph.yaml.jinja2     # Simple linear graph
│   └── orchestrator_graph.yaml.jinja2 # Complex orchestration
│
├── deployment/                       # Deployment configs
│   ├── Dockerfile.jinja2             # Docker container
│   ├── k8s_deployment.yaml.jinja2   # Kubernetes deployment
│   ├── k8s_service.yaml.jinja2      # Kubernetes service
│   ├── k8s_configmap.yaml.jinja2    # ConfigMap
│   └── docker-compose.yaml.jinja2   # Docker Compose
│
├── monitoring/                       # Monitoring configs
│   ├── grafana_dashboard.json.jinja2 # Grafana dashboard
│   ├── prometheus_alerts.yaml.jinja2 # Prometheus alerts
│   └── metrics.py.jinja2             # Metrics collection
│
├── docs/                             # Documentation templates
│   ├── README.md.jinja2              # Main README
│   ├── ARCHITECTURE.md.jinja2        # Architecture doc
│   ├── API.md.jinja2                 # API reference
│   ├── TOOLS.md.jinja2               # Tool specifications
│   └── EXAMPLES.md.jinja2            # Usage examples
│
└── config/                           # Configuration templates
    ├── config.py.jinja2              # Agent configuration
    ├── prompts.py.jinja2             # Prompt templates
    ├── requirements.txt.jinja2       # Python dependencies
    ├── pytest.ini.jinja2             # Pytest config
    └── .env.template.jinja2          # Environment variables
```

### 1.2 Template Engine Configuration

```python
# template_engine.py

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from typing import Dict, Any, List
import re


class GreenLangTemplateEngine:
    """
    Template engine for GreenLang agent generation.

    Features:
    - Jinja2 with custom filters for GreenLang patterns
    - Template inheritance and composition
    - Type-safe code generation
    - Automatic formatting and linting
    """

    def __init__(self, template_dir: Path):
        """
        Initialize template engine.

        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Register custom filters
        self._register_filters()

        # Register custom tests
        self._register_tests()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self.env.filters['dtype_to_python'] = self.dtype_to_python_type
        self.env.filters['snake_case'] = self.to_snake_case
        self.env.filters['pascal_case'] = self.to_pascal_case
        self.env.filters['camel_case'] = self.to_camel_case
        self.env.filters['format_docstring'] = self.format_docstring
        self.env.filters['format_import'] = self.format_import
        self.env.filters['constraint_to_pydantic'] = self.constraint_to_pydantic
        self.env.filters['safe_identifier'] = self.safe_identifier
        self.env.filters['extract_module'] = self.extract_module
        self.env.filters['extract_function'] = self.extract_function

    def _register_tests(self) -> None:
        """Register custom Jinja2 tests."""
        self.env.tests['calculator_tool'] = lambda tool: tool.get('safety') == 'deterministic'
        self.env.tests['llm_tool'] = lambda tool: tool.get('safety') != 'deterministic'
        self.env.tests['required_field'] = lambda field: field.get('required', False)

    # Custom Filters Implementation

    @staticmethod
    def dtype_to_python_type(dtype: str) -> str:
        """
        Convert AgentSpec dtype to Python type hint.

        Args:
            dtype: AgentSpec data type

        Returns:
            Python type hint string
        """
        mapping = {
            "string": "str",
            "float64": "float",
            "float32": "float",
            "int64": "int",
            "int32": "int",
            "bool": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
            "datetime": "datetime"
        }
        return mapping.get(dtype, "Any")

    @staticmethod
    def to_snake_case(text: str) -> str:
        """Convert text to snake_case."""
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)
        return text.lower()

    @staticmethod
    def to_pascal_case(text: str) -> str:
        """Convert text to PascalCase."""
        words = re.split(r'[_\s-]+', text)
        return ''.join(word.capitalize() for word in words)

    @staticmethod
    def to_camel_case(text: str) -> str:
        """Convert text to camelCase."""
        pascal = GreenLangTemplateEngine.to_pascal_case(text)
        return pascal[0].lower() + pascal[1:] if pascal else ""

    @staticmethod
    def format_docstring(text: str, indent: int = 4) -> str:
        """
        Format text as Python docstring.

        Args:
            text: Docstring text
            indent: Indentation level in spaces

        Returns:
            Formatted docstring
        """
        lines = text.strip().split('\n')
        indent_str = ' ' * indent
        formatted_lines = [f'{indent_str}{line}' for line in lines]
        return '\n'.join(formatted_lines)

    @staticmethod
    def format_import(impl_path: str) -> Dict[str, str]:
        """
        Parse python:// implementation path.

        Args:
            impl_path: Path like "python://module.path:function_name"

        Returns:
            Dict with 'module' and 'function' keys
        """
        if not impl_path.startswith('python://'):
            raise ValueError(f"Invalid python:// path: {impl_path}")

        path = impl_path.replace('python://', '')
        if ':' not in path:
            raise ValueError(f"Missing ':' in python:// path: {impl_path}")

        module_path, function_name = path.split(':', 1)
        return {
            'module': module_path,
            'function': function_name
        }

    @staticmethod
    def constraint_to_pydantic(constraints: Dict[str, Any]) -> str:
        """
        Convert AgentSpec constraints to Pydantic Field parameters.

        Args:
            constraints: Constraint dictionary

        Returns:
            Pydantic Field parameter string
        """
        params = []

        if 'ge' in constraints:
            params.append(f"ge={constraints['ge']}")
        if 'le' in constraints:
            params.append(f"le={constraints['le']}")
        if 'gt' in constraints:
            params.append(f"gt={constraints['gt']}")
        if 'lt' in constraints:
            params.append(f"lt={constraints['lt']}")
        if 'min_length' in constraints:
            params.append(f"min_length={constraints['min_length']}")
        if 'max_length' in constraints:
            params.append(f"max_length={constraints['max_length']}")
        if 'pattern' in constraints:
            params.append(f'regex=r"{constraints["pattern"]}"')

        return ', '.join(params)

    @staticmethod
    def safe_identifier(text: str) -> str:
        """
        Convert text to safe Python identifier.

        Args:
            text: Input text

        Returns:
            Safe identifier
        """
        # Replace invalid chars with underscore
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', text)

        # Ensure doesn't start with digit
        if safe[0].isdigit():
            safe = f'_{safe}'

        return safe

    @staticmethod
    def extract_module(impl_path: str) -> str:
        """Extract module path from python:// URI."""
        parsed = GreenLangTemplateEngine.format_import(impl_path)
        return parsed['module']

    @staticmethod
    def extract_function(impl_path: str) -> str:
        """Extract function name from python:// URI."""
        parsed = GreenLangTemplateEngine.format_import(impl_path)
        return parsed['function']

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render template with context.

        Args:
            template_name: Template file name
            context: Template context dictionary

        Returns:
            Rendered template string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
```

---

## 2. Base Agent Template

### 2.1 Base Agent Class Template

```jinja2
{# templates/agent_class/base_agent.py.jinja2 #}
"""
{{ spec.name }}

{{ spec.summary }}

Generated from AgentSpec v2 on {{ timestamp }}
Agent ID: {{ spec.id }}
Version: {{ spec.version }}
License: {{ spec.metadata.license }}
Owners: {{ spec.metadata.owners | join(', ') }}
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime

from greenlang_core import BaseAgent, AgentConfig
from greenlang_validation import ValidationResult
from greenlang_provenance import ProvenanceTracker

{% if spec.ai.tools %}
# Import tools from AgentSpec
{% for tool in spec.ai.tools %}
from {{ tool.impl | extract_module }} import {{ tool.impl | extract_function }}
{% endfor %}
{% endif %}

logger = logging.getLogger(__name__)


{# Generate Input Model #}
class {{ agent_name }}Input(BaseModel):
    """Input data model for {{ agent_name }}."""

{% for field_name, field_spec in spec.compute.inputs.items() %}
    {{ field_name }}: {{ field_spec.dtype | dtype_to_python }} = Field(
        {% if field_spec.required %}...,{% else %}default={{ field_spec.default | tojson }},{% endif %}
        description="{{ field_spec.description }}"
        {% if field_spec.constraints %}
        {{ field_spec.constraints | constraint_to_pydantic }}
        {% endif %}
    )
{% endfor %}

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"


{# Generate Output Model #}
class {{ agent_name }}Output(BaseModel):
    """Output data model for {{ agent_name }}."""

{% for field_name, field_spec in spec.compute.outputs.items() %}
    {{ field_name }}: {{ field_spec.dtype | dtype_to_python }} = Field(
        ...,
        description="{{ field_spec.description }}"
    )
{% endfor %}

    # Auto-added provenance fields
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )

    processing_time_ms: float = Field(
        ...,
        description="Processing duration in milliseconds"
    )

    agent_id: str = Field(
        default="{{ spec.id }}",
        description="Agent identifier"
    )

    agent_version: str = Field(
        default="{{ spec.version }}",
        description="Agent version"
    )


{# Generate Agent Class #}
class {{ agent_name }}(BaseAgent):
    """
    {{ spec.name }}.

    {{ spec.summary }}

    Generated from AgentSpec v2: {{ spec.id }}
    Deterministic: {{ spec.compute.deterministic }}
    Zero Hallucination: {{ spec.provenance.zero_hallucination | default(true) }}

    Attributes:
        AGENT_ID: Agent identifier from spec
        AGENT_VERSION: Agent version from spec
        config: Agent configuration
        provenance_tracker: Provenance tracking system
    """

    # AgentSpec metadata
    AGENT_ID = "{{ spec.id }}"
    AGENT_VERSION = "{{ spec.version }}"
    SCHEMA_VERSION = "{{ spec.schema_version }}"
    TAGS = {{ spec.metadata.tags | tojson }}
    OWNERS = {{ spec.metadata.owners | tojson }}
    LICENSE = "{{ spec.metadata.license }}"

    def __init__(self, config: AgentConfig):
        """
        Initialize {{ agent_name }}.

        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self.provenance_tracker = ProvenanceTracker()

        {% if spec.compute.factors %}
        # Load emission factors
        self.emission_factors = self._load_emission_factors()
        {% endif %}

        logger.info(
            f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}",
            extra={"agent_id": self.AGENT_ID}
        )

    def process(self, input_data: {{ agent_name }}Input) -> {{ agent_name }}Output:
        """
        Main processing method.

        Args:
            input_data: Validated input data

        Returns:
            Processed output with provenance hash

        Raises:
            ValueError: If input data fails validation
            ProcessingError: If processing fails
        """
        start_time = datetime.now()

        try:
            # Step 1: Validate input
            logger.debug("Validating input")
            validation_result = self._validate_input(input_data)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Input validation failed: {validation_result.errors}"
                )

            # Step 2: Process data (ZERO HALLUCINATION)
            logger.debug("Processing data")
            processed_data = self._process_core_logic(input_data)

            # Step 3: Validate output
            logger.debug("Validating output")
            output_validation = self._validate_output(processed_data)
            if not output_validation.is_valid:
                raise ValueError(
                    f"Output validation failed: {output_validation.errors}"
                )

            # Step 4: Calculate provenance hash
            logger.debug("Calculating provenance")
            provenance_hash = self._calculate_provenance(
                input_data, processed_data
            )

            # Step 5: Create output
            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            output = {{ agent_name }}Output(
                **processed_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                agent_id=self.AGENT_ID,
                agent_version=self.AGENT_VERSION
            )

            logger.info(
                f"Processing complete: {processing_time:.2f}ms",
                extra={
                    "agent_id": self.AGENT_ID,
                    "processing_time_ms": processing_time
                }
            )

            return output

        except Exception as e:
            logger.error(
                f"{self.AGENT_ID} processing failed: {str(e)}",
                exc_info=True
            )
            raise

    def _process_core_logic(
        self,
        input_data: {{ agent_name }}Input
    ) -> Dict[str, Any]:
        """
        Core processing logic - ZERO HALLUCINATION.

        This method implements deterministic processing only.
        Uses calculators from:
        {% for tool in spec.ai.tools %}
        - {{ tool.name }}: {{ tool.impl }}
        {% endfor %}

        Args:
            input_data: Validated input data

        Returns:
            Processing result dictionary
        """
        # TODO: Implement using GreenLang calculators
        # Example:
        {% if spec.ai.tools and spec.ai.tools[0] is calculator_tool %}
        # result = {{ spec.ai.tools[0].impl | extract_function }}(
        #     {% for field_name in spec.compute.inputs.keys() %}
        #     {{ field_name }}=input_data.{{ field_name }},
        #     {% endfor %}
        # )
        {% endif %}

        raise NotImplementedError(
            "Implement using GreenLang calculators (zero-hallucination)"
        )

    def _validate_input(
        self,
        input_data: {{ agent_name }}Input
    ) -> ValidationResult:
        """
        Validate input data meets all requirements.

        Args:
            input_data: Input data to validate

        Returns:
            Validation result
        """
        # Pydantic handles basic validation
        # Add custom validation here if needed
        return ValidationResult(is_valid=True, errors=[])

    def _validate_output(
        self,
        output_data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate output meets all requirements.

        Args:
            output_data: Output data to validate

        Returns:
            Validation result
        """
        # Check all required outputs are present
        required_outputs = {
            {% for field_name in spec.compute.outputs.keys() %}
            "{{ field_name }}",
            {% endfor %}
        }

        errors = []
        for field in required_outputs:
            if field not in output_data:
                errors.append(f"Required output '{field}' missing")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

    def _calculate_provenance(
        self,
        input_data: {{ agent_name }}Input,
        output_data: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        Args:
            input_data: Input data
            output_data: Output data

        Returns:
            SHA-256 hash string
        """
        import json

        provenance_str = (
            f"{input_data.json()}"
            f"{json.dumps(output_data, sort_keys=True)}"
            f"{self.AGENT_ID}"
            f"{self.AGENT_VERSION}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    {% if spec.compute.factors %}
    def _load_emission_factors(self) -> Dict[str, Any]:
        """
        Load emission factors from database.

        References:
        {% for factor in spec.compute.factors %}
        - {{ factor.ref }} (GWP: {{ factor.gwp_set }})
        {% endfor %}

        Returns:
            Emission factors dictionary
        """
        # TODO: Implement emission factor loading
        from greenlang.emission_factors import EmissionFactorRegistry

        registry = EmissionFactorRegistry()
        factors = {}

        {% for factor in spec.compute.factors %}
        factors["{{ factor.ref }}"] = registry.get(
            ref="{{ factor.ref }}",
            gwp_set="{{ factor.gwp_set }}"
        )
        {% endfor %}

        return factors
    {% endif %}


{# Generate entrypoint function #}
def compute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for {{ agent_name }}.

    Defined in AgentSpec: {{ spec.compute.entrypoint }}

    Args:
        inputs: Input dictionary matching {{ agent_name }}Input schema

    Returns:
        Output dictionary matching {{ agent_name }}Output schema
    """
    config = AgentConfig(
        deterministic={{ spec.compute.deterministic }},
        timeout_seconds={{ spec.compute.timeout_seconds | default(30) }}
    )
    agent = {{ agent_name }}(config)
    input_model = {{ agent_name }}Input(**inputs)
    result = agent.process(input_model)
    return result.dict()
```

---

## 3. Calculator Tool Template

### 3.1 Deterministic Calculator Tool

```jinja2
{# templates/tools/calculator_tool.py.jinja2 #}
"""
{{ tool.name }} - Deterministic Calculator Tool

SAFETY: deterministic (zero-hallucination)
Implementation: {{ tool.impl }}
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, validator

from {{ tool.impl | extract_module }} import {{ tool.impl | extract_function }}


{# Generate input schema #}
class {{ tool.name | pascal_case }}Input(BaseModel):
    """Input schema for {{ tool.name }} tool."""

{% for prop_name, prop_spec in tool.schema_in.properties.items() %}
    {{ prop_name }}: {{ prop_spec.type | dtype_to_python }} = Field(
        {% if prop_name in tool.schema_in.required %}...,{% else %}default=None,{% endif %}
        description="{{ prop_spec.description | default('') }}"
    )
{% endfor %}


{# Generate output schema #}
class {{ tool.name | pascal_case }}Output(BaseModel):
    """Output schema for {{ tool.name }} tool."""

{% for prop_name, prop_spec in tool.schema_out.properties.items() %}
    {{ prop_name }}: {{ prop_spec.type | dtype_to_python }} = Field(
        ...,
        description="{{ prop_spec.description | default('') }}"
    )
{% endfor %}


def {{ tool.name | snake_case }}_tool(
    {% for prop_name in tool.schema_in.required %}
    {{ prop_name }}: {{ tool.schema_in.properties[prop_name].type | dtype_to_python }},
    {% endfor %}
    {% for prop_name in tool.schema_in.properties.keys() %}
    {% if prop_name not in tool.schema_in.required %}
    {{ prop_name }}: {{ tool.schema_in.properties[prop_name].type | dtype_to_python }} = None,
    {% endif %}
    {% endfor %}
) -> {{ tool.name | pascal_case }}Output:
    """
    {{ tool.description | default(tool.name) }}

    SAFETY: {{ tool.safety }}
    Implementation: {{ tool.impl }}

    Args:
    {% for prop_name, prop_spec in tool.schema_in.properties.items() %}
        {{ prop_name }}: {{ prop_spec.description | default('') }}
    {% endfor %}

    Returns:
        {{ tool.name }} calculation result

    Raises:
        ValueError: If input validation fails
        CalculationError: If calculation fails
    """
    # Validate input
    input_data = {{ tool.name | pascal_case }}Input(
        {% for prop_name in tool.schema_in.properties.keys() %}
        {{ prop_name }}={{ prop_name }},
        {% endfor %}
    )

    # Call GreenLang calculator (ZERO HALLUCINATION)
    result = {{ tool.impl | extract_function }}(
        {% for prop_name in tool.schema_in.required %}
        {{ prop_name }}=input_data.{{ prop_name }},
        {% endfor %}
    )

    # Validate output
    output = {{ tool.name | pascal_case }}Output(
        {% for prop_name in tool.schema_out.required %}
        {{ prop_name }}=result.{{ prop_name }},
        {% endfor %}
    )

    return output


# LLM tool schema (for Anthropic API)
{{ tool.name | snake_case | upper }}_TOOL_SCHEMA = {
    "name": "{{ tool.name }}",
    "description": "{{ tool.description | default(tool.name) }}",
    "input_schema": {
        "type": "object",
        "required": {{ tool.schema_in.required | tojson }},
        "properties": {
        {% for prop_name, prop_spec in tool.schema_in.properties.items() %}
            "{{ prop_name }}": {
                "type": "{{ prop_spec.type }}",
                {% if prop_spec.description %}
                "description": "{{ prop_spec.description }}",
                {% endif %}
                {% if prop_spec.enum %}
                "enum": {{ prop_spec.enum | tojson }},
                {% endif %}
            },
        {% endfor %}
        }
    }
}
```

---

## 4. Test Template

### 4.1 Agent Unit Test Template

```jinja2
{# templates/tests/test_agent.py.jinja2 #}
"""
Unit tests for {{ agent_name }}.

Auto-generated from AgentSpec v2 on {{ timestamp }}
Agent ID: {{ spec.id }}
"""

import pytest
from pathlib import Path
from datetime import datetime

from {{ module_name }} import (
    {{ agent_name }},
    {{ agent_name }}Input,
    {{ agent_name }}Output,
    AgentConfig
)


class Test{{ agent_name }}:
    """Test suite for {{ agent_name }}."""

    @pytest.fixture
    def config(self):
        """Agent configuration fixture."""
        return AgentConfig(
            deterministic={{ spec.compute.deterministic }},
            seed=42,
            timeout_seconds={{ spec.compute.timeout_seconds | default(30) }}
        )

    @pytest.fixture
    def agent(self, config):
        """Agent instance fixture."""
        return {{ agent_name }}(config)

    @pytest.fixture
    def valid_input(self):
        """Valid input fixture."""
        return {{ agent_name }}Input(
        {% for field_name, field_spec in spec.compute.inputs.items() %}
            {% if field_spec.required %}
            {{ field_name }}={{ field_spec.example | default('None') | tojson }},
            {% endif %}
        {% endfor %}
        )

    # Basic Functionality Tests

    def test_agent_initialization(self, config):
        """Test that agent initializes correctly."""
        agent = {{ agent_name }}(config)

        assert agent.AGENT_ID == "{{ spec.id }}"
        assert agent.AGENT_VERSION == "{{ spec.version }}"
        assert agent.config == config

    def test_valid_input_processes_successfully(self, agent, valid_input):
        """Test that valid input processes successfully."""
        result = agent.process(valid_input)

        assert isinstance(result, {{ agent_name }}Output)
        {% for field_name in spec.compute.outputs.keys() %}
        assert hasattr(result, '{{ field_name }}')
        {% endfor %}
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0

    # Input Validation Tests

    {% for field_name, field_spec in spec.compute.inputs.items() %}
    {% if field_spec.required %}
    def test_missing_{{ field_name }}_raises_error(self):
        """Test that missing {{ field_name }} raises validation error."""
        with pytest.raises(ValueError, match="{{ field_name }}"):
            {{ agent_name }}Input(
                # Missing {{ field_name }} (required field)
                {% for other_field, other_spec in spec.compute.inputs.items() %}
                {% if other_field != field_name and other_spec.required %}
                {{ other_field }}={{ other_spec.example | default('None') | tojson }},
                {% endif %}
                {% endfor %}
            )
    {% endif %}
    {% endfor %}

    {% for field_name, field_spec in spec.compute.inputs.items() %}
    {% if field_spec.constraints and field_spec.constraints.ge is defined %}
    def test_{{ field_name }}_below_minimum_raises_error(self):
        """Test that {{ field_name }} below minimum raises error."""
        with pytest.raises(ValueError):
            {{ agent_name }}Input(
                {% for other_field, other_spec in spec.compute.inputs.items() %}
                {% if other_field == field_name %}
                {{ field_name }}={{ field_spec.constraints.ge - 1 }},  # Below minimum
                {% elif other_spec.required %}
                {{ other_field }}={{ other_spec.example | default('None') | tojson }},
                {% endif %}
                {% endfor %}
            )
    {% endif %}
    {% endfor %}

    # Determinism Tests
    {% if spec.compute.deterministic %}

    def test_deterministic_execution(self, agent, valid_input):
        """Test that execution is deterministic."""
        result1 = agent.process(valid_input)
        result2 = agent.process(valid_input)

        {% for field_name in spec.compute.outputs.keys() %}
        assert result1.{{ field_name }} == result2.{{ field_name }}
        {% endfor %}
        assert result1.provenance_hash == result2.provenance_hash

    def test_same_input_produces_same_hash(self, agent, valid_input):
        """Test that same input produces same provenance hash."""
        result1 = agent.process(valid_input)
        result2 = agent.process(valid_input)

        assert result1.provenance_hash == result2.provenance_hash
    {% endif %}

    # Provenance Tests

    def test_provenance_hash_is_sha256(self, agent, valid_input):
        """Test that provenance hash is valid SHA-256."""
        result = agent.process(valid_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex digest
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_output_includes_agent_metadata(self, agent, valid_input):
        """Test that output includes agent metadata."""
        result = agent.process(valid_input)

        assert result.agent_id == "{{ spec.id }}"
        assert result.agent_version == "{{ spec.version }}"

    # Output Validation Tests

    def test_output_schema_compliance(self, agent, valid_input):
        """Test that output complies with schema."""
        result = agent.process(valid_input)

        # All required outputs present
        {% for field_name, field_spec in spec.compute.outputs.items() %}
        assert hasattr(result, '{{ field_name }}')
        {% endfor %}

        # Correct types
        {% for field_name, field_spec in spec.compute.outputs.items() %}
        assert isinstance(
            result.{{ field_name }},
            {{ field_spec.dtype | dtype_to_python }}
        )
        {% endfor %}

    # Performance Tests

    def test_processing_time_within_timeout(self, agent, valid_input):
        """Test that processing completes within timeout."""
        result = agent.process(valid_input)

        assert result.processing_time_ms < {{ spec.compute.timeout_seconds | default(30) }} * 1000
```

---

## 5. Documentation Templates

### 5.1 README Template

```jinja2
{# templates/docs/README.md.jinja2 #}
# {{ spec.name }}

{{ spec.summary }}

**Agent ID**: `{{ spec.id }}`
**Version**: `{{ spec.version }}`
**License**: {{ spec.metadata.license }}
**Owners**: {{ spec.metadata.owners | join(', ') }}
**Tags**: {{ spec.metadata.tags | join(', ') }}

---

## Overview

{{ spec.summary }}

This agent was auto-generated from AgentSpec v2 on {{ timestamp }}.

### Features

- **Deterministic**: {% if spec.compute.deterministic %}Guaranteed reproducible results (temperature=0.0, seed=42){% else %}Non-deterministic execution{% endif %}
- **Zero Hallucination**: Uses GreenLang calculators for all numeric calculations
- **Complete Provenance**: Full audit trail with SHA-256 hashing
- **Type Safe**: Pydantic models with comprehensive validation
- **Production Ready**: 85%+ test coverage, documented, monitored

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from {{ module_name }} import {{ agent_name }}, {{ agent_name }}Input, AgentConfig

# Create agent
config = AgentConfig(deterministic={{ spec.compute.deterministic }})
agent = {{ agent_name }}(config)

# Process input
input_data = {{ agent_name }}Input(
{% for field_name, field_spec in spec.compute.inputs.items() %}
    {% if field_spec.required %}
    {{ field_name }}={{ field_spec.example | default('None') | tojson }},
    {% endif %}
{% endfor %}
)

result = agent.process(input_data)
{% if spec.compute.outputs %}
print(f"Result: {result.{{ spec.compute.outputs.keys() | first }}}")
{% endif %}
```

---

## API Reference

### Input Schema

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
{% for field_name, field_spec in spec.compute.inputs.items() %}
| `{{ field_name }}` | {{ field_spec.dtype }} | {% if field_spec.required %}Yes{% else %}No{% endif %} | {{ field_spec.description }} | {% if field_spec.constraints %}{{ field_spec.constraints }}{% else %}-{% endif %} |
{% endfor %}

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
{% for field_name, field_spec in spec.compute.outputs.items() %}
| `{{ field_name }}` | {{ field_spec.dtype }} | {{ field_spec.description }} |
{% endfor %}

---

## Tools

{% for tool in spec.ai.tools %}
### {{ tool.name }}

{{ tool.description | default('No description provided') }}

**Safety**: {{ tool.safety }}
**Implementation**: `{{ tool.impl }}`

{% endfor %}

---

## Testing

```bash
pytest tests/ -v
```

---

## Deployment

See [deployment/](deployment/) for:
- Kubernetes manifests
- Docker configuration
- Monitoring setup

---

## Documentation

- [Architecture](ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Tool Specifications](docs/TOOLS.md)
- [Examples](docs/EXAMPLES.md)

---

## License

{{ spec.metadata.license }}

---

## Support

- Owners: {{ spec.metadata.owners | join(', ') }}
- Issues: [GitHub Issues](https://github.com/greenlang/agents/issues)

---

**Auto-generated from AgentSpec v2** on {{ timestamp }}
```

---

## 6. Template Context Builder

### 6.1 Context Generation

```python
# context_builder.py

from typing import Dict, Any
from datetime import datetime
from pathlib import Path


class TemplateContextBuilder:
    """Build template context from AgentSpec."""

    def build_context(self, spec: AgentSpec) -> Dict[str, Any]:
        """
        Build complete template context.

        Args:
            spec: AgentSpec v2 model

        Returns:
            Template context dictionary
        """
        agent_name = self._generate_agent_name(spec)
        module_name = self._generate_module_name(spec)

        return {
            # Spec reference
            "spec": spec,

            # Naming
            "agent_name": agent_name,
            "agent_class": f"{agent_name}",
            "module_name": module_name,

            # Metadata
            "timestamp": datetime.now().isoformat(),
            "generator_version": "1.0.0",

            # Convenience accessors
            "inputs": spec.compute.inputs,
            "outputs": spec.compute.outputs,
            "tools": spec.ai.tools if spec.ai else [],
            "deterministic": spec.compute.deterministic,
            "zero_hallucination": spec.provenance.zero_hallucination if spec.provenance else True,

            # Paths
            "pack_dir": self._generate_pack_dir(spec),
        }

    def _generate_agent_name(self, spec: AgentSpec) -> str:
        """Generate agent class name from spec."""
        # Convert "emissions/fuel_agent_v1" -> "FuelAgent"
        name_part = spec.id.split('/')[-1]
        name_part = name_part.replace('_v1', '').replace('_v2', '')
        return ''.join(word.capitalize() for word in name_part.split('_'))

    def _generate_module_name(self, spec: AgentSpec) -> str:
        """Generate module name from spec."""
        # Convert "emissions/fuel_agent_v1" -> "fuel_agent"
        name_part = spec.id.split('/')[-1]
        return name_part.replace('_v1', '').replace('_v2', '')

    def _generate_pack_dir(self, spec: AgentSpec) -> Path:
        """Generate pack directory name."""
        return Path(f"packs/{self._generate_module_name(spec)}")
```

---

## 7. Template Selection Strategy

### 7.1 Agent Type Detection

```python
class TemplateSelector:
    """Select appropriate templates based on AgentSpec."""

    def select_agent_template(self, spec: AgentSpec) -> str:
        """
        Select agent class template based on spec.

        Returns:
            Template file name
        """
        if self._is_pure_calculator(spec):
            return "agent_class/calculator_agent.py.jinja2"
        elif self._is_llm_agent(spec):
            return "agent_class/llm_agent.py.jinja2"
        elif self._is_orchestrator(spec):
            return "agent_class/orchestrator_agent.py.jinja2"
        elif self._is_hybrid(spec):
            return "agent_class/hybrid_agent.py.jinja2"
        else:
            return "agent_class/base_agent.py.jinja2"

    def _is_pure_calculator(self, spec: AgentSpec) -> bool:
        """Check if agent uses only deterministic calculators."""
        if not spec.ai or not spec.ai.tools:
            return False
        return all(
            tool.safety == "deterministic"
            for tool in spec.ai.tools
        )

    def _is_llm_agent(self, spec: AgentSpec) -> bool:
        """Check if agent is LLM-powered."""
        return spec.ai is not None and spec.ai.system_prompt is not None

    def _is_orchestrator(self, spec: AgentSpec) -> bool:
        """Check if agent orchestrates other agents."""
        # Check for orchestration patterns in spec
        return "orchestrator" in spec.name.lower()

    def _is_hybrid(self, spec: AgentSpec) -> bool:
        """Check if agent combines LLM and calculators."""
        if not spec.ai or not spec.ai.tools:
            return False
        has_deterministic = any(
            tool.safety == "deterministic"
            for tool in spec.ai.tools
        )
        has_llm = spec.ai.system_prompt is not None
        return has_deterministic and has_llm
```

---

## 8. Template Validation

### 8.1 Post-Generation Validation

```python
class GeneratedCodeValidator:
    """Validate generated code quality."""

    def validate(self, generated_code: str) -> ValidationResult:
        """
        Validate generated code.

        Checks:
        - Valid Python syntax
        - Type hints present
        - Docstrings present
        - No security issues
        """
        errors = []

        # Syntax check
        try:
            compile(generated_code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Type hint check
        if not self._has_type_hints(generated_code):
            errors.append("Missing type hints")

        # Docstring check
        if not self._has_docstrings(generated_code):
            errors.append("Missing docstrings")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
```

---

## Summary

The Template System provides:

1. **Jinja2 Templates**: Type-safe code generation with custom filters
2. **Template Hierarchy**: Specialized templates for different agent types
3. **Context Builder**: Rich context from AgentSpec for templates
4. **Template Selection**: Automatic selection based on agent type
5. **Validation**: Post-generation code quality checks

**Next Step**: Implement generation workflows and CLI commands.

---

**Document Status**: Design Complete
**Implementation Status**: Pending
