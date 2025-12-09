# Agent Generator Architecture Specification

## 1. Executive Summary

The Agent Generator is a code generation system that transforms AgentSpec YAML definitions (pack.yaml) into production-ready Python agents. It ensures:

- **Zero-Hallucination Compliance**: All generated calculations use deterministic lookup tables, not LLM inference
- **Complete Provenance Tracking**: SHA-256 hashing of all calculation steps
- **GreenLang Pattern Conformance**: Generated code matches existing hand-written agents
- **Production Readiness**: Type hints, docstrings, validation, and test generation

### Key Metrics
- **Target Generation Time**: <5 seconds per agent
- **Test Coverage**: 85%+ for generated code
- **Type Safety**: 100% type annotations
- **Zero Manual Editing Required**: Generated agents are immediately deployable

---

## 2. System Architecture Overview

```
                                    Agent Generator System
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   +-----------------+     +------------------+     +-------------------+          |
|   |                 |     |                  |     |                   |          |
|   |   pack.yaml     +---->+   YAML Parser    +---->+   Spec Validator  |          |
|   |   (AgentSpec)   |     |                  |     |                   |          |
|   +-----------------+     +------------------+     +---------+---------+          |
|                                                              |                    |
|                                                              v                    |
|   +----------------------------------------------------------+----------------+   |
|   |                                                                           |   |
|   |                        Generation Engine (Orchestrator)                   |   |
|   |                                                                           |   |
|   +----+---------------+---------------+---------------+---------------+------+   |
|        |               |               |               |               |          |
|        v               v               v               v               v          |
|   +---------+    +---------+    +---------+    +---------+    +---------+        |
|   | Agent   |    | Model   |    | Tool    |    | Test    |    | Config  |        |
|   | Gen     |    | Gen     |    | Gen     |    | Gen     |    | Gen     |        |
|   +---------+    +---------+    +---------+    +---------+    +---------+        |
|        |               |               |               |               |          |
|        v               v               v               v               v          |
|   +---------+    +---------+    +---------+    +---------+    +---------+        |
|   | agent   |    | models  |    | tools   |    | tests   |    | config  |        |
|   | .py     |    | .py     |    | .py     |    | .py     |    | .yaml   |        |
|   +---------+    +---------+    +---------+    +---------+    +---------+        |
|                                                                                   |
|   +---------------------------+    +---------------------------+                  |
|   |   Jinja2 Templates        |    |   BaseCalculator/         |                  |
|   |   (agent.py.j2, etc.)     |    |   ProvenanceMixin         |                  |
|   +---------------------------+    +---------------------------+                  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Module Structure

```
backend/agent_generator/
|-- __init__.py                 # Public API exports
|-- engine.py                   # Core generation orchestrator
|-- config.py                   # Generator configuration
|
|-- parser/
|   |-- __init__.py             # Parser exports
|   |-- yaml_parser.py          # YAML loading with schema validation
|   |-- spec_validator.py       # AgentSpec validation rules
|   |-- schema.py               # Pydantic models for pack.yaml
|
|-- generators/
|   |-- __init__.py             # Generator exports
|   |-- base.py                 # Abstract base generator
|   |-- agent_gen.py            # Agent class generator
|   |-- model_gen.py            # Pydantic model generator
|   |-- tool_gen.py             # Tool wrapper generator
|   |-- test_gen.py             # Test file generator
|   |-- init_gen.py             # __init__.py generator
|
|-- templates/
|   |-- agent.py.j2             # Main agent class template
|   |-- models.py.j2            # Pydantic models template
|   |-- tools.py.j2             # Tool wrappers template
|   |-- test_agent.py.j2        # Pytest test template
|   |-- __init__.py.j2          # Package init template
|   |-- pack_spec.py.j2         # PACK_SPEC dictionary template
|
|-- validators/
|   |-- __init__.py             # Validator exports
|   |-- zero_hallucination.py   # Zero-hallucination rule checker
|   |-- provenance.py           # Provenance tracking validator
|   |-- compliance.py           # Regulatory compliance validator
|
|-- utils/
|   |-- __init__.py             # Utility exports
|   |-- naming.py               # Naming conventions (snake_case, PascalCase)
|   |-- type_mapping.py         # YAML type to Python type mapping
|   |-- hash_utils.py           # SHA-256 provenance utilities
```

---

## 4. Data Flow Diagram

```
                                   Data Flow

     +------------+
     | pack.yaml  |
     +-----+------+
           |
           | (1) Load YAML
           v
     +-----+------+
     | YAMLParser |----> Validates YAML syntax
     +-----+------+      Resolves file references
           |
           | (2) Parse to Schema
           v
     +-----+------+
     | AgentSpec  |----> Pydantic model with:
     | (Pydantic) |      - pack metadata
     +-----+------+      - agents list
           |             - tools list
           | (3) Validate              - validation rules
           v             - golden tests
     +-----+------+
     | Validator  |----> Checks:
     |            |      - Required fields present
     +-----+------+      - Types match schema
           |             - Zero-hallucination compliance
           | (4) Generate              - Tool references valid
           v
     +-----+------+
     | Generator  |----> Produces:
     |  Engine    |      - agent.py
     +-----+------+      - models.py
           |             - tools.py (if needed)
           | (5) Emit                  - test_agent.py
           v             - __init__.py
     +------------+
     | Output Dir |
     | gl_XXX_... |
     +------------+
```

---

## 5. Component Specifications

### 5.1 YAMLParser (parser/yaml_parser.py)

**Purpose**: Load and parse pack.yaml files into structured Python objects.

```python
class YAMLParser:
    """
    Parses AgentSpec YAML files into Pydantic models.

    Features:
    - Multi-file resolution (external schema references)
    - Environment variable interpolation
    - Default value injection
    - Syntax error handling with line numbers
    """

    def parse(self, yaml_path: Path) -> AgentSpec:
        """Parse pack.yaml to AgentSpec model."""

    def resolve_references(self, spec: Dict) -> Dict:
        """Resolve $ref pointers to external files."""

    def interpolate_env(self, value: str) -> str:
        """Replace ${VAR} with environment values."""
```

**Inputs**: Path to pack.yaml
**Outputs**: `AgentSpec` Pydantic model

---

### 5.2 SpecValidator (parser/spec_validator.py)

**Purpose**: Validate AgentSpec against GreenLang requirements.

```python
class SpecValidator:
    """
    Validates AgentSpec for GreenLang compliance.

    Validation Rules:
    1. Required fields present (id, name, version)
    2. At least one agent defined
    3. All tool references resolvable
    4. Zero-hallucination rules followed
    5. Provenance configuration complete
    """

    def validate(self, spec: AgentSpec) -> ValidationResult:
        """Run all validations, return errors/warnings."""

    def check_zero_hallucination(self, agent: AgentDef) -> List[str]:
        """Verify agent doesn't use LLM for calculations."""

    def check_tool_references(self, agent: AgentDef, tools: List[ToolDef]) -> List[str]:
        """Verify all referenced tools are defined."""
```

**Validation Categories**:
| Category | Rule | Severity |
|----------|------|----------|
| Structure | `pack.id` required | ERROR |
| Structure | At least 1 agent | ERROR |
| Compliance | deterministic=True for calculators | ERROR |
| Compliance | provenance.enable_audit=True | WARNING |
| References | Tool IDs must exist | ERROR |
| Types | Input/output types valid | ERROR |

---

### 5.3 GeneratorEngine (engine.py)

**Purpose**: Orchestrate the complete generation pipeline.

```python
class GeneratorEngine:
    """
    Main orchestrator for agent code generation.

    Pipeline:
    1. Parse YAML spec
    2. Validate spec
    3. Generate all code files
    4. Write to output directory
    5. Run validation on generated code
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.parser = YAMLParser()
        self.validator = SpecValidator()
        self.generators = {
            'agent': AgentGenerator(),
            'model': ModelGenerator(),
            'tool': ToolGenerator(),
            'test': TestGenerator(),
            'init': InitGenerator(),
        }

    def generate(self, spec_path: Path, output_dir: Path) -> GenerationResult:
        """Execute complete generation pipeline."""

    def preview(self, spec_path: Path) -> Dict[str, str]:
        """Generate without writing, return file contents."""
```

**Class Diagram**:
```
+------------------+       +------------------+
| GeneratorEngine  |------>| GeneratorConfig  |
+------------------+       +------------------+
        |
        | uses
        v
+------------------+       +------------------+
| YAMLParser       |------>| AgentSpec        |
+------------------+       +------------------+
        |                          |
        v                          v
+------------------+       +------------------+
| SpecValidator    |       | ValidationResult |
+------------------+       +------------------+
        |
        | orchestrates
        v
+------------------+
| BaseGenerator    |<-----+
+------------------+      |
        ^                 |
        |                 | implements
+-------+-------+---------+-------+
|       |       |         |       |
v       v       v         v       v
Agent  Model   Tool      Test    Init
Gen    Gen     Gen       Gen     Gen
```

---

### 5.4 AgentGenerator (generators/agent_gen.py)

**Purpose**: Generate the main agent.py file.

```python
class AgentGenerator(BaseGenerator):
    """
    Generates agent.py from AgentSpec.

    Generated Code Structure:
    1. Module docstring
    2. Imports
    3. Enums (from inputs with enum types)
    4. Input model (Pydantic)
    5. Output model (Pydantic)
    6. Agent class with:
       - AGENT_ID, VERSION, DESCRIPTION
       - Lookup tables (emission factors, etc.)
       - __init__()
       - run() method
       - Calculation methods
       - Provenance tracking
    7. PACK_SPEC dictionary
    """

    def generate(self, agent_def: AgentDef, pack: PackMeta) -> str:
        """Generate complete agent.py content."""

    def _generate_imports(self, agent_def: AgentDef) -> str:
        """Generate import statements."""

    def _generate_enums(self, inputs: List[InputDef]) -> str:
        """Generate enum classes from inputs with enum types."""

    def _generate_run_method(self, agent_def: AgentDef) -> str:
        """Generate the run() method with provenance tracking."""
```

**Template Variables**:
```python
template_context = {
    'agent_id': 'regulatory/eudr_compliance_v1',
    'agent_class_name': 'EUDRComplianceAgent',
    'version': '1.0.0',
    'description': 'EU Deforestation Regulation compliance validator',
    'imports': [...],
    'enums': [...],
    'input_model_name': 'EUDRInput',
    'input_fields': [...],
    'output_model_name': 'EUDROutput',
    'output_fields': [...],
    'lookup_tables': [...],
    'calculation_steps': [...],
    'tools': [...],
}
```

---

### 5.5 ModelGenerator (generators/model_gen.py)

**Purpose**: Generate Pydantic models for inputs/outputs.

```python
class ModelGenerator(BaseGenerator):
    """
    Generates Pydantic models from input/output definitions.

    Features:
    - Field type mapping (YAML type -> Python type)
    - Validation rules (min, max, pattern)
    - Field descriptions
    - Optional/required handling
    - Nested model support
    """

    TYPE_MAPPING = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'date': 'date',
        'datetime': 'datetime',
        'array': 'List',
        'object': 'Dict[str, Any]',
        'GeoJSONGeometry': 'GeoLocation',
    }

    def generate_input_model(self, inputs: List[InputDef], name: str) -> str:
        """Generate input Pydantic model."""

    def generate_output_model(self, outputs: List[OutputDef], name: str) -> str:
        """Generate output Pydantic model."""
```

---

### 5.6 ToolGenerator (generators/tool_gen.py)

**Purpose**: Generate tool wrapper classes.

```python
class ToolGenerator(BaseGenerator):
    """
    Generates tool wrapper classes from tool definitions.

    Tool Types:
    - deterministic: Pure functions with lookup tables
    - external_api: HTTP client wrappers
    - ml_model: ML inference wrappers

    Generated Code:
    - Input validation
    - Error handling
    - Rate limiting (for APIs)
    - Caching support
    """

    def generate(self, tool_def: ToolDef) -> str:
        """Generate tool wrapper class."""

    def _generate_deterministic_tool(self, tool_def: ToolDef) -> str:
        """Generate deterministic lookup tool."""

    def _generate_api_tool(self, tool_def: ToolDef) -> str:
        """Generate API client tool with retry logic."""
```

---

### 5.7 TestGenerator (generators/test_gen.py)

**Purpose**: Generate pytest test files.

```python
class TestGenerator(BaseGenerator):
    """
    Generates pytest test files from golden tests.

    Generated Tests:
    - Unit tests for each calculation
    - Integration tests for run() method
    - Validation tests (invalid inputs)
    - Provenance verification tests
    - Edge case tests
    """

    def generate(
        self,
        agent_def: AgentDef,
        golden_tests: GoldenTestSpec,
    ) -> str:
        """Generate test_agent.py content."""

    def _generate_test_case(self, test: GoldenTest) -> str:
        """Generate single test function."""
```

---

## 6. Pydantic Schema Models (parser/schema.py)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class AgentType(str, Enum):
    """Agent classification types."""
    DUE_DILIGENCE_VALIDATOR = "due-diligence-validator"
    DETERMINISTIC_CALCULATOR = "deterministic-calculator"
    ML_CLASSIFIER = "ml-classifier"
    REPORT_GENERATOR = "report-generator"

class ToolType(str, Enum):
    """Tool classification types."""
    DETERMINISTIC = "deterministic"
    EXTERNAL_API = "external_api"
    ML_MODEL = "ml_model"

class InputDef(BaseModel):
    """Input field definition."""
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    description: str = ""
    validation: Optional[Dict[str, Any]] = None

class OutputDef(BaseModel):
    """Output field definition."""
    name: str
    type: str
    description: str = ""

class AgentDef(BaseModel):
    """Agent definition from pack.yaml."""
    id: str
    name: str
    type: AgentType
    description: str
    inputs: List[InputDef]
    outputs: List[OutputDef]
    tools: List[str] = []

class ToolConfig(BaseModel):
    """Tool configuration for external APIs."""
    base_url: Optional[str] = None
    auth_type: Optional[str] = None
    rate_limit: Optional[int] = None
    timeout: Optional[int] = 30

class ToolDef(BaseModel):
    """Tool definition from pack.yaml."""
    id: str
    name: str
    description: str
    type: ToolType
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    config: Optional[ToolConfig] = None

class ValidationRule(BaseModel):
    """Validation rule definition."""
    id: str
    description: str
    condition: str
    severity: str = "error"

class GoldenTestCategory(BaseModel):
    """Golden test category."""
    name: str
    tests: int
    description: str = ""

class GoldenTestSpec(BaseModel):
    """Golden tests specification."""
    count: int
    categories: List[GoldenTestCategory] = []

class MetadataRegulation(BaseModel):
    """Regulatory metadata."""
    name: str
    reference: str
    official_journal: Optional[str] = None
    enforcement_date: Optional[str] = None

class PackMeta(BaseModel):
    """Pack metadata from pack.yaml."""
    id: str
    name: str
    description: str
    version: str
    author: str = "GreenLang"
    license: str = "Proprietary"
    category: str = ""
    priority: str = ""
    deadline: Optional[str] = None

class AgentSpec(BaseModel):
    """Complete agent specification from pack.yaml."""
    pack: PackMeta
    metadata: Dict[str, Any] = {}
    agents: List[AgentDef]
    tools: List[ToolDef] = []
    validation: Optional[Dict[str, Any]] = None
    golden_tests: Optional[GoldenTestSpec] = None
    deployment: Optional[Dict[str, Any]] = None
    documentation: Optional[Dict[str, Any]] = None
```

---

## 7. Zero-Hallucination Enforcement

### 7.1 Design Principles

**CRITICAL**: The generator MUST enforce zero-hallucination for all numeric calculations.

```
+------------------------------------------------------------------+
|                   ZERO-HALLUCINATION BOUNDARY                     |
|                                                                   |
|   LLM ALLOWED (Classification)    |    LLM PROHIBITED            |
|   --------------------------------|---------------------------   |
|   - Entity resolution             |    - Emission factor lookup  |
|   - Commodity classification      |    - Unit conversion         |
|   - Text extraction               |    - Risk score calculation  |
|   - Narrative generation          |    - Compliance thresholds   |
|   - Materiality assessment        |    - Any numeric formula     |
|                                                                   |
+------------------------------------------------------------------+
```

### 7.2 Enforcement Points

1. **Spec Validation**: Check `compute.deterministic: true` for calculators
2. **Tool Type Check**: `deterministic` tools must not call LLM
3. **Formula Verification**: All formulas use lookup tables or pure math
4. **Generated Code Audit**: No `openai`, `anthropic`, `langchain` imports in calculation path

### 7.3 Validator Implementation

```python
class ZeroHallucinationValidator:
    """
    Validates that agents follow zero-hallucination principles.
    """

    PROHIBITED_IMPORTS = ['openai', 'anthropic', 'langchain', 'llm']

    def validate_agent(self, agent_def: AgentDef) -> List[str]:
        """Check agent follows zero-hallucination rules."""
        errors = []

        if agent_def.type == AgentType.DETERMINISTIC_CALCULATOR:
            # Must not reference ML tools
            for tool in agent_def.tools:
                tool_def = self._get_tool(tool)
                if tool_def and tool_def.type == ToolType.ML_MODEL:
                    errors.append(
                        f"Deterministic agent {agent_def.id} cannot use "
                        f"ML tool {tool}. Use lookup tables instead."
                    )

        return errors

    def validate_generated_code(self, code: str) -> List[str]:
        """Scan generated code for prohibited patterns."""
        errors = []

        for prohibited in self.PROHIBITED_IMPORTS:
            if f"import {prohibited}" in code:
                errors.append(f"Prohibited import: {prohibited}")

        return errors
```

---

## 8. Provenance Tracking Integration

### 8.1 Generated Provenance Code

All generated agents must include:

```python
class GeneratedAgent:
    def run(self, input_data: InputModel) -> OutputModel:
        self._provenance_steps = []
        start_time = datetime.utcnow()

        # Step 1: Input validation
        self._track_step("input_validation", {
            "input_hash": self._hash_inputs(input_data),
            "valid": True,
        })

        # Step 2: Lookup
        value = self._lookup_table[key]
        self._track_step("lookup", {
            "table": "emission_factors",
            "key": key,
            "value": value,
            "source": "EPA 2024",
        })

        # Step 3: Calculation
        result = quantity * value
        self._track_step("calculation", {
            "formula": "result = quantity * emission_factor",
            "inputs": {"quantity": quantity, "emission_factor": value},
            "output": result,
        })

        # Final: Compute provenance hash
        provenance_hash = self._calculate_provenance_hash()

        return OutputModel(
            result=result,
            provenance_hash=provenance_hash,
        )

    def _track_step(self, step_type: str, data: Dict) -> None:
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()
```

### 8.2 Template Integration

The `agent.py.j2` template includes provenance tracking:

```jinja2
def run(self, input_data: {{ input_model_name }}) -> {{ output_model_name }}:
    """
    Execute the {{ agent_name }} calculation.

    ZERO-HALLUCINATION: All calculations use deterministic lookups.
    """
    start_time = datetime.utcnow()
    self._provenance_steps = []

    logger.info(
        f"Executing {{ agent_id }}: "
        f"{% for inp in key_inputs %}{{ inp.name }}={input_data.{{ inp.name }}}{% if not loop.last %}, {% endif %}{% endfor %}"
    )

    try:
        {% for step in calculation_steps %}
        # Step {{ loop.index }}: {{ step.description }}
        {{ step.code }}
        self._track_step("{{ step.type }}", {
            "formula": "{{ step.formula }}",
            "inputs": {{ step.inputs_dict }},
            "output": {{ step.output_var }},
        })

        {% endfor %}
        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash()

        output = {{ output_model_name }}(
            {% for field in output_fields %}
            {{ field.name }}={{ field.value }},
            {% endfor %}
            provenance_hash=provenance_hash,
            calculated_at=datetime.utcnow(),
        )

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Calculation complete: {{ primary_output }}={{ output.{{ primary_output }} }} "
            f"(duration: {duration_ms:.2f}ms, provenance: {provenance_hash[:16]}...)"
        )

        return output

    except Exception as e:
        logger.error(f"Calculation failed: {str(e)}", exc_info=True)
        raise
```

---

## 9. Jinja2 Templates

### 9.1 Template Directory Structure

```
templates/
|-- agent.py.j2           # Main agent class
|-- models.py.j2          # Pydantic models (standalone file)
|-- tools.py.j2           # Tool wrappers
|-- test_agent.py.j2      # Pytest tests
|-- __init__.py.j2        # Package exports
|
|-- partials/
|   |-- imports.j2        # Standard imports
|   |-- docstring.j2      # Module/class docstrings
|   |-- enum.j2           # Enum class
|   |-- pydantic_field.j2 # Single Pydantic field
|   |-- provenance.j2     # Provenance tracking methods
|   |-- lookup_table.j2   # Lookup table definition
```

### 9.2 Key Template: agent.py.j2

```jinja2
{% include 'partials/docstring.j2' %}

{% include 'partials/imports.j2' %}

logger = logging.getLogger(__name__)


{% for enum in enums %}
{% include 'partials/enum.j2' %}


{% endfor %}
class {{ input_model_name }}(BaseModel):
    """
    Input model for {{ agent_name }}.

    Attributes:
    {% for field in input_fields %}
        {{ field.name }}: {{ field.description }}
    {% endfor %}
    """

    {% for field in input_fields %}
    {{ field.name }}: {{ field.type }}{% if field.default is not none %} = {{ field.default }}{% elif not field.required %} = None{% endif %}
    {% endfor %}


class {{ output_model_name }}(BaseModel):
    """
    Output model for {{ agent_name }}.
    """

    {% for field in output_fields %}
    {{ field.name }}: {{ field.type }} = Field(..., description="{{ field.description }}")
    {% endfor %}
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class {{ agent_class_name }}:
    """
    {{ agent_id }}: {{ agent_name }}

    {{ description }}

    Uses zero-hallucination deterministic calculations.
    All numeric computations use validated lookup tables.

    Attributes:
        {% for table in lookup_tables %}
        {{ table.name }}: {{ table.description }}
        {% endfor %}
    """

    AGENT_ID = "{{ agent_id }}"
    VERSION = "{{ version }}"
    DESCRIPTION = "{{ description }}"

    {% for table in lookup_tables %}
    {{ table.name }}: Dict[str, {{ table.value_type }}] = {
        {% for key, value in table.entries.items() %}
        "{{ key }}": {{ value }},
        {% endfor %}
    }

    {% endfor %}
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the {{ agent_name }}."""
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"{{ agent_class_name }} initialized (version {self.VERSION})")

    {{ run_method }}

    {% include 'partials/provenance.j2' %}


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "{{ agent_id }}",
    "name": "{{ agent_name }}",
    "version": "{{ version }}",
    "summary": "{{ description }}",
    "tags": {{ tags | tojson }},
    "owners": {{ owners | tojson }},
    "compute": {
        "entrypoint": "python://agents.{{ module_name }}.agent:{{ agent_class_name }}",
        "deterministic": {{ is_deterministic | lower }},
    },
    "provenance": {
        "enable_audit": True,
    },
}
```

---

## 10. Integration Points

### 10.1 Backend Integration

```
backend/
|-- agents/
|   |-- gl_001_carbon_emissions/    # Existing hand-written
|   |-- gl_002_cbam_compliance/     # Existing hand-written
|   |-- gl_XXX_generated/           # <-- Generator output
|       |-- agent.py
|       |-- __init__.py
|       |-- test_agent.py
|
|-- agent_generator/                 # Generator module
|   |-- engine.py
|   |-- ...
|
|-- engines/
|   |-- base_calculator.py          # Inherited by generated agents
|
|-- app/
|   |-- api/
|       |-- v1/
|           |-- agents.py           # Agent registration endpoint
```

### 10.2 BaseCalculator Integration

Generated agents that perform calculations should extend `BaseCalculator`:

```python
from backend.engines.base_calculator import BaseCalculator, ProvenanceMixin

class GeneratedCalculatorAgent(BaseCalculator):
    """Generated agent extending BaseCalculator for provenance."""

    def calculate(self, inputs: Dict[str, Any]) -> CalculationResult:
        self._start_calculation()

        # ... generated calculation steps ...

        return self._build_result(
            formula_id=self.AGENT_ID,
            formula_version=self.VERSION,
            value=result,
            unit=unit,
            inputs_summary=inputs,
            emission_factors_used=ef_list,
        )
```

### 10.3 CLI Integration

```python
# backend/agent_generator/cli.py

import click
from pathlib import Path
from .engine import GeneratorEngine
from .config import GeneratorConfig

@click.group()
def cli():
    """GreenLang Agent Generator CLI."""
    pass

@cli.command()
@click.argument('spec_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--preview', is_flag=True, help='Preview without writing')
def generate(spec_path: str, output: str, preview: bool):
    """Generate agent from pack.yaml specification."""
    engine = GeneratorEngine(GeneratorConfig())

    if preview:
        files = engine.preview(Path(spec_path))
        for path, content in files.items():
            click.echo(f"\n--- {path} ---\n")
            click.echo(content)
    else:
        result = engine.generate(
            Path(spec_path),
            Path(output) if output else None
        )
        click.echo(f"Generated {len(result.files)} files in {result.output_dir}")

@cli.command()
@click.argument('spec_path', type=click.Path(exists=True))
def validate(spec_path: str):
    """Validate pack.yaml specification."""
    engine = GeneratorEngine(GeneratorConfig())
    result = engine.validate(Path(spec_path))

    if result.errors:
        click.echo("Errors:")
        for error in result.errors:
            click.echo(f"  - {error}")

    if result.warnings:
        click.echo("Warnings:")
        for warning in result.warnings:
            click.echo(f"  - {warning}")

    if result.is_valid:
        click.echo("Specification is valid.")
```

---

## 11. Configuration (config.py)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GeneratorConfig:
    """Configuration for the Agent Generator."""

    # Output settings
    output_base_dir: Path = Path("backend/agents")
    module_prefix: str = "gl"

    # Code generation settings
    python_version: str = "3.11"
    line_length: int = 100
    include_type_hints: bool = True
    include_docstrings: bool = True

    # Template settings
    template_dir: Path = Path(__file__).parent / "templates"

    # Validation settings
    enforce_zero_hallucination: bool = True
    require_provenance: bool = True
    require_golden_tests: bool = True
    min_test_coverage: float = 0.85

    # Import settings
    base_calculator_import: str = "backend.engines.base_calculator"

    # Naming conventions
    agent_suffix: str = "Agent"
    input_suffix: str = "Input"
    output_suffix: str = "Output"

    # File names
    agent_filename: str = "agent.py"
    test_filename: str = "test_agent.py"
    init_filename: str = "__init__.py"

    # Generation options
    generate_tests: bool = True
    generate_tools: bool = True
    generate_models_separate: bool = False  # If True, models.py separate file

    # Formatting
    use_black: bool = True
    use_isort: bool = True
```

---

## 12. Generated Output Structure

For a pack.yaml defining an EUDR agent, the generator produces:

```
backend/agents/gl_XXX_eudr_compliance/
|-- __init__.py
|   """
|   GL-XXX: EUDR Compliance Agent
|
|   Validates commodities against EU Deforestation Regulation 2023/1115.
|   """
|   from .agent import EUDRComplianceAgent, EUDRInput, EUDROutput
|   __all__ = ['EUDRComplianceAgent', 'EUDRInput', 'EUDROutput']
|
|-- agent.py
|   """
|   Complete agent implementation with:
|   - Enums (CommodityType, RiskLevel, etc.)
|   - Input/Output Pydantic models
|   - Agent class with run() method
|   - Lookup tables (country risks, CN codes)
|   - Provenance tracking
|   - PACK_SPEC dictionary
|   """
|
|-- test_agent.py
|   """
|   Pytest tests including:
|   - Happy path tests
|   - Edge case tests
|   - Validation tests
|   - Provenance verification
|   """
|
|-- tools/                  # If tools defined
|   |-- __init__.py
|   |-- geojson_parser.py
|   |-- country_lookup.py
```

---

## 13. Error Handling

### 13.1 Exception Hierarchy

```python
class GeneratorError(Exception):
    """Base exception for generator errors."""
    pass

class ParseError(GeneratorError):
    """YAML parsing failed."""
    def __init__(self, message: str, line: int = None, column: int = None):
        self.line = line
        self.column = column
        super().__init__(f"{message} at line {line}, column {column}")

class ValidationError(GeneratorError):
    """Spec validation failed."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")

class TemplateError(GeneratorError):
    """Template rendering failed."""
    pass

class ZeroHallucinationViolation(ValidationError):
    """Agent violates zero-hallucination rules."""
    pass
```

### 13.2 Error Messages

| Error Type | Example Message |
|------------|-----------------|
| ParseError | "Invalid YAML syntax at line 45: expected ':' but found '='" |
| ValidationError | "Required field 'pack.id' is missing" |
| ValidationError | "Agent 'eudr-validator' references undefined tool 'satellite_checker'" |
| ZeroHallucinationViolation | "Deterministic agent cannot use ML tool 'forest_change_detector'" |
| TemplateError | "Template variable 'agent_class_name' is undefined" |

---

## 14. Testing Strategy

### 14.1 Generator Tests

```python
# tests/test_agent_generator/test_engine.py

def test_generate_from_eudr_spec():
    """Test complete generation from EUDR pack.yaml."""
    engine = GeneratorEngine(GeneratorConfig())

    result = engine.generate(
        Path("08-regulatory-agents/eudr/pack.yaml"),
        Path("/tmp/generated"),
    )

    assert result.success
    assert "agent.py" in result.files
    assert "test_agent.py" in result.files

def test_zero_hallucination_enforcement():
    """Test that ML tools in calculators raise error."""
    spec = AgentSpec(
        pack=PackMeta(id="test", name="Test", version="1.0.0", description=""),
        agents=[
            AgentDef(
                id="test-calc",
                name="Test Calculator",
                type=AgentType.DETERMINISTIC_CALCULATOR,
                description="",
                inputs=[],
                outputs=[],
                tools=["ml_detector"],  # ML tool in calculator!
            )
        ],
        tools=[
            ToolDef(
                id="ml_detector",
                name="ML Detector",
                type=ToolType.ML_MODEL,  # ML model type
                description="",
                input_schema={},
                output_schema={},
            )
        ],
    )

    validator = SpecValidator()
    result = validator.validate(spec)

    assert not result.is_valid
    assert "cannot use ML tool" in str(result.errors)
```

### 14.2 Generated Code Tests

The generator produces test files that verify:

```python
# Generated: test_agent.py

class TestEUDRComplianceAgent:
    """Tests for generated EUDR agent."""

    @pytest.fixture
    def agent(self):
        return EUDRComplianceAgent()

    def test_valid_coffee_brazil(self, agent):
        """Test valid coffee import from Brazil."""
        result = agent.run(EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=50000,
            country_of_origin="BR",
            geolocation=GeoLocation(type="Point", coordinates=[-47.5, -15.5]),
            production_date=date(2024, 6, 1),
        ))

        assert result.risk_level == "high"
        assert result.country_risk_score == 75.0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_verification(self, agent):
        """Test that provenance hash is valid."""
        result = agent.run(EUDRInput(...))

        # Provenance should be consistent
        result2 = agent.run(EUDRInput(...))
        assert result.provenance_hash == result2.provenance_hash

    def test_invalid_geolocation_rejected(self, agent):
        """Test that invalid coordinates raise error."""
        with pytest.raises(ValueError):
            agent.run(EUDRInput(
                ...,
                geolocation=GeoLocation(type="Point", coordinates=[999, 999]),
            ))
```

---

## 15. Development Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create module structure with __init__.py files
- [ ] Implement YAMLParser with schema validation
- [ ] Implement SpecValidator with core rules
- [ ] Create GeneratorConfig dataclass

### Phase 2: Core Generators (Week 2)
- [ ] Implement AgentGenerator
- [ ] Implement ModelGenerator
- [ ] Create Jinja2 templates
- [ ] Integrate with BaseCalculator

### Phase 3: Advanced Features (Week 3)
- [ ] Implement ToolGenerator
- [ ] Implement TestGenerator
- [ ] Add zero-hallucination validation
- [ ] Add provenance tracking integration

### Phase 4: Testing & Polish (Week 4)
- [ ] Comprehensive unit tests
- [ ] Integration tests with existing agents
- [ ] CLI implementation
- [ ] Documentation

---

## 16. Appendix: Type Mapping Reference

| YAML Type | Python Type | Pydantic Field |
|-----------|-------------|----------------|
| string | str | Field(...) |
| integer | int | Field(..., ge=0) |
| number | float | Field(...) |
| boolean | bool | Field(default=False) |
| date | date | Field(...) |
| datetime | datetime | Field(default_factory=datetime.utcnow) |
| array | List[T] | Field(default_factory=list) |
| object | Dict[str, Any] | Field(default_factory=dict) |
| GeoJSONGeometry | GeoLocation | Field(...) |
| CommodityType | CommodityType (Enum) | Field(...) |

---

## 17. Appendix: Naming Conventions

| Context | Convention | Example |
|---------|------------|---------|
| Module name | snake_case with prefix | gl_004_eudr_compliance |
| Agent class | PascalCase + Agent | EUDRComplianceAgent |
| Input model | PascalCase + Input | EUDRInput |
| Output model | PascalCase + Output | EUDROutput |
| Enum class | PascalCase | CommodityType |
| Enum member | UPPER_SNAKE_CASE | PALM_OIL |
| Method | snake_case | calculate_traceability |
| Private method | _snake_case | _track_step |
| Constant | UPPER_SNAKE_CASE | AGENT_ID |
| Lookup table | UPPER_SNAKE_CASE | CN_TO_COMMODITY |

---

*Document Version: 1.0.0*
*Last Updated: 2025-12-09*
*Author: GreenLang Architecture Team*
