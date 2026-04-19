# Agent Generator Architecture

**Version**: 1.0.0
**Status**: Design
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Executive Summary

The Agent Generator is the core engine of the GreenLang Agent Factory. It transforms AgentSpec YAML definitions into production-ready agent packs with complete implementation skeletons, test suites, and deployment configurations.

**Key Capabilities**:
- Input: AgentSpec v2 YAML (single source of truth)
- Output: Complete agent pack (code, tests, docs, deployment)
- Zero-hallucination code generation (template-based, deterministic)
- Integration with GreenLang calculators, tools, and infrastructure
- Production-ready output (85%+ test coverage, type-safe, documented)

**Processing Pipeline**:
```
AgentSpec YAML → Parser → Validator → Code Generator → Test Generator → Pack Assembler → Complete Agent Pack
```

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Generator Engine                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   AgentSpec  │────▶│   Validator  │────▶│    Parser    │    │
│  │     YAML     │     │   (Schema)   │     │  (AST/Model) │    │
│  └──────────────┘     └──────────────┘     └──────┬───────┘    │
│                                                     │            │
│                       ┌─────────────────────────────┘            │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Code Generation Engine                     │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │    │
│  │  │Agent Class   │  │Tool Wrappers │  │Graph Config │  │    │
│  │  │Generator     │  │Generator     │  │Generator    │  │    │
│  │  └──────────────┘  └──────────────┘  └─────────────┘  │    │
│  │                                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │    │
│  │  │Test Suite    │  │Prompt        │  │Integration  │  │    │
│  │  │Generator     │  │Generator     │  │Generator    │  │    │
│  │  └──────────────┘  └──────────────┘  └─────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Pack Assembly Engine                     │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │  - Directory structure creation                       │      │
│  │  - File organization and naming                       │      │
│  │  - Metadata generation (README, docs)                 │      │
│  │  - Dependency resolution and requirements.txt         │      │
│  │  - Configuration file generation                      │      │
│  │  - Deployment manifests (K8s, Docker)                 │      │
│  └──────────────────────────────────────────────────────┘      │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Complete Agent Pack                      │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │  ├── agent_code/                                      │      │
│  │  ├── tools/                                           │      │
│  │  ├── tests/                                           │      │
│  │  ├── deployment/                                      │      │
│  │  ├── docs/                                            │      │
│  │  ├── pack.yaml                                        │      │
│  │  └── README.md                                        │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Architecture

```
AgentGenerator/
├── parser/
│   ├── __init__.py
│   ├── yaml_parser.py          # YAML → Pydantic models
│   ├── spec_validator.py       # AgentSpec v2 validation
│   └── spec_model.py           # Pydantic models for spec
│
├── validators/
│   ├── __init__.py
│   ├── schema_validator.py     # JSON Schema validation
│   ├── semantic_validator.py   # Business logic validation
│   ├── dependency_validator.py # Dependency resolution
│   └── compliance_validator.py # Standards compliance checks
│
├── generators/
│   ├── __init__.py
│   ├── agent_class_gen.py      # Agent class skeleton
│   ├── tool_wrapper_gen.py     # Tool integration code
│   ├── graph_config_gen.py     # LangGraph configuration
│   ├── prompt_gen.py           # Prompt templates
│   ├── test_gen.py             # Test suite generation
│   ├── integration_gen.py      # Integration stubs
│   └── deployment_gen.py       # Deployment configs
│
├── templates/
│   ├── agent_class/
│   │   ├── base_agent.py.jinja2
│   │   ├── calculator_agent.py.jinja2
│   │   ├── llm_agent.py.jinja2
│   │   └── orchestrator_agent.py.jinja2
│   │
│   ├── tools/
│   │   ├── calculator_tool.py.jinja2
│   │   └── llm_tool.py.jinja2
│   │
│   ├── tests/
│   │   ├── test_agent.py.jinja2
│   │   ├── test_tools.py.jinja2
│   │   └── test_integration.py.jinja2
│   │
│   ├── graph/
│   │   └── agent_graph.yaml.jinja2
│   │
│   ├── deployment/
│   │   ├── Dockerfile.jinja2
│   │   ├── k8s_manifest.yaml.jinja2
│   │   └── docker-compose.yaml.jinja2
│   │
│   └── docs/
│       ├── README.md.jinja2
│       ├── ARCHITECTURE.md.jinja2
│       └── API.md.jinja2
│
├── assembler/
│   ├── __init__.py
│   ├── pack_builder.py         # Assembles complete pack
│   ├── directory_builder.py    # Creates directory structure
│   └── file_writer.py          # Writes files to disk
│
├── engine/
│   ├── __init__.py
│   └── generator_engine.py     # Main orchestration engine
│
└── cli/
    ├── __init__.py
    └── commands.py             # CLI command implementations
```

---

## 2. Input: AgentSpec YAML

### 2.1 AgentSpec v2 Schema

The generator accepts AgentSpec v2 YAML as input. Key sections:

```yaml
schema_version: "2.0.0"
id: "emissions/fuel_agent_v1"
name: "Fuel Emissions Agent"
version: "1.0.0"
summary: "Calculate fuel combustion emissions"

metadata:
  tags: ["emissions", "scope1", "fuel"]
  owners: ["greenlang-team"]
  license: "MIT"

compute:
  entrypoint: "python://greenlang.agents.fuel_agent_ai:compute"
  deterministic: true
  timeout_seconds: 30

  inputs:
    fuel_type:
      dtype: "string"
      required: true
      constraints:
        enum: ["natural_gas", "coal", "diesel"]

    amount:
      dtype: "float64"
      unit: "therm"
      required: true
      constraints:
        ge: 0.0

  outputs:
    co2e_emissions_kg:
      dtype: "float64"
      unit: "kgCO2e"

  factors:
    - ref: "ef://ipcc/natural-gas-combustion"
      gwp_set: "AR6GWP100"

ai:
  json_mode: true
  system_prompt: "Calculate fuel emissions using provided tools."
  budget:
    max_cost_usd: 1.0
    max_tokens: 15000
  tools:
    - name: "calculate_emissions"
      impl: "python://greenlang.calculators.fuel:calculate_fuel_emissions"
      safety: "deterministic"

realtime:
  default_mode: "replay"
  snapshot_path: "./snapshots/"

provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"
  record: ["seed", "temperature", "tool_calls", "ef_cids"]
```

### 2.2 Validation Requirements

Before generation, the spec must pass:

1. **Schema Validation**: Valid AgentSpec v2 YAML structure
2. **Semantic Validation**: Business logic constraints
3. **Dependency Validation**: All referenced tools/calculators exist
4. **Compliance Validation**: Meets GreenLang standards

---

## 3. Output: Complete Agent Pack

### 3.1 Generated Directory Structure

```
GL-XXX-AgentName/
├── pack.yaml                     # AgentSpec v2 manifest (copied)
├── README.md                     # Auto-generated documentation
├── ARCHITECTURE.md               # Architecture overview
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── pytest.ini                    # Test configuration
├── .env.template                 # Environment variables template
│
├── agent_name/
│   ├── __init__.py
│   ├── agent.py                  # Main agent class
│   ├── config.py                 # Configuration models
│   ├── tools.py                  # Tool implementations
│   └── prompts.py                # Prompt templates
│
├── graph/
│   └── agent_graph.yaml          # LangGraph configuration
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py             # Agent tests
│   ├── test_tools.py             # Tool tests
│   ├── test_integration.py       # Integration tests
│   ├── test_determinism.py       # Determinism tests
│   └── fixtures/
│       └── test_data.yaml        # Test fixtures
│
├── deployment/
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   └── docker-compose.yaml
│
├── docs/
│   ├── API.md                    # API documentation
│   ├── TOOLS.md                  # Tool specifications
│   └── EXAMPLES.md               # Usage examples
│
├── monitoring/
│   ├── grafana/
│   │   └── dashboard.json
│   └── alerts/
│       └── alert_rules.yaml
│
└── sbom/
    └── sbom.json                 # Software Bill of Materials
```

### 3.2 Generated Code Structure

#### Agent Class Skeleton

```python
"""
{AgentName}Agent - {Summary from AgentSpec}

Generated from AgentSpec v2 on {timestamp}
AgentSpec ID: {spec.id}
Version: {spec.version}
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime

from greenlang_core import BaseAgent, AgentConfig
from greenlang_validation import ValidationResult
from greenlang_provenance import ProvenanceTracker

# Import calculators referenced in spec
{for tool in spec.ai.tools}
from {tool.impl_module} import {tool.impl_function}
{endfor}

logger = logging.getLogger(__name__)


class {AgentName}Input(BaseModel):
    """Input data model for {AgentName}Agent."""

    {for field_name, field_spec in spec.compute.inputs.items()}
    {field_name}: {field_spec.dtype_to_python_type()} = Field(
        ...,
        description="{field_spec.description}",
        {if field_spec.constraints}
        {if field_spec.constraints.ge}ge={field_spec.constraints.ge},{endif}
        {if field_spec.constraints.le}le={field_spec.constraints.le},{endif}
        {endif}
    )
    {endfor}


class {AgentName}Output(BaseModel):
    """Output data model for {AgentName}Agent."""

    {for field_name, field_spec in spec.compute.outputs.items()}
    {field_name}: {field_spec.dtype_to_python_type()} = Field(
        ..., description="{field_spec.description}"
    )
    {endfor}

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")


class {AgentName}Agent(BaseAgent):
    """
    {AgentName}Agent implementation.

    {spec.summary}

    Generated from AgentSpec v2: {spec.id}
    Deterministic: {spec.compute.deterministic}
    Zero Hallucination: {spec.provenance.zero_hallucination}
    """

    def __init__(self, config: AgentConfig):
        """Initialize {AgentName}Agent."""
        super().__init__(config)
        self.provenance_tracker = ProvenanceTracker()
        self.spec_id = "{spec.id}"
        self.spec_version = "{spec.version}"

        # Load emission factors
        {if spec.compute.factors}
        self.emission_factors = self._load_emission_factors()
        {endif}

    def process(self, input_data: {AgentName}Input) -> {AgentName}Output:
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
            validation_result = self._validate_input(input_data)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.errors}")

            # Step 2: Process data (ZERO HALLUCINATION)
            processed_data = self._process_core_logic(input_data)

            # Step 3: Calculate provenance hash
            provenance_hash = self._calculate_provenance(input_data, processed_data)

            # Step 4: Validate output
            output_validation = self._validate_output(processed_data)

            # Step 5: Create output
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return {AgentName}Output(
                **processed_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"{AgentName}Agent processing failed: {str(e)}", exc_info=True)
            raise

    def _process_core_logic(self, input_data: {AgentName}Input) -> Dict[str, Any]:
        """
        Core processing logic - ZERO HALLUCINATION.

        This method implements deterministic processing only.
        Uses calculators from: {spec.ai.tools}
        """
        # TODO: Implement using GreenLang calculators
        # Example:
        # result = calculate_fuel_emissions(
        #     fuel_type=input_data.fuel_type,
        #     amount=input_data.amount
        # )
        raise NotImplementedError("Implement using GreenLang calculators")

    def _validate_input(self, input_data: {AgentName}Input) -> ValidationResult:
        """Validate input data meets all requirements."""
        # Pydantic handles basic validation
        # Add custom validation here
        return ValidationResult(is_valid=True, errors=[])

    def _validate_output(self, output_data: Dict[str, Any]) -> ValidationResult:
        """Validate output meets all requirements."""
        # Check all required outputs are present
        required_outputs = {spec.compute.outputs.keys()}
        for field in required_outputs:
            if field not in output_data:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Required output '{field}' missing"]
                )
        return ValidationResult(is_valid=True, errors=[])

    def _calculate_provenance(self, input_data: Any, output_data: Any) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_str = f"{input_data.json()}{output_data}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    {if spec.compute.factors}
    def _load_emission_factors(self) -> Dict[str, Any]:
        """Load emission factors from database."""
        # TODO: Implement emission factor loading
        # References: {spec.compute.factors}
        return {}
    {endif}
```

---

## 4. Processing Pipeline

### 4.1 Generation Pipeline Stages

```
Stage 1: Parse & Validate
├── Load AgentSpec YAML
├── Parse into Pydantic models
├── Validate schema (AgentSpec v2)
├── Validate semantics (business logic)
└── Validate dependencies (tools exist)
    ↓
Stage 2: Code Generation
├── Generate agent class skeleton
├── Generate tool wrappers
├── Generate graph configuration
├── Generate prompt templates
└── Generate integration stubs
    ↓
Stage 3: Test Generation
├── Generate unit tests
├── Generate integration tests
├── Generate determinism tests
├── Generate fixtures
└── Generate evaluation scenarios
    ↓
Stage 4: Documentation Generation
├── Generate README.md
├── Generate ARCHITECTURE.md
├── Generate API.md
├── Generate TOOLS.md
└── Generate EXAMPLES.md
    ↓
Stage 5: Deployment Generation
├── Generate Dockerfile
├── Generate K8s manifests
├── Generate docker-compose.yaml
├── Generate monitoring configs
└── Generate SBOM
    ↓
Stage 6: Pack Assembly
├── Create directory structure
├── Write all files
├── Copy static assets
├── Generate metadata
└── Validate complete pack
    ↓
Complete Agent Pack
```

### 4.2 Pipeline Implementation

```python
class AgentGeneratorEngine:
    """Main orchestration engine for agent generation."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.parser = YAMLParser()
        self.validator = SpecValidator()
        self.code_gen = CodeGenerator()
        self.test_gen = TestGenerator()
        self.doc_gen = DocumentationGenerator()
        self.deploy_gen = DeploymentGenerator()
        self.assembler = PackAssembler()

    def generate_agent(
        self,
        spec_path: Path,
        output_dir: Path,
        overwrite: bool = False
    ) -> AgentPack:
        """
        Generate complete agent pack from AgentSpec YAML.

        Args:
            spec_path: Path to AgentSpec v2 YAML file
            output_dir: Output directory for agent pack
            overwrite: Whether to overwrite existing pack

        Returns:
            AgentPack with metadata and file paths

        Raises:
            ValidationError: If spec validation fails
            GenerationError: If code generation fails
        """
        # Stage 1: Parse & Validate
        logger.info(f"Parsing AgentSpec: {spec_path}")
        spec = self.parser.parse(spec_path)

        logger.info(f"Validating AgentSpec: {spec.id}")
        validation_result = self.validator.validate(spec)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)

        # Stage 2: Code Generation
        logger.info("Generating agent code")
        agent_code = self.code_gen.generate_agent_class(spec)
        tool_code = self.code_gen.generate_tool_wrappers(spec)
        graph_config = self.code_gen.generate_graph_config(spec)
        prompts = self.code_gen.generate_prompts(spec)

        # Stage 3: Test Generation
        logger.info("Generating test suite")
        tests = self.test_gen.generate_tests(spec)
        fixtures = self.test_gen.generate_fixtures(spec)

        # Stage 4: Documentation Generation
        logger.info("Generating documentation")
        docs = self.doc_gen.generate_docs(spec)

        # Stage 5: Deployment Generation
        logger.info("Generating deployment configs")
        deployment = self.deploy_gen.generate_deployment(spec)

        # Stage 6: Pack Assembly
        logger.info(f"Assembling pack at {output_dir}")
        pack = self.assembler.assemble(
            spec=spec,
            code=agent_code,
            tools=tool_code,
            graph=graph_config,
            prompts=prompts,
            tests=tests,
            fixtures=fixtures,
            docs=docs,
            deployment=deployment,
            output_dir=output_dir,
            overwrite=overwrite
        )

        logger.info(f"Agent pack generated: {pack.pack_id}")
        return pack
```

---

## 5. Template System Integration

### 5.1 Template Engine: Jinja2

The generator uses Jinja2 for all code templates with custom filters:

```python
from jinja2 import Environment, FileSystemLoader

class TemplateEngine:
    """Template rendering engine with custom filters."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Register custom filters
        self.env.filters['dtype_to_python'] = self._dtype_to_python_type
        self.env.filters['snake_case'] = self._to_snake_case
        self.env.filters['pascal_case'] = self._to_pascal_case
        self.env.filters['format_docstring'] = self._format_docstring

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render template with context."""
        template = self.env.get_template(template_name)
        return template.render(**context)

    @staticmethod
    def _dtype_to_python_type(dtype: str) -> str:
        """Convert AgentSpec dtype to Python type hint."""
        mapping = {
            "string": "str",
            "float64": "float",
            "float32": "float",
            "int64": "int",
            "int32": "int",
            "bool": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]"
        }
        return mapping.get(dtype, "Any")
```

### 5.2 Template Context

Templates receive rich context from AgentSpec:

```python
context = {
    "spec": spec,  # Complete AgentSpec model
    "agent_name": "FuelAgent",
    "agent_class": "FuelAgentAI",
    "module_name": "fuel_agent_ai",
    "timestamp": datetime.now().isoformat(),
    "generator_version": "1.0.0",
    "inputs": spec.compute.inputs,
    "outputs": spec.compute.outputs,
    "tools": spec.ai.tools,
    "deterministic": spec.compute.deterministic,
    "zero_hallucination": spec.provenance.zero_hallucination,
}
```

---

## 6. Integration with GreenLang Infrastructure

### 6.1 Calculator Integration

The generator maps tool definitions to GreenLang calculators:

```yaml
# AgentSpec
ai:
  tools:
    - name: "calculate_emissions"
      impl: "python://greenlang.calculators.fuel:calculate_fuel_emissions"
      safety: "deterministic"
```

Generated code:

```python
from greenlang.calculators.fuel import calculate_fuel_emissions

def _process_core_logic(self, input_data: FuelInput) -> Dict[str, Any]:
    """Core processing using GreenLang calculator."""
    result = calculate_fuel_emissions(
        fuel_type=input_data.fuel_type,
        amount=input_data.amount,
        unit=input_data.unit
    )
    return {
        "co2e_emissions_kg": result.emissions_kg_co2e,
        "emission_factor": result.emission_factor
    }
```

### 6.2 Validation Integration

Uses existing GreenLang validation framework:

```python
from greenlang.validation import ValidationEngine, ValidationResult

def _validate_input(self, input_data: Input) -> ValidationResult:
    """Validate using GreenLang validation engine."""
    validator = ValidationEngine(spec=self.spec)
    return validator.validate_input(input_data)
```

### 6.3 Provenance Integration

Integrates with GreenLang provenance tracking:

```python
from greenlang.provenance import ProvenanceTracker, ProvenanceRecord

def _track_provenance(self, input_data, output_data, tool_calls):
    """Track complete provenance."""
    record = ProvenanceRecord(
        agent_id=self.spec_id,
        agent_version=self.spec_version,
        input_hash=self._hash_data(input_data),
        output_hash=self._hash_data(output_data),
        tool_calls=tool_calls,
        ef_cids=self._extract_ef_cids(tool_calls),
        seed=self.config.seed,
        temperature=self.config.temperature
    )
    self.provenance_tracker.record(record)
```

---

## 7. Error Handling & Validation

### 7.1 Generation-Time Validation

Before generating code, validate:

1. **Spec Schema**: Valid AgentSpec v2 YAML
2. **Required Fields**: All mandatory fields present
3. **Tool References**: All tools exist in GreenLang
4. **Calculator References**: All calculators available
5. **Dependency Resolution**: All dependencies installable
6. **Naming Conflicts**: No duplicate names in inputs/outputs

### 7.2 Error Reporting

Clear, actionable error messages:

```python
class GenerationError(Exception):
    """Base class for generation errors."""

    def __init__(self, code: str, message: str, path: List[str]):
        self.code = code
        self.message = message
        self.path = path
        super().__init__(f"[{code}] {'.'.join(path)}: {message}")


# Example error
raise GenerationError(
    code="MISSING_CALCULATOR",
    message="Calculator 'calculate_fuel_emissions' not found in greenlang.calculators.fuel",
    path=["ai", "tools", "0", "impl"]
)
```

---

## 8. Performance & Optimization

### 8.1 Performance Targets

- **Spec Parsing**: <100ms
- **Validation**: <200ms
- **Code Generation**: <500ms per template
- **Complete Pack Generation**: <5 seconds
- **Parallel Generation**: Support for batch generation

### 8.2 Caching Strategy

```python
class CachedGenerator:
    """Generator with template caching."""

    def __init__(self):
        self.template_cache: Dict[str, str] = {}
        self.spec_cache: Dict[str, AgentSpec] = {}

    @lru_cache(maxsize=100)
    def load_template(self, template_name: str) -> str:
        """Load template with caching."""
        return self.template_engine.load(template_name)
```

---

## 9. Extensibility

### 9.1 Custom Template Support

Users can provide custom templates:

```bash
gl agent create \
  --spec specs/agent.yaml \
  --template-dir ./custom_templates \
  --template-name my_custom_agent.py.jinja2
```

### 9.2 Plugin System

```python
class GeneratorPlugin(ABC):
    """Base class for generator plugins."""

    @abstractmethod
    def pre_generate(self, spec: AgentSpec) -> AgentSpec:
        """Hook before generation."""
        pass

    @abstractmethod
    def post_generate(self, pack: AgentPack) -> AgentPack:
        """Hook after generation."""
        pass

# Example plugin
class CompliancePlugin(GeneratorPlugin):
    """Add compliance documentation."""

    def post_generate(self, pack: AgentPack) -> AgentPack:
        """Generate compliance matrix."""
        compliance_doc = self.generate_compliance_matrix(pack.spec)
        pack.add_file("COMPLIANCE_MATRIX.md", compliance_doc)
        return pack
```

---

## 10. Security Considerations

### 10.1 Code Injection Prevention

- Template escaping enabled
- No `eval()` or `exec()` in generated code
- Strict YAML parsing (no arbitrary code execution)
- Sandboxed template rendering

### 10.2 Dependency Validation

- All dependencies pinned with version ranges
- Vulnerability scanning on generated requirements.txt
- No untrusted sources in dependency list

---

## 11. Monitoring & Observability

### 11.1 Generation Metrics

Track generation metrics:

```python
class GenerationMetrics:
    """Metrics for generation process."""

    generation_count = Counter('agent_generation_total', 'Total agents generated')
    generation_duration = Histogram('agent_generation_duration_seconds', 'Generation time')
    generation_errors = Counter('agent_generation_errors_total', 'Generation errors')
    template_cache_hits = Counter('template_cache_hits', 'Template cache hits')
```

### 11.2 Logging

Structured logging at each stage:

```python
logger.info(
    "Agent generation started",
    extra={
        "spec_id": spec.id,
        "spec_version": spec.version,
        "output_dir": str(output_dir),
        "stage": "parse"
    }
)
```

---

## 12. Future Enhancements

### 12.1 Phase 2 Features

- AI-assisted code completion for TODO sections
- Automatic calculator discovery from spec
- Multi-language support (TypeScript, Go)
- Visual graph editor for agent workflows
- Hot reload for template development

### 12.2 Phase 3 Features

- Agent evolution (update existing agents from new specs)
- Diff generation (show changes between versions)
- Migration scripts (v1 → v2 spec conversion)
- Automated refactoring tools

---

## 13. Success Metrics

### 13.1 Code Quality Metrics

Generated code must meet:

- **Type Coverage**: 100% (all methods have type hints)
- **Docstring Coverage**: 100% (all public methods documented)
- **Test Coverage**: 85%+ (generated tests)
- **Linting**: Passes Ruff with zero errors
- **Type Checking**: Passes Mypy with zero errors

### 13.2 Generation Quality Metrics

- **Success Rate**: >99% (valid specs generate successfully)
- **Error Clarity**: 100% (all errors have clear messages)
- **Template Coverage**: 100% (all spec sections have templates)

---

## 14. References

- **AgentSpec v2**: `C:\Users\aksha\Code-V1_GreenLang\AGENTSPEC_V2_FOUNDATION_GUIDE.md`
- **Migration Guide**: `C:\Users\aksha\Code-V1_GreenLang\AGENTSPEC_V2_MIGRATION_GUIDE.md`
- **Example Pack**: `C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-001\pack.yaml`

---

**Document Status**: Design Complete
**Implementation Status**: Pending
**Next Step**: Implement generator engine and templates
