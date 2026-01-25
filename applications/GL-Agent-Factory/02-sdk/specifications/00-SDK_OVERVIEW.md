# Agent SDK v1 - Comprehensive Overview

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Specification

## Executive Summary

The GreenLang Agent SDK v1 provides a standardized, production-ready framework for building deterministic, auditable, and composable agents for climate and regulatory compliance applications. Built on the foundation of AgentSpec v2, the SDK delivers a comprehensive set of base classes, patterns, tools, and utilities that enable rapid development of high-quality agents while enforcing GreenLang's zero-hallucination guarantee.

**Key Value Propositions:**
- **50% reduction in agent development time** through reusable base classes and patterns
- **Zero-hallucination guarantee** through tool-first architecture
- **100% audit trail** with SHA-256 provenance tracking
- **Type-safe development** with comprehensive Pydantic models
- **Production-ready patterns** extracted from GL-001 through GL-007
- **Seamless integration** with existing AgentSpec v2 infrastructure

---

## Table of Contents

1. [Vision and Goals](#vision-and-goals)
2. [Core Architecture](#core-architecture)
3. [Integration with AgentSpec v2](#integration-with-agentspec-v2)
4. [SDK Components](#sdk-components)
5. [Design Principles](#design-principles)
6. [Technology Stack](#technology-stack)
7. [Development Workflow](#development-workflow)
8. [Success Metrics](#success-metrics)

---

## Vision and Goals

### Vision Statement

The GreenLang Agent SDK v1 establishes the industry standard for building deterministic AI agents for climate and regulatory applications, enabling enterprises to rapidly deploy auditable, compliant, and composable agents at scale.

### Primary Goals

1. **Standardization**
   - Unified base class for all GreenLang agents
   - Consistent lifecycle management (initialize → validate → execute → finalize)
   - Standard error handling and logging patterns
   - Common metadata and provenance structures

2. **Productivity**
   - Reduce agent development time from weeks to days
   - Provide reusable components for common patterns
   - Auto-generate boilerplate code
   - Comprehensive documentation and examples

3. **Quality Assurance**
   - Built-in validation at input/output boundaries
   - Automated schema checking against pack.yaml
   - Type safety with Pydantic models
   - Comprehensive testing utilities

4. **Zero-Hallucination Guarantee**
   - Tool-first architecture for all calculations
   - No LLM-generated numeric values
   - Deterministic computation (temperature=0.0, seed=42)
   - Complete reproducibility

5. **Audit Compliance**
   - SHA-256 provenance tracking
   - Immutable audit logs
   - Citation tracking and validation
   - Regulatory framework mapping

6. **Composability**
   - Agent graph patterns (intake → validation → calculation → reporting)
   - Multi-agent orchestration support
   - Event-driven communication
   - Flexible composition strategies

---

## Core Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GreenLang Agent SDK v1                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Core Base Classes Layer                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ AgentBase    │  │ CalculatorBase│  │ ValidatorBase│  │   │
│  │  │ (AgentSpec   │  │              │  │              │  │   │
│  │  │  v2 Base)    │  │              │  │              │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ RegulatoryBase│  │ ReportingBase│  │IntegrationBase│ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Agent Graph Patterns Layer                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ IntakeAgent  │→ │ValidationAgent│→│CalculationAgent│ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                          ↓                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ ReportingAgent│ │AuditAgent    │  │OrchestratorAgent│ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Tools & Utilities Layer                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Calculator   │  │ Emission     │  │ Regulatory   │  │   │
│  │  │ Tools        │  │ Factor Tools │  │ Engine Tools │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Provenance   │  │ Validation   │  │ Integration  │  │   │
│  │  │ Tracker      │  │ Tools        │  │ Adapters     │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           AgentSpec v2 Integration Layer                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ AgentSpecV2  │  │ pack.yaml    │  │ Schema       │  │   │
│  │  │ Base         │  │ Loader       │  │ Validator    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **AgentBase** | Unified base class for all agents | Lifecycle management, validation, provenance |
| **CalculatorBase** | Base for calculation agents | Zero-hallucination, deterministic compute |
| **ValidatorBase** | Base for validation agents | Schema validation, constraint checking |
| **RegulatoryBase** | Base for compliance agents | Framework mapping, audit logging |
| **ReportingBase** | Base for reporting agents | Template rendering, data aggregation |
| **IntegrationBase** | Base for integration agents | ERP/SCADA connectors, data sync |

---

## Integration with AgentSpec v2

### Building on AgentSpec v2 Foundation

The Agent SDK v1 is built **on top of** AgentSpec v2, not as a replacement. This design ensures:

1. **Full Backward Compatibility**: All existing AgentSpec v2 agents continue to work
2. **Incremental Adoption**: Teams can adopt SDK patterns gradually
3. **Standard Compliance**: SDK agents are AgentSpec v2 compliant by default
4. **Pack.yaml Integration**: Automatic schema validation against pack.yaml

### AgentSpec v2 Base Class

```python
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base
from greenlang.agents.agentspec_v2_base import AgentExecutionContext

# AgentSpec v2 provides:
# - Generic typing: AgentSpecV2Base[InT, OutT]
# - Standard lifecycle: initialize → validate → execute → finalize
# - Schema validation: Automated input/output checking
# - Citation tracking: Built-in support
# - Metrics collection: Execution time, success rate
```

### SDK Enhancement Strategy

```
AgentSpec v2 (Foundation)
         ↓
    SDK v1 (Enhancement)
         ↓
    Domain-Specific Agents
         ↓
    Application Solutions
```

**Enhancement Layers:**

1. **AgentSpec v2**: Core lifecycle and validation
2. **SDK Base Classes**: Domain-specific patterns (Calculator, Validator, Regulatory)
3. **Agent Graphs**: Composition patterns for multi-agent workflows
4. **Tools Library**: Reusable calculation and integration tools
5. **Utilities**: Testing, deployment, monitoring support

### SDK Inheritance Model

```python
# Base hierarchy
AgentSpecV2Base[InT, OutT]  # From core framework
    ↓
SDKAgentBase[InT, OutT]  # SDK additions
    ↓
CalculatorAgentBase[InT, OutT]  # Domain-specific
    ↓
CarbonCalculatorAgent  # Concrete implementation
```

**What SDK Adds:**

- Domain-specific validation logic (emissions, energy, materiality)
- Tool wrapper patterns (calculator tools, EF lookups, regulatory engines)
- Common data models (EmissionsData, EnergyData, MaterialityData)
- Integration adapters (SCADA, ERP, CMMS connectors)
- Reporting templates (GRI, SASB, TCFD, CDP)
- Testing utilities (fixtures, mocks, assertion helpers)

---

## SDK Components

### 1. Base Classes (`greenlang_sdk.base`)

**Purpose**: Reusable base classes for common agent types

**Components:**
- `SDKAgentBase`: Enhanced AgentSpec v2 base with SDK features
- `CalculatorAgentBase`: Zero-hallucination calculation agents
- `ValidatorAgentBase`: Schema and constraint validation agents
- `RegulatoryAgentBase`: Compliance and audit agents
- `ReportingAgentBase`: Report generation agents
- `IntegrationAgentBase`: System integration agents

**Key Features:**
- Type-safe input/output with Pydantic models
- Automatic provenance tracking
- Built-in error handling patterns
- Lifecycle hook system
- Metrics collection

### 2. Agent Graph Patterns (`greenlang_sdk.patterns`)

**Purpose**: Composition patterns for multi-agent workflows

**Patterns:**
- **Linear Pipeline**: Intake → Validation → Calculation → Reporting
- **Parallel Processing**: Fan-out/fan-in for batch operations
- **Conditional Routing**: Dynamic agent selection based on input
- **Hierarchical Orchestration**: Master-slave coordination
- **Event-Driven**: Pub-sub messaging patterns

**Examples:**
- Process Heat Orchestration (GL-001 pattern)
- Boiler Optimization Pipeline (GL-002 pattern)
- Multi-framework Reporting (GL-003 pattern)

### 3. Tools Library (`greenlang_sdk.tools`)

**Purpose**: Reusable deterministic tools for agent operations

**Categories:**

**Calculation Tools:**
- `EmissionsCalculator`: GHG emissions calculations (Scope 1/2/3)
- `EnergyEfficiencyCalculator`: Energy performance metrics
- `MaterialityCalculator`: ESG materiality assessments
- `FinancialImpactCalculator`: Cost/benefit analysis

**Data Tools:**
- `EmissionFactorLookup`: EF database queries with provenance
- `UnitConverter`: Climate unit conversions (kWh, BTU, GJ, etc.)
- `DataValidator`: Schema and constraint validation
- `DataNormalizer`: Format standardization

**Regulatory Tools:**
- `FrameworkMapper`: Map data to GRI/SASB/TCFD/CDP
- `ComplianceChecker`: Regulatory rule validation
- `AuditLogger`: Immutable audit trail creation

**Integration Tools:**
- `SCADAConnector`: OPC UA/Modbus/MQTT integration
- `ERPConnector`: REST/SOAP ERP integration
- `CMMSConnector`: Maintenance system integration

### 4. Data Models (`greenlang_sdk.models`)

**Purpose**: Type-safe data models for common domains

**Models:**
- `EmissionsData`: GHG emissions with provenance
- `EnergyData`: Energy consumption/generation
- `MaterialityData`: ESG materiality assessments
- `ComplianceData`: Regulatory compliance records
- `ProvenanceRecord`: Audit trail metadata
- `Citation`: Source citation with validation

### 5. Validation (`greenlang_sdk.validation`)

**Purpose**: Comprehensive validation utilities

**Validators:**
- `SchemaValidator`: JSON schema validation
- `ConstraintValidator`: Numeric constraints (ge, le, enum)
- `UnitValidator`: Climate unit whitelist checking
- `ProvenanceValidator`: Audit trail integrity checking
- `CitationValidator`: Source citation validation

### 6. Utilities (`greenlang_sdk.utils`)

**Purpose**: Helper functions and utilities

**Utilities:**
- `ProvenanceTracker`: SHA-256 hash calculation
- `DeterministicClock`: Deterministic timestamps
- `Logger`: Structured logging
- `ConfigLoader`: Configuration management
- `ErrorHandler`: Exception handling patterns

### 7. Testing (`greenlang_sdk.testing`)

**Purpose**: Testing utilities for agent development

**Components:**
- `AgentTestCase`: Base test class with fixtures
- `MockTools`: Mock tool implementations
- `Fixtures`: Common test data
- `Assertions`: Custom assertion helpers
- `CoverageReporter`: Test coverage analysis

---

## Design Principles

### 1. Zero-Hallucination Architecture

**Principle**: No LLM-generated numeric values in calculation path

**Implementation:**
```python
# ✅ ALLOWED: Tool-based calculation
result = self.calculator_tool.calculate_emissions(
    activity_data=1000,
    emission_factor=53.06  # From database
)

# ❌ NOT ALLOWED: LLM calculation
result = self.llm.calculate_emissions(
    "Calculate emissions for 1000 kg of natural gas"
)
```

**Allowed LLM Usage:**
- Text classification and categorization
- Entity resolution and matching
- Narrative generation (summaries, explanations)
- Materiality assessment (qualitative)

**Not Allowed LLM Usage:**
- Numeric calculations (emissions, energy, financial)
- Regulatory compliance values
- Audit trail values
- Data validation outcomes

### 2. Tool-First Design

**Principle**: All calculations performed by deterministic tools

**Tool Properties:**
- Deterministic (same input → same output)
- Reproducible (seed=42, temperature=0.0)
- Auditable (input/output hashing)
- Validated (unit tests, integration tests)
- Documented (formulas, standards, references)

**Tool Execution Pattern:**
```python
class ToolExecutor:
    def execute(self, tool: Tool, params: Dict) -> ToolResult:
        # 1. Validate inputs
        validated_params = self._validate(params, tool.input_schema)

        # 2. Calculate input hash
        input_hash = sha256(validated_params)

        # 3. Execute deterministic calculation
        result = tool.calculate(validated_params)

        # 4. Calculate output hash
        output_hash = sha256(result)

        # 5. Create provenance record
        provenance = ProvenanceRecord(
            tool_id=tool.id,
            input_hash=input_hash,
            output_hash=output_hash,
            formula=tool.formula,
            standards=tool.standards
        )

        return ToolResult(data=result, provenance=provenance)
```

### 3. Type Safety First

**Principle**: Comprehensive type hints and Pydantic models

**Implementation:**
```python
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)

class SDKAgentBase(AgentSpecV2Base[InT, OutT]):
    """Type-safe agent base with generic input/output."""

    def execute_impl(
        self,
        validated_input: InT,  # Type-checked by IDE
        context: AgentExecutionContext
    ) -> OutT:  # Return type enforced
        pass
```

**Benefits:**
- IDE autocomplete and type checking
- Runtime validation with Pydantic
- Self-documenting code
- Reduced bugs from type errors

### 4. Fail-Safe Design

**Principle**: Graceful degradation with safety prioritization

**Error Handling Hierarchy:**
1. **Validation Errors**: Fail fast with clear error messages
2. **Calculation Errors**: Return error result, log details
3. **Integration Errors**: Use cached data, trigger alert
4. **Critical Errors**: Emergency shutdown, notify operators

**Example:**
```python
def execute(self, input_data: Input) -> Result:
    try:
        # Validate input
        validated = self.validate_input(input_data)

        # Execute calculation
        result = self.calculate(validated)

        return Result(success=True, data=result)

    except ValidationError as e:
        # Fail fast - bad input
        return Result(success=False, error=f"Validation failed: {e}")

    except CalculationError as e:
        # Log and return error - calculation issue
        logger.error(f"Calculation failed: {e}", exc_info=True)
        return Result(success=False, error=f"Calculation failed: {e}")

    except IntegrationError as e:
        # Use fallback - external system issue
        logger.warning(f"Integration failed, using cached data: {e}")
        result = self.get_cached_data()
        return Result(success=True, data=result, warning="Using cached data")

    except CriticalError as e:
        # Emergency shutdown - safety issue
        logger.critical(f"CRITICAL ERROR: {e}", exc_info=True)
        self.emergency_shutdown()
        raise
```

### 5. Composability

**Principle**: Agents compose into larger workflows

**Composition Patterns:**

**Linear Pipeline:**
```python
workflow = Pipeline([
    IntakeAgent(),
    ValidationAgent(),
    CalculationAgent(),
    ReportingAgent()
])
```

**Parallel Processing:**
```python
workflow = ParallelProcessor([
    Scope1Calculator(),
    Scope2Calculator(),
    Scope3Calculator()
]).then(AggregatorAgent())
```

**Conditional Routing:**
```python
workflow = Router({
    "emissions": EmissionsAgent(),
    "energy": EnergyAgent(),
    "water": WaterAgent()
}).route_by(lambda input: input.category)
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Core development language |
| **Type System** | Pydantic | 2.5+ | Data validation and serialization |
| **Base Framework** | AgentSpec v2 | 2.0.0 | Agent lifecycle management |
| **Logging** | structlog | 23.2+ | Structured logging |
| **Testing** | pytest | 7.4+ | Unit and integration testing |

### Tool Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Calculation** | NumPy, SciPy | Numeric computation |
| **Data Processing** | Pandas | Data manipulation |
| **Hashing** | hashlib (SHA-256) | Provenance tracking |
| **Integration** | OPC UA, Modbus, MQTT | Industrial protocols |
| **API** | httpx, FastAPI | HTTP/REST integration |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Ruff** | Linting and formatting |
| **Mypy** | Static type checking |
| **Black** | Code formatting |
| **pytest-cov** | Test coverage |
| **Bandit** | Security scanning |

---

## Development Workflow

### Agent Development Lifecycle

```
1. Design
   ↓
2. Implement Base Class
   ↓
3. Add Tools
   ↓
4. Write Tests
   ↓
5. Create pack.yaml
   ↓
6. Validate
   ↓
7. Deploy
```

### Step-by-Step Guide

**1. Design Agent**
- Define input/output schema
- Identify required tools
- Map regulatory requirements
- Design provenance strategy

**2. Implement Base Class**
```python
from greenlang_sdk.base import CalculatorAgentBase

class MyAgent(CalculatorAgentBase[MyInput, MyOutput]):
    def execute_impl(self, input: MyInput, context) -> MyOutput:
        # Implementation
        pass
```

**3. Add Tools**
```python
from greenlang_sdk.tools import EmissionsCalculator

class MyAgent(CalculatorAgentBase[MyInput, MyOutput]):
    def __init__(self):
        super().__init__()
        self.calculator = EmissionsCalculator()

    def execute_impl(self, input: MyInput, context) -> MyOutput:
        result = self.calculator.calculate(
            activity_data=input.amount,
            emission_factor=self.get_ef(input.fuel_type)
        )
        return MyOutput(emissions=result.value)
```

**4. Write Tests**
```python
from greenlang_sdk.testing import AgentTestCase

class TestMyAgent(AgentTestCase):
    def test_calculation(self):
        agent = MyAgent()
        result = agent.run(MyInput(amount=1000, fuel_type="natural_gas"))
        assert result.success
        assert result.data.emissions > 0
```

**5. Create pack.yaml**
```yaml
schema_version: "2.0.0"
id: "emissions/my_agent_v1"
name: "My Emissions Agent"
version: "1.0.0"

compute:
  entrypoint: "python://my_module:compute"
  inputs:
    amount:
      dtype: "float64"
      unit: "kg"
      required: true
  outputs:
    emissions:
      dtype: "float64"
      unit: "kgCO2e"
```

**6. Validate**
```bash
# Run tests
pytest tests/

# Validate pack.yaml
greenlang validate pack.yaml

# Check type hints
mypy my_agent.py

# Lint code
ruff check my_agent.py
```

**7. Deploy**
```bash
# Build agent pack
greenlang pack build

# Test in staging
greenlang pack deploy --env staging

# Deploy to production
greenlang pack deploy --env production
```

---

## Success Metrics

### Development Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Development Time** | 50% reduction | Time from spec to production |
| **Code Reuse** | 60%+ | Percentage using SDK base classes |
| **Test Coverage** | 85%+ | Unit + integration test coverage |
| **Type Coverage** | 100% | Percentage with type hints |
| **Documentation** | 100% | Public methods with docstrings |

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Zero Hallucinations** | 100% | No LLM-generated numeric values |
| **Determinism** | 100% | Reproducible results |
| **Audit Compliance** | 100% | SHA-256 provenance for all calculations |
| **Schema Validation** | 100% | All agents validate against pack.yaml |
| **Error Handling** | 100% | All public methods with try/except |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Agent Creation** | <100ms | Time to instantiate agent |
| **Validation** | <50ms | Input/output validation time |
| **Tool Execution** | <500ms | Average tool execution time |
| **E2E Latency** | <2000ms | End-to-end request processing |
| **Throughput** | 100+ rps | Requests per second per agent |

### Adoption Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **SDK Adoption** | 80%+ new agents | Percentage using SDK |
| **Developer Satisfaction** | 4.5/5 | Survey score |
| **Support Tickets** | <10/month | SDK-related tickets |
| **Community Contributions** | 5+/quarter | External contributions |

---

## Next Steps

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement `SDKAgentBase` with enhanced features
- [ ] Create `CalculatorAgentBase` with zero-hallucination patterns
- [ ] Build provenance tracking utilities
- [ ] Write comprehensive tests

### Phase 2: Tools & Patterns (Weeks 3-4)
- [ ] Implement core calculation tools
- [ ] Create agent graph patterns
- [ ] Build integration adapters
- [ ] Document tool specifications

### Phase 3: Documentation & Examples (Week 5)
- [ ] Write developer guide
- [ ] Create agent examples
- [ ] Build tutorial series
- [ ] Record demo videos

### Phase 4: Validation & Launch (Week 6)
- [ ] Beta testing with 3 teams
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] Production launch

---

## References

- **AgentSpec v2 Base**: `greenlang/agents/agentspec_v2_base.py`
- **AgentSpec v2 Guide**: `AGENTSPEC_V2_FOUNDATION_GUIDE.md`
- **GL-001 Architecture**: `docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-001/ARCHITECTURE.md`
- **GL-002 Architecture**: `docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-002/ARCHITECTURE.md`
- **Python SDK**: `sdks/python/greenlang_sdk/`

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-03
**Author**: GL-BackendDeveloper
**Status**: Specification - Ready for Implementation
