# GreenLang Architecture Overview

**System Architecture, Design Principles, and Extension Points**

This document provides a comprehensive overview of GreenLang's architecture, design decisions, and how components fit together.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Design Principles](#design-principles)
4. [Core Components](#core-components)
5. [Extension Points](#extension-points)
6. [Performance Considerations](#performance-considerations)
7. [Security Model](#security-model)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐             │
│  │ Python SDK   │  │     CLI      │  │  REST API │             │
│  │              │  │   (gl cmd)   │  │ (Future)  │             │
│  └──────────────┘  └──────────────┘  └───────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 CLIMATE INTELLIGENCE LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │ AI Agents    │  │ ML Models    │  │ RAG System        │    │
│  │ (100+)       │  │ (Forecasting)│  │ (Compliance)      │    │
│  └──────────────┘  └──────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FRAMEWORK CORE LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Agent Base │  │ Pipeline   │  │ Context  │  │ Provenance │ │
│  │ Classes    │  │ Orchestrator│ │ Manager  │  │ Tracking   │ │
│  └────────────┘  └────────────┘  └──────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Pack       │  │ Security   │  │ Artifact │  │ Runtime    │ │
│  │ System     │  │ (SBOM,Sign)│  │ Storage  │  │ Executor   │ │
│  └────────────┘  └────────────┘  └──────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT LAYER                             │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐ │
│  │   Local    │  │   Docker   │  │ Kubernetes│ │   Cloud    │ │
│  │   (Dev)    │  │ (Container)│  │  (Prod)   │ │ (AWS/Azure)│ │
│  └────────────┘  └────────────┘  └──────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Input Data → Validation → Agent Processing → Provenance → Output
     ↓           ↓              ↓                ↓           ↓
 CSV/JSON    Pydantic      Calculations     Automatic    JSON/CSV
   API       Models         +Caching        Tracking     Reports
  Files      Schema         +Error          +Signing    Artifacts
            Validation      Handling        +Audit
```

---

## Module Dependency Graph

### Core Module Dependencies

```
greenlang/
├── sdk/                    (Base abstractions - no dependencies)
│   ├── base.py            → Agent, Pipeline, Result
│   ├── context.py         → Context, Artifact
│   └── __init__.py
│
├── provenance/            (Depends on: sdk)
│   ├── decorators.py     → @traced, @track_provenance
│   ├── records.py        → ProvenanceContext, ProvenanceRecord
│   ├── ledger.py         → ProvenanceLedger
│   └── hashing.py        → Hashing utilities
│
├── emissions/             (Depends on: sdk)
│   ├── factors.py        → EmissionFactorService
│   ├── calculations.py   → Emission calculators
│   └── models.py         → FuelConsumption, EmissionResult
│
├── agents/                (Depends on: sdk, provenance, emissions)
│   ├── building.py       → BuildingAgent
│   ├── calculator.py     → CalculatorAgent
│   └── reporter.py       → ReporterAgent
│
├── packs/                 (Depends on: sdk)
│   ├── manifest.py       → Pack manifest handling
│   ├── loader.py         → Pack loading
│   └── registry.py       → Pack registry
│
├── runtime/               (Depends on: sdk, packs, security)
│   ├── executor.py       → Pipeline executor
│   └── golden.py         → Golden dataset management
│
├── security/              (No framework dependencies)
│   ├── signing.py        → Sigstore signing
│   ├── sbom.py           → SBOM generation
│   └── network.py        → Network security
│
├── cli/                   (Depends on: all above)
│   ├── main.py           → CLI entry point
│   ├── cmd_run.py        → Run command
│   ├── cmd_pack.py       → Pack commands
│   └── cmd_verify.py     → Verification commands
│
└── intelligence/          (Depends on: sdk, agents)
    ├── llm/              → LLM integration
    ├── rag/              → RAG system
    └── ml/               → ML models
```

### Dependency Rules

1. **No Circular Dependencies**: Enforced through careful module design
2. **SDK is Foundation**: All modules depend on `sdk`, but `sdk` depends on nothing
3. **Provenance is Isolated**: Can be used independently
4. **CLI is Top Layer**: Depends on everything, but nothing depends on CLI

---

## Design Principles

### 1. Agent-Based Architecture

**Principle**: Every computation is an Agent with well-defined inputs and outputs.

```python
# Good: Agent-based
class EmissionsAgent(Agent[FuelInput, EmissionOutput]):
    def process(self, input_data: FuelInput) -> EmissionOutput:
        return calculate(input_data)

# Bad: Function-based (legacy pattern)
def calculate_emissions(fuel_type, consumption):
    return consumption * 0.417  # Hardcoded
```

**Benefits:**
- Composability: Agents can be chained in pipelines
- Testability: Each agent is independently testable
- Reusability: Agents can be packaged and shared
- Maintainability: Clear boundaries and responsibilities

### 2. Provenance by Default

**Principle**: All operations are tracked automatically with provenance.

```python
@traced(save_path="provenance/calc.json")
def process(self, input_data):
    # Automatic provenance tracking:
    # - Inputs captured
    # - Outputs captured
    # - Execution time tracked
    # - Environment recorded
    result = calculate(input_data)
    return result
```

**Benefits:**
- Regulatory compliance (audit trails)
- Reproducibility (can replay calculations)
- Debugging (understand what happened)
- Trust (verify data lineage)

### 3. Validation First

**Principle**: Use Pydantic models for automatic validation.

```python
class FuelInput(BaseModel):
    fuel_type: str = Field(..., regex="^(electricity|gas|diesel)$")
    consumption: float = Field(..., gt=0)
    unit: str = Field(default="kWh")

# Validation happens automatically
input_data = FuelInput(fuel_type="electricity", consumption=1000)
```

**Benefits:**
- Fail fast: Invalid data rejected immediately
- Clear contracts: Input/output schemas documented
- Type safety: IDE support and static analysis
- Self-documenting: Schema serves as documentation

### 4. Framework Over Library

**Principle**: Provide structure and conventions, not just utilities.

```
Library Approach (❌):           Framework Approach (✅):
- Provides functions            - Provides base classes
- User writes structure         - User extends framework
- Flexible but chaotic          - Structured but extensible
- Each project different        - Consistent across projects
```

### 5. Security by Design

**Principle**: Security is not optional, it's built-in.

- Zero hardcoded secrets
- Sigstore signing for all artifacts
- SBOM generation for supply chain security
- Policy enforcement with OPA

---

## Core Components

### Component 1: Agent System

**Purpose**: Stateless computation units for climate calculations.

**Key Classes:**
- `Agent[TInput, TOutput]`: Base agent class
- `Result`: Standard result container
- `Metadata`: Agent metadata (id, version, description)

**Design Decisions:**
- Generics for type safety
- Abstract base class forces implementation
- `run()` method handles validation + error handling automatically

**Example:**
```python
class BuildingAgent(Agent[BuildingInput, BuildingOutput]):
    def validate(self, input_data: BuildingInput) -> bool:
        return input_data.area_sqft > 0

    @traced()
    def process(self, input_data: BuildingInput) -> BuildingOutput:
        emissions = calculate_building_emissions(input_data)
        return BuildingOutput(emissions=emissions)
```

### Component 2: Pipeline Orchestration

**Purpose**: Coordinate multiple agents in workflows.

**Key Classes:**
- `Pipeline`: Base pipeline class
- `Context`: Execution context with artifacts
- `PipelineRunner`: YAML pipeline executor

**Design Decisions:**
- Context object passed between stages
- Artifacts stored automatically
- Provenance tracks entire pipeline

**Example:**
```python
class EmissionsPipeline(Pipeline):
    def execute(self, input_data: dict) -> Result:
        ctx = Context(inputs=input_data)

        # Stage 1
        result1 = self.loader.run(input_data)
        ctx.add_step_result("load", result1)

        # Stage 2
        result2 = self.calculator.run(result1.data)
        ctx.add_step_result("calculate", result2)

        return ctx.to_result()
```

### Component 3: Provenance Framework

**Purpose**: Automatic tracking of data lineage and operations.

**Key Classes:**
- `ProvenanceContext`: Tracks single operation
- `ProvenanceRecord`: Immutable record
- `ProvenanceLedger`: Chain of records

**Design Decisions:**
- Decorators for automatic tracking
- Context managers for manual tracking
- Cryptographic hashing for integrity
- JSON serialization for storage

### Component 4: Pack System

**Purpose**: Modular, reusable components for climate intelligence.

**Key Classes:**
- `PackManifest`: Pack metadata and dependencies
- `PackLoader`: Load and validate packs
- `PackRegistry`: Discover and manage packs

**Design Decisions:**
- YAML manifests for pack definition
- Semantic versioning for compatibility
- Dependency resolution with version constraints
- Sandboxed execution for security

### Component 5: Security Infrastructure

**Purpose**: Enterprise-grade security for compliance.

**Key Classes:**
- `SigningService`: Sigstore integration
- `SBOMGenerator`: CycloneDX/SPDX generation
- `PolicyEnforcer`: OPA policy enforcement

**Design Decisions:**
- Sigstore for keyless signing
- Industry-standard SBOM formats
- Policy-as-code with Rego
- Network sandboxing for pack execution

---

## Extension Points

### 1. Custom Agents

**Extension Point**: Subclass `Agent` to create custom agents.

```python
from greenlang.sdk import Agent

class CustomEmissionsAgent(Agent[CustomInput, CustomOutput]):
    """Your custom agent"""

    def validate(self, input_data: CustomInput) -> bool:
        # Your validation logic
        return True

    def process(self, input_data: CustomInput) -> CustomOutput:
        # Your processing logic
        return calculate(input_data)
```

### 2. Custom Emission Factors

**Extension Point**: Register custom emission factors.

```python
from greenlang.emissions import EmissionFactorService

ef_service = EmissionFactorService()

# Add custom factor
ef_service.register_factor(
    fuel_type="biofuel",
    factor=0.12,  # kgCO2e per unit
    unit="liters",
    region="custom",
    source="internal_research_2024"
)
```

### 3. Custom Pipelines

**Extension Point**: Create complex multi-stage pipelines.

```python
from greenlang.sdk import Pipeline

class MultiRegionPipeline(Pipeline):
    """Process data for multiple regions"""

    def execute(self, input_data: dict) -> Result:
        results = {}
        for region in input_data["regions"]:
            agent = self.get_region_agent(region)
            results[region] = agent.run(input_data)
        return Result(success=True, data=results)
```

### 4. Custom Provenance Handlers

**Extension Point**: Customize provenance storage.

```python
from greenlang.provenance.records import ProvenanceContext

class DatabaseProvenanceContext(ProvenanceContext):
    """Store provenance in database instead of files"""

    def finalize(self, output_path=None):
        record = super().finalize(output_path=None)
        # Store in database
        db.save_provenance(record)
        return record
```

### 5. Custom Packs

**Extension Point**: Create and distribute custom packs.

```yaml
# pack.yaml
name: my-custom-pack
version: 1.0.0
description: Custom emissions calculations for manufacturing

agents:
  - name: ManufacturingEmissionsAgent
    type: calculation
    inputs:
      - name: process_type
        type: string
      - name: output_tons
        type: number
    outputs:
      - name: emissions_tons
        type: number

dependencies:
  - greenlang-core>=0.3.0
  - numpy>=1.20.0
```

---

## Performance Considerations

### Caching Strategy

```python
# Agent-level caching
class CachedAgent(Agent):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def process(self, input_data):
        cache_key = hash(input_data)
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = expensive_calculation(input_data)
        self.cache[cache_key] = result
        return result
```

### Batch Processing

```python
from greenlang.utils import BatchProcessor

# Process 10,000 buildings in parallel
processor = BatchProcessor(
    agent=emissions_agent,
    batch_size=500,
    parallel=True,
    num_workers=8
)

results = processor.process_batch(buildings_data)
```

### Lazy Loading

```python
# Lazy load emission factors
class LazyEmissionFactorService:
    def __init__(self):
        self._factors = None

    @property
    def factors(self):
        if self._factors is None:
            self._factors = load_emission_factors()
        return self._factors
```

### Performance Metrics

| Operation | Typical Latency | Optimization |
|-----------|----------------|--------------|
| Agent validation (Pydantic) | 1-2ms | Acceptable overhead |
| Emission calculation | <5ms | Cached factors |
| Provenance tracking | 5-10ms | Disable in perf-critical paths |
| Pipeline execution | 50-200ms | Parallel stages |
| Batch processing (1000 items) | 2-5s | Parallelization |

---

## Security Model

### Zero Trust Architecture

```
Every component is untrusted by default:
├── Agents: Sandboxed execution
├── Packs: Signature verification required
├── Inputs: Validated with Pydantic
├── Artifacts: Signed with Sigstore
└── Network: Restricted by policy
```

### Security Layers

1. **Input Validation**: Pydantic models prevent injection attacks
2. **Pack Verification**: Sigstore signatures ensure authenticity
3. **SBOM Generation**: Track all dependencies for supply chain security
4. **Policy Enforcement**: OPA policies restrict operations
5. **Network Sandboxing**: Packs cannot access arbitrary URLs

### Secrets Management

```python
# ❌ NEVER do this
API_KEY = "hardcoded_secret"

# ✅ DO this
import os
API_KEY = os.getenv("GREENLANG_API_KEY")
if not API_KEY:
    raise ValueError("API key not configured")
```

---

## Deployment Architecture

### Local Development

```
Developer Machine
├── Python 3.10+ (venv)
├── GreenLang CLI (pip install)
├── Local artifacts (./out/)
└── Development profile
```

### Docker Deployment

```
Docker Container
├── Alpine Linux (minimal)
├── Python 3.11
├── GreenLang (baked in)
├── Volume mounts (/data, /config)
└── Production profile
```

### Kubernetes Deployment

```
Kubernetes Cluster
├── Deployment (greenlang-api)
│   ├── Replicas: 3-10 (autoscaling)
│   ├── Resources: 2 CPU, 4GB RAM
│   └── Probes: liveness, readiness
├── Service (LoadBalancer)
├── ConfigMap (emission factors, config)
├── Secrets (API keys, credentials)
└── PersistentVolume (artifacts, provenance)
```

---

## Design Patterns Used

1. **Strategy Pattern**: Agents as interchangeable strategies
2. **Template Method**: `Agent.run()` implements template
3. **Decorator Pattern**: `@traced` adds provenance
4. **Factory Pattern**: `PackLoader` creates agents
5. **Context Object**: `Context` passes data between stages
6. **Chain of Responsibility**: Pipeline stages
7. **Observer Pattern**: Provenance tracking

---

## Future Architecture Plans

**v1.0 (June 2026):**
- Real-time streaming pipelines
- GraphQL API layer
- Multi-tenant SaaS infrastructure
- Advanced caching (Redis, Memcached)

**v2.0 (June 2027):**
- Distributed execution (Spark, Dask)
- ML model serving (TensorFlow Serving)
- Global edge network (Cloudflare Workers)
- Advanced RAG with vector databases

**v3.0 (June 2028):**
- Climate OS platform
- Marketplace for agents and packs
- Federated learning infrastructure
- Blockchain provenance (optional)

---

## References

- [Design Patterns](https://refactoring.guru/design-patterns)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Twelve-Factor App](https://12factor.net/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---

**GreenLang v0.3.0 - The Climate Intelligence Platform**
