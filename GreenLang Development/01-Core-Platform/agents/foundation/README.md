# GreenLang Foundation Layer Agents

The Foundation Layer provides the core infrastructure that all other GreenLang agents depend on. These 10 agents form the "operating system" of the GreenLang Climate OS.

## Agent Catalog

| Agent ID | Name | Description | Status |
|----------|------|-------------|--------|
| GL-FOUND-X-001 | GreenLang Orchestrator | DAG execution engine for multi-agent pipelines | ✅ Complete |
| GL-FOUND-X-002 | Schema Compiler & Validator | Validates input payloads against schemas | 🔄 Building |
| GL-FOUND-X-003 | Unit & Reference Normalizer | Converts and normalizes units | 🔄 Building |
| GL-FOUND-X-004 | Assumptions Registry | Version-controlled assumption management | 🔄 Building |
| GL-FOUND-X-005 | Citations & Evidence Agent | Tracks data provenance and citations | 🔄 Building |
| GL-FOUND-X-006 | Access & Policy Guard | Authorization and policy enforcement | 🔄 Building |
| GL-FOUND-X-007 | Versioned Agent Registry | Catalogs all available agents | 🔄 Building |
| GL-FOUND-X-008 | Run Reproducibility Agent | Ensures deterministic execution | 🔄 Building |
| GL-FOUND-X-009 | QA Test Harness Agent | Testing framework for agents | 🔄 Building |
| GL-FOUND-X-010 | Observability Agent | Metrics, logging, and tracing | 🔄 Building |

## Zero-Hallucination Guarantees

All Foundation Layer agents enforce the GreenLang zero-hallucination principles:

1. **Complete Lineage** - Every output has traceable inputs
2. **Deterministic Execution** - Same inputs always produce same outputs
3. **Citation Required** - All data sources are attributed
4. **Assumption Tracking** - All assumptions are versioned and logged
5. **Audit Trail** - Complete history of all operations

## Usage

```python
from greenlang.agents.foundation import (
    GreenLangOrchestrator,
    SchemaCompiler,
    UnitNormalizer,
    AssumptionsRegistry,
    CitationsAgent,
    PolicyGuard,
    AgentRegistry,
    ReproducibilityAgent,
    QATestHarness,
    ObservabilityAgent,
)

# Create orchestrator
orchestrator = GreenLangOrchestrator()

# Register other foundation agents
orchestrator.register_agent("GL-FOUND-X-002", SchemaCompiler)
orchestrator.register_agent("GL-FOUND-X-003", UnitNormalizer)
# ... register all agents

# Execute a pipeline
result = await orchestrator.execute_pipeline(dag_definition)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Foundation Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐│
│  │  Orchestrator   │───▶│  Agent Registry │───▶│ Policy Guard ││
│  │  (GL-FOUND-001) │    │  (GL-FOUND-007) │    │ (GL-FOUND-006││
│  └─────────────────┘    └─────────────────┘    └──────────────┘│
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐│
│  │ Schema Compiler │───▶│ Unit Normalizer │───▶│  Citations   ││
│  │  (GL-FOUND-002) │    │  (GL-FOUND-003) │    │ (GL-FOUND-005││
│  └─────────────────┘    └─────────────────┘    └──────────────┘│
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐│
│  │  Assumptions    │───▶│ Reproducibility │───▶│ Observability││
│  │  (GL-FOUND-004) │    │  (GL-FOUND-008) │    │ (GL-FOUND-010││
│  └─────────────────┘    └─────────────────┘    └──────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    QA Test Harness (GL-FOUND-009)           ││
│  │                    Tests all agents above                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies

Each Foundation agent has specific dependencies:

- **GL-FOUND-X-001** (Orchestrator): No deps - root agent
- **GL-FOUND-X-002** (Schema): No deps
- **GL-FOUND-X-003** (Units): Depends on GL-FOUND-X-002
- **GL-FOUND-X-004** (Assumptions): No deps
- **GL-FOUND-X-005** (Citations): No deps
- **GL-FOUND-X-006** (Policy): Depends on GL-FOUND-X-007
- **GL-FOUND-X-007** (Registry): No deps
- **GL-FOUND-X-008** (Reproducibility): Depends on GL-FOUND-X-001
- **GL-FOUND-X-009** (QA): Depends on all above
- **GL-FOUND-X-010** (Observability): No deps

## Testing

Run all Foundation Layer tests:

```bash
pytest tests/agents/foundation/ -v
```

Run specific agent tests:

```bash
pytest tests/agents/foundation/test_orchestrator.py -v
pytest tests/agents/foundation/test_schema_compiler.py -v
```

## Configuration

Foundation agents are configured via the Agent Factory:

```yaml
# agent-factory-config.yaml
foundation:
  orchestrator:
    max_parallel: 10
    default_timeout: 300
    checkpoint_interval: 10
  schema_compiler:
    strict_mode: true
    coerce_types: true
  unit_normalizer:
    default_unit_system: "SI"
    allow_custom_factors: true
```
