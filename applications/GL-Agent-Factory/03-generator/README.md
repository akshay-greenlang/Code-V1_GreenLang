# Agent Generator - Design Documentation

**Version**: 1.0.0
**Status**: Design Complete
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Overview

The Agent Generator is the core engine of the GreenLang Agent Factory. It transforms AgentSpec v2 YAML definitions into production-ready agent packs with complete implementation skeletons, test suites, documentation, and deployment configurations.

**Purpose**: Enable rapid development of GreenLang agents with zero-hallucination guarantees, complete type safety, and production-grade quality.

---

## Document Structure

This directory contains the complete design documentation for the Agent Generator:

### 1. Architecture & Core Design

**File**: `design/00-GENERATOR_ARCHITECTURE.md`

Defines the complete architecture of the generator system:
- High-level component architecture
- Processing pipeline (parse → validate → generate → assemble)
- Template system integration
- Integration with GreenLang infrastructure (calculators, validation, provenance)
- Performance targets and caching strategies
- Security considerations
- Monitoring and observability

**Key Sections**:
- Generator Engine architecture
- Input (AgentSpec YAML) and Output (Agent Pack) specifications
- Code generation pipeline stages
- Integration points with GreenLang calculators and tools
- Error handling and validation checkpoints
- Extensibility (plugins, custom templates)

### 2. Specification to Code Mapping

**File**: `design/01-SPEC_TO_CODE_MAPPING.md`

Comprehensive mapping from AgentSpec v2 sections to generated code:
- Metadata → Module docstrings and class attributes
- Compute → Configuration models and entrypoint functions
- Inputs/Outputs → Pydantic models with validators
- AI section → LLM config, prompt templates, RAG integration
- Tools → Deterministic calculator wrappers
- Factors → Emission factor loading and references
- Realtime → Snapshot management and replay mode
- Provenance → SHA-256 hashing and audit trails

**Key Sections**:
- Complete type mapping (AgentSpec dtype → Python type hints)
- Constraint mapping (ge/le/enum → Pydantic Field parameters)
- Tool wrapper generation (calculator tools vs LLM tools)
- Test generation from spec
- Documentation generation from spec

### 3. Template System

**File**: `templates/00-TEMPLATE_SYSTEM.md`

Design of the Jinja2-based template system:
- Template directory structure (agent_class, tools, tests, graph, deployment, docs)
- Template engine with custom filters (dtype_to_python, snake_case, pascal_case)
- Base templates with inheritance hierarchy
- Template selection strategy (calculator vs LLM vs orchestrator agents)
- Context building from AgentSpec
- Post-generation validation

**Key Sections**:
- Template engine implementation (Jinja2 + custom filters)
- Base agent template with all lifecycle methods
- Calculator tool template (zero-hallucination wrappers)
- Test suite template (unit, integration, determinism tests)
- Documentation templates (README, ARCHITECTURE, API)
- Template context builder

### 4. Generation Workflows

**File**: `workflows/00-GENERATION_WORKFLOWS.md`

Step-by-step workflows for all generator operations:
- **Create Workflow**: Generate new agent pack (11 stages)
- **Update Workflow**: Update existing pack from modified spec
- **Validate Workflow**: Multi-stage spec validation
- **Test Workflow**: Complete test suite execution
- **Publish Workflow**: Package signing and registry upload
- Error handling and recovery strategies
- Progress reporting with Rich CLI
- Dry-run mode for preview

**Key Sections**:
- Complete create workflow with detailed implementation
- Update workflow with diff detection and backup
- Validation stages (YAML, schema, semantics, dependencies, compliance)
- Test execution with coverage reporting
- Publishing pipeline with SBOM generation and signing

### 5. CLI Specification

**File**: `tooling/00-CLI_SPECIFICATION.md`

Complete CLI interface design using Typer:
- `gl agent create`: Generate new agent pack
- `gl agent update`: Update existing pack
- `gl agent validate`: Validate AgentSpec
- `gl agent test`: Run test suite
- `gl agent publish`: Publish to registry
- `gl agent list`: List available agents
- `gl agent info`: Show agent details
- `gl agent init`: Interactive spec template creation

**Key Sections**:
- Command signatures with arguments and options
- Implementation examples with Typer
- Rich CLI formatting (progress bars, tables, colors)
- Interactive prompts for `init` command
- Configuration file format
- Shell completion support

---

## Quick Reference

### Generator Pipeline

```
AgentSpec YAML
     │
     ▼
┌─────────────────┐
│ 1. Parse & Val  │  Load YAML, validate schema
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Build Context│  Create template context
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Select Temps │  Choose templates by agent type
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Generate Code│  Render all templates
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Validate Code│  Syntax, types, quality
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. Assemble Pack│  Write to disk
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. Run Tests    │  Execute generated tests
└────────┬────────┘
         ▼
Complete Agent Pack
```

### Generated Pack Structure

```
GL-XXX-AgentName/
├── pack.yaml                  # AgentSpec v2 (copied)
├── README.md                  # Auto-generated docs
├── requirements.txt           # Python dependencies
├── agent_name/
│   ├── agent.py               # Main agent class
│   ├── tools.py               # Tool wrappers
│   ├── prompts.py             # Prompt templates
│   └── config.py              # Configuration
├── tests/
│   ├── test_agent.py          # Unit tests
│   ├── test_integration.py    # Integration tests
│   └── test_determinism.py    # Determinism tests
├── graph/
│   └── agent_graph.yaml       # LangGraph config
├── deployment/
│   ├── Dockerfile
│   └── kubernetes/
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    └── EXAMPLES.md
```

### CLI Commands

```bash
# Create new agent
gl agent create specs/fuel_agent.yaml

# Validate spec
gl agent validate specs/fuel_agent.yaml --strict

# Update existing agent
gl agent update agents/fuel_agent --spec specs/fuel_v2.yaml

# Run tests
gl agent test agents/fuel_agent --coverage

# Publish to registry
gl agent publish agents/fuel_agent

# Interactive init
gl agent init
```

### Key Design Principles

1. **Zero Hallucination**: All generated code uses deterministic calculators
2. **Type Safety**: 100% type hints, Pydantic models, Mypy validation
3. **Production Quality**: 85%+ test coverage, comprehensive error handling
4. **Maintainability**: Clear code, complete docstrings, self-documenting
5. **Provenance**: SHA-256 hashing, complete audit trails
6. **Determinism**: Reproducible results (temperature=0.0, seed=42)
7. **Compliance**: GreenLang standards, regulatory frameworks

---

## Implementation Roadmap

### Phase 1: Core Generator (Weeks 1-2)
- [ ] Implement generator engine (`generator_engine.py`)
- [ ] Implement template engine with custom filters (`template_engine.py`)
- [ ] Implement spec parser and validator (`yaml_parser.py`, `spec_validator.py`)
- [ ] Create base templates (agent_class, tools, tests)

### Phase 2: Code Generation (Weeks 3-4)
- [ ] Implement code generators (`agent_class_gen.py`, `tool_wrapper_gen.py`)
- [ ] Implement test generator (`test_gen.py`)
- [ ] Implement documentation generator (`doc_gen.py`)
- [ ] Implement deployment generator (`deployment_gen.py`)

### Phase 3: CLI (Week 5)
- [ ] Implement CLI commands with Typer (`cli/commands.py`)
- [ ] Implement rich progress reporting
- [ ] Implement interactive prompts (`init` command)
- [ ] Add shell completion support

### Phase 4: Workflows (Week 6)
- [ ] Implement create workflow
- [ ] Implement update workflow with diff detection
- [ ] Implement validate workflow
- [ ] Implement test workflow
- [ ] Implement publish workflow

### Phase 5: Testing & Documentation (Week 7)
- [ ] Write unit tests for generator components (85%+ coverage)
- [ ] Write integration tests (end-to-end generation)
- [ ] Create example AgentSpecs for testing
- [ ] Write user guide and tutorials

### Phase 6: Polish & Release (Week 8)
- [ ] Performance optimization (caching, parallel generation)
- [ ] Error message improvements
- [ ] CLI UX refinements
- [ ] Beta release and feedback collection

---

## Success Metrics

### Code Quality Metrics
- **Type Coverage**: 100% (all methods have type hints)
- **Docstring Coverage**: 100% (all public methods documented)
- **Test Coverage**: 85%+ (unit + integration tests)
- **Linting**: Passes Ruff with zero errors
- **Type Checking**: Passes Mypy with zero errors

### Generation Quality Metrics
- **Success Rate**: >99% (valid specs generate successfully)
- **Error Clarity**: 100% (all errors have actionable messages)
- **Template Coverage**: 100% (all spec sections have templates)
- **Generated Code Quality**: Passes same standards as hand-written code

### Performance Metrics
- **Spec Parsing**: <100ms
- **Validation**: <200ms
- **Code Generation**: <500ms per template
- **Complete Pack Generation**: <5 seconds
- **Test Execution**: <30 seconds

---

## Dependencies

### Core Dependencies
- **Python**: >=3.11
- **Jinja2**: >=3.1.0 (template engine)
- **Typer**: >=0.9.0 (CLI framework)
- **Rich**: >=13.0.0 (CLI formatting)
- **Pydantic**: >=2.0.0 (validation)
- **PyYAML**: >=6.0.0 (YAML parsing)

### Development Dependencies
- **Pytest**: >=7.0.0 (testing)
- **Mypy**: >=1.0.0 (type checking)
- **Ruff**: >=0.1.0 (linting)
- **Coverage**: >=7.0.0 (coverage reporting)

---

## References

### AgentSpec v2 Documentation
- `C:\Users\aksha\Code-V1_GreenLang\AGENTSPEC_V2_FOUNDATION_GUIDE.md`
- `C:\Users\aksha\Code-V1_GreenLang\AGENTSPEC_V2_MIGRATION_GUIDE.md`

### Example Packs
- `C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-001\pack.yaml`
- `C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-002\pack.yaml`

### GreenLang Infrastructure
- **Calculators**: `greenlang.calculators.*`
- **Validation**: `greenlang.validation.*`
- **Provenance**: `greenlang.provenance.*`
- **Emission Factors**: `greenlang.emission_factors.*`

---

## Support

For questions or issues:
- **Design Review**: GL Backend Developer
- **Implementation**: GreenLang Engineering Team
- **Documentation**: See individual design documents in this directory

---

## Changelog

### 2025-12-03 - Initial Design
- Created complete design documentation
- Defined architecture, mappings, templates, workflows, and CLI
- Ready for implementation

---

**Status**: ✅ Design Complete - Ready for Implementation
**Next Step**: Begin Phase 1 implementation (Core Generator)
