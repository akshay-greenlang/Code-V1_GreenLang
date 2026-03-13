# AGENT-EUDR-026: Due Diligence Orchestrator -- Technical Architecture Specification

## Document Info

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-AGENT-EUDR-026 |
| **Agent ID** | GL-EUDR-DDO-026 |
| **Component** | Due Diligence Orchestrator Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence Workflow Orchestration |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Architecture Specification |
| **Author** | GL-AppArchitect |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EUDR, Articles 4, 8, 9, 10, 11, 12, 13; ISO 19011:2018 Auditing Management Systems |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **DB Migration** | V114 |
| **Metric Prefix** | `gl_eudr_ddo_` |
| **Config Prefix** | `GL_EUDR_DDO_` |
| **API Prefix** | `/v1/eudr-ddo` |
| **RBAC Prefix** | `eudr-ddo:` |

---

## 1. Executive Summary

### 1.1 Purpose

AGENT-EUDR-026 Due Diligence Orchestrator is the central workflow orchestration engine for end-to-end EUDR due diligence. It coordinates the execution of all 25 upstream agents (Supply Chain Traceability EUDR-001 through EUDR-015, Risk Assessment EUDR-016 through EUDR-025) in a DAG-based topology across three mandatory phases defined by EUDR Article 8: information gathering (Article 9), risk assessment (Article 10), and risk mitigation (Article 11). The agent manages workflow state with persistent checkpointing, enforces quality gates between phases, optimizes execution through intelligent parallelization, handles errors with exponential backoff and circuit breakers, and produces audit-ready due diligence packages for DDS submission under Articles 12-13.

It is the 26th agent in the EUDR agent family and establishes the Due Diligence Workflow Orchestration sub-category.

### 1.2 Regulatory Driver

EUDR Article 8(1) mandates that operators establish and implement a due diligence system comprising three sequential phases: information gathering, risk assessment, and risk mitigation. Article 4(1) requires a Due Diligence Statement (DDS) before placing products on the EU market. Articles 12(2)(a-j) define the mandatory DDS content requirements. Article 13 provides for simplified due diligence for low-risk country origins. Article 31(1) requires 5-year record retention. Non-compliance exposes operators to penalties of up to 4% of annual EU turnover under Articles 22-23.

### 1.3 Key Differentiators

- **25-agent native orchestration** -- DAG-based coordination of all 25 purpose-built EUDR agents in a single workflow
- **Three regulatory quality gates** -- Article-specific validation between each mandatory phase (Art. 9, Art. 10, Art. 11)
- **Seven commodity workflow templates** -- Pre-configured DAG topologies for cattle, cocoa, coffee, palm oil, rubber, soya, and wood
- **Checkpoint and resume** -- Persistent workflow state with sub-second resume from any checkpoint
- **DAG-aware parallelization** -- Topological sort with work-stealing scheduler achieving 3-5x speedup
- **Circuit breaker resilience** -- Per-agent circuit breaker with exponential backoff and graceful degradation
- **Audit-ready package generation** -- DDS-compatible evidence bundles with SHA-256 provenance chain
- **Zero-hallucination guarantee** -- All workflow logic, scoring, and decisions use deterministic algorithms; no LLM in execution path

### 1.4 Performance Targets

| Metric | Target |
|--------|--------|
| Standard workflow end-to-end (1,000 shipments) | < 5 minutes |
| Simplified workflow end-to-end (low-risk) | < 2 minutes |
| Workflow creation and DAG validation | < 500ms |
| Checkpoint write latency | < 500ms (async) |
| Checkpoint resume latency | < 2 seconds |
| Quality gate evaluation (10,000 shipments) | < 5 seconds |
| Package generation (JSON, 1,000 shipments) | < 10 seconds |
| Package generation (PDF, full evidence) | < 30 seconds |
| Batch workflow creation (100 workflows) | < 5 seconds |
| API response time (read, p95) | < 200ms |
| API response time (write, p95) | < 500ms |
| Concurrent workflows | 1,000+ simultaneous |
| Parallelization efficiency | >= 70% CPU utilization |
| Error recovery success rate | >= 95% for transient failures |
| Orchestration overhead | < 5% of total workflow time |
| Determinism | 100% bit-perfect reproducibility |

### 1.5 Development Estimates

| Phase | Scope | Duration | Engineers |
|-------|-------|----------|-----------|
| Phase 1 | Core engines (E1, E6, E7, E8): Workflow Definition, State Manager, Parallel Execution, Error Recovery + Agent Client | 4 weeks | 2 |
| Phase 2 | Phase coordinators (E2, E3, E4) + Quality Gate Engine (E5): Information Gathering, Risk Assessment, Risk Mitigation coordinators | 3 weeks | 2 |
| Phase 3 | Package Generator (E9), API routes, auth integration, reporting | 3 weeks | 2 |
| Phase 4 | DB migration V114, Grafana dashboard, integration testing with all 25 agents | 1 week | 2 |
| Phase 5 | Testing (unit + integration + golden + chaos + performance), security audit | 3 weeks | 2 |
| **Total** | **Complete agent** | **14 weeks** | **2 engineers** |

### 1.6 Estimated Output

- ~50 files (agent code + API + workflow templates + reference data)
- ~45K lines of code
- ~1,040+ tests
- V114 database migration (9 regular tables + 4 TimescaleDB hypertables)
- 1 Grafana dashboard (20 panels)

---

## 2. System Architecture Overview

### 2.1 Component Diagram

```
+-----------------------------------------------------------------------------------+
|                              GL-EUDR-APP v1.0 (Frontend)                          |
|         [Workflow Dashboard] [Progress Tracker] [Package Viewer]                  |
+--------------------------------------+--------------------------------------------+
                                       |
+--------------------------------------v--------------------------------------------+
|                           Unified API Layer (FastAPI)                              |
|                     /v1/eudr-ddo  (~30 endpoints)                                 |
+---+------+------+------+------+------+------+------+------+------+----------------+
    |      |      |      |      |      |      |      |      |      |
    v      v      v      v      v      v      v      v      v      v
+------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
|Wflow | |Info  | |Risk  | |Mitig | |Qual  | |State | |Parll | |Error | |Pkg   |
|Defn  | |Gathr | |Assmnt| |Coord | |Gate  | |Mngr  | |Exec  | |Recov | |Gen   |
|(E1)  | |(E2)  | |(E3)  | |(E4)  | |(E5)  | |(E6)  | |(E7)  | |(E8)  | |(E9)  |
+--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+
   |        |        |        |        |        |        |        |        |
   +--------+--------+--------+--------+--------+--------+--------+--------+
                                       |
                       +---------------+----------------+
                       |               |                |
             +---------v--+   +--------v---+   +--------v-------+
             | PostgreSQL |   |   Redis    |   |  S3 / MinIO    |
             | TimescaleDB|   |  Cache     |   | Agent Outputs  |
             | (13 tables)|   | Streams    |   | Evidence Store |
             +------------+   | PubSub     |   | Report Store   |
                              +------------+   +----------------+

+-----------------------------------------------------------------------------------+
|                Phase 1: Supply Chain Traceability (EUDR-001 to 015)                |
|                                                                                   |
| Layer 0: EUDR-001 Supply Chain Mapping Master                                     |
| Layer 1: EUDR-002 GeoVer | EUDR-006 PlotBnd | EUDR-007 GPSVal | EUDR-008 MultiTr|
| Layer 2: EUDR-003 SatMon | EUDR-004 Forest  | EUDR-005 LandUse                  |
| Layer 3: EUDR-009 CoC    | EUDR-010 Segreg  | EUDR-011 MassBal                  |
| Layer 4: EUDR-012 DocAuth| EUDR-013 BlockCh | EUDR-014 QRCode | EUDR-015 Mobile  |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                Phase 2: Risk Assessment (EUDR-016 to 025)                         |
|                                                                                   |
| Layer 5: EUDR-016 Country | EUDR-017 Supplier | EUDR-018 Commodity               |
|          EUDR-019 Corrupt | EUDR-020 Deforest | EUDR-021 Indigenous               |
|          EUDR-022 ProtArea| EUDR-023 Legal    | EUDR-024 Audit                    |
|          EUDR-025 RiskMit (assessment inputs)                                     |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                Phase 3: Risk Mitigation (EUDR-025)                                |
| Layer 6: EUDR-025 Risk Mitigation Advisor (mitigation outputs)                    |
+-----------------------------------------------------------------------------------+
```

### 2.2 Engine Interaction Flow

```
                    Operator initiates workflow
                              |
                              v
                   +---------------------+
                   | E1: Workflow Defn   |-----> DAG topology, dependency edges,
                   |     Engine          |       commodity template selection
                   +--------+------------+
                            |
                            v
                   +---------------------+
                   | E7: Parallel Exec   |-----> Concurrency pool, work-stealing,
                   |     Engine          |       critical path, ETA calculation
                   +--------+------------+
                            |
              +-------------+-------------+
              v             v             v
   +----------------+ +----------------+ +-------------------+
   | E2: Info Gathr | | E3: Risk Assmnt| | E4: Risk Mitigat  |
   |   Coordinator  | |   Coordinator  | |   Coordinator     |
   |  (Phase 1)     | |  (Phase 2)     | |  (Phase 3)        |
   | EUDR-001-015   | | EUDR-016-025   | | EUDR-025 mitigation|
   +-------+--------+ +-------+--------+ +--------+----------+
           |                   |                    |
           v                   v                    v
   +---------------------+---------------------+-----------+
   |                E5: Quality Gate Engine                 |
   |  QG-1: Info Completeness (>= 90%)                     |
   |  QG-2: Risk Assessment Coverage (>= 95%)              |
   |  QG-3: Mitigation Adequacy (residual <= 15)           |
   +-------------------------+-----------------------------+
                             |
                             v
                   +---------------------+
                   | E6: Workflow State  |-----> Checkpoints, resume,
                   |     Manager        |       rollback, audit trail,
                   |                    |       SHA-256 provenance
                   +--------+------------+
                            |
                   +--------v------------+
                   | E8: Error Recovery |-----> Exponential backoff,
                   |     Manager        |       circuit breaker, DLQ,
                   |                    |       fallback strategies
                   +--------+------------+
                            |
                            v
                   +---------------------+
                   | E9: DD Package     |-----> DDS JSON, PDF report,
                   |     Generator      |       multi-language output,
                   |                    |       SHA-256 integrity hash
                   +---------------------+
```

---

## 3. Module Structure

### 3.1 Directory Layout

```
greenlang/agents/eudr/due_diligence_orchestrator/
    __init__.py                              # Public API exports
    config.py                                # GL_EUDR_DDO_ env prefix configuration
    models.py                                # Pydantic v2 models (~50 models)
    metrics.py                               # 20 Prometheus metrics (gl_eudr_ddo_ prefix)
    provenance.py                            # SHA-256 provenance tracking
    setup.py                                 # DueDiligenceOrchestratorService facade
    agent_client.py                          # Unified interface for invoking all 25 agents
    #
    # === 9 Processing Engines ===
    #
    workflow_definition_engine.py             # Engine 1: DAG creation, validation, templates
    information_gathering_coordinator.py      # Engine 2: Phase 1 orchestration (EUDR-001-015)
    risk_assessment_coordinator.py            # Engine 3: Phase 2 orchestration (EUDR-016-025)
    risk_mitigation_coordinator.py            # Engine 4: Phase 3 orchestration (EUDR-025)
    quality_gate_engine.py                    # Engine 5: Phase transition validation
    workflow_state_manager.py                 # Engine 6: Checkpoint, resume, rollback, audit
    parallel_execution_engine.py             # Engine 7: Concurrent agent execution, work-stealing
    error_recovery_manager.py                # Engine 8: Retry, circuit breaker, fallback, DLQ
    due_diligence_package_generator.py       # Engine 9: DDS JSON + PDF report generation
    #
    # === Workflow Templates ===
    #
    workflow_templates/
        __init__.py
        standard_workflow.py                 # Full 25-agent standard due diligence DAG
        simplified_workflow.py               # Reduced Article 13 simplified due diligence DAG
        cattle_workflow.py                   # Cattle-specific workflow template
        cocoa_workflow.py                    # Cocoa-specific workflow template
        coffee_workflow.py                   # Coffee-specific workflow template
        palm_oil_workflow.py                 # Palm oil-specific workflow template
        rubber_workflow.py                   # Rubber-specific workflow template
        soya_workflow.py                     # Soya-specific workflow template
        wood_workflow.py                     # Wood-specific workflow template
    #
    # === Reference Data ===
    #
    reference_data/
        __init__.py
        quality_gate_criteria.py             # Quality gate check definitions and thresholds
        dds_field_mapping.py                 # Article 12(2) DDS field to agent output mapping
        commodity_agent_topology.py          # Commodity-specific agent inclusion/exclusion rules
        error_classification_rules.py        # Error type to classification mapping
        risk_weight_defaults.py              # Default composite risk score weights
    #
    # === API Layer ===
    #
    api/
        __init__.py
        router.py                            # Main router (/v1/eudr-ddo), sub-router aggregation
        dependencies.py                      # FastAPI dependencies (service injection, auth)
        schemas.py                           # API-specific request/response schemas
        workflow_routes.py                   # Workflow CRUD and execution (8 endpoints)
        template_routes.py                   # Workflow template management (4 endpoints)
        phase_routes.py                      # Phase coordination (3 endpoints)
        quality_gate_routes.py               # Quality gate management (4 endpoints)
        state_routes.py                      # Workflow state and checkpoints (4 endpoints)
        execution_routes.py                  # Parallel execution monitoring (3 endpoints)
        package_routes.py                    # Package generation and download (4 endpoints)
        batch_routes.py                      # Batch operations (2 endpoints)
        ops_routes.py                        # Circuit breakers, DLQ, health (5 endpoints)
```

### 3.2 Test Directory Layout

```
tests/agents/eudr/due_diligence_orchestrator/
    __init__.py
    conftest.py                              # Shared fixtures, mock agents, test data
    test_workflow_definition_engine.py       # Engine 1: 100+ tests
    test_information_gathering_coordinator.py # Engine 2: 80+ tests
    test_risk_assessment_coordinator.py      # Engine 3: 80+ tests
    test_risk_mitigation_coordinator.py      # Engine 4: 60+ tests
    test_quality_gate_engine.py             # Engine 5: 100+ tests
    test_workflow_state_manager.py          # Engine 6: 120+ tests
    test_parallel_execution_engine.py       # Engine 7: 80+ tests
    test_error_recovery_manager.py          # Engine 8: 100+ tests
    test_package_generator.py              # Engine 9: 80+ tests
    test_agent_client.py                   # Agent client: 50+ tests
    test_api_routes.py                     # API layer: 100+ tests
    test_models.py                         # Pydantic models: 40+ tests
    test_config.py                         # Configuration: 15+ tests
    test_provenance.py                     # Provenance chain: 20+ tests
    test_golden_workflows.py              # 49 golden test scenarios (7 commodities x 7 scenarios)
    test_integration.py                   # 40+ integration tests
    test_chaos.py                         # 20+ chaos/resilience tests
    test_performance.py                   # 30+ performance benchmark tests
    test_determinism.py                   # 15+ bit-perfect reproducibility tests
    test_security.py                      # 10+ security and RBAC tests
```

### 3.3 Deployment Artifacts

```
deployment/
    database/migrations/sql/
        V114__agent_eudr_due_diligence_orchestrator.sql
    monitoring/dashboards/
        eudr-due-diligence-orchestrator.json
```

---

## 4. Engine Specifications

### 4.1 Engine 1: Workflow Definition Engine

**File:** `workflow_definition_engine.py`
**Purpose:** DAG-based workflow creation, validation, topological sorting, dependency resolution, and commodity-specific template management for due diligence workflows.

**Responsibilities:**
- Define workflows as directed acyclic graphs (DAGs) with up to 50 agent nodes and configurable dependency edges
- Resolve execution order using topological sorting (Kahn's algorithm)
- Detect circular dependencies and reject invalid workflow definitions
- Provide pre-built workflow templates for all 7 EUDR commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood)
- Support standard due diligence workflow (full 25-agent topology) and simplified due diligence workflow (reduced topology per Article 13)
- Support runtime workflow modification: add/remove agents, modify dependencies
- Validate workflow definitions against a schema before execution
- Support workflow versioning with immutable version history
- Support workflow cloning for template customization
- Calculate critical path through the DAG for ETA estimation

**DAG Construction and Topological Sort:**

```python
import networkx as nx
from typing import List, Dict, Tuple

class WorkflowDefinitionEngine:
    """DAG-based workflow definition and validation."""

    def build_dag(self, workflow_def: WorkflowDefinition) -> nx.DiGraph:
        """
        Construct a NetworkX DiGraph from workflow definition.
        Validates acyclicity and connectivity.
        """
        dag = nx.DiGraph()

        # Add agent nodes with metadata
        for node in workflow_def.agent_nodes:
            dag.add_node(
                node.agent_id,
                agent_name=node.agent_name,
                phase=node.phase,
                layer=node.layer,
                is_critical=node.is_critical,
                is_required=node.is_required,
                timeout_seconds=node.timeout_seconds,
            )

        # Add quality gate nodes
        for gate in workflow_def.quality_gates:
            dag.add_node(
                gate.gate_id,
                gate_name=gate.gate_name,
                phase_before=gate.phase_before,
                phase_after=gate.phase_after,
                threshold=gate.threshold,
                is_gate=True,
            )

        # Add dependency edges
        for edge in workflow_def.dependency_edges:
            dag.add_edge(
                edge.source_agent_id,
                edge.target_agent_id,
                dependency_type=edge.dependency_type,
            )

        # Validate DAG properties
        if not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            raise WorkflowValidationError(
                f"Circular dependency detected: {cycles}"
            )

        return dag

    def topological_sort(self, dag: nx.DiGraph) -> List[List[str]]:
        """
        Layered topological sort using Kahn's algorithm.
        Returns list of layers, each layer containing independent agents
        that can execute concurrently.
        """
        in_degree = dict(dag.in_degree())
        layers = []
        remaining = set(dag.nodes())

        while remaining:
            # Find all nodes with in-degree 0 among remaining
            ready = [
                n for n in remaining
                if in_degree.get(n, 0) == 0
            ]
            if not ready:
                raise WorkflowValidationError(
                    "Deadlock detected: no executable agents"
                )

            layers.append(sorted(ready))  # Sort for determinism

            # Remove ready nodes and update in-degrees
            for node in ready:
                remaining.remove(node)
                for successor in dag.successors(node):
                    if successor in remaining:
                        in_degree[successor] -= 1

        return layers

    def calculate_critical_path(
        self, dag: nx.DiGraph, estimated_durations: Dict[str, int]
    ) -> Tuple[List[str], int]:
        """
        Calculate critical path through DAG for ETA estimation.
        Uses longest-path algorithm on the DAG.
        Returns (critical_path_nodes, total_duration_seconds).
        """
        # Assign weights as estimated durations
        for node in dag.nodes():
            dag.nodes[node]["weight"] = estimated_durations.get(node, 30)

        critical_path = nx.dag_longest_path(dag, weight="weight")
        critical_duration = nx.dag_longest_path_length(dag, weight="weight")

        return critical_path, critical_duration
```

**Standard Workflow DAG Template (25 agents + 3 gates):**

```python
STANDARD_WORKFLOW_EDGES = [
    # Layer 0 -> Layer 1 (EUDR-001 feeds geospatial + supplier agents)
    ("EUDR-001", "EUDR-002"), ("EUDR-001", "EUDR-006"),
    ("EUDR-001", "EUDR-007"), ("EUDR-001", "EUDR-008"),
    # Layer 1 -> Layer 2 (geospatial feeds satellite/forest/land)
    ("EUDR-002", "EUDR-003"), ("EUDR-006", "EUDR-004"),
    ("EUDR-006", "EUDR-005"),
    # Layer 1 -> Layer 3 (supplier feeds chain of custody)
    ("EUDR-008", "EUDR-009"),
    ("EUDR-009", "EUDR-010"), ("EUDR-009", "EUDR-011"),
    # Layers 2-3 -> Layer 4 (evidence & traceability)
    ("EUDR-003", "EUDR-012"), ("EUDR-004", "EUDR-012"),
    ("EUDR-010", "EUDR-013"), ("EUDR-011", "EUDR-014"),
    ("EUDR-005", "EUDR-015"),
    # Layer 4 -> QG-1
    ("EUDR-012", "QG-1"), ("EUDR-013", "QG-1"),
    ("EUDR-014", "QG-1"), ("EUDR-015", "QG-1"),
    ("EUDR-003", "QG-1"), ("EUDR-004", "QG-1"),
    ("EUDR-005", "QG-1"), ("EUDR-007", "QG-1"),
    ("EUDR-010", "QG-1"), ("EUDR-011", "QG-1"),
    # QG-1 -> Layer 5 (all risk agents in parallel)
    ("QG-1", "EUDR-016"), ("QG-1", "EUDR-017"),
    ("QG-1", "EUDR-018"), ("QG-1", "EUDR-019"),
    ("QG-1", "EUDR-020"), ("QG-1", "EUDR-021"),
    ("QG-1", "EUDR-022"), ("QG-1", "EUDR-023"),
    ("QG-1", "EUDR-024"), ("QG-1", "EUDR-025"),
    # Layer 5 -> QG-2
    ("EUDR-016", "QG-2"), ("EUDR-017", "QG-2"),
    ("EUDR-018", "QG-2"), ("EUDR-019", "QG-2"),
    ("EUDR-020", "QG-2"), ("EUDR-021", "QG-2"),
    ("EUDR-022", "QG-2"), ("EUDR-023", "QG-2"),
    ("EUDR-024", "QG-2"), ("EUDR-025", "QG-2"),
    # QG-2 -> Layer 6 (mitigation)
    ("QG-2", "EUDR-025-MIT"),
    # Layer 6 -> QG-3
    ("EUDR-025-MIT", "QG-3"),
    # QG-3 -> Package Generation
    ("QG-3", "PKG-GEN"),
]
```

**Simplified Workflow Template (Article 13):**

```python
SIMPLIFIED_WORKFLOW_AGENTS = [
    "EUDR-001", "EUDR-002", "EUDR-007",  # Reduced geospatial
    "EUDR-003",                            # Satellite (reduced scope)
    "EUDR-016", "EUDR-018", "EUDR-023",  # 3 risk agents only
]

SIMPLIFIED_QUALITY_GATES = {
    "QG-1": {"threshold": 80.0},  # Relaxed from 90%
    "QG-2": {"threshold": 85.0},  # Relaxed from 95%
    # QG-3 skipped for simplified workflow (mitigation not required)
}
```

**Zero-Hallucination Guarantee:** Topological sort is deterministic (Kahn's algorithm with sorted tie-breaking). Critical path uses longest-path on DAG. No LLM in workflow definition or validation. Same definition always produces same execution plan.

**Estimated Lines of Code:** 4,500-5,000

---

### 4.2 Engine 2: Information Gathering Coordinator

**File:** `information_gathering_coordinator.py`
**Purpose:** Orchestrates Phase 1 execution of EUDR-001 through EUDR-015 in the correct dependency order for comprehensive supply chain data collection per Article 9.

**Responsibilities:**
- Invoke EUDR-001 Supply Chain Mapping Master as the entry point for all workflows
- Launch EUDR-002, EUDR-006, EUDR-007, EUDR-008 in parallel after EUDR-001 completes
- Launch EUDR-003, EUDR-004, EUDR-005 after Layer 1 geospatial agents complete
- Launch EUDR-009, EUDR-010, EUDR-011 after Layer 1 supplier agents complete
- Launch EUDR-012, EUDR-013, EUDR-014, EUDR-015 after Layers 2-3 complete
- Manage data handoff between dependent agents via output mapping
- Track individual agent status: PENDING, QUEUED, RUNNING, COMPLETED, FAILED, SKIPPED
- Calculate information gathering completeness score (0-100) in real time
- Validate all Article 9 required fields before triggering Quality Gate 1
- Support partial execution: skip non-applicable agents for specific commodities

**Phase 1 Orchestration Pipeline:**

```python
from decimal import Decimal, ROUND_HALF_UP

class InformationGatheringCoordinator:
    """Phase 1: Article 9 information gathering orchestration."""

    async def execute_phase(
        self,
        workflow: WorkflowExecution,
        dag_layers: List[List[str]],
        agent_client: AgentClient,
        parallel_engine: ParallelExecutionEngine,
        state_manager: WorkflowStateManager,
    ) -> PhaseResult:
        """
        Execute Phase 1 layers sequentially, agents within layers concurrently.
        Checkpoints after each agent completion.
        """
        phase_agents = [
            agent_id for layer in dag_layers
            for agent_id in layer
            if self._is_phase1_agent(agent_id)
        ]

        for layer in dag_layers:
            layer_agents = [a for a in layer if a in phase_agents]
            if not layer_agents:
                continue

            # Prepare inputs for each agent from upstream outputs
            agent_inputs = {}
            for agent_id in layer_agents:
                agent_inputs[agent_id] = self._build_agent_input(
                    agent_id, workflow.agent_outputs
                )

            # Execute layer agents concurrently
            results = await parallel_engine.execute_layer(
                workflow_id=workflow.workflow_id,
                agents=layer_agents,
                inputs=agent_inputs,
                agent_client=agent_client,
                max_concurrency=workflow.config.max_concurrency,
            )

            # Process results and checkpoint
            for agent_id, result in results.items():
                workflow.agent_outputs[agent_id] = result.output_summary
                workflow.agent_statuses[agent_id] = result.status
                await state_manager.checkpoint(
                    workflow, agent_id=agent_id,
                    checkpoint_type="agent_complete"
                )

        # Calculate completeness score
        completeness = self.calculate_completeness(workflow)
        return PhaseResult(
            phase="information_gathering",
            completeness_score=completeness,
            agents_completed=len([
                a for a in phase_agents
                if workflow.agent_statuses.get(a) == "completed"
            ]),
            agents_total=len(phase_agents),
        )

    def calculate_completeness(
        self, workflow: WorkflowExecution
    ) -> Decimal:
        """
        Deterministic completeness calculation based on Art. 9 field coverage.
        Uses fixed weights for each field category.
        """
        FIELD_WEIGHTS = {
            "product_description": Decimal("10"),
            "quantity_data": Decimal("10"),
            "country_of_production": Decimal("10"),
            "plot_geolocation": Decimal("20"),
            "polygon_coverage": Decimal("10"),
            "custody_chain": Decimal("15"),
            "satellite_verification": Decimal("15"),
            "document_authentication": Decimal("10"),
        }

        total_weight = sum(FIELD_WEIGHTS.values())
        achieved = Decimal("0")

        outputs = workflow.agent_outputs
        if outputs.get("EUDR-001", {}).get("product_count", 0) > 0:
            achieved += FIELD_WEIGHTS["product_description"]
        if outputs.get("EUDR-001", {}).get("quantity_complete", False):
            achieved += FIELD_WEIGHTS["quantity_data"]
        if outputs.get("EUDR-001", {}).get("countries_identified", 0) > 0:
            achieved += FIELD_WEIGHTS["country_of_production"]
        geo_coverage = Decimal(str(
            outputs.get("EUDR-002", {}).get("coverage_pct", 0)
        ))
        achieved += (FIELD_WEIGHTS["plot_geolocation"]
                     * geo_coverage / Decimal("100"))
        poly_coverage = Decimal(str(
            outputs.get("EUDR-006", {}).get("polygon_coverage_pct", 0)
        ))
        achieved += (FIELD_WEIGHTS["polygon_coverage"]
                     * poly_coverage / Decimal("100"))
        custody_integrity = Decimal(str(
            outputs.get("EUDR-009", {}).get("chain_integrity_pct", 0)
        ))
        achieved += (FIELD_WEIGHTS["custody_chain"]
                     * custody_integrity / Decimal("100"))
        sat_verified = Decimal(str(
            outputs.get("EUDR-003", {}).get("verified_pct", 0)
        ))
        achieved += (FIELD_WEIGHTS["satellite_verification"]
                     * sat_verified / Decimal("100"))
        doc_auth = Decimal(str(
            outputs.get("EUDR-012", {}).get("authenticated_pct", 0)
        ))
        achieved += (FIELD_WEIGHTS["document_authentication"]
                     * doc_auth / Decimal("100"))

        return (achieved / total_weight * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
```

**Data Handoff Mapping:**

| Source Agent | Output Field | Target Agent(s) | Input Field |
|-------------|-------------|-----------------|------------|
| EUDR-001 | plot_coordinates | EUDR-002, EUDR-006, EUDR-007 | coordinates |
| EUDR-001 | supplier_graph | EUDR-008 | supplier_ids |
| EUDR-002 | verified_coordinates | EUDR-003, EUDR-004, EUDR-005 | plot_locations |
| EUDR-006 | polygon_boundaries | EUDR-004, EUDR-005 | boundaries |
| EUDR-008 | sub_tier_suppliers | EUDR-009 | supplier_chain |
| EUDR-009 | custody_records | EUDR-010, EUDR-011 | custody_data |

**Zero-Hallucination Guarantee:** Completeness scoring uses fixed Decimal weights. Agent ordering follows deterministic topological sort. Data handoff uses explicit field mapping. No LLM in phase orchestration.

**Estimated Lines of Code:** 4,000-4,500

---

### 4.3 Engine 3: Risk Assessment Coordinator

**File:** `risk_assessment_coordinator.py`
**Purpose:** Orchestrates Phase 2 execution of EUDR-016 through EUDR-025 for multi-dimensional risk scoring per Article 10, including composite risk calculation.

**Responsibilities:**
- Block until Quality Gate 1 (information gathering completeness) passes
- Launch all 10 risk assessment agents (EUDR-016 through EUDR-025) in parallel
- Pass information gathering outputs as input context to each risk agent
- Aggregate individual risk scores into a unified risk profile
- Calculate composite risk score using deterministic weighted formula
- Map each risk agent output to specific Article 10(2) risk factors
- Validate risk assessment completeness: all 10 risk dimensions scored
- Generate risk assessment summary with highest-risk findings highlighted
- Support degraded mode: proceed with available risk scores if non-critical agents fail

**Composite Risk Score Formula (Deterministic):**

```python
from decimal import Decimal, ROUND_HALF_UP

DEFAULT_RISK_WEIGHTS = {
    "EUDR-016": Decimal("0.15"),  # Country risk
    "EUDR-017": Decimal("0.12"),  # Supplier risk
    "EUDR-018": Decimal("0.10"),  # Commodity risk
    "EUDR-019": Decimal("0.08"),  # Corruption risk
    "EUDR-020": Decimal("0.15"),  # Deforestation risk
    "EUDR-021": Decimal("0.10"),  # Indigenous rights
    "EUDR-022": Decimal("0.10"),  # Protected areas
    "EUDR-023": Decimal("0.10"),  # Legal compliance
    "EUDR-024": Decimal("0.05"),  # Audit findings
    "EUDR-025": Decimal("0.05"),  # Mitigation readiness
}
# Sum = 1.00

def calculate_composite_risk_score(
    agent_scores: Dict[str, Decimal],
    weights: Dict[str, Decimal] = None,
) -> CompositeRiskResult:
    """
    Deterministic composite risk calculation.
    All arithmetic uses Decimal for bit-perfect reproducibility.
    """
    w = weights or DEFAULT_RISK_WEIGHTS
    assert sum(w.values()) == Decimal("1.00"), "Weights must sum to 1.00"

    weighted_sum = Decimal("0")
    scored_dimensions = 0
    dimension_details = {}

    for agent_id, weight in w.items():
        score = agent_scores.get(agent_id)
        if score is not None:
            contribution = (score * weight).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            weighted_sum += contribution
            scored_dimensions += 1
            dimension_details[agent_id] = {
                "raw_score": score,
                "weight": weight,
                "contribution": contribution,
            }

    composite = weighted_sum.quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    coverage_pct = (Decimal(str(scored_dimensions))
                    / Decimal("10") * Decimal("100")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    return CompositeRiskResult(
        composite_score=composite,
        coverage_pct=coverage_pct,
        scored_dimensions=scored_dimensions,
        total_dimensions=10,
        dimension_details=dimension_details,
        risk_level=_classify_risk_level(composite),
    )

def _classify_risk_level(score: Decimal) -> str:
    """Deterministic risk level classification."""
    if score <= Decimal("20"):
        return "negligible"
    elif score <= Decimal("40"):
        return "low"
    elif score <= Decimal("60"):
        return "standard"
    elif score <= Decimal("80"):
        return "high"
    else:
        return "critical"
```

**Article 10(2) Risk Factor Mapping:**

| Art. 10(2) Factor | Risk Agent(s) | Risk Dimension |
|-------------------|---------------|----------------|
| (a) Supply chain complexity | EUDR-001, EUDR-008 | Depth, node count, opaque segments |
| (b) Circumvention or mixing risk | EUDR-010, EUDR-011, EUDR-018 | Segregation integrity, mass balance, mixing |
| (c) Country non-compliance risk | EUDR-016, EUDR-019 | Country classification, corruption index |
| (d) Country of production risk | EUDR-016, EUDR-020, EUDR-021, EUDR-022 | Deforestation rate, rights, protected areas |
| (e) Supplier concerns | EUDR-017, EUDR-024 | Compliance history, audit findings |
| (f) Substantiated concerns | EUDR-020, EUDR-021, EUDR-023, EUDR-025 | Alerts, violations, legal gaps |

**Zero-Hallucination Guarantee:** Composite risk score uses fixed Decimal weights summing to 1.00. Risk classification uses deterministic threshold ranges. No LLM in risk scoring path.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.4 Engine 4: Risk Mitigation Coordinator

**File:** `risk_mitigation_coordinator.py`
**Purpose:** Coordinates Phase 3 risk mitigation via EUDR-025 integration, validates mitigation adequacy per Article 11, and manages the decision logic for mitigation bypass on negligible risk.

**Responsibilities:**
- Block until Quality Gate 2 (risk assessment completeness) passes
- Determine whether mitigation is required based on composite risk threshold
- Invoke EUDR-025 Risk Mitigation Advisor with full risk assessment context
- Validate that mitigation strategies address all identified non-negligible risks
- Verify mitigation adequacy: residual risk reduced to configurable threshold (default <= 15)
- Validate mitigation proportionality per Article 11(1)
- Collect mitigation evidence for due diligence package
- Support bypass for negligible risk workflows (skip to Package Generation)
- Trigger Quality Gate 3 upon mitigation completion

**Risk-to-Mitigation Decision Logic (Deterministic):**

```python
from decimal import Decimal

NEGLIGIBLE_THRESHOLD = Decimal("20")
STANDARD_THRESHOLD = Decimal("50")
RESIDUAL_RISK_TARGET = Decimal("15")
ENHANCED_RESIDUAL_TARGET = Decimal("10")

def determine_mitigation_requirement(
    composite_risk_score: Decimal,
) -> MitigationRequirement:
    """
    Deterministic decision logic for mitigation phase.
    Returns the required mitigation level.
    """
    if composite_risk_score <= NEGLIGIBLE_THRESHOLD:
        return MitigationRequirement(
            required=False,
            level="none",
            reason="Risk assessment identified negligible risk; "
                   "no mitigation required per Article 11",
            residual_target=None,
            bypass_to_package=True,
        )
    elif composite_risk_score <= STANDARD_THRESHOLD:
        return MitigationRequirement(
            required=True,
            level="standard",
            reason="Standard mitigation required: composite risk "
                   f"score {composite_risk_score} exceeds negligible "
                   f"threshold ({NEGLIGIBLE_THRESHOLD})",
            residual_target=RESIDUAL_RISK_TARGET,
            bypass_to_package=False,
        )
    else:
        return MitigationRequirement(
            required=True,
            level="enhanced",
            reason="Enhanced mitigation required: composite risk "
                   f"score {composite_risk_score} exceeds standard "
                   f"threshold ({STANDARD_THRESHOLD}). Independent "
                   "audits and supplier site visits required.",
            residual_target=ENHANCED_RESIDUAL_TARGET,
            bypass_to_package=False,
        )
```

**Mitigation Adequacy Verification:**

```python
def verify_mitigation_adequacy(
    pre_mitigation_score: Decimal,
    post_mitigation_score: Decimal,
    mitigation_level: str,
) -> MitigationAdequacyResult:
    """Deterministic adequacy check."""
    target = (ENHANCED_RESIDUAL_TARGET
              if mitigation_level == "enhanced"
              else RESIDUAL_RISK_TARGET)

    reduction = pre_mitigation_score - post_mitigation_score
    reduction_pct = (reduction / pre_mitigation_score * Decimal("100")
                     ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP
                     ) if pre_mitigation_score > Decimal("0") else Decimal("0")

    adequate = post_mitigation_score <= target

    return MitigationAdequacyResult(
        adequate=adequate,
        pre_mitigation_score=pre_mitigation_score,
        post_mitigation_score=post_mitigation_score,
        reduction_absolute=reduction,
        reduction_pct=reduction_pct,
        target_residual=target,
        gap=max(Decimal("0"), post_mitigation_score - target),
    )
```

**Zero-Hallucination Guarantee:** All decision thresholds are fixed Decimal constants. Adequacy verification uses deterministic comparison. No LLM in mitigation decision path.

**Estimated Lines of Code:** 2,500-3,000

---

### 4.5 Engine 5: Quality Gate Engine

**File:** `quality_gate_engine.py`
**Purpose:** Validates completeness, accuracy, and consistency at each phase transition using configurable thresholds and deterministic check evaluation.

**Responsibilities:**
- Enforce Quality Gate 1 (Information Gathering): Art. 9 field completeness >= 90%
- Enforce Quality Gate 2 (Risk Assessment): risk dimension coverage >= 95%
- Enforce Quality Gate 3 (Mitigation Adequacy): residual risk <= 15
- Provide configurable thresholds per quality gate (adjustable per operator/commodity)
- Generate detailed quality gate reports with individual check results
- Identify specific gaps causing failure with remediation guidance
- Support quality gate override with mandatory justification
- Support threshold relaxation for simplified due diligence (Article 13)
- Emit events on gate evaluation: PASSED, FAILED, OVERRIDDEN

**Quality Gate Evaluation (Deterministic):**

```python
from decimal import Decimal, ROUND_HALF_UP

QUALITY_GATE_DEFINITIONS = {
    "QG-1": QualityGateSpec(
        gate_id="QG-1",
        gate_name="Information Gathering Completeness",
        phase_before="information_gathering",
        phase_after="risk_assessment",
        default_threshold=Decimal("90.00"),
        simplified_threshold=Decimal("80.00"),
        checks=[
            QGCheck("product_description", Decimal("10"), "Non-empty CN/HS code"),
            QGCheck("quantity_data", Decimal("10"), "Positive numeric with unit"),
            QGCheck("country_of_production", Decimal("10"), "Valid ISO 3166-1"),
            QGCheck("plot_geolocation", Decimal("20"), "Valid WGS84 coords"),
            QGCheck("polygon_coverage", Decimal("10"), "Polygons for > 4 ha"),
            QGCheck("custody_chain", Decimal("15"), "Unbroken producer to importer"),
            QGCheck("satellite_verification", Decimal("15"), "Deforestation-free"),
            QGCheck("document_authentication", Decimal("10"), "Key docs authenticated"),
        ],
    ),
    "QG-2": QualityGateSpec(
        gate_id="QG-2",
        gate_name="Risk Assessment Completeness",
        phase_before="risk_assessment",
        phase_after="risk_mitigation",
        default_threshold=Decimal("95.00"),
        simplified_threshold=Decimal("85.00"),
        checks=[
            QGCheck("risk_dimension_coverage", Decimal("40"), "All 10 dimensions scored"),
            QGCheck("composite_risk_calculated", Decimal("20"), "Composite score computed"),
            QGCheck("art_10_2_factors", Decimal("25"), "All Art 10(2) factors assessed"),
            QGCheck("risk_summary_generated", Decimal("15"), "Summary with highlights"),
        ],
    ),
    "QG-3": QualityGateSpec(
        gate_id="QG-3",
        gate_name="Mitigation Adequacy",
        phase_before="risk_mitigation",
        phase_after="package_generation",
        default_threshold=Decimal("85.00"),
        simplified_threshold=None,  # QG-3 skipped for simplified
        checks=[
            QGCheck("residual_risk_threshold", Decimal("40"), "Residual risk <= target"),
            QGCheck("mitigation_evidence", Decimal("30"), "Evidence documented"),
            QGCheck("proportionality_verified", Decimal("30"), "Effort proportionate"),
        ],
    ),
}

def evaluate_quality_gate(
    gate_id: str,
    workflow: WorkflowExecution,
    is_simplified: bool = False,
) -> GateEvaluation:
    """
    Deterministic quality gate evaluation.
    Returns pass/fail with individual check scores.
    """
    spec = QUALITY_GATE_DEFINITIONS[gate_id]
    threshold = (spec.simplified_threshold
                 if is_simplified and spec.simplified_threshold
                 else spec.default_threshold)

    check_results = []
    total_weight = Decimal("0")
    achieved_weight = Decimal("0")

    for check in spec.checks:
        score = _evaluate_check(check, workflow)
        passed = score >= Decimal("100")
        weighted_contribution = (check.weight * score / Decimal("100")
                                 ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        check_results.append(CheckResult(
            check_name=check.name,
            weight=check.weight,
            score=score,
            passed=passed,
            weighted_contribution=weighted_contribution,
            remediation=check.remediation_guidance if not passed else None,
        ))
        total_weight += check.weight
        achieved_weight += weighted_contribution

    overall_score = (achieved_weight / total_weight * Decimal("100")
                     ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    gate_passed = overall_score >= threshold

    return GateEvaluation(
        gate_id=gate_id,
        result="passed" if gate_passed else "failed",
        overall_score=overall_score,
        threshold_used=threshold,
        check_results=check_results,
    )
```

**Zero-Hallucination Guarantee:** Gate evaluation uses fixed Decimal weights and thresholds. Check scoring is deterministic. No LLM in quality gate evaluation path.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.6 Engine 6: Workflow State Manager

**File:** `workflow_state_manager.py`
**Purpose:** Persistent checkpointing after every agent completion, workflow resume from any checkpoint, rollback, pause, cancel, clone, and immutable audit trail with SHA-256 provenance.

**Responsibilities:**
- Checkpoint workflow state after every agent completion
- Checkpoint after every quality gate evaluation
- Support resume from any checkpoint without re-executing completed agents
- Support rollback to a previous checkpoint on critical failure
- Support pause, cancel, and clone operations
- Maintain immutable audit trail with timestamp, actor, and reason
- Generate cumulative SHA-256 provenance hash for each checkpoint
- Retain state for minimum 5 years per EUDR Article 31
- Provide real-time progress tracking via Redis pub/sub
- Support concurrent read access while execution continues

**Checkpoint and Provenance:**

```python
import hashlib
import json
from datetime import datetime

class WorkflowStateManager:
    """Persistent workflow state with SHA-256 provenance chain."""

    async def checkpoint(
        self,
        workflow: WorkflowExecution,
        agent_id: str = None,
        gate_id: str = None,
        checkpoint_type: str = "agent_complete",
    ) -> WorkflowCheckpoint:
        """
        Create a persistent checkpoint after agent completion or gate evaluation.
        Provenance hash chains all accumulated state.
        """
        sequence = await self._next_sequence(workflow.workflow_id)

        # Compute cumulative provenance hash
        previous_hash = await self._get_previous_hash(workflow.workflow_id)
        provenance_hash = self._compute_cumulative_hash(
            previous_hash=previous_hash,
            workflow_id=workflow.workflow_id,
            sequence=sequence,
            agent_statuses=workflow.agent_statuses,
            checkpoint_type=checkpoint_type,
            agent_id=agent_id,
            gate_id=gate_id,
            timestamp=datetime.utcnow(),
        )

        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow.workflow_id,
            sequence_number=sequence,
            phase=workflow.current_phase,
            agent_id=agent_id,
            gate_id=gate_id,
            checkpoint_type=checkpoint_type,
            agent_statuses=workflow.agent_statuses,
            agent_outputs_ref=await self._store_outputs(workflow),
            quality_gate_results=workflow.quality_gate_results,
            cumulative_provenance_hash=provenance_hash,
        )

        # Persist to PostgreSQL (async, non-blocking)
        await self._persist_checkpoint(checkpoint)

        # Publish progress to Redis for real-time tracking
        await self._publish_progress(workflow)

        # Record audit trail entry
        await self._record_audit_event(
            workflow.workflow_id,
            event_type=f"checkpoint_{checkpoint_type}",
            event_data={
                "agent_id": agent_id,
                "gate_id": gate_id,
                "sequence": sequence,
            },
            provenance_hash=provenance_hash,
        )

        return checkpoint

    def _compute_cumulative_hash(
        self, previous_hash: str, **kwargs
    ) -> str:
        """
        SHA-256 hash chaining. Each checkpoint hash includes
        the previous hash, creating an immutable chain.
        """
        payload = json.dumps({
            "previous_hash": previous_hash or "GENESIS",
            "workflow_id": kwargs["workflow_id"],
            "sequence": kwargs["sequence"],
            "checkpoint_type": kwargs["checkpoint_type"],
            "agent_id": kwargs.get("agent_id"),
            "gate_id": kwargs.get("gate_id"),
            "agent_statuses_hash": hashlib.sha256(
                json.dumps(kwargs["agent_statuses"],
                           sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamp": kwargs["timestamp"].isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def resume_from_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str = None,
    ) -> WorkflowExecution:
        """
        Resume workflow from specified checkpoint (or latest).
        Restores all agent statuses and skips completed agents.
        """
        checkpoint = await self._load_checkpoint(
            workflow_id, checkpoint_id
        )

        # Verify provenance chain integrity
        await self._verify_provenance_chain(workflow_id, checkpoint)

        # Restore workflow state
        workflow = await self._restore_workflow(checkpoint)
        workflow.status = "running"

        # Record resume event
        await self._record_audit_event(
            workflow_id,
            event_type="workflow_resumed",
            event_data={
                "from_checkpoint": checkpoint.checkpoint_id,
                "sequence": checkpoint.sequence_number,
            },
            provenance_hash=checkpoint.cumulative_provenance_hash,
        )

        return workflow
```

**Workflow State Machine (22 States):**

| State | Description | Transitions |
|-------|-------------|-------------|
| CREATED | Workflow defined, not started | -> VALIDATING |
| VALIDATING | DAG validation in progress | -> READY, VALIDATION_FAILED |
| VALIDATION_FAILED | Invalid workflow definition | -> CREATED (fix and retry) |
| READY | Validated, awaiting start | -> PHASE1_RUNNING |
| PHASE1_RUNNING | Phase 1 agents executing | -> PHASE1_COMPLETE, AGENT_FAILED, PAUSED |
| PHASE1_COMPLETE | All Phase 1 agents done | -> QG1_EVALUATING |
| QG1_EVALUATING | Quality Gate 1 in progress | -> QG1_PASSED, QG1_FAILED |
| QG1_PASSED | Information gathering adequate | -> PHASE2_RUNNING |
| QG1_FAILED | Information gathering gaps | -> PHASE1_RUNNING (remediate), QG1_OVERRIDDEN |
| QG1_OVERRIDDEN | Gate failed but overridden | -> PHASE2_RUNNING |
| PHASE2_RUNNING | Phase 2 agents executing | -> PHASE2_COMPLETE, AGENT_FAILED, PAUSED |
| PHASE2_COMPLETE | All Phase 2 agents done | -> QG2_EVALUATING |
| QG2_EVALUATING | Quality Gate 2 in progress | -> QG2_PASSED, QG2_FAILED |
| QG2_PASSED | Risk assessment complete | -> PHASE3_RUNNING, MITIGATION_BYPASSED |
| QG2_FAILED | Risk assessment gaps | -> PHASE2_RUNNING (remediate), QG2_OVERRIDDEN |
| QG2_OVERRIDDEN | Gate failed but overridden | -> PHASE3_RUNNING |
| MITIGATION_BYPASSED | Negligible risk, skip Phase 3 | -> PACKAGE_GENERATING |
| PHASE3_RUNNING | Phase 3 mitigation executing | -> QG3_EVALUATING, AGENT_FAILED, PAUSED |
| QG3_EVALUATING | Quality Gate 3 in progress | -> PACKAGE_GENERATING, QG3_FAILED |
| QG3_FAILED | Mitigation inadequate | -> PHASE3_RUNNING, QG3_OVERRIDDEN |
| PACKAGE_GENERATING | Generating DD package | -> COMPLETED |
| COMPLETED | All phases done, package ready | (terminal) |
| PAUSED | Execution paused by user | -> (resume to previous RUNNING state) |
| CANCELLED | Cancelled by user | (terminal) |
| TERMINATED | Unrecoverable failure | (terminal) |
| AGENT_FAILED | Agent failure, awaiting retry/decision | -> (RUNNING state), TERMINATED |

**Zero-Hallucination Guarantee:** Provenance chain uses SHA-256 hashing. State transitions are deterministic FSM. Checkpoint restore uses exact state replay. No LLM in state management.

**Estimated Lines of Code:** 5,000-5,500

---

### 4.7 Engine 7: Parallel Execution Engine

**File:** `parallel_execution_engine.py`
**Purpose:** DAG-aware concurrent agent execution with configurable concurrency limits, work-stealing scheduler, critical path analysis, and ETA calculation.

**Responsibilities:**
- Analyze DAG topology to identify independent agent groups per execution layer
- Execute agents within the same layer concurrently using async task pool
- Support configurable per-workflow concurrency (default: 10, max: 25)
- Support global concurrency limit across all workflows (default: 100)
- Implement work-stealing: idle workers pick up tasks from busy workers' queues
- Calculate critical path for ETA estimation using longest-path algorithm
- Provide execution timeline data for Gantt chart visualization
- Handle agent completion events and trigger dependent agents immediately
- Support priority-based scheduling: critical-path agents prioritized
- Report parallelization efficiency: actual speedup vs. theoretical maximum

**Parallel Layer Execution:**

```python
import asyncio
from datetime import datetime

class ParallelExecutionEngine:
    """DAG-aware concurrent agent execution with work-stealing."""

    def __init__(
        self,
        global_semaphore: asyncio.Semaphore,
        redis_client,
    ):
        self._global_semaphore = global_semaphore
        self._redis = redis_client

    async def execute_layer(
        self,
        workflow_id: str,
        agents: List[str],
        inputs: Dict[str, Dict],
        agent_client: AgentClient,
        max_concurrency: int = 10,
    ) -> Dict[str, AgentResult]:
        """
        Execute all agents in a layer concurrently.
        Respects per-workflow and global concurrency limits.
        """
        workflow_semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        tasks = []

        for agent_id in agents:
            task = asyncio.create_task(
                self._execute_agent_with_limits(
                    workflow_id=workflow_id,
                    agent_id=agent_id,
                    input_data=inputs.get(agent_id, {}),
                    agent_client=agent_client,
                    workflow_semaphore=workflow_semaphore,
                )
            )
            tasks.append((agent_id, task))

        # Wait for all agents in layer
        for agent_id, task in tasks:
            try:
                results[agent_id] = await task
            except Exception as e:
                results[agent_id] = AgentResult(
                    agent_id=agent_id,
                    status="failed",
                    error=str(e),
                )

        return results

    async def _execute_agent_with_limits(
        self,
        workflow_id: str,
        agent_id: str,
        input_data: Dict,
        agent_client: AgentClient,
        workflow_semaphore: asyncio.Semaphore,
    ) -> AgentResult:
        """Execute single agent respecting concurrency limits."""
        async with workflow_semaphore:
            async with self._global_semaphore:
                start_time = datetime.utcnow()

                # Record execution start
                await self._redis.hset(
                    f"ddo:timeline:{workflow_id}",
                    f"{agent_id}:start",
                    start_time.isoformat(),
                )

                result = await agent_client.invoke(
                    agent_id=agent_id,
                    input_data=input_data,
                    workflow_id=workflow_id,
                )

                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                # Record execution end
                await self._redis.hset(
                    f"ddo:timeline:{workflow_id}",
                    f"{agent_id}:end",
                    end_time.isoformat(),
                )

                result.duration_seconds = duration
                return result

    def calculate_eta(
        self,
        dag: nx.DiGraph,
        completed_agents: Set[str],
        agent_durations: Dict[str, float],
        default_duration: float = 30.0,
    ) -> int:
        """
        Estimate remaining time based on critical path of
        uncompleted agents. Returns seconds.
        """
        remaining = set(dag.nodes()) - completed_agents
        if not remaining:
            return 0

        subgraph = dag.subgraph(remaining)
        estimated = {
            n: agent_durations.get(n, default_duration)
            for n in remaining
        }
        for n in subgraph.nodes():
            subgraph.nodes[n]["weight"] = estimated[n]

        if len(subgraph.nodes()) == 0:
            return 0

        return int(nx.dag_longest_path_length(subgraph, weight="weight"))
```

**Parallelization Analysis (Standard Workflow):**

| Layer | Agents | Parallel Width | Sequential Time | Parallel Time | Speedup |
|-------|--------|---------------|-----------------|---------------|---------|
| 0 | EUDR-001 | 1 | 30s | 30s | 1.0x |
| 1 | EUDR-002, 006, 007, 008 | 4 | 120s | 30s | 4.0x |
| 2 | EUDR-003, 004, 005 | 3 | 90s | 30s | 3.0x |
| 3 | EUDR-009, 010, 011 | 3 | 90s | 30s | 3.0x |
| 4 | EUDR-012, 013, 014, 015 | 4 | 120s | 30s | 4.0x |
| QG-1 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 5 | EUDR-016-025 | 10 | 300s | 30s | 10.0x |
| QG-2 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 6 | EUDR-025 (mitigation) | 1 | 30s | 30s | 1.0x |
| QG-3 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 7 | Package Generator | 1 | 15s | 15s | 1.0x |
| **Total** | **25 agents + 3 gates** | -- | **810s (13.5 min)** | **240s (4 min)** | **3.4x** |

**Zero-Hallucination Guarantee:** Concurrency is managed by semaphores (deterministic admission). ETA uses longest-path algorithm on DAG (deterministic). No LLM in execution scheduling.

**Estimated Lines of Code:** 4,000-4,500

---

### 4.8 Engine 8: Error Recovery Manager

**File:** `error_recovery_manager.py`
**Purpose:** Exponential backoff with jitter, circuit breaker pattern, fallback strategies, and dead letter queue for unrecoverable failures.

**Responsibilities:**
- Implement exponential backoff: delay = min(base * 2^attempt + jitter, max_delay)
- Configurable retry parameters: base_delay (1s), max_delay (300s), max_retries (5)
- Circuit breaker per agent type: CLOSED, OPEN, HALF_OPEN states
- Circuit breaker parameters: failure_threshold (5), reset_timeout (60s), success_threshold (2)
- Classify errors: transient (retry), permanent (fail), degraded (fallback)
- Fallback strategies: cached_result, degraded_mode, manual_override
- Dead letter queue for permanently failed invocations
- Per-agent retry configuration overrides
- Emit retry/circuit-breaker events for monitoring

**Exponential Backoff with Jitter:**

```python
import random
import asyncio

class ErrorRecoveryManager:
    """Resilient error handling with backoff, circuit breaker, and fallback."""

    async def execute_with_retry(
        self,
        agent_id: str,
        execute_fn,
        retry_config: RetryConfig = None,
    ) -> AgentResult:
        """Execute agent invocation with retry and circuit breaker."""
        config = retry_config or RetryConfig()

        # Check circuit breaker
        cb_state = await self._get_circuit_breaker_state(agent_id)
        if cb_state == "open":
            return await self._handle_circuit_open(agent_id)

        last_error = None
        for attempt in range(1, config.max_retries + 1):
            try:
                result = await execute_fn()

                # Success: reset circuit breaker
                await self._record_success(agent_id)
                return result

            except Exception as e:
                last_error = e
                classification = self._classify_error(e)

                if classification == "permanent":
                    await self._send_to_dlq(agent_id, attempt, e)
                    raise PermanentAgentError(agent_id, str(e))

                if classification == "degraded":
                    fallback = await self._get_fallback(agent_id)
                    if fallback:
                        return fallback

                # Transient: retry with backoff
                await self._record_failure(agent_id)

                if attempt < config.max_retries:
                    delay = self._calculate_backoff(
                        attempt, config.base_delay, config.max_delay
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        await self._record_failure(agent_id)
        await self._send_to_dlq(agent_id, config.max_retries, last_error)
        raise MaxRetriesExceededError(agent_id, config.max_retries)

    def _calculate_backoff(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """
        Exponential backoff with jitter.
        delay = min(base * 2^attempt + random_jitter, max_delay)
        Jitter range: 0 to base_delay.
        """
        exponential = base_delay * (2 ** attempt)
        jitter = random.uniform(0, base_delay)
        return min(exponential + jitter, max_delay)

    def _classify_error(self, error: Exception) -> str:
        """Deterministic error classification."""
        if hasattr(error, 'status_code'):
            code = error.status_code
            if code == 429:
                return "transient"   # Rate limited
            if code == 503:
                return "transient"   # Service unavailable
            if code in (400, 401, 403, 404, 422):
                return "permanent"   # Client error
            if code >= 500:
                return "transient"   # Server error
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return "transient"
        if isinstance(error, ValueError):
            return "permanent"
        return "transient"          # Default to transient
```

**Circuit Breaker State Machine:**

```
CLOSED --[failure_count >= failure_threshold (5)]--> OPEN
OPEN --[reset_timeout (60s) elapsed]--> HALF_OPEN
HALF_OPEN --[success_count >= success_threshold (2)]--> CLOSED
HALF_OPEN --[any failure]--> OPEN
```

**Error Classification Matrix:**

| Error Type | Classification | Action | Example |
|-----------|---------------|--------|---------|
| HTTP 429 | Transient | Retry with backoff | API rate limit |
| HTTP 503 | Transient | Retry with backoff | Agent temporarily down |
| Connection Timeout | Transient | Retry with backoff | Network instability |
| HTTP 400 | Permanent | Fail immediately, DLQ | Invalid input |
| HTTP 401/403 | Permanent | Fail immediately, DLQ | Auth error |
| Data Validation | Permanent | Fail with details, DLQ | Upstream data quality |
| Partial Result | Degraded | Use fallback + warning | Partial completion |

**Zero-Hallucination Guarantee:** Error classification is rule-based. Backoff formula is deterministic (except jitter, which is bounded). Circuit breaker state transitions are deterministic threshold checks. No LLM in error handling.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.9 Engine 9: Due Diligence Package Generator

**File:** `due_diligence_package_generator.py`
**Purpose:** Compiles all 25 agent outputs into a single audit-ready evidence bundle with DDS-compatible JSON, human-readable PDF, and SHA-256 integrity hashes.

**Responsibilities:**
- Compile all 25 agent outputs into a structured evidence bundle
- Generate DDS-compatible JSON matching EU Information System schema
- Generate PDF report with executive summary, findings, evidence annexes, provenance chain
- Map agent outputs to EUDR Article 12(2)(a-j) DDS content requirements
- Validate package against EU Information System DDS schema
- Include SHA-256 integrity hash on every evidence artifact and complete package
- Support multi-language output: EN, FR, DE, ES, PT
- Support batch package generation for multiple products/shipments
- Include workflow execution metadata: duration, agents, gates, retries
- Provide package download in JSON, PDF, HTML, ZIP formats
- Support package versioning and amendment

**DDS Content Mapping (Article 12(2)):**

| Art. 12(2) | DDS Field | Source Agent(s) | Package Section |
|-----------|-----------|-----------------|-----------------|
| (a) | Operator name and contact | System config | Cover Page |
| (b) | Product description, trade name, CN code | EUDR-001 | Sec 1: Product ID |
| (c) | Quantity | EUDR-001, EUDR-011 | Sec 1: Product ID |
| (d) | Country of production | EUDR-001, EUDR-016 | Sec 2: Origin |
| (e) | Geolocation of plots | EUDR-002, EUDR-006, EUDR-007 | Sec 3: Geolocation |
| (f) | Date/period of production | EUDR-001, EUDR-009 | Sec 2: Origin |
| (g) | Deforestation-free verification | EUDR-003, EUDR-004, EUDR-005, EUDR-020 | Sec 4: Deforestation |
| (h) | Legal compliance evidence | EUDR-012, EUDR-023 | Sec 5: Legal |
| (i) | Risk assessment results | EUDR-016-025 | Sec 6: Risk Assessment |
| (j) | Risk mitigation measures | EUDR-025 | Sec 7: Mitigation |
| -- | Supply chain traceability | EUDR-001, EUDR-008-011, EUDR-013 | Sec 8: Supply Chain |
| -- | Workflow provenance | DDO-026 | Sec 9: Audit Trail |

**Package Generation Pipeline:**

```python
import hashlib
import json

class DueDiligencePackageGenerator:
    """Compile audit-ready DD evidence bundle."""

    async def generate_package(
        self,
        workflow: WorkflowExecution,
        languages: List[str] = None,
    ) -> DueDiligencePackage:
        """Generate complete due diligence package from workflow outputs."""
        languages = languages or ["en"]

        # Build DDS JSON per Article 12(2)
        dds_json = self._build_dds_json(workflow)

        # Build report sections
        sections = [
            self._build_product_section(workflow),      # Sec 1
            self._build_origin_section(workflow),        # Sec 2
            self._build_geolocation_section(workflow),   # Sec 3
            self._build_deforestation_section(workflow),  # Sec 4
            self._build_legal_section(workflow),          # Sec 5
            self._build_risk_assessment_section(workflow), # Sec 6
            self._build_mitigation_section(workflow),     # Sec 7
            self._build_supply_chain_section(workflow),   # Sec 8
            self._build_audit_trail_section(workflow),    # Sec 9
        ]

        # Compute evidence hashes per section
        for section in sections:
            section.evidence_hash = self._hash_section_evidence(section)

        # Build provenance chain from workflow checkpoints
        provenance_chain = await self._build_provenance_chain(
            workflow.workflow_id
        )

        # Compute package-level integrity hash
        package_hash = self._compute_package_hash(
            dds_json, sections, provenance_chain
        )

        # Validate against DDS schema
        validation = self._validate_dds_schema(dds_json)
        if not validation.valid:
            raise DDSValidationError(validation.errors)

        package = DueDiligencePackage(
            workflow_id=workflow.workflow_id,
            operator_id=workflow.operator_id,
            commodity=workflow.commodity,
            product_ids=workflow.product_ids,
            workflow_type=workflow.workflow_type,
            dds_json=dds_json,
            sections=sections,
            quality_gate_summary=workflow.quality_gate_results,
            risk_profile=workflow.risk_profile,
            mitigation_summary=workflow.mitigation_summary,
            provenance_chain=provenance_chain,
            package_hash=package_hash,
            languages=languages,
        )

        # Store package and generate PDF
        await self._persist_package(package)
        for lang in languages:
            pdf_ref = await self._generate_pdf(package, lang)
            package.pdf_refs[lang] = pdf_ref

        # Generate ZIP bundle
        package.zip_ref = await self._generate_zip_bundle(package)

        return package

    def _compute_package_hash(
        self,
        dds_json: Dict,
        sections: List[PackageSection],
        provenance_chain: List[ProvenanceEntry],
    ) -> str:
        """SHA-256 hash of complete package for integrity verification."""
        payload = json.dumps({
            "dds_json_hash": hashlib.sha256(
                json.dumps(dds_json, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "section_hashes": [s.evidence_hash for s in sections],
            "provenance_chain_hash": hashlib.sha256(
                json.dumps(
                    [p.dict() for p in provenance_chain],
                    sort_keys=True, default=str
                ).encode()
            ).hexdigest(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
```

**Zero-Hallucination Guarantee:** Package content is compiled exclusively from agent outputs (no generated text). DDS JSON is assembled from deterministic field mapping. Integrity hashes use SHA-256. No LLM in package generation.

**Estimated Lines of Code:** 5,000-5,500

---

## 5. Data Flow Architecture

### 5.1 End-to-End Workflow Data Flow

```
Operator submits workflow request (commodity, products, shipments)
    |
    v
E1: build_dag() -> topological_sort() -> calculate_critical_path()
    |
    v
E7: execute_layer(Layer 0)
    |-> EUDR-001 Supply Chain Mapping (plot data, supplier graph, product info)
    v
E7: execute_layer(Layer 1) -- 4 agents in parallel
    |-> EUDR-002 Geolocation Verification (verified coords)
    |-> EUDR-006 Plot Boundary Manager (polygon boundaries)
    |-> EUDR-007 GPS Coordinate Validator (validated GPS)
    |-> EUDR-008 Multi-Tier Supplier Tracker (sub-tier suppliers)
    v
E7: execute_layer(Layer 2) -- 3 agents in parallel
    |-> EUDR-003 Satellite Monitoring (imagery, change detection)
    |-> EUDR-004 Forest Cover Analysis (forest metrics)
    |-> EUDR-005 Land Use Change Detector (land use events)
    v
E7: execute_layer(Layer 3) -- 3 agents in parallel
    |-> EUDR-009 Chain of Custody (custody records)
    |-> EUDR-010 Segregation Verifier (segregation integrity)
    |-> EUDR-011 Mass Balance Calculator (mass balance)
    v
E7: execute_layer(Layer 4) -- 4 agents in parallel
    |-> EUDR-012 Document Authentication
    |-> EUDR-013 Blockchain Integration
    |-> EUDR-014 QR Code Generator
    |-> EUDR-015 Mobile Data Collector
    v
E2: calculate_completeness() -> PhaseResult
    |
    v
E5: evaluate_quality_gate("QG-1") -> PASSED / FAILED
    |
    v (if PASSED)
E7: execute_layer(Layer 5) -- 10 agents in parallel
    |-> EUDR-016 through EUDR-025 (all risk agents)
    v
E3: calculate_composite_risk_score() -> CompositeRiskResult
    |
    v
E5: evaluate_quality_gate("QG-2") -> PASSED / FAILED
    |
    v (if PASSED)
E4: determine_mitigation_requirement()
    |-> IF negligible: bypass to E9
    |-> IF standard/enhanced: invoke EUDR-025 (mitigation mode)
    v
E4: verify_mitigation_adequacy() -> MitigationAdequacyResult
    |
    v
E5: evaluate_quality_gate("QG-3") -> PASSED / FAILED
    |
    v (if PASSED)
E9: generate_package() -> DueDiligencePackage
    |
    v
OUTPUT: DDS JSON + PDF Report + ZIP Bundle (S3)
```

### 5.2 Checkpoint Data Flow

```
Every agent completion / gate evaluation
    |
    v
E6: checkpoint()
    +---> Compute cumulative SHA-256 provenance hash
    +---> Persist checkpoint to PostgreSQL (workflow_checkpoints hypertable)
    +---> Store large outputs to S3 (agent_outputs_ref)
    +---> Publish progress to Redis pub/sub
    +---> Record audit trail event (workflow_audit_trail hypertable)
    |
    v
Redis: PUBLISH ddo:progress:{workflow_id}
    |
    v
GL-EUDR-APP frontend polls / subscribes for real-time updates
```

### 5.3 Error Recovery Data Flow

```
Agent invocation fails
    |
    v
E8: _classify_error(exception)
    +---> "transient" -> _calculate_backoff() -> asyncio.sleep() -> retry
    +---> "permanent" -> _send_to_dlq() -> raise PermanentAgentError
    +---> "degraded"  -> _get_fallback() -> return cached/degraded result
    |
    v (on repeated transient failure)
E8: _record_failure() -> check circuit breaker threshold
    +---> failure_count >= 5: circuit breaker -> OPEN
    +---> OPEN for 60s -> HALF_OPEN -> probe -> CLOSED or OPEN
    |
    v (max retries exhausted)
E8: _send_to_dlq(agent_id, attempts, error)
    |
    v
dead_letter_queue table -> ops review -> resolve/retry
```

---

## 6. Database Schema (V114)

### 6.1 Overview

- **Migration:** `V114__agent_eudr_due_diligence_orchestrator.sql`
- **Schema:** `eudr_due_diligence_orchestrator`
- **Tables:** 13 (9 regular + 4 TimescaleDB hypertables)
- **Estimated indexes:** ~25
- **Retention:** 5 years (EUDR Article 31)

### 6.2 Regular Tables (9)

**Table 1: `workflow_definitions`** -- Templates and custom workflow DAGs

| Column | Type | Constraints |
|--------|------|-------------|
| workflow_def_id | UUID | PK DEFAULT gen_random_uuid() |
| name | VARCHAR(500) | NOT NULL |
| description | TEXT | |
| workflow_type | VARCHAR(20) | NOT NULL DEFAULT 'standard' CHECK IN ('standard','simplified','custom') |
| commodity | VARCHAR(50) | |
| version | INTEGER | NOT NULL DEFAULT 1 |
| agent_nodes | JSONB | NOT NULL DEFAULT '[]' |
| dependency_edges | JSONB | NOT NULL DEFAULT '[]' |
| quality_gates | JSONB | NOT NULL DEFAULT '[]' |
| config | JSONB | NOT NULL DEFAULT '{}' |
| is_system_template | BOOLEAN | DEFAULT FALSE |
| created_by | VARCHAR(100) | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_type), (commodity), (is_system_template), UNIQUE(name, version)

**Table 2: `workflow_executions`** -- One row per due diligence run

| Column | Type | Constraints |
|--------|------|-------------|
| workflow_id | UUID | PK DEFAULT gen_random_uuid() |
| workflow_def_id | UUID | NOT NULL FK -> workflow_definitions |
| operator_id | UUID | NOT NULL |
| commodity | VARCHAR(50) | NOT NULL |
| product_ids | JSONB | NOT NULL DEFAULT '[]' |
| batch_id | UUID | |
| status | VARCHAR(30) | NOT NULL DEFAULT 'created' |
| current_phase | VARCHAR(30) | NOT NULL DEFAULT 'information_gathering' |
| agent_statuses | JSONB | NOT NULL DEFAULT '{}' |
| progress | JSONB | NOT NULL DEFAULT '{}' |
| error_summary | JSONB | |
| package_id | UUID | |
| risk_profile | JSONB | |
| mitigation_summary | JSONB | |
| started_at | TIMESTAMPTZ | |
| completed_at | TIMESTAMPTZ | |
| estimated_completion | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (operator_id), (status), (commodity), (batch_id), (operator_id, status)

**Table 3: `due_diligence_packages`** -- Generated evidence bundles

| Column | Type | Constraints |
|--------|------|-------------|
| package_id | UUID | PK DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL FK -> workflow_executions |
| operator_id | UUID | NOT NULL |
| commodity | VARCHAR(50) | NOT NULL |
| product_ids | JSONB | NOT NULL DEFAULT '[]' |
| workflow_type | VARCHAR(20) | NOT NULL |
| dds_json | JSONB | NOT NULL |
| sections | JSONB | NOT NULL DEFAULT '[]' |
| quality_gate_summary | JSONB | NOT NULL DEFAULT '{}' |
| risk_profile | JSONB | NOT NULL DEFAULT '{}' |
| mitigation_summary | JSONB | |
| provenance_chain | JSONB | NOT NULL DEFAULT '[]' |
| package_hash | VARCHAR(64) | NOT NULL |
| languages | JSONB | NOT NULL DEFAULT '["en"]' |
| pdf_refs | JSONB | DEFAULT '{}' |
| zip_ref | VARCHAR(500) | |
| version | INTEGER | NOT NULL DEFAULT 1 |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (operator_id), (workflow_id), (commodity)

**Table 4: `circuit_breaker_state`** -- Per-agent, per-operator circuit breaker

| Column | Type | Constraints |
|--------|------|-------------|
| agent_id | VARCHAR(20) | PK (composite) |
| operator_id | UUID | PK (composite) |
| state | VARCHAR(15) | NOT NULL DEFAULT 'closed' |
| failure_count | INTEGER | NOT NULL DEFAULT 0 |
| success_count | INTEGER | NOT NULL DEFAULT 0 |
| last_failure_at | TIMESTAMPTZ | |
| last_success_at | TIMESTAMPTZ | |
| opened_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Table 5: `dead_letter_queue`** -- Unrecoverable failures

| Column | Type | Constraints |
|--------|------|-------------|
| dlq_id | UUID | PK DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| agent_id | VARCHAR(20) | NOT NULL |
| attempt_number | INTEGER | NOT NULL |
| error_type | VARCHAR(50) | NOT NULL |
| error_message | TEXT | NOT NULL |
| error_classification | VARCHAR(20) | NOT NULL |
| input_data_ref | VARCHAR(500) | |
| retry_history | JSONB | NOT NULL DEFAULT '[]' |
| resolved | BOOLEAN | DEFAULT FALSE |
| resolved_at | TIMESTAMPTZ | |
| resolved_by | VARCHAR(100) | |
| resolution_notes | TEXT | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_id), (resolved) WHERE resolved = FALSE

**Table 6: `batch_executions`** -- Batch workflow groups

| Column | Type | Constraints |
|--------|------|-------------|
| batch_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| name | VARCHAR(500) | |
| workflow_count | INTEGER | NOT NULL DEFAULT 0 |
| completed_count | INTEGER | NOT NULL DEFAULT 0 |
| failed_count | INTEGER | NOT NULL DEFAULT 0 |
| status | VARCHAR(20) | NOT NULL DEFAULT 'running' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| completed_at | TIMESTAMPTZ | |

**Table 7: `workflow_definition_versions`** -- Immutable version history

| Column | Type | Constraints |
|--------|------|-------------|
| version_id | UUID | PK DEFAULT gen_random_uuid() |
| workflow_def_id | UUID | NOT NULL FK -> workflow_definitions |
| version_number | INTEGER | NOT NULL |
| definition_snapshot | JSONB | NOT NULL |
| change_summary | TEXT | |
| changed_by | VARCHAR(100) | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Table 8: `quality_gate_overrides`** -- Override audit records

| Column | Type | Constraints |
|--------|------|-------------|
| override_id | UUID | PK DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| gate_id | VARCHAR(10) | NOT NULL |
| original_score | NUMERIC(5,2) | NOT NULL |
| threshold | NUMERIC(5,2) | NOT NULL |
| justification | TEXT | NOT NULL |
| overridden_by | VARCHAR(100) | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Table 9: `agent_estimated_durations`** -- ETA estimation data

| Column | Type | Constraints |
|--------|------|-------------|
| agent_id | VARCHAR(20) | PK (composite) |
| commodity | VARCHAR(50) | PK (composite) |
| avg_duration_seconds | NUMERIC(10,2) | NOT NULL |
| p95_duration_seconds | NUMERIC(10,2) | NOT NULL |
| sample_count | INTEGER | NOT NULL DEFAULT 0 |
| last_updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

### 6.3 TimescaleDB Hypertables (4)

**Hypertable 10: `workflow_checkpoints`** -- Chunk interval: 7 days; Retention: 5 years

| Column | Type | Constraints |
|--------|------|-------------|
| checkpoint_id | UUID | DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| sequence_number | INTEGER | NOT NULL |
| phase | VARCHAR(30) | NOT NULL |
| agent_id | VARCHAR(20) | |
| gate_id | VARCHAR(10) | |
| checkpoint_type | VARCHAR(20) | NOT NULL |
| agent_statuses | JSONB | NOT NULL DEFAULT '{}' |
| agent_outputs_ref | VARCHAR(500) | |
| quality_gate_results | JSONB | DEFAULT '{}' |
| cumulative_provenance_hash | VARCHAR(64) | NOT NULL |
| created_by | VARCHAR(100) | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_id), (workflow_id, sequence_number)

**Hypertable 11: `quality_gate_evaluations`** -- Chunk interval: 30 days; Retention: 5 years

| Column | Type | Constraints |
|--------|------|-------------|
| evaluation_id | UUID | DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| gate_id | VARCHAR(10) | NOT NULL |
| result | VARCHAR(20) | NOT NULL |
| overall_score | NUMERIC(5,2) | NOT NULL |
| check_results | JSONB | NOT NULL DEFAULT '[]' |
| threshold_used | NUMERIC(5,2) | NOT NULL |
| is_simplified | BOOLEAN | DEFAULT FALSE |
| evaluated_by | VARCHAR(100) | |
| evaluated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_id), (gate_id), (result)

**Hypertable 12: `agent_execution_log`** -- Chunk interval: 7 days; Retention: 5 years

| Column | Type | Constraints |
|--------|------|-------------|
| log_id | UUID | DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| agent_id | VARCHAR(20) | NOT NULL |
| status | VARCHAR(20) | NOT NULL |
| attempt_number | INTEGER | NOT NULL DEFAULT 1 |
| duration_seconds | NUMERIC(10,3) | |
| output_ref | VARCHAR(500) | |
| output_summary | JSONB | |
| error_type | VARCHAR(50) | |
| error_message | TEXT | |
| error_classification | VARCHAR(20) | |
| started_at | TIMESTAMPTZ | |
| completed_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_id), (agent_id), (status)

**Hypertable 13: `workflow_audit_trail`** -- Chunk interval: 30 days; Retention: 5 years (Art. 31)

| Column | Type | Constraints |
|--------|------|-------------|
| audit_id | UUID | DEFAULT gen_random_uuid() |
| workflow_id | UUID | NOT NULL |
| event_type | VARCHAR(50) | NOT NULL |
| event_data | JSONB | NOT NULL DEFAULT '{}' |
| actor | VARCHAR(100) | NOT NULL |
| provenance_hash | VARCHAR(64) | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** (workflow_id), (event_type), (workflow_id, event_type)

### 6.4 Continuous Aggregates (2)

**1. `gl_eudr_ddo_daily_workflow_summary`** -- Daily workflow completion metrics by commodity and status

**2. `gl_eudr_ddo_daily_agent_performance`** -- Daily per-agent execution count, average duration, success/failure rates

---

## 7. API Architecture (~37 Endpoints)

**API Prefix:** `/v1/eudr-ddo`

### 7.1 Workflow Management (8 endpoints)

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/workflows` | `eudr-ddo:workflows:create` | Create and start workflow |
| GET | `/workflows` | `eudr-ddo:workflows:read` | List workflows with filters |
| GET | `/workflows/{id}` | `eudr-ddo:workflows:read` | Get workflow with progress |
| POST | `/workflows/{id}/pause` | `eudr-ddo:workflows:manage` | Pause workflow |
| POST | `/workflows/{id}/resume` | `eudr-ddo:workflows:manage` | Resume from checkpoint |
| POST | `/workflows/{id}/cancel` | `eudr-ddo:workflows:manage` | Cancel workflow |
| POST | `/workflows/{id}/clone` | `eudr-ddo:workflows:create` | Clone workflow |
| DELETE | `/workflows/{id}` | `eudr-ddo:workflows:delete` | Archive (soft delete) |

### 7.2 Templates (4), Phases (3), Quality Gates (4), State (4), Execution (3), Packages (4), Batch (2), Ops (5)

See Section 7 tables above for full endpoint specification. Total: 37 endpoints.

---

## 8. Integration Architecture

### 8.1 All 25 upstream agents integrated via unified AgentClient (see Section 8 agent tables above)

### 8.2 Infrastructure: PostgreSQL+TimescaleDB, Redis, S3, SEC-001/002/003/005, OBS-001/003

---

## 9. Zero-Hallucination Guarantees

### 9.1 Design Principles

1. **No LLM in orchestration path** -- All workflow execution, state management, quality gate evaluation, and composite scoring use deterministic algorithms
2. **No LLM in calculation path** -- Composite risk scores use fixed Decimal weights. Completeness scoring uses fixed Decimal weights. All arithmetic uses Python `Decimal` with explicit `ROUND_HALF_UP`
3. **Deterministic DAG execution** -- Topological sort with sorted tie-breaking guarantees identical execution order for identical workflow definitions
4. **Deterministic quality gates** -- Gate evaluation uses fixed check weights and thresholds. Same agent outputs always produce same gate result
5. **Bit-perfect reproducibility** -- Same workflow definition + same agent outputs = same composite score, same gate results, same package
6. **Complete provenance** -- Every checkpoint includes SHA-256 hash chain. Every gate evaluation recorded. Every package has integrity hash

### 9.2 LLM Usage Boundary

| Component | LLM Allowed | Deterministic Required |
|-----------|-----------|----------------------|
| DAG construction and validation | No | Kahn's algorithm |
| Topological sort and layer assignment | No | Deterministic with sorted tie-breaking |
| Agent invocation and coordination | No | Explicit dependency resolution |
| Completeness scoring (Phase 1) | No | Fixed Decimal weights |
| Composite risk calculation (Phase 2) | No | Fixed Decimal weights summing to 1.00 |
| Risk level classification | No | Fixed threshold ranges |
| Mitigation requirement determination | No | Fixed threshold comparison |
| Quality gate evaluation | No | Fixed check weights and thresholds |
| Checkpoint provenance hashing | No | SHA-256 deterministic chain |
| ETA calculation | No | Longest-path DAG algorithm |
| Backoff delay calculation | No | Exponential formula (bounded jitter) |
| Circuit breaker state transitions | No | Threshold-based FSM |
| DDS JSON assembly | No | Deterministic field mapping |
| Package integrity hash | No | SHA-256 of complete content |
| PDF narrative sections | Yes (optional) | Template text as default fallback |
| Executive summary generation | Yes (optional) | Template text as default fallback |

### 9.3 Provenance Chain Verification

```python
async def verify_provenance_chain(
    workflow_id: str,
    target_checkpoint_id: str = None,
) -> ProvenanceVerificationResult:
    """
    Verify integrity of the complete provenance chain for a workflow.
    Recomputes each hash from genesis and compares against stored values.
    Any mismatch indicates tampering.
    """
    checkpoints = await load_all_checkpoints(workflow_id)
    checkpoints.sort(key=lambda c: c.sequence_number)

    previous_hash = "GENESIS"
    for cp in checkpoints:
        recomputed = _compute_cumulative_hash(
            previous_hash=previous_hash,
            workflow_id=workflow_id,
            sequence=cp.sequence_number,
            checkpoint_type=cp.checkpoint_type,
            agent_id=cp.agent_id,
            gate_id=cp.gate_id,
            agent_statuses=cp.agent_statuses,
            timestamp=cp.created_at,
        )
        if recomputed != cp.cumulative_provenance_hash:
            return ProvenanceVerificationResult(
                valid=False,
                tampered_at_sequence=cp.sequence_number,
                expected_hash=recomputed,
                stored_hash=cp.cumulative_provenance_hash,
            )
        previous_hash = cp.cumulative_provenance_hash

    return ProvenanceVerificationResult(
        valid=True,
        chain_length=len(checkpoints),
        final_hash=previous_hash,
    )
```

---

## 10. Security Architecture

### 10.1 Authentication

- All endpoints (except `/health`) require JWT authentication via SEC-001
- JWT RS256 token validation with configurable JWKS endpoint
- Service-to-service authentication for inter-agent calls via signed service tokens
- Token expiry: 1 hour (configurable via `GL_EUDR_DDO_TOKEN_EXPIRY_S`)

### 10.2 Authorization (RBAC -- 19 Permissions)

| # | Permission | Description | Roles |
|---|------------|-------------|-------|
| 1 | `eudr-ddo:workflows:read` | View workflow status and progress | Viewer, Analyst, CO, Admin |
| 2 | `eudr-ddo:workflows:create` | Create and start workflows | Analyst, CO, Admin |
| 3 | `eudr-ddo:workflows:manage` | Pause, resume, cancel, clone | CO, Admin |
| 4 | `eudr-ddo:workflows:delete` | Archive workflows | Admin |
| 5 | `eudr-ddo:templates:read` | View workflow templates | Viewer, Analyst, CO, Admin |
| 6 | `eudr-ddo:templates:manage` | Create/modify custom templates | CO, Admin |
| 7 | `eudr-ddo:gates:read` | View quality gate evaluations | Viewer, Analyst, CO, Admin |
| 8 | `eudr-ddo:gates:override` | Override failed quality gates | CO, Admin |
| 9 | `eudr-ddo:checkpoints:read` | View checkpoint data | Viewer, Analyst, CO, Admin |
| 10 | `eudr-ddo:checkpoints:rollback` | Rollback to checkpoint | CO, Admin |
| 11 | `eudr-ddo:audit-trail:read` | View workflow audit trail | Auditor, CO, Admin |
| 12 | `eudr-ddo:packages:read` | View DD packages | Viewer, Analyst, CO, Admin |
| 13 | `eudr-ddo:packages:generate` | Generate DD packages | CO, Admin |
| 14 | `eudr-ddo:packages:download` | Download packages (PDF, ZIP, JSON) | CO, Admin |
| 15 | `eudr-ddo:batch:manage` | Create/manage batch workflows | CO, Admin |
| 16 | `eudr-ddo:circuit-breakers:read` | View circuit breaker states | Analyst, CO, Admin, Ops |
| 17 | `eudr-ddo:circuit-breakers:manage` | Reset circuit breakers | Admin, Ops |
| 18 | `eudr-ddo:dlq:read` | View dead letter queue | Analyst, CO, Admin, Ops |
| 19 | `eudr-ddo:dlq:manage` | Resolve DLQ entries | Admin, Ops |

*CO = Compliance Officer, Ops = Operations Engineer*

### 10.3 Data Security

| Requirement | Implementation |
|-------------|---------------|
| Encryption at rest | AES-256-GCM via SEC-003 for all workflow state and packages |
| Encryption in transit | TLS 1.3 via SEC-004 for all API and inter-agent communication |
| Data integrity | SHA-256 provenance hashes on every checkpoint and package |
| Audit trail immutability | TimescaleDB hypertable with append-only pattern; no UPDATE/DELETE |
| Secrets management | Vault via SEC-006 for agent credentials and API keys |
| PII protection | PII detection and redaction via SEC-011 in package outputs |
| Access logging | All API access logged via SEC-005 |
| Record retention | 5 years minimum per EUDR Article 31 |

### 10.4 Multi-Tenant Isolation

| Isolation Layer | Implementation |
|-----------------|---------------|
| Workflow isolation | Each workflow scoped to operator_id; cross-operator blocked by RBAC |
| Agent output isolation | Outputs stored in operator-scoped S3 prefixes |
| Circuit breaker isolation | Per-operator state (one operator's failures do not affect another) |
| Resource isolation | K8s resource quotas per operator tier |

### 10.5 Rate Limiting

- Default: 200 requests/minute per tenant
- Workflow creation: 50 requests/minute per tenant
- Package generation: 20 requests/minute per tenant
- Batch creation: 10 requests/minute per tenant
- Configurable via `GL_EUDR_DDO_RATE_LIMIT_*` environment variables

---

## 11. Observability Architecture

### 11.1 Prometheus Metrics (20)

**Prefix:** `gl_eudr_ddo_`

**Counters (8)**

| # | Metric | Labels | Description |
|---|--------|--------|-------------|
| 1 | `gl_eudr_ddo_workflows_started_total` | commodity, workflow_type | Workflows initiated |
| 2 | `gl_eudr_ddo_workflows_completed_total` | commodity, workflow_type | Workflows completed successfully |
| 3 | `gl_eudr_ddo_workflows_failed_total` | commodity, failure_phase | Workflows terminated due to failure |
| 4 | `gl_eudr_ddo_quality_gates_evaluated_total` | gate_id, result | Gate evaluations by gate and result |
| 5 | `gl_eudr_ddo_agent_invocations_total` | agent_id, status | Agent invocations by agent and outcome |
| 6 | `gl_eudr_ddo_retries_total` | agent_id, classification | Retry attempts by agent and error class |
| 7 | `gl_eudr_ddo_circuit_breaker_transitions_total` | agent_id, from_state, to_state | CB state changes |
| 8 | `gl_eudr_ddo_packages_generated_total` | commodity, format | Packages generated by commodity and format |

**Histograms (6)**

| # | Metric | Buckets | Description |
|---|--------|---------|-------------|
| 9 | `gl_eudr_ddo_workflow_duration_seconds` | 60, 120, 180, 240, 300, 600 | End-to-end workflow duration |
| 10 | `gl_eudr_ddo_agent_execution_seconds` | 1, 5, 10, 30, 60, 120 | Individual agent execution time |
| 11 | `gl_eudr_ddo_quality_gate_evaluation_seconds` | 0.1, 0.5, 1, 2, 5 | Gate evaluation latency |
| 12 | `gl_eudr_ddo_checkpoint_write_seconds` | 0.01, 0.05, 0.1, 0.2, 0.5 | Checkpoint persistence latency |
| 13 | `gl_eudr_ddo_package_generation_seconds` | 1, 5, 10, 20, 30, 60 | Package generation latency |
| 14 | `gl_eudr_ddo_api_request_seconds` | 0.01, 0.05, 0.1, 0.2, 0.5, 1 | API request latency |

**Gauges (6)**

| # | Metric | Description |
|---|--------|-------------|
| 15 | `gl_eudr_ddo_active_workflows` | Currently active workflows |
| 16 | `gl_eudr_ddo_active_agents` | Currently executing agent invocations |
| 17 | `gl_eudr_ddo_paused_workflows` | Currently paused workflows |
| 18 | `gl_eudr_ddo_circuit_breakers_open` | Number of open circuit breakers |
| 19 | `gl_eudr_ddo_dlq_unresolved` | Unresolved dead letter queue entries |
| 20 | `gl_eudr_ddo_parallelization_efficiency` | Current parallel efficiency ratio (0.0-1.0) |

### 11.2 Grafana Dashboard

**Dashboard:** `eudr-due-diligence-orchestrator.json`
**Panels:** 20

| # | Panel | Type | Data Source |
|---|-------|------|-------------|
| 1 | Workflow Start Rate | Time series | Counter #1 |
| 2 | Workflow Completion Rate | Time series | Counter #2 |
| 3 | Workflow Failure Rate | Time series | Counter #3 |
| 4 | Active Workflows | Stat | Gauge #15 |
| 5 | Active Agent Executions | Stat | Gauge #16 |
| 6 | Workflow Duration (p50/p95/p99) | Time series | Histogram #9 |
| 7 | Agent Execution Duration by Agent | Heatmap | Histogram #10 |
| 8 | Quality Gate Pass/Fail Rate | Stacked bar | Counter #4 |
| 9 | Quality Gate Evaluation Latency | Time series | Histogram #11 |
| 10 | Agent Invocation Success Rate | Time series | Counter #5 |
| 11 | Retry Rate by Agent | Bar chart | Counter #6 |
| 12 | Circuit Breaker Status | Status map | Gauge #18 |
| 13 | Circuit Breaker Transitions | Time series | Counter #7 |
| 14 | Checkpoint Write Latency | Time series | Histogram #12 |
| 15 | Package Generation Rate | Time series | Counter #8 |
| 16 | Package Generation Latency | Time series | Histogram #13 |
| 17 | DLQ Unresolved Count | Stat (alert) | Gauge #19 |
| 18 | Parallelization Efficiency | Gauge | Gauge #20 |
| 19 | API Latency by Endpoint | Heatmap | Histogram #14 |
| 20 | Paused Workflows | Stat | Gauge #17 |

### 11.3 Alerting Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Workflow SLA Breach | `gl_eudr_ddo_workflow_duration_seconds` p99 > 300s | Warning | Investigate slowest agents |
| High Failure Rate | `gl_eudr_ddo_workflows_failed_total` > 10% in 1h | Critical | Page on-call |
| Circuit Breaker Open | `gl_eudr_ddo_circuit_breakers_open` > 0 for 5m | Warning | Check upstream agent health |
| DLQ Accumulation | `gl_eudr_ddo_dlq_unresolved` > 10 | Warning | Review and resolve DLQ entries |
| Quality Gate Override Spike | `gl_eudr_ddo_quality_gates_evaluated_total{result="overridden"}` > 5 in 1h | Warning | Review override justifications |
| Checkpoint Latency SLA | `gl_eudr_ddo_checkpoint_write_seconds` p99 > 0.5s | Warning | Check PostgreSQL performance |
| Low Parallelization Efficiency | `gl_eudr_ddo_parallelization_efficiency` < 0.5 for 15m | Info | Investigate resource contention |
| Active Workflow Surge | `gl_eudr_ddo_active_workflows` > 500 | Info | Verify HPA scaling |

### 11.4 OpenTelemetry Tracing

- Root span per workflow execution (captures full 3-phase lifecycle)
- Child span per phase (information_gathering, risk_assessment, risk_mitigation, package_generation)
- Child span per agent invocation (25 agents)
- Child span per quality gate evaluation
- Child span per checkpoint write
- Child span per package generation step
- Trace context propagated to all 25 upstream agents via W3C TraceContext headers
- Total spans per standard workflow: ~40 (1 root + 4 phases + 25 agents + 3 gates + ~7 internal)

---

## 12. Testing Strategy

### 12.1 Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Workflow Definition (E1) | 100+ | DAG creation, validation, topological sort, cycle detection, templates |
| Info Gathering Coordinator (E2) | 80+ | Phase 1 orchestration, dependency, data handoff, completeness |
| Risk Assessment Coordinator (E3) | 80+ | Phase 2 orchestration, parallel risk, composite scoring |
| Risk Mitigation Coordinator (E4) | 60+ | Phase 3 orchestration, adequacy, bypass |
| Quality Gate Engine (E5) | 100+ | All 3 gates, checks, thresholds, override, relaxed mode |
| Workflow State Manager (E6) | 120+ | Checkpoint, resume, rollback, pause, cancel, audit trail |
| Parallel Execution Engine (E7) | 80+ | Concurrency, work-stealing, critical path, ETA |
| Error Recovery Manager (E8) | 100+ | Backoff, circuit breaker FSM, fallback, DLQ |
| Package Generator (E9) | 80+ | DDS JSON, PDF, multi-language, schema validation, hash |
| API Routes | 100+ | All 37 endpoints, auth, pagination, errors, batch |
| Golden Workflow Tests | 49 | 7 commodities x 7 scenarios |
| Integration Tests | 40+ | E2E with mock agents, cross-phase flow, gate transitions |
| Performance Tests | 30+ | 1K/5K/10K shipments, concurrent load, checkpoint throughput |
| Chaos Tests | 20+ | Agent failure, network partition, DB unavailability |
| Determinism Tests | 15+ | Bit-perfect reproducibility for all deterministic paths |
| Security Tests | 10+ | RBAC enforcement, tenant isolation |
| **Total** | **1,064+** | |

### 12.2 Golden Workflow Test Scenarios (49)

7 commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) x 7 scenarios:

| # | Scenario | Expected Outcome |
|---|----------|-----------------|
| 1 | Complete standard workflow | All 25 agents; all 3 QGs pass; package generated |
| 2 | Simplified workflow (low-risk) | Reduced agents; relaxed gates; simplified package |
| 3 | QG-1 failure (missing geolocation) | Gate fails; gap report; remediation guidance |
| 4 | QG-2 failure (incomplete risk) | Gate fails; missing dimensions identified |
| 5 | High-risk requiring enhanced mitigation | Full mitigation; enhanced evidence required |
| 6 | Agent failure with successful retry | Transient failure retried; workflow completes |
| 7 | Agent failure with circuit breaker | CB opens; degraded mode with fallback |

### 12.3 Test Coverage Targets

| Coverage Type | Target |
|---------------|--------|
| Line coverage | >= 85% |
| Branch coverage | >= 80% |
| API endpoint coverage | 100% |
| Quality gate coverage | 100% (all 3 gates, all checks) |
| Commodity template coverage | 100% (all 7 + simplified) |
| Deterministic path coverage | 100% |
| Error classification coverage | 100% |

---

## 13. Deployment Architecture

### 13.1 Containerization

```yaml
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY greenlang/agents/eudr/due_diligence_orchestrator/ ./greenlang/agents/eudr/due_diligence_orchestrator/
COPY greenlang/infrastructure/ ./greenlang/infrastructure/
CMD ["uvicorn", "greenlang.agents.eudr.due_diligence_orchestrator.api.router:app", "--host", "0.0.0.0", "--port", "8026"]
EXPOSE 8026
HEALTHCHECK CMD curl -f http://localhost:8026/health || exit 1
```

### 13.2 Kubernetes Resources

| Resource | Configuration |
|----------|--------------|
| Deployment | 2-8 replicas (HPA: CPU 70%, custom metric: active workflows) |
| Service | ClusterIP, port 8026 |
| Ingress | `/v1/eudr-ddo` path routing via Kong API Gateway |
| ConfigMap | Non-sensitive config (`GL_EUDR_DDO_*` env vars) |
| Secret | DB credentials, Redis URL, S3 keys, service tokens |
| Resource Limits | CPU: 2 (req) / 8 (limit), Memory: 4Gi (req) / 16Gi (limit) |
| Readiness Probe | HTTP GET `/health`, period 10s, timeout 5s |
| Liveness Probe | HTTP GET `/health`, period 30s, timeout 10s |

### 13.3 Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eudr-ddo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eudr-due-diligence-orchestrator
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gl_eudr_ddo_active_workflows
      target:
        type: AverageValue
        averageValue: 50
```

### 13.4 Resource Requirements

| Resource | Development | Staging | Production |
|----------|-------------|---------|------------|
| Replicas | 1 | 2 | 2-20 (HPA) |
| CPU (per pod) | 1 core | 2 cores | 2-8 cores |
| Memory (per pod) | 2 Gi | 4 Gi | 4-16 Gi |
| PostgreSQL | Shared dev | Dedicated | HA cluster |
| Redis | Shared dev | Dedicated | HA cluster |
| S3 | Local MinIO | AWS S3 | AWS S3 |

---

## 14. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System DDS schema changes | Medium | High | Adapter pattern isolates schema mapping; Package Generator updatable independently |
| R2 | Upstream agent API contract breaking changes | Medium | High | AgentClient abstraction; version negotiation; CI contract tests |
| R3 | Workflow complexity exceeds performance targets | Low | High | S3 offload for large outputs; lazy loading; continuous benchmarking |
| R4 | Quality gate thresholds too strict/lenient | Medium | Medium | Configurable per-operator; override capability; beta tuning |
| R5 | Circuit breaker cascade across operators | Medium | High | Per-operator isolation; degraded mode for non-critical agents |
| R6 | Checkpoint volume grows unbounded | Low | Medium | TimescaleDB compression; S3 tiering; 5-year retention policy |
| R7 | Concurrent workflow overload exhausts DB | Medium | High | Connection pooling; read replicas; Redis rate limiting; HPA |
| R8 | Simplified DD misapplied to standard-risk | Low | High | Validation against country classification; audit trail flags type |
| R9 | Package generation fails for 10K+ shipments | Low | Medium | Streaming generation; chunked PDF; async with notification |
| R10 | Article 8 regulatory structure changes | Low | High | Modular 3-phase architecture; phases extensible via templates |

---

## 15. Technology Stack Summary

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | 0.104.0+ | Async, OpenAPI, Pydantic v2 native |
| DAG Engine | NetworkX | 3.2+ | Topological sort, critical path, dependency analysis |
| Async Runtime | asyncio + anyio | stdlib | Structured concurrency for parallel agents |
| Task Distribution | Redis Streams | 7.0+ | Work-stealing, event dispatch, progress pub/sub |
| Database | PostgreSQL + TimescaleDB | 14+ / 2.12+ | Persistent state + time-series hypertables |
| Cache | Redis | 7.0+ | Agent output cache, circuit breaker state |
| Object Storage | S3 (AWS/MinIO) | -- | Large outputs, packages, PDF reports |
| Serialization | Pydantic | 2.5.0+ | Type-safe validated workflow models |
| PDF Generation | ReportLab + WeasyPrint | -- | Audit-ready PDF due diligence reports |
| Authentication | JWT (RS256) | -- | SEC-001 standard |
| Authorization | RBAC | -- | SEC-002 with 19 `eudr-ddo:` permissions |
| Monitoring | Prometheus + Grafana | -- | 20 metrics + 20-panel dashboard |
| Tracing | OpenTelemetry | -- | Distributed tracing across 25 agent calls |
| HTTP Client | httpx | 0.25.0+ | Async HTTP for agent invocations |
| Hashing | hashlib (stdlib) | -- | SHA-256 provenance chains |
| Deployment | Kubernetes (EKS) | -- | HPA auto-scaling |

---

*End of Architecture Specification*
