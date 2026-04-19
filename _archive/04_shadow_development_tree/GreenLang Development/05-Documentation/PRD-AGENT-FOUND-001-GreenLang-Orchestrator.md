# PRD: AGENT-FOUND-001 - GreenLang Orchestrator
# DAG Execution Engine with Retry/Timeout & Determinism Guarantees

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-001 |
| **Title** | GreenLang Orchestrator - DAG Execution Engine |
| **Priority** | P0 - Critical Foundation |
| **Category** | Foundations Agent |
| **Status** | Draft |
| **Author** | GreenLang Platform Team |
| **Date** | 2026-02-07 |
| **Target** | Production Ready |

---

## 1. Executive Summary

AGENT-FOUND-001 delivers a production-grade **Directed Acyclic Graph (DAG) execution engine** that replaces the current linear workflow orchestrator with a proper graph-based scheduler. This is a P0 foundation component - every GreenLang application (CSRD, CBAM, VCCI, SB253, EUDR, Taxonomy, etc.) depends on reliable, deterministic, and auditable agent orchestration.

The engine provides:
- **DAG-based workflow definition** with explicit dependency edges, topological sort, and cycle detection
- **Parallel level execution** - automatically identifies and parallelizes independent nodes
- **Per-node retry/timeout policies** - configurable exponential backoff, jitter, and timeout per DAG node
- **Deterministic execution** - sorted scheduling, execution replay, hash-based provenance for regulatory audit
- **DAG-aware checkpointing** - node-level checkpoint/resume, never re-execute completed nodes
- **Zero-hallucination integration** - wraps calculation nodes with determinism guarantees

---

## 2. Current State Analysis

### 2.1 Existing Components (to integrate with, NOT replace)

| Component | Location | What It Provides | Gap |
|-----------|----------|------------------|-----|
| Orchestrator | `execution/core/orchestrator.py` | Linear sequential execution with policy + checkpoint | No DAG, no parallel |
| AsyncOrchestrator | `execution/core/async_orchestrator.py` | Basic parallel grouping by input_mapping | Heuristic-based, not true DAG |
| Workflow/WorkflowStep | `execution/core/workflow.py` | Linear step list, YAML/JSON loading | No dependency edges |
| DependencyGraph | `agent_factory/dependencies/graph.py` | DAG for agent version resolution | Not for execution scheduling |
| CoordinationLayer | `execution/core/coordination_layer.py` | Locks, sagas, consensus | Not DAG execution |
| Composability (GLEL) | `execution/core/composability.py` | Pipe chains, parallel branches | Not DAG |
| CheckpointManager | `execution/pipeline/checkpointing.py` | Per-pipeline file/DB/Redis checkpoints | Not DAG-node-aware |
| Retry decorators | `execution/resilience/retry.py` | Exponential backoff, jitter | Standalone decorators |
| Determinism | `greenlang/determinism.py` | Clock, random, decimal, UUID, hash | Not integrated into execution |

### 2.2 Architecture Decision

AGENT-FOUND-001 is a **new module** at `greenlang/orchestrator/` that:
- **Imports and builds upon** existing resilience, determinism, and checkpoint primitives
- **Does NOT modify** existing orchestrator/workflow code (backward compatible)
- **Provides** a new `DAGOrchestrator` that is the recommended orchestration path going forward
- **Integrates** with existing auth, audit, alerting, and metrics infrastructure

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-001: DAG Workflow Definition
- Define workflows as directed acyclic graphs with explicit node dependencies
- Support `depends_on` declarations for each node (list of predecessor node IDs)
- Nodes reference agents by agent_id with input/output mapping
- YAML and JSON workflow loading/export
- Programmatic builder pattern for DAG construction
- Workflow versioning with hash-based integrity

#### FR-002: DAG Validation
- Cycle detection using DFS-based algorithm
- Unreachable node detection (nodes with no path from any root)
- Missing dependency validation (edges to non-existent nodes)
- Duplicate node ID detection
- Self-dependency detection
- Validation returns structured error list (not exceptions for warnings)

#### FR-003: Topological Sort & Level Scheduling
- Kahn's algorithm for topological ordering (deterministic via sorted tie-breaking)
- Level-based grouping: nodes at same depth can execute in parallel
- Concurrency limits per level (configurable max_parallel_nodes)
- Priority-based tie-breaking within levels (node priority field)

#### FR-004: DAG Execution Engine
- Async execution loop processing one level at a time
- Within each level, independent nodes execute concurrently (asyncio.gather)
- Node execution wraps agent.run() / agent.run_async() calls
- Execution context propagation (results from predecessors available to successors)
- Conditional node execution (skip based on predecessor results)
- Dynamic DAG modification at runtime (optional nodes based on conditions)
- Resource-aware scheduling (respect concurrency limits)

#### FR-005: Per-Node Retry/Timeout Policies
- Each node has its own `RetryPolicy`: max_retries, strategy (exponential/linear/constant/fibonacci), base_delay, max_delay, jitter, retryable_exceptions
- Each node has its own `TimeoutPolicy`: timeout_seconds, on_timeout (fail/skip/compensate)
- Default policies inherited from DAG-level configuration
- Node-level override of DAG-level defaults
- Integration with existing `RetryConfig` from `execution/resilience/retry.py`

#### FR-006: Deterministic Execution
- Sorted node ordering within parallel levels (alphabetical by node_id for deterministic scheduling)
- `DeterministicClock` for all timestamps in execution traces
- `content_hash` (SHA-256) for all node inputs and outputs
- Execution replay: given same DAG + same inputs, produce identical execution trace
- Seed-based randomness for any stochastic operations
- Deterministic execution IDs using `deterministic_uuid`

#### FR-007: DAG-Aware Checkpointing
- Checkpoint after each node completes (node_id, status, outputs, hash)
- Resume from any point: load checkpoint, skip completed nodes, re-execute from failure point
- Checkpoint storage: in-memory (dev), file (staging), PostgreSQL (prod)
- Checkpoint integrity verification (hash comparison on resume)
- Automatic checkpoint cleanup (configurable retention)

#### FR-008: Execution Provenance & Audit Trail
- Complete execution trace stored per run: DAG topology, execution order, timing, I/O hashes
- Each node produces a `NodeProvenance` record: input_hash, output_hash, duration_ms, attempt_count, parent_hashes
- Chain provenance: SHA-256 hash chain linking all node provenances
- Export execution trace as JSON for regulatory audit
- Queryable execution history (by dag_id, date range, status)

#### FR-009: On-Failure Strategies
- Per-node `on_failure`: stop (halt DAG), skip (continue without result), compensate (run compensation handler)
- DAG-level `on_failure`: fail_fast (stop at first failure), continue (execute all possible nodes), compensate (saga-style rollback)
- Compensation handlers: optional per-node rollback functions
- Partial result collection even on failure

#### FR-010: REST API (20 endpoints)
- `POST /api/v1/orchestrator/dags` - Create DAG workflow
- `GET /api/v1/orchestrator/dags` - List DAG workflows
- `GET /api/v1/orchestrator/dags/{dag_id}` - Get DAG workflow
- `PUT /api/v1/orchestrator/dags/{dag_id}` - Update DAG workflow
- `DELETE /api/v1/orchestrator/dags/{dag_id}` - Delete DAG workflow
- `POST /api/v1/orchestrator/dags/{dag_id}/validate` - Validate DAG
- `POST /api/v1/orchestrator/dags/{dag_id}/execute` - Execute DAG
- `GET /api/v1/orchestrator/executions` - List executions
- `GET /api/v1/orchestrator/executions/{execution_id}` - Get execution details
- `GET /api/v1/orchestrator/executions/{execution_id}/trace` - Get execution trace
- `POST /api/v1/orchestrator/executions/{execution_id}/cancel` - Cancel execution
- `POST /api/v1/orchestrator/executions/{execution_id}/resume` - Resume from checkpoint
- `GET /api/v1/orchestrator/executions/{execution_id}/provenance` - Get provenance chain
- `GET /api/v1/orchestrator/checkpoints/{execution_id}` - Get checkpoints
- `DELETE /api/v1/orchestrator/checkpoints/{execution_id}` - Delete checkpoints
- `GET /api/v1/orchestrator/metrics` - Get orchestrator metrics
- `POST /api/v1/orchestrator/dags/import` - Import DAG from YAML
- `GET /api/v1/orchestrator/dags/{dag_id}/export` - Export DAG to YAML
- `GET /api/v1/orchestrator/dags/{dag_id}/visualize` - Get DAG visualization (Mermaid)
- `GET /api/v1/orchestrator/health` - Health check

#### FR-011: Prometheus Metrics (12 metrics)
- `gl_orchestrator_dag_executions_total` (counter, labels: dag_id, status)
- `gl_orchestrator_dag_execution_duration_seconds` (histogram, labels: dag_id)
- `gl_orchestrator_node_executions_total` (counter, labels: dag_id, node_id, status)
- `gl_orchestrator_node_execution_duration_seconds` (histogram, labels: dag_id, node_id)
- `gl_orchestrator_node_retries_total` (counter, labels: dag_id, node_id)
- `gl_orchestrator_node_timeouts_total` (counter, labels: dag_id, node_id)
- `gl_orchestrator_active_executions` (gauge)
- `gl_orchestrator_checkpoint_operations_total` (counter, labels: operation)
- `gl_orchestrator_checkpoint_size_bytes` (histogram)
- `gl_orchestrator_provenance_chain_length` (histogram, labels: dag_id)
- `gl_orchestrator_parallel_nodes_per_level` (histogram, labels: dag_id)
- `gl_orchestrator_dag_validation_errors_total` (counter, labels: error_type)

### 3.2 Non-Functional Requirements

| NFR | Requirement | Target |
|-----|-------------|--------|
| NFR-001 | DAG validation latency | < 10ms for 100-node DAG |
| NFR-002 | Execution overhead per node | < 5ms scheduling overhead |
| NFR-003 | Checkpoint save latency | < 50ms (memory), < 200ms (file) |
| NFR-004 | Concurrent DAG executions | 50+ simultaneous |
| NFR-005 | Max DAG size | 500+ nodes |
| NFR-006 | Deterministic replay | 100% identical traces for same inputs |
| NFR-007 | Test coverage | >= 85% |
| NFR-008 | Zero-hallucination | All calculation nodes wrapped |

---

## 4. Architecture

### 4.1 Module Structure

```
greenlang/orchestrator/
    __init__.py              # Public API exports
    config.py                # OrchestratorConfig with GL_ORCHESTRATOR_ env prefix
    models.py                # DAGWorkflow, DAGNode, DAGEdge, ExecutionTrace, NodeProvenance
    dag_builder.py           # DAGBuilder programmatic construction + YAML/JSON loading
    dag_validator.py         # Cycle detection, unreachable nodes, structural validation
    topological_sort.py      # Kahn's algorithm with deterministic tie-breaking
    dag_executor.py          # Core DAG execution engine (async, level-based parallel)
    node_runner.py           # Individual node execution with retry/timeout
    retry_policy.py          # Per-node retry policy (wraps execution/resilience/retry.py)
    timeout_policy.py        # Per-node timeout policy
    checkpoint_store.py      # DAG-aware checkpoint storage (memory, file, PostgreSQL)
    provenance.py            # Execution provenance tracking and chain hashing
    determinism.py           # Deterministic scheduling, replay support
    metrics.py               # Prometheus metric definitions
    setup.py                 # configure_orchestrator(app) + DAGOrchestrator facade
    api/
        __init__.py          # API package
        router.py            # FastAPI router with 20 endpoints
```

### 4.2 Key Classes

```python
# models.py
class DAGNode:
    node_id: str
    agent_id: str
    depends_on: List[str]        # predecessor node IDs
    input_mapping: Dict[str, str]
    output_key: str
    condition: Optional[str]
    retry_policy: Optional[RetryPolicy]
    timeout_policy: Optional[TimeoutPolicy]
    on_failure: str              # stop, skip, compensate
    compensation_handler: Optional[str]
    priority: int                # tie-breaking within levels
    metadata: Dict[str, Any]

class DAGWorkflow:
    dag_id: str
    name: str
    description: str
    version: str
    nodes: Dict[str, DAGNode]
    default_retry_policy: RetryPolicy
    default_timeout_policy: TimeoutPolicy
    on_failure: str              # fail_fast, continue, compensate
    max_parallel_nodes: int
    metadata: Dict[str, Any]

class ExecutionTrace:
    execution_id: str
    dag_id: str
    status: ExecutionStatus
    node_traces: Dict[str, NodeTrace]
    topology_levels: List[List[str]]
    start_time: datetime
    end_time: Optional[datetime]
    provenance_chain_hash: str

class NodeProvenance:
    node_id: str
    input_hash: str
    output_hash: str
    duration_ms: float
    attempt_count: int
    parent_hashes: List[str]     # hashes of predecessor provenances
    chain_hash: str
```

### 4.3 Execution Flow

```
1. DAG Definition -> Validate (cycle check, structure) -> Store
2. Execute Request -> Load DAG -> Topological Sort -> Level Grouping
3. For each level:
   a. Collect ready nodes (all predecessors completed)
   b. Sort nodes deterministically (by node_id)
   c. Execute nodes in parallel (up to max_parallel_nodes)
   d. For each node:
      - Prepare input (map from predecessor outputs)
      - Apply retry policy (exponential backoff with jitter)
      - Apply timeout policy
      - Execute agent.run() / agent.run_async()
      - Calculate provenance (input/output hash)
      - Save checkpoint
      - Handle failure (stop/skip/compensate)
4. Collect results -> Build execution trace -> Store provenance
```

### 4.4 Integration Points

| System | Integration |
|--------|-------------|
| Auth (SEC-001/002) | JWT + RBAC on all API endpoints |
| Audit (SEC-005) | Execution start/complete/fail events |
| Alerting (OBS-004) | Execution failure alerts via AlertingBridge |
| Prometheus (OBS-001) | 12 metrics exposed at /metrics |
| Tracing (OBS-003) | OTel spans per DAG execution + per node |
| Feature Flags (INFRA-008) | Feature-gate new orchestrator rollout |
| Agent Factory (INFRA-010) | Resolve agent instances for node execution |

---

## 5. Database Schema

### V021: Orchestrator Service Migration

```sql
-- DAG workflow definitions
CREATE TABLE dag_workflows (
    dag_id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    version VARCHAR(32) NOT NULL DEFAULT '1.0.0',
    definition JSONB NOT NULL,  -- Full DAG definition
    hash VARCHAR(64) NOT NULL,  -- SHA-256 of definition
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128),
    tenant_id VARCHAR(128)
);

-- DAG execution records
CREATE TABLE dag_executions (
    execution_id VARCHAR(128) PRIMARY KEY,
    dag_id VARCHAR(128) NOT NULL REFERENCES dag_workflows(dag_id),
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    topology_levels JSONB,     -- Level grouping snapshot
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT,
    provenance_chain_hash VARCHAR(64),
    tenant_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Node execution traces (hypertable for time-series)
CREATE TABLE node_traces (
    trace_id VARCHAR(128) PRIMARY KEY,
    execution_id VARCHAR(128) NOT NULL REFERENCES dag_executions(execution_id),
    node_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    duration_ms DOUBLE PRECISION,
    attempt_count INTEGER DEFAULT 1,
    error TEXT,
    provenance_hash VARCHAR(64),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE (execution_id, node_id)
);
SELECT create_hypertable('node_traces', 'started_at', if_not_exists => TRUE);

-- DAG checkpoints
CREATE TABLE dag_checkpoints (
    checkpoint_id VARCHAR(128) PRIMARY KEY,
    execution_id VARCHAR(128) NOT NULL REFERENCES dag_executions(execution_id),
    node_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    outputs JSONB,
    output_hash VARCHAR(64),
    attempt_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (execution_id, node_id)
);

-- Execution provenance chain
CREATE TABLE execution_provenance (
    provenance_id VARCHAR(128) PRIMARY KEY,
    execution_id VARCHAR(128) NOT NULL REFERENCES dag_executions(execution_id),
    node_id VARCHAR(128) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    parent_hashes JSONB NOT NULL DEFAULT '[]',
    chain_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_dag_executions_dag_id ON dag_executions(dag_id);
CREATE INDEX idx_dag_executions_status ON dag_executions(status);
CREATE INDEX idx_dag_executions_tenant ON dag_executions(tenant_id);
CREATE INDEX idx_node_traces_execution ON node_traces(execution_id);
CREATE INDEX idx_dag_checkpoints_execution ON dag_checkpoints(execution_id);
CREATE INDEX idx_execution_provenance_execution ON execution_provenance(execution_id);

-- RLS policies
ALTER TABLE dag_workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE dag_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE node_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE dag_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE execution_provenance ENABLE ROW LEVEL SECURITY;
```

---

## 6. API Specification

### 6.1 Create DAG Workflow
```
POST /api/v1/orchestrator/dags
Content-Type: application/json

{
  "name": "emissions-calculation",
  "description": "Calculate Scope 1+2+3 emissions",
  "nodes": {
    "intake": {
      "agent_id": "intake_agent",
      "depends_on": [],
      "output_key": "raw_data"
    },
    "validate": {
      "agent_id": "validation_agent",
      "depends_on": ["intake"],
      "input_mapping": {"data": "results.intake.raw_data"},
      "output_key": "validated_data"
    },
    "scope1": {
      "agent_id": "scope1_calc_agent",
      "depends_on": ["validate"],
      "retry_policy": {"max_retries": 3, "strategy": "exponential"},
      "timeout_policy": {"timeout_seconds": 120}
    },
    "scope2": {
      "agent_id": "scope2_calc_agent",
      "depends_on": ["validate"],
      "retry_policy": {"max_retries": 3, "strategy": "exponential"}
    },
    "aggregate": {
      "agent_id": "aggregation_agent",
      "depends_on": ["scope1", "scope2"],
      "output_key": "total_emissions"
    },
    "report": {
      "agent_id": "reporting_agent",
      "depends_on": ["aggregate"]
    }
  },
  "default_retry_policy": {"max_retries": 2, "strategy": "exponential", "base_delay": 1.0},
  "default_timeout_policy": {"timeout_seconds": 60},
  "max_parallel_nodes": 10
}
```

Response: `201 Created` with DAG definition including computed topology levels.

### 6.2 Execute DAG
```
POST /api/v1/orchestrator/dags/{dag_id}/execute
Content-Type: application/json

{
  "input_data": {"facility_id": "FAC-001", "period": "2025-Q4"},
  "execution_options": {
    "checkpoint_enabled": true,
    "deterministic_mode": true
  }
}
```

Response: `202 Accepted` with execution_id for tracking.

### 6.3 Get Execution Trace
```
GET /api/v1/orchestrator/executions/{execution_id}/trace
```

Response: Full execution trace with topology, node timings, provenance hashes.

---

## 7. Testing Requirements

### 7.1 Unit Tests (target: 250+)
- DAG validation (cycle detection, unreachable nodes, duplicates)
- Topological sort (deterministic ordering, level grouping)
- Node runner (retry, timeout, failure handling)
- Checkpoint store (save, load, resume, cleanup)
- Provenance (hash calculation, chain linking)
- Determinism (replay produces identical traces)
- Config (env vars, defaults, validation)
- Models (serialization, YAML/JSON loading)
- Metrics (counter increments, histogram observations)

### 7.2 Integration Tests (target: 40+)
- End-to-end DAG execution (simple, complex, diamond)
- Checkpoint resume after failure
- Concurrent DAG executions
- API endpoint testing
- Auth integration
- Database persistence

### 7.3 Load Tests (target: 10+)
- 50 concurrent DAG executions
- 500-node DAG validation
- Checkpoint throughput
- API latency under load

### Coverage Target: >= 85%

---

## 8. Success Criteria

| Criteria | Target |
|----------|--------|
| All unit tests pass | 250+ tests, 0 failures |
| Integration tests pass | 40+ tests, 0 failures |
| Test coverage | >= 85% |
| DAG validation < 10ms | For 100-node DAGs |
| Node scheduling overhead < 5ms | Per node |
| Deterministic replay | 100% identical traces |
| Checkpoint resume works | After any node failure |
| API endpoints functional | All 20 endpoints |
| Prometheus metrics exposed | All 12 metrics |
| Backward compatible | Existing orchestrator unchanged |

---

## 9. Rollout Plan

### Phase 1: Core Engine (Priority)
- Models, config, DAG builder, validator, topological sort
- DAG executor, node runner, retry/timeout policies
- In-memory checkpoint store
- Provenance tracking, determinism

### Phase 2: Infrastructure
- Database migration (V021)
- PostgreSQL/file checkpoint stores
- Kubernetes manifests
- CI/CD pipeline

### Phase 3: API & Integration
- REST API (20 endpoints)
- Auth integration (JWT + RBAC)
- Prometheus metrics
- Alerting bridge (OBS-004)

### Phase 4: Testing & Documentation
- Unit tests (250+)
- Integration tests (40+)
- Load tests (10+)
- Runbooks (4)
- API documentation

---

## 10. Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >= 3.11 | Runtime |
| FastAPI | >= 0.104 | REST API |
| Pydantic | >= 2.0 | Data models |
| asyncio | stdlib | Async execution |
| psycopg | >= 3.1 | PostgreSQL (optional) |
| redis | >= 5.0 | Redis cache (optional) |
| PyYAML | >= 6.0 | YAML loading |
| prometheus_client | >= 0.19 | Metrics |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Complex DAGs cause deadlocks | High | Cycle detection at validation; timeout per execution |
| Non-deterministic parallel execution | High | Sorted node ordering; deterministic clock |
| Checkpoint corruption | Medium | SHA-256 integrity verification on resume |
| Memory pressure from large DAGs | Medium | Streaming execution; configurable concurrency limits |
| Backward compatibility issues | Low | New module, existing code untouched |

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| DAG | Directed Acyclic Graph - workflow where nodes have directed edges with no cycles |
| Topological Sort | Linear ordering of nodes where every edge u->v has u before v |
| Level | Group of nodes at same depth that can execute in parallel |
| Provenance | Audit trail of execution with cryptographic hash chain |
| Checkpoint | Saved state of DAG execution for resume capability |
| Burn Rate | Not applicable (see OBS-005 for SLO burn rate) |
