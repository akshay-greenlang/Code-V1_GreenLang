# AGENT-FOUND-001: GreenLang Orchestrator - Development Tasks

## Phase 1: Core Engine (Backend Developer)

- [ ] Create `greenlang/orchestrator/__init__.py` with public API exports (50+ symbols)
- [ ] Create `greenlang/orchestrator/config.py` - OrchestratorConfig with `GL_ORCHESTRATOR_` env prefix, defaults, validation
- [ ] Create `greenlang/orchestrator/models.py` - DAGNode, DAGEdge, DAGWorkflow, RetryPolicy, TimeoutPolicy, ExecutionStatus, ExecutionTrace, NodeTrace, NodeProvenance, DAGCheckpoint (all Pydantic v2 / dataclass)
- [ ] Create `greenlang/orchestrator/dag_builder.py` - DAGBuilder programmatic construction, from_yaml(), from_json(), from_dict(), to_yaml(), to_json(), add_node(), add_edge(), with_defaults()
- [ ] Create `greenlang/orchestrator/dag_validator.py` - validate_dag(): cycle detection (DFS), unreachable nodes, missing dependencies, duplicate IDs, self-dependencies; returns structured error list
- [ ] Create `greenlang/orchestrator/topological_sort.py` - Kahn's algorithm with deterministic tie-breaking (sorted by node_id), level_grouping() returns List[List[str]] of parallel levels
- [ ] Create `greenlang/orchestrator/node_runner.py` - NodeRunner class: execute single node with retry policy + timeout policy, wraps agent.run()/run_async(), handles sync/async agents, returns NodeExecutionResult
- [ ] Create `greenlang/orchestrator/retry_policy.py` - RetryPolicy class integrating with `execution/resilience/retry.py` RetryConfig, per-node configuration, exponential/linear/constant/fibonacci strategies
- [ ] Create `greenlang/orchestrator/timeout_policy.py` - TimeoutPolicy class, asyncio.wait_for wrapping, on_timeout strategies (fail/skip/compensate)
- [ ] Create `greenlang/orchestrator/dag_executor.py` - DAGExecutor: core engine, topological sort -> level grouping -> parallel execution per level, context propagation, conditional nodes, on_failure handling (fail_fast/continue/compensate), compensation rollback
- [ ] Create `greenlang/orchestrator/checkpoint_store.py` - DAGCheckpointStore ABC, MemoryCheckpointStore, FileCheckpointStore, checkpoint after each node, resume from any point, SHA-256 integrity verification
- [ ] Create `greenlang/orchestrator/provenance.py` - NodeProvenance tracking, chain hashing (SHA-256), execution trace building, export as JSON, parent hash linking
- [ ] Create `greenlang/orchestrator/determinism.py` - Deterministic scheduling (sorted node ordering), execution replay support, DeterministicClock integration, deterministic_uuid for execution IDs
- [ ] Create `greenlang/orchestrator/metrics.py` - 12 Prometheus metrics (counters, histograms, gauges) with prometheus_client graceful fallback
- [ ] Create `greenlang/orchestrator/api/__init__.py` - API package init
- [ ] Create `greenlang/orchestrator/api/router.py` - FastAPI router with 20 endpoints: CRUD for DAGs, execute, list executions, get trace, cancel, resume, provenance, checkpoints, metrics, import/export, visualize (Mermaid), health
- [ ] Create `greenlang/orchestrator/setup.py` - DAGOrchestrator facade class, configure_orchestrator(app), get_orchestrator(app), lifespan management

## Phase 2: Infrastructure (DevOps Engineer)

- [ ] Create `deployment/database/migrations/sql/V021__orchestrator_service.sql` - 5 tables (dag_workflows, dag_executions, node_traces, dag_checkpoints, execution_provenance) + hypertable + indexes + RLS + seed data
- [ ] Create `deployment/kubernetes/orchestrator-service/deployment.yaml` - 2 replicas, resource limits, probes, init containers
- [ ] Create `deployment/kubernetes/orchestrator-service/service.yaml` - ClusterIP port 8080
- [ ] Create `deployment/kubernetes/orchestrator-service/configmap.yaml` - Full configuration
- [ ] Create `deployment/kubernetes/orchestrator-service/hpa.yaml` - HPA min=2 max=6 + PDB
- [ ] Create `deployment/kubernetes/orchestrator-service/networkpolicy.yaml` - Default deny + specific rules
- [ ] Create `deployment/kubernetes/orchestrator-service/servicemonitor.yaml` - Prometheus scrape config
- [ ] Create `deployment/kubernetes/orchestrator-service/kustomization.yaml` - Kustomize base
- [ ] Create `deployment/monitoring/dashboards/orchestrator-service.json` - Grafana dashboard (20+ panels)
- [ ] Create `deployment/monitoring/alerts/orchestrator-alerts.yaml` - Alert rules (12+ alerts)
- [ ] Create `.github/workflows/orchestrator-ci.yml` - 7-job CI/CD pipeline (lint, type-check, unit-test, integration-test, coverage, build, deploy)

## Phase 3: Tests (Test Engineer)

- [ ] Create `tests/unit/orchestrator/conftest.py` - Shared fixtures: sample DAGs (linear, diamond, complex, invalid), mock agents, retry policies
- [ ] Create `tests/unit/orchestrator/test_config.py` - Config creation, env var overrides, defaults, validation
- [ ] Create `tests/unit/orchestrator/test_models.py` - Model serialization, YAML/JSON round-trip, validation
- [ ] Create `tests/unit/orchestrator/test_dag_builder.py` - Builder pattern, from_yaml, from_json, from_dict, add_node, add_edge
- [ ] Create `tests/unit/orchestrator/test_dag_validator.py` - Cycle detection (simple, complex, self-loop), unreachable nodes, missing deps, duplicates, valid DAGs
- [ ] Create `tests/unit/orchestrator/test_topological_sort.py` - Kahn's algorithm, deterministic ordering, level grouping, empty DAG, single node, diamond pattern, wide parallel
- [ ] Create `tests/unit/orchestrator/test_node_runner.py` - Sync/async agent execution, retry (succeed on Nth attempt), timeout (exceed + within), failure handling
- [ ] Create `tests/unit/orchestrator/test_retry_policy.py` - All strategies (exponential, linear, constant, fibonacci), jitter, delay calculation, should_retry logic
- [ ] Create `tests/unit/orchestrator/test_timeout_policy.py` - Timeout enforcement, on_timeout strategies, cancellation
- [ ] Create `tests/unit/orchestrator/test_dag_executor.py` - Linear DAG, diamond DAG, wide parallel, conditional nodes, fail_fast, continue mode, compensation
- [ ] Create `tests/unit/orchestrator/test_checkpoint_store.py` - Memory store: save, load, resume, list, delete, cleanup, integrity verification
- [ ] Create `tests/unit/orchestrator/test_provenance.py` - Hash calculation, chain linking, export JSON, parent hash validation
- [ ] Create `tests/unit/orchestrator/test_determinism.py` - Sorted scheduling, replay produces identical traces, deterministic IDs
- [ ] Create `tests/unit/orchestrator/test_metrics.py` - Counter increments, histogram observations, gauge values
- [ ] Create `tests/unit/orchestrator/test_setup.py` - Facade class, configure_orchestrator, get_orchestrator
- [ ] Create `tests/unit/orchestrator/test_api_router.py` - All 20 API endpoints, request/response validation, error handling
- [ ] Create `tests/integration/orchestrator/conftest.py` - Integration fixtures, mock agent registry
- [ ] Create `tests/integration/orchestrator/test_end_to_end.py` - Full DAG execution (simple, complex, diamond), result propagation
- [ ] Create `tests/integration/orchestrator/test_checkpoint_resume.py` - Execute, fail at node, resume, verify skipped nodes
- [ ] Create `tests/integration/orchestrator/test_concurrent_execution.py` - 10 concurrent DAG executions
- [ ] Create `tests/integration/orchestrator/test_deterministic_replay.py` - Execute twice with same inputs, verify identical traces
- [ ] Create `tests/integration/orchestrator/test_api_integration.py` - API with FastAPI TestClient
- [ ] Create `tests/integration/orchestrator/test_provenance_chain.py` - Full provenance chain verification
- [ ] Create `tests/load/orchestrator/test_dag_load.py` - 50 concurrent executions, 500-node DAG validation, checkpoint throughput

## Phase 4: Documentation (Tech Writer)

- [ ] Create `docs/runbooks/orchestrator-service-down.md` - Detection, diagnosis, recovery
- [ ] Create `docs/runbooks/dag-execution-stuck.md` - Identify stuck executions, checkpoint inspection, manual resume
- [ ] Create `docs/runbooks/checkpoint-corruption.md` - Integrity verification, cleanup, re-execution
- [ ] Create `docs/runbooks/high-execution-latency.md` - Diagnosis, concurrency tuning, resource scaling
