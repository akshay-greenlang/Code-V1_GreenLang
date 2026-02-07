# INFRA-010: Agent Factory v1.0 - Development Tasks

**Status:** IN PROGRESS
**Created:** 2026-02-05
**Priority:** P0 - CRITICAL

---

## Phase 1: Core Infrastructure (P0)

### 1.1 Package Structure
- [ ] Create `greenlang/infrastructure/agent_factory/__init__.py` - Public API exports
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/queue/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/sandbox/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/metering/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/config/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/dependencies/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/hub/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/api/__init__.py`

### 1.2 Lifecycle Manager
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/states.py` - Agent state machine (CREATED->VALIDATING->VALIDATED->DEPLOYING->WARMING_UP->RUNNING->DRAINING->RETIRED + FAILED/DEGRADED/FORCE_STOPPED)
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/manager.py` - Central lifecycle coordinator with distributed locking, event emission, concurrent operations
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/health.py` - Health check registry (liveness, readiness, startup probes), check scheduling, aggregated status
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/warmup.py` - Warm-up strategies (cache priming, model loading, connection pool), configurable timeout
- [ ] Create `greenlang/infrastructure/agent_factory/lifecycle/shutdown.py` - Graceful shutdown coordinator (drain timeout, in-flight completion, force stop)

### 1.3 Task Queue
- [ ] Create `greenlang/infrastructure/agent_factory/queue/task_queue.py` - Redis Streams distributed queue with consumer groups, at-least-once delivery, idempotency keys
- [ ] Create `greenlang/infrastructure/agent_factory/queue/priority.py` - Priority scheduler (CRITICAL=0, HIGH=1, NORMAL=2, LOW=3, BACKGROUND=4)
- [ ] Create `greenlang/infrastructure/agent_factory/queue/dead_letter.py` - DLQ with retry policies, configurable retention, reprocessing
- [ ] Create `greenlang/infrastructure/agent_factory/queue/workers.py` - Worker pool management (concurrency limits, affinity, graceful shutdown)
- [ ] Create `greenlang/infrastructure/agent_factory/queue/scheduler.py` - Cron-based and event-driven scheduling

### 1.4 Inter-Agent Messaging
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/protocol.py` - MessageEnvelope, MessageType, typed payloads, schema validation
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/router.py` - Point-to-point, pub/sub, broadcast routing
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/serialization.py` - JSON (default) and MessagePack serialization
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/acknowledgment.py` - Ack/nack, at-least-once delivery guarantees, timeout
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/channels.py` - Named channels, topic-based routing, subscription management

### 1.5 Database
- [ ] Create `deployment/database/migrations/sql/V008__agent_factory.sql` - 8 tables, 2 hypertables, 1 continuous aggregate, RLS policies, triggers

---

## Phase 2: Resilience & Runtime (P0)

### 2.1 Circuit Breaker & Resilience
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/circuit_breaker.py` - Per-agent circuit breaker (closed->open->half_open), configurable thresholds, Prometheus metrics
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/fallback.py` - Fallback chain manager (primary->secondary->default), chain configuration
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/bulkhead.py` - Bulkhead isolation (semaphore-based), max concurrent per agent
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/retry.py` - Retry policies (exponential backoff, jitter, max attempts, retryable exceptions)
- [ ] Create `greenlang/infrastructure/agent_factory/resilience/timeout.py` - Execution timeout with cancellation, SIGTERM->SIGKILL escalation

### 2.2 Sandbox
- [ ] Create `greenlang/infrastructure/agent_factory/sandbox/executor.py` - Process-level isolation, subprocess management, clean state per invocation
- [ ] Create `greenlang/infrastructure/agent_factory/sandbox/resource_limits.py` - CPU/memory/disk/network limits via resource module and psutil
- [ ] Create `greenlang/infrastructure/agent_factory/sandbox/timeout_guard.py` - Execution timeout with forced termination
- [ ] Create `greenlang/infrastructure/agent_factory/sandbox/audit.py` - Sandbox audit trail (resource usage, syscalls, network)

### 2.3 Cost Metering
- [ ] Create `greenlang/infrastructure/agent_factory/metering/cost_tracker.py` - Per-agent cost tracking (compute, tokens, storage, network)
- [ ] Create `greenlang/infrastructure/agent_factory/metering/budget.py` - Budget allocation, threshold alerts (80%/100%), hard/soft limits
- [ ] Create `greenlang/infrastructure/agent_factory/metering/resource_quotas.py` - Max concurrent/daily execution enforcement
- [ ] Create `greenlang/infrastructure/agent_factory/metering/billing_events.py` - Structured billing events for downstream accounting

### 2.4 Config Hot-Reload
- [ ] Create `greenlang/infrastructure/agent_factory/config/hot_reload.py` - Redis pub/sub watcher, change detection, reload callbacks
- [ ] Create `greenlang/infrastructure/agent_factory/config/schema.py` - Agent config schema validation (Pydantic models)
- [ ] Create `greenlang/infrastructure/agent_factory/config/store.py` - Config store (Redis cache + PostgreSQL persistence)
- [ ] Create `greenlang/infrastructure/agent_factory/config/diff.py` - Config diff computation and rollback support

---

## Phase 3: Packaging & Versioning (P1)

### 3.1 Agent Packaging
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/pack_format.py` - agent.pack.yaml spec parser and validator
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/builder.py` - Package builder (collect, validate, archive as .glpack)
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/resolver.py` - Dependency resolver (semver ranges, conflict detection, diamond resolution)
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/installer.py` - Package installer (unpack, validate checksums, register)
- [ ] Create `greenlang/infrastructure/agent_factory/packaging/manifest.py` - Manifest generator (SHA-256 checksums, signatures)

### 3.2 Versioning & Deployment
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/semver.py` - Semantic version parsing, comparison, range matching
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/compatibility.py` - Version compatibility matrix (agent-to-agent, agent-to-platform)
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/migration.py` - Version migration framework (up/down scripts)
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/canary.py` - Canary deployment controller (traffic split, metrics eval, auto-promote/rollback)
- [ ] Create `greenlang/infrastructure/agent_factory/versioning/rollback.py` - Automated rollback (error rate, latency, health check triggers)

### 3.3 Dependency Graph
- [ ] Create `greenlang/infrastructure/agent_factory/dependencies/graph.py` - Agent dependency DAG builder with adjacency list
- [ ] Create `greenlang/infrastructure/agent_factory/dependencies/resolver.py` - Topological sort, execution ordering, parallel group detection
- [ ] Create `greenlang/infrastructure/agent_factory/dependencies/cycle_detector.py` - Circular dependency detection with diagnostic path
- [ ] Create `greenlang/infrastructure/agent_factory/dependencies/visualizer.py` - DOT format graph export for Graphviz

### 3.4 Agent Hub
- [ ] Create `greenlang/infrastructure/agent_factory/hub/registry.py` - Hub registry (publish, search, download, list versions)
- [ ] Create `greenlang/infrastructure/agent_factory/hub/client.py` - Hub API client (REST, auth, retry)
- [ ] Create `greenlang/infrastructure/agent_factory/hub/index.py` - Local agent index and package cache
- [ ] Create `greenlang/infrastructure/agent_factory/hub/validator.py` - Package validation before publish

---

## Phase 4: Developer Experience (P1)

### 4.1 CLI Commands
- [ ] Create `greenlang/cli/commands/agent/__init__.py` - Agent CLI command group
- [ ] Create `greenlang/cli/commands/agent/create.py` - Scaffold agent from template
- [ ] Create `greenlang/cli/commands/agent/test.py` - Run agent tests (unit/integration/e2e/coverage)
- [ ] Create `greenlang/cli/commands/agent/deploy.py` - Deploy agent (canary/blue-green/rolling)
- [ ] Create `greenlang/cli/commands/agent/rollback.py` - Rollback agent version
- [ ] Create `greenlang/cli/commands/agent/status.py` - Show agent status and health
- [ ] Create `greenlang/cli/commands/agent/logs.py` - Stream agent logs (tail/follow)
- [ ] Create `greenlang/cli/commands/agent/inspect.py` - Inspect agent (deps/config/metrics)
- [ ] Create `greenlang/cli/commands/agent/pack.py` - Build distributable package
- [ ] Create `greenlang/cli/commands/agent/publish.py` - Publish to Agent Hub

### 4.2 Telemetry
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/tracer.py` - OpenTelemetry tracer setup and configuration
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/spans.py` - Agent-specific span types and attributes
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/correlation.py` - Correlation ID propagation across chains
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/exporter.py` - OTLP/Jaeger/Zipkin export configuration
- [ ] Create `greenlang/infrastructure/agent_factory/telemetry/metrics_collector.py` - Per-agent metrics aggregation

### 4.3 Factory API
- [ ] Create `greenlang/infrastructure/agent_factory/api/factory_routes.py` - Factory CRUD and management endpoints
- [ ] Create `greenlang/infrastructure/agent_factory/api/lifecycle_routes.py` - Lifecycle management endpoints
- [ ] Create `greenlang/infrastructure/agent_factory/api/queue_routes.py` - Task queue management endpoints
- [ ] Create `greenlang/infrastructure/agent_factory/api/hub_routes.py` - Hub registry endpoints
- [ ] Create `greenlang/infrastructure/agent_factory/api/middleware.py` - Factory-specific middleware

---

## Phase 5: Deployment & Monitoring (P2)

### 5.1 Kubernetes Manifests
- [ ] Create `deployment/kubernetes/agent-factory/namespace.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/deployment-lifecycle-mgr.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/deployment-queue-worker.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/deployment-hub.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/service.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/hpa.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/configmap.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/networkpolicy.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/rbac.yaml`
- [ ] Create `deployment/kubernetes/agent-factory/kustomization.yaml`

### 5.2 Helm Charts
- [ ] Create `deployment/helm/greenlang-agents/templates/deployment-lifecycle-mgr.yaml`
- [ ] Create `deployment/helm/greenlang-agents/templates/deployment-queue-worker.yaml`
- [ ] Create `deployment/helm/greenlang-agents/templates/deployment-hub.yaml`

### 5.3 Monitoring
- [ ] Create `deployment/monitoring/dashboards/agent-factory-v1.json` - 20+ panel Grafana dashboard
- [ ] Create `deployment/monitoring/alerts/agent-factory-alerts.yaml` - 15+ Prometheus alert rules

---

## Phase 6: Testing (P2)

### 6.1 Unit Tests
- [ ] Create `tests/unit/agent_factory/__init__.py`
- [ ] Create `tests/unit/agent_factory/test_lifecycle_manager.py` - 20+ tests
- [ ] Create `tests/unit/agent_factory/test_task_queue.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_messaging_protocol.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_circuit_breaker.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_packaging.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_versioning.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_sandbox.py` - 10+ tests
- [ ] Create `tests/unit/agent_factory/test_cost_metering.py` - 10+ tests
- [ ] Create `tests/unit/agent_factory/test_hot_reload.py` - 10+ tests
- [ ] Create `tests/unit/agent_factory/test_dependency_graph.py` - 15+ tests
- [ ] Create `tests/unit/agent_factory/test_telemetry.py` - 10+ tests
- [ ] Create `tests/unit/agent_factory/test_hub_registry.py` - 10+ tests

### 6.2 Integration Tests
- [ ] Create `tests/integration/agent_factory/__init__.py`
- [ ] Create `tests/integration/agent_factory/test_lifecycle_integration.py` - 10+ tests
- [ ] Create `tests/integration/agent_factory/test_queue_integration.py` - 10+ tests
- [ ] Create `tests/integration/agent_factory/test_messaging_integration.py` - 10+ tests
- [ ] Create `tests/integration/agent_factory/test_factory_e2e.py` - 10+ tests

### 6.3 Load Tests
- [ ] Create `tests/load/agent_factory/__init__.py`
- [ ] Create `tests/load/agent_factory/test_queue_throughput.py` - Queue throughput benchmarks

---

## Summary

| Phase | Files | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Core Infrastructure | 22 | P0 | COMPLETE |
| Phase 2: Resilience & Runtime | 17 | P0 | COMPLETE |
| Phase 3: Packaging & Versioning | 18 | P1 | COMPLETE |
| Phase 4: Developer Experience | 20 | P1 | COMPLETE |
| Phase 5: Deployment & Monitoring | 15 | P2 | COMPLETE |
| Phase 6: Testing | 18 | P2 | COMPLETE |
| **v1.0 TOTAL** | **110** | - | **COMPLETE** |
| Phase 7: CTO v1.1 Improvements | 20 | P0 | IN PROGRESS |
| **GRAND TOTAL** | **130** | - | **IN PROGRESS** |

---

## Phase 7: CTO v1.1 Improvements (P0)

### 7.1 PRD Updates (CTO Review)
- [x] Add Execution Model section (hybrid topology: Pool vs Dedicated mode)
- [x] Rewrite TR-003 with dual-transport messaging (Redis Streams + Pub/Sub)
- [x] Align TR-004 circuit breaker with rate-based DB schema
- [x] Add agent_tenant_config table (9th table) for multi-tenant overrides
- [x] Add Operations API pattern (202 Accepted for long-running ops)
- [x] Trim TR-008 sandbox to achievable v1 scope
- [x] Add Backward Compatibility & Migration section for 119 agents
- [x] Add Open Decisions section for engineering

### 7.2 Messaging Module (NEW - was missing entirely)
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/protocol.py` - MessageEnvelope, ChannelType, MessageType
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/router.py` - MessageRouter, DurableTransport, EphemeralTransport
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/serialization.py` - JSON + MessagePack serialization
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/acknowledgment.py` - AcknowledgmentTracker, DLQ management
- [ ] Create `greenlang/infrastructure/agent_factory/messaging/channels.py` - ChannelManager, Channel, ChannelConfig

### 7.3 DB Migration Updates
- [ ] Update V008 circuit_breaker table: rate-based config JSONB fields
- [ ] Add agent_tenant_config table to V008
- [ ] Add agent_operations table to V008 (async 202 pattern)

### 7.4 Operations API
- [ ] Create `greenlang/infrastructure/agent_factory/api/operations_routes.py` - 202 Accepted pattern
- [ ] Update factory_routes.py deploy/rollback to delegate to operations API

### 7.5 Legacy Agent Migration
- [ ] Create `greenlang/infrastructure/agent_factory/legacy/__init__.py`
- [ ] Create `greenlang/infrastructure/agent_factory/legacy/discovery.py` - LegacyAgentDiscovery
- [ ] Create `greenlang/infrastructure/agent_factory/legacy/registrar.py` - LegacyRegistrar
- [ ] Create `greenlang/infrastructure/agent_factory/legacy/pack_generator.py` - LegacyPackGenerator

### 7.6 Sandbox Trimming
- [ ] Update sandbox/executor.py - process-level for pool mode
- [ ] Update sandbox/audit.py - simplified audit trail (no syscall logging)
- [ ] Update sandbox/resource_limits.py - setrlimit on Linux, best-effort otherwise

### 7.7 Tests for v1.1 Changes
- [ ] Create `tests/unit/agent_factory/test_messaging_protocol.py` - dual-transport tests
- [ ] Create `tests/unit/agent_factory/test_operations_api.py` - 202 Accepted pattern tests
- [ ] Create `tests/unit/agent_factory/test_legacy_migration.py` - discovery + registration tests
- [ ] Create `tests/unit/agent_factory/test_tenant_config.py` - multi-tenant override tests
