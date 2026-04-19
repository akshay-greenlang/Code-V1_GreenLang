# PRD: INFRA-010 - Agent Factory v1.0

**Document Version:** 1.1
**Date:** February 5, 2026
**Status:** READY FOR EXECUTION (CTO-reviewed, ambiguities resolved)
**Priority:** P0 - CRITICAL
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-010

---

## Executive Summary

Deploy a production-ready Agent Factory v1.0 for GreenLang Climate OS that provides a complete agent lifecycle platform: creation, testing, packaging, deployment, versioning, orchestration, monitoring, and retirement across all 47+ agent types and 402 planned agents. The Agent Factory consolidates and extends the existing LLM-powered code generation pipeline (`greenlang/utilities/factory/`, 5 files), AgentSpec v2 framework (`greenlang/agents/agentspec_v2_base.py`), agent registry (`greenlang/cli/agent_registry.py`), and agent API (`greenlang/execution/infrastructure/api/routes/agents_routes.py`) into a unified, production-grade platform with centralized lifecycle management, distributed task queues, inter-agent communication protocol, circuit breakers, agent packaging, developer CLI, cost metering, and canary deployment support.

### Current State
- LLM-powered agent code generation: `greenlang/utilities/factory/agent_factory.py` (multi-step pipeline: tools -> agent -> tests -> docs -> demo)
- Factory supporting files: `prompts.py`, `templates.py`, `validators.py`, Python/TypeScript SDKs
- AgentSpec v2 base: `greenlang/agents/agentspec_v2_base.py` (Generic Agent[InT, OutT] with lifecycle hooks)
- 3-path base classes: `greenlang/agents/base_agents.py` (DeterministicAgent, ReasoningAgent, InsightAgent)
- Agent registry: `greenlang/cli/agent_registry.py` (core, entry-point, filesystem, YAML discovery)
- Agent registry info: `greenlang/agents/registry.py` (AgentInfo, AgentStatus, ExecutionMode)
- Agent REST API: `greenlang/execution/infrastructure/api/routes/agents_routes.py` (4 endpoints)
- Agent service: `greenlang/execution/infrastructure/api/services/agent_service.py` (singleton lifecycle)
- Agent state store: `greenlang/execution/infrastructure/api/storage/agent_state_store.py`
- Agent templates: `greenlang/execution/infrastructure/agent_templates.py`
- Orchestration: `greenlang/core/orchestrator.py`, `greenlang/execution/core/workflow.py`, `greenlang/execution/core/composability.py`
- Message bus: `greenlang/execution/core/message_bus.py` (basic inter-agent messaging)
- Agent RBAC: `greenlang/governance/policy/agent_rbac.py`
- K8s manifests: `deployment/kubernetes/agents/` (6 manifests, 4 agent deployments)
- Helm chart: `deployment/helm/greenlang-agents/` (Chart.yaml, values.yaml, 4 deployment templates)
- Docker compose: `deployment/docker-compose.agents.yml`
- Monitoring: `deployment/kubernetes/monitoring/dashboards/agent-factory-overview.json`, `agent-health.json`
- Tests: 80+ test files across `tests/agents/`, `tests/unit/agents/`, `tests/integration/`, `tests/factory/`
- 119 of 402 target agents built (30% complete)
- No centralized lifecycle manager (startup/shutdown/health coordination)
- No agent circuit breakers or fallback chains
- No inter-agent communication protocol (standardized message envelopes)
- No distributed task queue for agent execution
- No agent packaging format or hub
- No developer CLI (`gl agent create/test/deploy/rollback`)
- No per-agent cost metering or resource quotas
- No agent versioning enforcement or canary deployment
- No agent sandbox/isolation runtime layer
- No agent dependency graph resolution
- No hot-reload for agent configuration
- No agent telemetry with distributed tracing correlation IDs

### Target State
- Centralized Agent Lifecycle Manager: init -> warm-up -> health-check -> execute -> graceful-shutdown -> retire
- Distributed task queue with priority scheduling and dead-letter queue
- Standardized inter-agent communication protocol with typed message envelopes
- Per-agent circuit breakers with configurable thresholds and fallback chains
- Agent packaging format (`agent.pack.yaml`) with dependency resolution
- Developer CLI: `gl agent create`, `gl agent test`, `gl agent deploy`, `gl agent rollback`, `gl agent status`
- Agent versioning with semantic version enforcement and canary deployment support
- Agent sandbox/isolation via process isolation and resource cgroup limits
- Per-agent cost metering (compute, LLM tokens, storage, network) with budget alerts
- Hot-reload for agent configuration without service restart via Redis pub/sub
- Agent dependency graph with topological ordering and circular dependency detection
- Distributed tracing with OpenTelemetry correlation IDs through agent execution chains
- Agent Hub registry for discovery, sharing, and version management
- Enhanced monitoring: 20+ panel dashboard, 15+ alert rules, SLA tracking per agent
- 85%+ test coverage across all new modules

---

## Scope

### In Scope
1. Agent Lifecycle Manager: `greenlang/infrastructure/agent_factory/lifecycle/` (Python package)
2. Agent Task Queue: `greenlang/infrastructure/agent_factory/queue/` (async distributed queue with Redis + PostgreSQL)
3. Inter-Agent Communication Protocol: `greenlang/infrastructure/agent_factory/messaging/` (typed envelopes, routing, serialization)
4. Agent Circuit Breaker: `greenlang/infrastructure/agent_factory/resilience/` (per-agent breakers, fallback chains, bulkhead isolation)
5. Agent Packaging: `greenlang/infrastructure/agent_factory/packaging/` (pack format, dependency resolver, builder)
6. Agent Developer CLI: `greenlang/cli/commands/agent/` (create, test, deploy, rollback, status, logs, inspect)
7. Agent Versioning: `greenlang/infrastructure/agent_factory/versioning/` (semver, compatibility matrix, migration)
8. Agent Sandbox: `greenlang/infrastructure/agent_factory/sandbox/` (process isolation, resource limits, timeout enforcement)
9. Agent Cost Metering: `greenlang/infrastructure/agent_factory/metering/` (cost tracker, budget alerts, resource quotas)
10. Agent Config Hot-Reload: `greenlang/infrastructure/agent_factory/config/` (Redis pub/sub watcher, schema validation, rollback)
11. Agent Dependency Graph: `greenlang/infrastructure/agent_factory/dependencies/` (DAG builder, topological sort, cycle detection)
12. Agent Telemetry: `greenlang/infrastructure/agent_factory/telemetry/` (OpenTelemetry spans, correlation IDs, trace export)
13. Agent Hub Registry: `greenlang/infrastructure/agent_factory/hub/` (registry client, publish, search, download)
14. Enhanced Agent API: extend `greenlang/execution/infrastructure/api/routes/agents_routes.py` (15+ new endpoints)
15. PostgreSQL migration V008: `deployment/database/migrations/sql/V008__agent_factory.sql` (9 tables)
16. Helm chart updates: `deployment/helm/greenlang-agents/` (new templates for factory services)
17. K8s manifests: `deployment/kubernetes/agent-factory/` (factory controller, queue workers, hub)
18. Monitoring dashboard: `deployment/monitoring/dashboards/agent-factory-v1.json` (20+ panels)
19. Alert rules: `deployment/monitoring/alerts/agent-factory-alerts.yaml` (15+ rules)
20. Comprehensive test suite: `tests/unit/agent_factory/`, `tests/integration/agent_factory/` (200+ tests, 85%+ coverage)

### Out of Scope
- Visual agent pipeline builder UI (future phase - API-first approach)
- Multi-region agent deployment synchronization (leverages existing DR replication)
- Agent marketplace with billing/payments (internal Hub only in v1)
- Machine learning-based agent auto-scaling (rule-based in v1)
- Natural language agent creation (existing LLM factory handles this separately)
- Agent-to-agent negotiation protocols (simple request/response in v1)
- Custom agent runtime environments (Python 3.11+ only in v1)

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Agent Factory v1.0                                │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ Developer CLI   │  │ REST API       │  │ Agent Hub Registry         │ │
│  │ (gl agent ...)  │  │ /api/v1/factory│  │ (publish, search, install) │ │
│  └───────┬────────┘  └───────┬────────┘  └────────────┬───────────────┘ │
│          │                   │                        │                  │
│          ▼                   ▼                        ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Agent Lifecycle Manager                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │   │
│  │  │ Create   │→│ Validate │→│ Deploy   │→│ Monitor  │→│ Retire │ │   │
│  │  │ (Scaffold│ │ (Test,   │ │ (Canary, │ │ (Health, │ │ (Drain,│ │   │
│  │  │  +Build) │ │  Lint,   │ │  Blue-   │ │  Metrics,│ │  Archive│ │   │
│  │  │          │ │  Pack)   │ │  Green)  │ │  Alerts) │ │  )     │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│          │                   │                        │                  │
│          ▼                   ▼                        ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Execution Layer                                 │   │
│  │                                                                   │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │   │
│  │  │ Task Queue       │  │ Message Bus      │  │ Dependency Graph │  │   │
│  │  │ (Priority,       │  │ (Typed Envelopes,│  │ (DAG, Topological│  │   │
│  │  │  DLQ, Retry)     │  │  Routing, Ack)   │  │  Sort, Cycles)   │  │   │
│  │  └────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘  │   │
│  │           │                    │                      │            │   │
│  │           ▼                    ▼                      ▼            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                Agent Runtime                                │  │   │
│  │  │  ┌───────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐ │  │   │
│  │  │  │ Sandbox   │ │ Circuit    │ │ Hot-Reload│ │ Cost      │ │  │   │
│  │  │  │ (Isolation│ │ Breaker    │ │ (Config   │ │ Metering  │ │  │   │
│  │  │  │  +Limits) │ │ (Fallback) │ │  Watcher) │ │ (Budget)  │ │  │   │
│  │  │  └───────────┘ └────────────┘ └───────────┘ └───────────┘ │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│          │                   │                        │                  │
│          ▼                   ▼                        ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Observability Layer                             │   │
│  │  ┌───────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │ Telemetry │ │ Versioning │ │ SLA Tracker  │ │ Audit Log    │ │   │
│  │  │ (OTel,    │ │ (Semver,   │ │ (P99, Avail, │ │ (Execution   │ │   │
│  │  │  Traces)  │ │  Canary)   │ │  Error Rate) │ │  History)    │ │   │
│  │  └───────────┘ └────────────┘ └──────────────┘ └──────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌─────────────────┐ ┌────────────────┐ ┌──────────────────────────────┐
│ PostgreSQL      │ │ Redis          │ │ S3 (Agent Packages,          │
│ (Agent Registry,│ │ (Task Queue,   │ │  Artifacts, Telemetry)       │
│  Audit, Metrics)│ │  Config Cache, │ │                              │
│                 │ │  Pub/Sub)      │ │                              │
└─────────────────┘ └────────────────┘ └──────────────────────────────┘
```

### Component Architecture

```
greenlang/infrastructure/agent_factory/
├── __init__.py                          # Public API exports
├── lifecycle/
│   ├── __init__.py
│   ├── manager.py                       # AgentLifecycleManager - central coordinator
│   ├── states.py                        # Agent state machine (CREATED→VALIDATING→DEPLOYING→RUNNING→DRAINING→RETIRED)
│   ├── health.py                        # Health check registry (liveness, readiness, startup probes)
│   ├── warmup.py                        # Agent warm-up strategies (preload models, cache priming)
│   └── shutdown.py                      # Graceful shutdown coordinator (drain, timeout, force)
├── queue/
│   ├── __init__.py
│   ├── task_queue.py                    # Distributed task queue (Redis Streams + PostgreSQL)
│   ├── priority.py                      # Priority scheduler (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
│   ├── dead_letter.py                   # Dead-letter queue with retry policies
│   ├── workers.py                       # Worker pool management (concurrency, affinity)
│   └── scheduler.py                     # Cron-based and event-driven agent scheduling
├── messaging/
│   ├── __init__.py
│   ├── protocol.py                      # Inter-agent communication protocol (MessageEnvelope)
│   ├── router.py                        # Message routing (point-to-point, pub/sub, broadcast)
│   ├── serialization.py                 # Message serialization (JSON, MessagePack, Protobuf)
│   ├── acknowledgment.py               # Message acknowledgment and delivery guarantees
│   └── channels.py                      # Named channels with topic-based routing
├── resilience/
│   ├── __init__.py
│   ├── circuit_breaker.py               # Per-agent circuit breaker (closed→open→half-open)
│   ├── fallback.py                      # Fallback chain manager (primary→secondary→default)
│   ├── bulkhead.py                      # Bulkhead isolation (thread/process/semaphore)
│   ├── retry.py                         # Retry policies (exponential backoff, jitter, max attempts)
│   └── timeout.py                       # Execution timeout enforcement with cancellation
├── packaging/
│   ├── __init__.py
│   ├── pack_format.py                   # agent.pack.yaml specification and parser
│   ├── builder.py                       # Agent package builder (collect, validate, archive)
│   ├── resolver.py                      # Dependency resolver (version ranges, conflict detection)
│   ├── installer.py                     # Agent package installer (unpack, validate, register)
│   └── manifest.py                      # Package manifest generator (checksums, signatures)
├── versioning/
│   ├── __init__.py
│   ├── semver.py                        # Semantic version enforcement and comparison
│   ├── compatibility.py                 # Version compatibility matrix (agent-to-agent, agent-to-platform)
│   ├── migration.py                     # Version migration framework (up/down scripts)
│   ├── canary.py                        # Canary deployment controller (traffic splitting, metrics)
│   └── rollback.py                      # Automated rollback (error rate threshold, latency spike)
├── sandbox/
│   ├── __init__.py
│   ├── executor.py                      # Sandboxed agent executor (process isolation)
│   ├── resource_limits.py               # Resource limit enforcement (CPU, memory, disk, network)
│   ├── timeout_guard.py                 # Execution timeout with forced termination
│   └── audit.py                         # Sandbox execution audit trail
├── metering/
│   ├── __init__.py
│   ├── cost_tracker.py                  # Per-agent cost tracking (compute, tokens, storage)
│   ├── budget.py                        # Budget allocation and alerts (per-agent, per-tenant)
│   ├── resource_quotas.py               # Resource quota enforcement and reporting
│   └── billing_events.py               # Billing event emission for downstream accounting
├── config/
│   ├── __init__.py
│   ├── hot_reload.py                    # Config hot-reload via Redis pub/sub
│   ├── schema.py                        # Agent configuration schema validation
│   ├── store.py                         # Configuration store (Redis + PostgreSQL)
│   └── diff.py                          # Configuration diff and rollback
├── dependencies/
│   ├── __init__.py
│   ├── graph.py                         # Agent dependency DAG builder
│   ├── resolver.py                      # Topological sort and execution ordering
│   ├── cycle_detector.py                # Circular dependency detection with diagnostic
│   └── visualizer.py                    # Dependency graph visualization (DOT format)
├── telemetry/
│   ├── __init__.py
│   ├── tracer.py                        # OpenTelemetry tracer for agent execution
│   ├── spans.py                         # Agent-specific span types and attributes
│   ├── correlation.py                   # Correlation ID propagation across agent chains
│   ├── exporter.py                      # Telemetry export (OTLP, Jaeger, Zipkin)
│   └── metrics_collector.py             # Per-agent metrics aggregation
├── hub/
│   ├── __init__.py
│   ├── registry.py                      # Agent Hub registry (publish, search, download)
│   ├── client.py                        # Hub API client
│   ├── index.py                         # Local agent index and cache
│   └── validator.py                     # Package validation before publish
└── api/
    ├── __init__.py
    ├── factory_routes.py                # Factory REST API endpoints
    ├── lifecycle_routes.py              # Lifecycle management endpoints
    ├── queue_routes.py                  # Task queue management endpoints
    ├── hub_routes.py                    # Hub registry endpoints
    └── middleware.py                    # Factory-specific middleware

greenlang/cli/commands/agent/
├── __init__.py
├── create.py                            # gl agent create [--template] [--spec]
├── test.py                              # gl agent test [--unit] [--integration] [--e2e]
├── deploy.py                            # gl agent deploy [--env] [--canary] [--blue-green]
├── rollback.py                          # gl agent rollback [--version] [--immediate]
├── status.py                            # gl agent status [--all] [--agent-id]
├── logs.py                              # gl agent logs [--agent-id] [--tail] [--follow]
├── inspect.py                           # gl agent inspect [--agent-id] [--deps] [--config]
├── pack.py                              # gl agent pack [--output] [--sign]
└── publish.py                           # gl agent publish [--hub] [--tag]

deployment/
├── database/
│   └── migrations/
│       └── sql/
│           └── V008__agent_factory.sql  # Agent factory database schema
├── kubernetes/
│   └── agent-factory/
│       ├── namespace.yaml               # greenlang-agent-factory namespace
│       ├── deployment-lifecycle-mgr.yaml # Lifecycle manager deployment
│       ├── deployment-queue-worker.yaml  # Task queue worker deployment
│       ├── deployment-hub.yaml           # Agent Hub registry deployment
│       ├── service.yaml                  # Factory services (ClusterIP)
│       ├── hpa.yaml                      # HPA for queue workers
│       ├── configmap.yaml               # Factory configuration
│       ├── networkpolicy.yaml           # Network isolation policies
│       ├── rbac.yaml                     # Factory RBAC (ServiceAccount, Role, RoleBinding)
│       └── kustomization.yaml           # Kustomize base
├── helm/
│   └── greenlang-agents/
│       └── templates/
│           ├── deployment-lifecycle-mgr.yaml
│           ├── deployment-queue-worker.yaml
│           └── deployment-hub.yaml
└── monitoring/
    ├── dashboards/
    │   └── agent-factory-v1.json        # 20+ panel Grafana dashboard
    └── alerts/
        └── agent-factory-alerts.yaml    # 15+ Prometheus alert rules

tests/
├── unit/
│   └── agent_factory/
│       ├── test_lifecycle_manager.py
│       ├── test_task_queue.py
│       ├── test_messaging_protocol.py
│       ├── test_circuit_breaker.py
│       ├── test_packaging.py
│       ├── test_versioning.py
│       ├── test_sandbox.py
│       ├── test_cost_metering.py
│       ├── test_hot_reload.py
│       ├── test_dependency_graph.py
│       ├── test_telemetry.py
│       └── test_hub_registry.py
├── integration/
│   └── agent_factory/
│       ├── test_lifecycle_integration.py
│       ├── test_queue_integration.py
│       ├── test_messaging_integration.py
│       └── test_factory_e2e.py
└── load/
    └── agent_factory/
        └── test_queue_throughput.py
```

### Execution Model (DECISION: Hybrid Topology)

The Agent Factory uses a **hybrid execution model** that balances cost efficiency with isolation guarantees:

```
┌─────────────────────────────────────────────────────────────┐
│                   Request Entry Points                       │
│   REST API (/api/v1/factory)  │  CLI (gl agent ...)         │
│   Kong Gateway                │  Cron Scheduler             │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Version Selection & Routing                     │
│                                                             │
│  Lifecycle Manager resolves: agent_key + tenant → version   │
│  Canary controller applies: traffic_pct → old_ver/new_ver   │
│  Feature flag gate: skip if agent disabled for tenant       │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
     ┌─────────┴─────────┐     ┌─────────┴──────────┐
     │  POOL MODE (default)│     │  DEDICATED MODE     │
     │  Long-tail agents   │     │  Hot / high-risk    │
     │                     │     │  agents              │
     │  Shared worker pool │     │  Own K8s Deployment  │
     │  loads agent.pack   │     │  + Service + HPA     │
     │  as Python module   │     │  Istio traffic split │
     │                     │     │  for canary           │
     │  Process sandbox    │     │  Pod-level isolation  │
     │  for untrusted      │     │                      │
     └─────────────────────┘     └──────────────────────┘
```

**Execution Modes:**

| Mode | When | Isolation | Deploy Means | Canary Means |
|---|---|---|---|---|
| **Pool** (default) | Agent < 100 exec/day, no special resource needs | Process sandbox in shared worker | Register .glpack in Hub, workers load on demand | Version-selection in scheduler (weighted random) |
| **Dedicated** | Agent > 100 exec/day, custom resource needs, or `dedicated: true` in pack.yaml | Own Pod with K8s resource limits + NetworkPolicy | Roll out new K8s Deployment via ArgoCD | Istio VirtualService traffic splitting |

**How requests flow:**
1. API/CLI receives execution request with `agent_key` and optional `tenant_id`
2. Lifecycle Manager looks up agent in registry → resolves active version (respecting canary weights and tenant version pins)
3. If Pool mode: Task Queue dispatches to shared worker pool → worker loads agent module from .glpack → executes in process sandbox
4. If Dedicated mode: Request routed to agent's K8s Service → Istio handles canary traffic split → Pod executes directly
5. Both modes: telemetry spans created, cost metered, circuit breaker checked, results stored

**Promotion criteria** (Pool → Dedicated):
- Agent execution count > 100/day sustained for 7 days
- Agent requires GPU or > 2Gi memory
- Agent has `dedicated: true` in agent.pack.yaml
- Tenant requests dedicated isolation (compliance requirement)

### Data Flow

```
1. Agent Creation Request arrives (CLI: gl agent create, or REST API)
   ├── Lifecycle Manager validates the request
   ├── AgentSpec v2 schema validation
   ├── Dependency graph analysis (detect cycles, resolve ordering)
   └── Template selection (DeterministicAgent / ReasoningAgent / InsightAgent)

2. Agent Build & Validation
   ├── Code generation (if from spec) via existing LLM factory
   ├── Package builder creates agent.pack.yaml
   ├── Validator runs lint, type-check, unit tests
   ├── Sandbox executor runs isolated integration tests
   └── Package manifest with checksums generated

3. Agent Deployment
   ├── Version registered in Agent Hub
   ├── Canary deployment: 5% → 25% → 50% → 100% traffic
   ├── Health checks validate startup/readiness/liveness
   ├── Circuit breaker initialized with thresholds
   ├── Cost metering starts tracking
   └── Config hot-reload watcher registered

4. Agent Execution
   ├── Task Queue receives execution request (priority-sorted)
   ├── Worker pool dispatches to appropriate agent
   ├── Sandbox enforces resource limits (CPU, memory, timeout)
   ├── Inter-agent messages routed via Message Bus
   ├── Telemetry spans created with correlation IDs
   └── Results stored with full provenance chain

5. Agent Monitoring & Retirement
   ├── SLA tracker monitors P99 latency, error rate, availability
   ├── Cost metering generates budget alerts when threshold exceeded
   ├── Stale agent detection (no executions in 30 days)
   ├── Graceful drain: stop accepting new tasks, complete in-flight
   └── Archive: move to RETIRED state, preserve audit trail
```

---

## Technical Requirements

### TR-001: Agent Lifecycle Manager

The lifecycle manager is the central coordinator for all agent operations.

**State Machine:**
```
CREATED → VALIDATING → VALIDATED → DEPLOYING → WARMING_UP → RUNNING → DRAINING → RETIRED
                ↓                                    ↓            ↓
             FAILED                               DEGRADED    FORCE_STOPPED
```

**Requirements:**
1. Centralized lifecycle coordinator managing all agent state transitions
2. Health check registry supporting liveness, readiness, and startup probes
3. Warm-up strategy execution before accepting traffic (cache priming, model loading)
4. Graceful shutdown with configurable drain timeout (default: 30s, max: 300s)
5. Event emission on every state transition (for audit trail and monitoring)
6. Concurrent lifecycle operations with distributed locking (Redis)
7. Automatic restart of crashed agents with backoff (max 5 attempts)

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Health Check Interval | 30s | 15s | 10s |
| Drain Timeout | 10s | 30s | 60s |
| Max Restart Attempts | 3 | 5 | 5 |
| Warm-up Timeout | 30s | 60s | 120s |
| State Transition Lock TTL | 10s | 10s | 10s |

### TR-002: Distributed Task Queue

Redis Streams-backed task queue for agent execution requests.

**Requirements:**
1. Priority-based scheduling: CRITICAL (0) > HIGH (1) > NORMAL (2) > LOW (3) > BACKGROUND (4)
2. Consumer group-based worker pool with at-least-once delivery
3. Dead-letter queue for failed tasks after max retries (default: 3)
4. Task deduplication via idempotency keys (24h window)
5. Task TTL with automatic expiration (default: 1h, max: 24h)
6. Visibility timeout for in-flight tasks (default: 300s)
7. Cron-based recurring task scheduling (cron expression syntax)
8. Event-driven task triggering via message bus subscriptions
9. Task progress tracking with percentage updates
10. Batch task submission with atomic enqueue

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Max Queue Depth | 1,000 | 10,000 | 100,000 |
| Worker Pool Size | 2 | 5 | 20 |
| Max Retries | 2 | 3 | 3 |
| Visibility Timeout | 120s | 300s | 300s |
| Task TTL Default | 30m | 1h | 1h |
| DLQ Retention | 24h | 72h | 168h (7d) |

### TR-003: Inter-Agent Communication Protocol

Standardized message protocol for agent-to-agent and agent-to-system communication.

**DECISION: Dual-transport architecture** — delivery guarantees must match the underlying transport.

**Message Envelope:**
```python
@dataclass
class MessageEnvelope:
    id: UUID                         # Unique message ID
    correlation_id: UUID             # Trace correlation ID
    source_agent: str                # Source agent identifier
    target_agent: str                # Target agent identifier (or channel name)
    message_type: MessageType        # REQUEST, RESPONSE, EVENT, COMMAND, QUERY
    channel_type: ChannelType        # DURABLE or EPHEMERAL
    payload: Dict[str, Any]          # Typed payload
    metadata: Dict[str, str]         # Headers (tenant_id, priority, ttl)
    timestamp: datetime              # UTC creation timestamp
    schema_version: str              # Message schema version
    reply_to: Optional[str]          # Reply channel for request/response
```

**Channel Types and Transport Mapping:**

| Channel Type | Transport | Guarantee | Ack | DLQ | Use Cases |
|---|---|---|---|---|---|
| **Durable** (request/response, commands) | Redis Streams (XADD/XREADGROUP/XACK) | At-least-once, persistent | Yes, explicit XACK | Yes, after max retries | Agent-to-agent work requests, task delegation, pipeline steps |
| **Ephemeral** (events, notifications) | Redis Pub/Sub | Best-effort, fire-and-forget | No | No | Config change notifications, telemetry signals, health broadcasts, metric events |
| **Config reload** | Redis Pub/Sub + periodic PostgreSQL reconcile | Eventual consistency (pub/sub + 30s reconcile) | No | No | Config hot-reload propagation |

**Requirements:**
1. Typed message envelopes with schema validation and explicit `channel_type` field
2. **Durable channels** (Redis Streams): point-to-point messaging with consumer groups, acknowledgment, retry, and DLQ
3. **Durable channels**: request/response pattern with reply streams and configurable timeout
4. **Ephemeral channels** (Redis Pub/Sub): broadcast and fan-out events with no delivery guarantee
5. **Ephemeral channels**: named topics with subscriber management (subscribe/unsubscribe)
6. Message serialization: JSON (default), MessagePack (high-throughput durable channels)
7. Named channels with topic-based routing (e.g., `durable:carbon.calculate`, `ephemeral:agent.health`)
8. Message TTL and automatic expiration (durable only — Redis Stream MAXLEN/MINID trimming)
9. Dead-letter routing for undeliverable durable messages (after max retries)
10. Clear API distinction: `send_durable(envelope)` vs `publish_event(envelope)` — no ambiguity

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Max Message Size | 256KB | 1MB | 1MB |
| Durable Message TTL | 5m | 15m | 30m |
| Durable Ack Timeout | 30s | 60s | 60s |
| Durable Max Retries | 2 | 3 | 3 |
| Ephemeral Max Subscribers | 50 | 200 | 1000 |
| Config Reconcile Interval | 10s | 30s | 30s |
| Serialization (Durable) | JSON | JSON | MessagePack |
| Serialization (Ephemeral) | JSON | JSON | JSON |

### TR-004: Agent Circuit Breaker

Per-agent circuit breaker to prevent cascading failures.

**State Machine:**
```
CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
                                          ↓
                                       CLOSED (recovered) or OPEN (still failing)
```

**Requirements:**
1. Per-agent circuit breaker with configurable thresholds
2. Failure rate threshold: open circuit when error rate > 50% in 60s window
3. Slow call threshold: open circuit when P99 > 5x target latency
4. Half-open state: allow 3 test requests before deciding recovery
5. Fallback chain: primary agent → secondary agent → default response
6. Bulkhead isolation: limit concurrent executions per agent (semaphore-based)
7. Retry with exponential backoff: base 1s, max 30s, jitter 0-500ms
8. Execution timeout: configurable per agent (default: 60s)
9. Prometheus metrics: circuit state, failure count, fallback invocations
10. Event emission on state transitions (for alerting)

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Failure Rate Threshold | 60% | 50% | 50% |
| Slow Call Duration | 10s | 5s | 5s |
| Wait in Open | 30s | 60s | 60s |
| Half-Open Test Requests | 2 | 3 | 3 |
| Max Concurrent Per Agent | 5 | 20 | 50 |
| Default Timeout | 30s | 60s | 60s |

### TR-005: Agent Packaging

Standardized agent packaging format for distribution and deployment.

**Pack Format (agent.pack.yaml):**
```yaml
pack:
  name: gl-carbon-agent
  version: 2.1.0
  description: "Carbon emission calculation agent"
  agent_type: deterministic          # deterministic | reasoning | insight
  spec_version: "2.0"

  entry_point: agent.py              # Main agent module
  base_class: DeterministicAgent     # Base class used

  dependencies:
    agents:
      - name: gl-unit-normalizer
        version: ">=1.0.0,<2.0.0"
      - name: gl-emission-factor-cache
        version: "^3.0.0"
    python:
      - numpy>=1.24.0
      - pydantic>=2.0.0

  inputs:
    schema: schemas/input.json
  outputs:
    schema: schemas/output.json

  resources:
    cpu_limit: "500m"
    memory_limit: "512Mi"
    timeout_seconds: 60

  metadata:
    author: GreenLang Framework Team
    license: Apache-2.0
    tags: [carbon, emissions, scope1, scope2]
    regulatory: [GHG Protocol, ISO 14064]
```

**Requirements:**
1. `agent.pack.yaml` specification parser and validator
2. Package builder: collect source, tests, schemas, config into distributable archive
3. Dependency resolver: semantic version ranges, conflict detection, diamond dependency resolution
4. Package installer: unpack, validate checksums, register in local index
5. Package manifest: SHA-256 checksums for all files, optional GPG signatures
6. Package archive format: `.glpack` (tar.gz with metadata header)
7. Backward compatibility checking: detect breaking changes in input/output schemas
8. Size limits: max 50MB per package (configurable)

### TR-006: Agent Developer CLI

Command-line interface for agent development workflow.

**Commands:**

| Command | Description | Example |
|---|---|---|
| `gl agent create` | Scaffold new agent from template | `gl agent create --template deterministic --name carbon-v3` |
| `gl agent test` | Run agent test suite | `gl agent test --unit --integration --coverage` |
| `gl agent deploy` | Deploy agent to environment | `gl agent deploy --env staging --canary 5%` |
| `gl agent rollback` | Rollback agent to previous version | `gl agent rollback --version 2.0.1` |
| `gl agent status` | Show agent status and health | `gl agent status --all` or `gl agent status carbon-agent` |
| `gl agent logs` | Stream agent execution logs | `gl agent logs carbon-agent --tail 100 --follow` |
| `gl agent inspect` | Inspect agent details | `gl agent inspect carbon-agent --deps --config` |
| `gl agent pack` | Build distributable package | `gl agent pack --output ./dist --sign` |
| `gl agent publish` | Publish to Agent Hub | `gl agent publish --hub internal --tag latest` |

**Requirements:**
1. Rich CLI output with colors, tables, progress bars (via `rich` library)
2. Interactive agent creation wizard with template selection
3. Dry-run mode for deploy/rollback commands
4. JSON output mode for CI/CD integration (`--output json`)
5. Configuration file support (`.gl-agent.yaml` in agent directory)
6. Bash/Zsh completion support
7. Verbose/debug logging modes (`-v`, `-vv`)

### TR-007: Agent Versioning & Canary Deployment

**Requirements:**
1. Semantic versioning enforcement (MAJOR.MINOR.PATCH)
2. Version compatibility matrix tracking (agent-to-agent, agent-to-platform)
3. Breaking change detection via input/output schema comparison
4. Canary deployment controller: configurable traffic percentages (5% → 25% → 50% → 100%)
5. Canary metrics evaluation: auto-promote if error rate < threshold, auto-rollback otherwise
6. Blue-green deployment support: instant traffic switch with rollback
7. Version migration framework: up/down migration scripts per version
8. Automated rollback triggers: error rate > 5%, P99 > 2x baseline, health check failures

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Canary Steps | 50%, 100% | 5%, 25%, 100% | 5%, 10%, 25%, 50%, 100% |
| Canary Step Duration | 1m | 5m | 15m |
| Error Rate Rollback | 20% | 10% | 5% |
| P99 Rollback Multiplier | 5x | 3x | 2x |
| Min Canary Requests | 10 | 50 | 200 |

### TR-008: Agent Sandbox & Isolation

**DECISION: Two-tier isolation (trimmed from original spec).** Process-level for pool-mode agents (v1 scope), pod-level for untrusted agents (K8s Job in restricted namespace). "Log all system calls" is deferred to v2 — v1 tracks resource usage and exit codes only.

**v1 Scope (Process-level sandbox for pool-mode agents):**
1. Process-level isolation via `asyncio.create_subprocess_exec` with clean environment
2. Resource limits: memory (via `resource.setrlimit` on Linux, best-effort on macOS/Windows), timeout enforcement
3. Execution timeout with forced termination (SIGTERM → wait 5s → SIGKILL)
4. Filesystem isolation: writable temp directory per invocation, cleaned up after execution
5. Environment variable filtering: only `GL_*`, `OTEL_*`, and explicit allowlist passed to subprocess
6. Shared-nothing execution: each agent invocation gets clean state and fresh temp directory
7. Audit trail: record exit code, duration, peak memory, stdout/stderr (capped at 10MB), error category

**v1 Scope (Pod-level sandbox for untrusted/dedicated agents):**
8. Untrusted agents run as K8s Job in `greenlang-agent-sandbox` namespace with restricted PSS
9. K8s resource limits (CPU, memory) and NetworkPolicy (egress allowlist only)
10. Pod-level timeout via `activeDeadlineSeconds`

**Deferred to v2:**
- Syscall auditing (seccomp profiles)
- Fine-grained disk I/O limits (cgroup v2)
- Network bandwidth throttling
- Vault integration for secrets injection

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Process CPU Limit | best-effort | best-effort | best-effort |
| Process Memory Limit | 512Mi | 1Gi | 2Gi |
| Max Execution Time | 120s | 300s | 600s |
| Temp Dir Size | 100MB | 500MB | 1GB |
| Pod CPU Limit (untrusted) | 1 core | 2 cores | 4 cores |
| Pod Memory Limit (untrusted) | 1Gi | 2Gi | 4Gi |

### TR-009: Agent Cost Metering

**Requirements:**
1. Per-agent cost tracking: compute time, LLM token usage, storage, network I/O
2. Budget allocation per agent with configurable thresholds (warn at 80%, block at 100%)
3. Per-tenant cost attribution for multi-tenant deployments
4. Resource quota enforcement: max concurrent executions, max daily executions
5. Billing event emission: structured events for downstream accounting
6. Cost dashboard integration: per-agent, per-tenant, per-environment views
7. Historical cost analysis: trend detection, anomaly alerting
8. Cost optimization recommendations: idle agents, oversized resource limits

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Default Budget/Agent/Day | $10 | $50 | $200 |
| Budget Alert Threshold | 90% | 80% | 80% |
| Max Concurrent Executions | 5 | 20 | 100 |
| Max Daily Executions | 100 | 1,000 | 50,000 |
| Metrics Retention | 7d | 30d | 365d |

### TR-010: Agent Telemetry & Distributed Tracing

**Requirements:**
1. OpenTelemetry integration for all agent execution paths
2. Span creation: one parent span per agent invocation, child spans for lifecycle phases
3. Correlation ID propagation through inter-agent message chains
4. Automatic context injection: agent_id, version, tenant_id, environment
5. Trace export: OTLP (Jaeger/Tempo), configurable sampling (default: 10% in prod)
6. Custom metrics: execution duration, queue wait time, retry count, cost
7. Exemplar linking: connect metrics to trace IDs for drill-down
8. Error classification: categorize failures (input_error, timeout, dependency_failure, internal_error)

---

## Operations API Pattern (DECISION: Async 202 Accepted)

Long-running operations (deploy, rollback, package build) use the **202 Accepted** pattern to avoid client timeouts and enable idempotent retries.

**Flow:**
```
POST /api/v1/factory/agents/{key}/deploy
  → 202 Accepted
  → Body: { "operation_id": "op-abc123", "status": "pending", "poll_url": "/api/v1/factory/operations/op-abc123" }

GET /api/v1/factory/operations/op-abc123
  → 200 OK
  → Body: { "operation_id": "op-abc123", "status": "running", "progress_pct": 45, "started_at": "..." }

DELETE /api/v1/factory/operations/op-abc123
  → 200 OK (cancellation requested)
```

**Operations that return 202:**

| Endpoint | Duration | Idempotency Key |
|---|---|---|
| `POST /agents/{key}/deploy` | 30s - 5m | `deploy:{agent_key}:{version}:{env}` |
| `POST /agents/{key}/rollback` | 10s - 2m | `rollback:{agent_key}:{target_version}` |
| `POST /agents/{key}/pack` | 5s - 30s | `pack:{agent_key}:{version}:{checksum}` |
| `POST /hub/publish` | 5s - 60s | `publish:{package_name}:{version}` |

**Requirements:**
1. All long-running ops return `202 Accepted` with `operation_id` and `poll_url`
2. `GET /operations/{id}` returns status, progress_pct, started_at, error (if failed)
3. `DELETE /operations/{id}` requests cancellation (best-effort, may not be immediate)
4. Idempotency keys prevent duplicate operations (key stored in Redis with 24h TTL)
5. Operations stored in PostgreSQL `infrastructure.agent_operations` table with status history
6. WebSocket endpoint `/ws/operations/{id}` for real-time status (optional, P2)

---

## Integration Points

### Integration with Existing Agent Infrastructure

```python
# 1. Existing base classes are preserved - Factory wraps them
from greenlang.agents.base_agents import DeterministicAgent, ReasoningAgent, InsightAgent
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base

# 2. Factory lifecycle hooks into AgentSpec v2 lifecycle
class ManagedAgent(AgentSpecV2Base[InT, OutT]):
    """Agent with factory lifecycle management."""

    async def on_factory_deploy(self, context: DeployContext) -> None:
        """Called by lifecycle manager during deployment."""
        await self.initialize()
        await self._warmup(context)

    async def on_factory_drain(self, timeout: float) -> None:
        """Called during graceful shutdown."""
        await self._complete_inflight()
        await self.finalize()
```

### Integration with INFRA-001 (Kubernetes)
- Agent Factory controller runs as Deployment in `greenlang-agent-factory` namespace
- Queue workers scale via HPA based on queue depth
- Canary deployments use Kubernetes traffic splitting (Istio/native)

### Integration with INFRA-002 (PostgreSQL)
- Agent registry, audit trail, cost metrics stored in `infrastructure.agent_*` tables
- Migration V008 extends existing schema with TimescaleDB hypertables for metrics

### Integration with INFRA-003 (Redis)
- Task queue uses Redis Streams with consumer groups
- Config hot-reload uses Redis pub/sub
- Circuit breaker state stored in Redis with TTL
- Message bus uses Redis pub/sub for inter-agent messaging

### Integration with INFRA-006 (Kong API Gateway)
- Factory API exposed through Kong at `/api/v1/factory/*`
- Rate limiting per consumer for factory endpoints
- JWT authentication required for all factory operations

### Integration with INFRA-007 (CI/CD)
- `gl agent test` integrates with GitHub Actions for automated testing
- `gl agent deploy` triggers ArgoCD sync for Kubernetes deployment
- Package checksums verified in CI pipeline before deployment

### Integration with INFRA-008 (Feature Flags)
- Agent canary deployments controlled by feature flags
- New agents gated behind feature flags during rollout
- Flag evaluation integrated into agent lifecycle (skip disabled agents)

### Integration with INFRA-009 (Log Aggregation)
- Agent execution logs sent to Loki with structured labels
- Correlation IDs from telemetry included in log entries
- Agent-specific log retention policies (operational: 30d, audit: 365d)

---

## Database Migration

### V008__agent_factory.sql

```sql
-- ============================================================
-- V008: Agent Factory v1.0 Schema
-- ============================================================

-- Agent Registry Table
CREATE TABLE IF NOT EXISTS infrastructure.agent_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_key VARCHAR(128) NOT NULL UNIQUE,
    display_name VARCHAR(256) NOT NULL,
    description TEXT,
    agent_type VARCHAR(32) NOT NULL CHECK (agent_type IN ('deterministic', 'reasoning', 'insight')),
    base_class VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'created' CHECK (status IN (
        'created', 'validating', 'validated', 'deploying', 'warming_up',
        'running', 'degraded', 'draining', 'retired', 'failed', 'force_stopped'
    )),
    entry_point VARCHAR(512) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    resource_limits JSONB NOT NULL DEFAULT '{"cpu": "500m", "memory": "512Mi", "timeout_seconds": 60}',
    metadata JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',
    tenant_id UUID,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

CREATE INDEX idx_ar_key_version ON infrastructure.agent_registry(agent_key, version);
CREATE INDEX idx_ar_status ON infrastructure.agent_registry(status);
CREATE INDEX idx_ar_type ON infrastructure.agent_registry(agent_type);
CREATE INDEX idx_ar_tenant ON infrastructure.agent_registry(tenant_id);
CREATE INDEX idx_ar_tags ON infrastructure.agent_registry USING GIN(tags);

-- Agent Versions Table (version history)
CREATE TABLE IF NOT EXISTS infrastructure.agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    version VARCHAR(32) NOT NULL,
    previous_version VARCHAR(32),
    changelog TEXT,
    input_schema JSONB,
    output_schema JSONB,
    pack_checksum VARCHAR(128),
    pack_url VARCHAR(1024),
    is_breaking BOOLEAN NOT NULL DEFAULT false,
    deployment_strategy VARCHAR(32) NOT NULL DEFAULT 'rolling' CHECK (
        deployment_strategy IN ('rolling', 'canary', 'blue_green', 'recreate')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128) NOT NULL,
    UNIQUE(agent_id, version)
);

CREATE INDEX idx_av_agent ON infrastructure.agent_versions(agent_id);
CREATE INDEX idx_av_version ON infrastructure.agent_versions(version);

-- Agent Dependencies Table (dependency graph)
CREATE TABLE IF NOT EXISTS infrastructure.agent_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    depends_on_agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    version_constraint VARCHAR(64) NOT NULL DEFAULT '*',
    dependency_type VARCHAR(32) NOT NULL DEFAULT 'runtime' CHECK (
        dependency_type IN ('runtime', 'build', 'test', 'optional')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, depends_on_agent_id, dependency_type),
    CHECK(agent_id != depends_on_agent_id)
);

CREATE INDEX idx_ad_agent ON infrastructure.agent_dependencies(agent_id);
CREATE INDEX idx_ad_depends ON infrastructure.agent_dependencies(depends_on_agent_id);

-- Agent Executions Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS infrastructure.agent_executions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    agent_key VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL,
    tenant_id UUID,
    correlation_id UUID,
    status VARCHAR(32) NOT NULL CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'timeout', 'cancelled'
    )),
    priority INTEGER NOT NULL DEFAULT 2,
    input_hash VARCHAR(128),
    output_hash VARCHAR(128),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    error_message TEXT,
    error_category VARCHAR(64),
    retry_count INTEGER NOT NULL DEFAULT 0,
    cost_compute_usd NUMERIC(10, 6) DEFAULT 0,
    cost_tokens_usd NUMERIC(10, 6) DEFAULT 0,
    cost_storage_usd NUMERIC(10, 6) DEFAULT 0,
    cost_total_usd NUMERIC(10, 6) DEFAULT 0,
    resource_cpu_ms BIGINT DEFAULT 0,
    resource_memory_mb_peak INTEGER DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for time-series metrics
SELECT create_hypertable('infrastructure.agent_executions', 'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_ae_agent_time ON infrastructure.agent_executions(agent_id, created_at DESC);
CREATE INDEX idx_ae_tenant_time ON infrastructure.agent_executions(tenant_id, created_at DESC);
CREATE INDEX idx_ae_correlation ON infrastructure.agent_executions(correlation_id);
CREATE INDEX idx_ae_status ON infrastructure.agent_executions(status, created_at DESC);

-- Agent Circuit Breaker State Table
-- DECISION: Rate-based thresholds (not count-based) to match the sliding-window
-- implementation in resilience/circuit_breaker.py. The config JSONB stores
-- window_seconds, error_rate_threshold_pct, and slow_call_p99_ms_threshold.
CREATE TABLE IF NOT EXISTS infrastructure.agent_circuit_breaker (
    agent_key VARCHAR(128) PRIMARY KEY,
    state VARCHAR(16) NOT NULL DEFAULT 'closed' CHECK (state IN ('closed', 'open', 'half_open')),
    failure_count INTEGER NOT NULL DEFAULT 0,       -- current window failures (for observability)
    success_count INTEGER NOT NULL DEFAULT 0,       -- current window successes (for observability)
    total_calls_in_window INTEGER NOT NULL DEFAULT 0,
    error_rate_pct NUMERIC(5, 2) NOT NULL DEFAULT 0, -- current error rate (for dashboards)
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    half_opened_at TIMESTAMPTZ,
    config JSONB NOT NULL DEFAULT '{
        "window_seconds": 60,
        "error_rate_threshold_pct": 50,
        "slow_call_p99_ms_threshold": 5000,
        "slow_call_rate_threshold_pct": 80,
        "wait_in_open_seconds": 60,
        "half_open_test_requests": 3,
        "minimum_calls_in_window": 5
    }',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent Cost Budgets Table
CREATE TABLE IF NOT EXISTS infrastructure.agent_cost_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES infrastructure.agent_registry(id),
    tenant_id UUID,
    budget_period VARCHAR(16) NOT NULL DEFAULT 'daily' CHECK (budget_period IN ('hourly', 'daily', 'weekly', 'monthly')),
    budget_amount_usd NUMERIC(10, 2) NOT NULL,
    spent_amount_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    alert_threshold_pct INTEGER NOT NULL DEFAULT 80,
    is_hard_limit BOOLEAN NOT NULL DEFAULT false,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_acb_agent ON infrastructure.agent_cost_budgets(agent_id);
CREATE INDEX idx_acb_tenant ON infrastructure.agent_cost_budgets(tenant_id);
CREATE INDEX idx_acb_period ON infrastructure.agent_cost_budgets(period_start, period_end);

-- Agent Audit Log Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS infrastructure.agent_audit_log (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_id UUID,
    agent_key VARCHAR(128),
    action VARCHAR(64) NOT NULL,
    actor VARCHAR(128) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    previous_state JSONB,
    new_state JSONB,
    ip_address INET,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('infrastructure.agent_audit_log', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_aal_agent ON infrastructure.agent_audit_log(agent_id, created_at DESC);
CREATE INDEX idx_aal_action ON infrastructure.agent_audit_log(action, created_at DESC);
CREATE INDEX idx_aal_actor ON infrastructure.agent_audit_log(actor, created_at DESC);

-- Agent Tenant Configuration Table (per-tenant overrides)
-- Supports version pinning, quota overrides, and feature gating per tenant.
CREATE TABLE IF NOT EXISTS infrastructure.agent_tenant_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    agent_key VARCHAR(128) NOT NULL,
    pinned_version VARCHAR(32),            -- NULL = follow default (latest stable)
    execution_mode VARCHAR(16) CHECK (execution_mode IN ('pool', 'dedicated')),
    max_concurrent INTEGER,                -- overrides global default
    max_daily_executions INTEGER,           -- overrides global default
    budget_override_usd NUMERIC(10, 2),     -- overrides agent default budget
    config_overrides JSONB NOT NULL DEFAULT '{}',  -- merged on top of agent config
    enabled BOOLEAN NOT NULL DEFAULT true,   -- kill switch per tenant
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, agent_key)
);

CREATE INDEX idx_atc_tenant ON infrastructure.agent_tenant_config(tenant_id);
CREATE INDEX idx_atc_agent ON infrastructure.agent_tenant_config(agent_key);

ALTER TABLE infrastructure.agent_tenant_config ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_tenant_config_isolation ON infrastructure.agent_tenant_config
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE TRIGGER trg_agent_tenant_config_updated
    BEFORE UPDATE ON infrastructure.agent_tenant_config
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

-- Agent Execution Metrics - Continuous Aggregate (5-minute buckets)
CREATE MATERIALIZED VIEW IF NOT EXISTS infrastructure.agent_metrics_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', created_at) AS bucket,
    agent_key,
    COUNT(*) AS execution_count,
    COUNT(*) FILTER (WHERE status = 'completed') AS success_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failure_count,
    COUNT(*) FILTER (WHERE status = 'timeout') AS timeout_count,
    AVG(duration_ms) AS avg_duration_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) AS p50_duration_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_duration_ms,
    SUM(cost_total_usd) AS total_cost_usd,
    AVG(resource_cpu_ms) AS avg_cpu_ms,
    MAX(resource_memory_mb_peak) AS max_memory_mb
FROM infrastructure.agent_executions
GROUP BY bucket, agent_key
WITH NO DATA;

SELECT add_continuous_aggregate_policy('infrastructure.agent_metrics_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Retention policies
SELECT add_retention_policy('infrastructure.agent_executions', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('infrastructure.agent_audit_log', INTERVAL '365 days', if_not_exists => TRUE);

-- Row-Level Security
ALTER TABLE infrastructure.agent_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_cost_budgets ENABLE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY agent_registry_tenant_isolation ON infrastructure.agent_registry
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_executions_tenant_isolation ON infrastructure.agent_executions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_audit_tenant_isolation ON infrastructure.agent_audit_log
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_budgets_tenant_isolation ON infrastructure.agent_cost_budgets
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

-- Updated-at trigger
CREATE OR REPLACE FUNCTION infrastructure.update_agent_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_agent_registry_updated
    BEFORE UPDATE ON infrastructure.agent_registry
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

CREATE TRIGGER trg_agent_budgets_updated
    BEFORE UPDATE ON infrastructure.agent_cost_budgets
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();
```

---

## Migration Plan

### Phase 1: Core Infrastructure (P0)
1. Create `greenlang/infrastructure/agent_factory/` package structure
2. Implement lifecycle manager: states, health checks, warm-up, shutdown
3. Implement distributed task queue: Redis Streams, priority, DLQ, workers
4. Implement inter-agent messaging protocol: envelopes, routing, acknowledgment
5. Run database migration V008 (9 tables, 2 hypertables, 1 continuous aggregate)
6. Deploy lifecycle manager to K8s

### Phase 2: Resilience & Runtime (P0)
1. Implement circuit breaker: state machine, thresholds, fallback chains
2. Implement bulkhead isolation and retry policies
3. Implement agent sandbox: process isolation, resource limits, timeout guard
4. Implement cost metering: tracker, budgets, quotas, billing events
5. Implement config hot-reload: Redis pub/sub watcher, validation, rollback

### Phase 3: Packaging & Versioning (P1)
1. Implement agent packaging: pack format, builder, resolver, installer
2. Implement versioning: semver, compatibility, canary controller, rollback
3. Implement dependency graph: DAG builder, topological sort, cycle detection
4. Implement Agent Hub registry: publish, search, download, validate

### Phase 4: Developer Experience (P1)
1. Implement CLI commands: create, test, deploy, rollback, status, logs, inspect, pack, publish
2. Implement agent telemetry: OpenTelemetry tracer, spans, correlation, export
3. Extend REST API with 15+ new endpoints for factory operations
4. Implement factory-specific FastAPI middleware

### Phase 5: Deployment & Monitoring (P2)
1. Create K8s manifests: namespace, deployments, services, HPA, networkpolicy, RBAC
2. Create/update Helm chart templates for factory services
3. Create Grafana dashboard (20+ panels): agent health, queue depth, cost, SLA
4. Create Prometheus alert rules (15+ rules): circuit breaker, queue backlog, budget, latency

### Phase 6: Testing & Validation (P2)
1. Create unit tests for all 13 modules (12 test files, 200+ tests)
2. Create integration tests (4 test files: lifecycle, queue, messaging, e2e)
3. Create load tests (queue throughput, concurrent agent execution)
4. Validate 85%+ code coverage across all new modules

---

## Backward Compatibility & Migration (119 Existing Agents)

The Agent Factory must support the 119 agents already built without requiring immediate rewrite. Migration is opt-in and incremental.

### Legacy Compatibility Mode

Agents that do **not** have an `agent.pack.yaml` run in **legacy mode**:

1. **Auto-discovery**: At startup, the factory scans `greenlang/agents/` for classes extending `AgentSpecV2Base`, `DeterministicAgent`, `ReasoningAgent`, or `InsightAgent`
2. **Synthetic registration**: Each discovered agent is registered in `infrastructure.agent_registry` with:
   - `agent_key`: derived from module path (e.g., `greenlang.agents.intake.carbon_intake` → `gl-carbon-intake`)
   - `version`: `"0.1.0"` (retroactive baseline)
   - `status`: `"running"` (already deployed)
   - `entry_point`: Python module path
   - `metadata`: `{"legacy": true, "migrated": false}`
3. **No packaging required**: Legacy agents skip pack validation and run directly from the Python module path
4. **Opt-in lifecycle**: Legacy agents get health checks and telemetry but skip circuit breakers (can be enabled per-agent)
5. **Migration path**: Developers run `gl agent pack --legacy <module_path>` to generate `agent.pack.yaml` from existing code

### Migration Playbook

| Step | Action | Owner | Timeline |
|---|---|---|---|
| 1 | Run auto-discovery, register all 119 agents | Agent Factory (automatic) | Day 1 |
| 2 | Add telemetry and health checks to legacy agents | Agent Factory (automatic) | Day 1 |
| 3 | Categorize agents by execution frequency | Platform team | Week 1 |
| 4 | Pack top-20 agents (by execution count) | Agent owners | Week 2-3 |
| 5 | Enable circuit breakers for top-20 packed agents | Platform team | Week 3 |
| 6 | Pack remaining agents in batches of 20 | Agent owners | Week 4-8 |
| 7 | Promote high-traffic agents to dedicated mode | Platform team | Week 6+ |

### Migration CLI

```bash
# Generate pack.yaml from existing agent module
gl agent pack --legacy greenlang.agents.intake.carbon_intake

# Validate legacy agent compatibility
gl agent test --legacy greenlang.agents.intake.carbon_intake

# Register legacy agent in factory without packaging
gl agent register --legacy greenlang.agents.intake.carbon_intake --version 0.1.0
```

---

## Acceptance Criteria

1. Agent lifecycle state machine transitions are correct and atomic with distributed locking
2. Task queue processes 1,000+ tasks/minute with priority ordering and at-least-once delivery
3. Inter-agent message delivery latency < 10ms P99 (same cluster, Redis pub/sub)
4. Circuit breaker opens within 5 seconds of failure threshold breach
5. Agent packages build, install, and validate with dependency resolution in < 30 seconds
6. `gl agent create` scaffolds a working agent with tests in < 5 seconds
7. Canary deployment promotes/rollbacks automatically based on metrics thresholds
8. Sandbox enforces CPU/memory limits with < 5% overhead
9. Cost metering tracks per-agent costs with < 1% error margin
10. Config hot-reload propagates changes to all instances within 5 seconds
11. Dependency graph detects circular dependencies and reports diagnostic path
12. OpenTelemetry traces cover 100% of agent execution paths with correlation IDs
13. Agent Hub supports publish/search/download with package integrity verification
14. REST API returns < 100ms P99 for all factory endpoints
15. All new modules achieve 85%+ test coverage
16. Database migration V008 runs cleanly with zero downtime (online migration)

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| INFRA-001: EKS Cluster | COMPLETE | Factory services deploy on EKS |
| INFRA-002: PostgreSQL | COMPLETE | Agent registry, audit trail, metrics |
| INFRA-003: Redis | COMPLETE | Task queue, config cache, message bus, circuit breaker state |
| INFRA-004: S3 | COMPLETE | Agent packages, artifacts storage |
| INFRA-005: pgvector | COMPLETE | Agent semantic search (optional) |
| INFRA-006: Kong API Gateway | COMPLETE | Factory API routing, auth, rate limiting |
| INFRA-007: CI/CD Pipelines | COMPLETE | Agent testing and deployment automation |
| INFRA-008: Feature Flags | COMPLETE | Canary deployment gates, agent rollout control |
| INFRA-009: Log Aggregation | COMPLETE | Agent execution logs, audit trail |
| AgentSpec v2 | EXISTS | `greenlang/agents/agentspec_v2_base.py` |
| Agent Base Classes | EXISTS | `greenlang/agents/base_agents.py` |
| Agent Registry | EXISTS | `greenlang/cli/agent_registry.py`, `greenlang/agents/registry.py` |
| Agent API | EXISTS | `greenlang/execution/infrastructure/api/routes/agents_routes.py` |
| Existing Factory | EXISTS | `greenlang/utilities/factory/agent_factory.py` |
| Message Bus | EXISTS | `greenlang/execution/core/message_bus.py` |

---

## Development Tasks (Ralphy-Compatible)

### Phase 1: Core Infrastructure
- [ ] Create package: `greenlang/infrastructure/agent_factory/__init__.py`
- [ ] Create lifecycle states: `greenlang/infrastructure/agent_factory/lifecycle/states.py`
- [ ] Create lifecycle manager: `greenlang/infrastructure/agent_factory/lifecycle/manager.py`
- [ ] Create health checks: `greenlang/infrastructure/agent_factory/lifecycle/health.py`
- [ ] Create warm-up: `greenlang/infrastructure/agent_factory/lifecycle/warmup.py`
- [ ] Create shutdown: `greenlang/infrastructure/agent_factory/lifecycle/shutdown.py`
- [ ] Create task queue: `greenlang/infrastructure/agent_factory/queue/task_queue.py`
- [ ] Create priority scheduler: `greenlang/infrastructure/agent_factory/queue/priority.py`
- [ ] Create dead-letter queue: `greenlang/infrastructure/agent_factory/queue/dead_letter.py`
- [ ] Create workers: `greenlang/infrastructure/agent_factory/queue/workers.py`
- [ ] Create scheduler: `greenlang/infrastructure/agent_factory/queue/scheduler.py`
- [ ] Create messaging protocol: `greenlang/infrastructure/agent_factory/messaging/protocol.py`
- [ ] Create message router: `greenlang/infrastructure/agent_factory/messaging/router.py`
- [ ] Create serialization: `greenlang/infrastructure/agent_factory/messaging/serialization.py`
- [ ] Create acknowledgment: `greenlang/infrastructure/agent_factory/messaging/acknowledgment.py`
- [ ] Create channels: `greenlang/infrastructure/agent_factory/messaging/channels.py`
- [ ] Create database migration: `deployment/database/migrations/sql/V008__agent_factory.sql`

### Phase 2: Resilience & Runtime
- [ ] Create circuit breaker: `greenlang/infrastructure/agent_factory/resilience/circuit_breaker.py`
- [ ] Create fallback chains: `greenlang/infrastructure/agent_factory/resilience/fallback.py`
- [ ] Create bulkhead: `greenlang/infrastructure/agent_factory/resilience/bulkhead.py`
- [ ] Create retry policies: `greenlang/infrastructure/agent_factory/resilience/retry.py`
- [ ] Create timeout: `greenlang/infrastructure/agent_factory/resilience/timeout.py`
- [ ] Create sandbox executor: `greenlang/infrastructure/agent_factory/sandbox/executor.py`
- [ ] Create resource limits: `greenlang/infrastructure/agent_factory/sandbox/resource_limits.py`
- [ ] Create timeout guard: `greenlang/infrastructure/agent_factory/sandbox/timeout_guard.py`
- [ ] Create sandbox audit: `greenlang/infrastructure/agent_factory/sandbox/audit.py`
- [ ] Create cost tracker: `greenlang/infrastructure/agent_factory/metering/cost_tracker.py`
- [ ] Create budget manager: `greenlang/infrastructure/agent_factory/metering/budget.py`
- [ ] Create resource quotas: `greenlang/infrastructure/agent_factory/metering/resource_quotas.py`
- [ ] Create billing events: `greenlang/infrastructure/agent_factory/metering/billing_events.py`
- [ ] Create hot-reload: `greenlang/infrastructure/agent_factory/config/hot_reload.py`
- [ ] Create config schema: `greenlang/infrastructure/agent_factory/config/schema.py`
- [ ] Create config store: `greenlang/infrastructure/agent_factory/config/store.py`
- [ ] Create config diff: `greenlang/infrastructure/agent_factory/config/diff.py`

### Phase 3: Packaging & Versioning
- [ ] Create pack format: `greenlang/infrastructure/agent_factory/packaging/pack_format.py`
- [ ] Create package builder: `greenlang/infrastructure/agent_factory/packaging/builder.py`
- [ ] Create dependency resolver: `greenlang/infrastructure/agent_factory/packaging/resolver.py`
- [ ] Create package installer: `greenlang/infrastructure/agent_factory/packaging/installer.py`
- [ ] Create manifest generator: `greenlang/infrastructure/agent_factory/packaging/manifest.py`
- [ ] Create semver module: `greenlang/infrastructure/agent_factory/versioning/semver.py`
- [ ] Create compatibility matrix: `greenlang/infrastructure/agent_factory/versioning/compatibility.py`
- [ ] Create migration framework: `greenlang/infrastructure/agent_factory/versioning/migration.py`
- [ ] Create canary controller: `greenlang/infrastructure/agent_factory/versioning/canary.py`
- [ ] Create rollback controller: `greenlang/infrastructure/agent_factory/versioning/rollback.py`
- [ ] Create dependency graph: `greenlang/infrastructure/agent_factory/dependencies/graph.py`
- [ ] Create topological resolver: `greenlang/infrastructure/agent_factory/dependencies/resolver.py`
- [ ] Create cycle detector: `greenlang/infrastructure/agent_factory/dependencies/cycle_detector.py`
- [ ] Create graph visualizer: `greenlang/infrastructure/agent_factory/dependencies/visualizer.py`
- [ ] Create Hub registry: `greenlang/infrastructure/agent_factory/hub/registry.py`
- [ ] Create Hub client: `greenlang/infrastructure/agent_factory/hub/client.py`
- [ ] Create Hub index: `greenlang/infrastructure/agent_factory/hub/index.py`
- [ ] Create Hub validator: `greenlang/infrastructure/agent_factory/hub/validator.py`

### Phase 4: Developer Experience
- [ ] Create CLI agent create: `greenlang/cli/commands/agent/create.py`
- [ ] Create CLI agent test: `greenlang/cli/commands/agent/test.py`
- [ ] Create CLI agent deploy: `greenlang/cli/commands/agent/deploy.py`
- [ ] Create CLI agent rollback: `greenlang/cli/commands/agent/rollback.py`
- [ ] Create CLI agent status: `greenlang/cli/commands/agent/status.py`
- [ ] Create CLI agent logs: `greenlang/cli/commands/agent/logs.py`
- [ ] Create CLI agent inspect: `greenlang/cli/commands/agent/inspect.py`
- [ ] Create CLI agent pack: `greenlang/cli/commands/agent/pack.py`
- [ ] Create CLI agent publish: `greenlang/cli/commands/agent/publish.py`
- [ ] Create telemetry tracer: `greenlang/infrastructure/agent_factory/telemetry/tracer.py`
- [ ] Create telemetry spans: `greenlang/infrastructure/agent_factory/telemetry/spans.py`
- [ ] Create telemetry correlation: `greenlang/infrastructure/agent_factory/telemetry/correlation.py`
- [ ] Create telemetry exporter: `greenlang/infrastructure/agent_factory/telemetry/exporter.py`
- [ ] Create metrics collector: `greenlang/infrastructure/agent_factory/telemetry/metrics_collector.py`
- [ ] Create factory API routes: `greenlang/infrastructure/agent_factory/api/factory_routes.py`
- [ ] Create lifecycle API routes: `greenlang/infrastructure/agent_factory/api/lifecycle_routes.py`
- [ ] Create queue API routes: `greenlang/infrastructure/agent_factory/api/queue_routes.py`
- [ ] Create hub API routes: `greenlang/infrastructure/agent_factory/api/hub_routes.py`
- [ ] Create factory middleware: `greenlang/infrastructure/agent_factory/api/middleware.py`

### Phase 5: Deployment & Monitoring
- [ ] Create K8s namespace: `deployment/kubernetes/agent-factory/namespace.yaml`
- [ ] Create K8s lifecycle deployment: `deployment/kubernetes/agent-factory/deployment-lifecycle-mgr.yaml`
- [ ] Create K8s queue worker deployment: `deployment/kubernetes/agent-factory/deployment-queue-worker.yaml`
- [ ] Create K8s hub deployment: `deployment/kubernetes/agent-factory/deployment-hub.yaml`
- [ ] Create K8s services: `deployment/kubernetes/agent-factory/service.yaml`
- [ ] Create K8s HPA: `deployment/kubernetes/agent-factory/hpa.yaml`
- [ ] Create K8s ConfigMap: `deployment/kubernetes/agent-factory/configmap.yaml`
- [ ] Create K8s NetworkPolicy: `deployment/kubernetes/agent-factory/networkpolicy.yaml`
- [ ] Create K8s RBAC: `deployment/kubernetes/agent-factory/rbac.yaml`
- [ ] Create K8s Kustomization: `deployment/kubernetes/agent-factory/kustomization.yaml`
- [ ] Create Helm lifecycle template: `deployment/helm/greenlang-agents/templates/deployment-lifecycle-mgr.yaml`
- [ ] Create Helm queue worker template: `deployment/helm/greenlang-agents/templates/deployment-queue-worker.yaml`
- [ ] Create Helm hub template: `deployment/helm/greenlang-agents/templates/deployment-hub.yaml`
- [ ] Create Grafana dashboard: `deployment/monitoring/dashboards/agent-factory-v1.json`
- [ ] Create Prometheus alerts: `deployment/monitoring/alerts/agent-factory-alerts.yaml`

### Phase 6: Testing
- [ ] Create test: `tests/unit/agent_factory/test_lifecycle_manager.py`
- [ ] Create test: `tests/unit/agent_factory/test_task_queue.py`
- [ ] Create test: `tests/unit/agent_factory/test_messaging_protocol.py`
- [ ] Create test: `tests/unit/agent_factory/test_circuit_breaker.py`
- [ ] Create test: `tests/unit/agent_factory/test_packaging.py`
- [ ] Create test: `tests/unit/agent_factory/test_versioning.py`
- [ ] Create test: `tests/unit/agent_factory/test_sandbox.py`
- [ ] Create test: `tests/unit/agent_factory/test_cost_metering.py`
- [ ] Create test: `tests/unit/agent_factory/test_hot_reload.py`
- [ ] Create test: `tests/unit/agent_factory/test_dependency_graph.py`
- [ ] Create test: `tests/unit/agent_factory/test_telemetry.py`
- [ ] Create test: `tests/unit/agent_factory/test_hub_registry.py`
- [ ] Create test: `tests/integration/agent_factory/test_lifecycle_integration.py`
- [ ] Create test: `tests/integration/agent_factory/test_queue_integration.py`
- [ ] Create test: `tests/integration/agent_factory/test_messaging_integration.py`
- [ ] Create test: `tests/integration/agent_factory/test_factory_e2e.py`
- [ ] Create test: `tests/load/agent_factory/test_queue_throughput.py`

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Redis Streams complexity for task queue | Medium | High | Use well-tested patterns, fallback to PostgreSQL-only queue |
| Circuit breaker false positives | Medium | Medium | Conservative thresholds, manual override capability |
| Agent sandbox performance overhead | Low | Medium | Benchmark overhead < 5%, disable sandbox for trusted agents |
| Package dependency resolution complexity | Medium | Medium | Limit dependency depth to 3, clear error messages |
| Canary deployment metrics noise | Medium | Medium | Minimum request threshold before evaluation |
| Hot-reload race conditions | Low | High | Distributed locking, version stamps, rollback on failure |

---

## Open Decisions (Engineering)

These items are intentionally left open for the implementing engineers to decide during development:

1. **Redis Streams consumer group naming**: Use `{agent_key}-workers` or `factory-workers-{shard}`? Depends on whether we want per-agent consumer groups or shared pools.
2. **Canary metric source**: Pull from Prometheus or push from agent telemetry? Prometheus is simpler but adds latency; push is faster but requires metric buffer.
3. **Pack signature format**: GPG, Sigstore/cosign, or simple SHA-256 + HMAC? GPG is heavyweight for internal use; Sigstore requires external infra; HMAC is simplest for v1.
4. **Hub storage backend**: S3 only, or S3 + local filesystem cache? Cache improves install speed but adds complexity.
5. **Legacy agent scan frequency**: One-time at factory boot, or periodic re-scan? Periodic catches new agents but may cause registration churn.
6. **Operations table**: Separate `infrastructure.agent_operations` table, or reuse `agent_audit_log` with filtered queries? Separate is cleaner; reuse avoids schema proliferation.
7. **WebSocket for operation status**: Implement in v1 or defer? If deferred, polling interval should be configurable (default 2s).
8. **Config hot-reload transport**: Redis pub/sub only, or add PostgreSQL LISTEN/NOTIFY as fallback? Pub/sub is faster; LISTEN/NOTIFY survives Redis outage.

---

## Appendix: Existing Code Integration Points

| Existing File | Integration Action |
|---|---|
| `greenlang/utilities/factory/agent_factory.py` | Wrap with lifecycle manager, add packaging output |
| `greenlang/agents/agentspec_v2_base.py` | Add factory lifecycle hooks (deploy, drain) |
| `greenlang/agents/base_agents.py` | Register with agent registry on import |
| `greenlang/cli/agent_registry.py` | Delegate to Hub registry, add version tracking |
| `greenlang/agents/registry.py` | Extend AgentInfo with packaging metadata |
| `greenlang/execution/infrastructure/api/routes/agents_routes.py` | Add factory endpoints, link to lifecycle manager |
| `greenlang/execution/infrastructure/api/services/agent_service.py` | Integrate with lifecycle manager states |
| `greenlang/execution/core/message_bus.py` | Wrap with typed messaging protocol |
| `greenlang/execution/core/workflow.py` | Add dependency graph resolution |
| `greenlang/governance/policy/agent_rbac.py` | Extend with factory-specific permissions |
| `deployment/kubernetes/agents/` | Keep existing, add factory-specific manifests |
| `deployment/helm/greenlang-agents/` | Add factory deployment templates |
