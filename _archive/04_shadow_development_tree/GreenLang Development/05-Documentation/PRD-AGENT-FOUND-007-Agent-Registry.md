# PRD: AGENT-FOUND-007 - GreenLang Agent Registry

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-007 |
| **Agent ID** | GL-FOUND-X-007 |
| **Component** | Agent Registry & Service Catalog |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical (service catalog backbone for all agents) |
| **Status** | Layer 1 Complete (~1,990 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS operates 47+ agents across 11 layers. Every agent must be
discoverable, version-tracked, health-monitored, and dependency-resolved before
it can participate in DAG-orchestrated pipelines. Without a production-grade
agent registry:

- **No service catalog**: Agents cannot be discovered by capability, layer, or sector
- **No health monitoring**: Unhealthy agents silently fail in pipelines
- **No version management**: Multiple agent versions cannot coexist safely
- **No dependency resolution**: Pipeline construction requires manual wiring
- **No GLIP v1 support**: Container-based agents cannot be registered with K8s specs
- **No hot-reload**: Agent updates require full system restarts
- **No audit trail**: Registration changes are untracked

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/agent_registry.py` (1,990 lines)
- `VersionedAgentRegistry` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-010)
- 7 enums: AgentLayer(11), SectorClassification(10), AgentHealthStatus(5), ExecutionMode(3), IdempotencySupport(3), CapabilityCategory(10)
- 12 Pydantic models: ResourceProfile, ContainerSpec, LegacyHttpConfig, SemanticVersion, AgentCapability, AgentVariant, AgentDependency, AgentMetadataEntry, RegistryQueryInput, RegistryQueryOutput, DependencyResolutionInput, DependencyResolutionOutput
- Registry operations: register_agent, unregister_agent, query_agents, get_agent, list_versions, get_agent_class
- Indexes: by_layer, by_sector, by_capability, by_tag (O(1) lookups)
- Health tracking: check_agent_health, set_agent_health
- Hot reload: register_reload_callback, hot_reload_agent
- GLIP v1 support: find_glip_compatible_agents, migrate_agent_to_glip, get_execution_context
- Dependency resolution: topological sort with cycle detection
- Export/Import: full registry serialization
- Factory functions: create_agent_metadata, create_glip_agent_metadata, create_legacy_agent_metadata
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
**File**: `tests/agents/foundation/test_agent_registry.py` (if exists)

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/agent_registry/` package providing a clean SDK for other agents/services.

### Gap 2: No Prometheus Metrics
No `greenlang/agent_registry/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_agent_registry(app)` / `get_agent_registry(app)` pattern.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 has in-memory storage; doesn't delegate to persistent integration module.

### Gap 5: No REST API Router
No `greenlang/agent_registry/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/agent-registry-service/` manifests.

### Gap 7: No Database Migration
No `V027__agent_registry_service.sql` for persistent registry storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/agent-registry-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for agent registry operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/agent_registry/
  __init__.py             # Public API exports
  config.py               # AgentRegistryConfig with GL_AGENT_REGISTRY_ env prefix
  models.py               # Pydantic v2 models (re-export + enhance from foundation agent)
  registry.py             # AgentRegistry: CRUD, indexing, versioning, hot-reload
  health_checker.py       # HealthChecker: probe agents, track status, TTL-based refresh
  dependency_resolver.py  # DependencyResolver: topological sort, cycle detection, version constraints
  capability_matcher.py   # CapabilityMatcher: find agents by required capabilities
  provenance.py           # ProvenanceTracker: SHA-256 hash chain for registry mutations
  metrics.py              # 12 Prometheus metrics
  setup.py                # AgentRegistryService facade, configure/get
  api/
    __init__.py
    router.py             # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V027)
```sql
CREATE SCHEMA agent_registry_service;
-- agents (agent metadata with version history)
-- agent_versions (individual version records)
-- agent_capabilities (capability definitions linked to agents)
-- agent_dependencies (dependency graph)
-- health_checks (hypertable - health probe results)
-- registry_audit_log (hypertable - registration/mutation audit trail)
-- agent_variants (geographic/fuel-type specializations)
-- service_catalog (published service catalog entries)
```

### 5.3 Prometheus Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_agent_registry_registrations_total` | Counter | Total agent registrations by layer |
| 2 | `gl_agent_registry_unregistrations_total` | Counter | Total agent unregistrations |
| 3 | `gl_agent_registry_queries_total` | Counter | Total registry queries by filter type |
| 4 | `gl_agent_registry_query_duration_seconds` | Histogram | Query latency |
| 5 | `gl_agent_registry_health_checks_total` | Counter | Health checks by status |
| 6 | `gl_agent_registry_unhealthy_agents_total` | Gauge | Currently unhealthy agents |
| 7 | `gl_agent_registry_agents_total` | Gauge | Total registered agents |
| 8 | `gl_agent_registry_versions_total` | Gauge | Total agent versions |
| 9 | `gl_agent_registry_dependency_resolutions_total` | Counter | Dependency resolutions by result |
| 10 | `gl_agent_registry_hot_reloads_total` | Counter | Hot reload operations |
| 11 | `gl_agent_registry_cache_hits_total` | Counter | Registry cache hits |
| 12 | `gl_agent_registry_cache_misses_total` | Counter | Registry cache misses |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/agents` | Register a new agent |
| GET | `/v1/agents` | List/query agents (with filters) |
| GET | `/v1/agents/{agent_id}` | Get agent metadata (latest version) |
| GET | `/v1/agents/{agent_id}/versions` | List all versions of an agent |
| GET | `/v1/agents/{agent_id}/versions/{version}` | Get specific version |
| PUT | `/v1/agents/{agent_id}` | Update agent metadata |
| DELETE | `/v1/agents/{agent_id}` | Unregister agent (all versions) |
| DELETE | `/v1/agents/{agent_id}/versions/{version}` | Unregister specific version |
| POST | `/v1/agents/{agent_id}/reload` | Hot-reload an agent |
| POST | `/v1/discover` | Discover agents by capabilities |
| POST | `/v1/dependencies/resolve` | Resolve dependency graph |
| GET | `/v1/health/{agent_id}` | Get agent health status |
| POST | `/v1/health/{agent_id}/check` | Trigger health check |
| PUT | `/v1/health/{agent_id}` | Set agent health status |
| GET | `/v1/catalog` | Get published service catalog |
| GET | `/v1/layers` | List agent layers with counts |
| GET | `/v1/statistics` | Get registry statistics |
| POST | `/v1/export` | Export registry data |
| POST | `/v1/import` | Import registry data |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Deterministic discovery**: No ML/probabilistic matching - all exact specification matching
2. **Version-safe**: Semantic versioning with compatibility checking
3. **Health-aware**: Proactive health probing with configurable intervals
4. **Hot-reload**: Zero-downtime agent updates
5. **GLIP v1 native**: First-class support for K8s Job-based agents
6. **Complete audit trail**: Every registration mutation logged with SHA-256 provenance
7. **Index-based queries**: O(1) lookups by layer, sector, capability, tag

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/agent_registry/__init__.py` - Public API exports
2. Create `greenlang/agent_registry/config.py` - AgentRegistryConfig with GL_AGENT_REGISTRY_ env prefix
3. Create `greenlang/agent_registry/models.py` - Pydantic v2 models
4. Create `greenlang/agent_registry/registry.py` - AgentRegistry with CRUD, indexing, versioning
5. Create `greenlang/agent_registry/health_checker.py` - HealthChecker with probing and TTL refresh
6. Create `greenlang/agent_registry/dependency_resolver.py` - DependencyResolver with topological sort
7. Create `greenlang/agent_registry/capability_matcher.py` - CapabilityMatcher for discovery
8. Create `greenlang/agent_registry/provenance.py` - ProvenanceTracker
9. Create `greenlang/agent_registry/metrics.py` - 12 Prometheus metrics
10. Create `greenlang/agent_registry/api/router.py` - FastAPI router with 20 endpoints
11. Create `greenlang/agent_registry/setup.py` - AgentRegistryService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V027__agent_registry_service.sql`
2. Create K8s manifests in `deployment/kubernetes/agent-registry-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests

## 7. Success Criteria
- Integration module provides clean SDK for all registry operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V027 database migration for persistent registry storage
- 20 REST API endpoints operational
- 500+ tests passing
- Complete audit trail for every registry mutation
- Health probing with configurable intervals
- Hot-reload without service interruption
- GLIP v1 agent registration with container specs

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-001 Orchestrator**: Uses registry for agent discovery in DAG execution
- **AGENT-FOUND-006 Access Guard**: Authorization for registry operations

### 8.2 Downstream Consumers
- **All agents (001-006+)**: Must register on startup
- **Orchestrator DAG execution**: Queries registry for pipeline construction
- **Admin dashboard**: Service catalog visualization
- **Health monitoring**: Proactive health checks

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent registry and audit storage (V027 migration)
- **Redis**: Registry caching, health status
- **Prometheus**: 12 observability metrics
- **Grafana**: Agent registry dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
