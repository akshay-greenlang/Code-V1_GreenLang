# PRD: INFRA-008 - Feature Flags System

**Document Version:** 1.0
**Date:** February 4, 2026
**Status:** READY FOR EXECUTION
**Priority:** P1 - HIGH
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-008

---

## Executive Summary

Deploy a production-ready, distributed feature flags system for GreenLang Climate OS that enables controlled rollouts, A/B testing, tenant-specific feature gating, kill switches, and compliance-safe flag management across all 47+ agents and microservices. The system extends the existing `FeatureFlagService` (765 lines, GL-Agent-Factory) into a platform-wide infrastructure component with Redis-backed distributed state, PostgreSQL audit trails, REST API management, and deep integration with the ExecutionContext, Kong API Gateway, and CI/CD pipelines.

### Current State
- Feature flag service exists at `applications/GL-Agent-Factory/backend/services/feature_flags.py` (765 lines)
- Supports 6 flag types: Boolean, Percentage, User List, Environment, Segment, Scheduled
- Storage backends: InMemoryFlagStorage, FileFlagStorage only
- 8 pre-configured flags for GreenLang-specific features
- `ExecutionContext` has `features: Dict[str, bool]` field with `is_feature_enabled()` method
- No distributed state (single-instance only)
- No REST API for flag management
- No database persistence or audit trail
- No admin dashboard or UI
- No monitoring, alerting, or analytics
- No multi-tenant flag isolation
- No A/B testing variant tracking
- No integration with Kong API Gateway or CI/CD

### Target State
- Distributed feature flags with Redis L1 cache + PostgreSQL L2 persistence
- Sub-millisecond flag evaluation (<1ms P99) via local cache + Redis
- REST API for full CRUD operations, bulk management, and evaluation
- Multi-tenant flag isolation (per-tenant overrides, segment targeting)
- A/B testing with variant assignment and metric tracking
- PostgreSQL audit trail for all flag changes (SOC 2 compliance)
- Integration with ExecutionContext for per-request flag state
- Kong API Gateway integration for route-level feature gating
- FastAPI middleware for automatic flag injection
- Prometheus metrics and Grafana dashboard
- Kill switch mechanism for emergency feature disable
- Stale flag detection and lifecycle management
- OpenFeature-compatible evaluation interface

---

## Scope

### In Scope
1. Core feature flags module: `greenlang/infrastructure/feature_flags/` (Python package)
2. Redis storage backend with connection pooling and circuit breaker
3. PostgreSQL storage backend with audit trail and versioning
4. Multi-layer caching: L1 (in-memory, 30s TTL) + L2 (Redis, 5m TTL) + L3 (PostgreSQL)
5. REST API endpoints via FastAPI (CRUD, evaluation, bulk operations, audit)
6. FastAPI middleware for automatic flag injection into ExecutionContext
7. A/B testing: variant assignment, consistent hashing, metric collection
8. Multi-tenant support: tenant-specific overrides, segment-based targeting
9. Kill switch: instant flag disable via Redis pub/sub propagation (<100ms)
10. PostgreSQL migration (V007) for flag tables, audit log, variants
11. Prometheus metrics: evaluation count/latency, flag state changes, error rates
12. Grafana dashboard: feature-flags.json (12+ panels)
13. Prometheus alert rules: feature-flags-alerts.yaml (10+ rules)
14. Database migration: `deployment/database/migrations/sql/V007__feature_flags.sql`
15. Helm chart integration: feature flag ConfigMap for infrastructure-level flags
16. Kong integration: custom plugin for flag-based route gating
17. Comprehensive test suite (unit, integration, load tests)

### Out of Scope
- Frontend admin UI/dashboard (future phase - API-first approach)
- Feature flag as a service for external customers (internal only)
- Machine learning-based flag optimization
- Multi-region flag synchronization (future - leverages existing DR replication)
- GraphQL API for flags (REST only in v1)
- Feature flag billing/metering

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Flag Clients                          │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐│
│  │ Python SDK  │  │ REST API    │  │ Kong Gateway Plugin      ││
│  │ (Agents,    │  │ (External   │  │ (Route-level gating)     ││
│  │  Pipelines) │  │  Services)  │  │                          ││
│  └──────┬──────┘  └──────┬──────┘  └────────────┬─────────────┘│
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Feature Flag Evaluation Engine                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Evaluation Pipeline                                       │  │
│  │  1. Check kill switch cache     (instant, <0.1ms)         │  │
│  │  2. Check local override        (testing only)            │  │
│  │  3. Check L1 in-memory cache    (30s TTL, <0.01ms)       │  │
│  │  4. Check L2 Redis cache        (5m TTL, <0.5ms)         │  │
│  │  5. Query L3 PostgreSQL         (source of truth, <5ms)  │  │
│  │  6. Apply targeting rules       (user/tenant/segment)     │  │
│  │  7. Evaluate percentage/variant (consistent hashing)      │  │
│  │  8. Record metrics              (async, non-blocking)     │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────┐  ┌─────────────────┐  ┌──────────────────────────┐
│ L1: Memory  │  │ L2: Redis       │  │ L3: PostgreSQL           │
│ (30s TTL)   │  │ (5m TTL)        │  │ (Source of Truth)        │
│ Per-process │  │ Shared cluster  │  │ Audit trail + versioning │
│ <0.01ms     │  │ <0.5ms          │  │ <5ms                     │
└─────────────┘  └─────────────────┘  └──────────────────────────┘
                         │
                  ┌──────┴──────┐
                  │ Redis PubSub│ (Kill switch + invalidation)
                  │ Channel:    │
                  │ ff:updates  │
                  └─────────────┘
```

### Component Architecture

```
greenlang/infrastructure/feature_flags/
├── __init__.py              # Public API exports
├── models.py                # Pydantic models (Flag, Rule, Variant, Context)
├── engine.py                # Core evaluation engine
├── storage/
│   ├── __init__.py
│   ├── base.py              # Abstract storage interface (IFlagStorage)
│   ├── memory.py            # L1 in-memory cache storage
│   ├── redis_store.py       # L2 Redis storage backend
│   ├── postgres_store.py    # L3 PostgreSQL storage backend
│   └── multi_layer.py       # Multi-layer cache orchestrator
├── targeting/
│   ├── __init__.py
│   ├── rules.py             # Targeting rule evaluation
│   ├── segments.py          # User/tenant segment matching
│   └── percentage.py        # Consistent hashing for rollouts
├── api/
│   ├── __init__.py
│   ├── router.py            # FastAPI router (CRUD + evaluation endpoints)
│   ├── middleware.py         # Request middleware for flag injection
│   └── schemas.py           # API request/response schemas
├── analytics/
│   ├── __init__.py
│   ├── collector.py         # Async metrics collector
│   └── metrics.py           # Prometheus metric definitions
├── lifecycle/
│   ├── __init__.py
│   ├── manager.py           # Flag lifecycle management
│   └── stale_detector.py    # Stale flag detection
├── kill_switch.py           # Emergency kill switch via Redis pub/sub
├── service.py               # High-level FeatureFlagService facade
└── config.py                # Configuration and environment settings
```

### Data Flow

```
Request arrives at Kong Gateway
    │
    ├── Kong ff-gate plugin checks route-level flags (Redis direct)
    │
    ▼
FastAPI Middleware (FeatureFlagMiddleware)
    │
    ├── Extract user_id, tenant_id, environment from JWT/headers
    ├── Build EvaluationContext
    ├── Evaluate all active flags for context
    ├── Inject flag results into ExecutionContext.features
    │
    ▼
Agent/Pipeline Execution
    │
    ├── context.is_feature_enabled("new_calculation_engine")
    ├── Reads from pre-populated ExecutionContext.features
    ├── No additional storage calls needed (already resolved)
    │
    ▼
Response + Metrics
    │
    ├── Flag evaluation metrics emitted to Prometheus
    ├── A/B variant assignments recorded
    └── Audit events logged to PostgreSQL (async)
```

---

## Technical Requirements

### TR-001: Core Evaluation Engine
- Evaluate flags in <1ms P99 (L1 cache hit) and <5ms P99 (L3 miss)
- Support 6 flag types: Boolean, Percentage, UserList, Environment, Segment, Scheduled
- Support multi-variant flags (A/B/C testing with string variants)
- Consistent hashing for deterministic percentage rollouts (MD5-based, keyed on `{flag_key}:{user_id}`)
- Thread-safe evaluation with asyncio support
- Evaluation order: kill_switch → override → L1_cache → L2_redis → L3_postgres → rules → default
- Support for flag dependencies (flag A requires flag B enabled)
- Maximum 1000 active flags per environment

### TR-002: Multi-Layer Storage
- **L1 (Memory):** Per-process dict with 30-second TTL, max 1000 entries, LRU eviction
- **L2 (Redis):** Shared across instances, 5-minute TTL, key pattern `ff:{env}:{flag_key}`
  - Connection pooling via existing `greenlang/utilities/cache/redis_client.py` patterns
  - Circuit breaker (5 failures → open for 60s)
  - Graceful degradation to L3 on Redis failure
- **L3 (PostgreSQL):** Source of truth, versioned flag definitions with full audit trail
  - Async via existing `psycopg` + `psycopg_pool` patterns
  - Read replicas for evaluation queries
  - Writer endpoint for flag management operations
- **Cache Invalidation:** Redis pub/sub channel `ff:updates` for cross-instance cache busting

### TR-003: Multi-Tenant Support
- Tenant-specific flag overrides (tenant A gets flag X enabled, tenant B disabled)
- Tenant segment targeting (segment: "enterprise", "startup", "government")
- Flag inheritance: global defaults → environment overrides → tenant overrides
- Tenant isolation in storage: PostgreSQL RLS policies, Redis key namespacing
- Maximum 100 tenant overrides per flag

### TR-004: A/B Testing & Variants
- Multi-variant flags: control + up to 10 treatment variants
- Consistent variant assignment via hashing (`{flag_key}:{variant_salt}:{user_id}`)
- Variant weight distribution (e.g., control: 50%, variant_a: 25%, variant_b: 25%)
- Variant assignment logging for analytics pipeline
- Sticky sessions: once assigned, variant persists until flag reset
- Experiment metadata: hypothesis, start_date, end_date, success_metrics

### TR-005: REST API
- Base path: `/api/v1/flags`
- Endpoints:
  - `GET /api/v1/flags` - List all flags (paginated, filterable by status/type/tag)
  - `POST /api/v1/flags` - Create flag
  - `GET /api/v1/flags/{key}` - Get flag details
  - `PUT /api/v1/flags/{key}` - Update flag
  - `DELETE /api/v1/flags/{key}` - Archive flag (soft delete)
  - `POST /api/v1/flags/{key}/evaluate` - Evaluate flag for context
  - `POST /api/v1/flags/evaluate-batch` - Evaluate multiple flags
  - `PUT /api/v1/flags/{key}/rollout` - Set rollout percentage
  - `POST /api/v1/flags/{key}/kill` - Activate kill switch
  - `GET /api/v1/flags/{key}/audit` - Get audit trail
  - `GET /api/v1/flags/{key}/metrics` - Get flag metrics
  - `POST /api/v1/flags/{key}/variants` - Configure A/B variants
  - `GET /api/v1/flags/stale` - List stale flags (no evaluations in 30 days)
- Authentication: JWT via Kong (admin group for write operations)
- Rate limiting: 100 req/min for management, 10000 req/min for evaluation

### TR-006: Kill Switch
- Instant flag disable propagated via Redis pub/sub (<100ms across all instances)
- Kill switch state cached in L1 memory (bypasses all other evaluation)
- Kill switch activation logged as critical audit event
- Kill switch can be activated via API or CLI
- Auto-rollback: optional timer to re-enable after N minutes
- Dedicated Redis channel: `ff:killswitch`

### TR-007: Lifecycle Management
- Flag states: `draft` → `active` → `rolled_out` → `permanent` → `archived`
- Stale flag detection: flags with zero evaluations in 30 days flagged for review
- Automatic archival recommendations for flags at 100% rollout for >14 days
- Flag ownership: every flag must have an owner (team or individual)
- Tag-based organization: regulatory, performance, beta, experiment, infrastructure
- Maximum flag age alerts (>90 days at partial rollout)

### TR-008: Database Schema (PostgreSQL)
- Table: `infrastructure.feature_flags` - Flag definitions
- Table: `infrastructure.feature_flag_rules` - Targeting rules
- Table: `infrastructure.feature_flag_variants` - A/B variants
- Table: `infrastructure.feature_flag_overrides` - Tenant/user overrides
- Table: `infrastructure.feature_flag_audit_log` - Change audit trail
- Table: `infrastructure.feature_flag_evaluations` - Evaluation metrics (TimescaleDB hypertable)
- Indexes: B-tree on flag_key, environment; GIN on tags; partial on status=active
- Row-Level Security: tenant isolation for multi-tenant queries

### TR-009: Monitoring & Alerting
- Prometheus metrics:
  - `ff_evaluation_total` (counter): flag_key, result, environment, tenant
  - `ff_evaluation_duration_seconds` (histogram): flag_key, cache_layer
  - `ff_flag_state` (gauge): flag_key, status (active/inactive/archived)
  - `ff_cache_hit_total` (counter): layer (l1/l2/l3)
  - `ff_cache_miss_total` (counter): layer
  - `ff_kill_switch_active` (gauge): flag_key
  - `ff_stale_flags_total` (gauge): environment
  - `ff_storage_errors_total` (counter): backend (redis/postgres)
- Grafana dashboard: 12+ panels covering evaluation rates, latency, cache hit rates, flag distribution
- Alert rules (10+):
  - FeatureFlagEvaluationErrorRate > 1% for 5m (critical)
  - FeatureFlagRedisDown for 1m (critical)
  - FeatureFlagHighLatency P99 > 5ms for 10m (warning)
  - FeatureFlagKillSwitchActive (info, immediate)
  - FeatureFlagStaleFlagsHigh > 10 (warning)
  - FeatureFlagStorageDesync for 5m (warning)
  - FeatureFlagCacheHitRateLow < 80% for 10m (warning)
  - FeatureFlagAuditLogFailure (critical)
  - FeatureFlagHighEvaluationRate > 100k/min (warning)
  - FeatureFlagRolloutStuck > 14 days at partial (info)

### TR-010: Environment Configuration

| Parameter | Dev | Staging | Prod |
|---|---|---|---|
| L1 Cache TTL | 10s | 30s | 30s |
| L2 Redis TTL | 60s | 300s | 300s |
| Max Active Flags | 100 | 500 | 1000 |
| Evaluation Logging | verbose | sampled (10%) | sampled (1%) |
| Kill Switch Enabled | true | true | true |
| Stale Detection Days | 7 | 14 | 30 |
| Audit Log Retention | 30 days | 90 days | 365 days |
| A/B Testing Enabled | true | true | true |
| Redis Backend | localhost | ElastiCache | ElastiCache |
| PostgreSQL Backend | local | Aurora Reader | Aurora Reader |
| PubSub Enabled | false | true | true |

---

## Integration Points

### Integration with ExecutionContext
```python
# In FastAPI middleware (automatic):
context = ExecutionContext(user_id=user_id, tenant_id=tenant_id)
flags = await flag_service.evaluate_all(context)
context.features = flags  # Dict[str, bool]

# In agent code (transparent):
if context.is_feature_enabled("new_calculation_engine"):
    result = await new_engine.calculate(data)
else:
    result = await legacy_engine.calculate(data)
```

### Integration with Kong API Gateway
```lua
-- Kong plugin: gl-feature-gate
-- Checks Redis directly for route-level flags
local flag_key = "route:" .. kong.router.get_route().name
local enabled = redis:get("ff:prod:" .. flag_key)
if enabled == "false" then
    return kong.response.exit(503, { message = "Feature temporarily unavailable" })
end
```

### Integration with CI/CD (GitHub Actions)
```yaml
# In deployment workflow
- name: Check feature flags
  run: |
    # Verify no kill switches active before deploy
    curl -s $FF_API/flags?status=killed | jq '.count == 0'
    # Verify no stale flags blocking deploy
    curl -s $FF_API/flags/stale | jq '.count < 10'
```

---

## File Structure

```
greenlang/infrastructure/feature_flags/
├── __init__.py
├── models.py
├── engine.py
├── service.py
├── config.py
├── kill_switch.py
├── storage/
│   ├── __init__.py
│   ├── base.py
│   ├── memory.py
│   ├── redis_store.py
│   ├── postgres_store.py
│   └── multi_layer.py
├── targeting/
│   ├── __init__.py
│   ├── rules.py
│   ├── segments.py
│   └── percentage.py
├── api/
│   ├── __init__.py
│   ├── router.py
│   ├── middleware.py
│   └── schemas.py
├── analytics/
│   ├── __init__.py
│   ├── collector.py
│   └── metrics.py
└── lifecycle/
    ├── __init__.py
    ├── manager.py
    └── stale_detector.py

deployment/
├── database/
│   └── migrations/
│       └── sql/
│           └── V007__feature_flags.sql
├── monitoring/
│   ├── dashboards/
│   │   └── feature-flags.json
│   └── alerts/
│       └── feature-flags-alerts.yaml
├── config/
│   └── kong/
│       └── custom-plugins/
│           └── gl-feature-gate/
│               ├── handler.lua
│               └── schema.lua
└── kubernetes/
    └── feature-flags/
        └── configmap.yaml

tests/
├── unit/
│   └── test_feature_flags/
│       ├── test_engine.py
│       ├── test_models.py
│       ├── test_targeting.py
│       ├── test_storage.py
│       └── test_kill_switch.py
├── integration/
│   └── test_feature_flags_integration.py
└── load/
    └── test_feature_flags_load.py
```

---

## Database Migration: V007__feature_flags.sql

```sql
-- Schema: infrastructure (create if not exists)
CREATE SCHEMA IF NOT EXISTS infrastructure;

-- Feature Flags table
CREATE TABLE infrastructure.feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(128) NOT NULL UNIQUE,
    name VARCHAR(256) NOT NULL,
    description TEXT DEFAULT '',
    flag_type VARCHAR(32) NOT NULL DEFAULT 'boolean',
    status VARCHAR(32) NOT NULL DEFAULT 'draft',
    default_value BOOLEAN NOT NULL DEFAULT false,
    rollout_percentage DECIMAL(5,2) DEFAULT 0.00,
    environments TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    owner VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT valid_flag_type CHECK (flag_type IN ('boolean','percentage','user_list','environment','segment','scheduled','multivariate')),
    CONSTRAINT valid_status CHECK (status IN ('draft','active','rolled_out','permanent','archived','killed')),
    CONSTRAINT valid_percentage CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100)
);

-- Targeting Rules
CREATE TABLE infrastructure.feature_flag_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID NOT NULL REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,
    rule_type VARCHAR(32) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    conditions JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- A/B Variants
CREATE TABLE infrastructure.feature_flag_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID NOT NULL REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,
    variant_key VARCHAR(64) NOT NULL,
    variant_value JSONB NOT NULL DEFAULT '{}',
    weight DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    description TEXT DEFAULT '',
    UNIQUE(flag_id, variant_key)
);

-- Tenant/User Overrides
CREATE TABLE infrastructure.feature_flag_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID NOT NULL REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,
    scope_type VARCHAR(32) NOT NULL,
    scope_value VARCHAR(256) NOT NULL,
    enabled BOOLEAN NOT NULL,
    variant_key VARCHAR(64),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128),
    CONSTRAINT valid_scope CHECK (scope_type IN ('user','tenant','segment','environment')),
    UNIQUE(flag_id, scope_type, scope_value)
);

-- Audit Log
CREATE TABLE infrastructure.feature_flag_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_key VARCHAR(128) NOT NULL,
    action VARCHAR(32) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    changed_by VARCHAR(128) NOT NULL,
    change_reason TEXT,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Evaluation Metrics (TimescaleDB hypertable for time-series)
CREATE TABLE infrastructure.feature_flag_evaluations (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    flag_key VARCHAR(128) NOT NULL,
    environment VARCHAR(32) NOT NULL DEFAULT 'production',
    tenant_id VARCHAR(128),
    result BOOLEAN NOT NULL,
    variant_key VARCHAR(64),
    cache_layer VARCHAR(8),
    duration_us INTEGER,
    context_hash VARCHAR(32)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('infrastructure.feature_flag_evaluations', 'time',
    chunk_time_interval => INTERVAL '1 day');

-- Indexes
CREATE INDEX idx_ff_key_status ON infrastructure.feature_flags(key, status);
CREATE INDEX idx_ff_status ON infrastructure.feature_flags(status) WHERE status = 'active';
CREATE INDEX idx_ff_tags ON infrastructure.feature_flags USING GIN(tags);
CREATE INDEX idx_ff_owner ON infrastructure.feature_flags(owner);
CREATE INDEX idx_ff_rules_flag ON infrastructure.feature_flag_rules(flag_id);
CREATE INDEX idx_ff_variants_flag ON infrastructure.feature_flag_variants(flag_id);
CREATE INDEX idx_ff_overrides_flag ON infrastructure.feature_flag_overrides(flag_id);
CREATE INDEX idx_ff_overrides_scope ON infrastructure.feature_flag_overrides(scope_type, scope_value);
CREATE INDEX idx_ff_audit_key ON infrastructure.feature_flag_audit_log(flag_key, created_at DESC);
CREATE INDEX idx_ff_audit_time ON infrastructure.feature_flag_audit_log(created_at DESC);
CREATE INDEX idx_ff_evals_key_time ON infrastructure.feature_flag_evaluations(flag_key, time DESC);

-- Continuous Aggregate: Hourly evaluation stats
CREATE MATERIALIZED VIEW infrastructure.ff_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    flag_key,
    environment,
    COUNT(*) AS total_evaluations,
    COUNT(*) FILTER (WHERE result = true) AS enabled_count,
    COUNT(*) FILTER (WHERE result = false) AS disabled_count,
    AVG(duration_us) AS avg_duration_us,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_us) AS p99_duration_us
FROM infrastructure.feature_flag_evaluations
GROUP BY bucket, flag_key, environment;

-- Refresh policy: every 5 minutes, refresh last hour
SELECT add_continuous_aggregate_policy('infrastructure.ff_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Retention: 90 days for raw evaluations, 365 days for hourly aggregates
SELECT add_retention_policy('infrastructure.feature_flag_evaluations', INTERVAL '90 days');

-- Compression: compress after 7 days
SELECT add_compression_policy('infrastructure.feature_flag_evaluations', INTERVAL '7 days');

-- Row-Level Security for multi-tenant isolation
ALTER TABLE infrastructure.feature_flag_overrides ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON infrastructure.feature_flag_overrides
    FOR ALL
    USING (scope_type != 'tenant' OR scope_value = current_setting('app.tenant_id', true));

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION infrastructure.update_ff_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_ff_updated
    BEFORE UPDATE ON infrastructure.feature_flags
    FOR EACH ROW
    EXECUTE FUNCTION infrastructure.update_ff_timestamp();
```

---

## Migration Plan

### Phase 1: Core Infrastructure (P0)
1. Create `greenlang/infrastructure/feature_flags/` module
2. Implement models, engine, config, storage backends (memory, Redis, PostgreSQL)
3. Run database migration V007
4. Deploy with in-memory + Redis storage

### Phase 2: API & Integration (P1)
1. Deploy REST API endpoints
2. Implement FastAPI middleware for flag injection
3. Integrate with ExecutionContext
4. Migrate existing GL-Agent-Factory flags to new system

### Phase 3: Advanced Features (P2)
1. A/B testing variant support
2. Kill switch mechanism
3. Lifecycle management and stale detection
4. Kong gl-feature-gate plugin

### Phase 4: Monitoring & Operations (P2)
1. Prometheus metrics and Grafana dashboard
2. Alert rules deployment
3. Audit trail verification
4. Load testing and performance validation

---

## Acceptance Criteria

1. Flag evaluation P99 latency < 1ms with L1 cache hit, < 5ms with L3 miss
2. 1000 concurrent flag evaluations per second with zero errors
3. Kill switch propagation across all instances in < 100ms
4. REST API supports full CRUD with pagination, filtering, and audit trail
5. Multi-tenant flag isolation verified (tenant A cannot see tenant B overrides)
6. A/B variant assignment is deterministic (same user always gets same variant)
7. PostgreSQL audit log captures all flag changes with user attribution
8. Grafana dashboard shows real-time evaluation metrics
9. Stale flag detection identifies flags with no evaluations in configured period
10. All existing 8 GL-Agent-Factory flags migrated to new system
11. Zero-downtime migration from existing FeatureFlagService
12. 85%+ test coverage across all modules

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| INFRA-001: EKS Cluster | COMPLETE | Feature flags deploy on EKS |
| INFRA-002: PostgreSQL | COMPLETE | Audit trail and flag persistence |
| INFRA-003: Redis | COMPLETE | L2 cache and pub/sub |
| INFRA-004: S3 | COMPLETE | Flag config backups |
| INFRA-005: pgvector | COMPLETE | No direct dependency |
| INFRA-006: Kong Gateway | COMPLETE | Route-level feature gating |
| INFRA-007: CI/CD | COMPLETE | Flag checks in deploy pipeline |
| ExecutionContext | EXISTS | `greenlang/execution/core/context.py` |
| Existing FeatureFlagService | EXISTS | `applications/GL-Agent-Factory/backend/services/feature_flags.py` |

---

## Development Tasks (Ralphy-Compatible)

### Phase 1: Core Infrastructure
- [ ] Create module: greenlang/infrastructure/feature_flags/__init__.py
- [ ] Create models: greenlang/infrastructure/feature_flags/models.py
- [ ] Create config: greenlang/infrastructure/feature_flags/config.py
- [ ] Create engine: greenlang/infrastructure/feature_flags/engine.py
- [ ] Create service facade: greenlang/infrastructure/feature_flags/service.py
- [ ] Create storage base: greenlang/infrastructure/feature_flags/storage/base.py
- [ ] Create memory storage: greenlang/infrastructure/feature_flags/storage/memory.py
- [ ] Create Redis storage: greenlang/infrastructure/feature_flags/storage/redis_store.py
- [ ] Create PostgreSQL storage: greenlang/infrastructure/feature_flags/storage/postgres_store.py
- [ ] Create multi-layer cache: greenlang/infrastructure/feature_flags/storage/multi_layer.py
- [ ] Create targeting rules: greenlang/infrastructure/feature_flags/targeting/rules.py
- [ ] Create segment matching: greenlang/infrastructure/feature_flags/targeting/segments.py
- [ ] Create percentage rollout: greenlang/infrastructure/feature_flags/targeting/percentage.py
- [ ] Create kill switch: greenlang/infrastructure/feature_flags/kill_switch.py
- [ ] Create database migration: deployment/database/migrations/sql/V007__feature_flags.sql

### Phase 2: API & Middleware
- [ ] Create API schemas: greenlang/infrastructure/feature_flags/api/schemas.py
- [ ] Create API router: greenlang/infrastructure/feature_flags/api/router.py
- [ ] Create API middleware: greenlang/infrastructure/feature_flags/api/middleware.py

### Phase 3: Analytics & Lifecycle
- [ ] Create metrics definitions: greenlang/infrastructure/feature_flags/analytics/metrics.py
- [ ] Create metrics collector: greenlang/infrastructure/feature_flags/analytics/collector.py
- [ ] Create lifecycle manager: greenlang/infrastructure/feature_flags/lifecycle/manager.py
- [ ] Create stale detector: greenlang/infrastructure/feature_flags/lifecycle/stale_detector.py

### Phase 4: Infrastructure
- [ ] Create Grafana dashboard: deployment/monitoring/dashboards/feature-flags.json
- [ ] Create alert rules: deployment/monitoring/alerts/feature-flags-alerts.yaml
- [ ] Create Kong plugin handler: deployment/config/kong/custom-plugins/gl-feature-gate/handler.lua
- [ ] Create Kong plugin schema: deployment/config/kong/custom-plugins/gl-feature-gate/schema.lua
- [ ] Create K8s ConfigMap: deployment/kubernetes/feature-flags/configmap.yaml

### Phase 5: Tests
- [ ] Create unit tests: tests/unit/test_feature_flags/test_engine.py
- [ ] Create unit tests: tests/unit/test_feature_flags/test_models.py
- [ ] Create unit tests: tests/unit/test_feature_flags/test_targeting.py
- [ ] Create unit tests: tests/unit/test_feature_flags/test_storage.py
- [ ] Create unit tests: tests/unit/test_feature_flags/test_kill_switch.py
- [ ] Create integration tests: tests/integration/test_feature_flags_integration.py
- [ ] Create load tests: tests/load/test_feature_flags_load.py

### Phase 6: Documentation & Cleanup
- [ ] Create Ralphy task file: .ralphy/INFRA-008-tasks.md
- [ ] Update MEMORY.md with INFRA-008 status
