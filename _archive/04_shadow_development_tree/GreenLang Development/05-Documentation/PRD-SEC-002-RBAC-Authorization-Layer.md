# PRD: SEC-002 - RBAC Authorization Layer

**Document Version:** 1.0
**Date:** February 5, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Security Team
**Ralphy Task ID:** SEC-002
**Depends On:** SEC-001 (JWT Authentication Service)

---

## Executive Summary

Build a production-ready RBAC (Role-Based Access Control) authorization layer that bridges the gap between the extensive existing auth library code (`greenlang/auth/`, 12+ files, ~5,000+ lines) and actual database-backed, API-managed, runtime-enforced authorization. The library modules (RBACManager, PermissionEvaluator, RoleHierarchy, ABAC engine) are well-implemented but operate **entirely in-memory** with no database persistence, no REST management API, no role seeding, and no Redis-backed caching for permission lookups. SEC-002 creates the service layer, database schema, management API, and integration plumbing to make authorization production-ready.

### Current State
- RBACManager (`greenlang/auth/rbac.py`, 588 lines): role hierarchy, 6 built-in roles, wildcard matching -- IN-MEMORY ONLY
- PermissionEvaluator (`greenlang/auth/permissions.py`, 857 lines): ABAC conditions, caching, deny-wins -- IN-MEMORY ONLY
- RoleManager (`greenlang/auth/roles.py`, 879 lines): role hierarchy, temporal assignments, 10 built-in roles -- IN-MEMORY ONLY
- ABACEvaluator (`greenlang/auth/abac.py`, 753 lines): policy engine, attribute providers -- IN-MEMORY ONLY
- SQLAlchemy models (`greenlang/db/models_auth.py`, 467 lines): User, Role, Permission, UserRole, Session, APIKey, AuditLog -- MODELS EXIST, NO MIGRATION
- PostgreSQL backend (`greenlang/auth/backends/postgresql.py`, ~500 lines): ORM models for persistence -- PARTIAL, NO CRUD
- Agent RBAC (`greenlang/governance/policy/agent_rbac.py`, ~250 lines): 4 agent roles, 8 permissions -- IN-MEMORY ONLY
- Route protector (`greenlang/infrastructure/auth_service/route_protector.py`): 43 permission mappings, `protect_router()` -- APPLIED TO ROUTES (SEC-001)
- Kong ACL: 4 static groups (admin, standard, readonly, agent-executor) -- WORKING but STATIC
- Database migration V009: auth tables (token_blacklist, refresh_tokens, password_history, login_attempts) -- NO RBAC TABLES
- Role assignment: NO API, NO database queries, NO runtime assignment
- Permission evaluation: NO database-backed lookup, NO Redis cache for authorization decisions

### Target State
- Database migration V010: roles, permissions, user_roles, role_hierarchy tables in `security` schema
- RBAC service layer: `greenlang/infrastructure/rbac_service/` bridging library code to database
- Database-backed RoleManager: CRUD roles, assign/revoke user roles, persist to PostgreSQL
- Database-backed PermissionEvaluator: load permissions from DB, evaluate with Redis cache
- REST API for RBAC management: `/api/v1/rbac/roles`, `/api/v1/rbac/permissions`, `/api/v1/rbac/assignments`
- Default role seeding: 6 system roles + 43 permissions seeded in migration
- Redis-cached permission lookups: L1 Redis (5-minute TTL) + L2 PostgreSQL
- RBAC audit trail: all role/permission changes logged to Loki (INFRA-009)
- RBAC metrics: Prometheus counters for authorization decisions
- Cache invalidation: Redis pub/sub on role/permission changes
- Tenant-scoped RBAC: roles and assignments scoped to tenant_id
- 85%+ test coverage

---

## Scope

### In Scope
1. RBAC Service: `greenlang/infrastructure/rbac_service/` (new service module)
2. Database migration V010: roles, permissions, user_roles, role_hierarchy tables
3. Default role seeding: system roles (super_admin, admin, manager, operator, analyst, viewer, auditor, developer, service_account, guest) + standard permissions
4. Database-backed RoleService: CRUD roles with hierarchy, wrapping existing `RoleManager`
5. Database-backed PermissionService: CRUD permissions, evaluation with DB + Redis caching
6. User role assignment/revocation with temporal support (expires_at)
7. REST API: role CRUD, permission CRUD, assignment management, permission checking
8. Redis-cached authorization: L1 Redis cache with TTL + cache invalidation via pub/sub
9. RBAC audit logging: role changes, permission changes, assignment changes to Loki
10. RBAC Prometheus metrics: authorization decisions, cache hits, role counts
11. Integration with SEC-001: AuthContext enrichment with DB-loaded roles/permissions
12. Tenant-scoped RBAC: all roles/permissions/assignments scoped by tenant_id
13. K8s deployment manifests and monitoring dashboard
14. Comprehensive test suite (200+ tests)

### Out of Scope
- ABAC policy management API (SEC-003 or future)
- OPA (Open Policy Agent) integration (SEC-003 or future)
- Delegation management API (SEC-005 or future)
- User management CRUD (separate concern from RBAC)
- SSO/SAML/LDAP role mapping (SEC-007 or future)
- UI for role management (frontend concern)

---

## Architecture

### Component Architecture

```
greenlang/infrastructure/rbac_service/
|-- __init__.py                          # Public API exports
|-- role_service.py                      # Role CRUD + hierarchy (wraps greenlang.auth.roles)
|-- permission_service.py                # Permission CRUD + evaluation (wraps greenlang.auth.permissions)
|-- assignment_service.py                # User-role assignment + revocation
|-- rbac_cache.py                        # Redis-backed permission cache with invalidation
|-- rbac_audit.py                        # RBAC change audit logging for Loki
|-- rbac_metrics.py                      # Prometheus metrics for authorization decisions
|-- rbac_seeder.py                       # Default role/permission seeding
|-- api/
|   |-- __init__.py
|   |-- roles_routes.py                  # /api/v1/rbac/roles/* endpoints
|   |-- permissions_routes.py            # /api/v1/rbac/permissions/* endpoints
|   |-- assignments_routes.py            # /api/v1/rbac/assignments/* endpoints
|   |-- check_routes.py                  # /api/v1/rbac/check permission evaluation endpoint

deployment/
|-- database/
|   |-- migrations/
|       |-- sql/
|           |-- V010__rbac_authorization.sql  # RBAC tables + seeding
|-- kubernetes/
|   |-- rbac-service/                    # K8s manifests (reuses auth-service namespace)
|-- monitoring/
|   |-- dashboards/
|   |   |-- rbac-service.json            # Grafana dashboard
|   |-- alerts/
|       |-- rbac-service-alerts.yaml     # Prometheus alerts

tests/
|-- unit/
|   |-- rbac_service/
|       |-- test_role_service.py
|       |-- test_permission_service.py
|       |-- test_assignment_service.py
|       |-- test_rbac_cache.py
|       |-- test_rbac_audit.py
|       |-- test_roles_routes.py
|       |-- test_assignments_routes.py
|-- integration/
|   |-- rbac_service/
|       |-- test_rbac_flow_e2e.py
|       |-- test_rbac_cache_integration.py
|-- load/
    |-- rbac_service/
        |-- test_rbac_throughput.py
```

### Authorization Flow

```
1. Request arrives at FastAPI endpoint:
   Client --> GET /api/v1/agents (Authorization: Bearer <token>)

2. AuthenticationMiddleware (SEC-001) validates JWT:
   --> Extracts user_id, tenant_id from JWT claims
   --> Sets request.state.auth = AuthContext(user_id, tenant_id, ...)

3. AuthContext enrichment (SEC-002 -- NEW):
   --> RBACService.get_user_permissions(user_id, tenant_id)
   --> Check Redis cache first (key: gl:rbac:perms:{tenant_id}:{user_id})
   --> Cache miss: query PostgreSQL security.user_roles + security.role_permissions
   --> Aggregate permissions from all assigned roles (with hierarchy)
   --> Populate AuthContext.roles and AuthContext.permissions
   --> Cache result in Redis with 5-minute TTL

4. PermissionDependency (SEC-001) enforces:
   --> @require_permissions("agents:list")
   --> Checks AuthContext.permissions for match (wildcard-aware)
   --> 403 Forbidden if denied

5. Route handler executes with guaranteed authorization context.
```

### Cache Architecture

```
Permission Lookup:
  1. Check Redis HASH "gl:rbac:perms:{tenant_id}:{user_id}" --> O(1)
  2. If miss, query PostgreSQL:
     security.user_roles JOIN security.roles JOIN security.role_permissions
  3. Cache result in Redis with TTL = 300s (5 min)
  4. Return aggregated permissions list

Cache Invalidation:
  On role/permission change:
  1. Update PostgreSQL
  2. PUBLISH to Redis channel "gl:rbac:invalidate"
     payload: { "type": "role_change"|"permission_change"|"assignment_change",
                "tenant_id": "...", "user_id": "..." (optional) }
  3. Subscribers clear matching cache keys
  4. Next permission check triggers fresh DB lookup
```

---

## Technical Requirements

### TR-001: Database Schema (V010)

RBAC tables in `security` schema with RLS for tenant isolation.

**Tables:**

```sql
-- 1. Roles table
CREATE TABLE security.roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,                                    -- NULL for system roles
    name VARCHAR(128) NOT NULL,
    display_name VARCHAR(256),
    description TEXT,
    parent_role_id UUID REFERENCES security.roles(id),
    is_system_role BOOLEAN NOT NULL DEFAULT false,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_role_name_tenant UNIQUE (tenant_id, name)
);

-- 2. Permissions table
CREATE TABLE security.permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource VARCHAR(256) NOT NULL,                    -- e.g., "agents", "emissions"
    action VARCHAR(128) NOT NULL,                      -- e.g., "list", "read", "execute"
    description TEXT,
    is_system_permission BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_permission UNIQUE (resource, action)
);

-- 3. Role-Permission mapping
CREATE TABLE security.role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id UUID NOT NULL REFERENCES security.roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES security.permissions(id) ON DELETE CASCADE,
    effect VARCHAR(8) NOT NULL DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')),
    conditions JSONB DEFAULT '{}',
    scope VARCHAR(256),                                -- optional scope restriction
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    granted_by VARCHAR(128),
    CONSTRAINT uq_role_permission UNIQUE (role_id, permission_id)
);

-- 4. User-Role assignments
CREATE TABLE security.user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    role_id UUID NOT NULL REFERENCES security.roles(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assigned_by VARCHAR(128),
    expires_at TIMESTAMPTZ,                            -- NULL = never expires
    revoked_at TIMESTAMPTZ,
    revoked_by VARCHAR(128),
    is_active BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT uq_user_role_tenant UNIQUE (user_id, role_id, tenant_id)
);

-- 5. RBAC audit log
CREATE TABLE security.rbac_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    actor_id UUID NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    target_type VARCHAR(64) NOT NULL,                  -- "role", "permission", "assignment"
    target_id UUID NOT NULL,
    action VARCHAR(64) NOT NULL,                       -- "create", "update", "delete", "assign", "revoke"
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    correlation_id UUID,
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Requirements:**
1. All tables in `security` schema with RLS policies for tenant isolation
2. System roles have `tenant_id = NULL` (visible to all tenants)
3. Self-referencing FK on `roles.parent_role_id` for hierarchy
4. Temporal assignments via `expires_at` on `user_roles`
5. Soft-delete via `is_active` + `revoked_at` on `user_roles`
6. RBAC audit log for all changes (separate from auth audit in V009)
7. Proper indexes on all FK columns and frequently queried fields

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Max Roles per Tenant | 50 | 100 | 200 |
| Max Permissions | 200 | 500 | 1000 |
| Max Hierarchy Depth | 5 | 5 | 5 |
| Cache TTL | 60s | 180s | 300s |

### TR-002: Default Role & Permission Seeding

Seed system roles and standard permissions in V010 migration.

**System Roles:**

| Role | Parent | Description |
|---|---|---|
| `super_admin` | None | Full platform access, all tenants |
| `admin` | None | Tenant administrator |
| `manager` | None | Team manager with elevated access |
| `developer` | None | Development and configuration access |
| `operator` | None | Operational execution access |
| `analyst` | None | Read + calculate access |
| `viewer` | None | Read-only access |
| `auditor` | None | Audit log and compliance report access |
| `service_account` | None | Machine-to-machine access |
| `guest` | None | Minimal read-only access |

**Standard Permissions (43 from PERMISSION_MAP + management):**

| Resource | Actions |
|---|---|
| `agents` | list, read, execute, configure, create, delete |
| `emissions` | list, read, calculate |
| `jobs` | list, read, cancel |
| `compliance` | list, read, create |
| `factory` | list, read, create, update, delete, execute, deploy, rollback, metrics |
| `flags` | list, read, create, update, delete, evaluate, rollout, kill, restore |
| `admin` | users:list, users:read, users:unlock, users:revoke, users:reset, users:mfa |
| `admin` | sessions:list, sessions:terminate, audit:read, lockouts:list |
| `rbac` | roles:list, roles:read, roles:create, roles:update, roles:delete |
| `rbac` | permissions:list, assignments:list, assignments:create, assignments:revoke |
| `rbac` | check |

### TR-003: Role Service

Database-backed role management wrapping existing `greenlang.auth.roles.RoleManager`.

**Requirements:**
1. CRUD operations for roles (create, read, update, delete)
2. Role hierarchy management with cycle detection
3. Max hierarchy depth of 5 levels enforced
4. System role protection (cannot delete/modify system roles)
5. Tenant-scoped queries (non-system roles filtered by tenant_id)
6. Role enable/disable without deletion
7. Role metadata storage (JSON)

### TR-004: Permission Service

Database-backed permission management wrapping existing `greenlang.auth.permissions.PermissionEvaluator`.

**Requirements:**
1. CRUD operations for permissions
2. System permission protection (cannot delete system permissions)
3. Role-permission association management (grant/revoke)
4. Permission evaluation with deny-wins conflict resolution
5. Wildcard-aware matching (e.g., `agents:*` grants `agents:list`)
6. Scope-restricted permissions (e.g., `agents:execute` only in tenant X)
7. Condition-based permissions (ABAC-style JSONB conditions)

### TR-005: Assignment Service

User-role assignment management with temporal support.

**Requirements:**
1. Assign role to user with optional expiry
2. Revoke role from user (soft-delete)
3. List user roles (with hierarchy expansion)
4. Get aggregated permissions for user (all roles + inherited)
5. Temporal assignment: auto-expire roles after `expires_at`
6. Bulk assignment (assign role to multiple users)
7. Assignment audit trail (who assigned, when, why)

### TR-006: RBAC Cache

Redis-backed permission cache with pub/sub invalidation.

**Requirements:**
1. Cache key format: `gl:rbac:perms:{tenant_id}:{user_id}`
2. Cache value: JSON array of permission strings
3. TTL: 300 seconds (production), configurable per environment
4. Cache invalidation via Redis pub/sub channel `gl:rbac:invalidate`
5. Selective invalidation: clear specific user, tenant, or all
6. Cache warming on service startup (optional)
7. Fallback to PostgreSQL on Redis failure

### TR-007: REST API

Management endpoints for RBAC administration.

**Endpoints:**

| Method | Path | Description | Required Permission |
|---|---|---|---|
| `GET` | `/api/v1/rbac/roles` | List roles (paginated) | `rbac:roles:list` |
| `POST` | `/api/v1/rbac/roles` | Create role | `rbac:roles:create` |
| `GET` | `/api/v1/rbac/roles/{role_id}` | Get role details | `rbac:roles:read` |
| `PUT` | `/api/v1/rbac/roles/{role_id}` | Update role | `rbac:roles:update` |
| `DELETE` | `/api/v1/rbac/roles/{role_id}` | Delete role | `rbac:roles:delete` |
| `GET` | `/api/v1/rbac/roles/{role_id}/permissions` | List role permissions | `rbac:roles:read` |
| `POST` | `/api/v1/rbac/roles/{role_id}/permissions` | Grant permission to role | `rbac:permissions:grant` |
| `DELETE` | `/api/v1/rbac/roles/{role_id}/permissions/{perm_id}` | Revoke permission from role | `rbac:permissions:revoke` |
| `GET` | `/api/v1/rbac/permissions` | List all permissions | `rbac:permissions:list` |
| `GET` | `/api/v1/rbac/assignments` | List role assignments | `rbac:assignments:list` |
| `POST` | `/api/v1/rbac/assignments` | Assign role to user | `rbac:assignments:create` |
| `DELETE` | `/api/v1/rbac/assignments/{assignment_id}` | Revoke assignment | `rbac:assignments:revoke` |
| `GET` | `/api/v1/rbac/users/{user_id}/roles` | Get user roles | `rbac:assignments:list` |
| `GET` | `/api/v1/rbac/users/{user_id}/permissions` | Get user effective permissions | `rbac:check` |
| `POST` | `/api/v1/rbac/check` | Check if user has permission | `rbac:check` |

### TR-008: RBAC Audit & Metrics

**Audit Events:**
- `role_created`, `role_updated`, `role_deleted`, `role_enabled`, `role_disabled`
- `permission_granted`, `permission_revoked`
- `role_assigned`, `role_revoked`, `role_expired`
- `authorization_allowed`, `authorization_denied`
- `cache_invalidated`

**Prometheus Metrics:**

| Metric | Type | Labels |
|---|---|---|
| `gl_rbac_authorization_total` | Counter | `result`, `resource`, `action`, `tenant_id` |
| `gl_rbac_authorization_duration_seconds` | Histogram | `result`, `cache_hit` |
| `gl_rbac_cache_hits_total` | Counter | `tenant_id` |
| `gl_rbac_cache_misses_total` | Counter | `tenant_id` |
| `gl_rbac_roles_total` | Gauge | `tenant_id`, `is_system` |
| `gl_rbac_assignments_total` | Gauge | `tenant_id` |
| `gl_rbac_role_changes_total` | Counter | `action`, `tenant_id` |

---

## Integration Points

### Integration with SEC-001 (JWT Authentication)
- AuthContext enrichment: after JWT validation, load user roles/permissions from DB
- `configure_auth(app)` extended to also register RBAC middleware
- AuthContext.roles and AuthContext.permissions populated from database

### Integration with INFRA-002 (PostgreSQL)
- V010 migration creates RBAC tables in `security` schema
- Async queries via `psycopg` + `psycopg_pool`

### Integration with INFRA-003 (Redis)
- Permission cache: `gl:rbac:perms:{tenant_id}:{user_id}`
- Cache invalidation pub/sub: `gl:rbac:invalidate` channel

### Integration with INFRA-006 (Kong API Gateway)
- RBAC management endpoints exposed at `/api/v1/rbac/*`
- Kong ACL groups map to RBAC roles

### Integration with INFRA-009 (Log Aggregation)
- RBAC audit events logged with `event_category=rbac` label
- Loki retention: 365 days for RBAC audit events

### Integration with INFRA-010 (Agent Factory)
- Agent-specific permissions: `factory:deploy`, `factory:execute`
- Agent role assignments for machine-to-machine auth

---

## Acceptance Criteria

1. V010 migration creates 5 RBAC tables and seeds 10 system roles + 60+ permissions
2. Role CRUD API allows creating custom roles with permission inheritance
3. Permission evaluation loads from database, caches in Redis, evaluates in < 2ms (cache hit)
4. User role assignment persists to database with audit trail
5. Temporal assignments automatically expire (enforced at query time)
6. Role hierarchy limited to 5 levels with cycle detection
7. Cache invalidation propagates within 1 second of role/permission change
8. All RBAC changes logged as structured JSON to Loki with correlation IDs
9. System roles cannot be deleted or disabled
10. Tenant isolation: users can only see/manage roles within their tenant
11. All RBAC management endpoints require `rbac:*` permissions
12. AuthContext enrichment populates roles/permissions from database
13. Permission evaluation follows deny-wins conflict resolution
14. Redis failure falls back to PostgreSQL-only lookups
15. 85%+ test coverage on all new code

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| SEC-001: JWT Authentication | COMPLETE | Auth middleware, route protection, AuthContext |
| INFRA-002: PostgreSQL | COMPLETE | Security schema, async queries |
| INFRA-003: Redis | COMPLETE | Permission cache, pub/sub invalidation |
| INFRA-006: Kong API Gateway | COMPLETE | RBAC route exposure |
| INFRA-009: Log Aggregation | COMPLETE | RBAC audit to Loki |
| greenlang/auth/roles.py | EXISTS | RoleManager, RoleHierarchy (in-memory) |
| greenlang/auth/permissions.py | EXISTS | PermissionEvaluator (in-memory) |
| greenlang/auth/rbac.py | EXISTS | RBACManager (in-memory) |
| greenlang/db/models_auth.py | EXISTS | SQLAlchemy models (ORM definitions) |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Redis cache stale after role change | Medium | High | Pub/sub invalidation + short TTL (5 min) |
| Permission evaluation latency on cold cache | Low | Medium | Warm cache on startup, async prefetch |
| Circular role hierarchy | Low | Critical | Cycle detection in RoleService, max depth = 5 |
| Tenant data leakage via RBAC API | Medium | Critical | RLS policies, tenant_id enforcement in all queries |
| System role deletion breaks platform | Low | Critical | is_system_role flag prevents deletion |
| High cardinality in permission cache | Low | Medium | Per-user cache keys, TTL expiry |

---

## Development Tasks (Ralphy-Compatible)

### Phase 1: Database & Seeding
- [ ] Create `deployment/database/migrations/sql/V010__rbac_authorization.sql` - 5 tables, RLS, indexes, seeding
- [ ] Create `greenlang/infrastructure/rbac_service/rbac_seeder.py` - Role/permission seeding logic

### Phase 2: Core Services
- [ ] Create `greenlang/infrastructure/rbac_service/__init__.py` - Public API exports
- [ ] Create `greenlang/infrastructure/rbac_service/role_service.py` - Role CRUD + hierarchy
- [ ] Create `greenlang/infrastructure/rbac_service/permission_service.py` - Permission CRUD + evaluation
- [ ] Create `greenlang/infrastructure/rbac_service/assignment_service.py` - User-role assignment/revocation
- [ ] Create `greenlang/infrastructure/rbac_service/rbac_cache.py` - Redis permission cache + invalidation

### Phase 3: Audit, Metrics & API
- [ ] Create `greenlang/infrastructure/rbac_service/rbac_audit.py` - RBAC audit logging
- [ ] Create `greenlang/infrastructure/rbac_service/rbac_metrics.py` - Prometheus metrics
- [ ] Create `greenlang/infrastructure/rbac_service/api/__init__.py` - API router exports
- [ ] Create `greenlang/infrastructure/rbac_service/api/roles_routes.py` - Role management endpoints
- [ ] Create `greenlang/infrastructure/rbac_service/api/permissions_routes.py` - Permission management endpoints
- [ ] Create `greenlang/infrastructure/rbac_service/api/assignments_routes.py` - Assignment endpoints
- [ ] Create `greenlang/infrastructure/rbac_service/api/check_routes.py` - Permission check endpoint

### Phase 4: Integration
- [ ] Modify `greenlang/infrastructure/auth_service/auth_setup.py` - Register RBAC middleware + include RBAC routers
- [ ] Update `greenlang/infrastructure/auth_service/route_protector.py` - Add RBAC permission mappings

### Phase 5: Deployment & Monitoring
- [ ] Create `deployment/monitoring/dashboards/rbac-service.json` - Grafana dashboard
- [ ] Create `deployment/monitoring/alerts/rbac-service-alerts.yaml` - Prometheus alerts

### Phase 6: Testing
- [ ] Create `tests/unit/rbac_service/test_role_service.py` - 30+ tests
- [ ] Create `tests/unit/rbac_service/test_permission_service.py` - 30+ tests
- [ ] Create `tests/unit/rbac_service/test_assignment_service.py` - 25+ tests
- [ ] Create `tests/unit/rbac_service/test_rbac_cache.py` - 20+ tests
- [ ] Create `tests/unit/rbac_service/test_rbac_audit.py` - 15+ tests
- [ ] Create `tests/unit/rbac_service/test_roles_routes.py` - 25+ tests
- [ ] Create `tests/unit/rbac_service/test_assignments_routes.py` - 20+ tests
- [ ] Create `tests/integration/rbac_service/test_rbac_flow_e2e.py` - 15+ tests
- [ ] Create `tests/integration/rbac_service/test_rbac_cache_integration.py` - 10+ tests
- [ ] Create `tests/load/rbac_service/test_rbac_throughput.py` - 5+ tests
