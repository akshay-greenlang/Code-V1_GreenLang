# SEC-002: RBAC Authorization Layer - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-05
**Completed:** 2026-02-05
**Priority:** P0 - CRITICAL
**Depends On:** SEC-001 (JWT Authentication Service)
**Result:** 29 new files + 2 modified, ~13,016 lines

---

## Phase 1: Database & Seeding (P0)

### 1.1 Database Migration
- [x] Create `deployment/database/migrations/sql/V010__rbac_authorization.sql` - 5 RBAC tables (roles, permissions, role_permissions, user_roles, rbac_audit_log) with RLS, indexes, 10 system roles, 61 permissions, default role-permission mappings, cleanup functions

### 1.2 RBAC Seeder
- [x] Create `greenlang/infrastructure/rbac_service/rbac_seeder.py` - RBACSeeder class with seed_roles(), seed_permissions(), seed_role_permissions(), seed_all(), idempotent ON CONFLICT

---

## Phase 2: Core Services (P0)

### 2.1 Package Init
- [x] Create `greenlang/infrastructure/rbac_service/__init__.py` - Public API exports (17 symbols), RBACServiceConfig dataclass

### 2.2 Role Service
- [x] Create `greenlang/infrastructure/rbac_service/role_service.py` - RoleService CRUD + hierarchy, cycle detection, system role protection, tenant scoping, 5 exception classes

### 2.3 Permission Service
- [x] Create `greenlang/infrastructure/rbac_service/permission_service.py` - PermissionService CRUD + 6-step evaluation pipeline (cache -> DB -> hierarchy -> deny-wins -> cache -> wildcard match)

### 2.4 Assignment Service
- [x] Create `greenlang/infrastructure/rbac_service/assignment_service.py` - AssignmentService assign/revoke/bulk/expire, cache invalidation, audit logging, 3 exception classes

### 2.5 RBAC Cache
- [x] Create `greenlang/infrastructure/rbac_service/rbac_cache.py` - RBACCache Redis-backed with pub/sub invalidation, graceful Redis failure handling

---

## Phase 3: Audit, Metrics & API (P1)

### 3.1 RBAC Audit Logger
- [x] Create `greenlang/infrastructure/rbac_service/rbac_audit.py` - 13 event types, structured JSON for Loki, async fire-and-forget DB writes, PII redaction

### 3.2 RBAC Metrics
- [x] Create `greenlang/infrastructure/rbac_service/rbac_metrics.py` - 7 Prometheus metrics with lazy initialization pattern

### 3.3 API Router Init
- [x] Create `greenlang/infrastructure/rbac_service/api/__init__.py` - Combined rbac_router with 4 sub-routers

### 3.4 Roles API Routes
- [x] Create `greenlang/infrastructure/rbac_service/api/roles_routes.py` - 6 endpoints (GET/POST roles, GET/PUT/DELETE roles/{id}, GET roles/{id}/permissions)

### 3.5 Permissions API Routes
- [x] Create `greenlang/infrastructure/rbac_service/api/permissions_routes.py` - 3 endpoints (GET permissions, POST/DELETE role permissions)

### 3.6 Assignments API Routes
- [x] Create `greenlang/infrastructure/rbac_service/api/assignments_routes.py` - 5 endpoints (GET/POST assignments, DELETE assignments/{id}, GET users/{id}/roles, GET users/{id}/permissions)

### 3.7 Check API Route
- [x] Create `greenlang/infrastructure/rbac_service/api/check_routes.py` - 1 endpoint (POST /api/v1/rbac/check)

---

## Phase 4: Integration (P1)

### 4.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py` - Added RBAC router inclusion, RBACEnrichmentDependency, enrich_with_rbac parameter

### 4.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py` - Added 15 RBAC permission mappings (PERMISSION_MAP now 58 entries)

---

## Phase 5: Deployment & Monitoring (P2)

### 5.1 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/rbac-service.json` - 15-panel dashboard (authorization decisions, cache hit rate, latency, roles, assignments, changes, denied permissions)

### 5.2 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/rbac-service-alerts.yaml` - 11 alert rules across 4 groups

---

## Phase 6: Testing (P2)

### 6.1 Unit Tests
- [x] Create `tests/unit/rbac_service/__init__.py`
- [x] Create `tests/unit/rbac_service/test_role_service.py` - 31 tests
- [x] Create `tests/unit/rbac_service/test_permission_service.py` - 30 tests
- [x] Create `tests/unit/rbac_service/test_assignment_service.py` - 25 tests
- [x] Create `tests/unit/rbac_service/test_rbac_cache.py` - 20 tests
- [x] Create `tests/unit/rbac_service/test_rbac_audit.py` - 15 tests
- [x] Create `tests/unit/rbac_service/test_roles_routes.py` - 25 tests
- [x] Create `tests/unit/rbac_service/test_assignments_routes.py` - 22 tests

### 6.2 Integration Tests
- [x] Create `tests/integration/rbac_service/__init__.py`
- [x] Create `tests/integration/rbac_service/test_rbac_flow_e2e.py` - 15 tests
- [x] Create `tests/integration/rbac_service/test_rbac_cache_integration.py` - 10 tests

### 6.3 Load Tests
- [x] Create `tests/load/rbac_service/__init__.py`
- [x] Create `tests/load/rbac_service/test_rbac_throughput.py` - 7 tests

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Database & Seeding | 2/2 | P0 | COMPLETE |
| Phase 2: Core Services | 5/5 | P0 | COMPLETE |
| Phase 3: Audit, Metrics & API | 7/7 | P1 | COMPLETE |
| Phase 4: Integration | 2/2 | P1 | COMPLETE |
| Phase 5: Deployment & Monitoring | 2/2 | P2 | COMPLETE |
| Phase 6: Testing | 14/14 | P2 | COMPLETE |
| **TOTAL** | **32/32** | - | **COMPLETE** |
