# PRD: SEC-001 - JWT Authentication Service

**Document Version:** 1.0
**Date:** February 5, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Security Team
**Ralphy Task ID:** SEC-001

---

## Executive Summary

Deploy a production-ready JWT Authentication Service for GreenLang Climate OS that bridges the gap between the existing auth modules (`greenlang/auth/`, 24 files, ~600KB of code) and actual application-level enforcement. The auth modules (JWT handler, RBAC, ABAC, OAuth2/OIDC, MFA, API keys, middleware) are well-implemented but **not wired into the running application**. No API routes enforce authentication, no token lifecycle endpoints exist, and no token revocation system is in production. SEC-001 consolidates these modules into a deployable authentication service with REST endpoints, middleware enforcement on all routes, token revocation, refresh token flow, audit logging, and compliance controls.

### Current State
- JWT handler (`greenlang/auth/jwt_handler.py`, 648 lines): RS256 signing, expiry, JWKS, validation — COMPLETE
- Auth manager (`greenlang/auth/auth.py`, 738 lines): user creation, password hashing, MFA, API keys — COMPLETE
- OAuth2/OIDC provider (`greenlang/auth/oauth_provider.py`, 857 lines): Authorization Code, PKCE, multi-provider — COMPLETE
- RBAC (`greenlang/auth/rbac.py`, 588 lines): role hierarchy, wildcard permissions, scopes — COMPLETE
- Fine-grained permissions (`greenlang/auth/permissions.py`, 857 lines): ALLOW/DENY with caching — COMPLETE
- Auth middleware (`greenlang/auth/middleware.py`): JWTAuthBackend, APIKeyAuthBackend, decorators — EXISTS but NOT REGISTERED
- DB models (`greenlang/db/models_auth.py`, 467 lines): 10 SQLAlchemy models — COMPLETE
- Kong JWT plugin (INFRA-006): validates signature + expiry at gateway — COMPLETE
- Kong ACL groups: admin, standard, readonly, agent-executor — COMPLETE
- Kong tenant isolation plugin: extracts tenant_id from JWT — COMPLETE
- API routes (`greenlang/execution/infrastructure/api/routes/`): NO auth decorators applied
- Token revocation: JTI concept exists, NO Redis/PostgreSQL blacklist in production
- Auth API endpoints: NONE (no /auth/login, /auth/token, /auth/refresh, /auth/revoke)
- Refresh token flow: NOT IMPLEMENTED
- Auth audit logging: NOT CONNECTED to Loki
- Password policies: min length only, no complexity/history/expiry
- Account lockout: tracking exists, enforcement NOT IMPLEMENTED

### Target State
- JWT Authentication Service with REST endpoints: `/auth/login`, `/auth/token`, `/auth/refresh`, `/auth/revoke`, `/auth/validate`, `/auth/me`
- Auth middleware registered globally on FastAPI app with `AuthContext` injection
- All existing API routes protected with `@require_auth` decorators and endpoint-specific permissions
- Token revocation via Redis JTI blacklist (L1) + PostgreSQL (L2) with cleanup job
- Refresh token rotation: long-lived refresh tokens (7d) with rotation on use
- Password policies: complexity requirements, history tracking, expiry
- Account lockout: automatic after N failed attempts, timed unlock + admin unlock
- Auth event audit logging integrated with Loki (INFRA-009)
- Defense-in-depth: Kong validates signature → app middleware validates JTI + permissions
- Service-to-service auth via dedicated JWT consumers with short-lived tokens
- JWKS endpoint for external token validation
- 85%+ test coverage on all new code
- Grafana dashboard for auth metrics (login rate, failure rate, revocations, MFA usage)

---

## Scope

### In Scope
1. Auth Service API: `greenlang/infrastructure/auth_service/` (new service module)
2. Auth REST endpoints: `/auth/login`, `/auth/token`, `/auth/refresh`, `/auth/revoke`, `/auth/validate`, `/auth/me`, `/auth/jwks`
3. Token revocation system: Redis JTI blacklist + PostgreSQL fallback + cleanup CronJob
4. Refresh token rotation: issue/validate/rotate refresh tokens with family tracking
5. Auth middleware registration: wire `AuthenticationMiddleware` into FastAPI app startup
6. Route protection: add `@require_auth`/`@require_permissions` to ALL existing API routes
7. Password policies: complexity, history (last 5), expiry (90d default), forced reset
8. Account lockout: 5 failed attempts → 15min lock, progressive backoff, admin unlock
9. Auth audit logging: all auth events → structured logs → Loki with correlation IDs
10. Auth metrics: Prometheus counters/histograms for login, validation, revocation, MFA
11. JWKS endpoint: public key distribution for external services
12. DB migration V009: auth service tables (token_blacklist, refresh_tokens, password_history, login_attempts)
13. K8s deployment: auth service Deployment, Service, HPA, NetworkPolicy
14. Monitoring dashboard: auth-specific Grafana dashboard (15+ panels)
15. Alert rules: auth-specific Prometheus alerts (10+ rules)
16. Comprehensive test suite: 200+ tests, 85%+ coverage

### Out of Scope
- WebAuthn/FIDO2 hardware key support (SEC-002 or future)
- OAuth2 Authorization Server (GreenLang as IdP — future)
- mTLS service mesh (Istio-level — SEC-004)
- Secrets management integration with HashiCorp Vault (SEC-006)
- GDPR data deletion workflows (SEC-010)
- User management admin UI (frontend concern)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JWT Authentication Service (SEC-001)                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Auth REST API                                │   │
│  │                                                                  │   │
│  │  POST /auth/login     POST /auth/token    POST /auth/refresh    │   │
│  │  POST /auth/revoke    GET  /auth/validate  GET  /auth/me        │   │
│  │  GET  /auth/jwks      POST /auth/logout                         │   │
│  └──────────────┬───────────────────────────┬──────────────────────┘   │
│                 │                           │                          │
│                 ▼                           ▼                          │
│  ┌──────────────────────────┐  ┌───────────────────────────────────┐  │
│  │   Token Service           │  │   Auth Middleware                  │  │
│  │                           │  │                                   │  │
│  │  ┌──────────────────┐    │  │  ┌─────────────────────────────┐  │  │
│  │  │ JWT Issuer        │    │  │  │ AuthenticationMiddleware     │  │  │
│  │  │ (RS256, claims,   │    │  │  │ (JWT → API Key → reject)    │  │  │
│  │  │  expiry, JTI)     │    │  │  └─────────────────────────────┘  │  │
│  │  └──────────────────┘    │  │                                   │  │
│  │  ┌──────────────────┐    │  │  ┌─────────────────────────────┐  │  │
│  │  │ Refresh Token     │    │  │  │ AuthContext Injection        │  │  │
│  │  │ Manager (rotation,│    │  │  │ (user_id, tenant_id, roles, │  │  │
│  │  │  family tracking) │    │  │  │  permissions, scopes)       │  │  │
│  │  └──────────────────┘    │  │  └─────────────────────────────┘  │  │
│  │  ┌──────────────────┐    │  │                                   │  │
│  │  │ Revocation Service│    │  │  ┌─────────────────────────────┐  │  │
│  │  │ (Redis L1 + PG L2)│    │  │  │ Permission Decorators        │  │  │
│  │  └──────────────────┘    │  │  │ @require_auth                │  │  │
│  └──────────────────────────┘  │  │ @require_roles("admin")      │  │  │
│                                 │  │ @require_permissions("r:w")  │  │  │
│  ┌──────────────────────────┐  │  │ @require_tenant              │  │  │
│  │   Security Controls       │  │  └─────────────────────────────┘  │  │
│  │                           │  └───────────────────────────────────┘  │
│  │  ┌──────────────────┐    │                                         │
│  │  │ Password Policy   │    │  ┌───────────────────────────────────┐  │
│  │  │ (complexity,      │    │  │   Audit & Observability           │  │
│  │  │  history, expiry) │    │  │                                   │  │
│  │  └──────────────────┘    │  │  ┌─────────────────────────────┐  │  │
│  │  ┌──────────────────┐    │  │  │ Auth Event Logger            │  │  │
│  │  │ Account Lockout   │    │  │  │ (login, validate, revoke,   │  │  │
│  │  │ (progressive      │    │  │  │  lockout, MFA events)       │  │  │
│  │  │  backoff, unlock) │    │  │  └─────────────────────────────┘  │  │
│  │  └──────────────────┘    │  │  ┌─────────────────────────────┐  │  │
│  │  ┌──────────────────┐    │  │  │ Auth Metrics (Prometheus)    │  │  │
│  │  │ Rate Limiter       │    │  │  │ (counters, histograms,     │  │  │
│  │  │ (login attempts,  │    │  │  │  latency, error rate)       │  │  │
│  │  │  per-IP throttle) │    │  │  └─────────────────────────────┘  │  │
│  │  └──────────────────┘    │  └───────────────────────────────────┘  │
│  └──────────────────────────┘                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌─────────────────┐ ┌────────────────┐ ┌──────────────────────────────┐
│ PostgreSQL       │ │ Redis          │ │ Loki (Auth Audit Logs)       │
│ (users, roles,   │ │ (JTI blacklist,│ │                              │
│  sessions,       │ │  refresh cache,│ │ Labels: event_type, user_id, │
│  audit_log,      │ │  rate limits,  │ │  tenant_id, result, ip      │
│  password_hist)  │ │  session store)│ │                              │
└─────────────────┘ └────────────────┘ └──────────────────────────────┘
```

### Component Architecture

```
greenlang/infrastructure/auth_service/
├── __init__.py                          # Public API exports
├── token_service.py                     # JWT access + refresh token lifecycle
├── revocation.py                        # Token revocation (Redis L1 + PostgreSQL L2)
├── refresh_tokens.py                    # Refresh token rotation with family tracking
├── password_policy.py                   # Complexity, history, expiry enforcement
├── account_lockout.py                   # Progressive lockout after failed attempts
├── auth_audit.py                        # Structured auth event logging for Loki
├── auth_metrics.py                      # Prometheus counters/histograms
├── rate_limiter.py                      # Login attempt rate limiting (per-IP, per-user)
├── jwks_endpoint.py                     # JWKS public key distribution
├── route_protector.py                   # Utility to apply auth decorators to routers
└── api/
    ├── __init__.py
    ├── auth_routes.py                   # /auth/* endpoints (login, token, refresh, revoke, etc.)
    ├── user_routes.py                   # /auth/users/* endpoints (profile, password change)
    └── admin_routes.py                  # /auth/admin/* endpoints (user mgmt, lockout mgmt)

deployment/
├── database/
│   └── migrations/
│       └── sql/
│           └── V009__auth_service.sql   # Auth service tables
├── kubernetes/
│   └── auth-service/
│       ├── namespace.yaml               # greenlang-auth namespace
│       ├── deployment.yaml              # Auth service deployment
│       ├── service.yaml                 # ClusterIP service
│       ├── hpa.yaml                     # HPA for auth service
│       ├── networkpolicy.yaml           # Network isolation
│       ├── configmap.yaml               # Auth configuration
│       ├── cronjob-token-cleanup.yaml   # JTI blacklist cleanup
│       └── kustomization.yaml           # Kustomize base
├── monitoring/
│   ├── dashboards/
│   │   └── auth-service.json            # 15+ panel Grafana dashboard
│   └── alerts/
│       └── auth-service-alerts.yaml     # 10+ Prometheus alert rules
└── helm/
    └── greenlang-agents/
        └── templates/
            └── deployment-auth-service.yaml  # Helm template

tests/
├── unit/
│   └── auth_service/
│       ├── test_token_service.py
│       ├── test_revocation.py
│       ├── test_refresh_tokens.py
│       ├── test_password_policy.py
│       ├── test_account_lockout.py
│       ├── test_auth_routes.py
│       ├── test_auth_audit.py
│       └── test_route_protector.py
├── integration/
│   └── auth_service/
│       ├── test_auth_flow_e2e.py
│       └── test_token_lifecycle.py
└── load/
    └── auth_service/
        └── test_auth_throughput.py
```

### Token Flow

```
1. Login Flow:
   Client → POST /auth/login (username, password)
        → Validate credentials (bcrypt hash comparison)
        → Check account lockout status
        → Verify MFA if enabled (TOTP code)
        → Issue access_token (RS256, 1h expiry, JTI)
        → Issue refresh_token (opaque, 7d expiry, family_id)
        → Log auth event to Loki
        → Return { access_token, refresh_token, expires_in, token_type }

2. API Request Flow:
   Client → GET /api/v1/agents (Authorization: Bearer <access_token>)
        → Kong validates JWT signature + expiry
        → Kong injects X-Tenant-ID from JWT claim
        → AuthenticationMiddleware extracts JWT
        → Check JTI against Redis blacklist (L1)
        → If miss, check PostgreSQL blacklist (L2)
        → Build AuthContext (user_id, tenant_id, roles, permissions)
        → @require_auth decorator validates auth present
        → @require_permissions("agents:list") validates permission
        → Route handler executes with guaranteed auth context

3. Refresh Flow:
   Client → POST /auth/refresh (refresh_token)
        → Validate refresh token exists and not expired
        → Check token family for reuse detection
        → Rotate: invalidate old refresh token, issue new pair
        → Return { access_token, refresh_token, expires_in }

4. Revocation Flow:
   Client → POST /auth/revoke (token)
        → Add JTI to Redis blacklist (TTL = remaining token life)
        → Write to PostgreSQL token_blacklist table
        → If refresh token, invalidate entire family
        → Log revocation event
        → Return 200 OK

5. Logout Flow:
   Client → POST /auth/logout
        → Revoke current access_token (JTI → blacklist)
        → Revoke all refresh tokens for session
        → Invalidate session record
        → Log logout event
        → Return 200 OK
```

---

## Technical Requirements

### TR-001: Auth REST API Endpoints

Production-grade auth endpoints built on FastAPI, leveraging existing `greenlang/auth/` modules.

**Endpoints:**

| Method | Path | Description | Auth Required | Rate Limit |
|---|---|---|---|---|
| `POST` | `/auth/login` | Authenticate with username/password + optional MFA | No | 10/min per IP |
| `POST` | `/auth/token` | Issue token via client_credentials grant | No | 20/min per IP |
| `POST` | `/auth/refresh` | Refresh access token using refresh token | No | 30/min per IP |
| `POST` | `/auth/revoke` | Revoke access or refresh token | Yes | 60/min |
| `POST` | `/auth/logout` | Logout (revoke session + tokens) | Yes | 60/min |
| `GET` | `/auth/validate` | Validate token and return claims | Yes | 1000/min |
| `GET` | `/auth/me` | Get current user profile | Yes | 100/min |
| `GET` | `/auth/jwks` | JWKS public key endpoint | No | 100/min |
| `POST` | `/auth/password/change` | Change own password | Yes | 5/min |
| `POST` | `/auth/password/reset` | Request password reset | No | 3/min per IP |
| `POST` | `/auth/mfa/setup` | Set up MFA for current user | Yes | 5/min |
| `POST` | `/auth/mfa/verify` | Verify MFA code | Yes | 10/min |

**Request/Response Models:**

```python
class LoginRequest(BaseModel):
    username: str                        # Username or email
    password: str
    mfa_code: Optional[str] = None       # TOTP code if MFA enabled
    tenant_id: Optional[str] = None      # Optional tenant context

class TokenResponse(BaseModel):
    access_token: str                    # JWT (RS256)
    refresh_token: str                   # Opaque token
    token_type: str = "Bearer"
    expires_in: int                      # Seconds until access_token expiry
    scope: str                           # Space-separated scopes
    tenant_id: Optional[str] = None

class TokenValidationResponse(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    scopes: List[str] = []
    expires_at: Optional[str] = None
```

**Requirements:**
1. All endpoints return standard JSON error responses with error codes
2. Login endpoint supports username OR email authentication
3. MFA verification required when user has MFA enabled
4. Token response includes standard OAuth2 fields (access_token, token_type, expires_in)
5. JWKS endpoint returns RSA public key in standard JWK format
6. All endpoints include `X-Request-ID` in response headers
7. OpenAPI/Swagger documentation auto-generated with security schemes

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Access Token Expiry | 24h | 4h | 1h |
| Refresh Token Expiry | 30d | 14d | 7d |
| Login Rate Limit (per IP) | 30/min | 15/min | 10/min |
| Token Validation Rate | 5000/min | 2000/min | 1000/min |
| Max Concurrent Sessions | 20 | 10 | 5 |

### TR-002: Token Revocation System

Two-layer token revocation with Redis (hot) + PostgreSQL (cold) for defense-in-depth.

**Architecture:**
```
Token validation check:
  1. Check Redis SET "gl:auth:blacklist" for JTI → O(1) lookup
  2. If Redis miss, check PostgreSQL token_blacklist table → indexed lookup
  3. Cache result in Redis with TTL = remaining token lifetime
```

**Requirements:**
1. Redis SET-based JTI blacklist with TTL matching remaining token lifetime
2. PostgreSQL `token_blacklist` table as durable fallback
3. Bulk revocation: revoke all tokens for a user (e.g., password change, account compromise)
4. Family revocation: revoke entire refresh token family on reuse detection (prevents token theft)
5. Cleanup CronJob: purge expired entries from PostgreSQL daily
6. Revocation propagation: < 1 second from revocation to enforcement
7. Graceful Redis failure: fall back to PostgreSQL-only validation (slower but correct)

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Redis Blacklist TTL | Match token | Match token | Match token |
| PostgreSQL Cleanup Interval | 1h | 6h | 24h |
| Max Blacklist Size (Redis) | 10K | 100K | 1M |
| Revocation Propagation | < 5s | < 2s | < 1s |

### TR-003: Refresh Token Rotation

Secure refresh token implementation with rotation and reuse detection.

**Requirements:**
1. Opaque refresh tokens (not JWT) stored as SHA-256 hash in PostgreSQL
2. Token rotation on every use: old token invalidated, new token issued
3. Token family tracking: all tokens from same login share a `family_id`
4. Reuse detection: if a rotated-out token is presented, revoke entire family (token theft signal)
5. Absolute expiry: refresh tokens expire after 7 days regardless of rotation
6. Device binding: optional — refresh tokens can be bound to device fingerprint
7. Concurrent refresh handling: only first refresh succeeds, subsequent requests get new family

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Refresh Token Lifetime | 30d | 14d | 7d |
| Max Tokens Per Family | 100 | 50 | 30 |
| Reuse Grace Period | 30s | 10s | 5s |
| Family Cleanup After | 30d | 14d | 7d |

### TR-004: Application-Level Auth Middleware

Register authentication middleware on the FastAPI application and protect all existing routes.

**Requirements:**
1. Register `AuthenticationMiddleware` in FastAPI app startup (`app.add_middleware(...)`)
2. Middleware chain: JWT validation → JTI revocation check → AuthContext injection
3. AuthContext available via `request.state.auth` with: user_id, tenant_id, roles, permissions, scopes, auth_method
4. Public endpoints whitelist: `/health`, `/readyz`, `/metrics`, `/auth/login`, `/auth/token`, `/auth/refresh`, `/auth/jwks`, `/docs`, `/openapi.json`
5. Protected endpoints get `@require_auth` decorator automatically
6. Endpoint-specific permissions via `@require_permissions("agents:execute")`
7. Tenant isolation enforcement: request tenant_id must match JWT tenant_id claim
8. Support both JWT (Authorization: Bearer) and API Key (X-API-Key header) authentication
9. Fallback ordering: JWT → API Key → 401 Unauthorized

**Route Protection Map:**

| Route Group | Auth | Required Permissions |
|---|---|---|
| `/api/v1/agents` (GET) | Required | `agents:list` |
| `/api/v1/agents/{id}` (GET) | Required | `agents:read` |
| `/api/v1/agents/{id}/execute` (POST) | Required | `agents:execute` |
| `/api/v1/agents/{id}/config` (PATCH) | Required | `agents:configure` |
| `/api/v1/emissions/calculate` (POST) | Required | `emissions:calculate` |
| `/api/v1/emissions/{id}` (GET) | Required | `emissions:read` |
| `/api/v1/factory/*` (ALL) | Required | `factory:*` (varies) |
| `/api/v1/flags/*` (ALL) | Required | `flags:*` (varies) |
| `/auth/*` (login, token, refresh, jwks) | Public | None |
| `/auth/*` (revoke, logout, me, password) | Required | Authenticated |
| `/auth/admin/*` | Required | `admin:users` |

### TR-005: Password Policy Enforcement

**Requirements:**
1. Minimum length: 12 characters (configurable)
2. Complexity: at least 1 uppercase, 1 lowercase, 1 digit, 1 special character
3. Password history: prevent reuse of last 5 passwords (SHA-256 hashed)
4. Password expiry: 90 days default, configurable per tenant
5. Forced password reset: admin can force user to change password on next login
6. Breach detection: optional check against HaveIBeenPwned API (k-anonymity model)
7. Common password rejection: block top 10,000 common passwords

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Min Length | 8 | 10 | 12 |
| Require Uppercase | No | Yes | Yes |
| Require Special Char | No | Yes | Yes |
| Password History Depth | 3 | 5 | 5 |
| Expiry Days | None | 180 | 90 |

### TR-006: Account Lockout

**Requirements:**
1. Lock account after 5 consecutive failed login attempts
2. Progressive lockout: 15min → 30min → 1h → 4h → 24h
3. Automatic unlock after lockout period expires
4. Admin unlock: immediate unlock via admin API
5. Lockout tracking: per-user with IP address logging
6. Rate limiting: separate from lockout — limits login attempts per IP address
7. Notification: emit event on lockout (for alerting integration)
8. Lockout bypass: service accounts exempt from lockout (use API keys instead)

| Parameter | Dev | Staging | Production |
|---|---|---|---|
| Max Failed Attempts | 10 | 7 | 5 |
| Initial Lockout Duration | 5min | 10min | 15min |
| Max Lockout Duration | 1h | 12h | 24h |
| Progressive Multiplier | 2x | 2x | 2x |
| IP Rate Limit (login) | 50/min | 20/min | 10/min |

### TR-007: Auth Audit Logging

**Requirements:**
1. Log ALL auth events as structured JSON to application logger (→ Loki via INFRA-009)
2. Event types: `login_success`, `login_failure`, `token_issued`, `token_validated`, `token_revoked`, `token_refreshed`, `logout`, `password_changed`, `password_reset_requested`, `mfa_setup`, `mfa_verified`, `account_locked`, `account_unlocked`, `permission_denied`
3. Each event includes: timestamp, event_type, user_id, tenant_id, ip_address, user_agent, correlation_id, details
4. Sensitive data redaction: never log passwords, tokens, or MFA secrets
5. Auth events tagged with Loki label `event_category=auth` for dedicated log stream
6. Retention: auth audit logs retained for 365 days (compliance requirement)
7. Prometheus metrics emitted alongside each event (counters by event type)

### TR-008: Auth Metrics & Monitoring

**Prometheus Metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `gl_auth_login_total` | Counter | `result`, `method`, `tenant_id` | Total login attempts |
| `gl_auth_login_duration_seconds` | Histogram | `result`, `method` | Login processing time |
| `gl_auth_token_issued_total` | Counter | `type`, `tenant_id` | Tokens issued (access/refresh) |
| `gl_auth_token_validated_total` | Counter | `result`, `method` | Token validation attempts |
| `gl_auth_token_revoked_total` | Counter | `reason`, `tenant_id` | Token revocations |
| `gl_auth_active_sessions` | Gauge | `tenant_id` | Current active sessions |
| `gl_auth_lockout_total` | Counter | `tenant_id` | Account lockouts |
| `gl_auth_permission_denied_total` | Counter | `permission`, `tenant_id` | Permission denials |
| `gl_auth_mfa_verification_total` | Counter | `result`, `tenant_id` | MFA verification attempts |
| `gl_auth_password_change_total` | Counter | `reason`, `tenant_id` | Password changes |

**Requirements:**
1. All metrics exposed at `/metrics` endpoint (Prometheus scrape)
2. Histogram buckets for latency: 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s
3. Metrics reset on service restart (counters are cumulative in Prometheus)

---

## Integration Points

### Integration with greenlang/auth/ (Existing Modules)
```python
# SEC-001 wraps existing modules — does NOT rewrite them
from greenlang.auth.jwt_handler import JWTHandler          # Token creation/validation
from greenlang.auth.auth import AuthManager                 # User authentication
from greenlang.auth.rbac import RBACManager                 # Role-based access
from greenlang.auth.permissions import PermissionEvaluator  # Fine-grained permissions
from greenlang.auth.middleware import AuthenticationMiddleware, AuthContext
from greenlang.auth.mfa import MFAManager                   # MFA verification
from greenlang.auth.api_key_manager import APIKeyManager    # API key validation
```

### Integration with INFRA-002 (PostgreSQL)
- Auth tables in `security` schema (V009 migration)
- Token blacklist, refresh tokens, password history, login attempts tables
- TimescaleDB hypertable for auth_events (if high volume)

### Integration with INFRA-003 (Redis)
- JTI blacklist: Redis SET with per-key TTL
- Session cache: active session metadata
- Rate limiting: sliding window counters per IP
- Refresh token lookup cache

### Integration with INFRA-006 (Kong API Gateway)
- Kong validates JWT signature (first layer)
- Auth service validates JTI revocation + permissions (second layer)
- Auth endpoints exposed through Kong at `/auth/*`
- Kong rate limiting as outer defense, app rate limiting as inner defense

### Integration with INFRA-008 (Feature Flags)
- MFA enforcement gated by feature flag `gl.auth.mfa_required`
- Password complexity rules gated by `gl.auth.password_policy_v2`
- Refresh token rotation gated by `gl.auth.refresh_rotation`

### Integration with INFRA-009 (Log Aggregation)
- Auth events logged with structured JSON format
- Loki labels: `event_category=auth`, `event_type=<type>`
- Correlation IDs from INFRA-010 telemetry propagated through auth events

### Integration with INFRA-010 (Agent Factory)
- Agent Factory API routes protected with `agents:*` permissions
- Service-to-service auth for agent execution via dedicated JWT consumer
- Agent pack publish requires `hub:publish` permission

---

## Database Migration

### V009__auth_service.sql

```sql
-- ============================================================
-- V009: JWT Authentication Service Schema
-- ============================================================
-- PRD: SEC-001
-- Creates 4 tables for token revocation, refresh tokens,
-- password history, and login attempt tracking.
-- ============================================================

-- Ensure security schema exists
CREATE SCHEMA IF NOT EXISTS security;

-- Token Blacklist Table (revoked JTIs)
CREATE TABLE IF NOT EXISTS security.token_blacklist (
    jti UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    tenant_id UUID,
    token_type VARCHAR(16) NOT NULL DEFAULT 'access' CHECK (token_type IN ('access', 'refresh')),
    revoked_by VARCHAR(128) NOT NULL,
    revoke_reason VARCHAR(64) NOT NULL DEFAULT 'logout' CHECK (revoke_reason IN (
        'logout', 'password_change', 'admin_revoke', 'token_theft',
        'session_expire', 'family_revoke', 'bulk_revoke'
    )),
    original_expiry TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tb_user ON security.token_blacklist(user_id);
CREATE INDEX idx_tb_tenant ON security.token_blacklist(tenant_id);
CREATE INDEX idx_tb_expiry ON security.token_blacklist(original_expiry);

-- Refresh Tokens Table
CREATE TABLE IF NOT EXISTS security.refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash VARCHAR(128) NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    tenant_id UUID,
    family_id UUID NOT NULL,
    device_fingerprint VARCHAR(256),
    ip_address INET,
    user_agent VARCHAR(512),
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_rotated BOOLEAN NOT NULL DEFAULT false,
    rotated_at TIMESTAMPTZ,
    rotated_to UUID REFERENCES security.refresh_tokens(id),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_rt_user ON security.refresh_tokens(user_id);
CREATE INDEX idx_rt_family ON security.refresh_tokens(family_id);
CREATE INDEX idx_rt_active ON security.refresh_tokens(is_active, expires_at) WHERE is_active = true;
CREATE INDEX idx_rt_hash ON security.refresh_tokens(token_hash);

-- Password History Table
CREATE TABLE IF NOT EXISTS security.password_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by VARCHAR(128) NOT NULL DEFAULT 'self',
    change_reason VARCHAR(64) NOT NULL DEFAULT 'user_change' CHECK (change_reason IN (
        'user_change', 'admin_reset', 'forced_reset', 'expiry', 'policy_change'
    ))
);

CREATE INDEX idx_ph_user ON security.password_history(user_id, changed_at DESC);

-- Login Attempts Table (for lockout tracking)
CREATE TABLE IF NOT EXISTS security.login_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    username VARCHAR(256) NOT NULL,
    tenant_id UUID,
    ip_address INET NOT NULL,
    user_agent VARCHAR(512),
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR(64),
    lockout_until TIMESTAMPTZ,
    attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_la_user ON security.login_attempts(user_id, attempt_at DESC);
CREATE INDEX idx_la_ip ON security.login_attempts(ip_address, attempt_at DESC);
CREATE INDEX idx_la_username ON security.login_attempts(username, attempt_at DESC);

-- Row-Level Security
ALTER TABLE security.token_blacklist ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.refresh_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.password_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.login_attempts ENABLE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY token_blacklist_tenant ON security.token_blacklist
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY refresh_tokens_tenant ON security.refresh_tokens
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY login_attempts_tenant ON security.login_attempts
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

-- Cleanup function for expired blacklist entries
CREATE OR REPLACE FUNCTION security.cleanup_expired_blacklist()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security.token_blacklist
    WHERE original_expiry < NOW() - INTERVAL '1 hour';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for expired refresh tokens
CREATE OR REPLACE FUNCTION security.cleanup_expired_refresh_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security.refresh_tokens
    WHERE expires_at < NOW() - INTERVAL '1 day'
       OR (is_rotated = true AND rotated_at < NOW() - INTERVAL '7 days');
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

---

## Migration Plan

### Phase 1: Core Service (P0)
1. Create `greenlang/infrastructure/auth_service/` package structure
2. Implement token_service.py: wraps existing jwt_handler.py with JTI tracking
3. Implement revocation.py: Redis L1 + PostgreSQL L2 JTI blacklist
4. Implement refresh_tokens.py: rotation, family tracking, reuse detection
5. Run database migration V009 (4 tables)
6. Implement auth REST API endpoints (auth_routes.py)

### Phase 2: Security Controls (P0)
1. Implement password_policy.py: complexity, history, expiry
2. Implement account_lockout.py: progressive backoff, admin unlock
3. Implement rate_limiter.py: per-IP sliding window
4. Implement auth_audit.py: structured event logging
5. Implement auth_metrics.py: Prometheus counters/histograms

### Phase 3: Middleware & Route Protection (P1)
1. Implement route_protector.py: utility to apply decorators to existing routers
2. Register AuthenticationMiddleware in FastAPI app startup
3. Apply @require_auth to all existing API routes
4. Apply endpoint-specific @require_permissions decorators
5. Test defense-in-depth: Kong + app middleware

### Phase 4: Deployment & Monitoring (P2)
1. Create K8s manifests: deployment, service, HPA, networkpolicy, configmap
2. Create CronJob for token blacklist cleanup
3. Create Grafana dashboard (15+ panels)
4. Create Prometheus alert rules (10+ rules)
5. Create Helm template for auth service

### Phase 5: Testing (P2)
1. Unit tests for all modules (8 test files, 200+ tests)
2. Integration tests for auth flows (2 test files)
3. Load tests for auth throughput (1 test file)
4. Validate 85%+ code coverage

---

## Acceptance Criteria

1. Login endpoint authenticates users with username/password + optional MFA and returns JWT + refresh token
2. Access tokens are RS256-signed with standard claims (sub, iss, aud, exp, iat, jti, tenant_id, roles, permissions)
3. Token revocation propagates to enforcement within 1 second (Redis blacklist)
4. Refresh token rotation invalidates old token and detects reuse (family revocation)
5. All existing API routes return 401 when no valid token is provided
6. All existing API routes enforce endpoint-specific permissions
7. Account locks after 5 failed login attempts with progressive backoff
8. Password changes enforce complexity + history policies
9. Auth events are logged as structured JSON to Loki with correlation IDs
10. JWKS endpoint returns valid JWK for external token validation
11. Token validation latency < 5ms P99 (Redis hit) and < 20ms P99 (PostgreSQL fallback)
12. Auth service handles 1000+ token validations/second
13. Cleanup CronJob purges expired blacklist entries without impacting service
14. Grafana dashboard shows real-time auth metrics (logins, failures, lockouts, revocations)
15. All new modules achieve 85%+ test coverage

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| INFRA-001: EKS Cluster | COMPLETE | Auth service deploys on EKS |
| INFRA-002: PostgreSQL | COMPLETE | Auth tables in `security` schema |
| INFRA-003: Redis | COMPLETE | JTI blacklist, rate limits, session cache |
| INFRA-006: Kong API Gateway | COMPLETE | JWT validation (first layer), auth route exposure |
| INFRA-008: Feature Flags | COMPLETE | Gate MFA, password policy features |
| INFRA-009: Log Aggregation | COMPLETE | Auth event logging to Loki |
| INFRA-010: Agent Factory | COMPLETE | Route protection for factory endpoints |
| greenlang/auth/ | EXISTS | Core auth modules (JWT, RBAC, OAuth2, MFA, middleware) |
| greenlang/db/models_auth.py | EXISTS | SQLAlchemy models (User, Role, Permission, Session, etc.) |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Redis blacklist failure causes token acceptance | Medium | Critical | PostgreSQL fallback, circuit breaker on Redis |
| Refresh token reuse detection false positives | Low | High | Grace period (5s) for concurrent requests |
| Password policy breaks existing user logins | Medium | Medium | Feature flag gate, migration period with warnings |
| Account lockout DoS (attacker locks legitimate users) | Medium | High | CAPTCHA after 3 attempts, IP-based rate limiting |
| Auth middleware latency impacts all API requests | Low | High | Redis-first validation, async audit logging |
| Key rotation causes token validation failures | Low | Critical | JWKS with multiple keys, overlap period during rotation |

---

## Development Tasks (Ralphy-Compatible)

### Phase 1: Core Service
- [ ] Create package: `greenlang/infrastructure/auth_service/__init__.py`
- [ ] Create token service: `greenlang/infrastructure/auth_service/token_service.py`
- [ ] Create revocation: `greenlang/infrastructure/auth_service/revocation.py`
- [ ] Create refresh tokens: `greenlang/infrastructure/auth_service/refresh_tokens.py`
- [ ] Create auth API routes: `greenlang/infrastructure/auth_service/api/auth_routes.py`
- [ ] Create user routes: `greenlang/infrastructure/auth_service/api/user_routes.py`
- [ ] Create API init: `greenlang/infrastructure/auth_service/api/__init__.py`
- [ ] Create database migration: `deployment/database/migrations/sql/V009__auth_service.sql`

### Phase 2: Security Controls
- [ ] Create password policy: `greenlang/infrastructure/auth_service/password_policy.py`
- [ ] Create account lockout: `greenlang/infrastructure/auth_service/account_lockout.py`
- [ ] Create rate limiter: `greenlang/infrastructure/auth_service/rate_limiter.py`
- [ ] Create auth audit: `greenlang/infrastructure/auth_service/auth_audit.py`
- [ ] Create auth metrics: `greenlang/infrastructure/auth_service/auth_metrics.py`
- [ ] Create JWKS endpoint: `greenlang/infrastructure/auth_service/jwks_endpoint.py`

### Phase 3: Middleware & Route Protection
- [ ] Create route protector: `greenlang/infrastructure/auth_service/route_protector.py`
- [ ] Create admin routes: `greenlang/infrastructure/auth_service/api/admin_routes.py`
- [ ] Apply auth middleware to FastAPI app
- [ ] Apply @require_auth to all existing API routes
- [ ] Apply @require_permissions to all existing API routes

### Phase 4: Deployment & Monitoring
- [ ] Create K8s namespace: `deployment/kubernetes/auth-service/namespace.yaml`
- [ ] Create K8s deployment: `deployment/kubernetes/auth-service/deployment.yaml`
- [ ] Create K8s service: `deployment/kubernetes/auth-service/service.yaml`
- [ ] Create K8s HPA: `deployment/kubernetes/auth-service/hpa.yaml`
- [ ] Create K8s networkpolicy: `deployment/kubernetes/auth-service/networkpolicy.yaml`
- [ ] Create K8s configmap: `deployment/kubernetes/auth-service/configmap.yaml`
- [ ] Create K8s CronJob: `deployment/kubernetes/auth-service/cronjob-token-cleanup.yaml`
- [ ] Create K8s kustomization: `deployment/kubernetes/auth-service/kustomization.yaml`
- [ ] Create Helm template: `deployment/helm/greenlang-agents/templates/deployment-auth-service.yaml`
- [ ] Create Grafana dashboard: `deployment/monitoring/dashboards/auth-service.json`
- [ ] Create Prometheus alerts: `deployment/monitoring/alerts/auth-service-alerts.yaml`

### Phase 5: Testing
- [ ] Create test: `tests/unit/auth_service/test_token_service.py`
- [ ] Create test: `tests/unit/auth_service/test_revocation.py`
- [ ] Create test: `tests/unit/auth_service/test_refresh_tokens.py`
- [ ] Create test: `tests/unit/auth_service/test_password_policy.py`
- [ ] Create test: `tests/unit/auth_service/test_account_lockout.py`
- [ ] Create test: `tests/unit/auth_service/test_auth_routes.py`
- [ ] Create test: `tests/unit/auth_service/test_auth_audit.py`
- [ ] Create test: `tests/unit/auth_service/test_route_protector.py`
- [ ] Create test: `tests/integration/auth_service/test_auth_flow_e2e.py`
- [ ] Create test: `tests/integration/auth_service/test_token_lifecycle.py`
- [ ] Create test: `tests/load/auth_service/test_auth_throughput.py`
