# PRD: INFRA-006 - Deploy API Gateway (Kong)

**Document Version:** 1.0
**Date:** February 4, 2026
**Status:** READY FOR EXECUTION
**Priority:** P1 - HIGH
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-006

---

## Executive Summary

Deploy a production-ready Kong API Gateway on AWS EKS to provide centralized API management, traffic control, authentication enforcement, rate limiting, request/response transformation, and observability for all GreenLang Climate OS services. Kong replaces the current distributed middleware approach (NGINX Ingress + Istio EnvoyFilters + FastAPI middleware) with a unified, plugin-driven API gateway layer.

### Current State
- NGINX Ingress Controller handles basic routing and TLS termination
- Istio Service Mesh provides mTLS, VirtualServices, DestinationRules
- OAuth2 Proxy + Keycloak handle authentication
- Rate limiting implemented at application layer (FastAPI middleware with Redis)
- No centralized API management, analytics, or developer portal
- No gateway-level request/response transformation
- No centralized API versioning management
- No unified API consumer management

### Target State
- Kong Gateway deployed as the primary API entry point on EKS
- Centralized rate limiting (Redis-backed, replacing application middleware)
- Gateway-level JWT/OAuth2 authentication with Keycloak integration
- Request/response transformation at gateway layer
- API versioning management via Kong routes
- Developer portal with OpenAPI documentation
- Comprehensive gateway metrics and Grafana dashboards
- Kong Ingress Controller (KIC) managing Kubernetes-native CRDs
- Canary/blue-green deployments via Kong traffic splitting
- Circuit breaking and health checking at gateway level

---

## Scope

### In Scope
1. Kong Gateway deployment on EKS (DB-less mode with declarative config)
2. Kong Ingress Controller (KIC) for Kubernetes-native management
3. Terraform module for Kong infrastructure (namespace, RBAC, HPA, PDB)
4. Helm chart with environment-specific values (dev/staging/prod)
5. Kong plugins: rate-limiting, jwt, oauth2, cors, request-transformer, response-transformer, prometheus, http-log, ip-restriction, bot-detection, request-size-limiting, acl
6. Kong declarative configuration (kong.yaml) with all routes, services, upstreams
7. Integration with existing Redis cluster for rate limiting state
8. Integration with existing Keycloak for OAuth2/OIDC
9. Integration with existing Istio mesh (mTLS preserved)
10. Grafana dashboards and Prometheus alert rules
11. Kong custom plugin for GreenLang tenant isolation
12. Migration plan from NGINX Ingress to Kong Ingress Controller

### Out of Scope
- Kong Enterprise features (Dev Portal, Vitals, RBAC - use OSS equivalent)
- Multi-region gateway deployment (future phase)
- GraphQL gateway (future consideration)
- API monetization and billing
- Kong Konnect cloud management plane

---

## Architecture

### High-Level Architecture

```
Internet
    |
Route53 DNS (api.greenlang.io)
    |
AWS NLB (Layer 4, TLS passthrough)
    |
Kong Gateway (Layer 7 - API Gateway)
    ├── TLS Termination
    ├── Rate Limiting (Redis-backed)
    ├── JWT/OAuth2 Authentication
    ├── Request/Response Transformation
    ├── IP Restriction & Bot Detection
    ├── Logging & Metrics (Prometheus)
    ├── Circuit Breaking & Health Checks
    └── Traffic Splitting (Canary/Blue-Green)
    |
Istio Service Mesh (mTLS between services)
    |
    ├── greenlang-api (FastAPI :8080)
    ├── greenlang-web (React :3000)
    ├── greenlang-admin (Admin :8080)
    ├── greenlang-grpc (gRPC :50051)
    ├── greenlang-worker (Worker :8080)
    └── agent-services (Fuel, CBAM, EUDR, etc.)
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Kong Gateway Pod                       │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Kong Proxy   │  │ Kong Admin   │  │ Kong Status  │ │
│  │ :8000 (HTTP) │  │ :8001        │  │ :8100        │ │
│  │ :8443 (HTTPS)│  │ (internal)   │  │ (metrics)    │ │
│  └──────┬───────┘  └──────────────┘  └──────────────┘ │
│         │                                               │
│  ┌──────┴───────────────────────────────────────────┐  │
│  │                 Plugin Chain                      │  │
│  │  1. ip-restriction → 2. bot-detection            │  │
│  │  3. cors → 4. jwt/oauth2 → 5. acl               │  │
│  │  6. request-size-limiting → 7. rate-limiting     │  │
│  │  8. request-transformer → 9. response-transformer│  │
│  │  10. prometheus → 11. http-log                   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │  Redis  │ (Rate limit counters, session cache)
    └─────────┘
```

### Kong Ingress Controller (KIC) Architecture

```
┌─────────────────────────────────────────────┐
│       Kong Ingress Controller (KIC)         │
│                                             │
│  Watches Kubernetes resources:              │
│  ├── Ingress / IngressClass                 │
│  ├── KongIngress (CRD)                      │
│  ├── KongPlugin (CRD)                       │
│  ├── KongClusterPlugin (CRD)               │
│  ├── KongConsumer (CRD)                     │
│  ├── KongConsumerGroup (CRD)               │
│  ├── TCPIngress (CRD)                       │
│  └── UDPIngress (CRD)                       │
│                                             │
│  Generates Kong declarative config          │
│  Pushes to Kong Admin API                   │
└─────────────────────────────────────────────┘
```

---

## Technical Requirements

### TR-001: Kong Gateway Deployment
- Deploy Kong OSS 3.x in DB-less mode (declarative YAML config)
- 3 replicas minimum in production (across 3 AZs)
- HPA: min 3, max 10, target CPU 70%
- PDB: minAvailable 2
- Resource requests: 500m CPU, 512Mi memory
- Resource limits: 2000m CPU, 2Gi memory
- Rolling update strategy (maxSurge: 1, maxUnavailable: 0)
- Pod anti-affinity for zone distribution
- Topology spread constraints across AZs

### TR-002: Kong Ingress Controller
- Deploy KIC 3.x alongside Kong Gateway
- IngressClass: kong (set as default for greenlang namespace)
- Watch namespaces: greenlang, greenlang-agents
- Kubernetes CRD-based route management
- Automatic config sync to Kong proxy

### TR-003: Rate Limiting (Redis-Backed)
- Plugin: rate-limiting-advanced (or rate-limiting with redis policy)
- Redis connection: reuse existing ElastiCache cluster
- Tiers:
  - Free tier: 100 req/min, 1000 req/hour
  - Standard tier: 1000 req/min, 50000 req/hour
  - Enterprise tier: 10000 req/min, unlimited
- Per-consumer, per-route, and global limits
- Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
- Burst allowance: 20% above limit for short bursts

### TR-004: Authentication & Authorization
- JWT plugin for API key validation
- OAuth2 plugin integrated with Keycloak OIDC
- OIDC Discovery URL: https://auth.greenlang.io/.well-known/openid-configuration
- JWT claims mapping: sub → consumer_id, tenant_id → header
- ACL plugin for role-based access control
- Consumer groups: admin, standard, readonly, agent-executor
- Anonymous access allowed: /health, /metrics, /api/docs, /api/openapi.json

### TR-005: Request/Response Transformation
- Add X-Request-ID header (UUID) to all requests
- Add X-Response-Time header to all responses
- Inject X-Tenant-ID from JWT claims into upstream headers
- Strip sensitive headers from responses (Server, X-Powered-By)
- Add HSTS, CSP, X-Frame-Options, X-Content-Type-Options headers
- Rename/normalize API version headers

### TR-006: Traffic Management
- Canary deployments: route 5%/10%/25%/50%/100% traffic splits
- Blue-green deployments via upstream target weighting
- A/B testing via header-based routing (X-API-Version)
- Circuit breaking: 5 consecutive 5xx errors → open circuit for 30s
- Health checks: active (HTTP GET /health every 10s) and passive
- Upstream connection pooling: max 100 connections per upstream
- Request timeout: 60s default, 300s for batch endpoints

### TR-007: Observability
- Prometheus plugin exposing /metrics on port 8100
- Metrics: request count, latency histogram, bandwidth, upstream health
- Per-route and per-consumer metric labels
- HTTP log plugin sending structured JSON to Loki/FluentBit
- Request/response logging for audit (configurable per route)
- Grafana dashboard: kong-gateway.json
- Prometheus alerts: kong-alerts.yaml

### TR-008: Security
- IP restriction plugin for admin endpoints (internal CIDR only)
- Bot detection plugin for public endpoints
- Request size limiting: 10MB default, 50MB for batch endpoints
- TLS 1.2+ only (TLS 1.3 preferred)
- mTLS with Istio sidecar for upstream connections
- CORS plugin with environment-specific origins
- Content-Security-Policy headers

### TR-009: Kong Routes & Services

| Route Pattern | Service | Rate Limit | Auth Required |
|---|---|---|---|
| /api/v1/factors/* | greenlang-api:8080 | 1000/min | Yes |
| /api/v1/calculate/* | greenlang-api:8080 | 500/min | Yes |
| /api/v1/calculate/batch | greenlang-api:8080 | 100/min | Yes |
| /api/v1/agents/* | greenlang-api:8080 | 500/min | Yes |
| /api/v1/stats/* | greenlang-api:8080 | 100/min | Yes |
| /api/v1/webhooks/* | greenlang-api:8080 | 200/min | Yes |
| /api/v1/health | greenlang-api:8080 | Unlimited | No |
| /api/docs | greenlang-api:8080 | 100/min | No |
| /api/openapi.json | greenlang-api:8080 | 100/min | No |
| /grpc.* | greenlang-grpc:50051 | 500/min | Yes |
| /ws/* | greenlang-websocket:8081 | 50/min | Yes |
| /app/* | greenlang-web:3000 | 1000/min | No |
| /admin/* | greenlang-admin:8080 | 200/min | Yes (admin) |

### TR-010: Environment Configuration

| Parameter | Dev | Staging | Prod |
|---|---|---|---|
| Replicas | 1 | 2 | 3 |
| HPA Max | 2 | 5 | 10 |
| CPU Request | 250m | 500m | 500m |
| Memory Request | 256Mi | 512Mi | 512Mi |
| CPU Limit | 500m | 1000m | 2000m |
| Memory Limit | 512Mi | 1Gi | 2Gi |
| Rate Limit Redis | localhost | ElastiCache | ElastiCache |
| Log Level | debug | info | warn |
| Admin API | enabled | enabled | disabled (internal only) |
| Proxy Cache | disabled | enabled | enabled |
| TLS | self-signed | staging LE | prod LE |

---

## File Structure

```
deployment/
├── terraform/
│   └── modules/
│       └── kong-gateway/
│           ├── main.tf              # Kong namespace, service accounts, IAM
│           ├── variables.tf         # Module input variables
│           └── outputs.tf           # Module outputs
├── helm/
│   └── kong-gateway/
│       ├── Chart.yaml              # Helm chart metadata
│       ├── values.yaml             # Default values (production)
│       ├── values-dev.yaml         # Development overrides
│       ├── values-staging.yaml     # Staging overrides
│       └── templates/
│           ├── _helpers.tpl        # Template helpers
│           ├── deployment.yaml     # Kong Gateway deployment
│           ├── service.yaml        # Kong services (proxy, admin, status)
│           ├── hpa.yaml            # Horizontal Pod Autoscaler
│           ├── pdb.yaml            # Pod Disruption Budget
│           ├── configmap.yaml      # Kong declarative config
│           ├── secret.yaml         # Kong secrets
│           └── servicemonitor.yaml # Prometheus ServiceMonitor
├── kubernetes/
│   └── kong-gateway/
│       ├── kong-plugins.yaml       # KongPlugin CRDs
│       ├── kong-consumers.yaml     # KongConsumer CRDs
│       ├── kong-routes.yaml        # KongIngress CRDs + Ingress routes
│       └── networkpolicy.yaml      # Kong NetworkPolicies
├── monitoring/
│   ├── dashboards/
│   │   └── kong-gateway.json       # Grafana dashboard
│   └── alerts/
│       └── kong-alerts.yaml        # Prometheus alert rules
└── config/
    └── kong/
        ├── kong.yaml               # Declarative config (routes, services, plugins)
        └── custom-plugins/
            └── gl-tenant-isolation/ # Custom GreenLang plugin
                ├── handler.lua
                └── schema.lua
```

---

## Migration Plan

### Phase 1: Parallel Deployment (Week 1)
1. Deploy Kong alongside existing NGINX Ingress
2. Route /health and /api/docs through Kong (non-critical)
3. Validate Kong metrics and logging
4. Compare latency: Kong vs NGINX Ingress

### Phase 2: Gradual Migration (Week 2)
1. Migrate /api/v1/factors routes to Kong
2. Migrate /api/v1/stats routes to Kong
3. Enable rate limiting at Kong (disable in FastAPI middleware)
4. Validate auth flow through Kong JWT/OAuth2 plugins

### Phase 3: Full Migration (Week 3)
1. Migrate all remaining API routes to Kong
2. Migrate gRPC and WebSocket routes
3. Migrate admin portal routes
4. Remove rate limiting middleware from FastAPI application
5. Update DNS records to point to Kong proxy service

### Phase 4: Cleanup (Week 4)
1. Remove NGINX Ingress Controller (keep as fallback)
2. Remove OAuth2 Proxy (replaced by Kong OAuth2 plugin)
3. Remove application-level rate limiting code
4. Update documentation and runbooks
5. Final performance validation

---

## Acceptance Criteria

1. Kong Gateway processes 100% of external API traffic
2. P99 latency < 10ms added by gateway (proxy overhead)
3. Rate limiting enforced at gateway level (not application)
4. JWT/OAuth2 authentication at gateway level
5. All routes defined in Kong declarative config
6. Grafana dashboard showing gateway metrics
7. Prometheus alerts configured for gateway health
8. Zero-downtime migration from NGINX Ingress
9. Canary deployment capability demonstrated
10. All existing API tests pass through Kong gateway

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| INFRA-001: EKS Cluster | COMPLETE | Kong deploys on EKS |
| INFRA-002: PostgreSQL | COMPLETE | Not needed (DB-less mode) |
| INFRA-003: Redis | COMPLETE | Used for rate limiting |
| INFRA-004: S3 | COMPLETE | Used for config backups |
| INFRA-005: pgvector | COMPLETE | No direct dependency |
| Keycloak/OAuth2 | EXISTS | Kong integrates with existing |
| cert-manager | EXISTS | Kong uses existing TLS certs |
| Istio Mesh | EXISTS | Kong integrates with Istio |

---

## Development Tasks (Ralphy-Compatible)

- [x] Create Terraform module: deployment/terraform/modules/kong-gateway/main.tf
- [x] Create Terraform variables: deployment/terraform/modules/kong-gateway/variables.tf
- [x] Create Terraform outputs: deployment/terraform/modules/kong-gateway/outputs.tf
- [x] Create Helm Chart.yaml: deployment/helm/kong-gateway/Chart.yaml
- [x] Create Helm values.yaml: deployment/helm/kong-gateway/values.yaml
- [x] Create Helm values-dev.yaml: deployment/helm/kong-gateway/values-dev.yaml
- [x] Create Helm values-staging.yaml: deployment/helm/kong-gateway/values-staging.yaml
- [x] Create Helm template _helpers.tpl
- [x] Create Helm template deployment.yaml (Kong Gateway + KIC)
- [x] Create Helm template service.yaml (proxy, admin, status)
- [x] Create Helm template hpa.yaml
- [x] Create Helm template pdb.yaml
- [x] Create Helm template configmap.yaml (declarative config mount)
- [x] Create Helm template secret.yaml
- [x] Create Helm template servicemonitor.yaml
- [x] Create Kong declarative config: deployment/config/kong/kong.yaml
- [x] Create Kong rate-limiting plugin config (Redis-backed)
- [x] Create Kong jwt plugin config
- [x] Create Kong oauth2 plugin config (Keycloak)
- [x] Create Kong cors plugin config
- [x] Create Kong request-transformer plugin config
- [x] Create Kong response-transformer plugin config
- [x] Create Kong ip-restriction plugin config
- [x] Create Kong bot-detection plugin config
- [x] Create Kong request-size-limiting plugin config
- [x] Create Kong prometheus plugin config
- [x] Create Kong http-log plugin config
- [x] Create Kong acl plugin config
- [x] Create KongPlugin CRDs: deployment/kubernetes/kong-gateway/kong-plugins.yaml
- [x] Create KongConsumer CRDs: deployment/kubernetes/kong-gateway/kong-consumers.yaml
- [x] Create Kong Ingress routes: deployment/kubernetes/kong-gateway/kong-routes.yaml
- [x] Create Kong NetworkPolicy: deployment/kubernetes/kong-gateway/networkpolicy.yaml
- [x] Create custom plugin: gl-tenant-isolation handler.lua
- [x] Create custom plugin: gl-tenant-isolation schema.lua
- [x] Create Grafana dashboard: deployment/monitoring/dashboards/kong-gateway.json
- [x] Create Prometheus alerts: deployment/monitoring/alerts/kong-alerts.yaml
- [x] Create Ralphy task file: .ralphy/INFRA-006-tasks.md
- [x] Update MEMORY.md with INFRA-006 status
