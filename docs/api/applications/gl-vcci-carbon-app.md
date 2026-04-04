# GL-VCCI-Carbon-APP -- VCCI Scope 3 Carbon Intelligence Platform API Reference

**Source:** `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/main.py`
**Calculator Routes:** `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/calculator/routes.py`
**Title:** GL-VCCI Scope 3 Carbon Intelligence API
**Version:** 2.0.0

---

## Overview

Enterprise-grade Scope 3 emissions tracking platform with AI-powered intelligence. The API uses JWT authentication (SEC-001) on all agent routes, rate limiting via `slowapi`, security headers middleware, Sentry error tracking, and Prometheus instrumentation.

**Authentication:** JWT Bearer Token via `Authorization: Bearer <token>` (all `/api/v1/` routes)
**Docs:** `/docs`, `/redoc` (disabled in production)

---

## Registered Routers

| Prefix | Tag | Router Module | Description |
|--------|-----|---------------|-------------|
| `/api/v1/intake` | Intake Agent | `services.agents.intake.routes` | Data ingestion and validation |
| `/api/v1/calculator` | Calculator Agent | `services.agents.calculator.routes` | Scope 3 calculations (15 categories) |
| `/api/v1/hotspot` | Hotspot Agent | `services.agents.hotspot.routes` | Emission hotspot identification |
| `/api/v1/engagement` | Engagement Agent | `services.agents.engagement.routes` | Supplier engagement workflows |
| `/api/v1/reporting` | Reporting Agent | `services.agents.reporting.routes` | Report generation |
| `/api/v1/factors` | Factor Broker | `services.factor_broker.routes` | Emission factor management |
| `/api/v1/methodologies` | Methodologies | `services.methodologies.routes` | Calculation methodology registry |
| `/api/v1/connectors` | ERP Connectors | `connectors.routes` | SAP/Oracle/Workday integration |

All routers require `Authorization: Bearer <token>` via the `verify_token` dependency.

---

## Health Check Endpoints

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| GET | `/` | Root -- API information | None |
| GET | `/health/live` | Kubernetes liveness probe | None |
| GET | `/health/ready` | Kubernetes readiness probe (DB + Redis) | None |
| GET | `/health/startup` | Kubernetes startup probe | None |
| GET | `/health/detailed` | Full dependency health with circuit breaker states | None |

### GET /health/detailed

Returns comprehensive health including latencies and circuit breaker states for external dependencies (Factor Broker, LLM Provider, ERP SAP).

**Response (200):**

```json
{
  "status": "healthy",
  "timestamp": "2026-04-04T12:00:00Z",
  "service": "gl-vcci-api",
  "version": "2.0.0",
  "dependencies": {
    "database": {
      "status": "healthy",
      "latency_ms": 2.5,
      "type": "postgresql"
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 0.8,
      "type": "cache"
    },
    "factor_broker": {
      "status": "healthy",
      "circuit_breaker": "closed",
      "failure_count": 0,
      "success_count": 1250,
      "total_calls": 1250,
      "type": "external_api"
    }
  }
}
```

**Status values:** `healthy`, `degraded` (circuit breaker half-open or open), `unhealthy` (DB/Redis down).

**HTTP 503** returned when status is `unhealthy`.

---

## Calculator Agent Endpoints

**Router Prefix:** `/api/v1/calculator/calculate`
**Tags:** `Calculator`

The Scope 3 Calculator Agent supports all 15 GHG Protocol Scope 3 categories using a 3-tier waterfall approach.

### Category-Specific Calculation Endpoints

| Method | Path | Category | Description |
|--------|------|----------|-------------|
| POST | `/category/1` | Cat 1 | Purchased Goods and Services |
| POST | `/category/2` | Cat 2 | Capital Goods |
| POST | `/category/3` | Cat 3 | Fuel and Energy-Related Activities |
| POST | `/category/4` | Cat 4 | Upstream Transportation and Distribution (ISO 14083) |
| POST | `/category/5` | Cat 5 | Waste Generated in Operations |
| POST | `/category/6` | Cat 6 | Business Travel (flights, hotels, ground transport) |
| POST | `/category/7` | Cat 7 | Employee Commuting |
| POST | `/category/8` | Cat 8 | Upstream Leased Assets |
| POST | `/category/9` | Cat 9 | Downstream Transportation and Distribution |
| POST | `/category/10` | Cat 10 | Processing of Sold Products |
| POST | `/category/11` | Cat 11 | Use of Sold Products |
| POST | `/category/12` | Cat 12 | End-of-Life Treatment of Sold Products |
| POST | `/category/13` | Cat 13 | Downstream Leased Assets |
| POST | `/category/14` | Cat 14 | Franchises |
| POST | `/category/15` | Cat 15 | Investments (PCAF Standard) |

### Dynamic and Batch Endpoints

| Method | Path | Summary |
|--------|------|---------|
| POST | `/{category}` | Calculate by category number (1-15) |
| POST | `/batch` | Batch calculation for multiple records |
| POST | `/all` | Calculate multiple categories in one request |

### Metadata and Utility

| Method | Path | Summary |
|--------|------|---------|
| GET | `/categories` | List all 15 categories with descriptions |
| GET | `/stats` | Performance statistics |
| POST | `/stats/reset` | Reset statistics (204 No Content) |
| GET | `/health` | Calculator health check |
| GET | `/config` | Current calculator configuration |

### Example: POST /api/v1/calculator/calculate/category/1

**Request Body (Category1Input):**

```json
{
  "product_name": "Steel",
  "quantity": 1000,
  "quantity_unit": "kg",
  "region": "US",
  "supplier_pcf": 1.85
}
```

**Response (CalculationResult):**

```json
{
  "category": 1,
  "category_name": "Purchased Goods & Services",
  "total_emissions_tco2e": 1.85,
  "calculation_tier": "tier_1",
  "data_quality_score": 0.95,
  "uncertainty_pct": 5.0,
  "provenance_hash": "sha256:abc123...",
  "methodology": "GHG Protocol Scope 3 Category 1",
  "emission_factor_source": "Supplier PCF"
}
```

### Example: POST /api/v1/calculator/calculate/batch

**Request Body:**

```json
{
  "category": 1,
  "records": [
    {"product_name": "Steel", "quantity": 1000, "quantity_unit": "kg", "region": "US", "supplier_pcf": 1.85},
    {"product_name": "Aluminum", "quantity": 500, "quantity_unit": "kg", "region": "EU"}
  ]
}
```

**Response (BatchResult):**

```json
{
  "total_records": 2,
  "successful_records": 2,
  "failed_records": 0,
  "total_emissions_tco2e": 6.35,
  "results": [...]
}
```

**Error (207 Multi-Status):** Returned when some records fail in batch processing.

---

## Metrics Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/metrics` | VCCI-specific Prometheus metrics |
| GET | `/metrics/http` | Default HTTP instrumentation metrics |

---

## Source Files

- `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/main.py`
- `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/calculator/routes.py`
