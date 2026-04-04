# GreenLang Applications API Reference -- Overview

**Version:** 1.0.0
**Last Updated:** 2026-04-04
**Base URL (per-app standalone):** `http://localhost:8000`
**Base URL (CBAM Pack shell):** `http://localhost:8000` (multi-app shell surface)

---

## Applications Inventory

GreenLang ships 10 compliance applications. Each application exposes a FastAPI REST API for programmatic access, health monitoring, and pipeline execution.

| # | App ID | Name | Regulation / Standard | API Prefix | Route Modules |
|---|--------|------|----------------------|------------|---------------|
| 1 | GL-CSRD-APP | CSRD Reporting Platform | EU CSRD / ESRS | `/api/v1/` | `server.py` (Pipeline, Validation, Calculation, Reporting, Materiality) |
| 2 | GL-CBAM-APP | CBAM Importer Copilot | EU CBAM 2023/956 | `/api/v1/` | `app.py`, `certificate_routes.py`, `quarterly_routes.py`, `supplier_routes.py` |
| 3 | GL-VCCI-Carbon-APP | VCCI Scope 3 Platform | GHG Protocol Scope 3 | `/api/v1/` | `main.py` + 8 agent/utility routers |
| 4 | GL-EUDR-APP | EUDR Compliance Platform | EU Deforestation Regulation | `/api/v1/` | 8 core routers + AGENT-EUDR-001 SCM router |
| 5 | GL-GHG-APP | GHG Corporate Platform | GHG Protocol Corporate | `/api/v1/` | 8 route modules (scope1, scope2, scope3, inventory, reporting, verification, target, dashboard, settings) |
| 6 | GL-ISO14064-APP | ISO 14064 Compliance | ISO 14064-1:2018 | `/api/v1/iso14064/` | 13 route modules (inventory, boundary, organization, quantification, removals, quality, verification, significance, crosswalk, management, reports, dashboard, settings) |
| 7 | GL-CDP-APP | CDP Disclosure Platform | CDP Climate Change 2024+ | `/api/v1/cdp/` | 10 route modules (questionnaires, responses, scoring, gap_analysis, benchmarking, supply_chain, transition_plan, reporting, dashboard, settings) |
| 8 | GL-TCFD-APP | TCFD Disclosure Platform | TCFD / ISSB IFRS S2 | `/api/v1/tcfd/` | 15 route modules (governance, strategy, risk_management, metrics, scenario, physical_risk, transition_risk, opportunity, financial, disclosure, issb, gap, dashboard, settings) |
| 9 | GL-SBTi-APP | SBTi Target Platform | SBTi v2.1 Criteria | `/api/v1/sbti/` | 16 route modules (targets, pathway, validation, sector, progress, temperature, recalculation, framework, review, scope3, fi, flag, gap, reporting, dashboard, settings) |
| 10 | GL-Taxonomy-APP | EU Taxonomy Platform | EU Taxonomy Regulation | `/api/v1/taxonomy/` | 16 route modules (screening, alignment, sc, dnsh, safeguards, gar, kpi, activity, portfolio, regulatory, data_quality, gap, reporting, dashboard, settings) |

---

## Unified Shell API (CBAM Pack MVP)

In addition to standalone per-app APIs, all applications are accessible through the **GreenLang Enterprise Shell** (`cbam-pack-mvp/src/cbam_pack/web/app.py`). The shell exposes a unified REST surface for multi-app pipeline execution, run management, governance, and administration.

See: [CBAM Pack Endpoints](../cbam-pack/endpoints.md)

---

## Common Patterns

### Authentication

All application APIs support one or more of the following authentication methods:

| Method | Header | Used By |
|--------|--------|---------|
| JWT Bearer Token | `Authorization: Bearer <token>` | GL-VCCI, GL-EUDR (SEC-001 JWT + SEC-002 RBAC) |
| API Key | `x-api-key: <key>` | CBAM Pack shell endpoints |
| None (development) | -- | Standalone dev mode with docs enabled |

### Rate Limiting

All applications use `slowapi` for rate limiting.

| Endpoint Category | Default Limit |
|-------------------|---------------|
| Health / Info | Unlimited |
| Pipeline Execution | 10 requests/minute |
| Validation / Calculation | 60 requests/minute |
| General API | 100 requests/minute |
| CBAM Pack Shell | 60 requests/minute (per IP) |

### Standard Error Response

All applications return errors in a consistent JSON envelope:

```json
{
  "error": "validation_error",
  "message": "Human-readable error description",
  "path": "/api/v1/pipeline/run",
  "request_id": "abc123"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (async job queued) |
| 207 | Multi-Status (partial batch success) |
| 400 | Bad Request -- invalid input |
| 401 | Unauthorized -- missing or invalid auth |
| 404 | Not Found |
| 413 | Payload Too Large (>10 MB upload) |
| 422 | Unprocessable Entity -- validation error |
| 429 | Too Many Requests -- rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- backend not configured |

### Health Check Endpoints

Every application exposes health check endpoints compatible with Kubernetes probes:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Basic liveness |
| `GET /health/live` | Kubernetes liveness probe |
| `GET /health/ready` | Kubernetes readiness probe (checks DB, Redis) |
| `GET /health/startup` | Kubernetes startup probe |
| `GET /health/detailed` | Full dependency health with latencies |
| `GET /metrics` | Prometheus-compatible metrics |

---

## Per-Application Documentation

- [GL-CSRD-APP](./gl-csrd-app.md) -- CSRD/ESRS Digital Reporting
- [GL-CBAM-APP](./gl-cbam-app.md) -- CBAM Importer Copilot
- [GL-VCCI-Carbon-APP](./gl-vcci-carbon-app.md) -- VCCI Scope 3 Carbon Intelligence
- [GL-EUDR-APP](./gl-eudr-app.md) -- EUDR Compliance Platform
- [GL-GHG-APP](./gl-ghg-app.md) -- GHG Corporate Platform
- [GL-ISO14064-APP](./gl-iso14064-app.md) -- ISO 14064 Compliance
- [GL-CDP-APP](./gl-cdp-app.md) -- CDP Disclosure Platform
- [GL-TCFD-APP](./gl-tcfd-app.md) -- TCFD Disclosure Platform
- [GL-SBTi-APP](./gl-sbti-app.md) -- SBTi Target Platform
- [GL-Taxonomy-APP](./gl-taxonomy-app.md) -- EU Taxonomy Platform

---

## Source Files

| Application | Primary Source |
|-------------|---------------|
| GL-CSRD-APP | `applications/GL-CSRD-APP/CSRD-Reporting-Platform/api/server.py` |
| GL-CBAM-APP | `applications/GL-CBAM-APP/CBAM-Importer-Copilot/backend/app.py` |
| GL-CBAM-APP Certificate Engine | `applications/GL-CBAM-APP/CBAM-Importer-Copilot/certificate_engine/api/certificate_routes.py` |
| GL-CBAM-APP Quarterly Engine | `applications/GL-CBAM-APP/CBAM-Importer-Copilot/quarterly_engine/api/quarterly_routes.py` |
| GL-CBAM-APP Supplier Portal | `applications/GL-CBAM-APP/CBAM-Importer-Copilot/supplier_portal/api/supplier_routes.py` |
| GL-VCCI-Carbon-APP | `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/main.py` |
| GL-EUDR-APP | `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/routers.py` |
| GL-GHG-APP | `applications/GL-GHG-APP/GHG-Corporate-Platform/services/api/*.py` |
| GL-ISO14064-APP | `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/*.py` |
| GL-CDP-APP | `applications/GL-CDP-APP/CDP-Disclosure-Platform/services/api/*.py` |
| GL-TCFD-APP | `applications/GL-TCFD-APP/TCFD-Disclosure-Platform/services/api/*.py` |
| GL-SBTi-APP | `applications/GL-SBTi-APP/SBTi-Target-Platform/services/api/*.py` |
| GL-Taxonomy-APP | `applications/GL-Taxonomy-APP/EU-Taxonomy-Platform/services/api/*.py` |
| CBAM Pack Shell | `cbam-pack-mvp/src/cbam_pack/web/app.py` |
