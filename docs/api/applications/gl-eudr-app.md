# GL-EUDR-APP -- EUDR Compliance Platform API Reference

**Source (router registration):** `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/routers.py`
**Route modules:** `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/*_routes.py`
**Version:** 1.0

---

## Overview

The GL-EUDR-APP provides a REST API for EU Deforestation Regulation (EUDR) compliance. It manages supplier onboarding, geolocation plot tracking, due diligence statements (DDS), document management, pipeline orchestration, risk assessment, and dashboards.

The API is assembled via `register_all_routers(app)` which mounts 8 core platform routers and optionally the AGENT-EUDR-001 Supply Chain Mapper router (25+ endpoints).

**Authentication:** SEC-001 JWT + SEC-002 RBAC permission checks on all routes.

---

## Core Platform Routers (8)

| Prefix | Tag | Module | Description |
|--------|-----|--------|-------------|
| `/api/v1/suppliers` | Suppliers | `supplier_routes.py` | Supplier CRUD, bulk import, compliance and risk summaries |
| `/api/v1/plots` | Plots | `plot_routes.py` | Geolocation plot management |
| `/api/v1/dds` | Due Diligence | `dds_routes.py` | Due diligence statement lifecycle |
| `/api/v1/documents` | Documents | `document_routes.py` | Document upload and management |
| `/api/v1/pipeline` | Pipeline | `pipeline_routes.py` | EUDR pipeline orchestration |
| `/api/v1/risk` | Risk | `risk_routes.py` | Risk assessment and scoring |
| `/api/v1/dashboard` | Dashboard | `dashboard_routes.py` | Dashboard aggregations |
| `/api/v1/settings` | Settings | `settings_routes.py` | Application configuration |

---

## Supplier Endpoints

**Prefix:** `/api/v1/suppliers`

| Method | Path | Summary |
|--------|------|---------|
| POST | `/` | Create a new supplier |
| GET | `/` | List suppliers (paginated) |
| PUT | `/{supplier_id}` | Update supplier |
| GET | `/{supplier_id}` | Get supplier details |
| POST | `/bulk-import` | Bulk import suppliers |
| GET | `/compliance-summary` | Compliance status overview |
| GET | `/risk-summary` | Risk assessment summary |
| DELETE | `/{supplier_id}` | Deactivate supplier |

### POST /api/v1/suppliers

**Request Body:**

```json
{
  "name": "Amazonia Timber Co.",
  "country": "BR",
  "tax_id": "12.345.678/0001-90",
  "commodities": ["timber", "soy"],
  "address": {
    "city": "Manaus",
    "country": "BR"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Legal entity name (1-255 chars) |
| `country` | string | Yes | ISO country code (2 or 3-letter) |
| `tax_id` | string | No | Tax identification number |
| `commodities` | array[string] | Yes | EUDR commodities (min 1) |
| `address` | object | No | Business address |

**Valid commodities:** `cattle`, `cocoa`, `coffee`, `oil_palm`, `rubber`, `soya`, `wood`

**Response (201):**

```json
{
  "supplier_id": "sup_abc123",
  "name": "Amazonia Timber Co.",
  "country": "BR",
  "commodities": ["timber", "soy"],
  "compliance_status": "pending",
  "risk_level": null,
  "created_at": "2026-04-04T12:00:00Z"
}
```

---

## Supply Chain Mapper (AGENT-EUDR-001)

**Prefix:** `/v1/supply-chain/v1/eudr-scm`
**Tag:** EUDR Supply Chain Mapper
**Endpoints:** 25+

When enabled (`scm_enabled=True`), the AGENT-EUDR-001 Supply Chain Mapper provides:

| Sub-prefix | Description |
|------------|-------------|
| `/graphs` | Supply chain graph construction and querying |
| `/mapping` | Commodity-to-supplier mapping |
| `/traceability` | End-to-end traceability records |
| `/risk` | Supply chain risk scoring |
| `/gaps` | Gap identification in supply chains |
| `/visualization` | Graph visualization data |
| `/onboarding` | Supplier onboarding workflows |

The SCM router is imported from `greenlang.agents.eudr.supply_chain_mapper.api.router` and is optional. If the module is not installed, the routes are silently skipped.

---

## Source Files

- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/routers.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/supplier_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/plot_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/dds_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/document_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/pipeline_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/risk_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/dashboard_routes.py`
- `applications/GL-EUDR-APP/EUDR-Compliance-Platform/services/api/settings_routes.py`
