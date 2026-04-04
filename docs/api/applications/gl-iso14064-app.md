# GL-ISO14064-APP -- ISO 14064 Compliance Platform API Reference

**Source:** `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/*.py`
**Version:** 1.0

---

## Overview

The GL-ISO14064-APP provides a REST API for ISO 14064-1:2018 GHG inventory compliance. It manages organization profiles, inventory lifecycle (draft through published), organizational and operational boundaries, emission quantification, GHG removals, data quality management, verification workflows, significance testing, framework crosswalks, management plans, and reporting.

---

## Route Modules (13)

| Prefix | Tag | Module | Endpoints | Description |
|--------|-----|--------|-----------|-------------|
| `/api/v1/iso14064/inventories` | Inventories | `inventory_routes.py` | 5 | Inventory CRUD and status transitions |
| `/api/v1/iso14064/organizations` | Organizations | `organization_routes.py` | 6 | Organization profiles and facilities |
| `/api/v1/iso14064/boundaries` | Boundaries | `boundary_routes.py` | 6 | Organizational and operational boundaries |
| `/api/v1/iso14064/quantification` | Quantification | `quantification_routes.py` | 6 | Emission source quantification |
| `/api/v1/iso14064/removals` | Removals | `removals_routes.py` | 6 | GHG removal/sink tracking |
| `/api/v1/iso14064/quality` | Quality | `quality_routes.py` | 12 | Data quality management and improvement plans |
| `/api/v1/iso14064/verification` | Verification | `verification_routes.py` | 7 | Third-party verification workflow |
| `/api/v1/iso14064/significance` | Significance | `significance_routes.py` | 4 | Significance testing for emission sources |
| `/api/v1/iso14064/crosswalk` | Crosswalk | `crosswalk_routes.py` | 3 | Framework crosswalk mappings |
| `/api/v1/iso14064/management` | Management | `management_routes.py` | 7 | Management plans and corrective actions |
| `/api/v1/iso14064/reports` | Reports | `reports_routes.py` | 4 | Report generation and export |
| `/api/v1/iso14064/dashboard` | Dashboard | `dashboard_routes.py` | 1 | Dashboard aggregations |
| `/api/v1/iso14064/settings` | Settings | `settings_routes.py` | 16 | GWP sources, EF databases, thresholds, templates |

---

## Inventory Lifecycle

Inventories progress through a defined lifecycle:

```
draft -> in_review -> approved -> verified -> published
```

### Inventory Endpoints

| Method | Path | Summary |
|--------|------|---------|
| POST | `/` | Create inventory |
| GET | `/` | List inventories (paginated) |
| GET | `/{inventory_id}` | Get inventory details |
| PUT | `/{inventory_id}` | Update inventory |
| POST | `/{inventory_id}/transition` | Transition inventory status |

### POST /api/v1/iso14064/inventories

**Request Body:**

```json
{
  "org_id": "org_abc123",
  "reporting_year": 2025,
  "consolidation_approach": "operational_control",
  "gwp_source": "ar5"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `org_id` | string | Yes | Organization ID |
| `reporting_year` | integer | Yes | Reporting year (1990-2100) |
| `consolidation_approach` | string | No | `operational_control` (default), `financial_control`, `equity_share` |
| `gwp_source` | string | No | `ar5` (default), `ar6`, `custom` |

### POST /api/v1/iso14064/inventories/{inventory_id}/transition

**Request Body:**

```json
{
  "target_status": "in_review",
  "notes": "Ready for internal review"
}
```

---

## Settings Endpoints (16)

The settings module is the most granular, providing fine-grained control over ISO 14064 compliance parameters.

| Method | Path | Summary |
|--------|------|---------|
| GET | `/general` | Get general settings |
| PUT | `/general` | Update general settings |
| GET | `/gwp-sources` | List available GWP sources |
| PUT | `/gwp-source` | Set active GWP source |
| GET | `/emission-factors` | List emission factor databases |
| PUT | `/emission-factors/default` | Set default EF database |
| GET | `/thresholds` | Get threshold settings |
| PUT | `/thresholds` | Update threshold settings |
| GET | `/notification-preferences` | Get notification preferences |
| PUT | `/notification-preferences` | Update notification preferences |
| GET | `/export-templates` | List export templates |
| POST | `/export-templates` | Create export template |

---

## Source Files

- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/inventory_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/organization_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/boundary_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/quantification_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/removals_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/quality_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/verification_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/significance_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/crosswalk_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/management_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/reports_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/dashboard_routes.py`
- `applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/api/settings_routes.py`
