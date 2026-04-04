# GL-CBAM-APP -- CBAM Importer Copilot API Reference

**Source (main):** `applications/GL-CBAM-APP/CBAM-Importer-Copilot/backend/app.py`
**Source (certificates):** `applications/GL-CBAM-APP/CBAM-Importer-Copilot/certificate_engine/api/certificate_routes.py`
**Source (quarterly):** `applications/GL-CBAM-APP/CBAM-Importer-Copilot/quarterly_engine/api/quarterly_routes.py`
**Source (suppliers):** `applications/GL-CBAM-APP/CBAM-Importer-Copilot/supplier_portal/api/supplier_routes.py`
**Title:** CBAM Importer Copilot
**Version:** 1.0.0 (main), 1.1.0 (engines)

---

## Overview

The GL-CBAM-APP implements the EU Carbon Border Adjustment Mechanism (Regulation 2023/956) for importers. It comprises four API modules:

1. **Main Application** -- Health, metrics, pipeline execution, and API information
2. **Certificate Engine** -- Certificate obligation calculations, EU ETS pricing, free allocation benchmarks, carbon price deductions
3. **Quarterly Engine** -- Quarterly report generation, submission, amendment workflow, deadline monitoring
4. **Supplier Portal** -- Supplier registration, installation management, emissions data submission, importer data exchange

All calculation endpoints use deterministic Decimal arithmetic (zero hallucination guarantee).

---

## Main Application Endpoints

**Prefix:** None (root-level)

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| GET | `/` | Root -- service information | Rate limited (100/min) |
| GET | `/api/v1/info` | Application capabilities and SLA | None |
| POST | `/api/v1/pipeline/execute` | Execute CBAM pipeline | Rate limited (10/min) |
| GET | `/health` | Basic health check | None |
| GET | `/health/ready` | Readiness probe | None |
| GET | `/health/live` | Liveness probe | None |
| GET | `/metrics` | Prometheus metrics | None |

### GET /api/v1/info

**Response (200):**

```json
{
  "name": "CBAM Importer Copilot",
  "version": "1.0.0",
  "description": "EU CBAM Transitional Registry Reporting",
  "capabilities": [
    "Shipment intake and validation",
    "Emissions calculation (zero hallucination)",
    "CBAM report generation",
    "Provenance tracking",
    "Health monitoring",
    "Prometheus metrics"
  ],
  "monitoring": {
    "health_checks": true,
    "metrics": true,
    "structured_logging": true,
    "correlation_ids": true
  },
  "sla": {
    "availability": "99.9%",
    "success_rate": "99%",
    "latency_p95": "10 minutes"
  }
}
```

### POST /api/v1/pipeline/execute

**Response (200):**

```json
{
  "status": "success",
  "message": "Pipeline execution completed",
  "report_id": "CBAM-2024-Q4-001",
  "records_processed": 1000,
  "emissions_tco2": 12345.67
}
```

---

## Certificate Engine Endpoints

**Router Prefix:** `/api/v1/certificates`
**Tags:** `certificates`

### Obligation Endpoints

| Method | Path | Summary |
|--------|------|---------|
| POST | `/obligations/calculate` | Calculate annual certificate obligation |
| GET | `/obligations/{importer_id}/{year}` | Get obligation summary |
| GET | `/obligations/{importer_id}/{year}/breakdown/cn` | Breakdown by CN code |
| GET | `/obligations/{importer_id}/{year}/breakdown/country` | Breakdown by country |
| POST | `/obligations/{importer_id}/{year}/project` | Cost projection |

### Holdings Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/holdings/{importer_id}/{year}/{quarter}` | Quarterly holding check |
| POST | `/holdings/{importer_id}/{year}/record` | Record certificates held |

### EU ETS Price Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/ets-price/current` | Current ETS price |
| GET | `/ets-price/weekly/{date}` | Weekly average price |
| GET | `/ets-price/quarterly/{year}/{quarter}` | Quarterly average price |
| GET | `/ets-price/annual/{year}` | Annual average price |
| GET | `/ets-price/history` | Date range price history |
| GET | `/ets-price/trend` | Price trend analysis |
| POST | `/ets-price/manual` | Manual price entry |
| POST | `/ets-price/import` | Bulk price import |

### Free Allocation Endpoints

| Method | Path | Summary |
|--------|------|---------|
| GET | `/free-allocation/schedule` | Phase-out schedule (2026-2034) |
| GET | `/free-allocation/benchmarks` | All product benchmarks |
| GET | `/free-allocation/{cn_code}/{year}` | Specific allocation factor |
| PUT | `/free-allocation/{cn_code}/{year}` | Update benchmark |
| GET | `/free-allocation/compare` | Year-over-year comparison |

### Carbon Price Deduction Endpoints

| Method | Path | Summary |
|--------|------|---------|
| POST | `/deductions/register` | Register carbon price deduction |
| GET | `/deductions/{importer_id}/{year}` | List deductions |
| GET | `/deductions/detail/{deduction_id}` | Get specific deduction |
| POST | `/deductions/{deduction_id}/verify` | Verify deduction |
| POST | `/deductions/{deduction_id}/approve` | Approve deduction |
| POST | `/deductions/{deduction_id}/reject` | Reject deduction |
| POST | `/deductions/{deduction_id}/evidence` | Add evidence document |
| GET | `/deductions/{importer_id}/{year}/summary` | Deduction summary |
| GET | `/country-pricing/{country}` | Country carbon pricing info |

---

## Quarterly Engine Endpoints

**Router Prefix:** `/api/v1/cbam/quarterly`
**Tags:** `cbam-quarterly`

| Method | Path | Summary |
|--------|------|---------|
| GET | `/calendar/{year}` | Quarterly reporting calendar |
| GET | `/current` | Current quarter details |
| POST | `/reports/generate` | Trigger report generation |
| GET | `/reports` | List quarterly reports |
| GET | `/reports/{report_id}` | Get report details |
| GET | `/reports/{report_id}/xml` | Download XML output |
| GET | `/reports/{report_id}/summary` | Download markdown summary |
| PUT | `/reports/{report_id}/submit` | Submit report for review |
| POST | `/reports/{report_id}/amend` | Create amendment (T+60 day window) |
| GET | `/reports/{report_id}/amendments` | List amendments |
| GET | `/reports/{report_id}/amendments/{id}/diff` | Get amendment diff |
| GET | `/deadlines` | Upcoming deadlines |
| GET | `/deadlines/overdue` | Overdue reports |
| PUT | `/deadlines/{alert_id}/acknowledge` | Acknowledge alert |
| GET | `/notifications` | Notification history |
| PUT | `/notifications/configure` | Configure notification recipients |

---

## Supplier Portal Endpoints

**Router Prefix:** `/api/v1/cbam/suppliers`
**Tags:** `cbam-suppliers`

The Supplier Portal enables third-country suppliers to register installations, submit emissions data, manage data exchange with EU importers, and view dashboard analytics.

### Supplier Management

| Method | Path | Summary |
|--------|------|---------|
| POST | `/` | Register new supplier |
| GET | `/` | List suppliers (paginated) |
| PUT | `/{supplier_id}` | Update supplier profile |
| GET | `/{supplier_id}` | Get supplier details |
| POST | `/bulk-import` | Bulk supplier import |
| GET | `/compliance-summary` | Compliance status summary |
| GET | `/risk-summary` | Risk assessment summary |
| DELETE | `/{supplier_id}` | Deactivate supplier |

CBAM product groups: `cement`, `steel`, `aluminum`, `fertilizers`, `hydrogen`, `electricity`.

---

## Source Files

- `applications/GL-CBAM-APP/CBAM-Importer-Copilot/backend/app.py`
- `applications/GL-CBAM-APP/CBAM-Importer-Copilot/certificate_engine/api/certificate_routes.py`
- `applications/GL-CBAM-APP/CBAM-Importer-Copilot/quarterly_engine/api/quarterly_routes.py`
- `applications/GL-CBAM-APP/CBAM-Importer-Copilot/supplier_portal/api/supplier_routes.py`
