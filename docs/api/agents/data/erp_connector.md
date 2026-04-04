# ERP/Finance Connector API Reference

**Agent:** AGENT-DATA-003
**Prefix:** `/api/v1/erp-connector`
**Source:** `greenlang/agents/data/erp_connector/api/router.py`
**Status:** Production Ready

## Overview

The ERP/Finance Connector agent integrates with enterprise resource planning systems (SAP, Oracle, NetSuite, Dynamics) for spend data sync, purchase order sync, inventory sync, vendor/material mapping to Scope 3 categories, and Scope 3 emission calculation.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/connections` | Register ERP connection | Yes |
| 2 | GET | `/v1/connections` | List connections | Yes |
| 3 | GET | `/v1/connections/{connection_id}` | Get connection details | Yes |
| 4 | POST | `/v1/connections/{connection_id}/test` | Test connectivity | Yes |
| 5 | DELETE | `/v1/connections/{connection_id}` | Remove connection | Yes |
| 6 | POST | `/v1/spend/sync` | Sync spend data | Yes |
| 7 | GET | `/v1/spend` | Query spend records | Yes |
| 8 | GET | `/v1/spend/summary` | Get spend summary | Yes |
| 9 | POST | `/v1/purchase-orders/sync` | Sync purchase orders | Yes |
| 10 | GET | `/v1/purchase-orders` | Query purchase orders | Yes |
| 11 | GET | `/v1/purchase-orders/{po_number}` | Get single purchase order | Yes |
| 12 | POST | `/v1/inventory/sync` | Sync inventory | Yes |
| 13 | GET | `/v1/inventory` | Query inventory | Yes |
| 14 | POST | `/v1/mappings/vendors` | Map vendor to Scope 3 category | Yes |
| 15 | GET | `/v1/mappings/vendors` | List vendor mappings | Yes |
| 16 | POST | `/v1/mappings/materials` | Map material to Scope 3 category | Yes |
| 17 | POST | `/v1/emissions/calculate` | Calculate Scope 3 emissions | Yes |
| 18 | GET | `/v1/emissions/summary` | Get emissions summary | Yes |
| 19 | GET | `/v1/statistics` | Get service statistics | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 1. Register ERP Connection

```http
POST /api/v1/erp-connector/v1/connections
```

**Request Body:**

```json
{
  "erp_system": "sap",
  "host": "sap-prod.example.com",
  "port": 443,
  "username": "greenlang_svc",
  "password": "encrypted_password",
  "tenant_id": "tenant_abc",
  "database_name": "PROD",
  "connection_params": {
    "client": "100",
    "language": "EN"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `erp_system` | string | Yes | ERP type: `sap`, `oracle`, `netsuite`, `dynamics`, `simulated` |
| `host` | string | Yes | ERP host address or URL |
| `port` | integer | Optional | Connection port (default: 443) |
| `username` | string | Yes | Authentication username |
| `password` | string | Optional | Password (stored encrypted) |
| `tenant_id` | string | Optional | Tenant identifier |
| `database_name` | string | Optional | Database or instance name |
| `connection_params` | object | Optional | Additional connection parameters |

### 17. Calculate Scope 3 Emissions

```http
POST /api/v1/erp-connector/v1/emissions/calculate
```

**Request Body:**

```json
{
  "connection_id": "conn_abc123",
  "start_date": "2026-01-01",
  "end_date": "2026-03-31",
  "methodology": "spend_based",
  "scope3_categories": ["cat1_purchased_goods", "cat4_upstream_transport"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connection_id` | string | Yes | ERP connection with synced data |
| `start_date` | string | Yes | Calculation period start (ISO 8601) |
| `end_date` | string | Yes | Calculation period end (ISO 8601) |
| `methodology` | string | Optional | Method: `eeio`, `spend_based`, `hybrid`, `supplier_specific` |
| `scope3_categories` | string[] | Optional | Scope 3 category filter |

**Response:**

```json
{
  "calculation_id": "calc_xyz789",
  "total_co2e_tonnes": 1250.75,
  "methodology": "spend_based",
  "by_category": {
    "cat1_purchased_goods": 980.50,
    "cat4_upstream_transport": 270.25
  },
  "vendor_count": 145,
  "record_count": 2340,
  "provenance_hash": "sha256:..."
}
```
