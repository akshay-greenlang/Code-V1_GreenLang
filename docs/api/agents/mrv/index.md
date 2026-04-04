# MRV Agents API Reference -- Index

> **Base URL:** `https://api.greenlang.io`
> **Authentication:** JWT Bearer token (OAuth2, RS256)
> **Version:** 1.0
> **Total Agents:** 30 (Scope 1: 8, Scope 2: 5, Scope 3: 15, Cross-cutting: 2)

---

## Agent Registry

### Scope 1 -- Direct Emissions (8 agents)

| # | Agent ID | Name | API Prefix | Doc |
|---|----------|------|------------|-----|
| 001 | GL-MRV-SCOPE1-001 | [Stationary Combustion](./stationary_combustion.md) | `/api/v1/stationary-combustion` | AGENT-MRV-001 |
| 002 | GL-MRV-SCOPE1-002 | [Refrigerants & F-Gas](./refrigerants_fgas.md) | `/api/v1/refrigerants-fgas` | AGENT-MRV-002 |
| 003 | GL-MRV-SCOPE1-003 | Mobile Combustion | `/api/v1/mobile-combustion` | AGENT-MRV-003 |
| 004 | GL-MRV-SCOPE1-004 | [Process Emissions](./process_emissions.md) | `/api/v1/process-emissions` | AGENT-MRV-004 |
| 005 | GL-MRV-SCOPE1-005 | Fugitive Emissions | `/api/v1/fugitive-emissions` | AGENT-MRV-005 |
| 006 | GL-MRV-SCOPE1-006 | Land Use Emissions | `/api/v1/land-use-emissions` | AGENT-MRV-006 |
| 007 | GL-MRV-SCOPE1-007 | Waste Treatment Emissions | `/api/v1/waste-treatment-emissions` | AGENT-MRV-007 |
| 008 | GL-MRV-SCOPE1-008 | Agricultural Emissions | `/api/v1/agricultural-emissions` | AGENT-MRV-008 |

### Scope 2 -- Indirect Energy Emissions (5 agents)

| # | Agent ID | Name | API Prefix | Doc |
|---|----------|------|------------|-----|
| 009 | GL-MRV-SCOPE2-001 | [Scope 2 Location-Based](./scope2_location.md) | `/api/v1/scope2-location` | AGENT-MRV-009 |
| 010 | GL-MRV-SCOPE2-002 | [Scope 2 Market-Based](./scope2_market.md) | `/api/v1/scope2-market` | AGENT-MRV-010 |
| 011 | GL-MRV-SCOPE2-003 | Steam & Heat Purchase | `/api/v1/steam-heat-purchase` | AGENT-MRV-011 |
| 012 | GL-MRV-SCOPE2-004 | Cooling Purchase | `/api/v1/cooling-purchase` | AGENT-MRV-012 |
| 013 | GL-MRV-SCOPE2-005 | Dual Reporting Reconciliation | `/api/v1/dual-reporting-reconciliation` | AGENT-MRV-013 |

### Scope 3 -- Value Chain Emissions (15 agents)

| # | Agent ID | Name | API Prefix | Doc |
|---|----------|------|------------|-----|
| 014 | GL-MRV-S3-001 | [Purchased Goods & Services](./purchased_goods_services.md) | `/api/v1/purchased-goods` | AGENT-MRV-014 |
| 015 | GL-MRV-S3-002 | Capital Goods | `/api/v1/capital-goods` | AGENT-MRV-015 |
| 016 | GL-MRV-S3-003 | Fuel & Energy Activities | `/api/v1/fuel-energy-activities` | AGENT-MRV-016 |
| 017 | GL-MRV-S3-004 | [Upstream Transportation](./upstream_transportation.md) | `/api/v1/upstream-transportation` | AGENT-MRV-017 |
| 018 | GL-MRV-S3-005 | Waste Generated | `/api/v1/waste-generated` | AGENT-MRV-018 |
| 019 | GL-MRV-S3-006 | [Business Travel](./business_travel.md) | `/api/v1/business-travel` | AGENT-MRV-019 |
| 020 | GL-MRV-S3-007 | Employee Commuting | `/api/v1/employee-commuting` | AGENT-MRV-020 |
| 021 | GL-MRV-S3-008 | Upstream Leased Assets | `/api/v1/upstream-leased-assets` | AGENT-MRV-021 |
| 022 | GL-MRV-S3-009 | Downstream Transportation | `/api/v1/downstream-transportation` | AGENT-MRV-022 |
| 023 | GL-MRV-S3-010 | Processing of Sold Products | `/api/v1/processing-sold-products` | AGENT-MRV-023 |
| 024 | GL-MRV-S3-011 | Use of Sold Products | `/api/v1/use-of-sold-products` | AGENT-MRV-024 |
| 025 | GL-MRV-S3-012 | End-of-Life Treatment | `/api/v1/end-of-life-treatment` | AGENT-MRV-025 |
| 026 | GL-MRV-S3-013 | Downstream Leased Assets | `/api/v1/downstream-leased-assets` | AGENT-MRV-026 |
| 027 | GL-MRV-S3-014 | Franchises | `/api/v1/franchises` | AGENT-MRV-027 |
| 028 | GL-MRV-S3-015 | Investments | `/api/v1/investments` | AGENT-MRV-028 |

### Cross-Cutting (2 agents)

| # | Agent ID | Name | API Prefix | Doc |
|---|----------|------|------------|-----|
| 029 | GL-MRV-X-041 | Category Mapper | `/api/v1/category-mapper` | AGENT-MRV-029 |
| 030 | GL-MRV-X-042 | [Audit Trail & Lineage](./audit_trail_lineage.md) | `/api/v1/audit-trail-lineage` | AGENT-MRV-030 |

---

## Common API Patterns

All 30 MRV agents share the following conventions.

### Authentication

```http
Authorization: Bearer <jwt_token>
```

All endpoints except `/health` require a valid JWT Bearer token issued via the GreenLang Auth service (`/api/v1/auth/token`).

### Standard Endpoint Layout

Every MRV agent exposes exactly 20 endpoints organized into functional groups:

| Group | Typical Endpoints | Count |
|-------|-------------------|-------|
| Calculations | `/calculate`, `/calculate/batch`, `/calculations`, `/calculations/{id}` | 4-5 |
| Entity Management | `/fuels`, `/equipment`, `/facilities`, `/factors`, etc. | 4-6 |
| Compliance | `/compliance/check`, `/compliance/{id}` | 1-2 |
| Uncertainty | `/uncertainty` | 1 |
| Aggregation | `/aggregations`, `/hot-spots` | 1-2 |
| Export / Reports | `/export`, `/reports` | 0-1 |
| Audit Trail | `/audit/{calc_id}` | 1 |
| Health & Stats | `/health`, `/stats` | 2 |

### Zero-Hallucination Calculations

Every MRV agent follows GreenLang's zero-hallucination principle:

- All emission calculations use deterministic formulas (Python `Decimal` arithmetic).
- No LLM calls in the calculation path.
- Every result includes a `provenance_hash` (SHA-256) for tamper-evidence and audit trail integrity.

### Provenance Hash

Every calculation response includes a `provenance_hash` field containing a SHA-256 digest of all calculation inputs and outputs. This enables end-to-end auditability and tamper detection.

```json
{
  "provenance_hash": "sha256:a1b2c3d4e5f6..."
}
```

### Pagination

List endpoints use offset-based pagination:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` or `page_size` | integer | 20-100 | Maximum items per page |
| `offset` or `skip` or `page` | integer | 0 or 1 | Page offset or number |

### Error Responses

All MRV agents return consistent error responses:

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error (missing fields, invalid ranges) |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, entity, or resource not found |
| 422 | Unprocessable Entity -- semantic validation failure |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- agent service not initialized |

```json
{
  "detail": "Calculation calc_abc123 not found"
}
```

### Compliance Frameworks

MRV agents check against up to 7 regulatory frameworks:

| Framework | Description |
|-----------|-------------|
| GHG Protocol | Corporate Accounting and Reporting Standard |
| ISO 14064 | Greenhouse gas quantification and reporting |
| CSRD / ESRS E1 | EU Corporate Sustainability Reporting Directive |
| EPA 40 CFR Part 98 | US EPA Mandatory GHG Reporting Rule |
| UK SECR | UK Streamlined Energy and Carbon Reporting |
| CDP | Carbon Disclosure Project questionnaire |
| EU ETS | EU Emissions Trading System |

### Uncertainty Analysis

All calculation agents support Monte Carlo and/or analytical uncertainty analysis:

```json
{
  "calculation_id": "calc_abc123",
  "method": "monte_carlo",
  "iterations": 10000
}
```

---

## Quick Start (Python)

```python
import requests

BASE = "https://api.greenlang.io"

# Authenticate
token = requests.post(f"{BASE}/api/v1/auth/token", data={
    "grant_type": "client_credentials",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
}).json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Scope 1: Stationary Combustion
calc = requests.post(f"{BASE}/api/v1/stationary-combustion/calculate", headers=headers, json={
    "fuel_type": "natural_gas",
    "quantity": 10000.0,
    "unit": "therms",
    "facility_id": "facility_hq",
}).json()

print(f"Emissions: {calc['total_co2e_kg']} kg CO2e")
print(f"Provenance: {calc['provenance_hash']}")

# Scope 2: Location-Based
scope2 = requests.post(f"{BASE}/api/v1/scope2-location/calculations", headers=headers, json={
    "facility_id": "facility_hq",
    "energy_type": "electricity",
    "consumption_kwh": 500000.0,
    "grid_region": "US-RFCW",
    "year": 2025,
}).json()

print(f"Scope 2: {scope2['total_co2e_kg']} kg CO2e")
```

---

## Documented Agents (Top 10)

| # | Agent | Scope | Documentation |
|---|-------|-------|---------------|
| 1 | Stationary Combustion | Scope 1 | [stationary_combustion.md](./stationary_combustion.md) |
| 2 | Refrigerants & F-Gas | Scope 1 | [refrigerants_fgas.md](./refrigerants_fgas.md) |
| 3 | Process Emissions | Scope 1 | [process_emissions.md](./process_emissions.md) |
| 4 | Scope 2 Location-Based | Scope 2 | [scope2_location.md](./scope2_location.md) |
| 5 | Scope 2 Market-Based | Scope 2 | [scope2_market.md](./scope2_market.md) |
| 6 | Purchased Goods & Services | Scope 3 Cat 1 | [purchased_goods_services.md](./purchased_goods_services.md) |
| 7 | Upstream Transportation | Scope 3 Cat 4 | [upstream_transportation.md](./upstream_transportation.md) |
| 8 | Business Travel | Scope 3 Cat 6 | [business_travel.md](./business_travel.md) |
| 9 | Audit Trail & Lineage | Cross-cutting | [audit_trail_lineage.md](./audit_trail_lineage.md) |
| 10 | Mobile Combustion | Scope 1 | [mobile_combustion.md](./mobile_combustion.md) |
