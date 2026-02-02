# API Reference Guide

## GL-VCCI Scope 3 Platform API

Complete API reference organized by functional area with code examples and common workflows.

---

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Common Workflows](#common-workflows)
- [Supplier Management](#supplier-management)
- [Procurement Data](#procurement-data)
- [Emissions Calculations](#emissions-calculations)
- [PCF Exchange](#pcf-exchange)
- [Reporting](#reporting)
- [Administration](#administration)

---

## Base URL

```
Production: https://api.vcci.greenlang.io/v2
Staging: https://api-staging.vcci.greenlang.io/v2
```

---

## Authentication

All requests require authentication via:
- **API Key:** Include `X-API-Key` header
- **OAuth 2.0:** Include `Authorization: Bearer TOKEN` header

See [Authentication Guide](./AUTHENTICATION.md) for details.

---

## Common Workflows

### Workflow 1: Complete Scope 3 Calculation

```python
# 1. Create/resolve supplier
resolution = client.suppliers.resolve(supplier_name="Acme Corporation")
supplier_id = resolution.matched_supplier_id

# 2. Create transaction
transaction = client.procurement.create_transaction(
    transaction_type="purchase_order",
    transaction_id="PO-2024-001",
    transaction_date="2024-11-01",
    supplier_name="Acme Corporation",
    product_name="Steel beams",
    quantity=1000,
    unit="kg",
    spend_usd=5000.00
)

# 3. Calculate emissions
calculation = client.emissions.calculate(
    transactions=[transaction.id],
    gwp_standard="AR6"
)

# 4. Generate report
report = client.reports.generate(
    report_type="esrs_e1",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### Workflow 2: Bulk Data Import

```python
# 1. Upload CSV file
with open("procurement_2024.csv", "rb") as f:
    job = client.procurement.upload_file(f, file_type="csv")

# 2. Wait for processing
completed_job = client.procurement.wait_for_job(job.job_id)

# 3. Batch calculate emissions
transactions = client.procurement.list_transactions(limit=1000)
calculation = client.emissions.calculate(
    transactions=[t.id for t in transactions]
)

# 4. Get aggregated results
aggregate = client.emissions.aggregate(
    date_from="2024-01-01",
    date_to="2024-12-31",
    group_by=["category", "supplier"]
)
```

### Workflow 3: Supplier PCF Request

```python
# 1. Create engagement campaign
campaign = client.engagement.create_campaign(
    name="Q4 PCF Request",
    type="pcf_request",
    target_suppliers=["sup_1", "sup_2"],
    message_template="Dear {supplier_name}, we request your PCF data..."
)

# 2. Grant portal access
for supplier_id in campaign.target_suppliers:
    access = client.engagement.grant_portal_access(
        supplier_id=supplier_id,
        expiration_days=90
    )

# 3. Monitor responses (via webhooks or polling)
# When PCF received, tier upgrades automatically
```

---

## Supplier Management

### Resolve Supplier Entity

**Purpose:** Match ambiguous supplier names to canonical entities

**Endpoint:** `POST /v2/suppliers/resolve`

**Request:**
```json
{
  "supplier_name": "ACME CORP",
  "hints": {
    "country": "US",
    "lei": "529900XYZU1234567890"
  },
  "confidence_threshold": 0.90,
  "max_candidates": 5
}
```

**Response:**
```json
{
  "status": "auto_matched",
  "matched_supplier_id": "sup_abc123",
  "confidence": 0.98,
  "candidates": [
    {
      "supplier_id": "sup_abc123",
      "canonical_name": "Acme Corporation",
      "confidence": 0.98,
      "matching_features": [
        "name_similarity: 0.95",
        "lei_match: exact"
      ]
    }
  ]
}
```

**Python Example:**
```python
resolution = client.suppliers.resolve(
    supplier_name="ACME CORP",
    hints={"country": "US"}
)

if resolution.status == "auto_matched":
    print(f"Matched: {resolution.matched_supplier_id}")
else:
    print("Human review required")
```

**JavaScript Example:**
```javascript
const resolution = await client.suppliers.resolve({
  supplier_name: 'ACME CORP',
  hints: { country: 'US' }
});

if (resolution.status === 'auto_matched') {
  console.log(`Matched: ${resolution.matched_supplier_id}`);
}
```

**cURL Example:**
```bash
curl -X POST https://api.vcci.greenlang.io/v2/suppliers/resolve \
  -H "X-API-Key: $VCCI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "supplier_name": "ACME CORP",
    "hints": {"country": "US"}
  }'
```

---

### Batch Resolve Suppliers

**Purpose:** Resolve multiple suppliers efficiently

**Endpoint:** `POST /v2/suppliers/batch/resolve`

**Request:**
```json
{
  "suppliers": [
    {
      "id": "row_001",
      "supplier_name": "ACME CORP",
      "hints": {"country": "US"}
    },
    {
      "id": "row_002",
      "supplier_name": "Steel Inc"
    }
  ],
  "confidence_threshold": 0.90
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "row_001",
      "status": "auto_matched",
      "matched_supplier_id": "sup_abc123",
      "confidence": 0.98
    },
    {
      "id": "row_002",
      "status": "human_review_required",
      "confidence": 0.75
    }
  ],
  "summary": {
    "total": 2,
    "auto_matched": 1,
    "human_review_required": 1,
    "no_match": 0
  }
}
```

---

### Enrich Supplier Data

**Purpose:** Add external data (LEI, DUNS, OpenCorporates)

**Endpoint:** `POST /v2/suppliers/{supplier_id}/enrich`

**Request:**
```json
{
  "sources": ["lei", "duns"]
}
```

**Response:**
```json
{
  "id": "sup_abc123",
  "canonical_name": "Acme Corporation",
  "enrichment": {
    "headquarters": {
      "country": "US",
      "city": "New York",
      "address": "123 Main St"
    },
    "industry": "Manufacturing",
    "revenue_usd": 50000000,
    "employee_count": 500,
    "parent_company": "Acme Holdings Inc"
  }
}
```

---

## Procurement Data

### Upload File

**Purpose:** Bulk import procurement data from CSV/Excel

**Endpoint:** `POST /v2/procurement/transactions/upload`

**Request (multipart/form-data):**
```
file: procurement_2024.csv (binary)
file_type: csv
column_mapping: {"PO Number": "transaction_id", "Vendor": "supplier_name"}
skip_rows: 1
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "file_name": "procurement_2024.csv",
  "file_size_bytes": 2048576,
  "estimated_rows": 5000,
  "status": "queued"
}
```

**Python Example:**
```python
with open("procurement_2024.csv", "rb") as f:
    job = client.procurement.upload_file(
        file=f,
        file_type="csv",
        column_mapping={
            "PO Number": "transaction_id",
            "Vendor": "supplier_name",
            "Amount": "spend_usd"
        },
        skip_rows=1
    )

# Wait for completion
completed = client.procurement.wait_for_job(job.job_id)
print(f"Processed: {completed.results.success} rows")
```

---

### Get Job Status

**Purpose:** Check batch processing status

**Endpoint:** `GET /v2/procurement/jobs/{job_id}`

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "progress": {
    "processed": 2500,
    "total": 5000,
    "percentage": 50
  },
  "created_at": "2024-11-06T10:00:00Z",
  "started_at": "2024-11-06T10:00:15Z"
}
```

**Completed Response:**
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "progress": {
    "processed": 5000,
    "total": 5000,
    "percentage": 100
  },
  "results": {
    "success": 4850,
    "errors": 150,
    "warnings": 200
  },
  "completed_at": "2024-11-06T10:05:30Z"
}
```

---

## Emissions Calculations

### Calculate Emissions

**Purpose:** Calculate Scope 3 emissions with uncertainty

**Endpoint:** `POST /v2/emissions/calculate`

**Request:**
```json
{
  "transactions": [
    {"transaction_id": "txn_abc123"},
    {"transaction_id": "txn_def456"}
  ],
  "options": {
    "gwp_standard": "AR6",
    "uncertainty_method": "monte_carlo",
    "monte_carlo_iterations": 10000
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "calc_abc123",
      "transaction_id": "txn_abc123",
      "category": "1",
      "tier": "2",
      "emissions_kg_co2e": 1250.5,
      "uncertainty": {
        "method": "monte_carlo",
        "range": {
          "lower_bound": 1000.4,
          "upper_bound": 1500.6
        },
        "confidence_level": 0.95
      },
      "data_quality": {
        "dqi_score": 3.8,
        "pedigree": "good"
      },
      "emission_factor": {
        "source": "ecoinvent",
        "source_version": "3.10",
        "factor_name": "Steel, hot rolled",
        "value": 1.25,
        "unit": "kg CO2e per kg"
      }
    }
  ],
  "summary": {
    "total_emissions_kg_co2e": 2501.0,
    "uncertainty_range": {
      "lower_bound": 2000.8,
      "upper_bound": 3001.2
    },
    "tier_distribution": {
      "tier_1_count": 0,
      "tier_2_count": 2,
      "tier_3_count": 0
    }
  }
}
```

---

### Aggregated Emissions

**Purpose:** Get aggregated emissions with flexible grouping

**Endpoint:** `GET /v2/emissions/aggregate`

**Query Parameters:**
- `date_from` (required): Start date (YYYY-MM-DD)
- `date_to` (required): End date (YYYY-MM-DD)
- `group_by`: Comma-separated fields (category, supplier, product, month, quarter, year)
- `supplier_id`: Filter by supplier
- `category`: Filter by Scope 3 category

**Request:**
```
GET /v2/emissions/aggregate?date_from=2024-01-01&date_to=2024-12-31&group_by=category,supplier
```

**Response:**
```json
{
  "data": [
    {
      "group": {
        "category": "1",
        "supplier": "Acme Corporation"
      },
      "total_emissions_kg_co2e": 50000.0,
      "transaction_count": 100,
      "uncertainty_range": {
        "lower_bound": 45000.0,
        "upper_bound": 55000.0
      },
      "avg_dqi_score": 3.7
    }
  ],
  "summary": {
    "total_emissions_kg_co2e": 150000.0,
    "date_from": "2024-01-01",
    "date_to": "2024-12-31"
  }
}
```

---

## PCF Exchange

### Import Supplier PCFs

**Purpose:** Import PACT Pathfinder or Catena-X PCFs

**Endpoint:** `POST /v2/pcf/import`

**Request:**
```json
{
  "format": "pact",
  "pcfs": [
    {
      "id": "pcf_supplier_001",
      "productId": "PROD-12345",
      "productName": "Steel beams",
      "pcf": {
        "declaredUnit": "kilogram",
        "unitary": true,
        "pCfExcludingBiogenic": 850.2
      },
      "companyIds": ["urn:epc:id:sgln:529900.XYZU.0"],
      "validityPeriodStart": "2024-01-01T00:00:00Z",
      "validityPeriodEnd": "2024-12-31T23:59:59Z"
    }
  ],
  "validate_only": false
}
```

**Response:**
```json
{
  "imported": 1,
  "errors": []
}
```

---

### Export PCFs

**Purpose:** Generate PCF export in PACT/Catena-X format

**Endpoint:** `POST /v2/pcf/export`

**Request:**
```json
{
  "product_ids": ["prod_1", "prod_2"],
  "format": "pact",
  "include_verification": true
}
```

**Response:**
```json
{
  "export_id": "export_abc123",
  "format": "pact",
  "pcfs": [
    {
      "id": "pcf_001",
      "productId": "PROD-12345",
      "productName": "Steel beams",
      "pcf": {
        "declaredUnit": "kilogram",
        "unitary": true,
        "pCfExcludingBiogenic": 850.2
      }
    }
  ],
  "generated_at": "2024-11-06T10:00:00Z"
}
```

---

## Reporting

### Generate Report

**Purpose:** Create standards-compliant reports (ESRS, CDP, IFRS S2)

**Endpoint:** `POST /v2/reports/generate`

**Request:**
```json
{
  "report_type": "esrs_e1",
  "reporting_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "output_format": "pdf",
  "include_charts": true,
  "categories": ["1", "4", "6"]
}
```

**Response:**
```json
{
  "report_id": "rpt_abc123",
  "status": "queued",
  "estimated_completion": "2024-11-06T10:05:00Z"
}
```

---

### Download Report

**Purpose:** Get generated report download URL

**Endpoint:** `GET /v2/reports/{report_id}`

**Response (completed):**
```json
{
  "report_id": "rpt_abc123",
  "status": "completed",
  "report_type": "esrs_e1",
  "reporting_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "download_url": "https://reports.vcci.greenlang.io/...",
  "generated_at": "2024-11-06T10:03:45Z"
}
```

**Python Example:**
```python
# Generate report
report = client.reports.generate(
    report_type="esrs_e1",
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_format="pdf"
)

# Wait for completion
completed = client.reports.wait_for_report(report.report_id)

# Download
report_data = client.reports.download(report.report_id)
with open("esrs_e1_2024.pdf", "wb") as f:
    f.write(report_data)
```

---

## Administration

### Create User

**Purpose:** Add user to tenant

**Endpoint:** `POST /v2/admin/users`

**Request:**
```json
{
  "email": "newuser@company.com",
  "name": "John Doe",
  "role": "user"
}
```

**Response:**
```json
{
  "id": "user_abc123",
  "email": "newuser@company.com",
  "name": "John Doe",
  "role": "user",
  "tenant_id": "tenant_xyz789",
  "created_at": "2024-11-06T10:00:00Z"
}
```

---

### List API Keys

**Purpose:** View all API keys for tenant

**Endpoint:** `GET /v2/auth/api-keys`

**Response:**
```json
{
  "data": [
    {
      "id": "key_abc123",
      "name": "Production Key",
      "prefix": "sk_live_abc",
      "scopes": ["read", "write"],
      "created_at": "2024-01-15T10:00:00Z",
      "expires_at": "2025-01-15T10:00:00Z",
      "last_used_at": "2024-11-06T09:30:00Z"
    }
  ]
}
```

---

## Error Responses

All errors follow consistent format:

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "Invalid input parameters",
    "details": [
      {
        "field": "supplier_name",
        "message": "Required field missing"
      }
    ]
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `BAD_REQUEST` | 400 | Invalid input |
| `UNAUTHORIZED` | 401 | Invalid auth credentials |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limiting

All responses include rate limit headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1699275600
```

See [Rate Limits Guide](./RATE_LIMITS.md) for optimization strategies.

---

## Pagination

List endpoints support pagination:

**Request:**
```
GET /v2/suppliers?limit=100&offset=200
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "limit": 100,
    "offset": 200,
    "total": 1543,
    "has_more": true
  }
}
```

---

## Best Practices

### 1. Use Batch Endpoints

```python
# ❌ Bad: 1000 requests
for txn in transactions:
    create_transaction(txn)

# ✅ Good: 1 request
batch_create_transactions(transactions)
```

### 2. Implement Retry Logic

```python
from time import sleep

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            sleep(int(response.headers.get('Retry-After', 60)))
            continue
        return response
```

### 3. Cache Frequently Accessed Data

```python
# Cache supplier lookups
supplier_cache = {}

def get_supplier_cached(supplier_id):
    if supplier_id not in supplier_cache:
        supplier_cache[supplier_id] = client.suppliers.get(supplier_id)
    return supplier_cache[supplier_id]
```

---

## Resources

- **OpenAPI Spec:** [openapi.yaml](./openapi.yaml)
- **Authentication:** [AUTHENTICATION.md](./AUTHENTICATION.md)
- **Rate Limits:** [RATE_LIMITS.md](./RATE_LIMITS.md)
- **Webhooks:** [WEBHOOKS.md](./WEBHOOKS.md)
- **Quickstart:** [integrations/QUICKSTART.md](./integrations/QUICKSTART.md)
- **Python SDK:** [integrations/PYTHON_SDK.md](./integrations/PYTHON_SDK.md)
- **JavaScript SDK:** [integrations/JAVASCRIPT_SDK.md](./integrations/JAVASCRIPT_SDK.md)

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
