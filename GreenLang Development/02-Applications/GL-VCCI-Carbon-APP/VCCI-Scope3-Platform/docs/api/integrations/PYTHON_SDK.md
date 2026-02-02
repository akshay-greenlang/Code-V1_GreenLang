# Python SDK Guide

## GL-VCCI Python SDK

Official Python SDK for the GL-VCCI Scope 3 Carbon Intelligence Platform.

---

## Installation

```bash
pip install vcci-python
```

Or install from source:

```bash
git clone https://github.com/greenlang/vcci-python.git
cd vcci-python
pip install -e .
```

**Requirements:** Python 3.10+

---

## Quick Start

```python
from vcci import Client

# Initialize client
client = Client(api_key="sk_live_your_key_here")

# Or use OAuth
client = Client(
    client_id="vcci_client_abc123",
    client_secret="sk_live_secret123"
)

# Create supplier
supplier = client.suppliers.create(
    canonical_name="Acme Corporation",
    aliases=["ACME Corp"]
)

# Create transaction
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

# Calculate emissions
calculation = client.emissions.calculate(
    transactions=[transaction.id],
    gwp_standard="AR6"
)

print(f"Emissions: {calculation.total_emissions_kg_co2e} kg CO2e")
```

---

## Client Initialization

### API Key Authentication

```python
from vcci import Client

client = Client(api_key="sk_live_your_key_here")
```

### OAuth Authentication

```python
client = Client(
    client_id="vcci_client_abc123",
    client_secret="sk_live_secret123",
    auth_method="oauth"
)
```

### Custom Configuration

```python
client = Client(
    api_key="sk_live_your_key_here",
    base_url="https://api-staging.vcci.greenlang.io/v2",
    timeout=60,
    max_retries=5,
    debug=True
)
```

---

## Suppliers

### Create Supplier

```python
supplier = client.suppliers.create(
    canonical_name="Acme Steel Corporation",
    aliases=["ACME Steel", "Acme Corp"],
    identifiers={
        "lei": "529900XYZU1234567890",
        "duns": "123456789"
    },
    enrichment={
        "headquarters": {
            "country": "US",
            "city": "Pittsburgh"
        },
        "industry": "Steel Manufacturing",
        "revenue_usd": 50000000
    }
)

print(supplier.id)  # "sup_abc123"
print(supplier.canonical_name)  # "Acme Steel Corporation"
```

### List Suppliers

```python
# Basic listing
suppliers = client.suppliers.list(limit=100)

for supplier in suppliers:
    print(f"{supplier.canonical_name} ({supplier.id})")

# With filters
suppliers = client.suppliers.list(
    search="Acme",
    has_lei=True,
    status="active",
    sort_by="name"
)
```

### Pagination

```python
# Manual pagination
suppliers = client.suppliers.list(limit=100, offset=0)
more_suppliers = client.suppliers.list(limit=100, offset=100)

# Auto-pagination
for supplier in client.suppliers.list().auto_paginate():
    print(supplier.canonical_name)
```

### Get Supplier

```python
supplier = client.suppliers.get("sup_abc123")

print(supplier.canonical_name)
print(supplier.enrichment.headquarters.country)
```

### Update Supplier

```python
supplier = client.suppliers.update(
    "sup_abc123",
    canonical_name="Acme Steel Corporation (Updated)",
    status="inactive"
)
```

### Delete Supplier

```python
# Soft delete
client.suppliers.delete("sup_abc123")

# Hard delete (admin only)
client.suppliers.delete("sup_abc123", permanent=True)
```

### Resolve Supplier

```python
# Basic resolution
resolution = client.suppliers.resolve(
    supplier_name="ACME CORP"
)

if resolution.status == "auto_matched":
    print(f"Matched to: {resolution.matched_supplier_id}")
    print(f"Confidence: {resolution.confidence}")
else:
    print("Requires human review")
    for candidate in resolution.candidates:
        print(f"  - {candidate.canonical_name}: {candidate.confidence}")

# With hints
resolution = client.suppliers.resolve(
    supplier_name="ACME CORP",
    hints={
        "country": "US",
        "lei": "529900XYZU1234567890"
    },
    confidence_threshold=0.95
)
```

### Batch Resolve

```python
suppliers_to_resolve = [
    {"id": "row_001", "supplier_name": "ACME CORP"},
    {"id": "row_002", "supplier_name": "Steel Inc"},
    {"id": "row_003", "supplier_name": "Metal Works LLC"}
]

results = client.suppliers.batch_resolve(suppliers_to_resolve)

print(f"Auto-matched: {results.summary.auto_matched}")
print(f"Human review: {results.summary.human_review_required}")

for result in results.results:
    if result.status == "auto_matched":
        print(f"{result.id}: Matched to {result.matched_supplier_id}")
```

### Enrich Supplier

```python
# Enrich from all sources
supplier = client.suppliers.enrich("sup_abc123")

# Enrich from specific sources
supplier = client.suppliers.enrich(
    "sup_abc123",
    sources=["lei", "duns"]
)

print(supplier.enrichment.headquarters)
print(supplier.enrichment.parent_company)
```

---

## Procurement

### Create Transaction

```python
transaction = client.procurement.create_transaction(
    transaction_type="purchase_order",
    transaction_id="PO-2024-001",
    transaction_date="2024-11-01",
    supplier_name="Acme Corporation",
    product_name="Hot rolled steel coil",
    quantity=1000,
    unit="kg",
    spend_usd=5000.00,
    currency="USD"
)

print(transaction.id)
print(transaction.status)  # "pending" or "processed"
```

### List Transactions

```python
transactions = client.procurement.list_transactions(
    supplier_id="sup_abc123",
    date_from="2024-01-01",
    date_to="2024-12-31",
    status="processed",
    limit=100
)

for txn in transactions:
    print(f"{txn.transaction_id}: ${txn.spend_usd}")
```

### Batch Create Transactions

```python
transactions_data = [
    {
        "transaction_type": "purchase_order",
        "transaction_id": f"PO-2024-{i:03d}",
        "transaction_date": "2024-11-01",
        "supplier_name": "Acme Corporation",
        "product_name": "Steel beams",
        "quantity": 100 * i,
        "unit": "kg",
        "spend_usd": 500.00 * i
    }
    for i in range(1, 101)
]

# Synchronous (blocks until complete)
results = client.procurement.batch_create_transactions(
    transactions_data,
    async_mode=False
)

print(f"Success: {results.summary.success}")
print(f"Errors: {results.summary.error}")

# Asynchronous (returns job ID)
job = client.procurement.batch_create_transactions(
    transactions_data,
    async_mode=True
)

# Poll for job status
while job.status in ["queued", "processing"]:
    time.sleep(5)
    job = client.procurement.get_job(job.job_id)

print(f"Job complete: {job.results.success} successful")
```

### Upload File

```python
# Upload CSV file
with open("procurement_data.csv", "rb") as f:
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

print(f"Job ID: {job.job_id}")
print(f"Status: {job.status}")

# Monitor job
job = client.procurement.wait_for_job(job.job_id, poll_interval=5)
print(f"Complete: {job.results.success} rows processed")
```

---

## Emissions Calculations

### Calculate Emissions

```python
# Single transaction
calculation = client.emissions.calculate(
    transactions=["txn_abc123"],
    gwp_standard="AR6",
    uncertainty_method="monte_carlo",
    monte_carlo_iterations=10000
)

print(f"Total: {calculation.summary.total_emissions_kg_co2e} kg CO2e")

for result in calculation.results:
    print(f"Transaction {result.transaction_id}:")
    print(f"  Emissions: {result.emissions_kg_co2e} kg CO2e")
    print(f"  Tier: {result.tier}")
    print(f"  Uncertainty: {result.uncertainty.range.lower_bound} - {result.uncertainty.range.upper_bound}")
    print(f"  DQI Score: {result.data_quality.dqi_score}")
```

### Get Calculation

```python
calc = client.emissions.get_calculation("calc_abc123")

print(calc.emissions_kg_co2e)
print(calc.tier)
print(calc.emission_factor.source)
print(calc.provenance.calculation_hash)
```

### Aggregated Emissions

```python
# Basic aggregation
aggregate = client.emissions.aggregate(
    date_from="2024-01-01",
    date_to="2024-12-31"
)

print(f"Total: {aggregate.summary.total_emissions_kg_co2e} kg CO2e")

# Group by category and supplier
aggregate = client.emissions.aggregate(
    date_from="2024-01-01",
    date_to="2024-12-31",
    group_by=["category", "supplier"],
    supplier_id="sup_abc123",
    category="1"
)

for group_data in aggregate.data:
    print(f"{group_data.group}: {group_data.total_emissions_kg_co2e} kg CO2e")
```

---

## Emission Factors

### Resolve Factor

```python
factor = client.factors.resolve(
    activity_type="steel_production",
    product_name="Steel, hot rolled coil",
    unit="kg",
    region="US",
    gwp_standard="AR6"
)

print(f"Factor: {factor.value} {factor.unit}")
print(f"Source: {factor.source} v{factor.source_version}")
print(f"Uncertainty: {factor.uncertainty.min} - {factor.uncertainty.max}")
```

### List Factor Sources

```python
sources = client.factors.list_sources()

for source in sources.sources:
    print(f"{source.name} v{source.version}")
    print(f"  Coverage: {', '.join(source.coverage.regions)}")
    print(f"  Last updated: {source.last_updated}")
```

---

## PCF Exchange

### List PCFs

```python
pcfs = client.pcf.list(
    supplier_id="sup_abc123",
    has_verification=True
)

for pcf in pcfs:
    print(f"{pcf.product_name}: {pcf.pcf_value_kg_co2e} kg CO2e")
```

### Create PCF

```python
pcf = client.pcf.create(
    product_id="prod_123",
    product_name="Steel beams",
    supplier_id="sup_abc123",
    pcf_value_kg_co2e=850.2,
    functional_unit="1 kg",
    boundary="cradle_to_gate",
    reference_period={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    },
    verification={
        "verified": True,
        "verifier": "TÜV SÜD",
        "verification_date": "2024-10-15"
    }
)
```

### Import PCFs

```python
# PACT Pathfinder format
import json

with open("supplier_pcfs_pact.json") as f:
    pact_data = json.load(f)

results = client.pcf.import_pcfs(
    pcfs=pact_data,
    format="pact",
    validate_only=False
)

print(f"Imported: {results.imported}")
print(f"Errors: {len(results.errors)}")
```

### Export PCFs

```python
export = client.pcf.export(
    product_ids=["prod_1", "prod_2", "prod_3"],
    format="pact",
    include_verification=True
)

# Save to file
with open("exported_pcfs.json", "w") as f:
    json.dump(export.pcfs, f, indent=2)
```

---

## Reports

### Generate Report

```python
# ESRS E1 report
report = client.reports.generate(
    report_type="esrs_e1",
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_format="pdf",
    include_charts=True,
    categories=["1", "4", "6"]
)

print(f"Report ID: {report.report_id}")
print(f"Status: {report.status}")

# Wait for completion
report = client.reports.wait_for_report(report.report_id)
print(f"Download URL: {report.download_url}")

# Download report
report_data = client.reports.download(report.report_id)

with open("esrs_report.pdf", "wb") as f:
    f.write(report_data)
```

### List Reports

```python
reports = client.reports.list(
    report_type="esrs_e1",
    status="completed",
    limit=10
)

for report in reports:
    print(f"{report.report_id}: {report.report_type} ({report.status})")
```

---

## Policies

### List Policies

```python
policies = client.policies.list()

for policy in policies.policies:
    print(f"{policy.name} ({policy.category})")
```

### Create Policy

```python
rego_code = """
package vcci.category1

import future.keywords.if

# Category 1 calculation policy
allow if {
    input.tier == "1"
    input.pcf_available
}

tier := "1" if {
    input.pcf_available
} else := "2" if {
    input.average_data_available
} else := "3"
"""

policy = client.policies.create(
    name="Category 1 Enhanced Policy",
    description="Prioritizes PCF data for Cat 1",
    category="1",
    rego_code=rego_code
)
```

### Evaluate Policy

```python
result = client.policies.evaluate(
    policy_id="pol_abc123",
    input={
        "transaction_id": "txn_123",
        "pcf_available": True,
        "tier": "1"
    }
)

print(f"Decision: {result.decision}")
print(f"Result: {result.result}")
```

---

## Review Workflows

### List Review Queue

```python
items = client.workflows.list_review_queue(
    queue_type="entity_resolution",
    status="pending",
    limit=50
)

for item in items:
    print(f"{item.id}: {item.queue_type} ({item.priority})")
```

### Resolve Review Item

```python
resolved = client.workflows.resolve_review_item(
    item_id="review_abc123",
    action="approve",
    resolution_data={
        "matched_supplier_id": "sup_xyz789"
    },
    notes="Verified via DUNS number match"
)

print(f"Resolved: {resolved.status}")
```

---

## Supplier Engagement

### List Campaigns

```python
campaigns = client.engagement.list_campaigns(status="active")

for campaign in campaigns:
    print(f"{campaign.name}: {campaign.metrics.response_rate}% response")
```

### Create Campaign

```python
campaign = client.engagement.create_campaign(
    name="Q4 2024 PCF Request Campaign",
    type="pcf_request",
    target_suppliers=["sup_1", "sup_2", "sup_3"],
    message_template="""
    Dear {supplier_name},

    We would like to request your Product Carbon Footprint (PCF)
    data for the products we purchase from you...
    """,
    scheduled_at="2024-11-15T09:00:00Z"
)
```

### Grant Portal Access

```python
access = client.engagement.grant_portal_access(
    supplier_id="sup_abc123",
    contact_email="contact@acmecorp.com",
    expiration_days=90,
    send_email=True
)

print(f"Portal URL: {access.access_url}")
```

---

## Administration

### List Users

```python
users = client.admin.list_users(role="admin")

for user in users:
    print(f"{user.name} ({user.email})")
```

### Create User

```python
user = client.admin.create_user(
    email="newuser@company.com",
    name="John Doe",
    role="user"
)
```

---

## Error Handling

```python
from vcci.exceptions import (
    VCCIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError
)

try:
    supplier = client.suppliers.get("sup_invalid")
except NotFoundError:
    print("Supplier not found")
except AuthenticationError:
    print("Invalid credentials")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation error: {e.errors}")
except VCCIError as e:
    print(f"API error: {e.message}")
```

---

## Advanced Features

### Async Client

```python
import asyncio
from vcci import AsyncClient

async def main():
    client = AsyncClient(api_key="sk_live_your_key")

    # Concurrent requests
    suppliers, transactions = await asyncio.gather(
        client.suppliers.list(limit=100),
        client.procurement.list_transactions(limit=100)
    )

    print(f"Suppliers: {len(suppliers)}")
    print(f"Transactions: {len(transactions)}")

asyncio.run(main())
```

### Webhooks

```python
from vcci.webhooks import verify_signature

def handle_webhook(request):
    payload = request.body
    signature = request.headers['X-VCCI-Signature']
    secret = "whsec_abc123"

    if not verify_signature(payload, signature, secret):
        return {"error": "Invalid signature"}, 401

    event = request.json()

    if event['type'] == 'job.completed':
        handle_job_completed(event['data'])

    return {"status": "success"}, 200
```

### Custom Retry Logic

```python
from vcci import Client
from vcci.retry import ExponentialBackoff

client = Client(
    api_key="sk_live_your_key",
    retry_strategy=ExponentialBackoff(
        max_retries=5,
        initial_delay=1,
        max_delay=60,
        multiplier=2
    )
)
```

---

## Complete Example

```python
from vcci import Client
import time

# Initialize client
client = Client(api_key="sk_live_your_key_here")

# 1. Create supplier
supplier = client.suppliers.create(
    canonical_name="Green Energy Solutions Ltd",
    enrichment={
        "headquarters": {"country": "GB", "city": "London"},
        "industry": "Renewable Energy"
    }
)

# 2. Upload procurement data
with open("procurement_2024.csv", "rb") as f:
    job = client.procurement.upload_file(f, file_type="csv")

# 3. Wait for processing
job = client.procurement.wait_for_job(job.job_id)
print(f"Processed {job.results.success} transactions")

# 4. Calculate emissions
transactions = client.procurement.list_transactions(limit=1000)
calculation = client.emissions.calculate(
    transactions=[t.id for t in transactions],
    gwp_standard="AR6"
)

# 5. Get aggregated results
aggregate = client.emissions.aggregate(
    date_from="2024-01-01",
    date_to="2024-12-31",
    group_by=["category"]
)

print(f"Total Scope 3 emissions: {aggregate.summary.total_emissions_kg_co2e} kg CO2e")

# 6. Generate ESRS report
report = client.reports.generate(
    report_type="esrs_e1",
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_format="pdf"
)

report = client.reports.wait_for_report(report.report_id)
report_data = client.reports.download(report.report_id)

with open("esrs_e1_2024.pdf", "wb") as f:
    f.write(report_data)

print("✓ ESRS E1 report generated")
```

---

## Testing

```python
# Use test API key for testing
client = Client(api_key="sk_test_your_test_key")

# Or use test mode
client = Client(api_key="sk_live_your_key", test_mode=True)
```

---

## Resources

- **GitHub:** https://github.com/greenlang/vcci-python
- **PyPI:** https://pypi.org/project/vcci-python/
- **Changelog:** https://github.com/greenlang/vcci-python/blob/main/CHANGELOG.md
- **API Docs:** https://docs.vcci.greenlang.io

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
