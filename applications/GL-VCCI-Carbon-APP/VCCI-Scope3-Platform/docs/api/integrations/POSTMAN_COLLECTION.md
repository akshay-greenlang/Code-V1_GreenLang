# Postman Collection Guide

## GL-VCCI API Postman Collection

Complete Postman collection for testing and exploring the GL-VCCI API.

---

## Download Collection

**Postman Collection:** [Download VCCI-API-v2.postman_collection.json](https://api.vcci.greenlang.io/docs/postman/collection)

**Environment Files:**
- Production: [vcci-production.postman_environment.json](https://api.vcci.greenlang.io/docs/postman/env-production)
- Staging: [vcci-staging.postman_environment.json](https://api.vcci.greenlang.io/docs/postman/env-staging)

---

## Quick Setup (5 minutes)

### 1. Import Collection

1. Open Postman
2. Click **Import** button (top left)
3. Drag `VCCI-API-v2.postman_collection.json` or paste URL
4. Click **Import**

### 2. Import Environment

1. Click **Environments** (left sidebar)
2. Click **Import**
3. Import `vcci-production.postman_environment.json`
4. Select **vcci-production** from environment dropdown (top right)

### 3. Configure API Key

1. Click **Environments** → **vcci-production**
2. Set `api_key` variable to your API key: `sk_live_your_key_here`
3. Save environment (Ctrl+S)

### 4. Test Connection

1. Open **Health Check** request in collection
2. Click **Send**
3. Verify 200 OK response

---

## Environment Variables

The collection uses environment variables for easy configuration:

| Variable | Description | Example |
|----------|-------------|---------|
| `base_url` | API base URL | `https://api.vcci.greenlang.io/v2` |
| `api_key` | Your API key | `sk_live_abc123...` |
| `supplier_id` | Auto-set from create responses | `sup_abc123` |
| `transaction_id` | Auto-set from create responses | `txn_abc123` |
| `calculation_id` | Auto-set from calculations | `calc_abc123` |
| `report_id` | Auto-set from report generation | `rpt_abc123` |

These variables are automatically set by test scripts in the collection.

---

## Collection Structure

```
VCCI API v2.0
├── Authentication
│   ├── Get OAuth Token
│   ├── Refresh Token
│   ├── Create API Key
│   └── List API Keys
├── Suppliers
│   ├── List Suppliers
│   ├── Create Supplier
│   ├── Get Supplier
│   ├── Update Supplier
│   ├── Delete Supplier
│   ├── Resolve Supplier
│   ├── Batch Resolve Suppliers
│   └── Enrich Supplier
├── Procurement
│   ├── Create Transaction
│   ├── List Transactions
│   ├── Batch Create Transactions
│   ├── Upload File
│   └── Get Job Status
├── Emissions
│   ├── Calculate Emissions
│   ├── Get Calculation
│   └── Aggregated Emissions
├── Emission Factors
│   ├── Resolve Factor
│   └── List Factor Sources
├── PCF Exchange
│   ├── List PCFs
│   ├── Create PCF
│   ├── Import PCFs
│   └── Export PCFs
├── Reports
│   ├── Generate Report
│   ├── Get Report Status
│   └── List Reports
├── Policies
│   ├── List Policies
│   ├── Create Policy
│   ├── Get Policy
│   └── Evaluate Policy
├── Workflows
│   ├── List Review Queue
│   └── Resolve Review Item
├── Engagement
│   ├── List Campaigns
│   ├── Create Campaign
│   └── Grant Portal Access
└── Admin
    ├── List Users
    ├── Create User
    └── List Tenants
```

---

## Pre-Request Scripts

The collection includes global pre-request scripts that:

1. **Auto-inject API key** from environment variable
2. **Set timestamps** for date fields
3. **Generate request IDs** for tracking
4. **Validate required variables** are set

Example pre-request script:

```javascript
// Auto-inject API key
pm.request.headers.add({
    key: 'X-API-Key',
    value: pm.environment.get('api_key')
});

// Set current date for transaction_date
pm.variables.set('current_date', new Date().toISOString().split('T')[0]);
```

---

## Test Scripts

Collection includes automated tests that:

1. **Verify response status codes**
2. **Validate response schema**
3. **Extract and save IDs** to environment variables
4. **Check rate limit headers**

Example test script:

```javascript
// Test: Successful response
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

// Test: Response has data
pm.test("Response has data", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('data');
});

// Save supplier_id to environment
var jsonData = pm.response.json();
if (jsonData.id) {
    pm.environment.set("supplier_id", jsonData.id);
}

// Check rate limit
pm.test("Rate limit not exceeded", function () {
    var remaining = pm.response.headers.get('X-RateLimit-Remaining');
    pm.expect(parseInt(remaining)).to.be.above(0);
});
```

---

## Example Requests

### 1. Create Supplier

**Endpoint:** `POST {{base_url}}/suppliers`

**Headers:**
```
X-API-Key: {{api_key}}
Content-Type: application/json
```

**Body:**
```json
{
  "canonical_name": "Acme Steel Corporation",
  "aliases": ["ACME Steel", "Acme Corp"],
  "identifiers": {
    "lei": "529900XYZU1234567890",
    "duns": "123456789"
  },
  "enrichment": {
    "headquarters": {
      "country": "US",
      "city": "Pittsburgh"
    },
    "industry": "Steel Manufacturing"
  }
}
```

**Tests:**
```javascript
pm.test("Supplier created", function () {
    pm.response.to.have.status(201);
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('id');
    pm.environment.set("supplier_id", jsonData.id);
});
```

---

### 2. Calculate Emissions

**Endpoint:** `POST {{base_url}}/emissions/calculate`

**Body:**
```json
{
  "transactions": [
    {"transaction_id": "{{transaction_id}}"}
  ],
  "options": {
    "gwp_standard": "AR6",
    "uncertainty_method": "monte_carlo",
    "monte_carlo_iterations": 10000
  }
}
```

**Tests:**
```javascript
pm.test("Calculation successful", function () {
    pm.response.to.have.status(200);
    var jsonData = pm.response.json();

    pm.expect(jsonData.results).to.be.an('array');
    pm.expect(jsonData.results[0]).to.have.property('emissions_kg_co2e');

    // Save calculation ID
    pm.environment.set("calculation_id", jsonData.results[0].id);
});
```

---

### 3. Generate Report

**Endpoint:** `POST {{base_url}}/reports/generate`

**Body:**
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

**Tests:**
```javascript
pm.test("Report queued", function () {
    pm.response.to.have.status(202);
    var jsonData = pm.response.json();

    pm.expect(jsonData).to.have.property('report_id');
    pm.environment.set("report_id", jsonData.report_id);
});
```

---

## Running Collection

### Run Individual Request

1. Select request from collection
2. Ensure correct environment is selected
3. Click **Send**
4. Review response and test results

### Run Folder

1. Right-click folder (e.g., "Suppliers")
2. Select **Run folder**
3. Configure run settings
4. Click **Run**
5. View results summary

### Run Entire Collection

1. Click **...** next to collection name
2. Select **Run collection**
3. Configure:
   - Environment: **vcci-production**
   - Iterations: **1**
   - Delay: **100ms** (to avoid rate limits)
4. Click **Run VCCI API v2.0**

---

## Collection Runner

### Basic Run

```
Collection: VCCI API v2.0
Environment: vcci-production
Iterations: 1
Delay: 100ms
Data File: None
```

### Data-Driven Run

Create CSV file `suppliers.csv`:

```csv
canonical_name,country,city,industry
Acme Corporation,US,New York,Manufacturing
Steel Works Inc,UK,London,Steel Production
Green Energy Ltd,DE,Berlin,Renewable Energy
```

Run configuration:

```
Collection: VCCI API v2.0 → Suppliers → Create Supplier
Environment: vcci-production
Iterations: Auto (from CSV)
Data File: suppliers.csv
```

The request body uses CSV columns:

```json
{
  "canonical_name": "{{canonical_name}}",
  "enrichment": {
    "headquarters": {
      "country": "{{country}}",
      "city": "{{city}}"
    },
    "industry": "{{industry}}"
  }
}
```

---

## Workflows

### Workflow 1: Complete Scope 3 Calculation

Run these requests in order:

1. **Suppliers → Create Supplier** (or Resolve Supplier)
2. **Procurement → Create Transaction**
3. **Emissions → Calculate Emissions**
4. **Emissions → Aggregated Emissions**
5. **Reports → Generate Report**

### Workflow 2: Bulk Import

1. **Procurement → Upload File**
2. **Procurement → Get Job Status** (poll until complete)
3. **Emissions → Calculate Emissions** (batch)
4. **Reports → Generate Report**

---

## Mock Server

Postman collection includes mock server for testing integrations without live API calls.

### Using Mock Server

1. Click **...** next to collection
2. Select **Mock collection**
3. Name: **VCCI API Mock**
4. Click **Create Mock Server**
5. Copy mock server URL: `https://abc123.mock.pstmn.io`

Update environment:

```json
{
  "base_url": "https://abc123.mock.pstmn.io/v2"
}
```

Mock responses include:
- Realistic sample data
- Proper status codes
- Rate limit headers
- Error responses

---

## Debugging

### Enable Request Logging

```javascript
// In pre-request script
console.log('Request URL:', pm.request.url.toString());
console.log('Request Body:', pm.request.body.raw);
```

### Enable Response Logging

```javascript
// In test script
console.log('Response Status:', pm.response.code);
console.log('Response Body:', pm.response.text());
console.log('Response Time:', pm.response.responseTime + 'ms');
```

### View Postman Console

1. Click **Console** button (bottom left)
2. View all requests, responses, and console.log output
3. Filter by status code, method, etc.

---

## Advanced Features

### Dynamic Variables

Use Postman's dynamic variables:

```json
{
  "transaction_id": "PO-{{$timestamp}}",
  "transaction_date": "{{$isoTimestamp}}",
  "reference": "REF-{{$randomUUID}}"
}
```

### Conditional Logic

```javascript
// Skip request if supplier already exists
if (pm.environment.get("supplier_id")) {
    console.log("Supplier already exists, skipping...");
    pm.execution.skipRequest();
}
```

### Request Chaining

```javascript
// In test script: Trigger next request
if (pm.response.code === 202) {
    // Job created, wait 5 seconds then check status
    setTimeout(function() {
        postman.setNextRequest("Get Job Status");
    }, 5000);
}
```

---

## CI/CD Integration

### Newman (Command Line)

Install Newman:

```bash
npm install -g newman
```

Run collection:

```bash
newman run VCCI-API-v2.postman_collection.json \
  --environment vcci-production.postman_environment.json \
  --reporters cli,json \
  --reporter-json-export results.json
```

### GitHub Actions

```yaml
name: API Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Newman
        run: npm install -g newman

      - name: Run API Tests
        run: |
          newman run VCCI-API-v2.postman_collection.json \
            --environment vcci-production.postman_environment.json \
            --env-var "api_key=${{ secrets.VCCI_API_KEY }}"
```

---

## Troubleshooting

### "Unauthorized" Errors

1. Verify API key is set in environment
2. Check API key hasn't expired
3. Ensure correct environment is selected

### "Rate Limit Exceeded"

1. Add delays between requests (100-500ms)
2. Use batch endpoints instead of individual calls
3. Reduce iterations in Collection Runner

### Variables Not Saving

1. Check test scripts are running (green checkmarks)
2. Verify variable names match exactly
3. Save environment after changes (Ctrl+S)

---

## Best Practices

### 1. Use Separate Environments

Create environments for each stage:
- `vcci-local` (for local dev)
- `vcci-staging` (for testing)
- `vcci-production` (for production)

### 2. Organize Collections

```
VCCI API v2.0
├── 00 - Setup & Health
├── 01 - Authentication
├── 02 - Suppliers
├── 03 - Procurement
└── ...
```

### 3. Add Request Descriptions

Document each request with:
- Purpose
- Required parameters
- Expected response
- Common errors

### 4. Share Collections

1. Create team workspace
2. Share collection with team
3. Version control with Git

---

## Resources

- **Postman Documentation:** https://learning.postman.com/
- **Newman Documentation:** https://learning.postman.com/docs/running-collections/using-newman-cli/
- **API Documentation:** https://docs.vcci.greenlang.io
- **Support:** support@greenlang.io

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
