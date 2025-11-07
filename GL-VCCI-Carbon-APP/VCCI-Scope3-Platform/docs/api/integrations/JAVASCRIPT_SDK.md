# JavaScript/Node.js SDK Guide

## GL-VCCI JavaScript SDK

Official JavaScript/TypeScript SDK for the GL-VCCI Scope 3 Carbon Intelligence Platform.

---

## Installation

```bash
npm install @vcci/sdk
# or
yarn add @vcci/sdk
```

**Requirements:** Node.js 14+ or modern browsers (ES2020+)

---

## Quick Start

```javascript
const { VCCIClient } = require('@vcci/sdk');

// Initialize client
const client = new VCCIClient({
  apiKey: 'sk_live_your_key_here'
});

// Create supplier
const supplier = await client.suppliers.create({
  canonical_name: 'Acme Corporation',
  aliases: ['ACME Corp']
});

// Create transaction
const transaction = await client.procurement.createTransaction({
  transaction_type: 'purchase_order',
  transaction_id: 'PO-2024-001',
  transaction_date: '2024-11-01',
  supplier_name: 'Acme Corporation',
  product_name: 'Steel beams',
  quantity: 1000,
  unit: 'kg',
  spend_usd: 5000.00
});

// Calculate emissions
const calculation = await client.emissions.calculate({
  transactions: [transaction.id],
  options: { gwp_standard: 'AR6' }
});

console.log(`Emissions: ${calculation.summary.total_emissions_kg_co2e} kg CO2e`);
```

---

## TypeScript Support

```typescript
import { VCCIClient, Supplier, Transaction, EmissionCalculation } from '@vcci/sdk';

const client = new VCCIClient({
  apiKey: process.env.VCCI_API_KEY
});

const supplier: Supplier = await client.suppliers.create({
  canonical_name: 'Acme Corporation'
});

const calculation: EmissionCalculation = await client.emissions.calculate({
  transactions: ['txn_abc123']
});
```

---

## Client Initialization

### API Key Authentication

```javascript
const client = new VCCIClient({
  apiKey: 'sk_live_your_key_here'
});
```

### OAuth Authentication

```javascript
const client = new VCCIClient({
  auth: {
    type: 'oauth',
    clientId: 'vcci_client_abc123',
    clientSecret: 'sk_live_secret123'
  }
});
```

### Custom Configuration

```javascript
const client = new VCCIClient({
  apiKey: 'sk_live_your_key_here',
  baseUrl: 'https://api-staging.vcci.greenlang.io/v2',
  timeout: 60000,
  maxRetries: 5,
  debug: true
});
```

---

## Suppliers

### Create Supplier

```javascript
const supplier = await client.suppliers.create({
  canonical_name: 'Acme Steel Corporation',
  aliases: ['ACME Steel', 'Acme Corp'],
  identifiers: {
    lei: '529900XYZU1234567890',
    duns: '123456789'
  },
  enrichment: {
    headquarters: {
      country: 'US',
      city: 'Pittsburgh'
    },
    industry: 'Steel Manufacturing',
    revenue_usd: 50000000
  }
});

console.log(supplier.id);  // "sup_abc123"
console.log(supplier.canonical_name);
```

### List Suppliers

```javascript
// Basic listing
const { data: suppliers } = await client.suppliers.list({ limit: 100 });

suppliers.forEach(supplier => {
  console.log(`${supplier.canonical_name} (${supplier.id})`);
});

// With filters
const { data, pagination } = await client.suppliers.list({
  search: 'Acme',
  has_lei: true,
  status: 'active',
  sort_by: 'name',
  limit: 50
});
```

### Auto-Pagination

```javascript
// Iterate through all suppliers
for await (const supplier of client.suppliers.listAll()) {
  console.log(supplier.canonical_name);
}

// Or collect all
const allSuppliers = await client.suppliers.listAll().toArray();
```

### Get Supplier

```javascript
const supplier = await client.suppliers.get('sup_abc123');

console.log(supplier.canonical_name);
console.log(supplier.enrichment.headquarters.country);
```

### Update Supplier

```javascript
const updated = await client.suppliers.update('sup_abc123', {
  canonical_name: 'Acme Steel Corporation (Updated)',
  status: 'inactive'
});
```

### Delete Supplier

```javascript
// Soft delete
await client.suppliers.delete('sup_abc123');

// Hard delete (admin only)
await client.suppliers.delete('sup_abc123', { permanent: true });
```

### Resolve Supplier

```javascript
// Basic resolution
const resolution = await client.suppliers.resolve({
  supplier_name: 'ACME CORP'
});

if (resolution.status === 'auto_matched') {
  console.log(`Matched to: ${resolution.matched_supplier_id}`);
  console.log(`Confidence: ${resolution.confidence}`);
} else {
  console.log('Requires human review');
  resolution.candidates.forEach(candidate => {
    console.log(`  - ${candidate.canonical_name}: ${candidate.confidence}`);
  });
}

// With hints
const resolution = await client.suppliers.resolve({
  supplier_name: 'ACME CORP',
  hints: {
    country: 'US',
    lei: '529900XYZU1234567890'
  },
  confidence_threshold: 0.95
});
```

### Batch Resolve

```javascript
const suppliersToResolve = [
  { id: 'row_001', supplier_name: 'ACME CORP' },
  { id: 'row_002', supplier_name: 'Steel Inc' },
  { id: 'row_003', supplier_name: 'Metal Works LLC' }
];

const results = await client.suppliers.batchResolve({
  suppliers: suppliersToResolve
});

console.log(`Auto-matched: ${results.summary.auto_matched}`);
console.log(`Human review: ${results.summary.human_review_required}`);

results.results.forEach(result => {
  if (result.status === 'auto_matched') {
    console.log(`${result.id}: Matched to ${result.matched_supplier_id}`);
  }
});
```

### Enrich Supplier

```javascript
// Enrich from all sources
const enriched = await client.suppliers.enrich('sup_abc123');

// Enrich from specific sources
const enriched = await client.suppliers.enrich('sup_abc123', {
  sources: ['lei', 'duns']
});

console.log(enriched.enrichment.headquarters);
console.log(enriched.enrichment.parent_company);
```

---

## Procurement

### Create Transaction

```javascript
const transaction = await client.procurement.createTransaction({
  transaction_type: 'purchase_order',
  transaction_id: 'PO-2024-001',
  transaction_date: '2024-11-01',
  supplier_name: 'Acme Corporation',
  product_name: 'Hot rolled steel coil',
  quantity: 1000,
  unit: 'kg',
  spend_usd: 5000.00,
  currency: 'USD'
});

console.log(transaction.id);
console.log(transaction.status);
```

### List Transactions

```javascript
const { data: transactions } = await client.procurement.listTransactions({
  supplier_id: 'sup_abc123',
  date_from: '2024-01-01',
  date_to: '2024-12-31',
  status: 'processed',
  limit: 100
});

transactions.forEach(txn => {
  console.log(`${txn.transaction_id}: $${txn.spend_usd}`);
});
```

### Batch Create Transactions

```javascript
const transactionsData = Array.from({ length: 100 }, (_, i) => ({
  transaction_type: 'purchase_order',
  transaction_id: `PO-2024-${String(i + 1).padStart(3, '0')}`,
  transaction_date: '2024-11-01',
  supplier_name: 'Acme Corporation',
  product_name: 'Steel beams',
  quantity: 100 * (i + 1),
  unit: 'kg',
  spend_usd: 500.00 * (i + 1)
}));

// Synchronous (blocks until complete)
const results = await client.procurement.batchCreateTransactions({
  transactions: transactionsData,
  async: false
});

console.log(`Success: ${results.summary.success}`);
console.log(`Errors: ${results.summary.error}`);

// Asynchronous (returns job ID)
const job = await client.procurement.batchCreateTransactions({
  transactions: transactionsData,
  async: true
});

// Poll for job status
const completedJob = await client.procurement.waitForJob(job.job_id);
console.log(`Job complete: ${completedJob.results.success} successful`);
```

### Upload File

```javascript
const fs = require('fs');
const FormData = require('form-data');

// Node.js
const formData = new FormData();
formData.append('file', fs.createReadStream('procurement_data.csv'));
formData.append('file_type', 'csv');
formData.append('column_mapping', JSON.stringify({
  'PO Number': 'transaction_id',
  'Vendor': 'supplier_name',
  'Amount': 'spend_usd'
}));
formData.append('skip_rows', '1');

const job = await client.procurement.uploadFile(formData);

console.log(`Job ID: ${job.job_id}`);
console.log(`Status: ${job.status}`);

// Monitor job
const completedJob = await client.procurement.waitForJob(job.job_id, {
  pollInterval: 5000
});
console.log(`Complete: ${completedJob.results.success} rows processed`);
```

### Browser File Upload

```javascript
// React example
const handleFileUpload = async (event) => {
  const file = event.target.files[0];

  const formData = new FormData();
  formData.append('file', file);
  formData.append('file_type', 'csv');

  const job = await client.procurement.uploadFile(formData);

  // Show progress
  const interval = setInterval(async () => {
    const status = await client.procurement.getJob(job.job_id);

    if (status.status === 'completed') {
      clearInterval(interval);
      console.log('Upload complete!');
    } else if (status.status === 'processing') {
      console.log(`Progress: ${status.progress.percentage}%`);
    }
  }, 2000);
};
```

---

## Emissions Calculations

### Calculate Emissions

```javascript
// Single transaction
const calculation = await client.emissions.calculate({
  transactions: [{ transaction_id: 'txn_abc123' }],
  options: {
    gwp_standard: 'AR6',
    uncertainty_method: 'monte_carlo',
    monte_carlo_iterations: 10000
  }
});

console.log(`Total: ${calculation.summary.total_emissions_kg_co2e} kg CO2e`);

calculation.results.forEach(result => {
  console.log(`Transaction ${result.transaction_id}:`);
  console.log(`  Emissions: ${result.emissions_kg_co2e} kg CO2e`);
  console.log(`  Tier: ${result.tier}`);
  console.log(`  Uncertainty: ${result.uncertainty.range.lower_bound} - ${result.uncertainty.range.upper_bound}`);
  console.log(`  DQI Score: ${result.data_quality.dqi_score}`);
});
```

### Get Calculation

```javascript
const calc = await client.emissions.getCalculation('calc_abc123');

console.log(calc.emissions_kg_co2e);
console.log(calc.tier);
console.log(calc.emission_factor.source);
console.log(calc.provenance.calculation_hash);
```

### Aggregated Emissions

```javascript
// Basic aggregation
const aggregate = await client.emissions.aggregate({
  date_from: '2024-01-01',
  date_to: '2024-12-31'
});

console.log(`Total: ${aggregate.summary.total_emissions_kg_co2e} kg CO2e`);

// Group by category and supplier
const aggregate = await client.emissions.aggregate({
  date_from: '2024-01-01',
  date_to: '2024-12-31',
  group_by: 'category,supplier',
  supplier_id: 'sup_abc123',
  category: '1'
});

aggregate.data.forEach(groupData => {
  console.log(`${JSON.stringify(groupData.group)}: ${groupData.total_emissions_kg_co2e} kg CO2e`);
});
```

---

## Emission Factors

### Resolve Factor

```javascript
const factor = await client.factors.resolve({
  activity_type: 'steel_production',
  product_name: 'Steel, hot rolled coil',
  unit: 'kg',
  region: 'US',
  gwp_standard: 'AR6'
});

console.log(`Factor: ${factor.value} ${factor.unit}`);
console.log(`Source: ${factor.source} v${factor.source_version}`);
console.log(`Uncertainty: ${factor.uncertainty.min} - ${factor.uncertainty.max}`);
```

### List Factor Sources

```javascript
const { sources } = await client.factors.listSources();

sources.forEach(source => {
  console.log(`${source.name} v${source.version}`);
  console.log(`  Coverage: ${source.coverage.regions.join(', ')}`);
  console.log(`  Last updated: ${source.last_updated}`);
});
```

---

## PCF Exchange

### List PCFs

```javascript
const { data: pcfs } = await client.pcf.list({
  supplier_id: 'sup_abc123',
  has_verification: true
});

pcfs.forEach(pcf => {
  console.log(`${pcf.product_name}: ${pcf.pcf_value_kg_co2e} kg CO2e`);
});
```

### Create PCF

```javascript
const pcf = await client.pcf.create({
  product_id: 'prod_123',
  product_name: 'Steel beams',
  supplier_id: 'sup_abc123',
  pcf_value_kg_co2e: 850.2,
  functional_unit: '1 kg',
  boundary: 'cradle_to_gate',
  reference_period: {
    start_date: '2024-01-01',
    end_date: '2024-12-31'
  },
  verification: {
    verified: true,
    verifier: 'TÜV SÜD',
    verification_date: '2024-10-15'
  }
});
```

### Import PCFs

```javascript
const fs = require('fs');
const pactData = JSON.parse(fs.readFileSync('supplier_pcfs_pact.json', 'utf8'));

const results = await client.pcf.importPCFs({
  pcfs: pactData,
  format: 'pact',
  validate_only: false
});

console.log(`Imported: ${results.imported}`);
console.log(`Errors: ${results.errors.length}`);
```

### Export PCFs

```javascript
const exportData = await client.pcf.export({
  product_ids: ['prod_1', 'prod_2', 'prod_3'],
  format: 'pact',
  include_verification: true
});

// Save to file
fs.writeFileSync('exported_pcfs.json', JSON.stringify(exportData.pcfs, null, 2));
```

---

## Reports

### Generate Report

```javascript
// ESRS E1 report
const report = await client.reports.generate({
  report_type: 'esrs_e1',
  reporting_period: {
    start_date: '2024-01-01',
    end_date: '2024-12-31'
  },
  output_format: 'pdf',
  include_charts: true,
  categories: ['1', '4', '6']
});

console.log(`Report ID: ${report.report_id}`);
console.log(`Status: ${report.status}`);

// Wait for completion
const completedReport = await client.reports.waitForReport(report.report_id);
console.log(`Download URL: ${completedReport.download_url}`);

// Download report
const reportData = await client.reports.download(report.report_id);

// Node.js - save to file
fs.writeFileSync('esrs_report.pdf', reportData);

// Browser - trigger download
const blob = new Blob([reportData], { type: 'application/pdf' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'esrs_report.pdf';
a.click();
```

### List Reports

```javascript
const { data: reports } = await client.reports.list({
  report_type: 'esrs_e1',
  status: 'completed',
  limit: 10
});

reports.forEach(report => {
  console.log(`${report.report_id}: ${report.report_type} (${report.status})`);
});
```

---

## Policies

### List Policies

```javascript
const { policies } = await client.policies.list();

policies.forEach(policy => {
  console.log(`${policy.name} (${policy.category})`);
});
```

### Create Policy

```javascript
const regoCode = `
package vcci.category1

import future.keywords.if

allow if {
    input.tier == "1"
    input.pcf_available
}

tier := "1" if {
    input.pcf_available
} else := "2" if {
    input.average_data_available
} else := "3"
`;

const policy = await client.policies.create({
  name: 'Category 1 Enhanced Policy',
  description: 'Prioritizes PCF data for Cat 1',
  category: '1',
  rego_code: regoCode
});
```

### Evaluate Policy

```javascript
const result = await client.policies.evaluate('pol_abc123', {
  input: {
    transaction_id: 'txn_123',
    pcf_available: true,
    tier: '1'
  }
});

console.log(`Decision: ${result.decision}`);
console.log(`Result: ${JSON.stringify(result.result)}`);
```

---

## Review Workflows

### List Review Queue

```javascript
const { data: items } = await client.workflows.listReviewQueue({
  queue_type: 'entity_resolution',
  status: 'pending',
  limit: 50
});

items.forEach(item => {
  console.log(`${item.id}: ${item.queue_type} (${item.priority})`);
});
```

### Resolve Review Item

```javascript
const resolved = await client.workflows.resolveReviewItem('review_abc123', {
  action: 'approve',
  resolution_data: {
    matched_supplier_id: 'sup_xyz789'
  },
  notes: 'Verified via DUNS number match'
});

console.log(`Resolved: ${resolved.status}`);
```

---

## Supplier Engagement

### List Campaigns

```javascript
const { data: campaigns } = await client.engagement.listCampaigns({
  status: 'active'
});

campaigns.forEach(campaign => {
  console.log(`${campaign.name}: ${campaign.metrics.response_rate}% response`);
});
```

### Create Campaign

```javascript
const campaign = await client.engagement.createCampaign({
  name: 'Q4 2024 PCF Request Campaign',
  type: 'pcf_request',
  target_suppliers: ['sup_1', 'sup_2', 'sup_3'],
  message_template: `
    Dear {supplier_name},

    We would like to request your Product Carbon Footprint (PCF)
    data for the products we purchase from you...
  `,
  scheduled_at: '2024-11-15T09:00:00Z'
});
```

### Grant Portal Access

```javascript
const access = await client.engagement.grantPortalAccess('sup_abc123', {
  contact_email: 'contact@acmecorp.com',
  expiration_days: 90,
  send_email: true
});

console.log(`Portal URL: ${access.access_url}`);
```

---

## Error Handling

```javascript
const {
  VCCIError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  NotFoundError
} = require('@vcci/sdk');

try {
  const supplier = await client.suppliers.get('sup_invalid');
} catch (error) {
  if (error instanceof NotFoundError) {
    console.log('Supplier not found');
  } else if (error instanceof AuthenticationError) {
    console.log('Invalid credentials');
  } else if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfter} seconds`);
  } else if (error instanceof ValidationError) {
    console.log(`Validation error: ${JSON.stringify(error.errors)}`);
  } else if (error instanceof VCCIError) {
    console.log(`API error: ${error.message}`);
  }
}
```

---

## Webhooks

### Verify Signature

```javascript
const { verifyWebhookSignature } = require('@vcci/sdk/webhooks');

// Express.js example
app.post('/webhooks/vcci', express.raw({ type: 'application/json' }), (req, res) => {
  const signature = req.headers['x-vcci-signature'];
  const secret = 'whsec_abc123';

  if (!verifyWebhookSignature(req.body, signature, secret)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }

  const event = JSON.parse(req.body);

  if (event.type === 'job.completed') {
    handleJobCompleted(event.data);
  }

  res.json({ status: 'success' });
});
```

---

## React Integration

```javascript
import React, { useEffect, useState } from 'react';
import { VCCIClient } from '@vcci/sdk';

const client = new VCCIClient({
  apiKey: process.env.REACT_APP_VCCI_API_KEY
});

function SuppliersList() {
  const [suppliers, setSuppliers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSuppliers() {
      try {
        const { data } = await client.suppliers.list({ limit: 100 });
        setSuppliers(data);
      } catch (error) {
        console.error('Error fetching suppliers:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchSuppliers();
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <ul>
      {suppliers.map(supplier => (
        <li key={supplier.id}>{supplier.canonical_name}</li>
      ))}
    </ul>
  );
}
```

---

## Complete Example

```javascript
const { VCCIClient } = require('@vcci/sdk');
const fs = require('fs');

async function main() {
  // Initialize client
  const client = new VCCIClient({
    apiKey: process.env.VCCI_API_KEY
  });

  // 1. Create supplier
  const supplier = await client.suppliers.create({
    canonical_name: 'Green Energy Solutions Ltd',
    enrichment: {
      headquarters: { country: 'GB', city: 'London' },
      industry: 'Renewable Energy'
    }
  });

  // 2. Upload procurement data
  const formData = new FormData();
  formData.append('file', fs.createReadStream('procurement_2024.csv'));

  const job = await client.procurement.uploadFile(formData);

  // 3. Wait for processing
  const completedJob = await client.procurement.waitForJob(job.job_id);
  console.log(`Processed ${completedJob.results.success} transactions`);

  // 4. Calculate emissions
  const { data: transactions } = await client.procurement.listTransactions({
    limit: 1000
  });

  const calculation = await client.emissions.calculate({
    transactions: transactions.map(t => ({ transaction_id: t.id })),
    options: { gwp_standard: 'AR6' }
  });

  // 5. Get aggregated results
  const aggregate = await client.emissions.aggregate({
    date_from: '2024-01-01',
    date_to: '2024-12-31',
    group_by: 'category'
  });

  console.log(`Total Scope 3 emissions: ${aggregate.summary.total_emissions_kg_co2e} kg CO2e`);

  // 6. Generate ESRS report
  const report = await client.reports.generate({
    report_type: 'esrs_e1',
    reporting_period: {
      start_date: '2024-01-01',
      end_date: '2024-12-31'
    },
    output_format: 'pdf'
  });

  const completedReport = await client.reports.waitForReport(report.report_id);
  const reportData = await client.reports.download(report.report_id);

  fs.writeFileSync('esrs_e1_2024.pdf', reportData);

  console.log('✓ ESRS E1 report generated');
}

main().catch(console.error);
```

---

## Resources

- **GitHub:** https://github.com/greenlang/vcci-js
- **npm:** https://www.npmjs.com/package/@vcci/sdk
- **Changelog:** https://github.com/greenlang/vcci-js/blob/main/CHANGELOG.md
- **API Docs:** https://docs.vcci.greenlang.io

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
