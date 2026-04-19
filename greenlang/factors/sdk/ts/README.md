# @greenlang/factors-sdk

TypeScript/JavaScript SDK for the [GreenLang Emission Factors API](https://greenlang.io). Query, search, match, and calculate with 327+ audited emission factors -- fully typed, zero runtime dependencies.

## Installation

```bash
npm install @greenlang/factors-sdk
# or
yarn add @greenlang/factors-sdk
# or
pnpm add @greenlang/factors-sdk
```

Requires Node.js 18+ (uses native `fetch`).

## Quick Start

```typescript
import { FactorsClient } from '@greenlang/factors-sdk';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1',
  apiKey: 'gl_your_api_key_here',
});

// Search for emission factors
const results = await client.search('diesel combustion US');
for (const factor of results.factors) {
  console.log(`${factor.factor_id}: ${factor.co2e_per_unit} kgCO2e/${factor.unit}`);
}
```

## Authentication

### API Key (recommended)

```typescript
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1',
  apiKey: 'gl_your_api_key_here',
});
```

### JWT Token

```typescript
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1',
  apiKey: 'eyJhbGciOiJIUzI1NiIs...', // JWT token
});
```

## Configuration

```typescript
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1', // API base URL
  apiKey: 'gl_...',           // API key or JWT token
  edition: '2026-Q1',         // Pin to a specific catalog edition
  timeout: 60_000,            // Request timeout in milliseconds (default: 60000)
  maxRetries: 3,              // Retries on transient errors (default: 3)
  retryBackoff: 1_000,        // Initial backoff in milliseconds (default: 1000)
});
```

## Common Operations

### Search Factors

```typescript
// Basic search
const results = await client.search('natural gas');

// Search with filters
const filtered = await client.search('electricity grid', {
  geography: 'EU',
  limit: 10,
  includePreview: false,
});

// Advanced search (v2) with sorting and pagination
const advanced = await client.searchV2({
  query: 'diesel',
  geography: 'US',
  scope: '1',
  sort_by: 'dqs_score',
  sort_order: 'desc',
  offset: 0,
  limit: 20,
});
```

### Get Factor Details

```typescript
// Get a specific factor by ID
const factor = await client.getFactor('ef_us_diesel_scope1_v2');

// Get provenance and license info
const provenance = await client.getProvenance('ef_us_diesel_scope1_v2');

// Get deprecation replacement chain
const replacements = await client.getReplacements('ef_us_diesel_scope1_v1');

// Field-by-field diff between editions
const diff = await client.diffFactor('ef_us_diesel_scope1_v2', '2025-Q4', '2026-Q1');
```

### Match Activity to Factors

```typescript
const matches = await client.match({
  activity_description: 'Burning 500 gallons of diesel fuel in a generator',
  geography: 'US',
  scope: '1',
  limit: 5,
});
for (const candidate of matches.candidates) {
  console.log(`${candidate.factor_id} (score: ${candidate.score})`);
}
```

### Calculate Emissions

```typescript
// Single calculation
const result = await client.calculate({
  fuel_type: 'diesel',
  activity_amount: 1000.0,
  activity_unit: 'gallons',
  geography: 'US',
  scope: '1',
  boundary: 'combustion',
});
console.log(`Emissions: ${result.emissions_tonnes_co2e} tCO2e`);

// Batch calculation
const batch = await client.calculateBatch([
  {
    fuel_type: 'diesel',
    activity_amount: 500.0,
    activity_unit: 'gallons',
    geography: 'US',
    scope: '1',
  },
  {
    fuel_type: 'natural_gas',
    activity_amount: 10000.0,
    activity_unit: 'therms',
    geography: 'US',
    scope: '1',
  },
]);
console.log(`Total: ${batch.total_emissions_tonnes_co2e} tCO2e`);
```

### List and Compare Editions

```typescript
// List available catalog editions
const editions = await client.listEditions();

// Compare two editions
const comparison = await client.compareEditions('2025-Q4', '2026-Q1');

// Get changelog for an edition
const changelog = await client.getChangelog('2026-Q1');
```

### Edition Pinning

```typescript
// Pin at client level for reproducible results
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1',
  apiKey: 'gl_...',
  edition: '2026-Q1', // All requests use this edition
});
```

### System Endpoints

```typescript
// Health check
const health = await client.health();

// API statistics
const stats = await client.stats();

// Coverage breakdown
const coverage = await client.coverage();

// Source registry
const sources = await client.sourceRegistry();
```

## Error Handling

```typescript
import { FactorsClient, FactorsApiError, FactorsConnectionError } from '@greenlang/factors-sdk';

try {
  const factor = await client.getFactor('nonexistent_id');
} catch (err) {
  if (err instanceof FactorsApiError) {
    console.error(`API error: HTTP ${err.statusCode} - ${err.message}`);
    if (err.statusCode === 404) {
      console.error('Factor not found');
    } else if (err.statusCode === 429) {
      console.error('Rate limit exceeded');
    }
  } else if (err instanceof FactorsConnectionError) {
    console.error(`Connection error: ${err.message}`);
  }
}
```

The SDK automatically retries on transient errors (HTTP 429, 500, 502, 503, 504) with exponential backoff.

## Tiers

| Tier | Price | Monthly Quota | Overage |
|------|-------|---------------|---------|
| Community | Free | 1,000 requests | N/A |
| Pro | $299/mo | 50,000 requests | $0.01/req |
| Enterprise | $999/mo | 500,000 requests | $0.005/req |

## License

MIT License.
