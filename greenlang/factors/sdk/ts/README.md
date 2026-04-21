# @greenlang/factors

TypeScript/JavaScript SDK for the [GreenLang Emission Factors API](https://greenlang.io).
Search, resolve, diff, and audit 300+ peer-reviewed emission factors
from a single typed client. Full parity with the Python SDK.

## Installation

```bash
npm install @greenlang/factors
# or
yarn add @greenlang/factors
# or
pnpm add @greenlang/factors
```

Requires Node.js 18+ (uses native `fetch` and WebCrypto) or any
modern browser. For Node 16, install `undici` and pass its `fetch`
via `fetchImpl`.

## Quick Start

```ts
import { FactorsClient } from '@greenlang/factors';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: process.env.GL_FACTORS_API_KEY!,
  edition: 'ef_2026_q1',
});

// Basic search
const hits = await client.search('diesel combustion US');
for (const f of hits.factors) {
  console.log(`${f.factor_id}: ${f.co2e_per_unit} kgCO2e/${f.unit}`);
}

// 7-step cascade resolution with full explain
const resolved = await client.resolve({
  activity: 'diesel combustion',
  method_profile: 'corporate_scope1',
  jurisdiction: 'US-CA',
  facility_id: 'plant-42',
});
console.log(resolved.chosen_factor_id, resolved.why_chosen);
```

## Authentication

```ts
// API key (recommended)
new FactorsClient({ baseUrl, apiKey: 'gl_fac_...' });

// JWT bearer
new FactorsClient({ baseUrl, jwtToken: 'eyJ...' });

// HMAC request signing (Pro+)
import { APIKeyAuth, HMACAuth } from '@greenlang/factors';
new FactorsClient({
  baseUrl,
  auth: new HMACAuth({
    apiKeyId: 'key-id',
    secret: process.env.HMAC_SECRET!,
    primary: new APIKeyAuth({ apiKey: 'gl_fac_...' }),
  }),
});
```

See [docs/sdk/typescript/AUTHENTICATION.md](../../../../docs/sdk/typescript/AUTHENTICATION.md) for full details.

## Key features

- **Typed models** — `Factor`, `ResolvedFactor`, `Edition`, `AuditBundle`, `Override`, and 13 more
- **Full parity** with the Python SDK (same endpoint list, same error hierarchy)
- **Automatic retries** on 429 / 5xx with exponential backoff and `Retry-After` support
- **ETag caching** — transparent `If-None-Match`/`304` on GETs
- **Webhook verification** — HMAC-SHA256 over canonical JSON, byte-compatible with Python
- **Async iterators** — `for await (const f of client.paginateSearch(...))`
- **Works everywhere** — Node 18+, browsers, Cloudflare Workers, Deno, Bun

## Documentation

- [README.md](../../../../docs/sdk/typescript/README.md) — overview
- [AUTHENTICATION.md](../../../../docs/sdk/typescript/AUTHENTICATION.md)
- [RESOLUTION.md](../../../../docs/sdk/typescript/RESOLUTION.md)
- [ERROR_HANDLING.md](../../../../docs/sdk/typescript/ERROR_HANDLING.md)
- [VERSION_PINNING.md](../../../../docs/sdk/typescript/VERSION_PINNING.md)
- [BROWSER_VS_NODE.md](../../../../docs/sdk/typescript/BROWSER_VS_NODE.md)

## Example scripts

Runnable samples live under [`/examples/factors_sdk/typescript/`](../../../../examples/factors_sdk/typescript/):

| Script                        | What it does                                  |
|-------------------------------|-----------------------------------------------|
| `01_basic_search.ts`          | Basic and advanced `/search` + `/search/v2`   |
| `02_resolve_with_explain.ts`  | 7-step cascade resolution (Pro+ tier)         |
| `03_batch_resolution.ts`      | Submit batch jobs and poll to completion      |
| `04_tenant_override.ts`       | Write a tenant-scoped override (Platform tier)|
| `05_audit_export.ts`          | Download audit bundles (Enterprise tier)      |

## CLI

```bash
npm install -g @greenlang/factors
export GL_FACTORS_API_KEY=gl_fac_...
glfactors search "diesel" --geography US --limit 5
glfactors resolve ./request.json
glfactors explain ef_us_diesel_scope1_v2
glfactors coverage
```

## Build & test

```bash
cd greenlang/factors/sdk/ts
npm install
npm test              # jest
npm run typecheck     # tsc --noEmit (strict)
npm run build         # ESM + CJS dual output
```

## License

MIT.
