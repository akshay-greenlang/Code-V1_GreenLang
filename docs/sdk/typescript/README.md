# @greenlang/factors — TypeScript SDK

Typed client for the GreenLang Emission Factors REST API. Full parity with
the Python SDK (`greenlang.factors.sdk.python`): same endpoints, same
error hierarchy, same webhook signature scheme.

## Installation

```bash
npm install @greenlang/factors
# or
yarn add @greenlang/factors
# or
pnpm add @greenlang/factors
```

- Node 18+ (native `fetch`), or any modern browser.
- For Node 16, `npm install undici` and pass its `fetch` via `fetchImpl`.
- No runtime dependencies besides the global `fetch` / `crypto.subtle`.

## Quick start

```ts
import { FactorsClient } from '@greenlang/factors';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: process.env.GL_FACTORS_API_KEY!,
  edition: 'ef_2026_q1',        // optional — pin for reproducibility
  methodProfile: 'corporate_scope1',
});

const hits = await client.search('diesel US scope 1');
for (const f of hits.factors) {
  console.log(f.factor_id, f.co2e_per_unit);
}
```

## Feature matrix

| Capability                       | Notes                                      |
|----------------------------------|--------------------------------------------|
| Auth: API key / JWT / HMAC       | `APIKeyAuth`, `JWTAuth`, `HMACAuth`        |
| 7-step cascade resolution        | `resolve` / `resolveExplain` / `alternates`|
| Batch resolution + polling       | `resolveBatch` + `waitForBatch`            |
| Edition pinning                  | `X-Factors-Edition` header                 |
| Diff across editions             | `diff(factorId, left, right)`              |
| Audit bundle export              | `auditBundle(factorId)` (Enterprise)       |
| Tenant overrides                 | `setOverride` / `listOverrides`            |
| Pagination (offset + cursor)     | `paginateSearch`, `OffsetPaginator`        |
| Retry + exponential backoff      | On 429 / 5xx / network errors              |
| ETag cache                       | Transparent `If-None-Match` / 304 handling |
| Rate-limit headers               | `X-RateLimit-*`, `Retry-After`             |
| Webhook verification             | HMAC-SHA256 over canonical JSON            |
| TypeScript types                 | All 18 models (`Factor`, `ResolvedFactor`, …) |

## Documentation

- [AUTHENTICATION.md](./AUTHENTICATION.md) — API key, JWT, HMAC.
- [RESOLUTION.md](./RESOLUTION.md) — the 7-step cascade and explain payload.
- [ERROR_HANDLING.md](./ERROR_HANDLING.md) — exception hierarchy and remediation hints.
- [VERSION_PINNING.md](./VERSION_PINNING.md) — editions, diff, audit bundles.
- [BROWSER_VS_NODE.md](./BROWSER_VS_NODE.md) — environment notes.

## CLI

A minimal CLI is bundled (install globally or via `npx`):

```bash
export GL_FACTORS_API_KEY=gl_...
glfactors search "diesel" --geography US --limit 5
glfactors explain ef_us_diesel_scope1_v2
glfactors coverage
glfactors resolve ./request.json
```

## Parity with the Python SDK

Every Python method has a camelCase TS equivalent:

| Python                           | TypeScript                   |
|----------------------------------|------------------------------|
| `search`                         | `search`                     |
| `search_v2`                      | `searchV2`                   |
| `list_factors`                   | `listFactors`                |
| `paginate_search`                | `paginateSearch`             |
| `get_factor`                     | `getFactor`                  |
| `match`                          | `match`                      |
| `coverage`                       | `coverage`                   |
| `resolve_explain`                | `resolveExplain`             |
| `resolve`                        | `resolve`                    |
| `alternates`                     | `alternates`                 |
| `resolve_batch`                  | `resolveBatch`               |
| `get_batch_job`                  | `getBatchJob`                |
| `wait_for_batch`                 | `waitForBatch`               |
| `list_editions`                  | `listEditions`               |
| `get_edition`                    | `getEdition`                 |
| `diff`                           | `diff`                       |
| `audit_bundle`                   | `auditBundle`                |
| `list_sources` / `get_source`    | `listSources` / `getSource`  |
| `list_method_packs` / `get_method_pack` | `listMethodPacks` / `getMethodPack` |
| `set_override` / `list_overrides` | `setOverride` / `listOverrides` |

The webhook verifier is byte-compatible: both implementations
serialise payloads with `json.dumps(..., sort_keys=True, default=str)` /
`canonicalJsonStringify(...)`, which produces the same UTF-8 bytes
and therefore the same HMAC-SHA256 digest.

## Building & testing

```bash
cd greenlang/factors/sdk/ts
npm install
npm run build        # ESM + CJS dual output into dist/
npm run typecheck    # tsc --noEmit in strict mode
npm test             # jest
```

## License

MIT.
