# TypeScript SDK Quickstart

The TypeScript SDK is a strict parity implementation of the Python client. Same endpoints, same models, same ETag caching, same signed-receipt verification.

**Source:** `greenlang/factors/sdk/ts/src/client.ts`
**Package:** `@greenlang/factors` on npm
**Minimum Node:** 18 (uses native `fetch`).

---

## Install

```bash
npm install @greenlang/factors
# or
pnpm add @greenlang/factors
# or
yarn add @greenlang/factors
```

Zero dependencies. `undici` is an optional peer if you want to pin a specific `fetch` implementation in non-Node runtimes.

---

## Initialise a client

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_API_KEY,           // or jwtToken: "..."
  defaultEdition: "2027.Q1-electricity",     // sent as X-Factors-Edition
  timeoutMs: 30_000,
  maxRetries: 3,
});
```

Options are defined by `FactorsClientOptions` in `greenlang/factors/sdk/ts/src/client.ts`. Both `defaultEdition` and `edition` are accepted aliases.

---

## Step 1: Resolve a factor

```ts
import { FactorsClient, ResolutionRequest } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_API_KEY!,
});

const req: ResolutionRequest = {
  activity: "diesel combustion stationary",
  method_profile: "corporate_scope1",
  jurisdiction: "US",
  reporting_date: "2026-06-01",
};

const resolved = await client.resolveExplain(req, {
  edition: "2027.Q1-electricity",
});

console.log(resolved.factor_id);           // EF:US:diesel:2024:v1
console.log(resolved.co2e_per_unit, resolved.unit);
console.log("cascade step:", resolved.fallback_rank);

for (const alt of resolved.alternates.slice(0, 3)) {
  console.log("  alt:", alt.factor_id, alt.score);
}
```

Under the hood: `POST /api/v1/factors/resolve-explain`.

---

## Step 2: Explain a specific factor

```ts
const explain = await client.explainFactor({
  factorId: "EF:US:diesel:2024:v1",
  methodProfile: "corporate_scope1",
  limit: 10,
});

console.log("Tie-break reasons:", explain.tie_break_reasons);
console.log("Gas breakdown:", explain.gas_breakdown);
// { co2: 10.15, ch4: 0.04, n2o: 0.02, hfcs: null, ... }
```

Under the hood: `GET /api/v1/factors/{factor_id}/explain`.

---

## Step 3: Pin an edition

```ts
// Option A: per-client default (sent on every request).
const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_API_KEY!,
  defaultEdition: "2027.Q1-electricity",
});

// Option B: per-call override.
const resolved = await client.resolveExplain(req, {
  edition: "2027.Q1-electricity",
});

console.log("Server used edition:", resolved.edition_id);

// List available editions.
const editions = await client.listEditions({ status: "stable" });
for (const e of editions) {
  console.log(e.edition_id, e.created_at, e.factor_count);
}
```

---

## Step 4: Verify the signed receipt

```ts
import { verifyReceipt } from "@greenlang/factors";

const ok = await verifyReceipt({
  body: resolved,
  hmacSecret: process.env.GL_FACTORS_SIGNING_SECRET!,
});

if (!ok) {
  throw new Error("Signed receipt failed verification - payload tampered");
}
```

For Ed25519 (Consulting / Enterprise tiers):

```ts
await verifyReceipt({
  body: resolved,
  ed25519PublicKey: process.env.GL_FACTORS_ED25519_PUBLIC_KEY!,
});
```

The helper re-derives the canonical JSON hash (see `greenlang/factors/sdk/ts/src/canonical.ts` + `hash.ts`) and compares against `_signed_receipt.signature`. See [signed-receipts](../concepts/signed-receipts.md) for the full algorithm.

---

## Complete minimal example

```ts
import { FactorsClient } from "@greenlang/factors";

async function main() {
  const client = new FactorsClient({
    baseUrl: "https://api.greenlang.io",
    apiKey: process.env.GL_API_KEY!,
    defaultEdition: "2027.Q1-electricity",
  });

  const resolved = await client.resolveExplain({
    activity: "diesel combustion stationary",
    method_profile: "corporate_scope1",
    jurisdiction: "US",
    reporting_date: "2026-06-01",
  });

  console.log(`${resolved.factor_id}: ${resolved.co2e_per_unit} ${resolved.unit}`);
  console.log(`Edition: ${resolved.edition_id}`);
  console.log(`Receipt: ${resolved.signed_receipt.algorithm} @ ${resolved.signed_receipt.signed_at}`);
}

main().catch(console.error);
```

Expected output:

```
EF:US:diesel:2024:v1: 10.21 kg/gal
Edition: 2027.Q1-electricity
Receipt: sha256-hmac @ 2026-04-22T14:33:02Z
```

---

## Error handling

```ts
import { FactorsAPIError } from "@greenlang/factors";

try {
  const resolved = await client.resolveExplain(req);
} catch (err) {
  if (err instanceof FactorsAPIError) {
    if (err.statusCode === 429) {
      console.log(`Rate limited, retry after ${err.retryAfter}s`);
    } else if (err.statusCode === 451) {
      console.log("License forbids this factor - upgrade or switch connector");
    } else {
      console.log(`${err.statusCode} ${err.errorCode}: ${err.message}`);
    }
  } else {
    throw err;
  }
}
```

---

## Pagination

All list endpoints expose an `OffsetPaginator` (see `greenlang/factors/sdk/ts/src/pagination.ts`):

```ts
const it = client.searchPaginated({ q: "natural gas US" });

for await (const factor of it) {
  console.log(factor.factor_id);
}
```

---

## Next steps

- [Python SDK quickstart](./python-sdk.md) for the same flow in Python.
- [cURL recipes](./curl-recipes.md) for shell scripts and CI jobs.
- [Resolution cascade](../concepts/resolution-cascade.md) to understand what the engine does.
- [Method packs](../concepts/method-packs.md) to pick the right `method_profile`.

---

## File citations

| SDK piece | File |
|---|---|
| `FactorsClient`, `AsyncFactorsClient` equivalent | `greenlang/factors/sdk/ts/src/client.ts` |
| Auth providers | `greenlang/factors/sdk/ts/src/auth.ts` |
| HTTP transport (retries, ETag cache) | `greenlang/factors/sdk/ts/src/transport.ts` |
| Models | `greenlang/factors/sdk/ts/src/models.ts` |
| Errors | `greenlang/factors/sdk/ts/src/errors.ts` |
| Canonical JSON for receipt hashing | `greenlang/factors/sdk/ts/src/canonical.ts`, `hash.ts` |
| CLI (`glfactors` bin) | `greenlang/factors/sdk/ts/src/cli.ts` |
| Package manifest | `greenlang/factors/sdk/ts/package.json` |
