# TypeScript SDK — `@greenlang/factors`

Official TypeScript/JavaScript SDK for the GreenLang Factors API. Tested against Node 18, 20, 22. Ships dual ESM + CJS, with full type declarations.

**Canonical changelog:** [`greenlang/factors/sdk/CHANGELOG.md`](../../../greenlang/factors/sdk/CHANGELOG.md).

---

## Install

```bash
npm install @greenlang/factors@1.2.0
# or
pnpm add @greenlang/factors@1.2.0
# or
yarn add @greenlang/factors@1.2.0
```

Pin the minor. The `1.x` line maintains API stability; 2.0 will remove the back-compat receipt-key aliases documented in the SDK changelog.

---

## Auth

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  apiKey: process.env.GL_API_KEY!,
  baseUrl: "https://api.greenlang.io",
});
```

OAuth2:

```ts
const client = await FactorsClient.fromOAuth({
  clientId: process.env.GL_CLIENT_ID!,
  clientSecret: process.env.GL_CLIENT_SECRET!,
});
```

See [`authentication.md`](../authentication.md).

---

## Resolve

```ts
const result = await client.resolve({
  factorFamily: "electricity",
  quantity: 12500,
  unit: "kWh",
  methodProfile: "corporate_scope2_location_based",
  jurisdiction: "IN",
  validAt: "2026-12-31",
});

console.log(result.chosenFactor.factorId);
console.log(result.chosenFactor.factorVersion);
console.log(result.chosenFactor.releaseVersion);
console.log(result.emissions.co2eKg);
console.log(result.quality.compositeFqs0_100);
console.log(result.fallbackRank);
console.log(result.licensing.redistributionClass);
console.log(result.assumptions);
console.log(result.auditText);
console.log(result.auditTextDraft);
```

See [`api-reference/resolve.md`](../api-reference/resolve.md).

---

## Explain

```ts
const explanation = await client.explain({
  factorId: "EF:IN:grid:CEA:FY2024-25:v1",
  methodProfile: "corporate_scope2_location_based",
  quantity: 12500,
  unit: "kWh",
  jurisdiction: "IN",
});

for (const tier of explanation.cascade) {
  console.log(tier.rank, tier.label, tier.outcome);
}
```

---

## Verify a signed receipt (offline)

```ts
import { verifyReceipt } from "@greenlang/factors";

const verified = await verifyReceipt(result.rawResponse);
console.log(verified.valid);
console.log(verified.alg);                    // Ed25519 or HS256
console.log(verified.verificationKeyHint);    // key id
console.log(verified.algorithm);              // legacy alias; removed in 2.0
```

`verifyReceipt` uses `jose` for Ed25519 (JWKS fetched from `/.well-known/jwks.json`) and `crypto` for HS256. Configure via `{ jwksUrl, hmacSecret }` options. See [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).

---

## Batch

```ts
const batch = await client.batchResolve({
  items: [
    { rowId: "r1", factorFamily: "electricity", quantity: 12500, unit: "kWh",
      methodProfile: "corporate_scope2_location_based", jurisdiction: "IN",
      validAt: "2026-12-31" },
  ],
});

for (const r of batch.results) { console.log(r.rowId, r.emissions.co2eKg); }
for (const e of batch.errors)   { console.log(e.rowId, e.errorCode); }
```

See [`api-reference/batch.md`](../api-reference/batch.md).

---

## Errors

The SDK exports a typed exception hierarchy:

```ts
import {
  FactorCannotResolveSafelyError,
  LicensingGapError,
  EntitlementError,
  EditionMismatchError,
  EditionPinError,
  RateLimitError,
  UnauthorizedError,
  BadRequestError,
} from "@greenlang/factors";

try {
  await client.resolve(...);
} catch (e) {
  if (e instanceof FactorCannotResolveSafelyError) {
    console.log(e.packId, e.methodProfile, e.evaluatedCandidatesCount);
  } else if (e instanceof RateLimitError) {
    await new Promise(r => setTimeout(r, e.retryAfter * 1000));
  } else { throw e; }
}
```

See full mapping in [`error-codes.md`](../error-codes.md).

---

## Version pinning

```ts
const pinned = client.pinEdition("builtin-v1.0.0");
const result = await pinned.resolve(...);

// Or scoped
await client.withEdition("builtin-v1.0.0", async (c) => {
  return c.resolve(...);
});
```

---

## Rate-limit-aware retries

The transport honours `Retry-After` on 429 responses (default: 1 retry). For caller-side backoff, catch `RateLimitError` and wait `retryAfter` seconds.

---

## Runtime requirements

- Node 18+ (ESM or CJS).
- Browsers: supported via bundler (webpack, Vite, esbuild). TLS 1.3 required.

---

## Related

- [SDK changelog](../../../greenlang/factors/sdk/CHANGELOG.md).
- [Python SDK](python.md), [CLI](cli.md).
- [`api-reference/resolve.md`](../api-reference/resolve.md), [`api-reference/batch.md`](../api-reference/batch.md).
