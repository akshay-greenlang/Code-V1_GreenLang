---
title: "SDK: TypeScript"
description: "@greenlang/factors -- the typed TS / Node client for the GreenLang Factors API."
---

# TypeScript SDK -- `@greenlang/factors`

```sh
npm install @greenlang/factors@1.0.0
```

Requires Node 18+. Dual-bundled (ESM + CJS) with full type declarations.

## Imports

```ts
import {
  FactorsClient,
  FactorsAPIError,
  AuthError,
  TierError,
  LicenseError,
  LicensingGapError,
  EntitlementError,
  FactorNotFoundError,
  ValidationError,
  RateLimitError,
  EditionPinError,
  EditionMismatchError,
  ReceiptVerificationError,
  verifyReceipt,
  computeSubresourceIntegrity,
} from "@greenlang/factors";
```

## Constructor

```ts
const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_FACTORS_API_KEY!, // or jwtToken
  defaultEdition: undefined,
  pinnedEdition: undefined,
  verifyGreenlangCert: true,                // TLS cert pinning (Node only)
  timeoutMs: 30_000,
  maxRetries: 3,
});
```

## Method surface (parity with Python)

```ts
const hits      = await client.search("diesel");
const all       = await client.searchV2("diesel", { jurisdiction: "US" });
const factor    = await client.getFactor("ef:co2:diesel:us:2026");
const matched   = await client.match("natural gas combustion");
const coverage  = await client.coverage();
const resolved  = await client.resolve({ activity: "diesel", ... });
const explain   = await client.resolveExplain("ef:co2:diesel:us:2026", { alternates: 5 });
const editions  = await client.listEditions({ includePending: false });
const diff      = await client.diff("ef:...", "2026.Q4", "2027.Q1");
const bundle    = await client.auditBundle("ef:...");
```

## Edition pinning

```ts
await client.withEdition("2027.Q1-electricity", async (scoped) => {
  const resolved = await scoped.resolve(request);
  return resolved;
});
```

## Receipt verification

```ts
const response = await client.resolve(request);
const summary  = await client.verifyReceipt(response, {
  // secret: ...     // for HMAC
  // jwksUrl: ...    // for Ed25519
});
console.log(summary.verified, summary.key_id);
```

For Ed25519 receipts the SDK uses the `jose` peer dependency:

```sh
npm install jose
```

## CLI

```sh
glfactors search "diesel"
glfactors resolve "natural gas" --jurisdiction US
glfactors explain ef:co2:diesel:us:2026 --alternates 5
```
