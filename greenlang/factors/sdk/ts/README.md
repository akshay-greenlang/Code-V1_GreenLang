# @greenlang/factors -- TypeScript SDK (v1.0.0)

Production-grade TypeScript / Node client for the [GreenLang Factors REST API](https://developers.greenlang.ai). Search, resolve, and audit emission factors across the global open + licensed catalog with edition pinning, signed-receipt verification, and rate-limit-aware retries.

```sh
npm install @greenlang/factors
# or
pnpm add @greenlang/factors
# or
yarn add @greenlang/factors
```

Requires Node 18+. Dual-bundled (ESM + CJS) with full type declarations.

## 60-second quickstart

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_FACTORS_API_KEY!,
});

// 1. Search the catalog
const hits = await client.search("natural gas US Scope 1", { limit: 5 });
for (const f of hits.factors) {
  console.log(f.factor_id, f.co2e_per_unit, f.unit);
}

// 2. Resolve an activity into a chosen factor
const resolved = await client.resolve({
  activity: "natural gas combustion",
  method_profile: "corporate_scope1",
  jurisdiction: "US",
  reporting_date: "2026-04-01",
  quantity: 1000,
  unit: "therm",
});
console.log("chosen:", resolved.chosen.factor_id);
console.log("co2e:", resolved.computed_total);
```

## Edition pinning

Pin every request to a specific catalog edition so reports remain reproducible across catalog updates:

```ts
await client.withEdition("2027.Q1-electricity", async (scoped) => {
  const resolved = await scoped.resolve({
    activity: "electricity consumption",
    method_profile: "corporate_scope2_location_based",
    jurisdiction: "US-CA",
    quantity: 5000,
    unit: "kWh",
  });
  // If the server returns a different edition than the pin, an
  // EditionMismatchError is thrown -- we never silently accept drift.
});
```

The accepted edition-id formats are:

* `v1.0.0`, `v1`, `v2.1` -- semantic-version style
* `2027.Q1`, `2027.Q1-electricity` -- quarterly + scope
* `2027-04-01-freight` -- date + scope

Anything else throws `EditionPinError` before the request goes out.

## Offline signed-receipt verification

Every Pro+ response can carry a signed receipt. Verify it offline -- no network call back to GreenLang -- so audit packages remain self-contained.

```ts
import { ReceiptVerificationError } from "@greenlang/factors";

const response = await client.resolve(request);
try {
  const summary = await client.verifyReceipt(response, {
    // secret: ...     for HMAC-SHA256
    // jwksUrl: ...    for Ed25519 (defaults to GreenLang's public JWKS)
  });
  console.log("verified by", summary.key_id, "at", summary.signed_at);
} catch (err) {
  if (err instanceof ReceiptVerificationError) {
    console.error("AUDIT FAILURE:", err.message);
  }
}
```

Two algorithms supported:

| Algorithm    | Tier                         | Key material                                            |
|--------------|------------------------------|---------------------------------------------------------|
| HMAC-SHA256  | Community / Developer Pro    | Shared secret (`GL_FACTORS_SIGNING_SECRET`)             |
| Ed25519      | Consulting / Platform / Ent. | JWKS at `https://api.greenlang.io/.well-known/jwks.json`|

The Ed25519 path uses the optional `jose` peer dependency:

```sh
npm install jose
```

## Rate-limit-aware retries

Built-in: when the server returns `429 Too Many Requests`, the transport reads `Retry-After` and waits exactly that long before retrying (capped at 60s, up to `maxRetries` attempts). On the final attempt a `RateLimitError` exposes the `retryAfter` field so caller code can also back off.

```ts
import { RateLimitError } from "@greenlang/factors";

try {
  await client.resolve(request);
} catch (err) {
  if (err instanceof RateLimitError) {
    console.log(`slow down -- retry after ${err.retryAfter}s`);
  }
}
```

## Typed exceptions

| Exception              | Trigger                                                           |
|------------------------|-------------------------------------------------------------------|
| `AuthError`            | 401 -- bad/missing API key or JWT                                 |
| `TierError`            | 403 -- caller's tier insufficient                                 |
| `LicenseError`         | 403 -- factor is `connector_only` and caller lacks permission     |
| `LicensingGapError`    | 403 -- requested licensed pack not in contract                    |
| `EntitlementError`     | 403 -- plan does not include the requested feature                |
| `FactorNotFoundError`  | 404 -- factor id missing in this edition                          |
| `ValidationError`      | 400 / 422 -- bad request body                                     |
| `RateLimitError`       | 429 -- exceeded tier rate limit                                   |
| `EditionPinError`      | client-side: bad edition id, or 409/410 from server               |
| `EditionMismatchError` | server returned a different edition than the pin                  |
| `FactorsAPIError`      | catch-all base class                                              |

## CLI

```
glfactors search "diesel US Scope 1"
glfactors get-factor ef:co2:diesel:us:2026
glfactors resolve "natural gas combustion" --jurisdiction US
glfactors explain ef:co2:elec:us-ca:2027 --alternates 5
glfactors list-editions
```

Authentication is sourced from environment variables (mirrors the Python SDK):

```
GREENLANG_FACTORS_BASE_URL    # default: http://localhost:8000
GREENLANG_FACTORS_API_KEY
GREENLANG_FACTORS_JWT
GREENLANG_FACTORS_EDITION
GL_FACTORS_SIGNING_SECRET     # for HMAC receipt verification
GL_FACTORS_JWKS_URL           # for Ed25519 receipt verification
```

## Links

* Pricing -- https://greenlang.ai/pricing
* Documentation -- https://developers.greenlang.ai
* Changelog -- https://github.com/greenlang/greenlang/blob/master/greenlang/factors/sdk/CHANGELOG.md
* Source -- https://github.com/greenlang/greenlang
