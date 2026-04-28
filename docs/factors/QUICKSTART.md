# GreenLang Factors — 5-Minute Quickstart

This is the fastest path from zero to your first factor resolution with full
explainability, edition pinning, and signed receipts.

If you've never touched GreenLang before, set aside 5 minutes. By the end you'll
have:

1. Resolved a real activity (12,500 kWh of electricity in India) into a CO2e
   value via the `/resolve-explain` API.
2. Read the explain trace — chosen factor, alternates considered, why-this-won,
   source/version, Factor Quality Score, per-gas breakdown, GWP basis.
3. Pinned a specific edition so the same call is byte-for-byte reproducible
   next year.

---

## 1. Install

### Python (>= 3.10)

```bash
pip install greenlang-factors-sdk
```

The wheel is stdlib-only — no transitive dependencies. Confirmed package name
is `greenlang-factors-sdk` (`greenlang/factors/sdk/pyproject.toml`,
v1.1.0). The import path is `greenlang.factors.sdk.python`.

### TypeScript / Node (>= 18)

```bash
npm install @greenlang/factors
```

Browser builds work too — see
[BROWSER_VS_NODE.md](../sdk/typescript/BROWSER_VS_NODE.md). Optional `undici`
peer for Node fetch hardening.

### CLI (bundled with the Python SDK)

```bash
glfactors --help
```

---

## 2. Get an API key

Free Community-tier keys are issued from the pricing page (rate limit:
60 RPM, 10 burst, 1 export per
15 minutes — see `greenlang/factors/middleware/rate_limiter.py`).

Pro and higher tiers are purchased via Stripe Checkout from the same page;
Enterprise is a sales-led contract with SSO/SCIM, VPC deployment, and a private
factor registry. Open-core vs. enterprise scope is documented in
[OPEN_CORE_BOUNDARY.md](OPEN_CORE_BOUNDARY.md).

```bash
export GREENLANG_API_KEY=glk_live_...
```

---

## 3. Your first factor resolution

Resolve 12,500 kWh of grid electricity consumed in India during FY27 under
GHG Protocol Scope 2 location-based methodology.

### Python

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="glk_live_...",
)

result = client.resolve(
    request={
        "activity": "purchased grid electricity",
        "method_profile": "corporate_scope2_location_based",
        "jurisdiction": "IN",
        "quantity": 12500,
        "unit": "kWh",
        "reporting_date": "2027-06-30",
    },
    alternates=True,   # return the runner-up factors too
)

print(result.chosen_factor_id, result.gas_breakdown.co2e_kg)
```

### TypeScript

```typescript
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: "glk_live_...",
});

const result = await client.resolve(
  {
    activity: "purchased grid electricity",
    method_profile: "corporate_scope2_location_based",
    jurisdiction: "IN",
    quantity: 12500,
    unit: "kWh",
    reporting_date: "2027-06-30",
  },
  { alternates: true },
);

console.log(result.chosen_factor_id, result.gas_breakdown.co2e_kg);
```

### curl

```bash
curl -X POST https://api.greenlang.io/api/v1/factors/resolve-explain \
  -H "Authorization: Bearer ${GREENLANG_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "activity": "purchased grid electricity",
    "method_profile": "corporate_scope2_location_based",
    "jurisdiction": "IN",
    "quantity": 12500,
    "unit": "kWh",
    "reporting_date": "2027-06-30",
    "alternates": true
  }'
```

`activity` is a free-text description (the resolution engine maps it to a
factor family via the mapping layer). For other examples, swap in any of the
seven v1 Certified slices:

| Slice            | activity                                | method_profile                          | jurisdiction |
| ---------------- | --------------------------------------- | --------------------------------------- | ------------ |
| Electricity      | `"purchased grid electricity"`          | `corporate_scope2_location_based`       | `IN`         |
| Combustion       | `"diesel B7 stationary genset"`         | `corporate_scope1`                      | `EU`         |
| Freight          | `"sea container TEU Asia to Rotterdam"` | `freight_iso_14083`                     | `EU`         |
| Material / CBAM  | `"hot-rolled steel coil"`               | `eu_cbam`                               | `EU`         |
| Land             | `"reforestation tropical hardwood"`     | `land_removals`                         | `BR`         |
| Product          | `"smartphone cradle-to-gate PCF"`       | `product_carbon`                        | `CN`         |
| Finance          | `"listed equity portfolio investee"`    | `finance_proxy`                         | `US`         |

The full canonical request shape is in
[`docs/api/factors-v1.yaml`](../api/factors-v1.yaml#/components/schemas/ResolutionRequest)
and the canonical factor record is in
[`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md).

---

## 4. Read the explain trace

Every response carries a full audit trace — never just a number. Inspect:

```python
print(result.chosen_factor_id)        # e.g. "EF:IN:electricity:cea:v12:2027"
print(result.step_label)              # which cascade step won (1..7)
print(result.fallback_rank)           # 1 = best match
print(result.why_chosen)              # one-sentence rationale
for alt in result.alternates:
    print(alt.factor_id, alt.score)   # runners-up

# Per-gas breakdown — never just CO2e (CTO non-negotiable)
print(result.gas_breakdown.co2_kg)
print(result.gas_breakdown.ch4_kg)
print(result.gas_breakdown.n2o_kg)
print(result.gas_breakdown.co2e_kg)
print(result.gas_breakdown.gwp_basis)  # e.g. "IPCC_AR6_100"

# Quality + caveats
print(result.quality_score)            # composite FQS 0-100
print(result.assumptions)              # list[str]
print(result.deprecation_status)
print(result.edition_id)               # e.g. "2027.04.0"
```

The 7-step cascade is: **(1)** customer-specific override → **(2)** supplier
factor → **(3)** facility/asset factor → **(4)** utility / grid-subregion →
**(5)** country / sector → **(6)** method-pack default → **(7)** global default.
Tie-breaking weighs geography, time, technology, unit compatibility, methodology
fit, source authority, verification, uncertainty, recency, and license
availability.

The composite Factor Quality Score (0-100) blends temporal, geographic,
technological, representativeness, and methodological dimensions — see
[developer_guide.md → Quality](developer_guide.md).

---

## 5. Pin an edition for reproducibility

Inventories filed with regulators must be re-runnable years later, byte-for-byte
identical (CTO non-negotiable: *never overwrite a factor; version everything*).
Pin the edition you used at filing time and every future call returns the same
data.

### Python — single call

```python
result = client.resolve(request, edition="2027.04.0")
```

### Python — sticky pin for a session

```python
client.pin_edition("2027.04.0")
# all subsequent calls on this client use 2027.04.0 unless overridden
```

### Python — scoped context manager

```python
with client.edition("2027.04.0"):
    result_a = client.resolve(request_a)
    result_b = client.resolve(request_b)
# pin reverts after the block
```

### TypeScript — scoped

```typescript
await client.edition("2027.04.0", async (scoped) => {
  const a = await scoped.resolve(reqA);
  const b = await scoped.resolve(reqB);
});
```

### curl

```bash
curl ... -H "X-GreenLang-Edition: 2027.04.0" ...
```

If the requested edition is unavailable for your tier the SDK raises
`EditionMismatchError`. Deeper coverage:
[`docs/sdk/python/VERSION_PINNING.md`](../sdk/python/VERSION_PINNING.md) and
[`docs/sdk/typescript/VERSION_PINNING.md`](../sdk/typescript/VERSION_PINNING.md).

---

## 6. Subscribe to factor-change webhooks

When a method-pack publisher (DEFRA, EPA, IPCC, India CEA, AIB residual mix...)
issues a refresh, GreenLang re-ingests, re-validates, and emits webhooks so you
can re-run affected inventories instead of polling.

The Python and TypeScript SDKs ship the **verifier** helpers but not a
subscription client — subscribe via REST:

```bash
curl -X POST https://api.greenlang.io/api/v1/webhooks/subscribe \
  -H "Authorization: Bearer ${GREENLANG_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.example.com/hooks/greenlang",
    "events": ["factor.updated", "factor.deprecated", "edition.published"],
    "secret": "whsec_..."
  }'
```

Verify deliveries on your end (HMAC-SHA256 over canonical JSON, sent in the
`X-GL-Signature: sha256=<hex>` header):

```python
from greenlang.factors.sdk.python import verify_webhook_strict

ok = verify_webhook_strict(
    payload=raw_body_bytes,
    signature=request.headers["X-GL-Signature"],
    secret="whsec_...",
)
```

```typescript
import { verifyWebhookStrict } from "@greenlang/factors";

const ok = verifyWebhookStrict({
  payload: rawBody,
  signature: req.headers["x-gl-signature"] as string,
  secret: "whsec_...",
});
```

Event taxonomy and delivery semantics:
[`docs/factors/watch-pipeline-ops.md`](watch-pipeline-ops.md).

---

## 7. Verify a signed receipt

Every successful response on `/api/v1/factors` carries a tamper-evident receipt
the platform signs over `{body_hash, edition_id, path, method, status_code}`.
You see the receipt in two places:

- **JSON envelope:** `_signed_receipt` field on the response body.
- **Headers:** `X-GreenLang-Receipt-Signature`, `X-GreenLang-Receipt-Algorithm`,
  `X-GreenLang-Receipt-Key-Id`, `X-GreenLang-Receipt-Signed-At`,
  `X-GreenLang-Receipt-Hash`.

Algorithm by tier (`greenlang/factors/middleware/signed_receipts.py`):

| Tier                                       | Algorithm                                |
| ------------------------------------------ | ---------------------------------------- |
| Community / Pro / Internal                 | HMAC-SHA256                              |
| Consulting / Platform / Enterprise         | Ed25519 (HMAC-SHA256 fallback)           |

Verify HMAC receipts with the bundled helper:

```python
from greenlang.factors.signing import verify_sha256_hmac

ok = verify_sha256_hmac(
    payload=response_body_bytes,
    receipt=response_json["_signed_receipt"],
    secret="your_receipt_signing_secret",
)
```

Ed25519 verification uses your standard crypto library plus the public key
served from `/api/v1/factors/keys` — covered in
[hosted_api.md → Receipts & Verification](hosted_api.md).

---

## 8. Where to next

- **Method packs** — the seven v1 profiles each encode their own selection,
  boundary, and fallback rules. Source:
  [`greenlang/factors/method_packs/`](../../greenlang/factors/method_packs/)
  (`corporate.py`, `electricity.py`, `freight.py`, `eu_policy.py`,
  `land_removals.py`, `product_carbon.py`, `finance_proxy.py`).
- **Full API reference** — [`docs/api/factors-v1.yaml`](../api/factors-v1.yaml)
  (OpenAPI 3.1).
- **Canonical factor record** —
  [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md).
- **SDK references** — Python: [`docs/sdk/python/`](../sdk/python/) ·
  TypeScript: [`docs/sdk/typescript/`](../sdk/typescript/).
- **Long-form developer guide** —
  [`docs/factors/developer_guide.md`](developer_guide.md).
- **Commercial features by tier** —
  [`docs/factors/commercial_feature_matrix.md`](commercial_feature_matrix.md).
- **OEM / white-label deployment** —
  [`docs/factors/OEM_DEPLOYMENT.md`](OEM_DEPLOYMENT.md).
- **Pricing & SKUs** — pricing page.
