# Quickstart — resolve a factor in 10 minutes

This path takes you from zero to a verified signed receipt for the canonical demo request: **12,500 kWh, India, FY2027, Corporate Inventory Scope 2 location-based.**

Every snippet below runs against staging (`https://api.staging.greenlang.io`).

---

## 1. Sign up and get an API key (2 min)

1. Go to `https://app.greenlang.io/signup`.
2. Confirm your email.
3. In the dashboard, open **Settings → API Keys → Create key**. Copy the key; it starts with `glk_`. Store it in `GL_API_KEY`.

```bash
export GL_API_KEY="glk_...."
```

---

## 2. Install the SDK (1 min)

**Python 3.10+:**

```bash
pip install greenlang-factors==1.2.0
```

**Node 18+:**

```bash
npm install @greenlang/factors@1.2.0
```

Pin the minor. See [`sdks/python.md`](sdks/python.md) and [`sdks/typescript.md`](sdks/typescript.md) for details on authentication and version-pinning.

---

## 3. Resolve the canonical demo factor (4 min)

**Python:**

```python
from greenlang_factors import FactorsClient

client = FactorsClient(
    api_key=os.environ["GL_API_KEY"],
    base_url="https://api.staging.greenlang.io",
)

result = client.resolve(
    factor_family="electricity",
    quantity=12500,
    unit="kWh",
    method_profile="corporate_scope2_location_based",
    jurisdiction="IN",
    valid_at="2026-12-31",
)

print(result.chosen_factor.factor_id)          # e.g., EF:IN:grid:CEA:FY2024-25:v1
print(result.emissions.co2e_kg)                 # e.g., 9950.0
print(result.quality.composite_fqs_0_100)       # e.g., 82.0
print(result.signed_receipt.receipt_id)         # proof envelope
```

**TypeScript:**

```ts
import { FactorsClient } from "@greenlang/factors";

const client = new FactorsClient({
  apiKey: process.env.GL_API_KEY!,
  baseUrl: "https://api.staging.greenlang.io",
});

const result = await client.resolve({
  factorFamily: "electricity",
  quantity: 12500,
  unit: "kWh",
  methodProfile: "corporate_scope2_location_based",
  jurisdiction: "IN",
  validAt: "2026-12-31",
});

console.log(result.chosenFactor.factorId);
console.log(result.emissions.co2eKg);
console.log(result.signedReceipt.receiptId);
```

**curl:**

```bash
curl -X POST "https://api.staging.greenlang.io/v1/factors/resolve" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_family": "electricity",
    "quantity": 12500,
    "unit": "kWh",
    "method_profile": "corporate_scope2_location_based",
    "jurisdiction": "IN",
    "valid_at": "2026-12-31"
  }'
```

You should receive a 200 with `chosen_factor`, `alternates[]`, `emissions`, `quality`, `licensing`, `assumptions[]`, `fallback_rank`, and `signed_receipt`. That envelope matches the CTO definition of done (section 30, canonical demo).

See [`api-reference/resolve.md`](api-reference/resolve.md) for field-by-field reference.

---

## 4. Verify the signed receipt (3 min)

The receipt proves the server signed the payload with a key that resolves via `https://api.greenlang.io/.well-known/jwks.json` (or an HMAC shared secret for private deployments). Verification runs **offline** — no network call required once the JWKS is cached.

**Python:**

```python
from greenlang_factors.verify import verify_receipt

verified = verify_receipt(result.raw_response)
assert verified.valid is True
print(verified.alg)                  # "Ed25519" or "HS256"
print(verified.verification_key_hint)
```

**CLI:**

```bash
gl-factors verify-receipt ./response.json
# Prints: valid=true alg=Ed25519 verification_key_hint=jwk-2026Q2-primary
```

If `verify_receipt` returns `valid=false`, the response has been tampered with or the key rotated. See [`concepts/signed_receipt.md`](concepts/signed_receipt.md).

---

## You now have

- A working API key
- An installed SDK
- A resolved factor with full gas vector, quality score, and assumptions
- A verified signed receipt

That is the minimal surface every customer ships against. The same four steps work in production — change the base URL to `https://api.greenlang.io` and use a production key.

**Next:** read [`concepts/factor.md`](concepts/factor.md) and [`concepts/method_pack.md`](concepts/method_pack.md) to understand what the resolver actually did; read [`api-reference/explain.md`](api-reference/explain.md) to trace the 7-step cascade that picked this factor.
