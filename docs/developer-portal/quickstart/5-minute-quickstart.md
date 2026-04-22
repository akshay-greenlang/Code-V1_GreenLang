# 5-Minute Quickstart

**Goal:** resolve your first emission factor, extract the signed receipt, verify it with HMAC.

**What you need:** `curl`, `jq`, and `openssl` (all standard on macOS / Linux; on Windows use WSL or Git Bash). No SDK.

**What you will learn:**

1. Authenticate with an API key.
2. POST a resolution request for diesel combustion in the US.
3. Pin the response to an edition with `X-Factors-Edition`.
4. Verify the `_signed_receipt` the API attached to your response.

---

## 0. Prerequisites

```bash
export GL_API_BASE="https://api.greenlang.io"
export GL_API_KEY="gl_pk_your_key_here"          # Dashboard -> API Keys
export GL_FACTORS_SIGNING_SECRET="shh_hmac_key"   # Dashboard -> Signing Secrets
```

If you do not have keys yet, run the pilot flow: `POST /api/v1/onboarding/signup` or ask your admin.

---

## 1. Resolve a factor (one call)

The `/resolve-explain` endpoint runs the full 7-step cascade and returns the winning factor plus up to 20 alternates. It requires **Pro, Consulting, Enterprise, or Internal** tier.

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: 2027.Q1-electricity" \
  -d '{
    "activity": "diesel combustion stationary",
    "method_profile": "corporate_scope1",
    "jurisdiction": "US",
    "reporting_date": "2026-06-01"
  }' \
  -o resolved.json -D response.headers
```

**Look at the response headers** — `response.headers` now contains:

```
X-GreenLang-Edition: 2027.Q1-electricity
X-Factors-Edition: 2027.Q1-electricity
X-GreenLang-Method-Profile: corporate_scope1
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 599
```

**Inspect the body** — `resolved.json` includes the chosen factor and the signed receipt:

```bash
jq '{factor_id: .factor_id, co2e: .co2e_per_unit, receipt: ._signed_receipt}' resolved.json
```

```json
{
  "factor_id": "EF:US:diesel:2024:v1",
  "co2e": 10.21,
  "receipt": {
    "signature": "Q2FtYnJpZGdlQW5hbHl0aWNh...",
    "algorithm": "sha256-hmac",
    "signed_at": "2026-04-22T14:33:02Z",
    "key_id": "gl-factors-v1",
    "payload_hash": "b31d...9f0a"
  }
}
```

If you got a `403` here, your key is on the Community tier. See [authentication](../api-reference/authentication.md).

---

## 2. Pin an edition (for audit)

The request header `X-Factors-Edition: 2027.Q1-electricity` told the API "use this immutable snapshot." The response echoes the edition back on `X-GreenLang-Edition` and the signed payload includes `edition_id` so tampering is detectable.

Any submission you hand to an auditor should cite the edition id. Save it:

```bash
EDITION_ID=$(grep -i '^X-GreenLang-Edition:' response.headers | awk '{print $2}' | tr -d '\r')
echo "Pinned to: $EDITION_ID"
```

See [version-pinning](../concepts/version-pinning.md) for rollback semantics.

---

## 3. Verify the signed receipt

GreenLang signs every 2xx JSON response with either HMAC-SHA256 (open-core / Pro) or Ed25519 (Consulting / Enterprise / Platform). The receipt binds the response hash, the edition id, and a timestamp. Verify in three steps.

### 3a. Extract the receipt, strip it, canonicalise the body

The receipt is injected as top-level `_signed_receipt`. Remove it, re-serialise deterministically, and hash.

```bash
# Strip the receipt and compute the payload hash the server would have signed.
jq 'del(._signed_receipt)' resolved.json \
  | jq -S -c '.' \
  > payload.json

PAYLOAD_HASH=$(openssl dgst -sha256 -hex payload.json | awk '{print $2}')
RECEIPT_HASH=$(jq -r '._signed_receipt.payload_hash' resolved.json)

[ "$PAYLOAD_HASH" = "$RECEIPT_HASH" ] \
  && echo "OK: payload hash matches receipt" \
  || { echo "FAIL: payload tampered"; exit 1; }
```

Server-side canonicalisation is `json.dumps(payload, sort_keys=True, default=str)` — see `_canonical_hash` in `greenlang/factors/signing.py`. `jq -S -c` gives you the equivalent key-sorted compact JSON.

### 3b. Re-derive the HMAC

```bash
SIG=$(jq -r '._signed_receipt.signature' resolved.json)
ALG=$(jq -r '._signed_receipt.algorithm' resolved.json)

if [ "$ALG" = "sha256-hmac" ]; then
  EXPECTED=$(openssl dgst -sha256 -hmac "$GL_FACTORS_SIGNING_SECRET" -binary payload.json | base64)
  [ "$SIG" = "$EXPECTED" ] && echo "OK: HMAC signature valid" || echo "FAIL: HMAC mismatch"
else
  echo "Response used $ALG - see docs/developer-portal/concepts/signed-receipts.md for Ed25519 verification"
fi
```

### 3c. Check the edition binding

The server embeds `edition_id` into the payload it signs (see `middleware/signed_receipts.py`, "Edition-pin into the receipt"). If anyone rewrites the `X-GreenLang-Edition` header or the body's edition after the fact, step 3a fails.

```bash
jq '.edition_id // "(not echoed in body)"' resolved.json
```

---

## 4. You are done

You have:

- Authenticated with an API key.
- Pinned your resolution to an immutable edition.
- Received and verified a cryptographically signed receipt.
- Produced audit-ready evidence of exactly which factor was used and when.

Total wall-clock: under 5 minutes with keys already in hand.

---

## Next steps

- **Understand the cascade that picked this factor** &rarr; [Resolution cascade](../concepts/resolution-cascade.md)
- **Use the Python SDK instead of curl** &rarr; [Python SDK quickstart](./python-sdk.md)
- **Pick the right method profile** (Scope 1 vs 2 vs 3 vs CBAM) &rarr; [Method packs](../concepts/method-packs.md)
- **CI/CD integration** &rarr; [cURL recipes](./curl-recipes.md)
- **Error handling** &rarr; [Errors](../api-reference/errors.md)

---

## File citations

| Thing you used | Where it lives |
|---|---|
| `/api/v1/factors/resolve-explain` | `greenlang/integration/api/routes/factors.py` (line 987) |
| 7-step cascade | `greenlang/factors/resolution/engine.py` |
| Signed receipt injection | `greenlang/factors/middleware/signed_receipts.py` |
| HMAC signing | `greenlang/factors/signing.py::sign_sha256_hmac` |
| Edition id pinning | `greenlang/factors/service.py::resolve_edition_id`, `greenlang/factors/edition_manifest.py` |
| Tier gate | `greenlang/factors/tier_enforcement.py` |
