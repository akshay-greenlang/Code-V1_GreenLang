# Signed Receipts

Every 2xx response from the GreenLang Factors API carries cryptographic proof of **what** we returned and **when**. The receipt binds a SHA-256 hash of the response body, the edition id used to produce it, a timestamp, and a signing key id. Clients can re-derive and verify that signature offline months later.

Two algorithms are supported:

- **HMAC-SHA256** (symmetric, Pro and Community tiers) — fast, shared secret.
- **Ed25519** (asymmetric, Consulting / Enterprise / Platform tiers) — GreenLang signs with a private key held in Vault; customers verify with the published public key.

**Signing core:** `greenlang/factors/signing.py`
**FastAPI middleware that attaches receipts to responses:** `greenlang/factors/middleware/signed_receipts.py`

---

## Receipt shape

Receipts are injected into JSON responses as a top-level `_signed_receipt` object:

```json
{
  "factor_id": "EF:US:diesel:2024:v1",
  "co2e_per_unit": 10.21,
  "edition_id": "2027.Q1-electricity",
  "_signed_receipt": {
    "signature": "Q2FtYnJpZGdlQW5hbHl0aWNh...",
    "algorithm": "sha256-hmac",
    "signed_at": "2026-04-22T14:33:02Z",
    "key_id": "gl-factors-v1",
    "payload_hash": "b31d9a6f7c2e..."
  }
}
```

For non-JSON responses (streaming, NDJSON, binary), the receipt is delivered as HTTP headers instead:

- `X-GreenLang-Receipt-Signature`
- `X-GreenLang-Receipt-Algorithm`
- `X-GreenLang-Receipt-Key-Id`
- `X-GreenLang-Receipt-Signed-At`
- `X-GreenLang-Receipt-Hash`

---

## What is signed

The receipt's `payload_hash` is the SHA-256 of `json.dumps(payload, sort_keys=True, default=str)` — i.e. a canonical, key-sorted, UTF-8 encoded JSON serialisation. The helper is `_canonical_hash` in `greenlang/factors/signing.py`.

Crucially, the payload **includes `edition_id`** when the response carries `X-GreenLang-Edition`. See `middleware/signed_receipts.py`:

> **Edition-pin into the receipt.** If the response carries `X-GreenLang-Edition` (set by the explain / resolve / quality / detail routes), the signed payload includes `edition_id` so customers can independently verify "this response was built from edition X at time Y". Tampering with the edition header after the fact invalidates the signature.

So if a middleman rewrites either the body's numbers or the edition header, step 3 below fails.

---

## Algorithm per tier

| Tier | Algorithm | Notes |
|---|---|---|
| community | `sha256-hmac` | Single shared secret per environment. |
| pro | `sha256-hmac` | Tenant-scoped secret rotations. |
| consulting | `ed25519` (fallback `sha256-hmac`) | Ed25519 preferred; falls back if private key unset. |
| platform | `ed25519` (fallback `sha256-hmac`) | Same. |
| enterprise | `ed25519` (fallback `sha256-hmac`) | Same. |
| internal | `sha256-hmac` | Internal tooling only. |

See `algorithm_for_tier()` in `middleware/signed_receipts.py`.

Fallback semantics: if Ed25519 is policy for your tier but `GL_FACTORS_ED25519_PRIVATE_KEY` is unset, the middleware emits an HMAC receipt and logs a warning **once** per process. Production deployments should monitor that log line and alert.

---

## Verification procedure (HMAC)

### Python

```python
import hashlib, hmac, json, base64

def verify_hmac_receipt(body: dict, hmac_secret: str) -> bool:
    receipt = body.pop("_signed_receipt")
    canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(expected_hash, receipt["payload_hash"]):
        return False
    expected_sig = base64.b64encode(
        hmac.new(hmac_secret.encode("utf-8"), canonical, hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(expected_sig, receipt["signature"])
```

### TypeScript

```ts
import { createHash, createHmac } from "node:crypto";

export function verifyHmacReceipt(body: any, hmacSecret: string): boolean {
  const receipt = body._signed_receipt;
  const copy = { ...body };
  delete copy._signed_receipt;
  const canonical = Buffer.from(canonicalJson(copy), "utf-8");   // keys sorted
  const expectedHash = createHash("sha256").update(canonical).digest("hex");
  if (expectedHash !== receipt.payload_hash) return false;
  const expectedSig = createHmac("sha256", hmacSecret).update(canonical).digest("base64");
  return expectedSig === receipt.signature;
}
```

`canonicalJson` lives in `greenlang/factors/sdk/ts/src/canonical.ts` and matches Python's key-sorted output.

### Shell (for CI pipelines)

```bash
jq 'del(._signed_receipt)' resolved.json | jq -S -c '.' > payload.json

# Verify payload_hash.
EXPECTED_HASH=$(openssl dgst -sha256 -hex payload.json | awk '{print $2}')
ACTUAL_HASH=$(jq -r '._signed_receipt.payload_hash' resolved.json)
[ "$EXPECTED_HASH" = "$ACTUAL_HASH" ] || exit 1

# Verify HMAC.
SIG=$(jq -r '._signed_receipt.signature' resolved.json)
EXPECTED_SIG=$(openssl dgst -sha256 -hmac "$GL_FACTORS_SIGNING_SECRET" -binary payload.json | base64)
[ "$SIG" = "$EXPECTED_SIG" ]
```

---

## Verification procedure (Ed25519)

GreenLang publishes the current Ed25519 public keys at:

```
GET /api/v1/keys/factors
```

Response:

```json
{
  "keys": [
    {
      "key_id": "gl-factors-ed25519-2026-Q2",
      "algorithm": "ed25519",
      "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n",
      "not_before": "2026-04-01T00:00:00Z",
      "not_after":  "2026-10-01T00:00:00Z"
    },
    {
      "key_id": "gl-factors-ed25519-2026-Q1",
      "algorithm": "ed25519",
      "public_key_pem": "...",
      "not_before": "2026-01-01T00:00:00Z",
      "not_after":  "2026-07-01T00:00:00Z",
      "retired": false
    }
  ]
}
```

Pick the key whose `key_id` matches your receipt's `key_id`, then:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
import base64, hashlib, json

def verify_ed25519_receipt(body: dict, public_key_pem: str) -> bool:
    receipt = body.pop("_signed_receipt")
    canonical = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if expected_hash != receipt["payload_hash"]:
        return False
    pk = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    try:
        pk.verify(base64.b64decode(receipt["signature"]), canonical)
        return True
    except Exception:
        return False
```

---

## Key rotation

Both HMAC and Ed25519 support rotation without code change. The receipt carries `key_id` so clients always verify against the right material.

### HMAC rotation (operator)

1. Generate new secret: `openssl rand -hex 32`.
2. Distribute the new secret to all verifying parties out-of-band (dashboard, email, vault sync).
3. Update `GL_FACTORS_SIGNING_SECRET`. New receipts carry the new `key_id` (e.g. `gl-factors-v2`).
4. Keep the old secret in a secondary verifier slot for 30 days to validate historical receipts.

### Ed25519 rotation (operator)

1. Generate a new keypair in Vault.
2. Publish the new public key via `GET /api/v1/keys/factors`.
3. Flip the signing key id. Old public key stays published until its `not_after` elapses.
4. Clients that cache keys should refresh on an unknown `key_id`.

### Verifier behaviour

Your verifier should:

1. Read `key_id` from the receipt.
2. Fetch the matching key (HMAC shared secret or Ed25519 public key).
3. Verify. If the `key_id` is unknown, fetch `/api/v1/keys/factors` and retry once.
4. If still unknown, reject the receipt.

---

## Edition id binding

Because `edition_id` is canonicalised into the signed payload, the following tamper-detection properties hold:

- Change a number in the body → `payload_hash` mismatch → fail.
- Change `edition_id` in the body → `payload_hash` mismatch → fail.
- Strip `_signed_receipt` entirely → no receipt → fail the "receipt present" check your verifier should run before verifying.
- Replace `X-GreenLang-Edition` header without touching the body → header ignored by the verifier; body is what is signed.

The signed receipt is therefore the **canonical evidence** of which edition produced your number — not the HTTP header.

---

## What does NOT get signed

- 4xx / 5xx responses. The policy is "GreenLang stands behind this answer"; we do not stand behind errors. A 403 or 451 response carries no receipt.
- Streaming / NDJSON / binary responses have a header-only receipt (body mutation would break streaming). The header-only receipt still binds the full response byte-content's hash.
- Pre-auth responses (401 without token).

---

## Operational notes

- Receipts add ~300 bytes per response.
- HMAC signing adds < 0.1ms per response; Ed25519 adds ~1ms.
- On signing failure the response is returned unsigned with an audit log entry; the middleware never crashes the response. See "Signing never crashes the response" invariant in `middleware/signed_receipts.py`.
- Constant-time comparisons are used throughout (`hmac.compare_digest`, Ed25519 library primitives).

---

## See also

- [5-minute quickstart](../quickstart/5-minute-quickstart.md) — step 3 walks through HMAC verification end-to-end.
- [cURL recipes](../quickstart/curl-recipes.md) — recipe 4 ships a shell script you can drop into CI.
- [Version pinning](./version-pinning.md) — edition-id binding relies on the pin mechanism.
- [Webhooks](../api-reference/webhooks.md) — webhook payloads are signed with the same primitives.

---

## File citations

| Piece | File |
|---|---|
| Receipt dataclass + signer functions (`sign_sha256_hmac`, `sign_ed25519`) | `greenlang/factors/signing.py` |
| FastAPI middleware (attaches receipt to every 2xx) | `greenlang/factors/middleware/signed_receipts.py` |
| Algorithm-per-tier table (`_ED25519_TIERS`, `_HMAC_TIERS`) | `greenlang/factors/middleware/signed_receipts.py` (lines 89-90) |
| Canonical JSON hashing (`_canonical_hash`) | `greenlang/factors/signing.py` (line 62) |
| Python SDK `verify_receipt` helper | `greenlang/factors/sdk/python/client.py` |
| TypeScript canonical JSON + hash | `greenlang/factors/sdk/ts/src/canonical.ts`, `hash.ts` |
