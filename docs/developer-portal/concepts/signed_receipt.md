# Concept — Signed Receipt

Every `/resolve` and `/explain` response includes a **signed receipt**: a small envelope that lets any third party verify, offline, that GreenLang actually produced the response and that the payload has not been tampered with since. Signed receipts are the mechanism by which auditors, regulators, and downstream assurance providers gain cryptographic evidence of a resolution.

## The envelope

```json
"signed_receipt": {
  "receipt_id": "rcpt_2026Q2_01J...",
  "signature": "base64url(Ed25519-signature)",
  "verification_key_hint": "jwk-2026Q2-primary",
  "alg": "Ed25519",
  "payload_hash": "sha256:3f7bc0e8f4d1e2a9c5b8f6d4e2a1c9b7d5e3a2c1b0f9e8d7c6b5a4c3d2e1f0a9",
  "issued_at": "2026-04-23T14:30:00Z"
}
```

Fields:

| Field | Purpose |
|---|---|
| `receipt_id` | ULID unique to this receipt. Used for audit lookup. |
| `signature` | Base64url signature over `payload_hash`. |
| `verification_key_hint` | Key ID. Resolves via `https://api.greenlang.io/.well-known/jwks.json` (Ed25519) or the tenant's configured HMAC secret (HS256). |
| `alg` | `Ed25519` (default) or `HS256` (private deployments). |
| `payload_hash` | SHA-256 over the canonical JCS-normalised response body minus the `signed_receipt` block itself. |
| `issued_at` | RFC 3339 timestamp. Does not affect signature validity. |

## Verification procedure (offline)

1. Parse the JSON response.
2. Extract `signed_receipt`; remove the block from the body.
3. Canonicalise the remaining body using JSON Canonicalization Scheme (RFC 8785).
4. Hash with SHA-256; compare to `payload_hash`. If mismatch → **REJECT**.
5. Resolve `verification_key_hint` to a public key (JWKS for Ed25519) or a shared HMAC secret (HS256).
6. Verify `signature` over `payload_hash`. If invalid → **REJECT**.
7. Record `receipt_id` and `verification_key_hint` in your audit trail.

The SDKs do steps 2-6 for you:

```python
from greenlang_factors.verify import verify_receipt
verified = verify_receipt(response_json)
assert verified.valid
```

```ts
import { verifyReceipt } from "@greenlang/factors";
const verified = await verifyReceipt(responseJson);
```

```bash
gl-factors verify-receipt ./response.json --key /secrets/hmac.key
```

## Key management

- **Ed25519** — GreenLang publishes a JWKS at `https://api.greenlang.io/.well-known/jwks.json`. Keys rotate quarterly; old keys remain in the JWKS for 18 months so historical receipts verify.
- **HS256** — private deployments configure a shared secret; clients read from a file via `--key` or from `GL_FACTORS_HMAC_KEY` env var.

## What a receipt proves

- The server ran the resolution at `issued_at`.
- The payload returned was exactly the one hashed.
- The signer holds the private key that matches `verification_key_hint`.

## What a receipt does NOT prove

- The underlying factor values are correct (that is what sources and FQS are for).
- The caller was entitled to the factors returned (that is what the tenant audit log proves).
- The receipt is the latest — receipts do not supersede each other; pair with `edition_id` for reproducibility.

## Audit bundles

An audit bundle (`POST /v1/audit/bundle`) is a batch of receipts plus their resolved responses, co-signed by a customer-owned key. This is the artefact third-party assurance providers consume. See [`api-reference/explain.md`](../api-reference/explain.md) and [legal source binder](../../launch/legal_source_rights_binder.md).

**See also:** [`edition`](edition.md), [`factor`](factor.md), [API authentication](../authentication.md).
