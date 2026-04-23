# API — `POST /v1/factors/batch-resolve`

Resolve up to 5,000 activities in a single call. Returns an array of the same envelope shape as [`/resolve`](resolve.md), plus per-item error entries for rows that fail to resolve.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: batchResolve`).

---

## Endpoint

```
POST https://api.greenlang.io/v1/factors/batch-resolve
```

Batch resolution is the right tool for:
- Bulk inventory imports (ERP activity exports).
- CI validation of a configuration change against 100+ test activities.
- Large scenario simulations.

For streaming workloads above 5,000 items, chunk the input and pipeline the calls.

---

## Request body

```json
{
  "items": [
    {
      "row_id": "client_row_001",
      "factor_family": "electricity",
      "quantity": 12500,
      "unit": "kWh",
      "method_profile": "corporate_scope2_location_based",
      "jurisdiction": "IN",
      "valid_at": "2026-12-31"
    },
    {
      "row_id": "client_row_002",
      "factor_family": "combustion",
      "quantity": 500,
      "unit": "gal",
      "method_profile": "corporate_scope1",
      "jurisdiction": "US-TX",
      "valid_at": "2026-12-31",
      "activity_code": "NAICS:221112"
    }
  ],
  "on_error": "collect"
}
```

| Field | Required | Notes |
|---|---|---|
| `items[]` | yes | Up to 5,000 items. Each item has the same shape as a `/resolve` request plus a caller-supplied `row_id`. |
| `on_error` | no | `collect` (default) returns per-item errors in `errors[]`. `fail_fast` aborts on first error. |

---

## curl

```bash
curl -X POST "https://api.greenlang.io/v1/factors/batch-resolve" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  --data @batch.json
```

## Python

```python
batch = client.batch_resolve(items=[
    {"row_id": "r1", "factor_family": "electricity", "quantity": 12500,
     "unit": "kWh", "method_profile": "corporate_scope2_location_based",
     "jurisdiction": "IN", "valid_at": "2026-12-31"},
    {"row_id": "r2", "factor_family": "combustion", "quantity": 500,
     "unit": "gal", "method_profile": "corporate_scope1",
     "jurisdiction": "US-TX", "valid_at": "2026-12-31"},
])

for r in batch.results:
    print(r.row_id, r.emissions.co2e_kg)
for e in batch.errors:
    print(e.row_id, e.error_code)
```

## TypeScript

```ts
const batch = await client.batchResolve({
  items: [
    { rowId: "r1", factorFamily: "electricity", quantity: 12500, unit: "kWh",
      methodProfile: "corporate_scope2_location_based", jurisdiction: "IN",
      validAt: "2026-12-31" },
    { rowId: "r2", factorFamily: "combustion", quantity: 500, unit: "gal",
      methodProfile: "corporate_scope1", jurisdiction: "US-TX",
      validAt: "2026-12-31" },
  ],
});
```

---

## Response (200 OK)

```json
{
  "edition_id": "builtin-v1.0.0",
  "results": [
    { "row_id": "client_row_001", "chosen_factor": { "...": "..." }, "emissions": { "co2e_kg": 9950.0 } },
    { "row_id": "client_row_002", "chosen_factor": { "...": "..." }, "emissions": { "co2e_kg": 5107.0 } }
  ],
  "errors": [],
  "batch_receipt": {
    "receipt_id": "brcpt_2026Q2_...",
    "signature": "...",
    "payload_hash": "sha256:...",
    "alg": "Ed25519"
  }
}
```

Each `results[i]` contains the same envelope as a single `/resolve` response EXCEPT `signed_receipt` — the batch carries one aggregate `batch_receipt` covering the whole payload. Use the batch receipt in audit bundles.

When `on_error: collect`, items that could not resolve appear in `errors[]`:

```json
{
  "errors": [
    {
      "row_id": "client_row_017",
      "error_code": "factor_cannot_resolve_safely",
      "message": "No candidate satisfies pack rules for method_profile=eu_cbam, jurisdiction=XX",
      "details": { "pack_id": "eu_cbam", "evaluated_candidates_count": 0 }
    }
  ]
}
```

---

## Rate limits and batching

- A single batch counts as ONE request for rate limiting but `len(items)` for billing.
- Recommended batch size: 500-2,000 items. Above 2,000, response latency grows; below 500, throughput drops.
- Items are processed in parallel server-side; order is preserved in `results[]`.

---

## Errors (top-level)

| Status | Code | When |
|---|---|---|
| 400 | `bad_request` | Invalid item shape, missing required fields, `items[]` empty. |
| 413 | `batch_too_large` | `items.length > 5000`. |
| 422 | `fail_fast_abort` | Used `on_error: fail_fast` and a row failed. |
| 429 | `rate_limited` | See [`error-codes.md`](../error-codes.md). |

Per-row errors do NOT count toward the top-level status; they appear in `errors[]`.

## Related

- [`/resolve`](resolve.md), [`/explain`](explain.md).
- [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).
