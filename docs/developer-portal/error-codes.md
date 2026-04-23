# Error codes

Every API error carries an HTTP status, a stable `error_code`, a human explanation, and an optional `details` bag. SDKs map each `error_code` to a typed exception.

Response shape:

```json
{
  "error_code": "factor_cannot_resolve_safely",
  "message": "No candidate satisfies pack rules for method_profile=eu_cbam, jurisdiction=XX",
  "http_status": 422,
  "details": {
    "pack_id": "eu_cbam",
    "method_profile": "eu_cbam",
    "evaluated_candidates_count": 0
  },
  "request_id": "req_01J..."
}
```

---

## Resolution errors

### `factor_cannot_resolve_safely` (422)

**When:** The resolver evaluated every tier in the cascade and no candidate satisfied both the pack's `SelectionRule` and `BoundaryRule`. The pack has `fallback.cannot_resolve_action = raise_no_safe_match` (required for Certified packs). This is CTO non-negotiable #3 made explicit — the resolver does not silently fall back to a weak global default.

**Python:** `FactorCannotResolveSafelyError`
**TypeScript:** `FactorCannotResolveSafelyError`
**Details:** `pack_id`, `method_profile`, `evaluated_candidates_count`, `tier_trace[]`.

**Remediation:**

1. Supply a narrower jurisdiction (e.g., `US-CA` instead of `XX`).
2. Supply `supplier_id` / `facility_id` / `utility_or_grid_region` for higher-tier matches.
3. Verify the activity is in-scope for the method pack (e.g., `eu_cbam` covers only Annex I goods).
4. If exploratory use is acceptable, set `include_preview: true` to surface preview-status factors.

---

## Licensing / entitlement errors

### `payment_required` (402)

**When:** The caller's subscription does not include the SKU that covers the requested `method_profile` or `source_id`.

**Python:** `PaymentRequiredError`
**Details:** `required_sku`, `upgrade_url`.

**Remediation:** Upgrade the tenant's SKU at `https://app.greenlang.io/billing`.

### `entitlement_gap` (403)

**When:** The caller holds a paid SKU but the specific source the resolver chose is not entitled under that SKU (e.g., holds Freight SKU but needs Finance SKU for PCAF).

**Python:** `EntitlementError`
**Details:** `source_id`, `required_entitlement`.

**Remediation:** Verify the right SKU is active, or re-issue with a broader method pack.

### `licensing_gap` (403 or 451)

**When:** The record's `redistribution_class` is incompatible with the caller's context — e.g., a bulk-export request hit a `licensed_embedded` factor. Returned as 451 (Unavailable For Legal Reasons) when a legal boundary is crossed.

**Python:** `LicensingGapError`
**Details:** `redistribution_class`, `blocked_reason`.

**Remediation:** Use the API query surface instead of bulk export, or add BYO-credentials for the source (see [`licensing.md`](licensing.md)).

### `legal_blocked` (451)

**When:** The combination of caller context, jurisdiction, and record licensing would violate a redistribution boundary. Used for cross-border export restrictions and sanction-list protections.

**Remediation:** Contact `legal@greenlang.io` — this is not a client-side fix.

---

## Edition errors

### `edition_mismatch` (409)

**When:** Caller pinned an `edition_id` the server cannot serve (retired / unknown / not synced to this cluster).

**Python:** `EditionMismatchError`
**Details:** `requested_edition`, `available_editions[]`.

**Remediation:** Pick an edition from `available_editions[]`, or unpin to receive the current Certified edition. See [`concepts/edition.md`](concepts/edition.md).

### `edition_pin_error` (400)

**When:** The caller's pinned `edition_id` is malformed (e.g., bad semver, wrong channel prefix).

**Python:** `EditionPinError`

**Remediation:** Use the form `<channel>-v<semver>`, e.g., `builtin-v1.0.0`, `preview-v1.1.0-rc.1`.

### `edition_retired` (410)

**When:** Requested edition was retired from active serving; retrievable only via the audit reconstruction endpoint.

**Remediation:** Use the Archive API or contact `support@greenlang.io` for retired-edition retrieval.

---

## Validation errors

### `bad_request` (400)

**When:** Request body fails JSON-Schema validation, missing required fields, wrong types, out-of-range values.

**Python:** `BadRequestError`
**Details:** `field_path`, `constraint`, `actual_value`.

**Remediation:** Fix the request per the field path. See [`api-reference/resolve.md`](api-reference/resolve.md).

### `activity_ambiguous` (422)

**When:** `activity_text` matched multiple factors with ties in the semantic index. The resolver refuses to guess.

**Details:** `candidate_factor_ids[]`.

**Remediation:** Provide `factor_family` + `activity_code` to disambiguate, or pick a candidate from the `candidates[]` array.

### `unit_incompatible` (422)

**When:** The caller's `unit` cannot be converted to the factor's denominator unit via the ontology (e.g., `kWh` → `USD`).

**Details:** `from_unit`, `to_unit`.

**Remediation:** Convert upstream, or use a different factor family.

---

## Auth errors

### `unauthorized` (401)

**When:** Missing / invalid / revoked API key or OAuth2 token.

**Python:** `UnauthorizedError`

**Remediation:** Check `Authorization` header format (`Bearer <token>`). Rotate stale keys at the dashboard.

### `forbidden` (403)

**When:** Authenticated but lacks the scope for the operation (e.g., read-only key attempted a mutation).

**Remediation:** Mint a new key with the required scope.

---

## Rate limiting

### `rate_limited` (429)

**When:** Exceeded 100 req/min per key or 1000 req/min per tenant.

**Python:** `RateLimitError`
**Details:** `retry_after` (seconds), `limit`, `remaining`.
**Response header:** `Retry-After: <seconds>`.

**Remediation:** Honour `Retry-After`. SDK transports retry once automatically; for caller-side backoff, catch the exception and wait `retry_after` seconds.

---

## Webhook errors

### `webhook_duplicate` (409)

Same URL + events combination already subscribed for the tenant.

### `webhook_url_unreachable` (422)

GreenLang could not open a TLS connection during validation. Webhook URLs must be HTTPS and publicly reachable.

---

## Infrastructure errors

### `server_error` (500)

Unexpected server error. `request_id` in the body — include it in support tickets.

### `service_unavailable` (503)

Planned maintenance or transient cluster health issue. See `https://status.greenlang.io` and retry with backoff.

### `bad_gateway` (502)

Upstream connector (BYO-source) unreachable. Usually means the customer's ecoinvent / IEA / Electricity Maps credential expired.

**Remediation:** Verify the BYO credential at `gl-factors connector test --source <id>`.

---

## SDK exception hierarchy

Python and TypeScript SDKs share the same shape:

```
FactorsError
 ├── BadRequestError
 ├── UnauthorizedError
 ├── PaymentRequiredError
 ├── EntitlementError
 ├── LicensingGapError
 ├── NotFoundError
 ├── EditionMismatchError
 ├── EditionPinError
 ├── EditionRetiredError
 ├── FactorCannotResolveSafelyError
 ├── ActivityAmbiguousError
 ├── UnitIncompatibleError
 ├── RateLimitError
 ├── WebhookDuplicateError
 ├── WebhookUnreachableError
 ├── ServerError
 └── ServiceUnavailableError
```

See [`sdks/python.md`](sdks/python.md#errors) and [`sdks/typescript.md`](sdks/typescript.md#errors) for import paths.

---

## Related

- [`sdks/python.md`](sdks/python.md), [`sdks/typescript.md`](sdks/typescript.md).
- [`api-reference/resolve.md`](api-reference/resolve.md), [`concepts/edition.md`](concepts/edition.md), [`licensing.md`](licensing.md).
