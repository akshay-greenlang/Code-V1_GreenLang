# Errors

Every error response shares the same envelope:

```json
{
  "error": "<machine_code>",
  "message": "<human sentence>",
  "details": { "...": "context the client can act on" }
}
```

`error` is stable and safe to switch on. `message` is human-readable but may change. `details` carries actionable context (the conflicting field, the required tier, the missing license, etc.) whose keys are stable per error code.

**Error envelope schema:** `greenlang/integration/api/models.py::ErrorResponse`
**Error catalog (see also):** `docs/api/error-codes.md`

---

## Full table

### 4xx — client errors

| HTTP | `error` | Meaning | Fixable by |
|---|---|---|---|
| 400 | `validation_error` | Body or query parameter failed schema validation. | Read `details.field`; send a valid value. |
| 400 | `invalid_edition` | `X-Factors-Edition` / `?edition=` references an unknown or retired edition. | Call `GET /api/v1/editions?status=stable` and pick a valid id. |
| 400 | `invalid_method_profile` | `method_profile` is not one of the registered profiles. | Use one of `corporate_scope1`, `corporate_scope2_location_based`, `corporate_scope2_market_based`, `corporate_scope3`, `product_carbon`, `freight_iso_14083`, `land_removals`, `finance_proxy`, `eu_cbam`, `eu_dpp`. |
| 400 | `incompatible_units` | Activity unit cannot be converted to the factor's native unit. | Fix the activity unit or supply `extras.unit_graph_overrides`. |
| 400 | `missing_required_field` | The method pack requires a field that you omitted (e.g. PCAF needs `asset_class`). | Check the method-pack doc for required fields. |
| 401 | `invalid_credentials` | Missing / malformed / expired credential. | Re-auth. See [authentication](./authentication.md). |
| 401 | `token_expired` | JWT `exp` is past. | Refresh the token. |
| 402 | `billing_limit_exceeded` | Authenticated but plan has no remaining capacity (seats, batch slots, etc.). | Upgrade or wait for the period to reset. |
| 403 | `tier_insufficient` | Endpoint requires a higher tier. | Upgrade at `https://dashboard.greenlang.io/plans`. |
| 403 | `scope_denied` | API key missing a required scope. | Re-issue with broader scope. |
| 403 | `tenant_mismatch` | Resource belongs to another tenant. | Use the correct credential. |
| 403 | `license_class_forbidden` | Factor's `license_class` is above your tier's ceiling. | Upgrade or use a different source. |
| 404 | `factor_not_found` | `factor_id` does not exist in the resolved edition. | Check the id and edition pin. |
| 404 | `edition_not_found` | Edition id does not exist. | List editions. |
| 404 | `subscription_not_found` | Webhook subscription id unknown. | List subscriptions. |
| 409 | `edition_promotion_conflict` | Rollback target is already the active default. | No action needed. |
| 409 | `rollback_in_progress` | Another rollback is running. | Wait or check `/rollback/history`. |
| 422 | `resolution_failed` | The 7-step cascade produced no eligible candidate. | Broaden activity, relax `preferred_sources`, or supply a supplier / facility hint. See [resolution-cascade](../concepts/resolution-cascade.md). |
| 422 | `license_class_mixed` | Homogeneity rule would force mixing `open` and `restricted` factors. | Supply `preferred_sources` to pin a single licence family. |
| 422 | `method_profile_gate_failed` | Candidate violates the method pack's `SelectionRule` (e.g. CBAM rejected a Preview factor). | Upgrade factor quality or relax the profile's strictness (most callers: switch profile). |
| 429 | `rate_limit_exceeded` | RPM or export budget exceeded. | Honour `Retry-After`. See [rate-limits](./rate-limits.md). |
| 451 | `license_required` | Routed to a licensed connector with no tenant key. | Configure the license key, or drop the `preferred_sources` hint. See [licensing-classes](../concepts/licensing-classes.md). |

### 5xx — server errors

| HTTP | `error` | Meaning | Fixable by |
|---|---|---|---|
| 500 | `internal_server_error` | Unexpected failure. | Retry with backoff. Report `request_id` in `details`. |
| 502 | `connector_unavailable` | Upstream licensed source unavailable (e.g. ecoinvent API timeout). | Retry. Check `status.greenlang.io`. |
| 503 | `service_unavailable` | Planned maintenance or graceful shutdown. | Honour `Retry-After`. |
| 504 | `upstream_timeout` | Upstream exceeded the GreenLang timeout budget. | Retry. If persistent, simplify the request. |

---

## Worked examples

### 400 `validation_error`

```json
{
  "error": "validation_error",
  "message": "reporting_date must be a valid ISO-8601 date.",
  "details": {
    "field": "reporting_date",
    "value": "06/01/2026",
    "resolution": "Use YYYY-MM-DD (e.g. 2026-06-01)."
  }
}
```

### 403 `tier_insufficient`

```json
{
  "error": "tier_insufficient",
  "message": "Factor explain endpoints require Pro, Consulting, Enterprise, or Internal tier.",
  "details": {
    "current_tier": "community",
    "required_tiers": ["pro", "consulting", "enterprise", "internal"],
    "resolution": "Upgrade at https://dashboard.greenlang.io/plans."
  }
}
```

### 422 `resolution_failed`

```json
{
  "error": "resolution_failed",
  "message": "No factor could be resolved for activity='obscure_process_gas' jurisdiction='XX' method_profile='corporate_scope1'.",
  "details": {
    "method_profile": "corporate_scope1",
    "steps_tried": 7,
    "candidates_considered": 12,
    "rejections_by_step": {
      "5": {"count": 4, "reason": "geography mismatch"},
      "6": {"count": 8, "reason": "allowed_statuses exclusion"}
    },
    "resolution": "Broaden the activity term, supply a supplier_id, or relax preferred_sources."
  }
}
```

### 429 `rate_limit_exceeded`

See [rate-limits](./rate-limits.md).

### 451 `license_required`

```json
{
  "error": "license_required",
  "message": "Resolution routed to licensed source 'ecoinvent' but no active license key is configured for tenant 'acme-climate'.",
  "details": {
    "connector_id": "ecoinvent",
    "license_class": "commercial_connector",
    "resolution": "Configure GL_FACTORS_LICENSE_ECOINVENT env var, or contact sales@greenlang.io for a connector key.",
    "alternate_resolution": "Re-run without preferred_sources to fall back to open-licensed alternatives."
  }
}
```

---

## Handling patterns

### Retry matrix

| HTTP | Retry? | How |
|---|---|---|
| 400, 401, 403, 404, 409, 422 | No | Fix the request. |
| 402 | No | Upgrade or wait for period reset. |
| 429 | Yes | Honour `Retry-After`. |
| 451 | No | Configure license or switch source. |
| 500, 502, 503, 504 | Yes | Exponential backoff with jitter, max 5 retries. |

### Exception types (SDK)

Python — `greenlang/factors/sdk/python/errors.py`:

- `FactorsAPIError` — base; carries `status_code`, `error_code`, `message`, `details`, `request_id`.
- `RateLimitError(FactorsAPIError)` — subclass; carries `retry_after` (seconds, pre-parsed).
- `ValidationError` — raised before a network call if the SDK detects a bad request shape.
- `AuthenticationError` — 401.
- `LicenseRequiredError` — 451; carries `connector_id`.

TypeScript — `greenlang/factors/sdk/ts/src/errors.ts`:

- `FactorsAPIError` — base.
- Shapes mirror Python. Use `err.statusCode === 429` etc.

### Idempotency

`POST /factors/resolve-explain` is idempotent — the same request payload produces the same response under the same edition pin. So retrying on a transient 5xx is always safe.

Batch submissions accept an optional `Idempotency-Key` header (UUID). The server caches completed responses under that key for 24h, so a retry after a network partition will not duplicate the job.

---

## Request id

Every response — including errors — carries `X-Request-Id`. Include this when filing a support ticket. It is also in `details.request_id` on the error body.

```http
HTTP/1.1 500 Internal Server Error
X-Request-Id: 018f5c7e-8c2a-7d05-a9e3-1d0a2b77cc10

{
  "error": "internal_server_error",
  "message": "Unexpected failure.",
  "details": {"request_id": "018f5c7e-8c2a-7d05-a9e3-1d0a2b77cc10"}
}
```

---

## See also

- [Authentication](./authentication.md)
- [Rate limits](./rate-limits.md)
- [Licensing classes](../concepts/licensing-classes.md)
- [Resolution cascade](../concepts/resolution-cascade.md) — 422 `resolution_failed` context.

---

## File citations

| Piece | File |
|---|---|
| `ErrorResponse` model | `greenlang/integration/api/models.py` |
| Route error emitters | `greenlang/integration/api/routes/factors.py` |
| `ResolutionError` (422 path) | `greenlang/factors/resolution/engine.py` |
| License-required (451) | `greenlang/factors/connectors/license_manager.py` |
| Tier gate (403) | `greenlang/factors/tier_enforcement.py` |
| Rate limiter (429) | `greenlang/factors/middleware/rate_limiter.py` |
| Authoritative error catalog | `docs/api/error-codes.md` |
