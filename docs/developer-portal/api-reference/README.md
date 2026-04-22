# API Reference

**Base URL:** `https://api.greenlang.io`
**Prefix:** `/api/v1`
**OpenAPI spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml)
**Authentication:** JWT Bearer token or API key (see [authentication](./authentication.md)).
**Rate limits:** Per-tier sliding window (see [rate-limits](./rate-limits.md)).

---

## How to read this reference

The full machine-readable spec lives at `docs/api/factors-v1.yaml`. Point any OpenAPI tooling (Swagger UI, Redoc, Postman, Stoplight, Insomnia) at it to get a browsable, request-building UI.

The human-language pages in this folder cover the cross-cutting concerns that the OpenAPI spec alone cannot carry well:

- [Authentication](./authentication.md) — API keys, JWTs, tier detection, 401 / 402 / 403 / 451 semantics.
- [Rate limits](./rate-limits.md) — per-tier limits, headers, 429 behaviour, bursts, export budgets.
- [Errors](./errors.md) — full error-code table with remediation.
- [Webhooks](./webhooks.md) — all 11 event types, signature verification, delivery retries, idempotency.

---

## The endpoints

Grouped by subsystem. Every route lives under `/api/v1`.

### Factors — discovery

| Method | Path | Summary |
|---|---|---|
| `GET` | `/factors` | List factors with pagination (filters: `fuel_type`, `geography`, `scope`, `boundary`). |
| `GET` | `/factors/search` | Simple text search. |
| `POST` | `/factors/search/v2` | Advanced search with sort + pagination. |
| `GET` | `/factors/search/facets` | Facet counts for filter UIs. |
| `POST` | `/factors/match` | Activity-to-factor matching. |
| `GET` | `/factors/coverage` | Coverage statistics per geography / sector. |
| `GET` | `/factors/status/summary` | Three-label dashboard (certified / preview / connector_only). |
| `GET` | `/factors/{factor_id}` | Get a single factor. |

### Factors — resolution (the 7-step cascade)

| Method | Path | Summary | Tier |
|---|---|---|---|
| `POST` | `/factors/resolve-explain` | Full cascade from a `ResolutionRequest`. | Pro+ |
| `GET` | `/factors/{factor_id}/explain` | Explain a specific factor under a default context. | Pro+ |
| `GET` | `/factors/{factor_id}/alternates` | Top-N alternate candidates the cascade would also consider. | Pro+ |
| `GET` | `/factors/{factor_id}/quality` | Composite FQS 0-100 + 5-component DQS + rating + promotion eligibility. | Any |

See [resolution-cascade](../concepts/resolution-cascade.md).

### Factors — audit + provenance

| Method | Path | Summary | Tier |
|---|---|---|---|
| `GET` | `/factors/{factor_id}/audit-bundle` | Full provenance, DQS breakdown, GWP table, checksums. | Enterprise |
| `GET` | `/factors/{factor_id}/diff?from=X&to=Y` | Field-by-field diff across two editions. | Pro+ |
| `GET` | `/factors/{factor_id}/dependent-computations` | List computations that cite this factor. | Pro+ |

### Factors — rollback

| Method | Path | Summary | Tier |
|---|---|---|---|
| `POST` | `/factors/{factor_id}/rollback/plan` | Dry-run impact preview. | Pro+ |
| `POST` | `/factors/{factor_id}/rollback/execute` | Commit the rollback (requires signed approval). | Pro+ |
| `GET` | `/factors/{factor_id}/rollback/history` | Audit log. | Pro+ |
| `GET` | `/factors/rollback/{rollback_id}` | Single rollback record. | Pro+ |

See [version-pinning](../concepts/version-pinning.md).

### Factors — batch

| Method | Path | Summary | Tier |
|---|---|---|---|
| `POST` | `/factors/batch/submit` | Submit a batch resolution job. | Pro+ |
| `GET` | `/factors/batch` | List batch jobs. | Pro+ |
| `GET` | `/factors/batch/{job_id}` | Job status + progress. | Pro+ |
| `GET` | `/factors/batch/{job_id}/results` | Paginated results. | Pro+ |
| `POST` | `/factors/batch/{job_id}/cancel` | Cancel a queued or running job. | Pro+ |
| `DELETE` | `/factors/batch/{job_id}` | Delete a completed job's storage. | Pro+ |

### Factors — export

| Method | Path | Summary | Tier |
|---|---|---|---|
| `GET` | `/factors/export` | Bulk export JSON Lines. | Pro+ (subject to per-15min export budget) |

### Impact simulation

| Method | Path | Summary | Tier |
|---|---|---|---|
| `POST` | `/factors/impact-simulation` | Preview one factor change. | Pro+ |
| `POST` | `/factors/impact-simulation/batch` | Preview many factor changes. | Pro+ |

### Editions

| Method | Path | Summary |
|---|---|---|
| `GET` | `/editions` | List published editions (filter by `status`). |
| `GET` | `/editions/{edition_id}` | Get a single edition manifest. |
| `GET` | `/editions/{edition_id}/changelog` | Human-readable changelog. |
| `GET` | `/editions/watch/status` | Per-source watch-pipeline status. |

### Webhooks

| Method | Path | Summary |
|---|---|---|
| `POST` | `/webhooks/subscriptions` | Create a subscription. |
| `GET` | `/webhooks/subscriptions` | List subscriptions. |
| `DELETE` | `/webhooks/subscriptions/{sub_id}` | Delete a subscription. |
| `POST` | `/webhooks/subscriptions/{sub_id}/test` | Emit a test delivery. |

See [webhooks](./webhooks.md).

### Keys

| Method | Path | Summary |
|---|---|---|
| `GET` | `/keys/factors` | Published HMAC key ids + Ed25519 public keys for receipt verification. |

See [signed-receipts](../concepts/signed-receipts.md).

---

## Common request headers

| Header | Set by | Purpose |
|---|---|---|
| `Authorization: Bearer <jwt>` | client | JWT auth (see [authentication](./authentication.md)). |
| `X-API-Key: <key>` | client | API key auth. |
| `X-Factors-Edition: <edition_id>` | client | Pin to a specific edition (see [version-pinning](../concepts/version-pinning.md)). |
| `Content-Type: application/json` | client | POST bodies must be JSON. |
| `If-None-Match: "<etag>"` | client | Conditional GET for explain endpoints. |

## Common response headers

| Header | Purpose |
|---|---|
| `X-GreenLang-Edition` | Edition id the response was produced from. |
| `X-Factors-Edition` | Legacy alias for `X-GreenLang-Edition`. |
| `X-GreenLang-Method-Profile` | Method profile used for explain / resolve calls. |
| `X-RateLimit-Limit` | Tier RPM cap. |
| `X-RateLimit-Remaining` | Requests left in the current window. |
| `X-RateLimit-Reset` | UTC epoch when the window resets. |
| `Retry-After` | Seconds to wait before retrying (on 429). |
| `ETag` | For cache-friendly repeat reads. |
| `X-GreenLang-Receipt-*` | Signed receipt headers on non-JSON responses. See [signed-receipts](../concepts/signed-receipts.md). |

---

## Language-specific quickstarts

- [Python SDK](../quickstart/python-sdk.md)
- [TypeScript SDK](../quickstart/typescript-sdk.md)
- [cURL recipes](../quickstart/curl-recipes.md)

---

## File citations

| Piece | File |
|---|---|
| Route definitions | `greenlang/integration/api/routes/factors.py` |
| Editions routes | `greenlang/integration/api/routes/editions.py` |
| Health + marketplace + dashboards | `greenlang/integration/api/routes/` |
| OpenAPI spec | `docs/api/factors-v1.yaml` |
| Models | `greenlang/integration/api/models.py` |
