# API — `POST /v1/factors/search` and `GET /v1/factors/{factor_id}`

Query the catalog. Search is scoped to a single edition; lookup returns a single factor record in canonical v1 shape.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: searchFactors`, `getFactor`).

---

## Search — `POST /v1/factors/search`

Use search when you need to browse the catalog. For single-activity resolution, use [`/resolve`](resolve.md) — it runs the cascade and returns a full envelope.

### Request

```json
{
  "filters": {
    "factor_family": ["electricity"],
    "jurisdiction.country": ["IN", "US"],
    "status": ["active"],
    "method_profile": ["corporate_scope2_location_based"],
    "source_id": ["india_cea_co2_baseline", "egrid"]
  },
  "sort": [
    { "field": "quality.composite_fqs", "order": "desc" },
    { "field": "valid_from", "order": "desc" }
  ],
  "offset": 0,
  "limit": 50
}
```

Supported filter fields: every top-level path in the [canonical schema](../schema.md). Sort fields: `quality.composite_fqs`, `valid_from`, `valid_to`, `factor_version`, `source_version`.

### curl

```bash
curl -X POST "https://api.greenlang.io/v1/factors/search" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-GreenLang-Edition: builtin-v1.0.0" \
  -d '{
    "filters": { "factor_family": ["electricity"], "jurisdiction.country": ["IN"] },
    "sort": [{ "field": "quality.composite_fqs", "order": "desc" }],
    "offset": 0,
    "limit": 20
  }'
```

### Python

```python
page = client.search_factors(
    filters={"factor_family": ["electricity"], "jurisdiction.country": ["IN"]},
    sort=[{"field": "quality.composite_fqs", "order": "desc"}],
    limit=20,
)
for f in page.items:
    print(f.factor_id, f.quality.composite_fqs_0_100)
```

### TypeScript

```ts
const page = await client.searchFactors({
  filters: { factorFamily: ["electricity"], "jurisdiction.country": ["IN"] },
  sort: [{ field: "quality.composite_fqs", order: "desc" }],
  limit: 20,
});
```

### Response

```json
{
  "edition_id": "builtin-v1.0.0",
  "total": 42,
  "offset": 0,
  "limit": 20,
  "items": [
    { "factor_id": "EF:IN:grid:CEA:FY2024-25:v1", "factor_version": "1.0.0", "...": "..." }
  ]
}
```

Pagination is offset-based. Server caps `limit` at 500. For bulk export, use [`/bulk-export`](releases.md#bulk-export) (open-class factors only).

---

## Lookup — `GET /v1/factors/{factor_id}`

Fetch a single record by ID.

```bash
FID=$(python -c "import urllib.parse; print(urllib.parse.quote('EF:IN:grid:CEA:FY2024-25:v1', safe=''))")

curl "https://api.greenlang.io/v1/factors/$FID" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Response

Full canonical record per [`schema.md`](../schema.md). Includes `numerator`, `denominator`, `parameters`, `quality`, `lineage`, `licensing`, `explainability`.

### Query parameters

| Param | Default | Notes |
|---|---|---|
| `version` | latest | Specific `factor_version` (semver). |
| `edition` | header or latest | `edition_id` to scope lookup. |

---

## Diff — `GET /v1/factors/{factor_id}/diff`

Structured diff between two `factor_version`s of the same `factor_id`. Used by the Operator Console's diff viewer.

```bash
curl "https://api.greenlang.io/v1/factors/$FID/diff?from=1.0.0&to=1.1.0" \
  -H "Authorization: Bearer $GL_API_KEY"
```

Response highlights changes in gas vectors, parameters, quality scores, and lineage.

---

## Errors

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | Missing token. |
| 403 | `forbidden` | Tenant lacks entitlement for a filtered `source_id`. |
| 404 | `not_found` | Factor ID / version unknown in the requested edition. |
| 409 | `edition_mismatch` | Pinned edition not servable. |
| 429 | `rate_limited` | See [`error-codes.md`](../error-codes.md). |

## Related

- [`/resolve`](resolve.md), [`/explain`](explain.md).
- [`/sources`](sources.md), [`/method-packs`](method-packs.md), [`/editions`](releases.md).
