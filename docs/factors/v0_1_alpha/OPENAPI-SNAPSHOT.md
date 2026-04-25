# GreenLang Factors v0.1 Alpha — OpenAPI 3.1 Snapshot

> **Status:** FROZEN for the v0.1 Alpha release (CTO doc §19.1).
> **Generated from:** `greenlang.factors.factors_app:create_factors_app()`
> with `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`.
> **Snapshot fixture:** [`tests/factors/v0_1_alpha/openapi_alpha_v0_1.json`](../../../tests/factors/v0_1_alpha/openapi_alpha_v0_1.json)
> **Drift detector:** `tests/factors/v0_1_alpha/test_alpha_api_contract.py::test_openapi_alpha_v0_1_matches_snapshot`
> **Regenerate:** `UPDATE_OPENAPI_SNAPSHOT=1 pytest tests/factors/v0_1_alpha/test_alpha_api_contract.py`

The v0.1 Alpha public surface is **exactly five** read-only `GET` endpoints
plus a `/api/v1/{path:path}` 410-Gone catch-all (the catch-all is hidden
from OpenAPI via `include_in_schema=False`). Every other surface
(resolve / explain / batch / coverage / fqs / editions / admin / billing /
oem / graphql / method-packs) is gated by `release_profile` and returns
404 under alpha.

---

## Endpoint Summary

| # | Method | Path | Response model | Auth | Notes |
|---|--------|------|----------------|------|-------|
| 1 | `GET`  | `/v1/healthz`           | `HealthzResponse`     | unauthenticated | Liveness + edition id + release profile + schema id + git commit. |
| 2 | `GET`  | `/v1/factors`           | `FactorListResponse`  | API key / JWT   | Cursor-paginated list, filtered by `geography_urn` / `source_urn` / `pack_urn` / `category` / `vintage_*`. 400 on invalid `category`. |
| 3 | `GET`  | `/v1/factors/{urn}`     | `FactorV0_1`          | API key / JWT   | URL-encoded canonical URN (`urn:gl:factor:<source>:<ns>:<id>:v<n>`). 404 with typed `ErrorResponse` body if unknown. |
| 4 | `GET`  | `/v1/sources`           | `SourceListResponse`  | API key / JWT   | The 6 alpha-flagged registry rows (`alpha_v0_1: true`). |
| 5 | `GET`  | `/v1/packs`             | `PackListResponse`    | API key / JWT   | One synthetic pack per alpha source. Filterable by `source_urn`. |
| ‑ | *any*  | `/api/v1/{path:path}`   | inline JSON           | unauthenticated | **410 Gone** with `alpha_endpoints` migration list; hidden from OpenAPI. |

Every successful response also stamps `X-GL-Release-Profile: alpha-v0.1`
so legacy clients can detect they hit a profile-locked surface.

---

## Component Schemas (typed response models)

All five endpoints declare a `200` response that `$ref`s into
`components.schemas`:

| Schema | Used by | Required keys |
|--------|---------|---------------|
| `HealthzResponse`     | `/v1/healthz`            | `status`, `service`, `release_profile`, `schema_id` |
| `FactorListResponse`  | `/v1/factors`            | (`data`, `next_cursor`, `edition` all optional) |
| `FactorV0_1`          | `/v1/factors/{urn}` (item shape used by list too) | `urn` |
| `SourceListResponse`  | `/v1/sources`            | (none — empty default) |
| `SourceV0_1`          | item under `/v1/sources` | `urn`, `source_id` |
| `PackListResponse`    | `/v1/packs`              | (none — empty default) |
| `PackV0_1`            | item under `/v1/packs`   | `urn`, `source_urn`, `pack_id`, `version` |
| `ErrorResponse`       | typed 404 on `/v1/factors/{urn}` | `error`, `message` |

`HTTPValidationError` and `ValidationError` are FastAPI's stock 422
schemas, included automatically when query/path params are typed.

---

## What's NOT in this surface

The following surfaces are **intentionally absent** under `alpha-v0.1`
and return 404. Each is gated by an entry in `FEATURES` in
`greenlang/factors/release_profile.py` and unlocks at `beta-v0.5` or later:

* `/v1/resolve` (POST + GET) — full 7-step resolve cascade
* `/v1/explain`, `/v1/factors/{id}/explain` — explain payloads
* `/v1/batch` — batched resolve
* `/v1/coverage`, `/v1/quality/fqs` — coverage / FQS endpoints
* `/v1/editions/{id}` — signed edition manifest
* `/v1/method-packs/*` — method-pack coverage
* `/v1/admin/*` — operator console
* `/v1/billing/*`, `/v1/oem/*` — commercial surfaces
* `/v1/graphql` — GraphQL gateway

---

## OpenAPI 3.1 Spec (verbatim)

The JSON below is the exact byte-for-byte content of
[`tests/factors/v0_1_alpha/openapi_alpha_v0_1.json`](../../../tests/factors/v0_1_alpha/openapi_alpha_v0_1.json)
(26,645 bytes; 983 lines). The contract test
`test_openapi_alpha_v0_1_matches_snapshot` fails CI if the live
`/openapi.json` drifts from this fixture.

The volatile `servers` block is stripped before snapshotting (it embeds
the dynamic FastAPI test-host URL and is not part of the contract).

```json
{
  "components": {
    "schemas": {
      "ErrorResponse": {
        "description": "Stable error payload shape for the alpha surface.",
        "properties": {
          "allowed": {
            "anyOf": [
              { "items": { "type": "string" }, "type": "array" },
              { "type": "null" }
            ],
            "title": "Allowed"
          },
          "error": { "title": "Error", "type": "string" },
          "message": { "title": "Message", "type": "string" },
          "urn": {
            "anyOf": [{ "type": "string" }, { "type": "null" }],
            "title": "Urn"
          }
        },
        "required": ["error", "message"],
        "title": "ErrorResponse",
        "type": "object"
      },
      "FactorListResponse": {
        "description": "Cursor-paginated list response for ``GET /v1/factors``.",
        "properties": {
          "data": {
            "items": { "$ref": "#/components/schemas/FactorV0_1" },
            "title": "Data",
            "type": "array"
          },
          "edition": {
            "anyOf": [{ "type": "string" }, { "type": "null" }],
            "title": "Edition"
          },
          "next_cursor": {
            "anyOf": [{ "type": "string" }, { "type": "null" }],
            "title": "Next Cursor"
          }
        },
        "title": "FactorListResponse",
        "type": "object"
      },
      "FactorV0_1": {
        "description": "Single factor record in the v0.1 alpha shape.",
        "properties": {
          "boundary": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Boundary" },
          "category": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Category" },
          "citations": {
            "items": { "additionalProperties": true, "type": "object" },
            "title": "Citations",
            "type": "array"
          },
          "description": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Description" },
          "extraction": { "additionalProperties": true, "title": "Extraction", "type": "object" },
          "factor_id_alias": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Factor Id Alias" },
          "factor_pack_urn": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Factor Pack Urn" },
          "geography_urn": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Geography Urn" },
          "gwp_basis": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Gwp Basis" },
          "gwp_horizon": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "title": "Gwp Horizon" },
          "licence": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Licence" },
          "methodology_urn": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Methodology Urn" },
          "name": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Name" },
          "published_at": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Published At" },
          "resolution": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Resolution" },
          "review": { "additionalProperties": true, "title": "Review", "type": "object" },
          "source_urn": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Source Urn" },
          "tags": { "items": { "type": "string" }, "title": "Tags", "type": "array" },
          "unit_urn": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Unit Urn" },
          "urn": { "description": "Canonical factor URN.", "title": "Urn", "type": "string" },
          "value": { "anyOf": [{ "type": "number" }, { "type": "null" }], "title": "Value" },
          "vintage_end": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Vintage End" },
          "vintage_start": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Vintage Start" }
        },
        "required": ["urn"],
        "title": "FactorV0_1",
        "type": "object"
      },
      "HealthzResponse": {
        "description": "Response shape for ``GET /v1/healthz``.",
        "properties": {
          "edition": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Edition" },
          "git_commit": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Git Commit" },
          "release_profile": { "title": "Release Profile", "type": "string" },
          "schema_id": { "title": "Schema Id", "type": "string" },
          "service": { "title": "Service", "type": "string" },
          "status": { "title": "Status", "type": "string" },
          "version": { "default": "0.1.0", "title": "Version", "type": "string" }
        },
        "required": ["status", "service", "release_profile", "schema_id"],
        "title": "HealthzResponse",
        "type": "object"
      },
      "PackListResponse": {
        "description": "Response shape for ``GET /v1/packs``.",
        "properties": {
          "count": { "default": 0, "title": "Count", "type": "integer" },
          "data": {
            "items": { "$ref": "#/components/schemas/PackV0_1" },
            "title": "Data",
            "type": "array"
          }
        },
        "title": "PackListResponse",
        "type": "object"
      },
      "PackV0_1": {
        "description": "Factor pack (alpha shape).",
        "properties": {
          "display_name": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Display Name" },
          "factor_count": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "title": "Factor Count" },
          "pack_id": { "title": "Pack Id", "type": "string" },
          "source_urn": { "title": "Source Urn", "type": "string" },
          "urn": { "title": "Urn", "type": "string" },
          "version": { "title": "Version", "type": "string" }
        },
        "required": ["urn", "source_urn", "pack_id", "version"],
        "title": "PackV0_1",
        "type": "object"
      },
      "SourceListResponse": {
        "description": "Response shape for ``GET /v1/sources``.",
        "properties": {
          "count": { "default": 0, "title": "Count", "type": "integer" },
          "data": {
            "items": { "$ref": "#/components/schemas/SourceV0_1" },
            "title": "Data",
            "type": "array"
          }
        },
        "title": "SourceListResponse",
        "type": "object"
      },
      "SourceV0_1": {
        "description": "Public source registry entry in the v0.1 shape.",
        "properties": {
          "cadence": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Cadence" },
          "citation_text": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Citation Text" },
          "display_name": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Display Name" },
          "jurisdiction": {
            "anyOf": [
              { "type": "string" },
              { "items": { "type": "string" }, "type": "array" },
              { "type": "null" }
            ],
            "title": "Jurisdiction"
          },
          "latest_ingestion_at": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Latest Ingestion At" },
          "license_class": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "License Class" },
          "provenance_completeness_score": { "anyOf": [{ "type": "number" }, { "type": "null" }], "title": "Provenance Completeness Score" },
          "publication_url": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Publication Url" },
          "publisher": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Publisher" },
          "source_id": { "title": "Source Id", "type": "string" },
          "source_version": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Source Version" },
          "urn": { "title": "Urn", "type": "string" }
        },
        "required": ["urn", "source_id"],
        "title": "SourceV0_1",
        "type": "object"
      }
    }
  },
  "info": {
    "description": "Canonical climate reference layer.",
    "title": "GreenLang Factors API",
    "version": "1.0.0"
  },
  "openapi": "3.1.0",
  "paths": {
    "/v1/healthz":         { "get": { "operationId": "healthz_v1_healthz_get",       "summary": "Service health + edition id",       "tags": ["factors-v0.1-alpha"], "responses": { "200": { "description": "Successful Response", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/HealthzResponse" } } } } } } },
    "/v1/factors":         { "get": { "operationId": "list_factors_v1_factors_get",  "summary": "List factors (cursor-paginated, filtered)", "tags": ["factors-v0.1-alpha"], "responses": { "200": { "description": "Successful Response", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FactorListResponse" } } } } } } },
    "/v1/factors/{urn}":   { "get": { "operationId": "get_factor_v1_factors__urn__get", "summary": "Get one factor by URN",          "tags": ["factors-v0.1-alpha"], "responses": { "200": { "description": "Successful Response", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FactorV0_1" } } } }, "404": { "description": "Factor not found in the active edition.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } } } } },
    "/v1/sources":         { "get": { "operationId": "list_sources_v1_sources_get",  "summary": "List registered alpha sources",     "tags": ["factors-v0.1-alpha"], "responses": { "200": { "description": "Successful Response", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SourceListResponse" } } } } } } },
    "/v1/packs":           { "get": { "operationId": "list_packs_v1_packs_get",      "summary": "List factor packs grouped by source", "tags": ["factors-v0.1-alpha"], "responses": { "200": { "description": "Successful Response", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/PackListResponse" } } } } } } }
  }
}
```

> **NOTE:** The block above is a human-readable digest with operations
> collapsed onto one line per path for visual scanning. The
> byte-for-byte authoritative spec lives in
> `tests/factors/v0_1_alpha/openapi_alpha_v0_1.json`. Drift between the
> digest and the JSON fixture is **expected** when query-parameter
> documentation gets richer; only the JSON fixture is contract-tested.

---

## Sample Responses

### `GET /v1/healthz` (200)

```json
{
  "status": "ok",
  "service": "greenlang-factors",
  "release_profile": "alpha-v0.1",
  "schema_id": "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json",
  "edition": "builtin-v1.0.0",
  "git_commit": "72154a93917d",
  "version": "0.1.0"
}
```

### `GET /v1/factors/urn:gl:factor:nope:nope:nope:v1` (404)

```json
{
  "error": "factor_not_found",
  "message": "Factor 'urn:gl:factor:nope:nope:nope:v1' not found in edition 'builtin-v1.0.0'.",
  "urn": "urn:gl:factor:nope:nope:nope:v1"
}
```

### `GET /v1/factors?category=invalid` (400)

```json
{
  "error": "invalid_category",
  "message": "category='invalid' is not in the alpha v0.1 enum.",
  "allowed": [
    "scope1", "scope2_location_based", "scope2_market_based",
    "grid_intensity", "fuel", "refrigerant", "fugitive",
    "process", "cbam_default"
  ]
}
```

### `GET /api/v1/factors` (410 Gone)

```json
{
  "error": "endpoint_gone",
  "message": "/api/v1 is not part of the v0.1 alpha contract; use /v1/...",
  "alpha_endpoints": [
    "/v1/healthz",
    "/v1/factors",
    "/v1/factors/{urn}",
    "/v1/sources",
    "/v1/packs"
  ],
  "requested_path": "/api/v1/factors"
}
```

---

## CI Drift Detection

The contract test
[`test_alpha_api_contract.py`](../../../tests/factors/v0_1_alpha/test_alpha_api_contract.py)
runs in every PR pipeline. It asserts:

1. The OpenAPI `paths` keys equal exactly the 5 alpha endpoints.
2. Every alpha operation is HTTP `GET` only.
3. Every alpha 200 response declares a `$ref` into `components.schemas`.
4. The live spec matches the saved JSON snapshot byte-for-byte (after
   stripping `servers`).

If a PR intentionally evolves the alpha contract, regenerate the
snapshot:

```bash
UPDATE_OPENAPI_SNAPSHOT=1 pytest \
  tests/factors/v0_1_alpha/test_alpha_api_contract.py::test_openapi_alpha_v0_1_matches_snapshot
```

…then commit the updated `openapi_alpha_v0_1.json` AND update this
document's endpoint table / schema list.
