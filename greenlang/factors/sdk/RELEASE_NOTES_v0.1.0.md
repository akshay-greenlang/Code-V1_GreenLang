# greenlang-factors 0.1.0 — v0.1 Alpha (FY27 Q1)

**Release date**: 2026-04-25
**Status**: Alpha (Development Status :: 3 - Alpha)
**Audience**: Internal + 2 design partners (one India-linked exporter, one EU-facing manufacturer)

This is the first alpha of the GreenLang Factors Python SDK. It ships ONLY the read-only contract defined by the CTO source-of-truth document §19.1. Breaking changes are expected at every minor version until 1.0 GA.

## Why we renumbered from 1.3.0 → 0.1.0

The 1.x line was forward-development released too aggressively. Per CTO doc §19.1: "SDK marked 0.x, with a clear note that breaking changes are expected until v1.0 GA." The distribution name `greenlang-factors` is preserved; only the version number is collapsed. The forward-dev surface (resolve, explain, batch, edition pinning, signed-receipt verification) remains in the codebase but is gated behind the `release_profile.feature_enabled(...)` flag — under `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1` those methods raise `ProfileGatedError`. They re-enable at `beta-v0.5` and higher.

## What ships in 0.1.0

Five read-only HTTP GETs, typed Pydantic v2 models, retries with exponential backoff, request timeouts, `py.typed`, basic filter args:

| SDK call | Wire endpoint | Notes |
|---|---|---|
| `client.health()` | `GET /v1/healthz` | Unauthenticated. Returns service status + edition id. |
| `client.list_factors(...)` | `GET /v1/factors` | Filters: `geography_urn`, `source_urn`, `pack_urn`, `category`, `vintage_start_after`, `vintage_end_before`. Cursor-paginated. |
| `client.get_factor(urn)` | `GET /v1/factors/{urn}` | URN as canonical primary id. |
| `client.list_sources()` | `GET /v1/sources` | Lists the 6 alpha sources. |
| `client.list_packs()` | `GET /v1/packs` | Lists factor packs grouped by source. |

## What is OUT OF SCOPE for 0.1.0

Per CTO doc §19.1 "Explicitly out of scope":

- No resolve / explain / batch endpoints
- No signed-receipt verification, no offline verifier
- No edition-pinning headers
- No CLI beyond `gl factors get|list|sources|packs|health`
- No GraphQL, no SQL-over-HTTP
- No commercial billing / OEM / admin
- No async client surface change vs sync (one client class)
- No TypeScript SDK (deferred to v0.5)

## Wire contract — what an alpha factor returns

```json
{
  "urn": "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1",
  "factor_id_alias": "EF:IPCC:stationary-combustion:natural-gas-residential:v1",
  "source_urn": "urn:gl:source:ipcc-ar6",
  "factor_pack_urn": "urn:gl:pack:ipcc-ar6:tier-1-defaults:v1",
  "name": "Stationary combustion of natural gas (residential), CO2e",
  "description": "...",
  "category": "fuel",
  "value": 56100.0,
  "unit_urn": "urn:gl:unit:kgco2e/tj",
  "gwp_basis": "ar6",
  "gwp_horizon": 100,
  "geography_urn": "urn:gl:geo:global:world",
  "vintage_start": "2021-01-01",
  "vintage_end": "2099-12-31",
  "resolution": "annual",
  "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
  "boundary": "...",
  "licence": "IPCC-PUBLIC",
  "citations": [...],
  "published_at": "2026-04-25T12:00:00Z",
  "extraction": { "source_url": "...", "raw_artifact_sha256": "...", "parser_id": "...", "parser_version": "0.1.0", "parser_commit": "...", ... },
  "review": { "review_status": "approved", "approved_by": "human:methodology-lead@greenlang.io", ... }
}
```

Validates against `factor_record_v0_1.schema.json` ($id: `https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json`).

## Authentication

Alpha is **read-only** for design partners. Pass an API key via `X-GL-API-Key` header. No JWT, no OAuth2, no HMAC signing in alpha.

```python
from greenlang.factors.sdk.python import FactorsClient, APIKeyAuth

client = FactorsClient(
    base_url="https://factors-alpha.greenlang.io",
    auth=APIKeyAuth(api_key="gl_alpha_..."),
)

print(client.health())
factor = client.get_factor("urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1")
print(factor.value, factor.unit_urn)
```

## Compatibility

- Python: 3.10, 3.11, 3.12, 3.13
- Wire schema: `factor_record_v0_1` (frozen 2026-04-25)
- Server: GreenLang Factors v0.1 Alpha (`GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`)

## Breaking changes vs hypothetical 1.x

- Distribution version is now `0.1.0`, not `1.3.0`. If you pinned `greenlang-factors==1.3.0`, switch to `==0.1.0` and reduce the call surface to the 5 GETs.
- Factor primary id is `urn` (not `factor_id`). The `EF:...` form is still surfaced as `factor_id_alias` for one release.
- `gwp_basis` is `ar6` only in alpha; AR4/AR5 records are rejected.

## What's next (v0.5 closed beta — FY27 Q2)

- Add `client.resolve(activity, geography, vintage, methodology)` returning a chosen factor + alternates + explain payload
- Add `client.get_factor_explain(urn)` returning the audit-grade explanation
- Add edition pinning via `X-GL-Edition` header
- TypeScript SDK at `@greenlang/factors==0.5.0`
- Add CBAM defaults + India BEE + IEA (commercial connector)

## Acknowledgements

This release closes the v0.1 Alpha exit criteria from the CTO source-of-truth document §19.1: schema approved, first parsers working, basic API + SDK, provenance fields complete for alpha sources, two design partners productive on the SDK.
