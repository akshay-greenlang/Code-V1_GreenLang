# Phase 2 / WS2 — API + SDK URN Primary-ID Audit

> **Authority**: CTO Phase 2 brief Section 2.2 — "Confirm API
> `/v0_1/alpha/factors` and SDK `factors.get_by_urn()` expose `urn` as
> primary identifier in every response shape."
> **Date**: 2026-04-27
> **Owner**: GL-BackendDeveloper
> **Status**: PASSED — `urn` is the primary identifier on every alpha
> response shape today; the only follow-up was the addition of
> `find_by_alias` / `GET /v1/factors/by-alias/{legacy_id}` and the
> matching SDK helper `client.get_by_alias()`.

---

## Audit scope

The Phase 2 contract is that the v0.1 alpha public API and SDK MUST
return the canonical `urn:gl:factor:...` URN as the **primary**
identifier on every response shape, with the legacy `EF:...` form
demoted to an optional `factor_id_alias` sibling. The frozen JSON
schema (`config/schemas/factor_record_v0_1.schema.json`) already
enforces this — the audit confirms each downstream consumer agrees.

Files examined:

| Layer  | File | Verdict |
|---|---|---|
| API models | `greenlang/factors/api_v0_1_alpha_models.py` | PASS — `FactorV0_1.urn` is the only required `str` identifier; `factor_id_alias` is `Optional[str]` |
| API routes | `greenlang/factors/api_v0_1_alpha_routes.py` | PASS — `_coerce_v0_1` builds canonical URN as primary, demotes legacy `EF:...` to `factor_id_alias`. `/v1/factors/{urn}` route uses URN as the path key, not legacy id. |
| SDK models | `greenlang/factors/sdk/python/models.py` | PASS — `AlphaFactor.urn` is required; `AlphaFactor.factor_id_alias` is `Optional`. The legacy `Factor` (v1.x) class still carries `factor_id` at the top of the file but is gated behind beta+ profile and never returned by alpha endpoints. |
| SDK client | `greenlang/factors/sdk/python/client.py` | PASS — `client.get_factor(urn)` validates URN client-side and parses into `AlphaFactor`. Legacy id strings raise `ValueError` BEFORE any network round-trip. |

No client code returned `factor_id` as the primary identifier. The
single audit gap was the missing alias-lookup helper covered by the
new code added in this work-stream.

---

## What was already correct

### `api_v0_1_alpha_models.FactorV0_1`

```python
class FactorV0_1(_AlphaBase):
    urn: str = Field(..., description="Canonical factor URN.")
    factor_id_alias: Optional[str] = None
    # ...
```

- `urn` required, type `str`.
- `factor_id_alias` optional, type `Optional[str]`.
- Matches the JSON schema's `required: ["urn", ...]` and the
  `factor_id_alias` non-required slot.

### `api_v0_1_alpha_routes._coerce_v0_1`

When converting a legacy `EmissionFactorRecord` into the v0.1 wire
shape, the helper builds a canonical URN:

```python
if isinstance(factor_id, str) and factor_id.startswith("urn:gl:factor:"):
    urn = factor_id
    alias = None
else:
    urn = f"urn:gl:factor:{src_slug}:{namespace}:{leaf}:v1"
    alias = factor_id if factor_id else None

return {
    "urn": urn,
    "factor_id_alias": alias,
    # ...
}
```

The output dict carries `urn` first (literally — Python dict key order)
and demotes the legacy `EF:...` to `factor_id_alias`. The
`/v1/factors/{urn:path}` route receives the URN as the path argument
and never accepts a legacy id at the URL level.

### `sdk.python.models.AlphaFactor`

```python
class AlphaFactor(BaseModel):
    urn: str = Field(..., description="Canonical primary id (urn:gl:factor:...).")
    factor_id_alias: Optional[str] = Field(
        None, description="Legacy EF: identifier — non-canonical alias only."
    )
    # ...
```

- `urn` required.
- `factor_id_alias` optional.
- The model lives alongside the legacy `Factor` (v1.x) class, but
  `client.list_factors()` / `client.get_factor()` always model-validate
  into `AlphaFactor`, never the legacy `Factor`. Beta+ continues to use
  the legacy class — that's by design (the v1.x surface is gated
  behind `_gate("resolve_endpoint", ...)`).

### `sdk.python.client.FactorsClient.get_factor` / `.list_factors`

Both methods return `AlphaFactor` / `ListFactorsResponse` (which wraps
`AlphaFactor`). `get_factor` validates the URN argument client-side
and raises `ValueError` if the caller passes a legacy id, so the
network never sees a legacy-id primary key.

---

## What needed adding (Phase 2 / WS2 deliverables)

### Backend repository — already shipped by WS7

The repository at `greenlang/factors/repositories/alpha_v0_1_repository.py`
already exposed:

* `find_by_alias(legacy_id) -> Optional[Dict]` — joins the SQLite
  mirror table `alpha_factor_aliases_v0_1` (or the Postgres
  `factors_v0_1.factor_aliases` table) against the canonical factor
  table and returns the full record dict.
* `register_alias(urn, legacy_id, kind='EF')` — append-only writer.

These were verified end-to-end against the new tests below.

### API route — added in this work-stream

```text
GET /v1/factors/by-alias/{legacy_id}
```

Resolves a legacy id to its canonical record. Returns the same
`FactorV0_1` shape as `/v1/factors/{urn}` (so the response carries
`urn` as primary, `factor_id_alias` as secondary). Returns 404 with a
stable JSON body when no alias matches. The route is registered
**before** the catch-all `/factors/{urn:path}` so the `by-alias`
prefix wins the route-matching contest.

### SDK helper — added in this work-stream

```python
factor = client.get_by_alias("EF:US:grid:eGRID-SERC:2024:v1")
if factor is None:
    raise SystemExit("alias not found")
print(factor.urn)              # urn:gl:factor:...
print(factor.factor_id_alias)  # EF:US:grid:eGRID-SERC:2024:v1
```

Both `FactorsClient.get_by_alias` (sync) and
`AsyncFactorsClient.get_by_alias` (async) are wired. Both return
`Optional[AlphaFactor]` so `None` denotes "no alias matched" while
non-404 server errors still raise `FactorsAPIError` for the caller to
handle.

### `ALPHA_ENDPOINTS_PUBLIC` updated

The deprecated 410 catch-all (`/api/v1/{path:path}`) embeds the public
endpoint list in its response so clients hitting the legacy prefix can
discover the new surface. We added `/v1/factors/by-alias/{legacy_id}`
to the list.

---

## Files modified

| File | Change |
|---|---|
| `greenlang/factors/api_v0_1_alpha_routes.py` | +new route `GET /v1/factors/by-alias/{legacy_id}`; +entry in `ALPHA_ENDPOINTS_PUBLIC`. |
| `greenlang/factors/sdk/python/client.py` | +`FactorsClient.get_by_alias` (sync); +`AsyncFactorsClient.get_by_alias` (async). |
| `docs/factors/PHASE_2_API_URN_AUDIT.md` | NEW — this doc. |

## Files NOT modified

The audit found nothing wrong with the following — they remain
canonical:

* `greenlang/factors/api_v0_1_alpha_models.py` — `FactorV0_1.urn` is
  primary, `factor_id_alias` is optional. No change needed.
* `greenlang/factors/sdk/python/models.py` — `AlphaFactor.urn` is
  primary; `AlphaFactor.factor_id_alias` is optional. No change needed.
* `greenlang/factors/ontology/urn.py` — load-bearing for 56+ existing
  tests. NOT touched.
* `config/schemas/factor_record_v0_1.schema.json` — frozen since
  2026-04-25; not touched.

---

## Tests updated / added

| Test file | Purpose |
|---|---|
| `tests/factors/v0_1_alpha/phase2/test_urn_property_roundtrip.py` | Property-based round-trip + curated negative corpus across every URN kind except `activity` (owned by WS5). |
| `tests/factors/v0_1_alpha/phase2/test_urn_lowercase_sweep.py` | Walks every catalog seed + source registry YAML/JSON; asserts every `urn:gl:` literal parses cleanly and respects the lowercase invariant. |
| `tests/factors/v0_1_alpha/phase2/test_urn_uniqueness_db.py` | sqlite-parameterised: duplicate URN INSERT raises; alias UNIQUE constraint rejects duplicate `legacy_id`. |
| `tests/factors/v0_1_alpha/phase2/test_api_urn_primary.py` | Exercises every alpha API + SDK path; asserts the response carries `urn` (not `factor_id`) as the primary id. |
| `tests/factors/v0_1_alpha/phase2/test_alias_backfill_idempotency.py` | Runs the backfill twice; asserts the second run inserts zero rows. |

The pre-existing test suite (56 cases in `tests/factors/v0_1_alpha/test_urn.py`,
plus the alpha API contract / SDK surface / publisher tests) continues
to pass — the audit changes are strictly additive.

---

## Backward compatibility

* Existing clients that read `factor_id_alias` keep working — the
  field stays in the response. Servers that emit `EF:...` continue to
  do so.
* The new `/v1/factors/by-alias/{legacy_id}` endpoint is additive; no
  legacy route was renamed or removed.
* The SDK `client.get_by_alias()` helper is additive; no legacy SDK
  method changed signature.
* The frozen JSON Schema's contract (`urn` required,
  `factor_id_alias` optional) was unchanged.

The only documented change for clients is the optional new alias-lookup
endpoint — there is no breaking change.
