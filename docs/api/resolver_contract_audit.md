# Resolver Contract Audit — `/v1/resolve` and `/v1/explain`

**Date:** 2026-04-23
**Scope:** GL-APIDeveloper audit for CTO Master ToDo tasks **R4**, **R6**, **API2** — Wave 1 (plan only; Wave 2 implements).
**Start-point files:**
- `greenlang/factors/api_v1_routes.py`
- `greenlang/factors/api_endpoints.py`
- `greenlang/factors/resolution/result.py`
- `greenlang/factors/resolution/engine.py`
- `greenlang/factors/middleware/signed_receipts.py`
- `greenlang/factors/middleware/edition_pin.py`
- `greenlang/factors/factors_app.py`
- `docs/api/factors-v1.yaml`

**Routes under audit:**
- `POST /v1/resolve` (`api_v1_routes.py:119`) — body-form cascade resolve
- `GET  /v1/factors/{factor_id}` (`api_v1_routes.py:144`) — factor-by-id with explain
- `GET  /v1/factors/{factor_id}/explain` (`api_v1_routes.py:169`) — explain-only payload

---

## 1. 16-element contract coverage

> **Response-shape reality:** both endpoints currently return **the raw dict produced by `build_factor_explain` / `build_resolution_explain`**, which is `ResolvedFactor.model_dump()` plus `alternates` + `explain` siblings. This is **not** the `ResolvedFactorResponse` envelope (`{resolved, edition_id, edition_source, signed_receipt}`) that `docs/api/factors-v1.yaml` advertises at line 174 and line 218. That shape mismatch is the root of several P0 gaps below.
>
> The `signed_receipt` IS attached at runtime, but under the key `_signed_receipt` (an envelope injected post-hoc by `SignedReceiptsMiddleware._inject_receipt_into_json` at `signed_receipts.py:222`), not the OpenAPI-declared `signed_receipt`. The middleware IS installed via `factors_app.py:101` — contradicting the MEMORY.md claim that it is dormant; it is fully wired but under a non-spec key.

| # | Required field | Present in `/v1/resolve`? | Present in `/v1/explain`? | Evidence (file:line) | Gap | Severity |
|---|---|---|---|---|---|---|
| 1 | `chosen_factor` `{id, name, version, factor_family}` | **partial** — flat `chosen_factor_id`, `chosen_factor_name`, `factor_version` at top level; **no** nested `chosen_factor` object; **no** `factor_family` | **partial** — `explain.chosen` has `factor_id`, `factor_name`, `factor_version` but **no** `factor_family` | `result.py:56-60`, `result.py:101-111`, `factors-v1.yaml:1386` (FactorFamily enum exists on `Factor`, not on `ResolvedFactor`) | No `factor_family` populated in `_build_resolved_factor`; not projected through to `ResolvedFactor` model or `explain()` view | **P0** |
| 2 | `alternates[]` (top-3), `{factor_id, reason_lost}` | **yes** — `alternates[]` present, cap via `alternates_limit` (default 5, max 20); uses key `why_not_chosen` (not `reason_lost`) | **yes** — same array attached at `explain.alternates` and root-level `alternates` | `api_endpoints.py:688-693`, `result.py:38-46` | **Field name drift**: spec calls for `reason_lost`; code emits `why_not_chosen`. Top-N default is 5, spec says "up to 3". Cosmetic but SDK-breaking. | **P1** |
| 3 | `why_this_won` (non-empty string) | **partial** — present as `why_chosen` (different name) | **partial** — present as `explain.derivation.why_chosen` | `result.py:64`, `engine.py:329` | **Field name drift**: spec demands `why_this_won`, code emits `why_chosen`. Always populated (falls back to a templated sentence), so semantics OK. | **P1** |
| 4 | `source` `{id, version, name, authority}` | **partial** — flat `source_id`, `source_version` only; **no** `source_name`, **no** `authority` | **partial** — `explain.chosen.source` is a **string** (source_id alone), no nested object | `result.py:57-59`, `result.py:105-107`, `engine.py:331-342` | Source is coerced into a single string at `engine.py:331-336`; spec requires a four-field nested object. `authority` (regulator / publisher) is never projected. | **P0** |
| 5 | `factor_version` AND `release_version` (distinct) | **partial** — `factor_version` present; `release_version` is read from record if present (`engine.py:339`) but **never surfaced** on `ResolvedFactor` (collapsed into `source_version`) | **partial** — same collapse | `result.py:60`, `engine.py:337-342` | `ResolvedFactor` has no `release_version` attribute — both are aliased into one `source_version`. Loses CTO-required distinct semantics (catalog release vs. individual factor version). | **P0** |
| 6 | `method_pack` and `method_pack_version` | **partial** — `method_pack_version` present; `method_pack` (pack name/id, e.g. `ghg_protocol_corporate`) is **not** emitted | **partial** — `explain.meta.method_pack_version` only | `result.py:96`, `result.py:142` | `ResolvedFactor` exposes only `method_profile` (string like `corporate_scope2_location_based`) and `method_pack_version`; the pack's canonical id is dropped. | **P1** |
| 7 | `valid_from` / `valid_to` (ISO dates) | **no** — neither field is on `ResolvedFactor` | **no** — explain payload does not project them | `result.py:49-97` (absent) | The underlying `EmissionFactorRecord` has `valid_from`, `valid_to`; the resolver never reads or emits them. Critical for vintage defense. | **P0** |
| 8 | `gas_breakdown` `{co2, ch4, n2o, hfcs, pfcs, sf6, nf3, biogenic_co2, co2e_total, gwp_basis}` | **yes** — all 10 sub-fields present (suffix `_kg`: `co2_kg`, `ch4_kg`, …, `co2e_total_kg`, `gwp_basis`) | **yes** — mirrored as `explain.emissions` | `result.py:13-25`, `engine.py:580-597` | **Field-name drift**: spec asks for `co2`, code emits `co2_kg` (unit suffix). Present in full; rename-only concern. | **P1** |
| 9 | `co2e_basis` (GWP set + horizon, e.g. `"IPCC_AR6_100"`) | **partial** — only inside `gas_breakdown.gwp_basis`; no top-level `co2e_basis` field | **partial** — same, via `explain.emissions.gwp_basis` | `result.py:25`, `engine.py:580-596` | CTO spec calls out `co2e_basis` as a top-level field; currently only reachable via `gas_breakdown.gwp_basis`. Worth a top-level alias for developer ergonomics. | **P1** |
| 10 | `quality` `{composite_fqs_0_100, temporal_score, geographic_score, technology_score, verification_score, completeness_score}` | **no** — `quality_score` is a single scalar (DQS `overall_score`, 1-5 scale) | **no** — `explain.quality.score` is same scalar | `result.py:77`, `engine.py:317-319`, `composite_fqs.py` (exists but not called on this path) | `ResolutionEngine._build_resolved_factor` copies `dqs.overall_score` only; it never calls `compute_fqs()` from `greenlang/factors/quality/composite_fqs.py`, and never projects the 5 per-dimension scores. The 0-100 surface exists in code but is disconnected from `/resolve` and `/explain`. | **P0** |
| 11 | `uncertainty` `{type, low, high, distribution}` | **partial** — `UncertaintyBand` has `distribution`, `ci_95_percent`, `low`, `high`, `note`; no discriminator `type` (`"95_percent_ci"` vs `"qualitative"`); `low`/`high` are typically null because only `ci_95_percent` gets populated at `engine.py:313` | **partial** — same | `result.py:28-35`, `engine.py:311-315` | `low` and `high` are never computed from `ci_95_percent` + central estimate; `type` field missing. Consumers cannot distinguish quantitative from qualitative uncertainty. | **P0** |
| 12 | `licensing` `{redistribution_class}` (enum: open \| licensed_embedded \| customer_private \| oem_redistributable) | **partial** — `redistribution_class` is flat on `ResolvedFactor`; no nested `licensing` object | **partial** — `explain.chosen.redistribution_class` only | `result.py:63`, `engine.py:352` | Per CTO spec #12, the envelope must be `licensing: { redistribution_class }` (room for `customer_entitlement_required`, watermark hints). Currently only the flat scalar. Enum values ARE defined in `factors-v1.yaml:1044`. | **P1** |
| 13 | `assumptions[]` (array of strings) | **yes** — always non-empty (falls back to `[f"Resolved via step {rank} ({label})."]` at `engine.py:306-307`) | **yes** — `explain.derivation.assumptions` | `result.py:71`, `engine.py:303-307` | None — fully present. | — |
| 14 | `fallback_rank` (int 1-7) | **yes** — integer in `[1, 7]`; step label matches the 7-tier order defined at `engine.py:375-383` | **yes** — `explain.derivation.fallback_rank` | `result.py:67`, `engine.py:375-383` | None — fully present and correct. | — |
| 15 | `deprecation_status` `{status: active\|deprecated\|superseded, replacement_pointer_factor_id: nullable}` | **partial** — `deprecation_status` is a **string or null** ("deprecated" only; no "active" / "superseded"); `deprecation_replacement` is flat, not nested under `replacement_pointer_factor_id` | **partial** — same, under `explain.derivation` | `result.py:72-73`, `engine.py:357-358` | (a) Status is `None` when factor is active instead of the explicit `"active"` enum; (b) `"superseded"` state is never produced; (c) nested envelope `{status, replacement_pointer_factor_id}` missing. No helper module `resolution/deprecation_pointer.py` exists. | **P0** |
| 16 | `signed_receipt` `{receipt_id, signature, verification_key_hint, alg}` | **partial** — receipt IS attached by `SignedReceiptsMiddleware` (installed at `factors_app.py:101`), but under the key `_signed_receipt`, not `signed_receipt`; fields are `signature, algorithm, signed_at, key_id, signed_over` — no `receipt_id`; the OpenAPI `Receipt` schema at `factors-v1.yaml:2347` specifies `signature, algorithm, signed_at, key_id, payload_hash` | **partial** — same middleware path, same key-name drift | `signed_receipts.py:218-233`, `signed_receipts.py:395-471`, `factors_app.py:99-103` | (a) Top-level key is `_signed_receipt`, not `signed_receipt` (breaks OpenAPI contract which requires `signed_receipt` as a required envelope field); (b) no `receipt_id` field; (c) no `verification_key_hint` (closest is `key_id`); (d) `alg` vs `algorithm` minor drift. Middleware is **NOT dormant** — it is wired and signs every 2xx /v1 response. | **P0** |

**Legend:** yes = field present with correct name and shape; partial = present but mis-named, flat-instead-of-nested, or incomplete; no = absent entirely.

---

## 2. Secondary verifications

### 2.1 Does `/v1/factors/{id}/explain` exist and work standalone?
**Yes** — defined at `api_v1_routes.py:169-187`. It delegates to `build_factor_explain` (`api_endpoints.py:650-693`), returns `payload.get("explain", payload)`. It is standalone (no body required). **But the shape drift** is severe: it returns the compact `explain()` dict (`{chosen, derivation, quality, uncertainty, emissions, unit_conversion, alternates, meta}`) defined at `result.py:99-144`, **not** the OpenAPI `ExplainResponse` envelope (`factors-v1.yaml:1807-1827`) which requires top-level `chosen_factor_id, method_profile, fallback_rank, step_label, gas_breakdown, uncertainty, alternates, explain, edition_id, signed_receipt`. **OpenAPI drift — P0.**

### 2.2 Does `/v1/resolve` return the explain block by default with `compact=true` opt-out?
**Yes** — `api_v1_routes.py:119-141`. `compact: bool = Query(False)` defaults to False (explain included). When `compact=True`, `_strip_explain` pops `explain` AND `alternates` and adds `_compact: True` (`api_v1_routes.py:70-77`). This satisfies CTO non-negotiable #3 ("never hide fallback logic") in the affirmative default.

### 2.3 Is `X-GL-Edition` header respected for version pinning?
**Yes** — `EditionPinMiddleware` at `edition_pin.py:24-74`. Resolution order: `X-GL-Edition` header → `?edition=` query → service default. The resolved id is stashed on `request.state.edition_id` and echoed on response as `X-GreenLang-Edition`. Validation calls `service.repo.resolve_edition(edition_id)`; an unknown edition raises `_UnknownEditionError` which the middleware converts to a 404 JSON body. Installation order (outer-in): `EditionPin` is added **last** (`factors_app.py:125`) which in Starlette means it runs **outermost** — so `request.state.edition_id` is set before every inner route. **Correct.**

### 2.4 Is the signed receipt attached to every factor-returning response by default?
**Yes — contrary to MEMORY.md** — `SignedReceiptsMiddleware` is added at `factors_app.py:100-103`. It signs every 2xx JSON response whose path starts with `/v1` (`signed_receipts.py:404-412`), excluding `/v1/health`, `/openapi.json`, `/docs`, `/redoc`, `/metrics`. It injects under **`_signed_receipt`** (not `signed_receipt`) and also sets 4 `X-GreenLang-Receipt-*` headers plus 3 `X-GL-Signature*` headers. **However:** the key-name mismatch means the OpenAPI-declared required field `signed_receipt` is **never** present in the actual response body. Every SDK that validates against the spec will reject the response.

### 2.5 OpenAPI drift summary
| Item | OpenAPI says | Code returns | Impact |
|---|---|---|---|
| `/v1/resolve` 200 body | `ResolvedFactorResponse = {resolved, edition_id, edition_source, signed_receipt}` | Flat `ResolvedFactor.model_dump()` + `alternates` + `explain` + middleware-injected `_signed_receipt` | **P0** — SDK validators fail; no `edition_id` inside body (only in header); `signed_receipt` key wrong |
| `/v1/factors/{id}/explain` 200 body | `ExplainResponse = {chosen_factor_id, method_profile, fallback_rank, step_label, gas_breakdown, uncertainty, alternates, explain, edition_id, signed_receipt}` | Compact `explain()` dict `{chosen, derivation, quality, uncertainty, emissions, unit_conversion, alternates, meta}` | **P0** — every top-level required field absent |
| `/v1/factors/{id}` 200 body | `FactorEnvelope` (`factors-v1.yaml:1529 = {factor, edition_id, content_hash, signed_receipt}`) | Flat resolved-factor payload from `build_factor_explain` | **P0** — same class of drift |
| `signed_receipt` schema | `{signature, algorithm, signed_at, key_id, payload_hash}` | `{signature, algorithm, signed_at, key_id, signed_over}` under wrong key `_signed_receipt` | **P0** — field `payload_hash` missing; key-name drift |
| `alternates[]` item `factor_family` | Not declared in `AlternateCandidate` | Not emitted | OK — spec already matches code here |

---

## 3. Severity tally

| Severity | Count | Elements |
|---|---|---|
| **P0** | **8** | #1 factor_family, #4 source{…}, #5 release_version, #7 valid_from/valid_to, #10 composite FQS + per-dim scores, #11 uncertainty type/low/high, #15 deprecation envelope, #16 signed_receipt key + payload_hash |
| **P1** | **6** | #2 `reason_lost` name, #3 `why_this_won` name, #6 method_pack name, #8 `_kg` suffix, #9 top-level co2e_basis, #12 licensing envelope |
| **OK** | **2** | #13 assumptions, #14 fallback_rank |

**Net:** 8 contract-breaking P0 gaps + pervasive OpenAPI drift at the envelope level. A design partner cannot validate a response against the published OpenAPI today.
