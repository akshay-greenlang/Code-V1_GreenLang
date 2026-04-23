# Resolver Contract — Code Patch Plan (Wave 2)

**Date:** 2026-04-23
**Scope:** Concrete code-change recipes for every P0 gap identified in `resolver_contract_audit.md`. **Planning artefact only — no production code is modified in Wave 1.**
**CTO tasks covered:** R4 (explain contract hardening), R6 (FQS 0-100 surface on resolve), API2 (signed-receipt key rename + OpenAPI drift).

Conventions:
- Every plan item names: target **file:line**, target **function**, **what to add**, **LOC estimate**, **risk**, and **sequencing**.
- All risks are on a 3-level scale: **low** (pure addition, no branch change), **medium** (touches cascade + serialization), **high** (affects tenant overlay / signing / middleware wiring).

---

## 1. P0 gaps — per-gap plan

### GAP-1 — `chosen_factor {id, name, version, factor_family}` envelope
- **Source files:**
  - `greenlang/factors/resolution/result.py:49-97` — add field `chosen_factor: ChosenFactor` to `ResolvedFactor`.
  - `greenlang/factors/resolution/engine.py:295-367` — populate it in `_build_resolved_factor` (read `record.factor_family`).
  - `greenlang/factors/api_endpoints.py:650-728` — ensure `build_factor_explain` / `build_resolution_explain` surface it (pydantic round-trips).
- **New helper:** none required.
- **New pydantic model:** `ChosenFactor(GreenLangBase)` — 4 fields.
- **LOC estimate:** **~40** (20 in result.py + 15 in engine.py + 5 in api_endpoints.py).
- **Risk:** **low** — pure projection from existing fields. `factor_family` already lives on `EmissionFactorRecord` via `greenlang.schemas.enums.FactorFamily`.
- **Sequence:** blocks GAP-10 (FQS display) and GAP-15 (deprecation envelope), because all three are projected inside `_build_resolved_factor`. Do first.

### GAP-4 — `source {id, version, name, authority}` envelope
- **Source files:**
  - `greenlang/factors/resolution/result.py:56-60` — replace flat `source_id`/`source_version` with `source: SourceDescriptor` (keep flat fields in parallel for deprecation window).
  - `greenlang/factors/resolution/engine.py:331-342` — extend the projector to produce a full `SourceDescriptor` (read `provenance.source_org` for `authority`, `provenance.source_publication` for `name`, `record.source_id` for `id`).
- **New pydantic model:** `SourceDescriptor(GreenLangBase)` — 4 required fields.
- **LOC estimate:** **~35**.
- **Risk:** **low**. The projector already walks `provenance`; this is a restructure, not new I/O.
- **Sequence:** independent — can ship parallel to GAP-1.

### GAP-5 — distinct `factor_version` AND `release_version`
- **Source files:**
  - `greenlang/factors/resolution/result.py:60` — add `release_version: Optional[str]` as a first-class field.
  - `greenlang/factors/resolution/engine.py:337-342` — stop collapsing `release_version` into `source_version`; project each separately.
  - `greenlang/factors/catalog_repository.py` (verify-only) — ensure `EmissionFactorRecord.release_version` is populated on every load path (Postgres + in-memory). Already exists; confirm migration guarantees non-null on Certified rows.
- **New helper:** none.
- **LOC estimate:** **~15**.
- **Risk:** **low** — restructure only.
- **Sequence:** independent. Ship with GAP-4.

### GAP-7 — `valid_from` / `valid_to` projection
- **Source files:**
  - `greenlang/factors/resolution/result.py:49-97` — add `valid_from: date` and `valid_to: Optional[date]`.
  - `greenlang/factors/resolution/engine.py:343-367` — read `record.valid_from`, `record.valid_to` (guaranteed present on `EmissionFactorRecord`, see `factors-v1.yaml:1421-1425`).
- **LOC estimate:** **~10**.
- **Risk:** **low**. Pure projection.
- **Sequence:** independent. Bundle with GAP-1/4/5 in a single "projection pass" PR.

### GAP-10 — composite FQS 0-100 + 5 per-dimension scores
- **Source files:**
  - `greenlang/factors/resolution/result.py:77` — replace scalar `quality_score: Optional[float]` with `quality: QualityEnvelope` (keep scalar for deprecation window).
  - `greenlang/factors/resolution/engine.py:317-319` — stop copying `dqs.overall_score`; call `greenlang.factors.quality.composite_fqs.compute_fqs(record.dqs)` and project its `FQSResult` into `QualityEnvelope`.
  - `greenlang/factors/quality/composite_fqs.py` (verify-only) — confirm existing `compute_fqs()` returns per-dimension 0-100 scores; it already does (weighted rescale at `composite_fqs.py:99+`).
- **New pydantic model:** `QualityEnvelope(GreenLangBase)` — 6 required 0-100 fields + rating + formula_version + weights.
- **New helper / module:** **none** — `composite_fqs.py` already exists; just wire the call.
- **LOC estimate:** **~60** (new pydantic model 25 + engine call 15 + projector 20).
- **Risk:** **medium** — this touches the serialized payload for every single `/resolve` call. Extensive unit coverage required for CI (golden payloads).
- **Sequence:** ships after GAP-1 (both modify `_build_resolved_factor`). Independent of middleware / signing work.

### GAP-11 — `uncertainty {type, low, high, distribution}` with numeric bounds
- **Source files:**
  - `greenlang/factors/resolution/result.py:28-35` — extend `UncertaintyBand` with `type: Literal["95_percent_ci", "qualitative"]`; upgrade `low` / `high` from existing-but-unused fields to populated numerics.
  - `greenlang/factors/resolution/engine.py:311-315` — call a new helper to derive `low`, `high` from `ci_95_percent` + `co2e_total_kg`.
  - **New helper:** `greenlang/factors/resolution/uncertainty_propagation.py` — 1 function `derive_bounds(ci_95_percent: Optional[float], central: float, distribution: str) -> Tuple[Optional[float], Optional[float]]`. Handles lognormal vs. normal vs. uniform heuristically. Default for unknown distribution is `±ci_95_percent * central`.
- **LOC estimate:** **~80** (helper ~40, engine hookup 15, result.py 15, model tests 10).
- **Risk:** **medium** — numeric formula has to match the audit defensibility docs. Reviewer: CTO + DQ lead.
- **Sequence:** after GAP-10 (shares the serialization payload); before GAP-16 because the expanded payload changes `payload_hash`.

### GAP-15 — `deprecation_status` envelope with `"active"` default and `replacement_pointer_factor_id`
- **Source files:**
  - `greenlang/factors/resolution/result.py:72-73` — replace nullable scalar `deprecation_status` + `deprecation_replacement` with `deprecation_status: DeprecationStatus` (always present; default `status="active"`).
  - `greenlang/factors/resolution/engine.py:357-358` — branch on `record.factor_status`: certified/preview → `active`; deprecated → `deprecated` + `replacement_pointer_factor_id = record.replacement_factor_id`; connector_only + superseded edition → `superseded`.
  - **New module:** `greenlang/factors/resolution/deprecation_pointer.py` — 1 function `resolve_deprecation_status(record, edition_id, repo) -> DeprecationStatus`. Encapsulates the 3-branch logic and provides a stable import path for tests + CLI.
- **LOC estimate:** **~100** (new module ~60, engine wiring 20, pydantic model 20).
- **Risk:** **medium** — introduces a new `"superseded"` status that the UI, SDK, and release pipeline don't yet consume. Coordination with the front-end team required.
- **Sequence:** after GAP-1 / GAP-10 (all three mutate `_build_resolved_factor`); before Wave-2 frontend work on the deprecation banner.

### GAP-16 — `signed_receipt` envelope (rename from `_signed_receipt`; add `receipt_id`, `verification_key_hint`, `payload_hash`; rename `algorithm` → `alg`)
- **Source files:**
  - `greenlang/factors/middleware/signed_receipts.py:218-233` — change the injected key from `"_signed_receipt"` to `"signed_receipt"`.
  - `greenlang/factors/middleware/signed_receipts.py:222-232` — extend the injected dict with `receipt_id` (uuid7), `verification_key_hint` (mirror of `key_id`), `alg` (mirror of `algorithm`), `payload_hash` (mirror of `body_hash`).
  - `greenlang/factors/signing.py` — add helper `mint_receipt_id() -> str` returning UUIDv7; extend `Receipt.to_dict()` with the 4 new keys (keep the old keys as aliases for one release).
  - `greenlang/factors/middleware/signed_receipts.py:395-471` — same changes in `SignedReceiptsMiddleware.dispatch` (two copies exist in this file — update both).
  - `greenlang/factors/sdk/python/client.py` & `greenlang/factors/sdk/ts/src/client.ts` — teach the SDK to prefer `signed_receipt` over `_signed_receipt`.
- **LOC estimate:** **~120** (middleware 50, signing.py 30, two SDKs 40).
- **Risk:** **high** — this is the most visible client-facing change. Every design-partner integration verifies receipts; a silent key rename breaks every downstream consumer. **Deprecation plan:** emit BOTH `_signed_receipt` and `signed_receipt` for one release cycle; SDKs prefer the new name.
- **Sequence:** ship **last** because the receipt signs the whole body — all other gap fixes must be in place before the final payload hash is signed.

---

## 2. P1 gaps — short-form

| Gap | File/function | Change | LOC | Risk |
|---|---|---|---|---|
| #2 `reason_lost` alias | `result.py:38-46` `AlternateCandidate` | Add `reason_lost: Optional[str]` populated = `why_not_chosen` | 5 | low |
| #3 `why_this_won` alias | `result.py:49-97` `ResolvedFactor` | Add field mirroring `why_chosen` | 5 | low |
| #6 `method_pack` name | `result.py:96` | Add `method_pack: Optional[str]` populated from `pack.pack_id` | 10 | low |
| #8 `_kg` suffix aliases | `result.py:13-25` `GasBreakdown` | Add computed properties `co2`, `ch4`, ... returning `*_kg` | 20 | low |
| #9 `co2e_basis` top-level | `result.py:49-97` | Derive from `gas_breakdown.gwp_basis` on model-dump | 10 | low |
| #12 `licensing` envelope | `result.py:49-97` | Add `licensing: LicensingEnvelope` projected from `record.licensing` | 15 | low |

All P1 gaps can land in a single "cosmetic-aliases" PR after P0 is merged. No risk to production.

---

## 3. Route-level changes

### 3a. `/v1/resolve` return-shape (envelope wrap)
- **File:** `greenlang/factors/api_v1_routes.py:119-141`.
- **Change:** wrap `payload` returned by `build_resolution_explain` into a `ResolveResponseV2` envelope:
  ```
  return {
      "resolved": payload_minus_explain,
      "edition_id": edition,
      "edition_source": _source_label(request),
      "signed_receipt": None,   # middleware fills it
      "explain": payload["explain"] if not compact else None,
  }
  ```
- **LOC:** ~30.
- **Risk:** **medium** — every existing client receives the new nested shape. Publish a `?legacy_shape=true` flag for one release as escape hatch.

### 3b. `/v1/factors/{id}/explain` return-shape (envelope + top-level fields)
- **File:** `api_v1_routes.py:169-187`.
- **Change:** stop returning `payload.get("explain", payload)`; instead build an `ExplainResponseV2` envelope with `chosen_factor`, `method_pack(+_version)`, `fallback_rank`, `step_label`, `gas_breakdown`, `uncertainty`, `alternates`, `explain`, `edition_id`, `signed_receipt`.
- **LOC:** ~50.
- **Risk:** **medium** — same deprecation concerns; use `?legacy_shape=true`.

### 3c. `/v1/factors/{factor_id}` return-shape (envelope)
- **File:** `api_v1_routes.py:144-166`.
- **Change:** return a `FactorEnvelope` (`{factor, edition_id, content_hash, signed_receipt}`) so it matches `factors-v1.yaml:1529`.
- **LOC:** ~40.
- **Risk:** **medium** — same class of drift as 3a/3b.

---

## 4. New modules / helpers summary

| New file | Purpose | LOC |
|---|---|---|
| `greenlang/factors/resolution/deprecation_pointer.py` | Classifies factor status (active / deprecated / superseded) and produces the `DeprecationStatus` envelope with the replacement pointer. | 60 |
| `greenlang/factors/resolution/uncertainty_propagation.py` | Derives numeric `low` / `high` bounds from `ci_95_percent` + central estimate based on declared distribution. | 40 |
| `greenlang/factors/resolution/envelope_projection.py` | Central place for projecting `ChosenFactor`, `SourceDescriptor`, `LicensingEnvelope`, `QualityEnvelope`, `Co2eBasis` from a `record` + `pack` + `resolved` triple. Keeps `engine._build_resolved_factor` readable (it's approaching 70 lines already). | 120 |

**Total new-module LOC:** ~220. Each module owns 3-5 unit tests (see §5).

---

## 5. Test plan

All new tests under `tests/factors/` (existing structure). Filenames below are **new files** unless marked (+add-cases).

### 5a. `tests/factors/api/`
| File | What it asserts |
|---|---|
| `test_resolve_contract_16_fields.py` | Canonical demo: `POST /v1/resolve` with body `{activity:"electricity", jurisdiction:"IN", method_profile:"corporate_scope2_location_based", quantity:12500, unit:"kWh"}` — response contains all 16 required elements at the documented paths. One parametrized test per element + one all-at-once structural test. |
| `test_explain_contract_16_fields.py` | Same 16-element check for `GET /v1/factors/{id}/explain`. |
| `test_resolve_compact_mode.py` | `?compact=true` drops `explain` but keeps `signed_receipt` + `edition_id`. |
| `test_resolve_envelope_shape.py` (+add-cases) | `resolved` + `edition_id` + `edition_source` + `signed_receipt` siblings present; no top-level leakage of internal fields. |
| `test_signed_receipt_rename.py` | Payload contains `signed_receipt` (new) AND `_signed_receipt` (legacy) during the deprecation window; they carry identical signatures. |
| `test_signed_receipt_fields.py` | `receipt_id` is a UUIDv7; `verification_key_hint` matches env-configured hint; `payload_hash` is hex SHA-256 of body-excluding-receipt; `alg ∈ {sha256-hmac, ed25519}`. |
| `test_openapi_response_validity.py` | Every 2xx response from canonical calls validates against the updated `factors-v1.yaml` (use `openapi-schema-validator`). Regression barrier against future drift. |
| `test_edition_pin_respected.py` (+add-cases) | `X-GL-Edition` on request echoes as `X-GreenLang-Edition` on response AND ends up inside `signed_receipt.signed_over.edition_id`. |

### 5b. `tests/factors/resolution/`
| File | What it asserts |
|---|---|
| `test_chosen_factor_envelope.py` | `ChosenFactor{id,name,version,factor_family}` is fully populated for all 7 method packs (electricity, combustion, freight, material, land, product, finance). |
| `test_source_descriptor.py` | `source.authority` is non-empty for every Certified source in the built-in DB. |
| `test_release_version_vs_factor_version.py` | The two fields produce distinct values for ≥3 records where the catalog release was bumped without bumping the factor. |
| `test_quality_envelope.py` | `composite_fqs_0_100` equals `compute_fqs(record.dqs).score` for a sample of 100 records; per-dimension scores are linearly rescaled from the 1-5 DQS (score 3 → 60). |
| `test_uncertainty_propagation.py` | For lognormal distribution, `derive_bounds(0.05, central=1.0, "lognormal")` produces `(≈0.905, ≈1.105)`; for unknown distribution, bounds are exactly `±0.05`; qualitative returns `(None, None)`. |
| `test_deprecation_pointer.py` | Active factor → `status="active"`, `replacement_pointer_factor_id=None`. Deprecated factor → `status="deprecated"`, pointer populated. Superseded edition → `status="superseded"`. |
| `test_valid_dates_projection.py` | `valid_from` and `valid_to` are correctly projected and serialized as ISO dates. |
| `test_resolution_backcompat.py` | The old flat `chosen_factor_id`, `source_id`, `why_chosen`, `quality_score` fields are STILL present on v2 responses (deprecation window check). |

### 5c. Golden-payload tests
- `tests/factors/api/golden/canonical_demo_resolve.json` — complete expected response for the canonical demo (India, 12,500 kWh, scope 2 location-based). Pinned at Wave 2 ship.
- `tests/factors/api/golden/canonical_demo_explain.json` — same factor via `/v1/factors/{id}/explain`.
- `test_golden_canonical_demo.py` — diff the live response against the golden, allowing only `resolved_at` / `signed_at` / `signature` to differ.

---

## 6. Risk + LOC totals

| Bucket | LOC | Risk profile |
|---|---|---|
| New pydantic models (ChosenFactor, SourceDescriptor, QualityEnvelope, UncertaintyEnvelope, LicensingEnvelope, DeprecationStatus, Co2eBasis, ResolvedFactorV2) | ~260 | low |
| Engine projection rewiring (`_build_resolved_factor`) | ~180 | medium |
| New helper modules (deprecation_pointer, uncertainty_propagation, envelope_projection) | ~220 | low-medium |
| Signing middleware + signing.py (key rename + 4 new fields) | ~120 | high |
| Route envelope wrapping (`/v1/resolve`, `/v1/factors/{id}`, `/v1/factors/{id}/explain`) | ~120 | medium |
| SDK updates (Python + TS) | ~80 | medium |
| Tests (api + resolution + golden) | ~1200 | — |
| **Total production LOC** | **~980** | — |
| **Total production + tests** | **~2180** | — |

---

## 7. Sequencing (ship order)

1. **Preparatory PR (low risk, no behavior change):**
   - Add all new pydantic models + `envelope_projection.py`.
   - Add `deprecation_pointer.py`, `uncertainty_propagation.py`.
   - Extend `factors-v1.yaml` per `resolver_contract_openapi_patch.yaml` (additive only).
   - **Unblocks:** everything below.

2. **Projection PR (medium risk):**
   - Rewire `_build_resolved_factor` to populate all new envelopes (GAP-1, 4, 5, 7, 10, 11, 12, 15 + P1 #6, #9, #12).
   - Keep every legacy scalar for backwards-compat.
   - Lands the 16-field coverage for `/v1/resolve` and `/v1/factors/{id}/explain` internally.

3. **Route-envelope PR (medium risk):**
   - Switch `/v1/resolve`, `/v1/factors/{id}`, `/v1/factors/{id}/explain` to the new OpenAPI response envelopes.
   - Add `?legacy_shape=true` escape hatch.
   - Flip OpenAPI `paths_updates` refs.

4. **Signing PR (high risk):**
   - Rename `_signed_receipt` → `signed_receipt`; emit both for one release.
   - Add `receipt_id`, `verification_key_hint`, `payload_hash`, `alg`.
   - SDK client updates.

5. **SDK + docs + FE consumers (parallel with PR 3 and 4).**

6. **Deprecation-window cleanup PR (next release):**
   - Remove `_signed_receipt`, `why_chosen`, `source_id` scalars, etc.
   - Remove `?legacy_shape=true`.

---

## 8. What this plan does NOT change

- `EmissionFactorRecord` on-disk schema — **not touched**.
- Database migrations — none required; every new surface is derived from existing record fields.
- Signing algorithm — HMAC + Ed25519 stay unchanged.
- Route paths and auth model — unchanged.
- Cascade order — 7-step remains per `engine.py:375-383`.
- `/v1/resolve` default `compact` behaviour — still returns `explain` by default; `?compact=true` still suppresses it (CTO non-negotiable #3 preserved).

---

## 9. Canonical demo readiness after Wave 2

After the four PRs above land, the canonical demo (resolve 12,500 kWh India FY27 `corporate_scope2_location_based`) returns a response that:

- Validates against the updated `factors-v1.yaml` schema.
- Contains all 16 required fields at the documented paths (see `test_resolve_contract_16_fields.py`).
- Carries an `X-GreenLang-Edition` header AND a `signed_receipt.signed_over.edition_id` fold-in.
- Is reproducibility-verifiable offline via the SDK's `verify_receipt()` helper.

---

**End of plan.** Wave 2 implementers: start at §7, step 1.
