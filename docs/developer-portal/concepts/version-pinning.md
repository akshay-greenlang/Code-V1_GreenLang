# Version Pinning & Editions

An **edition** is an immutable, content-hashed snapshot of the Factors catalog. Every edition has a deterministic fingerprint, a changelog, a list of deprecations, and a set of policy-rule references it binds to. You pin to one with a single HTTP header and your downstream artefacts become exactly reproducible.

**Manifest:** `greenlang/factors/edition_manifest.py` (`EditionManifest`)
**Resolution:** `greenlang/factors/service.py::resolve_edition_id`
**Rollback:** `greenlang/factors/watch/rollback_edition.py`

---

## The header

- **Request** → `X-Factors-Edition: <edition_id>`
- **Response** → `X-GreenLang-Edition: <edition_id>` (primary) and `X-Factors-Edition: <edition_id>` (legacy, kept for backward compatibility)

Edition ids follow the pattern `vYYYY.Q<n>[-slice]`, e.g. `2027.Q1-electricity`, `2026.Q4`, `v1.0.0-certified`.

```bash
curl -H "X-Factors-Edition: 2027.Q1-electricity" \
     -H "Authorization: Bearer $GL_API_KEY" \
     "$GL_API_BASE/api/v1/factors/resolve-explain" \
     -d @request.json
```

Response headers:

```
X-GreenLang-Edition: 2027.Q1-electricity
X-Factors-Edition: 2027.Q1-electricity
X-GreenLang-Method-Profile: corporate_scope1
```

If you do not send the header, `resolve_edition_id` (in `greenlang/factors/service.py`) falls back in this order:

1. `?edition=` query parameter.
2. `GL_FACTORS_FORCE_EDITION` env var (emergency override).
3. Repository's current active default edition.

The chosen value is echoed on the response header, and the edition id is embedded into the signed-receipt payload (see [signed-receipts](./signed-receipts.md)) so tampering is detectable.

---

## What an edition manifest looks like

```json
{
  "edition_id": "2027.Q1-electricity",
  "status": "stable",
  "created_at": "2027-01-14T09:00:00Z",
  "factor_count": 4127,
  "aggregate_content_hash": "b31d9a6f7c2e...",
  "per_source_hashes": {
    "egrid": "a31d...",
    "epa_hub": "c84e...",
    "desnz_ghg_conversion": "7f19...",
    "defra_conversion": "9c03...",
    "australia_nga_factors": "1b78...",
    "japan_meti_electric_emission_factors": "44de..."
  },
  "deprecations": [
    "EF:US:electricity:2020:egrid -> superseded by EF:US:electricity:2024:egrid"
  ],
  "changelog": [
    "+ added India CEA v20 baseline (2026)",
    "+ refreshed eGRID subregion SERC",
    "! deprecated DEFRA 2022 conversions"
  ],
  "policy_rule_refs": [
    "S1..S9 release signoff",
    "CBAM Implementing Act 2023/1773 art.4"
  ]
}
```

See `EditionManifest.to_dict()` in `greenlang/factors/edition_manifest.py`.

### Deterministic fingerprint

`manifest_fingerprint()` hashes everything except `created_at`. Two identical edition bundles from two environments will share a fingerprint, so you can diff CI-built editions against the production catalog byte-for-byte.

---

## When an edition id changes

A new edition id is minted whenever **any** of the following happen:

1. A factor's numeric value, gas breakdown, uncertainty, or unit changes.
2. A new factor is added.
3. A factor is deprecated (replacement pointer added).
4. A method pack's selection rule changes (e.g. a new allowed GWP set).
5. A source release is ingested (eGRID annual, DEFRA annual, DESNZ annual, IEA annual).

The watch pipeline (`greenlang/factors/watch/pipeline.py`) drives this. Editions are promoted through three stages:

```
pending --(passes S1..S9 gate)--> stable --(hotfix fails)--> deprecated
```

See [Changelog](../migration/CHANGELOG.md) for the published edition history.

---

## How to pin

### Per-request (ad hoc)

```bash
curl -H "X-Factors-Edition: 2027.Q1-electricity" ...
```

### Per-client (recommended for production systems)

Python:

```python
from greenlang.factors.sdk.python import FactorsClient
client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-electricity",   # sent on every request
)
```

TypeScript:

```ts
const client = new FactorsClient({
  baseUrl: "https://api.greenlang.io",
  apiKey: process.env.GL_API_KEY!,
  defaultEdition: "2027.Q1-electricity",
});
```

### Env var (for emergency freezes)

```bash
export GL_FACTORS_FORCE_EDITION="2027.Q1-electricity"
```

This wins over query and header params. Use only for incident response — it affects all callers in the process.

---

## Discovering available editions

```bash
curl -H "Authorization: Bearer $GL_API_KEY" \
     "$GL_API_BASE/api/v1/editions?status=stable"
```

Response:

```json
{
  "editions": [
    {"edition_id":"2027.Q1-electricity","status":"stable","created_at":"2027-01-14T09:00:00Z","factor_count":4127},
    {"edition_id":"2026.Q4","status":"stable","created_at":"2026-10-03T08:00:00Z","factor_count":3941},
    {"edition_id":"2026.Q3","status":"deprecated","created_at":"2026-07-05T08:00:00Z","factor_count":3820}
  ]
}
```

---

## Rollback

Rollback is a **pointer flip**, never an in-place mutation. See `greenlang/factors/watch/rollback_edition.py`.

The flow:

1. An issue is detected in the current stable edition (S-gate regression, bad source ingest, legal objection).
2. Ops runs `rollback_edition.py` with a target prior edition id.
3. The tool validates the target (exists, has factors, not already active).
4. The default-edition pointer flips. The deprecated edition transitions to `deprecated` status. A `RollbackResult` record is persisted.
5. Clients with `default_edition` unset automatically use the prior stable; clients who pinned by id keep using whatever they pinned.

The API surface:

- `POST /api/v1/factors/{factor_id}/rollback/plan` — dry-run impact preview.
- `POST /api/v1/factors/{factor_id}/rollback/execute` — commit the rollback (Pro+ tier, signed approval).
- `GET  /api/v1/factors/{factor_id}/rollback/history` — audit log.
- `GET  /api/v1/factors/rollback/{rollback_id}` — single rollback record.

### When does the edition id change on you?

If you pin explicitly: never. Your client always asks for `2027.Q1-electricity`; if that edition has been demoted you will either keep getting the demoted-but-still-served bytes (safe), or receive a `410 Gone` if it has been physically purged (rare, > 2 years old).

If you do not pin: the server substitutes the currently active default. A rollback flips that default; your next call may return a different edition id. **This is why production systems should always pin.**

---

## Verifying pin integrity

Because the edition id is embedded into the signed-receipt payload ("Edition-pin into the receipt" in `middleware/signed_receipts.py`), tampering with the header or the `edition_id` field in the body invalidates the signature. See [signed-receipts](./signed-receipts.md) for the verification procedure.

---

## See also

- [Signed receipts](./signed-receipts.md)
- [Changelog](../migration/CHANGELOG.md)
- [Quality scores](./quality-scores.md) — S1..S9 gate and FQS minima per promotion tier.
- [Resolution cascade](./resolution-cascade.md)

---

## File citations

| Piece | File |
|---|---|
| `EditionManifest` dataclass | `greenlang/factors/edition_manifest.py` |
| Edition resolution order (header -> query -> env -> default) | `greenlang/factors/service.py::resolve_edition_id` |
| Rollback validator + executor | `greenlang/factors/watch/rollback_edition.py` |
| Rollback CLI | `greenlang/factors/watch/rollback_cli.py` |
| Change classification + changelog drafting | `greenlang/factors/watch/change_classification.py`, `changelog_draft.py` |
| Release orchestrator (promotion state machine) | `greenlang/factors/watch/release_orchestrator.py` |
| Cut-list anchor (v1.0 Certified) | `docs/editions/v1-certified-cutlist.md` |
