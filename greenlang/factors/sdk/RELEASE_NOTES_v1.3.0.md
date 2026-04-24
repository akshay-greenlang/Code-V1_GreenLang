# `factors-sdk` v1.3.0 — Wave 5 contract cleanup release

> Python (`greenlang-factors`) and TypeScript (`@greenlang/factors`)
> **publish together** under the same semver.

## Headline

Wave 5 closes the three contract ambiguities flagged during Wave 4
customer integrations:

1. `uncertainty` no longer ambiguous between absolute and percentage forms.
2. `deprecation_status` is now always an object — no more `string | object`
   union to defend against in caller code.
3. `/v1/method-packs/coverage` returns one canonical shape regardless of
   whether `?pack=<slug>` is supplied.

All three changes are **additive on the wire and backward-compatible in
the SDK** — older clients keep working, and the SDK normalises legacy
shapes for you with a single-release deprecation window.

## What's new vs 1.2.0

### 1. Disambiguated uncertainty fields

`UncertaintyEnvelope` (Python) / `UncertaintyEnvelope` (TS) now exposes
both forms side-by-side:

| Field | Meaning |
|---|---|
| `uncertainty` | Absolute uncertainty in the factor's native unit (e.g. kg CO2e per activity unit). |
| `uncertainty_percent` | Relative uncertainty as a percentage (e.g. `5.0` = 5 %). |

```python
resolved = client.resolve(request)
abs_unc = resolved.uncertainty.uncertainty
pct_unc = resolved.uncertainty.uncertainty_percent
# Use whichever your downstream tooling expects.
```

```ts
const u = resolved.uncertainty;
const abs = u.uncertainty;            // native unit
const pct = u.uncertainty_percent;    // percent
```

The resolver emits both whenever it can compute them. If a source only
publishes one form, the other is `null`. Old SDK callers reading the
single `uncertainty` field continue to work — that field has not changed
shape or meaning.

### 2. `deprecation_status` always an object

Wire-level shapes the SDK now accepts on input:

```jsonc
// Legacy bare string (Wave 1–4 server may still emit this for active records)
"deprecation_status": "active"

// Wave 5 canonical object
"deprecation_status": {
  "status": "deprecated",
  "successor_id": "EF:...",
  "reason": "Replaced by AR6 GWP set",
  "deprecated_at": "2026-04-01T00:00:00Z"
}
```

What you read out of the SDK is **always** the typed object form:

```python
ds = resolved.deprecation_status   # DeprecationStatus | None
print(ds.status, ds.successor_id, ds.reason, ds.deprecated_at)
```

```ts
const ds = resolved.deprecation_status;  // DeprecationStatusEnvelope | null
console.log(ds?.status, ds?.successor_id);
```

If the wire payload is a bare string `s`, the SDK inflates it to
`{status: s, successor_id: null, reason: null, deprecated_at: null}`. No
client-side `isinstance` / `typeof` branching needed.

If your code did the branching workaround documented in v1.2 release
notes, you can delete it.

### 3. `/v1/method-packs/coverage` — one canonical shape

Old behaviour: passing `?pack=<slug>` returned a single-pack object;
omitting it returned an array. Wave 5 returns the same envelope either
way:

```jsonc
{
  "packs": [
    {
      "slug": "corporate",
      "version": "0.2.0",
      "total_activities": 412,
      "covered": 387,
      "fraction": 0.939,
      "by_family": { ... },
      "by_jurisdiction": { ... }
    }
  ],
  "overall": {
    "total_activities": 412,
    "covered": 387,
    "fraction": 0.939,
    "by_family": { ... },
    "by_jurisdiction": { ... }
  }
}
```

Single-pack request still filters `packs[]` to one entry; `overall`
mirrors that pack's numbers when only one is requested. Multi-pack
request fills `packs[]` with all matches and `overall` with the
aggregate.

```python
cov = client.method_pack_coverage(slug="corporate")
my_pack = cov.packs[0]
print(my_pack.fraction, my_pack.by_jurisdiction["IN"])
```

```ts
const cov = await client.methodPackCoverage({ slug: 'corporate' });
console.log(cov.packs[0].fraction);
```

The SDK accepts the legacy single-object response from older servers and
normalises it into this shape — no migration required if you point a new
SDK at an older API.

## Migration guide

| You did this in v1.2 | You can do this in v1.3 |
|---|---|
| `if isinstance(ds, str): ...` | Delete the branch — `ds` is always `DeprecationStatus`. |
| Read `.uncertainty` and assume "%" | Read `.uncertainty_percent` instead, or keep `.uncertainty` if absolute is what you wanted. |
| Branch on `coverage` response shape | One shape only — `cov.packs[*]` + `cov.overall`. |
| Everything else | No changes required. |

## Install

```bash
pip install -U greenlang-factors==1.3.0
# or
npm install @greenlang/factors@1.3.0
```

## Compatibility

| | Supported |
|---|---|
| **API server** | factors-api with Wave 1–5 patches. SDK v1.3 also normalises Wave 1–4 server responses. |
| **Python** | 3.10, 3.11, 3.12, 3.13 |
| **Node** | 18, 20, 22 |
| **Auth** | JWT or API-Key |

## Tag + publish

```bash
git tag factors-sdk-v1.3.0
git push origin factors-sdk-v1.3.0
```

Tag push fires `.github/workflows/sdk_release.yml` (OIDC trusted-publisher),
which:

1. Validates version references match the tag.
2. Runs `pytest tests/factors/sdk/` and `npm test` in
   `greenlang/factors/sdk/ts/`.
3. Builds + publishes Python (PyPI) and TypeScript (npm).
4. Creates a GitHub Release with this file as the body.

---

*Cut: 2026-04-24. Backward-compatible with SDK 1.2.0 callers and with
Wave 1–4 servers. No deprecations introduced; this release retires the
v1.2-era `isinstance(ds, str)` workaround in favour of unconditional
typed access.*
