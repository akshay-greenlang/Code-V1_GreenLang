# Version Pinning & Reproducibility

Emission factor catalogs evolve — new publications, methodology updates, deprecations. To guarantee that an emissions calculation today returns the exact same answer a year from now, every request can (and in regulated contexts MUST) pin an **edition**.

## What is an edition?

An edition is an immutable snapshot of the full factor catalog, identified by a string id like `ef_2026_q1`. Each edition carries:

- A SHA-256 `manifest_hash` covering every factor record in the edition.
- A `status` (`published | pending | deprecated`).
- A changelog (`get_edition(edition_id)`).

## Three ways to pin

### 1. Per-client default (recommended for batch jobs)

```python
from greenlang.factors.sdk.python import FactorsClient

with FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_fac_...",
    default_edition="ef_2026_q1",   # sent as X-Factors-Edition on every call
) as client:
    factor = client.get_factor("EF:US:diesel:2024:v1")
```

### 2. Per-call override

```python
factor = client.get_factor("EF:US:diesel:2024:v1", edition="ef_2025_q4")
```

### 3. Header override (lower-level)

For integrations that talk to the server directly, the server also accepts `X-Factors-Edition` as a request header. The SDK does this for you when you set `default_edition`.

## Reproducibility checklist

For audit-grade reproducibility, persist alongside your calculation:

- `edition_id` (e.g. `ef_2026_q1`)
- `manifest_hash` (returned by `list_editions()` or `get_edition()`)
- The exact `method_profile`
- `sdk_version` (from `greenlang.factors.sdk.python.__version__`)
- Per-factor `content_hash` (present on Factor responses)

Together these let an auditor re-run the same calculation years later and bit-for-bit match your result.

## Comparing editions

The diff endpoint shows field-by-field changes between two editions:

```python
diff = client.diff("EF:US:diesel:2024:v1", "ef_2025_q4", "ef_2026_q1")
print(diff.status)        # "changed" | "unchanged" | "added" | "removed"
for change in diff.changes:
    print(change["field"], change.get("old_value"), "->", change.get("new_value"))
```

## Discovering available editions

```python
for e in client.list_editions(include_pending=False):
    print(e.edition_id, e.status, e.manifest_hash)
```

## Response headers carry the authoritative edition

Every successful response includes `X-Factors-Edition` (and/or `X-GreenLang-Edition` for explain routes). When using a pinned default, double-check this matches by reading the `TransportResponse.edition` attribute (surfaced via the SDK's internal transport).

## ETag + edition = safe caching

The SDK's ETag cache is keyed on `(method, url, params)`. Since the edition is passed as either a header (`default_edition`) or a query param, cached entries will not cross-contaminate between editions — you always get the version you asked for.

## Deprecations

If the server returns a factor with `factor_status == "deprecated"`, the response includes `replacement_factor_id`. Use `client.resolve_explain(...)` to see the full deprecation chain and its replacement, and subscribe to the `factor.deprecated` webhook event (see `greenlang.factors.webhooks`) to be notified when factors you pinned move to deprecated.
