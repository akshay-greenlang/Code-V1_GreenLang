---
title: "Concept: editions"
description: How GreenLang Factors versions the catalog, and how to pin to a specific edition.
---

# Editions

The Factors catalog is **versioned**. Every change -- a new factor, a corrected uncertainty band, an upgraded source vintage -- ships as a new **edition** with its own immutable id.

## Why editions matter

If you call `/resolve` today and again in six months, the catalog underneath has likely moved. Without edition pinning your second call may return a *different factor* with a *different number*. That is fine for live dashboards but unacceptable for:

* Regulated reports (CSRD, SB-253, CBAM).
* Audited inventories.
* Reproducibility-mandated workflows (academic publication, internal audit trails).

Pin an edition and the same call always returns the same factor.

## Edition id format

Editions follow one of three id shapes:

| Shape                    | Example                  | Use                                    |
|--------------------------|--------------------------|----------------------------------------|
| `vN.N.N`                 | `v1.0.0`                 | Catalog-wide semver release.           |
| `YYYY.QN`                | `2027.Q1`                | Quarterly catalog cut.                 |
| `YYYY.QN-<scope>`        | `2027.Q1-electricity`    | Quarterly cut scoped to a method pack. |
| `YYYY-MM-DD-<scope>`     | `2027-04-01-freight`     | Date-pinned scoped release.            |

The SDK validates this format **client-side** before sending the request, so a typo never reaches the server.

## Pinning in code

```python
# Python
with client.with_edition("2027.Q1-electricity") as scoped:
    resolved = scoped.resolve(request)
```

```ts
// TypeScript
await client.withEdition("2027.Q1-electricity", async (scoped) => {
  const resolved = await scoped.resolve(request);
});
```

Both the Python and TypeScript SDKs send the pin as `X-GreenLang-Edition: <edition_id>` on every request and **validate the response header** on the way back. Drift raises `EditionMismatchError`.

## Listing available editions

```python
for ed in client.list_editions(include_pending=False):
    print(ed.edition_id, ed.published_at, ed.scope)
```

## Listing changes between editions

```python
diff = client.diff(
    factor_id="ef:co2:diesel:us:2026",
    left_edition="2026.Q4",
    right_edition="2027.Q1",
)
print(diff.summary)
```

Diffs cover:

* Numeric changes (with delta + percent).
* License-class transitions.
* Uncertainty changes.
* Provenance changes (new approval, source updated).

## When to bump your pin

* **For live dashboards**: do NOT pin. Ride the default edition so users always see the freshest data.
* **For reports**: pin once at the start of a reporting period and do not move until the period closes.
* **For audited submissions**: pin to the exact edition that backed the report. Store the edition id alongside the figure in your evidence package.
