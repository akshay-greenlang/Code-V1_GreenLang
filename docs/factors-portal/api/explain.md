---
title: "API: /v1/factors/{id}/explain"
description: Explain why a particular factor was chosen, with alternates and tie-break reasoning.
---

# `GET /api/v1/factors/{factor_id}/explain`

Returns the full explain payload for a factor that was previously chosen by `/resolve-explain` (or returned by any other endpoint). This is the endpoint your audit team uses to understand *why* a number is what it is.

## Authentication

`X-API-Key: <gl_fac_...>` or `Authorization: Bearer <jwt>`. **Pro+** endpoint.

## Request

```http
GET /api/v1/factors/ef:co2:natgas:us:2026/explain?method_profile=corporate_scope1&alternates=5
X-API-Key: gl_fac_...
```

| Query param        | Type    | Default | Meaning                                          |
|--------------------|---------|---------|--------------------------------------------------|
| `method_profile`   | string  | -       | Method profile under which the explain was generated. |
| `alternates`       | integer | 3       | How many next-best factors to surface (max 10).  |
| `edition`          | string  | -       | Pin to a specific edition.                       |

## Response

```json
{
  "factor": { /* CanonicalFactorRecord */ },
  "alternates": [
    { "factor": { /* ... */ }, "rank": 1, "score": 0.93 },
    { "factor": { /* ... */ }, "rank": 2, "score": 0.81 }
  ],
  "tie_break_reasons": [
    "preferred jurisdiction match",
    "newer publication year"
  ],
  "assumptions": [
    "Natural gas density assumed at standard conditions"
  ],
  "uncertainty_reasoning": [
    "DQS axes consistent across reliability + completeness",
    "Wide temporal axis (vintage 2018 vs reporting 2026) -> +5% on upper bound"
  ],
  "deprecation": null,
  "edition_id": "2027.Q1",
  "receipt": { /* signed receipt */ }
}
```

## Use cases

* **Audit defense**: drop the explain payload into your evidence package.
* **Sensitivity analysis**: pull `alternates` and re-run your inventory against each to bound the variance.
* **Methodology review**: surface `assumptions` for your verifier to inspect.

## SDK examples

```python
explain = client.resolve_explain(
    "ef:co2:natgas:us:2026",
    method_profile="corporate_scope1",
    alternates=5,
)
for alt in explain.alternates:
    print(alt.rank, alt.score, alt.factor.factor_id)
```

```ts
const explain = await client.resolveExplain("ef:co2:natgas:us:2026", {
  methodProfile: "corporate_scope1",
  alternates: 5,
});
explain.alternates.forEach((alt) =>
  console.log(alt.rank, alt.score, alt.factor.factor_id),
);
```
