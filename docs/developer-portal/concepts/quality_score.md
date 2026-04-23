# Concept — Quality Score (FQS)

The **Factor Quality Score (FQS)** is a 0-100 composite that summarises how trustworthy a factor is for a given use. FQS is derived from five integer component scores (1-5) using GHG Protocol Product Standard conventions. The composite is stored on the record and returned on every resolution.

## The five components

| Component | Weight | What it measures |
|---|-----:|---|
| `temporal_score` | 0.25 | How close the factor's `valid_from..valid_to` is to the activity's reporting period. 5 = overlaps, 1 = more than 10 years off. |
| `geographic_score` | 0.25 | How well the factor's jurisdiction matches the activity. 5 = exact facility / grid-subregion match, 1 = global default. |
| `technology_score` | 0.20 | How well the factor represents the specific technology / process. 5 = primary supplier data, 1 = broad industry proxy. |
| `verification_score` | 0.15 | Degree of external / third-party verification. 5 = regulator-approved, 4 = third-party verified, 3 = publisher-verified, 1 = unverified. |
| `completeness_score` | 0.15 | Fraction of relevant gases and sub-processes captured. 5 = all in-scope gases including biogenic and F-gases, 1 = CO2 only. |

## The composite

```
composite_fqs = 20 * (0.25*temporal + 0.25*geographic + 0.20*technology + 0.15*verification + 0.15*completeness)
```

Examples:

- All 5's → `20 * 5.00 = 100`
- All 1's → `20 * 1.00 = 20`
- `(5, 5, 4, 3, 4)` → `20 * 4.35 = 87.0`

The schema invariant is `|composite_fqs - (weighted formula)| <= 0.5`. The writer recomputes on every mutation.

## How FQS is surfaced

`/resolve` and `/explain` return the composite inside `quality.composite_fqs_0_100` and the five components inside `quality.components`. The SDKs expose both as typed fields on `ResolvedFactor.quality`. The three-label dashboard in the Operator Console groups factors into high (>=80), medium (60-79), and low (<60) bands.

## What FQS is NOT

- **Not a pass/fail gate.** A factor with FQS 40 is allowed in resolution; the pack's `stale_factor_cutoff_days` and `require_verification` rules gate by stricter criteria.
- **Not comparable across families.** An FQS 80 combustion factor and an FQS 80 spend-proxy factor are not equally authoritative for a given activity; the component scores are what auditors inspect.
- **Not a substitute for uncertainty.** `parameters.uncertainty_low` / `uncertainty_high` carry the 95% CI; FQS summarises methodological quality.

## Use in regulatory packs

- **CBAM** requires `temporal_score >= 4` and `geographic_score >= 4` for iron/steel, aluminium, cement, fertilizer, electricity, hydrogen.
- **PACT** requires `primary_data_share >= 0.5`, which drives `technology_score >= 4`.
- **PCAF** scales its own "data quality score" (Score 1-5) onto `verification_score` and `technology_score`; see [PCAF pack](../method-packs/finance_proxy.md).

**See also:** [`factor`](factor.md), [API `/quality`](../api-reference/quality.md), [`schema.md`](../schema.md).
