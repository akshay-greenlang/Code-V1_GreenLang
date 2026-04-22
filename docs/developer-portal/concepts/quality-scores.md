# Data Quality Score (DQS) & Composite Factor Quality Score (FQS)

Every factor carries **five** per-dimension quality scores on a 1-5 Pedigree-matrix scale, plus a composite **Factor Quality Score (FQS)** on a 0-100 scale. You use the 5 components to explain specific weaknesses ("the technology mix is from 2016"), and you use the composite to sort, filter, and gate.

**Per-factor DQS storage:** `greenlang/data/emission_factor_record.py::DataQualityScore` (1-5 scale, GHG-Protocol-aligned).
**Composite FQS + bands:** `greenlang/factors/quality/composite_fqs.py`.
**Release-signoff gate:** `greenlang/factors/quality/release_signoff.py`.

---

## The 5 components

| Internal name (storage) | CTO-spec alias (external) | 1 = worst | 5 = best | Weight in FQS |
|---|---|---|---|---|
| `temporal` | `temporal_representativeness` | > 10 years old | reporting-year match | **0.25** |
| `geographical` | `geographic_representativeness` | global default | exact sub-national match | **0.25** |
| `technological` | `technology_representativeness` | generic mix | exact activity sub-category | **0.20** |
| `representativeness` | `verification` | not reviewed | independently verified | **0.15** |
| `methodological` | `completeness` | partial methodology doc | full LCA documentation | **0.15** |

The CTO alias map (`CTO_SPEC_ALIASES` in `composite_fqs.py`) lets API responses surface both vocabularies so docs citing the spec language still read correctly.

### Why these weights?

Temporal and geographic specificity predict audit outcomes more strongly than methodology completeness. A "2019 France" factor applied to a "2026 US reporter" breaks compliance defensibility in a way that a slightly thinner methodology doc does not. Methodology is still weighted because it affects restatement risk.

Weights are tagged with `FORMULA_VERSION` (currently `fqs-1.0.0`). If weights change, the version bumps so historical responses are still interpretable.

---

## The composite (FQS, 0-100)

```
composite_fqs = round(
  (temporal      / 5 * 25) +
  (geographical  / 5 * 25) +
  (technological / 5 * 20) +
  (representativeness / 5 * 15) +
  (methodological / 5 * 15)
, 1)
```

Example: `(5/5 * 25) + (4/5 * 25) + (4/5 * 20) + (3/5 * 15) + (4/5 * 15) = 25 + 20 + 16 + 9 + 12 = 82 / 100`.

A factor's 0-100 composite is exposed on every response:

```json
"data_quality": {
  "temporal": 5,
  "geographical": 4,
  "technological": 4,
  "representativeness": 3,
  "methodological": 4,
  "overall_score": 82,
  "rating": "good",
  "formula_version": "fqs-1.0.0",
  "aliases": {
    "temporal": "temporal_representativeness",
    "geographical": "geographic_representativeness",
    "technological": "technology_representativeness",
    "representativeness": "verification",
    "methodological": "completeness"
  }
}
```

---

## Rating bands (for humans)

| Rating | FQS range | Typical use |
|---|---|---|
| `excellent` | `>= 85` | Direct use in certified submission without justification. |
| `good` | `70..84` | Direct use with a brief methodology note. |
| `fair` | `50..69` | Use for screening / Scope 3 only; flag in methodology. |
| `poor` | `< 50` | Screening only, not for disclosure. |

Constants: `RATING_BAND_EXCELLENT_MIN=85`, `RATING_BAND_GOOD_MIN=70`, `RATING_BAND_FAIR_MIN=50`. See `rating_label()` in `composite_fqs.py`.

---

## Promotion eligibility bands (for the release gate)

The release-signoff pipeline uses a different set of bands because "can a human use this" and "can it enter the Certified edition" are different policy questions.

| Promotion label | Minimum FQS | Gate |
|---|---|---|
| `certified` | `>= 75` | Can ship in a Certified edition. CBAM, PEF, `corporate_scope1`, `corporate_scope2_location_based` all require `certified`. |
| `preview` | `>= 50` | Can ship as Preview. Allowed in `corporate_scope3`, `freight_iso_14083`, `product_carbon` with methodology note. |
| `connector_only` | any | Visible only as a connector-layer response (licensed source), with source-side attribution. |

Constants: `RATING_BAND_CERTIFIED_MIN=75`, `RATING_BAND_PREVIEW_MIN=50`. See `promotion_eligibility()` in `composite_fqs.py`.

### Per-component minima (Certified edition)

In addition to the composite, each component must meet a minimum on the 1-5 scale for Certified. From `docs/editions/v1-certified-cutlist.md`:

| Component | Certified min (1-5) | Preview min (1-5) |
|---|---|---|
| temporal | 3 | 2 |
| geographical | 3 | 2 |
| technological | 3 | 2 |
| representativeness (verification) | 3 | 2 |
| methodological (completeness) | 3 | 2 |
| composite FQS | 75 / 100 | 50 / 100 |

A factor with `methodological=2` cannot enter the Certified edition even if its composite is 80.

---

## Filtering and sorting by FQS

Search:

```bash
# Only high-quality factors.
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/search?q=natural+gas+US&min_fqs=70"
```

Resolve + sort alternates by FQS:

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/EF:US:naturalgas:2024:v1/alternates?limit=10&sort=fqs_desc"
```

Quality endpoint (direct):

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/EF:US:naturalgas:2024:v1/quality" \
  | jq '.composite_fqs, .rating, .components'
```

See `greenlang/integration/api/routes/factors.py` (`/quality` endpoint, line 1177).

---

## FQS in the tie-breaker

The resolution engine uses FQS as **signal #4** in the tie-breaker (see [resolution-cascade](./resolution-cascade.md)), after geographic, temporal, and technological specificity. Two equally specific candidates at the same step are ranked by their FQS; the alternates list returned in the response shows the FQS-sorted second-place factors.

---

## Monitoring the gate

The catalog exposes `/api/v1/factors/status/summary` (three-label dashboard) which aggregates:

- Count of `certified` / `preview` / `connector_only` factors per method profile.
- Median, p25, p75 FQS per profile.
- Count of factors failing Q1..Q6 QA gates.

Useful for platform buyers monitoring the catalog before a release cutover.

---

## See also

- [Method packs](./method-packs.md) — how `allowed_statuses` interacts with promotion bands.
- [Resolution cascade](./resolution-cascade.md) — FQS as tie-breaker.
- [Version pinning](./version-pinning.md) — how S1..S9 uses promotion bands.

---

## File citations

| Piece | File |
|---|---|
| Composite FQS + rating + promotion bands | `greenlang/factors/quality/composite_fqs.py` |
| Weights (DEFAULT_WEIGHTS) | `greenlang/factors/quality/composite_fqs.py` (line 55) |
| CTO-spec aliases (`CTO_SPEC_ALIASES`) | `greenlang/factors/quality/composite_fqs.py` (line 76) |
| Rating-band constants | `greenlang/factors/quality/composite_fqs.py` (lines 89-95) |
| 1-5 DQS storage on records | `greenlang/data/emission_factor_record.py::DataQualityScore` |
| Release signoff (S1..S9) gate | `greenlang/factors/quality/release_signoff.py` |
| Q1..Q6 QA gates | `greenlang/factors/quality/batch_qa.py`, `validators.py` |
| `/quality` endpoint | `greenlang/integration/api/routes/factors.py` (line 1177) |
| Certified cutlist minima | `docs/editions/v1-certified-cutlist.md` |
