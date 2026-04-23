# GreenLang Factors - Public Gold-Label Evaluation Set

This directory holds the **gold-label evaluation set** used to measure how
well the factor matching + resolution pipeline picks the right emission
factor for a given activity.

It backs Track B-1 of `FY27_Factors_Launch_Checklist.md` ("Public gold-label
evaluation set, >=300 representative activities"), plus Track T3/T4 of
`GreenLang_Factors_CTO_Master_ToDo.md` (the precision/recall CI gate), and
is the data the `factors-gold-eval` GitHub Actions job consumes on every PR
that touches `greenlang/factors/**`.

---

## Layout

```
gold_set/
├── README.md                         # this file
└── v1/
    ├── index.json                    # manifest (schema + case_count per file)
    ├── electricity.json              # 60 cases  (v1.0)
    ├── fuel_combustion.json          # 50 cases  (v1.0)
    ├── refrigerants.json             # 32 cases  (v1.0)
    ├── refrigerants_expanded.json    #  8 cases  (v1.1) - HFO/R-32/R-1234yf
    ├── freight.json                  # 50 cases  (v1.0)
    ├── materials.json                # 50 cases  (v1.0)
    ├── cbam.json                     # 35 cases  (v1.0)
    ├── methodology_profiles.json     # 25 cases  (v1.0) - cross-profile routing
    ├── purchased_goods_proxy.json    # 40 cases  (v1.1) - spend-based screening
    └── waste.json                    # 20 cases  (v1.1) - landfill/incin/recycle/compost
```

Total: **370 cases** (>= 300 acceptance bar, under the 500 upper bound).

The files split into two schema generations — both are loaded by the CI gate
and normalized to the same internal `GoldEntry` shape.

---

## Case schemas

### Schema v1.1 (new, authored 2026-04-23 for the T3/T4 CI gate)

```json
{
  "id": "gs_waste_US_2026_001",
  "activity_description": "Mixed municipal solid waste disposed to US sanitary landfill with LFG collection",
  "amount": 185,
  "unit": "tonne",
  "expected_method_profile": "corporate_scope3",
  "expected_factor_family": "waste_treatment",
  "expected_jurisdiction": {"country": "US", "region": null},
  "expected_factor_id_pattern": "EF:EPA:waste_landfill_msw.*:US:.*",
  "tier_acceptance": ["primary", "alternate_top3"],
  "notes": "optional"
}
```

| Field | Description |
|---|---|
| `id` | Globally unique slug. Convention: `gs_<family>_<jurisdiction>_<year>_<nnn>`. |
| `activity_description` | Free-text activity. Fed to `ResolutionRequest.activity`. |
| `amount` / `unit` | Activity quantity + canonical unit. |
| `expected_method_profile` | Required. One of the `MethodProfile` enum values. |
| `expected_factor_family` | One of the `FactorFamily` enum values. |
| `expected_jurisdiction.country` | ISO 3166-1 alpha-2 (`US`, `IN`, `EU`, `UK`, ...). |
| `expected_jurisdiction.region` | Sub-national code (`US-CA`, `IN-KA`, ...); nullable. |
| `expected_factor_id_pattern` | **Python regex** that a returned `factor_id` must match to count as a hit. Use an exact-anchor pattern (`^...$`) when the factor is already seeded; use a family-level pattern (`EF:EPA:waste_.*:US:.*`) when only the bucket is known. |
| `tier_acceptance` | What counts as a hit. `primary` = chosen factor matches; `alternate_top3` = one of the top-3 alternates matches. Most entries allow both. |
| `notes` | Optional prose context / caveat. |

### Schema v1.0 (legacy, pre-dates T3/T4)

```json
{
  "case_id": "elec_in_northern_grid_2027_lb_002",
  "activity": {
    "description": "...", "quantity": 8200, "unit": "kWh",
    "metadata": {"country": "IN", "grid_region": "northern_grid", ...}
  },
  "method_profile": "corporate_scope2_location_based",
  "expected": {
    "factor_id": "EF:IN:northern_grid:2026-27:cea-v20.0",
    "factor_family": "grid_intensity",
    "source_authority": "CEA",
    "fallback_rank": 4,
    "co2e_per_unit_min": 0.50,
    "co2e_per_unit_max": 0.56,
    "co2e_unit": "kgCO2e/kWh",
    "must_include_assumptions": ["grid average", "location-based", "CEA"]
  },
  "tags": [...]
}
```

v1.0 entries retain their original semantics — when `expected.factor_id`
is `null`, the loader synthesizes a permissive family/subject-level pattern
so the entry still participates in the P@1 / R@3 aggregate without
blanket-failing for reasons unrelated to resolver quality.

---

## Coverage per family

| Family | Count | Schema | Coverage target |
|---|---:|:---:|---:|
| electricity | 60 | v1.0 | 80 (target), 60 (min) |
| fuel_combustion | 50 | v1.0 | 60 (target), 50 (min) |
| refrigerants | 40 | v1.0 + v1.1 | 40 (min) |
| freight | 50 | v1.0 | 60 (target), 50 (min) |
| materials | 50 | v1.0 | 40 (min) |
| waste | 20 | v1.1 | 20 (min) |
| purchased_goods_proxy | 40 | v1.1 | 40 (min) |
| cbam | 35 | v1.0 | (cross-cut of materials) |
| methodology_profiles | 25 | v1.0 | (cross-cut) |
| **TOTAL** | **370** | | **>=300** |

Per-family minimums are enforced by
`tests/factors/matching/test_gold_eval_gate.py::test_minimum_coverage_per_family`.

---

## Authenticity rules

1. **Use real industrial activity descriptions.** Acceptable anchor
   sources for paraphrase: CDP disclosures, IEA World Energy Outlook
   worked examples, DEFRA/DESNZ GHG Conversion Factors usage notes,
   PCAF Global GHG Accounting Standard case studies, EU CBAM transitional
   guidance. Verbatim copying is NOT required — paraphrase plausible
   real-world scenarios.

2. **Do NOT invent obscure scenarios just to hit a count.** Each entry
   must be independently resolvable (a reasonable method-profile exists
   and an authoritative factor source can plausibly produce the answer).

3. **Mix clean, ambiguous, and edge-case descriptions** to stress the
   resolver. Examples of intentional edge cases we include:
   - HFO refrigerant blends (R-1234yf, R-1234ze, R-452A) — catalog may not
     index every blend composition; we flag these so methodology can seed
     them.
   - Indian unmanaged dumpsites (IPCC default MCF 0.6).
   - CBAM goods with non-primary origin (Korea, Turkey, Ukraine).
   - Spend-based screening that intentionally overlaps with EPD-available
     products — a good resolver should still pick the spend-proxy factor
     when the `finance_proxy` profile is requested.

4. **`expected_factor_id_pattern` must map to a real factor-id convention.**
   Legal prefixes include `EF:CBAM:*`, `EF:IPCC:*`, `EF:IEA:*`,
   `EF:DESNZ:*`, `EF:EPA:*`, `EF:CEA:*`, `EF:EEIO:*`, `EF:EXIOBASE:*`,
   `EF:DEFRA_IO:*`, `EF:GLEC:*`, `EF:AIB:*`, `EF:EEA:*`. Cross-reference
   against `greenlang/factors/ingestion/parsers/` when adding new
   patterns.

5. **Honest reporting over coverage theatre.** The CI gate will fail
   loudly when accuracy is below the floor. Do NOT adjust patterns to
   match whatever the resolver currently returns — that hides the bug.

---

## Acceptance bars

| Gate | Threshold | Source of truth |
|---|---|---|
| Overall Precision@1 | `>= 0.85` | `test_overall_precision_at_1_above_floor` |
| Overall Recall@3 | `>= 0.95` | `test_overall_recall_at_3_above_floor` |
| Per-family coverage min | see table above | `test_minimum_coverage_per_family` |
| Total size | `300 <= n <= 1000` | `test_total_entries_within_bounds` |
| Per-family P@1 (legacy v1.0) | `>= 0.85` | `tests/factors/test_gold_set_eval.py` |
| Delta vs `main` baseline | drop `<= 2 pp` | CI workflow `delta` step |

A PR that drops overall P@1 by more than 2 pp vs the most recent
successful `main` run blocks merge.

---

## Current baseline (recorded 2026-04-23)

Run: `pytest tests/factors/matching/test_gold_eval_gate.py -v`

```
family                         N      P@1      R@3      MRR   errs
------------------------------------------------------------------
cbam                          35    0.00%    0.00%   0.0000     35
electricity                   60    0.00%    0.00%   0.0000     60
freight                       50    0.00%    0.00%   0.0000     23
fuel_combustion               50    0.00%    0.00%   0.0000     18
materials                     50    0.00%    0.00%   0.0000     14
methodology_profiles          25    0.00%    0.00%   0.0080     17
purchased_goods_proxy         40    0.00%    0.00%   0.0000      3
refrigerants                  40    0.00%    0.00%   0.0000      4
waste                         20    0.00%    0.00%   0.0000     20
------------------------------------------------------------------
OVERALL                      370    0.00%    0.00%   0.0005    230
```

**Status: BELOW THRESHOLD (gate fails).** This is the honest signal the
T3/T4 workstream is meant to surface. Root cause: 230 of 370 entries
(62.2%) raise `FactorCannotResolveSafelyError` because the built-in
`EmissionFactorDatabase` only ships 8 factors — the majority of the
catalog required to satisfy the gold set has not yet been ingested. The
remaining 140 entries also miss because the candidate pool is too shallow
for the resolver's selection rule to land on a match.

Actions required (owners noted):
- **Agent A/E (resolver)**: confirm the cascade correctly prefers the
  highest-specificity candidate when multiple are available; no fix
  needed for this baseline since the pool is too small to exercise it.
- **Catalog seeding (parsers)**: once parsers under
  `greenlang/factors/ingestion/parsers/` load CBAM, CEA, DESNZ, GLEC, and
  EPA WARM feeds into the built-in DB, re-run the gate and record the
  new baseline in this README. The gate itself is unchanged.
- **Methodology owners (HFO refrigerants)**: 8 of the 40 refrigerant
  entries reference R-1234yf/R-1234ze/R-452A/R-513A/R-448A — add these to
  the IPCC GWP factor list if not already present, or flag that these
  blends intentionally fall through to the method-pack default step.

---

## How to add a case

1. Pick the right file:
   - New cases in the **v1.1 schema** go under `v1/<family>.json` (append
     to the existing array).
   - Legacy v1.0 cases stay in their current files; do NOT mix schemas
     within a single file unless the file is explicitly new (e.g.
     `refrigerants_expanded.json` is v1.1-only).

2. For v1.1 entries:
   - Use `gs_<family>_<country>_<year>_<nnn>` for the id.
   - Paraphrase a real activity (see Authenticity rules).
   - Pick a regex pattern that matches the authoritative factor-id your
     organization would expect. When in doubt, anchor loosely to the
     family + country (`EF:EPA:waste_landfill_.*:US:.*`) and let the
     resolver's specificity scoring do the rest.
   - Set `tier_acceptance` to `["primary"]` for high-confidence entries
     (factor is definitely seeded) and `["primary", "alternate_top3"]`
     for entries where the top-3 pool is the real signal.

3. Run the gate locally:
   ```
   pytest tests/factors/matching/test_gold_eval_gate.py -v
   ```
   Inspect `build/gold_eval_report.csv` for the per-case outcome before
   pushing.

4. If precision/recall regresses, do NOT tweak the pattern to pass —
   open a ticket against the resolver/catalog instead.

---

## Versioning

- **v1.0** — original schema (2026-04-23 initial drop).
- **v1.1** — new schema (this commit) with
  `expected_factor_id_pattern` (regex) + `tier_acceptance`. The two
  co-exist: the loader auto-detects schema per entry.

When we make a backward-incompatible schema change (e.g. renaming a
required field) we copy `v1/` to `v2/` and keep `v1/` for one release
before removing it. The CI job pins the version it runs against via the
entry loader in `tests/factors/matching/test_gold_eval_gate.py`.
