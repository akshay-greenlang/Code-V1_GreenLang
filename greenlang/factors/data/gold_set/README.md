# GreenLang Factors — Public Gold-Label Evaluation Set

This directory holds the **gold-label evaluation set** used to measure how well
the factor matching + resolution pipeline picks the right emission factor for
a given activity.

It backs Track B-1 of `FY27_Factors_Launch_Checklist.md` ("Public gold-label
evaluation set, ≥300 representative activities") and is the data the
`factors-gold-eval` GitHub Actions job consumes on every PR that touches
`greenlang/factors/**`.

---

## Layout

```
gold_set/
├── README.md                    # this file
└── v1/
    ├── index.json               # case_count + per-family manifest
    ├── electricity.json         # 60 cases — IN/EU/UK/US, location/market/residual
    ├── fuel_combustion.json     # 50 cases — NG, diesel, gasoline, coal, LPG, fuel oil
    ├── refrigerants.json        # 32 cases — R-22 / R-32 / R-134a / R-410A / R-404A / R-507 / R-1234yf
    ├── freight.json             # 50 cases — road, sea, air, rail, intermodal (WTW/TTW)
    ├── materials.json           # 50 cases — steel, aluminium, cement, plastic, paper, fertilizer, glass
    ├── cbam.json                # 35 cases — CN-coded steel, aluminium, cement, fertilizer, hydrogen, electricity
    └── methodology_profiles.json # 25 cases — cross-cutting profile-routing exercises
```

Total: **302 cases** (≥ 300 acceptance bar).

---

## Case schema (v1.0)

Each case is a single JSON object inside the family file (the file is a
top-level array of cases):

```json
{
  "case_id": "elec_in_northern_grid_2027_lb_002",
  "activity": {
    "description": "Purchased grid electricity, India Northern Grid (FY2026-27), location-based",
    "quantity": 8200,
    "unit": "kWh",
    "metadata": {"country": "IN", "grid_region": "northern_grid", "year": 2027, "scope": "scope2"}
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
  "tags": ["electricity", "india", "location_based", "northern_grid", "fy27_launch"]
}
```

Field-by-field:

| Field | Description |
|---|---|
| `case_id` | Globally unique slug. Convention: `<family>_<geo>_<descriptor>_<seq>`. |
| `activity.description` | Free-text activity. Fed straight to `match_activity()`. |
| `activity.quantity` / `unit` | Activity quantity + canonical unit. Used by the resolver to apply unit conversion. |
| `activity.metadata` | Free-form dict; common keys: `country`, `year`, `scope`, `grid_region`, `instrument`, `cn_code`, `gwp_basis`, `boundary`. |
| `method_profile` | Required. One of the `MethodProfile` enum values from `greenlang.data.canonical_v2`. |
| `expected.factor_id` | The exact `factor_id` we expect at rank 1. **`null` is allowed** when the factor is not yet seeded in the catalog — the test then asserts on `factor_family` + `co2e_per_unit_min/max` instead. |
| `expected.factor_family` | One of the `FactorFamily` enum values. Always required — it is the fallback assertion when `factor_id` is `null`. |
| `expected.source_authority` | Display authority (CEA / DESNZ / EPA / IPCC / GLEC / EU CBAM / supplier / etc.). |
| `expected.fallback_rank` | Cascade step we expect the resolver to land on (1 = customer override, 7 = global default). |
| `expected.co2e_per_unit_min` / `_max` | Acceptable range for the resolved factor's `co2e_per_unit`. Tight where we have published authority data, loose where we are quoting industry ranges. |
| `expected.co2e_unit` | Canonical unit of the range (`kgCO2e/kWh`, `kgCO2e/tonne`, etc.). |
| `expected.must_include_assumptions` | Substrings that must appear (case-insensitive) in the resolved factor's assumptions list. |
| `tags` | Free-form filter tags; every case carries the `fy27_launch` tag so CI can filter on it. |

---

## Factor-id convention

The factor-id strings used in this set match the on-disk parsers under
`greenlang/factors/ingestion/parsers/`:

```
EF:<authority_or_jurisdiction>:<fuel_or_product>:<geo>:<year>:v<n>
```

Real-world examples emitted by parsers in the catalog seed:

| Parser | Example id |
|---|---|
| `india_cea.py` | `EF:IN:northern_grid:2026-27:cea-v20.0` |
| `desnz_uk.py` | `EF:DESNZ:s1_natural_gas_kwh:UK:2026:v1` |
| `iea.py` | `EF:IEA:diesel:US:2026:v1` |
| `aib_residual_mix.py` | `EF:DE:residual_mix:2026:v1.0` |
| `eGRID` (planned) | `EF:eGRID:rfce:US:2026:v1` |
| CBAM (`policy_factor_map.example.yaml`) | `EF:CBAM:steel:CN:2024:v1` |

**Where we don't yet have a real seeded factor** (e.g. supplier-specific
PPAs, EU country-level location-based factors not yet in the catalog,
GLEC freight lanes not yet ingested) we set `expected.factor_id` to
`null`. The test runner then drops the rank-1 id assertion and instead
asserts that:

1. The pipeline returns *some* candidate, and
2. the resolved factor's family matches `expected.factor_family`, and
3. the per-unit CO2e value falls inside `[co2e_per_unit_min,
   co2e_per_unit_max]`.

This keeps the gold set useful **today** (before the catalog seed lands)
and **tomorrow** (once it does — at which point the `null` ids should
be filled in with the real seeded ids).

---

## Acceptance bar (precision@1)

The runner (`tests/factors/test_gold_set_eval.py`) computes
**precision@1** globally and per family:

```
P@1 = correct_rank_1_predictions / total_cases
```

A case is "correct at rank 1" if either:

- `expected.factor_id` is non-null and the matcher returns it as the
  top-ranked `factor_id`, OR
- `expected.factor_id` is null and the matcher returns *any* candidate
  whose family + co2e-range pass the relaxed checks.

| Bar | Threshold | Status |
|---|---|---|
| Initial (B-1, FY27 launch checklist) | **P@1 ≥ 0.85** | enforced today |
| Post-launch (B-3) | P@1 ≥ 0.90 | scheduled for raise |
| CI delta vs `main` | drop ≤ 2 points | enforced today |

The CI job uploads `artifacts/gold_eval_summary.json` and posts a delta
comment on every PR. A drop > 2 points blocks merge.

---

## How to add a case

1. Edit `scripts/generate_gold_set_v1.py` (the canonical generator), or
   hand-edit the relevant family JSON file directly. Keep both in sync.
2. For each new case:
   - Pick a unique `case_id` (`<family>_<geo>_<seq>`).
   - Use a real activity description, not a synthetic one.
   - Use a published authority value for `co2e_per_unit_min/max`. Cite
     the source in the case-generation script comments. If you can't
     verify, set a wide range (±20 %) and add a `tags: ["low_confidence"]`
     entry so we can sweep these later.
   - Set `expected.factor_id` only if the factor really exists in the
     catalog seed — otherwise leave it `null`.
   - Pick the *correct* `method_profile` (this is what most callers get
     wrong: scope-2 market vs location is the most common mistake).
3. Re-run `python scripts/generate_gold_set_v1.py` to refresh the file
   and `index.json`.
4. Run `pytest tests/factors/test_gold_set_eval.py -v` locally to confirm
   the new case is parsed and that precision@1 is still ≥ 0.85.

---

## Versioning

This is `v1`. When we make a *backward-incompatible* schema change
(e.g. renaming `expected.factor_family`), we copy `v1/` to `v2/` and
keep `v1/` for one release before removing it. The CI job pins the
version it runs against in `.github/workflows/factors_gold_eval.yml`.
