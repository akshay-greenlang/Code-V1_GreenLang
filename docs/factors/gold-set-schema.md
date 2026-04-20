# Factor-Matching Gold Set — Schema & Rationale

> **What.** The canonical evaluation set that the Factor-matching pipeline is tested against on every merge to `master`.
> **Where.** `tests/factors/fixtures/gold_eval_curated.json` (380 cases).
> **Source of truth.** `tests/factors/fixtures/gold_eval_full.json` (511 candidate cases) + `scripts/curate_gold_eval.py`.

---

## 1. Why a gold set

Factor matching is the step where a free-text activity description (e.g. "diesel combustion stationary US") is mapped to a specific factor in the catalog. This is the one place the pipeline uses semantic matching + LLM rerank, and it's the one place where we can silently regress. The gold set locks in a corpus of `(activity → expected_factor)` pairs and measures precision / recall on every CI run.

## 2. Case schema (v1.0)

Every case in the curated file has **exactly these six fields**:

```json
{
  "id": "g042",
  "activity": "grid electricity for data center California",
  "expected_fuel_type": "electricity",
  "geography": "US",
  "domain": "energy",
  "difficulty": "medium"
}
```

### Field meanings

| Field | Required | Meaning | Example values |
|---|---|---|---|
| `id` | yes | Stable identifier; never renumber | `g042` |
| `activity` | yes | Free-text activity description | `"coal combustion boiler UK"` |
| `expected_fuel_type` | yes | Canonical fuel id the matcher should select | `"diesel"`, `"electricity"`, `"natural_gas"`, `"coal"`, `"gasoline"` |
| `geography` | yes | ISO country/region code | `"US"`, `"EU"`, `"UK"`, `"IN"`, `"CN"` |
| `domain` | yes | Activity taxonomy bucket | `"energy"`, `"transport"`, `"buildings"`, `"industry"`, `"agriculture"`, `"regulatory"`, or one of the "hard" markers below |
| `difficulty` | yes | Heuristic difficulty tier | `"easy"` \| `"medium"` \| `"hard"` |

### "Hard" markers

Cases deliberately probing matcher weaknesses use these `domain` values:

| Marker | Probes |
|---|---|
| `misspelling` | Typos in activity text ("diesl") |
| `abbreviation` | Short forms ("elec", "NG") |
| `cross_geography` | Cross-regional descriptions ("fleet operating in US and EU") |
| `ambiguous` | Multi-fuel wording ("blended fuel") |
| `edge` | Rare activities (propane in industrial context) |
| `multilingual` | Non-English terms (de/fr/es) |
| `scoped` | Scope-tagged activity ("scope 2 market-based electricity") |
| `units` | Non-standard units ("MMBTU", "therms") |
| `temporal` | Time-sensitive phrasing ("2023 electricity grid") |

Each `hard` marker maps automatically to `difficulty: "hard"`.

## 3. Curation process

### 3.1 Source → curated

Run:

```bash
python scripts/curate_gold_eval.py                 # writes curated set
python scripts/curate_gold_eval.py --audit-only    # prints coverage without writing
```

The script:

1. Loads `gold_eval_full.json` (511 raw candidate cases).
2. **Enriches** any case missing `geography`, `domain`, or `difficulty` via keyword heuristics. Defaults: `geography → US`, `domain → energy`, `difficulty → medium` (with `easy` for ≤ 3-word activities and `hard` for any "hard marker" match).
3. **Stratified-samples** by `(expected_fuel_type, geography)` to hit a target of **380 cases** (tunable via `--target`).
4. Writes a versioned JSON envelope:

```json
{
  "schema_version": "1.0",
  "generated_from": "tests/factors/fixtures/gold_eval_full.json",
  "curation_seed": 20260420,
  "target_size": 380,
  "actual_size": 380,
  "cases": [...]
}
```

The `curation_seed` is baked into the output so results are **bit-reproducible** — reruns produce the same file.

### 3.2 Coverage (after curation)

Current distribution (2026-04-20 run):

| Fuel type | Count |
|---|---|
| electricity | 99 |
| diesel | 90 |
| natural_gas | 79 |
| gasoline | 61 |
| coal | 51 |

| Geography | Count |
|---|---|
| US | 263 |
| EU | 70 |
| UK | 46 |
| IN | 1 |

| Difficulty | Count | Share |
|---|---|---|
| easy | 212 | 55.8 % |
| medium | 91 | 23.9 % |
| hard | 77 | 20.3 % |

Domain distribution (hard markers total 77 — exactly 20 % as required):

```
energy(199)  transport(35)  buildings(25)  industry(22)
cross_geography(16)  units(11)  multilingual(10)  edge(10)
misspelling(8)  scoped(8)  regulatory(8)  agriculture(7)
ghg_protocol(7)  ambiguous(5)  temporal(5)  abbreviation(4)
```

## 4. Evaluation harness

The gold set is consumed by `greenlang/factors/matching/evaluation.py::MatchEvaluator`. The harness:

- Loads `gold_eval_curated.json`.
- Runs the current matcher against each `activity`.
- Measures per-case pass/fail: the matcher's top-1 candidate must match `expected_fuel_type` **and** geographic compatibility (US → US or US-CA; EU → any EU-27; UK → UK only).
- Aggregates metrics per domain and per difficulty.

Typical output (written to `results/eval_report.json`):

```json
{
  "run_id": "<sha>",
  "total_cases": 380,
  "pass_rate_overall": 0.892,
  "pass_rate_by_difficulty": {
    "easy": 0.986, "medium": 0.912, "hard": 0.675
  },
  "pass_rate_by_domain": {
    "energy": 0.95, "transport": 0.91, "misspelling": 0.56, ...
  }
}
```

## 5. CI workflow

The curated gold set is exercised weekly and on any change to the matcher via `.github/workflows/factors-gold-eval.yml` (Phase 5.2). The workflow:

1. Regenerates `gold_eval_curated.json` from source to verify bit-reproducibility.
2. Runs `MatchEvaluator.evaluate()`.
3. Writes `results/eval_report.json` as a CI artifact.
4. Fails the run if overall pass-rate drops more than 2 percentage points vs the previous merge's report.

## 6. Updating the gold set

- **Adding cases:** append to `gold_eval_full.json` with a new `g<nnn>` id. Re-run `scripts/curate_gold_eval.py`. Commit both files.
- **Fixing labels:** edit `gold_eval_full.json` in place; the curator regenerates.
- **Retiring cases:** never delete — set `difficulty: "retired"` and the evaluator skips them. Keeps historical ids stable.
- **Schema-breaking changes:** bump `schema_version` in the output envelope + the evaluator's assertion.

## 7. References

- `scripts/curate_gold_eval.py` — the curator.
- `greenlang/factors/matching/evaluation.py` — the evaluator.
- `tests/factors/fixtures/gold_eval_full.json` — 511-case source.
- `tests/factors/fixtures/gold_eval_curated.json` — 380-case canonical eval set.
- `.github/workflows/factors-gold-eval.yml` — weekly CI job.
- `docs/factors/hosted_api.md` — how matching fits in the hosted API.

---

*Last updated: 2026-04-20. Source: FY27_vs_Reality_Analysis.md §5.2 + FY27 Factors proposal §10.*
