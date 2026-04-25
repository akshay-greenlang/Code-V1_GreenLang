# GreenLang Factors — Incident-Drill SOP

**Owner**: data-integration team
**Last updated**: 2026-04-25 (after first real drill — see
[postmortem](../postmortems/2026-Q1-desnz-parser-drift-drill.md))

---

## Purpose

Per CTO doc §19.1 acceptance criterion ("Postmortem from the first
real-incident drill (intentional ingestion fail) completed and filed"),
EVERY alpha-source parser MUST be drilled before the alpha launch flag
flips, and EVERY release MUST repeat the drill on every source. This
document is the standard operating procedure (SOP) the operator follows.

The drill mimics the most-likely real failure mode per CTO doc §19.1
risk: **"Parser drift on DEFRA or EPA Excel workbooks (column positions
shift year-over-year)"**.

---

## Pre-drill checklist (5 min)

Before starting any drill:

- [ ] Confirm the alpha namespace pods are HEALTHY (`kubectl get pods -n
      greenlang-factors -l release_profile=alpha-v0.1`).
- [ ] Confirm Prometheus is scraping the factors job (`curl -s
      http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[]
      | select(.labels.job=="factors")'`).
- [ ] Pull up Grafana dashboard `factors-v0.1-alpha` in a browser tab
      with auto-refresh = 30 s. Pin panels #5 (schema validation
      failures), #6 (provenance gate rejection rate), and #8 (parser
      error rate).
- [ ] Confirm the production seed under
      `greenlang/factors/data/catalog_seed/_inputs/<source>.json` is
      clean (`git status` shows no modifications).
- [ ] Set drill-mode env: `export GL_FACTORS_DRILL=1` (read by the
      drill executor; harmless in any other context).

---

## Per-source drill procedure (10 min each)

For each of the 6 alpha sources, follow steps 1-6 below.

### Step 1 — Build the corrupted fixture

Copy the production seed to the drill-fixture directory, then apply the
"easy" corruption from the table below:

```bash
SRC=desnz_ghg_conversion       # one of the 6 alpha source_ids
SEED_FILE=desnz_uk.json        # see registry mapping in
                               # scripts/factors_alpha_v0_1_backfill.py:53
DRILL_FIX=tests/factors/v0_1_alpha/drill_fixtures/${SRC}_corrupted.json

cp greenlang/factors/data/catalog_seed/_inputs/${SEED_FILE} ${DRILL_FIX}
# Then hand-edit DRILL_FIX to apply the source-specific corruption below.
```

> **CRITICAL**: never edit the production seed itself. The drill fixture
> ALWAYS lives under `tests/factors/v0_1_alpha/drill_fixtures/`.

### Step 2 — Run the drill executor

```bash
python tests/factors/v0_1_alpha/drill_fixtures/_capture_drill_output.py
```

Or, equivalently, point the production backfill script at the drill
fixture (override the input path via the `_seed_path_for` mapping in
`scripts/factors_alpha_v0_1_backfill.py:63-65`).

### Step 3 — Verify the parser failure mode

Inspect the captured stack trace at
`docs/factors/postmortems/evidence/<date>-<source>-stack.txt`. Confirm:

- The expected exception class fires (see per-source table below), OR
- The parser is tolerant AND the normalizer / gate catches every
  record (acceptable if AI-1 is still open).

### Step 4 — Verify the counter delta

Inspect `docs/factors/postmortems/evidence/<date>-<source>-counters.json`
and confirm:

- `factors_parser_errors_total` increments by ≥ 1 (or by # of failed
  rows if AI-1 has shipped).
- `factors_alpha_provenance_gate_rejections_total` increments by 1 per
  produced-but-rejected record.

### Step 5 — Verify Grafana panels light up

Within 30 s of the drill (= one Grafana refresh), confirm in the
browser tab:

- [ ] Panel #5 (schema validation failures) shows a non-zero rate.
- [ ] Panel #6 (provenance gate rejection rate) shows a non-zero rate.
- [ ] Panel #8 (parser error rate) shows a spike (or stays flat-zero
      if AI-1 is still open — note this in the postmortem).

Capture a screenshot of all three panels and save it to
`docs/factors/postmortems/evidence/<date>-<source>-grafana.png`.

### Step 6 — Revert and verify clean state

```bash
# Verify production seed is byte-identical to its committed state.
git diff -- greenlang/factors/data/catalog_seed/_inputs/${SEED_FILE}
# Expected: no output. If output appears, revert immediately:
#   git checkout -- greenlang/factors/data/catalog_seed/_inputs/${SEED_FILE}

# Drill fixture STAYS in the repo as drill evidence — do NOT delete.
# It's protected by the regression test
#   tests/factors/v0_1_alpha/test_drill_fixtures.py
# which asserts the corruption marker only ever appears in the drill fixture.

# Re-run the backfill against the production seed to confirm clean state:
python scripts/factors_alpha_v0_1_backfill.py --source ${SRC} --dry-run
# Expected: 0 failures, 0 skips (or only documented sequestration skips
# for IPCC).
```

---

## Per-source corruption matrix

Each row tells the operator what to corrupt, what to expect, and which
runbook page handles the resulting alert.

| source_id | "Easy" corruption (top-level key rename) | Expected exception class | Expected counter delta | Runbook page |
|-----------|------------------------------------------|--------------------------|------------------------|--------------|
| `desnz_ghg_conversion` | Row-level rename `co2_factor` → `co2_factor_RENAMED_FOR_DRILL` (etc. for ch4/n2o) on every `scope1_fuels` row | `NonPositiveValueError` (one-stage downstream of parser); SHOULD become `KeyError` after AI-1 ships | `factors_alpha_provenance_gate_rejections_total{source="desnz",reason="schema_required_field_missing"}` += N (= # of parser rows) | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |
| `epa_hub` | Rename `metadata.fuels_table` → `*_RENAMED_FOR_DRILL` (the parser's primary section anchor) | `NonPositiveValueError` (today); `KeyError` (post-AI-1) | `factors_alpha_provenance_gate_rejections_total{source="epa_hub",reason="schema_required_field_missing"}` += N | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |
| `egrid` | Rename `egrid_subregion` → `*_RENAMED_FOR_DRILL` on the egrid subregion records | `NonPositiveValueError` (today); `KeyError` (post-AI-1) | `factors_alpha_provenance_gate_rejections_total{source="egrid",reason="schema_required_field_missing"}` += N | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |
| `india_cea_co2_baseline` | Rename `rows[*].emission_factor` → `*_RENAMED_FOR_DRILL` | `NonPositiveValueError` (today); `KeyError` (post-AI-1) | `factors_alpha_provenance_gate_rejections_total{source="india_cea",reason="schema_required_field_missing"}` += N | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |
| `ipcc_2006_nggi` | Rename `default_factor` → `*_RENAMED_FOR_DRILL` on every record | `NonPositiveValueError` (today); `KeyError` (post-AI-1) | `factors_alpha_provenance_gate_rejections_total{source="ipcc",reason="schema_required_field_missing"}` += N | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |
| `cbam_default_values` | Rename `embedded_emissions_intensity` → `*_RENAMED_FOR_DRILL` | `NonPositiveValueError` (today); `KeyError` (post-AI-1) | `factors_alpha_provenance_gate_rejections_total{source="cbam",reason="schema_required_field_missing"}` += N | [factors-v0.1-alpha-alerts.md § Panel 8](./factors-v0.1-alpha-alerts.md#panel-8) |

> **Note on "today" vs "post-AI-1"**: today every parser is tolerant of
> the rename and the rejection only fires at the v0.1 normalizer step,
> producing `NonPositiveValueError`. After AI-1 ships
> (`raw_record_v1.schema.json` per source), the parser itself will
> raise a structured `KeyError` and `factors_parser_errors_total` will
> increment directly. The operator should annotate which path fires in
> the captured evidence file.

---

## Pre-alpha-launch sign-off checklist

Before flipping the alpha-launch flag, the operator MUST have run the
drill against EVERY alpha source AND filed a postmortem (or appended
to the existing one) covering each:

- [ ] DESNZ (`desnz_ghg_conversion`) — drilled 2026-04-25 (postmortem:
      [2026-Q1-desnz-parser-drift-drill.md](../postmortems/2026-Q1-desnz-parser-drift-drill.md)) — DONE
- [ ] EPA GHG Hub (`epa_hub`) — NOT YET DRILLED
- [ ] eGRID (`egrid`) — NOT YET DRILLED
- [ ] India CEA (`india_cea_co2_baseline`) — NOT YET DRILLED
- [ ] IPCC 2006 NGGI (`ipcc_2006_nggi`) — NOT YET DRILLED
- [ ] EU CBAM (`cbam_default_values`) — NOT YET DRILLED

The same checklist also applies to EVERY release after alpha launch:
each new edition's release notes MUST include a drill confirmation per
source.

---

## After every drill — required outputs

1. Drill fixture committed at
   `tests/factors/v0_1_alpha/drill_fixtures/<source>_corrupted_<vN>.json`.
2. Drill evidence files written to
   `docs/factors/postmortems/evidence/<date>-<source>-stack.txt` and
   `<date>-<source>-counters.json`.
3. Grafana screenshot at
   `docs/factors/postmortems/evidence/<date>-<source>-grafana.png`.
4. Postmortem entry under `docs/factors/postmortems/<quarter>-<source>-drill.md`
   following the template established in
   [`2026-Q1-desnz-parser-drift-drill.md`](../postmortems/2026-Q1-desnz-parser-drift-drill.md):
   Summary → Timeline → What worked → What didn't work → 5 whys →
   Action items → Evidence → Lessons applied.
5. Regression test in `tests/factors/v0_1_alpha/test_drill_fixtures.py`
   that asserts the parser-and-gate behaviour observed during the
   drill, marked `@pytest.mark.drill`.

---

## When NOT to drill

The drill is fault-injection. It is SAFE because the corrupted fixture
lives only in `tests/`, but the operator MUST avoid the following
anti-patterns:

- DO NOT corrupt the production seed file directly. The regression test
  `test_drill_fixture_exists_and_is_isolated_from_production` asserts
  the marker token `RENAMED_FOR_DRILL` never appears in production
  seeds; if it does, the build fails.
- DO NOT run the drill against the live alpha API endpoint. Drills run
  in-process via the parser → normalizer → gate path; they never issue
  HTTP requests to the alpha pod.
- DO NOT skip the post-drill `git diff` verification. Even though the
  drill fixture is the only thing that should change, a slipped pointer
  could leak corruption into the production seed.
- DO NOT drill during a customer-facing window. Although the drill is
  in-process, the counter increments are visible to anyone watching the
  Grafana dashboard, and a non-operator could mistake them for real
  ingestion failures.

---

## Cross-references

- **CTO doc**: §19.1 (FY27 Q1 alpha — operations & oversight, postmortem
  acceptance criterion).
- **Postmortem template**:
  [`2026-Q1-desnz-parser-drift-drill.md`](../postmortems/2026-Q1-desnz-parser-drift-drill.md).
- **Alert runbook**:
  [`factors-v0.1-alpha-alerts.md`](./factors-v0.1-alpha-alerts.md).
- **Backfill script**:
  [`scripts/factors_alpha_v0_1_backfill.py`](../../../scripts/factors_alpha_v0_1_backfill.py).
- **Alpha provenance gate**:
  [`greenlang/factors/quality/alpha_provenance_gate.py`](../../../greenlang/factors/quality/alpha_provenance_gate.py).
- **v0.1 normalizer**:
  [`greenlang/factors/etl/alpha_v0_1_normalizer.py`](../../../greenlang/factors/etl/alpha_v0_1_normalizer.py).
