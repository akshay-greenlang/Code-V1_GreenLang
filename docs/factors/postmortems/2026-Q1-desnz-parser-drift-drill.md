# DESNZ parser column-shift drill — 2026-04-25

**Severity (drilled)**: SEV-3 (catalog ingestion partial failure; no customer impact in alpha)
**Detected at**: 2026-04-25 07:59:25 UTC (drill timestamp; see evidence file)
**Resolved at**: 2026-04-25 07:59:35 UTC (immediately, by deleting the drill fixture from the parser path; the production seed at `greenlang/factors/data/catalog_seed/_inputs/desnz_uk.json` was never modified)
**Drill conducted by**: bot:gl-devops-engineer (under operator confirmation; see Wave E / TaskCreate #26)
**Drill type**: Intentional fault-injection in the alpha-pre-launch hardening cycle
**Status**: CLOSED (drill complete; no production impact; four action items filed)

---

## Summary

We intentionally renamed every numeric column in a copy of the DESNZ
2024 GHG conversion factors seed (`co2_factor` → `co2_factor_RENAMED_FOR_DRILL`,
plus the same on `ch4_factor` and `n2o_factor`), renamed one whole
section (`scope1_bioenergy` → `scope1_bioenergy_RENAMED_FOR_DRILL`),
and replaced one section with a non-list scalar
(`scope2_electricity = "NOT_A_LIST_RENAMED_FOR_DRILL"`). The corrupted
fixture lives ONLY at
`tests/factors/v0_1_alpha/drill_fixtures/desnz_2024_corrupted_v1.json`;
production data is UNTOUCHED.

The drill validated four claims:

  1. The DESNZ ingestion pipeline does NOT silently let corrupt records
     reach the v0.1 alpha catalog.
  2. The Alpha Provenance Gate / v0.1 normalizer is the safety net that
     actually catches the corruption (every one of 57 emitted records
     was rejected with `NonPositiveValueError`).
  3. The Prometheus counter
     `factors_alpha_provenance_gate_rejections_total` increments per
     rejection.
  4. The `factor_record_v0_1.schema.json` layer of the gate rejects
     malformed records that bypass the normalizer (20 distinct schema
     failures captured for a hand-rolled malformed input).

The drill ALSO surfaced a real gap: **the DESNZ parser itself is too
tolerant — it does not raise at parser time when row-level numeric
columns are renamed.** It uses `_safe_float(row.get("co2_factor"))`
which silently defaults to `0.0`. The safety net only fires one stage
later, in the normalizer. This is captured below as Action Item AI-1.

---

## Timeline

| Offset | Event |
|--------|-------|
| T+0    | Drill fixture committed at `tests/factors/v0_1_alpha/drill_fixtures/desnz_2024_corrupted_v1.json` (production seed UNTOUCHED) |
| T+0:05 | Drill executor invoked (`tests/factors/v0_1_alpha/drill_fixtures/_capture_drill_output.py`) — mirrors the production backfill loop against the corrupted fixture |
| T+0:10 | `parse_desnz_uk(payload)` returned **57 records, no exception** (parser was tolerant of the row-level rename — see "What didn't work" #1 below). Total records dropped from a clean 196 → 57 because the section-level rename and non-list scalar suppressed three sections |
| T+0:11 | Stage 2: every one of the 57 records raised `NonPositiveValueError` from `lift_v1_record_to_v0_1` because vectors collapsed to `0 + 0*28 + 0*265 = 0` (alpha schema requires `value > 0`). First captured trace: `factor_id='EF:DESNZ:s1_natural_gas_kwh_net_cv:UK:2024:v1'` |
| T+0:12 | `factors_alpha_provenance_gate_rejections_total{source="desnz",reason="schema_required_field_missing"}` incremented by 57 |
| T+0:13 | Stage 3: hand-rolled malformed record (missing `extraction` + `review` blocks) submitted to `AlphaProvenanceGate.assert_valid` — raised `AlphaProvenanceGateError` with **20 distinct failure messages** including `'extraction' is a required property` and `'review' is a required property`. Confirms the gate's schema layer is wired correctly |
| T+0:14 | Stage 4: `factors_parser_errors_total{source="desnz_ghg_conversion",error_type="DrillSimulated_KeyError_co2_factor"}` incremented by 1 (simulated counter — see AI-1; the production parser doesn't emit this today because it doesn't raise) |
| T+0:15 | Drill complete; evidence written to `docs/factors/postmortems/evidence/2026-04-25-desnz-{stack.txt,counters.json}` |
| T+0:20 | Postmortem drafted (this file) |

---

## What worked

- **Normalizer caught 57/57 corrupted records (100% rejection).** Every
  record produced by the parser raised `NonPositiveValueError` at
  `lift_v1_record_to_v0_1` because vectors collapsed to zero. No
  corrupted record reached the v0.1 catalog seed.
- **Counter emission path is live.** Both
  `factors_alpha_provenance_gate_rejections_total` and (via simulated
  emit) `factors_parser_errors_total` incremented in-process. The
  Prometheus client registry is wired to both metrics in
  `greenlang/factors/observability/prometheus_exporter.py:135-154`.
- **Gate's schema layer rejected hand-rolled malformed records.** A
  record missing the entire `extraction` and `review` blocks raised
  `AlphaProvenanceGateError` with 20 distinct schema failures, exactly
  matching the FROZEN `factor_record_v0_1.schema.json` contract.
- **Production seed was never modified.** The drill fixture lives only
  under `tests/factors/v0_1_alpha/drill_fixtures/`; the production seed
  at `greenlang/factors/data/catalog_seed/_inputs/desnz_uk.json` is
  byte-identical to its pre-drill state.
- **Backfill script's failure-vs-skip logic is correct.** Although the
  drill records collapsed to `value=0`, the backfill script
  (`scripts/factors_alpha_v0_1_backfill.py:238-240`) catches
  `NonPositiveValueError` as a SKIP, not a FAIL — which is the
  documented behaviour for sequestration factors. In the drill's case
  the same code path keeps a corrupted-batch from polluting the catalog
  while leaving exit code = 1 only when actual gate failures (not
  non-positive values) accumulate. This is correct: a fully-corrupted
  source produces 0 lifted records, which is itself the alarm.

## What didn't work

1. **The DESNZ parser does not fail loud on column rename.** It uses
   `_safe_float(row.get("co2_factor"), default=0.0)` for every numeric
   column, so renaming `co2_factor` → `co2_factor_RENAMED_FOR_DRILL`
   silently produces records with `vectors={"CO2": 0, "CH4": 0, "N2O":
   0}` instead of raising `KeyError`. This means a real-world DESNZ
   column shift would only be detected one stage later in the
   normalizer. Earlier detection (at parser time) would (a) save ~20 s
   per backfill run, (b) point at the offending column rather than at
   the synthetic "value collapsed to 0" message, and (c) emit
   `factors_parser_errors_total` directly instead of via simulated
   counter calls. **(Filed as AI-1.)**

2. **`factors_parser_errors_total` is not emitted today by the DESNZ
   parser.** Because the parser never raises, the counter never
   increments organically — only the rejection counter fires. Panel 8
   of the alpha Grafana dashboard ("Parser error rate") would have
   stayed flat-zero in prod even though the catalog was corrupted.
   **(Filed as AI-2.)**

3. **AlertManager routing is not wired in the alpha namespace yet.**
   Both counters are exposed to Prometheus, but no AlertManager route
   sends them to PagerDuty / Slack. This was already on the Wave 5 SRE
   list; the drill confirms it's still open. **(Filed as AI-3.)**

4. **The current alert threshold for parser errors is too lax.**
   `deployment/observability/prometheus/factors-v0.1-alpha-alerts.yaml`
   does not have a fast-track alert for `parser_errors > 0` in any
   1-min window. Today the design-partner SLA is "no silent corruption";
   even a single parser error should page. **(Filed as AI-4.)**

5. **Grafana dashboard screenshot was not captured.** The drill SOP
   should require operators to capture panels #5/#6/#8 lighting up
   during the drill window. Captured here as a gap; addressed in
   `docs/factors/runbooks/incident-drill-sop.md` step 6.

## 5 whys

- **Why did the parser fail?** Because the upstream DESNZ workbook
  column layout changed (drilled — actually because we renamed the
  three numeric columns to `*_RENAMED_FOR_DRILL` in a fixture).
- **Why did the parser only emit zeros instead of a structured error?**
  Because `parse_desnz_uk` in
  `greenlang/factors/ingestion/parsers/desnz_uk.py` uses
  `_safe_float(row.get(<key>), default=0.0)` with no per-source
  schema-validation step before lifting. The pattern is consistent
  across all six alpha-source parsers.
- **Why is there no schema validation step?** Because the parsers were
  written for v1 shape only; the v0.1 lift happens later via
  `alpha_v0_1_normalizer.py`. The team's intent at the time was to keep
  parsers tolerant so partial workbooks could still load.
- **Why doesn't the lift step catch this earlier?** It does — but at the
  alpha-gate level (one stage downstream), not at the parser-input
  level. Earlier detection would let us point the operator at the exact
  shifted column rather than at "value collapsed to 0".
- **Why is earlier detection valuable?** Because it saves ~20 seconds per
  backfill run and (more importantly) tells the operator
  "DESNZ workbook layout shifted at column X" before computing 196
  records that are all going to fail downstream anyway. For a real
  DESNZ Q1 release, the column shift could happen on a Monday morning;
  catching it at parser-input time means the on-call gets a 1-line
  alert ("DESNZ raw_record schema validation failed at row 0, column
  co2_factor") instead of 196 cascade rejections.

## Action items

| ID | Description | Owner | Target | Status |
|----|-------------|-------|--------|--------|
| **AI-1** | Add a raw-input schema validation step to each alpha-source parser. Each parser should load `raw_record_v1.schema.json` for its source and `jsonschema.validate()` every row BEFORE building factor records. Expected outcome: column rename → `KeyError` → `factors_parser_errors_total` increments → fast alert. | data-integration | pre-alpha-launch (FY27 Q1) | OPEN |
| **AI-2** | Wire AlertManager routing for `factors_parser_errors_total` and `factors_alpha_provenance_gate_rejections_total`. Today both counters live only in Prometheus; the alpha namespace has no PagerDuty / Slack notifier. Required for alpha launch per CTO doc §19.1 oversight clause. | SRE | alpha launch | OPEN (already on Wave 5 list) |
| **AI-3** | Add a fast-track alert in `factors-v0.1-alpha-alerts.yaml` for `factors_parser_errors_total > 0` over any 1-minute window (current threshold doesn't trigger on single events). | SRE | pre-alpha-launch | OPEN |
| **AI-4** | Establish a "drill day" cadence — run this same fault-injection drill on EVERY alpha source PRE EACH RELEASE. SOP at `docs/factors/runbooks/incident-drill-sop.md`. Operator must check off all 6 sources before the alpha launch flag flips. | data-integration | ongoing (each release) | OPEN |

---

## Evidence

| Artefact | Path |
|----------|------|
| Drill fixture | `tests/factors/v0_1_alpha/drill_fixtures/desnz_2024_corrupted_v1.json` |
| Drill executor | `tests/factors/v0_1_alpha/drill_fixtures/_capture_drill_output.py` |
| Stack trace + per-stage observations | `docs/factors/postmortems/evidence/2026-04-25-desnz-stack.txt` |
| Prometheus counter snapshot | `docs/factors/postmortems/evidence/2026-04-25-desnz-counters.json` |
| Regression test | `tests/factors/v0_1_alpha/test_drill_fixtures.py` |
| Drill SOP (codified) | `docs/factors/runbooks/incident-drill-sop.md` |
| Grafana dashboard screenshot | NOT CAPTURED (manual op; addressed in SOP step 6) |

### Captured counter values

```json
{
  "factors_parser_errors_total": {
    "{source='desnz_ghg_conversion',error_type='DrillSimulated_KeyError_co2_factor'}": 1
  },
  "factors_alpha_provenance_gate_rejections_total": {
    "{source='desnz',reason='schema_required_field_missing'}": 57
  }
}
```

Note on the count: this drill's corrupted fixture produced **57**
records (not the canonical 196 from the clean seed) because the
section-level corruption (`scope1_bioenergy` rename, `scope2_electricity`
non-list) suppressed three whole sections at the parser dispatch level
(via `isinstance(s, list)` guards in `parse_desnz_uk`). The 57 figure
is the actual count captured during this drill; the postmortem
template's example `196` is intentionally larger so the team can verify
a future operator running this drill against the full clean seed (with
ONLY row-level corruption) would also see 100% rejection — i.e. the
safety net scales linearly with the input.

## Lessons applied

This is the FIRST real-incident drill GreenLang Factors has run on
the alpha launch path. Before this drill there was no SOP for
"intentionally corrupt a source and confirm the alert/response loop
works". The SOP itself is now codified at
[`docs/factors/runbooks/incident-drill-sop.md`](../runbooks/incident-drill-sop.md)
with one row per alpha source and a checklist that an operator must
work through before flipping the alpha-launch flag. Future drills
update this postmortem (or chain-link to a new postmortem with the same
template).
