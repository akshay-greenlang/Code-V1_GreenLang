# Design partner pilot kit (DP1–DP3)

## Onboarding

- [ ] Edition pin documented (`X-Factors-Edition` + changelog review).
- [ ] NDA and data-rights addendum signed for connector sources.
- [ ] Sample audit export (`greenlang.factors.quality.audit_export.build_audit_bundle_dict`).

## Instrumentation (DP2)

- [ ] Time-to-first-correct-factor (median / p95).
- [ ] Mismatch rate vs gold eval (`tests/factors/fixtures/gold_eval_smoke.json`).
- [ ] Citation completeness score from provenance payloads.

## Feedback loop (DP3)

- [ ] Mismatches filed as `qa_reviews` rows (SQLite) with repro text.
- [ ] Gold set updated monthly from pilot telemetry.
