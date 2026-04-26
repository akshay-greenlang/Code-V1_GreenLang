# Acceptance Checklist — `<release-id>`

> Template. Copy to
> `docs/factors/release-templates/instances/<release-id>/ACCEPTANCE_CHECKLIST.md`
> and tick every box. Missing signoffs block the release per
> source-of-truth governance rule #5.

## Identity

| Field             | Value                                  |
| ----------------- | -------------------------------------- |
| Release id        | `<release-id>`                         |
| Release profile   | `<alpha-v0.1 \| beta-v0.5 \| ...>`      |
| Target environment| `<production \| staging \| canary>`     |
| Sign-off date     | `<ISO-8601>`                           |

## Exit criteria

Each item below MUST be checked OR explicitly waived (with a
linked ADR or release-council minute).

### Engineering quality

* [ ] `pytest tests/factors/v0_1_alpha -q` reports `0 failed`.
* [ ] URN canonical-parse regression passes
      (`test_seed_urns_canonical_parse.py`).
* [ ] OpenAPI snapshot pinned + matches the served `/openapi.json`.
* [ ] Schema validation green
      (`test_schema_validates_alpha_catalog.py`).
* [ ] Provenance gate green for every alpha source.
* [ ] Performance budget met (see TEST_REPORT.md table).
* [ ] No new SEV-1 / SEV-2 alerts firing in staging at the time
      of cut.
* [ ] Release-profile route filter verified
      (`test_alpha_api_contract.py::test_openapi_documents_only_5_alpha_endpoints`
      for alpha).

### Product / methodology

* [ ] Source registry up-to-date for every served source.
* [ ] Methodology exceptions documented under
      `docs/factors/v0_1_alpha/methodology-exceptions/` if any.
* [ ] Source vintage drift drill executed at least once in the
      release window
      (`docs/factors/runbooks/2026-Q1-desnz-parser-drift-drill.md`).

### Security / compliance

* [ ] No secrets committed in this release diff
      (`gl-secscan` clean).
* [ ] Source licences in served catalog all match the
      source registry licence allow-list
      (Compliance/Security gate).
* [ ] JWT_SECRET present in the target environment.
* [ ] Audit log enabled and verified to receive at least one
      event during the release smoke test.
* [ ] PII detector / redactor enabled where applicable
      (v0.9+ for ML resolve).

### Operations

* [ ] Grafana dashboard receives metrics from the new build
      (`deployment/.../grafana/dashboards/factors-v<rel>.json`).
* [ ] Prometheus alerts loaded
      (`deployment/.../prometheus/factors-v<rel>-alerts.yaml`).
* [ ] AlertManager routing set per environment.
* [ ] Rollback plan in `RELEASE_MANIFEST.md` reviewed by SRE
      Lead.
* [ ] On-call notified of release window.

### Customer / partner readiness

* [ ] Affected partners notified at least 5 business days in
      advance for breaking changes.
* [ ] Customer-impact memo prepared if applicable (template:
      `CUSTOMER_IMPACT_TEMPLATE.md`).
* [ ] Release notes drafted (`releases/<release-id>/RELEASE_NOTES.md`).

### Documentation

* [ ] OpenAPI snapshot updated.
* [ ] SDK readme + version bumped.
* [ ] Public docs (Quickstart, hosted_api.md) updated.
* [ ] CHANGELOG entry written.

## Sign-off

Each role below signs once their gates above are met. The release
PR cannot be merged unless ALL of (R, A) plus the indicated (C)
roles have signed.

| Role                       | RACI | Name           | Decision    | Signed at (ISO-8601) |
| -------------------------- | ---- | -------------- | ----------- | -------------------- |
| Platform/Data Lead         | R    | `<name>`       | `<approve>` | `<timestamp>`        |
| CTO                        | A    | `<name>`       | `<approve>` | `<timestamp>`        |
| SRE Lead                   | C    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Backend/API Lead           | C    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Climate Methodology Lead   | C    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Compliance/Security Lead   | C    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Partner Success Lead       | C/I  | `<name>`       | informed (alpha) / concur (beta+) | `<timestamp>` |
| DevRel / Docs              | I    | `<name>`       | informed    | `<timestamp>`        |

## Waivers

If any exit-criteria item is waived, document it here with the
waiver authority (CTO for product/security/methodology, Platform/Data
Lead for engineering quality / ops):

| Waived item | Reason | Waived by | Authority | Mitigation | Target to clear |
| ----------- | ------ | --------- | --------- | ---------- | --------------- |
|             |        |           |           |            |                 |
