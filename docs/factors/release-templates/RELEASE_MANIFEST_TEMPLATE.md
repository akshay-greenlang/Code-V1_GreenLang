# Release Manifest — `<release-id>`

> Template. Copy to
> `docs/factors/release-templates/instances/<release-id>/RELEASE_MANIFEST.md`
> and fill every `<placeholder>`. Submit via the release PR.

## Identity

| Field             | Value                                                |
| ----------------- | ---------------------------------------------------- |
| Release id        | `<release-id>` (e.g. `factors-v0.1.0-alpha-2026-04-25`) |
| Release version   | `<semver>` (e.g. `0.1.0-alpha`)                      |
| Release profile   | `<alpha-v0.1 \| beta-v0.5 \| rc-v0.9 \| ga-v1.0>`     |
| Release type      | `<edition-cut \| feature \| hotfix>`                 |
| Release timestamp | `<ISO-8601 UTC>`                                     |
| Release commit    | `<git SHA, 40-hex>`                                  |
| Release tag       | `<git tag name>`                                     |

## Source versions

List every upstream source that contributed records to this
release. Pull from the per-source manifest (template:
`SOURCE_MANIFEST_TEMPLATE.md`).

| source_id          | source URN                              | source_version | parser_version | parser_commit |
| ------------------ | --------------------------------------- | -------------- | -------------- | ------------- |
| `<source_id>`      | `<urn:gl:source:...>`                   | `<x.y>`        | `<x.y.z>`      | `<7-40 hex>`  |

## Pack versions

| pack URN                                      | pack id              | version |
| --------------------------------------------- | -------------------- | ------- |
| `urn:gl:pack:<source>:<pack-id>:v<n>`         | `<pack-id>`          | `v<n>`  |

## Schema version

| Schema                                        | $id                                                                                | version |
| --------------------------------------------- | ---------------------------------------------------------------------------------- | ------- |
| Factor Record                                 | `https://schemas.greenlang.io/factors/factor_record_v<n>.schema.json`              | `v<n>`  |

## Migrations

| Migration id                                  | Type                  | Applied at                    |
| --------------------------------------------- | --------------------- | ----------------------------- |
| `<V###__name.sql>` or `<alembic-rev-id>`      | `<DDL \| data \| both>` | `<env: dev/staging/prod>`     |

## Approvers (RACI #4 Release Approval)

| Role                       | Name           | Decision    | Signed at (ISO-8601) |
| -------------------------- | -------------- | ----------- | -------------------- |
| Platform/Data Lead (R)     | `<name>`       | `<approve>` | `<timestamp>`        |
| CTO (A)                    | `<name>`       | `<approve>` | `<timestamp>`        |
| SRE Lead (C)               | `<name>`       | `<concur>`  | `<timestamp>`        |
| Backend/API Lead (C)       | `<name>`       | `<concur>`  | `<timestamp>`        |
| Climate Methodology (C)    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Compliance/Security (C)    | `<name>`       | `<concur>`  | `<timestamp>`        |
| Partner Success Lead (I)   | `<name>`       | informed    | `<timestamp>`        |

## Rollback Plan

Document EXACTLY how to roll this release back if it goes wrong.

* **Trigger:** what symptom flips the rollback decision?
* **Owner:** who decides? (default: Platform/Data Lead, escalate
  to CTO for customer-impacting rollbacks)
* **Steps:**
  1. `<step 1>`
  2. `<step 2>`
  3. ...
* **Recovery time objective (RTO):** `<minutes>`
* **Recovery point objective (RPO):** `<minutes>`
* **Verification after rollback:** which command(s) prove the
  rollback succeeded?

## Linked Artifacts

| Artifact                | Path                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| Source Manifest         | `docs/factors/release-templates/instances/<release-id>/SOURCE_MANIFEST.md`                         |
| Test Report             | `docs/factors/release-templates/instances/<release-id>/TEST_REPORT.md`                             |
| Acceptance Checklist    | `docs/factors/release-templates/instances/<release-id>/ACCEPTANCE_CHECKLIST.md`                    |
| Customer-Impact Memo    | `docs/factors/release-templates/instances/<release-id>/CUSTOMER_IMPACT.md` (only when applicable) |
| Signed manifest blob    | `releases/<release-id>/manifest.json`                                                              |
| Manifest signature      | `releases/<release-id>/manifest.json.sig` (or `.sig.placeholder` for alpha)                        |
| Release notes           | `releases/<release-id>/RELEASE_NOTES.md`                                                           |
