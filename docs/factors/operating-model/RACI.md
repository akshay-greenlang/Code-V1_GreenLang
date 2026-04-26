# GreenLang Factors — RACI Matrix

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Accepted - Phase 0 closed                                                   |
| Date             | 2026-04-26                                                                  |
| Owner            | CTO (binds the matrix); Engineering Manager Factors (operates it)           |
| Related          | `OPERATING_MODEL.md`, `engineering/ENGINEERING_CHARTER.md`, release templates |

RACI assigns exactly one Accountable owner per critical decision,
plus the Responsible (does the work), Consulted (input required),
and Informed (after the fact) roles. **No row may have more than
one Accountable owner.**

## Conventions

* **R** — Responsible: does the work and produces the artifact.
* **A** — Accountable: single sign-off authority. The decision is
  legally / operationally theirs.
* **C** — Consulted: must give input before the decision lands.
* **I** — Informed: notified after the decision lands.
* Blank — no formal RACI role for that area in this decision.

## 1. Schema Approval

Decisions about adding, modifying, or deprecating a field in the
canonical Factor Record schema (`config/schemas/factor_record_v0_*.schema.json`).

| Role                          | RACI |
| ----------------------------- | ---- |
| Platform/Data Lead            | R    |
| CTO                           | A    |
| Climate Methodology Lead      | C    |
| Backend/API Lead              | C    |
| Data Engineering Lead         | C    |
| DevRel / Docs                 | I    |
| SRE Lead                      | I    |
| Compliance/Security Lead      | I    |

Trigger: any change to the schema's `$id`, `properties`, `required`
list, or pattern. Output: bumped `$id` (for breaking changes) +
ADR.

## 2. Source Licensing

Decisions about adding a new upstream source, changing licence
terms, or restricting/expanding redistribution.

| Role                          | RACI |
| ----------------------------- | ---- |
| Compliance/Security Lead      | R    |
| CTO                           | A    |
| Climate Methodology Lead      | C    |
| Data Engineering Lead         | C    |
| Backend/API Lead              | I    |
| Platform/Data Lead            | I    |
| Partner Success Lead          | I    |

Trigger: PR that touches `source_registry.yaml` `licence` field, or
a new source addition. Output: signed source registry entry + (for
new sources) source manifest in the next release.

## 3. Parser Approval

Decisions about merging a new source parser, changing an existing
parser, or pinning a parser version.

| Role                          | RACI |
| ----------------------------- | ---- |
| Data Engineering Lead         | R    |
| Climate Methodology Lead      | A    |
| Platform/Data Lead            | C    |
| Backend/API Lead              | C    |
| SRE Lead                      | I    |

Trigger: PR adding/modifying any `greenlang/factors/ingestion/parsers/*`
module, or changing the `parser_module` / `parser_function` /
`parser_version` fields in `source_registry.yaml`. Output: parser
snapshot test + signed parser version recorded in the next release
manifest.

## 4. Release Approval

Decisions about cutting + publishing a release (alpha, beta, rc,
GA, or hotfix).

| Role                          | RACI |
| ----------------------------- | ---- |
| Platform/Data Lead            | R    |
| CTO                           | A    |
| SRE Lead                      | C    |
| Backend/API Lead              | C    |
| Climate Methodology Lead      | C    |
| Compliance/Security Lead      | C    |
| Partner Success Lead          | I    |
| DevRel / Docs                 | I    |

Trigger: a tagged release candidate. Output: filled
`docs/factors/release-templates/RELEASE_MANIFEST_TEMPLATE.md` +
`ACCEPTANCE_CHECKLIST_TEMPLATE.md` + signature.

## 5. Hotfix Approval

Decisions about emergency fixes that bypass the normal release
cadence (e.g. correcting a published value due to upstream errata,
patching a P1 security issue).

| Role                          | RACI                                              |
| ----------------------------- | ------------------------------------------------- |
| Backend/API Lead OR Data Engineering Lead | R (whichever component is affected)   |
| Platform/Data Lead            | A                                                 |
| CTO                           | C (mandatory for any customer-impacting hotfix)   |
| SRE Lead                      | C                                                 |
| Compliance/Security Lead      | C (for any security-classified hotfix)            |
| Partner Success Lead          | I (must notify affected partners within 24h)      |
| DevRel / Docs                 | I                                                 |

Trigger: SEV-1 or SEV-2 incident requiring out-of-band release.
Output: hotfix release manifest + post-mortem in
`docs/factors/postmortems/` + customer-impact memo (template under
`docs/factors/release-templates/CUSTOMER_IMPACT_TEMPLATE.md`).

## 6. Customer-Impact Communication

Decisions about what to tell affected partners / tenants when
something goes wrong (or right) at the customer-visible boundary.

| Role                          | RACI |
| ----------------------------- | ---- |
| Partner Success Lead          | R    |
| CTO OR Product Lead           | A    |
| SRE Lead                      | C    |
| Backend/API Lead              | C    |
| Climate Methodology Lead      | C    |
| Compliance/Security Lead      | C (when communications touch licensing, security, privacy) |
| DevRel / Docs                 | I    |

Trigger: any customer-visible event — incident, value correction,
deprecation, breaking schema change, source decommissioning.
Output: customer-impact memo using the template; partner-tracker
entries updated.

## Acceptance Criteria

* Every decision area above has exactly one **A**.
* Every named role appears in `OPERATING_MODEL.md`.
* Every release template references the matching RACI row.
* No open governance issue cites "unclear ownership" against any of
  these 6 areas.

## Change-Control

Adding a new RACI row requires CTO sign-off plus updates to this
file AND `OPERATING_MODEL.md`. Removing or transferring an
Accountable assignment requires CTO sign-off + an entry in the
factors changelog (`epics/` updates).
