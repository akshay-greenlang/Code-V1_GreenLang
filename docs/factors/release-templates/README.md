# GreenLang Factors — Release Templates

This directory holds the **5 release artifact templates** required
by the engineering charter and source-of-truth governance. Every
release leaves behind a filled-in instance under `instances/<release-id>/`.

## Templates

| Template                              | Purpose                                                       | RACI ref     |
| ------------------------------------- | ------------------------------------------------------------- | ------------ |
| `RELEASE_MANIFEST_TEMPLATE.md`        | Identifies the release; signed by Platform/Data + CTO         | #4 Release Approval |
| `SOURCE_MANIFEST_TEMPLATE.md`         | Per-source: licence, version, parser, checksum, reviewer      | #2 Source Licensing |
| `TEST_REPORT_TEMPLATE.md`             | Test results + warnings + skipped + known risks               | #4 Release Approval |
| `ACCEPTANCE_CHECKLIST_TEMPLATE.md`    | Exit-criteria checklist; multi-signoff                        | #4 Release Approval |
| `CUSTOMER_IMPACT_TEMPLATE.md`         | Customer-impact memo (incidents, hotfixes, deprecations)      | #6 Customer-Impact Communication |

## Instance Layout

For each release `<release-id>` (e.g. `factors-v0.1.0-alpha-2026-04-25`):

```
docs/factors/release-templates/instances/<release-id>/
├── RELEASE_MANIFEST.md
├── SOURCE_MANIFEST.md
├── TEST_REPORT.md
├── ACCEPTANCE_CHECKLIST.md
└── CUSTOMER_IMPACT.md           # optional; required only when applicable
```

## Rules

1. Templates are **read-only**. Edits to a template require a PR
   reviewed by Platform/Data Lead.
2. Filled instances are **append-only**. Once a release ships, the
   instance directory is frozen. Corrections happen via a new
   instance referenced from the original.
3. A git tag for a release MUST have a matching instance directory
   on the same commit; CI checks this at tag time.
4. Signoffs are recorded in-line in the filled checklist (name +
   role + ISO timestamp).

## Where existing release artifacts live

The first signed alpha release is at
`releases/factors-v0.1.0-alpha-2026-04-25/` (manifest, manifest
hash, signature placeholder, release notes). Phase 0 wires that
release directory into the new template structure as the v0.1
reference instance.
