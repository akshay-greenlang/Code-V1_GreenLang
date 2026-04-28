---
schema_id: https://schemas.greenlang.io/factors/factor_record_vX_Y.schema.json
change_class: additive | breaking | deprecated | removed
effective_version: vX.Y
supersedes: https://schemas.greenlang.io/factors/factor_record_vX_PRIOR.schema.json
approver: cto@greenlang.io
methodology_lead_signoff: methodology-lead@greenlang.io
legal_signoff: legal@greenlang.io
effective_date: YYYY-MM-DD
---

# Migration Note: <Short title of the change>

> Replace every `<placeholder>` below before submitting. Delete sections that
> truly do not apply (e.g. "Data Migration Steps" for a pure documentation
> change), but prefer to leave the heading with `n/a` so reviewers can see
> you considered it. The CI gate
> (`scripts/ci/check_schema_migration_notes.py`, owned by WS9-B) reads the
> YAML frontmatter and the CHANGELOG anchor; the body sections are for human
> reviewers.

## Summary

<One paragraph describing the change in plain English. State the schema
version it ships in, the change class, and the customer-visible effect.>

## Motivation

<Why is this change being made now? What user problem, regulator requirement,
or methodology gap does it close? Link to the originating issue, PRD section,
or external reference.>

## Field Diff

<Show the exact JSON Schema diff using a unified-diff code block. Include
file path on both sides. Keep it minimal but complete.>

```diff
--- a/config/schemas/factor_record_vX_PRIOR.schema.json
+++ b/config/schemas/factor_record_vX_Y.schema.json
@@ -<line>,<n> +<line>,<n> @@
   "properties": {
+    "new_optional_field": {
+      "type": "string",
+      "description": "<purpose>"
+    },
     "existing_field": { ... }
   }
```

## Data Migration Steps

<Ordered list of steps required to migrate stored data, if any. For
`additive` changes this is usually `n/a`. For `breaking` changes, list the
exact backfill / transformation / validation steps with idempotency notes.>

1. <Step 1>
2. <Step 2>
3. <Step 3>

## Code Migration Steps

<Ordered list of steps required for producers, consumers, the resolver, the
SDK, the API, and any downstream packs to adopt the new schema. Reference
specific modules under `greenlang/factors/` and `packs/` where relevant.>

1. <Producer-side change>
2. <Resolver/validator update>
3. <SDK release>
4. <API surface change>
5. <Downstream pack update>

## Customer Impact

<Who is affected, how, and what they need to do. Distinguish between alpha
design partners, GA customers under v1.0 lock, and consumers of overlapping
older versions. Cite the CTO lock/overlap windows from
`docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md` §5 and §6.>

- **Alpha (v0.x)**: <impact>
- **GA producers (v1.0)**: <impact>
- **GA consumers (v1.0)**: <impact>
- **Overlap-window users (vN-1)**: <impact>

## Rollback Plan

<Exact steps to revert this change if a regression is detected post-merge.
Include: git revert SHA, schema-file restoration, registry rollback, data
backfill rollback (if applicable), CHANGELOG amendment (mark as
`reverted-YYYY-MM-DD`), and customer communication trigger.>

1. <Revert SHA>
2. <Restore schema file>
3. <Update registry>
4. <Notify affected customers>
