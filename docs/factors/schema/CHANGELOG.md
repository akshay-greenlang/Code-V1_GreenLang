# GreenLang Factors - Schema CHANGELOG

> **Append-only**. Newest entries first. Every change to a
> `config/schemas/factor_record_*.schema.json` MUST add a section above the
> existing entries before the PR can merge.

## Format spec (CI-parsed)

Every entry header MUST conform to the strict regex below. The CI gate
(`scripts/ci/check_schema_migration_notes.py`, owned by WS9-B) parses every
top-level `##` line against this regex and FAILS the build if any matched
section header is malformed.

```
^## v(\d+)\.(\d+) - (\d{4}-\d{2}-\d{2}) - (additive|breaking|deprecated|removed)\s*$
```

Rules for header authors:

- Use ASCII hyphens (`-`, U+002D) only. Do NOT use en-dash (U+2013) or
  em-dash (U+2014). The CI regex will reject the latter two.
- Use a single space on either side of each hyphen separator.
- Date is ISO 8601 calendar date, `YYYY-MM-DD`, no timezone, no time.
- Compatibility label is exactly one of `additive`, `breaking`, `deprecated`,
  `removed` - lowercase, no synonyms.
- No trailing punctuation, no parenthetical tags.

A correctly formatted header looks like:

```
## v1.0 - 2026-12-01 - breaking
```

Each entry below the header MUST include, at minimum:

- `schema_id:` - the JSON Schema `$id` URL for the version this entry
  describes.
- A short `Summary` paragraph.
- A `Migration:` block, either inline (for trivial `additive` changes) or
  linked to a migration-note file authored from
  `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md`.
- An `Approvers:` block listing Methodology Lead, CTO, and (for `breaking`
  or `removed`) Legal.

See `docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md` for full rules and §7
for approver requirements.

---

## v1.0 - <DRAFT> - breaking

- schema_id: https://schemas.greenlang.io/factors/factor_record_v1.schema.json
- Status: DRAFT - not yet promoted to frozen. The placeholder header above
  uses the literal string `<DRAFT>` in place of an ISO date so this entry
  WILL fail the CI date regex until the schema is promoted; that is
  intentional. CI is expected to skip `<DRAFT>` entries (the gate's
  responsibility, owned by WS9-B).
- Summary: v1.0 GA contract. v1.0 substitutes its own `$id` distinct from
  `factor_record_v0_1.schema.json`. v0.x records DO NOT satisfy v1.0
  validation per `SCHEMA_EVOLUTION_POLICY.md` §4.
- Migration: To be authored when v1.0 is promoted from draft to frozen. The
  migration note will document the field diff between v0.1 and v1.0, the
  data-backfill plan for alpha records, and the parallel-running window for
  alpha producers.
- Approvers: pending - Methodology Lead, CTO, Legal (required for the
  `breaking` boundary at v0.x -> v1.0).

---

## v0.1 - 2026-04-27 - additive

**schema_id**: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json

**Summary**: Add 5 optional Phase 2 contract fields: `activity_taxonomy_urn`, `confidence`, `created_at`, `updated_at`, `superseded_by_urn`. All optional; no existing record needs migration.

**Migration**: None required. Records that omit these fields validate as before. Records that include them must satisfy the new pattern/range constraints.

**Approver**: pending — methodology lead + CTO sign-off.

---

## v0.1 - 2026-04-25 - additive

- schema_id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
- Summary: Initial freeze of the v0.1 alpha contract. Establishes the
  21-required-field shape consumed by the 6 alpha-source parsers (IPCC AR6,
  DEFRA 2025, EPA GHG Hub 2025, EPA eGRID 2024, India CEA latest, EU CBAM
  defaults).
- Migration: Producers must populate the 21 required fields before publish.
  Legacy `EF:...` identifiers are accepted only via the `factor_id_alias`
  field; the canonical public identifier is `urn`. No prior schema exists,
  so there is no data-migration step from a predecessor - this entry is
  classified `additive` because it adds the schema where none existed.
- Approvers: pending - Methodology Lead, CTO, Legal.
