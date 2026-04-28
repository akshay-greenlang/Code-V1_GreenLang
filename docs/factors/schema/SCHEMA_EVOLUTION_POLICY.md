# GreenLang Factors - Schema Evolution Policy

> **Status**: Draft (pending CTO countersign 2026-04-27).
> **Authority**: CTO Phase 2 brief, section 2.6 (Schema Evolution).
> **Owner**: GL-SpecGuardian + GL-TechWriter.
> **Single source of truth for schema-evolution rules**. Schema files MUST NOT
> embed policy text; they MUST back-reference this document by URL. Effective
> dates and computed lock/overlap windows live in the version registry
> (`greenlang/factors/schemas/_version_registry.py`); this document is
> authoritative for *rules*, the registry is authoritative for *dates*.

---

## §1 Schema versioning ladder

GreenLang Factors schema versions follow a strict ladder. Each version has a
distinct JSON Schema `$id` and is **not substitutable** for any other version
(see §4). Lock and overlap windows are mandated by the CTO and are computed
from the version's `effective_date` in the registry.

| Version | Scope of change permitted | Lock duration (no breaking changes) | Overlap with next major | Deprecation policy | Customer migration window |
|---------|---------------------------|--------------------------------------|--------------------------|--------------------|----------------------------|
| **v0.x** (alpha) | Additive within minor; breaking allowed across minor (v0.1 -> v0.2) with migration notes + Methodology Lead approval | None (alpha) | None | Field marked `deprecated`; removal allowed in next minor | Best-effort; alpha customers under design-partner agreement |
| **v1.0** (GA) | **Locked for 24 months from GA effective date.** Only bug fixes that do not alter the public contract are permitted. Optional fields may be added only if classified `additive` and approved by both Methodology Lead and CTO. | **24 months** | **12 months** with v2.0 | Deprecated fields remain readable for the full 24-month lock plus the 12-month overlap | 36 months end-to-end (24-month lock + 12-month overlap) |
| **v2.0** | Major schema step. Breaking changes permitted at boundary. New `$id`. | (Per separate review at v2 GA) | **18 months** with v3.0 | Deprecated v1 fields remain readable during the 12-month v1<->v2 overlap, then removed | 12 months v1<->v2 overlap |
| **v3.0** | Major schema step. Breaking changes permitted at boundary. New `$id`. | (Per separate review at v3 GA) | (Per separate review at v3 GA) | Deprecated v2 fields remain readable during the 18-month v2<->v3 overlap, then removed | 18 months v2<->v3 overlap |

The version registry computes `lock_until` and `overlap_until` from the
`effective_date` so customers and CI gates always have machine-readable
expiry dates.

---

## §2 Change classification

Every change to a `factor_record_*.schema.json` MUST be classified into
exactly one of four categories. The classification is recorded in the
CHANGELOG header and consumed by the CI gate.

### `additive`

A change that **does not break** any conformant producer or consumer of the
prior schema. Concretely:

- Adding a new **optional** field (no `required` impact).
- Widening a `pattern` to a provable superset of the prior pattern.
- Widening a numeric `minimum`/`maximum` outward.
- Adding a new `enum` value (consumers MAY treat unknown values as opaque).
- Adding a new `$defs` entry that is only referenced by new optional fields.
- Adding documentation in `description` fields.

### `breaking`

A change that **invalidates** at least one previously-conformant record OR
that requires a previously-conformant consumer to alter its behaviour.
Concretely:

- Adding a new **required** field.
- Removing a field (also classified `removed`; see below).
- Tightening a `pattern` (allowed range strictly shrinks).
- Tightening `minimum`/`maximum` inward.
- Removing an `enum` value.
- Renaming a field.
- Changing a field's type (e.g. `string` -> `number`).
- Changing `additionalProperties` from `true` to `false`.

A `pattern` change that is provably a strict superset of the prior pattern is
classified `additive`, not `breaking`. The PR description MUST include the
proof (regex superset argument or property-test evidence).

### `deprecated`

A field is marked deprecated but still present in the schema. Concretely:

- Setting `"deprecated": true` on a property (JSON Schema 2020-12 supports
  this keyword).
- Adding a `deprecated_since` annotation in the field `description`.
- Updating the field reference doc with a deprecation banner.

A `deprecated` change is non-breaking and may ship at any time within a lock
window. The field MUST remain functionally equivalent for the duration of
the lock + overlap windows.

### `removed`

A field that was previously `deprecated` is eliminated from the schema.
Removal is always classified `removed` (and is, by definition, also
breaking; the CI gate accepts `removed` and treats it equivalently to
`breaking` for approval-gate purposes).

A field MUST have shipped as `deprecated` in at least one prior schema
version before it can be `removed`. Direct field removal without prior
deprecation is forbidden.

---

## §3 Migration note requirements

Every PR that modifies a `config/schemas/factor_record_*.schema.json` file
MUST:

1. Append a CHANGELOG entry to `docs/factors/schema/CHANGELOG.md` whose
   header conforms to the strict regex specified at the top of that file
   (newest entries first; append-only).
2. Include or link a migration note authored from
   `docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md`. For trivial `additive`
   changes (e.g. adding a single optional field with documentation), a short
   inline note inside the CHANGELOG entry is acceptable in lieu of a
   separate migration-note file.
3. Identify approvers per §7.
4. Link to the version registry entry (or open a registry PR adding one for
   a new `$id`).

The CI gate (`scripts/ci/check_schema_migration_notes.py`, owned by WS9-B)
diffs any `config/schemas/factor_record_*.schema.json` against the merge
base and FAILS the build if no matching CHANGELOG entry with a valid
compatibility label exists.

---

## §4 Substitutability rules

This is the cornerstone rule of the policy.

> **v0.x records DO NOT satisfy v1.0 schema, and v1.0 records DO NOT satisfy
> v0.x schema.** Each major version has its own JSON Schema `$id`. Every
> persisted record carries a `$schema` declaration (or an equivalent server-
> side annotation) that identifies the schema it was produced against. The
> resolver dispatches validation and field interpretation on that declared
> `$schema`. There is no implicit upcasting and no implicit downcasting.

Concrete consequences:

- A v0.1 record loaded into a v1.0 store is rejected at the gate.
- A v1.0 record served to a v0.1 SDK is wrapped in a v0.1-compatible
  projection if and only if a registered `supersedes` projection exists;
  otherwise the SDK receives a `406 Not Acceptable` style error.
- The `$id` of `factor_record_v0_1.schema.json` is permanent for the
  lifetime of any v0.1 record in the catalog. It is never reissued.
- Every breaking change requires a new `$id` (with a new version segment in
  the URL), a new registry entry, and a new CHANGELOG header.
- The resolver MUST NOT silently accept a record whose `$schema` declaration
  refers to an unknown `$id`. Unknown `$id` is a hard failure.

This rule supersedes any apparent backwards-compatibility logic that may
appear elsewhere in the codebase. If you find code that bypasses it, file a
P0 bug and link this section.

---

## §5 Lock obligations

During the 24-month lock window after v1.0 GA effective date:

- **No `breaking` change is allowed** in `factor_record_v1.schema.json`,
  full stop. PRs proposing one MUST be rejected by the CI gate.
- `additive` changes are permitted, subject to §7 approval, on the
  understanding that adding optional fields does not break v1.0
  conformance for existing producers or consumers.
- `deprecated` changes are permitted; the deprecated field MUST remain
  functionally equivalent for the full 24-month lock plus the subsequent
  12-month v1<->v2 overlap window (36 months total).
- `removed` changes are FORBIDDEN within the lock window. Removal can only
  happen at v2.0 boundary.
- Bug fixes that do not alter the public contract (e.g. typo in a
  `description`, tightening a JSON pointer that was always intended) are
  permitted without `breaking` classification, but MUST still be recorded
  in the CHANGELOG as `additive` with a `bug-fix` annotation.

The lock end date for any version is `effective_date + lock_months`,
computed by the registry's `lock_until` property.

---

## §6 Overlap window protocol

When v2.0 ships, v1.0 schema validation MUST remain functional for **12
months** from the v2.0 effective date. When v3.0 ships, v2.0 validation MUST
remain functional for **18 months**.

Operational rules during overlap:

- The resolver continues to dispatch on each record's declared `$schema`.
- Both schema validators are loaded; both `$id`s are advertised by the
  schema-discovery API.
- Old data is **read-only** after the prior version's lock window expires.
  Producers cannot publish new records against the deprecated version.
- Consumers of the deprecated version receive an HTTP `Deprecation` header
  (per RFC 9745) on every response, with a `Sunset` header set to the
  overlap-window end date.
- After the overlap window expires:
  - The deprecated `$id` remains resolvable for archival reads.
  - New publish attempts against the deprecated `$id` return `410 Gone`.
  - The version registry status flips to `removed`.

The overlap end date is `effective_date_of_next_major + overlap_months`,
computed by the registry's `overlap_until` property on the *prior* version
once the next major is registered.

---

## §7 Approval requirements

Every change to a `factor_record_*.schema.json` requires sign-off from:

- **Methodology Lead** - confirms scientific/methodological correctness.
- **CTO** - confirms architectural and contract correctness.

Additionally, every `breaking` and `removed` change requires:

- **Legal sign-off** - confirms no contractual or licence obligation is
  violated by the change (e.g. a deprecated source field tied to a
  licensor's attribution clause).

Approvals are recorded in:

1. The migration-note YAML frontmatter (`approver`, `methodology_lead_signoff`,
   `legal_signoff`).
2. The CHANGELOG entry (final lines listing approvers by name and date).
3. The PR's GitHub review trail.

The CI gate validates that the migration-note frontmatter matches the
classification: `breaking` or `removed` without `legal_signoff` set to a
non-`n/a` value FAILS.

---

## §8 Single-source-of-truth rule

To prevent policy drift, the following invariants are enforced:

- Policy text (rules, classifications, lock-window definitions, overlap
  rules, approval requirements) lives **only** in this document.
- Schema `description` fields MUST NOT embed policy text. They MAY include
  a back-reference URL to this document.
- Effective dates, lock-end dates, and overlap-end dates live **only** in
  `greenlang/factors/schemas/_version_registry.py`. The registry is
  authoritative.
- The CHANGELOG records *what* changed and *when*; the registry computes
  *until-when* the change holds. The registry's date arithmetic is the
  single source of truth for time-bound assertions.
- Tests, CI gates, and runtime code MUST import dates from the registry,
  not parse them from documentation.

If you need to reference a policy rule from code, link to the section
anchor in this document (e.g. `SCHEMA_EVOLUTION_POLICY.md#4-substitutability-rules`).
If you need a date, call the registry.

---

## Revision history

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-04-27 | 0.1-draft | GL-TechWriter | Initial draft per CTO Phase 2 brief §2.6. Awaiting CTO countersign. |
