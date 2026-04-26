# ADR-001: GreenLang Factors FY27–FY31 Product Source of Truth

| Field      | Value                                                              |
| ---------- | ------------------------------------------------------------------ |
| Status     | Accepted                                                           |
| Date       | 2026-04-26                                                         |
| Author     | Engineering (under CTO direction)                                  |
| Supersedes | None                                                               |
| Related    | `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md`, Phase 0 plan, `ADR-002-pack-urn-version-segments.md` |

## Countersign

| Field              | Value |
| ------------------ | ----- |
| Decision owner     | CTO (Akshay) |
| Countersign status | Accepted by CTO-delegated user request in the Codex thread on 2026-04-26 |
| Evidence           | Phase 0 governance closure request: "can you complete this for me" after CTO cleanup summary |
| Follow-up          | Add formal e-signature or initials later if company policy requires a human-signature artifact outside git |

## Context

The GreenLang Factors product has multiple, overlapping plan
artifacts in flight: prior PRDs (`docs/factors/PRD_FY27_*`), execution
plans (`FACTORS_EXECUTION_PLAN.md`), gap audits, and the most recent
CTO-authored documents shipped 2026-04-24/25:

* `Final_GreenLang_Factors.docx`
* `GreenLang_Climate_OS_Final_Product_Definition_Roadmap_FY27_FY31_CTO_Final.docx`

The repo also contains forward-looking code for v0.5–v3.0 features
(GraphQL, ML resolve, billing, OEM, SQL-over-HTTP, real-time grid)
already merged behind the release-profile gate at
`greenlang/factors/release_profile.py`. Without an explicit baseline,
contributors mix scope across the v0.1 Alpha cleanup and later
roadmap work.

## Decision

The two CTO-authored documents listed above, copied verbatim into
`docs/factors/roadmap/` on 2026-04-26 and recorded in
`SOURCE_OF_TRUTH_MANIFEST.md` with their SHA-256 checksums, are the
**authoritative product contract** for GreenLang Factors through FY31
Q1 (v3.0).

Specifically:

1. The 8 release milestones (v0.1, v0.5, v0.9, v1.0, v1.5, v2.0, v2.5,
   v3.0) and their exit criteria, scope, and out-of-scope items in the
   document govern engineering planning. The matching epics live under
   `docs/factors/epics/`.
2. The URN spec (Section 6.1.1) and Factor Record schema (Section 19)
   are normative. The canonical URN parser at
   `greenlang.factors.ontology.urn` is the implementation of record;
   any schema regex elsewhere in the repo MUST stay in lockstep with
   it.
3. The release-profile feature gate
   (`greenlang/factors/release_profile.py`) is the only mechanism for
   exposing v0.5+ surfaces in production builds prior to their
   document-allowed milestone.

## Conflict-Resolution Rule

| Conflict                                                       | Resolution                                                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Repo has feature that document defers to a later release       | Keep code; gate it behind the corresponding release profile. Default production profile is `alpha-v0.1`.  |
| Repo lacks a feature the document requires for a release       | Open a backlog item under the matching epic (`docs/factors/epics/<release>.md`).                          |
| Team wants to change product scope                             | Update the document AND submit a new ADR under `docs/factors/adr/` BEFORE merging code that relies on it. |

## Consequences

* All v0.1 Alpha work is constrained to the 5 alpha-allowed endpoints
  in `release_profile.ALPHA_ALLOWED_PATHS`.
* Future-roadmap code remains in the tree but is invisible in
  `alpha-v0.1` runtime. The release-profile test suite
  (`tests/factors/v0_1_alpha/test_release_profile.py`) enforces this.
* Any breaking change to the URN spec or Factor Record schema requires
  a new schema `$id` and a new ADR.
* The frozen documents are checksummed; a hash drift requires either
  a re-issued document with a new ADR or a revert to the frozen
  contents.

## Linked Decisions

* Pack URN version segment policy is decided in
  `docs/factors/adr/ADR-002-pack-urn-version-segments.md`.
* The legacy `EF:` factor-id alias retains uppercase country codes by
  design (alias schema permits it). The canonical `urn:` field MUST
  be all-lowercase (enforced by
  `tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py`).
