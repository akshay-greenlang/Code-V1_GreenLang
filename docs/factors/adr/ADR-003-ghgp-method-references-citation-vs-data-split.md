# ADR-003: GHGP Method References — Citation Availability vs Data Shipment Split

| Field      | Value                                                              |
| ---------- | ------------------------------------------------------------------ |
| Status     | Accepted (CTO-delegated, 2026-04-26)                               |
| Date       | 2026-04-26                                                         |
| Author     | Engineering (under CTO direction)                                  |
| Supersedes | None                                                               |
| Related    | ADR-001 (source-of-truth), `epics/epic-v0.1-alpha.md`, `source_registry.yaml` |

## Context

The CTO Phase 1 plan (Section 2 — "Complete source-specific licensing
work") lists **GHGP Product Standard method references** under the
v0.1 Alpha column. The Phase 1 follow-up audit found the registry
entry `ghgp_method_refs` at `release_milestone: v1.0`, not `v0.1` —
flagging an apparent contradiction with the CTO plan.

The contradiction is only apparent. There are two distinct things
the CTO plan can reasonably mean by "GHGP method references":

1. **Citation availability** — every v0.1 factor record carries a
   `methodology_urn` field that may point to a GHGP methodology
   (`urn:gl:methodology:ghgp-corporate-scope1`,
   `urn:gl:methodology:ghgp-corporate-scope2-location`,
   `urn:gl:methodology:ghgp-corporate-scope2-market`). This makes
   GHGP methodology citations **available** to v0.1 catalog
   consumers from day 1.
2. **GHGP source data shipment** — the registry row `ghgp_method_refs`
   describes the upstream source (the GHGP Product Standard PDF
   and the WRI/WBCSD framework text). Shipping this row as an
   ingestible source means we accept GHGP methodology *records*
   as a publishable source — and that triggers separate licence
   handling because GHGP methodology text falls under WRI/WBCSD
   reuse terms (`licence_class: method_only`).

These two are independent. (1) is satisfied by the methodology_urn
in factor records — already shipping in v0.1. (2) is the data
ingestion of GHGP-source rows — deferred to v1.0 when the
methodology pack registry is built out (`epic-v0.5-closed-beta.md`
+ `epic-v1.0-ga.md`).

## Decision

The registry's `ghgp_method_refs` row stays at
`release_milestone: v1.0`. The CTO Phase-1 listing of GHGP under
v0.1 refers to **citation availability via methodology_urn**, which
is satisfied today by every v0.1 factor record. The two are not the
same and conflating them led to the Phase-1 follow-up review
flagging a false contradiction.

Specifically:

* v0.1 factor records cite GHGP via `methodology_urn` strings
  (`urn:gl:methodology:ghgp-corporate-scope2-location`, etc.).
  These citations are validated by the schema's `methodology_urn`
  pattern. No `ghgp_method_refs` row is required for this.
* The `ghgp_method_refs` row in `source_registry.yaml` is what
  governs **ingesting GHGP-published methodology records as a
  source** (so the catalog could surface, e.g., the GHGP Scope 2
  Quality Criteria as factor-pack metadata records). That ingestion
  is scheduled for v1.0 alongside the methodology pack registry.
* The `ghgp_method_refs` row's `licence_class: method_only` and
  `redistribution_class: metadata_only` correctly capture the
  reuse-restricted nature of GHGP framework text — irrespective of
  release milestone.

## Consequences

* No code or registry change is required to satisfy the CTO Phase-1
  v0.1 GHGP citation requirement; it is already satisfied.
* The Phase-1 exit checklist's "GHGP method-reference metadata
  approved for v0.1 citation/metadata use while remaining
  `method_only` / `metadata_only`" line accurately reflects the
  state and is now linked to this ADR for clarity.
* No legal-signoff change is required for v0.1 alpha launch on
  GHGP grounds — the methodology citations carry no upstream
  copyright concern (the methodology URN is a GreenLang-namespace
  identifier; it does not embed GHGP framework text).
* Post-v1.0, when `ghgp_method_refs` graduates from
  `pending_legal_review` to `approved`, the WRI/WBCSD reuse terms
  apply and the existing `method_only` + `metadata_only` semantics
  govern serving (no values returned in bulk; method references
  served as metadata).

## Alternative Rejected

An alternative was to bump `ghgp_method_refs` to
`release_milestone: v0.1` and approve its legal_signoff under the
CTO-delegated pattern used for the other 6 v0.1 sources. That was
rejected because:

* The GHGP framework text is published under WRI/WBCSD reuse
  terms that are more restrictive than the public-domain /
  open-government licences underpinning the other 6 v0.1 sources.
  Approving its legal status under CTO-delegated authority would
  short-circuit the legal review needed to confirm method_only
  serving terms.
* The v0.1 catalog seed contains zero records whose `source_urn`
  is `urn:gl:source:ghgp-method-refs`, so promoting the row to v0.1
  would have no operational effect — only documentation theatre.
* The CTO Phase-1 listing more naturally reads as "GHGP method
  CITATIONS available at v0.1" than "GHGP DATA shipped at v0.1",
  consistent with how DEFRA / EPA / IPCC are listed (those ARE
  shipped as DATA at v0.1 and have records in the catalog seed).

## Linked Decisions

* `source_registry.yaml#ghgp_method_refs` — release_milestone v1.0
  with method_only / metadata_only semantics.
* `epic-v0.1-alpha.md` — v0.1 acceptance criteria.
* `epic-v1.0-ga.md` — methodology pack registry rollout.
* `docs/factors/PHASE_1_EXIT_CHECKLIST.md` — links to this ADR.
