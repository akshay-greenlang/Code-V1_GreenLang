# GreenLang Factors — Source-of-Truth Governance

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Accepted - binding after Phase 0 closure.                                    |
| Date             | 2026-04-26                                                                  |
| Owner            | CTO (governance); Platform/Data Lead (operational)                          |
| Related          | `product/MASTER_PRD.md`, `engineering/ENGINEERING_CHARTER.md`, ADR-001/002  |

This document declares the **6 authoritative artifacts** for the
GreenLang Factors product. Each artifact has a single owner and a
single canonical location. There is exactly one source of truth per
contract — duplicates are noise.

## 1. The 6 Source-of-Truth Artifacts

| #   | Artifact         | Purpose                                                                    | Canonical location                                                                                | Owner                       |
| --- | ---------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------- |
| 1   | Roadmap artifact | Release scope through FY31; the product contract                           | `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md` (frozen `.docx` + SHA-256) + ADR-001          | CTO                         |
| 2   | Source registry  | Source licensing, cadence, parser, owner per upstream source               | `greenlang/factors/data/source_registry.yaml` (+ accessor `source_registry.py`)                  | Climate Methodology Lead    |
| 3   | Schema registry  | Factor Record schema, Source schema, Pack schema                           | `config/schemas/` (versioned JSON Schema files; `factor_record_v0_*.schema.json` is the alpha-frozen contract) | Platform/Data Lead          |
| 4   | API contract     | OpenAPI spec; endpoint behavior; error envelope; auth contract             | `tests/factors/v0_1_alpha/openapi_alpha_v0_1.json` (snapshot); served at `/openapi.json`         | Backend/API Lead            |
| 5   | SDK contract     | Per-language SDK behavior (Python first; TS / Java / Go later)             | `greenlang/factors/sdk/python/` + per-language SDK readme + SDK contract test                   | DevRel / SDK Owner          |
| 6   | Release checklist | Release exit gates and signoff template                                   | `docs/factors/release-templates/ACCEPTANCE_CHECKLIST_TEMPLATE.md` (filled per release in `release-templates/instances/`) | Platform/Data Lead          |

## 2. Governance Rules

These rules are non-negotiable and enforced by the release pipeline
+ PR review:

1. **Source registry is the only source list.** No source can be
   served to clients unless it exists in
   `greenlang/factors/data/source_registry.yaml` with a valid
   licence + parser entry. The release manifest fails CI if a
   served URN points to a source not in the registry.

2. **Schema registry is the only schema.** No factor can ship
   unless it validates against the schema registry. The
   provenance gate runs on every record at backfill time + at
   release time.

3. **OpenAPI is the only API contract.** No API change ships
   unless OpenAPI is updated. `test_alpha_api_contract.py` pins
   the OpenAPI snapshot for alpha.

4. **SDK contract tests are the only SDK contract.** No SDK
   change ships unless SDK contract tests pass. The Python SDK
   owns the contract; other languages must implement parity
   tests against the same gold corpus.

5. **Release checklist is the only release gate.** No release
   ships without a filled, signed
   `ACCEPTANCE_CHECKLIST_TEMPLATE.md`. A release tagged in git
   without a corresponding checklist instance is treated as a
   policy violation.

6. **Conflict between repo and frozen roadmap → ADR.** If
   engineering needs to deviate from the roadmap document, the
   deviation requires a new ADR under `docs/factors/adr/`. The
   ADR cites the roadmap section it overrides and the CTO
   countersign.

## 3. Change Procedure

Per-artifact change procedure:

| Artifact          | Change procedure                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Roadmap           | New ADR + CTO countersign + bump SHA-256 in `SOURCE_OF_TRUTH_MANIFEST.md`. Old document archived under `docs/factors/roadmap/_archive/`. |
| Source registry   | PR with Climate Methodology + Compliance/Security review. Licence changes require Compliance/Security A-sign per RACI.           |
| Schema registry   | PR with Platform/Data + Climate Methodology + Backend/API review. Schema-breaking changes bump `$id` and trigger a new ADR.      |
| API contract      | PR updates OpenAPI snapshot + endpoint code together. CI rejects PRs where they diverge.                                          |
| SDK contract      | PR updates SDK + contract tests + SDK readme together. SDK version bump follows semver; major bump requires release-council review. |
| Release checklist | PR updates the template; existing filled instances are append-only (immutable history of past releases).                         |

## 4. Relationship Between Artifacts

```
                                 +--------------------+
                                 | Roadmap (1)        |
                                 |  CTO-owned         |
                                 +---------+----------+
                                           |
                                           v
                  +------------------------+-----------------------+
                  |                                                |
                  v                                                v
        +-------------------+                       +---------------------------+
        | Schema reg (3)    |<--------------------- | Source registry (2)       |
        |  Platform/Data    |   factor_id_alias →   |  Climate Methodology      |
        +---------+---------+   source URN          +-------------+-------------+
                  |                                               |
                  | every record validates                       | every parser fetches
                  v                                               v
        +-------------------+                       +---------------------------+
        | API contract (4)  |---- exposes -------- >| Released catalog edition  |
        |  Backend/API      |                       |  Platform/Data            |
        +---------+---------+                       +-------------+-------------+
                  |                                               |
                  | served via                                    | shipped via
                  v                                               v
        +-------------------+                       +---------------------------+
        | SDK contract (5)  |                       | Release checklist (6)     |
        |  DevRel / SDK     |                       |  Platform/Data            |
        +-------------------+                       +---------------------------+
```

## 5. Acceptance Criteria

* All 6 artifacts exist at their canonical location.
* Each has a named owner role and interim named-human owner in
  `operating-model/OPERATING_MODEL.md`.
* The release pipeline rejects releases that bypass any of the 6
  rules in Section 2.
* No two locations claim to be the canonical source for the same
  contract.

## 6. Forward Items

* SDK contract for TS / Java / Go — currently only Python ships;
  other languages start at v0.5.
* Release-template instances directory
  (`docs/factors/release-templates/instances/`) — created in
  Phase 0 deliverables; first instance is the v0.1 alpha
  release manifest filled by Platform/Data Lead at the next
  edition cut.
