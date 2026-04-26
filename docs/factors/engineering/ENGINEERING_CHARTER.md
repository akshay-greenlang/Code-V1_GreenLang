# GreenLang Factors — Engineering Charter

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Approved v1 - Phase 0 closed by delegated CTO + Platform/Data record        |
| Date             | 2026-04-26                                                                  |
| Author           | Engineering (under CTO direction)                                           |
| Related          | `product/MASTER_PRD.md`, ADR-001, ADR-002                                   |
| Enforcement      | Charter rules are enforced via release-checklist gates, CI tests, and ADRs  |

## 1. Engineering Principles (Non-Negotiable)

These principles are the bar every change must clear. PRs that
violate one require a justifying ADR before merge.

1. **Deterministic serving.** Identical request × identical edition
   pin → byte-identical response. No per-instance randomness, no
   wall-clock leakage in payloads, no float instability across
   architectures (use stable formatters, fixed-precision rounding
   only at the API surface).

2. **Provenance-first records.** Every factor record carries
   `extraction.source_url`, `extraction.raw_artifact_sha256`,
   `extraction.parser_id`, `extraction.parser_commit`, `published_at`,
   and `review.approved_by`. Records missing any of these are
   rejected by the provenance gate.

3. **Source-aware licensing.** Every served factor carries a
   `licence` field. Licences with redistribution restrictions
   (Ecoinvent, Sphera GaBi, etc.) are served only to tenants whose
   entitlement record proves upstream licence ownership. The
   licence enforcement happens at the API gateway, not at the
   client.

4. **Read-only public surfaces unless explicitly released.**
   Mutation of the canonical catalog happens through the
   release pipeline (parser → normalizer → provenance gate →
   methodology approve → release manifest → publish), never via a
   public POST. v0.5 introduces signed-receipt POST and
   gold-set submission POST; both are private to specific
   tenant entitlements.

5. **Canonical URNs.** Every public id is a URN
   (`urn:gl:<kind>:...`). Integer primary keys remain internal
   implementation details. Lowercase namespace + id segments per
   the canonical parser at `greenlang.factors.ontology.urn`.

6. **Schema-controlled APIs.** Every request and response is
   validated against the schema registry. Drift between OpenAPI
   and the actual response shape is a release blocker.

7. **No silent fallback geography or methodology.** If the
   resolution engine cannot find an exact match, the response
   either returns no factor (with a structured `not_found` error
   carrying the search vector) or returns a clearly-flagged
   approximate match (`match_quality: approximate`,
   `applied_fallbacks: [...]`). Implicit world-default substitution
   without flagging is forbidden.

8. **No certified factor without citation and source artifact.**
   Every factor traces to a fetchable upstream artifact (URL +
   SHA-256). Hand-typed factors without source attribution are
   rejected by the provenance gate.

9. **Future features must be release-profile gated.** Code for
   v0.5+ surfaces lives in the repo but is hidden by
   `release_profile.feature_enabled`. The default production
   profile is `alpha-v0.1`; future surfaces only mount when the
   release owner explicitly promotes the profile.

10. **Backward compatibility for URNs is forever.** A published URN
    must continue to resolve, with the same value, for as long as
    the source is supported. New editions get new URNs (different
    `vintage` or `version`); they never overwrite published values.

## 2. Architecture Ownership

The codebase is organized around the components below. Each has a
single accountable owner role; the named human owner sits in
`operating-model/OPERATING_MODEL.md`.

| Component                  | Repo path                                       | Accountable role            |
| -------------------------- | ----------------------------------------------- | --------------------------- |
| Source Registry            | `greenlang/factors/data/source_registry.yaml` + `source_registry.py` | Climate Methodology Lead |
| Ingestion Service          | `greenlang/factors/ingestion/`                  | Data Engineering Lead       |
| Parser Service             | `greenlang/factors/ingestion/parsers/`          | Data Engineering Lead       |
| Canonical Normalizer       | `greenlang/factors/etl/`                        | Data Engineering Lead       |
| Quality Engine             | `greenlang/factors/quality/`                    | Climate Methodology Lead    |
| Policy and Method Store    | `greenlang/factors/methodology/` (v0.5+)        | Climate Methodology Lead    |
| Search and Matching Service| `greenlang/factors/resolve/` (v0.5+)            | Backend/API Lead            |
| Release Manager            | `greenlang/factors/release/`                    | Platform/Data Lead          |
| API Gateway                | `greenlang/factors/api_v0_1_alpha_routes.py` + `factors_app.py` + `middleware/` | Backend/API Lead |
| SDK / CLI                  | `greenlang/factors/sdk/`                        | DevRel / SDK Owner          |
| Admin Console              | `greenlang/factors/admin/` (v0.5+)              | Backend/API Lead            |

## 3. Release Profile Rules

The release profile is the single mechanism that controls which
public surface a deployed instance exposes. It is read from
`GL_FACTORS_RELEASE_PROFILE` and defaults to `alpha-v0.1` in any
environment whose `GL_ENV` / `APP_ENV` / `ENVIRONMENT` indicates
production. Local dev defaults to `dev` (everything on).

| Profile      | Public meaning                                        |
| ------------ | ----------------------------------------------------- |
| `alpha-v0.1` | Read-only Alpha; 5 endpoints. Default in production.  |
| `beta-v0.5`  | Closed beta; resolve / explain / batch / coverage / fqs / edition / signed receipts / admin / TS SDK / extended CLI. |
| `rc-v0.9`    | Public beta / release candidate; adds GraphQL + ML resolve. |
| `ga-v1.0`    | GA commercial release; adds billing, OEM, SQL-over-HTTP, commercial packs, real-time grid. |
| `dev`        | Local development only; every feature on; never deploy in production. |

Promotion rules:

* Profile promotion in production requires Platform/Data Lead
  approval and a signed release manifest.
* A `dev`-profile build MUST NOT be promoted to production. The
  release pipeline rejects images whose embedded
  `GL_FACTORS_RELEASE_PROFILE_DEFAULT_BUILD` is `dev`.
* Adding a feature to the `FEATURES` table requires an ADR
  identifying the minimum profile and the expected promotion path.

## 4. Quality Bar

Every release must pass the following gates before tag + publish.
Gate evidence is captured under `docs/factors/release-templates/`
filled-in instances.

| Gate                          | Mechanism                                                                                  | Owner                       |
| ----------------------------- | ------------------------------------------------------------------------------------------ | --------------------------- |
| Schema validation             | `pytest tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py`                   | Platform/Data Lead          |
| Parser snapshot tests         | per-parser fixture-vs-output diff in `tests/factors/v0_1_alpha/test_alpha_*_normalizer.py` | Data Engineering Lead       |
| Provenance completeness tests | `AlphaProvenanceGate.validate` + the corresponding test                                    | Climate Methodology Lead    |
| API contract tests            | `test_alpha_api_contract.py` + OpenAPI snapshot pin                                        | Backend/API Lead            |
| SDK tests                     | `test_sdk_alpha_surface.py` + per-language SDK CI                                          | DevRel / SDK Owner          |
| URN canonical-parse           | `test_seed_urns_canonical_parse.py` (every seed URN passes the canonical parser)           | Platform/Data Lead          |
| Licence enforcement tests     | per-source licence allow-list; tests added with each restricted-licence source             | Compliance/Security Lead    |
| Performance budget            | `test_perf_p95_lookup.py` (p95 ≤ 100 ms list, ≤ 300 ms lookup, ≤ 50 ms healthz)            | SRE Lead                    |
| Release-profile route filter  | `test_alpha_api_contract.py::test_openapi_documents_only_5_alpha_endpoints`                | Backend/API Lead            |
| Release manifest generation   | `scripts/factors_alpha_publish.py --dry-run` succeeds                                      | Platform/Data Lead          |
| Source manifest generation    | per-source manifest emitted by `release/alpha_edition_manifest.py`                          | Platform/Data Lead          |
| Acceptance checklist          | filled `docs/factors/release-templates/ACCEPTANCE_CHECKLIST_TEMPLATE.md` instance          | Platform/Data Lead          |

A release with any red gate is rejected. A release with a yellow
(documented-known-risk) gate requires CTO sign-off in the release
manifest.

## 5. Security and Compliance Baseline

| Control                               | Status (2026-04-26) | Reference                                                |
| ------------------------------------- | ------------------- | -------------------------------------------------------- |
| API keys (per tenant, hashed at rest) | Built (Wave 5)      | `greenlang/factors/middleware/auth_metering.py` + SEC-001 |
| JWT / OAuth                           | Built               | SEC-001                                                  |
| RBAC                                  | Built               | SEC-002                                                  |
| Audit logs                            | Built               | SEC-005                                                  |
| Secrets management (Vault)            | Built               | SEC-006                                                  |
| Source licence enforcement            | v0.1 baseline; tightened in v0.5 with Ecoinvent gating | `source_registry.yaml.licence` |
| Customer data boundaries              | Tenant isolation enforced at API gateway              | INFRA-006 + SEC-002 |
| SOC 2 path                            | Type II target = v1.0 GA                              | SEC-009 |
| Incident response                     | Runbook in place                                       | `docs/factors/runbooks/incident-drill-sop.md` |
| PII detection / redaction             | Built (SEC-011); applied in ML resolve from v0.9      | SEC-011 |
| Supply-chain security (SBOM, sigs)    | Per release; `gl-supply-chain-sentinel` gate           | SEC-007 |
| Penetration test                      | Required before opening public sign-up at v0.9        | Compliance/Security Lead |

The Security and Compliance Baseline is owned by the
Compliance/Security Lead. Promoting a release profile (alpha → beta
→ rc → GA) requires re-confirming the controls above.

## 6. Code, Test, and Doc Conventions

* **Python:** PEP 8; Ruff + Black; type hints required on public
  functions; Pydantic v2 base classes from
  `greenlang.schemas.base.GreenLangBase` (see
  `MEMORY.md` migration log).
* **Tests:** pytest; per-feature directory under `tests/factors/`;
  no test relies on real network for the alpha; perf tests
  emit JSON reports under `out/factors/`.
* **Docs:** Every public module has a module-level docstring; every
  endpoint has an OpenAPI summary + at least one example.
* **Logging:** module-level `logger = logging.getLogger(__name__)`;
  use `%`-format placeholders (already standardised — see
  `MEMORY.md` Logging Standardization).

## 7. Charter Approval

This charter is binding once approved by:

1. CTO (governance)
2. Platform/Data Lead (operational)

Changes require co-signature of both above plus a new ADR.

### Phase 0 Approval Record

| Approver role      | Named approver / delegate      | Decision | Recorded at |
| ------------------ | ------------------------------ | -------- | ----------- |
| CTO                | Akshay                         | Approved governance rules and release-profile discipline | 2026-04-26T14:54:26+05:30 |
| Platform/Data Lead | Akshay (interim platform/data owner) | Approved operational binding through v0.5 handoff | 2026-04-26T14:54:26+05:30 |

This is a repo-recorded approval based on the delegated Phase 0
closure request. Formal e-signature may be attached outside git if
company policy requires it.

## 8. Implementation Follow-Ups

* Wire URN-canonical-parse test into CI (currently runs locally;
  CI integration tracked under `epic-v0.1-alpha.md` Phase 0
  cleanup tickets).
* Decide CI provider for the SDK matrix tests (Python multiple
  versions × OS) — current CI lane is GitHub Actions per
  INFRA-007.
* Wire `gl-supply-chain-sentinel` to the release-publish step.
