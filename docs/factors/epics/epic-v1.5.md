# Epic: v1.5 (FY28 Q3)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | post-GA (still `ga-v1.0` features by default; new flags) |
| Target quarter   | FY28 Q3                                                  |
| Owner            | ML and Community Lead                                    |
| Status           | Planned                                                  |

## Scope

Per source-of-truth document, v1.5 expands the catalog and tightens
audit primitives.

* Private packs (per-tenant uploads) — read-only, validated through
  the same provenance gate.
* Audit-grade export (FY-locked editions with signed manifests for
  CSRD / SEC climate-rule submissions).
* Source coverage push: +Brazil (PRO-A), +China (CMA), +Japan
  (J-CRP), +Mexico (Conuee), +ASEAN grid factors.
* Calc explainability v2 — per-figure citation lineage all the way
  to the upstream PDF page.

## Out of scope

* Marketplace open to 3rd-party publishers — v2.0.
* Agent-native ingestion — v2.5.

## Deliverables

* Private-pack ingestion API (`/v1/private-packs`) with row-level
  isolation.
* FY-locked edition snapshots with reproducible byte-identical
  manifests.
* +5 country-level source integrations.
* Citation lineage UI in admin console.

## Acceptance criteria

* Private-pack ingestion: 0 cross-tenant data leaks in red-team test
  (1k tenants × 100 packs).
* FY-edition manifest verifies byte-identical across 3 independent
  environments.
* Each new source has ≥ 95% of factors with a working PDF-page
  citation.

## Tickets

* [ ] Private-pack ingestion endpoint + validator.
* [ ] FY-edition manifest format + signer.
* [ ] Country source onboarding × 5.
* [ ] Citation lineage UI.

## Dependencies

* v1.0 GA shipped + 6 months of production seasoning.

## Release risks

* Private-pack ingestion is the highest tenant-isolation risk to
  date — needs full SEC review.
* International source onboarding cadence is bound by methodology-lead
  capacity.
