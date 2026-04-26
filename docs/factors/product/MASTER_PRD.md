# GreenLang Factors — Master PRD

| Field            | Value                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------- |
| Status           | Approved v1 - Phase 0 closed by delegated CTO/Product/Engineering record               |
| Date             | 2026-04-26                                                                             |
| Owner            | CTO (Akshay) — product-of-record sign-off                                              |
| Editors          | Platform/Data Lead, Backend/API Lead, Climate Methodology Lead                         |
| Source-of-truth  | `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md` (frozen 2026-04-26)                 |
| Related          | ADR-001, ADR-002, `engineering/ENGINEERING_CHARTER.md`, `epics/`                       |
| Approval gate    | Phase 0 exit (`docs/factors/PHASE_0_EXIT_CHECKLIST.md`)                                |

## 1. Product Definition

GreenLang Factors is the **canonical factor operating layer** for
GreenLang and external developers building climate, carbon, energy,
and sustainability software.

It owns and serves, as a single typed primitive across the stack:

* **emission factors** — Scope 1/2/3 CO₂e per activity unit;
* **conversion factors** — fuel calorific values, density,
  net-vs-gross calorific value;
* **grid factors** — location-based and market-based grid intensity
  by jurisdiction, subregion, balancing authority, hourly bidding
  zone;
* **fuel factors** — combustion + upstream WTW for liquids, gas,
  solid fuels;
* **material factors** — embodied carbon for materials and
  intermediate products (steel, cement, polymers, etc.);
* **refrigerant factors** — GWP, leak rates, end-of-life recovery
  factors;
* **freight factors** — tonne-km, vehicle-km, mode-specific
  intensities (GLEC-aligned);
* **land-use factors** — direct LUC, iLUC factors, peatland and
  carbon-stock change;
* **methodology metadata** — GHG Protocol, ISO 14064, GRI, CDP, IPCC,
  CSRD, SEC climate rule references for each factor;
* **canonical URNs** — globally unique, immutable, content-addressed
  ids per CTO doc §6.1.1;
* **factor packs** — versioned, signed bundles of factors by source +
  vintage (e.g. `urn:gl:pack:defra-2025:conversion-factors:v1`);
* **provenance** — upstream source URL, raw artifact hash, parser
  module + commit, ingest timestamp, reviewer and approver
  identities;
* **release manifests** — signed snapshots of catalog editions;
* **source licensing** — per-source licence terms and redistribution
  rules enforced at API serve time;
* **confidence and uncertainty** — per-factor data-quality tags,
  pedigree-matrix scores, uncertainty bands;
* **APIs and SDKs** — REST, GraphQL (v0.9+), Python / TypeScript /
  Java / Go SDKs for retrieval and resolution.

This is an **operating layer**, not a static database. The same
data that backs the public API also backs internal calculations,
audit-trail rendering, and downstream GreenLang products. There is
exactly **one** authoritative factor for any
`(activity, geography, vintage, methodology)` tuple at any point in
time — pinned by URN — and history is preserved as immutable
versions.

### What Factors does NOT own

To prevent scope creep, Factors explicitly does **not** own:

* full Scope 1/2/3 inventory calculation workflows (those live in
  Scope Engine);
* CBAM submission logic, registry filing, or DG TAXUD integration
  (those live in the CBAM application);
* PCF / LCA modeling logic, scenario rollups, allocation rules
  beyond what a factor publication carries (those live in PCF
  Studio);
* evidence storage of customer activity data, invoices, meter
  reads (Evidence Vault);
* legal interpretation of policy applicability, regulator
  filings, audit attestation (Policy Graph + downstream apps);
* carbon-credit registry logic, retirement, MRV verification of
  projects (carbon-credit application);
* customer disclosure filing, board-facing dashboards,
  CSRD/ESRS XBRL packaging (CSRD application).

## 2. Product Boundary (Authoritative)

Factors is the substrate. Other GreenLang products consume it but do
not become it.

| Product                         | Factors role                                                                                              | What Factors does NOT do                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Scope Engine                    | Supplies factor URNs, values, units, GWP basis, provenance for every emissions calculation                | Does not compute corporate inventory aggregation, organizational boundary logic, or rollups    |
| CBAM application                | Supplies the official CBAM default-values pack + verified actual-value packs                              | Does not submit CBAM declarations; does not run quarterly reporting workflows                  |
| PCF Studio                      | Supplies material / process / product factors and methodology metadata                                    | Does not run full LCA modeling, scenario branching, or allocation across products              |
| Policy Graph                    | Links each factor to the regulations / methodologies that authorize or require it                         | Does not decide legal applicability of a policy to a tenant; that is a policy-graph judgement  |
| Evidence Vault                  | References the upstream source artifacts (PDFs, datasets) and signed release manifests                    | Does not store all customer activity data, invoices, primary evidence                          |
| EUDR application                | Supplies deforestation-risk factor packs (commodity × jurisdiction × vintage)                             | Does not run satellite parcel verification, supply-chain mapping, or DDS submission            |
| GL-CSRD-APP                     | Supplies the disclosure-grade factor packs aligned to ESRS E1                                             | Does not compose the XBRL filing                                                               |
| GL-VCCI-APP, GL-CSDDD-APP, ...  | Each consumes the canonical factor primitive                                                              | Does not become each vertical product; never re-implements factor storage                      |
| Future GreenLang products       | Provides canonical factor primitive via stable SDK contract                                               | Does not become each vertical product                                                          |

This boundary is enforced two ways:

1. **Code:** every downstream GreenLang application MUST import
   factors via the `greenlang.factors.sdk.python` SDK — never by
   reading raw seed JSON or re-implementing a parser.
2. **Governance:** new product surfaces that look like calculation
   workflows, submission logic, or evidence stores require a
   product-boundary ADR before they can be added to the Factors
   roadmap.

## 3. User Personas

Phase 0 assumes the following primary personas. Each has a documented
job-to-be-done; SDK / API / docs design is judged against them.

| Persona                             | Job to be done                                                                                       | Primary surface                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Developer / sustainability engineer | "Give me the right factor for this activity, with provenance, in 1 SDK call."                        | Python / TS SDK; `/v1/factors`             |
| Consultant / LCA practitioner       | "Show me all factors for material X across DEFRA / EPA / Ecoinvent and let me pick + cite."          | `/v1/factors` filter + admin console       |
| Enterprise sustainability analyst   | "Resolve my activity description to a defensible factor; show the audit trail."                      | `/v1/resolve` + `/v1/explain` (v0.5+)      |
| Auditor / verifier                  | "Verify the receipt: which factor was used, which source, which edition, which methodology."         | `/v1/factors/{urn}` + signed receipts      |
| Internal GreenLang product team     | "Bind to a factor primitive that doesn't break my downstream contracts when a source ships an edit." | SDK + URN immutability contract            |
| Regulator / public-sector reviewer  | "Show me sources, licences, and how a published value traces back to the regulator publication."     | `/v1/sources` + `/v1/packs` + provenance   |

## 4. Release Scope

Each milestone has its own epic file under `docs/factors/epics/`.

| Milestone | Quarter   | Profile        | Epic                                                |
| --------- | --------- | -------------- | --------------------------------------------------- |
| v0.1 Alpha | FY27 Q1  | `alpha-v0.1`   | [`epic-v0.1-alpha.md`](../epics/epic-v0.1-alpha.md) |
| v0.5 Closed Beta | FY27 Q2 | `beta-v0.5` | [`epic-v0.5-closed-beta.md`](../epics/epic-v0.5-closed-beta.md) |
| v0.9 Public Beta | FY27 Q3 | `rc-v0.9`   | [`epic-v0.9-public-beta.md`](../epics/epic-v0.9-public-beta.md) |
| v1.0 GA    | FY27 Q4  | `ga-v1.0`      | [`epic-v1.0-ga.md`](../epics/epic-v1.0-ga.md)       |
| v1.5       | FY28 Q3  | post-GA        | [`epic-v1.5.md`](../epics/epic-v1.5.md)             |
| v2.0       | FY29 Q2  | post-GA        | [`epic-v2.0.md`](../epics/epic-v2.0.md)             |
| v2.5       | FY30 Q2  | post-GA        | [`epic-v2.5.md`](../epics/epic-v2.5.md)             |
| v3.0       | FY31 Q1  | post-GA        | [`epic-v3.0.md`](../epics/epic-v3.0.md)             |

All scope, out-of-scope, deliverables, acceptance criteria, source
coverage, API/SDK expectations, security gates, owner, dependencies
and risks live in the per-epic file. The PRD is the gating contract;
the epics are the scope-of-record.

## 5. Non-Goals (Explicit)

These are stated in the negative because they are the most common
forms of scope creep the team will encounter from internal and
external stakeholders:

* Factors **is not** Scope Engine. We do not compute organizational
  totals, organizational-boundary logic, or rollups.
* Factors **is not** the CBAM application. We do not file CBAM
  declarations.
* Factors **is not** PCF Studio. We do not run LCA scenarios,
  product-system modeling, or allocation across product trees.
* Factors **is not** a static CSV database. The canonical surface
  is the API + SDK, not flat-file downloads.
* Factors **is not** an autonomous AI factor generator. AI-assisted
  ingestion (v2.5) always lands in a human-reviewed staging gate.
  No factor is published into the canonical catalog without a
  named human reviewer + approver.
* Factors **is not** a commercial-LCA-database reseller without
  licence controls. Every record carries its source licence; the
  API serves licence-aware (Ecoinvent, Sphera GaBi, etc. require
  customer-side proof of upstream entitlement).
* Factors **is not** an evidence vault. We point to upstream
  source artifacts (URLs, S3 hashes); we do not store the entire
  customer evidence corpus.
* Factors **is not** an attestation service. We do not sign
  emissions claims; we sign the catalog edition that a claim
  references.

## 6. Acceptance Criteria for the PRD

This PRD ships only when:

1. CTO has approved Sections 1, 2, and 5 (definition, boundary,
   non-goals).
2. Product lead has approved Sections 3 and 4 (personas, release
   scope).
3. Engineering manager (Factors) has approved Section 6 (this
   section) plus the link-out to `ENGINEERING_CHARTER.md`.
4. Every downstream GreenLang product PM has acknowledged the
   product-boundary row that names their product (Scope Engine,
   CBAM, PCF Studio, Policy Graph, Evidence Vault, EUDR, CSRD,
   VCCI, CSDDD, BuildingBPS, Taxonomy, Green Claims, Product PCF).
5. Every release epic links back to this PRD (verified by a doc
   audit; all 8 epics already include this back-link).
6. No "what is Factors?" ambiguity remains in any open issue or PR
   description.

### Phase 0 Approval Record

| Approver role                 | Named approver / delegate        | Decision | Recorded at |
| ----------------------------- | -------------------------------- | -------- | ----------- |
| CTO                           | Akshay                           | Approved Sections 1, 2, and 5 | 2026-04-26T14:54:26+05:30 |
| Product lead                  | Akshay (interim product owner)   | Approved Sections 3 and 4 | 2026-04-26T14:54:26+05:30 |
| Engineering Manager, Factors  | Akshay (interim engineering delegate) | Approved Section 6 and engineering-charter link | 2026-04-26T14:54:26+05:30 |
| Downstream product PMs        | CTO-delegated Phase 0 acknowledgement | Boundary rows acknowledged for roadmap planning | 2026-04-26T14:54:26+05:30 |

This is a repo-recorded approval based on the delegated Phase 0
closure request. If company policy later requires formal e-signatures,
attach them outside git without changing the product decision.

## 7. Change-Control

Changes to Sections 1, 2, 5 require a new ADR under
`docs/factors/adr/` and a fresh CTO countersign. Changes to Sections
3, 4 require Product + Engineering co-sign. Changes to Section 6
require Engineering Manager (Factors) sign-off only.

## 8. Forward Items

* Non-CBAM family resolver wiring (CBAM stuck at 0% in current
  gold-set evaluation) — tracked under `epic-v0.5-closed-beta.md`.
