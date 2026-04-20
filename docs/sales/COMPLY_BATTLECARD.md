# GreenLang Comply — Sales Battlecard

> **One-liner.** One auditable substrate for emission factors, activity data, and policy logic — with CBAM, CSRD/ESRS, SB 253, TCFD, SBTi, ISO 14064, CDP, and EU Taxonomy applications on top. Buy once, add modules as regulations hit your org.

**Target buyer.** Mid-market to enterprise with 2+ overlapping climate regulations. Most common: EU-operating groups facing CBAM + CSRD simultaneously; US multinationals facing SB 253 + CSRD (EU subsidiary); Indian exporters facing CSRD cascade + CBAM.

---

## The v3 pitch in 3 bullets

1. **Substrate + Applications, not another report generator.** The Climate Ledger, Evidence Vault, Policy Graph, and Factors catalog underpin *every* module. Buy CBAM, add CSRD later — same entity graph, same evidence, same ledger.
2. **Zero-hallucination by design.** Calculations are deterministic Python + catalog lookups. AI is used only for matching free-text activity descriptions to candidate factors — every match is human-reviewable before sign-off.
3. **One signed audit bundle for all regulations.** `gl comply run` calls Policy Graph → Scope Engine → Evidence Vault → Climate Ledger, and emits a single zip of everything an auditor needs, cross-regulation.

## Why this beats unbundled tools

| Problem with status quo | How Comply fixes it |
|---|---|
| Separate SaaS for CSRD, SBTi, CDP — each re-ingesting the same activity data | One `ComplianceRunRequest` fan-outs across all applicable frameworks |
| Policy logic duplicated inside every pack | `PolicyGraph.applies_to(entity, activity, jurisdiction, date)` — single source of truth |
| Factor numbers not defensible to auditors | Every factor carries license + source lineage + coverage label (Certified / Preview / Connector-only) |
| Evidence scattered across emails, spreadsheets, consultant files | Evidence Vault bundle: raw sources + parser logs + reviewer decisions, signed ZIP, one command |
| No chain of reasoning from final number back to source | Climate Ledger: every activity + every emission + every framework view has a SHA-256 chain hash |

## Who we beat

- **Watershed / Persefoni / Sweep:** strong front-end, black-box calculation path. Our edge: auditor-defensible evidence bundle + Policy Graph.
- **Workiva / Sphera:** good disclosure authoring, weak numeric provenance. Our edge: chain-hashed Climate Ledger + signed bundle.
- **Big-4 consultants:** great relationships, expensive, manual. Our edge: tooling the consultant's team uses, not a substitute for it.
- **LLM-first entrants:** hallucinate factors. Our edge: deterministic calculation, auditable at every step.

## What ships today (FY27)

- `greenlang.comply.ComplyOrchestrator` + `gl comply run` CLI.
- Unified meta-pack (`packs/eu-compliance/PACK-Comply-000-unified/`) composing CSRD (PACK-001..003, PACK-012..017), CBAM (PACK-004, PACK-005), Scope 1/2/3 (PACK-041..043).
- Substrate: Factors (FY27-ready), Climate Ledger (Postgres migration `V439`), Evidence Vault (`V440`), Entity Graph (`V441`), Policy Graph (CBAM + CSRD + SB-253 + TCFD + GHG Protocol rules), Scope Engine (5 framework adapters), Connect (SAP + Workday + Snowflake + Databricks + AWS).
- Applications: GL-CBAM-APP (flagship, 332 MB), GL-CSRD-APP, GL-SB253-APP, GL-SBTi-APP, GL-TCFD-APP, GL-ISO14064-APP, GL-CDP-APP, GL-Taxonomy-APP.

## Proof points (repo assets)

- `greenlang/comply/orchestrator.py` — end-to-end pipeline
- `tests/comply/test_orchestrator.py` — 9 passing tests exercising the orchestration
- `docs/sales/CBAM_BATTLECARD.md`, `docs/sales/CBAM_PILOT_RUNBOOK.md` — CBAM-specific companion
- `docs/sales/COMPLY_PRICING.md` — tier + module pricing
- `packs/eu-compliance/PACK-Comply-000-unified/pack.yaml` — the meta-pack manifest

## Typical pilot (4 weeks)

1. **Week 1 — applicability.** Run `gl policy-graph applies-to` against the customer's entity profile. Output: the exact set of regulations in scope and deadlines per regulation. Scoping conversation anchored in facts, not hypothesis.
2. **Week 2 — single-framework.** Pilot against one regulation (typically CBAM if EU importer, CSRD if EU operator, SB 253 if CA > $1B revenue). Full `gl comply run` with real data.
3. **Week 3 — multi-framework.** Add a second regulation. Demonstrate the substrate reuse — same entities, same activities, same evidence pool.
4. **Week 4 — handover.** Production tenant provisioned, runbook captured, signed Year-1 contract.

## Pricing

- **Essentials $100k/yr** — 1 framework
- **Professional $250k/yr** — up to 5 frameworks (most common)
- **Enterprise $500k+/yr** — all 10 frameworks + multi-entity consolidation

See `docs/sales/COMPLY_PRICING.md` for the a-la-carte module price list and discount schedule.

## FAQs

**Q: Do we have to use your Factors catalog?** No. Tenant overlays let you override any factor with your own attested or licensed data. Catalog factors are used only when you don't override.

**Q: How does this compare to buying our CSRD tool separately?** Buying CSRD alone costs ~$80–150k at competitors. GreenLang CSRD Professional is $75k — and if you add any second regulation (CBAM, SBTi, TCFD) you're already cheaper than buying both separately.

**Q: What's the auditor story?** The Evidence Vault bundle is the auditor package. It contains raw inputs + parser logs + reviewer decisions + signed ZIP manifest. Many customers run it past their external auditor during the pilot before signing.

**Q: Can we self-host?** Yes, the Helm charts ship in `deployment/helm/`. Self-hosted pricing is a separate SKU — ask.

---

*Last updated: 2026-04-20. Owner: GreenLang sales (Comply). Complement: `docs/sales/COMPLY_PRICING.md`, `docs/sales/CBAM_BATTLECARD.md`, `docs/sales/SCOPE_ENGINE_BATTLECARD.md`.*
