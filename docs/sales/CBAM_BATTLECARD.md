# GreenLang CBAM — Sales Battlecard

> **One-liner.** GreenLang CBAM is the audit-ready, zero-hallucination reporting engine for EU Carbon Border Adjustment Mechanism declarations. Deterministic calculation, signed evidence, and a ledger that survives auditor review.

**Target buyer.** EU importers of covered goods (steel, aluminium, cement, fertilizers, hydrogen, electricity); India-linked exporters with EU OEM customers; CBAM-certified consultants serving both.

**Regulation status (FY27).** Definitive period began **1 January 2026**. Quarterly declarations due within 1 month of each quarter-end. Failure to file: €10–50 per tonne CO₂e unreported, plus loss of "authorised CBAM declarant" status.

---

## Why us — in 10 seconds

| Claim | Proof |
|---|---|
| **Zero hallucination** | All numeric paths are deterministic Python + factor-catalog lookups. The factor DB (`greenlang/factors/data/*.yaml`) ships ~25,600 lines with full source lineage. No LLM is invoked inside a calculation. |
| **Audit-ready evidence** | Every CBAM run writes into the **Evidence Vault** (raw inputs + parser logs + reviewer decisions) and produces a signed ZIP bundle: `gl evidence bundle --case <id>`. |
| **Provenance by default** | Every activity and every emission goes into the **Climate Ledger** with SHA-256 chain hashing + append-only triggers (Postgres + SQLite). Rebuilds the line from CBAM number → Factors edition → Policy Graph decision. |
| **Fast to pilot** | 2-week pilot path with sample EU importer dataset already in the repo (`applications/GL-CBAM-APP/CBAM-Importer-Copilot/examples/`). |
| **Unified with other regs** | The same substrate runs CSRD, SB 253, TCFD, SBTi, ISO 14064, CDP. Customer invests once, adds modules as they need them. |

---

## The ask

- **Per importer:** starting at **$100k/yr** (Professional tier). Volume pricing above 10,000 shipments/year; Enterprise tier for multi-entity group declarations.
- **Per consultant:** white-label tier — ask.
- **Pilot:** 2-week fixed-fee engagement against a real prior-quarter dataset. Outcome: signed CBAM XML + evidence bundle + gap report.

## What ships

- CBAM pack (`packs/eu-compliance/PACK-005-cbam-complete`) — the 3-agent import pipeline (routing → emissions → reporting).
- Comply umbrella orchestrator (`greenlang.comply.ComplyOrchestrator`) — calls Policy Graph to confirm CBAM applicability + triggers CBAM reporting alongside CSRD / SB 253 / GHG Protocol if the entity also matches.
- `gl run cbam ...` CLI for the full import pipeline (config + shipments → XML + audit bundle).
- `gl evidence bundle --case <id>` to produce the auditor package.
- `gl ledger verify <entity>` for chain-integrity audit.
- Kubernetes-ready deployment (Helm charts: `applications/GL-CBAM-APP/deployment/`).

## Where a competitor is weak

- **"Spreadsheet + consultant" incumbents:** manual, re-done every quarter, no traceability between version N and version N+1. GreenLang ledger keeps the whole chain.
- **Black-box SaaS:** can't explain a factor choice to an auditor. GreenLang Evidence Vault returns the raw source PDF + parser log + reviewer decision as a signed ZIP.
- **General ESG tools (Workiva, Sphera, Watershed):** no CBAM-specific XML output, no factor-class labels (Certified / Preview / Connector-only). We are narrow and deep in CBAM; they are wide and shallow.
- **LLM-first startups:** hallucinate factors. We don't. Calculation = deterministic Python; AI is used only for matching free-text descriptions to candidate factors — and every match is human-reviewable before sign-off.

## What a pilot looks like (2 weeks)

| Day | Deliverable |
|---|---|
| Day 0 | Kick-off; customer shares 1 prior quarter's shipments + supplier list. |
| Day 2 | Data loaded. `gl run cbam` produces a first-pass XML + audit folder. |
| Day 4 | Gap report: missing supplier GPS coords, missing CN-code refinements, factor-class mismatches. |
| Day 6 | Remediation; `gl factors ingest-paths` loads any customer-specific factors into a tenant overlay. |
| Day 9 | Final run + evidence bundle + chain-of-custody ledger export. |
| Day 10 | Review with customer's compliance officer + auditor. Sign-off. |
| Day 14 | Handover: runbook (`docs/sales/CBAM_PILOT_RUNBOOK.md`), tenant credentials, support contacts, pricing for Year 1 production. |

## Proof points (repo assets)

- `applications/GL-CBAM-APP/CBAM-Importer-Copilot/` — the 332 MB flagship application (332 MB includes K8s manifests + fixtures + Grafana dashboards).
- `packs/eu-compliance/PACK-004-cbam-readiness` + `PACK-005-cbam-complete` — the installable packs.
- `tests/factors/test_parser_cbam_full.py` — end-to-end parser + rules coverage.
- `docs/sales/CBAM_PILOT_RUNBOOK.md` — the operator runbook.
- Migration `V439` + `V440` + `V441` — Postgres schema for Ledger / Vault / Entity Graph backing a production tenant.

## FAQs

**Q: What EU jurisdictions are in scope?** All 27 member states. Non-EU exporters into EU supply chains are the most common buyer profile today.

**Q: Does this replace our CBAM consultant?** No — it replaces the spreadsheet the consultant maintains. The consultant stays valuable for strategy and auditor relationships.

**Q: How do we integrate with our SAP / Oracle ERP?** `greenlang.connect.SAPS4HanaConnector` ships today. Custom connector work is a ~2-week engagement.

**Q: What if CBAM rules change mid-year?** Rules live in `packs/eu-compliance/PACK-005-cbam-complete/rules/` and are version-pinned. A rules update ships as a pack version bump; old runs remain reproducible from the ledger.

---

*Last updated: 2026-04-20. Owner: GreenLang sales (CBAM). Source: FY27_vs_Reality_Analysis.md §3.4 and §7.4.*
