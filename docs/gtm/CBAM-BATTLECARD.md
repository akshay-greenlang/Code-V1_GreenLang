# GreenLang CBAM — Sales Battlecard

**For:** Sales, founders, and the first 5-8 FY27 design partners
**Edition tag:** `2027.Q4-material-cbam` (ref. `docs/editions/v1-certified-cutlist.md` §4)
**Owner:** GTM Lead
**Status:** Ready to quote

---

## Problem

The **EU CBAM definitive period began 2026-01-01** under Regulation (EU) 2023/956 + Implementing Regulation (EU) 2023/1773. Quarterly reports are due, and the covered-goods list is now in force: **iron & steel (CN 72, 73), aluminium (CN 76), cement (CN 2523), fertilizers (CN 2808, 3102, 3105), hydrogen (CN 2804.10.00), and grid electricity imports**.

Who feels it first:

- **India-linked manufacturers** exporting CBAM-covered goods to the EU (Tata Steel, JSW, Hindalco, Vedanta, UltraTech, Ambuja, Coromandel — and 200+ mid-market exporters).
- **EU importers** who need embedded-emissions data from their Indian, Turkish, Chinese, and Vietnamese suppliers to fill the Transitional Registry and, from 2026 onwards, buy CBAM certificates.
- **Customs brokers and Big-4 engagements** who are billing €50k-€200k per client per filing cycle with spreadsheet-based manual processes.

Non-compliance penalty: **€10-€50 per tonne CO2e misreported, plus market access blocked** (no import without a declared, verified embedded-emissions number).

## Fit

**Primary buyer:** Indian exporter with an EU supply chain. Sustainability controller or export head. 500-10,000 employees. Already has ISO 14001. Needs CN-code-aligned embedded emissions per shipment, per installation, per quarter — defensible to a Big-4 verifier.

**Secondary buyer:** EU importer needing supplier data. Procurement or ESG lead. Cannot get numbers from suppliers in the format the Transitional Registry wants.

**We replace:**
- Excel + PDF reference tables (IPCC defaults + CEA grid factors + country EF lookups, manually joined).
- Big-4 advisory engagements (€50k-€200k, 6-12 weeks per filing cycle, no data platform handover).
- Point tools that only cover EU CBAM defaults (not EU importer's primary-data workflow, not operator verification).

## Proof (code references)

| Asset | Path | What ships today |
|---|---|---|
| CBAM method pack | `greenlang/factors/method_packs/eu_policy.py` | `EU_CBAM` MethodProfile registered, `allowed_statuses=("certified",)`, `require_verification=True`, biogenic treatment = EXCLUDED per regulation Article 4(2) |
| GL-CBAM-APP | `applications/GL-CBAM-APP/` | 324MB — largest application in the repo; CN-code ingest, embedded-emissions calc, JSON export for EU Transitional Registry |
| Packs | `packs/eu-compliance/PACK-004-cbam-readiness`, `PACK-005-cbam-complete` | Readiness assessment + full declaration workflow |
| Parser | `greenlang/factors/ingestion/parsers/cbam_full.py` | EU CBAM Annex III default values + revised 2024-12 values |
| Slice cutlist | `docs/editions/v1-certified-cutlist.md` §4 | Edition `2027.Q4-material-cbam`, Week 10 promotion, CBAM-routed factors require DQS 85/100, 0.90 gold-eval precision |
| CLI | `gl run cbam` | Canonical operator path |

Zero-hallucination guarantee: every embedded-emissions number carries `fallback_rank` (1-7), SHA-256 content hash, and `/explain` endpoint output showing the 7-step cascade. Auditor opens one URL and sees the full chain.

## Pricing

| SKU | Annual | What is included |
|---|---|---|
| **Pilot** | **$25k (2 weeks, cash-back guarantee)** | Tenant provisioned, 50 activities resolved, first audit bundle, Q-report JSON draft |
| **Annual — single installation** | **$75k** | Up to 3 CN-code groups, 1 installation, quarterly filings, operator portal, audit bundle export, 99.9% SLA |
| **Annual — multi-installation** | **$125k-$200k** | 5-20 installations, multi-plant roll-up, supplier portal for sub-tier data collection, EU importer view, custom factor overlays |
| **CBAM declarant add-on** | **+$36k/yr** | Direct submission workflow to EU Transitional Registry, primary-operator data ingestion, verifier handoff pack (ref. PRD §7.3 `cbam_premium` Enterprise) |

Volume drivers: number of CN codes in scope, number of EU-bound installations, whether supplier primary data is ingested, custom factor overrides.

## Pilot offer — 2 weeks, cash-back guarantee

**Week 1 — Data in.**
- Mon-Tue: kickoff, NDA, tenant provisioned (`pilot/provisioner.py`), CSV template delivered. Customer provides 50 representative activity rows (shipments, fuel, grid, feedstock).
- Wed-Thu: customer engineer sits next to a GreenLang engineer; 50 activities resolved through `/v1/factors/resolve` with `method_profile=eu_cbam`. Any that fail are routed to `suggestion_agent.py` and manually mapped.
- Fri: first audit bundle generated via `POST /v1/audit-bundle` and opened in Excel.

**Week 2 — Sign off.**
- Mon-Tue: methodology review with customer's sustainability controller + designated auditor. Walk through every fallback_rank, every source citation, every Article 4(2) compliance flag.
- Wed: integration demo — Excel export + live API call from the customer's ERP (SAP / Oracle).
- Thu: sign-off discussion.
- Fri: green-light for annual, OR written write-up of blockers and full pilot fee returned. No lock-in.

**Success criteria:** ≥80% top-1 factor match on the 50 activities, 100% explain-coverage, auditor confirms bundle is Big-4 ready.

## Competitive landscape

| Option | Typical cost | Why we win |
|---|---|---|
| **Manual (Excel + PDFs)** | 40-80 hours per cycle | We do it in <2 hours per cycle; version-pinned factors survive an audit, Excel does not |
| **Big-4 advisory** | €50k-€200k per filing cycle, 6-12 weeks | We ship the platform the Big-4 would build if they weren't billing hourly; we are 20-30% of the cost and the customer keeps the tooling |
| **Generic carbon tools** (Watershed, Persefoni, Normative) | $75k-$150k ACV bundled with a Scope 1/2/3 app | They do not ship CN-code-aligned CBAM defaults, they do not carry `factor_status=certified` + `require_verification=True`, and they do not expose the 7-step cascade |
| **Point CBAM tools** (EU-only, regulator-registry integration) | €15k-€40k | They cover only EU importer side; they do not serve Indian exporter's primary-data workflow; they do not include supplier data collection or operator verification |
| **ecoinvent + consultant** | $10k seat + consultant hours | We layer resolution on top of licensed LCI; customers bring their own ecoinvent license or buy our Product Carbon Premium addon |

## Objection handlers

**"We already have Excel + a consultant; why switch?"**
Because CBAM is quarterly, your auditor asks for SHA-256-traceable provenance, and the default-value table changed in December 2024 (and will change again). Every spreadsheet cycle is a rebuild; our platform handles the change detection, re-runs your Q-report against the new defaults, and flags deviations >5% for re-verification. The first three quarterly cycles pay back the license.

**"We only import from India; our suppliers can't give us primary data."**
That is exactly what our supplier portal is built for. The EU importer signs up at $125k/yr. The Indian supplier gets a free co-branded tenant from you; they submit their activity data through the portal; you see resolved embedded emissions in the Transitional Registry format. When your supplier moves from EU default values to primary operator data, your CBAM certificate obligation drops — the ROI is on the importer side, directly.

**"Why not just use ecoinvent or Climatiq?"**
ecoinvent is LCI methodology, not CBAM-compliant defaults. Climatiq does not enforce `allowed_statuses=("certified",) AND require_verification=True` on CBAM-routed queries — meaning they will silently serve you a Preview factor that your auditor will reject. We refuse to serve Preview factors on a CBAM request, by regulation, in code.

## Call to action

Ask for the 2-week pilot by email to **partners@greenlang.io** with:
1. CN codes you export.
2. Number of installations.
3. EU importer counterparty (if supplier-facing).

Tenant provisioned within 48 hours. Week 1 kickoff within 5 business days.

---

*Footer: GreenLang Factors v1.0 Certified Edition, CBAM slice promotion: Week 10. See `docs/editions/v1-certified-cutlist.md` §4 for sign-off criteria. See PRD `docs/product/PRD-FY27-Factors.md` §7 for full pricing architecture.*
