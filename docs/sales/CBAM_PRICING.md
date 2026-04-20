# GreenLang CBAM — Pricing

> **Who this is for:** EU importers of CBAM-covered goods, their consultants, and India-linked exporters selling into EU supply chains.
> **FY27 regulatory trigger:** CBAM definitive period is live as of 1 January 2026. Quarterly declarations due within 1 month of quarter-end.

---

## Tier summary

| Tier | Annual | Best for | CBAM-specific inclusions |
|---|---|---|---|
| **Essentials** | **$100,000** | SMB importers (1–2 EU member states, ≤ 2,500 shipments/quarter) | CBAM pack (PACK-005), standard factor catalog (Certified + Preview), standard SLA 99.5% |
| **Professional** | **$250,000** | Mid-market (multiple EU countries, ≤ 10,000 shipments/quarter, group declaration) | Everything in Essentials + multi-entity group declaration + tenant factor overlay + 99.9% SLA + priority support |
| **Enterprise** | **$500,000+** | Global enterprises, group-wide CBAM + CSRD + SBTi coverage | Everything in Professional + unlimited entities/shipments + dedicated implementation + 99.99% SLA + white-glove onboarding |

All tiers include:

- CBAM pack (`PACK-005-cbam-complete`) with quarterly XML output + audit bundle
- Policy Graph applicability (`gl policy-graph applies-to`) across CBAM + adjacent regulations
- Evidence Vault signed ZIP bundles + Climate Ledger chain-of-custody
- Hosted factor API reads against the Certified + Preview tiers
- `gl run cbam` flagship CLI + production K8s deployment guide

## What's extra on top

| Add-on | Price | When you'd buy it |
|---|---|---|
| **Implementation services** | $200/hr or flat $25k for 2-week pilot | First production cutover |
| **ERP connector customization** | ~$20k per connector | Customer has SAP / Oracle / Workday but needs a bespoke OData / BAPI mapping |
| **Custom factors** | $5k per factor bundle per year | Customer has proprietary emission factors (in-house LCA, supplier attestation) that must override catalog defaults |
| **Extra API calls** (Essentials only) | $0.01 per 1,000 calls | Automation beyond the 100k/month included |
| **Extra storage** | $10/GB/month (Essentials); $8 (Pro); $5 (Enterprise) | Full historical evidence retention |

## Discounts

| Condition | Discount |
|---|---|
| Multi-year commitment (3 years) | 15 % off Years 2–3 |
| Pilot-to-production conversion within 30 days | 10 % off Year 1 |
| Consultant/reseller white-label (with > 5 end-clients) | Custom |

## Comparison against alternatives

| Approach | Annual cost to customer | GreenLang CBAM savings | Why |
|---|---|---|---|
| **Spreadsheet + in-house analyst** | $60–120k analyst salary + $0 tool | Break-even Year 1, **$60k+ saved** by Year 2 | Analyst reallocated to strategy, not data entry; no more manual rework per quarter |
| **Big-4 / consultant-only** | $250–500k recurring fees | **$150–400k/yr** | Ownership of the tool shifts in-house; consultant stays for strategy + auditor relationships |
| **Generic ESG SaaS** (Watershed, Persefoni, Sphera) | $80–300k | Narrower-better-fit **and** part of an integrated substrate | CBAM XML output is first-class, not a feature on top |
| **LLM-first startup** | Varies | Value: **zero hallucination guarantee** | Our calculation path has no LLM; auditor-safe |

## Procurement

- **Commercial terms.** Net 30, USD, pro-rated monthly after Year 1 start.
- **Data residency.** EU, US, IN regions available (Postgres + S3-compatible).
- **Security.** SOC 2 Type II prep complete (see `SEC-009` in platform infra). Full encryption at rest (AES-256) and in transit (TLS 1.3).
- **Contract templates.** MSA + SOW + DPA + BAA available.

## Next steps

1. Book a 30-min pilot scoping call.
2. Send sample quarter's shipment CSV + supplier YAML (see `docs/sales/CBAM_PILOT_RUNBOOK.md` §1 for format).
3. 2-week fixed-fee pilot against your prior-quarter data.
4. Year-1 production starting next CBAM deadline.

---

*Last updated: 2026-04-20. Source: `docs/business/pricing_model.md` + `FY27_vs_Reality_Analysis.md` §5.1.*
