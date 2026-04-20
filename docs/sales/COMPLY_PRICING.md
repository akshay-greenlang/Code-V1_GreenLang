# GreenLang Comply — Pricing

> **What you buy.** One platform subscription + one or more framework modules (CBAM, CSRD, SB 253, TCFD, SBTi, ISO 14064, CDP, EU Taxonomy). All modules run on the same substrate (Factors + Climate Ledger + Evidence Vault + Entity Graph + Policy Graph + Scope Engine) with one Comply orchestrator on top.

---

## Tier summary

| Tier | Annual | Best for | Frameworks included |
|---|---|---|---|
| **Essentials** | **$100,000** | Single framework, small org | Any **1** of: CBAM, CSRD Starter (PACK-001), SB 253, GHG Protocol + ISO 14064 |
| **Professional** | **$250,000** | Multi-framework mid-market | Up to **5** frameworks; includes CSRD Professional (PACK-002), CBAM Complete (PACK-005), Scope 1/2 complete (PACK-041), Scope 3 starter (PACK-042) |
| **Enterprise** | **$500,000+** | Global group reporting | **All 10 frameworks**: CBAM, CSRD Enterprise (PACK-003), full ESRS coverage (PACK-017), Scope 3 complete (PACK-043), TCFD, SBTi, ISO 14064, CDP, EU Taxonomy, SB 253 |

## Per-framework module pricing (a la carte on top of Essentials tier)

| Framework | Add-on price/year | Primary deliverable |
|---|---|---|
| CBAM (PACK-005) | $50,000 | Quarterly CBAM XML + audit bundle |
| CSRD Starter (PACK-001) | $40,000 | Basic ESRS disclosure; first-wave compliance |
| CSRD Professional (PACK-002) | $75,000 | Sector-specific ESRS + double materiality |
| CSRD Enterprise (PACK-003) | $150,000 | Group-wide CSRD + multi-entity consolidation |
| SB 253 | $35,000 | California Scope 1+2 (Aug 2026) + Scope 3 (2027) |
| TCFD | $30,000 | Scenario analysis + climate risk disclosure |
| SBTi | $25,000 | Target setting + progress tracking |
| ISO 14064 | $25,000 | Verification-ready GHG inventory |
| CDP | $20,000 | Climate disclosure questionnaire automation |
| EU Taxonomy | $40,000 | DNSH screening + alignment ratios |

> **Bundle logic.** Add-ons are discounted when bought together. The Professional tier *replaces* buying 5 add-ons individually (Professional = 33 % cheaper than à la carte for 5 frameworks).

## What's shared across tiers

- Full substrate: Factors API, Entity Graph, Climate Ledger (append-only signed records), Evidence Vault (signed ZIP bundles), Policy Graph (`applies_to()` applicability API), Scope Engine (5 framework adapters).
- `gl comply run <request.json>` orchestrator.
- Hosted multi-tenant deployment (AWS or GCP, EU/US/IN regions).
- SLA 99.5 % (Essentials) / 99.9 % (Professional) / 99.99 % (Enterprise).
- Support: email (Essentials) / priority (Professional) / dedicated TAM (Enterprise).

## What's extra on top

| Add-on | Price | When you'd buy it |
|---|---|---|
| Implementation services | $200/hr; 2-week pilot flat $25k | First production cutover |
| ERP connector customization | ~$20k per connector | Bespoke SAP / Oracle / Workday mapping beyond the 5 shipped connectors |
| Custom factors (tenant overlay) | $5k per bundle/yr | Proprietary emission factors |
| Policy Graph rule extension | $10k per custom regulation | Sector-specific or jurisdiction-specific rule |
| Additional storage beyond base | $10/GB/month (Essentials); $5 (Enterprise) | Full historical retention |

## Discounts

| Condition | Discount |
|---|---|
| 3-year commitment | 15 % off Years 2–3 |
| 5-year commitment | 20 % off |
| Add 2 modules in a single order | 10 % off module price |
| Pilot → production within 30 days | 10 % off Year 1 |
| Non-profit / academic | Contact us |

## Value comparison

**Against a multi-vendor stack.** A typical mid-market Comply customer who would otherwise buy Watershed ($120k) + Workiva ($80k) + consultant fees ($150k/yr) pays **$350k/yr**. GreenLang Professional ($250k) is **~29 % cheaper** while providing a single substrate with auditable evidence and zero hallucination on the calculation path.

**Against Big-4 consultant-only.** Annual CSRD + CBAM engagement quotes regularly exceed **$500k**. GreenLang Professional delivers the tooling in-house; the customer's consultant stays for strategy and auditor work.

## Procurement

- **Commercial.** Net 30, USD, pro-rated monthly after Year 1 start.
- **Data residency.** EU, US, IN regions available.
- **Security.** SOC 2 Type II prep complete (SEC-009). Full AES-256 + TLS 1.3 + HashiCorp Vault secrets.
- **Contracts.** MSA + SOW + DPA + BAA available. White-label for consultancy partners.

## Next steps

1. Identify which frameworks your org is obligated under (`gl policy-graph applies-to ...`).
2. Book a 30-min scoping call to map frameworks → tier + add-ons.
3. 2-week pilot against a real reporting period.
4. Year-1 production — align with your next regulatory deadline.

---

*Last updated: 2026-04-20. Source: `docs/business/pricing_model.md` + `FY27_vs_Reality_Analysis.md` §3.4.*
