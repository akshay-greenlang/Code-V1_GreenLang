# GreenLang Factors API — Pricing Sheet

**For:** Sales, founders, developer marketing
**Edition:** FY27 v1.0 (all 7 method packs, 4 tiers, 7 Premium Data Pack SKUs)
**Source of truth:** `docs/product/PRD-FY27-Factors.md` §7, `docs/editions/v1-certified-cutlist.md` §§1-7

---

## The four tiers

| Dimension | **Community** | **Developer Pro** | **Consulting / Platform** | **Enterprise** |
|---|---|---|---|---|
| **Price** | Free | $49 / $199 / $499 per month (3 sub-tiers) | $25k-$75k per year + usage | $100k-$300k ACV per year |
| **API calls / month** | 1,000 | 25k / 100k / 500k | 1M / 5M | 10M+ (negotiated) |
| **Batch items / month** | 1,000 | 25k / 100k / 500k | 2M | 10M+ |
| **Factor visibility** | Certified only | Certified + Preview | Certified + Preview | All incl. Connector-only |
| **Premium packs** | None | Per-pack addon ($99-$999/mo) | 3 packs included | All 7 packs addon-available |
| **Overlay / private-registry entries** | 0 | 50 per project | Multi-client sub-tenants | Unlimited + 4-eyes approval + SCIM |
| **Audit bundle export** | No | No | Yes (per-run) | Yes (per-run, signed) |
| **Bulk export rows** | None | 5,000 | 50,000 | 1M+ |
| **Signed receipts** | HMAC | HMAC + Ed25519 opt-in | Ed25519 | Ed25519 + customer-managed keys |
| **SLA** | None | 99.5% | 99.9% (5% credit on miss) | 99.95% (10%/25% credits) |
| **Support** | Community forum | Email, business hours | Email + Slack, 1 BD | Named CSM, 4h P1, 24x7 |
| **SSO / SCIM** | No | No | $500/mo addon | Included |
| **VPC peering / EU data residency** | No | No | No | Yes |
| **OEM white-label** | No | No | Included | $50k+ addon |
| **Target customer** | Solo dev, student, prototype | Climate startup eng, in-house consultant tech lead | 5-50 person ESG boutique, Big-4 climate practice | Fortune 1000, CBAM-obligated exporter, bank, OEM platform vendor |

Upgrade path: **Community → Pro ($49 or $199 or $499) → Consulting ($25k-$75k) → Enterprise ($100k-$300k) → OEM addon (+$50k)**. Each step adds pack entitlements, overlay slots, SLA, and audit-export scope. Self-service to Pro; annual MSA from Consulting up.

---

## Premium Data Pack SKUs (add to any tier)

| SKU | Pro addon | Enterprise addon | Cutlist reference |
|---|---|---|---|
| `electricity_premium` | **$99/mo** | $12k/yr | `docs/editions/v1-certified-cutlist.md` §1 — eGRID, AIB, DESNZ, Green-e residual, India CEA, METI, NGA + Electricity Maps connector |
| `combustion_premium` | **$149/mo** | $18k/yr | §2 — IPCC + EPA Hub + DESNZ + TCR + India CEA combustion; LHV/HHV + oxidation + fossil/biogenic split |
| `freight_premium` | **$199/mo** | $18k/yr | §3 — ISO 14083 + GLEC v3.0 + 6 shipping corridors + refrigeration uplift |
| `material_cbam_premium` | **$299/mo** | $36k/yr | §4 — EU CBAM defaults (CN 72/73/76/2523/2808/3102/3105/2804.10) + EC3 EPD + PACT |
| `land_premium` | **$149/mo** | $18k/yr | §5 — LSR land-use + active removals + permanence class (Premium-only at launch; no free tier) |
| `product_carbon_premium` | **$499/mo** | $40k/yr (+ ecoinvent license chain) | §6 — ecoinvent v3.11 connector (~50k activities) + EF 3.1 + PACT + PEF/OEF variants |
| `finance_premium` | **$299/mo** | $36k/yr (+ PCAF license chain) | §7 — PCAF asset-class proxies + NACE/GICS/NAICS cross-map + EXIOBASE/CEDA/SUSEEIO connectors |

Stacking rule: any combination of packs is supported. Licensing classes **cannot mix in one response** (CTO non-negotiable #4 enforced by `enforce_license_class_homogeneity()` in `greenlang/data/canonical_v2.py`); a cross-class query returns one response per class with explicit labels.

---

## Usage calculator — three example quotes

### Example 1 — Solo developer, open-source climate-dashboard project
- **Volume:** 500 API calls/month, no private overlays, US eGRID + IPCC defaults only.
- **Plan:** Community tier (free). Attribution required.
- **Year 1 cost:** **$0.**
- Upgrade trigger: first paid customer, or need of Preview factors.

### Example 2 — 200-employee consultancy running 20 client pilots per year
- **Volume:** 20 client sub-tenants, each ~50k calls/year = 1M calls/year, 200 private overlays (10 per client), quarterly audit bundles per client.
- **Plan:** Consulting tier at $50k/yr (mid-point). Add `freight_premium` + `material_cbam_premium`.
- **Line items:**
  - Consulting base: $50,000
  - `freight_premium` included (1 of 3)
  - `material_cbam_premium` included (1 of 3)
  - `electricity_premium` included (1 of 3)
  - Overage (>1M calls): none at this scale
  - SSO addon (Okta): $6,000
- **Year 1 cost:** **~$56,000** for 20 clients = **$2,800 per client** vs. €50k Big-4 per client. Margin absorbs per-client effort.

### Example 3 — Fortune-500 Indian steel exporter with 6 CBAM installations
- **Volume:** 6 installations, ~3M resolve calls/year (quarterly re-runs of ~50k shipments), 500 supplier overlays, monthly audit bundles, SSO + VPC peering + EU data residency, OEM branding for supplier portal.
- **Plan:** Enterprise at $200k ACV + Premium packs + CBAM declarant addon.
- **Line items:**
  - Enterprise base: $200,000
  - `material_cbam_premium`: $36,000
  - `combustion_premium`: $18,000
  - `electricity_premium`: $12,000
  - `freight_premium`: $18,000
  - CBAM declarant addon: $36,000
  - OEM white-label supplier portal: $50,000
- **Year 1 ACV:** **$370,000.**
- Comparable Watershed + Persefoni + Big-4 stack: ~$500k-$700k per year with no substrate handover.

---

## Competitive positioning

| Competitor | Their price | Their weakness | GreenLang positioning |
|---|---|---|---|
| **ecoinvent** | ~$10,000 per seat for the LCI library | No resolution layer, no API, no method packs, no policy alignment | We do not sell raw ecoinvent. We sell **resolution on top**. Customers bring their own ecoinvent license (or buy the Product Carbon Premium with OEM arrangement) and we turn their 50,000 LCI activities into auditable resolved factors for CBAM / CSRD / Scope 3 |
| **Climatiq** | $0.01-$0.10 per API call (~$100-$1,000 per 10k calls) | No 7-step cascade, no `/explain`, no audit bundle, no method packs (they sell a factor lookup, not a compliance-ready answer), no Certified/Preview/Connector-only label discipline | At our **Pro $499/mo tier (500k calls)** the effective per-call cost is **$0.001** — we beat them by 10-100x. At Consulting/Enterprise volumes we lead on explainability + method packs + signed receipts — the pieces auditors actually need |
| **Watershed** | $100k+ ACV (bundled with their Scope 1/2/3 app) | Proprietary factor library, no version pinning visible to customers, not API-first, not a substrate | We are the **substrate under** Watershed-class apps. Watershed-tier customers buy our Enterprise tier at $150-$250k with 2-3 Premium packs and keep their own app or build one. Or they OEM-embed us |
| **Persefoni** | $75k-$100k+ ACV (bundled) | Same as Watershed | Same positioning. An OEM deal ($50k addon) turns them from a competitor into a reseller |
| **Sweep, Normative** | $40k-$80k ACV | Smaller factor libraries (10-15 sources), no CBAM-specific compliance, no open-core motion | Our Community tier + Pro competes for the developer audience; our Consulting tier competes for the ESG consultant audience; they are not in the Enterprise conversation |
| **Open-source (Open Supply Hub, OpenLCA, Climate TRACE)** | Free | No SLA, no enterprise support, no audit bundle, no version pinning, no method packs | Our Community tier matches their openness. We layer Certified labels, `/explain`, editions, audit bundles, and SLA on top for paying customers |

---

## Upgrade paths (one-liners)

- **Community → Pro ($49)** when you need Preview factors, >1,000 calls/mo, or Excel export.
- **Pro $49 → Pro $199** when you cross 25,000 calls/mo or you need 50 overlays per project.
- **Pro $499 → Consulting** when you have 3+ clients under management and want multi-sub-tenant isolation + audit bundles.
- **Consulting → Enterprise** when you need Connector-only factors (ecoinvent, IEA, EXIOBASE), SCIM, VPC, >5M calls/mo, or 99.95% SLA.
- **Enterprise → Enterprise + OEM** when you want to white-label our Factor Explorer or route API through your domain.

---

## How to buy

- **Community:** sign up at `developers.greenlang.io`, click-through terms.
- **Pro ($49/$199/$499):** self-service at `developers.greenlang.io/pricing`, Stripe checkout, live in 2 minutes.
- **Consulting / Platform ($25k-$75k):** sales-assisted, 14-day typical close, annual MSA, Stripe metered billing for overage.
- **Enterprise ($100k-$300k):** founder-led sale, 30-60 day close, annual MSA + DPA, manual invoicing via `greenlang/factors/ga/billing.py`.

Contact: **sales@greenlang.io**. Stripe SKUs provisioned via `greenlang/factors/billing/stripe_provider.py` + `greenlang/factors/ga/sku_catalog.py`.

---

*Pricing principle (CTO non-negotiable): **price is never by factor count.** Price is by API calls, batch volume, Premium pack entitlements, overlay usage, tenant count, OEM rights, SLA. Counting factors would punish the customer for choosing better coverage — the opposite of the intended outcome. See PRD §7.1.*
