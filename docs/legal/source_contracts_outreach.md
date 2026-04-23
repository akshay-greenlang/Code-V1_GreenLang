# GreenLang Factors v1 — Source Contract Outreach + Carve-Out Posture

**Prepared for:** Legal + Product + Commercial  
**Prepared on:** 2026-04-23  
**Purpose:** Unblock v1 Certified edition cut without waiting for all 5 premium-source contracts to close.

The audit in `docs/legal/source_rights_matrix.md` flagged 5 blockers: **ecoinvent, IEA, Green-e, GLEC, EC3**. Worst-case closure is 90 days. This memo:
1. Provides ready-to-send outreach drafts for each.
2. Documents a **carve-out posture** that lets v1 ship on time even if ALL 5 slip past the deadline — by gating these sources as connector-only / BYO behind Premium SKUs, instead of bundling them into the Public Default Pack.

---

## Part 1 — Carve-out posture (ships today, no contract dependency)

### Rule: tier-to-source mapping for v1 Certified

| Tier | Sources in pack |
|---|---|
| Community (free, public default pack) | EPA GHG Factor Hub, EPA eGRID, UK DESNZ (OGL v3), India CEA (public), GHG Protocol guidance, IPCC AR6 GWP tables (factual-constants basis, legal memo required) |
| Developer Pro | Community pack + published explain logs + batch API |
| Electricity Premium SKU | AIB residual mix (if agreement), Green-e **BYO credentials only** until contract closes |
| Freight Premium SKU | GLEC **BYO credentials only** until SFBG membership closes |
| Product Carbon / LCI Premium SKU | ecoinvent **BYO customer credentials only** (no redistribution), EC3 **BYO credentials only** until Building Transparency partnership closes |
| Enterprise / CBAM Premium SKU | IEA **BYO credentials only** (never embedded), EU CBAM default values (DG TAXUD — verify license) |

### "BYO credentials" defined

A BYO-credentials source is:
- Configured at runtime via tenant-supplied API key / license code
- Factors resolved via connector AT RUNTIME; NOT ingested into the GreenLang catalog
- Responses NOT stored in any shared factor registry
- Tenant sees the data; tenant bears the license; GreenLang is a transport, not a redistributor

This is the safe posture for any commercial-license source. It ships TODAY. Contract upgrades later convert BYO → embedded-redistributable.

### Documentation deliverables (not contract-dependent)

1. **Public license docs**: `docs/legal/LICENSE_POSTURE.md` stating which sources are open/licensed/BYO/customer-private. Links to source license pages.
2. **SDK/CLI prompts**: CLI `factors connector add --source ecoinvent` walks user through BYO setup, surfaces the original source's license terms in-line.
3. **SKU descriptions in pricing page**: for each Premium SKU, call out which sources are embedded vs BYO.
4. **Audit export disclaimer**: exported audit bundle carries per-source license attribution + BYO notice where applicable.

### Outcome
v1 Certified edition cuts on schedule. Premium SKUs launch but their full data embedding unlocks as contracts close. Customers experience a consistent product today with upgrade paths clearly marked.

---

## Part 2 — Contract outreach (parallel, 90-day clock starts when Legal sends)

### A. ecoinvent — Product/LCI data

**Target:** ecoinvent Association (Zurich)  
**Relationship:** commercial data licensee  
**Estimated timeline:** 60–90 days  
**Recommended plan:** Association membership + redistribution addendum

**Outreach email draft**
> Subject: GreenLang Factors — ecoinvent redistribution license inquiry
>
> Dear ecoinvent Association,
>
> GreenLang is launching Factors, a canonical emissions-factor and climate-reference registry for enterprise sustainability reporting (CSRD, CBAM, SEC Climate Disclosure), in FY27. We are seeking a commercial license that permits redistribution of ecoinvent LCI data within our governed registry, accessible to our Enterprise-tier customers under entitlement.
>
> In scope for our license inquiry:
> - the cutoff sectors most relevant to our first launch: steel, aluminium, cement, hydrogen, fertilizers (CBAM-aligned)
> - redistribution through our API with per-activity attribution + version pinning
> - audit bundle export including ecoinvent source citations
>
> We are happy to structure this as Association membership + a redistribution addendum if that fits your licensing model. We can provide our proposed architecture for per-factor attribution, version pinning, and customer-entitlement-gated access under an NDA.
>
> Could we schedule a 45-minute call to discuss terms and pricing?
>
> Best regards,  
> [Founder/Commercial Lead], GreenLang

**If the deal takes longer than 90 days:** ecoinvent stays in the BYO-credentials posture (customers use their own ecoinvent account through our connector). No breach; no redistribution.

### B. IEA — International Energy Agency data

**Target:** IEA Data Licensing (Paris)  
**Relationship:** data licensee  
**Estimated timeline:** 60–90 days  
**Recommended plan:** redistribution license with restricted downstream use

**Outreach email draft**
> Subject: GreenLang Factors — IEA data redistribution license inquiry
>
> Dear IEA Data Licensing Team,
>
> GreenLang is launching Factors, a governed emissions-factor registry for enterprise climate reporting. We would like to discuss a commercial redistribution license for select IEA datasets — specifically energy and emissions statistics relevant to Scope 2 (market-based) and residual-mix calculations for jurisdictions not covered by AIB / eGRID / India CEA.
>
> We understand IEA license terms are strict regarding redistribution and derivative works. We propose:
> - limited, attribution-preserving redistribution under GreenLang's Enterprise tier
> - no derivative publication or bulk export
> - customer-level audit trails for usage attribution back to IEA
>
> If a redistribution license is not available, we would welcome a discussion of BYO-customer-credential access terms — where our customers present their own IEA subscription credentials through our connector.
>
> Could we schedule a call?
>
> Best regards,  
> [Founder/Commercial Lead], GreenLang

**If the deal takes longer than 90 days:** IEA stays in the BYO-credentials posture. Only customers with their own IEA subscription consume IEA data through our connector.

### C. Green-e Residual Mix — US/CA market-based Scope 2

**Target:** Center for Resource Solutions (CRS) — Oakland, CA  
**Relationship:** data licensee  
**Estimated timeline:** 30–45 days  
**Recommended plan:** redistribution license + attribution

**Outreach email draft**
> Subject: GreenLang Factors — Green-e residual mix redistribution license inquiry
>
> Dear Center for Resource Solutions,
>
> GreenLang is launching an enterprise emissions-factor registry in FY27 and we are seeking a commercial redistribution license for Green-e residual mix data. Green-e is the only public residual-mix source for US and Canadian markets, which makes it essential for GHG Protocol Scope 2 market-based accounting.
>
> In scope:
> - redistribution under GreenLang's Electricity Premium SKU with explicit CRS attribution on every response
> - version pinning so customer inventories remain reproducible across Green-e updates
> - per-factor citation in audit bundles and explain payloads
>
> We would like to discuss annual license terms. Could we schedule a 30-minute call?
>
> Best regards,  
> [Founder/Commercial Lead], GreenLang

**If the deal takes longer than 45 days:** Green-e stays in BYO-credentials posture OR GreenLang ships US/CA Scope 2 market-based as "Preview" only (unsupported for audited filings) until the deal closes. Cross-signal to design partners: US/CA market-based delayed if no Green-e by Week 10.

### D. GLEC — Freight

**Target:** Smart Freight Centre — Amsterdam  
**Relationship:** Smart Freight Buyer Group member  
**Estimated timeline:** 30 days  
**Recommended plan:** join Smart Freight Buyer Group + redistribution rights

**Outreach email draft**
> Subject: GreenLang — Smart Freight Buyer Group membership inquiry
>
> Dear Smart Freight Centre,
>
> GreenLang is launching Factors, a climate-reference registry for enterprise reporting, in FY27. Our Freight Premium pack aligns with GLEC Framework and ISO 14083 by design. We would like to discuss joining the Smart Freight Buyer Group and obtaining redistribution rights for GLEC lane, mode, and utilization factors under our Enterprise entitlement model.
>
> We can share our Freight Pack architecture (WTW/TTW labeling, consignment modes, ISO 14083 chain calculations) under NDA if helpful.
>
> Could we schedule a call?
>
> Best regards,  
> [Founder/Commercial Lead], GreenLang

**If the deal takes longer than 30 days:** GLEC data stays in BYO posture; GreenLang's internal Freight Pack resolution logic (ISO 14083 selection rules) works against open sources (DEFRA freight, EcoTransIT open) + BYO-GLEC for customers who have their own subscription.

### E. EC3 — Embodied Carbon / EPD

**Target:** Building Transparency (EC3)  
**Relationship:** API partner  
**Estimated timeline:** 45–60 days  
**Recommended plan:** partnership agreement + API access

**Outreach email draft**
> Subject: GreenLang Factors — EC3 API partnership inquiry
>
> Dear Building Transparency Team,
>
> GreenLang is launching Factors, an enterprise climate reference registry. Our Construction/EPD Premium pack will align with EC3's embodied-carbon methodology and PCR structure. We would like to discuss an API partnership that lets GreenLang resolve EPDs through EC3 with appropriate attribution and per-customer entitlement controls.
>
> Key questions:
> - permissioned API access scope and pricing tiers
> - whether we can cache EPD records at customer-level (for reproducibility of audited filings)
> - attribution and citation requirements
>
> Could we schedule a call?
>
> Best regards,  
> [Founder/Commercial Lead], GreenLang

**If the deal takes longer than 60 days:** EC3 stays in BYO posture; customers with their own EC3 API keys consume EPDs through our connector. Construction Premium pack launches at "Preview" status until embedded redistribution closes.

---

## Part 3 — IPCC legal opinion (separate track, 15 business days)

IPCC copyright posture requires a Legal opinion memo invoking the numerical-fact doctrine, board-recorded, before IPCC data (AR4/AR5/AR6 GWP tables, 2006 + 2019 Inventory Guidelines, EFDB) ships in the Public Default Pack.

### Recommended Legal opinion structure
1. **Jurisdiction:** Swiss (IPCC TFI is Japanese but the Panel is UN-adjacent; factual constants defence is strongest under WIPO Article 2.8)
2. **Claim:** numerical emission factors published for inventorying purposes are factual constants, not copyrighted expression
3. **Citation precedent:** *Feist Publications v. Rural Telephone Service* (US 1991), WIPO Copyright Treaty Article 2(8), EU Database Directive Article 7(5)
4. **Board resolution:** record the opinion in meeting minutes; attach to audit-evidence folder

### Parallel track
Send a courtesy inquiry letter to IPCC TFI Technical Support Unit (Tokyo) notifying them of our intended use. Not a license request — a notification + request for comment. Silence is tacit acceptance under the Swiss framework.

---

## Part 4 — Attribution strings for open sources (not contract-dependent)

Use these verbatim in explain payloads and audit exports. Reviewed by Legal every 12 months or on source version bump.

| Source | Attribution string | Where applied |
|---|---|---|
| UK DESNZ conversion factors | "Contains public sector information licensed under the Open Government Licence v3.0. © Crown copyright." | Every factor from `desnz_uk` parser |
| EPA GHG Factor Hub | "Source: US EPA GHG Emission Factors Hub (public domain)." | Every factor from `epa_ghg_hub` parser |
| EPA eGRID | "Source: US EPA eGRID (public domain). Year: {year}. Subregion: {subregion}." | Every factor from `egrid` parser |
| India CEA | "Source: Central Electricity Authority, Government of India. CO2 Baseline Database version {version}." | Every factor from `india_cea` parser |
| AIB residual mix | "Source: Association of Issuing Bodies (AIB) — European Residual Mixes. Year: {year}. Used with permission per AIB standard terms." | Every factor from `aib_residual_mix` parser |
| IPCC AR6 GWP | "GWP values from IPCC Sixth Assessment Report (2021), Working Group I, Chapter 7, Table 7.SM.7 (factual constants)." | Every record with `gwp_set = IPCC_AR6_100` |
| GHG Protocol | "Methodology aligned with GHG Protocol Corporate Standard and Scope 2 Guidance (WRI/WBCSD)." | Every factor resolved under Corporate Inventory method profile |

---

## Part 5 — Sign-off

| Role | Action | Owner | Date |
|---|---|---|---|
| Legal kickoff on 5 contracts | Send outreach emails in Part 2 | Legal | Week 1 |
| IPCC opinion memo | Draft + board-record | Legal | Week 1–3 |
| Carve-out posture published | Update `docs/legal/LICENSE_POSTURE.md` + pricing page | Product + DevRel | Week 1 |
| SDK/CLI BYO flows | Implement `factors connector add --source` prompts | DevRel + Backend | Week 2–3 |
| Attribution injection | Wire attribution strings into explain payloads | Backend | Week 2 |
| Track progress | Weekly update to CTO with per-contract status | Legal | Ongoing |

**Outcome:** v1 Certified edition cuts on schedule regardless of contract timing. Each closed contract upgrades a BYO posture to embedded-redistribution without breaking any existing customer flow.
