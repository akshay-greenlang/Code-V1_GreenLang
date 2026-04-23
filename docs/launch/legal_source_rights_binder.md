# GreenLang Factors — Legal Source Rights Binder

**Audience:** Legal, third-party auditors, procurement counterparties, enterprise legal review teams.
**Purpose:** Consolidated legal evidence pack. Every source in the v1 Certified edition, with license posture, attribution string, contract status, and `v1_gate_status` (Safe-to-Certify / Needs-Legal-Review / Blocked-Contract-Required).

**Source of truth:** [`docs/legal/source_rights_matrix.md`](../legal/source_rights_matrix.md) — audit-generated document, dated 2026-04-23.
**Contract outreach:** [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md) — ready-to-send drafts + carve-out posture.

> **Disclaimer.** This binder consolidates the programmer-facing rights matrix into a legal-review-ready package. Nothing here is legal advice; any row marked `Needs-Legal-Review` requires Legal to validate against the publisher's current license page before Certified cut.

---

## 1. The four data classes (enforced)

Per CTO non-negotiable #4, every factor record carries exactly one `licensing.redistribution_class`. A single API response NEVER mixes classes.

| Class | Storage | Export | Purpose |
|---|---|---|---|
| `open` | Public bucket, no entitlement check | Community tier + Certified bulk export | Public-domain and open-license publisher data (EPA, DESNZ, CEA, DCCEEW, IPCC). |
| `licensed_embedded` | Separate namespace with per-pack entitlement | API only; NEVER in bulk Certified export | Licensed commercial data embedded in a GreenLang Premium Pack under contract. |
| `customer_private` | Tenant-scoped; zero cross-tenant visibility | Never exported outside tenant | Tenant-uploaded primary data (facility factor, supplier PCF). |
| `oem_redistributable` | Separate OEM namespace | OEM tenants redistribute to sub-tenants under upstream contract | White-label partners. Empty until first OEM deal closes. |

---

## 2. v1 Certified edition cut-list (per source)

### 2.1 Safe-to-Certify — Open class

These ship in Community and Certified bulk exports with attribution.

| `source_id` | Publisher | License | Attribution string |
|---|---|---|---|
| `epa_hub` | US EPA | US-Gov-PD (17 U.S.C. §105) | "U.S. Environmental Protection Agency, GHG Emission Factors Hub." |
| `egrid` | US EPA | US-Gov-PD | "U.S. EPA eGRID subregion emission rates." |
| `desnz_ghg_conversion` | UK DESNZ | OGL-UK-v3 | "Contains public sector information licensed under the Open Government Licence v3.0 — UK DESNZ, GHG conversion factors." |
| `defra_conversion` | UK DEFRA | OGL-UK-v3 | Same as DESNZ (confirm duplicate vs separate dataset). |
| `beis_uk_residual` | UK DESNZ | OGL-UK-v3 | "Contains public sector information licensed under the Open Government Licence v3.0." |
| `india_cea_co2_baseline` | India CEA (Government of India) | Section 52(1)(q) Copyright Act 1957 (government works exception) | "CO2 Baseline Database for the Indian Power Sector, Central Electricity Authority (Government of India), latest edition." |
| `india_ccts_baselines` | India BEE / MoEFCC | Section 52(1)(q) | "Carbon Credit Trading Scheme baseline emission intensities, BEE, MoEFCC, GOI, G.S.R. 443(E) dated 28 June 2023." |
| `australia_nga_factors` | Australia DCCEEW | CC-BY-4.0 | "National Greenhouse Accounts Factors © Commonwealth of Australia (DCCEEW), licensed under CC BY 4.0." |
| `nger_au_state_residual` | Australia DCCEEW / CER | CC-BY-4.0 | "NGER state-level residual emission factors, CER and DCCEEW, CC BY 4.0." |
| `cer_canada_residual` | Canada CER + ECCC | OGL-Canada 2.0 | "Contains information licensed under the Open Government Licence – Canada. Provincial Electricity Intensity Factors, Canada Energy Regulator." |
| `us_epa_suseeio` | US EPA NRMRL | US-Gov-PD | "US EPA, Supply Chain GHG Emission Factors for US Industries and Commodities (SUSEEIO), v2." |
| `greenlang_builtin` | GreenLang curated | GreenLang Terms | "GreenLang Factors curated edition." |

### 2.2 Needs-Legal-Review — Open class (pending confirmation)

Must be validated by Legal before Certified cut.

| `source_id` | Concern |
|---|---|
| `ipcc_2006_nggi`, `ipcc_2006_afolu_v2019` | IPCC copyright posture — factual-constant defence is defensible but requires board-recorded Legal memo. |
| AR4/AR5/AR6 GWP tables | Same IPCC posture. |
| `aib_residual_mix_eu` | Parser tags `RESTRICTED`, registry tags `open` — reconcile by reading AIB Terms directly. |
| `japan_meti_electric_emission_factors` | Japan 政府標準利用規約 applies (CC-BY-4.0-compatible) but per-dataset licence tag needs confirmation. |
| `kemco_korea_residual` | KOGL type (1 vs 2 vs 3) determines commercial redistribution. |
| `ema_singapore_residual` | Confirm Singapore ODL vs EMA-specific licence. |
| `defra_conversion` | Confirm separate dataset or alias of `desnz_ghg_conversion`. |
| `ghgp_method_refs` | Boundary between methodology text (restricted) and factor values derived via Scope 3 tool (ambiguous). |
| `eu_cbam` | EU Commission publications reusable per Decision 2011/833/EU, but CBAM default tables may be governed by Implementing Regulation (EU) 2023/1773 — confirm. |
| `ef_3_1_secondary` | JRC licence varies per dataset within the EF 3.1 bundle. |
| `waste_treatment` | GreenLang-curated; each underlying provenance chain needs Legal-approved mapping. |

### 2.3 Licensed-Embedded — Premium tier

Ship only via API with per-source entitlement. NEVER in bulk Certified export.

| `source_id` | Publisher | License | Status |
|---|---|---|---|
| `ghgp_method_refs` | WRI/WBCSD | WRI-WBCSD-Terms | Needs-Legal-Review (methodology-text vs values boundary). |
| `pact_pathfinder` | WBCSD PACT | WRI-WBCSD-Terms | Needs-Legal-Review (PACT network membership scope). |
| `pcaf_global_std_v2` | PCAF | PCAF Attribution Terms | Needs-Legal-Review (derived-proxy redistribution). |
| `lsr_removals` | WRI/WBCSD | WRI-WBCSD-Terms | Needs-Legal-Review. |

### 2.4 Blocked-Contract-Required — NOT in Certified at v1

These MUST be BYO-credentials until a signed contract or membership is on file. See [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md) for outreach drafts.

| `source_id` | Contract needed | Timeline |
|---|---|---|
| `ecoinvent` | ecoinvent Association membership + redistribution addendum | 60-90 days |
| `iea` | IEA data subscription + redistribution addendum | 60-90 days |
| `green_e_residual`, `green_e_residual_mix` | CRS (Center for Resource Solutions) data licence | 30-45 days |
| `glec_framework`, `freight_lanes` | Smart Freight Buyer Group membership + redistribution rights | 30 days |
| `ec3_buildings_epd` | Building Transparency EC3 data partnership | 45-60 days |
| `tcr_grp_defaults` | TCR data-use agreement | 30-60 days |
| `exiobase_v3` | EXIOBASE Consortium commercial licence | per-partner |
| `ceda_pbe` | Profundo CEDA commercial licence + OEM addendum | per-partner |
| `electricity_maps` | Electricity Maps OEM agreement (otherwise BYO-customer-credentials only) | per-partner |
| EPD International raw | Per-EPD / program-operator agreements | per-EPD |

---

## 3. BYO-credentials carve-out (ships v1)

Per [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md) Part 1, every Blocked-Contract-Required source ships at v1 in the **bring-your-own-credentials** posture:

- Customer registers their own license key in the dashboard.
- GreenLang resolves factors via a connector at query time.
- Factor values are NOT persisted in the shared catalog.
- The customer bears the upstream license; GreenLang is a transport, not a redistributor.
- Every API response still carries the publisher's required attribution string.
- Audit bundle records the connector call plus the tenant's credential ID (never the secret).

This lets the v1 Certified edition cut on schedule with NO contract dependency, while preserving a clean upgrade path: each closed contract flips a source from BYO to `licensed_embedded` and existing customers automatically start receiving embedded values via webhook notification.

---

## 4. CI guardrail (L9 license scanner)

`.github/workflows/factors_ci.yml` runs the license scanner before every edition cut. The scanner fails the build if:

1. Any source row flagged `Blocked-Contract-Required` appears in a Certified-cut candidate without a `legal_signoff_artifact` field populated.
2. Any source row marked `redistribution_class: customer_private` has `source_id` not matching `tenant:<uuid>`.
3. Any bulk export includes a factor with `licensing.redistribution_class != "open"`.
4. Any API response in the gold-set test mixes more than one `redistribution_class` value.

---

## 5. Top-5 residual legal risks (ranked)

Per [`docs/legal/source_rights_matrix.md`](../legal/source_rights_matrix.md) §4:

1. **ecoinvent** — per-seat license explicitly prohibits redistribution. Action: execute commercial addendum or hold BYO indefinitely.
2. **IEA** — most restrictive of all major energy-data publishers. Action: subscribe + redistribution addendum, OR connector-only.
3. **Green-e Residual Mix** — only public US+CA residual-mix source; essential for Scope 2 market-based filings. Action: execute CRS data licence.
4. **GLEC / EC3** — gate Freight Premium and Construction Premium respectively. Action: SFBG membership + Building Transparency partnership.
5. **IPCC copyright posture** — AR6 GWPs, 2006 Guidelines, EFDB — numerical-fact doctrine defence. Action: IPCC permission letter + Legal memo + board resolution.

---

## 6. Open items for Legal (carried forward)

1. Reconcile `desnz_ghg_conversion` vs `defra_conversion` (alias or separate).
2. Reconcile `green_e_residual` vs `green_e_residual_mix` (dedupe registry).
3. Reconcile `aib_residual_mix_eu` parser-vs-registry disagreement.
4. Produce written IPCC permission OR legal-memo covering factual-constants doctrine.
5. Produce the exact attribution rendering spec per source (OGL v3, OGL-Canada 2.0, CC-BY-4.0 specifics).
6. Produce the "Licensed-Embedded redistribution matrix" used by the resolver's L9 entitlement check.
7. Add missing parsers for: IPCC EFDB, `cer_canada_residual`, `kemco_korea_residual`, `ema_singapore_residual`, `us_epa_suseeio`, `exiobase_v3`, `ceda_pbe`, `ef_3_1_secondary`.
8. Populate `legal_signoff_artifact` field in `source_registry.yaml` as contracts close.

---

## 7. Related

- Source matrix (primary audit): [`docs/legal/source_rights_matrix.md`](../legal/source_rights_matrix.md).
- Contract outreach: [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md).
- Public posture: [`docs/developer-portal/licensing.md`](../developer-portal/licensing.md).
- License-class concept: [`docs/developer-portal/concepts/license_class.md`](../developer-portal/concepts/license_class.md).
- Architecture: [`cto_architecture_deck.md`](cto_architecture_deck.md).
