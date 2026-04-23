# Licensing and BYO-credentials posture

This page is the authoritative reference for which sources ship in which SKU, what each data class permits, and how the **bring-your-own-credentials (BYO)** posture works for commercial publishers that do not permit GreenLang to redistribute their values at v1 launch.

**Authoritative legal reference:** [`docs/legal/source_rights_matrix.md`](../legal/source_rights_matrix.md).
**Contract outreach posture:** [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md).

---

## 1. The four data classes

Per CTO non-negotiable #4, every factor carries exactly one `licensing.redistribution_class`, and a single API response never mixes classes. See [`concepts/license_class.md`](concepts/license_class.md).

| Class | Storage policy | Export policy |
|---|---|---|
| `open` | Public bucket, no entitlement check | Community-tier AND Certified bulk export |
| `licensed_embedded` | Separate namespace with per-pack entitlement | API only; NEVER in bulk Certified export |
| `customer_private` | Tenant-scoped; zero cross-tenant visibility | Never exported outside tenant |
| `oem_redistributable` | Separate OEM namespace | OEM tenants redistribute to sub-tenants under contract |

---

## 2. What ships in each tier at v1

### Community tier (free, public default pack)

Open-class sources only:

- **EPA GHG Emission Factors Hub** (US, public domain).
- **EPA eGRID** (US, public domain; 27 subregions).
- **UK DESNZ** GHG conversion factors (OGL v3).
- **India CEA** CO2 Baseline Database.
- **India CCTS** baselines.
- **Australia DCCEEW NGA** (CC-BY-4.0).
- **Canada CER** provincial electricity intensity (OGL-Canada).
- **IPCC AR6 GWP tables** (pending Legal opinion on numerical-fact doctrine).
- **IPCC 2006 Guidelines + 2019 Refinement** (pending IPCC copyright confirmation).

Full list: [`source_rights_matrix.md`](../legal/source_rights_matrix.md) §3.1.

### Developer Pro SKU

Community pack + published `/explain` payloads + batch API.

### Electricity Premium SKU

- **AIB European Residual Mixes** (if agreement closes)
- **Green-e Residual Mix** (US + CA): **BYO-credentials only** until CRS contract closes.

### Freight Premium SKU

- **GLEC Framework v3.0**: **BYO-credentials only** until Smart Freight Buyer Group membership closes.
- Open fallbacks: DEFRA freight, EcoTransIT open, IPCC fuel defaults.

### Product Carbon / LCI Premium SKU

- **ecoinvent v3.10**: **BYO-customer-credentials only** — no redistribution at v1.
- **EC3 EPD library**: **BYO-credentials only** until Building Transparency partnership closes.
- **EXIOBASE v3**, **CEDA/PBE**: BYO or post-contract licensed.
- **PACT Pathfinder**: `licensed_embedded` (methodology text; data objects via PACT network).

### Enterprise / CBAM Premium SKU

- **IEA** statistics: **BYO-credentials only** — IEA default terms forbid redistribution.
- **EU CBAM default values** (DG TAXUD): open-class pending Legal review.

---

## 3. What "BYO-credentials" means

A BYO-credentials source is:

- Configured at runtime via **tenant-supplied API key / license code**.
- Factors resolved via **connector at query time**; NOT ingested into the GreenLang catalog.
- Responses **not stored** in any shared factor registry.
- The tenant sees the data; the tenant bears the upstream license; GreenLang is a **transport**, not a redistributor.
- Every API response still carries the publisher's required attribution string.
- Audit bundle records the connector call plus the tenant's credential ID (not the secret).

Set up a BYO connector with:

```bash
gl-factors connector add \
  --source ecoinvent \
  --credential-id "$ECOINVENT_LICENSE_ID"
```

The command displays the publisher's license terms in-line before the credential is registered. See [`sdks/cli.md`](sdks/cli.md).

This is the **safe posture** for any commercial-license source. v1 Certified edition ships on schedule regardless of contract timing. Each closed contract upgrades a BYO posture to `licensed_embedded` without breaking existing flows.

---

## 4. Attribution strings (required in every response)

The server inserts `licensing.attribution_text` on every resolved response. Callers MUST render attribution in any human-readable report. Exact strings are reviewed by Legal every 12 months.

| Source | Required text |
|---|---|
| UK DESNZ | "Contains public sector information licensed under the Open Government Licence v3.0. © Crown copyright." |
| EPA GHG Factor Hub | "Source: US EPA GHG Emission Factors Hub (public domain)." |
| EPA eGRID | "Source: US EPA eGRID (public domain). Year: {year}. Subregion: {subregion}." |
| India CEA | "Source: Central Electricity Authority, Government of India. CO2 Baseline Database v{version}." |
| AIB Residual Mix | "Source: Association of Issuing Bodies (AIB) — European Residual Mixes. Year: {year}." |
| IPCC AR6 GWP | "GWP values from IPCC Sixth Assessment Report (2021), Working Group I, Chapter 7, Table 7.SM.7 (factual constants)." |
| GHG Protocol | "Methodology aligned with GHG Protocol Corporate Standard and Scope 2 Guidance (WRI/WBCSD)." |
| GLEC | "GLEC Framework for Logistics Emissions Accounting and Reporting, Smart Freight Centre, v3.0 (2023)." |
| PCAF | "PCAF, The Global GHG Accounting and Reporting Standard for the Financial Industry, Part A (v2.0) & Part B." |
| EC3 | "Embodied Carbon in Construction Calculator (EC3), Building Transparency." |
| EU CBAM | "European Commission, Carbon Border Adjustment Mechanism default values, DG TAXUD." |

Full table: [`source_rights_matrix.md`](../legal/source_rights_matrix.md) §3.5 and [`source_contracts_outreach.md`](../legal/source_contracts_outreach.md) Part 4.

---

## 5. What we do NOT promise

- We do NOT redistribute ecoinvent, IEA, Electricity Maps, EC3, TCR, Green-e, or GLEC values at v1. Use BYO-credentials.
- We do NOT guarantee IEA / ecoinvent / Electricity Maps response reproducibility across connector refreshes — the upstream subscription controls cadence.
- We do NOT ship `licensed_embedded` factors in a bulk export, ever. Bulk exports are `open`-class only.
- We do NOT include `customer_private` factors in any cross-tenant response, OEM parent view, or audit bundle outside the originating tenant.

---

## 6. Upgrade path as contracts close

When an outreach closes (see [`source_contracts_outreach.md`](../legal/source_contracts_outreach.md) Part 2):

1. The source transitions from BYO to `licensed_embedded` in the catalog.
2. Tenants holding the relevant Premium SKU automatically start receiving embedded values (no client code change).
3. BYO connectors remain active for tenants who prefer their own subscription (e.g., existing enterprise ecoinvent customers).
4. A `source.updated` webhook event fires to notify affected tenants.

---

## Related

- [`concepts/license_class.md`](concepts/license_class.md), [`concepts/source.md`](concepts/source.md).
- [Source rights matrix](../legal/source_rights_matrix.md), [Contract outreach](../legal/source_contracts_outreach.md).
- [Launch legal binder](../launch/legal_source_rights_binder.md).
