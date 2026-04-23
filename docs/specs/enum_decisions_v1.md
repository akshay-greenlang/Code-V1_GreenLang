# GreenLang Factors v1 — Enum Decision Memo

**Prepared for:** CTO sign-off  
**Prepared on:** 2026-04-23  
**Purpose:** Lock three enum choices before the canonical-record migration (Wave 2b, Agent D) begins. A wrong choice costs ~1 week of rework.

These are **proposed defaults** — each has a sign-off checkbox below for CTO to approve, override, or defer.

---

## 1. `status` enum — factor lifecycle state

### Problem
Three parallel sets exist today:
- Code (`emission_factor_record.py`): `certified | preview | connector_only | deprecated`
- Frozen schema draft: `draft | under_review | active | deprecated | retired`
- User/founder brief: adds `superseded | private`

A factor's lifecycle needs to cover authoring (draft), release candidates (preview), the live certified state, and three terminal states that are NOT equivalent (deprecated ≠ superseded ≠ retired).

### Proposal — 7 values

| Value | Semantic | When applied |
|---|---|---|
| `draft` | In methodology authoring, not yet published | Ingestion pipeline default |
| `preview` | Published but NOT Certified. Usable but no SLA. Connector-only sources stay here. | Release candidate / licensed-connector default |
| `certified` | Public Certified-edition eligible. Full SLA. | After methodology + QA + legal signoff |
| `deprecated` | Phase-out announced; replacement_pack_id **required**. Still resolvable until end of notice period. | Methodology refresh / source update |
| `superseded` | Replaced by a newer factor_version of the SAME factor_id. Returned from resolver only when explicitly version-pinned. | On every new factor_version |
| `retired` | Removed from active catalog. Archival only — retrievable by pinned edition for reproducibility. Not listed in search. | End of notice + operational cutover |
| `private` | Customer-private tenant factor. Never in public catalog or Certified edition. | Tenant-override workflow |

### Migration from legacy
- `certified` → `certified` (no change)
- `preview` → `preview` (no change)
- `connector_only` → `preview` WITH a separate `connector_only: bool` flag on the record (orthogonal to status, not a status value)
- `deprecated` → `deprecated`
- New values (`draft`, `superseded`, `retired`, `private`) have no legacy rows to migrate

### CTO decision
- [ ] Approve as proposed (7 values)
- [ ] Reduce to 5 (drop `superseded` and `private` — handle via flags)
- [ ] Reduce to 4 (keep legacy, add `retired` only)
- [ ] Other — please annotate

---

## 2. `redistribution_class` enum — data licensing class

### Problem
CTO spec names 4 classes explicitly. Code today mixes boolean flags + a 5-value soft enum + free-text `license_class`. Frozen schema wants one 4-value enum.

### Proposal — 4 values (CTO spec is explicit; no ambiguity, just enforcement)

| Value | Semantic | Example |
|---|---|---|
| `open` | Public data; redistributable under stated attribution. May appear in Community tier. | UK DESNZ (OGL v3), EPA public domain, India CEA |
| `licensed_embedded` | Licensed commercial data embedded in a GreenLang Premium Pack under contract. Caller must have Premium SKU entitlement. Not redistributable by caller. | ecoinvent, IEA, Green-e, GLEC (post-contract) |
| `customer_private` | Tenant-owned data; visible only to that tenant. Never in any public edition or cross-tenant search. | Customer facility-specific factor, supplier primary data uploaded by tenant |
| `oem_redistributable` | Platform/OEM tenant can redistribute to sub-tenants under an OEM license. | White-label partner redistributing a curated subset |

### Migration from legacy
- boolean `redistribution_allowed=True` + `commercial_use_allowed=True` → `open`
- boolean `redistribution_allowed=False` + licensed commercial source → `licensed_embedded`
- tenant-scoped records → `customer_private`
- existing 5-value soft enum `restricted` → map to `licensed_embedded` for commercial sources; `customer_private` for tenant-owned

### CTO decision
- [ ] Approve as proposed (4 values, matches CTO spec)
- [ ] Add a 5th value — please specify
- [ ] Other — please annotate

**Note:** This choice is the most mature — it matches the CTO spec verbatim. Recommended: approve.

---

## 3. `gwp_set` enum — GWP reference set

### Problem
Code has `gwp_set` as a free-text string. Spec requires enumerated values with stable names so reports filed under pinned editions can be reproduced. IPCC has released multiple assessments with materially different GWP values for CH4/N2O/fluorinated gases.

### Proposal — 6 values

| Value | Source | Horizon | Use case |
|---|---|---|---|
| `IPCC_AR4_100` | IPCC Fourth Assessment (2007) | 100 yr | Legacy filings pre-2024; some regulators still require |
| `IPCC_AR5_100` | IPCC Fifth Assessment (2013) | 100 yr | Common reporting baseline 2015–2024 |
| `IPCC_AR6_100` | IPCC Sixth Assessment (2021) — **DEFAULT** | 100 yr | Current best-available; default for new Certified factors |
| `IPCC_AR5_20` | IPCC Fifth Assessment | 20 yr | Short-horizon alternative (e.g., CH4 sensitivity analysis) |
| `IPCC_AR6_20` | IPCC Sixth Assessment | 20 yr | Same, current science |
| `Kyoto_SAR_100` | IPCC Second Assessment (1995) | 100 yr | Kyoto Protocol legacy; may be required for very old inventory re-runs |

### Default for new factors
`IPCC_AR6_100`. Record-level `gwp_set` is stored but a new resolver parameter `?gwp=IPCC_AR5_100` derives CO2e under the requested set on read. Gas components (CO2, CH4, N2O, fluorinated gases) stored separately, CO2e is DERIVED — matching N1 non-negotiable.

### Migration from legacy
- Records with no `gwp_set` → tag `IPCC_AR5_100` (most likely historical default; methodology lead confirms)
- Records with `gwp_set == "AR5"` → `IPCC_AR5_100`
- Records with `gwp_set == "AR6"` → `IPCC_AR6_100`

### CTO decision
- [ ] Approve as proposed (6 values)
- [ ] Drop `Kyoto_SAR_100` (very rare usage)
- [ ] Drop 20-year horizons (scope to 100yr only for v1)
- [ ] Add WMO/WRF variants for ozone-depleting substances
- [ ] Other — please annotate

---

## Sign-off

| Approver | Role | Decision | Date | Signature |
|---|---|---|---|---|
| CTO | Technical authority | | | |
| Methodology Lead | Scientific authority | | | |
| Product Lead | Scope authority | | | |

---

## What happens when you sign

On your approval, Agent D (gl-calculator-engineer) runs the canonical-record migration per `docs/specs/schema_v1_gap_report.md`:

- ~1,850 LOC across 11 migration tickets
- Adds the 6 missing `parameters` discriminated-union models (transport, materials, refrigerants, land_removals, finance_proxy, waste)
- Consolidates the `lineage` sub-object
- Rebinds `GHGVectors` → spec numerator shape (f_gases dict)
- Applies the status/licensing/gwp mappings chosen above
- Flattens `jurisdiction` to `{country, region, grid_region}`
- Rescales DQS from 1–5 mean to 0–100 weighted composite
- Externalises GWP coefficients to `greenlang/data/gwp_registry.py`
- Extends `MethodProfile` enum to 14 explicit values
- Restructures `activity_schema` into `{category, sub_category, classification_codes[]}`

Estimated: 3 engineer-weeks as a single agent run, or ~5 engineer-days with parallelism. Reversible via branch if any choice needs changing within 72 hours of merge.

**Without these three decisions locked, Agent D is held.** The rest of Wave 2 (A, B, C, E, F, G) runs in parallel and is not blocked by this memo.
