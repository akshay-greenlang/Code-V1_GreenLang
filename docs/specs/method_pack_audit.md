# Method Pack Audit — v0.1

**Status:** DRAFT v0.1 — MP2 deliverable for GreenLang Factors FY27.
**Template reference:** `docs/specs/method_pack_template.md` (MP1).
**Date:** 2026-04-23.

Legend for cell status:

- `DONE` — section is fully populated, citation-backed, consistent with template.
- `PARTIAL` — section is populated but missing one or more required
  sub-fields (exclusion lists, label_logic, audit_texts placeholders,
  deprecation replacement_pointer, etc.).
- `MISSING` — section is absent or so under-specified that it cannot
  pass the YAML schema without scaffolding.

File:line references point to the construction site of the pack (i.e.
the `MethodPack(...)` call) or to the surrounding rule objects in
`greenlang/factors/method_packs/*.py`.

---

## 1. corporate.py — Corporate Inventory (GHG Protocol + Scope 2 + Scope 3)

Covers four packs: `CORPORATE_SCOPE1`, `CORPORATE_SCOPE2_LOCATION`,
`CORPORATE_SCOPE2_MARKET`, `CORPORATE_SCOPE3`.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `corporate.py:27-74` | `pack_name`, `version`, `tags` present; **MISSING** `status`, `owner_methodology_lead`, `last_reviewed_iso`, `standards_alignment` as structured key. |
| 2 | Selection rules | DONE | `corporate.py:35-45, 84-88, 128-136, 180-198` | Families + formula types + statuses all set per pack. |
| 3 | Boundary rules | PARTIAL | `corporate.py:46-51, 89-95, 137-142, 199-204` | `allowed_scopes` + `allowed_boundaries` set; **MISSING** `system_boundary` explicit tag (gate_to_gate vs cradle_to_gate), `lca_mode`, `scope3_categories` explicit list, `functional_unit`. |
| 4 | Inclusion / exclusion | MISSING | n/a | No `inclusion.activity_categories`, no `exclusion.*` fields anywhere. |
| 5 | Gas-to-CO2e basis | PARTIAL | `corporate.py:52, 96, 143, 205` | `gwp_basis="IPCC_AR6_100"` present; **MISSING** `allowed_override_sets`, `horizon_years` explicit, `metric`, `gwp_star_allowed`. |
| 6 | Biogenic treatment | DONE | `corporate.py:49, 92, 140, 202` | `REPORTED_SEPARATELY` on all four. |
| 7 | Market instruments | DONE | `corporate.py:50, 93, 141, 203` | NOT_APPLICABLE / PROHIBITED / ALLOWED / NOT_APPLICABLE — correct per GHG Protocol. **PARTIAL** on `quality_criteria` list (not expressed). |
| 8 | Region hierarchy | DONE | `corporate.py:53, 97, 144, 206` | Uses `DEFAULT_FALLBACK`. |
| 9 | Fallback logic | PARTIAL | base.py DEFAULT_FALLBACK | `DEFAULT_FALLBACK` exists; **MISSING** `cannot_resolve_action`, `global_default_tier_allowed`, `stale_factor_cutoff_days` as structured config. |
| 10 | Reporting labels | DONE | `corporate.py:55-66, 99-109, 146-157, 208-217` | Rich tag sets per pack. **PARTIAL** on `label_logic` — no conditional add-on logic encoded. |
| 11 | Audit-text template | PARTIAL | `corporate.py:67-71, 110-114, 158-162, 222-228` | Inline strings, not externalised `.j2` files; Scope 3 pack has `cat11_use_phase_block` render helper (`corporate.py:244`). Needs migration to `audit_texts/corporate.j2`. |
| 12 | Deprecation policy | PARTIAL | `corporate.py:24, 54, 98, 145, 207` | `_DEPRECATION` set (max_age 4y, grace 1y); **MISSING** `replacement_pointer`, `migration_notes`, `webhook_fan_out`, `advance_notice_days` (grace != notice). |

**Top 3 P0 gaps for GA (MP3):**

1. **Inclusion / exclusion lists** — GHG Protocol Corporate requires an
   explicit negative list (e.g. employee commuting under `3.7`, biogenic
   CO2 below the de-minimis threshold). Without documented allow/deny
   sets, CBAM and ESRS audits will flag unsubstantiated inclusions.
2. **Structured deprecation policy** — `replacement_pointer` MUST exist
   on every certified pack; regulators will not accept a sunset without
   a named successor. Add default 180-day advance notice + webhook
   fan-out config.
3. **Audit-text externalisation** — inline strings block i18n, prevent
   CI linting of required placeholders, and cannot be version-controlled
   independently of the pack code. Move to `audit_texts/corporate.j2`
   with `render_cat11_use_phase_block` becoming a Jinja filter.

---

## 2. electricity.py — Electricity (Location / Market / Supplier / Residual Mix)

Covers 11 packs: `ELECTRICITY_LOCATION`, `ELECTRICITY_MARKET`,
`ELECTRICITY_RESIDUAL_MIX_EU/US/AU/JP/CA/UK_NATIONAL/AU_STATE/KR/SG`.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `electricity.py:37-68, 71-101, 138-171, ...` | `pack_version` + `tags` present; **MISSING** structured `status`, `standards_alignment`, `owner_methodology_lead`. |
| 2 | Selection rules | DONE | `electricity.py:45-49, 79-82, 146-153, 181-189, 223-230, 266-272, 340-347, 385-392, 432-440, 477-485, 521-529` | Jurisdiction filters via `_jurisdiction_filter` helper (`electricity.py:310-327`). Excellent jurisdictional discipline. |
| 3 | Boundary rules | PARTIAL | `electricity.py:50-56, 84-89, 153-159, ...` | Scope 2 + combustion + transmission_losses=false consistent; **MISSING** `system_boundary` tag, `functional_unit` (`"1 kWh delivered"`). |
| 4 | Inclusion / exclusion | MISSING | n/a | No explicit allow/deny of activity categories or gases. |
| 5 | Gas-to-CO2e basis | PARTIAL | all MethodPack `gwp_basis` fields | Same partial state as corporate.py — override set + metric + horizon not encoded. |
| 6 | Biogenic treatment | DONE | all BoundaryRule blocks | `REPORTED_SEPARATELY` consistently. |
| 7 | Market instruments | PARTIAL | all BoundaryRule blocks | Enum values correct (PROHIBITED for location, ALLOWED for market/residual). **MISSING** `quality_criteria` list (GHG Protocol Scope 2 QC 1-7) and `allowed_instruments` explicit enumeration. |
| 8 | Region hierarchy | DONE | `electricity.py:120-135` + routing map `556-587` | `RESIDUAL_MIX_FALLBACK` + country routing table are strong. |
| 9 | Fallback logic | PARTIAL | `get_residual_mix_pack` fn `593-615` | Falls back to `ELECTRICITY_RESIDUAL_MIX_EU` when unknown country — this violates "cannot resolve safely" for non-EU unknown jurisdictions. **MISSING** structured `cannot_resolve_action`. |
| 10 | Reporting labels | PARTIAL | all MethodPack `reporting_labels` | Labels strong per pack; **MISSING** `label_logic` for WTT/WTW-not-applicable-to-Scope-2 tagging. |
| 11 | Audit-text template | PARTIAL | `electricity.py:61-64, 94-97, 164-167, ...` | Inline string templates; no externalised `.j2`. |
| 12 | Deprecation policy | PARTIAL | `electricity.py:34` (_DEPRECATION max_age 3y, grace 180d) | Shared instance used 11 times; **MISSING** per-pack `replacement_pointer` (critical for residual-mix packs where vintage rolls annually). |

**Top 3 P0 gaps for GA (MP4):**

1. **Market-instrument quality criteria encoded** — currently the pack
   says `MarketInstrumentTreatment.ALLOWED` but nowhere encodes the
   GHG Protocol Scope 2 QC 1-7 that a REC/GO/PPA must satisfy. Auditors
   will reject market-based claims without this.
2. **Residual-mix fallback to EU for unknown countries is unsafe** —
   `get_residual_mix_pack("BR")` today returns the AIB EU pack, which
   is methodologically invalid for Brazil. Must raise `NoSafeMatch`.
3. **Annual vintage roll + replacement_pointer** — residual mix is
   published annually (AIB, Green-e, NGA, METI, DESNZ, CER, KEMCO,
   EMA). Each pack needs `replacement_pointer` AND
   `max_factor_age_days = 730` so stale factors downgrade.

---

## 3. freight.py — Freight (ISO 14083 + GLEC)

Single pack: `FREIGHT_ISO_14083`.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `freight.py:22-59` | `pack_version`, `tags=("freight","licensed")` present; **MISSING** `status`, `owner_methodology_lead`. |
| 2 | Selection rules | DONE | `freight.py:31-42` | TRANSPORT_LANE + EMISSIONS + ENERGY_CONVERSION; certified + preview. |
| 3 | Boundary rules | PARTIAL | `freight.py:43-48` | Scope 3, `WTW`+`WTT` — good; **MISSING** `system_boundary`, `functional_unit` (`"1 t.km"` vs `"1 TEU.km"`). |
| 4 | Inclusion / exclusion | MISSING | n/a | No transport-mode allow/deny (ISO 14083 excludes pipeline + cableway unless opted-in). |
| 5 | Gas-to-CO2e basis | PARTIAL | `freight.py:49` | `IPCC_AR6_100` only. |
| 6 | Biogenic treatment | DONE | `freight.py:46` | `REPORTED_SEPARATELY` — consistent with GLEC v3.0. |
| 7 | Market instruments | DONE | `freight.py:47` | `NOT_APPLICABLE` — correct for freight. |
| 8 | Region hierarchy | DONE | `freight.py:50` | `DEFAULT_FALLBACK`. |
| 9 | Fallback logic | PARTIAL | n/a | No structured `cannot_resolve_action`. |
| 10 | Reporting labels | PARTIAL | `freight.py:52` | `("ISO_14083", "GLEC")` only; **MISSING** Scope 3 + CSRD_E1 + supply-chain labels, and `label_logic` for WTW vs TTW tagging per leg. |
| 11 | Audit-text template | PARTIAL | `freight.py:53-56` | Inline; **MISSING** `leg_id` loop template for multi-leg consignments; no externalised `.j2`. |
| 12 | Deprecation policy | PARTIAL | `freight.py:51` | Inline `DeprecationRule`; **MISSING** `replacement_pointer`, `webhook_fan_out`. |

**Top 3 P0 gaps for GA (MP6 — Preview for FY27 H1):**

1. **Multi-leg audit template** — ISO 14083 requires per-leg disclosure
   (mode, distance, payload, fuel, WTT + TTW + WTW). Current template
   only renders one leg.
2. **Mode allow/deny list** — Pipeline, cableway, and pedestrian
   delivery are excluded unless explicitly opted-in per GLEC.
3. **Label logic for WTW/TTW** — labels must be conditional on the leg
   fuel + reporting scope, not a static tuple.

---

## 4. eu_policy.py — EU Policy (CBAM + DPP + Battery)

Covers three packs: `EU_CBAM`, `EU_DPP`, `EU_BATTERY` + helpers.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `eu_policy.py:26-73, 76-109, 302-352` | `pack_version` + `tags=("eu_policy","licensed",...)`; **MISSING** `status`, `owner_methodology_lead`, `last_reviewed_iso`. |
| 2 | Selection rules | DONE | `eu_policy.py:36-52, 85-92, 315-327` | `require_verification=True` on CBAM + Battery (correct for regulated filings). |
| 3 | Boundary rules | PARTIAL | `eu_policy.py:53-60, 93-98, 328-336` | Scopes + biogenic enforced; **MISSING** explicit `system_boundary` tag, `functional_unit` (CBAM `"1 t of CN-coded good"`, Battery `"1 kWh delivered over service life"` — latter in audit text but not structured field). |
| 4 | Inclusion / exclusion | MISSING | n/a | CBAM has a regulation-defined allow-list (cement, iron & steel, aluminium, fertilisers, electricity, hydrogen) but it's only in prose in `description`. Must become a structured `inclusion.activity_categories` list. |
| 5 | Gas-to-CO2e basis | PARTIAL | `eu_policy.py:61, 99, 337` | AR6 100; **MISSING** — CBAM implementing regulation references AR6 specifically, so `allowed_override_sets` should be empty (locked). |
| 6 | Biogenic treatment | DONE | `eu_policy.py:58, 96, 334` | CBAM `EXCLUDED`; DPP + Battery `REPORTED_SEPARATELY` — correct. |
| 7 | Market instruments | PARTIAL | `eu_policy.py:59, 97, 335` | CBAM `REQUIRE_CERTIFICATE` — right enum; **MISSING** `allowed_instruments` list (CBAM accepts only primary operator data + limited proxies). |
| 8 | Region hierarchy | PARTIAL | `eu_policy.py:62, 100, 338` | `DEFAULT_FALLBACK` — **wrong** for CBAM. CBAM has its own operator-first hierarchy that disallows global default; using 7-step default silently lets a global proxy through. |
| 9 | Fallback logic | MISSING | n/a | CBAM regulation Art. 4(2) REQUIRES documented reason when falling back to EU default values. Currently no `cannot_resolve_action: raise_no_safe_match`. |
| 10 | Reporting labels | PARTIAL | `eu_policy.py:64, 102, 340` | `("EU_CBAM",)`, `("EU_DPP", "ESPR")`, `("EU_Battery_Regulation", "EU_DPP", "Article_7_CFP")` — **MISSING** `label_logic` (e.g. surcharge-tier label when fallback tier > 1). |
| 11 | Audit-text template | PARTIAL | `eu_policy.py:65-70, 103-106, 341-348` | Good structure (`cn_code`, `verification_status`, `battery_class`); **MISSING** externalised `.j2`, fallback trace disclosure, Article 4(2) justification block. |
| 12 | Deprecation policy | PARTIAL | `eu_policy.py:63, 101, 339` | 2y CBAM / 3y DPP+Battery — tight because regulation updates quickly. **MISSING** `replacement_pointer`, `migration_notes`. |

**Top 3 P0 gaps for GA (MP5):**

1. **CBAM-specific fallback hierarchy** — DEFAULT_FALLBACK allows a
   tier-7 global default, which CBAM regulation forbids. Must replace
   with a 5-tier CBAM-specific chain ending in "EU default value with
   surcharge" and `cannot_resolve_action = raise_no_safe_match`.
2. **CBAM CN-code inclusion list** — must be a structured
   `inclusion.activity_categories` matching Annex I of EU 2023/956
   (not just prose).
3. **Article 4(2) justification block in audit template** — every CBAM
   record that used an EU default (instead of primary operator data)
   MUST render a legally-required justification paragraph.

---

## 5. land_removals.py — Land Sector & Removals (GHG Protocol LSR)

Covers five packs: `GHG_LSR_LAND_USE_EMISSIONS`, `GHG_LSR_LAND_MANAGEMENT`,
`GHG_LSR_REMOVALS`, `GHG_LSR_STORAGE`, `LAND_REMOVALS` umbrella.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `land_removals.py:342-364, 390-409, 439-462, 506-527, 558-574` | `pack_version`, `tags` present; LSR-specific `LSRPackMetadata` sidecar is excellent. **MISSING** structured `status`, `owner_methodology_lead`. |
| 2 | Selection rules | DONE | `land_removals.py:282-297` | `_lsr_selection_rule` shared helper. |
| 3 | Boundary rules | DONE | `land_removals.py:318-323` | `cradle_to_grave` for all LSR. |
| 4 | Inclusion / exclusion | PARTIAL | `LSRPackMetadata.allowed_removal_types` `land_removals.py:168` | Positive list exists via `allowed_removal_types` + `direct_land_use_included` / `iluc_included` / `soc_tracked` flags — good; **MISSING** explicit `exclusion.source_ids` (e.g. exclude non-ICVCM-CCP approved standards). |
| 5 | Gas-to-CO2e basis | PARTIAL | `land_removals.py:324` | AR6 100; **MISSING** `gwp_star_allowed=true` for methane-heavy peatland / livestock contexts. |
| 6 | Biogenic treatment | DONE | `land_removals.py:361, 406, 459, 524, 572` | `BiogenicTreatment.INCLUDED` correctly; extended `BiogenicAccountingTreatment` (CARBON_NEUTRAL/SEQUESTRATION_TRACKED/STORAGE_TRACKED) gives extra granularity. |
| 7 | Market instruments | DONE | `land_removals.py:362, 407, 460, 525, 573` | Correct per variant (NOT_APPLICABLE for emissions + management; REQUIRE_CERTIFICATE for removals + storage). |
| 8 | Region hierarchy | PARTIAL | LSR_FALLBACK_HIERARCHY `land_removals.py:198-207` | **LSR-specific chain exists but is NOT wired** — `_build_pack` uses `DEFAULT_FALLBACK` (`line 325`). Needs swap. |
| 9 | Fallback logic | MISSING | n/a | No `cannot_resolve_action`; LSR should raise when both project data and IPCC Tier 1 default fail. |
| 10 | Reporting labels | PARTIAL | `land_removals.py:326-331` | `("GHG_Protocol_LSR", "IPCC_2006_GL", "IPCC_2019_Refinement")`; **MISSING** ICVCM-CCP label, per-variant label_logic. |
| 11 | Audit-text template | PARTIAL | `land_removals.py:353-360, 400-405, 451-458, 516-523, 567-571` | Strong per-variant context (buffer_pct, reversal_risk, permanence_class); **MISSING** externalisation and cross-variant shared header. |
| 12 | Deprecation policy | PARTIAL | `land_removals.py:279` | 5y / 2y grace; **MISSING** `replacement_pointer`, `migration_notes`. |

**Top 3 P0 gaps for GA (MP7 — Preview for FY27 H1):**

1. **Wire `LSR_FALLBACK_HIERARCHY` into `_build_pack`** — currently a
   dead symbol. Without it, project-specific data does not outrank
   Tier 1 defaults at resolution time.
2. **Buffer-pool + reversal-risk contract** — the sidecar metadata is
   rich but not exposed to the resolver for use in the receipt.
3. **ICVCM-CCP label + verification_standard enforcement** in the
   `require_verification` path of `SelectionRule`.

---

## 6. product_carbon.py + product_lca_variants.py — Product Carbon (ISO 14067 / GHG PS / PACT / PAS 2050 / PEF / OEF)

Covers four packs: `PRODUCT_CARBON`, `PAS_2050`, `PEF`, `OEF`.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `product_carbon.py:22-58`, `product_lca_variants.py:43-81, 88-129, 136-176` | PEF has `require_verification=True` (correct); **MISSING** structured `status`, `owner_methodology_lead`. |
| 2 | Selection rules | DONE | inline per pack | Families + formula types consistent. |
| 3 | Boundary rules | PARTIAL | inline per pack | Scope 3 + cradle_to_gate/cradle_to_grave; **MISSING** `functional_unit` as structured field (product-level LCA REQUIRES it). |
| 4 | Inclusion / exclusion | PARTIAL | `product_lca_variants.py:67` | PAS 2050 has `BiogenicTreatment.EXCLUDED` (per §7.4) — good; **MISSING** explicit gas-level exclusion list for PAS 2050. |
| 5 | Gas-to-CO2e basis | PARTIAL | `product_lca_variants.py:70` (PAS_2050) uses `IPCC_AR5_100` | PAS 2050 historical alignment is AR5 — correct. Others AR6 100. **MISSING** `allowed_override_sets`, `gwp_star_allowed`. |
| 6 | Biogenic treatment | DONE | inline | PAS 2050 EXCLUDED, others REPORTED_SEPARATELY. |
| 7 | Market instruments | DONE | inline | PRODUCT_CARBON + PAS_2050 + PEF: NOT_APPLICABLE; OEF: ALLOWED (matches entity-level scope 2). |
| 8 | Region hierarchy | DONE | all `DEFAULT_FALLBACK` | Acceptable for product LCA. |
| 9 | Fallback logic | MISSING | n/a | No `cannot_resolve_action`. Product LCA with missing material factors should raise. |
| 10 | Reporting labels | DONE | inline | `ISO_14067`, `GHG_Protocol_Product`, `PACT`, `PAS_2050`, `EU_PEF`, `EF_3_1`, `PEFCR`, `EU_OEF`, `OEFSR`, `ESRS_E1`. |
| 11 | Audit-text template | PARTIAL | inline per pack | Includes `allocation_method`, `recycled_content_pct`, `functional_unit`, `pefcr_id`. **MISSING** externalised `.j2`, PEFCR-per-category routing. |
| 12 | Deprecation policy | PARTIAL | 5y / 1y | **MISSING** `replacement_pointer`. |

**Top 3 P0 gaps for GA (MP8 — Preview for FY27 H1):**

1. **Functional unit structured** — product LCA without a machine-
   readable functional unit cannot be integrated with downstream
   CBAM / DPP / Battery packs.
2. **PEFCR routing table** — PEF packs must have a category-rule
   routing table (each product category has its own PEFCR).
3. **Attributional vs consequential tag** — default attributional
   today but not explicit; consequential ecoinvent datasets should be
   blocked via `exclusion.source_ids`.

---

## 7. finance_proxy.py — PCAF Financed Emissions

Covers eight packs: `FINANCE_PROXY` umbrella + 7 asset-class variants.

| # | Section | Status | Reference | Notes |
|---|---------|--------|-----------|-------|
| 1 | Metadata | PARTIAL | `finance_proxy.py:333, 369, 405, 445, 483, 526, 571, 614` | `PCAFPackMetadata` sidecar is excellent (`attribution_method`, `dqs_scale`, `attribution_formula`). **MISSING** structured `status`, `owner_methodology_lead`. |
| 2 | Selection rules | DONE | `finance_proxy.py:262-280` | `_pcaf_selection_rule` shared helper. |
| 3 | Boundary rules | DONE | `finance_proxy.py:283-289` | Scope 1+2+3 with high-emitting-sector override. |
| 4 | Inclusion / exclusion | PARTIAL | `_PCAF_SCOPE3_SECTORS` `finance_proxy.py:239-256` | Positive inclusion (sectors requiring Scope 3) exists; **MISSING** exclusion list (PCAF v2 explicitly excludes sovereign debt + central-bank reserves from the asset-class set). |
| 5 | Gas-to-CO2e basis | PARTIAL | `finance_proxy.py:314` | AR6 100. |
| 6 | Biogenic treatment | DONE | `finance_proxy.py:287` | REPORTED_SEPARATELY. |
| 7 | Market instruments | DONE | `finance_proxy.py:288` | NOT_APPLICABLE — correct (RECs applied at counterparty, not financier). |
| 8 | Region hierarchy | PARTIAL | attribution hierarchy `finance_proxy.py:125-136` | **PCAF_ATTRIBUTION_HIERARCHY** exists as a 5-step chain but `region_hierarchy=DEFAULT_FALLBACK` (line 315). Similar bug to LSR: the custom chain is a dead field on the metadata side. |
| 9 | Fallback logic | PARTIAL | n/a | `uncertainty_band_required_dqs=4` ensures DQS 4-5 triggers uncertainty disclosure (good); **MISSING** `cannot_resolve_action`. |
| 10 | Reporting labels | DONE | `finance_proxy.py:317-322` | `("PCAF","PCAF_Part_A_v2.0","GHG_Protocol_Scope3_Cat15","IFRS_S2")`. |
| 11 | Audit-text template | DONE | inline per variant | Attribution factor + DQS + emissions per class. **PARTIAL** on externalisation. |
| 12 | Deprecation policy | PARTIAL | `finance_proxy.py:259` | 4y / 1y; **MISSING** `replacement_pointer`. |

**Top 3 P0 gaps for GA (MP9 — Preview for FY27 H1):**

1. **Wire `PCAF_ATTRIBUTION_HIERARCHY` into every variant pack** —
   currently the 5-step attribution hierarchy is only in sidecar metadata,
   not in the pack's own `region_hierarchy`. Resolver uses
   `DEFAULT_FALLBACK` which is not PCAF-shaped.
2. **PCAF sovereign exclusion list** — explicitly deny sovereign debt,
   central-bank reserves, and gold from the eligible scope.
3. **DQS-based selection** — PCAF requires portfolio weighted DQS
   reporting. Today `SelectionRule` doesn't emit DQS on the receipt;
   add DQS as a required placeholder in the audit template (not just
   a sidecar field).

---

## Summary table — audit score at a glance

| Pack | Template sections DONE | PARTIAL | MISSING | Overall |
|---|---|---|---|---|
| corporate.py | 4 | 7 | 1 | ~55% |
| electricity.py | 3 | 8 | 1 | ~50% |
| freight.py | 3 | 8 | 1 | ~45% |
| eu_policy.py | 3 | 7 | 2 | ~45% |
| land_removals.py | 5 | 6 | 1 | ~60% |
| product_carbon.py (+variants) | 3 | 8 | 1 | ~50% |
| finance_proxy.py | 5 | 7 | 0 | ~70% |

(*"Overall" is a rough qualitative score: 100% = DONE, 50% = PARTIAL, 0% = MISSING, averaged over 12 template sections.*)
