# Method Pack Specification Template

**Status:** DRAFT v0.1 — MP1/MP2 deliverable for GreenLang Factors FY27.
**Owner:** GL-FormulaLibraryCurator (Methodology WG).
**Scope:** Normative template that every method pack under
`greenlang/factors/method_packs/*.py` MUST satisfy before it can be
promoted to `status = certified` inside a v1.0 Certified edition cut.

This document has two parts:

1. A **markdown description** of each required section — what the section
   means, what it controls at runtime, how the resolution engine reads
   it, and what is expected in each field.
2. A **YAML schema block** that can be used to validate a pack definition
   in CI. The YAML schema is the authoritative contract; the markdown
   sections exist only to explain it.

A pack author satisfies the template when every required key in the
YAML schema block below is populated with a defensible, citation-backed
value AND every "TODO(MP3|MP4|MP5)" marker has been cleared.

---

## 1. Metadata

Required identity + governance fields for every pack.

| Field | Type | Example | Notes |
|---|---|---|---|
| `pack_id` | `str` | `"corporate_scope1"` | Snake-case, globally unique across the registry. Matches `MethodProfile` value or variant name. |
| `pack_name` | `str` | `"Corporate Inventory - Scope 1 (GHG Protocol)"` | Human-readable, appears in `/explain` + UI. |
| `version` | `semver` | `"1.0.0"` | **Semver MAJOR.MINOR.PATCH.** MAJOR bumps on breaking selection/boundary changes; MINOR on new capabilities (e.g. added labels, new formula type); PATCH on audit-text / doc-only fixes. |
| `status` | `enum` | `certified` | One of `draft \| preview \| certified \| deprecated`. Only `certified` packs are eligible for regulated disclosures (CBAM, SECR, CSRD). |
| `standards_alignment` | `list[str]` | `["GHG Protocol Corporate", "IFRS S2", "CSRD_E1"]` | Normative standards the pack claims conformance with; each entry must resolve to an entry in `reporting_labels` table. |
| `deprecation_window_days` | `int` | `180` | Advance notice (in days) before the pack moves from `certified` -> `deprecated`. Min 90, default 180 for product packs, 365 for Corporate/Scope 2. |
| `owner_methodology_lead` | `str` | `"methodology-wg@greenlang.io"` | Accountable human owner (email or slack handle) for methodology questions + deprecation decisions. |
| `last_reviewed_iso` | `str` | `"2026-04-22"` | ISO-8601 date of last methodology-review-board sign-off. |
| `pack_source_file` | `str` | `"greenlang/factors/method_packs/corporate.py"` | Path to the Python module that constructs this pack. |

---

## 2. Factor Selection Rules

Controls which records from the canonical factor catalogue this pack will
accept when the resolver returns candidates. Evaluated **after** the
candidate list is built (see `SelectionRule.accepts` in
`greenlang/factors/method_packs/base.py`).

| Field | Type | Notes |
|---|---|---|
| `selection.allowed_families` | `list[FactorFamily]` | Required. Pack returns NO_MATCH if all candidates are in families not on this list. |
| `selection.allowed_formula_types` | `list[FormulaType]` | Required. Filters by canonical v2 `formula_type`. |
| `selection.allowed_statuses` | `list[str]` | Default `["certified"]`. Regulated packs (CBAM, Battery) MUST NOT include `preview`. |
| `selection.require_verification` | `bool` | When true, only records with `verification.status in {external_verified, regulator_approved}` pass. |
| `selection.require_primary_data` | `bool` | When true, only records with `primary_data_flag in {primary, primary_modeled}` pass. |
| `selection.jurisdiction_hierarchy` | `list[str]` | Ordered list, most-specific first, e.g. `["country", "region", "subregion"]`. Resolver iterates this list to narrow the candidate pool. |
| `selection.priority_tiers` | `list[str]` | Ordered priority e.g. `["supplier", "facility", "utility", "grid", "country", "global"]`. Maps directly to the 7-tier resolver fallback. |
| `selection.custom_filter` | `callable` | Optional picklable predicate — see `_jurisdiction_filter` in `electricity.py` for a documented example. |

---

## 3. Boundary Rules

Controls the **emission boundary** enforced on every factor returned by
the pack. The resolver refuses to use a factor whose scope / boundary
metadata falls outside the pack's `BoundaryRule`.

| Field | Type | Notes |
|---|---|---|
| `boundary.allowed_scopes` | `list[str]` | Subset of `{"1","2","3"}` and Scope 3 category strings (`"3.1"`, ... `"3.15"`). |
| `boundary.scope3_categories` | `list[str]` | Optional explicit allow-list of the 15 Scope 3 categories; omit for "all". |
| `boundary.system_boundary` | `enum` | One of `cradle_to_gate \| gate_to_gate \| cradle_to_grave \| consequential`. |
| `boundary.lca_mode` | `enum` | One of `attributional \| consequential`. Defaults to `attributional`. Consequential is only for ESRS E1 marginal analysis + scenario packs. |
| `boundary.allowed_boundaries` | `list[str]` | Canonical boundary tags e.g. `["combustion", "WTT", "WTW", "cradle_to_gate"]`. |
| `boundary.include_transmission_losses` | `bool \| null` | Scope 2 packs only; `null` = not applicable. |
| `boundary.functional_unit` | `str` | Product packs only. E.g. `"1 kWh delivered"`, `"1 tonne of steel"`, `"1 passenger-km"`. |

---

## 4. Inclusion / Exclusion Rules

Explicit allow/deny lists that sit **above** the selection rule so a
pack can whitelist / blacklist activities or gases without re-writing
the selection rule.

| Field | Type | Notes |
|---|---|---|
| `inclusion.activity_categories` | `list[str]` | If set, only activities whose category matches are admitted. Empty list = no restriction. |
| `inclusion.gases` | `list[str]` | If set, only records whose `gas_level_vector` covers a superset of these gases are admitted. |
| `exclusion.activity_categories` | `list[str]` | Hard deny. CBAM denies `electricity_imports_non_cbam`; PAS 2050 denies biogenic CH4 from managed forests. |
| `exclusion.gases` | `list[str]` | Hard deny, e.g. PAS 2050 excludes biogenic CO2 from the headline value. |
| `exclusion.source_ids` | `list[str]` | Hard deny of specific source datasets (e.g. `["ecoinvent_consequential_v3"]` for an attributional pack). |

Every deny-list entry MUST carry a comment pointing to the
paragraph of the underlying standard that drives the exclusion.

---

## 5. Gas-to-CO2e Basis

Per CTO non-negotiable #1: gas-level vectors are stored, CO2e is
derived at read-time via a GWP set.

| Field | Type | Notes |
|---|---|---|
| `gas_to_co2e.default_gwp_set` | `str` | E.g. `"IPCC_AR6_100"`. Canonical GWP-set identifier from `greenlang.data.gwp_sets`. |
| `gas_to_co2e.allowed_override_sets` | `list[str]` | E.g. `["IPCC_AR5_100", "IPCC_AR4_100"]`. Any set that a caller may switch to via `?gwp_set=` without a methodology-review. |
| `gas_to_co2e.horizon_years` | `int` | One of `20 \| 100 \| 500`. Defaults to 100. |
| `gas_to_co2e.metric` | `enum` | One of `GWP \| GTP \| GWP*`. Defaults to `GWP`. |
| `gas_to_co2e.gwp_star_allowed` | `bool` | Opt-in for methane-heavy packs (agriculture, LSR) only. |

---

## 6. Biogenic Carbon Treatment

| Value | Meaning |
|---|---|
| `fossil_only` | Biogenic CO2 is explicitly excluded from the pack's headline number. Used by CBAM, PAS 2050 §7.4. |
| `include_biogenic` | Biogenic CO2 is part of the headline value (LSR variants, full-lifecycle products). |
| `separate_report` | Biogenic CO2 is calculated and surfaced separately from the headline. Default for Corporate Scope 1/2/3, ISO 14067. |
| `neutral_with_lulucf` | Biogenic CO2 is assumed neutral PROVIDED an LULUCF accounting line covers land-use impacts. GHG Protocol legacy convention. |

Field: `biogenic_treatment: <enum>` (required).

---

## 7. Market-Instrument Treatment

Applies to Scope 2 electricity and selected Scope 3 energy packs.

| Instrument | Allowed when | Required quality criteria |
|---|---|---|
| RECs / I-RECs | Market-based Scope 2 packs only. | GHG Protocol Scope 2 QC 1-7 all satisfied; tracking-system documented; vintage within 15 months of consumption; single-count retirement proof. |
| GOs (EU) | Market-based Scope 2 EU packs. | AIB compliant; production-device attributes match consumption country or interconnected region. |
| PPAs (physical + virtual) | Market-based only. | Contractual proof of attributes; additionality not required for GHG Protocol but required for RE100. |
| Residual mix | Fallback when above fail. | Must come from an AIB / Green-e / NGA / METI / DESNZ / CER / KEMCO / EMA published source. |
| Carbon offsets / removals | NEVER in Scope 1-3 gross inventories; only in a separate "net" disclosure block. | VCS / Gold Standard / Puro.earth / Isometric verified AND ICVCM-CCP approved. |

| Field | Type | Notes |
|---|---|---|
| `market_instruments.treatment` | `enum` | One of `not_applicable \| allowed \| require_certificate \| prohibited`. |
| `market_instruments.quality_criteria` | `list[str]` | E.g. `["vintage_within_15mo", "single_count_retirement", "bundled_or_tracking_system_documented"]`. |
| `market_instruments.allowed_instruments` | `list[str]` | Subset of `{rec, i_rec, go, ppa_physical, ppa_virtual, residual_mix, offsets_separate_block}`. |

---

## 8. Region Hierarchy

Ordered fallback chain the resolver walks when a candidate set is
spatially ambiguous. Default chain is `DEFAULT_FALLBACK` in `base.py`
(7 steps, customer_override -> global_default). Packs with narrower
data (residual mix) override with a 4-5 step chain.

| Field | Type | Notes |
|---|---|---|
| `region_hierarchy` | `list[FallbackStep]` | Each step has `rank`, `label`, `description`. Rank 1 = customer override; highest rank = global default. |
| `region_hierarchy_fallback_trigger` | `list[str]` | Conditions that force advancement to the next rank — e.g. `["no_supplier_match", "dqs_below_threshold", "stale_factor"]`. |

---

## 9. Fallback Logic + "cannot resolve safely"

The resolver maps each pack's `region_hierarchy` onto the 7-tier
fallback order in `greenlang.factors.resolution.engine`. When no tier
returns a candidate that satisfies BOTH `SelectionRule.accepts` AND
`BoundaryRule` constraints, the resolver MUST return a structured
`NoSafeMatch` error instead of a weak default.

| Field | Type | Notes |
|---|---|---|
| `fallback.cannot_resolve_action` | `enum` | One of `raise_no_safe_match \| return_global_default \| return_null`. Default `raise_no_safe_match` — required for all Certified packs. |
| `fallback.global_default_tier_allowed` | `bool` | Opt-in permission to fall back to tier 7. Certified regulatory packs (CBAM, Battery) MUST set false. |
| `fallback.stale_factor_cutoff_days` | `int` | Any certified factor older than this count forces downgrade to preview status at resolution time. |

---

## 10. Reporting Labels

Tag set automatically applied to every emission record returned through
this pack. Used for:
- WTW / TTW split (freight, fuels).
- Scope 1 / Scope 2 / Scope 3 classification.
- Location-based vs market-based flag.
- Fossil vs biogenic flag.
- Regulatory-framework alignment (`CSRD_E1`, `CA_SB253`, `UK_SECR`, ...).

| Field | Type | Notes |
|---|---|---|
| `reporting_labels` | `list[str]` | Flat tag list. Every entry MUST exist in `greenlang.reporting.labels.REGISTERED_LABELS`. |
| `label_logic` | `list[LabelRule]` | Optional list of conditional rules — when a leg is diesel + heavy-freight -> add `WTW`; when grid factor is location-based -> add `GHG_Protocol_Scope2_LocationBased`, etc. |

---

## 11. Audit-Text Templates

Natural-language paragraph emitted in `/explain` payloads. Used by
auditors, regulators, and tenant admins to understand the computation.

Templates use Jinja-style placeholders. Required placeholders vary by
factor family and scope. Missing placeholders must degrade gracefully
(empty string) via the accompanying Python render helper.

| Field | Type | Notes |
|---|---|---|
| `audit_texts.template_file` | `str` | Path relative to `greenlang/factors/method_packs/audit_texts/` — e.g. `"corporate.j2"`. |
| `audit_texts.required_placeholders` | `list[str]` | E.g. `["factor_id", "source_org", "source_year", "gwp_basis"]`. |
| `audit_texts.optional_placeholders` | `list[str]` | E.g. `["supplier", "certificate", "pcaf_dqs", "battery_class"]`. |
| `audit_texts.per_family_blocks` | `dict[str,str]` | Mapping of `FactorFamily` to the sub-template filename that renders when the resolved factor is in that family. |

Standard Jinja blocks every pack template SHOULD define:

```jinja
{# header #}
{{ pack_name }} — factor {{ factor_id }} (source: {{ source_org }} {{ source_year }}).

{# boundary #}
Scope: {{ scope }} / Boundary: {{ system_boundary }} / GWP: {{ gwp_basis }}.

{# gases #}
{% if gas_vector %}Gas contribution: {% for gas, pct in gas_vector.items() %}{{ gas }}={{ "%.1f"|format(pct*100) }}%{% if not loop.last %}, {% endif %}{% endfor %}.{% endif %}

{# market instruments #}
{% if market_instrument %}Market instrument applied: {{ market_instrument }} ({{ certificate_id }}).{% endif %}

{# fallback trace #}
Fallback trace: {% for step in fallback_trace %}{{ step.rank }}->{{ step.label }}{% if not loop.last %}; {% endif %}{% endfor %}.
```

---

## 12. Deprecation Policy

| Field | Type | Notes |
|---|---|---|
| `deprecation.advance_notice_days` | `int` | Minimum 90; defaults 180 product, 365 corporate / Scope 2. |
| `deprecation.replacement_pointer` | `str` | `pack_id` of the successor pack. Required on every deprecation notice. |
| `deprecation.migration_notes` | `str` | Free-text guidance + link to methodology-review-board decision. |
| `deprecation.webhook_fan_out` | `list[str]` | Notification endpoints — default `["factors.deprecations", "factors.methodology"]`. |
| `deprecation.grace_period_days` | `int` | How long deprecated factors remain resolvable after the advance-notice window. |
| `deprecation.max_factor_age_days` | `int` | Any certified factor older than this count is rolled back to `preview`. |

---

## YAML Schema Block (authoritative)

```yaml
# method_pack.schema.yaml — v0.1 DRAFT (MP1/MP2)
$schema: https://greenlang.io/schemas/method_pack/v0.1.json
type: object
required:
  - pack_id
  - pack_name
  - version
  - status
  - standards_alignment
  - owner_methodology_lead
  - last_reviewed_iso
  - pack_source_file
  - selection
  - boundary
  - gas_to_co2e
  - biogenic_treatment
  - market_instruments
  - region_hierarchy
  - fallback
  - reporting_labels
  - audit_texts
  - deprecation

properties:

  # --- 1. Metadata -----------------------------------------------------
  pack_id: {type: string, pattern: "^[a-z][a-z0-9_]*$"}
  pack_name: {type: string, minLength: 6, maxLength: 120}
  version: {type: string, pattern: "^[0-9]+\\.[0-9]+\\.[0-9]+$"}
  status:
    type: string
    enum: [draft, preview, certified, deprecated]
  standards_alignment:
    type: array
    items: {type: string}
    minItems: 1
  deprecation_window_days: {type: integer, minimum: 90, default: 180}
  owner_methodology_lead: {type: string}
  last_reviewed_iso: {type: string, format: date}
  pack_source_file: {type: string}

  # --- 2. Factor selection rules --------------------------------------
  selection:
    type: object
    required: [allowed_families, allowed_formula_types, allowed_statuses]
    properties:
      allowed_families:
        type: array
        items:
          type: string
          enum:
            - emissions
            - heating_value
            - refrigerant_gwp
            - oxidation
            - carbon_content
            - grid_intensity
            - residual_mix
            - energy_conversion
            - material_embodied
            - transport_lane
            - waste_treatment
            - finance_proxy
            - land_use_removals
            - classification_mapping
      allowed_formula_types:
        type: array
        items:
          type: string
          enum:
            - direct_factor
            - combustion
            - lca
            - transport_chain
            - spend_proxy
            - residual_mix
            - carbon_budget
      allowed_statuses:
        type: array
        items: {type: string, enum: [draft, preview, certified, deprecated]}
      require_verification: {type: boolean, default: false}
      require_primary_data: {type: boolean, default: false}
      jurisdiction_hierarchy:
        type: array
        items: {type: string, enum: [country, region, subregion, utility, facility]}
      priority_tiers:
        type: array
        items: {type: string}

  # --- 3. Boundary rules ----------------------------------------------
  boundary:
    type: object
    required: [allowed_scopes, system_boundary, allowed_boundaries]
    properties:
      allowed_scopes:
        type: array
        items: {type: string, pattern: "^(1|2|3|3\\.[0-9]{1,2})$"}
      scope3_categories:
        type: array
        items: {type: string}
      system_boundary:
        type: string
        enum: [cradle_to_gate, gate_to_gate, cradle_to_grave, consequential]
      lca_mode:
        type: string
        enum: [attributional, consequential]
        default: attributional
      allowed_boundaries:
        type: array
        items: {type: string}
      include_transmission_losses: {type: [boolean, "null"]}
      functional_unit: {type: string}

  # --- 4. Inclusion / exclusion ---------------------------------------
  inclusion:
    type: object
    properties:
      activity_categories: {type: array, items: {type: string}}
      gases: {type: array, items: {type: string}}
  exclusion:
    type: object
    properties:
      activity_categories: {type: array, items: {type: string}}
      gases: {type: array, items: {type: string}}
      source_ids: {type: array, items: {type: string}}

  # --- 5. Gas-to-CO2e basis -------------------------------------------
  gas_to_co2e:
    type: object
    required: [default_gwp_set, horizon_years, metric]
    properties:
      default_gwp_set: {type: string}
      allowed_override_sets:
        type: array
        items: {type: string}
      horizon_years: {type: integer, enum: [20, 100, 500], default: 100}
      metric: {type: string, enum: [GWP, GTP, "GWP*"], default: GWP}
      gwp_star_allowed: {type: boolean, default: false}

  # --- 6. Biogenic carbon ---------------------------------------------
  biogenic_treatment:
    type: string
    enum: [fossil_only, include_biogenic, separate_report, neutral_with_lulucf]

  # --- 7. Market instruments ------------------------------------------
  market_instruments:
    type: object
    required: [treatment]
    properties:
      treatment:
        type: string
        enum: [not_applicable, allowed, require_certificate, prohibited]
      quality_criteria:
        type: array
        items: {type: string}
      allowed_instruments:
        type: array
        items:
          type: string
          enum: [rec, i_rec, go, ppa_physical, ppa_virtual, residual_mix, offsets_separate_block]

  # --- 8. Region hierarchy --------------------------------------------
  region_hierarchy:
    type: array
    minItems: 2
    items:
      type: object
      required: [rank, label, description]
      properties:
        rank: {type: integer, minimum: 1}
        label: {type: string}
        description: {type: string}
  region_hierarchy_fallback_trigger:
    type: array
    items: {type: string}

  # --- 9. Fallback logic ----------------------------------------------
  fallback:
    type: object
    required: [cannot_resolve_action, global_default_tier_allowed]
    properties:
      cannot_resolve_action:
        type: string
        enum: [raise_no_safe_match, return_global_default, return_null]
        default: raise_no_safe_match
      global_default_tier_allowed: {type: boolean, default: false}
      stale_factor_cutoff_days: {type: integer, minimum: 30}

  # --- 10. Reporting labels -------------------------------------------
  reporting_labels:
    type: array
    minItems: 1
    items: {type: string}
  label_logic:
    type: array
    items:
      type: object
      required: [when, add_labels]
      properties:
        when: {type: string}
        add_labels: {type: array, items: {type: string}}

  # --- 11. Audit text -------------------------------------------------
  audit_texts:
    type: object
    required: [template_file, required_placeholders]
    properties:
      template_file: {type: string, pattern: "\\.j2$"}
      required_placeholders:
        type: array
        items: {type: string}
      optional_placeholders:
        type: array
        items: {type: string}
      per_family_blocks:
        type: object
        additionalProperties: {type: string}

  # --- 12. Deprecation policy -----------------------------------------
  deprecation:
    type: object
    required: [advance_notice_days, replacement_pointer, grace_period_days]
    properties:
      advance_notice_days: {type: integer, minimum: 90, default: 180}
      replacement_pointer: {type: string}
      migration_notes: {type: string}
      webhook_fan_out:
        type: array
        items: {type: string}
      grace_period_days: {type: integer, minimum: 30, default: 365}
      max_factor_age_days: {type: integer, minimum: 365}
```

---

## Appendix A — Minimal-conformant pack example

```yaml
pack_id: corporate_scope1
pack_name: "Corporate Inventory - Scope 1 (GHG Protocol)"
version: "1.0.0"
status: certified
standards_alignment: [GHG_Protocol_Corporate, IFRS_S2, ISO_14064, CSRD_E1]
deprecation_window_days: 365
owner_methodology_lead: methodology-wg@greenlang.io
last_reviewed_iso: "2026-04-22"
pack_source_file: greenlang/factors/method_packs/corporate.py

selection:
  allowed_families: [emissions, heating_value, refrigerant_gwp, oxidation, carbon_content]
  allowed_formula_types: [direct_factor, combustion]
  allowed_statuses: [certified]
  jurisdiction_hierarchy: [country, region, subregion]
  priority_tiers: [supplier, facility, utility, country, global]

boundary:
  allowed_scopes: ["1"]
  system_boundary: gate_to_gate
  allowed_boundaries: [combustion]
  include_transmission_losses: null

inclusion:
  activity_categories: []           # no restriction
  gases: []
exclusion:
  activity_categories: []
  gases: []
  source_ids: []

gas_to_co2e:
  default_gwp_set: IPCC_AR6_100
  allowed_override_sets: [IPCC_AR5_100]
  horizon_years: 100
  metric: GWP
  gwp_star_allowed: false

biogenic_treatment: separate_report

market_instruments:
  treatment: not_applicable
  quality_criteria: []
  allowed_instruments: []

region_hierarchy:
  - {rank: 1, label: customer_override, description: "Tenant-supplied factor overlay"}
  - {rank: 2, label: supplier_specific, description: "Supplier or manufacturer disclosure"}
  - {rank: 3, label: facility_specific, description: "Facility / asset-specific measurement"}
  - {rank: 4, label: utility_or_grid_subregion, description: "Utility tariff or grid sub-region"}
  - {rank: 5, label: country_or_sector_average, description: "National / sectoral average"}
  - {rank: 6, label: method_pack_default, description: "Pack default for this method profile"}
  - {rank: 7, label: global_default, description: "Global default (last resort)"}
region_hierarchy_fallback_trigger: [no_supplier_match, stale_factor]

fallback:
  cannot_resolve_action: raise_no_safe_match
  global_default_tier_allowed: false
  stale_factor_cutoff_days: 1460   # 4 years

reporting_labels: [GHG_Protocol, IFRS_S2, ISO_14064, CSRD_E1, CA_SB253, UK_SECR, India_BRSR, TCFD, SBTi, CDP]
label_logic: []

audit_texts:
  template_file: corporate.j2
  required_placeholders: [factor_id, source_org, source_year, gwp_basis]
  optional_placeholders: [supplier, certificate, cat11_use_phase_block]
  per_family_blocks:
    refrigerant_gwp: corporate_refrigerant.j2
    land_use_removals: corporate_lsr_ref.j2

deprecation:
  advance_notice_days: 365
  replacement_pointer: corporate_scope1_v2
  migration_notes: "Bump AR-set when WG1 publishes new 100-year GWPs."
  webhook_fan_out: [factors.deprecations, factors.methodology]
  grace_period_days: 365
  max_factor_age_days: 1460
```
