# Concept — Method Pack

A **method pack** is a methodology binding. It captures the selection rules, boundary rules, gas treatment, biogenic-carbon handling, market-instrument rules, and fallback logic that a specific standard (GHG Protocol Corporate, ISO 14083, CBAM, PCAF, ...) imposes on factor choice. Every resolution call MUST specify a `method_profile`; the resolver refuses to return a factor whose record-level `method_profile` does not match the caller's declared methodology. This is CTO non-negotiable #6 — policy workflows never touch raw factors.

## The fourteen profiles

| `method_profile` | Pack |
|---|---|
| `corporate_scope1` | [Corporate](../method-packs/corporate.md) |
| `corporate_scope2_location_based` | [Corporate](../method-packs/corporate.md) + [Electricity](../method-packs/electricity.md) |
| `corporate_scope2_market_based` | [Corporate](../method-packs/corporate.md) + [Electricity](../method-packs/electricity.md) |
| `corporate_scope3` | [Corporate](../method-packs/corporate.md) |
| `product_carbon` | [Product Carbon](../method-packs/product_carbon.md) |
| `product_iso14067` | [Product Carbon](../method-packs/product_carbon.md) (ISO 14067 variant) |
| `product_pact` | [Product Carbon](../method-packs/product_carbon.md) (PACT Pathfinder variant) |
| `freight_iso14083_glec_wtw` | [Freight](../method-packs/freight.md) (WTW) |
| `freight_iso14083_glec_ttw` | [Freight](../method-packs/freight.md) (TTW) |
| `land_removals_ghgp_lsr` | [Land & Removals](../method-packs/land_removals.md) |
| `finance_proxies_pcaf` | [Finance](../method-packs/finance_proxy.md) |
| `eu_cbam` | [EU Policy](../method-packs/eu_policy.md) (CBAM) |
| `eu_dpp_battery` | [EU Policy](../method-packs/eu_policy.md) (Battery DPP) |
| `eu_dpp_textile` | [EU Policy](../method-packs/eu_policy.md) (Textile DPP) |

## What a pack controls

- **Factor selection** — allowed families, formula types, statuses, jurisdiction hierarchy, priority tiers.
- **Boundary** — allowed scopes, system boundary (cradle-to-gate / cradle-to-grave), WTW vs TTW, transmission losses yes/no.
- **Inclusion / exclusion** — activity categories, gases, and source IDs explicitly allowed or denied.
- **Gas-to-CO2e** — default `gwp_set` (AR6-100 by default), allowed overrides, horizon, metric (GWP / GTP / GWP*).
- **Biogenic treatment** — `fossil_only`, `include_biogenic`, `separate_report`, or `neutral_with_lulucf`.
- **Market-instrument treatment** — RECs, GOs, PPAs, residual mix, offsets (for Scope 2 electricity / selected Scope 3 energy packs).
- **Region hierarchy + fallback** — which tiers may be walked and what triggers advancement.
- **`cannot_resolve_action`** — `raise_no_safe_match` (required for Certified packs) vs weak defaults.
- **Reporting labels** — tags applied to every returned record (`WTW`, `GHG_Protocol_Scope2_LocationBased`, `CSRD_E1`, ...).
- **Audit-text template** — the paragraph emitted by `/explain`.

The normative template is in [`docs/specs/method_pack_template.md`](../../specs/method_pack_template.md).

## Pack lifecycle

Packs are themselves semver-versioned (`pack_id` + `version`). Statuses: `draft`, `preview`, `certified`, `deprecated`. Only `certified` packs are eligible for regulated disclosures (CBAM, SECR, CSRD). A certified pack has a minimum 180-day deprecation window (365 days for Corporate / Scope 2). Audit-text templates carry frontmatter signoff; unapproved templates are rendered with a `[Draft — Methodology Review Required]` banner and normative language stripped (see [`docs/specs/audit_text_template_policy.md`](../../specs/audit_text_template_policy.md)).

**See also:** [`factor`](factor.md), [API `/method-packs`](../api-reference/method-packs.md), [methodology manual](../../launch/methodology_manual.md).
