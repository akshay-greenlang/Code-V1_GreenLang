# GreenLang Factors — Methodology Manual

**Audience:** Methodology Lead, Methodology Working Group, reviewers, external assurance providers.
**Owner:** `methodology-wg@greenlang.io`.
**Purpose:** Reference manual for how GreenLang Factors handles methodology — which standards are supported, how packs are authored, how fallback and FQS work, how audit-text templates are reviewed, and the review cadence.

---

## 1. Supported standards

### 1.1 Corporate inventory

- **GHG Protocol Corporate Accounting and Reporting Standard** (WRI/WBCSD, 2004; revised 2015). [Link](https://ghgprotocol.org/corporate-standard).
- **GHG Protocol Scope 2 Guidance** (2015). [Link](https://ghgprotocol.org/scope_2_guidance).
- **GHG Protocol Scope 3 Standard** (2011) + **Scope 3 Calculation Tool** (2013).

### 1.2 Product

- **ISO 14067:2018** — Carbon footprint of products. [Link](https://www.iso.org/standard/71206.html).
- **WBCSD PACT Pathfinder Framework v3.0** — Product carbon data exchange. [Link](https://www.carbon-transparency.com/).
- **GHG Protocol Product Standard** (2011).

### 1.3 Transport

- **ISO 14083:2023** — GHG emissions from transport chain operations. Clauses 5, 6, 7, 9, 10.
- **SFC GLEC Framework v3.0** (2023) — aligned with ISO 14083.

### 1.4 Land & Removals

- **GHG Protocol Land Sector and Removals Guidance** (2024).
- **IPCC 2006 Guidelines Vol 4 (AFOLU) + 2019 Refinement** — Tier 1 defaults.
- **SBTi FLAG** — forest, land, agriculture target setting.

### 1.5 Finance

- **PCAF Global GHG Accounting Standard Part A (v2.0, 2022)** — financed emissions.
- **PCAF Part B (2024)** — facilitated emissions.
- **PCAF Part C** — insurance-associated emissions.

### 1.6 EU regulatory

- **Regulation (EU) 2023/956** — CBAM. Annex I (in-scope goods), Annex III (default values).
- **Implementing Regulation (EU) 2023/1773** — CBAM transitional period reporting.
- **Regulation (EU) 2023/1542** — Batteries Regulation + Annex II carbon footprint methodology.
- **Regulation (EU) 2024/1781** — Ecodesign for Sustainable Products Regulation (ESPR) / DPP framework.

### 1.7 Underlying physics

- **IPCC AR6** (2021) — default GWP set `IPCC_AR6_100`.
- **IPCC 2006 Guidelines** — underlying combustion factors.

Full standards mapping: [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md) §7.

---

## 2. Method pack authoring rules

Normative template: [`docs/specs/method_pack_template.md`](../specs/method_pack_template.md).

### 2.1 Metadata (required)

- `pack_id`, `pack_name`, `version` (semver), `status` (draft / preview / certified / deprecated).
- `standards_alignment[]` — every standard the pack claims conformance with.
- `deprecation_window_days` — min 90, default 180, 365 for Corporate/Scope 2.
- `owner_methodology_lead` — accountable human.
- `last_reviewed_iso` — date of last review-board signoff.

### 2.2 Selection rules

- `allowed_families`, `allowed_formula_types`, `allowed_statuses`.
- `require_verification` — forces `verification.status in {external_verified, regulator_approved}`.
- `require_primary_data` — forces `primary_data_flag in {primary, primary_modeled}`.
- `jurisdiction_hierarchy` — most-specific first.
- `priority_tiers` — maps to the 7-tier resolver fallback.

### 2.3 Boundary rules

- `allowed_scopes` + `scope3_categories`.
- `system_boundary` — cradle_to_gate / gate_to_gate / cradle_to_grave / consequential.
- `lca_mode` — attributional (default) or consequential.
- `functional_unit` — required for product packs.

### 2.4 Gas-to-CO2e

- `default_gwp_set` — `IPCC_AR6_100` for all new packs.
- `allowed_override_sets` — subset callers may request via `?gwp=...`.
- `horizon_years` — 20 / 100 / 500 (default 100).
- `metric` — GWP / GTP / GWP*. GWP* only for methane-heavy packs with explicit opt-in.

### 2.5 Biogenic treatment

`fossil_only` / `include_biogenic` / `separate_report` / `neutral_with_lulucf`.

Default: `separate_report` for Corporate Scope 1/2/3 and ISO 14067. `fossil_only` for CBAM (per Annex I). `include_biogenic` for cradle-to-gate PACT products.

### 2.6 Fallback

`cannot_resolve_action` — MUST be `raise_no_safe_match` for all Certified packs. `global_default_tier_allowed` — MUST be `false` for CBAM / Battery DPP / Textile DPP and any regulatory Certified pack.

---

## 3. Factor Quality Score (FQS) rubric

Canonical definition: [`docs/developer-portal/concepts/quality_score.md`](../developer-portal/concepts/quality_score.md).

| Component | Weight | Scale (reviewer rubric) |
|---|-----:|---|
| `temporal_score` | 0.25 | 5=overlaps reporting period; 4=within 1yr; 3=within 3yr; 2=within 5yr; 1=>10yr or undated. |
| `geographic_score` | 0.25 | 5=facility/grid-subregion exact; 4=country exact; 3=region; 2=continental; 1=global. |
| `technology_score` | 0.20 | 5=primary supplier; 4=industry-specific verified; 3=industry average; 2=sector proxy; 1=broad EEIO. |
| `verification_score` | 0.15 | 5=regulator-approved; 4=third-party verified (ISO 14065); 3=publisher-verified; 2=internal review; 1=unverified. |
| `completeness_score` | 0.15 | 5=all in-scope gases + biogenic + F-gases; 4=CO2+CH4+N2O; 3=CO2+CH4; 2=CO2 only; 1=partial coverage. |

Composite formula: `composite_fqs = 20 * (0.25*T + 0.25*G + 0.20*Tech + 0.15*V + 0.15*C)`.

Reviewer obligation: every `active` factor MUST have all five components populated with a defensible justification (stored in `quality.rationale_per_component`).

---

## 4. Audit-text template authoring

Policy: [`docs/specs/audit_text_template_policy.md`](../specs/audit_text_template_policy.md).

### 4.1 Frontmatter contract

Every template at `greenlang/factors/method_packs/audit_texts/*.j2` MUST start with:

```jinja
{# ---
approved: false
approved_by: null
approved_at: null
methodology_lead: null
standard_citation: "GHG Protocol Corporate Standard §4.1"
last_reviewed: null
--- #}
```

Until a Methodology Lead flips `approved: true`, the renderer prepends `[Draft — Methodology Review Required — do not rely on for regulatory filing]` and strips normative phrases (`"in compliance with"`, `"certified under"`, etc.).

### 4.2 Writing rules

1. **Cite the standard.** `standard_citation` is mandatory with clause/section. If you cannot cite, do not write the template.
2. **Use placeholders, not claims.** `{{ factor.chosen_factor.name }}` is good; "is the correct factor" is bad.
3. **Mark assumptions explicitly.** Use `"Based on the assumption that ..."` or `"Subject to ..."`.
4. **Never claim regulatory sufficiency.** Do not write "audit-ready", "CSRD-compliant", or "third-party assured". The customer's auditor decides.

### 4.3 Approval flow

1. Methodology Lead reads template body + `standard_citation`.
2. If aligned: edit frontmatter, set `approved: true`, `approved_by`, `approved_at`, `last_reviewed`. Commit directly.
3. If misaligned: return to author with comments; `approved: false` stays.

No PR process required for an approval flip that does not change the template body.

---

## 5. Review cadence (180 days)

Every 180 days, the Methodology WG reviews:

1. **Certified packs** — regression check against latest standard text (e.g., IPCC corrigenda, GHG Protocol clarifications).
2. **Audit-text templates** — `last_reviewed` must be within 180 days (CI guardrail fails otherwise for Certified packs).
3. **FQS component scores** — spot-check a random sample of 20 factors per family for score justification.
4. **Gold set** — refresh the canonical 150-request gold set against the latest edition.
5. **Standards changelog** — publish a minor version bump of any pack whose underlying standard has released an erratum.

CI guardrail: `.github/workflows/audit_text_gate.yml` fails the build if any Certified pack carries an unreviewed template.

---

## 6. Deprecation and successor handling

When a factor is deprecated:

1. Set `status = "deprecated"`.
2. Populate `explainability.rationale` with the successor `factor_id` and reason.
3. Keep serving the deprecated row for the deprecation window (180 / 365 days).
4. Emit `factor.deprecated` webhook.

When a method pack is deprecated:

1. Set pack `status = "deprecated"`.
2. Specify `replacement_pack_id` in metadata.
3. Notify tenants using the pack (via dashboard + email).
4. Keep serving for `deprecation_window_days`.

---

## 7. Related

- Architecture: [`cto_architecture_deck.md`](cto_architecture_deck.md).
- Engineering: [`engineering_runbook.md`](engineering_runbook.md).
- Legal: [`legal_source_rights_binder.md`](legal_source_rights_binder.md).
- Public concepts: [`docs/developer-portal/concepts/method_pack.md`](../developer-portal/concepts/method_pack.md).
- Audit-text policy: [`docs/specs/audit_text_template_policy.md`](../specs/audit_text_template_policy.md).
