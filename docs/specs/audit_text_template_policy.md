# Method-Pack Audit Text Template Policy

**Prepared for:** Methodology Lead + Backend  
**Prepared on:** 2026-04-23  
**Purpose:** Let `/explain` payloads emit audit-text paragraphs today, using draft templates, with a review banner — without waiting for methodology lead signoff on every template. Flip `approved: true` per template as methodology lands.

---

## Why a policy?

Agent E's Wave 2 work includes authoring audit-text Jinja templates for Corporate, Electricity, and EU Policy packs. Each template is a paragraph emitted in `/explain` responses describing WHY a factor was chosen, under WHICH standard, with WHAT caveats. These paragraphs have regulatory weight: consultants and auditors cite them. They cannot be emitted without methodology-lead review.

Without this policy, every template change would block behind methodology review — making Agent E's work unusable for weeks. With this policy, templates emit today with a visible draft banner and a safety markup that strips any normative language until approved.

---

## Frontmatter contract

Every template file under `greenlang/factors/method_packs/audit_texts/*.j2` MUST start with:

```
{# ---
approved: false
approved_by: null
approved_at: null
methodology_lead: null
standard_citation: "GHG Protocol Corporate Standard §X.Y"
last_reviewed: null
--- #}
```

Fields:
- `approved` (bool) — flip to `true` only after methodology lead signoff
- `approved_by` (str) — name of approver
- `approved_at` (ISO timestamp) — when approved
- `methodology_lead` (str) — current methodology-lead owner
- `standard_citation` (str) — the specific clause/section of the standard the template aligns with. MANDATORY even for drafts. Forces template author to ground the text.
- `last_reviewed` (ISO timestamp) — when last reviewed; SLA: reviewed every 180 days minimum

---

## Renderer behaviour

`greenlang.factors.method_packs.render_audit_text(pack_id, factor)` MUST:

### When `approved: false`

1. Prepend the banner: `[Draft — Methodology Review Required — do not rely on for regulatory filing]`
2. Wrap the entire rendered body in a second banner at the end: `[End of Draft — This audit text has not been reviewed by the GreenLang Methodology Lead.]`
3. Strip normative language: any phrase matching these patterns is replaced with `[REDACTED-PENDING-REVIEW]`:
   - `"in compliance with"`, `"fully conforms to"`, `"certified under"`, `"approved for regulatory filing"`, `"audit-ready"`, `"authoritative"`, `"definitive"`
4. Emit the rendered text BUT: mark the `explain.audit_text_draft: true` in the JSON envelope so SDKs can flag it in UIs.
5. Log a metric: `factors_audit_text_draft_emitted_total{pack_id}` (counter) — visibility signal for methodology lead.

### When `approved: true`

1. Render plainly, no banner, no stripping.
2. Emit `explain.audit_text_draft: false`.

### When frontmatter is missing or malformed

1. FAIL the template load. Never emit.
2. Raise `AuditTextFrontmatterMissingError`.
3. Log a metric: `factors_audit_text_load_failure_total{pack_id, reason}`.

---

## CI guardrail

Add a CI check `.github/workflows/audit_text_gate.yml` that fails the build if:

1. Any Certified-status pack (`status: certified` in pack metadata) has any template with `approved: false`.
2. Any template's `last_reviewed` is more than 180 days old.
3. Any template body contains normative language (the stripped-pattern list above) AND `approved: false` — prevents drafts that sneak normative claims past the strip list.

Non-Certified packs (`draft`, `preview`) can ship with `approved: false` templates — that's the whole point of the policy.

---

## Methodology review flow

When methodology lead is ready to review a template:

1. Methodology lead reads the template body + the `standard_citation` frontmatter field.
2. If aligned with the standard: edit frontmatter, set `approved: true`, `approved_by`, `approved_at`, `last_reviewed` — commit directly.
3. If misaligned: return to author with comments; leave `approved: false`.
4. Methodology lead owns per-template review cadence (180 days).

No PR process required for an approval flip that doesn't change the template body. The frontmatter is the signoff record.

---

## Template author guardrails (for Agent E and future authors)

When writing a new or revised template:

1. **Cite the standard** — `standard_citation` is mandatory. If you can't cite a specific clause, don't write the template.
2. **Use placeholders, not claims** — `{{ factor.chosen_factor.name }}` good; "is the correct factor" bad. Describe the selection, don't assert correctness.
3. **Mark assumptions explicitly** — use `Based on the assumption that ...` or `Subject to ...` phrasing.
4. **Never claim regulatory sufficiency** — never write "audit-ready" or "CSRD-compliant". Let the customer's auditor decide.
5. **Keep it short** — 3–6 sentences. `/explain` is a developer-facing payload; long prose is noise.

---

## Example — Corporate Inventory Pack (draft template)

`greenlang/factors/method_packs/audit_texts/corporate.j2`:

```
{# ---
approved: false
approved_by: null
approved_at: null
methodology_lead: null
standard_citation: "GHG Protocol Corporate Standard (WRI/WBCSD, 2004) + Scope 2 Guidance (2015)"
last_reviewed: null
--- #}
Factor {{ factor.chosen_factor.name }} was selected under the Corporate Inventory method profile. The resolver evaluated the activity against GHG Protocol Corporate Standard scope boundaries and chose this factor at fallback tier {{ factor.fallback_rank }} of 7. Source: {{ factor.source.authority }} dataset version {{ factor.source.version }}, valid {{ factor.valid_from }} to {{ factor.valid_to }}. GWP basis: {{ factor.co2e_basis }}. Factor Quality Score: {{ factor.quality.composite_fqs_0_100 }}/100. {% if factor.assumptions %}Assumptions considered in selection: {{ factor.assumptions | join("; ") }}.{% endif %}
```

Renderer output when `approved: false` (the default):

```
[Draft — Methodology Review Required — do not rely on for regulatory filing]

Factor "India CEA grid electricity FY2027" was selected under the Corporate Inventory method profile. The resolver evaluated the activity against GHG Protocol Corporate Standard scope boundaries and chose this factor at fallback tier 5 of 7. Source: Central Electricity Authority of India dataset version 12.0, valid 2026-04-01 to 2027-03-31. GWP basis: IPCC_AR6_100. Factor Quality Score: 84/100. Assumptions considered in selection: Used for purchased grid electricity in India when supplier-specific factor is unavailable.

[End of Draft — This audit text has not been reviewed by the GreenLang Methodology Lead.]
```

Renderer output when `approved: true`:

```
Factor "India CEA grid electricity FY2027" was selected under the Corporate Inventory method profile. The resolver evaluated the activity against GHG Protocol Corporate Standard scope boundaries and chose this factor at fallback tier 5 of 7. Source: Central Electricity Authority of India dataset version 12.0, valid 2026-04-01 to 2027-03-31. GWP basis: IPCC_AR6_100. Factor Quality Score: 84/100. Assumptions considered in selection: Used for purchased grid electricity in India when supplier-specific factor is unavailable.
```

Same text; no banner; customer-facing audit bundle can cite it.

---

## Sign-off

| Role | Action | Owner |
|---|---|---|
| Policy adopted | Review + approve this file | Methodology Lead |
| Frontmatter schema implemented | Load/validate frontmatter in renderer | Backend (Agent E scope) |
| CI guardrail | Add `.github/workflows/audit_text_gate.yml` | DevOps |
| Per-template review cadence | 180-day review SLA | Methodology Lead |
| Template authoring guidelines | Read + follow when writing new templates | Template authors |

This policy enables v1 Certified edition to cut with Corporate, Electricity, and EU Policy pack drafts emitting audit text SAFELY today. Certified packs flip to `approved: true` as methodology review lands, at which point the banner disappears and the audit bundle ships clean.
