# Design Partner Pilot Playbook — 2 Weeks

**For:** Founders, GTM Lead, lead engineer, and the first 5-8 FY27 design partners
**Objective:** Convert a design partner from conversation to a green-light annual contract (or a gracefully-no) in **14 calendar days**
**Pilot fee:** $25,000, **cash-back guarantee** if we cannot produce a path to production
**Targets:** India-linked manufacturers/exporters into EU supply chains, California SB 253 Scope 1+2 reporters (Aug 10 2026 deadline), EU CSRD year-one filers
**Ground truth:** `docs/product/PRD-FY27-Factors.md` §11 (KPIs), `docs/editions/v1-certified-cutlist.md` (slice readiness), `FY27_vs_Reality_Analysis.md` §7 (pitch spine)

---

## Week 1 — Data in. First audit bundle out.

### Mon (Day 1) — Kickoff

- 09:00 — 60-min video kickoff. Attendees: customer sustainability controller, customer IT lead, GreenLang founder or GTM lead, GreenLang solutions engineer.
- Agenda: introductions, scope confirmation, success criteria review (see §Success criteria below), NDA signature, data-transfer plan, comms channel (Slack Connect or shared Teams).
- NDA signed and countersigned by EOD. Use the **mutual 1-page NDA** in `templates/nda-v1.md`; nothing else.
- Solutions engineer sends the intake email (template below) by EOD.

### Tue (Day 2) — Tenant provisioned, CSV template delivered

- 09:00 — Engineer runs `gl pilot provision --customer <slug>` (maps to `greenlang/factors/pilot/provisioner.py`). Tenant is live in staging in <10 minutes with unique API keys, overlay vault, and audit-bundle scope.
- 10:00 — Customer receives: (a) Staging portal URL, (b) API key in a Bitwarden share, (c) CSV intake template with 15 required columns (activity_category, sub_category, classification_codes, jurisdiction country/region, date, quantity, unit, supplier_id, facility_id, scope, method_profile, notes, file_attachment_ref, currency_if_spend, data_quality_flag).
- 11:00 — Customer commits to delivering **50 representative activity rows** by EOD Wednesday. Rows should span the customer's 3-5 most material emission categories.
- Customer questionnaire sent (template below).

### Wed (Day 3) — Data resolved through the API, engineer-in-the-room

- 09:00 — Customer uploads 50 rows to the pilot staging tenant.
- 09:30 - 12:30 — **Working session. Screen-share.** GreenLang engineer runs `/v1/factors/batch` with customer engineer watching every call. Each row returns a `ResolvedFactor` with `fallback_rank`, `explanation`, `alternates`, `signed_receipt`. Customer sees live how the 7-step cascade picks step-1 (their own supplier override, if any), steps 2-6 (our method pack selection), or step 7 (global default).
- Expected outcome: ≥40 of 50 rows resolve on first pass with `fallback_rank ≤ 5` (tenant/supplier/facility/region/method-pack specific).
- For rows that do not resolve cleanly, `suggestion_agent.py` offers 3-5 alternate matches; customer picks the right one together with engineer and the decision is recorded as a mapping.

### Thu (Day 4) — Second-pass resolve + cascade review

- 09:00 — Re-run the 50 rows after mapping corrections. Target: **100% resolve, ≥80% top-1 match, 100% explain-coverage** (see Success criteria).
- 10:00 — Customer walks `/v1/factors/{id}/explain` for 5 random resolutions. They read the methodology note from `method_packs/*.py` and confirm it matches their internal interpretation.
- 14:00 — Any factor they dispute gets logged as either (a) a tenant overlay, via `POST /v1/overlays`, or (b) a mapping issue, escalated to GreenLang matching team for a hot fix.

### Fri (Day 5) — First audit bundle generated

- 10:00 — Engineer runs `POST /v1/audit-bundle` with `{scope: "run_id", run_id: <Wed run>}`. Returns a signed ZIP in ~60 seconds.
- 10:30 — Customer controller opens the ZIP in Excel. Bundle contains: one JSON file per resolved factor, every raw source artifact (PDF/CSV), parser logs, reviewer decisions, SHA-256 chain.
- 11:00 — **Week-1 demo call.** Customer walks the bundle, GreenLang walks the cascade, everyone confirms they can hand this to an auditor.
- Week-1 exit checkpoint: customer says "yes, keep going" OR "stop — this won't work because X." No middle.

---

## Week 2 — Methodology review, ERP integration, sign-off.

### Mon (Day 8) — Methodology review with customer's sustainability controller

- 09:00 - 12:00 — Deep review session. Customer controller + their designated auditor (if they have engaged one) + GreenLang methodology engineer.
- Walk the method pack the customer depends on most (CBAM / CSRD E1 / SB 253 Scope 1+2).
- Read the full methodology note (`method_packs/<profile>.py :: methodology_notes()`), map each selection rule to the regulatory citation, confirm the biogenic/LCA/GWP handling matches their reading.
- Log any methodology divergence as a private factor pack or tenant overlay. **Divergences that cannot be resolved in code = pilot failure — we return the fee.**

### Tue (Day 9) — Methodology sign-off + overlay approval flow

- 09:00 — Customer creates 3-5 tenant overlays for supplier-specific or facility-specific factors via `POST /v1/overlays`. We walk the 4-eyes approval workflow: author → reviewer (different user) → publisher. Enforced by `review_workflow.py`.
- 11:00 — Customer's methodology lead signs off on the approach. Decision recorded in `pilot/feedback.py`.

### Wed (Day 10) — Integration demo: Excel + ERP API call

- 09:00 - 10:00 — **Excel export demo.** Run `GET /v1/factors/export?format=xlsx` with the customer's filter (last quarter's inventory). Open in Excel. Confirm one row per resolution with factor_id, version, source, fallback_rank, explanation, license class.
- 10:00 - 12:00 — **ERP API call demo.** Customer IT pulls up their SAP / Oracle / Workday / Databricks environment. Together we write a 30-line integration script that calls `POST /v1/factors/batch` against their actual activity-data table. Target: 10,000 rows resolved in <60 seconds. This is the moment the customer sees the platform, not the pilot.
- 14:00 — Provide them with sample integration code in Python + TypeScript from `greenlang/factors/sdk/`.

### Thu (Day 11) — Sign-off discussion

- 14:00 - 15:30 — **Commercial conversation.** Founder or GTM lead leads. Attendees: customer CFO or procurement, customer sustainability controller.
- Present: pilot results (resolved rows, top-1 match rate, audit bundle, methodology sign-off, ERP integration).
- Present: annual quote — $75k single-installation, $125k-$200k multi-installation, plus relevant Premium packs (ref. `docs/gtm/FACTORS-API-PRICING-SHEET.md`).
- Ask: "Can you confirm green-light for the annual contract in writing by Friday, or should we wrap the pilot and refund?"

### Fri (Day 12) — Green-light or gracefully-no

- 10:00 — Customer decision due by email.
- **Green-light path:** MSA + DPA sent by 14:00. Production tenant provisioned Monday of Week 3 via `onboarding/partner_setup.py`. Stripe or manual invoice billed per choice.
- **Gracefully-no path:** pilot fee refunded via same payment channel by 17:00. One-page **"What would make this work" memo** delivered by EOD — what has to be true (new method pack, new source, new integration, new pricing) for a future engagement. Relationship preserved.

---

## Success criteria (all must be true to claim a successful pilot)

| Metric | Target | Measured via |
|---|---|---|
| Top-1 factor match on the 50 activities | **≥ 80%** | `matching/evaluation.py` run on customer set |
| Explain-coverage | **100%** — every resolved row returns a valid 7-step cascade | `GET /v1/factors/{id}/explain` inspection |
| Audit bundle export | **Opens in Excel**, customer controller can read every cell | Week 1 Fri demo |
| Methodology sign-off | Customer's sustainability controller + their auditor (if engaged) agree the method-pack interpretation is defensible | Week 2 Mon session + signed memo |
| Customer next-questions | Customer names the **next 5 questions** they want GreenLang to answer (new method packs, new sources, new integrations) | Captured in `pilot/feedback.py` structured form |
| Written green-light or gracefully-no | By Day 12 Friday 10:00 | Email archived |

If any item is missed and cannot be remediated by Week 2 Wednesday, we refund the pilot fee. No exceptions.

## Exit metrics to capture (for every pilot, green-light or not)

These roll up into the `pilot/registry.py` + `pilot/telemetry.py` dashboards. All 5-8 pilots feed the same board.

1. **Time to first resolve** (from API key issue to first successful `/v1/factors/resolve` 200 response). Target <1 hour.
2. **Top-1 match rate** on the customer's 50-row gold set. Target ≥80%.
3. **Explain-coverage rate.** Target 100%.
4. **Audit-bundle download time.** Target <90 seconds for a 50-row run.
5. **Methodology-divergence count.** Count of disputed factor routings. Target ≤5.
6. **Tenant overlay count created.** Actual number; no target — it signals domain depth.
7. **ERP integration time** (first successful batch call from customer's production ERP). Target <2 hours.
8. **NPS from customer controller + IT lead + CFO.** Target ≥40 per PRD §11.2.
9. **Green-light rate across all pilots.** Target ≥50% of 5-8 pilots = 3-4 paid annual logos. Combined with 3 Developer Pro self-service signups + 2 Consulting/Platform closes, this delivers the FY27 target of 8 paying logos per PRD §1.5.
10. **Regrets captured.** What did the customer ask for that we do not yet have? Product backlog input.

---

## Template — Intake email (send by EOD Day 1)

**Subject:** GreenLang Factors Pilot — Kickoff materials + data checklist

Hi {first name},

Thank you for confirming the 2-week pilot. This email has everything you need for Day 2.

**What you will get from us today:**
1. Staging portal URL (will arrive in a separate Bitwarden share).
2. One API key scoped to your pilot tenant.
3. A CSV intake template (attached) with 15 required columns.
4. A 1-page questionnaire (attached) so our methodology engineer can prep the right method pack for your sign-off session in Week 2.

**What we need from you by EOD Wednesday, Day 3:**
- **50 representative activity rows** covering your 3-5 most material emission categories, populated into the CSV template. These can be real data or anonymized/scaled; it does not matter for the pilot. What matters is that they represent the shape of data you report quarterly.
- Name + email of: (a) your sustainability controller, (b) your IT integration lead, (c) your designated CBAM/CSRD/SB253 auditor (if engaged).
- Confirmation of your current factor sources (IPCC defaults, DEFRA, EPA, country-specific, consultant spreadsheet, etc.) so we can run a cross-source consistency check alongside the pilot.

**What happens next:**
- Wednesday 09:00-12:30 local: screen-share working session where we run your 50 rows through the API live. Customer engineer on the call, GreenLang engineer on the keyboard.
- Thursday: second-pass resolve + cascade review.
- Friday: first audit bundle generated + demo call at 11:00 local.

**Our cash-back guarantee:** if by Day 12 (end of Week 2) we have not produced a path to production that your methodology controller + IT lead + CFO can all green-light, we refund the $25,000 pilot fee and hand you a 1-page "what would make this work" memo. No lock-in.

Shared channel: Slack Connect invite landing in your inbox in the next 30 minutes.

Questions: reply here or ping me on Slack.

{GTM Lead name}
GreenLang | sales@greenlang.io

---

## Template — Questionnaire (send with the intake email)

*Answer inline, return by EOD Day 3. ~12 questions, 20 minutes.*

1. **Regulatory driver.** What triggered this pilot — CBAM quarterly report, CSRD year-one filing, SB 253 Aug 2026 deadline, internal board pressure, other?
2. **Coverage.** Which of the 7 method packs matter to you in the next 90 days? Rank 1-7: corporate Scope 1, corporate Scope 2 (location), corporate Scope 2 (market), corporate Scope 3, product carbon / LCA, freight (ISO 14083), land & removals, finance proxy (PCAF), EU CBAM.
3. **Jurisdictions.** Where do you operate? Where do you sell? Which grid(s) do you consume?
4. **Installations / sites.** How many plants, facilities, offices, or legal entities are in scope?
5. **CN codes (CBAM pilots only).** Which CN codes do you export to the EU?
6. **Current practice.** How do you resolve emission factors today — internal spreadsheet, consultant, commercial tool (which?), open source?
7. **Auditor.** Who is your auditor (Big-4, local firm, none yet)? Have they raised specific objections to your current factor choices?
8. **Factor sources in use.** Which factor sources do you currently cite (IPCC, IEA, EPA, DEFRA, CEA India, AIB, ecoinvent, country-specific, industry average)?
9. **Integration target.** Will we feed an ERP (which one), a data warehouse (Snowflake / Databricks / BigQuery), or an Excel workflow?
10. **Scale.** How many activity rows per quarter do you process? Today: ___. Expected in 12 months: ___.
11. **Deal-breakers.** What would make you say "no" on Day 12? (Data residency, SSO, price point, methodology divergence, missing source, integration type.)
12. **Success statement.** Complete this sentence: "This pilot is a success if by Day 12, GreenLang has shown us that ____." (One sentence.)

---

## Pilot checklist — one screen for the GreenLang engineer

```
[  ] Day 1 Mon  — Kickoff call done, NDA signed, Slack channel live
[  ] Day 1 Mon  — Intake email + questionnaire sent
[  ] Day 2 Tue  — `gl pilot provision --customer <slug>` executed; tenant live
[  ] Day 2 Tue  — API key + portal URL delivered via Bitwarden
[  ] Day 3 Wed  — 50 activity rows received, uploaded, first batch resolve run
[  ] Day 3 Wed  — `fallback_rank` distribution recorded for all 50 rows
[  ] Day 4 Thu  — Mapping issues resolved, second-pass resolve hits ≥80% top-1
[  ] Day 4 Thu  — `/v1/factors/{id}/explain` reviewed on 5 random rows
[  ] Day 5 Fri  — `POST /v1/audit-bundle` executed; ZIP delivered
[  ] Day 5 Fri  — Week-1 demo call; customer writes "keep going" or "stop"
[  ] Day 8 Mon  — Methodology review session; divergences logged
[  ] Day 9 Tue  — 3-5 tenant overlays created via 4-eyes approval flow
[  ] Day 10 Wed — Excel export demo + ERP live batch call (≥10k rows)
[  ] Day 11 Thu — Commercial discussion; quote sent in writing
[  ] Day 12 Fri — Green-light email archived, OR refund issued + memo delivered
[  ] Day 12 Fri — Exit metrics captured in pilot/telemetry
[  ] Day 12 Fri — `pilot/feedback.py` form submitted with "next 5 questions"
```

---

*Playbook v1.0. Owned by GTM Lead. Updated after every completed pilot. A pilot that does not strictly follow this 14-day cadence is flagged to the founder on Day 8 for rescope or abort.*
