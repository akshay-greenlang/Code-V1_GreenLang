# Design Partner Feedback Memo (Template)

**To be completed by:** Partner technical lead + business sponsor (joint sign-off).
**Submission target:** End of FY27 Q1 (`2026-06-30`).
**File on receipt as:** `docs/factors/design-partners/feedback/<partner-slug>-memo-<quarter>.md` (e.g., `IN-EXPORT-01-memo-2026Q1.md`).
**Explicit non-goal:** This is **NOT** a vendor NPS, customer-satisfaction score, or any single-number metric. It is a written, qualitative memo intended to inform the alpha→beta transition gate and v0.5 roadmap.

---

## Header

| Field | Value |
|-------|-------|
| Partner slug | `<IN-EXPORT-01 | EU-MFG-01>` |
| Pilot window | `<start-date>` to `<end-date>` |
| Total SDK calls during pilot | `<integer>` |
| Total successful end-to-end calculations | `<integer>` |
| Memo author (technical lead) | `<name>` |
| Memo co-author (business sponsor) | `<name>` |

---

## Section A — Workflow Fit (1 paragraph)

> Describe in your own words how the GreenLang Factors SDK fit (or did not fit) into the calculation workflow your team already runs. What process step did it replace? What process step did it add? Where did it slot in cleanly, where did it create friction? One paragraph, no bullet points.

`<your paragraph here>`

---

## Section B — SDK Ergonomics

### What worked (3 bullets)

1. `<bullet 1>`
2. `<bullet 2>`
3. `<bullet 3>`

### What didn't (3 bullets)

1. `<bullet 1>`
2. `<bullet 2>`
3. `<bullet 3>`

---

## Section C — Data Quality Observations

Comment on factor data quality across the dimensions below. Be specific (cite a `factor_id` or a `source` if you can).

- **Accuracy** (factor values vs. your internal reference): `<comments>`
- **Completeness** (coverage of activities/regions/CN codes you needed): `<comments>`
- **Vintage gaps** (currency of factor data; any out-of-date sources you hit): `<comments>`

---

## Section D — Provenance Audit Experience

> The platform claims every factor lookup carries a SHA-256 provenance bundle traceable to its source document. Could you, in practice, take a single factor result your team produced and trace it back to the originating publication (e.g., DEFRA 2025, India CEA baseline, EU CBAM Annex I)? Walk through one concrete trace you attempted.

`<your trace narrative here, including which factor_id you traced and how far back you got>`

---

## Section E — Three Things That Would Block GA Adoption

If GreenLang Factors went GA tomorrow at v1.0, what 3 things in its current shape would stop your team from rolling it out broadly inside your company? Be concrete; "the SDK is too slow" is not concrete — "median get_factor latency exceeded 800ms which is unworkable for our 50k-row batch job" is concrete.

1. `<blocker 1>`
2. `<blocker 2>`
3. `<blocker 3>`

---

## Section F — Would You Recommend (Qualitative)

> Would you recommend GreenLang Factors to a peer team in your industry today, at v0.1 alpha? Answer in your own words: yes / no / yes-with-caveats, and explain why. **Do NOT** provide a 0-10 numeric NPS-style score; the question is intentionally qualitative.

`<your answer here>`

---

## Sign-off

| Role | Name | Date | Signature / Acknowledgement |
|------|------|------|-----------------------------|
| Partner technical lead | `<name>` | `<YYYY-MM-DD>` | `<acknowledged>` |
| Partner business sponsor | `<name>` | `<YYYY-MM-DD>` | `<acknowledged>` |
| GreenLang PM (counter-sign) | `<name>` | `<YYYY-MM-DD>` | `<acknowledged>` |
