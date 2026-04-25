# Methodology Exception — DESNZ UK GHG Conversion Factors

**source_id:** `desnz_ghg_conversion`
**Current `source_version`:** `2024.1`
**CTO target vintage:** `2025`
**Status:** `preview`
**Created:** 2026-04-25
**Re-audit due:** alpha launch (target 2026-Q2)

## Why the CTO target vintage is not yet shipped

The UK Department for Energy Security and Net Zero (DESNZ) publishes
the *Greenhouse gas reporting: conversion factors* spreadsheet
mid-year. The 2025 spreadsheet was scheduled for July 2025; ingestion
into the GreenLang catalog has been queued but the parser fixture
update is not yet committed (the wired `parse_desnz_uk` accepts the
2025 column shape unchanged — the gap is purely in the catalog seed,
not the parser).

The 2024 conversion factors are the current shipped vintage and
remain valid for backward-looking 2024-FY reporting; v0.1 alpha
explicitly advertises that customers using DESNZ-rooted factors are
on the 2024 vintage until the audit at launch flips the status.

## What is lost vs the target

The DESNZ 2024 -> 2025 release cycle revises:

- **Stationary fuel factors** (natural gas, gas oil, kerosene): typical
  drift <1% on the headline `kgCO2e/kWh` figure. These are dominated
  by stoichiometric chemistry of combustion and are extremely stable
  year-over-year.
- **Grid electricity factors** (UK Scope 2): typical drift 4-7% as the
  UK grid continues to decarbonise. A factor that was 0.207 kgCO2e/kWh
  in DESNZ 2024 is expected to be ~0.193 kgCO2e/kWh in DESNZ 2025
  (representative; subject to ingest verification).
- **Refrigerant GWPs**: under DESNZ 2025 the AR6 GWP set is preserved
  (R-32 GWP-100 stays at 675; R-410A stays at 2088). No expected
  change in alpha refrigerant outputs.
- **Scope 3 freight & business travel**: typical drift <2%; negligible
  for alpha use cases.
- **WTT (well-to-tank)**: typical drift <2%.

## Materiality estimate (representative factors)

| Factor family | DESNZ 2024 -> 2025 expected drift | Alpha materiality |
|---|---|---|
| Natural gas combustion | <1% | LOW |
| Liquid fuels | <1% | LOW |
| UK grid electricity (Scope 2) | 4-7% | MEDIUM (alpha customers using UK Scope 2 see a vintage stamp) |
| Refrigerant GWPs | unchanged (AR6 stable) | NONE |
| Freight (HGV, rail) | <2% | LOW |
| Business travel (flights) | <2% | LOW |
| WTT | <2% | LOW |
| Materials (paper) | <2% | LOW |

## Mitigation

The SDK does **not** surface this exception in its API response; v0.1
alpha is shipped under design-partner agreement. Mitigation lives in:

1. **API response provenance fields**: every alpha factor now carries
   `provenance.source_year` (= 2024) and `provenance.version`
   (= "2024" or "2024.1"). Customers can filter on this.
2. **Documentation callout**: this file plus the SOURCE-VINTAGE-AUDIT
   appear in the alpha onboarding pack (linked from
   `docs/factors/v0_1_alpha/QUICKSTART.md` once that lands).
3. **Re-audit trigger**: the methodology lead is required to verify
   DESNZ 2025 availability at the alpha-launch checkpoint and bump
   `source_version` to `2025.1` if the spreadsheet is published.

## Re-audit due date

**Target:** 2026-Q2 (alpha launch). The re-audit MUST confirm:
- DESNZ has or has not published the 2025 spreadsheet.
- If yes: parse with the current `parse_desnz_uk`, verify column
  shape unchanged via the snapshot test, bump
  `source_version` to `2025.1`, flip `alpha_v0_1_status` to `locked`,
  archive this file.
- If no: re-date this exception, push the next re-audit to the date
  DESNZ confirms publication; do not block alpha launch.

## Sign-off

methodology lead: human:methodology-lead@greenlang.io  approved_at: <pending — sign before alpha launch>
