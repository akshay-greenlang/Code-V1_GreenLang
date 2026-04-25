# Methodology Exception — US EPA eGRID

**source_id:** `egrid`
**Current `source_version`:** `2022.1`
**CTO target vintage:** `2024`
**Status:** `preview`
**Created:** 2026-04-25
**Re-audit due:** alpha launch (target 2026-Q2)

## Why the CTO target vintage is not yet shipped

eGRID (Emissions and Generation Resource Integrated Database) is
published by the US EPA on a roughly annual cycle, but with a 2-year
lag — eGRID 2022 (covering reporting year 2022) was the latest
published edition as of the audit date. The cadence has historically
been:

- eGRID 2020 — published Feb 2022
- eGRID 2021 — published Jan 2023
- eGRID 2022 — published Q1 2024
- eGRID 2023 — expected Q4 2025 (not yet published as of audit date)
- eGRID 2024 — expected Q4 2026 (not yet published as of audit date)

The CTO target ("eGRID 2024") therefore reflects the latest expected
publication available within the alpha-launch window, not what is
currently in the database. Alpha v0.1 ships with eGRID 2022.

## What is lost vs the target

eGRID 2022 -> 2024 historical drift on US grid CO2 intensities:

- **National rollup**: ~5-10% drop (852 lb/MWh in eGRID 2022 -> est.
  ~770-810 lb/MWh in eGRID 2024) as coal continues to retire and gas
  + renewables grow.
- **CAMX (California)**: ~10-15% drop (496 -> est. ~420-450 lb/MWh)
  as solar penetration deepens.
- **SRMW (SERC Midwest, coal-heavy)**: ~3-8% drop — slower decarb
  pace.
- **NEWE (NPCC New England)**: ~5-10% drop.
- **NWPP (Northwest, hydro-heavy)**: relatively stable; <3% drift as
  the underlying mix is already low-carbon.
- **NYUP (Upstate NY)**: relatively stable; <3% drift.
- **Sub-pollutants** (CH4, N2O): drift in the same direction but
  absolute CO2e contribution is <2% of the total.

## Materiality estimate (representative factors)

| Factor family | eGRID 2022 -> 2024 expected drift | Alpha materiality |
|---|---|---|
| US national average grid | 5-10% | MEDIUM |
| CAMX (CA) | 10-15% | HIGH (alpha customers in CA see notable difference) |
| SRMW (Midwest coal) | 3-8% | MEDIUM |
| NEWE (New England) | 5-10% | MEDIUM |
| NWPP (Northwest hydro) | <3% | LOW |
| NYUP (Upstate NY) | <3% | LOW |
| State-level rollups | 3-10% | MEDIUM (varies) |

This is the largest expected drift among the six alpha sources. The
2-year vintage gap on a decarbonising grid produces material
under-counting (using eGRID 2022 today over-states emissions by
roughly 5-10% on the national rollup).

## Mitigation

1. **API response provenance**: `provenance.source_year: 2022` is
   stamped on every eGRID-rooted factor; customers can filter.
2. **Vintage stamping**: alpha customers using US Scope 2 electricity
   factors see explicit `2022` vintage in the response — they can
   choose to apply a forward-correction at their own discretion (the
   alpha SDK does not auto-correct).
3. **Customer-facing language**: when discussing eGRID with design
   partners, the methodology team should explicitly call out the
   2-year vintage and the grid-decarb direction.
4. **Documentation callout**: this file appears in the alpha
   onboarding pack.
5. **Update path**: the moment EPA publishes eGRID 2023 OR 2024, the
   ETL pipeline can ingest within 1 day (parser is unchanged; column
   shape has been stable since eGRID 2018). The snapshot test will
   verify column-shape stability automatically.

## Re-audit due date

**Target:** 2026-Q2 (alpha launch), and again Q4 2026 if eGRID 2024
publishes mid-year. The re-audit MUST confirm:
- EPA has or has not published eGRID 2023 or 2024.
- If yes (any newer than 2022): parse, verify, bump `source_version`,
  flip `alpha_v0_1_status` to `locked`, archive this file. The
  preferred bump is to the latest published edition — even if eGRID
  2023 (not 2024) is what's available, that's still an improvement
  over eGRID 2022.
- If no: re-date this exception. Do not block alpha launch.

## Sign-off

methodology lead: human:methodology-lead@greenlang.io  approved_at: <pending — sign before alpha launch>
