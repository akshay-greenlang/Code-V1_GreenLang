# Methodology Exception — US EPA GHG Emission Factors Hub

**source_id:** `epa_hub`
**Current `source_version`:** `2024.1`
**CTO target vintage:** `2025`
**Status:** `exception_accepted`
**Created:** 2026-04-25
**Re-audit due:** alpha launch (target 2026-Q2)

## Why the CTO target vintage is not yet shipped

The US EPA publishes the *GHG Emission Factors Hub* as a single
PDF/Excel companion to the Climate Leadership program. The 2025
edition release schedule is Q1-Q2 2026 — the precise publication
date is not announced ahead of time (recent editions have shipped
within a 30-day window of `epa.gov/climateleadership` posting). At
the audit date (2026-04-25), we cannot confirm whether the 2025 edition
has yet been posted; the parser is wired and column shape is stable,
so a refresh-once-published is a 1-day operation.

Alpha v0.1 ships with the 2024 edition (the most recent confirmed
published version at PR-cut time). Customers using EPA-rooted
factors get an explicit `provenance.source_year: 2024` stamp on every
factor.

## What is lost vs the target

The EPA Hub 2024 -> 2025 release cycle is expected to revise:

- **Stationary combustion** (Tables 1-2): natural gas, distillate fuel
  oil No. 2, propane, kerosene, coal types — expected drift <0.5% on
  CO2 factors (heating-value-driven; very stable). CH4 and N2O may
  shift up to 5-10% as EPA refines combustion-by-equipment-class
  factors but the absolute contribution to CO2e is <0.1%.
- **Mobile combustion** (Table 4): on-road gasoline/diesel — expected
  drift <1%.
- **Electricity (Scope 2)**: the EPA Hub electricity table is a
  rollup of eGRID — the alpha *defers* Scope 2 electricity to the
  `egrid` source (see its own exception). EPA Hub electricity factors
  are NOT used in alpha resolution.
- **Steam and heat (Scope 2)**: typical drift <2%.
- **Scope 3 upstream (Tables 9-13)**: spend-based emission factors;
  drift up to 5% as EPA refreshes the SUSEEIO commodity-by-industry
  matrix. Alpha use of Scope 3 spend factors is opt-in.

## Materiality estimate (representative factors)

| Factor family | EPA 2024 -> 2025 expected drift | Alpha materiality |
|---|---|---|
| Natural gas (stationary) | <0.5% | LOW |
| Distillate fuel oil No. 2 | <0.5% | LOW |
| Motor gasoline (mobile) | <1% | LOW |
| Propane / Kerosene | <0.5% | LOW |
| Steam / district heat | <2% | LOW |
| Scope 3 upstream (spend) | up to 5% | MEDIUM (opt-in only) |

## Mitigation

1. **API response provenance**: `provenance.source_year: 2024` is
   stamped on every EPA-rooted factor; customers filter on this.
2. **Scope 2 electricity**: alpha resolver routes US Scope 2
   electricity to `egrid` (its own source), not EPA Hub electricity
   table. Drift on the alpha-default Scope 2 path is bounded by
   eGRID's own re-audit (see `egrid_v2022.1.md`).
3. **Documentation callout**: this file appears in the alpha
   onboarding pack.

## Re-audit due date

**Target:** 2026-Q2 (alpha launch). The re-audit MUST confirm:
- EPA has or has not published the 2025 Hub.
- If yes: parse with the current `parse_epa_ghg_hub`, verify column
  shape unchanged via the snapshot test, bump `source_version` to
  `2025.1`, flip `alpha_v0_1_status` to `locked`, archive this file.
- If no: re-date this exception and push the next re-audit by 30
  days. Do not block alpha launch.

## Sign-off

CTO-delegated exception acceptance:
`human:cto-delegated@greenlang.io`, approved at `2026-04-26T00:00:00+00:00`.

Permanent methodology-lead countersign remains part of the alpha
release review, but this exception no longer blocks Phase 1 source
rights closure.
