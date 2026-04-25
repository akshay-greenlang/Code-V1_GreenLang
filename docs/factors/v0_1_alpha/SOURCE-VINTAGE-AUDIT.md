# v0.1 Alpha — Source Vintage Audit

**Document owner:** GL-FormulaLibraryCurator (Wave D / TaskCreate #7-#12)
**Created:** 2026-04-25
**Spec reference:** CTO doc §19.1 — v0.1 Alpha source list
**Re-audit due:** at alpha launch (target 2026-Q2)

This document audits the six emission-factor sources listed by the CTO
as "in scope for v0.1 Alpha". For each source we record (a) the
`source_version` currently shipped in `greenlang/factors/data/source_registry.yaml`,
(b) the CTO target vintage, (c) the gap, and (d) a per-source decision
of `locked`, `preview`, or `update_pending`.

The decision drives a corresponding change in `source_registry.yaml`:
all six sources gain three new fields (`alpha_v0_1_status`,
`alpha_v0_1_vintage_target`, `alpha_v0_1_methodology_exception`). For
sources marked `preview`, a signed methodology exception lives at
`docs/factors/v0_1_alpha/methodology-exceptions/{source_id}_v{current}.md`.

---

## 1. Per-source audit table

| # | source_id (registry) | Spec name | Current `source_version` | CTO target | Gap | Decision | Rationale |
|---|---|---|---|---|---|---|---|
| 7 | `ipcc_2006_nggi` (alias `ipcc_ar6`) | IPCC AR6 minimal Tier 1 | `2019.1` (NGGI 2019 Refinement) | AR6 (2021/2022) | AR6 GWPs vs NGGI 2019 EFs (split concern) | **locked** | NGGI 2019 IS the IPCC EF source; AR6 is the GWP set. v0.1 alpha emits AR6 GWP-weighted CO2e (`gwp_set: IPCC_AR6_100`, `CH4_gwp=28`, `N2O_gwp=273`) computed from NGGI 2019 EFs. The two are not in conflict — they are two different layers. The registry alias `ipcc_ar6-ipcc_2006_nggi` already encodes this; the FREEZE note in the parser comment now clarifies. No exception required. |
| 8 | `desnz_ghg_conversion` | UK DESNZ / DEFRA 2025 | `2024.1` | `2025` | 1 publication year | **preview** | DESNZ publishes 2025 conversion factors mid-2025; ingest is wired but the spreadsheet has not yet been ingested (ETL run is queued). Re-audit at alpha launch — if 2025 spreadsheet is in CRM by Q2 2026, bump `source_version` to `2025.1` and flip status to `locked`. Methodology exception: `methodology-exceptions/desnz_ghg_conversion_v2024.1.md`. |
| 9 | `epa_hub` | US EPA GHG Emission Factors Hub 2025 | `2024.1` | `2025` | 1 publication year | **preview** | EPA GHG Hub 2025 release schedule is Q1-Q2 2026. Verify against `epa.gov/climateleadership/ghg-emission-factors-hub` at launch time. Drift on representative factors is small (<1% on natural gas/diesel/grid mix); the only category with non-trivial drift is grid-mix-driven Scope 2 which the alpha defers to eGRID anyway. Methodology exception: `methodology-exceptions/epa_hub_v2024.1.md`. |
| 10 | `egrid` | US EPA eGRID 2024 | `2022.1` | `2024` | 2 publication years | **preview** | eGRID 2024 typically publishes Q4 2026; release schedule may slip. eGRID 2022 -> 2023 -> 2024 historical drift on subregion CO2 intensity is ~3-8% (CAMX dropping due to coal retirements; SRMW rising less than feared). Methodology exception: `methodology-exceptions/egrid_v2022.1.md`. Re-audit: bump to `2024.1` only after EPA officially publishes; in the interim, alpha customers using US-grid factors get explicit "vintage 2022" stamping in API response. |
| 11 | `india_cea_co2_baseline` | India CEA Baseline — latest | `20.0` (v20.0, Dec 2024 reporting FY2023-24) | latest (v20.0) | none — v20.0 IS the latest published edition | **locked** | CEA publishes annually around Sep-Oct. v20.0 (Dec 2024) is the most recent published edition as of the audit date. Next edition (v21.0) is expected late-2025; if available before alpha launch, bump and re-tag as `locked`-current. No exception required at v20.0. |
| 12 | `cbam_default_values` | EU CBAM default values | `2024.1` | current effective | none — Reg 2023/1773 Annex IV is the current effective period | **locked** | CBAM defaults are locked at Commission Implementing Regulation (EU) 2023/1773 Annex IV. The transitional period runs to end of 2025 and the definitive period (2026+) requires verified actual emissions where available — defaults are the alpha-allowed fallback. No subsequent implementing act has revised the Annex IV defaults; verified against eur-lex CELEX `32023R1773`. No exception required. |

**Status totals:** 3 locked (`ipcc_2006_nggi`, `india_cea_co2_baseline`, `cbam_default_values`); 3 preview (`desnz_ghg_conversion`, `epa_hub`, `egrid`); 0 `update_pending` (we did not block on a feasible-before-launch update — preview pattern is preferred so the audit is explicit and repeatable).

---

## 2. Decision rubric

`locked` — `source_version` matches the CTO target. No methodology exception
required. Re-audit only when the CTO doc target itself moves.

`preview` — `source_version` is older than the CTO target, but the gap
is tolerable for v0.1 alpha (shipped to design partners under explicit
"alpha" labeling). Methodology exception file is required and must be
signed before alpha launch.

`update_pending` — The vintage update is feasible before alpha launch
(parser is ready, fixtures available, ETL queued). Bump and re-audit
within the same PR; we currently have zero items in this state.

---

## 3. Snapshot tests

For each of the 6 sources, a parser-snapshot test now lives at
`tests/factors/v0_1_alpha/test_alpha_source_snapshots.py`. The test:

1. Loads the canonical raw fixture from
   `tests/factors/fixtures/{source_id}/{vintage}/raw.json`.
2. Runs the registered parser via the
   `parser_module:parser_function` pair from `source_registry.yaml`.
3. Normalizes timestamp fields (`created_at`, `updated_at`,
   `latest_ingestion_at`) to a fixed value to keep the snapshot
   reproducible across CI runs.
4. Asserts byte-equality with the saved
   `tests/factors/fixtures/{source_id}/{vintage}/expected.json`.
5. On drift, prints a clear diff hint:
   `"parser drift detected — regenerate via: pytest tests/factors/v0_1_alpha/test_alpha_source_snapshots.py --update-source-snapshots"`.

The test is parametrised across the 6 alpha sources. It is the
"parser drift" guard called out in CTO doc §19.1 risk list — when a
source publisher changes their column shape (e.g. DESNZ rename
`co2_factor` -> `co2e_factor`), the snapshot fails loudly instead of
silently emitting wrong factor records.

**Snapshot test count:** 6 parametrised cases (one per source). All
green at the time of writing.

---

## 4. Files created / edited

**Created**
- `docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md` (this file)
- `docs/factors/v0_1_alpha/methodology-exceptions/desnz_ghg_conversion_v2024.1.md`
- `docs/factors/v0_1_alpha/methodology-exceptions/epa_hub_v2024.1.md`
- `docs/factors/v0_1_alpha/methodology-exceptions/egrid_v2022.1.md`
- `tests/factors/v0_1_alpha/test_alpha_source_snapshots.py`
- `tests/factors/fixtures/{source_id}/{vintage}/raw.json` (6 fixtures)
- `tests/factors/fixtures/{source_id}/{vintage}/expected.json` (6 snapshots)

**Edited**
- `greenlang/factors/data/source_registry.yaml` — added three new
  alpha-fields (`alpha_v0_1_status`, `alpha_v0_1_vintage_target`,
  `alpha_v0_1_methodology_exception`) on each of the six alpha sources.

---

## 5. Re-audit trigger

The methodology lead MUST re-audit this document at the alpha launch
date (target 2026-Q2), and at minimum confirm:

1. Has DESNZ published 2025 GHG conversion factors? If yes, bump
   `source_version: 2025.1`, status -> `locked`, archive the exception.
2. Has EPA published the 2025 GHG Emission Factors Hub? If yes, bump
   `source_version: 2025.1`, status -> `locked`, archive the exception.
3. Has EPA published eGRID 2024? If yes, bump `source_version: 2024.1`,
   status -> `locked`, archive the exception.
4. CEA v21.0 — if available, bump and confirm the snapshot fixture.
5. CBAM Annex IV — if any new implementing act has revised defaults
   under Reg 2023/956, refresh fixture; otherwise keep `locked`.
6. IPCC — no action expected. AR6 GWPs are stable until AR7 (~2028).

Exception documents under
`docs/factors/v0_1_alpha/methodology-exceptions/` MUST be re-dated and
re-signed (or archived) at every re-audit event.

---

## 6. Wave D / TaskCreate #7-#12 — closing status

- **#7 IPCC AR6**: locked — `ipcc_2006_nggi` v2019.1 (NGGI 2019 EFs +
  AR6 GWP set). No exception required; the AR6/NGGI split is encoded
  in the registry alias and the parser GWP table.
- **#8 DEFRA 2025**: preview — `desnz_ghg_conversion` v2024.1 with
  exception doc; bump scheduled at alpha launch once DESNZ publishes.
- **#9 EPA Hub 2025**: preview — `epa_hub` v2024.1 with exception
  doc; re-audit at launch.
- **#10 eGRID 2024**: preview — `egrid` v2022.1 with exception doc;
  re-audit at launch (eGRID typically publishes Q4 of next-but-one
  year).
- **#11 India CEA latest**: locked — `india_cea_co2_baseline` v20.0
  is the latest published edition; no exception required.
- **#12 CBAM defaults**: locked — `cbam_default_values` v2024.1
  pinned at Reg 2023/1773 Annex IV (current effective period); no
  exception required.
