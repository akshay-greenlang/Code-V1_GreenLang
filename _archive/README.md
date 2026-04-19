# _archive/ — Archived Code & Assets

**Created:** 2026-04-19
**Purpose:** Make the repo legible in 5 minutes to engineers, design partners, and investors.
**Policy:** Nothing is deleted. Everything moved here is still in git history and browsable.

## Bucket Index

| # | Bucket | What's Inside |
|---|--------|---------------|
| 01 | `windows_path_garbage/` | Files with literal Windows paths as filenames (created by a path-escaping bug) |
| 02 | `root_scratch_output/` | Test output files, temp counts, EUDR test results, closure reports, customer artifacts, old audit reports |
| 03 | `audit_and_prd_history/` | Completed PRDs, audit reports, consolidation/merge/standardize/unify PRDs, status trackers |
| 04 | `shadow_development_tree/` | `GreenLang Development/` — ~648MB duplicate of applications/ + greenlang/ tree |
| 05 | `duplicate_agent_versions/` | Duplicate agent variants (_v2, _v3, _v4, _ai, _intelligent, _async, _sync) |
| 06 | `fy28_plus_premature_apps/` | GL-016 to GL-020, GL Agents, gl_agents, GL-Agent-Factory, GL-VCCI-Carbon-APP, etc. |
| 07 | `fy29_plus_premature_packs/` | 36 packs not in FY27 scope: energy-efficiency, net-zero, GHG 044-050, EU 006-011/018-020 |
| 08 | `legacy_v1_v2_runtime/` | greenlang/v1/, greenlang/v2/ legacy runtimes |
| 09 | `tmp_smoke_outputs/` | tmp_v1_smoke/, test_out_csrd/, phase1_evidence/, out/, logs/, test-reports/, reports/ |
| 10 | `misc_prd_mvp/` | 2026_PRD_MVP/, GreenLang_Agents_PRD_402/ |
| 11 | `ralphy_and_extras/` | ralphy-agent/, .ralphy/ |
| 12 | `large_binaries/` | tools/jq.exe, tools/syft.exe, tools/syft.zip |

## FY27 Active Code (what stays in main tree)

**Core:** `greenlang/` (factors, agents, schemas, infrastructure, etc.)
**Apps:** GL-CBAM-APP, GL-CSRD-APP, GL-GHG-APP, GL-SBTi-APP, GL-TCFD-APP, GL-ISO14064-APP, GL-SB253-APP, GL-CDP-APP, GL-Taxonomy-APP, GL-EUDR-APP
**Packs (13):** PACK-001 to PACK-005, PACK-012 to PACK-017, PACK-041 to PACK-043
**Infra:** deployment/, config/, scripts/, tests/, docs/, frontend/, examples/

## How to Restore

Any archived item can be restored with:
```bash
git mv _archive/<bucket>/<item> <original_location>
```
