# GreenLang Audit Rebaseline (Mar 27, 2026)

This file revalidates the March 26 report against the current repository state.

## Overall Verdict

- `Backend`: Mostly complete, with specific gaps in CSRD API execution wiring.
- `Frontend`: Functional but fragmented across multiple stacks.
- `Platform`: Strong v1 framework, but signed-pack policy evidence is inconsistent in-repo.

## Claim Matrix

| Area | Claim | Status | Evidence |
|---|---|---|---|
| CBAM web | Export gate unified to `PipelineResult.can_export` | Done | `cbam-pack-mvp/src/cbam_pack/web/app.py` |
| CBAM CLI | Legacy drift resolved via proxy/re-export | Done | `greenlang/cli/cmd_run.py`, `greenlang/cli/__init__.py` |
| CBAM CI | Web extras installed in CI | Done | `.github/workflows/cbam-mvp-ci.yml` |
| CBAM XML | XSD fallback exists (embedded schema) | Done | `cbam-pack-mvp/src/cbam_pack/exporters/xml_generator.py` |
| CBAM tests | Excel bit-for-bit determinism test missing | Not done (claim stale) | `cbam-pack-mvp/tests/test_excel_output_determinism.py` exists |
| CBAM CI quality | Coverage/lint/type gates missing | Not done (claim stale) | `.github/workflows/cbam-mvp-ci.yml` has `--cov`, `ruff`, `mypy` |
| CBAM packaging | Non-standard monorepo path injection remains | Partial | `greenlang/cli/main.py` |
| CSRD pipeline | 6-agent backend pipeline exists | Done | `applications/GL-CSRD-APP/CSRD-Reporting-Platform/csrd_pipeline.py` |
| CSRD API | Real async execution wired from `/pipeline/run` | Partial | `applications/GL-CSRD-APP/CSRD-Reporting-Platform/api/server.py` (queue present, execution/commented placeholders) |
| CSRD validation API | Intake-based validation endpoint implemented | Partial | `applications/GL-CSRD-APP/CSRD-Reporting-Platform/api/server.py` placeholder response |
| VCCI backend | Backend/API and service modules exist | Done | `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/main.py` + `services/` |
| V1 core | Runtime/conformance/contracts/backends/profiles present | Done | `greenlang/v1/*.py`, `greenlang/cli/cmd_v1.py` |
| V1 contracts | `gl.yaml`, `pack.yaml`, `sbom.spdx.json` across 3 apps | Done | `applications/*/v1/` |
| V1 signing | `pack.sig` present for all 3 apps in-repo | Partial | `greenlang/v1/conformance.py` enforces signatures; `pack.sig` files not committed |
| V1 CI | 3-job lane (`smoke`, `gates`, `matrix`) exists | Done | `.github/workflows/greenlang-v1-platform-ci.yml` |
| Frontend stack | CBAM shell is vanilla HTML/CSS/JS | Done | `cbam-pack-mvp/src/cbam_pack/web/index.html`, `ui_shared.js` |
| Frontend stack | VCCI standalone uses React + MUI + Redux + Recharts | Done | `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/frontend/package.json` |
| Frontend stack | Agent Factory uses React + Tailwind + Radix + Framer Motion | Done | `applications/GL-Agent-Factory/frontend/package.json` |
| Frontend architecture | Three design systems are fragmented | Done | Shell HTML/CSS vs MUI React vs Tailwind/Radix app |
| Root hygiene | `tmp_*` root directories currently present | Not done (claim stale) | No root matches for `tmp_*` in current workspace scan |

## Priority Fixes Applied Next

1. Wire CSRD `/api/v1/pipeline/run` to execute real pipeline jobs.
2. Wire CSRD `/api/v1/validate` to use IntakeAgent and real file validation.
3. Make v1 signed-pack checks reproducible in CI by generating signatures before checks.
4. Normalize CBAM dependency loading to avoid default `sys.path` fallback behavior.
