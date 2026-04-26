# Phase 0 Audit Archive — 2026-04-26

These files were temporary audit artifacts from the GreenLang Factors
pre-Phase-0 audit. The CTO Phase 0 plan ("Immediate Audit Baseline And
Repo Cleanup") required them to be reviewed and either deleted or
archived once they were no longer needed.

| File                          | Origin                                                                            | Replaced by                                                       |
| ----------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `_tmp_factors_doc.txt`        | text extraction of `Final_GreenLang_Factors.docx`                                 | `docs/factors/roadmap/GreenLang_Factors_FY27_FY31_Source_of_Truth.docx` (frozen w/ SHA256) |
| `_tmp_factors_doc_full.txt`   | full text extraction of `Final_GreenLang_Factors.docx`                            | same as above                                                     |
| `_tmp_roadmap_doc.txt`        | text extraction of `GreenLang_Climate_OS_..._CTO_Final.docx`                      | `docs/factors/roadmap/GreenLang_Climate_OS_Roadmap_FY27_FY31_CTO_Final.docx` (frozen w/ SHA256) |
| `_tmp_probe_resolve.py`       | one-off probe script for `/v1/resolve` (which is gated off in `alpha-v0.1`)       | n/a — the live test path is `tests/factors/v0_1_alpha/`            |

Kept here rather than deleted so the audit trail is reconstructible.
The frozen source-of-truth manifest is at
`docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md` and ADR-001 at
`docs/factors/adr/ADR-001-greenlang-factors-source-of-truth.md`.
