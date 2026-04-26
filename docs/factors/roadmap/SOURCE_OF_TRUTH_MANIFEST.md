# GreenLang Factors — Source-of-Truth Manifest (Frozen)

**Status:** FROZEN
**Frozen on:** 2026-04-26
**Owner:** CTO (Akshay)
**Approval:** Approved via ADR-001 countersign on 2026-04-26

This manifest records the document baseline that controls the GreenLang
Factors product roadmap and release scope through FY31. Conflicts
between repo state and these documents are resolved in favor of the
documents until leadership explicitly amends them via an updated ADR.

## Frozen Documents

| Field            | Value                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| Filename         | `GreenLang_Factors_FY27_FY31_Source_of_Truth.docx`                                          |
| Original name    | `Final_GreenLang_Factors.docx` (repo root)                                                  |
| Version / Date   | Final, dated 2026-04-25 (saved by CTO 09:20 IST)                                            |
| Owner            | CTO (Akshay)                                                                                |
| Approval status  | Approved via ADR-001 countersign on 2026-04-26                                              |
| SHA-256          | `9060794dd8d82388e9f676c408ab0a4a8f68b4d7866aba02a740a14312ddcefd`                          |
| Source location  | Repo root, copied 2026-04-26 to `docs/factors/roadmap/`                                     |
| Linked decision  | `docs/factors/adr/ADR-001-greenlang-factors-source-of-truth.md`                             |
| Scope            | GreenLang Factors product contract (v0.1 Alpha through v3.0, FY27 Q1 through FY31 Q1)       |

| Field            | Value                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| Filename         | `GreenLang_Climate_OS_Roadmap_FY27_FY31_CTO_Final.docx`                                     |
| Original name    | `GreenLang_Climate_OS_Final_Product_Definition_Roadmap_FY27_FY31_CTO_Final.docx`            |
| Version / Date   | CTO Final, dated 2026-04-24                                                                 |
| Owner            | CTO (Akshay)                                                                                |
| Approval status  | Approved via ADR-001 countersign on 2026-04-26                                              |
| SHA-256          | `23952d2d8bfdf9bedadd8271d185b890f37779b198f83213a4be3031075a4adc`                          |
| Source location  | Repo root, copied 2026-04-26 to `docs/factors/roadmap/`                                     |
| Linked decision  | `docs/factors/adr/ADR-001-greenlang-factors-source-of-truth.md`                             |
| Scope            | Cross-product Climate OS roadmap; Factors is the first product to ship                      |

## Conflict-Resolution Rule

1. If the repo implements a future feature earlier than the document
   allows, keep it but feature-gate it behind a release profile higher
   than the current target (see
   `greenlang/factors/release_profile.py`).
2. If the repo lacks a document-required feature, create a backlog
   item under the appropriate epic (`docs/factors/epics/`).
3. If the team wants to change scope, update the document AND submit a
   new ADR under `docs/factors/adr/` BEFORE landing code.

## Hash Verification

To verify the frozen documents have not drifted, run:

```bash
python -c "import hashlib; \
print(hashlib.sha256(open('docs/factors/roadmap/GreenLang_Factors_FY27_FY31_Source_of_Truth.docx','rb').read()).hexdigest())"
```

The output MUST equal the SHA-256 above. Any drift requires a new ADR.
