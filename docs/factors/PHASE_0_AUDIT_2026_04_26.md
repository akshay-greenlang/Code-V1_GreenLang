# Pre-Phase-0 Audit Cleanup — 2026-04-26

This document records the actions taken during the pre-Phase-0 audit
("Immediate Audit Baseline And Repo Cleanup") for GreenLang Factors,
per the CTO's plan dated 2026-04-25.

This is **pre-Phase-0** — it creates the control surface (frozen
source-of-truth, epics, release-profile discipline, URN spec
compliance) that lets Phase 0 work start. It is NOT itself building
new product features.

## KPIs vs targets

| KPI                                                             | Target | Actual    | Status |
| --------------------------------------------------------------- | ------ | --------- | ------ |
| Product source-of-truth frozen                                  | 1      | 2 docs    | DONE   |
| Release milestones converted to epics                           | 8/8    | 8/8       | DONE   |
| Acceptance criteria captured                                    | 100%   | per epic  | DONE   |
| Worktree triaged                                                | 0 unclassified files | 0 | DONE   |
| Temporary audit files removed or archived                       | 100%   | 4/4 archived | DONE |
| v0.1 alpha test blocker fixed                                   | green  | 1275 passed, 7 skipped, 0 failed | DONE |
| Invalid URNs in alpha catalog                                   | 0      | 0         | DONE (691 fixed) |
| Release profiles documented                                     | 100%   | 100%      | DONE (already shipped Wave 5) |
| Production default profile                                      | `alpha-v0.1` | `alpha-v0.1` | DONE (already shipped Wave 5) |
| PR hygiene                                                      | v0.1 isolated | n/a — last commit (`b251b1d3`) already bundled v0.1 work; future PRs MUST stay isolated per ADR-001 |

## What changed

### 1. Source-of-truth documents frozen

| File | SHA-256 | Frozen at |
| --- | --- | --- |
| `docs/factors/roadmap/GreenLang_Factors_FY27_FY31_Source_of_Truth.docx` | `9060794dd8d82388e9f676c408ab0a4a8f68b4d7866aba02a740a14312ddcefd` | 2026-04-26 |
| `docs/factors/roadmap/GreenLang_Climate_OS_Roadmap_FY27_FY31_CTO_Final.docx` | `23952d2d8bfdf9bedadd8271d185b890f37779b198f83213a4be3031075a4adc` | 2026-04-26 |

Manifest: `docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md`
ADR: `docs/factors/adr/ADR-001-greenlang-factors-source-of-truth.md`

### 2. 8 release milestone epics created

`docs/factors/epics/`:

* `epic-v0.1-alpha.md` (FY27 Q1, profile `alpha-v0.1`)
* `epic-v0.5-closed-beta.md` (FY27 Q2, profile `beta-v0.5`)
* `epic-v0.9-public-beta.md` (FY27 Q3, profile `rc-v0.9`)
* `epic-v1.0-ga.md` (FY27 Q4, profile `ga-v1.0`)
* `epic-v1.5.md` (FY28 Q3)
* `epic-v2.0.md` (FY29 Q2)
* `epic-v2.5.md` (FY30 Q2)
* `epic-v3.0.md` (FY31 Q1)

Each has scope, out-of-scope, deliverables, acceptance criteria,
source coverage, API/SDK expectations, security gates, owner, target
quarter, dependencies, risks.

Owner assignments recorded 2026-04-26:

| Epic | Owner |
| ---- | ----- |
| v0.1 Alpha | Platform/Data Lead (interim accountable: CTO until named lead is assigned) |
| v0.5 Closed Beta | Backend/API Lead |
| v0.9 Public Beta | Developer Experience Lead |
| v1.0 GA | Engineering Manager, Factors |
| v1.5 | ML and Community Lead |
| v2.0 | Enterprise Platform Lead |
| v2.5 | Streaming/SRE Lead |
| v3.0 | CTO / Factors GM |

### 3. Worktree triaged

| File / dir                              | Bucket                       | Action              |
| --------------------------------------- | ---------------------------- | ------------------- |
| All `_tmp_factors_doc*.txt`             | temp audit                   | archived under `_archive/12_phase0_audit_2026_04_26/` |
| `_tmp_roadmap_doc.txt`                  | temp audit                   | archived            |
| `_tmp_probe_resolve.py`                 | one-off probe (resolve gated off in alpha) | archived |
| `_archive/11_ralphy_and_extras/ralphy-agent/` | unrelated archive (pre-existing untracked) | retain local-only for 30 days; exclude from git via `.gitignore`; delete after 2026-05-26 unless CTO requests a sanitized archive |
| All other repo state                    | clean (per `git status --short`) | no action needed |

### 4. v0.1 alpha test blocker — URN spec compliance

**Root cause:** the schema regex
`config/schemas/factor_record_v0_1.schema.json` permitted uppercase
namespace + id segments (`[A-Za-z0-9._-]+`) while the canonical URN
parser at `greenlang.factors.ontology.urn` requires lowercase. The
first batch of seed JSONs (`greenlang/factors/data/catalog_seed_v0_1/`)
was generated through a normalizer
(`greenlang.factors.etl.alpha_v0_1_normalizer.coerce_factor_id_to_urn`)
that did not lowercase before composing the URN. The provenance gate
test passed because it used the schema regex, not the canonical
parser.

**Fix (5 changes):**

1. `scripts/factors_alpha_v0_1_lowercase_urns.py` — one-shot
   in-place lowercaser; reports per-source counts. **Re-run is a
   no-op** (idempotent).
2. `greenlang/factors/etl/alpha_v0_1_normalizer.py` —
   `coerce_factor_id_to_urn` now lowercases namespace + id segments
   at generation time. Internal `_URN_FACTOR_RE` tightened.
3. `config/schemas/factor_record_v0_1.schema.json` — `urn` pattern
   tightened to `^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$`.
4. `tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py` —
   new regression test with three guards:
   * sanity: seeds present;
   * **parametric**: every factor URN passes
     `greenlang.factors.ontology.urn.parse`;
   * explicit guard: no factor URN's namespace segment contains an
     uppercase ASCII letter.
5. Catalog seeds: 691 URNs rewritten across 6 sources (CBAM 60,
   DESNZ 195, eGRID 79, EPA 84, India CEA 38, IPCC 235).

**Known follow-up (not in pre-Phase-0 scope):**

* `factor_pack_urn` migration is complete. ADR-002 requires public
  pack URNs to use final `v<int>` version segments. The 691 alpha seed
  pack URNs now end in `v1`; upstream dotted versions (`2024.1`,
  `2022.1`, `20.0`, `2019.1`) remain in
  `extraction.source_version`.

### 5. Release-profile discipline — verified, no changes needed

`greenlang/factors/release_profile.py` was already shipped in Wave 5
with:

* All 5 required profiles (`dev`, `alpha-v0.1`, `beta-v0.5`,
  `rc-v0.9`, `ga-v1.0`).
* Production default `alpha-v0.1` (env-driven; verified in
  `test_release_profile.py::test_current_profile_default_is_alpha_in_production`).
* `ALPHA_ALLOWED_PATHS` frozenset locks the 5 alpha-allowed v1
  paths.
* `filter_app_routes` enforces alpha minimality at runtime.
* `tests/factors/v0_1_alpha/test_release_profile.py` (60+
  assertions) covers ordering, per-feature gates, route filtering,
  fallback behavior.
* `tests/factors/v0_1_alpha/test_alpha_api_contract.py` asserts the
  OpenAPI spec exposes only the 5 alpha endpoints in alpha profile.

No changes were needed in pre-Phase-0.

## Verification

Run from repo root:

```bash
# 1. Verify URN seeds compile cleanly through canonical parser
python -m pytest tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py -q

# 2. Verify full v0.1 alpha suite is green
python -m pytest tests/factors/v0_1_alpha -q

# 3. Verify lowercase-URN script is idempotent (should report 0 changed)
python scripts/factors_alpha_v0_1_lowercase_urns.py

# 4. Verify source-of-truth document hashes unchanged
python -c "import hashlib; \
print(hashlib.sha256(open('docs/factors/roadmap/GreenLang_Factors_FY27_FY31_Source_of_Truth.docx','rb').read()).hexdigest())"
# expect: 9060794dd8d82388e9f676c408ab0a4a8f68b4d7866aba02a740a14312ddcefd
```

## CTO action closure

1. **Countersign ADR-001** - CLOSED 2026-04-26. ADR-001 status is
   `Accepted`; the source-of-truth manifest is binding.
2. **Assign owners** - CLOSED 2026-04-26. Owner roles are recorded in
   all 8 epic files and in `docs/factors/epics/README.md`.
3. **Pack URN version-segment policy** - CLOSED 2026-04-26. ADR-002
   chooses canonical public pack URNs ending in `v<int>` and stores
   upstream dotted versions in metadata.
4. **Retention for `_archive/11_ralphy_and_extras/ralphy-agent/`** -
   CLOSED 2026-04-26. Retain local-only for 30 days, exclude from git,
   delete after 2026-05-26 unless CTO requests a sanitized archive.
5. **Branch policy going forward** - last commit (`b251b1d3
   "update"`) bundled the entire v0.1 alpha stabilization plus the
   Wave 5 catalog expansion. Future PRs MUST stay isolated per
   ADR-001 conflict-resolution rule.

## Phase 0 entry criteria — status

Per the CTO's "Phase 0 Exit Criteria" (which becomes Phase 0 ENTRY
criteria for subsequent waves):

| Criterion                                                       | Status                |
| --------------------------------------------------------------- | --------------------- |
| source-of-truth document is frozen and approved                 | DONE - frozen and approved via ADR-001 countersign |
| all 8 roadmap epics exist                                       | DONE                  |
| worktree has no unclassified changes                            | DONE                  |
| v0.1 work is isolated on its own branch/PR                      | n/a (already merged); future PRs gated by ADR-001 |
| temporary audit files are removed or archived                   | DONE                  |
| v0.1 alpha tests are fully green                                | DONE — 1275 passed, 7 skipped, 0 failed (`pytest tests/factors/v0_1_alpha -q`, 2026-04-26 run) |
| invalid URNs are eliminated                                     | DONE (691 fixed)      |
| production defaults to alpha-v0.1                               | DONE (already shipped) |
| release-profile tests prove future features are hidden          | DONE (already shipped) |
