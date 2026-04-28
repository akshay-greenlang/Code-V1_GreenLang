# GreenLang Factors — Source Registry CHANGELOG

> **Authority**: CTO Phase 3 brief 2026-04-28, Block 7 box 2.
> **Purpose**: every bump of `parser_version` for any source in
> `greenlang/factors/data/source_registry.yaml` MUST land here as a section
> entry. The Phase 3 CI gate `scripts/ci/check_source_registry_version.py`
> blocks PRs that bump a parser_version without a matching CHANGELOG entry.
>
> **Format**: each entry is a top-level Markdown section with the header
> pattern `## <source_id> <new_parser_version>` (case-insensitive). The
> regex used by the gate is:
>
>     ^## .*<source_id>.* <new_version>
>
> Add a short rationale + a pointer to the parser commit + the path to the
> regenerated snapshot. Do not delete past entries — append-only.

---

## Template (copy and rename when bumping a parser)

## <source_id> <new_parser_version> — YYYY-MM-DD

- Reason: (e.g. upstream column rename, new product code, fix for X)
- Parser file: `greenlang/factors/ingestion/parsers/<source_id>.py`
- Snapshot regenerated: `tests/factors/v0_1_alpha/phase3/parser_snapshots/<source_id>_<vintage>__<parser_version>.golden.json`
- Tested-by: `tests/factors/v0_1_alpha/phase3/test_<family>_e2e.py`
- Reviewed-by: methodology-lead@greenlang.io

---

## Bootstrap entry — 2026-04-28

This file was created by Phase 3 Wave 3.0. Existing parser_version values in
`source_registry.yaml` (all `0.1.0` at file-creation time) are considered the
baseline and do not require retro-active CHANGELOG entries. The gate begins
enforcing on the FIRST parser_version bump after this commit.
