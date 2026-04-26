# ADR-002: Pack URN Version-Segment Policy

| Field      | Value |
| ---------- | ----- |
| Status     | Accepted |
| Date       | 2026-04-26 |
| Owner      | CTO (Akshay) |
| Author     | Engineering (under CTO direction) |
| Related    | `ADR-001-greenlang-factors-source-of-truth.md`, `docs/factors/epics/epic-v0.1-alpha.md` |

## Context

The v0.1 alpha seed catalog currently contains `factor_pack_urn`
values with upstream-style version segments such as `2024.1` and
`20.0`. Those values are useful source-version labels, but they do
not match the canonical GreenLang URN discipline used elsewhere,
where public version segments are stable product-release identifiers
of the form `v<int>`.

Leaving upstream dotted versions inside public pack URNs would make
pack IDs harder to validate and would couple GreenLang pack identity
to upstream publisher versioning conventions.

## Decision

Public GreenLang pack URNs MUST use a canonical final version segment
of the form `v<int>`.

Examples:

* `urn:gl:pack:defra-2025:v1`
* `urn:gl:pack:epa-ghg-hub-2025:v1`
* `urn:gl:pack:egrid-2024:v1`
* `urn:gl:pack:india-cea:v1`
* `urn:gl:pack:cbam-defaults-2024:v1`

Upstream source versions such as `2024.1`, `20.0`, `v22.0`, workbook
revisions, publication dates, or errata labels MUST be preserved in
metadata fields, not in the public pack URN final segment. Use fields
such as `source_version`, `upstream_version`, `source_publication`,
`release_manifest.source_version`, and source artifact checksums.

## Compatibility

The v0.1 alpha catalog may continue to read legacy dotted
`factor_pack_urn` values as aliases during stabilization, but publish
and API responses should migrate to canonical `v<int>` pack URNs
before v0.5 Closed Beta.

Any dotted upstream version found in a public pack URN after this ADR
is accepted is a migration bug, not a product decision.

## Consequences

* Pack URN parsing can be tightened without losing upstream source
  version traceability.
* Source-version freshness remains auditable through metadata and
  release manifests.
* v0.1 has a clear follow-up implementation task: migrate alpha seed
  `factor_pack_urn` values to canonical pack URNs and preserve the
  current dotted values in metadata.
