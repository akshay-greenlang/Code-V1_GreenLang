# Legal Note - GHG Protocol Method References

| Field | Value |
| --- | --- |
| source_id | `ghgp_method_refs` |
| source_urn | `urn:gl:source:ghgp-method-refs` |
| licence_class | `method_only` |
| redistribution_class | `metadata_only` |
| entitlement_model | `public_no_entitlement` |
| Phase 1 scope | v0.1 method-reference licensing baseline |
| Approval status | CTO-delegated approval for citation and metadata only |

## Decision

GreenLang may cite GHG Protocol method references in v0.1 alpha
metadata, methodology descriptions, and documentation. GreenLang must
not redistribute copyrighted standard text, tables, paywalled licensed
content, or extracted method values from the source under this approval.

This approval is intentionally narrower than a factor-source approval:
the row remains `method_only` and `metadata_only`. It supports naming
the method basis for alpha factors and partner documentation, not
serving GHG Protocol content as an emission-factor pack.

## Conditions

- API and SDK responses may include `citation_text`, source display
  name, source URN, and methodology linkage.
- Bulk factor values are not published from this source in v0.1.
- Long excerpts of GHG Protocol standard text are not stored in factor
  records or public docs.
- Any future paid/licensed content or verbatim method text requires a
  separate Compliance/Security Lead review before release.

## Evidence

- Public source landing page: `https://ghgprotocol.org/`
- Registry row: `greenlang/factors/data/source_registry.yaml`
- Human-readable matrix: `docs/factors/source-rights/SOURCE_RIGHTS_MATRIX.md`

## Sign-off

CTO-delegated engineering/legal baseline:
`human:cto-delegated@greenlang.io`, approved at `2026-04-26T00:00:00+00:00`.

Permanent Compliance/Security Lead countersign remains part of the
Phase 1 human sign-off block.
