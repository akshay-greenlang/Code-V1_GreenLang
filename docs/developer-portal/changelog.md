# Public changelog

Versioned release notes for the GreenLang Factors platform. This page is auto-published by the Factors release webhook on every `edition.cut` event.

For SDK-specific release notes see [`greenlang/factors/sdk/CHANGELOG.md`](../../greenlang/factors/sdk/CHANGELOG.md).

---

## v1.0.0 Certified edition (upcoming)

**Cut date:** planned 2026-05-01. **Edition ID:** `builtin-v1.0.0`.

### Highlights

- First **Certified** edition of the GreenLang Factors catalog.
- Canonical factor record v1.0 frozen (2026-04-22). See [`schema.md`](schema.md).
- **14 method profiles** across Corporate (4), Product (3), Freight (2), Land & Removals (1), Finance (1), EU Policy (3).
- **Seven-step resolution cascade** with `raise_no_safe_match` enforced on every Certified pack.
- **Signed receipts** — Ed25519 by default, HMAC-SHA256 for private deployments.
- **Bulk export** streams open-class factors as Parquet / JSON Lines / CSV with a signed manifest.
- **BYO-credentials carve-out** for ecoinvent, IEA, Electricity Maps, EC3, Green-e, GLEC, TCR — see [`licensing.md`](licensing.md).

### Sources bundled (Open class)

EPA GHG Factor Hub, EPA eGRID, UK DESNZ, UK DEFRA, India CEA, India CCTS, Australia DCCEEW NGA, Canada CER, US EPA SUSEEIO, IPCC 2006 Guidelines + 2019 Refinement, IPCC AR6 GWP tables, GreenLang curated built-in.

### Sources via BYO-credentials at v1 launch

ecoinvent, IEA, Electricity Maps, EC3 EPD library, Green-e Residual Mix, GLEC Framework, TCR General Reporting Protocol, PACT Pathfinder. Contract upgrades described in [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md).

### SDK versions

- Python SDK `greenlang-factors` 1.0.0 (rename from dev-release `greenlang-factors-sdk`).
- TypeScript SDK `@greenlang/factors` 1.0.0.

---

## v1.2.0 SDK — 2026-04-23

Pre-GA SDK release that exposes the Wave 2 / Wave 2a / Wave 2.5 envelope surface.

### Added

- Typed envelope models (`ChosenFactor`, `SourceDescriptor`, `QualityEnvelope`, `UncertaintyEnvelope`, `LicensingEnvelope`, `DeprecationStatus`, `SignedReceipt`).
- `audit_text` + `audit_text_draft` fields — surfaces the methodology-review draft banner per [`docs/specs/audit_text_template_policy.md`](../specs/audit_text_template_policy.md).
- `FactorCannotResolveSafelyError` exception (maps 422 `factor_cannot_resolve_safely`).
- CLI: `gl-factors resolve --show-full-audit`; `gl-factors explain --pretty`; `gl-factors verify-receipt --key <path>`.
- TS: `FactorsClient.verifyReceipt()` returns `alg`, `receipt_id`, `verification_key_hint`.

### Changed

- Signed receipt key names: `receipt_id`, `signature`, `verification_key_hint`, `alg`, `payload_hash` (new). SDK reads deprecated aliases (`_signed_receipt`, `algorithm`, `signed_over`) for one release with a deprecation warning.
- `ResolvedFactor.deprecation_status` is now `str | DeprecationStatus | None`.

### Deprecated (removed in v2.0.0)

- `_signed_receipt` top-level alias (use `signed_receipt`).
- `algorithm` field inside receipts (use `alg`).
- `signed_over` field inside receipts (use `payload_hash`).

Full SDK notes: [`greenlang/factors/sdk/CHANGELOG.md`](../../greenlang/factors/sdk/CHANGELOG.md).

---

## Edition release cadence

- **Certified** — quarterly minor releases (`v1.1`, `v1.2`, ...). Backward-compatible schema; additive source / factor additions; no retroactive factor value changes.
- **Patch** — as-needed (`v1.0.1`, ...) for audit-text fixes, documentation, and non-value-impacting corrections.
- **Preview** — rolling. Preview editions carry `preview`-status factors and connector-only sources that have not cleared Certified gating.

Every edition is signed with an Ed25519 key published at `https://api.greenlang.io/.well-known/jwks.json`. Old keys remain in the JWKS for 18 months so historical receipts verify.

---

## Related

- [`concepts/edition.md`](concepts/edition.md), [`roadmap.md`](roadmap.md).
- [SDK changelog](../../greenlang/factors/sdk/CHANGELOG.md).
