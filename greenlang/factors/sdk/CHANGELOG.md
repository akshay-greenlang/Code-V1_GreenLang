# Changelog — `greenlang-factors-sdk` / `@greenlang/factors-sdk`

All notable changes to the GreenLang Factors SDKs (Python + TypeScript)
are documented here. Semantic Versioning applies. The Python and
TypeScript SDKs share a version number.

## [1.1.0] — 2026-04-20

First release cut against the **100 %-CTO-spec Factors platform** (F1–F10
execution). This release focuses on SDK surface additions that expose the
new server capabilities; the client contract is backward compatible with
1.0.0 clients.

### Added

- **Resolution + explain.** New `FactorsClient.resolve(...)` + `.explain(...)`
  methods map to `POST /api/v1/factors/resolve` and
  `GET /api/v1/factors/{factor_id}/explain`. Returns the full `ResolvedFactor`
  payload — chosen factor, alternates considered, tie-break reasons,
  assumptions, gas breakdown, uncertainty band, and deprecation status.
- **Method profile support.** All resolve / match requests now accept a
  required `method_profile` parameter (enum: `corporate_scope1`,
  `corporate_scope2_location_based`, `corporate_scope2_market_based`,
  `corporate_scope3`, `product_carbon`, `freight_iso_14083`,
  `land_removals`, `finance_proxy`, `eu_cbam`, `eu_dpp`). Non-negotiable #6.
- **Mapping layer** client-side helpers: `map_fuel`, `map_transport`,
  `map_material`, `map_waste`, `map_electricity_market`,
  `map_classification`, `map_spend`.
- **Signed receipts.** `FactorsClient.verify_receipt(response, receipt)`
  validates server-issued HMAC-SHA256 or Ed25519 receipts attached to
  API responses.
- **Webhook registration.** `FactorsClient.register_webhook(url, event_types)`
  calls `POST /api/v1/factors/webhooks`. Customers receive HMAC-signed
  notifications on factor deprecation, license change, or impact-sim
  completion.
- **Status + watch endpoints.** `FactorsClient.status_summary()` +
  `FactorsClient.watch_status()` (both unauthenticated, cached 5 min).
- **TypeScript SDK:** mirrors the Python API surface; dual ESM + CJS.

### Changed

- `FactorsClient.match(...)` now surfaces `redistribution_class`,
  `factor_family`, `method_profile`, `formula_type`, and
  `explainability.fallback_rank` on every returned candidate.
- Client-side defaults align with server tier enforcement; Developer
  tier gets `certified` + `preview`; Enterprise adds `connector_only`.

### Non-breaking

No fields removed, no method signatures changed. All new parameters
are keyword-only with safe defaults.

### Tested against

- Server: factors-api v1.1.0 (F1–F10 merge).
- Python: 3.9, 3.10, 3.11, 3.12, 3.13.
- Node: 18, 20, 22.

## [1.0.0] — 2026-02

Initial publishable release.

- `FactorsClient.search()`, `search_v2()`, `match()`, `calculate()`,
  `calculate_batch()`, `export()`, `list_editions()`, `get_factor()`,
  `diff_factor()`, `get_audit_bundle()`.
- Edition pinning via constructor config.
- Zero runtime dependencies (stdlib only on Python; dual-build ESM/CJS on TS).

---

## Release process

1. Bump `pyproject.toml[project.version]` **and** `ts/package.json.version`.
2. Add a dated entry to this file.
3. Tag: `git tag factors-sdk-v1.1.0 && git push --tags`.
4. Tag push fires `.github/workflows/factors-sdk-publish.yml`:
   - Validates tag version matches the manifest.
   - Runs pytest + vitest.
   - Builds sdist + wheel (Python) and ESM + CJS (TS).
   - Publishes to PyPI (`PYPI_API_TOKEN`) and npm (`NPM_TOKEN`).
   - Creates a GitHub Release with these changelog notes.
