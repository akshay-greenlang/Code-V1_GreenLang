# GreenLang Factors Developer Portal

The source of truth for emission factors, with signed receipts, edition pinning, and a 7-step resolution cascade you can defend in front of an auditor.

---

## Get going

<table>
<tr>
<td width="33%"><b>Quickstart (5 min)</b><br/>Resolve your first factor, pull the signed receipt, verify it with HMAC. <code>curl</code> only, no SDK required.<br/><br/><a href="./quickstart/5-minute-quickstart.md">Start &rarr;</a></td>
<td width="33%"><b>Concepts</b><br/>Understand the resolution cascade, method packs, edition manifests, licensing classes, data quality scores, and signed receipts.<br/><br/><a href="./concepts/resolution-cascade.md">Read &rarr;</a></td>
<td width="33%"><b>API Reference</b><br/>Every endpoint, every header, every error code. Linked to the OpenAPI spec at <code>docs/api/factors-v1.yaml</code>.<br/><br/><a href="./api-reference/README.md">Browse &rarr;</a></td>
</tr>
</table>

---

## Who this is for

- **Climate-startup developers** building carbon-accounting products who need defensible factors with provenance.
- **In-house consultant engineers** producing CBAM / CSRD / TCFD / SBTi submissions who need to prove which edition was used.
- **Platform integrators** embedding GreenLang Factors into ERPs, spend-based tooling, PCF engines, and product passports.

If you finish the quickstart and cannot explain "why this factor won" to a teammate, file an issue. The portal is wrong, not you.

---

## Sitemap

### Quickstart

- [5-minute quickstart](./quickstart/5-minute-quickstart.md) — `curl` + HMAC receipt verification.
- [Python SDK](./quickstart/python-sdk.md) — `pip install greenlang-factors`, resolve + explain + pin edition.
- [TypeScript SDK](./quickstart/typescript-sdk.md) — `npm install @greenlang/factors`, same flow in Node.
- [cURL recipes](./quickstart/curl-recipes.md) — CI/CD friendly snippets.

### Concepts

- [Resolution cascade](./concepts/resolution-cascade.md) — the 7 steps that pick a factor.
- [Method packs](./concepts/method-packs.md) — the 10 methodology profiles you must choose from.
- [Version pinning & editions](./concepts/version-pinning.md) — how `X-GreenLang-Edition` survives audits.
- [Licensing classes](./concepts/licensing-classes.md) — open vs restricted vs licensed vs customer_private.
- [Quality scores (DQS + FQS)](./concepts/quality-scores.md) — the 5 dimensions and the 0-100 composite.
- [Signed receipts](./concepts/signed-receipts.md) — HMAC and Ed25519, edition binding, key rotation.
- [Gas breakdown vs CO2e](./concepts/gas-breakdown-vs-co2e.md) — why we never store CO2e only.

### API Reference

- [Overview](./api-reference/README.md)
- [Authentication](./api-reference/authentication.md) — JWT, API keys, tiers.
- [Rate limits](./api-reference/rate-limits.md) — per-tier sliding window.
- [Errors](./api-reference/errors.md) — full error-code table.
- [Webhooks](./api-reference/webhooks.md) — all 11 event types.

### Cookbook

- [CBAM resolver: hot-rolled steel coil from India](./cookbook/cbam-resolver.md)
- [Scope 2 location-based vs market-based](./cookbook/scope-2-location-vs-market.md)
- [Freight: WTW vs TTW under ISO 14083 / GLEC](./cookbook/freight-wtw-vs-ttw.md)
- [Refrigerant GWP selection](./cookbook/refrigerant-gwp-selection.md)
- [Product carbon: exporting a PACT-compatible footprint](./cookbook/product-carbon-pact-export.md)
- [Financed emissions (PCAF Scope 3 Cat 15)](./cookbook/financed-emissions-pcaf.md)

### Migration

- [Changelog](./migration/CHANGELOG.md) — v1.0 Certified and beyond.
- [From Climatiq](./migration/from-climatiq.md) — field mapping table.
- [From ecoinvent](./migration/from-ecoinvent.md) — connector + licensing.

---

## Support

- API status: `https://status.greenlang.io`
- Support: `support@greenlang.io`
- Source: `https://github.com/greenlang/greenlang`
- SDK source (Python): `greenlang/factors/sdk/python/`
- SDK source (TypeScript): `greenlang/factors/sdk/ts/`
