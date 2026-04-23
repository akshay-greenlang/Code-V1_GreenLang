# GreenLang Factors — Developer Portal

> **GreenLang Factors = Factor Registry + Resolution Engine + Method Packs + Governance + API.**

A single, signed, reproducible surface for emission factors and climate reference data. Every response carries a pinned `edition_id`, a signed receipt, and a fully explainable resolution trace. Callers cannot obtain a raw factor without first binding to a methodology profile (GHG Protocol Corporate, ISO 14083, CBAM, PCAF, etc.).

This portal is for **developers, auditors, consultants, and platform partners** integrating GreenLang Factors. It is API-first. If you want marketing copy, use `https://greenlang.io`; if you want to ship working code, start here.

---

## What this platform does

| Pillar | One-line contract |
|---|---|
| **Registry** | Every factor stored with gas vectors (CO2/CH4/N2O/F-gases), not just CO2e. Never overwritten; always semver-versioned. |
| **Resolution** | 7-step fallback cascade from `customer_override` to `global_default`. Never hides fallback. Returns a structured `FactorCannotResolveSafelyError` rather than a weak default. |
| **Method packs** | Every resolution binds a `method_profile` (Corporate, Electricity, Freight, CBAM, LSR, Product, Finance). The resolver refuses mismatched pairs. |
| **Governance** | Immutable `edition_id` for reproducibility. Signed receipts (HMAC-SHA256 or Ed25519). Licensing classes never mix in a single response. |
| **API** | REST + GraphQL + webhooks. OpenAPI 3.1 at `docs/api/factors-v1.yaml`. |

See the spec: [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md).

---

## Quick paths

| I want to... | Start here |
|---|---|
| Ship a first `/resolve` call in 10 minutes | [Quickstart](quickstart.md) |
| Understand factors, sources, editions, packs | [Concepts](#concepts) |
| See every endpoint + example | [API Reference](#api-reference) |
| Install an SDK | [SDKs](#sdks) |
| Understand what ships in each SKU | [Licensing](licensing.md) |
| Verify a signed receipt offline | [`concepts/signed_receipt.md`](concepts/signed_receipt.md) |
| See the coverage dashboard | [Coverage](coverage.md) |

---

## Concepts

Short, cross-linked pages. 200-400 words each.

- [Factor](concepts/factor.md) — the unit of data
- [Source](concepts/source.md) — the upstream publisher
- [Method Pack](concepts/method_pack.md) — methodology binding
- [Edition](concepts/edition.md) — immutable catalog snapshot
- [License Class](concepts/license_class.md) — the 4 data classes
- [Quality Score](concepts/quality_score.md) — FQS 0-100
- [Signed Receipt](concepts/signed_receipt.md) — proof-of-resolution envelope

---

## API reference

Each endpoint page has curl + Python + TypeScript examples.

- [`POST /v1/factors/resolve`](api-reference/resolve.md) — 7-step cascade
- [`POST /v1/factors/{id}/explain`](api-reference/explain.md) — full derivation trace
- [`POST /v1/factors/search`](api-reference/factors.md) — catalog search
- [`GET /v1/sources`](api-reference/sources.md) — publisher catalog
- [`GET /v1/method-packs`](api-reference/method-packs.md) — methodology profiles
- [`GET /v1/editions`](api-reference/releases.md) — pinned snapshots
- [`POST /v1/factors/batch-resolve`](api-reference/batch.md) — batched resolution
- [`POST /v1/webhooks`](api-reference/webhooks.md) — factor-change events
- [`POST /v1/graphql`](api-reference/graphql.md) — GraphQL surface
- [`GET /v1/quality/{factor_id}`](api-reference/quality.md) — FQS surface

---

## SDKs

- [Python — `greenlang-factors`](sdks/python.md)
- [TypeScript — `@greenlang/factors`](sdks/typescript.md)
- [CLI — `gl-factors`](sdks/cli.md)

---

## Method packs

- [Corporate (GHG Protocol)](method-packs/corporate.md)
- [Electricity](method-packs/electricity.md)
- [Freight (ISO 14083 + GLEC)](method-packs/freight.md)
- [EU Policy (CBAM + DPP)](method-packs/eu_policy.md)
- [Land & Removals (GHG Protocol LSR)](method-packs/land_removals.md)
- [Product Carbon (ISO 14067 + PACT)](method-packs/product_carbon.md)
- [Finance (PCAF)](method-packs/finance_proxy.md)

---

## Reference material

- [Canonical factor record schema](schema.md)
- [Licensing posture and BYO credentials](licensing.md)
- [Error codes](error-codes.md)
- [Changelog](changelog.md)
- [Public roadmap](roadmap.md)
- [Coverage dashboard](coverage.md)

---

## Support

- OpenAPI spec: [`docs/api/factors-v1.yaml`](../api/factors-v1.yaml)
- Issues: `platform@greenlang.io`
- Status: `https://status.greenlang.io`
