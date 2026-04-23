# Public roadmap

What ships when and under what boundary (open-core, commercial, deferred). This is the public commitment; dates slip only with a webhook notification plus a changelog entry.

Last updated: 2026-04-23.

---

## Now — v1.0 Certified (cut target 2026-05-01)

Open-core:
- Canonical factor record v1 (frozen 2026-04-22).
- REST + GraphQL API surface.
- Seven-step resolution cascade with `raise_no_safe_match`.
- Signed receipts (Ed25519 + HMAC-SHA256).
- Python + TypeScript SDKs 1.0.
- `gl-factors` CLI.
- Bulk export (open-class only).
- Community tier factors (EPA, UK DESNZ, India CEA, Australia DCCEEW, Canada CER, US SUSEEIO, IPCC defaults).
- 14 method profiles (Corporate, Product, Freight, LSR, Finance, CBAM, DPP).

Commercial:
- Developer Pro SKU (Community + published `/explain` + batch API).
- Electricity Premium, Freight Premium, Product Carbon Premium, Finance Premium SKUs — **BYO-credentials** at launch for commercial sources. See [`licensing.md`](licensing.md).
- Enterprise / CBAM SKU (BYO IEA, EU CBAM default values).

---

## Next — v1.1 (Q3 2026)

Open-core:
- Additional jurisdictions: Brazil (MMA), Mexico (SEMARNAT), Indonesia (KLHK).
- Scope 2 residual-mix coverage for non-AIB markets (Japan, Korea, Singapore confirmed; US/CA pending Green-e contract).
- Webhooks v2: filtered subscriptions by `factor_family` / `method_profile`.
- Reproducibility CLI: `gl-factors replay --edition <id> --batch <file>` for regression runs against past editions.

Commercial:
- **ecoinvent embedded redistribution** — upgrades ecoinvent from BYO to `licensed_embedded` upon contract close (target Q3 2026).
- **GLEC Framework embedded** — upgrades from BYO upon Smart Freight Buyer Group membership (target Q3 2026).
- **Green-e Residual Mix embedded** — upgrades from BYO upon CRS contract close (target Q3 2026).

---

## After — v1.2 (Q4 2026)

Open-core:
- `GWP*` metric opt-in for methane-heavy packs (agriculture, LSR).
- CBAM definitive-period submission helper (XML export per Implementing Regulation (EU) 2023/1773).
- CSRD ESRS E1 disclosure package — factor-level evidence pack for independent limited assurance.

Commercial:
- **IEA embedded** — only with redistribution addendum; if negotiation stalls, IEA remains BYO indefinitely.
- **EC3 EPD library embedded** — upon Building Transparency partnership close.
- OEM white-label flow: sub-tenant resolution with parent-tenant entitlement inheritance. `oem_redistributable` class becomes populated.

---

## Deferred — not on v1 path

- Real-time grid intensity (Electricity Maps embedded). Electricity Maps' license forbids redistribution; remains BYO indefinitely unless terms change.
- Consequential LCA mode. Attributional LCA is the v1 default; consequential is under methodology review.
- `GTP` (Global Temperature Potential) as a default metric. Supported as an override only.
- Monetized damage factors (social cost of carbon translation). Out of scope for Factors; will ship as a separate module.
- Non-CO2 air pollutant factors (SOx, NOx, PM). Pipeline under consideration for Factors v2.
- ISO 14097 climate-related investment reporting. Adjacent to PCAF; pipeline-deferred.

---

## Clear boundaries

### Open-core

Everything needed to resolve a factor from an Open-class source, verify a signed receipt, and export a reproducible audit trail. Released under MIT for the catalog schemas and BSD-3 for the SDK reference implementations.

### Commercial

Premium Packs (Freight, Product Carbon, Electricity, Finance, CBAM, DPP) — entitlements on top of the open API. Subscriptions track the contract stage of each upstream publisher. SKU details at `https://greenlang.io/pricing`.

### Deferred

Items explicitly not in the v1 path. We list them so customers know what not to wait for.

---

## Customer commitments

- **Reproducibility**: Any inventory filed under edition `X` is exactly reproducible under edition `X` at any future date (up to 10 years).
- **Receipt verification**: Ed25519 public keys remain in the JWKS for 18 months after retirement.
- **Pack deprecation window**: Certified pack → deprecated requires 180 days notice (365 for Corporate / Scope 2). No retroactive factor value changes — always a new `factor_version`.
- **BYO transparency**: Every BYO source is listed at [`licensing.md`](licensing.md); upgrades to `licensed_embedded` fire a `source.updated` webhook.

---

## Related

- [`changelog.md`](changelog.md), [`licensing.md`](licensing.md), [`concepts/edition.md`](concepts/edition.md).
- Contract posture: [`docs/legal/source_contracts_outreach.md`](../legal/source_contracts_outreach.md).
