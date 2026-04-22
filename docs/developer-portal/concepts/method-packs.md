# Method Packs

A method pack is a bundle of methodology rules attached to a specific reporting standard (GHG Protocol, ISO 14067, ISO 14083, PCAF, CBAM, etc.). Every resolution call must carry a `method_profile` identifying which pack to use; that requirement is **CTO non-negotiable #6** — there is no way to bypass it without re-writing the engine.

A pack bundles:

- **Selection rule** — which factor statuses, GWP sets, licensing, and verification attributes are acceptable.
- **Candidate generator** — extra filtering beyond the cascade (e.g. "Scope 2 market-based must have a contractual instrument attached").
- **Defaults** — the step-6 fallback factor when nothing more specific exists.
- **Output shape** — additional fields the resolved factor must carry (e.g. CBAM requires the CN8 code).

**Registry:** `greenlang/factors/method_packs/registry.py` (`register_pack`, `get_pack`).
**Base class:** `greenlang/factors/method_packs/base.py` (`MethodPack`).
**Profile enum:** `greenlang/data/canonical_v2.py::MethodProfile`.

---

## Why they matter

Without method packs, the same request ("diesel combustion, US") could return a factor that:

- Uses AR4 GWPs when your regulator requires AR6.
- Is in `preview` status when your regulator requires `certified` only.
- Mixes a `restricted` licensed factor with open government data (non-redistributable output).
- Omits a field (CBAM CN8 code) that makes the resolution unusable in your submission.

The method pack pins all of that to the reporting standard so the engine cannot produce an ineligible answer.

---

## The 10 registered profiles

The `MethodProfile` enum exposes these values (from `greenlang/data/canonical_v2.py`). Every resolution request must pass one of them.

| `method_profile` (wire value) | Standard anchor | One-liner | Implementation |
|---|---|---|---|
| `corporate_scope1` | GHG Protocol Scope 1 | Direct emissions from owned/controlled sources (stationary, mobile, process, fugitive). | `method_packs/corporate.py` |
| `corporate_scope2_location_based` | GHG Protocol Scope 2 Guidance | Location-based electricity / steam / heat / cooling (grid-average). | `method_packs/electricity.py` |
| `corporate_scope2_market_based` | GHG Protocol Scope 2 Guidance | Market-based electricity using contractual instruments (RECs, GOs, I-RECs, PPAs) and residual mix. | `method_packs/electricity.py` |
| `corporate_scope3` | GHG Protocol Scope 3 Standard | Value-chain emissions across Categories 1-15 (upstream + downstream). | `method_packs/corporate.py` |
| `product_carbon` | ISO 14067 / GHG Product Standard / PACT | Product-level cradle-to-gate or cradle-to-grave carbon footprint. | `method_packs/product_carbon.py`, `product_lca_variants.py` |
| `freight_iso_14083` | ISO 14083 (GLEC-aligned) | Freight transport emissions by lane, mode, payload, utilisation, WTW/TTW boundary. | `method_packs/freight.py` |
| `land_removals` | GHG LSR (Land Sector & Removals) | Biogenic removals and land-use emissions with permanence accounting. | `method_packs/land_removals.py` |
| `finance_proxy` | PCAF Global GHG Accounting Standard | Scope 3 Category 15 financed emissions by asset class (listed equity, corporate bonds, mortgages, CRE, motor, project finance). | `method_packs/finance_proxy.py` |
| `eu_cbam` | EU CBAM regulation 2023/956 + Implementing Act 2023/1773 | Embedded emissions for CBAM goods (cement, iron & steel, aluminium, fertilisers, hydrogen, electricity). | `method_packs/eu_policy.py` |
| `eu_dpp` | EU Digital Product Passport (ESPR + sector delegated acts) | Product-passport embedded emissions (batteries and textiles in first wave). | `method_packs/eu_policy.py` |

The CTO cut-list also references additional slices — `product_iso14067`, `product_pact`, `freight_iso14083_glec_wtw`, `freight_iso14083_glec_ttw`, `land_removals_ghgp_lsr`, `finance_proxies_pcaf`, `eu_dpp_battery`, `eu_dpp_textile` — but these are **sub-profiles** served by the same registered packs above. The engine routes a PACT request by accepting `method_profile=product_carbon` plus `extras={"output_shape":"pact"}`; the pack adapts output without needing a new enum value.

---

## Corporate Scope 1

```python
from greenlang.factors.method_packs import get_pack
from greenlang.data.canonical_v2 import MethodProfile

pack = get_pack(MethodProfile.CORPORATE_SCOPE1)
# pack.selection_rule.allowed_statuses -> ("certified",)
# pack.selection_rule.allowed_gwp_sets -> (IPCC_AR6_100, IPCC_AR5_100)
```

Only `certified` factors. Preview is forbidden so that annual inventories remain defensible under ISO 14064-1.

---

## Corporate Scope 2 (location-based vs market-based)

Two distinct packs because Scope 2 Guidance mandates **dual reporting**:

- **Location-based** uses the grid-average emission factor for the physical location.
- **Market-based** uses contractual instruments first (GO, I-REC, REC, PPA), falls back to supplier-specific mix, then residual mix, then grid average.

See [scope-2-location-vs-market cookbook](../cookbook/scope-2-location-vs-market.md) for a worked example.

---

## Corporate Scope 3

All 15 GHG Protocol categories share a single pack that multiplexes on `request.extras.scope3_category` (integer 1..15). The pack's selection rule permits `certified` and `preview` — Scope 3 reporting tolerates higher uncertainty than Scope 1/2 under the standard.

---

## Product Carbon (ISO 14067 / GHG PS / PACT)

`require_verification=True` and `allowed_statuses=("certified",)` — product claims carry brand risk, so the bar is raised. Output adapter supports PACT (Partnership for Carbon Transparency) export shape via `extras.output_shape="pact"`. See [product-carbon-pact-export cookbook](../cookbook/product-carbon-pact-export.md).

---

## Freight ISO 14083

GLEC-aligned lane calculations with explicit `well_to_wheel` (WTW) or `tank_to_wheel` (TTW) boundary. Payload and utilisation are first-class inputs; the pack requires mode (road/rail/air/sea/inland-waterway/pipeline) in `extras`. Boundary selection: `extras.boundary = "WTW" | "TTW"`.

---

## Land Removals (GHG LSR)

Permanence, baseline, additionality, and monitoring cadence are enforced. The pack emits biogenic CO2 separately from fossil per the CTO gas-breakdown rule. Removal factors carry a negative CO2e.

---

## Finance Proxies (PCAF)

Scope 3 Cat 15 via PCAF asset-class methodologies. The pack routes by `extras.asset_class` (listed_equity, corporate_bonds, project_finance, commercial_real_estate, mortgages, motor_vehicle_loans, sovereign_debt). Each asset class has its own data-quality tier (1-5) per PCAF; the pack surfaces that tier on the response.

---

## EU CBAM

The strictest profile. `allowed_statuses=("certified",)`, `require_verification=True`, and the selection rule forbids Preview even at step 7. Every resolved factor carries:

- CN8 code (8-digit Combined Nomenclature).
- Embedded emissions methodology marker (`default` or `actual` per CBAM Implementing Act).
- Country of origin (supplier primary data path is preferred — cascade step 2).

See [CBAM resolver cookbook](../cookbook/cbam-resolver.md).

---

## EU DPP

Digital Product Passport. Sub-profiles per delegated act — batteries first (EU 2023/1542), textiles second. Routes on `extras.dpp_variant = "battery" | "textile"` for now.

---

## How the registry discovers packs

Packs self-register at import time. `greenlang/factors/method_packs/__init__.py` imports each module, which calls `register_pack(pack_instance)` at module load. At request time, `get_pack(profile)` returns the registered instance or raises `MethodPackNotFound`.

```python
from greenlang.factors.method_packs.registry import (
    register_pack, get_pack, list_packs, registered_profiles,
)

for p in registered_profiles():
    print(p.value)
# corporate_scope1
# corporate_scope2_location_based
# corporate_scope2_market_based
# corporate_scope3
# eu_cbam
# eu_dpp
# finance_proxy
# freight_iso_14083
# land_removals
# product_carbon
```

---

## See also

- [Resolution cascade](./resolution-cascade.md) — how `SelectionRule` filters candidates at each step.
- [Quality scores](./quality-scores.md) — how `allowed_statuses` relates to FQS bands.
- [Licensing classes](./licensing-classes.md) — how `license_class_homogeneity` is enforced across a resolution.
- [CBAM cookbook](../cookbook/cbam-resolver.md)
- [Scope 2 cookbook](../cookbook/scope-2-location-vs-market.md)
- [Freight cookbook](../cookbook/freight-wtw-vs-ttw.md)
- [Product carbon cookbook](../cookbook/product-carbon-pact-export.md)
- [PCAF cookbook](../cookbook/financed-emissions-pcaf.md)

---

## File citations

| Piece | File |
|---|---|
| `MethodProfile` enum | `greenlang/data/canonical_v2.py` (line 137) |
| Registry | `greenlang/factors/method_packs/registry.py` |
| Base class | `greenlang/factors/method_packs/base.py` |
| Corporate Scope 1 / 3 pack | `greenlang/factors/method_packs/corporate.py` |
| Electricity (Scope 2 LB / MB) | `greenlang/factors/method_packs/electricity.py` |
| Product carbon | `greenlang/factors/method_packs/product_carbon.py`, `product_lca_variants.py` |
| Freight ISO 14083 | `greenlang/factors/method_packs/freight.py` |
| Land removals | `greenlang/factors/method_packs/land_removals.py` |
| PCAF | `greenlang/factors/method_packs/finance_proxy.py` |
| EU policy (CBAM, DPP) | `greenlang/factors/method_packs/eu_policy.py` |
