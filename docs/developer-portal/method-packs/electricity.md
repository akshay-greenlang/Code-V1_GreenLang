# Method Pack — Electricity

Implements **GHG Protocol Scope 2 Guidance** (2015) for both location-based and market-based accounting. Handles grid-average factors, supplier-specific tariffs, REC/GO/I-REC certificates, PPAs, and residual mix.

This is not a standalone `method_profile` — electricity factors are consumed under `corporate_scope2_location_based` or `corporate_scope2_market_based`. The pack captures the electricity-specific rules those profiles inherit.

---

## Standards alignment

- **GHG Protocol Scope 2 Guidance** (WRI/WBCSD, 2015) — authoritative reference. §6.1 (location-based), §6.2 (market-based), §7 (Quality Criteria 1-7). [Link](https://ghgprotocol.org/scope_2_guidance).
- **AIB European Residual Mixes** — annual publication for EU/EEA residual mix.
- **IPCC AR6** — GWP basis for CH4 and N2O on grid factors.

---

## `electricity_basis`

Every electricity factor carries `parameters.electricity_basis`:

| Value | Meaning | When used |
|---|---|---|
| `location` | Grid-average intensity for the jurisdiction. | `corporate_scope2_location_based` |
| `market` | Supplier-specific, contracted, or certificate-claimed. | `corporate_scope2_market_based` |
| `supplier` | Supplier-reported but without certificate. | `market` mode fallback |
| `residual` | Residual mix — grid average minus claimed low-carbon MWh. | Market-based fallback when no certificate held |

The resolver refuses to return a `residual` factor under a location-based request (methodological error; would double-count claimed MWh).

---

## Supported sources

### Location-based (grid average)

| Source | Coverage | License class |
|---|---|---|
| EPA eGRID | US (27 subregions) | `open` |
| UK DESNZ GHG Conversion Factors | UK | `open` (OGL v3) |
| India CEA CO2 Baseline Database | India (national) | `open` |
| Australia DCCEEW National Greenhouse Accounts Factors | Australia | `open` (CC-BY-4.0) |
| Canada CER Provincial Electricity Intensity | Canada (provincial) | `open` (OGL-Canada 2.0) |
| Japan METI/MOEJ Electric Utility Emission Factors | Japan (per-utility) | `open` pending Legal |
| Korea KEMCO | South Korea | `open` pending Legal |
| Singapore EMA Grid Emission Factor | Singapore | `open` pending Legal |

### Market-based residual mix

| Source | Coverage | License class |
|---|---|---|
| AIB European Residual Mixes | EU/EEA | `open` pending parser/registry reconciliation |
| Green-e Residual Mix | US + CA | `licensed_embedded` (BYO at launch; contract in progress) |
| UK DESNZ Residual | UK | `open` |
| Canada CER Residual | Canada provincial | `open` |
| Australia NGER state residual | Australia | `open` |
| Japan METI residual derivation | Japan | `open` |
| Korea KEMCO residual | South Korea | `open` |
| Singapore EMA residual | Singapore | `open` |

Per source rights, see [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md).

---

## Selection rules

- `selection.allowed_families`: `["electricity", "residual_mix"]`.
- `selection.jurisdiction_hierarchy`: `["grid_region", "region", "country", "global"]`.
- `selection.priority_tiers`:
  Location-based: `["utility", "grid_subregion", "country", "global"]`.
  Market-based: `["supplier_contracted", "certificate", "utility_mix", "residual_mix"]`.

---

## Boundary

- `boundary.include_transmission_losses`: `false` by default (busbar basis). Set `true` only when the factor explicitly includes T&D losses (delivered basis).

---

## Certificate handling (market-based)

`parameters.certificate_handling` values: `GO` (EU), `REC` (US), `I-REC` (global residual markets), or `null` when no certificate is claimed.

Quality Criteria 1-7 (GHG Protocol Scope 2 §7) MUST all be satisfied before a certificate-based factor is returned:

1. Attribute conveyance in contract
2. Claim and retirement on reporter's behalf
3. Tracking system documented
4. Vintage ≤ 15 months from consumption
5. Same market
6. Market publishes residual mix (avoids double counting)
7. Single-count retirement proof

The resolver rejects a certificate-based factor if any QC is undocumented.

---

## Fallback

`fallback.cannot_resolve_action = raise_no_safe_match`. Residual mix is the final fallback in market-based mode; if no residual-mix factor is available for the jurisdiction, the resolver raises `FactorCannotResolveSafelyError` rather than fall through to a location-based factor (would be a methodological error).

---

## Related

- [Corporate pack](corporate.md), [`/resolve`](../api-reference/resolve.md).
- [`concepts/method_pack.md`](../concepts/method_pack.md).
