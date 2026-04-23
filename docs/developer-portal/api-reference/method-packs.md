# API — `GET /v1/method-packs` and `GET /v1/method-packs/{pack_id}`

List installed method packs and inspect their selection / boundary / fallback rules.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: listMethodPacks`, `getMethodPack`).

---

## List — `GET /v1/method-packs`

```bash
curl "https://api.greenlang.io/v1/method-packs" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Response

```json
{
  "items": [
    {
      "pack_id": "corporate_scope2_location_based",
      "pack_name": "Corporate Inventory - Scope 2 (Location-Based)",
      "version": "1.0.0",
      "status": "certified",
      "standards_alignment": ["GHG Protocol Corporate", "GHG Protocol Scope 2 Guidance", "CSRD_E1"],
      "deprecation_window_days": 365,
      "owner_methodology_lead": "methodology-wg@greenlang.io",
      "last_reviewed_iso": "2026-04-22"
    },
    {
      "pack_id": "freight_iso14083_glec_wtw",
      "pack_name": "Freight - ISO 14083 / GLEC WTW",
      "version": "1.0.0",
      "status": "certified",
      "standards_alignment": ["ISO 14083:2023", "GLEC Framework v3.0"]
    },
    { "pack_id": "eu_cbam", "version": "1.0.0", "status": "certified",
      "standards_alignment": ["EU CBAM Regulation (EU) 2023/956 Annex I"] }
  ]
}
```

---

## Get — `GET /v1/method-packs/{pack_id}`

```bash
curl "https://api.greenlang.io/v1/method-packs/corporate_scope2_location_based" \
  -H "Authorization: Bearer $GL_API_KEY"
```

Returns the pack's full configuration:

- `selection` — allowed families, formula types, jurisdiction hierarchy, priority tiers.
- `boundary` — scopes, system boundary, WTW/TTW, T&D losses.
- `gas_to_co2e` — default GWP set, allowed overrides.
- `biogenic_treatment` — `fossil_only`, `include_biogenic`, `separate_report`, `neutral_with_lulucf`.
- `market_instruments` — REC/GO/PPA/residual-mix treatment rules.
- `region_hierarchy` — the ordered fallback chain.
- `fallback.cannot_resolve_action` — `raise_no_safe_match` for all Certified packs.
- `reporting_labels` — tags auto-applied to every returned record.
- `audit_text_template_id` — which template `/explain` uses.

See the full spec at [`docs/specs/method_pack_template.md`](../../specs/method_pack_template.md).

---

## Python / TypeScript

```python
pack = client.get_method_pack("eu_cbam")
print(pack.version, pack.standards_alignment)
print(pack.fallback.cannot_resolve_action)  # "raise_no_safe_match"
```

```ts
const pack = await client.getMethodPack("eu_cbam");
```

---

## Listing the fourteen profiles

The pack ID usually matches the `method_profile` value, with two exceptions:

- `freight_iso14083_glec_wtw` and `freight_iso14083_glec_ttw` share one pack with two boundary variants (`WTW` / `TTW`).
- `product_carbon`, `product_iso14067`, and `product_pact` share one pack with three variants.

For a full list, see [`concepts/method_pack.md`](../concepts/method_pack.md).

---

## Errors

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | Missing token. |
| 402 | `payment_required` | Pack requires a paid SKU (e.g. `eu_cbam` in Enterprise SKU). |
| 404 | `not_found` | Unknown `pack_id`. |
| 410 | `pack_deprecated` | Deprecated pack requested; includes `replacement_pack_id`. |

See [`error-codes.md`](../error-codes.md).

## Related

- [`concepts/method_pack.md`](../concepts/method_pack.md), [method pack spec](../../specs/method_pack_template.md).
- Per-pack docs: [corporate](../method-packs/corporate.md), [electricity](../method-packs/electricity.md), [freight](../method-packs/freight.md), [eu_policy](../method-packs/eu_policy.md), [land_removals](../method-packs/land_removals.md), [product_carbon](../method-packs/product_carbon.md), [finance_proxy](../method-packs/finance_proxy.md).
