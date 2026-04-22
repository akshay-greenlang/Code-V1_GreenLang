# Cookbook: CBAM — Hot-Rolled Steel Coil Imported from India

**Scenario:** You are an EU importer. In Q2 2026 you landed 500 tonnes of hot-rolled steel coil, CN8 `72083610`, produced by a specific Indian steel mill. You need to declare the embedded emissions for CBAM.

**Method profile:** `eu_cbam`.

**Goal:** pick the supplier's primary-data factor if they provide one, fall back to an open-licensed country-average factor if they do not, and generate a signed, edition-pinned artefact you can attach to your CBAM declaration.

---

## Step 1: Pin the edition

CBAM declarations are audited years later. Pin the edition the moment the reporting period closes.

```bash
export GL_EDITION="2027.Q1-eu-cbam"   # or whichever stable edition is current
```

---

## Step 2: Resolve with supplier primary data (ideal path)

If your supplier gave you a verified PCF (they should, under CBAM Implementing Act 2023/1773), load it into the tenant overlay first, then resolve:

```bash
# 2a. Upload the supplier's PCF to tenant overlay (one-time per supplier release).
curl -sS -X POST "$GL_API_BASE/api/v1/overrides" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_id": "OVR:IN:steel_hrc:supplier_acme_steel:2026",
    "method_profile": "eu_cbam",
    "supplier_id": "acme-steel-india",
    "jurisdiction": "IN",
    "activity": "hot_rolled_steel_coil",
    "unit": "kg_CO2e_per_t",
    "cn8_code": "72083610",
    "gas_breakdown": {"co2": 2140.0, "ch4": 0.3, "n2o": 0.1},
    "methodology": "actual",
    "verification_status": "verified",
    "verifier": "TUV_Nord_IN_2026_certif_8821",
    "valid_from": "2026-01-01",
    "valid_to":   "2026-12-31"
  }'
```

```bash
# 2b. Resolve.
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d '{
    "activity": "hot_rolled_steel_coil",
    "method_profile": "eu_cbam",
    "jurisdiction": "IN",
    "reporting_date": "2026-06-30",
    "supplier_id": "acme-steel-india",
    "extras": {
      "cn8_code": "72083610",
      "methodology_preference": "actual"
    }
  }'
```

Expected response (truncated):

```json
{
  "factor_id": "OVR:IN:steel_hrc:supplier_acme_steel:2026",
  "co2e_per_unit": 2140.4,
  "unit": "kg_CO2e_per_t",
  "fallback_rank": 2,
  "method_profile": "eu_cbam",
  "edition_id": "2027.Q1-eu-cbam",
  "source": {
    "organization": "Acme Steel India (tenant overlay)",
    "methodology": "actual",
    "verification": {"status": "verified", "verifier": "TUV_Nord_IN_2026_certif_8821"}
  },
  "cn8_code": "72083610",
  "license_class": "customer_private",
  "data_quality": {"overall_score": 92, "rating": "excellent"},
  "gas_breakdown": {"co2": 2140.0, "ch4": 0.3, "n2o": 0.1}
}
```

`fallback_rank = 2` tells you the **supplier-specific** step won (step 2 in the cascade). That is what CBAM wants — `methodology = "actual"`.

---

## Step 3: Without supplier data (fallback path)

If the supplier did not provide primary data, the cascade skips step 2 and proceeds through facility (3), utility/grid (4), country/sector average (5), method-pack default (6), global default (7).

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d '{
    "activity": "hot_rolled_steel_coil",
    "method_profile": "eu_cbam",
    "jurisdiction": "IN",
    "reporting_date": "2026-06-30",
    "extras": {"cn8_code": "72083610"}
  }'
```

Response (truncated):

```json
{
  "factor_id": "EF:IN:steel_hrc:country_avg:2024:v1",
  "co2e_per_unit": 2460.0,
  "unit": "kg_CO2e_per_t",
  "fallback_rank": 5,
  "method_profile": "eu_cbam",
  "edition_id": "2027.Q1-eu-cbam",
  "source": {
    "organization": "JRC CBAM Default Values 2024",
    "methodology": "default"
  },
  "cn8_code": "72083610",
  "license_class": "open",
  "data_quality": {"overall_score": 76, "rating": "good"}
}
```

Under CBAM Implementing Act 2023/1773 Article 4(3), default values are permitted only when actual values cannot reasonably be obtained. The response shows `methodology = "default"` and you must document in your CBAM declaration why supplier data was not available.

---

## Step 4: What the CBAM method pack enforces

The `eu_cbam` pack is the strictest profile:

- `allowed_statuses = ("certified",)` — Preview factors are forbidden.
- `require_verification = True` for `methodology = "actual"` factors.
- `license_class_homogeneity` enforced — you cannot accidentally mix an open default with a restricted-licence alternative.
- Response carries `cn8_code`, `methodology` (default | actual), `verification_status`.
- Uses **IPCC AR6 100-year** GWPs per Article 4 of Implementing Act 2023/1773.

See [method-packs](../concepts/method-packs.md) and the pack source at `greenlang/factors/method_packs/eu_policy.py`.

---

## Step 5: Scale to a shipment

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-eu-cbam",
)

SHIPMENT_TONNES = 500.0

resolved = client.resolve_explain({
    "activity": "hot_rolled_steel_coil",
    "method_profile": "eu_cbam",
    "jurisdiction": "IN",
    "reporting_date": "2026-06-30",
    "supplier_id": "acme-steel-india",
    "extras": {"cn8_code": "72083610"},
})

embedded = resolved.co2e_per_unit * SHIPMENT_TONNES
print(f"Embedded emissions: {embedded:,.1f} kgCO2e ({embedded/1000:,.2f} tCO2e)")
# Embedded emissions: 1,070,200.0 kgCO2e (1,070.20 tCO2e)
```

---

## Step 6: Produce the CBAM evidence package

The audit-bundle endpoint gives you a single archive you can attach to the CBAM declaration:

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
     -H "X-Factors-Edition: $GL_EDITION" \
     "$GL_API_BASE/api/v1/factors/OVR:IN:steel_hrc:supplier_acme_steel:2026/audit-bundle" \
     > cbam_evidence.json
```

The bundle includes:

- The resolved factor's full `ResolvedFactor` payload.
- Gas breakdown and GWP table used.
- Provenance (source, vintage, methodology, verifier).
- Edition manifest fingerprint.
- Signed receipt (edition-bound).
- License attribution text.

Store this file with your declaration. During audit, the verifier can re-derive the signed receipt offline to prove the number has not been altered.

---

## Common pitfalls

- **Wrong CN8:** CBAM goods are identified by 8-digit CN. If your `extras.cn8_code` does not match, the method pack rejects the candidate at step 5 and you drop to the global default. Check `docs/api/factors-v1.yaml` for the CBAM goods list.
- **Missing verification:** `methodology=actual` requires an accredited verifier. If `verification_status != "verified"`, the pack downgrades it to `methodology=default` fallback logic.
- **Scope 2 of the supplier:** Steel mill's Scope 2 (purchased electricity) is in-scope for CBAM embedded emissions. When loading supplier PCFs, include the electricity-attributed portion in the total, not just direct emissions.
- **Forgot to pin edition:** Without an edition pin, a future eGRID / JRC default refresh will silently change your numbers. Always pin.

---

## See also

- [Method packs — EU CBAM](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md)
- [Version pinning](../concepts/version-pinning.md)
- [Licensing classes](../concepts/licensing-classes.md)
- [Signed receipts](../concepts/signed-receipts.md)

---

## File citations

| Piece | File |
|---|---|
| EU CBAM method pack | `greenlang/factors/method_packs/eu_policy.py` |
| Method profile enum | `greenlang/data/canonical_v2.py::MethodProfile.EU_CBAM` |
| Resolution engine (cascade step 2 for supplier match) | `greenlang/factors/resolution/engine.py` |
| Tenant overlay writer | `greenlang/factors/tenant_overlay.py` |
