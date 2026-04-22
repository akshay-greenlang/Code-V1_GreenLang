# Cookbook: Product Carbon — Exporting a PACT-Compatible Footprint

The **Partnership for Carbon Transparency (PACT)** publishes a Pathfinder Framework + Data Exchange Protocol that lets suppliers and customers exchange product carbon footprints (PCFs) in a standard JSON shape. Any PCF you compute via GreenLang Factors can be rendered as a PACT-compatible `ProductFootprint` object.

**Method profile:** `product_carbon` (with `extras.output_shape = "pact"`).
**Implementation:** `greenlang/factors/method_packs/product_carbon.py`, `product_lca_variants.py`.

---

## Scenario

You manufacture a printed-circuit-board assembly (PCBA) in Vietnam. Primary data:

- Bill of materials: 12 g solder, 42 g PCB substrate (FR-4), 18 g copper, 4.5 g gold plating, 120 g plastic enclosure (ABS), 380 g aluminium heatsink.
- Factory electricity: 0.85 kWh per unit (VN grid).
- Factory heat: 12 MJ per unit (natural gas boiler).
- Transport to Rotterdam: 10,500 km sea freight.

Boundary: **cradle-to-gate** (raw materials + manufacturing, no use phase or end-of-life for this CBAM-focused declaration).

GWP set: AR6 100-year.

---

## Step 1: Resolve the constituent factors

Use one `resolve-explain` call per line item, then aggregate on the client side. A batch job would also work for high-volume cases.

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-product-carbon",
)

def kg_co2e(activity, jurisdiction, mass_or_energy, unit_hint=None, extras=None):
    r = client.resolve_explain({
        "activity": activity,
        "method_profile": "product_carbon",
        "jurisdiction": jurisdiction,
        "reporting_date": "2026-06-01",
        "gwp_set": "IPCC_AR6_100",
        "extras": extras or {},
    })
    return r.co2e_per_unit * mass_or_energy, r.factor_id, r.edition_id

# Materials (cradle-to-gate).
contributions = [
    kg_co2e("solder_sac305_production",       "GLOBAL", 0.012),
    kg_co2e("pcb_substrate_fr4_production",    "GLOBAL", 0.042),
    kg_co2e("copper_primary_production",       "GLOBAL", 0.018),
    kg_co2e("gold_plating_production",         "GLOBAL", 0.0045),
    kg_co2e("abs_plastic_production",          "GLOBAL", 0.120),
    kg_co2e("aluminium_primary_production",    "GLOBAL", 0.380),

    # Energy inputs at the Vietnamese factory.
    kg_co2e("purchased_electricity", "VN", 0.85,
            extras={"method_profile_override":"corporate_scope2_location_based"}),
    kg_co2e("natural_gas_combustion_stationary", "VN", 12/1000),  # MJ -> GJ

    # Transport: sea freight, GLEC container defaults, 10500 km.
    kg_co2e("freight_transport", "GLOBAL", 10500,
            extras={"mode":"sea","boundary":"WTW","payload_tonnes":0.0005,
                    "vehicle_class":"container_ship_generic_HFO",
                    "utilisation_pct":70}),
]
```

Each tuple is `(kg_co2e_contribution, factor_id, edition_id)`. Keep the factor_ids — they go into the PACT output.

---

## Step 2: Aggregate

```python
total_kgco2e = sum(c[0] for c in contributions)
print(f"Cradle-to-gate PCF: {total_kgco2e:.3f} kgCO2e per unit")
```

---

## Step 3: Request the PACT output shape

Re-run the resolution with `extras.output_shape = "pact"` and the pre-computed contributions as an attached bill-of-materials block. The pack emits a PACT `ProductFootprint` object directly:

```python
pf = client.resolve_explain({
    "activity": "product_carbon_footprint",
    "method_profile": "product_carbon",
    "jurisdiction": "VN",
    "reporting_date": "2026-06-01",
    "gwp_set": "IPCC_AR6_100",
    "extras": {
        "output_shape": "pact",
        "product_ids": {
            "sku": "ACME-PCBA-001",
            "gtin": "05012345678900",
            "internal_id": "ACME-PCBA-001"
        },
        "boundary": "cradle_to_gate",
        "functional_unit": {"value": 1, "unit": "unit"},
        "bill_of_materials": [
            {"factor_id": fid, "mass_or_energy": moe, "contribution_kgco2e": kg}
            for (kg, fid, _) in contributions
            for moe in [None]       # populate from your own BOM record
        ],
        "precomputed_total_kgco2e": total_kgco2e
    }
})

print(pf.pact_product_footprint)
```

---

## Step 4: The PACT ProductFootprint output

```json
{
  "specVersion": "2.0.0",
  "id": "urn:pf:01HYPW8A0...",
  "version": 1,
  "created": "2026-06-01T10:00:00Z",
  "status": "Active",
  "validityPeriodStart": "2026-01-01T00:00:00Z",
  "validityPeriodEnd":   "2026-12-31T23:59:59Z",
  "companyName": "Acme Manufacturing Vietnam",
  "companyIds": ["urn:epc:id:gln:0614141.00001.0"],
  "productDescription": "PCBA for industrial IoT gateway",
  "productIds": ["urn:gtin:05012345678900"],
  "productCategoryCpc": "4737",
  "productNameCompany": "ACME-PCBA-001",
  "comment": "Cradle-to-gate PCF computed via GreenLang Factors, edition 2027.Q1-product-carbon.",

  "pcf": {
    "declaredUnit": "unit",
    "unitaryProductAmount": 1,
    "pCfExcludingBiogenic": 1.824,
    "pCfIncludingBiogenic": 1.824,
    "fossilGhgEmissions":   1.811,
    "biogenicCarbonWithdrawal": 0,
    "biogenicCarbonEmissions": 0,
    "characterizationFactors": "AR6",
    "characterizationFactorsSources": ["IPCC AR6 100-year"],
    "ipccCharacterizationFactorsSources": ["AR6"],
    "crossSectoralStandardsUsed": ["ISO_14067", "GHG_PRODUCT"],
    "productOrSectorSpecificRules": [],
    "boundaryProcessesDescription": "Cradle-to-gate. Raw material extraction + manufacturing + inbound logistics. Excludes use phase and end-of-life.",
    "referencePeriodStart": "2026-01-01T00:00:00Z",
    "referencePeriodEnd":   "2026-12-31T23:59:59Z",
    "geographyCountry": "VN",
    "primaryDataShare": 62,
    "dqi": {
      "coveragePercent": 100,
      "technologicalDqr": 2.3,
      "temporalDqr": 2.1,
      "geographicalDqr": 2.5,
      "completenessDqr": 2.0,
      "reliabilityDqr": 2.5
    },
    "assurance": null
  },

  "extensions": {
    "greenlang.edition_id": "2027.Q1-product-carbon",
    "greenlang.receipt_key_id": "gl-factors-v1",
    "greenlang.line_items": [
      {"factor_id":"EF:GLOBAL:solder_sac305:2024:v1","contribution_kgco2e":0.097},
      {"factor_id":"EF:GLOBAL:pcb_fr4:2024:v1","contribution_kgco2e":0.132},
      ...
    ]
  },

  "_signed_receipt": {
    "signature": "...",
    "algorithm": "sha256-hmac",
    "signed_at": "2026-06-01T10:00:00Z",
    "key_id": "gl-factors-v1",
    "payload_hash": "..."
  }
}
```

Key points:

- `characterizationFactors: "AR6"` — the GWP family used.
- `primaryDataShare: 62` — how much of the PCF came from primary (supplier-specific) data vs secondary factors.
- `dqi` — PACT's Pedigree-matrix scores (derived from GreenLang's 5-component DQS).
- `extensions.greenlang.edition_id` — the edition you pinned. Store this.
- `_signed_receipt` — the PACT spec does not require it, but PACT's own Exchange Protocol allows arbitrary extensions, and the receipt is how you prove non-tampering.

---

## Step 5: Exchange the footprint via PACT

You can now POST the object to any PACT-conformant Exchange endpoint:

```bash
curl -sS -X POST "https://pact-exchange.customer.com/2/footprints" \
  -H "Authorization: Bearer $PACT_EXCHANGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d @pcf.json
```

---

## Primary-data vs secondary-data mix

The `primaryDataShare` field is computed as:

```
primaryDataShare = 100 * sum(contribution where fallback_rank in [1,2,3]) / total
```

Cascade steps 1 (customer override), 2 (supplier-specific), and 3 (facility-specific) count as primary data. Steps 4-7 count as secondary. PACT recommends `primaryDataShare >= 50%` for a "good quality" PCF; the regulator-specific thresholds differ (EU Battery Passport requires `>= 70%` for battery cell data by 2027).

If your share is low, you either need better supplier data (upload to tenant overlay or get supplier-provided PACT footprints) or better facility data (install metering).

---

## Common pitfalls

- **Unit confusion:** PACT's `declaredUnit` must match the units of `unitaryProductAmount`. If your BOM is per 1000 units, scale accordingly.
- **Biogenic carbon:** If your product contains biogenic content (paper, wood, bio-plastic), separate the biogenic flow. PACT has explicit `biogenicCarbonWithdrawal` (negative during growth) and `biogenicCarbonEmissions` (positive at end-of-life) fields. GreenLang's `biogenic_co2` slot carries this.
- **Scope boundary:** Do not mix cradle-to-gate and cradle-to-grave in one exchange; they are not comparable.
- **Edition changes:** If your supplier publishes a new PACT footprint next quarter under a newer edition, treat it as a new `ProductFootprint.id` and a new version, not an update.

---

## See also

- [Method packs — product_carbon](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md)
- [Quality scores](../concepts/quality-scores.md) — DQI derivation.
- [Signed receipts](../concepts/signed-receipts.md)

---

## File citations

| Piece | File |
|---|---|
| Product carbon pack | `greenlang/factors/method_packs/product_carbon.py` |
| PACT / PEF / ISO 14067 output variants | `greenlang/factors/method_packs/product_lca_variants.py` |
| Profile enum | `greenlang/data/canonical_v2.py::MethodProfile.PRODUCT_CARBON` |
| DQS to PACT DQI mapping | `greenlang/factors/quality/composite_fqs.py` (`CTO_SPEC_ALIASES`) |
