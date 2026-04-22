# Cookbook: Scope 2 — Location-Based vs Market-Based

GHG Protocol Scope 2 Guidance (2015 amendment) requires **dual reporting**: every reporter must publish both a location-based and a market-based Scope 2 number. The two use different factor selection rules, so you resolve them through two distinct method profiles.

**Method profiles:** `corporate_scope2_location_based`, `corporate_scope2_market_based`.
**Implementation:** `greenlang/factors/method_packs/electricity.py`.

---

## The rule in one sentence

- **Location-based** — use the grid-average emission factor for the physical location where the electricity was consumed. Ignore contracts.
- **Market-based** — use contractual instruments (GO, I-REC, REC, PPA) first; fall back to supplier-specific residual mix; then residual mix for the jurisdiction; then grid average.

---

## Scenario

A multi-site EU company consumes electricity in 2026:

| Facility | Country | MWh | Contract |
|---|---|---|---|
| Copenhagen HQ | DK | 1,200 | 100% Swedish hydropower via GO (EECS Guarantee of Origin) |
| Berlin office | DE | 800 | no contract (grid-supplied) |
| Stockholm lab | SE | 400 | no contract (grid-supplied) |

Report for reporting year 2026, resolved on 2026-12-31.

---

## Step 1: Location-based

Always use grid-average for the physical location.

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-electricity",
)

sites = [
    {"facility": "cph-hq", "country": "DK", "mwh": 1200},
    {"facility": "ber-office", "country": "DE", "mwh": 800},
    {"facility": "sto-lab", "country": "SE", "mwh": 400},
]

total_lb_kgco2e = 0.0
for site in sites:
    resolved = client.resolve_explain({
        "activity": "purchased_electricity",
        "method_profile": "corporate_scope2_location_based",
        "jurisdiction": site["country"],
        "reporting_date": "2026-12-31",
        "facility_id": site["facility"],
    })
    kg = resolved.co2e_per_unit * site["mwh"] * 1000   # kgCO2e/kWh * kWh
    total_lb_kgco2e += kg
    print(f"{site['facility']}  LB  {site['country']}: {kg:,.0f} kgCO2e")

print(f"Total Scope 2 location-based: {total_lb_kgco2e/1000:,.1f} tCO2e")
```

Typical output (numbers illustrative):

```
cph-hq      LB DK: 180,000 kgCO2e     (150 g/kWh grid average)
ber-office  LB DE: 296,000 kgCO2e     (370 g/kWh)
sto-lab     LB SE:   6,400 kgCO2e     (16 g/kWh)
Total Scope 2 location-based: 482.4 tCO2e
```

---

## Step 2: Market-based

For the Copenhagen HQ, pass the contractual instrument. For the others, the pack will route to supplier mix then residual mix.

```python
total_mb_kgco2e = 0.0

# Copenhagen: 100% GO-backed Swedish hydro.
resolved = client.resolve_explain({
    "activity": "purchased_electricity",
    "method_profile": "corporate_scope2_market_based",
    "jurisdiction": "DK",
    "reporting_date": "2026-12-31",
    "facility_id": "cph-hq",
    "extras": {
        "contractual_instrument": {
            "type": "GO",
            "registry": "AIB",
            "vintage_year": 2026,
            "source_country": "SE",
            "technology": "hydropower",
            "mwh": 1200,
            "certificate_ids": ["AIB-SE-2026-HYDRO-0001"]
        }
    }
})
kg_cph = resolved.co2e_per_unit * 1200 * 1000
total_mb_kgco2e += kg_cph
print(f"cph-hq      MB DK (GO-backed hydro):  {kg_cph:,.0f} kgCO2e")

# Berlin: no contract. Pack falls through to supplier mix, then residual mix.
resolved = client.resolve_explain({
    "activity": "purchased_electricity",
    "method_profile": "corporate_scope2_market_based",
    "jurisdiction": "DE",
    "reporting_date": "2026-12-31",
    "facility_id": "ber-office"
})
kg_ber = resolved.co2e_per_unit * 800 * 1000
total_mb_kgco2e += kg_ber
print(f"ber-office  MB DE (residual mix):     {kg_ber:,.0f} kgCO2e")

# Stockholm: no contract. Same logic.
resolved = client.resolve_explain({
    "activity": "purchased_electricity",
    "method_profile": "corporate_scope2_market_based",
    "jurisdiction": "SE",
    "reporting_date": "2026-12-31",
    "facility_id": "sto-lab"
})
kg_sto = resolved.co2e_per_unit * 400 * 1000
total_mb_kgco2e += kg_sto
print(f"sto-lab     MB SE (residual mix):     {kg_sto:,.0f} kgCO2e")

print(f"Total Scope 2 market-based: {total_mb_kgco2e/1000:,.1f} tCO2e")
```

Typical output:

```
cph-hq      MB DK (GO-backed hydro):      24,000 kgCO2e   (20 g/kWh specified residual for hydro)
ber-office  MB DE (residual mix):        360,000 kgCO2e   (450 g/kWh DE residual)
sto-lab     MB SE (residual mix):          8,800 kgCO2e   (22 g/kWh SE residual)
Total Scope 2 market-based: 392.8 tCO2e
```

Market-based is lower here because the Copenhagen PPA backs out of the grid-average.

---

## Attaching certificates (I-REC / GO / RECs)

Each contractual instrument carries:

- `type` — one of `GO`, `I-REC`, `REC`, `PPA`, `gPPA`, `virtual_PPA`.
- `registry` — `AIB` (EU GOs), `I-RECS` (global I-REC), `Green-e` (US RECs), issuing body.
- `vintage_year` — must match or immediately precede the reporting year.
- `source_country` — where the generation physically occurred.
- `technology` — hydropower / onshore_wind / offshore_wind / solar_pv / solar_thermal / biomass / geothermal / nuclear.
- `mwh` — volume the certificate covers.
- `certificate_ids` — audit trail. Store these; the verifier will demand them.

The pack enforces "no double counting" by checking `certificate_ids` against the tenant's prior submissions for the same vintage year.

---

## Why the Copenhagen pack returns non-zero

You might expect "100% GO-backed hydro" to equal 0 gCO2e. It does not, for two reasons:

1. **Residual upstream emissions** — hydropower has ~20 gCO2e/kWh life-cycle (dam construction, methane from reservoirs). The factor table reflects this.
2. **AR6 GWPs applied to small CH4 flux** — reservoirs leak CH4; AR6 gives that more weight than AR4.

If your reporter explicitly claims zero-Scope-2, they should cite GHG Protocol Scope 2 Guidance appendix for what "zero" means (operational, not life-cycle). The pack's numbers are life-cycle.

---

## Dual-report output shape

Publish both totals together:

```json
{
  "reporting_year": 2026,
  "scope_2_location_based_tco2e": 482.4,
  "scope_2_market_based_tco2e":   392.8,
  "methodology": "GHG Protocol Scope 2 Guidance (2015 amendment), dual-report.",
  "edition_id": "2027.Q1-electricity",
  "breakdown": [
    {
      "facility": "cph-hq", "country": "DK", "mwh": 1200,
      "lb_tco2e": 180.0, "mb_tco2e": 24.0,
      "mb_instrument": {"type":"GO","registry":"AIB","vintage":2026,"technology":"hydropower"}
    },
    ...
  ]
}
```

---

## Common pitfalls

- **Vintage mismatch:** A 2024 GO cannot cover 2026 consumption. The pack rejects mismatched vintages at cascade step 4.
- **Double counting:** Using the same `certificate_id` across two reporting entities. The pack detects this and returns `422 license_class_mixed` (used as a generic double-use error).
- **Wrong residual mix:** Pre-2023, residual mixes were published per-country with 18+ month lag. The pack pins to the AIB/Green-e residual-mix release corresponding to your reporting year; it will **not** use a later release retroactively.
- **Mixing scopes:** Heat / steam / cooling have their own profiles (`corporate_scope1` for self-generated, `corporate_scope2_location_based` for purchased). Do not roll them into electricity.

---

## See also

- [Method packs](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md) — step 4 (utility / tariff / grid sub-region).
- [Version pinning](../concepts/version-pinning.md) — residual-mix release vintage pinning.

---

## File citations

| Piece | File |
|---|---|
| Scope 2 location / market packs | `greenlang/factors/method_packs/electricity.py` |
| Profile enum | `greenlang/data/canonical_v2.py::MethodProfile.CORPORATE_SCOPE2_LOCATION`, `.CORPORATE_SCOPE2_MARKET` |
| eGRID parser | `greenlang/factors/ingestion/parsers/` (eGRID sub-region routing) |
| Residual-mix ingestion | `greenlang/factors/connectors/` (Green-e Residual Mix, AIB) |
