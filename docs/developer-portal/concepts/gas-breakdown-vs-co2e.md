# Gas Breakdown vs CO2e-Only

**CTO non-negotiable #1:** The GreenLang Factors catalog never stores a factor as CO2e-only. We keep the gas vector (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3, biogenic_CO2) and compute CO2e on demand using the caller's chosen GWP set.

If you have ever had to restate a year of inventory because IPCC shifted GWPs (AR4 → AR5 → AR6), you know why.

**Source:** `greenlang/factors/ontology/gwp_sets.py`

---

## Why CO2e-only is unsafe

CO2e is a function of **(gas vector, GWP set)**. A factor published as "0.35 kgCO2e / kWh" is useless without knowing:

1. Which gases were present in the original measurement.
2. Which GWP set was used to roll them up.

When AR6 raised methane's 100-year GWP from 28 (AR5) to 27.9 (AR6, without feedbacks) and dropped N2O's slightly, every single CO2e number depending on a CH4 or N2O component changed — but nobody can re-compute them if only the CO2e was stored.

By storing gas breakdowns, you can:

- Re-compute CO2e under any GWP set (AR4, AR5, AR6 — 20-year or 100-year horizons).
- Isolate fugitive HFC emissions for F-gas accounting.
- Report biogenic CO2 separately from fossil (required under ISO 14067, CORSIA, GHG PS).
- Restate a prior inventory cleanly when your regulator updates its recommended GWP.

---

## What the catalog stores

Each factor carries a `GasBreakdown` (see `greenlang/factors/resolution/result.py`):

```json
"gas_breakdown": {
  "co2":        10.15,
  "ch4":         0.04,
  "n2o":         0.02,
  "biogenic_co2": null,
  "hfcs":        null,
  "pfcs":        null,
  "sf6":         null,
  "nf3":         null
}
```

Units are `kg / <activity unit>`. `null` means "not applicable / not present" (an electricity factor does not emit SF6 directly). Zero means "measured, confirmed zero."

Fugitive refrigerant factors use the HFC / PFC / SF6 / NF3 slots. Biogenic CO2 stays in `biogenic_co2` so it never rolls up into fossil CO2e unintentionally.

---

## Supported GWP sets

From `GWPSet` enum in `gwp_sets.py`:

| Enum value | Wire string | Horizon | Source |
|---|---|---|---|
| `IPCC_AR4_100` | `IPCC_AR4_100` | 100-year | IPCC AR4 (2007), WG1 Ch.2 Table 2.14 |
| `IPCC_AR4_20` | `IPCC_AR4_20` | 20-year | IPCC AR4 (2007) |
| `IPCC_AR5_100` | `IPCC_AR5_100` | 100-year | IPCC AR5 (2013), WG1 Ch.8 Table 8.7 (no climate-carbon feedback, per UNFCCC guidance) |
| `IPCC_AR5_20` | `IPCC_AR5_20` | 20-year | IPCC AR5 (2013) |
| `IPCC_AR6_100` | `IPCC_AR6_100` | 100-year | IPCC AR6 (2021), WG1 Ch.7 Table 7.15 (without feedbacks) |
| `IPCC_AR6_20` | `IPCC_AR6_20` | 20-year | IPCC AR6 (2021) |

**Default:** `IPCC_AR6_100` (`DEFAULT_GWP_SET`). This is what UNFCCC and the EU CSRD guidance currently recommend.

---

## Selecting a GWP set

### Server-side (resolve under a specific set)

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "activity": "natural gas combustion stationary",
    "method_profile": "corporate_scope1",
    "jurisdiction": "US",
    "reporting_date": "2026-06-01",
    "gwp_set": "IPCC_AR6_100"
  }'
```

The response carries the CO2e computed under the requested set:

```json
{
  "co2e_per_unit": 53.07,
  "unit": "kg/mmBtu",
  "gwp_set": "IPCC_AR6_100",
  "ch4_gwp": 27.9,
  "n2o_gwp": 273,
  "gas_breakdown": { "co2": 53.02, "ch4": 0.001, "n2o": 0.0001 }
}
```

Check `co2e_per_unit` vs `gas_breakdown`: `53.02 + 0.001*27.9 + 0.0001*273 = 53.07`. The math is deterministic and done via `convert_co2e()` in `gwp_sets.py`.

### Client-side re-computation

Sometimes you have a bunch of pre-resolved factors and need to re-roll them under a new GWP set (e.g. for a retrospective restatement). Do it locally — no need to re-query the API:

```python
from greenlang.factors.ontology.gwp_sets import convert_co2e, GWPSet

breakdown = {"CO2": 53.02, "CH4": 0.001, "N2O": 0.0001}

co2e_ar5 = convert_co2e(breakdown, to_set=GWPSet.IPCC_AR5_100)
co2e_ar6 = convert_co2e(breakdown, to_set=GWPSet.IPCC_AR6_100)

print(f"AR5 100y: {co2e_ar5} kg")
print(f"AR6 100y: {co2e_ar6} kg")
```

This is faster, uses no API quota, and is reproducible against committed IPCC tables.

---

## When to use AR6 vs AR5 vs AR4

- **Almost everything new:** AR6 100-year. This is what IPCC AR6 recommends and what UNFCCC's NDC guidance and EU CSRD expect.
- **Legacy inventories:** Check your standard. Many pre-2021 inventories use AR4 or AR5 100-year; restating them to AR6 is a policy decision, not automatic.
- **F-gas-heavy activities (refrigeration, semiconductors):** Consider presenting both 100-year and 20-year under AR6 so short-lived-climate-pollutant risk is visible.
- **CORSIA aviation:** the program pins to a specific IPCC horizon; check your scheme's current version.
- **CBAM:** Uses AR6 100-year per Implementing Act 2023/1773.

---

## The "never CO2e only" rule in the API

- Every `ResolvedFactor` response carries a `gas_breakdown` block, even when all gases other than CO2 are `null`.
- Edit / override endpoints refuse a payload that provides only `co2e_per_unit` without a breakdown.
- The audit bundle (`/api/v1/factors/{factor_id}/audit-bundle`) includes the gas breakdown and the GWP values used at render time.
- The signed-receipt payload hash covers both the breakdown and the CO2e, so tampering with one and not the other is detectable.

---

## See also

- [Refrigerant GWP selection cookbook](../cookbook/refrigerant-gwp-selection.md)
- [Resolution cascade](./resolution-cascade.md)
- [Quality scores](./quality-scores.md)
- [Method packs](./method-packs.md) — some profiles forbid certain GWP sets.

---

## File citations

| Piece | File |
|---|---|
| GWP sets registry (AR4 / AR5 / AR6 tables) | `greenlang/factors/ontology/gwp_sets.py` |
| `GWPSet` enum | `greenlang/factors/ontology/gwp_sets.py` (line 52) |
| `DEFAULT_GWP_SET` = AR6 100 | `greenlang/factors/ontology/gwp_sets.py` (line 68) |
| `convert_co2e()` | `greenlang/factors/ontology/gwp_sets.py` |
| `GasBreakdown` on resolved factors | `greenlang/factors/resolution/result.py` |
| Gas fields on records | `greenlang/data/emission_factor_record.py` (`gwp_100yr`, `gwp_20yr`) |
