# Cookbook: Refrigerant GWP Selection

Refrigerants, solvents, SF6 insulation, and semiconductor-process gases are all high-GWP non-CO2 pollutants. Their contribution to your Scope 1 inventory depends entirely on **which GWP set you apply**. AR4 methane was 25; AR5 was 28; AR6 is 27.9 (without feedbacks) or 29.8 (with feedbacks). HFC-134a moved from 1430 (AR4) to 1300 (AR5) to 1530 (AR6).

Pick the wrong set and your fugitive-emissions number is off by 5-15%.

**GWP tables:** `greenlang/factors/ontology/gwp_sets.py`.
**Principle:** the catalog stores the gas vector; CO2e is computed on demand (see [gas-breakdown-vs-co2e](../concepts/gas-breakdown-vs-co2e.md)).

---

## Scenario

A data centre in Frankfurt operates:

- 12 water-cooled chillers, refrigerant **R-134a**, 450 kg total charge, 3% annual leakage.
- 6 refrigerant racks, **R-410A**, 120 kg total charge, 2% annual leakage.
- 2 HV switchgears, **SF6**, 18 kg total charge, 0.5% annual leakage.

Reporting year 2026. Compute under **IPCC AR6 100-year** (regulator default).

---

## Step 1: Understand what the catalog stores

Query the raw factor for R-134a leakage:

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/search?q=HFC-134a+leakage&scope=1" \
  | jq '.factors[0]'
```

You will see a gas breakdown with just the HFC component and a null CO2/CH4/N2O. That is correct — the refrigerant IS the greenhouse gas.

```json
{
  "factor_id": "EF:GLOBAL:refrigerant_leakage:HFC-134a:v1",
  "fuel_type": "HFC-134a",
  "unit": "kg_refrigerant",
  "gas_breakdown": {"co2": null, "ch4": null, "n2o": null, "hfcs": 1.0},
  "gwp_set": "IPCC_AR6_100",
  "ch4_gwp": 27.9,
  "n2o_gwp": 273,
  "co2e_per_unit": 1530.0
}
```

`hfcs: 1.0` means "each kg of refrigerant leakage contributes 1.0 kg of HFC-134a to the atmosphere." CO2e is `1.0 * GWP(HFC-134a, AR6_100) = 1530`.

---

## Step 2: Resolve under AR6 100-year (default)

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-electricity",
)

leaks = [
    {"gas": "HFC-134a", "leakage_kg": 450 * 0.03},
    {"gas": "R-410A",   "leakage_kg": 120 * 0.02},
    {"gas": "SF6",      "leakage_kg":  18 * 0.005},
]

total_ar6_kgco2e = 0.0
for row in leaks:
    resolved = client.resolve_explain({
        "activity": "refrigerant_leakage",
        "method_profile": "corporate_scope1",
        "jurisdiction": "DE",
        "reporting_date": "2026-12-31",
        "extras": {"gas_code": row["gas"]},
        "gwp_set": "IPCC_AR6_100"
    })
    kg = resolved.co2e_per_unit * row["leakage_kg"]
    total_ar6_kgco2e += kg
    print(f"{row['gas']:10s} {row['leakage_kg']:6.2f} kg  x  GWP {resolved.co2e_per_unit:6.0f}  =  {kg:,.0f} kgCO2e")

print(f"Total fugitive F-gas (AR6 100y): {total_ar6_kgco2e/1000:,.1f} tCO2e")
```

Expected:

```
HFC-134a    13.50 kg  x  GWP  1530  =   20,655 kgCO2e
R-410A       2.40 kg  x  GWP  2088  =    5,011 kgCO2e    (R-410A is a blend: 50% HFC-32, 50% HFC-125; blended GWP)
SF6          0.09 kg  x  GWP 25200  =    2,268 kgCO2e
Total fugitive F-gas (AR6 100y): 27.9 tCO2e
```

---

## Step 3: Re-compute under AR5 for historical comparability

Your prior-year inventory was published under AR5. To compute a like-for-like prior-year restatement:

```python
from greenlang.factors.ontology.gwp_sets import GWPSet, convert_co2e

# Reuse the gas breakdowns from step 2 (access resolved.gas_breakdown).
for row in leaks:
    resolved = client.resolve_explain({
        "activity": "refrigerant_leakage",
        "method_profile": "corporate_scope1",
        "jurisdiction": "DE",
        "reporting_date": "2026-12-31",
        "extras": {"gas_code": row["gas"]},
        "gwp_set": "IPCC_AR5_100"
    })
    kg_ar5 = resolved.co2e_per_unit * row["leakage_kg"]
    print(f"{row['gas']:10s} AR5: {kg_ar5:,.0f} kgCO2e")
```

Because the catalog stores the gas vector (not CO2e), the only thing that changes between runs is the GWP set applied. No repository look-up, no new factor — same gas breakdown, different multiplier.

---

## Step 4: AR6 20-year for short-lived-climate-pollutant reporting

Some investor-driven disclosures (CDP climate transition plans, TCFD scenario analysis) ask for a 20-year view of short-lived climate pollutants. Refrigerants are the largest single component for many tech operations.

```python
resolved = client.resolve_explain({
    "activity": "refrigerant_leakage",
    "method_profile": "corporate_scope1",
    "jurisdiction": "DE",
    "reporting_date": "2026-12-31",
    "extras": {"gas_code": "HFC-134a"},
    "gwp_set": "IPCC_AR6_20"
})
# HFC-134a 20-year GWP is about 4144 (vs 1530 for 100-year).
print(resolved.co2e_per_unit)
```

Report both windows side by side so the reader sees the near-term climate urgency.

---

## Blended refrigerants

Many modern refrigerants are blends (R-410A, R-404A, R-407C, R-448A, R-454B, R-513A). The catalog resolves them by:

1. Looking up the blend's composition in `chemistry.py` (mass fractions of pure refrigerants).
2. Computing the blended GWP as the weighted average of component GWPs under the selected GWP set.
3. Returning a gas breakdown with the blend components separated.

You can inspect the breakdown:

```python
resolved = client.resolve_explain({
    "activity": "refrigerant_leakage",
    "method_profile": "corporate_scope1",
    "jurisdiction": "DE",
    "extras": {"gas_code": "R-410A"}
})

print(resolved.gas_breakdown)
# {"hfc_32": 0.5, "hfc_125": 0.5, "hfcs": 1.0}
```

So if your regulator demands component-level reporting, you have it.

---

## PFCs and NF3

Semiconductor fabs emit PFC (CF4, C2F6, C3F8) and NF3. Same mechanism:

```python
resolved = client.resolve_explain({
    "activity": "semiconductor_process_emissions",
    "method_profile": "corporate_scope1",
    "jurisdiction": "TW",
    "extras": {"gas_code": "NF3", "process_step": "chamber_cleaning"}
})
# NF3 AR6 100y GWP = 17400.
```

---

## Common pitfalls

- **CO2e-only source records:** Older datasets sometimes publish refrigerants as CO2e with GWP unspecified. The catalog rejects ingesting those (CTO non-negotiable #1). If you import legacy data, you must supply the GWP set they were computed under or re-baseline.
- **Wrong GWP horizon:** 20-year and 100-year GWPs differ by a factor of 2-5 for HFCs. Pick one and stick with it; annotate prominently.
- **Mixing feedbacks:** IPCC AR5 published two tables (with / without climate-carbon feedbacks). UNFCCC reporting uses "without." The catalog exclusively stores "without." Do not cross-compare to external "with-feedback" tables.
- **Charge-based vs leakage-based:** Regulators want leakage, not charge. Multiply charge by leakage rate; do not report the installed charge.

---

## See also

- [Gas breakdown vs CO2e](../concepts/gas-breakdown-vs-co2e.md)
- [Method packs](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md)

---

## File citations

| Piece | File |
|---|---|
| GWP sets (AR4 / AR5 / AR6, 20y / 100y) | `greenlang/factors/ontology/gwp_sets.py` |
| Default GWP set (`DEFAULT_GWP_SET = IPCC_AR6_100`) | `greenlang/factors/ontology/gwp_sets.py` (line 68) |
| Blend composition registry | `greenlang/factors/ontology/chemistry.py` |
| `convert_co2e()` | `greenlang/factors/ontology/gwp_sets.py` |
| Gas vector storage on records | `greenlang/data/emission_factor_record.py::GWP100yr` |
