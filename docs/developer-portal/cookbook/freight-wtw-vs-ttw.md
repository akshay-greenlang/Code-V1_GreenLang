# Cookbook: Freight ISO 14083 / GLEC — WTW vs TTW

Freight emissions under ISO 14083 (GLEC-aligned) require you to choose a boundary:

- **Tank-to-Wheel (TTW)** — combustion emissions only. Fuel exits the tank, enters the atmosphere.
- **Well-to-Wheel (WTW)** — TTW plus upstream fuel production (extraction, refining, transport of the fuel). This is the default for GHG Protocol Scope 3 Category 4 / 9 and for regulatory submissions (CSRD, CDP).

**Method profile:** `freight_iso_14083` (with `extras.boundary = "WTW" | "TTW"`).
**Implementation:** `greenlang/factors/method_packs/freight.py`.

---

## Scenario

You ship a **26-tonne payload of finished-goods pallets** from Rotterdam to Milan:

- Leg 1: Road haul Rotterdam → Frankfurt (450 km), 40t semi-trailer diesel, 70% utilisation.
- Leg 2: Rail intermodal Frankfurt → Milan (700 km), electric traction (average EU grid mix).
- Shipping date: 2026-06-01.

You want both WTW and TTW numbers for GHG Protocol Scope 3 Cat 4.

---

## Step 1: Leg 1, road haul

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-freight",
)

def resolve_leg(leg):
    return client.resolve_explain({
        "activity": "freight_transport",
        "method_profile": "freight_iso_14083",
        "jurisdiction": leg["jurisdiction"],
        "reporting_date": "2026-06-01",
        "extras": {
            "mode": leg["mode"],
            "boundary": leg["boundary"],
            "payload_tonnes": leg["payload_t"],
            "distance_km": leg["distance_km"],
            "vehicle_class": leg.get("vehicle_class"),
            "utilisation_pct": leg.get("utilisation_pct"),
            "traction": leg.get("traction"),
        }
    })

leg1_ttw = resolve_leg({
    "mode": "road",
    "jurisdiction": "DE",
    "boundary": "TTW",
    "payload_t": 26,
    "distance_km": 450,
    "vehicle_class": "hgv_40t_articulated_diesel_euro6",
    "utilisation_pct": 70,
})
leg1_wtw = resolve_leg({**leg1_ttw.request.dict(), "boundary": "WTW"})
# or just resolve_leg(...) a second time.

# Factor unit is kg_CO2e per tonne-km.
tkm = 26 * 450
leg1_ttw_kg = leg1_ttw.co2e_per_unit * tkm
leg1_wtw_kg = leg1_wtw.co2e_per_unit * tkm

print(f"Leg 1 TTW: {leg1_ttw_kg:,.1f} kgCO2e  ({leg1_ttw.co2e_per_unit*1000:.1f} g/tkm)")
print(f"Leg 1 WTW: {leg1_wtw_kg:,.1f} kgCO2e  ({leg1_wtw.co2e_per_unit*1000:.1f} g/tkm)")
```

Expected output (illustrative):

```
Leg 1 TTW:   780.3 kgCO2e   (66.7 g/tkm)
Leg 1 WTW:   932.6 kgCO2e   (79.7 g/tkm)
```

The 20% WTW-vs-TTW uplift on road diesel reflects upstream refining and transport of crude oil.

---

## Step 2: Leg 2, rail intermodal

```python
leg2_ttw = resolve_leg({
    "mode": "rail",
    "jurisdiction": "EU",
    "boundary": "TTW",
    "payload_t": 26,
    "distance_km": 700,
    "traction": "electric",
})
leg2_wtw = resolve_leg({**leg2_ttw.request.dict(), "boundary": "WTW"})

tkm2 = 26 * 700
leg2_ttw_kg = leg2_ttw.co2e_per_unit * tkm2
leg2_wtw_kg = leg2_wtw.co2e_per_unit * tkm2
```

Rail electric TTW is effectively zero (the train burns no fuel onboard). WTW picks up the upstream grid emissions for the traction electricity.

---

## Step 3: Total

```python
total_ttw = leg1_ttw_kg + leg2_ttw_kg
total_wtw = leg1_wtw_kg + leg2_wtw_kg

print(f"Total TTW: {total_ttw:,.1f} kgCO2e")
print(f"Total WTW: {total_wtw:,.1f} kgCO2e")
```

For regulatory reporting (GHG Protocol Scope 3 Cat 4, EU CSRD ESRS E1, CDP C6), use **WTW**. TTW is informational.

---

## The `extras` fields that matter

| Field | Meaning | Required |
|---|---|---|
| `mode` | `road` / `rail` / `sea` / `air` / `inland_waterway` / `pipeline` | yes |
| `boundary` | `WTW` or `TTW` | yes |
| `payload_tonnes` | Mass of goods being moved. | yes |
| `distance_km` | Great-circle for air/sea; routed distance for road/rail. | yes |
| `vehicle_class` | Road: `hgv_7_5t_rigid_diesel_euro5`, `hgv_40t_articulated_diesel_euro6`, etc. Sea: `container_ship_<teu>_<fuel>`. | mode-dependent |
| `utilisation_pct` | 0-100; defaults per mode (road 85%, rail 65%, sea 70%, air 65%). | no |
| `traction` | Rail: `diesel` / `electric` / `hybrid`. | rail only |
| `fuel_type` | Override default (e.g. `HVO` for HVO-running road fleet, `LNG` for marine). | no |
| `return_journey` | `true` if the return leg was empty; the pack doubles the effective tkm. | no |

`utilisation_pct` is the single biggest uncertainty lever — a 40%-utilised truck emits per tkm twice as much as an 80%-utilised one. Supply measured values whenever possible.

---

## Fuel type override

Default road factors assume EU-average diesel. Override when you run biofuels or e-fuels:

```python
resolve_leg({
    "mode": "road",
    "jurisdiction": "DE",
    "boundary": "WTW",
    "payload_t": 26,
    "distance_km": 450,
    "vehicle_class": "hgv_40t_articulated_diesel_euro6",
    "fuel_type": "HVO100",       # 100% hydrotreated vegetable oil
    "utilisation_pct": 70,
})
```

The pack will swap the upstream fuel factor while keeping TTW combustion emissions (HVO burns nearly as cleanly as diesel; the WTW delta is the big win). `fuel_type` values live in `greenlang/factors/ontology/chemistry.py` and `freight.py`.

---

## Refrigerated transport

Reefer containers / reefer trucks add a second emission stream (mechanical cooling, often an APU). Pass the reefer load:

```python
"extras": {
  "mode": "road",
  "boundary": "WTW",
  "payload_tonnes": 20,
  "distance_km": 600,
  "vehicle_class": "hgv_40t_articulated_diesel_euro6",
  "reefer": {
    "refrigeration_fuel": "diesel",
    "refrigeration_load_fraction": 0.35   # 35% of vehicle fuel goes to cooling
  }
}
```

The pack returns two factors (transport + refrigeration) and a summed `co2e_per_unit`.

---

## Common pitfalls

- **Boundary silently wrong:** Omitting `boundary` defaults to WTW. If you need TTW explicitly, set it.
- **Distance not routed:** For road/rail, use actual routed distance (Google Maps, road network). Great-circle underestimates by 15-25% on typical EU lanes.
- **Empty-leg undercount:** If you run dedicated rather than hired-capacity routes, set `return_journey: true`. Hired capacity (3PL, rail freight company) already bakes empty-leg averages into its utilisation defaults.
- **Mixing modes under one call:** Do not try to compute a multi-modal shipment in one resolution. One call per leg, then sum.

---

## See also

- [Method packs — freight_iso_14083](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md)
- [Gas breakdown vs CO2e](../concepts/gas-breakdown-vs-co2e.md)

---

## File citations

| Piece | File |
|---|---|
| Freight method pack (WTW/TTW split) | `greenlang/factors/method_packs/freight.py` |
| Profile enum | `greenlang/data/canonical_v2.py::MethodProfile.FREIGHT_ISO_14083` |
| Fuel type registry | `greenlang/factors/ontology/chemistry.py` |
| GLEC / Smart Freight Centre source integration | `greenlang/factors/data/source_registry.yaml` |
