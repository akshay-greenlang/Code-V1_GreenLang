# Method Pack — Freight

Implements **ISO 14083:2023 Greenhouse gas emissions quantification and reporting** and the **Smart Freight Centre GLEC Framework v3.0**. Two variants cover the two energy boundaries.

| Profile | Boundary | What's included |
|---|---|---|
| `freight_iso14083_glec_wtw` | WTW (well-to-wheel) | Upstream fuel production + combustion + vehicle operation |
| `freight_iso14083_glec_ttw` | TTW (tank-to-wheel) | Combustion + vehicle operation only |

---

## Standards alignment

- **ISO 14083:2023** — *Greenhouse gases — Quantification and reporting of greenhouse gas emissions arising from transport chain operations*. [Link](https://www.iso.org/standard/78864.html). Clauses 5 (boundaries), 6 (allocation), 7 (data quality), 9 (GHG intensity factors), 10 (reporting).
- **SFC GLEC Framework v3.0** — Smart Freight Centre, 2023. Aligned with ISO 14083 by design. [Link](https://www.smartfreightcentre.org/en/our-programs/global-logistics-emissions-council/).
- **EN 16258:2012** — Predecessor European standard (for historical comparability only).

---

## Covered modes

- **Road** — HGV rigid / articulated / van, multiple tonnage classes (3.5t, 7.5t, 12t, 17t, 26t, 40t). Empty running and utilization defaults per GLEC.
- **Rail** — Electric vs diesel traction; electricity mix tied to grid factor in operating jurisdiction.
- **Sea** — Container (TEU-km), bulk, tanker, ro-ro. Vessel size classes per GLEC.
- **Air** — Short-haul, medium-haul, long-haul; passenger-km vs freight-km split.
- **Inland waterway** — Barge self-propelled / pushed convoy.

---

## Parameters

Every freight factor carries the full GLEC parameter set:

- `mode` — road / rail / sea / air / inland_waterway
- `vehicle_class` — GLEC code (e.g., `HGV_rigid_17t`)
- `payload_basis` — `t-km`, `v-km`, `TEU-km`, `pax-km`
- `distance_basis` — `great_circle`, `route`, `route_with_empty`
- `empty_running_assumption` — 0..1 (e.g., 0.25 for GLEC rigid default)
- `utilization_rate` — 0..1 (e.g., 0.70 EU27 average)
- `refrigerated` — boolean
- `energy_basis` — `WTW`, `TTW`, `TTT`, `WTT`

---

## Selection rules

- `selection.allowed_families`: `["transport"]`.
- `selection.allowed_formula_types`: `["transport_chain", "direct_factor"]`.
- `selection.jurisdiction_hierarchy`: `["country", "region", "global"]`. Most freight factors are global/regional; jurisdiction usually refines the grid factor for rail electrification.
- `selection.priority_tiers`: `["carrier_specific", "lane_specific", "mode_default", "global_default"]`.

---

## Boundary

- `boundary.system_boundary`: `gate_to_gate` (single transport leg) or `cradle_to_grave` (multi-leg transport chain, aggregated via ISO 14083 chain calculation).
- `boundary.allowed_boundaries`: `["WTW"]` or `["TTW"]` depending on profile variant.

---

## Chain calculation (ISO 14083 Clause 9)

When a request describes a multi-leg journey, the resolver decomposes the journey into legs, resolves a factor per leg, then aggregates via ISO 14083 Clause 9 formulas. Each leg carries its own `assumptions[]` in the explain output. The aggregate `co2e_kg` is the sum of per-leg contributions.

---

## Market instruments

`market_instruments.treatment = not_applicable`. Biofuel blends and sustainable aviation fuel (SAF) are handled through separate fuel factors referenced by the transport chain — not through market certificates.

---

## Licensing

GLEC factor values are `licensed_embedded` (GLEC / SFC Terms). At v1 launch the pack ships with:

- Open sources: **DEFRA freight conversion factors** (UK), **EcoTransIT open subset**, **IPCC AFOLU Vol 2** underlying fuel combustion.
- BYO-credentials: full **GLEC Framework v3.0** via Smart Freight Centre subscription.

Contract to upgrade GLEC from BYO to `licensed_embedded` is in outreach; see [`docs/legal/source_contracts_outreach.md`](../../legal/source_contracts_outreach.md) Part 2.D.

---

## Related

- [`/resolve`](../api-reference/resolve.md), [`concepts/method_pack.md`](../concepts/method_pack.md).
- [`licensing.md`](../licensing.md) for BYO-credentials details.
