# -*- coding: utf-8 -*-
"""Activity-unit ontology subset (S2). Full conversion engine hooks EmissionFactorDatabase."""

from __future__ import annotations

from typing import Optional

KNOWN_DENOMINATORS = frozenset(
    {
        "kwh",
        "mwh",
        "gwh",
        "therms",
        "mmbtu",
        "gj",
        "mj",
        "liters",
        "litres",
        "gallons",
        "kg",
        "tonnes",
        "t",
        "lb",
        "miles",
        "km",
        "passenger_km",
        "tonne_km",
        "usd",
        "eur",
        "m3",
        "scf",
    }
)


def is_known_activity_unit(unit: str) -> bool:
    u = (unit or "").strip().lower().replace(" ", "_")
    return u in KNOWN_DENOMINATORS


def suggest_si_base(unit: str) -> Optional[str]:
    u = (unit or "").strip().lower()
    if u in ("kwh", "mwh", "gwh"):
        return "J"
    if u in ("kg", "t", "tonnes", "lb"):
        return "kg"
    return None


# Energy: factors stored relative to kWh baseline for consistent comparisons (S2).
_KWH_PER_UNIT = {
    "kwh": 1.0,
    "mwh": 1000.0,
    "gwh": 1_000_000.0,
    "gj": 277.778,  # 1 GJ ≈ 277.778 kWh
    "mj": 0.277778,
    "therms": 29.3001,  # US therm → kWh
    "mmbtu": 293.071,  # million BTU → kWh
}


def convert_energy_to_kwh(amount: float, unit: str) -> float:
    """Convert *amount* in *unit* (energy denominator) to kilowatt-hours."""
    u = (unit or "").strip().lower().replace(" ", "_")
    if u not in _KWH_PER_UNIT:
        raise ValueError(f"unsupported energy unit for kWh conversion: {unit!r}")
    return float(amount) * _KWH_PER_UNIT[u]


def convert_energy(amount: float, from_unit: str, to_unit: str) -> float:
    """Convert *amount* between two energy units in ``_KWH_PER_UNIT``."""
    kwh = convert_energy_to_kwh(amount, from_unit)
    to = (to_unit or "").strip().lower().replace(" ", "_")
    if to not in _KWH_PER_UNIT:
        raise ValueError(f"unsupported target energy unit: {to_unit!r}")
    return kwh / _KWH_PER_UNIT[to]
