# -*- coding: utf-8 -*-
"""
Density converter — mass ↔ volume for fuels + common materials (Phase F5).

Density values are pressure- and temperature-sensitive.  This module ships
a compact reference table with per-fuel temperature/pressure defaults
that match the assumptions used by the majority of GHG inventories:

- **Liquid fuels** — 15 °C / 1 atm (IPCC 2006 Vol. 2 Ch. 1, EPA 40 CFR §98)
- **Gaseous fuels** — 15 °C / 1 atm (dry gas)
- **Materials** — 20 °C / 1 atm (standard LCA reference)

Callers that need temperature-corrected densities (e.g., road-freight
fuel in tropical vs arctic regions) pass ``adjust_temperature_c`` and we
apply a linear thermal-expansion correction using each fuel's
coefficient (β, per °C).

Non-negotiable impact: feeds the Scope Engine's combustion formula
(``mass × LHV × oxidation × factor``) so downstream emissions are never
computed on a mismatched unit basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DensityRecord:
    """Per-material density reference + temperature correction."""

    material: str
    density_kg_per_l: float                # kg/L at reference conditions
    reference_temperature_c: float         # usually 15 °C for fuels
    reference_pressure_bar: float          # usually 1.01325
    thermal_expansion_coef_per_c: float    # β, typical 0.0007 for liquids
    source: str

    def density_at(
        self,
        *,
        temperature_c: Optional[float] = None,
    ) -> float:
        """Return density (kg/L) at the given temperature."""
        if temperature_c is None:
            return self.density_kg_per_l
        delta_t = temperature_c - self.reference_temperature_c
        return self.density_kg_per_l * (1.0 - self.thermal_expansion_coef_per_c * delta_t)


# ---------------------------------------------------------------------------
# Reference table
# ---------------------------------------------------------------------------

# Key = canonical fuel / material identifier (matches greenlang.factors.mapping.fuels).
# Density values follow IPCC 2006 Vol.2 Ch.1 Table 1.2 for fuels and
# typical LCA references for materials.  Thermal expansion coefficients
# from API MPMS Ch.11 (fuels) and standard physics references.

_DEFAULT_LIQUID_BETA = 0.0007          # per °C, typical petroleum liquids
_DEFAULT_SOLID_BETA = 0.0              # assumed negligible for solids


DENSITY_TABLE: Dict[str, DensityRecord] = {
    # --- Liquid fuels ---
    "diesel": DensityRecord(
        material="diesel",
        density_kg_per_l=0.8400,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_LIQUID_BETA,
        source="IPCC 2006 Vol.2 Table 1.2 / API MPMS",
    ),
    "gasoline": DensityRecord(
        material="gasoline",
        density_kg_per_l=0.7450,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00094,
        source="IPCC 2006 / API MPMS",
    ),
    "kerosene": DensityRecord(
        material="kerosene",
        density_kg_per_l=0.8100,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00079,
        source="IPCC 2006",
    ),
    "jet_fuel": DensityRecord(
        material="jet_fuel",
        density_kg_per_l=0.8000,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00083,
        source="IPCC 2006 / ASTM D1655 Jet A-1",
    ),
    "avgas": DensityRecord(
        material="avgas",
        density_kg_per_l=0.7100,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00095,
        source="IPCC 2006 / ASTM D910 Avgas 100LL",
    ),
    "fuel_oil": DensityRecord(
        material="fuel_oil",
        density_kg_per_l=0.9600,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00064,
        source="IPCC 2006 / ISO 8217 RMG-380",
    ),
    "biodiesel": DensityRecord(
        material="biodiesel",
        density_kg_per_l=0.8800,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00080,
        source="EN 14214",
    ),
    "ethanol": DensityRecord(
        material="ethanol",
        density_kg_per_l=0.7890,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00108,
        source="CRC Handbook",
    ),
    "propane": DensityRecord(
        material="propane",
        density_kg_per_l=0.5100,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,      # liquid-state propane
        thermal_expansion_coef_per_c=0.00160,
        source="IPCC 2006 / LPG industry standard",
    ),
    "butane": DensityRecord(
        material="butane",
        density_kg_per_l=0.5840,
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00140,
        source="CRC Handbook",
    ),

    # --- Gaseous fuels (kg/L at 15 °C, 1 atm) ---
    "natural_gas": DensityRecord(
        material="natural_gas",
        density_kg_per_l=0.000712,           # 712 g/m3
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00366,  # ideal-gas ≈ 1/288.15 at 15°C
        source="IPCC 2006 / ISO 6976",
    ),
    "hydrogen": DensityRecord(
        material="hydrogen",
        density_kg_per_l=0.0000851,          # 85.1 g/m3 at 15 °C, 1 atm
        reference_temperature_c=15.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00366,
        source="CRC Handbook",
    ),

    # --- Solid fuels ---
    "coal": DensityRecord(
        material="coal",
        density_kg_per_l=1.3500,              # bulk density of bituminous
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="IEA Coal Information",
    ),
    "biomass": DensityRecord(
        material="biomass",
        density_kg_per_l=0.6500,              # typical wood pellets
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="EN ISO 17225-2 wood pellets",
    ),

    # --- Materials (Scope 3 Cat 1 / Product carbon) ---
    "steel_hot_rolled_coil": DensityRecord(
        material="steel_hot_rolled_coil",
        density_kg_per_l=7.85,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="ASTM A370",
    ),
    "aluminium_ingot_primary": DensityRecord(
        material="aluminium_ingot_primary",
        density_kg_per_l=2.70,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="CRC Handbook",
    ),
    "cement_portland": DensityRecord(
        material="cement_portland",
        density_kg_per_l=3.15,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="ACI 211.1",
    ),
    "concrete_ready_mix": DensityRecord(
        material="concrete_ready_mix",
        density_kg_per_l=2.40,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=_DEFAULT_SOLID_BETA,
        source="ACI 318",
    ),
    "water": DensityRecord(
        material="water",
        density_kg_per_l=0.99823,
        reference_temperature_c=20.0,
        reference_pressure_bar=1.01325,
        thermal_expansion_coef_per_c=0.00021,
        source="CRC Handbook (4 °C = 1.0000)",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DensityLookupError(KeyError):
    """Raised when the requested material isn't in the density table."""


def get_density(material: str) -> DensityRecord:
    key = material.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in DENSITY_TABLE:
        raise DensityLookupError(
            "No density reference for material %r; available: %s"
            % (material, sorted(DENSITY_TABLE))
        )
    return DENSITY_TABLE[key]


def mass_to_volume_l(
    *, material: str, mass_kg: float, temperature_c: Optional[float] = None
) -> float:
    """Convert a mass (kg) to a volume (litres) at the given temperature."""
    rec = get_density(material)
    density = rec.density_at(temperature_c=temperature_c)
    if density <= 0:
        raise ValueError("computed density is non-positive; check inputs")
    return mass_kg / density


def volume_l_to_mass(
    *, material: str, volume_l: float, temperature_c: Optional[float] = None
) -> float:
    """Convert a volume (litres) to mass (kg) at the given temperature."""
    rec = get_density(material)
    density = rec.density_at(temperature_c=temperature_c)
    return volume_l * density


def convert(
    *,
    material: str,
    value: float,
    from_unit: str,
    to_unit: str,
    temperature_c: Optional[float] = None,
) -> float:
    """High-level converter — accepts {kg, t, L, m3} on either side."""
    canonical_mass = _to_kg(value, from_unit)
    if canonical_mass is not None and _is_mass_unit(to_unit):
        return _from_kg(canonical_mass, to_unit)
    canonical_volume = _to_l(value, from_unit)
    if canonical_volume is not None and _is_volume_unit(to_unit):
        return _from_l(canonical_volume, to_unit)

    # mass → volume
    if canonical_mass is not None and _is_volume_unit(to_unit):
        litres = mass_to_volume_l(
            material=material, mass_kg=canonical_mass, temperature_c=temperature_c
        )
        return _from_l(litres, to_unit)
    # volume → mass
    if canonical_volume is not None and _is_mass_unit(to_unit):
        kg = volume_l_to_mass(
            material=material, volume_l=canonical_volume, temperature_c=temperature_c
        )
        return _from_kg(kg, to_unit)

    raise ValueError(
        "unsupported conversion %s → %s for material %r" % (from_unit, to_unit, material)
    )


_MASS_FACTORS = {"kg": 1.0, "g": 0.001, "t": 1000.0, "tonne": 1000.0, "tonnes": 1000.0, "lb": 0.45359237}
_VOLUME_FACTORS = {"l": 1.0, "liter": 1.0, "liters": 1.0, "ml": 0.001, "m3": 1000.0, "gallon": 3.78541, "gal": 3.78541}


def _is_mass_unit(unit: str) -> bool:
    return unit.lower() in _MASS_FACTORS


def _is_volume_unit(unit: str) -> bool:
    return unit.lower() in _VOLUME_FACTORS


def _to_kg(value: float, unit: str) -> Optional[float]:
    f = _MASS_FACTORS.get(unit.lower())
    return value * f if f is not None else None


def _from_kg(kg: float, unit: str) -> float:
    f = _MASS_FACTORS[unit.lower()]
    return kg / f


def _to_l(value: float, unit: str) -> Optional[float]:
    f = _VOLUME_FACTORS.get(unit.lower())
    return value * f if f is not None else None


def _from_l(litres: float, unit: str) -> float:
    f = _VOLUME_FACTORS[unit.lower()]
    return litres / f


__all__ = [
    "DensityRecord",
    "DensityLookupError",
    "DENSITY_TABLE",
    "get_density",
    "mass_to_volume_l",
    "volume_l_to_mass",
    "convert",
]
