# -*- coding: utf-8 -*-
"""
Fuel Heating Values Registry - Standardised LHV/HHV data for ~40 fuels.

This module ships a deterministic, fully-cited registry of lower/higher
heating values and bulk physico-chemical properties for the fuels most
commonly encountered in GHG inventories, including fossil, bio- and
synthetic fuels.

Sources
-------
- IPCC 2006 Guidelines for National GHG Inventories, Vol. 2 (Energy),
  Chapter 1 (Introduction) - Table 1.2 (default NCV).
- UK DEFRA / DESNZ 2025 GHG Conversion Factors - "Fuels" sheet.
- US EPA Emission Factors Hub, March 2024 (HHV and carbon content).
- IEA World Energy Statistics 2023 (densities, moisture typical).
- NIST Webbook (ethanol, methanol thermophysical data).
- ISO 16559 (solid biofuels moisture) & DIN 51900-2.

Conventions
-----------
- LHV (Lower Heating Value) = Net Calorific Value (NCV).
- HHV (Higher Heating Value) = Gross Calorific Value (GCV).
- Heating values expressed in MJ per kg on an as-received basis at 25 C.
- Densities at 15 C for liquids, 0 C / 1 atm for gases (STP convention).
- Moisture / ash / sulfur / carbon contents are mass fractions (0..1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)


HeatingBasis = Literal["LHV", "HHV"]


@dataclass(frozen=True)
class FuelHeatingValue:
    """Canonical heating-value record for a fuel.

    Attributes:
        fuel_code: Snake-case fuel identifier (e.g. ``"natural_gas"``).
        LHV_MJ_per_kg: Lower heating value (net) at reference conditions.
        HHV_MJ_per_kg: Higher heating value (gross) at reference conditions.
        density_kg_per_m3: Bulk density at reference state.
        moisture_content_fraction: As-received moisture mass fraction (0..1).
        ash_content_fraction: As-received ash mass fraction (0..1).
        sulfur_content_fraction: As-received sulfur mass fraction (0..1).
        carbon_content_fraction: As-received carbon mass fraction (0..1).
        temperature_reference_C: Reference temperature in Celsius.
        source_citation: Short textual citation.
        aliases: Alternative fuel codes that map to this record.
    """

    fuel_code: str
    LHV_MJ_per_kg: float
    HHV_MJ_per_kg: float
    density_kg_per_m3: float
    moisture_content_fraction: float = 0.0
    ash_content_fraction: float = 0.0
    sulfur_content_fraction: float = 0.0
    carbon_content_fraction: float = 0.0
    temperature_reference_C: float = 25.0
    source_citation: str = ""
    aliases: tuple = field(default_factory=tuple)

    def get(self, basis: HeatingBasis = "LHV") -> float:
        """Return LHV or HHV in MJ/kg."""
        if basis == "LHV":
            return self.LHV_MJ_per_kg
        if basis == "HHV":
            return self.HHV_MJ_per_kg
        raise ValueError("basis must be 'LHV' or 'HHV', got %r" % basis)


def _fuel(
    code: str,
    lhv: float,
    hhv: float,
    density: float,
    *,
    moisture: float = 0.0,
    ash: float = 0.0,
    sulfur: float = 0.0,
    carbon: float = 0.0,
    tref: float = 25.0,
    cite: str = "",
    aliases: tuple = (),
) -> FuelHeatingValue:
    return FuelHeatingValue(
        fuel_code=code,
        LHV_MJ_per_kg=lhv,
        HHV_MJ_per_kg=hhv,
        density_kg_per_m3=density,
        moisture_content_fraction=moisture,
        ash_content_fraction=ash,
        sulfur_content_fraction=sulfur,
        carbon_content_fraction=carbon,
        temperature_reference_C=tref,
        source_citation=cite,
        aliases=aliases,
    )


_CITE_IPCC = "IPCC 2006 GL Vol.2 Ch.1 Table 1.2"
_CITE_DEFRA = "DEFRA/DESNZ 2025 GHG Conversion Factors"
_CITE_EPA = "US EPA Emission Factors Hub 2024"
_CITE_IEA = "IEA World Energy Statistics 2023"
_CITE_NIST = "NIST Webbook thermophysical data"


_REGISTRY: Dict[str, FuelHeatingValue] = {
    # ---- Coal family ----
    "anthracite": _fuel(
        "anthracite", 26.7, 27.8, 1506.0,
        moisture=0.05, ash=0.10, sulfur=0.008, carbon=0.805,
        cite=_CITE_IPCC, aliases=("anthracite_coal",),
    ),
    "bituminous_coal": _fuel(
        "bituminous_coal", 25.8, 27.0, 1346.0,
        moisture=0.08, ash=0.09, sulfur=0.013, carbon=0.70,
        cite=_CITE_IPCC, aliases=("hard_coal", "bituminous"),
    ),
    "sub_bituminous_coal": _fuel(
        "sub_bituminous_coal", 18.9, 20.0, 1346.0,
        moisture=0.15, ash=0.07, sulfur=0.006, carbon=0.55,
        cite=_CITE_IPCC, aliases=("subbituminous",),
    ),
    "lignite": _fuel(
        "lignite", 11.9, 13.4, 801.0,
        moisture=0.30, ash=0.10, sulfur=0.010, carbon=0.35,
        cite=_CITE_IPCC, aliases=("brown_coal",),
    ),
    "coking_coal": _fuel(
        "coking_coal", 28.2, 29.3, 1400.0,
        moisture=0.05, ash=0.08, sulfur=0.008, carbon=0.78,
        cite=_CITE_IPCC,
    ),
    "coke_oven_coke": _fuel(
        "coke_oven_coke", 28.2, 29.5, 1350.0,
        moisture=0.04, ash=0.10, sulfur=0.007, carbon=0.85,
        cite=_CITE_IPCC, aliases=("metallurgical_coke",),
    ),
    "peat": _fuel(
        "peat", 9.76, 11.0, 365.0,
        moisture=0.45, ash=0.05, sulfur=0.002, carbon=0.28,
        cite=_CITE_IPCC,
    ),
    # ---- Gaseous fuels ----
    "natural_gas": _fuel(
        "natural_gas", 48.0, 53.1, 0.7215,
        moisture=0.0, sulfur=0.00001, carbon=0.75,
        cite=_CITE_DEFRA, aliases=("ng", "pipeline_gas"),
    ),
    "lng": _fuel(
        "lng", 48.6, 55.1, 450.0,
        carbon=0.74, cite=_CITE_IEA,
        aliases=("liquefied_natural_gas",),
    ),
    "cng": _fuel(
        "cng", 48.0, 53.1, 180.0,
        carbon=0.75, cite=_CITE_DEFRA,
        aliases=("compressed_natural_gas",),
    ),
    "lpg": _fuel(
        "lpg", 46.4, 50.3, 540.0,
        carbon=0.82, cite=_CITE_DEFRA,
        aliases=("liquefied_petroleum_gas",),
    ),
    "propane": _fuel(
        "propane", 46.3, 50.3, 1.91,
        carbon=0.8182, cite=_CITE_EPA,
    ),
    "butane": _fuel(
        "butane", 45.7, 49.5, 2.52,
        carbon=0.8265, cite=_CITE_EPA,
    ),
    "biogas": _fuel(
        "biogas", 20.0, 22.0, 1.15,
        carbon=0.45, cite=_CITE_IPCC,
        aliases=("raw_biogas",),
    ),
    "biomethane": _fuel(
        "biomethane", 48.0, 53.1, 0.717,
        carbon=0.75, cite=_CITE_IEA,
        aliases=("upgraded_biogas", "rng"),
    ),
    "synthetic_methane": _fuel(
        "synthetic_methane", 50.0, 55.5, 0.717,
        carbon=0.75, cite=_CITE_IEA,
        aliases=("e_methane", "power_to_methane"),
    ),
    # ---- Liquid fuels - petroleum ----
    "crude_oil": _fuel(
        "crude_oil", 42.3, 45.1, 870.0,
        sulfur=0.015, carbon=0.85, cite=_CITE_IPCC,
    ),
    "diesel": _fuel(
        "diesel", 43.0, 45.6, 835.0,
        sulfur=0.001, carbon=0.867, cite=_CITE_DEFRA,
        aliases=("diesel_fuel", "gasoil", "dfo"),
    ),
    "gasoline": _fuel(
        "gasoline", 44.3, 47.1, 745.0,
        sulfur=0.0003, carbon=0.866, cite=_CITE_DEFRA,
        aliases=("petrol", "motor_gasoline"),
    ),
    "kerosene": _fuel(
        "kerosene", 43.8, 46.4, 810.0,
        sulfur=0.002, carbon=0.858, cite=_CITE_IPCC,
    ),
    "jet_fuel": _fuel(
        "jet_fuel", 44.1, 47.0, 800.0,
        sulfur=0.0005, carbon=0.859, cite=_CITE_DEFRA,
        aliases=("jet_a1", "aviation_kerosene"),
    ),
    "aviation_gasoline": _fuel(
        "aviation_gasoline", 44.3, 47.0, 720.0,
        carbon=0.853, cite=_CITE_IPCC, aliases=("avgas",),
    ),
    "residual_fuel_oil": _fuel(
        "residual_fuel_oil", 40.4, 43.1, 990.0,
        sulfur=0.025, carbon=0.857, cite=_CITE_IPCC,
        aliases=("heavy_fuel_oil", "hfo", "bunker_c"),
    ),
    "bunker_fuel": _fuel(
        "bunker_fuel", 40.2, 42.9, 985.0,
        sulfur=0.035, carbon=0.855, cite=_CITE_IPCC,
        aliases=("marine_fuel_oil", "bunker_oil", "ifo"),
    ),
    "marine_diesel_oil": _fuel(
        "marine_diesel_oil", 42.7, 45.4, 890.0,
        sulfur=0.010, carbon=0.860, cite=_CITE_IPCC,
        aliases=("mdo", "marine_gas_oil"),
    ),
    "shale_oil": _fuel(
        "shale_oil", 36.0, 38.8, 900.0,
        sulfur=0.005, carbon=0.76, cite=_CITE_IPCC,
    ),
    # ---- Biofuels ----
    "ethanol": _fuel(
        "ethanol", 26.7, 29.7, 789.0,
        carbon=0.521, cite=_CITE_NIST,
        aliases=("bioethanol", "ethyl_alcohol"),
    ),
    "methanol": _fuel(
        "methanol", 19.9, 22.7, 791.0,
        carbon=0.374, cite=_CITE_NIST,
        aliases=("methyl_alcohol",),
    ),
    "biodiesel": _fuel(
        "biodiesel", 37.0, 40.2, 880.0,
        carbon=0.77, cite=_CITE_DEFRA,
        aliases=("fame", "b100"),
    ),
    "renewable_diesel": _fuel(
        "renewable_diesel", 44.0, 47.0, 780.0,
        carbon=0.85, cite=_CITE_EPA,
        aliases=("hvo", "hydrotreated_vegetable_oil"),
    ),
    "saf": _fuel(
        "saf", 44.0, 47.0, 780.0,
        carbon=0.85, cite=_CITE_IEA,
        aliases=("sustainable_aviation_fuel", "bio_jet"),
    ),
    "wood": _fuel(
        "wood", 15.6, 18.6, 450.0,
        moisture=0.20, ash=0.01, carbon=0.48,
        cite=_CITE_IPCC, aliases=("firewood", "wood_logs"),
    ),
    "wood_pellets": _fuel(
        "wood_pellets", 17.0, 19.0, 650.0,
        moisture=0.08, ash=0.007, carbon=0.50,
        cite=_CITE_DEFRA, aliases=("biomass_pellets",),
    ),
    "wood_chips": _fuel(
        "wood_chips", 12.5, 15.5, 300.0,
        moisture=0.35, ash=0.012, carbon=0.48,
        cite=_CITE_IPCC,
    ),
    "bagasse": _fuel(
        "bagasse", 9.6, 11.0, 150.0,
        moisture=0.50, ash=0.03, carbon=0.47,
        cite=_CITE_IPCC,
    ),
    "agricultural_residue": _fuel(
        "agricultural_residue", 12.0, 14.5, 120.0,
        moisture=0.25, ash=0.05, carbon=0.45,
        cite=_CITE_IPCC, aliases=("straw", "crop_residue"),
    ),
    "charcoal": _fuel(
        "charcoal", 29.5, 32.0, 300.0,
        moisture=0.05, ash=0.03, carbon=0.75,
        cite=_CITE_IPCC,
    ),
    # ---- Waste-derived ----
    "municipal_solid_waste": _fuel(
        "municipal_solid_waste", 10.0, 12.0, 250.0,
        moisture=0.25, ash=0.20, carbon=0.30,
        cite=_CITE_IPCC, aliases=("msw",),
    ),
    "refuse_derived_fuel": _fuel(
        "refuse_derived_fuel", 15.0, 17.5, 200.0,
        moisture=0.15, ash=0.10, carbon=0.40,
        cite=_CITE_IPCC, aliases=("rdf",),
    ),
    # ---- Hydrogen family ----
    "hydrogen": _fuel(
        "hydrogen", 120.0, 141.8, 0.08988,
        carbon=0.0, cite=_CITE_NIST, aliases=("h2",),
    ),
    "green_hydrogen": _fuel(
        "green_hydrogen", 120.0, 141.8, 0.08988,
        carbon=0.0, cite=_CITE_NIST,
        aliases=("renewable_hydrogen", "h2_green"),
    ),
    "blue_hydrogen": _fuel(
        "blue_hydrogen", 120.0, 141.8, 0.08988,
        carbon=0.0, cite=_CITE_IEA,
        aliases=("h2_blue", "ccus_hydrogen"),
    ),
    "ammonia": _fuel(
        "ammonia", 18.6, 22.5, 0.73,
        carbon=0.0, cite=_CITE_NIST, aliases=("nh3",),
    ),
}


# Build reverse alias map at import.
_ALIAS_MAP: Dict[str, str] = {}
for _code, _fv in _REGISTRY.items():
    _ALIAS_MAP[_code] = _code
    for _alias in _fv.aliases:
        _ALIAS_MAP[_alias.lower()] = _code


def _resolve(fuel_code: str) -> str:
    if not fuel_code:
        raise ValueError("fuel_code cannot be empty")
    key = fuel_code.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    raise KeyError("unknown fuel_code: %r" % fuel_code)


def get_fuel(fuel_code: str) -> FuelHeatingValue:
    """Return the :class:`FuelHeatingValue` for *fuel_code*."""
    return _REGISTRY[_resolve(fuel_code)]


def get_heating_value(
    fuel_code: str, basis: HeatingBasis = "LHV"
) -> float:
    """Return LHV or HHV in MJ/kg for *fuel_code*."""
    return get_fuel(fuel_code).get(basis)


def list_fuels() -> list:
    """Return sorted list of canonical fuel codes."""
    return sorted(_REGISTRY.keys())


def convert_mass_to_energy(
    mass_kg: float, fuel_code: str, basis: HeatingBasis = "LHV"
) -> float:
    """Convert *mass_kg* of fuel to energy (MJ) using LHV or HHV."""
    if mass_kg < 0:
        raise ValueError("mass_kg must be non-negative, got %r" % mass_kg)
    hv = get_heating_value(fuel_code, basis)
    return float(mass_kg) * float(hv)


def convert_volume_to_energy(
    volume_m3: float, fuel_code: str, basis: HeatingBasis = "LHV"
) -> float:
    """Convert *volume_m3* of fuel to energy (MJ) via density x HV."""
    if volume_m3 < 0:
        raise ValueError("volume_m3 must be non-negative, got %r" % volume_m3)
    fv = get_fuel(fuel_code)
    mass_kg = float(volume_m3) * float(fv.density_kg_per_m3)
    return convert_mass_to_energy(mass_kg, fuel_code, basis)


def apply_moisture_correction(
    heating_value_mj_per_kg: float,
    moisture_fraction: float,
    *,
    dry_basis_value: bool = True,
    latent_heat_water_mj_per_kg: float = 2.45,
) -> float:
    """Apply moisture correction to a heating value.

    Formula (as-received LHV from dry LHV)::

        LHV_ar = LHV_dry * (1 - M) - L * M

    where ``M`` is the moisture fraction (0..1) and ``L`` is the latent
    heat of vaporisation (approx 2.45 MJ/kg at 20 C).
    """
    if not 0.0 <= moisture_fraction < 1.0:
        raise ValueError(
            "moisture_fraction must be in [0, 1), got %r" % moisture_fraction
        )
    if heating_value_mj_per_kg < 0:
        raise ValueError("heating_value cannot be negative")
    if dry_basis_value:
        corrected = (
            heating_value_mj_per_kg * (1.0 - moisture_fraction)
            - latent_heat_water_mj_per_kg * moisture_fraction
        )
    else:
        corrected = (
            heating_value_mj_per_kg + latent_heat_water_mj_per_kg * moisture_fraction
        ) / (1.0 - moisture_fraction)
    return max(corrected, 0.0)


def apply_temperature_correction(
    heating_value_mj_per_kg: float,
    temperature_C: float,
    fuel_code: str,
    *,
    coefficient_per_C: float = 0.0004,
) -> float:
    """Apply a first-order temperature correction to a heating value.

    HV(T) = HV(Tref) * (1 - k * (T - Tref)) with k defaulting to
    4e-4 /K (IPCC 2006 Vol.2 Annex 1).
    """
    fv = get_fuel(fuel_code)
    delta = temperature_C - fv.temperature_reference_C
    factor = 1.0 - coefficient_per_C * delta
    factor = max(factor, 0.5)
    return float(heating_value_mj_per_kg) * factor


def with_overrides(
    fuel_code: str, **overrides: float
) -> FuelHeatingValue:
    """Return a copy of a fuel record with provided fields overridden."""
    base = get_fuel(fuel_code)
    return replace(base, **overrides)


__all__ = [
    "FuelHeatingValue",
    "HeatingBasis",
    "get_fuel",
    "get_heating_value",
    "list_fuels",
    "convert_mass_to_energy",
    "convert_volume_to_energy",
    "apply_moisture_correction",
    "apply_temperature_correction",
    "with_overrides",
]
