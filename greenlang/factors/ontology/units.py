# -*- coding: utf-8 -*-
"""
Activity-unit ontology (S2) with LHV/HHV, advanced fuels, and gas-law helpers.

Extended in GAP-1 closure to support:

* LHV/HHV basis flag on energy-content conversions.
* Temperature-corrected gas volume -> energy via ideal-gas law.
* Advanced fuels: green hydrogen (120 MJ/kg LHV), synthetic methane, SAF,
  and configurable biofuels (routed through the heating-values registry).
* Steam enthalpy lookup helper (saturated steam at 1-10 bar g, IAPWS-IF97
  subset — deterministic table).

Sources
-------
- IPCC 2006 GL Vol.2 Ch.1 Table 1.2 (net calorific values).
- NIST Webbook (hydrogen LHV/HHV 120 / 141.8 MJ/kg).
- ISO 6976:2016 (natural-gas calorific value calculation).
- IAPWS-IF97 saturated steam tables (subset).
"""

from __future__ import annotations

from typing import Literal, Optional

from greenlang.factors.ontology.heating_values import (
    HeatingBasis,
    convert_mass_to_energy,
    get_fuel,
    get_heating_value,
)

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
    if u in ("gj", "mj", "mmbtu", "therms"):
        return "J"
    if u in ("m3", "liters", "litres", "gallons", "scf"):
        return "m3"
    return None


# Energy: factors stored relative to kWh baseline for consistent comparisons (S2).
_KWH_PER_UNIT = {
    "kwh": 1.0,
    "mwh": 1000.0,
    "gwh": 1_000_000.0,
    "gj": 277.778,  # 1 GJ ~ 277.778 kWh
    "mj": 0.277778,
    "therms": 29.3001,  # US therm -> kWh
    "mmbtu": 293.071,  # million BTU -> kWh
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


# =============================================================================
# Ideal-gas law helpers
# =============================================================================

#: Universal gas constant in J/(mol*K).
R_UNIVERSAL_J_PER_MOL_K: float = 8.314462618

#: Reference STP conditions (DESNZ / IEA convention): 0 C, 1 atm.
STP_TEMPERATURE_K: float = 273.15
STP_PRESSURE_PA: float = 101_325.0

#: Molar mass of commonly converted gases (kg/mol).
_MOLAR_MASS_KG_PER_MOL = {
    "natural_gas": 0.01704,   # ~17.04 g/mol for typical NG composition
    "biomethane": 0.01604,
    "synthetic_methane": 0.01604,
    "methane": 0.01604,
    "ch4": 0.01604,
    "hydrogen": 0.00201588,
    "h2": 0.00201588,
    "green_hydrogen": 0.00201588,
    "blue_hydrogen": 0.00201588,
    "ammonia": 0.01703,
    "nh3": 0.01703,
    "propane": 0.04409,
    "butane": 0.05812,
    "lpg": 0.04800,
    "cng": 0.01704,
    "co2": 0.04401,
    "biogas": 0.02800,  # composition-averaged (~55% CH4 / 45% CO2)
}


def gas_volume_to_mass_kg(
    volume_m3: float,
    fuel_code: str,
    *,
    temperature_C: float = 0.0,
    pressure_pa: float = STP_PRESSURE_PA,
) -> float:
    """Convert gas *volume_m3* to mass (kg) via the ideal-gas law.

    Uses::

        n = P*V / (R*T)       [mol]
        mass = n * M          [kg]

    Args:
        volume_m3: Gas volume in cubic metres (at the given T and P).
        fuel_code: Gas identifier with a known molar mass.
        temperature_C: Gas temperature (Celsius). Default 0 C (STP).
        pressure_pa: Pressure in Pa. Default 101325 Pa (1 atm).
    """
    if volume_m3 < 0:
        raise ValueError("volume_m3 must be non-negative")
    if pressure_pa <= 0:
        raise ValueError("pressure_pa must be positive")
    t_k = temperature_C + 273.15
    if t_k <= 0:
        raise ValueError("temperature must be above absolute zero")
    key = fuel_code.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _MOLAR_MASS_KG_PER_MOL:
        molar_mass = _MOLAR_MASS_KG_PER_MOL[key]
    else:
        # fall back via heating-values registry lookup
        try:
            fv = get_fuel(fuel_code)
            key2 = fv.fuel_code
        except KeyError as exc:
            raise ValueError(
                "no molar mass registered for %r" % fuel_code
            ) from exc
        if key2 not in _MOLAR_MASS_KG_PER_MOL:
            raise ValueError(
                "no molar mass registered for %r" % fuel_code
            )
        molar_mass = _MOLAR_MASS_KG_PER_MOL[key2]

    moles = (pressure_pa * volume_m3) / (R_UNIVERSAL_J_PER_MOL_K * t_k)
    return moles * molar_mass


def gas_volume_to_energy_mj(
    volume_m3: float,
    fuel_code: str,
    *,
    temperature_C: float = 0.0,
    pressure_pa: float = STP_PRESSURE_PA,
    basis: HeatingBasis = "LHV",
) -> float:
    """Convert gas *volume_m3* directly to energy (MJ) via ideal gas + HV.

    Pipeline::

        V, T, P -> mass via ideal-gas law -> energy via LHV/HHV

    Args:
        volume_m3: Gas volume in cubic metres.
        fuel_code: Gas identifier (must have molar mass + HV data).
        temperature_C: Actual temperature (C).
        pressure_pa: Actual pressure (Pa).
        basis: ``"LHV"`` (default) or ``"HHV"``.
    """
    mass = gas_volume_to_mass_kg(
        volume_m3, fuel_code,
        temperature_C=temperature_C, pressure_pa=pressure_pa,
    )
    return convert_mass_to_energy(mass, fuel_code, basis)


def fuel_energy_content(
    amount: float,
    unit: str,
    fuel_code: str,
    *,
    basis: HeatingBasis = "LHV",
    temperature_C: Optional[float] = None,
    pressure_pa: Optional[float] = None,
) -> float:
    """Unified entry point: energy (MJ) from a fuel quantity in any supported unit.

    Supported input units (case-insensitive):

    * Mass: ``kg``, ``g``, ``t``, ``tonnes``, ``lb``
    * Volume: ``m3``, ``L``, ``litres``, ``liters``, ``gallons``, ``scf``

    When ``unit`` is ``m3`` or ``scf`` AND ``temperature_C`` is supplied,
    the ideal-gas pipeline is used; otherwise the fuel's registry density
    is used (suitable for liquid fuels).

    Args:
        amount: Numeric amount.
        unit: Unit code.
        fuel_code: Fuel identifier.
        basis: ``"LHV"`` (default) or ``"HHV"``.
        temperature_C: Optional gas temperature for volumetric gas input.
        pressure_pa: Optional gas pressure for volumetric gas input.
    """
    if amount < 0:
        raise ValueError("amount must be non-negative")
    u = unit.strip().lower()
    fv = get_fuel(fuel_code)
    hv = fv.get(basis)

    # mass-like units
    if u in ("kg",):
        return float(amount) * hv
    if u in ("g", "gram", "grams"):
        return float(amount) / 1000.0 * hv
    if u in ("t", "tonne", "tonnes", "metric_ton"):
        return float(amount) * 1000.0 * hv
    if u in ("lb", "pound", "pounds"):
        return float(amount) * 0.45359237 * hv

    # volume-like units
    if u in ("m3", "cubic_meter", "cubic_metre"):
        # For gases, use ideal-gas law if T is provided or density is small.
        if temperature_C is not None or fv.density_kg_per_m3 < 5.0:
            return gas_volume_to_energy_mj(
                float(amount), fuel_code,
                temperature_C=temperature_C if temperature_C is not None else 0.0,
                pressure_pa=pressure_pa if pressure_pa is not None else STP_PRESSURE_PA,
                basis=basis,
            )
        # Liquid/solid: density-based
        return float(amount) * fv.density_kg_per_m3 * hv

    if u in ("l", "liter", "liters", "litre", "litres"):
        m3 = float(amount) / 1000.0
        return fuel_energy_content(
            m3, "m3", fuel_code,
            basis=basis, temperature_C=temperature_C, pressure_pa=pressure_pa,
        )

    if u in ("gallon", "gallons", "us_gallon"):
        liters = float(amount) * 3.78541
        return fuel_energy_content(
            liters, "liters", fuel_code,
            basis=basis, temperature_C=temperature_C, pressure_pa=pressure_pa,
        )

    if u in ("scf", "standard_cubic_foot", "standard_cubic_feet"):
        # Standard cubic foot = 0.0283168 m3 at 60 F / 14.73 psi; approximate as STP.
        m3 = float(amount) * 0.0283168
        return gas_volume_to_energy_mj(
            m3, fuel_code,
            temperature_C=15.56 if temperature_C is None else temperature_C,
            pressure_pa=101_560.0 if pressure_pa is None else pressure_pa,
            basis=basis,
        )

    raise ValueError("unsupported unit for fuel_energy_content: %r" % unit)


# =============================================================================
# Steam enthalpy (saturated steam table, IAPWS-IF97 subset)
# =============================================================================

#: Saturated-steam specific enthalpy (kJ/kg) at bar-gauge pressure, from
#: IAPWS-IF97 industrial formulation.
_STEAM_ENTHALPY_KJ_PER_KG = {
    # bar gauge -> (hf, hg) enthalpies of sat. liquid and sat. vapour
    0: (419.04, 2676.0),
    1: (504.7, 2706.7),
    2: (562.2, 2725.5),
    3: (605.3, 2738.1),
    4: (640.1, 2747.5),
    5: (670.6, 2755.0),
    6: (697.1, 2761.0),
    7: (720.9, 2766.0),
    8: (742.6, 2770.1),
    9: (762.6, 2773.5),
    10: (781.1, 2776.2),
}


def steam_enthalpy_kj_per_kg(
    pressure_bar_gauge: float,
    *,
    phase: Literal["vapour", "liquid"] = "vapour",
) -> float:
    """Return saturated-steam specific enthalpy in kJ/kg (IAPWS-IF97 subset).

    Args:
        pressure_bar_gauge: Gauge pressure in bar (0..10, linearly
            interpolated between table rows).
        phase: ``"vapour"`` (default) or ``"liquid"``.
    """
    if pressure_bar_gauge < 0 or pressure_bar_gauge > 10:
        raise ValueError(
            "pressure_bar_gauge must be in [0, 10], got %r" % pressure_bar_gauge
        )
    idx = phase == "vapour"
    lower = int(pressure_bar_gauge)
    upper = lower + 1 if lower < 10 else 10
    frac = pressure_bar_gauge - lower
    lo = _STEAM_ENTHALPY_KJ_PER_KG[lower][1 if idx else 0]
    hi = _STEAM_ENTHALPY_KJ_PER_KG[upper][1 if idx else 0]
    return lo + (hi - lo) * frac


def steam_energy_mj(
    mass_kg: float,
    pressure_bar_gauge: float,
    *,
    feedwater_temperature_C: float = 20.0,
) -> float:
    """Return net steam energy (MJ) above feedwater reference.

    Approximates feedwater enthalpy as h_fw ~= 4.186 * T_fw (kJ/kg) —
    acceptable for T < 100 C.
    """
    if mass_kg < 0:
        raise ValueError("mass_kg must be non-negative")
    h_steam = steam_enthalpy_kj_per_kg(pressure_bar_gauge, phase="vapour")
    h_fw = 4.186 * float(feedwater_temperature_C)
    delta = h_steam - h_fw
    return float(mass_kg) * delta / 1000.0  # kJ -> MJ


# =============================================================================
# Energy-content convenience (LHV/HHV aware)
# =============================================================================


def convert_fuel_to_kwh(
    amount: float, unit: str, fuel_code: str,
    *, basis: HeatingBasis = "LHV",
    temperature_C: Optional[float] = None,
    pressure_pa: Optional[float] = None,
) -> float:
    """Convert any fuel-quantity input to kWh via LHV/HHV.

    Thin wrapper: MJ -> kWh using 1 kWh = 3.6 MJ.
    """
    mj = fuel_energy_content(
        amount, unit, fuel_code,
        basis=basis, temperature_C=temperature_C, pressure_pa=pressure_pa,
    )
    return mj / 3.6


__all__ = [
    "KNOWN_DENOMINATORS",
    "is_known_activity_unit",
    "suggest_si_base",
    "convert_energy_to_kwh",
    "convert_energy",
    "R_UNIVERSAL_J_PER_MOL_K",
    "STP_TEMPERATURE_K",
    "STP_PRESSURE_PA",
    "gas_volume_to_mass_kg",
    "gas_volume_to_energy_mj",
    "fuel_energy_content",
    "convert_fuel_to_kwh",
    "steam_enthalpy_kj_per_kg",
    "steam_energy_mj",
    "get_heating_value",
]
