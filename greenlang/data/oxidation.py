# -*- coding: utf-8 -*-
"""
Oxidation factor engine (Phase F5).

IPCC guidelines prescribe per-fuel, per-technology oxidation factors
(f_ox) that represent the fraction of fuel carbon actually oxidised to
CO2 during combustion.  Older (IPCC 1996) guidelines used lower defaults
(coal=0.98, oil=0.99, gas=0.995); IPCC 2006 moved toward full 1.00
assumption with explicit subtraction of any non-CO2 carbon later.
GreenLang supports both conventions via ``basis``.

Formula::

    fossil_CO2 = mass_fuel × carbon_content × oxidation_factor × 44/12

The oxidation engine also captures the tier-specific values for special
cases (fluidised-bed coal combustion, incomplete flaring, peat-specific
factors).

Used by:

- :class:`MethodProfile.CORPORATE_SCOPE1`
- :class:`MethodProfile.EU_CBAM` (CBAM Annex III references IPCC oxidation)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class OxidationBasis(str, Enum):
    """Which IPCC guideline the oxidation factor follows."""

    IPCC_1996 = "IPCC_1996"             # lower defaults (coal=0.98, oil=0.99, gas=0.995)
    IPCC_2006 = "IPCC_2006"             # full 1.00 baseline, exceptions explicit
    IPCC_2019_REFINEMENT = "IPCC_2019"  # refined defaults for fluidized-bed, etc.
    CUSTOM = "custom"


@dataclass(frozen=True)
class OxidationFactor:
    """One oxidation factor row."""

    fuel: str
    technology: str                         # e.g., "default", "fluidized_bed", "pulverized"
    factor: float                           # 0.0–1.0
    basis: OxidationBasis
    source: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Reference table
# ---------------------------------------------------------------------------

# Keyed by (fuel, technology, basis).  Coverage: the fuels that IPCC 2006
# Vol.2 Ch.1 Table 1.4 enumerates + the two most common exceptions
# (fluidised-bed coal, flaring).

OXIDATION_TABLE: Dict[tuple, OxidationFactor] = {
    # --- IPCC 2006 baseline (full oxidation assumption) ---
    ("diesel", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="diesel", technology="default", factor=1.00,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.1 Table 1.4",
    ),
    ("gasoline", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="gasoline", technology="default", factor=1.00,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.1 Table 1.4",
    ),
    ("natural_gas", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="natural_gas", technology="default", factor=1.00,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.1 Table 1.4",
    ),
    ("coal", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="coal", technology="default", factor=1.00,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.1 Table 1.4",
    ),
    ("coal", "fluidized_bed", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="coal", technology="fluidized_bed", factor=0.98,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.2 Table 2.4 (FBC)",
        notes="Fluidized-bed combustion: ~2 % unburned carbon captured in ash",
    ),
    ("coal", "pulverized", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="coal", technology="pulverized", factor=0.99,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.2",
    ),
    ("fuel_oil", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="fuel_oil", technology="default", factor=1.00,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.1",
    ),
    ("biomass", "default", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="biomass", technology="default", factor=0.92,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.2 Table 2.6",
        notes="Biomass combustion typically ~8 % unburned + char",
    ),
    # --- IPCC 1996 legacy (for backward compatibility) ---
    ("diesel", "default", OxidationBasis.IPCC_1996): OxidationFactor(
        fuel="diesel", technology="default", factor=0.99,
        basis=OxidationBasis.IPCC_1996,
        source="IPCC 1996 Revised Vol.2",
    ),
    ("gasoline", "default", OxidationBasis.IPCC_1996): OxidationFactor(
        fuel="gasoline", technology="default", factor=0.99,
        basis=OxidationBasis.IPCC_1996,
        source="IPCC 1996 Revised Vol.2",
    ),
    ("natural_gas", "default", OxidationBasis.IPCC_1996): OxidationFactor(
        fuel="natural_gas", technology="default", factor=0.995,
        basis=OxidationBasis.IPCC_1996,
        source="IPCC 1996 Revised Vol.2",
    ),
    ("coal", "default", OxidationBasis.IPCC_1996): OxidationFactor(
        fuel="coal", technology="default", factor=0.98,
        basis=OxidationBasis.IPCC_1996,
        source="IPCC 1996 Revised Vol.2",
    ),
    # --- Special cases ---
    ("natural_gas", "flaring", OxidationBasis.IPCC_2006): OxidationFactor(
        fuel="natural_gas", technology="flaring", factor=0.995,
        basis=OxidationBasis.IPCC_2006,
        source="IPCC 2006 Vol.2 Ch.4 (oil & gas)",
        notes="Flares typically achieve 99.5 % destruction efficiency",
    ),
}


class OxidationLookupError(KeyError):
    pass


def get_oxidation_factor(
    *,
    fuel: str,
    technology: str = "default",
    basis: OxidationBasis = OxidationBasis.IPCC_2006,
) -> OxidationFactor:
    """Return the oxidation factor row for ``(fuel, technology, basis)``.

    Falls back to ``technology='default'`` when an exact tech match is missing.
    """
    fuel_norm = fuel.strip().lower()
    key = (fuel_norm, technology.lower(), basis)
    rec = OXIDATION_TABLE.get(key)
    if rec is None:
        rec = OXIDATION_TABLE.get((fuel_norm, "default", basis))
    if rec is None:
        raise OxidationLookupError(
            "No oxidation factor for (%r, %r, %s)" % (fuel, technology, basis.value)
        )
    return rec


def apply_oxidation(
    *,
    mass_carbon_kg: float,
    fuel: str,
    technology: str = "default",
    basis: OxidationBasis = OxidationBasis.IPCC_2006,
) -> float:
    """Return oxidised carbon (kg) = mass_carbon × oxidation_factor."""
    rec = get_oxidation_factor(fuel=fuel, technology=technology, basis=basis)
    return mass_carbon_kg * rec.factor


__all__ = [
    "OxidationBasis",
    "OxidationFactor",
    "OXIDATION_TABLE",
    "OxidationLookupError",
    "get_oxidation_factor",
    "apply_oxidation",
]
