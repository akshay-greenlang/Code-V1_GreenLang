# -*- coding: utf-8 -*-
"""
Fossil / biogenic split calculator (Phase F5).

GHG Protocol requires Scope 1 to report **fossil** CO2 and Scope 2/3
biogenic CO2 **separately**.  Many mixed fuels (B20 biodiesel blend,
B5 diesel, municipal solid waste, co-fired biomass-coal plants) carry
both fossil and biogenic carbon and need an explicit split.

This module exposes:

- ``biogenic_share(fuel, blend)`` — returns the biogenic fraction of
  carbon mass (0.0 = pure fossil, 1.0 = pure biogenic)
- ``split_emissions(co2_total_kg, biogenic_fraction)`` — returns a
  ``FossilBiogenicSplit`` with fossil + biogenic components
- Pre-registered fuel blends + MSW composition defaults
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FossilBiogenicSplit:
    """Per-fuel fossil / biogenic carbon split."""

    fossil_co2_kg: float
    biogenic_co2_kg: float
    biogenic_fraction: float          # 0.0–1.0
    source: str = ""

    @property
    def total_co2_kg(self) -> float:
        return self.fossil_co2_kg + self.biogenic_co2_kg


# ---------------------------------------------------------------------------
# Default biogenic shares by fuel blend
# ---------------------------------------------------------------------------

#: Biogenic share expressed as **fraction of total carbon**.
BIOGENIC_SHARE_TABLE: Dict[str, Dict[str, float]] = {
    "diesel": {
        "pure": 0.0,
        "b5": 0.05,                     # 5 % biodiesel blend
        "b7": 0.07,
        "b20": 0.20,
        "b100": 1.00,
    },
    "gasoline": {
        "pure": 0.0,
        "e10": 0.10,
        "e15": 0.15,
        "e20": 0.20,
        "e85": 0.85,
        "e100": 1.00,
    },
    "natural_gas": {
        "pure": 0.0,
        "biomethane_blend_10": 0.10,
        "biomethane_blend_50": 0.50,
        "biomethane_pure": 1.00,
    },
    "coal": {
        "pure": 0.0,
        "cofiring_5_biomass": 0.05,
        "cofiring_10_biomass": 0.10,
        "cofiring_20_biomass": 0.20,
    },
    "msw": {
        "default_global": 0.45,         # IPCC 2006 MSW composition avg
        "europe_default": 0.55,
        "us_default": 0.40,
    },
    "biomass": {
        "pure": 1.00,
    },
    "biodiesel": {
        "pure": 1.00,
    },
    "ethanol": {
        "pure": 1.00,
    },
}


class BiogenicShareError(KeyError):
    pass


def biogenic_share(fuel: str, blend: str = "pure") -> float:
    """Return the biogenic fraction of a fuel blend (0.0–1.0)."""
    key = fuel.strip().lower()
    blends = BIOGENIC_SHARE_TABLE.get(key)
    if blends is None:
        raise BiogenicShareError("No biogenic-share table for fuel %r" % fuel)
    share = blends.get(blend.lower())
    if share is None:
        raise BiogenicShareError(
            "Blend %r not registered for fuel %r; available: %s"
            % (blend, fuel, sorted(blends))
        )
    return share


def split_emissions(
    *,
    co2_total_kg: float,
    biogenic_fraction: float,
    source: str = "",
) -> FossilBiogenicSplit:
    """Split a total CO2 mass into fossil + biogenic components.

    Non-negotiable #1 compliant: we do NOT derive biogenic CO2 by
    subtraction — caller must supply either the fraction or query the
    table.  This keeps the split auditable.
    """
    if not 0.0 <= biogenic_fraction <= 1.0:
        raise ValueError(
            "biogenic_fraction must be 0.0 ≤ f ≤ 1.0 (got %r)" % biogenic_fraction
        )
    biogenic = co2_total_kg * biogenic_fraction
    fossil = co2_total_kg - biogenic
    return FossilBiogenicSplit(
        fossil_co2_kg=fossil,
        biogenic_co2_kg=biogenic,
        biogenic_fraction=biogenic_fraction,
        source=source,
    )


def split_fuel(
    *,
    co2_total_kg: float,
    fuel: str,
    blend: str = "pure",
) -> FossilBiogenicSplit:
    """High-level helper combining lookup + split."""
    share = biogenic_share(fuel, blend)
    return split_emissions(
        co2_total_kg=co2_total_kg,
        biogenic_fraction=share,
        source=f"BIOGENIC_SHARE_TABLE:{fuel}:{blend}",
    )


__all__ = [
    "BIOGENIC_SHARE_TABLE",
    "BiogenicShareError",
    "FossilBiogenicSplit",
    "biogenic_share",
    "split_emissions",
    "split_fuel",
]
