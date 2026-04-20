# -*- coding: utf-8 -*-
"""
Moisture adjustment engine (Phase F5).

Climate inventories report fuel consumption on either a wet basis
(as-received / ar) or a dry basis (ad).  IPCC defaults use as-received,
but plant-level fuel measurements are frequently reported dry.  Getting
this conversion wrong produces errors of 5–20 % on coal and biomass
emissions.

Conversions::

    mass_dry = mass_wet × (1 - moisture_fraction)
    mass_wet = mass_dry / (1 - moisture_fraction)

Heating value conversions::

    LHV_wet = LHV_dry × (1 - m) - 2.447 × m      (MJ/kg; 2.447 = latent heat of water at 25 °C)
    LHV_dry = (LHV_wet + 2.447 × m) / (1 - m)

Moisture fraction MUST be expressed as kg water / kg wet fuel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MoistureBasis(str, Enum):
    AS_RECEIVED = "as_received"         # wet basis (standard for IPCC)
    DRY = "dry"                         # dry basis (plant-level reports)
    ASH_FREE_DRY = "ash_free_dry"       # daf — chemistry labs


class MoistureError(ValueError):
    pass


@dataclass
class MoistureConversion:
    moisture_fraction: float            # 0.0–1.0 kg water / kg wet fuel
    from_basis: MoistureBasis
    to_basis: MoistureBasis
    converted_mass_kg: float
    notes: str = ""


def _validate(m: float) -> None:
    if not 0.0 <= m < 1.0:
        raise MoistureError(
            "moisture_fraction must be 0.0 ≤ m < 1.0 (got %r); express as "
            "kg water / kg wet fuel" % m
        )


def convert_mass(
    *,
    mass_kg: float,
    moisture_fraction: float,
    from_basis: MoistureBasis,
    to_basis: MoistureBasis,
) -> MoistureConversion:
    """Convert fuel mass between wet and dry bases."""
    _validate(moisture_fraction)
    if from_basis == to_basis:
        return MoistureConversion(
            moisture_fraction=moisture_fraction,
            from_basis=from_basis,
            to_basis=to_basis,
            converted_mass_kg=mass_kg,
            notes="identity",
        )
    if from_basis == MoistureBasis.AS_RECEIVED and to_basis == MoistureBasis.DRY:
        converted = mass_kg * (1.0 - moisture_fraction)
        return MoistureConversion(
            moisture_fraction=moisture_fraction,
            from_basis=from_basis,
            to_basis=to_basis,
            converted_mass_kg=converted,
            notes="wet → dry: mass × (1 - m)",
        )
    if from_basis == MoistureBasis.DRY and to_basis == MoistureBasis.AS_RECEIVED:
        converted = mass_kg / (1.0 - moisture_fraction)
        return MoistureConversion(
            moisture_fraction=moisture_fraction,
            from_basis=from_basis,
            to_basis=to_basis,
            converted_mass_kg=converted,
            notes="dry → wet: mass / (1 - m)",
        )
    # ash-free dry paths require ash content — caller passes it via notes
    raise MoistureError(
        "ash_free_dry conversions require explicit ash content; not implemented here"
    )


LATENT_HEAT_WATER_MJ_PER_KG = 2.447              # at 25 °C


def convert_lhv(
    *,
    lhv_mj_per_kg: float,
    moisture_fraction: float,
    from_basis: MoistureBasis,
    to_basis: MoistureBasis,
) -> float:
    """Convert a lower heating value between wet and dry bases.

    Formula (from IPCC 2006 Vol.2 Annex 1)::

        LHV_wet = LHV_dry × (1 - m) - 2.447 × m
        LHV_dry = (LHV_wet + 2.447 × m) / (1 - m)
    """
    _validate(moisture_fraction)
    if from_basis == to_basis:
        return lhv_mj_per_kg
    if from_basis == MoistureBasis.DRY and to_basis == MoistureBasis.AS_RECEIVED:
        return (
            lhv_mj_per_kg * (1.0 - moisture_fraction)
            - LATENT_HEAT_WATER_MJ_PER_KG * moisture_fraction
        )
    if from_basis == MoistureBasis.AS_RECEIVED and to_basis == MoistureBasis.DRY:
        return (lhv_mj_per_kg + LATENT_HEAT_WATER_MJ_PER_KG * moisture_fraction) / (
            1.0 - moisture_fraction
        )
    raise MoistureError(
        "ash_free_dry LHV conversions require ash content; not implemented here"
    )


__all__ = [
    "MoistureBasis",
    "MoistureConversion",
    "MoistureError",
    "LATENT_HEAT_WATER_MJ_PER_KG",
    "convert_mass",
    "convert_lhv",
]
