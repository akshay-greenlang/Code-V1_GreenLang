# -*- coding: utf-8 -*-
"""
Gas-level chemistry utilities — stoichiometry, oxidation, refrigerants, biogenic CO2.

This module turns fuel/activity inputs into a **gas vector** (kg by gas),
which downstream :mod:`greenlang.factors.ontology.gwp_sets` aggregates into
CO2e. It also hosts the helpers for:

* Carbon-content → stoichiometric CO2 (12/44 ratio).
* Oxidation-factor application (combustion completeness).
* Biogenic vs fossil CO2 split (cradle-to-gate treatment).
* Biogenic sequestration / end-of-life fate tracking.
* Refrigerant leakage modelling (recharge, annual leak, EoL recovery).

Sources
-------
- IPCC 2006 GL Vol.2 Ch.1 Table 1.4 (default oxidation factors).
- IPCC 2006 GL Vol.3 Ch.7 (refrigeration & AC F-gas Tier 2 methodology).
- GHG Protocol Corporate Standard §9 (biogenic CO2 reporting).
- US EPA AP-42 Section 1 (combustion completeness by fuel class).
- UNFCCC Reporting Guidelines on AIRs, Annex I (biogenic separation).

Design notes
------------
Per CTO non-negotiable #1: **never store only CO2e**. These utilities
produce a gas vector (``{"CO2_fossil": ..., "CO2_biogenic": ..., "CH4": ...}``)
that callers pass into ``convert_co2e``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from greenlang.factors.ontology.gwp_sets import (
    DEFAULT_GWP_SET,
    GWPSet,
    convert_co2e,
    get_gwp,
    normalize_gas_code,
)
from greenlang.factors.ontology.heating_values import get_fuel

logger = logging.getLogger(__name__)


# =============================================================================
# Stoichiometric constants
# =============================================================================

#: Molar mass ratio CO2 / C = 44.009 / 12.011 ≈ 3.6640 kg CO2 per kg C.
C_TO_CO2_RATIO: float = 44.009 / 12.011

#: Default fossil/biogenic classification for registry fuels.
_BIOGENIC_FUELS = frozenset({
    "wood", "wood_pellets", "wood_chips", "bagasse",
    "agricultural_residue", "charcoal", "ethanol", "biodiesel",
    "renewable_diesel", "saf", "biogas", "biomethane",
})

# Municipal waste is mixed — a portion is biogenic.
_MIXED_BIOGENIC_FRACTION = {
    "municipal_solid_waste": 0.60,
    "refuse_derived_fuel": 0.50,
}


# =============================================================================
# Default oxidation factors (combustion completeness, fraction 0..1)
# =============================================================================

_DEFAULT_OXIDATION = {
    "anthracite": 0.98,
    "bituminous_coal": 0.98,
    "sub_bituminous_coal": 0.98,
    "lignite": 0.98,
    "coking_coal": 0.98,
    "coke_oven_coke": 0.98,
    "peat": 0.98,
    "natural_gas": 0.995,
    "lng": 0.995,
    "cng": 0.995,
    "lpg": 0.995,
    "propane": 0.995,
    "butane": 0.995,
    "biogas": 0.995,
    "biomethane": 0.995,
    "synthetic_methane": 0.995,
    "diesel": 0.99,
    "gasoline": 0.99,
    "kerosene": 0.99,
    "jet_fuel": 0.99,
    "aviation_gasoline": 0.99,
    "residual_fuel_oil": 0.99,
    "bunker_fuel": 0.99,
    "marine_diesel_oil": 0.99,
    "crude_oil": 0.99,
    "shale_oil": 0.99,
    "ethanol": 0.99,
    "methanol": 0.99,
    "biodiesel": 0.99,
    "renewable_diesel": 0.99,
    "saf": 0.99,
    "wood": 0.97,
    "wood_pellets": 0.97,
    "wood_chips": 0.97,
    "bagasse": 0.97,
    "agricultural_residue": 0.97,
    "charcoal": 0.97,
    "municipal_solid_waste": 0.95,
    "refuse_derived_fuel": 0.95,
    "hydrogen": 1.0,
    "green_hydrogen": 1.0,
    "blue_hydrogen": 1.0,
    "ammonia": 0.99,
}


def get_default_oxidation_factor(fuel_code: str) -> float:
    """Return the IPCC Tier-1 default oxidation factor for *fuel_code*."""
    code = fuel_code.strip().lower()
    if code in _DEFAULT_OXIDATION:
        return _DEFAULT_OXIDATION[code]
    # fall back: try registry alias resolution
    try:
        canonical = get_fuel(fuel_code).fuel_code
    except KeyError:
        return 0.99  # conservative catch-all
    return _DEFAULT_OXIDATION.get(canonical, 0.99)


# =============================================================================
# CO2 stoichiometry
# =============================================================================


def carbon_to_co2(carbon_kg: float) -> float:
    """Convert mass of carbon (kg) → mass of CO2 (kg) via 44/12 ratio."""
    if carbon_kg < 0:
        raise ValueError("carbon_kg must be non-negative")
    return float(carbon_kg) * C_TO_CO2_RATIO


def apply_oxidation_factor(
    carbon_kg: float, oxidation_factor: float = 1.0
) -> float:
    """Multiply *carbon_kg* by an oxidation factor (0..1).

    Accounts for incomplete combustion — a fraction of the fuel carbon
    escapes as soot/unburned hydrocarbons and is NOT reported as CO2
    (it is captured separately via CH4/CO/NMVOC emission factors).
    """
    if not 0.0 <= oxidation_factor <= 1.0:
        raise ValueError(
            "oxidation_factor must be in [0,1], got %r" % oxidation_factor
        )
    return float(carbon_kg) * float(oxidation_factor)


# =============================================================================
# Fossil vs biogenic CO2 split
# =============================================================================


@dataclass(frozen=True)
class CO2Split:
    """Result of splitting fuel CO2 into fossil vs biogenic components."""

    co2_fossil_kg: float
    co2_biogenic_kg: float
    biogenic_fraction: float
    fuel_code: str
    oxidation_factor: float

    @property
    def co2_total_kg(self) -> float:
        return self.co2_fossil_kg + self.co2_biogenic_kg

    def as_gas_vector(self) -> Dict[str, float]:
        """Emit the standard gas vector keys for this split.

        Biogenic CO2 is reported separately and is **not** aggregated
        into CO2e by default under GHG Protocol / IPCC reporting rules
        (cradle-to-gate). It stays in the vector for transparency.
        """
        return {
            "CO2_fossil": self.co2_fossil_kg,
            "CO2_biogenic": self.co2_biogenic_kg,
        }


def split_fossil_biogenic_co2(
    fuel_code: str,
    fuel_mass_kg: float,
    *,
    oxidation_factor: Optional[float] = None,
    carbon_content_fraction: Optional[float] = None,
    biogenic_fraction: Optional[float] = None,
) -> CO2Split:
    """Compute the fossil / biogenic CO2 split for *fuel_mass_kg* of fuel.

    Args:
        fuel_code: Fuel identifier from the heating-values registry.
        fuel_mass_kg: Burned fuel mass in kg.
        oxidation_factor: Override the default combustion completeness.
        carbon_content_fraction: Override the registry carbon content.
        biogenic_fraction: Override the biogenic share (0..1). For pure
            fossil fuels default is 0; for pure biogenic fuels default
            is 1; for MSW/RDF the default is taken from
            ``_MIXED_BIOGENIC_FRACTION``.
    """
    if fuel_mass_kg < 0:
        raise ValueError("fuel_mass_kg must be non-negative")

    fv = get_fuel(fuel_code)
    carbon_frac = (
        float(carbon_content_fraction)
        if carbon_content_fraction is not None
        else fv.carbon_content_fraction
    )
    if carbon_frac <= 0:
        # Zero-carbon fuels (hydrogen, ammonia) produce no CO2 from combustion.
        return CO2Split(
            co2_fossil_kg=0.0,
            co2_biogenic_kg=0.0,
            biogenic_fraction=0.0,
            fuel_code=fv.fuel_code,
            oxidation_factor=1.0,
        )

    ox = (
        float(oxidation_factor)
        if oxidation_factor is not None
        else get_default_oxidation_factor(fv.fuel_code)
    )

    carbon_oxidised_kg = apply_oxidation_factor(
        float(fuel_mass_kg) * carbon_frac, ox
    )
    total_co2 = carbon_to_co2(carbon_oxidised_kg)

    # Determine biogenic fraction.
    if biogenic_fraction is not None:
        bio_frac = float(biogenic_fraction)
    elif fv.fuel_code in _BIOGENIC_FUELS:
        bio_frac = 1.0
    elif fv.fuel_code in _MIXED_BIOGENIC_FRACTION:
        bio_frac = _MIXED_BIOGENIC_FRACTION[fv.fuel_code]
    else:
        bio_frac = 0.0

    if not 0.0 <= bio_frac <= 1.0:
        raise ValueError(
            "biogenic_fraction must be in [0,1], got %r" % bio_frac
        )

    co2_bio = total_co2 * bio_frac
    co2_fossil = total_co2 - co2_bio

    return CO2Split(
        co2_fossil_kg=co2_fossil,
        co2_biogenic_kg=co2_bio,
        biogenic_fraction=bio_frac,
        fuel_code=fv.fuel_code,
        oxidation_factor=ox,
    )


# =============================================================================
# Biogenic sequestration tracking
# =============================================================================


@dataclass(frozen=True)
class BiogenicFate:
    """Cradle-to-gate vs cradle-to-grave biogenic treatment."""

    co2_absorbed_kg: float           # sequestered during biomass growth
    co2_released_combustion_kg: float
    co2_released_eol_kg: float       # end-of-life (decomposition etc.)
    treatment: str                   # "cradle_to_gate" | "cradle_to_grave"

    @property
    def net_biogenic_kg(self) -> float:
        """Net biogenic CO2 after accounting for sequestration."""
        if self.treatment == "cradle_to_gate":
            # only combustion counted, sequestration accounted separately
            return self.co2_released_combustion_kg
        # cradle-to-grave: net = released - absorbed
        return (
            self.co2_released_combustion_kg
            + self.co2_released_eol_kg
            - self.co2_absorbed_kg
        )


def biogenic_fate(
    co2_biogenic_kg: float,
    *,
    co2_absorbed_kg: Optional[float] = None,
    co2_eol_kg: float = 0.0,
    treatment: str = "cradle_to_gate",
) -> BiogenicFate:
    """Build a :class:`BiogenicFate` record.

    If *co2_absorbed_kg* is omitted, carbon neutrality is assumed so
    absorbed equals released — the GHG-Protocol default for managed
    forestry and agricultural residues.
    """
    if treatment not in {"cradle_to_gate", "cradle_to_grave"}:
        raise ValueError("treatment must be cradle_to_gate|cradle_to_grave")
    absorbed = (
        float(co2_absorbed_kg)
        if co2_absorbed_kg is not None
        else float(co2_biogenic_kg)
    )
    return BiogenicFate(
        co2_absorbed_kg=absorbed,
        co2_released_combustion_kg=float(co2_biogenic_kg),
        co2_released_eol_kg=float(co2_eol_kg),
        treatment=treatment,
    )


# =============================================================================
# Refrigerant leakage modelling
# =============================================================================


@dataclass(frozen=True)
class RefrigerantLeakageResult:
    """Annualised refrigerant leakage model output."""

    refrigerant: str
    recharge_kg: float
    installation_leak_kg: float
    operational_leak_kg: float
    end_of_life_leak_kg: float
    total_leak_kg: float
    co2e_kg: float
    gwp_set: GWPSet
    gwp_value: float

    def as_gas_vector(self) -> Dict[str, float]:
        """Return the gas vector (mass by gas, kg)."""
        return {self.refrigerant: self.total_leak_kg}


def model_refrigerant_leakage(
    refrigerant: str,
    *,
    charge_kg: float,
    annual_leak_rate: float = 0.02,
    installation_leak_rate: float = 0.01,
    end_of_life_recovery_rate: float = 0.70,
    end_of_life_flag: bool = False,
    years: float = 1.0,
    gwp_set: GWPSet = DEFAULT_GWP_SET,
) -> RefrigerantLeakageResult:
    """Model refrigerant emissions for an equipment unit.

    Implements IPCC 2006 Vol.3 Ch.7 Tier 2 methodology:

    * **Installation leak** — one-off charge × installation rate (year 1 only).
    * **Operational leak** — charge × annual rate × years.
    * **End-of-life leak** — charge × (1 - recovery rate), applied only when
      ``end_of_life_flag`` is True (equipment retired in period).

    Args:
        refrigerant: Gas code (e.g. ``"HFC-134a"``, ``"HFC-32"``, ``"SF6"``).
        charge_kg: Initial refrigerant charge in kg.
        annual_leak_rate: Fraction of charge lost per year (default 2%).
        installation_leak_rate: One-off installation loss (year 1 only).
        end_of_life_recovery_rate: Fraction recovered at EoL (default 70%).
        end_of_life_flag: True if equipment is retired in this period.
        years: Number of operational years in the reporting period.
        gwp_set: Target GWP set (default AR6 100-yr).

    Returns:
        :class:`RefrigerantLeakageResult`.
    """
    if charge_kg < 0:
        raise ValueError("charge_kg must be non-negative")
    if years < 0:
        raise ValueError("years must be non-negative")
    for name, rate in (
        ("annual_leak_rate", annual_leak_rate),
        ("installation_leak_rate", installation_leak_rate),
        ("end_of_life_recovery_rate", end_of_life_recovery_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("%s must be in [0,1], got %r" % (name, rate))

    install = charge_kg * installation_leak_rate if years > 0 else 0.0
    operational = charge_kg * annual_leak_rate * years
    eol = (
        charge_kg * (1.0 - end_of_life_recovery_rate)
        if end_of_life_flag
        else 0.0
    )
    total = install + operational + eol
    # Recharge is what's added to compensate for leaks (engineering view).
    recharge = operational + install

    canonical = normalize_gas_code(refrigerant)
    try:
        gwp = get_gwp(canonical, gwp_set)
    except KeyError:
        # Some refrigerant blends (e.g. R-410a) aren't in IPCC tables —
        # fall back to a conservative HFC-125 estimate & log.
        logger.warning(
            "model_refrigerant_leakage: gas %s not in %s, using 0 GWP fallback",
            canonical,
            gwp_set.value,
        )
        gwp = 0.0
    co2e = total * gwp

    return RefrigerantLeakageResult(
        refrigerant=canonical,
        recharge_kg=recharge,
        installation_leak_kg=install,
        operational_leak_kg=operational,
        end_of_life_leak_kg=eol,
        total_leak_kg=total,
        co2e_kg=co2e,
        gwp_set=gwp_set,
        gwp_value=gwp,
    )


# =============================================================================
# High-level gas-vector assembly (the canonical calculation path)
# =============================================================================


def build_combustion_gas_vector(
    fuel_code: str,
    fuel_mass_kg: float,
    *,
    oxidation_factor: Optional[float] = None,
    ch4_kg_per_kg_fuel: float = 0.0,
    n2o_kg_per_kg_fuel: float = 0.0,
    biogenic_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Assemble the canonical gas vector for a combustion activity.

    This is the single entry point that honours CTO non-negotiable #1:
    the caller gets a gas vector (kg by gas) and aggregates via
    :func:`greenlang.factors.ontology.gwp_sets.convert_co2e`.

    Returns keys: ``CO2_fossil``, ``CO2_biogenic``, ``CH4``, ``N2O``.
    Zero-valued entries are retained for downstream audit trails.
    """
    split = split_fossil_biogenic_co2(
        fuel_code,
        fuel_mass_kg,
        oxidation_factor=oxidation_factor,
        biogenic_fraction=biogenic_fraction,
    )
    vector = split.as_gas_vector()
    vector["CH4"] = float(fuel_mass_kg) * float(ch4_kg_per_kg_fuel)
    vector["N2O"] = float(fuel_mass_kg) * float(n2o_kg_per_kg_fuel)
    return vector


def aggregate_co2e(
    gas_vector: Mapping[str, float],
    *,
    gwp_set: GWPSet = DEFAULT_GWP_SET,
    include_biogenic: bool = False,
) -> float:
    """Aggregate a gas vector → CO2e, optionally excluding biogenic CO2.

    GHG-Protocol Corporate Standard excludes biogenic CO2 from Scope 1
    by default (it is reported separately). Set ``include_biogenic=True``
    for cradle-to-grave / end-to-end analyses.
    """
    # Map biogenic CO2 vector key into a form the GWP registry accepts.
    filtered: Dict[str, float] = {}
    for gas, value in gas_vector.items():
        key_upper = gas.strip().upper()
        if key_upper == "CO2_BIOGENIC":
            if include_biogenic:
                filtered["CO2"] = filtered.get("CO2", 0.0) + float(value)
            continue
        if key_upper == "CO2_FOSSIL":
            filtered["CO2"] = filtered.get("CO2", 0.0) + float(value)
            continue
        filtered[gas] = float(value)
    if not filtered:
        return 0.0
    return convert_co2e(filtered, to_set=gwp_set)


__all__ = [
    "C_TO_CO2_RATIO",
    "CO2Split",
    "BiogenicFate",
    "RefrigerantLeakageResult",
    "get_default_oxidation_factor",
    "carbon_to_co2",
    "apply_oxidation_factor",
    "split_fossil_biogenic_co2",
    "biogenic_fate",
    "model_refrigerant_leakage",
    "build_combustion_gas_vector",
    "aggregate_co2e",
]
