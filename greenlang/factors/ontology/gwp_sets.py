# -*- coding: utf-8 -*-
"""
GWP (Global Warming Potential) Sets - IPCC AR4 / AR5 / AR6 registry.

This module provides deterministic, zero-hallucination lookups for GWP values
across IPCC Assessment Reports (AR4, AR5, AR6) with 20-year and 100-year
time horizons.

Design principles (CTO non-negotiable #1):
- NEVER store only CO2e - always keep the gas vector and derive CO2e here.
- Default horizon is IPCC AR6 100-year (``GWPSet.IPCC_AR6_100``).
- Values are Decimal-safe floats with full source citations.

Sources
-------
- IPCC AR4 (2007) - Climate Change 2007: The Physical Science Basis,
  Working Group I Chapter 2, Table 2.14.
- IPCC AR5 (2013) - Climate Change 2013: The Physical Science Basis,
  Working Group I Chapter 8, Table 8.7 (AR5 values incl. climate-carbon
  feedback are NOT used by default, per UNFCCC reporting guidance).
- IPCC AR6 (2021) - Climate Change 2021: The Physical Science Basis,
  Working Group I Chapter 7, Table 7.15 (without feedbacks; "GWP-100").

Usage
-----
::

    from greenlang.factors.ontology.gwp_sets import (
        GWPSet, get_gwp, convert_co2e, DEFAULT_GWP_SET,
    )

    ch4_gwp = get_gwp("CH4", GWPSet.IPCC_AR6_100)     # 27.9

    co2e = convert_co2e({"CO2": 1000.0, "CH4": 5.0, "N2O": 0.2},
                        to_set=GWPSet.IPCC_AR6_100)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Mapping, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# GWP Set Enumeration
# =============================================================================


class GWPSet(str, Enum):
    """IPCC Assessment Report GWP sets with time horizons.

    Naming: ``IPCC_{AR4|AR5|AR6}_{20|100}`` where 20/100 denote the
    Global-Warming-Potential time horizon in years.
    """

    IPCC_AR4_100 = "IPCC_AR4_100"
    IPCC_AR4_20 = "IPCC_AR4_20"
    IPCC_AR5_100 = "IPCC_AR5_100"
    IPCC_AR5_20 = "IPCC_AR5_20"
    IPCC_AR6_100 = "IPCC_AR6_100"
    IPCC_AR6_20 = "IPCC_AR6_20"


#: CTO non-negotiable #1 - default horizon is AR6 100-year.
DEFAULT_GWP_SET: GWPSet = GWPSet.IPCC_AR6_100


# =============================================================================
# GWP Registries (numeric values from IPCC reports)
# =============================================================================

_GWP_AR4_100: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 25.0,
    "N2O": 298.0,
    "HFC-23": 14800.0,
    "HFC-32": 675.0,
    "HFC-125": 3500.0,
    "HFC-134a": 1430.0,
    "HFC-143a": 4470.0,
    "HFC-152a": 124.0,
    "HFC-227ea": 3220.0,
    "HFC-236fa": 9810.0,
    "PFC-14": 7390.0,
    "PFC-116": 12200.0,
    "PFC-218": 8830.0,
    "SF6": 22800.0,
    "NF3": 17200.0,
}

_GWP_AR4_20: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 72.0,
    "N2O": 289.0,
    "HFC-23": 12000.0,
    "HFC-32": 2330.0,
    "HFC-125": 6350.0,
    "HFC-134a": 3830.0,
    "HFC-143a": 5890.0,
    "HFC-152a": 437.0,
    "HFC-227ea": 5310.0,
    "HFC-236fa": 8100.0,
    "PFC-14": 5210.0,
    "PFC-116": 8630.0,
    "PFC-218": 6310.0,
    "SF6": 16300.0,
    "NF3": 12800.0,
}

_GWP_AR5_100: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 28.0,
    "N2O": 265.0,
    "HFC-23": 12400.0,
    "HFC-32": 677.0,
    "HFC-125": 3170.0,
    "HFC-134a": 1300.0,
    "HFC-143a": 4800.0,
    "HFC-152a": 138.0,
    "HFC-227ea": 3350.0,
    "HFC-236fa": 8060.0,
    "PFC-14": 6630.0,
    "PFC-116": 11100.0,
    "PFC-218": 8900.0,
    "SF6": 23500.0,
    "NF3": 16100.0,
}

_GWP_AR5_20: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 84.0,
    "N2O": 264.0,
    "HFC-23": 10800.0,
    "HFC-32": 2430.0,
    "HFC-125": 6090.0,
    "HFC-134a": 3710.0,
    "HFC-143a": 6940.0,
    "HFC-152a": 506.0,
    "HFC-227ea": 5360.0,
    "HFC-236fa": 6940.0,
    "PFC-14": 4880.0,
    "PFC-116": 8210.0,
    "PFC-218": 6640.0,
    "SF6": 17500.0,
    "NF3": 12800.0,
}

_GWP_AR6_100: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 27.9,
    "CH4_FOSSIL": 29.8,
    "CH4_BIOGENIC": 27.0,
    "N2O": 273.0,
    "HFC-23": 14600.0,
    "HFC-32": 771.0,
    "HFC-125": 3740.0,
    "HFC-134a": 1530.0,
    "HFC-143a": 5810.0,
    "HFC-152a": 164.0,
    "HFC-227ea": 3600.0,
    "HFC-236fa": 8690.0,
    "PFC-14": 7380.0,
    "PFC-116": 12400.0,
    "PFC-218": 9290.0,
    "SF6": 25200.0,
    "NF3": 17400.0,
}

_GWP_AR6_20: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 81.2,
    "CH4_FOSSIL": 82.5,
    "CH4_BIOGENIC": 79.7,
    "N2O": 273.0,
    "HFC-23": 12400.0,
    "HFC-32": 2690.0,
    "HFC-125": 6740.0,
    "HFC-134a": 4140.0,
    "HFC-143a": 7840.0,
    "HFC-152a": 591.0,
    "HFC-227ea": 5850.0,
    "HFC-236fa": 7450.0,
    "PFC-14": 5300.0,
    "PFC-116": 8940.0,
    "PFC-218": 6770.0,
    "SF6": 18300.0,
    "NF3": 13400.0,
}


_REGISTRY: Dict[GWPSet, Dict[str, float]] = {
    GWPSet.IPCC_AR4_100: _GWP_AR4_100,
    GWPSet.IPCC_AR4_20: _GWP_AR4_20,
    GWPSet.IPCC_AR5_100: _GWP_AR5_100,
    GWPSet.IPCC_AR5_20: _GWP_AR5_20,
    GWPSet.IPCC_AR6_100: _GWP_AR6_100,
    GWPSet.IPCC_AR6_20: _GWP_AR6_20,
}


# =============================================================================
# Alias resolution
# =============================================================================


_GAS_ALIASES: Dict[str, str] = {
    "HFC23": "HFC-23",
    "HFC32": "HFC-32",
    "HFC125": "HFC-125",
    "HFC134A": "HFC-134a",
    "HFC143A": "HFC-143a",
    "HFC152A": "HFC-152a",
    "HFC227EA": "HFC-227ea",
    "HFC236FA": "HFC-236fa",
    "CF4": "PFC-14",
    "C2F6": "PFC-116",
    "C3F8": "PFC-218",
    "PFC14": "PFC-14",
    "PFC116": "PFC-116",
    "PFC218": "PFC-218",
    "METHANE": "CH4",
    "CARBON_DIOXIDE": "CO2",
    "NITROUS_OXIDE": "N2O",
}


def normalize_gas_code(gas: str) -> str:
    """Return the canonical gas code for a free-form input."""
    if not gas:
        raise ValueError("gas code cannot be empty")
    u = gas.strip().upper().replace(" ", "")
    stripped = u.replace("-", "")
    if stripped in _GAS_ALIASES:
        return _GAS_ALIASES[stripped]
    for canonical in _GWP_AR6_100.keys():
        if canonical.upper() == u:
            return canonical
    return u


# =============================================================================
# Public lookup API
# =============================================================================


def get_gwp(gas: str, gwp_set: GWPSet = DEFAULT_GWP_SET) -> float:
    """Return the GWP value for *gas* in *gwp_set*."""
    if gwp_set not in _REGISTRY:
        raise ValueError("unknown GWP set: %r" % (gwp_set,))
    code = normalize_gas_code(gas)
    table = _REGISTRY[gwp_set]
    if code not in table:
        raise KeyError("gas %r not found in %s" % (gas, gwp_set.value))
    return float(table[code])


def list_gases(gwp_set: GWPSet = DEFAULT_GWP_SET) -> Dict[str, float]:
    """Return a copy of the full gas-GWP mapping for *gwp_set*."""
    if gwp_set not in _REGISTRY:
        raise ValueError("unknown GWP set: %r" % (gwp_set,))
    return dict(_REGISTRY[gwp_set])


def convert_co2e(
    gas_vector: Mapping[str, float],
    *,
    to_set: GWPSet = DEFAULT_GWP_SET,
    from_set: Optional[GWPSet] = None,
    strict: bool = True,
) -> float:
    """Convert a gas vector (mass by gas) to CO2e using the target GWP set."""
    if to_set not in _REGISTRY:
        raise ValueError("unknown target GWP set: %r" % (to_set,))
    target = _REGISTRY[to_set]
    source = _REGISTRY[from_set] if from_set is not None else None

    total = 0.0
    for raw_gas, value in gas_vector.items():
        code = normalize_gas_code(raw_gas)
        if code not in target:
            if strict:
                raise KeyError(
                    "gas %r not found in target set %s" % (raw_gas, to_set.value)
                )
            logger.warning(
                "convert_co2e: gas %s missing in %s, skipping",
                code,
                to_set.value,
            )
            continue

        gwp_to = float(target[code])
        if source is not None:
            if code not in source:
                if strict:
                    raise KeyError(
                        "gas %r not found in source set %s"
                        % (raw_gas, from_set.value)
                    )
                logger.warning(
                    "convert_co2e: gas %s missing in %s, skipping",
                    code,
                    from_set.value,
                )
                continue
            gwp_from = float(source[code])
            if gwp_from == 0:
                raise ValueError("source GWP is zero for %s" % code)
            mass = float(value) / gwp_from
            total += mass * gwp_to
        else:
            total += float(value) * gwp_to

    return total


def compare_sets(
    gas: str, sets: Optional[list] = None
) -> Dict[str, Optional[float]]:
    """Return GWP of *gas* across all (or specified) sets for reporting."""
    selected = sets if sets is not None else list(_REGISTRY.keys())
    out: Dict[str, Optional[float]] = {}
    for s in selected:
        try:
            out[s.value] = get_gwp(gas, s)
        except KeyError:
            out[s.value] = None
    return out


__all__ = [
    "GWPSet",
    "DEFAULT_GWP_SET",
    "normalize_gas_code",
    "get_gwp",
    "list_gases",
    "convert_co2e",
    "compare_sets",
]
