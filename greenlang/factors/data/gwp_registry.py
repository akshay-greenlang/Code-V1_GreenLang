# -*- coding: utf-8 -*-
"""External GWP coefficient registry (W4-A / M20 / X06).

The legacy :class:`~greenlang.data.emission_factor_record.GHGVectors` class
embedded a ``GWP_VALUES`` dict on every instance — hundreds of kilobytes of
duplicated lookup tables across the ingested catalog. The v1 canonical record
stores only the :class:`GWPSet` enum string; the actual CH4/N2O/F-gas
coefficients are looked up HERE.

This module is the single source of truth for GWP coefficients in GreenLang.
All new code should call :func:`lookup` / :func:`co2e` rather than embedding
tables of its own.

Sources (all authoritative):

* IPCC AR6 (2021) — Working Group I, Table 7.SM.7 (100-yr) & 7.SM.6 (20-yr).
* IPCC AR5 (2013) — Working Group I, Chapter 8, Table 8.7.
* IPCC AR4 (2007) — Working Group I, Table 2.14.
* IPCC SAR (1995) — the Kyoto Protocol reference set.

Non-negotiable #1 (CO2e is DERIVED, never stored only) is enforced by
:func:`co2e` which returns ``Decimal`` for bit-perfect reproducibility.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Final, Mapping, Optional


# ---------------------------------------------------------------------------
# Enum values — duplicated here as strings to avoid a circular import with
# :mod:`canonical_v1` (which imports this module).  MUST stay in sync with
# ``GWP_SET_ENUM`` in canonical_v1.
# ---------------------------------------------------------------------------

_IPCC_AR4_100 = "IPCC_AR4_100"
_IPCC_AR5_100 = "IPCC_AR5_100"
_IPCC_AR5_20 = "IPCC_AR5_20"
_IPCC_AR6_100 = "IPCC_AR6_100"
_IPCC_AR6_20 = "IPCC_AR6_20"
_KYOTO_SAR_100 = "Kyoto_SAR_100"

ALL_GWP_SETS: Final[tuple[str, ...]] = (
    _IPCC_AR4_100,
    _IPCC_AR5_100,
    _IPCC_AR5_20,
    _IPCC_AR6_100,
    _IPCC_AR6_20,
    _KYOTO_SAR_100,
)

DEFAULT_GWP_SET: Final[str] = _IPCC_AR6_100


# ---------------------------------------------------------------------------
# Coefficient table.  Keys are (gwp_set, gas_code) tuples.
# Values are stored as Decimal for precision.
# ---------------------------------------------------------------------------


def _d(v: float | int) -> Decimal:
    return Decimal(str(v))


# CO2 is always 1.0 by definition — stored explicitly so callers can iterate.
_GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
    _IPCC_AR6_100: {
        "CO2": _d(1.0),
        "CH4": _d(27.9),         # fossil CH4, includes C-cycle feedback
        "CH4_biogenic": _d(27.2),
        "N2O": _d(273.0),
        # F-gas point values (per gas)
        "HFC-23": _d(14600),
        "HFC-32": _d(771),
        "HFC-125": _d(3740),
        "HFC-134a": _d(1530),
        "HFC-143a": _d(5810),
        "HFC-152a": _d(164),
        "HFC-227ea": _d(3600),
        "HFC-245fa": _d(962),
        "HFC-365mfc": _d(914),
        "HFC-4310mee": _d(1600),
        # Blend generics — using R-410a / R-404a / R-407c families
        "R-134a": _d(1530),
        "R-410A": _d(2256),
        "R-404A": _d(4728),
        "R-407C": _d(1908),
        "R-507A": _d(4468),
        "PFC-14": _d(7380),     # CF4
        "PFC-116": _d(12400),   # C2F6
        "PFC-218": _d(9290),    # C3F8
        "SF6": _d(25200),
        "NF3": _d(17400),
        # Aggregate fallbacks (used when a record only lists "HFCs"/"PFCs")
        "HFCs": _d(1526),
        "PFCs": _d(7380),
    },
    _IPCC_AR6_20: {
        "CO2": _d(1.0),
        "CH4": _d(82.5),
        "CH4_biogenic": _d(80.8),
        "N2O": _d(273.0),
        "HFC-23": _d(12900),
        "HFC-32": _d(2530),
        "HFC-125": _d(6500),
        "HFC-134a": _d(4140),
        "HFC-143a": _d(7840),
        "HFC-152a": _d(591),
        "HFC-227ea": _d(5850),
        "HFC-245fa": _d(3170),
        "R-134a": _d(4140),
        "R-410A": _d(4715),
        "R-404A": _d(6013),
        "R-407C": _d(4158),
        "PFC-14": _d(5300),
        "PFC-116": _d(8940),
        "SF6": _d(18300),
        "NF3": _d(13400),
        "HFCs": _d(4140),
        "PFCs": _d(5300),
    },
    _IPCC_AR5_100: {
        "CO2": _d(1.0),
        "CH4": _d(28.0),
        "CH4_biogenic": _d(28.0),
        "N2O": _d(265.0),
        "HFC-23": _d(12400),
        "HFC-32": _d(677),
        "HFC-125": _d(3170),
        "HFC-134a": _d(1300),
        "HFC-143a": _d(4800),
        "HFC-152a": _d(138),
        "HFC-227ea": _d(3350),
        "HFC-245fa": _d(858),
        "R-134a": _d(1300),
        "R-410A": _d(1924),
        "R-404A": _d(3922),
        "R-407C": _d(1624),
        "R-507A": _d(3985),
        "PFC-14": _d(6630),
        "PFC-116": _d(11100),
        "SF6": _d(23500),
        "NF3": _d(16100),
        "HFCs": _d(1300),
        "PFCs": _d(6630),
    },
    _IPCC_AR5_20: {
        "CO2": _d(1.0),
        "CH4": _d(84.0),
        "CH4_biogenic": _d(84.0),
        "N2O": _d(264.0),
        "HFC-23": _d(10800),
        "HFC-32": _d(2430),
        "HFC-125": _d(6090),
        "HFC-134a": _d(3710),
        "HFC-143a": _d(6940),
        "HFC-152a": _d(506),
        "R-134a": _d(3710),
        "R-410A": _d(4340),
        "R-404A": _d(6010),
        "R-407C": _d(3820),
        "PFC-14": _d(4880),
        "PFC-116": _d(8210),
        "SF6": _d(17500),
        "NF3": _d(12800),
        "HFCs": _d(3710),
        "PFCs": _d(4880),
    },
    _IPCC_AR4_100: {
        "CO2": _d(1.0),
        "CH4": _d(25.0),
        "CH4_biogenic": _d(25.0),
        "N2O": _d(298.0),
        "HFC-23": _d(14800),
        "HFC-32": _d(675),
        "HFC-125": _d(3500),
        "HFC-134a": _d(1430),
        "HFC-143a": _d(4470),
        "HFC-152a": _d(124),
        "HFC-227ea": _d(3220),
        "HFC-245fa": _d(1030),
        "R-134a": _d(1430),
        "R-410A": _d(2088),
        "R-404A": _d(3922),
        "R-407C": _d(1774),
        "R-507A": _d(3985),
        "PFC-14": _d(7390),
        "PFC-116": _d(12200),
        "SF6": _d(22800),
        "NF3": _d(17200),
        "HFCs": _d(1430),
        "PFCs": _d(7390),
    },
    _KYOTO_SAR_100: {
        "CO2": _d(1.0),
        "CH4": _d(21.0),
        "CH4_biogenic": _d(21.0),
        "N2O": _d(310.0),
        "HFC-23": _d(11700),
        "HFC-32": _d(650),
        "HFC-125": _d(2800),
        "HFC-134a": _d(1300),
        "HFC-143a": _d(3800),
        "HFC-152a": _d(140),
        "R-134a": _d(1300),
        "R-410A": _d(1725),
        "R-404A": _d(3260),
        "R-407C": _d(1525),
        "PFC-14": _d(6500),
        "PFC-116": _d(9200),
        "SF6": _d(23900),
        "NF3": _d(8000),
        "HFCs": _d(1300),
        "PFCs": _d(6500),
    },
}


# Aliases (accept legacy names without warning)
_SET_ALIASES: Dict[str, str] = {
    "AR6_100": _IPCC_AR6_100,
    "AR6_20": _IPCC_AR6_20,
    "AR5_100": _IPCC_AR5_100,
    "AR5_20": _IPCC_AR5_20,
    "AR4_100": _IPCC_AR4_100,
    "SAR_100": _KYOTO_SAR_100,
    "IPCC_SAR_100": _KYOTO_SAR_100,  # pre-v1 name
}


def normalize_gwp_set(gwp_set: str) -> str:
    """Return the canonical v1 enum string for ``gwp_set``.

    Accepts legacy aliases (``AR6_100``, ``IPCC_SAR_100``, ...) and returns the
    canonical v1 value. Raises :class:`ValueError` for unknown strings.
    """
    if gwp_set in _GWP_TABLE:
        return gwp_set
    canonical = _SET_ALIASES.get(gwp_set)
    if canonical is None:
        raise ValueError(
            f"Unknown gwp_set {gwp_set!r}. Valid: {sorted(_GWP_TABLE)}"
        )
    return canonical


def lookup(gwp_set: str, gas: str) -> Decimal:
    """Look up the GWP coefficient for ``(gwp_set, gas)``.

    Parameters
    ----------
    gwp_set
        One of :data:`ALL_GWP_SETS` (or a legacy alias — auto-normalised).
    gas
        Gas code (e.g., ``"CH4"``, ``"HFC-134a"``, ``"SF6"``).  Aggregate
        fallbacks ``"HFCs"`` / ``"PFCs"`` are accepted when a record does
        not list the specific species.

    Returns
    -------
    Decimal
        The GWP coefficient.

    Raises
    ------
    ValueError
        If ``gwp_set`` or ``gas`` is unknown in that set.
    """
    canonical_set = normalize_gwp_set(gwp_set)
    table = _GWP_TABLE[canonical_set]
    if gas not in table:
        # Forgiving fallback for common refrigerant-family lookups.
        # E.g. asking for "HFC-134a" when only "R-134a" is present.
        alt_gas = {
            "HFC-134a": "R-134a",
            "HFC-32": "HFC-32",
        }.get(gas)
        if alt_gas and alt_gas in table:
            return table[alt_gas]
        raise ValueError(
            f"Gas {gas!r} not in gwp_set {canonical_set!r}. "
            f"Known: {sorted(table)}"
        )
    return table[gas]


def available_gases(gwp_set: str) -> list[str]:
    """Return the list of gas codes registered for ``gwp_set``."""
    return sorted(_GWP_TABLE[normalize_gwp_set(gwp_set)])


def co2e(
    gases: Mapping[str, float | int | Decimal | None],
    gwp_set: str = DEFAULT_GWP_SET,
    *,
    f_gases: Optional[Mapping[str, float | int | Decimal | None]] = None,
) -> Decimal:
    """Compute CO2e from a dict of gas masses.

    Non-negotiable #1 — CO2e is DERIVED, never stored only.

    Parameters
    ----------
    gases
        Mapping of gas code to mass (any numeric). CO2/CH4/N2O at minimum.
    gwp_set
        Reference set (default :data:`DEFAULT_GWP_SET`).
    f_gases
        Optional separate mapping for F-gas contributions (keeps the v1 spec
        shape where ``numerator.f_gases`` is a dict).

    Returns
    -------
    Decimal
        Total CO2-equivalent (kg CO2e per denominator unit).

    Raises
    ------
    ValueError
        On unknown gas / set combinations.
    """
    total = Decimal("0")
    iter_maps = [gases]
    if f_gases:
        iter_maps.append(f_gases)
    for mapping in iter_maps:
        for gas, mass in mapping.items():
            if mass is None:
                continue
            mass_dec = mass if isinstance(mass, Decimal) else Decimal(str(mass))
            if mass_dec == 0:
                continue
            total += mass_dec * lookup(gwp_set, gas)
    return total


def all_sets() -> tuple[str, ...]:
    """Return all registered canonical gwp_set values."""
    return ALL_GWP_SETS


__all__ = [
    "ALL_GWP_SETS",
    "DEFAULT_GWP_SET",
    "available_gases",
    "all_sets",
    "co2e",
    "lookup",
    "normalize_gwp_set",
]
