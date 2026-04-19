# -*- coding: utf-8 -*-
"""Global Warming Potential (GWP) table.

Values from IPCC AR4, AR5, AR6 assessment reports. Used to convert non-CO2
GHG masses to CO2-equivalents.
"""

from __future__ import annotations

from decimal import Decimal

from greenlang.scope_engine.models import GHGGas, GWPBasis

# IPCC AR6 WG1 Chapter 7, 100-year GWP (Forster et al. 2021)
# HFC/PFC use representative values (HFC-134a, CF4) — production factor lookups
# should pull gas-specific values from the factor catalog instead.
_GWP_TABLE: dict[GWPBasis, dict[GHGGas, Decimal]] = {
    GWPBasis.AR4_100YR: {
        GHGGas.CO2: Decimal(1),
        GHGGas.CH4: Decimal(25),
        GHGGas.N2O: Decimal(298),
        GHGGas.HFC: Decimal(1430),
        GHGGas.PFC: Decimal(7390),
        GHGGas.SF6: Decimal(22800),
        GHGGas.NF3: Decimal(17200),
        GHGGas.CO2E: Decimal(1),
    },
    GWPBasis.AR5_100YR: {
        GHGGas.CO2: Decimal(1),
        GHGGas.CH4: Decimal(28),
        GHGGas.N2O: Decimal(265),
        GHGGas.HFC: Decimal(1300),
        GHGGas.PFC: Decimal(6630),
        GHGGas.SF6: Decimal(23500),
        GHGGas.NF3: Decimal(16100),
        GHGGas.CO2E: Decimal(1),
    },
    GWPBasis.AR5_20YR: {
        GHGGas.CO2: Decimal(1),
        GHGGas.CH4: Decimal(84),
        GHGGas.N2O: Decimal(264),
        GHGGas.HFC: Decimal(3710),
        GHGGas.PFC: Decimal(4880),
        GHGGas.SF6: Decimal(17500),
        GHGGas.NF3: Decimal(12800),
        GHGGas.CO2E: Decimal(1),
    },
    GWPBasis.AR6_100YR: {
        GHGGas.CO2: Decimal(1),
        GHGGas.CH4: Decimal("29.8"),
        GHGGas.N2O: Decimal(273),
        GHGGas.HFC: Decimal(1530),
        GHGGas.PFC: Decimal(7380),
        GHGGas.SF6: Decimal(25200),
        GHGGas.NF3: Decimal(17400),
        GHGGas.CO2E: Decimal(1),
    },
    GWPBasis.AR6_20YR: {
        GHGGas.CO2: Decimal(1),
        GHGGas.CH4: Decimal(82),
        GHGGas.N2O: Decimal(273),
        GHGGas.HFC: Decimal(4140),
        GHGGas.PFC: Decimal(5300),
        GHGGas.SF6: Decimal(18200),
        GHGGas.NF3: Decimal(13400),
        GHGGas.CO2E: Decimal(1),
    },
}


def gwp_factor(gas: GHGGas, basis: GWPBasis) -> Decimal:
    try:
        return _GWP_TABLE[basis][gas]
    except KeyError as e:
        raise ValueError(f"No GWP value for gas={gas} basis={basis}") from e


def to_co2e(gas: GHGGas, mass_kg: Decimal, basis: GWPBasis) -> Decimal:
    return Decimal(mass_kg) * gwp_factor(gas, basis)
