# -*- coding: utf-8 -*-
"""GWP table correctness tests (against published IPCC values)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.scope_engine.gwp import gwp_factor, to_co2e
from greenlang.scope_engine.models import GHGGas, GWPBasis


@pytest.mark.parametrize(
    "gas,basis,expected",
    [
        # IPCC AR6 100yr (Forster et al. 2021, Table 7.SM.7)
        (GHGGas.CO2, GWPBasis.AR6_100YR, Decimal(1)),
        (GHGGas.CH4, GWPBasis.AR6_100YR, Decimal("29.8")),
        (GHGGas.N2O, GWPBasis.AR6_100YR, Decimal(273)),
        (GHGGas.SF6, GWPBasis.AR6_100YR, Decimal(25200)),
        # IPCC AR5 100yr
        (GHGGas.CH4, GWPBasis.AR5_100YR, Decimal(28)),
        (GHGGas.N2O, GWPBasis.AR5_100YR, Decimal(265)),
        # IPCC AR4 (legacy GHG Protocol default)
        (GHGGas.CH4, GWPBasis.AR4_100YR, Decimal(25)),
        (GHGGas.N2O, GWPBasis.AR4_100YR, Decimal(298)),
    ],
)
def test_gwp_factor_matches_ipcc(gas, basis, expected):
    assert gwp_factor(gas, basis) == expected


def test_to_co2e_multiplies_correctly():
    # 1 kg CH4 at AR6-100yr = 29.8 kg CO2e
    assert to_co2e(GHGGas.CH4, Decimal(1), GWPBasis.AR6_100YR) == Decimal("29.8")
    # 10 kg N2O at AR6-100yr = 2730 kg CO2e
    assert to_co2e(GHGGas.N2O, Decimal(10), GWPBasis.AR6_100YR) == Decimal(2730)


def test_gwp_basis_enum_roundtrips():
    # Valid enum parsing
    assert GWPBasis("AR6-100yr") == GWPBasis.AR6_100YR
    # gwp_factor returns for valid combos
    assert gwp_factor(GHGGas.CO2, GWPBasis.AR6_100YR) == Decimal(1)
