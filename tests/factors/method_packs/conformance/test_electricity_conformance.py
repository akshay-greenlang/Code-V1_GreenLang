# -*- coding: utf-8 -*-
"""MP15 conformance tests — Electricity method pack.

Canonical cases: location-based grid (3 jurisdictions), market-based
supplier contract, residual-mix fallback.
"""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import electricity  # noqa: F401
from greenlang.factors.method_packs.exceptions import (
    FactorCannotResolveSafelyError,
)

from tests.factors.method_packs.conformance.conftest import (
    ConformanceRecord,
    assert_chosen_matches,
    resolve_case,
)


class TestElectricityConformance:
    """Conformance — Electricity packs must resolve canonical grid cases."""

    def test_location_based_gb_grid(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="GB",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:DESNZ:GRID_LOCATION:GB:2024",
                        factor_family="grid_intensity",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:DESNZ:GRID_LOCATION:GB:\d{4}$")

    def test_location_based_us_egrid_subregion(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="US",
            records_by_step={
                "utility_or_grid_subregion": [
                    ConformanceRecord(
                        factor_id="EF:EPA:EGRID_SERC:US:2024",
                        factor_family="grid_intensity",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EPA:EGRID_.*:US:\d{4}$")

    def test_location_based_india_cea(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="IN",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:CEA:GRID_LOCATION:IN:2024",
                        factor_family="grid_intensity",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:CEA:GRID_LOCATION:IN:\d{4}$")

    def test_market_based_supplier_contract(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
            jurisdiction="DE",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:TENNET:PPA_DE:2024",
                        factor_family="grid_intensity",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:.*:PPA_.*:\d{4}$")

    def test_market_based_residual_mix_fallback(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
            jurisdiction="NL",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:AIB:RESIDUAL_MIX:NL:2024",
                        factor_family="residual_mix",
                        formula_type="residual_mix",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:AIB:RESIDUAL_MIX:NL:\d{4}$")
