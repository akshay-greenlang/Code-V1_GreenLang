# -*- coding: utf-8 -*-
"""MP15 conformance tests — Corporate Inventory method pack.

Five canonical cases exercised against the Corporate Scope 1 / 2-loc /
2-mkt / 3 packs. Each asserts a chosen-factor-id PATTERN, not a specific
value (see conftest.assert_chosen_matches for policy).
"""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import corporate  # noqa: F401
from greenlang.factors.method_packs.exceptions import (
    FactorCannotResolveSafelyError,
)

from tests.factors.method_packs.conformance.conftest import (
    ConformanceRecord,
    assert_chosen_matches,
    resolve_case,
)


class TestCorporateConformance:
    """Conformance — Corporate Scope 1/2/3 packs must resolve canonical cases."""

    def test_scope1_combustion_ng_in_country(self):
        """Natural gas stationary combustion, country-tier factor."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="GB",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:DEFRA:NG_COMBUSTION:GB:2024",
                    ),
                ],
            },
        )
        assert err is None, f"unexpected error: {err!r}"
        assert_chosen_matches(resolved, r"^EF:DEFRA:.*:GB:\d{4}$")

    def test_scope1_combustion_refrigerant_leak(self):
        """Refrigerant leak factor family pulled via facility tier."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
            records_by_step={
                "facility_specific": [
                    ConformanceRecord(
                        factor_id="EF:EPA:REFRIG_R410A:US:2024",
                        factor_family="refrigerant_gwp",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EPA:REFRIG_.*:US:\d{4}$")

    def test_scope2_location_in_eu(self):
        """Location-based Scope 2 grid factor for EU country."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="DE",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:AIB:GRID_LOCATION:DE:2024",
                        factor_family="grid_intensity",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:AIB:GRID_LOCATION:DE:\d{4}$")

    def test_scope2_market_rejects_offset_record(self):
        """Offsets excluded from market-based inventory per Scope 2 Guidance §7.

        The record is tagged as an offset; the SelectionRule MUST reject
        it and surface no-safe-match.
        """
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
            jurisdiction="US",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="OFFSET:VERRA:US:2024",
                        factor_family="grid_intensity",
                        activity_category="carbon_offsets",
                    ),
                ],
            },
        )
        assert resolved is None, (
            "offset record must be rejected from market-based Scope 2"
        )
        assert isinstance(err, FactorCannotResolveSafelyError), (
            f"expected FactorCannotResolveSafelyError, got {err!r}"
        )

    def test_scope3_cat1_material_via_supplier(self):
        """Scope 3 Category 1 (purchased goods) — supplier-specific material."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.CORPORATE_SCOPE3,
            jurisdiction="IN",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:ECOINVENT:STEEL_HOTROLL:IN:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                        activity_category="3.1",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:.*:STEEL_.*:IN:\d{4}$")
