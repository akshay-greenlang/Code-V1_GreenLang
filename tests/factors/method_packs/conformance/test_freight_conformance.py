# -*- coding: utf-8 -*-
"""MP15 conformance tests — Freight ISO 14083 / GLEC method pack.

Canonical cases cover each of road / sea / air / rail modes plus a WTW
multi-leg scenario.
"""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import freight  # noqa: F401

from tests.factors.method_packs.conformance.conftest import (
    ConformanceRecord,
    assert_chosen_matches,
    resolve_case,
)


class TestFreightConformance:
    """Conformance — FREIGHT_ISO_14083 pack must resolve mode-specific lanes."""

    def test_road_diesel_wtw(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.FREIGHT_ISO_14083,
            jurisdiction="DE",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:GLEC:ROAD_DIESEL_WTW:DE:2024",
                        factor_family="transport_lane",
                        formula_type="transport_chain",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:GLEC:ROAD_.*:DE:\d{4}$")

    def test_sea_container_vessel(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.FREIGHT_ISO_14083,
            jurisdiction="SG",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:GLEC:SEA_CONTAINER_5000TEU:SG:2024",
                        factor_family="transport_lane",
                        formula_type="transport_chain",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:GLEC:SEA_.*:SG:\d{4}$")

    def test_air_belly_freight(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.FREIGHT_ISO_14083,
            jurisdiction="US",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:ICAO:AIR_BELLY_MED:US:2024",
                        factor_family="transport_lane",
                        formula_type="transport_chain",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:ICAO:AIR_.*:US:\d{4}$")

    def test_rail_electric_eu(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.FREIGHT_ISO_14083,
            jurisdiction="FR",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:GLEC:RAIL_ELECTRIC:FR:2024",
                        factor_family="transport_lane",
                        formula_type="transport_chain",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:GLEC:RAIL_.*:FR:\d{4}$")
