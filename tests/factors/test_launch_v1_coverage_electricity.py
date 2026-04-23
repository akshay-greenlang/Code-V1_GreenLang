# -*- coding: utf-8 -*-
"""Launch-v1 coverage matrix — Electricity (Track B-4).

CTO launch scope: India + EU/UK + US, location-based + market-based +
residual-mix paths. Each test resolves a representative activity through
the full 7-step cascade and asserts the explain payload includes source,
version, fallback rank, and method profile.
"""

from __future__ import annotations

import pytest

from tests.factors._launch_v1_helpers import (
    assert_launch_explain_contract,
    get_service_or_skip,
    resolve_or_skip,
)

pytestmark = pytest.mark.launch_v1


@pytest.fixture(scope="module")
def svc():
    return get_service_or_skip()


def test_india_grid_average_location_based(svc):
    payload = resolve_or_skip(
        svc, activity="purchased grid electricity",
        method_profile="corporate_scope2_location_based",
        jurisdiction="IN",
    )
    assert_launch_explain_contract(payload)


def test_india_grid_market_based(svc):
    payload = resolve_or_skip(
        svc, activity="purchased grid electricity",
        method_profile="corporate_scope2_market_based",
        jurisdiction="IN",
    )
    assert_launch_explain_contract(payload)


def test_eu_residual_mix(svc):
    payload = resolve_or_skip(
        svc, activity="purchased grid electricity",
        method_profile="corporate_scope2_market_based",
        jurisdiction="DE",
    )
    assert_launch_explain_contract(payload)


def test_uk_grid_desnz(svc):
    payload = resolve_or_skip(
        svc, activity="purchased grid electricity",
        method_profile="corporate_scope2_location_based",
        jurisdiction="UK",
    )
    assert_launch_explain_contract(payload)


def test_us_egrid_subregion(svc):
    payload = resolve_or_skip(
        svc, activity="purchased grid electricity",
        method_profile="corporate_scope2_location_based",
        jurisdiction="US",
    )
    assert_launch_explain_contract(payload)
