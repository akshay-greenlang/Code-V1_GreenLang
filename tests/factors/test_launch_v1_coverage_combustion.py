# -*- coding: utf-8 -*-
"""Launch-v1 coverage — Fuel combustion."""

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


@pytest.mark.parametrize(
    "fuel,jurisdiction",
    [
        ("natural gas", "IN"),
        ("natural gas", "EU"),
        ("natural gas", "US"),
        ("diesel", "IN"),
        ("diesel", "EU"),
        ("diesel", "US"),
        ("coal", "IN"),
        ("coal", "EU"),
        ("LPG", "IN"),
        ("LPG", "EU"),
    ],
)
def test_stationary_combustion(svc, fuel, jurisdiction):
    payload = resolve_or_skip(
        svc,
        activity=f"stationary combustion {fuel}",
        method_profile="corporate_scope1",
        jurisdiction=jurisdiction,
    )
    assert_launch_explain_contract(payload)
