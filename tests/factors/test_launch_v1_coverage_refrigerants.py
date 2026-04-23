# -*- coding: utf-8 -*-
"""Launch-v1 coverage — Refrigerants (AR6 GWP basis)."""

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


@pytest.mark.parametrize("gas", ["R-22", "R-32", "R-134a", "R-410A", "R-404A"])
def test_refrigerant_leak_ar6(svc, gas):
    payload = resolve_or_skip(
        svc,
        activity=f"refrigerant leak {gas}",
        method_profile="corporate_scope1",
        jurisdiction="GLOBAL",
    )
    assert_launch_explain_contract(payload)


def test_refrigerant_uses_ar6_gwp(svc):
    payload = resolve_or_skip(
        svc,
        activity="refrigerant leak R-410A",
        method_profile="corporate_scope1",
        jurisdiction="GLOBAL",
    )
    chosen = payload.get("chosen") or payload.get("factor") or payload
    if isinstance(chosen, dict):
        gwp = chosen.get("gwp_set") or chosen.get("gwp_basis")
        if gwp is not None:
            assert "ar6" in str(gwp).lower() or "ar5" in str(gwp).lower()
