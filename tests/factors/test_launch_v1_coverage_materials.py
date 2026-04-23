# -*- coding: utf-8 -*-
"""Launch-v1 coverage — Purchased goods proxies."""

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
    "material",
    ["steel", "aluminium", "cement", "fertilizer", "plastics", "paper"],
)
def test_purchased_goods_proxy(svc, material):
    payload = resolve_or_skip(
        svc,
        activity=f"purchased {material}",
        method_profile="corporate_scope3_purchased_goods",
        jurisdiction="GLOBAL",
    )
    assert_launch_explain_contract(payload)
