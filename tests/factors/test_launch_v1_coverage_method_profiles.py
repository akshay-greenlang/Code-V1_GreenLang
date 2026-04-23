# -*- coding: utf-8 -*-
"""Launch-v1 coverage — method profile routing.

Asserts that the launch-blocking method profiles all resolve through the
engine. This is what makes Track B-6 (CTO non-negotiable #6: policy
workflows must call method profiles, not raw factors) a pass-able bar.
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


@pytest.mark.parametrize(
    "method_profile,activity,jurisdiction",
    [
        ("corporate_scope1", "natural gas", "EU"),
        ("corporate_scope2_location_based", "purchased grid electricity", "IN"),
        ("corporate_scope2_market_based", "purchased grid electricity", "EU"),
        ("corporate_scope3_purchased_goods", "purchased steel", "GLOBAL"),
        ("freight_iso14083", "road freight HGV", "EU"),
        ("eu_policy_cbam", "purchased steel", "EU"),
    ],
)
def test_method_profile_routes(svc, method_profile, activity, jurisdiction):
    payload = resolve_or_skip(
        svc, activity=activity, method_profile=method_profile,
        jurisdiction=jurisdiction,
    )
    assert_launch_explain_contract(payload)
    assert payload.get("method_profile") == method_profile
