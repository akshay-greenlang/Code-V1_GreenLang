# -*- coding: utf-8 -*-
"""MP15 conformance tests — EU Policy method packs (CBAM / DPP / Battery)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import eu_policy  # noqa: F401

from tests.factors.method_packs.conformance.conftest import (
    ConformanceRecord,
    assert_chosen_matches,
    resolve_case,
)


def _verified_record(fid: str, **kwargs) -> ConformanceRecord:
    """Build a ConformanceRecord with a ``verification.status=external_verified``
    sidecar — required by CBAM / Battery packs' SelectionRule."""
    rec = ConformanceRecord(factor_id=fid, **kwargs)
    # attach a SimpleNamespace so getattr(record, "verification").status works
    object.__setattr__(
        rec,
        "verification",
        SimpleNamespace(status="external_verified"),
    )
    return rec


class TestEUPolicyConformance:
    """Conformance — EU CBAM / DPP / Battery regulated packs."""

    def test_cbam_steel_cn_code(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.EU_CBAM,
            jurisdiction="DE",
            records_by_step={
                "supplier_specific": [
                    _verified_record(
                        fid="EF:EUCBAM:STEEL_HR_CN72:DE:2024",
                        factor_family="material_embodied",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EUCBAM:.*:DE:\d{4}$")

    def test_cbam_cement_cn_code(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.EU_CBAM,
            jurisdiction="ES",
            records_by_step={
                "supplier_specific": [
                    _verified_record(
                        fid="EF:EUCBAM:CEMENT_CN2523:ES:2024",
                        factor_family="material_embodied",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EUCBAM:CEMENT_.*:ES:\d{4}$")

    def test_eu_dpp_generic_product(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.EU_DPP,
            jurisdiction="FR",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:ESPR:ELECTRONICS:FR:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:ESPR:.*:FR:\d{4}$")

    def test_eu_battery_regulation(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.EU_DPP_BATTERY,
            jurisdiction="FR",
            records_by_step={
                "supplier_specific": [
                    _verified_record(
                        fid="EF:EUBAT:LIB_NMC:FR:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EUBAT:.*:FR:\d{4}$")
