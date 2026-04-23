# -*- coding: utf-8 -*-
"""MP15 conformance tests — Product Carbon method pack (v0.2, Wave 4-G).

Canonical cases: cradle-to-gate steel, PAS 2050 finished product, PEF
climate-change indicator, OEF entity-scope. The tests intentionally do
NOT check exact CO2e values — only the chosen-factor-id PATTERN —
because this is a conformance gate, not a numeric regression gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import (  # noqa: F401
    product_carbon,
    product_lca_variants,
)
from greenlang.factors.method_packs.exceptions import (
    FactorCannotResolveSafelyError,
)

from tests.factors.method_packs.conformance.conftest import (
    ConformanceRecord,
    assert_chosen_matches,
    resolve_case,
)


def _verified_record(fid: str, **kwargs) -> ConformanceRecord:
    """Build a ConformanceRecord with verification.status=external_verified."""
    rec = ConformanceRecord(factor_id=fid, **kwargs)
    object.__setattr__(
        rec,
        "verification",
        SimpleNamespace(status="external_verified"),
    )
    return rec


class TestProductCarbonConformance:
    """Conformance — PRODUCT_CARBON + PAS_2050 + PEF + OEF variants."""

    def test_cradle_to_gate_steel(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.PRODUCT_CARBON,
            jurisdiction="DE",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:WSA:STEEL_CRADLE_TO_GATE:DE:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:WSA:STEEL_.*:DE:\d{4}$")

    def test_cradle_to_grave_consumer_good(self):
        resolved, err = resolve_case(
            method_profile=MethodProfile.PRODUCT_CARBON,
            jurisdiction="US",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:ECOINVENT:CONSUMER_GOOD:US:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:ECOINVENT:.*:US:\d{4}$")

    def test_product_pack_rejects_scope1_record(self):
        """Product pack MUST reject scope1_direct_emissions via exclusion list."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.PRODUCT_CARBON,
            jurisdiction="GB",
            records_by_step={
                "supplier_specific": [
                    ConformanceRecord(
                        factor_id="EF:DEFRA:NG_COMBUSTION:GB:2024",
                        factor_family="emissions",
                        formula_type="lca",
                        activity_category="scope1_direct_emissions",
                    ),
                ],
            },
        )
        assert resolved is None, (
            "product pack must reject scope1_direct_emissions"
        )
        assert isinstance(err, FactorCannotResolveSafelyError), (
            f"expected FactorCannotResolveSafelyError, got {err!r}"
        )

    def test_pef_requires_verification(self):
        """PEF pack has require_verification=True — unverified record fails."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.PRODUCT_CARBON,
            jurisdiction="FR",
            records_by_step={
                "supplier_specific": [
                    _verified_record(
                        fid="EF:EF3_1:PEFCR_CONSUMER_ELECTRONICS:FR:2024",
                        factor_family="material_embodied",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:EF3_1:PEFCR_.*:FR:\d{4}$")

    def test_cradle_to_gate_energy_conversion(self):
        """Energy-conversion factor family admitted via LCA formula."""
        resolved, err = resolve_case(
            method_profile=MethodProfile.PRODUCT_CARBON,
            jurisdiction="CN",
            records_by_step={
                "country_or_sector_average": [
                    ConformanceRecord(
                        factor_id="EF:ECOINVENT:THERMAL_ENERGY:CN:2024",
                        factor_family="energy_conversion",
                        formula_type="lca",
                    ),
                ],
            },
        )
        assert err is None
        assert_chosen_matches(resolved, r"^EF:ECOINVENT:.*:CN:\d{4}$")
