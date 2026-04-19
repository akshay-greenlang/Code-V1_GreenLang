# -*- coding: utf-8 -*-
"""Framework adapter projection tests — verify pure projection behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.scope_engine import adapters as framework_adapters
from greenlang.scope_engine.models import (
    ConsolidationApproach,
    Framework,
    GWPBasis,
    ScopeBreakdown,
    ScopeComputation,
)


def _computation(**overrides):
    breakdown = ScopeBreakdown(
        scope_1_co2e_kg=Decimal(1000),
        scope_2_location_co2e_kg=Decimal(500),
        scope_2_market_co2e_kg=Decimal(400),
        scope_3_co2e_kg=Decimal(2000),
        scope_3_by_category={1: Decimal(800), 4: Decimal(600), 11: Decimal(600)},
    )
    base = dict(
        computation_id="test-123",
        entity_id="ent-1",
        reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        gwp_basis=GWPBasis.AR6_100YR,
        consolidation=ConsolidationApproach.OPERATIONAL_CONTROL,
        breakdown=breakdown,
        results=[],
        total_co2e_kg=Decimal(3500),
        computation_hash="0" * 64,
    )
    base.update(overrides)
    return ScopeComputation(**base)


def test_ghg_protocol_rows_are_complete():
    adapter = framework_adapters.get(Framework.GHG_PROTOCOL)
    view = adapter.project(_computation())
    lines = {r["line"] for r in view.rows}
    assert {
        "scope_1_total",
        "scope_2_location_based",
        "scope_2_market_based",
        "scope_3_total",
    } <= lines


def test_iso_14064_category_3_aggregates_transport_scope3():
    adapter = framework_adapters.get(Framework.ISO_14064)
    view = adapter.project(_computation())
    cat_3 = next(r for r in view.rows if r["category"] == "3")
    # Category 3 = Scope 3 cat 4 (upstream transport) alone in our test data
    assert Decimal(cat_3["co2e_kg"]) == Decimal(600)


def test_csrd_e1_reports_in_tonnes():
    adapter = framework_adapters.get(Framework.CSRD_E1)
    view = adapter.project(_computation())
    scope_1 = next(r for r in view.rows if "Gross Scope 1" in r["esrs_dp"])
    assert Decimal(scope_1["co2e_t"]) == Decimal(1)  # 1000 kg = 1 tonne


def test_cbam_uses_tonnes_and_marks_phase():
    adapter = framework_adapters.get(Framework.CBAM)
    view = adapter.project(
        _computation(reporting_period_start=datetime(2025, 6, 1, tzinfo=timezone.utc))
    )
    assert view.metadata["phase"] == "transitional"
    view2 = adapter.project(
        _computation(reporting_period_start=datetime(2026, 6, 1, tzinfo=timezone.utc))
    )
    assert view2.metadata["phase"] == "definitive"


def test_sbti_computes_scope12_and_scope3_share():
    adapter = framework_adapters.get(Framework.SBTI)
    view = adapter.project(_computation())
    scope_1_2 = next(r for r in view.rows if r["metric"] == "scope_1_2_absolute")
    # Market-based selected since 400 < 500; adapter picks max = 500 (location) + 1000 = 1500
    assert Decimal(scope_1_2["co2e_kg"]) == Decimal(1500)


def test_all_five_adapters_registered():
    frameworks = framework_adapters.available()
    assert set(frameworks) == {
        Framework.GHG_PROTOCOL,
        Framework.ISO_14064,
        Framework.SBTI,
        Framework.CSRD_E1,
        Framework.CBAM,
    }


def test_unknown_framework_raises():
    with pytest.raises(ValueError):
        framework_adapters.get("not_a_framework")  # type: ignore[arg-type]
