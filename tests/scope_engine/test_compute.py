# -*- coding: utf-8 -*-
"""Scope Engine end-to-end compute tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.scope_engine import ScopeEngineService
from greenlang.scope_engine.models import (
    ActivityRecord,
    ComputationRequest,
    Framework,
    GHGGas,
    GWPBasis,
)


def _req(**overrides):
    base = dict(
        reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        gwp_basis=GWPBasis.AR6_100YR,
        entity_id="ent-test",
        activities=[
            ActivityRecord(
                activity_id="a1",
                activity_type="stationary_combustion",
                fuel_type="diesel",
                quantity=Decimal("1000"),
                unit="gallons",
                year=2024,
            )
        ],
    )
    base.update(overrides)
    return ComputationRequest(**base)


def test_compute_scope_1_diesel():
    svc = ScopeEngineService()
    resp = svc.compute(_req())
    c = resp.computation
    assert c.total_co2e_kg > Decimal(10000)
    assert c.total_co2e_kg < Decimal(11000)  # ~10249 kg for 1000 gal diesel
    assert c.breakdown.scope_1_co2e_kg == c.total_co2e_kg
    assert c.breakdown.scope_2_location_co2e_kg == Decimal(0)
    assert c.breakdown.scope_3_co2e_kg == Decimal(0)


def test_compute_emits_multi_gas_results():
    svc = ScopeEngineService()
    resp = svc.compute(_req())
    gases = {r.gas for r in resp.computation.results}
    assert GHGGas.CO2 in gases
    assert GHGGas.CH4 in gases
    assert GHGGas.N2O in gases


def test_compute_deterministic_hash():
    svc = ScopeEngineService()
    h1 = svc.compute(_req()).computation.computation_hash
    h2 = svc.compute(_req()).computation.computation_hash
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_unique_computation_ids():
    svc = ScopeEngineService()
    r1 = svc.compute(_req()).computation
    r2 = svc.compute(_req()).computation
    assert r1.computation_id != r2.computation_id  # UUIDs differ
    assert r1.computation_hash == r2.computation_hash  # but content hash equal


def test_compute_validates_empty_activities():
    svc = ScopeEngineService()
    with pytest.raises(ValueError, match="activities"):
        svc.compute(
            ComputationRequest(
                reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
                activities=[],
            )
        )


def test_compute_validates_reversed_period():
    svc = ScopeEngineService()
    with pytest.raises(ValueError, match="reporting_period_end"):
        svc.compute(
            _req(
                reporting_period_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
                reporting_period_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )


def test_framework_projection_ghg_protocol():
    svc = ScopeEngineService()
    resp = svc.compute(_req(frameworks=[Framework.GHG_PROTOCOL]))
    assert Framework.GHG_PROTOCOL in resp.framework_views
    view = resp.framework_views[Framework.GHG_PROTOCOL]
    assert len(view.rows) == 4
    scope_1_row = next(r for r in view.rows if r["line"] == "scope_1_total")
    assert Decimal(scope_1_row["co2e_kg"]) > Decimal(10000)


def test_framework_projection_all_five():
    svc = ScopeEngineService()
    frameworks = [
        Framework.GHG_PROTOCOL,
        Framework.ISO_14064,
        Framework.SBTI,
        Framework.CSRD_E1,
        Framework.CBAM,
    ]
    resp = svc.compute(_req(frameworks=frameworks))
    assert set(resp.framework_views.keys()) == set(frameworks)
    assert all(len(v.rows) >= 3 for v in resp.framework_views.values())


def test_gwp_basis_ar5_vs_ar6_differ():
    svc = ScopeEngineService()
    ar5 = svc.compute(_req(gwp_basis=GWPBasis.AR5_100YR)).computation.total_co2e_kg
    ar6 = svc.compute(_req(gwp_basis=GWPBasis.AR6_100YR)).computation.total_co2e_kg
    # Same CO2 mass, different CH4/N2O GWPs — totals should differ slightly
    assert ar5 != ar6
    # Both should be in the same order of magnitude
    ratio = ar5 / ar6
    assert Decimal("0.99") < ratio < Decimal("1.01")


def test_multi_activity_aggregation():
    svc = ScopeEngineService()
    req = _req(
        activities=[
            ActivityRecord(
                activity_id="a1",
                activity_type="stationary_combustion",
                fuel_type="diesel",
                quantity=Decimal("1000"),
                unit="gallons",
                year=2024,
            ),
            ActivityRecord(
                activity_id="a2",
                activity_type="stationary_combustion",
                fuel_type="natural_gas",
                quantity=Decimal("500"),
                unit="therms",
                year=2024,
            ),
        ]
    )
    resp = svc.compute(req)
    # Should have results for both activities
    activity_ids = {r.activity_id for r in resp.computation.results}
    assert activity_ids == {"a1", "a2"}
    # Total equals sum of results
    total = sum(r.co2e_kg for r in resp.computation.results)
    assert resp.computation.total_co2e_kg == total
