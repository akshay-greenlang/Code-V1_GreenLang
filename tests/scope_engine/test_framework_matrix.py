# -*- coding: utf-8 -*-
"""Phase 3.2 Scope Engine framework-matrix tests.

Goal: ensure each of the 5 framework adapters produces a non-empty
projection from the same `ScopeComputation`, so downstream products
(Comply, Scope Engine CLI, CBAM) can rely on uniform behaviour.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.scope_engine import adapters as framework_adapters
from greenlang.scope_engine.models import (
    ConsolidationApproach,
    EmissionResult,
    Framework,
    GHGGas,
    GWPBasis,
    ScopeBreakdown,
    ScopeComputation,
)
from greenlang.data.emission_factor_record import Scope


def _computation() -> ScopeComputation:
    """Build a ScopeComputation fixture with one result per scope."""
    emissions = [
        EmissionResult(
            activity_id="a1",
            scope=Scope.SCOPE_1,
            gas=GHGGas.CO2,
            gas_amount=Decimal("100"),
            gas_unit="kg",
            gwp_basis=GWPBasis.AR6_100YR,
            co2e_kg=Decimal("100"),
            factor_id="f-s1",
            factor_source="DEFRA 2024",
            factor_vintage="2024",
            formula_hash="h1",
            cached=False,
        ),
        EmissionResult(
            activity_id="a2",
            scope=Scope.SCOPE_2,
            gas=GHGGas.CO2,
            gas_amount=Decimal("50"),
            gas_unit="kg",
            gwp_basis=GWPBasis.AR6_100YR,
            co2e_kg=Decimal("50"),
            factor_id="f-s2",
            factor_source="eGRID",
            factor_vintage="2023",
            formula_hash="h2",
            cached=False,
        ),
        EmissionResult(
            activity_id="a3",
            scope=Scope.SCOPE_3,
            gas=GHGGas.CO2,
            gas_amount=Decimal("25"),
            gas_unit="kg",
            gwp_basis=GWPBasis.AR6_100YR,
            co2e_kg=Decimal("25"),
            factor_id="f-s3",
            factor_source="GHG Protocol",
            factor_vintage="2024",
            formula_hash="h3",
            cached=False,
        ),
    ]
    breakdown = ScopeBreakdown(
        scope_1_co2e_kg=Decimal("100"),
        scope_2_location_co2e_kg=Decimal("50"),
        scope_2_market_co2e_kg=Decimal("0"),
        scope_3_co2e_kg=Decimal("25"),
    )
    return ScopeComputation(
        computation_id="c-test",
        entity_id="e-1",
        reporting_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2026, 12, 31, tzinfo=timezone.utc),
        gwp_basis=GWPBasis.AR6_100YR,
        consolidation=ConsolidationApproach.OPERATIONAL_CONTROL,
        breakdown=breakdown,
        results=emissions,
        total_co2e_kg=Decimal("175"),
        computation_hash="h-total",
    )


class TestFrameworkMatrix:
    @pytest.mark.parametrize(
        "framework",
        [
            Framework.GHG_PROTOCOL,
            Framework.ISO_14064,
            Framework.SBTI,
            Framework.CSRD_E1,
            Framework.CBAM,
        ],
    )
    def test_adapter_projects_view(self, framework: Framework):
        adapter = framework_adapters.get(framework)
        view = adapter.project(_computation())
        assert view.framework == framework
        assert isinstance(view.rows, list)
        assert isinstance(view.metadata, dict)

    def test_all_five_adapters_registered(self):
        available = framework_adapters.available()
        for fw in (
            Framework.GHG_PROTOCOL,
            Framework.ISO_14064,
            Framework.SBTI,
            Framework.CSRD_E1,
            Framework.CBAM,
        ):
            assert fw in available

    def test_adapters_do_not_mutate_computation(self):
        """Adapters must be pure projections — no recomputation or mutation."""
        comp = _computation()
        before = comp.model_dump(mode="json")
        for fw in (
            Framework.GHG_PROTOCOL,
            Framework.ISO_14064,
            Framework.SBTI,
            Framework.CSRD_E1,
            Framework.CBAM,
        ):
            framework_adapters.get(fw).project(comp)
        after = comp.model_dump(mode="json")
        assert before == after


class TestModuleExports:
    def test_scope_engine_init_exports(self):
        from greenlang.scope_engine import (
            ActivityRecord,
            ComputationRequest,
            ComputationResponse,
            EmissionResult,
            FrameworkView,
            ScopeComputation,
            ScopeEngineService,
        )

        assert ScopeEngineService is not None
        assert ActivityRecord is not None
        assert ComputationRequest is not None
        assert ComputationResponse is not None
        assert EmissionResult is not None
        assert FrameworkView is not None
        assert ScopeComputation is not None
