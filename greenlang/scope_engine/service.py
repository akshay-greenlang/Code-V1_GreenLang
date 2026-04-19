# -*- coding: utf-8 -*-
"""ScopeEngineService — main compute orchestrator.

Pipeline: validate -> dispatch -> resolve factor -> per-gas CO2e -> aggregate
-> emit ScopeComputation with SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from greenlang.scope_engine import adapters as framework_adapters
from greenlang.scope_engine import dispatcher
from greenlang.scope_engine.config import ScopeEngineConfig
from greenlang.scope_engine.factor_adapter import FactorAdapter, FactorLookupKey, ResolvedFactor
from greenlang.scope_engine.gwp import gwp_factor
from greenlang.scope_engine.models import (
    ActivityRecord,
    ComputationRequest,
    ComputationResponse,
    EmissionResult,
    Framework,
    FrameworkView,
    GHGGas,
    GWPBasis,
    ScopeBreakdown,
    ScopeComputation,
)

logger = logging.getLogger(__name__)


# Mapping from GHGVectors field name -> GHGGas enum (plural -> singular enum value)
_VECTOR_TO_GAS: dict[str, GHGGas] = {
    "CO2": GHGGas.CO2,
    "CH4": GHGGas.CH4,
    "N2O": GHGGas.N2O,
    "HFCs": GHGGas.HFC,
    "PFCs": GHGGas.PFC,
    "SF6": GHGGas.SF6,
    "NF3": GHGGas.NF3,
}


class ScopeEngineService:
    def __init__(
        self,
        factor_adapter: FactorAdapter | None = None,
        config: ScopeEngineConfig | None = None,
    ) -> None:
        self._factors = factor_adapter or FactorAdapter()
        self._config = config or ScopeEngineConfig()

    def compute(self, request: ComputationRequest) -> ComputationResponse:
        self._validate(request)
        all_results: list[EmissionResult] = []
        for activity in request.activities:
            activity_results = self._compute_activity(activity, request.gwp_basis)
            all_results.extend(activity_results)

        breakdown = self._aggregate(all_results)
        total = sum((r.co2e_kg for r in all_results), Decimal(0))
        computation = ScopeComputation(
            computation_id=str(uuid.uuid4()),
            entity_id=request.entity_id,
            reporting_period_start=request.reporting_period_start,
            reporting_period_end=request.reporting_period_end,
            gwp_basis=request.gwp_basis,
            consolidation=request.consolidation,
            breakdown=breakdown,
            results=all_results,
            total_co2e_kg=total,
            computation_hash=self._hash(request, all_results),
        )

        framework_views: dict[Framework, FrameworkView] = {}
        for framework in request.frameworks:
            try:
                framework_views[framework] = framework_adapters.get(framework).project(
                    computation
                )
            except ValueError:
                logger.warning("No adapter for %s; skipping", framework)
        return ComputationResponse(
            computation=computation, framework_views=framework_views
        )

    # ---- internals ----

    def _validate(self, request: ComputationRequest) -> None:
        if not request.activities:
            raise ValueError("ComputationRequest.activities must not be empty")
        if len(request.activities) > self._config.max_activities_per_request:
            raise ValueError(
                f"Too many activities: {len(request.activities)} > "
                f"{self._config.max_activities_per_request}"
            )
        if request.reporting_period_end < request.reporting_period_start:
            raise ValueError("reporting_period_end must be >= reporting_period_start")

    def _compute_activity(
        self, activity: ActivityRecord, gwp_basis: GWPBasis
    ) -> list[EmissionResult]:
        route = dispatcher.resolve(activity)
        logger.debug("Dispatch %s -> %s", activity.activity_id, route.agent_id)

        # Factor catalog uses fuel_type for lookup; prefer explicit fuel_type,
        # fall back to activity_type (which may itself be a fuel name).
        lookup_name = activity.fuel_type or activity.activity_type
        key = FactorLookupKey(
            activity_type=lookup_name,
            region=activity.region,
            year=activity.year,
            scope=route.scope,
            methodology=activity.methodology,
        )
        factor = self._factors.resolve(key, activity.factor_override_id)
        return self._apply_factor(activity, route, factor, gwp_basis)

    def _apply_factor(
        self,
        activity: ActivityRecord,
        route,
        factor: ResolvedFactor,
        gwp_basis: GWPBasis,
    ) -> list[EmissionResult]:
        quantity = Decimal(str(activity.quantity))
        vectors = getattr(factor.raw_record, "vectors", None)
        if vectors is None:
            raise ValueError(
                f"Factor {factor.factor_id} has no GHG vectors — cannot compute"
            )

        results: list[EmissionResult] = []
        for field_name, gas in _VECTOR_TO_GAS.items():
            raw = getattr(vectors, field_name, None)
            if raw is None:
                continue
            gas_per_unit = Decimal(str(raw))
            if gas_per_unit == 0:
                continue
            gas_amount_kg = quantity * gas_per_unit
            co2e_kg = gas_amount_kg * gwp_factor(gas, gwp_basis)
            results.append(
                EmissionResult(
                    activity_id=activity.activity_id,
                    scope=route.scope,
                    gas=gas,
                    gas_amount=gas_amount_kg,
                    gas_unit="kg",
                    gwp_basis=gwp_basis,
                    co2e_kg=co2e_kg,
                    factor_id=factor.factor_id,
                    factor_source=factor.source,
                    factor_vintage=factor.vintage,
                    formula_hash=_formula_hash(activity, factor, gas, gwp_basis),
                    cached=False,
                )
            )
        return results

    @staticmethod
    def _aggregate(results: list[EmissionResult]) -> ScopeBreakdown:
        breakdown = ScopeBreakdown()
        for r in results:
            if r.scope.value == "1":
                breakdown.scope_1_co2e_kg += r.co2e_kg
            elif r.scope.value == "2":
                breakdown.scope_2_location_co2e_kg += r.co2e_kg
            elif r.scope.value == "3":
                breakdown.scope_3_co2e_kg += r.co2e_kg
        return breakdown

    @staticmethod
    def _hash(request: ComputationRequest, results: list[EmissionResult]) -> str:
        payload = {
            "req": request.model_dump(mode="json"),
            "res": [r.model_dump(mode="json") for r in results],
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


def _formula_hash(
    activity: ActivityRecord,
    factor: ResolvedFactor,
    gas: GHGGas,
    basis: GWPBasis,
) -> str:
    payload = {
        "activity_id": activity.activity_id,
        "quantity": str(activity.quantity),
        "unit": activity.unit,
        "factor_id": factor.factor_id,
        "factor_vintage": factor.vintage,
        "gas": gas.value,
        "gwp_basis": basis.value,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
