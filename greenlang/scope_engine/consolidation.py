# -*- coding: utf-8 -*-
"""Organizational consolidation layer.

Applies GHG Protocol Corporate Standard consolidation approaches:
- Equity share: multiply each entity's emissions by the reporting org's ownership %
- Operational control: include 100% of emissions from entities the org controls
- Financial control: include 100% of emissions from entities the org financially controls

Full entity_graph wiring lives in SCOPE-ENG 4 (task #24). This module provides
the algorithm; the graph traversal integration is wired during productionization.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from greenlang.scope_engine.models import ConsolidationApproach, EmissionResult


@dataclass(frozen=True)
class EntityOwnership:
    entity_id: str
    ownership_share: Decimal = Decimal(1)
    operational_control: bool = True
    financial_control: bool = True


def apply_consolidation(
    results: list[EmissionResult],
    ownerships: dict[str, EntityOwnership],
    approach: ConsolidationApproach,
) -> list[EmissionResult]:
    """Return a new list of results with co2e_kg scaled by consolidation rules.

    If ``ownerships`` is empty, returns ``results`` unchanged (single-entity case).
    """
    if not ownerships:
        return list(results)

    adjusted: list[EmissionResult] = []
    for r in results:
        # Activity-level entity is carried via EmissionResult.activity_id -> external
        # mapping (resolved by the caller). For now we look up by a metadata marker
        # if present; otherwise assume the default ownership '__self__'.
        entity_id = getattr(r, "_entity_id", None) or "__self__"
        ownership = ownerships.get(entity_id, EntityOwnership(entity_id))
        factor = _consolidation_factor(ownership, approach)
        if factor == Decimal(1):
            adjusted.append(r)
            continue
        adjusted.append(
            r.model_copy(
                update={
                    "co2e_kg": r.co2e_kg * factor,
                    "gas_amount": r.gas_amount * factor,
                }
            )
        )
    return adjusted


def _consolidation_factor(
    ownership: EntityOwnership, approach: ConsolidationApproach
) -> Decimal:
    if approach == ConsolidationApproach.EQUITY_SHARE:
        return Decimal(ownership.ownership_share)
    if approach == ConsolidationApproach.OPERATIONAL_CONTROL:
        return Decimal(1) if ownership.operational_control else Decimal(0)
    if approach == ConsolidationApproach.FINANCIAL_CONTROL:
        return Decimal(1) if ownership.financial_control else Decimal(0)
    return Decimal(1)
