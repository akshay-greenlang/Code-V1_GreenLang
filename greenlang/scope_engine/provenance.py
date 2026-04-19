# -*- coding: utf-8 -*-
"""Scope Engine provenance — emits ledger entries per computation + per result.

Wraps greenlang.climate_ledger.ClimateLedger. Optional: if the ledger import
fails (minimal install), provenance emission is a no-op so compute never breaks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.scope_engine.models import EmissionResult, ScopeComputation

logger = logging.getLogger(__name__)


class _NullLedger:
    def record_entry(self, **kwargs) -> str:
        return "null"


def build_ledger() -> Any:
    try:
        from greenlang.climate_ledger import ClimateLedger

        return ClimateLedger()
    except Exception as exc:  # pragma: no cover
        logger.debug("ClimateLedger unavailable, using null ledger: %s", exc)
        return _NullLedger()


class ProvenanceRecorder:
    def __init__(self, ledger: Optional[Any] = None, enabled: bool = True) -> None:
        self._ledger = ledger or build_ledger()
        self._enabled = enabled

    def record_computation(
        self, computation: ScopeComputation, metadata: Optional[dict] = None
    ) -> str:
        if not self._enabled:
            return ""
        try:
            return self._ledger.record_entry(
                entity_type="scope_computation",
                entity_id=computation.computation_id,
                operation="compute",
                content_hash=computation.computation_hash,
                metadata={
                    "entity": computation.entity_id,
                    "gwp_basis": computation.gwp_basis.value,
                    "consolidation": computation.consolidation.value,
                    "total_co2e_kg": str(computation.total_co2e_kg),
                    **(metadata or {}),
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Provenance recording failed: %s", exc)
            return ""

    def record_result(self, computation_id: str, result: EmissionResult) -> str:
        if not self._enabled:
            return ""
        try:
            return self._ledger.record_entry(
                entity_type="emission",
                entity_id=f"{computation_id}:{result.activity_id}:{result.gas.value}",
                operation="calculate",
                content_hash=result.formula_hash,
                metadata={
                    "scope": result.scope.value,
                    "gas": result.gas.value,
                    "factor_id": result.factor_id,
                    "factor_source": result.factor_source,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Result provenance failed: %s", exc)
            return ""
