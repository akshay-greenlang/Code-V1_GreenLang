# -*- coding: utf-8 -*-
"""GreenLang Unified Scope Engine — Scope 1/2/3 GHG computation across frameworks.

Entry points:
- ScopeEngineService.compute(request) -> ScopeComputation
- FrameworkView adapters (ghg_protocol, iso_14064, sbti, csrd_e1, cbam)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ScopeEngineService",
    "ActivityRecord",
    "EmissionResult",
    "ScopeComputation",
    "ComputationRequest",
    "ComputationResponse",
    "FrameworkView",
    "ScopeEngineError",
]


def __getattr__(name: str) -> Any:
    if name == "ScopeEngineService":
        from greenlang.scope_engine.service import ScopeEngineService

        return ScopeEngineService
    if name in {
        "ActivityRecord",
        "EmissionResult",
        "ScopeComputation",
        "ComputationRequest",
        "ComputationResponse",
        "FrameworkView",
    }:
        from greenlang.scope_engine import models

        return getattr(models, name)
    if name == "ScopeEngineError":
        from greenlang.exceptions.calculation import CalculationError

        return CalculationError
    raise AttributeError(name)
