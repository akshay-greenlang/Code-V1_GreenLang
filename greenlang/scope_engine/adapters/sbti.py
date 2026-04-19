# -*- coding: utf-8 -*-
"""SBTi target baseline projection."""

from __future__ import annotations

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation


class SBTiAdapter:
    framework = Framework.SBTI

    def project(self, computation: ScopeComputation) -> FrameworkView:
        b = computation.breakdown
        scope_1_2 = b.scope_1_co2e_kg + max(
            b.scope_2_market_co2e_kg, b.scope_2_location_co2e_kg
        )
        rows = [
            {"metric": "scope_1_2_absolute",
             "co2e_kg": str(scope_1_2),
             "method": "market_based_if_available_else_location"},
            {"metric": "scope_3_absolute",
             "co2e_kg": str(b.scope_3_co2e_kg)},
            {"metric": "scope_3_share_of_total",
             "value": self._share(b.scope_3_co2e_kg, scope_1_2 + b.scope_3_co2e_kg),
             "unit": "fraction"},
            {"metric": "total_co2e_absolute",
             "co2e_kg": str(computation.total_co2e_kg)},
        ]
        return FrameworkView(
            framework=self.framework,
            rows=rows,
            metadata={"standard": "SBTi Corporate Net-Zero Standard v1.2"},
        )

    @staticmethod
    def _share(numer, denom) -> str:
        if not denom:
            return "0"
        try:
            return str(numer / denom)
        except (ZeroDivisionError, TypeError):
            return "0"
