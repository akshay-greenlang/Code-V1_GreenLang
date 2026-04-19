# -*- coding: utf-8 -*-
"""GHG Protocol Corporate Standard projection.

Emits rows matching the GHG Protocol Corporate Standard Chapter 9 disclosure
format: Scope 1, Scope 2 (location + market), Scope 3 by category.
"""

from __future__ import annotations

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation


class GHGProtocolAdapter:
    framework = Framework.GHG_PROTOCOL

    def project(self, computation: ScopeComputation) -> FrameworkView:
        b = computation.breakdown
        rows: list[dict] = [
            {
                "line": "scope_1_total",
                "co2e_kg": str(b.scope_1_co2e_kg),
                "unit": "kg CO2e",
            },
            {
                "line": "scope_2_location_based",
                "co2e_kg": str(b.scope_2_location_co2e_kg),
                "unit": "kg CO2e",
            },
            {
                "line": "scope_2_market_based",
                "co2e_kg": str(b.scope_2_market_co2e_kg),
                "unit": "kg CO2e",
            },
            {
                "line": "scope_3_total",
                "co2e_kg": str(b.scope_3_co2e_kg),
                "unit": "kg CO2e",
            },
        ]
        for cat, value in sorted(b.scope_3_by_category.items()):
            rows.append(
                {
                    "line": f"scope_3_cat_{cat:02d}",
                    "co2e_kg": str(value),
                    "unit": "kg CO2e",
                }
            )
        return FrameworkView(
            framework=self.framework,
            rows=rows,
            metadata={
                "standard": "GHG Protocol Corporate Standard (Revised)",
                "gwp_basis": computation.gwp_basis.value,
                "consolidation": computation.consolidation.value,
            },
        )
