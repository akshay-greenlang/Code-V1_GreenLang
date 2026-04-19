# -*- coding: utf-8 -*-
"""CSRD ESRS E1 Climate Change projection.

Maps into ESRS E1-6 (Gross Scopes 1, 2, 3 and Total GHG emissions) disclosure
requirements.
"""

from __future__ import annotations

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation


class CSRDE1Adapter:
    framework = Framework.CSRD_E1

    def project(self, computation: ScopeComputation) -> FrameworkView:
        b = computation.breakdown
        rows = [
            {"esrs_dp": "E1-6 Gross Scope 1 GHG emissions",
             "co2e_t": self._to_tonnes(b.scope_1_co2e_kg)},
            {"esrs_dp": "E1-6 Gross location-based Scope 2 GHG emissions",
             "co2e_t": self._to_tonnes(b.scope_2_location_co2e_kg)},
            {"esrs_dp": "E1-6 Gross market-based Scope 2 GHG emissions",
             "co2e_t": self._to_tonnes(b.scope_2_market_co2e_kg)},
            {"esrs_dp": "E1-6 Total Gross Scope 3 GHG emissions",
             "co2e_t": self._to_tonnes(b.scope_3_co2e_kg)},
            {"esrs_dp": "E1-6 Total GHG emissions (location-based)",
             "co2e_t": self._to_tonnes(
                 b.scope_1_co2e_kg + b.scope_2_location_co2e_kg + b.scope_3_co2e_kg
             )},
            {"esrs_dp": "E1-6 Total GHG emissions (market-based)",
             "co2e_t": self._to_tonnes(
                 b.scope_1_co2e_kg + b.scope_2_market_co2e_kg + b.scope_3_co2e_kg
             )},
        ]
        for cat, value in sorted(b.scope_3_by_category.items()):
            rows.append(
                {"esrs_dp": f"E1-6 Scope 3 Category {cat}",
                 "co2e_t": self._to_tonnes(value)}
            )
        return FrameworkView(
            framework=self.framework,
            rows=rows,
            metadata={"standard": "ESRS E1 Climate Change (CSRD)"},
        )

    @staticmethod
    def _to_tonnes(kg) -> str:
        try:
            return str(kg / 1000)
        except TypeError:
            return "0"
