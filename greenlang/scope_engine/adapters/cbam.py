# -*- coding: utf-8 -*-
"""CBAM embedded emissions projection.

CBAM reports direct + indirect embedded emissions on CBAM-scope goods at the
import boundary. This adapter projects the engine result into the CBAM quarterly
report line format. Upstream caller must filter activities to CBAM-scope goods
before compute.
"""

from __future__ import annotations

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation


class CBAMAdapter:
    framework = Framework.CBAM

    def project(self, computation: ScopeComputation) -> FrameworkView:
        b = computation.breakdown
        # CBAM: direct emissions map to Scope 1; indirect (electricity) map to Scope 2
        rows = [
            {"line": "direct_embedded_emissions",
             "co2e_t": self._to_tonnes(b.scope_1_co2e_kg),
             "boundary": "direct"},
            {"line": "indirect_embedded_emissions",
             "co2e_t": self._to_tonnes(b.scope_2_location_co2e_kg),
             "boundary": "indirect_electricity"},
            {"line": "total_embedded_emissions",
             "co2e_t": self._to_tonnes(
                 b.scope_1_co2e_kg + b.scope_2_location_co2e_kg
             ),
             "boundary": "total"},
        ]
        return FrameworkView(
            framework=self.framework,
            rows=rows,
            metadata={
                "regulation": "Regulation (EU) 2023/956 (CBAM)",
                "phase": "transitional" if computation.reporting_period_start.year < 2026 else "definitive",
            },
        )

    @staticmethod
    def _to_tonnes(kg) -> str:
        try:
            return str(kg / 1000)
        except TypeError:
            return "0"
