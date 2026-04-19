# -*- coding: utf-8 -*-
"""California SB 253 adapter.

SB 253 requires covered entities to disclose Scope 1/2/3 emissions annually
using GHG Protocol methodology. This adapter uses the GHG Protocol view
then surfaces SB 253-specific metrics (revenue threshold, disclosure year).
"""

from __future__ import annotations

from typing import Any

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class SB253Adapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.SB253

    def _extract_metrics(self, computation: Any, view_rows: list) -> dict:
        metrics = super()._extract_metrics(computation, view_rows)
        metrics.update(
            {
                "sb253_scope_1_tonnes": str(
                    computation.breakdown.scope_1_co2e_kg / 1000
                ),
                "sb253_scope_2_tonnes": str(
                    max(
                        computation.breakdown.scope_2_market_co2e_kg,
                        computation.breakdown.scope_2_location_co2e_kg,
                    )
                    / 1000
                ),
                "sb253_scope_3_tonnes": str(
                    computation.breakdown.scope_3_co2e_kg / 1000
                ),
                "sb253_disclosure_year": computation.reporting_period_start.year,
            }
        )
        return metrics
