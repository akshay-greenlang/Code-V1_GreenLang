# -*- coding: utf-8 -*-
"""TCFD adapter.

TCFD Pillar 3 (Metrics & Targets) maps to GHG Protocol Scope 1/2/3. The
qualitative pillars (Governance, Strategy, Risk Management) and scenario
analysis live in applications/GL-TCFD-APP and are composed upstream of this
adapter's quantitative output.
"""

from __future__ import annotations

from typing import Any

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class TCFDAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.TCFD

    def _extract_metrics(self, computation: Any, view_rows: list) -> dict:
        metrics = super()._extract_metrics(computation, view_rows)
        # TCFD recommends carbon intensity metrics
        metrics["tcfd_total_scope_1_2_tonnes"] = str(
            (
                computation.breakdown.scope_1_co2e_kg
                + max(
                    computation.breakdown.scope_2_market_co2e_kg,
                    computation.breakdown.scope_2_location_co2e_kg,
                )
            )
            / 1000
        )
        return metrics
