# -*- coding: utf-8 -*-
"""EUDR deforestation-compliance adapter.

EUDR is NOT a GHG inventory framework — it concerns deforestation traceability
for imported commodities. This adapter delegates to applications/GL-EUDR-APP
pipeline if available; otherwise emits a PENDING result indicating the dedicated
app must run.
"""

from __future__ import annotations

import time

from schemas.models import ComplianceRequest, FrameworkEnum, FrameworkResult
from services.adapters.base import ScopeEngineAdapterBase
from greenlang.schemas.enums import ComplianceStatus


class EUDRAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.EUDR

    async def run(self, request: ComplianceRequest) -> FrameworkResult:
        start = time.perf_counter()
        commodities = (
            request.data_sources.get("eudr_commodities", [])
            if request.data_sources
            else []
        )
        if not commodities:
            return FrameworkResult(
                framework=self.framework,
                compliance_status=ComplianceStatus.UNDER_REVIEW,
                findings_summary=(
                    "EUDR requires commodity-level traceability data "
                    "(data_sources['eudr_commodities']). None provided."
                ),
                metrics={"commodities_count": 0},
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
        # Full EUDR check lives in GL-EUDR-APP; this adapter returns a summary
        return FrameworkResult(
            framework=self.framework,
            compliance_status=ComplianceStatus.UNDER_REVIEW,
            findings_summary=(
                f"Received {len(commodities)} commodity declarations. "
                "Full EUDR due-diligence check requires GL-EUDR-APP pipeline "
                "(satellite deforestation check + geolocation verification)."
            ),
            metrics={"commodities_count": len(commodities)},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
