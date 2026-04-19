# -*- coding: utf-8 -*-
"""Shared base for framework adapters.

Most adapters are thin projections over the Unified Scope Engine + framework
views. Base class handles:
- Building ActivityRecord list from ComplianceRequest.data_sources["activities"]
- Invoking ScopeEngineService.compute with the matching Framework
- Wrapping the result into a FrameworkResult
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, ClassVar, Optional

from schemas.models import (
    ComplianceRequest,
    FrameworkEnum,
    FrameworkResult,
)
from greenlang.schemas.enums import ComplianceStatus
from greenlang.scope_engine import ScopeEngineService
from greenlang.scope_engine.models import (
    ActivityRecord,
    ComputationRequest,
    Framework,
    GWPBasis,
)

logger = logging.getLogger(__name__)

# Map Comply-Hub FrameworkEnum -> Scope Engine Framework (for view projection)
_SCOPE_ENGINE_FRAMEWORK: dict[FrameworkEnum, Framework] = {
    FrameworkEnum.GHG_PROTOCOL: Framework.GHG_PROTOCOL,
    FrameworkEnum.ISO_14064: Framework.ISO_14064,
    FrameworkEnum.SBTI: Framework.SBTI,
    FrameworkEnum.CSRD: Framework.CSRD_E1,
    FrameworkEnum.CBAM: Framework.CBAM,
}


class ScopeEngineAdapterBase:
    """Base class for framework adapters that delegate to the Scope Engine.

    Subclasses set ``framework`` (Comply-Hub FrameworkEnum) and optionally
    override ``_extract_metrics`` to surface framework-specific KPIs.
    """

    framework: ClassVar[FrameworkEnum]
    _service: ClassVar[Optional[ScopeEngineService]] = None

    @classmethod
    def _get_service(cls) -> ScopeEngineService:
        if cls._service is None:
            cls._service = ScopeEngineService()
        return cls._service

    async def run(self, request: ComplianceRequest) -> FrameworkResult:
        start = time.perf_counter()
        activities = self._extract_activities(request)
        if not activities:
            return self._pending(
                request,
                "No activity data provided for framework "
                f"{self.framework.value} (missing data_sources['activities'])",
                start,
            )

        engine_framework = _SCOPE_ENGINE_FRAMEWORK.get(self.framework)
        frameworks = [engine_framework] if engine_framework else []

        try:
            response = self._get_service().compute(
                ComputationRequest(
                    reporting_period_start=request.reporting_period_start,
                    reporting_period_end=request.reporting_period_end,
                    gwp_basis=GWPBasis.AR6_100YR,
                    entity_id=request.entity.entity_id,
                    activities=activities,
                    frameworks=frameworks,
                )
            )
        except Exception as exc:
            logger.exception("Scope Engine compute failed for %s", self.framework)
            return self._pending(request, f"Compute error: {exc}", start)

        return self._finalize(request, response, start)

    # ---- subclass hooks ----

    def _finalize(
        self,
        request: ComplianceRequest,
        response: Any,
        start: float,
    ) -> FrameworkResult:
        computation = response.computation
        engine_framework = _SCOPE_ENGINE_FRAMEWORK.get(self.framework)
        view_rows: list = []
        if engine_framework and engine_framework in response.framework_views:
            view_rows = response.framework_views[engine_framework].rows

        metrics = self._extract_metrics(computation, view_rows)
        status = (
            ComplianceStatus.COMPLIANT
            if computation.total_co2e_kg >= Decimal(0)
            else ComplianceStatus.NON_COMPLIANT
        )
        return FrameworkResult(
            framework=self.framework,
            compliance_status=status,
            findings_summary=self._findings(computation, view_rows),
            provenance_hash=computation.computation_hash,
            metrics=metrics,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    def _extract_metrics(self, computation: Any, view_rows: list) -> dict:
        return {
            "total_co2e_kg": str(computation.total_co2e_kg),
            "scope_1_co2e_kg": str(computation.breakdown.scope_1_co2e_kg),
            "scope_2_location_co2e_kg": str(
                computation.breakdown.scope_2_location_co2e_kg
            ),
            "scope_3_co2e_kg": str(computation.breakdown.scope_3_co2e_kg),
            "framework_rows": len(view_rows),
        }

    def _findings(self, computation: Any, view_rows: list) -> str:
        return (
            f"Scope Engine computed {len(computation.results)} gas-level results "
            f"totalling {float(computation.total_co2e_kg):,.2f} kg CO2e; "
            f"{len(view_rows)} rows projected into {self.framework.value}."
        )

    @staticmethod
    def _extract_activities(request: ComplianceRequest) -> list[ActivityRecord]:
        raw = request.data_sources.get("activities", []) if request.data_sources else []
        activities: list[ActivityRecord] = []
        for item in raw:
            try:
                activities.append(ActivityRecord.model_validate(item))
            except Exception as exc:
                logger.warning("Invalid activity skipped: %s (%s)", exc, item)
        return activities

    def _pending(
        self, request: ComplianceRequest, reason: str, start: float
    ) -> FrameworkResult:
        return FrameworkResult(
            framework=self.framework,
            compliance_status=ComplianceStatus.UNDER_REVIEW,
            findings_summary=reason,
            metrics={"reason": reason},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
