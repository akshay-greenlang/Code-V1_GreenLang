# -*- coding: utf-8 -*-
"""EU Taxonomy adapter.

Taxonomy alignment is a financial/revenue classification, not a GHG inventory.
This adapter expects revenue breakdown by Taxonomy activity from
data_sources['taxonomy_activities']; delegates full alignment scoring to
applications/GL-Taxonomy-APP.
"""

from __future__ import annotations

import time
from decimal import Decimal

from schemas.models import ComplianceRequest, FrameworkEnum, FrameworkResult
from services.adapters.base import ScopeEngineAdapterBase
from greenlang.schemas.enums import ComplianceStatus


class TaxonomyAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.EU_TAXONOMY

    async def run(self, request: ComplianceRequest) -> FrameworkResult:
        start = time.perf_counter()
        activities = (
            request.data_sources.get("taxonomy_activities", [])
            if request.data_sources
            else []
        )
        if not activities:
            return FrameworkResult(
                framework=self.framework,
                compliance_status=ComplianceStatus.COMPLIANT,
                findings_summary=(
                    "EU Taxonomy alignment requires activity-level revenue data "
                    "(data_sources['taxonomy_activities'])."
                ),
                metrics={"activities_count": 0},
                duration_ms=int((time.perf_counter() - start) * 1000),
            )

        total_turnover = Decimal(0)
        eligible_turnover = Decimal(0)
        aligned_turnover = Decimal(0)
        for a in activities:
            total_turnover += Decimal(str(a.get("turnover_eur", 0)))
            if a.get("eligible"):
                eligible_turnover += Decimal(str(a.get("turnover_eur", 0)))
            if a.get("aligned"):
                aligned_turnover += Decimal(str(a.get("turnover_eur", 0)))

        eligible_pct = (
            float(eligible_turnover / total_turnover) if total_turnover else 0.0
        )
        aligned_pct = (
            float(aligned_turnover / total_turnover) if total_turnover else 0.0
        )
        return FrameworkResult(
            framework=self.framework,
            compliance_status=ComplianceStatus.COMPLIANT,
            findings_summary=(
                f"Taxonomy: {eligible_pct:.1%} eligible, {aligned_pct:.1%} aligned "
                f"across {len(activities)} activities "
                f"(total turnover EUR {float(total_turnover):,.0f})."
            ),
            metrics={
                "activities_count": len(activities),
                "total_turnover_eur": str(total_turnover),
                "eligible_turnover_eur": str(eligible_turnover),
                "aligned_turnover_eur": str(aligned_turnover),
                "eligible_share": str(eligible_pct),
                "aligned_share": str(aligned_pct),
            },
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
