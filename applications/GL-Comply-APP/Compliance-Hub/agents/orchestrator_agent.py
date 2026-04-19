# -*- coding: utf-8 -*-
"""Compliance orchestrator agent.

Production orchestration:
- Parallel async dispatch to framework adapters
- Per-adapter timeouts (configurable)
- Partial-failure semantics: successes recorded, failures surfaced in gap_analysis
- Overall status computed from per-framework results
- SHA-256 aggregate provenance hash
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone

from schemas.models import (
    ComplianceRequest,
    FrameworkEnum,
    FrameworkResult,
    UnifiedComplianceReport,
)
from services import normalizer, registry
from greenlang.schemas.enums import ComplianceStatus, ReportFormat

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_TIMEOUT_SECONDS = 120.0


class ComplianceOrchestrator:
    def __init__(self, adapter_timeout_seconds: float = DEFAULT_ADAPTER_TIMEOUT_SECONDS):
        self._timeout = adapter_timeout_seconds

    async def run(self, request: ComplianceRequest) -> UnifiedComplianceReport:
        # Normalize first (entity resolution, activity dedup, unit coercion)
        request = normalizer.normalize(request)

        tasks = [
            asyncio.wait_for(self._run_framework(fw, request), timeout=self._timeout)
            for fw in request.frameworks
        ]
        framework_outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[FrameworkEnum, FrameworkResult] = {}
        gap_analysis: list[dict] = []
        for fw, outcome in zip(request.frameworks, framework_outcomes):
            if isinstance(outcome, asyncio.TimeoutError):
                gap_analysis.append(
                    {
                        "framework": fw.value,
                        "severity": "error",
                        "reason": f"Adapter timed out after {self._timeout}s",
                    }
                )
                logger.error("Adapter %s timed out", fw)
                continue
            if isinstance(outcome, Exception):
                gap_analysis.append(
                    {
                        "framework": fw.value,
                        "severity": "error",
                        "reason": f"{type(outcome).__name__}: {outcome}",
                    }
                )
                logger.exception("Adapter %s raised", fw, exc_info=outcome)
                continue
            results[fw] = outcome  # type: ignore[assignment]
            if outcome.compliance_status in (
                ComplianceStatus.NON_COMPLIANT,
                ComplianceStatus.UNDER_REVIEW,
            ):
                gap_analysis.append(
                    {
                        "framework": fw.value,
                        "severity": "warning"
                        if outcome.compliance_status == ComplianceStatus.UNDER_REVIEW
                        else "error",
                        "reason": outcome.findings_summary,
                    }
                )

        overall_status = self._compute_overall_status(results)

        return UnifiedComplianceReport(
            job_id=str(uuid.uuid4()),
            entity_id=request.entity.entity_id,
            reporting_period_start=request.reporting_period_start,
            reporting_period_end=request.reporting_period_end,
            frameworks=list(results.keys()),
            results=results,
            overall_status=overall_status,
            gap_analysis=gap_analysis,
            report_format=ReportFormat.PDF,
            aggregate_provenance_hash=self._aggregate_hash(request, results),
        )

    async def _run_framework(
        self, framework: FrameworkEnum, request: ComplianceRequest
    ) -> FrameworkResult:
        adapter = registry.get(framework)
        return await adapter.run(request)

    @staticmethod
    def _compute_overall_status(
        results: dict[FrameworkEnum, FrameworkResult],
    ) -> ComplianceStatus:
        if not results:
            return ComplianceStatus.NOT_ASSESSED
        counts = Counter(r.compliance_status for r in results.values())
        if counts[ComplianceStatus.NON_COMPLIANT] > 0:
            return ComplianceStatus.NON_COMPLIANT
        if (
            counts[ComplianceStatus.UNDER_REVIEW] > 0
            or counts[ComplianceStatus.PARTIALLY_COMPLIANT] > 0
        ):
            return ComplianceStatus.PARTIALLY_COMPLIANT
        if counts[ComplianceStatus.COMPLIANT] == len(results):
            return ComplianceStatus.COMPLIANT
        return ComplianceStatus.PARTIALLY_COMPLIANT

    @staticmethod
    def _aggregate_hash(
        request: ComplianceRequest,
        results: dict[FrameworkEnum, FrameworkResult],
    ) -> str:
        """Deterministic aggregate hash over request + framework result content.

        Excludes wall-clock timestamps (created_at, updated_at) and generated
        IDs so the same logical computation produces the same hash across runs.
        """
        def _strip(d):
            if isinstance(d, dict):
                return {
                    k: _strip(v)
                    for k, v in d.items()
                    if k not in _NON_DETERMINISTIC_FIELDS
                }
            if isinstance(d, list):
                return [_strip(x) for x in d]
            return d

        payload = {
            "req": _strip(request.model_dump(mode="json")),
            "res": {
                k.value: _strip(v.model_dump(mode="json")) for k, v in results.items()
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


_NON_DETERMINISTIC_FIELDS = frozenset(
    {
        "created_at",
        "updated_at",
        "timestamp",
        "id",
        "record_id",
        "job_id",
        "duration_ms",
    }
)
