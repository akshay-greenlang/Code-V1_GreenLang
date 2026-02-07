# -*- coding: utf-8 -*-
"""
Compliance Metrics Collector - SEC-009 Phase 9

Calculates and aggregates compliance metrics for SOC 2 Type II dashboards.
Provides readiness scores, evidence coverage, test pass rates, finding metrics,
SLA compliance, and attestation status.

Classes:
    - EvidenceCoverage: Evidence collection metrics
    - TestMetrics: Control test result metrics
    - FindingMetrics: Audit finding metrics
    - SLAMetrics: Auditor request SLA metrics
    - AttestationMetrics: Attestation status metrics
    - DashboardSummary: Combined dashboard metrics
    - ComplianceMetrics: Main metrics calculation class

Example:
    >>> metrics = ComplianceMetrics(config, timeline, task_manager, finding_tracker)
    >>> summary = await metrics.get_dashboard_summary()
    >>> readiness = await metrics.calculate_readiness_score()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics Models
# ---------------------------------------------------------------------------


class EvidenceCoverage(BaseModel):
    """Evidence collection metrics.

    Attributes:
        collected: Number of evidence items collected.
        required: Total required evidence items.
        percentage: Coverage percentage.
        by_category: Breakdown by control category.
    """

    model_config = ConfigDict(extra="forbid")

    collected: int = Field(default=0, description="Number of evidence items collected.")
    required: int = Field(default=0, description="Total required evidence items.")
    percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Coverage percentage."
    )
    by_category: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Breakdown by control category."
    )


class TestMetrics(BaseModel):
    """Control test result metrics.

    Attributes:
        passed: Number of tests passed.
        failed: Number of tests failed.
        pending: Number of tests pending.
        rate: Pass rate percentage.
        by_criterion: Breakdown by trust service criterion.
    """

    model_config = ConfigDict(extra="forbid")

    passed: int = Field(default=0, description="Number of tests passed.")
    failed: int = Field(default=0, description="Number of tests failed.")
    pending: int = Field(default=0, description="Number of tests pending.")
    rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Pass rate percentage.")
    by_criterion: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Breakdown by trust service criterion."
    )


class FindingMetrics(BaseModel):
    """Audit finding metrics.

    Attributes:
        by_severity: Finding counts by severity level.
        by_status: Finding counts by status.
        aging: Aging analysis (days open by severity).
        total_open: Total open findings.
        avg_resolution_days: Average days to resolve findings.
    """

    model_config = ConfigDict(extra="forbid")

    by_severity: Dict[str, int] = Field(
        default_factory=dict, description="Finding counts by severity."
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict, description="Finding counts by status."
    )
    aging: Dict[str, float] = Field(
        default_factory=dict, description="Average days open by severity."
    )
    total_open: int = Field(default=0, description="Total open findings.")
    avg_resolution_days: float = Field(
        default=0.0, description="Average days to resolve findings."
    )


class SLAMetrics(BaseModel):
    """Auditor request SLA metrics.

    Attributes:
        within_sla: Requests completed within SLA.
        breached: Requests that breached SLA.
        percentage: SLA compliance percentage.
        avg_response_hours: Average response time in hours.
        by_priority: Breakdown by priority level.
    """

    model_config = ConfigDict(extra="forbid")

    within_sla: int = Field(default=0, description="Requests completed within SLA.")
    breached: int = Field(default=0, description="Requests that breached SLA.")
    percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="SLA compliance percentage."
    )
    avg_response_hours: float = Field(
        default=0.0, description="Average response time in hours."
    )
    by_priority: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Breakdown by priority level."
    )


class AttestationMetrics(BaseModel):
    """Attestation status metrics.

    Attributes:
        signed: Number of signed attestations.
        pending: Number of pending attestations.
        total: Total attestations.
        by_type: Breakdown by attestation type.
    """

    model_config = ConfigDict(extra="forbid")

    signed: int = Field(default=0, description="Number of signed attestations.")
    pending: int = Field(default=0, description="Number of pending attestations.")
    total: int = Field(default=0, description="Total attestations.")
    by_type: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Breakdown by attestation type."
    )


class DashboardSummary(BaseModel):
    """Combined dashboard metrics summary.

    Attributes:
        readiness_score: Overall SOC 2 readiness score (0-100).
        evidence_coverage: Evidence collection metrics.
        test_metrics: Control test metrics.
        finding_metrics: Audit finding metrics.
        sla_metrics: Auditor request SLA metrics.
        attestation_metrics: Attestation status metrics.
        health_status: Overall health status (healthy/at_risk/critical).
        generated_at: Timestamp of metrics generation.
    """

    model_config = ConfigDict(extra="forbid")

    readiness_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall readiness score."
    )
    evidence_coverage: EvidenceCoverage = Field(
        default_factory=EvidenceCoverage, description="Evidence metrics."
    )
    test_metrics: TestMetrics = Field(
        default_factory=TestMetrics, description="Test metrics."
    )
    finding_metrics: FindingMetrics = Field(
        default_factory=FindingMetrics, description="Finding metrics."
    )
    sla_metrics: SLAMetrics = Field(
        default_factory=SLAMetrics, description="SLA metrics."
    )
    attestation_metrics: AttestationMetrics = Field(
        default_factory=AttestationMetrics, description="Attestation metrics."
    )
    health_status: str = Field(
        default="unknown", description="Overall health status."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Metrics generation timestamp.",
    )


# ---------------------------------------------------------------------------
# Compliance Metrics
# ---------------------------------------------------------------------------


class ComplianceMetrics:
    """Compliance metrics calculation for SOC 2 dashboards.

    Calculates and aggregates metrics from various data sources to provide
    a comprehensive view of SOC 2 readiness and compliance status.

    Attributes:
        config: Configuration instance.
        timeline: AuditTimeline for project data.
        task_manager: AuditTaskManager for task data.
        finding_tracker: FindingTracker for finding data.

    Example:
        >>> metrics = ComplianceMetrics(config)
        >>> summary = await metrics.get_dashboard_summary()
        >>> print(f"Readiness: {summary.readiness_score}%")
    """

    def __init__(
        self,
        config: Any = None,
        timeline: Any = None,
        task_manager: Any = None,
        finding_tracker: Any = None,
        attestation_workflow: Any = None,
    ) -> None:
        """Initialize ComplianceMetrics.

        Args:
            config: SOC2Config instance.
            timeline: AuditTimeline instance.
            task_manager: AuditTaskManager instance.
            finding_tracker: FindingTracker instance.
            attestation_workflow: AttestationWorkflow instance.
        """
        self.config = config
        self.timeline = timeline
        self.task_manager = task_manager
        self.finding_tracker = finding_tracker
        self.attestation_workflow = attestation_workflow

        # Internal metrics storage (for demo/testing)
        self._evidence_items: List[Dict[str, Any]] = []
        self._test_results: List[Dict[str, Any]] = []
        self._findings: List[Dict[str, Any]] = []
        self._requests: List[Dict[str, Any]] = []
        self._attestations: List[Dict[str, Any]] = []

        logger.info("ComplianceMetrics initialized")

    async def calculate_readiness_score(self) -> float:
        """Calculate overall SOC 2 readiness score.

        The readiness score is a weighted average of:
        - Evidence coverage (30%)
        - Test pass rate (30%)
        - Finding resolution (20%)
        - SLA compliance (10%)
        - Attestation completion (10%)

        Returns:
            Readiness score as percentage (0-100).
        """
        evidence = await self.calculate_evidence_coverage()
        tests = await self.calculate_test_pass_rate()
        findings = await self.calculate_finding_metrics()
        sla = await self.calculate_request_sla()
        attestations = await self.calculate_attestation_status()

        # Calculate component scores
        evidence_score = evidence.percentage
        test_score = tests.rate
        finding_score = self._calculate_finding_score(findings)
        sla_score = sla.percentage
        attestation_score = (
            (attestations.signed / attestations.total * 100)
            if attestations.total > 0
            else 0.0
        )

        # Weighted average
        readiness = (
            evidence_score * 0.30
            + test_score * 0.30
            + finding_score * 0.20
            + sla_score * 0.10
            + attestation_score * 0.10
        )

        logger.debug(
            "Readiness score calculated: %.1f%% (evidence=%.1f, test=%.1f, "
            "finding=%.1f, sla=%.1f, attestation=%.1f)",
            readiness,
            evidence_score,
            test_score,
            finding_score,
            sla_score,
            attestation_score,
        )

        return round(readiness, 1)

    async def calculate_evidence_coverage(self) -> EvidenceCoverage:
        """Calculate evidence collection coverage.

        Returns:
            EvidenceCoverage with collection metrics.
        """
        # In production, query from evidence storage
        # For demo, use internal storage
        collected = len([e for e in self._evidence_items if e.get("status") == "collected"])
        required = len(self._evidence_items) if self._evidence_items else 100  # Default

        percentage = (collected / required * 100) if required > 0 else 0.0

        # Calculate by category
        by_category: Dict[str, Dict[str, int]] = {}
        for item in self._evidence_items:
            cat = item.get("category", "other")
            if cat not in by_category:
                by_category[cat] = {"collected": 0, "required": 0}
            by_category[cat]["required"] += 1
            if item.get("status") == "collected":
                by_category[cat]["collected"] += 1

        return EvidenceCoverage(
            collected=collected,
            required=required,
            percentage=round(percentage, 1),
            by_category=by_category,
        )

    async def calculate_test_pass_rate(self) -> TestMetrics:
        """Calculate control test pass rate.

        Returns:
            TestMetrics with test result metrics.
        """
        passed = len([t for t in self._test_results if t.get("result") == "pass"])
        failed = len([t for t in self._test_results if t.get("result") == "fail"])
        pending = len([t for t in self._test_results if t.get("result") == "pending"])

        total_executed = passed + failed
        rate = (passed / total_executed * 100) if total_executed > 0 else 0.0

        # Calculate by criterion
        by_criterion: Dict[str, Dict[str, int]] = {}
        for test in self._test_results:
            crit = test.get("criterion", "unknown")
            if crit not in by_criterion:
                by_criterion[crit] = {"passed": 0, "failed": 0, "pending": 0}
            result = test.get("result", "pending")
            if result == "pass":
                by_criterion[crit]["passed"] += 1
            elif result == "fail":
                by_criterion[crit]["failed"] += 1
            else:
                by_criterion[crit]["pending"] += 1

        return TestMetrics(
            passed=passed,
            failed=failed,
            pending=pending,
            rate=round(rate, 1),
            by_criterion=by_criterion,
        )

    async def calculate_finding_metrics(self) -> FindingMetrics:
        """Calculate audit finding metrics.

        Returns:
            FindingMetrics with finding analysis.
        """
        by_severity: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        by_status: Dict[str, int] = {
            "open": 0,
            "in_remediation": 0,
            "resolved": 0,
            "accepted": 0,
        }
        aging: Dict[str, List[float]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        now = datetime.now(timezone.utc)
        total_resolution_days: List[float] = []

        for finding in self._findings:
            severity = finding.get("severity", "medium")
            status = finding.get("status", "open")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1

            # Calculate aging for open findings
            if status in ("open", "in_remediation"):
                created = finding.get("created_at")
                if created:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    days_open = (now - created).days
                    aging[severity].append(days_open)

            # Calculate resolution time for resolved findings
            if status == "resolved":
                created = finding.get("created_at")
                resolved = finding.get("resolved_at")
                if created and resolved:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if isinstance(resolved, str):
                        resolved = datetime.fromisoformat(resolved.replace("Z", "+00:00"))
                    resolution_days = (resolved - created).days
                    total_resolution_days.append(resolution_days)

        # Calculate average aging
        aging_avg = {
            sev: (sum(days) / len(days) if days else 0.0) for sev, days in aging.items()
        }

        total_open = by_status.get("open", 0) + by_status.get("in_remediation", 0)
        avg_resolution = (
            sum(total_resolution_days) / len(total_resolution_days)
            if total_resolution_days
            else 0.0
        )

        return FindingMetrics(
            by_severity=by_severity,
            by_status=by_status,
            aging=aging_avg,
            total_open=total_open,
            avg_resolution_days=round(avg_resolution, 1),
        )

    async def calculate_request_sla(self) -> SLAMetrics:
        """Calculate auditor request SLA metrics.

        Returns:
            SLAMetrics with SLA compliance data.
        """
        # Get SLA thresholds from config
        sla_hours = {
            "critical": getattr(self.config, "sla_critical_hours", 4) if self.config else 4,
            "high": getattr(self.config, "sla_high_hours", 24) if self.config else 24,
            "normal": getattr(self.config, "sla_normal_hours", 48) if self.config else 48,
            "low": getattr(self.config, "sla_low_hours", 72) if self.config else 72,
        }

        within_sla = 0
        breached = 0
        response_hours: List[float] = []
        by_priority: Dict[str, Dict[str, int]] = {
            "critical": {"within_sla": 0, "breached": 0},
            "high": {"within_sla": 0, "breached": 0},
            "normal": {"within_sla": 0, "breached": 0},
            "low": {"within_sla": 0, "breached": 0},
        }

        for request in self._requests:
            priority = request.get("priority", "normal")
            created = request.get("created_at")
            responded = request.get("responded_at")

            if created and responded:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if isinstance(responded, str):
                    responded = datetime.fromisoformat(responded.replace("Z", "+00:00"))

                hours = (responded - created).total_seconds() / 3600
                response_hours.append(hours)

                threshold = sla_hours.get(priority, 48)
                if hours <= threshold:
                    within_sla += 1
                    by_priority[priority]["within_sla"] += 1
                else:
                    breached += 1
                    by_priority[priority]["breached"] += 1

        total = within_sla + breached
        percentage = (within_sla / total * 100) if total > 0 else 100.0
        avg_hours = sum(response_hours) / len(response_hours) if response_hours else 0.0

        return SLAMetrics(
            within_sla=within_sla,
            breached=breached,
            percentage=round(percentage, 1),
            avg_response_hours=round(avg_hours, 1),
            by_priority=by_priority,
        )

    async def calculate_attestation_status(self) -> AttestationMetrics:
        """Calculate attestation status metrics.

        Returns:
            AttestationMetrics with attestation data.
        """
        signed = 0
        pending = 0
        by_type: Dict[str, Dict[str, int]] = {}

        # Use attestation workflow if available
        if self.attestation_workflow:
            attestations = await self.attestation_workflow.list_attestations()
            for att in attestations:
                att_type = att.attestation_type.value
                if att_type not in by_type:
                    by_type[att_type] = {"signed": 0, "pending": 0}

                if att.status.value in ("signed", "active"):
                    signed += 1
                    by_type[att_type]["signed"] += 1
                else:
                    pending += 1
                    by_type[att_type]["pending"] += 1
        else:
            # Use internal storage
            for att in self._attestations:
                att_type = att.get("type", "other")
                if att_type not in by_type:
                    by_type[att_type] = {"signed": 0, "pending": 0}

                if att.get("status") in ("signed", "active"):
                    signed += 1
                    by_type[att_type]["signed"] += 1
                else:
                    pending += 1
                    by_type[att_type]["pending"] += 1

        return AttestationMetrics(
            signed=signed,
            pending=pending,
            total=signed + pending,
            by_type=by_type,
        )

    async def get_dashboard_summary(self) -> DashboardSummary:
        """Get combined dashboard metrics summary.

        Returns:
            DashboardSummary with all metrics combined.
        """
        start_time = datetime.now(timezone.utc)

        readiness = await self.calculate_readiness_score()
        evidence = await self.calculate_evidence_coverage()
        tests = await self.calculate_test_pass_rate()
        findings = await self.calculate_finding_metrics()
        sla = await self.calculate_request_sla()
        attestations = await self.calculate_attestation_status()

        # Determine health status
        if readiness >= 90 and findings.total_open == 0:
            health_status = "healthy"
        elif readiness >= 70 and findings.by_severity.get("critical", 0) == 0:
            health_status = "at_risk"
        else:
            health_status = "critical"

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Dashboard summary generated: readiness=%.1f%%, health=%s, elapsed=%.2fms",
            readiness,
            health_status,
            elapsed_ms,
        )

        return DashboardSummary(
            readiness_score=readiness,
            evidence_coverage=evidence,
            test_metrics=tests,
            finding_metrics=findings,
            sla_metrics=sla,
            attestation_metrics=attestations,
            health_status=health_status,
        )

    def _calculate_finding_score(self, findings: FindingMetrics) -> float:
        """Calculate finding score for readiness (higher is better).

        Args:
            findings: FindingMetrics instance.

        Returns:
            Score from 0-100 based on finding status.
        """
        # Penalties for open findings
        critical_penalty = findings.by_severity.get("critical", 0) * 20
        high_penalty = findings.by_severity.get("high", 0) * 10
        medium_penalty = findings.by_severity.get("medium", 0) * 5
        low_penalty = findings.by_severity.get("low", 0) * 2

        total_penalty = critical_penalty + high_penalty + medium_penalty + low_penalty

        # Calculate open finding ratio
        total_findings = sum(findings.by_status.values())
        resolved = findings.by_status.get("resolved", 0) + findings.by_status.get(
            "accepted", 0
        )
        resolution_ratio = (resolved / total_findings * 100) if total_findings > 0 else 100.0

        # Combine: base on resolution ratio minus penalties
        score = resolution_ratio - total_penalty
        return max(0.0, min(100.0, score))

    # -----------------------------------------------------------------------
    # Data Management (for testing/demo)
    # -----------------------------------------------------------------------

    def add_evidence_item(self, item: Dict[str, Any]) -> None:
        """Add an evidence item for metrics calculation."""
        self._evidence_items.append(item)

    def add_test_result(self, result: Dict[str, Any]) -> None:
        """Add a test result for metrics calculation."""
        self._test_results.append(result)

    def add_finding(self, finding: Dict[str, Any]) -> None:
        """Add a finding for metrics calculation."""
        self._findings.append(finding)

    def add_request(self, request: Dict[str, Any]) -> None:
        """Add an auditor request for metrics calculation."""
        self._requests.append(request)

    def add_attestation(self, attestation: Dict[str, Any]) -> None:
        """Add an attestation for metrics calculation."""
        self._attestations.append(attestation)


__all__ = [
    "EvidenceCoverage",
    "TestMetrics",
    "FindingMetrics",
    "SLAMetrics",
    "AttestationMetrics",
    "DashboardSummary",
    "ComplianceMetrics",
]
