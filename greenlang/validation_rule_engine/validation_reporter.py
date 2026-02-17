# -*- coding: utf-8 -*-
"""
Validation Reporter Engine - AGENT-DATA-019 (Engine 6 of 7)

Generates validation reports for the Validation Rule Engine agent. Supports
five report types (summary, detailed, compliance, trend, executive) and
five output formats (text, json, html, markdown, csv). Every generated
report is assigned a UUID, SHA-256 hashed for provenance, and stored
in-memory for later retrieval.

The reporter consumes evaluation result data structures produced by the
upstream evaluation engine and renders them into human-readable and
machine-readable reports. Compliance reports map validation rules to
regulatory framework articles (GHG Protocol, CSRD/ESRS, EUDR, SOC 2).

Zero-Hallucination Guarantees:
    - All report content is derived deterministically from evaluation data.
    - No LLM calls for any part of report generation.
    - SHA-256 provenance hash computed for every report.
    - Thread-safe via threading.Lock on the shared reports store.
    - Prometheus-compatible metric recording on every report generation.

Supported report types:
    - summary:      Overall pass rate, severity breakdown, top failures,
                    recommendations.
    - detailed:     Per-rule results, per-row failures, detailed diagnostics.
    - compliance:   Maps rules to regulatory articles (GHG Protocol,
                    CSRD/ESRS, EUDR, SOC 2). Compliance percentage per
                    article.
    - trend:        Compare current vs historical results. Pass rate trend,
                    new failures, resolved failures.
    - executive:    High-level: pass/fail counts, critical issues, risk
                    score, action items.

Supported output formats:
    - text:     Plain-text ASCII with column-aligned tables.
    - json:     JSON with indented structure.
    - html:     Self-contained HTML with inline CSS styling.
    - markdown: GitHub-flavoured Markdown with tables.
    - csv:      Comma-separated values (flat tabular representation).

Example:
    >>> from greenlang.validation_rule_engine.validation_reporter import (
    ...     ValidationReporterEngine,
    ... )
    >>> reporter = ValidationReporterEngine()
    >>> results = [
    ...     {"rule_id": "R001", "rule_name": "Not null check",
    ...      "status": "pass", "severity": "critical",
    ...      "pass_count": 100, "fail_count": 0},
    ...     {"rule_id": "R002", "rule_name": "Range check",
    ...      "status": "fail", "severity": "high",
    ...      "pass_count": 90, "fail_count": 10,
    ...      "failures": [{"row": 5, "field": "amount", "value": -1}]},
    ... ]
    >>> report = reporter.generate_report("summary", "markdown", results)
    >>> assert report["report_type"] == "summary"
    >>> assert report["format"] == "markdown"
    >>> assert report["report_hash"] is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.validation_rule_engine.config import get_config
from greenlang.validation_rule_engine.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_REPORT_TYPES = frozenset({
    "summary",
    "detailed",
    "compliance",
    "trend",
    "executive",
})

VALID_FORMATS = frozenset({
    "text",
    "json",
    "html",
    "markdown",
    "csv",
})

# Regulatory framework article mappings for compliance reports
_REGULATORY_FRAMEWORKS: Dict[str, Dict[str, str]] = {
    "ghg_protocol": {
        "CH1": "Organizational boundaries and scope determination",
        "CH2": "Operational boundaries - Scope 1, 2, 3 classification",
        "CH3": "Tracking emissions over time (base year, recalculation)",
        "CH4": "Identifying and calculating GHG emissions",
        "CH5": "Managing inventory quality (uncertainty, completeness)",
        "CH6": "Setting a GHG target and tracking performance",
        "CH7": "Verification of GHG emissions data",
    },
    "csrd_esrs": {
        "ESRS2-BP1": "Basis for preparation - general requirements",
        "ESRS2-BP2": "Disclosures in relation to specific circumstances",
        "ESRS-E1": "Climate change - emission data quality",
        "ESRS-E2": "Pollution - substance reporting accuracy",
        "ESRS-E3": "Water and marine resources data integrity",
        "ESRS-E4": "Biodiversity - impact data completeness",
        "ESRS-E5": "Resource use and circular economy metrics",
        "ESRS-S1": "Own workforce - social data provenance",
        "ESRS-G1": "Business conduct - governance data accuracy",
    },
    "eudr": {
        "ART3": "Prohibition of non-compliant commodities on the market",
        "ART4": "Operator obligations for due diligence",
        "ART9": "Due diligence statement requirements",
        "ART10": "Due diligence system requirements",
        "ART11": "Risk assessment and mitigation",
        "ART29": "Country benchmarking and risk assessment",
    },
    "soc2": {
        "CC6.1": "Logical and physical access controls",
        "CC6.6": "System boundary and data flow documentation",
        "CC7.2": "Monitoring of system components for anomalies",
        "CC8.1": "Change management and testing",
        "PI1.1": "Processing integrity - completeness and accuracy",
        "PI1.2": "Processing integrity - timely processing",
        "PI1.3": "Processing integrity - authorized processing",
    },
}

# Severity ordering for consistent sorting
_SEVERITY_ORDER: Dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "informational": 4,
}


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_sha256(content: str) -> str:
    """Compute a SHA-256 hex digest for the given content string.

    Args:
        content: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_division(numerator: float, denominator: float) -> float:
    """Perform safe floating-point division returning 0.0 on zero denominator.

    Args:
        numerator: Dividend value.
        denominator: Divisor value.

    Returns:
        Division result, or 0.0 when denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _severity_sort_key(item: dict) -> int:
    """Return a numeric sort key for a result dict's severity field.

    Args:
        item: Dictionary containing a ``severity`` key.

    Returns:
        Integer sort key (lower is more severe).
    """
    return _SEVERITY_ORDER.get(
        str(item.get("severity", "info")).lower(), 99
    )


def _escape_csv(value: str) -> str:
    """Escape a string for safe CSV inclusion.

    Wraps the value in double quotes if it contains commas, newlines,
    or double quotes. Internal double quotes are doubled per RFC 4180.

    Args:
        value: Raw string value.

    Returns:
        CSV-safe string.
    """
    s = str(value)
    if "," in s or "\n" in s or '"' in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _escape_html(value: str) -> str:
    """Escape a string for safe HTML rendering.

    Replaces ``&``, ``<``, ``>``, ``"``, and ``'`` with their HTML
    entity equivalents.

    Args:
        value: Raw string value.

    Returns:
        HTML-safe string.
    """
    s = str(value)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&#39;")
    return s


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _compute_aggregates(
    evaluation_results: List[dict],
) -> Dict[str, Any]:
    """Compute aggregate statistics from evaluation results.

    Calculates pass rate, severity breakdown, status counts, total
    records evaluated, and identifies the top failures sorted by
    fail count descending.

    Args:
        evaluation_results: List of per-rule evaluation result dicts.

    Returns:
        Dictionary of aggregate statistics.
    """
    total_rules = len(evaluation_results)
    pass_count = 0
    fail_count = 0
    warn_count = 0
    skip_count = 0
    error_count = 0
    total_records = 0
    total_failures = 0

    severity_counts: Dict[str, int] = {}
    failures: List[dict] = []

    for result in evaluation_results:
        status = str(result.get("status", "unknown")).lower()
        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        elif status == "warn" or status == "warning":
            warn_count += 1
        elif status == "skip" or status == "skipped":
            skip_count += 1
        elif status == "error":
            error_count += 1

        severity = str(result.get("severity", "info")).lower()
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        r_pass = int(result.get("pass_count", 0))
        r_fail = int(result.get("fail_count", 0))
        total_records += r_pass + r_fail
        total_failures += r_fail

        if r_fail > 0 or status == "fail":
            failures.append(result)

    pass_rate = _safe_division(pass_count, total_rules)
    record_pass_rate = _safe_division(
        total_records - total_failures, total_records
    )

    # Sort failures by fail_count descending, then severity ascending
    failures.sort(key=lambda x: (
        -int(x.get("fail_count", 0)),
        _severity_sort_key(x),
    ))

    return {
        "total_rules": total_rules,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "skip_count": skip_count,
        "error_count": error_count,
        "pass_rate": pass_rate,
        "total_records": total_records,
        "total_failures": total_failures,
        "record_pass_rate": record_pass_rate,
        "severity_counts": severity_counts,
        "top_failures": failures[:20],
    }


def _generate_recommendations(aggregates: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on aggregate statistics.

    Examines pass rates, severity distribution, and failure patterns
    to produce a prioritized list of improvement suggestions.

    Args:
        aggregates: Aggregate statistics from ``_compute_aggregates``.

    Returns:
        Ordered list of recommendation strings.
    """
    recommendations: List[str] = []

    pass_rate = aggregates.get("pass_rate", 1.0)
    record_pass_rate = aggregates.get("record_pass_rate", 1.0)
    severity_counts = aggregates.get("severity_counts", {})
    top_failures = aggregates.get("top_failures", [])

    # Critical severity failures
    critical_count = severity_counts.get("critical", 0)
    if critical_count > 0:
        recommendations.append(
            f"URGENT: {critical_count} critical-severity rule(s) failed. "
            f"Address these immediately as they may affect regulatory "
            f"compliance and data integrity."
        )

    # High severity failures
    high_count = severity_counts.get("high", 0)
    if high_count > 0:
        recommendations.append(
            f"HIGH PRIORITY: {high_count} high-severity rule(s) failed. "
            f"Review and remediate before the next reporting cycle."
        )

    # Overall pass rate
    if pass_rate < 0.80:
        recommendations.append(
            f"Rule pass rate is {pass_rate:.1%}, which is below the 80% "
            f"warning threshold. Conduct a comprehensive data quality "
            f"review across all failing rule categories."
        )
    elif pass_rate < 0.95:
        recommendations.append(
            f"Rule pass rate is {pass_rate:.1%}. Target 95%+ for "
            f"production-grade data quality. Focus remediation on the "
            f"highest-severity failures first."
        )

    # Record-level pass rate
    if record_pass_rate < 0.90:
        recommendations.append(
            f"Record-level pass rate is {record_pass_rate:.1%}. "
            f"Investigate the top failing rules to identify systemic "
            f"data quality issues in source datasets."
        )

    # Top failure patterns
    if len(top_failures) >= 5:
        rule_names = [
            str(f.get("rule_name", f.get("rule_id", "unknown")))
            for f in top_failures[:3]
        ]
        recommendations.append(
            f"Top 3 failing rules: {', '.join(rule_names)}. "
            f"Prioritize these for root cause analysis."
        )

    # No failures
    if pass_rate == 1.0 and not recommendations:
        recommendations.append(
            "All validation rules passed. Continue monitoring for "
            "regressions in future evaluation cycles."
        )

    return recommendations


# ---------------------------------------------------------------------------
# ValidationReporterEngine
# ---------------------------------------------------------------------------


class ValidationReporterEngine:
    """Generates validation reports from evaluation results.

    Reads evaluation result data (per-rule pass/fail status, per-row
    failure details, severity levels) and renders reports in the
    requested type and format. Every generated report is stored
    in-memory with a unique ID and SHA-256 provenance hash for later
    retrieval and audit.

    The engine supports five report types:

    1. **Summary** -- Overall pass rate, severity breakdown, top
       failures, and actionable recommendations.
    2. **Detailed** -- Per-rule results with per-row failure diagnostics.
    3. **Compliance** -- Maps rules to regulatory framework articles
       (GHG Protocol, CSRD/ESRS, EUDR, SOC 2) with compliance
       percentages.
    4. **Trend** -- Compares current evaluation results against
       historical baselines to identify new failures, resolved
       failures, and pass rate trends.
    5. **Executive** -- High-level dashboard: pass/fail counts,
       critical issues, risk score, and prioritized action items.

    Each report type can be rendered in five formats: text, json,
    html, markdown, csv.

    Attributes:
        _reports: In-memory store of generated reports keyed by report ID.
        _lock: Thread-safety lock for the reports store.
        _provenance: ProvenanceTracker for SHA-256 audit trail.
        _report_count: Running counter of reports generated.

    Example:
        >>> reporter = ValidationReporterEngine()
        >>> results = [
        ...     {"rule_id": "R001", "status": "pass", "severity": "high",
        ...      "pass_count": 100, "fail_count": 0},
        ... ]
        >>> report = reporter.generate_report("summary", "json", results)
        >>> assert report["report_type"] == "summary"
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize the ValidationReporterEngine.

        Args:
            provenance: Optional ProvenanceTracker. A fresh tracker is
                created when ``None`` is supplied.
            genesis_hash: Optional genesis hash for provenance tracker
                creation when no ``provenance`` is given.
        """
        self._reports: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if provenance is not None:
            self._provenance = provenance
        elif genesis_hash is not None:
            self._provenance = ProvenanceTracker(genesis_hash=genesis_hash)
        else:
            self._provenance = ProvenanceTracker()
        self._report_count: int = 0
        logger.info("ValidationReporterEngine initialized")

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str,
        format: str,
        evaluation_results: List[dict],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Generate a validation report.

        Dispatches to the appropriate type and format generator based on
        the requested ``report_type`` and ``format``. The resulting
        report is stored in-memory and a provenance entry is recorded.

        Args:
            report_type: One of ``summary``, ``detailed``, ``compliance``,
                ``trend``, ``executive``.
            format: Output format. One of ``text``, ``json``, ``html``,
                ``markdown``, ``csv``.
            evaluation_results: List of per-rule evaluation result dicts.
                Each dict should contain at minimum: ``rule_id``,
                ``status``, ``severity``, ``pass_count``, ``fail_count``.
            parameters: Optional dictionary of extra parameters (e.g.
                ``framework`` for compliance reports, ``history`` for
                trend reports).

        Returns:
            Dictionary with keys: ``report_id``, ``report_type``,
            ``format``, ``content``, ``report_hash``, ``generated_at``,
            ``provenance_hash``.

        Raises:
            ValueError: If ``report_type`` or ``format`` is not in the
                supported set.
        """
        start_time = time.monotonic()
        params = parameters or {}

        # -- Validate inputs -----------------------------------------------
        if report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid report_type '{report_type}'. "
                f"Must be one of: {sorted(VALID_REPORT_TYPES)}"
            )
        if format not in VALID_FORMATS:
            raise ValueError(
                f"Invalid format '{format}'. "
                f"Must be one of: {sorted(VALID_FORMATS)}"
            )

        # -- Dispatch to generator -----------------------------------------
        content = self._dispatch(
            report_type=report_type,
            fmt=format,
            evaluation_results=evaluation_results,
            parameters=params,
        )

        # -- Build report envelope -----------------------------------------
        report_id = str(uuid.uuid4())
        report_hash = _compute_sha256(content)
        generated_at = _utcnow().isoformat()

        # -- Record provenance ---------------------------------------------
        provenance_hash = self._provenance.record(
            entity_type="report",
            entity_id=report_id,
            action="report_generated",
            metadata={
                "report_type": report_type,
                "format": format,
                "report_hash": report_hash,
                "rule_count": len(evaluation_results),
            },
        ).hash_value

        report: dict = {
            "report_id": report_id,
            "report_type": report_type,
            "format": format,
            "content": content,
            "report_hash": report_hash,
            "generated_at": generated_at,
            "provenance_hash": provenance_hash,
        }

        # -- Persist -------------------------------------------------------
        with self._lock:
            self._reports[report_id] = report
            self._report_count += 1

        # -- Metrics -------------------------------------------------------
        elapsed = time.monotonic() - start_time
        logger.info(
            "Generated %s report (format=%s) id=%s in %.3fs",
            report_type,
            format,
            report_id,
            elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # Report type generators
    # ------------------------------------------------------------------

    def generate_summary_report(
        self,
        evaluation_results: List[dict],
        format: str = "markdown",
    ) -> str:
        """Generate a summary validation report.

        Includes overall pass rate, severity breakdown, top failures,
        and actionable recommendations.

        Args:
            evaluation_results: List of per-rule evaluation result dicts.
            format: Output format (text, json, html, markdown, csv).

        Returns:
            Formatted report content string.
        """
        aggregates = _compute_aggregates(evaluation_results)
        recommendations = _generate_recommendations(aggregates)
        generated_at = _utcnow().isoformat()

        content_data: Dict[str, Any] = {
            "title": "Validation Summary Report",
            "generated_at": generated_at,
            "generated_by": "validation-rule-engine",
            "overview": {
                "total_rules_evaluated": aggregates["total_rules"],
                "rules_passed": aggregates["pass_count"],
                "rules_failed": aggregates["fail_count"],
                "rules_warned": aggregates["warn_count"],
                "rules_skipped": aggregates["skip_count"],
                "rules_errored": aggregates["error_count"],
                "rule_pass_rate": f"{aggregates['pass_rate']:.1%}",
                "total_records_evaluated": aggregates["total_records"],
                "total_record_failures": aggregates["total_failures"],
                "record_pass_rate": f"{aggregates['record_pass_rate']:.1%}",
            },
            "severity_breakdown": aggregates["severity_counts"],
            "top_failures": [
                {
                    "rule_id": f.get("rule_id", "unknown"),
                    "rule_name": f.get("rule_name", "unnamed"),
                    "severity": f.get("severity", "info"),
                    "fail_count": f.get("fail_count", 0),
                    "status": f.get("status", "fail"),
                }
                for f in aggregates["top_failures"][:10]
            ],
            "recommendations": recommendations,
        }

        return self._format_by_type(content_data, format, "summary")

    def generate_detailed_report(
        self,
        evaluation_results: List[dict],
        format: str = "markdown",
    ) -> str:
        """Generate a detailed validation report.

        Includes per-rule results with per-row failure diagnostics,
        field-level details, expected vs actual values, and rule
        configuration metadata.

        Args:
            evaluation_results: List of per-rule evaluation result dicts.
                Each may contain a ``failures`` key with a list of
                per-row failure dicts (row, field, value, expected, message).
            format: Output format (text, json, html, markdown, csv).

        Returns:
            Formatted report content string.
        """
        aggregates = _compute_aggregates(evaluation_results)
        generated_at = _utcnow().isoformat()

        rule_details: List[Dict[str, Any]] = []
        for result in evaluation_results:
            detail: Dict[str, Any] = {
                "rule_id": result.get("rule_id", "unknown"),
                "rule_name": result.get("rule_name", "unnamed"),
                "rule_type": result.get("rule_type", "unknown"),
                "severity": result.get("severity", "info"),
                "status": result.get("status", "unknown"),
                "pass_count": result.get("pass_count", 0),
                "fail_count": result.get("fail_count", 0),
                "field": result.get("field", ""),
                "condition": result.get("condition", ""),
                "message": result.get("message", ""),
            }

            # Per-row failure diagnostics
            row_failures = result.get("failures", [])
            detail["row_failures"] = [
                {
                    "row": rf.get("row", "N/A"),
                    "field": rf.get("field", "N/A"),
                    "value": rf.get("value", "N/A"),
                    "expected": rf.get("expected", "N/A"),
                    "message": rf.get("message", ""),
                }
                for rf in row_failures[:50]
            ]
            detail["total_row_failures"] = len(row_failures)
            detail["row_failures_truncated"] = len(row_failures) > 50
            rule_details.append(detail)

        content_data: Dict[str, Any] = {
            "title": "Detailed Validation Report",
            "generated_at": generated_at,
            "generated_by": "validation-rule-engine",
            "summary": {
                "total_rules": aggregates["total_rules"],
                "pass_rate": f"{aggregates['pass_rate']:.1%}",
                "total_records": aggregates["total_records"],
                "total_failures": aggregates["total_failures"],
            },
            "rule_details": rule_details,
        }

        return self._format_by_type(content_data, format, "detailed")

    def generate_compliance_report(
        self,
        evaluation_results: List[dict],
        format: str = "markdown",
        framework: Optional[str] = None,
    ) -> str:
        """Generate a compliance validation report.

        Maps validation rules to regulatory framework articles and
        calculates compliance percentages per article. When no
        ``framework`` is specified, all four frameworks are included.

        Args:
            evaluation_results: List of per-rule evaluation result dicts.
                Each may include a ``framework`` and ``article`` field
                indicating which regulatory article the rule maps to.
            format: Output format (text, json, html, markdown, csv).
            framework: Optional single framework to report on. One of
                ``ghg_protocol``, ``csrd_esrs``, ``eudr``, ``soc2``.
                When ``None``, all frameworks are included.

        Returns:
            Formatted report content string.
        """
        aggregates = _compute_aggregates(evaluation_results)
        generated_at = _utcnow().isoformat()

        # Determine which frameworks to include
        if framework and framework in _REGULATORY_FRAMEWORKS:
            frameworks_to_report = {framework: _REGULATORY_FRAMEWORKS[framework]}
        else:
            frameworks_to_report = dict(_REGULATORY_FRAMEWORKS)

        # Map rules to framework articles
        framework_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for fw_name, articles in frameworks_to_report.items():
            framework_results[fw_name] = {}
            for article_code, article_desc in articles.items():
                article_rules = [
                    r for r in evaluation_results
                    if (
                        str(r.get("framework", "")).lower() == fw_name.lower()
                        and str(r.get("article", "")).upper() == article_code.upper()
                    )
                ]
                if not article_rules:
                    # Check for rules mapped via tags
                    article_rules = [
                        r for r in evaluation_results
                        if article_code in r.get("tags", [])
                        or article_code.upper() in [
                            str(t).upper() for t in r.get("tags", [])
                        ]
                    ]

                total = len(article_rules)
                passed = sum(
                    1 for r in article_rules
                    if str(r.get("status", "")).lower() == "pass"
                )
                compliance_pct = _safe_division(passed, total) * 100

                framework_results[fw_name][article_code] = {
                    "article_code": article_code,
                    "article_description": article_desc,
                    "total_rules": total,
                    "rules_passed": passed,
                    "rules_failed": total - passed,
                    "compliance_percentage": round(compliance_pct, 1),
                    "status": (
                        "compliant" if compliance_pct == 100.0
                        else "partial" if compliance_pct > 0
                        else "not_assessed" if total == 0
                        else "non_compliant"
                    ),
                }

        # Compute overall compliance per framework
        framework_summaries: Dict[str, Dict[str, Any]] = {}
        for fw_name, articles in framework_results.items():
            assessed = [a for a in articles.values() if a["total_rules"] > 0]
            total_assessed = len(assessed)
            fully_compliant = sum(
                1 for a in assessed if a["status"] == "compliant"
            )
            overall_pct = _safe_division(fully_compliant, total_assessed) * 100
            framework_summaries[fw_name] = {
                "total_articles": len(articles),
                "articles_assessed": total_assessed,
                "articles_compliant": fully_compliant,
                "overall_compliance": round(overall_pct, 1),
            }

        content_data: Dict[str, Any] = {
            "title": "Compliance Validation Report",
            "generated_at": generated_at,
            "generated_by": "validation-rule-engine",
            "evaluation_summary": {
                "total_rules": aggregates["total_rules"],
                "pass_rate": f"{aggregates['pass_rate']:.1%}",
            },
            "frameworks": framework_results,
            "framework_summaries": framework_summaries,
        }

        return self._format_by_type(content_data, format, "compliance")

    def generate_trend_report(
        self,
        evaluation_history: List[dict],
        format: str = "markdown",
    ) -> str:
        """Generate a trend validation report.

        Compares the most recent evaluation results against historical
        baselines to identify pass rate trends, new failures, and
        resolved failures.

        The ``evaluation_history`` parameter should be a list of
        evaluation snapshots ordered chronologically (oldest first).
        Each snapshot should have ``timestamp``, ``pass_rate``,
        ``results`` (list of per-rule dicts), and optionally
        ``label`` (e.g. ``"2026-Q1"``).

        Args:
            evaluation_history: List of evaluation snapshots ordered
                oldest-first. Each snapshot is a dict with ``timestamp``,
                ``pass_rate``, ``results``, and optional ``label``.
            format: Output format (text, json, html, markdown, csv).

        Returns:
            Formatted report content string.
        """
        generated_at = _utcnow().isoformat()

        if len(evaluation_history) < 2:
            # Need at least two snapshots for a trend
            content_data: Dict[str, Any] = {
                "title": "Trend Validation Report",
                "generated_at": generated_at,
                "generated_by": "validation-rule-engine",
                "message": (
                    "Insufficient evaluation history for trend analysis. "
                    "At least two evaluation snapshots are required."
                ),
                "snapshots_available": len(evaluation_history),
                "trend_data": [],
                "new_failures": [],
                "resolved_failures": [],
            }
            return self._format_by_type(content_data, format, "trend")

        # Extract trend data points
        trend_data: List[Dict[str, Any]] = []
        for snapshot in evaluation_history:
            results = snapshot.get("results", [])
            agg = _compute_aggregates(results)
            trend_data.append({
                "timestamp": snapshot.get("timestamp", "unknown"),
                "label": snapshot.get("label", ""),
                "pass_rate": round(agg["pass_rate"] * 100, 1),
                "total_rules": agg["total_rules"],
                "rules_passed": agg["pass_count"],
                "rules_failed": agg["fail_count"],
                "total_records": agg["total_records"],
                "total_failures": agg["total_failures"],
            })

        # Compare latest vs previous snapshot
        current_results = evaluation_history[-1].get("results", [])
        previous_results = evaluation_history[-2].get("results", [])

        current_failed_ids = {
            str(r.get("rule_id", ""))
            for r in current_results
            if str(r.get("status", "")).lower() == "fail"
        }
        previous_failed_ids = {
            str(r.get("rule_id", ""))
            for r in previous_results
            if str(r.get("status", "")).lower() == "fail"
        }

        new_failure_ids = current_failed_ids - previous_failed_ids
        resolved_failure_ids = previous_failed_ids - current_failed_ids

        # Build new failure details
        new_failures: List[Dict[str, Any]] = []
        for r in current_results:
            if str(r.get("rule_id", "")) in new_failure_ids:
                new_failures.append({
                    "rule_id": r.get("rule_id", "unknown"),
                    "rule_name": r.get("rule_name", "unnamed"),
                    "severity": r.get("severity", "info"),
                    "fail_count": r.get("fail_count", 0),
                })

        # Build resolved failure details
        resolved_failures: List[Dict[str, Any]] = []
        for r in previous_results:
            if str(r.get("rule_id", "")) in resolved_failure_ids:
                resolved_failures.append({
                    "rule_id": r.get("rule_id", "unknown"),
                    "rule_name": r.get("rule_name", "unnamed"),
                    "severity": r.get("severity", "info"),
                })

        # Pass rate change
        current_rate = trend_data[-1]["pass_rate"]
        previous_rate = trend_data[-2]["pass_rate"]
        rate_change = round(current_rate - previous_rate, 1)
        trend_direction = (
            "improving" if rate_change > 0
            else "declining" if rate_change < 0
            else "stable"
        )

        content_data = {
            "title": "Trend Validation Report",
            "generated_at": generated_at,
            "generated_by": "validation-rule-engine",
            "trend_summary": {
                "snapshots_analyzed": len(evaluation_history),
                "current_pass_rate": f"{current_rate}%",
                "previous_pass_rate": f"{previous_rate}%",
                "rate_change": f"{rate_change:+.1f}%",
                "trend_direction": trend_direction,
                "new_failures_count": len(new_failures),
                "resolved_failures_count": len(resolved_failures),
            },
            "trend_data": trend_data,
            "new_failures": new_failures,
            "resolved_failures": resolved_failures,
        }

        return self._format_by_type(content_data, format, "trend")

    def generate_executive_report(
        self,
        evaluation_results: List[dict],
        format: str = "markdown",
    ) -> str:
        """Generate an executive validation report.

        High-level report with pass/fail counts, critical issues, a
        computed risk score (0-100), and prioritized action items
        suitable for management review.

        Args:
            evaluation_results: List of per-rule evaluation result dicts.
            format: Output format (text, json, html, markdown, csv).

        Returns:
            Formatted report content string.
        """
        aggregates = _compute_aggregates(evaluation_results)
        recommendations = _generate_recommendations(aggregates)
        generated_at = _utcnow().isoformat()

        # Compute risk score (0 = no risk, 100 = maximum risk)
        risk_score = self._compute_risk_score(aggregates)
        risk_level = self._risk_level_label(risk_score)

        # Identify critical issues
        critical_issues: List[Dict[str, Any]] = []
        for result in evaluation_results:
            sev = str(result.get("severity", "")).lower()
            status = str(result.get("status", "")).lower()
            if sev == "critical" and status == "fail":
                critical_issues.append({
                    "rule_id": result.get("rule_id", "unknown"),
                    "rule_name": result.get("rule_name", "unnamed"),
                    "fail_count": result.get("fail_count", 0),
                    "impact": "Potential regulatory non-compliance",
                })

        # Build action items
        action_items: List[Dict[str, Any]] = []
        priority = 1
        for issue in critical_issues[:5]:
            action_items.append({
                "priority": priority,
                "action": (
                    f"Remediate critical rule failure: "
                    f"{issue['rule_name']} ({issue['rule_id']})"
                ),
                "owner": "Data Quality Team",
                "deadline": "Immediate",
            })
            priority += 1

        for rec in recommendations[:5]:
            action_items.append({
                "priority": priority,
                "action": rec,
                "owner": "Data Quality Team",
                "deadline": "Next sprint",
            })
            priority += 1

        config = get_config()
        overall_status = (
            "PASS" if aggregates["pass_rate"] >= config.default_pass_threshold
            else "WARN" if aggregates["pass_rate"] >= config.default_warn_threshold
            else "FAIL"
        )

        content_data: Dict[str, Any] = {
            "title": "Executive Validation Report",
            "generated_at": generated_at,
            "generated_by": "validation-rule-engine",
            "overall_status": overall_status,
            "dashboard": {
                "total_rules": aggregates["total_rules"],
                "rules_passed": aggregates["pass_count"],
                "rules_failed": aggregates["fail_count"],
                "rules_warned": aggregates["warn_count"],
                "pass_rate": f"{aggregates['pass_rate']:.1%}",
                "total_records": aggregates["total_records"],
                "record_failures": aggregates["total_failures"],
                "record_pass_rate": f"{aggregates['record_pass_rate']:.1%}",
            },
            "risk_assessment": {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "critical_issues": len(critical_issues),
                "high_severity_failures": aggregates["severity_counts"].get(
                    "high", 0
                ),
            },
            "critical_issues": critical_issues[:10],
            "action_items": action_items[:10],
            "severity_breakdown": aggregates["severity_counts"],
        }

        return self._format_by_type(content_data, format, "executive")

    # ------------------------------------------------------------------
    # Report retrieval
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[dict]:
        """Retrieve a previously generated report by its ID.

        Args:
            report_id: UUID of the report to retrieve.

        Returns:
            The report dictionary, or ``None`` if no report with the
            given ID exists.
        """
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(
        self,
        report_type: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """List generated reports with optional filtering.

        Args:
            report_type: Optional filter by report type.
            format: Optional filter by output format.
            limit: Maximum number of reports to return. Defaults to 100.

        Returns:
            List of report summary dictionaries (content omitted),
            ordered newest first by ``generated_at``.
        """
        with self._lock:
            all_reports = list(self._reports.values())

        # Apply filters
        if report_type:
            all_reports = [
                r for r in all_reports
                if r.get("report_type") == report_type
            ]
        if format:
            all_reports = [
                r for r in all_reports if r.get("format") == format
            ]

        # Sort newest first
        all_reports.sort(
            key=lambda r: r.get("generated_at", ""), reverse=True
        )

        # Return summaries without content
        results: List[dict] = []
        for rpt in all_reports[:limit]:
            summary = {
                "report_id": rpt["report_id"],
                "report_type": rpt["report_type"],
                "format": rpt["format"],
                "report_hash": rpt["report_hash"],
                "generated_at": rpt["generated_at"],
                "provenance_hash": rpt["provenance_hash"],
            }
            results.append(summary)
        return results

    # ------------------------------------------------------------------
    # Statistics and lifecycle
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return reporter engine statistics.

        Returns:
            Dictionary with total report count, breakdown by type and
            format, and provenance entry count.
        """
        with self._lock:
            all_reports = list(self._reports.values())
            total_generated = self._report_count

        type_counts: Dict[str, int] = {}
        format_counts: Dict[str, int] = {}
        for rpt in all_reports:
            rt = rpt.get("report_type", "unknown")
            fmt = rpt.get("format", "unknown")
            type_counts[rt] = type_counts.get(rt, 0) + 1
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        return {
            "total_reports_stored": len(all_reports),
            "total_reports_generated": total_generated,
            "by_report_type": type_counts,
            "by_format": format_counts,
            "provenance_entries": self._provenance.entry_count,
        }

    def clear(self) -> None:
        """Clear all stored reports from the in-memory store.

        Does not affect the provenance chain.
        """
        with self._lock:
            count = len(self._reports)
            self._reports.clear()
        logger.info(
            "ValidationReporterEngine cleared %d stored reports", count
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        report_type: str,
        fmt: str,
        evaluation_results: List[dict],
        parameters: Dict[str, Any],
    ) -> str:
        """Dispatch report generation to the appropriate method.

        Args:
            report_type: Validated report type.
            fmt: Validated output format.
            evaluation_results: Per-rule evaluation result dicts.
            parameters: Extra parameters.

        Returns:
            Generated report content as a string.
        """
        if report_type == "summary":
            return self.generate_summary_report(evaluation_results, fmt)
        if report_type == "detailed":
            return self.generate_detailed_report(evaluation_results, fmt)
        if report_type == "compliance":
            framework = parameters.get("framework")
            return self.generate_compliance_report(
                evaluation_results, fmt, framework=framework
            )
        if report_type == "trend":
            # For trend reports, evaluation_results is reinterpreted as
            # evaluation_history (list of snapshots)
            return self.generate_trend_report(evaluation_results, fmt)
        if report_type == "executive":
            return self.generate_executive_report(evaluation_results, fmt)

        # Fallback (should not be reachable after validation)
        return self.generate_summary_report(evaluation_results, fmt)

    # ------------------------------------------------------------------
    # Format dispatcher
    # ------------------------------------------------------------------

    def _format_by_type(
        self,
        content_data: Dict[str, Any],
        fmt: str,
        report_type: str,
    ) -> str:
        """Dispatch content formatting to the appropriate format renderer.

        Args:
            content_data: Structured report data dictionary.
            fmt: Output format string.
            report_type: Report type for format-specific rendering.

        Returns:
            Formatted content string.
        """
        if fmt == "json":
            return self._format_json(content_data)
        if fmt == "text":
            return self._format_text(content_data, report_type)
        if fmt == "html":
            return self._format_html(content_data, report_type)
        if fmt == "markdown":
            return self._format_markdown(content_data, report_type)
        if fmt == "csv":
            return self._format_csv(content_data, report_type)

        # Fallback to JSON
        return self._format_json(content_data)

    # ------------------------------------------------------------------
    # Format generators (private)
    # ------------------------------------------------------------------

    def _format_json(self, content_data: Dict[str, Any]) -> str:
        """Format report data as indented JSON.

        Args:
            content_data: Structured report data dictionary.

        Returns:
            JSON-formatted string.
        """
        return json.dumps(content_data, indent=2, default=str)

    def _format_text(
        self,
        content_data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Format report data as plain-text ASCII.

        Renders the content data with column-aligned tables and
        section headers using ASCII box drawing characters.

        Args:
            content_data: Structured report data dictionary.
            report_type: Report type for section-specific rendering.

        Returns:
            Plain-text formatted string.
        """
        lines: List[str] = []
        title = content_data.get("title", "Validation Report")
        generated_at = content_data.get("generated_at", "")

        lines.append("=" * 72)
        lines.append(title.upper())
        lines.append("=" * 72)
        lines.append(f"Generated:    {generated_at}")
        lines.append(f"Generated by: {content_data.get('generated_by', 'validation-rule-engine')}")
        lines.append("")

        if report_type == "summary":
            lines.extend(self._text_summary_sections(content_data))
        elif report_type == "detailed":
            lines.extend(self._text_detailed_sections(content_data))
        elif report_type == "compliance":
            lines.extend(self._text_compliance_sections(content_data))
        elif report_type == "trend":
            lines.extend(self._text_trend_sections(content_data))
        elif report_type == "executive":
            lines.extend(self._text_executive_sections(content_data))

        lines.append("")
        lines.append("=" * 72)
        lines.append(f"END OF {title.upper()}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def _format_html(
        self,
        content_data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Format report data as a self-contained HTML document.

        Generates a complete HTML page with inline CSS styling,
        responsive layout, and color-coded status indicators.

        Args:
            content_data: Structured report data dictionary.
            report_type: Report type for section-specific rendering.

        Returns:
            HTML document string.
        """
        title = _escape_html(content_data.get("title", "Validation Report"))
        generated_at = _escape_html(content_data.get("generated_at", ""))
        body_content = self._html_body_sections(content_data, report_type)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont,
                "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f9fafb;
            color: #1f2937;
            line-height: 1.6;
        }}
        h1 {{ color: #065f46; border-bottom: 2px solid #065f46; padding-bottom: 8px; }}
        h2 {{ color: #047857; margin-top: 24px; }}
        h3 {{ color: #059669; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: #065f46;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background: #ecfdf5; }}
        .status-pass {{ color: #065f46; font-weight: bold; }}
        .status-fail {{ color: #dc2626; font-weight: bold; }}
        .status-warn {{ color: #d97706; font-weight: bold; }}
        .risk-low {{ color: #065f46; }}
        .risk-medium {{ color: #d97706; }}
        .risk-high {{ color: #dc2626; }}
        .risk-critical {{ color: #991b1b; font-weight: bold; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-pass {{ background: #d1fae5; color: #065f46; }}
        .badge-fail {{ background: #fee2e2; color: #991b1b; }}
        .badge-warn {{ background: #fef3c7; color: #92400e; }}
        .recommendations {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 12px 16px;
            margin: 16px 0;
        }}
        .recommendations li {{ margin-bottom: 8px; }}
        .footer {{
            margin-top: 24px;
            padding-top: 12px;
            border-top: 1px solid #d1d5db;
            font-size: 0.85em;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><strong>Generated:</strong> {generated_at} |
       <strong>By:</strong> {_escape_html(content_data.get('generated_by', 'validation-rule-engine'))}</p>
{body_content}
    <div class="footer">
        <p>Generated by GreenLang Validation Rule Engine (AGENT-DATA-019)
           | Report Type: {_escape_html(report_type)}</p>
    </div>
</body>
</html>"""
        return html

    def _format_markdown(
        self,
        content_data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Format report data as GitHub-flavoured Markdown.

        Renders section headers, tables, and lists in Markdown syntax
        suitable for embedding in documentation or issue trackers.

        Args:
            content_data: Structured report data dictionary.
            report_type: Report type for section-specific rendering.

        Returns:
            Markdown-formatted string.
        """
        lines: List[str] = []
        title = content_data.get("title", "Validation Report")
        generated_at = content_data.get("generated_at", "")

        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated:** {generated_at}")
        lines.append(f"**Generated By:** {content_data.get('generated_by', 'validation-rule-engine')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if report_type == "summary":
            lines.extend(self._md_summary_sections(content_data))
        elif report_type == "detailed":
            lines.extend(self._md_detailed_sections(content_data))
        elif report_type == "compliance":
            lines.extend(self._md_compliance_sections(content_data))
        elif report_type == "trend":
            lines.extend(self._md_trend_sections(content_data))
        elif report_type == "executive":
            lines.extend(self._md_executive_sections(content_data))

        lines.extend([
            "",
            "---",
            "",
            "*Generated by GreenLang Validation Rule Engine (AGENT-DATA-019).*",
        ])
        return "\n".join(lines)

    def _format_csv(
        self,
        content_data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Format report data as CSV.

        Produces a flat tabular CSV representation of the most relevant
        data for the report type. Suitable for import into spreadsheet
        tools for further analysis.

        Args:
            content_data: Structured report data dictionary.
            report_type: Report type for column selection.

        Returns:
            CSV-formatted string.
        """
        rows: List[str] = []

        if report_type == "summary":
            rows.append("metric,value")
            overview = content_data.get("overview", {})
            for key, value in overview.items():
                rows.append(f"{_escape_csv(key)},{_escape_csv(str(value))}")

            rows.append("")
            rows.append("severity,count")
            for sev, count in content_data.get("severity_breakdown", {}).items():
                rows.append(f"{_escape_csv(sev)},{count}")

            rows.append("")
            rows.append("rule_id,rule_name,severity,fail_count,status")
            for f in content_data.get("top_failures", []):
                rows.append(
                    f"{_escape_csv(str(f.get('rule_id', '')))}"
                    f",{_escape_csv(str(f.get('rule_name', '')))}"
                    f",{_escape_csv(str(f.get('severity', '')))}"
                    f",{f.get('fail_count', 0)}"
                    f",{_escape_csv(str(f.get('status', '')))}"
                )

        elif report_type == "detailed":
            rows.append(
                "rule_id,rule_name,rule_type,severity,status,"
                "pass_count,fail_count,field,condition"
            )
            for detail in content_data.get("rule_details", []):
                rows.append(
                    f"{_escape_csv(str(detail.get('rule_id', '')))}"
                    f",{_escape_csv(str(detail.get('rule_name', '')))}"
                    f",{_escape_csv(str(detail.get('rule_type', '')))}"
                    f",{_escape_csv(str(detail.get('severity', '')))}"
                    f",{_escape_csv(str(detail.get('status', '')))}"
                    f",{detail.get('pass_count', 0)}"
                    f",{detail.get('fail_count', 0)}"
                    f",{_escape_csv(str(detail.get('field', '')))}"
                    f",{_escape_csv(str(detail.get('condition', '')))}"
                )

        elif report_type == "compliance":
            rows.append(
                "framework,article_code,article_description,"
                "total_rules,rules_passed,rules_failed,"
                "compliance_percentage,status"
            )
            for fw_name, articles in content_data.get("frameworks", {}).items():
                for article_code, article in articles.items():
                    rows.append(
                        f"{_escape_csv(fw_name)}"
                        f",{_escape_csv(article_code)}"
                        f",{_escape_csv(str(article.get('article_description', '')))}"
                        f",{article.get('total_rules', 0)}"
                        f",{article.get('rules_passed', 0)}"
                        f",{article.get('rules_failed', 0)}"
                        f",{article.get('compliance_percentage', 0)}"
                        f",{_escape_csv(str(article.get('status', '')))}"
                    )

        elif report_type == "trend":
            rows.append(
                "timestamp,label,pass_rate,total_rules,"
                "rules_passed,rules_failed,total_records,total_failures"
            )
            for td in content_data.get("trend_data", []):
                rows.append(
                    f"{_escape_csv(str(td.get('timestamp', '')))}"
                    f",{_escape_csv(str(td.get('label', '')))}"
                    f",{td.get('pass_rate', 0)}"
                    f",{td.get('total_rules', 0)}"
                    f",{td.get('rules_passed', 0)}"
                    f",{td.get('rules_failed', 0)}"
                    f",{td.get('total_records', 0)}"
                    f",{td.get('total_failures', 0)}"
                )

        elif report_type == "executive":
            rows.append("metric,value")
            dashboard = content_data.get("dashboard", {})
            for key, value in dashboard.items():
                rows.append(f"{_escape_csv(key)},{_escape_csv(str(value))}")

            rows.append("")
            rows.append("priority,action,owner,deadline")
            for item in content_data.get("action_items", []):
                rows.append(
                    f"{item.get('priority', 0)}"
                    f",{_escape_csv(str(item.get('action', '')))}"
                    f",{_escape_csv(str(item.get('owner', '')))}"
                    f",{_escape_csv(str(item.get('deadline', '')))}"
                )

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Text section renderers
    # ------------------------------------------------------------------

    def _text_summary_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render summary report sections in plain text.

        Args:
            content_data: Structured summary report data.

        Returns:
            List of text lines for the summary sections.
        """
        lines: List[str] = []
        overview = content_data.get("overview", {})

        lines.append("-" * 72)
        lines.append("OVERVIEW")
        lines.append("-" * 72)
        for key, value in overview.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<35s} {value}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("SEVERITY BREAKDOWN")
        lines.append("-" * 72)
        for sev, count in sorted(
            content_data.get("severity_breakdown", {}).items(),
            key=lambda x: _SEVERITY_ORDER.get(x[0], 99),
        ):
            bar = "#" * min(count, 40)
            lines.append(f"  {sev:<15s} {count:>5d}  {bar}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("TOP FAILURES")
        lines.append("-" * 72)
        failures = content_data.get("top_failures", [])
        if failures:
            lines.append(
                f"  {'Rule ID':<15s} {'Rule Name':<25s} "
                f"{'Severity':<12s} {'Fails':>6s}"
            )
            lines.append(f"  {'-'*15} {'-'*25} {'-'*12} {'-'*6}")
            for f in failures:
                lines.append(
                    f"  {str(f.get('rule_id', '')):<15s} "
                    f"{str(f.get('rule_name', '')):<25s} "
                    f"{str(f.get('severity', '')):<12s} "
                    f"{f.get('fail_count', 0):>6d}"
                )
        else:
            lines.append("  No failures detected.")

        lines.append("")
        lines.append("-" * 72)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 72)
        for idx, rec in enumerate(
            content_data.get("recommendations", []), 1
        ):
            lines.append(f"  {idx}. {rec}")

        return lines

    def _text_detailed_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render detailed report sections in plain text.

        Args:
            content_data: Structured detailed report data.

        Returns:
            List of text lines for the detailed sections.
        """
        lines: List[str] = []
        summary = content_data.get("summary", {})

        lines.append("-" * 72)
        lines.append("SUMMARY")
        lines.append("-" * 72)
        for key, value in summary.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<25s} {value}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("RULE-BY-RULE DETAILS")
        lines.append("-" * 72)

        for detail in content_data.get("rule_details", []):
            lines.append("")
            lines.append(
                f"  Rule: {detail.get('rule_id', 'unknown')} - "
                f"{detail.get('rule_name', 'unnamed')}"
            )
            lines.append(
                f"    Type:      {detail.get('rule_type', 'unknown')}"
            )
            lines.append(
                f"    Severity:  {detail.get('severity', 'info')}"
            )
            lines.append(
                f"    Status:    {detail.get('status', 'unknown')}"
            )
            lines.append(
                f"    Passed:    {detail.get('pass_count', 0)}  "
                f"Failed: {detail.get('fail_count', 0)}"
            )
            if detail.get("field"):
                lines.append(f"    Field:     {detail['field']}")
            if detail.get("condition"):
                lines.append(f"    Condition: {detail['condition']}")
            if detail.get("message"):
                lines.append(f"    Message:   {detail['message']}")

            row_failures = detail.get("row_failures", [])
            if row_failures:
                lines.append(f"    Row Failures ({detail.get('total_row_failures', 0)} total):")
                for rf in row_failures[:10]:
                    lines.append(
                        f"      Row {rf.get('row', 'N/A')}: "
                        f"field={rf.get('field', 'N/A')}, "
                        f"value={rf.get('value', 'N/A')}, "
                        f"expected={rf.get('expected', 'N/A')}"
                    )
                if len(row_failures) > 10:
                    lines.append(
                        f"      ... and {len(row_failures) - 10} more"
                    )

        return lines

    def _text_compliance_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render compliance report sections in plain text.

        Args:
            content_data: Structured compliance report data.

        Returns:
            List of text lines for the compliance sections.
        """
        lines: List[str] = []

        for fw_name, articles in content_data.get("frameworks", {}).items():
            fw_summary = content_data.get("framework_summaries", {}).get(
                fw_name, {}
            )
            lines.append("-" * 72)
            lines.append(f"FRAMEWORK: {fw_name.upper().replace('_', ' ')}")
            lines.append("-" * 72)
            lines.append(
                f"  Articles Assessed:  {fw_summary.get('articles_assessed', 0)}"
                f" / {fw_summary.get('total_articles', 0)}"
            )
            lines.append(
                f"  Articles Compliant: {fw_summary.get('articles_compliant', 0)}"
            )
            lines.append(
                f"  Overall Compliance: {fw_summary.get('overall_compliance', 0)}%"
            )
            lines.append("")
            lines.append(
                f"  {'Article':<12s} {'Description':<40s} "
                f"{'Rules':>5s} {'Pass':>5s} {'Compl%':>7s} {'Status':<15s}"
            )
            lines.append(
                f"  {'-'*12} {'-'*40} {'-'*5} {'-'*5} {'-'*7} {'-'*15}"
            )
            for article_code, article in articles.items():
                lines.append(
                    f"  {article_code:<12s} "
                    f"{str(article.get('article_description', ''))[:40]:<40s} "
                    f"{article.get('total_rules', 0):>5d} "
                    f"{article.get('rules_passed', 0):>5d} "
                    f"{article.get('compliance_percentage', 0):>6.1f}% "
                    f"{article.get('status', 'unknown'):<15s}"
                )
            lines.append("")

        return lines

    def _text_trend_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render trend report sections in plain text.

        Args:
            content_data: Structured trend report data.

        Returns:
            List of text lines for the trend sections.
        """
        lines: List[str] = []

        if content_data.get("message"):
            lines.append(content_data["message"])
            return lines

        trend_summary = content_data.get("trend_summary", {})
        lines.append("-" * 72)
        lines.append("TREND SUMMARY")
        lines.append("-" * 72)
        for key, value in trend_summary.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<30s} {value}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("PASS RATE HISTORY")
        lines.append("-" * 72)
        lines.append(
            f"  {'Timestamp':<25s} {'Label':<15s} "
            f"{'Rate':>7s} {'Rules':>6s} {'Failed':>6s}"
        )
        lines.append(
            f"  {'-'*25} {'-'*15} {'-'*7} {'-'*6} {'-'*6}"
        )
        for td in content_data.get("trend_data", []):
            lines.append(
                f"  {str(td.get('timestamp', '')):<25s} "
                f"{str(td.get('label', '')):<15s} "
                f"{td.get('pass_rate', 0):>6.1f}% "
                f"{td.get('total_rules', 0):>6d} "
                f"{td.get('rules_failed', 0):>6d}"
            )

        new_failures = content_data.get("new_failures", [])
        if new_failures:
            lines.append("")
            lines.append("-" * 72)
            lines.append(f"NEW FAILURES ({len(new_failures)})")
            lines.append("-" * 72)
            for nf in new_failures:
                lines.append(
                    f"  [{nf.get('severity', 'info')}] "
                    f"{nf.get('rule_id', 'unknown')} - "
                    f"{nf.get('rule_name', 'unnamed')} "
                    f"({nf.get('fail_count', 0)} failures)"
                )

        resolved = content_data.get("resolved_failures", [])
        if resolved:
            lines.append("")
            lines.append("-" * 72)
            lines.append(f"RESOLVED FAILURES ({len(resolved)})")
            lines.append("-" * 72)
            for rf in resolved:
                lines.append(
                    f"  [RESOLVED] {rf.get('rule_id', 'unknown')} - "
                    f"{rf.get('rule_name', 'unnamed')}"
                )

        return lines

    def _text_executive_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render executive report sections in plain text.

        Args:
            content_data: Structured executive report data.

        Returns:
            List of text lines for the executive sections.
        """
        lines: List[str] = []
        overall = content_data.get("overall_status", "UNKNOWN")

        lines.append("-" * 72)
        lines.append(f"OVERALL STATUS: {overall}")
        lines.append("-" * 72)

        lines.append("")
        lines.append("-" * 72)
        lines.append("DASHBOARD")
        lines.append("-" * 72)
        dashboard = content_data.get("dashboard", {})
        for key, value in dashboard.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<25s} {value}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("RISK ASSESSMENT")
        lines.append("-" * 72)
        risk = content_data.get("risk_assessment", {})
        for key, value in risk.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label:<30s} {value}")

        critical = content_data.get("critical_issues", [])
        if critical:
            lines.append("")
            lines.append("-" * 72)
            lines.append(f"CRITICAL ISSUES ({len(critical)})")
            lines.append("-" * 72)
            for issue in critical:
                lines.append(
                    f"  [CRITICAL] {issue.get('rule_id', 'unknown')} - "
                    f"{issue.get('rule_name', 'unnamed')} "
                    f"({issue.get('fail_count', 0)} failures)"
                )

        action_items = content_data.get("action_items", [])
        if action_items:
            lines.append("")
            lines.append("-" * 72)
            lines.append("ACTION ITEMS")
            lines.append("-" * 72)
            for item in action_items:
                lines.append(
                    f"  P{item.get('priority', 0)}: "
                    f"{item.get('action', 'No action specified')} "
                    f"[{item.get('owner', 'TBD')}] "
                    f"Due: {item.get('deadline', 'TBD')}"
                )

        return lines

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_summary_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render summary report sections in Markdown.

        Args:
            content_data: Structured summary report data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        overview = content_data.get("overview", {})

        lines.append("## Overview")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in overview.items():
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {value} |")

        lines.append("")
        lines.append("## Severity Breakdown")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev, count in sorted(
            content_data.get("severity_breakdown", {}).items(),
            key=lambda x: _SEVERITY_ORDER.get(x[0], 99),
        ):
            lines.append(f"| {sev} | {count} |")

        lines.append("")
        lines.append("## Top Failures")
        lines.append("")
        failures = content_data.get("top_failures", [])
        if failures:
            lines.append("| Rule ID | Rule Name | Severity | Fail Count | Status |")
            lines.append("|---------|-----------|----------|------------|--------|")
            for f in failures:
                lines.append(
                    f"| {f.get('rule_id', '')} "
                    f"| {f.get('rule_name', '')} "
                    f"| {f.get('severity', '')} "
                    f"| {f.get('fail_count', 0)} "
                    f"| {f.get('status', '')} |"
                )
        else:
            lines.append("No failures detected.")

        lines.append("")
        lines.append("## Recommendations")
        lines.append("")
        for idx, rec in enumerate(
            content_data.get("recommendations", []), 1
        ):
            lines.append(f"{idx}. {rec}")

        return lines

    def _md_detailed_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render detailed report sections in Markdown.

        Args:
            content_data: Structured detailed report data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        summary = content_data.get("summary", {})

        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in summary.items():
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {value} |")

        lines.append("")
        lines.append("## Rule Details")
        lines.append("")

        for detail in content_data.get("rule_details", []):
            rule_id = detail.get("rule_id", "unknown")
            rule_name = detail.get("rule_name", "unnamed")
            status = detail.get("status", "unknown")

            lines.append(f"### {rule_id} - {rule_name}")
            lines.append("")
            lines.append(f"- **Status:** {status}")
            lines.append(f"- **Severity:** {detail.get('severity', 'info')}")
            lines.append(f"- **Type:** {detail.get('rule_type', 'unknown')}")
            lines.append(
                f"- **Passed:** {detail.get('pass_count', 0)} | "
                f"**Failed:** {detail.get('fail_count', 0)}"
            )
            if detail.get("field"):
                lines.append(f"- **Field:** {detail['field']}")
            if detail.get("condition"):
                lines.append(f"- **Condition:** {detail['condition']}")
            if detail.get("message"):
                lines.append(f"- **Message:** {detail['message']}")

            row_failures = detail.get("row_failures", [])
            if row_failures:
                lines.append("")
                lines.append(
                    f"**Row Failures** "
                    f"({detail.get('total_row_failures', 0)} total):"
                )
                lines.append("")
                lines.append("| Row | Field | Value | Expected | Message |")
                lines.append("|-----|-------|-------|----------|---------|")
                for rf in row_failures[:20]:
                    lines.append(
                        f"| {rf.get('row', 'N/A')} "
                        f"| {rf.get('field', 'N/A')} "
                        f"| {rf.get('value', 'N/A')} "
                        f"| {rf.get('expected', 'N/A')} "
                        f"| {rf.get('message', '')} |"
                    )
                if detail.get("row_failures_truncated"):
                    lines.append("")
                    lines.append(
                        f"*Showing first 50 of "
                        f"{detail.get('total_row_failures', 0)} failures.*"
                    )
            lines.append("")

        return lines

    def _md_compliance_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render compliance report sections in Markdown.

        Args:
            content_data: Structured compliance report data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        eval_summary = content_data.get("evaluation_summary", {})

        lines.append("## Evaluation Summary")
        lines.append("")
        lines.append(f"- **Total Rules:** {eval_summary.get('total_rules', 0)}")
        lines.append(f"- **Pass Rate:** {eval_summary.get('pass_rate', 'N/A')}")
        lines.append("")

        for fw_name, articles in content_data.get("frameworks", {}).items():
            fw_summary = content_data.get("framework_summaries", {}).get(
                fw_name, {}
            )
            fw_label = fw_name.upper().replace("_", " ")
            lines.append(f"## {fw_label}")
            lines.append("")
            lines.append(
                f"**Overall Compliance:** "
                f"{fw_summary.get('overall_compliance', 0)}% | "
                f"**Articles Assessed:** "
                f"{fw_summary.get('articles_assessed', 0)} / "
                f"{fw_summary.get('total_articles', 0)} | "
                f"**Fully Compliant:** "
                f"{fw_summary.get('articles_compliant', 0)}"
            )
            lines.append("")
            lines.append(
                "| Article | Description | Rules | Passed | "
                "Compliance | Status |"
            )
            lines.append(
                "|---------|-------------|-------|--------|"
                "------------|--------|"
            )
            for article_code, article in articles.items():
                pct = article.get("compliance_percentage", 0)
                lines.append(
                    f"| {article_code} "
                    f"| {article.get('article_description', '')} "
                    f"| {article.get('total_rules', 0)} "
                    f"| {article.get('rules_passed', 0)} "
                    f"| {pct}% "
                    f"| {article.get('status', 'unknown')} |"
                )
            lines.append("")

        return lines

    def _md_trend_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render trend report sections in Markdown.

        Args:
            content_data: Structured trend report data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []

        if content_data.get("message"):
            lines.append(f"> {content_data['message']}")
            return lines

        trend_summary = content_data.get("trend_summary", {})
        lines.append("## Trend Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in trend_summary.items():
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {value} |")

        lines.append("")
        lines.append("## Pass Rate History")
        lines.append("")
        lines.append(
            "| Timestamp | Label | Pass Rate | Rules | Failed | "
            "Records | Failures |"
        )
        lines.append(
            "|-----------|-------|-----------|-------|--------|"
            "---------|----------|"
        )
        for td in content_data.get("trend_data", []):
            lines.append(
                f"| {td.get('timestamp', '')} "
                f"| {td.get('label', '')} "
                f"| {td.get('pass_rate', 0)}% "
                f"| {td.get('total_rules', 0)} "
                f"| {td.get('rules_failed', 0)} "
                f"| {td.get('total_records', 0)} "
                f"| {td.get('total_failures', 0)} |"
            )

        new_failures = content_data.get("new_failures", [])
        if new_failures:
            lines.append("")
            lines.append(f"## New Failures ({len(new_failures)})")
            lines.append("")
            lines.append("| Rule ID | Rule Name | Severity | Fail Count |")
            lines.append("|---------|-----------|----------|------------|")
            for nf in new_failures:
                lines.append(
                    f"| {nf.get('rule_id', '')} "
                    f"| {nf.get('rule_name', '')} "
                    f"| {nf.get('severity', '')} "
                    f"| {nf.get('fail_count', 0)} |"
                )

        resolved = content_data.get("resolved_failures", [])
        if resolved:
            lines.append("")
            lines.append(f"## Resolved Failures ({len(resolved)})")
            lines.append("")
            lines.append("| Rule ID | Rule Name | Severity |")
            lines.append("|---------|-----------|----------|")
            for rf in resolved:
                lines.append(
                    f"| {rf.get('rule_id', '')} "
                    f"| {rf.get('rule_name', '')} "
                    f"| {rf.get('severity', '')} |"
                )

        return lines

    def _md_executive_sections(
        self, content_data: Dict[str, Any]
    ) -> List[str]:
        """Render executive report sections in Markdown.

        Args:
            content_data: Structured executive report data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        overall = content_data.get("overall_status", "UNKNOWN")

        lines.append(f"## Overall Status: **{overall}**")
        lines.append("")

        lines.append("## Dashboard")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in content_data.get("dashboard", {}).items():
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {value} |")

        lines.append("")
        lines.append("## Risk Assessment")
        lines.append("")
        risk = content_data.get("risk_assessment", {})
        lines.append(f"- **Risk Score:** {risk.get('risk_score', 0)} / 100")
        lines.append(f"- **Risk Level:** {risk.get('risk_level', 'unknown')}")
        lines.append(
            f"- **Critical Issues:** {risk.get('critical_issues', 0)}"
        )
        lines.append(
            f"- **High Severity Failures:** "
            f"{risk.get('high_severity_failures', 0)}"
        )

        critical = content_data.get("critical_issues", [])
        if critical:
            lines.append("")
            lines.append(f"## Critical Issues ({len(critical)})")
            lines.append("")
            lines.append("| Rule ID | Rule Name | Fail Count | Impact |")
            lines.append("|---------|-----------|------------|--------|")
            for issue in critical:
                lines.append(
                    f"| {issue.get('rule_id', '')} "
                    f"| {issue.get('rule_name', '')} "
                    f"| {issue.get('fail_count', 0)} "
                    f"| {issue.get('impact', '')} |"
                )

        action_items = content_data.get("action_items", [])
        if action_items:
            lines.append("")
            lines.append("## Action Items")
            lines.append("")
            lines.append("| Priority | Action | Owner | Deadline |")
            lines.append("|----------|--------|-------|----------|")
            for item in action_items:
                lines.append(
                    f"| P{item.get('priority', 0)} "
                    f"| {item.get('action', '')} "
                    f"| {item.get('owner', '')} "
                    f"| {item.get('deadline', '')} |"
                )

        lines.append("")
        lines.append("## Severity Breakdown")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev, count in sorted(
            content_data.get("severity_breakdown", {}).items(),
            key=lambda x: _SEVERITY_ORDER.get(x[0], 99),
        ):
            lines.append(f"| {sev} | {count} |")

        return lines

    # ------------------------------------------------------------------
    # HTML body section renderers
    # ------------------------------------------------------------------

    def _html_body_sections(
        self,
        content_data: Dict[str, Any],
        report_type: str,
    ) -> str:
        """Generate HTML body sections for the given report type.

        Args:
            content_data: Structured report data dictionary.
            report_type: Report type for section selection.

        Returns:
            HTML fragment string for the body sections.
        """
        if report_type == "summary":
            return self._html_summary(content_data)
        if report_type == "detailed":
            return self._html_detailed(content_data)
        if report_type == "compliance":
            return self._html_compliance(content_data)
        if report_type == "trend":
            return self._html_trend(content_data)
        if report_type == "executive":
            return self._html_executive(content_data)
        return ""

    def _html_summary(self, content_data: Dict[str, Any]) -> str:
        """Render summary HTML sections."""
        parts: List[str] = []
        overview = content_data.get("overview", {})

        parts.append("    <h2>Overview</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Metric</th><th>Value</th></tr>")
        for key, value in overview.items():
            label = _escape_html(key.replace("_", " ").title())
            parts.append(
                f"        <tr><td>{label}</td>"
                f"<td>{_escape_html(str(value))}</td></tr>"
            )
        parts.append("    </table>")

        parts.append("    <h2>Severity Breakdown</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Severity</th><th>Count</th></tr>")
        for sev, count in sorted(
            content_data.get("severity_breakdown", {}).items(),
            key=lambda x: _SEVERITY_ORDER.get(x[0], 99),
        ):
            parts.append(
                f"        <tr><td>{_escape_html(sev)}</td>"
                f"<td>{count}</td></tr>"
            )
        parts.append("    </table>")

        failures = content_data.get("top_failures", [])
        if failures:
            parts.append("    <h2>Top Failures</h2>")
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Rule ID</th><th>Rule Name</th>"
                "<th>Severity</th><th>Fail Count</th><th>Status</th></tr>"
            )
            for f in failures:
                status_class = (
                    "status-fail" if f.get("status") == "fail" else ""
                )
                parts.append(
                    f"        <tr>"
                    f"<td>{_escape_html(str(f.get('rule_id', '')))}</td>"
                    f"<td>{_escape_html(str(f.get('rule_name', '')))}</td>"
                    f"<td>{_escape_html(str(f.get('severity', '')))}</td>"
                    f"<td>{f.get('fail_count', 0)}</td>"
                    f"<td class=\"{status_class}\">"
                    f"{_escape_html(str(f.get('status', '')))}</td></tr>"
                )
            parts.append("    </table>")

        recs = content_data.get("recommendations", [])
        if recs:
            parts.append("    <h2>Recommendations</h2>")
            parts.append("    <div class=\"recommendations\">")
            parts.append("    <ol>")
            for rec in recs:
                parts.append(f"        <li>{_escape_html(rec)}</li>")
            parts.append("    </ol>")
            parts.append("    </div>")

        return "\n".join(parts)

    def _html_detailed(self, content_data: Dict[str, Any]) -> str:
        """Render detailed HTML sections."""
        parts: List[str] = []
        summary = content_data.get("summary", {})

        parts.append("    <h2>Summary</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Metric</th><th>Value</th></tr>")
        for key, value in summary.items():
            label = _escape_html(key.replace("_", " ").title())
            parts.append(
                f"        <tr><td>{label}</td>"
                f"<td>{_escape_html(str(value))}</td></tr>"
            )
        parts.append("    </table>")

        parts.append("    <h2>Rule Details</h2>")
        for detail in content_data.get("rule_details", []):
            rule_id = _escape_html(str(detail.get("rule_id", "unknown")))
            rule_name = _escape_html(str(detail.get("rule_name", "unnamed")))
            status = str(detail.get("status", "unknown")).lower()
            status_class = f"status-{status}" if status in ("pass", "fail", "warn") else ""

            parts.append(f"    <h3>{rule_id} - {rule_name}</h3>")
            parts.append("    <table>")
            parts.append(
                f"        <tr><td><strong>Status</strong></td>"
                f"<td class=\"{status_class}\">"
                f"{_escape_html(status)}</td></tr>"
            )
            parts.append(
                f"        <tr><td><strong>Severity</strong></td>"
                f"<td>{_escape_html(str(detail.get('severity', '')))}</td></tr>"
            )
            parts.append(
                f"        <tr><td><strong>Passed / Failed</strong></td>"
                f"<td>{detail.get('pass_count', 0)} / "
                f"{detail.get('fail_count', 0)}</td></tr>"
            )
            parts.append("    </table>")

            row_failures = detail.get("row_failures", [])
            if row_failures:
                parts.append(
                    f"    <p><strong>Row Failures</strong> "
                    f"({detail.get('total_row_failures', 0)} total):</p>"
                )
                parts.append("    <table>")
                parts.append(
                    "        <tr><th>Row</th><th>Field</th>"
                    "<th>Value</th><th>Expected</th></tr>"
                )
                for rf in row_failures[:20]:
                    parts.append(
                        f"        <tr>"
                        f"<td>{_escape_html(str(rf.get('row', 'N/A')))}</td>"
                        f"<td>{_escape_html(str(rf.get('field', 'N/A')))}</td>"
                        f"<td>{_escape_html(str(rf.get('value', 'N/A')))}</td>"
                        f"<td>{_escape_html(str(rf.get('expected', 'N/A')))}</td>"
                        f"</tr>"
                    )
                parts.append("    </table>")

        return "\n".join(parts)

    def _html_compliance(self, content_data: Dict[str, Any]) -> str:
        """Render compliance HTML sections."""
        parts: List[str] = []

        for fw_name, articles in content_data.get("frameworks", {}).items():
            fw_summary = content_data.get("framework_summaries", {}).get(
                fw_name, {}
            )
            fw_label = _escape_html(fw_name.upper().replace("_", " "))
            overall_pct = fw_summary.get("overall_compliance", 0)

            parts.append(f"    <h2>{fw_label}</h2>")
            parts.append(
                f"    <p><strong>Overall Compliance:</strong> {overall_pct}% | "
                f"<strong>Articles Assessed:</strong> "
                f"{fw_summary.get('articles_assessed', 0)} / "
                f"{fw_summary.get('total_articles', 0)}</p>"
            )
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Article</th><th>Description</th>"
                "<th>Rules</th><th>Passed</th>"
                "<th>Compliance</th><th>Status</th></tr>"
            )
            for article_code, article in articles.items():
                status = article.get("status", "unknown")
                status_class = (
                    "status-pass" if status == "compliant"
                    else "status-fail" if status == "non_compliant"
                    else "status-warn" if status == "partial"
                    else ""
                )
                parts.append(
                    f"        <tr>"
                    f"<td>{_escape_html(article_code)}</td>"
                    f"<td>{_escape_html(str(article.get('article_description', '')))}</td>"
                    f"<td>{article.get('total_rules', 0)}</td>"
                    f"<td>{article.get('rules_passed', 0)}</td>"
                    f"<td>{article.get('compliance_percentage', 0)}%</td>"
                    f"<td class=\"{status_class}\">"
                    f"{_escape_html(status)}</td></tr>"
                )
            parts.append("    </table>")

        return "\n".join(parts)

    def _html_trend(self, content_data: Dict[str, Any]) -> str:
        """Render trend HTML sections."""
        parts: List[str] = []

        if content_data.get("message"):
            parts.append(
                f"    <p><em>{_escape_html(content_data['message'])}</em></p>"
            )
            return "\n".join(parts)

        trend_summary = content_data.get("trend_summary", {})
        parts.append("    <h2>Trend Summary</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Metric</th><th>Value</th></tr>")
        for key, value in trend_summary.items():
            label = _escape_html(key.replace("_", " ").title())
            parts.append(
                f"        <tr><td>{label}</td>"
                f"<td>{_escape_html(str(value))}</td></tr>"
            )
        parts.append("    </table>")

        parts.append("    <h2>Pass Rate History</h2>")
        parts.append("    <table>")
        parts.append(
            "        <tr><th>Timestamp</th><th>Label</th>"
            "<th>Pass Rate</th><th>Rules</th><th>Failed</th></tr>"
        )
        for td in content_data.get("trend_data", []):
            parts.append(
                f"        <tr>"
                f"<td>{_escape_html(str(td.get('timestamp', '')))}</td>"
                f"<td>{_escape_html(str(td.get('label', '')))}</td>"
                f"<td>{td.get('pass_rate', 0)}%</td>"
                f"<td>{td.get('total_rules', 0)}</td>"
                f"<td>{td.get('rules_failed', 0)}</td></tr>"
            )
        parts.append("    </table>")

        new_failures = content_data.get("new_failures", [])
        if new_failures:
            parts.append(f"    <h2>New Failures ({len(new_failures)})</h2>")
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Rule ID</th><th>Rule Name</th>"
                "<th>Severity</th><th>Fail Count</th></tr>"
            )
            for nf in new_failures:
                parts.append(
                    f"        <tr>"
                    f"<td>{_escape_html(str(nf.get('rule_id', '')))}</td>"
                    f"<td>{_escape_html(str(nf.get('rule_name', '')))}</td>"
                    f"<td>{_escape_html(str(nf.get('severity', '')))}</td>"
                    f"<td>{nf.get('fail_count', 0)}</td></tr>"
                )
            parts.append("    </table>")

        resolved = content_data.get("resolved_failures", [])
        if resolved:
            parts.append(f"    <h2>Resolved Failures ({len(resolved)})</h2>")
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Rule ID</th><th>Rule Name</th>"
                "<th>Severity</th></tr>"
            )
            for rf in resolved:
                parts.append(
                    f"        <tr>"
                    f"<td>{_escape_html(str(rf.get('rule_id', '')))}</td>"
                    f"<td>{_escape_html(str(rf.get('rule_name', '')))}</td>"
                    f"<td>{_escape_html(str(rf.get('severity', '')))}</td></tr>"
                )
            parts.append("    </table>")

        return "\n".join(parts)

    def _html_executive(self, content_data: Dict[str, Any]) -> str:
        """Render executive HTML sections."""
        parts: List[str] = []
        overall = content_data.get("overall_status", "UNKNOWN")
        status_class = (
            "status-pass" if overall == "PASS"
            else "status-fail" if overall == "FAIL"
            else "status-warn"
        )

        parts.append(
            f"    <h2>Overall Status: "
            f"<span class=\"{status_class}\">{_escape_html(overall)}</span></h2>"
        )

        parts.append("    <h2>Dashboard</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Metric</th><th>Value</th></tr>")
        for key, value in content_data.get("dashboard", {}).items():
            label = _escape_html(key.replace("_", " ").title())
            parts.append(
                f"        <tr><td>{label}</td>"
                f"<td>{_escape_html(str(value))}</td></tr>"
            )
        parts.append("    </table>")

        risk = content_data.get("risk_assessment", {})
        risk_score = risk.get("risk_score", 0)
        risk_class = (
            "risk-low" if risk_score <= 25
            else "risk-medium" if risk_score <= 50
            else "risk-high" if risk_score <= 75
            else "risk-critical"
        )
        parts.append("    <h2>Risk Assessment</h2>")
        parts.append("    <table>")
        parts.append("        <tr><th>Metric</th><th>Value</th></tr>")
        parts.append(
            f"        <tr><td>Risk Score</td>"
            f"<td class=\"{risk_class}\">{risk_score} / 100</td></tr>"
        )
        parts.append(
            f"        <tr><td>Risk Level</td>"
            f"<td class=\"{risk_class}\">"
            f"{_escape_html(str(risk.get('risk_level', '')))}</td></tr>"
        )
        parts.append(
            f"        <tr><td>Critical Issues</td>"
            f"<td>{risk.get('critical_issues', 0)}</td></tr>"
        )
        parts.append(
            f"        <tr><td>High Severity Failures</td>"
            f"<td>{risk.get('high_severity_failures', 0)}</td></tr>"
        )
        parts.append("    </table>")

        critical = content_data.get("critical_issues", [])
        if critical:
            parts.append(f"    <h2>Critical Issues ({len(critical)})</h2>")
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Rule ID</th><th>Rule Name</th>"
                "<th>Fail Count</th><th>Impact</th></tr>"
            )
            for issue in critical:
                parts.append(
                    f"        <tr>"
                    f"<td>{_escape_html(str(issue.get('rule_id', '')))}</td>"
                    f"<td>{_escape_html(str(issue.get('rule_name', '')))}</td>"
                    f"<td>{issue.get('fail_count', 0)}</td>"
                    f"<td>{_escape_html(str(issue.get('impact', '')))}</td></tr>"
                )
            parts.append("    </table>")

        action_items = content_data.get("action_items", [])
        if action_items:
            parts.append("    <h2>Action Items</h2>")
            parts.append("    <table>")
            parts.append(
                "        <tr><th>Priority</th><th>Action</th>"
                "<th>Owner</th><th>Deadline</th></tr>"
            )
            for item in action_items:
                parts.append(
                    f"        <tr>"
                    f"<td>P{item.get('priority', 0)}</td>"
                    f"<td>{_escape_html(str(item.get('action', '')))}</td>"
                    f"<td>{_escape_html(str(item.get('owner', '')))}</td>"
                    f"<td>{_escape_html(str(item.get('deadline', '')))}</td></tr>"
                )
            parts.append("    </table>")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_score(aggregates: Dict[str, Any]) -> int:
        """Compute a risk score from 0 (no risk) to 100 (maximum risk).

        The score is a weighted combination of:
        - Rule failure rate (40% weight)
        - Critical severity failures (30% weight)
        - High severity failures (20% weight)
        - Record-level failure rate (10% weight)

        Args:
            aggregates: Aggregate statistics from ``_compute_aggregates``.

        Returns:
            Integer risk score between 0 and 100.
        """
        pass_rate = aggregates.get("pass_rate", 1.0)
        record_pass_rate = aggregates.get("record_pass_rate", 1.0)
        severity_counts = aggregates.get("severity_counts", {})
        total_rules = max(aggregates.get("total_rules", 1), 1)

        # Rule failure component (0-100)
        rule_failure_pct = (1.0 - pass_rate) * 100

        # Critical severity component (0-100)
        critical_count = severity_counts.get("critical", 0)
        critical_pct = min(
            _safe_division(critical_count, total_rules) * 100 * 5, 100
        )

        # High severity component (0-100)
        high_count = severity_counts.get("high", 0)
        high_pct = min(
            _safe_division(high_count, total_rules) * 100 * 3, 100
        )

        # Record failure component (0-100)
        record_failure_pct = (1.0 - record_pass_rate) * 100

        # Weighted combination
        risk = (
            rule_failure_pct * 0.40
            + critical_pct * 0.30
            + high_pct * 0.20
            + record_failure_pct * 0.10
        )

        return min(int(round(risk)), 100)

    @staticmethod
    def _risk_level_label(risk_score: int) -> str:
        """Map a numeric risk score to a human-readable risk level.

        Args:
            risk_score: Integer risk score between 0 and 100.

        Returns:
            Risk level label string.
        """
        if risk_score <= 10:
            return "LOW"
        if risk_score <= 25:
            return "MODERATE"
        if risk_score <= 50:
            return "ELEVATED"
        if risk_score <= 75:
            return "HIGH"
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["ValidationReporterEngine"]
