# -*- coding: utf-8 -*-
"""
Article 9 Compliance Reporter - AGENT-EUDR-002

Generates EUDR Article 9 compliance reports for geolocation verification data.
Provides per-plot, per-commodity, and per-operator compliance assessments
showing verification status, identified issues, remediation requirements,
and compliance readiness scores.

Article 9 of EU Regulation 2023/1115 (EUDR) sets out the geolocation
requirements that operators must satisfy for each production plot:
    (1)(a) Coordinates of the production plot(s)
    (1)(b-c) Polygon boundary if >4 hectares
    (1)(d) Valid polygon vertices with sufficient precision
    + No overlap with protected areas
    + No deforestation detected post-cutoff (2020-12-31)

This reporter evaluates verification results against these requirements
and produces structured compliance reports with SHA-256 provenance hashes
for tamper-detection and audit trail integrity.

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean logic.
    - Compliance rates use standard arithmetic (no ML/LLM).
    - SHA-256 provenance hashes on all report objects.
    - No probabilistic models in compliance determination.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
Regulation: EU 2023/1115 (EUDR) Article 9
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateValidationResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    QualityTier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARTICLE_9_POLYGON_THRESHOLD_HA: float = 4.0
"""Plots larger than 4 hectares require polygon boundaries per Art 9(1)(b-c)."""

_DEFAULT_AVG_FIX_TIME_HOURS: float = 2.0
"""Default estimated hours to remediate a single non-compliant plot."""

_REPORT_VERSION: str = "1.0.0"
"""Report schema version for forward compatibility."""

# ---------------------------------------------------------------------------
# EUDR-relevant country name mapping (ISO 3166-1 alpha-2 -> name)
# ---------------------------------------------------------------------------

COUNTRY_NAMES: Dict[str, str] = {
    # Major EUDR commodity-producing countries
    "BR": "Brazil",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "CO": "Colombia",
    "PE": "Peru",
    "BO": "Bolivia",
    "EC": "Ecuador",
    "VE": "Venezuela",
    "GY": "Guyana",
    "SR": "Suriname",
    "PY": "Paraguay",
    "AR": "Argentina",
    "UY": "Uruguay",
    "CL": "Chile",
    # Central America
    "MX": "Mexico",
    "GT": "Guatemala",
    "HN": "Honduras",
    "NI": "Nicaragua",
    "CR": "Costa Rica",
    "PA": "Panama",
    "BZ": "Belize",
    "SV": "El Salvador",
    # Africa
    "GH": "Ghana",
    "CI": "Cote d'Ivoire",
    "CM": "Cameroon",
    "NG": "Nigeria",
    "CD": "Democratic Republic of the Congo",
    "CG": "Republic of the Congo",
    "GA": "Gabon",
    "GQ": "Equatorial Guinea",
    "ET": "Ethiopia",
    "KE": "Kenya",
    "TZ": "Tanzania",
    "UG": "Uganda",
    "RW": "Rwanda",
    "MG": "Madagascar",
    "MZ": "Mozambique",
    "ZA": "South Africa",
    "LR": "Liberia",
    "SL": "Sierra Leone",
    "GN": "Guinea",
    "SN": "Senegal",
    "ML": "Mali",
    "BF": "Burkina Faso",
    "TG": "Togo",
    "BJ": "Benin",
    # Southeast Asia
    "TH": "Thailand",
    "VN": "Vietnam",
    "PH": "Philippines",
    "MM": "Myanmar",
    "KH": "Cambodia",
    "LA": "Laos",
    "PG": "Papua New Guinea",
    "LK": "Sri Lanka",
    "IN": "India",
    "CN": "China",
    # EU / trade partners
    "DE": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "BE": "Belgium",
    "ES": "Spain",
    "IT": "Italy",
    "PT": "Portugal",
    "GB": "United Kingdom",
    "US": "United States",
    "CA": "Canada",
    "AU": "Australia",
    "JP": "Japan",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Data classes for Article 9 compliance reporting
# ---------------------------------------------------------------------------

@dataclass
class Article9Check:
    """Result of a single Article 9 requirement check.

    Attributes:
        requirement: Human-readable description of the requirement.
        article_reference: EUDR article/paragraph reference (e.g. 'Art 9(1)(a)').
        status: Check outcome -- 'passed', 'failed', or 'not_applicable'.
        details: Human-readable explanation of the check result.
    """

    requirement: str = ""
    article_reference: str = ""
    status: str = "failed"
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "requirement": self.requirement,
            "article_reference": self.article_reference,
            "status": self.status,
            "details": self.details,
        }

@dataclass
class PlotComplianceStatus:
    """Per-plot compliance assessment against Article 9 requirements.

    Attributes:
        plot_id: Unique plot identifier.
        status: Overall compliance status -- 'compliant', 'non_compliant', or 'pending'.
        checks: List of individual Article 9 requirement checks.
        issues_count: Total number of issues (all severities).
        critical_issues: Number of critical issues that block compliance.
        remediation_needed: Human-readable list of remediation actions required.
    """

    plot_id: str = ""
    status: str = "pending"
    checks: List[Article9Check] = field(default_factory=list)
    issues_count: int = 0
    critical_issues: int = 0
    remediation_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "plot_id": self.plot_id,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
            "issues_count": self.issues_count,
            "critical_issues": self.critical_issues,
            "remediation_needed": self.remediation_needed,
        }

@dataclass
class CommoditySummary:
    """Per-commodity compliance summary across all plots.

    Attributes:
        commodity: EUDR commodity identifier (e.g. 'palm_oil', 'soy', 'cocoa').
        total_plots: Total number of plots for this commodity.
        compliant: Number of fully compliant plots.
        non_compliant: Number of non-compliant plots.
        pending: Number of plots still awaiting verification.
        compliance_rate: Compliance rate as a percentage (0.0 - 100.0).
        average_accuracy_score: Average geolocation accuracy score (0.0 - 100.0).
        top_issues: Most frequently occurring issue types for this commodity.
    """

    commodity: str = ""
    total_plots: int = 0
    compliant: int = 0
    non_compliant: int = 0
    pending: int = 0
    compliance_rate: float = 0.0
    average_accuracy_score: float = 0.0
    top_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "commodity": self.commodity,
            "total_plots": self.total_plots,
            "compliant": self.compliant,
            "non_compliant": self.non_compliant,
            "pending": self.pending,
            "compliance_rate": round(self.compliance_rate, 2),
            "average_accuracy_score": round(self.average_accuracy_score, 2),
            "top_issues": self.top_issues,
        }

@dataclass
class CountrySummary:
    """Per-country compliance summary across all plots.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        country_name: Human-readable country name.
        total_plots: Total number of plots in this country.
        compliant: Number of compliant plots.
        non_compliant: Number of non-compliant plots.
        compliance_rate: Compliance rate as a percentage (0.0 - 100.0).
        risk_level: Country risk classification (low, medium, high, critical).
    """

    country_code: str = ""
    country_name: str = ""
    total_plots: int = 0
    compliant: int = 0
    non_compliant: int = 0
    compliance_rate: float = 0.0
    risk_level: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "country_code": self.country_code,
            "country_name": self.country_name,
            "total_plots": self.total_plots,
            "compliant": self.compliant,
            "non_compliant": self.non_compliant,
            "compliance_rate": round(self.compliance_rate, 2),
            "risk_level": self.risk_level,
        }

@dataclass
class RemediationPriority:
    """A prioritized remediation item for a non-compliant plot.

    Attributes:
        plot_id: Plot requiring remediation.
        priority_rank: Priority rank (1 = highest priority).
        issues: List of issue descriptions requiring remediation.
        estimated_effort_hours: Estimated hours to remediate all issues.
        impact_description: Human-readable impact of non-compliance.
    """

    plot_id: str = ""
    priority_rank: int = 0
    issues: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 0.0
    impact_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "plot_id": self.plot_id,
            "priority_rank": self.priority_rank,
            "issues": self.issues,
            "estimated_effort_hours": round(self.estimated_effort_hours, 2),
            "impact_description": self.impact_description,
        }

@dataclass
class TrendPoint:
    """A single data point in a compliance trend time-series.

    Attributes:
        date: ISO 8601 date string (YYYY-MM-DD).
        compliance_rate: Compliance rate at this point in time (0.0 - 100.0).
        total_plots: Total plots at this snapshot.
        compliant_plots: Compliant plots at this snapshot.
    """

    date: str = ""
    compliance_rate: float = 0.0
    total_plots: int = 0
    compliant_plots: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "date": self.date,
            "compliance_rate": round(self.compliance_rate, 2),
            "total_plots": self.total_plots,
            "compliant_plots": self.compliant_plots,
        }

@dataclass
class ComplianceReport:
    """Complete Article 9 compliance report for an operator.

    This is the top-level report structure returned by
    ``Article9ComplianceReporter.generate_report()``. It aggregates
    per-plot compliance statuses, per-commodity and per-country summaries,
    remediation priorities, and trend data.

    Attributes:
        report_id: Unique report identifier.
        operator_id: Operator for whom the report was generated.
        generated_at: UTC timestamp of report generation.
        report_version: Schema version for forward compatibility.
        total_plots: Total number of plots evaluated.
        compliant_plots: Number of fully compliant plots.
        non_compliant_plots: Number of non-compliant plots.
        pending_plots: Number of plots pending verification.
        overall_compliance_rate: Overall compliance rate (0.0 - 100.0).
        plot_statuses: Per-plot compliance assessment results.
        commodity_summaries: Per-commodity compliance summaries.
        country_summaries: Per-country compliance summaries.
        remediation_priorities: Prioritized remediation items.
        trend_data: Historical compliance trend data points.
        estimated_total_effort: Estimated total remediation effort.
        provenance_hash: SHA-256 hash for tamper detection.
        processing_time_ms: Report generation duration in milliseconds.
    """

    report_id: str = field(default_factory=lambda: _generate_id("RPT"))
    operator_id: str = ""
    generated_at: datetime = field(default_factory=utcnow)
    report_version: str = _REPORT_VERSION
    total_plots: int = 0
    compliant_plots: int = 0
    non_compliant_plots: int = 0
    pending_plots: int = 0
    overall_compliance_rate: float = 0.0
    plot_statuses: List[PlotComplianceStatus] = field(default_factory=list)
    commodity_summaries: Dict[str, CommoditySummary] = field(default_factory=dict)
    country_summaries: Dict[str, CountrySummary] = field(default_factory=dict)
    remediation_priorities: List[RemediationPriority] = field(default_factory=list)
    trend_data: List[TrendPoint] = field(default_factory=list)
    estimated_total_effort: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "report_id": self.report_id,
            "operator_id": self.operator_id,
            "generated_at": self.generated_at.isoformat(),
            "report_version": self.report_version,
            "total_plots": self.total_plots,
            "compliant_plots": self.compliant_plots,
            "non_compliant_plots": self.non_compliant_plots,
            "pending_plots": self.pending_plots,
            "overall_compliance_rate": round(self.overall_compliance_rate, 2),
            "plot_statuses": [p.to_dict() for p in self.plot_statuses],
            "commodity_summaries": {
                k: v.to_dict() for k, v in self.commodity_summaries.items()
            },
            "country_summaries": {
                k: v.to_dict() for k, v in self.country_summaries.items()
            },
            "remediation_priorities": [
                r.to_dict() for r in self.remediation_priorities
            ],
            "trend_data": [t.to_dict() for t in self.trend_data],
            "estimated_total_effort": self.estimated_total_effort,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Verification result bundle (passed per-plot to the reporter)
# ---------------------------------------------------------------------------

@dataclass
class PlotVerificationBundle:
    """Bundle of all verification results for a single plot.

    This dataclass groups together the results from every verification
    engine for a given plot, enabling the Article 9 reporter to perform
    a comprehensive compliance assessment.

    Attributes:
        plot_id: Unique plot identifier.
        commodity: EUDR commodity type.
        country_code: ISO 3166-1 alpha-2 country code.
        declared_area_ha: Operator-declared area in hectares.
        has_coordinates: Whether GPS coordinates were provided.
        has_polygon: Whether polygon boundary was provided.
        polygon_vertex_count: Number of polygon vertices (0 if no polygon).
        coordinate_result: Coordinate validation result (or None).
        polygon_result: Polygon verification result (or None).
        protected_area_result: Protected area check result (or None).
        deforestation_result: Deforestation verification result (or None).
        accuracy_score: Geolocation accuracy score (or None).
    """

    plot_id: str = ""
    commodity: str = ""
    country_code: str = ""
    declared_area_ha: float = 0.0
    has_coordinates: bool = False
    has_polygon: bool = False
    polygon_vertex_count: int = 0
    coordinate_result: Optional[CoordinateValidationResult] = None
    polygon_result: Optional[PolygonVerificationResult] = None
    protected_area_result: Optional[ProtectedAreaCheckResult] = None
    deforestation_result: Optional[DeforestationVerificationResult] = None
    accuracy_score: Optional[GeolocationAccuracyScore] = None

# ---------------------------------------------------------------------------
# Article9ComplianceReporter
# ---------------------------------------------------------------------------

class Article9ComplianceReporter:
    """Generates Article 9-specific compliance reports for EUDR geolocation data.

    Provides per-plot, per-commodity, and per-operator compliance assessments
    showing verification status, identified issues, remediation requirements,
    and compliance readiness scores.

    This reporter evaluates verification results from the CoordinateValidator,
    PolygonTopologyVerifier, ProtectedAreaChecker, DeforestationCutoffVerifier,
    and AccuracyScoringEngine against the six core Article 9 requirements:

        1. Has coordinates (Art 9(1)(a))
        2. Has polygon if plot >4 ha (Art 9(1)(b-c))
        3. Polygon vertices are valid (Art 9(1)(d))
        4. Coordinates verified against country (Art 9 + due diligence)
        5. No protected area overlap (Art 3 + Annex)
        6. No deforestation detected (Art 3(a) + Art 2(1))

    All compliance logic is deterministic (pure boolean checks + arithmetic).
    SHA-256 provenance hashes are computed on every report for tamper detection.

    Attributes:
        polygon_threshold_ha: Area threshold above which polygon is required.

    Example:
        >>> reporter = Article9ComplianceReporter()
        >>> bundles = [PlotVerificationBundle(plot_id="P1", ...)]
        >>> report = reporter.generate_report("OP-001", bundles)
        >>> assert report.provenance_hash != ""
        >>> assert report.overall_compliance_rate >= 0.0
    """

    def __init__(
        self,
        polygon_threshold_ha: float = _ARTICLE_9_POLYGON_THRESHOLD_HA,
        avg_fix_time_hours: float = _DEFAULT_AVG_FIX_TIME_HOURS,
    ) -> None:
        """Initialize Article9ComplianceReporter.

        Args:
            polygon_threshold_ha: Area threshold in hectares above which
                a polygon boundary is required per Art 9(1)(b-c). Default 4.0.
            avg_fix_time_hours: Average estimated hours to remediate a
                single non-compliant plot. Used for effort estimation.
        """
        self._polygon_threshold_ha = polygon_threshold_ha
        self._avg_fix_time_hours = avg_fix_time_hours
        logger.info(
            "Article9ComplianceReporter initialized: "
            "polygon_threshold=%.1f ha, avg_fix_time=%.1f hrs",
            self._polygon_threshold_ha,
            self._avg_fix_time_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        operator_id: str,
        plots_with_results: List[PlotVerificationBundle],
        commodity_filter: Optional[str] = None,
        export_format: str = "json",
        historical_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> ComplianceReport:
        """Generate a complete Article 9 compliance report for an operator.

        Evaluates all plots against Article 9 requirements, aggregates
        per-commodity and per-country summaries, identifies remediation
        priorities, computes trend data from historical snapshots, and
        signs the report with a SHA-256 provenance hash.

        Args:
            operator_id: Unique operator identifier.
            plots_with_results: List of PlotVerificationBundle objects,
                one per production plot, each containing verification
                results from all engines.
            commodity_filter: Optional commodity to filter by. If set,
                only plots matching this commodity are included.
            export_format: Report export format ('json', 'csv', 'pdf_data').
                This does not change the return type -- the ComplianceReport
                is always returned, but its internal representation can
                be accessed via _export_as_json/_export_as_csv/_export_as_pdf_data.
            historical_snapshots: Optional list of historical compliance
                snapshots for trend analysis. Each snapshot must include
                'date', 'total_plots', and 'compliant_plots' keys.

        Returns:
            ComplianceReport with all assessments, summaries, and provenance hash.

        Raises:
            ValueError: If operator_id is empty.
        """
        start_time = time.monotonic()

        if not operator_id:
            raise ValueError("operator_id must not be empty")

        logger.info(
            "Generating Article 9 compliance report: operator=%s, "
            "plots=%d, commodity_filter=%s, format=%s",
            operator_id,
            len(plots_with_results),
            commodity_filter or "all",
            export_format,
        )

        # Step 1: Apply commodity filter
        filtered_plots = self._apply_commodity_filter(
            plots_with_results, commodity_filter
        )

        # Step 2: Assess compliance for each plot
        plot_statuses = self._assess_all_plots(filtered_plots)

        # Step 3: Count overall compliance
        compliant_count, non_compliant_count, pending_count = (
            self._count_compliance(plot_statuses)
        )
        total_count = len(plot_statuses)

        # Step 4: Calculate overall compliance rate
        overall_rate = self._calculate_compliance_rate(
            total_count, compliant_count
        )

        # Step 5: Generate commodity summaries
        commodity_summaries = self._generate_commodity_summary(
            filtered_plots, plot_statuses
        )

        # Step 6: Generate country summaries
        country_summaries = self._generate_country_summary(
            filtered_plots, plot_statuses
        )

        # Step 7: Identify remediation priorities
        non_compliant_statuses = [
            ps for ps in plot_statuses if ps.status == "non_compliant"
        ]
        remediation_priorities = self._identify_remediation_priorities(
            non_compliant_statuses
        )

        # Step 8: Calculate estimated effort
        estimated_effort = self._calculate_estimated_effort(
            len(non_compliant_statuses), self._avg_fix_time_hours
        )

        # Step 9: Generate trend data
        trend_data = self._generate_trend_data(
            historical_snapshots or []
        )

        # Step 10: Build the report
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        report = ComplianceReport(
            operator_id=operator_id,
            generated_at=utcnow(),
            total_plots=total_count,
            compliant_plots=compliant_count,
            non_compliant_plots=non_compliant_count,
            pending_plots=pending_count,
            overall_compliance_rate=overall_rate,
            plot_statuses=plot_statuses,
            commodity_summaries=commodity_summaries,
            country_summaries=country_summaries,
            remediation_priorities=remediation_priorities,
            trend_data=trend_data,
            estimated_total_effort=estimated_effort,
            processing_time_ms=elapsed_ms,
        )

        # Step 11: Compute provenance hash
        report.provenance_hash = self._compute_report_hash(report)

        logger.info(
            "Article 9 compliance report generated: operator=%s, "
            "total=%d, compliant=%d (%.1f%%), non_compliant=%d, "
            "pending=%d, hash=%s, elapsed=%.1fms",
            operator_id,
            total_count,
            compliant_count,
            overall_rate,
            non_compliant_count,
            pending_count,
            report.provenance_hash[:12],
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Plot-level compliance assessment
    # ------------------------------------------------------------------

    def _assess_plot_compliance(
        self,
        plot_id: str,
        bundle: PlotVerificationBundle,
    ) -> PlotComplianceStatus:
        """Assess a single plot against all Article 9 requirements.

        Runs each Article 9 check and derives overall compliance status.
        A plot is 'compliant' only if all checks pass. A single failed
        check makes the plot 'non_compliant'. If verification data is
        incomplete, the plot is 'pending'.

        Args:
            plot_id: Unique plot identifier.
            bundle: PlotVerificationBundle with all engine results.

        Returns:
            PlotComplianceStatus with detailed check results.
        """
        checks = self._check_article9_requirements(bundle)

        # Count issues
        failed_checks = [c for c in checks if c.status == "failed"]
        issues_count = len(failed_checks)

        # Critical issues are coordinate missing, deforestation detected,
        # or protected area overlap
        critical_refs = {"Art 9(1)(a)", "Art 3(a)", "Art 3 + Annex"}
        critical_issues = sum(
            1 for c in failed_checks if c.article_reference in critical_refs
        )

        # Build remediation list from failed checks
        remediation_needed = [
            f"[{c.article_reference}] {c.details}"
            for c in failed_checks
        ]

        # Determine overall status
        has_pending = any(c.status == "not_applicable" for c in checks)
        if issues_count == 0 and not has_pending:
            status = "compliant"
        elif issues_count > 0:
            status = "non_compliant"
        else:
            status = "pending"

        return PlotComplianceStatus(
            plot_id=plot_id,
            status=status,
            checks=checks,
            issues_count=issues_count,
            critical_issues=critical_issues,
            remediation_needed=remediation_needed,
        )

    def _check_article9_requirements(
        self,
        bundle: PlotVerificationBundle,
    ) -> List[Article9Check]:
        """Run all six Article 9 requirement checks against a plot.

        The six checks are:
            1. Has coordinates (Art 9(1)(a))
            2. Has polygon if >4 ha (Art 9(1)(b-c))
            3. Polygon vertices valid (Art 9(1)(d))
            4. Coordinates verified (Art 9 due diligence)
            5. No protected area overlap (Art 3 + Annex)
            6. No deforestation detected (Art 3(a))

        Args:
            bundle: PlotVerificationBundle with all verification results.

        Returns:
            List of Article9Check results, one per requirement.
        """
        checks: List[Article9Check] = []

        # Check 1: Has coordinates (Art 9(1)(a))
        checks.append(self._check_has_coordinates(bundle))

        # Check 2: Has polygon if >4 ha (Art 9(1)(b-c))
        checks.append(self._check_polygon_required(bundle))

        # Check 3: Polygon vertices valid (Art 9(1)(d))
        checks.append(self._check_polygon_validity(bundle))

        # Check 4: Coordinates verified (Art 9 due diligence)
        checks.append(self._check_coordinates_verified(bundle))

        # Check 5: No protected area overlap (Art 3 + Annex)
        checks.append(self._check_protected_area(bundle))

        # Check 6: No deforestation detected (Art 3(a))
        checks.append(self._check_deforestation(bundle))

        return checks

    # ------------------------------------------------------------------
    # Individual Article 9 checks
    # ------------------------------------------------------------------

    def _check_has_coordinates(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check whether the plot has GPS coordinates (Art 9(1)(a)).

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail result.
        """
        if bundle.has_coordinates:
            return Article9Check(
                requirement="Plot must have GPS coordinates",
                article_reference="Art 9(1)(a)",
                status="passed",
                details="GPS coordinates provided",
            )
        return Article9Check(
            requirement="Plot must have GPS coordinates",
            article_reference="Art 9(1)(a)",
            status="failed",
            details="No GPS coordinates provided for this plot",
        )

    def _check_polygon_required(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check whether polygon boundary is provided if area >4 ha (Art 9(1)(b-c)).

        Plots at or below the threshold only require point coordinates.
        Plots above the threshold require a polygon boundary definition.

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail/not_applicable result.
        """
        if bundle.declared_area_ha <= self._polygon_threshold_ha:
            return Article9Check(
                requirement=(
                    f"Polygon required if area >{self._polygon_threshold_ha} ha"
                ),
                article_reference="Art 9(1)(b-c)",
                status="not_applicable",
                details=(
                    f"Plot area ({bundle.declared_area_ha:.2f} ha) is at or below "
                    f"the {self._polygon_threshold_ha} ha threshold; "
                    "point coordinates suffice"
                ),
            )

        if bundle.has_polygon:
            return Article9Check(
                requirement=(
                    f"Polygon required if area >{self._polygon_threshold_ha} ha"
                ),
                article_reference="Art 9(1)(b-c)",
                status="passed",
                details=(
                    f"Polygon boundary provided for plot with area "
                    f"{bundle.declared_area_ha:.2f} ha"
                ),
            )

        return Article9Check(
            requirement=(
                f"Polygon required if area >{self._polygon_threshold_ha} ha"
            ),
            article_reference="Art 9(1)(b-c)",
            status="failed",
            details=(
                f"Plot area ({bundle.declared_area_ha:.2f} ha) exceeds "
                f"{self._polygon_threshold_ha} ha but no polygon boundary provided"
            ),
        )

    def _check_polygon_validity(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check whether polygon vertices are valid (Art 9(1)(d)).

        Verifies that the polygon passes topology verification:
        ring closure, no self-intersection, no slivers/spikes, and
        area within declared tolerance.

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail/not_applicable result.
        """
        if not bundle.has_polygon:
            return Article9Check(
                requirement="Polygon vertices must be valid",
                article_reference="Art 9(1)(d)",
                status="not_applicable",
                details="No polygon provided; vertex validation not required",
            )

        if bundle.polygon_result is None:
            return Article9Check(
                requirement="Polygon vertices must be valid",
                article_reference="Art 9(1)(d)",
                status="not_applicable",
                details="Polygon verification not yet performed",
            )

        if bundle.polygon_result.is_valid:
            return Article9Check(
                requirement="Polygon vertices must be valid",
                article_reference="Art 9(1)(d)",
                status="passed",
                details=(
                    f"Polygon topology valid: {bundle.polygon_result.vertex_count} "
                    f"vertices, area={bundle.polygon_result.calculated_area_ha:.2f} ha"
                ),
            )

        # Build failure details from polygon issues
        issue_codes = [
            iss.code for iss in bundle.polygon_result.issues
        ]
        return Article9Check(
            requirement="Polygon vertices must be valid",
            article_reference="Art 9(1)(d)",
            status="failed",
            details=(
                f"Polygon topology invalid: {len(issue_codes)} issue(s) "
                f"detected [{', '.join(issue_codes[:5])}]"
            ),
        )

    def _check_coordinates_verified(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check whether coordinates have been verified (Art 9 due diligence).

        Verifies that coordinate validation passed (WGS84 valid, country
        match, precision adequate, on land).

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail/not_applicable result.
        """
        if bundle.coordinate_result is None:
            return Article9Check(
                requirement="Coordinates must be verified",
                article_reference="Art 9(1)(a) DD",
                status="not_applicable",
                details="Coordinate verification not yet performed",
            )

        if bundle.coordinate_result.is_valid:
            return Article9Check(
                requirement="Coordinates must be verified",
                article_reference="Art 9(1)(a) DD",
                status="passed",
                details=(
                    f"Coordinates verified: precision={bundle.coordinate_result.precision_decimal_places} "
                    f"decimals, country_match={bundle.coordinate_result.country_match}"
                ),
            )

        issue_count = len(bundle.coordinate_result.issues)
        return Article9Check(
            requirement="Coordinates must be verified",
            article_reference="Art 9(1)(a) DD",
            status="failed",
            details=(
                f"Coordinate verification failed: {issue_count} issue(s) detected"
            ),
        )

    def _check_protected_area(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check that plot does not overlap with protected areas (Art 3 + Annex).

        EUDR prohibits commodities sourced from plots that overlap with
        legally protected forests or areas of high conservation value.

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail/not_applicable result.
        """
        if bundle.protected_area_result is None:
            return Article9Check(
                requirement="No overlap with protected areas",
                article_reference="Art 3 + Annex",
                status="not_applicable",
                details="Protected area check not yet performed",
            )

        if not bundle.protected_area_result.overlaps_protected:
            return Article9Check(
                requirement="No overlap with protected areas",
                article_reference="Art 3 + Annex",
                status="passed",
                details="No protected area overlap detected",
            )

        area_name = (
            bundle.protected_area_result.protected_area_name or "unknown"
        )
        overlap_pct = bundle.protected_area_result.overlap_percentage
        return Article9Check(
            requirement="No overlap with protected areas",
            article_reference="Art 3 + Annex",
            status="failed",
            details=(
                f"Overlaps protected area '{area_name}' "
                f"({overlap_pct:.1f}% overlap)"
            ),
        )

    def _check_deforestation(
        self, bundle: PlotVerificationBundle
    ) -> Article9Check:
        """Check that no deforestation was detected post-cutoff (Art 3(a)).

        EUDR requires that commodities are 'deforestation-free', meaning
        produced on land not subject to deforestation after 31 Dec 2020.

        Args:
            bundle: Plot verification bundle.

        Returns:
            Article9Check with pass/fail/not_applicable result.
        """
        if bundle.deforestation_result is None:
            return Article9Check(
                requirement="No deforestation detected post-cutoff",
                article_reference="Art 3(a)",
                status="not_applicable",
                details="Deforestation verification not yet performed",
            )

        if not bundle.deforestation_result.deforestation_detected:
            return Article9Check(
                requirement="No deforestation detected post-cutoff",
                article_reference="Art 3(a)",
                status="passed",
                details=(
                    f"No deforestation detected after "
                    f"{bundle.deforestation_result.cutoff_date}"
                ),
            )

        return Article9Check(
            requirement="No deforestation detected post-cutoff",
            article_reference="Art 3(a)",
            status="failed",
            details=(
                f"Deforestation detected: {bundle.deforestation_result.alert_count} "
                f"alert(s), {bundle.deforestation_result.forest_loss_ha:.2f} ha "
                f"forest loss after {bundle.deforestation_result.cutoff_date} "
                f"(confidence: {bundle.deforestation_result.confidence:.0%})"
            ),
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _apply_commodity_filter(
        self,
        plots: List[PlotVerificationBundle],
        commodity_filter: Optional[str],
    ) -> List[PlotVerificationBundle]:
        """Filter plots by commodity if a filter is specified.

        Args:
            plots: All plot bundles.
            commodity_filter: Commodity to filter by, or None for all.

        Returns:
            Filtered list of plot bundles.
        """
        if commodity_filter is None:
            return plots
        filtered = [
            p for p in plots
            if p.commodity.lower() == commodity_filter.lower()
        ]
        logger.debug(
            "Commodity filter '%s' applied: %d of %d plots match",
            commodity_filter,
            len(filtered),
            len(plots),
        )
        return filtered

    def _assess_all_plots(
        self,
        plots: List[PlotVerificationBundle],
    ) -> List[PlotComplianceStatus]:
        """Assess compliance for all plots.

        Args:
            plots: List of plot bundles to assess.

        Returns:
            List of PlotComplianceStatus results.
        """
        statuses: List[PlotComplianceStatus] = []
        for bundle in plots:
            status = self._assess_plot_compliance(bundle.plot_id, bundle)
            statuses.append(status)
        return statuses

    def _count_compliance(
        self,
        plot_statuses: List[PlotComplianceStatus],
    ) -> Tuple[int, int, int]:
        """Count compliant, non-compliant, and pending plots.

        Args:
            plot_statuses: List of per-plot compliance statuses.

        Returns:
            Tuple of (compliant_count, non_compliant_count, pending_count).
        """
        compliant = sum(
            1 for ps in plot_statuses if ps.status == "compliant"
        )
        non_compliant = sum(
            1 for ps in plot_statuses if ps.status == "non_compliant"
        )
        pending = sum(
            1 for ps in plot_statuses if ps.status == "pending"
        )
        return compliant, non_compliant, pending

    def _generate_commodity_summary(
        self,
        plots: List[PlotVerificationBundle],
        plot_statuses: List[PlotComplianceStatus],
    ) -> Dict[str, CommoditySummary]:
        """Generate per-commodity compliance summaries.

        Groups plots by commodity, counts compliance outcomes, calculates
        average accuracy scores, and identifies the most common issues.

        Args:
            plots: List of plot bundles.
            plot_statuses: Corresponding compliance statuses.

        Returns:
            Dictionary mapping commodity name to CommoditySummary.
        """
        # Build a map from plot_id to status
        status_map: Dict[str, PlotComplianceStatus] = {
            ps.plot_id: ps for ps in plot_statuses
        }

        # Group plots by commodity
        commodity_groups: Dict[str, List[PlotVerificationBundle]] = {}
        for bundle in plots:
            commodity = bundle.commodity or "unknown"
            commodity_groups.setdefault(commodity, []).append(bundle)

        summaries: Dict[str, CommoditySummary] = {}
        for commodity, bundles in commodity_groups.items():
            compliant = 0
            non_compliant = 0
            pending = 0
            accuracy_scores: List[float] = []
            issue_counter: Dict[str, int] = {}

            for bundle in bundles:
                ps = status_map.get(bundle.plot_id)
                if ps is None:
                    pending += 1
                    continue

                if ps.status == "compliant":
                    compliant += 1
                elif ps.status == "non_compliant":
                    non_compliant += 1
                else:
                    pending += 1

                # Collect accuracy scores
                if bundle.accuracy_score is not None:
                    accuracy_scores.append(
                        float(bundle.accuracy_score.total_score)
                    )

                # Count issue types from failed checks
                for check in ps.checks:
                    if check.status == "failed":
                        issue_counter[check.requirement] = (
                            issue_counter.get(check.requirement, 0) + 1
                        )

            total = len(bundles)
            avg_score = (
                sum(accuracy_scores) / len(accuracy_scores)
                if accuracy_scores
                else 0.0
            )

            # Top issues sorted by frequency
            top_issues = sorted(
                issue_counter.keys(),
                key=lambda k: issue_counter[k],
                reverse=True,
            )[:5]

            summaries[commodity] = CommoditySummary(
                commodity=commodity,
                total_plots=total,
                compliant=compliant,
                non_compliant=non_compliant,
                pending=pending,
                compliance_rate=self._calculate_compliance_rate(
                    total, compliant
                ),
                average_accuracy_score=avg_score,
                top_issues=top_issues,
            )

        return summaries

    def _generate_country_summary(
        self,
        plots: List[PlotVerificationBundle],
        plot_statuses: List[PlotComplianceStatus],
    ) -> Dict[str, CountrySummary]:
        """Generate per-country compliance summaries.

        Groups plots by country code, counts compliance outcomes, and
        assigns a risk level based on compliance rate:
            >= 90%: low
            >= 70%: medium
            >= 50%: high
            <  50%: critical

        Args:
            plots: List of plot bundles.
            plot_statuses: Corresponding compliance statuses.

        Returns:
            Dictionary mapping country code to CountrySummary.
        """
        status_map: Dict[str, PlotComplianceStatus] = {
            ps.plot_id: ps for ps in plot_statuses
        }

        # Group plots by country
        country_groups: Dict[str, List[PlotVerificationBundle]] = {}
        for bundle in plots:
            country = bundle.country_code or "XX"
            country_groups.setdefault(country, []).append(bundle)

        summaries: Dict[str, CountrySummary] = {}
        for country_code, bundles in country_groups.items():
            compliant = 0
            non_compliant = 0

            for bundle in bundles:
                ps = status_map.get(bundle.plot_id)
                if ps is None:
                    continue
                if ps.status == "compliant":
                    compliant += 1
                elif ps.status == "non_compliant":
                    non_compliant += 1

            total = len(bundles)
            rate = self._calculate_compliance_rate(total, compliant)
            risk_level = self._derive_country_risk_level(rate)

            summaries[country_code] = CountrySummary(
                country_code=country_code,
                country_name=COUNTRY_NAMES.get(country_code, country_code),
                total_plots=total,
                compliant=compliant,
                non_compliant=non_compliant,
                compliance_rate=rate,
                risk_level=risk_level,
            )

        return summaries

    def _derive_country_risk_level(self, compliance_rate: float) -> str:
        """Derive country risk level from compliance rate.

        Args:
            compliance_rate: Compliance rate as a percentage (0.0 - 100.0).

        Returns:
            Risk level string: 'low', 'medium', 'high', or 'critical'.
        """
        if compliance_rate >= 90.0:
            return "low"
        if compliance_rate >= 70.0:
            return "medium"
        if compliance_rate >= 50.0:
            return "high"
        return "critical"

    def _identify_remediation_priorities(
        self,
        non_compliant_statuses: List[PlotComplianceStatus],
    ) -> List[RemediationPriority]:
        """Identify and rank remediation priorities for non-compliant plots.

        Plots are ranked by number of critical issues (descending), then
        by total issue count (descending). Each remediation item includes
        estimated effort and an impact description.

        Args:
            non_compliant_statuses: List of non-compliant plot statuses.

        Returns:
            List of RemediationPriority items, sorted by priority rank.
        """
        if not non_compliant_statuses:
            return []

        # Sort by critical issues (desc), then total issues (desc)
        sorted_statuses = sorted(
            non_compliant_statuses,
            key=lambda ps: (ps.critical_issues, ps.issues_count),
            reverse=True,
        )

        priorities: List[RemediationPriority] = []
        for rank, ps in enumerate(sorted_statuses, start=1):
            # Estimate effort: base time per issue + extra for critical
            effort = (
                ps.issues_count * self._avg_fix_time_hours
                + ps.critical_issues * 1.0  # extra hour per critical issue
            )

            # Build impact description
            if ps.critical_issues > 0:
                impact = (
                    f"CRITICAL: {ps.critical_issues} blocking issue(s) "
                    f"prevent EUDR compliance submission"
                )
            else:
                impact = (
                    f"{ps.issues_count} issue(s) require remediation "
                    f"before compliance submission"
                )

            priorities.append(RemediationPriority(
                plot_id=ps.plot_id,
                priority_rank=rank,
                issues=ps.remediation_needed,
                estimated_effort_hours=effort,
                impact_description=impact,
            ))

        return priorities

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _calculate_compliance_rate(
        self,
        total: int,
        compliant: int,
    ) -> float:
        """Calculate compliance rate as a percentage.

        Args:
            total: Total number of plots.
            compliant: Number of compliant plots.

        Returns:
            Compliance rate as a percentage (0.0 - 100.0).
            Returns 0.0 if total is zero to avoid division by zero.
        """
        if total <= 0:
            return 0.0
        return round((compliant / total) * 100.0, 2)

    def _calculate_estimated_effort(
        self,
        non_compliant_count: int,
        avg_fix_time_hours: float,
    ) -> Dict[str, Any]:
        """Calculate estimated total remediation effort.

        Args:
            non_compliant_count: Number of non-compliant plots.
            avg_fix_time_hours: Average hours to fix one plot.

        Returns:
            Dictionary with effort breakdown:
                - non_compliant_plots: Count of non-compliant plots
                - avg_fix_time_hours: Average fix time per plot
                - total_estimated_hours: Total estimated hours
                - total_estimated_days: Total estimated working days (8h/day)
                - fte_weeks: Full-time equivalent weeks (40h/week)
        """
        total_hours = non_compliant_count * avg_fix_time_hours
        total_days = total_hours / 8.0 if total_hours > 0 else 0.0
        fte_weeks = total_hours / 40.0 if total_hours > 0 else 0.0

        return {
            "non_compliant_plots": non_compliant_count,
            "avg_fix_time_hours": avg_fix_time_hours,
            "total_estimated_hours": round(total_hours, 2),
            "total_estimated_days": round(total_days, 2),
            "fte_weeks": round(fte_weeks, 2),
        }

    # ------------------------------------------------------------------
    # Trend data
    # ------------------------------------------------------------------

    def _generate_trend_data(
        self,
        historical_snapshots: List[Dict[str, Any]],
    ) -> List[TrendPoint]:
        """Generate compliance trend data from historical snapshots.

        Each snapshot must contain at minimum 'date', 'total_plots', and
        'compliant_plots' keys. Snapshots are sorted chronologically.

        Args:
            historical_snapshots: List of historical snapshot dictionaries.

        Returns:
            List of TrendPoint objects in chronological order.
        """
        if not historical_snapshots:
            return []

        trend_points: List[TrendPoint] = []
        for snapshot in historical_snapshots:
            date_str = str(snapshot.get("date", ""))
            total = int(snapshot.get("total_plots", 0))
            compliant = int(snapshot.get("compliant_plots", 0))
            rate = self._calculate_compliance_rate(total, compliant)

            trend_points.append(TrendPoint(
                date=date_str,
                compliance_rate=rate,
                total_plots=total,
                compliant_plots=compliant,
            ))

        # Sort chronologically by date
        trend_points.sort(key=lambda tp: tp.date)

        return trend_points

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def _export_as_json(self, report: ComplianceReport) -> str:
        """Export the compliance report as a JSON string.

        Produces a pretty-printed JSON representation of the full report
        including all plot statuses, summaries, and remediation priorities.

        Args:
            report: ComplianceReport to export.

        Returns:
            JSON string with 2-space indentation.
        """
        report_dict = report.to_dict()
        return json.dumps(report_dict, indent=2, default=str, ensure_ascii=False)

    def _export_as_csv(self, report: ComplianceReport) -> str:
        """Export the compliance report as a CSV string.

        Produces a CSV with one row per plot showing compliance status,
        issue counts, and Article 9 check results. Suitable for import
        into spreadsheets or BI tools.

        Columns:
            plot_id, status, issues_count, critical_issues,
            art9_1a, art9_1bc, art9_1d, art9_dd, art3_annex, art3a,
            remediation_needed

        Args:
            report: ComplianceReport to export.

        Returns:
            CSV string with header row and one data row per plot.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "plot_id",
            "status",
            "issues_count",
            "critical_issues",
            "art_9_1_a_coordinates",
            "art_9_1_bc_polygon",
            "art_9_1_d_vertices",
            "art_9_dd_verification",
            "art_3_annex_protected",
            "art_3_a_deforestation",
            "remediation_needed",
        ])

        # Data rows
        for ps in report.plot_statuses:
            # Map checks to columns by article reference
            check_map: Dict[str, str] = {}
            for check in ps.checks:
                check_map[check.article_reference] = check.status

            writer.writerow([
                ps.plot_id,
                ps.status,
                ps.issues_count,
                ps.critical_issues,
                check_map.get("Art 9(1)(a)", ""),
                check_map.get("Art 9(1)(b-c)", ""),
                check_map.get("Art 9(1)(d)", ""),
                check_map.get("Art 9(1)(a) DD", ""),
                check_map.get("Art 3 + Annex", ""),
                check_map.get("Art 3(a)", ""),
                "; ".join(ps.remediation_needed),
            ])

        return output.getvalue()

    def _export_as_pdf_data(self, report: ComplianceReport) -> Dict[str, Any]:
        """Export the compliance report as PDF-ready structured data.

        Returns a dictionary structured for PDF template rendering with
        sections for summary, per-plot details, commodity breakdown,
        country breakdown, and remediation priorities.

        Args:
            report: ComplianceReport to export.

        Returns:
            Dictionary with sections suitable for PDF template rendering.
        """
        return {
            "title": f"EUDR Article 9 Compliance Report - {report.operator_id}",
            "subtitle": (
                f"Generated {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            "report_id": report.report_id,
            "version": report.report_version,
            "provenance_hash": report.provenance_hash,
            "summary": {
                "total_plots": report.total_plots,
                "compliant_plots": report.compliant_plots,
                "non_compliant_plots": report.non_compliant_plots,
                "pending_plots": report.pending_plots,
                "overall_compliance_rate": round(
                    report.overall_compliance_rate, 2
                ),
                "processing_time_ms": round(report.processing_time_ms, 2),
            },
            "commodity_breakdown": {
                k: v.to_dict()
                for k, v in report.commodity_summaries.items()
            },
            "country_breakdown": {
                k: v.to_dict()
                for k, v in report.country_summaries.items()
            },
            "remediation_priorities": [
                r.to_dict() for r in report.remediation_priorities[:20]
            ],
            "estimated_effort": report.estimated_total_effort,
            "trend_data": [t.to_dict() for t in report.trend_data],
            "plot_details": [
                ps.to_dict() for ps in report.plot_statuses
            ],
        }

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_report_hash(self, report: ComplianceReport) -> str:
        """Compute SHA-256 hash of the compliance report for tamper detection.

        The hash covers the operator ID, generation timestamp, all plot
        compliance statuses, commodity summaries, and country summaries.
        This provides a cryptographic seal ensuring the report has not
        been altered after generation.

        Args:
            report: ComplianceReport to hash.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        # Build deterministic hash input from report contents
        hash_parts: List[str] = [
            report.report_id,
            report.operator_id,
            report.generated_at.isoformat(),
            str(report.total_plots),
            str(report.compliant_plots),
            str(report.non_compliant_plots),
            str(report.pending_plots),
            f"{report.overall_compliance_rate:.4f}",
        ]

        # Include plot statuses
        for ps in report.plot_statuses:
            hash_parts.append(f"{ps.plot_id}:{ps.status}:{ps.issues_count}")

        # Include commodity summaries
        for commodity in sorted(report.commodity_summaries.keys()):
            cs = report.commodity_summaries[commodity]
            hash_parts.append(
                f"{cs.commodity}:{cs.total_plots}:{cs.compliant}:"
                f"{cs.non_compliant}:{cs.compliance_rate:.4f}"
            )

        # Include country summaries
        for country in sorted(report.country_summaries.keys()):
            csm = report.country_summaries[country]
            hash_parts.append(
                f"{csm.country_code}:{csm.total_plots}:{csm.compliant}:"
                f"{csm.non_compliant}:{csm.compliance_rate:.4f}"
            )

        hash_input = "|".join(hash_parts)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Reporter
    "Article9ComplianceReporter",
    # Data classes
    "Article9Check",
    "PlotComplianceStatus",
    "CommoditySummary",
    "CountrySummary",
    "RemediationPriority",
    "TrendPoint",
    "ComplianceReport",
    "PlotVerificationBundle",
    # Constants
    "COUNTRY_NAMES",
]
