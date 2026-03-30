# -*- coding: utf-8 -*-
"""
Compliance Reporter Engine - AGENT-EUDR-010: Segregation Verifier (Feature 8)

Generates segregation compliance reports for EUDR regulatory submission,
competent authority inspections, internal audits, and trend analysis.
Supports four output formats (JSON, CSV, PDF data, EUDR XML), five report
types (audit, contamination, evidence, trend, supply_chain_summary), batch
generation, and five-year report retention tracking.

Zero-Hallucination Guarantees:
    - All report content is assembled from deterministic input data
    - Score summaries use arithmetic aggregation (no ML/LLM)
    - XML generation uses static namespace and schema templates
    - CSV formatting uses deterministic column ordering
    - Trend analysis uses simple time-series arithmetic
    - SHA-256 provenance hashes on all generated reports
    - No ML/LLM used for any report generation or summarization

Performance Targets:
    - Single report generation (JSON): <50ms
    - Single report generation (EUDR XML): <100ms
    - Batch report generation (100 facilities): <5 seconds
    - Report listing/retrieval: <10ms

Regulatory References:
    - EUDR Article 4: Due diligence statement requirements
    - EUDR Article 9: Geolocation and traceability data reporting
    - EUDR Article 10(2)(f): Segregation verification evidence
    - EUDR Article 14: Competent authority inspection support
    - EUDR Article 31: Five-year record retention
    - ISO 22095:2020: Chain of Custody reporting requirements

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 8: Segregation Compliance Reporting)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "rpt") -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported report types.
REPORT_TYPES: List[str] = [
    "audit",
    "contamination",
    "evidence",
    "trend",
    "supply_chain_summary",
]

#: Supported output formats.
SUPPORTED_FORMATS: List[str] = ["json", "csv", "pdf", "eudr_xml"]

#: EUDR XML namespace for segregation reports.
EUDR_XML_NAMESPACE: str = "urn:eu:eudr:segregation:2024"

#: Default report retention period in days (5 years per EUDR Article 31).
REPORT_RETENTION_DAYS: int = 1825

#: XML declaration header.
XML_DECLARATION: str = '<?xml version="1.0" encoding="UTF-8"?>'

#: EUDR XML schema location reference.
EUDR_XML_SCHEMA_LOCATION: str = (
    "urn:eu:eudr:segregation:2024 "
    "https://eudr.europa.eu/schemas/segregation-2024.xsd"
)

#: CSV column orderings for each report type.
CSV_COLUMNS: Dict[str, List[str]] = {
    "audit": [
        "report_id", "facility_id", "report_type", "overall_score",
        "scp_coverage", "storage_score", "transport_score",
        "processing_score", "labeling_score", "contamination_score",
        "recommendations_count", "generated_at",
    ],
    "contamination": [
        "report_id", "event_id", "facility_id", "pathway_type",
        "severity", "affected_batches_count", "affected_quantity_kg",
        "root_cause", "corrective_action", "resolved", "event_date",
        "generated_at",
    ],
    "evidence": [
        "report_id", "facility_id", "regulatory_framework",
        "evidence_items_count", "certifications_count",
        "assessment_count", "compliance_status", "generated_at",
    ],
    "trend": [
        "report_id", "facility_id", "period_months",
        "start_score", "end_score", "score_change",
        "trend_direction", "contamination_events_count",
        "generated_at",
    ],
    "supply_chain_summary": [
        "report_id", "supply_chain_id", "facilities_count",
        "avg_score", "min_score", "max_score",
        "overall_compliance_status", "generated_at",
    ],
}

# ---------------------------------------------------------------------------
# Internal Dataclass Result Types
# ---------------------------------------------------------------------------

@dataclass
class SegregationAuditReport:
    """Comprehensive segregation audit report for a facility.

    Attributes:
        report_id: Unique report identifier.
        facility_id: Assessed facility identifier.
        report_type: Always 'audit'.
        assessment_summary: Assessment scores and capability level.
        scp_coverage: SCP coverage statistics.
        storage_audit: Storage zone audit findings.
        transport_audit: Transport verification findings.
        processing_audit: Processing line verification findings.
        labeling_audit: Labeling compliance findings.
        contamination_summary: Contamination event summary.
        overall_score: Overall segregation compliance score (0-100).
        recommendations: Prioritized improvement recommendations.
        generated_at: ISO timestamp of report generation.
        format: Report output format.
        provenance_hash: SHA-256 hash for audit trail.
    """

    report_id: str
    facility_id: str
    report_type: str
    assessment_summary: Dict[str, Any]
    scp_coverage: Dict[str, Any]
    storage_audit: Dict[str, Any]
    transport_audit: Dict[str, Any]
    processing_audit: Dict[str, Any]
    labeling_audit: Dict[str, Any]
    contamination_summary: Dict[str, Any]
    overall_score: float
    recommendations: List[str]
    generated_at: str
    format: str
    provenance_hash: str = ""

@dataclass
class ContaminationReport:
    """Detailed contamination event report.

    Attributes:
        report_id: Unique report identifier.
        event_details: Full contamination event data.
        affected_batches: List of affected batch details.
        root_cause_analysis: Root cause analysis data.
        corrective_actions: Corrective actions taken or planned.
        impact_assessment: Downstream impact assessment data.
        timeline: Chronological event timeline.
        generated_at: ISO timestamp of report generation.
        format: Report output format.
        provenance_hash: SHA-256 hash for audit trail.
    """

    report_id: str
    event_details: Dict[str, Any]
    affected_batches: List[Dict[str, Any]]
    root_cause_analysis: Dict[str, Any]
    corrective_actions: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    generated_at: str
    format: str
    provenance_hash: str = ""

@dataclass
class EvidencePackage:
    """Evidence package for competent authority inspection.

    Attributes:
        report_id: Unique report identifier.
        facility_id: Facility identifier.
        regulatory_framework: Target regulatory framework.
        evidence_items: List of evidence item records.
        certifications: List of relevant certification records.
        assessment_history: Historical assessment records.
        compliance_status: Current compliance status.
        generated_at: ISO timestamp of package generation.
        format: Report output format.
        provenance_hash: SHA-256 hash for audit trail.
    """

    report_id: str
    facility_id: str
    regulatory_framework: str
    evidence_items: List[Dict[str, Any]]
    certifications: List[Dict[str, Any]]
    assessment_history: List[Dict[str, Any]]
    compliance_status: str
    generated_at: str
    format: str
    provenance_hash: str = ""

@dataclass
class TrendReport:
    """Score trend report over a period for a facility.

    Attributes:
        report_id: Unique report identifier.
        facility_id: Facility identifier.
        period: Period description (e.g., '12 months').
        score_trend: List of score data points over time.
        contamination_trend: List of contamination event counts over time.
        improvement_trajectory: Improvement analysis data.
        generated_at: ISO timestamp of report generation.
        format: Report output format.
        provenance_hash: SHA-256 hash for audit trail.
    """

    report_id: str
    facility_id: str
    period: str
    score_trend: List[Dict[str, Any]]
    contamination_trend: List[Dict[str, Any]]
    improvement_trajectory: Dict[str, Any]
    generated_at: str
    format: str
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# ComplianceReporter Engine
# ---------------------------------------------------------------------------

class ComplianceReporter:
    """Generates segregation compliance reports for EUDR regulatory needs.

    Supports five report types across four output formats:
    - Audit reports: Comprehensive facility segregation assessment
    - Contamination reports: Detailed contamination event analysis
    - Evidence packages: Competent authority inspection readiness
    - Trend reports: Score and contamination trend analysis
    - Supply chain summaries: Multi-facility compliance overview

    All report content is assembled from deterministic input data
    with arithmetic score aggregation. No ML/LLM is used for
    report generation or narrative content.

    Attributes:
        _reports: In-memory store of generated reports keyed by report_id.
        _facility_reports: Mapping of facility_id to report_id list.

    Example:
        >>> reporter = ComplianceReporter()
        >>> report = reporter.generate_audit_report(
        ...     facility_id="fac-001",
        ...     assessment_data={...},
        ...     scp_data={...},
        ...     storage_data={...},
        ...     transport_data={...},
        ...     processing_data={...},
        ...     labeling_data={...},
        ...     contamination_data={...},
        ... )
        >>> assert report.overall_score >= 0.0
    """

    def __init__(self) -> None:
        """Initialize ComplianceReporter."""
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._facility_reports: Dict[str, List[str]] = {}
        logger.info(
            "ComplianceReporter initialized: "
            "report_types=%d, formats=%d, retention=%d days, "
            "module_version=%s",
            len(REPORT_TYPES),
            len(SUPPORTED_FORMATS),
            REPORT_RETENTION_DAYS,
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Report Generation
    # ------------------------------------------------------------------

    def generate_audit_report(
        self,
        facility_id: str,
        assessment_data: Dict[str, Any],
        scp_data: Dict[str, Any],
        storage_data: Dict[str, Any],
        transport_data: Dict[str, Any],
        processing_data: Dict[str, Any],
        labeling_data: Dict[str, Any],
        contamination_data: Dict[str, Any],
        format: str = "json",
    ) -> SegregationAuditReport:
        """Generate a comprehensive segregation audit report.

        Args:
            facility_id: Facility identifier.
            assessment_data: Assessment scores and capability level.
            scp_data: SCP coverage statistics.
            storage_data: Storage zone audit findings.
            transport_data: Transport verification findings.
            processing_data: Processing line verification findings.
            labeling_data: Labeling compliance findings.
            contamination_data: Contamination event summary.
            format: Output format (json/csv/pdf/eudr_xml).

        Returns:
            SegregationAuditReport with all audit data.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, got '{format}'"
            )

        report_id = _generate_id("rpt-audit")
        now = utcnow()

        # Compute overall score from assessment data
        overall_score = float(assessment_data.get("overall_score", 0.0))

        # Build recommendations from assessment and findings
        recommendations = self._build_audit_recommendations(
            assessment_data, storage_data, transport_data,
            processing_data, labeling_data, contamination_data,
        )

        report = SegregationAuditReport(
            report_id=report_id,
            facility_id=facility_id,
            report_type="audit",
            assessment_summary=dict(assessment_data),
            scp_coverage=dict(scp_data),
            storage_audit=dict(storage_data),
            transport_audit=dict(transport_data),
            processing_audit=dict(processing_data),
            labeling_audit=dict(labeling_data),
            contamination_summary=dict(contamination_data),
            overall_score=round(overall_score, 2),
            recommendations=recommendations,
            generated_at=now.isoformat(),
            format=format,
        )
        report.provenance_hash = _compute_hash({
            "report_id": report_id,
            "facility_id": facility_id,
            "report_type": "audit",
            "overall_score": report.overall_score,
            "generated_at": report.generated_at,
            "module_version": _MODULE_VERSION,
        })

        # Store report
        self._store_report(report_id, facility_id, "audit", report)

        logger.info(
            "Audit report generated: id=%s, facility=%s, "
            "score=%.1f, format=%s",
            report_id,
            facility_id,
            overall_score,
            format,
        )
        return report

    def generate_contamination_report(
        self,
        event_id: str,
        event_data: Dict[str, Any],
        impact_data: Dict[str, Any],
        format: str = "json",
    ) -> ContaminationReport:
        """Generate a detailed contamination event report.

        Args:
            event_id: Contamination event identifier.
            event_data: Full contamination event data with keys:
                facility_id, pathway_type, severity, affected_batch_ids,
                affected_quantity_kg, root_cause, corrective_action,
                resolved, timestamp.
            impact_data: Downstream impact assessment data with keys:
                downstream_batch_ids, total_affected_quantity_kg,
                status_downgrades, propagation_depth.
            format: Output format (json/csv/pdf/eudr_xml).

        Returns:
            ContaminationReport with event analysis.
        """
        if not event_id:
            raise ValueError("event_id must not be empty")
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, got '{format}'"
            )

        report_id = _generate_id("rpt-contam")
        now = utcnow()
        facility_id = event_data.get("facility_id", "unknown")

        # Build affected batches list
        batch_ids = event_data.get("affected_batch_ids", [])
        downstream_ids = impact_data.get("downstream_batch_ids", [])
        affected_batches = [
            {"batch_id": bid, "source": "direct", "depth": 0}
            for bid in batch_ids
        ]
        for i, bid in enumerate(downstream_ids):
            affected_batches.append({
                "batch_id": bid,
                "source": "downstream",
                "depth": i + 1,
            })

        # Root cause analysis
        root_cause_analysis = {
            "root_cause": event_data.get("root_cause", "Under investigation"),
            "pathway_type": event_data.get("pathway_type", "unknown"),
            "severity": event_data.get("severity", "unknown"),
            "contributing_factors": self._identify_contributing_factors(
                event_data,
            ),
        }

        # Corrective actions
        corrective_action = event_data.get("corrective_action", "")
        corrective_actions = []
        if corrective_action:
            corrective_actions.append({
                "action": corrective_action,
                "status": "completed" if event_data.get("resolved") else "planned",
                "date": event_data.get("resolved_date", ""),
            })

        # Impact assessment
        impact_assessment = {
            "direct_batches": len(batch_ids),
            "downstream_batches": len(downstream_ids),
            "total_affected_quantity_kg": float(
                impact_data.get("total_affected_quantity_kg", 0.0)
            ),
            "propagation_depth": int(
                impact_data.get("propagation_depth", 0)
            ),
            "status_downgrades": impact_data.get("status_downgrades", []),
        }

        # Timeline
        timeline = self._build_contamination_timeline(event_data, impact_data)

        report = ContaminationReport(
            report_id=report_id,
            event_details=dict(event_data),
            affected_batches=affected_batches,
            root_cause_analysis=root_cause_analysis,
            corrective_actions=corrective_actions,
            impact_assessment=impact_assessment,
            timeline=timeline,
            generated_at=now.isoformat(),
            format=format,
        )
        report.provenance_hash = _compute_hash({
            "report_id": report_id,
            "event_id": event_id,
            "affected_batches": len(affected_batches),
            "generated_at": report.generated_at,
            "module_version": _MODULE_VERSION,
        })

        # Store report
        self._store_report(report_id, facility_id, "contamination", report)

        logger.info(
            "Contamination report generated: id=%s, event=%s, "
            "batches=%d, format=%s",
            report_id,
            event_id,
            len(affected_batches),
            format,
        )
        return report

    def generate_evidence_package(
        self,
        facility_id: str,
        regulatory_framework: str,
        assessment_history: List[Dict[str, Any]],
        format: str = "json",
    ) -> EvidencePackage:
        """Generate an evidence package for competent authority inspection.

        Assembles all relevant evidence for demonstrating EUDR
        segregation compliance to regulatory inspectors.

        Args:
            facility_id: Facility identifier.
            regulatory_framework: Target regulatory framework
                (EUDR/FSC/RSPO/ISCC).
            assessment_history: List of past assessment records.
            format: Output format (json/csv/pdf/eudr_xml).

        Returns:
            EvidencePackage with compiled evidence.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, got '{format}'"
            )

        report_id = _generate_id("rpt-evid")
        now = utcnow()

        # Build evidence items from assessment history
        evidence_items = self._compile_evidence_items(
            facility_id, regulatory_framework, assessment_history,
        )

        # Extract certifications from assessment history
        certifications = self._extract_certifications(assessment_history)

        # Determine current compliance status
        compliance_status = self._determine_compliance_status(
            assessment_history,
        )

        report = EvidencePackage(
            report_id=report_id,
            facility_id=facility_id,
            regulatory_framework=regulatory_framework,
            evidence_items=evidence_items,
            certifications=certifications,
            assessment_history=list(assessment_history),
            compliance_status=compliance_status,
            generated_at=now.isoformat(),
            format=format,
        )
        report.provenance_hash = _compute_hash({
            "report_id": report_id,
            "facility_id": facility_id,
            "regulatory_framework": regulatory_framework,
            "evidence_count": len(evidence_items),
            "compliance_status": compliance_status,
            "generated_at": report.generated_at,
            "module_version": _MODULE_VERSION,
        })

        # Store report
        self._store_report(report_id, facility_id, "evidence", report)

        logger.info(
            "Evidence package generated: id=%s, facility=%s, "
            "framework=%s, evidence_items=%d, format=%s",
            report_id,
            facility_id,
            regulatory_framework,
            len(evidence_items),
            format,
        )
        return report

    def generate_trend_report(
        self,
        facility_id: str,
        period_months: int = 12,
        format: str = "json",
    ) -> TrendReport:
        """Generate a trend analysis report for a facility.

        Args:
            facility_id: Facility identifier.
            period_months: Number of months to analyze.
            format: Output format (json/csv/pdf/eudr_xml).

        Returns:
            TrendReport with score and contamination trends.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, got '{format}'"
            )

        report_id = _generate_id("rpt-trend")
        now = utcnow()

        # Build score trend from stored reports
        score_trend = self._build_score_trend(facility_id, period_months)
        contamination_trend = self._build_contamination_trend(
            facility_id, period_months,
        )

        # Improvement trajectory
        improvement_trajectory = self._compute_improvement_trajectory(
            score_trend,
        )

        report = TrendReport(
            report_id=report_id,
            facility_id=facility_id,
            period=f"{period_months} months",
            score_trend=score_trend,
            contamination_trend=contamination_trend,
            improvement_trajectory=improvement_trajectory,
            generated_at=now.isoformat(),
            format=format,
        )
        report.provenance_hash = _compute_hash({
            "report_id": report_id,
            "facility_id": facility_id,
            "period_months": period_months,
            "data_points": len(score_trend),
            "generated_at": report.generated_at,
            "module_version": _MODULE_VERSION,
        })

        # Store report
        self._store_report(report_id, facility_id, "trend", report)

        logger.info(
            "Trend report generated: id=%s, facility=%s, "
            "period=%d months, data_points=%d, format=%s",
            report_id,
            facility_id,
            period_months,
            len(score_trend),
            format,
        )
        return report

    def generate_supply_chain_summary(
        self,
        supply_chain_id: str,
        segregation_data: List[Dict[str, Any]],
        format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a supply chain segregation summary report.

        Aggregates segregation data across multiple facilities in
        a supply chain to provide an overview of compliance posture.

        Args:
            supply_chain_id: Supply chain identifier.
            segregation_data: List of facility segregation data dicts
                with keys: facility_id, overall_score, capability_level,
                contamination_events_count, scp_coverage.
            format: Output format (json/csv/pdf/eudr_xml).

        Returns:
            Dictionary with supply chain summary data.
        """
        if not supply_chain_id:
            raise ValueError("supply_chain_id must not be empty")
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, got '{format}'"
            )

        report_id = _generate_id("rpt-sc")
        now = utcnow()

        # Aggregate scores
        scores = [
            float(d.get("overall_score", 0.0))
            for d in segregation_data
        ]
        avg_score = sum(scores) / max(len(scores), 1) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # Count capability levels
        level_distribution: Dict[str, int] = {}
        for d in segregation_data:
            level = d.get("capability_level", "level_0")
            level_distribution[level] = level_distribution.get(level, 0) + 1

        # Total contamination events
        total_contamination = sum(
            int(d.get("contamination_events_count", 0))
            for d in segregation_data
        )

        # Determine overall compliance status
        if avg_score >= 75.0 and min_score >= 60.0:
            overall_status = "compliant"
        elif avg_score >= 50.0:
            overall_status = "partially_compliant"
        else:
            overall_status = "non_compliant"

        summary = {
            "report_id": report_id,
            "supply_chain_id": supply_chain_id,
            "report_type": "supply_chain_summary",
            "facilities_count": len(segregation_data),
            "avg_score": round(avg_score, 2),
            "min_score": round(min_score, 2),
            "max_score": round(max_score, 2),
            "level_distribution": level_distribution,
            "total_contamination_events": total_contamination,
            "overall_compliance_status": overall_status,
            "facility_details": list(segregation_data),
            "generated_at": now.isoformat(),
            "format": format,
            "provenance_hash": _compute_hash({
                "report_id": report_id,
                "supply_chain_id": supply_chain_id,
                "facilities_count": len(segregation_data),
                "avg_score": round(avg_score, 2),
                "generated_at": now.isoformat(),
                "module_version": _MODULE_VERSION,
            }),
        }

        # Store report
        self._reports[report_id] = summary
        logger.info(
            "Supply chain summary generated: id=%s, sc=%s, "
            "facilities=%d, avg_score=%.1f, status=%s",
            report_id,
            supply_chain_id,
            len(segregation_data),
            avg_score,
            overall_status,
        )
        return summary

    # ------------------------------------------------------------------
    # Public API: Format Conversion
    # ------------------------------------------------------------------

    def format_report_json(self, report: Any) -> str:
        """Format a report as pretty-printed JSON string.

        Args:
            report: Report dataclass or dictionary.

        Returns:
            Formatted JSON string.
        """
        if hasattr(report, "__dataclass_fields__"):
            data = self._dataclass_to_dict(report)
        elif isinstance(report, dict):
            data = report
        else:
            data = {"data": str(report)}

        return json.dumps(data, indent=2, default=str, ensure_ascii=False)

    def format_report_csv(self, report: Any) -> str:
        """Format a report as CSV string.

        Args:
            report: Report dataclass or dictionary.

        Returns:
            CSV formatted string with headers.
        """
        if hasattr(report, "__dataclass_fields__"):
            data = self._dataclass_to_dict(report)
        elif isinstance(report, dict):
            data = report
        else:
            data = {"data": str(report)}

        report_type = data.get("report_type", "audit")
        columns = CSV_COLUMNS.get(report_type, list(data.keys()))

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        # Flatten nested data for CSV
        flat_row = self._flatten_for_csv(data, columns)
        writer.writerow(flat_row)

        return output.getvalue()

    def format_report_pdf_data(self, report: Any) -> Dict[str, Any]:
        """Format a report as structured dict for PDF generation.

        Produces a dictionary with sections, tables, and metadata
        suitable for consumption by a PDF rendering engine.

        Args:
            report: Report dataclass or dictionary.

        Returns:
            Structured dictionary with PDF layout sections.
        """
        if hasattr(report, "__dataclass_fields__"):
            data = self._dataclass_to_dict(report)
        elif isinstance(report, dict):
            data = report
        else:
            data = {"data": str(report)}

        report_type = data.get("report_type", "audit")
        now = utcnow()

        pdf_data = {
            "title": f"EUDR Segregation Verification Report - {report_type.title()}",
            "subtitle": f"Facility: {data.get('facility_id', 'N/A')}",
            "generated_at": now.isoformat(),
            "report_id": data.get("report_id", "N/A"),
            "format": "pdf",
            "sections": self._build_pdf_sections(data, report_type),
            "footer": {
                "disclaimer": (
                    "This report is generated by GreenLang EUDR Segregation "
                    "Verifier (GL-EUDR-SGV-010) using deterministic "
                    "calculations. All scores are provenance-tracked with "
                    "SHA-256 hashes."
                ),
                "retention": (
                    f"This report must be retained for {REPORT_RETENTION_DAYS} "
                    f"days per EUDR Article 31."
                ),
                "provenance_hash": data.get("provenance_hash", ""),
            },
        }

        return pdf_data

    def format_report_eudr_xml(self, report: Any) -> str:
        """Format a report as EUDR-compliant XML string.

        Produces XML output conforming to the EUDR segregation
        reporting schema for submission to the EU Information System.

        Args:
            report: Report dataclass or dictionary.

        Returns:
            EUDR-compliant XML string.
        """
        if hasattr(report, "__dataclass_fields__"):
            data = self._dataclass_to_dict(report)
        elif isinstance(report, dict):
            data = report
        else:
            data = {"data": str(report)}

        report_type = data.get("report_type", "audit")
        facility_id = data.get("facility_id", "unknown")
        report_id = data.get("report_id", "unknown")
        generated_at = data.get("generated_at", utcnow().isoformat())
        overall_score = data.get("overall_score", 0.0)
        provenance_hash = data.get("provenance_hash", "")

        xml_parts: List[str] = [
            XML_DECLARATION,
            f'<SegregationReport xmlns="{EUDR_XML_NAMESPACE}"'
            f' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            f' xsi:schemaLocation="{EUDR_XML_SCHEMA_LOCATION}">',
            f"  <ReportId>{xml_escape(str(report_id))}</ReportId>",
            f"  <ReportType>{xml_escape(str(report_type))}</ReportType>",
            f"  <FacilityId>{xml_escape(str(facility_id))}</FacilityId>",
            f"  <GeneratedAt>{xml_escape(str(generated_at))}</GeneratedAt>",
            f"  <OverallScore>{overall_score}</OverallScore>",
        ]

        # Add assessment summary if present
        assessment = data.get("assessment_summary", {})
        if assessment:
            xml_parts.append("  <AssessmentSummary>")
            capability = assessment.get("capability_level", "unknown")
            xml_parts.append(
                f"    <CapabilityLevel>{xml_escape(str(capability))}"
                f"</CapabilityLevel>"
            )
            for dim in ("layout", "protocols", "history", "labeling",
                        "documentation"):
                dim_score = assessment.get(f"{dim}_score", 0.0)
                xml_parts.append(
                    f"    <DimensionScore dimension=\"{dim}\">"
                    f"{dim_score}</DimensionScore>"
                )
            xml_parts.append("  </AssessmentSummary>")

        # Add SCP coverage if present
        scp = data.get("scp_coverage", {})
        if scp:
            xml_parts.append("  <SCPCoverage>")
            for key, val in scp.items():
                safe_key = xml_escape(str(key))
                safe_val = xml_escape(str(val))
                xml_parts.append(f"    <{safe_key}>{safe_val}</{safe_key}>")
            xml_parts.append("  </SCPCoverage>")

        # Add contamination summary if present
        contam = data.get("contamination_summary", {})
        if contam:
            xml_parts.append("  <ContaminationSummary>")
            for key, val in contam.items():
                safe_key = xml_escape(str(key))
                safe_val = xml_escape(str(val))
                xml_parts.append(f"    <{safe_key}>{safe_val}</{safe_key}>")
            xml_parts.append("  </ContaminationSummary>")

        # Add recommendations
        recommendations = data.get("recommendations", [])
        if recommendations:
            xml_parts.append("  <Recommendations>")
            for rec in recommendations:
                xml_parts.append(
                    f"    <Recommendation>{xml_escape(str(rec))}"
                    f"</Recommendation>"
                )
            xml_parts.append("  </Recommendations>")

        # Provenance
        xml_parts.append(
            f"  <ProvenanceHash>{xml_escape(str(provenance_hash))}"
            f"</ProvenanceHash>"
        )
        xml_parts.append(
            f"  <ModuleVersion>{_MODULE_VERSION}</ModuleVersion>"
        )
        xml_parts.append("</SegregationReport>")

        return "\n".join(xml_parts)

    # ------------------------------------------------------------------
    # Public API: Report Retrieval
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a generated report by its identifier.

        Args:
            report_id: Report identifier.

        Returns:
            Report data dictionary if found, None otherwise.
        """
        return self._reports.get(report_id)

    def list_reports(
        self,
        facility_id: Optional[str] = None,
        report_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List reports with optional filters.

        Args:
            facility_id: Optional filter by facility.
            report_type: Optional filter by report type.
            start_date: Optional ISO date string for range start.
            end_date: Optional ISO date string for range end.

        Returns:
            List of matching report metadata dictionaries.
        """
        if facility_id:
            report_ids = self._facility_reports.get(facility_id, [])
            reports = [
                self._reports[rid] for rid in report_ids
                if rid in self._reports
            ]
        else:
            reports = list(self._reports.values())

        if report_type:
            reports = [
                r for r in reports
                if r.get("report_type") == report_type
            ]

        if start_date:
            reports = [
                r for r in reports
                if str(r.get("generated_at", "")) >= start_date
            ]

        if end_date:
            reports = [
                r for r in reports
                if str(r.get("generated_at", "")) <= end_date
            ]

        # Sort by generation date descending
        reports.sort(
            key=lambda r: r.get("generated_at", ""),
            reverse=True,
        )

        return reports

    def batch_generate_reports(
        self,
        facility_ids: List[str],
        report_type: str,
        format: str = "json",
    ) -> List[Dict[str, Any]]:
        """Generate reports for multiple facilities in batch.

        Args:
            facility_ids: List of facility identifiers.
            report_type: Type of report to generate for each facility.
            format: Output format.

        Returns:
            List of generation result dictionaries with keys:
            facility_id, report_id, status, error.
        """
        if report_type not in REPORT_TYPES:
            raise ValueError(
                f"report_type must be one of {REPORT_TYPES}, "
                f"got '{report_type}'"
            )
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format must be one of {SUPPORTED_FORMATS}, "
                f"got '{format}'"
            )

        results: List[Dict[str, Any]] = []

        for fac_id in facility_ids:
            try:
                if report_type == "audit":
                    report = self.generate_audit_report(
                        facility_id=fac_id,
                        assessment_data={},
                        scp_data={},
                        storage_data={},
                        transport_data={},
                        processing_data={},
                        labeling_data={},
                        contamination_data={},
                        format=format,
                    )
                    results.append({
                        "facility_id": fac_id,
                        "report_id": report.report_id,
                        "status": "success",
                        "error": None,
                    })
                elif report_type == "trend":
                    report = self.generate_trend_report(
                        facility_id=fac_id,
                        period_months=12,
                        format=format,
                    )
                    results.append({
                        "facility_id": fac_id,
                        "report_id": report.report_id,
                        "status": "success",
                        "error": None,
                    })
                elif report_type == "evidence":
                    report = self.generate_evidence_package(
                        facility_id=fac_id,
                        regulatory_framework="EUDR",
                        assessment_history=[],
                        format=format,
                    )
                    results.append({
                        "facility_id": fac_id,
                        "report_id": report.report_id,
                        "status": "success",
                        "error": None,
                    })
                else:
                    results.append({
                        "facility_id": fac_id,
                        "report_id": None,
                        "status": "skipped",
                        "error": (
                            f"Batch generation not supported for "
                            f"report_type '{report_type}' without "
                            f"event-specific data"
                        ),
                    })

            except Exception as exc:
                logger.error(
                    "Batch report generation failed for facility=%s: %s",
                    fac_id,
                    str(exc),
                )
                results.append({
                    "facility_id": fac_id,
                    "report_id": None,
                    "status": "failure",
                    "error": str(exc),
                })

        success_count = sum(
            1 for r in results if r["status"] == "success"
        )
        logger.info(
            "Batch report generation: %d/%d succeeded, type=%s, format=%s",
            success_count,
            len(facility_ids),
            report_type,
            format,
        )
        return results

    # ------------------------------------------------------------------
    # Internal Helpers: Report Storage
    # ------------------------------------------------------------------

    def _store_report(
        self,
        report_id: str,
        facility_id: str,
        report_type: str,
        report: Any,
    ) -> None:
        """Store a report in the in-memory report registry.

        Args:
            report_id: Report identifier.
            facility_id: Facility identifier.
            report_type: Report type string.
            report: Report dataclass or dictionary.
        """
        if hasattr(report, "__dataclass_fields__"):
            data = self._dataclass_to_dict(report)
        elif isinstance(report, dict):
            data = dict(report)
        else:
            data = {"report_id": report_id, "data": str(report)}

        data["report_type"] = report_type
        self._reports[report_id] = data
        self._facility_reports.setdefault(facility_id, []).append(report_id)

    # ------------------------------------------------------------------
    # Internal Helpers: Data Assembly
    # ------------------------------------------------------------------

    def _build_audit_recommendations(
        self,
        assessment_data: Dict[str, Any],
        storage_data: Dict[str, Any],
        transport_data: Dict[str, Any],
        processing_data: Dict[str, Any],
        labeling_data: Dict[str, Any],
        contamination_data: Dict[str, Any],
    ) -> List[str]:
        """Build prioritized recommendations from audit findings.

        Args:
            assessment_data: Assessment scores.
            storage_data: Storage audit findings.
            transport_data: Transport verification findings.
            processing_data: Processing verification findings.
            labeling_data: Labeling compliance findings.
            contamination_data: Contamination event summary.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        overall = float(assessment_data.get("overall_score", 0.0))
        if overall < 40.0:
            recommendations.append(
                "CRITICAL: Facility segregation capability is below "
                "minimum requirements. Immediate infrastructure and "
                "protocol improvements required."
            )
        elif overall < 70.0:
            recommendations.append(
                "Facility segregation capability needs improvement. "
                "Address identified gaps in layout, protocols, and labeling."
            )

        # Dimension-specific recommendations
        layout_score = float(assessment_data.get("layout_score", 0.0))
        if layout_score < 60.0:
            recommendations.append(
                "Layout: Install or upgrade physical barriers between "
                "compliant and non-compliant material zones."
            )

        protocol_score = float(assessment_data.get("protocol_score", 0.0))
        if protocol_score < 60.0:
            recommendations.append(
                "Protocols: Develop comprehensive segregation SOPs and "
                "conduct mandatory training for all personnel."
            )

        labeling_score = float(assessment_data.get("labeling_score", 0.0))
        if labeling_score < 60.0:
            recommendations.append(
                "Labeling: Apply compliance labels to all SCPs and "
                "implement consistent color coding across zones."
            )

        # Contamination-related
        contam_count = int(contamination_data.get("total_events", 0))
        unresolved = int(contamination_data.get("unresolved_events", 0))
        if unresolved > 0:
            recommendations.append(
                f"Resolve {unresolved} unresolved contamination events "
                f"with documented corrective actions."
            )

        if not recommendations:
            recommendations.append(
                "Continue current segregation practices and maintain "
                "regular verification schedules."
            )

        return recommendations

    def _identify_contributing_factors(
        self,
        event_data: Dict[str, Any],
    ) -> List[str]:
        """Identify contributing factors for a contamination event.

        Uses deterministic rules based on pathway type and severity.

        Args:
            event_data: Event data dictionary.

        Returns:
            List of contributing factor strings.
        """
        factors: List[str] = []
        pathway = event_data.get("pathway_type", "unknown")
        severity = event_data.get("severity", "unknown")

        pathway_factors = {
            "shared_storage": [
                "Insufficient physical barriers",
                "Zone capacity constraints",
            ],
            "shared_transport": [
                "Inadequate vehicle cleaning procedures",
                "Fleet scheduling constraints",
            ],
            "shared_processing": [
                "Insufficient changeover time",
                "Inadequate flush procedures",
            ],
            "shared_equipment": [
                "Limited dedicated equipment",
                "Cleaning protocol gaps",
            ],
            "temporal_overlap": [
                "Production schedule compression",
                "Insufficient buffer time allocation",
            ],
            "adjacent_storage": [
                "Facility layout constraints",
                "Inadequate separation distance",
            ],
            "residual_material": [
                "Equipment dead zones",
                "Cleaning verification gaps",
            ],
            "handling_error": [
                "Personnel training gaps",
                "Insufficient visual identification",
            ],
            "labeling_error": [
                "Label durability issues",
                "Application procedure gaps",
            ],
            "documentation_error": [
                "Record-keeping system limitations",
                "Process documentation gaps",
            ],
        }

        factors = pathway_factors.get(pathway, [
            "Root cause investigation in progress",
        ])

        if severity == "critical":
            factors.append("Systemic control failure indicated")

        return factors

    def _build_contamination_timeline(
        self,
        event_data: Dict[str, Any],
        impact_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build chronological timeline for a contamination event.

        Args:
            event_data: Event data dictionary.
            impact_data: Impact data dictionary.

        Returns:
            List of timeline entry dictionaries.
        """
        timeline: List[Dict[str, Any]] = []

        event_ts = event_data.get("timestamp", "")
        if event_ts:
            timeline.append({
                "timestamp": event_ts,
                "event": "contamination_detected",
                "description": (
                    f"Contamination detected via {event_data.get('pathway_type', 'unknown')} "
                    f"pathway (severity: {event_data.get('severity', 'unknown')})"
                ),
            })

        batch_count = len(event_data.get("affected_batch_ids", []))
        if batch_count > 0:
            timeline.append({
                "timestamp": event_ts,
                "event": "batches_identified",
                "description": (
                    f"{batch_count} directly affected batches identified"
                ),
            })

        downstream_count = len(
            impact_data.get("downstream_batch_ids", [])
        )
        if downstream_count > 0:
            timeline.append({
                "timestamp": event_ts,
                "event": "impact_traced",
                "description": (
                    f"{downstream_count} downstream batches identified "
                    f"(propagation depth: "
                    f"{impact_data.get('propagation_depth', 0)})"
                ),
            })

        resolved_date = event_data.get("resolved_date")
        if resolved_date:
            timeline.append({
                "timestamp": resolved_date,
                "event": "contamination_resolved",
                "description": (
                    f"Event resolved: {event_data.get('corrective_action', 'N/A')}"
                ),
            })

        return timeline

    def _compile_evidence_items(
        self,
        facility_id: str,
        regulatory_framework: str,
        assessment_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compile evidence items for an evidence package.

        Args:
            facility_id: Facility identifier.
            regulatory_framework: Target regulatory framework.
            assessment_history: Historical assessment records.

        Returns:
            List of evidence item dictionaries.
        """
        items: List[Dict[str, Any]] = []

        # Assessment records as evidence
        for i, asmt in enumerate(assessment_history):
            items.append({
                "evidence_id": f"ev-asmt-{i:03d}",
                "type": "assessment_record",
                "description": (
                    f"Facility assessment dated "
                    f"{asmt.get('assessment_date', 'N/A')}"
                ),
                "score": asmt.get("overall_score", 0.0),
                "capability_level": asmt.get("capability_level", "unknown"),
                "reference": asmt.get("assessment_id", ""),
            })

        # Standard evidence items always included
        standard_items = [
            {
                "evidence_id": "ev-sop-001",
                "type": "documentation",
                "description": "Segregation Standard Operating Procedures",
                "reference": f"{regulatory_framework}-SOP",
            },
            {
                "evidence_id": "ev-train-001",
                "type": "training_record",
                "description": "Personnel segregation training records",
                "reference": f"{regulatory_framework}-TRAINING",
            },
            {
                "evidence_id": "ev-layout-001",
                "type": "facility_layout",
                "description": "Facility layout with zone markings",
                "reference": f"{regulatory_framework}-LAYOUT",
            },
            {
                "evidence_id": "ev-label-001",
                "type": "labeling_audit",
                "description": "Labeling compliance audit results",
                "reference": f"{regulatory_framework}-LABEL",
            },
            {
                "evidence_id": "ev-contam-001",
                "type": "contamination_log",
                "description": "Contamination event log and resolutions",
                "reference": f"{regulatory_framework}-CONTAM",
            },
        ]
        items.extend(standard_items)

        return items

    def _extract_certifications(
        self,
        assessment_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract certification records from assessment history.

        Args:
            assessment_history: Historical assessment records.

        Returns:
            List of certification record dictionaries.
        """
        certifications: List[Dict[str, Any]] = []

        for asmt in assessment_history:
            cert_readiness = asmt.get("certification_readiness", {})
            for standard, readiness in cert_readiness.items():
                if isinstance(readiness, dict) and readiness.get("ready"):
                    certifications.append({
                        "standard": standard,
                        "status": "ready",
                        "assessment_id": asmt.get("assessment_id", ""),
                        "date": asmt.get("assessment_date", ""),
                    })

        return certifications

    def _determine_compliance_status(
        self,
        assessment_history: List[Dict[str, Any]],
    ) -> str:
        """Determine current compliance status from assessment history.

        Args:
            assessment_history: Historical assessment records.

        Returns:
            Compliance status string.
        """
        if not assessment_history:
            return "unknown"

        # Use the most recent assessment
        sorted_history = sorted(
            assessment_history,
            key=lambda a: a.get("assessment_date", ""),
            reverse=True,
        )
        latest = sorted_history[0]
        score = float(latest.get("overall_score", 0.0))

        if score >= 75.0:
            return "compliant"
        elif score >= 50.0:
            return "partially_compliant"
        elif score > 0.0:
            return "non_compliant"
        return "not_assessed"

    def _build_score_trend(
        self,
        facility_id: str,
        period_months: int,
    ) -> List[Dict[str, Any]]:
        """Build score trend data from stored reports.

        Args:
            facility_id: Facility identifier.
            period_months: Number of months to include.

        Returns:
            List of score data points.
        """
        now = utcnow()
        cutoff = now - timedelta(days=period_months * 30)
        cutoff_str = cutoff.isoformat()

        report_ids = self._facility_reports.get(facility_id, [])
        audit_reports = [
            self._reports[rid]
            for rid in report_ids
            if rid in self._reports
            and self._reports[rid].get("report_type") == "audit"
            and str(self._reports[rid].get("generated_at", "")) >= cutoff_str
        ]

        audit_reports.sort(key=lambda r: r.get("generated_at", ""))

        trend: List[Dict[str, Any]] = []
        for rpt in audit_reports:
            trend.append({
                "date": rpt.get("generated_at", ""),
                "overall_score": float(rpt.get("overall_score", 0.0)),
                "report_id": rpt.get("report_id", ""),
            })

        return trend

    def _build_contamination_trend(
        self,
        facility_id: str,
        period_months: int,
    ) -> List[Dict[str, Any]]:
        """Build contamination event trend data.

        Args:
            facility_id: Facility identifier.
            period_months: Number of months to include.

        Returns:
            List of contamination event count data points.
        """
        now = utcnow()
        cutoff = now - timedelta(days=period_months * 30)
        cutoff_str = cutoff.isoformat()

        report_ids = self._facility_reports.get(facility_id, [])
        contam_reports = [
            self._reports[rid]
            for rid in report_ids
            if rid in self._reports
            and self._reports[rid].get("report_type") == "contamination"
            and str(self._reports[rid].get("generated_at", "")) >= cutoff_str
        ]

        # Group by month
        monthly_counts: Dict[str, int] = {}
        for rpt in contam_reports:
            gen_at = str(rpt.get("generated_at", ""))[:7]  # YYYY-MM
            monthly_counts[gen_at] = monthly_counts.get(gen_at, 0) + 1

        trend: List[Dict[str, Any]] = []
        for month in sorted(monthly_counts.keys()):
            trend.append({
                "month": month,
                "event_count": monthly_counts[month],
            })

        return trend

    def _compute_improvement_trajectory(
        self,
        score_trend: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute improvement trajectory from score trend.

        Args:
            score_trend: List of score data points with 'overall_score'.

        Returns:
            Trajectory analysis dictionary.
        """
        if not score_trend:
            return {
                "trend": "no_data",
                "data_points": 0,
                "score_change": 0.0,
                "monthly_rate": 0.0,
            }

        scores = [
            float(dp.get("overall_score", 0.0))
            for dp in score_trend
        ]

        if len(scores) < 2:
            return {
                "trend": "insufficient_data",
                "data_points": len(scores),
                "score_change": 0.0,
                "monthly_rate": 0.0,
                "current_score": scores[0] if scores else 0.0,
            }

        first_score = scores[0]
        last_score = scores[-1]
        change = last_score - first_score
        data_points = len(scores)

        # Estimate monthly rate
        monthly_rate = change / max(data_points - 1, 1)

        if change > 2.0:
            trend = "improving"
        elif change < -2.0:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": data_points,
            "first_score": round(first_score, 2),
            "last_score": round(last_score, 2),
            "score_change": round(change, 2),
            "monthly_rate": round(monthly_rate, 2),
        }

    # ------------------------------------------------------------------
    # Internal Helpers: Format Conversion
    # ------------------------------------------------------------------

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert a dataclass to a plain dictionary.

        Args:
            obj: Dataclass instance.

        Returns:
            Dictionary representation.
        """
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if hasattr(value, "__dataclass_fields__"):
                    result[field_name] = self._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[field_name] = [
                        self._dataclass_to_dict(item)
                        if hasattr(item, "__dataclass_fields__")
                        else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: self._dataclass_to_dict(v)
                        if hasattr(v, "__dataclass_fields__")
                        else v
                        for k, v in value.items()
                    }
                else:
                    result[field_name] = value
            return result
        return {"value": obj}

    def _flatten_for_csv(
        self,
        data: Dict[str, Any],
        columns: List[str],
    ) -> Dict[str, Any]:
        """Flatten nested data for CSV output.

        Args:
            data: Report data dictionary.
            columns: Target CSV column names.

        Returns:
            Flattened dictionary with only target columns.
        """
        flat: Dict[str, Any] = {}

        for col in columns:
            if col in data:
                value = data[col]
                if isinstance(value, (dict, list)):
                    flat[col] = json.dumps(value, default=str)
                else:
                    flat[col] = value
            else:
                # Try to extract from nested structures
                flat[col] = self._extract_nested_value(data, col)

        return flat

    def _extract_nested_value(
        self,
        data: Dict[str, Any],
        key: str,
    ) -> Any:
        """Extract a value from nested data structures.

        Args:
            data: Data dictionary to search.
            key: Key to find.

        Returns:
            Found value or empty string.
        """
        # Direct lookup
        if key in data:
            return data[key]

        # Search one level deep in sub-dicts
        for sub_key, sub_val in data.items():
            if isinstance(sub_val, dict) and key in sub_val:
                return sub_val[key]

        # Count-based keys
        if key.endswith("_count"):
            base = key[:-6]  # Remove _count
            if base in data and isinstance(data[base], list):
                return len(data[base])

        return ""

    def _build_pdf_sections(
        self,
        data: Dict[str, Any],
        report_type: str,
    ) -> List[Dict[str, Any]]:
        """Build PDF layout sections from report data.

        Args:
            data: Report data dictionary.
            report_type: Report type string.

        Returns:
            List of section dictionaries for PDF rendering.
        """
        sections: List[Dict[str, Any]] = []

        if report_type == "audit":
            sections.append({
                "title": "Executive Summary",
                "content_type": "text",
                "content": (
                    f"Overall segregation compliance score: "
                    f"{data.get('overall_score', 0.0)}/100"
                ),
            })
            sections.append({
                "title": "Assessment Summary",
                "content_type": "table",
                "content": data.get("assessment_summary", {}),
            })
            sections.append({
                "title": "SCP Coverage",
                "content_type": "table",
                "content": data.get("scp_coverage", {}),
            })
            sections.append({
                "title": "Storage Audit",
                "content_type": "table",
                "content": data.get("storage_audit", {}),
            })
            sections.append({
                "title": "Transport Audit",
                "content_type": "table",
                "content": data.get("transport_audit", {}),
            })
            sections.append({
                "title": "Processing Audit",
                "content_type": "table",
                "content": data.get("processing_audit", {}),
            })
            sections.append({
                "title": "Labeling Audit",
                "content_type": "table",
                "content": data.get("labeling_audit", {}),
            })
            sections.append({
                "title": "Contamination Summary",
                "content_type": "table",
                "content": data.get("contamination_summary", {}),
            })
            sections.append({
                "title": "Recommendations",
                "content_type": "list",
                "content": data.get("recommendations", []),
            })
        elif report_type == "contamination":
            sections.append({
                "title": "Event Details",
                "content_type": "table",
                "content": data.get("event_details", {}),
            })
            sections.append({
                "title": "Root Cause Analysis",
                "content_type": "table",
                "content": data.get("root_cause_analysis", {}),
            })
            sections.append({
                "title": "Impact Assessment",
                "content_type": "table",
                "content": data.get("impact_assessment", {}),
            })
            sections.append({
                "title": "Corrective Actions",
                "content_type": "list",
                "content": data.get("corrective_actions", []),
            })
            sections.append({
                "title": "Timeline",
                "content_type": "timeline",
                "content": data.get("timeline", []),
            })
        elif report_type == "evidence":
            sections.append({
                "title": "Compliance Status",
                "content_type": "text",
                "content": (
                    f"Current status: {data.get('compliance_status', 'unknown')}"
                ),
            })
            sections.append({
                "title": "Evidence Items",
                "content_type": "list",
                "content": data.get("evidence_items", []),
            })
            sections.append({
                "title": "Certifications",
                "content_type": "list",
                "content": data.get("certifications", []),
            })
        elif report_type == "trend":
            sections.append({
                "title": "Score Trend",
                "content_type": "chart_data",
                "content": data.get("score_trend", []),
            })
            sections.append({
                "title": "Contamination Trend",
                "content_type": "chart_data",
                "content": data.get("contamination_trend", []),
            })
            sections.append({
                "title": "Improvement Trajectory",
                "content_type": "table",
                "content": data.get("improvement_trajectory", {}),
            })
        else:
            sections.append({
                "title": "Report Data",
                "content_type": "table",
                "content": data,
            })

        return sections

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "REPORT_TYPES",
    "SUPPORTED_FORMATS",
    "EUDR_XML_NAMESPACE",
    "REPORT_RETENTION_DAYS",
    "CSV_COLUMNS",
    # Result types
    "SegregationAuditReport",
    "ContaminationReport",
    "EvidencePackage",
    "TrendReport",
    # Engine
    "ComplianceReporter",
]
