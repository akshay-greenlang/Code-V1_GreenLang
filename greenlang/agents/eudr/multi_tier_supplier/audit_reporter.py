# -*- coding: utf-8 -*-
"""
Audit Reporter - AGENT-EUDR-008 Engine 8

Production-grade audit reporting engine for multi-tier supplier tracking
under the EU Deforestation Regulation (EUDR). Generates EUDR Article 14
audit reports, tier depth summaries, risk propagation reports, gap
analysis reports, DDS readiness assessments, and supply chain verification
certificates. Exports in JSON, CSV, PDF-ready data structures, and EUDR
DDS XML format.

Zero-Hallucination Guarantees:
    - All report generation is deterministic template-driven
    - Certificate IDs are UUID4 with SHA-256 provenance hash
    - XML output follows EUDR DDS namespace conventions
    - No ML/LLM used in any report generation logic
    - SHA-256 provenance chain hashing on all reports
    - Scoring and metrics are pure arithmetic

Performance Targets:
    - Single audit report: <2 seconds
    - Tier summary: <500ms
    - Certificate generation: <10ms
    - Batch multi-format export: <5 seconds per 100 chains

Regulatory References:
    - EUDR Article 4: Due Diligence obligations
    - EUDR Article 9: Traceability information
    - EUDR Article 10: Trader obligations
    - EUDR Article 14: Competent authority audit (5-year retention)
    - EUDR Annex II: Due Diligence Statement content

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-008 (Engine 8: Audit Reporting)
Agent ID: GL-EUDR-MST-008
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

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

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Types of reports that can be generated."""

    FULL_AUDIT = "full_audit"
    TIER_SUMMARY = "tier_summary"
    RISK_REPORT = "risk_report"
    GAP_REPORT = "gap_report"
    DDS_READINESS = "dds_readiness"
    CHAIN_CERTIFICATE = "chain_certificate"

class CertificateStatus(str, Enum):
    """Status of a supply chain verification certificate."""

    VALID = "valid"
    CONDITIONAL = "conditional"
    INVALID = "invalid"
    EXPIRED = "expired"

class DdsReadinessLevel(str, Enum):
    """DDS submission readiness level."""

    READY = "ready"
    ALMOST_READY = "almost_ready"
    NOT_READY = "not_ready"
    BLOCKED = "blocked"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR DDS XML namespace.
EUDR_DDS_NAMESPACE: str = "urn:eu:eudr:dds:supplier-chain:2024"

#: Certificate validity duration in days.
CERTIFICATE_VALIDITY_DAYS: int = 365

#: DDS readiness thresholds.
DDS_READINESS_THRESHOLDS: Dict[str, float] = {
    DdsReadinessLevel.READY.value: 90.0,
    DdsReadinessLevel.ALMOST_READY.value: 70.0,
    DdsReadinessLevel.NOT_READY.value: 50.0,
    DdsReadinessLevel.BLOCKED.value: 0.0,
}

#: Minimum compliance score for DDS inclusion.
MIN_DDS_COMPLIANCE_SCORE: float = 60.0

#: GreenLang agent identifier for provenance.
AGENT_ID: str = "GL-EUDR-MST-008"

# ---------------------------------------------------------------------------
# Input Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SupplierChainEntry:
    """A single supplier entry within a supply chain for reporting.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Legal entity name.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Tier level (1 = direct supplier).
        commodity_types: EUDR commodity types handled.
        risk_score: Composite risk score (0-100).
        risk_level: Risk level classification.
        compliance_score: Composite compliance score (0-100).
        compliance_status: Compliance status classification.
        gps_latitude: GPS latitude (WGS84).
        gps_longitude: GPS longitude (WGS84).
        certifications: List of certification records.
        dds_references: List of DDS references.
        gaps: List of detected gap descriptions.
        gap_count: Total number of gaps.
        critical_gap_count: Number of critical gaps.
        deforestation_free_status: Deforestation verification status.
        relationship_status: Relationship status (active, suspended, etc.).
        annual_volume_tonnes: Annual commodity volume.
        upstream_supplier_ids: IDs of upstream (sub-tier) suppliers.
        registration_id: Company registration number.
        last_updated: Last profile update ISO string.
    """

    supplier_id: str = ""
    legal_name: str = ""
    country_iso: str = ""
    tier: int = 1
    commodity_types: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_level: str = ""
    compliance_score: float = 0.0
    compliance_status: str = ""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    dds_references: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    gap_count: int = 0
    critical_gap_count: int = 0
    deforestation_free_status: str = ""
    relationship_status: str = "active"
    annual_volume_tonnes: Optional[float] = None
    upstream_supplier_ids: List[str] = field(default_factory=list)
    registration_id: str = ""
    last_updated: str = ""

@dataclass
class SupplierChain:
    """A complete supplier chain for reporting.

    Attributes:
        chain_id: Unique identifier for this chain.
        operator_name: Name of the EU operator (importer).
        operator_id: Operator registration ID.
        commodity: Primary commodity of this chain.
        entries: Supplier entries in the chain.
        deepest_tier: Deepest tier reached.
        total_suppliers: Total suppliers in chain.
        created_at: Chain creation timestamp.
    """

    chain_id: str = ""
    operator_name: str = ""
    operator_id: str = ""
    commodity: str = ""
    entries: List[SupplierChainEntry] = field(default_factory=list)
    deepest_tier: int = 0
    total_suppliers: int = 0
    created_at: str = ""

# ---------------------------------------------------------------------------
# Output Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TierSummaryRow:
    """Summary data for a single tier level.

    Attributes:
        tier_level: Tier depth.
        supplier_count: Number of suppliers at this tier.
        avg_risk_score: Average risk score.
        avg_compliance_score: Average compliance score.
        coverage_pct: Percentage of expected suppliers identified.
        countries: Unique countries at this tier.
        with_gps: Number with GPS coordinates.
        with_certification: Number with valid certifications.
        with_dds: Number with DDS references.
        critical_gaps: Total critical gaps at this tier.
    """

    tier_level: int = 0
    supplier_count: int = 0
    avg_risk_score: float = 0.0
    avg_compliance_score: float = 0.0
    coverage_pct: float = 0.0
    countries: List[str] = field(default_factory=list)
    with_gps: int = 0
    with_certification: int = 0
    with_dds: int = 0
    critical_gaps: int = 0

@dataclass
class AuditReport:
    """Full EUDR Article 14 audit report.

    Attributes:
        report_id: Unique UUID4 report identifier.
        report_type: Type of report generated.
        chain_id: Supply chain identifier.
        operator_name: EU operator name.
        commodity: Primary commodity.
        generated_at: UTC ISO timestamp.
        valid_until: Report validity expiry.
        total_suppliers: Total suppliers in chain.
        deepest_tier: Deepest tier reached.
        tier_summary: Per-tier summary rows.
        risk_summary: Risk aggregation data.
        compliance_summary: Compliance aggregation data.
        gap_summary: Gap analysis summary.
        dds_readiness: DDS readiness assessment.
        supplier_details: Full supplier detail list.
        processing_time_ms: Report generation time.
        provenance_hash: SHA-256 hash for audit trail.
        engine_version: Engine version.
    """

    report_id: str = ""
    report_type: str = ReportType.FULL_AUDIT.value
    chain_id: str = ""
    operator_name: str = ""
    commodity: str = ""
    generated_at: str = ""
    valid_until: str = ""
    total_suppliers: int = 0
    deepest_tier: int = 0
    tier_summary: List[TierSummaryRow] = field(default_factory=list)
    risk_summary: Dict[str, Any] = field(default_factory=dict)
    compliance_summary: Dict[str, Any] = field(default_factory=dict)
    gap_summary: Dict[str, Any] = field(default_factory=dict)
    dds_readiness: Dict[str, Any] = field(default_factory=dict)
    supplier_details: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    engine_version: str = _MODULE_VERSION

@dataclass
class ChainCertificate:
    """Supply chain verification certificate.

    Attributes:
        certificate_id: Unique UUID4 certificate identifier.
        chain_id: Supply chain identifier.
        operator_name: EU operator name.
        commodity: Primary commodity.
        status: Certificate status.
        compliance_score: Chain-level compliance score.
        risk_score: Chain-level risk score.
        tier_depth: Number of tiers covered.
        total_suppliers: Total suppliers verified.
        compliant_suppliers: Number of compliant suppliers.
        issued_at: UTC ISO issuance timestamp.
        valid_until: Certificate validity expiry.
        issuing_agent: Agent ID that issued the certificate.
        findings: Key findings list.
        conditions: Conditions (for conditional certificates).
        provenance_hash: SHA-256 hash for tamper detection.
        engine_version: Engine version.
    """

    certificate_id: str = ""
    chain_id: str = ""
    operator_name: str = ""
    commodity: str = ""
    status: str = CertificateStatus.INVALID.value
    compliance_score: float = 0.0
    risk_score: float = 0.0
    tier_depth: int = 0
    total_suppliers: int = 0
    compliant_suppliers: int = 0
    issued_at: str = ""
    valid_until: str = ""
    issuing_agent: str = AGENT_ID
    findings: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    engine_version: str = _MODULE_VERSION

@dataclass
class DdsReadinessAssessment:
    """DDS submission readiness assessment for a supply chain.

    Attributes:
        assessment_id: Unique UUID4 assessment identifier.
        chain_id: Supply chain identifier.
        readiness_level: Overall readiness level.
        readiness_score: Numeric readiness score (0-100).
        blocking_issues: Issues that block DDS submission.
        warnings: Non-blocking warnings.
        recommendations: Improvement recommendations.
        eligible_supplier_count: Suppliers eligible for DDS inclusion.
        total_supplier_count: Total suppliers in chain.
        eligible_pct: Percentage of eligible suppliers.
        assessed_at: UTC ISO timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    assessment_id: str = ""
    chain_id: str = ""
    readiness_level: str = DdsReadinessLevel.NOT_READY.value
    readiness_score: float = 0.0
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    eligible_supplier_count: int = 0
    total_supplier_count: int = 0
    eligible_pct: float = 0.0
    assessed_at: str = ""
    provenance_hash: str = ""

@dataclass
class BatchReportResult:
    """Batch report generation result.

    Attributes:
        batch_id: Unique UUID4 batch identifier.
        total_chains: Total chains processed.
        successful: Number of successfully generated reports.
        failed: Number of failed reports.
        reports: List of generated reports (format-keyed).
        processing_time_ms: Total batch duration.
        provenance_hash: SHA-256 hash of batch.
    """

    batch_id: str = ""
    total_chains: int = 0
    successful: int = 0
    failed: int = 0
    reports: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

# ===========================================================================
# AuditReporter
# ===========================================================================

class AuditReporter:
    """Production-grade audit reporting engine for EUDR multi-tier suppliers.

    Generates EUDR Article 14 audit-ready reports, tier summaries, risk
    reports, gap reports, DDS readiness assessments, and supply chain
    verification certificates. Supports export in JSON, CSV, PDF-ready
    data, and EUDR DDS XML formats.

    All report generation is deterministic and template-driven with
    zero LLM/ML involvement.

    Attributes:
        _report_count: Running count of reports generated.
        _certificate_count: Running count of certificates issued.

    Example::

        reporter = AuditReporter()
        chain = SupplierChain(
            chain_id="CHAIN-001",
            operator_name="EU Importer GmbH",
            commodity="cocoa",
            entries=[...],
        )
        report = reporter.generate_audit_report(chain)
        assert report.report_id != ""
    """

    def __init__(self) -> None:
        """Initialize AuditReporter."""
        self._report_count: int = 0
        self._certificate_count: int = 0

        logger.info(
            "AuditReporter initialized (version=%s)", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Public API: Full Audit Report
    # ------------------------------------------------------------------

    def generate_audit_report(
        self,
        supplier_chain: SupplierChain,
    ) -> AuditReport:
        """Generate full EUDR Article 14 audit report.

        Creates a comprehensive audit report covering tier summary, risk
        assessment, compliance status, gap analysis, and DDS readiness
        for an entire supplier chain.

        Args:
            supplier_chain: Complete supplier chain data.

        Returns:
            AuditReport with all sections populated.

        Raises:
            ValueError: If supplier_chain has no chain_id.
        """
        if not supplier_chain.chain_id:
            raise ValueError("supplier_chain must have a chain_id")

        t_start = time.monotonic()
        report_id = str(uuid.uuid4())
        now = utcnow()

        # Generate each report section
        tier_summary = self._build_tier_summary(supplier_chain)
        risk_summary = self._build_risk_summary(supplier_chain)
        compliance_summary = self._build_compliance_summary(supplier_chain)
        gap_summary = self._build_gap_summary(supplier_chain)
        dds_readiness = self._build_dds_readiness_summary(supplier_chain)
        supplier_details = self._build_supplier_details(supplier_chain)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Calculate deepest tier
        deepest_tier = max(
            (e.tier for e in supplier_chain.entries), default=0
        )

        provenance_data = {
            "report_id": report_id,
            "chain_id": supplier_chain.chain_id,
            "operator_name": supplier_chain.operator_name,
            "commodity": supplier_chain.commodity,
            "total_suppliers": len(supplier_chain.entries),
            "deepest_tier": deepest_tier,
            "generated_at": now.isoformat(),
            "engine_version": _MODULE_VERSION,
        }
        provenance_hash = _compute_hash(provenance_data)

        valid_until = (
            now + timedelta(days=CERTIFICATE_VALIDITY_DAYS)
        ).isoformat()

        report = AuditReport(
            report_id=report_id,
            report_type=ReportType.FULL_AUDIT.value,
            chain_id=supplier_chain.chain_id,
            operator_name=supplier_chain.operator_name,
            commodity=supplier_chain.commodity,
            generated_at=now.isoformat(),
            valid_until=valid_until,
            total_suppliers=len(supplier_chain.entries),
            deepest_tier=deepest_tier,
            tier_summary=tier_summary,
            risk_summary=risk_summary,
            compliance_summary=compliance_summary,
            gap_summary=gap_summary,
            dds_readiness=dds_readiness,
            supplier_details=supplier_details,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
            engine_version=_MODULE_VERSION,
        )

        self._report_count += 1

        logger.info(
            "Audit report generated: report=%s chain=%s operator=%s "
            "commodity=%s suppliers=%d tiers=%d time=%.2fms",
            report_id,
            supplier_chain.chain_id,
            supplier_chain.operator_name,
            supplier_chain.commodity,
            len(supplier_chain.entries),
            deepest_tier,
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: Tier Summary
    # ------------------------------------------------------------------

    def generate_tier_summary(
        self,
        chain: SupplierChain,
    ) -> List[TierSummaryRow]:
        """Generate tier depth summary with coverage metrics.

        Creates a summary row for each tier level in the supply chain
        with aggregate metrics.

        Args:
            chain: Complete supplier chain data.

        Returns:
            List of TierSummaryRow objects.
        """
        return self._build_tier_summary(chain)

    # ------------------------------------------------------------------
    # Public API: Risk Report
    # ------------------------------------------------------------------

    def generate_risk_report(
        self,
        chain: SupplierChain,
    ) -> Dict[str, Any]:
        """Generate risk propagation inheritance paths report.

        Analyzes risk scores across the supply chain and identifies
        the highest-risk paths.

        Args:
            chain: Complete supplier chain data.

        Returns:
            Dict with risk report data.
        """
        t_start = time.monotonic()
        report_id = str(uuid.uuid4())

        risk_summary = self._build_risk_summary(chain)

        # Build risk paths: for each tier, list suppliers sorted by risk
        tier_risk_paths: Dict[int, List[Dict[str, Any]]] = {}
        for entry in chain.entries:
            if entry.tier not in tier_risk_paths:
                tier_risk_paths[entry.tier] = []
            tier_risk_paths[entry.tier].append({
                "supplier_id": entry.supplier_id,
                "legal_name": entry.legal_name,
                "risk_score": entry.risk_score,
                "risk_level": entry.risk_level,
                "country_iso": entry.country_iso,
            })

        # Sort each tier by risk score descending
        for tier in tier_risk_paths:
            tier_risk_paths[tier].sort(
                key=lambda x: x["risk_score"], reverse=True
            )

        # Find highest-risk path through tiers
        highest_risk_path = self._find_highest_risk_path(
            chain.entries
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        provenance_data = {
            "report_id": report_id,
            "chain_id": chain.chain_id,
            "type": "risk_report",
        }

        report = {
            "report_id": report_id,
            "report_type": ReportType.RISK_REPORT.value,
            "chain_id": chain.chain_id,
            "risk_summary": risk_summary,
            "tier_risk_paths": {
                str(k): v for k, v in tier_risk_paths.items()
            },
            "highest_risk_path": highest_risk_path,
            "generated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": _compute_hash(provenance_data),
        }

        self._report_count += 1

        logger.info(
            "Risk report generated: report=%s chain=%s time=%.2fms",
            report_id,
            chain.chain_id,
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: Gap Report
    # ------------------------------------------------------------------

    def generate_gap_report(
        self,
        gaps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate gap analysis report.

        Summarizes detected gaps by severity, category, and supplier.

        Args:
            gaps: List of gap dictionaries with at least "severity",
                "category", "supplier_id", "description" keys.

        Returns:
            Dict with gap report data.
        """
        t_start = time.monotonic()
        report_id = str(uuid.uuid4())

        severity_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        supplier_gap_counts: Dict[str, int] = {}

        for gap in gaps:
            severity = gap.get("severity", "unknown")
            category = gap.get("category", "unknown")
            supplier_id = gap.get("supplier_id", "unknown")

            severity_counts[severity] = (
                severity_counts.get(severity, 0) + 1
            )
            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )
            supplier_gap_counts[supplier_id] = (
                supplier_gap_counts.get(supplier_id, 0) + 1
            )

        # Top suppliers by gap count
        top_gap_suppliers = sorted(
            supplier_gap_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        provenance_data = {
            "report_id": report_id,
            "total_gaps": len(gaps),
            "type": "gap_report",
        }

        report = {
            "report_id": report_id,
            "report_type": ReportType.GAP_REPORT.value,
            "total_gaps": len(gaps),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "top_gap_suppliers": [
                {"supplier_id": s, "gap_count": c}
                for s, c in top_gap_suppliers
            ],
            "gap_details": gaps,
            "generated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": _compute_hash(provenance_data),
        }

        self._report_count += 1

        logger.info(
            "Gap report generated: report=%s total_gaps=%d time=%.2fms",
            report_id,
            len(gaps),
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: DDS Readiness Assessment
    # ------------------------------------------------------------------

    def generate_dds_readiness(
        self,
        chain: SupplierChain,
    ) -> DdsReadinessAssessment:
        """Generate DDS submission readiness assessment.

        Evaluates whether the supply chain is ready for DDS submission
        based on compliance scores, gap analysis, and data completeness.

        Args:
            chain: Complete supplier chain data.

        Returns:
            DdsReadinessAssessment with readiness level and issues.
        """
        t_start = time.monotonic()
        assessment_id = str(uuid.uuid4())
        blocking_issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        eligible_count = 0
        total_count = len(chain.entries)

        for entry in chain.entries:
            is_eligible = True

            # Check for critical gaps
            if entry.critical_gap_count > 0:
                is_eligible = False
                blocking_issues.append(
                    f"Supplier {entry.supplier_id} ({entry.legal_name}): "
                    f"{entry.critical_gap_count} critical gaps block "
                    "DDS inclusion"
                )

            # Check compliance score
            if entry.compliance_score < MIN_DDS_COMPLIANCE_SCORE:
                is_eligible = False
                blocking_issues.append(
                    f"Supplier {entry.supplier_id} ({entry.legal_name}): "
                    f"compliance score {entry.compliance_score:.1f} is below "
                    f"minimum {MIN_DDS_COMPLIANCE_SCORE:.1f}"
                )

            # Check for missing GPS (EUDR Article 9 requirement)
            if entry.gps_latitude is None or entry.gps_longitude is None:
                is_eligible = False
                blocking_issues.append(
                    f"Supplier {entry.supplier_id} ({entry.legal_name}): "
                    "missing GPS coordinates (EUDR Article 9(1)(d))"
                )

            # Check compliance status
            if entry.compliance_status in ("non_compliant", "expired"):
                is_eligible = False

            # Warnings (non-blocking)
            if entry.compliance_status == "conditionally_compliant":
                warnings.append(
                    f"Supplier {entry.supplier_id}: conditionally compliant "
                    "- disclose conditions in DDS"
                )

            if entry.risk_score >= 60.0:
                warnings.append(
                    f"Supplier {entry.supplier_id}: high risk score "
                    f"({entry.risk_score:.1f}) - enhanced monitoring "
                    "recommended"
                )

            if is_eligible:
                eligible_count += 1

        # Calculate readiness score
        eligible_pct = (
            (eligible_count / total_count * 100.0)
            if total_count > 0
            else 0.0
        )

        # Classify readiness level
        if len(blocking_issues) == 0 and eligible_pct >= 90.0:
            readiness_level = DdsReadinessLevel.READY
            readiness_score = eligible_pct
        elif eligible_pct >= 70.0:
            readiness_level = DdsReadinessLevel.ALMOST_READY
            readiness_score = eligible_pct
        elif eligible_pct >= 50.0:
            readiness_level = DdsReadinessLevel.NOT_READY
            readiness_score = eligible_pct
        else:
            readiness_level = DdsReadinessLevel.BLOCKED
            readiness_score = eligible_pct

        # Generate recommendations
        if blocking_issues:
            recommendations.append(
                "Resolve all blocking issues before DDS submission"
            )
        if eligible_pct < 100.0:
            non_eligible = total_count - eligible_count
            recommendations.append(
                f"Address {non_eligible} non-eligible suppliers to "
                "achieve full chain coverage"
            )
        if warnings:
            recommendations.append(
                "Review all warnings and document mitigation measures"
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        provenance_data = {
            "assessment_id": assessment_id,
            "chain_id": chain.chain_id,
            "readiness_level": readiness_level.value,
            "readiness_score": round(readiness_score, 4),
            "eligible_count": eligible_count,
            "total_count": total_count,
        }
        provenance_hash = _compute_hash(provenance_data)

        assessment = DdsReadinessAssessment(
            assessment_id=assessment_id,
            chain_id=chain.chain_id,
            readiness_level=readiness_level.value,
            readiness_score=round(readiness_score, 2),
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
            eligible_supplier_count=eligible_count,
            total_supplier_count=total_count,
            eligible_pct=round(eligible_pct, 2),
            assessed_at=utcnow().isoformat(),
            provenance_hash=provenance_hash,
        )

        self._report_count += 1

        logger.info(
            "DDS readiness assessment: chain=%s level=%s score=%.1f "
            "eligible=%d/%d blocking=%d",
            chain.chain_id,
            readiness_level.value,
            readiness_score,
            eligible_count,
            total_count,
            len(blocking_issues),
        )

        return assessment

    # ------------------------------------------------------------------
    # Public API: Export Methods
    # ------------------------------------------------------------------

    def export_json(
        self,
        report: AuditReport,
    ) -> str:
        """Export audit report as formatted JSON string.

        Args:
            report: AuditReport to export.

        Returns:
            Formatted JSON string.
        """
        t_start = time.monotonic()

        data = {
            "report_id": report.report_id,
            "report_type": report.report_type,
            "chain_id": report.chain_id,
            "operator_name": report.operator_name,
            "commodity": report.commodity,
            "generated_at": report.generated_at,
            "valid_until": report.valid_until,
            "total_suppliers": report.total_suppliers,
            "deepest_tier": report.deepest_tier,
            "tier_summary": [
                {
                    "tier_level": ts.tier_level,
                    "supplier_count": ts.supplier_count,
                    "avg_risk_score": ts.avg_risk_score,
                    "avg_compliance_score": ts.avg_compliance_score,
                    "coverage_pct": ts.coverage_pct,
                    "countries": ts.countries,
                    "with_gps": ts.with_gps,
                    "with_certification": ts.with_certification,
                    "with_dds": ts.with_dds,
                    "critical_gaps": ts.critical_gaps,
                }
                for ts in report.tier_summary
            ],
            "risk_summary": report.risk_summary,
            "compliance_summary": report.compliance_summary,
            "gap_summary": report.gap_summary,
            "dds_readiness": report.dds_readiness,
            "supplier_details": report.supplier_details,
            "processing_time_ms": report.processing_time_ms,
            "provenance_hash": report.provenance_hash,
            "engine_version": report.engine_version,
        }

        result = json.dumps(data, indent=2, default=str, ensure_ascii=False)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        logger.debug(
            "JSON export completed: report=%s size=%d bytes time=%.2fms",
            report.report_id,
            len(result),
            elapsed_ms,
        )

        return result

    def export_csv(
        self,
        report: AuditReport,
    ) -> str:
        """Export audit report supplier details as CSV string.

        Args:
            report: AuditReport to export.

        Returns:
            CSV formatted string.
        """
        t_start = time.monotonic()
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        headers = [
            "supplier_id",
            "legal_name",
            "country_iso",
            "tier",
            "commodity_types",
            "risk_score",
            "risk_level",
            "compliance_score",
            "compliance_status",
            "gps_latitude",
            "gps_longitude",
            "certifications",
            "dds_references",
            "gap_count",
            "critical_gap_count",
            "deforestation_free_status",
            "relationship_status",
            "annual_volume_tonnes",
            "registration_id",
        ]
        writer.writerow(headers)

        # Data rows
        for detail in report.supplier_details:
            row = [
                detail.get("supplier_id", ""),
                detail.get("legal_name", ""),
                detail.get("country_iso", ""),
                detail.get("tier", ""),
                "|".join(detail.get("commodity_types", [])),
                detail.get("risk_score", ""),
                detail.get("risk_level", ""),
                detail.get("compliance_score", ""),
                detail.get("compliance_status", ""),
                detail.get("gps_latitude", ""),
                detail.get("gps_longitude", ""),
                str(len(detail.get("certifications", []))),
                str(len(detail.get("dds_references", []))),
                detail.get("gap_count", 0),
                detail.get("critical_gap_count", 0),
                detail.get("deforestation_free_status", ""),
                detail.get("relationship_status", ""),
                detail.get("annual_volume_tonnes", ""),
                detail.get("registration_id", ""),
            ]
            writer.writerow(row)

        result = output.getvalue()
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        logger.debug(
            "CSV export completed: report=%s rows=%d time=%.2fms",
            report.report_id,
            len(report.supplier_details),
            elapsed_ms,
        )

        return result

    def export_pdf_data(
        self,
        report: AuditReport,
    ) -> Dict[str, Any]:
        """Export audit report as PDF-ready data structure.

        Creates a structured dict suitable for rendering into a PDF
        template. Does not generate the PDF itself (that is handled
        by the presentation layer).

        Args:
            report: AuditReport to prepare for PDF rendering.

        Returns:
            Dict with PDF template-ready data.
        """
        t_start = time.monotonic()

        pdf_data: Dict[str, Any] = {
            "title": "EUDR Supply Chain Audit Report",
            "subtitle": (
                f"Article 14 Compliance Report - "
                f"{report.commodity.title()}"
            ),
            "metadata": {
                "report_id": report.report_id,
                "chain_id": report.chain_id,
                "operator": report.operator_name,
                "commodity": report.commodity,
                "generated": report.generated_at,
                "valid_until": report.valid_until,
                "agent_id": AGENT_ID,
                "engine_version": report.engine_version,
            },
            "executive_summary": {
                "total_suppliers": report.total_suppliers,
                "deepest_tier": report.deepest_tier,
                "overall_risk": report.risk_summary.get(
                    "average_risk_score", 0.0
                ),
                "overall_compliance": report.compliance_summary.get(
                    "average_compliance_score", 0.0
                ),
                "dds_readiness": report.dds_readiness.get(
                    "readiness_level", "unknown"
                ),
                "total_gaps": report.gap_summary.get("total_gaps", 0),
                "critical_gaps": report.gap_summary.get(
                    "critical_gaps", 0
                ),
            },
            "tier_summary_table": [
                {
                    "tier": ts.tier_level,
                    "suppliers": ts.supplier_count,
                    "avg_risk": f"{ts.avg_risk_score:.1f}",
                    "avg_compliance": f"{ts.avg_compliance_score:.1f}",
                    "gps_coverage": (
                        f"{ts.with_gps}/{ts.supplier_count}"
                    ),
                    "certified": (
                        f"{ts.with_certification}/{ts.supplier_count}"
                    ),
                    "countries": ", ".join(ts.countries[:5]),
                }
                for ts in report.tier_summary
            ],
            "risk_section": report.risk_summary,
            "compliance_section": report.compliance_summary,
            "gap_section": report.gap_summary,
            "dds_readiness_section": report.dds_readiness,
            "supplier_table": report.supplier_details,
            "provenance": {
                "hash": report.provenance_hash,
                "algorithm": "SHA-256",
                "generated_by": AGENT_ID,
            },
        }

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        logger.debug(
            "PDF data export completed: report=%s time=%.2fms",
            report.report_id,
            elapsed_ms,
        )

        return pdf_data

    def export_eudr_xml(
        self,
        report: AuditReport,
    ) -> str:
        """Export audit report in EUDR DDS XML format.

        Generates XML following the EUDR DDS namespace conventions
        for supply chain traceability data submission.

        Args:
            report: AuditReport to export.

        Returns:
            XML formatted string.
        """
        t_start = time.monotonic()

        lines: List[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            f'<SupplierChainReport xmlns="{EUDR_DDS_NAMESPACE}" '
            f'version="{_MODULE_VERSION}">'
        )

        # Report metadata
        lines.append("  <ReportMetadata>")
        lines.append(f"    <ReportId>{_xml_escape(report.report_id)}</ReportId>")
        lines.append(f"    <ChainId>{_xml_escape(report.chain_id)}</ChainId>")
        lines.append(
            f"    <OperatorName>{_xml_escape(report.operator_name)}"
            "</OperatorName>"
        )
        lines.append(f"    <Commodity>{_xml_escape(report.commodity)}</Commodity>")
        lines.append(
            f"    <GeneratedAt>{_xml_escape(report.generated_at)}"
            "</GeneratedAt>"
        )
        lines.append(
            f"    <ValidUntil>{_xml_escape(report.valid_until)}"
            "</ValidUntil>"
        )
        lines.append(
            f"    <TotalSuppliers>{report.total_suppliers}"
            "</TotalSuppliers>"
        )
        lines.append(
            f"    <DeepestTier>{report.deepest_tier}</DeepestTier>"
        )
        lines.append(
            f"    <ProvenanceHash>{_xml_escape(report.provenance_hash)}"
            "</ProvenanceHash>"
        )
        lines.append(
            f"    <EngineVersion>{_xml_escape(report.engine_version)}"
            "</EngineVersion>"
        )
        lines.append("  </ReportMetadata>")

        # Tier summary
        lines.append("  <TierSummary>")
        for ts in report.tier_summary:
            lines.append(f'    <Tier level="{ts.tier_level}">')
            lines.append(
                f"      <SupplierCount>{ts.supplier_count}"
                "</SupplierCount>"
            )
            lines.append(
                f"      <AvgRiskScore>{ts.avg_risk_score:.2f}"
                "</AvgRiskScore>"
            )
            lines.append(
                f"      <AvgComplianceScore>"
                f"{ts.avg_compliance_score:.2f}"
                "</AvgComplianceScore>"
            )
            lines.append(
                f"      <WithGPS>{ts.with_gps}</WithGPS>"
            )
            lines.append(
                f"      <WithCertification>{ts.with_certification}"
                "</WithCertification>"
            )
            lines.append(
                f"      <WithDDS>{ts.with_dds}</WithDDS>"
            )
            lines.append(
                f"      <CriticalGaps>{ts.critical_gaps}"
                "</CriticalGaps>"
            )
            lines.append("    </Tier>")
        lines.append("  </TierSummary>")

        # Supplier details
        lines.append("  <Suppliers>")
        for detail in report.supplier_details:
            sid = _xml_escape(str(detail.get("supplier_id", "")))
            lines.append(f'    <Supplier id="{sid}">')
            lines.append(
                f"      <LegalName>"
                f"{_xml_escape(str(detail.get('legal_name', '')))}"
                "</LegalName>"
            )
            lines.append(
                f"      <CountryISO>"
                f"{_xml_escape(str(detail.get('country_iso', '')))}"
                "</CountryISO>"
            )
            lines.append(
                f"      <Tier>{detail.get('tier', 0)}</Tier>"
            )

            lat = detail.get("gps_latitude")
            lon = detail.get("gps_longitude")
            if lat is not None and lon is not None:
                lines.append(
                    f"      <Geolocation>"
                    f"<Latitude>{lat}</Latitude>"
                    f"<Longitude>{lon}</Longitude>"
                    "</Geolocation>"
                )

            lines.append(
                f"      <RiskScore>"
                f"{detail.get('risk_score', 0.0):.2f}"
                "</RiskScore>"
            )
            lines.append(
                f"      <ComplianceScore>"
                f"{detail.get('compliance_score', 0.0):.2f}"
                "</ComplianceScore>"
            )
            lines.append(
                f"      <ComplianceStatus>"
                f"{_xml_escape(str(detail.get('compliance_status', '')))}"
                "</ComplianceStatus>"
            )
            lines.append(
                f"      <DeforestationFreeStatus>"
                f"{_xml_escape(str(detail.get('deforestation_free_status', '')))}"
                "</DeforestationFreeStatus>"
            )

            # Commodities
            commodities = detail.get("commodity_types", [])
            if commodities:
                lines.append("      <Commodities>")
                for comm in commodities:
                    lines.append(
                        f"        <Commodity>{_xml_escape(comm)}"
                        "</Commodity>"
                    )
                lines.append("      </Commodities>")

            lines.append("    </Supplier>")
        lines.append("  </Suppliers>")

        lines.append("</SupplierChainReport>")

        result = "\n".join(lines)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        logger.debug(
            "EUDR XML export completed: report=%s size=%d bytes "
            "time=%.2fms",
            report.report_id,
            len(result),
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Supply Chain Certificate
    # ------------------------------------------------------------------

    def generate_supplier_chain_certificate(
        self,
        chain: SupplierChain,
    ) -> ChainCertificate:
        """Generate supply chain verification certificate.

        Creates a certificate attesting to the verification status of
        a complete supply chain with validity period and provenance hash.

        Args:
            chain: Complete supplier chain data.

        Returns:
            ChainCertificate with status, findings, and provenance.
        """
        t_start = time.monotonic()
        certificate_id = str(uuid.uuid4())
        now = utcnow()
        findings: List[str] = []
        conditions: List[str] = []

        # Calculate chain-level metrics
        if not chain.entries:
            avg_compliance = 0.0
            avg_risk = 0.0
            compliant_count = 0
        else:
            compliance_scores = [
                e.compliance_score for e in chain.entries
            ]
            risk_scores = [e.risk_score for e in chain.entries]
            avg_compliance = sum(compliance_scores) / len(compliance_scores)
            avg_risk = sum(risk_scores) / len(risk_scores)
            compliant_count = sum(
                1
                for e in chain.entries
                if e.compliance_status in ("compliant", "conditionally_compliant")
            )

        deepest_tier = max(
            (e.tier for e in chain.entries), default=0
        )

        # Check for critical issues
        critical_gap_suppliers = [
            e
            for e in chain.entries
            if e.critical_gap_count > 0
        ]
        non_compliant_suppliers = [
            e
            for e in chain.entries
            if e.compliance_status in ("non_compliant", "expired")
        ]
        missing_gps_suppliers = [
            e
            for e in chain.entries
            if e.gps_latitude is None or e.gps_longitude is None
        ]

        # Determine certificate status
        if (
            not critical_gap_suppliers
            and not non_compliant_suppliers
            and not missing_gps_suppliers
            and avg_compliance >= 80.0
        ):
            status = CertificateStatus.VALID
            findings.append(
                "All suppliers meet EUDR compliance requirements"
            )
            findings.append(
                f"Supply chain coverage: {deepest_tier} tiers, "
                f"{len(chain.entries)} suppliers"
            )
        elif (
            len(critical_gap_suppliers) <= 2
            and not non_compliant_suppliers
            and avg_compliance >= 60.0
        ):
            status = CertificateStatus.CONDITIONAL
            findings.append(
                f"{len(critical_gap_suppliers)} suppliers have "
                "critical data gaps requiring remediation"
            )
            if missing_gps_suppliers:
                conditions.append(
                    f"{len(missing_gps_suppliers)} suppliers missing GPS "
                    "coordinates - must be resolved before DDS submission"
                )
            conditions.append(
                "Certificate valid subject to resolution of "
                "identified conditions within 30 days"
            )
        else:
            status = CertificateStatus.INVALID
            if critical_gap_suppliers:
                findings.append(
                    f"{len(critical_gap_suppliers)} suppliers have "
                    "critical data gaps"
                )
            if non_compliant_suppliers:
                findings.append(
                    f"{len(non_compliant_suppliers)} suppliers are "
                    "non-compliant"
                )
            if missing_gps_suppliers:
                findings.append(
                    f"{len(missing_gps_suppliers)} suppliers missing "
                    "GPS coordinates"
                )
            findings.append(
                "Supply chain does not meet EUDR compliance requirements"
            )

        valid_until = (
            now + timedelta(days=CERTIFICATE_VALIDITY_DAYS)
        ).isoformat()

        # Build provenance hash
        provenance_data = {
            "certificate_id": certificate_id,
            "chain_id": chain.chain_id,
            "operator_name": chain.operator_name,
            "commodity": chain.commodity,
            "status": status.value,
            "avg_compliance": round(avg_compliance, 4),
            "avg_risk": round(avg_risk, 4),
            "total_suppliers": len(chain.entries),
            "compliant_suppliers": compliant_count,
            "issued_at": now.isoformat(),
            "engine_version": _MODULE_VERSION,
        }
        provenance_hash = _compute_hash(provenance_data)

        certificate = ChainCertificate(
            certificate_id=certificate_id,
            chain_id=chain.chain_id,
            operator_name=chain.operator_name,
            commodity=chain.commodity,
            status=status.value,
            compliance_score=round(avg_compliance, 2),
            risk_score=round(avg_risk, 2),
            tier_depth=deepest_tier,
            total_suppliers=len(chain.entries),
            compliant_suppliers=compliant_count,
            issued_at=now.isoformat(),
            valid_until=valid_until,
            issuing_agent=AGENT_ID,
            findings=findings,
            conditions=conditions,
            provenance_hash=provenance_hash,
            engine_version=_MODULE_VERSION,
        )

        self._certificate_count += 1
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        logger.info(
            "Chain certificate issued: cert=%s chain=%s status=%s "
            "compliance=%.1f risk=%.1f suppliers=%d/%d time=%.2fms",
            certificate_id,
            chain.chain_id,
            status.value,
            avg_compliance,
            avg_risk,
            compliant_count,
            len(chain.entries),
            elapsed_ms,
        )

        return certificate

    # ------------------------------------------------------------------
    # Public API: Batch Reporting
    # ------------------------------------------------------------------

    def batch_report(
        self,
        chains: List[SupplierChain],
        formats: Optional[List[ReportFormat]] = None,
    ) -> BatchReportResult:
        """Generate batch multi-format reports for multiple chains.

        Args:
            chains: List of supplier chains to report on.
            formats: List of export formats (default: JSON only).

        Returns:
            BatchReportResult with all generated reports.
        """
        if formats is None:
            formats = [ReportFormat.JSON]

        t_start = time.monotonic()
        batch_id = str(uuid.uuid4())
        reports: List[Dict[str, Any]] = []
        failed_count = 0

        logger.info(
            "Starting batch report generation: batch=%s chains=%d "
            "formats=%s",
            batch_id,
            len(chains),
            [f.value for f in formats],
        )

        for chain in chains:
            try:
                audit_report = self.generate_audit_report(chain)
                chain_result: Dict[str, Any] = {
                    "chain_id": chain.chain_id,
                    "report_id": audit_report.report_id,
                    "exports": {},
                }

                for fmt in formats:
                    if fmt == ReportFormat.JSON:
                        chain_result["exports"]["json"] = self.export_json(
                            audit_report
                        )
                    elif fmt == ReportFormat.CSV:
                        chain_result["exports"]["csv"] = self.export_csv(
                            audit_report
                        )
                    elif fmt == ReportFormat.PDF_DATA:
                        chain_result["exports"]["pdf_data"] = (
                            self.export_pdf_data(audit_report)
                        )
                    elif fmt == ReportFormat.EUDR_XML:
                        chain_result["exports"]["eudr_xml"] = (
                            self.export_eudr_xml(audit_report)
                        )

                reports.append(chain_result)
            except Exception as exc:
                failed_count += 1
                logger.warning(
                    "Batch report failed for chain=%s: %s",
                    chain.chain_id,
                    str(exc),
                )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        successful = len(reports)

        provenance_data = {
            "batch_id": batch_id,
            "total_chains": len(chains),
            "successful": successful,
            "failed": failed_count,
            "formats": [f.value for f in formats],
        }
        provenance_hash = _compute_hash(provenance_data)

        logger.info(
            "Batch report generation completed: batch=%s total=%d "
            "success=%d failed=%d formats=%s time=%.2fms",
            batch_id,
            len(chains),
            successful,
            failed_count,
            [f.value for f in formats],
            elapsed_ms,
        )

        return BatchReportResult(
            batch_id=batch_id,
            total_chains=len(chains),
            successful=successful,
            failed=failed_count,
            reports=reports,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def report_count(self) -> int:
        """Return total number of reports generated."""
        return self._report_count

    @property
    def certificate_count(self) -> int:
        """Return total number of certificates issued."""
        return self._certificate_count

    # ------------------------------------------------------------------
    # Internal Helpers: Report Section Builders
    # ------------------------------------------------------------------

    def _build_tier_summary(
        self,
        chain: SupplierChain,
    ) -> List[TierSummaryRow]:
        """Build tier-level summary rows from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            List of TierSummaryRow, one per tier level.
        """
        tier_data: Dict[int, List[SupplierChainEntry]] = {}

        for entry in chain.entries:
            if entry.tier not in tier_data:
                tier_data[entry.tier] = []
            tier_data[entry.tier].append(entry)

        summary: List[TierSummaryRow] = []

        for tier_level in sorted(tier_data.keys()):
            entries = tier_data[tier_level]
            count = len(entries)

            risk_scores = [e.risk_score for e in entries]
            compliance_scores = [e.compliance_score for e in entries]
            avg_risk = sum(risk_scores) / count if count > 0 else 0.0
            avg_compliance = (
                sum(compliance_scores) / count if count > 0 else 0.0
            )

            countries = sorted(
                set(
                    e.country_iso
                    for e in entries
                    if e.country_iso
                )
            )
            with_gps = sum(
                1
                for e in entries
                if e.gps_latitude is not None
                and e.gps_longitude is not None
            )
            with_cert = sum(
                1 for e in entries if e.certifications
            )
            with_dds = sum(
                1 for e in entries if e.dds_references
            )
            critical = sum(e.critical_gap_count for e in entries)

            summary.append(
                TierSummaryRow(
                    tier_level=tier_level,
                    supplier_count=count,
                    avg_risk_score=round(avg_risk, 2),
                    avg_compliance_score=round(avg_compliance, 2),
                    coverage_pct=100.0,  # Known coverage is 100% by definition
                    countries=countries,
                    with_gps=with_gps,
                    with_certification=with_cert,
                    with_dds=with_dds,
                    critical_gaps=critical,
                )
            )

        return summary

    def _build_risk_summary(
        self,
        chain: SupplierChain,
    ) -> Dict[str, Any]:
        """Build risk summary from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            Dict with risk summary statistics.
        """
        if not chain.entries:
            return {
                "average_risk_score": 0.0,
                "max_risk_score": 0.0,
                "min_risk_score": 0.0,
                "risk_level_distribution": {},
                "high_risk_suppliers": [],
            }

        risk_scores = [e.risk_score for e in chain.entries]
        risk_levels: Dict[str, int] = {}
        for e in chain.entries:
            level = e.risk_level or "unknown"
            risk_levels[level] = risk_levels.get(level, 0) + 1

        high_risk = [
            {
                "supplier_id": e.supplier_id,
                "legal_name": e.legal_name,
                "risk_score": e.risk_score,
                "risk_level": e.risk_level,
                "tier": e.tier,
                "country_iso": e.country_iso,
            }
            for e in chain.entries
            if e.risk_score >= 60.0
        ]
        high_risk.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "average_risk_score": round(
                sum(risk_scores) / len(risk_scores), 2
            ),
            "max_risk_score": round(max(risk_scores), 2),
            "min_risk_score": round(min(risk_scores), 2),
            "risk_level_distribution": risk_levels,
            "high_risk_suppliers": high_risk[:10],
        }

    def _build_compliance_summary(
        self,
        chain: SupplierChain,
    ) -> Dict[str, Any]:
        """Build compliance summary from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            Dict with compliance summary statistics.
        """
        if not chain.entries:
            return {
                "average_compliance_score": 0.0,
                "status_distribution": {},
                "dds_eligible_count": 0,
                "dds_eligible_pct": 0.0,
            }

        compliance_scores = [e.compliance_score for e in chain.entries]
        status_dist: Dict[str, int] = {}
        dds_eligible = 0

        for e in chain.entries:
            status = e.compliance_status or "unknown"
            status_dist[status] = status_dist.get(status, 0) + 1
            if status in ("compliant", "conditionally_compliant"):
                dds_eligible += 1

        total = len(chain.entries)

        return {
            "average_compliance_score": round(
                sum(compliance_scores) / total, 2
            ),
            "status_distribution": status_dist,
            "dds_eligible_count": dds_eligible,
            "dds_eligible_pct": round(
                dds_eligible / total * 100.0, 2
            ) if total > 0 else 0.0,
        }

    def _build_gap_summary(
        self,
        chain: SupplierChain,
    ) -> Dict[str, Any]:
        """Build gap summary from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            Dict with gap analysis summary.
        """
        total_gaps = sum(e.gap_count for e in chain.entries)
        critical_gaps = sum(e.critical_gap_count for e in chain.entries)
        suppliers_with_gaps = sum(
            1 for e in chain.entries if e.gap_count > 0
        )
        suppliers_with_critical = sum(
            1 for e in chain.entries if e.critical_gap_count > 0
        )

        return {
            "total_gaps": total_gaps,
            "critical_gaps": critical_gaps,
            "major_gaps": total_gaps - critical_gaps,  # Approximation
            "suppliers_with_gaps": suppliers_with_gaps,
            "suppliers_with_critical_gaps": suppliers_with_critical,
            "gap_free_pct": round(
                (len(chain.entries) - suppliers_with_gaps)
                / len(chain.entries)
                * 100.0,
                2,
            )
            if chain.entries
            else 0.0,
        }

    def _build_dds_readiness_summary(
        self,
        chain: SupplierChain,
    ) -> Dict[str, Any]:
        """Build DDS readiness summary from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            Dict with DDS readiness summary.
        """
        assessment = self.generate_dds_readiness(chain)
        # Undo the report count increment from the sub-call
        self._report_count -= 1

        return {
            "readiness_level": assessment.readiness_level,
            "readiness_score": assessment.readiness_score,
            "eligible_count": assessment.eligible_supplier_count,
            "total_count": assessment.total_supplier_count,
            "eligible_pct": assessment.eligible_pct,
            "blocking_issue_count": len(assessment.blocking_issues),
            "warning_count": len(assessment.warnings),
        }

    def _build_supplier_details(
        self,
        chain: SupplierChain,
    ) -> List[Dict[str, Any]]:
        """Build supplier detail list from a supplier chain.

        Args:
            chain: Supplier chain data.

        Returns:
            List of supplier detail dicts suitable for export.
        """
        details: List[Dict[str, Any]] = []

        for entry in chain.entries:
            detail: Dict[str, Any] = {
                "supplier_id": entry.supplier_id,
                "legal_name": entry.legal_name,
                "country_iso": entry.country_iso,
                "tier": entry.tier,
                "commodity_types": entry.commodity_types,
                "risk_score": entry.risk_score,
                "risk_level": entry.risk_level,
                "compliance_score": entry.compliance_score,
                "compliance_status": entry.compliance_status,
                "gps_latitude": entry.gps_latitude,
                "gps_longitude": entry.gps_longitude,
                "certifications": entry.certifications,
                "dds_references": entry.dds_references,
                "gap_count": entry.gap_count,
                "critical_gap_count": entry.critical_gap_count,
                "deforestation_free_status": entry.deforestation_free_status,
                "relationship_status": entry.relationship_status,
                "annual_volume_tonnes": entry.annual_volume_tonnes,
                "registration_id": entry.registration_id,
                "upstream_supplier_ids": entry.upstream_supplier_ids,
                "last_updated": entry.last_updated,
            }
            details.append(detail)

        return details

    # ------------------------------------------------------------------
    # Internal Helpers: Risk Path Finding
    # ------------------------------------------------------------------

    def _find_highest_risk_path(
        self,
        entries: List[SupplierChainEntry],
    ) -> List[Dict[str, Any]]:
        """Find the highest-risk path through the supply chain tiers.

        For each tier, selects the highest-risk supplier to form a
        risk inheritance path from Tier 1 to the deepest tier.

        Args:
            entries: List of supplier chain entries.

        Returns:
            List of dicts representing the highest-risk path.
        """
        tier_map: Dict[int, List[SupplierChainEntry]] = {}
        for entry in entries:
            if entry.tier not in tier_map:
                tier_map[entry.tier] = []
            tier_map[entry.tier].append(entry)

        path: List[Dict[str, Any]] = []
        for tier in sorted(tier_map.keys()):
            tier_entries = tier_map[tier]
            highest = max(tier_entries, key=lambda e: e.risk_score)
            path.append({
                "tier": tier,
                "supplier_id": highest.supplier_id,
                "legal_name": highest.legal_name,
                "risk_score": highest.risk_score,
                "risk_level": highest.risk_level,
                "country_iso": highest.country_iso,
            })

        return path

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"AuditReporter("
            f"reports={self._report_count}, "
            f"certificates={self._certificate_count}, "
            f"version={_MODULE_VERSION!r})"
        )

# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _xml_escape(value: str) -> str:
    """Escape special XML characters in a string.

    Args:
        value: String to escape.

    Returns:
        XML-safe string.
    """
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "ReportFormat",
    "ReportType",
    "CertificateStatus",
    "DdsReadinessLevel",
    # Constants
    "EUDR_DDS_NAMESPACE",
    "CERTIFICATE_VALIDITY_DAYS",
    "DDS_READINESS_THRESHOLDS",
    "MIN_DDS_COMPLIANCE_SCORE",
    "AGENT_ID",
    # Data classes - Input
    "SupplierChainEntry",
    "SupplierChain",
    # Data classes - Output
    "TierSummaryRow",
    "AuditReport",
    "ChainCertificate",
    "DdsReadinessAssessment",
    "BatchReportResult",
    # Engine
    "AuditReporter",
]
