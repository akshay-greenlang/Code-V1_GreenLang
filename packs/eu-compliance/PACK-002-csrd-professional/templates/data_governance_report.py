# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Data Governance Report Template
===================================================

Data governance status report template covering classification
coverage, retention policy compliance, GDPR request tracking,
data quality SLA adherence, and audit findings.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ClassificationLevel(str, Enum):
    """Data classification level."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class FindingSeverity(str, Enum):
    """Audit finding severity."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class FindingCategory(str, Enum):
    """Audit finding category."""
    ACCESS_CONTROL = "ACCESS_CONTROL"
    DATA_QUALITY = "DATA_QUALITY"
    RETENTION = "RETENTION"
    CLASSIFICATION = "CLASSIFICATION"
    PRIVACY = "PRIVACY"
    SECURITY = "SECURITY"
    PROCESS = "PROCESS"


class RemediationStatus(str, Enum):
    """Remediation status for audit findings."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    REMEDIATED = "REMEDIATED"
    ACCEPTED = "ACCEPTED"
    OVERDUE = "OVERDUE"


class SLAStatus(str, Enum):
    """SLA compliance status."""
    MEETING = "MEETING"
    AT_RISK = "AT_RISK"
    BREACHED = "BREACHED"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ClassificationSummary(BaseModel):
    """Data classification summary."""
    total_datasets: int = Field(0, ge=0, description="Total datasets")
    by_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Dataset count by classification level",
    )
    unclassified_count: int = Field(0, ge=0, description="Unclassified datasets")

    @property
    def classified_pct(self) -> float:
        """Percentage of datasets classified."""
        if self.total_datasets == 0:
            return 0.0
        classified = self.total_datasets - self.unclassified_count
        return (classified / self.total_datasets) * 100.0


class RetentionCompliance(BaseModel):
    """Retention policy compliance summary."""
    total_policies: int = Field(0, ge=0, description="Total retention policies")
    compliant_count: int = Field(0, ge=0, description="Compliant policies")
    overdue_count: int = Field(0, ge=0, description="Overdue for disposal")
    approaching_expiry: int = Field(
        0, ge=0, description="Approaching retention expiry"
    )

    @property
    def compliance_pct(self) -> float:
        """Retention compliance percentage."""
        if self.total_policies == 0:
            return 0.0
        return (self.compliant_count / self.total_policies) * 100.0


class GDPRStatus(BaseModel):
    """GDPR data subject request status."""
    pending_requests: int = Field(0, ge=0, description="Pending requests")
    completed_requests: int = Field(0, ge=0, description="Completed requests")
    overdue_requests: int = Field(0, ge=0, description="Overdue requests")
    avg_response_days: float = Field(
        0.0, ge=0.0, description="Average response time in days"
    )
    total_requests_ytd: Optional[int] = Field(
        None, ge=0, description="Total requests year-to-date"
    )


class SLATarget(BaseModel):
    """Data quality SLA target."""
    sla_name: str = Field(..., description="SLA name")
    target_value: float = Field(
        ..., ge=0.0, le=100.0, description="Target value %"
    )
    current_value: float = Field(
        0.0, ge=0.0, le=100.0, description="Current value %"
    )
    status: SLAStatus = Field(SLAStatus.MEETING, description="SLA status")
    measurement_frequency: Optional[str] = Field(
        None, description="Measurement frequency"
    )


class QualitySLA(BaseModel):
    """Data quality SLA status."""
    sla_targets: List[SLATarget] = Field(
        default_factory=list, description="SLA targets"
    )
    breaches: int = Field(0, ge=0, description="Total SLA breaches")
    breach_trend: Optional[str] = Field(None, description="Breach trend (up/down/stable)")


class AuditFinding(BaseModel):
    """Individual audit finding."""
    finding_id: str = Field(..., description="Finding identifier")
    severity: FindingSeverity = Field(..., description="Finding severity")
    category: FindingCategory = Field(..., description="Finding category")
    description: str = Field(..., description="Finding description")
    remediation_status: RemediationStatus = Field(
        RemediationStatus.OPEN, description="Remediation status"
    )
    owner: Optional[str] = Field(None, description="Remediation owner")
    due_date: Optional[date] = Field(None, description="Remediation due date")
    recommendation: Optional[str] = Field(None, description="Recommended action")


class DataGovernanceReportInput(BaseModel):
    """Complete input for the data governance report."""
    organization_name: str = Field(..., description="Organization name")
    reporting_date: date = Field(
        default_factory=date.today, description="Report date"
    )
    classification_summary: ClassificationSummary = Field(
        default_factory=ClassificationSummary,
        description="Classification summary",
    )
    retention_compliance: RetentionCompliance = Field(
        default_factory=RetentionCompliance,
        description="Retention compliance",
    )
    gdpr_status: GDPRStatus = Field(
        default_factory=GDPRStatus, description="GDPR status"
    )
    quality_sla: QualitySLA = Field(
        default_factory=QualitySLA, description="Quality SLA status"
    )
    audit_findings: List[AuditFinding] = Field(
        default_factory=list, description="Audit findings"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _severity_sort(severity: FindingSeverity) -> int:
    """Sort key for severity."""
    return {
        FindingSeverity.CRITICAL: 0,
        FindingSeverity.HIGH: 1,
        FindingSeverity.MEDIUM: 2,
        FindingSeverity.LOW: 3,
        FindingSeverity.INFORMATIONAL: 4,
    }.get(severity, 99)


def _severity_badge(severity: FindingSeverity) -> str:
    """Badge for severity."""
    return f"[{severity.value}]"


def _sla_badge(status: SLAStatus) -> str:
    """Badge for SLA status."""
    return f"[{status.value}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class DataGovernanceReportTemplate:
    """Generate data governance status report.

    Sections:
        1. Data Governance Overview
        2. Classification Coverage
        3. Retention Policy Compliance
        4. GDPR Request Tracker
        5. Data Quality SLA Status
        6. Audit Findings
        7. Recommendations

    Example:
        >>> template = DataGovernanceReportTemplate()
        >>> data = DataGovernanceReportInput(organization_name="Acme")
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "data_governance_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the data governance report template."""
        self._render_timestamp: Optional[datetime] = None

    def render_markdown(self, data: DataGovernanceReportInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_classification(data),
            self._md_retention(data),
            self._md_gdpr(data),
            self._md_quality_sla(data),
            self._md_audit_findings(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: DataGovernanceReportInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_overview(data),
            self._html_classification(data),
            self._html_retention(data),
            self._html_gdpr(data),
            self._html_quality_sla(data),
            self._html_audit_findings(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, body)

    def render_json(self, data: DataGovernanceReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        cs = data.classification_summary
        rc = data.retention_compliance
        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_date": data.reporting_date.isoformat(),
            "classification_summary": {
                **cs.model_dump(mode="json"),
                "classified_pct": cs.classified_pct,
            },
            "retention_compliance": {
                **rc.model_dump(mode="json"),
                "compliance_pct": rc.compliance_pct,
            },
            "gdpr_status": data.gdpr_status.model_dump(mode="json"),
            "quality_sla": data.quality_sla.model_dump(mode="json"),
            "audit_findings": [
                f.model_dump(mode="json") for f in data.audit_findings
            ],
        }

    def _compute_provenance(self, data: DataGovernanceReportInput) -> str:
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: DataGovernanceReportInput) -> str:
        return (
            f"# Data Governance Status Report - {data.organization_name}\n"
            f"**Report Date:** {data.reporting_date.isoformat()}\n\n---"
        )

    def _md_overview(self, data: DataGovernanceReportInput) -> str:
        cs = data.classification_summary
        rc = data.retention_compliance
        open_findings = sum(
            1 for f in data.audit_findings
            if f.remediation_status in (RemediationStatus.OPEN, RemediationStatus.OVERDUE)
        )
        critical_findings = sum(
            1 for f in data.audit_findings
            if f.severity == FindingSeverity.CRITICAL
            and f.remediation_status != RemediationStatus.REMEDIATED
        )
        return (
            "## 1. Data Governance Overview\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Classification Coverage | {_fmt_pct(cs.classified_pct)} |\n"
            f"| Retention Compliance | {_fmt_pct(rc.compliance_pct)} |\n"
            f"| GDPR Overdue Requests | {data.gdpr_status.overdue_requests} |\n"
            f"| SLA Breaches | {data.quality_sla.breaches} |\n"
            f"| Open Audit Findings | {open_findings} |\n"
            f"| Critical Findings | {critical_findings} |"
        )

    def _md_classification(self, data: DataGovernanceReportInput) -> str:
        cs = data.classification_summary
        lines = [
            "## 2. Classification Coverage",
            "",
            f"**Total Datasets:** {cs.total_datasets} | "
            f"**Classified:** {_fmt_pct(cs.classified_pct)} | "
            f"**Unclassified:** {cs.unclassified_count}",
            "",
            "| Level | Count |",
            "|-------|-------|",
        ]
        for level, count in sorted(cs.by_level.items()):
            lines.append(f"| {level} | {count} |")
        if cs.unclassified_count > 0:
            lines.append(f"| UNCLASSIFIED | {cs.unclassified_count} |")
        return "\n".join(lines)

    def _md_retention(self, data: DataGovernanceReportInput) -> str:
        rc = data.retention_compliance
        return (
            "## 3. Retention Policy Compliance\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Total Policies | {rc.total_policies} |\n"
            f"| Compliant | {rc.compliant_count} |\n"
            f"| Overdue for Disposal | {rc.overdue_count} |\n"
            f"| Approaching Expiry | {rc.approaching_expiry} |\n"
            f"| **Compliance Rate** | **{_fmt_pct(rc.compliance_pct)}** |"
        )

    def _md_gdpr(self, data: DataGovernanceReportInput) -> str:
        g = data.gdpr_status
        total_ytd = str(g.total_requests_ytd) if g.total_requests_ytd is not None else "N/A"
        return (
            "## 4. GDPR Request Tracker\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Pending Requests | {g.pending_requests} |\n"
            f"| Completed Requests | {g.completed_requests} |\n"
            f"| Overdue Requests | {g.overdue_requests} |\n"
            f"| Average Response (days) | {g.avg_response_days:.1f} |\n"
            f"| Total YTD | {total_ytd} |"
        )

    def _md_quality_sla(self, data: DataGovernanceReportInput) -> str:
        qs = data.quality_sla
        if not qs.sla_targets:
            return "## 5. Data Quality SLA Status\n\nNo SLA targets defined."
        trend = qs.breach_trend or "N/A"
        lines = [
            "## 5. Data Quality SLA Status",
            "",
            f"**Total Breaches:** {qs.breaches} | **Trend:** {trend}",
            "",
            "| SLA | Target | Current | Status | Frequency |",
            "|-----|--------|---------|--------|-----------|",
        ]
        for t in qs.sla_targets:
            freq = t.measurement_frequency or "-"
            lines.append(
                f"| {t.sla_name} | {_fmt_pct(t.target_value)} "
                f"| {_fmt_pct(t.current_value)} "
                f"| {_sla_badge(t.status)} | {freq} |"
            )
        return "\n".join(lines)

    def _md_audit_findings(self, data: DataGovernanceReportInput) -> str:
        if not data.audit_findings:
            return "## 6. Audit Findings\n\nNo audit findings."
        sorted_findings = sorted(
            data.audit_findings,
            key=lambda f: _severity_sort(f.severity),
        )
        lines = [
            "## 6. Audit Findings",
            "",
            "| ID | Severity | Category | Description | Status | Owner | Due |",
            "|----|----------|----------|-------------|--------|-------|-----|",
        ]
        for f in sorted_findings:
            owner = f.owner or "TBD"
            due = f.due_date.isoformat() if f.due_date else "TBD"
            lines.append(
                f"| {f.finding_id} | {_severity_badge(f.severity)} "
                f"| {f.category.value} | {f.description} "
                f"| {f.remediation_status.value} | {owner} | {due} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: DataGovernanceReportInput) -> str:
        lines = ["## 7. Recommendations", ""]
        recs = []
        cs = data.classification_summary
        if cs.unclassified_count > 0:
            recs.append(
                f"Classify {cs.unclassified_count} unclassified dataset(s) "
                f"to achieve full classification coverage."
            )
        rc = data.retention_compliance
        if rc.overdue_count > 0:
            recs.append(
                f"Process {rc.overdue_count} overdue retention disposal(s)."
            )
        if data.gdpr_status.overdue_requests > 0:
            recs.append(
                f"Resolve {data.gdpr_status.overdue_requests} overdue GDPR "
                f"request(s) to meet statutory deadlines."
            )
        critical_open = [
            f for f in data.audit_findings
            if f.severity == FindingSeverity.CRITICAL
            and f.remediation_status not in (RemediationStatus.REMEDIATED, RemediationStatus.ACCEPTED)
        ]
        if critical_open:
            recs.append(
                f"Address {len(critical_open)} critical audit finding(s) immediately."
            )
        if data.quality_sla.breaches > 0:
            recs.append(
                "Investigate and resolve SLA breaches to maintain data quality standards."
            )
        if not recs:
            recs.append("Continue current data governance practices.")
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: DataGovernanceReportInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Data Governance Report - {org}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".severity-critical{color:#cf222e;font-weight:bold;}\n"
            ".severity-high{color:#e36209;font-weight:bold;}\n"
            ".severity-medium{color:#b08800;}\n"
            ".severity-low{color:#1a7f37;}\n"
            ".sla-meeting{color:#1a7f37;font-weight:bold;}\n"
            ".sla-at-risk{color:#b08800;font-weight:bold;}\n"
            ".sla-breached{color:#cf222e;font-weight:bold;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: DataGovernanceReportInput) -> str:
        return (
            '<div class="section">\n'
            f"<h1>Data Governance Status Report &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Report Date:</strong> {data.reporting_date.isoformat()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overview(self, data: DataGovernanceReportInput) -> str:
        cs = data.classification_summary
        rc = data.retention_compliance
        open_f = sum(
            1 for f in data.audit_findings
            if f.remediation_status in (RemediationStatus.OPEN, RemediationStatus.OVERDUE)
        )
        cards = [
            (_fmt_pct(cs.classified_pct), "Classification"),
            (_fmt_pct(rc.compliance_pct), "Retention"),
            (str(data.gdpr_status.overdue_requests), "GDPR Overdue"),
            (str(data.quality_sla.breaches), "SLA Breaches"),
            (str(open_f), "Open Findings"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>1. Overview</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_classification(self, data: DataGovernanceReportInput) -> str:
        cs = data.classification_summary
        rows = []
        for level, count in sorted(cs.by_level.items()):
            rows.append(f"<tr><td>{level}</td><td>{count}</td></tr>")
        if cs.unclassified_count > 0:
            rows.append(
                f'<tr><td style="color:#cf222e">UNCLASSIFIED</td>'
                f"<td>{cs.unclassified_count}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. Classification Coverage</h2>\n'
            f"<p><strong>Total:</strong> {cs.total_datasets} | "
            f"<strong>Classified:</strong> {_fmt_pct(cs.classified_pct)}</p>\n"
            "<table><thead><tr><th>Level</th><th>Count</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_retention(self, data: DataGovernanceReportInput) -> str:
        rc = data.retention_compliance
        rows = [
            f"<tr><td>Total Policies</td><td>{rc.total_policies}</td></tr>",
            f"<tr><td>Compliant</td><td>{rc.compliant_count}</td></tr>",
            f"<tr><td>Overdue</td><td>{rc.overdue_count}</td></tr>",
            f"<tr><td>Approaching Expiry</td><td>{rc.approaching_expiry}</td></tr>",
            f"<tr style='font-weight:bold'><td>Compliance Rate</td>"
            f"<td>{_fmt_pct(rc.compliance_pct)}</td></tr>",
        ]
        return (
            '<div class="section">\n<h2>3. Retention Compliance</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_gdpr(self, data: DataGovernanceReportInput) -> str:
        g = data.gdpr_status
        cards = [
            (str(g.pending_requests), "Pending"),
            (str(g.completed_requests), "Completed"),
            (str(g.overdue_requests), "Overdue"),
            (f"{g.avg_response_days:.1f}d", "Avg Response"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>4. GDPR Request Tracker</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_quality_sla(self, data: DataGovernanceReportInput) -> str:
        qs = data.quality_sla
        if not qs.sla_targets:
            return (
                '<div class="section"><h2>5. Quality SLA</h2>'
                "<p>No SLAs defined.</p></div>"
            )
        rows = []
        for t in qs.sla_targets:
            css = f"sla-{t.status.value.lower().replace('_', '-')}"
            freq = t.measurement_frequency or "-"
            rows.append(
                f"<tr><td>{t.sla_name}</td><td>{_fmt_pct(t.target_value)}</td>"
                f"<td>{_fmt_pct(t.current_value)}</td>"
                f'<td class="{css}">{t.status.value}</td><td>{freq}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>5. Data Quality SLA Status</h2>\n'
            f"<p><strong>Breaches:</strong> {qs.breaches}</p>\n"
            "<table><thead><tr><th>SLA</th><th>Target</th><th>Current</th>"
            f"<th>Status</th><th>Frequency</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_audit_findings(self, data: DataGovernanceReportInput) -> str:
        if not data.audit_findings:
            return (
                '<div class="section"><h2>6. Audit Findings</h2>'
                "<p>No findings.</p></div>"
            )
        sorted_findings = sorted(
            data.audit_findings, key=lambda f: _severity_sort(f.severity)
        )
        rows = []
        for f in sorted_findings:
            css = f"severity-{f.severity.value.lower()}"
            owner = f.owner or "TBD"
            due = f.due_date.isoformat() if f.due_date else "TBD"
            rows.append(
                f"<tr><td>{f.finding_id}</td>"
                f'<td class="{css}">{f.severity.value}</td>'
                f"<td>{f.category.value}</td><td>{f.description}</td>"
                f"<td>{f.remediation_status.value}</td>"
                f"<td>{owner}</td><td>{due}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>6. Audit Findings</h2>\n'
            "<table><thead><tr><th>ID</th><th>Severity</th><th>Category</th>"
            "<th>Description</th><th>Status</th><th>Owner</th>"
            f"<th>Due</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: DataGovernanceReportInput) -> str:
        recs = []
        if data.classification_summary.unclassified_count > 0:
            recs.append(
                f"Classify {data.classification_summary.unclassified_count} "
                f"unclassified dataset(s)."
            )
        if data.retention_compliance.overdue_count > 0:
            recs.append(
                f"Process {data.retention_compliance.overdue_count} overdue disposal(s)."
            )
        if data.gdpr_status.overdue_requests > 0:
            recs.append(
                f"Resolve {data.gdpr_status.overdue_requests} overdue GDPR request(s)."
            )
        if not recs:
            recs.append("Continue current governance practices.")
        items = "".join(f"<li>{r}</li>" for r in recs)
        return (
            '<div class="section">\n<h2>7. Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: DataGovernanceReportInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
