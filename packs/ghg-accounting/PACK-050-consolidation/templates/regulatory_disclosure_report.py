# -*- coding: utf-8 -*-
"""
RegulatoryDisclosureReport - Multi-framework disclosure template for PACK-050.

Generates disclosure content mapped to multiple regulatory frameworks including
CSRD/ESRS E1, CDP Climate Change, GRI 305, TCFD Metrics, SEC Climate Rule,
SBTi progress, and IFRS S2. Includes a cross-reference table showing which
data point maps to which framework requirement.

Sections:
    1. Disclosure Summary (frameworks covered, completeness status)
    2. CSRD/ESRS E1 Disclosures (E1-1 through E1-9)
    3. CDP Climate Change Responses (C6, C7 questions)
    4. GRI 305 Disclosures (305-1 through 305-7)
    5. TCFD Metrics and Targets
    6. SEC Climate Rule Disclosures
    7. SBTi Progress Report
    8. IFRS S2 Climate Disclosures
    9. Cross-Reference Table
    10. Provenance Footer

Output Formats: Markdown, HTML, JSON, CSV

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class DisclosureItem(BaseModel):
    """Single disclosure data point."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    item_id: str = Field("")
    framework: str = Field("")
    requirement_ref: str = Field("", description="E.g., ESRS E1-6, GRI 305-1")
    requirement_name: str = Field("")
    data_value: str = Field("")
    data_unit: str = Field("")
    data_period: str = Field("")
    status: str = Field("", description="disclosed, partially_disclosed, not_disclosed, not_applicable")
    notes: str = Field("")
    assurance_level: str = Field("", description="reasonable, limited, none")


class CrossReference(BaseModel):
    """Cross-reference mapping between frameworks."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data_point: str = Field("")
    data_value: str = Field("")
    esrs_ref: str = Field("")
    cdp_ref: str = Field("")
    gri_ref: str = Field("")
    tcfd_ref: str = Field("")
    sec_ref: str = Field("")
    sbti_ref: str = Field("")
    ifrs_s2_ref: str = Field("")


class FrameworkCompleteness(BaseModel):
    """Completeness status for a single framework."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    framework: str = Field("")
    total_requirements: int = Field(0)
    disclosed: int = Field(0)
    partially_disclosed: int = Field(0)
    not_disclosed: int = Field(0)
    not_applicable: int = Field(0)
    completeness_pct: Decimal = Field(Decimal("0"))


class RegulatoryDisclosureReportInput(BaseModel):
    """Complete input for the regulatory disclosure report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    disclosures: List[Dict[str, Any]] = Field(default_factory=list)
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    framework_completeness: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class RegulatoryDisclosureReportOutput(BaseModel):
    """Rendered regulatory disclosure report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    frameworks_covered: int = Field(0)
    total_disclosures: int = Field(0)
    overall_completeness_pct: Decimal = Field(Decimal("0"))
    disclosures_by_framework: Dict[str, List[DisclosureItem]] = Field(default_factory=dict)
    framework_completeness: List[FrameworkCompleteness] = Field(default_factory=list)
    cross_references: List[CrossReference] = Field(default_factory=list)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class RegulatoryDisclosureReport:
    """
    Multi-framework regulatory disclosure report template for PACK-050.

    Produces disclosure content mapped to CSRD/ESRS E1, CDP, GRI 305,
    TCFD, SEC Climate Rule, SBTi, and IFRS S2 with a cross-reference table.

    Example:
        >>> tpl = RegulatoryDisclosureReport()
        >>> report = tpl.render(data)
        >>> md = tpl.export_markdown(report)
    """

    # Recognised frameworks for grouping
    FRAMEWORK_ORDER = ["ESRS", "CDP", "GRI", "TCFD", "SEC", "SBTi", "IFRS_S2"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> RegulatoryDisclosureReportOutput:
        """Render regulatory disclosure report from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = RegulatoryDisclosureReportInput(**data) if isinstance(data, dict) else data

        disclosures = [DisclosureItem(**d) if isinstance(d, dict) else d for d in inp.disclosures]
        cross_refs = [CrossReference(**c) if isinstance(c, dict) else c for c in inp.cross_references]
        fw_completeness = [
            FrameworkCompleteness(**f) if isinstance(f, dict) else f
            for f in inp.framework_completeness
        ]

        # Compute completeness percentages where missing
        for fc in fw_completeness:
            if fc.completeness_pct == Decimal("0") and fc.total_requirements > 0:
                applicable = fc.total_requirements - fc.not_applicable
                if applicable > 0:
                    fc.completeness_pct = (
                        Decimal(str(fc.disclosed)) / Decimal(str(applicable)) * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Group disclosures by framework
        by_framework: Dict[str, List[DisclosureItem]] = {}
        for d in disclosures:
            by_framework.setdefault(d.framework, []).append(d)

        frameworks_covered = len(by_framework)
        total_disclosures = len(disclosures)

        # Overall completeness
        overall_pct = Decimal("0")
        if fw_completeness:
            total_req = sum(f.total_requirements - f.not_applicable for f in fw_completeness)
            total_disc = sum(f.disclosed for f in fw_completeness)
            if total_req > 0:
                overall_pct = (
                    Decimal(str(total_disc)) / Decimal(str(total_req)) * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = RegulatoryDisclosureReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            frameworks_covered=frameworks_covered,
            total_disclosures=total_disclosures,
            overall_completeness_pct=overall_pct,
            disclosures_by_framework=by_framework,
            framework_completeness=fw_completeness,
            cross_references=cross_refs,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    # ------------------------------------------------------------------
    # CONVENIENCE RENDER METHODS
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)

    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    # ------------------------------------------------------------------
    # EXPORT METHODS
    # ------------------------------------------------------------------

    def export_markdown(self, r: RegulatoryDisclosureReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Regulatory Disclosure Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Frameworks:** {r.frameworks_covered} | **Overall Completeness:** {r.overall_completeness_pct}%")
        lines.append("")

        # Framework completeness summary
        if r.framework_completeness:
            lines.append("## Framework Completeness")
            lines.append("| Framework | Total | Disclosed | Partial | Not Disclosed | N/A | Completeness |")
            lines.append("|-----------|-------|-----------|---------|---------------|-----|-------------|")
            for fc in r.framework_completeness:
                lines.append(
                    f"| {fc.framework} | {fc.total_requirements} | {fc.disclosed} | "
                    f"{fc.partially_disclosed} | {fc.not_disclosed} | "
                    f"{fc.not_applicable} | {fc.completeness_pct}% |"
                )
            lines.append("")

        # Disclosures by framework
        framework_order = self.FRAMEWORK_ORDER
        ordered_keys = sorted(
            r.disclosures_by_framework.keys(),
            key=lambda k: framework_order.index(k) if k in framework_order else 999,
        )
        for fw in ordered_keys:
            items = r.disclosures_by_framework[fw]
            lines.append(f"## {fw} Disclosures ({len(items)} items)")
            lines.append("| Ref | Requirement | Value | Unit | Status | Assurance |")
            lines.append("|-----|-------------|-------|------|--------|-----------|")
            for d in items:
                lines.append(
                    f"| {d.requirement_ref} | {d.requirement_name} | "
                    f"{d.data_value} | {d.data_unit} | {d.status} | {d.assurance_level} |"
                )
            lines.append("")

        # Cross-reference table
        if r.cross_references:
            lines.append("## Cross-Reference Table")
            lines.append("| Data Point | Value | ESRS | CDP | GRI | TCFD | SEC | SBTi | IFRS S2 |")
            lines.append("|------------|-------|------|-----|-----|------|-----|------|---------|")
            for cr in r.cross_references:
                lines.append(
                    f"| {cr.data_point} | {cr.data_value} | {cr.esrs_ref} | "
                    f"{cr.cdp_ref} | {cr.gri_ref} | {cr.tcfd_ref} | "
                    f"{cr.sec_ref} | {cr.sbti_ref} | {cr.ifrs_s2_ref} |"
                )
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: RegulatoryDisclosureReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Regulatory Disclosure - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: RegulatoryDisclosureReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: RegulatoryDisclosureReportOutput) -> str:
        """Export all disclosures as CSV."""
        lines_out = [
            "item_id,framework,requirement_ref,requirement_name,"
            "data_value,data_unit,status,assurance_level"
        ]
        for fw, items in r.disclosures_by_framework.items():
            for d in items:
                lines_out.append(
                    f"{d.item_id},{d.framework},{d.requirement_ref},"
                    f'"{d.requirement_name}",{d.data_value},{d.data_unit},'
                    f"{d.status},{d.assurance_level}"
                )
        return "\n".join(lines_out)


__all__ = [
    "RegulatoryDisclosureReport",
    "RegulatoryDisclosureReportInput",
    "RegulatoryDisclosureReportOutput",
    "DisclosureItem",
    "CrossReference",
    "FrameworkCompleteness",
]
