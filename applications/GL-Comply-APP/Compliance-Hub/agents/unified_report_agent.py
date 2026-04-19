# -*- coding: utf-8 -*-
"""Unified report agent — merges per-framework outputs into one artifact.

Output formats:
- JSON: always supported, canonical content
- PDF: optional, requires reportlab; graceful text fallback if unavailable
- XBRL-lite: XML dump of quantitative metrics; full iXBRL authoring lives in
  GL-CSRD-APP reporting pipeline (out of scope for Comply-Hub v1)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from schemas.models import (
    FrameworkEnum,
    FrameworkResult,
    UnifiedComplianceReport,
)
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)


class UnifiedReportAgent:
    """Produces unified compliance-report artifacts.

    ``generate(report, format, output_dir)`` returns a dict with:
    - ``format``: requested format
    - ``path``: absolute path to the produced artifact
    - ``content_hash``: SHA-256 of the artifact bytes
    """

    def generate(
        self,
        report: UnifiedComplianceReport,
        output_format: ReportFormat = ReportFormat.JSON,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir or Path.cwd() / "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"comply-{report.job_id}"
        if output_format == ReportFormat.JSON:
            return self._generate_json(report, output_dir, stem)
        if output_format == ReportFormat.PDF:
            return self._generate_pdf(report, output_dir, stem)
        if output_format in (ReportFormat.XML,):
            return self._generate_xbrl_lite(report, output_dir, stem)
        raise ValueError(f"Unsupported format: {output_format}")

    # ---- generators ----

    def _generate_json(
        self, report: UnifiedComplianceReport, output_dir: Path, stem: str
    ) -> dict[str, Any]:
        path = output_dir / f"{stem}.json"
        payload = report.model_dump(mode="json")
        data = json.dumps(payload, indent=2, sort_keys=True)
        path.write_text(data, encoding="utf-8")
        return self._result(ReportFormat.JSON, path, data.encode("utf-8"))

    def _generate_pdf(
        self, report: UnifiedComplianceReport, output_dir: Path, stem: str
    ) -> dict[str, Any]:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

            path = output_dir / f"{stem}.pdf"
            styles = getSampleStyleSheet()
            doc = SimpleDocTemplate(str(path), pagesize=A4, title="Unified Compliance Report")
            story: list[Any] = [
                Paragraph("Unified Compliance Report", styles["Title"]),
                Paragraph(f"Job ID: {report.job_id}", styles["Normal"]),
                Paragraph(f"Entity: {report.entity_id}", styles["Normal"]),
                Paragraph(
                    f"Period: {report.reporting_period_start.date()} — "
                    f"{report.reporting_period_end.date()}",
                    styles["Normal"],
                ),
                Paragraph(
                    f"Overall status: {report.overall_status.value}", styles["Heading2"]
                ),
                Spacer(1, 12),
            ]
            table_rows = [
                ["Framework", "Status", "Duration (ms)", "Hash"],
                *[
                    [
                        fw.value,
                        r.compliance_status.value,
                        str(r.duration_ms or ""),
                        (r.provenance_hash or "")[:16],
                    ]
                    for fw, r in report.results.items()
                ],
            ]
            story.append(Table(table_rows))
            if report.gap_analysis:
                story.append(Spacer(1, 12))
                story.append(Paragraph("Gap analysis", styles["Heading2"]))
                for gap in report.gap_analysis:
                    story.append(
                        Paragraph(
                            f"[{gap.get('severity','info')}] "
                            f"{gap.get('framework','')}: {gap.get('reason','')}",
                            styles["Normal"],
                        )
                    )
            doc.build(story)
            return self._result(ReportFormat.PDF, path, path.read_bytes())
        except ImportError:
            logger.warning("reportlab not installed; falling back to text PDF")
            return self._generate_pdf_text_fallback(report, output_dir, stem)

    @staticmethod
    def _generate_pdf_text_fallback(
        report: UnifiedComplianceReport, output_dir: Path, stem: str
    ) -> dict[str, Any]:
        path = output_dir / f"{stem}.pdf.txt"
        lines = [
            "Unified Compliance Report",
            f"Job ID: {report.job_id}",
            f"Entity: {report.entity_id}",
            f"Overall status: {report.overall_status.value}",
            "Framework results:",
        ]
        for fw, r in report.results.items():
            lines.append(f"  - {fw.value}: {r.compliance_status.value}")
        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")
        return UnifiedReportAgent._result(ReportFormat.PDF, path, text.encode("utf-8"))

    def _generate_xbrl_lite(
        self, report: UnifiedComplianceReport, output_dir: Path, stem: str
    ) -> dict[str, Any]:
        path = output_dir / f"{stem}.xml"
        lines = [
            "<?xml version='1.0' encoding='UTF-8'?>",
            "<ComplianceReport>",
            f"  <JobId>{report.job_id}</JobId>",
            f"  <EntityId>{report.entity_id}</EntityId>",
            f"  <Period start='{report.reporting_period_start.isoformat()}'"
            f" end='{report.reporting_period_end.isoformat()}'/>",
            f"  <OverallStatus>{report.overall_status.value}</OverallStatus>",
            "  <Frameworks>",
        ]
        for fw, r in report.results.items():
            lines.append(f"    <Framework name='{fw.value}'>")
            lines.append(f"      <Status>{r.compliance_status.value}</Status>")
            lines.append(
                f"      <ProvenanceHash>{r.provenance_hash or ''}</ProvenanceHash>"
            )
            for k, v in (r.metrics or {}).items():
                lines.append(f"      <Metric name='{k}'>{v}</Metric>")
            lines.append("    </Framework>")
        lines.append("  </Frameworks>")
        lines.append(
            f"  <AggregateProvenanceHash>"
            f"{report.aggregate_provenance_hash}</AggregateProvenanceHash>"
        )
        lines.append("</ComplianceReport>")
        data = "\n".join(lines)
        path.write_text(data, encoding="utf-8")
        return self._result(ReportFormat.XML, path, data.encode("utf-8"))

    @staticmethod
    def _result(fmt: ReportFormat, path: Path, content: bytes) -> dict[str, Any]:
        return {
            "format": fmt.value,
            "path": str(path.resolve()),
            "content_hash": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }


__all__ = ["UnifiedReportAgent"]
