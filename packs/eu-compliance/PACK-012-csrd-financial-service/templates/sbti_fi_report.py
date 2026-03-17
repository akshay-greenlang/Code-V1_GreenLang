# -*- coding: utf-8 -*-
"""
SBTiFIReportTemplate - SBTi Financial Institutions Progress Report

Generates SBTi-FI progress report covering sector targets, portfolio coverage
approach, temperature rating, and NZBA alignment tracking.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _esc(value: str) -> str:
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )



class SBTiFIReportData(BaseModel):
    """Data for SBTi-FI progress report."""
    institution_name: str = Field(default="")
    reporting_period: str = Field(default="")
    commitment_framework: str = Field(default="SBTi-FI")
    net_zero_target_year: int = Field(default=2050)
    interim_target_year: int = Field(default=2030)
    baseline_emissions_tco2e: float = Field(default=0.0)
    current_emissions_tco2e: float = Field(default=0.0)
    portfolio_intensity: float = Field(default=0.0)
    sector_targets: List[Dict[str, Any]] = Field(default_factory=list)
    portfolio_coverage_pct: float = Field(default=0.0)
    temperature_rating: Optional[float] = Field(None)
    nzba_status: str = Field(default="")
    credibility_score: float = Field(default=0.0)
    gaps: List[str] = Field(default_factory=list)
    exclusion_policies: List[str] = Field(default_factory=list)
    engagement_targets: List[str] = Field(default_factory=list)


class SBTiFIReportTemplate:
    """SBTi-FI progress report template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "sbti_fi_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format '{fmt}'.")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        sections: List[str] = []
        name = data.get("institution_name", "Financial Institution")
        period = data.get("reporting_period", "")
        framework = data.get("commitment_framework", "SBTi-FI")
        nz_year = data.get("net_zero_target_year", 2050)
        interim = data.get("interim_target_year", 2030)
        baseline = data.get("baseline_emissions_tco2e", 0.0)
        current = data.get("current_emissions_tco2e", 0.0)
        intensity = data.get("portfolio_intensity", 0.0)
        coverage = data.get("portfolio_coverage_pct", 0.0)
        temp = data.get("temperature_rating")
        nzba = data.get("nzba_status", "")
        cred = data.get("credibility_score", 0.0)

        reduction = round((baseline - current) / max(baseline, 0.001) * 100, 2) if baseline > 0 else 0.0

        sections.append(
            f"# SBTi Financial Institutions Progress Report\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Framework:** {framework}\n\n"
            f"**Net Zero Target:** {nz_year} | **Interim Target:** {interim}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # KPIs
        sections.append(
            f"## Progress Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Baseline Emissions | {baseline:,.2f} tCO2e |\n"
            f"| Current Emissions | {current:,.2f} tCO2e |\n"
            f"| Reduction from Baseline | {reduction:.1f}% |\n"
            f"| Portfolio Intensity | {intensity:,.4f} tCO2e/EUR M |\n"
            f"| Portfolio Coverage | {coverage:.1f}% |\n"
            f"| Temperature Rating | {f'{temp:.1f} C' if temp is not None else 'N/A'} |\n"
            f"| NZBA Status | {nzba or 'N/A'} |\n"
            f"| **Credibility Score** | **{cred:.1f}/100** |"
        )

        # Credibility gauge
        filled = int(cred / 2)
        sections.append(
            f"## Credibility Assessment\n\n```\n"
            f"  Score: [{('#' * filled):<50s}] {cred:.1f}/100\n"
            f"```"
        )

        # Sector targets
        targets = data.get("sector_targets", [])
        if targets:
            rows = ["## Sector Targets\n",
                     "| Sector | Baseline | Current | Target 2030 | Status |",
                     "|--------|----------|---------|-------------|--------|"]
            for t in targets:
                on_track = t.get("on_track")
                status = "On Track" if on_track is True else ("Off Track" if on_track is False else "N/A")
                rows.append(
                    f"| {t.get('sector', '')} ({t.get('sector_name', '')}) | "
                    f"{t.get('baseline_intensity', 0.0):.1f} | "
                    f"{t.get('current_intensity', 0.0):.1f} | "
                    f"{t.get('target_2030', 'N/A')} | {status} |"
                )
            sections.append("\n".join(rows))

        # Temperature rating
        if temp is not None:
            temp_bar = int(min(temp * 10, 50))
            target_bar = int(1.5 * 10)
            sections.append(
                f"## Temperature Rating\n\n```\n"
                f"  Current:  [{'!' * temp_bar:<50s}] {temp:.1f} C\n"
                f"  Target:   [{'|' * target_bar:<50s}] 1.5 C\n"
                f"```"
            )

        # Gaps
        gaps = data.get("gaps", [])
        if gaps:
            sections.append("## Identified Gaps\n")
            for g in gaps:
                sections.append(f"- {g}")

        # Exclusion policies
        excl = data.get("exclusion_policies", [])
        if excl:
            sections.append("## Exclusion Policies\n")
            for e in excl:
                sections.append(f"- {e}")

        # Engagement targets
        eng = data.get("engagement_targets", [])
        if eng:
            sections.append("## Engagement Targets\n")
            for e in eng:
                sections.append(f"- {e}")

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Financial Institution"))
        cred = data.get("credibility_score", 0.0)
        body = (
            f'<div class="section"><h2>SBTi-FI Progress</h2>'
            f'<p><strong>Credibility Score:</strong> {cred:.1f}/100</p>'
            f'<p><strong>Framework:</strong> {data.get("commitment_framework", "")}</p></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"SBTi-FI Report - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "sbti_fi_progress",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "framework": data.get("commitment_framework", "SBTi-FI"),
            "net_zero_target_year": data.get("net_zero_target_year", 2050),
            "progress": {
                "baseline_emissions": data.get("baseline_emissions_tco2e", 0.0),
                "current_emissions": data.get("current_emissions_tco2e", 0.0),
                "portfolio_intensity": data.get("portfolio_intensity", 0.0),
                "portfolio_coverage_pct": data.get("portfolio_coverage_pct", 0.0),
                "temperature_rating": data.get("temperature_rating"),
                "credibility_score": data.get("credibility_score", 0.0),
            },
            "sector_targets": data.get("sector_targets", []),
            "gaps": data.get("gaps", []),
        }
        cs = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(cs)
        return report

    def _md_footer(self, provenance_hash: str) -> str:
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:40px auto;max-width:1000px;"
            "color:#2c3e50;line-height:1.6}"
            "h1{color:#1a5276;border-bottom:3px solid #1abc9c;padding-bottom:10px}"
            "h2{color:#1a5276;margin-top:30px}"
            ".section{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px;border:1px solid #ecf0f1}"
            ".data-table{width:100%;border-collapse:collapse;margin:10px 0}"
            ".data-table td,.data-table th{padding:8px 12px;border:1px solid #ddd}"
            ".data-table th{background:#1a5276;color:white;text-align:left}"
            ".data-table tr:nth-child(even){background:#f2f3f4}"
            ".provenance{margin-top:40px;padding:10px;background:#eaf2f8;border-radius:4px;"
            "font-size:.85em;font-family:monospace}"
            ".footer{margin-top:30px;font-size:.85em;color:#7f8c8d}"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
