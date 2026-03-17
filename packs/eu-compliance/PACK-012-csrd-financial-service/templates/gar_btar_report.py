# -*- coding: utf-8 -*-
"""
GARBTARReportTemplate - EU Taxonomy Art 8 Delegated Act GAR/BTAR Disclosure

Generates Green Asset Ratio and Banking Book Taxonomy Alignment Ratio
disclosure reports per EBA ITS templates.

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



class GARBTARReportData(BaseModel):
    """Data for GAR/BTAR report."""
    institution_name: str = Field(default="")
    reporting_date: str = Field(default="")
    gar_pct: float = Field(default=0.0)
    btar_pct: float = Field(default=0.0)
    eligible_assets_eur: float = Field(default=0.0)
    aligned_assets_eur: float = Field(default=0.0)
    total_covered_assets_eur: float = Field(default=0.0)
    excluded_assets_eur: float = Field(default=0.0)
    gar_by_objective: Dict[str, float] = Field(default_factory=dict)
    transitional_pct: float = Field(default=0.0)
    enabling_pct: float = Field(default=0.0)
    counterparties_assessed: int = Field(default=0)
    flow_data: Dict[str, Any] = Field(default_factory=dict)
    qualitative_notes: str = Field(default="")


class GARBTARReportTemplate:
    """EU Taxonomy Art 8 DA - GAR/BTAR disclosure report template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "gar_btar_report"
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
        name = data.get("institution_name", "Credit Institution")
        date = data.get("reporting_date", "")
        gar = data.get("gar_pct", 0.0)
        btar = data.get("btar_pct", 0.0)
        eligible = data.get("eligible_assets_eur", 0.0)
        aligned = data.get("aligned_assets_eur", 0.0)
        covered = data.get("total_covered_assets_eur", 0.0)
        excluded = data.get("excluded_assets_eur", 0.0)

        sections.append(
            f"# EU Taxonomy Art 8 DA Disclosure - GAR/BTAR\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Date:** {date}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

        sections.append(
            f"## Key Performance Indicators\n\n"
            f"| KPI | Value |\n"
            f"|-----|-------|\n"
            f"| **Green Asset Ratio (GAR)** | **{gar:.2f}%** |\n"
            f"| **BTAR** | **{btar:.2f}%** |\n"
            f"| Eligible Assets | EUR {eligible:,.0f} |\n"
            f"| Aligned Assets | EUR {aligned:,.0f} |\n"
            f"| Covered Assets | EUR {covered:,.0f} |\n"
            f"| Excluded Assets | EUR {excluded:,.0f} |"
        )

        # GAR by objective
        by_obj = data.get("gar_by_objective", {})
        if by_obj:
            rows = ["## GAR by Environmental Objective\n",
                     "| Objective | GAR (%) |",
                     "|-----------|---------|"]
            obj_names = {
                "CLIMATE_MITIGATION": "Climate Change Mitigation",
                "CLIMATE_ADAPTATION": "Climate Change Adaptation",
                "WATER": "Water and Marine Resources",
                "CIRCULAR_ECONOMY": "Circular Economy",
                "POLLUTION": "Pollution Prevention",
                "BIODIVERSITY": "Biodiversity and Ecosystems",
            }
            for obj, pct in by_obj.items():
                display = obj_names.get(obj, obj)
                rows.append(f"| {display} | {pct:.2f}% |")
            sections.append("\n".join(rows))

        # Transitional/enabling
        trans = data.get("transitional_pct", 0.0)
        enab = data.get("enabling_pct", 0.0)
        sections.append(
            f"## Taxonomy Activity Types\n\n"
            f"| Type | Proportion |\n"
            f"|------|-----------|\n"
            f"| Transitional Activities | {trans:.2f}% |\n"
            f"| Enabling Activities | {enab:.2f}% |"
        )

        # ASCII bar
        sections.append(
            f"## GAR Visual\n\n```\n"
            f"  Aligned    [{'#' * int(gar / 2):<50s}] {gar:.2f}%\n"
            f"  Eligible   [{'#' * int(eligible / max(covered, 1) * 50):<50s}] "
            f"{eligible / max(covered, 1) * 100:.1f}%\n"
            f"```"
        )

        notes = data.get("qualitative_notes", "")
        if notes:
            sections.append(f"## Qualitative Notes\n\n{notes}")

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Credit Institution"))
        gar = data.get("gar_pct", 0.0)
        btar = data.get("btar_pct", 0.0)
        body = (
            f'<div class="section"><h2>Key Performance Indicators</h2>'
            f'<table class="data-table">'
            f'<tr><th>KPI</th><th>Value</th></tr>'
            f'<tr><td>GAR</td><td>{gar:.2f}%</td></tr>'
            f'<tr><td>BTAR</td><td>{btar:.2f}%</td></tr>'
            f'</table></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"EU Taxonomy GAR/BTAR - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "eu_taxonomy_gar_btar",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "reporting_date": data.get("reporting_date", ""),
            "kpis": {
                "gar_pct": data.get("gar_pct", 0.0),
                "btar_pct": data.get("btar_pct", 0.0),
                "eligible_assets_eur": data.get("eligible_assets_eur", 0.0),
                "aligned_assets_eur": data.get("aligned_assets_eur", 0.0),
                "covered_assets_eur": data.get("total_covered_assets_eur", 0.0),
            },
            "gar_by_objective": data.get("gar_by_objective", {}),
            "transitional_pct": data.get("transitional_pct", 0.0),
            "enabling_pct": data.get("enabling_pct", 0.0),
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
            "<style>body{font-family:Arial,sans-serif;margin:40px auto;max-width:1000px;color:#2c3e50}"
            "h1{color:#1a5276;border-bottom:3px solid #1abc9c;padding-bottom:10px}"
            "h2{color:#1a5276}.section{margin:30px 0;padding:15px;background:#fafafa;border-radius:6px}"
            ".data-table{width:100%;border-collapse:collapse}.data-table td,.data-table th"
            "{padding:8px 12px;border:1px solid #ddd}.data-table th{background:#1a5276;color:#fff}"
            ".provenance{margin-top:40px;padding:10px;background:#eaf2f8;font-family:monospace;font-size:.85em}"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n{body}\n"
            f'<div class="provenance">Provenance: {provenance_hash}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
