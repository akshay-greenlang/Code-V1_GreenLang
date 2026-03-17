# -*- coding: utf-8 -*-
"""
Pillar3ESGTemplate - EBA Pillar 3 ESG ITS Disclosure

Generates EBA Pillar 3 ESG ITS quantitative and qualitative templates
for credit institutions under CRR3/CRD6.

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



class Pillar3ESGData(BaseModel):
    """Data for Pillar 3 ESG template."""
    institution_name: str = Field(default="")
    lei: str = Field(default="")
    reporting_date: str = Field(default="")
    templates_populated: int = Field(default=0)
    templates_total: int = Field(default=10)
    completeness_pct: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0)
    filing_ready: bool = Field(default=False)
    gar_pct: Optional[float] = Field(None)
    btar_pct: Optional[float] = Field(None)
    financed_emissions_tco2e: Optional[float] = Field(None)
    total_exposure_eur: Optional[float] = Field(None)
    top_20_carbon_intensive: List[Dict[str, Any]] = Field(default_factory=list)
    template_data: Dict[str, Any] = Field(default_factory=dict)
    qualitative_disclosures: Dict[str, str] = Field(default_factory=dict)
    validation_issues: List[Dict[str, Any]] = Field(default_factory=list)


class Pillar3ESGTemplate:
    """EBA Pillar 3 ESG ITS disclosure template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "pillar3_esg_template"
    VERSION = "1.0"

    EBA_TEMPLATE_NAMES = [
        "Template 1: Banking book - Climate change transition risk",
        "Template 2: Banking book - Climate change physical risk",
        "Template 3: Real estate - Energy efficiency (EPC)",
        "Template 4: Alignment metrics (GAR/BTAR)",
        "Template 5: Exposures to top 20 carbon-intensive firms",
        "Template 6: Trading book climate risk",
        "Template 7: ESG risks in the banking book",
        "Template 8: Qualitative ESG disclosures",
        "Template 9: Mitigating actions",
        "Template 10: Key performance indicators",
    ]

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
        lei = data.get("lei", "")
        date = data.get("reporting_date", "")
        populated = data.get("templates_populated", 0)
        total = data.get("templates_total", 10)
        dq = data.get("data_quality_score", 0.0)
        ready = data.get("filing_ready", False)

        sections.append(
            f"# EBA Pillar 3 ESG Disclosure\n\n"
            f"**Institution:** {name}\n\n"
            f"**LEI:** {lei}\n\n"
            f"**Reporting Date:** {date}\n\n"
            f"**Filing Status:** {'READY' if ready else 'NOT READY'}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # Template coverage
        sections.append(
            f"## Template Coverage\n\n"
            f"**Populated:** {populated}/{total} templates "
            f"({data.get('completeness_pct', 0.0):.1f}%)\n\n"
            f"**Data Quality Score:** {dq:.1f}/100\n\n"
            f"| # | Template | Status |\n"
            f"|---|----------|--------|"
        )
        for i, tname in enumerate(self.EBA_TEMPLATE_NAMES, 1):
            status_icon = "Populated" if i <= populated else "Pending"
            sections[-1] += f"\n| {i} | {tname} | {status_icon} |"

        # Key metrics
        gar = data.get("gar_pct")
        fe = data.get("financed_emissions_tco2e")
        exp = data.get("total_exposure_eur")
        if gar is not None or fe is not None:
            rows = ["## Key Alignment Metrics\n",
                     "| Metric | Value |",
                     "|--------|-------|"]
            if gar is not None:
                rows.append(f"| GAR | {gar:.2f}% |")
            btar = data.get("btar_pct")
            if btar is not None:
                rows.append(f"| BTAR | {btar:.2f}% |")
            if fe is not None:
                rows.append(f"| Financed Emissions | {fe:,.2f} tCO2e |")
            if exp is not None:
                rows.append(f"| Total Exposure | EUR {exp:,.0f} |")
            sections.append("\n".join(rows))

        # Top 20
        top20 = data.get("top_20_carbon_intensive", [])
        if top20:
            rows = ["## Top 20 Carbon-Intensive Exposures\n",
                     "| # | Counterparty | Sector | Exposure (EUR) |",
                     "|---|-------------|--------|----------------|"]
            for i, t in enumerate(top20[:20], 1):
                rows.append(
                    f"| {i} | {t.get('name', '')} | {t.get('sector', '')} | "
                    f"{t.get('exposure', 0.0):,.0f} |"
                )
            sections.append("\n".join(rows))

        # Validation issues
        issues = data.get("validation_issues", [])
        if issues:
            rows = ["## Validation Issues\n",
                     "| Severity | Issue |",
                     "|----------|-------|"]
            for iss in issues:
                rows.append(f"| {iss.get('severity', '')} | {iss.get('issue', '')} |")
            sections.append("\n".join(rows))

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Credit Institution"))
        body = f'<div class="section"><h2>Pillar 3 ESG Summary</h2>'
        body += f'<p>Templates Populated: {data.get("templates_populated", 0)}/{data.get("templates_total", 10)}</p>'
        body += f'<p>Filing Ready: {"Yes" if data.get("filing_ready") else "No"}</p></div>'
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"EBA Pillar 3 ESG - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "pillar3_esg_its",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "lei": data.get("lei", ""),
            "reporting_date": data.get("reporting_date", ""),
            "templates_populated": data.get("templates_populated", 0),
            "templates_total": data.get("templates_total", 10),
            "completeness_pct": data.get("completeness_pct", 0.0),
            "data_quality_score": data.get("data_quality_score", 0.0),
            "filing_ready": data.get("filing_ready", False),
            "alignment_metrics": {
                "gar_pct": data.get("gar_pct"),
                "btar_pct": data.get("btar_pct"),
                "financed_emissions_tco2e": data.get("financed_emissions_tco2e"),
            },
            "top_20_carbon_intensive": data.get("top_20_carbon_intensive", []),
            "validation_issues": data.get("validation_issues", []),
        }
        cs = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(cs)
        return report

    def _md_footer(self, provenance_hash: str) -> str:
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>body{font-family:Arial,sans-serif;margin:40px auto;max-width:1000px}"
            "h1{color:#1a5276;border-bottom:3px solid #2ecc71}h2{color:#1a5276}"
            ".section{margin:20px 0;padding:15px;background:#fafafa;border:1px solid #eee;border-radius:6px}"
            ".data-table{width:100%;border-collapse:collapse}.data-table td,.data-table th{padding:8px;border:1px solid #ddd}"
            ".data-table th{background:#1a5276;color:#fff}"
            ".provenance{margin-top:30px;padding:10px;background:#eaf2f8;font-family:monospace;font-size:.85em}"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n{body}\n"
            f'<div class="provenance">Provenance: {provenance_hash}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
