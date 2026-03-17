# -*- coding: utf-8 -*-
"""
InsuranceESGTemplate - Insurance ESG Disclosure

Generates insurance-specific ESG disclosure covering underwriting emissions,
Solvency II ESG integration, and responsible underwriting practices.

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



class InsuranceESGData(BaseModel):
    """Data for insurance ESG template."""
    institution_name: str = Field(default="")
    reporting_period: str = Field(default="")
    gross_attributed_emissions_tco2e: float = Field(default=0.0)
    net_attributed_emissions_tco2e: float = Field(default=0.0)
    reinsurance_adjustment_tco2e: float = Field(default=0.0)
    total_gwp_eur: float = Field(default=0.0)
    emission_intensity_per_meur_gwp: float = Field(default=0.0)
    policies_covered: int = Field(default=0)
    weighted_data_quality_score: float = Field(default=5.0)
    by_line_of_business: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    solvency_ii_esg: Dict[str, Any] = Field(default_factory=dict)
    responsible_underwriting: Dict[str, Any] = Field(default_factory=dict)
    exclusion_policies: List[str] = Field(default_factory=list)
    engagement_activities: List[str] = Field(default_factory=list)


class InsuranceESGTemplate:
    """Insurance ESG disclosure template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "insurance_esg_template"
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
        name = data.get("institution_name", "Insurance Undertaking")
        period = data.get("reporting_period", "")
        gross = data.get("gross_attributed_emissions_tco2e", 0.0)
        net = data.get("net_attributed_emissions_tco2e", 0.0)
        ceded = data.get("reinsurance_adjustment_tco2e", 0.0)
        gwp = data.get("total_gwp_eur", 0.0)
        intensity = data.get("emission_intensity_per_meur_gwp", 0.0)
        policies = data.get("policies_covered", 0)
        wdq = data.get("weighted_data_quality_score", 5.0)

        sections.append(
            f"# Insurance ESG Disclosure\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        sections.append(
            f"## Underwriting Emissions Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Gross Attributed Emissions | {gross:,.2f} tCO2e |\n"
            f"| Reinsurance Adjustment | ({ceded:,.2f}) tCO2e |\n"
            f"| **Net Attributed Emissions** | **{net:,.2f} tCO2e** |\n"
            f"| Total GWP | EUR {gwp:,.0f} |\n"
            f"| Emission Intensity | {intensity:,.2f} tCO2e/EUR M GWP |\n"
            f"| Policies Covered | {policies:,} |\n"
            f"| Data Quality Score | {wdq:.2f} / 5.0 |"
        )

        # By line of business
        by_lob = data.get("by_line_of_business", {})
        if by_lob:
            rows = ["## Emissions by Line of Business\n",
                     "| Line of Business | GWP (EUR) | Gross Emissions | Net Emissions |",
                     "|-----------------|-----------|-----------------|---------------|"]
            for lob, vals in by_lob.items():
                rows.append(
                    f"| {lob} | {vals.get('gwp', 0.0):,.0f} | "
                    f"{vals.get('gross', 0.0):,.2f} | {vals.get('net', 0.0):,.2f} |"
                )
            sections.append("\n".join(rows))

        # Solvency II
        sol2 = data.get("solvency_ii_esg", {})
        if sol2:
            sections.append("## Solvency II ESG Integration\n")
            for k, v in sol2.items():
                sections.append(f"**{k}:** {v}\n")

        # Responsible underwriting
        ru = data.get("responsible_underwriting", {})
        if ru:
            sections.append("## Responsible Underwriting\n")
            for k, v in ru.items():
                sections.append(f"**{k}:** {v}\n")

        # Exclusion policies
        excl = data.get("exclusion_policies", [])
        if excl:
            sections.append("## Exclusion Policies\n")
            for e in excl:
                sections.append(f"- {e}")

        # Engagement
        eng = data.get("engagement_activities", [])
        if eng:
            sections.append("## Engagement Activities\n")
            for e in eng:
                sections.append(f"- {e}")

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Insurance Undertaking"))
        net = data.get("net_attributed_emissions_tco2e", 0.0)
        body = (
            f'<div class="section"><h2>Underwriting Emissions</h2>'
            f'<p><strong>Net Attributed:</strong> {net:,.2f} tCO2e</p></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"Insurance ESG - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "insurance_esg_disclosure",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "emissions": {
                "gross": data.get("gross_attributed_emissions_tco2e", 0.0),
                "net": data.get("net_attributed_emissions_tco2e", 0.0),
                "reinsurance_adjustment": data.get("reinsurance_adjustment_tco2e", 0.0),
            },
            "gwp_eur": data.get("total_gwp_eur", 0.0),
            "intensity": data.get("emission_intensity_per_meur_gwp", 0.0),
            "by_line_of_business": data.get("by_line_of_business", {}),
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
