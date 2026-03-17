# -*- coding: utf-8 -*-
"""
FSESRSChapterTemplate - FI-specific ESRS Chapters

Generates financial institution-specific ESRS chapters covering E1 (financed
emissions), S1-S4 (financial inclusion, responsible lending), and G1
(board governance, ESG remuneration linkage).

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



class FSESRSChapterData(BaseModel):
    """Data for FI-specific ESRS chapters."""
    institution_name: str = Field(default="")
    reporting_period: str = Field(default="")
    # E1 - Climate
    financed_emissions_tco2e: float = Field(default=0.0)
    portfolio_intensity: float = Field(default=0.0)
    transition_plan_summary: str = Field(default="")
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    # S1-S4
    total_employees: int = Field(default=0)
    financial_inclusion_metrics: Dict[str, Any] = Field(default_factory=dict)
    responsible_lending_metrics: Dict[str, Any] = Field(default_factory=dict)
    consumer_protection_metrics: Dict[str, Any] = Field(default_factory=dict)
    # G1
    board_esg_oversight: str = Field(default="")
    esg_remuneration_linkage: str = Field(default="")
    anti_money_laundering: str = Field(default="")
    whistleblower_policy: str = Field(default="")
    # Materiality
    material_topics: List[Dict[str, Any]] = Field(default_factory=list)


class FSESRSChapterTemplate:
    """FI-specific ESRS chapter template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "fs_esrs_chapter"
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

        sections.append(
            f"# ESRS Disclosure - Financial Institution Specific\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # E1 - Climate (FI-specific)
        fe = data.get("financed_emissions_tco2e", 0.0)
        pi = data.get("portfolio_intensity", 0.0)
        tp = data.get("transition_plan_summary", "")

        sections.append(
            f"## ESRS E1 - Climate Change (FI-Specific)\n\n"
            f"### Financed Emissions (Scope 3 Category 15)\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Financed Emissions | {fe:,.2f} tCO2e |\n"
            f"| Portfolio Carbon Intensity | {pi:,.4f} tCO2e/EUR M |"
        )

        if tp:
            sections.append(f"### Transition Plan Summary\n\n{tp}")

        targets = data.get("targets", [])
        if targets:
            rows = ["### Decarbonization Targets\n",
                     "| Sector | Baseline | Target 2030 | On Track |",
                     "|--------|----------|-------------|----------|"]
            for t in targets:
                rows.append(
                    f"| {t.get('sector', '')} | {t.get('baseline', 0.0):.1f} | "
                    f"{t.get('target_2030', 'N/A')} | {t.get('on_track', 'N/A')} |"
                )
            sections.append("\n".join(rows))

        # S1-S4
        fi_metrics = data.get("financial_inclusion_metrics", {})
        rl_metrics = data.get("responsible_lending_metrics", {})
        cp_metrics = data.get("consumer_protection_metrics", {})

        sections.append("## ESRS S1-S4 - Social (FI-Specific)")

        if fi_metrics:
            rows = ["### Financial Inclusion\n",
                     "| Metric | Value |",
                     "|--------|-------|"]
            for k, v in fi_metrics.items():
                rows.append(f"| {k} | {v} |")
            sections.append("\n".join(rows))

        if rl_metrics:
            rows = ["### Responsible Lending\n",
                     "| Metric | Value |",
                     "|--------|-------|"]
            for k, v in rl_metrics.items():
                rows.append(f"| {k} | {v} |")
            sections.append("\n".join(rows))

        if cp_metrics:
            rows = ["### Consumer Protection\n",
                     "| Metric | Value |",
                     "|--------|-------|"]
            for k, v in cp_metrics.items():
                rows.append(f"| {k} | {v} |")
            sections.append("\n".join(rows))

        # G1 - Governance
        sections.append("## ESRS G1 - Business Conduct (FI-Specific)")

        board = data.get("board_esg_oversight", "")
        if board:
            sections.append(f"### Board ESG Oversight\n\n{board}")

        remun = data.get("esg_remuneration_linkage", "")
        if remun:
            sections.append(f"### ESG-Linked Remuneration\n\n{remun}")

        aml = data.get("anti_money_laundering", "")
        if aml:
            sections.append(f"### Anti-Money Laundering\n\n{aml}")

        wb = data.get("whistleblower_policy", "")
        if wb:
            sections.append(f"### Whistleblower Policy\n\n{wb}")

        # Materiality
        mat = data.get("material_topics", [])
        if mat:
            rows = ["## Material Topics (Double Materiality)\n",
                     "| Topic | ESRS | Impact | Financial | Classification |",
                     "|-------|------|--------|-----------|----------------|"]
            for m in mat:
                rows.append(
                    f"| {m.get('name', '')} | {m.get('esrs_reference', '')} | "
                    f"{m.get('impact_score', 0.0):.1f} | {m.get('financial_score', 0.0):.1f} | "
                    f"{m.get('classification', '')} |"
                )
            sections.append("\n".join(rows))

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Financial Institution"))
        fe = data.get("financed_emissions_tco2e", 0.0)
        body = (
            f'<div class="section"><h2>E1 - Financed Emissions</h2>'
            f'<p>Total: {fe:,.2f} tCO2e</p></div>'
            f'<div class="section"><h2>S1-S4 - Social</h2>'
            f'<p>FI-specific social metrics</p></div>'
            f'<div class="section"><h2>G1 - Governance</h2>'
            f'<p>Board ESG oversight and conduct</p></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"ESRS FI-Specific Chapters - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "fs_esrs_chapters",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "e1_climate": {
                "financed_emissions_tco2e": data.get("financed_emissions_tco2e", 0.0),
                "portfolio_intensity": data.get("portfolio_intensity", 0.0),
                "targets": data.get("targets", []),
            },
            "social": {
                "financial_inclusion": data.get("financial_inclusion_metrics", {}),
                "responsible_lending": data.get("responsible_lending_metrics", {}),
                "consumer_protection": data.get("consumer_protection_metrics", {}),
            },
            "governance": {
                "board_esg_oversight": data.get("board_esg_oversight", ""),
                "esg_remuneration_linkage": data.get("esg_remuneration_linkage", ""),
                "anti_money_laundering": data.get("anti_money_laundering", ""),
            },
            "material_topics": data.get("material_topics", []),
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
