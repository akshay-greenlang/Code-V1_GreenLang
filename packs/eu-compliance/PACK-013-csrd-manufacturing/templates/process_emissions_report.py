# -*- coding: utf-8 -*-
"""
ProcessEmissionsReportTemplate - Manufacturing Process Emissions Report

Generates process emissions breakdown reports for manufacturing facilities
including facility summary, process line details, sub-sector comparison,
CBAM embedded emissions, ETS benchmark, and abatement tracking.

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
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class ProcessEmissionsReportData(BaseModel):
    """Data model for process emissions report."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    facilities: List[Dict[str, Any]] = Field(default_factory=list)
    scope1_total: float = Field(default=0.0)
    process_emissions: float = Field(default=0.0)
    combustion_emissions: float = Field(default=0.0)
    process_lines: List[Dict[str, Any]] = Field(default_factory=list)
    sub_sector_comparison: List[Dict[str, Any]] = Field(default_factory=list)
    cbam_embedded: Dict[str, Any] = Field(default_factory=dict)
    ets_benchmark: Dict[str, Any] = Field(default_factory=dict)
    abatement_measures: List[Dict[str, Any]] = Field(default_factory=list)


class ProcessEmissionsReportTemplate:
    """
    Manufacturing process emissions breakdown report template.

    Generates reports covering facility summary, process line details,
    CBAM embedded emissions, ETS benchmarks, and abatement tracking.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "process_emissions_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render report in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format '{fmt}'.")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render as GitHub-flavored Markdown."""
        sections: List[str] = []
        name = data.get("company_name", "Manufacturing Company")
        year = data.get("reporting_year", 2024)
        s1 = data.get("scope1_total", 0.0)
        proc = data.get("process_emissions", 0.0)
        comb = data.get("combustion_emissions", 0.0)

        sections.append(
            f"# Process Emissions Report\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        sections.append(
            f"## Scope 1 Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| **Total Scope 1** | **{s1:,.2f} tCO2e** |\n"
            f"| Process Emissions | {proc:,.2f} tCO2e |\n"
            f"| Combustion Emissions | {comb:,.2f} tCO2e |"
        )

        facilities = data.get("facilities", [])
        if facilities:
            rows = ["## Facility Breakdown\n",
                     "| Facility | Country | Scope 1 (tCO2e) | Process (tCO2e) |",
                     "|----------|---------|-----------------|-----------------|"]
            for f in facilities:
                rows.append(f"| {f.get('name', '')} | {f.get('country', '')} | "
                            f"{f.get('scope1', 0.0):,.2f} | {f.get('process', 0.0):,.2f} |")
            sections.append("\n".join(rows))

        lines = data.get("process_lines", [])
        if lines:
            rows = ["## Process Line Details\n",
                     "| Line | Product | Emissions (tCO2e) | Intensity (tCO2e/t) |",
                     "|------|---------|-------------------|---------------------|"]
            for ln in lines:
                rows.append(f"| {ln.get('line_name', '')} | {ln.get('product', '')} | "
                            f"{ln.get('emissions', 0.0):,.2f} | {ln.get('intensity', 0.0):,.4f} |")
            sections.append("\n".join(rows))

        cbam = data.get("cbam_embedded", {})
        if cbam:
            sections.append(
                f"## CBAM Embedded Emissions\n\n"
                f"| Product | Embedded (tCO2e/t) | CBAM Liability (EUR) |\n"
                f"|---------|--------------------|-----------------------|\n"
                + "\n".join(
                    f"| {p.get('product', '')} | {p.get('embedded', 0.0):,.4f} | "
                    f"{p.get('liability_eur', 0.0):,.0f} |"
                    for p in cbam.get("products", [])
                )
            )

        ets = data.get("ets_benchmark", {})
        if ets:
            sections.append(
                f"## ETS Benchmark\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Benchmark (tCO2e/t) | {ets.get('benchmark', 0.0):,.4f} |\n"
                f"| Actual (tCO2e/t) | {ets.get('actual', 0.0):,.4f} |\n"
                f"| Free Allocation (tCO2e) | {ets.get('free_allocation', 0.0):,.0f} |\n"
                f"| Shortfall (tCO2e) | {ets.get('shortfall', 0.0):,.0f} |"
            )

        abatement = data.get("abatement_measures", [])
        if abatement:
            rows = ["## Abatement Tracking\n",
                     "| Measure | Reduction (tCO2e) | Status | Investment (EUR) |",
                     "|---------|-------------------|--------|------------------|"]
            for a in abatement:
                rows.append(f"| {a.get('name', '')} | {a.get('reduction', 0.0):,.0f} | "
                            f"{a.get('status', '')} | {a.get('investment', 0.0):,.0f} |")
            sections.append("\n".join(rows))

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n"
        content += f"**Provenance:** `{ph}`\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render as self-contained HTML."""
        name = _esc(data.get("company_name", "Manufacturing Company"))
        s1 = data.get("scope1_total", 0.0)
        body = (
            f'<div class="section"><h2>Scope 1 Summary</h2>'
            f'<table class="data-table"><tr><th>Metric</th><th>Value</th></tr>'
            f'<tr><td>Total Scope 1</td><td>{s1:,.2f} tCO2e</td></tr>'
            f'</table></div>'
        )
        ph = self._provenance(body)
        return self._wrap_html(f"Process Emissions - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        report = {"report_type": self.TEMPLATE_NAME, "pack_id": self.PACK_ID,
                  "version": self.VERSION, "generated_at": self.generated_at, **data}
        report["provenance_hash"] = self._provenance(json.dumps(report, default=str, sort_keys=True))
        return report

    @staticmethod
    def _provenance(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, title: str, body: str, ph: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<title>{_esc(title)}</title>'
            f'<style>body{{font-family:sans-serif;max-width:1000px;margin:40px auto}}'
            f'.data-table{{width:100%;border-collapse:collapse}}'
            f'.data-table td,.data-table th{{padding:8px;border:1px solid #ddd}}'
            f'.data-table th{{background:#1a5276;color:#fff}}</style>'
            f'</head><body><h1>{_esc(title)}</h1>{body}'
            f'<div style="margin-top:30px;font-family:monospace;font-size:0.85em">'
            f'Provenance: {ph}</div><!-- provenance_hash: {ph} --></body></html>'
        )
