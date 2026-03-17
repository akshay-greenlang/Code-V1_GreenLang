# -*- coding: utf-8 -*-
"""
EnergyPerformanceReportTemplate - Energy Performance Dashboard

Generates energy performance dashboard reports for manufacturing facilities
including total energy, SEC by product, energy mix, benchmark comparison,
EED compliance, decarbonization opportunities, and ISO 50001 status.

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
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class EnergyPerformanceReportData(BaseModel):
    """Data model for energy performance report."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    total_energy_mwh: float = Field(default=0.0)
    energy_by_source: Dict[str, float] = Field(default_factory=dict)
    sec_by_product: List[Dict[str, Any]] = Field(default_factory=list)
    benchmark_comparison: Dict[str, Any] = Field(default_factory=dict)
    eed_compliance: Dict[str, Any] = Field(default_factory=dict)
    decarbonization_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    iso50001_status: Dict[str, Any] = Field(default_factory=dict)
    renewable_pct: float = Field(default=0.0)


class EnergyPerformanceReportTemplate:
    """
    Energy performance dashboard report template.

    Covers total energy consumption, specific energy consumption (SEC),
    energy mix analysis, benchmarking, EED compliance, and ISO 50001 status.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "energy_performance_report"
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
        name = data.get("company_name", "Manufacturing Company")
        year = data.get("reporting_year", 2024)
        total = data.get("total_energy_mwh", 0.0)
        renewable = data.get("renewable_pct", 0.0)

        sections.append(
            f"# Energy Performance Report\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        sections.append(
            f"## Energy Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| **Total Energy** | **{total:,.0f} MWh** |\n"
            f"| Renewable Share | {renewable:,.1f}% |\n"
            f"| Non-Renewable | {100 - renewable:,.1f}% |"
        )

        by_source = data.get("energy_by_source", {})
        if by_source:
            rows = ["## Energy Mix\n",
                     "| Source | MWh | Share (%) |",
                     "|--------|-----|-----------|"]
            for src, mwh in by_source.items():
                pct = (mwh / total * 100) if total > 0 else 0.0
                rows.append(f"| {src.replace('_', ' ').title()} | {mwh:,.0f} | {pct:,.1f}% |")
            sections.append("\n".join(rows))

        sec = data.get("sec_by_product", [])
        if sec:
            rows = ["## Specific Energy Consumption (SEC)\n",
                     "| Product | SEC (MWh/tonne) | Benchmark | Gap (%) |",
                     "|---------|-----------------|-----------|---------|"]
            for s in sec:
                rows.append(
                    f"| {s.get('product', '')} | {s.get('sec', 0.0):,.3f} | "
                    f"{s.get('benchmark', 0.0):,.3f} | {s.get('gap_pct', 0.0):+,.1f}% |"
                )
            sections.append("\n".join(rows))

        bench = data.get("benchmark_comparison", {})
        if bench:
            sections.append(
                f"## Benchmark Comparison\n\n"
                f"| Metric | Facility | Sector Avg | Percentile |\n"
                f"|--------|----------|------------|------------|\n"
                f"| Energy Intensity | {bench.get('facility', 0.0):,.3f} | "
                f"{bench.get('sector_avg', 0.0):,.3f} | {bench.get('percentile', 0)}th |"
            )

        eed = data.get("eed_compliance", {})
        if eed:
            sections.append(
                f"## EED Compliance\n\n"
                f"| Requirement | Status |\n|-------------|--------|\n"
                f"| Energy Audit | {eed.get('audit_status', 'N/A')} |\n"
                f"| EnMS Certified | {eed.get('enms_certified', False)} |\n"
                f"| Reporting Complete | {eed.get('reporting_complete', False)} |"
            )

        opps = data.get("decarbonization_opportunities", [])
        if opps:
            rows = ["## Decarbonization Opportunities\n",
                     "| Opportunity | Savings (MWh/yr) | Investment (EUR) | Payback (yrs) |",
                     "|------------|-------------------|------------------|---------------|"]
            for o in opps:
                rows.append(
                    f"| {o.get('name', '')} | {o.get('savings_mwh', 0.0):,.0f} | "
                    f"{o.get('investment', 0.0):,.0f} | {o.get('payback_years', 0.0):,.1f} |"
                )
            sections.append("\n".join(rows))

        iso = data.get("iso50001_status", {})
        if iso:
            sections.append(
                f"## ISO 50001 Status\n\n"
                f"**Certified:** {iso.get('certified', False)} | "
                f"**Scope:** {iso.get('scope', 'N/A')} | "
                f"**Next Audit:** {iso.get('next_audit', 'N/A')}"
            )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        total = data.get("total_energy_mwh", 0.0)
        body = (
            f'<div class="section"><h2>Energy Summary</h2>'
            f'<p style="font-size:2em"><strong>{total:,.0f} MWh</strong></p></div>'
        )
        ph = self._provenance(body)
        return self._wrap_html(f"Energy Performance - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
            f'.data-table th{{background:#1a5276;color:#fff}}'
            f'.section{{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px}}</style>'
            f'</head><body><h1>{_esc(title)}</h1>{body}'
            f'<div style="margin-top:30px;font-family:monospace;font-size:0.85em">'
            f'Provenance: {ph}</div></body></html>'
        )
