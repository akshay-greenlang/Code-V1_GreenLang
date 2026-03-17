# -*- coding: utf-8 -*-
"""
ManufacturingScorecardTemplate - Sustainability Scorecard

Generates manufacturing sustainability scorecards with KPI dashboard,
peer percentile rankings, SBTi alignment status, trajectory analysis,
improvement priorities, and OEE sustainability overlay.

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


class ManufacturingScorecardData(BaseModel):
    """Data model for manufacturing sustainability scorecard."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    kpi_dashboard: List[Dict[str, Any]] = Field(default_factory=list)
    peer_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_alignment: Dict[str, Any] = Field(default_factory=dict)
    trajectory_analysis: Dict[str, Any] = Field(default_factory=dict)
    improvement_priorities: List[Dict[str, Any]] = Field(default_factory=list)
    oee_overlay: Dict[str, Any] = Field(default_factory=dict)
    overall_score: float = Field(default=0.0)


class ManufacturingScorecardTemplate:
    """
    Manufacturing sustainability scorecard template.

    Generates a comprehensive scorecard with KPI tracking, peer
    benchmarking, SBTi alignment, and OEE sustainability overlay.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "manufacturing_scorecard"
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
        score = data.get("overall_score", 0.0)

        grade = self._grade(score)

        sections.append(
            f"# Manufacturing Sustainability Scorecard\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Overall Score:** **{score:.0f}/100** ({grade})\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # KPI Dashboard
        kpis = data.get("kpi_dashboard", [])
        if kpis:
            rows = ["## KPI Dashboard\n",
                     "| KPI | Value | Target | Status | Trend |",
                     "|-----|-------|--------|--------|-------|"]
            for k in kpis:
                status = k.get("status", "N/A")
                trend = k.get("trend", "stable")
                trend_sym = {"improving": "^", "stable": "-", "declining": "v"}.get(trend, "-")
                rows.append(
                    f"| {k.get('kpi_name', '')} | {k.get('value', '')} | "
                    f"{k.get('target', '')} | {status} | {trend_sym} |"
                )
            sections.append("\n".join(rows))

        # Peer rankings
        peers = data.get("peer_rankings", [])
        if peers:
            rows = ["## Peer Percentile Rankings\n",
                     "| Metric | Your Value | Peer Median | Percentile |",
                     "|--------|-----------|-------------|------------|"]
            for p in peers:
                pctl = p.get("percentile", 50)
                rows.append(
                    f"| {p.get('metric', '')} | {p.get('your_value', '')} | "
                    f"{p.get('peer_median', '')} | {pctl}th |"
                )
            sections.append("\n".join(rows))

        # SBTi alignment
        sbti = data.get("sbti_alignment", {})
        if sbti:
            aligned = sbti.get("aligned", False)
            sections.append(
                f"## SBTi Alignment\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Pathway | {sbti.get('pathway', '1.5C')} |\n"
                f"| Status | {'Aligned' if aligned else 'Gap Identified'} |\n"
                f"| Annual Rate | {sbti.get('annual_rate', 0.0):,.1f}% |\n"
                f"| Required Rate | {sbti.get('required_rate', 4.2):,.1f}% |\n"
                f"| Gap | {sbti.get('gap_pct', 0.0):,.1f}% |"
            )

        # Trajectory analysis
        traj = data.get("trajectory_analysis", {})
        if traj:
            sections.append(
                f"## Trajectory Analysis\n\n"
                f"**Baseline ({traj.get('baseline_year', 'N/A')}):** "
                f"{traj.get('baseline_emissions', 0.0):,.0f} tCO2e\n\n"
                f"**Current ({traj.get('current_year', 'N/A')}):** "
                f"{traj.get('current_emissions', 0.0):,.0f} tCO2e\n\n"
                f"**Reduction:** {traj.get('reduction_pct', 0.0):,.1f}%\n\n"
                f"**On Track:** {traj.get('on_track', False)}"
            )

        # Improvement priorities
        priorities = data.get("improvement_priorities", [])
        if priorities:
            rows = ["## Improvement Priorities\n",
                     "| Priority | Area | Impact | Effort | ROI |",
                     "|----------|------|--------|--------|-----|"]
            for idx, p in enumerate(priorities):
                rows.append(
                    f"| {idx + 1} | {p.get('area', '')} | {p.get('impact', '')} | "
                    f"{p.get('effort', '')} | {p.get('roi', '')} |"
                )
            sections.append("\n".join(rows))

        # OEE overlay
        oee = data.get("oee_overlay", {})
        if oee:
            sections.append(
                f"## OEE Sustainability Overlay\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| OEE | {oee.get('oee_pct', 0.0):,.1f}% |\n"
                f"| Availability | {oee.get('availability_pct', 0.0):,.1f}% |\n"
                f"| Performance | {oee.get('performance_pct', 0.0):,.1f}% |\n"
                f"| Quality | {oee.get('quality_pct', 0.0):,.1f}% |\n"
                f"| Energy per Good Unit | {oee.get('energy_per_good_unit', 0.0):,.3f} MWh |\n"
                f"| Emissions per Good Unit | {oee.get('emissions_per_good_unit', 0.0):,.4f} tCO2e |"
            )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        score = data.get("overall_score", 0.0)
        grade = self._grade(score)
        color = "#27ae60" if score >= 70 else ("#f39c12" if score >= 40 else "#e74c3c")
        body = (
            f'<div class="section" style="text-align:center">'
            f'<h2>Sustainability Score</h2>'
            f'<p style="font-size:3em;color:{color}"><strong>{score:.0f}/100</strong></p>'
            f'<p style="font-size:1.5em">{grade}</p></div>'
        )
        ph = self._provenance(body)
        return self._wrap_html(f"Scorecard - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report = {"report_type": self.TEMPLATE_NAME, "pack_id": self.PACK_ID,
                  "version": self.VERSION, "generated_at": self.generated_at, **data}
        report["provenance_hash"] = self._provenance(json.dumps(report, default=str, sort_keys=True))
        return report

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A - Leader"
        if score >= 75:
            return "B - Advanced"
        if score >= 60:
            return "C - Progressing"
        if score >= 40:
            return "D - Developing"
        return "E - Beginning"

    @staticmethod
    def _provenance(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, title: str, body: str, ph: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<title>{_esc(title)}</title>'
            f'<style>body{{font-family:sans-serif;max-width:1000px;margin:40px auto}}'
            f'.section{{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px}}</style>'
            f'</head><body><h1>{_esc(title)}</h1>{body}'
            f'<div style="margin-top:30px;font-family:monospace;font-size:0.85em">'
            f'Provenance: {ph}</div></body></html>'
        )
