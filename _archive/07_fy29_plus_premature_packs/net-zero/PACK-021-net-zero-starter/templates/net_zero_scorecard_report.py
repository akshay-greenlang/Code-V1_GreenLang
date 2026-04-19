# -*- coding: utf-8 -*-
"""
NetZeroScorecardReportTemplate - Net-zero maturity scorecard for PACK-021.

Renders a maturity scorecard across 8 dimensions of net-zero readiness,
with overall scores, dimension details, radar chart data, priority
recommendations, improvement roadmap, and peer comparison context.

Sections:
    1. Overall Score & Maturity Level
    2. Dimension Scores (8 dimensions, radar chart data)
    3. Dimension Details
    4. Priority Recommendations (ranked)
    5. Improvement Roadmap
    6. Peer Comparison Context

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

_DIMENSIONS: List[Dict[str, str]] = [
    {"id": "governance", "name": "Governance & Accountability",
     "description": "Board oversight, committee structure, incentive alignment"},
    {"id": "baseline", "name": "GHG Baseline & Inventory",
     "description": "Completeness, accuracy, verification of emissions data"},
    {"id": "targets", "name": "Target Setting",
     "description": "SBTi alignment, ambition, scope coverage, timeline"},
    {"id": "reduction", "name": "Reduction Strategy",
     "description": "MACC analysis, roadmap, quick wins, investment planning"},
    {"id": "data_quality", "name": "Data & Measurement",
     "description": "Primary data share, EF quality, MRV systems"},
    {"id": "engagement", "name": "Stakeholder Engagement",
     "description": "Supplier programs, employee training, investor communication"},
    {"id": "offsets", "name": "Offset & Neutralization",
     "description": "Portfolio quality, removal share, VCMI alignment"},
    {"id": "disclosure", "name": "Disclosure & Reporting",
     "description": "CDP, TCFD, CSRD, voluntary disclosures, transparency"},
]

_MATURITY_LEVELS: List[Dict[str, Any]] = [
    {"min": 0, "max": 20, "level": "Nascent", "description": "Early awareness, minimal action"},
    {"min": 20, "max": 40, "level": "Developing", "description": "Initial commitments, partial data"},
    {"min": 40, "max": 60, "level": "Established", "description": "Targets set, plans in progress"},
    {"min": 60, "max": 80, "level": "Advanced", "description": "Comprehensive strategy, strong execution"},
    {"min": 80, "max": 100, "level": "Leading", "description": "Best-in-class, continuous improvement"},
]

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _get_maturity(score: float) -> Dict[str, Any]:
    for level in _MATURITY_LEVELS:
        if level["min"] <= score < level["max"]:
            return level
    return _MATURITY_LEVELS[-1]

class NetZeroScorecardReportTemplate:
    """
    Net-zero maturity scorecard template.

    Assesses organizational readiness for net-zero across 8 dimensions
    with scoring, maturity levels, recommendations, and peer context.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_dimension_scores(data),
            self._md_dimension_details(data),
            self._md_recommendations(data),
            self._md_improvement_roadmap(data),
            self._md_peer_comparison(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overall_score(data),
            self._html_dimension_scores(data),
            self._html_dimension_details(data),
            self._html_recommendations(data),
            self._html_improvement_roadmap(data),
            self._html_peer_comparison(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Net Zero Scorecard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        dim_scores = data.get("dimension_scores", {})
        overall = self._calc_overall(dim_scores)
        maturity = _get_maturity(overall)

        result: Dict[str, Any] = {
            "template": "net_zero_scorecard_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "overall": {
                "score": _dec(overall),
                "maturity_level": maturity["level"],
                "maturity_description": maturity["description"],
            },
            "dimensions": [
                {
                    "id": dim["id"],
                    "name": dim["name"],
                    "description": dim["description"],
                    "score": _dec(dim_scores.get(dim["id"], {}).get("score", 0)),
                    "maturity": _get_maturity(
                        float(Decimal(str(dim_scores.get(dim["id"], {}).get("score", 0))))
                    )["level"],
                    "key_findings": dim_scores.get(dim["id"], {}).get("findings", []),
                    "strengths": dim_scores.get(dim["id"], {}).get("strengths", []),
                    "gaps": dim_scores.get(dim["id"], {}).get("gaps", []),
                }
                for dim in _DIMENSIONS
            ],
            "radar_chart_data": {
                "labels": [d["name"] for d in _DIMENSIONS],
                "values": [
                    float(Decimal(str(dim_scores.get(d["id"], {}).get("score", 0))))
                    for d in _DIMENSIONS
                ],
            },
            "recommendations": data.get("recommendations", []),
            "improvement_roadmap": data.get("improvement_roadmap", []),
            "peer_comparison": data.get("peer_comparison", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Net Zero Maturity Scorecard\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Assessment Date:** {data.get('assessment_date', ts)}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        overall = self._calc_overall(dim_scores)
        maturity = _get_maturity(overall)
        return (
            f"## 1. Overall Score & Maturity Level\n\n"
            f"### {_dec(overall)} / 100 - {maturity['level']}\n\n"
            f"*{maturity['description']}*\n\n"
            f"| Level | Range | Status |\n"
            f"|-------|-------|--------|\n"
            + "\n".join(
                f"| {ml['level']} | {ml['min']}-{ml['max']} | "
                f"{'**CURRENT**' if ml['level'] == maturity['level'] else ''} |"
                for ml in _MATURITY_LEVELS
            )
        )

    def _md_dimension_scores(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        lines = [
            "## 2. Dimension Scores\n",
            "*Data suitable for radar/spider chart visualization.*\n",
            "| Dimension | Score | Maturity Level |",
            "|-----------|------:|---------------|",
        ]
        for dim in _DIMENSIONS:
            score = float(Decimal(str(dim_scores.get(dim["id"], {}).get("score", 0))))
            maturity = _get_maturity(score)
            lines.append(
                f"| {dim['name']} | {_dec(score)} | {maturity['level']} |"
            )
        return "\n".join(lines)

    def _md_dimension_details(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        sections = ["## 3. Dimension Details\n"]
        for dim in _DIMENSIONS:
            dim_data = dim_scores.get(dim["id"], {})
            score = float(Decimal(str(dim_data.get("score", 0))))
            maturity = _get_maturity(score)
            findings = dim_data.get("findings", [])
            strengths = dim_data.get("strengths", [])
            gaps = dim_data.get("gaps", [])

            sections.append(
                f"### {dim['name']} - {_dec(score)}/100 ({maturity['level']})\n"
            )
            sections.append(f"_{dim['description']}_\n")
            if findings:
                sections.append("**Key Findings:**")
                for f in findings:
                    sections.append(f"- {f}")
            if strengths:
                sections.append("\n**Strengths:**")
                for s in strengths:
                    sections.append(f"- {s}")
            if gaps:
                sections.append("\n**Gaps:**")
                for g in gaps:
                    sections.append(f"- {g}")
            sections.append("")
        return "\n".join(sections)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 4. Priority Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                priority = rec.get("priority", "MEDIUM")
                lines.append(
                    f"### {i}. [{priority}] {rec.get('title', '')}\n"
                )
                lines.append(f"**Dimension:** {rec.get('dimension', '-')}")
                lines.append(f"**Impact:** {rec.get('impact', '-')}")
                lines.append(f"**Effort:** {rec.get('effort', '-')}\n")
                lines.append(f"{rec.get('description', '')}\n")
        else:
            lines.append("_No recommendations at this time._")
        return "\n".join(lines)

    def _md_improvement_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("improvement_roadmap", [])
        lines = [
            "## 5. Improvement Roadmap\n",
            "| # | Action | Dimension | Target Score | Timeline | Status |",
            "|---|--------|-----------|------------:|----------|--------|",
        ]
        for i, item in enumerate(roadmap, 1):
            lines.append(
                f"| {i} | {item.get('action', '-')} "
                f"| {item.get('dimension', '-')} "
                f"| {_dec(item.get('target_score', 0))} "
                f"| {item.get('timeline', '-')} "
                f"| {item.get('status', '-')} |"
            )
        if not roadmap:
            lines.append("| - | _No roadmap actions defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        peer = data.get("peer_comparison", {})
        sector_avg = peer.get("sector_avg_score", 0)
        sector_leader = peer.get("sector_leader_score", 0)
        percentile = peer.get("percentile_rank", 0)
        dim_scores = data.get("dimension_scores", {})
        overall = self._calc_overall(dim_scores)
        return (
            "## 6. Peer Comparison Context\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Your Score | {_dec(overall)} |\n"
            f"| Sector Average | {_dec(sector_avg)} |\n"
            f"| Sector Leader | {_dec(sector_leader)} |\n"
            f"| Your Percentile | P{_dec(percentile, 0)} |\n"
            f"| Gap to Average | {_dec(overall - sector_avg)} |\n"
            f"| Gap to Leader | {_dec(overall - sector_leader)} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*8-dimension maturity framework applied.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f5f7f5;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".score-display{font-size:2.5em;font-weight:800;text-align:center;padding:24px;"
            "border-radius:12px;margin:20px 0;}"
            ".score-leading{background:#c8e6c9;color:#1b5e20;}"
            ".score-advanced{background:#dcedc8;color:#33691e;}"
            ".score-established{background:#fff9c4;color:#f57f17;}"
            ".score-developing{background:#ffe0b2;color:#e65100;}"
            ".score-nascent{background:#ffcdd2;color:#c62828;}"
            ".dim-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".dim-card{border:1px solid #c8e6c9;border-radius:10px;padding:20px;"
            "border-top:4px solid #43a047;}"
            ".dim-card-title{font-weight:600;color:#1b5e20;font-size:1.05em;}"
            ".dim-card-score{font-size:1.8em;font-weight:700;margin:8px 0;}"
            ".dim-card-level{font-size:0.85em;padding:3px 10px;border-radius:12px;"
            "display:inline-block;}"
            ".level-leading{background:#c8e6c9;color:#1b5e20;}"
            ".level-advanced{background:#dcedc8;color:#33691e;}"
            ".level-established{background:#fff9c4;color:#f57f17;}"
            ".level-developing{background:#ffe0b2;color:#e65100;}"
            ".level-nascent{background:#ffcdd2;color:#c62828;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:16px;overflow:hidden;"
            "margin:6px 0;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".fill-amber{background:linear-gradient(90deg,#ff8f00,#ffb300);}"
            ".fill-red{background:linear-gradient(90deg,#e53935,#ef5350);}"
            ".rec-card{border:1px solid #c8e6c9;border-radius:8px;padding:16px;"
            "margin:12px 0;border-left:4px solid #43a047;}"
            ".rec-high{border-left-color:#c62828;}"
            ".rec-medium{border-left-color:#ff8f00;}"
            ".rec-low{border-left-color:#43a047;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Net Zero Maturity Scorecard</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        overall = self._calc_overall(dim_scores)
        maturity = _get_maturity(overall)
        level_lower = maturity["level"].lower()
        score_cls = f"score-{level_lower}"
        return (
            f'<h2>1. Overall Score & Maturity Level</h2>\n'
            f'<div class="{score_cls} score-display">'
            f'{_dec(overall)} / 100 - {maturity["level"]}</div>\n'
            f'<p style="text-align:center;color:#666;">{maturity["description"]}</p>'
        )

    def _html_dimension_scores(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        cards = ""
        for dim in _DIMENSIONS:
            score = float(Decimal(str(dim_scores.get(dim["id"], {}).get("score", 0))))
            maturity = _get_maturity(score)
            level_lower = maturity["level"].lower()
            bar_color = "fill-green" if score >= 60 else "fill-amber" if score >= 40 else "fill-red"
            cards += (
                f'<div class="dim-card">'
                f'<div class="dim-card-title">{dim["name"]}</div>'
                f'<div class="dim-card-score" style="color:'
                f'{"#1b5e20" if score >= 60 else "#e65100" if score >= 40 else "#c62828"}">'
                f'{_dec(score)}</div>'
                f'<div class="progress-bar"><div class="progress-fill {bar_color}" '
                f'style="width:{score}%"></div></div>'
                f'<span class="dim-card-level level-{level_lower}">{maturity["level"]}</span>'
                f'</div>\n'
            )
        return (
            f'<h2>2. Dimension Scores</h2>\n'
            f'<div class="dim-grid">\n{cards}</div>'
        )

    def _html_dimension_details(self, data: Dict[str, Any]) -> str:
        dim_scores = data.get("dimension_scores", {})
        content = '<h2>3. Dimension Details</h2>\n'
        for dim in _DIMENSIONS:
            dim_data = dim_scores.get(dim["id"], {})
            score = float(Decimal(str(dim_data.get("score", 0))))
            maturity = _get_maturity(score)
            findings = dim_data.get("findings", [])
            strengths = dim_data.get("strengths", [])
            gaps = dim_data.get("gaps", [])

            findings_html = "".join(f"<li>{f}</li>" for f in findings)
            strengths_html = "".join(f"<li style='color:#1b5e20;'>{s}</li>" for s in strengths)
            gaps_html = "".join(f"<li style='color:#c62828;'>{g}</li>" for g in gaps)

            content += (
                f'<div style="margin:16px 0;padding:16px;border:1px solid #c8e6c9;'
                f'border-radius:8px;">'
                f'<h3>{dim["name"]} - {_dec(score)}/100 ({maturity["level"]})</h3>'
                f'<p style="color:#666;font-style:italic;">{dim["description"]}</p>'
                + (f'<strong>Key Findings:</strong><ul>{findings_html}</ul>' if findings else '')
                + (f'<strong>Strengths:</strong><ul>{strengths_html}</ul>' if strengths else '')
                + (f'<strong>Gaps:</strong><ul>{gaps_html}</ul>' if gaps else '')
                + '</div>\n'
            )
        return content

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "MEDIUM").lower()
            items += (
                f'<div class="rec-card rec-{priority}">'
                f'<strong>{i}. [{rec.get("priority", "MEDIUM")}] {rec.get("title", "")}</strong><br>'
                f'<small>Dimension: {rec.get("dimension", "-")} | '
                f'Impact: {rec.get("impact", "-")} | '
                f'Effort: {rec.get("effort", "-")}</small>'
                f'<p>{rec.get("description", "")}</p>'
                f'</div>\n'
            )
        return f'<h2>4. Priority Recommendations</h2>\n{items}'

    def _html_improvement_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("improvement_roadmap", [])
        rows = ""
        for i, item in enumerate(roadmap, 1):
            rows += (
                f'<tr><td>{i}</td><td>{item.get("action", "-")}</td>'
                f'<td>{item.get("dimension", "-")}</td>'
                f'<td>{_dec(item.get("target_score", 0))}</td>'
                f'<td>{item.get("timeline", "-")}</td>'
                f'<td>{item.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Improvement Roadmap</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Dimension</th>'
            f'<th>Target Score</th><th>Timeline</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_peer_comparison(self, data: Dict[str, Any]) -> str:
        peer = data.get("peer_comparison", {})
        sector_avg = float(Decimal(str(peer.get("sector_avg_score", 0))))
        sector_leader = float(Decimal(str(peer.get("sector_leader_score", 0))))
        dim_scores = data.get("dimension_scores", {})
        overall = self._calc_overall(dim_scores)
        percentile = peer.get("percentile_rank", 0)
        return (
            f'<h2>6. Peer Comparison Context</h2>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Your Score</td><td><strong>{_dec(overall)}</strong></td></tr>\n'
            f'<tr><td>Sector Average</td><td>{_dec(sector_avg)}</td></tr>\n'
            f'<tr><td>Sector Leader</td><td>{_dec(sector_leader)}</td></tr>\n'
            f'<tr><td>Percentile Rank</td><td>P{_dec(percentile, 0)}</td></tr>\n'
            f'<tr><td>Gap to Average</td><td>{_dec(overall - sector_avg)}</td></tr>\n'
            f'<tr><td>Gap to Leader</td><td>{_dec(overall - sector_leader)}</td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_overall(self, dim_scores: Dict[str, Any]) -> float:
        scores = []
        for dim in _DIMENSIONS:
            s = dim_scores.get(dim["id"], {}).get("score", 0)
            scores.append(float(Decimal(str(s))))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
