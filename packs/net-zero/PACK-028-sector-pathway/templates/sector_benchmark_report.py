# -*- coding: utf-8 -*-
"""
SectorBenchmarkReportTemplate - Multi-dimensional sector benchmarking for PACK-028.

Renders a comprehensive sector benchmarking dashboard with peer/leader/IEA
comparisons, percentile rankings, gap analysis, and multi-dimensional
scoring. Multi-format (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Benchmark Dimensions
    3.  Peer Comparison Table
    4.  Percentile Rankings
    5.  Gap-to-Leader Analysis
    6.  IEA Pathway Benchmark
    7.  SBTi-Validated Peer Comparison
    8.  Regional Benchmarking
    9.  Trend Analysis (Multi-Year)
    10. Improvement Opportunities
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "sector_benchmark_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_WARN = "#ef6c00"
_DANGER = "#c62828"
_SUCCESS = "#2e7d32"

BENCHMARK_DIMENSIONS: List[Dict[str, str]] = [
    {"id": "intensity", "name": "Emission Intensity", "description": "Scope 1+2 intensity per unit of output"},
    {"id": "reduction_rate", "name": "Annual Reduction Rate", "description": "Year-over-year intensity reduction"},
    {"id": "renewable_share", "name": "Renewable Energy Share", "description": "Percentage of electricity from renewables"},
    {"id": "target_ambition", "name": "Target Ambition Level", "description": "SBTi pathway alignment (1.5C/WB2C/2C)"},
    {"id": "scope3_coverage", "name": "Scope 3 Coverage", "description": "Percentage of Scope 3 categories reported"},
    {"id": "technology_adoption", "name": "Technology Adoption", "description": "Deployment of key decarbonization technologies"},
    {"id": "data_quality", "name": "Data Quality Score", "description": "Quality of emissions and activity data"},
    {"id": "disclosure", "name": "Disclosure Level", "description": "CDP, TCFD, CSRD reporting completeness"},
]

XBRL_BENCHMARK_TAGS: Dict[str, str] = {
    "overall_percentile": "gl:SectorBenchmarkOverallPercentile",
    "intensity_percentile": "gl:IntensityPercentileRanking",
    "gap_to_leader_pct": "gl:GapToSectorLeaderPercentage",
    "peer_count": "gl:BenchmarkPeerCount",
    "sbti_peers": "gl:SBTiValidatedPeerCount",
    "iea_alignment_pct": "gl:IEAPathwayAlignmentPercentage",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        neg = int_part.startswith("-")
        if neg:
            int_part = int_part[1:]
        fmt = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                fmt = "," + fmt
            fmt = ch + fmt
        if neg:
            fmt = "-" + fmt
        if len(parts) > 1:
            fmt += "." + parts[1]
        return fmt
    except Exception:
        return str(val)


def _quartile(percentile: float) -> str:
    if percentile >= 75:
        return "Q1 (Top 25%)"
    elif percentile >= 50:
        return "Q2 (25-50%)"
    elif percentile >= 25:
        return "Q3 (50-75%)"
    return "Q4 (Bottom 25%)"


def _quartile_color(percentile: float) -> str:
    if percentile >= 75:
        return _SUCCESS
    elif percentile >= 50:
        return _ACCENT
    elif percentile >= 25:
        return _WARN
    return _DANGER


class SectorBenchmarkReportTemplate:
    """
    Multi-dimensional sector benchmarking report template.

    Renders peer/leader/IEA comparison, percentile rankings, gap analysis,
    and improvement opportunities. Supports MD, HTML, JSON, and PDF.

    Example:
        >>> template = SectorBenchmarkReportTemplate()
        >>> data = {
        ...     "org_name": "SteelCo",
        ...     "sector_id": "steel",
        ...     "your_metrics": {"intensity": 1.72, ...},
        ...     "peers": [...],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_dimensions(data),
            self._md_peer_comparison(data),
            self._md_percentile_rankings(data),
            self._md_gap_to_leader(data),
            self._md_iea_benchmark(data),
            self._md_sbti_peers(data),
            self._md_regional(data),
            self._md_trend_analysis(data),
            self._md_opportunities(data),
            self._md_xbrl_tags(data),
            self._md_audit_trail(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_dimensions(data),
            self._html_peer_comparison(data),
            self._html_percentile_rankings(data),
            self._html_gap_to_leader(data),
            self._html_iea_benchmark(data),
            self._html_sbti_peers(data),
            self._html_regional(data),
            self._html_trend_analysis(data),
            self._html_opportunities(data),
            self._html_xbrl_tags(data),
            self._html_audit_trail(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Sector Benchmark - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        your_metrics = data.get("your_metrics", {})
        peers = data.get("peers", [])
        rankings = data.get("percentile_rankings", {})
        overall = float(rankings.get("overall", 50))
        leader = data.get("sector_leader", {})

        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector_id": data.get("sector_id", ""),
            "summary": {
                "overall_percentile": str(overall),
                "quartile": _quartile(overall),
                "peer_count": len(peers),
                "dimensions_assessed": len(BENCHMARK_DIMENSIONS),
            },
            "your_metrics": your_metrics,
            "sector_average": data.get("sector_average", {}),
            "sector_leader": leader,
            "peers": peers,
            "percentile_rankings": rankings,
            "iea_benchmark": data.get("iea_benchmark", {}),
            "regional_benchmarks": data.get("regional_benchmarks", []),
            "trend": data.get("trend", []),
            "opportunities": data.get("opportunities", []),
            "xbrl_tags": {k: XBRL_BENCHMARK_TAGS[k] for k in XBRL_BENCHMARK_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Sector Benchmark - {data.get('org_name', '')}",
                "author": "GreenLang PACK-028",
                "subject": "Multi-Dimensional Sector Benchmarking",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Sector Benchmark Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {data.get('sector_id', '').replace('_', ' ').title()}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        overall = float(rankings.get("overall", 50))
        peers = data.get("peers", [])
        your_metrics = data.get("your_metrics", {})
        avg = data.get("sector_average", {})
        leader = data.get("sector_leader", {})
        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |",
            f"|-----|-------|",
            f"| Overall Percentile | **{_dec(overall, 0)}th** ({_quartile(overall)}) |",
            f"| Peer Group Size | {len(peers)} companies |",
            f"| Your Intensity | {_dec(your_metrics.get('intensity', 0), 4)} |",
            f"| Sector Average | {_dec(avg.get('intensity', 0), 4)} |",
            f"| Sector Leader | {_dec(leader.get('intensity', 0), 4)} |",
            f"| Gap to Average | {_dec(float(your_metrics.get('intensity', 0)) - float(avg.get('intensity', 0)), 4)} |",
            f"| Gap to Leader | {_dec(float(your_metrics.get('intensity', 0)) - float(leader.get('intensity', 0)), 4)} |",
            f"| Dimensions Assessed | {len(BENCHMARK_DIMENSIONS)} |",
        ]
        return "\n".join(lines)

    def _md_dimensions(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        lines = [
            "## 2. Benchmark Dimensions\n",
            "| # | Dimension | Your Percentile | Quartile | Description |",
            "|---|-----------|----------------:|----------|-------------|",
        ]
        for i, dim in enumerate(BENCHMARK_DIMENSIONS, 1):
            pct = float(rankings.get(dim["id"], 50))
            lines.append(
                f"| {i} | {dim['name']} | {_dec(pct, 0)}th | {_quartile(pct)} | {dim['description']} |"
            )
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        peers = data.get("peers", [])
        your_metrics = data.get("your_metrics", {})
        your_int = float(your_metrics.get("intensity", 0))
        lines = [
            "## 3. Peer Comparison\n",
            "| Rank | Company | Intensity | Gap vs. You | SBTi | Region |",
            "|-----:|---------|----------:|------------:|:----:|--------|",
        ]
        sorted_peers = sorted(peers, key=lambda p: float(p.get("intensity", 999)))
        for i, p in enumerate(sorted_peers, 1):
            p_int = float(p.get("intensity", 0))
            gap = your_int - p_int
            sbti = "Yes" if p.get("sbti_committed", False) else "No"
            lines.append(
                f"| {i} | {p.get('name', '')} | {_dec(p_int, 4)} "
                f"| {'+' if gap > 0 else ''}{_dec(gap, 4)} | {sbti} | {p.get('region', '')} |"
            )
        if not peers:
            lines.append("| - | _No peer data available_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_percentile_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        lines = [
            "## 4. Percentile Rankings\n",
            "| Dimension | Percentile | Quartile | Interpretation |",
            "|-----------|----------:|----------|---------------|",
        ]
        for dim in BENCHMARK_DIMENSIONS:
            pct = float(rankings.get(dim["id"], 50))
            q = _quartile(pct)
            interp = (
                "Top performer" if pct >= 75 else
                "Above average" if pct >= 50 else
                "Below average" if pct >= 25 else
                "Needs improvement"
            )
            lines.append(f"| {dim['name']} | {_dec(pct, 0)}th | {q} | {interp} |")
        overall = float(rankings.get("overall", 50))
        lines.append(f"| **Overall** | **{_dec(overall, 0)}th** | **{_quartile(overall)}** | |")
        return "\n".join(lines)

    def _md_gap_to_leader(self, data: Dict[str, Any]) -> str:
        your_metrics = data.get("your_metrics", {})
        leader = data.get("sector_leader", {})
        lines = [
            "## 5. Gap-to-Leader Analysis\n",
            f"**Sector Leader:** {leader.get('name', 'Best-in-class')}\n",
            "| Dimension | Your Value | Leader Value | Gap | Gap (%) |",
            "|-----------|-----------|-------------|-----|---------|",
        ]
        for dim in BENCHMARK_DIMENSIONS:
            your_val = your_metrics.get(dim["id"], 0)
            leader_val = leader.get(dim["id"], 0)
            if isinstance(your_val, (int, float)) and isinstance(leader_val, (int, float)):
                gap = float(your_val) - float(leader_val)
                gap_pct = ((gap / float(leader_val)) * 100) if float(leader_val) != 0 else 0
                lines.append(
                    f"| {dim['name']} | {_dec(your_val, 4)} | {_dec(leader_val, 4)} "
                    f"| {'+' if gap > 0 else ''}{_dec(gap, 4)} | {_dec(gap_pct)}% |"
                )
            else:
                lines.append(
                    f"| {dim['name']} | {your_val} | {leader_val} | - | - |"
                )
        return "\n".join(lines)

    def _md_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        milestones = iea.get("milestones", [])
        lines = [
            "## 6. IEA Pathway Benchmark\n",
            f"**Your Intensity vs. IEA Milestones:**\n",
            "| Year | IEA Target | Your Projected | Gap | Status |",
            "|------|-----------|----------------|-----|--------|",
        ]
        for m in milestones:
            iea_val = float(m.get("iea_target", 0))
            your_val = float(m.get("your_projected", 0))
            gap = your_val - iea_val
            gap_pct = ((gap / iea_val) * 100) if iea_val > 0 else 0
            status = "ON TRACK" if gap_pct <= 5 else ("CLOSE" if gap_pct <= 15 else "OFF TRACK")
            lines.append(
                f"| {m.get('year', '')} | {_dec(iea_val, 4)} | {_dec(your_val, 4)} "
                f"| {_dec(gap_pct)}% | {status} |"
            )
        if not milestones:
            lines.append("| - | _IEA benchmark data not mapped_ | - | - | - |")
        return "\n".join(lines)

    def _md_sbti_peers(self, data: Dict[str, Any]) -> str:
        peers = data.get("peers", [])
        sbti_peers = [p for p in peers if p.get("sbti_committed", False)]
        your_int = float(data.get("your_metrics", {}).get("intensity", 0))
        lines = [
            "## 7. SBTi-Validated Peer Comparison\n",
            f"**SBTi-committed peers:** {len(sbti_peers)} / {len(peers)}\n",
        ]
        if sbti_peers:
            avg_sbti = sum(float(p.get("intensity", 0)) for p in sbti_peers) / len(sbti_peers)
            lines.append(f"**Average SBTi peer intensity:** {_dec(avg_sbti, 4)}")
            lines.append(f"**Your intensity vs SBTi average:** {'+' if your_int > avg_sbti else ''}{_dec(your_int - avg_sbti, 4)}\n")
            lines.append("| Company | Intensity | Target Type | Target Year | Ambition |")
            lines.append("|---------|----------:|------------|:-----------:|---------|")
            for p in sbti_peers:
                lines.append(
                    f"| {p.get('name', '')} | {_dec(float(p.get('intensity', 0)), 4)} "
                    f"| {p.get('target_type', 'SDA')} | {p.get('target_year', '')} "
                    f"| {p.get('ambition', '1.5C')} |"
                )
        else:
            lines.append("_No SBTi-committed peers in dataset._")
        return "\n".join(lines)

    def _md_regional(self, data: Dict[str, Any]) -> str:
        regional = data.get("regional_benchmarks", [])
        lines = [
            "## 8. Regional Benchmarking\n",
            "| Region | Average Intensity | Best-in-Class | Your Position | Regulatory Context |",
            "|--------|------------------:|--------------:|:-------------|-------------------|",
        ]
        for r in regional:
            lines.append(
                f"| {r.get('region', '')} | {_dec(r.get('average', 0), 4)} "
                f"| {_dec(r.get('best', 0), 4)} | {r.get('your_position', '')} "
                f"| {r.get('regulatory', '')} |"
            )
        if not regional:
            lines.append("| _No regional data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        trend = data.get("trend", [])
        lines = [
            "## 9. Multi-Year Trend\n",
            "| Year | Your Intensity | Sector Average | Percentile | Trend |",
            "|------|---------------:|--------------:|-----------|-------|",
        ]
        prev = None
        for t in trend:
            your_val = float(t.get("your_intensity", 0))
            trend_dir = ""
            if prev is not None:
                if your_val < prev:
                    trend_dir = "Improving"
                elif your_val > prev:
                    trend_dir = "Worsening"
                else:
                    trend_dir = "Stable"
            lines.append(
                f"| {t.get('year', '')} | {_dec(your_val, 4)} "
                f"| {_dec(t.get('sector_average', 0), 4)} "
                f"| {_dec(t.get('percentile', 50), 0)}th | {trend_dir} |"
            )
            prev = your_val
        if not trend:
            lines.append("| - | _Trend data not available_ | - | - | - |")
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        opportunities = data.get("opportunities", [])
        lines = [
            "## 10. Improvement Opportunities\n",
        ]
        if opportunities:
            lines.append("| Priority | Dimension | Current | Target | Action | Impact |")
            lines.append("|----------|-----------|---------|--------|--------|--------|")
            for i, o in enumerate(opportunities, 1):
                lines.append(
                    f"| {i} | {o.get('dimension', '')} | {o.get('current', '')} "
                    f"| {o.get('target', '')} | {o.get('action', '')} | {o.get('impact', '')} |"
                )
        else:
            lines.append("_Opportunities will be identified based on gap analysis results._")
        return "\n".join(lines)

    def _md_xbrl_tags(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        peers = data.get("peers", [])
        sbti_count = sum(1 for p in peers if p.get("sbti_committed", False))
        overall = float(rankings.get("overall", 50))
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
            f"| Overall Percentile | {XBRL_BENCHMARK_TAGS['overall_percentile']} | {_dec(overall, 0)}th |",
            f"| Intensity Percentile | {XBRL_BENCHMARK_TAGS['intensity_percentile']} | {_dec(float(rankings.get('intensity', 50)), 0)}th |",
            f"| Peer Count | {XBRL_BENCHMARK_TAGS['peer_count']} | {len(peers)} |",
            f"| SBTi Peers | {XBRL_BENCHMARK_TAGS['sbti_peers']} | {sbti_count} |",
        ]
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n*Multi-dimensional sector benchmarking dashboard.*"

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".pct-bar{{height:16px;background:#e0e0e0;border-radius:8px;overflow:hidden;}}"
            f".pct-fill{{height:16px;border-radius:8px;}}"
            f".q1{{background:{_SUCCESS};}}.q2{{background:{_ACCENT};}}.q3{{background:{_WARN};}}.q4{{background:{_DANGER};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Sector Benchmark Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name", "")} | <strong>Sector:</strong> {data.get("sector_id", "").replace("_", " ").title()} | <strong>Generated:</strong> {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        overall = float(rankings.get("overall", 50))
        peers = data.get("peers", [])
        your_int = float(data.get("your_metrics", {}).get("intensity", 0))
        avg_int = float(data.get("sector_average", {}).get("intensity", 0))
        leader_int = float(data.get("sector_leader", {}).get("intensity", 0))
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Overall Percentile</div><div class="card-value">{_dec(overall, 0)}th</div><div class="card-unit">{_quartile(overall)}</div></div>\n'
            f'<div class="card"><div class="card-label">Your Intensity</div><div class="card-value">{_dec(your_int, 4)}</div></div>\n'
            f'<div class="card"><div class="card-label">Sector Average</div><div class="card-value">{_dec(avg_int, 4)}</div></div>\n'
            f'<div class="card"><div class="card-label">Sector Leader</div><div class="card-value">{_dec(leader_int, 4)}</div></div>\n'
            f'<div class="card"><div class="card-label">Peers</div><div class="card-value">{len(peers)}</div></div>\n'
            f'</div>'
        )

    def _html_dimensions(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        rows = ""
        for i, dim in enumerate(BENCHMARK_DIMENSIONS, 1):
            pct = float(rankings.get(dim["id"], 50))
            q_cls = "q1" if pct >= 75 else ("q2" if pct >= 50 else ("q3" if pct >= 25 else "q4"))
            rows += (
                f'<tr><td>{i}</td><td>{dim["name"]}</td>'
                f'<td><div class="pct-bar"><div class="pct-fill {q_cls}" style="width:{pct}%"></div></div> {_dec(pct, 0)}th</td>'
                f'<td>{_quartile(pct)}</td></tr>\n'
            )
        return f'<h2>2. Benchmark Dimensions</h2>\n<table>\n<tr><th>#</th><th>Dimension</th><th>Percentile</th><th>Quartile</th></tr>\n{rows}</table>'

    def _html_peer_comparison(self, data: Dict[str, Any]) -> str:
        peers = data.get("peers", [])
        your_int = float(data.get("your_metrics", {}).get("intensity", 0))
        sorted_peers = sorted(peers, key=lambda p: float(p.get("intensity", 999)))
        rows = ""
        for i, p in enumerate(sorted_peers, 1):
            p_int = float(p.get("intensity", 0))
            gap = your_int - p_int
            rows += (
                f'<tr><td>{i}</td><td>{p.get("name", "")}</td><td>{_dec(p_int, 4)}</td>'
                f'<td>{"+" if gap > 0 else ""}{_dec(gap, 4)}</td>'
                f'<td>{"Yes" if p.get("sbti_committed") else "No"}</td><td>{p.get("region", "")}</td></tr>\n'
            )
        return f'<h2>3. Peer Comparison</h2>\n<table>\n<tr><th>#</th><th>Company</th><th>Intensity</th><th>Gap</th><th>SBTi</th><th>Region</th></tr>\n{rows}</table>'

    def _html_percentile_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        rows = ""
        for dim in BENCHMARK_DIMENSIONS:
            pct = float(rankings.get(dim["id"], 50))
            q_cls = "q1" if pct >= 75 else ("q2" if pct >= 50 else ("q3" if pct >= 25 else "q4"))
            rows += (
                f'<tr><td>{dim["name"]}</td>'
                f'<td><div class="pct-bar"><div class="pct-fill {q_cls}" style="width:{pct}%"></div></div></td>'
                f'<td>{_dec(pct, 0)}th</td><td>{_quartile(pct)}</td></tr>\n'
            )
        return f'<h2>4. Percentile Rankings</h2>\n<table>\n<tr><th>Dimension</th><th>Bar</th><th>Percentile</th><th>Quartile</th></tr>\n{rows}</table>'

    def _html_gap_to_leader(self, data: Dict[str, Any]) -> str:
        your_metrics = data.get("your_metrics", {})
        leader = data.get("sector_leader", {})
        rows = ""
        for dim in BENCHMARK_DIMENSIONS:
            yv = your_metrics.get(dim["id"], 0)
            lv = leader.get(dim["id"], 0)
            if isinstance(yv, (int, float)) and isinstance(lv, (int, float)):
                gap = float(yv) - float(lv)
                rows += f'<tr><td>{dim["name"]}</td><td>{_dec(yv, 4)}</td><td>{_dec(lv, 4)}</td><td>{"+" if gap > 0 else ""}{_dec(gap, 4)}</td></tr>\n'
            else:
                rows += f'<tr><td>{dim["name"]}</td><td>{yv}</td><td>{lv}</td><td>-</td></tr>\n'
        return f'<h2>5. Gap to Leader</h2>\n<table>\n<tr><th>Dimension</th><th>You</th><th>Leader</th><th>Gap</th></tr>\n{rows}</table>'

    def _html_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        milestones = iea.get("milestones", [])
        rows = ""
        for m in milestones:
            iea_val = float(m.get("iea_target", 0))
            your_val = float(m.get("your_projected", 0))
            gap_pct = ((your_val - iea_val) / iea_val * 100) if iea_val > 0 else 0
            status = "ON TRACK" if gap_pct <= 5 else ("CLOSE" if gap_pct <= 15 else "OFF TRACK")
            rows += f'<tr><td>{m.get("year", "")}</td><td>{_dec(iea_val, 4)}</td><td>{_dec(your_val, 4)}</td><td>{_dec(gap_pct)}%</td><td>{status}</td></tr>\n'
        return f'<h2>6. IEA Benchmark</h2>\n<table>\n<tr><th>Year</th><th>IEA</th><th>Projected</th><th>Gap</th><th>Status</th></tr>\n{rows}</table>'

    def _html_sbti_peers(self, data: Dict[str, Any]) -> str:
        peers = data.get("peers", [])
        sbti_peers = [p for p in peers if p.get("sbti_committed", False)]
        rows = ""
        for p in sbti_peers:
            rows += f'<tr><td>{p.get("name", "")}</td><td>{_dec(float(p.get("intensity", 0)), 4)}</td><td>{p.get("target_type", "SDA")}</td><td>{p.get("ambition", "1.5C")}</td></tr>\n'
        return f'<h2>7. SBTi Peers</h2>\n<p>{len(sbti_peers)} SBTi-committed peers</p>\n<table>\n<tr><th>Company</th><th>Intensity</th><th>Target Type</th><th>Ambition</th></tr>\n{rows}</table>'

    def _html_regional(self, data: Dict[str, Any]) -> str:
        regional = data.get("regional_benchmarks", [])
        rows = "".join(f'<tr><td>{r.get("region","")}</td><td>{_dec(r.get("average",0),4)}</td><td>{_dec(r.get("best",0),4)}</td><td>{r.get("your_position","")}</td></tr>\n' for r in regional)
        return f'<h2>8. Regional Benchmarking</h2>\n<table>\n<tr><th>Region</th><th>Average</th><th>Best</th><th>Position</th></tr>\n{rows}</table>'

    def _html_trend_analysis(self, data: Dict[str, Any]) -> str:
        trend = data.get("trend", [])
        rows = ""
        for t in trend:
            rows += f'<tr><td>{t.get("year","")}</td><td>{_dec(t.get("your_intensity",0),4)}</td><td>{_dec(t.get("sector_average",0),4)}</td><td>{_dec(t.get("percentile",50),0)}th</td></tr>\n'
        return f'<h2>9. Multi-Year Trend</h2>\n<table>\n<tr><th>Year</th><th>You</th><th>Sector Avg</th><th>Percentile</th></tr>\n{rows}</table>'

    def _html_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        rows = "".join(f'<tr><td>{i}</td><td>{o.get("dimension","")}</td><td>{o.get("action","")}</td><td>{o.get("impact","")}</td></tr>\n' for i, o in enumerate(opps, 1))
        return f'<h2>10. Opportunities</h2>\n<table>\n<tr><th>#</th><th>Dimension</th><th>Action</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_xbrl_tags(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        overall = float(rankings.get("overall", 50))
        peers = data.get("peers", [])
        return (
            f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n'
            f'<tr><td>Overall</td><td><code>{XBRL_BENCHMARK_TAGS["overall_percentile"]}</code></td><td>{_dec(overall, 0)}th</td></tr>\n'
            f'<tr><td>Peers</td><td><code>{XBRL_BENCHMARK_TAGS["peer_count"]}</code></td><td>{len(peers)}</td></tr>\n'
            f'</table>'
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-028 Sector Pathway Pack on {ts} - Sector benchmarking dashboard</div>'
