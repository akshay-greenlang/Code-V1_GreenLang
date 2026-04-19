# -*- coding: utf-8 -*-
"""
BenchmarkingReport - Peer Comparison and Rankings for PACK-044.

Generates a benchmarking report covering peer comparison of emissions
intensity, sector rankings, best practice identification, and
improvement opportunities.

Sections:
    1. Benchmarking Overview
    2. Intensity Metrics Comparison
    3. Sector Rankings
    4. Best Practices
    5. Improvement Opportunities

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class BenchmarkingReport:
    """
    Benchmarking report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BenchmarkingReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render benchmarking report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_intensity_comparison(data),
            self._md_rankings(data),
            self._md_best_practices(data),
            self._md_opportunities(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render benchmarking report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_intensity_comparison(data),
            self._html_rankings(data),
            self._html_best_practices(data),
            self._html_opportunities(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render benchmarking report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "benchmarking_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "sector": self._get_val(data, "sector", ""),
            "overview": data.get("overview", {}),
            "intensity_comparisons": data.get("intensity_comparisons", []),
            "rankings": data.get("rankings", []),
            "best_practices": data.get("best_practices", []),
            "opportunities": data.get("opportunities", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        sector = self._get_val(data, "sector", "")
        return (
            f"# Benchmarking Report - {company}\n\n"
            f"**Sector:** {sector} | **Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("overview", {})
        if not overview:
            return "## 1. Benchmarking Overview\n\nNo benchmark data available."
        return (
            "## 1. Benchmarking Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Peer Group Size | {overview.get('peer_group_size', 0)} |\n"
            f"| Sector | {overview.get('sector', '-')} |\n"
            f"| Data Year | {overview.get('data_year', '-')} |\n"
            f"| Percentile Rank | {overview.get('percentile_rank', 0):.0f}th |\n"
            f"| Overall Rating | **{overview.get('overall_rating', '-')}** |"
        )

    def _md_intensity_comparison(self, data: Dict[str, Any]) -> str:
        comps = data.get("intensity_comparisons", [])
        if not comps:
            return ""
        lines = [
            "## 2. Intensity Metrics Comparison", "",
            "| Metric | Your Value | Peer Median | Peer Best | Percentile | Unit |",
            "|--------|-----------|------------|----------|-----------|------|",
        ]
        for c in comps:
            lines.append(
                f"| {c.get('metric', '')} | {c.get('your_value', 0):.2f} | "
                f"{c.get('peer_median', 0):.2f} | {c.get('peer_best', 0):.2f} | "
                f"{c.get('percentile', 0):.0f}th | {c.get('unit', '')} |"
            )
        return "\n".join(lines)

    def _md_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("rankings", [])
        if not rankings:
            return ""
        lines = [
            "## 3. Sector Rankings", "",
            "| Category | Your Rank | Total Peers | Quartile | Trend |",
            "|---------|----------|------------|---------|-------|",
        ]
        for r in rankings:
            lines.append(
                f"| {r.get('category', '')} | {r.get('rank', '-')} | "
                f"{r.get('total_peers', '-')} | Q{r.get('quartile', '-')} | {r.get('trend', '-')} |"
            )
        return "\n".join(lines)

    def _md_best_practices(self, data: Dict[str, Any]) -> str:
        practices = data.get("best_practices", [])
        if not practices:
            return ""
        lines = ["## 4. Best Practices from Top Performers", ""]
        for p in practices:
            lines.append(
                f"- **{p.get('practice', '')}** ({p.get('category', '')})\n"
                f"  - Adopted by: {p.get('adoption_pct', 0):.0f}% of top quartile\n"
                f"  - Typical Impact: {p.get('typical_impact', '-')}"
            )
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        if not opps:
            return ""
        lines = [
            "## 5. Improvement Opportunities", "",
            "| Opportunity | Category | Gap to Median | Gap to Best | Priority |",
            "|-----------|----------|-------------|-----------|----------|",
        ]
        for o in opps:
            lines.append(
                f"| {o.get('opportunity', '')} | {o.get('category', '')} | "
                f"{o.get('gap_to_median', '-')} | {o.get('gap_to_best', '-')} | "
                f"**{o.get('priority', 'medium')}** |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return f"---\n\n*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n*Provenance Hash: `{provenance}`*"

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Benchmarking - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #e9c46a;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Benchmarking &mdash; {company}</h1><hr></div>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("overview", {})
        if not overview:
            return ""
        return f'<div><h2>1. Overview</h2><p>Peers: {overview.get("peer_group_size", 0)} | Rank: {overview.get("percentile_rank", 0):.0f}th percentile</p></div>'

    def _html_intensity_comparison(self, data: Dict[str, Any]) -> str:
        comps = data.get("intensity_comparisons", [])
        if not comps:
            return ""
        rows = ""
        for c in comps:
            rows += f"<tr><td>{c.get('metric', '')}</td><td>{c.get('your_value', 0):.2f}</td><td>{c.get('peer_median', 0):.2f}</td><td>{c.get('peer_best', 0):.2f}</td><td>{c.get('unit', '')}</td></tr>\n"
        return '<div><h2>2. Intensity Comparison</h2>\n<table><thead><tr><th>Metric</th><th>You</th><th>Median</th><th>Best</th><th>Unit</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("rankings", [])
        if not rankings:
            return ""
        rows = ""
        for r in rankings:
            rows += f"<tr><td>{r.get('category', '')}</td><td>{r.get('rank', '-')}</td><td>{r.get('total_peers', '-')}</td><td>Q{r.get('quartile', '-')}</td></tr>\n"
        return '<div><h2>3. Rankings</h2>\n<table><thead><tr><th>Category</th><th>Rank</th><th>Peers</th><th>Quartile</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_best_practices(self, data: Dict[str, Any]) -> str:
        practices = data.get("best_practices", [])
        if not practices:
            return ""
        li = ""
        for p in practices:
            li += f"<li><strong>{p.get('practice', '')}</strong> - {p.get('typical_impact', '')}</li>\n"
        return f'<div><h2>4. Best Practices</h2><ul>{li}</ul></div>'

    def _html_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        if not opps:
            return ""
        rows = ""
        for o in opps:
            rows += f"<tr><td>{o.get('opportunity', '')}</td><td>{o.get('category', '')}</td><td>{o.get('gap_to_median', '-')}</td><td><strong>{o.get('priority', 'medium')}</strong></td></tr>\n"
        return '<div><h2>5. Opportunities</h2>\n<table><thead><tr><th>Opportunity</th><th>Category</th><th>Gap to Median</th><th>Priority</th></tr></thead>\n<tbody>' + rows + '</tbody></table></div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return f'<div style="font-size:0.85rem;color:#666;"><hr><p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p><p class="provenance">Provenance Hash: {provenance}</p></div>'
