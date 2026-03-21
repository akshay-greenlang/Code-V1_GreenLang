# -*- coding: utf-8 -*-
"""
TemperatureRatingReportTemplate - Temperature scoring report for PACK-023.

Renders a comprehensive temperature rating report covering company and portfolio
temperature scores, implied temperature by scope and timeframe, six aggregation
methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS), contribution analysis by entity,
sector comparison benchmarks, and trajectory visualization against 1.5C and
2C pathways.

Sections:
    1. Temperature Score Summary (company-level scores)
    2. Score by Scope & Timeframe (S1+S2, S3, S1+S2+S3 x short/mid/long)
    3. Aggregation Methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS)
    4. Contribution Analysis (entity-level breakdown)
    5. Sector Comparison (company vs sector benchmarks)
    6. Trajectory Visualization (1.5C / 2C pathway alignment)
    7. Score Improvement Opportunities

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

# SBTi Temperature Rating aggregation method descriptions
AGGREGATION_METHODS = {
    "WATS": "Weighted Average Temperature Score",
    "TETS": "Total Emissions Weighted Temperature Score",
    "MOTS": "Market Owned Temperature Score",
    "EOTS": "Enterprise Owned Temperature Score",
    "ECOTS": "Enterprise Value + Cash Temperature Score",
    "AOTS": "Revenue Allocated Temperature Score",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


def _temp_label(temp: float) -> str:
    """Return alignment label based on temperature score."""
    if temp <= 1.5:
        return "1.5C aligned"
    elif temp <= 1.75:
        return "Below 2C"
    elif temp <= 2.0:
        return "2C aligned"
    elif temp <= 2.5:
        return "Above 2C"
    elif temp <= 3.2:
        return "Well above 2C"
    else:
        return "Strongly misaligned"


def _temp_color(temp: float) -> str:
    """Return CSS color class based on temperature score."""
    if temp <= 1.5:
        return "temp-15"
    elif temp <= 2.0:
        return "temp-20"
    elif temp <= 2.5:
        return "temp-25"
    elif temp <= 3.2:
        return "temp-32"
    else:
        return "temp-high"


class TemperatureRatingReportTemplate:
    """
    Temperature rating report template for SBTi alignment.

    Renders company and portfolio temperature scores using the SBTi
    Temperature Rating methodology with six aggregation methods
    (WATS/TETS/MOTS/EOTS/ECOTS/AOTS), contribution analysis, sector
    comparison, and trajectory visualization against 1.5C and 2C pathways.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TemperatureRatingReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render temperature rating report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_score_summary(data),
            self._md_scope_timeframe(data),
            self._md_aggregation_methods(data),
            self._md_contribution_analysis(data),
            self._md_sector_comparison(data),
            self._md_trajectory_visualization(data),
            self._md_improvement_opportunities(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render temperature rating report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_score_summary(data),
            self._html_scope_timeframe(data),
            self._html_aggregation_methods(data),
            self._html_contribution_analysis(data),
            self._html_sector_comparison(data),
            self._html_trajectory_visualization(data),
            self._html_improvement_opportunities(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Temperature Rating Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render temperature rating report as structured JSON."""
        self.generated_at = _utcnow()
        scores = data.get("scores", {})
        scope_timeframe = data.get("scope_timeframe", [])
        aggregation = data.get("aggregation_methods", [])
        contributions = data.get("contributions", [])
        sectors = data.get("sector_comparison", [])
        trajectory = data.get("trajectory", [])

        result: Dict[str, Any] = {
            "template": "temperature_rating_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "score_summary": {
                "company_score": scores.get("company_score", 0),
                "portfolio_score": scores.get("portfolio_score", 0),
                "alignment": _temp_label(
                    float(scores.get("company_score", 0))
                ),
            },
            "scope_timeframe": scope_timeframe,
            "aggregation_methods": aggregation,
            "contributions": contributions,
            "sector_comparison": sectors,
            "trajectory": trajectory,
            "improvement_opportunities": data.get("improvement_opportunities", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Temperature Rating Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Methodology:** SBTi Temperature Rating Methodology V2.0\n\n---"
        )

    def _md_score_summary(self, data: Dict[str, Any]) -> str:
        scores = data.get("scores", {})
        company = float(scores.get("company_score", 0))
        portfolio = float(scores.get("portfolio_score", 0))
        target_status = scores.get("target_status", "N/A")
        target_ambition = scores.get("target_ambition", "N/A")

        return (
            f"## 1. Temperature Score Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Company Temperature Score | {_dec(company, 2)}C |\n"
            f"| Alignment | {_temp_label(company)} |\n"
            f"| Portfolio Temperature Score | {_dec(portfolio, 2)}C |\n"
            f"| Portfolio Alignment | {_temp_label(portfolio)} |\n"
            f"| Target Status | {target_status} |\n"
            f"| Target Ambition | {target_ambition} |\n"
            f"| Score Type | {scores.get('score_type', 'Implied Temperature Rise')} |\n"
            f"| Default Score | {_dec(scores.get('default_score', 3.2), 2)}C "
            f"(companies without targets) |\n"
            f"| 1.5C Threshold | {'MET' if company <= 1.5 else 'NOT MET'} |\n"
            f"| 2C Threshold | {'MET' if company <= 2.0 else 'NOT MET'} |"
        )

    def _md_scope_timeframe(self, data: Dict[str, Any]) -> str:
        st = data.get("scope_timeframe", [])
        lines = [
            "## 2. Score by Scope & Timeframe\n",
            "Temperature scores segmented by emission scope and time horizon.\n",
            "| Scope | Short-Term (C) | Mid-Term (C) | Long-Term (C) "
            "| Combined (C) | Alignment |",
            "|-------|:--------------:|:------------:|:-------------:"
            "|:------------:|-----------|",
        ]
        for s in st:
            combined = float(s.get("combined", 0))
            lines.append(
                f"| {s.get('scope', '-')} "
                f"| {_dec(s.get('short_term', 0), 2)} "
                f"| {_dec(s.get('mid_term', 0), 2)} "
                f"| {_dec(s.get('long_term', 0), 2)} "
                f"| {_dec(combined, 2)} "
                f"| {_temp_label(combined)} |"
            )
        if not st:
            lines.append("| - | _No scope data_ | - | - | - | - |")

        # Default rows if scope_timeframe not provided
        defaults = data.get("scope_defaults", {})
        if defaults:
            lines.append("")
            lines.append("### Scope Details\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| S1+S2 Coverage | {_pct(defaults.get('s1s2_coverage', 0))} |\n"
                f"| S3 Coverage | {_pct(defaults.get('s3_coverage', 0))} |\n"
                f"| Companies with Targets | {defaults.get('with_targets', 0)} |\n"
                f"| Companies without Targets | {defaults.get('without_targets', 0)} |\n"
                f"| Default Score Applied | {_dec(defaults.get('default_score', 3.2), 2)}C |"
            )

        return "\n".join(lines)

    def _md_aggregation_methods(self, data: Dict[str, Any]) -> str:
        methods = data.get("aggregation_methods", [])
        lines = [
            "## 3. Aggregation Methods\n",
            "Temperature scores computed using six SBTi-defined aggregation methods.\n",
            "| Method | Full Name | S1+S2 (C) | S3 (C) "
            "| Combined (C) | Alignment |",
            "|--------|-----------|:---------:|:------:"
            "|:------------:|-----------|",
        ]
        for m in methods:
            code = m.get("method", "-")
            full_name = AGGREGATION_METHODS.get(code, code)
            combined = float(m.get("combined", 0))
            lines.append(
                f"| {code} | {full_name} "
                f"| {_dec(m.get('s1s2', 0), 2)} "
                f"| {_dec(m.get('s3', 0), 2)} "
                f"| {_dec(combined, 2)} "
                f"| {_temp_label(combined)} |"
            )
        if not methods:
            lines.append("| - | _No methods computed_ | - | - | - | - |")

        # Method selection rationale
        selected = data.get("selected_method", {})
        if selected:
            lines.append("")
            lines.append(
                f"**Selected Method:** {selected.get('method', 'N/A')} "
                f"({AGGREGATION_METHODS.get(selected.get('method', ''), '')})  \n"
                f"**Rationale:** {selected.get('rationale', 'N/A')}"
            )

        return "\n".join(lines)

    def _md_contribution_analysis(self, data: Dict[str, Any]) -> str:
        contributions = data.get("contributions", [])
        lines = [
            "## 4. Contribution Analysis\n",
            "Entity-level contribution to portfolio temperature score.\n",
            "| # | Entity | Weight (%) | Score (C) | Contribution (C) "
            "| Target Status | Sector |",
            "|---|--------|:----------:|:---------:|:----------------:"
            "|:-------------:|--------|",
        ]
        for i, c in enumerate(contributions, 1):
            lines.append(
                f"| {i} | {c.get('entity', '-')} "
                f"| {_pct(c.get('weight_pct', 0))} "
                f"| {_dec(c.get('score', 0), 2)} "
                f"| {_dec(c.get('contribution', 0), 3)} "
                f"| {c.get('target_status', '-')} "
                f"| {c.get('sector', '-')} |"
            )
        if not contributions:
            lines.append(
                "| - | _No entity contributions_ | - | - | - | - | - |"
            )

        # Top contributors summary
        if contributions:
            sorted_c = sorted(
                contributions,
                key=lambda x: float(x.get("contribution", 0)),
                reverse=True,
            )
            top5 = sorted_c[:5]
            lines.append("")
            lines.append("### Top 5 Contributors to Portfolio Score\n")
            lines.append("| Entity | Contribution (C) | Impact |")
            lines.append("|--------|:----------------:|--------|")
            for c in top5:
                lines.append(
                    f"| {c.get('entity', '-')} "
                    f"| {_dec(c.get('contribution', 0), 3)} "
                    f"| {c.get('impact', '-')} |"
                )

        return "\n".join(lines)

    def _md_sector_comparison(self, data: Dict[str, Any]) -> str:
        sectors = data.get("sector_comparison", [])
        lines = [
            "## 5. Sector Comparison\n",
            "Company temperature score benchmarked against sector averages.\n",
            "| Sector | Sector Avg (C) | Company (C) | Delta (C) "
            "| Percentile | Better/Worse |",
            "|--------|:--------------:|:-----------:|:---------:"
            "|:----------:|:------------:|",
        ]
        for s in sectors:
            company = float(s.get("company_score", 0))
            sector_avg = float(s.get("sector_avg", 0))
            delta = company - sector_avg
            status = "Better" if delta < 0 else "Worse" if delta > 0 else "Equal"
            lines.append(
                f"| {s.get('sector', '-')} "
                f"| {_dec(sector_avg, 2)} "
                f"| {_dec(company, 2)} "
                f"| {'+' if delta > 0 else ''}{_dec(delta, 2)} "
                f"| {s.get('percentile', '-')} "
                f"| {status} |"
            )
        if not sectors:
            lines.append(
                "| - | _No sector benchmarks_ | - | - | - | - |"
            )

        # Global benchmarks
        benchmarks = data.get("global_benchmarks", {})
        if benchmarks:
            lines.append("")
            lines.append("### Global Benchmarks\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Global Average | {_dec(benchmarks.get('global_avg', 0), 2)}C |\n"
                f"| Paris-Aligned Average | {_dec(benchmarks.get('paris_aligned_avg', 0), 2)}C |\n"
                f"| SBTi Committed Average | {_dec(benchmarks.get('sbti_committed_avg', 0), 2)}C |\n"
                f"| SBTi Validated Average | {_dec(benchmarks.get('sbti_validated_avg', 0), 2)}C |"
            )

        return "\n".join(lines)

    def _md_trajectory_visualization(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        lines = [
            "## 6. Trajectory Visualization\n",
            "Company temperature trajectory against 1.5C and 2C pathways.\n",
            "| Year | Company Score (C) | 1.5C Pathway (C) | 2C Pathway (C) "
            "| Delta vs 1.5C | Delta vs 2C | Status |",
            "|:----:|:-----------------:|:----------------:|:--------------:"
            "|:-------------:|:-----------:|--------|",
        ]
        for t in trajectory:
            company = float(t.get("company_score", 0))
            p15 = float(t.get("pathway_1_5c", 0))
            p20 = float(t.get("pathway_2c", 0))
            d15 = company - p15
            d20 = company - p20
            status = (
                "1.5C Aligned" if d15 <= 0
                else "2C Aligned" if d20 <= 0
                else "Misaligned"
            )
            lines.append(
                f"| {t.get('year', '-')} "
                f"| {_dec(company, 2)} "
                f"| {_dec(p15, 2)} "
                f"| {_dec(p20, 2)} "
                f"| {'+' if d15 > 0 else ''}{_dec(d15, 2)} "
                f"| {'+' if d20 > 0 else ''}{_dec(d20, 2)} "
                f"| {status} |"
            )
        if not trajectory:
            lines.append(
                "| - | _No trajectory data_ | - | - | - | - | - |"
            )

        # Trend summary
        if len(trajectory) >= 2:
            first = float(trajectory[0].get("company_score", 0))
            last = float(trajectory[-1].get("company_score", 0))
            change = last - first
            trend = "Improving" if change < 0 else "Worsening" if change > 0 else "Stable"
            lines.append("")
            lines.append(
                f"**Trend:** {trend} ({'+' if change > 0 else ''}"
                f"{_dec(change, 2)}C from {trajectory[0].get('year', '?')} "
                f"to {trajectory[-1].get('year', '?')})"
            )

        return "\n".join(lines)

    def _md_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("improvement_opportunities", [])
        lines = [
            "## 7. Score Improvement Opportunities\n",
            "Actions ranked by potential temperature score improvement.\n",
            "| # | Opportunity | Scope | Score Impact (C) "
            "| Effort | Priority | Timeline |",
            "|---|------------|-------|:----------------:"
            "|:------:|:--------:|:--------:|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('opportunity', '-')} "
                f"| {o.get('scope', '-')} "
                f"| {_dec(o.get('score_impact', 0), 2)} "
                f"| {o.get('effort', '-')} "
                f"| {o.get('priority', '-')} "
                f"| {o.get('timeline', '-')} |"
            )
        if not opps:
            lines.append(
                "| - | _No improvement opportunities identified_ "
                "| - | - | - | - | - |"
            )

        # Potential score after improvements
        potential = data.get("potential_score", {})
        if potential:
            lines.append("")
            lines.append(
                f"**Current Score:** {_dec(potential.get('current', 0), 2)}C  \n"
                f"**Potential Score (all actions):** "
                f"{_dec(potential.get('potential', 0), 2)}C  \n"
                f"**Improvement:** {_dec(potential.get('improvement', 0), 2)}C  \n"
                f"**Potential Alignment:** "
                f"{_temp_label(float(potential.get('potential', 0)))}"
            )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Temperature scoring per SBTi Temperature Rating Methodology V2.0 "
            f"with CDP-SBTi Temperature Ratings dataset.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".temp-15{color:#1b5e20;font-weight:700;}"
            ".temp-20{color:#388e3c;font-weight:600;}"
            ".temp-25{color:#f57c00;font-weight:600;}"
            ".temp-32{color:#e65100;font-weight:700;}"
            ".temp-high{color:#c62828;font-weight:700;}"
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-warn{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".temp-gauge{display:flex;align-items:center;gap:8px;margin:10px 0;}"
            ".temp-bar{flex:1;height:24px;background:linear-gradient(90deg,"
            "#1b5e20 0%,#43a047 25%,#ff9800 50%,#ef5350 75%,#b71c1c 100%);"
            "border-radius:12px;position:relative;}"
            ".temp-marker{width:4px;height:32px;background:#1a1a2e;position:absolute;"
            "top:-4px;border-radius:2px;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_temp_badge(self, temp: float) -> str:
        """Return an HTML badge for a temperature score."""
        cls = _temp_color(temp)
        label = _temp_label(temp)
        return f'<span class="{cls}">{_dec(temp, 2)}C ({label})</span>'

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Temperature Rating Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Methodology:</strong> SBTi Temperature Rating V2.0</p>'
        )

    def _html_score_summary(self, data: Dict[str, Any]) -> str:
        scores = data.get("scores", {})
        company = float(scores.get("company_score", 0))
        portfolio = float(scores.get("portfolio_score", 0))
        company_cls = _temp_color(company)
        portfolio_cls = _temp_color(portfolio)
        marker_pos = min(max((company - 1.0) / 4.0 * 100, 0), 100)

        return (
            f'<h2>1. Temperature Score Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Company Score</div>'
            f'<div class="card-value {company_cls}">{_dec(company, 2)}C</div>'
            f'<div class="card-unit">{_temp_label(company)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Portfolio Score</div>'
            f'<div class="card-value {portfolio_cls}">{_dec(portfolio, 2)}C</div>'
            f'<div class="card-unit">{_temp_label(portfolio)}</div></div>\n'
            f'  <div class="card"><div class="card-label">1.5C Test</div>'
            f'<div class="card-value">{"MET" if company <= 1.5 else "NOT MET"}</div></div>\n'
            f'  <div class="card"><div class="card-label">2C Test</div>'
            f'<div class="card-value">{"MET" if company <= 2.0 else "NOT MET"}</div></div>\n'
            f'</div>\n'
            f'<div class="temp-gauge">\n'
            f'  <span>1.0C</span>\n'
            f'  <div class="temp-bar">'
            f'<div class="temp-marker" style="left:{marker_pos}%;"></div></div>\n'
            f'  <span>5.0C</span>\n'
            f'</div>'
        )

    def _html_scope_timeframe(self, data: Dict[str, Any]) -> str:
        st = data.get("scope_timeframe", [])
        rows = ""
        for s in st:
            combined = float(s.get("combined", 0))
            cls = _temp_color(combined)
            rows += (
                f'<tr><td><strong>{s.get("scope", "-")}</strong></td>'
                f'<td>{_dec(s.get("short_term", 0), 2)}</td>'
                f'<td>{_dec(s.get("mid_term", 0), 2)}</td>'
                f'<td>{_dec(s.get("long_term", 0), 2)}</td>'
                f'<td class="{cls}">{_dec(combined, 2)}</td>'
                f'<td>{_temp_label(combined)}</td></tr>\n'
            )
        return (
            f'<h2>2. Score by Scope & Timeframe</h2>\n'
            f'<table>\n'
            f'<tr><th>Scope</th><th>Short-Term (C)</th><th>Mid-Term (C)</th>'
            f'<th>Long-Term (C)</th><th>Combined (C)</th><th>Alignment</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_aggregation_methods(self, data: Dict[str, Any]) -> str:
        methods = data.get("aggregation_methods", [])
        rows = ""
        for m in methods:
            code = m.get("method", "-")
            full_name = AGGREGATION_METHODS.get(code, code)
            combined = float(m.get("combined", 0))
            cls = _temp_color(combined)
            rows += (
                f'<tr><td><strong>{code}</strong></td>'
                f'<td>{full_name}</td>'
                f'<td>{_dec(m.get("s1s2", 0), 2)}</td>'
                f'<td>{_dec(m.get("s3", 0), 2)}</td>'
                f'<td class="{cls}">{_dec(combined, 2)}</td>'
                f'<td>{_temp_label(combined)}</td></tr>\n'
            )

        selected = data.get("selected_method", {})
        selected_html = ""
        if selected:
            selected_html = (
                f'<p><strong>Selected:</strong> {selected.get("method", "N/A")} '
                f'({AGGREGATION_METHODS.get(selected.get("method", ""), "")}) | '
                f'<strong>Rationale:</strong> {selected.get("rationale", "N/A")}</p>\n'
            )

        return (
            f'<h2>3. Aggregation Methods</h2>\n'
            f'<table>\n'
            f'<tr><th>Method</th><th>Full Name</th><th>S1+S2 (C)</th>'
            f'<th>S3 (C)</th><th>Combined (C)</th><th>Alignment</th></tr>\n'
            f'{rows}</table>\n'
            f'{selected_html}'
        )

    def _html_contribution_analysis(self, data: Dict[str, Any]) -> str:
        contributions = data.get("contributions", [])
        rows = ""
        for i, c in enumerate(contributions, 1):
            score = float(c.get("score", 0))
            cls = _temp_color(score)
            rows += (
                f'<tr><td>{i}</td><td>{c.get("entity", "-")}</td>'
                f'<td>{_pct(c.get("weight_pct", 0))}</td>'
                f'<td class="{cls}">{_dec(score, 2)}</td>'
                f'<td>{_dec(c.get("contribution", 0), 3)}</td>'
                f'<td>{c.get("target_status", "-")}</td>'
                f'<td>{c.get("sector", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Contribution Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Entity</th><th>Weight</th><th>Score (C)</th>'
            f'<th>Contribution (C)</th><th>Target Status</th><th>Sector</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sector_comparison(self, data: Dict[str, Any]) -> str:
        sectors = data.get("sector_comparison", [])
        rows = ""
        for s in sectors:
            company = float(s.get("company_score", 0))
            sector_avg = float(s.get("sector_avg", 0))
            delta = company - sector_avg
            cls = "temp-15" if delta < 0 else "temp-high" if delta > 0.5 else "temp-25"
            status = "Better" if delta < 0 else "Worse" if delta > 0 else "Equal"
            rows += (
                f'<tr><td>{s.get("sector", "-")}</td>'
                f'<td>{_dec(sector_avg, 2)}</td>'
                f'<td>{_dec(company, 2)}</td>'
                f'<td class="{cls}">{"+" if delta > 0 else ""}{_dec(delta, 2)}</td>'
                f'<td>{s.get("percentile", "-")}</td>'
                f'<td>{status}</td></tr>\n'
            )

        benchmarks = data.get("global_benchmarks", {})
        bench_html = ""
        if benchmarks:
            bench_html = (
                f'<h3>Global Benchmarks</h3>\n'
                f'<div class="summary-cards">\n'
                f'  <div class="card"><div class="card-label">Global Avg</div>'
                f'<div class="card-value">{_dec(benchmarks.get("global_avg", 0), 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">Paris-Aligned</div>'
                f'<div class="card-value">{_dec(benchmarks.get("paris_aligned_avg", 0), 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">SBTi Committed</div>'
                f'<div class="card-value">{_dec(benchmarks.get("sbti_committed_avg", 0), 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">SBTi Validated</div>'
                f'<div class="card-value">{_dec(benchmarks.get("sbti_validated_avg", 0), 2)}C</div></div>\n'
                f'</div>\n'
            )

        return (
            f'<h2>5. Sector Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>Sector</th><th>Sector Avg (C)</th><th>Company (C)</th>'
            f'<th>Delta (C)</th><th>Percentile</th><th>Status</th></tr>\n'
            f'{rows}</table>\n'
            f'{bench_html}'
        )

    def _html_trajectory_visualization(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        rows = ""
        for t in trajectory:
            company = float(t.get("company_score", 0))
            p15 = float(t.get("pathway_1_5c", 0))
            p20 = float(t.get("pathway_2c", 0))
            d15 = company - p15
            d20 = company - p20
            if d15 <= 0:
                status = '<span class="badge-pass">1.5C Aligned</span>'
            elif d20 <= 0:
                status = '<span class="badge-warn">2C Aligned</span>'
            else:
                status = '<span class="badge-fail">Misaligned</span>'
            rows += (
                f'<tr><td>{t.get("year", "-")}</td>'
                f'<td>{_dec(company, 2)}</td>'
                f'<td>{_dec(p15, 2)}</td>'
                f'<td>{_dec(p20, 2)}</td>'
                f'<td>{"+" if d15 > 0 else ""}{_dec(d15, 2)}</td>'
                f'<td>{"+" if d20 > 0 else ""}{_dec(d20, 2)}</td>'
                f'<td>{status}</td></tr>\n'
            )
        return (
            f'<h2>6. Trajectory Visualization</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Company (C)</th><th>1.5C Pathway</th>'
            f'<th>2C Pathway</th><th>Delta 1.5C</th><th>Delta 2C</th>'
            f'<th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("improvement_opportunities", [])
        rows = ""
        for i, o in enumerate(opps, 1):
            rows += (
                f'<tr><td>{i}</td><td>{o.get("opportunity", "-")}</td>'
                f'<td>{o.get("scope", "-")}</td>'
                f'<td>{_dec(o.get("score_impact", 0), 2)}</td>'
                f'<td>{o.get("effort", "-")}</td>'
                f'<td>{o.get("priority", "-")}</td>'
                f'<td>{o.get("timeline", "-")}</td></tr>\n'
            )

        potential = data.get("potential_score", {})
        potential_html = ""
        if potential:
            current = float(potential.get("current", 0))
            pot = float(potential.get("potential", 0))
            potential_html = (
                f'<div class="summary-cards">\n'
                f'  <div class="card"><div class="card-label">Current Score</div>'
                f'<div class="card-value">{_dec(current, 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">Potential Score</div>'
                f'<div class="card-value">{_dec(pot, 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">Improvement</div>'
                f'<div class="card-value">{_dec(potential.get("improvement", 0), 2)}C</div></div>\n'
                f'  <div class="card"><div class="card-label">Potential Alignment</div>'
                f'<div class="card-value">{_temp_label(pot)}</div></div>\n'
                f'</div>\n'
            )

        return (
            f'<h2>7. Score Improvement Opportunities</h2>\n'
            f'{potential_html}'
            f'<table>\n'
            f'<tr><th>#</th><th>Opportunity</th><th>Scope</th>'
            f'<th>Impact (C)</th><th>Effort</th><th>Priority</th>'
            f'<th>Timeline</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Temperature scoring per SBTi Temperature Rating Methodology V2.0 '
            f'with CDP-SBTi Temperature Ratings dataset.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
