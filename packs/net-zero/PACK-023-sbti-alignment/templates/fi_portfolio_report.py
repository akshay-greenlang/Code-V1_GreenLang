# -*- coding: utf-8 -*-
"""
FIPortfolioReportTemplate - Financial Institution portfolio report for PACK-023.

Renders a comprehensive FI portfolio-level target and coverage report per
SBTi Financial Institutions Net-Zero (FINZ) Standard V1.0 covering eight
asset classes, portfolio-level targets per class, PCAF data quality scores,
engagement target progress, temperature alignment per asset class, and
coverage dashboard.

Sections:
    1. Portfolio Overview (AUM, covered assets, temperature alignment)
    2. Asset Class Breakdown (8 classes: listed equity, corporate bonds,
       business loans, project finance, commercial RE, mortgages,
       sovereign debt, other)
    3. Portfolio-Level Targets per Asset Class
    4. PCAF Data Quality Scores
    5. Engagement Target Progress
    6. Temperature Alignment by Asset Class
    7. Coverage Dashboard

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

# FINZ V1.0 asset classes
FINZ_ASSET_CLASSES = [
    "Listed Equity & Corporate Bonds",
    "Business Loans",
    "Project Finance",
    "Commercial Real Estate",
    "Residential Mortgages",
    "Motor Vehicle Loans",
    "Sovereign Debt",
    "Other / Unclassified",
]

# PCAF data quality scale descriptions
PCAF_QUALITY_SCALE = {
    1: "Audited/verified data (highest quality)",
    2: "Non-audited but primary data from counterparties",
    3: "Estimated using production/physical activity data",
    4: "Estimated using economic activity data",
    5: "Estimated using asset-class-level data (lowest quality)",
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


def _pcaf_label(score: float) -> str:
    """Return PCAF quality label for a score."""
    rounded = round(score)
    if rounded < 1:
        rounded = 1
    if rounded > 5:
        rounded = 5
    return PCAF_QUALITY_SCALE.get(rounded, f"Score {score}")


class FIPortfolioReportTemplate:
    """
    FI portfolio report template for SBTi FINZ alignment.

    Renders a complete portfolio-level report for financial institutions
    per FINZ V1.0 covering eight asset classes, PCAF data quality scores,
    engagement targets, temperature alignment, and coverage tracking.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FIPortfolioReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render FI portfolio report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_portfolio_overview(data),
            self._md_asset_class_breakdown(data),
            self._md_portfolio_targets(data),
            self._md_pcaf_quality(data),
            self._md_engagement_progress(data),
            self._md_temperature_alignment(data),
            self._md_coverage_dashboard(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render FI portfolio report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_portfolio_overview(data),
            self._html_asset_class_breakdown(data),
            self._html_portfolio_targets(data),
            self._html_pcaf_quality(data),
            self._html_engagement_progress(data),
            self._html_temperature_alignment(data),
            self._html_coverage_dashboard(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>FI Portfolio Report - FINZ V1.0</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render FI portfolio report as structured JSON."""
        self.generated_at = _utcnow()
        overview = data.get("overview", {})
        asset_classes = data.get("asset_classes", [])
        targets = data.get("portfolio_targets", [])
        pcaf = data.get("pcaf_scores", [])
        engagement = data.get("engagement_targets", [])
        temperature = data.get("temperature_alignment", [])
        coverage = data.get("coverage", {})

        result: Dict[str, Any] = {
            "template": "fi_portfolio_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "overview": overview,
            "asset_classes": asset_classes,
            "portfolio_targets": targets,
            "pcaf_scores": pcaf,
            "engagement_targets": engagement,
            "temperature_alignment": temperature,
            "coverage": coverage,
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
            f"# FI Portfolio Report - FINZ V1.0\n\n"
            f"**Financial Institution:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** SBTi Financial Institutions Net-Zero (FINZ) V1.0\n\n---"
        )

    def _md_portfolio_overview(self, data: Dict[str, Any]) -> str:
        ov = data.get("overview", {})
        return (
            f"## 1. Portfolio Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total AUM | {_dec_comma(ov.get('total_aum', 0), 0)} {ov.get('currency', 'USD')} |\n"
            f"| Covered AUM | {_dec_comma(ov.get('covered_aum', 0), 0)} {ov.get('currency', 'USD')} |\n"
            f"| AUM Coverage | {_pct(ov.get('aum_coverage_pct', 0))} |\n"
            f"| Financed Emissions | {_dec_comma(ov.get('financed_emissions_tco2e', 0), 0)} tCO2e |\n"
            f"| Portfolio Temperature | {_dec(ov.get('portfolio_temperature', 0), 2)}C |\n"
            f"| Temperature Alignment | {ov.get('temperature_alignment', 'N/A')} |\n"
            f"| Number of Counterparties | {ov.get('num_counterparties', 0)} |\n"
            f"| Asset Classes Active | {ov.get('asset_classes_active', 0)} / 8 |\n"
            f"| PCAF Weighted Score | {_dec(ov.get('pcaf_weighted_score', 0), 1)} |\n"
            f"| Target Setting Method | {ov.get('target_method', 'Sectoral Decarbonization')} |"
        )

    def _md_asset_class_breakdown(self, data: Dict[str, Any]) -> str:
        classes = data.get("asset_classes", [])
        lines = [
            "## 2. Asset Class Breakdown\n",
            "Portfolio composition across FINZ V1.0 asset classes.\n",
            "| # | Asset Class | AUM | % of Total "
            "| Financed Emissions (tCO2e) | % of Emissions "
            "| Counterparties | PCAF Score |",
            "|---|------------|----:|:----------:"
            "|:---------------------------:|:--------------:"
            "|:--------------:|:----------:|",
        ]
        for i, ac in enumerate(classes, 1):
            lines.append(
                f"| {i} | {ac.get('asset_class', '-')} "
                f"| {_dec_comma(ac.get('aum', 0), 0)} "
                f"| {_pct(ac.get('pct_of_total_aum', 0))} "
                f"| {_dec_comma(ac.get('financed_emissions_tco2e', 0), 0)} "
                f"| {_pct(ac.get('pct_of_emissions', 0))} "
                f"| {ac.get('num_counterparties', 0)} "
                f"| {_dec(ac.get('pcaf_score', 0), 1)} |"
            )
        if not classes:
            lines.append(
                "| - | _No asset classes_ | - | - | - | - | - | - |"
            )

        # Intensity by asset class
        intensities = data.get("asset_class_intensities", [])
        if intensities:
            lines.append("")
            lines.append("### Emission Intensity by Asset Class\n")
            lines.append(
                "| Asset Class | Intensity | Unit | Benchmark | Status |"
            )
            lines.append(
                "|------------|:---------:|------|:---------:|--------|"
            )
            for ai in intensities:
                lines.append(
                    f"| {ai.get('asset_class', '-')} "
                    f"| {_dec(ai.get('intensity', 0), 4)} "
                    f"| {ai.get('unit', '-')} "
                    f"| {_dec(ai.get('benchmark', 0), 4)} "
                    f"| {ai.get('status', '-')} |"
                )

        return "\n".join(lines)

    def _md_portfolio_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("portfolio_targets", [])
        lines = [
            "## 3. Portfolio-Level Targets per Asset Class\n",
            "SBTi-aligned portfolio decarbonization targets.\n",
            "| Asset Class | Target Type | Base Year | Target Year "
            "| Base Intensity | Target Intensity | Reduction (%) "
            "| Pathway | Status |",
            "|------------|:-----------:|:---------:|:-----------:"
            "|:--------------:|:----------------:|:-------------:"
            "|---------|--------|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('asset_class', '-')} "
                f"| {t.get('target_type', '-')} "
                f"| {t.get('base_year', '-')} "
                f"| {t.get('target_year', '-')} "
                f"| {_dec(t.get('base_intensity', 0), 4)} "
                f"| {_dec(t.get('target_intensity', 0), 4)} "
                f"| {_pct(t.get('reduction_pct', 0))} "
                f"| {t.get('pathway', '-')} "
                f"| {t.get('status', '-')} |"
            )
        if not targets:
            lines.append(
                "| - | _No targets set_ | - | - | - | - | - | - | - |"
            )

        # Portfolio-wide aggregates
        agg = data.get("portfolio_aggregate", {})
        if agg:
            lines.append("")
            lines.append("### Portfolio Aggregate\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Portfolio Reduction Target | {_pct(agg.get('total_reduction_pct', 0))} |\n"
                f"| Weighted Average Target Year | {agg.get('avg_target_year', 'N/A')} |\n"
                f"| Assets with Targets | {_pct(agg.get('assets_with_targets_pct', 0))} |\n"
                f"| FINZ Compliance | {agg.get('finz_compliance', 'N/A')} |"
            )

        return "\n".join(lines)

    def _md_pcaf_quality(self, data: Dict[str, Any]) -> str:
        scores = data.get("pcaf_scores", [])
        lines = [
            "## 4. PCAF Data Quality Scores\n",
            "Partnership for Carbon Accounting Financials data quality assessment.\n",
            "| Asset Class | Score (1-5) | Description "
            "| % at Score 1-2 | % at Score 3 | % at Score 4-5 "
            "| Improvement Plan |",
            "|------------|:-----------:|-------------"
            "|:--------------:|:------------:|:--------------:"
            "|:----------------:|",
        ]
        for s in scores:
            score = float(s.get("score", 0))
            lines.append(
                f"| {s.get('asset_class', '-')} "
                f"| {_dec(score, 1)} "
                f"| {_pcaf_label(score)} "
                f"| {_pct(s.get('pct_score_1_2', 0))} "
                f"| {_pct(s.get('pct_score_3', 0))} "
                f"| {_pct(s.get('pct_score_4_5', 0))} "
                f"| {s.get('improvement_plan', '-')} |"
            )
        if not scores:
            lines.append(
                "| - | _No PCAF scores_ | - | - | - | - | - |"
            )

        # PCAF quality scale reference
        lines.append("")
        lines.append("### PCAF Data Quality Scale Reference\n")
        lines.append("| Score | Description |")
        lines.append("|:-----:|-------------|")
        for k, v in PCAF_QUALITY_SCALE.items():
            lines.append(f"| {k} | {v} |")

        return "\n".join(lines)

    def _md_engagement_progress(self, data: Dict[str, Any]) -> str:
        engagement = data.get("engagement_targets", [])
        lines = [
            "## 5. Engagement Target Progress\n",
            "Progress on counterparty engagement targets per FINZ requirements.\n",
            "| Asset Class | Engagement Scope | Total Counterparties "
            "| Engaged | % Engaged | Target (%) "
            "| Status | Notes |",
            "|------------|:----------------:|:--------------------:"
            "|:--------:|:---------:|:----------:"
            "|--------|-------|",
        ]
        for e in engagement:
            lines.append(
                f"| {e.get('asset_class', '-')} "
                f"| {e.get('engagement_scope', '-')} "
                f"| {e.get('total_counterparties', 0)} "
                f"| {e.get('engaged', 0)} "
                f"| {_pct(e.get('pct_engaged', 0))} "
                f"| {_pct(e.get('target_pct', 0))} "
                f"| {e.get('status', '-')} "
                f"| {e.get('notes', '-')} |"
            )
        if not engagement:
            lines.append(
                "| - | _No engagement targets_ | - | - | - | - | - | - |"
            )

        # Engagement summary
        eng_summary = data.get("engagement_summary", {})
        if eng_summary:
            lines.append("")
            lines.append(
                f"**Total Counterparties Engaged:** "
                f"{eng_summary.get('total_engaged', 0)} / "
                f"{eng_summary.get('total_counterparties', 0)}  \n"
                f"**Overall Engagement Rate:** "
                f"{_pct(eng_summary.get('overall_pct', 0))}  \n"
                f"**SBTi-Committed Counterparties:** "
                f"{eng_summary.get('sbti_committed', 0)}  \n"
                f"**SBTi-Validated Counterparties:** "
                f"{eng_summary.get('sbti_validated', 0)}"
            )

        return "\n".join(lines)

    def _md_temperature_alignment(self, data: Dict[str, Any]) -> str:
        alignment = data.get("temperature_alignment", [])
        lines = [
            "## 6. Temperature Alignment by Asset Class\n",
            "Implied temperature rise (ITR) per asset class using "
            "FINZ methodology.\n",
            "| Asset Class | ITR (C) | Alignment | 1.5C Gap "
            "| 2C Gap | Trend | Method |",
            "|------------|:-------:|-----------|:--------:"
            "|:------:|:-----:|--------|",
        ]
        for a in alignment:
            itr = float(a.get("itr", 0))
            gap_15 = itr - 1.5
            gap_20 = itr - 2.0
            alignment_label = (
                "1.5C Aligned" if itr <= 1.5
                else "Below 2C" if itr <= 2.0
                else "Above 2C" if itr <= 3.0
                else "Misaligned"
            )
            lines.append(
                f"| {a.get('asset_class', '-')} "
                f"| {_dec(itr, 2)} "
                f"| {alignment_label} "
                f"| {'+' if gap_15 > 0 else ''}{_dec(gap_15, 2)} "
                f"| {'+' if gap_20 > 0 else ''}{_dec(gap_20, 2)} "
                f"| {a.get('trend', '-')} "
                f"| {a.get('method', '-')} |"
            )
        if not alignment:
            lines.append(
                "| - | _No alignment data_ | - | - | - | - | - |"
            )

        return "\n".join(lines)

    def _md_coverage_dashboard(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        dimensions = cov.get("dimensions", [])
        lines = [
            "## 7. Coverage Dashboard\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| AUM Covered by Targets | {_pct(cov.get('aum_covered_pct', 0))} |\n"
            f"| Emissions Covered by Targets | {_pct(cov.get('emissions_covered_pct', 0))} |\n"
            f"| Asset Classes with Targets | {cov.get('classes_with_targets', 0)} / 8 |\n"
            f"| FINZ Minimum Coverage Met | {cov.get('finz_min_met', 'N/A')} |\n"
            f"| Data Quality Threshold Met | {cov.get('data_quality_met', 'N/A')} |\n"
            f"| Engagement Requirements Met | {cov.get('engagement_met', 'N/A')} |",
        ]

        if dimensions:
            lines.append("")
            lines.append("### Coverage by Dimension\n")
            lines.append(
                "| Dimension | Coverage (%) | Required (%) | Gap (%) | Status |"
            )
            lines.append(
                "|-----------|:------------:|:------------:|:-------:|--------|"
            )
            for d in dimensions:
                coverage_val = float(d.get("coverage_pct", 0))
                required_val = float(d.get("required_pct", 0))
                gap = coverage_val - required_val
                lines.append(
                    f"| {d.get('dimension', '-')} "
                    f"| {_pct(coverage_val)} "
                    f"| {_pct(required_val)} "
                    f"| {'+' if gap > 0 else ''}{_pct(gap)} "
                    f"| {d.get('status', '-')} |"
                )

        gaps = cov.get("gaps", [])
        if gaps:
            lines.append("")
            lines.append("### Coverage Gaps & Actions\n")
            lines.append("| # | Gap | Impact | Action Required | Priority |")
            lines.append("|---|-----|--------|-----------------|:--------:|")
            for i, g in enumerate(gaps, 1):
                lines.append(
                    f"| {i} | {g.get('gap', '-')} "
                    f"| {g.get('impact', '-')} "
                    f"| {g.get('action', '-')} "
                    f"| {g.get('priority', '-')} |"
                )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Portfolio targets per SBTi Financial Institutions Net-Zero (FINZ) "
            f"Standard V1.0 and PCAF Global GHG Standard.*"
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
            ".pcaf-1{background:#1b5e20;color:#fff;font-weight:700;"
            "padding:2px 8px;border-radius:4px;display:inline-block;}"
            ".pcaf-2{background:#43a047;color:#fff;font-weight:600;"
            "padding:2px 8px;border-radius:4px;display:inline-block;}"
            ".pcaf-3{background:#ff9800;color:#fff;font-weight:600;"
            "padding:2px 8px;border-radius:4px;display:inline-block;}"
            ".pcaf-4{background:#ef6c00;color:#fff;font-weight:600;"
            "padding:2px 8px;border-radius:4px;display:inline-block;}"
            ".pcaf-5{background:#ef5350;color:#fff;font-weight:700;"
            "padding:2px 8px;border-radius:4px;display:inline-block;}"
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-warn{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_pcaf_badge(self, score: float) -> str:
        """Return an HTML badge for PCAF score."""
        rounded = round(score)
        if rounded <= 1:
            return f'<span class="pcaf-1">{_dec(score, 1)}</span>'
        elif rounded <= 2:
            return f'<span class="pcaf-2">{_dec(score, 1)}</span>'
        elif rounded <= 3:
            return f'<span class="pcaf-3">{_dec(score, 1)}</span>'
        elif rounded <= 4:
            return f'<span class="pcaf-4">{_dec(score, 1)}</span>'
        else:
            return f'<span class="pcaf-5">{_dec(score, 1)}</span>'

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>FI Portfolio Report - FINZ V1.0</h1>\n'
            f'<p><strong>Financial Institution:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Standard:</strong> SBTi FINZ V1.0</p>'
        )

    def _html_portfolio_overview(self, data: Dict[str, Any]) -> str:
        ov = data.get("overview", {})
        temp = float(ov.get("portfolio_temperature", 0))
        return (
            f'<h2>1. Portfolio Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total AUM</div>'
            f'<div class="card-value">{_dec_comma(ov.get("total_aum", 0), 0)}</div>'
            f'<div class="card-unit">{ov.get("currency", "USD")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Coverage</div>'
            f'<div class="card-value">{_pct(ov.get("aum_coverage_pct", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Financed Emissions</div>'
            f'<div class="card-value">{_dec_comma(ov.get("financed_emissions_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Portfolio Temp</div>'
            f'<div class="card-value">{_dec(temp, 2)}C</div></div>\n'
            f'  <div class="card"><div class="card-label">PCAF Score</div>'
            f'<div class="card-value">{self._html_pcaf_badge(float(ov.get("pcaf_weighted_score", 0)))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Counterparties</div>'
            f'<div class="card-value">{ov.get("num_counterparties", 0)}</div></div>\n'
            f'</div>'
        )

    def _html_asset_class_breakdown(self, data: Dict[str, Any]) -> str:
        classes = data.get("asset_classes", [])
        rows = ""
        for i, ac in enumerate(classes, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td><strong>{ac.get("asset_class", "-")}</strong></td>'
                f'<td>{_dec_comma(ac.get("aum", 0), 0)}</td>'
                f'<td>{_pct(ac.get("pct_of_total_aum", 0))}</td>'
                f'<td>{_dec_comma(ac.get("financed_emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(ac.get("pct_of_emissions", 0))}</td>'
                f'<td>{ac.get("num_counterparties", 0)}</td>'
                f'<td>{self._html_pcaf_badge(float(ac.get("pcaf_score", 0)))}</td></tr>\n'
            )
        return (
            f'<h2>2. Asset Class Breakdown</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Asset Class</th><th>AUM</th>'
            f'<th>% of Total</th><th>Financed Emissions</th>'
            f'<th>% of Emissions</th><th>Counterparties</th>'
            f'<th>PCAF Score</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_portfolio_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("portfolio_targets", [])
        rows = ""
        for t in targets:
            rows += (
                f'<tr><td>{t.get("asset_class", "-")}</td>'
                f'<td>{t.get("target_type", "-")}</td>'
                f'<td>{t.get("base_year", "-")}</td>'
                f'<td>{t.get("target_year", "-")}</td>'
                f'<td>{_dec(t.get("base_intensity", 0), 4)}</td>'
                f'<td>{_dec(t.get("target_intensity", 0), 4)}</td>'
                f'<td>{_pct(t.get("reduction_pct", 0))}</td>'
                f'<td>{t.get("pathway", "-")}</td>'
                f'<td>{t.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Portfolio-Level Targets</h2>\n'
            f'<table>\n'
            f'<tr><th>Asset Class</th><th>Target Type</th><th>Base Year</th>'
            f'<th>Target Year</th><th>Base Intensity</th>'
            f'<th>Target Intensity</th><th>Reduction</th>'
            f'<th>Pathway</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_pcaf_quality(self, data: Dict[str, Any]) -> str:
        scores = data.get("pcaf_scores", [])
        rows = ""
        for s in scores:
            score = float(s.get("score", 0))
            rows += (
                f'<tr><td>{s.get("asset_class", "-")}</td>'
                f'<td>{self._html_pcaf_badge(score)}</td>'
                f'<td>{_pcaf_label(score)}</td>'
                f'<td>{_pct(s.get("pct_score_1_2", 0))}</td>'
                f'<td>{_pct(s.get("pct_score_3", 0))}</td>'
                f'<td>{_pct(s.get("pct_score_4_5", 0))}</td>'
                f'<td>{s.get("improvement_plan", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. PCAF Data Quality Scores</h2>\n'
            f'<table>\n'
            f'<tr><th>Asset Class</th><th>Score</th><th>Description</th>'
            f'<th>% Score 1-2</th><th>% Score 3</th><th>% Score 4-5</th>'
            f'<th>Improvement Plan</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_engagement_progress(self, data: Dict[str, Any]) -> str:
        engagement = data.get("engagement_targets", [])
        rows = ""
        for e in engagement:
            pct = float(e.get("pct_engaged", 0))
            target = float(e.get("target_pct", 0))
            bar_color = "#43a047" if pct >= target else "#ff9800" if pct >= target * 0.7 else "#ef5350"
            rows += (
                f'<tr><td>{e.get("asset_class", "-")}</td>'
                f'<td>{e.get("engagement_scope", "-")}</td>'
                f'<td>{e.get("total_counterparties", 0)}</td>'
                f'<td>{e.get("engaged", 0)}</td>'
                f'<td>{_pct(pct)}</td>'
                f'<td>{_pct(target)}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{min(pct, 100)}%;background:{bar_color};"></div>'
                f'</div></td>'
                f'<td>{e.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Engagement Target Progress</h2>\n'
            f'<table>\n'
            f'<tr><th>Asset Class</th><th>Scope</th><th>Total</th>'
            f'<th>Engaged</th><th>%</th><th>Target</th>'
            f'<th>Progress</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_temperature_alignment(self, data: Dict[str, Any]) -> str:
        alignment = data.get("temperature_alignment", [])
        rows = ""
        for a in alignment:
            itr = float(a.get("itr", 0))
            alignment_label = (
                "1.5C Aligned" if itr <= 1.5
                else "Below 2C" if itr <= 2.0
                else "Above 2C" if itr <= 3.0
                else "Misaligned"
            )
            badge_cls = (
                "badge-pass" if itr <= 1.5
                else "badge-warn" if itr <= 2.0
                else "badge-fail"
            )
            gap_15 = itr - 1.5
            gap_20 = itr - 2.0
            rows += (
                f'<tr><td>{a.get("asset_class", "-")}</td>'
                f'<td>{_dec(itr, 2)}</td>'
                f'<td><span class="{badge_cls}">{alignment_label}</span></td>'
                f'<td>{"+" if gap_15 > 0 else ""}{_dec(gap_15, 2)}</td>'
                f'<td>{"+" if gap_20 > 0 else ""}{_dec(gap_20, 2)}</td>'
                f'<td>{a.get("trend", "-")}</td>'
                f'<td>{a.get("method", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Temperature Alignment by Asset Class</h2>\n'
            f'<table>\n'
            f'<tr><th>Asset Class</th><th>ITR (C)</th><th>Alignment</th>'
            f'<th>1.5C Gap</th><th>2C Gap</th><th>Trend</th>'
            f'<th>Method</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_coverage_dashboard(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        aum_pct = float(cov.get("aum_covered_pct", 0))
        emissions_pct = float(cov.get("emissions_covered_pct", 0))

        dimensions = cov.get("dimensions", [])
        dim_rows = ""
        for d in dimensions:
            coverage_val = float(d.get("coverage_pct", 0))
            required_val = float(d.get("required_pct", 0))
            bar_color = "#43a047" if coverage_val >= required_val else "#ef5350"
            dim_rows += (
                f'<tr><td>{d.get("dimension", "-")}</td>'
                f'<td>{_pct(coverage_val)}</td>'
                f'<td>{_pct(required_val)}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{min(coverage_val, 100)}%;background:{bar_color};"></div>'
                f'</div></td>'
                f'<td>{d.get("status", "-")}</td></tr>\n'
            )
        dim_html = ""
        if dimensions:
            dim_html = (
                f'<h3>Coverage by Dimension</h3>\n'
                f'<table><tr><th>Dimension</th><th>Coverage</th><th>Required</th>'
                f'<th>Progress</th><th>Status</th></tr>\n{dim_rows}</table>\n'
            )

        return (
            f'<h2>7. Coverage Dashboard</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">AUM Covered</div>'
            f'<div class="card-value">{_pct(aum_pct)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Emissions Covered</div>'
            f'<div class="card-value">{_pct(emissions_pct)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Classes with Targets</div>'
            f'<div class="card-value">{cov.get("classes_with_targets", 0)}/8</div></div>\n'
            f'  <div class="card"><div class="card-label">FINZ Min Met</div>'
            f'<div class="card-value">{cov.get("finz_min_met", "N/A")}</div></div>\n'
            f'</div>\n'
            f'{dim_html}'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Portfolio targets per SBTi Financial Institutions Net-Zero '
            f'(FINZ) Standard V1.0 and PCAF Global GHG Standard.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
