# -*- coding: utf-8 -*-
"""
SectorDisclosureTemplate - Sector-Specific Scope 3 Disclosure for PACK-043.

Generates sector-specific Scope 3 reporting parameterized by sector_focus
configuration. Supports four sector profiles:
    - Finance: PCAF financed emissions disclosure
    - Retail: Packaging and logistics analysis
    - Manufacturing: Circular economy metrics
    - Technology: Cloud carbon report

Each sector profile includes tailored metrics, benchmarks, and disclosure
requirements relevant to the industry.

Sections:
    1. Sector Overview
    2. Sector-Specific Metrics
    3. Benchmark Comparison
    4. Sector Disclosure Requirements
    5. Material Categories Analysis
    6. Sector Action Plan

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained, sector-dependent color theme)
    - JSON (structured with chart-ready data)

Sector Color Themes:
    - Finance: #1A5276
    - Retail: #7D3C98
    - Manufacturing: #1E8449
    - Technology: #2E86C1

Author: GreenLang Team
Version: 43.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"

# ---------------------------------------------------------------------------
# Sector-specific theme configuration
# ---------------------------------------------------------------------------

_SECTOR_THEMES: Dict[str, Dict[str, str]] = {
    "finance": {
        "primary": "#1A5276",
        "primary_light": "#2471A3",
        "accent": "#5DADE2",
        "bg": "#EBF5FB",
        "label": "Financial Services",
    },
    "retail": {
        "primary": "#7D3C98",
        "primary_light": "#A569BD",
        "accent": "#D2B4DE",
        "bg": "#F4ECF7",
        "label": "Retail & Consumer",
    },
    "manufacturing": {
        "primary": "#1E8449",
        "primary_light": "#27AE60",
        "accent": "#82E0AA",
        "bg": "#EAFAF1",
        "label": "Manufacturing",
    },
    "technology": {
        "primary": "#2E86C1",
        "primary_light": "#5DADE2",
        "accent": "#85C1E9",
        "bg": "#EBF5FB",
        "label": "Technology",
    },
}

_DEFAULT_SECTOR = "manufacturing"


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _fmt_currency(value: Optional[float]) -> str:
    """Format currency value."""
    if value is None:
        return "N/A"
    return f"${_fmt_num(value)}"


class SectorDisclosureTemplate:
    """
    Sector-specific Scope 3 disclosure template.

    Renders sector-tailored disclosure reports for finance (PCAF),
    retail (packaging/logistics), manufacturing (circular economy),
    and technology (cloud carbon). The sector is selected via
    sector_focus config parameter. All outputs include SHA-256
    provenance.

    Attributes:
        config: Optional configuration overrides including sector_focus.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = SectorDisclosureTemplate(config={"sector_focus": "finance"})
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SectorDisclosureTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _sector(self, data: Dict[str, Any]) -> str:
        """Resolve sector focus from config or data."""
        sector = self._get_val(data, "sector_focus", _DEFAULT_SECTOR)
        return sector.lower() if sector else _DEFAULT_SECTOR

    def _theme(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get color theme for current sector."""
        sector = self._sector(data)
        return _SECTOR_THEMES.get(sector, _SECTOR_THEMES[_DEFAULT_SECTOR])

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render sector disclosure as Markdown.

        Args:
            data: Validated sector disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sector = self._sector(data)
        sections: List[str] = [
            self._md_header(data),
            self._md_sector_overview(data),
            self._md_sector_metrics(data, sector),
            self._md_benchmark(data),
            self._md_disclosure_requirements(data),
            self._md_material_categories(data),
            self._md_sector_action_plan(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render sector disclosure as HTML.

        Args:
            data: Validated sector disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sector = self._sector(data)
        body_parts: List[str] = [
            self._html_header(data),
            self._html_sector_overview(data),
            self._html_sector_metrics(data, sector),
            self._html_benchmark(data),
            self._html_disclosure_requirements(data),
            self._html_material_categories(data),
            self._html_sector_action_plan(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render sector disclosure as JSON-serializable dict.

        Args:
            data: Validated sector disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        sector = self._sector(data)
        return {
            "template": "sector_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "sector_focus": sector,
            "sector_label": _SECTOR_THEMES.get(sector, {}).get("label", sector),
            "sector_overview": data.get("sector_overview", {}),
            "sector_metrics": data.get("sector_metrics", {}),
            "benchmark_comparison": data.get("benchmark_comparison", []),
            "disclosure_requirements": data.get("disclosure_requirements", []),
            "material_categories": data.get("material_categories", []),
            "sector_action_plan": data.get("sector_action_plan", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        sector = self._sector(data)
        theme = self._theme(data)
        sector_label = theme.get("label", sector)
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# {sector_label} Scope 3 Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Sector:** {sector_label}\n\n"
            "---"
        )

    def _md_sector_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector overview."""
        overview = data.get("sector_overview", {})
        if not overview:
            return "## 1. Sector Overview\n\nNo sector overview available."
        total_scope3 = overview.get("total_scope3_tco2e")
        revenue = overview.get("revenue_mln")
        intensity = overview.get("intensity")
        lines = [
            "## 1. Sector Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if total_scope3 is not None:
            lines.append(f"| Total Scope 3 | {_fmt_tco2e(total_scope3)} |")
        if revenue is not None:
            lines.append(f"| Revenue | ${revenue:.1f}M |")
        if intensity is not None:
            lines.append(f"| Sector Intensity | {intensity:.2f} |")
        description = overview.get("description")
        if description:
            lines.append(f"\n{description}")
        return "\n".join(lines)

    def _md_sector_metrics(self, data: Dict[str, Any], sector: str) -> str:
        """Render Markdown sector-specific metrics."""
        metrics = data.get("sector_metrics", {})
        if not metrics:
            return "## 2. Sector-Specific Metrics\n\nNo metrics available."
        if sector == "finance":
            return self._md_finance_metrics(metrics)
        elif sector == "retail":
            return self._md_retail_metrics(metrics)
        elif sector == "manufacturing":
            return self._md_manufacturing_metrics(metrics)
        elif sector == "technology":
            return self._md_technology_metrics(metrics)
        return self._md_generic_metrics(metrics)

    def _md_finance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render PCAF financed emissions metrics."""
        lines = [
            "## 2. PCAF Financed Emissions",
            "",
            "| Asset Class | AuM ($M) | Financed Emissions (tCO2e) | Data Quality |",
            "|------------|---------|--------------------------|-------------|",
        ]
        asset_classes = metrics.get("asset_classes", [])
        for ac in asset_classes:
            name = ac.get("asset_class", "-")
            aum = ac.get("aum_mln")
            aum_str = f"${aum:.1f}M" if aum is not None else "-"
            emissions = _fmt_tco2e(ac.get("financed_emissions_tco2e"))
            dq = ac.get("data_quality_score")
            dq_str = f"{dq:.1f}" if dq is not None else "-"
            lines.append(f"| {name} | {aum_str} | {emissions} | {dq_str} |")
        total_fe = metrics.get("total_financed_emissions_tco2e")
        if total_fe is not None:
            lines.append(f"\n**Total Financed Emissions:** {_fmt_tco2e(total_fe)}")
        return "\n".join(lines)

    def _md_retail_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render retail packaging/logistics metrics."""
        lines = [
            "## 2. Packaging and Logistics Analysis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        packaging = metrics.get("packaging_emissions_tco2e")
        if packaging is not None:
            lines.append(f"| Packaging Emissions | {_fmt_tco2e(packaging)} |")
        logistics = metrics.get("logistics_emissions_tco2e")
        if logistics is not None:
            lines.append(f"| Logistics Emissions | {_fmt_tco2e(logistics)} |")
        recycled = metrics.get("recycled_packaging_pct")
        if recycled is not None:
            lines.append(f"| Recycled Packaging | {recycled:.1f}% |")
        last_mile = metrics.get("last_mile_emissions_tco2e")
        if last_mile is not None:
            lines.append(f"| Last-Mile Emissions | {_fmt_tco2e(last_mile)} |")
        return "\n".join(lines)

    def _md_manufacturing_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render manufacturing circular economy metrics."""
        lines = [
            "## 2. Circular Economy Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        recycled = metrics.get("recycled_input_pct")
        if recycled is not None:
            lines.append(f"| Recycled Input Rate | {recycled:.1f}% |")
        waste = metrics.get("waste_to_landfill_pct")
        if waste is not None:
            lines.append(f"| Waste to Landfill | {waste:.1f}% |")
        circular = metrics.get("circularity_index")
        if circular is not None:
            lines.append(f"| Circularity Index | {circular:.2f} |")
        supply_chain = metrics.get("supply_chain_emissions_tco2e")
        if supply_chain is not None:
            lines.append(f"| Supply Chain Emissions | {_fmt_tco2e(supply_chain)} |")
        return "\n".join(lines)

    def _md_technology_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render technology cloud carbon metrics."""
        lines = [
            "## 2. Cloud Carbon Report",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        cloud = metrics.get("cloud_emissions_tco2e")
        if cloud is not None:
            lines.append(f"| Cloud Emissions | {_fmt_tco2e(cloud)} |")
        pue = metrics.get("pue")
        if pue is not None:
            lines.append(f"| PUE (Power Usage Effectiveness) | {pue:.2f} |")
        renewable = metrics.get("renewable_energy_pct")
        if renewable is not None:
            lines.append(f"| Renewable Energy | {renewable:.1f}% |")
        devices = metrics.get("device_emissions_tco2e")
        if devices is not None:
            lines.append(f"| Employee Device Emissions | {_fmt_tco2e(devices)} |")
        return "\n".join(lines)

    def _md_generic_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render generic sector metrics."""
        lines = [
            "## 2. Sector-Specific Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"| {key.replace('_', ' ').title()} | {_fmt_num(value)} |")
        return "\n".join(lines)

    def _md_benchmark(self, data: Dict[str, Any]) -> str:
        """Render Markdown benchmark comparison."""
        benchmarks = data.get("benchmark_comparison", [])
        if not benchmarks:
            return "## 3. Benchmark Comparison\n\nNo benchmark data available."
        lines = [
            "## 3. Benchmark Comparison",
            "",
            "| Metric | Company | Sector Avg | Sector Best | Percentile |",
            "|--------|---------|-----------|-------------|-----------|",
        ]
        for bm in benchmarks:
            metric = bm.get("metric_name", "-")
            company_val = bm.get("company_value")
            co_str = f"{company_val}" if company_val is not None else "-"
            sector_avg = bm.get("sector_average")
            avg_str = f"{sector_avg}" if sector_avg is not None else "-"
            sector_best = bm.get("sector_best")
            best_str = f"{sector_best}" if sector_best is not None else "-"
            percentile = bm.get("percentile")
            pct_str = f"{percentile:.0f}th" if percentile is not None else "-"
            lines.append(f"| {metric} | {co_str} | {avg_str} | {best_str} | {pct_str} |")
        return "\n".join(lines)

    def _md_disclosure_requirements(self, data: Dict[str, Any]) -> str:
        """Render Markdown disclosure requirements."""
        reqs = data.get("disclosure_requirements", [])
        if not reqs:
            return "## 4. Disclosure Requirements\n\nNo requirements data available."
        lines = [
            "## 4. Sector Disclosure Requirements",
            "",
            "| Requirement | Framework | Status | Gap |",
            "|------------|----------|--------|-----|",
        ]
        for req in reqs:
            name = req.get("requirement", "-")
            framework = req.get("framework", "-")
            status = "Met" if req.get("met", False) else "Gap"
            gap = req.get("gap_description", "-") if not req.get("met", False) else "-"
            lines.append(f"| {name} | {framework} | {status} | {gap} |")
        return "\n".join(lines)

    def _md_material_categories(self, data: Dict[str, Any]) -> str:
        """Render Markdown material categories analysis."""
        categories = data.get("material_categories", [])
        if not categories:
            return "## 5. Material Categories\n\nNo category data available."
        lines = [
            "## 5. Material Categories Analysis",
            "",
            "| Category | Emissions (tCO2e) | % of Scope 3 | Sector Relevance |",
            "|----------|------------------|-------------|-----------------|",
        ]
        for cat in categories:
            name = f"Cat {cat.get('category_number', '?')} - {cat.get('category_name', 'Unknown')}"
            em = _fmt_tco2e(cat.get("emissions_tco2e"))
            pct = cat.get("pct_of_scope3")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            relevance = cat.get("sector_relevance", "-")
            lines.append(f"| {name} | {em} | {pct_str} | {relevance} |")
        return "\n".join(lines)

    def _md_sector_action_plan(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector action plan."""
        actions = data.get("sector_action_plan", [])
        if not actions:
            return "## 6. Sector Action Plan\n\nNo action plan defined."
        lines = ["## 6. Sector Action Plan", ""]
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            impact = action.get("expected_impact", "-")
            lines.append(f"### Action {i}: {title}")
            lines.append("")
            if desc:
                lines.append(desc)
                lines.append("")
            lines.append(f"- **Expected Impact:** {impact}")
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with sector-specific CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        theme = self._theme(data)
        primary = theme["primary"]
        primary_light = theme["primary_light"]
        accent = theme["accent"]
        bg = theme["bg"]
        label = theme.get("label", "Sector")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{label} Disclosure - {company} ({year})</title>\n"
            "<style>\n"
            f":root{{--primary:{primary};--primary-light:{primary_light};"
            f"--accent:{accent};--bg:{bg};"
            "--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "h3{color:var(--primary-light);}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            f"tr:nth-child(even){{background:{bg};}}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".met{color:var(--success);font-weight:700;}\n"
            ".gap{color:var(--danger);font-weight:700;}\n"
            ".action-card{border-left:4px solid var(--primary);"
            "padding:1rem;margin:0.5rem 0;border-radius:0 4px 4px 0;"
            f"background:{bg};}}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        theme = self._theme(data)
        label = theme.get("label", "Sector")
        return (
            '<div class="section">\n'
            f"<h1>{label} Scope 3 Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Sector:</strong> {label} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_sector_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML sector overview."""
        overview = data.get("sector_overview", {})
        if not overview:
            return ""
        rows = ""
        total_scope3 = overview.get("total_scope3_tco2e")
        if total_scope3 is not None:
            rows += f"<tr><td>Total Scope 3</td><td>{_fmt_tco2e(total_scope3)}</td></tr>\n"
        revenue = overview.get("revenue_mln")
        if revenue is not None:
            rows += f"<tr><td>Revenue</td><td>${revenue:.1f}M</td></tr>\n"
        intensity = overview.get("intensity")
        if intensity is not None:
            rows += f"<tr><td>Sector Intensity</td><td>{intensity:.2f}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>1. Sector Overview</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sector_metrics(self, data: Dict[str, Any], sector: str) -> str:
        """Render HTML sector-specific metrics."""
        metrics = data.get("sector_metrics", {})
        if not metrics:
            return ""
        if sector == "finance":
            return self._html_finance_metrics(metrics)
        elif sector == "retail":
            return self._html_retail_metrics(metrics)
        elif sector == "manufacturing":
            return self._html_manufacturing_metrics(metrics)
        elif sector == "technology":
            return self._html_technology_metrics(metrics)
        return self._html_generic_metrics(metrics)

    def _html_finance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render HTML PCAF financed emissions."""
        asset_classes = metrics.get("asset_classes", [])
        if not asset_classes:
            return ""
        rows = ""
        for ac in asset_classes:
            name = ac.get("asset_class", "-")
            aum = ac.get("aum_mln")
            aum_str = f"${aum:.1f}M" if aum is not None else "-"
            emissions = _fmt_tco2e(ac.get("financed_emissions_tco2e"))
            dq = ac.get("data_quality_score")
            dq_str = f"{dq:.1f}" if dq is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{aum_str}</td>"
                f"<td>{emissions}</td><td>{dq_str}</td></tr>\n"
            )
        total_fe = metrics.get("total_financed_emissions_tco2e")
        total_html = ""
        if total_fe is not None:
            total_html = f"<p><strong>Total Financed Emissions:</strong> {_fmt_tco2e(total_fe)}</p>"
        return (
            '<div class="section">\n'
            "<h2>2. PCAF Financed Emissions</h2>\n"
            "<table><thead><tr><th>Asset Class</th><th>AuM</th>"
            "<th>Financed Emissions</th><th>DQ Score</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{total_html}\n</div>"
        )

    def _html_retail_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render HTML retail metrics."""
        rows = ""
        packaging = metrics.get("packaging_emissions_tco2e")
        if packaging is not None:
            rows += f"<tr><td>Packaging Emissions</td><td>{_fmt_tco2e(packaging)}</td></tr>\n"
        logistics = metrics.get("logistics_emissions_tco2e")
        if logistics is not None:
            rows += f"<tr><td>Logistics Emissions</td><td>{_fmt_tco2e(logistics)}</td></tr>\n"
        recycled = metrics.get("recycled_packaging_pct")
        if recycled is not None:
            rows += f"<tr><td>Recycled Packaging</td><td>{recycled:.1f}%</td></tr>\n"
        last_mile = metrics.get("last_mile_emissions_tco2e")
        if last_mile is not None:
            rows += f"<tr><td>Last-Mile Emissions</td><td>{_fmt_tco2e(last_mile)}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>2. Packaging and Logistics Analysis</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_manufacturing_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render HTML manufacturing metrics."""
        rows = ""
        recycled = metrics.get("recycled_input_pct")
        if recycled is not None:
            rows += f"<tr><td>Recycled Input Rate</td><td>{recycled:.1f}%</td></tr>\n"
        waste = metrics.get("waste_to_landfill_pct")
        if waste is not None:
            rows += f"<tr><td>Waste to Landfill</td><td>{waste:.1f}%</td></tr>\n"
        circular = metrics.get("circularity_index")
        if circular is not None:
            rows += f"<tr><td>Circularity Index</td><td>{circular:.2f}</td></tr>\n"
        supply_chain = metrics.get("supply_chain_emissions_tco2e")
        if supply_chain is not None:
            rows += f"<tr><td>Supply Chain Emissions</td><td>{_fmt_tco2e(supply_chain)}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>2. Circular Economy Metrics</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_technology_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render HTML technology metrics."""
        rows = ""
        cloud = metrics.get("cloud_emissions_tco2e")
        if cloud is not None:
            rows += f"<tr><td>Cloud Emissions</td><td>{_fmt_tco2e(cloud)}</td></tr>\n"
        pue = metrics.get("pue")
        if pue is not None:
            rows += f"<tr><td>PUE</td><td>{pue:.2f}</td></tr>\n"
        renewable = metrics.get("renewable_energy_pct")
        if renewable is not None:
            rows += f"<tr><td>Renewable Energy</td><td>{renewable:.1f}%</td></tr>\n"
        devices = metrics.get("device_emissions_tco2e")
        if devices is not None:
            rows += f"<tr><td>Device Emissions</td><td>{_fmt_tco2e(devices)}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>2. Cloud Carbon Report</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_generic_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render HTML generic metrics."""
        rows = ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                rows += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{_fmt_num(value)}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>2. Sector-Specific Metrics</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark comparison."""
        benchmarks = data.get("benchmark_comparison", [])
        if not benchmarks:
            return ""
        rows = ""
        for bm in benchmarks:
            metric = bm.get("metric_name", "-")
            company_val = bm.get("company_value")
            co_str = f"{company_val}" if company_val is not None else "-"
            sector_avg = bm.get("sector_average")
            avg_str = f"{sector_avg}" if sector_avg is not None else "-"
            sector_best = bm.get("sector_best")
            best_str = f"{sector_best}" if sector_best is not None else "-"
            percentile = bm.get("percentile")
            pct_str = f"{percentile:.0f}th" if percentile is not None else "-"
            rows += (
                f"<tr><td>{metric}</td><td>{co_str}</td><td>{avg_str}</td>"
                f"<td>{best_str}</td><td>{pct_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Benchmark Comparison</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Company</th>"
            "<th>Sector Avg</th><th>Sector Best</th><th>Percentile</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_disclosure_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML disclosure requirements."""
        reqs = data.get("disclosure_requirements", [])
        if not reqs:
            return ""
        rows = ""
        for req in reqs:
            name = req.get("requirement", "-")
            framework = req.get("framework", "-")
            met = req.get("met", False)
            css = "met" if met else "gap"
            status = "Met" if met else "Gap"
            gap_desc = req.get("gap_description", "-") if not met else "-"
            rows += (
                f"<tr><td>{name}</td><td>{framework}</td>"
                f'<td class="{css}">{status}</td><td>{gap_desc}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>4. Sector Disclosure Requirements</h2>\n"
            "<table><thead><tr><th>Requirement</th><th>Framework</th>"
            "<th>Status</th><th>Gap</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_material_categories(self, data: Dict[str, Any]) -> str:
        """Render HTML material categories."""
        categories = data.get("material_categories", [])
        if not categories:
            return ""
        rows = ""
        for cat in categories:
            name = f"Cat {cat.get('category_number', '?')} - {cat.get('category_name', 'Unknown')}"
            em = _fmt_tco2e(cat.get("emissions_tco2e"))
            pct = cat.get("pct_of_scope3")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            relevance = cat.get("sector_relevance", "-")
            rows += (
                f"<tr><td>{name}</td><td>{em}</td>"
                f"<td>{pct_str}</td><td>{relevance}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Material Categories Analysis</h2>\n"
            "<table><thead><tr><th>Category</th><th>Emissions</th>"
            "<th>% of Scope 3</th><th>Sector Relevance</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sector_action_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML sector action plan."""
        actions = data.get("sector_action_plan", [])
        if not actions:
            return ""
        cards = ""
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            impact = action.get("expected_impact", "-")
            cards += (
                f'<div class="action-card">\n'
                f"<h3>Action {i}: {title}</h3>\n"
                f"<p>{desc}</p>\n"
                f"<p><strong>Expected Impact:</strong> {impact}</p>\n</div>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Sector Action Plan</h2>\n"
            f"{cards}</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
