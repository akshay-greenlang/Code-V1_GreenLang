# -*- coding: utf-8 -*-
"""
TargetProgressReport - Target vs Actual and SBTi Pathway for PACK-045.

Generates a target progress report covering target definitions, actual
vs target comparisons, SBTi pathway tracking, reduction attribution
analysis, and gap-to-target projections.

Sections:
    1. Target Summary
    2. Progress Against Targets
    3. SBTi Pathway Tracking
    4. Reduction Attribution
    5. Gap-to-Target Projections

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _on_track_badge(on_track: bool) -> str:
    """Return on-track status text."""
    return "ON TRACK" if on_track else "OFF TRACK"


def _progress_bar_text(pct: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    clamped = max(0.0, min(100.0, pct))
    filled = int(clamped / 100 * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:.1f}%"


class TargetProgressReport:
    """
    Target progress report template.

    Renders emission reduction targets vs actual performance, SBTi pathway
    alignment analysis, reduction attribution by initiative, and forward
    projections. All outputs include SHA-256 provenance hashing for audit
    trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = TargetProgressReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TargetProgressReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render target progress report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_target_summary(data),
            self._md_progress_table(data),
            self._md_sbti_pathway(data),
            self._md_reduction_attribution(data),
            self._md_projections(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render target progress report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_target_summary(data),
            self._html_progress_table(data),
            self._html_sbti_pathway(data),
            self._html_reduction_attribution(data),
            self._html_projections(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render target progress report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "target_progress_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "target_year": self._get_val(data, "target_year", ""),
            "targets": data.get("targets", []),
            "sbti_pathway": data.get("sbti_pathway", {}),
            "reduction_attribution": data.get("reduction_attribution", []),
            "projections": data.get("projections", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        target_year = self._get_val(data, "target_year", "")
        return (
            f"# Target Progress Report - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Target Year:** {target_year} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_target_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown target summary."""
        targets = data.get("targets", [])
        if not targets:
            return "## 1. Target Summary\n\nNo targets defined."
        lines = [
            "## 1. Target Summary",
            "",
            "| Target | Scope | Type | Reduction | Target Year | Status |",
            "|--------|-------|------|----------|------------|--------|",
        ]
        for t in targets:
            name = t.get("name", "")
            scope = t.get("scope", "")
            ttype = t.get("target_type", "absolute")
            reduction = f"{t.get('reduction_pct', 0):.1f}%"
            year = t.get("target_year", "")
            on_track = _on_track_badge(t.get("on_track", False))
            lines.append(f"| {name} | {scope} | {ttype} | {reduction} | {year} | **{on_track}** |")
        return "\n".join(lines)

    def _md_progress_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown progress against targets."""
        targets = data.get("targets", [])
        if not targets:
            return ""
        lines = [
            "## 2. Progress Against Targets",
            "",
            "| Target | Base (tCO2e) | Current (tCO2e) | Target (tCO2e) | Progress | Gap |",
            "|--------|-------------|----------------|---------------|----------|-----|",
        ]
        for t in targets:
            name = t.get("name", "")
            base = t.get("base_year_tco2e", 0)
            current = t.get("current_tco2e", 0)
            target = t.get("target_tco2e", 0)
            required_reduction = base - target
            actual_reduction = base - current
            pct_done = (actual_reduction / required_reduction * 100) if required_reduction else 0
            gap = current - target
            bar = _progress_bar_text(pct_done, 15)
            lines.append(
                f"| {name} | {base:,.1f} | {current:,.1f} | {target:,.1f} | "
                f"`{bar}` | {gap:+,.1f} |"
            )
        return "\n".join(lines)

    def _md_sbti_pathway(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi pathway tracking."""
        pathway = data.get("sbti_pathway", {})
        if not pathway:
            return ""
        scenario = pathway.get("scenario", "1.5C")
        annual_rate = pathway.get("required_annual_rate_pct", 4.2)
        actual_rate = pathway.get("actual_annual_rate_pct", 0)
        aligned = pathway.get("is_aligned", False)
        milestones = pathway.get("milestones", [])
        lines = [
            "## 3. SBTi Pathway Tracking",
            "",
            f"- **Scenario:** {scenario}",
            f"- **Required Annual Rate:** {annual_rate:.1f}%",
            f"- **Actual Annual Rate:** {actual_rate:.1f}%",
            f"- **Alignment:** {'ALIGNED' if aligned else 'NOT ALIGNED'}",
        ]
        if milestones:
            lines.extend(["", "**Milestones:**", ""])
            lines.append("| Year | Expected (tCO2e) | Actual (tCO2e) | On Track |")
            lines.append("|------|-----------------|---------------|----------|")
            for m in milestones:
                year = m.get("year", "")
                expected = m.get("expected_tco2e", 0)
                actual = m.get("actual_tco2e", 0)
                ot = "Yes" if m.get("on_track") else "No"
                lines.append(f"| {year} | {expected:,.1f} | {actual:,.1f} | **{ot}** |")
        return "\n".join(lines)

    def _md_reduction_attribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown reduction attribution."""
        attrs = data.get("reduction_attribution", [])
        if not attrs:
            return ""
        lines = [
            "## 4. Reduction Attribution",
            "",
            "| Initiative | Category | Reduction (tCO2e) | % of Total Reduction |",
            "|-----------|----------|------------------|---------------------|",
        ]
        total_reduction = sum(a.get("reduction_tco2e", 0) for a in attrs)
        for a in attrs:
            name = a.get("initiative", "")
            cat = a.get("category", "")
            reduction = a.get("reduction_tco2e", 0)
            pct = (reduction / total_reduction * 100) if total_reduction else 0
            lines.append(f"| {name} | {cat} | {reduction:,.1f} | {pct:.1f}% |")
        return "\n".join(lines)

    def _md_projections(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap-to-target projections."""
        projections = data.get("projections", [])
        if not projections:
            return ""
        lines = [
            "## 5. Gap-to-Target Projections",
            "",
            "| Year | Projected (tCO2e) | Target (tCO2e) | Gap (tCO2e) | On Track |",
            "|------|------------------|---------------|------------|----------|",
        ]
        for p in projections:
            year = p.get("year", "")
            projected = p.get("projected_tco2e", 0)
            target = p.get("target_tco2e", 0)
            gap = projected - target
            ot = "Yes" if p.get("on_track") else "No"
            lines.append(f"| {year} | {projected:,.1f} | {target:,.1f} | {gap:+,.1f} | **{ot}** |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Target Progress - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".on-track{color:#2a9d8f;font-weight:700;}\n"
            ".off-track{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".progress-bg{background:#e8e8e8;border-radius:4px;height:20px;width:200px;display:inline-block;}\n"
            ".progress-fill{border-radius:4px;height:20px;display:inline-block;background:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        target_year = self._get_val(data, "target_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Target Progress Report &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Target Year:</strong> {target_year}</p>\n<hr>\n</div>"
        )

    def _html_target_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML target summary table."""
        targets = data.get("targets", [])
        if not targets:
            return ""
        rows = ""
        for t in targets:
            name = t.get("name", "")
            scope = t.get("scope", "")
            reduction = f"{t.get('reduction_pct', 0):.1f}%"
            on_track = t.get("on_track", False)
            css = "on-track" if on_track else "off-track"
            label = "ON TRACK" if on_track else "OFF TRACK"
            rows += (
                f'<tr><td>{name}</td><td>{scope}</td><td>{reduction}</td>'
                f'<td class="{css}">{label}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>1. Target Summary</h2>\n'
            "<table><thead><tr><th>Target</th><th>Scope</th>"
            "<th>Reduction</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_progress_table(self, data: Dict[str, Any]) -> str:
        """Render HTML progress table with bars."""
        targets = data.get("targets", [])
        if not targets:
            return ""
        rows = ""
        for t in targets:
            name = t.get("name", "")
            base = t.get("base_year_tco2e", 0)
            current = t.get("current_tco2e", 0)
            target = t.get("target_tco2e", 0)
            required = base - target
            actual = base - current
            pct = (actual / required * 100) if required else 0
            bar_w = int(max(0, min(200, pct * 2)))
            rows += (
                f'<tr><td>{name}</td><td>{base:,.1f}</td><td>{current:,.1f}</td>'
                f'<td>{target:,.1f}</td><td><div class="progress-bg">'
                f'<div class="progress-fill" style="width:{bar_w}px"></div>'
                f"</div> {pct:.0f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Progress</h2>\n'
            "<table><thead><tr><th>Target</th><th>Base</th><th>Current</th>"
            "<th>Target</th><th>Progress</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sbti_pathway(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi pathway section."""
        pathway = data.get("sbti_pathway", {})
        if not pathway:
            return ""
        scenario = pathway.get("scenario", "1.5C")
        aligned = pathway.get("is_aligned", False)
        css = "on-track" if aligned else "off-track"
        label = "ALIGNED" if aligned else "NOT ALIGNED"
        return (
            '<div class="section">\n<h2>3. SBTi Pathway</h2>\n'
            f"<p><strong>Scenario:</strong> {scenario} | "
            f'<strong>Alignment:</strong> <span class="{css}">{label}</span></p>\n</div>'
        )

    def _html_reduction_attribution(self, data: Dict[str, Any]) -> str:
        """Render HTML reduction attribution table."""
        attrs = data.get("reduction_attribution", [])
        if not attrs:
            return ""
        total_reduction = sum(a.get("reduction_tco2e", 0) for a in attrs)
        rows = ""
        for a in attrs:
            name = a.get("initiative", "")
            cat = a.get("category", "")
            reduction = a.get("reduction_tco2e", 0)
            pct = (reduction / total_reduction * 100) if total_reduction else 0
            rows += f"<tr><td>{name}</td><td>{cat}</td><td>{reduction:,.1f}</td><td>{pct:.1f}%</td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Reduction Attribution</h2>\n'
            "<table><thead><tr><th>Initiative</th><th>Category</th>"
            "<th>tCO2e</th><th>%</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML projections table."""
        projections = data.get("projections", [])
        if not projections:
            return ""
        rows = ""
        for p in projections:
            year = p.get("year", "")
            projected = p.get("projected_tco2e", 0)
            target = p.get("target_tco2e", 0)
            gap = projected - target
            on_track = p.get("on_track", False)
            css = "on-track" if on_track else "off-track"
            rows += (
                f'<tr><td>{year}</td><td>{projected:,.1f}</td><td>{target:,.1f}</td>'
                f'<td>{gap:+,.1f}</td><td class="{css}">{"Yes" if on_track else "No"}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>5. Projections</h2>\n'
            "<table><thead><tr><th>Year</th><th>Projected</th><th>Target</th>"
            "<th>Gap</th><th>On Track</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
